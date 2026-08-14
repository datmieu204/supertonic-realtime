"""Incremental synthesis for realtime voice agents.

:meth:`supertonic.TTS.synthesize` is a batch API: it chunks the text,
synthesizes every chunk, concatenates, and returns one array. For a voice
agent that is the wrong shape — the listener waits for the *whole* answer
before hearing the first word.

This module streams instead. The unit of streaming is one **clause**, not one
audio frame: :meth:`supertonic.core.Supertonic.__call__` runs the vocoder over
a complete latent, so nothing can be emitted mid-chunk. What we can do is make
the first clause short and cheap, then keep the speaker fed while later clauses
are still being generated:

* :class:`ClauseBuffer` accepts text incrementally (e.g. token deltas from an
  LLM) and releases clauses at sentence/phrase boundaries as soon as one is
  complete — no need to wait for the full text.
* The first clause of a response uses a smaller character target and fewer
  diffusion steps (``first_chunk_steps``), because it is the only chunk whose
  synthesis the listener actually waits on.
* :func:`synthesize_stream` yields :class:`AudioChunk` objects as they are
  produced, and honours a ``cancel`` token between chunks so a voice agent can
  barge-in.

Example:
    ```python
    from supertonic import TTS
    from supertonic.streaming import synthesize_stream

    tts = TTS()
    style = tts.get_voice_style("M1")

    def llm_deltas():          # anything iterable works
        yield "Hello there. "
        yield "This is streamed "
        yield "one clause at a time."

    for chunk in synthesize_stream(tts, llm_deltas(), style, lang="en"):
        speaker.write(chunk.wav)      # chunk.wav is float32, mono, 1-D
    ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, TYPE_CHECKING, Union

import numpy as np

from .config import (
    DEFAULT_FIRST_CHUNK_CHARS,
    DEFAULT_FIRST_CHUNK_STEPS,
    DEFAULT_MIN_CHUNK_CHARS,
    DEFAULT_SPEED,
    DEFAULT_STREAM_CHUNK_CHARS,
    DEFAULT_TOTAL_STEPS,
    MIN_STREAM_CHUNK_CHARS,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core import Style
    from .pipeline import TTS

logger = logging.getLogger(__name__)


# Characters that end a sentence. A clause may be released here even when it is
# shorter than the character target — a natural boundary beats a full buffer.
_STRONG_BOUNDARIES = ".!?…。！？"

# Characters that end a phrase. Used only once the buffer has reached its
# target length, as a nicer split point than plain whitespace.
_WEAK_BOUNDARIES = ",;:—–、，；"

# Quotes/brackets that may trail a sentence-ending mark: `He said "go!"` ends
# after the closing quote, not at the exclamation mark.
_CLOSERS = "\"'’”)]}»›」』"

# Trailing forms that make a period *not* a sentence end. Mirrors the
# abbreviation list in :mod:`supertonic.utils`, lowercased for comparison.
_ABBREVIATIONS = (
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "ph.d.",
    "etc.",
    "e.g.",
    "i.e.",
    "vs.",
    "inc.",
    "ltd.",
    "co.",
    "corp.",
    "st.",
    "ave.",
    "blvd.",
)

# Longest abbreviation above, used to bound the backwards look.
_MAX_ABBREVIATION_LEN = max(len(a) for a in _ABBREVIATIONS)


@dataclass
class AudioChunk:
    """One synthesized clause.

    Attributes:
        index: Position within the current response, starting at 0.
        text: The clause that was synthesized.
        wav: Mono float32 waveform, shape ``(num_samples,)``.
        sample_rate: Sample rate of ``wav`` in Hz.
        duration_s: Duration of ``wav`` in seconds.
        total_steps: Diffusion steps actually used for this chunk (the first
            chunk of a response may use fewer).
    """

    index: int
    text: str
    wav: np.ndarray
    sample_rate: int
    duration_s: float
    total_steps: int


def _is_abbreviation(buf: str, i: int) -> bool:
    """True when the period at ``buf[i]`` belongs to an abbreviation.

    Covers both the known-abbreviation list (``Dr.``) and single-letter
    initials (``J. R. R.``), which would otherwise be read as sentence ends.
    """
    tail = buf[max(0, i - _MAX_ABBREVIATION_LEN + 1) : i + 1].lower()
    if tail.endswith(_ABBREVIATIONS):
        return True
    # Single capital letter initial: "A." but not "a." (end of a word).
    return i >= 1 and buf[i - 1].isupper() and (i < 2 or not buf[i - 2].isalpha())


def _sentence_end(buf: str, i: int) -> int:
    """Exclusive end index of the sentence closing at ``buf[i]``, or -1.

    A boundary is only recognized when the character *after* it (past any
    closing quotes) has already arrived and is whitespace. Mid-stream we
    cannot otherwise tell the ``.`` of ``3.14`` from the end of a sentence,
    and a premature split would put a pause in the middle of a number.
    """
    ch = buf[i]
    if ch == "\n":
        return i + 1
    if ch not in _STRONG_BOUNDARIES:
        return -1
    if ch == "." and _is_abbreviation(buf, i):
        return -1
    j = i + 1
    while j < len(buf) and buf[j] in _CLOSERS:
        j += 1
    if j >= len(buf):
        return -1  # need one more character to confirm
    if not buf[j].isspace():
        return -1
    return j


class ClauseBuffer:
    """Accumulates streamed text and releases it clause by clause.

    Feed it whatever arrives (single tokens, sentences, whole paragraphs) via
    :meth:`push`; it returns the clauses that are ready to synthesize and keeps
    the rest. Call :meth:`flush` when the text is finished to drain the tail.

    Args:
        first_chunk_chars: Character target for the first clause of a
            response. Smaller means lower time-to-first-audio.
        max_chunk_chars: Character target for every later clause. Larger means
            better prosody and less per-chunk overhead.
        min_chunk_chars: Never release a clause shorter than this at a sentence
            boundary; avoids one-word chunks like ``"Yes."`` being split off
            from the sentence that follows. Does **not** apply to the first
            clause of a response, where starting to speak sooner wins.

    Example:
        ```python
        buf = ClauseBuffer()
        buf.push("Hello there. This is ")   # -> ["Hello there."]
        buf.flush()                         # -> ["This is"]
        ```
    """

    def __init__(
        self,
        first_chunk_chars: int = DEFAULT_FIRST_CHUNK_CHARS,
        max_chunk_chars: int = DEFAULT_STREAM_CHUNK_CHARS,
        min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    ) -> None:
        for name, value in (
            ("first_chunk_chars", first_chunk_chars),
            ("max_chunk_chars", max_chunk_chars),
        ):
            if value < MIN_STREAM_CHUNK_CHARS:
                raise ValueError(
                    f"{name} must be at least {MIN_STREAM_CHUNK_CHARS}, got {value}. "
                    f"Very small chunks produce choppy speech."
                )
        if min_chunk_chars < 0:
            raise ValueError(f"min_chunk_chars must be non-negative, got {min_chunk_chars}")

        self.first_chunk_chars = first_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self._buf = ""
        self._emitted = 0

    @property
    def pending(self) -> str:
        """Text held back because no clause boundary has been reached yet."""
        return self._buf

    @property
    def emitted(self) -> int:
        """Number of clauses released since the last :meth:`reset`."""
        return self._emitted

    def reset(self) -> None:
        """Drop buffered text and restart first-chunk sizing.

        Called between responses, and on barge-in so a cancelled answer's tail
        never leaks into the next one.
        """
        self._buf = ""
        self._emitted = 0

    def push(self, text: str) -> List[str]:
        """Append ``text`` and return every clause that is now complete."""
        if not text:
            return []
        self._buf += text
        out: List[str] = []
        while True:
            end = self._next_clause_end()
            if end < 0:
                break
            clause = self._buf[:end].strip()
            self._buf = self._buf[end:].lstrip()
            if clause:
                out.append(clause)
                self._emitted += 1
        return out

    def flush(self) -> List[str]:
        """Release everything still buffered, splitting it if it is long."""
        out: List[str] = []
        while self._buf.strip():
            end = self._next_clause_end()
            if end < 0:
                # No boundary left: the tail is short enough to send as-is.
                clause = self._buf.strip()
                self._buf = ""
                out.append(clause)
                self._emitted += 1
                break
            clause = self._buf[:end].strip()
            self._buf = self._buf[end:].lstrip()
            if clause:
                out.append(clause)
                self._emitted += 1
        self._buf = ""
        return out

    def _target(self) -> int:
        return self.first_chunk_chars if self._emitted == 0 else self.max_chunk_chars

    def _next_clause_end(self) -> int:
        """Exclusive index where the next clause ends, or -1 if not ready.

        Preference order: sentence boundary, phrase boundary, whitespace, hard
        cut. Everything but the sentence boundary requires the buffer to have
        reached the character target first, so we never split a short sentence
        that is still growing.
        """
        buf = self._buf
        target = self._target()
        # `min_chunk_chars` deliberately does not apply to the first clause of
        # a response: "Hi." is a fine thing to start speaking immediately, and
        # the rest of the answer is synthesized while it plays. Later clauses
        # clamp the floor to the target so an over-large minimum only weakens
        # the preference for boundaries instead of disabling it.
        floor = 0 if self._emitted == 0 else min(max(0, self.min_chunk_chars - 1), target - 1)

        # 1. Sentence boundary at/after min_chunk_chars, within the target.
        for i in range(floor, min(len(buf), target)):
            end = _sentence_end(buf, i)
            if end > 0:
                return end

        if len(buf) < target:
            return -1

        # 2. Latest phrase boundary inside the target.
        for i in range(min(len(buf), target) - 1, floor - 1, -1):
            if buf[i] in _WEAK_BOUNDARIES and i + 1 < len(buf) and buf[i + 1].isspace():
                return i + 1

        # 3. Latest whitespace inside the target.
        for i in range(min(len(buf), target) - 1, floor - 1, -1):
            if buf[i].isspace():
                return i

        # 4. No boundary at all (e.g. one very long token) — cut at the target.
        return target


def _cancelled(cancel: Optional[object]) -> bool:
    """True when a cancel token (anything with ``is_set()``) has been set."""
    return cancel is not None and bool(cancel.is_set())  # type: ignore[attr-defined]


def synthesize_clause(
    tts: "TTS",
    clause: str,
    voice_style: "Style",
    *,
    index: int = 0,
    lang: Optional[str] = None,
    speed: float = DEFAULT_SPEED,
    total_steps: int = DEFAULT_TOTAL_STEPS,
    first_chunk_steps: Optional[int] = DEFAULT_FIRST_CHUNK_STEPS,
) -> AudioChunk:
    """Synthesize a single clause into an :class:`AudioChunk`.

    This is the blocking unit of work behind :func:`synthesize_stream`; the
    server calls it directly because it buffers clauses on the connection's
    event loop and runs only the synthesis in a worker thread.

    Args:
        tts: Loaded :class:`supertonic.TTS`.
        clause: Text to synthesize. Should already be clause-sized —
            ``max_chunk_length`` is set so the pipeline will not re-split it.
        voice_style: Voice style to speak with.
        index: Position in the current response; ``0`` selects
            ``first_chunk_steps``.
        lang: Language code, or ``None`` to let the pipeline decide.
        speed: Speech speed multiplier.
        total_steps: Diffusion steps for chunks after the first.
        first_chunk_steps: Diffusion steps for the first chunk. ``None`` uses
            ``total_steps`` for every chunk.

    Returns:
        The synthesized :class:`AudioChunk`.
    """
    steps = first_chunk_steps if (index == 0 and first_chunk_steps is not None) else total_steps
    wav, _ = tts.synthesize(
        text=clause,
        voice_style=voice_style,
        total_steps=steps,
        speed=speed,
        # The clause is already sized; passing its own length as the limit
        # keeps `chunk_text` from splitting it again and inserting silence.
        max_chunk_length=max(len(clause), 10),
        silence_duration=0.0,
        lang=lang,
    )
    mono = np.asarray(wav, dtype=np.float32).reshape(-1)
    return AudioChunk(
        index=index,
        text=clause,
        wav=mono,
        sample_rate=tts.sample_rate,
        duration_s=len(mono) / float(tts.sample_rate),
        total_steps=steps,
    )


def synthesize_stream(
    tts: "TTS",
    text: Union[str, Iterable[str]],
    voice_style: "Style",
    *,
    lang: Optional[str] = None,
    speed: float = DEFAULT_SPEED,
    total_steps: int = DEFAULT_TOTAL_STEPS,
    first_chunk_steps: Optional[int] = DEFAULT_FIRST_CHUNK_STEPS,
    first_chunk_chars: int = DEFAULT_FIRST_CHUNK_CHARS,
    max_chunk_chars: int = DEFAULT_STREAM_CHUNK_CHARS,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    cancel: Optional[object] = None,
) -> Iterator[AudioChunk]:
    """Synthesize text incrementally, yielding audio clause by clause.

    Args:
        tts: Loaded :class:`supertonic.TTS`.
        text: A complete string, or any iterable of text fragments (an LLM's
            streamed deltas, a file read line by line, ...). Fragments do not
            need to align with word or sentence boundaries.
        voice_style: Voice style to speak with.
        lang: Language code, or ``None`` to let the pipeline decide.
        speed: Speech speed multiplier.
        total_steps: Diffusion steps for chunks after the first.
        first_chunk_steps: Diffusion steps for the first chunk — lower trades
            a little quality for a faster first word. ``None`` disables the
            special case.
        first_chunk_chars: Character target for the first clause.
        max_chunk_chars: Character target for later clauses.
        min_chunk_chars: Shortest clause that may be split at a sentence end.
        cancel: Optional token with an ``is_set()`` method (e.g.
            :class:`threading.Event`). Checked before each chunk is
            synthesized and again before it is yielded, so a barge-in stops
            the stream within one chunk. Audio already yielded is unaffected.

    Yields:
        :class:`AudioChunk` in order, one per clause.

    Example:
        ```python
        stop = threading.Event()
        for chunk in synthesize_stream(tts, deltas, style, cancel=stop):
            play(chunk.wav)
        ```
    """
    buffer = ClauseBuffer(
        first_chunk_chars=first_chunk_chars,
        max_chunk_chars=max_chunk_chars,
        min_chunk_chars=min_chunk_chars,
    )
    deltas: Iterable[str] = [text] if isinstance(text, str) else text
    index = 0

    def _emit(clauses: List[str]) -> Iterator[AudioChunk]:
        nonlocal index
        for clause in clauses:
            if _cancelled(cancel):
                logger.debug("streaming cancelled before chunk %d", index)
                return
            chunk = synthesize_clause(
                tts,
                clause,
                voice_style,
                index=index,
                lang=lang,
                speed=speed,
                total_steps=total_steps,
                first_chunk_steps=first_chunk_steps,
            )
            if _cancelled(cancel):
                logger.debug("streaming cancelled after chunk %d", index)
                return
            index += 1
            yield chunk

    for delta in deltas:
        if _cancelled(cancel):
            return
        for chunk in _emit(buffer.push(delta)):
            yield chunk
    if _cancelled(cancel):
        return
    for chunk in _emit(buffer.flush()):
        yield chunk
