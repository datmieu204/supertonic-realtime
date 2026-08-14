"""Tests for :mod:`supertonic.streaming`.

The real ``TTS`` is replaced with a stand-in so these tests never touch ONNX
Runtime or download a model. Audio length is derived from the text length so
assertions about chunking stay meaningful.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from supertonic.streaming import (
    AudioChunk,
    ClauseBuffer,
    synthesize_clause,
    synthesize_stream,
)

SAMPLE_RATE = 44100


class FakeTTS:
    """Minimal stand-in for :class:`supertonic.TTS` used in streaming tests."""

    def __init__(self, on_synthesize=None) -> None:
        self.sample_rate = SAMPLE_RATE
        self.calls: list[dict] = []
        self._on_synthesize = on_synthesize

    def synthesize(
        self,
        text: str,
        voice_style,
        total_steps: int = 8,
        speed: float = 1.05,
        max_chunk_length=None,
        silence_duration: float = 0.3,
        lang=None,
        verbose: bool = False,
    ):
        self.calls.append(
            {
                "text": text,
                "total_steps": total_steps,
                "speed": speed,
                "max_chunk_length": max_chunk_length,
                "silence_duration": silence_duration,
                "lang": lang,
            }
        )
        if self._on_synthesize is not None:
            self._on_synthesize(text)
        # 10 samples per character, so chunk sizes are visible in the output.
        n = max(1, len(text) * 10)
        return np.zeros((1, n), dtype=np.float32), np.array([n / SAMPLE_RATE])


STYLE = object()


# --- ClauseBuffer: boundary detection -----------------------------------


def test_releases_first_sentence_immediately():
    buf = ClauseBuffer()
    # The trailing space confirms the period ends a sentence.
    assert buf.push("Hello there. ") == ["Hello there."]
    assert buf.pending == ""


def test_waits_for_the_character_after_a_period():
    buf = ClauseBuffer()
    # Without the following character we cannot tell a sentence end from "3.14".
    assert buf.push("Hello there.") == []
    assert buf.push(" And more") == ["Hello there."]


def test_decimal_number_is_not_a_boundary():
    buf = ClauseBuffer()
    assert buf.push("Pi is 3.14 and that is that. ") == ["Pi is 3.14 and that is that."]


def test_abbreviations_do_not_split():
    buf = ClauseBuffer()
    assert buf.push("Dr. Smith arrived. ") == ["Dr. Smith arrived."]


def test_single_letter_initials_do_not_split():
    buf = ClauseBuffer()
    assert buf.push("J. R. R. Tolkien wrote it. ") == ["J. R. R. Tolkien wrote it."]


def test_closing_quote_is_kept_with_the_sentence():
    buf = ClauseBuffer()
    assert buf.push('He said "go!" then left. ') == ['He said "go!"']
    assert buf.flush() == ["then left."]


def test_newline_is_a_boundary():
    buf = ClauseBuffer()
    assert buf.push("A short line\nnext") == ["A short line"]


def test_min_chunk_chars_applies_after_the_first_clause():
    buf = ClauseBuffer(min_chunk_chars=30)
    # First clause is released regardless of length — latency wins.
    assert buf.push("Hi. ") == ["Hi."]
    # The next short sentence is held back and merged with what follows.
    assert buf.push("Yes. ") == []
    out = buf.push("That is a considerably longer sentence to say. ")
    assert out == ["Yes. That is a considerably longer sentence to say."]


def test_falls_back_to_phrase_boundary_when_target_reached():
    buf = ClauseBuffer(first_chunk_chars=40)
    text = "this clause has no sentence end, but it does have a comma right there and more"
    (clause,) = buf.push(text)
    assert clause.endswith(",")
    assert len(clause) <= 40


def test_falls_back_to_whitespace_when_no_punctuation():
    buf = ClauseBuffer(first_chunk_chars=30)
    (clause,) = buf.push("word " * 20)
    assert len(clause) <= 30
    assert clause.endswith("word")


def test_hard_cut_when_a_single_token_exceeds_the_target():
    buf = ClauseBuffer(first_chunk_chars=10)
    (clause,) = buf.push("x" * 25)
    assert clause == "x" * 10


def test_later_clauses_use_the_larger_target():
    buf = ClauseBuffer(first_chunk_chars=20, max_chunk_chars=120, min_chunk_chars=0)
    sentence = "This sentence is about sixty characters long in total, yes. "
    first = buf.push(sentence * 3)
    # First clause capped near 20 chars, later ones allowed to grow past it.
    assert len(first[0]) <= 20
    assert any(len(c) > 20 for c in first[1:])


# --- ClauseBuffer: flush and reset ---------------------------------------


def test_flush_drains_the_tail():
    buf = ClauseBuffer()
    buf.push("Hello there. Unfinished tail")
    assert buf.flush() == ["Unfinished tail"]
    assert buf.pending == ""


def test_every_clause_respects_the_target_and_nothing_is_lost():
    buf = ClauseBuffer(first_chunk_chars=30, max_chunk_chars=30, min_chunk_chars=0)
    text = "word " * 40
    out = buf.push(text) + buf.flush()
    assert len(out) > 1
    assert all(len(c) <= 30 for c in out)
    assert " ".join(out) == text.strip()


def test_flush_of_empty_buffer_is_empty():
    assert ClauseBuffer().flush() == []
    assert ClauseBuffer().push("   ") == []


def test_reset_restores_first_chunk_sizing():
    buf = ClauseBuffer(first_chunk_chars=20, max_chunk_chars=200, min_chunk_chars=0)
    buf.push("A first clause here. ")
    assert buf.emitted == 1
    buf.reset()
    assert buf.emitted == 0
    # Sizing is back to the first-chunk target.
    (clause,) = buf.push("word " * 10)
    assert len(clause) <= 20


def test_rejects_tiny_targets():
    with pytest.raises(ValueError, match="first_chunk_chars"):
        ClauseBuffer(first_chunk_chars=3)
    with pytest.raises(ValueError, match="max_chunk_chars"):
        ClauseBuffer(max_chunk_chars=1)


# --- synthesize_clause ----------------------------------------------------


def test_synthesize_clause_returns_mono_audio():
    tts = FakeTTS()
    chunk = synthesize_clause(tts, "Hello there.", STYLE, index=0, lang="en")
    assert isinstance(chunk, AudioChunk)
    assert chunk.wav.ndim == 1
    assert chunk.wav.dtype == np.float32
    assert chunk.sample_rate == SAMPLE_RATE
    assert chunk.duration_s == pytest.approx(len(chunk.wav) / SAMPLE_RATE)


def test_synthesize_clause_disables_resplitting_and_silence():
    tts = FakeTTS()
    text = "A clause that should reach the pipeline in one piece."
    synthesize_clause(tts, text, STYLE)
    call = tts.calls[0]
    assert call["silence_duration"] == 0.0
    assert call["max_chunk_length"] >= len(text)


def test_first_chunk_uses_fewer_steps():
    tts = FakeTTS()
    synthesize_clause(tts, "First.", STYLE, index=0, total_steps=8, first_chunk_steps=3)
    synthesize_clause(tts, "Second.", STYLE, index=1, total_steps=8, first_chunk_steps=3)
    assert [c["total_steps"] for c in tts.calls] == [3, 8]


def test_first_chunk_steps_none_keeps_total_steps():
    tts = FakeTTS()
    synthesize_clause(tts, "First.", STYLE, index=0, total_steps=8, first_chunk_steps=None)
    assert tts.calls[0]["total_steps"] == 8


# --- synthesize_stream ----------------------------------------------------


def test_stream_from_a_plain_string():
    tts = FakeTTS()
    chunks = list(synthesize_stream(tts, "Hello there. How are you today?", STYLE, lang="en"))
    assert [c.text for c in chunks] == ["Hello there.", "How are you today?"]
    assert [c.index for c in chunks] == [0, 1]
    assert {c["lang"] for c in tts.calls} == {"en"}


def test_stream_from_deltas_ignores_fragment_boundaries():
    tts = FakeTTS()
    deltas = ["Hel", "lo the", "re. How ", "are you?", " Fine."]
    chunks = list(synthesize_stream(tts, iter(deltas), STYLE, min_chunk_chars=0))
    assert [c.text for c in chunks] == ["Hello there.", "How are you?", "Fine."]


def test_stream_is_lazy():
    """Nothing is synthesized until the consumer asks for a chunk."""
    tts = FakeTTS()
    stream = synthesize_stream(tts, "Hello there. How are you today?", STYLE)
    assert tts.calls == []
    next(stream)
    assert len(tts.calls) == 1


def test_stream_emits_before_the_text_ends():
    """The first chunk is available while later deltas are still arriving."""
    tts = FakeTTS()
    produced: list[str] = []

    def deltas():
        for part in ["Hello there. ", "Second sentence here. ", "Third one."]:
            produced.append(part)
            yield part

    stream = synthesize_stream(tts, deltas(), STYLE)
    first = next(stream)
    assert first.text == "Hello there."
    # Only the first delta has been pulled from the generator so far.
    assert produced == ["Hello there. "]


def test_cancel_stops_the_stream():
    stop = threading.Event()
    tts = FakeTTS(on_synthesize=lambda text: stop.set())
    chunks = list(synthesize_stream(tts, "One here. Two here. Three here.", STYLE, cancel=stop))
    # The in-flight chunk is dropped once the token is set, so nothing is
    # yielded after the cancel and no further synthesis happens.
    assert chunks == []
    assert len(tts.calls) == 1


def test_cancel_before_start_synthesizes_nothing():
    stop = threading.Event()
    stop.set()
    tts = FakeTTS()
    assert list(synthesize_stream(tts, "Hello there. ", STYLE, cancel=stop)) == []
    assert tts.calls == []


def test_stream_forwards_speed_and_steps():
    tts = FakeTTS()
    list(
        synthesize_stream(
            tts,
            "One here. Two here.",
            STYLE,
            speed=1.5,
            total_steps=6,
            first_chunk_steps=2,
        )
    )
    assert [c["speed"] for c in tts.calls] == [1.5, 1.5]
    assert [c["total_steps"] for c in tts.calls] == [2, 6]
