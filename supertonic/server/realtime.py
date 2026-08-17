"""Realtime streaming synthesis over WebSocket — ``WS /v1/realtime``.

The HTTP routes in :mod:`supertonic.server.routes` are request/response: the
caller sends the whole text and gets one encoded file back. A voice agent
cannot use that shape — it has an LLM producing tokens and a speaker that
should start playing before the sentence is finished.

This endpoint inverts it. The client pushes text as it becomes available; the
server releases audio clause by clause (see :mod:`supertonic.streaming` for why
a clause, and not an audio frame, is the smallest unit the vocoder can emit).

Protocol
--------

One WebSocket, JSON text frames in both directions, plus **binary frames from
the server carrying raw PCM**. Connection query parameters accept the same
fields as ``session.update`` (e.g. ``/v1/realtime?voice=F1&lang=en``).

Client → server::

    {"type": "session.update", "voice": "F1", "speed": 1.1}   # any subset
    {"type": "input.text.delta", "text": "Hello th"}          # append text
    {"type": "input.text.done"}                               # flush the tail
    {"type": "speak", "text": "One shot."}                    # delta + done
    {"type": "cancel"}                                        # barge-in
    {"type": "ping"}

Server → client::

    {"type": "session.created", "sample_rate": 44100, "format": "pcm_s16le", ...}
    {"type": "session.updated", ...}
    {"type": "audio.chunk", "index": 0, "text": "Hello there.", "bytes": 66150, ...}
    <binary frame: `bytes` bytes of PCM>
    {"type": "response.done", "chunks": 3, "duration_s": 4.12}
    {"type": "cancelled"}
    {"type": "pong"}
    {"type": "error", "error": {"message": "...", "type": "...", "code": "..."}}

Every ``audio.chunk`` is immediately followed by exactly one binary frame, so a
client can either read the metadata or ignore it and just play the bytes.
Audio is headerless PCM at the model's native sample rate (44.1 kHz) —
``session.created`` reports the rate and format; clients targeting WebRTC or
telephony must resample.

Barge-in
--------

``cancel`` bumps a response counter: queued clauses are dropped and the chunk
currently in the ONNX session is discarded when it finishes. ONNX Runtime
inference cannot be interrupted, so cancellation is effective within one chunk
— which is why a short first chunk matters for more than just latency.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import anyio
from anyio import to_thread
from fastapi import APIRouter, FastAPI, WebSocket
from pydantic import ValidationError

from .. import __version__
from ..config import (
    AVAILABLE_LANGUAGES,
    DEFAULT_FIRST_CHUNK_CHARS,
    DEFAULT_FIRST_CHUNK_STEPS,
    DEFAULT_MIN_CHUNK_CHARS,
    DEFAULT_SPEED,
    DEFAULT_STREAM_CHUNK_CHARS,
    DEFAULT_TOTAL_STEPS,
    MAX_TEXT_LENGTH,
)
from ..streaming import ClauseBuffer, synthesize_clause
from .audio import (
    DEFAULT_PCM_FORMAT,
    PCM_FORMATS,
    UnsupportedAudioFormat,
    coerce_pcm_format,
    encode_pcm,
)
from .routes import UnknownVoice, resolve_voice
from .schemas import RealtimeSessionUpdate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .app import ServerState

logger = logging.getLogger(__name__)


# Largest single ``input.text.delta``. Generous for an LLM token stream while
# bounding what one frame can allocate.
MAX_DELTA_CHARS = 10_000

# Largest amount of text that may sit unsynthesized in the clause buffer. A
# client that never sends a boundary character would otherwise grow it without
# limit. Matches the pipeline's own per-call ceiling.
MAX_PENDING_CHARS = MAX_TEXT_LENGTH

# WebSocket close codes.
_CLOSE_TRY_AGAIN_LATER = 1013

# Queue entry kinds.
_KIND_CHUNK = "chunk"
_KIND_DONE = "done"


@dataclass(frozen=True)
class _SessionConfig:
    """Per-connection synthesis settings, replaced wholesale on update."""

    voice: str = "M1"
    lang: Optional[str] = None
    speed: float = DEFAULT_SPEED
    steps: int = DEFAULT_TOTAL_STEPS
    first_chunk_steps: int = DEFAULT_FIRST_CHUNK_STEPS
    fmt: str = DEFAULT_PCM_FORMAT
    first_chunk_chars: int = DEFAULT_FIRST_CHUNK_CHARS
    max_chunk_chars: int = DEFAULT_STREAM_CHUNK_CHARS
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS


def _apply_update(cfg: _SessionConfig, upd: RealtimeSessionUpdate) -> _SessionConfig:
    """Return ``cfg`` with every non-``None`` field of ``upd`` applied."""
    changes: Dict[str, Any] = {}
    for field, attr in (
        ("voice", "voice"),
        ("lang", "lang"),
        ("speed", "speed"),
        ("steps", "steps"),
        ("first_chunk_steps", "first_chunk_steps"),
        ("format", "fmt"),
        ("first_chunk_chars", "first_chunk_chars"),
        ("max_chunk_chars", "max_chunk_chars"),
        ("min_chunk_chars", "min_chunk_chars"),
    ):
        value = getattr(upd, field)
        if value is not None:
            changes[attr] = value
    return replace(cfg, **changes)


class _RealtimeSession:
    """Drives one WebSocket connection.

    Two concurrent tasks share the connection:

    * :meth:`_receive_loop` owns the client's messages and the clause buffer.
      It only does string work, so it stays responsive to ``cancel`` even
      while a synthesis is in flight.
    * :meth:`_sender_loop` pulls clauses off the queue, runs the (blocking)
      ONNX synthesis in a worker thread, and writes audio back.

    They communicate through an in-memory stream of
    ``(kind, clause, generation)`` entries. ``generation`` is what makes
    barge-in work: :meth:`_cancel` increments it, and the sender drops any
    entry — or any finished audio — whose generation is stale.
    """

    def __init__(self, ws: WebSocket, state: "ServerState", cfg: _SessionConfig, style: Any):
        self._ws = ws
        self._state = state
        self._cfg = cfg
        self._style = style
        self._buffer = ClauseBuffer(
            first_chunk_chars=cfg.first_chunk_chars,
            max_chunk_chars=cfg.max_chunk_chars,
            min_chunk_chars=cfg.min_chunk_chars,
        )
        self._send, self._recv = anyio.create_memory_object_stream(math.inf)
        self._send_lock = anyio.Lock()
        # Current response id. Bumped by `cancel`; entries and finished audio
        # tagged with an older id are discarded.
        self._gen = 0
        # Sender-side accounting for the response being emitted.
        self._sender_gen = 0
        self._chunks = 0
        self._duration = 0.0

    # --- outbound ---------------------------------------------------------

    async def _send_json(self, payload: Dict[str, Any]) -> None:
        # Both tasks can write (errors from the receive loop, audio from the
        # sender); serialize so frames never interleave.
        async with self._send_lock:
            await self._ws.send_json(payload)

    async def _send_pcm(self, data: bytes) -> None:
        async with self._send_lock:
            await self._ws.send_bytes(data)

    async def _error(self, message: str, code: str, type_: str = "invalid_request_error") -> None:
        await self._send_json(
            {"type": "error", "error": {"message": message, "type": type_, "code": code}}
        )

    def config_payload(self) -> Dict[str, Any]:
        """Public view of the current session settings."""
        cfg = self._cfg
        return {
            "model": self._state.model,
            "sample_rate": self._state.tts.sample_rate if self._state.tts else None,
            "format": cfg.fmt,
            "voice": cfg.voice,
            "lang": cfg.lang,
            "speed": cfg.speed,
            "steps": cfg.steps,
            "first_chunk_steps": cfg.first_chunk_steps,
            "first_chunk_chars": cfg.first_chunk_chars,
            "max_chunk_chars": cfg.max_chunk_chars,
            "min_chunk_chars": cfg.min_chunk_chars,
            "version": __version__,
        }

    # --- lifecycle --------------------------------------------------------

    async def run(self) -> None:
        """Serve the connection until the client disconnects."""
        await self._send_json(dict(type="session.created", **self.config_payload()))
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._sender_loop)
            try:
                await self._receive_loop()
            finally:
                # Stop the sender even if it is parked on `receive` or waiting
                # for a worker thread.
                tg.cancel_scope.cancel()

    # --- inbound ----------------------------------------------------------

    async def _receive_loop(self) -> None:
        while True:
            message = await self._ws.receive()
            if message["type"] == "websocket.disconnect":
                return
            raw = message.get("text")
            if raw is None:
                await self._error("expected a text frame containing a JSON event", "invalid_frame")
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as e:
                await self._error(f"invalid JSON: {e}", "invalid_json")
                continue
            if not isinstance(event, dict):
                await self._error("event must be a JSON object", "invalid_event")
                continue
            await self._handle(event)

    async def _handle(self, event: Dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "input.text.delta":
            await self._on_delta(event.get("text"))
        elif etype == "input.text.done":
            self._commit()
        elif etype == "speak":
            if await self._on_delta(event.get("text")):
                self._commit()
        elif etype == "cancel":
            await self._cancel()
        elif etype == "session.update":
            await self._on_session_update(event)
        elif etype == "ping":
            await self._send_json({"type": "pong"})
        else:
            await self._error(f"unknown event type {etype!r}", "unknown_event")

    async def _on_delta(self, text: Any) -> bool:
        """Buffer a text delta; return True when it was accepted."""
        if not isinstance(text, str):
            await self._error("'text' must be a string", "invalid_text")
            return False
        if len(text) > MAX_DELTA_CHARS:
            await self._error(
                f"text delta of {len(text)} chars exceeds {MAX_DELTA_CHARS}",
                "payload_too_large",
            )
            return False
        if len(self._buffer.pending) + len(text) > MAX_PENDING_CHARS:
            self._buffer.reset()
            await self._error(
                f"unsynthesized text exceeded {MAX_PENDING_CHARS} chars; buffer cleared. "
                f"Send 'input.text.done' to flush, or include sentence punctuation.",
                "buffer_overflow",
            )
            return False
        for clause in self._buffer.push(text):
            self._enqueue(_KIND_CHUNK, clause)
        return True

    def _commit(self) -> None:
        for clause in self._buffer.flush():
            self._enqueue(_KIND_CHUNK, clause)
        # Restart first-chunk sizing so the next response also starts fast.
        self._buffer.reset()
        self._enqueue(_KIND_DONE, "")

    async def _cancel(self) -> None:
        self._gen += 1
        self._buffer.reset()
        await self._send_json({"type": "cancelled"})

    async def _on_session_update(self, event: Dict[str, Any]) -> None:
        payload = {k: v for k, v in event.items() if k != "type"}
        try:
            upd = RealtimeSessionUpdate(**payload)
        except ValidationError as e:
            await self._error(f"invalid session.update: {e}", "invalid_session_update")
            return
        if upd.lang is not None and upd.lang not in AVAILABLE_LANGUAGES:
            await self._error(
                f"unsupported lang {upd.lang!r}; valid: {', '.join(AVAILABLE_LANGUAGES)}",
                "unsupported_lang",
            )
            return
        if upd.format is not None:
            try:
                upd.format = coerce_pcm_format(upd.format)
            except UnsupportedAudioFormat:
                await self._error(
                    f"unsupported format {upd.format!r}; valid: {', '.join(PCM_FORMATS)}",
                    "unsupported_response_format",
                )
                return

        new_cfg = _apply_update(self._cfg, upd)

        if upd.voice is not None and upd.voice != self._cfg.voice:
            try:
                self._style = await to_thread.run_sync(
                    partial(resolve_voice, self._state, upd.voice)
                )
            except UnknownVoice:
                await self._error(f"unknown voice {upd.voice!r}", "unknown_voice")
                return

        sizing_changed = (
            new_cfg.first_chunk_chars,
            new_cfg.max_chunk_chars,
            new_cfg.min_chunk_chars,
        ) != (
            self._cfg.first_chunk_chars,
            self._cfg.max_chunk_chars,
            self._cfg.min_chunk_chars,
        )
        self._cfg = new_cfg
        if sizing_changed:
            # Re-run whatever is still buffered through a buffer with the new
            # targets, so the change takes effect immediately.
            pending = self._buffer.pending
            self._buffer = ClauseBuffer(
                first_chunk_chars=new_cfg.first_chunk_chars,
                max_chunk_chars=new_cfg.max_chunk_chars,
                min_chunk_chars=new_cfg.min_chunk_chars,
            )
            if pending:
                for clause in self._buffer.push(pending):
                    self._enqueue(_KIND_CHUNK, clause)

        await self._send_json(dict(type="session.updated", **self.config_payload()))

    def _enqueue(self, kind: str, clause: str) -> None:
        # The stream is unbounded, so this never blocks and the receive loop
        # stays available to handle `cancel`.
        self._send.send_nowait((kind, clause, self._gen))

    # --- synthesis --------------------------------------------------------

    async def _sender_loop(self) -> None:
        async for kind, clause, gen in self._recv:
            if gen != self._gen:
                continue  # cancelled while queued
            if gen != self._sender_gen:
                self._sender_gen = gen
                self._chunks = 0
                self._duration = 0.0
            if kind == _KIND_DONE:
                await self._send_json(
                    {
                        "type": "response.done",
                        "chunks": self._chunks,
                        "duration_s": round(self._duration, 3),
                    }
                )
                self._chunks = 0
                self._duration = 0.0
                continue
            await self._synthesize_and_send(clause, gen)

    async def _synthesize_and_send(self, clause: str, gen: int) -> None:
        cfg = self._cfg  # snapshot: a session.update mid-chunk applies next time
        style = self._style
        index = self._chunks
        try:
            chunk = await to_thread.run_sync(
                partial(self._blocking_synth, clause, style, cfg, index)
            )
        except Exception as e:
            logger.exception("realtime synthesis failed")
            await self._error(f"synthesis failed: {e}", "synthesis_failed", type_="server_error")
            return
        if gen != self._gen:
            return  # barged in while the vocoder was running
        try:
            pcm = encode_pcm(chunk.wav, cfg.fmt)
        except UnsupportedAudioFormat:
            await self._error(f"unsupported format {cfg.fmt!r}", "unsupported_response_format")
            return
        await self._send_json(
            {
                "type": "audio.chunk",
                "index": index,
                "text": chunk.text,
                "duration_s": round(chunk.duration_s, 3),
                "samples": int(chunk.wav.shape[0]),
                "bytes": len(pcm),
                "steps": chunk.total_steps,
            }
        )
        await self._send_pcm(pcm)
        self._chunks += 1
        self._duration += chunk.duration_s

    def _blocking_synth(self, clause: str, style: Any, cfg: _SessionConfig, index: int):
        # Runs in a worker thread. ONNX Runtime sessions are not safe under
        # concurrent use, so share the same lock the HTTP routes take.
        with self._state.synth_lock:
            return synthesize_clause(
                self._state.tts,
                clause,
                style,
                index=index,
                lang=cfg.lang,
                speed=cfg.speed,
                total_steps=cfg.steps,
                first_chunk_steps=cfg.first_chunk_steps,
            )


def _config_from_query(ws: WebSocket) -> Tuple[Optional[_SessionConfig], Optional[str]]:
    """Build the initial config from query parameters.

    Returns ``(config, None)`` or ``(None, error_message)``.
    """
    params = {k: v for k, v in ws.query_params.items() if v != ""}
    try:
        upd = RealtimeSessionUpdate(**params)
    except ValidationError as e:
        return None, f"invalid query parameters: {e}"
    if upd.lang is not None and upd.lang not in AVAILABLE_LANGUAGES:
        return None, f"unsupported lang {upd.lang!r}; valid: {', '.join(AVAILABLE_LANGUAGES)}"
    if upd.format is not None:
        try:
            upd.format = coerce_pcm_format(upd.format)
        except UnsupportedAudioFormat:
            return None, f"unsupported format {upd.format!r}; valid: {', '.join(PCM_FORMATS)}"
    return _apply_update(_SessionConfig(), upd), None


def register_realtime(app: FastAPI) -> None:
    """Attach ``WS /v1/realtime`` to ``app``.

    Called from :func:`supertonic.server.app.create_app` alongside
    :func:`supertonic.server.routes.register_routes`.
    """
    router = APIRouter()

    @router.websocket("/v1/realtime")
    async def realtime(ws: WebSocket) -> None:
        state: "ServerState" = ws.app.state.server_state
        await ws.accept()

        if state.tts is None:
            await ws.send_json(
                {
                    "type": "error",
                    "error": {
                        "message": "server not ready",
                        "type": "server_error",
                        "code": "not_ready",
                    },
                }
            )
            await ws.close(code=_CLOSE_TRY_AGAIN_LATER)
            return

        cfg, err = _config_from_query(ws)
        if cfg is None:
            await ws.send_json(
                {
                    "type": "error",
                    "error": {
                        "message": err,
                        "type": "invalid_request_error",
                        "code": "invalid_session",
                    },
                }
            )
            await ws.close()
            return

        try:
            style = await to_thread.run_sync(partial(resolve_voice, state, cfg.voice))
        except UnknownVoice:
            await ws.send_json(
                {
                    "type": "error",
                    "error": {
                        "message": f"unknown voice {cfg.voice!r}",
                        "type": "invalid_request_error",
                        "code": "unknown_voice",
                    },
                }
            )
            await ws.close()
            return

        session = _RealtimeSession(ws, state, cfg, style)
        try:
            await session.run()
        except Exception:
            logger.exception("realtime session failed")
            with anyio.CancelScope(shield=True):
                try:
                    await ws.close(code=1011)
                except Exception:
                    logger.debug("could not close realtime socket after failure", exc_info=True)

    app.include_router(router)
