"""Tests for the realtime WebSocket endpoint (``WS /v1/realtime``).

Like :mod:`tests.test_server_routes`, the real ``TTS`` is replaced with a
stand-in so no ONNX session is created and no model is downloaded. The focus
here is the wire protocol: event handling, the metadata-then-binary framing,
session updates, and barge-in.
"""

from __future__ import annotations

import struct
import threading

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from supertonic.server import ServerState, create_app  # noqa: E402

SAMPLE_RATE = 44100


class _FakeStyle:
    def __init__(self, source: str) -> None:
        self.source = source


class FakeTTS:
    """Stand-in for :class:`supertonic.TTS` with a controllable synth delay."""

    def __init__(self) -> None:
        self.sample_rate = SAMPLE_RATE
        self.voice_style_names = ["M1", "F1", "F2"]
        self.calls: list[dict] = []
        # Set to block inside synthesize(), so a test can send `cancel` while
        # a chunk is in the (fake) ONNX session.
        self.gate: threading.Event | None = None
        self.entered = threading.Event()

    def get_voice_style(self, name: str) -> _FakeStyle:
        if name not in self.voice_style_names:
            raise FileNotFoundError(f"voice not found: {name}")
        return _FakeStyle(source=f"builtin:{name}")

    def get_voice_style_from_path(self, path) -> _FakeStyle:
        return _FakeStyle(source=f"custom:{path}")

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
                "voice_style_source": getattr(voice_style, "source", None),
                "total_steps": total_steps,
                "speed": speed,
                "lang": lang,
            }
        )
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(timeout=5)
        n = max(1, len(text) * 10)
        return np.zeros((1, n), dtype=np.float32), np.array([n / SAMPLE_RATE])


@pytest.fixture
def fake_state(tmp_path):
    fake = FakeTTS()
    state = ServerState(model="supertonic-3", tts=fake, custom_styles_dir=tmp_path)
    return state, fake


@pytest.fixture
def client(fake_state):
    state, _ = fake_state
    app = create_app(state=state)
    with TestClient(app) as c:
        yield c


def _drain_until(ws, wanted: str, limit: int = 40):
    """Collect events until ``wanted`` arrives; return the list of JSON events.

    Binary audio frames are recorded as ``{"type": "__binary__", "size": n}``
    so ordering assertions can cover both frame kinds.
    """
    events = []
    for _ in range(limit):
        msg = ws.receive()
        if msg["type"] == "websocket.close":
            raise AssertionError(f"socket closed before {wanted!r}: {events}")
        if msg.get("bytes") is not None:
            events.append({"type": "__binary__", "size": len(msg["bytes"]), "raw": msg["bytes"]})
            continue
        import json

        event = json.loads(msg["text"])
        events.append(event)
        if event.get("type") == wanted:
            return events
    raise AssertionError(f"never received {wanted!r}: {events}")


# --- session setup --------------------------------------------------------


def test_session_created_announces_the_audio_contract(client):
    with client.websocket_connect("/v1/realtime") as ws:
        created = ws.receive_json()
    assert created["type"] == "session.created"
    assert created["sample_rate"] == SAMPLE_RATE
    assert created["format"] == "pcm_s16le"
    assert created["voice"] == "M1"
    assert created["model"] == "supertonic-3"


def test_query_parameters_configure_the_session(client):
    with client.websocket_connect("/v1/realtime?voice=F1&lang=en&speed=1.2&format=pcm_f32le") as ws:
        created = ws.receive_json()
    assert created["voice"] == "F1"
    assert created["lang"] == "en"
    assert created["speed"] == 1.2
    assert created["format"] == "pcm_f32le"


def test_unknown_voice_in_query_is_rejected(client):
    with client.websocket_connect("/v1/realtime?voice=nope") as ws:
        event = ws.receive_json()
    assert event["type"] == "error"
    assert event["error"]["code"] == "unknown_voice"


def test_unsupported_lang_in_query_is_rejected(client):
    with client.websocket_connect("/v1/realtime?lang=xx") as ws:
        event = ws.receive_json()
    assert event["type"] == "error"
    assert event["error"]["code"] == "invalid_session"


def test_unsupported_format_in_query_is_rejected(client):
    with client.websocket_connect("/v1/realtime?format=mp3") as ws:
        event = ws.receive_json()
    assert event["type"] == "error"
    assert event["error"]["code"] == "invalid_session"


def test_not_ready_closes_the_socket(tmp_path):
    state = ServerState(model="supertonic-3", tts=None, custom_styles_dir=tmp_path)
    app = create_app(state=state)
    # No lifespan: `state.tts` stays None, mimicking a connection that arrives
    # while the model is still loading.
    with TestClient(app).websocket_connect("/v1/realtime") as ws:
        event = ws.receive_json()
    assert event["error"]["code"] == "not_ready"


# --- speaking -------------------------------------------------------------


def test_speak_streams_metadata_then_binary(client, fake_state):
    _, fake = fake_state
    with client.websocket_connect("/v1/realtime?lang=en") as ws:
        ws.receive_json()  # session.created
        ws.send_json({"type": "speak", "text": "Hello there. How are you today, friend?"})
        events = _drain_until(ws, "response.done")

    kinds = [e["type"] for e in events]
    # Every audio.chunk is immediately followed by exactly one binary frame.
    assert kinds == ["audio.chunk", "__binary__", "audio.chunk", "__binary__", "response.done"]

    meta = [e for e in events if e["type"] == "audio.chunk"]
    binaries = [e for e in events if e["type"] == "__binary__"]
    assert [m["index"] for m in meta] == [0, 1]
    assert meta[0]["text"] == "Hello there."
    assert [m["bytes"] for m in meta] == [b["size"] for b in binaries]
    # 16-bit samples: two bytes per sample.
    assert all(m["bytes"] == m["samples"] * 2 for m in meta)

    done = events[-1]
    assert done["chunks"] == 2
    assert done["duration_s"] == pytest.approx(sum(m["duration_s"] for m in meta), abs=0.01)
    assert {c["lang"] for c in fake.calls} == {"en"}


def test_first_chunk_uses_fewer_steps(client, fake_state):
    _, fake = fake_state
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "speak", "text": "Hello there. How are you today, friend?"})
        _drain_until(ws, "response.done")
    assert [c["total_steps"] for c in fake.calls] == [4, 8]


def test_deltas_are_buffered_until_a_clause_completes(client):
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        # No boundary yet — nothing should be synthesized.
        ws.send_json({"type": "input.text.delta", "text": "Hello the"})
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"

        ws.send_json({"type": "input.text.delta", "text": "re. And a tail"})
        first = ws.receive_json()
        assert first["type"] == "audio.chunk"
        assert first["text"] == "Hello there."
        ws.receive_bytes()

        ws.send_json({"type": "input.text.done"})
        events = _drain_until(ws, "response.done")
    assert [e["type"] for e in events] == ["audio.chunk", "__binary__", "response.done"]
    assert events[0]["text"] == "And a tail"


def test_two_responses_each_restart_chunk_numbering(client, fake_state):
    _, fake = fake_state
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "speak", "text": "First answer here."})
        first = _drain_until(ws, "response.done")
        ws.send_json({"type": "speak", "text": "Second answer here."})
        second = _drain_until(ws, "response.done")
    assert first[0]["index"] == 0
    assert second[0]["index"] == 0
    # Both responses get the cheap first chunk.
    assert [c["total_steps"] for c in fake.calls] == [4, 4]


def test_pcm_f32le_emits_four_bytes_per_sample(client):
    with client.websocket_connect("/v1/realtime?format=pcm_f32le") as ws:
        ws.receive_json()
        ws.send_json({"type": "speak", "text": "Hello there."})
        meta = ws.receive_json()
        payload = ws.receive_bytes()
    assert meta["bytes"] == meta["samples"] * 4
    assert len(payload) == meta["samples"] * 4
    # Silence from the fake decodes to 0.0.
    assert struct.unpack("<f", payload[:4])[0] == 0.0


# --- session.update -------------------------------------------------------


def test_session_update_changes_voice(client, fake_state):
    _, fake = fake_state
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "session.update", "voice": "F2", "speed": 1.3})
        updated = ws.receive_json()
        assert updated["type"] == "session.updated"
        assert updated["voice"] == "F2"
        ws.send_json({"type": "speak", "text": "Hello there."})
        _drain_until(ws, "response.done")
    assert fake.calls[0]["voice_style_source"] == "builtin:F2"
    assert fake.calls[0]["speed"] == 1.3


def test_session_update_rejects_unknown_voice_without_dropping_the_session(client):
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "session.update", "voice": "nope"})
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["error"]["code"] == "unknown_voice"
        # Session survives: the old voice still works.
        ws.send_json({"type": "speak", "text": "Hello there."})
        events = _drain_until(ws, "response.done")
    assert events[0]["type"] == "audio.chunk"


def test_session_update_rejects_out_of_range_speed(client):
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "session.update", "speed": 9.0})
        err = ws.receive_json()
    assert err["error"]["code"] == "invalid_session_update"


def test_session_update_applies_new_chunk_sizing_to_buffered_text(client):
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "input.text.delta", "text": "one two three four five six seven"})
        ws.send_json({"type": "session.update", "max_chunk_chars": 10, "first_chunk_chars": 10})
        # The buffered text is re-run through the smaller buffer, so chunks
        # start arriving without any further input.
        first = ws.receive_json()
        while first["type"] == "session.updated":
            first = ws.receive_json()
    assert first["type"] == "audio.chunk"
    assert len(first["text"]) <= 10


# --- errors ---------------------------------------------------------------


def test_invalid_json_is_reported(client):
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_text("not json")
        err = ws.receive_json()
    assert err["error"]["code"] == "invalid_json"


def test_binary_frames_from_the_client_are_rejected(client):
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_bytes(b"\x00\x01")
        err = ws.receive_json()
    assert err["error"]["code"] == "invalid_frame"


def test_unknown_event_type_is_reported(client):
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "nope"})
        err = ws.receive_json()
    assert err["error"]["code"] == "unknown_event"


def test_non_string_text_is_reported(client):
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "input.text.delta", "text": 42})
        err = ws.receive_json()
    assert err["error"]["code"] == "invalid_text"


def test_oversized_delta_is_rejected(client):
    from supertonic.server.realtime import MAX_DELTA_CHARS

    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "input.text.delta", "text": "x" * (MAX_DELTA_CHARS + 1)})
        err = ws.receive_json()
    assert err["error"]["code"] == "payload_too_large"


def test_synthesis_failure_keeps_the_session_alive(client, fake_state, monkeypatch):
    _, fake = fake_state

    def boom(*args, **kwargs):
        raise RuntimeError("onnx exploded")

    monkeypatch.setattr(fake, "synthesize", boom)
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "speak", "text": "Hello there."})
        err = ws.receive_json()
        assert err["error"]["code"] == "synthesis_failed"
        # The response still terminates cleanly, with no chunks.
        done = ws.receive_json()
        assert done["type"] == "response.done"
        assert done["chunks"] == 0
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"


# --- barge-in -------------------------------------------------------------


def test_cancel_drops_queued_and_in_flight_audio(client, fake_state):
    _, fake = fake_state
    gate = threading.Event()
    fake.gate = gate

    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "speak",
                "text": "First sentence here. Second sentence here. Third sentence here.",
            }
        )
        # Wait until the first chunk is inside the (blocked) synthesizer, then
        # barge in: its audio must be discarded, and the queued clauses too.
        assert fake.entered.wait(timeout=5)
        ws.send_json({"type": "cancel"})
        assert ws.receive_json()["type"] == "cancelled"
        gate.set()

        # Nothing from the cancelled response may arrive; a new one still works.
        fake.gate = None
        ws.send_json({"type": "speak", "text": "Fresh answer here."})
        events = _drain_until(ws, "response.done")

    assert [e["type"] for e in events] == ["audio.chunk", "__binary__", "response.done"]
    assert events[0]["text"] == "Fresh answer here."
    assert events[0]["index"] == 0


def test_cancel_clears_buffered_text(client):
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "input.text.delta", "text": "half a sentence with no end"})
        ws.send_json({"type": "cancel"})
        assert ws.receive_json()["type"] == "cancelled"
        # The abandoned tail must not leak into the next response.
        ws.send_json({"type": "speak", "text": "Fresh answer here."})
        events = _drain_until(ws, "response.done")
    assert events[0]["text"] == "Fresh answer here."
