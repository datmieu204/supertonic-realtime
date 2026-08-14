"""
Example 11: Realtime WebSocket client (WS /v1/realtime)

Start the server first:

    supertonic serve                       # ws://127.0.0.1:7788/v1/realtime

Then run this script. It shows the two things a voice agent needs:

1. Push text as it is produced (`input.text.delta`) and receive audio clause by
   clause, without waiting for the answer to finish.
2. Barge-in: `cancel` drops queued clauses and discards the chunk currently in
   the synthesizer.

Requires the `websockets` package (installed with `supertonic[serve]` via
`uvicorn[standard]`); audio playback additionally needs `supertonic[playback]`.
"""

import asyncio
import json
import os

import numpy as np
import websockets

URL = os.environ.get("SUPERTONIC_REALTIME_URL", "ws://127.0.0.1:7788/v1/realtime")

ANSWER = (
    "Sure, I can help with that. "
    "Your flight leaves at nine forty in the morning, and the gate is B twelve. "
    "Would you like me to add it to your calendar?"
)


def pcm16_to_float(payload: bytes) -> np.ndarray:
    """Decode the server's default `pcm_s16le` frames."""
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32767.0


async def collect_response(ws, *, cancel_after=None):
    """Read one response; optionally send `cancel` after N chunks (barge-in)."""
    audio = []
    while True:
        message = await ws.recv()
        if isinstance(message, bytes):
            audio.append(pcm16_to_float(message))
            if cancel_after is not None and len(audio) >= cancel_after:
                print("  <barge-in> sending cancel")
                await ws.send(json.dumps({"type": "cancel"}))
                cancel_after = None
            continue

        event = json.loads(message)
        etype = event.get("type")
        if etype == "audio.chunk":
            print(f"  [{event['index']}] {event['duration_s']:5.2f}s  {event['text']}")
        elif etype == "response.done":
            print(f"  done: {event['chunks']} chunk(s), {event['duration_s']:.2f}s")
            return audio
        elif etype == "cancelled":
            print("  cancelled")
            return audio
        elif etype == "error":
            raise RuntimeError(event["error"]["message"])


async def main():
    async with websockets.connect(f"{URL}?voice=M1&lang=en") as ws:
        created = json.loads(await ws.recv())
        sample_rate = created["sample_rate"]
        print(f"connected: {created['model']} @ {sample_rate} Hz, format={created['format']}")

        # --- stream text in as an LLM would produce it ---------------------
        print("\nstreaming an answer:")
        for i in range(0, len(ANSWER), 7):
            await ws.send(json.dumps({"type": "input.text.delta", "text": ANSWER[i : i + 7]}))
            await asyncio.sleep(0.02)
        await ws.send(json.dumps({"type": "input.text.done"}))
        audio = await collect_response(ws)

        # --- barge-in: interrupt a long answer after its first chunk -------
        print("\ninterrupting a long answer:")
        await ws.send(json.dumps({"type": "speak", "text": ANSWER + " " + ANSWER}))
        await collect_response(ws, cancel_after=1)

    wav = np.concatenate(audio) if audio else np.zeros(1, dtype=np.float32)
    try:
        import sounddevice as sd

        print(f"\nplaying {len(wav) / sample_rate:.2f}s ...")
        sd.play(wav, sample_rate)
        sd.wait()
    except ImportError:
        import soundfile as sf

        os.makedirs("outputs/test11", exist_ok=True)
        sf.write("outputs/test11/realtime.wav", wav, sample_rate)
        print(f"\nsaved {len(wav) / sample_rate:.2f}s → outputs/test11/realtime.wav")


if __name__ == "__main__":
    asyncio.run(main())
