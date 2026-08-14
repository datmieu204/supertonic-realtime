"""
Example 10: Realtime Streaming (in-process)

`synthesize()` returns one array after the whole text is done. For a voice
agent that is too late — you want the first words playing while the rest is
still being generated.

`synthesize_stream()` yields audio clause by clause, so playback starts after
the first clause instead of the last. It accepts a plain string or any
iterable of text fragments (e.g. an LLM's streamed tokens).
"""

import os
import time

import numpy as np

from supertonic import TTS

os.makedirs("outputs/test10", exist_ok=True)

tts = TTS()
style = tts.get_voice_style("M1")


def llm_tokens():
    """Stand-in for an LLM streaming its answer a few characters at a time."""
    answer = (
        "Sure, I can help with that. "
        "Your flight leaves at nine forty in the morning, "
        "and the gate is B twelve. "
        "Would you like me to add it to your calendar?"
    )
    for i in range(0, len(answer), 7):
        time.sleep(0.02)  # pretend the model is thinking
        yield answer[i : i + 7]


# --- batch: nothing is heard until the whole answer is synthesized ---------
start = time.perf_counter()
wav_batch, _ = tts.synthesize("".join(llm_tokens()), voice_style=style, lang="en")
batch_ttfa = time.perf_counter() - start

# --- streaming: the first clause arrives while the rest is still generating -
start = time.perf_counter()
chunks = []
first_chunk_at = None
for chunk in tts.synthesize_stream(llm_tokens(), style, lang="en"):
    if first_chunk_at is None:
        first_chunk_at = time.perf_counter() - start
    chunks.append(chunk.wav)
    print(f"  [{chunk.index}] {chunk.duration_s:5.2f}s  steps={chunk.total_steps}  {chunk.text}")
stream_total = time.perf_counter() - start

wav_stream = np.concatenate(chunks)
tts.save_audio(wav_stream.reshape(1, -1), "outputs/test10/streamed.wav")

print()
print(f"batch      — first audio after {batch_ttfa:.2f}s")
print(f"streaming  — first audio after {first_chunk_at:.2f}s (full answer in {stream_total:.2f}s)")
print(f"saved {len(wav_stream) / tts.sample_rate:.2f}s → outputs/test10/streamed.wav")
