#!/usr/bin/env python
"""Cut a 16k/mono/16-bit WAV excerpt from any audio file miniaudio can decode.

Usage: make_excerpt.py IN START_S DUR_S OUT.wav
"""
import sys
import wave

import miniaudio

IN, START_S, DUR_S, OUT = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]

decoded = miniaudio.decode_file(
    IN, nchannels=1, sample_rate=16000, output_format=miniaudio.SampleFormat.FLOAT32
)
samples = decoded.samples
sr = decoded.sample_rate
total_s = len(samples) / sr
start_i = int(START_S * sr)
end_i = min(len(samples), start_i + int(DUR_S * sr))
if start_i >= len(samples):
    sys.exit(f"start {START_S}s beyond file duration {total_s:.1f}s")
chunk = samples[start_i:end_i]

peak = max((abs(s) for s in chunk), default=0.0)
if peak > 1.0:
    chunk = [s / peak * 0.999 for s in chunk]
ints = [max(-32768, min(32767, int(round(s * 32767.0)))) for s in chunk]

with wave.open(OUT, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(b"".join(int(x).to_bytes(2, "little", signed=True) for x in ints))

print(f"OK {OUT}: {len(chunk)/sr:.2f}s from [{START_S:.1f}, {START_S + len(chunk)/sr:.1f}] of {total_s:.1f}s, peak_in={peak:.3f}")
