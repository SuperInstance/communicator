#!/usr/bin/env python3
"""stt.py -- one WAV -> transcript, via the cached faster-whisper.

USAGE:  audio-venv-python browser_capture/stt.py <wav> [--start S] [--end S]
Prints {"text": "...", "duration_s": n} on stdout. The conversation
loop's ear.
"""
import argparse
import json
import sys
import wave
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--model", default="small.en")
    a = ap.parse_args()

    src = Path(a.wav)
    use = src
    tmp = None
    if a.start > 0 or a.end > 0:
        import numpy as np
        with wave.open(str(src)) as w:
            rate = w.getframerate()
            n = w.getnframes()
            s0 = int(a.start * rate)
            s1 = int(a.end * rate) if a.end > 0 else n
            w.setpos(min(s0, n))
            frames = w.readframes(max(0, s1 - min(s0, n)))
        x = np.frombuffer(frames, "<i2").astype(np.float32) / 32768
        tmp = Path("/tmp/stt_slice.wav")
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
            w.writeframes((x * 32767).astype("<i2").tobytes())
        use = tmp

    # faster-whisper 1.2.1 calls av.open(..., metadata_errors=...) but
    # pyav 19 dropped the kwarg (py3.14 has no older wheels). Shim it:
    # swallow the kwarg -- it was metadata-error tolerance, not behavior
    # we depend on for clean WAVs.
    import av as _av
    _orig_open = _av.open
    def _open_shim(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_open(*args, **kwargs)
    _av.open = _open_shim
    from faster_whisper import WhisperModel
    model = WhisperModel(a.model, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(use), language="en",
                                       vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()
    print(json.dumps({"text": text, "duration_s": round(info.duration, 1)}))

if __name__ == "__main__":
    main()
