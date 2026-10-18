"""
Day-0 real-speech measurement — reproducible recipe for EXPERIMENTS-AUDIO.md §3.

Decode (once, needs the scratch venv's miniaudio — no ffmpeg on this box):

    /home/proart/.local/audio-venv/bin/python - <<'PY'
    import miniaudio, wave, struct
    src = miniaudio.decode_file("alice01.mp3")
    sr, ch = src.sample_rate, src.nchannels
    s = struct.unpack(f"<{len(src.samples)}h", bytes(src.samples))
    n = len(s) // ch
    mono = [sum(s[i*ch:(i+1)*ch]) / ch / 32768.0
            for i in range(int(20*sr), int(110*sr))]
    with wave.open("alice01_excerpt.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(max(-1, min(1, x))*32767))
                               for x in mono))
    PY

Then this script (any python with the repo's stdlib modules):

    python examples/realaudio/measure_excerpt.py alice01_excerpt.wav

Numbers from the 2026-10-18 run are recorded in EXPERIMENTS-AUDIO.md §3.
"""
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import wavio                              # noqa: E402
from src.score import measure                      # noqa: E402


def main(path):
    audio = wavio.read_wav(path)
    m = measure(audio.samples, audio.sample_rate)
    pf, segs = m["pitch_frames"], m["segments"]
    voiced = [f for f in pf if f.hz is not None]
    hzs = sorted(f.hz for f in voiced)

    jumps = halvings = 0
    prev = None
    for f in voiced:
        if prev is not None:
            r = math.log2(f.hz / prev.hz)
            jumps += abs(r) > 0.75
            halvings += r < -0.75
        prev = f

    speech = [s for s in segs if s.speech]
    print(f"frames {len(pf)} · voiced {len(voiced)} "
          f"({100*len(voiced)/max(1,len(pf)):.0f}%) · poisoned {m['poisoned']}")
    if hzs:
        print(f"F0 median {statistics.median(hzs):.0f}Hz · "
              f"p10 {hzs[len(hzs)//10]:.0f} · p90 {hzs[9*len(hzs)//10]:.0f} · "
              f"min {hzs[0]:.0f} · max {hzs[-1]:.0f}")
    print(f"octave jumps {jumps}/{max(1,len(voiced)-1)} "
          f"({100*jumps/max(1,len(voiced)-1):.1f}%) · halvings {halvings}")
    print(f"segments {len(segs)} · speech {len(speech)} · "
          f"median phrase {statistics.median(s.duration() for s in speech):.2f}s")
    print(f"bpm {m['bpm'] and round(m['bpm'], 1)} (crude heuristic — see E4)")
    print("inflection", dict(Counter(e.direction.name
                                     for e in m["inflection_events"])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "alice01_excerpt.wav")
