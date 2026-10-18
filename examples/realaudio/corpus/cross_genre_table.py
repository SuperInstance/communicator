"""Cross-genre pipeline table: do the measured tracks separate the cells?

Day-0 stats per corpus cell: voiced %, F0 distribution, contiguous vs
cross-gap octave jumps (the Day-3 decomposition), phrase-tempo spread
(E4's map), segment structure. The corpus matrix was chosen to bracket
cadence (oration vs children's); this table asks whether the MEASURED
tracks alone already separate the genres.
"""
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src import wavio, tempo                          # noqa: E402
from src.pitch import PitchTracker, smooth_octaves    # noqa: E402
from src.score import measure                         # noqa: E402

rows = []
for wav in sorted((Path(__file__).parent).glob("*.wav")):
    a = wavio.read_wav(wav)
    m = measure(a.samples, a.sample_rate)
    pf = m["pitch_frames"]
    voiced = [f for f in pf if f.hz is not None]
    hzs = sorted(f.hz for f in voiced)
    contig = cross = 0
    prev = None
    for f in voiced:
        if prev is not None:
            r = math.log2(f.hz / prev.hz) if prev.hz else 0
            if abs(r) > 0.75:
                if f.index == prev.index + 1:
                    contig += 1
                else:
                    cross += 1
        prev = f
    sm = smooth_octaves(pf)
    svoiced = [f for f in sm if f.hz is not None]
    scontig = sum(1 for x, y in zip(svoiced, svoiced[1:])
                  if y.index == x.index + 1 and abs(math.log2(y.hz / x.hz)) > 0.75)
    tmap = [e["bpm"] for e in tempo.phrase_tempo_map(m["energy_frames"], m["segments"])
            if e["bpm"]]
    cell = wav.stem.split("_")[0]
    rows.append({
        "cell": cell, "dur": a.duration(),
        "voiced%": round(100 * len(voiced) / max(1, len(pf))),
        "F0med": round(statistics.median(hzs)) if hzs else 0,
        "F0p10": round(hzs[len(hzs) // 10]) if hzs else 0,
        "F0p90": round(hzs[9 * len(hzs) // 10]) if hzs else 0,
        "oct%": round(100 * (contig + cross) / max(1, len(voiced) - 1), 1),
        "contig": contig, "cross": cross,
        "after%": round(100 * scontig / max(1, len(svoiced) - 1), 1),
        "segs": len(m["segments"]),
        "medPhrase": round(statistics.median(
            s.duration() for s in m["segments"] if s.speech), 2) if any(
            s.speech for s in m["segments"]) else 0,
        "tempoSD%": round(100 * statistics.pstdev(tmap) / statistics.mean(tmap), 0)
        if len(tmap) > 2 else None,
    })

hdr = list(rows[0].keys())
print(" | ".join(f"{h:>10s}" for h in hdr))
for r in rows:
    print(" | ".join(f"{str(r[h]):>10s}" for h in hdr))
