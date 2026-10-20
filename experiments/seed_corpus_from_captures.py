"""
seed_corpus_from_captures.py -- build a corpus from the existing captures.

WHY.

The labelled conditions A, B and C already exist as raw NV12 captures from the
2026-10-18 session. The teacher benchmark needs them in the corpus format
(motion maps plus a label sidecar), and building that converter means the
benchmark can be run and VERIFIED tonight rather than waiting for a new
recording session.

WHAT THIS IS AND IS NOT.

It IS: a mechanical conversion, plus a genuine (if small) benchmark of a text
model on the ASCII motion representation.

It IS NOT: a good dataset. Three sessions, one per condition, one speaker, one
room, one lighting state. The B condition's head turn was too weak to register
at all -- `ROW-BAND-REMOVAL.md` established that facing-vs-turned is
indistinguishable from a time-shifted control -- so the "facing" question has
no signal in this data and any score on it is noise.

The "speaking" question IS valid here: A is speaking and C is silent, and
those differ by 4.914 counts against A's own within-condition split of 2.878
(1.7x), with
79% of the excess localised. That is a real, measured difference, so a teacher
that can read it should be able to tell A from C. This is the first honest test
of whether a text model can read motion rendered as characters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W, H = 160, 96
FB = W * H * 3 // 2
YS = W * H

SPECS = [
    ("A", "/mnt/c/proart_capture/A.nv12", True, "facing",
     "facing the camera, speaking"),
    ("B", "/mnt/c/proart_capture/B.nv12", True, "turned",
     "head turned away, speaking"),
    ("C", "/mnt/c/proart_capture/C.nv12", False, "facing",
     "facing the camera, silent"),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/c/proart_capture/corpus")
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--filter", default="row",
                    choices=("none", "row", "row+median3", "row+median5"),
                    help="which filter to apply before computing motion maps")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, path, speak, pose, note in SPECS:
        p = Path(path)
        if not p.is_file():
            print(f"skip {name}: {path} missing")
            continue
        raw = np.frombuffer(p.read_bytes(), dtype=np.uint8)
        n = min(a.max_frames, len(raw) // FB)
        Y = raw[:n * FB].reshape(n, FB)[:, :YS].reshape(n, H, W).astype(np.float64)

        Yr = Y
        if "row" in a.filter:
            prof = np.median(Y, axis=2)
            g = prof.mean(axis=1, keepdims=True)
            Yr = Y - (prof - g)[:, :, None]
        k = 3 if "median3" in a.filter else (5 if "median5" in a.filter else 1)
        if k > 1:
            pad = k // 2
            P = np.pad(Yr, ((pad, pad), (0, 0), (0, 0)), mode="edge")
            Yr = np.median(np.stack([P[i:i + len(Yr)] for i in range(k)]), axis=0)
        d = np.abs(np.diff(Yr, axis=0))
        maps = d.reshape(d.shape[0], H // 2, 2, W // 2, 2).mean(axis=(2, 4))

        hi = float(np.percentile(maps, 99.5))
        hi = max(hi, 1e-6)
        q = np.clip(maps / hi * 255.0, 0, 255).astype(np.uint8)

        sfx = f"_{a.tag}" if a.tag else ""
        npz = out / f"{name}{sfx}.npz"
        np.savez_compressed(npz, maps=q, scale=np.float32(hi))
        meta = {
            "condition": name, "kind": "condition", "pose": pose,
            "speak": speak, "note": note,
            "speech_frac": None, "n_frames": int(len(Y)),
            "seconds": len(Y) / 30.0, "map_shape": list(q.shape[1:]),
            "map_scale": hi,
            "drift_raw": float(np.abs(np.diff(Y, axis=0)).mean()),
            "drift_after_row_removal": float(np.abs(np.diff(Yr, axis=0)).mean()),
            "source": path, "seeded": True,
        }
        meta["filter"] = a.filter
        (out / f"{name}{sfx}.json").write_text(json.dumps(meta, indent=2))
        print(f"{name}: {len(Y)} frames -> {npz.name} "
              f"({npz.stat().st_size / 1024:.0f} kB)  "
              f"drift raw {meta['drift_raw']:.1f} -> row {meta['drift_after_row_removal']:.1f}")


if __name__ == "__main__":
    main()
