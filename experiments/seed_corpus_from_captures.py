"""
seed_corpus_from_captures.py -- build a corpus from the existing captures.

WHY.

The labelled conditions A, B and C already exist as raw NV12 captures from the
2026-10-18 session. The teacher benchmark needs them in the corpus format
(motion maps plus a label sidecar), and building that converter means the
benchmark can be run and VERIFIED tonight rather than waiting for a new
recording session.

THE FRAME LAYOUT IS DETECTED, NOT ASSUMED.

The captures on disk are NOT standard NV12. `C:\\proart_capture\\nv12small.py`
writes, per 160x96 frame:

    Y plane    15,360 bytes   (160 * 96)
    U,V         3,840 bytes   (80 x 24 samples, INTERLEAVED)
    -------------------------
    total      19,200 bytes   = W*H*5/4

So the stride is 19,200 bytes, not the `W*H*3//2` = 23,040 that every
experiment in this repo used to assume. Reading a 19,200-byte file at 23,040
tears all but every sixth frame -- rows 0..71 from frame k and rows 72..95 from
frame k+1 -- and the "drift" that measured was the tear, not the room.

The layout is therefore read from the file's CONTENT by `src/nv12.py`: the
candidate whose luma plane has the smallest mean consecutive |diff| wins, and
a file that cannot decide between candidates is an ERROR rather than a guess.
Nothing in this file computes a stride by hand. See `src/nv12.py` and
`docs/research/NV12-STRIDE.md`.

THE OLD CORPUS SURVIVES.

The first seeding wrote to `C:\\proart_capture\\corpus` with a 600-frame cap and
the wrong stride, so its motion maps are largely stride artefact. It is left
exactly where it is -- for comparison, and because overwriting a corpus other
experiments have already read is not this script's call. This one writes to
`C:\\proart_capture\\corpus_v2`.

WHAT THIS IS AND IS NOT.

It IS: a mechanical conversion, plus a genuine (if small) benchmark of a text
model on the ASCII motion representation.

It IS NOT: a good dataset. Three sessions, one per condition, one speaker, one
room, one lighting state. The B condition's head turn was too weak to register
at all -- `ROW-BAND-REMOVAL.md` established that facing-vs-turned is
indistinguishable from a time-shifted control -- so the "facing" question has
no signal in this data and any score on it is noise.

The "speaking" question IS valid here: A is speaking and C is silent, and
those differ by 0.2506 counts against A's own within-condition split of
0.2006 (1.25x), with
31% of the excess localised. That is a real, measured difference -- but it
does not clear the 1.25x margin `experiments/pose_test.py` was given before
the numbers were seen, so whether the camera channel carries speech at all is
OPEN, not settled. The first honest test of whether a text model can read
motion rendered as characters has not happened yet.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nv12 import read_luma  # noqa: E402

W, H = 160, 96
YS = W * H
FPS = 30.0

SPECS = [
    ("A", "/mnt/c/proart_capture/A.nv12", True, "facing",
     "facing the camera, speaking"),
    ("B", "/mnt/c/proart_capture/B.nv12", True, "turned",
     "head turned away, speaking"),
    ("C", "/mnt/c/proart_capture/C.nv12", False, "facing",
     "facing the camera, silent"),
]


def sha256_file(path) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/c/proart_capture/corpus_v2",
                    help="where to write the corpus. The FIRST corpus "
                         "(/mnt/c/proart_capture/corpus) is left alone: it was "
                         "seeded at the wrong stride and has to survive for "
                         "comparison.")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap the frames used per session. 0 (the default) "
                         "means every frame in the capture. The old corpus was "
                         "capped at 600, which truncated each 30 s session to "
                         "20 s.")
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

        # The layout is DETECTED from the file's content, by src/nv12.py. An
        # ambiguous file raises here rather than being read at a guessed
        # stride.
        luma, layout = read_luma(p, width=W, height=H)
        n = len(luma) if a.max_frames <= 0 else min(a.max_frames, len(luma))
        if n <= 0:
            print(f"skip {name}: no complete frames")
            continue
        Y = luma[:n].astype(np.float64)

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
            "seconds": len(Y) / FPS, "map_shape": list(q.shape[1:]),
            "map_scale": hi,
            "drift_raw": float(np.abs(np.diff(Y, axis=0)).mean()),
            "drift_after_row_removal": float(np.abs(np.diff(Yr, axis=0)).mean()),
            "source": path, "seeded": True,
            # -- the identity of the input, added 2026-10-20 ------------------
            # The frame layout that was actually read, and the evidence for
            # believing it. `layout_evidence` is the detection report from
            # src/nv12.py, including the margin over the runner-up: a layout
            # that cannot say how sure it is must not be acted on.
            "layout": layout.name,
            "stride_bytes": layout.stride,
            "source_sha256": sha256_file(p),
            "source_bytes": p.stat().st_size,
            "layout_evidence": layout.as_dict(),
            "fps": FPS,
            "provenance": {
                "generated_by": "experiments/seed_corpus_from_captures.py",
                "script_path": str(Path(__file__).resolve()),
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "python": sys.executable,
                "command_line": " ".join(sys.argv),
                "out_dir": str(out.resolve()),
                "max_frames": a.max_frames,
                "filter": a.filter,
            },
        }
        meta["filter"] = a.filter
        (out / f"{name}{sfx}.json").write_text(json.dumps(meta, indent=2))
        print(f"{name}: {len(Y)} frames -> {npz.name} "
              f"({npz.stat().st_size / 1024:.0f} kB)  "
              f"layout {layout.name} stride {layout.stride}  "
              f"drift raw {meta['drift_raw']:.1f} -> row {meta['drift_after_row_removal']:.1f}")


if __name__ == "__main__":
    main()
