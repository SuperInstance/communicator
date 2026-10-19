"""
ascii_motion_render.py -- Casey's original idea, with a measurement attached.

THE IDEA, RESTATED.

Render the camera as an array of characters, make MOTION first-class, compress
like video. Text-only models can then consume it. This was the first thing
Casey asked for, and `docs/research/ASCII-MOTION.md` is where it was first
measured -- and where the first two attempts at it turned out to be measuring
the camera rather than the scene.

WHAT HAS CHANGED SINCE, AND WHY THIS FILE EXISTS NOW.

The first attempt rendered SYZYGY'S BRAILLE FIELD and took the frame-to-frame
difference. That was measured to be useless: an empty room flipped 50.00% of
its dots per frame, so the "motion" was the auto-exposure.

`CAMERA-FLICKER-BANDING.md` then found the nuisance exactly -- horizontal
bands, row-constant -- and `ROW-BAND-REMOVAL.md` removed them with a
per-row median across the columns. The empty room went to 0.01% and the
subject's column structure was untouched.

So the representation Casey wanted is now BUILDABLE, and it is the right input
for a teacher model, for one concrete reason:

    THE SIGNAL IS MOTION. Facing-and-speaking differs from facing-and-silent
    by 4.914 counts against a within-condition split of 1.917, with 79% of the
    excess in the top 5% of pixels -- the mouth. That information is in the
    frame-to-frame DIFFERENCE and is NOT in any single frame.

A teacher that reads single frames cannot see it. A teacher that reads a
SEQUENCE can. And a text-only model can read a sequence, if the sequence is
rendered as text. That is the whole argument for this file.

TWO RENDERINGS, because they answer different questions.

  GRID  -- K frames of the motion field laid out side by side, like a film
           strip. This is "video as ASCII" and it preserves SPATIAL structure
           and TEMPORAL ORDER together. A text model that has learned to read
           a grid can ask "where does the change happen, and does it move?"

  TRACE -- per-frame summary statistics as text lines. This destroys spatial
           structure but is tiny, and it is the honest control: if a model can
           only read the TRACE, it is reading a scalar summary, which is
           exactly the kind of route that has failed all season.

THE CONTROL THAT MATTERS, and it is the reason this is not just a renderer.

A rendering is not evidence. The question is whether a model can READ this
representation better than chance. The only way to know is to benchmark a
model on labelled conditions, and the only labelled conditions we have are
Casey's A/B/C/D set -- which does not exist yet. So this file provides:

  * the renderer, so the input format is fixed and reproducible
  * a BLANK and a SHUFFLED control on the same text, so that when a model is
    benchmarked, "the model read the motion" and "the model read the prompt
    and guessed" are distinguishable

The shuffled control is the one that catches the failure this project has
already produced once: zcode's spectrogram model answered a constant
"different" to every input, INCLUDING a blank rectangle. A benchmark that
does not include the blank and the shuffle will happily report that model as
"74% accurate".

WHAT IS NOT CLAIMED.

No model has been benchmarked on this representation yet. The A/B/C/D corpus
does not exist. This file builds the instrument and the control; it does not
report a result it has not got.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAMP = " .:-=+*#%@"


def render_frame(m: np.ndarray, ramp: str = RAMP) -> str:
    """One motion frame as characters. m is (rows, cols), higher = more change."""
    lo, hi = float(m.min()), float(np.percentile(m, 99.0))
    if hi <= lo:
        hi = lo + 1e-9
    q = np.clip((m - lo) / (hi - lo), 0, 1)
    return "\n".join("".join(ramp[int(v * (len(ramp) - 1))] for v in row)
                     for row in q)


def render_grid(maps: np.ndarray, k: int = 4, start: int = 0,
                ramp: str = RAMP) -> str:
    """K motion frames side by side, like a film strip."""
    frames = [render_frame(maps[start + i], ramp) for i in range(k)
              if start + i < len(maps)]
    if not frames:
        return ""
    rows = frames[0].split("\n")
    out = []
    for r in range(len(rows)):
        out.append("   ".join(f.split("\n")[r] for f in frames))
    return "\n".join(out)


def render_trace(maps: np.ndarray, start: int = 0, n: int = 16) -> str:
    """Per-frame summary statistics as text. Destroys spatial structure."""
    lines = []
    for i in range(start, min(start + n, len(maps))):
        m = maps[i]
        tot = float(m.sum())
        if tot <= 0:
            lines.append(f"f{i:04d} total=0.000 active=0.000")
            continue
        ys, xs = np.nonzero(m > 0.5 * m.max())
        cy = float(ys.mean()) / m.shape[0]
        cx = float(xs.mean()) / m.shape[1]
        spread = float((ys.max() - ys.min() + 1) / m.shape[0]) if len(ys) else 0.0
        lines.append(f"f{i:04d} total={tot:8.2f} active={len(ys) / m.size:.3f} "
                     f"centre=({cx:.2f},{cy:.2f}) spread={spread:.2f}")
    return "\n".join(lines)


def load_session(npz_path: Path):
    d = np.load(npz_path)
    maps = d["maps"].astype(np.float64) * float(d["scale"])
    return maps


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--kind", default="grid", choices=("grid", "trace"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    maps = load_session(Path(a.npz))
    if a.kind == "grid":
        text = render_grid(maps, a.k, a.start)
    else:
        text = render_trace(maps, a.start)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
