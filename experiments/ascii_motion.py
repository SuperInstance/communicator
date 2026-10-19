"""
ascii_motion.py -- make MOTION first-class in a character array, and compress
it the way video compresses space.

!! READ docs/research/ASCII-MOTION.md FIRST !!

The numbers this script prints on a live webcam capture are real but they DO NOT
mean what they look like they mean. This file reports ~56% of cells changing per
frame on a stationary subject, and a resolution ladder pinned at its coarsest
rung on 24 of 25 frames. Both were initially read as "motion". They are not.

Measured afterwards, at dot (per-pixel) resolution:

  * dots sitting exactly on `braille_thresh` flip at 67.25%; dots more than 41
    counts away flip at 21.61% -- a monotone 3.1x gradient, so roughly half the
    delta below is threshold dither, not motion;
  * the 21.61% far-from-threshold floor is BELOW the 34.62% independence
    baseline, so that residual is structured and real motion IS present;
  * a rotated null delta selects the identical ladder rung on 100.0% of frames,
    so this ladder consumes only a change COUNT and is a rate meter, not a
    resolution knob.

The ladder below is therefore left as written, unmodified, because it is the
thing those findings are measurements OF. The fix is hysteresis in Syzygy's
threshold and a rung decision that consumes magnitude and position rather than a
count. Neither is a change to this file.

THE PROBLEM THIS SOLVES, AS MEASURED.

`docs/research/TWO-SPEAKER-TURNS.md`: at the moment of decision, END and RESUME
are byte-identical audio. The information is not in the frame. Silence duration
gives AUC 0.529; pitch cannot resolve individuals within a voice class. Every
route that reads a single instant failed, and they failed for the same reason.

So a representation that makes motion first-class is not a refinement here. It
is the only place the information can physically be.

THE REPRESENTATION.

Syzygy turns a camera frame into an array of characters -- Braille, edge glyphs,
a tone ramp -- in one integer-only pass, 0.1 ms for 160x96, byte-deterministic,
no model. Per-frame arrays are then treated the way video codecs treat frames:

    I-frame  the whole glyph field, emitted when the delta is large enough that
             a delta would cost more than a keyframe
    P-frame  the CHANGE, not the picture

Two consequences fall out of "the change, not the picture":

1. MOTION IS FIRST-CLASS. A P-frame is literally a motion map. Nothing has to be
   inferred from a pair of snapshots; the pipeline carries the difference
   explicitly.

2. RESOLUTION SCALES INVERSELY TO FRAME RATE. A delta's cost is proportional to
   how many cells changed. When much moved, spatial resolution is dropped and
   temporal resolution kept; when little moved, full spatial resolution is kept.
   Bandwidth stays roughly constant while the spatial detail degrades exactly
   when the picture is changing too fast for detail to be readable anyway --
   which is the same trade every video codec makes, applied to characters.

AND THE OUTPUT IS STILL TEXT, so a text-only model can consume it. That matters
more than it looks: `docs/research/SPECTROGRAM-TURN.md` found the only
image-capable model available locally answers a constant 'different' to every
spectrogram and to a blank rectangle. An array of characters does not need an
image model to read.

WHAT IS AND IS NOT PROVEN HERE.

This file MEASURES the representation: it computes motion maps, keyframe
decisions and the resolution ladder, and reports the numbers. It does NOT claim
that any of this separates "talking to me" from "talking to someone else" --
that needs labelled data from the three-condition protocol and is a separate
step. A demonstration that motion is visible is not a demonstration that motion
is sufficient.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAME_W, FRAME_H = 160, 96
# Resolution ladder. Each step halves the cell count in both axes, so bytes
# fall by 4x per step -- the knob the ladder trades.
LADDER = [(80, 24), (40, 12), (20, 6)]

RENDER_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include "syz_fused.h"
/* prints: F <cols> <rows> then <hex masks> then <hex tones> per frame */
int main(int argc, char **argv) {
    uint32_t W = (uint32_t)atoi(argv[1]), H = (uint32_t)atoi(argv[2]);
    FILE *f = fopen(argv[3], "rb"); if (!f) { perror("open"); return 1; }
    size_t fb = (size_t)W * H * 3 / 2;
    static uint8_t buf[64 << 20];
    size_t got = fread(buf, 1, fb * 2000, f); fclose(f);
    size_t nf = got / fb;
    static uint8_t pool[1 << 20], mask[4096], tone[4096];
    static uint32_t glyph[4096];
    for (size_t i = 0; i < nf; i++) {
        SyzArena a; syz_arena_init(&a, pool, sizeof pool);
        SyzNv12 nv = { .y = buf + i * fb, .uv = buf + i * fb + (size_t)W * H,
                       .w = W, .h = H, .y_stride = W, .uv_stride = W };
        SyzFusedParams p = { .braille_thresh = 100, .edge_thresh2 = 4000, .fft_row = 1 };
        SyzFusedOut o = { .cap = 4096, .mask = mask, .glyph = glyph, .tone = tone };
        if (syz_fused(&a, &nv, &p, &o)) { fprintf(stderr, "rc\n"); return 1; }
        printf("F %u %u\n", o.cols, o.rows);
        for (uint32_t k = 0; k < o.cols * o.rows; k++) printf("%02x", mask[k]);
        printf("\n");
    }
    return 0;
}
"""


def ensure(build_dir: Path) -> Path:
    d = ROOT / ".scratch" / "Syzygy"
    if not d.is_dir():
        sys.exit("run experiments/syzygy_camera.py once to fetch Syzygy")
    build_dir.mkdir(parents=True, exist_ok=True)
    binp = build_dir / "syz_stream"
    src = build_dir / "syz_stream.c"
    if not binp.is_file() or src.read_text() != RENDER_C:
        src.write_text(RENDER_C)
        subprocess.run(["cc", "-O2", f"-I{d/'include'}", str(src), "-o", str(binp)],
                       check=True)
    return binp


def fields(binp: Path, nv12: Path):
    r = subprocess.run([str(binp), str(FRAME_W), str(FRAME_H), str(nv12)],
                       capture_output=True, text=True, check=True)
    lines = r.stdout.splitlines()
    out, i = [], 0
    while i < len(lines):
        if not lines[i].startswith("F "):
            i += 1
            continue
        _, cs, rs = lines[i].split()
        out.append((int(cs), int(rs), lines[i + 1]))
        i += 2
    return out


def downsample(mask_hex: str, cols: int, rows: int, step: int):
    """Coarsen by `step`; a coarse cell is 1 if ANY fine cell in it is set.

    OR-downsampling, not averaging: the question a coarse cell answers is
    "did anything change here", and averaging would let a large faint change
    disappear -- exactly the motion we are trying to keep.
    """
    m = [int(mask_hex[k:k + 2], 16) for k in range(0, len(mask_hex), 2)]
    cw, ch = cols // step, rows // step
    out = []
    for r in range(ch):
        for c in range(cw):
            v = 0
            for dr in range(step):
                for dc in range(step):
                    idx = (r * step + dr) * cols + (c * step + dc)
                    if idx < len(m) and m[idx]:
                        v = 1
            out.append(v)
    return cw, ch, out


def bits_to_text(v):
    return "".join("#" if v else "." for v in v)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="NV12 written on Windows")
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    binp = ensure(Path("/tmp/syzbuild"))
    fr = fields(binp, Path(a.capture))
    print(f"frames={len(fr)}  cells={fr[0][0]}x{fr[0][1]}")

    # -- motion map per frame pair, at full resolution -------------------------
    motion = []
    for i in range(1, len(fr)):
        cols, rows, prev = fr[i - 1]
        _, _, cur = fr[i]
        d = [1 if prev[k * 2:k * 2 + 2] != cur[k * 2:k * 2 + 2] else 0
             for k in range(cols * rows)]
        motion.append(d)

    # -- the resolution ladder ------------------------------------------------
    # A delta costs bytes proportional to changed cells. Pick the coarsest
    # resolution whose delta still fits the budget, so bandwidth stays bounded
    # while spatial detail degrades only when motion is high.
    BUDGET = 40          # changed coarse cells per frame at full res
    rows_out = []
    for i, d in enumerate(motion, start=1):
        chosen, cw, ch, cd = LADDER[0][0], *downsample(
            "".join(f"{v:02x}" for v in d), fr[i][0], fr[i][1], 1)
        full_changed = sum(d)
        for (lw, lh), step in zip(LADDER, (1, 2, 4)):
            if lw // step == 0 or lh // step == 0:
                continue
            cw, ch, cd = downsample("".join(f"{v:02x}" for v in d),
                                   fr[i][0], fr[i][1], step)
            if sum(cd) <= BUDGET or step == 4:
                chosen = step
                break
        rows_out.append({"frame": i, "full_changed": full_changed,
                         "ladder_step": chosen,
                         "coarse_changed": sum(cd)})

    full = [r["full_changed"] for r in rows_out]
    report = {
        "note": ("Motion made first-class in a character array. full_changed "
                 "is how many of the 80x24 cells differ from the previous "
                 "frame; ladder_step is which rung of the resolution ladder "
                 "that frame was coded at under a fixed changed-cell budget. "
                 "Measures the representation only -- no claim is made here "
                 "that it separates any particular class."),
        "cells": fr[0][:2],
        "n_frames": len(fr),
        "budget_changed_cells": BUDGET,
        "stats": {
            "full_changed_min": min(full) if full else None,
            "full_changed_median": statistics.median(full) if full else None,
            "full_changed_max": max(full) if full else None,
            "frames_needing_downscale": sum(1 for r in rows_out
                                            if r["ladder_step"] > 1),
        },
        "per_frame": rows_out,
    }
    p = Path(a.out) / "ascii_motion_results.json"
    p.write_text(json.dumps(report, indent=2))

    print(json.dumps(report["stats"], indent=2))
    print(f"\nmotion map, frame {len(motion)//2} (cell = changed since previous):")
    mid = motion[len(motion) // 2]
    cols, rows = fr[0][0], fr[0][1]
    for r in range(rows):
        print("   ", bits_to_text(mid[r * cols:(r + 1) * cols]))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()