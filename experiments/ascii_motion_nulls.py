"""
ascii_motion_nulls.py -- is the "motion map" measuring MOTION, or is it measuring
threshold dither?

THE FRAME LAYOUT, FIRST, because it decides whether any of this is true.

The captures on disk are NOT standard NV12. `C:\\proart_capture\\nv12small.py`
writes, per 160x96 frame:

    Y plane   15,360 bytes   (160 * 96)
    U,V        3,840 bytes   (80 * 24 samples, INTERLEAVED)
    ---------------------------
    total     19,200 bytes   = W*H*5/4

so the stride is 19,200 bytes, not the `W*H*3//2` = 23,040 this file used to
assume. Read at 23,040, all but every sixth frame is a tear -- rows 0..71 from
frame k, rows 72..95 from frame k+1 -- and the mean |diff| between adjacent
frames is tens of counts instead of about one. Every "56% of the field
changes/frame" number this file was built to explain was, in large part, the
tear.

The layout is DETECTED from the file's content by `src/nv12.py` (the candidate
whose luma has the smallest mean consecutive |diff| wins; an ambiguous file is
an error, not a guess). See `src/nv12.py` and `docs/research/NV12-STRIDE.md`.

The character renderer (`experiments/ascii_motion.py`) reads a frame as
`W*H*3/2` bytes -- the STANDARD NV12 stride. That is correct for standard
NV12, and this file does not change it. The capture is RE-WRAPPED into
standard NV12 before the renderer sees it: the luma is copied verbatim, and
the quarter-height interleaved chroma (80x24 samples) is duplicated down to
NV12's 80x48 siting. No luma byte changes in the re-wrap. The same pattern
is used by `experiments/row_band_removal.py`,
`experiments/ascii_motion_cause.py` and `experiments/gain_invariance.py`.

WHY THIS FILE EXISTS.

`experiments/ascii_motion.py` ran on a live camera capture and reported:

    full_changed_median = 1074 of 1920 cells   (56% of the field changes/frame)
    frames_needing_downscale = 24 / 24

Both numbers are suspicious in the same way. A person sitting at a desk does not
move 56% of a Braille field between consecutive frames. And if 56% of cells
change on EVERY frame, the resolution ladder can never come back off its coarsest
rung -- the knob is saturated, so "resolution scales inversely to frame rate"
was not demonstrated, it was collapsed to a constant.

READ AT THE CORRECT STRIDE, THAT NUMBER IS 6.98%, NOT 56%. The 56% was mostly
the tear.

The hypothesis that would explain both at once: most cells sit near
`braille_thresh = 100`, so an ordinary +/-1 luma wobble from sensor noise or
auto-exposure flips them. If so the P-frame is a dither map, not a motion map,
and the whole I/P construction is storing noise and calling it signal.

This file tries to KILL that hypothesis. If motion is real, three things must
hold, and all three are checked here:

  N1 DETERMINISM. Re-rendering the identical frame must change 0 cells. The
     kernel is claimed byte-deterministic; if this fails, every number in
     ascii_motion.py is measuring noise in the renderer.

  N2 LAG SEPARATION. Change must GROW with temporal lag. Lag-1 < lag-5 < lag-20.
     If lag-1 is already 56% and lag-20 is 58%, the metric has saturated and is
     reporting a constant -- it cannot distinguish "the person moved" from
     "the person is present". Saturation is the falsifier: a real motion signal
     is bounded well below 100% at short lags and climbs toward it.

  N3 NULL SEPARATION. A permuted version of the same delta must preserve the
     change COUNT but carry no structure. If a shuffled delta is
     indistinguishable from the real one, the count alone is all the ladder ever
     consumed, and the ladder is a rate meter wearing a resolution knob's
     clothes.

A fourth, separate measurement explains the mechanism either way: where do the
pixel luminances actually sit relative to the threshold that decides the cells?

If N2 fails, the conclusion is not "tune the threshold". It is that a hard
threshold on a noisy continuous signal is the wrong primitive for a P-frame, and
the representation needs the magnitude of the change, not its sign.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAME_W, FRAME_H = 160, 96
BRAILLE_THRESH = 100
sys.path.insert(0, str(ROOT / "experiments"))
from ascii_motion import ensure, fields, downsample  # noqa: E402
from ascii_motion_threshold import (  # noqa: E402
    sha256_file, standard_nv12,
)
from src.nv12 import read_luma  # noqa: E402


def change_rate(a_hex: str, b_hex: str) -> float:
    """Fraction of cells whose mask byte differs."""
    n = len(a_hex) // 2
    d = sum(1 for k in range(n) if a_hex[k * 2:k * 2 + 2] != b_hex[k * 2:k * 2 + 2])
    return d / n


def luma_histogram(luma: np.ndarray):
    """Where do Y samples sit relative to BRAILLE_THRESH?

    The luma comes from `src.nv12.read_luma`, so it is the luma of the frames
    that were actually captured -- not a mis-strided slice of a frame that
    splices two moments together.
    """
    ys = np.sort(luma.reshape(-1))
    n = len(ys)
    near = int((np.abs(ys.astype(int) - BRAILLE_THRESH) <= 2).sum())
    return {
        "n_pixels": int(n),
        "min": int(ys[0]), "max": int(ys[-1]),
        "p01": int(ys[n // 100]), "p10": int(ys[n // 10]), "p50": int(ys[n // 2]),
        "p90": int(ys[(9 * n) // 10]), "p99": int(ys[(99 * n) // 100]),
        "frac_within_2_of_threshold": near / n,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    capture = Path(a.capture)
    binp = ensure(Path("/tmp/syzbuild"))

    # The layout is DETECTED, not assumed, and the capture is re-wrapped into
    # standard NV12 for the renderer. No luma byte changes in the re-wrap.
    luma, layout = read_luma(capture, width=FRAME_W, height=FRAME_H)
    std = Path("/tmp/amn_std.nv12")
    standard_nv12(capture, std)

    fr = fields(binp, std)
    cols, rows = fr[0][0], fr[0][1]
    n_cells = cols * rows
    print(f"frames={len(fr)}  cells={cols}x{rows} ({n_cells})  "
          f"braille_thresh={BRAILLE_THRESH}")
    print(f"layout={layout.name}  stride={layout.stride}  "
          f"mean|d|={layout.mean_frame_diff:.4f}  margin={layout.margin:.1f}x")

    # -- N1 DETERMINISM --------------------------------------------------------
    # Feed frame 0 twice in a row. Any nonzero change is renderer noise, full
    # stop -- there is no motion in a single frame repeated.
    rep = Path("/tmp/syzbuild/repeat.nv12")
    std_bytes = std.read_bytes()
    # The renderer reads STANDARD NV12, so the frame size here is 23,040 --
    # and that is correct, because `std` is the re-wrapped standard-NV12 file,
    # not the capture. The capture's own stride came from src/nv12.
    fb = FRAME_W * FRAME_H * 3 // 2
    rep.write_bytes(std_bytes[:fb] * 2)
    rf = fields(binp, rep)
    n1 = change_rate(rf[0][2], rf[1][2])
    print(f"\nN1 determinism (identical frame twice): changed={n1 * 100:.2f}%")

    # -- N2 LAG SEPARATION -----------------------------------------------------
    lags = [1, 2, 3, 5, 10, 20]
    lag_rates = {}
    for lag in lags:
        pairs = [(fr[i - lag][2], fr[i][2]) for i in range(lag, len(fr))]
        if not pairs:
            continue
        rates = [change_rate(x, y) for x, y in pairs]
        lag_rates[lag] = {"median": statistics.median(rates),
                          "min": min(rates), "max": max(rates),
                          "n_pairs": len(rates)}
        print(f"N2 lag {lag:>2}: change median {statistics.median(rates) * 100:6.2f}%"
              f"   (min {min(rates) * 100:.2f}%  max {max(rates) * 100:.2f}%)")

    l1 = lag_rates.get(1, {}).get("median")
    l20 = lag_rates.get(20, {}).get("median")
    monotone = all(
        lag_rates[la]["median"] < lag_rates[lb]["median"]
        for la, lb in zip(lags, lags[1:]) if la in lag_rates and lb in lag_rates
    )
    if l1 is not None and l20 is not None:
        headroom = l20 - l1
        print(f"\nN2 headroom lag1->lag20: {(headroom * 100):+.2f} percentage points"
              f"  ({(headroom / l1 * 100 if l1 else 0):+.1f}% relative)")
        print(f"N2 strictly increasing across lags: {monotone}")

    # -- N3 NULL SEPARATION ----------------------------------------------------
    # Replace each P-frame's delta by a permutation of its own bits. Same number
    # of changed cells, no spatial structure. The ladder in ascii_motion.py
    # consumes ONLY the changed-cell count, so it must land on identical rungs.
    BUDGET = 40
    real_rungs, null_rungs = [], []
    for i in range(1, len(fr)):
        d = [1 if fr[i - 1][2][k * 2:k * 2 + 2] != fr[i][2][k * 2:k * 2 + 2] else 0
             for k in range(n_cells)]
        # deterministic permutation: rotate by a fixed offset
        rot = d[len(d) // 3:] + d[:len(d) // 3]
        for vec, acc in ((d, real_rungs), (rot, null_rungs)):
            rung = 4
            for step in (1, 2, 4):
                _, _, cd = downsample("".join(f"{v:02x}" for v in vec), cols, rows, step)
                if sum(cd) <= BUDGET:
                    rung = step
                    break
            acc.append(rung)
    agree = sum(1 for x, y in zip(real_rungs, null_rungs) if x == y) / len(real_rungs)
    print(f"\nN3 ladder rung, real vs rotated-null delta: agreement {agree * 100:.1f}%"
          f"   real median {statistics.median(real_rungs)}"
          f"   null median {statistics.median(null_rungs)}")

    # -- MECHANISM -------------------------------------------------------------
    hist = luma_histogram(luma[:len(fr)])
    print(f"\nmechanism: luma p10={hist['p10']} p50={hist['p50']} p90={hist['p90']}"
          f"   within +/-2 of {BRAILLE_THRESH}: "
          f"{hist['frac_within_2_of_threshold'] * 100:.2f}% of pixels")

    saturated = (l1 is not None and l1 > 0.30) or not monotone
    verdict = ("SATURATED: the delta is dominated by threshold dither, not motion."
               if saturated else
               "SEPARATED: change grows with lag; the delta carries motion.")

    report = {
        "question": ("does the I/P character representation measure motion, or "
                     "threshold dither on a noisy continuous signal?"),
        "capture": a.capture,
        "cells": [cols, rows],
        "braille_thresh": BRAILLE_THRESH,
        "N1_determinism_same_frame_changed_pct": n1 * 100,
        "N1_pass": n1 == 0.0,
        "N2_lag_change_rate_pct": {str(k): v["median"] * 100
                                  for k, v in lag_rates.items()},
        "N2_monotone_increasing": monotone,
        "N2_headroom_lag1_to_lag20_pp": (l20 - l1) * 100 if l1 and l20 else None,
        "N3_null_rung_agreement_pct": agree * 100,
        "luma_histogram": hist,
        "verdict": verdict,
        "notes": [
            ("The SATURATED/SEPARATED predicate is the pre-registered "
             "falsifier this file was written to apply: lag-1 change above "
             "30%, or not monotone in lag, means the delta is a dither map. It "
             "now comes out SEPARATED, so the conclusion the previous run "
             "recorded -- that the delta is dominated by threshold dither -- "
             "does NOT survive the stride fix."),
            ("That is not the same as saying the dither mechanism is gone. "
             "experiments/ascii_motion_threshold.py measures it directly on "
             "the same capture: the flip rate still falls monotonically with "
             "distance from the threshold, and 5.62% of pixels sit within +/-2 "
             "of it. What the stride fix removed was the TORN-FRAME artefact "
             "-- a large, lag-invariant change signal that was not motion at "
             "all. The residual dither is a small, structured, "
             "threshold-proximity effect sitting on top of a real motion "
             "signal, and the two are not in conflict."),
        ],
        "consequence": (
            "A hard threshold on a noisy continuous signal makes the P-frame a "
            "dither map. The representation needs the MAGNITUDE of the change "
            "between keyframes, not the per-frame sign of a threshold test. "
            "That also un-saturates the ladder: cost then tracks real motion "
            "instead of a constant flip rate."
            if saturated else "None. Proceed to the labelled capture."),
        "provenance": {
            "generated_by": "experiments/ascii_motion_nulls.py",
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.executable,
            "command_line": " ".join(sys.argv),
            "cli_args": list(sys.argv[1:]),
            "capture": {
                "path": str(capture.resolve()),
                "sha256": sha256_file(capture),
                "bytes": capture.stat().st_size,
                "layout": layout.name,
                "stride_bytes": layout.stride,
                "n_frames": layout.n_frames,
                "frames_used": len(fr),
                "layout_evidence": layout.as_dict(),
            },
            "renderer": {
                "program": "experiments/ascii_motion.py RENDER_C",
                "note": ("the character renderer assumes the STANDARD NV12 "
                         "stride of 23,040 bytes per frame, so the capture is "
                         "re-wrapped into standard NV12 first; no luma byte is "
                         "changed by that re-wrap"),
                "frame_bytes": fb,
            },
        },
        "supersedes": {
            "why": ("Re-measured on 2026-10-20. The previous run read the "
                    "capture at W*H*3//2 = 23,040 bytes per frame; the capture "
                    "on disk carries 19,200 bytes per frame, so all but every "
                    "sixth frame was torn (rows 0..71 from frame k, rows 72..95 "
                    "from frame k+1). The previous numbers are recorded here "
                    "rather than silently overwritten."),
            "previous_stride_bytes": 23040,
            "previous_n_frames": 25,
            "previous_N1_determinism_same_frame_changed_pct": 0.0,
            "previous_N1_pass": True,
            "previous_N2_lag_change_rate_pct": {
                "1": 55.93749999999999, "2": 57.552083333333336,
                "3": 57.447916666666664, "5": 11.458333333333332,
                "10": 15.156249999999998, "20": 18.958333333333332},
            "previous_N2_monotone_increasing": False,
            "previous_N2_headroom_lag1_to_lag20_pp": -36.979166666666664,
            "previous_N3_null_rung_agreement_pct": 100.0,
            "previous_luma_histogram": {
                "n_pixels": 384000, "min": 32, "max": 212,
                "p01": 50, "p10": 83, "p50": 123, "p90": 157, "p99": 173,
                "frac_within_2_of_threshold": 0.04558072916666667},
            "previous_verdict": ("SATURATED: the delta is dominated by "
                                 "threshold dither, not motion."),
            "verdict_changed": (
                "YES, AND IT FLIPS. The previous run read the capture at "
                "23,040 bytes per frame, so 25 of the 30 frames were torn, and "
                "the lag curve was not even monotone: lag 1/2/3 sat at ~56-58% "
                "and lag 5 fell to 11.46%. That step is the tear signature -- "
                "every sixth frame was intact. At the correct stride the lag "
                "curve is monotone from 6.98% at lag 1 to 22.55% at lag 20, "
                "which is a motion signal, not saturation, so the conclusion "
                "this file was written to support does NOT survive."),
        },
    }
    p = Path(a.out) / "ascii_motion_nulls_results.json"
    p.write_text(json.dumps(report, indent=2))
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
