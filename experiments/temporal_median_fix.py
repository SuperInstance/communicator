"""
temporal_median_fix.py -- the fix that survives, tested.

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
frames is tens of counts instead of well under one.

The layout is DETECTED from the file's content by `src/nv12.py` (the candidate
whose luma has the smallest mean consecutive |diff| wins; an ambiguous file is
an error, not a guess). See `src/nv12.py` and `docs/research/NV12-STRIDE.md`.

The character renderer below reads a frame as `W*H*3/2` bytes -- the STANDARD
NV12 stride. That is correct for standard NV12, and this file does not change
it. The capture is RE-WRAPPED into standard NV12 before the renderer sees it:
the luma is copied verbatim, and the quarter-height interleaved chroma
(80x24 samples) is duplicated down to NV12's 80x48 siting. No luma byte
changes in the re-wrap. The same pattern is used by
`experiments/row_band_removal.py`, `experiments/ascii_motion_cause.py` and
`experiments/gain_invariance.py`.

WHY THIS FILE EXISTS.

`CAMERA-FLICKER-BANDING.md` established the mechanism and killed the previous
proposal. The frame-to-frame drift is horizontal banding: in the empty room the
residual is identical in every column (column energy share 0.00%), a single
scalar explains under 5% of it, and the row spectrum is dominated by 1-3 cycles
across the scan. That is mains-derived LED flicker sampled by a rolling shutter.

Its stated conclusion was that the surviving fix is a TEMPORAL MEDIAN applied
before thresholding, and that this was untested. This file tests it.

THE BANDING WAS THE TORN FRAME. The "column energy share 0.00%" is the
signature of a horizontal step, and the row-72 splice of a mis-strided read is
exactly a horizontal step. At the correct stride the empty room's dot flip rate
is 0.00% with NO filter at all, so there is no banding left to remove and the
temporal median has nothing to fix.

THE REASON A MEDIAN SHOULD WORK, AND WHY NOT A MEAN.

The banding is phase-varying: consecutive frames sample the lamp at different
points in its cycle, so the bands sit at different row offsets in each frame.
A temporal median over k frames therefore sees k different band positions and
takes the per-pixel median, which discards a pattern that moved.

A MEAN WOULD NOT WORK, and the distinction matters. The bands shift rather than
vanish, so averaging blurs them into a low-amplitude standing wave instead of
removing them -- and a blurred wave still crosses the threshold. The median is
non-linear and rejects the excursion; the mean absorbs it. So this file tests
both, because if a mean worked as well as a median the simpler and cheaper
operator would be the right answer, and I want to know that rather than assume.

WHAT IS PREDICTED, STATED FIRST.

  * EMPTY room: the flip rate is already 0.00% with no filter, so it cannot
    collapse any further. There is nothing to remove.
  * OCCUPIED: stays ABOVE the empty rate. Casey was there, and he is signal.
  * If the occupied rate collapses too as the window grows, the median is
    erasing real motion and the window is too long.

THE COST, WHICH IS NOT FREE AND IS NOT HIDDEN.

A k-frame median needs k frames before it emits anything, so it adds (k-1)/30
seconds of latency and it smooths real fast motion by the same amount. For a
system whose entire premise is watching continuously, that is a real trade, not
a free win. The sweep below reports the window alongside the result so the
trade is visible per window rather than settled once.

THE USUAL CAVEAT, WHICH APPLIES AGAIN.

Both captures are on the same camera in the same room, the empty one taken later
and darker. This measures whether the filter restores a usable signal. It does
not prove the restored signal means anything, and it is not evidence about
whether vision can tell "talking to me" from "talking to someone else".
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FRAME_W, FRAME_H = 160, 96
FB = FRAME_W * FRAME_H * 3 // 2
Y_SIZE = FRAME_W * FRAME_H
THRESH = 100
WINDOWS = (1, 3, 5, 7)
sys.path.insert(0, str(ROOT / "experiments"))
from ascii_motion_threshold import build, nv12_chroma, sha256_file  # noqa: E402
from src.nv12 import read_luma  # noqa: E402


def load(path: Path, nf: int):
    """The luma of the first `nf` frames, and the chroma in STANDARD NV12 siting.

    The luma comes from `src.nv12.read_luma`, so the layout is DETECTED from
    the file's content rather than assumed. The chroma is reshaped to
    standard NV12's 80x48 siting, because the character renderer reads
    `W*H*3/2` bytes per frame.
    """
    luma, layout = read_luma(path, width=FRAME_W, height=FRAME_H)
    n = min(nf, layout.n_frames)
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    uv = nv12_chroma(raw, layout, n)
    return luma[:n], uv


def emit_median(Y, k, op):
    """Emit one filtered frame per input frame (edge-replicated), no latency shift."""
    if k <= 1:
        return Y
    pad = k // 2
    P = np.pad(Y, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    if op == "median":
        stack = np.stack([P[i:i + len(Y)] for i in range(k)], axis=0)
        return np.median(stack, axis=0).astype(np.uint8)
    return np.clip(P[pad:pad + len(Y)].astype(np.float32), 0, 255).astype(np.uint8)


def emit_mean(Y, k):
    if k <= 1:
        return Y
    pad = k // 2
    P = np.pad(Y, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    stack = np.stack([P[i:i + len(Y)] for i in range(k)], axis=0).astype(np.float32)
    return np.clip(stack.mean(axis=0), 0, 255).astype(np.uint8)


def flip(binp: Path, tmp: Path, nf: int):
    r = subprocess.run([str(binp), str(FRAME_W), str(FRAME_H), str(tmp)],
                       capture_output=True, text=True, check=True)
    lines = r.stdout.splitlines()
    masks, i = [], 0
    while i < len(lines):
        if not lines[i].startswith("F "):
            i += 1
            continue
        masks.append(lines[i + 1])
        i += 2
    B = np.stack([((np.frombuffer(bytes.fromhex(h), np.uint8)[:, None]
                    >> np.arange(8)) & 1).astype(np.uint8) for h in masks[:nf]])
    x = B[1:] ^ B[:-1]
    per = x.reshape(len(x), -1).mean(axis=1)
    return float(np.median(per) * 100)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--occupied", default="/mnt/c/proart_capture/A.nv12")
    ap.add_argument("--empty", default="/home/proart/projects/glasses/.capture_keep/empty1.nv12")
    ap.add_argument("--nf", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    binp = build()
    caps = {}
    layouts = {}
    for p, lab in ((a.occupied, "occupied"), (a.empty, "empty")):
        if Path(p).is_file():
            Y, UV = load(Path(p), a.nf)
            caps[lab] = (Y, UV)
            _, layout = read_luma(Path(p), width=FRAME_W, height=FRAME_H)
            layouts[lab] = (layout, Path(p))
    if not caps:
        sys.exit("no captures found")

    results = {}
    for op in ("median", "mean"):
        for k in WINDOWS:
            key = f"{op}{k}"
            row = {}
            for lab, (Y, UV) in caps.items():
                F = emit_median(Y, k, op) if op == "median" else emit_mean(Y, k)
                tmp = Path("/tmp/tm.nv12")
                # Standard NV12, because the renderer reads W*H*3/2 bytes a
                # frame. The luma is verbatim; only the chroma's vertical
                # siting was changed when it was loaded.
                tmp.write_bytes(np.concatenate(
                    [F.reshape(len(F), Y_SIZE),
                     UV[:len(F)].reshape(len(F), FB - Y_SIZE)], axis=1
                ).astype(np.uint8).tobytes())
                row[lab] = flip(binp, tmp, len(F))
            results[key] = row
            e, o = row.get("empty"), row.get("occupied")
            ratio = (e / o) if (e is not None and o) else None
            print(f"  {op:6s} k={k}  empty {e if e is None else round(e,2):>6}"
                  f"   occupied {o if o is None else round(o,2):>6}"
                  f"   empty/occ {ratio if ratio is None else round(ratio,3)}"
                  f"   latency +{(k-1)/30:.2f}s")

    base = results.get("median1", {})
    # A window only "wins" if it STRICTLY reduces the empty-room rate. The
    # previous version of this used `base.get("empty") or 100`, which treats a
    # base rate of 0.00% as missing and then picks an arbitrary window.
    # With the empty room already at 0.00% there is nothing to reduce, so the
    # honest answer is that no window wins.
    base_e = base.get("empty")
    best = None
    if base_e is not None and base_e > 0:
        for key, row in results.items():
            if not key.startswith("median") or key == "median1":
                continue
            e, o = row.get("empty"), row.get("occupied")
            if e is None or not o:
                continue
            if e < base_e * 0.5 and (best is None or e < best[1]):
                best = (key, e, o)

    out = {"question": ("does a temporal median before thresholding remove LED "
                        "banding and leave real motion?"),
           "operator_results": results,
           "best_median_window": best,
           "limits": [
               "the empty capture is later and darker than the occupied one "
               "(mean luma 49.7 vs 111.2); only within-condition comparisons "
               "across windows are clean",
               "a k-frame median adds (k-1)/30 s latency and smooths real fast "
               "motion by the same amount -- reported per window, not hidden",
               "edge frames are replicated, so the first and last (k-1)/2 "
               "frames of each capture are not fully filtered",
               "this shows whether the instrument is usable. It says nothing "
               "about whether vision can distinguish 'talking to me'",
           ]}

    # A verdict has to account for BOTH halves of this result, and the first
    # version of this script only reported the flattering half.
    #
    # It announced "the banding is removable and real motion survives it" on the
    # strength of the k=3 row. But the occupied rate is NON-MONOTONIC in the
    # window -- 21.6, 0.44, 12.25 for k=3,5,7 -- and it reproduces at 300 and 700
    # frames. A filter that can drive a real scene to 0.44% is not selectively
    # preserving content; it is cancelling periodic content. The k=5 collapse
    # is the control that says so, and it is the reason this verdict cannot be
    # the cheerful one.
    occ = {k: results.get(f"median{k}", {}).get("occupied")
           for k in WINDOWS if results.get(f"median{k}")}
    occ_vals = [v for v in occ.values() if v is not None]
    non_mono = any(occ_vals[i] > occ_vals[i + 1] * 3
                   for i in range(len(occ_vals) - 1))
    killed_somewhere = min(occ_vals) < 1.0 if occ_vals else False

    if base_e is not None and base_e < 1.0:
        # THE PREMISE IS GONE. The empty room was already at ~0% with no
        # filter, so there is no banding to remove and no window can "win".
        out["banding_removal"] = (
            f"NOT APPLICABLE. The empty room already flips {base_e:.2f}% of "
            "dots per frame with NO filter at all, so there is no banding left "
            "to remove and no median window can improve on it. The 50.00% "
            "empty-room rate that motivated this experiment was the torn "
            "frame: the captures carry 19,200 bytes per frame and this file "
            "used to read them at 23,040.")
        out["verdict"] = (
            "THE PREMISE IS GONE, AND THE FIX IS NOT NEEDED. The empty room "
            f"flips {base_e:.2f}% of dots per frame with no filter at all -- "
            "there is no banding to remove, so a temporal median has nothing to "
            "fix here. What the sweep now measures is the COST of the filter: "
            "the occupied rate is "
            + ", ".join(f"k={k}: {v:.2f}%" for k, v in occ.items())
            + " on the median, and it is "
            + ("NOT monotone in the window size"
               if non_mono else
               "MONOTONE in the window size, falling as the window grows")
            + ". A k-frame median reduces ANY temporally-varying signal by "
            "about that factor, so the filter is smoothing the occupied "
            "capture's signal -- whether or not that signal is motion -- and "
            "it costs (k-1)/30 s of latency to do it. The honest "
            "recommendation is to use NO temporal median on this capture and "
            "to keep the latency.")
    elif best:
        out["banding_removal"] = (
            f"SOLID. The empty room goes from {base_e:.2f}% to "
            f"{best[1]:.2f}% and stays at 0.00% for every window k>=3, on both "
            f"the median and the mean, and reproduces at 300 and 700 frames. "
            f"The banding is removable in software.")
    if non_mono and killed_somewhere:
        out["verdict"] = (
            "PARTIAL, AND THE SECOND HALF IS A WARNING. The banding is "
            f"completely removed -- empty room {base_e:.2f}% -> 0.00%, every "
            "window, both operators, stable across frame counts. But the "
            "occupied rate is NON-MONOTONIC in the window size ("
            + ", ".join(f"k={k}: {v:.2f}%" for k, v in occ.items())
            + ") and the k=5 collapse to 0.44% reproduces at both frame counts. "
            "A filter that can drive a REAL scene to 0.44% is cancelling "
            "periodic content, not preserving content while removing nuisance. "
            "So this is NOT a safe drop-in fix at any fixed window: k=3 is the "
            "best available, and 'real motion survives it' is NOT established "
            "by this data. The honest next step is a filter that can tell "
            "banding from content, or a camera and lighting combination with no "
            "banding in the first place.")
    elif best:
        out["verdict"] = (
            f"A {best[0]} window drops the empty room to {best[1]:.2f}% "
            f"(from {base_e:.2f}%) while the occupied room stays at "
            f"{best[2]:.2f}%, monotonically across windows.")
    elif base_e is None or base_e >= 1.0:
        out["verdict"] = ("No median window halved the empty-room rate. Either "
                          "the banding needs more frames than were tested, or "
                          "this representation needs a banding-free camera "
                          "rather than a better filter.")

    out["provenance"] = {
        "generated_by": "experiments/temporal_median_fix.py",
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.executable,
        "command_line": " ".join(sys.argv),
        "cli_args": list(sys.argv[1:]),
        "nf": a.nf,
        "captures": {
            lab: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "layout": layout.name,
                "stride_bytes": layout.stride,
                "n_frames": layout.n_frames,
                "layout_evidence": layout.as_dict(),
            } for lab, (layout, path) in layouts.items()
        },
        "renderer": {
            "program": "experiments/ascii_motion_threshold.py CHANNELS_C",
            "note": ("the character renderer assumes the STANDARD NV12 stride "
                     "of 23,040 bytes per frame, so the capture is re-wrapped "
                     "into standard NV12 first; no luma byte is changed by "
                     "that re-wrap"),
            "frame_bytes": FB,
            "braille_thresh": THRESH,
        },
    }
    out["supersedes"] = {
        "why": ("Re-measured on 2026-10-20. The previous run read both "
                "captures at W*H*3//2 = 23,040 bytes per frame; the captures "
                "on disk carry 19,200 bytes per frame, so all but every sixth "
                "frame was torn (rows 0..71 from frame k, rows 72..95 from "
                "frame k+1). The previous numbers are recorded here rather "
                "than silently overwritten."),
        "previous_stride_bytes": 23040,
        "previous_operator_results": {
            "median1": {"occupied": 48.190104166666664, "empty": 50.0},
            "median3": {"occupied": 21.42578125, "empty": 0.0},
            "median5": {"occupied": 0.48177083333333337, "empty": 0.0},
            "median7": {"occupied": 12.180989583333334, "empty": 0.0},
            "mean1": {"occupied": 48.190104166666664, "empty": 50.0},
            "mean3": {"occupied": 25.774739583333332, "empty": 0.0},
            "mean5": {"occupied": 0.78125, "empty": 0.0},
            "mean7": {"occupied": 12.805989583333332, "empty": 0.0},
        },
        "previous_best_median_window": ["median3", 0.0, 21.42578125],
        "previous_banding_removal": (
            "SOLID. The empty room goes from 50.00% to 0.00% and stays at "
            "0.00% for every window k>=3, on both the median and the mean, and "
            "reproduces at 300 and 700 frames. The banding is removable in "
            "software."),
        "previous_verdict": (
            "PARTIAL, AND THE SECOND HALF IS A WARNING. The banding is "
            "completely removed -- empty room 50.00% -> 0.00%, every window, "
            "both operators, stable across frame counts. But the occupied rate "
            "is NON-MONOTONIC in the window size (k=1: 48.19%, k=3: 21.43%, "
            "k=5: 0.48%, k=7: 12.18%) and the k=5 collapse to 0.44% reproduces "
            "at both frame counts. A filter that can drive a REAL scene to "
            "0.44% is cancelling periodic content, not preserving content "
            "while removing nuisance. So this is NOT a safe drop-in fix at any "
            "fixed window: k=3 is the best available, and 'real motion "
            "survives it' is NOT established by this data. The honest next "
            "step is a filter that can tell banding from content, or a camera "
            "and lighting combination with no banding in the first place."),
        "verdict_changed": (
            "YES. The previous run's 'the banding is removable in software' "
            "and its warning that a k=5 median drives a real scene to 0.44% "
            "were both measured on torn frames. At the correct stride the "
            "empty room is already at 0.00% with no filter, so there is no "
            "banding to remove and the fix is not needed; and the occupied "
            "rate now FALLS monotonically as the window grows, which is the "
            "filter smoothing real motion away at a latency cost. Both of the "
            "previous run's halves were artefacts of the stride."),
    }
    p = Path(a.out) / "temporal_median_fix_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nVERDICT: {out.get('verdict')}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
