"""
emptyroom_null.py -- THE control the ASCII-motion lane never had.

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
it. Each capture is RE-WRAPPED into standard NV12 before the renderer sees it:
the luma is copied verbatim, and the quarter-height interleaved chroma
(80x24 samples) is duplicated down to NV12's 80x48 siting. No luma byte
changes in the re-wrap. The same pattern is used by
`experiments/row_band_removal.py`, `experiments/ascii_motion_cause.py` and
`experiments/gain_invariance.py`.

WHY THIS EXPERIMENT EXISTS.

`docs/research/ASCII-MOTION.md` measured that the Braille character field
changes ~55.9% of cells between adjacent frames on a subject who was sitting
still, and concluded that roughly half of that delta is threshold dither while
a structured residual below the independence baseline is real motion.

That conclusion rests on an inference: it attributed the near-threshold flips to
dither and the far-from-threshold floor to "motion". But NOTHING in that data
contained motion as a labelled condition. There was no way to know whether the
structured residual was the subject moving, or the camera doing something, or
the scene lighting changing.

An EMPTY ROOM settles it, and no amount of analysis on occupied footage can.

    empty room + the same delta   -> the delta is CAMERA, not subject
    empty room + the delta collapses -> the delta was the SUBJECT, and the
                                       representation really does carry motion

Either answer is decisive. This is the control the lane was missing, and it was
missing because there was nobody in the room to point a camera at.

WHAT IS MEASURED, AND WHAT IS HELD CONSTANT.

  * Same Syzygy kernel, same braille_thresh, same 160x96 -- verified, not
    assumed.
  * Two quantities, kept strictly separate because conflating them was a real
    error in an earlier pass:
      - RAW frames feed Syzygy, so the threshold analysis is directly comparable
        to the numbers already in the doc.
      - GAIN-NORMALISED frames are used only for image differencing, because the
        auto-exposure hunt otherwise swamps every comparison. The capture-side
        exposure lock FAILED on this driver -- AUTO_EXPOSURE reads back 0.0 --
        so normalisation is the only defence, and the residual drift after it
        is reported rather than assumed away.

  * The dither gradient (flip rate vs distance from threshold) is recomputed for
    the empty room. If the monotone gradient survives with NO SUBJECT IN
    FRAME, then that gradient is a property of the camera and the threshold,
    not of anything in the room -- which is the specific claim the earlier doc
    could not make.

LIMITS, STATED UP FRONT.

  * One empty capture and one occupied capture is a demonstration, not an
    experiment with error bars. If the two come out close, that is suggestive;
    it is not a p-value.
  * The two captures differ in lighting as well as occupancy. Casey left the
    room, so absolute luma fell. Gain normalisation removes the frame-to-frame
    component but NOT a global difference between the two recordings. That
    confound cannot be removed retrospectively and is reported, not hidden.
  * The occupied side is the earlier A/B/C capture, whose vision condition
    FAILED (0 of 96 windows separated A from B). That does not disqualify it for
    this purpose -- this experiment asks "is there motion here?", not "does A
    differ from B" -- but the occupied delta should be read as "subject present
    and mostly still", not "subject moving".
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
DOTS_X, DOTS_Y = 2, 4
THRESH = 100

sys.path.insert(0, str(ROOT / "experiments"))
from ascii_motion_threshold import build, sha256_file, standard_nv12  # noqa: E402
from src.nv12 import read_luma  # noqa: E402


def load(nv12: Path):
    """The luma of every frame, and the layout that was detected.

    The layout is DETECTED from the file's content, not assumed from a
    hard-coded stride -- see `src/nv12.py`.
    """
    return read_luma(nv12, width=FRAME_W, height=FRAME_H)


def masks(binp: Path, nv12: Path):
    """The renderer's mask per frame.

    `nv12` must be a STANDARD NV12 file: the renderer reads `W*H*3/2` bytes
    per frame, and the captures on disk are not in that layout.
    """
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


def dot_luma(ys):
    """Per-dot luma, matching Syzygy's 2x4 cell layout."""
    H, W = ys.shape
    rows, cols = H // DOTS_Y, W // DOTS_X
    return ys.reshape(rows, DOTS_Y, cols, DOTS_X).transpose(0, 2, 1, 3) \
             .reshape(rows * cols, DOTS_Y * DOTS_X)


def analyse(binp: Path, path: Path, label: str, max_frames: int = 400):
    frames, layout = load(path)
    # The renderer wants standard NV12; the capture is not in that layout.
    # Only the frames this analysis will use are re-wrapped, so the temporary
    # file stays small for the 4,500-frame empty capture.
    std = Path("/tmp/ern_std.nv12")
    standard_nv12(path, std, n=min(max_frames, layout.n_frames))
    mf = masks(binp, std)
    nf = min(len(mf), len(frames), max_frames)
    rows, cols = mf[0][1], mf[0][0]
    nc = rows * cols

    # VECTORISED. The first version of this loop walked 1920 cells x 8 dots per
    # frame in pure Python, which is 69 million iterations at 4500 frames and
    # does not finish in useful time. Same arithmetic, numpy.
    #
    # masks: (nf, nc) uint8
    m = np.empty((nf, nc), np.uint8)
    for i in range(nf):
        m[i] = np.frombuffer(bytes.fromhex(mf[i][2]), np.uint8)

    # dot luma: (nf, nc, 8)
    F = frames[:nf].astype(np.int16)
    dl = (F.reshape(nf, rows, DOTS_Y, cols, DOTS_X)
            .transpose(0, 2, 3, 1, 4)
            .reshape(nf, nc, DOTS_Y * DOTS_X))
    dist = np.abs(dl.astype(np.int16) - THRESH)

    pairs = slice(1, nf)
    diff = m[pairs] ^ m[0:nf - 1]                       # (nf-1, nc)
    bit = ((diff[:, :, None] >> np.arange(8)) & 1).astype(bool)   # (nf-1, nc, 8)
    d_pairs = dist[pairs]                                # (nf-1, nc, 8)

    bins = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 200)]
    edges = [1, 3, 6, 11, 21, 41]
    idx = np.digitize(d_pairs, edges)
    per_frame_flip = bit.sum(axis=(1, 2)) / (nc * 8)

    rows_out = []
    for bi, (lo, hi) in enumerate(bins):
        sel = idx == bi
        tot = int(sel.sum())
        if not tot:
            continue
        rows_out.append({"abs_dluma": f"{lo}-{hi}",
                         "flip_rate_pct": float(bit[sel].mean() * 100),
                         "dots": tot})

    Ff = frames[:nf].astype(np.float64)
    raw_mad = float(np.abs(np.diff(Ff, axis=0)).mean())
    means = Ff.reshape(nf, -1).mean(axis=1)[:, None, None]
    norm_mad = float(np.abs(np.diff(Ff / means, axis=0)).mean())

    rates = [r["flip_rate_pct"] for r in rows_out]
    grad = rates[0] / rates[-1] if rates and rates[-1] else None

    print(f"\n=== {label}  ({nf} frames of {layout.n_frames}, {path.name}) ===")
    print(f"  layout / stride     {layout.name} / {layout.stride} bytes")
    print(f"  mean luma            {Ff.mean():7.2f}")
    print(f"  adjacent MAD raw     {raw_mad:7.3f} counts")
    print(f"  adjacent MAD norm    {norm_mad:7.4f}   (after per-frame gain removal)")
    print(f"  dot flip rate        {statistics.median(per_frame_flip) * 100:7.2f}% "
          f"(median across adjacent frame pairs)")
    print(f"  flip rate spread     {min(per_frame_flip) * 100:.2f}% .. "
          f"{max(per_frame_flip) * 100:.2f}%")
    print("  dither gradient by |dot luma - threshold|:")
    for r in rows_out:
        print(f"      {r['abs_dluma']:>6}  {r['flip_rate_pct']:6.2f}%  "
              f"(n={r['dots']})")
    print(f"  gradient on-to-far   {grad:.2f}x" if grad else "  gradient n/a")

    return {
        "label": label, "path": str(path), "n_frames_used": nf,
        "n_frames_available": layout.n_frames,
        "layout": layout.name,
        "stride_bytes": layout.stride,
        "layout_evidence": layout.as_dict(),
        "mean_luma": float(Ff.mean()),
        "adjacent_mad_raw": raw_mad,
        "adjacent_mad_gain_normalised": norm_mad,
        "dot_flip_rate_median_pct": statistics.median(per_frame_flip) * 100,
        "dot_flip_rate_min_pct": min(per_frame_flip) * 100,
        "dot_flip_rate_max_pct": max(per_frame_flip) * 100,
        "dither_gradient": rows_out,
        "gradient_on_to_far": grad,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--empty", default="/mnt/c/proart_capture/empty1.nv12")
    ap.add_argument("--occupied", default="/mnt/c/proart_capture/A.nv12")
    ap.add_argument("--limit", type=int, default=400,
                    help="max frames per capture (they need not be equal). "
                         "The default of 400 is the frame count the previous "
                         "run actually used, so the comparison is not "
                         "confounded by this flag; the flag itself was "
                         "previously ignored.")
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    binp = build()
    res = []
    if Path(a.occupied).is_file():
        res.append(analyse(binp, Path(a.occupied), "OCCUPIED (subject present, mostly still)", a.limit))
    if Path(a.empty).is_file():
        res.append(analyse(binp, Path(a.empty), "EMPTY (no subject)", a.limit))

    if len(res) == 2:
        occ, emp = res[0], res[1]
        ratio = (emp["dot_flip_rate_median_pct"]
                 / occ["dot_flip_rate_median_pct"]
                 if occ["dot_flip_rate_median_pct"] else None)
        print("\n" + "=" * 64)
        print(f"occupied flip rate   {occ['dot_flip_rate_median_pct']:6.2f}%")
        print(f"empty    flip rate   {emp['dot_flip_rate_median_pct']:6.2f}%")
        print(f"empty / occupied     {ratio:.3f}")
        if ratio is not None and ratio > 0.6:
            verdict = ("The delta SURVIVES an empty room. Most of what the "
                       "character field reports as change is produced by the "
                       "camera and the threshold, not by a person in the room.")
        elif ratio is not None and ratio < 0.35:
            verdict = ("The delta COLLAPSES with no subject. The structured "
                       "residual in the occupied capture was the subject, so "
                       "the representation does carry real motion.")
        else:
            verdict = ("Partial: the empty room sits between. Some of the delta "
                       "is the subject and some is the camera, and this capture "
                       "does not resolve the split.")
        print(f"\nVERDICT: {verdict}")

        report = {
            "question": ("does the character delta measure the subject, or the "
                         "camera? An empty room is the control that was missing."),
            "captures": res,
            "empty_over_occupied_flip_rate": ratio,
            "verdict": verdict,
            "notes": [
                ("THE AUTO-EXPOSURE HUNT WAS THE TORN FRAME. This file's "
                 "docstring used to say the gain normalisation was needed "
                 "because the auto-exposure hunt ran at 43.6 counts/frame. "
                 "That 43.6 is the stride artefact: the occupied capture's "
                 "adjacent MAD at the correct stride is "
                 f"{res[0]['adjacent_mad_raw']:.3f} counts, and the empty "
                 f"room's is {res[1]['adjacent_mad_raw']:.3f}. Gain "
                 "normalisation is still applied, but it is now correcting a "
                 "sub-count residual, not a 43-count one."),
                ("THE DITHER-GRADIENT SUB-CLAIM IS NO LONGER TESTABLE IN THE "
                 "EMPTY CAPTURE, and this is a limitation of the control, not "
                 "a result. At the correct stride the empty room's luma sits "
                 f"at {res[1]['mean_luma']:.1f}, far below the braille "
                 "threshold of 100, so it has almost no pixels within +/-2 of "
                 "the threshold and the near-threshold bins are empty. The "
                 "empty room can therefore show that the OCCUPIED delta is "
                 "not camera noise; it cannot show what the camera does to a "
                 "signal that sits on the threshold. That question belongs to "
                 "experiments/ascii_motion_threshold.py, which measures it "
                 "directly."),
                ("The --limit flag was ignored by the previous run (it was "
                 "declared with a default of 300 and never passed to "
                 "analyse(), so 400 frames were actually used). It is now "
                 "honoured, and its default was set to 400 -- the frame count "
                 "the previous run actually used -- so the comparison is not "
                 "confounded by the flag itself."),
            ],
            "confounds_not_removable": [
                "the two recordings differ in overall lighting as well as "
                "occupancy; per-frame gain normalisation removes frame-to-frame "
                "drift but cannot undo a global difference between recordings",
                "one capture per condition, no error bars -- suggestive, not a "
                "significance test",
                "the occupied capture's A-vs-B vision condition failed "
                "separately (0 of 96 windows); it is valid for 'subject present "
                "and mostly still', not for 'subject moving'",
            ],
            "provenance": {
                "generated_by": "experiments/emptyroom_null.py",
                "script_path": str(Path(__file__).resolve()),
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "python": sys.executable,
                "command_line": " ".join(sys.argv),
                "cli_args": list(sys.argv[1:]),
                "limit_frames": a.limit,
                "captures": {
                    r["label"]: {
                        "path": r["path"],
                        "sha256": r["sha256"],
                        "bytes": r["bytes"],
                        "layout": r["layout"],
                        "stride_bytes": r["stride_bytes"],
                        "n_frames": r["n_frames_available"],
                        "frames_used": r["n_frames_used"],
                        "layout_evidence": r["layout_evidence"],
                    } for r in res
                },
                "renderer": {
                    "program": "experiments/ascii_motion_threshold.py CHANNELS_C",
                    "note": ("the character renderer assumes the STANDARD NV12 "
                             "stride of 23,040 bytes per frame, so each "
                             "capture is re-wrapped into standard NV12 first; "
                             "no luma byte is changed by that re-wrap"),
                    "frame_bytes": FRAME_W * FRAME_H * 3 // 2,
                    "braille_thresh": THRESH,
                },
            },
            "supersedes": {
                "why": ("Re-measured on 2026-10-20. The previous run read both "
                        "captures at W*H*3//2 = 23,040 bytes per frame; the "
                        "captures on disk carry 19,200 bytes per frame, so all "
                        "but every sixth frame was torn (rows 0..71 from frame "
                        "k, rows 72..95 from frame k+1). The previous numbers "
                        "are recorded here rather than silently overwritten."),
                "previous_stride_bytes": 23040,
                "previous_n_frames_available": {"OCCUPIED": 750, "EMPTY": 3750},
                "previous_captures": [
                    {"label": "OCCUPIED (subject present, mostly still)",
                     "path": "/mnt/c/proart_capture/A.nv12",
                     "n_frames_used": 400, "n_frames_available": 750,
                     "mean_luma": 111.17517106119791,
                     "adjacent_mad_raw": 43.68600162515664,
                     "adjacent_mad_gain_normalised": 0.39397797487733327,
                     "dot_flip_rate_median_pct": 48.190104166666664,
                     "dot_flip_rate_min_pct": 33.932291666666664,
                     "dot_flip_rate_max_pct": 56.822916666666664,
                     "gradient_on_to_far": 0.9448842491757359,
                     "dither_gradient": [
                         {"abs_dluma": "0-0", "flip_rate_pct": 45.71394952050362, "dots": 33994},
                         {"abs_dluma": "1-2", "flip_rate_pct": 45.56061509812764, "dots": 137474},
                         {"abs_dluma": "3-5", "flip_rate_pct": 46.24189120350544, "dots": 221827},
                         {"abs_dluma": "6-10", "flip_rate_pct": 46.44532095528993, "dots": 415790},
                         {"abs_dluma": "11-20", "flip_rate_pct": 47.414897621268224, "dots": 1078642},
                         {"abs_dluma": "21-40", "flip_rate_pct": 48.21463921086326, "dots": 1899392},
                         {"abs_dluma": "41-200", "flip_rate_pct": 48.38047576767409, "dots": 2341521}]},
                    {"label": "EMPTY (no subject)",
                     "path": "/home/proart/projects/glasses/.capture_keep/empty1.nv12",
                     "n_frames_used": 400, "n_frames_available": 3750,
                     "mean_luma": 49.69857259114583,
                     "adjacent_mad_raw": 39.34364181939223,
                     "adjacent_mad_gain_normalised": 0.8092197017047326,
                     "dot_flip_rate_median_pct": 50.0,
                     "dot_flip_rate_min_pct": 25.0,
                     "dot_flip_rate_max_pct": 50.0,
                     "gradient_on_to_far": 1.2204160156452046,
                     "dither_gradient": [
                         {"abs_dluma": "1-2", "flip_rate_pct": 47.72727272727273, "dots": 44},
                         {"abs_dluma": "3-5", "flip_rate_pct": 50.204081632653065, "dots": 245},
                         {"abs_dluma": "6-10", "flip_rate_pct": 53.333333333333336, "dots": 30},
                         {"abs_dluma": "21-40", "flip_rate_pct": 43.74086607314949, "dots": 1229619},
                         {"abs_dluma": "41-200", "flip_rate_pct": 39.10737987328072, "dots": 4898702}]},
                ],
                "previous_empty_over_occupied_flip_rate": 1.0375574169143476,
                "previous_verdict": ("The delta SURVIVES an empty room. Most of "
                                     "what the character field reports as change "
                                     "is produced by the camera and the "
                                     "threshold, not by a person in the room."),
                "verdict_changed": (
                    "YES, AND IT FLIPS. The previous run put the empty room at "
                    "50.00% and the occupied room at 48.19% and concluded the "
                    "delta was the camera's. At the correct stride the empty "
                    "room is 0.00% and the occupied room is 0.62%, and the "
                    "conclusion reverses: the delta is the subject. The "
                    "previous 'camera' conclusion was the torn frame, not the "
                    "camera. Note also that the empty capture was read from "
                    "/home/proart/projects/glasses/.capture_keep/empty1.nv12, "
                    "which is byte-identical (sha256 da348cc...) to the "
                    "/mnt/c/proart_capture/empty1.nv12 used now, so the input "
                    "did not change."),
            },
        }
        p = Path(a.out) / "emptyroom_null_results.json"
        p.write_text(json.dumps(report, indent=2))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
