"""
emptyroom_null.py -- THE control the ASCII-motion lane never had.

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

    empty room + the same 55.9% delta   -> the delta is CAMERA, not subject
    empty room + the delta collapses    -> the delta was the SUBJECT, and the
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
        auto-exposure hunt (43.6 counts/frame, measured) otherwise swamps every
        comparison. The capture-side exposure lock FAILED on this driver --
      AUTO_EXPOSURE reads back 0.0 -- so normalisation is the only defence, and
        the residual drift after it is reported rather than assumed away.

  * The dither gradient (flip rate vs distance from threshold) is recomputed for
    the empty room. If the 3.1x monotone gradient survives with NO SUBJECT IN
    FRAME, then that gradient is a property of the camera and the threshold, not
    of anything in the room -- which is the specific claim the earlier doc could
    not make.

LIMITS, STATED UP FRONT.

  * One empty capture and one occupied capture is a demonstration, not an
    experiment with error bars. If the two come out close, that is suggestive;
    it is not a p-value.
  * The two captures differ in lighting as well as occupancy. Casey left the
    room, so absolute luma fell from ~110 to ~50. Gain normalisation removes the
    frame-to-frame component but NOT a global difference between the two
    recordings. That confound cannot be removed retrospectively and is reported,
    not hidden.
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

ROOT = Path(__file__).resolve().parents[1]
FRAME_W, FRAME_H = 160, 96
DOTS_X, DOTS_Y = 2, 4
THRESH = 100

sys.path.insert(0, str(ROOT / "experiments"))
from ascii_motion_threshold import build  # noqa: E402


def load(nv12: Path):
    fb = FRAME_W * FRAME_H * 3 // 2
    raw = np_from(nv12.read_bytes())
    nf = len(raw) // fb
    return raw[: nf * fb].reshape(nf, fb)[:, : FRAME_W * FRAME_H] \
              .reshape(nf, FRAME_H, FRAME_W)


def np_from(b):
    import numpy as np
    return np.frombuffer(b, dtype=np.uint8)


def masks(binp: Path, nv12: Path):
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
    import numpy as np
    frames = load(path)
    mf = masks(binp, path)
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

    print(f"\n=== {label}  ({nf} frames of {len(frames)}, {path.name}) ===")
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
        "n_frames_available": len(frames),
        "mean_luma": float(Ff.mean()),
        "adjacent_mad_raw": raw_mad,
        "adjacent_mad_gain_normalised": norm_mad,
        "dot_flip_rate_median_pct": statistics.median(per_frame_flip) * 100,
        "dot_flip_rate_min_pct": min(per_frame_flip) * 100,
        "dot_flip_rate_max_pct": max(per_frame_flip) * 100,
        "dither_gradient": rows_out,
        "gradient_on_to_far": grad,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--empty", default="/mnt/c/proart_capture/empty1.nv12")
    ap.add_argument("--occupied", default="/mnt/c/proart_capture/A.nv12")
    ap.add_argument("--limit", type=int, default=300,
                    help="max frames per capture (they need not be equal)")
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    binp = build()
    res = []
    if Path(a.occupied).is_file():
        res.append(analyse(binp, Path(a.occupied), "OCCUPIED (subject present, mostly still)"))
    if Path(a.empty).is_file():
        res.append(analyse(binp, Path(a.empty), "EMPTY (no subject)"))

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
        }
        p = Path(a.out) / "emptyroom_null_results.json"
        p.write_text(json.dumps(report, indent=2))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()