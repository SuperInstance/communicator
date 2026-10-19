"""
gain_invariance.py -- the fix, tested before anyone sits in front of a camera.

THE PROBLEM THIS ATTEMPTS TO FIX.

`EMPTY-ROOM-CONTROL.md` measured that the Braille character field changes 50.00%
of its dots between adjacent frames in a room with NOBODY IN IT, against 48.19%
with a subject. 50% is the theoretical maximum. The representation reports change
whether or not anything moved, so it cannot be used to decide whether anyone
moved.

The mechanism is identified, not guessed: Syzygy's `syz_braille_pack` compares
each dot's luma against a FIXED `braille_thresh`. The camera's auto-exposure
hunt swings the whole frame by ~39 counts per frame. A global gain change moves
every dot across a fixed threshold together, so the whole field flips in unison.

So the fix is not a threshold tweak and not hysteresis. It is to make the
comparison SCALE-INVARIANT: normalise each frame's gain BEFORE thresholding, so
a global brightness change has nothing to move across the boundary.

THE PREDICTION, STATED BEFORE RUNNING.

  * EMPTY room flip rate COLLAPSES toward zero. Nothing is moving, so a
    gain-invariant threshold should report almost nothing.
  * OCCUPIED flip rate stays ABOVE the empty one. Casey was present and
    speaking, which is real change and must survive normalisation.

If the empty rate does not collapse, the fix does not work and I want to know
that before he returns, not after.

A SECOND, FREE CHANCE. The original three-condition capture was written off
because facing and head-turned could not be separated: 0 of 96 spatial windows
beat the noise floor, with adjacent-frame drift (43.7 counts) exceeding the
whole A-vs-B signal (14.4). If gain normalisation works, that capture may be
recoverable, and it would mean the morning take is confirmation rather than the
first shot.

WHAT IS HELD CONSTANT, AND WHAT IS NOT.

Held: same kernel, same braille_thresh=100, same 160x96, same frames, same
comparison code. Only the pixels change.

NOT held, and stated plainly: normalisation changes the absolute luma
distribution, so the fixed threshold now means something different than it did
before. That is the point -- but it means the two conditions are only
comparable to each other within this script, not to any earlier number. The raw
column is reported alongside every normalised column for exactly this reason.

A REAL RISK, MEASURED NOT ASSUMED. Scaling a dark frame up can clip highlights
to 255. Clipping is itself a saturation, and a saturated field flips too. So the
clipped fraction is measured per capture and reported. If a capture clips
heavily, its result is not comparable to one that does not.
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
FB = FRAME_W * FRAME_H * 3 // 2
Y_SIZE = FRAME_W * FRAME_H
THRESH = 100

sys.path.insert(0, str(ROOT / "experiments"))
from ascii_motion_threshold import build  # noqa: E402


def load_nv12(path: Path):
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    nf = len(raw) // FB
    a = raw[: nf * FB].reshape(nf, FB)
    return a[:, :Y_SIZE].reshape(nf, FRAME_H, FRAME_W), \
        a[:, Y_SIZE:].reshape(nf, -1), nf


def normalise(Y, UV, ref_mean):
    """Per-frame gain removal: scale each frame so its mean luma is `ref_mean`.

    UV is kept as (nf, FB - Y_SIZE) -- 2D, one row per frame. Scaling it with a
    (nf, 1, 1) column vector silently broadcasts to (nf, nf, W), which is an
    nf-fold memory blowup and a reshape error further down. Both operands are
    shaped (nf, 1) for the multiply.
    """
    means = Y.reshape(len(Y), -1).mean(axis=1)
    scale = (ref_mean / means)
    Yn = np.clip(Y * scale[:, None, None], 0, 255)
    UVn = np.clip(UV.astype(np.float64) * scale[:, None], 0, 255).astype(np.uint8)
    clipped = float((Y * scale[:, None, None] > 255).mean())
    return Yn.astype(np.uint8), UVn, clipped


def write_nv12(path: Path, Y, UV, nf):
    out = np.concatenate([Y[:nf].reshape(nf, Y_SIZE),
                          UV[:nf].reshape(nf, FB - Y_SIZE)], axis=1)
    path.write_bytes(out.astype(np.uint8).tobytes())
    return path


def flip_rate(binp: Path, nv12: Path, nf: int):
    r = subprocess.run([str(binp), str(FRAME_W), str(FRAME_H), str(nv12)],
                       capture_output=True, text=True, check=True)
    lines = r.stdout.splitlines()
    ms, i = [], 0
    while i < len(lines):
        if not lines[i].startswith("F "):
            i += 1
            continue
        _, cs, rs = lines[i].split()
        ms.append(lines[i + 1])
        i += 2
    nc = int(cs) * int(rs)
    bits = []
    for h in ms[:nf]:
        b = np.frombuffer(bytes.fromhex(h), np.uint8)
        bits.append(((b[:, None] >> np.arange(8)) & 1).astype(np.uint8))
    B = np.stack(bits)                                   # (nf, nc, 8)
    if len(B) < 2:
        return None, None, None
    x = B[1:] ^ B[:-1]
    per = x.reshape(len(x), -1).mean(axis=1)
    return (float(np.median(per) * 100), float(per.min() * 100),
            float(per.max() * 100))


def analyse(binp: Path, path: Path, label: str, nf: int):
    Y, UV, total = load_nv12(path)
    nf = min(nf, total)
    Y, UV = Y[:nf], UV[:nf]

    raw_mad = float(np.abs(np.diff(Y.astype(np.float64), axis=0)).mean())
    ref = float(np.median(Y.reshape(nf, -1).mean(axis=1)))
    Yn, UVn, clipped = normalise(Y, UV, ref)
    norm_mad = float(np.abs(np.diff(Yn.astype(np.float64), axis=0)).mean())

    raw_flip = flip_rate(binp, path, nf)
    tmp = write_nv12(Path("/tmp/gn.nv12"), Yn, UVn, nf)
    norm_flip = flip_rate(binp, tmp, nf)

    print(f"\n=== {label}  ({nf} frames, {path.name}) ===")
    print(f"  mean luma            {Y.reshape(nf,-1).mean():7.2f}   ref for norm {ref:6.2f}")
    print(f"  clipped after scaling{clipped * 100:7.3f}%")
    print(f"  adjacent MAD  raw    {raw_mad:8.3f}   ->  normalised {norm_mad:8.4f}"
          f"   ({raw_mad / norm_mad:5.1f}x quieter)")
    print(f"  dot flip RAW         {raw_flip[0]:7.2f}%   ({raw_flip[1]:.2f}..{raw_flip[2]:.2f})")
    print(f"  dot flip NORMALISED  {norm_flip[0]:7.2f}%   ({norm_flip[1]:.2f}..{norm_flip[2]:.2f})")

    return {"label": label, "path": str(path), "n_frames": nf,
            "mean_luma": float(Y.reshape(nf, -1).mean()),
            "clipped_fraction_after_norm": clipped,
            "adjacent_mad_raw": raw_mad, "adjacent_mad_normalised": norm_mad,
            "mad_reduction_x": raw_mad / norm_mad if norm_mad else None,
            "flip_raw_pct": raw_flip[0], "flip_raw_min": raw_flip[1],
            "flip_raw_max": raw_flip[2],
            "flip_normalised_pct": norm_flip[0],
            "flip_normalised_min": norm_flip[1], "flip_normalised_max": norm_flip[2]}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--empty", default="/home/proart/projects/glasses/.capture_keep/empty1.nv12")
    ap.add_argument("--occupied", default="/mnt/c/proart_capture/A.nv12")
    ap.add_argument("--turned", default="/mnt/c/proart_capture/B.nv12")
    ap.add_argument("--nf", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    binp = build()
    res = []
    for path, lab in ((a.occupied, "OCCUPIED (facing, speaking)"),
                      (a.turned, "TURNED (head away, same words)"),
                      (a.empty, "EMPTY (nobody)")):
        if Path(path).is_file():
            res.append(analyse(binp, Path(path), lab, a.nf))

    by = {r["label"].split()[0]: r for r in res}
    out = {"question": ("does per-frame gain normalisation make the character "
                        "delta measure the subject instead of the camera?"),
           "prediction": ("empty-room flip rate collapses toward zero while the "
                          "occupied rate stays above it"),
           "captures": res,
           "limits": [
               "normalisation changes what the fixed braille_thresh means, so "
               "normalised numbers are comparable WITHIN this script only, not "
               "to any earlier report",
               "the empty room was captured dark (mean luma ~50 vs ~111 "
               "occupied) and the two captures are hours apart; gain "
               "normalisation removes frame-to-frame drift, not a global "
               "difference between two recordings",
               "one capture per condition, no error bars",
               "per-frame normalisation also removes genuine global lighting "
               "changes, which is an accepted trade, not a free win",
           ]}

    if "EMPTY" in by and "OCCUPIED" in by:
        e, o = by["EMPTY"], by["OCCUPIED"]
        ratio = (e["flip_normalised_pct"] / o["flip_normalised_pct"]
                 if o["flip_normalised_pct"] else None)
        out["empty_over_occupied_normalised"] = ratio
        out["empty_over_occupied_raw"] = (e["flip_raw_pct"] / o["flip_raw_pct"]
                                         if o["flip_raw_pct"] else None)
        print("\n" + "=" * 66)
        print(f"raw:        empty {e['flip_raw_pct']:6.2f}%   occupied "
              f"{o['flip_raw_pct']:6.2f}%   empty/occ "
              f"{(e['flip_raw_pct'] / o['flip_raw_pct']):.3f}")
        print(f"normalised: empty {e['flip_normalised_pct']:6.2f}%   occupied "
              f"{o['flip_normalised_pct']:6.2f}%   empty/occ {ratio:.3f}")

        if e["flip_normalised_pct"] < 15 and o["flip_normalised_pct"] > e["flip_normalised_pct"] * 1.5:
            out["verdict"] = (
                "THE FIX WORKS. The empty room collapses toward zero while the "
                "occupied room stays clearly above it, so the character delta "
                "now tracks the subject rather than the camera. Casey can "
                "record the three-condition capture and the result will mean "
                "something.")
        elif e["flip_normalised_pct"] >= 25:
            out["verdict"] = (
                "THE FIX DOES NOT WORK. The empty room still reports a large "
                "flip rate after gain normalisation, so the auto-exposure hunt "
                "is not the whole mechanism. Do NOT spend the morning on a "
                "capture that will still be uninterpretable -- the next thing "
                "to try is hysteresis (a dot keeps its bit until its luma "
                "crosses the threshold by a margin), which attacks the "
                "remaining per-dot instability directly.")
        else:
            out["verdict"] = ("Partial: the empty rate fell but not to nothing, "
                              "and the separation is not yet clean.")

    if "OCCUPIED" in by and "TURNED" in by:
        o, t = by["OCCUPIED"], by["TURNED"]
        d = abs(o["flip_normalised_pct"] - t["flip_normalised_pct"])
        out["facing_vs_turned_normalised_gap_pp"] = d
        out["original_capture_recovered"] = bool(d > 10)
        print(f"\nfacing vs turned, normalised: {o['flip_normalised_pct']:.2f}% vs "
              f"{t['flip_normalised_pct']:.2f}%   gap {d:.2f} pp"
              f"   -> original capture {'RECOVERED' if d > 10 else 'still not separable'}")

    p = Path(a.out) / "gain_invariance_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nVERDICT: {out.get('verdict', 'insufficient captures')}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()