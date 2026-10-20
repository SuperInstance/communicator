"""
pose_test.py -- can the camera alone tell facing from head-turned?

THE FRAME LAYOUT, FIRST.

The captures on disk are NOT standard NV12. `C:\\proart_capture\\nv12small.py`
writes 15,360 luma bytes + 1,920 U + 1,920 V per 160x96 frame -- 19,200 bytes
-- not the `W*H*3//2` = 23,040 this file used to assume. Read at 23,040, all
but every sixth frame is torn (rows 0..71 from frame k, rows 72..95 from frame
k+1) and the 750 frames that come out are not 750 frames of anything.

The layout is DETECTED from the file's content by `src/nv12.py` -- the
candidate whose luma has the smallest mean consecutive |diff| wins, and an
ambiguous file is an error rather than a guess. Nothing here computes a stride
by hand. See `src/nv12.py` and `docs/research/NV12-STRIDE.md`.

THE TEST, PROPERLY SPECIFIED.

Everything this project tried before failed one of two ways. The audio routes
could not separate the conditions at all. The vision routes were swamped: the
character field changed ~50% of its dots per frame with nobody in the room, so
a facing-vs-turned comparison had no null to be measured against.

Now there is a null. `row_band_removal.py` shows that removing the row-constant
component takes an EMPTY room from 50.00% flip to 0.01%, while leaving the
subject's column structure untouched (col std 7.08 -> 7.18, column energy share
4.86% -> 26.20%). So the instrument has a floor, and the floor is essentially
zero.

That changes what a null result MEANS. Before, a null was uninterpretable,
because a null was what the noise produced anyway. Now a null is informative.

THE STATISTIC, AND WHY NOT THE FLIP RATE.

The flip RATE is a poor discriminator and it is worth saying why. A speaking
person changes a lot: even after row removal the occupied capture sits at ~48%
flips, which is near the theoretical maximum. A rate that saturates cannot
separate two conditions that are both near the ceiling. The facing and turned
captures are both at 40-49%; the rate is measuring "a person is moving", not
"which way the head points".

So the statistic is SPATIAL: WHERE the change happens.

  M_c(x, y)  = mean over the speech window of |frame(t+1) - frame(t)| at each
                pixel, for condition c, after row removal.

The empty room's M_empty is the NOISE FLOOR, a spatial map of what changes when
nothing does. The question is then:

  does |M_A - M_B| exceed M_empty, in a spatially LOCALISED way?

A localised excess means the two conditions differ in a specific part of the
frame -- which is what a head turn would look like. A diffuse excess means
something global differs, which is not a pose signal. And no excess means the
conditions really are the same, which is a finding about the capture, not about
vision.

THE CONTROLS, because a single comparison proves nothing.

  C1 EMPTY-ROOM NULL. The noise floor, measured, not assumed.
  C2 WITHIN-CONDITION SPLIT. Split each condition's speech window in half. The
     two halves of the SAME condition must be MORE similar to each other than
     A is to B. If not, the "difference" is just the scene changing over time
     and the test is measuring drift, not pose.
  C3 SILENT CONDITION. C is facing and silent. If C's change map matches A's
     closely and differs from B's, the difference is in the MOTION, not the
     pose -- which would be a different and more interesting result than the
     one we want, and must be distinguishable.
  C4 A-SHIFTED-B. Compare A against B shifted in time. If the real A-vs-B
     difference is no larger than a time-shifted one, there is no pose signal
     at all.

C2 and C4 are the ones that usually kill results like this, and they are the
reason the previous "7.85 pp gap" was not evidence of anything: that gap was
measured with a saturated rate against a noise floor of 50%, and row removal
SHRINKS it to 3.65 pp, which is consistent with the difference having been
mostly nuisance all along.

WHAT THIS CANNOT DO.

It cannot establish that vision solves "talking to me". It can only establish
whether the camera carries a usable pose signal in this room, with this
lighting, in this capture. If C2 and C4 pass, the morning capture is worth
recording. If they fail, the honest answer is that the manipulation was too
weak -- which is a statement about the capture, and it is the statement the
previous three-condition run deserved.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAME_W, FRAME_H = 160, 96
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.nv12 import read_luma  # noqa: E402


def sha256_file(path) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_y(path: Path, nf: int):
    """The luma of the first `nf` frames, as float64.

    The layout is detected, not assumed -- see `src/nv12.py`.
    """
    luma, layout = read_luma(path, width=FRAME_W, height=FRAME_H)
    n = min(nf, layout.n_frames)
    return luma[:n].astype(np.float64), layout


def row_remove(F):
    prof = np.median(F, axis=2)
    g = prof.mean(axis=1, keepdims=True)
    return F - (prof - g)[:, :, None]


def change_map(F, t0, t1):
    """Mean |frame(t+1)-frame(t)| over [t0,t1) seconds, per pixel."""
    a = int(t0 * 30)
    b = min(int(t1 * 30), len(F) - 1)
    if b - a < 5:
        return None
    d = np.abs(F[a + 1:b + 1] - F[a:b])
    return d.mean(axis=0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--occupied", default="/mnt/c/proart_capture/A.nv12")
    ap.add_argument("--turned", default="/mnt/c/proart_capture/B.nv12")
    ap.add_argument("--silent", default="/mnt/c/proart_capture/C.nv12")
    ap.add_argument("--empty", default="/home/proart/projects/glasses/.capture_keep/empty1.nv12")
    ap.add_argument("--t0", type=float, default=0.8)
    ap.add_argument("--t1", type=float, default=9.0)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    caps = {}
    inputs = {}
    for key, p in (("A_facing_speaking", a.occupied), ("B_turned_speaking", a.turned),
                   ("C_facing_silent", a.silent), ("EMPTY", a.empty)):
        if Path(p).is_file():
            F, layout = load_y(Path(p), 1200)
            caps[key] = row_remove(F)
            inputs[key] = {
                "path": str(Path(p).resolve()),
                "sha256": sha256_file(p),
                "bytes": Path(p).stat().st_size,
                "layout": layout.name,
                "stride_bytes": layout.stride,
                "n_frames": layout.n_frames,
                "frames_used": int(len(F)),
                "layout_evidence": layout.as_dict(),
            }

    if "A_facing_speaking" not in caps or "EMPTY" not in caps:
        sys.exit("need at least the facing capture and the empty capture")

    M = {k: change_map(v, a.t0, a.t1) for k, v in caps.items()}
    M = {k: v for k, v in M.items() if v is not None}
    null = M["EMPTY"]

    print(f"speech window {a.t0}-{a.t1}s")
    print(f"  EMPTY null map: mean {null.mean():.3f}  max {null.max():.3f}  "
          f"p99 {np.percentile(null, 99):.3f} counts")

    results = {}

    def compare(name, k1, k2):
        d = np.abs(M[k1] - M[k2])
        # how much of the difference is localised, i.e. above the null floor
        excess = np.clip(d - null, 0, None)
        frac = float((d > null).mean())
        peak = float(d.max())
        # localisation: the top 5% of pixels hold how much of the excess?
        flat = np.sort(excess.ravel())[::-1]
        top = flat[: max(1, len(flat) // 20)].sum()
        loc = float(top / (flat.sum() + 1e-9))
        print(f"  {name:34s} mean|d| {d.mean():6.3f}  max {peak:6.3f}  "
              f"frac>null {frac * 100:5.1f}%  top5%-holds {loc * 100:5.1f}% of excess")
        return {"mean_abs_diff": float(d.mean()), "max_diff": peak,
                "frac_above_null": frac, "top5pct_excess_share": loc,
                "excess_mean": float(excess.mean())}

    print("\npairwise differences in the spatial change map:")
    if "A_facing_speaking" in M and "B_turned_speaking" in M:
        results["A_vs_B"] = compare("A facing  vs  B turned", "A_facing_speaking",
                                    "B_turned_speaking")
    if "A_facing_speaking" in M and "C_facing_silent" in M:
        results["A_vs_C"] = compare("A speaking vs C silent", "A_facing_speaking",
                                    "C_facing_silent")
    if "B_turned_speaking" in M and "C_facing_silent" in M:
        results["B_vs_C"] = compare("B turned  vs  C silent", "B_turned_speaking",
                                    "C_facing_silent")

    # -- C2 WITHIN-CONDITION SPLIT -------------------------------------------
    print("\nC2  within-condition split (same condition, two halves of its window):")
    for key in ("A_facing_speaking", "B_turned_speaking"):
        if key not in caps:
            continue
        F = caps[key]
        mid = (a.t0 + a.t1) / 2
        m1 = change_map(F, a.t0, mid)
        m2 = change_map(F, mid, a.t1)
        if m1 is None or m2 is None:
            continue
        d = np.abs(m1 - m2)
        print(f"  {key:22s} half-vs-half mean|d| {d.mean():6.3f}  max {d.max():6.3f}")
        results[f"within_{key}"] = {"mean_abs_diff": float(d.mean()),
                                    "max_diff": float(d.max())}

    # -- C4 TIME-SHIFTED CONTROL ---------------------------------------------
    print("\nC4  time-shifted control (A vs B shifted in time -- should match A-vs-B if no pose signal):")
    if "A_facing_speaking" in caps and "B_turned_speaking" in caps:
        F = caps["B_turned_speaking"]
        shifted = change_map(F, a.t0 + 1.0, a.t1 + 1.0)
        if shifted is not None:
            d = np.abs(M["A_facing_speaking"] - shifted)
            print(f"  A vs B shifted 1.0s        mean|d| {d.mean():6.3f}  max {d.max():6.3f}")
            results["A_vs_B_shifted"] = {"mean_abs_diff": float(d.mean()),
                                          "max_diff": float(d.max())}

    out = {
        "question": ("does the camera carry a usable pose signal, measured "
                     "against a measured empty-room null?"),
        "window_s": [a.t0, a.t1],
        "null_map_mean": float(null.mean()), "null_map_max": float(null.max()),
        "null_map_p99": float(np.percentile(null, 99)),
        "results": results,
        "supersedes": {
            "why": ("Re-measured on 2026-10-20. The previous run read the "
                    "captures at W*H*3//2 = 23,040 bytes per frame; the "
                    "captures on disk carry 19,200 bytes per frame, so all "
                    "but every sixth frame was torn and the 'drift' and the "
                    "'difference' both measured that tear. The previous "
                    "numbers are recorded here rather than silently "
                    "overwritten."),
            "previous_stride_bytes": 23040,
            "previous_n_frames": {"A_facing_speaking": 750,
                                  "B_turned_speaking": 750,
                                  "C_facing_silent": 750,
                                  "EMPTY": 3750},
            "previous_null_map_mean": 43.560,
            "previous_results": {
                "A_vs_B": {"mean_abs_diff": 9.182},
                "A_vs_C": {"mean_abs_diff": 4.914},
                "within_A_facing_speaking": {"mean_abs_diff": 2.878},
                "within_B_turned_speaking": {"mean_abs_diff": 1.917},
                "A_vs_B_shifted": {"mean_abs_diff": 9.178},
            },
            "previous_verdict": (
                "NO POSE SIGNAL IN THIS STATISTIC. A-vs-B is 9.182 and the "
                "time-shifted control is 9.178 -- indistinguishable. The "
                "apparent difference between facing and turned is the scene "
                "changing over time, not a head direction. SPEECH IS VISIBLE. "
                "Facing-and-speaking vs facing-and-silent differ by 4.914 "
                "against A's own within-condition split of 2.878 (1.7x), and "
                "the excess is 79% concentrated in the top 5% of pixels -- it "
                "is LOCALISED, which is what a real signal looks like and "
                "what the A-vs-B comparison is not."),
        },
        "provenance": {
            "generated_by": "experiments/pose_test.py",
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.executable,
            "command_line": " ".join(sys.argv),
            "cli_args": list(sys.argv[1:]),
            "captures": inputs,
            "fps_assumed": 30.0,
            "window_s": [a.t0, a.t1],
        },
        "limits": [
            "the empty capture is later and darker than the others; the null "
            "floor is therefore probably an OVER-estimate, which makes the "
            "test conservative rather than optimistic",
            "THIS STATISTIC IS BLIND TO HEAD POSE, and that is now known "
            "rather than assumed. A separate search over fourteen candidate "
            "features (experiments/pose_feature_search.py) was RE-RUN at the "
            "correct frame stride on 2026-10-20 and now finds NO clean pose "
            "signal. mirror_cov, the frame's covariance with its own "
            "horizontal mirror, still separates facing from turned (margin "
            "+1.083) but ALSO separates the two FACING conditions (A-vs-C "
            "0.418 against a control of 0.331), so it reads as motion or "
            "session rather than as pose. The previous run's claim that "
            "mirror_cov was pose-consistent (margin +0.814, 'A and C agree, "
            "only B differs') was measured on torn frames and does not "
            "survive. So the null below is a property of the CHANGE-MAP "
            "MAGNITUDE, and the earlier explanation -- that the head turn "
            "was too small to register -- is an assumption resting on a "
            "blind instrument, which the re-run neither confirms nor "
            "refutes. See docs/research/POSE-FEATURE.md.",
            "one capture per condition, no error bars, no cross-validation",
            "this measures a POSE signal. It does not measure whether anyone "
            "is talking to you",
        ],
    }
    ab = results.get("A_vs_B", {}).get("mean_abs_diff")
    ab_sh = results.get("A_vs_B_shifted", {}).get("mean_abs_diff")
    # THE CONTROLS, AND WHICH ONE MATCHES WHICH CONTRAST.
    #
    # The first version took `min(...)` over every within-condition split,
    # which is the most FAVOURABLE null available rather than the matched
    # one: it picked condition B's 1.917 to divide the A-vs-C contrast of
    # 4.914, giving "2.6x". The condition in the A-vs-C contrast is A, so
    # the matched control is A's own within-split, 2.878, and the factor is
    # 1.71x. (NOTES-AUDIT.md F1.)
    #
    # For the A-vs-B contrast the matched control is not obvious -- the
    # conservative choice is the LARGER of the two splits, so that a claim
    # has to clear the least favourable null. The verdict is unchanged
    # either way: 9.182 against a time-shifted control of 9.178 is nothing.
    half_a = results.get("within_A_facing_speaking", {}).get("mean_abs_diff")
    half_b = results.get("within_B_turned_speaking", {}).get("mean_abs_diff")
    half = min((v["mean_abs_diff"] for k, v in results.items()
                if k.startswith("within_")), default=None)
    half_pose = max((x for x in (half_a, half_b) if x is not None), default=None)
    ac = results.get("A_vs_C", {}).get("mean_abs_diff")
    ac_loc = results.get("A_vs_C", {}).get("top5pct_excess_share")

    # A verdict needs a MARGIN, not a strict inequality. The first version of
    # this function concluded "there is a localised pose signal" because
    # 9.182 > 9.178 -- a difference of 0.004, i.e. nothing. The time-shifted
    # control is the whole point of the test, and a real pose signal has to
    # clear it by a wide margin, not by a rounding error.
    MARGIN = 1.25
    if ab is not None and ab_sh is not None and half_pose is not None:
        if ab > ab_sh * MARGIN and ab > half_pose * MARGIN:
            out["pose_verdict"] = (
                f"A-vs-B ({ab:.3f}) clears the time-shifted control "
                f"({ab_sh:.3f}) by more than {MARGIN}x. There is a localised "
                "pose signal above the null.")
        else:
            out["pose_verdict"] = (
                f"NO POSE SIGNAL IN THIS STATISTIC. A-vs-B is {ab:.3f} and the "
                f"time-shifted control is {ab_sh:.3f} -- indistinguishable. The "
                "apparent difference between facing and turned is the scene "
                "changing over time, not a head direction.")

    # THE OTHER HALF, and it is the better half.
    #
    # The control is A's OWN within-split, because A is the condition in the
    # A-vs-C contrast. The earlier version divided by the minimum split
    # across all conditions -- 1.917, which belongs to B -- and printed
    # "2.6x". Against the matched control it is 1.71x. The conclusion is
    # unchanged and the localisation is what carries it: 79% of the excess
    # sits in the top 5% of pixels, which is what a real signal looks like
    # and what the A-vs-B comparison is not.
    if ac is not None and half_a is not None:
        ratio = ac / half_a
        if ac > half_a * MARGIN:
            out["speech_verdict"] = (
                f"SPEECH IS VISIBLE. Facing-and-speaking vs facing-and-silent "
                f"differ by {ac:.3f} against A's own within-condition split of "
                f"{half_a:.3f} ({ratio:.1f}x), and the excess is "
                f"{ac_loc * 100:.0f}% concentrated in the top 5% of pixels -- "
                "it is LOCALISED, which is what a real signal looks like and "
                "what the A-vs-B comparison is not. The camera channel does "
                "carry information about whether someone is speaking.")
        else:
            # The previous run reported 4.914 against a split of 2.878 -- 1.7x
            # -- and called it "SPEECH IS VISIBLE". That was measured on a
            # torn signal. On the clean signal the ratio is at the margin
            # rather than over it, so the conclusion does not carry. It is
            # said here rather than left out, because a missing verdict reads
            # as "not measured" and this one WAS measured and FAILED.
            out["speech_verdict"] = (
                f"THE SPEECH SIGNAL DOES NOT CLEAR ITS OWN CONTROL. Facing-and-"
                f"speaking vs facing-and-silent differ by {ac:.4f} against A's "
                f"own within-condition split of {half_a:.4f} ({ratio:.3f}x), "
                f"which does not clear the {MARGIN}x margin this test was given "
                f"before the numbers were seen (it misses by "
                f"{MARGIN - ratio:.4f} in ratio, i.e. "
                f"{MARGIN * half_a - ac:.6f} counts), and the excess is only "
                f"{ac_loc * 100:.0f}% localised in the top 5% of pixels "
                "against 79% before. The previous run reported 1.71x and "
                "'SPEECH IS VISIBLE'; it was measuring a torn signal. On the "
                "clean signal this statistic does NOT establish that speech is "
                "visible in the camera channel, and the honest reading is that "
                "the question is open again rather than answered.")
    out["verdict"] = (out.get("pose_verdict", "") + " " +
                      out.get("speech_verdict", "")).strip()
    p = Path(a.out) / "pose_test_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nVERDICT: {out.get('verdict', 'insufficient data')}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
