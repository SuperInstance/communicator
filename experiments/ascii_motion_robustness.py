"""
ascii_motion_robustness.py -- is +0.702 a mechanism, or one leverage point?

WHY THIS FILE EXISTS.

`ascii_motion_cause.py` reported

    lag   mean|dY|   cell-change%
      1       1.804         55.30
      2       3.325         58.48
      3       3.562         58.55
      4       2.629         55.75
      5       1.393         14.52      <-- the only row where both are low
      6       2.306         55.57      <-- low dY but HIGH change

and then printed "correlation = +0.702" and a verdict about auto-exposure.

That verdict is not supported, and this file exists to retract it honestly.

Two things are wrong with the inference:

  1. The correlation rests on ONE row. Lag 6 has the second-lowest mean|dY|
     and the second-HIGHEST cell-change. Remove lag 5 and the two series have
     almost nothing in common. A correlation of six points in which one point
     carries the entire effect is an anecdote. Leave-one-out is computed here so
     the fragility is visible instead of hidden in a single number.

  2. A gain loop predicts a SMOOTH periodicity -- a trough in change at every
     multiple of the hunt period, as the mean-luma autocorrelation does
     (-0.45 at lag 2, -0.68 at lag 3). What the cell delta actually shows is a
     flat ~55-58% at lags 1,2,3,4,6 and a single sharp dip at lag 5. One dip at
     one lag is not a period. It is one unusual PAIR of frames.

The competing explanation is much more mundane and much more likely: the capture
contains a duplicated or near-duplicated frame, so one lag-5 comparison is
against a frame that is nearly itself. That would produce exactly this shape --
a uniform high baseline plus one anomalously low point -- with no appeal to
camera behaviour at all.

THE TEST.

  R1 WHICH PAIR. Find the lag-5 pair with anomalously low change and compare
     its frames' mean luma. If they are near-identical, it is a repeat, and the
     delta is not telling us anything about exposure.
  R2 LEAVE-ONE-OUT. Recompute the correlation with each lag removed. A real
     mechanism survives; a leverage point collapses.
  R3 MINIMUM-PAIR FLOOR. Find the single most-similar frame pair anywhere in the
     capture at ANY lag. If a pair exists that is nearly identical, a duplicate
     frame is present and the whole lag profile is contaminated by it.

If R1/R3 find a duplicate, the correct conclusion is not "AGC" and not
"threshold dither". It is: THIS CAPTURE IS UNUSABLE FOR MOTION WORK, because
one frame is a repeat, and the honest next step is a capture that records a
per-frame checksum so repeats are detected rather than inferred.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAME_W, FRAME_H = 160, 96


def pearson(a, b):
    n = len(a)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = (sum((x - ma) ** 2 for x in a)) ** 0.5
    db = (sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / (da * db) if da and db else None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "experiments"))
    from ascii_motion import ensure, fields

    binp = ensure(Path("/tmp/syzbuild"))
    fr = fields(binp, Path(a.capture))
    nf = len(fr)
    cols, rows = fr[0][0], fr[0][1]
    n_cells = cols * rows

    fb = FRAME_W * FRAME_H * 3 // 2
    raw = Path(a.capture).read_bytes()
    means = [sum(raw[i * fb:i * fb + FRAME_W * FRAME_H]) / (FRAME_W * FRAME_H)
             for i in range(nf)]

    def rate(i, j):
        return sum(1 for k in range(n_cells)
                   if fr[i][2][k * 2:k * 2 + 2] != fr[j][2][k * 2:k * 2 + 2]) / n_cells

    # -- R1 WHICH PAIR ---------------------------------------------------------
    lag5 = [(rate(i - 5, i), i - 5, i) for i in range(5, nf)]
    lag5_sorted = sorted(lag5)
    print("R1  lag-5 pairs, lowest change first:")
    for r, i, j in lag5_sorted[:4]:
        print(f"     frame {i:>2} vs {j:>2}: change {r * 100:6.2f}%   "
              f"mean luma {means[i]:6.2f} vs {means[j]:6.2f}  "
              f"(dY {abs(means[i] - means[j]):.3f})")

    worst_r, wi, wj = lag5_sorted[0]
    typical_r = statistics.median([r for r, _, _ in lag5])
    print(f"\n     median lag-5 change {typical_r * 100:.2f}%   "
          f"lowest {worst_r * 100:.2f}%   ratio {typical_r / max(worst_r, 1e-9):.1f}x")

    # -- R3 GLOBAL MINIMUM PAIR ------------------------------------------------
    best = min(((rate(i, j), i, j)
                for i in range(nf) for j in range(i + 1, nf)),
               key=lambda t: t[0])
    print(f"\nR3  most similar pair ANYWHERE in the capture: frames {best[1]} vs "
          f"{best[2]} -> {best[0] * 100:.2f}% change"
          f"   mean luma {means[best[1]]:.2f} vs {means[best[2]]:.2f}")

    # -- R2 LEAVE-ONE-OUT ------------------------------------------------------
    lags = list(range(1, 7))
    dY = [statistics.fmean([abs(means[i] - means[i - L]) for i in range(L, nf)])
          for L in lags]
    chg = [statistics.fmean([rate(i - L, i) for i in range(L, nf)]) for L in lags]

    full = pearson(dY, chg)
    print(f"\nR2  correlation with all lags: {full:+.3f}")
    loo = {}
    for drop in lags:
        keep = [k for k in lags if k != drop]
        v = pearson([dY[k - 1] for k in keep], [chg[k - 1] for k in keep])
        loo[f"drop_lag_{drop}"] = v
        print(f"     drop lag {drop}: {v:+.3f}")

    loo_vals = [v for v in loo.values() if v is not None]
    loo_min, loo_max = min(loo_vals), max(loo_vals)

    duplicate_suspected = (best[0] < 0.10) and (worst_r < 0.5 * typical_r)
    fragile = (loo_max - loo_min) > 0.8

    report = {
        "question": ("is the +0.702 exposure/delta correlation a real mechanism "
                     "or one leverage point?"),
        "capture": a.capture,
        "n_frames": nf,
        "lag_profile": {str(L): {"mean_abs_dY": dY[L - 1] * 100,
                                "cell_change_pct": chg[L - 1] * 100}
                        for L in lags},
        "R1_lowest_lag5_pair": {"frames": [wi, wj],
                                "change_pct": worst_r * 100,
                                "mean_luma_delta": abs(means[wi] - means[wj]),
                                "median_lag5_change_pct": typical_r * 100},
        "R3_global_min_pair": {"frames": [best[1], best[2]],
                               "change_pct": best[0] * 100},
        "R2_correlation_all_lags": full,
        "R2_leave_one_out": loo,
        "R2_spread": loo_max - loo_min,
        "duplicate_frame_suspected": bool(duplicate_suspected),
        "correlation_is_fragile": bool(fragile),
        "verdict": (
            "RETRACTED. The exposure verdict was carried by one row (lag 5). "
            "Removing any single lag swings the correlation across "
            f"{loo_min:+.3f} to {loo_max:+.3f}, so it is not a mechanism."
            if fragile else
            "The correlation survives leave-one-out and is not a leverage point."),
        "duplicate_note": (
            f"Most similar pair anywhere is frames {best[1]} and {best[2]} at "
            f"{best[0] * 100:.2f}% change. The capture contains a repeat, so the "
            "lag profile -- and anything computed from it -- is contaminated."
            if duplicate_suspected else
            "No near-duplicate frame pair found; the lag-5 dip is not a repeat."),
        "consequence": (
            "Neither auto-exposure nor threshold dither is established as the "
            "cause. The honest state is that the 55-58% baseline cell-change "
            "rate on a stationary subject is UNEXPLAINED, and the resolution "
            "ladder is saturated because of it. Before any motion claim, the "
            "capture must record a per-frame checksum so repeats are caught, "
            "and the ladder budget must be set from measured, not assumed, "
            "change rates."
        ),
    }
    p = Path(a.out) / "ascii_motion_robustness_results.json"
    p.write_text(json.dumps(report, indent=2))
    print(f"\nVERDICT: {report['verdict']}")
    print(f"NOTE:    {report['duplicate_note']}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()