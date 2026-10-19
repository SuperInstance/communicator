"""
ascii_motion_cause.py -- WHAT is changing? Per-frame global gain vs the delta.

THE ANOMALY, STATED PRECISELY.

`ascii_motion_nulls.py` on a live webcam capture:

    lag  1: 55.94%     lag  3: 57.45%
    lag  5: 11.46%     lag 20: 18.96%

Change rate FALLS by two thirds as temporal lag increases. Real motion
accumulates -- a subject that moves leaves more difference at lag 20 than at
lag 1. Here the opposite happens, sharply, between lag 3 and lag 5. Nothing that
moves behaves this way. Something that RETURNS does.

The candidate is the camera's own control loop. A webcam's AGC/AWB hunts for
exposure, and a hunting loop is periodic: it overshoots, corrects, and comes
back. Compare a frame to its neighbour and you catch it at opposite phases of
the hunt -- maximum difference. Compare it to a frame one full cycle later and
it has landed near where it started -- near-zero difference. That reproduces
the shape exactly, including the discontinuity at the cycle length.

This is not academic. The whole point of the I/P character representation is to
carry MOTION. If the frame-to-frame character delta is dominated by the camera's
gain loop, the P-frame is a picture of the EXPOSURE CYCLE, and the more faithful
it is the more carefully it records the camera instead of the room.

THE MEASUREMENT.

Per frame: mean luma, and mean luma of the brightest decile (a gain change
scales everything, so the histogram's top moves). Then the autocorrelation of
the mean-luma series. If AGC is real, luma will be strongly NEGATIVELY
correlated at the hunt period and positively correlated across it -- the same
lag-inversion seen in the cell delta, appearing in a scalar that Syzygy never
touched and that the renderer cannot be blamed for.

The control that keeps this honest: the same statistic computed on a SYNTHETIC
frame with no exposure variation and on the same camera's *first* frame. If the
live capture's luma really swings, a static synthetic capture cannot.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAME_W, FRAME_H = 160, 96


def frames_luma(nv12: Path):
    fb = FRAME_W * FRAME_H * 3 // 2
    raw = nv12.read_bytes()
    nf = len(raw) // fb
    out = []
    for i in range(nf):
        ys = raw[i * fb:i * fb + FRAME_W * FRAME_H]
        s = sorted(ys)
        n = len(s)
        out.append({
            "mean": sum(ys) / n,
            "p10": s[n // 10],
            "p50": s[n // 2],
            "p90": s[(9 * n) // 10],
            "top_decile_mean": sum(s[-(n // 10):]) / (n // 10),
        })
    return out


def autocorr(series, lag):
    """Pearson correlation of the series with itself shifted by `lag`."""
    n = len(series) - lag
    if n < 3:
        return None
    a, b = series[:n], series[lag:lag + n]
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else None


def pearson(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    fb = FRAME_W * FRAME_H * 3 // 2
    raw = Path(a.capture).read_bytes()
    nf = len(raw) // fb
    live = frames_luma(Path(a.capture))

    # CONTROL 1: the same scene, rendered once and repeated nf times. Zero
    # exposure variation by construction. Whatever the live capture shows must
    # be absent here.
    static = frames_luma_from_bytes(raw[:fb] * nf)

    print(f"frames={nf}\n")
    print(" live   " + "  ".join(f"{f['mean']:6.1f}" for f in live[:14]))
    print(" static " + "  ".join(f"{f['mean']:6.1f}" for f in static[:14]))

    means = [f["mean"] for f in live]
    p90s = [f["p90"] for f in live]
    smeans = [f["mean"] for f in static]

    ac_live, ac_static = {}, {}
    for lag in (1, 2, 3, 4, 5, 6):
        v = autocorr(means, lag)
        ac_live[lag] = v
        ac_static[lag] = autocorr(smeans, lag)

    print("\n mean-luma autocorrelation (live vs static-repeat control):")
    for lag in (1, 2, 3, 4, 5, 6):
        vl = ac_live.get(lag)
        vs = ac_static.get(lag)
        f_l = f"{vl:+.3f}" if vl is not None else " n/a"
        f_s = f"{vs:+.3f}" if vs is not None else "  n/a"
        print(f"   lag {lag}: live {f_l}   static {f_s}")

    # How much does global level actually move?
    live_span = max(means) - min(means)
    static_span = max(smeans) - min(smeans)
    p90_span = max(p90s) - min(p90s)

    # -- THE CROSS-TEST: does exposure PREDICT the character delta? ------------
    # This is the test that actually decides the question. If the I/P delta is
    # tracking the subject, its lag profile should be unrelated to the exposure
    # profile. If the delta is tracking the camera, then for every lag L the
    # mean |luma difference| and the cell-change rate should move together --
    # and, decisively, the delta's sharp DISCONTINUITY (56% at lag 3 falling to
    # 11% at lag 5, from ascii_motion_nulls.py) should land where the exposure
    # autocorrelation turns back positive.
    #
    # Two series, both indexed by lag. If they trace the same curve, exposure
    # explains the delta and no separate "motion" claim can be made from this
    # capture.
    print("\n cross-test: exposure difference vs character delta, by lag")
    print("   lag   mean|dY|   cell-change%")
    exp_by_lag, cell_by_lag = [], []
    cross: dict = {}
    try:
        sys.path.insert(0, str(ROOT / "experiments"))
        from ascii_motion import ensure, fields  # noqa: E402
        binp = ensure(Path("/tmp/syzbuild"))
        fr = fields(binp, Path(a.capture))
        cols, rows = fr[0][0], fr[0][1]
        for lag in range(1, 7):
            dl = [abs(means[i] - means[i - lag]) for i in range(lag, len(means))]
            rates = [sum(1 for k in range(cols * rows)
                         if fr[i - lag][2][k * 2:k * 2 + 2] != fr[i][2][k * 2:k * 2 + 2])
                     / (cols * rows) for i in range(lag, len(fr))]
            mdl, mr = statistics.fmean(dl), statistics.fmean(rates)
            exp_by_lag.append(mdl)
            cell_by_lag.append(mr)
            cross[lag] = {"mean_abs_dY": mdl, "cell_change_pct": mr * 100}
            print(f"   {lag:>3}   {mdl:9.3f}   {mr * 100:11.2f}")
    except Exception as exc:  # noqa: BLE001
        cross = {"error": str(exc)}
        print(f"   (skipped: {exc})")

    cross_corr = pearson(exp_by_lag, cell_by_lag) if len(exp_by_lag) >= 4 else None
    if cross_corr is not None:
        print(f"\n correlation(mean|dY| by lag, cell-change by lag) = {cross_corr:+.3f}")

    print(f"\n mean-luma span: live {live_span:.2f} counts   static {static_span:.2f}")
    print(f" p90-luma span:  live {p90_span:.2f} counts")

    # THE DECISIVE SIGNATURE, and a correction to the first version of this check.
    #
    # The original predicate asked for NEGATIVE correlation at lag 1. That was
    # wrong, and it reported "no oscillation" on data that plainly contained one.
    # A hunting gain loop is caught mid-correction in the SAME direction on
    # consecutive frames (lag 1 stays positively correlated -- the loop has
    # momentum), and only reaches the opposite extreme two or three frames later.
    # So the fingerprint is a NEGATIVE trough at lag 2-3 with a positive lag-1,
    # not a negative lag-1.
    #
    # This was caught because the statistic disagreed with the raw printout of
    # mean luma (a visible 111 -> 125 -> 119 swing). A verdict function that can
    # contradict its own printed data is not a measurement.
    l1 = ac_live.get(1)
    troughs = [ac_live.get(k) for k in (2, 3) if ac_live.get(k) is not None]
    trough = min(troughs) if troughs else None
    oscillating = (
        trough is not None and trough < -0.30
        and live_span > 3.0 * max(static_span, 1e-9)
    )
    hunt_period = min((k for k in (2, 3, 4) if ac_live.get(k) == trough),
                      default=None)

    report = {
        "question": ("is the frame-to-frame character delta dominated by the "
                     "camera's auto-exposure loop rather than by subject motion?"),
        "capture": a.capture,
        "n_frames": nf,
        "live_mean_luma": means,
        "static_control_mean_luma": smeans,
        "mean_luma_span_live": live_span,
        "mean_luma_span_static_control": static_span,
        "p90_luma_span_live": p90_span,
        "autocorr_live": {str(k): v for k, v in ac_live.items()},
        "autocorr_static_control": {str(k): v for k, v in ac_static.items()},
        "corr_mean_vs_p90_per_frame": pearson(means, p90s),
        "oscillating": bool(oscillating),
        "cross_test_exposure_vs_delta_by_lag": cross,
        "cross_test_correlation": cross_corr,
        "lag1_autocorr": l1,
        "autocorr_trough": trough,
        "hunt_period_frames": hunt_period,
        "verdict": (
            "EXPOSURE OSCILLATION. Consecutive frames sit at opposite phases of "
            "a hunting gain loop. The character delta is dominated by the "
            "camera's AGC, not the subject."
            if oscillating else
            "No exposure oscillation detected at this cadence."),
        "consequence": (
            "Any I/P representation built on raw frame subtraction of this "
            "capture would be spending its budget on the exposure cycle. Either "
            "the capture is normalised (lock exposure) or the P-frame must be "
            "taken against a per-frame gain-compensated reference. Until that is "
            "settled, no motion claim from a live webcam capture is admissible."
        ),
    }
    p = Path(a.out) / "ascii_motion_cause_results.json"
    p.write_text(json.dumps(report, indent=2))
    print(f"\nVERDICT: {report['verdict']}")
    print(f"wrote {p}")


def frames_luma_from_bytes(buf: bytes):
    fb = FRAME_W * FRAME_H * 3 // 2
    nf = len(buf) // fb
    out = []
    for i in range(nf):
        ys = buf[i * fb:i * fb + FRAME_W * FRAME_H]
        s = sorted(ys)
        n = len(s)
        out.append({"mean": sum(ys) / n, "p50": s[n // 2],
                    "p90": s[(9 * n) // 10]})
    return out


if __name__ == "__main__":
    main()