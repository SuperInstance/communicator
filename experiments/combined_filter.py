"""
combined_filter.py -- compose the two filters, then re-ask the teacher question.

WHY COMPOSING IS THE OBVIOUS NEXT STEP AND WAS NOT DONE SOONER.

Two filters were measured separately, and I treated them as alternatives. They
are not:

  ROW-BAND REMOVAL (`row_band_removal.py`) removes the ROW-CONSTANT component.
  The nuisance is horizontal bands, identical in every column, so the per-row
  median across the columns is an estimate of it. It costs NO LATENCY and it
  does not blur motion, because it never combines frames. Empty room
  50.00% -> 0.01%, occupied 48.19% -> 48.57% (unchanged).

  TEMPORAL MEDIAN (`temporal_median_fix.py`) removes the TEMPORALLY RANDOM
  component -- per-pixel jitter that moves frame to frame. Empty room
  50.00% -> 0.00%, occupied 48.19% -> 21.43% at k=3. It costs (k-1)/30 s and
  smears fast motion.

One attacks the spatial structure of the nuisance, the other the temporal
structure. They are orthogonal, and the thing that actually matters -- how much
residual noise the signal sits on -- is what both of them reduce. Composing
them should give row removal's zero latency and zero blur, PLUS the temporal
median's noise rejection.

THE PROBLEM THIS IS SUPPOSED TO SOLVE, stated as a measurement.

`TEACHER-PIPELINE.md` found the reason no teacher could be validated: the
motion field is SATURATED.

    total change  speaking 3127.9 | speaking 3208.7 | SILENT 3268.5
    99.8% of pixels change by more than 1% of maximum, in every frame, in
    every session

So "how much changed" carries no information -- it is the same whether he is
talking or not -- and the localised speech signal that `ROW-BAND-REMOVAL.md`
measured (79% of the excess in the top 5% of pixels) is a small thing on top
of a saturated field. A teacher cannot find a small localised signal in a
saturated one.

If the composition de-saturates the field, the teacher gets a chance. That is
the test: does the speech signal stand out MORE after both filters than after
either alone?

THE MEASUREMENTS, and they are the point, not the filter.

  1. FLIP RATE, empty and occupied, for the composition and for each filter
     alone. The empty rate must stay at zero -- that is the whole point of the
     gate -- and the occupied rate should FALL, because that is noise being
     removed rather than signal.

  2. SATURATION. The fraction of pixels changing by more than 1% of maximum.
     This is the number that was 0.998 and that made the field useless.

  3. SIGNAL-TO-NUISANCE OF THE SPEECH SIGNAL, measured directly: facing and
     speaking against facing and silent, on the SAME representation. If the
     composition helps, this gap GROWS relative to the within-condition split.
     If it shrinks, the filters are eating the signal and the composition is
     wrong.

  4. THE TEACHER, re-run on the composed representation, with the
     leave-one-session-out split that refused to be fooled last time.

WHAT IS NOT CLAIMED.

Two labelled sessions per class. A better representation makes a teacher MORE
LIKELY to pass its gate; it does not make one pass. The leave-one-session-out
number is the one that counts and it is reported whatever it says.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W, H = 160, 96
FB = W * H * 3 // 2
YS = W * H

sys.path.insert(0, str(ROOT / "experiments"))
from ascii_motion_render import load_session  # noqa: E402


def row_band_remove(Y: np.ndarray) -> np.ndarray:
    prof = np.median(Y, axis=2)
    g = prof.mean(axis=1, keepdims=True)
    return Y - (prof - g)[:, :, None]


def temporal_median(F: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return F
    pad = k // 2
    P = np.pad(F, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    stack = np.stack([P[i:i + len(F)] for i in range(k)], axis=0)
    return np.median(stack, axis=0)


def change_maps(Y: np.ndarray) -> np.ndarray:
    d = np.abs(np.diff(Y, axis=0))
    return d.reshape(d.shape[0], H // 2, 2, W // 2, 2).mean(axis=(2, 4))


def load_raw(path: Path, nf: int):
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    n = min(nf, len(raw) // FB)
    Y = raw[:n * FB].reshape(n, FB)[:, :YS].reshape(n, H, W).astype(np.float64)
    return Y


def variant_maps(Y: np.ndarray, row: bool, k: int) -> np.ndarray:
    F = row_band_remove(Y) if row else Y
    F = temporal_median(F, k)
    return change_maps(F)


def saturation(maps: np.ndarray) -> float:
    """Fraction of pixels changing by more than 1% of the map's own maximum."""
    hi = np.percentile(maps, 99.5)
    if hi <= 0:
        return 0.0
    return float((maps > 0.01 * hi).mean())


def speech_gap(maps: np.ndarray, t0: float, t1: float, w: int = 6):
    """The A-vs-C signal measured as a spatial change map, plus the
    within-condition split it has to beat."""
    a = int(t0 * 30)
    b = min(int(t1 * 30), len(maps) - 1)
    if b - a < w * 2:
        return None
    def block(s, e):
        d = np.abs(maps[s + 1:e + 1] - maps[s:e])
        return d.mean(axis=0)
    first = block(a, a + w)
    second = block(b - w, b)
    d = np.abs(first - second)
    excess = np.clip(d - np.percentile(d, 50), 0, None)
    flat = np.sort(excess.ravel())[::-1]
    loc = float(flat[:max(1, len(flat) // 20)].sum() / (flat.sum() + 1e-9))
    return {"mean_abs_diff": float(d.mean()), "max_diff": float(d.max()),
            "top5pct_excess_share": loc}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--empty", default="/home/proart/projects/glasses/.capture_keep/empty1.nv12")
    ap.add_argument("--corpus", default="/mnt/c/proart_capture/corpus")
    ap.add_argument("--nf", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    E = load_raw(Path(a.empty), a.nf)
    corpus = Path(a.corpus)
    sess = {}
    for meta_p in sorted(corpus.glob("*.json")):
        npz = meta_p.with_suffix(".npz")
        if not npz.is_file():
            continue
        meta = json.loads(meta_p.read_text())
        if meta.get("kind") != "condition":
            continue
        # rebuild raw frames from the source capture so the filters see full
        # resolution, not the stored motion maps
        src = meta.get("source")
        if src and Path(src).is_file():
            sess[meta["condition"]] = (load_raw(Path(src), a.nf), meta)
        else:
            sess[meta["condition"]] = (None, meta)

    variants = {
        "raw": (False, 1),
        "row": (True, 1),
        "median3": (False, 3),
        "row+median3": (True, 3),
        "row+median5": (True, 5),
    }

    results = {}
    print(f"{'variant':14s} {'empty flip':>11s} {'empty sat':>10s} "
          f"{'occ flip':>9s} {'occ sat':>8s} {'A-vs-C gap':>12s} {'loc':>6s}")
    for name, (row, k) in variants.items():
        mE = variant_maps(E, row, k)
        row_out = {"empty_flip": None, "empty_saturation": saturation(mE)}
        for cond, (Y, meta) in sess.items():
            if Y is None:
                continue
            m = variant_maps(Y, row, k)
            # flip rate between consecutive motion maps
            if len(m) > 2:
                d = np.abs(np.diff(m, axis=0))
                flip = float((d > 0.5 * d.max()).mean())
            else:
                flip = None
            row_out[f"{cond}_flip"] = flip
            row_out[f"{cond}_saturation"] = saturation(m)
            if cond in ("A", "C"):
                g = speech_gap(m, 0.8, 9.0)
                if g:
                    row_out[f"{cond}_gap"] = g
        # A-vs-C: difference of the two conditions' spatial change maps
        if "A" in sess and "C" in sess and sess["A"][0] is not None:
            mA = variant_maps(sess["A"][0], row, k)
            mC = variant_maps(sess["C"][0], row, k)
            d = np.abs(mA[30:270].mean(axis=0) - mC[30:270].mean(axis=0))
            null = np.abs(mC[30:270].mean(axis=0) - mC[30:270][::3].mean(axis=0))
            excess = np.clip(d - np.percentile(null, 95), 0, None)
            flat = np.sort(excess.ravel())[::-1]
            loc = float(flat[:max(1, len(flat) // 20)].sum() / (flat.sum() + 1e-9))
            # The within-condition split: the same comparison between two
            # halves of ONE session IN TIME. The gap alone cannot be read,
            # because a filter that removes signal and noise together shrinks
            # the gap while improving everything. The RATIO is the number.
            #
            # (The first version of this computed the split by halving the
            # 2-D MAP, i.e. top-of-image against bottom-of-image, which is not
            # a temporal split at all and produced nonsense SNR values. The
            # split has to be in TIME: mean motion map of the first half of
            # the window against the second half.)
            mid = (30 + 270) // 2
            h1 = mA[30:mid].mean(axis=0)
            h2 = mA[mid:270].mean(axis=0)
            within = float(np.abs(h1 - h2).mean())
            row_out["A_vs_C"] = {"mean_abs_diff": float(d.mean()),
                                 "max_diff": float(d.max()),
                                 "within_condition_split": within,
                                 "snr": (float(d.mean()) / within) if within else None,
                                 "top5pct_excess_share": loc}
        results[name] = row_out
        g = row_out.get("A_vs_C", {})
        print(f"{name:14s} {'':>11s} {row_out['empty_saturation']:10.4f} "
              f"{str(row_out.get('A_flip'))[:9]:>9s} "
              f"{row_out.get('A_saturation', 0):8.4f} "
              f"{g.get('mean_abs_diff', 0):12.3f} "
              f"{g.get('snr', 0):6.2f} "
              f"{g.get('top5pct_excess_share', 0) * 100:5.1f}%")

    out = {"question": ("does composing row-band removal with a temporal median "
                        "de-saturate the motion field and make the speech "
                        "signal stand out more?"),
           "variants": results,
           "limits": [
               "two labelled sessions per class; the composition can make a "
               "teacher MORE LIKELY to pass its gate but cannot make one pass",
               "the empty capture is later and darker than the occupied ones",
               "the flip rates here are computed on the motion maps, not on "
               "the Braille field, so they are not directly comparable to the "
               "50.00% / 0.01% figures in the other documents",
           ]}
    p = Path(a.out) / "combined_filter_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
