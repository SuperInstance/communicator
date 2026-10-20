"""
teacher_static_control.py -- does the teacher need the MOTION, or is it
reading a static picture of the camera?

WHY THIS FILE EXISTS.

`experiments/teacher_conventional.py` reports that the "map" feature set
scores 1.000 on held-out session B of the leave-one-session-out split, and
`docs/research/TEACHER-PIPELINE.md` reports that the map features are "the
only way a linear model can express 'the mouth moved'". The whole teacher
story rests on that.

But the `map` feature set is built from the MEAN motion map over a window of
consecutive frames, and the motion maps carry a large FIXED spatial pattern --
the same vertical bands in every frame, which is the camera's own structure,
not a person. Measured: the pixelwise temporal standard deviation is 64% of
the temporal mean, and the frame-to-frame correlation of consecutive maps is
only 0.35, so a 6-frame mean keeps a substantial part of that fixed
pattern.

So the "map" features could be reading a STATIC IMAGE of the camera and
using the speaking/silent difference in its AMPLITUDE, and the spatial
scramble control would not catch it, because scrambling the pixels destroys
the pattern (and so destroys the session identity the classifier was using).

This file runs the leave-one-session-out split on four variants of the same
motion maps, and the one that matters is `static`:

    real             the motion maps as they are.
    static           every frame replaced by the SESSION'S TEMPORAL MEAN map.
                     There is no temporal information left at all -- no
                     motion, no order, no change. If the teacher still
                     scores 1.000 on this, it is not measuring movement.
    time_shuffled    the frames permuted. The spatial pattern survives, the
                     temporal ORDER does not. This separates "the picture"
                     from "the order of the picture".
    spatial_scrambled the repo's existing control: every frame's pixels
                     rolled by a different amount, which destroys WHERE the
                     change was while keeping its energy.

AND THE CONTROL THAT SETTLES IT: a shuffled-TIME version of the speech
labels. If the teacher is reading a static picture, then permuting the
frames must not change the answer -- and if it does change the answer, the
motion was doing the work after all.

WHAT IT CANNOT SHOW.

It cannot show that the teacher is WRONG, only what it is reading. A
teacher that reads a static picture may still work on the session-level
task it was built for; it just is not a speaking detector, and the
difference matters for a peripheral that has to decide "is this person
speaking" every 200 ms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from teacher_conventional import build, fit_logistic, predict  # noqa: E402
from ascii_motion_render import load_session                # noqa: E402

# Session A's motion-map scale, used as a COMMON divisor so the
# between-session ratios are untouched and only the absolute scale moves.
SCALE_REFERENCE = 143.2708282470703

CORPUS = Path("/mnt/c/proart_capture/corpus")


def load_sessions(corpus: Path):
    out = []
    for meta_p in sorted(corpus.glob("*.json")):
        npz_p = meta_p.with_suffix(".npz")
        if not npz_p.is_file():
            continue
        meta = json.loads(meta_p.read_text())
        if meta.get("kind") != "condition":
            continue
        label = 1 if meta.get("speak") else 0
        out.append({"name": meta["condition"], "maps": load_session(npz_p),
                    "label": label, "npz": str(npz_p),
                    "pose": meta.get("pose"), "speak": meta.get("speak")})
    return out


def variant_unscaled(maps: np.ndarray) -> np.ndarray:
    """The SAME maps, shrunk by a common factor.

    `ascii_motion_render.load_session` returns `maps * scale`, where `scale`
    is a per-session display constant (143.27 for A, 140.05 for C) that
    recovers the original count scale of the motion. Nothing about the
    TEACHER needs that scale -- the spatial features are ratios, and the
    `total` feature is divided by 1e4 either way -- but the logistic
    regression is NOT scale-invariant: its regularisation is `l2 * w / n`
    against a fixed learning rate, so multiplying a feature by 143 changes
    the effective step size for that feature by 143.

    Dividing EVERY session by the same constant keeps the between-session
    ratios intact and changes only the absolute scale. So this variant is a
    control on the MODEL, not on the data: if the leave-one-session-out
    result changes when a display constant is removed, the result is a
    property of the optimiser's scale and not of the signal.
    """
    return maps / SCALE_REFERENCE


def variant_static(maps: np.ndarray) -> np.ndarray:
    """Every frame replaced by the session's temporal mean map."""
    mean = maps.astype(np.float64).mean(axis=0)
    return np.repeat(mean[None, :, :], len(maps), axis=0).astype(maps.dtype)


def variant_time_shuffled(maps: np.ndarray, seed: int = 11) -> np.ndarray:
    """The frames permuted: the picture survives, the order does not."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(maps))
    return maps[idx]


def fit_and_score(X, y, tr, te):
    m = X[tr].mean(axis=0)
    s = X[tr].std(axis=0) + 1e-9
    w = fit_logistic((X[tr] - m) / s, y[tr].astype(float))
    p, _ = predict((X[te] - m) / s, w)
    return float((p == y[te]).mean()), int(len(te))


def run(corpus: Path, modes, k: int, stride: int, max_per: int) -> dict:
    sessions = load_sessions(corpus)
    names = [s["name"] for s in sessions]
    results = {"corpus": str(corpus), "sessions": [
        {"name": s["name"], "label": s["label"], "pose": s["pose"],
         "speak": s["speak"]} for s in sessions],
        "feature_sets": {}}

    variants = {
        "real": lambda m: m,
        "static": variant_static,
        "time_shuffled": variant_time_shuffled,
        "unscaled": variant_unscaled,
    }

    for mode in modes:
        rows = []
        for vname, vfn in variants.items():
            # Build the variant of EVERY session, then hand the whole set to
            # `build` once. Doing it per session and concatenating is WRONG:
            # the list comprehension inside the loop shadows the loop
            # variable, so every session ends up with the LAST session's
            # maps, and the split quietly stops being a split. (It was caught
            # because the numbers came out different from the teacher's own
            # leave-one-session-out table, which is the only reason to run
            # this at all.)
            varied = []
            for s in sessions:
                varied.append((s["name"], vfn(s["maps"]), s["label"],
                               s["npz"], s))
            X, y, gs = build(varied, k, stride, max_per, mode=mode)
            for name in names:
                tr = [i for i, gg in enumerate(gs) if gg != name]
                te = [i for i, gg in enumerate(gs) if gg == name]
                if not te or len(set(y[tr])) < 2:
                    rows.append({"variant": vname, "held_out": name,
                                 "acc": None,
                                 "note": "degenerate fold: one class in training"})
                    continue
                acc, n = fit_and_score(X, y, tr, te)
                rows.append({"variant": vname, "held_out": name, "acc": acc,
                             "n": n, "held_out_label": int(y[te][0])})
        results["feature_sets"][mode] = {"rows": rows}

    # --- the headline: does STATIC score as well as REAL? -----------------
    headline = {}
    for mode, blob in results["feature_sets"].items():
        by = {}
        for row in blob["rows"]:
            by.setdefault(row["variant"], {})[row["held_out"]] = row["acc"]
        real = by.get("real", {})
        static = by.get("static", {})
        same = {k: (real.get(k) is not None and static.get(k) is not None
                    and abs(real[k] - static[k]) < 1e-9) for k in real}
        headline[mode] = {
            "real": real,
            "static": static,
            "time_shuffled": by.get("time_shuffled"),
            "the_teacher_reads_a_static_picture": (
                None if not same else all(same.values())),
            "reading": ("if `static` scores the same as `real` on every fold, "
                        "the temporal information contributes nothing and the "
                        "feature set is reading a PICTURE of the camera"),
        }
    results["headline"] = headline
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--modes", default="summary,map,both,region,region+map")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-per", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "experiments" /
                                         "teacher_static_control_results.json"))
    args = ap.parse_args(argv)
    res = run(Path(args.corpus),
              [m for m in args.modes.split(",") if m],
              args.k, args.stride, args.max_per)
    out = Path(args.out)
    out.write_text(json.dumps(res, indent=2) + "\n")
    for mode, blob in res["feature_sets"].items():
        print(f"--- {mode} ---")
        for row in blob["rows"]:
            if row.get("acc") is None:
                print(f"  {row['variant']:16s} held out {row['held_out']}: "
                      f"{row.get('note')}")
            else:
                print(f"  {row['variant']:16s} held out {row['held_out']}: "
                      f"{row['acc']:.3f}  (n={row['n']})")
    print(json.dumps(res["headline"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
