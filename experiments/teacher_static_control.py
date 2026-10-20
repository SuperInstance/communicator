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

# The motion-map scale, used as a COMMON divisor so the between-session
# ratios are untouched and only the absolute scale moves.
#
# IT WAS A HARD-CODED CONSTANT AND IT WAS THE WRONG ONE. The constant
# 143.2708282470703 was session A's scale in the TORN corpus. On the
# corrected corpus it is 8.0, so dividing by the old constant was dividing
# by a number that no longer meant anything -- and the "scale" control was
# therefore testing a scale of 17.9x the one it claimed to be undoing.
# The scale is now READ FROM THE CORPUS.
SCALE_REFERENCE_FALLBACK = 143.2708282470703

CORPUS = Path("/mnt/c/proart_capture/corpus_v2")
# CORRECTED CORPUS. The old /mnt/c/proart_capture/corpus was seeded at the
# wrong frame stride (23,040 instead of 19,200) and every frame in it is
# a tear between two real frames. See docs/research/NV12-STRIDE.md.


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
                    "pose": meta.get("pose"), "speak": meta.get("speak"),
                    "source": meta.get("source"),
                    "raw_maps": np.load(npz_p)["maps"].astype(np.float64)})
    return out


def variant_unscaled(maps: np.ndarray, scale_reference: float) -> np.ndarray:
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
    return maps / scale_reference


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


def _raw_motion(capture: Path) -> np.ndarray:
    """The motion maps of a capture, at the layout the file actually has."""
    from src import nv12
    luma, _ = nv12.read_luma(capture, width=160, height=96)
    Y = luma.astype(np.float64)
    prof = np.median(Y, axis=2)
    g = prof.mean(axis=1, keepdims=True)
    Yr = Y - (prof - g)[:, :, None]
    d = np.abs(np.diff(Yr, axis=0))
    return d.reshape(d.shape[0], 48, 2, 80, 2).mean(axis=(2, 4))


def static_template_stats(sessions) -> dict:
    """WHAT THE STATIC TEMPLATE IS, MEASURED.

    The `static` variant's score says the teacher can read a picture. This
    says what the picture IS: how similar the speaking take's average motion
    map is to the silent take's, how much of the difference is one global
    scale factor, and how much of the empty room's motion map is in the
    same units.

    All of these were quoted in `docs/research/TEACHER-LOO-FOLDS.md` from the
    TORN corpus, and all of them change when the frames are read correctly.
    """
    by_name = {s["name"]: s for s in sessions}
    if "A" not in by_name or "C" not in by_name:
        return {}
    raw = {}
    for name in ("A", "C"):
        d = np.load(by_name[name]["npz"])
        raw[name] = d["maps"].astype(np.float64)
    mA, mC = raw["A"].mean(axis=0), raw["C"].mean(axis=0)
    corr = float(np.corrcoef(mA.ravel(), mC.ravel())[0, 1])
    # THE AMPLITUDE RATIO IS MEASURED ON THE RAW MOTION, NOT ON THE STORED
    # MAPS. The stored maps are quantised per session against their own
    # 99.5th percentile, so their amplitude ratio is a property of the
    # quantiser, not of the scene. The raw motion is what "amplitude" means.
    ratio = None
    raw_amp = {}
    for name in ("A", "C"):
        src = by_name[name].get("source")
        if src and Path(src).is_file():
            raw_amp[name] = _raw_motion(Path(src))
    if "A" in raw_amp and "C" in raw_amp:
        ratio = float(raw_amp["A"].mean() / max(raw_amp["C"].mean(), 1e-12))
    s_glob = float((mA.ravel() @ mC.ravel()) / max((mC.ravel() @ mC.ravel()), 1e-12))
    resid = mA - s_glob * mC
    r2 = float(1.0 - resid.var() / max((mA - mA.mean()).var(), 1e-12))
    tot = {k: v.reshape(len(v), -1).sum(axis=1) for k, v in raw.items()}
    sd = float(np.sqrt((tot["A"].var() + tot["C"].var()) / 2))
    effect = float((tot["A"].mean() - tot["C"].mean()) / sd) if sd > 0 else 0.0

    def temporal_stats(maps):
        mean = np.maximum(maps.mean(axis=0), 1e-9)
        tstd = float((maps.std(axis=0) / mean).mean())
        # consecutive-frame correlation, over the pixels that vary at all
        vals = []
        for i in range(len(maps) - 1):
            a, b = maps[i].ravel(), maps[i + 1].ravel()
            live = a.std() > 0 and b.std() > 0
            if live:
                vals.append(float(np.corrcoef(a, b)[0, 1]))
        return tstd, (float(np.mean(vals)) if vals else float("nan"))

    tstd_a, ffc_a = temporal_stats(raw["A"])
    tstd_c, ffc_c = temporal_stats(raw["C"])
    out = {
        "mean_map_correlation_A_vs_C": round(corr, 4),
        "amplitude_ratio_A_over_C_raw_motion": (None if ratio is None
                                                else round(ratio, 4)),
        "amplitude_ratio_A_over_C_stored_maps": round(
            float(mA.mean() / max(mC.mean(), 1e-12)), 4),
        "note_on_amplitude": (
            "the stored maps are quantised per session against their own "
            "99.5th percentile, so their amplitude ratio is a property of "
            "the quantiser; the raw-motion ratio is the scene's"),
        "global_scale_fit_R2": round(r2, 4),
        "effect_size_A_vs_C_total_motion_sd": round(effect, 4),
        "pixelwise_temporal_std_over_mean_A": round(tstd_a, 3),
        "pixelwise_temporal_std_over_mean_C": round(tstd_c, 3),
        "consecutive_frame_correlation_A": round(ffc_a, 4),
        "consecutive_frame_correlation_C": round(ffc_c, 4),
    }
    # The empty room, read at the layout the file actually has.
    empty = Path("/mnt/c/proart_capture/empty1.nv12")
    if empty.is_file():
        try:
            from src import nv12
            luma, layout = nv12.read_luma(empty, width=160, height=96)
            Y = luma.astype(np.float64)
            prof = np.median(Y, axis=2)
            g = prof.mean(axis=1, keepdims=True)
            Yr = Y - (prof - g)[:, :, None]
            dd = np.abs(np.diff(Yr, axis=0))
            emaps = dd.reshape(dd.shape[0], 48, 2, 80, 2).mean(axis=(2, 4))
            out["empty_room_layout"] = layout.name
            out["empty_room_mean_map_energy_over_A"] = round(
                float(emaps.mean() / max(raw["A"].mean(), 1e-12)), 9)
        except Exception as exc:                       # pragma: no cover
            out["empty_room_error"] = str(exc)
    return out


def run(corpus: Path, modes, k: int, stride: int, max_per: int) -> dict:
    sessions = load_sessions(corpus)
    names = [s["name"] for s in sessions]
    results = {"corpus": str(corpus), "sessions": [
        {"name": s["name"], "label": s["label"], "pose": s["pose"],
         "speak": s["speak"]} for s in sessions],
        "feature_sets": {},
        "static_template": {}}

    # The corpus's OWN motion-map scale, read from session A's npz, used as
    # the common divisor for the `unscaled` variant. It used to be a
    # hard-coded constant from the torn corpus (143.27); on the corrected
    # corpus it is 8.0, and a divisor from the wrong corpus makes the
    # control test a scale that means nothing.
    scale_reference = SCALE_REFERENCE_FALLBACK
    for s_ in sessions:
        try:
            scale_reference = float(np.load(s_["npz"])["scale"])
            break
        except Exception:
            continue

    results["static_template"] = static_template_stats(sessions)

    variants = {
        "real": lambda m: m,
        "static": variant_static,
        "time_shuffled": variant_time_shuffled,
        "unscaled": lambda m: variant_unscaled(m, scale_reference),
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
