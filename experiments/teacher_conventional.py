"""
teacher_conventional.py -- the teacher should probably not be an LLM.

WHY THIS FILE IS THE ANSWER TO CASEY'S QUESTION.

"Could we simply record me and use a traditional model to analyze and be the
teacher for the system?"

Yes. And this file is the evidence for the second half of that sentence, because
the LLM version of the teacher was measured and it does not work.

WHAT WAS MEASURED FIRST, so this is not a preference.

`teacher_bench.py` on the A/C speaking task, with the ASCII motion grid as the
input:

  qwen3:0.6b    real 0.333  shuffled 0.333  blank 0.333   margin 0.0
  llama3.2:1b   real 0.333  shuffled 0.333  blank 0.333   margin 0.0
  gemma2:2b     real 0.333  shuffled 0.333  blank 0.333   margin 0.0
  qwen2.5-coder:7b   refuses the input
  tev1:latest   HTTP 400 (rejects the request shape)

Every model answers CONSTANTLY -- the same answer to the real input, the
shuffled input and the blank. The margin over the best control is exactly
zero, which is the bench's definition of a model that knows nothing. The
models cannot parse a dense character-art grid as a picture. (One real bug was
found and fixed on the way: qwen3 is a reasoning model and its entire token
budget was going into a `thinking` preamble, so every response came back
empty. That bug looked exactly like a constant-answering model in the summary
table, which is why the raw text is kept.)

So the teacher should be a CONVENTIONAL model over the motion maps, which is
what "traditional model" means in Casey's question. No LLM is required, and
none is used here.

THE MODEL.

Logistic regression, implemented in numpy because the repo is deliberately
lean and sklearn is not a dependency. Features are computed from the motion
field itself, and the important ones are TEMPORAL:

  per frame   total change, active fraction, centroid x and y, spread
  per window  the mean and std of each, the centroid VELOCITY (the change in
              centroid between consecutive frames), and the max of each

The centroid velocity is what makes this a motion model rather than a snapshot
model: a person who is speaking has a centroid that moves within a small
region frame to frame, and that motion is invisible in any single frame.
`ROW-BAND-REMOVAL.md` measured that the speaking/silent difference lives in
exactly that place -- 79% of the excess concentrated in the top 5% of pixels.

THE EVALUATION, AND WHY IT IS SPLIT BY TIME.

Three sessions, one per condition, one speaker, one room. That is not a
dataset; it is a PIPELINE VALIDATION. Two things make the number meaningful
anyway:

  SPLIT BY TIME. Train on the first 60% of every session, test on the last
    40%. A model that memorises a session's appearance would score badly on
    the test portion; only a model reading motion generalises. This is the
    only honest split with this little data, and it is still weak.
  THE SHUFFLED CONTROL. The same model, the same features, but the frames
    within each window permuted -- so the motion is destroyed and the spatial
    summary survives. If the model scores the same on shuffled input, it was
    reading a spatial snapshot and knows nothing about motion.

  THE BLANK CONTROL. Zero motion everywhere. Should score at the base rate.

WHAT IS NOT CLAIMED.

Two sessions, one speaker, no error bars, one lighting state, and a temporal
split of a 20-second window. This demonstrates that the pipeline works and
that the controls are wired up. It is NOT evidence that the task is solved,
and the performance number here should be treated as a smoke test that will
move when there is a real corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from ascii_motion_render import load_session  # noqa: E402


# -- features -----------------------------------------------------------------
def frame_feats(m: np.ndarray) -> np.ndarray:
    """Per-frame summary of a motion map. Returns 5 numbers."""
    tot = float(m.sum())
    if tot <= 0:
        return np.array([0.0, 0.0, 0.5, 0.5, 0.0])
    ys, xs = np.nonzero(m > 0.5 * m.max())
    if len(ys) == 0:
        return np.array([tot / 1e4, 0.0, 0.5, 0.5, 0.0])
    cy = float(ys.mean()) / m.shape[0]
    cx = float(xs.mean()) / m.shape[1]
    spread = float((ys.max() - ys.min() + 1) / m.shape[0])
    return np.array([tot / 1e4, len(ys) / m.size, cx, cy, spread])


def window_feats(maps: np.ndarray, mode: str = "summary") -> np.ndarray:
    """A window of consecutive motion frames -> one feature vector.

    mode "summary" -- per-frame totals, centroid, spread, and their temporal
    derivatives. Compact, but it CANNOT see a localised change: `ROW-BAND-
    REMOVAL.md` measured that the speaking/silent difference is 79%
    concentrated in the top 5% of pixels, and a total or a centroid is
    deliberately blind to where a change happened. Measured consequence: with
    these features the model scored 1.000 on a temporal split but 0.000 on
    leave-one-session-out, i.e. it was learning each session's appearance.

    mode "map" -- the window's MEAN motion map, downsampled to a coarse grid,
    concatenated with the temporal summary. This puts the spatial pattern of
    the change directly in the feature vector, which is the only way a linear
    model can express "the mouth moved".

    mode "both" -- summary and map together.
    """
    ff = np.stack([frame_feats(m) for m in maps])          # (k, 5)
    d = np.diff(ff, axis=0) if len(ff) > 1 else np.zeros((1, 5))
    vel = np.abs(d).mean(axis=0)                            # (5,)
    summ = np.concatenate([ff.mean(axis=0), ff.std(axis=0),
                           ff.max(axis=0), vel])
    if mode == "summary":
        return summ

    mean_map = maps.mean(axis=0)
    grid = (mean_map.reshape(6, mean_map.shape[0] // 6, 8,
                             mean_map.shape[1] // 8)
                  .mean(axis=(1, 3))).ravel()               # 48 numbers
    if mode == "map":
        return np.concatenate([grid, vel])
    return np.concatenate([summ, grid])


# -- logistic regression, numpy only ------------------------------------------
def fit_logistic(X, y, l2=1.0, iters=400, lr=0.5):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = Xb.T @ (p - y) / n
        g[:-1] += l2 * w[:-1] / n
        w -= lr * g
    return w


def predict(X, w):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    z = Xb @ w
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return (p >= 0.5).astype(int), p


def build(sessions, k, stride, max_per, shuffle_seed=None, mode="summary",
          spatial_scramble=False):
    """(X, y, groups) for each session's windows.

    `spatial_scramble` destroys the SPATIAL PATTERN of every frame while
    preserving its total change, by rolling each frame's pixels. This is the
    control that a spatial feature set needs, and it is different from
    permuting frames: the mean motion map is INVARIANT to frame order, so a
    frame-permutation control is vacuous for the "map" features -- it scored
    0.979 against a real-input 1.000 and told us nothing. Rolling the pixels
    keeps every frame's energy and destroys where it was.
    """
    Xs, ys, gs = [], [], []
    for name, maps, label, _npz, _meta in sessions:
        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed)
            idx = rng.permutation(len(maps))
            maps = maps[idx]
        if spatial_scramble:
            maps = np.stack([np.roll(m, shift=13 + 7 * (i % 5), axis=1)
                             for i, m in enumerate(maps)])
        starts = list(range(0, len(maps) - k + 1, stride))
        if not starts:
            continue
        if len(starts) > max_per:
            sel = np.linspace(0, len(starts) - 1, max_per).astype(int)
            starts = [starts[i] for i in sel]
        for s in starts:
            Xs.append(window_feats(maps[s:s + k], mode))
            ys.append(label)
            gs.append(name)
    return np.stack(Xs), np.array(ys), gs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/mnt/c/proart_capture/corpus")
    ap.add_argument("--task", default="speaking", choices=("speaking", "facing"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-per", type=int, default=40)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--features", default="both",
                    choices=("summary", "map", "both"))
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    corpus = Path(a.corpus)
    sessions = []
    for meta_p in sorted(corpus.glob("*.json")):
        npz_p = meta_p.with_suffix(".npz")
        if not npz_p.is_file():
            continue
        meta = json.loads(meta_p.read_text())
        if meta.get("kind") != "condition":
            continue
        if a.task == "speaking":
            lab = 1 if meta.get("speak") else 0
        else:
            lab = 1 if meta.get("pose") == "facing" else 0
        sessions.append((meta["condition"], load_session(npz_p), lab,
                         npz_p, meta))
    if len(sessions) < 2:
        sys.exit("need at least two condition sessions")

    X, y, groups = build(sessions, a.k, a.stride, a.max_per, mode=a.features)
    names = [s[0] for s in sessions]
    print(f"task={a.task}  windows={len(X)}  features={X.shape[1]}  "
          f"sessions={names}")
    print(f"  labels: " + ", ".join(f"{n}={s[2]}" for n, s in zip(names, sessions)))

    # -- time split: first train_frac of each session trains, the rest tests --
    tr, te = [], []
    for name in names:
        idx = [i for i, g in enumerate(groups) if g == name]
        cut = int(len(idx) * a.train_frac)
        tr += idx[:cut]
        te += idx[cut:]
    if not te or len(set(y[tr])) < 2:
        sys.exit("degenerate split")

    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-9
    Z = (X - mu) / sd

    w = fit_logistic(Z[tr], y[tr].astype(float))

    def acc(idx, Xs=X):
        pred, _ = predict(Z[idx], w)
        return float((pred == y[idx]).mean())

    res = {
        "task": a.task,
        "k_frames": a.k,
        "n_windows": int(len(X)),
        "train_windows": len(tr),
        "test_windows": len(te),
        "classes_in_train": sorted(set(int(v) for v in y[tr])),
        "acc_train": acc(tr),
        "acc_test": acc(te),
        "base_rate_test": float(max(y[te].mean(), 1 - y[te].mean())),
    }

    # -- CONTROL 1: shuffled frames ------------------------------------------
    Xs, ys, gs = build(sessions, a.k, a.stride, a.max_per, shuffle_seed=7,
                       mode=a.features, spatial_scramble=True)
    mu2, sd2 = Xs[tr].mean(axis=0), Xs[tr].std(axis=0) + 1e-9
    Zs = (Xs - mu2) / sd2
    w2 = fit_logistic(Zs[tr], ys[tr].astype(float))
    pred, _ = predict(Zs[te], w2)
    res["acc_test_spatially_scrambled"] = float((pred == ys[te]).mean())
    res["acc_test_spatially_scrambled_train"] = float(
        (predict(Zs[tr], w2)[0] == ys[tr]).mean())

    # -- CONTROL 2: blank ----------------------------------------------------
    Zb = np.zeros_like(Z)
    pred, _ = predict(Zb[te], w)
    res["acc_test_blank"] = float((pred == y[te]).mean())

    # -- CONTROL 3: leave-one-session-out -----------------------------------
    loo = []
    for name in names:
        tr2 = [i for i, g in enumerate(groups) if g != name]
        te2 = [i for i, g in enumerate(groups) if g == name]
        if len(set(y[tr2])) < 2 or not te2:
            continue
        m2, s2 = X[tr2].mean(axis=0), X[tr2].std(axis=0) + 1e-9
        w3 = fit_logistic((X[tr2] - m2) / s2, y[tr2].astype(float))
        p3, _ = predict((X[te2] - m2) / s2, w3)
        loo.append({"held_out": name, "acc": float((p3 == y[te2]).mean()),
                    "n": len(te2)})
    res["leave_one_session_out"] = loo

    margin = res["acc_test"] - max(res["acc_test_spatially_scrambled"],
                                   res["acc_test_blank"])
    res["margin_over_best_control"] = margin
    res["usable"] = bool(margin >= 0.10)

    print(f"\n  train acc            {res['acc_train']:.3f}")
    print(f"  TEST acc             {res['acc_test']:.3f}   "
          f"(base rate {res['base_rate_test']:.3f})")
    print(f"  control: scrambled   {res['acc_test_spatially_scrambled']:.3f}"
          "   (pixels rolled: energy kept, pattern destroyed)")
    print(f"  control: blank       {res['acc_test_blank']:.3f}")
    print(f"  MARGIN over controls {margin:+.3f}  -> "
          f"{'USABLE' if res['usable'] else 'NOT USABLE'}")
    print("  leave-one-session-out: " + ", ".join(
        f"{d['held_out']} {d['acc']:.3f}" for d in loo))

    res["limits"] = [
        "two labelled sessions, one speaker, one room, one lighting state",
        "the temporal split is over a ~20 s window, so train and test are "
        "highly correlated in time",
        "leave-one-session-out is reported but with two sessions it means "
        "training on one class only for a binary task, so it is a sanity "
        "check, not a measurement",
        "no error bars; one number from one split",
        "this validates the PIPELINE and the CONTROLS. It is not evidence "
        "that the task is solved.",
    ]
    res["features"] = a.features
    p = Path(a.out) / f"teacher_conventional_{a.task}_{a.features}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
