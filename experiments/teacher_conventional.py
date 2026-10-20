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

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from ascii_motion_render import load_session  # noqa: E402


# -- provenance ---------------------------------------------------------------
# A number without its inputs cannot be traced. The `both` artefact committed
# at HEAD carried a leave-one-session-out accuracy of 0.675 that no corpus on
# disk reproduces: the corpus that produced it is gone, and the file did not
# record WHICH corpus that was. So every result this script writes records
# the identity of its inputs -- the corpus, each session file and its
# SHA-256, the script's own SHA-256, and the exact command line -- and says
# so in one boolean a reader can check.
#
# `verify_provenance` re-checks a persisted result against what is on disk
# NOW. A result that cannot be verified is a bare number, and says so.

def sha256_file(path) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def session_fingerprint(npz_path) -> dict:
    """The identity of one corpus session: where it is and what it is."""
    p = Path(npz_path).resolve()
    out = {"npz_path": str(p)}
    if not p.is_file():
        out["problem"] = "the npz file is not on disk"
        return out
    out["npz_sha256"] = sha256_file(p)
    try:
        with np.load(p) as d:
            maps = d["maps"]
            out["n_frames"] = int(maps.shape[0])
            out["map_shape"] = [int(v) for v in maps.shape[1:]]
            out["scale"] = float(d["scale"])
    except Exception as exc:                     # noqa: BLE001
        out["problem"] = f"could not read the npz: {exc}"
    return out


def corpus_fingerprint(session_ids) -> str:
    """One hash over the whole loaded corpus, order-independent."""
    h = hashlib.sha256()
    for s in sorted(session_ids, key=lambda s: s.get("npz_path", "")):
        h.update(s.get("npz_path", "").encode())
        h.update(b"\0")
        h.update((s.get("npz_sha256") or "").encode())
        h.update(b"\n")
    return h.hexdigest()


def build_provenance(corpus_path, session_ids, cli_args, parameters) -> dict:
    """The provenance block, and whether it is complete."""
    script = Path(__file__).resolve()
    prov = {
        "generated_by": "experiments/teacher_conventional.py",
        "script_path": str(script),
        "script_sha256": sha256_file(script) if script.is_file() else None,
        "corpus_path": str(Path(corpus_path).resolve()),
        "cli_args": list(cli_args),
        "command_line": " ".join(str(x) for x in cli_args),
        "parameters": dict(parameters),
        "sessions": list(session_ids),
    }
    prov["corpus_fingerprint"] = corpus_fingerprint(prov["sessions"])
    prov["problems"] = [f"{s.get('npz_path')}: {s['problem']}"
                         for s in prov["sessions"] if s.get("problem")]
    return prov


def verify_provenance(result) -> dict:
    """
    Re-check a persisted result against what is on disk NOW.

    Returns {"ok": bool, "problems": [str]}. A result with no provenance block
    is not ok: it is a bare number.
    """
    problems = []
    prov = result.get("provenance")
    if not prov:
        return {"ok": False,
                "problems": ["the result carries no provenance block"]}
    for key in ("corpus_path", "script_sha256", "cli_args", "sessions"):
        if not prov.get(key):
            problems.append(f"provenance is missing {key}")
    script = prov.get("script_path")
    if script and Path(script).is_file():
        if sha256_file(script) != prov.get("script_sha256"):
            problems.append("the script has changed since the result was written")
    for s in prov.get("sessions", []):
        path = Path(s.get("npz_path", ""))
        if not path.is_file():
            problems.append(f"{path} is not on disk")
            continue
        if sha256_file(path) != s.get("npz_sha256"):
            problems.append(f"{path} has changed since the result was written")
    if not problems and corpus_fingerprint(prov.get("sessions", [])) != \
            prov.get("corpus_fingerprint"):
        problems.append("the corpus fingerprint does not match the sessions")
    return {"ok": not problems, "problems": problems}


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


def region_feats(maps: np.ndarray, top_frac: float = 0.05) -> np.ndarray:
    """Features that CANNOT be reduced to an energy statistic.

    TEACHER-PIPELINE.md and COMBINED-FILTER.md both ended at the same place:
    the model was reading a GLOBAL ENERGY STATISTIC, because that is what
    survives the spatially-scrambled control. And the global statistic is
    worthless here -- total change is 3128 / 3209 / 3268 for speaking /
    speaking / SILENT.

    So the features have to describe WHERE the change is and whether it stays
    there. Two of them:

      CONCENTRATION -- the fraction of all change that falls inside the most
        active `top_frac` of pixels. A uniform field gives top_frac (0.05); a
        mouth moving gives far more. This is scale-free: it is a shape, not
        an amount, so pixel-rolling cannot change it.

      PERSISTENCE -- the overlap of the active region between consecutive
        frames. Real speech keeps moving in the same small place; a person
        shifting in their chair, or camera noise, does not. This is the
        feature that says "a specific thing is moving", which is the thing
        the whole project is trying to detect.
    """
    conc, pers = [], []
    for m in maps:
        tot = float(m.sum())
        if tot <= 0:
            conc.append(0.0)
            # A zeros ARRAY, not the scalar 0.0. The scalar used to be
            # appended here, and the persistence loop below does
            # `pers[i] & pers[i+1]` -- which raises TypeError on a float.
            # So the BLANK control (zero motion everywhere) could not be
            # computed at all: the feature extractor crashed on it. The
            # committed results are unaffected, because no committed run
            # passes an all-zero motion field through this function.
            pers.append(np.zeros(m.shape, bool))
            continue
        k = max(1, int(m.size * top_frac))
        idx = np.argpartition(m.ravel(), -k)[-k:]
        conc.append(float(m.ravel()[idx].sum() / tot))
        act = np.zeros(m.size, bool)
        act[idx] = True
        pers.append(act.reshape(m.shape))
    conc = np.array(conc)
    if len(pers) > 1:
        ov = [float((pers[i] & pers[i + 1]).sum() / max(pers[i].sum(), 1))
              for i in range(len(pers) - 1)]
        persistence = float(np.mean(ov)) if ov else 0.0
    else:
        persistence = 0.0
    return np.array([conc.mean(), conc.std(), conc.max(), persistence])


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
    reg = region_feats(maps)
    if mode == "map":
        return np.concatenate([grid, vel])
    if mode == "region":
        return np.concatenate([reg, vel])
    if mode == "region+map":
        return np.concatenate([reg, grid, vel])
    return np.concatenate([summ, grid, reg])


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


def leave_one_session_out(X, y, groups, names, label_of=None,
                           corpus_fingerprint=None):
    """
    Leave-one-session-out, one row per session that CAN be held out.

    A fold that leaves a single class in the training set is not a fold -- it
    is a degenerate fit -- so it is reported in `skipped`, with the reason,
    instead of being quietly dropped. That is the whole point of the change:
    a table cell can only be transcribed from a row that exists, and the
    sessions with NO row are named, so "A 0.000, C 0.650" cannot be written
    for a session C that has no row.

    Returns (rows, skipped).
    """
    loo = []
    skipped = []
    for name in names:
        tr2 = [i for i, g in enumerate(groups) if g != name]
        te2 = [i for i, g in enumerate(groups) if g == name]
        if not te2:
            skipped.append({"held_out": name,
                            "reason": "this session contributed no windows"})
            continue
        if len(set(y[tr2])) < 2:
            left = sorted(set(int(v) for v in y[tr2]))
            skipped.append({
                "held_out": name,
                "reason": (f"holding out this session leaves a single class in "
                           f"the training set (labels {left}), so the fold is "
                           "not defined"),
            })
            continue
        m2, s2 = X[tr2].mean(axis=0), X[tr2].std(axis=0) + 1e-9
        w3 = fit_logistic((X[tr2] - m2) / s2, y[tr2].astype(float))
        p3, _ = predict((X[te2] - m2) / s2, w3)
        row = {"held_out": name, "acc": float((p3 == y[te2]).mean()),
               "n": len(te2)}
        if label_of is not None and name in label_of:
            row["held_out_label"] = int(label_of[name])
        if corpus_fingerprint is not None:
            # so a row cannot be attributed to the wrong corpus
            row["corpus_fingerprint"] = corpus_fingerprint
        loo.append(row)
    return loo, skipped


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/mnt/c/proart_capture/corpus_v2")
    # CORRECTED CORPUS. The old /mnt/c/proart_capture/corpus was seeded
# at the wrong frame stride (23,040 instead of 19,200) and every frame
# in it is a tear between two real frames. See
# docs/research/NV12-STRIDE.md.
    ap.add_argument("--task", default="speaking", choices=("speaking", "facing"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-per", type=int, default=40)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--features", default="both",
                    choices=("summary", "map", "both", "region",
                             "region+map"))
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args(argv)

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

    # -- provenance: what this run read, and what produced it ---------------
    used_argv = list(sys.argv[1:]) if argv is None else list(argv)
    session_ids = []
    for name, _maps, label, npz_p, _meta in sessions:
        fid = session_fingerprint(npz_p)
        fid["name"] = name
        fid["label"] = int(label)
        session_ids.append(fid)
    provenance = build_provenance(
        corpus, session_ids, used_argv,
        {"task": a.task, "features": a.features, "k": a.k, "stride": a.stride,
         "max_per": a.max_per, "train_frac": a.train_frac, "out": str(a.out)})
    fingerprint = provenance["corpus_fingerprint"]
    print(f"  corpus {provenance['corpus_path']}  "
          f"fingerprint {fingerprint[:16]}")

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
    label_of = {s[0]: s[2] for s in sessions}
    loo, loo_skipped = leave_one_session_out(X, y, groups, names, label_of,
                                             corpus_fingerprint=fingerprint)
    res["leave_one_session_out"] = loo

    margin = res["acc_test"] - max(res["acc_test_spatially_scrambled"],
                                   res["acc_test_blank"])
    res["margin_over_best_control"] = margin
    res["usable"] = bool(margin >= 0.10)

    # The LOO table, persisted in a shape that cannot be mis-transcribed: one
    # object KEYED BY FEATURE SET, each row carrying the feature set's name,
    # the held-out session, the accuracy and n -- and, at the feature-set
    # level, the margin over the best control and the two control values
    # themselves. The bare list above is kept for anything that reads it.
    res["leave_one_session_out_by_feature_set"] = {
        a.features: {
            "feature_set": a.features,
            "task": a.task,
            "margin_over_best_control": margin,
            "control_spatially_scrambled": res["acc_test_spatially_scrambled"],
            "control_blank": res["acc_test_blank"],
            "best_control": max(res["acc_test_spatially_scrambled"],
                                res["acc_test_blank"]),
            "rows": [dict(r, feature_set=a.features) for r in loo],
            "held_out_sessions": [r["held_out"] for r in loo],
            "skipped_sessions": loo_skipped,
            "n_rows": len(loo),
        }
    }
    # Every session in the run, with its label, so a reader can tell a
    # session that has no LOO row from a session that does not exist.
    res["sessions"] = [{"name": s[0], "label": int(s[2])} for s in sessions]

    # What this run read, and what produced it. A result that cannot say
    # where it came from is a bare number, and `reproducible` says so.
    res["provenance"] = provenance
    res["reproducible"] = bool(
        provenance.get("script_sha256")
        and provenance.get("corpus_fingerprint")
        and provenance.get("cli_args") is not None
        and all(s.get("npz_sha256") for s in provenance.get("sessions", []))
        and not provenance.get("problems"))

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
    for sk in loo_skipped:
        print(f"  leave-one-session-out: {sk['held_out']} -- NO ROW "
              f"({sk['reason']})")

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
