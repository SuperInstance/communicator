"""
teacher_loo_control.py -- give the leave-one-session-out fold the control it
never had, and ask whether the fold can identify "speaking" at all.

WHY THIS FILE EXISTS.

`experiments/teacher_conventional.py` computes two controls for its TEMPORAL
split -- a spatially scrambled input and a blank input -- and computes the
leave-one-session-out fold with NO controls at all. The LOO is the split that
the report itself calls "the only split here that can tell 'speaking' from
'this recording'", and it is the one number with nothing beside it.

This file runs the same LOO fold three times -- on the real motion maps, on
the spatially scrambled maps, and on blank maps -- so the fold's held-out
accuracy has its own margin, and it asks a second question that the accuracy
numbers cannot answer on their own.

THE IDENTIFIABILITY QUESTION.

Leave-one-session-out over three sessions cannot always tell "the person is
speaking" from "this recording looks different", and the difference between
the folds is not in the features at all -- it is in the DESIGN.

    fold A: train on {B = turned + speaking, C = facing + silent}
            The two training sessions differ in POSE and in SPEAKING at the
            same time. A model that learns "turned -> speaking" and a model
            that learns "speaking -> class 1" fit the training data
            EQUALLY WELL. The training set contains no pair of examples that
            differ in speaking while holding pose fixed, so the fold cannot
            separate the two explanations -- and the held-out session A
            (facing + speaking) is exactly the case that tells them apart.
            Whatever accuracy this fold reports, it is a statement about the
            training set's design, not about the features.

    fold B: train on {A = facing + speaking, C = facing + silent}
            Both training sessions are FACING, so pose cannot explain the
            labels; only speech can. The fold IS identified.

So the two folds are not two measurements of the same thing, and averaging
them, or quoting the worse one, or treating "LOO A = 0.000" as evidence
that the features cannot detect speech, are all mistakes. Which is why this
file reports the identifiability of each fold next to its accuracy.

WHAT IT DOES NOT DO.

It does not change `experiments/teacher_conventional.py`, and it does not
validate a teacher. It measures the existing pipeline's LOO fold properly
and says what the fold can and cannot support.
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

from teacher_conventional import (            # noqa: E402
    build, fit_logistic, predict,
)


def load_sessions(corpus: Path, task: str = "speaking"):
    """The same session loading the teacher uses, with the pose kept."""
    from ascii_motion_render import load_session
    out = []
    for meta_p in sorted(corpus.glob("*.json")):
        npz_p = meta_p.with_suffix(".npz")
        if not npz_p.is_file():
            continue
        meta = json.loads(meta_p.read_text())
        if meta.get("kind") != "condition":
            continue
        label = 1 if (meta.get("speak") if task == "speaking"
                      else meta.get("pose") == "facing") else 0
        out.append({"name": meta["condition"], "maps": load_session(npz_p),
                    "label": label, "pose": meta.get("pose"),
                    "speak": meta.get("speak"), "npz": str(npz_p)})
    return out


def blank_maps(maps):
    return np.zeros_like(maps)


def fit_and_score(X, y, groups, tr2, te2):
    m2, s2 = X[tr2].mean(axis=0), X[tr2].std(axis=0) + 1e-9
    w = fit_logistic((X[tr2] - m2) / s2, y[tr2].astype(float))
    p, _ = predict((X[te2] - m2) / s2, w)
    return float((p == y[te2]).mean()), int(len(te2))


def identifiability(train_sessions, label_key="speak") -> dict:
    """Can the training set distinguish "speaking" from an alternative?

    The alternative that matters here is POSE. If the training sessions
    differ in pose and in speaking together, the two explanations fit the
    training labels equally well and the fold is not identified.
    """
    poses = {s["pose"] for s in train_sessions}
    labels = {s["label"] for s in train_sessions}
    # Is the label a perfect function of the pose (and vice versa)?
    by_pose = {}
    for s in train_sessions:
        by_pose.setdefault(s["pose"], set()).add(s["label"])
    label_determined_by_pose = all(len(v) == 1 for v in by_pose.values())
    # Is the label a perfect function of the session identity?
    by_label = {}
    for s in train_sessions:
        by_label.setdefault(s["label"], set()).add(s["pose"])
    pose_determined_by_label = all(len(v) == 1 for v in by_label.values())
    confounded = label_determined_by_pose and pose_determined_by_label
    if confounded:
        why = ("the training sessions differ in pose and in speaking at the "
               "same time, so 'speaking' and 'pose' fit the training labels "
               "equally well and the fold cannot separate them")
    elif len(poses) == 1:
        why = ("every training session has the same pose, so pose cannot "
               "explain the labels; only the speaking label can")
    else:
        why = "the training set does not confound pose with speaking"
    return {
        "train_sessions": sorted(s["name"] for s in train_sessions),
        "train_poses": sorted(str(p) for p in poses),
        "train_labels": sorted(int(v) for v in labels),
        "label_determined_by_pose": bool(label_determined_by_pose),
        "pose_determined_by_label": bool(pose_determined_by_label),
        "identified": not confounded,
        "why": why,
    }


def run(corpus: Path, task: str, feature_sets, k: int, stride: int,
        max_per: int) -> dict:
    sessions = load_sessions(corpus, task)
    names = [s["name"] for s in sessions]
    out = {"corpus": str(corpus), "task": task,
           "sessions": [{"name": s["name"], "label": s["label"],
                         "pose": s["pose"], "speak": s["speak"]}
                        for s in sessions],
           "feature_sets": {}}

    # Build the three variants ONCE per feature set: real, scrambled, blank.
    variants = {}
    for mode in feature_sets:
        real = build([(s["name"], s["maps"], s["label"], s["npz"], s)
                      for s in sessions], k, stride, max_per, mode=mode)
        scr = build([(s["name"], s["maps"], s["label"], s["npz"], s)
                     for s in sessions], k, stride, max_per, mode=mode,
                    spatial_scramble=True)
        blank = build([(s["name"], blank_maps(s["maps"]), s["label"],
                        s["npz"], s) for s in sessions],
                      k, stride, max_per, mode=mode)
        variants[mode] = {"real": real, "scrambled": scr, "blank": blank}

    for mode, v in variants.items():
        Xr, yr, gr = v["real"]
        Xs, ys_, gs = v["scrambled"]
        Xb, yb, gb = v["blank"]
        rows = []
        for name in names:
            tr2 = [i for i, g in enumerate(gr) if g != name]
            te2 = [i for i, g in enumerate(gr) if g == name]
            if not te2:
                continue
            if len(set(yr[tr2])) < 2:
                rows.append({"held_out": name, "acc": None,
                             "note": "degenerate fold: one class in training"})
                continue
            acc_r, n = fit_and_score(Xr, yr, gr, tr2, te2)
            acc_s, _ = fit_and_score(Xs, yr, gr, tr2, te2)
            acc_b, _ = fit_and_score(Xb, yr, gr, tr2, te2)
            train_sessions = [s for s in sessions if s["name"] != name]
            ident = identifiability(train_sessions)
            rows.append({
                "held_out": name,
                "held_out_label": int(yr[te2][0]),
                "n": n,
                "acc_real": acc_r,
                "acc_spatially_scrambled": acc_s,
                "acc_blank": acc_b,
                "margin_over_own_scrambled_control": round(
                    acc_r - max(acc_s, acc_b), 4),
                "identifiability": ident,
            })
        out["feature_sets"][mode] = {"rows": rows}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="/mnt/c/proart_capture/corpus_v2")
    # CORRECTED CORPUS. The old /mnt/c/proart_capture/corpus was seeded
# at the wrong frame stride (23,040 instead of 19,200) and every frame
# in it is a tear between two real frames. See
# docs/research/NV12-STRIDE.md.
    ap.add_argument("--task", default="speaking",
                    choices=("speaking", "facing"))
    ap.add_argument("--features", default="summary,map,both,region,region+map")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-per", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "experiments" /
                                         "teacher_loo_control_results.json"))
    args = ap.parse_args(argv)

    res = run(Path(args.corpus), args.task,
              [f for f in args.features.split(",") if f],
              args.k, args.stride, args.max_per)
    out = Path(args.out)
    out.write_text(json.dumps(res, indent=2) + "\n")

    for mode, blob in res["feature_sets"].items():
        print(f"--- {mode} ---")
        for row in blob["rows"]:
            if row.get("acc_real") is None:
                print(f"  held out {row['held_out']}: {row.get('note')}")
                continue
            ident = row["identifiability"]
            print(f"  held out {row['held_out']} (label "
                  f"{row['held_out_label']}): real {row['acc_real']:.3f}  "
                  f"scrambled {row['acc_spatially_scrambled']:.3f}  "
                  f"blank {row['acc_blank']:.3f}  margin "
                  f"{row['margin_over_own_scrambled_control']:+.3f}  "
                  f"identified={ident['identified']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
