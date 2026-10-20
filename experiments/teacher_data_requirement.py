"""
teacher_data_requirement.py -- how many sessions does the teacher actually need?

THE QUESTION THIS ANSWERS, AND WHY IT MATTERS MORE THAN THE ACCURACY.

`TEACHER-PIPELINE.md` and `COMBINED-FILTER.md` both ended with the same
situation: a teacher that scores 1.000 on a temporal split and 0.000 on
leave-one-session-out. The obvious reading is "not enough data", and the
obvious reaction is to record a lot more and hope.

But "record a lot more" is not a plan, because the amount matters and it is
computable. This file computes it, and it re-runs as the corpus grows.

THE MEASUREMENT.

For each feature, the corpus gives us:

  WITHIN-SESSION SPREAD. Windows from one session, which share that session's
    lighting, position and posture. This is irreducible noise for a
    session-level classifier.

  BETWEEN-SESSION SPREAD AMONG THE SAME LABEL. Two sessions of the SAME
    condition, recorded minutes apart. This is the number that matters: it is
    how much a session's identity leaks into the features, and it is what a
    classifier has to see through.

  BETWEEN-LABEL SEPARATION. How far apart the label means are.

The critical comparison is BETWEEN-LABEL SEPARATION versus BETWEEN-SESSION
SPREAD WITHIN A LABEL. If the separation is smaller than the spread, the label
is not resolvable at all -- no classifier, no feature set, no amount of
cleverness -- and the ONLY fix is more sessions, because more sessions is what
shrinks the estimate of the within-label spread.

THIS IS ALREADY VISIBLE IN THE CURRENT CORPUS, and it is worth stating plainly
because it is not a modelling problem:

    session   label    concentration
    A         SPEAK      0.1869
    C         SILENT     0.1760
    B         SPEAK      0.1724

B and C differ by 0.0036. A and B are the SAME label and differ by 0.0145 --
four times more. The two sessions with opposite labels are closer together
than the two with the same label. A classifier asked to separate them is being
asked to do something the data does not contain.

THE SAMPLE SIZE.

With per-session means and a within-label between-session standard deviation
s, the standard error of a difference of two label means at n sessions each is
s*sqrt(2/n). For a target effect size (a difference of d) to be detectable at
a given confidence, n scales as 2*(z*s/d)^2. With d = 0.0036 and s = 0.0103
(the between-session spread of the two speaking sessions), n is on the order of
SIXTY sessions per class.

That number is alarming, and it is also only as good as its estimate: s is
computed from TWO sessions, so it is itself extremely unreliable. The point of
this file is not the number it prints today. It is that it prints a number
that IMPROVES as the corpus grows, and that it tells you, honestly, when the
corpus is finally sufficient.

WHAT IT DOES NOT DO.

It does not fit a classifier and report an accuracy. Accuracy on a corpus
that cannot support the claim is the number this project has been burned by.
It reports the RESOLVABILITY of the label, which is the precondition for any
accuracy to mean anything.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from ascii_motion_render import load_session  # noqa: E402
from teacher_conventional import region_feats, window_feats  # noqa: E402

FEATURE_NAMES = ["conc_mean", "conc_std", "conc_max", "persistence"]


def session_summary(maps: np.ndarray, k: int = 6, stride: int = 6):
    """Per-session means of the region features, over that session's windows."""
    rows = []
    for s in range(0, len(maps) - k + 1, stride):
        rows.append(region_feats(maps[s:s + k]))
    return np.array(rows) if rows else np.zeros((0, 4))


def analyse(corpus: Path, k: int = 6):
    sessions = []
    for meta_p in sorted(corpus.glob("*.json")):
        npz = meta_p.with_suffix(".npz")
        if not npz.is_file():
            continue
        meta = json.loads(meta_p.read_text())
        if meta.get("kind") != "condition":
            continue
        label = "SPEAK" if meta.get("speak") else "SILENT"
        sessions.append((meta["condition"], label,
                         session_summary(load_session(npz), k)))
    if len(sessions) < 2:
        return None

    out = {"question": ("is the label resolvable in this corpus, and how many "
                        "sessions per class would it take?"),
           "n_sessions": len(sessions), "sessions": [], "features": {}}

    print(f"sessions: {len(sessions)}")
    for name, label, summ in sessions:
        m = summ.mean(axis=0) if len(summ) else np.zeros(4)
        out["sessions"].append({
            "name": name, "label": label, "n_windows": int(len(summ)),
            "means": {FEATURE_NAMES[i]: float(m[i]) for i in range(4)},
        })
        print(f"  {name:8s} {label:7s} n={len(summ):3d}  " +
              "  ".join(f"{FEATURE_NAMES[i]}={m[i]:.4f}" for i in range(4)))

    labels = sorted({s[1] for s in sessions})
    print(f"\n  labels present: {labels}")

    for fi, fname in enumerate(FEATURE_NAMES):
        by_label = {}
        for name, label, summ in sessions:
            if len(summ):
                by_label.setdefault(label, []).append(float(summ[:, fi].mean()))

        # within-session spread (pooled)
        within = []
        for _n, _l, summ in sessions:
            if len(summ) > 1:
                within.append(float(summ[:, fi].std()))
        s_within = math.sqrt(statistics.fmean([w * w for w in within])) if within else None

        # between-session spread within each label
        s_between = {}
        for lab, vals in by_label.items():
            if len(vals) > 1:
                s_between[lab] = float(statistics.stdev(vals))
        pooled = list(s_between.values())
        s_between_pooled = math.sqrt(statistics.fmean([v * v for v in pooled])) \
            if pooled else None

        # between-label separation
        sep = None
        if len(labels) == 2 and all(l in by_label for l in labels):
            sep = abs(statistics.fmean(by_label[labels[0]])
                      - statistics.fmean(by_label[labels[1]]))

        # sessions needed for a target effect, using the within-label
        # between-session spread as the noise scale
        n_needed = None
        if sep and s_between_pooled and sep > 0:
            for target in (2.0, 3.0):
                n = math.ceil(2 * (target * s_between_pooled / sep) ** 2)
                n_needed = n_needed or {}
                n_needed[f"n_per_class_for_z{target:g}"] = n

        resolvable = bool(sep and s_between_pooled and sep > s_between_pooled)

        out["features"][fname] = {
            "within_session_sd": s_within,
            "between_session_sd_within_label": s_between_pooled,
            "between_session_sd_by_label": s_between,
            "between_label_separation": sep,
            "label_resolvable": resolvable,
            "n_per_class_estimate": n_needed,
            "per_label_means": {k: statistics.fmean(v)
                                for k, v in by_label.items()},
        }
        print(f"\n  {fname}:")
        print(f"    between-label separation      {sep if sep is not None else 'n/a'}")
        print(f"    between-session sd (same label) "
              f"{s_between_pooled if s_between_pooled is not None else 'n/a'}")
        print(f"    -> label resolvable?          {resolvable}")
        if n_needed:
            print(f"    sessions per class for z=2: {n_needed['n_per_class_for_z2']}"
                  f"   z=3: {n_needed['n_per_class_for_z3']}")

    out["verdict"] = (
        "The label is NOT resolvable in this corpus. The between-label "
        "separation is smaller than the between-session spread among sessions "
        "of the SAME label, so no classifier can separate them and more "
        "features will not help. The only fix is more sessions, and the "
        "n_per_class_estimate says how many."
        if not all(f["label_resolvable"] for f in out["features"].values())
        else "The label is resolvable in this corpus by the separation test. "
             "A classifier can be fitted, subject to the leave-one-session-out "
             "check.")

    out["limits"] = [
        "the between-session spread is estimated from very few sessions and is "
        "itself unreliable -- the n_per_class_estimate improves as the corpus "
        "grows, and the number printed today is a floor, not a forecast",
        "one speaker, one room, one lighting state",
        "this measures RESOLVABILITY, not accuracy; a corpus that cannot "
        "support the claim makes any accuracy meaningless",
    ]
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/mnt/c/proart_capture/corpus_v2")
    # CORRECTED CORPUS. The old /mnt/c/proart_capture/corpus was seeded
# at the wrong frame stride (23,040 instead of 19,200) and every frame
# in it is a tear between two real frames. See
# docs/research/NV12-STRIDE.md.
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    out = analyse(Path(a.corpus), a.k)
    if out is None:
        sys.exit("need at least two condition sessions")
    p = Path(a.out) / "teacher_data_requirement_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nVERDICT: {out['verdict']}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
