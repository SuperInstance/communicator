"""
Measure the calibration of the judged tracks. This is the gap Phase 2 left.

WHAT THIS MEASURES.

For each INTENTION level L, Jev returns a probability that the utterance sits
at L. Against labelled utterances, ECE asks: when Jev says 0.8 for level L, is
that right about 80% of the time?

WHY THIS MATTERS MORE THAN ACCURACY.

An agent that may act autonomously reads these numbers through a threshold. An
uncalibrated 0.9 that actually means 0.55 is worse than no number, because a
threshold on it fires at the wrong rate AND every individual decision still
looks reasonable. The failure is invisible by construction.

HONEST LIMITS, UP FRONT, because they bound the result.

  * THE LABELS ARE ONE AUTHOR'S JUDGEMENTS, NOT A USER'S. This measures
    agreement with a reasonable annotator. It is not ground truth about Casey,
    and it is not ground truth about anyone. If the labels are biased, the
    model can be perfectly calibrated against them and still wrong.
  * Every probe is a TYPED utterance. Nothing here says anything about speech.
  * Utterances marked AMBIGUOUS are excluded from the headline ECE and reported
    separately. A label a careful annotator cannot settle is not evidence.
  * The model is a sampler. Phase 2 measured five identical gate runs spanning
    0.913-0.940 AUC, so ECE is reported as mean +/- stdev over repeated passes.
    A single pass is not a number.

Run:
    set -a && . /home/proart/projects/glasses/.env && set +a
    .venv/bin/python experiments/judged_calibration.py --passes 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calib import (adaptive_calibration_error, expected_calibration_error,  # noqa: E402
                        worst_of)

# -- the probe corpus --------------------------------------------------------
#
# Four levels, matching src/judged.INTENTION_SCALE. Written so that a careful
# person would agree on the label without much argument -- that is the point:
# a disputed label measures annotator disagreement, not calibration.
#
# (text, level, ambiguous)
IDLE, ATTENTIVE, FOCUSED, URGENT = 0, 1, 2, 3

PROBES: list[tuple[str, int, bool]] = [
    # -- idle: ambient, reflexive, no intent behind it
    ("mm hmm", IDLE, False),
    ("yeah", IDLE, False),
    ("oh okay", IDLE, False),
    ("heh", IDLE, False),
    ("well anyway", IDLE, False),
    ("just thinking out loud here", IDLE, False),
    ("the kettle's on", IDLE, False),
    ("it's raining again", IDLE, False),
    ("oh, that's interesting", IDLE, False),
    ("huh", IDLE, False),
    ("right then", IDLE, False),
    ("lovely morning isn't it", IDLE, False),

    # -- attentive: mild interest, considering something
    ("hm, that's a fair point", ATTENTIVE, False),
    ("let me think about that", ATTENTIVE, False),
    ("I hadn't thought of it that way", ATTENTIVE, False),
    ("say more about that", ATTENTIVE, False),
    ("interesting, go on", ATTENTIVE, False),
    ("what do you make of that", ATTENTIVE, False),
    ("huh, that's odd", ATTENTIVE, False),
    ("I'm not sure I follow", ATTENTIVE, False),
    ("give me a second to think", ATTENTIVE, False),
    ("that's worth considering", ATTENTIVE, False),
    ("carry on", ATTENTIVE, False),
    ("so what happened next", ATTENTIVE, False),

    # -- focused: deliberate, means exactly what it says
    ("pull up the last commit", FOCUSED, False),
    ("show me the failing test", FOCUSED, False),
    ("rename that branch to release two", FOCUSED, False),
    ("what files changed in that commit", FOCUSED, False),
    ("open the config and read the timeout value", FOCUSED, False),
    ("tell me what broke the build", FOCUSED, False),
    ("list the branches merged this week", FOCUSED, False),
    ("explain what this function does", FOCUSED, False),
    ("run the suite and report the failures", FOCUSED, False),
    ("write that summary to notes file", FOCUSED, False),
    ("which of these two tests is slower", FOCUSED, False),
    ("show me the diff against main", FOCUSED, False),

    # -- urgent: time-sensitive, act now
    ("stop the deploy right now", URGENT, False),
    ("kill the process immediately", URGENT, False),
    ("the production database is down, fix it now", URGENT, False),
    ("cancel everything and roll back immediately", URGENT, False),
    ("I need that right now or we lose the window", URGENT, False),
    ("emergency, get me out of this building", URGENT, False),
    ("abort abort abort", URGENT, False),
    ("it's on fire", URGENT, False),
    ("call the ambulance now", URGENT, False),
    ("everything is burning, help", URGENT, False),
    ("take the site down this second", URGENT, False),
    ("I have four minutes, go", URGENT, False),

    # -- ambiguous: EXCLUDED from the headline, reported separately
    ("sure, whatever you think", ATTENTIVE, True),
    ("fine", FOCUSED, True),
    ("do it", FOCUSED, True),
    ("now would be good", ATTENTIVE, True),
    ("asap please", URGENT, True),
    ("handle it", ATTENTIVE, True),
    ("you know what to do", FOCUSED, True),
    ("right away", URGENT, True),
    ("can you take a look", ATTENTIVE, True),
    ("fix it", FOCUSED, True),
]


def measure_once(model: str, timeout: float = 15.0):
    """One pass over the whole corpus. Returns (pairs, ambiguous_pairs)."""
    from typesafe_sdk import TypeSafeClient
    from src.judged import INTENTION_SCALE, build_questions

    n_levels = len(INTENTION_SCALE)
    client = TypeSafeClient()
    per_level: dict[int, list[tuple[float, int]]] = {i: [] for i in range(n_levels)}
    amb: dict[int, list[tuple[float, int]]] = {i: [] for i in range(n_levels)}
    resolved = set()

    for text, label, ambiguous in PROBES:
        try:
            resp = client.system_one(state=text, questions=build_questions(),
                                     model=model, timeout=timeout)
        except Exception as exc:                      # noqa: BLE001
            print(f"  !! {text!r}: {type(exc).__name__}")
            continue
        resolved.add(str(getattr(resp, "model", model)))
        probs = resp.answers["intention"].probabilities
        target = amb if ambiguous else per_level
        for lvl in range(n_levels):
            target[lvl].append((float(probs.get(lvl, 0.0)),
                                1 if lvl == label else 0))

    pairs = [p for lvl in per_level.values() for p in lvl]
    amb_pairs = [p for lvl in amb.values() for p in lvl]
    return pairs, amb_pairs, resolved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--model", default="jev-1.14.0")
    args = ap.parse_args()

    key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFEAI_KEY")
    if not key:
        print("SKIP  no TYPESAFE_API_KEY")
        return 0

    from src.judged import INTENTION_SCALE

    n_clean = sum(1 for _, _, a in PROBES if not a)
    n_amb = len(PROBES) - n_clean
    print(f"\ncorpus: {len(PROBES)} probes ({n_clean} labelled, {n_amb} ambiguous "
          f"and excluded)\nmodel:  {args.model}\npasses: {args.passes}\n")

    ew, am, results, amb_ew = [], [], [], []
    for i in range(1, args.passes + 1):
        pairs, amb_pairs, resolved = measure_once(args.model)
        if not pairs:
            return 1
        r_eq = expected_calibration_error(pairs)
        r_am = adaptive_calibration_error(pairs)
        ew.append(r_eq.ece)
        am.append(r_am.ece)
        results.append(r_eq)
        a_eq = expected_calibration_error(amb_pairs) if amb_pairs else None
        if a_eq:
            amb_ew.append(a_eq.ece)
        print(f"  pass {i}: ECE(equal-width)={r_eq.ece:.4f}  "
              f"ECE(equal-mass)={r_am.ece:.4f}  "
              f"mean_conf={r_eq.mean_confidence:.3f}  base_rate={r_eq.base_rate:.3f}  "
              f"resolved={sorted(resolved)}")

    mean_ew, sd_ew, n = worst_of(results)
    mean_am = sum(am) / len(am)
    print(f"\n{'=' * 70}")
    print(f"INTENTION calibration, n={results[0].n} labelled probes x {args.passes} passes")
    print(f"{'-' * 70}")
    print(f"  ECE equal-width : {mean_ew:.4f} +/- {sd_ew:.4f}")
    print(f"  ECE equal-mass  : {mean_am:.4f}")
    if amb_ew:
        print(f"  (ambiguous subset, n={len(PROBES) - n_clean} probes) "
              f"ECE={sum(amb_ew) / len(amb_ew):.4f}")
    print(f"  base rate (level 3 share) : {results[0].base_rate:.3f}")
    print(f"  mean stated confidence    : {results[0].mean_confidence:.3f}")

    w = results[0].worst_bin()
    if w:
        print(f"\n  worst bin: conf {w.lo:.2f}-{w.hi:.2f}  "
              f"stated {w.confidence:.3f}  observed {w.accuracy:.3f}  "
              f"gap {w.gap:+.3f}  (n={w.n})")

    print(f"\n{'=' * 70}")
    print(f"per-bin, final pass")
    print(f"{'bin':<14}{'n':>5}{'stated':>9}{'observed':>10}{'gap':>9}")
    for b in results[0].bins:
        if b.n:
            print(f"{b.lo:.1f}-{b.hi:.1f}{'':<5}{b.n:>5}{b.confidence:>9.3f}"
                  f"{b.accuracy:>10.3f}{b.gap:>+9.3f}")

    out = ROOT / "experiments" / "judged_calibration_results.json"
    out.write_text(json.dumps({
        "model": args.model,
        "passes": args.passes,
        "n_labelled": results[0].n,
        "n_ambiguous": n_amb,
        "ece_equal_width": {"mean": mean_ew, "stdev": sd_ew, "per_pass": ew},
        "ece_equal_mass": {"mean": mean_am, "per_pass": am},
        "ece_ambiguous_subset": (sum(amb_ew) / len(amb_ew)) if amb_ew else None,
        "base_rate": results[0].base_rate,
        "mean_confidence": results[0].mean_confidence,
    }, indent=2))
    print(f"\n  wrote {out}")
    print("\n  REMINDER: these labels are one author's judgements, not a user's.")
    print("  This measures agreement with a reasonable annotator, not truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())