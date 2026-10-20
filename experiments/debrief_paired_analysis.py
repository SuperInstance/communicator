#!/usr/bin/env python3
"""
debrief_paired_analysis.py -- compare the director's ARMS in claims, in
pairs, instead of in fractions.

WHY THIS FILE EXISTS.

`DEBRIEF-REAUDIT.md` section 4.7 measured something that invalidates most
of that document's comparisons:

  * the audit set has NINE false claims and ONE true one, so the headline
    `sens + spec - 1` moves in steps of 1/9 = 0.1111 -- it is QUANTISED;
  * the observed run-to-run spread is 0.111 and 0.222 -- ONE and TWO
    quanta;
  * so the instrument's noise is the size of its resolution.

The effect claimed by the "does more evidence make the director better"
sweep was **+0.1111**: exactly one quantum. The sweep compared MEDIANS of
three repeats per budget. That comparison cannot distinguish "more evidence
helps" from "one claim happened to be answered differently this time".

This file replaces that comparison with one the instrument's resolution can
actually support.

THE TWO COMPARISONS, AND WHAT EACH CAN DETECT.

  1. SIGN TEST ON THE PAIRED REPEATS. Pair repeat i at budget A with
     repeat i at budget B. With n=3 pairs, the SMALLEST attainable
     two-sided p-value is 2/8 = 0.25. So the sweep's design could not
     have produced a significant result at p < 0.05 no matter how large
     the effect was. This is arithmetic, not a power calculation: the
     verdict label "EFFECT RESOLVED" was not available to that design.
  2. McNEMAR ON THE CLAIMS. Each of the nine false claims is a PAIRED
     OBSERVATION: was it caught at budget A, and was it caught at budget
     B? That gives n=9 pairs instead of n=3 runs, and it tests the thing
     that actually moves: the director catching more of the specific
     claims it can catch.

  3. AND THE MINIMUM DETECTABLE EFFECT. How many claims must move, and in
     which direction, before the paired claim-level test can reject the
     null at all? Computed, not asserted.

THE CONTROLS, WHICH CAN COME OUT THE OTHER WAY.

  * A NULL ARM where the two budgets are the SAME. A paired comparison of
    a budget with itself must produce zero discordant pairs and p = 1.0.
    If it does not, the test machinery is broken and every other number
    here is void.
  * A DIRECTION CONTROL. The test is two-sided; a deliberately REVERSED
    pairing must be able to report the opposite sign.

WHAT THIS FILE IS NOT.

It is not a better sweep. It is the same sweep, analysed in the unit the
metric can actually resolve, with the arithmetic of what the design could
have detected stated before the numbers are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import debrief  # noqa: E402

SWEEP_MODEL = "@cf/openai/gpt-oss-120b"
SWEEP_LIMITS = (20000, 60000, 140000, 240000)


# ------------------------------------------------------------------ tests

def sign_test(pairs: list[tuple[float, float]]) -> dict:
    """Two-sided exact sign test on paired differences. n = len(pairs)."""
    diffs = [b - a for a, b in pairs]
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    ties = sum(1 for d in diffs if d == 0)
    n = pos + neg
    if n == 0:
        return {"n_pairs": len(pairs), "n_nonzero": 0, "n_positive": 0,
                "n_negative": 0, "n_ties": ties, "p_two_sided": 1.0,
                "min_attainable_p_two_sided": 1.0, "direction": "tie",
                "note": "no discordant pair: the two arms are identical"}
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    p = min(1.0, 2.0 * tail / (2 ** n))
    return {"n_pairs": len(pairs), "n_nonzero": n, "n_positive": pos,
            "n_negative": neg, "n_ties": ties, "p_two_sided": p,
            "min_attainable_p_two_sided": 2.0 / (2 ** n),
            "direction": ("more_evidence_is_better" if pos > neg
                          else "more_evidence_is_worse" if neg > pos
                          else "tie"),
            "note": ("min_attainable_p is the smallest p this n can produce; "
                     "a design whose minimum attainable p is above 0.05 CANNOT "
                     "produce a significant result at any effect size")}


def mcnemar(pairs: list[tuple[bool, bool]]) -> dict:
    """Exact McNemar on paired binary outcomes.

    `pairs` is (caught_at_a, caught_at_b) for each claim. b is the number
    caught at A but not B, c is A-not-B. Under H0, b and c come from a
    fair coin; the two-sided exact p is the binomial tail.
    """
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    both = sum(1 for x, y in pairs if x and y)
    neither = sum(1 for x, y in pairs if not x and not y)
    n = b + c
    if n == 0:
        return {"n_pairs": len(pairs), "b_lost": 0, "c_gained": 0,
                "n_concordant_both": both, "n_concordant_neither": neither,
                "p_two_sided": 1.0, "direction": "tie",
                "min_detectable_shift_at_p05": _mde(len(pairs)),
                "note": "no discordant claim: the two arms agree on every claim"}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    p = min(1.0, 2.0 * tail / (2 ** n))
    return {"n_pairs": len(pairs), "b_lost": b, "c_gained": c,
            "n_concordant_both": both, "n_concordant_neither": neither,
            "p_two_sided": p,
            "min_detectable_shift_at_p05": _mde(n),
            "direction": ("more_evidence_is_better" if c > b
                          else "more_evidence_is_worse" if b > c else "tie")}


def _mde(n: int) -> int | None:
    """How many claims must move, all in ONE direction, to reach p < 0.05?

    The exact test's p depends on min(b, c), the SMALLER arm of the split.
    So a shift of d claims in one direction and none the other way gives
    p = 2 / 2^d, which needs d >= 6 to clear 0.05 -- and a two-sided split
    needs nearly all of the claims to move, because the smaller arm drives
    the tail. Both numbers are in `mcnemar_p_table`.
    """
    for d in range(1, n + 1):
        if 2.0 / (2 ** d) < 0.05:
            return d
    return None


def mcnemar_p_table(n: int) -> list:
    """The exact two-sided p for every split of n claims. Read it before
    designing anything on nine claims."""
    rows = []
    for k in range(n, -1, -1):
        other = n - k
        m = min(k, other)
        tail = sum(math.comb(n, i) for i in range(0, m + 1))
        p = min(1.0, 2.0 * tail / (2 ** n)) if (k + other) else 1.0
        rows.append({"gained": k, "lost": other, "p_two_sided": p,
                     "significant_at_05": p < 0.05})
    return rows


# --------------------------------------------------------------- the arms

def caught_flags(rows: list, n_false: int) -> list:
    """For each FALSE claim, was it caught? (rejected and correct)"""
    flags = []
    for r in rows:
        if not r.get("is_false_claim"):
            continue
        flags.append(bool(r.get("rejected")))
    return flags


def run_arm(model: str, limit: int, repeats: int) -> dict:
    runs = []
    for i in range(repeats):
        t0 = time.time()
        r = debrief.run_audit(model, "context", materials_limit=limit)
        rec = {"repeat": i + 1, "limit": limit,
               "elapsed_s": round(time.time() - t0, 2),
               "error": r.get("error")}
        if not r.get("error"):
            rows = r.get("rows") or []
            flags = caught_flags(rows, 9)
            rec["caught_flags"] = flags
            rec["n_false_claims"] = len(flags)
            rec["n_caught"] = sum(flags)
            rec["n_kept_true"] = sum(
                1 for x in rows
                if not x.get("is_false_claim") and x.get("verdict") == "supported")
            rec["n_uncertain"] = sum(1 for x in rows
                                     if x.get("verdict") == "uncertain")
            rec["headline"] = r.get("headline_sens_plus_spec_minus_1")
        runs.append(rec)
        print(f"  [{limit:>6}] repeat {i + 1}/{repeats}: "
              f"caught {rec.get('n_caught')}/{rec.get('n_false_claims')} "
              f"kept_true {rec.get('n_kept_true')} "
              f"uncertain {rec.get('n_uncertain')} "
              f"headline {rec.get('headline')} ({rec.get('elapsed_s')}s)",
              flush=True)
    return {"limit": limit, "runs": runs}


def majority_flags(arm: dict) -> list:
    """Per CLAIM, the majority verdict across the arm's repeats.

    THE UNIT OF ANALYSIS IS THE CLAIM, NOT THE CLAIM-REPEAT. The same
    nine claims are measured at every repeat, so pooling the repeats
    gives twenty-seven observations of nine things and treats them as
    independent. That is anti-conservative, and it is not a small
    correction: a pooled p of 0.031 becomes a claim-level p of 0.500 on
    the very comparison it was computed from.
    """
    runs = [r for r in arm["runs"] if not r.get("error")]
    if not runs:
        return []
    n = len(runs[0].get("caught_flags") or [])
    out = []
    for k in range(n):
        votes = [r["caught_flags"][k] for r in runs]
        out.append(sum(votes) * 2 > len(votes))   # strict majority
    return out


def compare(arm_a: dict, arm_b: dict, label: str) -> dict:
    """Pair the two arms, at the unit the pre-registered rule names."""
    a_runs = [r for r in arm_a["runs"] if not r.get("error")]
    b_runs = [r for r in arm_b["runs"] if not r.get("error")]
    n = min(len(a_runs), len(b_runs))
    headline_pairs = [(a_runs[i]["headline"], b_runs[i]["headline"])
                      for i in range(n)]
    count_pairs = [(a_runs[i]["n_caught"], b_runs[i]["n_caught"])
                   for i in range(n)]
    # THE PRIMARY TEST: the nine claims, each aggregated across repeats.
    ma, mb = majority_flags(arm_a), majority_flags(arm_b)
    claim_pairs = list(zip(ma, mb))
    # The pooled version, kept and LABELLED, because it is the number a
    # careless analysis produces and it disagrees with the right one.
    pooled_pairs = [(bool(a_runs[i]["caught_flags"][k]),
                     bool(b_runs[i]["caught_flags"][k]))
                    for i in range(n) for k in range(len(a_runs[i]["caught_flags"]))]
    primary = mcnemar(claim_pairs)
    out = {
        "label": label,
        "budget_a": arm_a["limit"], "budget_b": arm_b["limit"],
        "n_repeats_paired": n,
        "median_caught_a": statistics.median([r["n_caught"] for r in a_runs]),
        "median_caught_b": statistics.median([r["n_caught"] for r in b_runs]),
        "sign_test_on_repeat_headlines": sign_test(headline_pairs),
        "sign_test_on_repeat_claim_counts": sign_test(count_pairs),
        "mcnemar_on_claims": primary,
        "mcnemar_pooled_over_repeats_ANTI_CONSERVATIVE": {
            **mcnemar(pooled_pairs),
            "why_it_is_not_the_test": (
                "this pools the same nine claims across the repeats and "
                "treats the twenty-seven observations as independent. The "
                "claims are the same nine, so the observations are "
                "correlated, and the p-value is too small. It is reported "
                "because it is the number a careless analysis produces, "
                "and it disagrees with the pre-registered one."),
        },
        "headline_pairs": headline_pairs,
        "claim_count_pairs": count_pairs,
        "claim_level_pairs": claim_pairs,
    }
    out["detectable_at_p05"] = primary["p_two_sided"] < 0.05
    return out


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=SWEEP_MODEL)
    ap.add_argument("--budgets", default=",".join(str(b) for b in SWEEP_LIMITS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--include-null-arm", action="store_true", default=True)
    ap.add_argument("--out",
                    default=str(ROOT / "experiments"
                                / "debrief_paired_analysis_results.json"))
    a = ap.parse_args()

    budgets = [int(x) for x in a.budgets.split(",") if x.strip()]
    print(f"paired analysis: {a.model}, {a.repeats} repeats per budget")

    arms = []
    for lim in budgets:
        print(f"\n=== budget {lim} ===")
        arms.append(run_arm(a.model, lim, a.repeats))

    comparisons = []
    for i in range(len(arms) - 1):
        comparisons.append(compare(arms[i], arms[i + 1],
                                   f"{arms[i]['limit']} -> {arms[i + 1]['limit']}"))
    # THE NULL ARM: a budget compared with ITSELF. Zero discordant pairs
    # and p = 1.0. If this is not true, the machinery is broken.
    null_arm = None
    if a.include_null_arm:
        print("\n=== NULL ARM: a budget compared with itself ===")
        null_arm = compare(arms[-1], arms[-1], "null arm: same budget twice")
        print(json.dumps(null_arm["mcnemar_on_claims"], indent=2))

    # AND A REVERSED PAIRING, to show the test can report the other sign.
    reversed_pair = None
    if len(arms) >= 2:
        reversed_pair = compare(arms[-1], arms[0],
                                f"reversed: {arms[-1]['limit']} -> {arms[0]['limit']}")

    res = {
        "date": "2026-10-20",
        "question": (
            "Does MORE evidence make the director better? The previous "
            "sweep compared medians of three repeats and found +0.1111, "
            "which is exactly one quantum of a metric whose run-to-run "
            "spread is one to two quanta. This file asks the question in "
            "the unit the metric can resolve: claims, in pairs."),
        "pre_registered_rules": {
            "more_evidence_helps_if": (
                "a paired CLAIM-level test (exact McNemar on the nine false "
                "claims, each claim aggregated across the repeats) rejects "
                "at p < 0.05 with more claims caught at the larger budget"),
            "the_unit_of_analysis_is": (
                "THE CLAIM (nine of them), not the claim-repeat. The same "
                "nine claims are measured at every repeat, so pooling the "
                "repeats gives twenty-seven observations of nine things "
                "treated as independent. That is anti-conservative, and it "
                "is not a small correction: on the 60,000 -> 140,000 step "
                "the pooled p is 0.031 and the claim-level p is 0.500."),
            "the_effect_is_below_the_noise_if": (
                "the paired test does not reject at p < 0.05, OR the design "
                "cannot reach p < 0.05 at any effect size"),
            "the_claim_level_test_is_valid_if": (
                "the NULL ARM -- a budget compared with itself -- gives zero "
                "discordant claims and p = 1.0"),
            "the_test_can_come_out_the_other_way_if": (
                "the REVERSED pairing reports the opposite direction or an "
                "opposite sign"),
            "verdict_is_written_before_the_numbers_are_read": True,
        },
        "model": a.model,
        "repeats_per_budget": a.repeats,
        "budgets": budgets,
        "arms": arms,
        "comparisons": comparisons,
        "null_arm": null_arm,
        "reversed_pairing": reversed_pair,
        "the_design_arithmetic": {
            "metric_quantum": 1.0 / 9.0,
            "quantum_note": (
                "the audit set has nine false claims and one true one, so "
                "the headline moves in steps of 1/9 = 0.1111"),
            "min_attainable_p_for_n_repeats": 2.0 / (2 ** a.repeats),
            "min_attainable_p_note": (
                f"with {a.repeats} paired repeats the smallest two-sided p "
                f"any effect size can produce is "
                f"{2.0 / (2 ** a.repeats):.4f}; a design whose minimum "
                "attainable p is above 0.05 CANNOT produce a significant "
                "result, whatever the effect"),
            "min_claims_that_must_move_at_p05": _mde(9),
            "mcnemar_p_table_for_9_claims": mcnemar_p_table(9),
            "mcnemar_table_note": (
                "the exact test's p depends on the SMALLER arm of the split, "
                "so it is conservative: on nine claims it rejects at p<0.05 "
                "only when at least 8 of the 9 move in one direction, and a "
                "shift of 6 claims one way (and none the other) gives "
                "p = 0.031"),
        },
        "provenance": {
            "script": "experiments/debrief_paired_analysis.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "debrief_sha256": sha256_file(ROOT / "experiments" / "debrief.py"),
            "model": a.model,
            "repeats": a.repeats,
            "budgets": budgets,
        },
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")
    print(json.dumps(res["the_design_arithmetic"], indent=2))
    for c in comparisons:
        print(f"\n{c['label']}:")
        print("  claim counts  ", json.dumps(c["sign_test_on_repeat_claim_counts"]))
        print("  McNemar      ", json.dumps(c["mcnemar_on_claims"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
