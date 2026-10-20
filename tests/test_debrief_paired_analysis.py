"""
test_debrief_paired_analysis.py -- the test machinery, and the trap that
the machinery exists to catch.

WHY THESE TESTS EXIST.

The lane's first implementation of the paired claim-level test POOLEDED the
repeats: it ran McNemar on 27 pairs -- the same nine claims measured three
times -- and treated them as 27 independent observations. That produced a
p = 0.0391 on the one budget step that looked significant.

At the unit the pre-registered rule actually names ("exact McNemar on the
NINE false claims"), the same comparison gives p = 0.2500.

So the pooled number was a manufactured significance, and it was
manufactured by the most ordinary mistake in the book: counting the same
thing more than once and calling it more data.

These tests pin:

  1. the sign test's minimum attainable p, which is what makes a
     three-repeat design unable to reach significance at ANY effect size;
  2. the exact McNemar p table on nine claims, and the minimum detectable
     effect that follows from it;
  3. the majority aggregation, so that the unit of analysis is the claim;
  4. that the pooled and claim-level analyses DISAGREE, and that the
     claim-level one is the one the rule names;
  5. that the null arm is a true null and the reversed pairing reports the
     other direction -- a test that cannot come out the other way is not a
     test.

A mutation test at the end: if the majority aggregation is removed, the
suite must fail.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import debrief_paired_analysis as P  # noqa: E402

RESULTS = ROOT / "experiments" / "debrief_paired_analysis_results.json"


# ------------------------------------------------------------- sign test

def test_three_pairs_cannot_reach_significance():
    """The arithmetic that made the old sweep's design uninformative.

    A two-sided exact sign test on n paired samples has a smallest
    attainable p of 2/2^n. At n = 3 that is 0.25, so the sweep could not
    have produced a significant result at ANY effect size.
    """
    r = P.sign_test([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    assert r["min_attainable_p_two_sided"] == pytest.approx(0.25)
    assert r["p_two_sided"] == pytest.approx(0.25)
    assert r["n_positive"] == 3 and r["n_negative"] == 0
    assert r["direction"] == "more_evidence_is_better"


def test_the_sign_test_can_come_out_the_other_way():
    r = P.sign_test([(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)])
    assert r["n_negative"] == 3
    assert r["direction"] == "more_evidence_is_worse"


def test_a_tie_is_not_a_discordant_pair():
    r = P.sign_test([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)])
    assert r["n_nonzero"] == 0
    assert r["p_two_sided"] == 1.0
    assert r["direction"] == "tie"


# --------------------------------------------------------------- McNemar

def test_the_exact_mcnemar_table_on_nine_claims():
    """The exact test is conservative, and the table says by how much."""
    table = {r["gained"]: r for r in P.mcnemar_p_table(9)}
    assert table[9]["p_two_sided"] == pytest.approx(0.00390625)
    assert table[8]["p_two_sided"] == pytest.approx(0.0390625)
    assert table[7]["p_two_sided"] == pytest.approx(0.1796875)
    assert table[5]["p_two_sided"] == pytest.approx(1.0)
    assert table[0]["p_two_sided"] == pytest.approx(0.00390625)


def test_the_minimum_detectable_effect_on_nine_claims():
    """Six claims, one way, is the smallest shift that clears p = 0.05."""
    assert P._mde(9) == 6
    # and a 5-4 split cannot clear it: 4 lost, 5 gained
    r = P.mcnemar([(True, False)] * 4 + [(False, True)] * 5)
    assert r["b_lost"] == 4 and r["c_gained"] == 5
    assert r["p_two_sided"] == 1.0
    assert not (r["p_two_sided"] < 0.05)


def test_mcnemar_reports_a_direction_even_when_there_is_no_discordance():
    """The concordant branch must still carry a direction.

    It did not, once, and the results file grew a key that the summary
    code did not expect.
    """
    r = P.mcnemar([(True, True)] * 9)
    assert r["p_two_sided"] == 1.0
    assert r["direction"] == "tie"
    assert r["b_lost"] == 0 and r["c_gained"] == 0


# --------------------------------------------------- the unit of analysis

def _arm(flags_per_run):
    return {"limit": 1, "runs": [{"caught_flags": f, "n_caught": sum(f),
                                   "headline": 0.0}
                                  for f in flags_per_run]}


def test_the_majority_aggregates_across_repeats():
    arm = _arm([[True, False, True], [True, False, False], [False, False, True]])
    assert P.majority_flags(arm) == [True, False, True]


def test_a_tie_is_not_a_caught_claim():
    """One of three repeats caught it is a tie, and the strict majority
    says not caught."""
    arm = _arm([[True, True], [False, True], [False, False]])
    assert P.majority_flags(arm) == [False, True]


def test_pooling_the_repeats_manufactures_significance():
    """THE TRAP. The same nine claims, measured three times.

    Pooled over the repeats, the test says p = 0.0391 and calls the step
    significant. At the unit the pre-registered rule names -- the claim --
    it says p = 0.2500 and calls it not significant.
    """
    # Three repeats in which the same three claims are caught in every
    # repeat at arm B and none at arm A: nine claims, three of which
    # agree.
    a = _arm([[False] * 9, [False] * 9, [False] * 9])
    b = _arm([[True] * 3 + [False] * 6] * 3)
    out = P.compare(a, b, "trap")
    pooled = out["mcnemar_pooled_over_repeats_ANTI_CONSERVATIVE"]
    primary = out["mcnemar_on_claims"]
    # At the CLAIM level: three claims are caught at B and not at A, six
    # agree. Three discordant claims in one direction is p = 0.25.
    assert primary["c_gained"] == 3 and primary["b_lost"] == 0
    assert primary["p_two_sided"] == pytest.approx(0.25)
    assert not (primary["p_two_sided"] < 0.05)
    # Pooled over the repeats, the SAME three claims are counted nine
    # times and the p collapses to a manufactured significance.
    assert pooled["c_gained"] == 9 and pooled["b_lost"] == 0
    assert pooled["p_two_sided"] < 0.05, (
        "the pooled version is the one that manufactures significance")
    assert primary["p_two_sided"] != pooled["p_two_sided"]
    assert "why_it_is_not_the_test" in pooled


def test_the_null_arm_is_a_true_null():
    a = _arm([[True, False, True]] * 3)
    out = P.compare(a, a, "null")
    assert out["mcnemar_on_claims"]["p_two_sided"] == 1.0
    assert out["mcnemar_on_claims"]["b_lost"] == 0
    assert out["mcnemar_on_claims"]["c_gained"] == 0
    assert out["sign_test_on_repeat_claim_counts"]["n_nonzero"] == 0


def test_the_reversed_pairing_reports_the_other_direction():
    a = _arm([[True, True, False]] * 3)
    b = _arm([[False, False, False]] * 3)
    out = P.compare(a, b, "reversed")
    assert out["mcnemar_on_claims"]["direction"] == "more_evidence_is_worse"


# ------------------------------------------------------ the committed run

def test_the_committed_results_are_present_and_agree_with_the_rules():
    if not RESULTS.is_file():
        pytest.skip("the paired analysis has not been run")
    d = json.loads(RESULTS.read_text())

    assert d["pre_registered_rules"]["the_unit_of_analysis_is"]
    assert "nine" in d["pre_registered_rules"]["more_evidence_helps_if"]

    arith = d["the_design_arithmetic"]
    assert arith["metric_quantum"] == pytest.approx(1.0 / 9.0)
    assert arith["min_attainable_p_for_n_repeats"] == pytest.approx(0.25)
    assert arith["min_claims_that_must_move_at_p05"] == 6

    for c in d["comparisons"]:
        m = c["mcnemar_on_claims"]
        assert "direction" in m
        assert m["n_pairs"] == 9, "the unit of analysis is the CLAIM"

    null = d["null_arm"]["mcnemar_on_claims"]
    assert null["p_two_sided"] == 1.0
    assert null["b_lost"] == 0 and null["c_gained"] == 0

    rev = d["reversed_pairing"]["mcnemar_on_claims"]
    assert rev["direction"] in ("more_evidence_is_worse", "more_evidence_is_better")


# ----------------------------------------------------------- mutation test

def test_the_majority_aggregation_is_load_bearing():
    """Remove the claim-level aggregation and this must fail.

    A suite that cannot fail is not evidence. This is the check that the
    rest of this file is testing something real: if `majority_flags`
    returned the pooled per-repeat flags instead, the claim-level p would
    change and the trap test above would stop holding.
    """
    src = (ROOT / "experiments" / "debrief_paired_analysis.py").read_text()
    assert "def majority_flags" in src, (
        "the claim-level aggregation is gone; the rest of this file now "
        "tests an analysis that cannot tell a claim from a claim-repeat")
    assert "claim_pairs = list(zip(ma, mb))" in src, (
        "the primary test is no longer paired at the claim level")
