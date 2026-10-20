"""
Tests for the director's run-to-run repeatability.

These are deliberately network-free. The measurement itself needs a hosted
model; what needs testing is the machinery around it -- the scorer, the
control, and the analysis -- because a measurement whose harness cannot be
shown to be deterministic is not a measurement.

The rule this file follows: every test must be able to fail. In particular
the control test is an ASSERTION about the harness, not a demonstration that
the harness happened to work today.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import debrief  # noqa: E402
import director_repeatability as dr  # noqa: E402


# ------------------------------------------------------------------ the scorer

def _pairs(n_false=3, n_true=2):
    """A synthetic audit set: n_false known-false claims, n_true known-true."""
    return [(f"false claim {i}", True) for i in range(n_false)] + \
           [(f"true claim {i}", False) for i in range(n_true)]


def _verdicts(spec):
    """spec: list of verdict strings, one per claim."""
    return {"verdicts": [{"index": i, "verdict": v, "reason": "test"}
                         for i, v in enumerate(spec)]}


def test_scoring_is_deterministic_given_verdicts():
    """The same verdicts must always give the same headline.

    This is the property that lets any spread seen with a real model be
    attributed to the MODEL rather than to the scorer.
    """
    pairs = _pairs()
    obj = _verdicts(["unsupported", "unsupported", "unsupported",
                     "supported", "supported"])
    headlines = {debrief.score_audit(pairs, obj)["headline_sens_plus_spec_minus_1"]
                 for _ in range(20)}
    assert headlines == {1.0}, f"scoring moved between identical verdicts: {headlines}"


def test_perfect_director_scores_one():
    pairs = _pairs()
    res = debrief.score_audit(pairs, _verdicts(["unsupported"] * 3 + ["supported"] * 2))
    assert res["headline_sens_plus_spec_minus_1"] == pytest.approx(1.0)


def test_both_degenerate_directors_score_zero():
    """An always-accept and an always-reject director must BOTH score 0.0.

    This is the property the first version of the metric broke: `sens - spec`
    scores an always-reject director at +1.0, which looks like a perfect
    advisor. A metric that cannot fail is not a metric.
    """
    pairs = _pairs()
    accept = debrief.score_audit(pairs, _verdicts(["supported"] * 5))
    reject = debrief.score_audit(pairs, _verdicts(["unsupported"] * 5))
    assert accept["headline_sens_plus_spec_minus_1"] == pytest.approx(0.0)
    assert reject["headline_sens_plus_spec_minus_1"] == pytest.approx(0.0)
    # And the lenient variant must not let either one score well either.
    assert accept["headline_lenient"] == pytest.approx(0.0)
    assert reject["headline_lenient"] == pytest.approx(0.0)


def test_degenerate_baselines_helper_agrees_with_the_scorer():
    pairs = _pairs()
    b = dr.degenerate_baselines(pairs)
    assert b["always_accept"]["headline"] == pytest.approx(0.0)
    assert b["always_reject"]["headline"] == pytest.approx(0.0)
    assert b["always_accept"]["sensitivity"] == pytest.approx(0.0)
    assert b["always_reject"]["specificity"] == pytest.approx(0.0)


# ------------------------------------------------------------------ the control

def test_control_stub_is_deterministic():
    """The stub director must produce an identical headline every time.

    If this fails, the harness is adding variance and any spread measured
    with a real model is the harness's, not the model's. This is the test
    that makes the whole measurement falsifiable.
    """
    pairs = _pairs()
    scored = [debrief.score_audit(pairs, dr.stub_verdicts(len(pairs)))
              for _ in range(25)]
    summary = dr.summarise_runs(scored)
    assert summary["headline_spread"] == 0.0
    assert summary["n_distinct_headlines"] == 1
    assert summary["n_unstable_claims"] == 0


def test_control_stub_with_mixed_but_fixed_verdicts_is_also_deterministic():
    """A stub that is right about some claims and wrong about others.

    This is the case that could plausibly break: the headline depends on the
    set of verdicts, so a scoring bug could hide behind a degenerate stub
    where every claim has the same answer.

    The stub below catches 2 of the 3 false claims and endorses both true
    ones, so its headline is 2/3 + 1.0 - 1 = 2/3 -- a deliberately
    non-degenerate, non-zero score.
    """
    pairs = _pairs()
    spec = ["unsupported", "unsupported", "supported", "supported", "supported"]
    scored = [debrief.score_audit(pairs, _verdicts(spec)) for _ in range(25)]
    summary = dr.summarise_runs(scored)
    assert summary["headline_spread"] == 0.0
    assert summary["n_distinct_headlines"] == 1
    assert summary["n_unstable_claims"] == 0
    # And it must not accidentally look perfect.
    assert summary["headline_median"] == pytest.approx(2.0 / 3.0)
    assert summary["headline_median"] != pytest.approx(1.0)


# ------------------------------------------------------------------ the analysis

def test_summary_detects_a_varying_model():
    """Two runs with different headlines must be reported as two distinct ones.

    The failure this guards against is a summariser that averages the runs
    and reports one number -- which is precisely how the original +0.857
    became a property of the model instead of a property of a run.
    """
    pairs = _pairs()
    a = debrief.score_audit(pairs, _verdicts(
        ["unsupported", "unsupported", "unsupported", "supported", "supported"]))
    b = debrief.score_audit(pairs, _verdicts(
        ["unsupported", "unsupported", "supported", "supported", "unsupported"]))
    s = dr.summarise_runs([a, b])
    assert s["n_distinct_headlines"] == 2
    assert s["headline_spread"] > 0
    assert s["headline_min"] < s["headline_max"]
    assert s["headline_median"] is not None


def test_summary_reports_the_median_not_only_the_mean():
    """A bimodal set of runs must have a median that is one of the modes.

    A mean of a bimodal score is a number nobody should quote, so the
    summary must carry the median and the distinct set.

    The count is odd on purpose: with an even number of runs the median is
    the midpoint of the two middle values, which is a number that belongs to
    no run and would muddy exactly the comparison being made here.
    """
    pairs = _pairs()
    good = debrief.score_audit(pairs, _verdicts(
        ["unsupported", "unsupported", "unsupported", "supported", "supported"]))
    bad = debrief.score_audit(pairs, _verdicts(["supported"] * 5))
    s = dr.summarise_runs([good, bad, good, bad, good])
    assert s["n_distinct_headlines"] == 2
    assert s["headline_median"] in s["distinct_headlines"]
    assert s["headline_median"] == pytest.approx(1.0)
    assert len(s["distinct_headlines"]) == 2
    # And the mean must NOT be one of the runs' headlines -- that is the
    # number nobody should quote.
    assert s["headline_mean"] not in s["distinct_headlines"]


def test_summary_flags_an_unstable_claim():
    """A claim that draws different verdicts across runs must be flagged.

    And a claim that is known-FALSE and unstable is the dangerous kind: the
    director endorses it on some runs and catches it on others.
    """
    pairs = _pairs()
    a = debrief.score_audit(pairs, _verdicts(
        ["unsupported", "unsupported", "unsupported", "supported", "supported"]))
    b = debrief.score_audit(pairs, _verdicts(
        ["unsupported", "supported", "unsupported", "supported", "supported"]))
    s = dr.summarise_runs([a, b])
    assert s["n_unstable_claims"] == 1
    unstable = s["unstable_claims"][0]
    assert unstable["index"] == 1
    assert unstable["is_false_claim"] is True
    assert unstable["stable"] is False
    assert unstable["unstable_and_false_claim"] is True
    # The two runs must agree on the other four claims.
    assert sum(1 for e in s["per_claim"] if e["stable"]) == 4


def test_summary_of_no_runs_is_empty_not_a_crash():
    """Zero scored runs must give None, not raise and not divide by zero."""
    s = dr.summarise_runs([])
    assert s["n_runs"] == 0
    assert s["headline_min"] is None
    assert s["headline_median"] is None
    assert s["headline_spread"] is None
    assert s["n_unstable_claims"] == 0


def test_summary_counts_unparseable_runs():
    """A run that could not be parsed must be counted, not silently dropped."""
    pairs = _pairs()
    ok = debrief.score_audit(pairs, _verdicts(["supported"] * 5))
    s = dr.summarise_runs([ok, {"error": "unparseable"}])
    assert s["n_runs"] == 2
    assert s["n_unparseable"] == 1
    assert s["n_scored"] == 1


# ------------------------------------------------------------------ the prompt

def test_audit_prompt_is_a_pure_function_of_its_inputs():
    """The same seed must give the same prompt, byte for byte.

    This is what licenses the headline claim of the experiment: the prompt
    is held identical across runs, so the only thing varying is the model.
    """
    p1, _ = debrief.build_audit_prompt("blind", debrief.audit_pairs(20261019))
    p2, _ = debrief.build_audit_prompt("blind", debrief.audit_pairs(20261019))
    assert p1 == p2
    assert len(p1) > 100


def test_audit_prompt_changes_with_the_seed():
    """A different seed must shuffle the claims, so the prompt must differ.

    Otherwise the seed is not doing anything and the order is fixed in a way
    the experiment cannot vary.
    """
    p1, _ = debrief.build_audit_prompt("blind", debrief.audit_pairs(1))
    p2, _ = debrief.build_audit_prompt("blind", debrief.audit_pairs(2))
    assert p1 != p2


def test_audit_prompt_does_not_contain_the_answer_key():
    """The prompt must not name which claims are false.

    The model sees the claims and the evidence, not the labels. If the
    labels leaked into the prompt the whole audit would be worthless.
    """
    prompt, _ = debrief.build_audit_prompt("blind", debrief.audit_pairs(20261019))
    for _, is_false in debrief.AUDIT_SET:
        if is_false:
            continue
    # The prompt is built from the claim TEXTS only; assert that a claim
    # text appears and that no boolean marker was appended to it.
    first_claim = debrief.AUDIT_SET[0][0][:40]
    assert first_claim[:20] in prompt
    assert "is_false_claim" not in prompt
    assert "known-false" not in prompt.lower()


def test_context_prompt_contains_material_and_is_repeatable():
    """In context mode the material is included, and the same call gives the
    same prompt."""
    p1, lim1 = debrief.build_audit_prompt("context", debrief.audit_pairs(),
                                          materials_limit=2000)
    p2, lim2 = debrief.build_audit_prompt("context", debrief.audit_pairs(),
                                          materials_limit=2000)
    assert p1 == p2
    assert lim1 == lim2 == 2000
    assert "RESEARCH NOTES AND RESULTS" in p1


# ------------------------------------------------------- the majority vote

def _scored(pairs, spec_list):
    """A list of scored runs, one per verdict list."""
    return [debrief.score_audit(pairs, _verdicts(spec)) for spec in spec_list]


def test_majority_picks_the_verdict_the_runs_agree_on():
    pairs = _pairs()
    # 4 runs say "unsupported" on claim 0, 1 says "supported" -> unsupported.
    runs = _scored(pairs, [["unsupported"] * 5, ["unsupported"] * 5,
                           ["unsupported"] * 5, ["supported"] + ["unsupported"] * 4])
    _, summ = dr.majority_verdicts(runs, pairs)
    assert summ["per_claim"][0]["verdict"] == "unsupported"
    assert summ["per_claim"][0]["stable"] is False
    assert summ["per_claim"][0]["tie"] is False


def test_majority_tie_breaks_to_uncertain_not_to_supported():
    """A tie must not be broken towards "supported".

    Breaking a 2-2 tie towards "supported" would push the headline UP on
    exactly the claims where the model is least sure -- which is the
    direction that endorses false claims. It breaks to "uncertain", the
    safe answer for a reviewer: it neither endorses nor accuses.
    """
    pairs = _pairs()
    runs = _scored(pairs, [["supported"] * 5, ["supported"] * 5,
                           ["unsupported"] * 5, ["unsupported"] * 5])
    _, summ = dr.majority_verdicts(runs, pairs)
    assert summ["per_claim"][0]["verdict"] == "uncertain"
    assert summ["per_claim"][0]["tie"] is True
    assert summ["n_tied_claims"] == 5


def test_majority_of_identical_runs_reproduces_that_run():
    pairs = _pairs()
    spec = ["unsupported", "unsupported", "supported", "supported", "supported"]
    runs = _scored(pairs, [spec] * 5)
    _, summ = dr.majority_verdicts(runs, pairs)
    single = debrief.score_audit(pairs, _verdicts(spec))
    assert summ["headline"] == pytest.approx(
        single["headline_sens_plus_spec_minus_1"])


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    pairs = _pairs()
    runs = _scored(pairs, [["supported"] * 5, ["unsupported"] * 5,
                           ["supported"] * 5, ["unsupported"] * 5,
                           ["supported"] * 5])
    a = dr.bootstrap_majority(runs, pairs, 3, n_groups=200, seed=7)
    b = dr.bootstrap_majority(runs, pairs, 3, n_groups=200, seed=7)
    assert a["distinct_headlines"] == b["distinct_headlines"]
    assert a["counts"] == b["counts"]
    c = dr.bootstrap_majority(runs, pairs, 3, n_groups=200, seed=8)
    assert c["n_sampled"] == a["n_sampled"]


def test_voting_does_NOT_fix_the_instability():
    """THE RESULT, pinned.

    When the model splits on a claim, running it five times and taking the
    majority does not converge on an answer -- and the headline does not
    narrow at all. This is the measurement that says "run it more times" is
    not a fix, and it is falsifiable: if the majority vote DID narrow the
    distribution, this test would fail.

    The fixture is deliberately NOT the degenerate one. An always-accept and
    an always-reject director both score 0.0 -- that is what the metric is
    for -- so a fixture built from those two would show a spread of zero and
    prove nothing. Here the two run types are a strong director (1.000) and
    a mediocre one (0.167), and the majority vote over 3 or 5 runs still
    spans the whole range.
    """
    pairs = _pairs()
    strong = ["unsupported"] * 3 + ["supported"] * 2
    weak = ["unsupported", "unsupported", "supported", "supported", "unsupported"]
    runs = _scored(pairs, [strong] * 3 + [weak] * 3)
    single = dr.summarise_runs(runs)
    assert single["headline_spread"] > 0.8
    for gs in (3, 5):
        boot = dr.bootstrap_majority(runs, pairs, gs, n_groups=500, seed=11)
        # Voting does NOT narrow the range the single runs span.
        assert boot["spread"] >= single["headline_spread"] - 1e-9
        # And it does not settle on one answer either.
        assert boot["probability_of_the_mode"] < 0.75


def test_degenerate_directors_score_the_same_so_they_cannot_stand_in_for_a_fixture():
    """The reason the fixture above is not built from the degenerate pair.

    An always-accept and an always-reject director BOTH score 0.0, so a
    fixture made of them has a single-run spread of zero and would make any
    claim about stability vacuously true.
    """
    pairs = _pairs()
    accept = debrief.score_audit(pairs, _verdicts(["supported"] * 5))
    reject = debrief.score_audit(pairs, _verdicts(["unsupported"] * 5))
    assert accept["headline_sens_plus_spec_minus_1"] == pytest.approx(0.0)
    assert reject["headline_sens_plus_spec_minus_1"] == pytest.approx(0.0)
    runs = _scored(pairs, [["supported"] * 5] * 3 + [["unsupported"] * 5] * 3)
    assert dr.summarise_runs(runs)["headline_spread"] == pytest.approx(0.0)


def test_bootstrap_refuses_to_bootstrap_with_too_few_runs():
    pairs = _pairs()
    runs = _scored(pairs, [["supported"] * 5, ["unsupported"] * 5])
    out = dr.bootstrap_majority(runs, pairs, 5)
    assert "error" in out
    assert "need at least 5 runs" in out["error"]


# ------------------------------------------------------- rebuilding a file

def test_rebuild_from_persisted_runs_reproduces_the_headlines():
    """The persisted file stores verdicts, not rows; the rebuild is exact.

    If this drifts, every analysis run with --from-files is measuring
    something other than what the file says.
    """
    pairs = _pairs()
    specs = [["unsupported"] * 5, ["supported"] * 5, ["unsupported"] * 5]
    runs = _scored(pairs, specs)
    persisted = [{"headline": r["headline_sens_plus_spec_minus_1"],
                  "sensitivity": r["sensitivity_rejected_false_claims"],
                  "specificity": r["specificity_kept_true_claims"],
                  "verdicts": [row["verdict"] for row in r["rows"]]}
                 for r in runs]
    rebuilt = dr.rebuild_scored(persisted, pairs)
    assert len(rebuilt) == len(runs)
    for a, b in zip(runs, rebuilt):
        assert a["headline_sens_plus_spec_minus_1"] == pytest.approx(
            b["headline_sens_plus_spec_minus_1"])
