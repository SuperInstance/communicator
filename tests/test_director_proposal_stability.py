"""
Tests for the director's PROPOSAL stability.

Network-free. The measurement needs a hosted model; what needs testing is the
comparer and the control around it, because a measurement whose harness cannot
be shown deterministic is not a measurement.

As everywhere in this repo: every test must be able to fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import director_proposal_stability as dps  # noqa: E402


# ---------------------------------------------------------------- the control

def test_control_stub_is_completely_stable():
    """The stub must give one proposal, one set of unsafe claims, every time.

    If this fails, the comparer is adding variance and the measurement of the
    real model is the comparer's, not the model's.
    """
    c = dps.compare_debriefs([dps.stub_debrief() for _ in range(25)])
    assert c["n"] == 25
    assert c["n_distinct_titles"] == 1
    assert c["n_stable_fields"] == len(dps.REQUIRED_FIELDS)
    assert c["n_distinct_unsafe_claims"] == 1
    assert c["n_runs_with_all_required_fields"] == 25
    assert c["title_agreement"] == pytest.approx(1.0)
    assert c["unsafe_claim_agreement"] == pytest.approx(1.0)


def test_comparer_reports_an_unstable_set_as_unstable():
    """Two different proposals must be reported as two, with no agreement."""
    a = dps.stub_debrief()
    b = dps.stub_debrief()
    b["next_experiment"]["title"] = "something else entirely"
    b["unsafe_claims"] = [{"claim": "a different unsafe claim", "why": "x",
                            "action": "y"}]
    c = dps.compare_debriefs([a, b])
    assert c["n_distinct_titles"] == 2
    assert c["title_agreement"] == pytest.approx(0.5)
    assert c["n_distinct_unsafe_claims"] == 2
    assert c["n_stable_fields"] < len(dps.REQUIRED_FIELDS)


def test_comparer_counts_only_the_runs_that_have_every_field():
    """A run that drops the null or the falsifier is not a complete proposal."""
    complete = dps.stub_debrief()
    incomplete = dps.stub_debrief()
    del incomplete["next_experiment"]["falsifier"]
    c = dps.compare_debriefs([complete, incomplete])
    assert c["n_runs_with_all_required_fields"] == 1
    assert c["frac_runs_with_all_required_fields"] == pytest.approx(0.5)
    assert c["fields"]["falsifier"]["n_present"] == 1


def test_comparer_handles_an_empty_list():
    c = dps.compare_debriefs([])
    assert c["n"] == 0
    assert "error" in c


def test_comparer_handles_unparseable_debriefs():
    """A run the model could not produce a JSON object for is not a proposal."""
    c = dps.compare_debriefs([None, "not a dict", dps.stub_debrief()])
    assert c["n"] == 1
    assert c["n_distinct_titles"] == 1


# ---------------------------------------------------------------- the fields

def test_list_and_dict_fields_are_compared_as_text():
    """A field may be a string, a list or an object; it must compare stably.

    If the comparer only handled strings, a director that returns a list for
    `controls` on one run and a string on another would look unstable when it
    is not -- or worse, look stable when it is.
    """
    a = dps.stub_debrief()
    a["next_experiment"]["controls"] = ["control one", "control two"]
    b = dps.stub_debrief()
    b["next_experiment"]["controls"] = ["control one", "control two"]
    c = dps.compare_debriefs([a, b])
    assert c["fields"]["controls"]["stable"] is True
    assert c["fields"]["controls"]["n_distinct"] == 1

    d = dps.stub_debrief()
    d["next_experiment"]["controls"] = ["control one", "control THREE"]
    c2 = dps.compare_debriefs([a, d])
    assert c2["fields"]["controls"]["stable"] is False


def test_field_agreement_counts_distinct_values_per_field():
    vectors = [["t1", "h1"], ["t1", "h2"], ["t2", "h2"]]
    s = dps.summarise_agreement(vectors)
    assert s["0"]["n_distinct"] == 2
    assert s["0"]["agreement"] == pytest.approx(2 / 3)
    assert s["1"]["n_distinct"] == 2
    assert s["1"]["agreement"] == pytest.approx(2 / 3)


def test_field_agreement_of_no_vectors():
    assert "error" in dps.summarise_agreement([])


def test_required_fields_are_the_six_the_project_asked_for():
    """The six fields are the ones the debrief prompt demands.

    A proposal missing the null or the falsifier is not a test, and this
    check stops the list drifting silently.
    """
    assert set(dps.REQUIRED_FIELDS) == {
        "title", "hypothesis", "method", "controls", "null", "falsifier"}


# ------------------------------------------------------ the substance metric

def test_jaccard_is_one_for_the_same_text():
    assert dps.jaccard("cache pressure gate advantage",
                       "cache pressure gate advantage") == pytest.approx(1.0)


def test_jaccard_is_zero_for_disjoint_text():
    assert dps.jaccard("cache pressure hardware", "speech pitch tracker") == 0.0


def test_jaccard_separates_rewording_from_a_different_experiment():
    """The measurement that distinguishes "reworded" from "different".

    Four titles that all propose the same experiment in different words must
    score strictly higher against each other than any of them scores against
    a title for a DIFFERENT experiment. If this fails, the substance metric
    is measuring nothing and the "same experiment, different words" reading
    falls back to an eyeball.

    Note what the metric does and does not claim: the absolute overlap
    between the four is only 0.10-0.50, because "cache-pressure",
    "low-cache" and "cache-constrained" share only the word "cache". The
    separation is what is measured, not the magnitude.
    """
    same = ["Cache-pressure gate advantage test",
            "Gate-advantage on low-cache hardware",
            "Gate performance on low-memory target device",
            "Gate cost on cache-constrained hardware"]
    other = "Measure the spectral flatness of the room tone"
    sims = [dps.jaccard(a, b) for i, a in enumerate(same)
            for b in same[i + 1:]]
    cross = [dps.jaccard(a, other) for a in same]
    assert min(sims) > 0
    assert min(sims) > max(cross)
    # And the mean over the four is comfortably above the mean against the
    # unrelated experiment.
    assert sum(sims) / len(sims) > sum(cross) / len(cross)


def test_jaccard_ignores_filler_words():
    a = dps.jaccard("the gate cost on cache constrained hardware",
                    "gate cost cache constrained hardware")
    assert a == pytest.approx(1.0)


def test_pairwise_similarity_reports_median_and_extremes():
    s = dps.pairwise_similarity(["alpha beta gamma", "alpha beta gamma",
                                  "delta epsilon"])
    assert s["n"] == 3
    assert s["n_pairs"] == 3
    assert s["median"] is not None
    assert s["min"] < s["max"]


def test_pairwise_similarity_of_one_string_is_empty():
    s = dps.pairwise_similarity(["only one"])
    assert s["mean"] is None
    assert s["median"] is None


def test_pairwise_run_similarity_compares_runs_not_a_pooled_bag():
    """Two runs flagging the same two claims must score high on similarity.

    Pooling every claim from every run into one bag would measure the
    vocabulary of the corpus, not whether the runs agree. This test pins the
    difference: the two runs share their content, so they must score high.
    """
    a = {"unsafe_claims": [{"claim": "gain normalisation cuts drift 50x"},
                           {"claim": "the watcher runs in microseconds"}]}
    b = {"unsafe_claims": [{"claim": "gain normalisation cuts drift 50x"},
                           {"claim": "the watcher runs in microseconds"}]}
    c = {"unsafe_claims": [{"claim": "pitch cannot separate two speakers"}]}
    sim = dps.pairwise_run_similarity([a, b], lambda d: d["unsafe_claims"])
    assert sim["median"] == pytest.approx(1.0)
    assert sim["min"] == pytest.approx(1.0)
    mixed = dps.pairwise_run_similarity([a, c],
                                        lambda d: d["unsafe_claims"])
    assert mixed["min"] < 0.5
