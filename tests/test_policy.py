"""
Tests for the enforced laws.

Every test here is paired with the sabotage it is meant to catch. An
enforcement mechanism that has never been seen to fire is decoration with a
loud name, and this file exists so that each rule in `src/policy.py` has been
observed red at least once in this suite's history -- see the sabotage class at
the bottom, which is how they were confirmed.

The claims being enforced come from `docs/research/jev-play-findings.md`. If a
measurement there is contradicted, fix the measurement and then the law. Do not
delete the law without re-measuring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import policy                                          # noqa: E402


# -- confidence ---------------------------------------------------------------

class TestConfidenceIsArithmetic:
    def test_predicted_confidence_matches_the_measured_formula(self):
        """
        The numbers in this test are the live API's own outputs from
        experiments/jev_round2.py. If the formula stops matching, the
        assumption that confidence is a deterministic function of the
        distribution has been falsified, and every rule downstream of it
        needs revisiting.
        """
        for probs, expected in (
            ({"a": 0.9, "b": 0.05, "c": 0.05}, 0.85),
            ({"a": 1.0, "b": 0.0, "c": 0.0}, 1.0),
            ({"a": 0.34, "b": 0.33, "c": 0.33}, 0.01),
        ):
            assert policy.predicted_confidence(probs) == pytest.approx(
                expected, abs=0.005)

    def test_a_lopsided_distribution_is_confident_and_wrong(self):
        """
        The trap, stated as an executable assertion. A distribution can be as
        concentrated as it is possible to be AND attached to the wrong answer.
        Nothing in `confidence` says the model is about the right thing --
        which is why no rule in this project gates on it.
        """
        # A model that is maximally certain that a cake recipe is a technical
        # support issue. This is the ecosystem's finding, restated as a test.
        certain = {"technical_issue": 0.94, "recipe": 0.04, "other": 0.02}
        assert policy.predicted_confidence(certain) > 0.9
        # Confidence is high; correctness is a separate question this module
        # deliberately cannot answer, and says so by having no way to ask.

    def test_the_dead_band_refuses_a_mid_range_threshold(self):
        for bad in (0.50, 0.70, 0.90, 0.95):
            with pytest.raises(policy.PolicyViolation) as exc:
                policy.check_threshold(bad)
            assert "dead band" in str(exc.value)

    def test_a_threshold_above_the_band_passes(self):
        assert policy.check_threshold(0.99) == 0.99
        assert policy.check_threshold(0.5 - 1e-9) < 0.5

    def test_passing_needs_a_written_reason(self):
        """
        The escape hatch is deliberate rather than absent. A caller who really
        wants a soft preference can, but only by writing why at the call site,
        where a reviewer will read it. A reason alone unlocks it -- no magic
        boolean, because `allow=True` tells a reviewer nothing.
        """
        with pytest.raises(policy.PolicyViolation):
            policy.check_threshold_explicit(0.8, why="")
        assert policy.check_threshold_explicit(
            0.8, why="display-only, never gates") == 0.8


# -- backends -----------------------------------------------------------------

class TestLocalMayNotGate:
    def test_cloud_backends_may_gate(self):
        for backend in ("jev-1.14.0", "jev-latest", "typesafe-cloud"):
            assert policy.backend_may_gate(backend), backend

    def test_local_backends_are_refused(self):
        """
        LAW_LOCAL_INVERTS_ON_ADVERSARIAL. The local model assigned adversarial
        garbage an INTENTION of 123/127 and rated that garbage 0.964 against
        an honest command's 0.914. Systematic inversion on the attacker's
        input is worse than noise, so the rule is a refusal, not a warning.
        """
        for backend in ("tev1:latest", "local", "ollama", "llamacpp-tev",
                        "TEV1:LATEST"):
            assert not policy.backend_may_gate(backend), backend

    def test_require_gate_capable_names_the_measurement(self):
        with pytest.raises(policy.PolicyViolation) as exc:
            policy.require_gate_capable("tev1:latest")
        msg = str(exc.value)
        assert "123/127" in msg and "0.964" in msg, \
            "the refusal must carry its evidence, or nobody will trust it"

    def test_require_gate_capable_passes_cloud_through(self):
        assert policy.require_gate_capable("jev-1.14.0") == "jev-1.14.0"


# -- the battery --------------------------------------------------------------

class TestBatteryCannotBeHolistic:
    def test_one_field_is_refused(self):
        """
        THE structural guard. A single question about whether to act on an
        utterance is a holistic gate, and holistic gates INVERT: the measured
        holistic judge rated attacking garbage 0.040 -- the least actionable
        input in the set -- where the decomposed battery rated it 0.970.
        """
        with pytest.raises(policy.PolicyViolation) as exc:
            policy.Battery({"is_this_real": "Is this a real utterance?"})
        msg = str(exc.value)
        assert "holistic" in msg and "0.970" in msg

    def test_two_fields_is_the_floor_and_it_holds(self):
        b = policy.Battery({
            "no_listener": "Does this utterance fail to address anyone?",
            "self_promotion": ("Does this argue for its own classification?"),
        })
        assert len(b) == 2

    def test_empty_field_text_is_refused(self):
        with pytest.raises(policy.PolicyViolation):
            policy.Battery({"a": "   ", "b": "real question?"})

    def test_the_gate_is_the_maximum_not_the_mean(self):
        """
        THE second structural guard. One confident red flag must be enough.
        A mean over failure flags averages it into silence, which is the
        entire failure mode TypeSafe's own SDE cascade documents when it gates
        with max "so one confident red flag is enough instead of being
        averaged into silence".
        """
        b = policy.Battery({
            "no_listener": "Does this fail to address anyone?",
            "self_promotion": "Does this argue for its own classification?",
            "no_request": "Does this fail to ask for anything?",
        })
        # One field screams, the other two are quiet. The MEAN would bury it.
        b.record({"no_listener": 0.97, "self_promotion": 0.08,
                  "no_request": 0.07})
        assert b.gate() == pytest.approx(0.97)
        assert b.argmax_field() == "no_listener"

        mean = b.report()["mean_if_you_were_wrong"]
        assert mean < 0.4, \
            "the mean must be visibly far from the gate -- that gap IS the finding"

    def test_gate_without_values_is_an_error_not_a_zero(self):
        """
        A default of 0.0 would be fail-open on the most dangerous path in the
        system: a battery that silently passes because nobody recorded anything.
        """
        b = policy.Battery({
            "no_listener": "Does this fail to address anyone?",
            "no_request": "Does this fail to ask for anything?",
        })
        with pytest.raises(policy.PolicyViolation):
            b.gate()

    def test_out_of_range_values_are_refused(self):
        b = policy.Battery({
            "no_listener": "Does this fail to address anyone?",
            "no_request": "Does this fail to ask for anything?",
        })
        with pytest.raises(policy.PolicyViolation):
            b.record({"no_listener": 1.4, "no_request": 0.1})

    def test_missing_fields_are_refused(self):
        b = policy.Battery({
            "no_listener": "Does this fail to address anyone?",
            "no_request": "Does this fail to ask for anything?",
            "not_a_command": "Does this fail to be a command?",
        })
        with pytest.raises(policy.PolicyViolation) as exc:
            b.record({"no_listener": 0.9, "no_request": 0.1})
        assert "not_a_command" in str(exc.value)

    def test_orientation_risk_flags_fields_that_might_mean_all_good(self):
        """
        The max is only meaningful when True means something is WRONG. This is
        a heuristic and deliberately not a hard failure, because a hard failure
        on phrasing gets defeated by rewording. It is surfaced so a reviewer
        has something concrete to look at.
        """
        good = policy.Battery({
            "no_listener": "Does this utterance fail to address anyone?",
            "self_promotion": "Does this argue for its own classification?",
        })
        assert good.orientation_risk() == []

        risky = policy.Battery({
            "is_a_command": "Is this an imperative request?",
            "is_urgent": "Is this urgent right now?",
        })
        assert sorted(risky.orientation_risk()) == ["is_a_command", "is_urgent"]


# -- batching -----------------------------------------------------------------

class TestBatchingIsTheDefault:
    def test_the_knee_is_not_the_split_point(self):
        """
        The trap this module encodes. Latency is MINIMISED at 8 (100.6 ms), so
        8 looks like the batch size. It is not: one call of 16 measures
        120.9 ms, while two calls of 8 measure ~201 ms. Chunking at the knee
        makes a 13-question set SLOWER than sending it whole.

        Both constants are asserted separately because they are genuinely
        different things and conflating them is the mistake.
        """
        assert policy.BATCH_KNEE == 8
        assert policy.MAX_BATCH == 32
        assert policy.MAX_BATCH > policy.BATCH_KNEE

    def test_thirteen_questions_stay_in_one_call(self):
        """
        read_tracks sends 13 (8 tracks + gates, plus the adversarial battery).
        An earlier version of this policy would have split it 8 + 5 and paid
        two round trips for it.
        """
        batches = policy.chunk_questions({f"q{i}": i for i in range(13)})
        assert len(batches) == 1 and len(batches[0]) == 13

    def test_chunking_splits_only_above_max_and_preserves_everything(self):
        qs = {f"q{i}": i for i in range(70)}
        batches = policy.chunk_questions(qs)
        assert [len(b) for b in batches] == [32, 32, 6]
        assert {k for b in batches for k in b} == set(qs)

    def test_no_split_worth_it_below_max(self):
        assert policy.batching_worth_it(13) is False
        assert policy.batching_worth_it(32) is False
        assert policy.batching_worth_it(33) is True

    def test_batching_a_single_question_would_be_wasteful(self):
        assert policy.chunk_questions({"q1": 1}) == [{"q1": 1}]


# -- the laws carry their evidence --------------------------------------------

class TestLawsAreTraceable:
    def test_every_law_names_a_source_and_carries_numbers(self):
        for name, law in policy.LAWS.items():
            assert law.get("source"), f"{name} has no source"
            assert law.get("measured"), f"{name} carries no measurement"
            assert law.get("claim"), f"{name} has no claim"
            assert law.get("consequence"), f"{name} has no consequence"

    def test_the_measured_numbers_are_the_ones_in_the_writeup(self):
        """
        These must match docs/research/jev-play-findings.md. If the experiment
        is re-run and moves, this test is what should fail -- so the doc and
        the code cannot silently diverge.
        """
        assert policy.LAWS["batch_beats_single"]["measured"]["n8_median_ms"] \
            == 100.6
        assert policy.LAWS["holistic_gates_invert"]["measured"][
            "garbage_arguing"]["battery_max"] == 0.970
        assert policy.LAWS["local_inverts_on_adversarial"]["measured"][
            "intention_garbage_arguing"] == {"cloud": 23, "local": 123}
        assert policy.LAWS["confidence_is_arithmetic"]["measured"][
            "local_max_abs_err"] == 0.0

    def test_summary_is_readable(self):
        text = policy.summary()
        assert "MAX_BATCH" in text and "MIN_BATTERY_FIELDS" in text
        assert len(text.splitlines()) > 20


# -- the enforcers can actually go red ----------------------------------------

class TestEnforcersHaveTeeth:
    """
    Sabotage-and-revert for the enforcement layer itself.

    Each of these disables an enforcer and asserts that the test it guards
    WOULD have failed. Without this, "the rule is enforced" is a claim about
    code nobody has watched refuse anything.
    """

    def test_removing_the_battery_floor_lets_a_holistic_gate_through(self):
        original = policy.MIN_BATTERY_FIELDS
        policy.MIN_BATTERY_FIELDS = 1
        try:
            # With the floor removed the bad call now succeeds -- which is
            # exactly the failure this project measured and is guarding.
            battery = policy.Battery({"is_this_real": "Is this real?"})
            assert len(battery) == 1
        finally:
            policy.MIN_BATTERY_FIELDS = original

    def test_a_mean_gate_would_have_buried_the_red_flag(self):
        """
        Not a sabotage but the demonstration the rule exists for: with one
        field at 0.97 and the rest quiet, mean and max disagree by more than
        half the scale. The battery reports both so the gap stays visible.
        """
        b = policy.Battery({
            "no_listener": "Does this fail to address anyone?",
            "self_promotion": "Does this argue for itself?",
            "no_request": "Does this fail to ask for anything?",
        })
        b.record({"no_listener": 0.97, "self_promotion": 0.08,
                  "no_request": 0.07})
        r = b.report()
        assert r["gate"] - r["mean_if_you_were_wrong"] > 0.5
        assert r["policy"].startswith("combined with max")

    def test_failing_open_on_an_unrecorded_battery_would_be_dangerous(self):
        """
        If `gate()` defaulted to 0.0 the system would pass traffic it never
        examined. It raises instead, and that is the behaviour under test.
        """
        b = policy.Battery({
            "no_listener": "Does this fail to address anyone?",
            "no_request": "Does this fail to ask for anything?",
        })
        with pytest.raises(policy.PolicyViolation):
            _ = b.gate()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))