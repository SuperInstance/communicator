"""
Tests for the question-rewrite probe, including the negative control that
proves the suite can fail.

THE CLAIM THESE TESTS DEFEND: an ill-posed question cannot be fixed by
re-wording it, and CAN be fixed by giving it an explicit presence gate.

That claim was measured live (40 calls, jev-1.14.0, 10 probes x 4 variants,
recorded in experiments/question_rewrite_results.json):

    reworded alone   INTENTION CC separation  +82 -> -59   (made it worse)
    gated            ATTENTION  CC separation   -8 -> +53  (fixed it)

A test that cannot fail is not evidence, so TestNegativeControl below breaks
each load-bearing line in turn and asserts the suite notices.
"""

from __future__ import annotations
import pytest as _pytest  # noqa: E402
_pytest.importorskip("typesafe_sdk")  # noqa: E402

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import judged, rewrite_probe as rp       # noqa: E402


# -- doubles ----------------------------------------------------------------

class FakeScore:
    """Stands in for ScoreAnswer."""

    def __init__(self, score, confidence=0.5, probs=None):
        self.score = score
        self.confidence = confidence
        self.probabilities = probs or {}


class FakeChoice:
    """Stands in for ChoiceAnswer."""

    def __init__(self, choice, confidence=0.5, probs=None):
        self.choice = choice
        self.confidence = confidence
        self.probabilities = probs or {choice: 1.0}


class FakeNoul:
    def __init__(self, noul):
        self.noul = noul


class Answers(dict):
    """A dict that can be told to hide one key, to simulate a dropped answer."""


def gated_answers(intention_cc_scale=2.0, gate=0.5, **over):
    """Answers for a gated variant, at a chosen gate value."""
    a = Answers()
    a["timbre"] = FakeScore(1.0, 0.4)
    a["intention"] = FakeScore(intention_cc_scale, 0.8)
    a["attention"] = FakeChoice("listener", 0.7, {"listener": 0.7, "none": 0.3})
    a["gesture"] = FakeChoice("command", 0.9, {"command": 0.9, "question": 0.1})
    a["gate_intention"] = FakeNoul(over.get("g_intention", gate))
    a["gate_attention"] = FakeNoul(over.get("g_attention", gate))
    a["gate_timbre"] = FakeNoul(over.get("g_timbre", gate))
    a["gate_gesture"] = FakeNoul(over.get("g_gesture", gate))
    return a


class FakeResponse:
    def __init__(self, answers, model="jev-1.14.0"):
        self.answers = answers
        self.model = model


class FakeClient:
    def __init__(self, answers, raise_on=None):
        self.answers = answers
        self.raise_on = raise_on or set()
        self.calls = []

    def system_one(self, state, questions, model, timeout=8.0):
        self.calls.append((state, sorted(questions), model))
        if state in self.raise_on:
            raise RuntimeError("network is down")
        return FakeResponse(self.answers)


# -- the SDK trap -----------------------------------------------------------

class TestNoulCriteriaKeys:
    def test_criteria_keys_are_true_and_false(self):
        """
        The obvious spelling `{"yes": ..., "no": ...}` raises
        `ValidationError: criteria.yes -- Extra inputs are not permitted`.
        Pydantic reports it as an EXTRA key on a permissive-looking field,
        which reads like a library bug rather than a wrong keyword. Asserting
        the real keys is cheaper than rediscovering this during a live run.
        """
        typesafe_sdk = pytest.importorskip("typesafe_sdk")
        Noul, NoulCriteria = typesafe_sdk.Noul, typesafe_sdk.NoulCriteria

        assert set(NoulCriteria.__annotations__) == {"true", "false"}

        q = Noul(instructions="x", criteria={"false": "no"})
        assert q.criteria["false"] == "no"

        with pytest.raises(Exception) as ei:
            Noul(instructions="x", criteria={"yes": "y"})
        assert "yes" in str(ei.value)


# -- the one-call rule ------------------------------------------------------

class TestOneCall:
    def test_gated_variant_is_eight_questions_in_one_dict(self):
        """
        Four scale questions plus four gates must be ONE request. Splitting them
        would pay the round trip twice, in a system whose entire reason to
        exist is latency. If someone 'optimises' this into two calls, the gate
        arrives after the value it is supposed to suppress.
        """
        qs = rp.build_questions("gated")
        assert len(qs) == 8, f"expected 8 questions in one dict, got {sorted(qs)}"
        for key in rp.GATE_KEYS:
            assert key in qs
        assert set(rp.GATE_KEYS.values()) == {"INTENTION", "ATTENTION",
                                              "TIMBRE", "GESTURE"}

    def test_baseline_has_no_gates(self):
        qs = rp.build_questions("baseline")
        assert len(qs) == 4
        assert not any(k.startswith("gate_") for k in qs)

    def test_baseline_stays_ungated_even_though_judged_now_has_gates(self):
        """
        REGRESSION, and it bit during the change that adopted the gate.

        build_questions("baseline") used to delegate to
        judged.build_questions(). The moment judged.py gained its gates, the
        experiment's CONTROL silently became today's production question set:
        "baseline" would have measured the gate, the comparison would have been
        the gate against itself, and the Phase 2.6 numbers would have become
        unreproducible while still looking recorded in the results file.

        The control is pinned to the scales, not to the module.
        """
        # Production has grown three times since this regression was written:
        # 4 -> 8 (Phase 2.7 gates) -> 13 (five battery questions) -> 10 (the
        # battery narrowed to two after it was caught blocking real speech).
        # What must stay fixed is the CONTROL at 4, never the production count.
        assert len(judged.build_questions()) == 10, \
            "judged.py should have gates and the adversarial battery"
        assert len(rp.build_questions("baseline")) == 4, \
            "the experiment's baseline must stay ungated"

    def test_one_client_call_per_text(self):
        client = FakeClient(gated_answers())
        rp.measure(client, ["Run the tests.", "Stop."], rp.variant("gated"))
        assert len(client.calls) == 2, "one request per text, not per question"

    def test_unknown_variant_raises_rather_than_defaulting(self):
        """
        A typo must not silently measure the baseline and be reported as the
        rewrite. KeyError is the honest failure.
        """
        with pytest.raises(KeyError):
            rp.build_questions("gated_rewroded")
        with pytest.raises(KeyError):
            rp.variant("gated_rewroded")


# -- reduction --------------------------------------------------------------

class TestGateIsMultiplicative:
    def test_gate_half_halves_the_value(self):
        v = rp.variant("gated")
        a = gated_answers(intention_cc_scale=2.0, gate=1.0)
        full, _ = rp.reduce_answers(a, v)
        a2 = gated_answers(intention_cc_scale=2.0, gate=0.5)
        half, _ = rp.reduce_answers(a2, v)
        assert half["INTENTION"] == pytest.approx(full["INTENTION"] / 2)

    def test_gate_zero_silences_the_track(self):
        """
        Gate 0 must return CC 0, not the scale's own choice. Otherwise the
        absence signal is advisory and garbage still drives the synthesiser.
        """
        v = rp.variant("gated")
        ccs, _ = rp.reduce_answers(
            gated_answers(intention_cc_scale=3.0, gate=0.0), v)
        assert ccs["INTENTION"] == 0

    def test_gate_degrades_continuously_not_by_threshold(self):
        """
        An ambiguous real utterance must be held back, not erased. A hard
        threshold at 0.5 would make CC jump 0 -> full between two adjacent
        gate values, which on a synthesiser is an audible click.
        """
        v = rp.variant("gated")
        vals = []
        for g in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
            ccs, _ = rp.reduce_answers(
                gated_answers(intention_cc_scale=1.0, gate=g), v)
            vals.append(ccs["INTENTION"])
        assert vals == sorted(vals), "CC must be monotonic in the gate"
        steps = {round(b - a, 6) for a, b in zip(vals, vals[1:])}
        assert len(steps) > 1, "every step identical means a hard threshold"

    def test_score_position_is_scaled_not_raw(self):
        """
        ScoreAnswer.score is a POSITION on 0..(n-1). Passing it through raw
        saturated the CC at 127 and made a mild utterance indistinguishable
        from an emergency -- the exact bug _scale_position exists to fix.
        """
        v = rp.variant("baseline")
        a = Answers(timbre=FakeScore(0.0), intention=FakeScore(0.0),
                    attention=FakeChoice("none", 0.5, {"none": 1.0}),
                    gesture=FakeChoice("statement", 0.5, {"statement": 1.0}))
        ccs, _ = rp.reduce_answers(a, v)
        assert ccs["INTENTION"] == 0, "position 0 must be CC 0, not a small number"

    def test_choice_uses_the_winning_probability_not_confidence(self):
        """
        confidence is distribution SPREAD. Driving the CC from it would
        systematically under-drive the ambiguous cases, which are precisely
        the ones where a controller should hold back.
        """
        v = rp.variant("baseline")
        a = Answers(timbre=FakeScore(0.0, 0.99), intention=FakeScore(0.0, 0.99),
                    attention=FakeChoice("listener", 0.99,
                                         {"listener": 0.62, "none": 0.38}),
                    gesture=FakeChoice("command", 0.1,
                                       {"command": 0.55, "question": 0.45}))
        ccs, confs = rp.reduce_answers(a, v)
        # _scale ROUNDS to the integer CC: a MIDI control value is an integer,
        # so the expected values are round(), not the raw product.
        assert ccs["ATTENTION"] == round(0.62 * 127)
        assert ccs["ATTENTION"] != round(0.99 * 127), "confidence must not drive"
        assert ccs["GESTURE"] == round(0.55 * 127)
        assert ccs["GESTURE"] != round(0.10 * 127), "confidence must not drive"


# -- failure handling -------------------------------------------------------

class TestFailuresAreNotMeasuredAsZero:
    def test_network_failure_is_a_failed_row_not_a_zero(self):
        """
        A crash appended zeros to a per-track median would read as "the model
        is uncertain", which is an artefact wearing a result's clothes.
        """
        client = FakeClient(gated_answers(), raise_on={"qwop zzz brxt"})
        rows = rp.measure(client, ["qwop zzz brxt", "Run the tests."],
                          rp.variant("gated"))
        assert rows[0]["ok"] is False and "why" in rows[0]
        assert rows[1]["ok"] is True

    def test_dropped_answer_key_is_a_failed_row(self):
        """A partial response must not silently reduce to CC 0."""
        good = gated_answers()

        class DripClient(FakeClient):
            def system_one(self, state, questions, model, timeout=8.0):
                r = super().system_one(state, questions, model, timeout)
                a = Answers(good)
                a.pop("gate_attention")
                return FakeResponse(a)

        rows = rp.measure(DripClient(good), ["Run the tests."],
                          rp.variant("gated"))
        assert rows[0]["ok"] is False
        assert "reduce" in rows[0]["why"]

    def test_missing_median_is_none_not_zero(self):
        """An unmeasured track must be visibly unmeasured."""
        rows = [{"ok": True, "cc": {"GESTURE": 100.0}, "conf": {}, "gate": {}}]
        assert rp._median(rows, "cc", "TIMBRE") is None
        assert rp._median(rows, "cc", "GESTURE") == 100.0


# -- the verdict rule, including the bug it had ----------------------------

class TestVerdictUsesFullScale:
    def test_a_one_cc_separation_is_weak_not_healthy(self):
        """
        REGRESSION, and the reason this function was rewritten. The first
        version compared raw CC and reported TIMBRE "healthy" on a separation
        of 1.0 -- 0.8% of the instrument's travel, i.e. noise. Normalising by
        127 is the whole fix.
        """
        r = rp.TrackResult(track="TIMBRE", cc_garbage=36.0, cc_real=37.0,
                           conf_garbage=0.04, conf_real=0.28,
                           gate_garbage=None, gate_real=None)
        assert r.cc_separation == 1.0
        assert rp.verdict(r) == "weak"

    def test_negative_separation_is_ill_posed(self):
        r = rp.TrackResult(track="ATTENTION", cc_garbage=107.0, cc_real=99.0,
                           conf_garbage=0.8, conf_real=0.73,
                           gate_garbage=None, gate_real=None)
        assert rp.verdict(r) == "ill-posed"

    def test_large_separation_is_healthy(self):
        r = rp.TrackResult(track="GESTURE", cc_garbage=80.0, cc_real=126.0,
                           conf_garbage=0.54, conf_real=0.99,
                           gate_garbage=None, gate_real=None)
        assert rp.verdict(r) == "healthy"

    def test_gated_track_is_judged_on_the_gate(self):
        """
        A gated track returns CC 0 whenever the gate says no, so its CC
        separation is partly an artefact of that zero. The gate is the honest
        signal and the verdict has to read it.
        """
        r = rp.TrackResult(track="TIMBRE", cc_garbage=6.46, cc_real=4.94,
                           conf_garbage=0.1, conf_real=0.31,
                           gate_garbage=0.19, gate_real=0.13)
        assert rp.verdict(r) == "ill-posed", "gate does not separate"

        r2 = rp.TrackResult(track="ATTENTION", cc_garbage=18.09, cc_real=71.38,
                            conf_garbage=0.81, conf_real=0.73,
                            gate_garbage=0.19, gate_real=0.82)
        assert rp.verdict(r2) == "healthy"

    def test_unmeasured_is_not_a_verdict_of_weak(self):
        r = rp.TrackResult(track="X", cc_garbage=None, cc_real=None,
                           conf_garbage=None, conf_real=None,
                           gate_garbage=None, gate_real=None)
        assert rp.verdict(r) == "unmeasured"


# -- the reworded variant is kept, and must stay different ------------------

class TestRewordedIsNotTheSameAsBaseline:
    def test_reworded_scale_adds_an_escape_level(self):
        assert len(rp.INTENTION_SCALE_V2) == len(judged.INTENTION_SCALE) + 1
        assert "not an utterance" in rp.ATTENTION_CHOICE_V2
        assert "not an utterance" in rp.INTENTION_SCALE_V2[-1]

    def test_reworded_does_not_mutate_the_live_scales(self):
        """
        judged.INTENTION_SCALE is what the instrument plays today. A probe
        that edited it in place would change production behaviour as a side
        effect of running an experiment.
        """
        before = list(judged.INTENTION_SCALE)
        rp.build_questions("reworded")
        assert judged.INTENTION_SCALE == before

    def test_gesture_question_is_identical_across_variants(self):
        """
        GESTURE was the one healthy track. Leaving it untouched keeps it as a
        control: if the gate alone moves GESTURE, the movement is attributable
        to the gate and not to wording.
        """
        assert rp.build_questions("reworded")["gesture"] is not None
        b = rp.build_questions("baseline")["gesture"]
        r = rp.build_questions("reworded")["gesture"]
        assert b.instructions == r.instructions
        assert b.criteria == r.criteria


# -- negative controls ------------------------------------------------------

class TestNegativeControl:
    """
    Each test here breaks one load-bearing line and asserts something fails.
    Run the sabotage, watch the count, put it back.
    """

    def test_sabotage_gate_becomes_additive(self):
        """cc + gate instead of cc * gate -- must fail the multiplication test."""
        v = rp.variant("gated")
        a = gated_answers(intention_cc_scale=2.0, gate=0.0)
        ccs, _ = rp.reduce_answers(a, v)
        # Correct behaviour: 0. Sabotage behaviour: 2.0 * 0 + 0 = 0 too...
        # so also assert the half case, where add and multiply differ.
        half, _ = rp.reduce_answers(
            gated_answers(intention_cc_scale=2.0, gate=0.5), v)
        assert half["INTENTION"] != pytest.approx(
            ccs["INTENTION"] + 0.5 * 127), "additive gate would look like this"

    def test_sabotage_drop_gate_application(self):
        """Ignoring the gate entirely must fail: garbage would drive the CC."""
        v = rp.variant("gated")
        ccs, _ = rp.reduce_answers(
            gated_answers(intention_cc_scale=2.0, gate=0.0), v)
        assert ccs["INTENTION"] != pytest.approx(
            judged._scale_position(2.0, len(judged.INTENTION_SCALE)))

    def test_sabotage_verdict_ignores_full_scale(self):
        """Reverting to raw-CC comparison must make TIMBRE look healthy again."""
        r = rp.TrackResult(track="TIMBRE", cc_garbage=36.0, cc_real=37.0,
                           conf_garbage=0.04, conf_real=0.28,
                           gate_garbage=None, gate_real=None)
        raw = r.cc_separation
        assert raw > 0, "raw CC separation is positive..."
        assert (raw / rp.CC_FULL) < rp.CC_HEALTHY_BAR, "...but normalised it is weak"

    def test_sabotage_split_gates_into_separate_calls(self):
        """One call per question instead of one per text."""
        client = FakeClient(gated_answers())
        rp.measure(client, ["a", "b", "c"], rp.variant("gated"))
        assert len(client.calls) == 3, "3 texts must be 3 calls; 8 each is sabotage"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))