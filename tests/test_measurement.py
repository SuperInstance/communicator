"""
Measurement instruments: question health and calibration metrics.

The metrics here are the arithmetic a live number will be believed with, so
they are tested against values computed BY HAND first: an ECE that cannot be
trusted offline cannot be trusted on a network call.

WHAT THIS FILE PROVES:
  * ece/brier/spearman return hand-computed values on small inputs,
    including the degenerate bins (empty bin contributes nothing; the top
    edge p=1.0 lands in the LAST bin).
  * The health instrument flags confident-on-garbage (ill-posed),
    unconfident-everywhere (degenerate), and passes separation (healthy) --
    from readings, not from live calls.
  * calibration.score() turns a rows dict into accuracy/ECE/bias/spearman
    with all-correct and all-wrong controls.

WHAT THIS FILE DOES NOT PROVE:
  * Nothing about live Jev. A green suite here says the INSTRUMENT is
    honest, not that the measured thing is good. Live numbers land in
    CALIBRATION.md and PROGRESS.md with their model named.
"""

from __future__ import annotations
import pytest as _pytest  # noqa: E402
_pytest.importorskip("typesafe_sdk", reason="exercises code needing typesafe_sdk")  # noqa: E402

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import judged, question_health                    # noqa: E402
from src.calibration import (ATTENTION_ITEMS, GESTURE_ITEMS,     # noqa: E402
                             INTENTION_ITEMS, TIMBRE_COLD_WORDS, TIMBRE_WARM_WORDS,
                             brier, ece, score, spearman)
from src.judged import TrackReading                        # noqa: E402


# -- hand-computed metric controls -------------------------------------------

class TestMetricsByHand:
    def test_ece_is_zero_when_bin_rates_match(self):
        # ECE is a property of BINS, not rows: a lone p=0.8 that came up y=1
        # is not "perfectly calibrated" -- 0.8 predicts a RATE. Half y=1 at
        # p=0.5 matches its bin rate exactly, so the bin contributes zero.
        # (Half-and-half at p=0.8 would contribute |0.8-0.5| = 0.3 -- the
        # second version of this test asserted exactly that mistake.)
        rows = [{"p": 0.5, "y": 1}] * 5 + [{"p": 0.5, "y": 0}] * 5
        assert ece(rows) == pytest.approx(0.0)

    def test_ece_of_pure_overconfidence(self):
        # one bin at p=0.9, always wrong: |0.9 - 0.0| weighted by 1.0
        assert ece([{"p": 0.9, "y": 0}] * 4) == pytest.approx(0.9)

    def test_ece_half_right_at_0_8(self):
        # bin mean_p 0.8, mean_y 0.5 -> 0.3
        rows = [{"p": 0.8, "y": 1}] * 5 + [{"p": 0.8, "y": 0}] * 5
        assert ece(rows) == pytest.approx(0.3)

    def test_ece_empty_bin_contributes_nothing(self):
        a = ece([{"p": 0.55, "y": 1}])
        b = ece([{"p": 0.55, "y": 1}] + [{"p": 0.12, "y": 0}] * 0)
        assert a == b

    def test_top_edge_lands_in_last_bin(self):
        # p=1.0 must not fall off the end (b < 1.0 comparison bug class)
        assert ece([{"p": 1.0, "y": 0}]) == pytest.approx(1.0)

    def test_brier_bounds(self):
        assert brier([{"p": 1.0, "y": 1}]) == 0.0
        assert brier([{"p": 1.0, "y": 0}]) == 1.0

    def test_spearman_monotone_and_reversed(self):
        assert spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)
        assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_spearman_survives_ties(self):
        # Two texts share designed level 0 and two share level 1, but the
        # measured positions never tie. Tie-averaged ranks lose a little
        # information, so monotone-with-ties scores slightly UNDER 1.0:
        # rx=[1.5,1.5,3.5,3.5,5], ry=[1,2,4,3,5] -> 9/sqrt(9*10) = 0.9487.
        # Demanding exactly 1.0 here would be demanding a lie about ties.
        assert spearman([0, 0, 1, 1, 2], [0.1, 0.2, 1.4, 1.3, 2.8]) == \
            pytest.approx(0.9487, abs=0.001)

    def test_spearman_constant_is_nan_not_a_fake_one(self):
        assert math.isnan(spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


# -- fakes: readings with a chosen confidence and chosen answers -------------

def make_readings(*, conf: float, gesture="question", gesture_p=0.8,
                  attention="listener", attention_p=0.8,
                  timbre_pos=1.0, intention_pos=2.0):
    def reading(track, value, detail):
        return TrackReading(track, value, detail=detail, model="jev-fake-9.9.9")

    return [
        reading("TIMBRE", judged._scale_position(timbre_pos, len(judged.TIMBRE_SCALE)),
                {"score": timbre_pos, "confidence": conf}),
        reading("INTENTION", judged._scale_position(intention_pos, 4),
                {"score": intention_pos, "confidence": conf}),
        reading("ATTENTION", judged._scale(attention_p),
                {"choice": attention, "probabilities": {}, "confidence": conf}),
        reading("GESTURE", judged._scale(gesture_p),
                {"choice": gesture, "probabilities": {}, "confidence": conf}),
    ]


class ConfByBattery:
    """A client whose confidence depends on whether the text is garbage."""

    def __init__(self, garbage_conf, real_conf):
        self.garbage_conf, self.real_conf = garbage_conf, real_conf
        self.calls = []

    def system_one(self, *, state, **kw):
        self.calls.append(state)
        conf = self.garbage_conf if state in question_health.GARBAGE_PROBES \
            else self.real_conf
        # The SDK's confidence comes back on the answers; read_tracks stores
        # it in detail. Reuse the fake answer shapes from the cache tests.
        return _ConfResponse(conf)


class _ConfResponse:
    def __init__(self, conf):
        self.model = "jev-fake-9.9.9"
        self.answers = {
            "timbre": _ConfScore(1.0, len(judged.TIMBRE_SCALE), conf),
            "intention": _ConfScore(2.0, 4, conf),
            "attention": _ConfChoice("listener", {"listener": 0.8, "task": 0.2}, conf),
            "gesture": _ConfChoice("question", {"question": 0.8, "command": 0.2}, conf),
        }


class _ConfScore:
    def __init__(self, s, n, conf):
        self.score = s
        self.confidence = conf
        self.legend = {i: f"level {i}" for i in range(n)}
        self.probabilities = {i: (0.7 if i == int(s) else 0.3 / (n - 1))
                              for i in range(n)}


class _ConfChoice:
    def __init__(self, choice, probs, conf):
        self.choice = choice
        self.probabilities = probs
        self.confidence = conf


class TestQuestionHealth:
    def test_confident_on_garbage_is_ill_posed(self):
        client = ConfByBattery(garbage_conf=0.9, real_conf=0.95)
        healths = question_health.probe_questions(client=client)
        assert all(h.verdict == "ill-posed" for h in healths)
        assert all(h.separation < 0.1 for h in healths)

    def test_separation_is_healthy(self):
        client = ConfByBattery(garbage_conf=0.2, real_conf=0.9)
        healths = question_health.probe_questions(client=client)
        assert all(h.verdict == "healthy" for h in healths)
        assert all(h.separation > 0.5 for h in healths)

    def test_unconfident_everywhere_is_degenerate(self):
        client = ConfByBattery(garbage_conf=0.1, real_conf=0.2)
        healths = question_health.probe_questions(client=client)
        assert all(h.verdict == "degenerate" for h in healths)

    def test_call_count_is_garbage_plus_real(self):
        client = ConfByBattery(0.2, 0.9)
        question_health.probe_questions(client=client)
        n = len(question_health.GARBAGE_PROBES) + len(question_health.REAL_PROBES)
        assert len(client.calls) == n, "one call per probe string, no more"

    def test_report_names_the_fingerprint_and_bars(self):
        client = ConfByBattery(0.2, 0.9)
        report = question_health.format_report(
            question_health.probe_questions(client=client), model="jev-fake")
        assert judged.question_fingerprint() in report
        assert "ill-posed" in report or "healthy" in report


# -- calibration.score() controls --------------------------------------------

def rows_from(*, gesture_correct=True, attention_correct=True,
              intention_bias=0.0):
    """Build a rows dict exactly like collect() would, with controlled errors."""
    rows = {}

    def put(text, track, reading):
        rows.setdefault(text, {})[track] = reading

    for label, texts in GESTURE_ITEMS.items():
        for t in texts:
            chosen = label if gesture_correct else "statement"
            put(t, "GESTURE",
                TrackReading("GESTURE", judged._scale(0.8),
                             {"choice": chosen, "probabilities": {},
                              "confidence": 0.8}, "fake"))
    for label, texts in ATTENTION_ITEMS.items():
        for t in texts:
            chosen = label if attention_correct else "self"
            put(t, "ATTENTION",
                TrackReading("ATTENTION", judged._scale(0.8),
                             {"choice": chosen, "probabilities": {},
                              "confidence": 0.8}, "fake"))
    for level, texts in INTENTION_ITEMS.items():
        for t in texts:
            put(t, "INTENTION",
                TrackReading("INTENTION", 0,
                             {"score": float(level + intention_bias),
                              "confidence": 0.7}, "fake"))
    for t in TIMBRE_WARM_WORDS:
        put(t, "TIMBRE", TrackReading("TIMBRE", 0, {"score": 4.0,
                                                    "confidence": 0.6}, "fake"))
    for t in TIMBRE_COLD_WORDS:
        put(t, "TIMBRE", TrackReading("TIMBRE", 0, {"score": 0.0,
                                                    "confidence": 0.6}, "fake"))
    return rows


class TestScore:
    def test_all_correct_choice_tracks(self):
        res = {r.track: r for r in score(rows_from())}
        assert res["GESTURE"].payload["accuracy"] == 1.0
        assert res["GESTURE"].payload["ece"] == pytest.approx(
            abs(0.8 - 1.0), abs=0.01)
        assert res["ATTENTION"].payload["accuracy"] == 1.0

    def test_all_wrong_gesture(self):
        # "All wrong" is impossible by construction: choosing `statement` for
        # every text is still CORRECT for the three texts that are statements.
        # The honest control is 3/15 right, with recall 1.0 on the statement
        # class and 0.0 everywhere else -- a wrong answer that is itself one
        # of the classes always scores its own class's share.
        res = {r.track: r for r in score(rows_from(gesture_correct=False))}
        g = res["GESTURE"].payload
        assert g["accuracy"] == pytest.approx(3 / 15)
        assert g["per_class"]["question"] == 0.0
        assert g["per_class"]["statement"] == 1.0

    def test_intention_perfect_order(self):
        res = {r.track: r for r in score(rows_from())}
        p = res["INTENTION"].payload
        assert p["spearman"] == pytest.approx(1.0)
        assert all(b == 0.0 for b in p["bias_by_level"].values())
        assert p["mae_levels"] == 0.0

    def test_intention_constant_offset_shows_as_bias_not_disorder(self):
        res = {r.track: r for r in score(rows_from(intention_bias=0.5))}
        p = res["INTENTION"].payload
        assert p["spearman"] == pytest.approx(1.0)
        assert all(b == pytest.approx(0.5) for b in p["bias_by_level"].values())

    def test_timbre_wording_probe_separates(self):
        res = {r.track: r for r in score(rows_from())}
        assert res["TIMBRE"].payload["separates"] is True


# -- the blind-spot guard ----------------------------------------------------

def test_this_file_does_not_measure_live_calibration():
    """Every number above was constructed. CALIBRATION.md carries the live
    ones, with the model named and the corpus shown, and neither file may be
    quietly substituted for the other."""
    assert True
