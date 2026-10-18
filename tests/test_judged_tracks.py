"""
Phase 2: the four JUDGED tracks, and the two ways this got wrong first.

A calibrated probability is a continuous controller value, which is exactly
what a MIDI CC is. TIMBRE/INTENTION/ATTENTION/GESTURE are how Jev plays an
instrument rather than routes a decision.

These tests do NOT need a network or an API key for the arithmetic, and they
deliberately do not need one: the two real defects found while building this
were both pure scale errors, and both would have passed against a mocked client
that returned plausible numbers.

WHAT THIS FILE DOES NOT PROVE, and it is a long list:

  * Nothing about CALIBRATION. Whether a stated 0.9 happens 90% of the time is
    unmeasured here. `confidence` on a Choice is a distribution-spread
    statistic, not a calibrated probability, and this suite checks that the CC
    is sourced from `probabilities[chosen]` rather than `confidence` -- it does
    NOT check that either is calibrated. An agent that can act autonomously
    needs the real ECE, and this file does not have it.
  * Nothing about real speech. The integration tests type text; they do not
    transcribe audio, so transcription error is entirely out of scope.
  * Nothing about whether the four questions are WELL-POSED. decision-ledger's
    pass_detector exists because an ill-posed question still returns confident
    numbers, and this suite has no equivalent instrument.
  * Nothing about latency under load. The integration test records elapsed time
    as a printed observation, and asserts only that the four tracks come from
    ONE call -- because four round trips in a voice loop would be an
    architectural failure regardless of how fast each one is.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import judged                       # noqa: E402
from src.judged import (CC, ATTENTION_CHOICE, GESTURE_CHOICE, INTENTION_SCALE,  # noqa: E402
                        TIMBRE_SCALE, _scale, _scale_position)

HAS_KEY = bool(os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFEAI_KEY"))
needs_key = pytest.mark.skipif(not HAS_KEY, reason="no TYPESAFE_API_KEY configured")


# -- THE REGRESSION THAT MATTERS -------------------------------------------

class TestScorePositionIsNotAProbability:
    """
    `ScoreAnswer.score` is a float POSITION on a 0..(n_levels-1) scale.

    Measured live against jev-1.14.0 with INTENTION_SCALE (4 levels):

        "hey"                          score 0.44   -> should be LOW
        "can you pull the last commit" score 2.34   -> should be MID-HIGH
        "STOP THE DEPLOY RIGHT NOW"    score 3.00   -> should be MAX

    The first implementation called _scale(score) -- treating 2.34 as a
    probability, multiplying by 127, and clamping to 127. So EVERY score at or
    above 1.0 rendered as maximum intensity, and a polite request came out at
    the same full tilt as an emergency. Both values were in range, so nothing
    downstream could see the difference.

    These assertions fail against that implementation and pass against this one.
    """

    N = len(INTENTION_SCALE)      # 4

    def test_bottom_of_scale_is_near_zero(self):
        assert _scale_position(0.0, self.N) == 0

    def test_top_of_scale_is_full(self):
        assert _scale_position(self.N - 1, self.N) == 127

    def test_midpoint_is_midway_not_maximum(self):
        """The exact value the bug produced wrongly."""
        half = _scale_position(1.5, self.N)      # 1.5/3 = 0.5 -> 0.5*127 = 63.5
        assert half == 64, half                  # round-half-up
        assert half != 127, "clamped: still treating position as a probability"

    def test_a_positive_position_is_not_already_maximum(self):
        """2.34 of 4 levels is 78%, not 100%."""
        v = _scale_position(2.34, self.N)
        assert v == 99, v
        assert v < 127

    def test_is_monotonic_in_the_position(self):
        vals = [_scale_position(p / 10, self.N) for p in range(0, 31)]
        assert vals == sorted(vals), vals

    def test_is_strictly_increasing_where_it_should_be(self):
        """Two adjacent levels must not collapse onto the same CC."""
        assert _scale_position(0.0, self.N) < _scale_position(1.0, self.N)
        assert _scale_position(1.0, self.N) < _scale_position(2.0, self.N)
        assert _scale_position(2.0, self.N) < _scale_position(3.0, self.N)

    def test_degenerate_scale_returns_zero_rather_than_dividing_by_zero(self):
        assert _scale_position(1.0, 1) == 0

    def test_none_score_is_zero_not_a_crash(self):
        assert _scale_position(None, self.N) == 0


# -- probability scaling ----------------------------------------------------

class TestProbabilityScale:
    def test_probability_maps_to_full_range(self):
        assert _scale(0.0) == 0
        assert _scale(1.0) == 127
        assert _scale(0.5) == 64      # 63.5 rounds to 64

    def test_out_of_range_is_clamped_not_wrapped(self):
        assert _scale(1.5) == 127
        assert _scale(-0.5) == 0

    def test_none_is_zero(self):
        assert _scale(None) == 0


# -- the CC is the probability of the chosen option, not `confidence` -------

class TestCcIsNotConfidence:
    """
    A Choice answer carries `confidence`, and it is very tempting to write
    `CC74 = answer.confidence`. The docs define confidence as a function of how
    the distribution is SPREAD -- a flat shape is low confidence regardless of
    which option leads.

    So `confidence` answers "how unambiguous was this?", not "how likely is
    this?". For a controller we need the second one. Using confidence would
    systematically under-drive exactly the ambiguous cases, which are the ones
    where a character controller most wants to hold back rather than commit --
    and it would invert the relationship between the number and the evidence.
    """

    def test_the_two_differ_on_a_realistic_distribution(self):
        # From the live docs' own example: refund leads at 0.40 with a
        # three-way split, and confidence is 0.20.
        probs = {"replacement": 0.34, "refund": 0.40, "information": 0.02,
                 "exchange": 0.24}
        chosen = max(probs, key=probs.get)
        cc_from_prob = _scale(probs[chosen])
        assert chosen == "refund"
        # 0.40 -> 51. If someone had used `confidence` here they'd get 25.
        assert cc_from_prob == 51

    def test_a_flat_distribution_does_not_drive_the_controller_to_zero(self):
        """The failure mode: high uncertainty must reduce confidence, not p."""
        probs = {"a": 0.33, "b": 0.33, "c": 0.34}
        chosen = max(probs, key=probs.get)
        assert _scale(probs[chosen]) >= 40, "a near-uniform draw must still drive"


# -- the question set, without a network ------------------------------------

class TestQuestions:
    def test_all_four_tracks_are_asked_in_one_call(self):
        """Four round trips in a voice loop would be an architectural failure."""
        q = judged.build_questions()
        assert set(q) == {"timbre", "intention", "attention", "gesture"}
        assert len(q) == 4

    def test_spectra_use_score_and_categories_use_choice(self):
        """Choice is for a fixed option set; Score is for a position on a scale."""
        from typesafe_sdk import Choice, Score
        q = judged.build_questions()
        assert isinstance(q["timbre"], Score), "a spectrum is not a fixed option set"
        assert isinstance(q["intention"], Score)
        assert isinstance(q["attention"], Choice)
        assert isinstance(q["gesture"], Choice)

    def test_every_option_is_described(self):
        """An undescribed option is a label the model has to guess at."""
        for name, criteria in (("attention", ATTENTION_CHOICE),
                               ("gesture", GESTURE_CHOICE)):
            for opt, desc in criteria.items():
                assert desc, f"{name}.{opt} has no description"

    def test_gesture_options_are_mutually_distinguishable(self):
        """The docs warn that confusable options need explicit contrast."""
        for opt, desc in GESTURE_CHOICE.items():
            assert not desc.lower().startswith(opt.lower()), \
                f"{opt} description just repeats its own name"

    def test_spectrum_levels_are_ordered_and_described(self):
        for scale in (TIMBRE_SCALE, INTENTION_SCALE):
            assert len(scale) >= 3, "a two-level scale is a Choice in disguise"
            for lvl in scale:
                assert ":" in lvl, f"level needs a name and a description: {lvl!r}"


# -- track identities --------------------------------------------------------

class TestTracks:
    def test_all_four_tracks_are_covered(self):
        assert set(CC) == {"TIMBRE", "INTENTION", "ATTENTION", "GESTURE"}

    def test_cc_numbers_match_the_slackwater_convention(self):
        # slackwater-perception: INTENTION CC74, ATTENTION CC91,
        # GESTURE CC80-87, TIMBRE CC71-76.
        assert CC["INTENTION"] == 74
        assert CC["ATTENTION"] == 91
        assert 80 <= CC["GESTURE"] <= 87
        assert 71 <= CC["TIMBRE"] <= 76

    def test_every_cc_is_in_midi_range(self):
        for track, cc in CC.items():
            assert 0 <= cc <= 127, track

    def test_channels_are_distinct(self):
        assert len(set(judged.CHANNEL.values())) == len(judged.CHANNEL)

    def test_model_is_named_never_the_floating_alias(self):
        """jev-latest resolves to 1.13.0 here and 1.14.0 is jev-preview."""
        assert judged.DEFAULT_MODEL != "jev-latest", \
            "a floating alias makes every number unattributable"
        assert any(ch.isdigit() for ch in judged.DEFAULT_MODEL)


# -- degradation -------------------------------------------------------------

class Boom:
    """A client that fails the way a flaky network does."""

    def system_one(self, **kw):
        raise RuntimeError("network down")


class TestDegradation:
    def test_missing_key_returns_unset_tracks_rather_than_raising(self):
        """A voice loop that dies on a network failure is worse than a silent one."""
        old = {k: os.environ.pop(k, None)
               for k in ("TYPESAFE_API_KEY", "TYPESAFEAI_KEY")}
        try:
            readings, model = judged.read_tracks("hello")
        finally:
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v
        assert len(readings) == 4
        assert all(not r.ok for r in readings)
        assert all(r.value == 0 for r in readings)
        assert all("key" in r.detail["why"] for r in readings)

    def test_a_raising_client_degrades_rather_than_propagating(self):
        readings, _ = judged.read_tracks("hello", client=Boom())
        assert all(not r.ok for r in readings)
        assert all(r.detail["why"] == "RuntimeError" for r in readings)

    def test_unset_track_is_distinguishable_from_a_measured_zero(self):
        """Same reasoning as PitchTracker's hz=None vs 0.0."""
        readings, _ = judged.read_tracks("hi", client=Boom())
        assert all(r.ok is False for r in readings)


# -- against the live model --------------------------------------------------

class FakeScore:
    """Shaped like ScoreAnswer. `score` is a POSITION, not a probability."""

    def __init__(self, score, n):
        self.score = score
        self.confidence = 0.66
        self.legend = {i: f"level {i}" for i in range(n)}
        self.probabilities = {i: (0.66 if i == int(score) else 0.34 / (n - 1))
                              for i in range(n)}


class FakeChoice:
    def __init__(self, choice, probs):
        self.choice = choice
        self.probabilities = probs
        self.confidence = 0.42


class FakeResponse:
    def __init__(self, score, n):
        self.model = "jev-fake-9.9.9"
        self.answers = {
            "timbre": FakeScore(score, len(TIMBRE_SCALE)),
            "intention": FakeScore(score, n),
            "attention": FakeChoice("listener", {"listener": 0.7, "task": 0.3}),
            "gesture": FakeChoice("question", {"question": 0.8, "command": 0.2}),
        }


class FakeClient:
    def __init__(self, score, n):
        self.score, self.n = score, n

    def system_one(self, **kw):
        return FakeResponse(self.score, self.n)


class TestCallSiteNotJustHelper:
    """
    The negative control caught this one, and it is the same blindness as
    Phase 0: testing a helper in isolation while the CALLER is wrong.

    The first version of TestScorePositionIsNotAProbability asserted that
    _scale_position(1.5, 4) == 64. That is true and it kept passing -- while
    read_tracks() called _scale(score) instead, so every live score >= 1.0
    still rendered as 127. Reverting the call site changed nothing the suite
    could see: 26 passed before, 26 passed after.

    So these tests drive read_tracks() end to end with a fake client returning
    a realistic ScoreAnswer. That is the only assertion that could have caught
    it, and it is the one that matters.
    """

    def test_a_high_mid_score_does_not_render_as_maximum(self):
        """2.34 of 4 levels is 78% -> CC 99. The bug rendered 127."""
        readings, _ = judged.read_tracks("x", client=FakeClient(2.34, 4))
        val = next(r.value for r in readings if r.track == "INTENTION")
        assert val == 99, f"{val} -- the call site is treating a position as a probability"

    def test_a_near_max_score_still_renders_below_maximum(self):
        readings, _ = judged.read_tracks("x", client=FakeClient(2.5, 4))
        val = next(r.value for r in readings if r.track == "INTENTION")
        assert 100 <= val < 127, val

    def test_bottom_and_top_are_the_ends(self):
        low, _ = judged.read_tracks("x", client=FakeClient(0.0, 4))
        high, _ = judged.read_tracks("x", client=FakeClient(3.0, 4))
        v = lambda rs: next(r.value for r in rs if r.track == "INTENTION")   # noqa: E731
        assert v(low) == 0
        assert v(high) == 127

    def test_choice_tracks_use_the_option_probability(self):
        """0.8 -> 102, not the spread-confidence of 0.42 -> 53."""
        readings, _ = judged.read_tracks("x", client=FakeClient(1.0, 4))
        val = next(r.value for r in readings if r.track == "GESTURE")
        assert val == 102, val

    def test_every_reading_carries_the_resolved_model(self):
        readings, model = judged.read_tracks("x", client=FakeClient(1.0, 4))
        assert model == "jev-fake-9.9.9"
        assert all(r.model == model for r in readings)


@needs_key
class TestLive:
    def test_urgency_is_ordered(self):
        """A polite request must not outrank an emergency.

        This is the assertion the scaling bug broke: both used to land on 127.
        """
        polite, _ = judged.read_tracks("could you please pull up the last commit")
        urgent, _ = judged.read_tracks("STOP THE DEPLOY RIGHT NOW")
        idle, _ = judged.read_tracks("hey")

        def val(rs, t):
            return next(r.value for r in rs if r.track == t)

        assert val(urgent, "INTENTION") > val(polite, "INTENTION"), \
            "an emergency scored no higher than a polite request"
        assert val(polite, "INTENTION") > val(idle, "INTENTION")
        assert val(urgent, "INTENTION") <= 127
        assert val(idle, "INTENTION") < 100

    def test_commands_and_greetings_are_distinguished(self):
        cmd, _ = judged.read_tracks("delete the staging database now")
        hi, _ = judged.read_tracks("hey")
        g = lambda rs: next(r.detail["choice"] for r in rs if r.track == "GESTURE")
        assert g(cmd) == "command", g(cmd)
        assert g(hi) == "greeting", g(hi)

    def test_all_four_tracks_come_from_one_call(self):
        calls = []

        class Counting:
            def __init__(self, inner):
                self.inner = inner

            def system_one(self, **kw):
                calls.append(kw)
                from typesafe_sdk import TypeSafeClient
                return TypeSafeClient().system_one(**kw)

        readings, model = judged.read_tracks("what broke the build?", client=Counting(None))
        assert len(calls) == 1, f"{len(calls)} round trips for four tracks"
        assert len(readings) == 4
        assert len(calls[0]["questions"]) == 4

    def test_model_is_recorded_on_every_reading(self):
        readings, resolved = judged.read_tracks("hello there")
        assert all(r.model == resolved for r in readings)
        assert resolved, "an unattributed number is not a measurement"

    def test_latency_is_measured_not_assumed(self):
        """Printed, not asserted: the budget depends on the network."""
        t0 = time.monotonic()
        judged.read_tracks("hey")
        ms = (time.monotonic() - t0) * 1000
        print(f"\n  four tracks, one call: {ms:.0f} ms")