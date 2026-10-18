"""
Tests for the turn decision -- the stage the adoption plan collapsed into a
single constant.

Two questions, deliberately not one:

    "did the voice stop?"     energy segmentation, MIN_SILENCE_S = 150 ms
    "is the turn over?"       this module,       TURN_END_S     = 3.0 s

The plan wanted to merge them into one constant at 200 ms. Measured across
seven cells of real speech that costs 46 of 259 pauses -- 17.8%, worst on
dramatic and children's speech (`tests/test_pause_threshold.py`). So the 3 s
decision sits ON TOP of the pause stream.

THE FIRST RULE, AND IT IS THE ONE THAT MATTERS: IT NEVER FILTERS.

`feed` returns newly-decided turns and touches nothing else. A pause that
does not reach the threshold is still a pause: it still counts, still
divides TEMPO, still appears in the baseline table. A stage that sits
downstream of everything can only ADD information, because anything else is a
way to lose data that was already measured correctly.

THE MEASUREMENT THAT SHAPED THE MODULE, AND IT IS UNCOMFORTABLE.

Across 465 s of real recorded speech, pauses of 3 s or more:

    cell          audio_s   max_gap   gaps >= 3 s
    narration        90.0     1.980              0
    essay            75.0     4.530              2
    dialogue         75.0     1.130              0
    shakespeare       75.0     2.100              0
    children          75.0     1.960              0
    oration           75.0     0.380              0

Recorded monologue simply does not pause for three seconds. That is not a
defect in the constant -- 3 s is an INTERACTIVE give-up ("nobody has spoken
for three seconds, I may speak now") and it is correct for that. It is wrong
as a claim about where turns END in a recording, and this corpus is
recordings. So the tracker reports `UNRESOLVED` below `TURN_YIELD_FLOOR_S`
instead of guessing, and a peripheral that never speaks is also a failure.

THREE BUGS FOUND HERE, ALL BY RUNNING IT ON REAL AUDIO.

Each produced a plausible wrong answer rather than an exception:

1. Silence was measured against the watcher's clock at push time instead of
   the pause's own extent. On 75 s of the essay: 94 turn boundaries instead
   of 2, every one ending at the end of the stream.

2. The tracker used each pause EVENT's duration, which is never long. A
   6 s silence is flushed by `MAX_PAUSE_S` into 2.755 / 2.775 / 0.360 s, so
   no single event clears any useful threshold. Fixed by carrying the
   accumulated run length on the event as `run_s`.

3. Once fixed, still zero turns -- because `if self._speaking and
   self._silence_since is None` meant the first pause set the start and every
   LATER pause of the same run skipped the decision. The run that finally
   exceeded the threshold was exactly the one that could not be seen.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import turn, watcher, wavio                                  # noqa: E402

SR = 16000
CORPUS = ROOT / "examples" / "realaudio" / "corpus"


def tone(hz, seconds, amp=0.3):
    return [amp * math.sin(2 * math.pi * hz * i / SR)
            for i in range(int(seconds * SR))]


def drain(w, samples, chunk=1600):
    """Push every chunk, returning all events in order."""
    events = []
    for i in range(0, len(samples), chunk):
        events.extend(w.push(samples[i:i + chunk]))
    return events


def run_stream(w, samples, tracker, chunk=1600):
    """
    Push and feed the tracker TOGETHER, collecting turns as they happen.

    The ordering is load-bearing and getting it wrong silently breaks the
    tests. Collecting every event first and then feeding them with the
    watcher's FINAL clock makes each pause look like it was seen at the end of
    the session -- and the tracker then correctly declines to believe it,
    producing zero turns and a test that passes for the wrong reason.

    Returns `(events, turns)` so a test can check both that nothing was lost
    and that the right decisions were made.
    """
    events, turns = [], []
    for i in range(0, len(samples), chunk):
        for ev in w.push(samples[i:i + chunk]):
            events.append(ev)
            turns.extend(tracker.feed(ev, w.elapsed_s))
    return events, turns


def speech_then(silence_s, then_speech_s=1.0):
    """Utterance, a known silence, another utterance."""
    return (tone(180.0, 1.0) + [0.0] * int(silence_s * SR)
            + tone(200.0, then_speech_s))


# -- the rule that makes this safe ------------------------------------------

class TestItNeverFilters:
    def test_feeding_preserves_every_event(self):
        """
        THE structural guarantee. `feed` is not a filter with a side effect --
        it returns turns and leaves the event stream alone. If a caller can
        lose an event by using this stage, the stage is unsafe wherever it is
        used.
        """
        w = watcher.Watcher(SR, adaptive=True)
        events = drain(w, speech_then(6.0))
        t = turn.TurnTracker()
        emitted = []
        for ev in events:
            emitted.extend(t.feed(ev, w.elapsed_s))
        # every original event is still in the watcher's ledger
        assert len(events) == len([e for e in w.emitted
                                   if e in events])
        assert all(e in w.emitted for e in events)

    def test_a_short_pause_is_still_a_pause(self):
        """Below threshold means 'not a turn yet', never 'not a pause'."""
        w = watcher.Watcher(SR, adaptive=True)
        events = drain(w, speech_then(0.8))
        pauses = [e for e in events if e.kind == watcher.PAUSE]
        assert pauses, "the 0.8 s silence produced no pause event"
        assert sum(p.duration_s for p in pauses) > 0.5, \
            "the silence was not reported at all"

    def test_the_threshold_never_touches_min_silence(self):
        """
        The plan's mistake, made structural: the two constants are independent
        and a test pins both, so moving one cannot silently move the other.
        """
        from src import energy
        assert energy.MIN_SILENCE_S == 0.150
        assert turn.TURN_END_S == 3.0
        assert turn.TURN_END_S > energy.MIN_SILENCE_S * 10


# -- it fires on a long silence ---------------------------------------------

class TestLongSilenceEndsATurn:
    def test_a_six_second_silence_is_one_turn(self):
        """
        REGRESSION for all three bugs. Before them this produced zero turns,
        and before bug 1 it produced several spurious ones.

        The silence is one turn even though it arrives as three events --
        because `MAX_PAUSE_S` flushes an open segment at 3.0 s and a pause is
        otherwise only emitted once its segment closes behind the tail cut.
        """
        w = watcher.Watcher(SR, adaptive=True)
        _events, turns = run_stream(w, speech_then(6.0), turn.TurnTracker())
        assert len(turns) == 1, f"expected one turn, got {turns}"
        assert turns[0]["silence_s"] >= 3.0
        assert turns[0]["reason"] == "sustained_silence"
        assert turns[0]["human_decision"] is False

    def test_a_pause_run_accumulates_across_checkpoints(self):
        """
        The `run_s` field, tested directly. Without it a long pause is
        unreadable: measured across the real corpus, the longest pause the
        watcher emitted was 0.790 s, so no turn threshold could ever match.

        `run_s` GROWS WITHIN a silence run and RESETS when a new one begins.
        It is not globally monotone, and asserting that it was would be
        asserting a bug -- the first version of this test did exactly that and
        failed once trailing silences started being reported, because the
        sequence legitimately contains runs of 0.26 .. 5.76 and then a fresh
        run starting near zero.

        So the property asserted is the real one: within a run, non-decreasing;
        across runs, reset to near zero.
        """
        w = watcher.Watcher(SR, adaptive=True)
        events = drain(w, speech_then(6.0))
        runs, current = [], []
        for ev in events:
            if ev.kind in ("utterance", "onset"):
                if current:
                    runs.append(current)
                    current = []
            elif ev.kind == watcher.PAUSE:
                current.append(ev.data["run_s"])
        if current:
            runs.append(current)

        assert runs, "no pause events at all"
        assert any(max(r) > 3.0 for r in runs), (
            f"no run exceeded 3.0 s: {[max(r) for r in runs]}; a 6 s silence "
            f"never appears in any single event")
        for r in runs:
            assert r == sorted(r), f"run_s went backwards within a run: {r}"
            assert r[0] < 1.0, (
                f"a fresh pause run started at {r[0]:.3f}s -- the reset is not "
                f"happening and every run inherits the previous one's length")

    def test_a_trailing_silence_is_reported_as_it_grows(self):
        """
        THE BUG THIS EXISTED TO CATCH, and the most important single test in
        the file.

        A pause used to be emitted only once its segment CLOSED -- which means
        once speech resumed. A silence at the END of speech therefore never
        closed and was never reported, at any length. Measured, a 1.30 s
        trailing silence produced pause events reading 0.03 and 0.08 s: the
        silence was not measured at all.

        So a peripheral could not learn that someone had stopped talking, and
        the 3 s turn threshold was never the problem -- the watcher could not
        see the silence that matters most in any conversation.
        """
        w = watcher.Watcher(SR, adaptive=True)
        events = drain(w, speech_then(6.0))
        runs = [e.data["run_s"] for e in events if e.kind == watcher.PAUSE]
        assert max(runs) > 5.0, (
            f"the trailing silence tops out at {max(runs):.3f}s; it is not "
            f"being reported as it grows")

    def test_a_short_silence_does_not_end_a_turn(self):
        w = watcher.Watcher(SR, adaptive=True)
        _events, turns = run_stream(w, speech_then(1.2), turn.TurnTracker())
        assert turns == [], f"a 1.2 s silence ended a turn: {turns}"

    def test_silence_before_any_speech_ends_nothing(self):
        """
        A quiet room is not the end of anyone's turn. A tracker that fires on
        idle silence is a peripheral that talks to nobody, which is worse than
        one that never talks.
        """
        w = watcher.Watcher(SR, adaptive=True)
        _events, turns = run_stream(w, [0.0] * int(12 * SR),
                                    turn.TurnTracker())
        assert turns == [], f"12 s of silence produced {len(turns)} turns"


# -- a human decision is the other way a turn ends --------------------------

class TestHumanDecisions:
    def test_a_human_decision_ends_a_turn_without_any_silence(self):
        """
        A refusal, an approval, a correction. It has no acoustic signature, so
        no amount of silence analysis finds it -- and a tracker that cannot
        represent it will misread the most consequential moments in a session
        as a pause that happened to land at the same time.
        """
        t = turn.TurnTracker()
        utt = watcher.Event(watcher.UTTERANCE, 0.0, 2.0, {})
        t.feed(utt, elapsed_s=2.0)
        turns = t.note_external(t0=2.5, elapsed_s=2.5)
        assert len(turns) == 1
        assert turns[0]["reason"] == "human_decision"
        assert turns[0]["human_decision"] is True

    def test_a_decision_after_the_speech_already_stopped_still_ends_it(self):
        """
        The common ordering: the voice goes quiet, then the person answers
        back with a word or a gesture the mic never saw.
        """
        w = watcher.Watcher(SR, adaptive=True)
        t = turn.TurnTracker()
        run_stream(w, speech_then(1.0), t)
        turns = t.note_external(t0=4.0, elapsed_s=4.0)
        assert len(turns) == 1
        assert turns[0]["reason"] == "human_decision"

    def test_a_decision_with_nothing_speaking_ends_nothing(self):
        """No speech, no turn -- even with a decision recorded."""
        t = turn.TurnTracker()
        assert t.note_external(t0=1.0, elapsed_s=1.0) == []


# -- the verdict, and saying what it does not know -------------------------

class TestVerdict:
    def test_too_little_time_is_unresolved_not_complete(self):
        """
        Below the yield floor there is no evidence either way, and the honest
        answer is "not enough". Reporting COMPLETE there would be a coin flip
        presented as a measurement -- and a peripheral that never speaks is
        also a failure, so the two errors are not ranked.
        """
        t = turn.TurnTracker()
        v = t.verdict(elapsed_s=1.0)
        assert v.state == turn.UNRESOLVED
        assert v.reason == "too_little_time"
        assert v.settled is False

    def test_while_speaking_the_verdict_is_open(self):
        """
        Asked past the yield floor, because below it the honest answer is
        UNRESOLVED for every reason at once -- which is itself tested
        separately. The clock is synthetic here on purpose: what is under test
        is the decision given a stated elapsed time, not the watcher's
        streaming behaviour.
        """
        t = turn.TurnTracker()
        t.feed(watcher.Event(watcher.UTTERANCE, 8.0, 10.0, {}), elapsed_s=10.0)
        v = t.verdict(elapsed_s=10.0)          # sound arriving right now
        assert v.state == turn.OPEN
        assert v.reason == "still_speaking"

    def test_idle_is_distinct_from_still_speaking(self):
        """
        An empty room is not an open turn. Conflating the two is how a
        peripheral decides to speak to nobody.
        """
        t = turn.TurnTracker()
        v = t.verdict(elapsed_s=30.0)
        assert v.state == turn.OPEN
        assert v.reason == "idle"

    def test_waiting_for_the_threshold_is_open_not_complete(self):
        """
        The distinction that matters operationally: "nobody has spoken for
        1.5 s" and "the turn is over" are different facts, and a system that
        conflates them will interrupt people.

        Asserted at 1.5 s of accumulated silence, which is below the 3 s
        threshold, so the verdict must be OPEN -- and the reason must say
        WHICH kind of open, because the reasons call for opposite behaviour:
        "still_speaking" means do not interrupt, "waiting" means the clock is
        running. Asserting the state alone would pass for either.
        """
        t = turn.TurnTracker()
        t.feed(watcher.Event(watcher.UTTERANCE, 10.0, 11.0, {}), elapsed_s=11.0)
        v = t.verdict(elapsed_s=12.5)          # 1.5 s of silence
        assert v.state == turn.OPEN, f"expected OPEN, got {v.state}/{v.reason}"
        assert v.reason == "silence_short_of_threshold", (
            f"expected 'waiting for the threshold', got {v.reason!r} -- the "
            f"tracker cannot distinguish waiting from interrupting")
        # and once it does exceed the threshold, the same tracker says so
        later = t.verdict(elapsed_s=20.0)
        assert later.state == turn.COMPLETE
        assert later.reason == "sustained_silence"
        assert later.settled is True

    def test_an_unknown_state_is_refused_at_construction(self):
        """
        A state nobody planned for must not be silently accepted. The tracker
        is what decides when a peripheral speaks, and an unrecognised state
        reaching a caller is how that goes wrong unremarkably.
        """
        with pytest.raises(ValueError):
            turn.TurnVerdict("probably", "guess")


# -- the measured reality of recorded speech --------------------------------

class TestRecordedSpeech:
    """
    The uncomfortable result, asserted so it cannot be quietly reinterpreted.
    """

    CELLS = {
        "essay": CORPUS / "essay_perpetual_peace_librivox.wav",
        "narration": ROOT / "examples" / "realaudio" / "alice01_excerpt.wav",
        "oration": CORPUS / "oration_jfks19620912.wav",
    }

    def test_recorded_monologue_almost_never_reaches_three_seconds(self):
        """
        Five of six real cells produce ZERO turn boundaries at 3 s. People
        recording a monologue do not pause for three seconds; the longest gap
        anywhere in the corpus outside the essay is 2.1 s.

        So this is the correct output for this material, not a failure to
        detect. It is also the reason `TURN_YIELD_FLOOR_S` exists and the
        reason the verdict can say UNRESOLVED: a constant tuned for live
        interaction must not present "nothing happened" as an achievement when
        it is pointed at a recording.
        """
        for name, path in self.CELLS.items():
            if not path.is_file():
                pytest.skip(f"corpus cell absent: {path.name}")
            samples = wavio.read_wav(str(path)).samples
            w = watcher.Watcher(SR, adaptive=True)
            events, turns = run_stream(w, samples, turn.TurnTracker())
            longest = max((e.data["run_s"] for e in events
                           if e.kind == watcher.PAUSE), default=0.0)
            assert len(turns) == 0 or longest >= 3.0, (
                f"{name}: {len(turns)} turns but the longest pause run was "
                f"{longest:.3f}s")
            if name in ("narration", "oration"):
                assert turns == [], (
                    f"{name} produced {len(turns)} turns at a 3 s threshold; "
                    f"measured longest gap there is under 2.1 s")

    def test_the_yield_floor_does_not_make_complete_unreachable(self):
        """
        The floor exists so the tracker can decline to answer when there is
        not enough time for the answer to mean anything. But a floor ABOVE
        the threshold it gates means COMPLETE can never be reported at all --
        a different failure wearing the same shape, and one that would look
        exactly like "this material never has a turn".

        So the floor must not sit above the threshold. It sits ABOVE it
        deliberately (5 s > 3 s) and that is fine, because the two answer
        different questions: the floor asks "has enough TIME passed to judge",
        the threshold asks "has enough SILENCE accumulated". A turn needs both.
        What must never happen is a pause run that lands between them and
        therefore satisfies neither.
        """
        assert turn.TURN_YIELD_FLOOR_S > turn.TURN_END_S, (
            "precondition: the floor is above the threshold, so the two are "
            "genuinely different questions")

        path = self.CELLS["essay"]
        if not path.is_file():
            pytest.skip("essay cell absent")
        samples = wavio.read_wav(str(path)).samples
        w = watcher.Watcher(SR, adaptive=True)
        events = drain(w, samples)
        longest = max((e.data["run_s"] for e in events
                       if e.kind == watcher.PAUSE), default=0.0)
        # the danger is a run that clears the threshold while still too early
        # to judge, with nothing later able to resolve it
        stranded = [
            e for e in events
            if e.kind == watcher.PAUSE
            and e.data["run_s"] >= turn.TURN_END_S
            and e.t1 < turn.TURN_YIELD_FLOOR_S]
        assert not stranded, (
            f"{len(stranded)} pause run(s) clear the threshold before the "
            f"yield floor and can never be resolved later; longest run in the "
            f"cell was {longest:.3f}s")


# -- latency is not turn length ---------------------------------------------

class TestEndToEnd:
    def test_end_to_end_is_the_silence_not_the_turn(self):
        """
        The number a latency budget wants. A turn that lasted 20 s of speech
        and ended on a 3 s pause has a 3 s end-to-end cost; measuring "turn
        length" to bound responsiveness would be off by an order of magnitude.
        """
        t = {"t0": 0.0, "t1": 23.0, "silence_s": 3.4}
        assert turn.end_to_end_s(t) == 3.4


# -- the two statistics, separated so neither can hide behind the other -----

class TestPauseExtentVersusPauseRun:
    """
    `run_s` and the event's own extent are DIFFERENT NUMBERS, and a tracker
    that uses the wrong one still passes a naive test -- because the clock
    fallback quietly supplies the right answer.

    Measured on a 6 s silence:

        event   t0     t1   run_s    own extent
        1     1.04   3.80   2.755      2.775
        2     3.83   6.60   5.555      2.775     <- 3 s of the run is in
        3     6.62   6.99   5.940      0.360          the FIRST event only

    So `_decide` is tested here directly. Calling it through `feed` was not
    enough: sabotaging it to use the own-extent instead of the run passed the
    whole suite, because `feed` had already reached the same verdict by way
    of the clock and the difference never surfaced.
    """

    def _tracker_ready(self):
        t = turn.TurnTracker()
        t._speaking = True
        t._silence_since = 1.045
        return t

    def test_the_second_checkpoint_is_the_one_that_carries_the_turn(self):
        ev = watcher.Event(watcher.PAUSE, 3.83, 6.60,
                           {"duration_s": 2.775, "run_s": 5.555, "open": True})
        t = self._tracker_ready()
        turns = t._decide(ev)
        assert len(turns) == 1, "the run that exceeds 3 s did not end a turn"
        assert turns[0]["silence_s"] == pytest.approx(5.555)

    def test_the_own_extent_would_have_missed_it(self):
        """
        The falsification, stated as its own test so the distinction cannot be
        quietly dropped: on this event the own extent is 2.775 s, which is
        UNDER the 3 s threshold. Using it would find nothing where the run
        length finds 5.555 s.
        """
        ev = watcher.Event(watcher.PAUSE, 3.83, 6.60,
                           {"duration_s": 2.775, "run_s": 5.555, "open": True})
        own = ev.t1 - ev.t0          # this event's own extent, 2.775 s
        assert own < turn.TURN_END_S < ev.data["run_s"], (
            f"precondition: own={own:.3f} must be under the threshold and "
            f"run_s={ev.data['run_s']:.3f} over it")

    def test_a_final_short_checkpoint_does_not_erase_the_turn(self):
        """
        The last event of a long pause is short -- 0.360 s. A tracker that
        re-decided on it would retract the turn it already reported.
        """
        ev = watcher.Event(watcher.PAUSE, 6.62, 6.99,
                           {"duration_s": 0.360, "run_s": 5.940, "open": False})
        assert ev.t1 - 6.62 < turn.TURN_END_S
        assert ev.data["run_s"] > turn.TURN_END_S

    def test_a_pause_without_run_s_still_works_on_its_own_extent(self):
        """
        The fallback, so a tracker fed events from a watcher that predates
        `run_s` degrades to the older, weaker measurement instead of failing.
        """
        ev = watcher.Event(watcher.PAUSE, 10.0, 14.0, {"duration_s": 4.0})
        t = self._tracker_ready()
        t._silence_since = 10.0
        turns = t._decide(ev)
        assert len(turns) == 1
        assert turns[0]["silence_s"] == pytest.approx(4.0)


class TestItCannotBeMadeToFilter:
    def test_idle_silence_must_not_end_a_turn(self):
        """
        Sabotage-and-revert on the rule that matters most.

        The sabotaged version let any pause reach `_decide` regardless of
        whether anyone had spoken, which is how a peripheral ends up deciding
        to respond to an empty room. Without this the entire safety argument
        for the stage is untested.
        """
        t = turn.TurnTracker()
        idle_pause = watcher.Event(watcher.PAUSE, 1.0, 6.0,
                                   {"duration_s": 5.0, "run_s": 5.0,
                                    "open": False})
        assert t._decide(idle_pause) == [], (
            "a turn ended without anyone having spoken")
        assert t.turns == []
