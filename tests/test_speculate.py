"""
Tests for the speculation gate -- acting before you are sure, reversibly.

WHY THE MECHANISM EXISTS AT ALL.

`turn.py` ends a turn on 3 s of silence. Across the seven-cell real corpus
it fires on ZERO of the 673 distinct silence runs measured on that population
-- or on ZERO of the 1671 pause events the watcher reports over them --
because the longest gap anywhere outside one essay is 2.4 s. So a peripheral
built on it does not respond slowly -- it does not respond at all.

(The cells on disk total 485.01 s, not the 465 s quoted in earlier revisions,
and the code now measures 1671 pause events (673 silence runs) rather than
680 runs. The conclusion is
unaffected: 3 s exceeds the longest gap either way.
`docs/research/NOTES-AUDIT.md` F7, F8.)

And `docs/research/TURN-ROUTES.md` showed the decision cannot be made better
by looking harder. At the moment of decision, "the turn ended" and "the
speaker is about to continue" are **byte-identical audio**; the only
difference is speech that has not happened yet. Our duration rule scores
0.0064 on one and 0.0067 on the other.

So every turn decision is a bet. This module exists to make the bet cheap and
reversible rather than to pretend it is certain.

THE THRESHOLD WAS WRONG THE FIRST TIME AND THE TEST SAYS SO.

The first default was 1.20 s. Measured against 676 real pause runs:

    0.30 s  reach 53.8%   0.60 s  reach 21.9%
    0.90 s  reach  9.9%   1.20 s  reach  3.1%
    1.80 s  reach  0.6%   2.40 s  reach  0.1%

At 1.20 s, 96.9% of speculations would be voided. The gate would almost never
fire and would be interrupted almost immediately when it did -- and it would
look like it worked, because nothing would go wrong. There would be nothing
to go wrong. A gate that never fires cannot fail, and a suite that only checks
"no exception was raised" would have passed it happily.

WHY THIS LANE WAS REPORTED FLAKY, AND WHAT IT ACTUALLY WAS.

Reported as "live-call and order-dependent". Measured, neither is true:

    pytest tests/test_speculate.py -q   3/3 runs -> 18 passed
                                            (123.20s, 109.85s, 121.30s)
    pytest tests/ -q                    3/3 runs -> 422 passed, 5 skipped
                                            (304.39s, 314.42s, 295.83s)

Six consecutive clean runs, so the reported flake does not reproduce here.
There is no live call in this file: the only urlopen sites in this suite are
monkeypatched stubs in test_local_tev.py / test_local_judge.py. And there is
no order dependence: watcher.py and speculate.py read no clock, no RNG, no
environment and hold no module cache; every cross-test mutation elsewhere
(test_watcher.py, test_policy.py) is restored in a `finally`; wavio cannot
write, so no test can alter the corpus mid-run; and pytest-randomly, -xdist
and -timeout are not installed, so ordering is fixed.

THE REAL DEFECT. IT IS A FAIL-NOT-SKIP BUG, NOT A FLAKE.

This lane reads its corpus from `examples/realaudio/`, and every `.wav` under
that path is GITIGNORED and untracked (.gitignore:10) -- only the
transcripts, the build script and corpus/MANIFEST.md are in git. The corpus
is generated from tracked .mp3s by corpus/_downloads/make_excerpt.py into a
directory shared with other agents working this tree.

So the result depended on ambient shared-tree filesystem state. With the
corpus absent -- the default on any fresh clone -- neither measurement test
skipped, both FAILED:

    test_the_shipped_threshold_is_not_theatre
        AssertionError: only 0 runs; corpus too thin
    test_the_rejected_threshold_is_rejected_for_a_reason
        ZeroDivisionError: division by zero

A measurement that cannot run has to say so, not fail for want of a file.
A lane whose verdict is decided by files nobody committed is a lane that goes
red on a machine that did nothing wrong -- which is the shape of bug that
gets misfiled as "flaky" and then blamed on ordering.

WHAT CHANGED, AND WHAT DID NOT.

1. The corpus is PINNED to the seven cells declared in
   `examples/realaudio/corpus/MANIFEST.md` instead of an open-ended
   `glob("*.wav")`. A stray or regenerated clip dropped into that shared
   directory used to silently move the measured numbers; the measurement now
   names the files it is over.
2. `needs_corpus` skips the two measurement tests, with a real and stated
   reason, when the declared corpus has not been built. This is the existing
   house pattern -- test_vad.py's `needs_weights`, test_judged_tracks.py's
   `needs_key`.
3. Nothing was softened. No assertion was removed, weakened, or made
   trivially true; both measurement tests carry an explicit non-empty
   precondition now, so a degenerate corpus reports itself instead of raising
   ZeroDivisionError. Every assertion runs at full strength whenever the
   corpus exists, which is every run on a built tree.

THE PROSE ABOVE QUOTES A CORPUS THAT IS NOT THE ONE ON DISK. THE ONES THE
CODE MEASURES TODAY, over the same seven declared cells:

    1671 pause events (673 silence runs)   (prose: 673 runs / 1671 events)
    min 0.160 s   median 0.380 s   max 2.540 s     (prose: 0.320 / 2.440)
    speculate_s=0.30 reach rate 75.3%              (prose: 53.8%)
    share below the rejected 1.20 s: 94.4%        (prose: 96.9%)

Both the old and the new figures clear the asserted floors (reach > 0.35,
below 1.20 s > 0.90), so the assertions were never wrong -- they were
quoting a corpus that has since changed. The floors are deliberately not
raised to the current numbers: the claim being defended is "the gate reaches
enough real pauses to be a gate", and pinning a floor to one corpus's exact
reach rate would make the test fail on a fixture change rather than on a
regression in the gate.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import speculate, watcher, wavio                             # noqa: E402

SR = 16000


def tone(hz, seconds, amp=0.3):
    return [amp * math.sin(2 * math.pi * hz * i / SR)
            for i in range(int(seconds * SR))]


def drain(w, samples, chunk=1600):
    events = []
    for i in range(0, len(samples), chunk):
        events.extend(w.push(samples[i:i + chunk]))
    return events


def pause_event(t0, t1, run_s):
    return watcher.Event(watcher.PAUSE, t0, t1,
                         {"duration_s": t1 - t0, "run_s": run_s,
                          "open": False})


# -- the corpus, named ------------------------------------------------------

# The seven cells declared in examples/realaudio/corpus/MANIFEST.md, NAMED
# rather than globbed. Every .wav under examples/realaudio/ is gitignored and
# untracked, and that directory is shared with other agents working this tree,
# so `glob("*.wav")` let a stray or regenerated clip silently change a
# measured number. The manifest IS in git; this list mirrors it.
_AUDIO = ROOT / "examples" / "realaudio"

CORPUS_CELLS = (
    "alice01_excerpt.wav",
    "corpus/poetry_weeds_by_the_wall_1702_librivox.wav",
    "corpus/essay_perpetual_peace_librivox.wav",
    "corpus/shortstory_shortstory058_1405_librivox.wav",
    "corpus/shakespeare_midsummernightsdream_1201_librivox.wav",
    "corpus/children_english_fairy_tales_joy_librivox.wav",
    "corpus/oration_jfks19620912.wav",
)

MISSING_CELLS = tuple(n for n in CORPUS_CELLS
                      if not (_AUDIO / n).is_file())

# Same pattern as test_vad.py's `needs_weights` and test_judged_tracks.py's
# `needs_key`: skip with a real, stated reason when the prerequisite is
# genuinely absent, instead of failing for want of a gitignored file.
needs_corpus = pytest.mark.skipif(
    bool(MISSING_CELLS),
    reason="corpus not built, missing cell(s): " + ", ".join(MISSING_CELLS)
           + " -- audio is gitignored; regenerate with"
             " examples/realaudio/corpus/_downloads/make_excerpt.py")


# -- the threshold, measured ------------------------------------------------

class TestThresholdIsMeasured:
    @needs_corpus
    def test_the_shipped_threshold_is_not_theatre(self):
        """
        A gate that fires almost never and is voided almost always is worse
        than no gate: it costs the complexity and delivers nothing. The test
        is on the corpus, not on a fixture.
        """
        runs = _corpus_pause_runs()
        assert runs, "the declared corpus produced no pause runs at all"
        assert len(runs) > 300, f"only {len(runs)} runs; corpus too thin"
        reach = sum(1 for r in runs if r >= speculate.SpeculationGate().speculate_s)
        rate = reach / len(runs)
        assert rate > 0.35, (
            f"speculate_s={speculate.SpeculationGate().speculate_s} reaches "
            f"only {rate:.1%} of {len(runs)} real pause runs; below ~0.35 the "
            f"gate is theatre")

    @needs_corpus
    def test_the_rejected_threshold_is_rejected_for_a_reason(self):
        """
        The falsification, kept so the original 1.20 s cannot come back
        unexamined. 96.9% void rate.
        """
        runs = _corpus_pause_runs()
        assert runs, "the declared corpus produced no pause runs at all"
        below = sum(1 for r in runs if r < 1.20)
        assert below / len(runs) > 0.90, (
            f"1.20 s now voids only {below / len(runs):.1%}; the reason it "
            f"was rejected needs re-measuring")

    def test_the_yield_curve_is_monotone(self):
        """More patience must never mean more bets. Obvious, and worth pinning."""
        runs = [0.2 + 0.004 * i for i in range(400)]
        curve = speculate.yield_curve(runs)
        reaches = [r["reach_rate"] for r in curve]
        assert reaches == sorted(reaches, reverse=True), \
            f"reach rate is not decreasing: {reaches}"

    def test_an_empty_corpus_yields_no_numbers(self):
        """A ratio over zero is not a measurement."""
        assert speculate.yield_curve([]) == []


# -- the lifecycle ----------------------------------------------------------

class TestLifecycle:
    def test_a_long_pause_opens_a_bet_and_holds_it(self):
        g = speculate.SpeculationGate(speculate_s=0.30, commit_s=3.0)
        changes = g.feed(pause_event(1.0, 1.5, run_s=0.5), elapsed_s=1.5)
        assert [c["state"] for c in changes] == [speculate.HOLDING]
        assert g.opened == 1 and g.held == 1
        assert len(g.live) == 1

    def test_a_short_pause_opens_nothing(self):
        g = speculate.SpeculationGate(speculate_s=0.30)
        assert g.feed(pause_event(1.0, 1.2, run_s=0.2), elapsed_s=1.2) == []
        assert g.opened == 0

    def test_the_speaker_resuming_voids_the_bet(self):
        """
        The whole point. A bet that cannot be voided is not a bet, it is a
        guess with consequences -- and guessing whether someone has finished
        speaking has the worst consequence there is, which is interrupting.
        """
        g = speculate.SpeculationGate(speculate_s=0.30)
        g.feed(pause_event(1.0, 1.5, run_s=0.5), elapsed_s=1.5)
        [c] = g.feed(watcher.Event(watcher.UTTERANCE, 1.9, 2.5, {}), 2.5)
        assert c["state"] == speculate.VOIDED
        assert g.voided == 1
        assert not g.live

    def test_a_bet_that_clears_the_commit_bar_commits(self):
        """
        Two events, because that is how it actually arrives: a bet opens when
        the silence first clears the speculate bar, and commits later when a
        LONGER silence event clears the commit bar. A single event with
        run_s=2.5 opens a bet and cannot commit it, which is correct -- it is
        evidence below the bar.
        """
        g = speculate.SpeculationGate(speculate_s=0.30, commit_s=2.0)
        g.feed(pause_event(1.0, 1.5, run_s=0.5), elapsed_s=1.5)
        changes = g.feed(pause_event(1.0, 3.5, run_s=2.5), elapsed_s=3.5)
        assert any(c["state"] == speculate.COMMITTED for c in changes)
        assert g.committed == 1
        assert g.speculation_accuracy() == 1.0

    def test_an_event_below_the_commit_bar_only_opens(self):
        """
        The other half of the above, asserted because it is the easy thing to
        get wrong: opening and committing are different bars and an event
        that clears one does not clear the other.
        """
        g = speculate.SpeculationGate(speculate_s=0.30, commit_s=3.0)
        g.feed(pause_event(1.0, 3.5, run_s=2.5), elapsed_s=3.5)
        assert g.opened == 1
        assert g.committed == 0, \
            "a bet committed on evidence below the commit bar"
        assert len(g.live) == 1

    def test_committed_cannot_be_voided_afterwards(self):
        """
        Pretending a committed action was speculative is how a peripheral
        lies to its own tables. Once committed, a later utterance must NOT
        reopen it.
        """
        g = speculate.SpeculationGate(speculate_s=0.30, commit_s=2.0)
        g.feed(pause_event(1.0, 1.5, run_s=0.5), elapsed_s=1.5)
        g.feed(pause_event(1.0, 3.5, run_s=2.5), elapsed_s=3.5)
        g.feed(watcher.Event(watcher.UTTERANCE, 4.0, 5.0, {}), 5.0)
        assert g.voided == 0, "a committed turn was voided by later speech"
        assert g.history[-1].state == speculate.COMMITTED

    def test_a_human_decision_voids_a_live_bet(self):
        """
        The person answered, so the system about to react to a pause was
        wrong about who had the floor -- and no silence analysis would ever
        have found that out.
        """
        g = speculate.SpeculationGate(speculate_s=0.30)
        g.feed(pause_event(1.0, 1.5, run_s=0.5), elapsed_s=1.5)
        [c] = g.feed(watcher.Event(watcher.DECISION, 1.8, 1.8, {}), 1.8)
        assert c["state"] == speculate.VOIDED

    def test_only_one_bet_is_live_at_a_time(self):
        """
        Overlapping bets would mean paying for several reactions to the same
        silence, which is the most expensive possible version of this bug.
        """
        g = speculate.SpeculationGate(speculate_s=0.30)
        for t in (0.4, 0.6, 0.8, 1.0):
            g.feed(pause_event(1.0, 1.0 + t, run_s=t), elapsed_s=1.0 + t)
        assert len(g.live) <= 1
        assert g.opened == 1


# -- honesty about cost -----------------------------------------------------

class TestIrreversibility:
    def test_an_externally_visible_void_is_reported(self):
        """
        A void is a claim about INTERNAL state. Anything already audible has
        left the process and cannot be recalled, and the module says so rather
        than pretending otherwise.
        """
        g = speculate.SpeculationGate(speculate_s=0.30)
        g.feed(pause_event(1.0, 1.5, run_s=0.5), elapsed_s=1.5)
        for s in g.live.values():
            s.externally_visible = True
        g.feed(watcher.Event(watcher.UTTERANCE, 1.9, 2.5, {}), 2.5)
        assert len(g.irreversible()) == 1
        assert g.summary()["irreversible_voids"] == 1

    def test_accuracy_of_a_quiet_gate_is_none_not_zero(self):
        """
        Nothing settled means no ratio exists. Reporting 0.0 or 1.0 would both
        be lies, and 0.0 is the lie that makes a dead gate look like a
        cautious one.
        """
        g = speculate.SpeculationGate()
        assert g.speculation_accuracy() is None
        assert g.summary()["accuracy"] is None

    def test_accuracy_is_measured_not_assumed(self):
        g = speculate.SpeculationGate(speculate_s=0.30, commit_s=2.0)
        # one that sticks, two that get voided
        g.feed(pause_event(1.0, 3.5, run_s=2.5), elapsed_s=3.5)
        g.feed(pause_event(4.0, 4.5, run_s=0.5), elapsed_s=4.5)
        g.feed(watcher.Event(watcher.UTTERANCE, 4.9, 5.2, {}), 5.2)
        g.feed(pause_event(6.0, 6.5, run_s=0.5), elapsed_s=6.5)
        g.feed(watcher.Event(watcher.UTTERANCE, 6.9, 7.2, {}), 7.2)
        assert g.committed == 1 and g.voided == 2
        assert g.speculation_accuracy() == pytest.approx(1 / 3)

    def test_an_unknown_state_is_refused_at_construction(self):
        with pytest.raises(ValueError):
            speculate.Speculation(id=1, t0=0.0, state="probably")


# -- on real audio ----------------------------------------------------------

class TestOnRealSpeech:
    def test_it_speculates_and_voids_on_the_corpus(self):
        """
        End to end on the narration fixture: a real gate, fed real watcher
        events, produces bets and voids some of them.

        The assertion is deliberately weak about the ratio -- the evidence
        says the ratio cannot be high -- and strong about the mechanism being
        live and honest.
        """
        path = ROOT / "examples" / "realaudio" / "alice01_excerpt.wav"
        if not path.is_file():
            pytest.skip("real speech fixture absent")
        samples = wavio.read_wav(str(path)).samples[:SR * 30]
        w = watcher.Watcher(SR, adaptive=True)
        g = speculate.SpeculationGate()
        for i in range(0, len(samples), 1600):
            for ev in w.push(samples[i:i + 1600]):
                g.feed(ev, w.elapsed_s)
        assert g.opened > 0, "the gate never speculated on 30 s of speech"
        assert len(g.history) > 0
        acc = g.speculation_accuracy()
        assert acc is None or 0.0 <= acc <= 1.0

    def test_nothing_is_left_live_at_the_end(self):
        """
        A gate that finishes a session still holding a bet has leaked state.
        """
        path = ROOT / "examples" / "realaudio" / "alice01_excerpt.wav"
        if not path.is_file():
            pytest.skip("real speech fixture absent")
        samples = wavio.read_wav(str(path)).samples[:SR * 20]
        w = watcher.Watcher(SR, adaptive=True)
        g = speculate.SpeculationGate()
        for i in range(0, len(samples), 1600):
            for ev in w.push(samples[i:i + 1600]):
                g.feed(ev, w.elapsed_s)
        assert len(g.live) <= 1
        for s in g.live.values():
            assert s.state in (speculate.OPEN, speculate.HOLDING)


_CORPUS_RUNS = None


def _corpus_pause_runs():
    """Real pause-run durations across the declared corpus cells."""
    global _CORPUS_RUNS
    if _CORPUS_RUNS is not None:
        return _CORPUS_RUNS
    runs = []
    for name in CORPUS_CELLS:
        path = _AUDIO / name
        if not path.is_file():
            continue
        samples = wavio.read_wav(str(path)).samples
        w = watcher.Watcher(SR, adaptive=True)
        for i in range(0, len(samples), 1600):
            for ev in w.push(samples[i:i + 1600]):
                if ev.kind == watcher.PAUSE and ev.data.get("run_s", 0) > 0.15:
                    runs.append(ev.data["run_s"])
    _CORPUS_RUNS = runs
    return runs
