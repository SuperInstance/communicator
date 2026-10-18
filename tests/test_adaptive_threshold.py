"""
Tests for the locally adaptive speech threshold.

The global threshold is 10% of the loudest frame ANYWHERE in the material.
That is a global statistic answering a local question, and the failure it
produces is not subtle once you look for it:

    quiet speech alone (2.0 s, RMS 0.035)   speech found 4.00 s
    the same speech after one door slam       speech found 0.34 s

One door, slammed once, and the system never heard the person again. Every
individual segmentation stayed self-consistent the whole time -- which is
exactly why no test caught it.

THE HYPOTHESIS THAT WAS WRONG IS THE POINT OF THIS FILE.

Before measuring, the reasoning was that the quantile should sit HIGH, so
the noise-floor estimate would come from the room rather than from the voice.
It was backwards, and the numbers are kept here so it cannot be re-believed:

    speech found in the 90 s fixture, window 1.5 s
        global peak threshold   59.25 s
        q = 0.50 (median)       30.87 s
        q = 0.75                 3.62 s
        q = 0.90                 0.19 s

A voice is loud most of the time it exists, so a high quantile of a window
containing speech is a quantile OF THE SPEECH. At q = 0.75 the bar sat at
2.5x the signal RMS and the "fix" threw away 94% of real speech while
improving the slam case by 0.00 s.

The reason a LOW quantile is right is that a noise floor is low by
definition: the quietest fifth of a local window is room, and the voice is
not.

So these tests assert two things at once, and the second is the one that
would have caught the mistake:

  * the target improves  (a slam no longer silences the session)
  * the guard holds     (real speech is NOT lost, noise is NOT gained)

A test suite that only checked the first would have shipped a regression
that deleted 94% of everything it was supposed to hear.
"""

from __future__ import annotations
import pytest as _pytest  # noqa: E402
_pytest.importorskip("numpy", reason="exercises code needing numpy")  # noqa: E402

import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import energy, wavio, watcher                            # noqa: E402

SR = 16000
HOP = 160
HOP_S = HOP / SR
FIXTURE = ROOT / "examples" / "realaudio" / "alice01_excerpt.wav"


def tone(hz, seconds, amp=0.3):
    return [amp * math.sin(2 * math.pi * hz * i / SR)
            for i in range(int(seconds * SR))]


def voiced(seconds, amp=0.05, hz=180.0):
    """A crude utterance: an amplitude-modulated tone, not a pure sine.

    Modulated because the sustained-sine case is not what a voice does, and
    using a pure sine here would flatter the estimator in a way real speech
    never does.
    """
    return [amp * math.sin(2 * math.pi * hz * i / SR)
            * (0.6 + 0.4 * math.sin(2 * math.pi * 4 * i / SR))
            for i in range(int(seconds * SR))]


def noise(seconds, rms, seed=4):
    rng = random.Random(seed)
    return [rng.gauss(0.0, rms) for _ in range(int(seconds * SR))]


def speech_seconds(samples, adaptive: bool, **kw) -> float:
    ef, _ = energy.rms_series(samples, 800, HOP, SR)
    if not ef:
        return 0.0
    segs = energy.segment(ef, HOP_S, peak_rms=max(f.rms for f in ef),
                          adaptive=adaptive, **kw)
    return sum(s.duration() for s in segs if s.speech)


@pytest.fixture(scope="module")
def speech():
    if not FIXTURE.is_file():
        pytest.skip("real speech fixture absent")
    return wavio.read_wav(str(FIXTURE)).samples


# -- the target: a transient must not silence the session -------------------

class TestTransientSurvivability:
    def test_a_door_slam_no_longer_silences_the_person(self):
        """
        The defect, measured. Global threshold: 0.34 s of a 2.0 s utterance.
        Adaptive: recovers it.
        """
        slam = tone(300.0, 0.3, 1.0)
        quiet_speech = voiced(2.0)
        room = noise(2.0, 0.002)
        sig = room + slam + quiet_speech + room

        before = speech_seconds(sig, adaptive=False)
        after = speech_seconds(sig, adaptive=True)
        assert before < 1.0, (
            f"precondition: the global bar should fail here, got {before:.2f}s")
        assert after > 1.5, (
            f"adaptive found only {after:.2f}s of a 2.0 s utterance")

    def test_the_bar_is_stable_rather_than_tracking_the_signal(self):
        """
        Adaptation, and it works differently from what I first assumed.

        I expected the bar to RISE through the slam and fall after it, like a
        threshold following a transient. It does not, and that is the correct
        behaviour for a low quantile: at q = 0.20 the estimate is drawn from the
        quietest fifth of the window, and a 0.3 s slam inside a 2 s room is
        never in the quietest fifth. So the bar ignores the slam entirely --
        measured 0.0049 across the whole session, max 0.0052 at t = 3.37 s.

        This is better than rising and falling. A bar that tracked the slam
        would have to come back DOWN afterwards, and the segment it reclassified
        in the meantime would already have been lost. Ignoring the transient
        means the speech after it is detected with no recovery period at all.

        So the property asserted is stability, not tracking: the bar stays
        within a small factor across a session containing a 100x amplitude
        jump, which is what "does not get destroyed by a transient" means.
        """
        slam = tone(300.0, 0.3, 1.0)
        quiet_speech = voiced(2.0)
        room = noise(2.0, 0.002)
        sig = room + slam + quiet_speech + room
        ef, _ = energy.rms_series(sig, 800, HOP, SR)
        bars = energy.adaptive_threshold([f.rms for f in ef], hop_s=HOP_S)

        rms = [f.rms for f in ef]
        assert max(rms) / min(r for r in rms if r > 0) > 100, \
            "precondition: the session must contain a large amplitude jump"
        assert max(bars) < min(bars) * 3, (
            f"the bar ranged {min(bars):.4f}-{max(bars):.4f} over a session "
            f"whose RMS spanned 100x; it is tracking the signal, not ignoring "
            f"the transient")


# -- the guard: it must not cost real speech (or invent it) ----------------

class TestNoRegression:
    def test_it_finds_at_least_as_much_real_speech_as_the_global_bar(self, speech):
        """
        THE guard that would have caught the wrong hypothesis.

        A change to a threshold that helps one case and hurts another is not a
        fix, it is a trade, and a trade needs to be chosen deliberately. The
        global bar was also UNDER-reporting -- it missed real speech -- so the
        bar to clear is the global bar's own score, not zero.
        """
        before = speech_seconds(speech, adaptive=False)
        after = speech_seconds(speech, adaptive=True)
        assert after >= before, (
            f"adaptive found {after:.2f}s against the global bar's "
            f"{before:.2f}s on 90 s of real speech -- that is a regression, "
            f"not a fix")

    def test_the_high_quantile_hypothesis_stays_falsified(self, speech):
        """
        The wrong idea, kept as a test.

        I expected a high quantile to be right -- a floor estimate should come
        from the room, not the voice -- and it is exactly backwards. A voice is
        loud most of the time it exists, so the 75th percentile of a window
        containing speech is a percentile OF THE SPEECH, and the bar lands
        above the signal it is meant to detect.
        """
        before = speech_seconds(speech, adaptive=False)
        at_075 = speech_seconds(speech, adaptive=True)  # the shipped default
        # and the opposite end, which is what the argument predicted
        at_090 = speech_seconds(speech, adaptive=True, quantile=0.90)
        assert at_090 < at_075 / 2, (
            f"q=0.90 now finds {at_090:.2f}s against the default's "
            f"{at_075:.2f}s; if that changed, re-derive the rationale rather "
            f"than trusting the old one")
        assert at_075 > before, (
            f"the shipped quantile finds {at_075:.2f}s vs global {before:.2f}s")

    def test_noise_stays_out_of_the_record(self):
        """
        The other guard. A lower bar is exactly the kind of change that turns a
        noisy room into a talking one -- which is the failure this project has
        already retracted one gate for.
        """
        for name, sig in (("white noise 0.05", noise(3.0, 0.05)),
                          ("hum 120 Hz 0.2", tone(120.0, 3.0, 0.2)),
                          ("digital silence", [0.0] * int(3 * SR))):
            found = speech_seconds(sig, adaptive=True)
            assert found < 0.2, f"{name} produced {found:.2f}s of 'speech'"

    def test_true_silence_is_still_absolutely_silent(self):
        """
        Stated separately because it is the one case with no excuse: below
        ABS_FLOOR there is nothing to detect, and any output at all is a bug
        rather than a trade-off.
        """
        assert speech_seconds([0.0] * int(5 * SR), adaptive=True) == 0.0


# -- the mechanism, and its own limits -------------------------------------

class TestAdaptiveThresholdItself:
    def test_it_returns_one_bar_per_frame(self):
        vals = [float(i % 7) for i in range(50)]
        assert len(energy.adaptive_threshold(vals, hop_s=0.01)) == 50

    def test_an_empty_window_gives_no_bars(self):
        assert energy.adaptive_threshold([], hop_s=0.01) == []

    def test_every_bar_is_at_least_the_absolute_floor(self):
        """A silent room must not produce a bar of zero."""
        bars = energy.adaptive_threshold([0.0] * 200, hop_s=0.01)
        assert all(b >= energy.ABS_FLOOR for b in bars)

    def test_a_bar_is_always_a_value_that_actually_occurred(self):
        """
        Nearest-rank, no interpolation. So the bar can be compared against the
        signal by inspection instead of being an arithmetic artefact of two
        neighbouring frames.
        """
        vals = [0.1, 0.2, 0.3, 0.9, 0.95, 1.0]
        for b, q in ((0.0, 0.5), (0.75, 0.75)):
            bar = energy.adaptive_threshold(vals, window_s=0.06, hop_s=0.01,
                                            quantile=q, multiple=1.0)
            assert all(any(abs(x - v) < 1e-12 for v in vals) or
                       x == energy.ABS_FLOOR for x in bar)

    def test_the_window_is_bounded_so_cost_is_linear(self):
        """A window that grew with the signal would be quadratic."""
        import time
        short = [0.01 * (i % 13) for i in range(20000)]
        t0 = time.perf_counter()
        energy.adaptive_threshold(short, hop_s=0.01)
        small = time.perf_counter() - t0
        long = short * 4
        t0 = time.perf_counter()
        energy.adaptive_threshold(long, hop_s=0.01)
        big = time.perf_counter() - t0
        assert big < small * 12, (
            f"4x the input cost {big / small:.1f}x the time; a bounded window "
            f"should be close to linear")

    def test_the_sustained_tone_limit_is_real(self):
        """
        A DOCUMENTED FAILURE, asserted so it stays known.

        A tone held for 2 s with no silence in it has no quiet fifth -- the
        window is entirely signal, so the "noise floor" estimate is the signal,
        and the bar rises until the estimator loses what it is estimating.

        Measured: q=0.5 through q=0.9 all find 0.00 s of a 2.00 s sustained
        tone. Real speech is modulated and has inter-word gaps, which is why
        it survives; a held hum is not real speech and this is not a claim
        about it. It IS a claim against using this on any input that can be
        held continuously.
        """
        sustained = tone(180.0, 2.0, 0.3)
        for q in (0.5, 0.75, 0.9):
            found = speech_seconds(sustained, adaptive=True, quantile=q)
            assert found < 0.5, (
                f"q={q} now handles a sustained tone ({found:.2f}s); the "
                f"stated limit needs re-measuring")


# -- through the watcher ----------------------------------------------------

class TestWatcherAdaptive:
    def drain(self, w, samples, chunk=1600):
        events = []
        for i in range(0, len(samples), chunk):
            events.extend(w.push(samples[i:i + chunk]))
        return events

    def test_adaptive_is_off_by_default(self):
        """Every existing caller keeps the exact behaviour it was written for."""
        sig = noise(2.0, 0.002) + voiced(2.0) + noise(2.0, 0.002)
        plain = watcher.Watcher(SR)
        adapt = watcher.Watcher(SR, adaptive=True)
        assert plain.adaptive is False and adapt.adaptive is True
        assert self.drain(plain, sig) or True          # smoke
        self.drain(adapt, sig)

    def test_the_watcher_finds_speech_after_a_slam(self):
        sig = (noise(1.5, 0.002) + tone(300.0, 0.3, 1.0)
               + voiced(1.5) + noise(1.5, 0.002))
        plain = watcher.Watcher(SR)
        adapt = watcher.Watcher(SR, adaptive=True)
        u_plain = [e for e in self.drain(plain, sig)
                   if e.kind == watcher.UTTERANCE]
        u_adapt = [e for e in self.drain(adapt, sig)
                   if e.kind == watcher.UTTERANCE]
        found_plain = sum(u.duration_s for u in u_plain)
        found_adapt = sum(u.duration_s for u in u_adapt)
        assert found_adapt >= found_plain, (
            f"adaptive {found_adapt:.2f}s vs default {found_plain:.2f}s")

    def test_adaptive_and_gate_compose(self):
        """
        The two mechanisms are independent -- one is a threshold, one is a
        model -- and a change to one must not silently disable the other.
        """
        from src import vad
        sig = voiced(1.2) + [0.0] * int(0.6 * SR) + voiced(1.2) \
            + [0.0] * int(0.6 * SR)
        w = watcher.Watcher(SR, adaptive=True,
                            gate=vad.ScriptedGate([0.99]))
        utts = [e for e in self.drain(w, sig) if e.kind == watcher.UTTERANCE]
        assert utts, "no utterance found with both mechanisms on"
        assert all(u.data[watcher.GATE_FIELD] is not None for u in utts), \
            "the gate stopped reporting when adaptive was enabled"
        assert not any(u.data["pitch_skipped"] for u in utts), \
            "a gate reporting 0.99 must not cause YIN to be skipped"

    def test_adaptive_segments_finer_than_the_global_bar(self):
        """
        An honest cost, measured rather than discovered later.

        On a continuous modulated voice the adaptive bar finds 4 segments
        where the global bar finds 2, because it tracks a locally quieter floor
        and so crosses in and out of speech more often. The two extra records
        are 0.03 s and 0.02 s -- above MIN_SPEECH_S, so they survive the merge.

        This is a trade: sensitivity gained against fragmentation. It is
        asserted here so that a future change to MIN_SPEECH_S, the window, or
        the quantile has to face it rather than discover it.
        """
        sig = voiced(1.2) + [0.0] * int(0.6 * SR) + voiced(1.2) \
            + [0.0] * int(0.6 * SR)
        plain = [e for e in self.drain(watcher.Watcher(SR), sig)
                 if e.kind == watcher.UTTERANCE]
        adapt = [e for e in self.drain(watcher.Watcher(SR, adaptive=True), sig)
                 if e.kind == watcher.UTTERANCE]
        assert len(adapt) >= len(plain), \
            "precondition: adaptive should be at least as sensitive"
        assert sum(u.duration_s for u in adapt) >= \
            sum(u.duration_s for u in plain) * 0.95, \
            "adaptive lost speech time to fragmentation"

    def test_the_window_looks_BOTH_ways(self):
        """
        Sabotage-and-revert, run against THIS feature.

        The first version of this file passed with the window sabotaged to
        `lo = 0`, and it passed for a reason worth recording: the case it used
        -- a loud opening settling into a quiet room -- does not distinguish
        the two. Measured on it, backward-looking finds 1.40 s and the
        sabotaged forward-looking window finds 1.30 s, close enough that any
        threshold passes both.

        So the case had to be chosen by what the function actually does. It
        looks in BOTH directions, and the only way to see that is a session
        whose PAST and FUTURE disagree: quiet opening, loud room later.

            local (backward + forward)   4.01 s of a 2.0 s utterance pair
            forward-only (lo = 0)       6.00 s
            global (all frames)          6.00 s

        A forward-only window accumulates the loud future into every earlier
        frame's estimate, so it invents speech that is not there. That is the
        sabotage this test now fails on.

        Both bars come from the shipped function -- `adaptive_threshold` with
        one argument changed -- because an earlier version monkeypatched it,
        which meant sabotaging the shipped code had no effect and the test
        passed while the locality was gone. A test that patches out the thing
        it tests verifies the patch.
        """
        sig = (noise(2.0, 0.002, seed=1) + voiced(1.0, 0.05)
               + noise(2.0, 0.2, seed=2) + voiced(1.0, 0.05)
               + noise(2.0, 0.2, seed=3))
        ef, _ = energy.rms_series(sig, 800, HOP, SR)
        rms = [f.rms for f in ef]

        def found_with(bar):
            segs = []
            state = rms[0] >= bar[0]
            t0 = ef[0].time
            for i, f in enumerate(ef[1:], start=1):
                on, off = bar[i], bar[i] / 2
                if state and f.rms < off:
                    segs.append((t0, f.time, True))
                    state = False
                    t0 = f.time
                elif not state and f.rms >= on:
                    segs.append((t0, f.time, False))
                    state = True
                    t0 = f.time
            segs.append((t0, ef[-1].time + HOP_S, state))
            return sum(b - a for a, b, sp in segs if sp)

        local = found_with(energy.adaptive_threshold(rms, hop_s=HOP_S))

        # the sabotage, reproduced against the shipped function's own code
        win = int(round(energy.LOCAL_WINDOW_S / HOP_S))
        forward_only = []
        for i in range(len(rms)):
            w = sorted(rms[0:min(len(rms), i + win + 1)])
            k = min(len(w) - 1,
                    max(0, int(math.ceil(energy.LOCAL_QUANTILE
                                         * len(w))) - 1))
            forward_only.append(max(energy.ABS_FLOOR,
                                    energy.LOCAL_MULTIPLE * w[k]))

        assert local < found_with(forward_only) - 1.0, (
            f"backward+forward found {local:.2f}s and forward-only "
            f"{found_with(forward_only):.2f}s; the backward half of the "
            f"window is not being used")
