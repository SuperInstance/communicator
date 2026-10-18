"""
Tests for the voice-activity gate -- the thing in front of the watcher.

The gate exists because `watcher.py` classifies speech by energy alone and
there is a measured, reproducible failure: a room with a fan is not silent,
it is one continuous utterance that never ends. Measured on this machine:

    the watcher's segmenter vs signals whose truth we know
        steady 60 Hz hum      -> 89% "speech"
        hum + harmonics       -> 89%          (all identical: saturation)
        white noise           -> 89%
        digital silence       -> 89%
        120 Hz hum at amp 0.05 (the case the watcher docstring declares)
                               -> one continuous utterance, never ends

The 89% being identical for digital silence is the tell. The segmenter marks
the WHOLE clip, so "how much of this clip is speech" carries no information
at all in a noisy room.

SO THE GATE IS TESTED IN BOTH DIRECTIONS, AND THE SECOND DIRECTION IS THE
ONE THAT MATTERS.

A gate built for a threat and only tested against that threat will always
look good. The 2026 session that produced the adversarial battery retracted a
max-gate for exactly this reason: it scored an attack at 0.867 and ordinary
narration at 0.980, so it blocked ordinary conversation harder than the
attack. The retraction is now a fixture.

So this file does NOT ask "does the gate catch a hum". It asks both:
    * does it reject every non-speech signal tried?           (the threat)
    * does it ACCEPT quiet real speech?                      (what it must not block)

The second half is the one that decides whether this module is usable.

THE MEASURED SEPARATION, on 90 s of real LibriVox speech
(`examples/realaudio/alice01_excerpt.wav`, resampled to 16 kHz by
`wavio.read_wav`), scored with ONE gate held across the whole stream:

    real speech, per 32 ms frame     mean 0.7300   frac>=0.5 0.728
    120 Hz hum amp 0.05             mean 0.1579   frac>=0.5 0.000
    hum + harmonics                 mean 0.0205   frac>=0.5 0.000
    white noise                     mean 0.0118   frac>=0.5 0.000
    digital silence                 mean 0.0037   frac>=0.5 0.000

Two decisions came out of measuring rather than assuming, and both are
asserted below so neither can be quietly undone:

1. `max`, not `mean` or `frac`, is the statistic. Over 20 real 0.25-1.2 s
   speech segments, `max` vetoed 0/20; `mean` vetoed 1/20. A 0.5 threshold on
   the mean of a short segment throws away real speech for no benefit.

2. ONE gate for the whole stream, never restarted per segment. The model
   carries LSTM state. Restarting it per segment vetoes real speech that a
   continuous gate accepts, because the model has no context at the segment
   edge:

       level      vetoed (continuous)   vetoed (restart per segment)
         -10 dB            0/20                     2/20
         -20 dB            0/20                     4/20
         -30 dB            0/20                     8/20
         -40 dB            1/20                    18/20

   At -40 dB a restarted gate would drop 90% of real speech. This is the same
   class of defect as the watcher's file-global thresholds, and it is why
   `push` is documented as carrying state.

THE LIMIT, ASSERTED RATHER THAN DISCOVERED LATER. The gate is not a
classifier of everything non-speech. A pure tone at 150, 175 or 200 Hz
scores 0.66-0.82, and a harmonic stack at 200 Hz scores 0.97 -- indistinguishable
from speech by this statistic. So the gate CANNOT be used to decide whether a
segment is worth reporting; it is used to decide whether running YIN over it
is worth the CPU. That is the direction this repo already commits to in
`policy.backend_may_gate`: "local buys availability, not judgement" applies
with even more force to a 2 MB model on a peripheral.

Every test that needs the real weights is skipped when they are absent, so
the suite still runs on a machine with no onnxruntime and no 2.2 MB blob.
The contract tests run regardless.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import energy, vad                                          # noqa: E402

SR = 16000
FIXTURE = ROOT / "examples" / "realaudio" / "alice01_excerpt.wav"

_weights = vad.model_path()
try:
    import onnxruntime                                           # noqa: F401
    _ORT = True
except ImportError:
    _ORT = False

needs_weights = pytest.mark.skipif(
    not (_ORT and _weights),
    reason="silero weights/onnxruntime absent; run src.vad.fetch_model()")


# -- signal builders ---------------------------------------------------------

def tone(hz, seconds, amp=0.3, sr=SR):
    return [amp * math.sin(2 * math.pi * hz * i / sr)
            for i in range(int(seconds * sr))]


def harmonic_stack(hz, seconds, amp=0.3, harmonics=(1, 2, 3), sr=SR):
    out = [0.0] * int(seconds * sr)
    for k, harm in enumerate(harmonics):
        a = amp / (k + 1)
        for i in range(len(out)):
            out[i] += a * math.sin(2 * math.pi * hz * harm * i / sr)
    return out


def noise(seconds, rms, sr=SR, seed=11):
    rng = random.Random(seed)
    return [rng.gauss(0.0, rms) for _ in range(int(seconds * sr))]


def continuous_probs(samples, gate=None, chunk=1600):
    """One gate for the whole stream -- how the watcher actually uses it."""
    g = gate or vad.SileroGate()
    out = []
    for i in range(0, len(samples), chunk):
        out.extend(g.push(samples[i:i + chunk]))
    return out


def seg_max(probs, start_s, end_s):
    """Peak gate probability over a time range of a continuous prob series.

    `max` is the chosen statistic and this is its only implementation, so the
    choice is visible at every call site rather than buried in a helper.
    """
    i0 = max(0, int(start_s * SR) // vad.FRAME_SAMPLES)
    i1 = min(len(probs), int(end_s * SR) // vad.FRAME_SAMPLES)
    return max(probs[i0:i1]) if i1 > i0 else 0.0


@pytest.fixture(scope="module")
def speech():
    if not FIXTURE.is_file():
        pytest.skip("real speech fixture absent")
    from src import wavio
    return wavio.read_wav(str(FIXTURE)).samples


def speech_segments(samples):
    """Real speech segments the watcher's own segmenter would call speech."""
    ef, _ = energy.rms_series(samples, 800, 160, SR)
    assert ef, "fixture produced no frames"
    peak = max(f.rms for f in ef)
    return [s for s in energy.segment(ef, 160 / SR, peak_rms=peak)
            if s.speech and 0.25 <= s.duration() <= 1.2]


# -- the threat: non-speech the gate must reject -----------------------------

@needs_weights
class TestNonSpeechIsRejected:
    def test_the_declared_hum_failure_is_below_threshold(self):
        """
        The exact signal `watcher.py`'s docstring declares as its most likely
        real-world failure: 120 Hz at amplitude 0.05, which the energy-only
        segmenter reads as one continuous utterance.
        """
        probs = continuous_probs(tone(120.0, 3.0, 0.05))
        assert seg_max(probs, 0.0, 3.0) < 0.5, \
            "the declared failure is no longer rejected -- re-measure before " \
            "trusting this test"

    def test_a_harmonic_fan_is_below_threshold(self):
        """A fan is not one tone. Harmonics are what a fan actually has."""
        sig = harmonic_stack(60.0, 3.0, 0.20, (1, 2, 3, 4))
        assert seg_max(continuous_probs(sig), 0.0, 3.0) < 0.5

    def test_white_noise_and_digital_silence_are_below_threshold(self):
        for name, sig in (("white noise", noise(2.0, 0.3)),
                          ("digital silence", [0.0] * int(2.0 * SR))):
            assert seg_max(continuous_probs(sig), 0.0, 2.0) < 0.5, \
                f"{name} crossed the threshold"

    def test_a_full_clip_of_hum_produces_no_over_threshold_frame(self):
        """
        The stronger form: not merely a low peak but not one single frame
        above 0.5 in two seconds. This is the number that would make the
        watcher's continuous-utterance failure disappear, because the YIN
        work is skipped on frames the gate calls silent.
        """
        probs = continuous_probs(harmonic_stack(60.0, 2.0, 0.25))
        assert sum(1 for p in probs if p >= 0.5) == 0


# -- the direction that actually matters -------------------------------------

@needs_weights
class TestRealSpeechIsAccepted:
    def test_speech_frames_clear_the_threshold_with_room(self, speech):
        probs = continuous_probs(speech)
        over = [p for p in probs if p >= 0.5]
        assert len(over) / len(probs) > 0.5, \
            f"only {len(over) / len(probs):.3f} of 90 s of real speech " \
            f"clears 0.5 -- the threshold is too high"

    def test_no_real_speech_segment_is_vetoed(self, speech):
        """
        THE direction. Not "does it catch a hum" but "does it refuse to throw
        away a person talking". Over every speech segment the watcher's own
        segmenter produces in 90 s of LibriVox, none may be gated out.
        """
        segs = speech_segments(speech)
        assert len(segs) >= 10, f"only {len(segs)} segments; corpus too thin"
        probs = continuous_probs(speech)
        vetoed = [s for s in segs if seg_max(probs, s.start, s.end) < 0.5]
        assert not vetoed, \
            f"{len(vetoed)}/{len(segs)} real speech segments vetoed; " \
            f"first at {vetoed[0].start:.2f}s -- the gate would silence a " \
            f"real speaker"

    def test_quiet_speech_is_still_accepted(self, speech):
        """
        The retraction lesson, as a fixture. A gate that only works at a
        comfortable level is a gate that decides who gets heard.

        -20 dB is roughly someone talking from across a room, which is the
        entire premise of a peripheral system.
        """
        sig = [x * (10 ** (-20 / 20)) for x in speech]
        segs = speech_segments(sig)
        probs = continuous_probs(sig)
        vetoed = [s for s in segs if seg_max(probs, s.start, s.end) < 0.5]
        assert len(vetoed) <= 1, \
            f"{len(vetoed)}/{len(segs)} segments vetoed at -20 dB -- quiet " \
            f"speech is being blocked"

    def test_max_beats_mean_as_the_segment_statistic(self, speech):
        """
        WHY `max` AND NOT `mean`, measured on 20 real segments.

        `mean` vetoed 1 of 20 real speech segments. That is not a rounding
        error: it is a real utterance silently dropped because the average
        over a short window is pulled down by its own quiet edges, which is
        the same averaging bug the segmentation threshold had.
        """
        segs = speech_segments(speech)
        g = vad.SileroGate()
        probs = continuous_probs(speech, gate=g)
        vetoed_by_max = [s for s in segs
                         if seg_max(probs, s.start, s.end) < 0.5]
        # same segments, each scored by a gate RESTARTED on it -- the mean of
        # that restart is what a naive implementation would use
        vetoed_by_mean = 0
        for s in segs:
            p = vad.SileroGate().push(speech[int(s.start * SR):
                                              int(s.end * SR)])
            if p and sum(p) / len(p) < 0.5:
                vetoed_by_mean += 1
        assert len(vetoed_by_max) == 0
        assert vetoed_by_mean >= 1, \
            "mean no longer vetoes anything; re-measure the justification " \
            "for max before trusting this test"


# -- statefulness is load-bearing -------------------------------------------

@needs_weights
class TestStatefulness:
    def test_one_gate_for_the_stream_beats_one_gate_per_segment(self, speech):
        """
        The model carries LSTM state across frames. Restarting it per segment
        scores a segment with no context at its edges, and that is enough to
        veto real speech a continuous gate accepts.

        Measured on the same 90 s fixture:

            level   vetoed (continuous)   vetoed (restart per segment)
              -20 dB         0/20                     4/20
              -30 dB         0/20                     8/20

        So the watcher must hold ONE gate for its whole life. This test
        exists so that a future "optimisation" that reuses a gate per segment
        -- which looks obviously cheaper -- fails here instead of in the room.
        """
        sig = [x * (10 ** (-30 / 20)) for x in speech]
        segs = speech_segments(sig)
        assert len(segs) >= 10

        probs = continuous_probs(sig)
        cont_vetoed = sum(1 for s in segs
                          if seg_max(probs, s.start, s.end) < 0.5)

        restart_vetoed = 0
        for s in segs:
            p = vad.SileroGate().push(sig[int(s.start * SR):int(s.end * SR)])
            if not p or max(p) < 0.5:
                restart_vetoed += 1

        assert cont_vetoed <= 1, f"continuous gate vetoed {cont_vetoed}"
        assert restart_vetoed > cont_vetoed, \
            "restarting per segment is no longer worse; the statefulness " \
            "justification needs re-measuring"

    def test_reset_returns_the_gate_to_its_opening_state(self, speech):
        """A reset gate must score identically to a fresh one."""
        sig = speech[:16000 * 2]
        first = continuous_probs(sig, gate=vad.SileroGate())
        g = vad.SileroGate()
        continuous_probs(sig, gate=g)          # advance the LSTM state
        g.reset()
        after = continuous_probs(sig, gate=g)
        assert after == pytest.approx(first, abs=1e-9)

    def test_state_carries_forward_across_chunks(self, speech):
        """
        Pushing one whole clip and pushing it in 100 ms chunks must agree,
        because a live mic delivers chunks and the whole-clip case is only
        reachable in a test.
        """
        sig = speech[:16000 * 3]
        whole = continuous_probs(sig, chunk=len(sig))
        chunked = continuous_probs(sig, chunk=1600)
        assert len(whole) == len(chunked)
        assert chunked == pytest.approx(whole, abs=1e-6)


# -- chunk invariance (the gate's own contract) ------------------------------

@needs_weights
class TestChunking:
    def test_partial_frames_are_carried_not_dropped(self, speech):
        """
        Audio pushed mid-frame must not be discarded or double-counted. A
        1600-sample chunk is 3.125 frames, so most chunks end mid-frame and
        this is the normal case, not an edge case.
        """
        sig = speech[:16000 * 2]
        whole = continuous_probs(sig, chunk=len(sig))
        for chunk in (160, 512, 1600, 4096, 16000):
            assert continuous_probs(sig, chunk=chunk) == pytest.approx(
                whole, abs=1e-6), f"chunk={chunk} changed the scores"

    def test_an_empty_push_is_harmless(self):
        g = vad.SileroGate()
        assert g.push([]) == []
        assert g.push([]) == []

    def test_a_chunk_shorter_than_one_frame_returns_nothing(self):
        """Nothing is said until there is a whole frame to say it about."""
        g = vad.SileroGate()
        assert g.push([0.0] * 100) == []
        assert len(g.push([0.0] * 412)) == 1

    def test_output_is_one_probability_per_frame(self):
        g = vad.SileroGate()
        n = 16000 * 3
        assert len(g.push([0.0] * n)) == n // vad.FRAME_SAMPLES


# -- the limits, asserted so they cannot be forgotten ------------------------

@needs_weights
class TestDeclaredLimits:
    def test_a_tone_near_200hz_is_indistinguishable_from_speech(self):
        """
        A DOCUMENTED FAILURE of using this gate as a speech/no-speech
        classifier, asserted so nobody later assumes it can do that job.

        Pure tones at 150, 175 and 200 Hz score 0.66-0.82, and a harmonic
        stack at 200 Hz scores 0.97 -- real-speech territory. The model's own
        training contains voiced periodic sound, and a periodic sound with
        that structure is what it is looking for.
        """
        scored = {hz: seg_max(continuous_probs(tone(hz, 0.5, 0.3)), 0.0, 0.5)
                  for hz in (150, 175, 200)}
        assert any(v >= 0.5 for v in scored.values()), \
            f"no tone fooled the gate any more: {scored}"
        assert seg_max(continuous_probs(
            harmonic_stack(200.0, 0.5, 0.3)), 0.0, 0.5) >= 0.5

    def test_which_is_why_the_gate_may_not_decide_what_is_reported(self):
        """
        The consequence of the limit above, made executable.

        The gate answers "is running YIN over this worth the CPU". It may not
        answer "is this segment worth reporting", because it gets 200 Hz fans
        wrong. So the watcher's use of it must be confined to SKIPPING WORK,
        and every utterance record must still carry the probability so a human
        can audit which segments were decided by the gate alone.
        """
        # The watcher module must expose the gate's verdict as data, not as a
        # branch that can silently drop an event.
        from src import watcher
        assert hasattr(watcher, "GATE_FIELD"), \
            "the gate verdict must be a named field on the utterance record"

    def test_quiet_speech_below_about_40db_is_not_reliably_detected(self, speech):
        """
        The other direction's limit, measured rather than assumed: at -40 dB a
        continuous gate already vetoes some real speech, and by -50 dB it
        vetoes essentially all of it.

        So there is a level below which this gate cannot be trusted, and it is
        around -40 dB. Any claim that the watcher "always hears the room"
        would be false below that.
        """
        sig = [x * (10 ** (-50 / 20)) for x in speech]
        segs = speech_segments(sig)
        probs = continuous_probs(sig)
        vetoed = sum(1 for s in segs
                     if seg_max(probs, s.start, s.end) < 0.5)
        assert vetoed > len(segs) / 2, \
            "-50 dB speech is now reliably detected; the stated floor needs " \
            "re-measuring"

    def test_the_first_frames_after_a_silent_run_are_the_weakest(self):
        """
        A real artefact of the start-up state, measured: a 120 Hz hum at amp
        0.05 scores 0.0105 on its first frame and 0.3563 by its twelfth,
        because the LSTM state has to fill before it can judge.

        Consequence for integration: the first ~100 ms after a long silence
        is the least trustworthy region. The watcher must not treat a single
        early frame as decisive, which is one more reason the segment
        statistic is `max` over a whole segment and not a per-frame veto.
        """
        probs = continuous_probs(tone(120.0, 0.5, 0.05))
        assert probs[0] < probs[min(11, len(probs) - 1)], \
            "the start-up transient is gone; re-measure before relying on it"


# -- cost, and the contract that does not need the weights ------------------

class TestCost:
    @needs_weights
    def test_the_gate_is_cheap_enough_to_sit_in_front_of_everything(self):
        """
        The number that justifies putting it in the continuous path at all.

        Measured: 0.072-0.135 ms per 32 ms chunk, one thread. That is
        0.2-0.4% of a core, against the watcher's 39.13 ms per 100 ms --
        39.1% of a core. If this gate ever costs more than a few percent, it
        does not belong in front of the watcher and the whole argument for it
        changes.
        """
        import time
        g = vad.SileroGate()
        chunk = [0.0] * vad.FRAME_SAMPLES
        g.push(chunk)                                   # warm the session
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            g.push(chunk)
        per = (time.perf_counter() - t0) / n
        audio_s = vad.FRAME_SAMPLES / SR
        assert per < audio_s * 0.05, \
            f"{per * 1000:.3f} ms per {audio_s * 1000:.0f} ms chunk is " \
            f"{per / audio_s:.1%} of realtime -- too expensive for the " \
            f"continuous path"


class TestWeightsAreFetchedNotVendored:
    def test_model_path_is_optional_and_never_reaches_the_network(self):
        """
        Importing this module must never download 2.2 MB behind the caller's
        back, so `fetch_model` is a function and not an import side effect.
        """
        assert callable(vad.fetch_model)
        assert vad.model_path() is None or isinstance(vad.model_path(), str)

    def test_a_missing_model_is_a_downgrade_not_a_crash(self):
        """
        `GateUnavailable` is a distinct exception so a caller can fall back
        to energy-only behaviour, which is the shipped default when onnxruntime
        or the weights are absent. A peripheral that refuses to run without a
        2 MB model is not a peripheral.
        """
        with pytest.raises(vad.GateUnavailable):
            vad.SileroGate(model_path="/nonexistent/silero.onnx")

    def test_a_wrong_sample_rate_is_refused_rather_than_silently_wrong(self):
        """A 48 kHz graph fed 16 kHz audio produces plausible nonsense."""
        if not (_ORT and _weights):
            pytest.skip("silero weights/onnxruntime absent")
        with pytest.raises(ValueError):
            vad.SileroGate(sample_rate=48000)


# -- the stand-in used by tests and offline machines -------------------------

class TestScriptedGate:
    def test_it_speaks_the_same_protocol_as_the_real_gate(self):
        """
        `ScriptedGate` stands in for `SileroGate` so the suite can test the
        watcher's gate behaviour with no model and no onnxruntime. That only
        works while the two agree on the protocol, which is what this asserts.
        """
        g = vad.ScriptedGate([0.9, 0.9, 0.01, 0.01])
        n = 4 * vad.FRAME_SAMPLES
        assert g.push([0.0] * n) == [0.9, 0.9, 0.01, 0.01]

    def test_it_holds_its_last_value_when_the_script_runs_out(self):
        """A short script must not turn into silence at the end of a stream."""
        g = vad.ScriptedGate([0.9])
        assert len(g.push([0.0] * (10 * vad.FRAME_SAMPLES))) == 10
        assert all(p == 0.9 for p in
                   g.push([0.0] * (10 * vad.FRAME_SAMPLES)))

    def test_is_speech_and_mean_agree_with_push(self):
        g = vad.ScriptedGate([0.1, 0.9])
        assert g.push([0.0] * (2 * vad.FRAME_SAMPLES)) == [0.1, 0.9]
        g2 = vad.ScriptedGate([0.1, 0.9])
        assert g2.is_speech([0.0] * (2 * vad.FRAME_SAMPLES)) is True
        g3 = vad.ScriptedGate([0.1, 0.2])
        assert g3.is_speech([0.0] * (2 * vad.FRAME_SAMPLES)) is False

    def test_mean_probability_of_a_quiet_stream_is_low(self):
        g = vad.ScriptedGate([0.02])
        assert g.mean_probability([0.0] * (5 * vad.FRAME_SAMPLES)) == 0.02

    def test_push_of_a_short_chunk_yields_no_frames(self):
        g = vad.ScriptedGate([0.9])
        assert g.push([0.0] * 100) == []
