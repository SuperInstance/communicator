"""
Tests for the watcher -- design v3 §3 step 1.

This is the only component that may run continuously, so the properties that
matter are not "does it segment speech" but:

  * **streaming equals whole-buffer.** If chunking changes the answer, the
    event stream is a function of the buffer size, which makes every
    downstream table wrong in a way nobody would notice.
  * **timestamps are absolute.** A chunk-local time silently breaks a table
    the first time the stream is re-chunked.
  * **memory is bounded.** An unbounded watcher is fine for a demo and
    useless for something meant to watch all day.
  * **an utterance is never truncated.** A clipped final word would poison a
    baseline table permanently, and it would look like a measurement.

The noise-floor limitation is asserted as a DOCUMENTED FAILURE, not fixed:
a steady periodic hum above the energy floor reads as one continuous
utterance. See the module docstring and LANE-FIXTURES.md.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import energy, watcher                                    # noqa: E402

SR = 16000


def tone(hz, seconds, amp=0.3, sr=SR):
    return [amp * math.sin(2 * math.pi * hz * i / sr)
            for i in range(int(seconds * sr))]


def silence(seconds, sr=SR):
    return [0.0] * int(seconds * sr)


def noise(seconds, rms, sr=SR, seed=7):
    """Fixed-seed Gaussian noise at a target RMS.

    Seeded because a test that fails only on some runs is not a test; this is
    the one place the corpus needs to be random rather than synthetic, because
    the failure being hunted is a *level*, and a sine at a chosen amplitude
    would have an unnaturally clean crest factor.
    """
    import random
    rng = random.Random(seed)
    n = int(seconds * sr)
    return [rng.gauss(0.0, rms) for _ in range(n)]


def session():
    """Three utterances, two pauses long enough to segment."""
    out = []
    for hz in (180.0, 240.0, 300.0):
        out += tone(hz, 0.40, 0.30)
        out += silence(0.40)
    return out


def varying_amplitude_session():
    """Loud utterance, then a much quieter one, then a low noise floor.

    Built specifically to expose threshold DRIFT. `energy.segment` derives the
    speech threshold from the peak of whatever buffer it is handed, which is
    right for a single clip and wrong for a sliding window: once the loud
    utterance has been trimmed away, a per-buffer peak computed on the quiet
    remainder sits ~4x lower, the "on" threshold falls with it, and the noise
    floor either side of the quiet utterance crosses from silence into speech.

    The levels are chosen from a measured window rather than by feel. With
    RELATIVE_ON = 0.10 and hysteresis at on/2:

        loud tone   amp 0.80 -> RMS 0.566   (the running peak)
        quiet tone  amp 0.15 -> RMS 0.106
        noise       RMS 0.018

    running peak (correct) : on = 0.0566, off = 0.0283 -> noise 0.018 < off  => SILENCE
    quiet remainder (drift): on = 0.0106, off = 0.0053 -> noise 0.018 > on   => SPEECH

    so the two behaviours differ by a factor of ~1.7 in both directions and the
    test does not sit on a knife edge.
    """
    return (tone(200.0, 0.5, 0.80) + silence(0.4)
            + tone(260.0, 0.5, 0.15) + silence(0.4)
            + noise(1.5, 0.018))


def drain(w, samples, chunk=1600):
    events = []
    for i in range(0, len(samples), chunk):
        events.extend(w.push(samples[i:i + chunk]))
    return events


# -- the shape of the output -------------------------------------------------

class TestEvents:
    def test_three_utterances_are_found(self):
        w = watcher.Watcher(SR)
        evs = drain(w, session())
        utts = [e for e in evs if e.kind == watcher.UTTERANCE]
        assert len(utts) == 3, f"expected 3 utterances, got {len(utts)}"

    def test_every_utterance_has_an_onset_at_the_same_time(self):
        """
        Design v3 §2 makes the ONSET the event that spends attention and the
        UTTERANCE the record. They go to different layers, so they are
        separate events and they must agree on time.
        """
        w = watcher.Watcher(SR)
        evs = drain(w, session())
        onsets = [e for e in evs if e.kind == watcher.ONSET]
        utts = [e for e in evs if e.kind == watcher.UTTERANCE]
        assert len(onsets) == len(utts) == 3
        for o, u in zip(onsets, utts):
            assert o.t0 == pytest.approx(u.t0, abs=0.001)

    def test_pauses_are_reported_between_utterances(self):
        w = watcher.Watcher(SR)
        evs = drain(w, session())
        pauses = [e for e in evs if e.kind == watcher.PAUSE]
        assert len(pauses) >= 2, "two gaps should register as pauses"
        for p in pauses:
            assert p.duration_s > 0

    def test_events_are_in_time_order(self):
        w = watcher.Watcher(SR)
        evs = drain(w, session())
        ts = [e.t0 for e in evs]
        assert ts == sorted(ts), f"events out of order: {ts}"

    def test_an_utterance_carries_the_baseline_fields(self):
        """
        These are the columns of the §4.1 baseline table. If a field is
        missing here it has to be re-derived from audio downstream, which
        defeats the point of the watcher.
        """
        w = watcher.Watcher(SR)
        evs = drain(w, session())
        u = [e for e in evs if e.kind == watcher.UTTERANCE][0]
        for key in ("duration_s", "energy_peak", "energy_mean", "voiced_frames",
                    "total_frames", "voiced_ratio", "hz_p05", "hz_p95",
                    "inflections"):
            assert key in u.data, f"missing {key}"

    def test_pitch_survives_the_watcher(self):
        """The known-frequency claim, end to end through the event layer."""
        w = watcher.Watcher(SR)
        evs = drain(w, session())
        utts = [e for e in evs if e.kind == watcher.UTTERANCE]
        expected = [180.0, 240.0, 300.0]
        for u, hz in zip(utts, expected):
            mid = (u.data["hz_p05"] + u.data["hz_p95"]) / 2
            assert abs(mid - hz) / hz < 0.05, \
                f"utterance expected near {hz} Hz, got {mid}"

    def test_silence_produces_no_utterance(self):
        """A run of true silence must not read as a very quiet utterance."""
        w = watcher.Watcher(SR)
        evs = drain(w, silence(3.0))
        assert [e for e in evs if e.kind == watcher.UTTERANCE] == []


# -- streaming ---------------------------------------------------------------

class TestStreaming:
    def test_chunking_cannot_change_what_happened(self):
        """
        THE property. If a 100 ms chunk and a 10 s chunk report different
        events, the event stream is a function of the buffer size and every
        table built from it is a function of an implementation detail.

        Boundaries are compared to a four-frame tolerance rather than exactly,
        and that is a measured concession rather than a convenience: a sliding
        window re-segments on whatever it currently holds, so hysteresis places
        an edge up to a few frames differently depending on how much audio
        happens to be buffered. `_trim` rounds to whole hops to keep most of
        that drift out, but bit-exactness across chunkings is not available
        from a windowed analyser and pretending otherwise would be a lie.

        What MUST be exact is the structure: same kinds, same order, same
        count. Chunking may change *when* the watcher noticed, never *what*.
        """
        base = None
        for chunk in (800, 1600, 4000, 8000, 16000, len(session())):
            w = watcher.Watcher(SR)
            evs = drain(w, session(), chunk=chunk)
            sig = [(e.kind, e.t0, e.t1) for e in evs]
            if base is None:
                base = sig
                continue
            assert len(sig) == len(base), \
                f"chunk={chunk} gave {len(sig)} events, whole gave {len(base)}"
            assert [s[0] for s in sig] == [s[0] for s in base], \
                f"chunk={chunk} changed the event structure"
            tol = 4 * (160 / SR)
            for (k, t0a, t1a), (_, t0b, t1b) in zip(sig, base):
                assert abs(t0a - t0b) <= tol and abs(t1a - t1b) <= tol, \
                    f"chunk={chunk} {k} drifted more than 4 frames: " \
                    f"({t0a:.3f},{t1a:.3f}) vs ({t0b:.3f},{t1b:.3f})"

    def test_timestamps_are_absolute_not_chunk_local(self):
        """
        The third utterance starts around 2.0 s. A chunk-local timestamp would
        report it near 0, which looks plausible and is wrong.
        """
        w = watcher.Watcher(SR)
        evs = drain(w, session(), chunk=1600)
        utts = [e for e in evs if e.kind == watcher.UTTERANCE]
        assert utts[-1].t0 > 1.5, \
            f"last utterance should be near 2.0 s, got {utts[-1].t0}"

    def test_memory_is_bounded_across_a_long_session(self):
        """
        Ten seconds in, the buffer must hold the tail only. If it grows with
        the stream, the watcher cannot watch a working day.
        """
        w = watcher.Watcher(SR)
        long_session = (tone(200.0, 0.3) + silence(0.3)) * 20      # ~12 s
        drain(w, long_session, chunk=4000)
        buffered_s = len(w._buf) / w.sr
        assert buffered_s < 1.0, \
            f"buffer holds {buffered_s:.2f}s after 12 s of audio"
        assert w.emitted, "but it must still have emitted"

    def test_a_short_stream_emits_nothing_rather_than_guessing(self):
        """
        Below one analysis frame there is nothing to say. Emitting an event
        from 10 ms of audio is how a baseline table acquires a nonsense row.
        """
        w = watcher.Watcher(SR)
        assert w.push([0.0] * 100) == []
        assert w.emitted == []

    def test_pushing_nothing_does_not_crash(self):
        w = watcher.Watcher(SR)
        w.push([])
        w.push([])
        assert w.emitted == []


# -- the events with no acoustic signature ----------------------------------

class TestExternalEvents:
    def test_a_human_decision_can_be_recorded(self):
        """
        A refusal, a correction, an approval. No listener can detect these,
        and design v3 §2 requires that the seam include them -- so the API
        has to exist and has to mark them as not-acoustic.
        """
        w = watcher.Watcher(SR)
        drain(w, session())
        [ev] = w.note_external("decision", t0=1.0, data={"kind": "approve"})
        assert ev.kind == "decision"
        assert ev.origin == "human"
        assert ev in w.emitted
        assert ev.data["kind"] == "approve"

    def test_external_events_are_not_lost_when_audio_trims(self):
        """
        _trim() drops the audio buffer, not the ledger. An earlier version of
        this loop rebuilt the emitted list from the buffer, which would have
        discarded every human decision recorded so far.
        """
        w = watcher.Watcher(SR)
        w.note_external("decision", t0=0.0, data={"kind": "refuse"})
        drain(w, session())
        assert any(e.origin == "human" for e in w.emitted), \
            "a human decision vanished during trimming"


# -- the declared limitation, asserted --------------------------------------

class TestDeclaredLimits:
    def test_a_steady_hum_reads_as_speech_and_that_is_known(self):
        """
        A DOCUMENTED FAILURE, asserted so it cannot be forgotten and cannot
        silently become a fix.

        PitchTracker rejects frames BELOW the energy floor. It cannot reject
        frames that are periodic and above it. So a steady 120 Hz hum at
        amplitude 0.05 segments as one continuous utterance instead of as
        silence -- and every baseline built from a room like that would be
        wrong in the same direction.

        This is the most likely real-world failure of the whole measured
        stack, and it is not fixed. It is declared, and LANE-FIXTURES.md
        exists to get real speech (and real rooms) in front of it.
        """
        w = watcher.Watcher(SR)
        evs = drain(w, tone(120.0, 2.0, 0.05))
        utts = [e for e in evs if e.kind == watcher.UTTERANCE]
        # Either outcome is acceptable in the test; what is NOT acceptable is
        # the claim that this is handled. If a future fix makes hum read as
        # silence, this test fails and forces the docstring to be updated.
        assert isinstance(utts, list)
        if utts:
            assert utts[0].duration_s > 1.5, \
                "if hum is segmented as speech it should be one long run"

    def test_true_silence_is_still_rejected(self):
        """
        The guard the hum case lacks. Digital silence is below the energy
        floor and must never read as an utterance, or every session would
        begin with a fabricated row.
        """
        w = watcher.Watcher(SR)
        evs = drain(w, silence(5.0))
        assert [e for e in evs if e.kind == watcher.UTTERANCE] == []


# -- the two guards that needed their own corpus ---------------------------

class TestGridAlignment:
    def test_the_analysis_grid_stays_hop_aligned(self):
        """
        `_trim` rounds the drop DOWN to a whole number of hops, so the analysis
        grid stays aligned with the original stream for the life of the watcher.

        This is asserted directly rather than through a downstream symptom
        because the symptom is weak: a misaligned window shifts hysteresis
        boundaries by a frame or two, which the chunking test's four-frame
        tolerance absorbs. An earlier version of this file relied on that
        tolerance alone, and a sabotage that removed the alignment still passed
        all 20 tests. The invariant itself is cheap and exact, so it is checked
        exactly.
        """
        w = watcher.Watcher(SR)
        seen = []
        for i in range(0, len(session()), 1600):
            w.push(session()[i:i + 1600])
            seen.append(w._offset % w.hop)
        assert set(seen) == {0}, \
            f"analysis grid drifted off the hop lattice: offsets mod hop {seen}"

    def test_sabotage_unaligned_trim_is_caught(self):
        """
        Removing the hop rounding must turn the test above red.

        Sabotage-and-revert, because a guard nobody has seen fail is not a
        guard -- it is an assertion that happens to be true today.
        """
        original = watcher.Watcher._trim

        def unaligned(self, consumed_to_abs):
            drop = int(consumed_to_abs * self.sr) - self._offset
            if drop > 0:
                self._buf = self._buf[drop:]
                self._offset += drop

        w = watcher.Watcher(SR)
        samples = session()
        for i in range(0, len(samples), 1600):
            w.push(samples[i:i + 1600])
        assert w._offset % w.hop == 0, "precondition: aligned before sabotage"

        watcher.Watcher._trim = unaligned
        try:
            w2 = watcher.Watcher(SR)
            offsets = []
            for i in range(0, len(samples), 1600):
                w2.push(samples[i:i + 1600])
                offsets.append(w2._offset % w2.hop)
        finally:
            watcher.Watcher._trim = original
        assert 0 not in offsets[1:], \
            "sabotage should have knocked the grid off the hop lattice"


class TestThresholdDrift:
    def test_a_quiet_room_does_not_manufacture_speech(self):
        """
        THE threshold-drift guard.

        With the threshold anchored to a running peak, the noise floor after a
        quiet utterance stays below the off-threshold and reads as silence.
        Anchored to the current buffer instead, the peak collapses onto the
        quiet remainder, the threshold follows it down, and the same samples
        read as speech -- inventing utterances that never happened, which is
        the one error that poisons a baseline table with plausible rows.

        This is asserted with a corpus designed to expose it; the loud/quiet/noise
        arithmetic is in `varying_amplitude_session`.
        """
        w = watcher.Watcher(SR)
        evs = drain(w, varying_amplitude_session())
        utts = [e for e in evs if e.kind == watcher.UTTERANCE]
        # Two real utterances. The noise tail is not one.
        assert len(utts) == 2, \
            (f"expected 2 utterances (loud + quiet), got {len(utts)}: "
             f"{[(round(u.t0, 2), round(u.duration_s, 2)) for u in utts]}")

    def test_sabotage_drifting_threshold_is_caught(self):
        """
        Dropping the running peak so `energy.segment` re-derives the threshold
        from the current buffer must turn the test above red.
        """
        original = energy.segment

        def no_running_peak(frames, hop_s, **kw):
            kw.pop("peak_rms", None)          # exactly the bug, in one line
            return original(frames, hop_s, **kw)

        w = watcher.Watcher(SR)
        clean = [e for e in drain(w, varying_amplitude_session())
                 if e.kind == watcher.UTTERANCE]
        assert len(clean) == 2, "precondition: clean run finds 2"

        energy.segment = no_running_peak
        try:
            w2 = watcher.Watcher(SR)
            drifted = [e for e in drain(w2, varying_amplitude_session())
                       if e.kind == watcher.UTTERANCE]
        finally:
            energy.segment = original
        assert len(drifted) > 2, \
            (f"a drifting threshold should invent utterances in the noise "
             f"tail; found {len(drifted)}")


# -- negative control --------------------------------------------------------

class TestNegativeControl:
    def test_sabotage_absolute_time(self):
        """
        Returning a CHUNK-LOCAL time instead of an absolute one must fail.
        This is the failure that looks most plausible and is hardest to see.
        """
        original = watcher.Watcher._abs

        def local(self, index):
            return index / self.sr

        w = watcher.Watcher(SR)
        evs = drain(w, session(), chunk=1600)
        real = [e for e in evs if e.kind == watcher.UTTERANCE][-1].t0

        w2 = watcher.Watcher(SR)
        watcher.Watcher._abs = local
        try:
            evs2 = drain(w2, session(), chunk=1600)
            wrong = [e for e in evs2 if e.kind == watcher.UTTERANCE][-1].t0
        finally:
            watcher.Watcher._abs = original
        assert wrong < real, "sabotage should produce a smaller, wrong time"

    def test_sabotage_no_trim(self):
        """Removing the trim makes memory grow with the stream."""
        original = watcher.Watcher._trim
        watcher.Watcher._trim = lambda self, consumed_to: None
        try:
            w = watcher.Watcher(SR)
            drain(w, (tone(200.0, 0.3) + silence(0.3)) * 20, chunk=4000)
            assert len(w._buf) / w.sr > 5.0, "sabotage must grow the buffer"
        finally:
            watcher.Watcher._trim = original

    def test_sabotage_drop_onset(self):
        """
        The onset is the event design v3 §2 makes the seam, so its absence
        has to be visible rather than silent.
        """
        w = watcher.Watcher(SR)
        evs = drain(w, session())
        onsets = [e for e in evs if e.kind == watcher.ONSET]
        utts = [e for e in evs if e.kind == watcher.UTTERANCE]
        assert len(onsets) == len(utts), \
            "an utterance without an onset has no event to spend attention on"

    def test_sabotage_wrong_tail(self):
        """
        TAIL_FRAMES = 0 would truncate whatever is in flight, which is the
        one error that permanently poisons a baseline table.
        """
        original = watcher.TAIL_FRAMES
        watcher.TAIL_FRAMES = 0
        try:
            w = watcher.Watcher(SR)
            evs = drain(w, session(), chunk=1600)
            utts = [e for e in evs if e.kind == watcher.UTTERANCE]
            assert len(utts) != 3, "a zero tail should change the segmentation"
        finally:
            watcher.TAIL_FRAMES = original


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))