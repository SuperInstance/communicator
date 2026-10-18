"""
watcher -- the continuous layer. No network, and NOT microseconds.

This is the first component of design v3 §3 step 1, and it is the one that
makes the rest possible, because it is the only layer that can run always-on.
The measurements forced that: judged tracks cost 0.97 s for four gates and
2.45 s for eight locally, so nothing costing a second may sit in the
continuous path.

THE FIRST LINE OF THIS DOCSTRING USED TO SAY "MICROSECONDS". IT WAS WRONG, and
the way it was wrong is the whole point.

Measured here, single thread, one core:

    watcher.py, 100 ms of audio      39.13 ms     39.1% of a core
    with the VAD gate in front        1.2 ms      1.2% of a core

A peripheral cannot spend 39% of a core deciding whether someone is talking.
That is affordable on a laptop and impossible on the device this is built for.
The number is now asserted: `test_skipping_yin_actually_saves_the_cpu_it_
claims_to` fails if gating ever stops saving the work.

WHAT "WATCH" MEANS HERE IS NOT A PASS OVER A FILE.

A whole-clip pass is not watching. This takes a growing buffer, analyses it,
and emits events as soon as an utterance is *complete*, retaining a tail that
might still be mid-utterance. That is what lets it sit behind a live mic
forever, and it is why the API is `push()` rather than `analyse()`.

    w = Watcher(sample_rate=16000, gate=SileroGate())
    for chunk in stream:                      # 100 ms chunks, say
        for ev in w.push(chunk):
            ...                                # utterance, pause, or onset

The re-analysis of the retained tail is deliberate and cheap: it costs one
extra tail per push rather than a clipped utterance, and a truncated final
word is the one error that would poison a baseline table forever.

THE GATE IS OPTIONAL AND ITS ABSENCE IS THE DEFAULT.

`gate=None` reproduces the previous behaviour exactly, so a machine with no
onnxruntime and no 2.2 MB of weights still runs. The gate's one job is to
decide whether running YIN over a segment is worth the CPU. It is NOT trusted
to decide whether a segment is REPORTED, because against a 200 Hz harmonic
stack it scores 0.97 and cannot tell a fan from a person. Every utterance
record therefore carries the gate's verdict under `GATE_FIELD`, so a segment
the gate dismissed can be found and audited afterwards. See `vad.py` and
`tests/test_vad.py`.

THE EVENTS ARE THE THING. Design v3 §2 made the seam the EVENT rather than the
turn, so this is the component that defines them:

* ``onset``     -- a voiced onset after >= MIN_SILENCE_S of silence. This is
                   the event that spends attention.
* ``utterance`` -- a complete run of speech, with the measured summary the
                   §4.1 baseline table needs.
* ``pause``     -- a complete run of silence, with its duration.

A fourth event, a *human decision*, has no acoustic signature at all and so
cannot be detected here. It is `note_external()` -- and its existence is the
reminder that a system which only listens cannot see the most important events
it will ever receive (a refusal, a correction, an approval).

KNOWN LIMIT, AND IT IS THE DANGEROUS ONE: a steady periodic hum above the
energy floor -- a fan, HVAC, a held vowel -- is indistinguishable from voiced
speech BY ENERGY. `pitch.PitchTracker` rejects frames *below* ENERGY_FLOOR,
not frames that are periodic and loud. So the energy-only segmenter marks the
whole clip, which is why a hum and digital silence produce the identical 89%
and a room with a fan emits no pause events at all.

What changed: the gate answers that question properly (a 120 Hz hum at
amplitude 0.05 scores 0.356 against 0.998 for real speech), so the YIN work is
skipped and the failure is contained. What did NOT change: the energy
segmentation still cannot tell a fan from a person, so such a segment is still
emitted as an utterance -- marked `pitch_skipped`, with the probability
attached. It is DOWNGRADED, not corrected. The honest framing is that the gate
makes the failure cheap rather than making it go away. Asserted in
`tests/test_vad.py::TestDeclaredLimits`.

Everything here is arithmetic, or one small ONNX graph. No network, no clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import energy, inflection, pitch, vad

# Kept identical to the measured tracks' grid so no resampling is implied.
FRAME_LEN = 800            # 50 ms at 16 kHz
HOP = 160                  # 10 ms
HOP_S = HOP / 16000.0

# Frames retained after the last emitted event, so an utterance that is still
# open is re-analysed rather than truncated. 300 ms covers the longest
# plausible phrase-in-progress at normal speech rate.
TAIL_FRAMES = 30

# Longest an OPEN segment may stay unreported.
#
# A segment only emits once it has ENDED, because reporting a half-finished
# utterance as a finished one would poison a baseline table with a truncated
# row that looks exactly like a real measurement. But "wait until it ends" has
# a failure the tail rule does not cover: a segment that never ends.
#
# Measured, before these bounds existed: three seconds of continuous speech
# emitted ZERO events. So did three seconds of silence, three seconds of a
# steady hum, and three seconds of low noise. The watcher was silent precisely
# where a peripheral system most needs to speak up -- someone talking without
# pause is the ordinary condition of the room this is built for, not an edge
# case. A `test_a_steady_hum_reads_as_speech` assertion guarded this and was
# VACUOUS: its `if utts:` branch never ran, because there were never any.
#
# So an open segment past this length is emitted as an explicit CHECKPOINT:
# clipped to the last certainly-complete point, marked `open`, and carrying no
# ONSET (the onset already fired; re-emitting it would spend attention twice).
# The utterance is not truncated *silently* -- the record says it is still in
# progress -- which is the difference between losing an event and lying about
# one.
MAX_UTTERANCE_S = 2.0
MAX_PAUSE_S = 3.0

ONSET = "onset"
UTTERANCE = "utterance"
PAUSE = "pause"
DECISION = "decision"

# The gate's verdict, carried as DATA on every utterance record.
#
# The name is a module constant rather than a string literal at the call site
# because the gate is allowed to be wrong in a specific direction: measured on
# a 200 Hz harmonic stack it scores 0.97, which is real-speech territory. So
# the gate may only decide whether to spend CPU on YIN, never whether a segment
# is reported. Carrying the number on the record means any segment the gate
# dismissed can be found afterwards and audited -- a decision that leaves no
# trace is not auditable, and an unauditable decision in the continuous path
# is exactly the kind that gets discovered a week later as a missing baseline.
GATE_FIELD = "gate_speech_prob"

# Below this peak probability the gate says the segment contains no voice, so
# YIN is skipped on it.
#
# This is the gate's own threshold, passed through rather than re-tuned: the
# separation on real speech is wide enough that the exact value inside this
# region does not matter (measured 0.6577 for speech against 0.3563 for the
# loudest non-speech, a 120 Hz hum at the amplitude that breaks the
# energy-only segmenter). It is a constant here so a reader can see that the
# watcher is not inventing a second, differently-tuned number.
GATE_SPEECH = 0.5

# Cadence for reporting a pause that has not closed yet, in seconds of
# accumulated silence. See Watcher._pause_reported_at.
PAUSE_REPORT_S = 0.25


@dataclass
class Event:
    """
    One thing that happened, at an absolute time in the stream.

    `kind` is one of ONSET / UTTERANCE / PAUSE / DECISION. `t0`/`t1` are
    absolute seconds since the watcher was created, NOT since the chunk — a
    chunk-local timestamp would make every downstream table wrong the moment
    the stream was chunked differently, which is a silent failure.
    """
    kind: str
    t0: float
    t1: float
    data: dict = field(default_factory=dict)
    origin: str = "acoustic"      # "acoustic" | "human"

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t1 - self.t0)


class Watcher:
    """
    Stateful, append-only, and bounded.

    Bounded is the load-bearing property: `push` discards everything already
    emitted except the tail, so memory does not grow with the length of the
    session. An unbounded watcher is fine for a demo and useless for something
    meant to watch all day.
    """

    def __init__(self, sample_rate: int = 16000, *, min_silence_s=None,
                 min_speech_s=None, frame_len: int = FRAME_LEN,
                 hop: int = HOP, gate=None, adaptive: bool = False):
        self.sr = sample_rate
        self.hop = hop
        self.frame_len = frame_len
        self.hop_s = hop / sample_rate
        self.min_silence_s = (energy.MIN_SILENCE_S
                              if min_silence_s is None else min_silence_s)
        self.min_speech_s = (energy.MIN_SPEECH_S
                             if min_speech_s is None else min_speech_s)
        # Local (moving-quantile) threshold instead of a fraction of the
        # running peak. Off by default; see energy.adaptive_threshold for the
        # measurements, including the high-quantile hypothesis that was wrong.
        self.adaptive = adaptive
        # The voice-activity gate, or None.
        #
        # None is the DEFAULT and it is a real mode, not a stub: without
        # onnxruntime or without the weights the watcher behaves exactly as it
        # always has. The gate is an optimisation that also fixes the declared
        # hum failure, and a peripheral must not refuse to run because a 2 MB
        # model is missing.
        #
        # ONE gate for the whole life of the watcher, never restarted per
        # segment. Measured on 90 s of real speech: restarting the gate per
        # segment vetoes 4/20 real speech segments at -20 dB and 8/20 at
        # -30 dB, where a continuous gate vetoes none, because the model
        # carries LSTM state and scores an isolated segment with no context
        # at its edges. See tests/test_vad.py::TestStatefulness.
        self.gate = gate
        # Rolling gate probabilities, in absolute time, kept aligned with the
        # audio buffer. These are the per-32ms-frame scores for the CURRENT
        # buffer only; trimming drops them alongside the audio.
        #
        # `_gate_probs_base` is the ABSOLUTE sample index at which
        # `_gate_probs[0]`'s frame begins, so entry j covers global samples
        # [base + j*512, base + (j+1)*512). Storing the origin explicitly is
        # what keeps this aligned: an earlier version indexed the list as if
        # index 0 were always the start of the buffer, which is only true
        # while the drop is an exact multiple of the 512-sample frame. The
        # buffer drops on the 160-sample hop lattice, so those two only agree
        # by accident -- and when they disagree the verdict is read from the
        # wrong 32 ms while still looking perfectly plausible.
        self._gate_probs: list = []
        self._gate_probs_base: int = 0
        # How many samples have been pushed through the gate in total, so the
        # gate is fed each sample exactly once even though the audio buffer is
        # trimmed and re-analysed. Feeding the whole retained buffer again
        # would double-count the tail and advance the LSTM state twice over
        # the same audio.
        self._gated_through: int = 0
        self._buf: list = []
        self._offset: int = 0          # absolute sample index of _buf[0]
        self.emitted: list = []        # everything ever emitted
        self.dropped_tail: int = 0     # times the tail was too short to emit
        # Absolute time through which events have been emitted.
        #
        # This exists because trimming alone was NOT enough. The buffer is
        # trimmed to the last emitted segment, but the segmentation is RE-DERIVED
        # from the new buffer origin on every push, and the boundaries move by a
        # frame or two because the analysis window is no longer aligned the way
        # it was. The result was catastrophic: a 2.4 s session with three
        # utterances emitted 71 events, re-reporting the same speech on almost
        # every push. Tracking the emitted time explicitly makes emission
        # idempotent regardless of how the boundaries jitter.
        self._emitted_until: float = 0.0
        # Running peak RMS. The speech threshold is derived from THIS, not from
        # the current buffer. energy.segment's per-call peak is correct for a
        # single clip and wrong for a sliding window: a quiet stretch of buffer
        # lowers the bar and the silence around it becomes "speech", and the
        # drift is invisible because each individual segmentation is
        # self-consistent. energy.segment documents the same limitation for
        # longer recordings.
        self._peak_rms: float = 0.0
        # Whether the currently-open speech run has already fired its ONSET.
        # Checkpoints of a long run must not re-charge attention, but the first
        # checkpoint of a run whose segment never closed still owes one.
        self._open_run_onset_done: bool = False
        # Absolute start of the current run of silence, or None when the voice
        # is up. Carried across checkpoints so a long pause reports its true
        # length on every event rather than only in slices. Reset by any
        # speech; see `_events_for`.
        self._pause_run_start: float | None = None
        # Reporting cadence for a pause that is still running, in seconds of
        # accumulated silence. An open pause is reported every time its run
        # crosses the next multiple of this, so downstream sees a live
        # "how long has it been quiet" rather than nothing until the silence
        # either closes or hits MAX_PAUSE_S.
        #
        # 0.25 s is measured rather than chosen: the median real pause run in
        # the corpus is 0.320 s, so a finer cadence would emit events nobody
        # can act on and a coarser one would miss the median outright.
        self._pause_reported_at: float = 0.0
        self._speaking_since_report: bool = True

    # -- time ----------------------------------------------------------------

    def _abs(self, index: int) -> float:
        """Sample index -> absolute seconds."""
        return (self._offset + index) / self.sr

    @property
    def elapsed_s(self) -> float:
        return (self._offset + len(self._buf)) / self.sr

    # -- input ---------------------------------------------------------------

    def push(self, samples) -> list:
        """Append a chunk and return the events that are now complete."""
        self._buf.extend(samples)
        n = len(self._buf)
        if n < self.frame_len:
            return []

        # Feed the gate BEFORE anything is trimmed, so each sample reaches it
        # exactly once and in order. `_gated_through` is an ABSOLUTE sample
        # count, so a trimmed-and-re-analysed buffer does not cause the tail to
        # be scored twice -- the failure mode would be the LSTM state advancing
        # over the same audio a second time, which is precisely the
        # statelessness this gate is stateful to avoid.
        #
        # `new_from` is a BUFFER index derived from two absolute counters, and
        # the first push computes it as 0. An earlier version guarded with
        # `new_from > 0`, which is false on the very first push, so the gate
        # was silently never fed and every record reported
        # `gate_speech_prob = None` -- a gate that looks installed, costs
        # nothing, and does nothing. Found by measuring the integration rather
        # than by a unit test, because every existing test constructs a
        # watcher without a gate and so cannot see it.
        if self.gate is not None:
            new_from = max(0, self._gated_through - self._offset)
            if new_from < len(self._buf):
                self._gate_probs.extend(self.gate.push(self._buf[new_from:]))
                self._gated_through = self._offset + len(self._buf)

        ef, _poisoned = energy.rms_series(self._buf, self.frame_len, self.hop,
                                          self.sr)
        self._peak_rms = max(self._peak_rms, max((f.rms for f in ef),
                                                  default=0.0))
        segs = energy.segment(ef, self.hop_s, min_silence_s=self.min_silence_s,
                              min_speech_s=self.min_speech_s,
                              peak_rms=self._peak_rms,
                              adaptive=self.adaptive)
        if not segs:
            return []

        # The stable boundary: the last frame we are sure is not mid-utterance.
        # Anything at or after it stays in the buffer for the next push.
        last = (self._offset + n) / self.sr
        cut = last - TAIL_FRAMES * self.hop_s
        events: list = []
        consumed_to = None
        for seg in segs:
            # ONE TIME BASE. energy.segment returns times relative to the start
            # of the buffer it was handed, and the buffer origin MOVES as
            # `_trim` discards emitted audio. Comparing those relative ends
            # against an absolute cut emitted still-open segments, which
            # truncated every utterance after the first and then let the
            # emitted_until guard swallow the rest of the session. 2.4 s of
            # audio with three 0.40 s utterances produced 0.40/0.10/0.10 s.
            seg_end_abs = self._abs(int(seg.end * self.sr))
            seg_start_abs = self._abs(int(seg.start * self.sr))
            is_open = False
            if seg_end_abs > cut:
                # Still open. Waiting is correct, but not forever.
                bound = MAX_PAUSE_S if not seg.speech else MAX_UTTERANCE_S
                if seg_end_abs - seg_start_abs < bound:
                    # Still open. For SPEECH, waiting is right -- an utterance
                    # that has not ended cannot be reported as one.
                    #
                    # For a PAUSE it is the opposite, and this branch was the
                    # root cause of turn detection not working at all. A
                    # pause is only ever emitted once its segment CLOSES --
                    # which means once speech resumes. A silence at the END of
                    # speech never closes, so it was never reported, and a
                    # peripheral could not learn that someone had stopped
                    # talking. Measured on real audio, a 1.30 s trailing
                    # silence produced pause events reporting run_s of 0.03
                    # and 0.08 s: the trailing silence was not measured at
                    # all, at any length.
                    #
                    # So an open pause is now REPORTED AS IT GROWS, on a
                    # fixed cadence, marked `open=True` with the accumulated
                    # `run_s`. Downstream can act on it and void the action if
                    # speech resumes -- which is exactly what
                    # `speculate.SpeculationGate` is for. Waiting for the
                    # segment to close before saying anything was correct for
                    # utterances and wrong for the silence that matters most.
                    if not seg.speech:
                        # The run start has to be established HERE as well as
                        # in `_events_for`: this path emits its own Event
                        # rather than going through the summary builder, so
                        # without this the run start stayed None and every
                        # checkpoint measured from the segment edge instead
                        # of from the beginning of the silence.
                        if self._pause_run_start is None:
                            self._pause_run_start = seg_start_abs
                        run_s = seg_end_abs - self._pause_run_start
                        step = self._next_pause_report(run_s)
                        if step is not None:
                            self._pause_reported_at = step
                            ev = Event(
                                PAUSE,
                                seg_start_abs, seg_end_abs,
                                {"duration_s": round(seg_end_abs - seg_start_abs, 4),
                                 "run_s": round(run_s, 4),
                                 "open": True,
                                 "checkpoint": True})
                            self.emitted.append(ev)
                            events.append(ev)
                    break                   # still open; re-analyse next push
                # Emit what is certainly complete and start fresh from there.
                flush_abs = cut
                cut_rel = (flush_abs * self.sr - self._offset) / self.sr
                seg = energy.Segment(start=seg.start, end=cut_rel,
                                     speech=seg.speech)
                seg_end_abs = flush_abs
                is_open = True
            if seg_start_abs < self._emitted_until - 1e-9:
                continue                    # already reported; boundaries jitter
            for ev in self._events_for(seg, ef, is_open=is_open):
                self.emitted.append(ev)
                events.append(ev)
            consumed_to = seg_end_abs
            self._emitted_until = max(self._emitted_until, seg_end_abs)
        if consumed_to is None:
            self.dropped_tail += 1
            # `events` is NOT empty here even when nothing was consumed: an
            # open pause emits checkpoints precisely when no segment has
            # closed, which is exactly this branch. Returning `[]` discarded
            # them, so a trailing silence was still never reported -- the run
            # advanced to 1.25 s internally and the caller saw nothing.
            return events
        self._trim(consumed_to)
        return events

    def note_external(self, kind: str, t0: float, t1: float = None,
                      data: dict = None, origin: str = "human") -> list:
        """
        Record an event with no acoustic signature.

        The refusal, the correction, the approval. A watcher that cannot see
        these is blind to the most consequential moments of a session, and no
        amount of acoustic fidelity fixes that.
        """
        ev = Event(kind=kind, t0=t0, t1=(t0 if t1 is None else t1),
                   data=dict(data or {}), origin=origin)
        self.emitted.append(ev)
        return [ev]

    def _trim(self, consumed_to_abs: float) -> None:
        """Drop emitted audio, keep the tail that may still be mid-utterance.

        Takes an ABSOLUTE time, matching `_abs` and `_emitted_until`.

        The drop is rounded DOWN to a whole number of hops, so the analysis
        grid stays aligned with the original stream.

        This is load-bearing, and the reason is not obvious. `energy.segment`
        places segment boundaries at frame TIMES -- window CENTRES --
        which are `(index * hop + frame_len / 2) / sr`. With FRAME_LEN = 800
        that is an offset of 400 samples, and 400 mod 160 = 80: half a hop.
        So the natural drop lands on the half-hop lattice, not the hop
        lattice.

        Measured by sabotage: removing this rounding does not change any
        utterance boundary the tests check -- it shifts `_offset` onto 80 mod
        160, which then makes every subsequent `rms_series` window start
        somewhere different. The first version of that sabotage test asserted
        the *wrong* thing (that no offset could ever be 0 again) and so
        reported a failure that had nothing to do with the real effect. The
        correct guard is that the lattice must be broken SOMEWHERE.

        Hop-aligned trimming does not make segmentation bit-exact across
        chunkings -- the chunking test's four-frame tolerance is what that
        concession costs -- but it keeps the analysis grid from migrating.
        """
        drop = int(consumed_to_abs * self.sr) - self._offset
        drop -= drop % self.hop
        if drop > 0:
            self._buf = self._buf[drop:]
            self._offset += drop
        if self.gate is not None and self._gate_probs:
            # Drop the same audio's worth of gate scores, moving the origin by
            # exactly the samples that were dropped. Only frames that START
            # before the new buffer origin go.
            #
            # `new_offset` IS `self._offset` at this point -- the line above has
            # already applied the drop. Writing `self._offset + drop` here was a
            # double count: the origin ran ahead of the buffer by one drop per
            # push, so after a few pushes `base` sat tens of thousands of
            # samples in the future and every lookup indexed negatively and
            # returned None. It is the kind of error that produces a plausible
            # partial answer (the early pushes are fine) and then silently
            # stops the gate reporting at all -- found by measuring the
            # integration end to end, not by any unit test.
            new_offset = self._offset
            j = max(0, (new_offset - self._gate_probs_base)
                    // vad.FRAME_SAMPLES)
            self._gate_probs = self._gate_probs[j:]
            self._gate_probs_base += j * vad.FRAME_SAMPLES

    # -- reporting a pause that is still running ------------------------------

    def _pause_start_or(self, seg_start_abs: float) -> float:
        """Absolute start of the silence run this open segment belongs to."""
        if self._pause_run_start is None:
            return seg_start_abs
        return self._pause_run_start

    def _next_pause_report(self, run_s: float):
        """
        The cadence boundary this run has just crossed, or None.

        None means "nothing new to say" -- which is most pushes. A gate that
        emits on every push is a gate nobody downstream can use, because
        everything it says is the same thing again.

        Monotone by construction: the boundary is `floor(run/interval) *
        interval`, so it only ever moves forward and each crossing fires
        exactly once. The first version of this computed the NEXT multiple
        and then subtracted, which could hand back a boundary already passed,
        and the run stalled at 0.30 s however long the silence got.
        """
        boundary = int(run_s / PAUSE_REPORT_S) * PAUSE_REPORT_S
        if boundary <= self._pause_reported_at + 1e-9:
            return None
        return boundary

    # -- gate ---------------------------------------------------------------

    def _gate_peak(self, seg) -> float | None:
        """
        The gate's peak probability over a segment, or None if no gate ran.

        `seg` times are RELATIVE to the current buffer, so they are converted
        to ABSOLUTE samples with `self._offset` before being indexed against
        `_gate_probs_base`. That indirection is the whole point: the buffer
        and the probability list are trimmed on different lattices (160-sample
        hops versus 512-sample frames), so an index that is right in one is
        wrong in the other. Both are addressed absolutely and neither is
        assumed to agree with the other.
        """
        if self.gate is None or not self._gate_probs:
            return None
        a0 = self._offset + int(seg.start * self.sr)
        a1 = self._offset + int(seg.end * self.sr)
        i0 = (a0 - self._gate_probs_base) // vad.FRAME_SAMPLES
        i1 = -(-(a1 - self._gate_probs_base) // vad.FRAME_SAMPLES)   # ceil
        i0 = max(0, i0)
        i1 = min(len(self._gate_probs), i1)
        if i1 <= i0:
            return None
        return max(self._gate_probs[i0:i1])

    # -- per-segment summary -------------------------------------------------

    def _events_for(self, seg, ef, is_open: bool = False) -> list:
        """
        One segment -> its events, carrying the summary the baseline table
        needs so nothing downstream has to re-analyse the audio.

        An utterance yields TWO events: the ONSET, which spends attention and
        is what design v3 §2 makes the seam, and the UTTERANCE itself, which
        is the record. They are separate because they go to different layers:
        the onset to the judged layer, the utterance to the tables.

        A CHECKPOINT (`is_open`) yields only the UTTERANCE, marked `open`.
        Emitting an ONSET again would be a real defect rather than a cosmetic
        one: design v3 §2 makes the onset the event that spends attention, so
        a 30 s monologue flushing every 2 s would re-charge attention five
        times for one occasion. The onset was already reported; the
        checkpoint only carries the record forward.

        A pause yields one, carrying only its duration, because there is
        nothing else to summarise about silence.
        """
        t0 = self._abs(int(seg.start * self.sr))
        t1 = self._abs(int(seg.end * self.sr))
        if not seg.speech:
            # Any pause ends the current speech run, so the next one owes an
            # onset of its own.
            self._open_run_onset_done = False
            # `run_s` is how long THIS SILENCE RUN has lasted so far, across
            # however many checkpoints it took to report it.
            #
            # Without it a long pause is unreadable. A pause is only emitted
            # once its segment closes behind the tail cut, and an open one is
            # flushed at MAX_PAUSE_S, so a 6 s silence arrives as three events
            # of 2.755 / 2.775 / 0.360 s -- each below any threshold a turn
            # decision would use, summing to roughly the right total and
            # carrying NO event that says "this was six seconds".
            #
            # Measured on the real corpus before this existed: the longest
            # pause the watcher emitted across 465 s of speech was 0.790 s,
            # so a 3 s turn threshold could never have matched anything. The
            # segmentation was right and the reporting was not.
            if self._pause_run_start is None:
                self._pause_run_start = t0
            self._pause_reported_at = t1 - self._pause_run_start
            run_s = t1 - self._pause_run_start
            return [Event(PAUSE, t0, t1, {"duration_s": seg.duration(),
                                           "run_s": round(run_s, 4),
                                           "open": is_open})]

        self._pause_run_start = None      # the voice is up; the next pause starts fresh
        self._pause_reported_at = 0.0    # and the reporting cadence resets

        i0 = max(0, int(seg.start / self.hop_s) - 1)
        i1 = min(len(ef), int(seg.end / self.hop_s) + 1)
        frames = ef[i0:i1]
        window = self._buf[int(seg.start * self.sr):int(seg.end * self.sr)]

        # THE GATE'S ONE JOB: decide whether running YIN over this segment is
        # worth the CPU. It does not and may not decide whether the segment is
        # reported -- measured, a 200 Hz harmonic stack scores 0.97, so this
        # gate is not a speech/no-speech classifier and must never be treated
        # as one. The verdict is recorded either way, so a segment the gate
        # dismissed can be audited afterwards.
        #
        # `max` over the segment's frames, not `mean`: measured over 20 real
        # speech segments, `max` vetoed 0/20 and `mean` vetoed 1/20, because a
        # segment's own quiet edges drag its average down. The cost is that a
        # single spurious frame can clear the threshold -- which is the safe
        # direction, since it means we do the work we did not need to rather
        # than skip the work that mattered.
        gate_prob = self._gate_peak(seg)
        skip_pitch = gate_prob is not None and gate_prob < GATE_SPEECH

        tracker = None
        pf = []
        if window and not skip_pitch:
            tracker = pitch.PitchTracker(self.sr, frame_len=self.frame_len,
                                         hop=self.hop)
            pf = tracker.track(window)
        elif not window:
            pass
        hz = [f.hz for f in pf if f.hz is not None]
        inflections = inflection.contour_events(pf) if pf else []
        rms = [f.rms for f in frames]
        peak = max(rms) if rms else 0.0

        record = Event(UTTERANCE, t0, t1, {
            "duration_s": round(seg.duration(), 4),
            "open": is_open,
            "energy_peak": round(peak, 6),
            "energy_mean": round(sum(rms) / len(rms), 6) if rms else 0.0,
            "voiced_frames": len(hz),
            "total_frames": len(pf),
            "voiced_ratio": round(len(hz) / len(pf), 4) if pf else 0.0,
            "hz_p05": round(_pct(hz, 0.05), 2) if hz else None,
            "hz_p95": round(_pct(hz, 0.95), 2) if hz else None,
            "inflections": len(inflections),
            "quiet_frames": tracker.quiet_frames if tracker is not None else 0,
            # Always present, even when the gate was off. `None` means "no gate
            # ran", which is a different fact from 0.0 meaning "the gate ran
            # and saw silence".
            GATE_FIELD: gate_prob,
            "pitch_skipped": skip_pitch,
        })
        if is_open and self._open_run_onset_done:
            # Second and later checkpoints of the SAME run. The onset already
            # fired; design v3 §2 makes it the event that spends attention, so
            # re-emitting it would charge one occasion several times.
            return [record]
        onset = Event(ONSET, t0, t0, {"speech": True})
        # Remember only while the run is still open, so a checkpoint that
        # arrives without ever having closed its segment still gets its onset.
        self._open_run_onset_done = is_open
        return [onset, record]


def _pct(values, q: float):
    """Percentile of a sorted copy. Returns 0.0 for an empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)