"""
watcher -- the continuous layer. No model, no network, microseconds.

This is the first component of design v3 §3 step 1, and it is the one that
makes the rest possible, because it is the only layer that can run always-on.
The measurements forced that: judged tracks cost 0.97 s for four gates and
2.45 s for eight locally, so nothing costing a second may sit in the
continuous path.

WHAT "WATCH" MEANS HERE IS NOT A PASS OVER A FILE.

A whole-clip pass is not watching. This takes a growing buffer, analyses it,
and emits events as soon as an utterance is *complete*, retaining a tail that
might still be mid-utterance. That is what lets it sit behind a live mic
forever, and it is why the API is `push()` rather than `analyse()`.

    w = Watcher(sample_rate=16000)
    for chunk in stream:                      # 100 ms chunks, say
        for ev in w.push(chunk):
            ...                                # utterance, pause, or onset

The re-analysis of the retained tail is deliberate and cheap: it costs one
extra tail per push rather than a clipped utterance, and a truncated final
word is the one error that would poison a baseline table forever.

THE EVENTS ARE THE THING. Design v3 §2 made the seam the EVENT rather than the
turn, so this is the component that defines them:

* ``onset``     -- a voiced onset after >= MIN_SILENCE_S of silence. This is
                   the event that spends attention.
* ``utterance`` -- a complete run of speech, with the measured summary the
                   §4.1 baseline table needs.
* ``pause``     -- a complete run of silence, with its duration.

A fourth event, a *human decision*, has no acoustic signature at all and so
cannot be detected here. It is `note_external()` — and its existence is the
reminder that a system which only listens cannot see the most important events
it will ever receive (a refusal, a correction, an approval).

KNOWN LIMIT, STATED BECAUSE IT IS THE DANGEROUS ONE: a steady periodic hum
above the energy floor -- a fan, HVAC, a held vowel -- is indistinguishable
from voiced speech here. `pitch.PitchTracker` rejects frames *below*
ENERGY_FLOOR, not frames that are periodic and loud. So a noisy room reads as
one continuous utterance rather than as silence, and every baseline built from
it would be wrong in the same direction. This is the single most likely
real-world failure and it is not fixed, only declared. The real-speech fixture
lane (LANE-FIXTURES.md) is the thing that will catch it.

Everything here is arithmetic. No model, no I/O, no clock. Which is also what
makes it runnable on a microcontroller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import energy, inflection, pitch

# Kept identical to the measured tracks' grid so no resampling is implied.
FRAME_LEN = 800            # 50 ms at 16 kHz
HOP = 160                  # 10 ms
HOP_S = HOP / 16000.0

# Frames retained after the last emitted event, so an utterance that is still
# open is re-analysed rather than truncated. 300 ms covers the longest
# plausible phrase-in-progress at normal speech rate.
TAIL_FRAMES = 30

ONSET = "onset"
UTTERANCE = "utterance"
PAUSE = "pause"
DECISION = "decision"


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
                 hop: int = HOP):
        self.sr = sample_rate
        self.hop = hop
        self.frame_len = frame_len
        self.hop_s = hop / sample_rate
        self.min_silence_s = (energy.MIN_SILENCE_S
                              if min_silence_s is None else min_silence_s)
        self.min_speech_s = (energy.MIN_SPEECH_S
                             if min_speech_s is None else min_speech_s)
        self._buf: list = []
        self._offset: int = 0          # absolute sample index of _buf[0]
        self.emitted: list = []        # everything ever emitted
        self.dropped_tail: int = 0     # times the tail was too short to emit

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

        ef, _poisoned = energy.rms_series(self._buf, self.frame_len, self.hop,
                                          self.sr)
        segs = energy.segment(ef, self.hop_s, min_silence_s=self.min_silence_s,
                              min_speech_s=self.min_speech_s)
        if not segs:
            return []

        # The stable boundary: the last frame we are sure is not mid-utterance.
        # Anything at or after it stays in the buffer for the next push.
        last = (self._offset + n) / self.sr
        cut = last - TAIL_FRAMES * self.hop_s
        events: list = []
        consumed_to = None
        for seg in segs:
            if seg.end > cut:
                break                       # still open; re-analyse next push
            for ev in self._events_for(seg, ef):
                self.emitted.append(ev)
                events.append(ev)
            consumed_to = seg.end
        if consumed_to is None:
            self.dropped_tail += 1
            return []
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

    def _trim(self, consumed_to: float) -> None:
        """Drop emitted audio, keep the tail that may still be mid-utterance."""
        drop = int(consumed_to * self.sr) - self._offset
        if drop > 0:
            self._buf = self._buf[drop:]
            self._offset += drop

    # -- per-segment summary -------------------------------------------------

    def _events_for(self, seg, ef) -> list:
        """
        One segment -> its events, carrying the summary the baseline table
        needs so nothing downstream has to re-analyse the audio.

        An utterance yields TWO events: the ONSET, which spends attention and
        is what design v3 §2 makes the seam, and the UTTERANCE itself, which
        is the record. They are separate because they go to different layers:
        the onset to the judged layer, the utterance to the tables.

        A pause yields one, carrying only its duration, because there is
        nothing else to summarise about silence.
        """
        t0 = self._abs(int(seg.start * self.sr))
        t1 = self._abs(int(seg.end * self.sr))
        if not seg.speech:
            return [Event(PAUSE, t0, t1, {"duration_s": seg.duration()})]

        i0 = max(0, int(seg.start / self.hop_s) - 1)
        i1 = min(len(ef), int(seg.end / self.hop_s) + 1)
        frames = ef[i0:i1]
        window = self._buf[int(seg.start * self.sr):int(seg.end * self.sr)]
        tracker = pitch.PitchTracker(self.sr, frame_len=self.frame_len,
                                     hop=self.hop)
        pf = tracker.track(window) if window else []
        hz = [f.hz for f in pf if f.hz is not None]
        inflections = inflection.contour_events(pf) if pf else []
        rms = [f.rms for f in frames]
        peak = max(rms) if rms else 0.0

        return [
            Event(ONSET, t0, t0, {"speech": True}),
            Event(UTTERANCE, t0, t1, {
                "duration_s": round(seg.duration(), 4),
                "energy_peak": round(peak, 6),
                "energy_mean": round(sum(rms) / len(rms), 6) if rms else 0.0,
                "voiced_frames": len(hz),
                "total_frames": len(pf),
                "voiced_ratio": round(len(hz) / len(pf), 4) if pf else 0.0,
                "hz_p05": round(_pct(hz, 0.05), 2) if hz else None,
                "hz_p95": round(_pct(hz, 0.95), 2) if hz else None,
                "inflections": len(inflections),
                "quiet_frames": tracker.quiet_frames,
            }),
        ]


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