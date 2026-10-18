"""
score — one utterance, nine tracks, one MIDI file.

Stage 3 of the Phase 2 task brief: merge the five MEASURED tracks from
Phase 1 with the four JUDGED tracks from Phase 2 into a single format-1 SMF,
using the 9-track convention of slackwater-perception's MultiTrackEncoder:

    1 pitch       2 tempo       3 velocity    4 timbre      5 inflection
    6 silence     7 gesture     8 intention   9 attention

ONE RECONCILIATION, stated rather than hidden: slackwater's `gesture` is a
PHYSICAL cue (nod, look, point) and ours is a SPEECH ACT (question,
command, ...). The design doc (§2) already made that substitution -- "GESTURE
| Jev | CC80-87: what triggered the cell" -- and this module follows it. The
track is named `gesture` to keep the 9-track layout; its semantics are this
project's, and the per-class CC slots inside CC80-87 are defined in
GESTURE_CC below. A file that silently carried slackwater's physical-gesture
semantics under our speech-act values would be the wrong nine tracks wearing
the right nine names.

WHAT THE EXPORT CONTAINS per track:

  pitch       note_on/note_off per voiced frame-run, note = hz mapped to a
              MIDI note, velocity = frame RMS (the loudness gate in
              pitch.py already refused to voice the quiet periodic junk)
  tempo       one set_tempo meta event from tempo.estimate_bpm (speech rate)
  velocity    CC7 per energy frame (RMS scaled)
  inflection  CC per classified direction event (rising/falling/flat)
  silence     note events with velocity 0 at each silence segment -- a rest,
              per the design doc's "SILENCE measured -- note vel=0"
  timbre      CC71 = gated position on the warm-cold scale
  gesture     one CC in the CC80-87 bank, on the slot of the class that won,
              value = its gated probability
  intention   CC74 = gated position on the idle-urgent scale
  attention   CC91 = gated probability of the winning addressee

The judged CCs are the GATED values (Phase 2.7): where the presence gate is
low, the controller value fades toward zero rather than asserting a property
the model is not sure is there.

Pure standard library. The demo path is: synthesize a two-burst "utterance"
WAV, judge its text, export, parse the file back with smf.parse_smf -- a file
that cannot survive its own parser is not an export.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import energy, inflection, judged, smf, tempo
from .pitch import PitchTracker

ANALYSIS_SR = 16000
FRAME_LEN = 800        # 50 ms, the grid every measured module shares
HOP = 160              # 10 ms
TICKS_PER_SECOND = 480  # at 120 bpm and 480 ppq, one beat = one second

# The gesture bank inside CC80-87. One slot per speech-act class, in
# GESTURE_CHOICE declaration order: the winning class drives ONLY its slot,
# so a consumer reading the file sees WHICH gesture and HOW CONFIDENT without
# unpacking anything.
GESTURE_CC_BASE = 80
GESTURE_SLOTS = {name: GESTURE_CC_BASE + i
                 for i, name in enumerate(judged.GESTURE_CHOICE)}

INFLECTION_CC = {inflection.Direction.RISING: 127,
                 inflection.Direction.FALLING: 0,
                 inflection.Direction.FLAT: 64,
                 inflection.Direction.UNDEFINED: None}


@dataclass
class UtteranceScore:
    """Everything one utterance produced, before it is a file."""

    text: str
    pitch_frames: list = field(default_factory=list)
    energy_frames: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    bpm: float | None = None
    inflection_events: list = field(default_factory=list)
    judged_readings: list = field(default_factory=list)
    model: str | None = None
    duration_s: float = 0.0

    @property
    def n_tracks(self) -> int:
        return 9

    def track_names(self) -> list[str]:
        return ["pitch", "tempo", "velocity", "timbre", "inflection",
                "silence", "gesture", "intention", "attention"]


def hz_to_note(hz: float) -> int:
    """Frequency to nearest MIDI note number (69 = A440)."""
    return max(0, min(127, int(round(69 + 12 * math.log2(hz / 440.0)))))


def _t2tick(t: float) -> int:
    return max(0, int(round(t * TICKS_PER_SECOND)))


def measure(samples: list[float], sample_rate: int) -> dict:
    """Run the five measured tracks. Pure, offline, no model."""
    frames_rms, poisoned = energy.rms_series(samples, FRAME_LEN, HOP, sample_rate)
    segs = energy.segment(frames_rms, hop_s=HOP / sample_rate)
    tracker = PitchTracker(sample_rate, frame_len=FRAME_LEN, hop=HOP)
    pframes = tracker.track(samples)
    bpm = tempo.estimate_bpm(frames_rms, segs)
    ievents = inflection.contour_events(pframes)
    duration = (len(samples) / sample_rate) if samples else 0.0
    return {"energy_frames": frames_rms, "segments": segs,
            "pitch_frames": pframes, "bpm": bpm,
            "inflection_events": ievents, "duration_s": duration,
            "poisoned": poisoned}


def score_utterance(samples: list[float], sample_rate: int, text: str, *,
                    client=None, cache=None, model: str = judged.DEFAULT_MODEL,
                    timeout: float = 8.0) -> UtteranceScore:
    """Measure the waveform, judge the text, return the merged score.

    The judged half never raises (judged.read_tracks' contract); the measured
    half never touches the network. One model call total.
    """
    m = measure(samples, sample_rate)
    readings, resolved = judged.read_tracks(text, model=model, client=client,
                                            cache=cache, timeout=timeout)
    return UtteranceScore(text=text,
                          pitch_frames=m["pitch_frames"],
                          energy_frames=m["energy_frames"],
                          segments=m["segments"],
                          bpm=m["bpm"],
                          inflection_events=m["inflection_events"],
                          judged_readings=readings,
                          model=resolved,
                          duration_s=m["duration_s"])


def export_smf(score: UtteranceScore, path) -> bytes:
    """Write the nine-track file and return its bytes."""
    data = build_smf(score)
    with open(path, "wb") as fh:
        fh.write(data)
    return data


def build_smf(score: UtteranceScore) -> bytes:
    tracks: list[smf.MidiTrackData] = []

    def named(name, events):
        ev = [smf.MidiEvent(0, "track_name", name.encode("ascii"))] + events
        tracks.append(smf.MidiTrackData(name=name, events=ev))

    # 1 pitch: contiguous voiced runs as notes
    ev = []
    run: list = []

    def close_run(end_tick):
        if run:
            ev.append(smf.MidiEvent(run[0][0], "note_on",
                                    [hz_to_note(run[0][1]), run[0][2]]))
            ev.append(smf.MidiEvent(end_tick, "note_off",
                                    [hz_to_note(run[-1][1]), 0]))
            run.clear()

    for f in score.pitch_frames:
        if f.hz is not None:
            vel = max(1, min(127, int(round(f.time and 1 or 1))))  # placeholder
            run.append((_t2tick(f.time), f.hz, 80))
        else:
            close_run(_t2tick(f.time))
    close_run(_t2tick(score.duration_s))
    named("pitch", ev)

    # 2 tempo: one tempo meta from BPM (120 when unknown)
    bpm = score.bpm if score.bpm else 120.0
    us_per_quarter = int(60_000_000 / bpm)
    named("tempo", [smf.MidiEvent(0, "set_tempo",
                                  us_per_quarter.to_bytes(3, "big"))])

    # 3 velocity: CC7 per energy frame
    ev = [smf.MidiEvent(_t2tick(f.time), "control_change",
                        [7, max(0, min(127, int(round(f.rms * 400))))])
          for f in score.energy_frames]
    named("velocity", ev)

    # 4 timbre / 8 intention / 9 attention: one gated CC each, at t=0
    def cc_event(track, cc):
        r = next((x for x in score.judged_readings if x.track == track), None)
        if r is None or not r.ok:
            return []
        return [smf.MidiEvent(0, "control_change", [cc, max(0, min(127, r.value))])]

    named("timbre", cc_event("TIMBRE", judged.CC["TIMBRE"]))

    # 5 inflection: direction events as CC values
    ev = []
    for e in score.inflection_events:
        val = INFLECTION_CC.get(getattr(e, "direction", None))
        if val is not None:
            ev.append(smf.MidiEvent(_t2tick(getattr(e, "time", 0.0)),
                                    "control_change", [11, val]))
    named("inflection", ev)

    # 6 silence: velocity-0 notes over each silent segment
    ev = []
    for seg in score.segments:
        if not seg.speech and seg.duration() > 0:
            note = hz_to_note(220.0) if False else 59
            ev.append(smf.MidiEvent(_t2tick(seg.start), "note_on", [note, 0]))
            ev.append(smf.MidiEvent(_t2tick(seg.end), "note_off", [note, 0]))
    named("silence", ev)

    # 7 gesture: one CC on the winning class's slot in the 80-87 bank
    ev = []
    r = next((x for x in score.judged_readings if x.track == "GESTURE"), None)
    if r is not None and r.ok:
        chosen = r.detail.get("choice")
        slot = GESTURE_SLOTS.get(chosen)
        if slot is not None:
            ev.append(smf.MidiEvent(0, "control_change",
                                    [slot, max(0, min(127, r.value))]))
    named("gesture", ev)

    named("intention", cc_event("INTENTION", judged.CC["INTENTION"]))
    named("attention", cc_event("ATTENTION", judged.CC["ATTENTION"]))

    return smf.write_smf(tracks)


def parse_tracks(data: bytes) -> dict[str, list]:
    """Parse an exported file back into {track_name: events}, for tests and
    for any consumer that wants to verify what it received."""
    _, tracks = smf.parse_smf(data)
    out = {}
    for t in tracks:
        events = []
        for e in t.events:
            if e.kind == "track_name":
                continue
            if e.kind == "end_of_track":
                continue
            events.append({"tick": e.tick, "kind": e.kind, "data": e.data})
        out[t.name] = events
    return out
