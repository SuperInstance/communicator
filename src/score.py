"""
score -- the nine tracks, in one file, in one write.

THIS IS THE ASSET. Everything else in this repo exists to produce this object
and to read it back.

The design's central claim is that the durable record is the MIDI score and
not the transcript, because a score is re-analysable and a transcript is
lossy: once the words are gone, the pitch contour, the pauses and the
loudness envelope are gone with them.

PROVENANCE OF THIS FILE, because it was rewritten rather than written:

* The 9-track list, the CC80-87 gesture bank (one slot per speech-act class,
  so a reader sees WHICH gesture as well as how confident), and `parse_tracks`
  -- a READER, without which the re-analysis claim is just a slogan -- came
  from an earlier draft of this module.
* That draft had never been executed. Its first run raised
  `TypeError: byte indices must be integers or slices, not str`, because it
  called `MidiEvent(0, "track_name", name.encode("ascii"))` while the real
  signature is `MidiEvent(tick, kind, data: dict)`. It also defaulted a
  missing tempo to 120, scaled loudness by a bare `rms * 400`, and carried
  two dead placeholder expressions (`if False else 59`, a `vel =` line that
  computed nothing and was never used).

  So the lesson is the one this repo keeps re-learning: a module can be
  complete, documented, committed and published and still have never run.
  Nothing here is trusted because it is written -- `tests/test_score.py`
  drives a real WAV through to real bytes and parses them back.

TWO DECISIONS THAT COST SOMETHING, both deliberate:

1. **ONE TICK IS ONE MILLISECOND.** ppq 1000 with set_tempo 1,000,000 us/beat
   puts the tick on the millisecond exactly. The usual ppq 480 at 120 BPM
   makes a tick 1.0417 ms, so every note time is a rounding of a real time
   and re-deriving the acoustic tracks from the file drifts. At 1 tick = 1 ms
   the round trip is exact, and a future rewrite of the perception layer can
   be re-run against the same file with no error budget spent on quantisation.

2. **ONE PROPERTY PER TRACK.** Each of the nine carries exactly one thing so
   any of them can be re-analysed without the others being interpreted first.
   Folding loudness into note velocity is the idiomatic MIDI choice and it
   would weld VELOCITY to PITCH: you could no longer ask what the pitch was
   without also reading the loudness. So PITCH is a note stream at constant
   velocity and VELOCITY is CC7.

   Separation is by TRACK, not by channel, and that is forced by the writer:
   smf._encode_event emits a fixed status byte with no channel nibble, so a
   `channel` key in an event's data would be silently dropped and would imply
   a separation the file does not contain. Separate MTrk chunks are the
   encoding that actually survives the round trip.

WHAT IS NOT CLAIMED. The judged tracks are one model's reading of a text
utterance. Phase 2.7 measured that TIMBRE cannot be read from words at all --
its presence gate returns ~0.13 on real speech -- so a TIMBRE CC of 0 here is
a measurement of its own absence, not a missing measurement. That is recorded
in the provenance sidecar rather than hidden, because a score that quietly
contains a null track is worse than one that says so.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

from . import energy, inflection, judged, pitch, smf, tempo, wavio

# -- the clock ---------------------------------------------------------------

TICKS_PER_BEAT = 1000
TEMPO_US = 1_000_000        # 1000 ticks/beat, 1e6 us/beat -> 1 tick = 1 ms
MS_PER_TICK = 1000 / TICKS_PER_BEAT


def ms_to_ticks(seconds: float) -> int:
    """Seconds -> tick. One tick is one millisecond, by construction."""
    return int(round(seconds * 1000.0))


def ticks_to_ms(tick: int) -> float:
    return tick * MS_PER_TICK


# -- the nine ---------------------------------------------------------------

ANALYSIS_SR = 16000
FRAME_LEN = 800              # 50 ms, the grid every measured module shares
HOP = 160                    # 10 ms

TRACK_NAMES = ["pitch", "tempo", "velocity", "timbre", "inflection",
               "silence", "gesture", "intention", "attention"]

# measured vs judged is THE central design decision, so it is a constant a
# reader can import rather than a comment.
MEASURED = {"pitch", "tempo", "velocity", "inflection", "silence"}
JUDGED = {"timbre", "intention", "attention", "gesture"}

CH = {"pitch": 0, "velocity": 1, "silence": 2, "inflection": 3,
      "tempo": 4, "timbre": 5, "intention": 6, "attention": 7, "gesture": 8}

# slackwater's mapping: a direction is a controller value, not a bend offset.
INFLECTION_CC = {inflection.Direction.RISING: 127,
                 inflection.Direction.FALLING: 0,
                 inflection.Direction.FLAT: 64,
                 inflection.Direction.UNDEFINED: None}
INFLECTION_CC_NUM = 11

# The gesture bank inside CC80-87: one slot per speech-act class, in
# GESTURE_CHOICE declaration order. The winning class drives ONLY its own
# slot, so a consumer sees WHICH gesture and HOW CONFIDENT without unpacking
# anything -- and the other slots read 0, which is itself information.
GESTURE_CC_BASE = 80
GESTURE_SLOTS = {name: GESTURE_CC_BASE + i
                 for i, name in enumerate(judged.GESTURE_CHOICE)}

VELOCITY_CC = 7

# Measured speech rate. 60-240 BPM mapped onto 0-127; the CC is omitted when
# no tempo was measurable, so its absence means "not measured" rather than
# "60 BPM".
TEMPO_CC = 120


@dataclass
class UtteranceScore:
    """Everything one utterance produced, before it is a file."""
    text: str = ""
    pitch_frames: list = field(default_factory=list)
    energy_frames: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    bpm: float | None = None
    inflection_events: list = field(default_factory=list)
    judged_readings: list = field(default_factory=list)
    model: str | None = None
    duration_s: float = 0.0
    sample_rate: int = ANALYSIS_SR
    poisoned_samples: int = 0

    @property
    def n_tracks(self) -> int:
        return 9

    def track_names(self) -> list:
        return list(TRACK_NAMES)

    def provenance(self) -> dict:
        return {"measured": sorted(MEASURED), "judged": sorted(JUDGED)}


def hz_to_note(hz: float) -> int:
    """Hz -> MIDI note number (69 = A440), clamped to the keyboard."""
    return max(0, min(127, int(round(69 + 12 * math.log2(hz / 440.0)))))


# -- the measured half -------------------------------------------------------

def measure(samples: list, sample_rate: int = ANALYSIS_SR) -> dict:
    """
    Run the five measured tracks. Pure arithmetic: no model, no network.

    This half is what runs on the UNO Q, so it has to stay cheap and
    deterministic -- it is the only half guaranteed to work with the network
    down.
    """
    eframes, poisoned = energy.rms_series(samples, FRAME_LEN, HOP, sample_rate)
    segs = energy.segment(eframes, hop_s=HOP / sample_rate)
    tracker = pitch.PitchTracker(sample_rate, frame_len=FRAME_LEN, hop=HOP)
    pframes = tracker.track(samples)
    return {
        "energy_frames": eframes,
        "segments": segs,
        "pitch_frames": pframes,
        "bpm": tempo.estimate_bpm(eframes, segs),
        "inflection_events": inflection.contour_events(pframes),
        "duration_s": (len(samples) / sample_rate) if samples else 0.0,
        "poisoned": poisoned,
        "peak_rms": max((f.rms for f in eframes), default=0.0),
    }


# -- the SMF -----------------------------------------------------------------

def _named(name: str, events: list) -> smf.MidiTrackData:
    """
    One track, with its name written INTO the file as a track_name meta
    event. data is a dict -- MidiEvent's signature is (tick, kind, data), and
    passing bytes here is what stopped the previous draft from ever running.
    """
    head = [smf.MidiEvent(0, "track_name", {"name": name})]
    return smf.MidiTrackData(name=name, events=head + events)


def _pitch_events(score: UtteranceScore) -> list:
    """
    Contiguous voiced runs as notes, at CONSTANT velocity.

    Unvoiced frames close the current run and emit nothing. hz is None
    precisely so "no measurement" cannot be confused with "measured zero",
    and note 0 is a real, audible, wrong answer -- the bottom of the keyboard.
    """
    ev: list = []
    run: list = []

    def close(end_tick: int):
        if not run:
            return
        ev.append(smf.MidiEvent(run[0][0], "note_on",
                                {"note": hz_to_note(run[0][1]), "velocity": 64}))
        ev.append(smf.MidiEvent(max(end_tick, run[0][0] + 1), "note_off",
                                {"note": hz_to_note(run[-1][1]), "velocity": 0}))
        run.clear()

    for f in score.pitch_frames:
        t = ms_to_ticks(f.time)
        if f.hz is not None:
            run.append((t, f.hz))
        else:
            close(t)
    close(ms_to_ticks(score.duration_s))
    return ev


def _tempo_events(score: UtteranceScore) -> list:
    """
    Two different things, kept apart on purpose.

    1. The CLOCK. A set_tempo of 1,000,000 us/beat is written
       UNCONDITIONALLY, because that event is what makes one tick one
       millisecond. A reader that finds no tempo meta event assumes the
       standard 500,000 us/beat, which at ppq 1000 means a tick is 0.5 ms --
       so "one tick is one millisecond" would be true of this module and
       false of the file, and every time re-derived from the score would be
       off by a factor of two. The clock is a property of the encoding, not
       of the utterance, so it does not depend on whether tempo was measured.

    2. The MEASUREMENT. Speech rate goes on CC120 as a controller value,
       60-240 BPM mapped onto 0-127, and the CC is OMITTED entirely when no
       tempo was measurable.

    Not 120. The earlier draft wrote `bpm if bpm else 120.0`, which is a
    fabricated measurement that the next reader re-derives as real -- and it
    did so by silently redefining the clock for every other track too.
    """
    events = [smf.MidiEvent(0, "set_tempo", {"tempo": TEMPO_US})]
    bpm = score.bpm
    if bpm and bpm > 0:
        bpm = min(240.0, max(60.0, float(bpm)))
        events.append(smf.MidiEvent(0, "control_change",
                                    {"control": TEMPO_CC,
                                     "value": int(round(
                                         127.0 * (bpm - 60.0) / 180.0))}))
    return events


def _velocity_events(score: UtteranceScore) -> list:
    """
    Loudness as CC7, normalised to the clip's own peak.

    Normalised, not absolute and not `rms * 400`: a magic multiplier makes a
    quiet recording invisible and a loud one saturate, so the track would
    encode the recording's gain rather than its envelope. energy.rms_series
    already guarantees a finite value -- the NaN guard lives there, where the
    non-finite samples actually are.
    """
    peak = max((f.rms for f in score.energy_frames), default=0.0)
    if peak <= 0:
        return []
    return [smf.MidiEvent(ms_to_ticks(f.time), "control_change",
                          {"control": VELOCITY_CC,
                           "value": int(round(127.0 * min(1.0, max(
                               0.0, f.rms / peak))))})
            for f in score.energy_frames]


def _inflection_events(score: UtteranceScore) -> list:
    ev = []
    for e in score.inflection_events:
        val = INFLECTION_CC.get(getattr(e, "direction", None))
        if val is None:                      # UNDEFINED: no measurement
            continue
        ev.append(smf.MidiEvent(ms_to_ticks(getattr(e, "time", 0.0)),
                                "control_change",
                                {"control": INFLECTION_CC_NUM, "value": val}))
    return ev


def _silence_events(score: UtteranceScore) -> list:
    """
    slackwater's convention: silence is a note with velocity 0.

    One note per segment -- sustained through speech, velocity 0 through
    silence -- so the track reads as a mask without a per-frame note storm.
    """
    ev = []
    note = 59                                   # B3, one fixed carrier
    for seg in score.segments:
        start, end = ms_to_ticks(seg.start), ms_to_ticks(seg.end)
        if end <= start:
            continue
        ev.append(smf.MidiEvent(start, "note_on",
                                {"note": note,
                                 "velocity": 127 if seg.speech else 0}))
        ev.append(smf.MidiEvent(end, "note_off",
                                {"note": note, "velocity": 0}))
    return ev


def _judged_cc(score: UtteranceScore, track: str, cc: int) -> list:
    r = next((x for x in score.judged_readings if x.track == track), None)
    if r is None or not r.ok:
        return []
    return [smf.MidiEvent(0, "control_change",
                          {"control": cc, "value": int(max(0, min(127, r.value)))})]


def _gesture_events(score: UtteranceScore) -> list:
    r = next((x for x in score.judged_readings if x.track == "GESTURE"), None)
    if r is None or not r.ok:
        return []
    slot = GESTURE_SLOTS.get(r.detail.get("choice"))
    if slot is None:
        return []
    return [smf.MidiEvent(0, "control_change",
                          {"control": slot, "value": int(max(0, min(127, r.value)))})]


def build_smf(score: UtteranceScore) -> bytes:
    """The nine-track file."""
    tracks = [
        _named("pitch", _pitch_events(score)),
        _named("tempo", _tempo_events(score)),
        _named("velocity", _velocity_events(score)),
        _named("timbre", _judged_cc(score, "TIMBRE", judged.CC["TIMBRE"])),
        _named("inflection", _inflection_events(score)),
        _named("silence", _silence_events(score)),
        _named("gesture", _gesture_events(score)),
        _named("intention", _judged_cc(score, "INTENTION", judged.CC["INTENTION"])),
        _named("attention", _judged_cc(score, "ATTENTION", judged.CC["ATTENTION"])),
    ]
    return smf.write_smf(tracks, ticks_per_beat=TICKS_PER_BEAT)


def parse_tracks(data: bytes) -> dict:
    """
    Parse an exported file back to {track_name: events}.

    The reader half of the thesis. A score that cannot be read back is a
    screenshot; this is what makes the acoustic evidence re-analysable when
    the perception code is rewritten later.
    """
    _, parsed = smf.parse_smf(data)
    out = {}
    for t in parsed:
        events = []
        for e in t.events:
            if e.kind in ("track_name", "end_of_track"):
                continue
            events.append({"tick": e.tick, "kind": e.kind, "data": e.data})
        out[t.name] = events
    return out


def export_smf(score: UtteranceScore, path: str) -> bytes:
    data = build_smf(score)
    with open(path, "wb") as fh:
        fh.write(data)
    return data


# -- assembly ----------------------------------------------------------------

def score_utterance(samples: list, text: str, *, sample_rate: int = ANALYSIS_SR,
                    client=None, cache=None, model: str = judged.DEFAULT_MODEL,
                    timeout: float = 8.0) -> UtteranceScore:
    """
    Measure the waveform, judge the text, return the merged score.

    The judged half never raises (judged.read_tracks' contract); the measured
    half never touches the network. One model call total.
    """
    m = measure(samples, sample_rate)
    readings, resolved = judged.read_tracks(text, model=model, client=client,
                                            cache=cache, timeout=timeout)
    return UtteranceScore(
        text=text, pitch_frames=m["pitch_frames"],
        energy_frames=m["energy_frames"], segments=m["segments"],
        bpm=m["bpm"], inflection_events=m["inflection_events"],
        judged_readings=readings, model=resolved, duration_s=m["duration_s"],
        sample_rate=sample_rate, poisoned_samples=m["poisoned"])


def write_sidecar(score: UtteranceScore, path: str, extra: dict = None) -> str:
    """
    Provenance JSON: what was measured, what was guessed, by which model.

    Optional -- the SMF is the record. This exists so that a reader months
    later can tell a null track from a missing one without re-running the
    model, which is the difference between a measurement and a rumour.
    """
    meta = {
        "text": score.text,
        "duration_s": round(score.duration_s, 4),
        "sample_rate": score.sample_rate,
        "ticks_per_beat": TICKS_PER_BEAT,
        "tick_is": "1 millisecond",
        "bpm": score.bpm,
        "tempo_written": bool(score.bpm),
        "model": score.model,
        "question_fingerprint": judged.question_fingerprint()
        if score.model else None,
        "poisoned_samples": score.poisoned_samples,
        "provenance": score.provenance(),
        "tracks": score.track_names(),
    }
    if extra:
        meta.update(extra)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return path


def render(wav_path: str, text: str, out_prefix: str, **kw) -> dict:
    """WAV in, .mid + provenance out. The end-to-end path."""
    samples, sr = wavio.read_wav(wav_path)
    extra = {k: kw.pop(k) for k in list(kw)
             if k in ("client", "cache", "model", "timeout")}
    sc = score_utterance(samples, text, sample_rate=sr, **extra)
    mid = f"{out_prefix}.mid"
    side = f"{out_prefix}.provenance.json"
    export_smf(sc, mid)
    write_sidecar(sc, side, extra_meta={"source": os.path.basename(wav_path)})
    return {"mid": mid, "provenance": side, "score": sc}