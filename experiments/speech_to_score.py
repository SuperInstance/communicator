#!/usr/bin/env python3
"""
speech_to_score.py -- everything said, turned into a score.

WHAT THIS IS.

Casey, in the words the brief relays: "Use a speech-to-text model to
translate everything said, with timestamps. SEPARATELY measure the tones:
where the notes are in the speech. That makes a melody out of everything
said, and it lives in MIDI. Then those little notes become where the
syllables are -- the MIDI notes ARE the syllable skeleton of the
utterance."

So the pipeline has three halves that meet at a table:

  1. TRANSCRIBE.  A local Whisper-class model with WORD timestamps.
  2. PITCH.       F0 over the same audio, segmented into NOTE EVENTS, each
                   with an onset, an offset and a median pitch -- a MIDI
                   note number. Exported as a real .mid.
  3. ALIGN.       Note events mapped onto the transcript's words, and the
                   words onto their SYLLABLES. The output is one row per
                   syllable: when it started, when it ended, which frames it
                   spans, and which MIDI note it carries.

ONE SHARED CLOCK, which is the addendum and the keystone.

"Transcript words, MIDI notes, your face/turn segments and the video
frames must all live on the session's single timeline (timestamps.json is
the spine)."

So there is exactly ONE time base in this file: the session's own. The
audio's native sample rate defines the seconds; `timestamps.json` maps
those seconds onto frame indices; and every note, word and syllable
carries BOTH. The face/turn segments come from
`experiments/free_session_analyze.py` and are read, not recomputed.

The trap this avoids: the old project's blocker was that the audio and the
video ran on independent clocks, so no alignment could be trusted. The
free recorder was born with one clock, and nothing here may introduce a
second one. Concretely:

  * The analysis audio is the session's own WAV, resampled to the
    repository's 16 kHz analysis rate by `src/wavio.py`, whose resampler
    is an exact 3:1 decimation -- 16 kHz sample i IS 48 kHz sample 3i, so
    the time base is preserved EXACTLY. No shift, no drift.
  * Whisper is handed that same array, and its word timestamps are in the
    input's time base, so they are session seconds.
  * The pitch tracker is given that same array, and its frame times are
    session seconds.
  * `timestamps.json` is the ONLY thing that converts seconds to frames.

THE PARTS ARE THE REPOSITORY'S, NOT NEW.

  * F0: `src/pitch.py`, a YIN tracker with a stated confidence and a
    refusal to guess. Nothing here re-implements it.
  * Voicing and silence: `src/energy.py`.
  * Hz -> MIDI note: `src/score.py:hz_to_note`.
  * The .mid: `src/smf.py`, a hand-rolled Standard MIDI File writer, with
    the repository's own clock convention -- **one tick is one
    millisecond** (ppq 1000, set_tempo 1e6 us/beat, `src/score.py`). So
    the MIDI file's times ARE the session's milliseconds, and a note can
    be looked up in the score and found in the audio at the same instant.

  The only new machinery is Whisper (a local model, no paid API) and the
  syllable table.

WHAT THIS DOES NOT CLAIM.

  * The melody of speech is not the melody of a song. A note event here is
    a median pitch over a voiced run; it is a reading of where the voice
    was, not a performance of it.
  * The syllable count comes from a pronouncing dictionary when the word is
    in it, and from a STATED heuristic when it is not. The heuristic is
    named in the report and every word it touched is listed.
  * A syllable with no note is reported as having no note. Silence is
    not a pitch.

USAGE:

    # a free-session folder: the full contract, with frame indices
    /home/proart/projects/speech-venv/bin/python experiments/speech_to_score.py \\
        /mnt/c/proart_capture/corpus/free_sessions/1792553040 --out /tmp/s2s

    # a bare audio file: the same pipeline, with no frame clock to report
    ... speech_to_score.py some.wav --out /tmp/s2s
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import energy, pitch, score, smf, wavio  # noqa: E402

# -- the clock ---------------------------------------------------------------

#: The repository's analysis rate. `src/pitch.py` is designed for it
#: (50 ms window, 10 ms hop, 800/160 samples) and `src/wavio.py` resamples
#: to it exactly.
ANALYSIS_SR = 16000

#: A note shorter than this is not a note, it is a wobble.
MIN_NOTE_S = 0.060
#: A note is split from its neighbour when the median pitch moves by at
#: least this many semitones.
SPLIT_SEMITONES = 1.0
#: The window the split decision is made over. The repository's own
#: inflection module compares over 100 ms at this frame rate, and a
#: 10 ms move is not a musical event.
SPLIT_WINDOW_FRAMES = 10
#: A note is merged into its neighbour when their medians are within this
#: many semitones AND the shorter is below MIN_NOTE_S.
MERGE_SEMITONES = 1.0


# ============================================================================
# the clock
# ============================================================================

@dataclass
class SessionClock:
    """Seconds <-> frame indices, through the session's own timestamps.

    `timestamps.json` holds `frame_offsets_ms`: the time of every captured
    frame, relative to the recording's origin. It is the spine. A time in
    seconds becomes a frame range by asking THIS array and nothing else.
    """
    path: Path
    frame_offsets_s: np.ndarray = None
    wall_clock_start: str = ""
    n_frames: int = 0

    @property
    def has_frames(self) -> bool:
        return self.frame_offsets_s is not None and len(self.frame_offsets_s) > 0

    def frames_for_span(self, t0: float, t1: float) -> tuple[int | None, int | None]:
        """The frames a time span covers: the frames whose timestamps fall
        inside it. None when there is no frame clock to ask.

        The span is HALF-OPEN, [t0, t1): a frame at exactly t1 is the first
        frame of the NEXT span, and counting it in both would make two
        adjacent syllables claim the same frame.
        """
        if not self.has_frames:
            return None, None
        lo = int(np.searchsorted(self.frame_offsets_s, t0, side="left"))
        hi = int(np.searchsorted(self.frame_offsets_s, t1, side="left"))
        if lo >= len(self.frame_offsets_s):
            # the span is past the end of the recording
            return None, None
        if hi > lo:
            return lo, hi - 1
        return lo, lo

    def frame_at(self, t: float) -> int | None:
        f0, _ = self.frames_for_span(t, t)
        return f0

    def describe(self) -> dict:
        d = {"source": str(self.path), "has_frames": self.has_frames,
             "n_frames": self.n_frames, "wall_clock_start": self.wall_clock_start}
        if self.has_frames:
            d["first_frame_s"] = float(self.frame_offsets_s[0])
            d["last_frame_s"] = float(self.frame_offsets_s[-1])
            d["median_fps"] = float(1.0 / np.median(np.diff(self.frame_offsets_s)))
        return d


def load_clock(session_dir: Path | None) -> SessionClock:
    """The clock, from the session's own timestamps.json."""
    if session_dir is None:
        return SessionClock(path=Path("(none)"))
    session_dir = Path(session_dir)
    clock = SessionClock(path=session_dir / "timestamps.json")
    ts_path = session_dir / "timestamps.json"
    if ts_path.is_file():
        try:
            data = json.loads(ts_path.read_text())
        except Exception:  # noqa: BLE001
            data = {}
        offs = data.get("frame_offsets_ms")
        if isinstance(offs, list) and len(offs) > 1:
            clock.frame_offsets_s = np.asarray(offs, dtype=np.float64) / 1000.0
            clock.n_frames = len(clock.frame_offsets_s)
        clock.wall_clock_start = str(data.get("wall_clock_start", ""))
    meta_path = session_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            clock.wall_clock_start = str(meta.get("wall_clock_start",
                                                   clock.wall_clock_start))
        except Exception:  # noqa: BLE001
            pass
    return clock


# ============================================================================
# 1. TRANSCRIBE
# ============================================================================

@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float


def transcribe(audio: np.ndarray, model_name: str = "small.en",
               device: str = "auto", compute_type: str = "auto",
               beam_size: int = 5, language: str | None = "en") -> tuple[list[Word], dict]:
    """Transcribe with a LOCAL Whisper-class model, with word timestamps.

    The model is faster-whisper: local weights, no API, no subscription.
    The device is chosen here and REPORTED, because "it ran on the GPU" is
    a fact a reader should not have to take on trust.
    """
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    t0 = time.time()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    load_s = time.time() - t0

    t0 = time.time()
    segments, info = model.transcribe(audio, word_timestamps=True,
                                      beam_size=beam_size, language=language,
                                      vad_filter=True)
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            words.append(Word(text=w.word, start=float(w.start),
                              end=float(w.end), probability=float(w.probability)))
    meta = {
        "engine": "faster-whisper",
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "model_load_s": round(load_s, 2),
        "transcribe_s": round(time.time() - t0, 2),
        "language": getattr(info, "language", language),
        "audio_duration_s": float(getattr(info, "duration", len(audio) / ANALYSIS_SR)),
        "n_words": len(words),
        "vad_filter": True,
    }
    return words, meta


# ============================================================================
# 2. PITCH -> NOTE EVENTS
# ============================================================================

@dataclass
class NoteEvent:
    index: int
    start: float
    end: float
    hz: float
    midi: int
    confidence: float
    n_frames: int

    def as_dict(self) -> dict:
        return {"index": self.index, "start_s": round(self.start, 4),
                "end_s": round(self.end, 4), "duration_s": round(self.end - self.start, 4),
                "hz": round(self.hz, 3), "midi": self.midi,
                "midi_name": midi_name(self.midi),
                "confidence": round(self.confidence, 4),
                "n_pitch_frames": self.n_frames}


def midi_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"


def _semitones(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return abs(12.0 * math.log2(a / b))


def pitch_frames(audio: np.ndarray) -> list:
    """The repository's own YIN tracker over the analysis audio.

    `src/pitch.py:smooth_octaves` is the repo's own post-pass against
    YIN's classic octave error, measured at 3.0% of adjacent voiced pairs
    on real speech. It is applied because it is the repository's answer
    to a defect this data will otherwise have.
    """
    tracker = pitch.PitchTracker(ANALYSIS_SR, frame_len=800, hop=160)
    frames = tracker.track(list(audio))
    return pitch.smooth_octaves(frames)


def _step_index(hz: np.ndarray, lo: int, hi: int, old: float,
                new: float) -> int | None:
    """The frame inside [lo, hi) where the contour turns from old to new.

    The window is scanned for the first frame from which the REST of the
    window sits closer to the new level than to the old one. That frame is
    the step; everything before it in the window still belongs to the old
    note. If the contour never settles, the LAST frame on the new side is
    used, which is the conservative choice -- the note ends late rather
    than early.

    Returns None when the window says nothing, which leaves the contour
    unsplit rather than splitting it at a place the pitch does not support.
    """
    seg = hz[lo:hi]
    if len(seg) == 0:
        return None
    closer_new = np.abs(seg - new) < np.abs(seg - old)
    for k in range(len(seg)):
        if bool(closer_new[k:].all()):
            return lo + k
    idx = np.flatnonzero(closer_new)
    if len(idx) == 0:
        return None
    return lo + int(idx[-1])


def note_events(frames: list, min_note_s: float = MIN_NOTE_S,
                split_semitones: float = SPLIT_SEMITONES,
                window: int = SPLIT_WINDOW_FRAMES) -> list[NoteEvent]:
    """Segment the pitch contour into NOTE EVENTS.

    Two stages, and the order matters:

    1. VOICED RUNS. A frame with no pitch is an absence, and an absence
       ends a note. The tracker refuses to guess, so a run boundary is
       always a real boundary.
    2. SPLITS WITHIN A RUN. The contour is walked and a note is closed
       when the median pitch of the recent window has moved away from the
       note's own median by at least `split_semitones`, and the note being
       closed is already long enough to be a note.

    The note's own span is the window geometry, not the frame centres: the
    onset is the first frame's window START and the offset is the last
    frame's window END, and a split between two frames lands on the
    midpoint between their centres. So consecutive notes tile their run
    with no gap and no overlap -- the 50 ms analysis window is the only
    blur, and it is stated rather than hidden.
    """
    half = (800 / 2.0) / ANALYSIS_SR
    notes: list[NoteEvent] = []
    i = 0
    n = len(frames)
    while i < n:
        f = frames[i]
        if f.hz is None:
            i += 1
            continue
        j = i
        while j + 1 < n and frames[j + 1].hz is not None:
            j += 1
        # a voiced run: frames i..j inclusive
        run = frames[i:j + 1]
        hz = np.array([fr.hz for fr in run if fr.hz is not None], dtype=np.float64)
        conf = np.array([fr.confidence for fr in run], dtype=np.float64)
        times = np.array([fr.time for fr in run], dtype=np.float64)
        # The split is decided on the MEDIAN OF THE NOTE SO FAR vs the median
        # of the RECENT WINDOW, which is what makes a held note survive and a
        # moving contour break. Two pointers are needed and they are NOT the
        # same pointer:
        #
        #   note_start -- where the current note began. The note's AGE is
        #                 measured from here, so a note is only closed once it
        #                 is long enough to be a note.
        #   scan      -- the frame we are currently looking at. The RECENT
        #                 WINDOW is the `window` frames before it.
        #
        # The cut is placed at the STEP, not at the window edge: inside the
        # window the contour is walked back to the frame where the pitch is
        # closer to the new level than to the old one. That frame is the
        # boundary, and the note ends there.
        cuts = []
        note_start = 0
        scan = 0
        while scan < len(run):
            win_start = max(note_start, scan - window)
            if scan - win_start < 3:
                scan += 1
                continue
            recent = float(np.median(hz[win_start:scan]))
            established = hz[note_start:win_start]
            if len(established) >= 2:
                old = float(np.median(established))
            else:
                old = float(np.median(hz[note_start:scan]))
            if (_semitones(recent, old) >= split_semitones
                    and (times[scan - 1] - times[note_start]) >= min_note_s):
                cut = _step_index(hz, win_start, scan, old, recent)
                if cut is not None and note_start < cut < scan:
                    cuts.append(cut)
                    note_start = cut
                    scan = cut
                    continue
            scan += 1
        # build the notes
        bounds = [0] + cuts + [len(run)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b <= a:
                continue
            seg_hz = hz[a:b]
            if len(seg_hz) == 0:
                continue
            onset = times[a] - half
            offset = times[b - 1] + half
            if a > 0:
                onset = (times[a - 1] + times[a]) / 2.0
            if b < len(run):
                offset = (times[b - 1] + times[b]) / 2.0
            median_hz = float(np.median(seg_hz))
            notes.append(NoteEvent(
                index=0,
                start=max(0.0, onset),
                end=max(onset + 1e-3, offset),
                hz=median_hz,
                midi=score.hz_to_note(median_hz),
                confidence=float(np.median(conf[a:b])),
                n_frames=b - a))
        i = j + 1
    # drop the notes that are too short to be notes, and merge the ones
    # that are really the same note
    notes = [nt for nt in notes if (nt.end - nt.start) >= min_note_s]
    merged: list[NoteEvent] = []
    for nt in notes:
        if merged and _semitones(merged[-1].hz, nt.hz) <= MERGE_SEMITONES \
                and (nt.end - nt.start) < min_note_s * 2:
            prev = merged[-1]
            merged[-1] = NoteEvent(prev.index, prev.start, nt.end,
                                   (prev.hz * prev.n_frames + nt.hz * nt.n_frames)
                                   / (prev.n_frames + nt.n_frames),
                                   prev.midi, min(prev.confidence, nt.confidence),
                                   prev.n_frames + nt.n_frames)
            continue
        merged.append(nt)
    for k, nt in enumerate(merged):
        nt.index = k
        nt.midi = score.hz_to_note(nt.hz)
    return merged


# ============================================================================
# 3. THE .mid
# ============================================================================

def write_midi(notes: list[NoteEvent], words: list[Word], path: Path,
               track_name: str = "speech") -> dict:
    """The score: a real Standard MIDI File, on the repository's clock.

    `src/score.py` chose ONE TICK IS ONE MILLISECOND (ppq 1000, set_tempo
    1e6 us/beat) precisely so that a note time in the file IS a time in
    the audio. That choice is kept, so the .mid and the alignment JSON are
    the same clock to the millisecond.

    ONE track, carrying ONE property: the notes, at constant velocity, so
    the pitch is not welded to the loudness (the repository's own rule --
    "one property per track"). The words are NOT in the .mid: they are the
    transcript, which is a separate deliverable, and `src/smf.py` has no
    text-meta event to carry them honestly. A file that pretends to be
    more than it is would be worse than one that is exactly the melody.
    """
    n_words = len(words)
    label = f"{track_name}: {len(notes)} notes, {n_words} words"
    events = [smf.MidiEvent(0, "track_name", {"name": label[:120]}),
              smf.MidiEvent(0, "set_tempo", {"tempo": score.TEMPO_US})]
    for nt in notes:
        on = score.ms_to_ticks(nt.start)
        off = score.ms_to_ticks(nt.end)
        if off <= on:
            off = on + 1
        events.append(smf.MidiEvent(on, "note_on",
                                    {"note": nt.midi, "velocity": 64}))
        events.append(smf.MidiEvent(off, "note_off",
                                    {"note": nt.midi, "velocity": 0}))

    data = smf.write_smf([smf.MidiTrackData(name="pitch", events=events)],
                         ticks_per_beat=score.TICKS_PER_BEAT)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data),
            "ticks_per_beat": score.TICKS_PER_BEAT,
            "ms_per_tick": score.MS_PER_TICK,
            "n_notes": len(notes), "n_words": n_words,
            "n_tracks": 1, "track_name": label[:120]}


# ============================================================================
# 4. SYLLABLES
# ============================================================================

#: A CMUdict phoneme is a vowel when it ends in a stress digit. That is
#: the dictionary's own convention and it is how syllable counts are
#: derived from it.
def _is_vowel(phoneme: str) -> bool:
    return bool(phoneme) and phoneme[-1].isdigit()


@dataclass
class SyllablePlan:
    """How a word divides into syllables, and how it says so."""
    word: str
    n_syllables: int
    source: str            # "cmudict" | "heuristic"
    weights: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"word": self.word, "n_syllables": self.n_syllables,
                "source": self.source,
                "weights": [round(w, 4) for w in self.weights]}


class Syllabifier:
    """Syllable counts from a pronouncing dictionary, with a stated
    heuristic for the words the dictionary does not have.

    The heuristic, stated once and used only where the dictionary is
    silent: count the VOWEL LETTER GROUPS in the word (a, e, i, o, u, y
    when it is not the first letter), and treat a silent final `e` as not a
    vowel. That is the classic English rule and it is wrong sometimes; the
    report lists every word it touched, so the reader can see how much of
    the alignment rests on it.
    """

    def __init__(self):
        self.dict = {}
        self.source = "none"
        self.ambiguous: set = set()
        try:
            import cmudict
            self.dict = cmudict.dict()
            self.source = "cmudict"
        except Exception:  # noqa: BLE001
            self.dict = {}

    def plan(self, word: str) -> SyllablePlan:
        # The transcript's words carry a leading space (" delighted"), and a
        # dictionary lookup on " delighted" is a miss. Whitespace first,
        # then the punctuation that rides on the end of a word.
        w = word.strip().lower().strip(".,!?;:'\"()[]")
        if not w:
            return SyllablePlan(word=word, n_syllables=0, source="empty")
        if w in self.dict:
            # cmudict maps a word to a LIST OF PRONUNCIATIONS, and a word can
            # have more than one. The syllable COUNT is usually the same for
            # all of them, but not always, so the FIRST is taken and the
            # ambiguity is counted and reported rather than hidden.
            prons = self.dict[w]
            phones = prons[0] if prons else []
            groups: list[list[str]] = []
            current: list[str] = []
            for p in phones:
                current.append(p)
                if _is_vowel(p):
                    groups.append(current)
                    current = []
            if current and groups:
                groups[-1].extend(current)
            if groups:
                if len(prons) > 1:
                    self.ambiguous.add(w)
                weights = [float(len(g)) for g in groups]
                return SyllablePlan(word=word, n_syllables=len(groups),
                                    source="cmudict", weights=weights)
        return SyllablePlan(word=word, n_syllables=self._heuristic(w),
                            source="heuristic",
                            weights=[1.0] * max(1, self._heuristic(w)))

    @staticmethod
    def _heuristic(w: str) -> int:
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for i, ch in enumerate(w):
            is_v = ch in vowels
            if ch == "y" and i == 0:
                is_v = False
            if is_v and not prev_vowel:
                count += 1
            prev_vowel = is_v
        if w.endswith("e") and count > 1:
            count -= 1
        return max(1, count)


def split_word_into_syllables(word: str, plan: SyllablePlan,
                               start: float, end: float) -> list[tuple[float, float]]:
    """Split a word's span across its syllables.

    The split is by the PHONEME weight of each syllable, not evenly: a
    three-phoneme syllable takes three times as long as a one-phoneme
    one. That is a heuristic and it is the best available without a
    duration model -- it is stated here, and in the report.
    """
    n = max(1, plan.n_syllables)
    weights = plan.weights or [1.0] * n
    weights = weights[:n] if len(weights) >= n else weights + [1.0] * (n - len(weights))
    total = sum(weights) or float(n)
    span = max(0.0, end - start)
    out = []
    t = start
    for w in weights:
        d = span * (w / total)
        out.append((t, t + d))
        t += d
    return out


# ============================================================================
# 5. ALIGN
# ============================================================================

def drill_analysis(words: list, notes: list, energy_frames: list,
                   n_gram: int = 3) -> dict:
    """The drill-shaped analysis: repeated deliveries of the SAME words.

    The potato drill holds the words constant and varies the delivery. A
    transcript contains that shape by accident whenever a speaker repeats
    a phrase, and the oration is full of it -- "noted for knowledge ...
    noted for progress ... noted for strength". Those repetitions are the
    nearest thing to drill data that exists today, so they are measured
    the way the drill will measure it.

    For every instance of a repeated phrase: when it was said, the note
    sequence that carried it, how long it took, and how loud it was. Then
    the DELTAS between the instances, because the deltas are the tone map
    -- the words are identical, so any difference in the melody is the
    delivery and nothing else.
    """
    groups: dict = {}
    for i in range(len(words) - n_gram + 1):
        chunk = words[i:i + n_gram]
        key = " ".join(w.text.strip().lower() for w in chunk)
        if not any(c.isalpha() for c in key):
            continue
        groups.setdefault(key, []).append(chunk)

    repeated = {k: v for k, v in groups.items() if len(v) > 1}

    def rms_between(t0: float, t1: float) -> tuple[float, float]:
        vals = [f.rms for f in energy_frames
                if f.time >= t0 and f.time <= t1]
        if not vals:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.max(vals))

    out = []
    for phrase, instances in sorted(repeated.items()):
        rows = []
        for chunk in instances:
            s0 = chunk[0].start
            s1 = chunk[-1].end
            carried = [nt for nt in notes
                       if min(nt.end, s1) - max(nt.start, s0) > 0]
            mean_rms, peak_rms = rms_between(s0, s1)
            hz = [nt.hz for nt in carried]
            rows.append({
                "start_s": round(s0, 4),
                "end_s": round(s1, 4),
                "duration_s": round(s1 - s0, 4),
                "n_notes": len(carried),
                "midi": [nt.midi for nt in carried],
                "hz_median": round(float(np.median(hz)), 2) if hz else None,
                "note_durations_s": [round(nt.end - nt.start, 4) for nt in carried],
                "mean_rms": round(mean_rms, 6),
                "peak_rms": round(peak_rms, 6),
                "n_syllables_in_span": sum(1 for w in words
                                           if w.start >= s0 - 1e-6 and w.end <= s1 + 1e-6),
            })
        # the deltas: every instance against the first one
        base = rows[0]
        deltas = []
        for r in rows[1:]:
            dh = None
            if base["hz_median"] and r["hz_median"]:
                dh = round(12.0 * math.log2(r["hz_median"] / base["hz_median"]), 3)
            deltas.append({
                "against": r["start_s"],
                "pitch_delta_semitones": dh,
                "duration_delta_s": round(r["duration_s"] - base["duration_s"], 4),
                "note_count_delta": r["n_notes"] - base["n_notes"],
                "mean_rms_ratio": round(r["mean_rms"] / base["mean_rms"], 4)
                                  if base["mean_rms"] else None,
            })
        out.append({"phrase": phrase, "n_instances": len(rows),
                    "instances": rows, "deltas": deltas})

    return {
        "n_gram": n_gram,
        "n_repeated_phrases": len(out),
        "note": ("the words are identical across the instances, so every "
                 "delta in the melody, the timing and the energy is the "
                 "delivery and not the wording -- this is the drill's table, "
                 "measured on natural repetitions"),
        "phrases": out,
    }


def join_segments(rows: list, segments: list) -> dict:
    """Attach the analyzer's face and speech state to each syllable row.

    This is a LOOKUP, not a second analysis. The analyzer already measured
    the face state and the speech state on the session's own timeline, and
    every alignment row already carries frame indices on that same
    timeline -- the join key is the frame index, so there is still exactly
    one clock. Nothing here re-detects a face or re-thresholds an RMS.

    When a row has no frames (an audio file with no session beside it) the
    join falls back to time overlap, which is the same comparison in the
    same unit.
    """
    joined = 0
    for r in rows:
        best = None
        best_ov = 0.0
        # The analyzer's segment spans are HALF-OPEN (frame_end exclusive --
        # the next segment starts there), while an alignment row's frame_end
        # is INCLUSIVE (it is the last frame the syllable touches). The
        # row's end is made exclusive here so the two are comparable.
        r_end = None
        if r.get("frame_start") is not None and r.get("frame_end") is not None:
            r_end = r["frame_end"] + 1
        for si, s in enumerate(segments):
            if r_end is not None and s.get("frame_start") is not None \
                    and s.get("frame_end") is not None:
                ov = min(r_end, s["frame_end"]) - max(r["frame_start"], s["frame_start"])
            else:
                ov = min(r["end_s"], s.get("end_s", 0.0)) - max(r["start_s"], s.get("start_s", 0.0))
            if ov > best_ov:
                best_ov = ov
                best = (si, s)
        if best is not None:
            si, s = best
            r["segment_index"] = si
            r["segment_start_s"] = round(float(s.get("start_s", 0.0)), 4)
            r["segment_end_s"] = round(float(s.get("end_s", 0.0)), 4)
            r["face_state"] = s.get("face_state")
            r["speech_state"] = s.get("speech_state")
            r["segment_overlap_s"] = round(best_ov, 4)
            joined += 1
    return {"segments": len(segments), "rows_joined": joined,
            "rows_unjoined": len(rows) - joined,
            "source": "experiments/free_session_analyze.py segment table "
                      "(face state and speech state, joined on the session's "
                      "own frame indices -- no second clock, no re-measure)"}


def align(notes: list[NoteEvent], words: list[Word],
          clock: SessionClock, syllabifier: Syllabifier) -> dict:
    """The alignment table: one row per syllable.

    For each syllable: when it started, when it ended, WHICH FRAMES it
    spans, and which MIDI note it carries. The note is the one with the
    largest time overlap; a syllable with no overlapping note has no note,
    and says so.
    """
    rows = []
    n_ambiguous = 0
    for wi, w in enumerate(words):
        plan = syllabifier.plan(w.text)
        spans = split_word_into_syllables(w.text, plan, w.start, w.end)
        for si, (s0, s1) in enumerate(spans):
            # the notes that overlap this syllable
            best = None
            best_ov = 0.0
            for nt in notes:
                ov = min(nt.end, s1) - max(nt.start, s0)
                if ov > best_ov:
                    best_ov = ov
                    best = nt
            f0, f1 = clock.frames_for_span(s0, s1)
            row = {
                "syllable_index": len(rows),
                "word_index": wi,
                "word": w.text.strip(),
                "syllable_in_word": si,
                "n_syllables_in_word": plan.n_syllables,
                "syllable_source": plan.source,
                "start_s": round(s0, 4),
                "end_s": round(s1, 4),
                "duration_s": round(s1 - s0, 4),
                "frame_start": f0,
                "frame_end": f1,
                "midi": best.midi if best is not None else None,
                "midi_name": midi_name(best.midi) if best is not None else None,
                "note_index": best.index if best is not None else None,
                "note_overlap_s": round(best_ov, 4) if best is not None else 0.0,
                "word_probability": round(w.probability, 4),
            }
            if best is None:
                n_ambiguous += 1
            rows.append(row)
    return {"rows": rows, "n_syllables": len(rows), "n_without_note": n_ambiguous}


# ============================================================================
# the driver
# ============================================================================

def load_audio(path: Path) -> tuple[np.ndarray, dict]:
    """The session's own audio, at the repository's analysis rate.

    `src/wavio.py` decodes mono 16-bit PCM and resamples to 16 kHz with
    an EXACT 3:1 decimation from 48 kHz -- sample i at 16 kHz IS sample 3i
    at 48 kHz -- so the time base is the session's, to the sample.
    """
    audio = wavio.read_wav(str(path))
    arr = np.asarray(audio.samples, dtype=np.float32)
    meta = {"path": str(path), "source_sample_rate": audio.source_sample_rate,
            "analysis_sample_rate": audio.sample_rate,
            "n_samples": len(arr), "duration_s": len(arr) / float(audio.sample_rate),
            "poisoned_samples": audio.poisoned_samples,
            "resample_note": ("src/wavio.py resamples with linear interpolation "
                              "and does NOT anti-alias; the time base is exact "
                              "but content near the old Nyquist folds")}
    return arr, meta


def analyse(audio_path: Path, session_dir: Path | None, out_dir: Path,
            model: str = "small.en", device: str = "auto",
            compute_type: str = "auto", min_note_s: float = MIN_NOTE_S,
            split_semitones: float = SPLIT_SEMITONES,
            segments_path: Path | None = None,
            drill_n: int = 0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clock = load_clock(session_dir)
    arr, audio_meta = load_audio(audio_path)

    report: dict = {
        "session": str(session_dir) if session_dir else None,
        "audio": audio_meta,
        "clock": clock.describe(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # 1. the words
    t0 = time.time()
    words, wmeta = transcribe(arr, model_name=model, device=device,
                              compute_type=compute_type)
    report["transcribe"] = wmeta
    report["transcribe"]["total_s"] = round(time.time() - t0, 2)
    if not words:
        # honest, and specific: the audio carries no speech
        energy_frames, _ = energy.rms_series(list(arr), 800, 160, ANALYSIS_SR)
        rms = np.array([f.rms for f in energy_frames]) if energy_frames else np.zeros(1)
        report["no_speech"] = {
            "reason": "the transcript is empty",
            "audio_peak_rms": float(rms.max()) if rms.size else 0.0,
            "audio_mean_rms": float(rms.mean()) if rms.size else 0.0,
            "frames_over_energy_floor": int((rms > energy.ABS_FLOOR).sum()),
            "n_frames": int(len(rms)),
            "verdict": ("the audio is SILENT -- there is nothing to transcribe, "
                        "nothing to pitch and no syllables to align"),
        }
        report["notes"] = []
        report["alignment"] = {"rows": [], "n_syllables": 0, "n_without_note": 0}
        report["midi"] = write_midi([], [], out_dir / "session.mid")
        (out_dir / "transcript.json").write_text(json.dumps(
            {"words": [], "meta": wmeta}, indent=2))
        (out_dir / "alignment.json").write_text(json.dumps(report, indent=2))
        (out_dir / "transcript.txt").write_text(
            "(no speech in this audio -- see alignment.json: no_speech)\n")
        return report

    # 2. the pitch, and the notes
    t0 = time.time()
    frames = pitch_frames(arr)
    notes = note_events(frames, min_note_s=min_note_s,
                        split_semitones=split_semitones)
    report["pitch"] = {
        "tracker": "src/pitch.py (YIN, 50 ms window, 10 ms hop, 70-400 Hz)",
        "octave_smoothing": "src/inflection.py:smooth_octaves",
        "n_frames": len(frames),
        "n_voiced": sum(1 for f in frames if f.hz is not None),
        "voiced_fraction": round(sum(1 for f in frames if f.hz is not None)
                                 / max(1, len(frames)), 4),
        "min_note_s": min_note_s,
        "split_semitones": split_semitones,
        "n_notes": len(notes),
        "elapsed_s": round(time.time() - t0, 2),
    }
    report["notes"] = [nt.as_dict() for nt in notes]

    # 3. the score
    report["midi"] = write_midi(notes, words, out_dir / "session.mid")

    # 4. the alignment
    syllabifier = Syllabifier()
    al = align(notes, words, clock, syllabifier)
    report["alignment"] = al

    # 4b. the analyzer's face/speech segments, joined on the frame index
    if segments_path is not None:
        seg_path = Path(segments_path)
        if seg_path.is_file():
            seg_doc = json.loads(seg_path.read_text())
            segs = seg_doc.get("segments", [])
            report["segment_join"] = join_segments(al["rows"], segs)
            report["segment_join"]["path"] = str(seg_path)
        else:
            report["segment_join"] = {"error": f"no segment table at {seg_path}"}
    report["syllables"] = {
        "source": syllabifier.source,
        "n_words": len(words),
        "n_syllables": al["n_syllables"],
        "words_from_dictionary": sum(1 for r in al["rows"]
                                     if r["syllable_source"] == "cmudict"),
        "words_from_heuristic": sum(1 for r in al["rows"]
                                    if r["syllable_source"] == "heuristic"),
        "heuristic_words": sorted({r["word"] for r in al["rows"]
                                   if r["syllable_source"] == "heuristic"}),
        "ambiguous_pronunciations": sorted(syllabifier.ambiguous),
    }

    # 4c. the drill-shaped analysis: the same words delivered again
    if drill_n > 0:
        ef, _ = energy.rms_series(list(arr), 800, 160, ANALYSIS_SR)
        report["drill"] = drill_analysis(words, notes, ef, n_gram=drill_n)

    # 5. the outputs
    (out_dir / "transcript.json").write_text(json.dumps(
        {"words": [{"text": w.text, "start_s": round(w.start, 4),
                    "end_s": round(w.end, 4),
                    "probability": round(w.probability, 4),
                    "frame_start": clock.frames_for_span(w.start, w.end)[0],
                    "frame_end": clock.frames_for_span(w.start, w.end)[1]}
                   for w in words], "meta": wmeta}, indent=2))
    with (out_dir / "transcript.txt").open("w", encoding="utf-8") as f:
        for w in words:
            f.write(f"[{w.start:8.2f} - {w.end:8.2f}] {w.text.strip()}\n")
    (out_dir / "alignment.json").write_text(json.dumps(report, indent=2))
    return report


def format_alignment(report: dict, limit: int = 60) -> str:
    """The printed alignment table."""
    lines = []
    lines.append("ALIGNMENT -- one row per syllable")
    lines.append("-" * 96)
    lines.append(f"{'#':>4} {'start':>8} {'end':>7} {'dur':>6} {'frames':>13} "
                 f"{'midi':>5} {'name':>5}  word")
    rows = report.get("alignment", {}).get("rows", [])
    for r in rows[:limit]:
        fr = (f"{r['frame_start']}..{r['frame_end']}"
              if r["frame_start"] is not None else "-")
        midi = f"{r['midi']}" if r["midi"] is not None else "-"
        mname = r["midi_name"] or "-"
        lines.append(f"{r['syllable_index']:>4} {r['start_s']:8.2f} "
                     f"{r['end_s']:7.2f} {r['duration_s']:6.2f} {fr:>13} "
                     f"{midi:>5} {mname:>5}  {r['word']}")
    if len(rows) > limit:
        lines.append(f"  ... {len(rows) - limit} more rows in alignment.json")
    return "\n".join(lines)


def selftest() -> int:
    """The checks that do not need a model: the clock, the .mid round trip,
    the syllable split and the note segmentation."""
    import tempfile

    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  [{'ok' if got == want else 'FAIL'}] {name}: {got!r}")

    print("SELFTEST -- the clock, the score and the alignment")
    print("-" * 60)

    # 1. THE CLOCK. A session whose frames are at a known rate.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "sess"
        d.mkdir()
        offs = [round(i * 1000.0 / 60.0, 3) for i in range(120)]
        (d / "timestamps.json").write_text(json.dumps({
            "wall_clock_start": "2026-01-01T00:00:00Z", "frame_offsets_ms": offs}))
        (d / "meta.json").write_text(json.dumps({"n_frames": 120}))
        clock = load_clock(d)
        check("the clock has frames", clock.has_frames, True)
        check("the frame count", clock.n_frames, 120)
        check("the median fps", round(clock.describe()["median_fps"], 2), 60.0)
        # 1.0 s is frame 60 at 60 fps
        f0, f1 = clock.frames_for_span(1.0, 1.0)
        check("the frame at 1.0 s", f0, 60)
        f0, f1 = clock.frames_for_span(1.0, 1.5)
        check("the frames in 1.0-1.5 s", (f0, f1), (60, 89))
        f0, f1 = clock.frames_for_span(0.0, 0.0)
        check("the frame at 0.0 s", f0, 0)
        # a span past the end of the recording
        f0, f1 = clock.frames_for_span(100.0, 101.0)
        check("a span past the end has no frames", f0, None)
        # and the audio's seconds are the same seconds
        check("the clock and the audio share a unit",
              round(clock.frame_offsets_s[60], 6), 1.0)

    # 2. THE .mid ROUND TRIP. A writer without a reader is a slogan.
    with tempfile.TemporaryDirectory() as td:
        notes = [NoteEvent(0, 0.0, 0.5, 130.81, 57, 0.9, 50),
                 NoteEvent(1, 0.5, 1.25, 146.83, 59, 0.8, 75)]
        words = [Word("hello", 0.0, 0.5, 0.99)]
        out = Path(td) / "s.mid"
        info = write_midi(notes, words, out)
        check("the file was written", out.is_file(), True)
        ppq, tracks = smf.parse_smf(out.read_bytes())
        check("the file parses back", ppq, score.TICKS_PER_BEAT)
        check("with one track", len(tracks), 1)
        kinds = [(e.tick, e.kind, e.data) for e in tracks[0].events]
        ons = [k for k in kinds if k[1] == "note_on"]
        offs_ = [k for k in kinds if k[1] == "note_off"]
        check("both notes are there", len(ons), 2)
        check("and their note-offs", len(offs_), 2)
        check("the first note is on the millisecond",
              ons[0][0], score.ms_to_ticks(0.0))
        check("the first note's pitch survives",
              ons[0][2]["note"], notes[0].midi)
        check("the second note's onset survives",
              ons[1][0], score.ms_to_ticks(0.5))
        check("the second note's offset survives",
              offs_[1][0], score.ms_to_ticks(1.25))
        check("one tick is one millisecond", info["ms_per_tick"], 1.0)

    # 3. THE SYLLABLE SPLIT.
    syl = Syllabifier()
    if syl.source != "cmudict":
        # The dictionary is not installed in this environment. That is a
        # fact about the machine, not a failure of the tool, and the tool
        # handles it by falling back to the stated heuristic -- which is
        # checked below. The dictionary checks are skipped and say so.
        print("  [skip] the dictionary checks -- cmudict is not installed "
              f"here (the tool is using {syl.source!r})")
        plan = syl.plan("delighted")
        check("without the dictionary the heuristic is used",
              plan.source, "heuristic")
    else:
        plan = syl.plan("delighted")
        check("the dictionary found the word", plan.source, "cmudict")
    check("and counted its syllables", plan.n_syllables, 3)
    spans = split_word_into_syllables("delighted", plan, 1.0, 2.0)
    check("the spans tile the word", len(spans), 3)
    check("the spans are contiguous", round(spans[0][1], 9), round(spans[1][0], 9))
    check("and they end where the word ends", round(spans[-1][1], 9), 2.0)
    check("the weights sum to the span", round(spans[-1][1] - spans[0][0], 9), 1.0)
    hplan = syl.plan("qxzzy")
    check("an unknown word uses the heuristic", hplan.source, "heuristic")

    # 4. THE NOTE SEGMENTATION: a contour with a step is two notes.
    class F:
        def __init__(self, t, hz):
            self.time = t
            self.hz = hz
            self.confidence = 0.9
    contour = []
    for i in range(60):
        contour.append(F(i * 0.01, 120.0))       # 0.6 s at 120 Hz
    for i in range(60):
        contour.append(F(0.6 + i * 0.01, 180.0))  # 0.6 s at 180 Hz
    contour.append(F(1.2, None))
    evs = note_events(contour, min_note_s=0.05, split_semitones=1.0, window=5)
    check("a step in the contour is two notes", len(evs), 2)
    # 120 Hz is B2 (MIDI 47) and 180 Hz is D#3 (MIDI 54) -- the code's
    # hz_to_note is right and the expectation was wrong, once.
    check("the first note is the low one", evs[0].midi, 47)
    check("the second is the high one", evs[1].midi, 54)
    check("the notes tile the run", round(evs[0].end, 3), round(evs[1].start, 3))
    check("and neither is longer than the run", evs[1].end <= 1.25, True)
    # the cut lands AT THE STEP, not at the window edge: the step is at
    # 0.60 s and the window is 0.05 s wide, so the cut may be late by at
    # most one window.
    check("the cut is at the step, not the window edge",
          abs(evs[0].end - 0.60) <= 0.05, True)

    # 5. THE ALIGNMENT joins the right note.
    # The word must CROSS the boundary between the two notes, or both of its
    # syllables land on the first note and the second note is never tested.
    # "alpha" is AE L F AH -- two syllables weighted 1:3 -- so over 1.0-1.7 s
    # the first syllable is [1.0, 1.175] (inside note 0) and the second is
    # [1.175, 1.7], which overlaps note 1 for 0.3 s against note 0's 0.225 s.
    notes = [NoteEvent(0, 1.0, 1.4, 130.0, 57, 0.9, 40),
             NoteEvent(1, 1.4, 2.0, 150.0, 59, 0.9, 60)]
    words = [Word(" alpha ", 1.0, 1.7, 0.95)]
    clock = SessionClock(path=Path("(none)"))
    al = align(notes, words, clock, syl)
    check("one word gives its syllables", al["n_syllables"], 2)
    check("the first syllable takes the first note",
          al["rows"][0]["midi"], 57)
    check("the second takes the second",
          al["rows"][1]["midi"], 59)
    if syl.source == "cmudict":
        check("the word's leading space did not break the dictionary",
              al["rows"][0]["syllable_source"], "cmudict")
    else:
        print("  [skip] the dictionary check -- cmudict is not installed here")

    # 6. THE SEGMENT JOIN -- the composition with the analyzer, on the
    # frame index. The segment spans are half-open, the row's end is
    # inclusive; the join has to know that or it loses a frame.
    segs = [{"frame_start": 0, "frame_end": 50, "start_s": 0.0, "end_s": 1.0,
             "face_state": "facing", "speech_state": "speaking"},
            {"frame_start": 50, "frame_end": 100, "start_s": 1.0, "end_s": 2.0,
             "face_state": "turned", "speech_state": "silent"}]
    rows = [{"frame_start": 10, "frame_end": 20, "start_s": 0.2, "end_s": 0.4},
            {"frame_start": 60, "frame_end": 70, "start_s": 1.2, "end_s": 1.4},
            {"frame_start": 98, "frame_end": 99, "start_s": 1.9, "end_s": 2.0}]
    res = join_segments(rows, segs)
    check("every row joined a segment", res["rows_joined"], 3)
    check("the first row found the facing segment", rows[0]["face_state"], "facing")
    check("the second found the turned one", rows[1]["face_state"], "turned")
    check("and it carries the analyzer's speech state too",
          rows[1]["speech_state"], "silent")
    check("a row on the boundary joins the later segment",
          rows[2]["segment_index"], 1)

    print("-" * 60)
    if failures:
        print(f"SELFTEST FAILED: {len(failures)} check(s)")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SELFTEST PASSED")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", nargs="?", default=None,
                    help="a free-session folder, or an audio file")
    ap.add_argument("--out", default=None, help="where to write the outputs")
    ap.add_argument("--model", default="small.en",
                    help="the local Whisper model (default: small.en)")
    ap.add_argument("--device", default="auto", help="cuda | cpu | auto")
    ap.add_argument("--compute-type", default="auto",
                    help="float16 | int8 | auto")
    ap.add_argument("--min-note-s", type=float, default=MIN_NOTE_S)
    ap.add_argument("--split-semitones", type=float, default=SPLIT_SEMITONES)
    ap.add_argument("--selftest", action="store_true",
                    help="the checks that need no model: the clock, the .mid "
                         "round trip, the syllable split, the segmentation")
    ap.add_argument("--segments", default=None,
                    help="the analyzer's segment table (JSON) to join the "
                         "syllable rows against, on the session's frame index")
    ap.add_argument("--drill", type=int, default=0,
                    help="the drill-shaped analysis: find repeated phrases of "
                         "N words and tabulate each delivery's melody, timing "
                         "and energy (0 = off)")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if not a.input:
        ap.error("give a session folder or an audio file, or --selftest")
    src = Path(a.input)
    if src.is_dir():
        session_dir = src
        audio_path = src / "audio.wav"
        if not audio_path.is_file():
            print(f"no audio.wav in {src}", file=sys.stderr)
            return 2
    else:
        session_dir = None
        audio_path = src
    out_dir = Path(a.out) if a.out else Path.cwd() / "speech_to_score_out"

    report = analyse(audio_path, session_dir, out_dir, model=a.model,
                     device=a.device, compute_type=a.compute_type,
                     min_note_s=a.min_note_s,
                     split_semitones=a.split_semitones,
                     segments_path=Path(a.segments) if a.segments else None,
                     drill_n=a.drill)

    t = report.get("transcribe", {})
    print(f"input      {src}")
    print(f"out        {out_dir}")
    print(f"audio      {report['audio']['duration_s']:.2f} s @ "
          f"{report['audio']['source_sample_rate']} Hz "
          f"(analysis {report['audio']['analysis_sample_rate']} Hz)")
    print(f"clock      {report['clock'].get('n_frames')} frames, "
          f"median {report['clock'].get('median_fps', 0):.3f} fps")
    print(f"transcribe {t.get('engine')} {t.get('model')} on {t.get('device')} "
          f"({t.get('compute_type')}), {t.get('n_words')} words in "
          f"{t.get('total_s')} s")
    if report.get("no_speech"):
        ns = report["no_speech"]
        print(f"NO SPEECH  {ns['verdict']}")
        print(f"           peak RMS {ns['audio_peak_rms']:.8f}, "
              f"{ns['frames_over_energy_floor']} of {ns['n_frames']} frames "
              f"over the energy floor")
    else:
        p = report.get("pitch", {})
        s = report.get("syllables", {})
        print(f"pitch      {p.get('n_voiced')} voiced frames "
              f"({p.get('voiced_fraction') * 100:.1f}%), "
              f"{p.get('n_notes')} note events in {p.get('elapsed_s')} s")
        print(f"syllables  {s.get('n_syllables')} from {s.get('n_words')} words "
              f"({s.get('words_from_dictionary')} from {s.get('source')}, "
              f"{s.get('words_from_heuristic')} from the heuristic)")
        print(f"midi       {report['midi']['path']} "
              f"({report['midi']['bytes']} bytes, "
              f"{report['midi']['n_notes']} notes)")
        print()
        print(format_alignment(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
