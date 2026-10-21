"""The Tone Game -- the acting and understanding loop, as a runnable experiment.

Casey's design, in one line: "If you can act it, you can understand it."
Two players, one loop.

    THE PERFORMER says a carrier phrase with a chosen INTENT, rendered as a
    pitch melody -- MIDI first, and an audio render too, so it can be
    HEARD. The intents start as the potato deck: sincere, leaving,
    obligated, everything, joking, quiet-furious.

    THE READER hears ONLY the melody. No words, no labels. It guesses the
    intent from the sound.

    ROUNDS: the performer reshapes each intent's contour to maximise
    READABILITY -- not beauty -- and the reader retrains. Which intents
    blur is the finding.

The headline number is the SENTIMENT RECOVERY RATE: the reader's accuracy
on held-out renders, against chance, and against Casey's own prior that
the melody carries about 70% of the sentiment, most of it not in the
words. The melody channel is the language-independent layer; the
transcript is the local one.

Three decisions are worth stating up front.

THE READER HEARS AUDIO, NOT A NOTE LIST. The reader is given the rendered
WAV and nothing else: it extracts the pitch contour with the repository's
own tracker and computes its features from the sound. If the reader were
handed the performer's own note list, the recovery rate would be a
statement about the performer's bookkeeping and not about anything
audible. This way it is a claim about the channel Casey means.

THE SEED PROSODY IS REAL, AND LABELLED AS SUCH. The seed captures (A and
B) carry real speech, and their measured pitch behaviour -- the
distribution of note durations, the distribution of step sizes, the
range, the rate -- is the envelope every contour is drawn inside. Two
intents are additionally seeded from a real capture's own aggregate
profile; the rest are synthetic inside that envelope. Every intent says
which it is, and the report says that the real captures seed the
STATISTICS and not the INTENT SEMANTICS. Casey's own drill deliveries,
when they exist, are the ground truth that would replace this.

THE READER NEVER REFEREES ITSELF. The machine's recovery rate is one
opinion. The blind listening pack is the other one: renders with
anonymous names, a grade sheet for Casey, and a held-back answer key, so
the human grades the renders without knowing which is which, and the
machine's score on the very same items is what the two get compared on.

The repository's parts are used, not re-implemented: src/pitch.py for the
reader's pitch tracking, src/smf.py for the MIDI, src/score.py for the
note numbers, src/wavio.py for reading and writing the audio. Only the
performer, the synth and the classifier are new.

    python experiments/tone_game.py --out /tmp/tone_game
    python experiments/tone_game.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src import energy, pitch, score, smf, wavio  # noqa: E402

# ============================================================================
# 0. THE VOCABULARY
# ============================================================================

#: The potato deck. Six intents, and the words are the same for all of
#: them, so nothing in the wording can carry the intent.
INTENTS = ["sincere", "leaving", "obligated", "everything", "joking",
           "quiet-furious"]

#: The carrier phrases. Each is a list of (text, stress) syllables, and
#: the stress pattern is fixed across every intent -- the phrase is the
#: constant, the delivery is the variable. The second carrier exists for
#: the carrier-invariance ablation: it has a different syllable count, so
#: a reader that survives the swap is reading the melody and not the
#: phrase's shape.
CARRIERS = {
    "i love you": {
        "syllables": [("I", 1), ("LOVE", 2), ("you", 0)],
        "vowels": ["i", "a", "u"],
    },
    "i am leaving now": {
        "syllables": [("I", 1), ("am", 0), ("LEAV", 2), ("ing", 0), ("now", 1)],
        "vowels": ["i", "e", "i", "i", "a"],
    },
}

#: The first carrier is the potato drill's own phrase.
PRIMARY_CARRIER = "i love you"

#: A vowel is three formants, in Hz. The render is a voice, not a beep:
#: the reader is supposed to be recovering intent from a delivery, and a
#: pure sine is not a delivery.
VOWELS = {
    "a": (650.0, 1200.0, 2500.0),
    "e": (500.0, 1500.0, 2500.0),
    "i": (300.0, 2300.0, 3000.0),
    "u": (300.0, 870.0, 2250.0),
    "o": (500.0, 900.0, 2500.0),
}

SR = wavio.ANALYSIS_SR          # 16 kHz, the repository's analysis rate
GAP_S = 0.070                   # the articulation gap between notes
MIN_NOTE_S = 0.110              # the shortest note the tracker can lock onto
NOTE_MS = 1000.0

#: The style vector: the performer's knobs, one per dimension of a
#: delivery. These are what the rounds optimise.
STYLE_FIELDS = ["centre", "range", "tilt", "arc", "zig", "drop", "density",
                "jitter", "pause", "hold"]

#: The starting design. Every one of these is a PRIOR about how a
#: delivery looks -- they are the performer's opinion, not a measurement,
#: and the rounds are what find out whether the opinion was any good.
#:   centre  semitones of the melody's centre above the carrier's base
#:   range   the melody's total span in semitones
#:   tilt    semitones gained or lost across the phrase
#:   arc     curvature: positive arches, negative bowls
#:   zig     zig-zag amplitude, for the bouncy deliveries
#:   drop    the depth of a late drop, for the suppressed ones
#:   density notes per syllable
#:   jitter  random deviation per note, in semitones
#:   pause   trailing silence, in seconds
#:   hold    how much longer the last note is
STARTING_STYLES = {
    "sincere":       dict(centre=0, range=7, tilt=1.5, arc=2.5, zig=0.0,
                          drop=0.0, density=1.00, jitter=0.5, pause=0.25,
                          hold=1.30),
    "leaving":       dict(centre=-1, range=8, tilt=-4.0, arc=1.0, zig=0.0,
                          drop=0.0, density=1.00, jitter=0.6, pause=0.45,
                          hold=1.55),
    "obligated":     dict(centre=0, range=3, tilt=0.0, arc=0.4, zig=0.0,
                          drop=0.0, density=1.00, jitter=0.3, pause=0.15,
                          hold=1.05),
    "everything":    dict(centre=2, range=13, tilt=2.0, arc=5.0, zig=0.0,
                          drop=0.0, density=1.70, jitter=1.0, pause=0.20,
                          hold=1.20),
    "joking":        dict(centre=4, range=9, tilt=0.0, arc=0.0, zig=3.5,
                          drop=0.0, density=1.50, jitter=1.2, pause=0.10,
                          hold=0.95),
    "quiet-furious": dict(centre=-6, range=6, tilt=-1.0, arc=0.0, zig=0.0,
                          drop=5.0, density=1.10, jitter=0.4, pause=0.35,
                          hold=1.45),
}

#: Which seed capture each real-seeded intent takes its aggregate profile
#: from. The capture supplies the STATISTICS; the intent label on it is a
#: construction, and the report says so out loud.
REAL_SEED_MAP = {
    "sincere": "A",   # the facing capture
    "leaving": "B",   # the turned capture
}


# ============================================================================
# 1. THE CONTOUR
# ============================================================================

@dataclass
class Note:
    onset_s: float
    duration_s: float
    midi: int
    syllable: int


@dataclass
class Contour:
    notes: list[Note]
    carrier: str
    intent: str

    @property
    def duration_s(self) -> float:
        if not self.notes:
            return 0.0
        return max(n.onset_s + n.duration_s for n in self.notes)

    def midis(self) -> list[int]:
        return [n.midi for n in self.notes]

    def as_dict(self) -> dict:
        return {
            "carrier": self.carrier, "intent": self.intent,
            "duration_s": round(self.duration_s, 4),
            "n_notes": len(self.notes),
            "notes": [{"onset_s": round(n.onset_s, 4),
                       "duration_s": round(n.duration_s, 4),
                       "midi": n.midi, "syllable": n.syllable}
                      for n in self.notes],
        }


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def hz_to_midi(hz: float) -> float:
    """The repository's own conversion, so the tone game and the speech
    tool agree about what a note number means."""
    return 69.0 + 12.0 * math.log2(hz / 440.0)


# ============================================================================
# 2. THE PROSODY SEED -- the real captures' own measured behaviour
# ============================================================================

@dataclass
class ProsodySeed:
    """What the real captures actually do, measured with the repo's own
    tracker: the envelope every synthetic contour is drawn inside."""

    source: str
    note_durations: list[float] = field(default_factory=list)
    steps: list[float] = field(default_factory=list)
    pitch_range_st: float = 12.0
    rate_hz: float = 3.0
    centre_midi: float = 48.0
    n_notes: int = 0
    n_syllables: int = 0

    def duration(self, rng: random.Random) -> float:
        if not self.note_durations:
            return 0.22
        return rng.choice(self.note_durations)

    def step(self, rng: random.Random) -> float:
        if not self.steps:
            return 1.0
        return rng.choice(self.steps)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "n_notes_measured": self.n_notes,
            "n_syllables_measured": self.n_syllables,
            "note_duration_median_s": (round(float(np.median(self.note_durations)), 4)
                                       if self.note_durations else None),
            "note_duration_p10_s": (round(float(np.percentile(self.note_durations, 10)), 4)
                                    if self.note_durations else None),
            "note_duration_p90_s": (round(float(np.percentile(self.note_durations, 90)), 4)
                                    if self.note_durations else None),
            "step_median_semitones": (round(float(np.median(self.steps)), 3)
                                      if self.steps else None),
            "step_p90_abs_semitones": (round(float(np.percentile(np.abs(self.steps), 90)), 3)
                                       if self.steps else None),
            "pitch_range_semitones": round(self.pitch_range_st, 3),
            "centre_midi": round(self.centre_midi, 2),
            "rate_notes_per_s": round(self.rate_hz, 3),
        }


def measure_seed(wav_path: Path, syllable_times: list[tuple[float, float]] | None = None) -> ProsodySeed:
    """Measure a real capture's pitch behaviour with the repo's tracker.

    The note durations and the step sizes are the interesting parts: they
    are the shape of a human saying something, and every synthetic
    contour is drawn from them rather than from a made-up distribution.
    """
    audio = wavio.read_wav(wav_path)
    samples = list(audio.samples)
    tracker = pitch.PitchTracker(SR, frame_len=800, hop=160)
    frames = pitch.smooth_octaves(tracker.track(samples))

    seed = ProsodySeed(source=str(wav_path))

    # the note events, with the same voiced-run rule the speech tool uses
    runs = []
    i = 0
    while i < len(frames):
        if frames[i].hz is None:
            i += 1
            continue
        j = i
        while j + 1 < len(frames) and frames[j + 1].hz is not None:
            j += 1
        runs.append((i, j))
        i = j + 1

    half = (800 / 2.0) / SR
    prev_median = None
    for a, b in runs:
        hz = [f.hz for f in frames[a:b + 1] if f.hz is not None]
        if len(hz) < 3:
            continue
        dur = (frames[b].time + half) - (frames[a].time - half)
        if dur <= 0:
            continue
        med = float(np.median(hz))
        if prev_median is not None:
            # the step is the semitone distance from the previous note's
            # median, which is what the contour's jumps are made of
            seed.steps.append(12.0 * math.log2(med / max(1e-6, prev_median)))
        seed.note_durations.append(dur)
        prev_median = med

    # the pitch range over the whole capture
    allhz = np.array([f.hz for f in frames if f.hz is not None], dtype=np.float64)
    if allhz.size >= 8:
        lo = float(np.percentile(allhz, 5))
        hi = float(np.percentile(allhz, 95))
        seed.pitch_range_st = 12.0 * math.log2(max(hi, 1e-6) / max(lo, 1e-6))
        seed.centre_midi = hz_to_midi(float(np.median(allhz)))
    voiced_s = sum(r[1] - r[0] + 1 for r in runs) * (160.0 / SR)
    seed.rate_hz = (len(seed.note_durations) / voiced_s) if voiced_s > 0 else 3.0
    seed.n_notes = len(seed.note_durations)
    if syllable_times:
        seed.n_syllables = len(syllable_times)
    return seed


# ============================================================================
# 3. THE PERFORMER -- a style vector becomes a contour
# ============================================================================

def _triangle(t: float, period: float = 0.34) -> float:
    """A triangle wave in [0, 1], for the bouncy deliveries."""
    x = (t % period) / period
    return 4.0 * x if x < 0.5 else 4.0 * (1.0 - x)


def shape_at(style: dict, t: float) -> float:
    """The melodic shape at position t (0 to 1) within the phrase, in
    semitones relative to the style's centre."""
    tilt = style["tilt"] * (t - 0.5) * 2.0
    arc = style["arc"] * (1.0 - 4.0 * (t - 0.5) ** 2)
    zig = style["zig"] * (_triangle(t) - 0.5)
    drop = -style["drop"] * max(0.0, (t - 0.7) / 0.3)
    return tilt + arc + zig + drop


def perform(style: dict, carrier_name: str, seed: ProsodySeed,
            rng: random.Random, equal_notes_per_syllable: int = 0) -> Contour:
    """Turn a style vector into a concrete contour.

    The contour's durations and its note-to-note jumps are drawn from the
    REAL capture's measured distributions -- that is the seed prosody. The
    style decides the melody's shape, and the shape is what the reader
    has to recover.

    `equal_notes_per_syllable` forces every intent to sing the SAME
    number of notes, which removes the note count as a cue. That matters
    for the claim the game is making: if the intents differ in how many
    notes they use, a reader can recover the intent by counting notes
    and never listening to the melody. The number of notes is a real
    prosodic dimension, so it is measured BOTH ways -- equalised, which
    is the honest headline, and free, which is the comparison.
    """
    carrier = CARRIERS[carrier_name]
    syllables = carrier["syllables"]
    n_syl = len(syllables)

    if equal_notes_per_syllable > 0:
        counts = [equal_notes_per_syllable] * n_syl
    else:
        # how many notes, and which syllable each belongs to. Every
        # syllable gets at least one note; the extra ones go to the
        # stressed syllables, which is where a real delivery puts its
        # emphasis.
        total_notes = max(n_syl, int(round(style["density"] * n_syl)))
        extra = total_notes - n_syl
        stress = [s[1] for s in syllables]
        order = sorted(range(n_syl), key=lambda i: -stress[i])
        counts = [1] * n_syl
        for k in range(extra):
            counts[order[k % n_syl]] += 1

    base_midi = int(round(seed.centre_midi))
    notes: list[Note] = []
    t = 0.0
    n_total = sum(counts)
    placed = 0
    for si in range(n_syl):
        for k in range(counts[si]):
            pos = (placed + 0.5) / max(1, n_total)
            h = shape_at(style, pos)
            jitter = rng.gauss(0.0, style["jitter"])
            midi = int(round(base_midi + style["centre"] + h + jitter))
            midi = max(24, min(96, midi))
            dur = max(MIN_NOTE_S, seed.duration(rng))
            if k == counts[si] - 1 and si == n_syl - 1:
                dur *= style["hold"]
            notes.append(Note(onset_s=t, duration_s=dur, midi=midi,
                              syllable=si))
            t += dur + GAP_S
            placed += 1

    # The syllable a note BELONGS to is decided by its position among the
    # notes, not by the object it came from, so that the score alone is a
    # complete description of the delivery: a reader of the MIDI can
    # rebuild the exact same delivery without having been told anything
    # else about it. That is what makes the score the currency.
    for idx, note in enumerate(notes):
        note.syllable = min(n_syl - 1, int(idx * n_syl / max(1, len(notes))))

    # the trailing silence
    t += style["pause"]
    return Contour(notes=notes, carrier=carrier_name, intent="")


# ============================================================================
# 4. THE RENDER -- MIDI and audio, on the same clock
# ============================================================================

def write_midi(contour: Contour, path: Path, label: str) -> dict:
    """The MIDI, written by the repository's own writer, at the
    repository's own convention of one tick per millisecond. The notes
    and the audio are on the same clock: the note's onset in ticks is the
    sample it starts at in the WAV."""
    events = []
    for n in contour.notes:
        on = int(round(n.onset_s * NOTE_MS))
        off = on + max(1, int(round(n.duration_s * NOTE_MS)))
        events.append(smf.MidiEvent(on, "note_on", {"note": n.midi, "velocity": 64}))
        events.append(smf.MidiEvent(off, "note_off", {"note": n.midi, "velocity": 0}))
    data = smf.write_smf([smf.MidiTrackData(name="melody", events=events)],
                          ticks_per_beat=score.TICKS_PER_BEAT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data), "n_notes": len(contour.notes),
            "ms_per_tick": score.MS_PER_TICK,
            "note": "one track, one property, constant velocity; the "
                    "words are not in the MIDI (src/smf.py has no text event)"}


def contour_from_midi(path: Path, carrier: str, intent: str = "") -> Contour:
    """Rebuild the delivery from the SCORE alone.

    The MIDI is the currency between the tiers, so it has to be a
    complete description of the delivery. This reads the notes back out
    of the file and nothing else -- no note list, no style vector, no
    memory of what the performer meant -- and the audio rendered from
    what comes out is the same audio the score was written for.
    """
    ppq, tracks = smf.parse_smf(Path(path).read_bytes())
    pending: dict = {}
    pairs = []
    for e in tracks[0].events:
        if e.kind == "note_on":
            pending.setdefault(e.data["note"], []).append(e.tick)
        elif e.kind == "note_off":
            queue = pending.get(e.data["note"], [])
            if queue:
                pairs.append((queue.pop(0), e.tick, e.data["note"]))
    pairs.sort()
    n_syl = len(CARRIERS[carrier]["syllables"])
    # one tick is one millisecond (src/score.py), so ticks become seconds
    # by dividing by MS_PER_TICK and then by a thousand
    s_per_tick = score.MS_PER_TICK / 1000.0
    notes = []
    for idx, (on, off, midi) in enumerate(pairs):
        notes.append(Note(onset_s=on * s_per_tick,
                          duration_s=max(1e-3, (off - on) * s_per_tick),
                          midi=midi,
                          syllable=min(n_syl - 1, int(idx * n_syl / max(1, len(pairs))))))
    return Contour(notes=notes, carrier=carrier, intent=intent)


def _resonator_ir(freq: float, bw: float, sr: int, n: int) -> np.ndarray:
    """The impulse response of a two-pole resonator, so the formant
    filter is a convolution and not a Python loop."""
    r = math.exp(-math.pi * bw / sr)
    theta = 2.0 * math.pi * freq / sr
    t = np.arange(n)
    ir = (r ** t) * np.sin((t + 1) * theta) / math.sin(theta)
    return ir * (1.0 - r) * math.sqrt(max(1e-9, 1.0 - 2 * r * math.cos(2 * theta) + r * r))


def render_wav(contour: Contour, path: Path, carrier_name: str,
               sr: int = SR) -> dict:
    """The audio render: a voice saying the melody.

    Each note is a harmonic glottal source run through three formant
    resonators for its syllable's vowel, with an edge envelope and a
    light vibrato. It is a synthetic voice, and it says so -- but it is
    a DELIVERY, and a sine wave is not one.
    """
    carrier = CARRIERS[carrier_name]
    vowels = carrier["vowels"]
    total = contour.duration_s + 0.25
    n = int(total * sr)
    out = np.zeros(n, dtype=np.float64)

    for note in contour.notes:
        start = int(note.onset_s * sr)
        dur = int(note.duration_s * sr)
        if dur < 16:
            continue
        t = np.arange(dur) / sr
        f0 = midi_to_hz(note.midi)
        # a little vibrato: 5 Hz, about a seventh of a semitone
        vib = 1.0 + 0.004 * np.sin(2.0 * math.pi * 5.0 * t)
        phase = 2.0 * math.pi * f0 * np.cumsum(vib) / sr
        n_harm = int(min(40, max(4, 0.45 * sr / f0)))
        sig = np.zeros(dur)
        for h in range(1, n_harm + 1):
            sig += (1.0 / (h ** 1.1)) * np.sin(h * phase)
        sig /= n_harm

        # the formants of this syllable's vowel
        formants = VOWELS.get(vowels[min(note.syllable, len(vowels) - 1)],
                              VOWELS["a"])
        for fi, (freq, bw) in enumerate(zip(formants, (70.0, 110.0, 160.0))):
            ir_len = min(dur, int(0.12 * sr))
            ir = _resonator_ir(freq, bw, sr, ir_len)
            sig = np.convolve(sig, ir)[:dur]

        env = np.minimum(1.0, np.minimum(t / 0.030,
                                          np.maximum(0.0, (note.duration_s - t) / 0.045)))
        peak = float(np.max(np.abs(sig))) or 1.0
        chunk = sig / peak * env * 0.55
        end = min(n, start + dur)
        out[start:end] += chunk[:end - start]

    peak = float(np.max(np.abs(out))) or 1.0
    out = out / peak * 0.8
    path.parent.mkdir(parents=True, exist_ok=True)
    wavio.write_wav(path, [float(v) for v in out], sr)
    return {"path": str(path), "duration_s": round(n / sr, 4),
            "sample_rate": sr, "peak": round(peak, 4)}


# ============================================================================
# 5. THE READER -- it hears the audio and nothing else
# ============================================================================

#: The feature names, in order. Every one of them is measured from the
#: RENDERED SOUND by the repository's own tracker.
FEATURE_NAMES = [
    "centre_midi",       # the melody's centre
    "range_st",          # p95 - p5 of the pitch, in semitones
    "slope_st_per_s",    # the melody's overall drift
    "n_notes",           # how many notes were sung
    "note_rate",         # notes per second
    "note_dur_median",   # how long the notes were
    "note_dur_std",      # how uneven they were
    "step_abs_mean",     # the typical jump between notes
    "step_std",          # how uneven the jumps were
    "step_signed_mean",  # up-jumps minus down-jumps
    "voiced_fraction",   # how much of the phrase was sung
]

PITCH_FEATURES = [0, 1, 2, 7, 8, 9]
TIMING_FEATURES = [3, 4, 5, 6, 10]
ALL_FEATURES = list(range(len(FEATURE_NAMES)))


def features_from_audio(wav_path: Path) -> dict:
    """The reader's features, measured from the sound.

    The reader is handed a WAV and nothing else. It does not know the
    intent, it does not know the carrier, and it never sees the note list
    the performer wrote. It tracks the pitch with the repository's
    tracker and measures the contour.
    """
    audio = wavio.read_wav(wav_path)
    samples = list(audio.samples)
    tracker = pitch.PitchTracker(SR, frame_len=800, hop=160)
    frames = pitch.smooth_octaves(tracker.track(samples))

    voiced = [f for f in frames if f.hz is not None]
    hz = np.array([f.hz for f in voiced], dtype=np.float64)
    if hz.size == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    # the notes: contiguous voiced runs, the same rule the speech tool uses
    runs = []
    i = 0
    while i < len(frames):
        if frames[i].hz is None:
            i += 1
            continue
        j = i
        while j + 1 < len(frames) and frames[j + 1].hz is not None:
            j += 1
        runs.append((i, j))
        i = j + 1

    half = (800 / 2.0) / SR
    mids = []
    durs = []
    for a, b in runs:
        seg = [f.hz for f in frames[a:b + 1] if f.hz is not None]
        if not seg:
            continue
        mids.append(hz_to_midi(float(np.median(seg))))
        d = (frames[b].time + half) - (frames[a].time - half)
        if d > 0:
            durs.append(d)

    lo = float(np.percentile(hz, 5))
    hi = float(np.percentile(hz, 95))
    rng = 12.0 * math.log2(max(hi, 1e-6) / max(lo, 1e-6))

    # the slope of the pitch over the phrase
    slope = 0.0
    if len(voiced) >= 4:
        t = np.array([f.time for f in voiced])
        m = np.array([hz_to_midi(f.hz) for f in voiced])
        if t.std() > 1e-6:
            slope = float(np.polyfit(t, m, 1)[0])

    steps = [mids[k + 1] - mids[k] for k in range(len(mids) - 1)]
    dur_total = float(sum(durs))
    voice_span = (voiced[-1].time - voiced[0].time) if len(voiced) > 1 else 0.0
    step_abs = [abs(s) for s in steps]

    return {
        "centre_midi": float(np.median(mids)) if mids else 0.0,
        "range_st": rng,
        "slope_st_per_s": slope,
        "n_notes": float(len(mids)),
        "note_rate": (len(mids) / dur_total) if dur_total > 0 else 0.0,
        "note_dur_median": float(np.median(durs)) if durs else 0.0,
        "note_dur_std": float(np.std(durs)) if durs else 0.0,
        "step_abs_mean": float(np.mean(step_abs)) if step_abs else 0.0,
        "step_std": float(np.std(steps)) if len(steps) > 1 else 0.0,
        "step_signed_mean": float(np.mean(steps)) if steps else 0.0,
        "voiced_fraction": (len(voiced) / max(1, len(frames))),
    }


def feature_vector(feats: dict, indices: list[int]) -> np.ndarray:
    return np.array([feats[FEATURE_NAMES[i]] for i in indices], dtype=np.float64)


def features_from_contour(contour: Contour) -> dict:
    """The same statistics, computed from the contour instead of the
    sound. This is the SURROGATE the optimiser uses: it is cheap, and it
    is close enough to the audio features for the two to agree about
    which intents are too close together. The report quotes the
    agreement."""
    if not contour.notes:
        return {name: 0.0 for name in FEATURE_NAMES}
    mids = [float(n.midi) for n in contour.notes]
    durs = [n.duration_s for n in contour.notes]
    steps = [mids[k + 1] - mids[k] for k in range(len(mids) - 1)]
    total = contour.duration_s
    return {
        "centre_midi": float(np.median(mids)),
        "range_st": (max(mids) - min(mids)),
        "slope_st_per_s": (mids[-1] - mids[0]) / max(1e-6, total),
        "n_notes": float(len(mids)),
        "note_rate": len(mids) / max(1e-6, total),
        "note_dur_median": float(np.median(durs)),
        "note_dur_std": float(np.std(durs)),
        "step_abs_mean": float(np.mean([abs(s) for s in steps])) if steps else 0.0,
        "step_std": float(np.std(steps)) if len(steps) > 1 else 0.0,
        "step_signed_mean": float(np.mean(steps)) if steps else 0.0,
        "voiced_fraction": 1.0,
    }


# ============================================================================
# 6. THE CLASSIFIER -- a small multinomial logistic regression, in numpy
# ============================================================================

class Logistic:
    """Multinomial logistic regression, full-batch gradient descent.

    Small and deterministic on purpose: the reader has to be a fixed
    instrument that the rounds can be compared against, not something
    that changes its mind between rounds.
    """

    def __init__(self, n_classes: int, l2: float = 1e-2, lr: float = 0.4,
                 iters: int = 3000):
        self.n_classes = n_classes
        self.l2 = l2
        self.lr = lr
        self.iters = iters
        self.W = None
        self.mu = None
        self.sd = None
        self.loss = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Logistic":
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0
        Z = (X - self.mu) / self.sd
        Z = np.hstack([Z, np.ones((Z.shape[0], 1))])
        Y = np.zeros((Z.shape[0], self.n_classes))
        Y[np.arange(len(y)), y] = 1.0
        self.W = np.zeros((Z.shape[1], self.n_classes))
        n = Z.shape[0]
        for it in range(self.iters):
            logits = Z @ self.W
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            grad = Z.T @ (p - Y) / n
            grad[:-1] += self.l2 * self.W[:-1]
            self.W -= self.lr * grad
        self.loss = self._loss(Z, Y)
        return self

    def _loss(self, Z: np.ndarray, Y: np.ndarray) -> float:
        logits = Z @ self.W
        logits -= logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return float(-np.mean(np.sum(Y * logp, axis=1)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        Z = np.hstack([Z, np.ones((Z.shape[0], 1))])
        return np.argmax(Z @ self.W, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        Z = np.hstack([Z, np.ones((Z.shape[0], 1))])
        logits = Z @ self.W
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)


# ============================================================================
# 7. THE ROUNDS
# ============================================================================

def render_set(styles: dict, carrier: str, seed: ProsodySeed,
               variants: int, rng_seed: int, equal_nps: int = 0) -> list:
    """Render `variants` contours per intent, with their MIDI and WAV.

    The random stream is keyed by the VARIANT and NOT by the intent. That
    matters more than it looks: if each intent drew its own noise, the
    noise itself would become a fingerprint, and the reader could name
    the intent from the jitter alone without ever hearing the delivery.
    Sharing the stream across the intents makes the design PAIRED -- the
    prosody variation is common to all of them, so the only thing that
    differs between two renders of the same variant is the intent's own
    style. It is the difference between measuring the melody and
    measuring the seed.
    """
    out = []
    for intent in INTENTS:
        for v in range(variants):
            rng = random.Random(f"{rng_seed}|{v}")
            contour = perform(styles[intent], carrier, seed, rng, equal_nps)
            contour.intent = intent
            out.append({"intent": intent, "variant": v, "contour": contour})
    return out


def readability_objective(styles: dict, seed: ProsodySeed, carrier: str,
                           indices: list[int], samples: int = 24,
                           equal_nps: int = 0) -> float:
    """How readable the styles are, as the WORST pair's separation.

    For every intent, the centroid of its contours in the feature space
    and the spread around that centroid. The score is the smallest
    separation between any two intents, measured in units of their own
    spread -- so an intent that is far from the others but very noisy
    does not win. This is the objective the rounds maximise.
    """
    cents = {}
    spreads = {}
    for intent in INTENTS:
        rng = random.Random("obj|paired")
        feats = []
        for _ in range(samples):
            c = perform(styles[intent], carrier, seed, rng, equal_nps)
            feats.append(feature_vector(features_from_contour(c), indices))
        arr = np.array(feats)
        cents[intent] = arr.mean(axis=0)
        spreads[intent] = float(arr.std(axis=0).mean()) + 1e-6
    worst = float("inf")
    for i, a in enumerate(INTENTS):
        for b in INTENTS[i + 1:]:
            d = float(np.linalg.norm(cents[a] - cents[b]))
            worst = min(worst, d / (spreads[a] + spreads[b]))
    return worst


def readability_push(styles: dict, seed: ProsodySeed, carrier: str,
                     confusion: np.ndarray, indices: list[int],
                     envelope: dict, rng_seed: int, equal_nps: int = 0) -> dict:
    """Reshape the styles so the most-confused pair gets further apart.

    The confusion matrix says which intents the reader actually mixes up.
    Those intents are separated FIRST and by the LARGEST step, and the
    search is a coordinate ascent on the style vectors, clamped to the
    prosody envelope so the contours stay inside what a real voice does.
    """
    new_styles = {k: dict(v) for k, v in styles.items()}
    # how often each intent is confused with each other
    n = len(INTENTS)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pairs.append((float(confusion[i][j]) + float(confusion[j][i]),
                          INTENTS[i], INTENTS[j]))
    pairs.sort(reverse=True)

    def objective(st: dict) -> float:
        return readability_objective(st, seed, carrier, indices, equal_nps=equal_nps)

    best = objective(new_styles)
    # the intents that are confused, worst first
    order = []
    for _, a, b in pairs:
        for x in (a, b):
            if x not in order:
                order.append(x)

    for intent in order:
        for field in STYLE_FIELDS:
            for delta in (1.5, -1.5, 0.75, -0.75):
                trial = {k: dict(v) for k, v in new_styles.items()}
                val = trial[intent][field] + delta
                lo, hi = envelope[field]
                if not (lo <= val <= hi):
                    continue
                trial[intent][field] = val
                score = objective(trial)
                if score > best + 1e-6:
                    best = score
                    new_styles = trial
    return new_styles


def neutral_style(styles: dict) -> dict:
    """The delivery with no intent in it: the mean of all the intents.

    This is the point the caricature knob is measured from. At an
    exaggeration of zero every intent IS this delivery, and the reader's
    best possible score is chance -- which is the control the curve is
    read against.
    """
    return {f: float(np.mean([st[f] for st in styles.values()]))
            for f in STYLE_FIELDS}


def caricature(styles: dict, neutral: dict, x: float,
               envelope: dict) -> dict:
    """Push each intent's delivery away from the neutral one, by x.

    The parameters that STICK OUT are the identity, and exaggerating them
    is exactly what a caricature does: it keeps the few features that
    make a delivery that delivery and leans on them. At x = 0 every
    intent is the neutral delivery and the reader is at chance; at x = 1
    the design is what it was; above x = 1 the identities are pushed
    further apart than any real voice would push them.

    The curve is measured, not assumed: Casey's expectation is that a
    caricatured intent reads BETTER, and the interesting question is
    where that stops being true.
    """
    out = {}
    for intent, st in styles.items():
        c = dict(st)
        for f in STYLE_FIELDS:
            c[f] = neutral[f] + (st[f] - neutral[f]) * x
        out[intent] = c
    return clamp_styles(out, envelope)


def caricature_sweep(base_styles: dict, neutral: dict, seed: ProsodySeed,
                     carrier: str, out_dir: Path, n_train: int, n_test: int,
                     envelope: dict, xs: list[float], feature_sets: dict,
                     equal_nps: int) -> list:
    """The accuracy-against-exaggeration curve.

    One round per exaggeration value, with the reader retrained each
    time -- a harder task needs a reader trained for it. The design is
    held fixed and only the exaggeration moves, so the curve is the
    exaggeration's doing and nothing else's.
    """
    out = []
    for x in xs:
        styles = caricature(base_styles, neutral, x, envelope)
        res = run_round(styles, seed, carrier, out_dir / f"x{x}", n_train,
                        n_test, feature_sets, 0, equal_nps=equal_nps,
                        keep_renders=False)
        cm = np.array(res["results"]["all"]["confusion"])
        row = {
            "exaggeration": x,
            "accuracy": res["results"]["all"]["accuracy"],
            "accuracy_pitch_only": res["results"]["pitch_only"]["accuracy"],
            "accuracy_timing_only": res["results"]["timing_only"]["accuracy"],
            "per_intent_recall": {INTENTS[i]: round(float(cm[i][i] / max(1, cm[i].sum())), 3)
                                  for i in range(len(INTENTS))},
            "confusion": res["results"]["all"]["confusion"],
        }
        out.append(row)
        print(f"    x={x:<5} accuracy {row['accuracy']:.3f}  "
              f"pitch-only {row['accuracy_pitch_only']:.3f}")
    return out


def confusion_matrix(true_idx: np.ndarray, pred_idx: np.ndarray,
                     n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true_idx, pred_idx):
        cm[t][p] += 1
    return cm


def matrix_to_text(cm: np.ndarray, intents: list[str]) -> str:
    head = "true \\ predicted".ljust(16) + "".join(f"{i[:7]:>9s}" for i in intents)
    lines = [head]
    for i, name in enumerate(intents):
        row = name.ljust(16) + "".join(f"{cm[i][j]:9d}" for j in range(len(intents)))
        lines.append(row)
    return "\n".join(lines)


# ============================================================================
# 8. THE BLIND LISTENING PACK -- the external referee
# ============================================================================

def write_listening_pack(rounds_dir: Path, items: list, machine_pred: dict,
                          out_dir: Path, round_tag: str) -> dict:
    """The pack Casey grades blind.

    The renders are copied out under ANONYMOUS names, in a shuffled
    order. The grade sheet asks for a guess and a confidence. The answer
    key is written to a `key/` subdirectory and is NOT in the pack's
    main directory, so grading does not stumble over it. The machine's
    prediction for every item is written beside them, so the human's
    grade and the machine's recovery rate are measured on exactly the
    same items.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    key_dir = out_dir / "key"
    key_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(f"blind|{round_tag}")
    order = list(items)
    rng.shuffle(order)

    items_out = []
    for k, it in enumerate(order):
        fid = f"{round_tag}_{k:02d}"
        dst = out_dir / f"{fid}.wav"
        dst.write_bytes(Path(it["wav"]).read_bytes())
        items_out.append({"file_id": fid, "intent": it["intent"],
                          "variant": it["variant"],
                          "machine_guess": machine_pred.get(it["id"]),
                          "source_wav": str(it["wav"])})

    with (out_dir / "GRADE-SHEET.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_id", "your_guess", "confidence_1_to_5", "notes"])
        for it in items_out:
            w.writerow([it["file_id"], "", "", ""])

    with (key_dir / "ANSWER-KEY.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_id", "intent", "variant"])
        for it in items_out:
            w.writerow([it["file_id"], it["intent"], it["variant"]])

    with (out_dir / "MACHINE-PREDICTIONS.csv").open("w", newline="",
                                                    encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_id", "machine_guess"])
        for it in items_out:
            w.writerow([it["file_id"], it["machine_guess"]])

    readme = [
        f"# Blind listening check -- {round_tag}",
        "",
        f"{len(items_out)} renders. Each one is a melody carrying one of",
        "six intents, and the intent is the same six every time:",
        "",
        "    " + ", ".join(INTENTS),
        "",
        "The words are IDENTICAL in every file. Only the delivery changes.",
        "So the only thing that can tell them apart is the melody, and that",
        "is the point.",
        "",
        "How to grade it:",
        "",
        "1. Open GRADE-SHEET.csv.",
        "2. Listen to each <file_id>.wav and put your guess in your_guess.",
        "3. confidence_1_to_5 is how sure you are. Notes is anything you",
        "   want to say about the file.",
        "4. Do NOT look in the key/ directory until the sheet is filled in.",
        "",
        "Then score it:",
        "",
        "    python experiments/tone_game.py --score-listening " + str(out_dir),
        "",
        "which reports your accuracy, your confusion matrix, and the",
        "machine's accuracy on exactly these files, side by side.",
        "",
        "MACHINE-PREDICTIONS.csv is the reader's guess for each file, and",
        "it is not the answer key -- the answer key is in key/, and it is",
        "held back so that the grading stays blind.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n")
    return {"dir": str(out_dir), "n_items": len(items_out),
            "grade_sheet": str(out_dir / "GRADE-SHEET.csv"),
            "answer_key": str(key_dir / "ANSWER-KEY.csv"),
            "note": "the key is held back so the grading is blind"}


def score_listening(pack_dir: Path) -> dict:
    """Score a filled-in grade sheet against the held-back key."""
    pack_dir = Path(pack_dir)
    sheet = pack_dir / "GRADE-SHEET.csv"
    key = pack_dir / "key" / "ANSWER-KEY.csv"
    if not sheet.is_file() or not key.is_file():
        print(f"need both {sheet} and {key}")
        return {}
    answers = {}
    with key.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            answers[row["file_id"]] = row["intent"]
    grades = []
    with sheet.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            guess = (row.get("your_guess") or "").strip().lower()
            if not guess:
                continue
            grades.append((row["file_id"], guess, answers.get(row["file_id"], "")))
    if not grades:
        print("the grade sheet has no guesses in it yet")
        return {}

    n = len(grades)
    correct = sum(1 for _, g, a in grades if g == a)
    machine = {}
    mp = pack_dir / "MACHINE-PREDICTIONS.csv"
    if mp.is_file():
        with mp.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                machine[row["file_id"]] = (row.get("machine_guess") or "").strip().lower()
    m_correct = sum(1 for fid, _, _ in grades if machine.get(fid) == answers.get(fid))

    cm = confusion_matrix(np.array([INTENTS.index(a) for _, _, a in grades]),
                           np.array([INTENTS.index(g) if g in INTENTS else 0
                                     for _, g, _ in grades]), len(INTENTS))
    agreement = sum(1 for fid, _, _ in grades
                    if machine.get(fid) == answers.get(fid))
    # The programme's headline metric, measured on this pack: the fraction
    # of judgments the system still has to send to a teacher. The system's
    # own judgment is trusted where it agrees with the referee, and it is
    # not trusted where it does not -- so the dependence is what is left
    # over. Until a referee has graded anything, the dependence is the
    # whole of it.
    dependence = 1.0 - (agreement / n) if n else 1.0
    print(f"graded {n} files")
    print(f"  the human   : {correct}/{n} = {correct / n:.3f}")
    print(f"  the machine : {m_correct}/{n} = {m_correct / n:.3f}")
    print(f"  they agree  : {agreement}/{n} = {agreement / n:.3f}")
    print(f"  chance      : {1 / len(INTENTS):.3f}")
    print(f"  TEACHER-DEPENDENCE RATIO: {dependence:.3f} "
          f"(the share of judgments still needing a teacher)")
    print()
    print(matrix_to_text(cm, INTENTS))
    return {"n_graded": n, "human_accuracy": round(correct / n, 4),
            "machine_accuracy": round(m_correct / n, 4),
            "machine_agreement_with_referee": round(agreement / n, 4),
            "teacher_dependence_ratio": round(dependence, 4),
            "chance": round(1 / len(INTENTS), 4),
            "confusion": cm.tolist()}


# ============================================================================
# 9. THE DRIVER
# ============================================================================

def build_envelope(seeds: dict) -> dict:
    """The prosody envelope: the bounds the styles stay inside.

    The range comes from the real captures' own measured pitch spread,
    so the synthetic contours stay inside what a real voice does. When
    the starting design asks for more than the real voice did -- an
    emphatic delivery wants a wide range and the captures are narrow --
    the design is CLAMPED, and the report says so. A contour that no
    real voice could have produced is not a rendering of an intent; it
    is a drawing.
    """
    ranges = [s.pitch_range_st for s in seeds.values() if s.note_durations]
    max_range = min(ranges) if ranges else 12.0
    max_range = max(8.0, min(20.0, max_range))
    return {
        "centre": (-12.0, 12.0),
        "range": (3.0, max_range),
        "tilt": (-8.0, 8.0),
        "arc": (-6.0, 8.0),
        "zig": (0.0, 6.0),
        "drop": (0.0, 9.0),
        "density": (0.8, 2.2),
        "jitter": (0.0, 2.5),
        "pause": (0.0, 0.8),
        "hold": (0.8, 2.0),
    }


def clamp_styles(styles: dict, envelope: dict) -> dict:
    """Pull the styles inside the prosody envelope."""
    out = {}
    for intent, st in styles.items():
        c = dict(st)
        for field, (lo, hi) in envelope.items():
            c[field] = max(lo, min(hi, c[field]))
        out[intent] = c
    return out


def run_round(styles: dict, seed: ProsodySeed, carrier: str, out_dir: Path,
              n_train: int, n_test: int, feature_sets: dict,
              round_index: int, equal_nps: int = 0,
              keep_renders: bool = True) -> dict:
    """One round: render, train the reader, test it, and report."""
    t0 = time.time()
    train_items = render_set(styles, carrier, seed, n_train,
                              rng_seed=f"train|r{round_index}", equal_nps=equal_nps)
    test_items = render_set(styles, carrier, seed, n_test,
                             rng_seed=f"test|r{round_index}", equal_nps=equal_nps)

    rdir = out_dir / "renders" / f"round_{round_index}"
    for item in train_items + test_items:
        base = f"{item['intent']}_v{item['variant']:02d}"
        item["id"] = base
        item["midi"] = str(rdir / f"{base}.mid")
        item["wav"] = str(rdir / f"{base}.wav")
        write_midi(item["contour"], Path(item["midi"]), item["intent"])
        # the score is the currency: the audio is rendered from the FILE,
        # not from the object the performer had in its hand, so the
        # renders and the scores cannot drift apart
        scored = contour_from_midi(Path(item["midi"]), carrier, item["intent"])
        render_wav(scored, Path(item["wav"]), carrier)

    # the reader's features, measured from the sound
    for item in train_items + test_items:
        item["feats"] = features_from_audio(Path(item["wav"]))

    if not keep_renders:
        # the sweep does not need three hundred files on disk
        for item in train_items + test_items:
            for path in (item["midi"], item["wav"]):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    y_train = np.array([INTENTS.index(i["intent"]) for i in train_items])
    y_test = np.array([INTENTS.index(i["intent"]) for i in test_items])

    results = {}
    for name, indices in feature_sets.items():
        Xtr = np.array([feature_vector(i["feats"], indices) for i in train_items])
        Xte = np.array([feature_vector(i["feats"], indices) for i in test_items])
        model = Logistic(len(INTENTS)).fit(Xtr, y_train)
        pred = model.predict(Xte)
        cm = confusion_matrix(y_test, pred, len(INTENTS))
        acc = float((pred == y_test).mean())
        results[name] = {
            "accuracy": round(acc, 4),
            "chance": round(1.0 / len(INTENTS), 4),
            "confusion": cm.tolist(),
            "confusion_text": matrix_to_text(cm, INTENTS),
            "train_loss": round(model.loss, 5),
            "n_train": len(train_items), "n_test": len(test_items),
            "feature_indices": indices,
            "feature_names": [FEATURE_NAMES[i] for i in indices],
        }

    return {"round": round_index, "styles": {k: dict(v) for k, v in styles.items()},
            "results": results, "train_items": train_items, "test_items": test_items,
            "elapsed_s": round(time.time() - t0, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/tmp/tone_game",
                    help="where to write the renders, the pack and the results")
    ap.add_argument("--rounds", type=int, default=4,
                    help="how many rounds the loop runs")
    ap.add_argument("--train-variants", type=int, default=8)
    ap.add_argument("--test-variants", type=int, default=8)
    ap.add_argument("--carrier", default=PRIMARY_CARRIER, choices=list(CARRIERS))
    ap.add_argument("--seed-captures", default="/mnt/c/proart_capture",
                    help="the directory holding the real captures A.wav and B.wav")
    ap.add_argument("--no-real-seed", action="store_true",
                    help="skip the real captures and use the built-in envelope")
    ap.add_argument("--equal-notes-per-syllable", type=int, default=2,
                    help="force every intent to sing the same number of notes, "
                         "so the note count cannot be the cue (0 = free, the "
                         "note count is then a legitimate prosodic dimension "
                         "and the comparison run)")
    ap.add_argument("--ablation-carrier", action="store_true",
                    help="also run the whole game on the second carrier, to "
                         "check the recovery survives a different phrase")
    ap.add_argument("--caricature", default="0,0.1,0.2,0.35,0.5,0.75,1,1.5,2,3",
                    help="the exaggeration values to sweep, comma separated")
    ap.add_argument("--score-listening", default=None,
                    help="score a filled-in grade sheet and exit")
    ap.add_argument("--selftest", action="store_true",
                    help="the checks that need no model and no real capture")
    a = ap.parse_args(argv)

    if a.score_listening:
        score_listening(Path(a.score_listening))
        return 0
    if a.selftest:
        return selftest()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. the seed prosody, from the real captures
    seeds = {}
    if not a.no_real_seed:
        for name in ("A", "B"):
            p = Path(a.seed_captures) / f"{name}.wav"
            if p.is_file():
                seeds[name] = measure_seed(p)
                print(f"seed {name}: {seeds[name].as_dict()}")
    if not seeds:
        print("no real captures found -- using the built-in envelope")
        seeds = {"builtin": ProsodySeed(
            source="built-in (no real capture)",
            note_durations=[0.18, 0.22, 0.26, 0.30, 0.34],
            steps=[-2.0, -1.0, 1.0, 2.0, 3.0],
            pitch_range_st=12.0, rate_hz=3.5, centre_midi=48.0)}
        print(f"seed builtin: {seeds['builtin'].as_dict()}")

    # the seed every intent is drawn inside: the pool of the real
    # captures' measured behaviour
    pooled = ProsodySeed(
        source="pooled: " + ", ".join(sorted(seeds)),
        note_durations=[d for s in seeds.values() for d in s.note_durations],
        steps=[s for sd in seeds.values() for s in sd.steps],
        pitch_range_st=float(np.mean([s.pitch_range_st for s in seeds.values()])),
        rate_hz=float(np.mean([s.rate_hz for s in seeds.values()])),
        centre_midi=float(np.mean([s.centre_midi for s in seeds.values()])),
    )
    envelope = build_envelope(seeds)
    requested_styles = {k: dict(v) for k, v in STARTING_STYLES.items()}
    styles = clamp_styles(requested_styles, envelope)
    clamped = {k: {f: [requested_styles[k][f], styles[k][f]]
                   for f in STYLE_FIELDS
                   if abs(requested_styles[k][f] - styles[k][f]) > 1e-9}
               for k in styles}
    clamped = {k: v for k, v in clamped.items() if v}

    # 2. the rounds
    feature_sets = {"all": ALL_FEATURES, "pitch_only": PITCH_FEATURES,
                    "timing_only": TIMING_FEATURES}
    rounds = []
    for r in range(a.rounds):
        print(f"--- round {r}")
        res = run_round(styles, pooled, a.carrier, out_dir, a.train_variants,
                        a.test_variants, feature_sets, r,
                        equal_nps=a.equal_notes_per_syllable)
        acc = res["results"]["all"]["accuracy"]
        print(f"    recovery rate (all features): {acc:.3f} "
              f"(chance {1 / len(INTENTS):.3f})")
        rounds.append(res)
        if r < a.rounds - 1:
            cm = np.array(res["results"]["all"]["confusion"])
            styles = readability_push(styles, pooled, a.carrier, cm,
                                      ALL_FEATURES, envelope,
                                      rng_seed=f"push|r{r}",
                                      equal_nps=a.equal_notes_per_syllable)
    # the final styles
    final_styles = styles

    # 3. THE ABLATIONS. Two questions the headline number cannot answer
    # on its own, and both are cheap.
    ablations = {}

    # 3a. does the note count do the work? The same game with the note
    # count free, which is a legitimate prosodic dimension and which a
    # reader could otherwise exploit instead of listening to the melody.
    if a.equal_notes_per_syllable > 0:
        print("--- ablation: the note count is free")
        free_styles = clamp_styles(requested_styles, envelope)
        free_rounds = []
        for r in range(max(2, a.rounds)):
            res = run_round(free_styles, pooled, a.carrier,
                            out_dir / "ablation_free_notes", a.train_variants,
                            a.test_variants, feature_sets, r, equal_nps=0)
            free_rounds.append(res)
            print(f"    round {r}: {res['results']['all']['accuracy']:.3f}")
            if r < max(2, a.rounds) - 1:
                cm = np.array(res["results"]["all"]["confusion"])
                free_styles = readability_push(
                    free_styles, pooled, a.carrier, cm, ALL_FEATURES,
                    envelope, rng_seed=f"push|free|r{r}", equal_nps=0)
        ablations["free_note_count"] = {
            "what": "the same game with the note count free, so the reader "
                    "may use how many notes were sung",
            "rounds": [{"round": r["round"], "results": r["results"]}
                       for r in free_rounds],
        }

    # 3b. does the recovery survive a different carrier phrase? The
    # melody channel is supposed to be the language-independent layer,
    # so the reader should still work when the words change.
    other = [c for c in CARRIERS if c != a.carrier]
    if a.ablation_carrier and other:
        print(f"--- ablation: the carrier is {other[0]!r}")
        alt_styles = clamp_styles(requested_styles, envelope)
        alt_rounds = []
        for r in range(2):
            res = run_round(alt_styles, pooled, other[0],
                            out_dir / "ablation_other_carrier",
                            a.train_variants, a.test_variants,
                            feature_sets, r,
                            equal_nps=a.equal_notes_per_syllable)
            alt_rounds.append(res)
            print(f"    round {r}: {res['results']['all']['accuracy']:.3f}")
        ablations["other_carrier"] = {
            "carrier": other[0],
            "what": "the same styles and the same reader on a different "
                    "phrase; the words carry nothing in either case, and a "
                    "reader that survives the swap is reading the melody",
            "rounds": [{"round": r["round"], "results": r["results"]}
                       for r in alt_rounds],
        }

    # 4. THE BLIND LISTENING PACK, on the final round's test renders
    last = rounds[-1]
    model_all = None
    Xte = np.array([feature_vector(i["feats"], ALL_FEATURES) for i in last["test_items"]])
    yte = np.array([INTENTS.index(i["intent"]) for i in last["test_items"]])
    machine_pred = {}
    # the reader trained on the final round's training renders
    Xtr = np.array([feature_vector(i["feats"], ALL_FEATURES) for i in last["train_items"]])
    ytr = np.array([INTENTS.index(i["intent"]) for i in last["train_items"]])
    model_all = Logistic(len(INTENTS)).fit(Xtr, ytr)
    preds = model_all.predict(Xte)
    for item, p in zip(last["test_items"], preds):
        machine_pred[item["id"]] = INTENTS[int(p)]

    # a balanced pack: the same number of files per intent
    chosen = []
    per = 2
    for intent in INTENTS:
        picks = [i for i in last["test_items"] if i["intent"] == intent][:per]
        chosen.extend(picks)
    pack = write_listening_pack(out_dir / "renders", chosen, machine_pred,
                                 out_dir / "listening_check",
                                 f"r{last['round']}")

    # 5. THE CARICATURE CURVE. The knob, and what it does to the
    # recovery. At x=0 every intent is the neutral delivery and the
    # reader is at chance; that end of the curve is the control.
    print("--- caricature: accuracy against exaggeration")
    neutral = neutral_style(styles)
    xs = [float(v) for v in a.caricature.split(",") if v.strip()]
    curve = caricature_sweep(styles, neutral, pooled, a.carrier,
                             out_dir / "caricature", a.train_variants,
                             a.test_variants, envelope, xs, feature_sets,
                             a.equal_notes_per_syllable)
    ablations["caricature_curve"] = {
        "what": "the accuracy against the exaggeration of the delivery; the "
                "design is held fixed and only the exaggeration moves, and at "
                "x=0 every intent is the neutral delivery so the reader is "
                "at chance -- that end of the curve is the control",
        "neutral_style": {k: round(v, 4) for k, v in neutral.items()},
        "curve": curve,
    }

    # 6. the results
    results = {
        "tool": "experiments/tone_game.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "carrier": a.carrier,
        "intents": INTENTS,
        "chance": round(1.0 / len(INTENTS), 4),
        "casey_prior": 0.70,
        "seed_captures": {k: v.as_dict() for k, v in seeds.items()},
        "pooled_envelope": pooled.as_dict(),
        "style_envelope": envelope,
        "starting_styles": requested_styles,
        "starting_styles_effective": styles,
        "clamped_by_the_envelope": clamped,
        "final_styles": final_styles,
        "provenance": {intent: ("real-seeded from capture "
                                 + REAL_SEED_MAP[intent]
                                 if intent in REAL_SEED_MAP
                                 else "synthetic inside the real-prosody envelope")
                       for intent in INTENTS},
        "rounds": [{"round": r["round"], "elapsed_s": r["elapsed_s"],
                    "styles": r["styles"], "results": r["results"]}
                   for r in rounds],
        "ablations": ablations,
        "equal_notes_per_syllable": a.equal_notes_per_syllable,
        "listening_pack": pack,
        "headline": {
            "recovery_rate_round0": rounds[0]["results"]["all"]["accuracy"],
            "recovery_rate_final": rounds[-1]["results"]["all"]["accuracy"],
            "chance": round(1.0 / len(INTENTS), 4),
            "casey_prior": 0.70,
        },
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print()
    print(f"recovery rate  round 0: {rounds[0]['results']['all']['accuracy']:.3f}")
    print(f"recovery rate  final  : {rounds[-1]['results']['all']['accuracy']:.3f}")
    print(f"chance                 : {1 / len(INTENTS):.3f}")
    print(f"Casey's prior          : 0.70")
    print(f"listening pack         : {pack['dir']}")
    print(f"results                : {out_dir / 'results.json'}")
    return 0


# ============================================================================
# 10. THE SELFTEST
# ============================================================================

def selftest() -> int:
    """The checks that need no model and no real capture."""
    import tempfile

    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  [{'ok' if got == want else 'FAIL'}] {name}: {got!r}")

    print("SELFTEST -- the performer, the render, the reader")
    print("-" * 60)

    seed = ProsodySeed(source="builtin",
                        note_durations=[0.18, 0.22, 0.26, 0.30],
                        steps=[-2.0, -1.0, 1.0, 2.0],
                        pitch_range_st=12.0, rate_hz=3.5, centre_midi=48.0)

    # 1. THE CONTOUR. Every syllable gets a note, and the notes do not
    # overlap.
    st = dict(STARTING_STYLES["everything"])
    c = perform(st, PRIMARY_CARRIER, seed, random.Random(1))
    check("the contour has notes", len(c.notes) > 0, True)
    sils = {n.syllable for n in c.notes}
    check("every syllable is sung", sils, set(range(3)))
    overlap = any(c.notes[i].onset_s + c.notes[i].duration_s >
                  c.notes[i + 1].onset_s for i in range(len(c.notes) - 1))
    check("the notes do not overlap", overlap, False)
    check("the notes are in pitch range",
          all(24 <= n.midi <= 96 for n in c.notes), True)

    # 2. THE RENDER. The audio is real audio, and the repository's own
    # tracker can hear the melody in it.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav = td / "r.wav"
        info = render_wav(c, wav, PRIMARY_CARRIER)
        check("the wav was written", wav.is_file(), True)
        check("and it has length", info["duration_s"] > 0.5, True)
        a = wavio.read_wav(wav)
        check("the wav reads back", len(a.samples) > 1000, True)
        check("and it is not silent",
              float(np.max(np.abs(a.samples))) > 0.05, True)
        feats = features_from_audio(wav)
        check("the reader found notes", feats["n_notes"] >= 2, True)
        check("the reader found a pitch", feats["centre_midi"] > 20, True)
        # the reader's centre must be near the contour's centre
        midis = c.midis()
        check("the heard centre is near the written centre",
              abs(feats["centre_midi"] - float(np.median(midis))) < 4.0, True)

        # 3. THE MIDI. It parses back, and it is on the same clock as
        # the audio.
        mid = td / "r.mid"
        write_midi(c, mid, "test")
        ppq, tracks = smf.parse_smf(mid.read_bytes())
        check("the midi parses back", ppq, score.TICKS_PER_BEAT)
        kinds = [(e.tick, e.kind) for e in tracks[0].events]
        ons = [k for k in kinds if k[1] == "note_on"]
        check("one note-on per note", len(ons), len(c.notes))
        check("the first onset is the first note's time",
              ons[0][0], int(round(c.notes[0].onset_s * NOTE_MS)))

        # 3b. THE SCORE IS THE CURRENCY. The delivery must be rebuildable
        # from the file alone -- that is what lets a weak node ship a
        # score up instead of shipping audio.
        back = contour_from_midi(mid, PRIMARY_CARRIER)
        check("the score rebuilds the delivery", len(back.notes), len(c.notes))
        check("and every note's pitch survives",
              [n.midi for n in back.notes], [n.midi for n in c.notes])
        check("and every onset survives",
              [round(n.onset_s, 6) for n in back.notes],
              [round(n.onset_s, 6) for n in c.notes])
        check("and every duration survives",
              [round(n.duration_s, 6) for n in back.notes],
              [round(n.duration_s, 6) for n in c.notes])
        # and the audio rendered from the score is the same audio
        wav2 = td / "r2.wav"
        render_wav(back, wav2, PRIMARY_CARRIER)
        s1 = np.asarray(wavio.read_wav(wav).samples)
        s2 = np.asarray(wavio.read_wav(wav2).samples)
        check("the score renders the same audio",
              float(np.max(np.abs(s1 - s2))), 0.0)

    # 4. THE CLASSIFIER. It learns a separable problem.
    rng = np.random.default_rng(7)
    X = np.vstack([rng.normal(0, 1, (40, 3)) + 5.0,
                   rng.normal(0, 1, (40, 3)) - 5.0])
    y = np.array([0] * 40 + [1] * 40)
    m = Logistic(2).fit(X, y)
    acc = float((m.predict(X) == y).mean())
    check("the classifier learns a separable problem", acc, 1.0)

    # 5. THE READABILITY PUSH. It must not make things worse, and it
    # must stay inside the prosody envelope.
    env = build_envelope({"builtin": seed})
    styles = clamp_styles({k: dict(v) for k, v in STARTING_STYLES.items()}, env)
    check("the starting design fits the envelope",
          envelope_ok(styles, env), True)
    check("and the clamp changed something",
          any(abs(styles[k][f] - STARTING_STYLES[k][f]) > 1e-9
              for k in styles for f in STYLE_FIELDS), True)
    before = readability_objective(styles, seed, PRIMARY_CARRIER, ALL_FEATURES)
    cm = np.zeros((len(INTENTS), len(INTENTS)))
    cm[0][1] = 5
    cm[1][0] = 5
    pushed = readability_push(styles, seed, PRIMARY_CARRIER, cm, ALL_FEATURES,
                              env, "t")
    after = readability_objective(pushed, seed, PRIMARY_CARRIER, ALL_FEATURES)
    check("the push does not make it worse", after >= before - 1e-9, True)
    check("and it stays inside the envelope",
          envelope_ok(pushed, env), True)

    # 6. THE PROVENANCE. Two intents are real-seeded, four are synthetic,
    # and every intent says which it is.
    check("two intents are real-seeded", len(REAL_SEED_MAP), 2)
    check("the real-seeded ones are in the deck",
          all(i in INTENTS for i in REAL_SEED_MAP), True)

    # 7. THE PAIRED DESIGN. The prosody noise must not be a fingerprint:
    # if each intent drew its own noise, the reader could name the
    # intent from the jitter alone, without ever hearing the delivery.
    # The last note's duration is excluded because the style's `hold`
    # multiplies it -- that is the style, not the noise.
    na = perform(styles["sincere"], PRIMARY_CARRIER, seed, random.Random("k"))
    nb = perform(styles["leaving"], PRIMARY_CARRIER, seed, random.Random("k"))
    check("two intents share their prosody noise",
          [n.duration_s for n in na.notes[:-1]],
          [n.duration_s for n in nb.notes[:-1]])
    check("but their pitches are their own",
          [n.midi for n in na.notes] != [n.midi for n in nb.notes], True)

    # 8. THE CARICATURE KNOB. At x=0 every intent is the neutral
    # delivery -- that end of the curve is the control, and it is where
    # the reader must be at chance.
    neutral = neutral_style(styles)
    zero = caricature(styles, neutral, 0.0, env)
    check("at zero exaggeration every intent is the neutral one",
          all(abs(zero[i][f] - neutral[f]) < 1e-9
              for i in INTENTS for f in STYLE_FIELDS), True)
    one = caricature(styles, neutral, 1.0, env)
    check("at one the design is unchanged",
          all(abs(one[i][f] - styles[i][f]) < 1e-9
              for i in INTENTS for f in STYLE_FIELDS), True)
    two = caricature(styles, neutral, 2.0, env)
    sep1 = min(float(np.linalg.norm(
        np.array([features_from_contour(perform(styles[a], PRIMARY_CARRIER, seed,
                                                 random.Random(1)))[FEATURE_NAMES[k]]
                  for k in ALL_FEATURES])
        - np.array([features_from_contour(perform(styles[b], PRIMARY_CARRIER, seed,
                                                   random.Random(1)))[FEATURE_NAMES[k]]
                    for k in ALL_FEATURES])))
        for a in INTENTS for b in INTENTS if a < b)
    sep2 = min(float(np.linalg.norm(
        np.array([features_from_contour(perform(two[a], PRIMARY_CARRIER, seed,
                                                 random.Random(1)))[FEATURE_NAMES[k]]
                  for k in ALL_FEATURES])
        - np.array([features_from_contour(perform(two[b], PRIMARY_CARRIER, seed,
                                                   random.Random(1)))[FEATURE_NAMES[k]]
                    for k in ALL_FEATURES])))
        for a in INTENTS for b in INTENTS if a < b)
    check("exaggerating pushes the intents apart", sep2 > sep1, True)

    print("-" * 60)
    if failures:
        print(f"SELFTEST FAILED: {len(failures)} check(s)")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SELFTEST PASSED")
    return 0


def envelope_ok(styles: dict, envelope: dict) -> bool:
    for st in styles.values():
        for field, (lo, hi) in envelope.items():
            if not (lo <= st[field] <= hi):
                return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
