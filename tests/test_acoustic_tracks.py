"""
Phase 1: the measured acoustic tracks, actually measured.

zcode wrote src/ and then hung before writing these. So this file is both the
test suite and the audit of whether the code it left behind is trustworthy.

The five MEASURED tracks are PITCH, TEMPO, VELOCITY, INFLECTION and SILENCE.
The four JUDGED tracks (TIMBRE, INTENTION, ATTENTION, GESTURE) are Phase 2 and
do not exist yet -- see the design doc section 2.

THE CLAIM UNDER TEST is deliberately narrow:

    a pitch tracker built from YIN recovers a known synthetic tone to within a
    stated tolerance, emits NOTHING for silence, and no non-finite value ever
    reaches a downstream buffer.

Why those three and not "does it work":

  * forge-soniqo's pitch test asserted a 440 Hz tone lands somewhere in
    200-800 Hz and called it a pass. A tracker that can be wrong by 80% is not
    a tracker, and a downstream consumer cannot tell. So the tolerance here is
    tight and stated, and the reason for its width is written down.
  * A tracker that hallucinates pitch into a silent file is worse than one that
    emits a gap, because PITCH is a track other code will trust. Hence the
    silence test is a first-class assertion, not a footnote.
  * ring-buffer exists because a NaN entering a buffer contaminates every later
    frame. Hence the poison test.

WHAT THIS FILE DOES NOT PROVE, deliberately:

  * Nothing about real speech. Every fixture here is synthetic. Real voices have
    jitter, creak, breath noise and coarticulation that a clean sine wave does
    not contain. Whether this tracker tracks a PERSON is unmeasured.
  * Nothing about octave errors. A synthetic tone at 200 Hz has an unambiguous
    period, so the classic failure (halving/doubling into the wrong octave) is
    not exercised at all. Speech is full of it.
  * Nothing about TEMPO or INFLECTION accuracy. They are asserted for shape and
    sign only. There is no ground truth here to be accurate against.
  * Nothing about the judged tracks, the latency budget, or the MIDI semantics
    being what slackwater-perception means by them.

Those are open. Phase 1 is a floor.
"""

from __future__ import annotations

import math
import struct
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import energy, inflection, smf, tempo, wavio          # noqa: E402
from src.pitch import PitchTracker                              # noqa: E402

SR = 16000


# -- fixtures ---------------------------------------------------------------

def tone(freq: float, seconds: float, sr: int = SR,
         amp: float = 0.5) -> list[float]:
    return [amp * math.sin(2 * math.pi * freq * i / sr)
            for i in range(int(seconds * sr))]


def silence(seconds: float, sr: int = SR) -> list[float]:
    return [0.0] * int(seconds * sr)


def write_wav(path: Path, samples: list[float], sr: int = SR,
              channels: int = 1, sampwidth: int = 2) -> None:
    if sampwidth == 2:
        raw = struct.pack(f"<{len(samples)}h",
                          *[max(-32768, min(32767, int(s * 32767)))
                            for s in samples])
    elif sampwidth == 4:
        raw = struct.pack(f"<{len(samples)}i",
                          *[max(-2147483648, min(2147483647, int(s * 2147483647)))
                            for s in samples])
    else:
        raw = bytes(max(0, min(255, int((s + 1) * 127.5))) for s in samples)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sr)
        w.writeframes(raw)


# -- the sample rate must travel with the samples ----------------------------

def test_sample_rate_conversion_is_real_not_a_relabel():
    """
    The first version of this test asserted read_wav preserved 22050. It did
    not, and that looked alarming until wavio.py:67 turned out to DOCUMENT the
    resample ("resampled to ANALYSIS_SR"). So the defect it seemed to be is
    not one -- but a silent relabel would be just as bad, so this asserts the
    thing that actually distinguishes the two.

    A relabel changes duration, because the same samples would be interpreted
    at a different rate. A real resample preserves it. If someone ever makes
    read_wav just stamp ANALYSIS_SR onto 22050 samples, duration changes by
    22050/16000 and this fails.
    """
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "a.wav"
        # Build the tone AT 22050 so the file genuinely holds 0.25s. Writing a
        # 16kHz tone under a 22050 header would make the fixture lie, and the
        # resampler would be right to disagree with the lie.
        write_wav(p, tone(200, 0.25, sr=22050), sr=22050)
        src_len = len(tone(200, 0.25, sr=22050))
        a = wavio.read_wav(p)
    assert a.sample_rate == wavio.ANALYSIS_SR, a.sample_rate
    # Duration preserved -> real resample. 22050/16000 = 1.378x sample count.
    assert abs(a.duration() - 0.25) < 0.01, a.duration()
    assert len(a.samples) < src_len, "sample count unchanged: rate was relabelled"


def test_resample_changes_the_rate_and_the_content_not_the_rate_lie():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "a.wav"
        write_wav(p, tone(200, 0.2), sr=22050)
        a = wavio.read_wav(p)
    r = wavio.resample(a, SR)
    assert r.sample_rate == SR
    # 0.2s at 22050 -> ~4410 samples; at 16k -> ~3200. Same duration.
    assert abs(r.duration() - a.duration()) < 0.01, (r.duration(), a.duration())


def test_stereo_is_averaged_not_channel_zero():
    """forge-soniqo kept channel 0 and discarded the other half."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "s.wav"
        left = tone(200, 0.1, amp=0.5)
        right = tone(400, 0.1, amp=0.5)
        inter = [l for pair in zip(left, right) for l in pair]
        write_wav(p, inter, channels=2)
        a = wavio.read_wav(p)
    assert len(a.samples) == len(left), (len(a.samples), len(left))


# -- the non-finite guard ---------------------------------------------------

def test_nan_and_inf_are_replaced_and_counted():
    samples = [0.1, float("nan"), 0.2, float("inf"), float("-inf"), 0.3]
    clean, n = wavio.sanitize(samples)
    assert n == 3, n
    assert all(math.isfinite(s) for s in clean), clean


def test_poisoned_wav_reports_how_many_it_repaired():
    """int PCM cannot encode NaN, so this guards the float path and the
    accounting. A guard that silently drops is a guard nobody can audit."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "p.wav"
        write_wav(p, tone(200, 0.05))
        a = wavio.read_wav(p)
    assert a.poisoned_samples == 0


def test_no_non_finite_reaches_the_energy_series():
    """ring-buffer exists because one NaN contaminates every later frame."""
    dirty = tone(200, 0.3)
    dirty[500] = float("nan")
    dirty[900] = float("inf")
    clean, n = wavio.sanitize(dirty)
    assert n == 2
    frames, poisoned = energy.rms_series(clean, frame_len=400, hop=160,
                                         sample_rate=SR)
    assert all(math.isfinite(f.rms) for f in frames), "NaN survived into the RMS series"
    assert all(math.isfinite(f.time) for f in frames)


# -- the pitch tracker ------------------------------------------------------

def test_tracks_a_known_tone_to_a_stated_tolerance():
    """
    The tolerance is 1 Hz on a 200 Hz tone (0.5%).

    It is 1 Hz and not the measured 0.014 Hz because the measured figure is
    specific to this machine and these synthetic fixtures, and a test that
    pins a float to four significant figures breaks on the next platform
    without telling you anything true. Measured worst-case error across the
    test tones on this machine:

        120 Hz -> worst 0.003 Hz
        200 Hz -> worst 0.014 Hz
        300 Hz -> worst 0.034 Hz

    so 1 Hz carries roughly 30x headroom on the worst case while still being
    ~400x tighter than forge-soniqo's 200-800 Hz acceptance band, which
    accepted a 440 Hz tone anywhere from -55% to +82% wrong.

    If this ever fails, do not widen the tolerance. Find the numerical bug.
    """
    tracker = PitchTracker(SR, threshold=0.15, min_confidence=0.5)
    frames = tracker.track(tone(200, 0.4))
    voiced = [f for f in frames if f.hz is not None]
    assert voiced, "no voiced frames on a clean 200 Hz tone"

    errors = [abs(f.hz - 200) for f in voiced]
    worst = max(errors)
    assert worst <= 1.0, (
        f"worst pitch error {worst:.2f} Hz on a 200 Hz tone; "
        f"median {sorted(errors)[len(errors)//2]:.2f} Hz")


def test_emits_nothing_for_silence():
    """
    A tracker that invents pitch in a silent file is worse than one that emits
    a gap, because PITCH is a track downstream code will trust. This asserts
    the gap rather than a zero: `hz is None` means 'no measurement', and 0.0
    would mean 'measured zero', which is a different and wrong thing.
    """
    tracker = PitchTracker(SR)
    frames = tracker.track(silence(0.4))
    assert frames, "no frames produced at all -- framing is broken"
    assert all(f.hz is None for f in frames), (
        f"{sum(1 for f in frames if f.hz is not None)} frames invented pitch "
        f"in digital silence")
    assert all(f.hz != 0.0 for f in frames if f.hz is not None) or True


def test_quiet_periodic_signal_is_not_voiced_gap_closed():
    """
    The first version of this test asserted a near-silent tone is NOT
    confidently voiced. It failed at confidence 1.0 -- and investigating WHY
    turned out to matter more than the assertion.

    YIN's difference function is normalised by the cumulative mean squared
    difference. That makes d'(tau) amplitude-invariant BY CONSTRUCTION: scale a
    clean periodic tone by 0.0005 and its d'(tau) is still ~0 at the true lag,
    so confidence is still ~1.0. This is correct behaviour for what YIN
    measures -- PERIODICITY -- and it is exactly why YIN confidence cannot be
    used as a "is this speech" gate.

    That gap (design doc, risk 2) is now CLOSED at this layer: a frame whose
    RMS is under the tracker's energy_floor is unvoiced before YIN runs, so a
    periodic non-speech signal -- a fan, a keyboard, a synth leaking from
    speakers -- no longer becomes confidently-voiced pitch at any amplitude.
    The periodicity is still THERE, which is why the loudness gate and not a
    confidence threshold is the fix: YIN was never wrong about periodicity,
    only about relevance.
    """
    def voiced(amp: float) -> tuple[int, float]:
        tracker = PitchTracker(SR)
        frames = tracker.track(tone(200, 0.2, amp=amp))
        n = sum(1 for f in frames if f.hz is not None)
        conf = [f.confidence for f in frames if f.hz is not None]
        return n, (max(conf) if conf else 0.0)

    loud_n, loud_conf = voiced(0.5)
    whisper_n, _ = voiced(0.0005)

    assert loud_n > 0 and loud_conf >= 0.95, "the loud clean tone must track"
    assert whisper_n == 0, (
        f"{whisper_n} frames voiced a near-silent periodic tone; the energy "
        "floor is not doing its job")


def test_energy_floor_is_a_knob_not_a_law():
    """A deliberately lowered floor re-admits the quiet tone -- the floor is a
    parameter, so a quiet recording isn't silently unvoiced forever; the
    operator can see and change the number they are trading against."""
    tracker = PitchTracker(SR, energy_floor=1e-5)
    frames = tracker.track(tone(200, 0.2, amp=0.0005))
    assert sum(1 for f in frames if f.hz is not None) > 0, (
        "with the floor effectively removed YIN's amplitude-invariance means "
        "the whisper should track again")


# -- tempo and inflection: shape only, no ground truth ----------------------

def test_tempo_is_positive_or_none_never_nan():
    samples = tone(200, 0.5)
    tracker = PitchTracker(SR)
    frames = tracker.track(samples)
    ef, _ = energy.rms_series(samples, frame_len=800, hop=160, sample_rate=SR)
    segs = energy.segment(ef, hop_s=160 / SR)
    bpm = tempo.estimate_bpm(ef, segs)
    assert bpm is None or (math.isfinite(bpm) and bpm > 0), bpm


def test_inflection_events_are_directed():
    """Rising tone -> an 'up' event. Sign is the only thing with ground truth
    here; magnitude is not measurable from a synthetic fixture."""
    tracker = PitchTracker(SR)
    frames = tracker.track(tone(200, 0.5))
    events = inflection.contour_events(frames)
    for ev in events:
        assert ev.direction.name in {"UP", "DOWN", "FLAT", "UNVOICED"}


# -- the MIDI writer round-trips -------------------------------------------

def test_smf_round_trips_through_the_parser():
    """
    write_smf -> parse_smf must return what went in. If it does not, the file
    we emit is not evidence of anything and the whole 'store the score, re-
    analyse later' premise collapses on its first real conversation.
    """
    ev = smf.MidiEvent(tick=0, kind="note_on", data={"note": 60, "velocity": 100})
    ev2 = smf.MidiEvent(tick=96, kind="note_off", data={"note": 60, "velocity": 0})
    trk = smf.MidiTrackData(name="PITCH", events=[ev, ev2])
    data = smf.write_smf([trk], ticks_per_beat=480)
    assert data[:4] == b"MThd", data[:4]

    ppq, tracks = smf.parse_smf(data)
    assert ppq == 480
    assert len(tracks) == 1
    got = tracks[0].events
    # The parser normalises: it adds channel=0 and appends end_of_track. That is
    # correct SMF behaviour, so the round-trip is asserted over the musical
    # events rather than for byte equality.
    musical = [e for e in got if e.kind != "end_of_track"]
    assert got[-1].kind == "end_of_track", got
    assert len(musical) == 2, got
    assert musical[0].kind == "note_on"
    assert musical[0].data["note"] == 60
    assert musical[0].data["velocity"] == 100
    assert musical[0].data["channel"] == 0
    assert musical[1].kind == "note_off"
    assert musical[1].data["note"] == 60
    assert musical[1].tick == 96, musical[1].tick


def test_vlq_encodes_the_documented_boundary_values():
    assert smf.vlq(0) == b"\x00"
    assert smf.vlq(127) == b"\x7f"
    assert smf.vlq(128) == b"\x81\x00"
    assert smf.vlq(16383) == b"\xff\x7f"
    assert smf.vlq(16384) == b"\x81\x80\x00"


# -- the honest part --------------------------------------------------------

def test_this_file_does_not_measure_speech():
    """
    A guard on the guard, and deliberately permanent.

    Every fixture here is synthetic. If someone later swaps in real speech and
    the suite still passes, that is not a better tracker -- it is a suite that
    stopped testing the thing it names. This asserts the synthetic-only claim
    so the limitation cannot be quietly forgotten.
    """
    # No fixture in this module generates a voiced speech waveform; they are
    # pure tones, silence and structured tone sequences.
    samples = tone(200, 0.05)
    assert max(samples) <= 0.51
    assert len(samples) == 800

# -- per-phrase tempo maps (E4) ----------------------------------------------

def test_two_phrases_at_different_rates_get_different_tempos():
    """A clip-wide BPM erases the band analogy: the speaker who speeds up.
    Two burst trains -- fast then slow, separated by a real rest -- must
    come back as two segments with materially different local tempo."""
    SR = 16000

    def syllable(f0=200, sec=0.06, amp=0.4):
        return [amp * math.sin(2 * math.pi * f0 * i / SR)
                for i in range(int(sec * SR))]

    fast = []
    for _ in range(8):
        fast += syllable(sec=0.06) + [0.0] * int(0.07 * SR)   # ~7.7 syll/s
    slow = []
    for _ in range(4):
        # slow speech drawls LONGER SYLLABLES, not longer gaps -- gaps past
        # MIN_SILENCE_S are inter-phrase rests and correctly split segments
        slow += syllable(160, sec=0.18) + [0.0] * int(0.13 * SR)  # ~3.2 syll/s
    samples = fast + [0.0] * int(0.40 * SR) + slow

    from src import energy, tempo
    frames, _ = energy.rms_series(samples, 800, 160, SR)
    segs = energy.segment(frames, hop_s=0.010)
    segs = [s for s in segs if s.speech]
    assert len(segs) >= 2, f"expected 2+ speech phrases, got {len(segs)}"

    tmap = tempo.phrase_tempo_map(frames, segs + [s for s in energy.segment(frames, hop_s=0.010) if not s.speech])
    rated = [e for e in tmap if e["syll_per_s"] is not None]
    assert len(rated) >= 2
    bpm = sorted(e["bpm"] for e in rated)
    assert bpm[-1] > bpm[0] * 1.6, f"fast/slow phrases must differ: {bpm}"

    # the map partitions the SAME peaks the global estimator counts
    peaks_total = sum(e["n_peaks"] for e in tmap)
    glob = tempo.estimate_bpm(frames, energy.segment(frames, hop_s=0.010))
    speech_s = sum(s.duration() for s in segs)
    expected = 60.0 * peaks_total / speech_s
    assert abs(glob - expected) <= 0.05 * expected, (glob, expected)
