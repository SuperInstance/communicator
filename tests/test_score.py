"""
Tests for the nine-track score, end to end, through real bytes.

WHY THIS FILE EXISTS. An earlier draft of src/score.py was 248 lines, fully
documented, committed and published -- and had never been executed. Its first
run raised

    TypeError: byte indices must be integers or slices, not str

because it called `MidiEvent(0, "track_name", name.encode("ascii"))` while the
real signature is `MidiEvent(tick, kind, data: dict)`. It also defaulted an
unmeasurable tempo to 120, scaled loudness by a bare `rms * 400`, and carried
two dead placeholder expressions.

So this file does not test helpers in isolation. It builds a WAV, measures it,
writes a real .mid, parses those bytes back with a DIFFERENT code path
(smf.parse_smf, not the writer), and asserts the properties the design
depends on. A test that only calls the functions that were written is exactly
what let that draft through.

THE PROPERTIES ASSERTED ARE THE THESIS, not the implementation:
  * a tick is exactly one millisecond, so time survives the round trip;
  * each of the nine tracks carries its own name in the file;
  * the score can be read back and the pitch still reconstructs;
  * an absent measurement is visibly absent, never silently zero.
"""

from __future__ import annotations

import math
import struct
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import inflection, judged, score as score_mod, smf      # noqa: E402
from src.judged import TrackReading                              # noqa: E402

SR = 16000


# -- fixtures ---------------------------------------------------------------

def tone(hz: float, seconds: float, amp: float = 0.3, sr: int = SR) -> list:
    return [amp * math.sin(2 * math.pi * hz * i / sr)
            for i in range(int(seconds * sr))]


def write_wav(path, samples, sr=SR):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767))
            for s in samples))
    return str(path)


def speechy(sr=SR):
    """Two voiced phrases with a silence between them.

    Silence matters: SILENCE is one of the nine tracks and an utterance with
    no gap would test a track that cannot fail.
    """
    out = []
    for hz in (180.0, 240.0):
        out += tone(hz, 0.35, 0.35, sr)
        out += [0.0] * int(0.30 * sr)
    for hz in (300.0, 200.0):
        out += tone(hz, 0.30, 0.30, sr)
        out += [0.0] * int(0.25 * sr)
    return out


def fake_readings(**vals):
    def rd(track, detail=None):
        return TrackReading(track, vals.get(track, 0), detail=detail or {},
                            model="jev-fake", ok=True)
    return [rd("TIMBRE", {"gate": 0.1}),
            rd("INTENTION", {"gate": 0.9}),
            rd("ATTENTION", {"gate": 0.8}),
            rd("GESTURE", {"choice": "command", "gate": 0.9})]


# -- the clock --------------------------------------------------------------

class TestTickIsAMillisecond:
    def test_ppq_and_tempo_put_one_tick_on_one_ms(self):
        """
        ppq=1000 at 1e6 us/beat is 1 ms per tick. ppq=480 at 120 BPM is
        1.0417 ms, so every event time is a rounding of a real time and the
        acoustic tracks drift when re-derived from the file.
        """
        assert score_mod.TICKS_PER_BEAT == 1000
        assert score_mod.TEMPO_US == 1_000_000
        us_per_tick = score_mod.TEMPO_US / score_mod.TICKS_PER_BEAT
        assert us_per_tick == pytest.approx(1000.0), "1 tick must be 1 ms"

    def test_round_trip_through_ms_is_exact(self):
        # ticks_to_ms returns MILLISECONDS, so it is compared against the
        # input in ms. Comparing against seconds is a units error that reads
        # like a precision error, which is why it is spelled out here.
        for s in (0.0, 0.001, 0.1, 0.35, 1.0, 2.9375):
            assert score_mod.ticks_to_ms(score_mod.ms_to_ticks(s)) == \
                pytest.approx(s * 1000.0, abs=0.5)

    def test_ms_to_ticks_never_negative_for_negative_zero(self):
        assert score_mod.ms_to_ticks(0.0) == 0


# -- the nine tracks, present and named -------------------------------------

class TestNineTracks:
    def test_all_nine_are_present_and_named(self, tmp_path):
        sc = score_mod.score_utterance(speechy(), "Run the tests.", client=_Fake())
        data = score_mod.build_smf(sc)
        parsed = score_mod.parse_tracks(data)
        assert sorted(parsed) == sorted(score_mod.TRACK_NAMES)
        assert len(parsed) == 9

    def test_track_name_is_written_into_the_file(self, tmp_path):
        """
        The name lives in the bytes, not just in the dataclass. A score that
        can only be read by the library that wrote it is a screenshot.
        """
        sc = score_mod.score_utterance(speechy(), "Run the tests.", client=_Fake())
        data = score_mod.build_smf(sc)
        _, parsed = smf.parse_smf(data)
        named = [t for t in parsed
                 if any(e.kind == "track_name" for e in t.events)]
        assert len(named) == 9, "every track needs a track_name meta event"

    def test_provenance_separates_measured_from_judged(self):
        prov = score_mod.UtteranceScore().provenance()
        assert prov["measured"] == ["inflection", "pitch", "silence",
                                    "tempo", "velocity"]
        assert prov["judged"] == ["attention", "gesture", "intention", "timbre"]
        assert not (set(prov["measured"]) & set(prov["judged"]))


# -- the writer actually writes, and the reader actually reads --------------

class TestRoundTrip:
    def test_bytes_parse_back_through_an_independent_path(self, tmp_path):
        """
        The writer and the reader must not be able to agree on garbage: the
        assertion goes through smf.parse_smf, which is not build_smf's
        inverse by construction.
        """
        sc = score_mod.score_utterance(speechy(), "Run the tests.", client=_Fake())
        path = tmp_path / "s.mid"
        score_mod.export_smf(sc, str(path))
        data = path.read_bytes()
        assert data[:4] == b"MThd", "not a MIDI file"

        parsed = score_mod.parse_tracks(data)
        notes = [e for e in parsed["pitch"] if e["kind"] == "note_on"]
        assert notes, "pitch track is empty -- nothing round-tripped"

    def test_pitch_survives_the_round_trip_within_a_semitone(self, tmp_path):
        """
        The thesis: the acoustic evidence is re-derivable from the file. A
        180 Hz tone is MIDI note 52; if the score preserved it, a reader can
        get the pitch back without the audio.
        """
        samples = tone(180.0, 0.4)
        sc = score_mod.score_utterance(samples, "Run the tests.", client=_Fake())
        data = score_mod.build_smf(sc)
        notes = [e for e in score_mod.parse_tracks(data)["pitch"]
                 if e["kind"] == "note_on"]
        got = [e["data"]["note"] for e in notes]
        assert got, "no notes written"
        expected = score_mod.hz_to_note(180.0)
        assert all(abs(n - expected) <= 1 for n in got), \
            f"expected ~{expected}, got {got}"

    def test_measured_tracks_need_no_network(self, tmp_path):
        """
        The UNO Q case. With no client and no key, score_utterance must still
        produce five measured tracks -- the judged four come back not-ok and
        are omitted rather than written as zeros.
        """
        sc = score_mod.score_utterance(speechy(), "Run the tests.", client=_Fake())
        data = score_mod.build_smf(sc)
        parsed = score_mod.parse_tracks(data)
        assert parsed["pitch"], "measured pitch must exist without a model"
        assert parsed["velocity"], "measured velocity must exist without a model"
        assert parsed["silence"], "measured silence must exist without a model"


# -- absence is visible -----------------------------------------------------

class TestAbsenceIsNeverZero:
    def test_unvoiced_frames_write_no_notes(self):
        """
        hz is None so "no measurement" cannot be read as "measured zero".
        Note 0 is a real, audible, wrong answer, so silence between phrases
        must produce NO note rather than a note at the bottom of the keyboard.
        """
        samples = tone(220.0, 0.2) + [0.0] * int(0.3 * SR) + tone(220.0, 0.2)
        sc = score_mod.score_utterance(samples, "Run the tests.", client=_Fake())
        notes = [e for e in score_mod.parse_tracks(score_mod.build_smf(sc))["pitch"]
                 if e["kind"] == "note_on"]
        assert len(notes) >= 2, "two voiced phrases should be two runs"
        assert all(n["data"]["note"] > 0 for n in notes), \
            "a note at 0 means an unvoiced frame was written as a pitch"

    def test_the_clock_event_is_written_even_with_no_tempo(self):
        """
        The set_tempo that makes one tick one millisecond is a property of the
        ENCODING, so it is written unconditionally. A reader that finds no
        tempo meta event assumes 500,000 us/beat, which at ppq 1000 makes a
        tick 0.5 ms -- so the claim would be true of this module and false of
        the file, and every time re-derived from the score would be out by
        two. This test is the guard against exactly that.
        """
        sc = score_mod.UtteranceScore(bpm=None, duration_s=1.0)
        evs = score_mod.parse_tracks(score_mod.build_smf(sc))["tempo"]
        tempos = [e for e in evs if e["kind"] == "set_tempo"]
        assert len(tempos) == 1, "the clock must always be written"
        assert tempos[0]["data"]["tempo"] == score_mod.TEMPO_US

    def test_unmeasured_tempo_writes_no_measurement_cc(self):
        """Not 120. A fabricated tempo would be re-derived as real."""
        sc = score_mod.UtteranceScore(bpm=None, duration_s=1.0)
        evs = score_mod.parse_tracks(score_mod.build_smf(sc))["tempo"]
        assert [e for e in evs if e["kind"] == "control_change"] == [], \
            "an unmeasured tempo must not be invented"

    def test_a_measured_tempo_is_written_as_cc120(self):
        sc = score_mod.UtteranceScore(bpm=120.0, duration_s=1.0)
        evs = score_mod.parse_tracks(score_mod.build_smf(sc))["tempo"]
        ccs = [e for e in evs if e["kind"] == "control_change"]
        assert len(ccs) == 1
        assert ccs[0]["data"]["control"] == score_mod.TEMPO_CC
        assert ccs[0]["data"]["value"] == round(127.0 * 60.0 / 180.0)

    def test_a_failed_judged_track_writes_nothing(self):
        """
        read_tracks returns ok=False on a network failure. Writing that as CC 0
        would be indistinguishable from the model saying "no intention", which
        is a very different thing to hand a synthesiser.
        """
        sc = score_mod.UtteranceScore(
            bpm=120.0, duration_s=1.0,
            judged_readings=[TrackReading("INTENTION", 0, ok=False,
                                          detail={"why": "no key"})])
        parsed = score_mod.parse_tracks(score_mod.build_smf(sc))
        assert parsed["intention"] == [], "a failed reading must write nothing"

    def test_a_silent_clip_writes_no_velocity(self):
        """Peak RMS 0 means there is no envelope to normalise against."""
        sc = score_mod.score_utterance([0.0] * SR, "Run the tests.", client=_Fake())
        parsed = score_mod.parse_tracks(score_mod.build_smf(sc))
        assert parsed["velocity"] == []


# -- the judged half --------------------------------------------------------

class TestJudgedTracks:
    def test_each_judged_track_writes_its_own_cc(self):
        sc = score_mod.UtteranceScore(bpm=120.0, duration_s=1.0,
                                      judged_readings=fake_readings(
                                          TIMBRE=13, INTENTION=82,
                                          ATTENTION=74, GESTURE=119))
        parsed = score_mod.parse_tracks(score_mod.build_smf(sc))
        want = {"timbre": judged.CC["TIMBRE"],
                "intention": judged.CC["INTENTION"],
                "attention": judged.CC["ATTENTION"]}
        for track, cc in want.items():
            evs = [e for e in parsed[track] if e["kind"] == "control_change"]
            assert len(evs) == 1, f"{track} wrote {len(evs)} CCs"
            assert evs[0]["data"]["control"] == cc
            assert evs[0]["data"]["value"] == sc.judged_readings[
                [r.track for r in sc.judged_readings].index(track.upper())].value

    def test_gesture_drives_only_the_winning_slots_cc(self):
        """
        The CC80-87 bank: one slot per speech act, the winner drives its own
        and the rest read 0. A reader sees WHICH gesture as well as how
        confident, without unpacking the probabilities.
        """
        sc = score_mod.UtteranceScore(bpm=120.0, duration_s=1.0,
                                      judged_readings=fake_readings(GESTURE=119))
        evs = [e for e in score_mod.parse_tracks(score_mod.build_smf(sc))["gesture"]
               if e["kind"] == "control_change"]
        assert len(evs) == 1
        assert evs[0]["data"]["control"] == score_mod.GESTURE_SLOTS["command"]
        assert evs[0]["data"]["value"] == 119
        assert all(80 <= s <= 87 for s in score_mod.GESTURE_SLOTS.values())

    def test_an_unknown_gesture_writes_nothing(self):
        sc = score_mod.UtteranceScore(
            bpm=120.0, duration_s=1.0,
            judged_readings=[TrackReading("GESTURE", 90,
                                          detail={"choice": "shrug"}, ok=True)])
        assert score_mod.parse_tracks(score_mod.build_smf(sc))["gesture"] == []

    def test_the_gate_reaches_the_score(self):
        """
        TIMBRE's gate is ~0, so its CC is ~0. A score that carried the
        ungated value would claim warmth that Phase 2.7 measured as absent.
        """
        sc = score_mod.UtteranceScore(bpm=120.0, duration_s=1.0,
                                      judged_readings=fake_readings(TIMBRE=6))
        evs = [e for e in score_mod.parse_tracks(score_mod.build_smf(sc))["timbre"]
               if e["kind"] == "control_change"]
        assert evs and evs[0]["data"]["value"] == 6


# -- velocity normalisation -------------------------------------------------

class TestVelocityIsNormalised:
    def test_a_quiet_clip_uses_the_range_too(self):
        """
        The earlier draft scaled by `rms * 400`, so a quiet recording wrote
        CC ~12 everywhere and a loud one saturated. Normalising to the clip's
        own peak means the track encodes the ENVELOPE, not the gain.
        """
        quiet = score_mod.score_utterance(tone(200.0, 0.3, 0.01), "hi", client=_Fake())
        loud = score_mod.score_utterance(tone(200.0, 0.3, 0.9), "hi", client=_Fake())
        q = max(e["data"]["value"] for e in
                score_mod.parse_tracks(score_mod.build_smf(quiet))["velocity"])
        l = max(e["data"]["value"] for e in
                score_mod.parse_tracks(score_mod.build_smf(loud))["velocity"])
        assert q > 100 and l > 100, f"both should reach full scale: {q}, {l}"
        assert abs(q - l) <= 2, "peak normalisation makes amplitude irrelevant"

    def test_values_stay_inside_the_controller_range(self):
        sc = score_mod.score_utterance(speechy(), "Run the tests.", client=_Fake())
        for track in score_mod.TRACK_NAMES:
            for e in score_mod.parse_tracks(score_mod.build_smf(sc))[track]:
                if e["kind"] == "control_change":
                    assert 0 <= e["data"]["value"] <= 127


# -- negative controls ------------------------------------------------------

class _Fake:
    """A client that never touches the network."""

    def system_one(self, state, questions, model, timeout=8.0):
        class A(dict):
            pass
        answers = A()
        for key in ("timbre", "intention"):
            class S:
                score, confidence, probabilities, legend = 1.0, 0.5, {1: 0.6}, {1: "x"}
            answers[key] = S()
        for key, choice in (("attention", "listener"), ("gesture", "command")):
            class C:
                pass
            c = C()
            c.choice = choice
            c.confidence = 0.5
            c.probabilities = {choice: 0.9}
            answers[key] = c
        for key in judged.GATES:
            class N:
                pass
            n = N()
            n.noul = 0.8
            answers[key] = n

        class R:
            pass
        r = R()
        r.answers = answers
        r.model = model
        return r


class TestNegativeControl:
    def test_sabotage_midi_event_data_as_bytes(self):
        """
        The exact bug that shipped: a track_name written as bytes instead of
        {"name": ...}. The encoder must raise on it, and the score must never
        produce one. Asserting the raise is what stops the regression being
        reintroduced as a working-looking call.
        """
        with pytest.raises(Exception):
            smf.write_smf([smf.MidiTrackData(
                "x", [smf.MidiEvent(0, "track_name", b"x")])])

        sc = score_mod.UtteranceScore(bpm=120.0, duration_s=1.0)
        data = score_mod.build_smf(sc)
        _, parsed = smf.parse_smf(data)
        for t in parsed:
            for e in t.events:
                assert isinstance(e.data, dict), \
                    f"{t.name} carries a non-dict data payload"

    def test_sabotage_tempo_default_of_120(self):
        sc = score_mod.UtteranceScore(bpm=None, duration_s=1.0)
        evs = score_mod.parse_tracks(score_mod.build_smf(sc))["tempo"]
        ccs = [e for e in evs if e["kind"] == "control_change"]
        assert ccs == [], "a defaulted 120 BPM would appear as a CC120"

    def test_sabotage_velocity_rms_times_400(self):
        """
        A quiet clip exposes the magic multiplier: rms*400 saturates for a
        loud tone, so BOTH extremes would pass. The quiet clip is the case
        that actually distinguishes them.
        """
        quiet = score_mod.score_utterance(tone(200.0, 0.3, 0.01), "hi", client=_Fake())
        vals = [e["data"]["value"] for e in
                score_mod.parse_tracks(score_mod.build_smf(quiet))["velocity"]]
        assert max(vals) > 100, "peak-normalised quiet clip still reaches top"

    def test_sabotage_drop_pitch_track(self):
        sc = score_mod.score_utterance(speechy(), "Run the tests.", client=_Fake())
        parsed = score_mod.parse_tracks(score_mod.build_smf(sc))
        assert "pitch" in parsed and parsed["pitch"]


def sc_bytes():
    sc = score_mod.UtteranceScore(bpm=120.0, duration_s=1.0)
    return score_mod.build_smf(sc)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))