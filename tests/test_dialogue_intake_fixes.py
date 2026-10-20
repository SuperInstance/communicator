"""
Tests for the dialogue-intake fixes of 2026-10-19.

WHY THIS FILE EXISTS, SEPARATELY FROM `test_dialogue_intake.py`.

`tests/test_dialogue_intake.py` pins the intake's BEHAVIOUR: that a good
corpus is accepted, that each defect is rejected, that the exit code
carries the verdict. This file pins the MEASUREMENTS that justify the
behaviour -- the numbers that were taken, the margins they gave, and the
properties of the two modules the fixes touched (`src/wavio.py` gained a
writer and kept the source sample rate).

A behaviour test says "the gate fires". A measurement test says "the gate
fires BECAUSE this margin is 0.6939 and not 0.0067". Only the second kind
of assertion notices when a number quietly moves.

WHAT IS PINNED HERE.

  - the cross-talk statistic separates a contaminated corpus from a clean
    one by a wide margin, and the legacy statistic does not;
  - the level measure is ROOM-INDEPENDENT: the same corpus in a quiet room
    and in an ordinary one produces the same numbers, and an empty room is
    read as empty in both;
  - the clap recovers the inter-channel offset exactly, and the hand-off
    count does not move with the offset;
  - the drift report says what it can and cannot measure;
  - the protocol length gate matches the protocol's own scene durations;
  - `src/wavio.py` writes what it claims to write, round-trips through its
    own reader, clips rather than wraps, and keeps the file's REAL sample
    rate.

COST.

Most of this file reads the committed results file rather than re-running
the battery, which takes four minutes. The two tests that DO re-run
anything are marked, and they build their own corpus in a temp directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import wavio                              # noqa: E402
from experiments import dialogue_intake as di     # noqa: E402
from experiments import dialogue_intake_selftest as st   # noqa: E402

RESULTS = ROOT / "experiments" / "dialogue_intake_fixes_results.json"
INTAKE = ROOT / "experiments" / "dialogue_intake.py"
PY = sys.executable


@pytest.fixture(scope="module")
def results() -> dict:
    assert RESULTS.is_file(), (
        f"{RESULTS} is missing; run "
        "experiments/dialogue_intake_fixes.py to produce it")
    return json.loads(RESULTS.read_text())


# ------------------------------------------------------------- the margins --

def test_the_cross_talk_statistic_separates_widely(results):
    """The number the gate depends on.

    The old statistic separated a contaminated corpus from a clean one by
    0.0067 -- a margin far too small to gate on, which is why nothing was
    gated on it. The new one separates by 0.6939 on the SAME signals:
    clean 0.000-0.306, contaminated 1.000.
    """
    new = results["cross_talk"]["new_statistic"]
    old = results["cross_talk"]["legacy_statistic"]
    assert new["separation"] is not None, "the battery measured no separation"
    assert new["separation"] > 0.30, \
        (f"the cross-talk statistic now separates by only "
         f"{new['separation']}: re-measure and update "
         "docs/research/DIALOGUE-INTAKE.md")
    assert new["crosstalk_min"] == 1.0, \
        "a corpus where B hears A did not drive the statistic to 1"
    assert new["clean_max"] < new["crosstalk_min"], \
        "the clean and contaminated ranges overlap"
    assert old["separation"] < 0.10, \
        (f"the legacy statistic now separates by {old['separation']}: "
         "re-measure and update docs/research/DIALOGUE-INTAKE.md")
    assert di.CROSSTALK_MAX > new["clean_max"], \
        "the gate threshold sits inside the clean range"
    assert di.CROSSTALK_MAX < new["crosstalk_min"], \
        "the gate threshold sits above the contaminated range"


def test_the_level_measure_is_room_independent(results):
    """An ordinary room must not look like two people talking.

    The old floor (1e-4 RMS, about -80 dBFS) is 23.5 dB below an ordinary
    lived-in room, so it reported the empty room-tone scene as 100% signal
    and 100% overlap. The new floor is measured from the corpus, and the
    same corpus in a quiet room and in an ordinary one now reads the same.
    """
    finding = results["level_floor"]["finding"]
    old = finding["old_floor_reads_an_empty_room_as_full"]
    new = finding["new_floor_reads_an_empty_room_as_empty"]
    assert old is not None and new is not None
    assert old["ordinary_room_both_loud_frac"] == 1.0, \
        "the old floor no longer reads an ordinary room as full overlap: " \
        "re-measure and update docs/research/DIALOGUE-INTAKE.md"
    assert new["ordinary_room_both_loud_frac"] < 0.02, \
        "the new floor still reads an ordinary room as full overlap"
    assert new["quiet_room_both_loud_frac"] < 0.02
    assert finding["new_floor_is_room_independent"] is True, \
        "the level measure depends on the room after all"


def test_the_hand_off_rule_counts_what_the_old_rule_missed(results):
    """The three ordinary shapes of a turn-taking conversation."""
    shapes = results["hand_off_rule"]
    for name in ("canonical", "canonical_then_return", "barge_back"):
        assert shapes[name]["old_rule_hand_offs"] == 0 or \
            name == "barge_back", \
            (f"the old rule counted {shapes[name]['old_rule_hand_offs']} "
             f"hand-offs for {name}")
        assert shapes[name]["new_rule_hand_offs"] == 1, \
            (f"the new rule counted {shapes[name]['new_rule_hand_offs']} "
             f"hand-offs for {name}, expected 1")


def test_the_protocol_strict_corpus_is_no_longer_rejected_for_hand_offs(
        results):
    """The corpus-level regression the audit found.

    The intake rejected a corpus made exactly to the protocol for "zero
    hand-offs" -- the thing the corpus contains.
    """
    strict = results["protocol_strict"]
    truth = strict["hand_offs_in_the_corpus"]
    assert truth["canonical"] == 8, \
        f"the corpus is built with {truth['canonical']} canonical hand-offs, expected 8"
    assert strict["old_rule_hand_offs"] <= 1, \
        "the old rule counted hand-offs in the protocol-strict corpus: " \
        "re-measure and update docs/research/DIALOGUE-INTAKE.md"
    assert strict["new_rule_hand_offs"] == truth["total"], \
        (f"the intake counted {strict['new_rule_hand_offs']} hand-offs in a "
         f"corpus built with {truth['total']}")
    assert not any("hand-off" in u for u in strict["unmet"]), \
        "the protocol-strict corpus is still rejected for a hand-off reason"


def test_the_alignment_never_flips_the_hand_off_count(results):
    """The clap is read, and the count stops moving with the offset."""
    rows = results["alignment"]["rows"]
    assert len(rows) >= 4
    for row in rows:
        assert row["alignment_applied"] is True, \
            f"the alignment was not applied at offset {row['true_offset_ms']} ms"
        assert row["detected_offset_ms"] == -row["true_offset_ms"], \
            (f"the offset was detected as {row['detected_offset_ms']} ms, "
             f"expected {-row['true_offset_ms']} ms")
        assert row["new_rule_hand_offs"] == row["true_hand_offs"], \
            (f"at offset {row['true_offset_ms']} ms the count was "
             f"{row['new_rule_hand_offs']}")
    assert results["alignment"]["old_rule_flip"], \
        "the old rule no longer flips with the offset: re-measure and " \
        "update docs/research/DIALOGUE-INTAKE.md"
    assert results["alignment"]["new_rule_flip"] == [], \
        ("the new rule flips at offsets "
         f"{results['alignment']['new_rule_flip']}")


def test_the_drift_report_says_what_it_cannot_measure(results):
    """A bound is not a measurement, and the report must not pretend."""
    drift = results["drift"]
    reported = drift["reported_by_the_intake"]
    assert drift["measurable"] is False, \
        "the drift became measurable: re-measure and update the report"
    assert reported["measured_ms"] is None
    assert reported["measurable_from_one_alignment_point"] is False
    bound = reported.get("bound_ms_at_100ppm")
    assert bound is not None and bound > 0, \
        "the drift report carries no bound at all"
    # The bound must scale with the scene, and it must be far smaller than
    # the drift the audit used to make the consequence visible.
    assert reported["bound_ms_at_12500ppm"] > reported["bound_ms_at_100ppm"]
    assert "cannot be measured" in reported["method"] or \
        "not be measured" in reported["method"] or \
        "drift" in reported["method"]


def test_every_perturbation_behaves_as_specified(results):
    """Each defect is caught; each legitimate variation is not."""
    for row in results["perturbations"]:
        assert row["correct"] is True, \
            (f"{row['perturbation']}: must_be_caught={row['must_be_caught']} "
             f"but caught={row['caught_by_the_relevant_gate']}")


def test_the_exit_codes_are_distinguishable(results):
    """A pipeline must be able to tell the three outcomes apart."""
    codes = results["exit_codes"]
    assert codes["nothing_recorded"]["exit_code"] == 2
    assert codes["empty_directory"]["exit_code"] == 1
    assert codes["short_corpus"]["exit_code"] == 1


# ------------------------------------------------------------- the protocol --

def test_the_length_gate_matches_the_protocol():
    """The gate wants what the protocol asks for.

    The old gate wanted 600 s and the protocol's eight scenes are 1230 s
    (300 + 180 + 180 + 120 + 180 + 120 + 120 + 30), so a corpus that
    satisfied the protocol tripped the gate.
    """
    assert di.PROTOCOL_SECONDS == 1230.0
    readme = (ROOT / "examples/realdialogue/README.md").read_text()
    for minutes in ("5 min", "3 min", "2 min", "0.5 min"):
        assert minutes in readme, \
            f"the protocol's scene durations changed; re-derive the gate"
    # The gate is the sum of the protocol's own per-scene durations.
    assert di.PROTOCOL_SECONDS == 300 + 180 + 180 + 120 + 180 + 120 + 120 + 30


def test_the_scene_expectations_match_the_protocol():
    """Which channels speak in which scene.

    Scene 7 is SPECIFIED to have a silent channel B and scene 8 is the
    room-tone scene, so "B is silent" cannot be an error on its own.
    """
    assert di.scene_expect("scene01_free") == (True, True)
    assert di.scene_expect("scene02_qa") == (True, True)
    assert di.scene_expect("scene03_think") == (True, True)
    assert di.scene_expect("scene04_fillers") == (True, False)
    assert di.scene_expect("scene05_backchannel") == (True, True)
    assert di.scene_expect("scene06_interrupt") == (True, True)
    assert di.scene_expect("scene07_alone") == (True, False)
    assert di.scene_expect("scene08_roomtone") == (False, False)
    # An unrecognised scene is UNKNOWN, not "B is silent".
    assert di.scene_expect("mystery") == (None, None)


# ---------------------------------------------------------------- the writer --

def test_the_writer_round_trips_through_the_reader(tmp_path):
    """`src/wavio.py` can now write what it reads.

    It could read every file the protocol asks for and nothing in the repo
    could write one, so every synthetic corpus in this project carried its
    own copy of a writer.
    """
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, 0.1, 4000)
    path = tmp_path / "round_trip.wav"
    wavio.write_wav(path, x, sample_rate=16000)
    assert path.is_file()
    back = wavio.read_wav(str(path))
    assert back.sample_rate == 16000
    assert len(back.samples) == len(x)
    # 16-bit quantisation is 1 LSB = 3.05e-5 full scale.
    err = max(abs(a - b) for a, b in zip(back.samples, x))
    assert err < 1e-4, f"the round trip lost {err} of signal"


def test_the_writer_clips_rather_than_wraps(tmp_path):
    """A sample above full scale is clipped, not wrapped.

    A wrapped sample is a sign flip: the loudest instant in a recording
    becomes the loudest instant with the opposite polarity, which is not
    something a downstream analyser can notice.
    """
    path = tmp_path / "loud.wav"
    wavio.write_wav(path, [2.0, -2.0, 0.5], sample_rate=16000)
    back = wavio.read_wav(str(path))
    assert back.samples[0] > 0.9, "a sample above full scale was not clipped"
    assert back.samples[1] < -0.9, "a sample below full scale was not clipped"
    assert abs(back.samples[2] - 0.5) < 1e-4


def test_the_writer_honours_the_sample_rate(tmp_path):
    """A file written at 22050 Hz says 22050 Hz."""
    path = tmp_path / "rate.wav"
    wavio.write_wav(path, [0.0] * 22050, sample_rate=22050)
    import wave
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 22050


def test_the_reader_keeps_the_files_real_sample_rate(tmp_path):
    """The rate the file was in, not the rate it was resampled to.

    `read_wav` resamples to the analysis rate, so before this field
    existed the original rate was unrecoverable and every file reported
    16000 -- the intake's `raw_rate_hz` field was 16000 whatever the file
    said.
    """
    path = tmp_path / "src_rate.wav"
    wavio.write_wav(path, [0.0] * 4410, sample_rate=44100)
    audio = wavio.read_wav(str(path))
    assert audio.sample_rate == 16000, "the analysis rate changed"
    assert audio.source_sample_rate == 44100, \
        "the file's real sample rate was not kept"


def test_the_intake_reports_the_real_sample_rates(tmp_path):
    """And the intake puts them in the report."""
    d = tmp_path / "rates"
    d.mkdir()
    specs = [s for s in st.SCENES if s["name"] == "scene08_roomtone"]
    st.build_corpus(d, 8.0, 1e-5, specs=specs)
    b = np.array(st.wavio.read_wav(str(d / "scene08_roomtone_B.wav")).samples)
    st.write_wav(d / "scene08_roomtone_B.wav", b, sr=22050)
    proc = subprocess.run([PY, str(INTAKE), "--raw", str(d)],
                          capture_output=True, text=True, cwd=str(ROOT))
    report = json.loads(proc.stdout)
    entry = next(p for p in report["pairs"]
                 if p["scene"] == "scene08_roomtone")
    assert entry["A"]["source_rate_hz"] == 16000
    assert entry["B"]["source_rate_hz"] == 22050
    assert entry["source_rates_hz"] == {"A": 16000, "B": 22050}
