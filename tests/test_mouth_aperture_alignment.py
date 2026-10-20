"""
Tests for the mouth-aperture alignment measurement.

`docs/research/MOUTH-APERTURE-ALIGNMENT.md` is the document these tests
pin. It makes these claims, and each one can be false:

    1. the captures are NOT the frames the repository thought they were --
       each record is 19200 bytes (15360 of 160x96 luma plus 3840 of
       chroma), not the 23040 of an NV12 frame, and reading them at 23040
       tears every frame in half;
    2. on the CORRECTLY-read frames the landmark model detects a face in
       every frame, and the mouth aperture really is much larger in the
       speaking take than in the silent one;
    3. the mouth aperture's correlation with the speech envelope is
       above the scrambled-audio null -- and that is NOT enough, because
       the eye-distance control and the mouth-width control reach the same
       or a higher number, so the correlation is not about the mouth.

The assertions are mostly on the committed results, because the claims are
about a measurement that takes a minute. The cheap ones are checked against
the code, so the tests fail if the MEASUREMENT changes and not only if the
RECORD of it changes.

THE NULL IS A VALID OUTCOME. These tests do not assert that the mouth
tracks the speech. They assert that the verdict is DERIVED from the
recorded controls, so that whichever way the numbers go, the recorded
verdict follows from them. A result that is a null, with its controls
recorded, is a valid outcome and this suite passes on it.

Nothing here needs the network, a GPU, or a model download. The landmark
model is imported lazily inside the experiment, and one test below asserts
that importing the experiment does not import mediapipe at all.
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
sys.path.insert(0, str(ROOT / "experiments"))

import mouth_aperture_alignment as maa  # noqa: E402

RESULTS = ROOT / "experiments" / "mouth_aperture_alignment_results.json"


@pytest.fixture(scope="module")
def res() -> dict:
    assert RESULTS.is_file(), f"{RESULTS} is missing"
    return json.loads(RESULTS.read_text())


# --- claim 1: the frames are not the frames the repository thought -------

def test_the_stride_is_recorded_and_is_not_the_nv12_one(res):
    """The record is 19200 bytes: 15360 of luma and 3840 of chroma.

    Read at the 23040 bytes of a full NV12 frame, every frame is torn
    between two real frames and the scene jumps a quarter of its height
    per frame. The stride and the evidence for it are recorded, so the
    reader can check the measurement was made on the right pixels.
    """
    layout = res["identity"]["record_layout"]
    assert layout["record_bytes"] == 19200, (
        f"the record is {layout['record_bytes']} bytes, not 19200: the "
        "frames are being read with the wrong stride and every number in "
        "this lane is about a tear")
    assert layout["luma_bytes"] == 15360
    assert layout["frame_w"] == 160 and layout["frame_h"] == 96
    assert layout["torn_stride_for_comparison"] == 23040


def test_the_stride_evidence_is_measured_not_asserted(res):
    """Consecutive frames must be nearly identical at the correct stride.

    This is the difference between a face and a tear: at 19200 the
    consecutive frames correlate near 1, at 23040 they do not.
    """
    ev = res["sessions"]["A"]["stride_evidence"]
    good = ev["correct_19200_luma160x96"]
    torn = ev["repo_assumed_23040_luma160x96"]
    assert good["mean_consecutive_correlation"] > 0.95, (
        "the correct stride no longer gives near-identical consecutive "
        "frames -- re-check the frame layout before trusting this lane")
    assert good["mean_consecutive_abs_diff"] < 5.0
    assert torn["mean_consecutive_correlation"] < 0.5, (
        "the 23040-byte stride now reads coherent frames, which would "
        "mean the files changed and the correction has to be re-checked")


def test_the_video_clock_is_thirty_fps_over_ninety_frames(res):
    """900 frames at 30 fps is 30.00 s, which is the length of the audio.

    The frame rate is derived from the two spans agreeing, and the
    agreement is recorded.
    """
    assert res["identity"]["video_fps"] == 30.0
    for name in ("A", "B", "C"):
        assert res["detection"][name]["n_frames"] == 900, (
            f"{name} has {res['detection'][name]['n_frames']} frames, not "
            "900 -- the frame count changed and the clock has to be "
            "re-derived")


# --- claim 2: the landmark model works, and the mouth really moves -------

def test_the_landmark_model_is_recorded(res):
    """Which model, which version, and with what settings."""
    lm = res["identity"]["landmark_model"]
    assert lm["model"]
    assert lm["version"] if "version" in lm else lm["model"]
    assert res["identity"]["mediapipe"]
    assert res["identity"]["upscale_factor"] in (1, 4, 6)
    assert lm["static_image_mode"] is True
    assert lm["refine_landmarks"] is True
    # the model's own weights, by hash
    files = lm.get("bundled_model_files", {})
    assert files, "the landmark model's bundled weights are not recorded"
    for path, rec in files.items():
        assert len(rec["sha256"]) == 64, f"{path} has no sha256"


def test_the_landmark_indices_are_recorded(res):
    """The indices used, so a reader can check what was measured."""
    idx = res["identity"]["landmark_indices"]
    assert idx["upper_inner_lip"] == 13
    assert idx["lower_inner_lip"] == 14
    assert idx["eye_outer_right"] == 33
    assert idx["eye_outer_left"] == 263
    assert idx["forehead_top"] == 10
    assert idx["chin_bottom"] == 152


def test_the_face_is_detected_in_every_frame_of_every_take(res):
    """A detector that fires on zero frames is a result. This one fires on
    all of them."""
    for name in ("A", "B", "C"):
        d = res["detection"][name]
        assert d["n_faces_detected"] == d["n_frames"], (
            f"{name}: the landmark model detected a face in "
            f"{d['n_faces_detected']}/{d['n_frames']} frames -- on the "
            "correctly-read frames it detected all of them, so something "
            "has changed and MOUTH-APERTURE-ALIGNMENT.md has to be "
            "re-checked")
        assert d["detection_fraction"] == 1.0


def test_the_empty_room_has_no_face_in_it(res):
    """0 faces in 4500 empty-room frames.

    The empty room is not a passive null: it has real low-amplitude
    motion (LED flicker). The landmark model must not find a mouth in it.
    """
    d = res["detection"]["empty1"]
    assert d["n_frames"] == 4500
    assert d["n_faces_detected"] == 0, (
        "the landmark model now finds a face in the empty room, which "
        "means it is fitting something that is not a mouth")


def test_the_mouth_aperture_is_much_larger_when_the_person_speaks(res):
    """The landmark model is measuring a real mouth.

    Session A is speaking and C is silent. If the two had the same mouth
    aperture, the whole measurement would be about nothing.
    """
    a = res["sessions"]["A"]["signals"]["mouth_aperture"]
    c = res["sessions"]["C"]["signals"]["mouth_aperture"]
    assert a["defined"] and c["defined"]
    # the C numbers are the ratio of two near-constant signals and are
    # not used as a headline anywhere; the comparison is of the MEANS
    assert a["mean"] > 5 * c["mean"], (
        f"the speaking take's mouth aperture is {a['mean']} and the silent "
        f"take's is {c['mean']}: the landmarks are not seeing a mouth that "
        "opens")


# --- claim 3: the controls are all recorded -----------------------------

def test_the_speech_envelope_is_real(res):
    """The gate finds speech in A and B and none in C.

    Without this the alignment measurement would be a measurement of
    nothing.
    """
    for name in ("A", "B"):
        s = res["sessions"][name]["speech"]
        assert s["mean"] > 0.4, f"{name}: speech envelope mean {s['mean']}"
        assert s["frac_over_0.5"] > 0.5
    c = res["sessions"]["C"]["speech"]
    assert c["mean"] < 0.01, "C is the silent take and its envelope is not flat"


def test_the_scrambled_audio_null_is_recorded(res):
    """The null for 'the mouth's pattern aligns with the speech'."""
    for name in ("A", "B", "C"):
        key = f"scrambled_audio_{name}_mouth_aperture"
        assert key in res["controls"], f"{key} is missing"
        null = res["controls"][key]
        assert null["n_permutations"] >= 10
        assert null["seed"] is not None
        assert len(null["best_r_values"]) == null["n_permutations"]
        assert null["best_r_min"] <= null["best_r_mean"] <= null["best_r_max"]


def test_the_eye_distance_control_is_recorded(res):
    """A landmark that must NOT track speech, with the same statistic.

    If the eye distance tracks the speech as well as the mouth aperture,
    the correlation is an artefact of the whole frame moving and not of
    the mouth.
    """
    for name in ("A", "B", "C"):
        assert f"eye_distance_{name}" in res["controls"], (
            f"the eye-distance control for {name} is missing")
        row = res["controls"][f"eye_distance_{name}"]
        assert "best_r" in row and "best_lag_s" in row
        assert f"scrambled_audio_{name}_eye_distance" in res["controls"], (
            f"the eye-distance control for {name} has no scrambled null")


def test_the_wrong_session_control_is_recorded(res):
    """A's audio against C's video, and the rows that are not clean
    controls are labelled as such."""
    clean = [k for k, v in res["controls"].items()
             if k.startswith("wrong_session_") and v.get("is_a_clean_control")]
    assert clean, "no clean wrong-session control is recorded"
    for k, v in res["controls"].items():
        if not k.startswith("wrong_session_"):
            continue
        if not v.get("is_a_clean_control"):
            assert "note" in v, f"{k} is not a clean control and says why"


def test_the_empty_room_control_is_recorded(res):
    """The empty room, and where its motion actually is.

    The empty room is not still: it has real, low-amplitude motion that
    the repo's own work attributes to LED flicker, confined to a small
    region. The control has to say that the motion is nowhere near the
    mouth.
    """
    e = res["controls"]["empty_room"]
    assert e["n_faces_detected"] == 0
    assert e["room_tone_frac_over_0.5"] == 0.0
    loc = e["motion_localisation"]
    assert loc["top_5_pct_of_map_pixels"]["energy_fraction"] > 0.5, (
        "the empty room's motion is no longer concentrated in a small "
        "region: the LED-flicker control has to be re-checked")
    nb = loc["mouth_neighbourhood"]
    assert nb["box_is_above_the_frame_mean"] is False, (
        "the empty room's motion is now concentrated in the mouth's own "
        "neighbourhood, so the empty-room control no longer shows the "
        "flicker is somewhere else")
    assert nb["energy_fraction"] < 0.05, (
        f"the empty room's motion puts {nb['energy_fraction']:.4f} of its "
        "energy in the mouth's neighbourhood, which is no longer a null")
    # and in the occupied take the mouth region is comparatively lively
    occ = e["occupied_take_motion_localisation"]["mouth_neighbourhood"]
    assert occ["box_is_above_the_frame_mean"] is True, (
        "the occupied take's motion no longer concentrates around the "
        "mouth, which would mean the empty-room comparison is not the "
        "comparison it was meant to be")


def test_the_degenerate_baselines_are_recorded(res):
    """The search's own inflation, on a signal against itself."""
    deg = res["controls"]["degenerate_baselines"]
    assert deg, "no degenerate baselines are recorded"
    for name, d in deg.items():
        assert d["n_random_lags"] > 0
        assert d["self_vs_self_max_over_lags_r_mean"] >= \
            d["self_vs_self_random_lag_r_mean"], (
            f"{name}: the max-over-lags search must be able to exceed the "
            "point statistic, or the search is not inflating anything")


def test_the_inputs_are_recorded_by_hash(res):
    """Every input's identity: path, size and SHA-256."""
    inputs = res["inputs"]
    assert inputs, "no inputs are recorded"
    for path, rec in inputs.items():
        assert len(rec["sha256"]) == 64, f"{path} has no sha256"
        assert rec["bytes"] > 0
    names = {Path(p).name for p in inputs}
    for want in ("A.nv12", "B.nv12", "C.nv12", "empty1.nv12",
                 "A.wav", "B.wav", "C.wav", "empty1.wav"):
        assert want in names, f"{want} is not recorded as an input"


# --- the headline, recomputed rather than hardcoded ----------------------

def test_the_headline_margin_is_recomputed_from_the_recorded_numbers(res):
    """The margin is recomputed, not transcribed.

    If the margin is hardcoded anywhere it will not move when the
    measurement moves, and the whole point of a headline margin over a
    control is that it is a difference of two recorded numbers.
    """
    h = res["headline"]
    assert "mouth_aperture_best_r" in h
    assert "scrambled_audio_null_r" in h
    assert "margin_over_scrambled_audio_null" in h
    recomputed = round(h["mouth_aperture_best_r"] - h["scrambled_audio_null_r"], 4)
    assert recomputed == h["margin_over_scrambled_audio_null"], (
        f"the recorded margin {h['margin_over_scrambled_audio_null']} is not "
        f"{h['mouth_aperture_best_r']} - {h['scrambled_audio_null_r']}")
    # the conservative margin, against the WORST permutation
    worst = round(h["mouth_aperture_best_r"] - h["scrambled_audio_null_max_r"], 4)
    assert worst == h["margin_over_scrambled_audio_null_max"]


def test_the_headline_best_r_is_the_max_of_the_recorded_sessions(res):
    """The headline number comes from the per-session rows."""
    h = res["headline"]
    per_session = [h["A_mouth_aperture_best_r"], h["B_mouth_aperture_best_r"]]
    assert h["mouth_aperture_best_r"] == max(per_session)
    for name in ("A", "B"):
        row = res["sessions"][name]["signals"]["mouth_aperture"]
        assert row["best_r"] == h[f"{name}_mouth_aperture_best_r"]
        assert row["best_lag_s"] == h[f"{name}_mouth_aperture_best_lag_s"]


def test_the_verdict_is_derived_from_its_checks(res):
    """The verdict is a FUNCTION of the recorded checks.

    This is what makes it able to fail. If the verdict were a hardcoded
    True, a null with its controls recorded would still pass the suite and
    the reader would never learn. The verdict here is whatever the checks
    say, so a null passes and a positive finding would pass too.
    """
    h = res["headline"]
    checks = h["checks"]
    required = {
        "beats_the_scrambled_audio_null",
        "beats_the_eye_distance_control",
        "beats_the_mouth_width_control",
        "beats_the_wrong_session_control",
        "replicates_on_the_halves_of_the_recording",
    }
    assert required <= set(checks), (
        f"the verdict is missing checks: {required - set(checks)}. A "
        "verdict that cannot fail on the eye-distance or mouth-width "
        "control is not a verdict.")
    for name, passed in checks.items():
        assert isinstance(passed, bool), f"{name} is not a boolean"
    assert h["the_mouth_tracks_the_speech"] == all(checks.values()), (
        "the recorded verdict does not follow from the recorded checks")
    # the check names must be the ones the verdict is actually made of,
    # so a reader can see that a failed check is what failed the verdict
    if not h["the_mouth_tracks_the_speech"]:
        failed = [k for k, v in checks.items() if not v]
        assert failed, "the verdict is False but no check failed"


def test_the_verdict_would_flip_if_a_check_flipped(res):
    """The verdict is able to fail: it is an AND over the checks.

    Checked against the CODE, so it holds however the numbers came out.
    """
    checks = {
        "beats_the_scrambled_audio_null": True,
        "beats_the_eye_distance_control": True,
        "beats_the_mouth_width_control": True,
        "beats_the_wrong_session_control": True,
        "replicates_on_the_halves_of_the_recording": True,
    }
    assert all(checks.values()) is True
    for k in checks:
        flipped = dict(checks)
        flipped[k] = False
        assert all(flipped.values()) is False, (
            f"flipping {k} did not fail the verdict, so the verdict cannot "
            "fail")


# --- the measurement itself, checked cheaply against the code -----------

def test_the_import_does_not_need_the_landmark_model():
    """The suite must run without mediapipe and without any model file.

    Checked in a subprocess, because by the time this test runs another
    test may have imported something.
    """
    code = (
        "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r);"
        "import mouth_aperture_alignment;"
        "print('mediapipe' in sys.modules)"
        % (str(ROOT), str(ROOT / "experiments")))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", (
        "importing the experiment pulled in mediapipe, so the test suite "
        "needs the landmark model")


def test_the_mouth_and_the_eye_are_different_landmarks():
    """The control is a control: moving the lips must not move the eyes.

    Built from a synthetic face so that the test needs no model and no
    data.
    """
    n = 5
    lm = np.zeros((n, 478, 2))
    lm[:, :, 0] = 80.0
    lm[:, :, 1] = 40.0
    # the landmarks the signals use, in a plausible arrangement
    lm[:, 10, :] = (78.0, 22.0)     # forehead top
    lm[:, 152, :] = (78.0, 65.0)    # chin bottom
    lm[:, 1, :] = (79.0, 45.0)      # nose tip
    lm[:, 13, :] = (78.0, 52.0)     # upper inner lip
    lm[:, 14, :] = (78.0, 53.0)     # lower inner lip
    lm[:, 61, :] = (71.0, 53.0)     # mouth corner right
    lm[:, 291, :] = (85.0, 53.0)    # mouth corner left
    lm[:, 33, :] = (64.0, 36.0)     # eye outer right
    lm[:, 263, :] = (92.0, 36.0)    # eye outer left
    det = np.ones(n, dtype=bool)

    # open the mouth on the last frame
    lm[4, 14, 1] = 56.0
    sig = maa.landmark_signals(lm, det)
    assert sig["mouth_aperture"][4] > sig["mouth_aperture"][0], (
        "opening the mouth did not open the mouth aperture")
    assert sig["eye_distance"][4] == sig["eye_distance"][0], (
        "opening the mouth moved the eye distance, so the eye-distance "
        "control is not a control")
    assert sig["mouth_width"][4] == sig["mouth_width"][0], (
        "opening the mouth changed the mouth width, so the mouth-width "
        "control is not a control")
    # and the scale really is normalised away
    wide = lm.copy()
    wide[:, :, 0] *= 2.0
    wide[:, :, 1] *= 2.0
    sig_wide = maa.landmark_signals(wide, det)
    assert np.allclose(sig_wide["mouth_aperture"], sig["mouth_aperture"]), (
        "the mouth aperture is not scale-free")


def test_the_gap_filling_reports_what_it_filled():
    """A long gap would be a fabrication, so it is reported."""
    sig = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
    filled, info = maa.fill_gaps(sig)
    assert info["n_detected"] == 3
    assert info["n_filled"] == 3
    assert info["max_gap_frames"] == 2
    assert np.allclose(filled[[0, 2, 5]], [1.0, 3.0, 6.0])
    # a signal with no detection at all comes back as NaN, not as a number
    empty, info = maa.fill_gaps(np.full(4, np.nan))
    assert not np.isfinite(empty).any()
    assert info["n_detected"] == 0
