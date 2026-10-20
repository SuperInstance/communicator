"""
Tests for the mouth-detector measurement.

`docs/research/MOUTH-DETECTOR.md` is the document these tests pin. It makes
these claims, and each one can be false:

    1. the captures are read at the DETECTED stride (19,200 bytes), not at
       the 23,040 the repository used to assume, and every reader in this
       lane goes through `src/nv12.py`;
    2. a mouth-localised feature separates the speaking take from the
       silent take even with the head-motion proxy held constant -- the
       signal SURVIVES the confound, which is the central question of the
       lane;
    3. that signal does NOT beat its spatial control, so the result is not
       a claim that the detector is a MOUTH detector rather than a take
       detector;
    4. the leave-one-session-out accuracy does NOT beat a constant
       predictor, because the labels are session-level -- the metric is
       partly degenerate, and the document says so;
    5. the evaluation is SESSION-LEVEL only: the audio and the video are
       independent takes, so no frame-level audio<->video label is
       recoverable, and this lane does not pretend otherwise;
    6. the verdict is DERIVED from the recorded controls, so it can come
       out the other way.

THE NULL IS A VALID OUTCOME. These tests do not assert that the mouth
signal survives or that it fails -- they assert that the recorded verdict
follows from the recorded control numbers, that the verdict function can
produce the opposite answer, and that the specific findings this lane
recorded are recorded. A future run that flips a number will fail the
checks that pin it, which is the point.

Nothing here needs the network, a GPU, or a model download. The experiment
module is numpy-only and imports no landmark model at all.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import mouth_detector as md  # noqa: E402

RESULTS = ROOT / "experiments" / "mouth_detector_results.json"
SCRIPT = ROOT / "experiments" / "mouth_detector.py"


@pytest.fixture(scope="module")
def res() -> dict:
    assert RESULTS.is_file(), f"{RESULTS} is missing"
    return json.loads(RESULTS.read_text())


# --- claim 1: the frames are read at the right stride ---------------------

def test_the_stride_is_detected_and_is_not_the_nv12_one(res):
    """Each capture is read at the layout the reader DETECTS.

    The captures are 19,200-byte records (15,360 of 160x96 luma plus
    3,840 of chroma). Read at the 23,040 of a standard NV12 frame, every
    frame is torn between two real frames and every number in this lane
    would be about the tear.
    """
    layout = res["identity"]["record_layout"]
    for sess in ("A", "B", "C", "empty1"):
        rec = layout[sess]
        assert rec["stride_bytes"] == 19200, (
            f"{sess} is read at {rec['stride_bytes']} bytes per frame, not "
            "19,200: the frames are torn and this lane is about a tear")
        assert rec["layout"] == "y_uv_halfres"
        assert rec["margin_over_runner_up"] > 5.0, (
            f"{sess}'s layout is ambiguous (margin "
            f"{rec['margin_over_runner_up']}), so the reader guessed")
    assert layout["A"]["n_frames"] == 900
    assert layout["C"]["n_frames"] == 900
    assert layout["empty1"]["n_frames"] == 4500


def test_the_features_are_computed_on_the_distinct_frames(res):
    """The captures are not 30 DISTINCT frames per second, and the lane
    knows it.

    304 of A's 899 consecutive frame pairs are byte-identical: the camera
    delivered ~19.9 distinct images per second and the driver repeated
    frames to fill the 30 fps grid. Differencing the grid as recorded
    reads about a third of the frames as "nothing happened", which is a
    property of the capture driver and not of the wearer.
    """
    for sess in ("A", "B", "C"):
        cap = res["sessions"][sess]["capture"]
        assert cap["n_duplicate_consecutive_pairs"] > 250, (
            f"{sess} reports only {cap['n_duplicate_consecutive_pairs']} "
            "duplicated frames, which contradicts the measured corpus")
        assert cap["n_distinct_frames"] < cap["n_frames_on_disk"]
        assert res["sessions"][sess]["n_distinct_frames"] == \
            cap["n_distinct_frames"]
        assert 18.0 < cap["distinct_frame_rate_hz"] < 21.0, (
            f"{sess}'s distinct-frame rate is "
            f"{cap['distinct_frame_rate_hz']:.2f} Hz; the lane's reading of "
            "the capture has changed and the numbers must be re-checked")


# --- claim 2: the signal survives the head-motion control -----------------

def test_the_mouth_signal_survives_the_head_motion_control(res):
    """The conditional AUC is computed WITHIN strata of the head-motion
    proxy, so the head motion is held as constant as the strata allow.

    This is the lane's central claim and it can be false: if the mouth
    signal were reducible to head motion, the conditional AUC would sit
    at 0.50.
    """
    r1a = res["checks"]["R1a_conditional_auc"]
    assert r1a["value"] > 0.5, (
        f"the mouth feature's head-motion-controlled AUC is {r1a['value']:.4f}, "
        "at or below chance: the mouth signal does not survive the "
        "head-motion control")
    assert r1a["perm_z"] >= 2.0, (
        f"the conditional AUC's permutation z is {r1a['perm_z']:.2f}, under "
        "the pre-registered +2.0")
    assert r1a["null_mean"] == pytest.approx(0.5, abs=0.05), (
        "the conditional AUC's null is not centred on 0.50, so the "
        "statistic itself is miscalibrated")


def test_the_residualised_effect_is_positive_in_every_fold(res):
    """With the head-motion proxy regressed out, the mouth feature's effect
    is positive in every fold, with the regression fitted on the training
    sessions only."""
    r1b = res["checks"]["R1b_residual_d"]
    assert set(r1b["per_fold"]) == {"fold_A", "fold_B", "fold_C"}
    for fold, rec in r1b["per_fold"].items():
        assert rec["residual_d"] > 0, (
            f"{fold}: the residualised effect is {rec['residual_d']:+.4f} sd; "
            "the mouth signal does not survive the head-motion control")
        assert rec["perm_z"] >= 3.0, (
            f"{fold}: the residualised effect's permutation z is "
            f"{rec['perm_z']:.2f}, under the pre-registered +3.0")
        # the held-out session is never used to fit the covariate
        assert rec["train_sessions"] != []


def test_the_head_motion_covariate_actually_moves(res):
    """The covariate has to be a covariate.

    If the head-motion proxy did not separate the speaking take from the
    silent one, the control would be vacuous and the conditional AUC would
    be measuring nothing.
    """
    head = res["comparisons"]["head_proxies"]["H_upper"]["A_vs_C"]
    assert head["auc"] > 0.6, (
        f"the head-motion proxy separates A from C at only AUC "
        f"{head['auc']:.4f}: the central control is too weak to be a control")
    # and conditioning on it removes most of the proxy's own separation,
    # which is what shows that the conditioning works
    check = res["comparisons"]["boxes"]["mouth"][
        res["checks"]["headline_mouth_feature"]]["head_covariate_check"]
    assert check["conditional_auc_A_vs_C"] < 0.60, (
        "conditioning on the head-motion proxy did not remove the proxy's "
        "own separation, so the conditional AUC is not interpretable")


# --- claim 3: it does not beat the spatial control ------------------------

def test_the_spatial_control_is_measured_and_reported(res):
    """The mouth box's head-motion-controlled AUC is compared against the
    cheek control's, with a PAIRED null (the same label permutation is
    applied to both boxes).

    The comparison is recorded whether it passes or fails; the point is
    that it is made at all.
    """
    r2 = res["checks"]["R2_mouth_specific"]
    assert "mouth_conditional_auc" in r2 and "cheek_conditional_auc" in r2
    assert r2["difference"] == pytest.approx(
        r2["mouth_conditional_auc"] - r2["cheek_conditional_auc"], abs=1e-9)
    assert np.isfinite(r2["perm_z"])
    # the per-candidate table must show the same comparison for every
    # candidate, including the ones where the mouth wins
    per = res["checks"]["per_candidate_survival"]
    assert len(per) == len(md.FEATURES)
    diffs = [v["mouth_minus_cheek_conditional_auc"] for v in per.values()]
    assert max(diffs) > 0.05, (
        "no candidate feature beats the spatial control by more than 0.05; "
        "the per-candidate table must record the ones that do")
    assert min(diffs) < 0.0, (
        "no candidate feature loses to the spatial control; the table must "
        "record the ones that do, or the selection is not being reported "
        "honestly")


def test_the_geometry_of_the_spatial_control_is_recorded(res):
    """The 'cheek' control is not on a cheek.

    The face's right-hand edge is at x ~ 90 and the cheek control starts
    at column 92, so the control box STRADDLES the face's silhouette: a
    high-contrast edge that moves with the head passes through it. The
    lane measures this rather than assuming it, because a control that
    moves with the head is not a clean spatial control.
    """
    sd = res["geometry"]["temporal_sd_by_box"]
    assert sd["A"]["mouth"] > sd["A"]["bg"], (
        "the mouth box is no more alive than the background, so the mouth "
        "measurement is not measuring the wearer")
    assert sd["A"]["cheek"] > sd["A"]["bg"], (
        "the cheek control is static like the background, so it is not the "
        "moving-edge region the geometry says it is")
    assert "STRADDLES" in res["geometry"]["what_each_box_contains"]


# --- claim 4: the session-level metric is partly degenerate ---------------

def test_the_blank_control_is_reported_and_the_metric_is_degenerate(res):
    """A constant predictor is exactly right on a pure speaking session.

    This is the repo's standing finding (`docs/research/TEACHER-LOO-FOLDS.md`
    §1) and it applies here: the blank control must be reported beside
    every accuracy, and the lane must not claim to beat it if it does not.
    """
    blank = res["nulls"]["blank_constant_speaking"]
    assert blank["per_fold"]["A"] == 1.0
    assert blank["per_fold"]["B"] == 1.0
    assert blank["per_fold"]["C"] == 0.0
    assert res["nulls"]["blank_training_majority"]["macro_frame_recall"] == \
        pytest.approx(blank["macro_frame_recall"])
    r4 = res["checks"]["R4_beats_nulls"]
    assert r4["margins"]["blank"] == pytest.approx(
        r4["mouth_macro_frame_recall"] - r4["nulls"]["blank"], abs=1e-9)


def test_the_wrong_session_pairing_is_run(res):
    """A's video with C's labels and vice versa is the negative control the
    brief asks for, and it is reported with its number."""
    wrong = res["nulls"]["wrong_session_pairing"]
    assert wrong["labels"] == {"A": 0, "B": 1, "C": 1}
    assert 0.0 <= wrong["macro_frame_recall"] <= 1.0
    assert set(wrong["per_fold"]) == {"A", "B", "C"}


def test_the_permuted_label_nulls_are_run(res):
    """Both the session-label permutations and the frame-label shuffle are
    present, with their distributions."""
    perms = res["nulls"]["permuted_session_labels"]
    assert len(perms["permutations"]) == 2, (
        "there are exactly two alternative assignments of three sessions to "
        "{speaking, speaking, silent}; the null is weak and says so")
    shuffle = res["nulls"]["frame_label_shuffle"]
    assert shuffle["n_draws"] >= 100
    assert shuffle["sd"] > 0.0, (
        "the frame-label shuffle produced no spread, so the labels are not "
        "actually being permuted")


def test_the_empty_room_is_a_physical_control(res):
    """The empty room carries no person, so its mouth-box motion is the
    floor, and the detectors' false-alarm rate on its held-out slice is
    reported."""
    mouth = res["empty_room_control"]["boxes"]["mouth"]
    for feature, rec in mouth.items():
        assert rec["A_over_empty_ratio_of_means"] > 100.0, (
            f"the empty room is only {rec['A_over_empty_ratio_of_means']:.1f}x "
            f"quieter than A in the {feature} feature: the empty room is "
            "not acting as a physical control")
    fa = res["empty_room_false_alarm"]["per_fold"]
    assert set(fa) == set(md.BOXES)
    for box, rows in fa.items():
        assert len(rows) == 3
        for row in rows:
            assert 0.0 <= row["fraction_predicted_speaking"] <= 1.0


# --- claim 5: session level, not frame level ------------------------------

def test_the_session_level_versus_frame_level_distinction_is_pinned(res):
    """The evaluation is at the SESSION level and the lane says so.

    The audio and the video of A/B/C are independent takes: no lag in
    +/-20 s and no video time base in 10-60 Hz aligns them. So frame-level
    audio<->video labels are not recoverable from this corpus, and a
    detector evaluated against them would be a fiction.
    """
    gate = res["checks"]["R5_honesty_gate"]
    assert gate["frame_level_labels_recoverable"] is False
    assert "independent takes" in gate["why"]
    assert res["verdict"]["scope"].startswith("SESSION-LEVEL")
    # the folds are sessions, and there are exactly three of them
    for box in md.BOXES:
        folds = res["loo"]["boxes"][box]["per_fold_selected"]
        assert [f["test_session"] for f in folds] == ["A", "B", "C"]
        for fold in folds:
            assert fold["held_out_session_label"] in (0, 1)
    # the empty room is never a labelled fold: it is a physical control
    assert "empty1" not in [f["test_session"]
                            for f in res["loo"]["boxes"]["mouth"][
                                "per_fold_selected"]]


def test_the_lane_never_reads_the_audio(res):
    """No frame-level label can come from an audio file that is not aligned,
    so this lane must not read one at all.

    The script's source is checked, not just its results: an audio label
    could only enter through the audio.
    """
    source = SCRIPT.read_text()
    for banned in (".wav", "SileroGate", "vad", "speech_envelope"):
        assert banned not in source, (
            f"the experiment reads '{banned}': this lane has no aligned "
            "audio, so any audio-derived label would be a fiction")


# --- claim 6: the verdict is derived, and can come out the other way ------

def test_the_verdict_is_derived_from_the_recorded_controls(res):
    """The recorded verdict is the one the recorded numbers imply.

    This is the test that fails if the detector stops beating its
    controls: the checks are recomputed by the verdict function and the
    answer must be the recorded one.
    """
    recomputed = md.derive_verdict(res["checks"])
    assert recomputed["verdict"] == res["verdict"]["verdict"], (
        f"the recorded verdict ({res['verdict']['verdict']}) is not what "
        f"the recorded controls imply ({recomputed['verdict']})")


def test_the_check_flags_are_derived_from_their_own_numbers(res):
    """Each check's pass/fail is recomputed from its own recorded numbers.

    A number edited without its flag updated would fail here.
    """
    checks = res["checks"]
    r1a = checks["R1a_conditional_auc"]
    assert r1a["passes"] == bool(r1a["value"] > 0.5 and r1a["perm_z"] >= 2.0)
    r2 = checks["R2_mouth_specific"]
    assert r2["passes"] == bool(r2["difference"] >= 0.02 and r2["perm_z"] >= 2.0)
    r3 = checks["R3_beats_head_motion"]
    assert r3["passes"] == bool(r3["recall_margin"] >= 0.05
                                and r3["folds_not_worse"] >= 2)
    r4 = checks["R4_beats_nulls"]
    assert r4["passes"] == bool(all(v >= 0.05 for v in r4["margins"].values()))


def test_the_verdict_function_can_fail(res):
    """The verdict must be able to come out the other way.

    The mouth feature's head-motion-controlled AUC is set to chance and the
    verdict must change. If this test fails, the verdict is a constant, and
    a constant is not a measurement.
    """
    checks = json.loads(json.dumps(res["checks"]))
    checks["R1a_conditional_auc"]["value"] = 0.50
    checks["R1a_conditional_auc"]["perm_z"] = 0.0
    flipped = md.derive_verdict(checks)
    assert flipped["verdict"] == "mouth_signal_does_not_survive_head_motion"
    assert flipped["verdict"] != res["verdict"]["verdict"]

    checks = json.loads(json.dumps(res["checks"]))
    checks["R4_beats_nulls"]["margins"]["blank"] = -0.5
    checks["R2_mouth_specific"]["difference"] = 0.5
    checks["R2_mouth_specific"]["perm_z"] = 5.0
    flipped = md.derive_verdict(checks)
    assert flipped["verdict"] == "mouth_detector_beats_nothing"
    assert flipped["verdict"] != res["verdict"]["verdict"]


def test_the_pre_registered_rule_is_recorded_verbatim(res):
    """The rule that decided the verdict is in the results, and it is the
    rule the script carries, so it cannot be quietly rewritten."""
    assert res["pre_registered_rule_text"] == md.PRE_REGISTERED_RULE_TEXT
    assert res["pre_registered_rule"] == md.PRE_REGISTERED_RULE
    rule = res["pre_registered_rule_text"]
    for needed in ("R1a", "R1b", "R2", "R3", "R4", "R5",
                   "conditional_auc", "residual_d", "permutation"):
        assert needed in rule, f"the recorded rule does not mention {needed}"


def test_the_provenance_block_is_complete(res):
    """The script's own hash, the inputs' hashes, the CLI and the Python
    version, so the result can be reproduced or its absence noticed."""
    prov = res["provenance"]
    assert prov["script_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()).hexdigest(), (
        "the results file was produced by a different version of the "
        "script than the one on disk")
    assert prov["python"]
    assert "cli_args" in prov and "command_line" in prov
    inputs = prov["inputs"]
    for name in ("A.nv12", "B.nv12", "C.nv12", "empty1.nv12"):
        paths = [p for p in inputs if p.endswith(name)]
        assert paths, f"the provenance records no hash for {name}"
        assert len(inputs[paths[0]]["sha256"]) == 64


def test_the_published_mouth_effect_is_reproduced(res):
    """`docs/research/NV12-STRIDE.md` §5's mouth effect is re-measured here
    with the same statistic, on the 30 fps grid as recorded.

    If the lane's reading of the capture had changed, this number would
    change, and the earlier document's number would need a dated
    correction.
    """
    repro = res["published_claim_reproduction"]
    grid = repro["grid_30fps_as_recorded"]
    assert grid["mouth_A_minus_C"] == pytest.approx(
        repro["published"]["mouth_A_minus_C"], abs=0.05)
    assert grid["cheek_A_minus_C"] == pytest.approx(
        repro["published"]["cheek_A_minus_C"], abs=0.05)
    assert grid["mouth_over_cheek_ratio_of_effects"] == pytest.approx(
        repro["published"]["mouth_over_cheek_ratio_of_effects"], abs=0.05)
    assert grid["empty_room_share_of_A"] == pytest.approx(
        repro["published"]["empty_room_share_of_A"], abs=5e-5)


def test_the_experiment_imports_no_landmark_model():
    """This lane is numpy-only.

    The landmark-based mouth aperture is the subject of
    `docs/research/MOUTH-APERTURE-ALIGNMENT.md`, not of this lane, and the
    result must reproduce without a GPU.
    """
    assert "mediapipe" not in SCRIPT.read_text()
    assert "cv2" not in SCRIPT.read_text()
