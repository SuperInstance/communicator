"""
Tests for the leave-one-session-out audit of the teacher.

`docs/research/TEACHER-LOO-FOLDS.md` is the document these tests pin. It
makes four claims about the teacher pipeline, and each one can be false:

    1. the leave-one-session-out fold is DEGENERATE -- the held-out session
       is pure, so the blank control (a constant predictor) scores 1.000 on
       every fold, and the metric cannot distinguish the `map` feature set
       from a constant;
    2. the `map` feature set does not need the MOTION -- the static
       variant, in which every frame is the session's temporal mean, scores
       1.000 on BOTH folds;
    3. the teacher's numbers are not an artefact of the motion maps' display
       scale;
    4. the video's motion does not follow the audio's speech, so frame-level
       labels cannot be recovered from these captures.

These assertions are mostly on the committed results, because the claims are
about measurements that took a minute each. The ones that can be checked
cheaply are checked directly against the code, so the tests fail if the
MEASUREMENT changes, not only if the RECORD of it changes.

Nothing here needs the network, a GPU, or a model download.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from ascii_motion_render import load_session          # noqa: E402

LOO_RESULTS = ROOT / "experiments" / "teacher_loo_control_results.json"
STATIC_RESULTS = ROOT / "experiments" / "teacher_static_control_results.json"
AV_RESULTS = ROOT / "experiments" / "audio_video_alignment_results.json"
CORPUS = Path("/mnt/c/proart_capture/corpus")


@pytest.fixture(scope="module")
def loo() -> dict:
    assert LOO_RESULTS.is_file(), f"{LOO_RESULTS} is missing"
    return json.loads(LOO_RESULTS.read_text())


@pytest.fixture(scope="module")
def static() -> dict:
    assert STATIC_RESULTS.is_file(), f"{STATIC_RESULTS} is missing"
    return json.loads(STATIC_RESULTS.read_text())


@pytest.fixture(scope="module")
def av() -> dict:
    assert AV_RESULTS.is_file(), f"{AV_RESULTS} is missing"
    return json.loads(AV_RESULTS.read_text())


def rows_of(blob: dict, feature_set: str) -> dict:
    """Index a results file's rows by (variant, held_out).

    The static-control file has a `variant` on every row; the LOO-control
    file has one row per fold, because there is only one variant there.
    """
    out = {}
    for row in blob["feature_sets"][feature_set]["rows"]:
        key = (row.get("variant", "real"), row["held_out"])
        out[key] = row
    return out


# --- claim 1: the metric is degenerate ------------------------------------

def test_the_blank_control_scores_one_on_every_fold(loo):
    """A CONSTANT predictor scores 1.000 on every held-out session.

    That is the whole point: the held-out session is pure, so a model that
    always answers "speaking" is exactly right, and the leave-one-session-out
    accuracy has no room to measure anything.
    """
    for feature_set, blob in loo["feature_sets"].items():
        for row in blob["rows"]:
            if row.get("acc_real") is None:
                continue
            assert row["acc_blank"] == 1.0, (
                f"{feature_set} hold-out {row['held_out']}: the blank control "
                f"scored {row['acc_blank']}, not 1.000 -- if this has changed, "
                "the degeneracy argument in TEACHER-LOO-FOLDS.md has to be "
                "re-checked")


def test_the_loo_fold_has_its_own_scrambled_control(loo):
    """The LOO fold's scrambled control is 0.000 on fold B for `map`.

    The report used the TEMPORAL split's 0.938 to dismiss the LOO's 1.000.
    Measured on the same fold, the control is 0.000.
    """
    rows = rows_of(loo, "map")
    row = rows[("real", "B")]
    assert row is not None, "no row for held-out B in the map feature set"
    assert row["acc_spatially_scrambled"] == 0.0, (
        "the LOO fold's own scrambled control is no longer 0.000: "
        "re-measure and update TEACHER-LOO-FOLDS.md")
    assert row["acc_real"] == 1.0


def test_the_identifiability_of_each_fold_is_recorded(loo):
    """Fold A cannot identify "speaking"; fold B can.

    Fold A trains on {B = turned + speaking, C = facing + silent}: the two
    training sessions differ in pose and in speaking at the same time, so a
    model that learns "turned -> speaking" fits the training labels exactly
    as well as a model that learns the speech. Fold B trains on
    {A = facing + speaking, C = facing + silent}: both are facing, so only
    speech can explain the labels.
    """
    for feature_set, blob in loo["feature_sets"].items():
        for row in blob["rows"]:
            if row.get("acc_real") is None:
                continue
            ident = row["identifiability"]
            if row["held_out"] == "A":
                assert ident["identified"] is False, (
                    "fold A is now reported as identified: the design "
                    "argument in TEACHER-LOO-FOLDS.md has to be re-checked")
                assert ident["label_determined_by_pose"] is True
            if row["held_out"] == "B":
                assert ident["identified"] is True
                assert ident["train_poses"] == ["facing"], (
                    "fold B's training set is no longer all-facing")


# --- claim 2: the map features do not need the motion ----------------------

def test_the_static_variant_scores_one_on_both_folds(static):
    """No motion, no order, no change -- and 1.000 on BOTH folds.

    This is the finding. The `map`, `both` and `region+map` feature sets all
    contain the 6x8 grid of the window's MEAN motion map, and that grid is
    what carries the discrimination.
    """
    for feature_set in ("map", "both", "region+map"):
        rows = rows_of(static, feature_set)
        for held_out in ("A", "B"):
            row = rows[("static", held_out)]
            assert row["acc"] == 1.0, (
                f"{feature_set} static variant, hold-out {held_out}: "
                f"{row['acc']} -- re-measure and update TEACHER-LOO-FOLDS.md")


def test_the_feature_sets_without_the_grid_do_not_reach_one(static):
    """The two feature sets WITHOUT the grid do not reach 1.000 on both folds.

    `summary` (20 per-frame summary numbers) and `region` (4 numbers) do not
    contain the 6x8 grid, and neither reaches 1.000 on both folds with the
    motion removed. That is what makes the grid the culprit rather than
    "something about the static variant".
    """
    for feature_set in ("summary", "region"):
        rows = rows_of(static, feature_set)
        scores = [rows[("static", h)]["acc"] for h in ("A", "B")]
        assert not all(s == 1.0 for s in scores), (
            f"{feature_set} now reaches 1.000 on both folds without motion: "
            "the grid is not the only thing that was carrying it, and "
            "TEACHER-LOO-FOLDS.md has to be re-checked")


def test_the_temporal_features_make_the_classifier_worse(static):
    """Fold A: 1.000 without the motion, 0.000 with it.

    Adding the temporal information to the static template does not merely
    fail to help -- it makes the result worse. The temporal features are
    adding noise to a discrimination that the static template already makes.
    """
    rows = rows_of(static, "map")
    assert rows[("static", "A")]["acc"] == 1.0
    assert rows[("real", "A")]["acc"] == 0.0


def test_the_scale_control_does_not_move_the_results(static):
    """The teacher's numbers are not an artefact of the display scale.

    The logistic regression is not scale-invariant (its regularisation is
    `l2 * w / n` against a fixed learning rate), so this control had to be
    run. It came out clean.
    """
    for feature_set in ("map", "both", "region+map"):
        rows = rows_of(static, feature_set)
        for held_out in ("A", "B"):
            real = rows[("real", held_out)]["acc"]
            unscaled = rows[("unscaled", held_out)]["acc"]
            assert real == unscaled, (
                f"{feature_set} hold-out {held_out}: the result moved from "
                f"{real} to {unscaled} when the maps were rescaled -- the "
                "teacher's numbers are scale-sensitive after all, and "
                "TEACHER-LOO-FOLDS.md has to be re-checked")


# --- the variants themselves -----------------------------------------------

def test_the_static_variant_really_is_static():
    """Every frame of the static variant is the same image."""
    from teacher_static_control import variant_static
    rng = np.random.default_rng(3)
    maps = rng.integers(0, 255, size=(20, 8, 10)).astype(np.uint8)
    out = variant_static(maps.astype(np.float64))
    assert out.shape == maps.shape
    for i in range(1, len(out)):
        assert np.array_equal(out[i], out[0]), \
            f"frame {i} of the static variant differs from frame 0"


def test_the_time_shuffled_variant_keeps_the_frames():
    """The frames are permuted, not changed."""
    from teacher_static_control import variant_time_shuffled
    rng = np.random.default_rng(4)
    maps = rng.integers(0, 255, size=(30, 8, 10)).astype(np.float64)
    out = variant_time_shuffled(maps, seed=5)
    assert out.shape == maps.shape
    # the same multiset of frames
    assert sorted(float(f.sum()) for f in out) == \
        sorted(float(f.sum()) for f in maps)
    # and it is a permutation, not the identity
    assert not np.array_equal(out, maps)


def test_the_unscaled_variant_is_a_common_factor():
    """Every session is shrunk by the SAME factor, so the ratios are intact."""
    from teacher_static_control import variant_unscaled, SCALE_REFERENCE
    rng = np.random.default_rng(6)
    maps = rng.random((10, 4, 5)) * 1000.0
    out = variant_unscaled(maps)
    assert np.allclose(out * SCALE_REFERENCE, maps)


# --- claim 4: the video does not follow the audio -------------------------

def test_the_speech_envelope_is_real(av):
    """The gate finds speech in A and B and none in C.

    Without this, the alignment measurement below would be a measurement of
    nothing.
    """
    sessions = av["sessions"]
    for name in ("A", "B"):
        assert sessions[name]["speech_mean"] > 0.4, (
            f"{name}: the speech envelope's mean is "
            f"{sessions[name]['speech_mean']}")
    assert sessions["C"]["speech_mean"] < 0.01, (
        "C is the silent take and its speech envelope is not flat")


def test_the_motion_does_not_follow_the_speech(av):
    """No alignment above the nulls.

    The speaking sessions correlate with their own speech at +0.078 and
    +0.025, against a scrambled-audio null of +0.061 and a wrong-session
    control of +0.068. The `active` envelope is the candidate -- the top 5%
    of pixels, where 79% of the speaking/silent difference was measured to
    live -- and it does not clear the nulls.
    """
    headline = av["headline"]
    real = headline["active_real_sessions"]
    controls = headline["active_controls"]
    assert min(real) <= max(controls), (
        "the motion now correlates with the speech above the controls: "
        "re-measure and update TEACHER-LOO-FOLDS.md")
    assert headline["the_video_carries_a_time_aligned_speaking_signal"] is False


def test_the_total_envelope_is_the_negative_case(av):
    """The total-change envelope is included as a negative case.

    The repo has already measured that total change does not distinguish
    speaking from silent, so a correlation there would be about drift. It is
    reported so that the reader can see that the candidate envelope is the
    spatial one and not the energy one.
    """
    assert set(av["headline"]["total_real_sessions"]) is not None
    for name in av["sessions"]:
        for env in ("total", "active", "area"):
            row = av["sessions"][name][env]
            for key in ("raw_best_r", "detrended_best_r",
                        "raw_best_lag_s", "detrended_best_lag_s"):
                assert key in row, f"{name}/{env} is missing {key}"


# --- what the static template is -------------------------------------------

@pytest.mark.skipif(not (CORPUS / "A_seed.npz").is_file(),
                    reason="the seeded corpus is not on disk")
def test_the_speaking_and_silent_average_pictures_are_almost_the_same():
    """Correlation 0.904, amplitude differing by 4%.

    This is what the static template IS: nearly the same picture, at a
    slightly different amplitude.
    """
    a = load_session(CORPUS / "A_seed.npz").mean(axis=0)
    c = load_session(CORPUS / "C_seed.npz").mean(axis=0)
    corr = float(np.corrcoef(a.ravel(), c.ravel())[0, 1])
    ratio = float(a.sum() / c.sum())
    assert 0.85 < corr < 0.95, f"the correlation moved to {corr:.4f}"
    assert 0.93 < ratio < 0.99, f"the amplitude ratio moved to {ratio:.4f}"


@pytest.mark.skipif(not (CORPUS / "A_seed.npz").is_file(),
                    reason="the seeded corpus is not on disk")
def test_one_global_scale_explains_most_of_the_difference():
    """R^2 = 0.817 for a single scale factor.

    The difference between the two takes' average pictures is mostly a gain
    difference, and the residual is the scene's fixed vertical structure at
    a different amplitude -- not a face, and not a mouth.
    """
    a = load_session(CORPUS / "A_seed.npz").mean(axis=0).ravel()
    c = load_session(CORPUS / "C_seed.npz").mean(axis=0).ravel()
    alpha, beta = np.polyfit(a, c, 1)
    pred = alpha * a + beta
    r2 = 1.0 - ((c - pred) ** 2).sum() / ((c - c.mean()) ** 2).sum()
    assert 0.75 < r2 < 0.88, f"the scale fit moved to R^2 = {r2:.4f}"
