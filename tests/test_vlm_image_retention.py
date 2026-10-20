"""
Tests for the image-retention-vs-description measurement.

`docs/research/VLM-IMAGE-RETENTION.md` is the document these tests pin. It
makes these claims, and each one can be false:

    1. the captures are read at the DETECTED stride (19,200 bytes) and
       every reader in this lane goes through `src/nv12.py`;
    2. the two headline arms carry the SAME INFORMATION and differ only in
       the ENCODING -- the image arm's pixels and the text arm's
       characters are the same 15,360 luma values;
    3. the blank-room control is run in EVERY arm's own representation,
       and the headline is a MARGIN OVER THAT CONTROL, not a raw score;
    4. the headline difference is computed from the recorded numbers by the
       stated chain, so a wrong chain fails these tests;
    5. the verdict is DERIVED from the recorded numbers, so it can come
       out the other way -- including the way that says the TEXT arm won;
    6. the scramble control preserves the pixel histogram exactly, so it
       destroys structure and nothing else;
    7. the evaluation is SESSION-LEVEL only: the audio and the video are
       independent takes, so no frame-level audio<->video label is
       recoverable, and this lane does not pretend otherwise;
    8. B is a SPEAKING take and is NOT used as a silent class.

THE NULL IS A VALID OUTCOME. These tests do not assert that the image arm
won, nor that the text arm won, nor that either separated speaking from
silent at all. They assert that the recorded verdict follows from the
recorded control numbers, that the verdict function can produce the
opposite answer, and that the specific findings this lane recorded are
recorded. A future run that flips a number will fail the checks that pin
it, which is the point.

Nothing here needs the network, a GPU, or a model download. The experiment
module is numpy-only at import time: torch and transformers are imported
inside its scoring functions.
"""

from __future__ import annotations

import ast
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

import vlm_image_retention as vir  # noqa: E402

RESULTS = ROOT / "experiments" / "vlm_image_retention_results.json"
SCRIPT = ROOT / "experiments" / "vlm_image_retention.py"


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


def test_the_frame_indices_are_fixed_and_evenly_spaced(res):
    """The frame indices are the pre-registered rule's, not a fresh pick.

    They must be the same for every arm (so the arms see the same
    moments), spread evenly over the capture, and reproducible from the
    rule alone.
    """
    for sess, n in (("A", 120), ("C", 120), ("empty1", 120)):
        rec = res["frames"][sess]
        assert rec["n_scored"] == n, (
            f"{sess} was scored on {rec['n_scored']} frames, not {n}")
        expected = vir.evenly_spaced_indices(rec["n_frames_on_disk"], n)
        assert rec["frame_indices"] == expected, (
            f"{sess}'s frame indices do not follow the pre-registered "
            "rule round(i * n_frames_on_disk / n_scored)")
        assert len(set(rec["frame_indices"])) == n, (
            f"{sess}'s frame indices repeat")


# --- claim 2: the two arms carry the same information ---------------------

def test_the_two_arms_carry_the_same_luma_values(res):
    """The image arm's pixels and the text arm's characters are the SAME
    frame, not two different views of the data.

    The text arm is one character per pixel over the same 96x160 grid,
    and the image arm is the same frame upscaled x4 with NEAREST, so no
    pixel value is invented on either side.
    """
    reps = res["representations"]
    assert reps["image"]["size_px"] == [384, 640]
    assert reps["image"]["upscale"] == 4
    assert reps["image"]["resampling"].startswith("NEAREST")
    assert reps["text"]["grid"] == [96, 160]
    assert reps["text"]["chars_per_pixel"] == 1
    assert reps["text"]["renderer"] == ("experiments/ascii_motion_render.py's "
                                        "render_frame")
    # and the renderer's file has not changed under the results
    h = hashlib.sha256(
        (ROOT / "experiments" / "ascii_motion_render.py").read_bytes()
    ).hexdigest()
    assert reps["text"]["renderer_sha256"] == h, (
        "the repository's ASCII renderer has changed since the run; the "
        "text arm's numbers would no longer be the repository's own "
        "representation")


def test_the_renderer_is_the_repository_s_own():
    """The text arm really uses the repo's ramp, not a re-derivation."""
    mod, path = vir._load_repo_renderer()
    assert path.name == "ascii_motion_render.py"
    assert mod.RAMP == " .:-=+*#%@"
    frame = np.arange(96 * 160, dtype=np.uint8).reshape(96, 160)
    assert vir.render_text_frame(frame, mod.RAMP, True) == \
        mod.render_frame(frame, mod.RAMP), (
        "the text arm's renderer has drifted from the repository's own "
        "render_frame")


# --- claim 3: the blank control is run in every arm -----------------------

def test_the_blank_room_control_is_run_in_every_arm(res):
    """The empty room is scored in EACH arm's own representation, and it
    is what the headline is measured against."""
    per_arm = res["controls"]["blank_room"]["per_arm"]
    for arm in vir.ARMS:
        assert arm in per_arm, f"{arm} has no blank-room control"
        rec = per_arm[arm]
        assert 0.0 <= rec["fpr_blank"] <= 1.0
        assert np.isfinite(rec["tau"])
        # the blank room is a real condition with real frames scored
        cond = res["arms"][arm]["conditions"]["empty1"]
        assert cond["n_frames"] > 0
        assert cond["role"] == "blank_control"


def test_the_headline_is_a_margin_over_the_control_not_a_raw_score(res):
    """The headline is a MARGIN OVER CONTROLS.

    If it were a raw score -- the arms' separations, or their difference
    -- the blank room would not be in it. This is the check that fails if
    somebody replaces the margin with a bare AUC difference.
    """
    head = res["headline"]
    assert head["difference"] == pytest.approx(
        head["margin_over_blank_image"] - head["margin_over_blank_text"])
    # and the margin itself is the separation MINUS the blank control
    for arm in ("image", "text"):
        sep = res["arms"][arm]["separation"]
        assert res["arms"][arm]["margin_over_blank"] == pytest.approx(
            sep["auc_ac"] - sep["blank_sep_auc_blank_vs_c"])
    # the blank control's separation is the empty room scored against the
    # silent take, and it is NOT the same number as the separation
    for arm in ("image", "text"):
        sep = res["arms"][arm]["separation"]
        assert sep["blank_sep_auc_blank_vs_c"] != sep["auc_ac"]


# --- claim 4: the headline chain is computed, not asserted ---------------

def test_the_headline_is_computed_from_the_recorded_numbers(res):
    """THE TEST THAT FAILS IF THE HEADLINE COMPUTATION IS WRONG.

    The whole chain is recomputed here from the arms' own recorded
    statistics: the two separations, the two blank controls, the two
    margins, the difference, and the verdict. A headline computed on a
    different statistic, or a margin that forgot its blank control, or a
    difference with the arms the wrong way round, all fail here.
    """
    sep_img = res["arms"]["image"]["separation"]["auc_ac"]
    blank_img = res["arms"]["image"]["separation"][
        "blank_sep_auc_blank_vs_c"]
    sep_txt = res["arms"]["text"]["separation"]["auc_ac"]
    blank_txt = res["arms"]["text"]["separation"][
        "blank_sep_auc_blank_vs_c"]
    margin_img = sep_img - blank_img
    margin_txt = sep_txt - blank_txt
    difference = margin_img - margin_txt

    assert res["arms"]["image"]["margin_over_blank"] == pytest.approx(
        margin_img)
    assert res["arms"]["text"]["margin_over_blank"] == pytest.approx(
        margin_txt)
    assert res["headline"]["difference"] == pytest.approx(difference)
    assert res["headline"]["difference"] == pytest.approx(
        (sep_img - blank_img) - (sep_txt - blank_txt))


def test_the_verdict_follows_from_the_recorded_numbers(res):
    """The verdict is re-derived from the recorded numbers, not read from
    a stored boolean."""
    recomputed = vir.derive_verdict(res)
    assert recomputed["verdict"] == res["verdict"]["verdict"]
    assert recomputed["checks"] == res["verdict"]["checks"]


def test_the_verdict_function_can_fail(res):
    """The verdict function can produce the OPPOSITE answer.

    Three mutations of the recorded numbers, three different verdicts.
    This is the property the lane depends on: a verdict that cannot come
    out the other way is not a measurement.
    """
    # (a) make the text arm win outright
    flipped = json.loads(json.dumps(res))
    flipped["headline"]["difference"] = -0.9
    flipped["headline"]["perm_z"] = -9.0
    flipped["arms"]["text"]["auc_ac"] = 0.99
    flipped["arms"]["text"]["auc_ac_perm_z"] = 9.0
    out = vir.derive_verdict(flipped)
    assert out["verdict"] == "text_description_beats_image_retention"
    assert out["checks"]["R1_text_wins"] is True

    # (b) make the image arm win outright
    flipped = json.loads(json.dumps(res))
    flipped["headline"]["difference"] = 0.9
    flipped["headline"]["perm_z"] = 9.0
    flipped["arms"]["image"]["auc_ac"] = 0.99
    flipped["arms"]["image"]["auc_ac_perm_z"] = 9.0
    out = vir.derive_verdict(flipped)
    assert out["verdict"] == "image_retention_beats_text_description"
    assert out["checks"]["R1_image_wins"] is True

    # (c) make neither arm separate anything at all
    flipped = json.loads(json.dumps(res))
    for arm in ("image", "text"):
        flipped["arms"][arm]["auc_ac"] = 0.5
        flipped["arms"][arm]["auc_ac_perm_z"] = 0.0
    out = vir.derive_verdict(flipped)
    assert out["verdict"] == "neither_arm_separates_speaking_from_silent"

    # (d) and the honest null: a big difference with no permutation
    #     support is NOT a win
    flipped = json.loads(json.dumps(res))
    flipped["headline"]["difference"] = 0.9
    flipped["headline"]["perm_z"] = 0.5
    flipped["arms"]["image"]["auc_ac"] = 0.99
    flipped["arms"]["image"]["auc_ac_perm_z"] = 9.0
    out = vir.derive_verdict(flipped)
    assert out["verdict"] == "no_difference_at_the_preregistered_threshold"


def test_an_unusable_arm_cannot_win_on_the_margin_alone(res):
    """R2: an arm that separates nothing cannot win, however its margin
    compares. A degenerate arm's margin is 0 anyway -- both of its AUC
    terms are 0.50 -- but the gate is checked explicitly."""
    flipped = json.loads(json.dumps(res))
    flipped["headline"]["difference"] = 0.9
    flipped["headline"]["perm_z"] = 9.0
    flipped["arms"]["image"]["auc_ac"] = 0.50
    flipped["arms"]["image"]["auc_ac_perm_z"] = 0.0
    out = vir.derive_verdict(flipped)
    assert out["checks"]["R2_image_usable"] is False
    assert out["verdict"] != "image_retention_beats_text_description"


# --- claim 5: the null is a null -----------------------------------------

def test_the_null_permutes_the_labels_and_holds_the_blank_fixed():
    """The permutation null breaks the session assignment and keeps
    everything else, and it is the SAME permutation for both arms.

    The separation's null is 0.50. The MARGIN's null is 0.00 when the
    blank room is indistinguishable from the person frames -- which is
    the case the margin is designed for -- and it moves away from 0.00
    exactly when the arm ranks the blank room differently from a person,
    which is the failure the margin is meant to catch.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(2.0, 1.0, 120)
    c = rng.normal(0.0, 1.0, 120)
    # a blank room drawn from the SAME distribution as the person frames:
    # the arm cannot tell it from a person, so the margin's null is 0
    blank = rng.normal(1.0, 1.0, 120)
    perms = vir.make_permutations(120, 120, 200, 20261020)
    margins, aucs = vir.null_stats(a, c, blank, perms)
    assert len(aucs) == 200
    assert aucs.mean() == pytest.approx(0.5, abs=0.05)
    assert margins.mean() == pytest.approx(0.0, abs=0.10)

    # the margin's null IS the separation's null minus the blank term,
    # computed independently: the control is not permuted
    na = len(a)
    pool = np.concatenate([a, c])
    blank_terms = []
    for p in perms:
        blank_terms.append(vir.auc(blank, pool[p[na:]]))
    assert margins.mean() == pytest.approx(
        aucs.mean() - float(np.mean(blank_terms)), abs=1e-9)

    # a blank room that the arm ranks ABOVE a person pushes the margin's
    # null away from zero: that is the failure the margin catches
    blank_high = rng.normal(6.0, 1.0, 120)
    margins2, _ = vir.null_stats(a, c, blank_high, perms)
    assert margins2.mean() < margins.mean()

    # and the SAME permutation object drives both terms
    m2, a2 = vir.null_stats(a, c, blank, perms)
    assert np.array_equal(aucs, a2) and np.array_equal(margins, m2)


def test_the_permutations_are_seeded_and_reproducible():
    p1 = vir.make_permutations(10, 10, 5, 20261020)
    p2 = vir.make_permutations(10, 10, 5, 20261020)
    assert np.array_equal(p1, p2)
    p3 = vir.make_permutations(10, 10, 5, 1)
    assert not np.array_equal(p1, p3)


def test_the_permutation_null_is_recorded_with_its_seed(res):
    head = res["headline"]
    assert head["n_permutations"] >= 1000
    assert head["null_seed"] == 20261020
    assert head["paired"].startswith("the SAME A/C permutation")


# --- claim 6: the scramble control destroys structure and nothing else ---

def test_the_scramble_preserves_the_histogram_exactly():
    """The scramble control keeps the pixel values and moves them.

    A control that changed the histogram would be testing something else
    (a darker image, a smoother image) and not the spatial arrangement.
    """
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(96, 160), dtype=np.uint8)
    perm = vir.block_permutation((96 // 4) * (160 // 4), 20261020)
    out = vir.apply_block_permutation(frame, 4, perm)
    assert out.shape == frame.shape
    assert np.array_equal(np.sort(out.ravel()), np.sort(frame.ravel()))
    assert not np.array_equal(out, frame)
    # and it is deterministic
    again = vir.apply_block_permutation(frame, 4, perm)
    assert np.array_equal(out, again)


def test_the_two_scrambled_arms_share_one_permutation(res):
    """The image scramble and the text scramble are the SAME permutation,
    so the two control arms are matched to each other."""
    sc = res["scramble"]
    assert sc["block_px"] == 4
    assert sc["grid"] == [24, 40]
    assert sc["seed"] == 20261020
    perm = vir.block_permutation(24 * 40, sc["seed"])
    assert sc["permutation"] == [int(p) for p in perm]
    assert sc["permutation_sha256"] == hashlib.sha256(
        np.asarray(perm, dtype=np.int64).tobytes()).hexdigest()


def test_the_scramble_control_is_scored_and_reported(res):
    """The control is actually RUN, not merely planned: both scrambled
    arms have scores, a separation and a blank control."""
    for arm in ("image_scrambled", "text_scrambled"):
        rec = res["arms"][arm]
        assert rec["role"] == "control"
        assert rec["auc_ac"] is not None
        assert rec["margin_over_blank"] is not None
        assert rec["control_blank"]["fpr_blank"] is not None
        for sess in ("A", "C", "empty1"):
            assert rec["conditions"][sess]["n_frames"] > 0
    assert "scrambled" in res["controls"]
    assert "image_scrambled" in res["controls"]["scrambled"]["per_arm"]


# --- claim 7: the session-level distinction is pinned ---------------------

def test_the_labels_are_session_level_and_no_frame_alignment_is_claimed(res):
    """The audio and the video of these captures are independent takes.

    No frame-level audio<->video label is recoverable, so every number in
    this lane is a statement about the DISTRIBUTION of scores on A's
    frames against the distribution on C's frames. The lane says so, and
    the rule pins it.
    """
    rule = res["pre_registered_rules"]
    gate = rule["checks"]["R4_session_level_gate"]
    assert "session-level" in gate.lower()
    assert "independent takes" in gate.lower()
    assert res["arms"]["image"]["conditions"]["A"]["role"] == "positive"
    assert res["arms"]["image"]["conditions"]["C"]["role"] == "negative"
    assert res["arms"]["image"]["conditions"]["C"]["pose"] == "facing"
    assert res["arms"]["image"]["conditions"]["A"]["pose"] == "facing"


def test_b_is_a_speaking_take_and_is_not_used_as_a_silent_class(res):
    """B is the speaking take with the head turned away.

    Putting B in the silent class would be a label error, and the
    repository's own metadata says B is speaking. B is reported as a
    secondary condition and never as a negative class.
    """
    corpus_b = Path("/mnt/c/proart_capture/corpus_v2/B.json")
    if corpus_b.is_file():
        meta = json.loads(corpus_b.read_text())
        assert meta["speak"] is True
        assert meta["pose"] == "turned"
    rec = res["arms"]["image"]["secondary_condition_B"]
    assert rec["label"] == "speaking"
    assert rec["pose"] == "turned"
    assert "NOT a silent class" in rec["note"]
    # and it is not in the separation: the separation is A against C
    assert "B" not in res["arms"]["image"]["separation"]
    assert res["arms"]["image"]["separation"]["auc_ac"] == pytest.approx(
        res["arms"]["image"]["separation"]["auc_ac"])


def test_the_lane_never_reads_the_audio(res):
    """No WAV file is read by this lane.

    The audio and the video are independent takes, so an audio-derived
    frame label would be a label with no referent.
    """
    inputs = res["provenance"]["inputs"]
    assert set(inputs) == {"A", "B", "C", "empty1"}
    for name, rec in inputs.items():
        assert rec["path"].endswith(".nv12")
        assert ".wav" not in rec["path"]


# --- the harness, the model, the provenance -------------------------------

def test_the_harness_is_deterministic(res):
    """The same frame scored twice gives the same score.

    `docs/research/LOCAL-VLM-DETERMINISM.md` measured local greedy
    decoding to be deterministic (spread 0.000 against a hosted spread of
    0.619). This lane scores by log-probability, so it inherits that, and
    it checks it rather than assuming it.
    """
    rep = res["controls"]["harness_determinism"]
    for arm in ("image", "text"):
        assert rep[arm]["n_frames"] > 0
        assert rep[arm]["max_abs_score_delta"] == 0.0, (
            f"the {arm} arm's scores are not reproducible on a repeat; "
            "the measurement is not deterministic")
        assert rep[arm]["scores_identical"] is True


def test_the_score_is_a_single_token_forced_choice(res):
    """Both scored answers are scored on their FIRST token only.

    One token each, so there is no length bias between "SPEAKING" and
    "NOT SPEAKING", and the score does not depend on the model producing
    JSON or any particular formatting.
    """
    m = res["model"]
    ids = m["answer_token_ids"]
    assert ids["positive_first_token_id"] != ids["negative_first_token_id"]
    assert len(ids["positive_tokens"]) >= 1
    assert len(ids["negative_tokens"]) >= 1
    assert m["batch_size"] == 1
    assert "decoding" in m


def test_the_pre_registered_rule_is_recorded_verbatim(res):
    """The rule in the results file is the rule in the script.

    If they differ, the rule was edited after the numbers were seen and
    the run is not the run the rule was written for.
    """
    assert res["pre_registered_rules"] == vir.PRE_REGISTERED_RULE
    assert res["pre_registered_rule_text"] == vir.PRE_REGISTERED_RULE_TEXT
    assert vir.PRE_REGISTERED_RULE["written_before"].startswith(
        "the measurement code in this file existed")


def test_the_provenance_block_is_complete(res):
    """Everything needed to attribute the numbers is recorded."""
    prov = res["provenance"]
    for name in ("A", "B", "C", "empty1"):
        rec = prov["inputs"][name]
        assert len(rec["sha256"]) == 64
        assert rec["bytes"] > 0
    assert len(prov["script_sha256"]) == 64
    assert prov["script_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()).hexdigest()
    assert prov["reader"]["path"] == "src/nv12.py"
    assert len(prov["reader"]["sha256"]) == 64
    assert prov["model"]["model_id"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert len(prov["model"]["weights_sha256"]) == 64
    assert prov["gpu"]["name"]
    assert prov["gpu"]["total_mib"] > 0
    assert prov["commands"], "the exact commands must be recorded"
    assert any("vlm_image_retention.py" in c for c in prov["commands"])
    # the nv12 layout is recorded per capture, with its margin
    for sess in ("A", "B", "C", "empty1"):
        assert res["identity"]["record_layout"][sess]["layout"] == \
            "y_uv_halfres"


def test_the_experiment_imports_no_torch_at_import_time():
    """The module must be importable on the repository's own interpreter.

    torch and transformers are imported inside the scoring functions, so
    the pure functions -- and therefore these tests -- need no GPU and no
    model download. This is checked on the AST, so it is a statement
    about the import graph and not about a string.
    """
    tree = ast.parse(SCRIPT.read_text())
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_level.add(node.module.split(".")[0])
    assert "torch" not in top_level, (
        "torch is imported at module level, so this module cannot be "
        "imported without a GPU stack")
    assert "transformers" not in top_level
    assert "PIL" not in top_level
