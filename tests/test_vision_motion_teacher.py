"""
Tests for experiments/vision_motion_teacher.py -- does a SEQUENCE of frames
beat a single frame?

WHY THESE TESTS EXIST.

The headline of that experiment is a DIFFERENCE between two stimuli that
differ only in the ORDER of their tiles: `strip` (consecutive frames, in time
order) against `strip_shuffled` (the same frames, scrambled). That is a
delicate comparison, and a comparison this delicate has four ways of
producing a number that looks like a result and is not:

1. THE CONTROL IS NOT A CONTROL. If the shuffled strip quietly ended up
   holding different frames, or the same frames in the same order, then the
   "motion effect" it measured would be a content effect. The builders are
   tested directly here: the shuffled strip must hold exactly the same frames
   as the ordered one, in a different order, and the repeated strip must hold
   exactly one distinct frame.

2. THE HEADLINE IS HARD-CODED. A headline written into the results file by
   hand, or one that cannot move when the numbers underneath it move, is a
   claim and not a measurement. The headline is recomputed here from the
   recorded per-condition numbers, and the verdict function is fed perturbed
   inputs to prove it can change its mind.

3. THE DENOMINATORS ARE UNEQUAL. The project has shipped a number that came
   from two conditions with different frame counts. The results file has to
   carry its denominators, and they have to be equal.

4. THE FRAMES ARE TORN. The captures on disk are 19,200 bytes per frame
   (planar chroma), not the 23,040 that a standard 4:2:0 read assumes.
   Reading them at the wrong stride produces frames that do not exist, and
   it does so SILENTLY: the image looks like a plausible webcam frame and
   the "motion" between frames is a tear. Every number that touches those
   frames has to name the stride it was read at, so the stride and the mean
   consecutive |diff| are checked into the results file and checked here.

The tests also pin the IDENTITY fields, because this project's standing rule
is that a number whose inputs are not recorded is not a result: the model, the
request shape, the source file for every frame, the frame index, the stimulus
type, K, the upscale factor, the question text, the counts and the
denominators.

NO NETWORK. Every test here runs offline. The builder tests use synthetic
frames; the results-file tests read the committed JSON; the stride test reads
a local capture file and is SKIPPED when the capture drive is not attached.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import vision_motion_teacher as vmt  # noqa: E402

RESULTS = ROOT / "experiments" / "vision_motion_teacher_results.json"
CAPTURE = Path("/mnt/c/proart_capture/A.nv12")


@pytest.fixture(scope="module")
def results():
    assert RESULTS.is_file(), (
        f"{RESULTS} is missing. The results file is part of the deliverable: "
        f"a number whose inputs are not recorded is not a result.")
    import json
    return json.loads(RESULTS.read_text())


# -- synthetic fixtures -----------------------------------------------------

def _frames(k, h=96, w=160):
    """k frames, each a different flat grey so a tile is identifiable."""
    return [np.full((h, w), 10 + 20 * i, np.uint8) for i in range(k)]


def _pair_names(stimulus):
    """The (ordered, shuffled) partner of a stimulus name."""
    if stimulus.endswith("_shuffled"):
        return stimulus[: -len("_shuffled")], stimulus
    return stimulus, stimulus + "_shuffled"


# -- 1. the results file exists and carries its identity ---------------------

def test_results_file_has_identity_fields(results):
    for key in ("question_text", "request_shape", "image_construction",
                "frame_geometry", "conditions", "denominators",
                "degenerate_baselines", "models", "headlines", "calls",
                "models_probed"):
        assert key in results, f"results file is missing {key!r}"

    shape = results["request_shape"]
    assert shape["body_template"]["max_tokens"] == 200
    assert shape["body_template"]["temperature"] == 0.0
    body = shape["body_template"]
    content = body["messages"][0]["content"]
    assert [c["type"] for c in content] == ["text", "image_url"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert results["question_text"].strip(), "the question text must be recorded"


def test_every_call_records_its_inputs(results):
    """A number whose inputs are not recorded is not a result."""
    calls = results["calls"]
    assert calls, "there are no recorded calls"
    for c in calls:
        for key in ("model", "condition", "stimulus", "k", "scale", "order",
                    "moment", "source_video", "png_sha256",
                    "image_width", "image_height", "tile_source_frames",
                    "tile_multiset_sha256", "tile_order_sha256", "text",
                    "label", "error"):
            assert key in c, f"a call record is missing {key!r}: {c}"
        assert c["k"] >= 1
        assert c["scale"] >= 1
        assert c["condition"] in vmt.CONDITIONS
        assert c["source_video"], "every frame must name its source file"
        assert len(c["tile_source_frames"]) == c["k"], (
            "the tile count must be K: the identity of every frame in the "
            "stimulus has to be recorded")


# -- 2. the degenerate baselines are recorded -------------------------------

def test_degenerate_baselines_are_recorded(results):
    deg = results["degenerate_baselines"]
    assert "always_SPEAKING" in deg and "always_NOT_SPEAKING" in deg
    always = deg["always_SPEAKING"]
    never = deg["always_NOT_SPEAKING"]
    assert always["fpr_on_silent"] == 1.0
    assert always["fpr_on_blank"] == 1.0
    assert always["sensitivity_on_speak"] == 1.0
    assert never["sensitivity_on_speak"] == 0.0
    assert never["fpr_on_silent"] == 0.0
    # Both constant answers score a margin of exactly zero, which is the
    # whole reason the margin is the headline and the sensitivity alone is
    # not.
    assert always["margin_over_silent"] == 0.0
    assert never["margin_over_silent"] == 0.0


def test_base_rate_note_is_present(results):
    assert "base_rate_note" in results
    assert "0.4545" in results["base_rate_note"]


# -- 3. the denominators are equal ------------------------------------------

def test_denominators_are_equal_across_conditions(results):
    per = results["denominators"]["moments_per_condition"]
    assert set(per) == set(vmt.CONDITIONS)
    assert len(set(per.values())) == 1, (
        f"conditions have different denominators: {per}. A rate over a "
        f"denominator nobody can see is not a comparable rate.")
    assert results["denominators"]["equal_across_conditions"] is True


def test_scored_denominators_match_across_conditions(results):
    """The denominators must be equal in the SCORED numbers, not just in the
    declaration."""
    for model, stim in results["models"].items():
        for name, row in stim.items():
            counts = {c: row["conditions"][c]["n_stimuli"]
                      for c in vmt.CONDITIONS}
            assert len(set(counts.values())) == 1, (
                f"{model}/{name}: conditions scored different numbers of "
                f"stimuli: {counts}")
            for c in vmt.CONDITIONS:
                assert row["conditions"][c]["denominator"] == counts[c]


# -- 4. the stimulus construction is correct --------------------------------

def test_shuffled_strip_holds_the_same_frames_in_a_different_order():
    """THE CONTROL IS A CONTROL.

    The shuffled strip must contain exactly the frames of the ordered strip --
    same pixels, same count -- in a DIFFERENT order. If it held different
    frames, the headline would be measuring content, not motion.
    """
    frames = _frames(8)
    spec_id = {"k": 4, "scale": 2, "order": "identity"}
    spec_sh = {"k": 4, "scale": 2, "order": "shuffled"}
    perm = vmt.shuffle_order(4, np.random.default_rng(1))

    ordered, order_id = vmt.make_tiles(frames[:4], spec_id, perm)
    shuffled, order_sh = vmt.make_tiles(frames[:4], spec_sh, perm)

    assert order_id == [0, 1, 2, 3]
    assert sorted(order_sh) == [0, 1, 2, 3]
    assert order_sh != order_id, "the shuffled strip must differ in order"
    assert ordered.shape == shuffled.shape, (
        "the ordered and shuffled strips must be the SAME SIZE, so the "
        "comparison cannot be an image-size effect")
    # Same frames: the order-independent hash of the tiles is equal, and the
    # order-dependent one is not.
    tiles_id = [vmt.upscale(frames[j], 2) for j in order_id]
    tiles_sh = [vmt.upscale(frames[j], 2) for j in order_sh]
    assert vmt.tile_multiset_hash(tiles_id) == vmt.tile_multiset_hash(tiles_sh)
    assert vmt.tile_order_hash(tiles_id) != vmt.tile_order_hash(tiles_sh), (
        "the two orders must hash differently, or the order hash cannot "
        "prove that the order changed")
    # and the actual pixel content of each tile is the source frame
    for j, idx in enumerate(order_sh):
        tile = shuffled[:, j * 160 * 2:(j + 1) * 160 * 2]
        assert np.array_equal(tile, vmt.upscale(frames[idx], 2))


def test_repeated_strip_holds_one_distinct_frame():
    """The second control: K copies of ONE frame. No motion at all, and only
    ONE distinct image in the strip."""
    frames = _frames(8)
    spec = {"k": 4, "scale": 2, "order": "repeated"}
    img, order = vmt.make_tiles(frames[:4], spec, None)
    assert order == [0, 0, 0, 0]
    tile = img[:, 0:320]
    for j in range(1, 4):
        assert np.array_equal(img[:, j * 320:(j + 1) * 320], tile), (
            "every tile of the repeated strip must be the same frame")
    assert np.array_equal(tile, vmt.upscale(frames[0], 2))
    # and the repeated strip differs from the ordered one
    ordered, _ = vmt.make_tiles(frames[:4],
                                {"k": 4, "scale": 2, "order": "identity"}, None)
    assert not np.array_equal(img, ordered)


def test_shuffle_order_is_neither_identity_nor_a_rotation():
    """A rotation preserves k-1 of the k adjacencies, so it is not a
    motion-destroyed control. The builder must reject it."""
    rng = np.random.default_rng(7)
    for k in (4, 8):
        perm = vmt.shuffle_order(k, rng)
        assert sorted(perm) == list(range(k))
        assert perm != list(range(k))
        rotations = {tuple((i + r) % k for i in range(k)) for r in range(k)}
        assert tuple(perm) not in rotations


def test_build_stimulus_rejects_the_identity_as_a_control():
    frames = _frames(4)
    with pytest.raises(ValueError):
        vmt.build_stimulus(frames, 4, 2, "shuffled", list(range(4)))
    with pytest.raises(ValueError):
        vmt.build_stimulus(frames, 4, 2, "shuffled", [0, 0, 1, 2])


def test_ordered_and_shuffled_strips_in_the_results_hold_the_same_frames(results):
    """The committed run must actually contain the control it claims.

    For every model, condition and moment, an ordered strip and its shuffled
    partner must carry the SAME order-independent tile hash and DIFFERENT
    order-dependent tile hashes. That is the machine-checkable form of
    "appearance preserved, motion destroyed".
    """
    calls = results["calls"]
    assert calls
    by_key = {}
    for c in calls:
        key = (c["model"], c["condition"], c["stimulus"], c["moment"])
        assert key not in by_key, f"duplicate call record for {key}"
        by_key[key] = c

    checked = 0
    for (model, cond, stim, moment), rec in list(by_key.items()):
        if not stim.endswith("_shuffled"):
            continue
        ordered_name, shuffled_name = _pair_names(stim)
        other = by_key.get((model, cond, ordered_name, moment))
        assert other is not None, (
            f"{model}/{cond}/{moment}: the ordered partner of {stim} is "
            f"missing from the results")
        assert rec["tile_multiset_sha256"] == other["tile_multiset_sha256"], (
            f"{model}/{cond}/{moment}/{stim}: the shuffled strip does NOT "
            f"hold the same frames as the ordered strip")
        if cond == "blank":
            # The blank condition's tiles are all the same uniform frame, so
            # the ordered and the shuffled stimulus ARE the same image. That
            # is true by construction and is recorded in the results file.
            assert rec["png_sha256"] == other["png_sha256"]
            continue
        assert rec["tile_order_sha256"] != other["tile_order_sha256"], (
            f"{model}/{cond}/{moment}/{stim}: the shuffled strip has the "
            f"same ORDER as the ordered strip")
        assert rec["png_sha256"] != other["png_sha256"], (
            f"{model}/{cond}/{moment}/{stim}: the two stimuli are "
            f"byte-identical images")
        assert rec["image_width"] == other["image_width"]
        assert rec["image_height"] == other["image_height"]
        assert rec["k"] == other["k"]
        checked += 1
    assert checked, "no ordered/shuffled pairs were found in the results"


def test_strip_k4_window_is_a_prefix_of_strip_k8(results):
    """The two K arms are NESTED, so the K comparison is about the number of
    frames and not about different moments."""
    calls = results["calls"]
    by_key = {}
    for c in calls:
        by_key.setdefault((c["model"], c["condition"], c["moment"],
                           c["stimulus"]), []).append(c)
    checked = 0
    for (model, cond, moment, stim), group in by_key.items():
        if stim not in ("strip_k8", "strip_k8_torn"):
            continue
        four_name = ("strip_k4" if stim == "strip_k8" else "strip_k4_torn")
        eight = group[0]
        four = by_key.get((model, cond, moment, four_name))
        if not four:
            continue
        assert four[0]["tile_source_frames"] == \
            eight["tile_source_frames"][:4], (
            f"{model}/{cond}/{moment}/{stim}: the K=4 window is not the first "
            f"four frames of the K=8 window")
        checked += 1
    assert checked, "no nested K=4/K=8 pairs were found in the results"


# -- 5. the stride, and the torn control ------------------------------------

def test_results_record_the_stride_and_the_frame_statistics(results):
    """Every number must carry the identity of the pixels it came from."""
    geom = results["frame_geometry"]
    assert geom["measurement_stride_bytes"] == 19200
    assert geom["control_stride_bytes"] == 23040
    assert geom["frame_size"] == [160, 96]
    for cond, s in geom["sessions"].items():
        assert s["clean_read"]["n_frames"] == 900, (
            f"{cond}: the clean read must give 900 frames at stride 19200")
        assert s["torn_read"]["n_frames"] == 750, (
            f"{cond}: the wrong-stride read must give 750 frames")
        clean = s["clean_read"]["mean_abs_consecutive_diff"]
        torn = s["torn_read"]["mean_abs_consecutive_diff"]
        assert clean < 2.0, (
            f"{cond}: a real 30 fps scene must not change by {clean} grey "
            f"levels between consecutive frames")
        assert torn > 20.0, (
            f"{cond}: the wrong-stride read must change by tens of grey "
            f"levels -- that is the tear")


def test_the_torn_stimuli_are_marked_and_are_not_the_measurement(results):
    """The torn read is a CONTROL. It must be labelled as one, in the file
    and in the stimulus specs."""
    torn = [s for s in vmt.STIMULUS_SPECS if s.get("source") == "torn"]
    assert torn, "there must be a torn-strip control"
    for spec in vmt.STIMULUS_SPECS:
        if spec.get("source") != "torn":
            continue
        assert spec["name"].endswith("_torn") or "_torn_" in spec["name"], (
            f"{spec['name']}: a torn stimulus must be named as one")
    # and the frames it used are recorded as torn frames
    for c in results["calls"]:
        if c["condition"] == "blank":
            # the blank condition has no frames at all, torn or otherwise
            continue
        if c["stimulus"].endswith("_torn") or "_torn_" in c["stimulus"]:
            assert all(f.startswith("torn@") for f in c["tile_source_frames"]), (
                f"{c['stimulus']}: the torn stimulus must record torn frame "
                f"indices, not clean ones")
        else:
            assert not any(f.startswith("torn@")
                           for f in c["tile_source_frames"]), (
                f"{c['stimulus']}: a clean stimulus must not record torn "
                f"frame indices")


def test_stride_reader_reads_the_right_number_of_frames():
    """The reader, on the real capture, if the capture drive is attached."""
    if not CAPTURE.is_file():
        pytest.skip("capture drive not attached")
    clean = vmt.load_y_strided(CAPTURE, vmt.STRIDE_CORRECT)
    torn = vmt.load_y_strided(CAPTURE, vmt.STRIDE_TORN)
    assert clean.shape == (900, 96, 160)
    assert torn.shape == (750, 96, 160)
    assert vmt.frame_stats(clean)["mean_abs_consecutive_diff"] < 2.0
    assert vmt.frame_stats(torn)["mean_abs_consecutive_diff"] > 20.0


# -- 6. the headline is computed, not hardcoded ------------------------------

def test_headline_is_recomputed_from_the_recorded_numbers(results):
    """The stored headline must equal the headline recomputed from the
    per-condition numbers it claims to come from."""
    checked = 0
    for model, heads in results["headlines"].items():
        stim = results["models"][model]
        for kk, head in heads.items():
            s = stim[head["ordered_stimulus"]]
            sh = stim[head["shuffled_control"]]
            margin_s = (s["sensitivity_on_speak"]
                        - s["fpr_on_silent_person_present"])
            margin_sh = (sh["sensitivity_on_speak"]
                         - sh["fpr_on_silent_person_present"])
            assert head["margin_delta_strip_minus_shuffled"] == pytest.approx(
                margin_s - margin_sh), (
                f"{model}/{kk}: the stored headline is not the difference of "
                f"the recorded margins")
            sens_delta = (s["sensitivity_on_speak"]
                          - sh["sensitivity_on_speak"])
            assert head["sensitivity_delta_strip_minus_shuffled"] == \
                pytest.approx(sens_delta)
            assert head["strip"]["margin_over_silent"] == pytest.approx(
                margin_s)
            assert head["strip_shuffled"]["margin_over_silent"] == \
                pytest.approx(margin_sh)
            checked += 1
    assert checked, "no headlines were found in the results"


def test_the_verdict_function_can_change_its_mind(results):
    """A verdict that cannot come out the other way is not a measurement.

    Feed the headline function perturbed numbers and require the verdict and
    the margin delta to MOVE. A hard-coded verdict would not move.
    """
    model = next(iter(results["models"]))
    base = results["models"][model]

    # GOOD: the ordered strip is the better one, and BOTH strips run at a
    # matched false-positive rate (0 of 12), so the verdict is about motion
    # rather than about a difference in prompt-scale response.
    good = copy.deepcopy(base)
    good["strip_k4"]["sensitivity_on_speak"] = 1.0
    good["strip_k4"]["fpr_on_silent_person_present"] = 0.0
    good["strip_k4_shuffled"]["sensitivity_on_speak"] = 0.0
    good["strip_k4_shuffled"]["fpr_on_silent_person_present"] = 0.0
    h_good = vmt.headline_for(good)

    bad = copy.deepcopy(base)
    bad["strip_k4"]["sensitivity_on_speak"] = 0.0
    bad["strip_k4"]["fpr_on_silent_person_present"] = 0.0
    bad["strip_k4_shuffled"]["sensitivity_on_speak"] = 1.0
    bad["strip_k4_shuffled"]["fpr_on_silent_person_present"] = 0.0
    h_bad = vmt.headline_for(bad)

    dg = h_good["k4"]["margin_delta_strip_minus_shuffled"]
    db = h_bad["k4"]["margin_delta_strip_minus_shuffled"]
    assert dg != db, "the headline did not move when the numbers moved"
    assert dg > db
    assert h_good["k4"]["fpr_matched"] is True
    assert h_bad["k4"]["fpr_matched"] is True
    assert h_good["k4"]["verdict"] != h_bad["k4"]["verdict"]
    assert "MOTION HELPS" in h_good["k4"]["verdict"]
    assert "NO MOTION EFFECT" in h_bad["k4"]["verdict"]


def test_no_secret_in_the_results_file(results):
    """No secrets in files.

    The request shape IS recorded, and the shape includes the word "Bearer",
    so the check is that no real credential is present: the token value must
    not appear, and the only thing after "Bearer" may be the placeholder.
    """
    import os
    import re
    text = RESULTS.read_text()
    token = os.environ.get("CF_TOKEN")
    if token:
        assert token not in text
    for match in re.findall(r"Bearer\s+([A-Za-z0-9_\-.<>]+)", text):
        assert match == "<CF_TOKEN>", (
            f"the results file carries something after 'Bearer': {match!r}")
    for marker in ("CF_TOKEN=", "api_key", "apiKey", "secret"):
        assert marker not in text, (
            f"the results file contains {marker!r}")
