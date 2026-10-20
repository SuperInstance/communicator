"""
Tests for the corpus capture plan and its frame maths,
`experiments/corpus_capture_win.py`.

WHY THIS EXISTS. The corpus is the binding constraint of the project: the
teacher needs >= 63 INDEPENDENT sessions per class, and every session is one
entry of this plan. A malformed plan does not fail -- it produces a corpus
that is full and quietly wrong, with the takes labelled as conditions they
were not performed as. The condition set is small and the labels are the
whole point:

    A  facing,  speaking    the poem, to the camera
    B  turned,  speaking    the poem, head fully away -- SAME WORDS as A
    C  facing,  silent      present, mouth closed, nothing said
    D  turned,  silent      present and turned away, nothing said
    F  free                 no label asserted; the teacher labels these

D IS THE ONE THAT MATTERS MOST and is the one that has never been recorded.
Without it, "the model sees the head turn" and "the model sees the mouth
moving" are the same hypothesis, and no amount of data on A and B can
separate them. A plan edit that quietly drops D -- a rename, a reordering, a
filter that keeps only the speaking conditions -- would leave the corpus
looking complete while removing the comparison that de-confounds pose from
speech. So the test below asserts that a turned-and-SPEAKING condition AND a
turned-and-SILENT condition both exist, by name.

The other thing pinned here is the arithmetic of the stored frames. The
corpus stores row-band-removed change maps, quantised to uint8. Each of
those three steps can fail quietly:

  - `row_band_remove` must actually remove the LED flicker banding (a
    per-row constant offset). If it does not, the stored maps are dominated
    by the lighting and the subject's motion is a rounding error.
  - `change_maps` must be all zeros for a scene that did not move, and
    non-zero for one that did. A change map that is non-zero on a static
    scene is noise; one that is zero on a moving scene is a dead channel.
  - `quantise` must CLIP at 255. It must not wrap: a value above the scale
    that wrapped to 0 would be recorded as "no motion", which is worse than
    recording too much motion, because it is indistinguishable from a
    genuine zero.

None of these tests need the real captures: the fixtures are synthetic
arrays, so the suite is runnable on any machine, with no capture drive
attached.
"""

from __future__ import annotations
import pytest as _pytest  # noqa: E402
_pytest.importorskip("numpy", reason="exercises code needing numpy")  # noqa: E402

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import corpus_capture_win as cap      # noqa: E402
from src import stimulus                                # noqa: E402


# What each condition letter must mean. The letter IS the label the corpus
# will carry, so this table is the contract the plan is held to.
EXPECTED_SHAPE = {
    "A": (True, "facing"),      # facing, speaking
    "B": (True, "turned"),      # turned, speaking  -- same words as A
    "C": (False, "facing"),     # facing, silent
    "D": (False, "turned"),     # turned, silent    -- de-confounds pose/speech
    "F": (None, "any"),         # free session, no label asserted
    "FREE": (None, "any"),      # the trailing long free session
}


def base_name(name: str) -> str:
    """'A1' -> 'A'; 'FREE' -> 'FREE'."""
    if name.startswith("FREE"):
        return "FREE"
    return name.rstrip("0123456789")


# -- the shape of the plan ----------------------------------------------------

class TestPlanShape:
    def test_every_condition_appears_exactly_once(self):
        """
        A1..A3, B1..B3, C1..C3, D1..D3, F1..F3 and FREE: sixteen entries, each
        once. A duplicate is as bad as a gap -- two takes called A2 are two
        takes the teacher cannot tell apart, and one missing A2 is a
        repetition that was never recorded.
        """
        names = [p["name"] for p in cap.PLAN]
        expected = {f"{letter}{rep}" for letter in "ABCDF"
                    for rep in (1, 2, 3)} | {"FREE"}
        assert len(names) == len(set(names)), (
            f"duplicate entries in the plan: "
            f"{sorted({n for n in names if names.count(n) > 1})}")
        assert set(names) == expected, (
            f"missing {sorted(expected - set(names))}, "
            f"unexpected {sorted(set(names) - expected)}")
        assert len(cap.PLAN) == len(expected)

    def test_the_plan_repeats_each_condition_enough_times_to_be_independent(self):
        """
        The repetition count is the parameter that unlocks everything
        downstream: a teacher scored 1.000 on a temporal split of a single
        session and 0.000 on leave-one-session-out, because it had learned
        the room rather than speech. Three is the documented minimum.
        """
        assert cap.REPEATS >= 3, (
            f"REPEATS is {cap.REPEATS}; fewer than three independent sessions "
            f"per condition cannot be split leave-one-session-out")
        lettered = [k for k in EXPECTED_SHAPE if k != "FREE"]
        assert len(cap.PLAN) == cap.REPEATS * len(lettered) + 1
        for letter in "ABCDF":
            reps = sorted(p["rep"] for p in cap.PLAN
                          if base_name(p["name"]) == letter)
            assert reps == list(range(1, cap.REPEATS + 1)), (
                f"condition {letter} has reps {reps}")

    def test_the_trailing_free_session_is_the_last_entry(self):
        """
        The corpus the teacher will label is one long free session, and it
        runs at the end -- after the labelled conditions, so the teacher is
        benchmarked against ground truth before it is trusted with anything.
        """
        assert cap.PLAN[-1]["name"] == "FREE"
        assert cap.PLAN[-1].get("long") is True
        assert sum(1 for p in cap.PLAN if p.get("long")) == 1


# -- what each condition asserts ---------------------------------------------

class TestConditionSemantics:
    def test_each_letter_has_the_speech_and_pose_it_names(self):
        """
        The whole table at once:

          speak True  -> pose is 'facing' or 'turned' (concrete)
          speak False -> pose is 'facing' or 'turned' (concrete)
          speak None  -> pose is 'any' (free: nothing is asserted)

        A free session that claimed a pose, or a condition that claimed to be
        silent without saying which way the subject was facing, is a
        mislabelled take by construction.
        """
        for p in cap.PLAN:
            letter = base_name(p["name"])
            assert letter in EXPECTED_SHAPE, f"unknown condition {p['name']}"
            want_speak, want_pose = EXPECTED_SHAPE[letter]
            assert p["speak"] is want_speak, (
                f"{p['name']}: speak is {p['speak']!r}, expected {want_speak!r}")
            assert p["pose"] == want_pose, (
                f"{p['name']}: pose is {p['pose']!r}, expected {want_pose!r}")
            if p["speak"] is not None:
                assert p["pose"] in ("facing", "turned"), (
                    f"{p['name']} asserts speech={p['speak']} but has no "
                    f"concrete pose")

    def test_turned_speaking_and_turned_silent_both_exist(self):
        """
        THE regression guard. D -- turned and SILENT -- has never been
        recorded, and without it "the model sees the head turn" and "the
        model sees the mouth moving" are the same hypothesis: a classifier
        that keys on any visual change at all would score the same on A-vs-B
        and on C-vs-D, and the pose test would be worthless.

        So the plan must carry BOTH a turned-and-SPEAKING condition (B) and a
        turned-and-SILENT one (D). A plan that keeps only the speaking
        conditions must fail this test.
        """
        turned_speaking = [p for p in cap.PLAN
                           if p["pose"] == "turned" and p["speak"] is True]
        turned_silent = [p for p in cap.PLAN
                         if p["pose"] == "turned" and p["speak"] is False]
        assert turned_speaking, (
            "no turned-and-SPEAKING condition in the plan: A-vs-B can no "
            "longer be separated from any other visual difference")
        assert turned_silent, (
            "no turned-and-SILENT condition (D) in the plan: pose and speech "
            "are confounded and the head turn is unidentifiable")
        assert {base_name(p["name"]) for p in turned_speaking} == {"B"}
        assert {base_name(p["name"]) for p in turned_silent} == {"D"}
        assert len(turned_silent) == cap.REPEATS, (
            f"D is recorded {len(turned_silent)} times, not {cap.REPEATS}")

    def test_every_condition_has_a_note_a_duration_and_a_rep(self):
        """
        Every entry must be self-describing: the note goes into the take's
        metadata, the duration is what the capture loop runs, and the rep
        index is what makes the repetitions countable.
        """
        for p in cap.PLAN:
            assert isinstance(p.get("note"), str) and p["note"].strip(), \
                f"{p['name']}: no note"
            assert isinstance(p.get("seconds"), (int, float)), \
                f"{p['name']}: seconds is {p.get('seconds')!r}"
            assert p["seconds"] > 0, f"{p['name']}: seconds is {p['seconds']}"
            assert p.get("rep") is not None, f"{p['name']}: no rep"
            assert isinstance(p["kind"], str) and p["kind"] in (
                "condition", "free"), f"{p['name']}: kind is {p['kind']!r}"

    def test_speaking_conditions_outlast_the_stimulus(self):
        """
        A speaking take must be at least RECOMMENDED_SECONDS long, because
        that is the window the stimulus was timed for: the poem needs
        ESTIMATED_SECONDS, and the recommendation leaves room for the
        opening and the trailing silence. A window shorter than the poem
        truncates the recitation, and the truncated tail is read by the gate
        as the end of the utterance rather than the end of the take.
        """
        speaking = [p for p in cap.PLAN if p["speak"] is True]
        assert speaking, "the plan has no speaking conditions"
        for p in speaking:
            assert p["seconds"] >= stimulus.RECOMMENDED_SECONDS, (
                f"{p['name']}: {p['seconds']}s is shorter than the "
                f"{stimulus.RECOMMENDED_SECONDS}s the stimulus needs")


# -- the frame maths ----------------------------------------------------------

class TestRowBandRemoval:
    def test_row_band_remove_removes_a_constant_row_offset(self):
        """
        The camera's auto-exposure makes horizontal LED-flicker bands: a
        per-row constant offset, which flips 50% of an empty room's dots per
        frame. Removing it at capture time is what keeps the stored maps
        dominated by the subject rather than by the lighting.

        The fixture is a frame plus a random per-row offset. After the
        removal every row must have the same median again, and the result
        must be the un-banded image up to one global constant per frame --
        i.e. the offset is gone and nothing else was changed.
        """
        rng = np.random.default_rng(11)
        base = rng.normal(0.0, 10.0, size=(3, cap.H, cap.W))
        band = rng.normal(0.0, 25.0, size=cap.H)
        banded = base + band[None, :, None]

        # the banding is really there, or the test proves nothing
        assert np.median(banded, axis=2).std(axis=1).min() > 5.0, \
            "the synthetic fixture has no visible banding"

        out = cap.row_band_remove(banded)

        row_medians = np.median(out, axis=2)          # (frames, rows)
        spread = row_medians.max(axis=1) - row_medians.min(axis=1)
        assert spread.max() < 1e-9, (
            f"the row offsets survived the removal (spread {spread.max()})")

        reference = cap.row_band_remove(base)
        delta = out - reference
        per_frame = delta.max(axis=(1, 2)) - delta.min(axis=(1, 2))
        assert per_frame.max() < 1e-9, (
            "the removal changed more than the row-constant component")

    def test_row_band_remove_keeps_the_shape_of_the_frame(self):
        """The subject's column structure must survive the removal."""
        rng = np.random.default_rng(3)
        frame = rng.normal(0.0, 10.0, size=(1, cap.H, cap.W))
        frame[:, :, 20:40] += 60.0                      # a bright column block
        out = cap.row_band_remove(frame)
        assert out.shape == frame.shape
        cols = out[0].mean(axis=0)
        assert cols[20:40].mean() > cols[:20].mean() + 20.0, \
            "the subject's column structure was removed with the banding"


class TestChangeMaps:
    def test_a_static_scene_produces_all_zeros(self):
        """
        A scene that did not move must produce an all-zero change map. If it
        does not, the stored corpus is full of lighting noise and every
        downstream number is measuring the flicker.
        """
        rng = np.random.default_rng(5)
        still = rng.normal(0.0, 10.0, size=(1, cap.H, cap.W))
        scene = np.repeat(still, 5, axis=0)
        maps = cap.change_maps(scene)
        assert maps.shape == (4, cap.MAP_H, cap.MAP_W)
        assert np.abs(maps).max() == 0.0, \
            "a static scene produced a non-zero change map"

    def test_a_moving_edge_produces_a_non_zero_map_in_the_right_place(self):
        """
        The other direction: something that DID move must be seen, and seen
        where it moved. The fixture is a bright bar that walks across an
        otherwise constant frame, so the only non-zero columns of the map
        are the ones the bar vacated and the ones it reached.
        """
        scene = np.full((4, cap.H, cap.W), 40.0)
        for t in range(scene.shape[0]):
            scene[t, :, 20 + 10 * t: 30 + 10 * t] = 220.0

        maps = cap.change_maps(scene)
        assert maps.shape == (3, cap.MAP_H, cap.MAP_W)
        assert np.abs(maps).max() > 0.0, \
            "a moving edge produced an all-zero change map"

        columns = np.flatnonzero(np.abs(maps).max(axis=(0, 1)) > 0)
        assert columns.size > 0
        # the bar travelled from columns 20 to 60 of 160; the map is 80 wide
        assert columns.min() >= 10 and columns.max() < 31, (
            f"the motion shows up in columns {columns.min()}..{columns.max()}, "
            f"not where the edge actually was")


class TestQuantise:
    def test_quantise_returns_uint8_and_a_positive_finite_scale(self):
        """
        The scale is stored next to the maps so nothing is silently clipped:
        the quantiser divides by it, and a zero or non-finite scale would
        make every map either all zero or all 255.
        """
        rng = np.random.default_rng(17)
        maps = rng.random((7, cap.MAP_H, cap.MAP_W)) * 30.0
        q, scale = cap.quantise(maps)
        assert q.dtype == np.uint8
        assert np.isfinite(scale)
        assert scale > 0.0
        assert q.shape == maps.shape
        assert q.max() <= 255

    def test_quantise_clips_at_255_and_does_not_wrap(self):
        """
        The failure this guards is a WRAP: a value above the scale that
        overflowed uint8 would land on 0, and a stored 0 is indistinguishable
        from "no motion at all". A clipping implementation saturates at 255,
        which is loudly different from the noise floor.
        """
        maps = np.zeros(10000, dtype=np.float64)
        maps[-1] = 1000.0
        q, scale = cap.quantise(maps)
        assert q.dtype == np.uint8
        assert q[-1] == 255, (
            f"the oversized value quantised to {q[-1]}, not 255 -- it may "
            f"have wrapped")
        assert q[:-1].max() == 0, "the quiet part of the map was not left at 0"

    def test_quantise_keeps_the_ordering_of_the_values(self):
        """Monotone input must give monotone output, so motion still ranks."""
        maps = np.array([0.0, 1.0, 2.0, 4.0, 8.0], dtype=np.float64)
        q, _scale = cap.quantise(maps)
        assert list(q) == sorted(q)
