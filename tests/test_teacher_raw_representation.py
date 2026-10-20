"""
Tests for the teacher-on-the-raw-representation sweep.

`docs/research/TEACHER-RAW-REPRESENTATION.md` is the document these tests
pin. It makes four claims, and each one can be false:

    1. the four corpora differ ONLY by the filter -- same captures, same
       frames, same layout, one filter each;
    2. the teacher does NOT do better on the raw representation: on the
       temporal split the filtered representations win the feature sets, and
       on the leave-one-session-out folds the raw representation loses most
       of the cells;
    3. NO representation beats the teacher's own controls -- the margin over
       the fold's own scrambled control is not positive on both folds for any
       feature set of any representation, and the blank control (a constant
       predictor) scores 1.000 on every fold;
    4. the STATIC-template control still scores 1.000 on fold A on every
       representation, including the raw one -- so the raw representation
       does not rescue the teacher from reading a picture of the camera.

The tests are mostly on the committed results, because the claims are about
measurements that took a minute each. The ones that can be checked cheaply
are checked directly against the code, so they fail if the MEASUREMENT
changes, not only if the RECORD of it changes.

Two of them are about the verdict being able to come out the other way:
`test_every_branch_of_the_verdict_is_reachable` runs the verdict function on
synthetic tables and shows that all four of its verdicts are reachable, and
`test_the_verdict_follows_the_pre_registered_rule` recomputes the verdict
from the numbers in the results file, so a verdict that was written by hand
rather than computed fails.

Nothing here needs the network, a GPU, or a model download.
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

import teacher_raw_representation as trr            # noqa: E402

RESULTS = ROOT / "experiments" / "teacher_raw_representation_results.json"

FILTERS = trr.FILTERS
FEATURE_SETS = trr.FEATURE_SETS
CORPUS_ROOT = trr.CORPUS_ROOT


@pytest.fixture(scope="module")
def results() -> dict:
    assert RESULTS.is_file(), f"{RESULTS} is missing"
    return json.loads(RESULTS.read_text())


@pytest.fixture(scope="module")
def table(results) -> dict:
    return results["table"]


def _corpora_on_disk() -> bool:
    return all((CORPUS_ROOT / f"corpus_filter_{f}" / "A.npz").is_file()
               for f in FILTERS)


# --- claim 0: the rule was fixed before the numbers were seen -------------

def test_the_results_file_carries_the_pre_registered_rule(results):
    """The rule is IN the results file, with its hash, and the rule says
    what the decision rule is before the numbers are read.

    A result with no pre-registered rule cannot be checked for having been
    written after the fact.
    """
    rule = results["pre_registered_rule"]
    assert rule["text"], "the results file carries no rule text"
    assert rule["rule_sha256"] == hashlib.sha256(
        rule["text"].encode()).hexdigest()
    # The rule has to name the task, the quantities, the thresholds and the
    # tie-break, or it is not a rule.
    lowered = rule["text"].lower()
    for needle in ("speaking", "margin_over_best_control", "at least 3 of the 5",
                   "static", "blank", "recommend raw"):
        assert needle in lowered, f"the rule does not mention {needle!r}"


def test_the_rule_in_the_results_is_the_rule_in_the_script(results):
    """The rule was not edited after the numbers were seen.

    The rule text in the results file must be byte-identical to the
    `PRE_REGISTERED_RULE` constant in the script, and the script's SHA-256
    must be the one the results file recorded. If the script was edited after
    the run -- to change the rule, or for any other reason -- this fails.
    """
    assert results["pre_registered_rule"]["text"] == trr.PRE_REGISTERED_RULE, (
        "the rule in the results file is not the rule in the script: the "
        "rule was edited after the numbers were seen")
    recorded = results["provenance"]["script_sha256"]
    on_disk = hashlib.sha256(
        (ROOT / "experiments" / "teacher_raw_representation.py").read_bytes()
    ).hexdigest()
    assert recorded == on_disk, (
        "the script has changed since the results were written, so the rule "
        "that produced them cannot be read")


def test_the_results_file_records_the_corrections_to_its_own_verdict(results):
    """The first implementation of the rule was stricter than the rule.

    It read "raw beats none of the three" as "raw wins ZERO feature sets
    against every filter", and it returned MIXED where the rule text returns
    FILTERED_IS_BETTER. The rule text was NOT changed; the implementation
    was corrected to match it. That correction is in the record, not
    swallowed.
    """
    corrections = results.get("corrections") or []
    assert corrections, "the results file records no corrections to itself"
    text = " ".join(c.get("what", "") + " " + c.get("effect", "")
                    for c in corrections)
    assert "raw beats none of the three" in text
    assert "MIXED" in text and "FILTERED_IS_BETTER" in text


# --- claim 1: the corpora differ ONLY by the filter ------------------------

def test_the_four_corpora_differ_only_by_the_filter(results):
    """Same captures, same frames, same layout, same everything -- except
    the filter.

    A sweep in which anything else changed is not a sweep over the filter,
    and the whole comparison would be meaningless.
    """
    eq = results["corpus_equivalence"]
    assert eq["ok"], f"the corpus equivalence check failed: {eq['legs']}"

    # (a) identity: for each session, every shared field is the same in all
    # four corpora.
    for field, blob in eq["legs"]["a_identity"]["per_field"].items():
        for session, sblob in blob["per_session"].items():
            assert sblob["identical_across_the_four_corpora"], (
                f"{field} differs between the corpora for session {session}: "
                f"{sblob['values']}")

    # (b) content: each corpus holds what its label says.
    for key, blob in eq["legs"]["b_content"]["per_corpus"].items():
        assert blob["maps_identical"], (
            f"{key}: the stored motion maps do not equal the recomputation "
            "from the capture")
        assert blob["scale_matches"], f"{key}: the stored scale moved"

    # (c) separation: no two corpora are the same data.
    assert eq["legs"]["c_separation"]["all_npz_sha256s_distinct"], (
        "two of the four corpora have the same npz SHA-256, so the sweep did "
        "not change the data")


def test_each_corpus_declares_its_own_filter(results):
    """The label in the sidecar is the filter that was asked for."""
    for filt in FILTERS:
        for name in ("A", "B", "C"):
            s = results["corpora"][filt]["sessions"][name]
            assert s["declared_filter"] == filt
            assert s["declared_provenance_filter"] == filt
            assert s["declared_max_frames"] == 0, (
                f"{filt}/{name} was seeded with a frame cap, so the corpora "
                "do not hold the same frames")
            assert f"--filter {filt}" in s["declared_command_line"]


def test_the_row_corpus_reproduces_the_published_corpus_v2(results):
    """The EXTERNAL control: the `row` corpus here is byte-for-byte the
    corpus every published teacher number was measured on.

    If it were not, this sweep would not be comparable with the repo's own
    numbers, and the "the published numbers were measured on the row
    representation" claim would be a guess.
    """
    check = results["corpus_equivalence"]["legs"]["c_separation"][
        "row_corpus_reproduces_corpus_v2"]
    for name in ("A", "B", "C"):
        assert check[name]["known_corpus_present"], (
            f"corpus_v2/{name}.npz is not on disk, so the comparison cannot "
            "be made")
        assert check[name]["identical"], (
            f"the row corpus's {name}.npz does not reproduce "
            f"{check[name]['known_corpus_path']} byte for byte")


@pytest.mark.skipif(not _corpora_on_disk(),
                    reason="the four corpora are not on disk")
def test_the_corpora_are_rebuilt_from_the_capture_on_disk(results):
    """The strongest check there is: rebuild each corpus's motion maps from
    the capture file, through the shared reader, and compare.

    `src.nv12.read_luma` DETECTS the frame layout from the file's content,
    so this check does not assume a stride -- it re-derives the frames and
    the filter and compares the bytes with what the corpus holds.
    """
    for filt in FILTERS:
        for name in ("A", "B", "C"):
            s = results["corpora"][filt]["sessions"][name]
            q, hi, layout = trr.motion_maps_from_capture(
                Path(s["source"]), filt)
            with np.load(s["npz_path"]) as d:
                stored = d["maps"]
                stored_scale = float(d["scale"])
            assert np.array_equal(stored, q), (
                f"{filt}/{name}: the corpus does not hold what the filter "
                "produces from the capture")
            assert np.float32(stored_scale) == np.float32(hi)
            assert layout.stride == 19200, (
                f"{filt}/{name}: the reader's stride moved to {layout.stride}")
            assert layout.n_frames == s["n_frames"]


def test_the_capture_files_are_the_ones_the_corpus_names(results):
    """The corpora's inputs are the capture files, checked by hash.

    The file size and the hash come from the capture ON DISK, so this is a
    check against the world rather than against the seeding script.
    """
    for name, blob in results["provenance"]["captures"].items():
        on_disk = hashlib.sha256(Path(blob["path"]).read_bytes()).hexdigest()
        assert on_disk == blob["sha256"], f"{name}.nv12 has changed on disk"
        assert blob["bytes"] % 19200 == 0, (
            f"{name}.nv12 is not a whole number of 19,200-byte frames")


# --- claim 2: raw is NOT better --------------------------------------------

def test_raw_does_not_beat_the_filtered_representations(results):
    """The headline: the filtered representations win the feature sets.

    `row` wins all five feature sets against raw; the two median
    compositions win four each. Raw wins one feature set against
    `row+median3` and one against `row+median5`, and none against `row`.
    """
    r1 = results["verdict"]["rule_1_representation"]
    assert r1["headline"] == "FILTERED_IS_BETTER", (
        f"the headline verdict moved to {r1['headline']} -- re-measure and "
        "update TEACHER-RAW-REPRESENTATION.md")
    assert r1["feature_set_wins_over_raw"]["row"] == 5
    assert r1["feature_set_wins_for_raw"]["row"] == 0
    assert r1["best_filter"] == "row"
    for filt in ("row", "row+median3", "row+median5"):
        assert r1["feature_set_wins_over_raw"][filt] >= 3, (
            f"{filt} no longer beats raw on a majority of the feature sets")


def test_the_tie_handling_is_not_load_bearing(results):
    """The same verdict under a tie tolerance of 1e-9.

    The margins are differences of accuracies on 48 test windows, so two
    mathematically equal margins can differ by ~1e-16 in floating point. The
    headline and the recommendation must not depend on how that is read.
    """
    ts = results["verdict"]["rule_1_representation"]["tie_sensitivity"]
    assert ts["headline_unchanged"], (
        "the headline depends on how a 1e-16 tie is read, so the tie "
        "handling IS load-bearing and has to be argued for, not assumed")
    assert ts["headline_at_tolerance_1e-9"] == \
        results["verdict"]["rule_1_representation"]["headline"]


def test_the_leave_one_session_out_folds_also_favour_the_filters(results):
    """The split that can tell "speaking" from "this recording" also does
    not favour raw.

    Ten cells (five feature sets x two folds) of the held-out real
    accuracy. `row` wins seven, loses one, ties two.
    """
    h2h = results["verdict"]["loo_head_to_head"]
    assert h2h["row"]["cells_where_the_filter_beats_raw"] > \
        h2h["row"]["cells_where_raw_beats_the_filter"], (
        "raw now wins more leave-one-session-out cells than row does -- "
        "re-measure and update TEACHER-RAW-REPRESENTATION.md")
    assert h2h["row"]["n_cells"] == 10


def test_the_size_of_the_difference_is_reported_with_the_verdict(results):
    """The verdict counts a feature set as a win whatever the size of the
    difference, so the size has to be reported beside it.

    One test window is 1/48 = 0.0208, and the largest gap between any two
    representations is four windows.
    """
    es = results["verdict"]["descriptive_effect_size"]
    assert es["n_test_windows"] == 48
    assert abs(es["one_test_window"] - 1 / 48) < 1e-12
    assert es["max_abs_difference_in_test_windows"] <= 6, (
        "the largest gap between representations has grown past six test "
        "windows -- re-measure and update TEACHER-RAW-REPRESENTATION.md")


# --- claim 3: nothing beats the teacher's own controls ---------------------

def test_every_margin_is_reported_with_the_control_it_was_measured_against(
        table):
    """A margin without its control beside it is not a result.

    This is the test that fails if the comparison degenerates into a table
    of numbers with nothing to compare them to: every margin in the file
    must be recomputable from the controls reported next to it.
    """
    for filt in FILTERS:
        for feat in FEATURE_SETS:
            t = table[filt][feat]["speaking"]["temporal"]
            assert "control_spatially_scrambled" in t
            assert "control_blank" in t
            assert "acc_test" in t
            expected = t["acc_test"] - max(t["control_spatially_scrambled"],
                                           t["control_blank"])
            assert abs(t["margin_over_best_control"] - expected) < 1e-12
            assert abs(t["margin_over_scrambled"]
                       - (t["acc_test"] - t["control_spatially_scrambled"])) \
                < 1e-12
            for fold in ("A", "B"):
                f = table[filt][feat]["loo_folds"][fold]
                assert "acc_real" in f and "control_spatially_scrambled" in f \
                    and "control_blank" in f
                assert abs(f["margin_over_scrambled"]
                           - (f["acc_real"] - f["control_spatially_scrambled"])) \
                    < 1e-12
                assert abs(f["margin_over_blank"]
                           - (f["acc_real"] - f["control_blank"])) < 1e-12


def test_the_blank_control_is_one_on_every_fold_of_every_representation(
        table):
    """The BLANK control -- a model applied to a constant input, which
    predicts one class for everything -- scores 1.000 on every fold.

    It is degenerate because the labels are session-level, so it is reported
    as the null and never used as a target. The real features never beat it:
    a margin over the blank control on a pure fold is not interpretable.
    """
    for filt in FILTERS:
        for feat in FEATURE_SETS:
            for fold in ("A", "B"):
                f = table[filt][feat]["loo_folds"][fold]
                assert f["control_blank"] == 1.0, (
                    f"{filt}/{feat}/{fold}: the blank control scored "
                    f"{f['control_blank']}, not 1.000")
                assert f["acc_real"] <= f["control_blank"], (
                    f"{filt}/{feat}/{fold}: the real features scored "
                    f"{f['acc_real']}, ABOVE the constant predictor's "
                    f"{f['control_blank']} -- the degeneracy argument in "
                    "TEACHER-LOO-FOLDS.md has to be re-checked")


def test_no_representation_beats_the_teachers_own_controls(results):
    """Rule 2's outcome, for all four representations.

    Not one feature set of any representation has a positive margin over
    its own fold's scrambled control on BOTH folds. That is the finding.
    """
    r2 = results["verdict"]["rule_2_controls"]
    for filt in FILTERS:
        assert r2[filt]["is_usable"] is False, (
            f"{filt} is now usable -- the teacher beat its controls on the "
            "raw/filtered sweep, and TEACHER-RAW-REPRESENTATION.md has to be "
            "re-checked")
        for feat in FEATURE_SETS:
            pf = r2[filt]["per_feature_set"][feat]
            assert pf["loo_beats_its_own_control_on_both_folds"] is False, (
                f"{filt}/{feat}: the leave-one-session-out folds now beat "
                "the scrambled control on both folds")


def test_the_blank_control_is_named_as_degenerate_in_the_results(results):
    """The null is named as the null, not left for the reader to work out."""
    assert "degenerate" in results["definitions"]["L_blank(F,S,f)"]
    assert "never a target" in results["definitions"]["L_blank(F,S,f)"]
    assert "blank" in results["verdict_text"].lower()


def test_the_usability_threshold_is_the_teachers_own(results):
    """Rule 2 uses the 0.10 threshold `teacher_conventional.py` itself uses
    for `usable`, not one invented here."""
    assert ">= 0.10" in results["pre_registered_rule"]["text"]
    for filt in FILTERS:
        for feat in FEATURE_SETS:
            t = table_of(results)[filt][feat]["speaking"]["temporal"]
            assert t["usable_by_the_teachers_own_threshold"] == (
                t["margin_over_best_control"] >= 0.10)


def table_of(results) -> dict:
    return results["table"]


# --- claim 4: the static control, on every representation ------------------

def test_the_static_control_scores_one_on_fold_a_everywhere(table):
    """The teacher still reads a PICTURE of the camera -- on the raw
    representation too.

    Every frame replaced by the session's temporal mean map: no motion, no
    order, no change. It scores 1.000 on fold A for every feature set of
    every representation, including the raw one. So the raw representation
    does not rescue the teacher.
    """
    for filt in FILTERS:
        for feat in FEATURE_SETS:
            static = table[filt][feat]["loo_folds"]["A"]["static_control"]
            assert static["static"] == 1.0, (
                f"{filt}/{feat}: the static variant scored "
                f"{static['static']} on fold A, not 1.000 -- the teacher may "
                "need the motion after all, and TEACHER-RAW-REPRESENTATION.md "
                "has to be re-checked")


def test_the_static_control_is_reported_for_every_representation(table):
    """Every cell of every fold carries the static, time-shuffled and
    unscaled controls, not just the real one."""
    for filt in FILTERS:
        for feat in FEATURE_SETS:
            for fold in ("A", "B"):
                sc = table[filt][feat]["loo_folds"][fold]["static_control"]
                for variant in ("real", "static", "time_shuffled", "unscaled"):
                    assert variant in sc, (
                        f"{filt}/{feat}/{fold}: the {variant} control is "
                        "missing")


def test_the_scale_control_still_does_not_move_the_results(table):
    """The teacher's numbers are not an artefact of the maps' display scale.

    `unscaled` divides every session by the same constant, so the
    between-session ratios are untouched and only the absolute scale moves.
    It reproduces the real variant exactly, on all four corpora.
    """
    for filt in FILTERS:
        for feat in FEATURE_SETS:
            for fold in ("A", "B"):
                sc = table[filt][feat]["loo_folds"][fold]["static_control"]
                assert sc["real"] == sc["unscaled"], (
                    f"{filt}/{feat}/{fold}: the result moved from "
                    f"{sc['real']} to {sc['unscaled']} when the maps were "
                    "rescaled")


# --- the verdict itself ---------------------------------------------------

def test_the_verdict_follows_the_pre_registered_rule(results):
    """Recompute the verdict from the numbers in the results file.

    If the verdict was written by hand, or the code and the record have
    drifted apart, this fails.
    """
    recomputed = trr.evaluate(results["table"], trr.PRIMARY_TASK)
    recorded = results["verdict"]
    assert recomputed["rule_1_representation"]["headline"] == \
        recorded["rule_1_representation"]["headline"]
    assert recomputed["recommended_representation"] == \
        recorded["recommended_representation"]
    assert recomputed["rule_1_representation"]["feature_set_wins_over_raw"] \
        == recorded["rule_1_representation"]["feature_set_wins_over_raw"]
    for filt in FILTERS:
        assert recomputed["rule_2_controls"][filt]["is_usable"] == \
            recorded["rule_2_controls"][filt]["is_usable"]


def _synthetic_table(t_by_filter: dict) -> dict:
    """A table with the shape `evaluate` reads, from a dict of margins."""
    out = {}
    for filt in FILTERS:
        out[filt] = {}
        for feat in FEATURE_SETS:
            out[filt][feat] = {
                "speaking": {"temporal": {
                    "margin_over_best_control": t_by_filter[filt][feat],
                    "n_test_windows": 48}},
                "facing": {"temporal": {
                    "margin_over_best_control": t_by_filter[filt][feat],
                    "n_test_windows": 48}},
                "loo_folds": {
                    fold: {"margin_over_scrambled": 0.0} for fold in ("A", "B")},
            }
    return out


def test_every_branch_of_the_verdict_is_reachable(results):
    """A verdict that cannot come out the other way is not a verdict.

    Each of the four verdicts is produced from a synthetic table, so the
    function is shown failing -- in the sense of landing on a different
    answer -- when the numbers change.
    """
    zero = {f: {s: 0.0 for s in FEATURE_SETS} for f in FILTERS}

    # RAW_IS_BETTER: raw beats every filter on every feature set.
    raw_wins = {f: {s: 0.0 for s in FEATURE_SETS} for f in FILTERS}
    raw_wins["none"] = {s: 1.0 for s in FEATURE_SETS}
    assert trr.evaluate(_synthetic_table(raw_wins))["rule_1_representation"][
        "headline"] == "RAW_IS_BETTER"

    # FILTERED_IS_BETTER: one filter beats raw, raw beats none.
    filt_wins = {f: {s: 0.0 for s in FEATURE_SETS} for f in FILTERS}
    filt_wins["none"] = {s: 0.0 for s in FEATURE_SETS}
    filt_wins["row"] = {s: 1.0 for s in FEATURE_SETS}
    assert trr.evaluate(_synthetic_table(filt_wins))["rule_1_representation"][
        "headline"] == "FILTERED_IS_BETTER"

    # MIXED: one filter beats raw AND raw beats another.
    mixed = {f: {s: 0.0 for s in FEATURE_SETS} for f in FILTERS}
    mixed["row"] = {s: 1.0 for s in FEATURE_SETS}
    for s in ("both", "region", "region+map"):
        mixed["none"][s] = 0.5
    assert trr.evaluate(_synthetic_table(mixed))["rule_1_representation"][
        "headline"] == "MIXED"

    # NO_DIFFERENCE: nothing separates them.
    assert trr.evaluate(_synthetic_table(zero))["rule_1_representation"][
        "headline"] == "NO_DIFFERENCE"

    # And the recorded verdict is one of the four, not a fifth.
    assert results["verdict"]["rule_1_representation"]["headline"] in (
        "RAW_IS_BETTER", "FILTERED_IS_BETTER", "MIXED", "NO_DIFFERENCE")


def test_the_recommendation_is_one_of_the_four_representations(results):
    rec = results["verdict"]["recommended_representation"]
    assert rec in FILTERS
    assert rec in results["verdict"]["rule_2_controls"]


def test_the_provenance_names_every_input(results):
    """A number without its inputs cannot be traced."""
    prov = results["provenance"]
    for key in ("script_sha256", "python", "cli_args", "scripts_used_unchanged",
                "captures", "corpora_written"):
        assert key in prov, f"the provenance is missing {key}"
    assert isinstance(prov["cli_args"], list), (
        "the cli args are not recorded as a list, so the run cannot be "
        "re-run as it was")
    # Every script that produced the numbers is named, with its hash, so a
    # reader can go and read exactly what ran. Whether those files still
    # hash the same is NOT asserted here -- another lane may fix a docstring
    # in a shared script, and the question that matters is whether the
    # BEHAVIOUR is the same. The next two tests ask that.
    for name, sha in prov["scripts_used_unchanged"].items():
        assert sha, f"{name} has no recorded hash"
        assert (ROOT / "experiments" / name).is_file(), f"{name} is missing"
    assert results["reproducible"] is True


@pytest.mark.skipif(not _corpora_on_disk(),
                    reason="the four corpora are not on disk")
def test_the_seeding_script_still_produces_the_recorded_corpora(results,
                                                                 tmp_path):
    """Re-seed one corpus with the script as it is on disk NOW, and check it
    is byte-for-byte the corpus the results were measured on.

    This is a BEHAVIOURAL check rather than a hash of the script's file: a
    shared script's docstring may legitimately be fixed by another lane, and
    what matters is that the command still produces the same data. The
    `row` corpus is the one to check, because it is the one the published
    teacher numbers were measured on.
    """
    out = tmp_path / "reseed_row"
    trr.run_cmd([sys.executable, trr.SEED_SCRIPT, "--filter", "row",
                 "--out", str(out), "--max-frames", "0"])
    for name in ("A", "B", "C"):
        recorded = results["corpora"]["row"]["sessions"][name]["npz_sha256"]
        on_disk = hashlib.sha256((out / f"{name}.npz").read_bytes()).hexdigest()
        assert on_disk == recorded, (
            f"re-seeding with the script as it is now produces a different "
            f"{name}.npz: {on_disk} against the recorded {recorded}")


@pytest.mark.skipif(not _corpora_on_disk(),
                    reason="the four corpora are not on disk")
def test_the_teacher_script_still_produces_the_recorded_numbers(results,
                                                                tmp_path):
    """Re-run one teacher command with the script as it is on disk NOW, and
    check it reproduces the numbers in the table.

    Again behavioural, not a file hash: the point of the provenance block is
    that the recorded numbers can be reproduced, and that is what has to be
    true, whatever else a shared script's docstring says.
    """
    out = tmp_path / "teacher"
    out.mkdir()
    trr.run_cmd([sys.executable, trr.TEACHER_SCRIPT, "--corpus",
                 str(trr.corpus_dir("none")), "--task", "speaking",
                 "--features", "map", "--out", str(out)])
    written = json.loads((out / "teacher_conventional_speaking_map.json")
                         .read_text())
    recorded = results["table"]["none"]["map"]["speaking"]["temporal"]
    assert written["acc_test"] == recorded["acc_test"]
    assert written["acc_test_spatially_scrambled"] == \
        recorded["control_spatially_scrambled"]
    assert written["acc_test_blank"] == recorded["control_blank"]
    assert abs(written["margin_over_best_control"]
               - recorded["margin_over_best_control"]) < 1e-12
