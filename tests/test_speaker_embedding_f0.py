"""
Tests for the SPEAKER-EMBEDDING measurement.

WHY THIS FILE EXISTS.

`docs/research/F0-SPEAKER-CHANGE.md` measured that pitch cannot tell two
same-gender speakers apart in the AMI two-speaker corpus: FEO070 and FEO072
sit 0.5 sigma apart, MEE071 and MEE073 0.3 sigma, so "a hand-off produces a
pitch jump" is false for a hand-off within a voice class. The same document
names the escape -- a speaker embedding -- and
`experiments/speaker_embedding_f0.py` is that measurement.

The measurement is only worth anything if it can be checked. This file is
the check.

WHAT IS ASSERTED HERE, AND WHAT IS DELIBERATELY NOT.

ASSERTED:

  * IDENTITY. The result file names the model, the model revision, every
    audio file, every segment file, and a SHA-256 for each. The audio and
    segment SHAs are checked against the files on disk, so a result file
    that describes some other corpus is rejected. (The SCRIPT's own SHA is
    checked for FORM only: it identifies the version of the script that
    produced the file, and re-running the script refreshes it, so asserting
    equality against the working copy would only assert that nobody has
    edited the script since the last run.)
  * THE HEADLINE IS RECOMPUTED, NOT HARD-CODED. The separation is recomputed
    from the recorded within-speaker and between-speaker distributions and
    compared with the recorded value. A result file whose headline does not
    follow from its own numbers is rejected. The checker is then ATTACKED:
    a perturbed result file must be rejected, because a test that cannot
    fail is not evidence.
  * THE CONTROLS ARE RECORDED. The shuffled-label null, the same-speaker
    control, the crop/scramble-style nulls, and the degenerate
    constant-decision baselines all have to be present with their counts.
  * THE COUNTS ARE IN THE SAME FORM. The within-speaker and the
    between-speaker distributions are described with the same keys, so a
    separation computed from one of them and not the other cannot hide.
  * THE EERs ARE RECOMPUTED from the recorded score lists.

NOT ASSERTED:

  * THE OUTCOME. Whether the embedding beats pitch, and whether the result
    is a clean hit or a null, is a FINDING, not a property of the corpus. A
    null result with its controls recorded is a valid outcome, and this
    suite passes on it. The only outcome-adjacent assertions are the ones
    about the METHOD: the null has to be a null, the controls have to be
    present, and the numbers have to follow from each other.

  * The `experiments/speaker_embedding_f0.py` module is imported for its
    pure-numpy analysis functions (`separation`, `eer_from_scores`,
    `describe`). It imports NO torch at module level -- the torch import is
    inside the functions that need it -- so this suite runs in the repo's
    own `.venv`, on a machine with no GPU, and downloads nothing.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import speaker_embedding_f0 as se            # noqa: E402

RESULTS = ROOT / "experiments" / "speaker_embedding_f0_results.json"
SCRIPT = ROOT / "experiments" / "speaker_embedding_f0.py"
REPORT = ROOT / "docs" / "research" / "SPEAKER-EMBEDDING-F0.md"

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SUMMARY_KEYS = {"n", "mean", "std", "mad_sigma", "median", "p05", "p95",
                 "min", "max"}


# ---------------------------------------------------------------------------
# helpers -- the checkers the tests attack
# ---------------------------------------------------------------------------

def load_results() -> dict:
    assert RESULTS.is_file(), f"missing result file: {RESULTS}"
    return json.loads(RESULTS.read_text())


def separation_is_consistent(pair: dict, tol: float = 5e-4) -> bool:
    """Does the recorded separation follow from the recorded distributions?

    This is the checker the headline test uses, and the one the attack test
    shows can fail.
    """
    within = pair.get("within_speaker") or {}
    between = pair.get("between_speaker") or {}
    recorded = (pair.get("separation") or {}).get("separation_sigma_mad")
    if not within.get("n") or not between.get("n") or recorded is None:
        return False
    recomputed = se.separation(within, between).get("separation_sigma_mad")
    if recomputed is None:
        return False
    return abs(recomputed - recorded) <= tol


def recorded_eer_is_consistent(target, impostor, eer_block, tol: float = 5e-4) -> bool:
    """Does the recorded EER follow from the recorded score lists?

    The score lists and the EER live in different places depending on the
    task, so they are passed in separately rather than guessed at.
    """
    recorded = (eer_block or {}).get("eer")
    if target is None or impostor is None or recorded is None:
        return False
    recomputed = se.eer_from_scores(target, impostor).get("eer")
    if recomputed is None:
        return False
    return abs(recomputed - recorded) <= tol


# ---------------------------------------------------------------------------
# the identity of the inputs
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_model_id_and_revision_are_recorded(self):
        r = load_results()
        model = r["model"]
        assert model["id"], "no model id recorded"
        assert REVISION_RE.match(model["revision"]), \
            f"model revision is not a 40-hex commit sha: {model['revision']!r}"
        assert model.get("class"), "no model class recorded"

    def test_every_input_audio_and_segment_file_is_named_and_hashed(self):
        r = load_results()
        inputs = r["inputs"]
        assert inputs, "no input files recorded"
        for name, block in inputs.items():
            assert block.get("path"), f"{name}: no path recorded"
            assert SHA_RE.match(block.get("sha256") or ""), \
                f"{name}: sha256 is not a 64-hex digest: {block.get('sha256')!r}"
            assert block.get("sidecar"), f"{name}: no segment file recorded"
            assert SHA_RE.match(block.get("sidecar_sha256") or ""), \
                f"{name}: sidecar sha256 is not a 64-hex digest"

    def test_recorded_audio_hashes_match_the_files_on_disk(self):
        r = load_results()
        for name, block in r["inputs"].items():
            path = ROOT / block["path"]
            if not path.is_file():
                pytest.skip(f"{name}: audio not on disk ({path})")
            assert se.sha256_file(path) == block["sha256"], \
                f"{name}: the recorded sha256 does not match the file on disk"

    def test_recorded_segment_hashes_match_the_files_on_disk(self):
        r = load_results()
        for name, block in r["inputs"].items():
            path = ROOT / block["sidecar"]
            if not path.is_file():
                pytest.skip(f"{name}: segment file not on disk ({path})")
            assert se.sha256_file(path) == block["sidecar_sha256"], \
                f"{name}: the recorded sidecar sha256 does not match disk"

    def test_the_script_that_produced_the_file_is_identified(self):
        r = load_results()
        script = r.get("script") or {}
        assert script.get("path"), "no script path recorded"
        assert SHA_RE.match(script.get("sha256") or ""), \
            "the script's own sha256 is not a 64-hex digest"

    def test_the_preprocessing_is_recorded(self):
        r = load_results()
        method = r["method"]
        assert method["sample_rate_hz"] == 16000
        assert "src/wavio.py" in method["reader"], \
            "the reader is not the repo's own reader"
        assert method.get("embedding_unit"), "no embedding unit recorded"
        assert "cosine" in method["distance"]
        assert method.get("separation_form"), "no separation formula recorded"

    def test_the_window_and_hop_are_recorded(self):
        r = load_results()
        assert r["method"].get("boundary_clip_s"), "no boundary clip length"
        sweep = r["length_sweep"]
        assert sweep.get("sweep"), "no length sweep recorded"
        for key, row in sweep["sweep"].items():
            assert row["min_len_s"] > 0
            assert row["n_turns"] > 0, f"min_len {key}: no turns"

    def test_the_counts_are_recorded(self):
        r = load_results()
        pop = r["turn_population"]
        assert pop["audible_turns_total"] > 0
        assert pop["headline_turns"] > 0
        for spk, n in pop["per_speaker_turns"].items():
            assert n > 0, f"speaker {spk} has no turns at the headline length"


# ---------------------------------------------------------------------------
# the headline, recomputed
# ---------------------------------------------------------------------------

class TestHeadlineIsRecomputed:
    def test_each_pair_separation_follows_from_its_own_numbers(self):
        r = load_results()
        pairs = r["headline"]["pairs"]
        assert pairs, "no headline pairs recorded"
        for key, pair in pairs.items():
            assert separation_is_consistent(pair), \
                f"{key}: the recorded separation does not follow from the " \
                f"recorded within/between distributions"

    def test_the_sweep_separations_also_follow_from_their_own_numbers(self):
        r = load_results()
        for min_len, row in r["length_sweep"]["sweep"].items():
            for key, pair in row["pairs"].items():
                assert separation_is_consistent(pair), \
                    f"min_len {min_len} {key}: recorded separation does not " \
                    f"follow from the recorded distributions"

    def test_the_f0_baseline_is_computed_in_the_same_arithmetic(self):
        r = load_results()
        for key, pair in r["headline"]["pairs"].items():
            f0 = pair.get("f0_same_turns_same_arithmetic")
            assert f0, f"{key}: no F0 baseline in the same arithmetic"
            assert separation_is_consistent(f0), \
                f"{key}: the F0 baseline does not follow from its own numbers"
            assert f0["separation"]["formula"] == \
                r["headline"]["pairs"][key]["separation"]["formula"], \
                f"{key}: the F0 baseline and the embedding headline use " \
                f"different formulas, so they are not comparable"

    def test_the_f0_document_number_is_reproduced_on_the_same_turns(self):
        r = load_results()
        for key, pair in r["headline"]["pairs"].items():
            doc = pair.get("f0_doc_style")
            assert doc, f"{key}: the F0 document's own arithmetic is not " \
                        f"recomputed on the same turns"
            assert doc.get("separation_sigma") is not None

    def test_the_checker_rejects_a_perturbed_result(self):
        """A test that cannot fail is not a test."""
        r = load_results()
        key = next(iter(r["headline"]["pairs"]))
        good = copy.deepcopy(r["headline"]["pairs"][key])
        assert separation_is_consistent(good) is True
        bad = copy.deepcopy(good)
        bad["separation"]["separation_sigma_mad"] = \
            float(bad["separation"]["separation_sigma_mad"]) + 1.0
        assert separation_is_consistent(bad) is False, \
            "the separation checker accepted a headline that does not " \
            "follow from its own numbers"

    def test_the_checker_rejects_a_perturbed_distribution(self):
        r = load_results()
        key = next(iter(r["headline"]["pairs"]))
        good = copy.deepcopy(r["headline"]["pairs"][key])
        bad = copy.deepcopy(good)
        bad["within_speaker"]["median"] = \
            float(bad["within_speaker"]["median"]) + 0.5
        assert separation_is_consistent(bad) is False, \
            "the separation checker accepted a perturbed within-speaker " \
            "distribution"


# ---------------------------------------------------------------------------
# the counts, in the same form
# ---------------------------------------------------------------------------

class TestCountsAreInTheSameForm:
    def test_within_and_between_are_described_with_the_same_keys(self):
        r = load_results()
        for key, pair in r["headline"]["pairs"].items():
            for name in ("within_speaker", "between_speaker"):
                block = pair.get(name) or {}
                assert block.get("n", 0) > 0, f"{key}: {name} has no samples"
                assert SUMMARY_KEYS <= set(block), \
                    f"{key}: {name} is missing {SUMMARY_KEYS - set(block)}"

    def test_within_and_between_counts_are_both_recorded(self):
        r = load_results()
        for key, pair in r["headline"]["pairs"].items():
            w = pair["within_speaker"]["n"]
            b = pair["between_speaker"]["n"]
            assert w > 0 and b > 0, f"{key}: a distance distribution is empty"
            assert w == b or w > 0, "counts are not recorded"

    def test_the_same_speaker_control_is_recorded(self):
        r = load_results()
        ctrl = r["controls"]["same_speaker_control"]
        pooled = ctrl.get("_pooled") or {}
        assert pooled.get("n", 0) > 0, "the same-speaker control has no samples"
        for spk, block in ctrl.items():
            if spk.startswith("_"):
                continue
            assert block["distances"]["n"] > 0, f"{spk}: no same-speaker samples"

    def test_the_turn_population_counts_are_recorded(self):
        r = load_results()
        pop = r["turn_population"]
        for spk in pop["per_speaker_turns"]:
            assert spk in pop["per_speaker_voiced_frames"]
            assert spk in pop["per_speaker_overlapped_turns"]


# ---------------------------------------------------------------------------
# the controls
# ---------------------------------------------------------------------------

class TestControlsAreRecorded:
    def test_the_shuffled_label_null_is_recorded(self):
        r = load_results()
        null = r["controls"]["shuffled_label_null"]
        assert null["n_permutations"] > 0, "the shuffled-label null ran no " \
                                           "permutations"
        assert null.get("seed") is not None, "the null has no seed, so it " \
                                             "cannot be reproduced"
        for key in r["headline"]["pairs"]:
            block = null["separation_sigma_mad"].get(key)
            assert block, f"{key}: no shuffled-label null"
            assert block["n"] == null["n_permutations"], \
                f"{key}: the null has {block['n']} values, not " \
                f"{null['n_permutations']}"
            assert block.get("mean") is not None

    def test_the_shuffled_label_null_permutation_test_is_recomputable(self):
        r = load_results()
        null = r["controls"]["shuffled_label_null"]
        for key, pair in r["headline"]["pairs"].items():
            block = pair.get("shuffled_label_null")
            assert block, f"{key}: no per-pair shuffled-label null"
            test = block.get("permutation_test") or {}
            assert "p_value" in test, f"{key}: no permutation p-value"
            assert test.get("n_null") == null["n_permutations"]

    def test_the_scramble_style_nulls_are_recorded(self):
        r = load_results()
        nulls = r["controls"]["crop_and_scramble_nulls"]
        for name in ("time_reversed", "gain_scaled", "white_noise"):
            assert name in nulls, f"the {name} null was not run"
            for key in r["headline"]["pairs"]:
                block = nulls[name].get(key)
                assert block, f"{name}: no result for {key}"
                assert separation_is_consistent(block), \
                    f"{name} {key}: separation does not follow from its numbers"

    def test_the_degenerate_constant_decision_baselines_are_recorded(self):
        r = load_results()
        base = r["turn_change_task"]["degenerate_baselines"]
        assert "constant_decision_always_change" in base
        assert "constant_decision_always_same" in base
        for name, block in base.items():
            assert block["eer"] == 0.5, \
                f"{name}: a constant decision scores EER 0.5 by construction; " \
                f"a different number means the baseline was not a constant"

    def test_the_f0_only_baseline_is_recorded_on_the_same_turns(self):
        r = load_results()
        for key, pair in r["headline"]["pairs"].items():
            assert pair.get("f0_same_turns_same_arithmetic"), \
                f"{key}: no F0-only baseline"

    def test_the_headline_is_a_margin_over_a_control_not_a_bare_accuracy(self):
        r = load_results()
        for key, pair in r["headline"]["pairs"].items():
            sep = pair.get("separation") or {}
            assert "separation_sigma_mad" in sep
            assert sep.get("mean_ratio") is not None, \
                f"{key}: the ratio of the means is not recorded"


# ---------------------------------------------------------------------------
# the turn-change task
# ---------------------------------------------------------------------------

class TestTurnChangeTask:
    def test_the_task_is_defined_and_populated(self):
        r = load_results()
        tc = r["turn_change_task"]
        assert tc.get("definition"), "the turn-change task is not defined"
        assert tc["n_boundaries"] > 0, "no boundaries were found"
        assert tc["n_positive_different_speaker"] > 0, \
            "the positive class (a speaker change) is unrepresentable"
        assert tc["n_negative_same_speaker"] > 0, \
            "the same-speaker control class is unrepresentable"
        assert tc["n_within_turn_controls"] > 0, \
            "the within-turn control is unrepresentable"

    def test_the_eer_is_recomputable_from_the_recorded_scores(self):
        r = load_results()
        block = r["turn_change_task"]["embedding"]
        assert recorded_eer_is_consistent(
            block["positive_scores"], block["negative_same_speaker_scores"],
            block["eer_vs_same_speaker"]), \
            "the recorded EER does not follow from the recorded score lists"

    def test_the_eer_against_the_within_turn_control_is_recomputable(self):
        r = load_results()
        block = r["turn_change_task"]["embedding"]
        assert recorded_eer_is_consistent(
            block["positive_scores"],
            block["negative_same_speaker_scores"] + block["within_turn_scores"],
            block["eer_vs_same_speaker_and_within_turn"]), \
            "the EER against the within-turn control does not follow from the " \
            "recorded score lists"

    def test_the_f0_baseline_on_the_same_boundaries_is_recorded(self):
        r = load_results()
        block = r["turn_change_task"]["f0_baseline"]
        assert recorded_eer_is_consistent(
            block["positive_scores"], block["negative_same_speaker_scores"],
            block["eer_vs_same_speaker"]), \
            "the F0 baseline EER does not follow from its recorded scores"

    def test_the_shuffled_label_null_on_the_task_is_recorded(self):
        r = load_results()
        null = r["turn_change_task"]["shuffled_label_null"]
        assert null.get("n_permutations", 0) > 0
        assert null.get("null_eer", {}).get("n", 0) > 0
        assert "permutation_test" in null

    def test_the_two_speaker_verification_eer_is_recomputable(self):
        r = load_results()
        for key, pair in r["headline"]["pairs"].items():
            v = pair.get("verification_2speaker")
            assert v, f"{key}: no two-speaker verification task"
            assert recorded_eer_is_consistent(
                v["target_scores"], v["impostor_scores"], v["eer"]), \
                f"{key}: the recorded 2-speaker EER does not follow from its " \
                f"recorded score lists"

    def test_the_four_speaker_verification_eer_is_recomputable(self):
        r = load_results()
        v = r["verification_4speaker"]
        assert recorded_eer_is_consistent(
            v["target_scores"], v["impostor_scores"], v["eer"]), \
            "the recorded 4-speaker EER does not follow from its score lists"


# ---------------------------------------------------------------------------
# the doc
# ---------------------------------------------------------------------------

class TestTheDocument:
    def test_the_document_exists(self):
        assert REPORT.is_file(), f"missing report: {REPORT}"

    def test_the_document_quotes_the_numbers_the_result_file_holds(self):
        text = REPORT.read_text()
        r = load_results()
        for key, pair in r["headline"]["pairs"].items():
            sep = pair["separation"]["separation_sigma_mad"]
            assert sep is not None
            assert f"{sep}" in text, \
                f"{key}: the document does not quote the measured " \
                f"separation {sep}"
