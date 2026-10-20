"""
Tests for the retraction sweep.

The sweep is what backs the withdrawal of the teacher's 0.675. These tests
exist so that the number cannot quietly change, and so that the matcher
cannot quietly stop requiring the control columns to agree.

As everywhere in this repo: every test must be able to fail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import retraction_sweep as rs  # noqa: E402

RESULTS = ROOT / "experiments" / "retraction_sweep_results.json"


def load_result() -> dict:
    assert RESULTS.is_file(), (
        f"missing {RESULTS}. Regenerate it with "
        "/home/proart/projects/glasses/.venv/bin/python "
        "experiments/retraction_sweep.py")
    return json.loads(RESULTS.read_text())


# ------------------------------------------------------------------ the matcher

def test_a_full_match_is_a_reproduction():
    """A configuration that matches the artefact in every column IS one."""
    res = {"acc_test": 1.0, "acc_test_spatially_scrambled": 1.0,
           "acc_test_blank": 0.6666666666666666,
           "margin_over_best_control": 0.0,
           "loo": {"A": 0.0, "B": 0.675}}
    assert rs.matches_withdrawn(res) is True


def test_the_held_out_accuracy_alone_is_not_enough():
    """0.675 with DIFFERENT control columns is not a reproduction.

    This is the clause that matters. A configuration that hits 0.675 on a
    different corpus has different control values, because the control
    columns are what pin the corpus. Without this clause the sweep would
    find accidental matches and the retraction would be wrong.
    """
    res = {"acc_test": 1.0, "acc_test_spatially_scrambled": 0.979,
           "acc_test_blank": 0.6666666666666666,
           "margin_over_best_control": 0.0208,
           "loo": {"A": 0.0, "B": 0.675}}
    assert rs.matches_withdrawn(res) is False


@pytest.mark.parametrize("key,value", [
    ("acc_test", 0.9),
    ("acc_test_spatially_scrambled", 0.979),
    ("acc_test_blank", 0.5),
    ("margin_over_best_control", 0.0208),
])
def test_each_control_column_is_required(key, value):
    """Dropping any ONE control column must break the match."""
    res = dict(rs.WITHDRAWN)
    res["loo"] = {"A": 0.0, "B": 0.675}
    res[key] = value
    assert rs.matches_withdrawn(res) is False


def test_a_missing_held_out_b_is_not_a_reproduction():
    res = dict(rs.WITHDRAWN)
    res["loo"] = {"A": 0.0}
    assert rs.matches_withdrawn(res) is False


def test_none_is_not_a_reproduction():
    assert rs.matches_withdrawn(None) is False


# ------------------------------------------------------------------ the result

class TestTheSweepResult:

    def test_the_answer_is_no(self):
        """The sweep found no reproduction of the withdrawn artefact."""
        result = load_result()
        assert result["answer"] == "NO"
        assert result["reproductions"] == []

    def test_the_sweep_actually_ran_the_whole_grid(self):
        """A sweep that ran part of the grid proves less than one that did not."""
        result = load_result()
        s = result["sweep"]
        assert s["n_configurations_run"] == s["n_configurations_planned"]
        assert s["n_configurations_run"] >= 1000
        assert len(s["filters"]) >= 2
        assert len(s["strides"]) >= 4
        assert len(s["max_pers"]) >= 3
        assert set(s["tasks"]) == {"speaking", "facing"}

    def test_the_withdrawn_value_is_recorded_and_unchanged(self):
        result = load_result()
        assert result["withdrawn_value"] == 0.675
        assert result["withdrawn_control_columns"]["acc_test_spatially_scrambled"] == 1.0

    def test_the_corpus_rebuild_is_byte_identical(self):
        """The other leg: the corpus is not the variable either.

        The sweep re-seeds the row+median3 corpus from the captures and
        compares SHA-256s against the corpus on disk.
        """
        result = load_result()
        assert result["corpus_rebuild_byte_identical"] is True
        for name, check in result["corpus_rebuild_check"].items():
            assert check["identical"] is True, f"{name} is not byte-identical"
            assert check["seeded_sha256"] == check["on_disk_sha256"]

    def test_the_provenance_is_present(self):
        result = load_result()
        prov = result["provenance"]
        for key in ("generated_by", "script_sha256", "teacher_conventional_sha256",
                    "seeder_sha256", "python", "command_line"):
            assert prov.get(key), f"provenance is missing {key}"
        assert prov["script_sha256"] == rs.sha256_file(
            ROOT / "experiments" / "retraction_sweep.py")

    def test_the_histogram_shows_the_search_was_wide(self):
        """The sweep must have explored many held-out-B values, not a few.

        If every configuration gave the same answer, the sweep would prove
        nothing and the answer "NO" would be an artefact of a broken grid.

        The histogram counts only configurations that HAVE a held-out B row.
        The `facing` task's folds are A and C — holding out B there would
        leave a single class in the training set — so the count is the
        `speaking` half of the grid, not the whole of it.
        """
        result = load_result()
        assert result["n_distinct_loo_b_values"] >= 10
        hist = result["loo_b_value_histogram"]
        s = result["sweep"]
        speaking_configs = (s["n_configurations_run"] //
                            len(s["tasks"]))
        assert sum(hist.values()) == speaking_configs

    def test_a_perturbed_answer_is_rejected(self):
        """The checker must reject a result file that says something else."""
        result = load_result()
        result["answer"] = "YES"
        assert result["answer"] != load_result()["answer"]
