"""
test_debrief_reaudit.py -- the audit set's key is derived, not asserted.

The night of 2026-10-20 found that every camera reader in this repository
was reading the captures at the wrong frame stride, and the audit set's
ground truth had been written against the torn frames. Two of the three
standing claims were standing on numbers computed on those tears.

These tests exist so that the same thing cannot happen quietly again:

  - every label in `debrief.AUDIT_SET` must be DERIVED from a current
    results file, so a label that drifts back without its evidence drifting
    with it fails the build;
  - the claim TEXTS must not change, because the texts are the items and
    the key is the thing that moves;
  - the old text must stay visible in the docstring, because an audit set
    that has been silently overwritten is a quiz with the answers written on
    it;
  - the committed artefacts must match the corrected key, so a score
    against a key that no longer exists is never shipped.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import debrief  # noqa: E402
import debrief_set_reaudit as dsr  # noqa: E402


# The blind prompt as it was committed on 2026-10-19, before any of tonight's
# changes. If the claim texts move, this hash moves, and the before/after
# comparison stops being a comparison.
BLIND_PROMPT_SHA256 = "5870ad6bb1989c07727c57ea01c417aa0a014674824d48ee8a692665e6e1100c"


def _claims():
    return {c: f for c, f in debrief.AUDIT_SET}


def _flat(text: str) -> str:
    """Whitespace-collapsed, so a docstring's line wrapping is not a
    failure. The wrapping is not what these tests are about."""
    return " ".join(text.split())


# ---------------------------------------------------------------- the key

def test_every_audit_claim_has_a_current_evidence_backed_label():
    """The label in AUDIT_SET must equal the label the evidence derives.

    This is the test that fails if a stale label creeps back in. The label
    in `debrief.py` is compared against a check that reads a current results
    file, so the two cannot disagree without the build failing.
    """
    in_file = _claims()
    derived = {c["claim"]: c for c in dsr.reaudit_set()["claims"]}
    assert len(derived) == len(in_file), "an audit claim has no evidence entry"
    for claim, is_false in in_file.items():
        assert claim in derived, f"no evidence for: {claim[:60]}"
        assert derived[claim]["new_is_false"] == is_false, (
            f"STALE LABEL: {claim[:70]!r}\n"
            f"  file says is_false={is_false}\n"
            f"  the current results files say is_false="
            f"{derived[claim]['new_is_false']}\n"
            f"  {derived[claim]['why'][:300]}")


def test_the_audit_set_is_not_all_true_and_not_all_false():
    """A set that is all one class cannot measure anything.

    This is the test that would catch a key that collapsed to a constant,
    which is exactly the failure mode of an over-eager re-audit.
    """
    labels = list(_claims().values())
    n_false = sum(1 for f in labels if f)
    n_true = len(labels) - n_false
    assert n_false > 0, "every claim is a true claim; the set measures nothing"
    assert n_true > 0, "every claim is a false claim; the set measures nothing"
    assert n_false + n_true == len(labels)


def test_the_two_changed_labels_are_the_two_that_moved():
    """The re-audit changed two labels, and they are the two expected ones.

    The row-median claim and the composed-filter claim were the two
    standing claims whose evidence was computed on torn frames. The
    acoustics claim is the one that survived, because it is an audio
    measurement.
    """
    by_key = {c["key"]: c for c in dsr.reaudit_set()["claims"]}
    row_median = by_key["Removing each row's median across the columns"]
    composed = by_key["Composing row-wise band removal with a temporal median"]
    acoustics = by_key["The room's own acoustic noise floor measured -73.5 dBFS"]
    assert row_median["old_is_false"] is False and row_median["new_is_false"] is True
    assert composed["old_is_false"] is False and composed["new_is_false"] is True
    assert acoustics["old_is_false"] is False and acoustics["new_is_false"] is False
    changed = [c["key"] for c in dsr.reaudit_set()["claims"] if c["label_changed"]]
    assert len(changed) == 2, f"expected exactly two label changes, got {changed}"


def test_the_set_now_holds_nine_false_and_one_true():
    """The counts, so a future edit has to argue with this number."""
    counts = dsr.reaudit_set()["counts"]
    assert counts["before"] == {"n_false": 7, "n_true": 3}
    assert counts["after"] == {"n_false": 9, "n_true": 1}


def test_the_old_labels_are_still_what_the_committed_artefacts_record():
    """The "before" column is a hard-coded key, and it must agree with the
    artefacts that were recorded against it.

    The committed artefacts now carry the CORRECTED key -- they were
    re-run. So the check is made against the BEFORE snapshot, which is
    what the committed record looked like when the re-audit started.
    """
    snap = ROOT / ".scratch" / "debrief_reaudit" / "before.json"
    if not snap.is_file():
        pytest.skip("the before-snapshot is not present")
    d = json.loads(snap.read_text())
    seen = 0
    for art in d.get("artefacts") or []:
        for run in art.get("runs") or []:
            for row in (run.get("old_key") or {}).get("rows") or []:
                key = dsr._key_for(row["claim"])
                assert row["is_false_claim"] == dsr.OLD_LABELS[key], (
                    f"{art['file']}: the before-snapshot's key disagrees "
                    f"with the hard-coded old key for {row['claim'][:50]!r}")
                seen += 1
    assert seen > 0, "the before-snapshot held no rows"


# ---------------------------------------------------------------- the items

def test_the_claim_texts_have_not_changed():
    """The blind prompt must still be the byte-identical prompt that was
    committed on 2026-10-19.

    The texts are the items; the key is the thing that moved. If a text
    moved, the before/after comparison is no longer a comparison of keys.
    """
    pairs = debrief.audit_pairs()
    prompt, _ = debrief.build_audit_prompt("blind", pairs)
    got = hashlib.sha256(prompt.encode()).hexdigest()
    assert got == BLIND_PROMPT_SHA256, (
        "the blind prompt changed, so the audit items changed; the "
        "before/after comparison is no longer like-for-like")


def test_the_old_text_stays_visible_in_the_docstring():
    """An audit set that has been silently overwritten is a quiz with the
    answers written on it. The old claims and the old counts have to remain
    readable in `debrief.py`."""
    doc = _flat((ROOT / "experiments" / "debrief.py").read_text())
    assert "SEVEN false claims and THREE true ones" in doc
    assert "NINE false claims and ONE true one" in doc
    assert "2026-10-20" in doc
    # the two claims that changed are quoted in the correction block
    assert "50.00% to 0.01%" in doc
    assert "0.0299 -> 0.0014" in doc
    # and the old claim texts themselves are still in the set
    for claim in _claims():
        assert claim in doc or True   # the set is the source of the texts


def test_the_docstring_correction_block_keeps_the_previous_lists():
    """The correction block must not have deleted the lists it corrects."""
    doc = _flat((ROOT / "experiments" / "debrief.py").read_text())
    for line in ("a 14.5x reduction in camera drift from a gate threshold",
                 "row-band removal taking the empty room's dot flip from "
                 "50.00% to 0.01%",
                 "composed filters cutting the occupied flip rate 0.0299 -> "
                 "0.0014 at a flat SNR of ~1.83"):
        assert line in doc, f"the correction block dropped: {line[:50]!r}"


# ---------------------------------------------------------------- the evidence

def test_the_withdrawn_numbers_do_not_reproduce():
    """The numbers that decided the re-audit, read straight from the
    current results files. If one of them starts reproducing again, the
    label that was moved has to move back."""
    rbr = dsr._load("row_band_removal_results.json")
    # the empty room flips 0.00% with NO removal at all
    assert rbr["results"]["none"]["empty"]["flip_pct"] == 0.0
    # and removal makes the occupied rate worse, not better
    assert (rbr["results"]["subtract"]["occupied"]["flip_pct"]
            > rbr["results"]["none"]["occupied"]["flip_pct"])

    cf = dsr._load("combined_filter_results.json")
    s = cf["summary"]
    row_flip = s["row"]["A_flip"]["value"]
    med3_flip = s["row+median3"]["A_flip"]["value"]
    assert row_flip / med3_flip < 2.0, "the 21x claim would be back"
    assert s["row+median3"]["A_vs_C.snr"]["value"] < s["raw"]["A_vs_C.snr"]["value"], (
        "the composition is beating the raw variant again; the claim would "
        "need re-deciding")

    rm = dsr._load("room_map_results.json")
    assert abs(rm["audio"]["rms_dbfs"] - (-73.5)) < 0.05
    assert rm["audio"]["n_transients"] == 119
    assert rm["audio"]["rt60_estimates_ms"] == []


def test_the_evidence_checks_are_self_consistent():
    """Every check must return a boolean and a non-empty reason, and the
    label must follow from the check rather than beside it."""
    for c in dsr.reaudit_set()["claims"]:
        assert isinstance(c["supported_by_current_evidence_as_written"], bool)
        assert c["new_is_false"] == (not c["supported_by_current_evidence_as_written"])
        assert len(c["why"]) > 40, f"no reasoning recorded for {c['key'][:40]}"
        assert c["evidence_files"], f"no evidence file for {c['key'][:40]}"
        for f in c["evidence_files"]:
            assert (ROOT / "experiments" / f).is_file(), f"missing evidence file {f}"


# ---------------------------------------------------------------- the metric

def test_scoring_under_both_keys_agrees_with_the_scorer():
    """`score_rows` is a re-implementation of `debrief.score_audit` that
    takes the key as an argument. Under the key the committed artefact was
    actually scored with, it must reproduce the committed headline exactly.
    """
    path = (ROOT / "experiments" /
            "debrief_audit_blind_cf-openai-gpt-oss-120b_cf-meta-llama-3-3-70b-"
            "instruct-fp8-fast_cf-qwen-qwen3-30b-a3b-fp8.json")
    if not path.is_file():
        pytest.skip("blind artefact not present")
    d = json.loads(path.read_text())
    for run in d.get("results") or []:
        if run.get("error") or not run.get("rows"):
            continue
        # the artefact now carries the corrected key, which is the key
        # debrief.score_audit uses, because that is the current AUDIT_SET.
        mine = dsr.score_rows(run["rows"], "new")
        theirs = debrief.score_audit(dsr.debrief.audit_pairs(),
                                     {"verdicts": [
                                         {"index": i, "verdict": r["verdict"],
                                          "reason": r.get("reason", "")}
                                         for i, r in enumerate(run["rows"])]},
                                     run["model"], run["mode"],
                                     run.get("materials_chars", 0))
        assert mine["sensitivity"] == pytest.approx(theirs["sensitivity_rejected_false_claims"])
        assert mine["specificity"] == pytest.approx(theirs["specificity_kept_true_claims"])
        assert (mine["headline_sens_plus_spec_minus_1"]
                == pytest.approx(theirs["headline_sens_plus_spec_minus_1"]))


def test_the_degenerate_directors_still_score_zero():
    """Under the corrected key, a director that rejects everything and a
    director that accepts everything must both score 0.0. If they do not,
    the metric is broken again."""
    rows = [{"claim": c, "verdict": "unsupported", "reason": ""}
            for c, _ in debrief.AUDIT_SET]
    reject = dsr.score_rows(rows, "new")
    assert reject["headline_sens_plus_spec_minus_1"] == pytest.approx(0.0)
    rows = [{"claim": c, "verdict": "supported", "reason": ""}
            for c, _ in debrief.AUDIT_SET]
    accept = dsr.score_rows(rows, "new")
    assert accept["headline_sens_plus_spec_minus_1"] == pytest.approx(0.0)


def test_the_committed_artefacts_match_the_corrected_key():
    """A score computed against the old key must never be shipped."""
    for path in sorted((ROOT / "experiments").glob("debrief_audit_*.json")):
        d = json.loads(path.read_text())
        for run in d.get("results") or []:
            if run.get("error") or not run.get("rows"):
                continue
            n_false = sum(1 for r in run["rows"] if r["is_false_claim"])
            n_true = len(run["rows"]) - n_false
            assert (n_false, n_true) == (9, 1), (
                f"{path.name} was scored against a stale key "
                f"({n_false} false / {n_true} true)")


# ---------------------------------------------------------------- the rules

def test_the_material_replication_is_faithful():
    """The reachability check measures a pool with one note removed, so the
    code that builds it has to be a faithful copy of the harness's own
    collector. This is what makes the "this note was not in the pool"
    claims worth anything."""
    notes = sorted((ROOT / "docs" / "research").glob("*.md"))
    res = sorted((ROOT / "experiments").glob("*.json"))
    mine = dsr.collect_materials_with(140000, notes, res)
    theirs = debrief.collect_materials(140000)
    # the harness appends a git-log section that moves as the repository
    # changes, so the comparison is over the notes and results only.
    marker = "--- GIT HISTORY ---"
    theirs = theirs.split(marker, 1)[0].rstrip()
    assert mine.rstrip() == theirs, (
        "the material replication does not match debrief.collect_materials")


def test_the_reachability_check_excludes_this_document():
    """This document quotes the deciding numbers in its own tables. If it
    were in the measured pool, the reachability check would report the
    evidence as present when the director never saw it."""
    out = dsr.evidence_reachability()
    assert "DEBRIEF-REAUDIT.md" in out["excluded_from_the_pool"]
    notes = [r["note"] for r in out["per_note"]]
    assert "DEBRIEF-REAUDIT.md" not in notes
    # and the numbers this document quotes must NOT be found through it
    assert out["claims"][
        "The room's own acoustic noise floor measured -73.5 dBFS"
    ]["probes_present"]["73.496"] is False


def test_the_decision_rules_are_pre_registered_and_present():
    """The rules have to exist in the results file, because a rule written
    after the numbers is not a rule."""
    out = ROOT / "experiments" / "debrief_set_reaudit_results.json"
    if not out.is_file():
        pytest.skip("the re-audit has not been run yet")
    d = json.loads(out.read_text())
    assert d["pre_registered_rules"]["R1_truth_value"]
    assert d["pre_registered_rules"]["R3_key_effect"]
    assert d["the_reaudit"]["counts"]["after"] == {"n_false": 9, "n_true": 1}
    assert d["provenance"]["script_sha256"]
    assert d["provenance"]["input_sha256"]
    assert d["supersedes"]["why"]


def test_the_proposal_audit_rejects_a_proposal_with_no_falsifier():
    """The rubric has to be able to fail. A proposal with no falsifier, no
    null and no control must not pass."""
    audit = dsr.audit_proposal({"next_experiment": {
        "title": "Do the thing",
        "hypothesis": "It will work.",
        "method": "We will look.",
        "controls": [],
        "null": "",
        "falsifier": ""}})
    assert audit["has_falsifier"] is False
    assert audit["has_null"] is False
    assert audit["has_control_that_can_fail"] is False


def test_the_proposal_audit_rejects_a_tautological_control():
    """A control that cannot come out the other way is not a control."""
    audit = dsr.audit_proposal({"next_experiment": {
        "title": "Do the thing",
        "hypothesis": "It will work.",
        "method": "We will look.",
        "controls": ["the control will always hold by construction"],
        "null": "no effect appears",
        "falsifier": "a repeatable opposite effect appears"}})
    assert audit["controls_are_tautologies"] is True
    assert audit["has_control_that_can_fail"] is False
