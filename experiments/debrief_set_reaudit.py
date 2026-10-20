"""
debrief_set_reaudit.py -- re-audit the audit set, after the night the stride was found.

WHAT HAPPENED.

`experiments/debrief.py` audits an LLM "director" against ten claims: seven
this project has retracted or falsified, three that are still standing. The
ground truth lives in the script and is applied after the model has answered.

On 2026-10-19 every camera reader in this repository was reading the captures
at the wrong frame stride: 23,040 bytes per frame assumed, 19,200 bytes per
frame on disk. 80% of every "frame" was a tear between two real moments, so
the numbers behind several of those ten claims were computed on garbage.
`docs/research/NV12-STRIDE.md` is the full account.

**The answer key changed under the quiz.** Two of the three standing claims
are now false, and the evidence under two of the seven retracted claims has
been replaced. This file re-derives the truth value of all ten from the
CURRENT results files, records the evidence for each decision, re-runs the
audit against the corrected key, and audits the director's proposed next
experiment against this repository's own standard.

THE RULE, PRE-REGISTERED BEFORE ANY MODEL WAS CALLED.

    A claim is TRUE if and only if the CURRENT results files support it AS
    WRITTEN. A claim whose cited numbers have been withdrawn is FALSE as
    written, EVEN IF a related conclusion survives on different evidence.

That second sentence is the one that decides two of the ten claims, so it is
worth stating out loud: the audit set's items are the claims the model is
asked to judge, and the model is asked to judge the words in front of it. If
the words cite a number that no longer reproduces, the claim is false, and a
director that endorses it has made an error -- even where the underlying
subject has since been re-measured and the new measurement is favourable.

Every decision below is made by a CHECK that reads a results file and returns
a boolean. The label is not asserted here and then defended in prose: the
check derives it. `tests/test_debrief_reaudit.py` re-runs every check, so a
label that drifts back without its evidence drifting with it fails the build.

WHAT THIS FILE DOES NOT DO.

It does not change the instrument. `debrief.py`'s prompt, its scorer, its
shuffler and its seed are untouched; the only edit to that file is the two
labels this file derives and a dated correction block in its docstring that
keeps the old text visible. If the instrument needs changing -- and it may --
that is reported in `docs/research/DEBRIEF-REAUDIT.md` as a finding, not made
here.

The material pool that CONTEXT mode reads is `experiments/*.json` plus
`docs/research/*.md`, so this file's own output is part of the evidence the
next director sees. That is a property of the instrument, not a bug in it,
and it is recorded in the results file.

USAGE:
    python experiments/debrief_set_reaudit.py --phase before   # snapshot the old artefacts
    python experiments/debrief_set_reaudit.py --phase main     # everything else
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import debrief  # noqa: E402
from director_proposal_stability import _as_text, jaccard  # noqa: E402

RESULTS = ROOT / "experiments"
SCRATCH = ROOT / ".scratch" / "debrief_reaudit"
BEFORE_PATH = SCRATCH / "before.json"
OUT_PATH = RESULTS / "debrief_set_reaudit_results.json"

MODELS = [
    "@cf/openai/gpt-oss-120b",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/qwen/qwen3-30b-a3b-fp8",
]

# The material budget each model is given in CONTEXT mode. It is FIXED per
# model so that the repeats of one model are one prompt, not N prompts.
# llama-3.3-70b has a 24 000-token window; 140 000 characters does not fit
# it, and the adaptive ladder dropped it to 60 000 in the committed run.
CONTEXT_LIMIT = {
    "@cf/openai/gpt-oss-120b": 140000,
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast": 60000,
    "@cf/qwen/qwen3-30b-a3b-fp8": 140000,
}

N_BLIND = 4          # repeats per model, blind mode
N_CONTEXT = 3        # repeats per model, context mode
SWEEP_LIMITS = (20000, 60000, 140000, 240000)
SWEEP_MODEL = "@cf/openai/gpt-oss-120b"
N_SWEEP = 3


# ------------------------------------------------------------------ helpers

def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------- the rule

PRE_REGISTERED_RULES = {
    "registered": "2026-10-20, before any model was called for this file",
    "R1_truth_value": (
        "A claim is TRUE if and only if the CURRENT results files support it "
        "AS WRITTEN. A claim whose cited numbers have been withdrawn is "
        "FALSE as written, even if a related conclusion survives on "
        "different evidence. The truth value is DERIVED by a check that "
        "reads a results file, never asserted in prose."),
    "R2_class_balance": (
        "The composition of the set is whatever the evidence yields. It is "
        "not topped up to keep the set balanced, and the number of TRUE "
        "items is reported even when it makes specificity a one-bit "
        "measurement."),
    "R3_key_effect": (
        "The effect of the key change is measured by re-scoring the SAME "
        "verdict rows under the old and the new key. The effect of the run "
        "is measured by repeating the same prompt. The two are never "
        "confounded, and the headline quoted for the corrected key is the "
        "MEDIAN over the repeats, with min, max and spread."),
    "R4_spread": (
        "The hosted endpoint is not deterministic at temperature 0 (measured "
        "elsewhere in this repository at a 0.619 headline spread). A single "
        "run is a single sample. Every headline quoted in the results file "
        "and in the write-up carries its run count and its spread."),
    "R5_proposal_audit": (
        "The proposal passes if it has a falsifier, at least one control "
        "that could have failed, does not duplicate an experiment the "
        "repository's own notes already name, and does not re-propose "
        "something the repository has already falsified. Duplication and "
        "falsified-claim-revisit are checked by content-token overlap "
        "against the repository's own lists, using the same tokeniser the "
        "repository already uses in director_proposal_stability.py."),
    "R6_materials_sweep": (
        "Model held constant, budget varied. The null is 'more evidence does "
        "not make the director better'. The effect is called only if the "
        "headline at the LARGER budget exceeds the headline at the SMALLER "
        "budget by more than the run-to-run spread observed AT THE SMALLER "
        "BUDGET. Otherwise the answer is 'not resolved', and 'not resolved' "
        "is reported as 'not resolved'."),
    "R7_no_key_leak": (
        "The audit's ground truth comes from the results files. It is never "
        "taken from the claims themselves, and the CONTEXT materials are "
        "checked for the answer key before the context score is quoted."),
}


# ------------------------------------------------------------- the evidence
#
# One function per claim. Each returns:
#   supported : bool -- is the claim AS WRITTEN supported by the current
#               results files?  The label FOLLOWS from this boolean.
#   why       : str  -- the reasoning, in the repository's plain language.
#   numbers   : dict -- the deciding numbers, so a reader can check them.
#   files     : list -- the results files consulted.
#
# The thresholds inside the checks are the ones the repository already uses
# (a 0.5 rate difference, a 2.0 growth exponent, an exact figure match), and
# each is written into the check so that the threshold is visible next to the
# number it is applied to.

def _ev_gate_14_5x() -> dict:
    """The gate's field threshold did NOT remove 14.5x of the drift."""
    v = _load("gate_peak_verdict_results.json")
    cov = _load("gate_peak_coverage_results.json")
    numbers = {
        "gate_peak_verdict.median_ratio": v.get("median_ratio"),
        "gate_peak_verdict.worst_ratio": v.get("worst_ratio"),
        "gate_peak_verdict.field_matches_own_interval": v.get("field_matches_own_interval"),
        "gate_peak_verdict.verdict": (v.get("verdict") or "")[:300],
        "gate_peak_coverage.max_understatement_x":
            cov.get("summary", {}).get("max_understatement_x"),
    }
    # The claim survives only if the understatement is still there when the
    # gate is compared against the record's OWN interval. The verdict file
    # measured exactly that: the worst ratio over the record's own interval.
    supported = float(numbers["gate_peak_verdict.worst_ratio"] or 0) >= 14.5
    return {
        "supported": supported,
        "why": ("The 14.5x was a comparison over two different intervals: the "
                "gate's peak against records that did not contain the peak. "
                "Over the record's OWN interval the ratio is 1.0, and the "
                "verdict file says the earlier figure is withdrawn. The "
                "claim is false and its evidence is unchanged by the stride."),
        "numbers": numbers,
        "files": ["gate_peak_verdict_results.json",
                  "gate_peak_coverage_results.json"],
    }


def _ev_ascii_48_19() -> dict:
    """The ASCII map: occupied 48.19% against an empty room's 50.00%."""
    rbr = _load("row_band_removal_results.json")
    prev = rbr["supersedes"]["previous_results"]
    nulls = _load("ascii_motion_nulls_results.json")
    empty = _load("emptyroom_null_results.json")
    cur_empty = rbr["results"]["none"]["empty"]["flip_pct"]
    cur_occ = rbr["results"]["none"]["occupied"]["flip_pct"]
    numbers = {
        "row_band_removal.supersedes.previous_results.none": prev["none"],
        "row_band_removal.results.none.empty.flip_pct": cur_empty,
        "row_band_removal.results.none.occupied.flip_pct": cur_occ,
        "ascii_motion_nulls.verdict": nulls.get("verdict"),
        "emptyroom_null.empty_over_occupied_flip_rate":
            empty.get("empty_over_occupied_flip_rate"),
    }
    # The claim states the empty room flips 50.00% of its cells. It is
    # supported only if the CURRENT measurement still says that.
    supported = abs(float(cur_empty) - 50.0) < 0.05
    return {
        "supported": supported,
        "why": ("Both numbers in the claim were the tear. On the correct "
                "frames the empty room flips 0.00% of its cells and the "
                "occupied capture 0.62%, and the null test now verdicts "
                "SEPARATED -- the delta carries motion. So the CLAIM AS "
                "WRITTEN is false (its two numbers do not reproduce), even "
                "though the conclusion it draws is now supported by "
                "different evidence. That is the case this file's rule R1 "
                "exists for, and it is also the case a reader of the notes "
                "would catch."),
        "numbers": numbers,
        "files": ["row_band_removal_results.json",
                  "ascii_motion_nulls_results.json",
                  "emptyroom_null_results.json"],
    }


def _ev_watcher_quadratic() -> dict:
    """The watcher's memory does not grow quadratically."""
    w = _load("watcher_cost_results.json")
    exps = w.get("growth_exponents") or {}
    max_all = max((float(v.get("ungated_all", 0)) for v in exps.values()),
                  default=0.0)
    max_ge4 = max((float(v.get("ungated_ge4s", 0)) for v in exps.values()),
                  default=0.0)
    numbers = {
        "watcher_cost.growth_exponents": exps,
        "max_exponent_over_all_lengths": max_all,
        "max_exponent_beyond_4s": max_ge4,
        "watcher_cost.question": w.get("question"),
    }
    # A quadratic claim needs an exponent at or above 2. Nothing here is.
    supported = max_all >= 2.0
    return {
        "supported": supported,
        "why": ("The growth exponents are all below 1.2 over the whole "
                "length range and below 0.1 beyond four seconds, and the "
                "buffer is bounded by the analysis window rather than "
                "growing without bound. The original 'quadratic' reading "
                "was a numpy-fed harness artefact worth about 2x. Audio, "
                "not video: the stride cannot reach it."),
        "numbers": numbers,
        "files": ["watcher_cost_results.json"],
    }


def _ev_gain_43_3() -> dict:
    """Gain normalisation did not cut the drift by fifty times."""
    empty = _load("emptyroom_null_results.json")
    occ = empty["captures"][0]
    g = _load("gain_invariance_results.json")
    raw = float(occ.get("adjacent_mad_raw", 0))
    norm = float(occ.get("adjacent_mad_gain_normalised", 0))
    numbers = {
        "emptyroom_null.captures[0].adjacent_mad_raw": raw,
        "emptyroom_null.captures[0].adjacent_mad_gain_normalised": norm,
        "gain_invariance.empty_over_occupied_normalised":
            g.get("empty_over_occupied_normalised"),
        "gain_invariance.verdict": (g.get("verdict") or "")[:400],
    }
    # The claim names two numbers: 43.3 before and 0.81 after.
    supported = (abs(raw - 43.3) < 0.5) and (abs(norm - 0.81) < 0.05)
    return {
        "supported": supported,
        "why": ("The 43.3 was the torn stride. On the correct frames the "
                "occupied capture's mean consecutive |difference| is 0.615 "
                "counts BEFORE any normalisation -- the auto-exposure hunt "
                "was the tear, not a real drift. The claim stays false, but "
                "the reason has changed: it is not that normalisation "
                "fails to deliver 50x, it is that there was no 43.3 to "
                "reduce."),
        "numbers": numbers,
        "files": ["emptyroom_null_results.json", "gain_invariance_results.json"],
    }


def _ev_local_llm_ascii() -> dict:
    """A local LLM cannot label speech from a rendering, over a blank."""
    v = _load("local_vlm_vision_results.json")
    conds = v["vision_task"]["conditions"]
    speak = float(conds["speak"]["said_speaking_rate"])
    silent = float(conds["silent"]["said_speaking_rate"])
    blank = float(conds["blank"]["said_speaking_rate"])
    numbers = {
        "local_vlm_vision.vision_task.speak.said_speaking_rate": speak,
        "local_vlm_vision.vision_task.silent.said_speaking_rate": silent,
        "local_vlm_vision.vision_task.blank.said_speaking_rate": blank,
        "margin_over_blank_control": speak - blank,
        "local_vlm_vision.verdict.reason": (v.get("verdict") or {}).get("reason", "")[:400],
    }
    # "A useful margin over a blank control": a rate difference of at least
    # half the condition rate. Here the margin is exactly zero.
    supported = (speak - blank) >= 0.5
    return {
        "supported": supported,
        "why": ("The local model says SPEAKING to every frame of the "
                "speaking condition, every frame of the silent condition "
                "and every frame of the BLANK control -- all three rates "
                "are 1.000, so the margin over the blank control is "
                "exactly 0.000. It is a constant answer, and a constant "
                "answer is not a detector."),
        "numbers": numbers,
        "files": ["local_vlm_vision_results.json"],
    }


def _ev_31_8() -> dict:
    """The 31.8% figure was never a measurement."""
    w = _load("watcher_cost_results.json")
    measured = []
    for name, cell in (w.get("grid") or {}).items():
        for variant, body in cell.items():
            if isinstance(body, dict) and "cpu_pct" in body:
                measured.append(float(body["cpu_pct"]))
    numbers = {
        "watcher_cost.question": w.get("question"),
        "n_measured_cpu_cells": len(measured),
        "max_measured_cpu_pct": max(measured) if measured else None,
        "median_measured_cpu_pct": statistics.median(measured) if measured else None,
        "any_cell_is_31_8": any(abs(p - 31.8) < 0.05 for p in measured),
        "measured_ungated_tone200_30s":
            ((w.get("mechanism") or {}).get("tone200_30s_ungated") or {}).get("cpu_pct"),
    }
    # The claim is a figure of 31.8%. It is supported only if 31.8 is
    # somewhere in the measured cells. It is not.
    supported = bool(numbers["any_cell_is_31_8"])
    return {
        "supported": supported,
        "why": ("Three figures for the ungated watcher were in circulation "
                "-- 31.8%, 39.3% and 116.9% -- and 31.8% is the one that "
                "was never a measurement: no script, fixture, length or "
                "clock was ever recorded for it. The measured cells in the "
                "current results file do not contain it. The number was "
                "also a CPU percentage, not a count of unmeasured frames."),
        "numbers": numbers,
        "files": ["watcher_cost_results.json"],
    }


def _ev_row_median() -> dict:
    """Row-median removal: 50.00% -> 0.01%, leaving the subject alone."""
    r = _load("row_band_removal_results.json")
    res = r["results"]
    prev = r["supersedes"]["previous_results"]
    numbers = {
        "row_band_removal.results.none.empty.flip_pct":
            res["none"]["empty"]["flip_pct"],
        "row_band_removal.results.subtract.empty.flip_pct":
            res["subtract"]["empty"]["flip_pct"],
        "row_band_removal.results.none.occupied.flip_pct":
            res["none"]["occupied"]["flip_pct"],
        "row_band_removal.results.subtract.occupied.flip_pct":
            res["subtract"]["occupied"]["flip_pct"],
        "row_band_removal.supersedes.previous_results.subtract.empty":
            prev["subtract"]["empty"],
        "row_band_removal.supersedes.previous_results.none.occupied":
            prev["none"]["occupied"],
        "row_band_removal.verdict": (r.get("verdict") or "")[:400],
    }
    # The claim states a before and an after: 50.00% and 0.01%.
    supported = (abs(float(res["none"]["empty"]["flip_pct"]) - 50.0) < 0.05 and
                 abs(float(res["subtract"]["empty"]["flip_pct"]) - 0.01) < 0.005)
    return {
        "supported": supported,
        "why": ("LABEL CHANGED. The claim's premise WAS the tear: the "
                "empty room's 50.00% was a stride artefact, and with no "
                "filter at all the empty room now flips 0.00% of its "
                "cells. There is no banding for the filter to remove, and "
                "removal makes the occupied rate WORSE (1.79% against "
                "0.62% with none). The claim's second half -- that the "
                "occupied room is left alone -- is also false. The "
                "retraction of this claim is in "
                "docs/research/ROW-BAND-REMOVAL.md and NV12-STRIDE.md."),
        "numbers": numbers,
        "files": ["row_band_removal_results.json"],
    }


def _ev_teacher_loo() -> dict:
    """The teacher: 1.000 on a temporal split, 0.000 on a held-out session."""
    m = _load("teacher_conventional_speaking_map.json")
    loo = {r["held_out"]: r.get("acc") for r in m.get("leave_one_session_out") or []}
    numbers = {
        "teacher_conventional_speaking_map.acc_test": m.get("acc_test"),
        "teacher_conventional_speaking_map.leave_one_session_out": loo,
        "teacher_conventional_speaking_map.acc_test_spatially_scrambled":
            m.get("acc_test_spatially_scrambled"),
        "teacher_conventional_speaking_map.acc_test_blank": m.get("acc_test_blank"),
        "teacher_conventional_speaking_map.base_rate_test": m.get("base_rate_test"),
        "teacher_conventional_speaking_map.margin_over_best_control":
            m.get("margin_over_best_control"),
        "teacher_conventional_speaking_map.usable": m.get("usable"),
    }
    # The claim states 1.000 on the temporal split AND 0.000 on the held-out
    # session. Both halves have to hold.
    supported = (abs(float(m.get("acc_test", 0)) - 1.0) < 1e-9 and
                 any(abs(float(a) - 0.0) < 1e-9 for a in loo.values()))
    return {
        "supported": supported,
        "why": ("The claim is false, but the evidence under it has been "
                "replaced. The motion-map feature set still scores 1.000 "
                "on the temporal split, and now scores 0.15 (held out A) "
                "and 0.375 (held out B) where it used to score 0.000. Its "
                "1.000 is matched by a spatially scrambled control at "
                "0.9375 and its blank control is the 0.667 base rate, so "
                "the margin over the best control is 0.0625 and the "
                "pipeline is marked unusable. The teacher has not learned "
                "the recording; it has learned nothing."),
        "numbers": numbers,
        "files": ["teacher_conventional_speaking_map.json"],
    }


def _ev_room_acoustics() -> dict:
    """The room: -73.5 dBFS, 119 transients, zero valid RT60."""
    rm = _load("room_map_results.json")
    a = rm["audio"]
    numbers = {
        "room_map.audio.rms_dbfs": a.get("rms_dbfs"),
        "room_map.audio.peak_95pct": a.get("peak_95pct"),
        "room_map.audio.n_transients": a.get("n_transients"),
        "room_map.audio.rt60_estimates_ms": a.get("rt60_estimates_ms"),
        "room_map.audio.rt60_median_ms": a.get("rt60_median_ms"),
        "room_map.audio.duration_s": a.get("duration_s"),
    }
    supported = (abs(float(a.get("rms_dbfs", 0)) - (-73.5)) < 0.05 and
                 a.get("n_transients") == 119 and
                 not a.get("rt60_estimates_ms") and
                 a.get("rt60_median_ms") is None)
    return {
        "supported": supported,
        "why": ("SURVIVES. This is an AUDIO measurement of a recording, and "
                "the frame stride is a video defect, so the night could not "
                "have touched it. The current results file carries "
                "-73.496 dBFS, 119 transients and an empty RT60 list. It is "
                "the only one of the ten that is still standing, and the "
                "set now holds NINE false claims and ONE true one."),
        "numbers": numbers,
        "files": ["room_map_results.json"],
    }


def _ev_combined_filters() -> dict:
    """Composed filters: 0.0299 -> 0.0014, 21x, SNR flat at 1.83."""
    c = _load("combined_filter_results.json")
    s = c["summary"]
    row_flip = float(s["row"]["A_flip"]["value"])
    med3_flip = float(s["row+median3"]["A_flip"]["value"])
    snr_row = float(s["row"]["A_vs_C.snr"]["value"])
    snr_med3 = float(s["row+median3"]["A_vs_C.snr"]["value"])
    snr_raw = float(s["raw"]["A_vs_C.snr"]["value"])
    numbers = {
        "combined_filter.summary.row.A_flip": row_flip,
        "combined_filter.summary.row+median3.A_flip": med3_flip,
        "combined_filter.improvement_ratio": row_flip / med3_flip,
        "combined_filter.summary.row.A_vs_C.snr": snr_row,
        "combined_filter.summary.row+median3.A_vs_C.snr": snr_med3,
        "combined_filter.summary.raw.A_vs_C.snr": snr_raw,
        "combined_filter.supersedes.previous_variants.row.A_flip":
            c["supersedes"]["previous_variants"]["row"]["A_flip"],
        "combined_filter.supersedes.previous_variants.row+median3.A_flip":
            c["supersedes"]["previous_variants"]["row+median3"]["A_flip"],
        "combined_filter.supersedes.previous_variants.row+median3.A_vs_C.snr":
            c["supersedes"]["previous_variants"]["row+median3"]["A_vs_C"]["snr"],
        "combined_filter.verdict": (c.get("verdict") or "")[:400],
    }
    # The claim names four numbers: 0.0299, 0.0014, 21x, and SNR 1.83.
    supported = (abs(row_flip - 0.0299) < 1e-4 and
                 abs(med3_flip - 0.0014) < 1e-4 and
                 abs((row_flip / med3_flip) - 21.0) < 1.0 and
                 abs(snr_med3 - 1.83) < 0.01)
    return {
        "supported": supported,
        "why": ("LABEL CHANGED, AND THE CLAIM TEXT WAS NEVER FIXED. The "
                "docstring of debrief.py says this claim was 'corrected' "
                "once; what was corrected was its LABEL, not its TEXT, and "
                "the text still asserts 0.0299, 0.0014, 21x and an SNR of "
                "1.83. Every one of those numbers was measured on torn "
                "frames. On the correct frames the composition moves the "
                "occupied flip rate by 1.14x, not 21x, and the SNR is 1.386 "
                "against 1.410 for the RAW variant -- the raw "
                "representation is the better one, and the premise is "
                "withdrawn."),
        "numbers": numbers,
        "files": ["combined_filter_results.json"],
    }


# The evidence table, keyed by a stable prefix of each claim's text. The
# prefix is the identity of the item; the shuffle moves the items around,
# but the items themselves do not change.
#
# Each entry is MEMOISED: the checks read results files, and scoring a run
# asks for the key once per claim. The files do not change under a run, so
# the memo is safe, and it keeps the sweep from re-reading a 194 kB file
# twenty times.
EVIDENCE = {
    "Reducing the gate's field threshold to 0.30": _ev_gate_14_5x,
    "The ASCII character map reliably detects": _ev_ascii_48_19,
    "The watcher's memory use grows quadratically": _ev_watcher_quadratic,
    "Applying a per-frame gain normalisation": _ev_gain_43_3,
    "A local large language model given an ASCII rendering": _ev_local_llm_ascii,
    "31.8% of the frames the watcher processes": _ev_31_8,
    "Removing each row's median across the columns": _ev_row_median,
    "A conventional teacher model trained on motion maps": _ev_teacher_loo,
    "The room's own acoustic noise floor measured -73.5 dBFS": _ev_room_acoustics,
    "Composing row-wise band removal with a temporal median": _ev_combined_filters,
}
EVIDENCE = {k: functools.lru_cache(maxsize=None)(fn)
            for k, fn in EVIDENCE.items()}

# The labels the set carried before tonight, hard-coded so that the "before"
# column does not depend on the file that is being corrected. Cross-checked
# against the committed artefacts' own rows.
OLD_LABELS = {
    "Reducing the gate's field threshold to 0.30": True,
    "The ASCII character map reliably detects": True,
    "The watcher's memory use grows quadratically": True,
    "Applying a per-frame gain normalisation": True,
    "A local large language model given an ASCII rendering": True,
    "31.8% of the frames the watcher processes": True,
    "Removing each row's median across the columns": False,
    "A conventional teacher model trained on motion maps": True,
    "The room's own acoustic noise floor measured -73.5 dBFS": False,
    "Composing row-wise band removal with a temporal median": False,
}


def _key_for(claim: str) -> str:
    for k in EVIDENCE:
        if claim.startswith(k):
            return k
    raise KeyError(f"no evidence entry for claim: {claim[:60]!r}")


def reaudit_set() -> dict:
    """Re-derive the truth value of every claim from the current results.

    Returns the claim-by-claim table, the before/after counts, and the
    labels that changed.
    """
    claims = []
    for idx, (claim, old_is_false_in_file) in enumerate(debrief.AUDIT_SET):
        key = _key_for(claim)
        ev = EVIDENCE[key]()
        new_is_false = not ev["supported"]
        claims.append({
            "index_in_source_order": idx,
            "key": key,
            "claim": claim,
            "old_is_false": OLD_LABELS[key],
            "old_label": "FALSE" if OLD_LABELS[key] else "TRUE",
            "old_label_matches_the_committed_file": OLD_LABELS[key] == old_is_false_in_file,
            "new_is_false": new_is_false,
            "new_label": "FALSE" if new_is_false else "TRUE",
            "label_changed": OLD_LABELS[key] != new_is_false,
            "supported_by_current_evidence_as_written": ev["supported"],
            "why": ev["why"],
            "deciding_numbers": ev["numbers"],
            "evidence_files": ev["files"],
        })
    n_false = sum(1 for c in claims if c["new_is_false"])
    return {
        "rule": PRE_REGISTERED_RULES["R1_truth_value"],
        "claims": claims,
        "counts": {
            "before": {"n_false": sum(1 for c in claims if c["old_is_false"]),
                       "n_true": sum(1 for c in claims if not c["old_is_false"])},
            "after": {"n_false": n_false,
                      "n_true": len(claims) - n_false},
        },
        "labels_changed": [
            {"key": c["key"], "old": c["old_label"], "new": c["new_label"],
             "why": c["why"]}
            for c in claims if c["label_changed"]
        ],
    }


# --------------------------------------------------------------- the scoring

def label_map(key_name: str) -> dict:
    """claim text -> is_false, under the old or the new key."""
    out = {}
    for claim, _ in debrief.AUDIT_SET:
        k = _key_for(claim)
        out[claim] = OLD_LABELS[k] if key_name == "old" else (
            not EVIDENCE[k]()["supported"])
    return out


def score_rows(rows: list, key_name: str) -> dict:
    """Score a set of verdict rows under one key.

    This is a re-implementation of `debrief.score_audit` that takes the key
    as an argument, so the SAME verdicts can be scored under the old key and
    the new key. The formula is the repository's: sens + spec - 1, with the
    lenient variant beside it.
    """
    lab = label_map(key_name)
    n_false = n_true = det = fp = tp = unc = unparse = 0
    out = []
    for r in rows:
        claim = r.get("claim", "")
        is_false = lab.get(claim)
        if is_false is None:
            continue
        verdict = str(r.get("verdict", "")).strip().lower()
        if verdict not in ("supported", "unsupported", "uncertain"):
            verdict = "unparseable"
        rejected = verdict == "unsupported"
        correct = (rejected == is_false) if verdict != "unparseable" else False
        if is_false:
            n_false += 1
            if correct:
                det += 1
        else:
            n_true += 1
            if rejected:
                fp += 1
            if verdict == "supported":
                tp += 1
        if verdict == "uncertain":
            unc += 1
        if verdict == "unparseable":
            unparse += 1
        out.append({"claim": claim, "is_false_claim": is_false,
                    "verdict": verdict, "rejected": rejected,
                    "correct": correct,
                    "reason": str(r.get("reason", ""))[:300]})
    sens = det / n_false if n_false else None
    spec = tp / n_true if n_true else None
    headline = None if (sens is None or spec is None) else sens + spec - 1.0
    not_endorsed = sum(1 for x in out if x["is_false_claim"]
                       and x["verdict"] != "supported")
    not_accused = sum(1 for x in out if (not x["is_false_claim"])
                       and x["verdict"] != "unsupported")
    sens_l = not_endorsed / n_false if n_false else None
    spec_l = not_accused / n_true if n_true else None
    headline_l = None if (sens_l is None or spec_l is None) else sens_l + spec_l - 1.0
    return {"key": key_name, "sensitivity": sens, "specificity": spec,
            "headline_sens_plus_spec_minus_1": headline,
            "sensitivity_not_endorsing": sens_l, "specificity_not_accusing": spec_l,
            "headline_lenient": headline_l,
            "detections": det, "n_false": n_false, "n_true": n_true,
            "false_accusations": fp, "true_positives": tp,
            "uncertain": unc, "unparseable": unparse, "rows": out}


def score_artefact(path: Path) -> dict:
    """Score every run in a saved audit artefact under BOTH keys."""
    d = json.loads(path.read_text())
    out = []
    for r in d.get("results") or []:
        if r.get("error") or not r.get("rows"):
            out.append({"model": r.get("model"), "error": r.get("error")})
            continue
        out.append({
            "model": r.get("model"), "mode": r.get("mode"),
            "materials_chars": r.get("materials_chars"),
            "old_key": score_rows(r["rows"], "old"),
            "new_key": score_rows(r["rows"], "new"),
        })
    return {"file": path.name, "mode": d.get("mode"), "runs": out}


# ------------------------------------------------------------- the artefacts

def artefact_paths() -> list:
    return sorted(RESULTS.glob("debrief_audit_*.json"))


def capture_before() -> dict:
    """Snapshot the committed artefacts BEFORE they are overwritten.

    The committed artefacts carry the old key in their rows, and the whole
    point of the "before" column is that it is the state of the record as it
    stood before tonight's re-run. So it is captured first, and the snapshot
    is written to the scratch directory, which is git-ignored and is not
    part of the material pool.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    snap = {"captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "artefacts": [score_artefact(p) for p in artefact_paths()],
            "file_sha256": {p.name: sha256_file(p) for p in artefact_paths()}}
    BEFORE_PATH.write_text(json.dumps(snap, indent=2))
    return snap


# ------------------------------------------------------------ the proposals

# The repository's own next steps, as the notes state them. This is the list
# the director's proposal is checked against for DUPLICATION.
REPO_NEXT_STEPS = {
    "condition_d": (
        "Record condition D: turned AND silent. The single most valuable "
        "missing take; it makes every pose fold identifiable and separates "
        "'the head turned' from 'something else moved'."),
    "held_out_device": (
        "A held-out-device measurement, proxied by the gate's advantage "
        "under cache pressure on a low-memory target (ADOPTION-PLAN item 11)."),
    "real_two_mic_dialogue": (
        "Record real two-microphone dialogue: the one measurement the AMI "
        "corpus cannot supply (ADOPTION-PLAN item 10)."),
    "sessions_per_class": (
        "The teacher needs at least 63 independent sessions per class, and "
        "the task as constructed cannot be measured at all until it has "
        "within-session label variation."),
    "re_record_shared_clock": (
        "Re-record with the new capture tool, which writes one video frame, "
        "one audio block and one clock reading per iteration, so the "
        "frame-level audio-video alignment exists at all."),
}

# The claims this repository has already retracted. A proposal that
# re-proposes one of these is the dangerous case, and it is checked
# separately from duplication.
REPO_FALSIFIED = {
    "gate_14_5x": ("the gate's field threshold understates the gate by "
                   "14.5x (different intervals)"),
    "ascii_saturation": ("the ASCII character field flips 50% of its cells in "
                         "an empty room, so the delta is a dither map"),
    "dither_gradient": ("the dither gradient is a monotone 3.1x"),
    "watcher_saturation": ("watcher cost above 100% is a realtime saturation "
                           "floor"),
    "led_flicker_banding": ("LED flicker banding, a perfect 50.00% of pixels "
                            "in an empty room"),
    "gain_50x": ("gain normalisation cut the drift from 43.3 to 0.81 counts, "
                 "a fifty-fold reduction"),
    "row_band_removal": ("row-wise band removal takes the empty room from "
                         "50.00% to 0.01%"),
    "composed_filters": ("composing row removal with a temporal median cuts "
                         "the occupied flip rate 0.0299 -> 0.0014"),
    "teacher_usable": ("the teacher's USABLE verdicts (+0.208, +0.312)"),
    "pose_1_71x": ("a pose feature at 1.71x"),
    "figure_31_8": ("the 31.8% figure"),
    "teacher_all_zero": ("the conventional teacher scores 0.000 "
                         "leave-one-out for every feature set"),
}


def _ratio_bounds(text: str) -> list:
    """The comparison bounds in a proposal field, as (op, value).

    A number immediately followed by a multiplication sign is an ADVANTAGE
    multiplier ("1.1x advantage"), not a ratio, and is skipped: mixing the
    two quantities is how a contradiction gets manufactured by accident.
    """
    out = []
    for m in re.finditer(r"(≤|≥|<=|>=|<|>|=)\s*([0-9]+(?:\.[0-9]+)?)(\s*[×x])?", text):
        if m.group(3) and "×" in m.group(3):
            continue
        op = {"<=": "≤", ">=": "≥"}.get(m.group(1), m.group(1))
        out.append((op, float(m.group(2))))
    return out


def _satisfies(op: str, value: float, bound: float) -> bool:
    return {">": value > bound, "<": value < bound, "≥": value >= bound,
            "≤": value <= bound, "=": value == bound}.get(op, False)


def decision_structure(hypothesis: str, null: str, margin: str,
                       falsifier: str) -> dict:
    """Can the proposal's own decision structure come out the other way?

    A proposal can have all five required fields and still be unable to
    fail, which is the shape of claim this repository has had to retract
    four times. Two mechanical checks catch the two ways that happens here:

      1. THE NULL TRIGGERS THE FALSIFIER. If the null's own value also
         satisfies the falsifier's condition, then a null result and a
         falsifying result are the same observation, and the null cannot
         do its job.
      2. THE HYPOTHESIS AND THE MARGIN DISAGREE. If the hypothesis says the
         effect is LESS THAN N times and the margin says it must be AT LEAST
         N times, the proposal asks for two opposite things and one of them
         was written by mistake.
    """
    out = {"null_ratio_bounds": _ratio_bounds(null),
           "falsifier_ratio_bounds": _ratio_bounds(falsifier),
           "margin_ratio_bounds": _ratio_bounds(margin)}
    null_vals = [v for _, v in out["null_ratio_bounds"]]
    triggers = []
    for nv in null_vals:
        for op, bound in out["falsifier_ratio_bounds"]:
            if _satisfies(op, nv, bound):
                triggers.append({"null_value": nv, "falsifier_op": op,
                                 "falsifier_bound": bound})
    out["null_triggers_the_falsifier"] = bool(triggers)
    out["null_falsifier_overlap"] = triggers

    hyp_m = re.search(r"(less than|fewer than|below|under|shrinks? to less than)"
                       r"\s+(?:a|an|the)?\s*([0-9]+(?:\.[0-9]+)?)\s*[×x]",
                       hypothesis, re.I)
    mar_m = re.search(r"(at least|more than|greater than|≥|>=)\s*"
                       r"(?:a|an|the)?\s*([0-9]+(?:\.[0-9]+)?)\s*[×x]",
                       margin)
    out["hypothesis_multiplier"] = float(hyp_m.group(2)) if hyp_m else None
    out["margin_multiplier"] = float(mar_m.group(2)) if mar_m else None
    out["hypothesis_margin_contradiction"] = bool(
        hyp_m and mar_m and
        abs(float(hyp_m.group(2)) - float(mar_m.group(2))) < 1e-9)
    return out


def audit_evidence_review(debrief_obj: dict) -> dict:
    """Check the debrief's REVIEW of the evidence, not just its proposal.

    The dangerous case is not the proposal -- it is the debrief's list of
    what the evidence SUPPORTS. A claim the repository has retracted
    appearing there is the director endorsing a retraction, which is the
    failure mode this whole lane exists to look for.
    """
    supported = [str(c.get("claim", "")) for c in
                 (debrief_obj or {}).get("state_of_the_evidence") or []]
    unsafe = [str(c.get("claim", "")) for c in
              (debrief_obj or {}).get("unsafe_claims") or []]
    out = {"n_supported": len(supported), "n_unsafe": len(unsafe),
           "endorses_a_falsified_claim": [], "flags_a_falsified_claim": [],
           "supported_vs_falsified": {}, "unsafe_vs_falsified": {}}
    for k, v in REPO_FALSIFIED.items():
        best_s = max((round(jaccard(s, v), 4) for s in supported), default=0.0)
        best_u = max((round(jaccard(u, v), 4) for u in unsafe), default=0.0)
        out["supported_vs_falsified"][k] = best_s
        out["unsafe_vs_falsified"][k] = best_u
        if best_s >= 0.25:
            hit = max(supported, key=lambda s: jaccard(s, v))
            out["endorses_a_falsified_claim"].append(
                {"falsified_claim": k, "description": v, "jaccard": best_s,
                 "endorsed_as": hit})
        if best_u >= 0.25:
            hit = max(unsafe, key=lambda s: jaccard(s, v))
            out["flags_a_falsified_claim"].append(
                {"falsified_claim": k, "description": v, "jaccard": best_u,
                 "flagged_as": hit})
    return out


def previous_proposal() -> dict:
    """The proposal the committed record made BEFORE this re-run.

    Read from git, so the comparison does not depend on a file that has
    already been overwritten.
    """
    path = "experiments/debrief_cf-openai-gpt-oss-120b.json"
    try:
        raw = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    d = json.loads(raw)
    ne = d.get("debrief", {}).get("next_experiment") or {}
    return {"source": f"git HEAD:{path}", "title": ne.get("title"),
            "next_experiment": ne, "debrief": d.get("debrief")}


def audit_proposal(debrief_obj: dict) -> dict:
    """Audit a proposed next experiment against the repository's standard.

    The same four questions the audit set asks of a claim, applied to the
    director's own proposal.
    """
    ne = (debrief_obj or {}).get("next_experiment") or {}
    title = str(ne.get("title", ""))
    hypothesis = str(ne.get("hypothesis", ""))
    method = str(ne.get("method", ""))
    controls = ne.get("controls") or []
    controls_txt = _as_text(controls)
    null = str(ne.get("null", ""))
    margin = str(ne.get("margin_threshold", ""))
    falsifier = str(ne.get("falsifier", ""))
    whole = " ".join([title, hypothesis, method, controls_txt, null, margin,
                      falsifier])

    # 1. A FALSIFIER: a result that would have made it come out the other
    #    way. The field has to be present and have content.
    has_falsifier = bool(falsifier.strip()) and len(falsifier.strip()) > 20
    # 2. A CONTROL THAT CAN COME OUT THE OTHER WAY. A control that is
    #    phrased as a certainty, or that has no value to be compared
    #    against, is not a control.
    tautology_markers = ("will be", "always", "cannot fail", "must be",
                          "is guaranteed", "by construction")
    controls_are_tautologies = bool(controls) and all(
        any(m in str(c).lower() for m in tautology_markers) for c in controls)
    has_failable_control = bool(controls) and not controls_are_tautologies
    # 3. A NULL.
    has_null = bool(null.strip()) and len(null.strip()) > 10
    # 4. A margin, or the claim is absolute.
    has_margin = bool(margin.strip())

    # DUPLICATION, against the repository's own next steps.
    dup = {}
    for k, v in REPO_NEXT_STEPS.items():
        dup[k] = round(jaccard(whole, v), 4)
    dup_max = max(dup.values()) if dup else 0.0
    dup_best = max(dup, key=dup.get) if dup else None

    # REVISITS A FALSIFIED CLAIM, against the repository's own list.
    revisits = {}
    for k, v in REPO_FALSIFIED.items():
        revisits[k] = round(jaccard(whole, v), 4)
    revisits_max = max(revisits.values()) if revisits else 0.0
    revisits_best = max(revisits, key=revisits.get) if revisits else None

    # 5. Can the proposal's OWN decision structure fail? A proposal with
    #    all five fields can still be unable to come out the other way.
    structure = decision_structure(hypothesis, null, margin, falsifier)

    # 6. Does it mention the repository's own top-ranked missing take?
    #    "Condition D -- turned AND silent" is what PROGRESS.md and
    #    NV12-STRIDE.md name as the single most valuable missing take.
    hay = whole.lower()
    mentions_condition_d = any(
        k in hay for k in ("condition d", "turned and silent",
                           "turned and not speaking", "pose fold"))

    return {
        "proposal": {"title": title, "hypothesis": hypothesis,
                     "method": method, "controls": controls, "null": null,
                     "margin_threshold": margin, "falsifier": falsifier},
        "has_falsifier": has_falsifier,
        "has_null": has_null,
        "has_margin": has_margin,
        "has_control_that_can_fail": has_failable_control,
        "controls_are_tautologies": controls_are_tautologies,
        "decision_structure": structure,
        "can_come_out_the_other_way": bool(
            has_falsifier and has_null and has_failable_control and
            not structure["null_triggers_the_falsifier"] and
            not structure["hypothesis_margin_contradiction"]),
        "mentions_condition_d": mentions_condition_d,
        "duplication_vs_repo_next_steps": dup,
        "max_overlap_with_a_repo_next_step": dup_max,
        "closest_repo_next_step": dup_best,
        "overlap_with_repo_falsified_claims": revisits,
        "max_overlap_with_a_falsified_claim": revisits_max,
        "closest_falsified_claim": revisits_best,
        "required_fields_present": {
            k: bool(ne.get(k)) for k in
            ("hypothesis", "method", "controls", "null", "falsifier")},
    }


# ------------------------------------------------------------- running them

def run_repeats(mode: str, n: int) -> list:
    """Run the audit N times per model, at a FIXED material budget.

    The runs are CACHED to the scratch directory as they complete, so an
    interrupted sweep does not throw away the calls it already paid for.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cache_path = SCRATCH / f"repeats_{mode}.json"
    done = json.loads(cache_path.read_text()) if cache_path.is_file() else []
    # A run that FAILED for an infrastructure reason is not a measurement,
    # so it is dropped from the cache and retried rather than carried.
    done = [r for r in done if "new_key" in r]
    seen = {(r.get("model"), r.get("run")) for r in done}
    out = list(done)
    for model in MODELS:
        # CONTEXT mode uses the FIT LADDER, not a fixed budget. The material
        # pool grew tonight, and a budget that fitted yesterday does not fit
        # today: llama's 60 000 characters is now 21 001 input tokens against
        # a 24 000-token window, and qwen's 140 000 is 38 417 tokens against
        # a 32 768-token window. The ladder finds the largest that fits and
        # records it, and it is deterministic for a given set of files, so
        # every repeat of one model still gets the SAME prompt.
        lim = None
        for i in range(n):
            if (model, i + 1) in seen:
                continue
            t0 = time.time()
            r = debrief.run_audit(model, mode, materials_limit=lim)
            rec = {"model": model, "mode": mode, "run": i + 1,
                   "elapsed_s": round(time.time() - t0, 2)}
            if r.get("error"):
                rec["error"] = r.get("error")
                rec["detail"] = str(r.get("detail"))[:300]
                out.append(rec)
                cache_path.write_text(json.dumps(out, indent=2))
                print(f"  [{mode}] {model} run {i+1}/{n}: ERROR {rec['error']}",
                      flush=True)
                continue
            rec["old_key"] = score_rows(r["rows"], "old")
            rec["new_key"] = score_rows(r["rows"], "new")
            rec["materials_chars"] = r.get("materials_chars")
            out.append(rec)
            cache_path.write_text(json.dumps(out, indent=2))
            print(f"  [{mode}] {model} run {i+1}/{n}: "
                  f"new headline {rec['new_key']['headline_sens_plus_spec_minus_1']} "
                  f"(old key {rec['old_key']['headline_sens_plus_spec_minus_1']}) "
                  f"{rec['elapsed_s']}s", flush=True)
    return out


def summarise_runs(runs: list) -> dict:
    """Median, min, max and spread of the headline, per model and key."""
    out = {}
    for model in MODELS:
        rs = [r for r in runs if r.get("model") == model
              and "new_key" in r and "old_key" in r]
        if not rs:
            continue
        for key_name in ("old", "new"):
            hs = [r[f"{key_name}_key"]["headline_sens_plus_spec_minus_1"]
                  for r in rs]
            sens = [r[f"{key_name}_key"]["sensitivity"] for r in rs]
            spec = [r[f"{key_name}_key"]["specificity"] for r in rs]
            out.setdefault(model, {})[key_name] = {
                "n_runs": len(hs),
                "headlines": hs,
                "n_distinct_headlines": len(set(hs)),
                "headline_median": statistics.median(hs),
                "headline_min": min(hs),
                "headline_max": max(hs),
                "headline_spread": round(max(hs) - min(hs), 4),
                "sensitivity_median": statistics.median(sens),
                "specificity_median": statistics.median(spec),
            }
    return out


def error_pattern(runs: list, reachability: dict | None = None) -> dict:
    """Which claims the director misses, and what class they are in.

    The interesting question is not only HOW MANY the director misses but
    WHICH. A director whose false accusations cluster on one class of claim
    is telling us something about the instrument.
    """
    classes = {
        "Reducing the gate's field threshold to 0.30": "interval_mismatch",
        "The ASCII character map reliably detects": "unrun_control",
        "The watcher's memory use grows quadratically": "harness_artefact",
        "Applying a per-frame gain normalisation": "stride_artefact",
        "A local large language model given an ASCII rendering": "no_margin_over_blank",
        "31.8% of the frames the watcher processes": "never_measured",
        "Removing each row's median across the columns": "stride_artefact",
        "A conventional teacher model trained on motion maps": "control_collapse",
        "The room's own acoustic noise floor measured -73.5 dBFS": "acoustic_measurement",
        "Composing row-wise band removal with a temporal median": "stride_artefact",
    }
    reach = (reachability or {}).get("claims") or {}
    per_claim = {}
    for r in runs:
        if "new_key" not in r or "old_key" not in r:
            continue
        for row in r["new_key"]["rows"]:
            k = _key_for(row["claim"])
            c = per_claim.setdefault(k, {
                "key": k, "class": classes.get(k, "?"),
                "is_false_claim": row["is_false_claim"],
                "claim": row["claim"], "verdicts": [], "correct": []})
            c["verdicts"].append(row["verdict"])
            c["correct"].append(row["correct"])
    for k, c in per_claim.items():
        c["n_runs"] = len(c["verdicts"])
        c["distinct_verdicts"] = sorted(set(c["verdicts"]))
        c["n_correct"] = sum(c["correct"])
        c["accuracy"] = round(c["n_correct"] / c["n_runs"], 4)
        c["stable"] = len(c["distinct_verdicts"]) == 1
        c["verdict_counts"] = {v: c["verdicts"].count(v)
                               for v in c["distinct_verdicts"]}
        if k in reach:
            c["evidence_reachable_in_the_prompt"] = reach[k]["all_present"]
            c["probes_present"] = reach[k]["probes_present"]
    by_class = {}
    for c in per_claim.values():
        b = by_class.setdefault(c["class"], {"n_claims": 0, "n_correct": 0,
                                             "n_runs": 0})
        b["n_claims"] += 1
        b["n_correct"] += c["n_correct"]
        b["n_runs"] += c["n_runs"]
    for b in by_class.values():
        b["accuracy"] = round(b["n_correct"] / b["n_runs"], 4) if b["n_runs"] else None
    # The join that matters: is a claim the director gets wrong one whose
    # evidence did not reach the prompt?
    reach_yes, reach_no = [], []
    for c in per_claim.values():
        if "evidence_reachable_in_the_prompt" not in c:
            continue
        (reach_yes if c["evidence_reachable_in_the_prompt"] else reach_no).append(
            {"key": c["key"], "accuracy": c["accuracy"],
             "n_correct": c["n_correct"], "n_runs": c["n_runs"]})
    def _acc(lst):
        n = sum(x["n_runs"] for x in lst)
        k = sum(x["n_correct"] for x in lst)
        return {"n_claims": len(lst), "n_runs": n, "n_correct": k,
                "accuracy": round(k / n, 4) if n else None}
    return {"per_claim": per_claim, "by_class": by_class,
            "accuracy_when_the_evidence_reached_the_prompt": _acc(reach_yes),
            "accuracy_when_the_evidence_did_not_reach_the_prompt": _acc(reach_no),
            "claims_whose_evidence_did_not_reach_the_prompt": reach_no,
            "claims_whose_evidence_reached_the_prompt": reach_yes}


def materials_sweep(model: str = SWEEP_MODEL, limits=SWEEP_LIMITS,
                    n: int = N_SWEEP) -> dict:
    """Does MORE evidence make the director better? Model held constant."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cache_path = SCRATCH / "sweep.json"
    cached = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    out = {"model": model, "n_runs_per_budget": n, "budgets": {}}
    for lim in limits:
        runs = []
        errors = []
        key = f"{model}@{lim}"
        if key in cached:
            entry = cached[key]
            if isinstance(entry, dict):
                runs, errors = entry.get("runs", []), entry.get("errors", [])
            else:                       # the first version cached a bare list
                runs = entry
        else:
            for i in range(n):
                r = debrief.run_audit(model, "context", materials_limit=lim)
                if r.get("error"):
                    errors.append({"run": i + 1, "error": r.get("error"),
                                   "detail": str(r.get("detail"))[:300]})
                    print(f"  [sweep] {model} limit {lim} run {i+1}/{n}: "
                          f"ERROR {r.get('error')}", flush=True)
                    continue
                runs.append({"new_key": score_rows(r["rows"], "new"),
                             "old_key": score_rows(r["rows"], "old"),
                             "materials_chars": r.get("materials_chars")})
                print(f"  [sweep] {model} limit {lim} run {i+1}/{n}: "
                      f"{runs[-1]['new_key']['headline_sens_plus_spec_minus_1']}",
                      flush=True)
            cached[key] = {"runs": runs, "errors": errors}
            cache_path.write_text(json.dumps(cached, indent=2))
        if not runs:
            continue
        hs = [x["new_key"]["headline_sens_plus_spec_minus_1"] for x in runs]
        old_hs = [x["old_key"]["headline_sens_plus_spec_minus_1"]
                  for x in runs if x.get("old_key")]
        out["budgets"][str(lim)] = {
            "n_runs": len(hs), "headlines": hs,
            "headline_median": statistics.median(hs),
            "headline_min": min(hs), "headline_max": max(hs),
            "headline_spread": round(max(hs) - min(hs), 4),
            "n_failed_runs": len(errors),
            "failed_runs": errors,
            "old_key_headlines": old_hs,
            "old_key_headline_median": (statistics.median(old_hs)
                                        if old_hs else None),
            "materials_chars_actual": runs[0]["materials_chars"],
            "rows": [x["new_key"].get("rows", []) for x in runs],
        }
    # RULE R6: the effect is called only if the larger budget beats the
    # smaller by more than the spread observed AT THE SMALLER BUDGET.
    small = str(min(limits))
    large = str(max(limits))
    if small in out["budgets"] and large in out["budgets"]:
        s = out["budgets"][small]
        l = out["budgets"][large]
        delta = l["headline_median"] - s["headline_median"]
        out["decision"] = {
            "null": "more evidence does not make the director better",
            "small_budget": int(small), "large_budget": int(large),
            "small_median": s["headline_median"],
            "large_median": l["headline_median"],
            "delta_median": round(delta, 4),
            "spread_at_small_budget": s["headline_spread"],
            "rule": PRE_REGISTERED_RULES["R6_materials_sweep"],
            "verdict": ("EFFECT RESOLVED" if delta > s["headline_spread"]
                        else "NOT RESOLVED"),
        }
    return out


# The numbers each claim's truth value turns on, and whether the CONTEXT
# prompt actually carries them.
#
# The director's reason strings are often CORRECT given what it was shown
# ("No cited evidence provides the -73.5 dBFS floor"), so a wrong verdict
# is not automatically the model's mistake. This table lets the two be
# separated: if the number is not in the prompt, the verdict is the
# harness's fault, and the model is reading correctly.
CLAIM_PROBES = {
    "Reducing the gate's field threshold to 0.30": ["14.5", "GATE_FIELD"],
    "The ASCII character map reliably detects": ["48.19", "50.00"],
    "The watcher's memory use grows quadratically": ["quadratic", "39.13"],
    "Applying a per-frame gain normalisation": ["43.3", "0.81"],
    "A local large language model given an ASCII rendering": ["Qwen2-VL", "blank"],
    "31.8% of the frames the watcher processes": ["31.8"],
    "Removing each row's median across the columns": ["0.01", "1.79"],
    "A conventional teacher model trained on motion maps": ["1.000", "0.375"],
    "The room's own acoustic noise floor measured -73.5 dBFS": ["73.496", "119 transients"],
    "Composing row-wise band removal with a temporal median": ["0.0014", "1.83", "21"],
}

# The retraction text, searched for separately: a note can carry the OLD
# number and drop the notice that withdraws it, and that is the case that
# produced the director's one false endorsement tonight.
RETRACTION_PROBES = ("WITHDRAWN", "premise is withdrawn", "PREMISE IS WITHDRAWN")


def collect_materials_with(limit: int, notes: list, res: list) -> str:
    """`debrief.collect_materials`, with the note and result lists supplied.

    A faithful copy, so that the material pool can be measured against the
    state it was in when the runs were made. The copy is VERIFIED against
    `debrief.collect_materials` in
    `tests/test_debrief_reaudit.py::test_the_material_replication_is_faithful`,
    which is what makes the excluded version worth anything: if the copy
    were not faithful, the "this note was in the pool" claims would be
    worthless.

    > CORRECTION (2026-10-20, overnight). This copy carried the same
    > front-truncation the harness had -- `text[:per]` -- and the
    > faithfulness test caught the divergence the moment the harness was
    > fixed. That is the test doing its job. The copy now mirrors the
    > harness: the budget is spent by dropping lines from the MIDDLE of the
    > keep-set, and if even the head-and-tail slice is too big then each
    > end is cut to its own half so that BOTH survive. The numbers this
    > lane recorded were measured against the pre-fix pool and stay in
    > `debrief_set_reaudit_results.json` as the record of that run; what
    > the fix is worth is in `docs/research/DEBRIEF-MATERIAL-POOL.md`.
    """
    parts: list[str] = []
    note_pool = int(limit * 0.72)
    per = max(600, note_pool // max(1, len(notes)))
    for p in notes:
        try:
            body = p.read_text()
        except Exception:  # noqa: BLE001
            continue
        lines = body.splitlines()
        if len(body) <= per:
            parts.append(f"--- NOTE: {p.name} ---\n{body}")
            continue
        head_lines = max(3, int(len(lines) * 0.18))
        keep = set(range(head_lines))
        keep |= set(range(len(lines) - head_lines, len(lines)))
        for i, ln in enumerate(lines):
            if any(ch.isdigit() for ch in ln):
                keep.add(i)
        keep = sorted(keep)

        def assemble(ks: list[int]) -> str:
            out, prev = [], -2
            for i in ks:
                if i != prev + 1:
                    out.append(f"[... {i - prev - 1} lines elided ...]")
                out.append(lines[i])
                prev = i
            return "\n".join(out)

        n_head = max(3, len(keep) // 4)
        n_tail = max(3, len(keep) // 4)
        text = assemble(keep)
        if len(text) > per:
            # The budget is spent by dropping lines from the MIDDLE of the
            # keep-set. The head and the tail are what carry the retraction
            # notices, and the old front-truncation threw the tail away.
            ks = list(keep)
            while len(ks) > n_head + n_tail and len(assemble(ks)) > per:
                del ks[len(ks) // 2]
            keep = ks
            text = assemble(keep)
        if len(text) > per:
            # Even the head-and-tail slice is too big. Cut each end to its
            # own half of the budget so that BOTH survive.
            head_text = "\n".join(lines[i] for i in keep[:n_head])
            tail_text = "\n".join(lines[i] for i in keep[-n_tail:])
            half = per // 2
            text = (head_text[:half] + "\n[... truncated ...]\n"
                    + "[... middle elided ...]\n" + tail_text[-half:])
        parts.append(f"--- NOTE: {p.name} ---\n{text}")
    res_pool = int(limit * 0.24)
    per_res = max(300, res_pool // max(1, len(res)))
    for p in res:
        try:
            body = p.read_text()
        except Exception:  # noqa: BLE001
            continue
        parts.append(f"--- RESULT: {p.name} ---\n{body[:per_res]}")
    return "\n\n".join(parts)


def evidence_reachability(mode: str = "context", limit: int = 140000,
                          exclude: tuple = ("DEBRIEF-REAUDIT.md",)) -> dict:
    """Does the CONTEXT prompt actually carry each claim's evidence?

    `collect_materials` keeps each note's head, its tail and every line
    containing a digit. This measures what survives.

    > CORRECTION (2026-10-20, overnight). This docstring used to say: "The
    > cap is applied AFTER the keep-set is built, so a note whose kept text
    > is longer than the cap loses its tail, and the tail is where the
    > retraction notices live." That was a correct description of a DEFECT,
    > and the defect is now fixed: the cap is spent on the middle of the
    > keep-set, and both ends survive. So this function now measures the
    > FIXED pool, and the numbers it reports are higher than the ones
    > recorded in `debrief_set_reaudit_results.json` -- which were
    > measured against the pre-fix pool and stay there as the record of
    > that run. See `docs/research/DEBRIEF-MATERIAL-POOL.md`.

    The search is over the MATERIALS ONLY. The claims themselves are
    prepended to the prompt, and they carry the numbers -- so a naive
    search of the whole prompt finds every number in every claim and
    reports that the evidence is present. It is not. (That was the first
    version of this check, and it was wrong in the flattering direction.)

    THIS DOCUMENT IS EXCLUDED from the pool, and it has to be: it quotes
    the deciding numbers in its own tables, so a pool that includes it
    would report the evidence as present when the director never saw it.
    The pool measured here is the pool the runs were made against.
    """
    notes = [p for p in sorted((ROOT / "docs" / "research").glob("*.md"))
             if p.name not in exclude]
    res = sorted((ROOT / "experiments").glob("*.json"))
    mats = collect_materials_with(limit, notes, res)
    out = {"mode": mode, "materials_limit": limit,
           "materials_chars": len(mats),
           "excluded_from_the_pool": list(exclude),
           "note": ("the search is over the MATERIALS only; the claims "
                    "themselves carry the numbers and would make every "
                    "probe look present"),
           "claims": {}}
    for k, probes in CLAIM_PROBES.items():
        present = {p: (p in mats) for p in probes}
        out["claims"][k] = {
            "probes_present": present,
            "n_present": sum(present.values()),
            "n_probes": len(probes),
            "all_present": all(present.values()),
        }
    out["retraction_text_present"] = {
        p: (p in mats) for p in RETRACTION_PROBES}


    # The note-level diagnostic: how much of each note survives the cap.
    note_pool = int(limit * 0.72)
    per = max(600, note_pool // max(1, len(notes)))
    rows = []
    for p in notes:
        try:
            body = p.read_text()
        except Exception:  # noqa: BLE001
            continue
        lines = body.splitlines()
        head_lines = max(3, int(len(lines) * 0.18))
        keep = set(range(head_lines))
        keep |= set(range(len(lines) - head_lines, len(lines)))
        for i, ln in enumerate(lines):
            if any(ch.isdigit() for ch in ln):
                keep.add(i)
        keep = sorted(keep)
        out_lines, prev = [], -2
        for i in keep:
            if i != prev + 1:
                out_lines.append(f"[... {i - prev - 1} lines elided ...]")
            out_lines.append(lines[i])
            prev = i
        text = "\n".join(out_lines)
        final = text[:per] + "\n[... truncated ...]" if len(text) > per else text
        rows.append({
            "note": p.name, "bytes": len(body), "n_lines": len(lines),
            "n_lines_kept": len(keep), "kept_chars": len(text),
            "per_note_cap": per, "truncated": len(text) > per,
            "frac_of_kept_text_surviving": round(len(final) / max(1, len(text)), 4),
            "tail_survives": bool([l for l in lines[-3:] if l.strip() and l[:30] in final]),
        })
    out["per_note"] = rows
    out["n_notes"] = len(notes)
    out["per_note_cap"] = per
    out["n_notes_truncated"] = sum(1 for r in rows if r["truncated"])
    out["n_notes_whose_tail_survives"] = sum(1 for r in rows if r["tail_survives"])
    out["note"] = ("CORRECTION (2026-10-20, overnight). This note used to "
                   "say: 'The per-note cap is applied AFTER the head/tail/"
                   "digit-line keep-set is built, so a note longer than the "
                   "cap loses its tail -- which is where this repository "
                   "puts its retraction notices.' That described a DEFECT, "
                   "and the defect is fixed: the cap is now spent on the "
                   "MIDDLE of the keep-set and both ends survive. So the "
                   "numbers below are measured against the FIXED pool, and "
                   "the ones recorded in the previous version of this file "
                   "were measured against the pre-fix pool. What the fix is "
                   "worth is in docs/research/DEBRIEF-MATERIAL-POOL.md: the "
                   "acoustics figure now reaches the prompt, and the "
                   "director accuses the one TRUE claim anyway.")
    return out


def key_leak_check() -> dict:
    """Does the CONTEXT material pool contain the answer key?

    `collect_materials` reads `experiments/*.json`, and the audit artefacts
    live there. If the key leaked into the prompt, the context score would
    be contaminated. This checks the slice that is actually sent.
    """
    pairs = debrief.audit_pairs()
    lim = max(CONTEXT_LIMIT.values())
    res = sorted(RESULTS.glob("*.json"))
    res_pool = int(lim * 0.24)
    per = max(300, res_pool // max(1, len(res)))
    leaks = []
    for p in res:
        body = p.read_text()
        if "is_false_claim" in body[:per]:
            leaks.append(p.name)
    # And the aggregate scores of previous runs, which do leak.
    agg = []
    for p in artefact_paths():
        body = p.read_text()
        head = body[:per]
        if "sensitivity_rejected_false_claims" in head:
            agg.append(p.name)
    return {"materials_limit_used_for_the_check": lim,
            "per_result_file_chars": per,
            "n_result_files": len(res),
            "files_whose_slice_contains_the_answer_key": leaks,
            "files_whose_slice_contains_a_previous_run_aggregate": agg,
            "note": ("The per-claim key does not leak: the slice each result "
                     "file gets is the first few hundred characters, and the "
                     "per-claim labels sit far deeper. The previous run's "
                     "AGGREGATE scores do leak, and from them the class "
                     "balance is recoverable (0.2857 = 2/7). That is a real "
                     "property of the instrument and it changes when the "
                     "key changes: 7:3 before, 9:1 after. It is reported, "
                     "not fixed, because fixing it means changing the "
                     "instrument.")}


# ------------------------------------------------------------------- the main

def git_state() -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True).stdout.splitlines()
        return {"head": head, "n_uncommitted_paths": len(dirty),
                "uncommitted": dirty[:40]}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def build_provenance(cli_args: list) -> dict:
    inputs = {}
    for p in sorted(RESULTS.glob("*_results.json")):
        if p.name == OUT_PATH.name:
            continue          # a file cannot carry its own sha256
        inputs[p.name] = sha256_file(p)
    for p in sorted((ROOT / "docs" / "research").glob("*.md")):
        inputs["docs/research/" + p.name] = sha256_file(p)
    return {
        "generated_by": "experiments/debrief_set_reaudit.py",
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "debrief_py_sha256": sha256_file(RESULTS / "debrief.py"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command_line": " ".join(cli_args),
        "cli_args": cli_args,
        "input_sha256": inputs,
        "git": git_state(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["before", "reaudit", "main"],
                    default="main")
    ap.add_argument("--blind-runs", type=int, default=N_BLIND)
    ap.add_argument("--context-runs", type=int, default=N_CONTEXT)
    ap.add_argument("--sweep-runs", type=int, default=N_SWEEP)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-repeats", action="store_true")
    ap.add_argument("--out", default=str(OUT_PATH))
    a = ap.parse_args(argv)
    cli = [sys.executable, str(Path(__file__).resolve())] + list(argv or sys.argv[1:])

    if a.phase == "before":
        snap = capture_before()
        print(f"captured {len(snap['artefacts'])} artefacts -> {BEFORE_PATH}")
        for art in snap["artefacts"]:
            for r in art["runs"]:
                if "old_key" in r:
                    print(f"  {art['file']} {r['model']}: "
                          f"old headline {r['old_key']['headline_sens_plus_spec_minus_1']}")
        return

    reaudit = reaudit_set()

    if a.phase == "reaudit":
        out = {"pre_registered_rules": PRE_REGISTERED_RULES,
               "the_reaudit": reaudit}
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(json.dumps(reaudit["counts"], indent=2))
        for c in reaudit["claims"]:
            print(f"  {c['old_label']:5s} -> {c['new_label']:5s} "
                  f"{'CHANGED' if c['label_changed'] else '       '} "
                  f"{c['claim'][:64]}")
        return

    # -- phase main --------------------------------------------------------
    before = {}
    if BEFORE_PATH.is_file():
        before = json.loads(BEFORE_PATH.read_text())

    print(f"repeats: blind x{a.blind_runs}, context x{a.context_runs} "
          f"per model")
    blind_runs = [] if a.skip_repeats else run_repeats("blind", a.blind_runs)
    context_runs = [] if a.skip_repeats else run_repeats("context", a.context_runs)

    sweep = {}
    if not a.skip_sweep:
        print("materials sweep: model held constant, budget varied")
        sweep = materials_sweep(n=a.sweep_runs)

    # The committed artefacts, as they stand NOW (after the CLI re-runs).
    committed = [score_artefact(p) for p in artefact_paths()]

    # Does the CONTEXT prompt carry each claim's evidence? Measured once
    # and joined into the error pattern.
    reach = evidence_reachability()

    # The proposal: the committed production debrief, plus the repeats.
    proposal = {}
    deb_path = RESULTS / "debrief_cf-openai-gpt-oss-120b.json"
    if deb_path.is_file():
        d = json.loads(deb_path.read_text())
        proposal["committed_debrief"] = {
            "file": deb_path.name,
            "model": d.get("model"),
            "materials_chars": d.get("materials_chars"),
            "acceptable": d.get("acceptable"),
            "missing_required_fields": d.get("missing_required_fields"),
            "audit": audit_proposal(d.get("debrief") or {}),
            "evidence_review": audit_evidence_review(d.get("debrief") or {}),
        }

    # THE ECHO QUESTION: is the proposal the same experiment as the one the
    # committed record made before tonight, and the same as the repository's
    # own next step?
    prev = previous_proposal()
    if deb_path.is_file() and "error" not in prev:
        d = json.loads(deb_path.read_text())
        ne_new = (d.get("debrief") or {}).get("next_experiment") or {}
        ne_old = prev.get("next_experiment") or {}
        t_new = _as_text(ne_new)
        t_old = _as_text(ne_old)
        proposal["echo_check"] = {
            "question": ("does the proposal add information beyond the "
                         "repository's own notes, or does it echo them?"),
            "previous_title": prev.get("title"),
            "current_title": ne_new.get("title"),
            "jaccard_whole_proposal_vs_the_previous_committed_proposal":
                round(jaccard(t_new, t_old), 4),
            "jaccard_vs_repo_next_steps": {
                k: round(jaccard(t_new, v), 4)
                for k, v in REPO_NEXT_STEPS.items()},
            "max_overlap_with_a_repo_next_step":
                max(round(jaccard(t_new, v), 4) for v in REPO_NEXT_STEPS.values()),
            "jaccard_vs_repo_falsified_claims": {
                k: round(jaccard(t_new, v), 4)
                for k, v in REPO_FALSIFIED.items()},
            "max_overlap_with_a_falsified_claim":
                max(round(jaccard(t_new, v), 4) for v in REPO_FALSIFIED.values()),
            "the_repo_s_own_ranking": {
                "top_missing_take": "Condition D -- turned AND silent "
                                    "(PROGRESS.md, POSE-FEATURE.md, "
                                    "NV12-STRIDE.md section 8)",
                "plan_item_11": "a held-out-device measurement, proxied by "
                                "the gate's advantage under cache pressure",
                "the_proposal_names_the_top_missing_take":
                    audit_proposal(d.get("debrief") or {})["mentions_condition_d"],
                "the_proposal_names_plan_item_11": "cache" in t_new.lower(),
            },
        }
        # The unsafe-claim sets, old and new, against the repository's own
        # retractions: the director's REVIEW of the record.
        pu = [str(c.get("claim", "")) for c in
              (prev.get("debrief") or {}).get("unsafe_claims") or []]
        nu = [str(c.get("claim", "")) for c in
              (d.get("debrief") or {}).get("unsafe_claims") or []]
        vals = [jaccard(a, b) for a in pu for b in nu]
        proposal["echo_check"]["unsafe_claim_set_overlap"] = {
            "median": round(statistics.median(vals), 4) if vals else None,
            "min": round(min(vals), 4) if vals else None,
            "max": round(max(vals), 4) if vals else None,
            "n_pairs": len(vals),
        }
        proposal["echo_check"]["previous_unsafe_claims"] = pu
        proposal["echo_check"]["current_unsafe_claims"] = nu

    out = {
        "experiment": "debrief_set_reaudit",
        "date": time.strftime("%Y-%m-%d"),
        "question": ("the audit set's ground truth was written against torn "
                     "frames. What is the truth value of each of its ten "
                     "claims now, what does the corrected key do to the "
                     "director's score, and does the director's proposed "
                     "next experiment add information beyond this "
                     "repository's own notes?"),
        "pre_registered_rules": PRE_REGISTERED_RULES,
        "the_reaudit": reaudit,
        "the_key_change": {
            "how_it_was_measured": (
                "The SAME verdict rows, scored under the old key and under "
                "the new key. No model is involved, so the difference is the "
                "key and nothing else."),
            "committed_artefacts_now": committed,
            "the_before_snapshot": before,
        },
        "the_audit": {
            "blind_runs": blind_runs,
            "context_runs": context_runs,
            "summary": {"blind": summarise_runs(blind_runs),
                        "context": summarise_runs(context_runs)},
            "error_pattern": {"blind": error_pattern(blind_runs, reach),
                              "context": error_pattern(context_runs, reach)},
        },
        "materials_sweep": sweep,
        "proposal_audit": proposal,
        "key_leak_check": key_leak_check(),
        "evidence_reachability": reach,
        "provenance": build_provenance(cli),
        "supersedes": {
            "what": ("The audit set's labels as they stood before "
                     "2026-10-20, and every audit score computed against "
                     "them."),
            "superseded_by": "experiments/debrief_set_reaudit_results.json",
            "why": ("Every camera reader in the repository was reading the "
                    "captures at 23,040 bytes per frame when the files carry "
                    "19,200, so 80% of every frame was a tear between two "
                    "real moments. Two of the three standing claims in the "
                    "audit set were standing on numbers computed on those "
                    "tears, and the answer key changed under the quiz. The "
                    "old scores are kept in `the_before_snapshot` rather "
                    "than overwritten, because a score against a key that no "
                    "longer exists is not comparable to anything."),
            "the_old_key": "7 false claims and 3 true claims",
            "the_new_key": (f"{reaudit['counts']['after']['n_false']} false "
                            f"claims and {reaudit['counts']['after']['n_true']} "
                            "true claims"),
        },
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")
    print(json.dumps(reaudit["counts"], indent=2))
    for m, d in out["the_audit"]["summary"]["blind"].items():
        print(f"  blind {m}: new median {d['new']['headline_median']} "
              f"(spread {d['new']['headline_spread']})  "
              f"old-key median {d['old']['headline_median']}")
    for m, d in out["the_audit"]["summary"]["context"].items():
        print(f"  context {m}: new median {d['new']['headline_median']} "
              f"(spread {d['new']['headline_spread']})  "
              f"old-key median {d['old']['headline_median']}")


if __name__ == "__main__":
    main()
