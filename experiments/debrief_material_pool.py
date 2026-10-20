#!/usr/bin/env python3
"""
debrief_material_pool.py -- the evidence pool was dropping the retractions.

WHAT THIS IS ABOUT.

`experiments/debrief.py` audits an LLM "director" against ten claims drawn
from this project's own retracted history. Its material pool is supposed to
give every note an equal share of the character budget and to keep each
note's HEAD and its TAIL, because the dated correction blocks -- the ones
that say which of this repository's numbers were withdrawn -- live at both
ends of these notes.

It did not do that. The keep-set was built correctly and the assembled text
was then cut with `text[:per]`, from the FRONT. Every note whose slice
exceeded its share lost its TAIL.

    notes keeping their tail, at the committed 140,000-character budget
        BEFORE the fix:   2 of 50
        AFTER  the fix:  50 of 50   (and 50 of 50 keep their head too)

This file MEASURES that, and it records the A/B of the director's score
with the two pools, because the measurement of the defect is not the same
as the measurement of its effect.

AND THE EFFECT IS NOT WHAT ANYONE WOULD GUESS.

The honest summary of the A/B, run against ONE corpus with both pools, 3
repeats per model per pool:

  * **Specificity is 0.000 in 15 of 15 successful runs, in BOTH pools, at
    every budget.** The director accuses the one TRUE claim every single
    time. That is the failure, and it is not a budget problem.
  * **The pool effect cannot be separated from the endpoint's run-to-run
    variance.** gpt-oss-120b's sensitivity on the OLD pool was 0.889,
    0.333 and 1.000 on three IDENTICAL prompts -- a spread of 0.667 in
    one cell.
  * **The fix changes HOW MUCH EVIDENCE A MODEL GETS, and that is the
    finding with a consequence.** The tails are dense tables of numbers and
    they tokenise more densely than the head slices, so the same character
    budget costs more tokens and the fit ladder shrinks further:
    llama-3.3-70b went from 140,000 characters of evidence to 40,000, and
    qwen3-30b from 140,000 to 60,000. **Any comparison of "the same
    evidence" has to be made on TOKENS, not characters.**

WHAT THIS FILE IS NOT.

It is not a claim that the fix makes the director better. It does not, on
this evidence, and the file says so.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import debrief  # noqa: E402

RESULTS = ROOT / "experiments" / "debrief_material_pool_results.json"
SCRATCH = ROOT / ".scratch" / "debrief_reaudit"
AB_PATH = SCRATCH / "ab_pools_samecorpus.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def old_pool_tail_survival(limit: int = 140000) -> dict:
    """Reproduce the OLD assembly exactly and count the tails it kept.

    The old behaviour is reproduced here rather than quoted, so the "2 of
    50" is a measurement this file makes, not a number it inherited.
    """
    notes = sorted((ROOT / "docs" / "research").glob("*.md"))
    per = max(600, int(limit * 0.72) // max(1, len(notes)))
    kept_tail = kept_head = truncated = total = 0
    for p in notes:
        body = p.read_text()
        total += 1
        if len(body) <= per:
            kept_tail += 1
            kept_head += 1
            continue
        lines = body.splitlines()
        truncated += 1
        head_lines = max(3, int(len(lines) * 0.18))
        keep = set(range(head_lines))
        keep |= set(range(len(lines) - head_lines, len(lines)))
        for i, ln in enumerate(lines):
            if any(ch.isdigit() for ch in ln):
                keep.add(i)
        keep = sorted(keep)
        out, prev = [], -2
        for i in keep:
            if i != prev + 1:
                out.append(f"[... {i - prev - 1} lines elided ...]")
            out.append(lines[i])
            prev = i
        text = "\n".join(out)
        if len(text) > per:
            text = text[:per] + "\n[... truncated ...]"      # THE OLD BUG
        first_kept = next((lines[i] for i in keep if lines[i].strip()), "")
        if first_kept and first_kept in text:
            kept_head += 1
        if text.rstrip().endswith(lines[keep[-1]]):
            kept_tail += 1
    return {"notes_total": total, "notes_truncated": truncated,
            "notes_keeping_head": kept_head, "notes_keeping_tail": kept_tail}


def summarise_ab(path: Path = AB_PATH) -> dict:
    """The A/B, summarised from the raw runs."""
    if not path.is_file():
        return {"status": "no A/B run file at " + str(path)}
    runs = json.loads(path.read_text())
    out: dict = {"source": str(path), "pools": {}}
    for pool in ("old_pool", "new_pool"):
        ok = [r for r in runs if r.get("pool") == pool and "new_key" in r]
        err = [r for r in runs if r.get("pool") == pool and "new_key" not in r]
        if not ok:
            out["pools"][pool] = {"runs_ok": 0, "runs_unparseable": len(err)}
            continue
        hs = [r["new_key"]["headline_sens_plus_spec_minus_1"] for r in ok]
        sens = [r["new_key"]["sensitivity"] for r in ok]
        spec = [r["new_key"]["specificity"] for r in ok]
        by_model: dict = {}
        for r in ok:
            m = by_model.setdefault(r["model"], {"chars": set(), "headlines": []})
            m["chars"].add(r.get("materials_chars"))
            m["headlines"].append(
                round(r["new_key"]["headline_sens_plus_spec_minus_1"], 6))
        out["pools"][pool] = {
            "runs_ok": len(ok),
            "runs_unparseable": len(err),
            "sensitivity_values": [round(s, 6) for s in sens],
            "sensitivity_spread": round(max(sens) - min(sens), 6),
            "specificity_values": [round(s, 6) for s in spec],
            "specificity_is_zero_in_every_run": all(s == 0.0 for s in spec),
            "headlines": [round(h, 6) for h in hs],
            "headline_median": round(statistics.median(hs), 6),
            "headline_spread": round(max(hs) - min(hs), 6),
            "by_model": {k: {"materials_chars": sorted(v["chars"]),
                             "headlines": v["headlines"]}
                         for k, v in by_model.items()},
        }
    return out


def main() -> int:
    before = old_pool_tail_survival()
    after = debrief.collect_materials_stats(140000)
    ab = summarise_ab()

    spec_zero_both = all(
        p.get("specificity_is_zero_in_every_run")
        for p in ab.get("pools", {}).values() if p.get("runs_ok"))
    all_runs = sum(p.get("runs_ok", 0) for p in ab.get("pools", {}).values())

    res = {
        "date": "2026-10-20",
        "question": (
            "The material pool's head-and-tail design was defeated by a "
            "front-truncation, so the dated correction blocks -- the ones "
            "that say which of this repository's numbers were withdrawn -- "
            "never reached the prompt. Does fixing it make the director "
            "better?"),
        "the_defect": {
            "what": (
                "collect_materials built a keep-set holding each note's "
                "head, its tail and every digit-line, then cut the "
                "assembled text from the FRONT with text[:per]. Every note "
                "over its share lost its tail, and the retraction notices "
                "live in the tails."),
            "measured_before": before,
            "measured_after": after,
            "the_change": {
                "notes_keeping_tail_before": before["notes_keeping_tail"],
                "notes_keeping_tail_after": after["notes_keeping_tail"],
                "notes_total": after["notes_total"],
            },
        },
        "the_ab": ab,
        "the_finding": {
            "specificity_is_zero_in_every_successful_run": spec_zero_both,
            "successful_runs_in_the_ab": all_runs,
            "the_robust_result": (
                "The director accuses the one TRUE claim in every "
                "successful run, in both pools, at every budget. That is "
                "the failure and it is not a starvation problem."),
            "the_pool_effect_is_not_separable": (
                "gpt-oss-120b's sensitivity on the old pool was 0.889, "
                "0.333 and 1.000 on three IDENTICAL prompts, a spread of "
                "0.667 in one cell, so three repeats cannot separate a pool "
                "effect from the endpoint's variance."),
            "the_fix_changes_how_much_evidence_a_model_gets": (
                "The tails are dense tables of numbers and tokenise more "
                "densely than the head slices, so the same character "
                "budget costs more tokens and the fit ladder shrinks "
                "further: llama-3.3-70b went from 140,000 characters to "
                "40,000 and qwen3-30b from 140,000 to 60,000. A comparison "
                "of 'the same evidence' has to be made on TOKENS."),
        },
        "supersedes": {
            "what": (
                "The implicit claim, carried by the 2026-10-19 note in "
                "LLM-DIRECTOR.md, that the director's quality is a function "
                "of the evidence you hand it and that delivering the "
                "retraction notices would therefore help."),
            "status": (
                "PARTLY WITHDRAWN. The budget sweep still shows a strong "
                "effect of the MATERIALS BUDGET. The pool's internal "
                "COMPLETENESS is a different axis, and on this evidence "
                "fixing it does not help. The note's mechanism -- 'a "
                "starved evidence base produces a confident, specific, "
                "WRONG verdict' -- survives, and is now the best-supported "
                "thing in the instrument: the director accuses the true "
                "claim in 15 of 15 successful runs."),
        },
        "provenance": {
            "generated_by": "experiments/debrief_material_pool.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "debrief_py_sha256": sha256_file(ROOT / "experiments" / "debrief.py"),
            "command_line": "experiments/debrief_material_pool.py",
            "cli_args": [],
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "notes_read": sorted(p.name for p in
                                 (ROOT / "docs" / "research").glob("*.md")),
        },
    }
    RESULTS.write_text(json.dumps(res, indent=2))
    print(json.dumps({
        "notes_keeping_tail": {
            "before": before["notes_keeping_tail"],
            "after": after["notes_keeping_tail"],
            "of": after["notes_total"]},
        "specificity_zero_in_every_run": spec_zero_both,
        "successful_runs": all_runs,
        "wrote": str(RESULTS),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
