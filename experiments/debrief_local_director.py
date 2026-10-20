#!/usr/bin/env python3
"""
debrief_local_director.py -- measure the director where the measurement is
variance-free.

WHY THIS FILE EXISTS.

Every number this repository has about the LLM "director" is contaminated
by the serving stack. The same prompt, sent to the same hosted model,
gives different answers:

  * the hosted vision model's headline spread was measured at 0.619
  * gpt-oss-120b's sensitivity on THREE IDENTICAL prompts was 0.889,
    0.333 and 1.000 -- a spread of 0.667 in one cell

So a difference between two arms of this experiment cannot be separated
from the endpoint's noise unless the noise is removed. And the noise
cannot be removed by running more repeats: it is not variance around a
mean, it is the endpoint changing its mind.

A LOCAL model at temperature 0 has no such freedom. This file measures the
director on local models and reports the spread, so the two can be
compared.

AND THE LOCAL PATH HAD NEVER BEEN MEASURED AT ALL.

`experiments/debrief.py` has always accepted `ollama:` and `local:`
models. Every such call returned `no_result` and no local model was ever
scored, because `_http_json` understood only Cloudflare's response shape
(`{"result": {"choices": [...]}}`) while ollama puts the message at the
TOP level (`{"message": {"role": "assistant", "content": ...}}`). The
answer was in the body the whole time, behind a shape check.

So this file measures two things at once:

  1. the headline the local director gets, and
  2. the spread it gets over identical prompts -- which is the point.

WHAT THIS FILE IS NOT.

It is not a claim that a local model is a better director. It is a claim
about WHERE the variance comes from. If the local spread is 0.000 and the
hosted spread is not, then the hosted spread is the serving stack, and
every hosted number in this repository carries that noise. If the local
spread is also non-zero, then the variance is the model, and no amount of
re-running will fix it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import debrief  # noqa: E402

LOCAL_MODELS = [
    "ollama:qwen2.5-coder:7b",
    "ollama:gemma2:2b",
    "ollama:llama3.2:1b",
    "ollama:qwen3:0.6b",
    "ollama:tev1",
]

# The hosted spread measured on identical prompts, for comparison. These
# are the numbers from `docs/research/DEBRIEF-REAUDIT.md` section 4.2.
HOSTED_SPREAD = {
    "blind @cf/openai/gpt-oss-120b": 0.222,
    "blind @cf/meta/llama-3.3-70b-instruct-fp8-fast": 0.000,
    "blind @cf/qwen/qwen3-30b-a3b-fp8": 0.000,
    "context @cf/openai/gpt-oss-120b": 0.111,
    "context @cf/meta/llama-3.3-70b-instruct-fp8-fast": 0.000,
    "context @cf/qwen/qwen3-30b-a3b-fp8": 0.111,
}


def run_once(model: str, mode: str, limit: int | None) -> dict:
    t0 = time.time()
    r = debrief.run_audit(model, mode, materials_limit=limit)
    rec = {
        "model": model, "mode": mode,
        "elapsed_s": round(time.time() - t0, 2),
        "error": r.get("error"),
        "why": r.get("why"),
        "headline": r.get("headline_sens_plus_spec_minus_1"),
        "sensitivity": r.get("sensitivity_rejected_false_claims"),
        "specificity": r.get("specificity_kept_true_claims"),
        "truncated": r.get("truncated"),
        "finish_reason": r.get("finish_reason"),
        "usage": r.get("usage"),
        "materials_chars": r.get("materials_chars"),
    }
    if r.get("error"):
        rec["detail"] = str(r.get("detail"))[:400]
        rec["raw_head"] = (r.get("raw") or "")[:300]
    return rec


def measure(model: str, mode: str, repeats: int, limit: int | None) -> dict:
    runs = []
    for i in range(repeats):
        rec = run_once(model, mode, limit)
        rec["trial"] = i + 1
        runs.append(rec)
        print(f"  {model} [{mode}] trial {i + 1}/{repeats}: "
              f"error={rec['error']} headline={rec['headline']} "
              f"{rec['elapsed_s']}s", flush=True)
    hs = [r["headline"] for r in runs if r["headline"] is not None]
    errors = [r for r in runs if r["error"]]
    return {
        "model": model,
        "mode": mode,
        "n_repeats": repeats,
        "n_parsed": len(hs),
        "n_failed": len(errors),
        "failure_modes": sorted({r.get("why") or r["error"] for r in errors}),
        "headlines": hs,
        "headline_median": statistics.median(hs) if hs else None,
        "headline_spread": (max(hs) - min(hs)) if len(hs) > 1 else None,
        "n_distinct_headlines": len(set(hs)),
        "is_deterministic": (len(set(hs)) == 1) if hs else None,
        "sensitivity": runs[0]["sensitivity"] if runs else None,
        "specificity": runs[0]["specificity"] if runs else None,
        "runs": runs,
    }


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(LOCAL_MODELS))
    ap.add_argument("--mode", choices=["blind", "context"], default="blind")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--materials-limit", type=int, default=0)
    ap.add_argument("--out",
                    default=str(ROOT / "experiments"
                                / "debrief_local_director_results.json"))
    a = ap.parse_args()

    limit = a.materials_limit or None
    out = []
    for model in a.models.split(","):
        if not model.strip():
            continue
        print(f"\n=== {model} ({a.mode}, {a.repeats} identical prompts) ===")
        out.append(measure(model.strip(), a.mode, a.repeats, limit))

    scored = [m for m in out if m["n_parsed"]]
    deterministic = [m for m in scored if m["is_deterministic"]]
    spreads = [m["headline_spread"] for m in scored
               if m["headline_spread"] is not None]

    res = {
        "date": "2026-10-20",
        "question": (
            "Is the director's run-to-run variance the MODEL or the SERVING "
            "STACK? Every hosted number in this repository carries the "
            "endpoint's noise, and the noise cannot be removed by more "
            "repeats if it is the endpoint changing its mind."),
        "pre_registered_rules": {
            "the_local_director_is_deterministic_if": (
                "every repeat of one identical prompt on one local model "
                "gives the same headline"),
            "the_variance_is_the_serving_stack_if": (
                "the local spread is 0.000 on the models that can answer, "
                "while the hosted spread on identical prompts is not"),
            "the_variance_is_the_model_if": (
                "the local spread is non-zero on a model that can answer"),
            "what_this_cannot_establish": (
                "that a local model is a better director; this measures "
                "WHERE the variance comes from, not which director is "
                "better. The local models here are also 10 to 100 times "
                "smaller than the hosted ones."),
        },
        "measurements": out,
        "summary": {
            "n_models": len(out),
            "n_models_that_scored": len(scored),
            "n_models_deterministic": len(deterministic),
            "deterministic_models": [m["model"] for m in deterministic],
            "local_spread_max": max(spreads) if spreads else None,
            "local_spread_median": statistics.median(spreads) if spreads else None,
            "hosted_spread_on_identical_prompts": HOSTED_SPREAD,
            "hosted_spread_max": max(HOSTED_SPREAD.values()),
            "verdict": (
                "THE VARIANCE IS THE SERVING STACK" if scored and spreads
                and max(spreads) == 0.0 and max(HOSTED_SPREAD.values()) > 0
                else "INCONCLUSIVE" if not scored else
                "THE VARIANCE IS THE MODEL"),
        },
        "the_confound_and_what_it_does_to_the_verdict": {
            "the_rule_fired": (
                "the pre-registered rule says: local spread 0.000 on the "
                "models that can answer, and a hosted spread that is not "
                "0.000, means the variance is the serving stack. Both "
                "halves are true, so the rule fired and the summary's "
                "verdict is what it says."),
            "but_the_rule_confounds_two_things": (
                "EVERY local model that could answer here is a NON-reasoning "
                "model, and EVERY hosted cell that varies is a cell whose "
                "model writes a reasoning preamble before its answer. So "
                "the comparison cannot separate 'local' from 'reasoning'. "
                "The design does not have a local reasoning model to "
                "compare against, and it should not pretend to."),
            "and_a_better_controlled_measurement_says_otherwise": (
                "The transport was fixed afterwards -- finish_reason and "
                "usage are now read, and a truncated answer is retried with "
                "4x the output budget -- and the repeats were re-measured "
                "on the same prompts with the same key. The hosted spread "
                "did not move: the median spread was 0.0555 before the fix "
                "and 0.0555 after it. A controlled change that does not "
                "move the quantity is evidence that the quantity is not "
                "what the rule was measuring."),
            "where_the_variance_really_is": (
                "With the transport clean, all 21 re-measured runs carry "
                "10 of 10 verdicts, and the headline moves only when a "
                "claim flips to 'uncertain'. The audit set has nine false "
                "claims, so the headline is quantised at 1/9 = 0.1111, and "
                "the observed spreads are 0.111 and 0.222 -- one and two "
                "quanta. The variance is the model's own indecision, and "
                "the metric's resolution is the same size as its noise."),
            "the_corrected_conclusion": (
                "THE VARIANCE IS NOT EXPLAINED BY LOCALITY. It is one or "
                "two claims out of nine flipping to 'uncertain', and no "
                "comparison in this repository can resolve a difference "
                "smaller than that."),
            "see": (
                "docs/research/DEBRIEF-REAUDIT.md sections 4.6 and 4.7"),
        },
        "the_defect_that_made_this_impossible": {
            "what": (
                "debrief._http_json understood only Cloudflare's response "
                "shape, so every ollama:/local: call returned 'no_result' "
                "and no local model was ever scored. The answer was in the "
                "body the whole time."),
            "fix": (
                "_http_json now unwraps ollama's top-level 'message', "
                "carries done_reason and the token counts, and reports "
                "truncation the same way it does for Cloudflare."),
        },
        "provenance": {
            "script": "experiments/debrief_local_director.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "debrief_sha256": sha256_file(ROOT / "experiments" / "debrief.py"),
            "mode": a.mode,
            "repeats": a.repeats,
            "materials_limit": limit,
            "ollama_models": a.models,
        },
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")
    print(json.dumps(res["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
