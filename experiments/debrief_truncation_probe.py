#!/usr/bin/env python3
"""
debrief_truncation_probe.py -- the audit's "unparseable" runs were answers
that were CUT OFF, not answers that were never given.

WHAT THIS IS ABOUT.

Three of the nine committed audit runs came back "unparseable": the model's
reply was not valid JSON. The record said only "unparseable", and the
natural reading of that word is that the model failed to answer.

It did not. It answered, and the answer was cut off.

`experiments/debrief.py::_http_json` rebuilt every response as
`{"text": text}` and DROPPED `finish_reason` and `usage`. A response that
the output budget cut off mid-answer and a response the model finished are
the same shape on the wire once `finish_reason` is thrown away. The
symptom was qwen3 and gpt-oss writing a prose preamble, running out of
room before the JSON arrived, and the record saying nothing about why.

`cloud_llm` already had a retry for reasoning models -- it repeats the call
with 4x the output budget when `content` comes back EMPTY. The truncated
case never reached it, because the content was not empty; it was cut.

THE OBVIOUS EXPLANATION WAS TESTED AND IT IS FALSE.

The natural hypothesis was that the material was too large for the smaller
models' context windows, and that the fit ladder should have shrunk it.
It is not that, and there is a structural reason it cannot be that in the
mode where the failures happened:

  * In BLIND mode the material budget DOES NOT ENTER THE PROMPT AT ALL.
    `build_audit_prompt` returns the claims and nothing else for blind
    mode, so the prompt is 701 input tokens whether the budget is 140,000
    characters or 8,000. Every blind-mode failure therefore had a prompt
    of 701 tokens, and the context window was never the issue.
  * The failure happened at the SMALLEST budget measured and NOT at the
    largest: qwen3-30b at 40,000 characters came back unparseable, and at
    140,000 and 60,000 it came back parsed.

So the budget that ran out was the OUTPUT budget, not the INPUT budget.

WHAT THIS FILE MEASURES.

  1. The base rate of the failure, at the committed budget, for a named
     model and mode.
  2. What the API said about every response: finish_reason, the input
     tokens and the COMPLETION tokens.
  3. Whether 4x the output budget recovers the SAME prompt.
  4. The control that refutes the material hypothesis: the input-token
     count at three different material budgets, in each mode.

WHAT THIS FILE IS NOT.

It is not a claim that the director is better after the fix. It is a
measurement of the INSTRUMENT: a run that was silently discarded as a
model failure was in fact a budget failure, and the fix is a transport
fix. Whether the recovered runs change the director's score is a separate
question, answered by re-running the audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import debrief  # noqa: E402

MODELS = {
    "qwen3": "@cf/qwen/qwen3-30b-a3b-fp8",
    "gpt-oss": "@cf/openai/gpt-oss-120b",
    "llama": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
}
MAX_TOKENS = 3000
RETRY_MAX_TOKENS = 12000


def _call(model: str, prompt: str, max_tokens: int) -> dict:
    """One call, with the transport facts kept."""
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0}
    url = (f"https://api.cloudflare.com/client/v4/accounts/"
           f"{debrief.ACCOUNT}/ai/run/{model}")
    hdr = {"Authorization": f"Bearer {os.environ['CF_TOKEN']}",
           "Content-Type": "application/json"}
    return debrief._http_json(url, body, hdr)


def _describe(r: dict) -> dict:
    text = r.get("text") or ""
    usage = r.get("usage") or {}
    obj = debrief.extract_json(text)
    verdicts = (obj or {}).get("verdicts") or [] if obj else []
    return {
        "finish_reason": r.get("finish_reason"),
        "truncated": bool(r.get("truncated")),
        "text_len": len(text),
        "parsed": obj is not None,
        "n_verdicts": len(verdicts),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "error": r.get("error"),
    }


def probe(model: str, mode: str, budget: int, trials: int) -> list:
    """N identical calls, with a recovery attempt on every truncated one."""
    pairs = debrief.audit_pairs(20261019)
    prompt = debrief.build_audit_prompt(mode, pairs, budget)[0]
    out = []
    for i in range(trials):
        first = _describe(_call(model, prompt, MAX_TOKENS))
        rec = {"trial": i + 1, "model": model, "mode": mode,
               "materials_limit": budget, "max_tokens": MAX_TOKENS,
               "first_attempt": first}
        if first["truncated"] or first["error"]:
            second = _describe(_call(model, prompt, RETRY_MAX_TOKENS))
            rec["retry_4x_output_budget"] = second
            rec["recovered"] = bool(second["parsed"])
        out.append(rec)
        print(f"  [{mode}] {model.split('/')[-1]} trial {i + 1}/{trials}: "
              f"finish={first['finish_reason']} "
              f"completion_tokens={first['completion_tokens']} "
              f"parsed={first['parsed']}"
              + (f" -> 4x recovered={rec['recovered']}"
                 if "recovered" in rec else ""), flush=True)
    return out


def budget_control(model: str, mode: str,
                   budgets=(140000, 60000, 40000)) -> dict:
    """The control that refutes the "too much material" hypothesis.

    If the failure were caused by too much material, the input-token count
    would fall with the budget and the failure would follow it. In BLIND
    mode the material is not in the prompt at all, so the input-token count
    must be IDENTICAL at every budget -- and the failure rate must not care
    what the budget is.
    """
    pairs = debrief.audit_pairs(20261019)
    rows = []
    for b in budgets:
        prompt = debrief.build_audit_prompt(mode, pairs, b)[0]
        d = _describe(_call(model, prompt, MAX_TOKENS))
        rows.append({"materials_limit": b, "prompt_chars": len(prompt),
                     "prompt_tokens": d["prompt_tokens"],
                     "completion_tokens": d["completion_tokens"],
                     "finish_reason": d["finish_reason"],
                     "parsed": d["parsed"]})
        print(f"  control {mode} @{b}: prompt_chars={len(prompt)} "
              f"prompt_tokens={d['prompt_tokens']} parsed={d['parsed']}",
              flush=True)
    counts = {r["prompt_tokens"] for r in rows}
    return {
        "budgets": list(budgets),
        "rows": rows,
        "prompt_tokens_are_identical_across_budgets": len(counts) == 1,
        "prompt_tokens_values": sorted(counts),
    }


def summarise(runs: list) -> dict:
    n = len(runs)
    trunc = [r for r in runs if r["first_attempt"]["truncated"]]
    retried = [r for r in runs if "retry_4x_output_budget" in r]
    rec = [r for r in retried if r["recovered"]]
    comp = [r["first_attempt"]["completion_tokens"] for r in runs
            if r["first_attempt"]["completion_tokens"] is not None]
    # A TRUNCATED ANSWER CAN STILL PARSE. The JSON object closed before the
    # cut, and the model kept writing prose after it. That case is worse
    # than a lost run, because the OLD scorer did not lose it -- it scored
    # the claims the model never got to as "unparseable" rows, which count
    # as WRONG. So a truncated answer was silently charged to the director.
    trunc_parsed = [r for r in trunc if r["first_attempt"]["parsed"]]
    short_trunc = [r for r in trunc_parsed
                   if r["first_attempt"]["n_verdicts"] < len(debrief.AUDIT_SET)]
    return {
        "n_trials": n,
        "n_truncated": len(trunc),
        "truncation_base_rate": (len(trunc) / n) if n else None,
        "n_retried": len(retried),
        "n_recovered_by_4x_output_budget": len(rec),
        "recovery_rate": (len(rec) / len(retried)) if retried else None,
        "n_truncated_but_parsed": len(trunc_parsed),
        "n_truncated_and_missing_verdicts": len(short_trunc),
        "verdicts_in_truncated_but_parsed": sorted(
            r["first_attempt"]["n_verdicts"] for r in trunc_parsed),
        "n_claims_in_the_audit_set": len(debrief.AUDIT_SET),
        "completion_tokens_when_truncated": sorted(
            r["first_attempt"]["completion_tokens"] for r in trunc
            if r["first_attempt"]["completion_tokens"] is not None),
        "completion_tokens_when_complete": sorted(
            r["first_attempt"]["completion_tokens"] for r in runs
            if not r["first_attempt"]["truncated"]
            and r["first_attempt"]["completion_tokens"] is not None),
        "completion_tokens_median_complete": (
            statistics.median([r["first_attempt"]["completion_tokens"]
                               for r in runs
                               if not r["first_attempt"]["truncated"]
                               and r["first_attempt"]["completion_tokens"]
                               is not None]) if comp else None),
    }


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen3", choices=sorted(MODELS))
    ap.add_argument("--mode", choices=["blind", "context"], default="blind")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--budget", type=int, default=140000)
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--out",
                    default=str(ROOT / "experiments"
                                / "debrief_truncation_probe_results.json"))
    a = ap.parse_args()

    if not os.environ.get("CF_TOKEN"):
        print("source ~/.config/cloudflare/env", file=sys.stderr)
        return 2
    model = MODELS[a.model]
    if not debrief.cf_agree(model):
        print(f"model {model} is not reachable", file=sys.stderr)
        return 2

    print(f"probe: {model} {a.mode} @ {a.budget} chars, {a.trials} trials")
    runs = probe(model, a.mode, a.budget, a.trials)

    control = {}
    if not a.skip_control:
        print("control: input-token count across three material budgets")
        control = budget_control(model, a.mode)

    res = {
        "date": "2026-10-20",
        "question": (
            "Three committed audit runs were recorded as 'unparseable'. "
            "Did the model fail to answer, or was the answer cut off by the "
            "output budget -- and is the material budget implicated?"),
        "pre_registered_rules": {
            "a_truncation_is": (
                "finish_reason == 'length' on the API response"),
            "a_recovery_is": (
                "the SAME prompt, same model, same mode, with 4x the output "
                "token budget, returns a parseable JSON object"),
            "the_material_hypothesis_is_refuted_if": (
                "in blind mode the input-token count is IDENTICAL at every "
                "material budget, because the material is not in the blind "
                "prompt at all"),
            "verdict": (
                "the unparseable runs are an OUTPUT-budget failure if the "
                "truncation rate is non-zero and the 4x retry recovers them"),
        },
        "the_defect": {
            "what": (
                "debrief._http_json rebuilt every response as "
                "{'text': text} and discarded finish_reason and usage, so a "
                "response cut off at the output cap was indistinguishable "
                "from a finished one. cloud_llm's retry only fired on EMPTY "
                "content, which a truncated answer never has."),
            "fix": (
                "finish_reason and usage are kept and surfaced as "
                "truncated/usage; cloud_llm retries a truncated answer with "
                "4x the output budget; run_audit records the transport facts "
                "on every run and distinguishes 'response_was_truncated' "
                "from 'no_json_verdicts_object'."),
        },
        "runs": runs,
        "summary": summarise(runs),
        "the_material_control": control,
        "provenance": {
            "script": "experiments/debrief_truncation_probe.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "debrief_sha256": sha256_file(
                ROOT / "experiments" / "debrief.py"),
            "model": model,
            "mode": a.mode,
            "materials_limit": a.budget,
            "max_tokens": MAX_TOKENS,
            "retry_max_tokens": RETRY_MAX_TOKENS,
            "trials": a.trials,
            "git_head": debrief.git_head() if hasattr(debrief, "git_head")
            else None,
        },
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")
    print(json.dumps(res["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
