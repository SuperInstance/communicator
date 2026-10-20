"""
director_repeatability.py -- is the LLM director's score a property of the
model, or a property of the run?

WHY THIS FILE EXISTS.

The director is audited against this project's own retractions, and the audit
is reported as a headline: "gpt-oss-120b, context mode, +0.857". That number
is quoted as if it were a property of the model.

It is not. Two audits of the SAME model, the SAME claims, the SAME mode and
the SAME prompt were run and gave:

    blind, @cf/openai/gpt-oss-120b     +0.095
    blind, @cf/openai/gpt-oss-120b     +0.429

Same model. Same prompt. Same ten claims in the same shuffled order. The
client already sends `temperature: 0.0`, so this is not sampling noise: it is
hosted-inference non-determinism, and it is large enough to flip the
headline by a third of its range.

That matters more than it looks. A score that moves by 0.33 between two
identical runs cannot support the sentence "the director scores +0.857". What
it can support is a distribution.

THE MEASUREMENT.

Hold the prompt BYTE-IDENTICAL and vary only the number of times it is sent.
The prompt is built once, hashed, and reused -- which is the whole point, and
is why `debrief.py` was refactored to expose `build_audit_prompt()` and
`score_audit()` as separate pure functions.

Reported per model and mode:

  * the headline for every run
  * the number of DISTINCT headlines, and the spread (max - min)
  * the median, because a mean of a bimodal score is a number nobody should
    quote
  * PER CLAIM, how many distinct verdicts that claim drew across the runs.
    An unstable claim is the interesting one: a claim the director scores
    correctly on some runs and endorses on others is a claim the director is
    NOT reliable on, and hiding that inside an average is how a false claim
    gets signed off.

THE CONTROL, and it is not optional.

A deterministic stub director -- a fixed verdict list, scored N times -- must
produce a spread of exactly 0.0 on every axis. If it does not, the harness
is adding variance and the measurement is worthless. This is the control that
makes the result falsifiable: a claim of "the model varies" that a
deterministic stub also satisfies would be a bug, not a finding.

The two degenerate directors are also reported, computed rather than
asserted, because the metric's whole purpose is to put both at 0.0:

    always-accept  sens 0.0  spec 1.0  -> headline  0.0
    always-reject  sens 1.0  spec 0.0  -> headline  0.0

WHAT THIS DOES NOT CLAIM.

It does not claim the director is useless. A distribution with a median of
+0.86 and a spread of 0.33 is still a director worth having. It claims
something narrower and more useful: **quote the distribution, not the run.**
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import debrief  # noqa: E402


# ---------------------------------------------------------------- the control

def stub_verdicts(n_claims: int, verdict: str = "supported") -> dict:
    """A deterministic director: the same verdicts every time.

    This is the control. Scored N times it must give an identical headline
    every time, so that any spread seen with a real model is the MODEL's
    spread and not the harness's.
    """
    return {"verdicts": [{"index": i, "verdict": verdict, "reason": "stub"}
                         for i in range(n_claims)]}


def degenerate_baselines(pairs) -> dict:
    """The two directors the metric must score at 0.0, computed not asserted."""
    out = {}
    for name, verdict in (("always_accept", "supported"),
                          ("always_reject", "unsupported")):
        res = debrief.score_audit(pairs, stub_verdicts(len(pairs), verdict))
        out[name] = {
            "sensitivity": res["sensitivity_rejected_false_claims"],
            "specificity": res["specificity_kept_true_claims"],
            "headline": res["headline_sens_plus_spec_minus_1"],
        }
    return out


# ---------------------------------------------------------------- the analysis

def summarise_runs(scored: list[dict]) -> dict:
    """The distribution of a set of scored runs.

    Pure function of `scored`, so it is testable without a network.
    """
    headlines = [r["headline_sens_plus_spec_minus_1"] for r in scored
                 if r.get("headline_sens_plus_spec_minus_1") is not None]
    sens = [r["sensitivity_rejected_false_claims"] for r in scored
            if r.get("sensitivity_rejected_false_claims") is not None]
    spec = [r["specificity_kept_true_claims"] for r in scored
            if r.get("specificity_kept_true_claims") is not None]

    # Per claim: how many DISTINCT verdicts did it draw across the runs?
    per_claim = {}
    for r in scored:
        for row in r.get("rows") or []:
            i = row["index"]
            e = per_claim.setdefault(i, {
                "index": i, "claim": row["claim"],
                "is_false_claim": row["is_false_claim"],
                "verdicts": {}, "correct_count": 0})
            e["verdicts"][row["verdict"]] = e["verdicts"].get(row["verdict"], 0) + 1
            if row.get("correct"):
                e["correct_count"] += 1
    for e in per_claim.values():
        e["distinct_verdicts"] = len(e["verdicts"])
        e["verdict_counts"] = e["verdicts"]
        e["stable"] = len(e["verdicts"]) <= 1
        # An UNSTABLE claim on a false claim is the dangerous kind: the
        # director endorses it on some runs and catches it on others.
        e["unstable_and_false_claim"] = (not e["stable"]) and e["is_false_claim"]
        del e["verdicts"]

    unstable = [e for e in per_claim.values() if not e["stable"]]
    return {
        "n_runs": len(scored),
        "n_scored": len(headlines),
        "n_unparseable": sum(1 for r in scored if r.get("error")),
        "headlines": headlines,
        "distinct_headlines": sorted(set(headlines)),
        "n_distinct_headlines": len(set(headlines)),
        "headline_min": min(headlines) if headlines else None,
        "headline_max": max(headlines) if headlines else None,
        "headline_median": statistics.median(headlines) if headlines else None,
        "headline_mean": statistics.fmean(headlines) if headlines else None,
        "headline_spread": (max(headlines) - min(headlines)) if headlines else None,
        "sensitivity_min": min(sens) if sens else None,
        "sensitivity_max": max(sens) if sens else None,
        "specificity_min": min(spec) if spec else None,
        "specificity_max": max(spec) if spec else None,
        "n_unstable_claims": len(unstable),
        "unstable_claims": unstable,
        "per_claim": sorted(per_claim.values(), key=lambda e: e["index"]),
    }


# ---------------------------------------------------------------- the runner

def run_repeatability(model: str, mode: str, n: int, seed: int = 20261019,
                      materials_limit: int | None = None) -> dict:
    pairs = debrief.audit_pairs(seed)
    prompt, lim = debrief.build_audit_prompt(mode, pairs, materials_limit)
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()

    # -- CONTROL: the deterministic stub, scored n times. It must not move.
    control_runs = []
    for _ in range(n):
        res = debrief.score_audit(pairs, stub_verdicts(len(pairs)))
        control_runs.append(res)
    control = summarise_runs(control_runs)

    # -- the real model, the SAME prompt, n times.
    scored = []
    errors = []
    for i in range(n):
        r = debrief.call_model(model, prompt, 3000)
        if r.get("error"):
            errors.append({"run": i, "error": r.get("error"),
                           "detail": str(r.get("detail"))[:300]})
            scored.append({"error": r.get("error")})
            continue
        obj = debrief.extract_json(r.get("text") or "")
        if not obj or "verdicts" not in obj:
            errors.append({"run": i, "error": "unparseable",
                           "raw": (r.get("text") or "")[:300]})
            scored.append({"error": "unparseable"})
            continue
        scored.append(debrief.score_audit(pairs, obj, model, mode, lim,
                                           r.get("elapsed_s"),
                                           r.get("text") or ""))

    return {
        "model": model,
        "mode": mode,
        "n_requested": n,
        "seed": seed,
        "materials_chars": lim,
        "prompt_sha256": prompt_sha,
        "prompt_chars": len(prompt),
        "n_claims": len(pairs),
        "control": {
            "what": ("a deterministic stub director, scored the same number "
                     "of times with the same scorer"),
            "headline_spread": control["headline_spread"],
            "n_distinct_headlines": control["n_distinct_headlines"],
            "headlines": control["headlines"],
            "n_unstable_claims": control["n_unstable_claims"],
        },
        "degenerate_baselines": degenerate_baselines(pairs),
        "result": summarise_runs(scored),
        "errors": errors,
        "runs": [{"headline": r.get("headline_sens_plus_spec_minus_1"),
                  "sensitivity": r.get("sensitivity_rejected_false_claims"),
                  "specificity": r.get("specificity_kept_true_claims"),
                  "uncertain": r.get("uncertain"),
                  "unparseable": r.get("unparseable"),
                  "error": r.get("error"),
                  "elapsed_s": r.get("elapsed_s"),
                  "verdicts": [row["verdict"] for row in (r.get("rows") or [])]}
                 for r in scored],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="@cf/openai/gpt-oss-120b")
    ap.add_argument("--models", default=None,
                    help="comma-separated; each is measured separately")
    ap.add_argument("--mode", choices=("blind", "context"), default="blind")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--materials-limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args(argv)

    models = ([m.strip() for m in a.models.split(",") if m.strip()]
              if a.models else [a.model])
    lim = a.materials_limit or None

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for m in models:
        print(f"\n=== REPEATABILITY [{a.mode}] {m}  x{a.runs} ===")
        res = run_repeatability(m, a.mode, a.runs, materials_limit=lim)
        all_results.append(res)
        ctl = res["control"]
        r = res["result"]
        print(f"  prompt sha256   {res['prompt_sha256'][:16]}  "
              f"({res['prompt_chars']} chars, identical every run)")
        print(f"  CONTROL (stub)  spread {ctl['headline_spread']}  "
              f"distinct headlines {ctl['n_distinct_headlines']}  "
              f"unstable claims {ctl['n_unstable_claims']}")
        for name, b in res["degenerate_baselines"].items():
            print(f"  baseline {name:14s} headline {b['headline']:+.3f}")
        print(f"  headlines       {['%+.3f' % h for h in r['headlines']]}")
        print(f"  distinct        {r['n_distinct_headlines']}   "
              f"min {r['headline_min']:+.3f}  median {r['headline_median']:+.3f}  "
              f"max {r['headline_max']:+.3f}  spread {r['headline_spread']:.3f}")
        print(f"  unstable claims {r['n_unstable_claims']} of {res['n_claims']}")
        for e in r["unstable_claims"]:
            flag = " <-- FALSE CLAIM" if e["unstable_and_false_claim"] else ""
            print(f"    [{e['index']}] {e['verdict_counts']}  {e['claim'][:60]}{flag}")
        if res["errors"]:
            print(f"  errors: {len(res['errors'])}")
            for e in res["errors"][:3]:
                print(f"    run {e['run']}: {e.get('error')} {e.get('detail', '')}")

    p = outdir / (f"director_repeatability_{a.mode}"
                  + (f"_{lim}" if lim else "") + ".json")
    p.write_text(json.dumps({"mode": a.mode, "materials_limit": lim,
                              "results": all_results}, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
