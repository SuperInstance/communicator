"""
director_proposal_stability.py -- is the director's PROPOSAL stable?

WHY THIS FILE EXISTS.

`director_repeatability.py` measured the director's *verdicts*: on a
byte-identical prompt it produced four distinct audit headlines over a spread
of 0.619, and it endorsed a known-false claim on half the runs.

But an auditor is not the thing this project wants. What the plan wants is a
director that says **what to do next**. That is the `next_experiment` block
of the production debrief: title, hypothesis, method, controls, null,
falsifier. Those are the fields the project asked the LLM to fill, and the
plan explicitly puts the LLM "between experiments" to decide what to improve
and what experiment to run next.

**A proposal that changes every time it is asked for the same question is
not a plan.** And unlike an audit headline, a proposal has consequences: it
is what gets built next.

So this file asks the question that actually decides whether the director is
usable:

    Ask the director N times, with the SAME material and the SAME prompt,
    and count how many DIFFERENT experiments it proposes.

THE CONTROL, again, and it is not optional.

A deterministic stub -- a fixed debrief -- is scored N times with the same
comparer. It must give an identical proposal every time. If it does not, the
comparer is adding variance and the measurement is worthless.

WHAT IS REPORTED.

  * how many distinct `title`s, `hypothesis`s, `method`s, `controls`s,
    `null`s and `falsifier`s across the runs
  * a FIELD-STABILITY table: for each field, how many runs produced it and
    how many distinct values it took
  * how many runs produced ALL SIX required fields (a director that drops
    the null or the falsifier is not proposing a test)
  * the `unsafe_claims` the director raised, and how many DISTINCT unsafe
    claims it raised -- because the triage signal is the claim list, and a
    claim list that changes every run is a claim list nobody can act on
  * the proposal that appeared MOST OFTEN, with its share

WHAT THIS DOES NOT CLAIM.

It does not claim a stable proposal is a good proposal. It measures
reproducibility, which is a precondition for usefulness and not a substitute
for it. A director that proposes the same wrong experiment every time is
still wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import debrief  # noqa: E402


REQUIRED_FIELDS = ("title", "hypothesis", "method", "controls", "null",
                   "falsifier")


# ---------------------------------------------------------------- the control

def stub_debrief() -> dict:
    """A deterministic director. The control."""
    return {
        "state_of_the_evidence": [{"claim": "stub", "evidence": "stub"}],
        "unsafe_claims": [{"claim": "stub unsafe", "why": "stub",
                           "action": "stub"}],
        "next_experiment": {f: "stub" for f in REQUIRED_FIELDS},
    }


# ---------------------------------------------------------------- the analysis

def _as_text(v) -> str:
    """A comparable string for a field that may be a string or a list."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " | ".join(_as_text(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v).strip()


# Words that carry no content when comparing two proposals about the same
# thing. Stopping the comparison at exact string equality would measure
# phrasing, not substance -- and two proposals for the same experiment are
# rarely phrased identically.
_STOP = {
    "the", "a", "an", "of", "on", "in", "to", "for", "and", "or", "is", "are",
    "be", "with", "that", "this", "it", "as", "at", "by", "from", "we", "our",
    "test", "tests", "testing", "measure", "measurement", "measurements",
    "experiment", "experiments", "use", "using", "under", "over",
}


def _tokens(s: str) -> set:
    """Content tokens of a string, for a substance comparison.

    Hyphenated compounds are SPLIT, because that is where the vocabulary
    mismatch lives: "cache-pressure", "low-cache" and "cache-constrained" are
    three ways of naming the same condition, and a tokeniser that keeps them
    whole reports zero overlap between three identical proposals.
    """
    out = set()
    for w in "".join(ch.lower() if (ch.isalnum() or ch == "-") else " "
                      for ch in s).split():
        for part in w.split("-"):
            if len(part) > 2 and part not in _STOP:
                out.add(part)
    return out


def jaccard(a: str, b: str) -> float:
    """Similarity of two strings by content tokens. 1.0 = same content."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 1.0 if (not ta and not tb) else 0.0
    return len(ta & tb) / len(ta | tb)


def pairwise_similarity(strings: list[str]) -> dict:
    """Mean and median pairwise token similarity of a list of strings.

    This is the measurement that separates "the proposal changed" from "the
    proposal was reworded". Exact string equality answers the first question;
    it cannot answer the second, and two runs proposing the same experiment
    in different words score 0 on it.
    """
    s = [x for x in strings if x]
    if len(s) < 2:
        return {"n": len(s), "mean": None, "median": None, "min": None,
                "pairs": []}
    vals = []
    pairs = []
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            v = jaccard(s[i], s[j])
            vals.append(v)
            pairs.append({"a": i, "b": j, "jaccard": round(v, 4)})
    pairs.sort(key=lambda p: p["jaccard"])
    return {"n": len(s), "n_pairs": len(vals),
            "mean": statistics.fmean(vals), "median": statistics.median(vals),
            "min": min(vals), "max": max(vals),
            "least_similar_pairs": pairs[:3],
            "most_similar_pairs": pairs[-3:]}


def pairwise_run_similarity(debriefs: list[dict], getter) -> dict:
    """Pairwise token similarity of a per-run TEXT, one comparison per pair.

    `getter` maps a debrief to a string (or list of strings) for that run.
    Comparing run i's text with run j's text answers "did these two runs say
    the same thing", which is the question -- pooling every claim from every
    run into one bag would measure the vocabulary, not the agreement.
    """
    texts = []
    for d in debriefs:
        if not isinstance(d, dict):
            continue
        v = getter(d)
        if isinstance(v, (list, tuple)):
            v = " | ".join(_as_text(x) for x in v)
        texts.append(_as_text(v))
    return pairwise_similarity(texts)


def compare_debriefs(debriefs: list[dict]) -> dict:
    """How much do N debriefs agree? Pure function, so it is testable.

    `debriefs` is a list of PARSED debrief objects (the JSON the model
    returned), not the raw text.
    """
    parsed = [d for d in debriefs if isinstance(d, dict)]
    n = len(parsed)
    if n == 0:
        return {"n": 0, "error": "no parsed debriefs"}

    # -- the proposal fields
    fields = {}
    for f in REQUIRED_FIELDS:
        values = [_as_text((d.get("next_experiment") or {}).get(f))
                  for d in parsed]
        present = [v for v in values if v]
        fields[f] = {
            "n_present": len(present),
            "n_distinct": len(set(present)),
            "stable": len(set(present)) <= 1,
            "distinct_values": sorted(set(present))[:5],
        }

    titles = [_as_text((d.get("next_experiment") or {}).get("title"))
              for d in parsed]
    titles = [t for t in titles if t]

    # -- the triage signal: the unsafe claims
    unsafe = []
    for d in parsed:
        for c in (d.get("unsafe_claims") or []):
            if isinstance(c, dict):
                unsafe.append(_as_text(c.get("claim")))
            else:
                unsafe.append(_as_text(c))
    unsafe = [u for u in unsafe if u]
    unsafe_counts = Counter(unsafe)

    # -- the evidence claims
    evidence = []
    for d in parsed:
        for c in (d.get("state_of_the_evidence") or []):
            if isinstance(c, dict):
                evidence.append(_as_text(c.get("claim")))
    evidence = [e for e in evidence if e]

    complete = sum(1 for d in parsed
                   if all(_as_text((d.get("next_experiment") or {}).get(f))
                          for f in REQUIRED_FIELDS))

    return {
        "n": n,
        "n_distinct_titles": len(set(titles)),
        "titles": titles,
        "most_common_title": (Counter(titles).most_common(1)[0]
                              if titles else None),
        "title_agreement": (Counter(titles).most_common(1)[0][1] / n
                            if titles else None),
        # The SUBSTANCE of the agreement, as opposed to the wording. A
        # director that proposes the same experiment in four different
        # sentences scores 0 on exact equality and ~0.6 here, and the second
        # number is the one that says whether it is proposing the same thing.
        "title_similarity": pairwise_similarity(titles),
        "unsafe_claim_similarity": pairwise_run_similarity(
            parsed, lambda d: [c.get("claim") if isinstance(c, dict) else c
                               for c in (d.get("unsafe_claims") or [])]),
        "proposal_similarity": pairwise_run_similarity(
            parsed, lambda d: {k: v for k, v in
                               (d.get("next_experiment") or {}).items()
                               if k != "title"}),
        "fields": fields,
        "n_stable_fields": sum(1 for f in fields.values() if f["stable"]),
        "n_runs_with_all_required_fields": complete,
        "frac_runs_with_all_required_fields": complete / n,
        "unsafe_claims": unsafe,
        "n_unsafe_claims": len(unsafe),
        "n_distinct_unsafe_claims": len(set(unsafe)),
        "unsafe_claim_counts": {k: v for k, v in
                                unsafe_counts.most_common()},
        "unsafe_claim_agreement": (unsafe_counts.most_common(1)[0][1] / len(unsafe)
                                   if unsafe else None),
        "evidence_claims": evidence,
        "n_distinct_evidence_claims": len(set(evidence)),
    }


def summarise_agreement(vectors: list[list[str]]) -> dict:
    """Per-field agreement across runs: how many DISTINCT values each field took.

    `vectors` is one list of per-field strings per run.
    """
    if not vectors:
        return {"error": "no vectors"}
    out = {}
    n_fields = len(vectors[0])
    for i in range(n_fields):
        vals = [v[i] for v in vectors if len(v) > i]
        vals = [v for v in vals if v]
        out[str(i)] = {
            "n_present": len(vals),
            "n_distinct": len(set(vals)),
            "agreement": (Counter(vals).most_common(1)[0][1] / len(vals)
                          if vals else None),
        }
    return out


# ---------------------------------------------------------------- the runner

def run_proposal_stability(model: str, n: int, materials_limit: int = 140000,
                            seed: int = 20261019) -> dict:
    # The material is snapshotted ONCE, so the prompt is byte-identical across
    # every run. If it were rebuilt each time, a lane editing a note would
    # change the input and the measurement would be of two different questions.
    materials = debrief.collect_materials(limit=materials_limit)
    prompt = debrief.DEBRIEF_PROMPT.replace("{materials}", materials)
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()

    # -- the control
    control = compare_debriefs([stub_debrief() for _ in range(n)])

    # -- the real model
    parsed, errors, raw = [], [], []
    for i in range(n):
        r = debrief.call_model(model, prompt, 4000)
        if r.get("error"):
            errors.append({"run": i, "error": r.get("error"),
                           "detail": str(r.get("detail"))[:300]})
            continue
        obj = debrief.extract_json(r.get("text") or "")
        if not isinstance(obj, dict):
            errors.append({"run": i, "error": "unparseable",
                           "raw": (r.get("text") or "")[:300]})
            continue
        parsed.append(obj)
        raw.append((r.get("text") or "")[:2000])

    comparison = compare_debriefs(parsed)
    vectors = []
    for d in parsed:
        ne = d.get("next_experiment") or {}
        vectors.append([_as_text(ne.get(f)) for f in REQUIRED_FIELDS])

    return {
        "model": model,
        "n_requested": n,
        "n_parsed": len(parsed),
        "seed": seed,
        "materials_chars": len(materials),
        "materials_limit": materials_limit,
        "prompt_sha256": prompt_sha,
        "prompt_chars": len(prompt),
        "control": {
            "what": "a deterministic stub debrief, compared the same way",
            "n_distinct_titles": control.get("n_distinct_titles"),
            "n_stable_fields": control.get("n_stable_fields"),
            "n_distinct_unsafe_claims": control.get("n_distinct_unsafe_claims"),
        },
        "comparison": comparison,
        "field_agreement": summarise_agreement(vectors),
        "errors": errors,
        "runs": [{"title": v[0] if v else None,
                  "next_experiment": (d.get("next_experiment") or {}),
                  "unsafe_claims": [
                      (c.get("claim") if isinstance(c, dict) else c)
                      for c in (d.get("unsafe_claims") or [])],
                  "state_of_the_evidence": [
                      (c.get("claim") if isinstance(c, dict) else c)
                      for c in (d.get("state_of_the_evidence") or [])],
                  "n_unsafe_claims": len(d.get("unsafe_claims") or []),
                  "n_evidence_claims": len(d.get("state_of_the_evidence") or [])}
                 for d, v in zip(parsed, vectors)],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="@cf/openai/gpt-oss-120b")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--materials-limit", type=int, default=140000)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args(argv)

    res = run_proposal_stability(a.model, a.runs, a.materials_limit)
    c = res["control"]
    cmp_ = res["comparison"]
    print(f"\n=== PROPOSAL STABILITY [{a.model}] x{res['n_requested']} ===")
    print(f"  prompt sha256   {res['prompt_sha256'][:16]}  "
          f"({res['prompt_chars']} chars, identical every run)")
    print(f"  parsed          {res['n_parsed']} of {res['n_requested']}")
    print(f"  CONTROL (stub)  distinct titles {c['n_distinct_titles']}  "
          f"stable fields {c['n_stable_fields']}  "
          f"distinct unsafe claims {c['n_distinct_unsafe_claims']}")
    print(f"  DISTINCT TITLES {cmp_.get('n_distinct_titles')}  "
          f"agreement {cmp_.get('title_agreement')}")
    for t in cmp_.get("titles") or []:
        print(f"    - {t[:78]}")
    print(f"  fields:")
    for f in REQUIRED_FIELDS:
        fi = (cmp_.get("fields") or {}).get(f) or {}
        print(f"    {f:12s} present {fi.get('n_present')}/{res['n_parsed']}  "
              f"distinct {fi.get('n_distinct')}  "
              f"{'STABLE' if fi.get('stable') else 'VARIES'}")
    print(f"  runs with ALL six required fields: "
          f"{cmp_.get('n_runs_with_all_required_fields')}/{res['n_parsed']}")
    for label, key in (("title similarity", "title_similarity"),
                       ("proposal similarity", "proposal_similarity"),
                       ("unsafe-claim similarity", "unsafe_claim_similarity")):
        s = cmp_.get(key) or {}
        if s.get("median") is not None:
            print(f"  {label:24s} median jaccard {s['median']:.3f}  "
                  f"mean {s['mean']:.3f}  min {s['min']:.3f}  "
                  f"max {s['max']:.3f}")
    print(f"  unsafe claims: {cmp_.get('n_unsafe_claims')} total, "
          f"{cmp_.get('n_distinct_unsafe_claims')} distinct, "
          f"top agreement {cmp_.get('unsafe_claim_agreement')}")
    for claim, cnt in list((cmp_.get("unsafe_claim_counts") or {}).items())[:6]:
        print(f"    x{cnt}  {claim[:70]}")
    if res["errors"]:
        print(f"  errors: {len(res['errors'])}")
        for e in res["errors"][:3]:
            print(f"    run {e['run']}: {e.get('error')} {e.get('detail', '')}")

    p = Path(a.out) / "director_proposal_stability.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
