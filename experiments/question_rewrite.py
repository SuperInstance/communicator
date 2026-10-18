#!/usr/bin/env python
"""
Run the question-rewrite probe against the live model and print the table.

Paired design: for each probe text, every variant is asked in immediate
succession. Model drift within a run therefore lands on all variants roughly
equally, instead of making the variant that happened to run first look worse.

    set -a && . /home/proart/projects/glasses/.env && set +a
    /home/proart/projects/glasses/.venv/bin/python experiments/question_rewrite.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import judged, question_health, rewrite_probe      # noqa: E402

VARIANTS = ["baseline", "reworded", "gated", "gated_reworded"]


def main() -> int:
    model = os.environ.get("JEV_MODEL", "jev-1.14.0")
    if not (os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFEAI_KEY")):
        print("no TYPESAFE_API_KEY; nothing would be measured")
        return 2

    from typesafe_sdk import TypeSafeClient
    client = TypeSafeClient()

    garbage = question_health.GARBAGE_PROBES
    real = question_health.REAL_PROBES
    probes = [("garbage", t) for t in garbage] + [("real", t) for t in real]

    print(f"model pinned: {model}")
    print(f"fingerprints: " + ", ".join(
        f"{name}={rewrite_probe.build_questions(name) and _fp(name)}"
        for name in VARIANTS))
    print(f"{len(probes)} probes x {len(VARIANTS)} variants = "
          f"{len(probes) * len(VARIANTS)} calls\n")

    rows = {name: [] for name in VARIANTS}
    failures = []
    t0 = time.time()
    calls = 0
    for kind, text in probes:
        for name in VARIANTS:
            v = rewrite_probe.variant(name)
            got = rewrite_probe.measure(client, [text], v, model=model)
            calls += 1
            row = got[0]
            if not row["ok"]:
                failures.append((name, text, row.get("why")))
            rows[name].append((kind, row))
            mark = "ok " if row["ok"] else "ERR"
            cells = "  ".join(
                f"{t}={row['cc'].get(t, 0):.2f}" for t in
                sorted(row["cc"])) if row["ok"] else row.get("why", "")
            print(f"  [{mark}] {kind:7s} {name:16s} {cells}")
    dt = time.time() - t0

    print(f"\n{calls} calls in {dt:.1f}s ({dt/max(calls,1):.2f}s each)")

    scored = {}
    for name in VARIANTS:
        g = [r for k, r in rows[name] if k == "garbage"]
        rl = [r for k, r in rows[name] if k == "real"]
        scored[name] = rewrite_probe.score_variant(
            rewrite_probe.variant(name), g, rl)

    print()
    print(rewrite_probe.format_report(scored))

    if failures:
        print(f"## {len(failures)} failed calls — median of the rest only")
        for name, text, why in failures:
            print(f"  {name}: {why}  <- {text[:48]!r}")

    import json
    out = {
        "model": model,
        "calls": calls,
        "seconds": round(dt, 1),
        "failures": len(failures),
        "variants": {
            name: [
                {
                    "track": r.track,
                    "cc_garbage": r.cc_garbage,
                    "cc_real": r.cc_real,
                    "cc_separation": r.cc_separation,
                    "conf_garbage": r.conf_garbage,
                    "conf_real": r.conf_real,
                    "conf_separation": r.conf_separation,
                    "gate_garbage": r.gate_garbage,
                    "gate_real": r.gate_real,
                    "gate_separation": r.gate_separation,
                    "verdict": rewrite_probe.verdict(r),
                }
                for r in res
            ]
            for name, res in scored.items()
        },
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "question_rewrite_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}")
    return 0


def _fp(name: str) -> str:
    """Fingerprint of a variant's wording, so a result names what it asked."""
    import hashlib
    import json

    v = rewrite_probe.variant(name)
    spec = {"name": name, "specs": [list(s) for s in v.specs],
            "gates": sorted(v.gates)}
    if name == "baseline":
        spec["judged"] = judged.question_fingerprint()
    if name in ("reworded", "gated_reworded"):
        spec["scales"] = {
            "intention": rewrite_probe.INTENTION_SCALE_V2,
            "attention": rewrite_probe.ATTENTION_CHOICE_V2,
            "timbre": rewrite_probe.TIMBRE_SCALE_V2,
        }
    return hashlib.sha1(
        json.dumps(spec, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:12]


if __name__ == "__main__":
    raise SystemExit(main())