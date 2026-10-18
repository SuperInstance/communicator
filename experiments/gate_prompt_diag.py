#!/usr/bin/env python
"""
Why did every small model return CC 0 on every track and both batteries?

HYPOTHESIS, from reading the prompt rather than from the numbers: the gate
instruction contains an example value.

    "if the text is gibberish, noise or has no such property, report a LOW
     number near 0"

That sentence offers the model a CONSTANT answer. A 7B model answered around
it (garbage 0.10-0.38, real 0.85-0.97). Two different sub-1B models answered
it literally: every gate 0.0, every CC 0, on real speech as well as gibberish.
A model that follows an example value exactly is not measuring anything.

So the sweep's 0/4-healthy result is ambiguous between two very different
claims:

  * small models cannot do this task, OR
  * the prompt hands them a constant and they are obedient

This script separates them by printing the RAW model JSON, and by re-asking
with the example value removed. It is a diagnostic, not a benchmark: one probe
per condition, because the question is "what does the model literally emit".

    /home/proart/projects/glasses/.venv/bin/python experiments/gate_prompt_diag.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import local_judge                                  # noqa: E402

REAL = "Run the tests."
GARBAGE = "qwop zzz brxt"

# The current instruction, and a rewrite with no example value in it. The
# rewrite states the property and nothing about what the answer looks like;
# the model must decide from the utterance.
NO_LEAD = ("For each gate, report how confident you are that the property is "
           "genuinely present in this utterance. Do not default: decide from "
           "the utterance itself.")


def raw(model: str, text: str, system: str) -> tuple:
    import urllib.request
    payload = {"model": model,
               "messages": [{"role": "system", "content": system},
                            {"role": "user",
                             "content": local_judge.build_prompt(text)}],
               "stream": False, "format": local_judge.build_schema(),
               "options": {"temperature": 0}}
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            body = json.loads(r.read().decode())
    except Exception as exc:                                 # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}, time.time() - t0
    try:
        return json.loads(body["message"]["content"]), time.time() - t0
    except Exception:                                        # noqa: BLE001
        return {"raw": body}, time.time() - t0


def main() -> int:
    models = sys.argv[1:] or ["qwen2.5:0.5b", "qwen3:0.6b",
                              "qwen2.5-coder:7b"]
    for m in models:
        j = local_judge.LocalJudge(model=m)
        if not j.available():
            print(f"== {m}: no local server")
            continue
        j.judge("warm up the model")          # load weights, not recorded
        print(f"\n{'=' * 72}\n== {m}")
        for label, system in (("CURRENT prompt", local_judge.SYSTEM),
                              ("NO-LEADING prompt", local_judge.SYSTEM.replace(
                                  " -- if the text is "
                                  "gibberish, noise or has no such property, "
                                  "report a LOW number near 0", "")
                                  + " " + NO_LEAD)):
            for kind, text in (("garbage", GARBAGE), ("real", REAL)):
                obj, dt = raw(m, text, system)
                if "error" in obj or "raw" in obj:
                    print(f"  [{label:16s}] {kind:8s} ERROR {str(obj)[:90]}")
                    continue
                gates = {k: obj.get(k) for k in
                         ("gate_intention", "gate_attention", "gate_gesture")}
                gate_vals = [v for v in gates.values() if v is not None]
                print(f"  [{label:16s}] {kind:8s} {dt:6.2f}s gates={gates} "
                      f"intent_pos={obj.get('intention', {}).get('position')}")
                if not gate_vals:
                    print("      !! no gate keys in the response at all")
                elif all(v == 0 for v in gate_vals):
                    print("      !! every gate is exactly 0 -- constant answer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())