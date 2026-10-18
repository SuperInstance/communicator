#!/usr/bin/env python
"""
Can a SMALL local model hit voice-loop latency? Phase 2.8 says no for 7B.

Measured: cloud 0.13 s/turn, local qwen2.5-coder:7b 7.17 s/turn warm. That is
55x too slow to drive a synthesiser, so "local-first" currently means "local
for the measured arithmetic only" -- the five acoustic tracks need no model at
all, which is already true.

The question this answers is whether a smaller model can carry the JUDGED half.
Two things have to hold, and a sweep that only measures latency would hide
the failure:

  1. FAST. Under ~1 s/turn, or the voice loop cannot use it.
  2. STILL SEPARATES. The same garbage/real batteries as Phase 2.7-2.8, with
     the same normalisation. A 0.5B model that answers everything with 0.5 is
     fast and useless.

Latency without separation is the classic way to report a local-first win that
does not exist, so both are printed together and neither alone passes.

    /home/proart/projects/glasses/.venv/bin/python experiments/small_model_sweep.py
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import local_judge, question_health                    # noqa: E402

# Ordered smallest-first. Pulled on demand; a model that will not pull is
# recorded as unmeasured rather than skipped silently.
CANDIDATES = [
    "qwen2.5:0.5b",
    "qwen3:0.6b",
    "llama3.2:1b",
    "qwen2.5:1.5b",
    "gemma2:2b",
]

TRACKS = ["TIMBRE", "INTENTION", "ATTENTION", "GESTURE"]

# TIMBRE is excluded from the health count. Phase 2.7 measured its presence
# gate at 0.10-0.26 on BOTH batteries -- the words carry no vocal colour --
# and calibration.py reached that independently. A model that "separates"
# TIMBRE from text is therefore not scoring a win, it is inventing a
# dimension, and counting it as healthy would reward the exact failure the
# two instruments spent a phase ruling out.
HEALTH_TRACKS = ["INTENTION", "ATTENTION", "GESTURE"]
LATENCY_BAR = 1.0          # seconds per turn, the voice-loop budget
SEP_BAR = 0.30             # normalised CC separation, Phase 2.7's bar


def pull(model: str) -> bool:
    try:
        r = subprocess.run(["ollama", "pull", model], capture_output=True,
                           timeout=1800)
        return r.returncode == 0
    except Exception:                                      # noqa: BLE001
        return False


def present(model: str) -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags",
                                    timeout=5) as r:
            body = json.loads(r.read().decode())
        # EXACT name match, including the tag. The first version compared
        # only the part before the colon, so `qwen2.5:1.5b` counted as
        # present because `qwen2.5-coder:7b` was installed -- the pull was
        # skipped and the run 404'd. A missing row in a sweep reads as
        # "measured and failed" unless the check is exact.
        return any(m["name"] == model for m in body.get("models", []))
    except Exception:                                      # noqa: BLE001
        return False


def sweep(model: str, probes, n_reps: int = 1):
    j = local_judge.LocalJudge(model=model)
    # One warm-up call that is NOT recorded: the first request loads weights
    # into VRAM, and including it would make a small model look 50x slower
    # than it is -- which is the exact error Phase 2.8 nearly made.
    t0 = time.time()
    warm = j.judge(probes[0][1])
    load_s = time.time() - t0
    if not warm.ok:
        return {"model": model, "ok": False, "why": f"warmup: {warm.why}",
                "load_s": round(load_s, 2)}

    times, rows = [], []
    for kind, text in probes:
        for _ in range(n_reps):
            t0 = time.time()
            r = j.judge(text)
            times.append(time.time() - t0)
            if r.ok:
                rows.append({"kind": kind, "text": text, "cc": dict(r.ccs),
                             "gate": dict(r.gates)})
            else:
                rows.append({"kind": kind, "text": text, "cc": {}, "gate": {},
                             "why": r.why})

    def med(kind, field, track):
        v = [r[field][track] for r in rows
             if r["kind"] == kind and track in r[field]]
        return statistics.median(v) if v else None

    seps = {}
    for t in TRACKS:
        g, rr = med("garbage", "cc", t), med("real", "cc", t)
        seps[t] = None if g is None or rr is None else round((rr - g) / 127.0, 3)

    n_healthy = sum(1 for t in HEALTH_TRACKS
                    if seps.get(t) is not None and seps[t] >= SEP_BAR)
    # TIMBRE separation is reported separately and never counted: it is a
    # red flag, not a credit.
    timbre_suspect = (seps.get("TIMBRE") is not None
                      and seps["TIMBRE"] >= SEP_BAR)
    lat = statistics.median(times)
    return {
        "model": model, "ok": True, "load_s": round(load_s, 2),
        "latency_s": round(lat, 3), "latency_max_s": round(max(times), 3),
        "n_calls": len(times), "n_ok": sum(1 for r in rows if r["cc"]),
        "cc_garbage": {t: med("garbage", "cc", t) for t in TRACKS},
        "cc_real": {t: med("real", "cc", t) for t in TRACKS},
        "separation": seps, "n_healthy_tracks": n_healthy,
        "health_tracks": HEALTH_TRACKS,
        "timbre_separated_but_should_not": timbre_suspect,
        "fast_enough": lat < LATENCY_BAR,
        "verdict": ("USABLE" if lat < LATENCY_BAR and n_healthy >= 3
                    else ("fast but blind" if lat < LATENCY_BAR
                          else ("separates but too slow" if n_healthy >= 3
                                else "neither"))),
    }


def main() -> int:
    probes = ([("garbage", t) for t in question_health.GARBAGE_PROBES]
              + [("real", t) for t in question_health.REAL_PROBES])
    print(print(f"{len(probes)} probes per model; load time recorded separately "
          f"from steady-state latency")
    print(f"healthy tracks counted over {HEALTH_TRACKS}; TIMBRE separation "
          f"from text is reported as a FALSE POSITIVE, never as a win")
    print(f"bars: latency < {LATENCY_BAR}s, separation >= {SEP_BAR} "
          f"on at least 3 of 4 tracks\n")

    results = []
    for m in CANDIDATES:
        if not present(m):
            print(f"[pull] {m}")
            if not pull(m):
                print(f"  [skip] {m}: pull failed")
                results.append({"model": m, "ok": False, "why": "pull failed"})
                continue
        print(f"[run ] {m}")
        r = sweep(m, probes)
        results.append(r)
        if not r.get("ok"):
            print(f"  [fail] {r.get('why')}")
            continue
        print(f"  load {r['load_s']}s | latency {r['latency_s']}s "
              f"(max {r['latency_max_s']}s) | healthy tracks "
              f"{r['n_healthy_tracks']}/4 | {r['verdict']}")
        for t in TRACKS:
            mark = ("  <- TIMBRE separating from text is a FALSE POSITIVE"
                    if t == "TIMBRE" and r["timbre_separated_but_should_not"]
                    else ("  (not counted)" if t == "TIMBRE" else ""))
            print(f"    {t:10s} garbage {r['cc_garbage'][t]}  real "
                  f"{r['cc_real'][t]}  sep {r['separation'][t]}{mark}")

    print("\n## Summary")
    print(f"| model | load s | latency s | healthy tracks | verdict |")
    print("|---|---|---|---|---|")
    for r in results:
        if not r.get("ok"):
            print(f"| {r['model']} | - | - | - | {r.get('why')} |")
        else:
            print(f"| {r['model']} | {r['load_s']} | {r['latency_s']} | "
                  f"{r['n_healthy_tracks']}/4 | {r['verdict']} |")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "small_model_sweep_results.json")
    with open(out, "w") as fh:
        json.dump({"bars": {"latency_s": LATENCY_BAR, "separation": SEP_BAR},
                   "results": results}, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())