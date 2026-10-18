#!/usr/bin/env python
"""
The Phase 2.7 probe, run against the LOCAL decision model.

Phase 2.7 measured, on the cloud, that three of four judged questions were
ill-posed and that a presence gate fixed them (ATTENTION CC separation -8 ->
+53). Phase 2.9 found a local model that speaks the same primitives. The
question this answers is the one the whole local-first design rests on:

    does the local model need the gate too, or does it get the tracks right
    where the cloud model needed one?

If tev1 separates without needing the gate, the gate was a patch for one
model's behaviour. If it needs the gate the same way, the gate is a property
of the question -- which is what the design assumes.

Same batteries, same reduction, same normalisation as Phase 2.7, so the
numbers are comparable rather than merely parallel.

    /home/proart/projects/glasses/.venv/bin/python experiments/local_question_health.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import judged, local_tev, question_health            # noqa: E402

CC_FULL = 127.0
SEP_BAR = 0.30


def run(client, probes):
    rows = []
    for kind, text in probes:
        t0 = time.time()
        readings, resolved = judged.read_tracks(text, client=client)
        rows.append({"kind": kind, "text": text, "s": time.time() - t0,
                     "resolved": resolved,
                     "cc": {r.track: r.value for r in readings},
                     "gate": {r.track: r.detail.get("gate") for r in readings},
                     "ok": all(r.ok for r in readings)})
    return rows


def med(rows, kind, field, track):
    v = [r[field][track] for r in rows
         if r["kind"] == kind and r["ok"] and track in r[field]]
    return statistics.median(v) if v else None


def main() -> int:
    probes = ([("garbage", t) for t in question_health.GARBAGE_PROBES]
              + [("real", t) for t in question_health.REAL_PROBES])

    client = local_tev.SystemOneClient()
    if not (client.available() and client.is_decision_model()):
        print(f"no local decision model at {client.base_url} ({client.model})")
        return 2
    print(f"model {client.model} | decision capability "
          f"{client.is_decision_model()} | fingerprint "
          f"{judged.question_fingerprint()}")
    print(f"{len(probes)} probes, 8 questions in ONE call each\n")

    run(client, probes[:1])                      # warm the weights
    rows = run(client, probes)

    print("| Track | CC garbage | CC real | sep | norm | gate garbage | gate real | gate sep | verdict |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in sorted(judged.CC):
        g, r = med(rows, "garbage", "cc", t), med(rows, "real", "cc", t)
        gg, gr = med(rows, "garbage", "gate", t), med(rows, "real", "gate", t)
        if g is None or r is None:
            print(f"| {t} | - | - | - | - | - | - | - | unmeasured |")
            continue
        sep = r - g
        norm = sep / CC_FULL
        gsep = None if gg is None or gr is None else gr - gg
        verdict = ("healthy" if norm >= SEP_BAR else
                   "ill-posed" if norm < 0 else "weak")
        f = lambda x, p=0: "  -  " if x is None else (
            f"{x*100:5.0f}%" if p else f"{x:6.1f}")
        print(f"| {t} | {f(g)} | {f(r)} | {f(sep)} | {norm:+.3f} | "
              f"{f(gg,1)} | {f(gr,1)} | {f(gsep)} | {verdict} |")

    lat = [r["s"] for r in rows]
    print(f"\nmedian {statistics.median(lat):.2f}s per turn "
          f"(min {min(lat):.2f}, max {max(lat):.2f}); voice budget 1.0s")
    bad = [r for r in rows if not r["ok"]]
    if bad:
        print(f"{len(bad)} probes failed and are excluded from every median")
    print("\nNOTE TIMBRE: Phase 2.7 showed its presence gate reads ~0.02 on "
          "real speech, so a\nhealthy TIMBRE row here would be a red flag, "
          "not a win -- the dimension is not\nin the text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())