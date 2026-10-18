#!/usr/bin/env python
"""
Two models, the same questions, the same probes, the same reduction.

    /home/proart/projects/glasses/.venv/bin/python experiments/local_vs_cloud.py

The question this answers is NOT "is the local model as good as the cloud
one". It is:

    When the two disagree about an utterance, is the QUESTION ambiguous, or
    is one model better at it?

Nothing else in the repo can separate those cases. If both models place
garbage at full tilt on the same track, the defect is in the question. If
they diverge, at least one of them is answering something the other is not
seeing -- and the disagreement is the interesting place to look.

Both paths reduce through judged._scale_position / judged._scale, so every
difference below is a difference in the ANSWERS.
"""

from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import judged, local_judge, question_health          # noqa: E402

TRACKS = ["TIMBRE", "INTENTION", "ATTENTION", "GESTURE"]


def cloud_ccs(text, model="jev-1.14.0"):
    readings, _ = judged.read_tracks(text, model=model)
    return ({r.track: r.value for r in readings},
            {r.track: r.detail.get("gate") for r in readings})


def main() -> int:
    have_key = bool(os.environ.get("TYPESAFE_API_KEY")
                    or os.environ.get("TYPESAFEAI_KEY"))
    local = local_judge.LocalJudge()
    if not local.available():
        print(f"no local model at {local.base_url}")
        return 2
    print(f"local: {local.model} at {local.base_url}")
    print(f"cloud: jev-1.14.0 (key present: {have_key})")
    print("model note: `tev1:latest` declares capability 'decision' and "
          "refuses BOTH /api/generate and /api/chat on Ollama 0.40.5 "
          "(\"does not support\"), so it cannot serve as a judge here.\n")

    probes = ([("garbage", t) for t in question_health.GARBAGE_PROBES]
              + [("real", t) for t in question_health.REAL_PROBES])

    rows = []
    t0 = time.time()
    for kind, text in probes:
        loc = local.judge(text)
        row = {"kind": kind, "text": text,
               "local_ok": loc.ok, "local_why": loc.why,
               "local_cc": dict(loc.ccs) if loc.ok else {},
               "local_gate": dict(loc.gates) if loc.ok else {}}
        if have_key:
            ccs, gates = cloud_ccs(text)
            row["cloud_cc"], row["cloud_gate"] = ccs, gates
        rows.append(row)
        tag = "ok " if loc.ok else "ERR"
        print(f"  [{tag}] {kind:7s} {text[:40]:42s} "
              + "  ".join(f"{t[:4]}={row['local_cc'].get(t, -1):3d}"
                          for t in TRACKS))
    dt = time.time() - t0

    def med(kind, field, track):
        vals = [r[field][track] for r in rows
                if r["kind"] == kind and r["local_ok"] and track in r[field]]
        return statistics.median(vals) if vals else None

    def cmed(kind, field, track):
        vals = [r[field][track] for r in rows
                if r["kind"] == kind and "cloud_cc" in r and track in r[field]]
        return statistics.median(vals) if vals else None

    def norm(a, b):
        return None if a is None or b is None else (b - a) / 127.0

    print(f"\n## Separation (garbage -> real), normalised to the 0..127 CC range")
    print(f"\n| Track | local garbage | local real | local sep | cloud garbage | cloud real | cloud sep | both healthy |")
    print("|---|---|---|---|---|---|---|---|")

    disagreements = []
    for t in TRACKS:
        lg, lr = med("garbage", "local_cc", t), med("real", "local_cc", t)
        cg, cr = cmed("garbage", "cloud_cc", t), cmed("real", "cloud_cc", t)
        lsep, csep = norm(lg, lr), norm(cg, cr)
        lh = lsep is not None and lsep >= 0.30
        ch = csep is not None and csep >= 0.30
        f = lambda x: "  -   " if x is None else f"{x:6.3f}"
        print(f"| {t} | {f(lg)} | {f(lr)} | {f(lsep)} | {f(cg)} | {f(cr)} | "
              f"{f(csep)} | {'both' if lh and ch else ('LOCAL' if lh else ('CLOUD' if ch else 'NEITHER'))} |")
        if lg is not None and cg is not None and abs(lg - cg) >= 32:
            disagreements.append((t, lg, cg))

    print(f"\n## Gate separation (real minus garbage); both must be POSITIVE")
    print(f"\n| Track | local gate g/garbage | local gate g/real | local sep | cloud gate g/garbage | cloud gate g/real | cloud sep | both separate |")
    print("|---|---|---|---|---|---|---|---|")
    for t in TRACKS:
        lg = med("garbage", "local_gate", t)
        lr = med("real", "local_gate", t)
        cg = cmed("garbage", "cloud_gate", t)
        cr = cmed("real", "cloud_gate", t)
        ls = None if lg is None or lr is None else lr - lg
        cs = None if cg is None or cr is None else cr - cg
        # "Agree" in the first version of this table only compared SIGNS, so
        # two models that both fail printed "yes". Agreeing that a track does
        # not work is not agreeing it does. Both must separate positively.
        if ls is None or cs is None:
            same = "-"
        elif ls > 0 and cs > 0:
            same = "yes"
        elif ls <= 0 and cs <= 0:
            same = "**NEITHER**"
        else:
            same = "**SPLIT**"
        f = lambda x: "  -   " if x is None else f"{x:6.2f}"
        print(f"| {t} | {f(lg)} | {f(lr)} | {f(ls)} | {f(cg)} | {f(cr)} | {f(cs)} | {same} |")

    if disagreements:
        print("\n## Where the two models disagree by >= 32 CC on garbage")
        for t, lg, cg in disagreements:
            print(f"  {t}: local {lg} vs cloud {cg}")

    errs = [r for r in rows if not r["local_ok"]]
    if errs:
        print(f"\n## {len(errs)} local failures -- excluded from every median above")
        for r in errs:
            print(f"  {r['text'][:40]!r}: {r['local_why']}")

    print(f"\n{len(probes)} probes in {dt:.1f}s local "
          f"({dt/max(len(probes),1):.2f}s each, includes model load)")


if __name__ == "__main__":
    raise SystemExit(main())