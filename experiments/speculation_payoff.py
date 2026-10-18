"""
speculation_payoff.py -- does speculating actually beat waiting, or is it just
a more complicated way of doing nothing?

MOTIVATION.

`src/speculate.py` exists because `turn.py` fires on ZERO of 680 real pause
runs: the longest gap anywhere in the corpus outside one essay is 2.4 s, so a
peripheral that waits for 3 s never responds. The gate's premise is that a
retractable bet beats no response.

That premise deserves a number rather than an argument, and the number is not
"accuracy" -- accuracy is fixed by the evidence and cannot be tuned. The
number is the TRADE:

    latency      how long before the system starts acting
    interrupts   how often it acts on someone who was not finished

Those two move in opposite directions as the speculate bar drops, and the
interesting object is the frontier between them, not either endpoint.

WHAT IS MEASURED, AND WHAT IS NOT.

Ground truth is manufactured (`turn_routes.py`): real speech with a silence
inserted at a controlled point, so END / RESUME are known by construction.

For each speculate bar we report, over every case:

    latency_p50/p90   seconds of silence before the bet opens
    interrupts        bets that were VOIDED -- the system acted on a speaker
                      who was not finished. This is the failure that matters:
                      interrupting someone is not recoverable.
    reach             bets that stuck
    accuracy          reach / (reach + interrupts)

And critically: **what waiting costs.** A bar of 3.0 s produces zero bets on
this corpus, which is an accuracy of None and a latency of infinity. Reporting
that as "100% accurate" would be the single most misleading number available,
so it is reported as what it is -- no decision at all.

THE HONEST CONCLUSION THIS IS DESIGNED TO REACH.

It is entirely possible that the frontier is bad: that at every latency where
you act, you interrupt most people, and the correct answer is to stay silent
and say so. `turn.py` already has the vocabulary for that (`UNRESOLVED`,
`TURN_YIELD_FLOOR_S`). This script is built to be able to return that answer
rather than to argue for speculation.

Run: python experiments/speculation_payoff.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import speculate, turn, watcher, wavio                     # noqa: E402

SR = 16000
CORPUS = ROOT / "examples" / "realaudio"

CELLS = {
    "narration": CORPUS / "alice01_excerpt.wav",
    "poetry": CORPUS / "corpus" / "poetry_weeds_by_the_wall_1702_librivox.wav",
    "essay": CORPUS / "corpus" / "essay_perpetual_peace_librivox.wav",
    "dialogue": CORPUS / "corpus" / "shortstory_shortstory058_1405_librivox.wav",
    "shakespeare": CORPUS / "corpus"
                   / "shakespeare_midsummernightsdream_1201_librivox.wav",
    "children": CORPUS / "corpus"
                / "children_english_fairy_tales_joy_librivox.wav",
    "oration": CORPUS / "corpus" / "oration_jfks19620912.wav",
}

SILENCE_LENGTHS = (0.20, 0.35, 0.60, 0.90, 1.30, 1.80)


def voiced_runs(samples, min_s=0.8):
    from src import energy
    ef, _ = energy.rms_series(samples, 800, 160, SR)
    if not ef:
        return []
    segs = energy.segment(ef, 160 / SR, peak_rms=max(f.rms for f in ef))
    out = []
    for s in segs:
        if not s.speech:
            continue
        a, b = int(s.start * SR), int(s.end * SR)
        if (b - a) / SR >= min_s:
            out.append((a, b))
    return out


def build_cases(samples, cell, n_per_cell):
    """
    END and RESUME only. Those are the two that a turn decision has to tell
    apart, and RESUME is the one that punishes a system for acting early.

    MID is left out deliberately here: a pause inside continuous speech is
    not a turn boundary at all, so scoring against it would measure something
    nobody is trying to decide.
    """
    runs = voiced_runs(samples)
    if len(runs) < 4:
        return []
    step = max(1, len(runs) // max(1, n_per_cell))
    cases = []
    for k, (a, b) in enumerate(runs[::step][:n_per_cell]):
        span = b - a
        for sil in SILENCE_LENGTHS:
            gap = [0.0] * int(sil * SR)
            cases.append({"cell": cell, "silence_s": sil, "truth": "end",
                          "audio": samples[a:b] + gap})
            cut = a + int(span * 0.55)
            cases.append({"cell": cell, "silence_s": sil, "truth": "resume",
                          "audio": samples[a:cut] + gap + samples[cut:b]})
    return cases


def score_bar(cases, speculate_s, commit_s):
    """
    Run every case through the gate at one threshold.

    A bet on an END case that reaches commit_s is a correct action; a bet on
    a RESUME case is an interruption, however it later settles, because the
    system acted on a speaker who was not finished.
    """
    g = speculate.SpeculationGate(speculate_s=speculate_s, commit_s=commit_s)
    reached, interrupted = [], []
    for c in cases:
        w = watcher.Watcher(SR, adaptive=True)
        acted = False
        for i in range(0, len(c["audio"]), 1600):
            for ev in w.push(c["audio"][i:i + 1600]):
                for ch in g.feed(ev, w.elapsed_s):
                    if ch["state"] == speculate.HOLDING:
                        acted = True
                        if c["truth"] == "resume":
                            interrupted.append(c["silence_s"])
                        else:
                            reached.append(c["silence_s"])
    total = len(reached) + len(interrupted)
    return {
        "speculate_s": speculate_s,
        "acted": total,
        "reached": len(reached),
        "interrupted": len(interrupted),
        "accuracy": (round(len(reached) / total, 4) if total else None),
        # latency = how much silence the system waited before acting
        "latency_p50": (round(statistics.median(reached + interrupted), 3)
                        if total else None),
    }


def score_waiting(cases, commit_s):
    """
    What the strict alternative does on the same material.

    Reported as a refusal with a number, not as an accuracy of 1.0.
    """
    decided = sum(1 for c in cases if c["silence_s"] >= commit_s)
    return {
        "policy": f"wait {commit_s}s",
        "decided": decided,
        "of": len(cases),
        "decision_rate": round(decided / len(cases), 4),
        "note": ("no decision is not a correct decision; a peripheral that "
                 "never acts has 100% accuracy and 100% uselessness"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--commit-s", type=float, default=3.0)
    ap.add_argument("--bars", default="0.20,0.30,0.40,0.60,0.90,1.30,1.80,3.00")
    ap.add_argument("--out", default="experiments/speculation_payoff_results.json")
    args = ap.parse_args()

    cases = []
    for cell, path in CELLS.items():
        if not path.is_file():
            print(f"skip {cell}: absent")
            continue
        samples = wavio.read_wav(str(path)).samples
        built = build_cases(samples, cell, args.per_cell)
        print(f"{cell:12s} {len(built):4d} cases")
        cases.extend(built)
    print(f"\ntotal {len(cases)} cases "
          f"({sum(1 for c in cases if c['truth'] == 'end')} END / "
          f"{sum(1 for c in cases if c['truth'] == 'resume')} RESUME)\n")

    bars = [float(x) for x in args.bars.split(",")]
    rows = [score_bar(cases, b, args.commit_s) for b in bars]
    waiting = score_waiting(cases, args.commit_s)

    print("=" * 74)
    print("THE FRONTIER: what waiting buys and what acting costs")
    print("=" * 74)
    print(f"{'speculate_s':>11s} {'acted':>6s} {'reached':>8s} "
          f"{'interrupted':>12s} {'accuracy':>9s} {'latency_p50':>12s}")
    print("-" * 74)
    for r in rows:
        acc = "  none" if r["accuracy"] is None else f"{r['accuracy']:9.3f}"
        lat = "     -" if r["latency_p50"] is None else f"{r['latency_p50']:12.3f}"
        print(f"{r['speculate_s']:11.2f} {r['acted']:6d} {r['reached']:8d} "
              f"{r['interrupted']:12d} {acc} {lat}")

    print(f"\nSTRICT ALTERNATIVE -- {waiting['policy']}:")
    print(f"  decides on {waiting['decided']}/{waiting['of']} cases "
          f"({waiting['decision_rate']:.1%})")
    print(f"  {waiting['note']}")

    best = [r for r in rows if r["accuracy"] is not None]
    print("\nREADING IT")
    if not best:
        print("  no bar ever produced a decision on this material.")
    else:
        top = max(best, key=lambda r: r["accuracy"])
        print(f"  best accuracy {top['accuracy']:.3f} at "
              f"speculate_s={top['speculate_s']:.2f} "
              f"({top['interrupted']} interruptions)")
        safe = [r for r in best if r["accuracy"] >= 0.9]
        if safe:
            lo = min(safe, key=lambda r: r["speculate_s"])
            print(f"  cheapest bar reaching 90% accuracy: "
                  f"{lo['speculate_s']:.2f}s "
                  f"(latency {lo['latency_p50']:.2f}s, "
                  f"{lo['interrupted']} interruptions)")
        else:
            print("  NO bar reaches 90% accuracy. The evidence does not")
            print("  support acting at all on this material, and the honest")
            print("  default stays 'say you do not know'.")
        worst = min(best, key=lambda r: r["speculate_s"])
        if worst["accuracy"] is not None and worst["accuracy"] < 0.5:
            print(f"  at {worst['speculate_s']:.2f}s accuracy is "
                  f"{worst['accuracy']:.3f} -- worse than a coin flip.")

    out = Path(args.out)
    out.write_text(json.dumps({
        "commit_s": args.commit_s,
        "n_cases": len(cases),
        "bars": rows,
        "waiting": waiting,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()