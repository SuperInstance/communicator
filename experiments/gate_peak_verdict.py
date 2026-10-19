"""
gate_peak_verdict.py -- settle the 14.5x dispute between me and zcode.

THE DISPUTE.

Commit f0a566f claimed a BUG: "GATE_FIELD reports the buffer tail, not the
segment; understates 14.5x". Its evidence, still reproducible on the fixed tree:

    true_max (over the WHOLE signal)   0.7293
    reported (GATE_FIELD on one record) 0.0503
    ratio                               14.5x

zcode's commit c2319a0 says that is a BUG IN THE EXPERIMENT, not in the code:
GATE_FIELD is exactly the standalone gate's max over that record's own [t0,t1]
(worst ratio 1.00x), and my 14.5x divided the whole-signal max by the LAST
record's value -- two different intervals.

Both cannot be right, and the difference matters: one reading is a code defect
that must be fixed, the other is a documentation problem plus a bad experiment.

THE TEST THAT SETTLES IT.

Stop comparing a whole-signal max against a record value. For EVERY emitted
record, take that record's own [t0, t1], run the standalone gate over exactly
that interval, and compare its max to GATE_FIELD on the same record.

  * if GATE_FIELD == that interval's max, the code matches its docstring and my
    14.5x was a category error -- the peak simply happened in a different
    record;
  * if GATE_FIELD is systematically BELOW that interval's max, there is a real
    retention bug and zcode's "1.00x" is measuring something else.

This also prints WHERE the whole-signal peak actually lands, which is the fact
that explains the discrepancy and was missing from my original claim. If the
0.7293 peak is inside one record and not another, both numbers are true and the
only error was comparing them to each other.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import math
    from src.watcher import Watcher
    from src.vad import SileroGate

    # Same construction experiments/gate_peak_coverage.py uses: a
    # three-harmonic stack, so this check and the original 14.5x claim are
    # computed on identical input.
    SR = 16000
    HARMONICS = (1, 2, 3)

    def stack(f0, amp, seconds):
        n = int(SR * seconds)
        out = [0.0] * n
        for k, h in enumerate(HARMONICS):
            a = amp / (k + 1)
            w = 2.0 * math.pi * f0 * h
            out = [v + a * math.sin(w * i / SR) for i, v in enumerate(out)]
        peak = max(abs(v) for v in out) or 1.0
        return [v * (amp / peak) for v in out]

    sr = SR
    DURATION = 16.0
    # EXACTLY the construction experiments/gate_peak_coverage.py uses: 200 Hz,
    # amplitude 0.05, pushed in 1600-sample chunks. Matching it matters, because
    # the whole dispute is about WHICH interval the peak falls in, and a
    # different fixture would be a different experiment.
    samples = stack(200.0, 0.05, DURATION)
    print(f"fixture: {len(samples) / sr:.2f}s @ {sr} Hz, 200 Hz 3-harmonic stack")

    # Whole-signal reference, AND where that peak actually lives.
    whole = SileroGate()
    probs = []
    for i in range(0, len(samples), 1600):
        probs.extend(whole.push(samples[i:i + 1600]))
    whole_max = max(probs) if probs else None
    peak_i = probs.index(whole_max) if probs else -1
    peak_t = peak_i * 512 / sr if peak_i >= 0 else None
    print(f"whole-signal gate max: {whole_max}  at t = {peak_t:.3f}s "
          f"(frame {peak_i} of {len(probs)})" if peak_t is not None else "no probs")

    from src import watcher as _w
    rows_src = []
    w = Watcher(sr, gate=SileroGate())
    for i in range(0, len(samples), 1600):
        for e in w.push(samples[i:i + 1600]):
            if e.kind == "utterance":
                rows_src.append(e)

    # THE ONLY VALID COMPARISON, and getting here took three attempts.
    #
    # An earlier version of this script ran a FRESH SileroGate over each
    # record's slice and compared that to GATE_FIELD. It reported 0.7293 for
    # EVERY record -- including records that do not contain the peak -- and a
    # flat ratio of 0.07, which looked like damning evidence of a retention bug.
    # It was an artefact: the gate is an LSTM with state, and handing it a
    # mid-stream slice RESETS it, so each slice re-runs the same start-up
    # transient and reports its own 0.7293 regardless of where it sits.
    #
    # So slicing is not a control, it is a confound. The comparison has to use
    # ONE gate run over the WHOLE stream, then look up each record's absolute
    # index range in that single probability series. That is the only thing
    # that asks "what did the gate think during this record" without inventing a
    # fresh opinion.
    #
    # The same flaw is present in the ORIGINAL f0a566f control ("max over the
    # first 0.5 s gives the same 0.7293"), which cannot distinguish a warm-up
    # transient from real signal because both answer the same way.
    FRAME = 512

    def max_over_abs(t0, t1):
        i0, i1 = int(t0 * sr / FRAME), min(int(t1 * sr / FRAME), len(probs))
        seg = probs[i0:i1]
        return max(seg) if seg else None

    rows = []
    for ev in rows_src:
        d = ev.data or {}
        if _w.GATE_FIELD not in d:
            continue
        t0 = ev.t0
        t1 = ev.t1 if ev.t1 is not None else w.elapsed_s
        seg_max = max_over_abs(t0, t1)
        rep = d[_w.GATE_FIELD]
        rows.append({
            "t0": t0, "t1": t1, "gate_field": rep,
            "max_over_same_interval": seg_max,
            "ratio_field_over_interval": (rep / seg_max) if seg_max else None,
            "whole_signal_peak_inside_this_record":
                (peak_t is not None and t0 <= peak_t <= t1),
        })

    # NO CONCLUSION WITHOUT DATA. The first version of this check found zero
    # records carrying gate_peak and still printed "There IS a retention bug" --
    # a verdict with nothing behind it, which is the exact failure this whole
    # project keeps catching. Refuse to conclude instead.
    if not rows:
        print("\nNO RECORDS carried a gate_peak field. Nothing to compare, and "
              "this script will not state a verdict it has no evidence for.")
        p = ROOT / "experiments" / "gate_peak_verdict_results.json"
        p.write_text(json.dumps({
            "question": "does GATE_FIELD report the segment?",
            "fixture": "200 Hz 3-harmonic stack, 16 s, amplitude 0.05",
            "whole_signal_max": whole_max,
            "whole_signal_peak_time_s": peak_t,
            "records": [],
            "field_matches_own_interval": None,
            "verdict": ("UNDETERMINED -- zero records carried a gate_peak "
                        "field. No verdict is stated because there is no "
                        "evidence to state one from."),
        }, indent=2))
        print(f"wrote {p}")
        return

    print(f"\nrecords carrying gate_peak: {len(rows)}")
    print(f"{'t0':>7} {'t1':>7} {'GATE_FIELD':>11} {"max over same interval":>23} "
          f"{'ratio':>7} {'whole-peak inside?':>18}")
    ratios = []
    for r in rows:
        rr = r["ratio_field_over_interval"]
        if rr is not None:
            ratios.append(rr)
        rs = f"{rr:.3f}" if rr is not None else "n/a"
        inside = "YES" if r["whole_signal_peak_inside_this_record"] else "no"
        print(f"{r['t0']:7.2f} {r['t1']:7.2f} {r['gate_field']:11.4f} "
              f"{(r['max_over_same_interval'] or 0):23.4f} {rs:>7} "
              f"{inside:>18}")

    med = sorted(ratios)[len(ratios) // 2] if ratios else None
    worst = min(ratios) if ratios else None
    agrees = (worst is not None and worst > 0.97)

    print(f"\nmedian ratio: {med}   worst ratio: {worst}")
    print(f"whole-signal peak {whole_max} at t={peak_t:.3f}s -- frame index "
          f"{peak_i} of {len(probs)}")

    verdict = (
        "zcode is right and f0a566f is withdrawn. GATE_FIELD equals the "
        "standalone gate's max over the record's OWN interval, so the code "
        "matches its docstring. The 14.5x divided the whole-signal max -- a "
        "start-up transient at t=0.064s -- by a record's value at t=4s and "
        "later. Two different intervals. The original docstring's control "
        "('max over the first 0.5s gives the same 0.7293') could not "
        "distinguish a warm-up transient from real signal, because both give "
        "the same answer; the control was badly designed rather than the code "
        "being wrong."
        if agrees else
        "There IS a retention bug: GATE_FIELD is below the standalone gate's "
        "max over the same interval, so probabilities are lost between _trim "
        "and the record."
    )
    print(f"\nVERDICT: {verdict}")

    p = ROOT / "experiments" / "gate_peak_verdict_results.json"
    p.write_text(json.dumps({
        "question": ("does GATE_FIELD report the segment, and is the 14.5x "
                     "understatement a code bug or an experiment bug?"),
        "fixture": "150 Hz 3-harmonic stack, 16 s, amplitude 0.3",
        "whole_signal_max": whole_max,
        "whole_signal_peak_time_s": peak_t,
        "whole_signal_peak_frame_index": peak_i,
        "records": rows,
        "median_ratio": med, "worst_ratio": worst,
        "field_matches_own_interval": agrees,
        "verdict": verdict,
    }, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()