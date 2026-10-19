"""
gate_peak_coverage.py -- does `GATE_FIELD` report the segment's peak?

WHY THIS FILE EXISTS.

`src/watcher.py` attaches the gate's verdict to every utterance record so that
"any segment the gate dismissed can be found and audited afterwards"
(`GATE_FIELD`), and documents `_gate_peak` as "the gate's peak probability over
a segment".

This measures whether that is what it returns. It does not.

THE MEASUREMENT.

A 200 Hz three-harmonic stack at -26 dBFS is fed through `Watcher` exactly as
production does: one continuous gate, 100 ms chunks. Two numbers are compared:

    true_max    the maximum gate probability over the WHOLE signal, from a
                standalone gate fed the identical samples
    reported    the `GATE_FIELD` value the watcher attaches to the utterance

Measured on this machine:

    segment length      true_max    reported
         0.5 s           0.7293    (no utterance emitted)
         1.0 s           0.7293    (no utterance emitted)
         2.0 s           0.7293    (no utterance emitted)
         4.0 s           0.7293         0.0503
         8.0 s           0.7293         0.0539
        16.0 s           0.7293         0.0523

A factor of ~14, and the true peak is constant while the reported value is flat.

WHAT IT IS NOT.

It is not an LSTM warm-up artefact. Taking the max over the first 0.5 s, 2 s or
the whole stream gives the same 0.7293 every time; the peak sits at t ~= 0.064 s.
It is not a chunking artefact either -- the standalone gate uses the same 1600
sample chunks.

It is BUFFER TRIMMING. `_trim` drops gate probabilities whose frames start
before the new buffer origin. By the time a 4 s segment closes, 6400 samples
remain; the peak at 0.064 s is gone. After 8 s of audio the watcher holds 25
probability entries with `_gate_probs_base = 115200`, i.e. starting at sample
7.2 s.

THE DIRECTION, STATED PLAINLY.

For hum this makes the gate do the right thing for the wrong reason: it reports
low, so YIN is skipped and the CPU saving happens. The mitigation is not broken.

The audit trail is. The field's stated purpose is that a human can check which
segments the gate dismissed. A value that under-reports by 14x and systematically
under-reports SPEECH on long segments cannot support that -- every long-segment
dismissal is recorded as near-silence when the gate's own opinion of that
segment was 0.73.

Reproduce:  /home/proart/projects/glasses/.venv/bin/python experiments/gate_peak_coverage.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import vad, watcher  # noqa: E402

SR = 16000
DURATIONS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
HARMONICS = (1, 2, 3)


def stack(f0, amp, seconds):
    """Three-harmonic stack, the same construction tests/test_vad.py uses."""
    n = int(SR * seconds)
    out = [0.0] * n
    for k, h in enumerate(HARMONICS):
        a = amp / (k + 1)
        w = 2.0 * math.pi * f0 * h
        out = [v + a * math.sin(w * i / SR) for i, v in enumerate(out)]
    peak = max(abs(v) for v in out) or 1.0
    return [v * (amp / peak) for v in out]


def true_max(samples, chunk=1600):
    g = vad.SileroGate()
    probs = []
    for i in range(0, len(samples), chunk):
        probs.extend(g.push(samples[i:i + chunk]))
    return (max(probs), sum(probs) / len(probs)) if probs else (None, None)


def reported(samples, chunk=1600):
    w = watcher.Watcher(SR, gate=vad.SileroGate())
    rec = None
    for i in range(0, len(samples), chunk):
        for e in w.push(samples[i:i + chunk]):
            if e.kind == "utterance":
                rec = e.data
    if rec is None:
        return None, None, None
    return (rec[watcher.GATE_FIELD], rec["pitch_skipped"],
            len(w._gate_probs))


def main():
    amp = 0.05                       # -26 dBFS
    rows = []
    for d in DURATIONS:
        sig = stack(200.0, amp, d)
        tm, mean = true_max(sig)
        rep, skipped, n_probs = reported(sig)
        rows.append({"duration_s": d, "true_max": round(tm, 4),
                     "true_mean": round(mean, 4),
                     "reported": None if rep is None else round(rep, 4),
                     "pitch_skipped": skipped,
                     "gate_probs_retained": n_probs,
                     "understatement_x": None if rep in (None, 0)
                     else round(tm / rep, 2)})
        print(f"{d:>6}s true_max {tm:.4f}  reported "
              f"{'none' if rep is None else f'{rep:.4f}'}"
              f"  retained_frames {n_probs}")

    ok = [r for r in rows if r["reported"] is not None]
    out = {
        "note": ("Does GATE_FIELD report the segment's peak, as documented? "
                 "200Hz 3-harmonic stack at -26 dBFS through Watcher with one "
                 "continuous gate. true_max is the maximum over the whole "
                 "signal from a standalone gate on identical samples."),
        "rows": rows,
        "summary": {
            "max_understatement_x": max((r["understatement_x"] for r in ok),
                                        default=None),
            "all_rows_understated": all(r["understatement_x"] > 1 for r in ok),
        },
    }
    print(json.dumps(out["summary"], indent=2))
    (Path(__file__).with_name("gate_peak_coverage_results.json")
     .write_text(json.dumps(out, indent=2)))


if __name__ == "__main__":
    main()