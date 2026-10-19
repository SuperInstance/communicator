"""
hum_sweep.py -- where does the silero gate stop rejecting a hum, and what does
that cost?

MOTIVATION.

Commit `8eb05f6` put a silero-vad gate in front of the watcher on a measured
result: a 120 Hz hum at amplitude 0.05 took the watcher from 31.8% to 2.4% of
a core. That is the project's single most load-bearing performance claim, and
the hum fixture it rests on is a 60 Hz fundamental plus harmonics, which the
gate rejects cleanly (mean 0.0205).

The threat has a documented limit. `tests/test_vad.py::TestDeclaredLimits`
records that a pure tone at 150/175/200 Hz scores 0.66-0.82 and a 200 Hz
harmonic stack scores 0.97 -- real-speech territory. So the gate is known to
fail somewhere below 200 Hz. Nobody has measured WHERE.

That location is not academic, because 50 Hz mains and its harmonics are
exactly 50, 100, 150, 200, 250 Hz. Anything powered from the grid with a
transformer, a motor, or a rectifier lands on one of those fundamentals. The
shipped mitigation may not cover the hum it was built for.

WHAT IS MEASURED, AND WHY BOTH HALVES.

A probability alone would repeat the mistake already documented in
`test_vad.py`: scoring well against one threat proves nothing about the input
next to it. So every fundamental is measured two ways:

    gate       mean probability and the fraction of frames >= GATE_SPEECH
    cpu        percent of one core the WATCHER burns over 30 s of that signal,
               with the gate installed and with gate=None

The CPU column is the one that decides whether the mitigation worked, because
the gate's job is not to output zero -- it is to make the watcher not run YIN.
A signal the gate scores 0.97 defeats the mitigation no matter how the
probability is framed: YIN runs and the cost returns.

Both directions are reported, exactly as `test_vad.py` insists. The same
sweep is run over real speech at matched levels, so a fundamental that "looks
like speech" is compared against what speech actually scores.

AMPLITUDE IS SWEPT, NOT ASSUMED.

The original claim is amplitude-specific (0.05). A gate that rejects a hum at
0.05 and accepts it at 0.2 has a limit nobody wrote down. Each fundamental is
measured at three levels, and the reported finding is the WORST case across
levels -- the one that decides whether the gate holds in a loud room.

WHAT THIS FILE CANNOT SHOW.

Whether the hums here resemble any particular device. These are synthetic
harmonic stacks, not recordings of equipment, so they test the gate's
response to a frequency structure, not its response to a fan bearing or a
coil whine. That is a real limitation and it is the reason the conclusion is
phrased as a boundary in frequency rather than as a device verdict.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import vad, wavio, watcher  # noqa: E402

import numpy as np  # noqa: E402

SR = 16000
DUR_S = 30.0
CORPUS = Path(__file__).resolve().parents[1] / "examples/realaudio"
SPEECH = CORPUS / "alice01_excerpt.wav"

# Mains fundamentals first, then the band the gate is known to fail in.
FUNDAMENTALS = [50, 60, 100, 120, 150, 175, 200, 250, 300]
LEVELS = [0.02, 0.05, 0.10]
N_HARMONICS = 12


def harmonic_stack(f0, amp, n_harm=N_HARMONICS, rolloff=1.0):
    """A periodic buzz with 1/k amplitude rolloff -- a mains-driven device.

    Rolloff is 1/k rather than a flat stack because that is what a real
    harmonic source looks like: the fundamental dominates and the upper
    harmonics fall away. A flat stack would be an easier test than reality.

    numpy is used only to GENERATE the signal. It never touches the gate, the
    watcher, or any number on the hot path -- `src/` is stdlib and stays that
    way; this is an experiment harness and onnxruntime already requires numpy.
    """
    n = int(DUR_S * SR)
    t = np.arange(n, dtype=np.float64) / SR
    out = np.zeros(n, dtype=np.float64)
    for k in range(1, n_harm + 1):
        f = f0 * k
        if f >= SR / 2:
            break
        out += (amp / (k ** rolloff)) * np.sin(2.0 * math.pi * f * t)
    peak = np.abs(out).max() or 1.0
    return (out * (amp / peak)).astype(np.float32)


def gate_stats(samples, gate):
    """mean probability and frac>=threshold over the whole stream."""
    probs = []
    for i in range(0, len(samples), 1600):
        probs.extend(gate.push(samples[i:i + 1600]))
    if not probs:
        return {"mean": None, "frac": None}
    s = sorted(probs)
    return {"mean": round(sum(s) / len(s), 4),
            "frac": round(sum(1 for p in s if p >= watcher.GATE_SPEECH)
                          / len(s), 4)}


def cpu_percent(samples, gate):
    """Percent of one core to push DUR_S of audio through a Watcher."""
    w = watcher.Watcher(SR, gate=gate)
    t0 = time.perf_counter()
    for i in range(0, len(samples), 1600):
        w.push(samples[i:i + 1600])
    return (time.perf_counter() - t0) / DUR_S * 100.0


def tone(f0, amp):
    n = int(DUR_S * SR)
    t = np.arange(n, dtype=np.float64) / SR
    return (amp * np.sin(2.0 * math.pi * f0 * t)).astype(np.float32)


def main():
    if vad.model_path() is None:
        print(json.dumps({"error": "gate weights absent; run src.vad.fetch_model()"}))
        return
    gate = vad.SileroGate()
    rows = []

    for f0 in FUNDAMENTALS:
        for amp in LEVELS:
            sig = harmonic_stack(f0, amp)
            rows.append({"signal": f"harmonic_stack_{f0}Hz", "f0": f0,
                         "amp": amp, **gate_stats(sig, gate),
                         "cpu_gated_pct": round(cpu_percent(sig, gate), 2),
                         "cpu_ungated_pct": round(cpu_percent(sig, None), 2)})

    # The already-documented worst case, so this file is comparable to the
    # claim it is auditing rather than quietly using a different threat.
    for f0 in (150, 175, 200):
        sig = tone(f0, 0.05)
        rows.append({"signal": f"pure_tone_{f0}Hz", "f0": f0, "amp": 0.05,
                     **gate_stats(sig, gate),
                     "cpu_gated_pct": round(cpu_percent(sig, gate), 2),
                     "cpu_ungated_pct": round(cpu_percent(sig, None), 2)})

    # The direction that decides usability: the input the gate must not block.
    if SPEECH.is_file():
        speech = wavio.read_wav(str(SPEECH)).samples
        # Truncate to DUR_S. `cpu_percent` normalises by DUR_S, so pushing a
        # 90 s clip would understate the rate by 3x and silently flatter the
        # gate. The clip is longer than every other row here.
        speech = speech[:int(DUR_S * SR)]
        peak = max(abs(v) for v in speech) or 1.0
        for db in (-10, -20, -30, -40):
            a = 10 ** (db / 20.0)
            lv = np.asarray(speech, dtype=np.float32) / peak * a
            rows.append({"signal": "real_speech", "dbfs": db,
                         **gate_stats(lv, gate),
                         "cpu_gated_pct": round(cpu_percent(lv, gate), 2),
                         "cpu_ungated_pct": round(cpu_percent(lv, None), 2)})

    failing = [r for r in rows
               if r.get("f0") and (r["frac"] or 0) > 0.0]
    out = {
        "note": ("Does the silero gate hold against mains-frequency hum? Each "
                 "row is mean gate probability, fraction of frames clearing "
                 f"GATE_SPEECH={watcher.GATE_SPEECH}, and percent of one core "
                 "the watcher burns with and without the gate. A row with "
                 "frac>0 defeats the mitigation: YIN runs anyway."),
        "rows": rows,
        "summary": {
            "mains_fundamentals_rejected": sorted(
                {r["f0"] for r in rows
                 if r.get("f0") and r["f0"] in (50, 60, 100, 120)
                 and (r["frac"] or 0) == 0.0}),
            "fundamentals_that_clear_the_threshold": sorted(
                {r["f0"] for r in failing}),
            "max_cpu_saved_pct": round(
                max((r["cpu_ungated_pct"] - r["cpu_gated_pct"]) for r in rows), 2),
            "min_cpu_saved_pct": round(
                min((r["cpu_ungated_pct"] - r["cpu_gated_pct"]) for r in rows), 2),
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()