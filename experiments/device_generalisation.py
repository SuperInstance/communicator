#!/usr/bin/env python
"""
device_generalisation.py -- does the gate's advantage survive cache pressure?

MOTIVATION. `docs/research/ADOPTION-PLAN.md` section 8 names the thing that
would most change our minds about commit `8eb05f6`: every number behind the
silero gate -- 31.8% of a core down to 2.4% -- comes from ONE machine, an AMD
Ryzen AI 9 HX 370 with a 16 MB L3, inside WSL2. The target is a Pi-class
peripheral.

The specific worry is not "is it slower elsewhere" but a falsifiable one about
the SHAPE of the cost. The gate is a 2.2 MB ONNX graph; ungated, the watcher's
dominant cost is `pitch.PitchTracker` (YIN) over a few hundred KB of buffers. A
2.2 MB working set is nearly free in a 16 MB L3 and not nearly free in a Pi's
512 KB-1 MB L2. So the gate's cost could scale WORSE than YIN's as the working
set stops fitting in cache, and if it does, the 15x advantage shrinks or inverts
on the actual device. That has never been tested.

WHAT IS MEASURED. The same fixed audio, four ways:

    regime    gate                 what it means
    warm      SileroGate()         model and buffers resident (current claims)
    cold      SileroGate()         L1/L2/L3 flushed before every 100 ms chunk
    warm      gate=None            YIN always runs
    cold      gate=None            same, with cache pressure

and for each, the RATIO ungated/gated, which is the portable quantity. The
absolute percentages are machine-specific; the ratio is the architectural
decision. The hypothesis is falsified if the cold ratio falls materially below
the warm ratio -- that is, if the gate degrades faster under cache pressure
than YIN does.

THE FLUSH IS VERIFIED, NOT ASSUMED. "Cold cache" is a label, and an unverified
one is worth nothing. A deliberately L1-resident control loop is timed under the
identical flush and reported alongside: if the control does not slow down, the
flush is not doing anything and the cold column means nothing either. The
control ratio is reported in the output so the label can be checked.

WHAT THIS IS NOT -- the honest limit, stated up front.

This is a CACHE-SIZE PROXY for device generalisation. It is not a held-out
device. A Zen 5 L3 miss and a DRAM access under WSL2 are not an ARM Cortex-A72
with a 1 MB L2 and much slower memory; the proxy perturbs one axis (working set
vs cache) and leaves the others (core count, memory latency, ISA, ORT kernels,
thermal limits) untouched. A result here can FALSIFY the claim -- if the ratio
degrades under cache pressure, the claim does not survive. It cannot CONFIRM
generalisation to a device nobody ran it on.

The one thing already established is not redone: the gate is single-threaded
(`intra_op_num_threads = 1` in `vad.py`), so 1.00 cores busy, and the claim
carries no hidden parallelism to lose.

    /home/proart/projects/glasses/.venv/bin/python experiments/device_generalisation.py
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import vad, wavio, watcher  # noqa: E402

import numpy as np  # noqa: E402

SR = 16000
DUR_S = 30.0
CHUNK = 1600               # 100 ms, the same push granularity as hum_sweep.py
REPEATS = 5                # timed passes per cell; 1 untimed warmup before
WARMUP = 1

FLUSH_MB = 256             # ~16x this machine's 16 MB L3
CONTROL_BYTES = 32 * 1024  # deliberately L1-resident (L1d is 48 KB)

REPO = Path(__file__).resolve().parents[1]
SPEECH = REPO / "examples/realaudio/alice01_excerpt.wav"
RESULTS = REPO / "experiments/device_generalisation_results.json"

# Read-only sweep. A write sweep would also evict, but it leaves 256 MB of dirty
# lines being written back while the next timed region runs, which is noise this
# experiment would then be measuring instead of cache misses.
_FLUSH = np.ones((FLUSH_MB * 1024 * 1024) // 8, dtype=np.float64)


def flush_caches() -> None:
    """Evict L1/L2/L3 by streaming a buffer far larger than the last-level cache."""
    float(_FLUSH.sum())


# -- signals -----------------------------------------------------------------

def harmonic_stack(f0, amp, n_harm=12, rolloff=1.0):
    """The hum family the gate was measured against (hum_sweep.py's generator)."""
    n = int(DUR_S * SR)
    t = np.arange(n, dtype=np.float64) / SR
    out = np.zeros(n, dtype=np.float64)
    for k in range(1, n_harm + 1):
        f = f0 * k
        if f >= SR / 2:
            break
        out += (amp / (k ** rolloff)) * np.sin(2.0 * math.pi * f * t)
    peak = np.abs(out).max() or 1.0
    return (out * (amp / peak)).astype(np.float32).tolist()


def speech_signal():
    a = wavio.read_wav(str(SPEECH)).samples
    # `cpu_percent` normalises by DUR_S, so a longer clip would understate the
    # rate by its length ratio and silently flatter whichever side is measured.
    # hum_sweep.py truncates for the same reason.
    return a[:int(DUR_S * SR)]


# -- timing primitives --------------------------------------------------------

def summarise(values):
    return {
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "runs": [round(v, 3) for v in values],
    }


def measure_watcher(samples, gate_obj, flush, repeats=REPEATS):
    """Percent of one core to push DUR_S through a Watcher.

    The flush is timed SEPARATELY and excluded, and the timed region contains
    only the push -- otherwise "cold" would be measuring the flush.
    """
    rates, flush_s = [], []
    for r in range(repeats + WARMUP):
        w = watcher.Watcher(SR, gate=gate_obj)
        elapsed = fl = 0.0
        for i in range(0, len(samples), CHUNK):
            chunk = samples[i:i + CHUNK]
            if flush:
                f0 = time.perf_counter()
                flush_caches()
                fl += time.perf_counter() - f0
            t0 = time.perf_counter()
            w.push(chunk)
            elapsed += time.perf_counter() - t0
        if r < WARMUP:
            continue
        rates.append(elapsed / DUR_S * 100.0)
        flush_s.append(fl)
    out = summarise(rates)
    out["flush_s_per_run"] = round(statistics.median(flush_s), 2)
    return out


def measure_gate_alone(samples, gate_obj, flush, repeats=REPEATS):
    """The gate's own cost, same protocol, no watcher.

    This is the attribution half: if the gated column degrades under cache
    pressure, the question is whether it is the gate re-reading 2.2 MB of
    weights from DRAM on every chunk.
    """
    rates, flush_s = [], []
    for r in range(repeats + WARMUP):
        gate_obj.reset()
        elapsed = fl = 0.0
        for i in range(0, len(samples), CHUNK):
            chunk = samples[i:i + CHUNK]
            if flush:
                f0 = time.perf_counter()
                flush_caches()
                fl += time.perf_counter() - f0
            t0 = time.perf_counter()
            gate_obj.push(chunk)
            elapsed += time.perf_counter() - t0
        if r < WARMUP:
            continue
        rates.append(elapsed / DUR_S * 100.0)
        flush_s.append(fl)
    out = summarise(rates)
    out["flush_s_per_run"] = round(statistics.median(flush_s), 2)
    return out


def control_loop(flush, repeats=60):
    """A loop whose working set is deliberately L1-resident.

    Warm it is served from L1; under the same flush every line must come from
    DRAM. If these two are not measurably different, the flush is not working
    and every "cold" number below is just a warm number with a slower harness.
    """
    buf = np.ones(CONTROL_BYTES // 8, dtype=np.float64)
    times = []
    for _ in range(repeats):
        if flush:
            flush_caches()
        t0 = time.perf_counter()
        s = float(buf.sum())
        t1 = time.perf_counter()
        times.append(t1 - t0)
        del s
    return summarise([t * 1e6 for t in times])   # microseconds per pass


def gate_stats(samples, gate_obj):
    probs = []
    for i in range(0, len(samples), CHUNK):
        probs.extend(gate_obj.push(samples[i:i + CHUNK]))
    if not probs:
        return {"mean": None, "frac": None}
    return {"mean": round(sum(probs) / len(probs), 4),
            "frac": round(sum(1 for p in probs if p >= watcher.GATE_SPEECH)
                          / len(probs), 4)}


# -- cross-architecture probe -------------------------------------------------

def cross_arch_probe():
    """What is actually installed here. Reported, never assumed."""
    out = {}
    for b in ("qemu-aarch64", "qemu-aarch64-static", "qemu-user-static",
              "qemu-arm", "qemu-riscv64", "qemu-x86_64"):
        p = subprocess.run(["which", b], capture_output=True, text=True)
        out[b] = p.stdout.strip() or None
    try:
        d = subprocess.run(["dpkg", "-l"], capture_output=True, text=True,
                           timeout=30).stdout
        out["qemu_packages_installed"] = [
            l.split()[1] for l in d.splitlines() if " qemu" in l][:10] or None
    except Exception as exc:                                  # noqa: BLE001
        out["qemu_packages_installed"] = f"probe failed: {exc}"
    try:
        reg = Path("/proc/sys/fs/binfmt_misc").glob("*")
        out["binfmt_misc_handlers"] = sorted(p.name for p in reg) or None
    except Exception as exc:                                  # noqa: BLE001
        out["binfmt_misc_handlers"] = f"probe failed: {exc}"
    try:
        dk = subprocess.run(["docker", "info"], capture_output=True,
                            text=True, timeout=30)
        out["docker_cli_present"] = True
        out["docker_usable"] = dk.returncode == 0
        out["docker_detail"] = (dk.stderr or dk.stdout).strip()[:400]
    except FileNotFoundError:
        out["docker_cli_present"] = False
        out["docker_usable"] = False
    except Exception as exc:                                  # noqa: BLE001
        out["docker_cli_present"] = True
        out["docker_usable"] = False
        out["docker_detail"] = f"probe failed: {exc}"
    return out


def main():
    started = time.time()
    if vad.model_path() is None:
        print(json.dumps({"error": "gate weights absent; run src.vad.fetch_model()"}))
        return

    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    model = next((l.split(":", 1)[1].strip() for l in lscpu.splitlines()
                  if l.startswith("Model name")), "unknown")
    l3 = next((l.split(":", 1)[1].strip() for l in lscpu.splitlines()
               if l.startswith("L3 cache")), "unknown")

    # -- is the flush real? ---------------------------------------------------
    print("verifying the cache flush ...", file=sys.stderr)
    ctl_warm = control_loop(flush=False)
    ctl_cold = control_loop(flush=True)
    control_ratio = round(ctl_cold["median"] / ctl_warm["median"], 3)

    signals = {
        "speech_alice01": speech_signal(),
        "hum_120_harmonic_0.05": harmonic_stack(120, 0.05),
    }

    cells, gated_alone = {}, {}
    for name, samples in signals.items():
        g = vad.SileroGate()
        stats = gate_stats(samples, g)          # consumes state; reset below
        print(f"  {name}: gate mean={stats['mean']} frac={stats['frac']}",
              file=sys.stderr)
        cells[name] = {
            "gate_stats": stats,
            "warm_gated": measure_watcher(samples, g, flush=False),
            "cold_gated": measure_watcher(samples, g, flush=True),
            "warm_ungated": measure_watcher(samples, None, flush=False),
            "cold_ungated": measure_watcher(samples, None, flush=True),
        }
        print(f"  {name}: main cells done", file=sys.stderr)
        g.reset()
        gated_alone[name] = {
            "warm": measure_gate_alone(samples, g, flush=False),
            "cold": measure_gate_alone(samples, g, flush=True),
        }
        print(f"  {name}: isolated gate done", file=sys.stderr)

    for name in signals:
        c = cells[name]
        c["ratio_warm"] = round(c["warm_ungated"]["median"]
                                / c["warm_gated"]["median"], 2)
        c["ratio_cold"] = round(c["cold_ungated"]["median"]
                                / c["cold_gated"]["median"], 2)
        # The hypothesis in one number: the gate's cache-pressure penalty
        # relative to YIN's. >1 means the gate degrades faster.
        c["gated_inflation"] = round(c["cold_gated"]["median"]
                                     / c["warm_gated"]["median"], 3)
        c["ungated_inflation"] = round(c["cold_ungated"]["median"]
                                       / c["warm_ungated"]["median"], 3)
        c["relative_cache_penalty"] = round(c["gated_inflation"]
                                            / c["ungated_inflation"], 3)
        c["ratio_degrades"] = bool(c["ratio_cold"] < c["ratio_warm"] * 0.9)

    out = {
        "note": ("Does the silero gate's advantage over the ungated watcher "
                 "survive cache pressure? Percent of one core to push 30 s of "
                 "audio through a Watcher, warm (no flush) and cold (L1/L2/L3 "
                 "flushed before every 100 ms chunk, flush time excluded). "
                 "The RATIO ungated/gated is the portable quantity; the "
                 "absolute percentages are machine-specific."),
        "machine": {"cpu": model, "l3": l3, "flush_buffer_mb": FLUSH_MB,
                    "threads": "gate pinned to 1 (vad.py intra_op_num_threads=1)"},
        "protocol": {"audio_s": DUR_S, "chunk_samples": CHUNK,
                     "repeats": REPEATS, "warmup_passes": WARMUP,
                     "gate_threshold": watcher.GATE_SPEECH},
        "flush_verification": {
            "claim": ("A deliberately L1-resident control loop must measurably "
                      "slow under the same flush, or 'cold' is an unverified "
                      "label and the cold column means nothing."),
            "control_bytes": CONTROL_BYTES,
            "control_warm_us": ctl_warm,
            "control_cold_us": ctl_cold,
            "control_slowdown_x": control_ratio,
            "flush_is_real": control_ratio > 1.10,
        },
        "signals": cells,
        "gate_alone": gated_alone,
        "cross_architecture": cross_arch_probe(),
        "limits": [
            "Cache-size proxy, NOT a held-out device. Perturbs working set vs "
            "cache only; leaves core count, memory latency, ISA, ORT kernels "
            "and thermal behaviour untouched.",
            "Can falsify the 15x claim under cache pressure; cannot confirm "
            "generalisation to a device nobody ran it on.",
            "Absolute percent-of-a-core figures are specific to a Ryzen AI 9 "
            "HX 370 under WSL2 and are not portable; the ratio is.",
            "The flush evicts the watcher's own buffers as well as the gate's "
            "weights, so it is a harsher regime than a device that only has a "
            "small model competing for cache.",
        ],
        "elapsed_s": round(time.time() - started, 1),
    }
    RESULTS.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()