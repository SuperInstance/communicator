#!/usr/bin/env python
"""
watcher_cost.py -- what does the watcher actually cost, and what does that cost
depend on?

MOTIVATION. One number in this project is load-bearing and two independent
measurements disagreed with it by more than 3x:

    commit 8eb05f6 / src/vad.py / README.md   31.8% of a core   ("no gate")
    docs/research/DEVICE-GENERALISATION.md    39.3%  on 30 s of real speech
    docs/research/IN-THE-WILD-AUDIO.md      116.9%  on a 6 s 150 Hz stack

Those three were measured on three different fixtures at three different lengths
by three different harnesses, and none of the harnesses recorded the fixture and
length alongside the number. So the disagreement may not be a disagreement at
all: it may be three correct measurements of three different things.

THE HYPOTHESIS UNDER TEST. The watcher's per-second cost GROWS with the length
of a continuous signal, because on a signal with no pauses the retained buffer
is not trimmed at pause boundaries, so every push re-runs `energy.rms_series`
over a growing buffer -- which is O(buffer) per push and so O(n^2) in run
length if the buffer is never bounded.

"GROWTH" IS NOT ONE THING, AND CONFLATING THE TWO IS THE TRAP. A cost that is
superlinear in TOTAL TIME is not the same claim as a cost that is superlinear in
STREAMING RATE. A watcher with a bounded ramp -- buffer growing to a bound B,
trimmed, growing again -- has quadratic TOTAL cost over n >> B seconds only
until the ramp amortises, after which the streaming rate is constant. Both look
like "it grows" if you only plot percent-of-a-core against clip length. So this
experiment measures BOTH:

    total cost       -- CPU seconds against clip length (what a benchmark shows)
    streaming rate   -- CPU per second of audio AFTER the ramp (what a
                        deployment pays)

and the per-push trace is what separates them. If the per-push cost plateaus,
the growth is amortisation of a bounded ramp and the steady-state figure is the
defensible one. If the buffer itself never gets bounded, the ramp IS the
steady state and the growth is unbounded.

MEASUREMENT. `resource.getrusage(RUSAGE_SELF)` user+sys around the push loop,
normalised by the clip length. The watcher is pure stdlib and single-threaded,
so CPU-seconds/second-of-audio IS percent-of-one-core, and unlike wall clock it
cannot be flattered by an idle machine or inflated by another process. Wall
clock is recorded alongside every cell so the two can be compared rather than
one being assumed; a cpu/wall ratio well below 1.0 in the gated cells would mean
hidden parallelism, and it is reported rather than hidden.

THE CLOCK IS CONTAMINATED BY DEFAULT ON THIS BOX, AND THAT IS A FINDING. With
the thread environment untouched, the FIRST timed measurement in any fresh
process reports a cpu/wall ratio of 2.28 -- 128% of phantom CPU on a workload
that is one Python thread doing arithmetic. numpy's BLAS thread pool spin-waits
after import and its spinning is charged to whatever else the process is doing.
Wall clock shows 7.29%; rusage shows 16.6%. Every cell in this file therefore
sets `OMP/OPENBLAS/MKL_NUM_THREADS=1` BEFORE importing numpy, because a
percent-of-a-core figure that includes an idle thread pool burning CPU on
another core is not a percent of a core. This is also why WARMUP exists. A
harness that used `time.perf_counter()` alone would have quietly dodged the
problem and a harness that used rusage alone would have to solve it explicitly;
this one solves it explicitly and reports the ratio so the fix is checkable.

FIXTURES are the ones the disputed numbers were plausibly taken on, built with
the SAME generators the documents that produced them used:

    tone200        pure 200 Hz sine, the signal the growth claim was reported on
    hum120_005     12-harmonic 120 Hz stack, peak-normalised, amplitude 0.05
                   -- hum_sweep.py / device_generalisation.py's fixture
    testvad150     3-harmonic 150 Hz stack at amplitude 0.3, NOT peak
                   normalised -- tests/test_vad.py's fixture, and the one
                   IN-THE-WILD-AUDIO.md's 116.9% was measured on
    speech         examples/realaudio/alice01_excerpt.wav
    white          uniform noise at a fixed seed
    silence        exact zeros

THE CONTAINER IS PART OF THE FIXTURE, AND IT IS WORTH A FACTOR OF TWO. Every
signal here is handed to `push()` as a PYTHON LIST OF FLOATS, because that is
what `wavio.read_wav` returns and what a microphone callback yields. That is not
a cosmetic choice. `energy.frame_rms` is a pure-Python loop that calls
`math.isfinite(x)` once per sample, and `math.isfinite` on an `np.float32`
scalar pays a `__float__` conversion that a Python float does not:

    4 s of tone200, list of floats      39.7% of a core
    4 s of tone200, np.float32          94.2% of a core     2.37x
    30 s of speech, read_wav list       33.9% of a core
    30 s of speech, np.float32          69.2% of a core     2.04x

So the three disputed numbers now have a second, larger axis of variation than
length: `inthewild_probe.py` and `hum_sweep.py` build their fixtures with
numpy and push the array, while `device_generalisation.py` pushes what
`read_wav` returned, which is a list. The first harness pays ~2x. Reporting a
percent-of-a-core without saying which container was fed is reporting half the
measurement. `--containers` re-measures the same signals both ways so the
multiplier is in the results rather than in a comment.

WHAT THIS IS NOT. One machine (see RESULTS["machine"] for what it claims to be),
one chunk size, one gate build, python-level wall. It cannot tell you what a
Pi-class target does; see DEVICE-GENERALISATION.md, which is the document that
asks that question and answers it with a cache-size proxy. What it CAN do is
settle whether three quoted percentages are three measurements of one quantity,
which is a question about this repository's own bookkeeping.

    /home/proart/projects/glasses/.venv/bin/python experiments/watcher_cost.py
"""

from __future__ import annotations

import json
import math
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

# Before numpy, before onnxruntime, before anything that can start a thread
# pool. See the module docstring: unpinned, this inflates the first measurement
# by 128% on a single-threaded workload.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src import vad, wavio, watcher  # noqa: E402

SR = 16000
CHUNK = 1600               # 100 ms, the push granularity every disputed number used
LENGTHS = [1, 2, 4, 8, 16, 30, 60]
WARMUP = 1                 # untimed passes before the timed ones
REPEATS = 2                # timed passes per cell; median reported

REPO = Path(__file__).resolve().parents[1]
SPEECH = REPO / "examples/realaudio/alice01_excerpt.wav"
RESULTS = REPO / "experiments/watcher_cost_results.json"

_GATE = None


def fresh_gate():
    """One loaded ONNX session, reset between cells.

    silero carries RNN state across pushes, so a cell that inherited the
    previous cell's state would not be measuring the same thing. `reset()`
    clears the state and the carry buffer and keeps the graph loaded; building
    a session per cell costs ~1 s each and would dominate the short cells.
    """
    global _GATE
    if _GATE is None:
        _GATE = vad.SileroGate(SR)
    _GATE.reset()
    return _GATE


# -- signals ------------------------------------------------------------------

def tone200(amp: float = 0.5, n: int = 0) -> np.ndarray:
    """A pure 200 Hz sine. Continuous, periodic, never pauses."""
    t = np.arange(n or int(LENGTHS[-1] * SR)) / SR
    return (amp * np.sin(2.0 * math.pi * 200.0 * t)).astype(np.float32)


def harmonic_stack(f0: float, amp: float, n: int = 0, harms=range(1, 13),
                   normalise: bool = True) -> np.ndarray:
    """hum_sweep.py / inthewild_probe.py's generator, byte-for-byte in shape.

    `normalise=False` reproduces tests/test_vad.py's fixture, where `amp` is
    the fundamental's coefficient and the stack is NOT rescaled -- a fixture
    difference already documented as changing gate behaviour by 1.000 vs 0.000
    at the same nominal frequency (IN-THE-WILD-AUDIO.md section 7 item 1).
    """
    n = n or int(LENGTHS[-1] * SR)
    t = np.arange(n, dtype=np.float64) / SR
    out = np.zeros(n, dtype=np.float64)
    for k in harms:
        f = f0 * k
        if f >= SR / 2:
            break
        out += (amp / k) * np.sin(2.0 * math.pi * f * t)
    if not normalise:
        return out.astype(np.float32)
    peak = np.abs(out).max() or 1.0
    return (out * (amp / peak)).astype(np.float32)


def speech_signal(n: int = 0):
    """Real speech, exactly as device_generalisation.py read it: the Python
    list `read_wav` returns, truncated. NOT peak-normalised -- re-normalising
    would change which frames clear the energy bar and therefore which frames
    YIN is asked about."""
    a = wavio.read_wav(str(SPEECH)).samples
    return list(a[:n or int(LENGTHS[-1] * SR)])


def white_signal(n: int = 0) -> np.ndarray:
    rng = np.random.default_rng(20261018)
    return rng.uniform(-0.5, 0.5, n or int(LENGTHS[-1] * SR)).astype(np.float32)


def silence_signal(n: int = 0) -> np.ndarray:
    return np.zeros(n or int(LENGTHS[-1] * SR), dtype=np.float32)


def as_fed(sig: np.ndarray):
    """What actually reaches `push()`.

    A Python list of floats, which is what `wavio.read_wav` returns and what a
    microphone callback yields. See the module docstring: feeding the numpy
    array instead roughly doubles the measured cost, because `frame_rms` is a
    pure-Python loop that calls `math.isfinite` per sample.
    """
    return sig.tolist()


SIGNALS = {
    "tone200": lambda n: as_fed(tone200(0.5, n)),
    "hum120_005": lambda n: as_fed(harmonic_stack(120, 0.05, n)),
    "testvad150": lambda n: as_fed(harmonic_stack(150, 0.3, n, harms=(1, 2, 3),
                                                  normalise=False)),
    "speech": speech_signal,
    "white": lambda n: as_fed(white_signal(n)),
    "silence": lambda n: as_fed(silence_signal(n)),
}


# -- instrumentation ----------------------------------------------------------

class ProbeWatcher(watcher.Watcher):
    """A Watcher that reports on itself.

    The point of the experiment is the buffer, and the buffer is a private
    attribute of a class in `src/` that this experiment does not own and must
    not edit. Subclassing is the read-only way to observe it: `_trim` is
    overridden only to count, and every call is delegated upwards, so the code
    under measurement is byte-for-byte the shipped code.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.trim_calls = 0
        self.trace: list = []            # (elapsed_s, buf_len, cpu_s)

    def _trim(self, consumed_to_abs: float) -> None:
        self.trim_calls += 1
        super()._trim(consumed_to_abs)


def _cpu_now() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


# -- measurement --------------------------------------------------------------

def measure(samples, gated: bool, instrumented: bool = False) -> dict:
    """Percent of one core to push `samples` through a Watcher.

    `instrumented=True` adds a per-push trace (elapsed, buffer length, CPU
    delta). It is run as its own pass, never inside a timed one, so the trace
    cannot contaminate the percentages it is meant to explain.
    """
    gate = fresh_gate() if gated else None
    cls = ProbeWatcher if instrumented else watcher.Watcher
    w = cls(SR, gate=gate)

    c0, t0 = _cpu_now(), time.perf_counter()
    prev_c = c0
    for i in range(0, len(samples), CHUNK):
        w.push(samples[i:i + CHUNK])
        if instrumented:
            c = _cpu_now()
            w.trace.append(((i + CHUNK) / SR, len(w._buf), c - prev_c))
            prev_c = c
    cpu, wall = _cpu_now() - c0, time.perf_counter() - t0
    dur_s = len(samples) / SR

    out = {
        "cpu_pct": round(cpu / dur_s * 100.0, 2),
        "wall_pct": round(wall / dur_s * 100.0, 2),
        "cpu_s": round(cpu, 3),
        "threads_ratio": round(cpu / wall, 3) if wall > 0 else None,
        "pushes": len(range(0, len(samples), CHUNK)),
        "events": len(w.emitted),
        "utterances": sum(1 for e in w.emitted if e.kind == watcher.UTTERANCE),
    }
    if instrumented:
        lens = [t[1] for t in w.trace]
        out.update({
            "trim_calls": w.trim_calls,
            "buf_len_first": lens[0] if lens else None,
            "buf_len_last": lens[-1] if lens else None,
            "buf_len_max": max(lens) if lens else None,
            "buf_len_mean": round(statistics.mean(lens), 1) if lens else None,
            "buf_len_p50": statistics.median(lens) if lens else None,
            "trace": [[round(a, 4), b, round(c, 6)] for a, b, c in w.trace],
        })
    return out


def cell(samples, gated: bool) -> dict:
    """One (fixture, length, gate) cell: warmup, then timed passes, median."""
    runs = [measure(samples, gated) for _ in range(WARMUP + REPEATS)]
    timed = runs[WARMUP:]
    med = {}
    for k in ("cpu_pct", "wall_pct", "cpu_s", "threads_ratio"):
        vals = [r[k] for r in timed if r[k] is not None]
        if vals:
            med[k] = round(statistics.median(vals), 3)
    out = dict(med)
    out["cpu_pct_runs"] = [r["cpu_pct"] for r in timed]
    out["events"] = timed[-1]["events"]
    out["utterances"] = timed[-1]["utterances"]
    return out


# -- growth shape -------------------------------------------------------------

def growth_exponent(points) -> float | None:
    """Least-squares slope of log(cost) on log(length). 1.0 = linear,
    2.0 = quadratic. `points` is [(length_s, cpu_pct)]."""
    pts = [(math.log(L), math.log(c)) for L, c in points if L >= 1 and c > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    den = n * sxx - sx * sx
    if den == 0:
        return None
    return round((n * sxy - sx * sy) / den, 3)


def band_rate(trace, lo_s: float, hi_s: float) -> float | None:
    """CPU per second of audio over the band [lo_s, hi_s) of stream time.

    This is the quantity that separates "the total cost of an N-second clip
    grows with N" -- which is trivially true and true of any per-second cost,
    because the clip got longer -- from "the streaming rate grows with N",
    which is the claim that matters for a watcher. A bounded ramp amortises:
    the 0-2 s band is expensive, the steady band is not, and the headline
    percentage converges to the steady rate as the clip lengthens.
    """
    rows = [t for t in trace if lo_s <= t[0] < hi_s]
    if len(rows) < 2:
        return None
    cpu = sum(t[2] for t in rows)
    span = rows[-1][0] - rows[0][0]
    return round(cpu / span * 100.0, 2) if span > 0 else None


def container_ab(durs, gate_ok: bool) -> dict:
    """Same signal, same length, two containers. The factor-of-two axis.

    `hum_sweep.py` and `inthewild_probe.py` build fixtures with numpy and push
    the array; `device_generalisation.py` pushes what `read_wav` returned, which
    is a list. Nothing in either file says so, and the two disagree by ~2x.
    """
    out = {}
    for name in SIGNALS:
        rows = []
        for dur in durs:
            raw = SIGNALS[name](int(round(dur * SR)))
            as_np = np.asarray(raw, dtype=np.float32)
            lst = cell(list(raw), gated=False)
            npy = cell(as_np, gated=False)
            rows.append({"dur_s": dur, "list_pct": lst["cpu_pct"],
                         "numpy_pct": npy["cpu_pct"],
                         "ratio": round(npy["cpu_pct"] / lst["cpu_pct"], 2)
                         if lst["cpu_pct"] else None})
            print(f"    {name:<12} {dur:>5.2f}s  list {lst['cpu_pct']:>8.2f}%  "
                  f"numpy {npy['cpu_pct']:>8.2f}%  "
                  f"ratio {rows[-1]['ratio']}")
        out[name] = rows
    return out


def target_search(lengths, gate_ok: bool) -> dict:
    """Fine short-length sweep: which (fixture, length) lands on a disputed number?

    The headline claim "31.8% of a core" was never recorded against a fixture or
    a length. Rather than guess which of the three numbers is a typo, this asks
    a falsifiable question of the data: is there a (fixture, length) in this
    repo's own signal set whose ungated cost is 31.8%? Sweeping the short end
    densely is where the answer can be, because the coarse sweep shows cost
    rising steeply between 1 s and 4 s.

    For a continuous signal the interesting lengths are near the 2.0 s
    MAX_UTTERANCE_S checkpoint, because that is where one YIN window lands
    inside the clip or does not -- a cliff, not a slope.
    """
    out = {}
    for name in SIGNALS:
        rows = []
        for dur in lengths:
            samples = SIGNALS[name](int(round(dur * SR)))
            c = cell(samples, gated=False)
            g = cell(samples, gated=True) if gate_ok else None
            rows.append({"dur_s": round(dur, 3), "ungated": c["cpu_pct"],
                         "gated": g["cpu_pct"] if g else None})
            print(f"    {name:<12} {dur:>5.2f}s  ungated {c['cpu_pct']:>8.2f}%"
                  + (f"  gated {g['cpu_pct']:>8.2f}%" if g else ""))
        out[name] = rows
    return out


def machine_info() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        info["cpu"] = "unknown"
    try:
        import onnxruntime
        info["onnxruntime"] = onnxruntime.__version__
    except Exception:                                          # noqa: BLE001
        info["onnxruntime"] = None
    info["numpy"] = np.__version__
    return info


def main() -> None:
    quick = "--quick" in sys.argv
    gate_ok = vad.model_path() is not None

    # `--finish` completes a partially-written results file: the grid is
    # already on disk, so it re-measures only the mechanism trace and derives
    # the growth exponents from what is saved. Exists because the grid writes
    # itself after every fixture, so a run stopped before the end still leaves
    # usable data -- this picks it back up instead of re-running an hour.
    if "--finish" in sys.argv:
        results = json.loads(RESULTS.read_text())
        lengths = results["lengths_s"]
        names = sorted({k.rsplit("_", 1)[0] for k in results["grid"]})
        trace_cells = [("tone200", 30, False), ("tone200", 30, True),
                       ("tone200", 60, False), ("tone200", 60, True),
                       ("hum120_005", 30, False), ("hum120_005", 30, True),
                       ("speech", 30, False), ("speech", 30, True),
                       ("silence", 30, False)]
        for name, dur, gated in trace_cells:
            samples = SIGNALS[name](int(dur * SR))
            tr = measure(samples, gated, instrumented=True)
            mode = "gated" if gated else "ungated"
            results["mechanism"][f"{name}_{dur:g}s_{mode}"] = {
                "trim_calls": tr["trim_calls"], "buf_len_max": tr["buf_len_max"],
                "buf_len_mean": tr["buf_len_mean"],
                "buf_len_last": tr["buf_len_last"], "cpu_pct": tr["cpu_pct"],
                "rate_pct_0_to_2s": band_rate(tr["trace"], 0.0, 2.0),
                "rate_pct_2_to_4s": band_rate(tr["trace"], 2.0, 4.0),
                "rate_pct_after_4s": band_rate(tr["trace"], 4.0, float("inf")),
                "trace": tr["trace"],
            }
            m = results["mechanism"][f"{name}_{dur:g}s_{mode}"]
            print(f"  {name:<12} {dur:>3g}s {mode:<8} buf max "
                  f"{m['buf_len_max']:>7} mean {m['buf_len_mean']:>8} "
                  f"trims {m['trim_calls']:>4} bands 0-2s "
                  f"{m['rate_pct_0_to_2s']}% 2-4s {m['rate_pct_2_to_4s']}% "
                  f">4s {m['rate_pct_after_4s']}%")
        growth = {}
        for name in names:
            row = {
                "ungated_all": growth_exponent(
                    [(L, results["grid"][f"{name}_{L:g}s"]["ungated"]["cpu_pct"])
                     for L in lengths]),
                "ungated_ge4s": growth_exponent(
                    [(L, results["grid"][f"{name}_{L:g}s"]["ungated"]["cpu_pct"])
                     for L in lengths if L >= 4]),
            }
            if gate_ok and "gated" in results["grid"][f"{name}_{lengths[0]:g}s"]:
                row["gated_all"] = growth_exponent(
                    [(L, results["grid"][f"{name}_{L:g}s"]["gated"]["cpu_pct"])
                     for L in lengths])
                row["gated_ge4s"] = growth_exponent(
                    [(L, results["grid"][f"{name}_{L:g}s"]["gated"]["cpu_pct"])
                     for L in lengths if L >= 4])
            growth[name] = row
            print(f"  {name:<12} ungated {row['ungated_all']} (all) / "
                  f"{row['ungated_ge4s']} (>=4s)  gated "
                  f"{row.get('gated_all')} / {row.get('gated_ge4s')}")
        results["growth_exponents"] = growth
        RESULTS.write_text(json.dumps(results, indent=1) + "\n")
        print(f"\nwrote {RESULTS}")
        return

    # `--target-only` reuses the grid already on disk. The target search is a
    # separate question from the length sweep, and re-running a 50-minute grid
    # to answer it would be waste; if no results file exists it says so rather
    # than silently starting from an empty dict.
    if "--target-only" in sys.argv:
        if not RESULTS.exists():
            print(f"no {RESULTS}; run the full sweep first")
            return
        results = json.loads(RESULTS.read_text())
        fine = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.1, 2.25,
                2.5, 3.0, 3.5, 4.0]
        print("-- target search: dense short-length sweep --")
        results["target_search"] = target_search(fine, gate_ok)
        targets = {"31.8": 31.8, "39.1": 39.1, "116.9": 116.9}
        print("\n  nearest cell to each disputed figure (ungated):")
        for label, want in targets.items():
            best = None
            for name, rows in results["target_search"].items():
                for r in rows:
                    d = abs(r["ungated"] - want)
                    if best is None or d < best[0]:
                        best = (round(d, 2), name, r["dur_s"], r["ungated"])
            print(f"    {label:>6}%  ->  {best[1]} at {best[2]}s = {best[3]}%"
                  f"   (off by {best[0]} points)")
        RESULTS.write_text(json.dumps(results, indent=1) + "\n")
        print(f"\nwrote {RESULTS}")
        return

    lengths = [1, 2, 4, 8, 16, 30, 60] if "--lean" not in sys.argv else \
        [1, 2, 4, 8, 16, 30]
    names = ["tone200", "hum120_005", "testvad150", "speech", "white", "silence"]
    if quick:
        names = ["tone200", "speech", "hum120_005"]
    # The per-push trace costs a whole extra pass per cell, so it runs on the
    # fixtures that answer the growth question, not on every signal.
    trace_lengths = [2, 8] if quick else [2, 8, 30, 60]
    trace_names = ["tone200", "testvad150", "speech", "hum120_005"]
    if quick:
        trace_names = ["tone200"]

    results = {
        "question": "is the watcher's per-second cost a function of run length, "
                    "and which of 31.8 / 39.3 / 116.9 % is defensible?",
        "method": {
            "clock": "resource.getrusage(RUSAGE_SELF) utime+stime around the "
                     "push loop, divided by clip length; wall clock recorded "
                     "alongside in every cell",
            "thread_env": {v: os.environ.get(v) for v in
                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS")},
            "sr": SR, "chunk": CHUNK, "chunk_s": CHUNK / SR,
            "warmup_passes": WARMUP, "timed_passes": REPEATS,
            "statistic": "median of timed passes",
            "gate": "src.vad.SileroGate, one loaded session reset per cell",
        },
        "machine": machine_info(),
        "lengths_s": lengths,
        "grid": {},
        "mechanism": {},
    }

    print(f"lengths {lengths}s   fixtures {names}   gate available: {gate_ok}")

    for name in names:
        for dur in lengths:
            n = int(dur * SR)
            samples = SIGNALS[name](n)
            key = f"{name}_{dur:g}s"
            entry = {"ungated": cell(samples, gated=False)}
            if gate_ok:
                entry["gated"] = cell(samples, gated=True)
            results["grid"][key] = entry
            u = entry["ungated"]["cpu_pct"]
            g = entry.get("gated", {}).get("cpu_pct")
            print(f"  {key:<18} ungated {u:>8.2f}%   gated "
                  f"{('%8.2f%%' % g) if g is not None else '     n/a'}")
        # Written after every fixture, not only at the end. A full sweep is
        # ~50 minutes of mostly-CPU-bound work; a partial sweep with a JSON on
        # disk beats a complete one that a timeout deletes.
        RESULTS.write_text(json.dumps(results, indent=1) + "\n")

    print("\n-- mechanism: buffer length and per-push CPU --")
    for name in trace_names:
        for dur in trace_lengths:
            n = int(dur * SR)
            samples = SIGNALS[name](n)
            for gated in (False, True) if gate_ok else (False,):
                tr = measure(samples, gated, instrumented=True)
                mode = "gated" if gated else "ungated"
                results["mechanism"][f"{name}_{dur:g}s_{mode}"] = {
                    "trim_calls": tr["trim_calls"],
                    "buf_len_max": tr["buf_len_max"],
                    "buf_len_mean": tr["buf_len_mean"],
                    "buf_len_last": tr["buf_len_last"],
                    "cpu_pct": tr["cpu_pct"],
                    "rate_pct_0_to_2s": band_rate(tr["trace"], 0.0, 2.0),
                    "rate_pct_2_to_4s": band_rate(tr["trace"], 2.0, 4.0),
                    "rate_pct_after_4s": band_rate(tr["trace"], 4.0, float("inf")),
                    "trace": tr["trace"],
                }
            m = results["mechanism"][f"{name}_{dur:g}s_ungated"]
            g2 = results["mechanism"].get(f"{name}_{dur:g}s_gated")
            print(f"  {name:<12} {dur:>3g}s  buf max {m['buf_len_max']:>7} "
                  f"mean {m['buf_len_mean']:>8}  trims {m['trim_calls']:>4}  "
                  f"bands 0-2s {m['rate_pct_0_to_2s']}%  2-4s "
                  f"{m['rate_pct_2_to_4s']}%  >4s {m['rate_pct_after_4s']}%"
                  + (f"   | gated >4s {g2['rate_pct_after_4s']}%" if g2 else ""))

    print("\n-- growth shape (exponent of cpu_pct vs length; 1.0 linear, 2.0 quadratic) --")
    growth = {}
    for name in names:
        row = {
            "ungated_all": growth_exponent(
                [(L, results["grid"][f"{name}_{L:g}s"]["ungated"]["cpu_pct"])
                 for L in lengths]),
            "ungated_ge4s": growth_exponent(
                [(L, results["grid"][f"{name}_{L:g}s"]["ungated"]["cpu_pct"])
                 for L in lengths if L >= 4]),
        }
        if gate_ok:
            row["gated_all"] = growth_exponent(
                [(L, results["grid"][f"{name}_{L:g}s"]["gated"]["cpu_pct"])
                 for L in lengths])
            row["gated_ge4s"] = growth_exponent(
                [(L, results["grid"][f"{name}_{L:g}s"]["gated"]["cpu_pct"])
                 for L in lengths if L >= 4])
        growth[name] = row
        print(f"  {name:<12} ungated {row['ungated_all']} (all) / "
              f"{row['ungated_ge4s']} (>=4s)   gated "
              f"{row.get('gated_all')} / {row.get('gated_ge4s')}")
    results["growth_exponents"] = growth

    if "--target" in sys.argv:
        # Dense short-length sweep. Cheap: every cell here is under 2 s of
        # audio, so the whole sweep costs seconds, not the tens of minutes the
        # main grid costs.
        fine = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.1, 2.25,
                2.5, 3.0, 3.5, 4.0]
        print("\n-- target search: dense short-length sweep --")
        results["target_search"] = target_search(fine, gate_ok)
        targets = {"31.8": 31.8, "39.1": 39.1, "116.9": 116.9}
        print("\n  nearest cell to each disputed figure (ungated, list-fed):")
        for label, want in targets.items():
            best = None
            for name, rows in results["target_search"].items():
                for r in rows:
                    d = abs(r["ungated"] - want)
                    if best is None or d < best[0]:
                        best = (round(d, 2), name, r["dur_s"], r["ungated"])
            print(f"    {label:>6}%  ->  {best[1]} at {best[2]}s = {best[3]}%"
                  f"   (off by {best[0]} points)")

    if "--containers" in sys.argv or "--target" in sys.argv:
        print("\n-- container A/B: python list vs numpy float32 --")
        results["container_ab"] = container_ab([4, 30], gate_ok)

    RESULTS.write_text(json.dumps(results, indent=1) + "\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()