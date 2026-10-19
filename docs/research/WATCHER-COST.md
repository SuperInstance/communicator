# WATCHER COST — what the number is, and why there isn't one

**Experiment:** `experiments/watcher_cost.py`
**Data:** `experiments/watcher_cost_results.json`
**Date:** 2026-10-18 · AMD Ryzen AI 9 HX 370, WSL2 · Python 3.14.4 · numpy
2.5.4 · onnxruntime 1.31.0 · `getrusage` clock, threads pinned

---

## The dispute

One number is load-bearing in this project. Commit `8eb05f6` put a
voice-activity gate in front of the watcher and justified it with a saving
expressed in percent-of-a-core. Three figures are now in circulation for "the
ungated watcher", and no two agree:

| figure | source | fixture | length | container | clock |
|---|---|---|---|---|---|
| **31.8%** | `8eb05f6` subject, `src/vad.py:41`, `README.md:109` | *not recorded* | *not recorded* | *not recorded* | *not recorded* |
| **39.3%** | `DEVICE-GENERALISATION.md` §Signal 2 | `alice01_excerpt.wav` | 30 s | Python list | wall clock |
| **116.9%** | `IN-THE-WILD-AUDIO.md` §7.4 | `test_vad` 150 Hz stack | 6 s | numpy array | wall clock |

**There is no single number, and 31.8% is the only one of the three that was
never a measurement.** No experiment script in this repository has ever
produced it: `git log --all -S"39.13" -- experiments/` is empty, and across
all history that string appears only in prose — `src/vad.py`,
`src/watcher.py`, `tests/test_vad.py`, and three research docs. Worse, the
commit that introduced it recorded `39.13 ms per 100 ms` in the same breath,
and 39.13 ms per 100 ms **is 39.13%**. Two labels for one measurement,
disagreeing by 23%, with no fixture and no length attached to either. That is
the whole origin of the discrepancy: not three measurements, but one
unrecorded measurement quoted three ways.

---

## Method

`resource.getrusage(RUSAGE_SELF)` (`utime + stime`) around the push loop,
divided by clip length. The watcher is pure stdlib and single-threaded, so
CPU-seconds per second-of-audio **is** percent-of-one-core, and unlike wall
clock it is immune to a busy machine. Wall clock is recorded in every cell:
the largest cpu/wall ratio across all 72 cells is **1.003**, so the
single-threaded claim is checked rather than asserted.

Two harnesses in this repo would have reported something different, and both
would have been wrong:

- **Threads must be pinned first.** With the thread environment untouched, the
  *first* timed measurement in any fresh process reports a cpu/wall ratio of
  **2.28** — 128% of phantom CPU on a workload that is one Python thread doing
  arithmetic. numpy's BLAS pool spin-waits after import and its spinning is
  charged to whatever else the process is doing. This file sets
  `OMP/OPENBLAS/MKL_NUM_THREADS=1` before importing numpy.
- **The sample container is part of the fixture, and worth ~2×.**
  `energy.frame_rms` is a pure-Python loop calling `math.isfinite(x)` once per
  sample, and `math.isfinite` on an `np.float32` pays a `__float__` conversion
  a Python float does not:

  | signal | Python list of floats | numpy `float32` | ratio |
  |---|---:|---:|---:|
  | 4 s of 200 Hz tone | **39.7%** | 94.2% | 2.37× |
  | 30 s of real speech | **33.9%** | 69.2% | 2.04× |

  `wavio.read_wav` returns a **list**, so the list column is the shipped path
  and is what this file reports. But `hum_sweep.py` and `inthewild_probe.py`
  build fixtures with numpy and push the array — which is the 116.9% row.
  Neither file says which it did. Everything below is list-fed unless stated.

Fixtures use the generators of the documents that produced the disputed
numbers: `hum120_005` is `hum_sweep.py`'s 12-harmonic stack at 0.05;
`testvad150` is `tests/test_vad.py`'s 3-harmonic stack at 0.3, unnormalised;
`speech` is `alice01_excerpt.wav` unnormalised, as `device_generalisation.py`
reads it. Protocol: 100 ms pushes, 1 untimed warmup, 2 timed passes, median.
Run-to-run spread is tight (30 s speech: 29.87 / 30.02).

---

## The answer

**It is not quadratic.** Two separate things were being called "growth":

1. **A bounded startup ramp.** A 1–2 s clip never pays the 2 s checkpoint, so
   its percentage looks low. That is amortisation.
2. **A periodic cost that never stops.** Beyond ~4 s the per-second cost is
   *flat*, not rising.

Least-squares slope of log(cost) on log(length) — 1.0 is linear, 2.0 is
quadratic:

| fixture | exponent, all lengths | exponent, ≥ 4 s | verdict |
|---|---:|---:|---|
| `tone200` | 1.153 | **0.010** | flat beyond the ramp |
| `hum120_005` | 1.169 | **0.024** | flat |
| `testvad150` | 1.147 | **0.050** | flat |
| `white` | 1.193 | **0.025** | flat |
| `speech` | 0.575 | 0.100 | near-flat |
| `silence` | 0.314 | 0.090 | near-flat |

The full-range exponent of ~1.15 is what a bounded ramp looks like when you
fit a straight line through it; it is not a growth law. Beyond 4 s the
exponents are indistinguishable from zero. **Doubling the clip from 30 s to
60 s changes nothing** (`tone200` steady-state band: 39.5% at 30 s, 40.6% at
60 s).

---

## Cost against run length

Percent of one core. Ungated / gated. Every cell is single-threaded
(cpu/wall ≤ 1.003). **No cell reaches 100%** — the largest is 49.1%.

| fixture | 1 s | 2 s | 4 s | 8 s | 16 s | 30 s |
|---|---:|---:|---:|---:|---:|---:|
| `tone200` (continuous 200 Hz) | 1.1 / 1.4 | 2.2 / 2.5 | 39.6 / 39.8 | 49.1 / 40.3 | 41.0 / 40.5 | 43.0 / 40.4 |
| `hum120_005` (the hum fixture) | 1.0 / 1.3 | 2.0 / 2.3 | 41.4 / 22.5 | 41.4 / 12.3 | 41.3 / 7.5 | 43.8 / **5.3** |
| `testvad150` (the 116.9% fixture) | 1.0 / 1.3 | 2.0 / 2.3 | 37.3 / 37.7 | 37.1 / 37.5 | 37.1 / 33.3 | 41.7 / 19.1 |
| `speech` (real, `alice01`) | 2.2 / 2.5 | 21.8 / 21.9 | 24.2 / 24.6 | 25.5 / 25.8 | 26.6 / 26.8 | **29.9 / 30.2** |
| `white` (uniform noise) | 1.0 / 1.3 | 2.0 / 2.3 | 44.2 / **2.5** | 43.9 / 2.6 | 44.1 / 2.6 | 46.7 / 2.6 |
| `silence` (exact zeros) | 1.0 / 1.3 | 2.0 / 2.3 | 2.8 / 3.0 | 3.1 / 3.4 | 3.2 / 3.5 | 3.3 / 3.6 |

Two things to read off it. **The fixture moves the number by 14×** (3.3% for
silence, 46.7% for white noise, same code, same length). **The gate's value is
entirely which world the signal lands in**: it removes 44.1 pp from white
noise and 38.5 pp from the hum, and *adds* 0.26 pp to speech, because on
speech the gate accepts and YIN runs anyway.

---

## The mechanism, evidenced

**The buffer is bounded, and `_trim` does run.** The hypothesis was that on a
signal with no pauses nothing is ever trimmed and every push re-runs
`energy.rms_series` (`src/watcher.py:333`) over a growing buffer. That is
**refuted by direct observation**:

| signal | 30 s buffer max | mean | `_trim` calls | 60 s buffer max | `_trim` calls |
|---|---:|---:|---:|---:|---:|
| `tone200` | 32 160 (2.01 s) | 18 109 | 16 | 32 160 | 33 |
| `hum120_005` | 32 160 (2.01 s) | 18 109 | 16 | 32 160 | 33 |
| `speech` | 34 880 (2.18 s) | 14 741 | 30 | — | — |
| `silence` | 48 160 (3.01 s) | 25 790 | 10 | — | — |

The plateau is `MAX_UTTERANCE_S = 2.0` (`src/watcher.py:121`) for speech and
`MAX_PAUSE_S = 3.0` for silence. `_trim` fires every ~1.9 s, indefinitely: 33
flushes over 60 s of input, with the steady-state cost unchanged between 30 s
and 60 s. The buffer does not grow without bound; the per-push cost inside a
ramp is linear in buffer length, but the ramp is re-paid inside every cycle,
so it amortises rather than compounding.

**The 2 s → 4 s step is a discrete onset, and it has a name.** A dense sweep
across the boundary:

| clip length | `tone200`, ungated |
|---|---:|
| 1.75 s | 1.8% |
| 2.00 s | 2.0% |
| **2.10 s** | **35.0%** |

CPU-seconds go 0.044 (2 s) → 1.585 (4 s), a **36× step**. Quadratic growth
predicts 4× per doubling, so "quadratic" does not account for it, and the
parenthetical explanation is the right one to reject: this is not a slope, it
is a switch. At 2.0 s the open segment reaches `MAX_UTTERANCE_S`, is flushed
as a checkpoint, and `pitch.PitchTracker.track()` (`src/watcher.py:643-646`)
runs over the whole 2.0 s window. One 2 s YIN pass lands inside the clip and
the clip is suddenly an order of magnitude more expensive.

**The cost is YIN, not `rms_series`.** The per-push trace shows one push in
each 2 s cycle costing an order of magnitude more than its neighbours — that
is the flush — and the buffer control settles the attribution:

| 30 s signal | buffer max | steady-state rate | YIN runs? |
|---|---:|---:|---|
| `silence` | 48 160 (the **largest** buffer measured) | **3.3%** | never |
| `tone200` | 32 160 (smaller) | 39.5% | continuously |
| `speech` | 34 880 | 30.9% | over short utterance windows |

Silence runs `rms_series` over a bigger buffer than the tone does and costs
**12× less**, because it is marked silence and YIN is never asked. The base
cost of the whole buffer machinery — `rms_series` over a ~2 s buffer, every
push, forever — is about **3% of a core**. Everything above that is YIN.

So the cost model is roughly: **3% base + YIN duty cycle**, where the duty
cycle is ~1.0 for a continuous signal (a 2 s window every 2 s) and ~0.6–0.7
for real speech (many short utterance windows).

---

## Which figures are defensible

| figure | verdict |
|---|---|
| **31.8%** | **Not defensible as a measurement.** Never recorded against a fixture, a length or a clock. Arithmetically inconsistent with the `39.13 ms per 100 ms` printed beside it in the same commit. The nearest cell in this repo's own signal set is **30 s of real speech, ungated = 29.9%** — so the number is *approximately* right for a fixture nobody wrote down, which is worse than a typo: it looks reproducible. |
| **39.1% / 39.3%** | **Defensible for 30 s of real speech, ungated, list-fed.** `DEVICE-GENERALISATION.md`'s 39.25% reproduces the docstring's `39.13 ms per 100 ms` on the same fixture and length. This session measured 29.9% with a CPU clock on the same file — a 1.3× spread between two harnesses on one machine, which is the honest width of this number. |
| **116.9%** | **Not reproducible, and not explained by length.** The fixture is genuinely expensive — the gate scores it 0.94 and accepts it, so YIN runs either way — but on this machine it costs **37.3% list-fed at 4 s** and **91.8% numpy-fed at 4 s**. Neither reaches 116.9%. `inthewild_probe.py` measured wall clock, a single untimed pass, 128 ms chunks; wall clock on a shared machine is the obvious inflation path. The honest statement is "the fixture the gate cannot reject costs ~37% (shipped path)", not 116.9%. |

**The defensible summary, stated as a range rather than a number:** the
ungated watcher costs **3–47% of a core** depending on the fixture, is **flat
in run length beyond ~4 s**, and the gate takes a rejected signal from ~44% to
~3–5% while adding ~0.3 pp on real speech.

There is no averaging to be done here. Averaging 31.8% and 116.9% would
produce a number that describes no signal any of them measured.

---

## Does the gate remove the growth, or shrink the constant?

**It shrinks the constant, and only where the signal is rejected.** Measured at
30 s, ungated → gated:

| signal | ungated | gated | Δ | gated exponent ≥ 4 s |
|---|---:|---:|---:|---:|
| `white` | 46.7% | 2.6% | **−44.1** | 0.026 |
| `hum120_005` | 43.8% | 5.3% | **−38.5** | **−0.724** |
| `testvad150` | 41.7% | 19.1% | −22.6 | −0.315 |
| `tone200` | 43.0% | 40.4% | −2.6 | 0.007 |
| `speech` | 29.9% | 30.2% | **+0.3** | 0.097 |
| `silence` | 3.3% | 3.6% | +0.3 | 0.074 |

The **negative** exponents are the interesting ones: gated hum gets *cheaper*
with length (22.5% at 4 s → 5.3% at 30 s). That is not the gate improving. It
is one fixed cost amortising. On a 120 Hz hum at 0.05 the gate passes **1
frame in 937** (max 0.504, over the 0.5 threshold by 0.004) and that frame is
**index 2** — a start-up transient, which `tests/test_vad.py` already knows
about. Because `_gate_peak` (`src/watcher.py:552`) takes the **max** over the
segment, one such frame clears the gate for an entire 2.0 s window, so one
full YIN pass happens at the start of every run. On a 4 s clip that is 45% of
the clip; on a 30 s clip it is 6%. The gate's residual cost on hum is a
start-up artefact, not a hum leak.

For `testvad150` the gate's verdict is **length-dependent** — it accepts
everything over 6 s (`IN-THE-WILD-AUDIO.md`) but rejects some 2 s windows over
30 s, so the saving grows from −0.5 pp at 8 s to −22.6 pp at 30 s. That is
LSTM state drift, and it means any gated figure on that fixture is
length-specific.

`DEVICE-GENERALISATION.md` used 30 s. Its headline ratios (18.46 warm,
16.68 cold on hum; 1.02 on speech) are therefore length-specific results, and
should say so: they are ratios *at 30 s*, and the gated-hum ratio is exactly
the quantity that depends on length.

---

## What the `src/watcher.py` docstring should say

**Do not edit `src/` — another agent owns it. Recommended replacement for
`src/watcher.py:13-21`:**

```
Measured here, single thread, one core. THERE IS NO SINGLE NUMBER FOR THIS,
and quoting one bare is the mistake this section used to make. The cost
depends on the fixture and on the clip length, and both must be named:

    30 s of real speech, no gate          29.9% of a core
    30 s of a continuous 200 Hz tone      43.0%
    30 s of a 120 Hz hum, no gate         43.8%, with the gate 5.3%
    30 s of digital silence                3.3%

A continuous signal above the energy floor is marked speech from end to end,
so YIN runs continuously and the cost is ~3% base plus a near-unity duty
cycle. Real speech has pauses, so YIN runs over short utterance windows and
lands near 30%. The gate's whole value is which of those two worlds the signal
falls in; on speech it accepts and adds nothing.

The old "31.8% of a core" WAS NOT A MEASUREMENT ANYONE CAN POINT AT. The same
commit recorded "39.13 ms per 100 ms" beside it, which is 39.13%, and 39.3%
re-measures on 30 s of the same fixture today. No experiment in this repo
ever produced 31.8%. It is a fixture and a length that were never written down.

THE GROWTH IS BOUNDED, NOT QUADRATIC. On a signal with no pauses the retained
buffer grows until the open segment reaches MAX_UTTERANCE_S and is flushed, so
it plateaus at ~2.0 s (32 160 samples measured) and _trim runs every ~1.9 s
indefinitely -- 33 flushes over 60 s of input, per-second cost unchanged
between 30 s and 60 s, growth exponent 0.01 beyond 4 s. What a short clip
shows as "it gets worse with length" is the startup ramp amortising plus one
YIN pass landing inside the clip: the cost steps 36x between a 2.00 s clip
(2.0%) and a 2.10 s clip (35.0%), which is the checkpoint switching on, not a
slope.

So: quote a length AND a fixture, or quote nothing. Full grid, per-push traces
and the container artifact that doubles every numpy-fed harness:
docs/research/WATCHER-COST.md.
```

Two further recommendations for whoever owns `src/`, neither applied here:

- **`tests/test_vad.py` and `src/vad.py:41` still cite 39.13 ms per 100 ms as
  a fixed number.** That reference should carry the fixture and length with it,
  or the same drift recurs.
- **The gate's max-over-segment rule costs one full YIN window per run on a
  rejected hum**, from a single start-up frame scoring 0.504. Dropping the
  first N gate frames, or requiring a minimum count of passing frames rather
  than a max, would remove it. Not implemented here — `src/` is not this
  task's lane, and this is a behavioural change that wants its own measurement.

---

## Honest limits

- **One machine, one Python build.** Everything is x86-64, WSL2, CPython 3.14.
  Nothing here says what a Pi-class target does. `DEVICE-GENERALISATION.md`
  asks that question with a cache-size proxy and is the document that answers
  it; this one does not.
- **Percent-of-a-core is not portable, and this file demonstrates that against
  itself.** Two harnesses on one machine, same file, same 30 s, disagree by
  1.3× (29.9% here on a CPU clock, 39.25% there on a wall clock). The
  *container* effect is larger still and fully explained (~2×). Only ratios
  taken with a stated container and clock should travel.
- **`RUSAGE_SELF` counts every thread in the process.** Pinning the BLAS
  environment removes the observed contamination (max ratio 1.003), but an
  ONNX session loaded in the same process could still contribute. The gated
  cells carry their own `threads_ratio` for exactly this reason.
- **2 timed passes per cell.** Tight enough here (spread under 3%) to separate
  fixtures, not tight enough to resolve the ~10% cache effect
  `DEVICE-GENERALISATION.md` declined to claim.
- **The dense target search was truncated.** The 2.00/2.10 s cliff is measured
  and in the data; the full 90-cell sweep was stopped on time budget, so the
  "nearest cell to each disputed figure" summary is computed from the coarse
  grid plus the partial sweep, not from a complete one. The 31.8% attribution
  rests on the coarse grid (30 s speech = 29.9%) and on the arithmetic
  contradiction inside the commit, not on an exhaustive search.
- **Synthetic hums are not fans.** Every non-speech signal here is a generator.
  `IN-THE-WILD-AUDIO.md` §9 already says this; nothing here improves on it.