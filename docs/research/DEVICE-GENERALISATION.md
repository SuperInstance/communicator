# DEVICE GENERALISATION — does the gate's advantage survive cache pressure?

**Experiment:** `experiments/device_generalisation.py`
**Data:** `experiments/device_generalisation_results.json`
**Date:** 2026-10-18 · AMD Ryzen AI 9 HX 370 (24 threads, L1d 48 KiB/core, L2
1 MiB/core, L3 16 MiB shared), WSL2 · numpy 2.5.4, onnxruntime 1.31.0 ·
gate pinned to 1 thread (`vad.py` `intra_op_num_threads = 1`, already
established, not re-measured here)

---

## The question

`docs/research/ADOPTION-PLAN.md` §8 names the thing that would most change our
minds about commit `8eb05f6`: every number behind the silero gate — 31.8% of a
core down to 2.4% — comes from one machine.

The falsifiable worry is about the **shape** of the cost, not its size. The gate
is a 2,327,524-byte ONNX graph. Ungated, the watcher's dominant cost is
`pitch.PitchTracker` (YIN) over a few hundred KB of buffers. **2.2 MB does not
fit in this machine's 1 MiB per-core L2 — it only fits in the 16 MiB shared L3.**
On a Pi-class target there is no 16 MiB L3 to fall back on. So the gate's cost
could scale worse than YIN's once the working set stops fitting in cache, and if
it does, the 15× advantage shrinks or inverts on the actual device.

Stated before running, so a disagreeing result would be informative:

> **Hypothesis (strong form).** Under cache pressure the gate's advantage over
> the ungated watcher shrinks materially or inverts.
> **Hypothesis (weak form).** The gate degrades *faster* than YIN in relative
> terms, because its working set is 2.2 MB and YIN's is not.

---

## The measurement

Same fixed audio, four cells. Percent of one core to push 30 s through a
`Watcher`, 100 ms chunks (the granularity `hum_sweep.py` uses), 1 untimed warmup
pass then 5 timed passes, median reported.

| regime | gate | meaning |
|---|---|---|
| warm | `SileroGate()` | model and buffers resident — the conditions behind the existing claim |
| cold | `SileroGate()` | L1/L2/L3 flushed before every 100 ms chunk |
| warm | `None` | YIN always runs |
| cold | `None` | same, with cache pressure |

**The flush is verified, not assumed.** "Cold cache" is a label, and an
unverified label is worth nothing. A deliberately L1-resident control loop
(32 KiB, fits the 48 KiB L1d) was timed under the identical flush:

| control loop | median | stdev | runs |
|---|---|---|---|
| warm | **0.951 µs** | 6.761 | 88.7 (first touch, page fault), 5.01, 1.573, 1.032, 0.962, 1.012 … |
| cold | **18.340 µs** | 6.122 | 20.398, 18.595, 19.818, 18.575, 16.692, 18.065 … |

**19.29× slower** under the same flush. The cold column is a real cold column.
The flush itself costs 3.43–3.53 s per 30 s run and is timed **separately and
excluded** — otherwise "cold" would be measuring the flush.

### Signal 1 — hum, the case the claim was built on

`harmonic_stack(120, 0.05)`, the mains-family signal from `hum_sweep.py`.
Gate scores it **mean 0.0236, frac ≥0.5 = 0.0011** — rejected, so YIN never runs.

| cell | median | stdev | min | max |
|---|---|---|---|---|
| warm ungated | 61.268% | 2.482 | 56.820 | 62.385 |
| cold ungated | 65.831% | 3.358 | 62.103 | 70.061 |
| warm gated | 3.319% | 0.259 | 3.072 | 3.760 |
| cold gated | 3.946% | 0.378 | 3.438 | 4.404 |

| quantity | value |
|---|---|
| **ratio warm (ungated/gated)** | **18.46** |
| **ratio cold (ungated/gated)** | **16.68** |
| gated inflation (cold/warm) | 1.189 |
| ungated inflation (cold/warm) | 1.074 |
| **relative cache penalty** (gated ÷ ungated inflation) | **1.107** |

### Signal 2 — real speech, the case the gate must not break

`examples/realaudio/alice01_excerpt.wav`, read with `src.wavio.read_wav`,
truncated to 30 s. Gate scores it **mean 0.6854, frac = 0.6809** — accepted, so
YIN runs anyway. The gate is *not supposed* to help here, and it does not.

| cell | median | stdev |
|---|---|---|
| warm ungated | 39.252% | 3.257 |
| cold ungated | 39.287% | 1.032 |
| warm gated | 38.345% | 2.124 |
| cold gated | 36.398% | 3.081 |

ratio warm **1.02**, ratio cold **1.08**. Gate-alone inflation **1.83×**.

### Attribution — where the cold penalty actually comes from

The gate measured on its own, same protocol, no watcher:

| signal | warm | cold | inflation |
|---|---|---|---|
| speech | 0.414% | 0.756% | **1.83×** |
| hum | 0.511% | 0.818% | **1.60×** |

This is the structural confirmation: **the 2.2 MB model really is being re-read
from DRAM on every chunk.** The premise of the hypothesis was correct. It just
does not matter at this magnitude.

---

## What the data says

**The strong hypothesis is FALSIFIED.** The advantage does not shrink
materially and does not invert. On hum the gate is **15–20× cheaper ungated**
both with a warm cache and with the cache destroyed between every chunk. The
entire cache-pressure effect is a ~10% reduction in the ratio, from 18.46 to
16.68 — one point eight of eighteen and a half.

**The weak hypothesis is CONFIRMED, weakly.** The gate does degrade faster than
YIN in relative terms: relative cache penalty **1.107** on hum, and the gate in
isolation inflates **1.60–1.83×** against YIN's **1.074**. The prediction about
the *shape* was right. The magnitude is two orders of magnitude too small to
touch the decision.

**But the 10% is not cleanly resolved from noise.** Pairing all 5×5 run
combinations:

| ratio | min | median | max |
|---|---|---|---|
| hum warm | 15.11 | 18.46 | 20.31 |
| hum cold | 14.10 | 16.83 | 20.38 |
| speech warm | 0.85 | 1.02 | 1.24 |
| speech cold | 0.89 | 1.08 | 1.19 |

The warm and cold hum distributions overlap heavily. (The medians here are
over all 25 paired ratios, so they differ slightly from the ratio-of-medians
figures in the tables above — 16.83 vs 16.68. Both are reported rather than
picking the flattering one.) **The honest reading is "no detectable
degradation", not "9.6% degradation."** The gated medians also overlap at the
tails (warm max 3.760 vs cold min 3.438). A larger repeat count would be needed
to claim the 10% as a real effect. What is *not* in doubt is that the ratio
stays in the 14–20 band in every single cold run — nowhere near inversion.

**On speech the measurement cannot resolve anything smaller than ±20%**, and the
gate is a wash (ratio ≈ 1.0, consistent with it accepting 68% of frames). That
is the correct behaviour: the gate decides whether to pay for YIN, and on speech
the answer is yes.

### Two things the numbers say that the old claim does not

1. **Ungated hum costs MORE than ungated speech** (61.3% vs 39.3% of a core).
   The energy-only segmenter marks the whole hum clip as speech — the 89% figure
   already recorded in `vad.py`'s docstring — so YIN runs continuously rather
   than only over speech segments. The hum case is *worse* than the committed
   31.8%, which makes the gate's job there more valuable, not less.
2. **The absolute figures do not reproduce 31.8%.** Ungated speech measured
   39.3% here, not 31.8%. Different clip, level, run, and machine load. This is
   exactly why the ratio is the portable quantity and the percent-of-a-core
   figures are not.

---

## Is a held-out device reachable from here? No.

Checked, not assumed:

| probed | result |
|---|---|
| `qemu-aarch64`, `qemu-aarch64-static`, `qemu-user-static`, `qemu-arm`, `qemu-riscv64`, `qemu-x86_64` | **all absent** (`which` → not found) |
| installed qemu packages (`dpkg -l`) | **none** |
| `binfmt_misc` handlers | mounted, but only `WSLInterop` and `python3.14` — **no qemu-user interpreter registered** |
| Docker | CLI present (`/mnt/c/Program Files/Docker/Docker/resources/bin/docker`) but **not usable**: *"The command 'docker' could not be found in this WSL 2 distro. We recommend to activate the WSL integration in Docker Desktop settings."* WSL2 integration is **disabled** |
| podman | **absent** |
| passwordless sudo | **unverified, not "no"** — `sudo` exists at `/usr/bin/sudo`, this user *is* in the `sudo` group, and `/etc/sudoers.d/proart-nopasswd` exists, but it is root-owned `0440` and unreadable, and this session's permission rules block invoking `sudo` at all |

**Conclusion: no cross-architecture execution is available on this machine.**
Even if passwordless sudo did work, installing `qemu-user-static` is a
system-level change requiring network access and explicit approval — not
something to do unilaterally inside a measurement task.

No device measurement was performed, and none is claimed. Everything above is
x86-64.

---

## Honest limits

- **This is a cache-size PROXY for device generalisation. It is not a held-out
  device.** It perturbs one axis — working set vs cache size — and leaves core
  count, memory latency, ISA, ORT kernel selection, and thermal behaviour
  untouched. A Zen 5 L3 miss is not a Cortex-A72 miss.
- **It can falsify, not confirm.** Had the ratio collapsed under cache pressure,
  the claim would be dead and this would be the evidence. That it held here says
  the cache axis is not the killer. It says nothing about the axes this proxy
  does not move — most importantly ARM NEON kernels and single-core memory
  bandwidth, which is where a Pi-class target could still hurt.
- **The flush is harsher than reality.** It evicts the watcher's own buffers as
  well as the gate's weights, and does so every 100 ms. A real device has a
  small model competing with the analysis buffers, not a total cache wipe.
- **Absolute percent-of-a-core figures are not portable** and did not even
  reproduce the previously committed 31.8% on the same class of machine. Only
  the ratio should travel.
- **5 repeats is thin.** The headline ratio distributions overlap; the ~10% cold
  effect is at the edge of what this repeat count can resolve.
- **qemu-user would not have been a real device test anyway** — it emulates a
  foreign ISA under an x86 host's memory system, so it measures instruction
  translation, not a peripheral's cache and memory behaviour.

## What would actually settle it

A held-out device. Everything short of that — cache pressure, an emulator, a
penalty multiplier — is a proxy, and this document's value is that it is
explicit about which side of that line it sits on.