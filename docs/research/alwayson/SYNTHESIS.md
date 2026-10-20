# SYNTHESIS — what should the always-on hot path actually be?

**Recommendation: keep `src/watcher.py` as the hot path and put a silero-vad ONNX model in
front of it as a gate. Reject openWakeWord's weights on licensing. Reject pyannote
outright.**

The single new dependency is `onnxruntime`. The silero model is 2.220 MB, MIT, costs
**0.23% of one core**, and scores **0.0036 on steady hum against 0.75+ on real speech** —
which is the exact failure our watcher declares it cannot fix.

Everything below is measured on this machine (AMD Ryzen AI 9 HX 370, 24 cores,
onnxruntime 1.31.0, **1 thread**) by [`measure_hotpath.py`](measure_hotpath.py), or
`stat`ed from the real release artifacts. Script output:
[`hotpath_measurements.json`](hotpath_measurements.json).

---

## 1. The comparison that decides it

Same six signals, same 4-second clips, same machine, same thread.

| signal | **our watcher: s of "speech"** | **silero-vad v5: mean score** | **silero-vad v5: max** | frac > 0.5 |
|---|---:|---:|---:|---:|
| steady 60 Hz hum + harmonics | **3.550 / 4.000 (89%)** | **0.0036** | 0.0374 | **0.000** |
| fan noise (3.5 Hz AM) | 3.550 (89%) | 0.0063 | 0.0306 | 0.000 |
| keyboard clicks | 3.550 (89%) | 0.0054 | 0.0334 | 0.000 |
| music triad | 3.550 (89%) | 0.0052 | 0.1736 | 0.000 |
| flat white noise | 3.550 (89%) | 0.0081 | 0.0289 | 0.000 |
| near-silence | 3.550 (89%) | 0.0067 | 0.0230 | 0.000 |
| real speech ("alexa") | 3.460 | **0.8819** | 1.0000 | 0.880 |
| real speech ("hey mycroft") | 2.650 | **0.7642** | 1.0000 | 0.760 |
| real speech ("hey jane") | 2.300 | **0.7530** | 1.0000 | 0.752 |

Two columns, and they tell opposite stories.

**Our watcher calls 89% of a steady hum "speech", and it calls *silence* speech at exactly
the same rate** — 3.550 s in all six non-speech cases, which is a saturation artefact: the
segmenter is marking the whole clip. This is the declared failure at `src/watcher.py:40-47`,
now quantified. It is not a subtle degradation. On a room with a fan, our watcher has
**no silence events at all**, so every downstream table built on pause events is
structurally wrong in the same direction.

**silero-vad separates cleanly on every class**, including the hum that specifically breaks
us, and it does it at a threshold (0.5) sitting in a gap **93.0× wide on the mean**
(0.7530 worst speech mean ÷ 0.0081 worst non-speech mean) and **5.76× wide on the
worst-case max** (1.0000 worst speech max ÷ 0.1736 worst non-speech max, music).

> **CORRECTION (2026-10-19).** This said *"a gap 28× wide on the mean and 5.4×
> wide on the worst-case max (0.1736 music vs 0.7530 speech)"*. **No
> derivation in `hotpath_measurements.json` yields 28 or 5.4.** From
> `silero_v5_scores.current_v5_opset16`: worst speech mean **0.7530**
> (speech_jane) and max **1.0000**; worst non-speech mean **0.0081** (white
> noise) and max **0.1736** (music). So the derivable figures are **93.0× on
> the mean** (0.7530 / 0.0081) and **5.76× on the max** (1.0000 / 0.1736), and
> the note's own parenthetical pair (0.7530 / 0.1736) is **4.34×**. **The
> underlying separation is not wrong — the file supports a LARGER mean gap than
> 28×.** `NOTES-AUDIT.md` F10.

## 2. Cost: nobody here is close to the bottleneck

| Component | Cost, 1 thread | % of one core at 1× realtime |
|---|---:|---:|
| **our watcher** (`src/watcher.py`) | **39.13 ms per 100 ms push** (mean of the 6 non-speech fixtures) | **39.1%** |
| **silero-vad v5**, one 32 ms chunk | **74.9 μs** | **0.23%** |
| openWakeWord full stack, all 6 heads | 1.63 ms per 80 ms | 2.04% |
| openWakeWord, shared front only | 0.92 ms per 80 ms | 1.15% |
| pyannote diarization | **not published for CPU** | **unknown** |

**The uncomfortable finding about our own hot path:** `src/watcher.py:1` says
*"No model, no network, microseconds."* It is measured at **39.13 ms per 100 ms of audio**
— 39.1% of a core, a realtime ratio of only **2.56×**, and it is *model-free arithmetic*
that is 39% of a core. The docstring is wrong by three orders of magnitude.

**That 39.13 ms is the mean over the SIX NON-SPEECH fixtures** (hum 31.913, fan
41.095, clicks 48.387, music 33.877, white noise 39.307, silence 40.153 ms).
On **real speech** the watcher is **21.9–36.1 ms** (speech_jane 21.945,
speech_mycroft 24.666, speech_alexa 36.124; mean 27.578 ms) — a realtime ratio
of **2.77–4.56×**, against **2.1×** (48.387 ms, keyboard clicks) for the
non-speech worst case.

> **CORRECTION (2026-10-19) — "39.1% of a core" had no denominator.** From
> `hotpath_measurements.json` → `watcher`: the six non-speech fixtures above
> average **39.122 ms**; the three speech fixtures average **27.578 ms**. The
> rate is real and the trade-off is sound — "a gate that costs 0.23% of a core
> to delete a 39%-of-a-core failure mode" is correct precisely because 39.1%
> is the non-speech cost — but the population has to be named, and the **8× Pi
> projection (313% of a core) below is the non-speech worst case stated as the
> general case**. `NOTES-AUDIT.md` F11.

That does not disqualify it — 39% of one core to run all day is affordable, and on this CPU
it is well clear of saturation. But it changes the framing: **the hot path is not nearly
free, so anything that can be taken *out* of it should be.** A VAD gate that costs 0.23%
to delete a 39% failure mode is the best trade available in this lane.

Watcher realtime ratio varies by signal: 2.1× (keyboard clicks — the worst case, 48.4 ms
per push) to 4.6× (real speech, 21.9 ms per push). Cost tracks how much of the buffer is
classified as speech, because that is when the YIN pitch tracker runs.

**Caveat on every number here:** this is a 24-core x86 desktop CPU, not the target
peripheral. A Raspberry Pi 3 core is roughly 4–8× slower per core. At 8× pessimism:
silero 1.8% of a core (fine), watcher 313% of a core (**saturated — this becomes the
problem**; this is the **non-speech** projection — 39.122 ms mean, 48.387 ms worst case
— and on speech the same 8× gives 176–289% of a core (221% at the 27.578 ms speech
mean), so the saturation is real on both populations), openWakeWord 16% (fine).
**On a Pi-class target our watcher would not keep up,
and it is the thing that is easy to fix**: `src/watcher.py:319-325` runs a YIN pitch
tracker over the whole window of every speech segment, and it is the dominant cost.

## 3. Why each candidate wins or loses

### silero-vad — **ADOPT (the gate)**

| | |
|---|---|
| Size | 2.220 MB (`silero_vad.onnx`), 309,633 params in the 16 kHz variant |
| Latency | **74.9 μs per 32 ms chunk, 428× realtime**, 1 thread |
| CPU-only | yes, overwhelmingly |
| Licence | **MIT** — `LICENSE:1-3` |
| No DL framework | **The ONNX model: yes.** The pip package: **no** |

The dependency catch, stated precisely because it decides the implementation:
`pyproject.toml:32` puts `torch>=1.12.0` in **base** `dependencies`, and
`src/silero_vad/model.py:2` does `import torch` at module scope, reached by
`__init__.py:7`. So **`pip install silero-vad` installs torch even with `[onnx-cpu]`**,
because extras only ever add. In our runtime that path is simply unavailable.

The way through is 30 lines of numpy + onnxruntime — and **openWakeWord already wrote
them, MIT, with silero's licence header preserved** (`openwakeword/vad.py:15-39`). We take
the *technique and the MIT VAD copy* from openWakeWord, and the *weights* from silero.
That combination is both permissible and the cheapest possible path.

One improvement on openWakeWord's copy: it hardcodes `frame_size=480` (30 ms,
`vad.py:129`) whereas current silero is a 512-sample/32 ms model with a 64-sample context
(`utils_vad.py:112,130`). I verified the newer convention works and is measurably better
separated (0.0036 vs 0.0001 on hum, 0.88 vs 0.80 on speech).

### openWakeWord — **REJECT the weights; COPY three design ideas**

| | |
|---|---|
| Size | 2.303 MB fixed front + 0.195–1.662 MB per wake word |
| Latency | 1.63 ms per 80 ms frame, all 6 heads — genuinely fast |
| CPU-only | yes |
| Licence | code Apache-2.0, **pre-trained models CC BY-NC-SA 4.0 = NON-COMMERCIAL** |
| Benchmark in repo | **none that runs** — `benchmark/benchmark.py:29` passes `input_sizes=[16]`, a parameter that exists nowhere in v0.6.0 |

It is the fastest, smallest, best-engineered of the three, and we cannot ship its models.
It also wants six pip installs (`scipy`, `scikit-learn`, `requests`, `tqdm`,
`onnxruntime`, `ai-edge-litert`), and `import openwakeword` fails outright without
`requests` (`utils.py:28`) even on the ONNX path.

Do we need a wake word at all? **Not yet, and that is worth saying.** A wake word solves
"only react to *this* phrase". Our brief is "watch audio continuously and cheaply". The
VAD answers "did someone speak"; the wake word answers "did someone say *the thing*".
Those are different products and the second is a stage-2 concern. If we want it later, the
licence retrain path is a research environment (`setup.py:54-78`: torch, speechbrain,
tensorflow-cpu, deep-phonemizer) plus **~30,000 hours of negative data** — which is the
actual blocker, not the compute.

### pyannote-audio — **REJECT**

| | |
|---|---|
| Size | 31.301 MB for the pipeline (14× silero), 81% of it a speaker-embedding net |
| Latency | **not published on CPU.** Only `README.md:110`: an NVIDIA H100 80GB |
| CPU-only | **unknown — "runs" is not "fits a single core"**, and nobody has measured it |
| Licence | code MIT; **weights CC-BY-4.0/MIT — commercially usable, but gated behind click-through**; the 25 MB embedding additionally carries WeSpeaker's own `LICENSE.WeSpeaker` |
| ONNX path | **none — `grep -rn -i onnx` over the whole repo returns zero** |
| Deps | 21 packages, incl. `torch`, `lightning`, `torchcodec`, `torchmetrics` |

The decisive objection is **structural, not practical**. Diarization needs
1. segmentation over every frame, 2. an embedding per region, 3. **clustering** — and
clustering is order-dependent and non-incremental. You cannot decide two regions are the
same speaker until you have seen both, so an unbounded stream forces unbounded state or
periodic re-clustering. That is directly incompatible with the property our watcher is
deliberately built around: `push()` discards everything emitted except the tail so memory
does not grow with session length (`src/watcher.py:126-128`).

Worth taking as **ideas, not dependencies**: PLDA + agglomerative clustering as a cheap
non-neural "same speaker?" test, and DER as the evaluation standard if we ever want to
claim speaker awareness. If we ever do want speaker awareness in the hot path, the right
shape is openWakeWord's *verifier-on-the-onset* — embed the utterance we already have and
run one classifier — not diarization on the stream.

## 4. The recommended architecture

```
mic 16 kHz
  │
  ├── every 32 ms (512 samples):
  │       silero_vad.onnx  ──►  p_speech          0.23% of a core
  │       (numpy + onnxruntime; 64-sample context; carry state)
  │
  └── gate: if p_speech > threshold:
              existing src/watcher.py.push()      39.1% of a core,
              on the gated audio only             paid only when someone speaks
                    └──► onset / utterance / pause events, unchanged
```

Properties that matter:

- **Bounded state.** silero's state is `(2, 1, 128)` floats — 1 KB, fixed. Compatible
  with our bounded-buffer design.
- **Fail-open.** If the ONNX file is missing or onnxruntime fails to import, fall back to
  today's behaviour. This must not be able to stop the peripheral hearing anything.
- **The gate reduces the watcher's cost**, not just its false positives: today the YIN
  tracker runs over every loud periodic hum (48.4 ms per push on clicks, the worst case);
  gated, it runs only on real speech (21.9 ms per push measured).
- **Thresholds are ours.** silero's defaults are `threshold=0.5`,
  `neg_threshold=max(threshold-0.15, 0.01)` (`utils_vad.py:475-476`). The measured
  separation (worst non-speech max 0.1736, worst speech mean 0.7530) suggests we can sit
  well above 0.5, but that needs real-room audio, not synthetic clips.

### The known risk, stated plainly

My non-speech signals are synthetic and clean. Real rooms have reverb, competing speech,
music beds and HVAC that this battery does not model. **The 93.0× mean separation is measured
on 3 speech clips and 6 synthetic signals and is not a certification.** Before adopting, the
measurement to run is the real-room fixture lane (`LANE-FIXTURES.md`) with actual recordings
of the fan that is currently breaking us. That is a day of work and it is the gate on this
recommendation.

## 5. The 5 most stealable mechanisms

Each with a licence and a `file:line`.

| # | Mechanism | Where | Licence | Why it is worth taking |
|---|---|---|---|---|
| 1 | **VAD wrapper: 30 lines of numpy + onnxruntime, no torch** | `openwakeword/vad.py:54-130` (MIT header preserved at `:15-39`) | **MIT** (both silero's and openWakeWord's copy) | This is the whole adoption path. It proves the ONNX model needs no DL framework, and it is the code we would otherwise have to write. |
| 2 | **Hysteresis post-processing over a probability stream** | `silero-vad/src/silero_vad/utils_vad.py:421-...` (`get_speech_timestamps_from_probs`) | **MIT** | Separate enter/exit thresholds (`threshold=0.5`, `neg_threshold=threshold-0.15`, `:475-476`) plus `min_speech_duration_ms=250`, `min_silence_duration_ms=100`, `speech_pad_ms=30`. I verified its body contains **zero torch references** — it is a pure-Python state machine over a list of floats. Directly portable, and the hysteresis idea is worth having independently of silero. |
| 3 | **Frozen shared backbone + tiny per-key heads** | `openwakeword/openwakeword/model.py:143-182`; architecture rationale in README "Model Architecture" | **Apache-2.0 (code)** — the *design* is free even though the weights are not | Measured: the shared front is 92% of the cost and a 7th head costs 0.0103 ms. This is how you get N detectors for ~1× the price. We could apply it to whatever small acoustic features we trust later. |
| 4 | **Streaming melspectrogram with a bounded rolling buffer + fixed-stride embedding** | `openwakeword/openwakeword/utils.py:165-166` (`melspectrogram_max_len = 10*97`), `:229-235` (`step_size=8`, `window_size=76`) | **Apache-2.0 (code)** | The exact same bounded-buffer discipline our watcher already uses, for a neural front end. The magic numbers (10 s buffer, 76-frame window, 8-frame stride ≈ one embedding per 80 ms) are the reusable content. |
| 5 | **Zero the first 5 frames after model load** | `openwakeword/openwakeword/model.py:330-333` | **Apache-2.0 (code)** | Three lines that prevent a warm-up transient from firing a false activation at startup. We have exactly this bug class — `src/watcher.py` needed `MAX_UTTERANCE_S`/`_emitted_until` guards to stop emitting 71 events for 3 utterances. Cheap insurance against a cold-start false positive in a device that is listening from boot. |

Deliberately **not** on this list: the wake-word weights (NC), the silero pip package
(torch-gated), and anything from pyannote's pipeline. Also not listed: `MAX` over narrow
Nouls as a mechanism, because that is Jev, not audio — it is covered in
[JEV-AS-INSTRUMENT.md](JEV-AS-INSTRUMENT.md) §E2 and applies at the decision layer, not here.

## 6. Ranked summary

| Rank | Option | Size | CPU cost | Fixes hum | Licence | No-torch path | Verdict |
|---:|---|---:|---:|---|---|---|---|
| **1** | **watcher + silero gate** | +2.220 MB | +0.23% | **yes, 0.0036 vs 0.75** | MIT | **yes (reimplement)** | **ADOPT** |
| 2 | watcher as-is | 0 | 39.1% | **no** | ours | yes | keep, but **fix the hum and profile the YIN cost** |
| 3 | + openWakeWord | +2.3–8 MB | +2.04% | via its own VAD | **NC weights** | needs 6 installs | **REJECT weights, copy design** |
| 4 | pyannote diarization | +31.3 MB | unknown | n/a | gated | **no ONNX at all** | **REJECT** |

## 7. Next actions

1. Vendor a 30-line `numpy + onnxruntime` silero wrapper (MIT, attribution preserved) and
   gate `src/watcher.py` behind it. **Fail open.**
2. Measure on the real-room fixture lane (`LANE-FIXTURES.md`) with the actual fan/HVAC
   recording that is currently breaking us. This is the gate on the recommendation.
3. Tune the threshold on real audio; the measured gap suggests 0.5 is conservative.
4. Profile the YIN pitch tracker. At 39.1% of a core it is the dominant hot-path cost and
   on a Pi-class target it is the thing that does not fit. Running it only on gated speech
   is the cheap first win.
5. Drop openWakeWord. Record why: CC BY-NC-SA 4.0 on the weights, and a benchmark script
   that cannot run.