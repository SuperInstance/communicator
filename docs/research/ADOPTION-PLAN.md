# ADOPTION PLAN — five research lanes, one ranked set of decisions

Written after five parallel research lanes, each of which read source, ran
measurements on this machine, and wrote up its own learning journey:

| lane | repos | docs |
|---|---|---|
| audio-llms | AudioGPT, VoxCPM, FunASR | [audio-llms/](audio-llms/SYNTHESIS.md) |
| audio-engines | SuperCollider, Sonic Pi, ZzFX, glicol, Pizzicato, pts | [audio-engines/](audio-engines/SYNTHESIS.md) |
| audio-dsp | aubio, torch-audiomentations, figaro, mechvibes, noisecraft | [audio-dsp/](audio-dsp/SYNTHESIS.md) |
| alwayson | openWakeWord, silero-vad, pyannote | [alwayson/](alwayson/SYNTHESIS.md) |
| voiceloop | pipecat, MeloTTS, Coqui TTS | [voiceloop/](voiceloop/SYNTHESIS.md) |

Plus the TypeSafe/Jev work in [jev-play-findings.md](jev-play-findings.md).

This file is the join. It ranks what to do, says what **not** to do, and names
the one thing that would most change our minds.

---

## 1. The finding that changes the architecture

**Our hot path is expensive and wrong, and a 2.2 MB model fixes both.**

Measured on this machine, same six signals, same clips, one thread:

| signal | our `watcher.py` | silero-vad v5 |
|---|---:|---:|
| steady 60 Hz hum | **89% "speech"** | **0.0036** |
| fan noise | **89%** | 0.0063 |
| keyboard clicks | **89%** | 0.0054 |
| music triad | **89%** | 0.0052 |
| white noise | **89%** | 0.0081 |
| near-silence | **89%** | 0.0067 |
| real speech | 3.46 s | **0.88** |
| **cost, 1 thread** | **39.1% of a core** | **0.23% of a core** |

Two things, and both are uncomfortable.

**Our watcher calls a steady hum "speech" and calls *silence* speech at exactly
the same rate** — 3.550 s in all six non-speech cases. That is saturation: the
segmenter marks the whole clip. This is the failure `src/watcher.py:40-47`
declares, now quantified. It is not degradation, it is total. In a room with a
fan we emit **no pause events at all**, so every table built on pauses is
wrong in the same direction.

**The docstring is wrong by three orders of magnitude.** `src/watcher.py:1`
says *"No model, no network, microseconds."* It is measured at **39.13 ms per
100 ms of audio** — a realtime ratio of 2.56×, and it is model-free arithmetic.

On this desktop that is affordable. **On a Pi-class target it is not** — at 8×
pessimism the watcher needs 313% of a core while silero needs 1.8%. The
dominant cost is `src/watcher.py:319-325`, which runs the YIN tracker across
the whole window of every speech segment, and it runs precisely when the
segmenter is wrong about a hum.

**Decision: put a silero-vad ONNX gate in front of the watcher.** MIT, 2.220
MB, 309,633 params, 74.9 μs per 32 ms chunk. A gate that costs 0.23% of a core
to delete a 39%-of-a-core failure mode is the best trade available.

**The dependency trap, stated because it decides the implementation:**
`pip install silero-vad` **installs torch anyway** — `pyproject.toml:32` puts
`torch>=1.12.0` in *base* dependencies, and extras only ever add. The way
through is ~30 lines of numpy + onnxruntime, and openWakeWord already wrote
them under MIT with silero's licence header preserved. Take the *technique*
from openWakeWord and the *weights* from silero.

## 2. What we already have and should keep

**Our YIN is as good as aubio's, and better on four axes.** A lane read
`pitchyin.c:137-171` against `src/pitch.py:155-187` line by line: identical
difference function, identical cumulative-mean normalisation, identical global-min
fallback, identical parabolic refinement. The two thresholds converged
independently — aubio `tol = 0.15`, ours `threshold = 0.15`.

We beat it on: an **amplitude gate before periodicity** (aubio's YIN is
amplitude-invariant by construction, so it reports confident f0 for a fan),
non-finite rejection, confidence-gated gaps, and **octave repair** — aubio
ships no pYIN and no octave post-pass at all. aubio is **GPL-3.0**, which is
the decisive wall anyway, and its build fails against numpy 2.x
(`ufuncs.c:48`).

**Do not adopt it. Do mine it** — its `yinfft` (Wiener–Khinchin YIN, O(N log N))
and its psychoacoustic spectral weighting are the two things we lack.

**The one DSP mechanism worth stealing from that lane is not about pitch.**
`peakpicker.c:110-117` uses a **locally adaptive median threshold**. Our
thresholds are **file-global** (`energy.py:28-29`). A peripheral that runs all
night needs them re-anchored locally — this is the same class of defect as the
running-peak bug fixed earlier, one level up. GPL-3.0, so reimplement from the
description.

## 3. Turn-taking: our 150 ms is doing two jobs

pipecat's numbers are the reference: **~200 ms to trigger** a VAD stop, **3 s to
give up** and treat it as end-of-turn. Our single 150 ms threshold is a fine
trigger and a **20×-too-eager decision**.

The shape worth adopting is three-stage, and it is cheap:

1. VAD gate (ours already, plus silero).
2. A **binary classifier on the raw waveform, no transcript** — smart-turn v3.2,
   **8,679,182 bytes, bundled in the pip package, BSD-2**.
3. A **speculation gate**: `OPEN / HOLDING / DROPPING` with a
   predict → void → commit lifecycle. This is what makes an aggressive trigger
   *defensible*, because the reaction starts before the decision is safe.

Measured here: features 3.7 ms, ONNX inference 59 ms, **total 63–79 ms**
(±15% run to run on an idle box, single-threaded). And a finding that matters
more than the latency:

> **The smart-turn model says "turn complete" about silence.**
> P(complete): digital silence **0.987**, noise 0.962, pure tone 0.957,
> speech-like 0.0495.

It discriminates speech correctly and must still be **VAD-gated** — feed it raw
audio and it agrees with you constantly.

**One inversion to copy deliberately:** pipecat fails **open** on model timeout
(`SmartTurnTimeoutException → COMPLETE`). For a peripheral that is backwards —
an unwanted gesture cannot be recalled. **Fail toward `INCOMPLETE`.**

Also inherit their deadline arithmetic (`turn_analyzer_user_turn_stop_strategy.py:236`):
the VAD reports "stopped" only `stop_secs` late, so `now + stt_timeout`
double-counts. We would have shipped that bug the moment we added a second stage.

## 4. What not to build

- **No streaming TTS exists.** `grep -rn "yield\|stream"` over MeloTTS's API and
  models returns **zero**. Coqui sentence-splits and synthesises whole segments.
  Both are PyTorch-only with `librosa==0.9.1` and `transformers==4.27.4` pinned.
  **Stop looking.** We want prosodic response, not speech, so the relevant
  lesson is MeloTTS's stochastic duration predictor as a prosody model, not its
  audio path.
- **Do not build a node-graph DSP engine.** glicol needed `StableGraph` +
  persistent `DfsPostOrder` + `NodeData<T,N>` + a 1024-node capacity story, and
  its sequencer is still **O(events × samples)** with a per-sample modulo scan
  (`seq.rs:45-65`). It also allocates inside its own audio callback —
  `glicol-engine.js:143-173`, six allocations plus four `slice()` copies every
  128 frames, and an `alloc(128)` at `:143` that is never freed.
- **Do not vendor the best architectures.** The two best ring-buffer/scheduler
  designs are **GPL-3.0** (SuperCollider) and **AGPL-3.0** (Sonic Pi). Every
  mechanism worth taking is 30–60 lines and reimplementable:
  - `MsgFifo.h:84-117` — a lock-free SPSC ring where `Write` **returns false when
    full** and never blocks or allocates. That one line is the whole
    anti-dropout story.
  - `PriorityQueue.h:31-121` — fixed-capacity event heap with a **stability
    counter** so equal timestamps keep FIFO order. Two mature projects needed
    that tiebreak independently.
  - `scheduler.rb:417` — **publish your own next deadline**,
    `min` over all pending timers. This is our own T-minus timing law in one line.
  - `event_history.rb:178-209` — a bounded time-indexed "value at t?" store,
    MIT, exactly what a 9-track score needs. **Do not copy line 412**, which
    evicts on wall clock while events carry logical time.
- **pyannote: reject structurally, not practically.** Diarization needs
  clustering, which is order-dependent and non-incremental — you cannot decide
  two regions are the same speaker until you have seen both. That forces
  unbounded state on a stream whose central property is that `push()` discards
  everything but the tail.
- **openWakeWord: reject the weights, copy the design.** Code Apache-2.0 but
  **pre-trained models are CC BY-NC-SA 4.0 — non-commercial.** Retraining needs
  ~30,000 hours of negative data, which is the real blocker, not the compute.
  And we do not need a wake word yet: a VAD answers "did someone speak", a wake
  word answers "did someone say *the thing*". Different products; the second is
  stage 2.

## 5. Audio generation and STT

**FunASR is the one serious repo.** 121 test files, 19 CI workflows including a
real-time benchmark, and real `|| true`-free enforcement.

- **2pass ASR**: a streaming model for latency plus an offline model that
  corrects at *end of sentence*. This is exactly our "cheap first, verify after"
  shape, and it is the mechanism that makes an aggressive trigger safe.
- **`chunk_size` as an explicit latency dial** — `[5,10,5]` is a 600 ms window,
  300 ms lookback, 300 ms lookahead. A flag, not a code change.

**A methodological warning that outranks both:** RTF **swings 16× from the
serving stack alone** (FunASR's `RTFx 340` under vLLM vs `21` under PyTorch).
Any RTF figure quoted without its runtime is noise. **A reported RTF is not a
result until the serving stack is named.**

VoxCPM's honest number: **RTF 1.76 on an M4 Pro with Metal** — slower than real
time, on good hardware. The 4090 figures it leads with are irrelevant to us.
Its repo also has **1,485 lines of tests that CI never runs** — CI that exists,
is nontrivial, and cannot catch a regression because it has no test step. That is
a cousin of `|| true`, and worth remembering as a failure mode.

## 6. Jev as a research instrument — what it is and is not

Both research lanes used it as a tool while doing the research, and both
reported failures. Consolidated:

**Worked.**
- **Screening candidates against typed criteria** where the state contained real
  measurements. Its ranking agreed with the lane's own benchmark.
- **Reproducing the holistic-judge inversion** on a fresh question set.
- **Refusing to count.** Asked to count, it declined rather than guessing — which
  is the useful behaviour, and matches the vendor's own "count in code".

**Did not work.**
- **Every aggregator we built on top of it was worse than the rows underneath.**
  Good for one decomposed table in a single call; bad for verdicts. This is the
  single most consistent finding across both lanes.
- **It has no access to mechanism.** Asked whether three prosody knobs were
  separable and realtime-safe, it scored all three *identically* — spread
  exactly **0.000** on both substantive axes — because the descriptions read
  alike.
- **`Noul` has no `confidence` and no `probabilities` field at all.** The
  arithmetic finding covers `Choice` only, and the `Noul` gate is the one we use
  most.
- **The position control did not reproduce** with distinguishable options
  (7/7 picks travelled), and reproduced only partially with indistinguishable
  ones (forward 9/9 first, reversed 6/9). The honest reading is **instability**,
  not a clean positional bias.

**Verdict:** useful as a fast screen over a small set of options with measured
numbers in the state; not usable as an oracle, not usable for aggregation, and
never to be trusted for mechanism. This matches what `src/policy.py` already
encforces.

## 7. Ranked next actions

| # | action | cost | why now |
|---|---|---|---|
| 1 | **silero-vad ONNX gate in front of the watcher** | ~1 day | removes a total failure (hum) and 39% of a core |
| 2 | **Correct `src/watcher.py`'s docstring** | 5 min | it claims microseconds; it is 39% of a core |
| 3 | **Split our 150 ms into a 200 ms trigger and a 3 s decision** | 2 h | matches measured practice; removes 20× over-eagerness |
| 4 | **Locally adaptive median threshold** (reimplement `peakpicker.c:110-117`) | 3 h | file-global thresholds cannot survive a night |
| 5 | **Test the 3 s decision against a waveform-only classifier** (smart-turn v3.2, BSD-2, bundled) | 2 days | the measured 63–79 ms and the silence finding |
| 6 | Speculation gate (OPEN/HOLDING/DROPPING) | 3 days | what makes 5 defensible |
| 7 | FunASR 2pass for the cold path | 1 week | only after the hot path is cheap |
| 8 | Lock-free SPSC ring + fixed-capacity event heap | 1 week | reimplement; GPL/AGPL means reimplement |

## 8. What would most change our minds

A **held-out-device** measurement. Every number in this document comes from a
24-core x86 desktop. The literature's hardest axis is device generalisation
(DCASE 2025: city generalisation is ~1 pp held out; device generalisation drops
S1 62.8% → S6 42.6%), and we have never measured it, because everything so far
is synthetic or one machine. If silero holds at 8× pessimism and our watcher
does not, the decision is made for us. If both hold, the gate is still worth it
for the hum failure alone — which is a correctness argument, not a speed one.