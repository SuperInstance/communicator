# pipecat — voice agent control loops and turn-taking

- Repo: `pipecat-ai/pipecat`, default branch **`main`**, commit `e1d5c82`
- License: **BSD-2-Clause** (verified via GitHub `licenseInfo`; `main:src/pipecat/audio/turn/base_turn_analyzer.py:4` carries `SPDX-License-Identifier: BSD 2-Clause License`)
- 16,572 stars, **pushed 2026-10-18** — actively developed, not a stale snapshot
- Accessed via `gh api graphql` (REST was exhausted) and `raw.githubusercontent`

## Verdict up front

**This is the single most valuable repo in the lane, and not for the reason I
expected.** I expected a voice-bot framework whose turn-taking was a VAD knob.
What is actually there is a **two-stage detector where the second stage is a
binary classifier over raw audio with no transcript in the loop**, plus a
**speculative commit/rollback gate**, plus a first-class **TypeSafe/Jev**
classifier integration. All three map onto the peripheral problem almost
directly.

The headline correction to our own design: **pipecat uses two different silence
numbers, not one.** 200 ms to *trigger*, 3 s to *give up waiting*. Our single
150 ms is being asked to do both jobs.

---

## 1. Turn-taking: two-stage, VAD-triggered + semantic

`BaseTurnAnalyzer` (`audio/turn/base_turn_analyzer.py`) defines the contract.
The signature that matters:

```python
# base_turn_analyzer.py:97
def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState: ...
# base_turn_analyzer.py:110
async def analyze_end_of_turn(self) -> tuple[EndOfTurnState, MetricsData | None]: ...
```

`is_speech` is a **VAD boolean supplied from outside** (`:97`). The analyzer
does not detect voice activity itself. Feeding and deciding are separate calls,
so the expensive model runs only after something has already decided there is
speech.

The only two states are `COMPLETE` / `INCOMPLETE` (`:29-30`).

### The two thresholds — this is the finding

```python
# audio/turn/smart_turn/base_smart_turn.py:27-29
STOP_SECS = 3
PRE_SPEECH_MS = 500
MAX_DURATION_SECONDS = 8
```

and the fallback logic:

```python
# base_smart_turn.py:128-138
else:
    if self._speech_triggered:
        chunk_duration_ms = len(audio_int16) / (self._sample_rate / 1000)
        self._silence_ms += chunk_duration_ms
        # If silence exceeds threshold, mark end of turn
        if self._silence_ms >= self._stop_ms:
            state = EndOfTurnState.COMPLETE
```

So: **3 seconds** of accumulated silence *after speech was triggered* forces
`COMPLETE`, regardless of what the model thinks. Meanwhile the VAD layer's own
gate is far shorter — `audio/vad/vad_analyzer.py:25-28`:

```python
VAD_CONFIDENCE = 0.7
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.2
VAD_MIN_VOLUME = 0.6
```

**0.2 s to fire, 3 s to surrender.** A ~15× ratio. Our 150 ms is doing both
jobs with one number, which is exactly why it is naive: it is simultaneously far
too eager to *start looking* and far too impatient to *stop looking*.

There is a third safety net. If the model itself times out, the analyzer fails
**toward acting**:

```python
# base_smart_turn.py:267-271
except SmartTurnTimeoutException:
    state = EndOfTurnState.COMPLETE
```

Fail-open to `COMPLETE`. Worth noting as a policy choice: for a peripheral that
must almost never interrupt you would want the opposite default, and this is a
one-line inversion.

### No transcript in the loop

The prediction is made on the **waveform**:

```python
# base_smart_turn.py:279-282
@abstractmethod
def _predict_endpoint(self, audio_array: np.ndarray) -> dict[str, Any]:
```

`dict` with `prediction: 0|1` and `probability: float`. Nothing is transcribed
before the end-of-turn decision. **A peripheral that emits music has no need for
a transcript at all, and pipecat has already proven the waveform-only path
works.** This is the strongest single reason to steal their architecture.

### Buffering details worth copying

- 500 ms of **pre-speech** audio is included (`PRE_SPEECH_MS`, `:28`), giving
  the model the consonant onset — a turn that starts with a fricative is
  otherwise truncated.
- `effective_pre_speech_ms = pre_speech_ms + vad_start_secs*1000` (`:215`) —
  the pre-roll is extended by however long the VAD itself took to trigger.
- Segment capped at 8 s keeping the **tail** (`:233-236`), because the end of
  the utterance is what carries the completion signal.
- Unbounded-growth guard before speech starts (`:140-150`).
- Raw int16 kept as a view and converted once per turn, not per ~20 ms frame
  (`:110-118`) — a real optimisation with a good comment explaining why it is
  safe.

---

## 2. The model: smart-turn v3.2 CPU, ONNX, **8.3 MB**

```python
# audio/turn/smart_turn/local_smart_turn_v3.py:48-51
if not smart_turn_model_path:
    model_name = "smart-turn-v3.2-cpu.onnx"
    package_path = "pipecat.audio.turn.smart_turn.data"
```

**The model ships inside the pip package.** I measured the file:
**8,679,182 bytes (8.3 MB)**. Session options are CPU-tuned
(`local_smart_turn_v3.py:70-74`): `ORT_SEQUENTIAL`, `inter_op_num_threads=1`,
`intra_op_num_threads=cpu_count`, `ORT_ENABLE_ALL`.

Fixed input window, zero-padded at the head so the model always sees exactly
8 s (`local_smart_turn_v3.py:141-152`, `n_seconds=8`):

```python
elif len(audio_array) < max_samples:
    padding = max_samples - len(audio_array)
    return np.pad(audio_array, (padding, 0), mode="constant", constant_values=0)
```

Threshold is hardcoded (`local_smart_turn_v3.py:174`):

```python
prediction = 1 if probability > 0.5 else 0
```

Older v2 (`local_smart_turn_v2.py:110`) uses the same `> 0.5`.

### I ran it. Here are the numbers.

`bench_smartturn.py` in this directory, on this machine, single-threaded.
**Two independent runs**, because the run-to-run spread is itself a finding:

| stage | run 1 median | run 2 median | p95 (runs 1/2) |
|---|---|---|---|
| log-mel features (pure numpy) | 7.78 ms | 6.52 ms | 13.38 / 10.98 ms |
| ONNX inference | 60.88 ms | 72.87 ms | 89.75 / 99.19 ms |
| **total** | **68.66 ms** | **79.39 ms** | — |

**These are measured, not quoted, and the ±15% run-to-run spread on a quiet
machine is why I report a range.** Behaviour numbers were bit-identical across
runs; only timing varied. pipecat publishes **no** latency figures anywhere I
could find — the README says only "Ultra-low latency" (`README.md:29`) with no
number. So there was no benchmark to find; I made one.

Behaviour, `P(complete)`:

| input | P(complete) | prediction |
|---|---|---|
| speech-like, 4 syllables | **0.0495** | INCOMPLETE |
| speech-like, 1 syllable then silence | 0.8972 | COMPLETE |
| white noise | 0.9619 | COMPLETE |
| 220 Hz sine tone | 0.9574 | COMPLETE |
| **digital silence** | **0.987** | **COMPLETE** |

Two conclusions:

1. The model **does** discriminate speech from non-speech in the right
   direction (0.05 for continuing speech vs 0.90 for a finished one).
2. **Digital silence scores 0.987 — near-maximal confidence that the turn is
   over.** The model is *not* a silence detector and must never be fed silence
   directly. This is precisely why the architecture is VAD-gated: `_process_speech_segment`
   returns early on an empty buffer (`base_smart_turn.py:211-212`) and
   `append_audio` only accumulates after `is_speech`.

**Anyone lifting this model must gate it on VAD first.** Feeding it raw audio
gives you "the user finished" every time.

A second observation from the numbers: the answer depends strongly on the
**trailing** content of the window. One synthetic syllable followed by silence
reads 0.90; the same syllable with more speech after it reads low. That is the
model doing its job, but it means the window's end alignment is load-bearing —
you cannot slide an arbitrary 8 s window.

---

## 3. The feature extractor is pure numpy — and that answers our environment question

`_whisper_features.py` is a vendored, numpy-only Whisper log-mel extractor
(Apache-2.0 derived from HuggingFace, noted at `:6-15`). **torch, torchaudio,
librosa and scipy are all absent in my environment and the file still runs.**

Techniques worth stealing:

```python
# _whisper_features.py:126-129
windows = sliding_window_view(padded, frame_length)[::hop_length]
# One batched rFFT replaces ~800 Python-level calls into numpy.
spec = np.fft.rfft(windows * win, axis=-1)
```

`sliding_window_view` gives a zero-copy strided view, then one batched rFFT
replaces the per-frame loop. That single change is most of the feature cost.

```python
# _whisper_features.py:177
x = (x - x.mean()) / np.sqrt(x.var() + _NORM_VARIANCE_EPS)
```

Waveform normalised **before** the spectrogram — this is what makes the model
level-invariant, and it is why silence behaves as it does.

```python
# _whisper_features.py:183-184
log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
log_spec = (log_spec + 4.0) / 4.0
```

Dynamic-range clamp then rescale into roughly `[-1,1]`.

Output is fixed `(80, 800)` float32 (`:158-160`), matching the ONNX input
`[batch, 80, 800]` (confirmed from the loaded session).

---

## 4. Interruption / barge-in: task cancellation + queue flush

`processors/frame_processor.py:1152-1171`:

```python
async def _start_interruption(self):
    current = self.__process_current_frame
    current_is_uninterruptible = current is not None and not current.interruptible
    if current_is_uninterruptible:
        self.__reset_process_queue()
    else:
        await self.__cancel_process_task()
        self.__create_process_task()
```

The mechanism: **cancel the in-flight asyncio task and recreate it, flushing the
queue.** Frames carry an `interruptible` flag; an uninterruptible frame already
being processed is *not* cancelled, but queued interruptible frames are dropped
anyway.

The comment at `:1163-1169` is worth reading as a design note — it records a
real bug: skipping the cancel branch when the queue held an uninterruptible
frame caused "slow non-uninterruptible frames to block interruptions". The fix
is that cancelling is safe because task creation resets the queue while
preserving uninterruptible frames.

For a peripheral: interruption is *drop the pending output and reset*, not *wait
for a graceful finish*. If we are mid-way through a musical phrase and the human
resumes, the right move is to truncate and re-evaluate.

---

## 5. The stealable mechanism: speculate, then commit or roll back

`turns/speculation_gate.py`:

```python
class SpeculationState(Enum):
    OPEN = "open"          # emitting everything
    HOLDING = "holding"    # holding a speculative response until confirmed
    DROPPING = "dropping"  # discarding the rest of a withdrawn speculation
```

From the module docstring (`:44-53`):

> A speculative response is generated from an eager end of turn — a provisional
> guess that the user has finished talking — so it may answer a transcript the
> user never actually completed. This holds everything such a response produces
> until the turn is confirmed, then releases it, or discards it if the guess is
> withdrawn.

And the lifecycle it drives (`turns/eager_end_of_turn_mixin.py:38-52`) — three
calls, all safe when nothing is outstanding and all inert when the feature is
off:

```
_push_eager_end_of_turn()   # predict the turn has ended
_cancel_eager_end_of_turn() # user resumed speaking, prediction void
_clear_eager_end_of_turn()  # turn committed, prediction resolved
```

**Why this is the top mechanism for a peripheral.** Our outputs are MIDI CC
values — cheap to regenerate, and we cannot "un-send" a chord that has already
left the speaker. The fix is identical: compute the response speculatively as
soon as the trigger fires, **buffer the output frames, and only emit them once
the turn is confirmed**. Wrong guesses cost compute, not correctness. This lets
us keep the trigger aggressive (fast reaction) while keeping the output safe
(never act on a half-finished thought) — which is precisely the tension a
single threshold cannot express.

Note also: only one speculation is ever in flight (`:57-58`), so the frames do
not need to name which speculation they mean. That is a genuinely simplifying
design decision.

---

## 6. Latency budget: they separate *your padding* from *their compute*

`metrics/metrics.py`:

```python
# metrics.py:41-61
class TTFAMetricsData(MetricsData):
    """Time To First Audio (TTFA) ... i.e. time-to-first-byte plus any leading
    silence the service pads onto the start of its response. ``ttfa`` is
    reported with its breakdown so consumers can see how much of the perceived
    latency is silence padding rather than service response time..."""
    ttfa: float
    ttfb: float
    leading_silence: float
```

The same split for LLM reasoning time (`TTFATMetricsData`, `metrics.py:64-97`:
`ttfat = ttfb + thinking_time`), plus `TextAggregationMetricsData`
(`:203-213`) explicitly for "the latency cost of sentence aggregation in the TTS
pipeline", and `TurnMetricsData` / `SmartTurnMetricsData` (`:216-247`) carrying
`e2e_processing_time_ms`, `inference_time_ms`, `server_total_time_ms`.

**The idea to steal: report perceived latency decomposed into service time plus
the padding you chose.** Most of a voice agent's felt latency is the silence it
inserted waiting to be sure, not model compute. If we log only "reaction
latency" we cannot tell a slow decision from a cautious threshold. We should
log both.

### The double-counting bug they explicitly guard against

`turns/user_stop/turn_analyzer_user_turn_stop_strategy.py:225-236`:

```python
self._stop_secs = frame.stop_secs
...
# Anchoring the ...
stt_deadline = frame.timestamp - frame.stop_secs + self._stt_timeout
```

and again at `:302`:

```python
timeout = max(0, self._stt_timeout - self._stop_secs)
```

**The VAD only reports "stopped speaking" `stop_secs` after the speech actually
stopped.** So a naive `now + stt_timeout` deadline pays the silence window and
the STT wait back to back. They subtract it. If we ever add any downstream wait
after our energy gate, this is the bug we will hit — and their own warning at
`:247-262` fires if `stop_secs` drifts from the tuned default, because their
latency numbers are conditional on it.

### P99 as the design bound

`:236` uses `self._stt_timeout`, populated from the **P99** STT latency reported
in `STTMetadataFrame`, as the safety-net timeout. Budget the tail, not the mean.

### Getting STT off the critical path entirely

`turn_analyzer_user_turn_stop_strategy.py:36-45` documents `wait_for_transcript`:

> Set this to False when local turn detection is the intended driver of the
> conversation (e.g. with a realtime LLM service consuming audio directly), so
> transcripts are off the latency critical path.

**This is the mode a peripheral should run in.** We have no transcript and need
none; the waveform model decides alone. pipecat ships the switch; we should copy
the setting.

---

## 7. Plugging a typed decision model into the loop — **already done**

This was an open question in the brief. The answer is better than expected:
**pipecat ships a first-class TypeSafe/Jev classifier.**

`classifiers/base_classifier.py:8-11` states the integration contract:

> A classifier is a plain object. Whoever needs answers creates one, keeps it,
> and calls it. **It does not sit in a pipeline and no frames flow into it.**

So there are **two distinct shapes**, and the distinction is useful:

- **Pipeline** — `FrameProcessor` chain; each processor implements
  `process_frame(frame, direction)` (`pipeline/pipeline.py:186-198`), frames
  flow both ways, `enable_direct_mode=True` bypasses the queue for latency.
- **Classifier** — a plain awaited object. Not a frame processor.

The typed vocabulary maps almost one-to-one onto our questions:

```python
# classifiers/base_classifier.py:44-84
class YesNoQuestion(BaseModel):    instructions; yes; no
class ChoiceQuestion(BaseModel):   instructions; options: dict[str, str|...]
class ScoreQuestion(BaseModel):    instructions; levels: list  (min_length=2)
```

and results `YesNoResult{probability}`, `ChoiceResult{choice, probabilities,
confidence}`, `ScoreResult{score, levels, confidence}` (`:135-166`).

`JevClassifier` (`classifiers/typesafe/jev/classifier.py`) implements it against
the hosted model, with `JEV_MAX_CHOICE_OPTIONS = 255` (`:43`), mapping
`YesNoQuestion → {"type":"noul", "criteria":{"true":...,"false":...}}`
(`:172-174`) and `ChoiceQuestion → {"type":"choice","criteria":options}`
(`:181-185`).

Its docstring asserts, at `:52-53`:

> **Jev's probabilities are calibrated.**

I could not verify this in-repo — there is no calibration harness in the
repository, no ECE, no reliability curve. **Treat it as a vendor claim.** What
I *can* say from my own measurement is the narrower, checkable statement in
[JEV-AS-INSTRUMENT.md](JEV-AS-INSTRUMENT.md) §E1: `confidence` is exactly
`(p_max − 1/K)/(1 − 1/K)`, i.e. arithmetic on the distribution, which is
consistent with calibration but does not establish it.

**On thresholding**, `base_classifier.py:144-151`:

```python
@property
def is_yes(self) -> bool:
    """... Callers that need more certainty than that compare ``probability``
    with a threshold of their own."""
    return self.probability >= 0.5
```

The framework hands you a probability and pushes the threshold decision to the
caller. That is the correct stance for us: **the model supplies evidence, the
peripheral supplies the cost of being wrong.**

Other backends exist under `classifiers/`: `cloudflare/clef`, `llm`, `openai/decisions`,
`typesafe/jev`, `jev` (a 21-line shim). So a typed decision model is a
**first-class, multi-vendor category** in this framework, not an awkward
workaround.

---

## 8. Turn management as a composable strategy system

Beyond the analyzer, pipecat decomposes turn control into swappable strategies
(`turns/`):

```
turns/user_start/  vad | transcription | min_words | wake_phrase | external | krisp_viva_ip
turns/user_stop/   turn_analyzer | speech_timeout | llm_turn_completion | eager | deferred | external
turns/user_mute/   always | first_speech | function_call | mute_until_first_bot_complete
```

**Three genuinely different end-of-turn mechanisms coexist:**

1. **VAD/speech timeout** — `speech_timeout_user_turn_stop_strategy.py`
2. **Semantic waveform model** — `turn_analyzer_user_turn_stop_strategy.py`
3. **The LLM itself** — `llm_turn_completion_user_turn_stop_strategy.py`, which
   enables `filter_incomplete_user_turns` on the LLM and has the model emit a
   `●` completion marker. On `incomplete_short` / `incomplete_long` markers
   "the mixin re-prompts internally and no completion frame is emitted, so the
   public stop event stays deferred" (`llm_turn_completion_...py:36-39`).

That third one is an elegant trick: **an incomplete turn produces no visible
response at all**, because the model silently re-asks itself rather than
answering. A peripheral's analogue is obvious — if the decision is "not worth
reacting to", emit nothing rather than emitting a "no" gesture.

`deferred(...)` lets a detector strategy drive inference while leaving
finalization to another strategy (`llm_turn_completion_...py:41-45`) — separate
*when to compute* from *when to commit*, which is the same split as the
speculation gate.

---

## 9. Tests and CI

Turn-taking is genuinely tested — **18 turn-related test files**:

```
tests/test_base_smart_turn_buffer.py
tests/test_local_smart_turn_v3_features.py      <- tests the numpy extractor
tests/test_local_smart_turn_v3_resample.py
tests/test_user_turn_stop_strategy.py
tests/test_user_turn_start_strategy.py
tests/test_user_turn_controller.py
tests/test_user_turn_processor.py
tests/test_eager_end_of_turn_mixin.py
tests/test_user_turn_completion_mixin.py
tests/test_speech_timeout_user_turn_stop_strategy.py (implied by the group)
tests/test_turn_start_vad_state.py
tests/test_turn_trace_observer.py
tests/test_turn_tracking_observer.py
tests/test_empty_user_turn.py
tests/test_krisp_ip_user_turn_start_strategy.py
tests/test_cartesia_turns_stt.py
tests/test_azure_voice_live_text_turns.py
tests/test_wake_phrase_user_turn_start_strategy.py
tests/test_eager_user_turn_strategies.py
```

Notably `test_local_smart_turn_v3_features.py` — the pure-numpy feature
extractor is under test, which is why I was willing to depend on it.

**Not verified:** I did not confirm which CI jobs actually execute these on
every push. I did not inspect `.github/workflows/` (REST budget and time). The
existence of the test files is confirmed; their enforcement in CI is not. I
found no `|| true` guard in the files I read, but I did not read the workflows.

---

## 10. Source-vs-markdown discrepancies

| Markdown claim | Source reality |
|---|---|
| README:29 "Ultra-low latency interaction" | **No latency figure anywhere.** I had to build `bench_smartturn.py` to get 69–79 ms median. |
| `local_smart_turn_v3.py:170` "ONNX model returns sigmoid probabilities" | Output node is named **`logits`**; values observed in `[0,1]` (0.0495–0.987), so it is already squashed despite the name. Code uses it as a probability directly (`:174`). Comment is misleading, behaviour is fine. |
| `JevClassifier` docstring:52 "Jev's probabilities are calibrated" | No calibration harness, ECE, or reliability curve in the repo. Vendor claim, unverified. |
| README badge | Repo license is genuinely BSD-2-Clause — the one markdown claim that checks out. |

---

## 11. Environment feasibility for our box

| library | present here | needed by |
|---|---|---|
| `numpy` | **yes** | everything; the extractor is numpy-only |
| `onnxruntime` | **no** (I pip-installed it to measure) | smart-turn inference |
| `torch` / `torchaudio` / `librosa` / `scipy` | **no** | **not needed** for turn-taking |

The turn-taking path is **numpy + onnxruntime only**. That is a deliberate
property of the repo — pipecat vendored the feature extractor specifically so
this stays light — and it means we can adopt the whole detector without pulling
in the PyTorch stack that MeloTTS and Coqui both require.