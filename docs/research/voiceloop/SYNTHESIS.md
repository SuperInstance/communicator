# Synthesis — how should a peripheral decide end-of-turn and handle interruption?

**The question:** our watcher segments on energy with a **150 ms silence
threshold**, and we know that is naive. What replaces it?

**The one-line answer:** stop using one threshold for two jobs. pipecat uses
**~200 ms to start looking and 3 s to give up waiting** — a ~15× ratio — and
then runs a **binary classifier over the raw waveform with no transcript in the
loop**. Adopt the two-stage shape, steal the detector, and put a
**speculative commit/rollback gate** in front of the output so the trigger can
be aggressive without the system ever acting on a half-finished thought.

---

## 1. Why 150 ms is wrong — three distinct reasons

**(a) One number is doing two jobs.** pipecat splits them explicitly:
`VAD_STOP_SECS = 0.2` in `audio/vad/vad_analyzer.py:27` fires the trigger;
`STOP_SECS = 3` in `audio/turn/smart_turn/base_smart_turn.py:27` bounds the
semantic wait. 150 ms as a *trigger* is roughly right — it is within 50 ms of
theirs. 150 ms as a *decision* is 20× too eager. We have been treating the
trigger value as if it were the decision value.

**(b) Silence duration is not a completion signal.** `base_smart_turn.py:128-138`
shows what silence can legitimately mean: only "give up and act". The actual
completion signal is a classifier on the audio:
`_predict_endpoint(audio_array) -> {prediction: 0|1, probability: float}`
(`base_smart_turn.py:279-282`). A fixed silence interval cannot distinguish
"finished a clause" from "thinking of the next word", and no tuning of a single
scalar fixes that.

**(c) It also breaks downstream arithmetic.** pipecat explicitly subtracts the
VAD's own silence window when computing a downstream deadline
(`turn_analyzer_user_turn_stop_strategy.py:236`,
`stt_deadline = frame.timestamp - frame.stop_secs + self._stt_timeout`), because
the VAD reports "stopped speaking" only `stop_secs` late. Any wait we add after
our energy gate will be double-counted unless we do the same subtraction. Their
code warns at runtime if `stop_secs` drifts (`:247-262`), because their
published numbers are conditional on it.

---

## 2. The recommended architecture

```
audio in
   │
   ├─► [STAGE 1] energy/VAD gate ── stop_secs ~0.15-0.25 s
   │      cheap, runs always, decides only "is there speech to look at"
   │      ~200 ms in pipecat (vad_analyzer.py:27); ours at 150 ms is fine HERE
   │
   ├─► [STAGE 2] semantic completion model ── bounded wait ~0.6-3.0 s
   │      binary classifier on log-mel of raw waveform, NO transcript
   │      hard cap on how long we will wait for a confident answer
   │
   ├─► [STAGE 3] speculation gate ── OPEN / HOLDING / DROPPING
   │      compute the response eagerly, buffer the output, commit or discard
   │
   └─► MIDI CC out (only after the gate opens)
```

Three properties this buys that a single threshold cannot express:

1. **Fast trigger, safe output.** Stage 1 can stay at ~150-200 ms — reaction
   stays fast — because Stage 3 means an early wrong guess costs compute, not a
   wrong gesture.
2. **Level invariance.** Normalising the waveform before the spectrogram
   (`_whisper_features.py:177`) makes the decision independent of how loud the
   room is. An energy threshold is a threshold on absolute level.
3. **No transcript.** We produce music, so the transcript was never needed.
   pipecat has a switch for exactly this — `wait_for_transcript=False`,
   documented at `turn_analyzer_user_turn_stop_strategy.py:36-45` as putting
   "transcripts off the latency critical path". Run in that mode.

### Concrete recommendation

| parameter | today | recommended | basis |
|---|---|---|---|
| trigger silence | 150 ms | **keep 150-200 ms** | `VAD_STOP_SECS=0.2` (`vad_analyzer.py:27`) |
| decision wait | *(same 150 ms)* | **600 ms minimum floor**, adaptive | below |
| give-up cap | — | **3 s**, then force `COMPLETE` | `STOP_SECS=3` (`base_smart_turn.py:27`) |
| decision signal | energy | **log-mel waveform classifier** | `local_smart_turn_v3.py:138-183` |
| failure direction | unknown | **fail toward NOT acting** | invert `base_smart_turn.py:267-271` |

**On the two failure directions, be deliberate.** pipecat fails *open* — a
model timeout yields `COMPLETE` (`:267-271`) — because for a phone bot, silence
is expensive and talking over a pause is cheap. **For a peripheral it is the
reverse:** an unwanted musical gesture is intrusive and cannot be taken back.
Default to `INCOMPLETE` on timeout and keep waiting, up to the 3 s cap. This is
a one-line change to their code and it is the single most important
adaptation.

### Why adaptive beats a fixed 600 ms

A fixed floor is still a fixed floor. `smart_turn.py` and the VAD take a
`tts_p99_latency`-style bound — pipecat budgets the **P99**, not the mean
(`turn_analyzer_user_turn_stop_strategy.py:236`) — and tracks
`pre_speech_ms` plus the VAD's own trigger time to adapt the window to the
signal (`base_smart_turn.py:215`). Do the same for silence: track the
distribution of *this speaker's* trailing silences and set the floor from it.
Human thinking-silence is person-specific and state-dependent; a global
constant cannot be right for both a focused 45-minute coding block and a
tired end-of-day aside.

**Jev's read agrees on the direction and not the mechanism** (see
[JEV-AS-INSTRUMENT.md](JEV-AS-INSTRUMENT.md) §E7): it chose a long fixed
threshold (600 ms, p=0.85) over the short one because the short threshold scored
0.92 on "will this cut the human off mid-thought". But the adaptive option was
only ever offered as a Choice *label*, never as a framing of the symptoms, so
that comparison is untested — I would still choose adaptive, on the source
reading.

---

## 3. Interruption / barge-in

pipecat's mechanism is small and worth copying exactly
(`processors/frame_processor.py:1152-1171`):

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

**Cancel the in-flight work and flush the queue** — do not wait for a graceful
finish. Per-frame `interruptible` flags let a critical frame finish while
everything queued behind it is discarded. Their comment at `:1163-1169` records
a real bug this design fixed: slow non-interruptible frames used to block
interruptions entirely.

For a peripheral: if the human resumes mid-phrase, **truncate the phrase and
re-evaluate**, with the already-emitted prefix staying emitted and the
uncommitted remainder dropped. Treat a released MIDI CC as committed output
(cannot be recalled) and everything still buffered as discardable — which is
exactly the Stage 3 gate's job.

---

## 4. The top 5 stealable mechanisms overall

### 1. Speculative commit/rollback gate — *`BSD-2-Clause`* · **`turns/speculation_gate.py`, `turns/eager_end_of_turn_mixin.py`**

```python
class SpeculationState(Enum):
    OPEN = "open"          # emitting everything
    HOLDING = "holding"    # holding a speculative response until confirmed
    DROPPING = "dropping"  # discarding the rest of a withdrawn speculation
```

Three-call lifecycle (`eager_end_of_turn_mixin.py:38-52`):
`_push_eager_end_of_turn()` → `_cancel_eager_end_of_turn()` → `_clear_eager_end_of_turn()`.
All three are safe when nothing is outstanding and inert when the feature is
off. Only one speculation is ever in flight, so frames need not name which
(`speculation_gate.py:57-58`).

**Why #1 for a peripheral:** it decouples reaction speed from output safety.
Compute early, buffer the output, commit or discard. Wrong guesses cost
milliseconds, never correctness. This is the mechanism that makes an aggressive
trigger defensible.

### 2. Waveform-only end-of-turn classifier — *model 8.3 MB, code `BSD-2-Clause`* · **`audio/turn/smart_turn/`**

Binary classifier on Whisper-style log-mel, **no STT in the loop**
(`base_smart_turn.py:279-282`). Measured here across two runs: **6.5–7.8 ms
features + 61–73 ms inference = 69–79 ms total** (p95 up to 99 ms),
single-threaded, ONNX, and the model **ships inside the pip package** at
8,679,182 bytes.

**Critical caveat from my own measurements** (`bench_smartturn.py`): on
**digital silence the model outputs 0.987 — near-maximal "turn complete"**, and
on noise/tone 0.96. It is not a silence detector and **must be VAD-gated**. Feed
it raw audio and it will agree with you every time.

### 3. Pure-numpy Whisper log-mel extractor — *`Apache-2.0`-derived (NOTICE in file)* · **`audio/turn/smart_turn/_whisper_features.py`**

Vendored from HuggingFace so the detector needs no `torch`/`torchaudio`/
`librosa`/`scipy`. **Runs and is benchmarked in our environment with all four
absent.** The specific trick:

```python
windows = sliding_window_view(padded, frame_length)[::hop_length]
# One batched rFFT replaces ~800 Python-level calls into numpy.
spec = np.fft.rfft(windows * win, axis=-1)
```

Plus normalise-before-spectrogram (`:177`) for level invariance and a
`log_spec.max() - 8.0` dynamic-range clamp (`:183`). Directly reusable for any
audio-feature work we do, regardless of the turn-taking decision.

### 4. Latency decomposed into service time + padding you chose — *`BSD-2-Clause`* · **`metrics/metrics.py:41-97`**

```python
class TTFAMetricsData(MetricsData):
    ttfa: float; ttfb: float; leading_silence: float
class TTFATMetricsData(MetricsData):
    ttfat: float; ttfb: float; thinking_time: float
```

> so consumers can see how much of the perceived latency is silence padding
> rather than service response time

Most perceived latency in a voice system is **the silence the system inserted to
be sure**, not compute. If we log one "reaction latency" number we cannot
distinguish a slow decision from a cautious threshold — and those have opposite
fixes. Log `decision_latency` and `silence_waited` separately. Also adopt their
`TurnMetricsData` / `SmartTurnMetricsData` shape (`:216-247`) which carries
`e2e_processing_time_ms` alongside the `probability`.

### 5. Typed decision model as a first-class pipeline citizen — *`BSD-2-Clause`* · **`classifiers/`**

pipecat ships `classifiers/typesafe/jev/` (TypeSafe System One), plus
`cloudflare/clef`, `openai/decisions`, `llm`, and `jev` backends behind one
interface (`base_classifier.py:44-84`):

```python
class YesNoQuestion(BaseModel):  instructions; yes; no
class ChoiceQuestion(BaseModel): instructions; options: dict[str, str|...]
class ScoreQuestion(BaseModel):  instructions; levels: list  # min_length=2
```

So our unusual shape — probabilities out, no text generation — is a
**mainstream, multi-vendor integration category**, not a workaround.

The interface deliberately keeps the threshold with the caller
(`base_classifier.py:144-151`):

> Callers that need more certainty than that compare `probability` with a
> threshold of their own.

**The model supplies evidence; the peripheral supplies the cost of being wrong.**
That is the correct division for us and it is what Stage 2 should follow.

Two integration shapes exist and the distinction is useful: **classifiers do not
sit in the pipeline** — "no frames flow into it" (`base_classifier.py:10`) —
they are plain awaited objects, whereas the **pipeline** is a chain of
`process_frame(frame, direction)` processors (`pipeline/pipeline.py:186-198`)
with `enable_direct_mode` for latency. Use the classifier shape for the
decision; keep the pipeline shape for audio.

**Caveat on the vendor claim:** `JevClassifier`'s docstring asserts "Jev's
probabilities are calibrated" (`:52-53`). **Unverified** — no calibration
harness, ECE, or reliability curve exists in the repo. What I did verify is the
narrower, checkable claim in §E1 of the Jev doc: `confidence` is exactly
`(p_max − 1/K)/(1 − 1/K)`, arithmetic on the distribution.

---

## 5. Runner-up: three prosody scalars (MeloTTS, *MIT*)

`api.py:83` — `noise_scale` (0.6), `noise_scale_w` (0.8), `sdp_ratio` (0.2).

Steal the **vocabulary**, not the runtime:

- `noise_scale` → how much output varies event to event
- `noise_scale_w` → how much timing varies
- `sdp_ratio` → **0 = deterministic durations**, a legitimate mechanical mode,
  not just a degenerate one

MIT (`LICENSE:1-3`, `README.md:58`) so freely usable. But they are per-utterance
draws through a flow with internal state — **not real-time modulators**, and
the `noise_scale=0.667` default means consecutive utterances differ unless
pinned. And MeloTTS is PyTorch-only: no numpy/ONNX path exists.

---

## 6. Licensing summary for anything we adopt

| thing | licence | commercial? | evidence |
|---|---|---|---|
| pipecat turn-taking + gates + classifiers | **BSD-2-Clause** | **yes** | `licenseInfo`; `base_turn_analyzer.py:4` |
| smart-turn v3.2 ONNX weights | shipped in pipecat, BSD-2 | **yes** | 8,679,182 B, in-package |
| `_whisper_features.py` | **Apache-2.0**-derived w/ NOTICE | **yes** | `_whisper_features.py:6-15` |
| MeloTTS code + weights | **MIT** | **yes** | `LICENSE:1-3`, `README.md:58` |
| Coqui TTS code | **MPL-2.0** | **yes** (weak copyleft) | `LICENSE.txt:1-2` |
| Coqui models (Apache-2.0 ×36, BSD-3 ×24, MIT ×11, MPL ×7, CC-BY-SA ×8, CC-BY ×3) | permissive | **yes** | `TTS/.models.json` |
| **Coqui `xtts_v2` / `xtts_v1.1`** | **CPML** | **NO** — non-commercial **model and outputs** | `.models.json`, `tts-hub.github.io/cpml` |
| **Coqui `your_tts`, `fast_pitch`** | **CC BY-NC-ND 4.0** | **NO** — non-commercial, no derivatives | `.models.json` |

Only 2 of 95 Coqui models are CPML. The brief's "Coqui is non-commercial"
premise applies to the flagship XTTS models, not the repo (MPL-2.0) and not the
other 93 models.

> **CORRECTION (2026-10-19).** "the other 93 models" was a *not-CPML* count and
> read as though all 93 were fine to use. The manifest gives **89 of 95**
> commercially permissive (4 non-commercial: XTTS v1.1, XTTS v2, and two
> CC BY-NC-ND models; 2 restricted or unverified) — see
> [README.md](README.md) §licences and `TTS/.models.json`. The not-CPML count
> is still 93; the permissive count is 89. `NOTES-AUDIT-2.md` F9, F10.

**Maintenance note:** use `idiap/coqui-ai-TTS` (pushed today, 2026-10-18), not
`coqui-ai/TTS` (last push 2024-08-16). Idiap is the live continuation.

---

## 7. What I would NOT build

- **A per-model licence check at download time is worth adding; a Coqui
  dependency is not.** It is PyTorch-only, dormant, and its best-known model is
  the one we cannot use. The licence-manifest *pattern* is the takeaway.
- **Do not treat the VITS knobs as a real-time modulator** — per-utterance
  state, and the defaults deliberately randomise between takes.
- **Do not look for streaming synthesis in either TTS repo.** Zero `yield` or
  `stream` in MeloTTS; sentence-split-then-whole-segment in Coqui. If
  incremental audio is ever needed, look at neural vocoders (HiFi-GAN chunked
  inference, Vocos), not these text-to-waveform stacks.

---

## 8. Confidence and gaps

**Solid (measured or read in source):** the two-threshold split, the waveform-
only decision, the 8.3 MB model size, the 69–79 ms median latency,
latency, the silence-scores-0.987 behaviour, the `OPEN/HOLDING/DROPPING` gate,
the task-cancellation barge-in, the TTFA decomposition, every licence row,
MeloTTS having no streaming, both TTS repos requiring PyTorch.

**Weak / unverified:**
- My speech-like test signal is a synthetic formant stack, **not real speech**.
  The 0.05-vs-0.90 discrimination is directionally right and sufficient to
  justify the architecture, but the model's real accuracy is unmeasured.
- Latency is **this machine, single-threaded, two runs** (69–79 ms total,
  ±15% run-to-run). A tuned `intra_op_num_threads` or a faster CPU will move it;
  treat ~70 ms as an order of magnitude.
- pipecat publishes **no** latency figures at all — I had to build the
  benchmark. There is no upstream number to check mine against.
- Not verified: whether pipecat's 18 turn tests actually run in CI on every
  push (I did not read `.github/workflows/`).
- Coqui ONNX browser-inference artefacts have licences outside `.models.json`
  and are unverified.
- The 600 ms adaptive floor is **my synthesis**, not a number I found in any of
  these repos. pipecat's own numbers are 200 ms trigger / 3 s cap.