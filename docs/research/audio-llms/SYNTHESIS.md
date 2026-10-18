# SYNTHESIS — across AudioGPT, VoxCPM, FunASR

*2026-10-18. Companion to `README.md` (learning journey) and the three repo
files. Written for the question we actually care about: what is worth taking
into a continuous, cheap, single-mic, WSL voice peripheral.*

---

## 1. The framing correction

We entered treating these as three points on one curve — token rate vs
parameter count vs benchmark score — and picking the one on our frontier.

They are not on one curve. They are **three categories**:

- **AudioGPT** — an *integration layer*. A filename-passing ReAct agent.
  Contains no audio-conditioned model and no original audio research.
- **VoxCPM** — a *single generative model*. One system, LLM + tokenizer +
  flow-matching decoder, solving one problem well.
- **FunASR** — *infrastructure*. Dozens of small models, a C++ runtime,
  bindings for five platforms, benchmark methodology, and CI that runs.

The category difference decides the outcome. Only two of the three contain
anything transferable, and the most valuable transfers are **not the models**.

## 2. What recurs across the repos

Three genuine patterns appear more than once, which raises their prior:

### (a) Small specialists beat one big generalist, for peripherals

FunASR's zoo is built from components measured in single-digit-millions to
low-hundreds-of-millions: `fsmn-vad` **0.4M**, `cam++` **7.2M**,
SenseVoiceSmall **234M**, Paraformer **220M**. VoxCPM's own tokenizer proves
the same point internally — one learned vector per patch compresses audio to a
few tokens (`local_encoder.py:11`). AudioGPT reaches for 1550M Whisper-class
models and a cloud LLM to do what FunASR does with 0.4M.

This matches our "runs small enough for CPU or edge is very interesting"
constraint exactly, and it is the strongest convergent signal in the pass.

### (b) Latency is an explicit, tunable parameter — in the good projects

FunASR exposes lookahead as a runtime flag:
`[5,10,5]` = 600 ms window, 300 ms lookback, 300 ms lookahead
(`runtime/docs/SDK_tutorial_online.md:70`). VoxCPM exposes
`streaming_prefix_len`, chunk-level streaming, and per-step yields. AudioGPT
exposes nothing and cannot: a chat round-trip sits in its critical path.

Projects designed for latency make it a dial. Projects designed for
demonstration do not.

### (c) Reproducible measurement is treated as part of the deliverable

FunASR ships `rtf_reproducibility.md` with a required-fields table and a
three-run median protocol. VoxCPM publishes a full Seed-TTS-eval comparison
against 25 baselines. AudioGPT publishes **zero** numbers of any kind.

Note the asymmetry: the repo that bothers to specify *how* to measure is also
the one whose numbers we can act on.

## 3. What is genuinely novel

### The CLS-token audio tokenizer (VoxCPM) — best single idea here

`src/voxcpm/modules/locenc/local_encoder.py:11`:

```python
self.special_token = nn.Parameter(torch.randn(1, 1, 1, config.hidden_size))
x = torch.cat([special_tokens, x], dim=2)
outputs, _ = self.encoder(x, is_causal=False)
cls_output = outputs[:, 0, :]        # one token per audio patch
```

A learned query vector collapses every audio patch to exactly one token,
**using a backbone you already own** — the same MiniCPM-4 weights, with
`assert config.vocab_size == 0` (`local_encoder.py:17`) confirming it is the
same transformer with no output head. Cost: one parameter vector.

The value to us is not TTS. It is that this is a clean, minimal answer to
"how do I get audio into an LLM at a low token rate without training a
dedicated codec?" If we ever want an LLM to reason *about* audio rather than
merely transcribe it, this is the cheapest credible route.

### 2pass (FunASR) — best mechanism *for our stated problem*

Streaming Paraformer costs +0.61 CER on Aishell1 dev, +1.40 on test, +1.18 on
Aishell2 test_mic versus offline. 2pass runs the streaming model for latency
and the offline model to correct **at the end of each sentence** — where the
user is not waiting on the first token. `funasr-wss-server-2pass` ships as a
binary.

### The runaway guard (VoxCPM)

`voxcpm.py:488` — if `generated_patches >= text_length * 6.0`, the decode has
degenerated; bump the seed and retry, bounded at 3 attempts. No critic model.
Six lines. Applies to any generative stage.

## 4. The five most promising things to try, with the experiment each justifies

Ordered by expected value per unit of effort.

### 1. 2pass streaming ASR — *replaces our Whisper hot path*
**Justifies:** our constraint "local Whisper inference is not viable for a hot
path" plus "latency matters more than quality". 2pass is the only mechanism in
this pass that improves accuracy *without* costing first-token latency.

**Experiment.** Take one representative WAV corpus. Measure three
configurations at identical settings: (a) current faster-whisper CPU int8
baseline, (b) FunASR Paraformer-zh-streaming `online`, (c) FunASR `2pass`.
Report **first_text_ms_p50/p95** and **final_after_stop_ms_p50/p95** — FunASR's
own metric names, from `realtime_ws_benchmark.md` — plus CER on a
**self-transcribed** set, since their published CERs are Chinese-only. Kill
criterion: if 2pass first-text p95 does not beat the Whisper baseline, drop it.

### 2. Tune `chunk_size` as our latency budget — *not a config default*
**Justifies:** "latency matters more than quality", and the dial exists.

**Experiment.** Sweep `[0,10,5]` → `[0,10,2]` → `[0,5,2]`. Plot CER against
`first_text_ms_p95`. Pick the knee. The expected shape: lookahead is buying
accuracy with latency, so there is a knee worth finding. **This is cheap — a
flag, no code — and directly optimizes our actual objective.**

### 3. Adopt `rtf_reproducibility.md` as our own benchmark contract
**Justifies:** we have been burned by unreproducible claims; the brief cites
prior examples of overstated repos. This is the antidote, and it is a
*document*, not a project.

**Experiment.** Before any model is evaluated, write our own required-fields
table (data, model revision, runtime, hardware, software versions, pipeline,
batching, timing scope, quality method) and refuse to record any latency
number that cannot fill it. Adopt "≥3 runs, median + min/max". Immediate
effect: it makes it structurally impossible for us to repeat the mistake this
pass was commissioned to catch.

### 4. Probe the GGUF/llama.cpp edge path for the always-on VAD+ASR
**Justifies:** "runs small enough for CPU or edge is very interesting", and we
are on WSL with no audio hardware and a latency budget.

**Experiment.** Load `SenseVoiceSmall-GGUF` and `fsmn-vad` under llama.cpp,
CPU-only, on a WAV loop. Measure RTF and per-chunk latency. Combined ceiling:
234M + 0.4M. If RTF « 1 holds on CPU, the whole recognition path fits a cheap
always-on process — which is the whole thesis of the peripheral.

### 5. Port the runaway-length guard to our generative stages
**Justifies:** cheap insurance against a decoder degenerating in a
long-running process, at zero model cost.

**Experiment.** Take the ratio check (`voxcpm.py:488`) as a template. Where
*we* generate variable-length output from a short input, assert
`output_length ≤ k × input_length`, and on violation reseed and retry with a
bounded attempt count. First application: any TTS or audio-synthesis leg we
end up running in-process.

## 5. What we should NOT build

Stated bluntly, because negative results are the cheapest output of this pass.

- **Do not build a unified audio-token LLM.** That is the AudioGPT *paper's*
  ambition. The released code has no ATST, no audio path into the LLM, no
  benchmarks, no tests, no CI, and an **empty LICENSE (0 bytes)**. We would be
  starting from a paper sketch with no working reference.
- **Do not build an audio LLM by bolting tools to a text LLM.** AudioGPT's
  filename-as-audio-channel design tells us the LLM contributes nothing about
  audio. If we route audio through an LLM, it must be *actually conditioned* on
  audio tokens — or not done at all.
- **Do not adopt VoxCPM's model for a real-time voice loop.** RTF 1.76 on an
  M4 Pro with Metal is slower than real time, and that is on good hardware.
  Its Chinese-first profile (test-Hard CER 8.87%) and 48 kHz target are both
  wrong for an English peripheral.
- **Do not adopt the full FunASR C++ runtime for v1.** It is excellent and it
  is a large integration surface. Python + GGUF captures most of the value.
- **Do not build diarization.** FunASR itself says the path is offline and the
  labels are anonymous per-recording indices. A single-channel, single-user
  peripheral gains nothing, and it costs **2.2× throughput** (RTFx 102 → 46).
- **Do not port a multilingual 1.5B+ ASR.** We already carry a Whisper path;
  adding Qwen3-ASR or GLM-ASR-Nano is more GPU-hungry redundancy.
- **Do not chase 48 kHz or "studio quality".** Compute we do not have, for
  fidelity no peripheral listener will hear.

## 6. Honest negatives

- **FunASR's benchmark table is Chinese-only.** The English section renders
  empty. Every CER number we extracted is Mandarin. **None of it transfers to
  our English system without our own measurement.**
- **VoxCPM publishes throughput RTF but no first-token latency.** For a voice
  peripheral, first-token latency *is* the metric. It is not published. And no
  RTF exists for the 0.5B model on any non-4090 device.
- **We have no CPU RTF for Paraformer streaming** — the single most
  decision-relevant number in this entire pass. It is not published.
- **The `rtf_reproducibility.md` numbers are Fun-ASR-Nano (800M) on
  GPU/vLLM**, not the 220M streaming model on CPU. They establish the
  *sensitivity* of RTF to serving stack (16× swing), not our operating point.
- **Three different "RTF" meanings** appear across these repos — offline
  batch throughput, streaming first-token, and end-to-end service latency.
  They are not comparable, and FunASR is the only repo that says so in writing.
- **No repo here was executed.** Every finding is from source and published
  tables. Nothing in this pass is a measured result of ours.

## 7. The one-line summary

> FunASR answers our question (streaming: yes, and here is the measured cost
> and the mitigation); VoxCPM holds the best mechanisms (the CLS audio
> tokenizer, the runaway guard) and the worst news (RTF 1.76 on real local
> hardware); AudioGPT is a filename-passing agent with an empty license and
> should be closed out.