# FunASR — modelscope/FunASR

*Audited 2026-10-18 at commit `4bdf87e15169515c093f71b6aad9717aed9cd884`
(`main`). LICENSE present (1061 bytes). This is the only one of the three
repos that looks like production infrastructure.*

## Verdict up front

**THIS IS THE ONE.** It answers our streaming question with a yes and a
measured cost, ships a CPU/edge path via GGUF, has the test and CI discipline
the other two lack, and contains the mechanism — **2pass** — that directly
attacks our latency constraint. Its best contribution is arguably not the
models but `docs/benchmark/rtf_reproducibility.md`.

## What it is

An ASR toolkit from Alibaba DAMO (ModelScope), MIT-licensed, shipping ~50
models under `funasr/models/`. It is three things at once: a Python SDK
(`funasr/auto/`), a C++/ONNX runtime (`runtime/`, with C++/Java/C#/Go/Android/
iOS/WebRTC bindings and a Triton GPU path), and a model zoo.

## REAL vs described — and here the description *exceeds* the reality

This is the first repo in the pass where the docs are conservative rather than
inflated. Both headline claims I checked are caveated in place:

- README.md: "OpenMOSS publishes MOSS-Transcribe-Diarize; FunASR provides an
  adapter, **not ownership of its weights**." Model licenses are called out as
  separate from the toolkit's MIT.
- On diarization: "**Anonymous indices within a recording, not enrolled-
  speaker identification**", and "Its unified path is **offline**".

### Evidence of a real, maintained codebase

| Signal | Finding |
|---|---|
| Size | 1,448 `.py`, 4,153 tracked files |
| Tests | **121 test files, 2,017 asserts, 52 `pytest.raises`, 1 skip** |
| CI | **19 workflows**, path-filtered, actually executing |
| `\|\| true` | 4 total, all in `build-llamacpp-binaries.yml` file-copy steps |

The `|| true` instances I checked are legitimate — `cp build/bin/llama-funasr-*
pkg/ 2>/dev/null || true` copies binaries that may not have been produced.
None are attached to a correctness check. **No `continue-on-error` on any test
job.** This is what CI discipline actually looks like, and it is the opposite
of the VoxCPM pattern.

`test-realtime-benchmark.yml` is a good example: triggers on PR, installs
`pytest websockets`, runs `python -m pytest -q tests/test_realtime_ws_benchmark.py`
under a 5-minute timeout. Real, bounded, and CPU-only by design.

## Model zoo — with the parameters that matter

| Model | Task | Params | Edge path |
|---|---|---|---|
| **Paraformer-zh-streaming** | **streaming ASR** | **220M** | ONNX |
| Paraformer-zh | offline ASR + timestamps | 220M | ONNX |
| **SenseVoiceSmall** | ASR + emotion + event tags | **234M** | **GGUF** |
| Fun-ASR-Nano | ASR, zh/en/ja + dialects | 800M | **GGUF** |
| fsmn-vad | VAD | **0.4M** | ONNX |
| cam++ | speaker embeddings | **7.2M** | — |
| ct-punc | punctuation | 290M | ONNX |
| emotion2vec+large | emotion | 300M | — |
| Qwen3-ASR / GLM-ASR-Nano / Whisper-large-v3 | multilingual | 1.7B / 1.5B / 1550M | — |

The three small ones are the interesting ones for a peripheral: **fsmn-vad at
0.4M and cam++ at 7.2M are trivially CPU-viable**, and SenseVoiceSmall at 234M
has an official **GGUF edge checkpoint** for llama.cpp.

## YES, IT SUPPORTS STREAMING — three separate ways

### 1. Paraformer streaming ASR, with an explicit latency dial

`funasr/models/paraformer_streaming/model.py:520`:

```python
chunk_size = kwargs.get("chunk_size", [0, 10, 5])
```

That is `[left_context, current, right_context]` — **chunk-based streaming with
lookahead**. The runtime docs make the latency tradeoff explicit
(`runtime/docs/SDK_tutorial_online.md:70`):

> `--chunk_size` indicates the latency configuration of the streaming model.
> `[5,10,5]` indicates that the current audio is 600ms, with a **lookback of
> 300ms and a lookahead of 300ms**.

So latency is a **first-class tunable**, not a fixed property. Shrink lookahead
for latency, grow it for accuracy. Our constraint ("latency matters more than
quality") maps onto this dial directly.

### 2. Streaming VAD and keyword spotting

`fsmn_vad_streaming` and `sanm_kws_streaming` exist as first-class modules —
streaming VAD is exactly the endpointing primitive our system needs.

### 3. **2pass — the important one**

`runtime/docs/SDK_tutorial_online.md:69`:

> `--mode`: `offline` indicates that the inference mode is one-sentence
> recognition; `online` indicates real-time speech recognition; **`2pass`
> indicates real-time speech recognition, and offline models are used for error
> correction at the end of each sentence.**

`run_server_2pass.sh` wires it: `--model-dir` (offline Paraformer) *and*
`--online-model-dir` (streaming Paraformer) into the prebuilt
`funasr-wss-server-2pass` binary, plus VAD, punctuation, ITN, and an n-gram LM
rescoring stage.

**This is the answer to our latency problem.** Streaming ASR is measurably
worse; 2pass recovers offline accuracy at streaming latency by correcting
only at utterance end — where nobody is waiting on the first token.

## The measured numbers

### CER — streaming vs offline, from the repo's own leaderboard

`benchmarks/benchmark_pipeline_cer.md`, Aishell-1 (dev / test) and Aishell-2:

| Model | Mode | Aishell1 dev | Aishell1 test | Aishell2 test_mic |
|---|---|---|---|---|
| Paraformer-large | Offline | **1.76** | **1.94** | 3.03 |
| Paraformer-large-long | Offline | 1.80 | 2.10 | 3.11 |
| Paraformer-large-contextual | Offline | 1.76 | 2.02 | 2.95 |
| **Paraformer-large-online** | **Online** | **2.37** | **3.34** | **4.21** |
| UniASR | Online | 3.34 | 3.99 | 4.73 |
| UniASR-large | Offline | 2.93 | 3.48 | 4.16 |

**The cost of streaming, quantified: +0.61 CER absolute on Aishell1 dev
(1.76 → 2.37), +1.40 on test (1.94 → 3.34), +1.18 on Aishell2 test_mic.**

This is the trade we would have had to guess at. It is small enough that 2pass
correction is clearly worth attempting, which is presumably why FunASR ships
2pass as a first-class mode rather than an afterthought.

### RTF / RTFx — and the caveat that governs all of them

From `docs/benchmark/rtf_reproducibility.md`:

| Configuration | RTFx | CER |
|---|---|---|
| Fun-ASR-Nano, vLLM batch | **340** | 8.20% |
| Fun-ASR-Nano, PyTorch baseline | **21** | 8.06% |
| Fun-ASR-Nano, offline service, no diarization | 102 | 8.14% |
| Fun-ASR-Nano, offline service, with diarization | 46 | 8.19% |

**RTFx 340 vs RTFx 21 is a 16× swing from the serving stack alone.** Same
weights, same audio. Any RTF number quoted without its runtime is noise.

## The best document in this entire pass

`docs/benchmark/rtf_reproducibility.md` is not a model card — it is a
**methodology contract**, and it should be read as a template for our own
benchmarking. Its core warning:

> Do not compare an offline batch `RTFx` result with streaming first-token
> latency or end-to-end product latency. They measure different things.

It then specifies the fields any speed claim must report: **Data · Model ·
Runtime · Hardware · Software · Pipeline · Batching · Timing scope · Quality**
— where "timing scope" explicitly asks whether the number includes model
download, cold start, warmup, file I/O, resampling, VAD, and serialization.

And a five-step timing protocol: compute total duration first, warm the model
if claiming steady state, time exactly one scope, **run ≥3 times and report
median plus min/max**, and keep the transcripts and timing JSON with the run.

There is a companion, `docs/benchmark/realtime_ws_benchmark.md`, that measures
what we actually care about — *client-observable* behaviour — with metrics
like `first_text_ms_p50/p95`, `final_after_stop_ms_p50/p95`, and
`aggregate_audio_per_wall`. It is careful to say a `--no-pace` run is "a
throughput stress signal, not user-facing realtime latency".

## STEAL THIS

1. **2pass** (`runtime/docs/SDK_tutorial_online.md:69`,
   `runtime/run_server_2pass.sh`) — streaming model for latency, offline model
   correcting at utterance end. **Directly attacks our constraint.** The single
   highest-value idea across all three repos.
2. **`chunk_size=[left, current, right]` as an explicit latency dial**
   (`paraformer_streaming/model.py:520`) — make lookahead a tunable and tune it
   against our budget rather than accepting a fixed latency.
3. **`rtf_reproducibility.md` as a template** — adopt its required-fields table
   for *every* latency number we publish internally, and its "run ≥3 times,
   report median + min/max" rule. This is a direct answer to having been
   burned by un-reproducible claims elsewhere.
4. **The realtime ws benchmark metrics** (`realtime_ws_benchmark.md`) —
   `first_text_ms_p50/p95` and `final_after_stop_ms_p50/p95` are the correct
   metrics for a voice peripheral. Adopt the naming so our numbers are
   comparable.
5. **GGUF checkpoints for the edge path** — SenseVoiceSmall-GGUF and
   Fun-ASR-Nano-GGUF mean a llama.cpp route with no PyTorch, which fits a
   cheap always-on process. 234M params is genuinely small.
6. **fsmn-vad at 0.4M** — streaming VAD that small is a better endpointing
   candidate than anything threshold-based, and it pairs with the existing
   VAD-in-pipeline convention.
7. **n-gram LM rescoring** in the 2pass server (`--lm-dir`, FST-based) — a
   cheap post-hoc accuracy win on the corrected pass only.

## NOT USEFUL TO US

- **The heavyweight multilingual models.** Qwen3-ASR (1.7B), GLM-ASR-Nano
  (1.5B), Whisper-large-v3 (1550M) are all GPU-class and not CPU-viable. Also
  redundant with the Whisper path already in our stack.
- **Diarization.** Explicitly offline, anonymous indices, not
  identification. A single-channel peripheral with one user gains nothing from
  it, and it costs 2.2× throughput (RTFx 102 → 46).
- **The full C++/ONNX runtime for v1.** `runtime/` is excellent but is a large
  integration surface. The Python path plus GGUF gets us most of the value at a
  fraction of the cost. Revisit if the Python path misses our latency budget.
- **Aishell/Wenet-specific language models.** The Chinese leaderboard is their
  home turf; for English we would benchmark `paraformer-en` / SenseVoiceSmall
  ourselves before trusting any of these numbers.

## What I could not verify

- **No English CER table** could be extracted — the README's "English Dataset"
  section renders as an empty table in `benchmarks/benchmark_pipeline_cer.md`.
  Every number I extracted above is **Chinese**. For an English system this is
  a real gap; the numbers must not be carried over.
- **No CPU RTF for Paraformer streaming.** The published RTFx figures are
  Fun-ASR-Nano on GPU/vLLM. Our case is 220M Paraformer on CPU — unmeasured
  here. This is the number to obtain before committing.
- **SenseVoiceSmall's own CER/SIM.** The repo links the paper and checkpoint
  but ships no benchmark table of its own; the numbers are in the arXiv paper,
  which I did not fetch.
- Whether `fun_asr_nano` (800M) fits a 6 GB GPU in practice — params suggest
  yes but no VRAM figure is published.