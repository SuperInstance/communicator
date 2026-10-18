# VoxCPM — OpenBMB/VoxCPM

*Audited 2026-10-18 at commit `cce58a5b59303c9bd63d12f23afe6a49a4a80c59`
(`main`). LICENSE is present and real (Apache-2.0, 11298 bytes) — unlike
AudioGPT.*

## Verdict up front

**STEAL THE MECHANISMS, DO NOT ADOPT THE MODEL.** This is the most
interesting *engineering* of the three repos, and the one whose headline
performance numbers are least applicable to us. Its good RTF requires an RTX
4090; its CPU/edge path measures **RTF 1.76 — slower than real time**. Since
our constraint is "continuous, cheap, latency over quality", we are looking at
exactly the number VoxCPM leads least with.

## What it is

A monolithic TTS model: an LLM backbone (MiniCPM-4) that autoregresses audio
*tokens*, conditioned on text tokens and a voice prompt. Not an agent, not a
toolkit — one model with four cooperating parts:

| Module | File | Role |
|---|---|---|
| MiniCPM-4 | `src/voxcpm/modules/minicpm4/model.py` | LLM backbone, generates the token sequence |
| LocalEncoder (tokenizer) | `src/voxcpm/modules/locenc/local_encoder.py` | audio latents → one token per patch |
| LocalDiT (CFM) | `src/voxcpm/modules/locdit/local_dit.py`, `unified_cfm.py` | flow-matching refinement over audio |
| AudioVAE (vocoder) | `src/voxcpm/modules/audiovae/audio_vae.py` | tokens → waveform |

Plus `zipenhancer.py` (a neural audio enhancer), `lora.py` (fine-tuning), and
a `timestamps/` package for word/char timing.

## Architecture — the part worth stealing

### LocalEncoder: a CLS-token audio tokenizer, 30 lines

`src/voxcpm/modules/locenc/local_encoder.py:11` prepends a **learned
`special_token`** to every audio patch, runs the transformer bidirectionally
(`is_causal=False`), and keeps only position 0:

```python
self.special_token = nn.Parameter(torch.randn(1, 1, 1, config.hidden_size))  # :11
...
special_tokens = self.special_token.expand(B, T, 1, -1)
x = torch.cat([special_tokens, x], dim=2)
x = rearrange(x, "b t p c -> (b t) p c")
outputs, _ = self.encoder(x, is_causal=False)
cls_output = outputs[:, 0, :]
```

Every patch is compressed to one token by a learned pooling query, using the
*same* transformer weights as the generator (a config with
`vocab_size == 0`, `assert config.vocab_size == 0` at
`local_encoder.py:17` — the encoder shares the backbone but has no output
head). `patch_size: int = 2` (`voxcpm.py:78`).

**This is the single most transferable idea in the whole pass.** It is a
one-token-per-patch audio tokenizer for the cost of one learned vector,
reusing the backbone you already have. Not because we need TTS — but because
"compress each audio window to one embedding token with a learned query, using
an existing transformer's first-token output" is a pattern for *any*
low-rate audio representation feeding an LLM.

### Streaming that yields audio mid-generation

`generate_streaming` / `generate_with_prompt_cache_streaming`
(`voxcpm.py:356`, `:599`) yield decoded audio *per diffusion step* rather than
at the end, with `streaming_prefix_len: int = 3` (`:617`) reusing the first 3
prefix patches as context. Note the explicit, honest interaction constraint at
`voxcpm.py:375`:

```python
if retry_badcase and streaming:
    warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
```

Two mechanisms that cannot compose, documented rather than silently
mishandled. That is a quality signal about the codebase.

### retry_badcase: a 3-line length sanity guard (cheap, stealable)

`voxcpm.py:488-494`. If the generated audio runs away relative to the input
text, detect it and reseed:

```python
if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:  # 6.0
    print(f"  Badcase detected, audio_text_ratio=..., retrying...", file=sys.stderr)
    retry_badcase_times += 1
    current_seed += 1
    continue
```

Bounded at 3 attempts (`retry_badcase_max_times: int = 3`, `voxcpm.py:370`).
No neural critic, no scoring model — just "output length ÷ input length
exceeded 6×, so the decode degenerated; try another seed."

### StaticKVCache

`minicpm4/cache.py` preallocates a single
`zeros(2, num_layers, batch, kv_heads, max_length, dim)` tensor and walks a
`current_length` cursor via `step()`. Fixed footprint, no reallocation, and
`step()` raises on overflow rather than growing silently. Reasonable for a
server that must not fragment its allocator over a long-lived process.

## REAL vs described

### Genuinely real — unlike the other two repos

| Signal | Finding |
|---|---|
| Tests | 7 files, **108 asserts, 14 `pytest.raises`, 0 skips** |
| Tests hit real logic | e.g. `test_timestamps.py` asserts exact word→char duration splitting |
| Mocking is scoped, not evasive | 111 `MagicMock`s, all for model loading (`tests/test_cli.py:104`) |
| Weights | Not committed; downloaded from HF — normal practice |
| Benchmark | Seed-TTS-eval WER/CER + SIM published vs 25 baselines |

The tests are honest unit tests of the pure logic (timestamp splitting,
`torch.load` safety, LoRA checkpoint key handling, CLI arg parsing). They do
**not** download weights or prove the model synthesizes audio — which is
reasonable, but worth stating plainly: no test proves end-to-end speech output.

### The real gap: CI that never runs the tests

`.github/workflows/` contains exactly one workflow —
`publish-to-pypi.yml` — and it triggers only `on: release: types: [created]`.
It builds a wheel and pushes to PyPI. It **never invokes pytest**.

So: 1,485 lines of real tests, and no automated run of them. This is a
softer cousin of the `|| true` pattern — the CI exists, is nontrivial, and is
structurally incapable of catching a regression because it has no test step.
Worth naming as its own category: **tests that exist but are never executed.**

### Measured numbers

Seed-TTS-eval, from the README's own comparison table:

| Model | Params | test-EN WER↓ / SIM↑ | test-ZH CER↓ / SIM↑ | test-Hard CER↓ / SIM↑ |
|---|---|---|---|---|
| **VoxCPM2** | 2B | 1.84 / 75.3 | 0.97 / 79.5 | 8.13 / 75.3 |
| VoxCPM1.5 | 0.8B | 2.12 / 71.4 | 1.18 / 77.0 | 7.74 / 73.1 |
| **VoxCPM-0.5B** | 0.6B | 1.85 / 72.9 | **0.93** / 77.2 | 8.87 / 73.0 |
| CosyVoice2 | 0.5B | 3.09 / 65.9 | 1.38 / 75.7 | 6.83 / 72.4 |
| F5-TTS | 0.3B | 2.00 / 67.0 | 1.53 / 76.0 | 8.67 / 71.3 |

Minor inconsistency worth noting: the model-zoo table (README.md:370) lists
VoxCPM-0.5B as **0.5B** while the benchmark table (README.md:427) lists it as
**0.6B**. Both self-reported.

### Chinese vs English — the claim, checked

The README positions it as a bilingual zh/en model. The numbers support a
**real asymmetry that favours Chinese**:

- test-ZH CER **0.93%** vs test-EN WER **1.85%** — English intelligibility is
  roughly 2× worse in absolute error.
- test-Hard CER 8.87% is *nine times* worse than test-ZH. The hard subset is
  where the gap shows.
- SIM is comparable across both (77.2 ZH vs 72.9 EN), so the deficit is
  **intelligibility, not voice identity** — it mis-renders words, not the
  speaker.

For an English-language peripheral, that is the number that matters: **this is
a Chinese-first TTS.** VoxCPM2's 30-language support is a v2 feature and does
not change the evaluation shown.

### RTF — the finding that decides this for us

| Configuration | RTF |
|---|---|
| VoxCPM2, RTX 4090, PyTorch | ~0.30 |
| VoxCPM2, RTX 4090, Nano-vLLM | ~0.13 |
| **VoxCPM2, Q8_0, Apple M4 Pro / Metal (llama.cpp-omni)** | **~1.76** |
| VoxCPM1.5 / 0.5B, RTX 4090 | ~0.15 / ~0.17 |

The README leads with the 4090 numbers (README.md:53, :295). The llama.cpp
figure is demoted to a note (README.md:358) — but **RTF 1.76 means 1.76 s of
compute per 1 s of audio: slower than real time**, and it is measured on an
M4 Pro with Metal, which is a *good* local machine. A CPU-only or modest-GPU
peripheral is strictly worse.

Also note the serving-stack sensitivity: 0.30 → 0.13 is a **2.3× swing** from
Nano-vLLM alone. Any single RTF figure is a statement about a serving stack,
not about the weights.

## STEAL THIS

1. **The CLS-token audio tokenizer** (`local_encoder.py:11`) — one learned
   vector collapses each audio patch to one token using an already-purchased
   backbone. Cheap, and directly applicable to "get audio into an LLM" at low
   token rate. **Highest value in this repo.**
2. **The retry-on-runaway guard** (`voxcpm.py:488`) — a ratio sanity check with
   bounded reseeding. Cheap insurance against a generative decoder degenerating
   in production, at zero model cost. Applies to *any* generative stage in our
   stack.
3. **The streaming/non-streaming incompatibility being explicitly refused**
   (`voxcpm.py:375`) — warn and disable rather than silently produce
   different-quality output. We should adopt this contract wherever we add a
   fast path.
4. **`build_prompt_cache` / `generate_with_prompt_cache`** (`voxcpm.py:597`)
   — hoist a per-utterance prompt (the voice prompt) out of the hot path
   entirely. For a fixed synthetic voice in a peripheral this is exactly the
   right caching boundary.
5. **`StaticKVCache`** (`minicpm4/cache.py`) — fixed preallocated KV with an
   overflow *raise*. For a long-lived process, predictable memory.

## NOT USEFUL TO US

- **The model itself, at any speed we can run.** RTF 1.76 on good local
  hardware is disqualifying when latency leads. We need RTF « 1 with margin.
- **48 kHz studio-quality output** — a peripheral does not need it, and the
  sample rate costs compute we do not have.
- **The vLLM serving path** — assumes an RTX 4090-class GPU and a serving
  fleet. We are a single WSL process.
- **Chinese-first tuning** for an English system: test-Hard CER 8.87% is not
  a number to build a voice loop on.
- **The dependency on Llama-family LLM weights at all** for the *reasoning*
  part — we already have an LLM; we would only want the tokenizer and the
  guarded-decoding idea.

## What I could not verify

- No measured RTF for the **0.5B** model on CPU or on any non-4090 device. The
  0.17 figure is 4090-only. Given the 1.76-vs-0.30 gap between a good M4 Pro
  and a 4090, the 0.5B on our hardware is unknown and probably still > 1.
- No first-token latency (only throughput RTF). For our use — *when does the
  first sound come out* — that is the number that actually matters, and it is
  not published.
- Whether the CI-less test suite currently passes. Nothing runs it.