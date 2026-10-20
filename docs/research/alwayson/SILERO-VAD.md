# silero-vad — the hot-path candidate, measured

Repo: https://github.com/snakers4/silero-vad
Branch `master`, commit `1e261b036686cd0017d500ee96acd1c4ba572a9d`, last push 2026-09-29.
10,460 stars. **LICENSE: MIT** (`LICENSE:1-3`, `pyproject.toml:15`).
Package version 6.2.3 (`pyproject.toml:6`).

Every number here is measured by [`measure_hotpath.py`](measure_hotpath.py) on this
machine (AMD Ryzen AI 9 HX 370, 24 cores, onnxruntime 1.31.0, **1 thread**), or quoted
from silero's own wiki benchmark with the source named. Nothing here is from the README
except where the README is explicitly being quoted as a claim under test.

---

## 1. THE ANSWER TO OUR FAILURE MODE

Our declared failure (`src/watcher.py:40-47`) is that a steady periodic hum above the
energy floor reads as voiced speech. I built a battery of 4-second 16 kHz signals and ran
silero-vad over them with the exact call convention their own wrapper uses.

### Scores by signal, 512-sample (32 ms) chunks

| Signal | oww-vendored copy, mean | **current v5, mean** | current v5, max | frac > 0.5 |
|---|---:|---:|---:|---:|
| steady 60 Hz hum + harmonics | 0.0001 | **0.0036** | 0.0374 | **0.000** |
| fan noise (3.5 Hz AM) | 0.0212 | 0.0063 | 0.0306 | 0.000 |
| keyboard clicks (90 ms cadence) | 0.0123 | 0.0054 | 0.0334 | 0.000 |
| music triad + vibrato + swell | 0.0241 | 0.0052 | 0.1736 | 0.000 |
| flat white noise | 0.0330 | 0.0081 | 0.0289 | 0.000 |
| near-silence | 0.0256 | 0.0067 | 0.0230 | 0.000 |
| **real speech** ("alexa") | 0.7990 | **0.8819** | 1.0000 | 0.880 |
| **real speech** ("hey mycroft") | 0.8032 | 0.7642 | 1.0000 | 0.760 |
| **real speech** ("hey jane") | 0.6966 | 0.7530 | 1.0000 | 0.752 |

Speech fixtures are the repo's own test WAVs (`tests/data/alexa_test.wav` etc., real
16 kHz mono 16-bit speech).

**This is a clean separation.** Worst non-speech is music at 0.0052 mean / 0.1736 max.
Worst speech is 0.7530 mean. The threshold of 0.5 that silero uses by default
(`utils_vad.py:283`) sits in a gap that is **93.0× wide on the mean**
(0.7530 worst speech mean ÷ 0.0081 worst non-speech mean, white noise) and
**5.76× wide on the worst-case max** (1.0000 worst speech max ÷ 0.1736 worst
non-speech max, music). The separation holds across *every* non-speech class we
tested, including the periodic hum that specifically breaks us.

> **CORRECTION (2026-10-19).** This said *"28× wider on the mean"*. **No
> derivation in `hotpath_measurements.json` yields 28.** From
> `silero_v5_scores.current_v5_opset16`: worst speech mean **0.7530**
> (speech_jane), worst speech max **1.0000**; non-speech means 0.0036 (hum),
> 0.0063 (fan), 0.0054 (clicks), 0.0052 (music), **0.0081 (white noise)**,
> 0.0067 (silence); non-speech max **0.1736** (music). So the derivable
> figures are **93.0× on the mean** (0.7530 / 0.0081) and **5.76× on the
> worst-case max** (1.0000 / 0.1736); the note's own parenthetical pair
> (0.7530 / 0.1736) is **4.34×**. **The underlying separation is not wrong —
> the file supports a LARGER mean gap than 28×.** `NOTES-AUDIT.md` F10.

So: **silero-vad fixes the exact failure that our watcher declares, at 0.23% of a core.**
That is the single most important measured result in this lane.

Caveat stated honestly: these are 3 speech clips and 6 synthetic signals, and the
synthetic non-speech signals are clean by construction. A real room has reverb, competing
speech, and music beds this battery does not model. The separation is real but its margin
is measured, not certified.

## 2. Latency — their own benchmark, then mine

### Their benchmark (silero-vad wiki, *Performance-Metrics*)

> All speed test were run on AMD Ryzen Threadripper 3960X using only 1 thread and batch
> size equal to 1, 16000 sampling rate

| Single chunk (31.25 ms) processing speed, μs | RTS (×realtime) | Model |
|---:|---:|---|
| 830 | 36 | V4 JIT |
| 207 | 151 | V4 ONNX |
| 325 | 96 | V5 JIT |
| **189** | **165** | **V5 ONNX** |

The README's summary ("takes **less than 1 ms**", `README.md:117`) is **~5× conservative**
against their own table. The README also says "Under certain conditions ONNX may even run
up to 4-5x faster" — measured on their own numbers ONNX is 325/189 = **1.72×** faster than
JIT, not 4-5×.

Note this benchmark is on a **wiki page, not in the repo**. There is no benchmark script
in the repository and no benchmark table in the README. Reproducing it means calling their
API in a loop — which is what §3 does.

### My measurement (single thread, warm)

| Chunk | Median ms | RTS |
|---:|---:|---:|
| 480 samples (30 ms) | 0.0811 | **371×** |
| 512 samples (32 ms) | 0.0749 | **428×** |
| 1536 samples (96 ms) | 0.1045 | 919× |
| 15360 samples (960 ms) | 0.4929 | 1948× |
| cold `InferenceSession` construction | **11.06 ms** | — |

Cost is almost entirely per-call overhead, not per-sample: 15× the audio costs 6.6× the
time. **The 32 ms chunk is the right operating point** — smaller chunks cost the same
fixed overhead more often, and larger chunks are rejected by the model anyway (§4).

Budget check: at one 32 ms chunk per 32 ms of audio, silero costs **0.23% of one core**.
Our watcher costs 39.1% (mean over the six non-speech fixtures; 21.9–36.1 ms
on real speech). silero is not close to being the bottleneck and could be run at
1 kHz if we wanted.

## 3. Model sizes and parameter counts (from the artifacts)

Exact byte counts from `src/silero_vad/data/` at the pinned commit:

| File | Bytes | MB | Params | Opset |
|---|---:|---:|---:|---:|
| `silero_vad.jit` (TorchScript) | 2,272,526 | 2.167 | — | — |
| `silero_vad.onnx` (current, 8k+16k) | 2,327,524 | 2.220 | see below | 16 |
| `silero_vad_16k_op15.onnx` | 1,289,603 | 1.230 | **309,633** | 15 |
| `silero_vad_half.onnx` | 1,280,395 | 1.221 | 309,633 | 16 |
| `silero_vad_16k_sequence.onnx` | 1,246,165 | 1.188 | 309,633 | 16 |
| `silero_vad_op18_ifless.onnx` | 2,845,718 | 2.714 | 545,689 | 18 |
| `silero_vad_openvino_16k.onnx` | 1,288,203 | 1.229 | — | — |

### The "~1.8M params" figure is not supported by any artifact

Parameter counts are computed from the ONNX graph, recursively including `If` subgraphs
(`measure_hotpath.py:onnx_param_count`) — a naive `graph.initializer` count returns **0**
for `silero_vad.onnx` because the whole body lives inside `If` subgraphs, which is a trap
for anyone auditing this.

What the artifacts actually support:
- the 16 kHz-only variants are **309,633 parameters**
- the opset-18 "ifless" export of the same model is **545,689 parameters**
- the current 8k+16k `silero_vad.onnx` stores its weights in a form my counter does not
  resolve (0 initializers, 0 `Constant` tensors); its file size of 2,327,524 B is
  consistent with ~5×10⁵ fp32 weights, not with 1.8M

So the brief's "~1.8M params" is **roughly 3× too high** for any shipped ONNX, and the
README's own comparable claim is about *size* not params: "JIT model is around two
megabytes in size" (`README.md:121`) — which I confirm (2.167 MB).

Practical consequence: **silero-vad is a ~310k-parameter model**, which is why it is cheap,
and 2.2 MB is comfortably "small enough to load at boot".

### The `ifless` variant is a misnomer, but useful

`silero_vad_op18_ifless.onnx` still contains `If` nodes (ops: `Equal`, `Gather`, `If`,
`Reshape`). So "ifless" is not "no control flow". It is 2.714 MB and **is unreachable
through the public API**: `load_silero_vad` accepts only opset `[15, 16]`
(`model.py:8-10`) and `model.py:12-20` never names the op18 or half files. They are shipped
for downstream converters to use directly. Worth knowing, not worth adopting for no
measurable gain — opset 16 already loads in 11.06 ms.

## 4. THE DEPENDENCY PROBLEM — this is the real obstacle

The brief asked whether there is a numpy-only or ONNX-only path. **The answer is no, not
through the package**, and this is worth being precise about.

- `pyproject.toml:30-33` lists `dependencies = ["packaging", "torch>=1.12.0"]`.
  **torch is a hard, unconditional dependency of the package.**
- `[project.optional-dependencies] onnx-cpu = ["onnxruntime>=1.16.1", "numpy"]`
  (`pyproject.toml:52-55`) — the extra only *adds* onnxruntime. `pip install
  silero-vad[onnx-cpu]` **still installs torch**, because extras never remove base deps.
- `src/silero_vad/model.py:2` does `import torch` at **module scope**, and
  `src/silero_vad/__init__.py:7` imports from `silero_vad.model`. So `import silero_vad`
  requires torch **even when you pass `onnx=True`**.
- `OnnxWrapper` itself is not torch-free either: `utils_vad.py:104-108` builds state with
  `torch.zeros`, line 130 concatenates with `torch.cat`, lines 132/135 move tensors via
  `.numpy()` / `torch.from_numpy`. It is an **ONNX-inference** wrapper, not a
  **numpy-only** wrapper.

**So in our runtime (no torch), `pip install silero-vad` is not an option.** Not "a bigger
ask" — it is unavailable.

### The path that does work: the ONNX model directly

This is 30 lines of numpy + onnxruntime, and **openWakeWord already wrote it for us, MIT,
with silero's own licence header preserved** (`openwakeword/vad.py:15-39`). That file is the
answer to "does it have an ONNX-only path": yes, but the path is *someone else's
reimplementation of it*, and that someone else shipped it under a permissive licence.

The call convention, from `utils_vad.py:109-137`:

```
state  = zeros((2, 1, 128), float32)
context = zeros((1, 64), float32)
feed   = concat([context, chunk(1,512)], axis=1)     # 576 samples
out, state = session.run({"input": feed, "state": state, "sr": int64(16000)})[:2]
context = feed[:, -64:]                                # next context
```

Chunk size is **512 samples (32 ms) at 16 kHz**, hard-coded at `utils_vad.py:112`
(`num_samples = 512 if sr == 16000 else 256`) and `model.py:7` (`max_frames=512`). I
confirmed empirically: feeding 1536 samples raises inside the model's LSTM. There is no
"batch more audio per call to go faster" lever.

**Required new dependency: `onnxruntime` only.** Nothing else. I verified
`pip install onnxruntime` into a clean venv works and that
`providers=['CPUExecutionProvider']` is available; `inter_op_num_threads = 1` and
`intra_op_num_threads = 1` are set exactly as `utils_vad.py:69-74` does.

### The post-processing is torch-free and stealable

`get_speech_timestamps_from_probs` (`utils_vad.py:421-...`) takes a **plain list of
per-frame floats** and runs a hysteresis state machine. I grepped its entire body:
**zero references to torch**. It is MIT, and it is the part that turns a noisy probability
stream into speech segments:

| default | value | meaning |
|---|---:|---|
| `threshold` | 0.5 | above → speech |
| `neg_threshold` | `max(threshold - 0.15, 0.01)` (`utils_vad.py:475-476`) | hysteresis floor |
| `min_speech_duration_ms` | 250 | discard shorter |
| `min_silence_duration_ms` | 100 | wait before splitting |
| `speech_pad_ms` | 30 | pad each side |
| `max_speech_duration_s` | inf | no cap |

The **hysteresis** (`neg_threshold` ≠ `threshold`) is the interesting bit and is worth
copying regardless of silero: a single threshold on a probabilistic stream produces
flicker, and a separate lower exit threshold is the cheapest fix.

## 5. CI, tests, honesty — and one real problem

- `.github/workflows/test.yml:3-4`:
  ```yaml
  on:
    workflow_dispatch:       # запуск вручную
  ```
  **The test workflow only runs when someone clicks Run manually.** It does not run on
  push and not on pull_request. The green badge at the top of the README is for a
  workflow that does not gate anything. This is the opposite of openWakeWord's CI, which
  runs `pytest` on every push to `main` on Linux and Windows with no `|| true`.
- The second workflow (`python-publish.yml`) runs on `push: tags: ['*']` only, so it
  publishes without tests having run on the tag.
- **17 test functions** across 5 files.
- `tests/conftest.py` and `test_optional_deps.py` (6 tests) specifically cover the
  optional-dependency matrix, which is commendable.
- `test.yml:75-82` has a comment worth quoting because it is the right instinct:
  *"Fail loudly here rather than letting the backend tests skip silently."* Then
  `pytest tests -v`.
- No `|| true` anywhere.

So: honest commands, but a manual trigger. I would not treat "CI is green" as evidence for
this repo without checking when it last ran.

## 6. Source vs prose

Whole tree at the pinned commit (`.py` counts only the package + tests, so it is
deliberately small):

| | files | bytes |
|---|---:|---:|
| `.py` | 22 | 120,777 |
| `.md` | 19 | 51,493 |
| `.onnx` | 6 | 10,277,608 |
| `.wav` (fixtures) | 5 | 20,865,312 |

**2.3× more code than prose** — the most balanced of the three, and the only one whose
*actual weight files are in the repository* rather than behind a download. That matters
for reproducibility and it is a point in silero's favour.

The other 13 language ports (Java, Rust, Go, C#, C++, Haskell, C) total ~180 KB across
36 files — more port surface than the Python. For a 309k-parameter model whose inference
is one LSTM, that is a lot of duplicated I/O code.

## 7. Licence caveat, honestly

`LICENSE` is plain MIT and `pyproject.toml:15` says `License :: OSI Approved :: MIT
License`. **However**, `README.md:1` carries a contradictory badge — the link text is
`License: CC BY-NC 4.0` while the badge image alt-text says `License-MIT`. silero's
*other* models (TTS, STT) are CC BY-NC; the **VAD specifically is MIT**. Since the
contradictory string lives in the VAD repo's own README, I am recording it rather than
resolving it. If this ever matters commercially, confirm with Silero directly — the
`LICENSE` file and `pyproject.toml` are unambiguous MIT and the badge looks like a
copy-paste from silero-models.

## 8. Verdict for the hot path

| Question | Answer |
|---|---|
| Model size | 2.220 MB onnx; 309,633 params (16k variant) |
| Measured latency | **74.9 μs per 32 ms chunk, 428× realtime, 1 thread** |
| CPU-only viable | Yes, overwhelmingly |
| Fixes our hum failure | **Yes, measured: 0.0036 on hum vs 0.75+ on speech** |
| Licence | **MIT** |
| Runs without a DL framework | **The ONNX model does, with onnxruntime + numpy. The Python package does NOT** (torch is a hard dep, `pyproject.toml:32`) |
| Reproducible benchmark in-repo | No — it lives in the wiki |

**This is the hot path.** The only thing standing between us and adopting it is 30 lines
of MIT-licensed wrapper code, because we are not allowed to `pip install` it.