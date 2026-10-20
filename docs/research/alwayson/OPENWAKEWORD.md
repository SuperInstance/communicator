# openWakeWord — tiny wake-word models, measured

Repo: https://github.com/dscripka/openWakeWord
Branch `main`, commit `368c03716d1e92591906a84949bc477f3a834455`, last push 2025-12-30.
2,851 stars. License field: Apache-2.0. **The license field is only half the story — see §1.**

All latency and size numbers in this file were measured by me on this machine, not
read off the README. Script: [`measure_hotpath.py`](measure_hotpath.py). Raw output:
[`hotpath_measurements.json`](hotpath_measurements.json).

Host: AMD Ryzen AI 9 HX 370, 24 cores, onnxruntime 1.31.0, **1 thread**, CPUExecutionProvider.
This is a fast desktop x86 CPU, not a Raspberry Pi. Every number below is therefore an
*optimistic* bound for a peripheral. §7 says how optimistic.

---

## 1. THE LICENSING FINDING — the reason this repo is not the hot path

The README's own License section, read carefully:

> All of the code in this repository is licensed under the **Apache 2.0** license.
> All of the included pre-trained models are licensed under the
> **[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International]**
> license due to the inclusion of datasets with unknown or restrictive licensing
> as part of the training data.

**CC BY-NC-SA 4.0 is non-commercial.** So:

| What | License | Usable for us? |
|---|---|---|
| The Python library | Apache-2.0 | yes |
| The shared Google speech-embedding backbone | Apache-2.0 (README cites the TFHub module) | yes |
| **The 6 bundled pre-trained wake-word models** | **CC BY-NC-SA 4.0** | **no, if we ever ship** |
| Silero VAD copy vendored in the release | MIT | yes |

GitHub's `licenseInfo` reports `Apache-2.0` for the repo, which is true of the *code* and
wrong as a summary of what you actually download and run. Any automated licence check
against the repo metadata will give the wrong answer here.

Two further traps in that licence:
- **ShareAlike** — derivative works must carry the same terms.
- The models are unusable commercially, so "train your own" is not a trivial escape:
  training needs torch (see §6) plus several thousand synthetic clips.

`alexa_v0.1.onnx` etc. are **not in git**. `openwakeword/__init__.py:26-51` points at
`openwakeword/resources/models/*.tflite` with `download_url`s into the **release assets**.
The `.onnx` variants are found by string substitution:
`openwakeword/__init__.py:65-69` — `get_pretrained_model_paths` returns
`MODELS[i]["model_path"].replace(".tflite", ".onnx")`.

## 2. Exact sizes (byte counts from the v0.5.1 release assets)

Downloaded and `stat`ed, not estimated:

| File | Bytes | MB | Params (counted from ONNX) | Opset |
|---|---:|---:|---:|---:|
| `melspectrogram.onnx` | 1,087,958 | 1.038 | 271,395 | 13 |
| `embedding_model.onnx` | 1,326,578 | 1.265 | 328,137 | 13 |
| `alexa_v0.1.onnx` | 854,246 | 0.815 | 213,377 | 13 |
| `hey_mycroft_v0.1.onnx` | 857,691 | 0.818 | 213,889 | 13 |
| `hey_jarvis_v0.1.onnx` | 1,271,370 | 1.212 | 316,738 | 13 |
| `hey_rhasspy_v0.1.onnx` | 204,081 | **0.195** | 50,401 | 14 |
| `timer_v0.1.onnx` | 1,742,475 | 1.662 | 435,335 | 13 |
| `weather_v0.1.onnx` | 1,149,158 | 1.096 | 287,105 | 13 |
| `alexa_v0.1.tflite` | 855,312 | 0.816 | — | — |

So "a few MB" is right, with a structure worth noting: the **shared front is a fixed
2.303 MB** (melspec + embedding) and **each additional wake word costs 0.195–1.662 MB**,
because the backbone is frozen and shared. That sharing is the actual architectural idea
(`model.py:143-182` builds one `AudioFeatures` and N independent heads).

Parameter counts are from `measure_hotpath.py:onnx_param_count`, which walks ONNX
initializers recursively. Counts from the real graph, not from a README.

## 3. Measured latency

Per 80 ms frame (1280 samples), single-threaded, warm:

| Component | Median ms | Share of the 80 ms budget |
|---|---:|---:|
| melspectrogram | 0.0641 | 0.08% |
| embedding (the expensive part) | 0.8494 | 1.06% |
| shared front (both) | 0.9199 | 1.15% |
| one wake-word head | 0.0103 | 0.01% |
| **full stack, 1 head** | **1.1269** | **1.41%** |
| **full stack, all 6 heads** | **1.6281** | **2.04%** |

Two things worth stealing as *design facts* even though we are not adopting the repo:

1. **The embedding is 92% of the cost and it is shared.** Adding a 7th wake word costs
   0.0103 ms. That is the payoff of freezing a backbone.
2. **Latency is bought with frame length.** `model.py:287-298`: when more than 1280
   samples are prepared, the library iterates *backwards* over the frame and takes
   `max` across sub-frames. Longer frames amortise the shared front but delay the
   decision; the README says exactly this and the code implements it.

## 4. Is there a benchmark? (No — and the one that exists is broken)

The README claims "a single core of a Raspberry Pi 3 can run 15-20 openWakeWord models
simultaneously in real-time." There is **no script in the repo that reproduces it.**

- `benchmark/benchmark.py` is **stale and cannot run.** Line 29 passes
  `input_sizes=[16]` to `Model(...)`. The string `input_sizes` appears **nowhere** in
  `openwakeword/` in v0.6.0 — `Model.__init__` (`model.py:38-48`) has no such parameter,
  and `AudioFeatures.__init__` (`utils.py:38-45`) does not either. It would land in
  `**kwargs` and raise `TypeError`. Line 25-26 also globs a path layout that no longer
  matches the package.
- `examples/benchmark_efficiency.py` requires `pyaudio` and a **live microphone**, and
  computes its "could run N more models" figure from a formula on line 58 rather than by
  loading more models.

So the Pi-3 number is a README assertion with no reproducible artifact behind it in this
commit. My own 24-core-x86 measurement (§3) cannot confirm it either way.

## 5. Dependencies — the environment question

Runtime deps declared in `setup.py:32-40`:

```
onnxruntime>=1.10.0,<2
ai-edge-litert>=2.0.2,<3      # Linux/Darwin
speexdsp-ns>=0.1.2,<1         # Linux
tqdm, scipy, scikit-learn, requests
```

Our runtime has **none** of these (`numpy` and `yaml` only). The honest accounting:

- `torch` is **not** needed for inference. Good.
- But `pip install openwakeword` pulls `scipy`, `scikit-learn`, `requests`, `tqdm`,
  `onnxruntime` and `ai-edge-litert` — **six packages**, two of which (scipy, sklearn) are
  entirely gratuitous for inference.
- `openwakeword/__init__.py:2` → `model.py` → `utils.py:28` `import requests` at module
  scope, so **`import openwakeword` fails outright** without `requests`, and `utils.py:24`
  imports `tqdm`. The package cannot be imported in a requests-less runtime even if we only
  wanted the ONNX path.
- `vad.py:48` imports `onnxruntime` at module scope too.

**Verdict: the ONNX path exists but is not ONNX-only.** Reachable only after installing
`onnxruntime` + `requests` + `tqdm` (and it will try to import `ai_edge_litert` first,
falling back to onnx at `model.py:121-131`).

## 6. Custom wake-word training — needs the big hammer

`docs/custom_verifier_models.md` + `openwakeword/train.py` (46 KB). Training requires
the `full` extra (`setup.py:54-78`): `torch>=1.13.1`, `torchaudio`, `speechbrain`,
`torch-audiomentations`, `tensorflow-cpu==2.8.1`, `tensorflow_probability`,
`onnx_tf`, `deep-phonemizer`, `datasets`, `acoustics`. That is a research environment,
not a dependency we would take to escape the NC licence.

The method itself is the interesting part and is well documented: generate synthetic
speech with TTS, train a small classifier on top of the **frozen** shared embedding
(`model.py` architecture section, README "Model Architecture"). Negative data is the
bottleneck: "the included models were all trained with ~30,000 hours of negative data."
That number is why we could not simply retrain our own legal model.

The `custom_verifier_model` path is much lighter — it is a scikit-learn `Pipeline`
(`custom_verifier_model.py`) over the same embeddings — but it still needs sklearn, which
we lack.

## 7. CI, tests, and honesty

- `.github/workflows/tests.yml`: runs `pytest` on **ubuntu-latest** (py3.10) and
  **windows-latest** (py3.10). **No `|| true`, no `continue-on-error`, no `allow-failure`.**
  `grep -rn "|| true\|continue-on-error" .github/` returns nothing.
- 16 test functions in `tests/` (`test_models.py`, `test_custom_verifier_model.py`).
- Tests do need the real weights downloaded, so CI exercises the download path too.

This is the most honest CI of the three repos reviewed.

## 8. Source vs prose

From the git tree at the pinned commit (all extensions, so `.ipynb` counts as neither):

| | files | bytes |
|---|---:|---:|
| `.py` | 18 | 217,800 |
| `.md` | 11 | 67,006 |
| `.ipynb` (notebooks) | 4 | 862,476 |

**3.3× more code than prose.** openWakeWord is a real implementation with thin docs —
and the thin docs are exactly where its unverified claims live (the Pi-3 number is in the
README and has no script behind it, while `benchmark/benchmark.py` cannot run).

Worth noting: the 4 notebooks total 862 KB, **3.96×** the entire Python source — which
is a lot of notebook for a project whose training story is
"here is a Colab, run it for an hour".

> **CORRECTION (2026-10-19).** This said "openWakeWord **is the most notebook-heavy
> of the three**". It is not: pyannote's notebooks exceed its source by **6.6×**
> (5,264,103 ÷ 798,995) against openWakeWord's **3.96×** (862,476 ÷ 217,800). The
> claim was not in `NOTES-AUDIT-2.md` — it surfaced while applying that audit's
> corrections, when the two files stopped agreeing — so it is recorded here rather
> than left to be discovered twice. Note that the three repos are not compared on
> the same population (silero lists no `.ipynb` row at all, and pyannote's "6.6×"
> counts package + tests only), so "most notebook-heavy of the three" is not a
> well-posed comparison in the first place.

> **CORRECTION (2026-10-19).** This said the 4 notebooks total 862 KB, *"more than 4×
> the entire Python source"*. 862,476 ÷ 217,800 = **3.960** — just under 4×, not over
> it, so the claim landed on the wrong side of its own threshold. The neighbouring
> "3.3× more code than prose" is unchanged and correct (217,800 ÷ 67,006 = 3.25).
> Found by the notes audit — `NOTES-AUDIT-2.md` F4.

## 9. Verdict for the hot path

| Question | Answer |
|---|---|
| Model size | 2.303 MB fixed front + 0.195–1.662 MB per wake word. Fine. |
| Measured latency | 1.63 ms per 80 ms frame, all 6 heads, 1 thread. Cheap. |
| CPU-only viable | Yes, demonstrably. |
| Reproducible benchmark in-repo | **No** — `benchmark/benchmark.py` uses a removed API |
| Runs without a DL framework | Yes (onnxruntime), **but not without 6 pip installs** |
| **Licence permits our use** | **NO — weights are CC BY-NC-SA 4.0** |

**Not our hot path, on licensing.** It is fast, tiny, CPU-only and its shared-backbone
design is worth copying. But the only thing that makes openWakeWord *openWakeWord* — its
pre-trained weights — is non-commercial, and the retrain path needs a research-grade
environment we are not going to adopt to escape an NC clause.

Note the irony worth recording: openWakeWord already vendors silero-vad's MIT VAD
(`openwakeword/vad.py`, MIT header preserved at lines 15-39) and gates its own predictions
on it (`model.py:366-381`, `vad_threshold`). The part of this repo we may actually want is
the *third-party* part.