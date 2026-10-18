# pyannote-audio — speaker diarization, and why it cannot be the hot path

Repo: https://github.com/pyannote/pyannote-audio
Branch `main`, commit `b749285c5cdd4636b2edc7f766f1352c8dde9369`, last push 2026-10-18.
10,671 stars.

**This file answers the licence question first, because that is the question you asked to
be checked carefully, and because for this repo the licence and the dependency situation
are the whole answer.**

---

## 1. Licensing: code is MIT, the WEIGHTS are gated and separately licensed

Three different things are called "pyannote" and they do **not** share a licence.

### (a) The library — MIT, unambiguous

`LICENSE:1-3`:
```
MIT License
Copyright (c) 2020 CNRS
```
This is the code. We may use it, modify it, ship it, commercially.

### (b) The pretrained pipelines/models — CC-BY-4.0 or MIT, but GATED

The models live on HuggingFace under the `pyannote` org. From the HF API at the time of
writing:

| Model | `gated` | card licence | requires |
|---|---|---|---|
| `pyannote/speaker-diarization-community-1` | **`auto`** | cc-by-4.0 | HF account + accept conditions + token |
| `pyannote/speaker-diarization-3.1` | **`auto`** | mit | HF account + accept conditions + token |
| `pyannote/segmentation-3.0` | **`auto`** | mit | HF account + accept conditions + token |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | no | cc-by-4.0 | none |

`README.md:28` states the procedure in the install steps:
> 3. Accept [`pyannote/speaker-diarization-community-1`] user conditions
> 4. Create Huggingface access token at `hf.co/settings/tokens`

So: **the licence terms themselves are permissive** (CC-BY-4.0 and MIT both permit
commercial use; CC-BY-4.0 requires attribution, which is a documentation obligation, not a
restriction). The obstacle is not the licence, it is the **click-through**: our hot path
would depend on a HuggingFace account, a token, a licence acceptance that can be changed
or revoked by a third party, and a network fetch, before the first frame of audio is
processed.

`Model.from_pretrained` (`src/pyannote/audio/core/model.py:498-540`) takes
`token: str | bool | None = None` and accepts "identifier of a model on Huggingface hub"
(line 517) — so the gated download is the designed path, and a `None` token will fail.

**Answering the brief directly: we MAY use pyannote's licence — CC-BY-4.0/MIT both permit
it. What we may not do is make a licence acceptance a runtime dependency of an
always-on peripheral.** That is a different objection, and the more honest one.

### (c) Third-party embedding, separately licensed

The community-1 pipeline bundles `embedding/pytorch_model.bin` (25.412 MB) which is the
WeSpeaker ResNet34 described by `src/pyannote/audio/models/embedding/wespeaker/` — a
**different project with its own `LICENSE.WeSpeaker`** (present at
`src/pyannote/audio/models/embedding/wespeaker/LICENSE.WeSpeaker`). So the 25 MB of weights
in the pipeline are not covered by the MIT of pyannote's repo; they are covered by
WeSpeaker's terms. This is the kind of transitive licence detail that a repo-level
`licenseInfo: MIT` check will not catch, and it is the same *shape* of problem as
openWakeWord's NC weights.

## 2. Model sizes (exact, from the HF API with `blobs=true`)

| File | Bytes | MB |
|---|---:|---:|
| `embedding/pytorch_model.bin` | 26,646,242 | 25.412 |
| `segmentation/pytorch_model.bin` | 5,906,507 | 5.633 |
| `plda/plda.npz` | 133,852 | 0.128 |
| `plda/xvec_transform.npz` | 134,376 | 0.128 |
| **community-1 total** | **32,820,977** | **31.301** |
| `pyannote/segmentation-3.0` alone | 5,905,440 | 5.632 |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | 26,645,418 | 25.411 |

For comparison: the entire silero-vad hot path is **2.220 MB**. A diarization pipeline is
**14× larger**, and 81% of that is a speaker-embedding network we do not need to know
*that someone spoke*.

## 3. Does it need a GPU? What is actually published

**There is no CPU benchmark.** `README.md:110`:
> __Self-hosted speed on a NVIDIA H100 80GB HBM3__

That is the only speed figure in the repository. There is no CPU number, no real-time
factor, no memory figure for CPU inference. The README's DER benchmark table (`README.md:96`)
reports accuracy (DER) against RAMC and one other corpus, with no speed column.

Whether it *runs* on CPU: the library does not require CUDA. `Model(lightning.LightningModule)`
(`core/model.py:69`) will run on CPU, and `README.md:41` shows `# send pipeline to GPU
(when available)` — GPU is optional, not required. But "runs" is not "fast enough", and
**nobody has published the number that would tell us whether it is fast enough.** Given
that the published figures are H100-only and the workload is a ResNet34 embedding plus a
segmentation model over *every* frame of continuous audio, the assumption that it fits a
single-core peripheral is unfounded.

So the honest answer to "does it need a GPU?" is: **no, and we cannot prove it is viable
without one.** That is a bigger problem than "it needs a GPU", because it means the
evaluation has to be done from scratch before the dependency is even worth discussing.

## 4. NO ONNX PATH — verified, not assumed

`grep -rn -i "onnx"` across `*.py`, `*.toml`, `*.md` in the repository returns
**zero matches**. There is no ONNX export, no ONNX loader, no ONNX example.

Every model is a `torch.nn.Module` (a `lightning.LightningModule`) serialised as
`pytorch_model.bin`. There is no `safetensors` export of the pipelines in the repo either,
despite `safetensors>=0.5.2` being a dependency (it is used for the bundled WeSpeaker
weights, not the pipelines).

**Answer to the brief's question: pyannote has neither a numpy-only nor an ONNX-only path.
It is PyTorch-only.**

## 5. The dependency wall

`pyproject.toml:10-32` — the base dependencies, none optional:

```
asteroid-filterbanks, einops, huggingface-hub, lightning, opentelemetry-api,
opentelemetry-sdk, opentelemetry-exporter-otlp, pyannote-core, pyannote-database,
pyannote-metrics, pyannote-pipeline, pytorch-metric-learning, rich, safetensors,
torch-audiomentations, torch>=2.8.0, torchaudio>=2.8.0, torchcodec>=0.7.0,
torchmetrics>=1.6.1, matplotlib, pyannoteai-sdk
```

**21 packages, 4 of them large, and torch is one of them.** Compare:

| | openWakeWord | silero-vad | pyannote-audio |
|---|---:|---:|---:|
| base deps that pull torch | 0 | **1 (hard, unconditional)** | 1 |
| extra large deps | 2 (`ai-edge-litert`, `speexdsp-ns`) | 0 | ~3 (`lightning`, `torchmetrics`, `torch-audiomentations`) |
| packages with no ONNX path | 0 | 0 | **all** |
| our runtime (numpy + yaml only) can run it | no | **yes, via vendored wrapper** | **no** |

`lightning` in particular is a training framework, pulled into a *runtime* dependency
list. Installing pyannote-audio in our runtime is not "a bigger ask than pip install
torch" — it is a substantial fraction of the PyTorch ecosystem plus a HuggingFace account
plus a network fetch at model-load time.

## 6. Is diarization even the right job for a hot path?

Separate from feasibility. A peripheral that "must watch audio continuously and cheaply"
is asking a **detection** question: *did someone address me?* Diarization answers a much
harder question — *who spoke when* — which requires:

1. a segmentation model over every frame (5.633 MB),
2. a speaker **embedding** for each detected region (25.412 MB),
3. a clustering step to group embeddings into speakers (PLDA + agglomerative).

Step 3 is the killer for a hot path: clustering is **order-dependent and not incremental**.
You cannot diarize a stream forever without unbounded state or periodic re-clustering,
because whether two regions are the same speaker can only be decided after seeing both.
Our watcher is explicitly built the other way round — `push()` discards everything already
emitted except the tail so memory does not grow with session length
(`src/watcher.py:126-128`). Diarization's core loop is structurally incompatible with that
property.

If we ever want speaker awareness, the cheap version is a **verifier on the onset**, not
diarization on the stream: openWakeWord's own `custom_verifier_model` idea
(embeddings already computed, one classifier, per-utterance) is the shape that fits.

## 7. CI, tests, honesty

- `.github/workflows/test.yml:3-11`: triggers on `pull_request` → `develop` and
  `push` → `develop`, `master`, `release/*`. **Real triggers.**
- `run: uv run --extra test pytest` (line 36) — bare pytest, no masking.
- `grep -rn "|| true\|continue-on-error"` over `.github/` → nothing.
- Installs FFmpeg in CI (line 29-30), consistent with the `torchcodec` dependency.

**Honest CI — better than silero-vad's manual trigger.** Note one asymmetry: CI does not
run on pushes to `main`, only `develop`/`master`/`release/*`.

## 8. Source vs prose

Whole tree at the pinned commit:

| | files | bytes |
|---|---:|---:|
| `.py` | 103 | 798,995 |
| `.md` | 12 | 47,166 |
| `.ipynb` | 18 | 5,264,103 |

**16.9× more code than prose** — a serious research codebase. It is the largest of the
three by a wide margin (2.6× openWakeWord's Python, 6.6× silero's) and it is the only one
whose *notebooks* exceed its source by 6.6×, which is what a research group looks like.

## 9. Verdict for the hot path

| Question | Answer |
|---|---|
| Model size | 31.301 MB for the community-1 pipeline; 5.632 MB for segmentation alone |
| Measured latency | **Not published on CPU.** Only on an NVIDIA H100 80GB |
| CPU-only viable | Unknown. Runs on CPU; no evidence it fits a single core |
| Licence | Code MIT. **Weights CC-BY-4.0/MIT — commercially usable, but GATED behind click-through**; the 25 MB embedding carries WeSpeaker's own terms |
| Runs without a DL framework | **No. Zero ONNX anywhere in the repository** |
| Fits an all-day peripheral | **No — clustering is not incremental, so it cannot be a bounded always-on loop** |

**Rejected, and the reason is structural rather than merely practical.** It fails the
environment test (PyTorch-only, 21 deps), fails the size test (14× silero), has no CPU
evidence, adds a third-party licence click-through to the boot path, and its core
clustering step is incompatible with the bounded-memory property our watcher is
deliberately built around.

That said, two things in it are worth taking conceptually: the **PLDA + agglomerative**
pairing as a cheap non-neural alternative to a full embedding network for
"is this the same speaker", and the **diarization DER** metric as the evaluation standard
if we ever want to claim speaker awareness. Both are ideas, not dependencies.