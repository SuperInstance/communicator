# LOCAL VLM DETERMINISM — is the director's instability the serving stack's, or the task's?

**Code: `experiments/local_vlm_vision.py`. Measured 2026-10-20.**

---

## 1. The question

`docs/research/LLM-DIRECTOR.md` reports the director's headline as if it were a
property of the model, and §4 then records that it is not:

| | runs | distinct headlines | spread |
|---|---:|---:|---:|
| hosted `@cf/openai/gpt-oss-120b`, BLIND audit, temperature 0 | 8 | **4** | **0.619** |
| deterministic stub director, same scorer, same 8 runs | 8 | 1 | **0.000** |

Those are **the repo's committed measurements**, from
`experiments/director_repeatability_blind.json`. Nothing was re-measured here
and no Cloudflare token was used; the hosted row is copied into the artefact so
that every row of the comparison table in §3.1 comes from one file.

The note says *"this is hosted-inference non-determinism"*. That sentence was
an **inference**, and two explanations fit the same eight numbers:

- **(A) hosted-inference non-determinism** — the serving stack returns
  different answers for identical requests;
- **(B) the task is ambiguous** — any model asked it would answer differently,
  because the question admits several defensible readings.

They make **different predictions**. A **local** model with greedy decoding is
deterministic by construction: same weights, same input, same arithmetic, same
answer. If a local model at temperature 0 gives a spread of **0.000** on the
same prompt, then (B) is not the explanation and the instability belongs to
the serving stack. If the local model *also* gives a large spread at
temperature 0, then the task itself is the problem.

---

## 2. The instrument

**The local model.** `Qwen/Qwen2-VL-2B-Instruct`, revision
`895c3a49bc3fa70a340399125c650a463535e71c`, weights SHA-256
`148ca8369dd703e7bcef883c40f1ec7bc8ae36de520a3da382b4076073f90058`
(4,418,050,768 bytes), **float16**, on `cuda:0` of an RTX 4050 Laptop GPU
(6,140 MiB), 4.418 GB resident — **wholly on the GPU**, no CPU offload. The
weights are the whole picture: the SHA-256 is a hash over the safetensors
files, so the number below is attached to bytes and not to a model name.

**A second model, as a control on the model.** `Qwen/Qwen2.5-VL-3B-Instruct`,
revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, weights SHA-256
`0891a2d0e1a6f8132afe046017eef6b86e3e8f803b98d1a3e726722a077bb804`
(7,509,337,976 bytes), **float16**, `device_map="auto"` — 4.58 GB resident on
the GPU with **21 of 41 modules offloaded to the CPU**, because the weights
are 7.51 GB and the GPU has 6.14 GB. It is the **secondary** model, for the
reason in §7.

**The prompt is the hosted prompt, byte for byte.** Built by the repo's own
`debrief.build_audit_prompt('blind', pairs, None)`, SHA-256
`5870ad6bb1989c07727c57ea01c417aa0a014674824d48ee8a692665e6e1100c`, 3,031
chars — **the same SHA-256 as the committed hosted artefact**, and
`materials_chars = 0`: the BLIND prompt is the ten claims and nothing else.
**No research note, no result file, and no video frame enters it.**

That last point matters for what this measurement can settle, and it is easy
to get wrong: the audit arm is **text-only**, so the frame-stride defect in
§4 **cannot reach it**. A finding about the audit cannot be explained away by
the frames being torn.

**The regimes.** `greedy` is `do_sample=False`. `sampled` is `do_sample=True`,
`temperature=0.8`, `top_p=1.0`, `top_k=0`, with the seed set **once per
regime** (20261019) so that the N runs are N draws from one seeded stream —
the local analogue of N separate requests. Re-seeding before every run would
make the sampled regime identical to itself and the measurement vacuous.

**The scorer is the repo's.** `debrief.score_audit` scores the verdicts and
`director_repeatability.summarise_runs` computes the spread, so the number in
the table is produced by the same code that produced the hosted number.

**The control.** The repo's own deterministic stub director, scored N times by
the repo's own scorer: **spread 0.000, 1 distinct headline in 8**. And a
constant vision answer through the same spread statistic: **0.000**. A harness
that cannot produce 0.000 on a deterministic input is not measuring anything.

---

## 3. The results

### 3.1 The audit — the decisive comparison

**Local:** 8 runs per regime, the byte-identical BLIND audit prompt.

| model | regime | scorable | distinct headlines | **spread** | unstable claims |
|---|---|---:|---:|---:|---:|
| `Qwen2-VL-2B` | **greedy** (T=0) | 8/8 | **1** | **0.000** | **0 of 10** |
| `Qwen2-VL-2B` | sampled (T=0.8) | 7/8 | **5** | **0.952** | **10 of 10** |
| `Qwen2.5-VL-3B` | **greedy** (T=0) | 8/8 | **1** | **0.000** | **0 of 10** |
| `Qwen2.5-VL-3B` | sampled (T=0.8) | 8/8 | **7** | **0.810** | **7 of 10** |

The 2B's greedy headline is `-0.667` on all eight runs and its sampled
headlines are `-1.000, -0.238, -0.095, -1.000, -0.048, -0.095, -0.429`. The
3B's greedy headline is `0.000` on all eight runs and its sampled headlines are
`-0.333, -0.190, -0.333, -0.048, +0.143, -0.667, -0.333, -0.667`.

**The decisive comparison table.**

| | greedy (T=0) | sampled (T>0) |
|---|---|---|
| hosted `@cf/openai/gpt-oss-120b` (the repo's committed measurement) | spread **0.619** | **NOT MEASURED** |
| **local** `Qwen/Qwen2-VL-2B-Instruct` | spread **0.000** | spread **0.952** |
| **local** `Qwen/Qwen2.5-VL-3B-Instruct` | spread **0.000** | spread **0.810** |

The hosted-sampled cell is empty **on purpose**. The hosted model at
temperature > 0 was never run by this project, and it cannot be measured from
this machine. It is left empty rather than guessed, and the test asserts it
is empty.

**Two local models, two different personalities, the same determinism
result.** The 2B hedges (`uncertain` on 7 of 10 claims) and the 3B does not
(`supported` on 9 of 10). They do not agree about the claims. They agree
exactly about being repeatable.

### 3.2 What the local models actually are

They are **poor directors**, and the note has to say so. The 2B's greedy
headline of `-0.667` is `sensitivity 0.000 + specificity 0.333 − 1`: it
rejected **none** of the seven known-false claims and endorsed one of the
three known-true ones. Seven of its ten verdicts are `uncertain`. The 3B is
confident and no better: its headline of `0.000` is `sensitivity 0.000 +
specificity 1.000 − 1`, so it kept all three true claims and **endorsed six
of the seven false ones**.

This is a demonstration about **determinism**, not about the task being
well-posed and not about the models being any good. A model that answers
`uncertain` to almost everything is not evidence that the claims are unclear;
it is evidence that the model is small. A model that endorses six of seven
known-false claims is not evidence that the claims are clear; it is evidence
that the model is worse.

Both do, however, **discriminate**: each greedy arm uses two different
verdict labels, so each constant headline is a property of the model's
answer, not of a degenerate constant. The verdict logic checks that, and
would have answered *undetermined* if a model had answered every claim
identically.

### 3.3 The vision task

The cloud bench's question, verbatim, on the luma plane of this project's own
captures, at **11 frames per condition and equal denominators**:

> *"Is the person in this image SPEAKING (their mouth is moving or open, as
> if talking) or NOT SPEAKING (mouth closed, still)? Answer with exactly one
> word: SPEAKING or NOT SPEAKING."*

| condition | source | 2B said SPEAKING | 3B said SPEAKING |
|---|---|---:|---:|
| `speak` | `A.nv12`, VAD-aligned | 11/11 | 11/11 |
| `silent` | `C.nv12`, person present, not speaking | 11/11 | 11/11 |
| `blank` | flat grey 128 | 11/11 | 11/11 |

**Both local models: sensitivity 1.000, FPR on silent 1.000, FPR on blank
1.000, margin 0.000.**

The degenerate baselines, computed and recorded:

| director | sensitivity | FPR silent | FPR blank | margin |
|---|---:|---:|---:|---:|
| **always-SPEAKING** | 1.000 | **1.000** | **1.000** | 0.000 |
| **always-NOT-SPEAKING** | **0.000** | 0.000 | 0.000 | 0.000 |

**Both local models score exactly the always-SPEAKING baseline.** Their margin
over the silent class is **0.000**, the same as a model that says SPEAKING to
everything, which is what they do. **There is no usable local vision signal
here**: at 160×96, upscaled, a 2B and a 3B model answer SPEAKING to a closed
mouth and to a blank frame alike. The cloud bench's llama-3.2-11b-vision did
*better* than either (sensitivity 0.455, margin 0.455) on frames that were
torn (§4) — so the local models are not merely handicapped by the frames.

### 3.4 Clean frames against torn frames

The parent lane's correction (§4) says the repo's readers tear the frames.
Both readings were therefore measured, as two cells of one table rather than
one silent replacement of the other.

| model | stride | frames | fps | greedy spread | sampled spread |
|---|---|---:|---:|---:|---:|
| `Qwen2-VL-2B` | **clean 19,200** | 900 | 30.02 | **0.000** | **1.000** |
| `Qwen2-VL-2B` | torn 23,040 (repo's assumed stride) | 750 | 25.02 | **0.000** | **1.000** |
| `Qwen2.5-VL-3B` | **clean 19,200** | 900 | 30.02 | **0.000** | **1.000** |
| `Qwen2.5-VL-3B` | torn 23,040 (repo's assumed stride) | 750 | 25.02 | **0.000** | **1.000** |

Identical, in every condition (`speak`, `silent`, `blank`), for both models at
both strides. The **sampled answer sequences are the same sequence of tokens**
at both strides, for each model.

One small difference is worth naming because it is a **control that came out
the other way**: the 3B's sampled answers are *not* identical across
conditions — it says SPEAKING on 0.625 of the sampled `speak` runs but only
0.375 of the sampled `silent` and `blank` runs — and that difference is
**the same at the clean stride and at the torn stride**. So a torn frame does
not move the answer distribution; and where the distribution does move at all,
it moves with the condition, not with the stride.

**So the stride defect is not what makes the vision answer unstable, and it is
not what makes the local models degenerate.** A torn frame does not move the
answer distribution here — the models are answering from the prompt and from
prior, not from the picture, which is why the clean frames do not help them
either. The defect is still a real defect and it is still in every other
experiment in this repository; what it does **not** do is explain an
instability.

### 3.5 The control, and the sampled arm, together

| | greedy | sampled |
|---|---|---|
| deterministic stub director | **0.000** | — |
| deterministic stub vision answer | **0.000** | — |
| local 2B, audit | **0.000** | 0.952 |
| local 3B, audit | **0.000** | 0.810 |
| local 2B, vision | **0.000** | 1.000 |
| local 3B, vision | **0.000** | 1.000 |

The stub moves by nothing, so the harness adds no variance. The local models
move by **0.952** and **0.810** the moment they are allowed to sample, and by
nothing at all the moment they are not.

---

## 4. The correction, dated

> **CORRECTION (2026-10-20). The frame stride of the raw captures.** The
> captures on disk are **19,200 bytes per frame** — 15,360 of luma followed by
> 3,840 bytes of interleaved U,V, written by `/mnt/c/proart_capture/nv12small.py`
> — which is **900 frames at 30.02 fps**. The repo's readers assume the
> standard NV12 4:2:0 stride of **23,040 bytes**, which gives 750 frames, and
> at that stride **80% of every frame is a splice of two frames** (rows 0..71
> from one, rows 72..95 from the next).
>
> The evidence is physical, not documentary: the mean absolute difference
> between consecutive frames.
>
> | file | stride 19,200 | stride 23,040 | stride 15,360 |
> |---|---:|---:|---:|
> | `A.nv12` | **0.545** | 43.560 | 43.554 |
> | `B.nv12` | **0.433** | 37.740 | 37.730 |
> | `C.nv12` | **0.367** | 42.935 | 42.922 |
>
> The stride is now **detected per file** by that test, not assumed, and
> recorded; and the detector tries all three candidate layouts and reports
> every score. (A correction: this paragraph first said a third format
> exists because `experiments/corpus_capture_win.py` writes **luma only**,
> 15,360 bytes per frame. That is **false** — no script in this repo writes
> a luma-only frame stream; the capture tool writes `.npz` change-map
> archives. The `luma_only` layout is kept in the reader as something
> realisable, and it is measured here as the third arm.)
>
> **What this changes here:** the first run of this lane read the captures at
> 23,040 and at 25.00 fps. Its vision numbers (sensitivity 1.000, FPR silent
> 1.000, FPR blank 1.000, margin 0.000; greedy spread 0.000; sampled spread
> 1.000) are **superseded** and re-measured at both strides. They did not
> change, because the model was not reading the frames anyway (§3.4). The
> superseded run is kept in the artefact under `superseded_first_run`, not
> deleted.
>
> **What this does not change:** the audit arm. It is text-only (§2).

---

## 5. What this shows

**The evidence supports (A): the instability is a property of the serving
stack, not of the task.**

**The task is answerable **stably**, by two models that do not even agree
about it.** A local model, greedy, on the **byte-identical** prompt, gave
**one** headline out of eight, a spread of **0.000**, and **0 of 10 claims**
changed verdict — and the second local model, which is more confident and
gives a *different* verdict on every claim, did the same: **one** headline out
of eight, spread **0.000**, 0 of 10 claims unstable. Hypothesis (B) in its
strong form — *any model asked this would answer differently* — is therefore
**refuted**. The hosted 0.619 is not what ambiguity looks like.

**And the sampled arm says something sharper than "it is fine".** The local
models' spreads are **0.952** and **0.810** at temperature 0.8 and **0.000**
at temperature 0: they *have* a distribution over verdicts (10 of 10 claims
flip for the 2B when it is allowed to sample; 7 of 10 for the 3B), and greedy
decoding collapses that distribution to a point. A hosted stack that returns
four different headlines for four identical requests **at temperature 0** is
therefore behaving like a stack that is *not* actually greedy — or like one
whose arithmetic is not reproducible under its own batching. Either way the
thing that moved was the **stack**, not the model and not the claims.

**The local models are not the hosted model.** A 2B model answering
`uncertain` to seven of ten claims, and a 3B endorsing six of seven
known-false claims, are different instruments from a 120B model scoring
+0.429, and this measurement does **not** show that gpt-oss-120b would be
stable if it were run locally. What it shows is narrower and sufficient: **the
prompt admits a stable answer**, so "the task is ambiguous" cannot be the
explanation for the hosted spread.

---

## 6. What this does not show

- **It does not show the hosted model is locally stable.** It cannot; that
  model does not run here. The residual explanation this measurement cannot
  exclude is *model-specific* instability of gpt-oss-120b under hosted
  serving, as opposed to stack-wide non-determinism.
- **The hosted+sampled cell is unmeasured**, and stays unmeasured.
- **Two local models, one task, one session.** The local arm is a 2B and a 3B
  model on a ten-claim blind audit. `LLM-DIRECTOR.md` §8 already says the
  hosted repeatability measurement covers one model properly; this is the
  same limitation on the local side.
- **It does not measure the CONTEXT mode.** The 140,000-char prompt is the
  one that produced the quoted +0.857. The blind prompt is the one that
  produced the 0.619, and the blind prompt is the one that fits a 6 GB GPU
  with room to spare.
- **It does not clear the local models.** Sensitivity 0.000 on the false
  claims for both. They are stable answers to a task they are not good at.
- **It does not clear the frames.** The stride defect is real, it is in every
  other experiment in this repository, and §3.4 only shows that it does not
  explain *this* instability.

---

## 7. The models tried, and what happened to them

| model | revision | weights | outcome |
|---|---|---:|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` | `66285546d2b821cf421d4f5eb2576359d3770cd3` | 7,509,337,976 B | Ran. **fp16 weights are 7.51 GB and the GPU has 6.14 GB**, so `device_map="auto"` offloads **21 of 41 modules to the CPU**. It works, and it is the stronger model, but a model with half its weights in system RAM is a weaker demonstration of "deterministic by construction" than one wholly on the GPU, and it is roughly 13× slower per token — **234 s for one 549-token audit answer, against 24 s for the 2B**. It is the **secondary** model, and it was measured in a separate invocation because of that slowness (§8). Its result: **greedy spread 0.000, sampled spread 0.810** (§3.1). |
| `Qwen/Qwen2-VL-2B-Instruct` | `895c3a49bc3fa70a340399125c650a463535e71c` | 4,418,050,768 B | Ran. **4.42 GB resident on a 6.14 GB GPU, wholly on `cuda:0`.** The **primary**. |
| — | — | — | **Setup failure.** transformers 5.20.2 could not build the Qwen2-VL processor: *"Qwen2VLVideoProcessor requires the Torchvision library but it was not found"*. Fixed by installing `torchvision==0.21.0+cu124` into `/home/proart/projects/gpu-venv`, **outside the repository**. The repo's own `.venv` was not touched. |

No gated repository was used, and no model needing more than ~6 GB of VRAM was
run without offload.

**One more thing the harness had to fix, and it is in the record.** The **2B**'s
audit answers are JSON with an **unescaped double quote inside a `reason`
string**, so the outer object does not parse. `debrief.extract_json` returns
the **first** balanced `{...}` pair, which in that case is an *inner* verdict
object, and the repo's caller correctly rejects it as unparseable — so with
the strict parser alone, **0 of 8 greedy runs were scorable** and the
comparison could not have been made at all. This file therefore falls back to
scanning for every object carrying an `index` and a `verdict`, and **records
which parser each run used** (`repo_extract_json` or `fallback_scan`). The
repo's parser is always tried first, the fallback is applied to every run in
both regimes alike, and the count of runs that needed it is in the artefact.

The **3B** needed the fallback **not at all** — all sixteen of its audit runs
parsed with the repo's strict parser — so the fallback cannot be accused of
manufacturing the 3B's result. The hosted model's eight runs also all parsed
with the strict parser, so the hosted row is untouched by any of this.

---

## 8. Reproducing it

The primary, the secondary, and the merge are three scripted steps:

```
# 1. the primary model
/home/proart/projects/gpu-venv/bin/python experiments/local_vlm_vision.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --frames 11 --runs 8 --strides 19200,23040 \
    --out experiments/local_vlm_vision_results.json

# 2. the secondary model (slow: ~234 s per audit answer)
/home/proart/projects/gpu-venv/bin/python experiments/local_vlm_vision.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct --device-map auto \
    --frames 11 --runs 8 --strides 19200,23040 \
    --out /tmp/lvv_3b.json

# 3. merge the secondary's block into the artefact (scripted, not a hand edit)
/home/proart/projects/glasses/.venv/bin/python experiments/local_vlm_vision.py \
    --merge-secondary-from /tmp/lvv_3b.json \
    --out experiments/local_vlm_vision_results.json
```

Step 3 also exists in the `--recompute` form, which recomputes the summary
blocks from an artefact's **own recorded runs** with no re-measurement — which
is how a change in the analysis code is applied to a measurement that has
already been taken.

The artefact records, for every number above: the model id and revision, the
weights SHA-256, the dtype, the device, the exact prompt text and its
SHA-256, the generation parameters, the seed, the SHA-256 of every image's
bytes, the frame indices, the per-condition counts, and the per-run values
every spread is recomputed from.

`tests/test_local_vlm_vision.py` checks the artefact, and is network-free,
GPU-free and download-free:

```
/home/proart/projects/glasses/.venv/bin/python -m pytest tests/test_local_vlm_vision.py -q
```
