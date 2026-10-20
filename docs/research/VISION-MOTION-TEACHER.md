# THE VISION MOTION TEACHER — does a SEQUENCE of frames beat a single frame?

**Code: `experiments/vision_motion_teacher.py`. Measured 2026-10-20.**
Casey: *"the current stimulus gives the model one still image and asks it
about movement. The fix is to give it the evidence it needs: a short sequence
of consecutive frames, in order, so the motion is visible."*

---

## 1. The question, and the hole in the previous experiment

`experiments/cloud_multimodal.py` measures whether a vision-language model can
tell "this person is SPEAKING" from "this person is not speaking" off a single
160×96 greyscale webcam frame. The published result is Llama 3.2 Vision 11B,
sensitivity **0.4545**, FPR **0.0000** on a person-present-but-silent session,
headline margin **+0.4545 — "USABLE"**.

But look at the question that was actually asked:

> *"Is the person in this image SPEAKING (their mouth is **moving** or open, as
> if talking) or NOT SPEAKING (mouth closed, still)?"*

**"Their mouth is moving" is a claim about time. A single frame cannot show
motion.** The experiment gives the model one still image and scores whatever
it says about movement. Whether the 0.4545 is the model reading motion, or the
model reading a single frame's *mouth-open* cue and calling it "speaking", is
not distinguishable by that experiment.

This is the same shape as the OpenVLHarness result Casey pointed at
(https://huggingface.co/papers/2610.26860): a text description of an image is a
weak substitute for retaining the image, and the gain comes from the
capability layer that renders tool output into model-readable evidence. Here
the capability layer is **motion evidence**, and the evidence is frames.

**The hypothesis under test:** the extra frames are doing nothing, and the
published margin is a mouth-open cue read off a single still.

## 2. What was measured, and against what

The model, the frames, the question, the request shape and the moment count
are held fixed. Only the **arrangement of the same pixels** changes.

| stimulus | what it is | what it controls |
|---|---|---|
| `single` | one 160×96 frame | the published setup, re-measured here |
| `single_up` | the same frame, bilinear ×4 → 640×384 | resolution, not content |
| `strip_k4` | 4 **consecutive** frames, time order, tiled → 1280×192 | motion evidence |
| `strip_k4_shuffled` | **the same 4 frames, scrambled** | **appearance kept, motion destroyed** |
| `strip_k4_repeated` | 4 copies of **one** frame | no temporal information at all |
| `strip_k8` | 8 consecutive frames, time order → 2560×192 | more motion evidence |
| `strip_k8_shuffled` | **the same 8 frames, scrambled** | as above, at K=8 |
| `strip_k8_repeated` | 8 copies of one frame | no temporal information |
| `grid_k4` | the same 4 frames in a 2×2 grid → 640×384 | same pixel count as `single_up` |
| `grid_k4_shuffled` | the same 4 frames, scrambled, in the grid | as above, in a readable shape |
| `strip_k4_torn` | 4 frames read at the **wrong stride** → torn | frames that do not exist |
| `strip_k4_torn_shuffled` | the same torn frames, scrambled | torn, in both orders |
| `strip_k8_torn` | 8 torn frames, in order | as above, at K=8 |
| `strip_k8_torn_shuffled` | the same torn frames, scrambled | as above, at K=8 |

**The conditions** (all with the *same* number of moments, 12, so every rate
has the same denominator):

| condition | what it is | source |
|---|---|---|
| `speak` | frames inside DETECTED SPEECH | `A.nv12` / `A.wav` |
| `silent` | a person present, **not** speaking — the real negative class | `C.nv12` / `C.wav` |
| `turned` | head turned away, speaking — **reported separately, never pooled** | `B.nv12` / `B.wav` |
| `blank` | a frame of nothing — the prompt floor | generated |

**The degenerate baselines are mandatory**, and are written into the results
file so every margin is read against them:

| degenerate model | sensitivity | FPR (silent) | FPR (blank) | margin |
|---|---:|---:|---:|---:|
| always SPEAKING | 1.000 | 1.000 | 1.000 | **0.000** |
| always NOT SPEAKING | 0.000 | 0.000 | 0.000 | **0.000** |

**And the base-rate trap:** the published sensitivity of 0.4545 is not "45 % of
the way to 1.0". A constant-answer model produces a sensitivity too, in either
direction. Only the **margin** (sensitivity − FPR on a person who is present
and not speaking) is not fakeable by a constant.

**The resolution of this measurement is one stimulus out of twelve: 0.083.**
Every difference below should be read against that.

## 3. A dated correction: the frames were torn, and the stride is the story

**This is a correction, not a footnote, and it changes what the published
baseline means.**

Every reader in this repo assumed an NV12 frame is `W*H*3//2` = 23,040 bytes:
luma then *interleaved* chroma, standard 4:2:0. **The captures on disk are
not that.** They were written by `/mnt/c/proart_capture/nv12small.py` as

```
Y plane   15,360 bytes   (160×96)
U plane    1,920 bytes   (80×48, PLANAR — all of U, then all of V)
V plane    1,920 bytes
--------------------------
STRIDE    19,200 bytes   900 frames = 30.00 s at 30 fps
```

19,200 × 900 = 17,280,000 = exactly the file size, and the WAV beside each
capture is 29.981 s. Read at 23,040 instead, the same file yields 750 frames
for 30 s of audio, and 80 % of them are **torn**: rows from frame *k* mixed
with rows from frame *k+1*.

**The tell is one line of code.** The mean absolute difference between
*consecutive* frames of a static 30 fps webcam scene is a fraction of a grey
level. Measured here, on the real captures:

| capture | read at 19,200 (correct) | read at 23,040 (torn) | frames, correct → torn |
|---|---:|---:|---|
| A (speaking) | **0.545** | 43.560 | 900 → 750 |
| B (turned) | **0.433** | 37.740 | 900 → 750 |
| C (silent) | **0.367** | 42.935 | 900 → 750 |
| empty1 (empty room) | **0.005** | 39.295 | 4500 → 3750 |

The empty room at the right stride is 0.005 — a static scene. Every session at
the wrong stride lands on 37–43, and they all land on *the same number*,
because they are all measuring the tear and not the scene.

**So the "motion" the repo has been measuring at 160×96 was, to about 99 %,
stride artefact — and the published +0.4545 margin was measured on torn
frames.** Every number in this file is re-measured at the **19,200-byte**
stride, and the stride, the frame count and the mean |diff| are recorded in
`experiments/vision_motion_teacher_results.json` so the numbers carry the
identity of their inputs.

**The torn frames are also run, as a control.** `strip_k4_torn` and
`strip_k8_torn` are the *same anchors at the same instants* — the wrong-stride
read is mapped back to the wall-clock moment of each clean anchor
(clean frame *a* → torn frame ⌊a · 19200/23040⌋) — but with every frame torn.
If the model scores the torn strip like the clean one, the strip is carrying
**format**, not frames. And if the *torn* ordered strip beats the *torn*
shuffled strip, then the headline measure is not measuring motion.

**What the correction does NOT change.** A single torn frame still shows a
face, and a single clean frame still shows a face. The `single` arm is the arm
least affected by the defect, and it is the arm that can be compared with the
published number. It is reproduced below.

## 4. The result

**Llama 3.2 Vision 11B** (`@cf/meta/llama-3.2-11b-vision-instruct`), 12
moments per condition, 0 errors:

| stimulus | sens (speak) | FPR (silent) | FPR (blank) | **margin** | turned |
|---|---:|---:|---:|---:|---:|
| `single` | 0.583 (7/12) | **0.000** (0/12) | 0.000 (0/12) | **+0.583** | 0.833 |
| `single_up` | 0.667 (8/12) | 0.000 (0/12) | 0.000 (0/12) | +0.667 | 0.667 |
| `strip_k4` | 1.000 (12/12) | 0.583 (7/12) | **1.000** (12/12) | +0.417 | 0.250 |
| `strip_k4_shuffled` | 1.000 (12/12) | 0.500 (6/12) | 1.000 (12/12) | +0.500 | 0.333 |
| `strip_k4_repeated` | 1.000 (12/12) | 0.417 (5/12) | 0.917 (11/12) | +0.583 | 0.167 |
| `strip_k8` | 0.833 (10/12) | **0.000** (0/12) | 0.167 (2/12) | **+0.833** | 0.000 |
| `strip_k8_shuffled` | 0.667 (8/12) | 0.000 (0/12) | 0.000 (0/12) | +0.667 | 0.000 |
| `strip_k8_repeated` | 0.333 (4/12) | 0.083 (1/12) | 0.000 (0/12) | +0.250 | 0.000 |
| `grid_k4` | 1.000 (12/12) | 0.833 (10/12) | 0.000 (0/12) | +0.167 | 0.917 |
| `grid_k4_shuffled` | 1.000 (12/12) | 0.917 (11/12) | 0.000 (0/12) | +0.083 | 1.000 |
| `strip_k4_torn` | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) | **0.000** | 0.833 |
| `strip_k4_torn_shuffled` | 0.917 (11/12) | 1.000 (12/12) | 1.000 (12/12) | −0.083 | 1.000 |
| `strip_k8_torn` | 0.750 (9/12) | 0.333 (4/12) | 0.000 (0/12) | +0.417 | 0.250 |
| `strip_k8_torn_shuffled` | 0.917 (11/12) | 0.333 (4/12) | 0.083 (1/12) | +0.583 | 0.250 |

**Mistral Small 3.1 24B** (`@cf/mistralai/mistral-small-3.1-24b-instruct`), a
different model family, 12 moments per condition, 0 errors:

| stimulus | sens (speak) | FPR (silent) | FPR (blank) | **margin** | turned |
|---|---:|---:|---:|---:|---:|
| `single` | **0.083** (1/12) | 0.000 (0/12) | 0.000 (0/12) | +0.083 | 0.000 |
| `single_up` | 0.750 (9/12) | 0.000 (0/12) | 0.000 (0/12) | +0.750 | 1.000 |
| `strip_k4` | 0.750 (9/12) | 0.000 (0/12) | 0.000 (0/12) | +0.750 | 0.583 |
| `strip_k4_shuffled` | 0.750 (9/12) | 0.000 (0/12) | 0.000 (0/12) | +0.750 | 0.583 |
| `strip_k4_repeated` | 0.583 (7/12) | 0.000 (0/12) | 0.000 (0/12) | +0.583 | 0.500 |
| `strip_k8` | 1.000 (12/12) | 0.417 (5/12) | 0.000 (0/12) | +0.583 | 1.000 |
| `strip_k8_shuffled` | 1.000 (12/12) | 0.417 (5/12) | 0.000 (0/12) | +0.583 | 1.000 |
| `strip_k8_repeated` | 1.000 (12/12) | 0.667 (8/12) | 0.000 (0/12) | +0.333 | 1.000 |
| `grid_k4` | 0.833 (10/12) | 0.000 (0/12) | 0.000 (0/12) | +0.833 | 0.750 |
| `grid_k4_shuffled` | 0.833 (10/12) | 0.000 (0/12) | 0.000 (0/12) | +0.833 | 0.667 |
| `strip_k4_torn` | 0.833 (10/12) | 0.000 (0/12) | 0.000 (0/12) | +0.833 | 0.833 |
| `strip_k4_torn_shuffled` | 0.833 (10/12) | 0.000 (0/12) | 0.000 (0/12) | +0.833 | 0.750 |
| `strip_k8_torn` | 1.000 (12/12) | 0.500 (6/12) | 0.000 (0/12) | +0.500 | 1.000 |
| `strip_k8_torn_shuffled` | 0.833 (10/12) | 0.083 (1/12) | 0.000 (0/12) | +0.750 | 1.000 |

**GLM 5.3 Flash** (`@cf/zai-org/glm-5.3-flash`), a third family, 12 moments per
condition, 0 errors — **and 154 of its 672 answers were unparseable prose**
("I'm unable to answer this question…"), which are counted in the denominator
and broken out per condition in the results file:

| stimulus | sens (speak) | FPR (silent) | FPR (blank) | **margin** | turned |
|---|---:|---:|---:|---:|---:|
| `single` | 0.333 (4/12) | 0.000 (0/12) | 0.083 (1/12) | +0.333 | 0.750 |
| `single_up` | 1.000 (12/12) | 0.000 (0/12) | 0.000 (0/12) | +1.000 | 1.000 |
| `strip_k4` | 0.917 (11/12) | 0.000 (0/12) | 0.000 (0/12) | +0.917 | 0.917 |
| `strip_k4_shuffled` | 0.917 (11/12) | 0.000 (0/12) | 0.000 (0/12) | +0.917 | 0.917 |
| `strip_k4_repeated` | 0.833 (10/12) | 0.000 (0/12) | 0.083 (1/12) | +0.833 | 0.917 |
| `strip_k8` | 1.000 (12/12) | 0.000 (0/12) | 0.167 (2/12) | +1.000 | 1.000 |
| `strip_k8_shuffled` | 0.917 (11/12) | 0.000 (0/12) | 0.083 (1/12) | +0.917 | 1.000 |
| `strip_k8_repeated` | 0.833 (10/12) | 0.000 (0/12) | 0.000 (0/12) | +0.833 | 0.833 |
| `grid_k4` | 0.750 (9/12) | 0.000 (0/12) | 0.000 (0/12) | +0.750 | 0.667 |
| `grid_k4_shuffled` | 0.917 (11/12) | 0.000 (0/12) | 0.250 (3/12) | +0.917 | 0.917 |
| `strip_k4_torn` | 0.833 (10/12) | 0.000 (0/12) | 0.000 (0/12) | +0.833 | 0.917 |
| `strip_k4_torn_shuffled` | 0.833 (10/12) | 0.000 (0/12) | 0.000 (0/12) | +0.833 | 0.917 |
| `strip_k8_torn` | 1.000 (12/12) | 0.000 (0/12) | 0.083 (1/12) | +1.000 | 0.833 |
| `strip_k8_torn_shuffled` | 1.000 (12/12) | 0.000 (0/12) | 0.083 (1/12) | +1.000 | 0.917 |

## 5. The headline: the ordered strip does NOT beat the shuffled strip

The headline is the margin of the ordered strip over the shuffled strip, on
the speaking condition, at matched FPR. Materiality threshold **0.20**,
pre-registered before the run. FPR matching means the two strips ran at a
false-positive rate within one stimulus of each other.

**Llama 3.2 Vision 11B:**

| arm | margin (ordered) | margin (shuffled) | **Δ margin** | Δ sensitivity | FPR matched |
|---|---:|---:|---:|---:|---|
| K=4, horizontal strip | +0.417 | +0.500 | **−0.083** | +0.000 | yes (7 vs 6 of 12) |
| K=8, horizontal strip | +0.833 | +0.667 | **+0.167** | +0.167 | yes (0 vs 0 of 12) |
| K=4, 2×2 grid | +0.167 | +0.083 | **+0.083** | +0.000 | yes (10 vs 11 of 12) |
| K=4, **torn** | 0.000 | −0.083 | +0.083 | +0.083 | yes (12 vs 12) |
| K=8, **torn** | +0.417 | +0.583 | **−0.167** | −0.167 | yes (4 vs 4 of 12) |

**Mistral Small 3.1 24B:**

| arm | margin (ordered) | margin (shuffled) | **Δ margin** | Δ sensitivity | FPR matched |
|---|---:|---:|---:|---:|---|
| K=4, horizontal strip | +0.750 | +0.750 | **+0.000** | +0.000 | yes (0 vs 0) |
| K=8, horizontal strip | +0.583 | +0.583 | **+0.000** | +0.000 | yes (5 vs 5) |
| K=4, 2×2 grid | +0.833 | +0.833 | **+0.000** | +0.000 | yes (0 vs 0) |
| K=4, **torn** | +0.833 | +0.833 | **+0.000** | +0.000 | yes (0 vs 0) |
| K=8, **torn** | +0.500 | +0.750 | −0.250 | +0.167 | **no** (6 vs 1) |

**GLM 5.3 Flash:**

| arm | margin (ordered) | margin (shuffled) | **Δ margin** | Δ sensitivity | FPR matched |
|---|---:|---:|---:|---:|---|
| K=4, horizontal strip | +0.917 | +0.917 | **+0.000** | +0.000 | yes (0 vs 0) |
| K=8, horizontal strip | +1.000 | +0.917 | **+0.083** | +0.083 | yes (0 vs 0) |
| K=4, 2×2 grid | +0.750 | +0.917 | **−0.167** | −0.167 | yes (0 vs 0) |
| K=4, **torn** | +0.833 | +0.833 | **+0.000** | +0.000 | yes (0 vs 0) |
| K=8, **torn** | +1.000 | +1.000 | **+0.000** | +0.000 | yes (0 vs 0) |

**So: no.** In the fifteen arm/model combinations, the ordered strip beats the
shuffled strip in **four** — by +0.167 and by +0.083 three times — and **all
four are below the pre-registered 0.20 threshold**; one of the four is on the
**torn** strip, where the frames do not exist at all. In **seven** the two are
exactly equal. In **four** the ordered strip is **worse** than the shuffled
one. The largest positive Δ margin anywhere is +0.167 (two stimuli out of
twelve), and the largest negative is −0.250.

**The hypothesis is falsified.** For these models, at 160×96, the extra frames
carry no motion information. The published +0.4545 margin is not a motion
reading.

## 6. What DID move the numbers: image size, not temporal order

The null above is only interesting next to what *does* work, and the table
says it plainly.

**Mistral Small 3.1 is nearly blind at 160×96 and sharp at 640×384.** The
same frame, bilinear-upscaled ×4, takes its sensitivity from **0.083 (1/12) to
0.750 (9/12)** — nine times — with the FPR on both negative classes still
0.000. GLM goes **0.333 → 1.000** the same way. That is a **resolution**
result, and it is the cleanest positive finding in this file: at this camera's
native resolution these models cannot see the mouth, and giving them four
times the pixels lets them see it.

**And the same size effect shows up in the strips.** For Mistral, every
multi-panel stimulus scores between +0.583 and +0.833 and every ordered/
shuffled pair is *exactly equal*. For Llama 3.2, `single` → `single_up` →
`strip_k8` goes 0.583 → 0.667 → 0.833, again with the shuffled partner right
behind the ordered one. What the model responds to is **how many pixels the
mouth occupies**, not **whether the mouth moved**.

**The one thing that is not size.** At K=8, *all three* models score the
`strip_k8_repeated` control below the `strip_k8` strip that contains eight
distinct frames (Llama 3.2: +0.250 vs +0.833; Mistral: +0.333 vs +0.583;
GLM: +0.833 vs +1.000). So the presence of several **distinct** frames is
worth something — but **scrambling their order costs nothing**. Llama 3.2's
ordered K=8 strip beats its shuffled K=8 strip by 0.167 (2 stimuli), Mistral's
two are identical to the stimulus, and GLM's by 0.083 (1 stimulus). The model
is reacting to *variety across the panels*, not to *time running through
them*.

## 7. The strip format is itself a prompt floor, and the torn control proves it

`strip_k4_torn` is four **torn** frames — the wrong-stride read, at the same
instants as the clean strip. For Llama 3.2 it scores:

- sensitivity 1.000, FPR on the silent person 1.000, FPR on the **blank**
  1.000, **margin 0.000** — *exactly* the `always_SPEAKING` degenerate
  baseline, to three decimal places.

A 1280×192 strip of four blank panels makes this model answer SPEAKING. The
wide strip is not a neutral container for evidence; it is a stimulus that
drags the answer to a constant. That is why the shuffled control is not
optional: a strip arm that is degenerate on the blank would look like a
"margin of +0.417" (which is what `strip_k4` shows) and is not one.

The grid arm was added precisely to separate "the frames do not help" from
"the 6.67:1 shape does not help": `grid_k4` is the same four frames at the
same tile scale in a 2×2 arrangement, which makes it **640×384 — the exact
dimensions and pixel count of `single_up`**. It fixed the blank FPR (1.000 →
0.000) and made the model say SPEAKING on the silent person 10 times out of
12 instead of 7. **Neither shape gives a good margin at K=4.**

## 8. Models: what was used, and what failed

All 73 catalogue entries of account `049ff5e84ecf636b53b162cbb580aae6` were
called once with this exact request shape. Four accepted an image:

| model | outcome |
|---|---|
| `@cf/meta/llama-3.2-11b-vision-instruct` | **run** — the published baseline |
| `@cf/mistralai/mistral-small-3.1-24b-instruct` | **run** — second, different family |
| `@cf/zai-org/glm-5.3-flash` | **run** — third family, at ~18 calls/min |
| `@cf/meta-llama-4-scout-17b-16e-instruct` | **unavailable** |

**The model named in the brief does not exist on this account.**
`@cf/meta/llama-4-scout-17b-vision-instruct` is not a catalogue entry. The
catalogue entry is `@cf/meta-llama-4-scout-17b-16e-instruct` — the model the
published "+0.167, NOT usable" number was measured on — and **its run route is
intermittently absent**: it answered normally on some probes and returned
Cloudflare error code 7000 "No route for that URI" on others, repeatedly and
consistently through the run window. It was pre-flighted before the run and
dropped, with the pre-flight evidence recorded in the results file. **The
second model in this file is Mistral Small 3.1, and the third is GLM 5.3
Flash**, and that substitution is a fact about the account, not a choice.

**GLM 5.3 Flash needed a scheduler.** It accepts the image shape and is
genuinely multimodal, but the account rate-limits it to about 18 inference
calls per minute while the experiment needs 672 — and a fixed interval that
is too fast for one model turns that model's calls into a retry storm, where
the run gets slow and the arm comes back full of errors, which reads like a
model failure and is really a scheduling failure. The run therefore paces
each model adaptively: a rate-limit error doubles that model's interval and a
success walks it back down. **And 154 of GLM's 672 answers were unparseable
prose**, so its rates are computed over all twelve moments per condition with
the unparsed count broken out, and they should be read with that in mind.

Rejected at the request-shape level, with the error recorded:
`@cf/llava-hf/llava-1.5-7b-hf` (oneOf not met — the known failure),
`@cf/moondream/moondream3.1-9B-A2B` (HTTP 200, empty result), `@cf/zai-org/
glm-5.3` (empty content; its own reasoning says it has no multimodal input),
`@cf/cloudflare/clairic` (needs `{model, fields}`), `@cf/cloudflare/clef`,
`clef-flash`, `clef-omni` (need `{model, state, questions}`), `@cf/qwen/qwq-32b`
and `@cf/zai-org/glm-4.7-flash` and `glm-5.2` ("not a multimodal model"),
`@cf/deepseek-ai/deepseek-v4-flash-0731` (text only), `@cf/swiss-ai/apertus-
v1.5-8b` (403, account not allowed). Every remaining catalogue entry rejects
the image content type.

## 9. What was falsified

- **"A single frame cannot show motion, so the model was probably reading a
  mouth-open cue rather than motion."** — *confirmed, and then some.* Feeding
  the model real consecutive frames in time order does **not** improve on the
  single frame by more than 0.167 in margin, and the shuffled control shows
  why: the frames were not buying motion, they were buying pixels.
- **"A strip of frames gives the model motion evidence."** — **false** for
  these models at this resolution. Fifteen arm/model combinations, four
  positive effects of one or two stimuli, seven exactly zero, four negative.
- **"The 0.4545 margin is a stride artefact."** — **false.** The clean-frame
  `single` arm gives **+0.583** (7/12 sensitivity, 0/12 on both negative
  classes), slightly *better* than the published +0.4545. The stride defect
  did not manufacture the published margin. What it did manufacture is the
  repo's *motion* numbers (§3).
- **"The wide strip is a neutral container."** — **false.** A 1280×192 strip
  of four blank panels makes Llama 3.2 answer SPEAKING 12 times out of 12.
- **"Upscaling might not help."** — **false for Mistral, and strongly.**
  0.083 → 0.750 sensitivity from the same frame at 4× the linear size; and
  GLM goes 0.333 → 1.000 the same way.
- **"The null is one model's quirk."** — **not supported.** Three model
  families, fourteen stimuli, the same null.

## 10. What is NOT claimed

- One session per condition, one speaker, one room, one lighting state.
  Twelve moments per condition, and frames from one session are correlated,
  so the effective sample size is far below twelve. **The resolution of every
  number here is one stimulus: 0.083.** GLM 5.3 Flash's 154 unparseable
  answers out of 672 make its own numbers the softest of the three.
- The speech labels come from the Silero gate applied to the WAV. Per
  `AV-SYNC.md` A.wav and A.nv12 do not correspond, so the per-frame labels
  are provisional. The corrected stride makes the *durations* agree to 0.02 s,
  which is necessary but not sufficient for frame-level correspondence.
- The model's own internal image resizing is not observable from here, so
  what the model actually sees after resizing is unknown. The **order** of the
  tiles survives any whole-image resize, which is why the shuffled control is
  valid; the resolution question is not settled by it.
- The `turned` condition is reported and **not** pooled. It is worth noting
  and not worth claiming: Llama 3.2 says SPEAKING on 10/12 of the *turned*
  frames against 7/12 of the facing-and-speaking ones, i.e. it is **more**
  willing to say SPEAKING when the head is away. That is a separate question
  from this one.
- One measurement of one model on one capture is one measurement, not a
  property. The nulls here are nulls about **three models, one capture, one
  resolution, one question** — not about vision-language models.

## 11. What it changes

- **The published cloud teacher still stands on its own single frame**, and
  now has a clean-frame re-measurement (+0.583) and a control that explains
  what it was reading: a mouth-open cue, visible once the mouth is big enough
  to see.
- **The teacher to build is a RESOLUTION teacher, not a motion teacher.**
  A 160×96 frame is below the threshold at which this class of model can read
  a mouth at all (Mistral: 1 of 12). Upscaling is what buys the sensitivity,
  and it costs nothing but pixels.
- **If the project wants motion evidence out of a vision model, the tile
  format has to be designed around the model's own tiling.** Llama 3.2 Vision
  has no multi-image interface; it tiles ONE image, so a 6.67:1 strip arrives
  as a wide band of arbitrary crops, and a 2×2 grid arrives as something it
  can actually read. The shape of the container is not neutral and the blank
  control is what shows it.
- **The stride correction reaches past this file.** Every motion number in
  this repo that came from these three captures is a tear, and the empty-room
  capture says so in one line: 0.005 at the right stride, 39.295 at the
  wrong one.
