# VLM-IMAGE-RETENTION — describing the frame beat keeping it, and the blank room is why

**Date:** 2026-10-20
**Experiment:** `experiments/vlm_image_retention.py`
**Results:** `experiments/vlm_image_retention_results.json`
**Tests:** `tests/test_vlm_image_retention.py`
**Reader:** `src/nv12.py` — the shared reader; the frame layout is **detected** from the content, not assumed.

> ## The verdict, first
>
> **`no_difference_at_the_preregistered_threshold`**
>
> The pre-registered headline — the IMAGE arm's margin over its blank-room
> control minus the TEXT arm's — is **+0.8159** in the IMAGE arm's favour.
> That clears the ±0.05 margin threshold. **It does not clear the
> permutation gate: the paired permutation z is −0.41, against a required
> +2.0.** So the rule licenses neither "the image wins" nor "the text
> wins", and the verdict is a null.
>
> The numbers underneath the null are not null, and they do not all point
> the same way:
>
> * On the raw separation of the speaking take from the silent take, the
>   **TEXT arm was better**: AUC **0.9562** (permutation z **+12.26**)
>   against the IMAGE arm's **0.7784** (z **+7.43**). That is the
>   opposite of the paper's claim.
> * On the blank-room control, the **IMAGE arm passed and the TEXT arm
>   failed catastrophically**. The TEXT arm ranks the **empty room above
>   the silent person** — AUC **0.9970** — and above the speaking person
>   on the mean score (−0.145 against −0.320). The IMAGE arm ranks the
>   empty room *below* the silent person: AUC **0.0034**.
> * The pre-registered permutation null cannot tell those two facts
>   apart, because the null **preserves the blank-control term** and that
>   term is what dominates the headline. That is a property of the
>   statistic this lane pre-registered, and **§3.4** says exactly what it
>   does and does not license.

---

## 0. The question, and the claim it is testing

The external claim is the one in **"OpenVLHarness: Real World Visual Agentic
Tool-Use with Multimodal Memory"** (Yang et al., UCLA/Apple,
<https://huggingface.co/papers/2610.26860>), Appendix H:

> "The textual-memory step (+0.7 at 8B and +0.3 at 32B) is comparable to
> run-to-run variation, which is consistent with our conclusion that
> **text descriptions of images are a weak substitute for retaining the
> images themselves**."

and, in §2.2: *"Repeating a description of a previous result does not
provide this access."*

**This repository is the case that claim is about, and it had never tested
it.** The project has spent its life converting camera frames into
TEXT/ASCII representations and asking a language model to judge them
(`docs/research/ASCII-MOTION.md`, `experiments/teacher_conventional.py`,
`experiments/ascii_motion_render.py`). It had never handed a model the
frame itself. So the claim lands here as a testable proposition:

> **On this repository's own captures, does a vision-language model that
> receives the RAW IMAGE separate "speaking" from "silent" better than the
> SAME model receiving the SAME INFORMATION as a TEXT/ASCII DESCRIPTION of
> that image?**

### 0.1 What is controlled, and how

Both headline arms are handed **the same 15,360 luma values of the same
frame**. The IMAGE arm gets them as pixels. The TEXT arm gets them as
characters, using the repository's own tone ramp, one character per pixel.

Nothing else differs: same model, same weights, same prompt wording, same
frame indices, same scoring rule, same null, same permutation seed. The
**only** thing that varies between the two headline arms is the
**encoding**. That is what makes the comparison a statement about the
encoding rather than about the model or the frames.

### 0.2 The labels are the repository's own, and they are not what the brief said

| capture | `speak` | `pose` | note | role here |
|---|---|---|---|---|
| `A.nv12` | **True** | facing | facing the camera, speaking a poem | **positive class** |
| `B.nv12` | **True** | turned | head turned away, speaking | secondary condition |
| `C.nv12` | **False** | facing | facing the camera, silent | **negative class** |
| `empty1.nv12` | — | — | no person | **blank control** |

as recorded in `/mnt/c/proart_capture/corpus_v2/{A,B,C}.json` and in
`docs/research/MOUTH-DETECTOR.md` §2. **B is a SPEAKING take**, so B is
*not* a silent class and is not used as one: the negative class is **C**,
which is **pose-matched** with A (both face the camera), and B is reported
as a secondary condition only (§5). This was decided from the captures'
own metadata before any number in this lane was computed.

### 0.3 The fundamental limit, stated up front

The audio and the video of A/B/C are **independent takes**. No lag aligns
them (`docs/research/NV12-STRIDE.md` §6,
`docs/research/MOUTH-APERTURE-ALIGNMENT.md` §4). **Frame-level audio↔video
labels are not recoverable from this corpus** and this lane does not
attempt them. Every number here is a **session-level** statement: the
*distribution* of scores on A's frames against the distribution on C's
frames. The "speaking" label on an A frame means *"this frame comes from a
take in which the person was speaking"*, **not** *"the mouth was moving at
this instant"*.

---

## 1. The pre-registered rule

The rule was written into `experiments/vlm_image_retention.py` **before the
measurement code in that file existed and before any model was run**, and
it is copied verbatim into `experiments/vlm_image_retention_results.json`
under `pre_registered_rules`. `tests/test_vlm_image_retention.py` asserts
that the copy in the results file still equals the copy in the script, so
the rule cannot be quietly rewritten to fit the numbers.

| check | what it asks | passes when |
|---|---|---|
| **R1** | does one representation beat the other? | `D = margin_over_blank(image) − margin_over_blank(text)` is at least **+0.05** with a paired permutation **z ≥ +2.0** for the image arm, or at most **−0.05** with **z ≤ −2.0** for the text arm |
| **R2** | honesty gate | an arm is **usable** iff its own `auc_ac` is **> 0.50** with a permutation **z ≥ +2.0**. The "image beats text" verdict requires the image arm to be usable; "text beats image" requires the text arm to be usable. **An arm that separates nothing cannot win, however its margin compares** |
| **R3** | scramble control | the scrambled arms are reported with the same statistics. An arm whose margin survives the scramble is reading global statistics of the frame, not its spatial arrangement. **Reported, not decisive** |
| **R4** | session-level gate | always reported, never a pass or a fail: the labels are session-level, so a predictor that identifies the **session** rather than the speaking state scores the same as one that identifies the state |

### 1.1 The statistic, and why it is a MARGIN OVER A CONTROL

```
sep(arm)       = AUC(arm; A's frames vs C's frames)
blank_sep(arm) = AUC(arm; the EMPTY ROOM's frames vs C's frames)
margin_over_blank(arm) = sep(arm) − blank_sep(arm)

D = margin_over_blank(image) − margin_over_blank(text)
```

The blank control's separation is **oriented so that the failure mode is a
high value**: an arm that ranks the empty room as *more* speaking than a
silent person — a model that says "speaking" for the blank room — scores
high on `blank_sep`, and the margin subtracts that credit. An arm that says
the same thing about everything scores **0.50 on both terms**, so its
margin is **0** and it cannot win on the margin alone.

**Why the statistic is threshold-free.** A first version of this rule used
a thresholded margin (`TPR_A − FPR_C` at the empty room's 95th percentile).
It was replaced, before the full run and before any result was seen,
because a thresholded margin **saturates**: when the speaking take and the
silent take both sit on the same side of any blank-calibrated threshold it
reports zero even when the scores separate them cleanly. The thresholded
margin is still **computed and reported** as a secondary statistic (§4.3),
at the operating point the empty room imposes, because that is what a
deployable system would see.

### 1.2 The null

**2,000 permutations** of the A/C session labels over the pooled A and C
frames, seed **20261020**. The **empty room is not permuted**: it is the
control, and the control is what the margin is measured against. The
**same permutation is applied to both arms**, so `D` has its own null
rather than borrowing two separate ones.

### 1.3 The arms, all fixed before the run

| arm | role | what it is |
|---|---|---|
| `image` | **headline** | the frame's 15,360 luma values as pixels: replicated to RGB, upscaled **×4 NEAREST** to 640×384, written as a greyscale PNG whose bytes are hashed |
| `text` | **headline** | the **same** 15,360 values as characters: the repository's own tone ramp `" .:-=+*#%@"`, **96 rows of 160 characters, one character per pixel**, with the renderer's per-frame 1st/99th-percentile contrast stretch |
| `text_global` | robustness variant | the same ramp and grid with a **fixed 0–255** mapping, so the absolute luma level survives into the description |
| `image_scrambled` | control | the raw luma with its **4×4 blocks permuted** (fixed seed): identical pixel histogram, destroyed spatial arrangement |
| `text_scrambled` | control | the **same** block permutation applied to the same 96×160 character grid, so the two scrambled arms are matched to each other exactly |

---

## 2. The instrument

### 2.1 The frames

All four captures are read through `src/nv12.py`, which **detects** the
layout from the content and reports the margin over the runner-up:

| capture | layout | stride | frames | mean consecutive \|diff\| | margin over runner-up |
|---|---|---:|---:|---:|---:|
| A | `y_uv_halfres` | 19,200 | 900 | 0.544891 | **79.9×** |
| B | `y_uv_halfres` | 19,200 | 900 | 0.433 | **87.1×** |
| C | `y_uv_halfres` | 19,200 | 900 | 0.367 | **117.1×** |
| empty1 | `y_uv_halfres` | 19,200 | 4,500 | 0.005 | **7,446×** |

Nothing in this lane reads a torn frame. `A.nv12`'s SHA-256
(`b65cf944aa4e8a65932d4c140335a495fa6a1707f28025d40fb08d31fa33bc73`)
is the same one recorded in `corpus_v2/A.json`, so this is the file the
corpus was seeded from.

**Frame selection**: **120 frames** per main condition (A, C, empty1) and
**60** for the secondary condition B, chosen as
`round(i · n_frames_on_disk / n_scored)`, in order — for A and C, indices
0, 8, 15, 22, 30, 38, … 892; for empty1, 0, 38, 75, 112, … . **No selected
pair is byte-identical in any condition** (0 of 119, 0 of 119, 0 of 119,
0 of 59), so the driver-repeated frames do not inflate the sample.

### 2.2 The model and the scoring

| | |
|---|---|
| model | `Qwen/Qwen2.5-VL-3B-Instruct`, revision `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| weights | SHA-256 `0891a2d0e1a6f8132afe046017eef6b86e3e8f803b98d1a3e726722a077bb804`, 7,509,337,976 B — **the same bytes** `docs/research/LOCAL-VLM-DETERMINISM.md` §7 measured |
| device | float16, `device_map="auto"`, NVIDIA RTX 4050 Laptop GPU (6,140 MiB); **peak VRAM 4.58 GB**; 21 of 41 decoder modules offloaded to the CPU |
| decoding | **none** — one forward pass per frame, batch size 1, no generation, no sampling, so there is no generation seed to set |

The score is the **logit of the first token of `"SPEAKING"` minus the logit
of the first token of `"NOT SPEAKING"`** at the final prompt position
(token ids **50** and **14065**). One token each, so there is **no length
bias** between the two options, and the score does not depend on the
model producing JSON or any particular formatting. The prompt is the same
in every arm except for its content:

```
Here is a representation of one camera frame from a desk-mounted camera in a workshop.

<CONTENT>

Question: Is the person in this frame SPEAKING (their mouth is moving or open, as if
talking) or NOT SPEAKING (mouth closed, still)? Answer with exactly one word: SPEAKING
or NOT SPEAKING.
```

The model's **greedy** first token was one of the two scored options on
**every one of the 1,920 scored frames** (0 exceptions in any arm and any
condition).

**The harness is deterministic.** Eight frames per headline arm were scored
twice: **max |Δscore| = 0.0000**, bit-identical, in both arms. This is the
same property `docs/research/LOCAL-VLM-DETERMINISM.md` measured for local
greedy decoding (spread 0.000 against a hosted spread of 0.619).

**Cost**: 1,920 forward passes in **4,239 s** of scoring (71 min), plus 16
repeat scorings for the determinism control. The image arm ran at
**0.90 s/frame** (409 prompt tokens); the text arm at **2.00 s/frame**
(4,052 tokens on A's frames); the scrambled text arm at **16.7 s/frame**,
because scrambling destroys the runs of repeated characters that the
tokenizer was compressing (5,790 tokens).

---

## 3. The results

### 3.1 The separation: speaking take vs silent take

AUC of the score separating A's frames from C's frames, against the
permuted-label null of 0.50 (2,000 permutations):

| arm | AUC (A vs C) | perm z | Cohen's *d* | usable (R2)? |
|---|---:|---:|---:|:--:|
| **`image`** | **0.7784** | **+7.43** | +1.1103 | yes |
| **`text`** | **0.9562** | **+12.26** | +2.5925 | yes |
| `text_global` | 0.5265 | +0.72 | +0.0992 | no |
| `image_scrambled` | 0.4062 | −2.48 | −0.3785 | no |
| `text_scrambled` | 0.6438 | +3.88 | +0.5471 | yes |

**The text description separated the speaking take from the silent take
BETTER than the raw image did: 0.9562 against 0.7784, a difference of
+0.1778 in the text arm's favour.** It is also higher than **every** AUC
`docs/research/MOUTH-DETECTOR.md` records for this corpus — its best is
the mouth box's `motion` feature at 0.9158 — though that is a comparison
across different samples (§6). The image arm's separation is real
(z +7.43) but it is the weaker of the two.

The score distributions behind those AUCs:

| arm | A (speaking) | C (silent) | empty1 (blank) | B (speaking, turned) |
|---|---:|---:|---:|---:|
| `image` | **+1.4177** ± 0.4338 | +1.0401 ± 0.2076 | **+0.3488** ± 0.1530 | +1.2156 ± 0.3717 |
| `text` | **−0.3202** ± 0.2212 | −0.7750 ± 0.1123 | **−0.1454** ± 0.2325 | −0.5875 ± 0.2717 |
| `text_global` | +0.0885 ± 0.2017 | +0.0734 ± 0.0754 | **+0.3443** ± 0.0214 | — |

The image arm orders the three conditions **A > C > empty1**, which is the
order the world requires. The text arm orders them **empty1 > A > C**: the
**empty room outranks the speaking person**.

### 3.2 The blank-room control

`blank_sep` = AUC(empty room's frames vs C's frames), oriented so that a
**high** value means the arm ranks the empty room as *more* speaking than a
silent person:

| arm | blank_sep | what it means |
|---|---:|---|
| **`image`** | **0.0034** | the empty room is ranked below the silent person, essentially always — **the control passes** |
| **`text`** | **0.9970** | the empty room is ranked above the silent person, essentially always — **the control fails** |
| `text_global` | 0.9998 | fails, harder |
| `image_scrambled` | 0.5440 | the scramble destroys the ordering too |
| `text_scrambled` | 0.7316 | still fails the control |

**This is the sharpest number in the document.** The TEXT arm does not
merely fail to beat the blank room — it puts the **empty room** at the
**top** of the "speaking" scale, above both the silent person and the
speaking person. The IMAGE arm puts the empty room at the bottom.

A plausible mechanism, offered as a mechanism and not as a result: the
empty room's character grid is nearly empty (697 prompt tokens, against
4,183 for the silent person's), and a mostly-blank page of spaces reads to
this model as *less* like a scene, and therefore as *more* like "nobody is
speaking" in a way that raises the "speaking" logit rather than lowering
it. Whatever the mechanism, the consequence for the lane is the same: the
text arm's A-vs-C separation cannot be read as evidence that it detected
speech, because the same arm ranks a room with nobody in it as more
speaking than a man who is standing still.

### 3.3 The margin over the control, and the headline

```
margin_over_blank(arm) = sep(arm) − blank_sep(arm)
```

| arm | sep (A vs C) | blank_sep | **margin over blank** |
|---|---:|---:|---:|
| **`image`** | 0.7784 | 0.0034 | **+0.7750** |
| **`text`** | 0.9562 | 0.9970 | **−0.0408** |

| headline | value |
|---|---:|
| `margin_over_blank(image)` | **+0.7750** |
| `margin_over_blank(text)` | **−0.0408** |
| **`D = image − text`** | **+0.8159** |
| pre-registered margin threshold | ±0.05 |
| **paired permutation z of D** | **−0.41** |
| permutation null | mean **+0.8309**, sd **0.0370**, 2,000 permutations |

**D clears the margin threshold and fails the permutation gate.** The
observed D is *slightly below* the null's mean.

### 3.4 Why the permutation gate failed — and what it does and does not say

The null permutes the A/C labels over the pooled A and C frames and leaves
the **empty room alone**, because the empty room is the control. That is
correct for what the null is supposed to hold fixed, and it has a
consequence worth stating precisely:

> **The blank-control term of `D` is almost exactly invariant under the
> permutation**, because the two arms rank the empty room relative to
> *every* person frame the same way they rank it relative to C's frames
> (the image arm below, the text arm above). So the null's mean sits on
> top of the observed D, and the z compares two numbers that share their
> dominant term.

Decomposed, the observed D is

```
D_obs = (0.7784 − 0.0034) − (0.9562 − 0.9970) = +0.8159
       =    +0.7750         −        −0.0408
```

and the null's mean is +0.8309 — the same construction with the A-vs-C
separations replaced by their permutation means (≈ 0.50 each). The
observed D is **0.015 below** it, because the TEXT arm's A-vs-C separation
is 0.178 *better* than the IMAGE arm's, and the headline subtracts the
separations.

So the pre-registered statistic, on this data, is **dominated by the
blank-control term**, and its permutation z is therefore a test of the
**A-vs-C separation difference**, not of the difference in how the two arms
treat the empty room. The separation difference goes the **wrong way for
the image arm** (−0.178), which is why the z is negative.

**What this licenses and what it does not:**

* It **does not** license "retaining the image beats describing it".
  The pre-registered rule did not produce that verdict and this document
  does not claim it.
* It **does not** license "describing the image beats retaining it",
  either: the text arm's separation advantage is bought with a blank
  control that fails, and the pre-registered rule's gate on `D` is not
  met in that direction either.
* It **does** license the two component facts, each with its own
  permutation z: the TEXT arm separates A from C at AUC 0.9562
  (z +12.26) and the IMAGE arm at 0.7784 (z +7.43); and the TEXT arm
  ranks the empty room above the silent person (AUC 0.9970) while the
  IMAGE arm ranks it below (AUC 0.0034).

**If the headline had to be restated as a single sentence:** on this
corpus, **the description read the mouth better than the picture did, and
the picture was the only one of the two that still knew the difference
between a room and a man.**

### 3.5 A note on what the log-probability buys

At the model's own decision boundary (score > 0), the arms' behaviour is
exactly what `docs/research/LOCAL-VLM-DETERMINISM.md` §3.3 measured with
*generation*:

| arm | fraction scored "speaking" on A | on C (silent) | on empty1 |
|---|---:|---:|---:|
| `image` | **1.000** | **1.000** | **0.967** |
| `text` | 0.017 | 0.000 | 0.192 |

The image arm says "speaking" to the empty room 96.7% of the time — the
same degeneracy that lane recorded for the local 2B and 3B models on
**blank and silent frames alike**. **The continuous score is what carries
the signal**: the same forward passes that produce a degenerate
"always SPEAKING" answer separate A from C at AUC 0.7784. **Scoring by
forced-choice log-probability rather than by generation is the only reason
this lane produced a separation at all**, and that is a transferable
finding in its own right.

---

## 4. The controls

### 4.1 The scramble control (R3)

The scramble permutes the 4×4 blocks of the frame — **identical pixel
histogram, destroyed spatial arrangement** — with a fixed seed (20261020),
and the **same permutation** is applied to both scrambled arms so they are
matched to each other.

| arm | AUC (A vs C) | change from its own unscrambled arm | blank_sep | margin |
|---|---:|---:|---:|---:|
| `image` → `image_scrambled` | 0.7784 → **0.4062** | **−0.3722** (and it flips below chance, z −2.48) | 0.0034 → 0.5440 | +0.7750 → −0.1378 |
| `text` → `text_scrambled` | 0.9562 → **0.6438** | **−0.3124** (z +3.88) | 0.9970 → 0.7316 | −0.0408 → −0.0878 |

**Both arms' separations are spatial.** Scrambling the blocks removes
0.37 of AUC from the image arm and 0.31 from the text arm. Neither arm is
reading a global statistic such as mean brightness, and neither arm's
separation is an artefact of the frame's overall level. The image arm's
separation is the *more* fragile of the two under the scramble, and it
inverts rather than merely decays.

### 4.2 The robustness variant on the description

`text_global` — the same ramp and grid with a **fixed 0–255** mapping
instead of the renderer's per-frame contrast stretch — has AUC (A vs C) of
**0.5265** (z +0.72): **the text arm's 0.9562 depends on the repository's
per-frame stretch**. With the absolute level preserved, the character grid
of a person and of an empty room differ mostly in *how much of the ramp
they use*, and the separation collapses. `text_global` also fails the
blank control harder (0.9998) and scores the empty room above both people
(+0.3443 against +0.0885 and +0.0734).

**So the TEXT arm's strong separation is a property of one particular
renderer**, and it is not a property of "text descriptions" in general.

### 4.3 The leak-free thresholded margin

At the operating point the empty room imposes (τ = the 95th percentile of
that arm's blank-room scores):

| arm | τ | TPR on A | FPR on C | FPR on empty1 | margin |
|---|---:|---:|---:|---:|---:|
| `image` | +0.579 | 0.975 | 0.967 | 0.050 | **+0.0083** |
| `text` | +0.333 | 0.000 | 0.000 | 0.050 | **0.0000** |

**Both arms are worth nothing at a leak-free operating point.** The image
arm's separation of 0.7784 needs an oracle threshold: at the threshold the
empty room sets, it calls 96.7% of the silent person's frames "speaking".
Its raw AUC is real, but its usable margin is **0.008**. This is the
reason the thresholded margin was demoted to a secondary statistic — it
saturates — and it is also a finding in its own right: **on this corpus,
a 3B vision-language model's separation of speaking from silent does not
survive its own blank-room operating point.**

---

## 5. The secondary condition B

B is the **speaking** take with the head turned away. It is not a silent
class and is not used as one; it is reported because it separates
*facing* from *speaking*.

| arm | mean score | AUC (A vs B) | AUC (B vs C) |
|---|---:|---:|---:|
| `image` | +1.2156 | 0.6467 | **0.6726** |
| `text` | −0.5875 | 0.7685 | **0.7075** |

**Both arms rank the turned-away SPEAKING person above the facing SILENT
person** (AUC 0.6726 and 0.7075 for B against C). Neither arm is simply
detecting "a face looking at the camera": a person who has turned away and
stopped being visible as a face is still scored as more speaking than a
person standing still in full view. That is a genuine positive result,
and it is the same direction in both arms.

---

## 6. Against the repository's own measurements

`docs/research/MOUTH-DETECTOR.md` is the directly comparable measurement,
and its numbers are the ones this lane is measured against — not fresh
ones. Its AUCs are computed on the **distinct** frames (596 of A's 900)
with a moving-average window; this lane's are computed on 120 evenly
spaced grid frames. **The two are different samples, so the comparison is
indicative, not exact.**

| instrument | AUC (A vs C) | conditional AUC | perm z |
|---|---:|---:|---:|
| **TEXT arm (this lane)** | **0.9562** | — | **+12.26** |
| mouth box, `motion` (`MOUTH-DETECTOR.md` §3.1) | 0.9158 | 0.8499 | +16.55 |
| mouth box, `lumafluct` (the deployed feature) | 0.8000 | 0.7086 | +9.60 |
| head-motion proxy `H_upper` (the confound) | 0.8342 | **0.5340** | — |
| background box (rows 46–58, cols 130–157), `motion` | 0.7446 | 0.7037 | +9.49 |
| **IMAGE arm (this lane)** | **0.7784** | — | **+7.43** |

Three things follow, and only three:

1. **The TEXT arm's separation is higher than every AUC
   `docs/research/MOUTH-DETECTOR.md` records for this corpus** — above the
   mouth box's own `motion` feature (0.9158) and well above the
   head-motion proxy (0.8342). Across different samples, so indicative
   rather than exact.
2. **The IMAGE arm's separation is below the mouth box's `motion`
   feature** (0.7784 against 0.9158) and **below the head-motion proxy**
   (0.8342). **The repository's own cheap pixel statistic — the mean
   absolute consecutive-frame difference in a 13×25 box — separates the
   speaking take from the silent take better than a 3B vision-language
   model looking at the picture does.** That is a strong argument that the
   raw-luma route was not the bottleneck this repository was working on.
3. **The IMAGE arm does not clear the head-motion confound.** Its
   separation (0.7784) is *below* `H_upper`'s (0.8342), which is the
   number `docs/research/MOUTH-DETECTOR.md` §2 identifies as the
   instrument that separates the speaking take from the silent one by
   moving more, not by speaking. **This lane did not condition the VLM
   arms on the head-motion proxy** — that control was not in the
   pre-registered design — so **neither arm's A-vs-C separation can be
   called speech-specific.** The text arm's higher separation could still
   be head motion carried by the character grid's texture.

---

## 7. The verdict

> **`no_difference_at_the_preregistered_threshold`**

| check | result |
|---|---|
| **R1** the image arm beats the text arm | **no** — D = **+0.8159** clears the ±0.05 margin threshold, but the paired permutation z is **−0.41**, against the required **+2.0** |
| **R1** the text arm beats the image arm | **no** — D is nowhere near −0.05, and the z is not ≤ −2.0 |
| **R2** the image arm is usable | **yes** — AUC 0.7784, z +7.43 |
| **R2** the text arm is usable | **yes** — AUC 0.9562, z +12.26 |
| **R3** the scramble control | reported — both arms' separations are spatial; the image arm's loses 0.372 of AUC and inverts, the text arm's loses 0.312 |
| **R4** the session-level gate | reported — every number above is a distribution-level statement; no frame-level audio↔video claim is made or supported |

**Which way did it come out?** **Neither way, on the pre-registered
headline.** The two component facts point in opposite directions — the
text arm separates better, the image arm passes the blank control — and
the pre-registered statistic cannot adjudicate between them, because the
permutation it is tested against preserves the very term that dominates it
(§3.4).

**The verdict can come out the other way, and this document says so
plainly.** `derive_verdict()` in the experiment recomputes the verdict from
the recorded numbers, and `tests/test_vlm_image_retention.py` mutates
those numbers four ways and requires four different verdicts — image wins,
text wins, neither arm separates anything, and a big difference with no
permutation support is **not** a win. A verdict function that could not
come out the other way would not be a measurement.

**The falsified thing here is not the paper's claim.** It is the
expectation, natural to this repository, that handing a model the picture
would finally give it something the character rendering could not. On
this corpus, with this model, the **character rendering separated the
speaking take from the silent take better than the picture did** — and the
picture's only advantage was that it still ranked an empty room below a
silent man.

---

## 8. What this does not establish

- **It does not establish a frame-level claim.** The audio and the video
  of A/B/C are independent takes; no lag or time base aligns them
  (`docs/research/NV12-STRIDE.md` §6,
  `docs/research/MOUTH-APERTURE-ALIGNMENT.md` §4). "Speaking" here means
  "this frame comes from a take in which the person was speaking", never
  "the mouth was moving now".
- **It does not establish that either arm detected speech.** Neither arm
  was conditioned on the head-motion proxy, and the image arm's
  separation is *below* that proxy's own AUC. Both arms' separations are
  consistent with head motion.
- **It does not establish that the image arm is worse than the text
  arm**, either. The verdict is a null produced by a pre-registered gate
  that this data does not clear, and §3.4 explains exactly why the gate
  cannot separate the two facts it is being asked to separate.
- **It does not establish the paper's claim.** One 3B model, one corpus,
  one person, one room, one lighting state, one seed, one prompt, one
  description renderer, one split. The paper's claim is about 8B and 32B
  models on agentic tool-use tasks; this is a 3B model asked a binary
  question about a 160×96 frame.
- **It does not establish anything about "descriptions" in general.** The
  TEXT arm is one particular description: the repository's own tone ramp,
  one character per pixel, with a per-frame contrast stretch. A caption, a
  landmark list, or a motion map are different descriptions and are not
  tested here — and §4.2 shows that the choice of renderer inside the
  description moves the answer by 0.43 AUC.
- **It does not establish that the ASCII representation is good.** It
  establishes that *this* rendering, *this* model, *this* corpus give
  AUC 0.9562 — and that the same arm ranks the empty room above the
  silent person.
- **It does not clear the frame's own defects.** The face box straddles
  the silhouette edge (`docs/research/MOUTH-DETECTOR.md` §0); the frames
  are 160×96 and the camera delivers ~19.9 distinct images per second on
  a 30 fps grid; the capture is one person in one workshop.
- **It says nothing about throughput.** The text arm cost 2.00 s/frame
  and 4,052 prompt tokens against the image arm's 0.90 s and 409 tokens;
  the scrambled text arm cost 16.7 s/frame. Those are costs of *this*
  description at *full* resolution, and they are not a general statement
  about text-versus-image economics.
- **It does not license any use of the "always SPEAKING" behaviour.** At
  the model's own decision boundary the image arm calls the empty room
  "speaking" 96.7% of the time. Nothing downstream should be built on the
  model's word; only on its score.

---

## 9. Reproducing it

```bash
/home/proart/projects/gpu-venv/bin/python experiments/vlm_image_retention.py \
    --out experiments/vlm_image_retention_results.json
/home/proart/projects/glasses/.venv/bin/python -m pytest \
    tests/test_vlm_image_retention.py -q
```

The experiment needs the GPU venv (torch, transformers, Pillow); it takes
about 71 minutes of scoring on one RTX 4050 Laptop GPU and peaks at
4.58 GB of VRAM. **The test does not**: it runs on the repository venv,
downloads nothing, and needs no model file — torch and transformers are
imported inside the experiment's scoring functions, and one test asserts
on the AST that importing the experiment does not import torch at all.

### Provenance

| | |
|---|---|
| script | `experiments/vlm_image_retention.py`, SHA-256 `86b3e1feeb5f8388ab5ceb2ac33d601aae6ab7fb2bb8921eadcc39244a8877a5` |
| reader | `src/nv12.py` (layout detected, margins recorded) |
| renderer | `experiments/ascii_motion_render.py`, SHA-256 `a07ea4a6f84f8f871f2ff2b02544dc94e91700325b084007488ddf69c793c833` |
| inputs | `A.nv12` `b65cf944aa4e…` (17,280,000 B) · `B.nv12` `c87269568fab…` · `C.nv12` `937ac321bc43…` · `empty1.nv12` `da348ccaf08c…` (86,400,000 B) — SHA-256, in `experiments/vlm_image_retention_results.json` |
| model | `Qwen/Qwen2.5-VL-3B-Instruct` rev `66285546d2b821cf421d4f5eb2576359d3770cd3`, weights SHA-256 `0891a2d0e1a6f8132afe046017eef6b86e3e8f803b98d1a3e726722a077bb804` (7,509,337,976 B) — no download; the model was already cached |
| GPU | NVIDIA RTX 4050 Laptop GPU, 6,140 MiB total, float16, `device_map="auto"`, peak 4.58 GB |
| seeds | permutation 20261020 · scramble permutation 20261020 · **no sampling anywhere** (one deterministic forward pass per frame) |
| frames | 120 per main condition, 60 for B, `round(i · n / n_scored)`; **0 byte-identical selected pairs** in any condition |

The test asserts on the committed results, and **it passes on a null**:
the headline difference is recomputed from the recorded numbers rather
than transcribed, the verdict is recomputed from the recorded numbers
rather than read from a stored boolean, and the four verdict mutations
require the opposite answers. So the suite fails only if the measurement
stops being derivable from what was recorded — not if the answer changes.
