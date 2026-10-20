# MOUTH-DETECTOR — the mouth's signal survives head motion, and is still not a mouth detector

**Date:** 2026-10-20
**Experiment:** `experiments/mouth_detector.py`
**Results:** `experiments/mouth_detector_results.json`
**Tests:** `tests/test_mouth_detector.py`
**Reader:** `src/nv12.py` — the shared reader; the frame layout is **detected** from the content, not assumed.

**numpy only.** No scipy, no GPU, no landmark model. The result reproduces
on the repository's own interpreter (`/home/proart/projects/glasses/.venv/bin/python`).
The MediaPipe lane (`docs/research/MOUTH-APERTURE-ALIGNMENT.md`) is a
different instrument and a different statistic; §6 says how the two relate.

---

## 0. The question, and the two corrections that came with it

`docs/research/NV12-STRIDE.md` §5 measured, on the correctly-read frames, a
mouth effect: the mouth box (rows 46–58, cols 62–86) moves **+0.733 sd**
more when Casey speaks than when he is silent, against **+0.477 sd** for a
cheek control at the same rows, a **1.54×** ratio and a permutation z of
**+10.9**. The empty room carries **0.015%** of the mouth region's motion.

`docs/research/MOUTH-APERTURE-ALIGNMENT.md` then found, with a face
landmark model, that the mouth aperture's correlation with the speech
envelope **does not survive its controls**: mouth *width* scored +0.3038
(higher than the aperture's +0.2052) and head *height* scored +0.3316 with a
sharper peak. The correlation was about the head moving, not the mouth
opening.

So this lane asks the sharp version:

> **Can a mouth-localised detector separate speaking from silent once head
> motion is held constant — and if so, how far above its own controls does
> it sit?**

The answer has two halves, and only one of them is the one the brief hoped
for. The signal **does** survive the head-motion control. It **does not**
beat its spatial control, and at the session level it does not beat a
constant predictor.

### CORRECTION (2026-10-20) — the captures are not 30 *distinct* frames per second

`docs/research/NV12-STRIDE.md` §1 and `docs/research/MOUTH-APERTURE-ALIGNMENT.md`
§1 both state that the captures are **900 frames at 30 fps**, and the
argument for 30 fps (900 frames = 30.00 s against a 29.98 s recording) is
sound — **the time base is right**. What was not noticed is that the frames
are not 900 *distinct* images. Measured on the correctly-read luma:

| capture | consecutive pairs that are byte-identical | distinct frames | distinct rate |
|---|---:|---:|---:|
| A | **304** of 899 | 596 | **19.83 Hz** |
| B | **305** of 899 | 595 | 19.80 Hz |
| C | **305** of 899 | 595 | 19.80 Hz |
| empty1 | 149 of 4,499 | 4,351 | 29.00 Hz |

The writer (`/mnt/c/proart_capture/nv12small.py`) calls `cap.read()` 30
times per second of wall clock, but the camera delivered about **19.9
distinct frames per second** and the driver repeated frames to fill the
grid. The duplicate pattern is irregular (gaps of 2, 3 and 4 frames between
duplicates), so this is a driver behaviour, not a frame-rate round.

**Why it matters here.** Differencing the grid as recorded reads about a
third of every session's frames as "nothing happened", and the head-motion
proxy inherits a third of its mass at exactly zero — which is a property of
the capture, not of the wearer. This lane therefore computes its features
on the **distinct** frames, with the moving-average window expressed in
seconds of real time and converted to distinct frames at each session's own
measured rate (40 frames for A/B/C, 58 for the empty room, which was
captured in a different round at 29 Hz).

**The published effect survives this.** Re-measured with the same statistic
as `docs/research/NV12-STRIDE.md` §5 (§8), the mouth-over-cheek ratio is
**1.534×** on the grid as recorded (published: 1.54×) and **1.788×** on the
distinct frames. The correction makes the published effect *stronger*, not
weaker, so **no number in `docs/research/NV12-STRIDE.md` §5 is withdrawn.**

### CORRECTION (2026-10-20) — the "cheek control" is not on a cheek

`docs/research/NV12-STRIDE.md` §5 calls the box at rows 46–58, cols 92–116
a *cheek control*. The boxes were drawn on a 6× render of `A.nv12` for this
lane, and the cheek box **straddles the face's right-hand silhouette** —
the MediaPipe left eye corner in `docs/research/MOUTH-APERTURE-ALIGNMENT.md`
§2 is at x = 91.98, and the cheek box starts at column 92. It is a
face-*adjacent* box with a high-contrast moving edge through it, not a
cheek.

The measured per-pixel temporal standard deviation shows it:

| box | A | B | C |
|---|---:|---:|---:|
| mouth (46–58, 62–86) | 13.83 | 9.22 | 8.13 |
| "cheek" (46–58, 92–116) | **9.98** | **12.71** | 5.57 |
| upper face (20–44, 56–104) | 18.95 | 10.14 | 12.37 |
| background (46–58, 130–157) | 2.08 | 2.44 | 2.82 |

In **B** the "cheek" box is *more* alive than the mouth box (12.71 against
9.22), which is what a silhouette edge does when the head turns. The
consequence for this lane is stated where it bites (§4): the cheek box is a
valid control to *run*, and it is not a clean "face but not mouth" control.
A pure-background box at the same rows is added as the clean scene control.

---

## 1. The pre-registered rule

The rule was written into `experiments/mouth_detector.py` **before the
measurement code existed**, is copied verbatim into
`experiments/mouth_detector_results.json` under `pre_registered_rule`, and
is asserted by `tests/test_mouth_detector.py` to still be the copy in the
script — so it cannot be quietly rewritten to fit the numbers.

| check | what it asks | passes when |
|---|---|---|
| **R1a** | does the mouth feature still separate the takes with the head-motion proxy held constant? | the AUC computed **within strata of the proxy** is > 0.50 with a permutation z ≥ +2.0 (2,000 permutations of the session labels *within* the strata) |
| **R1b** | does it survive the proxy being regressed out? | the residualised effect size is positive with a permutation z ≥ +3.0 **in every fold**, the regression fitted on the training sessions only |
| **R2** | is it **mouth**-specific? | the mouth's conditional AUC exceeds the cheek's by ≥ 0.02 with a **paired** permutation z ≥ +2.0 |
| **R3** | does it beat the competing predictor? | the macro frame recall margin over the head-motion proxy is ≥ 0.05 **and** the mouth's session accuracy is at least the proxy's in ≥ 2 of 3 folds |
| **R4** | does it beat the nulls? | the macro frame recall exceeds **each** of the permuted-label null, the blank control and the session-fingerprint control by ≥ 0.05 |
| **R5** | the honesty gate | always reported, never a pass or a fail |

The verdict is the first of these that applies:
`mouth_signal_does_not_survive_head_motion` → `…survives_but_is_not_mouth_specific`
→ `mouth_detector_does_not_beat_head_motion` → `mouth_detector_beats_nothing`
→ `mouth_detector_survives_head_motion`, and `UNDETERMINED` if the data are
insufficient.

The verdict is computed by `derive_verdict()` from the recorded numbers:
each check's pass/fail is **recomputed** from its own numbers rather than
read from a stored boolean, so a changed number changes the verdict. Two
tests in `tests/test_mouth_detector.py` mutate a recorded number and
require the verdict to change — the verdict has to be able to come out the
other way.

---

## 2. The data, the features, and the covariate

**Reading.** `src/nv12.py` detects the layout from the content: all four
captures resolve to `y_uv_halfres`, stride **19,200**, with margins over
the runner-up of 79.9× (A), 87.1× (B), 117.1× (C) and 7,446× (empty1).
Nothing in this lane reads a torn frame.

**Boxes** (rows [r0, r1), cols [c0, c1), in the 160×96 frame):

| box | rows | cols | role |
|---|---|---|---|
| mouth | 46–58 | 62–86 | the signal |
| cheek | 46–58 | 92–116 | the brief's spatial control (face-*adjacent* — see §0) |
| fore | 20–44 | 56–104 | the upper face: the covariate's own box, and a third control |
| bg | 46–58 | 130–157 | a pure-scene control: the same rows, far from the wearer |

The mouth box contains the MediaPipe landmark positions recorded in
`docs/research/MOUTH-APERTURE-ALIGNMENT.md` §2 (upper inner lip at
(78.60, 52.52), lower inner lip at (78.66, 53.76)) and does not overlap
the cheek box.

**Six candidate features**, all per-frame, all "how much did this box
change", all computed on the distinct frames:

| feature | what it is | translation-sensitive? |
|---|---|---|
| `motion` | mean absolute consecutive-frame difference in the box | yes |
| `lumafluct` | the box's mean brightness minus its 2 s trend | no |
| `histdiff` | histogram-intersection distance between consecutive frames | nearly (a histogram barely cares about a slide) |
| `hfchange` | high-frequency energy (vertical Laplacian) minus its 2 s trend | yes |
| `vgradfluct` | vertical gradient energy minus its 2 s trend (the jaw moves vertically) | yes |
| `darkfracfluct` | the fraction of the box darker than its own 20th percentile, minus its 2 s trend | no |

**The head-motion covariate** `H_upper` is the mean absolute
consecutive-frame difference over the **upper-face box** — outside the mouth
box, so regressing the mouth feature on it does not regress it on itself.
Three variants are reported: `H_frame` (whole frame), `H_shift` and
`H_mouthshift` (the boxes' estimated integer translations).

The covariate is not a straw man. It separates the speaking take from the
silent one at **AUC 0.8342**, and the speaking take really does move more:

| | `H_upper` mean | `H_frame` mean |
|---|---:|---:|
| A (speaking, facing) | **2.0117** | 0.8219 |
| B (speaking, away) | 0.9636 | 0.6545 |
| C (silent, facing) | 0.9432 | 0.5540 |

A moves its upper face **2.1×** as much as C does. B, the speaking take
that faces away, moves about as much as the silent one — the confound is
specific to A, and the A-vs-C comparison is therefore the one where
conditioning matters most. That is the comparison this lane leans on, and it
is the **pose-matched** one (both face the camera).

---

## 3. The head-motion control: the signal survives it

Three head-motion-controlled versions of the same A-vs-C comparison are
computed, because they can disagree and the disagreement is the point.

**Conditional AUC**: the AUC computed *within* strata of the proxy, so the
proxy is held as constant as the strata allow. Its null is the session
labels permuted *within* the strata (2,000 draws), which keeps the matching
and breaks only the session assignment.

**Residualised effect**: Cohen's *d* after regressing the feature on the
proxy, with the regression fitted on the fold's **training** sessions only,
so the held-out session is never used to fit the covariate.

**Matched pairs**: frames matched one-to-one on the proxy (caliper ladder
0.05/0.10/0.25/0.50 sd, first rung forming ≥ 50 pairs), then the win rate.

### 3.1 The mouth box, every candidate (A vs C, both facing camera)

| feature | plain AUC | **conditional AUC** | null | perm z | residual *d* (worst fold) | z | matched win rate | z |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `motion` | 0.9158 | **0.8499** | 0.4999 | +16.55 | +0.7450 | +12.83 | 0.8669 | +12.49 |
| `hfchange` | 0.8913 | **0.8592** | 0.4997 | +17.37 | +0.9552 | +16.45 | 0.8635 | +12.44 |
| `vgradfluct` | 0.7921 | 0.7349 | 0.4997 | +11.05 | +0.6375 | +10.83 | 0.7406 | +8.03 |
| `lumafluct` | 0.8000 | 0.7086 | 0.4994 | +9.60 | +0.2646 | +4.49 | 0.7338 | +8.04 |
| `histdiff` | 0.8048 | 0.7080 | 0.4998 | +10.00 | +0.4916 | +8.41 | 0.6894 | +6.57 |
| `darkfracfluct` | 0.7269 | 0.6459 | 0.4997 | +6.92 | +0.1437 | +2.42 | 0.6314 | +4.57 |

**Every candidate survives.** The conditional AUC is between **0.646** and
**0.859** against a null of **0.499**, with permutation z between **+6.9**
and **+17.4**. The residualised effect is positive in every fold, from
**+0.14 sd** to **+0.96 sd**, and the head-motion-matched win rate runs
from **0.63** to **0.87**.

**The control is calibrated, and that is checkable.** Conditioning on the
proxy removes the proxy's own separation almost completely — `H_upper`'s AUC
falls from **0.8342** to **0.5340** — so the conditioning is doing what it
claims. The matched pairs are near-exact: **293 pairs**, mean absolute
proxy gap **0.0033** against a proxy whose standard deviation is ≈ 1.4. The
proxy's own matched win rate is **0.4096** rather than exactly 0.500,
because the greedy one-to-one matching leaves a hair of systematic
difference in the proxy (≈ 0.003 in proxy units, which moves the mouth
feature by ≈ 0.001 — three orders of magnitude below the effect being
measured).

**The headline feature.** The pre-registered rule selects the feature
inside each fold by training performance. Fold A and fold B select
`motion`; fold C selects `lumafluct`. The headline checks below use fold
C's selection, because fold C is the hardest fold. The table above shows
that the conclusion does not depend on that choice for R1.

### 3.2 R1b, per fold

| fold | regression fitted on | slope on the proxy | residual *d* | perm z |
|---|---|---:|---:|---:|
| A (test) | B, C | +0.2750 | **+0.6146** | +10.75 |
| B (test) | A, C | +0.3669 | **+0.2646** | +4.49 |
| C (test) | A, B | +0.3493 | **+0.5459** | +9.56 |

All three positive with z ≥ +3.0, so **R1b passes**. A pooled-fit
diagnostic (the regression fitted on the A and C frames themselves, which
does not extrapolate to an unseen session) gives +0.5292 sd for
`lumafluct` and +0.6752 sd for `motion` — the same sign and order of
magnitude, so the fold-wise fits are stable and not an artefact of the
extrapolation.

---

## 4. The spatial control: the signal does not clear it

The mouth's conditional AUC against the two other boxes, with a **paired**
null (the same label permutation applied to both boxes, so the difference
has its own null):

| feature | mouth | **cheek** | **background** | mouth − cheek | paired z | mouth − bg | paired z |
|---|---:|---:|---:|---:|---:|---:|---:|
| `motion` | 0.8499 | 0.8233 | 0.7037 | +0.0266 | +1.24 | **+0.1462** | +5.50 |
| `hfchange` | 0.8592 | 0.7699 | 0.6336 | **+0.0893** | +3.68 | **+0.2256** | +7.67 |
| `histdiff` | 0.7080 | 0.6001 | 0.5264 | **+0.1079** | +3.99 | **+0.1816** | +6.17 |
| `vgradfluct` | 0.7349 | 0.6452 | 0.5898 | **+0.0897** | +3.59 | **+0.1450** | +4.99 |
| `lumafluct` | 0.7086 | 0.7443 | 0.6860 | −0.0356 | −1.70 | +0.0227 | +1.10 |
| `darkfracfluct` | 0.6459 | 0.6891 | 0.3714 | −0.0432 | −1.61 | **+0.2745** | +9.90 |

**R2 fails for the deployed feature.** For `lumafluct` — the feature fold C
selects — the mouth's conditional AUC (**0.7086**) is *below* the "cheek"
box's (**0.7443**). The difference is **−0.0356** with a paired
permutation z of **−1.70**: not a significant deficit, but not the
required ≥ 0.02 margin either. R2's rule (margin ≥ 0.02 **and** paired z ≥
+2.0) is not met.

Two things must be said about this row, and neither rescues it:

* **The control is not clean.** The cheek box straddles the face's
  silhouette (§0), so a head-driven moving edge runs through it. A box that
  moves with the head is a poor control for a mouth-specific signal, and
  that is a reason to distrust the comparison, not a reason to dismiss it.
* **The mouth's point estimate beats the *background* for every candidate** —
  by +0.023 to +0.275 conditional AUC, with paired z from +1.10 to +9.90
  (significant for five of the six). The signal is therefore not a
  scene-wide difference between the two takes: a region of the same wall,
  at the same rows, with nothing of the wearer in it, separates far less.

**What this means.** The raw-luma mouth signal is real and is not reducible
to head motion. But the comparison that would make it a *mouth* detector —
does the mouth beat a nearby control that cannot open? — is not won by the
feature the fold-C selection actually deploys, and it is won by only four of
the six candidates, and by a margin that is small even then (+0.027 for
`motion`, whose paired z of +1.24 does not reach the pre-registered +2.0).

---

## 5. The leave-one-session-out evaluation, and why it is not the evidence

The classifier is deliberately plain: one feature, one threshold ("speaking"
iff the feature is above it), the threshold fitted to maximise balanced
accuracy on the **training** sessions, the session decided by the majority
of the held-out session's frames. The feature is selected inside each fold
by training performance; the test session is used for neither choice.

| fold | feature | threshold | held-out frame accuracy | session decision | truth |
|---|---|---:|---:|---:|---:|
| test A | `motion` | 0.79792 | 0.9983 | speaking | speaking ✓ |
| test B | `motion` | 0.89231 | 0.9983 | speaking | speaking ✓ |
| test C | `lumafluct` | 0.00000 | **0.0000** | speaking | silent ✗ |

**macro frame recall 0.6655**, session accuracy 0.6667.

Fold C deserves its own paragraph, because it is the fold that decides
R4. Its training set (A and B, both speaking) contains **no silent
person**, so the pre-registered rule supplies the empty room's first 50 s
as the negative class. The empty room is a *much* easier negative than a
silent person — its mouth motion is **10,748×** quieter than A's — so the
fitted threshold lands at zero and the detector fires on everything,
including the silent wearer. **The empty room is not a substitute for a
silent person**, and a detector trained against it does not discriminate
one. That is the result, not a bug in the fold.

### 5.1 The controls, side by side

| control | what it is | macro frame recall | margin over the mouth detector |
|---|---|---:|---:|
| **mouth detector** | the fold-selected mouth feature | **0.6655** | — |
| **blank (constant "speaking")** | a predictor that always says "speaking" | **0.6667** | **−0.0011** |
| **blank (training majority)** | the training set's majority class | 0.6667 | −0.0011 |
| **permuted session labels** | both alternative label assignments | **0.6661** | **−0.0006** |
| **session fingerprint** | the nearest session by the frame's mean luma | 0.3333 | +0.3322 |
| **frame-label shuffle** | training labels permuted across the pooled training frames, 200 draws | mean 0.7356, p95 0.9899 | — |
| **wrong-session pairing** | A's video labelled silent, C's labelled speaking | **0.6661** | −0.0006 |
| head-motion proxy `H_upper` | the competing predictor | 0.5098 | +0.1557 |
| head-motion proxy `H_frame` | whole-frame motion (contains the mouth box, so not independent) | 0.7546 | −0.0891 |

**The blank scores 1.000 on the two speaking folds and 0.000 on the silent
one**, and the mouth detector scores *exactly the same thing*. **R4 fails.**

**The wrong-session pairing is the sharpest number in this document.**
Swapping the labels of A and C changes the macro frame recall by
**−0.0006**. The metric cannot tell the right labelling from the wrong one.
That is the degeneracy `docs/research/TEACHER-LOO-FOLDS.md` §1 warns about,
measured here on this corpus: the labels are session-level, so a predictor
that identifies the **session** rather than the speaking state scores the
same as one that identifies the **state**.

**A detector that separates sessions is not a speaking detector.** Every
accuracy in this section has to be read with that sentence attached.

### 5.2 The empty room as a physical control

| feature | A's mouth motion | empty room | A / empty | empty room's share of A |
|---|---:|---:|---:|---:|
| `motion` | 1.9306 | 0.000180 | **10,748×** | 0.0093% |
| `lumafluct` | 1.5316 | 0.000192 | 7,975× | 0.0125% |
| `vgradfluct` | 0.6206 | 0.000385 | 1,610× | 0.0621% |
| `hfchange` | 188.997 | 0.001275 | 148,198× | 0.0007% |

The empty room's mouth-box motion is **0.0093%** of the speaking take's —
the published 0.015% (`docs/research/NV12-STRIDE.md` §5) reproduces in
order of magnitude on the distinct frames. The room is not the camera.

The **false-alarm rate** of each fold's detector on the empty room's
**held-out** slice (its last 100 s, which no detector was trained on):

| fold | mouth | cheek | upper face | background |
|---|---:|---:|---:|---:|
| A (`motion`) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B (`motion`) | 0.0000 | 0.0000 | 0.0000 | 0.0052 |
| C (`lumafluct`) | **0.3678** | 0.0000 | 0.5126 | 0.0000 |

Fold C's detector fires on 37% of an empty room's frames — the same
degeneracy as §5, seen from the other side.

---

## 6. The verdict

> **`mouth_signal_survives_but_is_not_mouth_specific`**

| check | result |
|---|---|
| **R1a** the mouth signal survives the head-motion control | **yes** — conditional AUC 0.7086 against a null of 0.4994, permutation z **+9.60** (and 0.8499 / z +16.55 for the `motion` feature that folds A and B select) |
| **R1b** it survives the proxy being regressed out | **yes** — residual *d* +0.6146 / +0.2646 / +0.5459 sd across the three folds, z +10.75 / +4.49 / +9.56 |
| **R2** it is mouth-specific | **NO** — the mouth's conditional AUC 0.7086 does not exceed the "cheek" box's 0.7443 (difference −0.0356, paired z −1.70) |
| **R3** it beats the head-motion proxy | **yes** — macro frame recall 0.6655 against 0.5098 (margin +0.1557), and it is at least as good on all three folds |
| **R4** it beats the nulls | **NO** — the blank control scores 0.6667 and the permuted-label null 0.6661, against the detector's 0.6655 |
| **R5** the honesty gate | the labels are session-level and the frame-level audio↔video labels are **not recoverable**; every number here is a session-level statement |

R2 and R4 are the failures, and the rule's precedence reports R2 first.

### Where each control is reported

| control | where | its number |
|---|---|---|
| cheek box (spatial) | §4 | mouth 0.7086 vs cheek 0.7443, paired z −1.70 |
| empty room (physical) | §5.2 | mouth motion 10,748× quieter; false-alarm rate 0.0000 / 0.0000 / 0.3678 |
| permuted session labels (null) | §5.1 | 0.6661 (margin −0.0006) |
| frame-label shuffle (null) | §5.1 | mean 0.7356, p95 0.9899 |
| wrong-session pairing (negative) | §5.1 | 0.6661 — swapping A's and C's labels moves the score by −0.0006 |
| head-motion proxy (competing predictor) | §3, §5.1 | AUC 0.8342, conditional 0.5340; macro frame recall 0.5098 |
| blank / constant predictor | §5.1 | 0.6667 (margin −0.0011) |
| session fingerprint | §5.1 | 0.3333 |

### What the signal actually is

**It is not head motion.** The MediaPipe lane's caveat — that the mouth
aperture's speech correlation was about the head moving — **does not carry
over to these raw-luma features**, and the reason is that this lane's
statistic is different: it asks whether the mouth region separates the two
*takes* at matched head motion, not whether the mouth's geometry correlates
with the audio envelope frame by frame. On that question the answer is
unambiguous: the conditional AUC is 0.65–0.86 against a null of 0.50, the
residualised effect is positive in every fold, and the covariate's own
separation collapses from 0.834 to 0.534 once it is conditioned on.

**But it is not yet a mouth.** The face-adjacent control box separates the
two takes almost as well, and the mouth's advantage over it is either
absent (the deployed feature) or small and not permutation-significant
(`motion`, +0.027, z +1.24). What the measurement cannot exclude is a
**take-level** difference: the speaking take and the silent take differ in
more than the mouth — in illumination, in posture, in how the person sits —
and the mouth box is one place where that difference shows up.

The one clean piece of evidence for the mouth specifically is the
**background** box: the same wall, at the same rows, with no wearer in it,
separates far less. The mouth's margin over the background is **+0.0227**
(`lumafluct`, paired z +1.10) to **+0.2745** (`darkfracfluct`, z +9.90), and
for the four candidates that beat the cheek it is **+0.1450** to **+0.2256**.
So the effect is not a scene-wide difference — but a face-adjacent
difference is exactly what a *head* difference looks like too, and the
cheek box straddles the silhouette edge, so this measurement cannot
separate "the mouth moved" from "the face moved".

---

## 7. The fundamental limit, stated plainly

**The audio and the video of A/B/C are independent takes.** No lag in ±20 s
and no video time base in 10–60 Hz aligns them
(`docs/research/NV12-STRIDE.md` §6; `docs/research/MOUTH-APERTURE-ALIGNMENT.md`
§4.3 and §5). **Frame-level audio↔video labels are not recoverable from
this corpus**, and this lane does not attempt them: the script does not
open a `.wav` file at all, and `tests/test_mouth_detector.py` asserts that
it does not.

Everything in this document is therefore a **session-level** statement:
*this take* is a speaking take, *that take* is a silent take. The labels
are session-level, which is precisely what makes the leave-one-session-out
accuracy a partly degenerate metric (§5.1). There is no version of this
measurement that says "the wearer was speaking at 12.4 s".

**The A↔B comparison is content-confounded and is not used to judge the
mouth.** A and B recite the same poem, so their speech envelopes agree at
+0.3537 (`docs/research/MOUTH-APERTURE-ALIGNMENT.md` §4.3) — any signal
that tracks speech *content* aligns with both. The mouth features here are
purely visual and cannot be inflated by that agreement, but the A-vs-B
comparison is still not a speaking-vs-silent comparison (both are speaking;
B simply faces away, so its "mouth box" sees a different scene — the
conditional AUC there is 0.62 for `lumafluct`). Every judgement in §3–§6
rests on **A vs C**, which is pose-matched and content-uncorrelated.

---

## 8. The published effect, re-measured

`docs/research/NV12-STRIDE.md` §5 is re-measured here with the **same
statistic** (the A−C difference divided by the mean of the two within-group
standard deviations), on both readings of the capture:

| | published | grid, 30 fps as recorded | distinct frames |
|---|---:|---:|---:|
| mouth A − C | +0.8044 | **+0.7944** | +1.1991 |
| mouth effect | +0.733 sd | +0.828 sd | +1.359 sd |
| cheek A − C | +0.3267 | **+0.3281** | +0.4952 |
| cheek effect | +0.477 sd | +0.540 sd | +0.760 sd |
| mouth-specific excess | +0.256 sd | **+0.288 sd** | +0.599 sd |
| mouth / cheek ratio | 1.54× | **1.534×** | 1.788× |
| empty room, share of A | 0.015% | 0.0136% | 0.0093% |
| A / empty room | 6,600× | 7,368× | 10,764× |

The published figures reproduce: the **1.54× ratio** is the ratio of the
two *effect sizes* (0.733 / 0.477), not the ratio of the raw differences
(the raw-difference ratio is **2.42×**), and it comes back as **1.534×**.
The published effect size itself corresponds to a slightly different
standard-deviation denominator; with the mean-of-within-group-sds
convention the mouth's effect is +0.828 sd and the cheek's +0.540 sd.
**No number in `docs/research/NV12-STRIDE.md` §5 is withdrawn by this
lane**, and the distinct-frame reading makes the effect *larger*.

---

## 9. What this does and does not establish

**Established:**

1. A mouth-localised raw-luma feature separates the speaking take from the
   silent take **with the head-motion proxy held constant**: conditional
   AUC 0.646–0.859 against a null of 0.499 (z +6.9 to +17.4), residualised
   effect +0.14 to +0.96 sd in every fold, head-motion-matched win rate
   0.63–0.87. **The head-motion confound does not explain it.**
2. The empty room is not the camera: the mouth box's motion there is
   **10,748×** quieter than the speaking take's, and the detectors' false
   alarm rate on its held-out slice is 0.000 for the two folds that have a
   real silent person to train on.
3. The mouth box's point estimate beats a pure-background control at the
   same rows for every candidate feature (paired z +1.10 to +9.90,
   significant for five of the six), so the effect is not a scene-wide
   difference between the two takes.
4. **The leave-one-session-out accuracy cannot support a speaking
   detector**, and this lane measures exactly how badly: the blank control
   scores the same (0.6667 vs 0.6655), and swapping A's and C's labels
   changes the score by −0.0006.

**Not established:**

1. **That this is a mouth detector.** The deployed feature does not beat
   the face-adjacent spatial control (R2 fails), and the control is not
   clean, so the measurement cannot separate "the mouth moved" from "the
   face moved".
2. **Anything at the frame level.** No frame-level speaking claim is
   admissible from these captures, and none is made.
3. **Anything about a wearable in the wild.** Three sessions, one person,
   one seat, one room; two of the three are speaking and one is silent;
   A and B recite the same poem; B faces away.
4. **That the head-motion proxy is the only confound.** It is the one that
   was measured. Illumination, posture and distance are not controlled
   between the takes, and the cheek box's geometry (§0) is a live example
   of a control that is not clean.

**What would move it.** The captures this lane needs are the ones
`docs/research/NV12-STRIDE.md` §8 already asks for: a single clock for the
audio and the video, so that frame-level labels exist at all; and a **silent
but facing, and speaking but turned** take (condition D) so that pose and
speaking are separable in the training set. With those, the same script
would answer the question this corpus cannot.

---

## 10. Reproducing it

```
cd /home/proart/projects/communicator
/home/proart/projects/glasses/.venv/bin/python experiments/mouth_detector.py
/home/proart/projects/glasses/.venv/bin/python -m pytest tests/test_mouth_detector.py -q
```

About 40 s, numpy only. The results file records the script's own SHA-256,
the SHA-256 of every input, the CLI and the Python version, and the
pre-registered rule verbatim; `tests/test_mouth_detector.py` fails if the
results were produced by a different version of the script than the one on
disk.
