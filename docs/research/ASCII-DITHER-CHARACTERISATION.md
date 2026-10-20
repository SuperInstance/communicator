# ASCII dither characterisation: the measured change is scene change, not dither

**Date:** 2026-10-20
**Experiment:** `experiments/ascii_dither_characterisation.py`
**Results:** `experiments/ascii_dither_characterisation_results.json`
**Tests:** `tests/test_ascii_dither_characterisation.py`

> ## CORRECTION (2026-10-20) — the bit/dot pairing in the per-dot table was wrong
>
> `experiments/ascii_motion_threshold.py` read the braille mask byte and
> paired **bit b** with the luma of **dot b**, where dot b is the b-th dot of
> the cell in raster order. Syzygy packs a cell with the standard Unicode
> braille bit layout: the **LEFT** column of the 2×4 block is bits 0,1,2,6
> and the **RIGHT** column is bits 3,4,5,7. The bit→dot mapping is
> therefore the **inverse list**
>
> ```
> bit 0 -> dot 0        bit 4 -> dot 3
> bit 1 -> dot 2        bit 5 -> dot 5
> bit 2 -> dot 4        bit 6 -> dot 6
> bit 3 -> dot 1        bit 7 -> dot 7
> ```
>
> — not the identity. Half the bits of every cell were paired with the wrong
> pixel's luma.
>
> The evidence is **exact, and the artefact computes it**. The renderer's
> ingest copies the Y plane **verbatim**
> (`.scratch/Syzygy/include/syz_yuv.h`: `out->luma[...] = src->y[...]`) and
> the braille packer compares each pixel of the block against
> `braille_thresh` with **no luma re-weighting**, so the mask bit for a dot
> *is* `(Y at that dot > 100)`. That is what makes the check exact rather
> than approximate, and it is why the mapping is the only thing that can be
> wrong.
>
> `experiments/ascii_dither_characterisation.py` measures it, over all
> 460,800 dot-decisions of `small.nv12`:
>
> | bit→dot mapping | mismatches |
> |---|---|
> | the renderer's own header, inverted — `[0,2,4,1,3,5,6,7]` | **`0`** |
> | the identity — `[0,1,2,3,4,5,6,7]`, what the old file used | **`24,894`** |
> | `b//2 + 4*(b%2)` — the formula this note first asserted | **`42,680`** |
>
> The old file's numbers are reproduced here on the old basis, bit for bit,
> and the difference is reported rather than overwritten.
>
> | band \|luma−100\| | published | corrected |
> |---|---|---|
> | 0 (on it) | `19.18%` | **`27.93%`** |
> | 11–20 | `1.76%` | **`0.74%`** |
> | 21–40 | `1.05%` | **`0.33%`** |
> | 41+ | `0.36%` | **`0.09%`** |
> | gradient on→far | `52.6×` | **`323.4×`** |
>
> **The published "signal floor" of 0.36% is overstated by about 4×.** The
> monotone fall survives; the numbers do not.

> ## CORRECTION (2026-10-20) — the formula this note first asserted was wrong
>
> The first version of the block above said the bit→dot mapping was
> `b//2 + 4*(b%2)`. That is the **column-major** order, which is not the
> order the scripts enumerate dots in, and it is wrong for six of the eight
> bits — `42,680` of the 460,800 dot-decisions, against the identity
> mapping's `24,894`.
>
> The mapping is an inverse **list**, and the script now computes it from
> `DOT_TO_BIT` so it cannot drift. The number above is produced by the
> artefact (`pairing_mismatch_counts()`), not asserted in prose.

> ## CORRECTION (2026-10-20) — "below the independence baseline" is not a null
>
> The published verdict was: *"the far-from-threshold floor of 0.36% is BELOW
> the 39.98% independence baseline, so that residual is structured, not
> noise"*.
>
> That inference does not hold, and the reason is specific. The independence
> baseline `2p(1−p)` is the rate at which two **independent random** masks of
> the same set density differ. It is a chance-coincidence rate for
> **uncorrelated content**. The far-from-threshold population is not
> uncorrelated content: it is a set of dots whose luma sits far from the
> threshold, which are therefore **stable** — and stability is what *any*
> static scene produces, including a scene in which nothing at all is moving.
> Sitting below the chance baseline says "these bits are sticky". It does not
> say "these bits carry motion".
>
> The null that applies is a **static scene with a real camera in it**, and
> the empty room is exactly that.

> ## CORRECTION (2026-10-20) — the tone-channel recommendation is withdrawn
>
> The published recommendation was reversed on the strength of `5.41%` per
> cell for the tone against `9.67%` per cell for the mask: *"the tone
> channel, not the mask, is the one to build motion on"*. That was the
> second reversal of the same recommendation in this lane, and both
> reversals were made on a **confounded comparison**.
>
> The two numbers do share a denominator — both are per-cell rates on the
> same frames — but the **inference does not follow**: a *lower* change rate
> is not a *better* channel. The mask is a bare threshold on 8 pixels; the
> tone is a 10-level ramp of the cell luma, whose quantisation step is about
> 28 counts. A lower rate is also a lower **sensitivity**.
>
> The corrected comparison is the **excess over a null scene**, and it
> disagrees with the raw rate: the raw rate prefers the tone, the excess
> prefers the **mask** (§8).

**Status: measured, with four external nulls.** The mechanism claim this
file tests survives in its conclusion and fails in its reasoning; the
recommendation it reversed is withdrawn; and a defect in the measurement
underneath it is corrected.

Capture: `/mnt/c/proart_capture/small.nv12`, 30 frames, 160×96 NV12,
`y_uv_halfres` at 19,200 bytes/frame, live ASUS FHD webcam, stationary
subject speaking. Nulls: `/mnt/c/proart_capture/empty1.nv12` (4,500 frames),
`/mnt/c/proart_capture/A.nv12` (900 frames), a synthetic dither calibration,
and a null scene built from the capture's own content.

---

## 1. The question

Two results in this lane had never been reconciled.

`ascii_motion_nulls.py` reported that change **grows with lag** — 6.98% at
lag 1 rising to 22.55% at lag 20, monotone — and concluded *"the delta
carries motion"*.

`ascii_motion_threshold.py` reported that the flip rate **falls 52.6×** from
the threshold outward, and concluded that *"a large part of the raw delta is
threshold dither"* while *"the far-from-threshold floor … is structured, not
noise"*.

Both can be true at once, and both were true of a statistic nobody had ever
compared with anything from outside the measurement. The question this file
answers is:

> **How much of the measured change is threshold dither, and how much is
> motion? And is the "the floor is structured, not noise" inference actually
> supported?**

The rule was fixed before the decisive runs and is written into the results
file (§9).

## 2. Four nulls, and what each one is for

A claim about dither cannot be checked with a number derived from the same
measurement. Four references were used, and each one answers a different
question.

| null | what it is | what it can kill |
|---|---|---|
| **empty room** | the same camera, the same renderer, 300 frames of `empty1.nv12` with nothing in the room | "does anything change when nothing moves" |
| **null scene** | the capture's own static content (per-pixel temporal median) **plus the capture's own global luma wander**, no per-pixel jitter | "the scene is still and only the brightness moves" |
| **pure static** | the same content, frozen, no wander, no jitter | anything at all |
| **synthetic calibration** | the same content with dither of **known** amplitude injected | "how much of this could a jitter of size *s* produce?" |

The empty room's luma sits at about 30 counts and is essentially never near
the threshold (`99.98%` of its pixels are 41+ counts away), so it cannot
probe threshold dither *directly*. It is a null for the frame-level question
and for the far band specifically — and for the far band that is exactly
what is needed.

## 3. The nulls, measured

```
empty room    mean consecutive |diff|  0.005277
              mask per cell        0.0000%      tone per cell  0.0000%
              far band (11+):  4,592,341 dot-observations, 0 flips
pure static   mask per cell        0.0000%      tone per cell  0.0000%
null scene    mask per cell        2.05%        tone per cell  1.59%
              far band (11+):  0 flips, at every lag
real capture  mask per cell        8.63%        tone per cell  4.38%
```

**The camera contributes no dither at all.** With nothing moving, the mask
changes `0.0000%` of cells and the tone changes `0.0000%` of cells. The
empty room's frames are, for these purposes, identical.

The null scene is not zero, and it should not be: the capture's own mean
luma wanders by 10.8 counts over the 30 frames, and a global shift moves
dots across the threshold. That is a **scene-wide** change, and it is
separated from the **local** change by construction.

## 4. The dither calibration, and why it fails to fit

A static scene with a perturbation of known amplitude was rendered through
the same renderer and measured on the same bands. The perturbation is a
temporally correlated Gaussian field of amplitude σ and lag-1
autocorrelation ρ, on luma and chroma, seeded.

The near-threshold bands were used to **fit**; the far bands were left to be
**predicted**.

```
observed    0-0: 27.93%   1-2: 12.42%   3-5:  3.79%
best fit    0-0: 30.38%   1-2: 10.35%   3-5:  0.06%     (σ=2.0, ρ=0.9)
            max band error 3.73 pp
```

The pre-registered rule required a fit within 0.5 pp on those three bands.
**It fails**, and that failure is the finding rather than an obstacle: no
temporally correlated Gaussian jitter reproduces the observed curve at any
amplitude. The observed curve falls too fast — the real change distribution
has heavier tails than a jitter, which is what scene change looks like.

The far-band prediction does not depend on the fit, and it is the same for
every grid point within 5 pp of the observed near bands:

| band | predicted | observed | 99% upper bound on the prediction |
|---|---|---|---|
| 11–20 | `0.0000%` | `0.74%` | `0.0119%` |
| 21–40 | `0.0000%` | `0.33%` | `0.0072%` |
| 41+ | `0.0000%` | `0.09%` | `0.0057%` |

**Dither is inert in the far-from-threshold band at every amplitude in the
grid.** A dot 41+ counts from the threshold does not flip because a pixel
jittered; it flipped because the scene moved it across the threshold. The
observed far-band rate is an **excess over dither**: on the closest far band
it is 62x the 99% upper bound of the prediction, and on all three bands it
lies beyond that bound.

The same property is asserted as a test, so it cannot rot:
`test_dither_is_inert_in_the_far_from_threshold_band` fails if any plausible
amplitude ever produces a far-band flip.

## 5. The lag curve is the discriminator

Dither is uncorrelated in time; motion is correlated. So the change rate of a
**static scene with uncorrelated jitter** must be the same at every lag, and
the real capture's must not be.

```
lag                        1       20
real capture  mask      8.63%   25.19%     near  5.89% -> 20.30%   far 0.31% -> 5.51%
null scene   mask      2.05%   14.79%     near  2.40% -> 15.74%   far 0.00% -> 0.00%
synthetic (rho=0)  mask      7.71%    7.64%     near  5.32% ->  5.34%   far 0.00% -> 0.00%
```

The uncorrelated synthetic is **flat**, as it must be. The real capture
rises by 2.9x over that range, and by 4.6x out to lag 29. So the rise is not
jitter.

And the null scene says *which part* is not local motion: it reproduces most
of the rise in the **near** band (`15.74%` against the real `20.30%` at lag
20) while producing **no far-band flips at any lag**. The
near band's growth is the scene's global brightness wander, amplified by the
threshold; the far band's growth is local scene change and nothing else.

On the longer capture (`A.nv12`, 300 frames, lags 1…40) the same picture
holds: mask `3.80%` → `19.68%`, near `3.41%` → `19.72%`, far `0.09%` →
`3.67%`. The far band's ratio is 40× over that range.

## 6. The split

| share of the measured change | value | measured against |
|---|---|---|
| **camera dither** | **`0.0%`** | the empty room (`0.0000%`) and the pure static scene (`0.0000%`) |
| **the scene's global brightness change, amplified by the threshold** | **`23.8%`** | the null scene (`2.05%` of `8.63%`) |
| **local scene change** | **`76.2%`** | the remainder |

At the dot level the split is different, and worth stating because it is
where the threshold actually acts: the threshold-adjacent bands (within 10
counts) carry `85.83%` of the dot-level flips, and the far bands carry the
rest. A dot 41+ counts from the threshold that flips must have moved at least 41
counts in one frame to cross it; across the whole far band (11+ counts away)
the measured flips moved a median of `34` counts.

**So the answer to the question asked:** none of the measured change is
threshold dither in the sense the mechanism claim uses, because the camera
produces none; `23.8%` of it is the scene's global brightness wander being
amplified by the threshold; and `76.2%` is local scene change.

## 7. The two files, on one basis

The two scripts were not computing the same statistic.

| | `ascii_motion_nulls.py` | `ascii_motion_threshold.py` | one basis |
|---|---|---|---|
| statistic | cells whose mask byte differs | same | same |
| aggregate | **median** over 29 lag-1 pairs | **mean** over 30 pairs | mean and median over 29 pairs |
| the 30th pair | — | the **wrap-around** pair (last frame vs first frame), 29 frames apart | excluded |
| per-dot pairing | n/a | **bit b paired with dot b** | the inverse list `[0,2,4,1,3,5,6,7]` |

On one basis (29 consecutive pairs, correct pairing):

```
mask per cell   mean 8.63%    median 6.98%
tone per cell   mean 4.38%    median 2.76%
```

The published `9.67%` is reproduced exactly from the published basis (30
pairs including the wrap-around one) and `6.98%` is reproduced exactly from
the median basis. The difference between them is entirely the aggregate and
the spurious pair — not a disagreement about the world.

The per-dot table is a separate matter: its **bit/dot pairing** was wrong,
and it is corrected in the correction block at the top of this file, where
the artefact's own measurement of the three candidate mappings is given.

## 8. The recommendation, re-tested

Same cells, same frames, same 29 pairs, same denominator. The criterion is
the **excess over the null scene** as a share of the channel's own observed
change, because a raw rate compares a channel's noise with its sensitivity.

| channel | observed | null scene | excess | excess share |
|---|---|---|---|---|
| mask | `8.63%` | `2.05%` | `6.58%` | **`76.2%`** |
| tone | `4.38%` | `1.59%` | `2.79%` | `63.6%` |

**The two criteria disagree, and the excess is the right one.** The raw rate
prefers the tone; the excess prefers the **mask**.

The sensitivity profile says the same thing from the other side — the share
of cells each channel reports as changed, by how much that cell's luma
actually moved:

| cell luma moved | mask | tone |
|---|---|---|
| < 0.5 counts | `5.65%` | `2.16%` |
| 0.5–1 | `10.33%` | `5.53%` |
| 1–2 | `15.46%` | `12.00%` |
| 2–5 | `30.15%` | `19.42%` |
| 5–10 | `58.81%` | `30.22%` |
| ≥ 10 | `84.76%` | `60.70%` |

The mask detects the scene's change at every magnitude more often. The tone's
lower raw rate is a lower sensitivity, not a cleaner signal.

**The recommendation is therefore: build motion on the MASK.** The published
reversal in favour of the tone is withdrawn, on the grounds that it was made
on a confounded comparison.

The exposure-drift control is consistent: with each frame re-centred on the
threshold, the mask's rate moves `8.63%` → `8.43%` and the tone's `4.38%` →
`4.11%`, so only `2.4%` of the mask's change and `6.1%` of the tone's is
attributable to the frame mean. The tone is the more exposure-sensitive
channel, as its quantisation step predicts.

## 9. The rule, and how it came out

The rule was fixed before the calibration fit, the empty-room run, the
extended lag sweep and the persistence test; the band table and the bit/dot
pairing defect had already been measured when it was written, and the results
file says so.

| rule | outcome |
|---|---|
| **R1** the calibration must reproduce the observed near bands within 0.5 pp | **FAILS** (best fit `3.73` pp off) — and that failure is reported as the finding: no jitter reproduces the observed curve |
| **R2** the "structured, not noise" claim holds only if the observed far-band rate exceeds the 99% bound of the prediction | **HOLDS**: `0.74%`/`0.33%`/`0.09%` observed against `0.0000%` predicted |
| **R3** the excess is motion only if the empty room does not reproduce it | the empty room's far band is `0.0000%` against the capture's `0.31%` — so the excess is **scene change**, but *not* provably the subject |
| **R4** near flat in lag and far growing = motion | **BOTH GROW**: no temporally uncorrelated component is visible in the real capture; the null scene localises the near band's growth to the global wander |
| **R5** the channel with the larger excess over the null is the one to build on | **the MASK** — so the published "the tone channel, not the mask" is **withdrawn** |
| **R6** the dither fraction is taken from the physical nulls | **`0.0%`** by the empty room; `23.8%` by the null scene |

## 10. What this does not establish

* **That the near-threshold flips are small *motion* rather than small
  *scene* change.** The null scene attributes most of them to the scene's
  global brightness wander; whether that wander is the camera's auto-exposure
  responding to the subject or the subject's own brightness is not settled
  here, and no measurement here separates a sub-pixel scene change from a
  content-dependent camera artefact.
* **That the far-band excess is the subject moving.** The empty room
  establishes only that it is not camera noise. Its flips are localised —
  the top 1% of cells carry `19.29%` of them, a nineteen-fold concentration —
  but that is consistency with an edge moving, not identification of the
  subject.
* **That this representation separates "talking to me" from "talking to
  someone else".** No labelled data is involved. A demonstration that motion
  is visible is not a demonstration that motion is sufficient.

## 11. What follows for the representation

* **Hysteresis is a trade, not a fix.** `ASCII-MOTION.md` §8 recommended
  hysteresis "sized by the measured gradient". The gradient is real
  (`27.93%` on the threshold against `0.09%` at 41+ counts), but the
  threshold-adjacent flips it would suppress are **not noise** — they are
  the scene's own small changes, and they are `85.83%` of the dot-level
  flips. Hysteresis buys a smaller P-frame by throwing away small motion.
  That can be the right trade, but it should be made knowingly, with the
  cost measured.
* **The fix the evidence supports is the one `ASCII-MOTION.md` already
  reached in its own conclusion**: the P-frame should carry the **magnitude**
  of the change, not the per-frame sign of a threshold test. The sign is
  what makes a 1-count wobble and a 40-count move look alike.
* **Keep the mask.** It is the more sensitive channel and it carries the
  larger share of change the nulls cannot explain.

## Reproduce

```bash
/home/proart/projects/glasses/.venv/bin/python experiments/ascii_dither_characterisation.py \
  --capture /mnt/c/proart_capture/small.nv12 \
  --long-capture /mnt/c/proart_capture/A.nv12 \
  --empty-capture /mnt/c/proart_capture/empty1.nv12
/home/proart/projects/glasses/.venv/bin/python -m pytest tests/test_ascii_dither_characterisation.py -q
```

## The generalisable lesson

The published inference compared a **rate** against a **chance baseline** and
called the difference "structure". The chance baseline was never a null of
the claim. A static scene is sticky by construction, so *every* static scene
sits below `2p(1−p)` — and a sticky scene with nothing moving would have
passed the same test.

The null that settles it was sitting on disk the whole time: an empty room,
in a different directory, in the same layout, needing no model and no
assumption. It changed `0.0000%` of cells. That single number is worth more
than the `52.6×` gradient it was supposed to be explained by.

And the recommendation that reversed twice did so on a comparison that could
not fail in the direction it failed: a *lower* change rate was read as a
*better* channel, when for these two channels it is a *lower sensitivity*.
A comparison that cannot come out the other way is not a comparison.
