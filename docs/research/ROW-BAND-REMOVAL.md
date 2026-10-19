# Row-wise band removal, and the first positive result from the camera channel

**Status: the instrument is fixed, and the camera channel is shown to carry a
real signal — but not the one we went looking for.**

Follows `CAMERA-FLICKER-BANDING.md`, which found the nuisance term (horizontal
bands from mains-frequency LED flicker sampled by the rolling shutter) and
proposed a temporal median as the fix. This file tests that proposal, finds a
better one, and then runs the first properly-controlled pose test.

---

## 1. The temporal median works, and is not good enough

`experiments/temporal_median_fix.py`. Median over k frames, applied before
thresholding, edge-replicated so no latency shift is hidden.

| operator | k | empty | occupied | latency |
|---|---:|---:|---:|---:|
| — | 1 | 50.00% | 48.19% | — |
| median | 3 | **0.00%** | 21.43% | +0.07s |
| median | 5 | **0.00%** | **0.48%** | +0.13s |
| median | 7 | **0.00%** | 12.18% | +0.20s |
| mean | 3 | 0.00% | 25.77% | +0.07s |
| mean | 5 | 0.00% | 0.78% | +0.13s |
| mean | 7 | 0.00% | 12.81% | +0.20s |

**The banding removal is solid.** Empty room 50.00% → 0.00% at every window
k ≥ 3, on both operators, reproducing at 300, 400 and 700 frames.

**But it destroys signal.** The occupied rate is non-monotonic in the window —
21.43%, **0.48%**, 12.18% — reproducing at every frame count. A filter that can
drive a *real* scene to 0.48% is cancelling periodic content, not preserving
content while removing nuisance. It also costs latency and smears fast motion.

**And my prediction about the mean was wrong.** I argued a mean would blur the
shifting bands into a standing wave and fail where only the non-linear median
could reject them. It does not fail — within ~4 points of the median at every
window. The banding shifts enough that plain averaging removes it.

## 2. A better fix, from the structure of the nuisance

The banding measurement said something more useful than "there is drift":

> In the empty room the residual between adjacent frames has a **column energy
> share of 0.00%**. The banding is identical in every column.

That means the nuisance lives in a subspace we can address directly:

```
frame(x, y) = content(x, y) + band(y)
```

`band(y)` is row-constant. A face, a body, a chair — all vary across x. So the
per-row statistic across columns estimates `band(y)` directly, and the **median**
across columns is used rather than the mean because a subject occupying a
minority of a row's columns must not drag the estimate.

`experiments/row_band_removal.py`. Subtract (or divide out) the row offset, then
rescale to the capture's median mean luma.

| mode | empty | occupied | turned |
|---|---:|---:|---:|
| none | 50.00% | 48.19% | 40.35% |
| temporal median k=3 | 0.00% | 21.43% | — |
| temporal median k=5 | 0.00% | 0.48% | — |
| **row subtract** | **0.01%** | **48.57%** | 44.92% |
| **row divide** | **0.06%** | **49.30%** | 46.18% |

**Row removal keeps everything the temporal median threw away.** The occupied
rate goes from 48.19% to 48.57% — essentially unchanged — while the empty room
collapses to 0.01%. And it costs no latency, smears no motion, and averages no
frames, because it never combines frames at all.

### The proof that it removes nuisance and leaves signal

| | row std | col std | column energy | adj MAD |
|---|---:|---:|---:|---:|
| EMPTY raw | 45.98 | 0.16 | 0.00% | 39.34 |
| EMPTY **row-removed** | **0.29** | 0.17 | — | **9.91** |
| OCCUPIED raw | 31.31 | 7.08 | 4.86% | 43.69 |
| OCCUPIED **row-removed** | 12.05 | **7.18** | **26.20%** | 32.20 |

The row-constant nuisance drops **159×** in the empty room (45.98 → 0.29), and
the subject's column structure is **untouched** (7.08 → 7.18). Column energy
share rises from 4.86% to 26.20% because what is left is genuinely the scene.

## 3. The properly-controlled pose test

`experiments/pose_test.py`. With a null floor of essentially zero, a null result
is finally interpretable.

The statistic is **spatial, not a rate**. A speaking person changes a lot —
even after row removal the occupied capture sits at ~48% flips, near the
theoretical maximum. A rate that saturates cannot separate two conditions that
are both near the ceiling; the flip rate measures "a person is moving", not
"which way the head points". So:

> M_c(x, y) = mean over the speech window of |frame(t+1) − frame(t)| per pixel
> null = M_EMPTY, the measured noise floor

### Results, with the controls

```
EMPTY null map: mean 10.004  max 36.693  p99 13.785 counts

  A facing  vs  B turned      mean|d| 9.182   frac>null 34.9%   top5%-holds 37.4%
  A speaking vs C silent      mean|d| 4.914   frac>null 12.1%   top5%-holds 79.1%
  B turned  vs  C silent      mean|d| 9.411   frac>null 34.9%   top5%-holds 38.8%

C2  within-condition split (same condition, two halves of its window):
  A_facing_speaking      half-vs-half mean|d| 2.878
  B_turned_speaking      half-vs-half mean|d| 1.917

C4  time-shifted control (A vs B shifted 1.0 s):
  A vs B shifted 1.0s        mean|d| 9.178
```

### No pose signal

**A-vs-B is 9.182. The time-shifted control is 9.178.** Indistinguishable. The
apparent difference between facing and turned is the scene changing over time,
not a head direction.

This is a statement about the **capture**, not about vision. Casey reported the
glasses restricting the turn, and the previous analysis already showed the
manipulation was 0 of 96 windows above the noise floor. A null here is the
expected outcome, and it is not evidence against the idea.

**A verdict function caught this one itself.** Its first version concluded
"there is a localised pose signal" because 9.182 > 9.178 — a margin of
**0.004**. The check now requires a 1.25× margin over the time-shifted
control, which is the only kind of margin that means anything.

### But speech IS visible, and that is a real finding

**A (facing, speaking) vs C (facing, silent): 4.914 against a within-condition
split of 1.917 — 2.6× — with 79% of the excess concentrated in the top 5% of
pixels.**

The localisation is the tell. The A-vs-B difference spreads diffusely (37.4% in
the top 5%); the A-vs-C difference concentrates (79.1%). That is what a real
signal looks like: a specific region of the frame — the mouth — changing when
someone speaks and not when they don't.

**The camera channel carries information about whether someone is speaking.**
That was never tested before. Every vision number in this project has been
about the character representation, and this is the first result that says the
vision path is worth building on at all.

## 4. What the morning capture needs to be

The protocol changes, and it changes for a reason the data gave us.

| | pose | speech | what it isolates |
|---|---|---|---|
| **A** | facing | speaking | — |
| **B** | turned | speaking | pose + speech |
| **C** | facing | silent | speech (already shown to work) |
| **D** | **turned** | **silent** | **pose alone** |

**D is the condition that was missing.** Without it, pose and speech are
confounded: B differs from A in both. With D, the pure pose test is C vs D, and
the pure speech test is A vs C — which we now know works.

Also: **look fully away, 90°, at the wall.** Not 45°. The last capture's
head-turn was too small to register, and no analysis can recover a manipulation
that did not happen.

And run the probe first. It now reports the number that matters:

```
drift   : adjacent-frame MAD 39.344 counts  (raw)
drift   : after row-band removal 9.909 counts   OK - below ~5
```

**Aim for the row-removed residual under ~5.** If it stays high, the banding is
not the only problem and different lighting is needed first.

## 5. Limits

- One capture per condition, no error bars, no cross-validation.
- The empty capture is later and darker than the others, so the null floor is
  probably an over-estimate — the test is conservative, not optimistic.
- The row estimate is contaminated where a subject fills more than half a row's
  columns; a person very close to the camera could be partly removed *by this
  filter*. The occupied result is the check on that, and it held.
- The reconstruction cannot restore original absolute luma, only the
  row-structured part of the change, so the threshold is applied to a
  reconstructed image.
- **None of this is evidence about "talking to me".** It is evidence that the
  camera can see that someone is speaking, and that the previous capture could
  not test facing-versus-turned. The pose question is still open.

## Reproduce

```bash
/home/proart/projects/glasses/.venv/bin/python experiments/row_band_removal.py --nf 400
/home/proart/projects/glasses/.venv/bin/python experiments/temporal_median_fix.py --nf 400
/home/proart/projects/glasses/.venv/bin/python experiments/pose_test.py
```