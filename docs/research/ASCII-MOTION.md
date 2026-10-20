# ASCII motion: the character delta is half threshold dither, and the tone channel is a dead end


> ## CORRECTION (2026-10-20) — the captures were read at the wrong frame stride
>
> Every number in this file was computed on **torn frames**. The captures
> carry **19,200 bytes per frame** (luma 160×96 + interleaved U,V chroma at
> 80×24), not the 23,040 of a standard NV12 frame, so every "frame" was
> rows 0..71 of one moment spliced to rows 72..95 of the next.
>
> The full account, with the evidence and the fix, is
> **`docs/research/NV12-STRIDE.md`**. The reader is now `src/nv12.py`, which
> detects the layout from the content. The corrected corpus is
> `/mnt/c/proart_capture/corpus_v2/`.
>
> **The measurements here are on torn frames.** The ASCII motion maps, the
> 50.00% empty-room banded fraction and the 600-frame / 20.0 s corpus
> geometry are all artefacts of the 23,040-byte stride. The corrected
> corpus is 900 frames at 30 fps = 30.0 s. This document's existing
> retraction of the threshold-dither mechanism stands on its own terms;
> the numbers it retracts are now retracted again for a different reason.

> ## ⚠ PARTIALLY RETRACTED — read `EMPTY-ROOM-CONTROL.md` first
>
> This document's central mechanism claim — "roughly half the delta is threshold
> dither, and the structured far-from-threshold residual is real motion" — **does
> not survive its control.**
>
> `EMPTY-ROOM-CONTROL.md` ran the same measurement with **nobody in the room**.
> Empty-room dot flip rate: **50.00%** (the theoretical maximum). Occupied:
> **48.19%**. The empty room flips *more*. So the structured residual is not
> motion — it is camera exposure drift plus the threshold, and an empty room
> produces the same maximal change.
>
> Additionally, the **3.1× dither gradient in §6 does not reproduce**: on a
> different capture of the same scene (`A.nv12`) the gradient is **0.94×, flat**.
> It was a property of one recording, not of the pipeline.
>
> What survives here: the ladder never operates (24/24 frames pinned at the
> coarsest rung), the rotated-null control shows it consumes only a change count
> (100.0% rung agreement), and the tone channel is worse than the mask at
> 70.91% — the control that killed my own proposed fix.
>
> What does not survive: the motion interpretation, and the gradient.

**Status: measured. The representation idea survives; two of its load-bearing
assumptions do not.**

Four experiments, run in order, each one correcting or refuting the last. Two of
the four intermediate conclusions in this document were wrong, and both are kept
here rather than edited out, because the sequence is the finding: *every*
measurement in this lane produced a plausible mechanism that a control killed.

Capture: `/mnt/c/proart_capture/small.nv12`, 25 frames, 160×96 NV12, live ASUS
FHD webcam, stationary subject (me, sitting, speaking).

---

## 1. The idea, and why it was worth trying

`docs/research/TWO-SPEAKER-TURNS.md` established that at the moment of decision,
END and RESUME are byte-identical audio. Every route that reads a single instant
failed — silence duration AUC 0.529, F0 0 of 6 cells monotone. So a
representation that makes *motion* first-class is not a refinement; it is one of
the few places the information can physically be.

Syzygy already does half of it: one integer-only pass turns a frame into an
array of characters (Braille masks, edge glyphs, a tone ramp), 0.1 ms for
160×96, byte-deterministic, no model. `experiments/ascii_motion.py` treats
consecutive glyph fields as video frames:

- **I-frame** — the whole field, emitted when the delta would cost more
- **P-frame** — the *change*, not the picture

Two promises follow. Motion becomes first-class because the pipeline carries the
difference explicitly. And resolution scales inversely to frame rate, because a
delta's cost is proportional to how many cells changed — drop spatial detail
when much moved, keep it when little did. Output stays text, so a text-only
model can read it, which matters more than it sounds: `SPECTROGRAM-TURN.md`
found the only image-capable model available locally answers a constant
"different" to every spectrogram *and to a blank rectangle*.

## 2. First run: the ladder never operates

`ascii_motion.py`, 25 frames, 80×24 cells:

| statistic | value |
|---|---|
| `full_changed` median | 1074 / 1920 cells (**55.9%**) |
| min / max | 890 / 1323 |
| `frames_needing_downscale` | **24 / 24** |

Two failures, same shape. 56% of the field changes per frame on a stationary
subject. And with the delta that large, the resolution ladder can never come off
its coarsest rung — so "resolution scales inversely to frame rate" was not
demonstrated, it was **collapsed to a constant**. A knob pinned at one end is
not a knob.

## 3. Nulls: the delta is not monotonic in lag

`ascii_motion_nulls.py`. Motion accumulates — lag-20 must differ from lag-1 more
than lag-1 differs from lag-0. Nothing that *moves* behaves otherwise. The
capture does the opposite, sharply:

```
lag  1: 55.94%    lag  2: 57.55%    lag  3: 57.45%
lag  5: 11.46%    lag 10: 15.16%    lag 20: 18.96%
```

Change *falls by two thirds* as lag grows, with a discontinuity between lag 3 and
lag 5.

- **N1 determinism — PASSES.** Frame 0 rendered twice: 0.00% change. The kernel
  is deterministic, so this is signal noise, not renderer noise.
- **N3 null — FAILS THE REPRESENTATION.** A rotated permutation of each delta
  (same change *count*, no structure) selects the **identical ladder rung on
  100.0% of frames.** The ladder consumes nothing but the count. It is a rate
  meter wearing a resolution knob's clothes.

## 4. Wrong #1: "auto-exposure oscillation"

Mean luma autocorrelation: lag 1 **+0.566**, lag 2 **−0.450**, lag 3 **−0.678**,
lag 5 **+0.436**, against a static-repeat control of exactly +1.000 throughout.
Mean-luma span 14.01 counts live vs 0.00 control. A negative trough at lag 2–3
with a positive lag-1 *is* the fingerprint of a hunting gain loop. The cross-test
agreed: correlation between mean |ΔY| and cell-change, indexed by lag, was
**+0.702**. I wrote the verdict "the character delta is dominated by the camera's
AGC".

**That verdict was retracted**, by the next experiment.

*(My first version of this check also reported "no oscillation" on data that
plainly contained one, because its predicate tested for negative correlation at
lag 1. A hunting loop has momentum — lag 1 stays positive, and only lags 2–3
reach the opposite extreme. A verdict function that can contradict its own
printed data is not a measurement.)*

## 5. Wrong #2: "+0.702 is one leverage point"

Reading the cross-test table again: lag 6 had the *second-lowest* |ΔY| and the
*second-highest* change. One row, lag 5, was low on both. A six-point
correlation carried by one point is an anecdote, so I ran leave-one-out
(`ascii_motion_robustness.py`), expecting it to collapse.

```
all lags            +0.702
drop lag 1  +0.849  drop lag 2  +0.685  drop lag 3  +0.717
drop lag 4  +0.702  drop lag 5  +0.943  drop lag 6  +0.736
```

It does not collapse. Removing the most influential point makes it **stronger**.
**My retraction was itself wrong**, and the exposure association stands. Also
checked and rejected: no near-duplicate frame pair exists (global minimum is
7.71% change), so the lag-5 dip is not a repeated frame.

Two wrong answers in a row, and the lesson is not "trust the third one". It is
that a mechanism can be real at the same time as a much larger effect sitting on
top of it.

## 6. The actual mechanism, at the right granularity

`ascii_motion_threshold.py`. `braille_thresh` is a `uint8` in `SyzFusedParams`
compared against cell luma. But 160×96 → 80×24 means each cell is **2 wide × 4
tall = 8 dots**, and the output byte is those 8 dots. The threshold applies to
**each pixel**, not to a cell average.

That granularity error was mine and it mattered. The first version of this test
binned *cells* by mean luma and reported "far-from-threshold cells still flip
47%" — a flat floor that looked like a refutation. Binning by cell mean measures
the wrong quantity: a cell averaging 40 counts above 100 can still contain a
pixel sitting exactly on 100. At dot resolution the effect is unambiguous.

| \|luma − 100\| | dots | flip rate |
|---|---|---|
| 0 (on it) | 3,487 | **67.25%** |
| 1–2 | 14,016 | 49.27% |
| 3–5 | 22,583 | 48.39% |
| 6–10 | 38,208 | 40.96% |
| 11–20 | 64,950 | 37.71% |
| 21–40 | 127,306 | 34.48% |
| 41+ | 113,450 | **21.61%** |

**Monotone, 3.1× end to end.** Dot set density 77.73%, so the independence
baseline for unchanging-but-uncorrelated content is 2p(1−p) = **34.62%**.

## 7. What is actually true

Both effects are real, and the interesting number is the *floor*, not the peak.

1. **Roughly half the raw delta is threshold dither.** Dots sitting on the
   threshold flip at 67.25%, nearly twice the independence baseline. A single
   bare comparison has no hysteresis, so any perturbation flips the bit. This is
   the noise the resolution ladder was spending its entire budget on.

2. **The residual is structured, not noise.** Far from any threshold, dots still
   change at **21.61% — well below the 34.62% independence baseline.** Sub-
   independent change cannot be random flicker; it means content that moves
   *coherently* between frames. **There is genuine motion in this capture and it
   is recoverable.**

3. **The tone channel is a dead end.** The control came back at **70.91%** —
   *worse* than the mask, not better. I had proposed the argmax ramp as the
   stable fallback channel on the reasoning that argmax is smoother than a bare
   threshold. The control refutes that. A P-frame built on tone would be strictly
   worse. This is the cleanest negative result in the lane: a plausible fix,
   killed by the control built specifically to test it.

## 8. What follows

**Hysteresis, sized by the measured gradient.** A dot's bit should be held until
its luma crosses the threshold by a margin, not on a single comparison. The
gradient gives the margin for free: beyond ~20 counts a dot is already 1.6×
more stable than one on the threshold, so a margin of that order separates the
two populations. This is the one concrete, measured fix the lane produces.

**The ladder needs measured inputs, not assumed ones.** With a 55.9% baseline it
saturated at the coarsest rung on 24/24 frames and a null delta reproduced its
decisions exactly. `BUDGET = 40` was chosen before any measurement existed. It
must be set from the *hysteresis-corrected* change rate, and the rung decision
must consume something a permutation cannot reproduce — position and magnitude,
not a count.

**The capture needs per-frame checksums.** Every stage of this investigation was
contaminated by not knowing which frames were real. Frame identity should be
recorded at capture time, not inferred afterward from a change rate.

**Still unproven, and it is the part that matters.** Nothing here shows this
representation separates *talking to me* from *talking to someone else*. A
demonstration that motion is visible is not a demonstration that motion is
sufficient. That needs labelled data from the three-condition protocol, and
nothing in this document substitutes for it.

## Reproduce

```bash
/home/proart/projects/glasses/.venv/bin/python experiments/ascii_motion.py \
  --capture /mnt/c/proart_capture/small.nv12
/home/proart/projects/glasses/.venv/bin/python experiments/ascii_motion_nulls.py \
  --capture /mnt/c/proart_capture/small.nv12
/home/proart/projects/glasses/.venv/bin/python experiments/ascii_motion_cause.py \
  --capture /mnt/c/proart_capture/small.nv12
/home/proart/projects/glasses/.venv/bin/python experiments/ascii_motion_robustness.py \
  --capture /mnt/c/proart_capture/small.nv12
/home/proart/projects/glasses/.venv/bin/python experiments/ascii_motion_threshold.py \
  --capture /mnt/c/proart_capture/small.nv12
```

## The generalisable lesson

Three plausible mechanisms were produced in a row — auto-exposure, a duplicate
frame, threshold dither — and the first two were wrong. The one that survived
was found only because the *granularity was wrong*: the effect was invisible at
cell resolution and unambiguous at dot resolution.

And the single most valuable number in the whole lane was the one designed to
kill my own preferred fix. The tone channel was proposed by me, as the obviously
more stable alternative to a hard threshold. It measured **worse**.