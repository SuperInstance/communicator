# Empty-room control: the character delta is the camera, not the subject


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
> **The empty room's 50.00% banded fraction was the stride defect.** With
> the correct 19,200-byte stride the empty room shows **0.0000%** banding
> and its mouth-region motion is **6,600× quieter** than the occupied one
> (0.015% of it; the whole frame is 103× quieter). The
> empty room is a good control; the mechanism it was read as supporting
> is not there. See `docs/research/NV12-STRIDE.md` §5.

**Status: decisive, and it retracts a mechanism claim in
`ASCII-MOTION.md`.**

Run overnight, 2026-10-18, with Casey asleep. 150 s, 4500 frames, empty room.
This is the control the ASCII-motion lane never had: every vision number in this
project had been measured on a scene with a person in it, so nothing could
distinguish "the subject moved" from "the camera did something".

---

## The result

`experiments/emptyroom_null.py`, same Syzygy kernel, same `braille_thresh=100`,
same 160×96. 400 adjacent-frame pairs per condition.

| | mean luma | adjacent MAD (raw) | **dot flip rate** | spread |
|---|---:|---:|---:|---|
| **OCCUPIED** — subject present, mostly still | 111.18 | 43.686 | **48.19%** | 33.93–56.82% |
| **EMPTY** — nobody in the room | 49.70 | 39.344 | **50.00%** | 25.00–50.00% |

**empty / occupied = 1.038**

The empty room produces a *higher* flip rate than the occupied one. And
**50.00% is the theoretical maximum** for a binary field — every dot flipping
every frame, consecutive masks exact complements. Many frames sit exactly at the
ceiling.

**Conclusion: the character delta does not measure motion.** An empty room
produces the same maximal change. Whatever Syzygy's Braille field is reporting
frame to frame, it is the camera and the threshold, not a person in the room.

This was foreseeable and is the lesson of the whole lane: a representation
demonstrated to *change* was never demonstrated to change *for a reason*.

## Retraction: the 3.1× dither gradient does not reproduce

`ASCII-MOTION.md` reported a monotone **3.1× gradient** in dot-flip rate with
distance from the threshold (67.25% on-threshold down to 21.61% far away), and
built a mechanism on it. Recomputed on the occupied capture:

| capture | on-threshold | far-from-threshold | gradient |
|---|---:|---:|---:|
| `small.nv12` (original) | 67.25% | 21.61% | **3.1×** |
| `A.nv12` (this run) | 45.71% | 48.38% | **0.94× — flat** |
| `empty1.nv12` | — | 39.11% | 1.22× |

The gradient is **absent** on a different capture of the same scene. It was a
property of one recording, not of the pipeline. The "half the delta is
threshold dither, half is motion" reading is withdrawn — the empty-room control
shows the far-from-threshold residual is not motion either.

## The vision map, which does work

Over 2000 frames of a scene nobody is touching, the **median composite** is
stable to a degree nothing else in this project has managed:

```
median-map step drift:  mean 0.0002   max 0.0005   (per-pixel mean luma)
```

A per-pixel median over a few thousand frames removes sensor noise and the
exposure hunt, because both are near zero-mean, and keeps the scene. This is the
cheapest denoiser available and it was never run here before, because every
capture to date has been 25–750 frames *of a moving subject* — a moving subject
is the one thing a median cannot denoise.

`experiments/room_median_map.pgm` (160×96, greyscale) is the map.

This also tells you the substrate is static: a room map built once stays valid,
so differences against any future capture are content, not camera drift.

## The sound: too quiet to characterise, and that is the finding

```
duration   149.9 s
rms        -73.5 dBFS
p95        12 / 32768
```

The room is essentially silent. No acoustic map is meaningful at this level, and
there is no reverberant tail to measure — **119 transients were detected and
zero produced a valid RT60 estimate**, which is exactly what a room with no
reverberant energy looks like.

Two bugs found and fixed while doing this, both worth recording:

**The Goertzel filter was fabricating its spectrum.** `k = int(2 * math.pi *
freq / sr)` truncates to 0 for every frequency below ~2.5 kHz, so `coeff =
2*cos(0) = 2` identically and it returned *the same power for 50, 100, 200 and
1000 Hz*. The tell was eight harmonics all printing `33.4` to one decimal, which
real noise cannot do. Had this gone unreported it would have become a room hum
table, because it looked entirely plausible.

## What has to change before any motion claim

> ### ⚠ THIS SECTION WAS WRONG — see `CAMERA-FLICKER-BANDING.md`
>
> The gain-invariance fix proposed below has been implemented and tested. **It
> does not work**: per-frame gain normalisation leaves the empty-room flip rate
> at exactly 50.00%. Measured afterward, a single scalar per frame explains only
> 2.80% (occupied) / 4.35% (empty) of the frame-to-frame variation. The
> dominant term is **horizontal banding from mains-frequency LED flicker
> sampled by the rolling shutter** — in the empty room the residual's column
> energy share is 0.00%, meaning it is identical in every column, which is what
> a light band looks like and what no room feature looks like.
>
> **A number in the text below is also an arithmetic artefact.** "Per-frame mean
> removal cuts adjacent drift from 39.3 to 0.81 counts" is wrong: it compares raw
> counts against relative units. 43.686/111.18 = 0.393 and 43.226/109.71 = 0.394,
> so the real reduction is **1.00×**. Normalisation is still needed to put two
> captures into the same units for comparison, but this document overstated what
> it achieves.
>
> The original text is kept below rather than deleted, because it is a fair
> record of the reasoning and of where it went wrong.

The fix is now specific and it is not a tuning knob.

**The threshold must be gain-invariant.** Syzygy compares raw luma to a fixed
`braille_thresh`. A global gain change moves *every* dot across the threshold
together, which is precisely the failure: an empty room at 39 counts of
per-frame drift produces a maximal flip rate. Thresholding a per-frame
gain-normalised image would make the comparison scale-invariant and remove the
mechanism outright.

**Post-hoc normalisation is a workaround, not a fix.** Per-frame mean removal
cuts adjacent drift from 39.3 to 0.81 counts, which makes comparison possible —
and this analysis used it — but it happens *after* thresholding, so it cannot
recover information the threshold already destroyed.

**Capture-side exposure lock FAILED on this driver.** The script requests
`CAP_PROP_AUTO_EXPOSURE = 0.75` and reads back `0.0`; drift stays at 39.3
counts. This is recorded rather than papered over, and it is why the room was
captured dark (mean luma 49.7 vs 111.2 occupied) — with the lights off the AGC
has more gain range to swing through. A driver-level exposure lock, or lighting
bright enough that AGC never hunts, is required.

## Limits, stated

- **One empty capture and one occupied capture.** No error bars. The direction
  and magnitude (1.038, at the theoretical ceiling) make this hard to explain as
  noise, but it is a demonstration, not a significance test.
- **The two recordings differ in lighting as well as occupancy.** Per-frame gain
  normalisation removes frame-to-frame drift but cannot undo a global difference
  between two recordings taken hours apart. This confound cannot be removed
  retrospectively and is the weakest link in the comparison.
- **The occupied capture's A-vs-B condition failed separately** (0 of 96 windows
  separated them). It is valid here as "subject present and mostly still", which
  is all this experiment asks of it.
- One room, one night, one machine.

## Not settled

Whether vision can distinguish *talking to me* from *talking to someone else* is
still completely open. This result removes a false positive — it is not evidence
against the idea, it is evidence that the current instrument cannot be trusted
to produce a negative. Casey redoing the three-condition capture in the morning
with a bigger head turn is still the right next step, and now has a specific
bar to clear: **the flip rate in the facing and head-turned conditions must
differ from the empty-room baseline of 50.00%.** Anything at or near 50% means
the camera is still dominating.

```bash
/home/proart/projects/glasses/.venv/bin/python experiments/emptyroom_null.py \
  --empty empty1.nv12 --occupied A.nv12 --limit 400
/home/proart/projects/glasses/.venv/bin/python experiments/room_map.py \
  --wav empty1.wav --nv12 empty1.nv12
```