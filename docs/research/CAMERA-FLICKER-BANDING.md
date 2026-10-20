# The camera's drift is LED flicker banding, not gain, and not noise


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
> **This document's mechanism claim is WITHDRAWN.** The horizontal bands
> were not LED flicker in the scene: they were the tear. On the corrected
> frames an **empty room with no filter at all shows 0.00% banding, where it
> showed a perfect 50.00%.** There is no flicker nuisance term to find.
> The row-removal pass proposed here is now unnecessary and actively
> harmful: with nothing to remove, it *raises* the occupied rate from
> 0.625% to 1.790%. See `docs/research/ROW-BAND-REMOVAL.md`.

**Status: mechanism identified with a measurement. The fix I proposed in
`EMPTY-ROOM-CONTROL.md` is wrong and is withdrawn here.**

This is the R&D that ran while Casey was asleep, and its first job was to test
whether the morning capture would be worth recording. It isn't — not until the
lighting changes.

---

## The fix I proposed, and why it failed

`EMPTY-ROOM-CONTROL.md` closed with:

> the threshold must be gain-invariant… a global gain change moves every dot
> across it together, which is exactly the failure mode.

`experiments/gain_invariance.py` implemented that. It failed:

```
OCCUPIED  raw flip 48.19%  ->  normalised 51.43%
EMPTY     raw flip 50.00%  ->  normalised 50.00%
```

Per-frame gain normalisation changed nothing.

## And a correction to a number I had already reported

The run exposed an arithmetic error in my own earlier claim. I had written that
normalisation "cuts adjacent drift from 39.3 to 0.81 counts". It does not:

```
raw         43.686 / 111.18  =  0.393
normalised  43.226 / 109.71  =  0.394
```

Dividing by a mean of 111.18 and dividing by a mean of 109.71 are the same
operation. The reduction was **1.00×, not ~50×**. I had compared raw counts
against relative units — a factor-of-100 illusion in a number I put in a
committed document.

This matters beyond the arithmetic: it is why the gain fix could not work. There
was nothing to remove.

## What the drift actually is

`experiments/drift_structure.py`. For every adjacent frame pair, fit the single
scalar *k* that best explains the pair (least squares through the origin), remove
it, and examine what is left.

| | gain explains | residual RMS | after 2×2 averaging | pure noise would give |
|---|---:|---:|---:|---:|
| OCCUPIED (lit) | **2.80%** | 52.32 | 0.969 | 0.50 |
| EMPTY (dark) | **4.35%** | 50.88 | **1.000** | 0.50 |

**Not gain.** One scalar per frame accounts for under 5%.

**Not pixel noise.** Independent noise drops to ~0.50 under 2×2 block
averaging, because averaging four samples halves their variance. This residual
barely moves at all — 0.97 and 1.00. It *survives* spatial averaging, so it is
large-scale structure.

**A third verdict function contradicted its own output while producing this.**
Its first version concluded "gain explains almost none of it → PIXEL-LEVEL NOISE"
while the line above it printed an averaging ratio of 1.00. That is the third
such failure in this project and the subject of
`HOW-WE-KEEP-FOOLING-OURSELVES.md`, which I wrote earlier the same day and then
immediately repeated. Each classification now cites the number supporting it.

## What structure: horizontal light bands

Separate the residual's variation along rows from its variation along columns.

| | row-profile std | col-profile std | column energy share |
|---|---:|---:|---:|
| **EMPTY (dark)** | **45.98** | **0.16** | **0.00%** |
| OCCUPIED (lit) | 31.31 | 7.08 | 4.86% |

In the empty room the residual is **identical in every column** — column energy
share 0.00%. A window, a chair, a face all vary horizontally as well as
vertically. A horizontal light band does not. This is banding and nothing else.

The row spectrum is dominated by **bins 1, 2 and 3** (46% of energy in the top
three) — 1–3 cycles across 96 rows scanned in 33 ms at 30 fps, i.e. **roughly
30–90 Hz**. That is consistent with mains-derived LED flicker sampled by the
rolling shutter: each frame scans the sensor at a different phase of the lamp
cycle, so the exposure varies down the frame and the bands move.

In the occupied capture the column share rises to 4.86% — that is Casey, on top
of the same banding.

**What is not established:** which lamp or driver is responsible. This cannot
separate 50 Hz from 60 Hz mains, nor flicker from a PWM driver at another
frequency. It identifies the *shape* — horizontal bands at mains-ish frequency —
not the source.

## What this means for the morning capture

**The capture is currently swamped by the lighting, not by your head turn.** In
the lit room the banding is 31 counts of row-profile variation against a total
adjacent-frame difference of 43.7 counts. There is not enough room left for a
pose change to register.

So the instruction I gave you this morning — bigger turn, clearer framing — is
the wrong lever. It would help a little, and change nothing about the fact that
~70% of the frame-to-frame variation is the lamps.

> **UNSETTLED (2026-10-19).** The "~70%" share is **not** derived from two
> quantities measured over the same thing: it divides a **spatial** standard
> deviation (31.312) by a **temporal** MAD (43.686). For a Gaussian,
> MAD = 0.6745σ, so on its own terms the share would be ≈48 %, not 71.7 %. The
> number is left as printed and is **not** settled. **What would settle it:**
> decompose each frame pair into a row-constant part and a column-varying part
> and report the fraction of the *temporal residual* explained by the row term.
> Flagged as a candidate, not a finding — `NOTES-AUDIT.md` F12.

### What to change instead

1. **Change the lighting.** This is the lever. A different lamp, or daylight,
   or a different room. The banding comes from a mains-driven source; swapping
   the source removes the measurement's dominant nuisance term. Everything
   downstream is easier once this is fixed.
2. **Try a shorter exposure.** OpenCV could not lock exposure on this driver
   (`CAP_PROP_AUTO_EXPOSURE` reads back 0.0), so a faster shutter is not
   reachable from the capture script. If the camera has a physical or driver
   setting for it, a scan faster than the flicker would average the bands out.
3. **Confirm before recording.** Run the 6-second probe and read the
   `adjacent-frame MAD` line. On a bad setup it prints `TOO HIGH`. Aim for
   **under ~10 counts** before you sit down and speak — that is the bar, and it
   is now measurable rather than a guess.

If the probe still reads `TOO HIGH` after a lighting change, the honest
conclusion is that this camera plus this room cannot currently support the
experiment, and that is worth knowing before spending a morning on it rather
than after.

## The fix that survives

Not gain normalisation. **Median filtering across time before thresholding.**
The banding is a phase-varying pattern, so consecutive frames see different band
positions, and a temporal median of 3–5 frames averages them out while a subject
present for more than half the window survives. It costs latency and it is a real
trade, but unlike the gain fix it targets the mechanism actually measured.

That is untested and is the next thing to run once there is a capture with the
banding reduced enough to measure against.

## Reproduce

```bash
/home/proart/projects/glasses/.venv/bin/python experiments/gain_invariance.py --nf 400
/home/proart/projects/glasses/.venv/bin/python experiments/drift_structure.py --nf 400
```

## Limits

- One camera, one room, one lighting state. The mechanism is identified here; it
  is not shown to be the dominant term under different lighting.
- The empty capture was taken in a dark room (mean luma 49.7 vs 111.2 occupied)
  and the two are hours apart. Absolute counts are not comparable between them;
  the row/column split and the averaging ratio are.
- The 30–90 Hz estimate comes from three FFT bins on 400 frames, not from a
  resolved peak. It is a band, not a frequency.
- Nothing here re-opens whether vision can distinguish "talking to me". It says
  the current instrument cannot answer that question yet.
---

## Addendum: the temporal median, tested

`experiments/temporal_median_fix.py`. Median over k frames applied *before*
thresholding, edge-replicated so no latency shift is hidden. Both operators are
tested, because if a mean worked as well as a median the cheaper operator would
be the right answer and I wanted to know that rather than assume it.

| operator | k | empty | occupied | latency |
|---|---:|---:|---:|---:|
| — | 1 | 50.00% | 48.19% | — |
| median | 3 | **0.00%** | 21.43% | +0.07s |
| median | 5 | **0.00%** | **0.48%** | +0.13s |
| median | 7 | **0.00%** | 12.18% | +0.20s |
| mean | 3 | 0.00% | 25.77% | +0.07s |
| mean | 5 | 0.00% | 0.78% | +0.13s |
| mean | 7 | 0.00% | 12.81% | +0.20s |

### The banding removal is solid

The empty room goes to **0.00%** at every window k ≥ 3, on **both** operators,
and reproduces at 300, 400 and 700 frames. That is the actionable result: the
50.00% ceiling is removable in software, and the camera is now usable.

### And my prediction about the mean was wrong

I argued in the script that a mean would blur the shifting bands into a
low-amplitude standing wave and fail, while only the non-linear median could
reject them. It did not fail — at every window the mean lands within ~4 points
of the median. The banding shifts enough that plain averaging removes it too.
The median is marginally better at k=3 and costs more to compute, but with one
capture per condition and no error bars that difference is not established.

### The second half is a warning, and it is the important half

The occupied rate is **non-monotonic** in the window: 21.43%, **0.48%**, 12.18%.
It reproduces at every frame count I ran, so it is structure, not noise.

A filter that can drive a *real* scene to 0.48% is not selectively preserving
content — it is cancelling periodic content. At k=5 it is destroying exactly the
signal we want. So:

- **k=3 is the best available**, and it is not a safe drop-in fix at any fixed
  window.
- **"Real motion survives the filter" is not established by this data.** My
  first version of this script's verdict claimed it, on the strength of the k=3
  row alone, while the k=5 row in the same table refuted it. That is the fourth
  verdict function in two days to contradict its own printed table.

The honest next step is a filter that can *tell banding from content* — the
banding is stationary in frequency and moving in phase, content is neither — or
a camera and lighting combination with no banding to remove.

## Practical status

The camera instrument now has a usable null: an empty room reports 0.00% rather
than 50.00%. **The morning capture is worth recording**, with a k=3 temporal
median applied before thresholding, and the bar is that the facing and
head-turned conditions must differ from **0.00%**, not from 50%.

The lighting advice from the section above still stands on its own merits —
reducing the banding at source is better than filtering it out — but it is no
longer blocking.
