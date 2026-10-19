# The camera's drift is LED flicker banding, not gain, and not noise

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