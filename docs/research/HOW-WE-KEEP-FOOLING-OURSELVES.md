# How we keep fooling ourselves: four retractions, one shape


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
> **A FIFTH instance, and it is the largest.** A hard-coded constant — the
> frame stride — sat in every reader, it was plausible, and every check on
> it was derived from itself: the frame count was verified against a
> frame count computed with the same wrong stride. The check has to come
> from outside the assumption (the file size, the script on disk, the
> audio's duration, a human eye). See `docs/research/NV12-STRIDE.md` §9.

**Status: a synthesis, not a finding.** Every instance below is documented in
detail elsewhere; this document exists because the *shape* repeats across three
independent lanes, and a shape that repeats is worth naming.

---

## The four

| # | Claim | What it actually was | Lane |
|---|---|---|---|
| 1 | `GATE_FIELD` understates the gate by **14.5×** because `_trim` drops its peak | The 0.7293 peak sits at **t = 0.064 s, frame 2** — an LSTM warm-up transient. I compared it to records at t≥4 s. Different intervals. No bug. | audio |
| 2 | Roughly half the character delta is **threshold dither**, the rest is motion | An **empty room** flips 50.00% of dots vs 48.19% occupied. 50% is the theoretical maximum. The delta is the camera. | vision |
| 3 | The dither gradient is a monotone **3.1×** | **0.94× on a different capture** — flat. It was a property of one recording. | vision |
| 4 | Watcher cost above 100% is a **realtime saturation floor** | A **numpy-fed harness** artifact worth ~2× (`math.isfinite` per sample against an `np.float32`). Nothing saturates; max is 49.1%. | cost |

Three of the four were mine. One (#2) was a control that had never been run.

> ## CORRECTION (2026-10-20) — a fifth retraction, and it is a different shape
>
> | # | Claim | What it actually was | Lane |
> |---|---|---|---|
> | 5 | "LED flicker banding" (a perfect **50.00%** in an empty room); the teacher's USABLE verdicts (**+0.208**, **+0.312**); a pose feature at **1.71×** | **Every reader in the repo hard-coded a 23,040-byte frame stride. The files are 19,200.** 80% of every "frame" was a tear between two real moments — rows 0..71 of frame *k* spliced to rows 72..95 of frame *k+1* — so the "motion" the repo measured was ~99% artefact. | vision |
>
> The full account is `docs/research/NV12-STRIDE.md`. Four of the five were
> mine; #2 was again a control that had never been run, and the empty room
> that killed it also killed #5 (an empty room shows **0.00%** banding on the
> corrected frames, where it showed a perfect 50.00%).
>
> **The fifth is NOT the shared shape of the first four, and that is why it
> is worth naming separately.** The first four all compared a quantity
> summarised over one interval against a quantity summarised over a
> different one — a mistake in the comparison. The fifth is a mistake in the
> *data*, and it is invisible to every kind of internal check:
>
> > **A constant that is inherited is a constant that is never tested.**
> > The stride was written down once, in the first reader, and every other
> > reader copied it — and every reader's *own* check confirmed it, because
> > each check was computed with the same wrong constant.

## The shared shape

Every wrong claim compared **a quantity summarised over one interval against a
quantity summarised over a different one.**

1. A whole-signal max against one record's max.
2. An occupied room against an empty room — never both, until last night.
3. One capture's gradient, called a pipeline property.
4. A cost figure from a harness that was not the harness anyone else used.

None of these are exotic. Each felt obviously right while I was writing it down.
That is the part worth taking seriously.

## The controls that would have caught them

Cheap, and each one is now in the repo:

- **Compare like with like.** Same interval, same container, same capture
  conditions, or the ratio is meaningless. This is what killed #1 and #4, and it
  is the whole of #3.
- **Run the null condition.** The empty room. A measurement of "motion" that has
  never seen a room without motion has not measured motion. This is what killed
  #2, and nothing else could have.
- **Change one thing and re-run it on a different sample.** The single most
  effective habit here. #3 died the moment it was run on `A.nv12`.
- **Ask what the number would look like if the claim were false.** A suspicious
  100%, or eight identical spectral values, or a ratio of exactly 1.000 — these
  are artefacts until proven otherwise. All three appeared in this session.

## The traps that actually bit

**Slicing a stateful model is not a control.** My re-check of #1 ran a fresh
`SileroGate()` per record and got a flat 0.07 ratio, which *appeared to confirm*
my own bug. The gate is an LSTM; every slice re-ran the same start-up
transient. The verification manufactured the evidence it appeared to produce.
Any control that involves re-initialising a stateful component needs to be
checked for that specifically.

**A control that cannot come out the other way is not a control.** I dismissed
the warm-up hypothesis with "max over the first 0.5 s gives the same 0.7293" —
but a warm-up transient and real signal answer identically, so that control was
incapable of refuting anything. Before believing a control, check what result
would have made me change my mind.

**Verdict functions must be able to fail.** One script of mine printed
"There IS a retention bug" having found **zero records to compare**. It is
now hard-coded to print `UNDETERMINED` and refuse to conclude. A function that
can contradict its own printed data is not a measurement.

**Fix the granularity before believing a null result.** The dither gradient was
invisible at cell resolution and unambiguous at dot resolution — because
Syzygy thresholds *each pixel* of a 2×4 cell, not a cell average. Measuring the
wrong quantity made a real effect look like its absence, and would equally have
made a real absence look like an effect.

**A check derived from the assumption it tests cannot fail.** The stride
survived because the frame count was verified against a frame count computed
with the same wrong divisor — a check that could only ever agree. The check
has to come from *outside* the assumption: the file size, the script on disk,
the audio's own duration, a human eye. Three independent lines of evidence
were needed (file size plus audio duration; the mean-|diff| collapse from
43.560 to 0.545; a human looking at the frames and seeing a face rather
than a tear). **Any one of them alone would have been arguable**, and the
file-size one is the one that would have caught it in a second.

## What this says about the project

Everything above is cheap to prevent and expensive to discover late. The cost
is not the wrong answers — it is that a wrong answer committed at 3am reads
exactly like a right one, and the next agent inherits it as settled.

So the standing practice, which this document exists to enforce:

> Every measurement names its condition, its interval, and its null.
> Every verdict is able to fail, and is shown failing when it is built.
> A number that has only ever been measured one way is reported as one
> measurement, not as a property.

## Not a criticism of the method

Four retractions out of roughly thirty claims is a healthy rate, and #2 is a
genuine advance — it was found *because* somebody asked for the null condition,
which is the behaviour that produces the other three. The failure mode to guard
against is not making errors. It is making an error and not noticing.

> ## CORRECTION (2026-10-20) — it is five, and #5 changed the standing practice
>
> Five retractions out of roughly thirty claims. The rate is still healthy; the
> *cost* of #5 was not, because it was inherited rather than owned: one constant
> in one reader, copied into every reader, and it invalidated **every camera
> measurement the project had ever made** at once. The other four were wrong
> claims about the world. #5 was wrong claims about the *data*, which is
> worse, because a wrong claim about the world can be caught by a new
> measurement and a wrong claim about the data cannot.
>
> So the standing practice above gains a fourth line, and it is the one #5
> earned:
>
> > Every measurement names its condition, its interval, and its null.
> > Every verdict is able to fail, and is shown failing when it is built.
> > A number that has only ever been measured one way is reported as one
> > measurement, not as a property.
> > **Every constant that is inherited is re-derived from the data, by a
> > check that comes from outside it.**