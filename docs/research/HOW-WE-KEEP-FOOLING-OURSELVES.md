# How we keep fooling ourselves: four retractions, one shape

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