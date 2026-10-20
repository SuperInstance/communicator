# SRC FIX BRIEF — three instances of one bug class

You own `src/`. Casey has authorised this work and said: **do it the right way,
even if it takes more time.** Do not rush this into a patch.

Read these first, in order. Do not skip to editing:

- `docs/research/GATE-PEAK-COVERAGE.md`
- `docs/research/MIC-HARDWARE.md`
- `docs/research/NOTCH-AB.md`
- `docs/research/WATCHER-COST.md`

## The thesis you are fixing

Three defects look unrelated and are the same defect:

> **A statistic summarised over the wrong interval.**

1. `GATE_FIELD` / `_gate_peak` in `src/watcher.py` — documented as "peak over a
   segment", but the running value is read from the retained **buffer tail**.
   Measured understatement: **14.5×** (true max 0.7293 vs reported 0.0503), at
   every duration. Cause is `_trim`.

2. `_gate_peak` then takes a **max over the whole segment window**. On the
   120 Hz hum fixture the gate passes **exactly 1 frame in 937** (max 0.504,
   index 2 — a start-up transient), and that single frame clears an entire 2 s
   YIN window. The gate admits hum because of a start-up artefact.

3. The pitch tracker has **no periodicity test**. `docs/research/NOTCH-AB.md`
   establishes the mechanism: a notch *does* remove the line (49.4 dB) but
   **cannot** fix the collapse, because **YIN measures period, not frequency**.
   Deleting the fundamental leaves 300/450/600 Hz — the same period. The
   deliberately-losing `hpf60` control *did* lose (R=0.370); the
   no-information control held. This is a mechanism, not a tuning problem.

## The standard

The established discipline in this repo, and the reason Casey is spending time
on it:

- **A suite that cannot fail is not evidence.** Every test you add MUST be
  demonstrated failing before your fix and passing after. Paste both runs in
  your final report. If you cannot show a red state, the test proves nothing.
- **Sabotage and revert.** After each fix, deliberately break it again and
  confirm the test catches the break. A test that survives sabotage is wrong.
- **Measure before and after.** Not "it should be faster now" — the number.
- **Keep falsified claims visible.** If something you expected turns out false,
  that belongs in the report, not in the bin.
- Cite a length AND a fixture for every cost figure, per `WATCHER-COST.md`.

## Constraints

- **Work on your own branch in this worktree. Do not push.** I merge.
- Do not edit `docs/research/` or `experiments/` — those are my lane.
- Do not edit existing tests to make them pass. If an existing test encodes the
  buggy behaviour, that is a finding: report it, do not quietly rewrite it.
- Keep the existing public behaviour (`Event` shape, `GATE_SPEECH`, timings)
  unless a change is required by the fix. If you change one, justify it.

## What "done" looks like

For each of the three:

1. A named test, its failing output, its passing output, and a sabotage run.
2. The before/after measurement.
3. A note on anything you found that contradicts the brief above. If the brief
   is wrong, say so plainly — I would rather have that than a green suite.

**If you conclude one of the three should not be fixed as specified, stop and
say why.** That is a legitimate result and a useful one. Do not implement
something merely because it was asked for.

## Suggested order

3 (pitch periodicity) is the deepest and the one where the mechanism is already
established — but it is also the one most likely to change behaviour, so it goes
last, after the two interval fixes are proven. 1 and 2 share a code path and
should be designed together, because fixing 1 alone leaves 2 still letting hum
through.