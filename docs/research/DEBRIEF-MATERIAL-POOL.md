# The director was being starved, and the fix for it did not help

**Status: a defect, fixed and measured — and a partial retraction of what
the fix was supposed to prove.**

Follows `DEBRIEF-REAUDIT.md`, which re-audited the answer key the LLM
director is scored against. That lane found the mechanism behind the
director's worst verdicts, and this lane fixes it and then measures what
the fix is worth.

---

## 1. The defect

`experiments/debrief.py` hands the director the repository's research
notes. The pool is supposed to give every note an **equal share** of the
character budget and to keep each note's **head and its tail** — because
the dated correction blocks, the ones that say which of this
repository's numbers were withdrawn, live at both ends of these notes.

It did not do that. The keep-set was built correctly:

```python
keep = set(range(head_lines))                      # the head
keep |= set(range(len(lines) - head_lines, len(lines)))   # the tail
for i, ln in enumerate(lines):
    if any(ch.isdigit() for ch in ln):
        keep.add(i)                                # every number
keep = sorted(keep)
```

and then the assembled text was cut **from the front**:

```python
text = "\n".join(out)
if len(text) > per:
    text = text[:per] + "\n[... truncated ...]"     # <-- the tail goes here
```

The design was right and one line of code undid it. Every note whose
assembled slice exceeded its share lost its tail, and the retraction
notices are in the tails.

**Measured, on the real notes, at the committed 140,000-character budget:**

| | notes keeping their tail | notes keeping their head |
|---|---:|---:|
| **before** | **2 of 50** | 50 of 50 |
| **after** | **50 of 50** | 50 of 50 |

The "before" number is reproduced by `experiments/debrief_material_pool.py`,
which re-implements the old assembly and counts its tails, so it is a
measurement this repository makes rather than a number it inherited.

## 2. The fix

The budget is now spent by dropping lines from the **middle** of the
keep-set, never from the end; and if even the head-and-tail slice is too
big, each end is cut to its own half of the budget so that **both**
survive.

`collect_materials_stats()` reports what the pool actually kept —
`notes_total`, `notes_kept_whole`, `notes_truncated`,
`notes_keeping_head`, `notes_keeping_tail`, `notes_keeping_both_ends` —
so the claim is a number in an artefact rather than a promise in a
docstring. `tests/test_debrief_material_pool.py` (8 tests) pins it,
including a test that **reproduces the old front-truncation and asserts
that it loses the tails**, so the fix cannot be reverted by someone who
"simplifies" it back.

## 3. The control that says the fix is inert where it must be inert

The audit runs in two modes: `blind` (the claims alone) and `context`
(the claims plus the evidence). The pool is only used in `context` mode, so
a correct fix must leave `blind` untouched.

| mode | model | headline before | headline after | delta |
|---|---|---:|---:|---:|
| blind | llama-3.3-70b | −0.778 | −0.778 | **+0.000** |
| blind | gpt-oss-120b | +0.222 | +0.222 | **+0.000** |
| blind | qwen3-30b | +0.444 | *unparseable* | — |

The three blind cells are identical. The fix changed what it was
supposed to change and nothing else.

## 4. The effect on the director: it does not help, and the reason matters

Both pools were run against **one corpus** — the same notes, the same
moment — 3 repeats per model per pool, corrected key throughout, at the
budget each model's window allows.

| | old pool (tails dropped) | new pool (tails kept) |
|---|---|---|
| successful runs | 7 (2 unparseable) | 8 (1 unparseable) |
| **specificity** | **0.000 in 7 of 7** | **0.000 in 8 of 8** |
| sensitivity spread | **1.000** (0.000 … 1.000) | 0.556 (0.333 … 0.889) |
| headline median | −0.778 | −0.611 |
| headline spread | 1.000 | 0.556 |
| characters of evidence, llama-3.3-70b | 140,000 | **40,000** |
| characters of evidence, qwen3-30b | 140,000 | **60,000** |

**THREE THINGS COME OUT OF THAT, AND ONLY ONE OF THEM IS THE HEADLINE.**

**1. THE ROBUST RESULT: THE DIRECTOR ACCUSES THE ONE TRUE CLAIM, EVERY
SINGLE TIME.** Specificity is **0.000 in 15 of 15 successful runs**, in
both pools, at every budget. The one true claim in the audit set is the
acoustic noise floor (−73.5 dBFS, 119 transients, zero RT60), and the
director rejects it in every run — while its sensitivity, the rate at
which it catches the false ones, moves around by 1.000 across identical
prompts.

**That is the failure, and it is not a starvation problem.** A director
that always accuses the one true claim has a floor of −1.000 on the
headline no matter how much evidence it is given, and no budget will lift
it.

**2. THE POOL EFFECT CANNOT BE SEPARATED FROM THE ENDPOINT'S VARIANCE.**
gpt-oss-120b's sensitivity on the **old** pool was **0.889, 0.333 and
1.000** on three **identical** prompts. Three repeats cannot separate a
pool effect from a 0.667 spread in one cell, and the new pool's median is
better by 0.167 — which is inside the old pool's own range. **The honest
statement is that this experiment did not show the fix helping, and could
not have shown it helping at this sample size.**

**3. THE FINDING WITH A CONSEQUENCE: THE FIX CHANGES HOW MUCH EVIDENCE A
MODEL ACTUALLY GETS.** The tails are dense tables of numbers, and they
tokenise more densely than the head slices. The same *character* budget
therefore costs more *tokens*, and the fit ladder shrinks further:
**llama-3.3-70b went from 140,000 characters of evidence to 40,000, and
qwen3-30b from 140,000 to 60,000.** llama's headline moved from −0.778 to
−0.667 across that 3.5× change in evidence, which is a reminder that a
character budget is not an evidence budget.

**Any comparison of "the same evidence" has to be made on TOKENS.**

## 5. What this withdraws, and what it does not

> **CORRECTION (2026-10-20, overnight).** `docs/research/LLM-DIRECTOR.md`
> §5 recorded, from the 2026-10-19 run: *"The fix was mechanical and is
> worth stating because it is the actual lesson"* — and the actual lesson
> it pointed at was that the director's quality is a function of the
> evidence you hand it, so delivering the retractions should help.
>
> **The budget half of that survives.** The materials sweep still shows a
> strong effect of the *size* of the evidence: median headline −0.444 at
> 20,000 characters, −0.222 at 60,000, 0.000 at 140,000, +1.000 at 240,000.
>
> **The pool half does not.** The pool's internal *completeness* is a
> different axis, and on this evidence fixing it does not help. What is now
> the best-supported thing in the instrument is the mechanism the note
> described: a starved evidence base produces a confident, specific, WRONG
> verdict — and the director's verdict on the one true claim is wrong in
> every single run, at every budget, with the retractions delivered and
> with them starved.
>
> **And the "+1.000 at 240,000 characters" arm is NOT reproducible.** A
> re-run of that arm against the current corpus gives **−0.111 three times
> out of three** for gpt-oss-120b, where the sweep recorded +0.889 twice.
> The corpus changed in between — four new research notes and six
> corrected ones — so the arm is a property of a *particular corpus*, not
> of the model or the budget. **It must not be quoted as a capability of
> the director.** (llama-3.3-70b and qwen3-30b cannot be given 240,000
> characters at all: HTTP 413, payload too large.)

## 6. What this does not establish

* It does not establish that the fixed pool is better for the director.
  It establishes that the fixed pool is **complete**, that the defect was
  real, and that the effect of fixing it is **not separable** from the
  endpoint's variance at this sample size.
* It does not establish a mechanism for *why* the director accuses the
  true claim. The candidate mechanism — that a dated correction block
  **restates the withdrawn numbers in order to withdraw them**, so a note
  containing "43.3 counts" reads as evidence *for* a claim that says 43.3
  — is plausible and consistent with what the lane observed, and it has
  **not** been tested. It is recorded here as a hypothesis to be measured,
  not as a finding.
* It does not change the re-audited answer key, the 9-false/1-true set, or
  the verdict in `DEBRIEF-REAUDIT.md`.

## Reproduce

```bash
/home/proart/projects/glasses/.venv/bin/python experiments/debrief_material_pool.py
/home/proart/projects/glasses/.venv/bin/python -m pytest tests/test_debrief_material_pool.py -q
```

The A/B itself is cloud calls against the same corpus with both pools; the
raw runs are in `.scratch/debrief_reaudit/ab_pools_samecorpus.json` and
the summary is in `experiments/debrief_material_pool_results.json`.

A note on the old pool's code, for anyone repeating this: it is
`git show 4b74128:experiments/debrief.py`, and the A/B imports it
side-by-side with the current one so that both pools see the same notes.
