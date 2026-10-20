# TEACHER-RAW-REPRESENTATION — the teacher on the raw representation, and the controls it still does not beat

**2026-10-20.** This is the measurement `experiments/combined_filter.py` left
open. It answers one question — *does the teacher do better on the RAW
representation than on the filtered ones, and does either beat the teacher's
own controls?* — and it answers it **no** on both halves: raw is not better
than the filtered representations, and **no** representation beats the
teacher's own controls. The reason nothing is usable is not the
representation.

The scripts are `experiments/teacher_raw_representation.py` (the driver) and
the three teacher scripts it runs unchanged: `experiments/teacher_conventional.py`,
`experiments/teacher_loo_control.py`, `experiments/teacher_static_control.py`.
The results are `experiments/teacher_raw_representation_results.json`. The
tests are `tests/test_teacher_raw_representation.py`.

> ## CORRECTION (2026-10-20) — the published teacher headline numbers are measured on the ROW representation
>
> Every teacher number in the repo — `docs/research/TEACHER-PIPELINE.md`,
> `docs/research/TEACHER-LOO-FOLDS.md`, `docs/research/COMBINED-FILTER.md`,
> and the committed `experiments/teacher_conventional_*.json` — was measured
> on `/mnt/c/proart_capture/corpus_v2`.
>
> **`corpus_v2` was seeded with `--filter row`.** That is recorded in the
> corpus's own sidecar (`"filter": "row"`, `"provenance": {"filter": "row"}`).
> And `experiments/combined_filter.py`, measured after the frame-stride fix,
> says, verbatim:
>
> > "THE FILTERS ARE NOT NEEDED, AND THE PREMISE IS WITHDRAWN. … The honest
> > recommendation is to use the raw representation, keep the zero latency,
> > and re-run the teacher on it."
>
> So the published headline numbers (speaking task: summary **+0.000**,
> region **+0.062**, map **+0.062**, both **+0.042**, region+map **+0.062**)
> describe a representation the repo's own composition experiment has
> **rejected**, and the clause "re-run the teacher on it" had never been
> carried out.
>
> This file carries that clause out. **The published numbers are not
> withdrawn** — they are reproduced here, exactly, as the `row` rows: the
> `row` corpus built in this sweep is **byte-for-byte identical** to
> `corpus_v2` (§1, leg c), so the published numbers and the `row` rows below
> are the same numbers.
>
> **What this file does not do:** it does not change the teacher, the
> features, the corpus format, or any published number. It adds the other
> three representations beside the one that was already there.

---

## 0. The question, and the rule that was fixed before the numbers

**Question.** (a) Does the teacher do better on the RAW representation
(`--filter none`) than on the filtered ones (`row`, `row+median3`,
`row+median5`)? (b) Does either beat the teacher's own controls?

**The decision rule was written before the run and is frozen.** It is a
literal constant, `PRE_REGISTERED_RULE`, in
`experiments/teacher_raw_representation.py`, and it is copied verbatim into
the results file together with that script's SHA-256. A reader can check
that the rule was not edited after the numbers were seen: if the rule text
in the results file differs from the constant in the script, or the script's
hash differs from the one the results file records, the run is not the run
the rule was written for.

The rule, in brief (the full text is in the results file and in the script):

- **Primary task: `speaking`.** Secondary and non-decisive: `facing`.
  Fixed in advance: the speaking task is the one with **both** a defined
  leave-one-session-out fold A and a defined fold B. On the facing labels
  (A = facing, B = turned, C = facing) holding out B leaves a single class
  in the training set, so fold B does not exist for that task.
- **The quantities.** `T(F,S)` is the teacher's own temporal-split margin,
  `acc_test(real) − max(acc_test(spatially scrambled), acc_test(blank))` —
  the quantity the published headline numbers are. `L(F,S,f)` is the
  leave-one-session-out margin over **that fold's own** scrambled control.
  `L_blank` is reported for every fold and is never a target: the blank
  control is 1.000 on every fold because the labels are session-level, so
  it is the degenerate null.
- **Rule 1 (the representation question).** A filter beats raw if it wins
  **at least 3 of the 5** feature sets on `T`. The headline is one of
  `RAW_IS_BETTER` (raw beats all three filters), `FILTERED_IS_BETTER` (some
  filter beats raw and raw beats none), `MIXED` (some filter beats raw and
  raw beats some) or `NO_DIFFERENCE` (nothing separates them). Each has a
  stated falsifier.
- **Rule 2 (the control question).** For the recommended representation, a
  feature set beats its controls iff `T ≥ 0.10` (the threshold
  `teacher_conventional.py` itself uses for `usable`) **and** `L > 0` on
  **both** defined folds.
- **The tie-break, fixed in advance.** If the verdict is MIXED or
  NO_DIFFERENCE, recommend **raw** — unless raw is not usable under Rule 2
  while some filter is, in which case recommend the usable one.

---

## 1. The sweep: four corpora that differ only by the filter

Four corpora, one per filter value, all seeded from the same three
captures with `experiments/seed_corpus_from_captures.py --max-frames 0`
(every frame) and the same defaults otherwise, so the filter is the only
thing that varies.

| filter | corpus | A.npz SHA-256 | B.npz SHA-256 | C.npz SHA-256 |
|---|---|---|---|---|
| none | `corpus_filter_none` | `2c05cb31dfd900c5…` | `61afc38c0bc0e7af…` | `8e3d55ebb7a46761…` |
| row | `corpus_filter_row` | `2d870ad1c942e655…` | `3c010a36f6e05832…` | `89cd606cc3762015…` |
| row+median3 | `corpus_filter_row+median3` | `d22823ab6c7d9813…` | `33569b6974ef57b6…` | `97342462ba346edb…` |
| row+median5 | `corpus_filter_row+median5` | `6ef8148e3ec353b3…` | `8d37047b95a45d91…` | `8fa04ee5de901b87…` |

(The full hashes are in the results file. The corpora live under
`/mnt/c/proart_capture/teacher_raw_representation/`.)

**Three legs of evidence that they differ only by the filter**, all recorded
in `corpus_equivalence` in the results file:

- **(a) Identity.** For every session, the four sidecars agree on the
  capture path, the capture's **SHA-256 and byte length**, the frame count
  (**900**), the seconds (**30.0**), the frame layout (`y_uv_halfres`), the
  stride (**19,200**), the map shape (48×80), the label fields and the
  fps. The capture hashes and byte lengths come from the capture files on
  disk, so this is a check against the world and not against the seeding
  script. Each sidecar also records `--filter <F>` and `max_frames: 0`.
- **(b) Content.** An independent recomputation — read the capture through
  the shared reader `src/nv12.py`, apply the filter, quantise — reproduces
  each corpus's stored motion maps **exactly**, byte for byte, for all
  twelve session/filter combinations. The reader *detects* the layout from
  the file's content, so this leg does not assume a stride either.
- **(c) Separation, and the external control.** All twelve npz hashes are
  distinct — no two corpora are the same data. And the `row` corpus here
  **reproduces `corpus_v2` byte for byte** (all three sessions). That is the
  check that ties this sweep to the repo's published numbers: the driver
  seeds exactly what the published numbers were measured on, and the only
  thing it then varies is the filter.

> **A stated limitation of leg (b).** The recomputation shares the filter
> *definition* with the seeding script, so it checks that each corpus holds
> what its label says, not that the filter definition is the right one. The
> identity of the *input* is checked externally, against the capture files'
> own hashes and byte lengths.

Nothing here reads, writes or moves `/mnt/c/proart_capture/corpus` (the torn
corpus) or `corpus_v2`. They are read-only in this lane.

---

## 2. Rule 1: which representation is better?

**The teacher does NOT do better on raw. The filtered representations win,
by a majority of the feature sets, on the split the rule was fixed on.**

The margin `T` is the teacher's own: `acc_test(real) − max(acc_test(spatially
scrambled), acc_test(blank))`. On the temporal split the blank control is
the base rate, 0.6667, for every cell, so `T` is the margin over the
spatially-scrambled control throughout. Each cell is written
**test | scrambled | blank → margin**.

### `none` (raw)

| feature set | test | scrambled | blank | margin `T` |
|---|---:|---:|---:|---:|
| summary | 0.9792 | 1.0000 | 0.6667 | **-0.0208** |
| map | 1.0000 | 0.9583 | 0.6667 | **+0.0417** |
| both | 1.0000 | 1.0000 | 0.6667 | **+0.0000** |
| region | 0.8542 | 0.8750 | 0.6667 | **-0.0208** |
| region+map | 1.0000 | 0.9583 | 0.6667 | **+0.0417** |

### `row` — this is `corpus_v2`, the published corpus

| feature set | test | scrambled | blank | margin `T` |
|---|---:|---:|---:|---:|
| summary | 0.9583 | 0.9583 | 0.6667 | **+0.0000** |
| map | 1.0000 | 0.9375 | 0.6667 | **+0.0625** |
| both | 1.0000 | 0.9583 | 0.6667 | **+0.0417** |
| region | 0.8958 | 0.8333 | 0.6667 | **+0.0625** |
| region+map | 1.0000 | 0.9375 | 0.6667 | **+0.0625** |

### `row+median3`

| feature set | test | scrambled | blank | margin `T` |
|---|---:|---:|---:|---:|
| summary | 0.9375 | 0.9167 | 0.6667 | **+0.0208** |
| map | 0.9583 | 0.9167 | 0.6667 | **+0.0417** |
| both | 1.0000 | 0.9375 | 0.6667 | **+0.0625** |
| region | 0.8542 | 0.8333 | 0.6667 | **+0.0208** |
| region+map | 0.9792 | 0.9583 | 0.6667 | **+0.0208** |

### `row+median5`

| feature set | test | scrambled | blank | margin `T` |
|---|---:|---:|---:|---:|
| summary | 0.9375 | 0.9792 | 0.6667 | **-0.0417** |
| map | 0.9583 | 0.8542 | 0.6667 | **+0.1042** |
| both | 1.0000 | 0.9375 | 0.6667 | **+0.0625** |
| region | 0.7917 | 0.7292 | 0.6667 | **+0.0625** |
| region+map | 0.9792 | 0.8958 | 0.6667 | **+0.0833** |

### The pairwise comparison, and the verdict

| filter | beats raw on | raw beats it on | undecided |
|---|---:|---:|---:|
| **row** | **5** of 5 | 0 | 0 |
| **row+median3** | **4** of 5 | 1 | 0 |
| **row+median5** | **4** of 5 | 1 | 0 |

> **CORRECTION (2026-10-20) — tie handling.** `row+median3`'s `map` cell
> and `row+median5`'s `both` cell are exact ties (both are 2/48 = 0.0417),
> but two mathematically equal margins can differ by ~1e-16 in floating
> point, and the literal reading of the rule ("`T(F,S) > T(none,S)`") turns
> one of them into a win. Under a tie tolerance of 1e-9 the counts become
> `row` 5, `row+median3` **3**, `row+median5` 4 — **the headline and the
> recommended representation are the same either way**, so the tie handling
> is not load-bearing. Both counts are in the results file
> (`verdict.rule_1_representation.tie_sensitivity`).

> **VERDICT (Rule 1): `FILTERED_IS_BETTER`, best filter `row`.** All three
> filtered representations beat raw on at least three of the five feature
> sets; raw beats **no** filter on three of five. The rule's falsifier for
> `FILTERED_IS_BETTER` — raw beats every filter — did not fire.

### How big is that difference?

**At most four test windows out of forty-eight.** The test split is 48
windows, so one window is 1/48 = 0.0208. The largest gap between any two
representations anywhere in the table above is **+0.0833 — four windows**
(`row` vs `none` on `region`). The `map` cell that reaches +0.1042 is one
window above the +0.0625 that `row` reaches on the same feature set.

The rule counts a feature set as a win whatever the size of the difference,
so this number is reported **beside** the verdict, not inside it. The
honest reading of Rule 1 is: **the filter moves the temporal margin by one
to four windows, and it moves it towards the filtered representations.**

### The leave-one-session-out folds agree, and are the better evidence

The temporal split is the split the teacher's headline uses; the
leave-one-session-out folds are the split that can tell "speaking" from
"this recording", so they are reported too. Real accuracy on the held-out
session, folds A and B (40 windows each; fold A is **not identified** —
its training set confounds pose with speech — and fold B **is**):

| feature set | none | row | row+median3 | row+median5 |
|---|---|---|---|---|
| summary | 0.800 / 0.025 | **1.000 / 0.000** | **1.000 / 0.000** | 0.925 / 0.150 |
| map | 0.125 / 0.200 | **0.150 / 0.375** | **0.150 / 0.350** | **0.200 / 0.225** |
| both | 0.575 / 0.025 | **0.950 / 0.100** | **0.875 / 0.250** | 0.375 / 0.200 |
| region | 1.000 / 0.000 | 1.000 / 0.000 | 0.975 / 0.150 | 0.725 / 0.000 |
| region+map | 0.075 / 0.075 | **0.150 / 0.275** | **0.175 / 0.275** | **0.225 / 0.100** |

Ten cells (five feature sets × two folds). Counting the cells the filtered
representation wins on real accuracy:

| filter | filter wins | raw wins | ties |
|---|---:|---:|---:|
| row | **7** | 1 | 2 |
| row+median3 | **8** | 2 | 0 |
| row+median5 | **7** | 2 | 1 |

**The folds agree with the temporal split: raw is not better, it is
slightly worse on the held-out sessions too.**

### The secondary task disagrees, and that is the most informative line here

The rule made `facing` secondary and non-decisive. Run anyway, the facing
task's temporal margins give the **opposite** verdict — `RAW_IS_BETTER`,
with raw beating all three filters:

| feature set | none | row | row+median3 | row+median5 |
|---|---:|---:|---:|---:|
| summary | +0.0625 | -0.0208 | +0.0208 | -0.0625 |
| map | +0.0417 | +0.0000 | +0.0625 | +0.1250 |
| both | +0.0208 | +0.0208 | +0.0000 | +0.0208 |
| region | +0.0833 | +0.0625 | +0.0208 | -0.0833 |
| region+map | +0.0208 | -0.0208 | +0.0208 | +0.1250 |

**A representation effect that flips sign between the two tasks is not a
property of the representations.** The facing task is the one with no
signal in this data (`seed_corpus_from_captures.py`: "the 'facing' question
has no signal in this data and any score on it is noise"), the one whose
fold B does not exist, and the one whose swings are **larger** than the
speaking task's (up to 0.1667, eight windows) while meaning less. This is
the same shape as every other retraction in `HOW-WE-KEEP-FOOLING-OURSELVES.md`:
a quantity summarised over one split against a quantity summarised over
another.

---

## 3. Rule 2: does any of them beat the teacher's own controls?

**No. Not one feature set of any representation beats the teacher's own
controls on both splits at once.** This is the answer to the second half of
the question, and it is the same answer for all four representations.

`L(F,S,f)` is the held-out accuracy minus **that fold's own** spatially
scrambled control. The **blank** control is 1.000 on every fold of every
representation, because the labels are session-level and a constant
predictor is exactly right on a pure held-out session; it is the degenerate
null and is never a target.

| feature set | none A / B | row A / B | row+median3 A / B | row+median5 A / B |
|---|---:|---:|---:|---:|
| summary | -0.200 / +0.025 | +0.000 / +0.000 | +0.000 / -0.050 | -0.075 / -0.050 |
| map | -0.275 / +0.175 | -0.300 / +0.275 | -0.400 / +0.225 | -0.325 / +0.075 |
| both | -0.300 / +0.025 | -0.025 / +0.100 | -0.100 / +0.175 | -0.350 / +0.000 |
| region | +0.000 / +0.000 | +0.000 / +0.000 | -0.025 / +0.125 | -0.075 / +0.000 |
| region+map | -0.325 / +0.075 | -0.300 / +0.250 | -0.500 / +0.175 | -0.300 / +0.000 |

**No cell is positive on fold A at all.** The largest fold-A margin anywhere
in the sweep is exactly **+0.000**, and the four cells that reach it
(`none`/region, `row`/summary, `row`/region, `row+median3`/summary) are at
or below zero on fold B as well (-0.050 for the last). Every cell that is
positive on fold B is negative on fold A. The only feature set anywhere in
the sweep that clears the temporal threshold `T ≥ 0.10` is `row+median5`'s
`map` (+0.1042) — and its leave-one-session-out margins are **-0.325** and
**+0.075**, so it fails Rule 2 on both folds.

| representation | usable under Rule 2 | usable feature sets |
|---|---|---|
| none | **NO** | — |
| row | **NO** | — |
| row+median3 | **NO** | — |
| row+median5 | **NO** | — |

**So: the raw representation is not better, and the filtered representations
are not usable either. The representation is not what is blocking the
teacher.**

---

## 4. The static control: the raw representation does not rescue the teacher

The STATIC-template control (`teacher_static_control.py`) replaces every
frame with the session's temporal mean map, so there is no motion, no order
and no change anywhere in the input — only a picture of the camera.

**It scores 1.000 on fold A for every feature set of every representation,
including the raw one.** All twenty cells.

The `unscaled` control (every session divided by the same constant, so the
between-session ratios are untouched and only the absolute scale moves)
**reproduces the real variant exactly** in all forty cells across the four
corpora. The teacher's numbers are not a scale artefact, on any of them.

And the picture the teacher reads does not change with the filter:

| quantity | none | row | row+median3 | row+median5 |
|---|---:|---:|---:|---:|
| correlation between the two takes' average motion map | 0.7058 | 0.7046 | 0.7619 | 0.8064 |
| fraction of their difference explained by one global scale | R² = 0.4900 | R² = 0.4724 | R² = 0.5274 | R² = 0.5981 |
| empty room's mean-map energy relative to A | 0.000333674 | 0.000254897 | 0.000336765 | 0.000412559 |

The raw and the row-filtered pictures correlate at 0.7058 and 0.7046 — the
same picture to within 0.002. **The filter does not change what the teacher
is reading.** It is still a session template, and a session template is
still not a speaking detector.

---

## 5. What this establishes, and what it does not

**What it establishes.**

1. **The answer to the open question is no, on both halves.** Raw is not
   better than the filtered representations: on the pre-registered primary
   (the speaking task's temporal margin) all three filters beat raw on a
   majority of the five feature sets, and on the leave-one-session-out folds
   the filtered representations win 7, 8 and 7 of the ten cells.
2. **The pre-registered verdict is `FILTERED_IS_BETTER`, best filter `row`** —
   which is the representation the repo already uses, and the one
   `corpus_v2` holds. `combined_filter.py`'s "use the raw representation"
   recommendation is **not** supported by the teacher, and this file does not
   adopt it.
3. **But the difference is at most four windows out of forty-eight**, and
   the secondary task gives the opposite verdict, so the effect is not a
   property of the representations either.
4. **No representation beats the teacher's own controls** — on any feature
   set, on any fold, in any of the four corpora. The blank control is
   1.000 everywhere and the static control is 1.000 on fold A everywhere.
   **The teacher is still blocked, and the representation is not the
   blocker.**
5. **The published numbers stand, with their label corrected.** They are the
   `row` rows, on a corpus this sweep reproduces byte for byte.

**What it does not establish.**

- **Not that `row` is the right representation to ship.** It wins a
  3-of-5 rule by one to four windows of forty-eight, and the facing task
  disagrees. The rule says what it says; the effect size says how little
  there is to have.
- **Not that raw is useless.** It is marginally worse, not worse in kind.
- **Not anything about a speaking detector.** On every representation the
  teacher's real features lose to a constant predictor on a pure held-out
  session. The blocker is the data and the label design, exactly as
  `TEACHER-LOO-FOLDS.md` §4 says.
- **Not generalisation.** Two sessions per class, one speaker, one room, one
  lighting state, one seed, one split, no error bars. The facing task has no
  signal in this data at all.
- **Not a claim about the filter's physical value.** `combined_filter.py`
  measured the motion *field*; this file measures the *teacher*. The field's
  statistics (SNR, saturation, localisation) are unchanged by this file.

---

## 6. Corrections to this file's own verdict

Both are recorded in the results file (`corrections`) rather than silently
overwritten.

> **CORRECTION (2026-10-20) — the first implementation of the rule was
> stricter than the rule, and the verdict moved.** The first run returned
> `MIXED`. The cause was in the code, not the rule: the implementation read
> *"raw beats none of the three"* as *"raw wins **zero** feature sets
> against every filter"*, which is stricter than the rule text, which says
> *"raw beats none of the three **by the 3-of-5 rule**"*. With the
> implementation matching the rule text the verdict is
> `FILTERED_IS_BETTER`.
>
> **The rule text was NOT changed.** The SHA-256 of `PRE_REGISTERED_RULE` is
> the same before and after the correction, and the rule text in the
> results file is byte-identical to the constant in the script. A verdict
> computed by code that does not implement its own rule is not the rule's
> verdict, whichever way it falls — which is the argument for having the
> rule as a constant in the first place.

> **CORRECTION (2026-10-20) — the tie handling, recorded because it is
> close to being load-bearing.** The margins are differences of accuracies on
> 48 test windows, so two mathematically equal margins can differ by ~1e-16
> in floating point, and the literal rule turns one such pair into a win.
> The verdict is the same under a tie tolerance of 1e-9 and the same
> recommended representation, so the tie handling is not load-bearing — but
> that is a result, not an assumption, and both counts are in the file.

---

## Reproducing this

```bash
/home/proart/projects/glasses/.venv/bin/python experiments/teacher_raw_representation.py
/home/proart/projects/glasses/.venv/bin/python -m pytest tests/test_teacher_raw_representation.py -q
```

The driver seeds the four corpora, re-checks them against the captures, runs
the three teacher scripts on each (48 subprocess calls in all), and writes
`experiments/teacher_raw_representation_results.json`. It takes about 90
seconds, uses only numpy and the repo's own `src/`, and is deterministic:
two independent runs produced byte-identical tables. The four corpora are
written under `/mnt/c/proart_capture/teacher_raw_representation/`; `corpus_v2`
and the torn `corpus` are read, never written.
