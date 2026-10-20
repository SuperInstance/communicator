# Composing the two filters, and what still blocks the teacher

> ## CORRECTION (2026-10-20, overnight) — the table below was measured on TORN FRAMES, and the "clean instrument" is withdrawn
>
> Every number in §1 and §2 of this file was computed on frames that were
> **torn between two real moments**: every reader in the repo read the
> captures at a 23,040-byte stride when the files are 19,200, so 80% of
> every "frame" was rows 0..71 of one moment spliced to rows 72..95 of the
> next. The full account is `docs/research/NV12-STRIDE.md`.
>
> **A tear IS a horizontal band.** The "LED flicker banding" this file was
> written to remove was the tear, so the whole nuisance was an artefact of
> the reader. `experiments/combined_filter.py`, re-run at the correct
> stride, now says:
>
> > "THE FILTERS ARE NOT NEEDED, AND THE PREMISE IS WITHDRAWN. At the
> > correct stride the motion field is not saturated: the raw variant's
> > saturation is 0.5999 on condition A and 0.0154 on the empty capture,
> > against 0.9973 and 0.4173 before the stride fix. The composition of
> > row-band removal with a temporal median does not improve the speech
> > signal-to-noise ratio -- 1.410 raw against 1.389 for the most
> > aggressive variant -- because there is no longer a nuisance for either
> > of them to remove."
>
> **The corrected numbers, in the same order as the table below:**
>
> | variant | empty saturation | A flip rate | A saturation | A-vs-C gap | SNR | top-5% share |
> |---|---:|---:|---:|---:|---:|---:|
> | raw | 0.0154 | 0.0000883 | 0.5999 | 0.2262 | **1.410** | 42.8% |
> | row | 0.0154 | 0.0000864 | 0.6529 | 0.2492 | 1.411 | 39.8% |
> | median3 | 0.0073 | 0.0000759 | 0.5120 | 0.2144 | 1.352 | 42.1% |
> | row+median3 | 0.0073 | 0.0000759 | 0.6023 | 0.2355 | 1.386 | 39.0% |
> | row+median5 | 0.0000 | 0.0000602 | 0.6378 | 0.2190 | 1.389 | 38.1% |
>
> **What this withdraws from this file:**
>
>  - **"The composition is worth having", and the 21× / 28× figures.** Every
>    flip rate is now of order **1e-4**, so a "21× improvement" over a
>    1e-4 number is a ratio of two noises. The SNR is flat at **1.35–1.41**
>    across all five variants, and the *raw* variant is within **0.001** of
>    the best. There is nothing for the filters to buy.
>  - **§4's "The representation is solved ... That is a clean instrument."**
>    The representation was never the problem. There is no saturation to
>    clean.
>  - **§2's "SNR is flat at ~1.85"** becomes 1.35–1.41, and the
>    interpretation stands in strengthened form: cleaning the field does not
>    buy contrast, because there is no noise to clean.
>
> **What survives, and it is the more important half:** the teacher is
> blocked on **data**, not on the representation. Two sessions per class
> cannot tell "speaking" from "this recording".
>
> **AND THE RECOMMENDATION WAS CARRIED OUT, AND IT DID NOT MATTER.** This
> file's own verdict said: "The honest recommendation is to use the raw
> representation, keep the zero latency, and re-run the teacher on it."
> That re-run is now done — `docs/research/TEACHER-RAW-REPRESENTATION.md` —
> and it ran the teacher on all four representations. The answer: **no
> representation beats the teacher's own controls**, and the best of them
> (`row`) beats raw by **four of 48 test windows** (+0.0833). The
> representation is not the blocker. Neither is the filter. It is the
> number of independent sessions, and condition D.
>
> The original text below is kept visible, per the house rule that a
> falsified claim stays in the record.

**Status: the representation is now clean. The teacher is still blocked — and
the blocker is data, not modelling.**

Follows `TEACHER-PIPELINE.md`, which built the teacher pipeline and found that
no teacher could be validated on two labelled sessions per class.

---

## 1. The two filters are complementary, and I treated them as alternatives

They are not. One attacks the **spatial** structure of the nuisance, the other
its **temporal** structure:

- **Row-band removal** subtracts each row's median across the columns. The
  nuisance is horizontal bands, identical in every column, so the row median
  estimates it. No latency, no blur — it never combines frames.
- **Temporal median** rejects per-pixel jitter that moves frame to frame.
  Costs (k−1)/30 s and smears fast motion.

Composed, they should give row removal's zero latency plus the temporal
median's noise rejection. `experiments/combined_filter.py`.

| variant | empty saturation | occupied flip (A) | A-vs-C gap | SNR | localisation |
|---|---:|---:|---:|---:|---:|
| raw | 0.4173 | 0.0663 | 3.941 | 1.99 | 24.9% |
| row | 0.4884 | 0.0299 | 4.481 | 1.84 | 28.1% |
| median3 | 0.0391 | 0.0409 | 2.416 | 1.91 | 23.4% |
| **row+median3** | **0.0348** | **0.0014** | 1.806 | 1.83 | **35.7%** |
| row+median5 | **0.0000** | 0.00013 | 0.188 | 1.83 | 23.9% |

**Occupied flip rate: 0.0299 (row) and 0.0409 (median3) alone, 0.0014
composed** — a **21×** improvement over row removal alone and **28×** over the
temporal median alone. The two really are orthogonal, and the composition is
worth having.

**On the EMPTY room the composition buys much less.** The empty-room
saturation goes 0.0391 → 0.0348, which is **14.0×** better than row removal
alone but only **1.12×** better than the temporal median alone: **most of the
empty-room benefit comes from the median, not from the composition.**

> **CORRECTION (2026-10-19).** This table's headers were shifted by one
> column against the script's own output. `experiments/combined_filter.py:185`
> prints `empty flip | empty sat | occ flip | occ sat | A-vs-C gap | loc`, and
> the first column here was labelled "empty saturation" while carrying
> `A_flip` — the **occupied** flip rate. `empty_flip` is `null` for every
> variant. So the earlier "an empty-room saturation of 0.0014" was false: the
> real empty-room saturation for row+median3 is **0.0348 (3.48 %)**, and for
> row+median5 it is **0.0**. Likewise "a 21× improvement over **either**
> filter on its own" used the better of the two singles. Found by the notes
> audit — `NOTES-AUDIT.md` F4. The table and the two sentences it supported
> are corrected above; the numbers are unchanged, only the labels moved.

## 2. The filters do not improve the signal-to-noise ratio

SNR is flat at ~1.85 across every variant. The filters cut the noise floor
dramatically **and cut the signal proportionally**. That is the honest result:
this is a noise-limited problem, and cleaning the field does not buy contrast.

Two things do improve: the **localisation** of the speech signal (24.9% → 35.7%
in the top 5% of pixels for `row+median3`) and the raw drift (43.6 → **9.4
counts** after composition, against 32.1 for row removal alone). A cleaner,
better-localised field is exactly what a teacher model needs, even if the
underlying contrast is unchanged.

> **CAVEAT.** The 9.4 and 32.1 pair has no counterpart in
> `combined_filter_results.json` — that file carries no `drift` field, and
> `drift_structure_results.json` holds 43.686 (raw) and 31.312 (row std) for
> the occupied clip, neither of which is 32.1. These two numbers may come
> from a run that is not in the repository. Flagged as an unknown rather than
> a finding — `NOTES-AUDIT.md` §4.

*A bug worth recording: the first version of this measurement computed its
within-condition split by halving the 2-D MAP — top-of-image against
bottom-of-image — which is not a temporal split at all, and produced nonsense
SNR values. The split has to be in time.*

## 3. The teacher is still not validated on the cleaner representation

Regenerated the corpus with `row+median3` and re-ran the teacher:

| features | test acc | scrambled | blank | margin | LOO |
|---|---:|---:|---:|---:|---|
| summary | 1.000 | 0.792 | 0.667 | +0.208 | A 0.000, B 0.000 |
| map | 1.000 | 0.938 | 0.667 | +0.062 | A 0.000, B 1.000 |
| both | 1.000 | 0.979 | 0.667 | +0.021 | A 0.000, B 0.000 |
| **both, row+median3** | 1.000 | **1.000** | 0.667 | **+0.000** | A 0.000, B 0.000 |

> **CORRECTION (2026-10-19) — both LOO cells in this table were wrong, and one
> of them was wrong in a way worth recording.**
>
> The `both` row read **"A 0.000, C 0.650"**. There is **no session C** with a
> leave-one-session-out row: holding out C leaves only label 1 in the training
> set, so the fold is undefined. The experiment now reports that as a *skipped*
> fold with its reason, rather than dropping it silently — a cell can only be
> transcribed from a row that exists. The cell is **A 0.000, B 0.000**.
>
> The `both, row+median3` row read **"A 0.000, B 0.675"**. That 0.675 was real
> in a committed artefact (`experiments/teacher_conventional_speaking_both.json`
> at `7a4dff5`) but **it does not reproduce.** Re-running the committed script
> against the committed corpus reproduces every other column of that artefact
> exactly — test 1.000, scrambled 1.000, blank 0.667, margin +0.000 — and
> returns LOO **A 0.000, B 0.000**. Neither the code nor the data is the
> variable: the corpus rebuilds **byte for byte** from the captures
> (`seed_corpus_from_captures.py --filter row+median3` yields three
> SHA-256-identical `.npz` files), and a sweep of **1280 configurations** over
> the parameters the artefact never recorded (4 filter variants × 4 frame
> counts × 8 strides × 5 max-pers × 2 tasks), committed as
> `experiments/retraction_sweep.py` and
> `experiments/retraction_sweep_results.json`, never produces 0.675
> *together with* the artefact's other columns. The cell is now **A 0.000, B 0.000**, and **0.675 is
> withdrawn as a measurement of this corpus.** `NOTES-AUDIT.md` F3,
> `ARTIFACT-REPRODUCIBILITY.md`.
>
> The lesson is not "the teacher got worse". The other four feature sets
> reproduce bit-exactly, so the numbers this document's argument rests on are
> unaffected. The lesson is that **a committed number is not a result until it
> carries the identity of its inputs** — and that one did not.

**The composed representation makes the teacher worse, not better** — the
spatially-scrambled control now scores 1.000, exactly as the real input does.

That is diagnostic. Scrambling the pixels keeps every frame's energy and
destroys where the change was. If the model scores the same on both, it is
reading a **global energy statistic** and not the spatial pattern. And we
already measured that the global statistic is worthless here: total change is
3128 / 3209 / **3268** for speaking / speaking / **silent** — the same whether
he is talking or not, slightly *higher* when he is quiet.

So the model is reading a global statistic, and on this data the global
statistic does not carry the label. It scores 1.000 on the temporal split
because intra-session drift is systematic (the first half of a session moves
more than the second half), and it scores 0.000 on leave-one-session-out
because that drift is not shared between sessions.

## 4. What this establishes, and what it does not

**The representation is solved.** Row removal plus a 3-frame temporal median
gives an **empty-room saturation of 0.0348** and an **occupied flip rate of
0.0014**, with the speech signal 35.7% concentrated in the top 5% of pixels.
That is a clean instrument.

> **CORRECTION (2026-10-20, overnight) — WITHDRAWN.** "The representation is
> solved" and "that is a clean instrument" are both false, and the sentence
> above was measured on torn frames. At the correct stride the empty room
> shows an empty saturation of **0.0154 with no filter at all** (0.0073 with
> the temporal median, 0.0000 with row+median5), and the occupied flip rate
> is of order **1e-4** for every variant. There is no saturation, so there
> was never a dirty instrument to clean, and the SNR is flat at 1.35–1.41
> across all five variants — the raw variant is within 0.001 of the best.
> The teacher re-run on all four representations
> (`docs/research/TEACHER-RAW-REPRESENTATION.md`) confirms it: **no
> representation beats the teacher's own controls.** The blocker was never
> here.

**The teacher is blocked on data.** The gap between a temporal split and
leave-one-session-out is not a modelling failure that more features or a
bigger network will close. It is that **two sessions per class cannot tell
"speaking" from "this recording"**, and the only split that can tell them
apart needs several independent sessions.

`corpus_capture_win.py` now records **each condition three times, as separate
recordings**, with the repetition count baked into the plan:

```
A1 B1 C1 D1 F1 | A2 B2 C2 D2 F2 | A3 B3 C3 D3 F3 | FREE(5 min)
```

Three is the minimum that makes leave-one-session-out meaningful. More is
better. The free sessions are the corpus the teacher labels, and they are
worthless until a teacher passes the gate on the conditions.

**The teacher also needs features that cannot be reduced to an energy
statistic.** The `map` feature set should express "the mouth moved", and its
control should punish that. It currently does not beat its own scramble. A
teacher that will work probably needs either a non-linear model over the map
grid, or a feature that explicitly compares a small region against the rest of
the frame — which is what the localised signal is.

## 5. Limits

- Two labelled sessions per class, one speaker, one room, one lighting state.
  Every number here is a pipeline validation.
- The empty capture is later and darker than the occupied ones, so its
  saturation is not directly comparable.
- The flip rates in this document are computed on the *motion maps*, not on the
  Braille field, so they are not comparable to the 50.00% / 0.01% figures
  elsewhere. The two measurements answer different questions.
- The teacher has not been tried with a non-linear model, a larger feature
  set, or the region-versus-rest contrast. Those are untried, not refuted.

## Reproduce

```bash
/home/proart/projects/glasses/.venv/bin/python experiments/combined_filter.py --nf 400
/home/proart/projects/glasses/.venv/bin/python experiments/seed_corpus_from_captures.py \
  --filter row+median3 --tag rm3 --out /mnt/c/proart_capture/corpus_rm3
/home/proart/projects/glasses/.venv/bin/python experiments/teacher_conventional.py \
  --corpus /mnt/c/proart_capture/corpus_rm3 --task speaking --features both
```