# The teacher: not an LLM, and not yet a model at all

**Status: the pipeline is built and the controls are working. No teacher is
validated, and the reason is a data requirement, not a modelling failure.**

Casey's question: *"could we simply record me and use a traditional model to
analyze and be the teacher for the system?"*

Short answer: **yes, record him — that is the highest-leverage thing available
right now. But the teacher should not be an LLM, and with the data that exists
today no teacher can be validated at all.** Both halves of that were measured,
not assumed.

---

## 1. The LLM teacher is dead, and it was measured

`experiments/teacher_bench.py` renders the motion field as characters and asks
a local model a question with a known answer. The question was the *speaking*
one, which has real signal: facing-and-speaking differs from facing-and-silent
by 4.914 counts against a within-condition split of **2.878**, with 79% of the
excess localised.

> **CORRECTION (2026-10-19).** The split was published as **1.917**, giving
> 2.6×. The contrast is **A-vs-C**, so the matched control is A's own
> within-split, `within_A_facing_speaking.mean_abs_diff` = **2.878**; 1.917 is
> condition B's split. The factor is **1.71×** (4.914 / 2.878). Settled by
> `experiments/pose_test_results.json`; `NOTES-AUDIT.md` F1. The same wrong
> pair is still in the `speech_verdict` and `verdict` strings inside that JSON
> and **still needs fixing in code** — the result file is not editable from
> the notes.

Every model, with the real input, the spatially-shuffled input, and a blank:

| model | real | shuffled | blank | margin |
|---|---:|---:|---:|---:|
| qwen3:0.6b | 0.333 | 0.333 | 0.333 | **0.0** |
| llama3.2:1b | 0.333 | 0.333 | 0.333 | **0.0** |
| gemma2:2b | 0.333 | 0.333 | 0.333 | **0.0** |
| qwen2.5-coder:7b | — | — | — | refuses the input |
| tev1:latest | — | — | — | HTTP 400 on the request |

A margin of exactly 0.0 is the bench's definition of a model that knows
nothing: it answers the same thing to the data, to a scramble of the data, and
to nothing at all. **The small text models cannot parse a dense character-art
grid as a picture.** The trace rendering (per-frame statistics as text) did not
help — same constant answers.

One real bug was found on the way, and it is worth recording because it looked
exactly like the failure it was hiding: qwen3 is a *reasoning* model, ollama
puts its reasoning in a separate `thinking` field, and with a small
`num_predict` the entire budget went into the preamble so every `response`
came back **empty**. In a summary table, "always empty" and "always the same
word" are identical, and both read as "model failed". The raw text is now kept
and printed.

## 2. So: a conventional model, which is what "traditional model" means

`experiments/teacher_conventional.py`. Logistic regression (numpy only — the
repo is deliberately lean and sklearn is not a dependency) over features of the
motion field: per-frame total change, active fraction, centroid, spread, their
temporal derivatives, and optionally the window's mean motion map as a coarse
grid.

The pipeline runs end to end and the controls are wired up:

- **Spatially scrambled control.** Every frame's pixels are rolled, which keeps
  each frame's energy and destroys where the change was. (Permuting *frames* is
  not a valid control for map features — the mean motion map is invariant to
  frame order, so that control scored 0.979 against a real-input 1.000 and
  told us nothing. The first version of this file had that bug.)
- **Blank control.** Zero motion everywhere.
- **Leave-one-session-out.** Train on all but one session, test on the held-out
  one.

## 3. The result, and it is a data verdict

| features | test acc | scrambled | blank | margin | LOO |
|---|---:|---:|---:|---:|---|
| summary | 1.000 | 0.792 | 0.667 | +0.208 | **A 0.000, B 0.000** |
| map | 1.000 | 0.938 | 0.667 | +0.062 | A 0.000, B 1.000 |
| both | 1.000 | 0.979 | 0.667 | +0.021 | A 0.000, B 0.000 |

The temporal split says **1.000**. Leave-one-session-out says **0.000** on
held-out A for every feature set, and **0.000 / 1.000 / 0.0 / 0.200** on
held-out B for summary / map / both / region+map.

> **CORRECTION (2026-10-19).** This said *"Leave-one-session-out says **0.000**
> on the held-out session for every feature set"*, and the `both` row read
> "A 0.000, **C** 0.650". Held-out accuracy per feature set, from
> `experiments/teacher_conventional_speaking{,_summary,_region,_map,_both,_region+map}.json`:
>
> | feature set | LOO A | LOO B |
> |---|---:|---:|
> | summary | 0.0 | 0.0 |
> | region | 0.0 | 0.0 |
> | map | 0.0 | **1.000** |
> | both | 0.0 | 0.0 |
> | region+map | 0.0 | **0.200** |
>
> **Two of five are not 0.000.** The notes audit settled this against a
> committed `speaking_both.json` holding B **0.675**; the experiment has since
> been re-run and that file now holds B **0.0**, with the scrambled control
> **0.979** and the margin **0.0208** — which is what the `both` row above
> already printed for its other columns.
>
> And the **0.650 has no source**: no result
> file contains a held-out session "C" for this task, and 0.650 matches neither
> candidate source (B 0.675 in `speaking_both.json`, B 0.200 in
> `speaking_region+map.json`). The cell is now B 0.000.
> `NOTES-AUDIT.md` F3.
>
> **FURTHER CORRECTION (2026-10-19) — the 0.675 is withdrawn too.** The block
> above still offers B 0.675 as a "candidate source" for the 0.650, which
> treats it as a measurement. It is not one. The 0.675 **does not reproduce**:
> re-running the committed script against the committed corpus reproduces every
> other column of that artefact exactly — test 1.000, scrambled 1.000, blank
> 0.667, margin +0.000 — and returns LOO **A 0.000, B 0.000**. Neither the code
> nor the data is the variable: the corpus rebuilds byte for byte from the
> captures, and a sweep of 1280 configurations over the parameters the artefact
> never recorded (4 filter variants × 4 frame counts × 8 strides × 5
> max-pers × 2 tasks), committed as `experiments/retraction_sweep.py` and
> `experiments/retraction_sweep_results.json`, never produces 0.675
> *together with* the artefact's other columns.
>
> The other four feature sets' artefacts reproduce bit-exactly, so the
> two-of-five verdict above stands on numbers that do reproduce — **map** B
> **1.000** and **region+map** B **0.200**. What was wrong was the *provenance*
> of one artefact, not the verdict. Result files now carry the identity of the
> corpus they were measured on, so a number that cannot be traced cannot be
> quoted again. See `COMBINED-FILTER.md`, `ARTIFACT-REPRODUCIBILITY.md`.

> **CORRECTION (2026-10-19, `TEACHER-LOO-FOLDS.md`).** The paragraph below
> is right about the temporal split and **wrong about the leave-one-session-out
> split**, in a way that matters.
>
> 1. **"its scrambled control is 0.938" is the TEMPORAL split's control, not
>    the LOO fold's own.** Measured on the same fold, the LOO's scrambled
>    control is **0.000**, so the margin there is +1.000 — the two numbers
>    were being compared across two different splits.
> 2. **But the LOO fold cannot carry the claim either, and the reason is
>    worse than a wrong control.** Every window of a held-out session carries
>    the same label, so the fold is a one-bit statistic, and the **blank
>    control — a constant predictor — scores 1.000 on every fold of every
>    feature set**. The `map` feature set's 1.000 on held-out B is worth
>    exactly what a constant predictor's 1.000 is worth. The LOO rows have no
>    control beside them, and the metric has no room to measure anything.
> 3. **And the `map` features turn out not to need the motion at all.**
>    Replacing every motion frame with the session's temporal mean map — no
>    motion, no order, no change — scores **1.000 on BOTH folds** for `map`,
>    `both` and `region+map`. The discriminative feature is a *static spatial
>    template of the session*, and adding the temporal information makes it
>    worse (fold A goes 1.000 → 0.000).
> 4. **The video does not follow the audio.** Cross-correlating the repo's
>    own VAD speech envelope against the video's motion envelope over a ±15 s
>    offset search gives +0.078 (A) and +0.025 (B) against a scrambled-audio
>    null of +0.061. So frame-level labels cannot be recovered from these
>    captures either.
>
> **The consequence for the recording plan: the 63-sessions-per-class figure
> is not the blocker.** That number is a precision requirement on an estimate;
> recording it does not make the fold gradated, does not give the fold a
> control, and does not make the motion follow the speech. The blocker is that
> **the task as constructed cannot be measured**, and the fix is a change of
> task, not a change of sample size. The full argument, with every control,
> is in `TEACHER-LOO-FOLDS.md`.

That contradiction is the finding. The temporal split puts train and test in
the same twenty-second window, minutes apart, and a model can score 1.000 by
learning *what this room looks like at 19:52*. The leave-one-session-out is
the only split here that can tell "speaking" from "this recording", and on the
one feature set with a real margin over its controls — `summary` — it says the
model learned the recording: **0.000 on both held-out sessions**. (`map` does
score 1.000 on held-out B, but its scrambled control is 0.938, so it is not
beating its own control either.)

**No teacher is validated. The controls are working exactly as intended** — this
is what a gate is for.

## 4. Why, underneath: the motion field is saturated

The features say why, and this is more useful than the accuracy numbers.

| session | label | total change | active fraction | change centroid x |
|---|---|---:|---:|---:|
| A | speaking | 3127.9 | 0.1013 | 0.546 |
| B | speaking | 3208.7 | 0.1295 | 0.530 |
| C | **silent** | **3268.5** | 0.1206 | 0.599 |

**The total motion energy is essentially the same whether he is speaking or
not — and it is slightly HIGHER when he is silent.** 99.8% of pixels change by
more than 1% of maximum in every frame, in all three sessions, even after
row-band removal.

So "how much changed" is not the signal, and the interesting difference — the
localised one that `ROW-BAND-REMOVAL.md` measured as 79% concentrated in the
top 5% of pixels — is a small excess on top of a saturated field. The change
centroid is the only feature that even hints at the head turn (A 0.546 facing,
C 0.599 facing, B 0.530 turned), and with one session per condition that is
indistinguishable from a session artifact.

## 5. What this means for the plan

**Record him. That part of the plan is right and the pipeline is ready.**
`experiments/corpus_capture_win.py` captures, removes the banding, computes the
motion maps, checks that each take actually achieved its condition (voice
present, voice absent, drift under 5), and writes a compact session — about
7 MB per minute instead of ~400 MB of raw video.

**But the binding constraint is the number of SESSIONS, not the number of
windows.** Windows from one session are not independent samples, and a temporal
split of a twenty-second window is not a test. Leave-one-session-out needs at
least **three to four independent sessions per condition** before it can
distinguish a teacher from a memoriser.

So the capture protocol becomes:

| condition | pose | speech | sessions needed |
|---|---|---|---|
| A | facing | speaking | ≥3 |
| B | turned | speaking | ≥3 |
| C | facing | silent | ≥3 |
| D | turned | silent | ≥3 |
| F | free | free | as much as you like |

Three sessions per condition, not one long one — start a new recording for
each, and vary the time and lighting a little if you can. The free sessions
are the corpus the teacher will label, and they are worth nothing until a
teacher passes the gate on the conditions.

**And the pose condition still needs a real 90° turn.** B's centroid moved but
the pose test could not separate facing from turned at all, so the D condition
has to be a genuine head turn, not a glance.

## 6. The architecture this implies

```
camera
  -> row-band removal           (measured: empty room 50.00% -> 0.01%)
  -> motion maps                (measured: speech signal localised, 79% top 5%)
  -> conventional teacher       (trained on conditions, GATED by controls)
  -> soft labels on the free corpus
  -> small student, local, always-on in the glasses
```

The teacher runs offline and slowly. The student runs always-on. The teacher's
only job is to label data Casey will not sit and label by hand, so the corpus
can be far larger than the condition set — and the gate on the conditions is
what makes that trustworthy.

**The LLM is not in this pipeline.** That was the working assumption, and it is
the one thing here that was measured and failed.

## Limits

- Two labelled sessions per class, one speaker, one room, one lighting state.
  Every number in §3 is a pipeline validation, not a performance claim.
- The `scrambled` control for the `summary` features reaches 0.792, so even the
  best margin (+0.208) comes from a feature set that is close to a global
  statistic. The spatial features are the ones that *should* capture the
  measured signal, and they currently do not beat their control.
- The teacher bench's `facing` question is meaningless on this corpus: B's head
  turn never registered.
- The student is not built. Building it before the teacher passes its gate
  would train a student on unvalidated labels, which is the failure mode this
  whole design is meant to prevent.

## Reproduce

```bash
# corpus
/home/proart/projects/glasses/.venv/bin/python experiments/seed_corpus_from_captures.py
# LLM teacher gate
/home/proart/projects/glasses/.venv/bin/python experiments/teacher_bench.py \
  --corpus /mnt/c/proart_capture/corpus --models qwen3:0.6b,llama3.2:1b \
  --question speaking --render grid
# conventional teacher
/home/proart/projects/glasses/.venv/bin/python experiments/teacher_conventional.py \
  --task speaking --features both
```