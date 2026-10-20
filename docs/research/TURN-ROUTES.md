# Turn-end: four routes, one of them mine, and the two experiments that were rigged

*Written after `experiments/turn_routes.py`. Every number here was measured on
this machine against audio in this repo. Two of the four experiments as first
written were circular and were caught by reading the output rather than the
code — those corrections are kept in place, because the corrections are the
result.*

## The problem

`src/turn.py` ends a turn on 3 s of sustained silence. Across the seven-cell
real corpus (485 s, **673 distinct silence runs / 1671 pause events** — the
population is corrected below) it fires on **zero**:

| cell | audio | longest pause run | turns at 3 s |
|---|---|---|---|
| narration | 90.0 s | 1.830 s | 0 |
| poetry | 20.011 s | 1.200 s | 0 |
| essay | 75.0 s | 1.560 s | 0 |
| dialogue | 75.0 s | 1.530 s | 0 |
| shakespeare | 75.0 s | 1.450 s | 0 |
| children | 75.0 s | 1.600 s | 0 |
| oration | 75.0 s | 2.540 s | 0 |

> **CORRECTION (2026-10-19) — the corpus length.** This said **465 s**; the
> seven cells in the table above sum to **485 s**, and measured on disk they
> total **485.01 s** (narration 90.000, poetry 20.011, and 75.000 each for
> essay / dialogue / shakespeare / children / oration —
> `examples/realaudio/alice01_excerpt.wav` plus `examples/realaudio/corpus/*.wav`).
> The 20 s discrepancy is exactly the poetry cell. **The same wrong 465 s is
> hard-coded in six code and test sites** — `src/turn.py:13`,
> `src/speculate.py:6`, `src/watcher.py:661`, `tests/test_turn.py:25`,
> `tests/test_speculate.py:7`, `experiments/turn_routes.py:8` — and **still
> needs fixing there**; notes cannot fix a docstring. `NOTES-AUDIT.md` F7.
>
> **CORRECTION (2026-10-19, later the same day) — the six sites have been
> fixed since.** The 465 s no longer stands as a live figure at any of them:
> `src/turn.py:18`, `src/speculate.py:13`, `src/watcher.py:662`,
> `tests/test_turn.py:25` and `tests/test_speculate.py:11` now quote 485.01 s
> and record the 465 s as the wrong earlier figure, and
> `experiments/turn_routes.py` carries no corpus length at all. What is
> **still** stale at those sites is the *population* wording, not the length —
> see the next two blocks.

> **CORRECTION (2026-10-19) — the pause-run population, re-measured.** The "680
> measured pause runs" this section used to claim is **not** what the
> repository measures today, and the population behind it no longer exists.
> Measured on the corpus that is on disk: **673 distinct silence runs / 1671
> pause events**. The 1671 reproduces `tests/test_speculate.py:99-100`
> **exactly** (min **0.160 s** / median **0.380 s** / max **2.540 s**) because
> it is an **event** count — the watcher re-reports an *open* pause every
> `PAUSE_REPORT_S = 0.25 s` (`src/watcher.py:202`), so one silence run yields
> several events. The 680 was a **run** count; it is 7 runs (**1.03%**) from
> today's 673, and it is **not exactly reproducible** under any method tried,
> because the count moves with the population: 673 distinct runs (adaptive),
> 1671 events (adaptive), 647 events (energy-only, adaptive off), 262 silence
> runs from the energy method at 0.150 s, and **665** runs (1656 events) on
> the hypothetical 465 s corpus that the misstated length implies.
> Truncating the poetry cell to the 15 s that 465 s implies moves the run
> count **down** to 665 — *away* from 680, not up towards it — so the 465 s
> misstatement does not explain the 680. Every rate computed on the 680
> population is a rate on a population that cannot be re-derived. The table
> above is now on the 673-run population. `docs/research/PAUSE-THRESHOLD.md` §5;
> `NOTES-AUDIT.md` F8.

> **STILL TO FIX (2026-10-19) — outside this file's write lane.** None of these
> were edited; a note cannot fix a docstring.
>
> * `experiments/turn_routes.py:8` — the module docstring still says "across
>   680 measured pause runs the longest is 2.1 s". Today's longest run is
>   **2.540 s** (oration) on **673** runs / 1671 events.
> * `experiments/turn_routes.py:125` — still quotes "real inter-utterance gaps
>   run 0.11-0.64 s". The **0.11 s** end of that range is the leading-silence
>   artefact (`src/energy.py:290`); the inter-utterance range starts at
>   **0.15 s**.
> * `src/speculate.py:7,14` and `tests/test_speculate.py:7,12,99` — still call
>   the 1671 "1671 pause **runs**". They are pause **events**; 673 is the
>   run count.
> * `src/turn.py:13,19-21` — "costs 46 pauses, 17.8% **of them all**" still
>   has no denominator (it is 46 of the 259 pauses the 150 ms trigger keeps),
>   and the claim that "the pause counts and the 17.8% were measured on that
>   older 465 s population and need re-measuring" is now **falsified**: every
>   cell of the pause table reproduces exactly on the current corpus. Only the
>   length stated beside the table was misstated.

A peripheral that never speaks is also a failure. "The threshold is correct and
the material is atypical" is not a place to stop.

## Ground truth by construction

There are no turn annotations and hand-labelling hours of audio was not on the
table. So the truth is manufactured: take continuous real speech and **insert
a silence** at a controlled place.

| case | construction | truth |
|---|---|---|
| **END** | speech, silence, nothing follows | turn ended |
| **MID** | speech, silence, same voice resumes | turn did not end |
| **RESUME** | speech, **long** silence, same voice resumes | turn did not end |
| **NONE** | untouched original, no silence | never ended |

`RESUME` exists to break a circularity — see below.

## Experiment 1 (rigged) and what it revealed

First run, scoring each route on the **complete clip**:

```
truth       n  acoustic smartturn
end        60     0.224     0.803
resume     48     0.005     0.437
mid        60     0.005     0.404
none       12     0.005     0.560

  acoustic     concordance 1.0000
  smartturn    concordance 0.7063
```

**My own route scored a perfect 1.0, and that was the tell.** `trailing_silence`
is computed over the whole clip, so it can see the speech that comes *after* the
pause. END has none, RESUME has 0.45–2.0 s of it. A trailing-silence rule
separates them **by definition** — it would score 1.0 on any corpus built this
way and says nothing about whether it could tell a speaker finishing from a
speaker breathing.

`RESUME` was added specifically to break that, and it is the case that matters
operationally: it has the *longest* silence in the set and the speaker still
comes back. A peripheral that answers "the turn ended" whenever it hears
silence will interrupt every pause a person ever takes — and the interruption
happens before the evidence that it was wrong arrives.

## Experiment 2 (also rigged) — the causal cut

The deeper problem: **trailing silence requires future audio.** A live
peripheral stands at the end of a pause with no idea what follows. So every
route was re-scored on `audio[:decision]`, where `decision` is the start of the
final silence. At that instant END and RESUME are byte-identical prefixes.

```
truth       n  acoustic smartturn          <- causal, no future
end        60     0.006     0.614
resume     48     0.007     0.478
mid        60     0.008     0.552
none       12     0.005     0.560

  acoustic     concordance 0.7848
  smartturn    concordance 0.5541
```

**Our duration rule: 0.0064 on END, 0.0067 on RESUME.** Identical. It cannot
see the future, and no amount of tuning changes that — the information is not
in the input it is given.

smart-turn gives **0.614 vs 0.478**, and it is real:

```
observed concordance 0.6247   null mean 0.5079 (chance 0.5)
permutations >= observed: 393/20000  ->  p = 0.0197
```

**Verdict: real separation.** A waveform classifier separates "finished" from
"about to continue" causally, at ~p=0.02, where a duration rule cannot do it at
all. That is a capability our rule does not have and cannot acquire.

But the controls are bad, and they are the reason this is not a win:

| contrast | end | other | concordance |
|---|---|---|---|
| end vs none | 0.614 | 0.560 | 0.549 |
| **resume vs none** | 0.478 | 0.560 | **0.436** |
| mid vs none | 0.552 | 0.560 | 0.464 |

Two of three are at or **below** chance. `resume vs none` at 0.436 is
*inverted* — it scores continuous speech as more turn-complete than speech with
a pause in it. smart-turn v3.2 has a real signal on the one contrast that
matters and is unreliable on the easy ones.

## Route 3: Jev, and a reusable method

A text model cannot hear. Its entire input is what I type. So the first version
described the structure — "and then RESUMES in the same voice" — and asked the
model to judge. **The description contains the answer.** A route that scores
well there learned my phrasing, not the speaker.

Both modes were run:

| case | CC (leak) | CC (neutral) | gap |
|---|---|---|---|
| END/0/0.15 | 0.980 | 0.680 | 0.300 |
| MID/0/0.15 | 1.000 | 0.680 | 0.320 |
| END/0/0.30 | 0.980 | 0.700 | 0.280 |
| MID/0/0.30 | 1.000 | 0.640 | 0.360 |
| END/0/0.60 | 0.980 | 0.620 | 0.360 |
| MID/0/0.60 | 1.000 | 0.640 | 0.360 |
| END/0/1.00 | 0.980 | 0.680 | 0.300 |
| MID/0/1.00 | 1.000 | 0.620 | 0.380 |
| END/0/1.50 | 0.980 | 0.720 | 0.260 |
| MID/0/1.50 | 1.000 | 0.740 | 0.260 |

**Framing gap: mean 0.318.**

Two findings, both useful beyond this experiment:

1. **With the answer in the prompt, Jev is near-certain and INVERTED** —
   `MID` (speaker resumes) scores **1.000**, higher than `END` at 0.980. Given
   a description saying the speaker resumes, it concludes the turn is *more*
   complete. Confident and backwards.
2. **Given identical neutral acoustic scalars, it says ~0.62–0.74 regardless
   of truth** — flat, with no discrimination.

**The gap is the instrument, and it is reusable.** Any time a language model is
pointed at a signal, the honest baseline is *the same model, same question, no
answer in the prompt*. A large gap means the result was framing. This is the
same failure the symbolic-audio experiment found (identical audio, 1/3 vs 3/3
depending only on how the shape was written down), now measured as a number.

## Route 4: Cloudflare — a working novel use and a broken one

### Works: `@cf/pipecat-ai/smart-turn-v2` on Workers

The turn detector as a Cloudflare AI model. Body is `{"state", "questions",
"audio": <raw PCM base64>}` — a bare string, not the documented object.

| input | `is_complete` | P | latency |
|---|---|---|---|
| 3 s real speech | False | 0.2227 | 906 ms |
| 3 s speech + 1.5 s silence | **True** | 0.7027 | 975 ms |
| 3 s speech + 3 s silence | **True** | 0.9333 | 1396 ms |
| white noise 3 s | True | 0.5461 | — |
| **digital silence 1/3/8 s** | **HTTP 500** | — | — |

This is the useful half and it is a real behavioural difference:

* **Cloud v2 discriminates.** 0.2227 → 0.7027 → 0.9333 as silence grows. That is
  a usable turn signal, and it is the same job smart-turn was doing locally.
* **Cloud v2 returns HTTP 500 on pure digital silence**, where **local v3.2
  returned P(complete)=0.987** — "turn complete" about nothing at all.

So the two implementations fail in **opposite directions on the same input**:
v3.2 answers confidently about silence, v2 crashes on it. **The bug is the
input validation.** A peripheral that feeds a silent frame to a turn model must
have a gate in front of it — which is exactly what `src/vad.py` now is.

Cost note: latency is 0.9–1.4 s per call, which is too slow for the continuous
path and fine for a cold decision. `usage` came back `null`, so per-call neuron
cost is **unmeasured** — not free, and not claimed to be.

### Broken: clef / clef-omni schema regression

The documented body — `{"model","state","questions","audio":[{content_type,
base64}]}` — worked earlier in this same session and now 400s on **every**
variant tried, including a text-only call with no audio at all:

```
no audio          400  required properties at '/' are 'model,state,questions'
audio [object]    400  required properties at '/audio/0' are 'content_type,base64'
audio {body,contentType}  400  required properties at '/audio/contentType' are 'content_type,base64'
```

The validator **rejects the exact key names it names as required.** With
`contentType` it demands `content_type`; with `content_type` it demands
`content_type` at `/audio/contentType`. `clef`, `clef-omni` and `clef-flash`
all fail identically while the token and account are valid.

Not worked around and not guessed at. **The audio-native route is unavailable
right now**, which leaves the four-route experiment with three routes.

## What is left standing

* **Duration cannot do turn-end at decision time.** 0.0064 vs 0.0067, and the
  two inputs are byte-identical. This is not a tuning failure; it is an
  information limit.
* **A waveform classifier can, weakly** (p=0.0197), and this is a capability
  the duration rule cannot have. But it is uncalibrated (0.614 for a true end)
  and inverted on two of three controls.
* **A language model contributes nothing here**, and when told the answer it is
  confident and backwards. The framing gap (0.318) is now a standard
  measurement for every future model-vs-signal claim.
* **Disagreement between routes was the hypothesis and it did not hold.** mean
  spread: end 0.304, resume 0.236, mid 0.272 — the truth conditions disagree by
  roughly the same amount. On this data **disagreement is noise** and the
  ensemble should be dropped, not adopted.

That last one is the finding I went looking for and did not get. Four routes
that each fail differently produced no usable combined signal. The honest
consequence is that turn-end on this material is an open problem, and the best
available answer is still *"wait longer, and be able to say you do not know"* —
which is what `TURN_YIELD_FLOOR_S` and the `UNRESOLVED` verdict already do.
---

## Addendum: the watcher could not see a trailing silence at all

Found after the four-route experiment above, by building
`experiments/speculation_payoff.py` to ask whether speculating beats waiting.

**Every conclusion above about "the 3 s threshold never fires" was measured
against a watcher that could not report the silence it was waiting for.**

A pause was emitted only once its segment **closed** — which means once speech
resumed. A silence at the *end* of speech never closes, so it was never
reported, at any length. Measured on real speech, identical construction,
three different silence lengths:

```
END silence=0.20s -> pause run_s = [0.03, 0.08]
END silence=0.60s -> pause run_s = [0.03, 0.08]
END silence=1.30s -> pause run_s = [0.03, 0.08]
```

The trailing silence was not measured **at all**. The watcher's own internal
state knew it — `elapsed_s - _pause_run_start` reached 2.625 s on the
narration fixture — but nothing was ever emitted.

So the 3 s threshold was never the problem. A peripheral could not learn that
someone had stopped talking, which is the single most useful thing a
peripheral needs to know.

**Fix:** an open pause is now reported as it grows, on a `PAUSE_REPORT_S =
0.25 s` cadence, marked `open=True, checkpoint=True`, carrying `run_s`.

```
END silence=0.20s -> [0.26, 0.56, 0.03, 0.08]
END silence=1.30s -> [... 0.3, 0.5, 0.8, 1.1, 1.26]
END silence=2.50s -> [... 1.8, 2.1, 2.3]

real corpus, longest run now visible:
  narration  1.830 s      (was invisible)
  oration    2.540 s      (was invisible)
```

Four defects in that fix, all found by measuring rather than reading:

1. The cadence arithmetic computed the *next* multiple and subtracted, which
   could return a boundary already passed — the run stalled at 0.30 s however
   long the silence got.
2. The checkpoint path emits its own `Event` instead of going through
   `_events_for`, so `_pause_run_start` was never established and every
   checkpoint measured from the segment edge instead of the run start.
3. **`if consumed_to is None: return []`** — the early return discarded the
   checkpoints it had just collected, and that branch is reached *precisely*
   when no segment closed, which is exactly when checkpoints matter. The
   internal counter advanced to 1.25 s while the caller saw nothing.
4. `_pause_reported_at` conflated two different numbers: which cadence steps
   had been emitted (dedup) and the largest `run_s` actually reported. With
   one variable the `MAX_PAUSE_S` flush reported 5.75 s after checkpoints had
   reached 5.7599, so a run appeared to go backwards.

`run_s` is now monotone within a silence run and resets when a new run begins.

### What the fix did to the numbers

Same experiment, same cases, before and after:

| speculate_s | acted | interrupted | accuracy (before) | accuracy (after) |
|---|---|---|---|---|
| 0.20 s | 106 | 76 | 0.283 | 0.528 |
| 0.30 s | 73 | 55 | 0.247 | 0.527 |
| 0.40 s | 70 | 52 | 0.257 | **0.536** |
| 0.60 s | 47 | 41 | 0.128 | 0.473 |
| 0.90 s | 28 | 28 | 0.000 | 0.455 |
| 1.80 s | 4 | 4 | 0.000 | 0.000 |

The pre-fix 0.283 was *worse than a coin flip*, and it was an artefact: the
broken watcher fed the gate almost only RESUME cases, so it could only
interrupt people. Fixing the watcher moved the best accuracy from 0.283 to
0.536 and the number of decisions from 106 to 250.

**But 0.536 is still barely a coin flip, with 84 interruptions.** So even with
the silence visible, speculation on this material is not good enough to ship as
an action trigger. No bar reaches 90%. The honest default stays: wait, and be
able to say you do not know.

---

## Addendum 2: the corpus cannot answer the turn question, and that is the finding

With the watcher fixed, pauses are visible and the question can finally be
asked of **real** speech rather than spliced audio. No manufactured data: for
every silence run the watcher reports, did the speaker continue afterwards?

A "no" at the very end of a file is ambiguous — the recording simply stopped —
so those are counted separately rather than called turn ends.

| threshold | silence runs | speaker resumed | at file end | would have interrupted |
|---|---|---|---|---|
| 0.6 s | 170 | 169 | 1 | **99.41%** |
| 0.9 s | 71 | 70 | 1 | **98.59%** |
| 1.2 s | 35 | 35 | 0 | **100.0%** |
| 1.5 s | 15 | 15 | 0 | **100.0%** |
| 1.8 s | 6 | 6 | 0 | **100.0%** |
| 2.1 s | 2 | 2 | 0 | **100.0%** |

> **CORRECTION (2026-10-19) — this table has been re-measured.** Every rate in
> the version published here (98.1% and 100.0% ×5) was computed on the
> **680-run** pause population, which the repository no longer measures — the
> 1671 it records is an **event** count, not a run count. The table above is
> re-measured on today's **673-run** population: 170 / 71 / 35 / 15 / 6 / 2
> silence runs, 99.41% / 98.59% / 100.0% ×4, and **1** run "at file end" at
> 0.6 s and 0.9 s. The three highest-threshold rows (1.5 / 1.8 / 2.1 s)
> reproduce the old table exactly; the three lowest do not. The *conclusion* —
> in recorded speech, silence is essentially never a turn end — is unchanged,
> and is now **verified on a population that can be re-derived**: at every
> threshold from 0.6 s up, at least 98% of the silence runs were followed by
> the speaker coming back, and the one exception is the end of a recording.
> Full measurement: `docs/research/PAUSE-THRESHOLD.md` §5 and
> `experiments/pause_threshold_results.json`. `NOTES-AUDIT.md` F8.

**In recorded speech, silence is essentially never a turn end. The speaker
comes back.**

That single fact explains every other number in this document:

* Why the 3 s threshold fires on nothing — the longest silence in 485 s of
  seven cells is 2.54 s.
* Why speculation scores ~0.536 with 84 interruptions — the manufactured END
  cases (speech, then silence, then *nothing*) do not resemble the corpus,
  because in the corpus the answer is almost always "they kept talking".
* Why the four routes disagreed without pattern: they were being asked about
  events that essentially do not occur in the material.

**THE REALISATION: this corpus cannot answer the turn question.**

Every file in `examples/realaudio/corpus/` is a recording of continuous
speech — a LibriVox chapter, a reading, an address. A recording is cut at an
arbitrary point and people keep talking until the cut. So it contains
phrase-level pauses, and essentially **zero** instances of a speaker
yielding the floor to someone else, because there is no someone else.

Tuning a turn detector on this corpus tunes it on a fiction: the only way a
pause becomes a turn end in this material is if the recording stops, which is
an artefact of the file, not a property of the conversation.

What would actually be needed:

* **Two speakers.** Turn-taking is a two-party phenomenon. Every corpus file
  here is monologues, so the positive class is not merely rare, it is
  *unrepresentable*.
* **Natural turn exchange**, not read speech. Read material has different
  pause structure from conversation and will mislead any threshold fitted to
  it.
* **Overlapping and interrupted turns**, which is where real systems fail and
  which no splice of a single voice can produce.

Until then the honest position is unchanged and now has a measured reason
behind it: `TURN_END_S = 3.0` is not wrong, it is **unfalsifiable with the
data available**, and `UNRESOLVED` is the correct output rather than a
failure to have tried harder.

## A harness error worth recording

Three separate measurements in this thread were wrong in the same way, and
each was caught by the answer being implausible rather than by a failing
assertion:

1. Events collected first, then fed to the tracker with the watcher's
   **final** clock — every pause looked like it was seen at the end of the
   session, and the tracker correctly declined to believe it.
2. Comparing each checkpoint against the file's final `elapsed_s` — produced
   a clean **100%** interrupt rate, which was obviously wrong.
3. Comparing a checkpoint against the clock at push time — produced a clean
   **0%**, because a checkpoint's `t1` *is* the current clock by construction.

Only the third formulation (look forward for a later utterance, with
file-end runs counted separately) is the question actually being asked.

A measurement that returns a suspiciously round 100% or 0% is usually a
harness bug, not a result.
