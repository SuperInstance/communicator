# NOTES AUDIT 3 — the work the last two audits produced, audited against its result files

**Measured 2026-10-19. 18 findings: 12 confirmed, 6 candidates. ~1,300 numeric
claims traced.**

This is round three of the same audit, applied to the documents written or
heavily edited by the last few sessions: `PAUSE-THRESHOLD.md`,
`SILENCE-LEADING-ARTEFACT.md`, `DIALOGUE-INTAKE.md`, `ARTIFACT-REPRODUCIBILITY.md`,
`LLM-DIRECTOR.md`, `PROGRESS.md`'s newest arc, the pause-statistics sections of
`ADOPTION-PLAN.md` and `TURN-ROUTES.md`, and `NOTES-AUDIT-2.md` itself.

**Method.** Every numeric claim was traced to the file that produced it, or
recomputed from that file's own script. Where a number is not in any result
file, the producing code was re-run: `pause_threshold.py`'s own `_EventProbe`
and `_runs_from_events` over the seven-cell corpus (`src/wavio.py` read, 16 kHz
analysis rate), `src/energy.py`'s own `segment()` over the same corpus, and
`director_proposal_stability.py`'s own `jaccard()` over the recorded runs.
Recomputations were validated first: the harness reproduces the committed
adaptive watcher population exactly, cell for cell (126 / 29 / 109 / 115 / 124 /
89 / 81 = 673 runs) and the committed per-cell gap counts exactly, so a
disagreement with it is a disagreement with the repo's own method and not with
my re-implementation. Numbers that exist only in git history were traced
through `git rev-parse` and `git show` at the commits named. numpy only; no
sklearn, pandas, PIL, cv2 or matplotlib.

**What the last two rounds found, and what this one finds.** Round one: 14
findings, 10 confirmed. Round two: 16 findings over 49 documents, 10 confirmed.
This round is quieter in absolute terms and sharper in kind. The confirmed
findings here are almost all of one family, and it is the family the 0.675
belongs to: **a document describing a state that a later commit has already
moved past.** A stale run's numbers, a stale working-tree claim, a stale
"this was not changed" note, a stale "these files were not edited" table. The
corrections themselves — the poetry `max_gap` in `src/turn.py`, the round-two
voiceloop edits, the `ADOPTION-PLAN.md` pause table — were checked against
their sources and are **sound**. A correction that introduced a new error
would have been the worst outcome; there is none of that here.

---

## Summary of findings

| id | doc | line | claim | verdict | severity |
|---|---|---|---|---|---|
| F1 | `PAUSE-THRESHOLD.md` | 285 | "watcher distinct runs, energy-only \| **215**" | recomputes to **216** | CONFIRMED |
| F2 | `PAUSE-THRESHOLD.md` | 288, 293; `TURN-ROUTES.md` | 57–59 | "hypothetical **465 s** corpus (poetry truncated to 15 s) \| 1656 events / **665** runs" | those are the **480 s** corpus; the true 465 s corpus gives **1612 / 644** | CONFIRMED |
| F3 | `DIALOGUE-INTAKE.md` | 156–159 | "the reported `signal_frame_frac` for the low channel (0.0042) is **the same** as for a digitally silent channel (0.0042)" | a digitally silent channel reports **0.0** | CONFIRMED |
| F4 | `DIALOGUE-INTAKE.md` | 88 | "wall time \| 10.4 s \| 69.4 s" | the file records **9.24 s** | CONFIRMED |
| F5 | `PROGRESS.md` | 1106 | director repeatability, 60,000-char row: "3 distinct \| −0.143 \| +0.262 \| +0.333 \| **0.476**" | the committed file gives **4 / +0.000 / +0.119 / +0.333 / 0.333** | CONFIRMED |
| F6 | `PROGRESS.md` | 1077–1078 | "The script at `7a4dff5` and at `HEAD` are **byte-identical**" | blobs `459e23c0…` vs `f8f89c96…` | CONFIRMED |
| F7 | `PROGRESS.md` | 1173–1174 | "the sixteen from round two are written up and **pending**" | the ten confirmed ones were applied at `c71a6c5` | CONFIRMED |
| F8 | `LLM-DIRECTOR.md` | 164–166 | "qwen3-30b-a3b-fp8 −0.083 / **+0.083** with **5 hedges** and **0 false accusations**" | file: −0.0833 / **−0.0833**, **0** uncertain, **1** false accusation | CONFIRMED |
| F9 | `LLM-DIRECTOR.md` | 178–180 | "qwen3-30b's low specificity is **hedging, not accusation**: 0 false accusations and **5 \"uncertain\" verdicts**" | no committed row has more than 2 uncertain verdicts | CONFIRMED |
| F10 | `SILENCE-LEADING-ARTEFACT.md` | 522–530 | "`src/turn.py:37` … reads `max_gap \"-\"` … **it is wrong today**" | corrected in the same commit that added the file this note audits | CONFIRMED |
| F11 | `SILENCE-LEADING-ARTEFACT.md`; `PAUSE-THRESHOLD.md` | 272, 278, 281–283, 296; 410–428 | cross-references between the three pause notes | the measurements agree; **the line pointers do not** | CONFIRMED |
| F12 | `ARTIFACT-REPRODUCIBILITY.md` | 666–671 | "another agent has **uncommitted changes** to `experiments/pause_threshold.py` and `…_results.json` right now" | both files are clean in `HEAD` | CONFIRMED |
| C1 | `PAUSE-THRESHOLD.md` | 280–288 | 647, 664, 622, 44, 1656, 665 in the candidate-population table | reproducible, but **in no result file** | CANDIDATE |
| C2 | `LLM-DIRECTOR.md`; `PROGRESS.md` | 393–396; 1212–1213 | "a title for a *different* experiment scores **0.000** against all five" | it is the **control stub's** title; no baseline stored | CANDIDATE |
| C3 | `LLM-DIRECTOR.md` | 182–188 | llama **+0.583** at 40 kB; gpt-oss **−0.083** blind | in **no committed result file** | CANDIDATE |
| C4 | `SILENCE-LEADING-ARTEFACT.md` | 441 | "3 failed, 606 passed, 5 skipped" | a test-suite result; not re-run here | CANDIDATE |
| C5 | `ARTIFACT-REPRODUCIBILITY.md` | 184–191 | "389 s / 350 s / 225 s"; "31 `REPRODUCIBLE` and 21 `NOT_RE_RUN`" at the 120 s default | the manifest is **not committed** | CANDIDATE |
| C6 | `LLM-DIRECTOR.md` | 534 | "The context materials are **139 kB** of a **372 kB** corpus" | the file records **140000** chars; 372 kB traces to `debrief.py:477` | CANDIDATE |

---

## 1. The confirmed findings

### F1 — "215" energy-only silence runs is 216

`PAUSE-THRESHOLD.md:283-285`, the candidate-population table:

> | watcher pause events, `run_s > 0.15`, energy-only (adaptive off) | 647 |
> | watcher pause events, no filter, energy-only | 664 |
> | watcher distinct runs, energy-only | **215** |

`experiments/pause_threshold.py` hardcodes `adaptive=True` (line 593), so the
energy-only population is **not** in `pause_threshold_results.json` at all. I
re-ran the script's own machinery with `adaptive=False`:

| cell | distinct silence runs |
|---|---:|
| narration | 45 |
| poetry | 6 |
| essay | 30 |
| dialogue | 38 |
| shakespeare | 47 |
| children | 44 |
| oration | 6 |
| **total** | **216** |

The harness is the script's, not a re-implementation: the same call with
`adaptive=True` reproduces the committed per-cell population exactly
(126 / 29 / 109 / 115 / 124 / 89 / 81 = **673**, matching
`watcher_pause_run_population.per_cell`). The two other numbers in the same
table (**647** events > 0.15 s, **664** events unfiltered) reproduce exactly.

**Correction.** `PAUSE-THRESHOLD.md:285`, current text `| watcher distinct runs, energy-only | 215 |`, replacement `| watcher distinct runs, energy-only | 216 |`.

### F2 — the "465 s corpus" is a 480 s corpus

`PAUSE-THRESHOLD.md:288` and `:293`, and the same passage in
`TURN-ROUTES.md:57-59`:

> | the same, on a hypothetical 465 s corpus (poetry truncated to 15 s) | 1656 events / **665** runs |
>
> … the **465 s** figure implies moves the run count *down* to 665 …

Both numbers reproduce exactly — for the corpus that exists when the poetry
cell is truncated to **15 s**:

| corpus | length | events > 0.15 s | distinct runs |
|---|---:|---:|---:|
| poetry truncated to 15 s | **480 s** | **1656** | **665** |
| poetry removed entirely | **465 s** | **1612** | **644** |

The mislabelling is arithmetic, not a typo. The seven-cell corpus is
485.011125 s; the published 465 s is that figure **minus the poetry cell
(20.011125 s)**. So 465 s implies **zero** seconds of poetry, not 15 s — and
`PAUSE-THRESHOLD.md:266-277` says so itself, in the same document, when it
identifies the 20.011 s gap as "exactly the poetry cell". Truncating poetry to
15 s yields 480 s, and the run count 665 belongs to that 480 s corpus.

The consequence for the argument is small but it is a population error in a
document whose whole subject is populations: the claim "**the 465 s
misstatement moves the count away from 680, not towards it**" is true of both
665 and 644, so the conclusion survives — but the number attached to it is
attached to the wrong corpus.

**Correction.** `PAUSE-THRESHOLD.md:288`, current text
`| the same, on a hypothetical 465 s corpus (poetry truncated to 15 s) | 1656 events / **665** runs |`,
replacement
`| the same, on a hypothetical 480 s corpus (poetry truncated to 15 s) | 1656 events / **665** runs | — the 465 s figure implies poetry *removed entirely*, which gives 1612 events / 644 runs |`.
Same edit at `TURN-ROUTES.md:57-59`: replace "the hypothetical 465 s corpus that
the misstated length implies" with "a hypothetical 480 s corpus (poetry truncated
to 15 s)" and "Truncating the poetry cell to the 15 s that 465 s implies" with
"Truncating the poetry cell to 15 s gives a 480 s corpus; the 465 s the
published length implies is the corpus with poetry removed, which gives 1612
events and 644 runs".

### F3 — a digitally silent channel reports `signal_frame_frac` 0.0, not 0.0042

`DIALOGUE-INTAKE.md:153-159`:

> A channel at RMS 2e-4 reports `silent: false`, `rms: 0.0002`,
> `signal_frame_frac: 0.0042`; at RMS 5e-5 it reports `silent: false`,
> `rms: 5e-05`, `signal_frame_frac: 0.0021`. Nothing is gated. Note that the
> reported `signal_frame_frac` for the low channel (0.0042) is **the same** as
> for a digitally silent channel (0.0042) — the report cannot tell "quiet but
> alive" from "dead mic" either.

The first two numbers are right (`experiments/dialogue_intake_selftest_results.json`,
`perturbations[low_level_B].evidence` → `signal_frame_frac: 0.0042`;
`perturbations[very_low_level_B].evidence` → `0.0021`). The equality is not.
A digitally silent channel reports **0.0**:

| perturbation | channel | field | value |
|---|---|---|---:|
| `digital_silence_B` | B (silent) | `B_signal_frac` | **0.0** |
| `digital_silence_B` | B (silent) | `B_rms` | 0.0 |
| `digital_silence_A` | A (silent) | `A_signal_frac` | **0.0** |
| `low_level_B` | B (RMS 2e-4) | `B_signal_frac` | 0.0042 |

`experiments/dialogue_intake.py:91` computes `"silent": peak == 0.0`, and
line 90 computes `signal_frame_frac` as frames above `SPEECH_FLOOR` — a
digitally silent channel has no frame above the floor, so the value is 0.0.

The 0.0042 that *does* exist belongs to a different pair: the baseline's
quiet-but-alive channel B (`B_rms: 0.006365` → `B_signal_frac: 0.0042`).
That is the real "cannot tell quiet-but-alive from dead-mic" instance — and it
is the low channel of §4 row 4 against the *baseline* channel B, not against a
digitally silent channel. The sentence's conclusion therefore needs rewording,
not just its number: `signal_frame_frac` does distinguish a digitally silent
channel (0.0) from a near-dead one (0.0042); what it cannot distinguish is the
baseline's quiet-but-alive B (0.0042 at RMS 0.006365) from the perturbed
low-B (0.0042 at RMS 0.0002).

**Correction.** `DIALOGUE-INTAKE.md:157-159`, current text "Note that the
reported `signal_frame_frac` for the low channel (0.0042) is **the same** as
for a digitally silent channel (0.0042) — the report cannot tell \"quiet but
alive\" from \"dead mic\" either.", replacement: "Note that the reported
`signal_frame_frac` for the low channel (0.0042) is **the same** as for the
baseline's quiet-but-alive channel B (0.0042 at RMS 0.006365) — the report
cannot tell \"quiet but alive\" from \"nearly dead\". A *digitally* silent
channel is different: it reports `signal_frame_frac` 0.0 and `rms` 0.0."

### F4 — the matrix wall time is 9.24 s, not 10.4 s

`DIALOGUE-INTAKE.md:88`, the "What the good corpora produce" table:

> | wall time | 10.4 s | 69.4 s |

`experiments/dialogue_intake_selftest_results.json` records:

| corpus | field | value |
|---|---|---:|
| matrix (96 s) | `baseline_good_corpus.wall_s` | **9.24** |
| protocol-length (615 s) | `protocol_length_corpus.wall_s` | **69.46** |

The protocol-length figure matches (69.46 → 69.4). The matrix figure does
not, and "10.4" appears nowhere in the file. Wall time is a property of the
machine and the load, not of the measurement, so this is the least consequential
finding in the set — but it is a number in the prose that does not trace, and
the sibling cell in the same row does, which is what makes it look like a
transcription error rather than a timing drift.

**Correction.** `DIALOGUE-INTAKE.md:88`, current text `| wall time | 10.4 s | 69.4 s |`,
replacement `| wall time | 9.24 s | 69.46 s |`.

### F5 — `PROGRESS.md` quotes the superseded 60,000-char director run

`PROGRESS.md:1103-1107`:

> | mode | material | runs | distinct headlines | min | median | max | spread |
> |---|---|---:|---:|---:|---:|---:|---:|
> | blind | none (3,031 chars) | 8 | **4** | −0.190 | +0.357 | +0.429 | **0.619** |
> | context | 60,000 chars | 6 | **3** | −0.143 | +0.262 | +0.333 | **0.476** |
> | context | 140,000 chars | 6 | **3** | +0.381 | +0.619 | **+1.000** | **0.619** |

The blind and 140,000-char rows are correct against the committed files. The
60,000-char row is not:

| | `director_repeatability_context_60000.json` |
|---|---|
| prompt SHA-256 | `72547b3d7a477718…` (prompt 75,800 chars) |
| runs | 6 |
| distinct headlines | **4** |
| headlines | +0.048, +0.190, +0.000, +0.333, +0.000, +0.190 |
| min / median / max | **+0.000 / +0.119 / +0.333** |
| spread | **0.333** |

The quoted row (−0.143 / +0.262 / +0.333, spread 0.476, 3 distinct) is the
**earlier** run of that arm — the one with the 715-char-smaller prompt, SHA
`6a2b547b7becf0f9`. `LLM-DIRECTOR.md:239-241` says so in as many words and
correctly frames it as superseded. `PROGRESS.md` does not: it prints the old
numbers next to two current ones, with nothing marking the difference, in a
table whose whole point is that the numbers are per-run.

**Correction.** `PROGRESS.md:1106`, current text
`| context | 60,000 chars | 6 | **3** | −0.143 | +0.262 | +0.333 | **0.476** |`,
replacement
`| context | 60,000 chars | 6 | **4** | +0.000 | +0.119 | +0.333 | **0.333** |`
— or, if the earlier run is worth keeping for the contrast it makes, label it
as the superseded run and give its prompt SHA, as `LLM-DIRECTOR.md` does.

### F6 — the script is no longer byte-identical across those two commits

`PROGRESS.md:1077-1078`:

> Neither the code nor the data is the variable. The script at `7a4dff5` and at
> `HEAD` are byte-identical and both return 0.000.

They are not:

```
$ git rev-parse 7a4dff5:experiments/teacher_conventional.py
459e23c0e621456219766ceda86abe3fa1cfe56d
$ git rev-parse HEAD:experiments/teacher_conventional.py
f8f89c961d3f3b6038964d48813ca087fdd15999
```

The script gained the provenance API in `d6baa6d`. `ARTIFACT-REPRODUCIBILITY.md:413-419`
identifies exactly this and says the same claim in `NOTES-AUDIT.md:93-94` is
**"no longer true"** — but the correction was never applied there, and
`PROGRESS.md` repeats the stale sentence in its newest arc.

The substantive claim survives: the artefact at `7a4dff5` holds LOO B = **0.675**,
and `d6baa6d`, `c71a6c5` and `HEAD` all hold **0.000**; the four teacher
feature sets reproduce bit-exactly at `HEAD`
(`speaking_both` A 0.000 / B 0.000, `speaking_map` A 0.000 / B **1.000**,
`speaking_region+map` A 0.000 / B **0.200**). Only the "byte-identical" clause
is false.

**Correction.** `PROGRESS.md:1077-1078`, current text "The script at `7a4dff5`
and at `HEAD` are byte-identical and both return 0.000.", replacement "The
script at `7a4dff5` and at `HEAD` are **not** the same file — it gained the
provenance API in `d6baa6d` (blob `459e23c0…` → `f8f89c96…`) — and **both**
versions return 0.000, so the conclusion does not depend on which one is run."
The same fix belongs at `NOTES-AUDIT.md:93-94`; that file was outside this
audit's write lane but inside the correction set this finding names.

### F7 — the round-two corrections are applied, not pending

`PROGRESS.md:1172-1174`:

> The four corrections from round one are in the tree; the sixteen from round two are
> written up and pending.

Round two's commit message is `c71a6c5` — *"Notes audit round two: ten confirmed
findings applied, and the five residuals they exposed"*. The ten confirmed
corrections are in the tree, as dated correction blocks in the project's
established style: `voiceloop/JEV-AS-INSTRUMENT.md:150-161` and `:256-261`
(F5, F7), `voiceloop/README.md:266-276` and `:288-300` (F6, F8),
`alwayson/PYANNOTE.md` and `alwayson/OPENWAKEWORD.md` (F1–F4). All were
verified against `docs/research/voiceloop/jev_voiceloop_results.json` and are
correct (see §3 below).

`PROGRESS.md`'s newest arc was added at `d6baa6d`, when "pending" was true, and
last edited at `d4a49c1` — **after** `c71a6c5` — without the sentence being
updated. Six of the sixteen remain outstanding (the five residuals plus one
unconfirmed); ten are done.

**Correction.** `PROGRESS.md:1173-1174`, current text "the sixteen from round two are
written up and pending.", replacement "the **ten confirmed** of the sixteen from
round two are applied (commit `c71a6c5`); the five residuals they exposed and
the one unconfirmed finding are still open."

### F8 — the qwen parenthetical in `LLM-DIRECTOR.md` contradicts the file it cites

`LLM-DIRECTOR.md:164-166`:

> (On the earlier, easier set, before the two corrections: gpt-oss-120b +0.833
> strict / +1.000 lenient, llama-3.3-70b +0.250 / +0.417, qwen3-30b-a3b-fp8
> **−0.083 / +0.083 with 5 hedges and 0 false accusations.**

The "earlier, easier set" is the 6-false / 4-true claim set, and the committed
record of it is `debrief_audit_blind_cf-meta-llama-…json` at `6d1456b`:

| field | qwen3-30b-a3b-fp8 at `6d1456b` |
|---|---:|
| `headline_sens_plus_spec_minus_1` | **−0.0833** ✓ |
| `headline_lenient` | **−0.0833** (not +0.083) |
| `uncertain` | **0** (not 5) |
| `false_positives` | **1** (not 0) |
| `n_false` / `n_true` | 6 / 4 |

The gpt-oss and llama figures in the same parenthetical are exact against that
commit (`+0.8333 / +1.0000` and `+0.2500 / +0.4167`). The qwen one has the
right strict score and three wrong numbers beside it. The lenient headline
being *negative* is the substantive point: with `uncertain: 0` and
`false_positives: 1`, qwen's low specificity came from an **actual accusation**,
not from hedging.

**Correction.** `LLM-DIRECTOR.md:165-166`, current text "qwen3-30b-a3b-fp8
−0.083 / +0.083 with 5 hedges and 0 false accusations.", replacement
"qwen3-30b-a3b-fp8 −0.083 / −0.083, with 0 uncertain verdicts and 1 false
accusation."

### F9 — "5 \"uncertain\" verdicts" is in no committed debrief row

`LLM-DIRECTOR.md:178-180`:

> qwen3-30b's low specificity is **hedging, not accusation**: 0 false
> accusations and 5 "uncertain" verdicts.

The full inventory of every committed debrief row in the repository's history
(13 distinct rows across 4 files and 6 commits) has `uncertain` ∈ {0, 1, 2}.
The current qwen rows have `uncertain: 1` (meta-first file) and `uncertain: 0`
(openai-first file); the 6d1456b row has 0. "0 false accusations" is correct
for the current set and wrong for the earlier one (F8). The hedging claim, in
the sentence that exists to make the point about qwen, is not supported by the
committed record.

**Correction.** `LLM-DIRECTOR.md:178-180`, current text "qwen3-30b's low
specificity is **hedging, not accusation**: 0 false accusations and 5
\"uncertain\" verdicts.", replacement "qwen3-30b's low specificity in the
current blind run is **hedging, not accusation**: 0 false accusations and 1
\"uncertain\" verdict." — or, if the intended reference is the earlier 6/4 set,
state that set and its actual numbers (0 uncertain, 1 false accusation).

### F10 — the `src/turn.py` note describes a state the same commit already fixed

`SILENCE-LEADING-ARTEFACT.md:522-530`:

> **One stale number found on the way, outside this lane and NOT changed:** the
> `src/turn.py:37` docstring row for poetry reads `max_gap "-"` and `gaps >= 3 s
> "-"`, but the whole-file energy population has a **5.100 s** gap in poetry …
> it is wrong today, and it is in a shipped file's docstring …

`src/turn.py:37` today reads:

```
    poetry           20.011    1.740              0
```

The correction was applied in **`d4a49c1`** — the same commit that added
`experiments/silence_leading_artefact_results.json`, the file this note
audits. So the note's own §7 paragraph, written before that commit, is now
false on its facts.

Two things worth stating, because the note and the fix chose different
readings of the same column and the fix is the better one:

- The note's diagnosis was that the row should read **5.100** (the whole-file
  energy population, `max_s: 5.1` in the results file).
- The applied fix set it to **1.740** — poetry's longest **inter-utterance**
  gap, which is the population `src/turn.py`'s "gaps ≥ 3 s" column counts.
  For the other six cells the two readings coincide (`max_gap_all` ==
  `max_inter_utterance_gap` for narration 1.98, essay 4.53, dialogue 1.13,
  shakespeare 2.10, children 1.96, oration 0.38), so making the column
  explicitly the inter-utterance one is the coherent choice and makes the
  table internally consistent.

The applied fix is **sound**. The note is what is now out of date.

**Correction.** `SILENCE-LEADING-ARTEFACT.md:522-530`, current text "**One
stale number found on the way, outside this lane and NOT changed:** the
`src/turn.py:37` docstring row for poetry reads `max_gap \"-\"` … it is wrong
today …", replacement "**One stale number found on the way, outside this lane
and not changed *here*:** the `src/turn.py:37` docstring row for poetry read
`max_gap \"-\"` and `gaps >= 3 s \"-\"`. It has since been corrected to
`poetry 20.011 1.740 0` — the row's `max_gap` column is the **inter-utterance**
population, which is what its `gaps >= 3 s` column counts, and poetry's
longest inter-utterance gap is 1.740 s. The whole-file energy population's
5.100 s is poetry's trailing silence, not a gap between utterances."

### F11 — the three pause notes agree on the numbers and disagree on the line numbers

The measurements in `PAUSE-THRESHOLD.md`, `SILENCE-LEADING-ARTEFACT.md` and
`ADOPTION-PLAN.md` are **identical** — I checked the per-cell tables, the
46-of-259 headline, the artefact-free shortests, the trailing-silence table
and the Addendum-2 rows against the two result files, and they agree
everywhere. What does not agree is the line-number apparatus around them.

`SILENCE-LEADING-ARTEFACT.md` §4.7 and `PAUSE-THRESHOLD.md` §7 were written
against the *pre-correction* `PAUSE-THRESHOLD.md` and `ADOPTION-PLAN.md`. The
docs moved; the pointers did not.

| pointer | what it points at now | where the number is |
|---|---|---|
| `SILENCE-LEADING-ARTEFACT.md:272` → `PAUSE-THRESHOLD.md:247` | "was never written down anywhere until now." | 262 is at `PAUSE-THRESHOLD.md:133`, `:158` |
| `SILENCE-LEADING-ARTEFACT.md:278` → `PAUSE-THRESHOLD.md:181-182` | the **shakespeare** and **children** rows of the defect table | dialogue is at `:180`; `:181-182` are the other two |
| `SILENCE-LEADING-ARTEFACT.md:281` → `PAUSE-THRESHOLD.md:241` | prose about two tables counting different populations | 1671 is at `:268` |
| `SILENCE-LEADING-ARTEFACT.md:282` → `PAUSE-THRESHOLD.md:242` | same prose | 1789 is at `:269` |
| `SILENCE-LEADING-ARTEFACT.md:283` → `PAUSE-THRESHOLD.md:243` | same prose | 673 is at `:270` |
| `SILENCE-LEADING-ARTEFACT.md:296` → `PAUSE-THRESHOLD.md:270-277` | the candidate-population table | the Addendum-2 watcher rows are at `:311-316` |

`PAUSE-THRESHOLD.md` §7 is stale in the other direction, and more seriously:
its header says *"**These files were not edited.** The corrections below are
for the parent agent to apply"*, and it lists six `ADOPTION-PLAN.md` items as
outstanding — `:121` (the 110 ms claim), `:129`/`:130`/`:131` (the per-cell
shortests), `:126-132` (the gaps column), `:147` (the denominator), `:154`
("sit *below*"). **All six have since been applied**: the corrected
inter-utterance table is at `ADOPTION-PLAN.md:126-135`, the correction blocks
at `:137` (the 110 ms claim), `:178` (the denominator) and `:193` ("sit
*below*"), and the line numbers in the §7 table now land on the *corrected*
text. `ADOPTION-PLAN.md:121` today reads "the shortest **inter-utterance**
pause a human voice actually takes is **150 ms**" — exactly what §7 asks for.

**Correction.** Two edits, both mechanical:

1. `PAUSE-THRESHOLD.md:412` (header) and the table at `:416-427`: change
   "**These files were not edited.** The corrections below are for the parent
   agent to apply" to a statement that the `ADOPTION-PLAN.md` corrections were
   applied in commit `d4a49c1`, and re-point the six rows at the correction
   blocks that now exist (`ADOPTION-PLAN.md:137`, `:123`, `:129-135`, `:178`,
   `:193`) rather than at the original line numbers.
2. `SILENCE-LEADING-ARTEFACT.md:272, 278, 281, 282, 283, 296`: re-point at
   `PAUSE-THRESHOLD.md:133` (262), `:180` (dialogue shortest), `:268` (1671),
   `:269` (1789), `:270` (673) and `:311-316` (the Addendum-2 rows).

### F12 — `ARTIFACT-REPRODUCIBILITY.md` describes a working tree that no longer exists

`ARTIFACT-REPRODUCIBILITY.md:666-671`:

> 6. **The audit is pinned to `c71a6c5`.** The working tree moved during the
>    work (another agent has uncommitted changes to
>    `experiments/pause_threshold.py` and
>    `experiments/pause_threshold_results.json` right now). …

Both files are clean in `HEAD` (`38f948c`):

```
$ git status --short
?? docs/research/ARTIFACT-REPRODUCIBILITY.md
?? experiments/artifact_repro.py
?? tests/test_artifact_repro.py
```

The two files named in the paragraph are not modified. The observation was true
when the report was written and is now a stale statement about someone else's
working tree — which is the same class of defect as the 0.675: a claim about a
state that a later commit moved past, which is why it is worth catching in the
audit of the audit.

**Correction.** `ARTIFACT-REPRODUCIBILITY.md:669-671`, current text "(another
agent has uncommitted changes to `experiments/pause_threshold.py` and
`experiments/pause_threshold_results.json` right now)", replacement "(the
working tree moved during the work; `experiments/pause_threshold.py` and
`experiments/pause_threshold_results.json` were regenerated and **committed**
in `d4a49c1`, so the audit's subject is `c71a6c5` while the tree now holds a
later version of both)".

---

## 2. The candidates

### C1 — six numbers in the candidate-population table are in no result file

`PAUSE-THRESHOLD.md:280-288` presents a table of eight populations as
"measured today on the current corpus". Four of the eight are in
`pause_threshold_results.json`; the rest are not, because the script hardcodes
`adaptive=True` and the energy-only and hypothetical-corpus variants were never
persisted.

| number | in a result file? | what I found |
|---|---|---|
| 647 | no | reproduces exactly (re-running `_EventProbe(adaptive=False)`) |
| 664 | no | reproduces exactly |
| **215** | no | **216** — see F1 |
| 622 | no | reproduces exactly, with `min_silence_s=0.0` and `min_speech_s=0.060` held |
| 44 | no | reproduces exactly on that population (44/622 = 7.07%) |
| 1656 / 665 | no | reproduce exactly — for the **480 s** corpus, see F2 |

The numbers are right; they are simply not recorded anywhere a reader can
check them, which is the standard this project has set itself in
`ARTIFACT-REPRODUCIBILITY.md`. **Suggested fix** (owner of
`pause_threshold.py`): persist the energy-only and no-filter populations in the
results file alongside the adaptive one, and drop the `min_silence_s=0` variant
into the same file rather than leaving it as a prose figure.

### C2 — "a title for a different experiment" is the control stub

`LLM-DIRECTOR.md:393-396` and `PROGRESS.md:1212-1213`:

> For contrast: a title for a *different* experiment scores **0.000** against
> all five of these.

I recomputed all three Jaccard families from
`experiments/director_proposal_stability.json` using the script's own
`jaccard()` and `pairwise_run_similarity()`, and they reproduce **exactly**:

| comparison | median | min | max | doc |
|---|---:|---:|---:|---|
| titles | 0.40179 | 0.09091 | 0.66667 | 0.402 / 0.091 / 0.667 ✓ |
| whole proposal | 0.29193 | 0.0 | 0.375 | 0.292 / 0.000 / 0.375 ✓ |
| unsafe-claim sets | 0.30217 | 0.21519 | 0.49057 | 0.302 / 0.215 / 0.491 ✓ |

The 0.000 is also right — but it is the **control stub's** title. The stub
debrief's `next_experiment` fields are all the literal string `"stub"`
(`director_proposal_stability.py:76-86`), whose content-token set is
`{"stub"}`, which shares nothing with any of the five titles, so the Jaccard is
0.000 against all five. That is what the file's `control` block measures — and
that block records only `n_distinct_titles`, `n_stable_fields` and
`n_distinct_unsafe_claims`; the 0.000 similarities are not stored. So the
sentence describes a deterministic stub as "a title for a different experiment",
and the number it quotes is not in the file.

**Suggested fix** (owner of `director_proposal_stability.py`): record the
control's similarity against the real titles in the `control` block, and reword
to "the control stub's title scores 0.000 against all five".

### C3 — the +0.583 / 40 kB llama run and the −0.083 blind run are not in any file

`LLM-DIRECTOR.md:182-188`:

> llama-3.3-70b scored **+0.583** on one run and **+0.250** on another — same
> model, same ten claims, same shuffled order, different material sizes (the two
> runs were handed 40 kB and 60 kB of notes …). In the BLIND mode … gpt-oss-120b
> scored **−0.083** on its first run and **−0.333** on its second.

The complete committed inventory of debrief rows is 13 distinct rows; the
material sizes recorded in them are 140000 and 60000. There is no 40 kB row and
no gpt-oss blind row with headline −0.083. The **+0.250** (llama, context,
60 kB, `6d1456b`) and the **−0.333** (gpt-oss, blind, `6d1456b`) do exist. The
other two numbers are almost certainly session-internal runs that were never
committed — but a reader cannot check them, and the document's own standard
(now written into `ARTIFACT-REPRODUCIBILITY.md`) is that a number a reader
cannot check is a rumour. **Suggested fix**: commit the two runs, or mark both
numbers as not reproducible from the tree.

### C4 — the test-suite result

`SILENCE-LEADING-ARTEFACT.md:441` records "3 failed, 606 passed, 5 skipped".
A test-run tally is not a result file and is not reproducible by re-running a
script; it is the same class of quantity `ARTIFACT-REPRODUCIBILITY.md` excludes
from its comparison set (`elapsed*`, `cpu_*`, `wall_*`). I did not re-run the
suite. **Unresolved, not wrong.**

### C5 — the timings and the 120-second tally

`ARTIFACT-REPRODUCIBILITY.md:184-191` quotes per-script run times
(`speculation_payoff.py` 389 s, `f0_speaker_change.py` 350 s,
`inthewild_probe.py` 225 s) and the tally at the 120 s default (31
`REPRODUCIBLE`, 21 `NOT_RE_RUN`). These come from the tool's JSON manifest,
which is **not committed** — there is no manifest in the tree and
`/tmp/manifest.json` does not exist. The internal arithmetic is consistent
(31 + 21 + 4 `NO_SCRIPT` + 1 `LEGACY_NO_SCRIPT` = 57; 35 − 31 = 4 = three
converted from `NOT_RE_RUN` plus `pause_threshold_results.json`, which the
120 s run could not attempt at all). **Unresolved without re-running the tool**,
which costs hours.

### C6 — "139 kB of a 372 kB corpus"

`LLM-DIRECTOR.md:534`: "The context materials are 139 kB of a 372 kB corpus".
The committed context artefacts record `materials_chars: 140000`. The 372 kB
figure traces to a source file, not a result file: `experiments/debrief.py:477`
says "With 30 notes totalling 372 kB, that meant SIX notes got in" — a
description of the **pre-fix** state, which is what the sentence is about, and
it is quoted correctly. The "139 kB" does not match the recorded 140000 and may
be the pre-truncation length of the material actually built. Note also that the
note pool is no longer 30 notes: `docs/research/*.md` is now 39 files and
633.5 kB, so the "30 notes" figure is historical in a second way.

---

## 3. Claims checked and found sound

Grouped by where they live. Every item below was traced to a result file, a git
object or a source file and **holds**.

**The pause populations — `PAUSE-THRESHOLD.md`, `SILENCE-LEADING-ARTEFACT.md`,
`ADOPTION-PLAN.md`, `TURN-ROUTES.md`, `src/turn.py`.** These five describe the
same measurement and agree with each other and with
`experiments/pause_threshold_results.json` on every one of these:

- the corpus table — narration 90.000000 s, poetry 20.011125 s, five cells at
  75.000000 s, **total 485.011125 s**; the file counts
  (`narration` 3 969 000 @ 44 100 Hz, `poetry` 320 178 @ 16 000 Hz, the rest
  1 200 000 @ 16 000 Hz);
- the per-cell gap table — narration 45/0.170/0.210/0.640/45/43/2/4.4 %,
  poetry 7/0.150/0.150/0.510/7/6/1/14.3 %, essay 33/0.150/0.160/0.760/33/24/9/27.3 %,
  dialogue 35/0.130/0.160/0.450/34/30/4/11.8 %, shakespeare 62/0.110/0.160/0.380/61/46/15/24.6 %,
  children 78/0.110/0.160/0.300/77/62/15/19.5 %, oration 2/0.310/0.310/0.380/2/2/0/0 %,
  **TOTAL 262 / 259 / 213 / 46 / 17.76 %**;
- the headline — **46 of 259 = 17.760618 %**, not 46 of 262 (17.56 %) and not
  44 of 622 (7.07 %); the denominator table's other two rows are arithmetically
  right as quoted;
- the two published-table conventions — the linear-interpolated p05 on poetry
  (0.255 vs the published 0.150) and on essay (0.166 vs 0.160); the standard
  median vs the published one on shakespeare (0.375 vs 0.380) and oration
  (0.345 vs 0.380); the file stores both conventions and the prose quotes the
  right one each time;
- the leading-silence artefact — dialogue 0.130 s (0.025 → 0.155), shakespeare
  0.110 s (0.025 → 0.135), children 0.110 s (0.025 → 0.135), the essay's
  **real** 0.760 s leading silence, and the artefact-free shortests
  0.160 / 0.150 / 0.150;
- the trailing-silence table, all seven cells, including poetry's **5.100 s**;
- the gap accounting — **262 = 257 inter-utterance + 1 real leading + 3 leading
  artefacts + 1 trailing**, and 257 = 45 + 6 + 32 + 34 + 61 + 77 + 2;
- the Addendum-2 rows on the 673-run population — 170/169/1/99.41 %,
  71/70/1/98.59 %, 35/35/0/100 %, 15/15/0, 6/6/0, 2/2/0, and the energy table
  96/95/1/98.96 % … 6/5/1/83.33 %, both of which I summed from the per-cell
  resume tables and got the same totals;
- the per-cell longest watcher runs — 1.830 / 1.200 / 1.560 / 1.530 / 1.450 /
  1.600 / 2.540 — and the 1671 event count with min 0.160 / median 0.380 /
  max 2.540;
- **the poetry `max_gap` correction in `src/turn.py:37`** — 20.011 s, 1.740 s,
  0 gaps ≥ 3 s. The applied value is poetry's longest **inter-utterance** gap
  (1.740), which is the population the same table's "gaps ≥ 3 s" column
  counts; for the other six cells the whole-file and inter-utterance maxima
  coincide, so the column is now internally consistent. "Six of seven cells
  produce ZERO turn boundaries at 3 s" is right: only the essay has any
  (2). **Sound.**

**Recomputed directly from the corpus** (not from a stored file), because the
note quotes them and the stored file does not hold them: poetry's trailing
silence runs **14.895 s → 19.995 s** of a 20.011125 s file (duration 5.100 s
exactly, on the 10 ms frame grid), and every per-cell shortest inter-utterance
gap in the table above. Both confirmed.

**`SILENCE-LEADING-ARTEFACT.md`.** The whole of §2's mechanism table — buffers
4851, current short-silence opens 576, artefact-free 4, differing pushes 2387,
all four summed and matched per cell; the three absorber durations
(1.06 → 1.19 s dialogue, 0.13 → 0.24 s shakespeare, 1.21 → 1.32 s children);
§4.1's per-cell table and the 17.760618 %; §4.2's watcher table (299 → 222,
59 → 46, … 1671 → 1241 events, 1789 → 1262, 673 → 588 runs, with every max/min/median
run); §4.3's speculation figures (1258/1671 = 75.2843 %, 935/1241 = 75.3425 %,
1578/1671 = 94.4345 %, 1167/1241 = 94.0371 %); §4.5's F0 population (684 → 671
utterance spans, 398 → 353 gaps); §4.6's Addendum-2 rows; and the file's own
`committed_comparison` against `pause_threshold_results.json` — the recorded
SHA-256 `168d2c48…` is the **current** file's SHA-256, `all_agree: true` and
`disagreements: []`. The pause file was regenerated after the leading-silence
findings were written, and the silence results file was written **after** that
regeneration, so the two are consistent with each other.

**`DIALOGUE-INTAKE.md`.** The two corpora (matrix 96.0 s / 8 pairs / 38
silences / 1 hand-off / status NOT YET USABLE / exit 0; protocol-length 615.0 s
/ 221 silences / 5 hand-offs / USABLE / `unmet` empty / exit 0); all **ten**
perturbation rows, including their `reported`/`gated` verdicts and
`unmet_added` lists; the footnote on row 10 (the unreadable channel was
`scene06_interrupt`, and `hand_offs_seen` falls 1 → 0 because it takes the
corpus's only barge-back with it); the cross-talk comparison table end to end
(clean 0.9738–0.9830 on 2 of 8 scenes, corrupted 0.9897–0.9984, separation
**0.0067**; zero-mean correlation −0.0267–0.0063 vs 0.7515–1.0000, separation
**0.7452**; "B loud when A loud" 0.000–0.3125 vs 1.0000, separation **0.6875**);
the alignment table (five offsets, the 5 ms and 100 ms columns, the error
columns, the hand-off flips 1 → 0 at 300 ms and 600 ms, the overlap boundary
3.2/3.2/3.2/0.6/0.9 s); the clap-correction table (peak r **0.7587** at every
offset, residual gap error **0.00 s**, hand-offs restored to 1); the drift
table (1.25 %, 250 ms over 20 s, 187.5 ms at the gap, 0.40 → 0.21 s, error
−0.19 s, clap offset 4 ms, no drift field); the hand-off semantics table
(canonical 0, canonical-later 0, barge-back 2, with the exact pause list
including the 20 ms dip); the level-floor table (23.5 dB, `signal_frame_frac`
→ 1.0, `both_loud_frac` → 1.0, cross-talk 0.968–0.997 with 0.413 in the
room-tone scene, hand-offs 1 → 38, 38 silences in both rooms); and the
proposed-rule counts (8 canonical, 0 today, **7** clap-excluded, **14**
clap-included).

**`LLM-DIRECTOR.md`.** The audit set (7 false / 3 true — `n_false: 7`,
`n_true: 3` in every committed debrief row); the BLIND table (gpt-oss
0.429/1.000/0/3/+0.429/+0.429, qwen 0.143/1.000/0/3/+0.143/+0.286, llama
0.286/0.667/1/3/−0.048/+0.095 — the meta-first file, which is the one the
table quotes); the CONTEXT table (gpt-oss 0.857/1.000/0/3/0/+0.857/+0.857,
llama 0.429/0.667/1/3/1/+0.095/+0.238); the correction block's "row moved from
`0.429 / 0.667 / 1/3 → +0.095` to `0.429 / 1.000 / 0/3 → +0.429`", "qwen3's
lenient moved from +0.143 to +0.286 because one verdict became `uncertain`",
and "llama's row did not move at all" — all three confirmed against the two
blind files; the repeatability table's blind and 140,000-char rows; the control
stub (8 / 6 / 6 control runs, **1** distinct headline, spread **0.000**, and
always-accept and always-reject both at **0.000**); the per-claim stability
table (three of four unstable claims are known-FALSE; the gain-normalisation
claim answered `unsupported` ×3 and `supported` ×3; the conventional-teacher
claim ×4/×2; the composed-filters claim supported ×3 and unparseable ×3; the
blind arm's four unstable claims with the ASCII claim at unsupported ×4 /
uncertain ×4 and the row-median claim carrying two false accusations in eight
runs); the majority-vote table (blind 0.619/0.619/0.619/0.485, context-140k
0.619/0.619/0.619/0.507, context-60k 0.333/0.333/0.333/0.470) and the
"probability 0.485–0.507 of being the mode"; the proposal-stability block
(prompt 143,852 chars, SHA `9ccc4179ea39bdfb`, 6 runs, 5 distinct titles,
agreement 0.167, 5 distinct values in every field, 5 of 6 complete proposals,
37 unsafe claims / 34 distinct / top agreement 0.054, the five titles, and the
three recurring unsafe claims — the ASCII labeling claim, the ASCII-motion
claim and the fifty-fold gain claim, each with count 2); §5's corrected
interpretation of the 2.99 % figure (`0.0299` is `A_flip`; row removal's real
empty saturation **0.48839** and row+median3's **0.034787** — both in
`combined_filter_results.json`); and §8's claim-set description (7 false, 3
true, 140 000 chars of material, the 8-repeat and 6-repeat headline sets).

**`ARTIFACT-REPRODUCIBILITY.md`.** **57** committed artefacts at `c71a6c5`
(46 `*_results.json` + 11 `teacher_conventional_*.json`), matching the
§5.1 table row for row and the §5 verdict counts (35 / 0 / 4 / 17 / 1); the
0.675 chain — `415ecb2` holds **0.65**, `7a4dff5` holds **0.675** with
`acc_test_spatially_scrambled` **1.0** and `margin_over_best_control` **0.0**,
`d6baa6d` and `c71a6c5` and `HEAD` hold **0.000** with scrambled
**0.9791666666666666** and margin **0.02083333333333337**; the script blobs
`459e23c0…` at `7a4dff5` and `f8f89c96…` from `d6baa6d` on; the legacy
orphan's contents (acc_test 1.0, acc_test_shuffled 0.9375, acc_test_shuffled_train
0.986, acc_test_blank 0.667, margin 0.0625, `usable` false, LOO A 0.0 and B 0.0)
and that no invocation of `teacher_conventional.py` can write that filename;
§9's two filename-permutation files and their differing gpt-oss rows
(`specificity_kept_true_claims` **1.0** vs **0.666…**, `false_positives` **0**
vs **1**) and the historical claim sets (6 false / 4 true at `6d1456b`, 7 / 3 at
`1c564c5`, both files now 7 / 3); §11's line references, every one of them
landing on a line that does quote the 0.675 (`TEACHER-PIPELINE.md` 102, 109,
113, 114, 115, 122; `NOTES-AUDIT.md` 81, 84, 101, 106; `COMBINED-FILTER.md`
99, 109, 110; `LLM-DIRECTOR.md` 53, 60) and the replacement values at
`TEACHER-PIPELINE.md` 70, 82, 104 and `COMBINED-FILTER.md` 87; and the **17
tests** in `tests/test_artifact_repro.py`, all of which collect.

**`PROGRESS.md`'s newest arc.** The 0.675 withdrawal and the LOO table
(`summary` A 0.000/B 0.000, `region` A 0.000/B 0.000, `map` A 0.000/B **1.000**,
`both` A 0.000/B 0.000, `region+map` A 0.000/B **0.200**); `src/energy.py:290`
being the `kept = [segs[0]]` line that lets the first segment escape the
filter; the artefact-free shortest inter-utterance pause **0.150 s**; the
673 runs against 680 at **1.03 %**; the poetry closing of the other end —
**5.100 s** from 14.895 s to 19.995 s of a 20.011 s file, poetry's longest
inter-utterance gap **1.740 s**, and the full 262 = 257 + 1 + 3 + 1 accounting;
and the proposal-stability block (5 distinct titles of 6 runs, every field
varies, one incomplete, 37 unsafe claims / 34 distinct, title Jaccard median
**0.402**, unsafe-claim Jaccard median **0.302**).

**`NOTES-AUDIT-2.md`, audited as the audit.** Its own headline numbers hold:
**49** documents, **16** findings, **10** confirmed; F1's arithmetic
(798 995 ÷ 217 800 = **3.67×**); F5/F6's spread table — recomputed from
`jev_voiceloop_results.json` E6, the per-axis spreads are `timing` **0.000**,
`prosody` **0.020**, `realtime_safe` **0.010**, `orthogonal` **0.010**, so
"only `timing` is bit-identical" is right and the applied corrections in
`voiceloop/JEV-AS-INSTRUMENT.md:256-261` and `voiceloop/README.md:288-300`
say exactly that; F7/F8's 6/9 → 3/9 — the reversed blocks give A/B **0/3**,
C/D **2/3**, E/F **1/3** = **3/9**, the forward blocks 9/9, and the "6" is
indeed the content-anchored count (3 + 1 + 2 = 6), with
`picked_first_option_rate = 0.5` confirmed as "3 of the 6 order-blocks were
all-first" at `jev_voiceloop.py:659`; and the applied corrections in
`voiceloop/README.md:266-276` and `JEV-AS-INSTRUMENT.md:153-161` reproduce all
of it. **F5, F6, F7 and F8 are correct and their corrections are correct.**

---

## 4. Claims I could not resolve

1. **The test-suite tally** in `SILENCE-LEADING-ARTEFACT.md:441` ("3 failed,
   606 passed, 5 skipped"). Not re-run here; it is a machine-and-load
   property, not a measurement.
2. **The `artifact_repro.py` run times and the 120 s-default tally** in
   `ARTIFACT-REPRODUCIBILITY.md:184-191`. The manifest is not committed and
   re-running the tool costs hours; the internal arithmetic is consistent.
3. **The corpus rebuild claim** — `PROGRESS.md:1078-1080`, "The corpus rebuilds
   **byte for byte** from the captures (`seed_corpus_from_captures.py --filter
   row+median3` produces three SHA-256-identical `.npz` files)". The corpus is
   on disk at `/mnt/c/proart_capture`, but the rebuild was not re-run here, and
   `ARTIFACT-REPRODUCIBILITY.md:450-453` explicitly declines to re-run it too.
   The claim is asserted flatly in `PROGRESS.md` and in `NOTES-AUDIT.md:95-97`
   without a result file behind it. **This is the one claim in the newest
   session arc I could not check**, and it is load-bearing: it is one of the two
   legs of "neither the code nor the data is the variable".
4. **The "352 configurations" sweep** (`PROGRESS.md:1083-1084`,
   `NOTES-AUDIT.md:100`, `TEACHER-PIPELINE.md:120`, `COMBINED-FILTER.md:107`).
   No sweep script and no sweep artefact are committed; the number appears in
   four documents with the same value, which is consistent with one uncommitted
   session run.
5. **The Windows bundle byte-for-byte claim** (`PROGRESS.md:1176-1178`) — the
   other machine is not reachable from here.
6. **The +0.583 / 40 kB llama run and the −0.083 blind gpt-oss run**
   (`LLM-DIRECTOR.md:182-188`). In no committed file; see C3.
7. **The 24 000-character budget, "30 notes totalling 372 kB" and "six notes
   got in"** (`LLM-DIRECTOR.md:455-462`). These trace to `experiments/debrief.py:476-481`,
   which is a source file rather than a result file, and they describe a
   **historical** state; the note pool is now 39 top-level files and 633.5 kB.
8. **The `proposal`-stability `field_agreement` values**
   (`director_proposal_stability.json`): each field's `agreement` is **0.2**,
   i.e. the modal value appears once among the five present values, on the five
   of six runs that produced the field. Internally consistent with the file's
   own `fields` block; no document quotes it, so not checked further.
9. **`experiments/teacher_conventional_*.json` — the other nine feature sets.**
   Their LOO values were read and are quoted correctly by
   `ARTIFACT-REPRODUCIBILITY.md` §10; they were not individually re-run.

---

## 5. Process note

The single most productive check in this round was not a number at all. It was
re-running `pause_threshold.py`'s **own** `_EventProbe` and `_runs_from_events`
with `adaptive=False` and checking the harness against the committed
adaptive population first: seven of seven cells matched exactly, 673 runs in
total. Only then did a one-run disagreement (215 vs 216) mean anything.

The second was reading the *commits* rather than the files. F5, F6, F7 and F10
are all the same defect seen four times — a note describing a state a later
commit moved past — and in each case the note's own text or the commit log
said so. `git log --oneline -- <file>` and `git show <ref>:<file>` answered in
seconds what re-deriving the measurement could not.

The third is the one to keep: the corrections themselves were checked against
their sources and none of them introduced a new error. The poetry `max_gap`
row, the round-two voiceloop edits, the `ADOPTION-PLAN.md` pause table, the
`SILENCE-LEADING-ARTEFACT.md` tables — every one holds. That is the result
that matters most in an audit of corrections, and it is the one the `0.675`
story could not deliver.
