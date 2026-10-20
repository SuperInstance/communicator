# PAUSE-THRESHOLD — the pause statistics, re-measured on the corpus that is on disk

**What this is.** `docs/research/ADOPTION-PLAN.md` §3 and
`docs/research/TURN-ROUTES.md` carry a pause-statistics table used to argue
that raising the VAD trigger from 150 ms to 200 ms destroys 17.8% of real
pauses. Those numbers used to live only in a test docstring
(`tests/test_pause_threshold.py`) and in prose. There was no result file, so
the numbers could not be traced to the audio they claimed to describe.

This is that result file's report. The measurements are in
`experiments/pause_threshold_results.json`, produced by
`experiments/pause_threshold.py`, and held to the corpus by
`tests/test_pause_threshold_stats.py`.

**The short version.**

1. The corpus is **485.011125 s** on disk, not the 465 s the prose claimed.
   The 20.011 s gap is exactly the poetry cell. (`NOTES-AUDIT.md` F7.)
2. **Every number in the published pause table reproduces exactly** on the
   current corpus. The table was measured on a population whose LENGTH was
   misstated; the COUNTS are not stale.
3. **The 17.8% survives**: 46 of the 259 pauses the 150 ms trigger keeps are
   shorter than 200 ms. The number is right; the phrase "of all real pauses"
   is not — the denominator is 259, not everything.
4. A **defect in the copied method** was found and is reported, not hidden:
   the file's *leading* silence escapes the trigger filter, which is where the
   published "shortest pause is 110 ms" comes from. It is not an
   inter-utterance pause. The headline does not depend on it.
5. The "**680 vs 1671**" pause-run discrepancy (`NOTES-AUDIT.md` F8) is
   **resolved**: 1671 is an *event* count and reproduces exactly; the
   distinct silence-*run* count is **673**, within 1.03% of the 680, which was
   a run count measured on a population this repository no longer has.

---

## 1. The method, and exactly what it was copied from

| | |
|---|---|
| Source | `tests/test_pause_threshold.py` |
| Gap helper | `_gaps()`, lines **90–99** — copied **byte for byte** |
| Corpus mapping | `CELLS`, lines **77–87** — copied **byte for byte** |
| Checked at run time | `experiments/pause_threshold.py --check-copy` |

A different gap definition would produce a different number, and a different
number would be a *different measurement*, not a re-measurement. The copy is
therefore byte-exact, and the script re-verifies it against the source file
every time it runs, so the two cannot drift apart silently.

The copied method, unchanged:

```python
def _gaps(path: Path):
    """Every silence run the segmenter finds, in seconds."""
    if not path.is_file():
        pytest.skip(f"corpus cell absent: {path.name}")
    samples = wavio.read_wav(str(path)).samples
    ef, _ = energy.rms_series(samples, 800, HOP, SR)
    if not ef:
        pytest.skip(f"{path.name} produced no frames")
    segs = energy.segment(ef, HOP_S, peak_rms=max(f.rms for f in ef))
    return [s.duration() for s in segs if not s.speech]
```

The exact parameters, all recorded in the result file:

| parameter | value | where |
|---|---|---|
| sample rate | 16000 Hz | `wavio.ANALYSIS_SR` (the reader resamples) |
| frame length | 800 samples (50 ms) | `energy.rms_series` |
| hop | 160 samples (10 ms) | `HOP` |
| frame grid | `energy.rms_series(samples, 800, 160, 16000)` | copied |
| segmenter | `src/energy.py:segment()` | copied |
| speech-on bar | 10% of the cell's own peak frame RMS | `energy.RELATIVE_ON` |
| `min_silence_s` | **0.150** | `energy.MIN_SILENCE_S` |
| `min_speech_s` | 0.060 | `energy.MIN_SPEECH_S` |
| adaptive threshold | **off** | the copied call does not pass `adaptive=True` |
| a "gap" | the duration of a non-speech segment | copied |

**Statistics conventions** (a percentile is a choice, and an unstated choice
is how a table stops meaning anything):

* **p05 — nearest rank**, `sorted[ceil(0.05·n) − 1]`. This is the convention
  that reproduces the published table. On poetry (n = 7) linear
  interpolation gives 0.255 where the published table says 0.150; on essay
  (n = 33) 0.166 against the published 0.160. Both values are recorded.
* **median — `sorted[n // 2]`**, the *upper* middle value for an even n. This
  is what the published table used, and it is identical to the standard
  median for every odd n, which is six of the seven cells. It differs on
  shakespeare (published 0.380, standard 0.375) and oration (published
  0.380, standard 0.345). Both values are recorded.

---

## 2. The corpus, measured on disk

Read from the WAV headers, not from any document:

| cell | file | frames | file rate | duration |
|---|---|---|---|---|
| narration | `examples/realaudio/alice01_excerpt.wav` | 3 969 000 | 44 100 Hz | 90.000000 s |
| poetry | `examples/realaudio/corpus/poetry_weeds_by_the_wall_1702_librivox.wav` | 320 178 | 16 000 Hz | 20.011125 s |
| essay | `examples/realaudio/corpus/essay_perpetual_peace_librivox.wav` | 1 200 000 | 16 000 Hz | 75.000000 s |
| dialogue | `examples/realaudio/corpus/shortstory_shortstory058_1405_librivox.wav` | 1 200 000 | 16 000 Hz | 75.000000 s |
| shakespeare | `examples/realaudio/corpus/shakespeare_midsummernightsdream_1201_librivox.wav` | 1 200 000 | 16 000 Hz | 75.000000 s |
| children | `examples/realaudio/corpus/children_english_fairy_tales_joy_librivox.wav` | 1 200 000 | 16 000 Hz | 75.000000 s |
| oration | `examples/realaudio/corpus/oration_jfks19620912.wav` | 1 200 000 | 16 000 Hz | 75.000000 s |
| **TOTAL** | | | | **485.011125 s** |

| claim | value | verdict |
|---|---|---|
| the prose's corpus length | 465 s | **wrong by 20.011 s** — exactly the poetry cell |
| the audit's corpus length | 485.01 s | **confirmed** (485.011125 s) |

The narration cell is 44.1 kHz on disk and is resampled to 16 kHz by
`wavio.read_wav`; the duration is unchanged by the resample (90.000 s).

---

## 3. The measured table

Measured with the copied method, verbatim, on the corpus above.

| cell | gaps | shortest | p05 | median | @0.150 | @0.200 | lost | lost% |
|---|---|---|---|---|---|---|---|---|
| narration | 45 | 0.170 | 0.210 | 0.640 | 45 | 43 | 2 | 4.4% |
| poetry | 7 | 0.150 | 0.150 | 0.510 | 7 | 6 | 1 | 14.3% |
| essay | 33 | 0.150 | 0.160 | 0.760 | 33 | 24 | 9 | 27.3% |
| dialogue | 35 | 0.130 | 0.160 | 0.450 | 34 | 30 | 4 | 11.8% |
| shakespeare | 62 | 0.110 | 0.160 | 0.380 | 61 | 46 | 15 | 24.6% |
| children | 78 | 0.110 | 0.160 | 0.300 | 77 | 62 | 15 | 19.5% |
| oration | 2 | 0.310 | 0.310 | 0.380 | 2 | 2 | 0 | 0.0% |
| **TOTAL** | **262** | | | | **259** | **213** | **46** | **17.8%** |

**Every cell of this table is identical to `ADOPTION-PLAN.md:126-145`.** The
machine-checked diff is in the result file at
`published_table_comparison`, and it is empty — per cell and in total. So the
published counts are **not stale**: they were measured on the corpus that is
on disk. What was wrong was the corpus *length* stated alongside them.

The **median** column is the published convention (`sorted[n//2]`). The
standard median differs in two cells: shakespeare **0.375** and oration
**0.345**.

### The headline

**17.8% survives.** 46 of the 259 pauses the 150 ms trigger keeps are
shorter than 200 ms: 46 / 259 = **17.76%**, which rounds to the published
17.8%.

The denominator has to be stated, and this is where the published wording is
loose:

| population | count | 46 lost as a share of it |
|---|---|---|
| the 259 pauses the **150 ms trigger** keeps | 259 | **17.8%** ← the published figure |
| all 262 silence runs the method returns | 262 | 17.6% |
| all 622 silence segments with no `min_silence_s` filter at all | 622 | 7.1% (44 lost) |

"17.8% of all real pauses disappear" is therefore true of the pauses the
shipped trigger reports, and of nothing larger. The claim survives; the
phrase "all real pauses" does not.

---

## 4. A defect in the copied method — reported, not hidden

`src/energy.py:290` keeps the first segment of a file unconditionally:

```python
kept = [segs[0]]                       # src/energy.py:290
```

so the first segment escapes the `min_silence_s` filter entirely. On this
corpus the first segment is the **silence before the recording starts**, and
it is therefore reported as a gap:

| cell | leading silence | published "shortest" | shortest *inter-utterance* pause |
|---|---|---|---|
| dialogue | 0.130 s (0.025 → 0.155) | 0.130 | **0.160** |
| shakespeare | 0.110 s (0.025 → 0.135) | 0.110 | **0.150** |
| children | 0.110 s (0.025 → 0.135) | 0.110 | **0.150** |

The other four cells start with speech and are unaffected. In every cell the
only silence shorter than the 0.150 s trigger is that leading one — the test
`tests/test_pause_threshold_stats.py` asserts exactly that, so a second
instance of the same defect cannot creep in unnoticed.

**Both measurements are reported, and the definition was not changed.**

| | verbatim method | leading artefact removed |
|---|---|---|
| gaps | 262 | 259 |
| @0.150 | 259 | 259 |
| @0.200 | 213 | 213 |
| lost | 46 | 46 |
| **lost%** | **17.76%** | **17.76%** |

The artefact is below 0.150 s, so the 0.150 trigger already excludes it. The
`@0.150` / `@0.200` / `lost` columns and the headline are **identical with
and without it**.

What it does change is the "shortest" column and the sentence built on it:

> the shortest pause a human voice actually takes is **110 ms**

That 110 ms is the silence *before the first word*, not a pause between
utterances. On the artefact-free population the shortest inter-utterance
pause in the corpus is **150 ms** — in poetry, essay, shakespeare and
children — which is *at* the shipped trigger, not below it.

### The other end of the file: trailing silence

The leading artefact is only half the edge problem, and the other half was
found later, while checking a claim about poetry. **The published gap list
is every SILENCE RUN in the file, which includes the silence after the last
word.** Poetry is the cell where it matters:

| cell | published gaps | inter-utterance gaps | leading silence | trailing silence | max gap (all) | max gap (inter-utterance) | gaps ≥ 3 s (all) | gaps ≥ 3 s (inter-utterance) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| narration | 45 | 45 | — | — | 1.980 | 1.980 | 0 | 0 |
| poetry | 7 | **6** | — | **5.100** | **5.100** | **1.740** | **1** | **0** |
| essay | 33 | 32 | 0.760 | — | 4.530 | 4.530 | 2 | 2 |
| dialogue | 35 | 34 | 0.130 | — | 1.130 | 1.130 | 0 | 0 |
| shakespeare | 62 | 61 | 0.110 | — | 2.100 | 2.100 | 0 | 0 |
| children | 78 | 77 | 0.110 | — | 1.960 | 1.960 | 0 | 0 |
| oration | 2 | 2 | — | — | 0.380 | 0.380 | 0 | 0 |

Poetry's 5.100 s silence runs from **14.895 s to 19.995 s of a 20.011 s
file** — it is the recording stopping. Poetry's longest *inter-utterance*
gap is **1.740 s**, so poetry produces no turn boundary at 3 s, like the
other five cells.

The full accounting closes exactly: **262 published gaps = 257
inter-utterance + 1 real leading silence (essay 0.760 s) + 3 leading
artefacts (dialogue 0.130, shakespeare 0.110, children 0.110) + 1 trailing
silence (poetry 5.100).**

**Why this is worth stating rather than fixing silently.** Two tables in
this repository count gaps and they use different populations, and always
have: the *pause* table counts silence runs, and the *turn* table
(`src/turn.py`) counts gaps with speech on both sides. Poetry has **1** gap
≥ 3 s in the pause table and **0** in the turn table. That is not a
contradiction — it is two questions, and the reason the numbers disagree
was never written down anywhere until now.

The measurement is in `experiments/pause_threshold_results.json` under
`cells.<name>.edge_silences`, and the edge window is one analysis frame
(0.05 s), so a silence has to actually touch the file edge to count as one.

---

## 5. The "680 vs 1671" pause-run question (`NOTES-AUDIT.md` F8)

**Verdict: the disagreement is an events-vs-runs confusion, and the 680 is
within 1% of today's run count but is not exactly reproducible.**

`tests/test_speculate.py:409-426` counts **PAUSE events** from the streaming
`Watcher` with `run_s > 0.15`. The watcher reports an *open* pause on a
fixed cadence — `PAUSE_REPORT_S = 0.25 s` (`src/watcher.py:202`) — so **one
silence run produces several events**. That is the whole disagreement: the
two numbers count different things.

Measured today on the current corpus:

| measurement | count | min | median | max |
|---|---|---|---|---|
| pause **events**, `run_s > 0.15` (what the repo measures) | **1671** | 0.160 | 0.380 | 2.540 |
| pause events, no filter | 1789 | | | |
| distinct silence **runs** | **673** | 0.010 | 0.350 | 2.540 |

The 1671 and its min / median / max reproduce
`tests/test_speculate.py:99-100` **exactly**. The distinct-run count is
**673**, and the prose's 680 is 7 runs away from it (1.03%).

The 680 is not exactly reproducible under any method tried:

| candidate population | count |
|---|---|
| watcher pause **events**, `run_s > 0.15`, adaptive | 1671 |
| watcher pause events, no filter, adaptive | 1789 |
| watcher **distinct runs**, adaptive | **673** |
| watcher pause events, `run_s > 0.15`, energy-only (adaptive off) | 647 |
| watcher pause events, no filter, energy-only | 664 |
| watcher distinct runs, energy-only | 215 |
| the energy method's silence runs @0.150 (the pause table's population) | 262 |
| the energy method's silence runs with no filter | 622 |
| the same, on a hypothetical 465 s corpus (poetry truncated to 15 s) | 1656 events / **665** runs |

The reading that fits all the evidence: **680 was a run count, 1671 is an
event count, and the 1.03% difference between 680 and today's 673 is watcher
drift, not corpus drift** — truncating the poetry cell to the 15 s the
465 s figure implies moves the run count *down* to 665, further from 680,
not up towards it. The population behind 680 no longer exists in this
repository, so the figure is reported as *not exactly reproducible* — and
every rate computed on it is on a population that cannot be re-derived.

One more piece of evidence for that reading: the published Addendum-2 table's
three highest-threshold rows (1.5 s → 15 runs, 1.8 s → 6, 2.1 s → 2)
**reproduce exactly** on today's run population. The three lowest-threshold
rows do not.

### Addendum 2 of `TURN-ROUTES.md`, re-measured

The Addendum-2 table is computed on the pause-run population, so it moves
with it. Re-measured on the **watcher run population** (673 runs) — the
like-for-like replacement, and the population the published table was on:

| threshold | silence runs | speaker resumed | at file end | would have interrupted | published |
|---|---|---|---|---|---|
| 0.6 s | 170 | 169 | 1 | 99.41% | 54 / 53 / 0 / 98.1% |
| 0.9 s | 71 | 70 | 1 | 98.59% | 38 / 38 / 0 / 100.0% |
| 1.2 s | 35 | 35 | 0 | 100.0% | 24 / 24 / 0 / 100.0% |
| 1.5 s | 15 | 15 | 0 | 100.0% | 15 / 15 / 0 / 100.0% |
| 1.8 s | 6 | 6 | 0 | 100.0% | 6 / 6 / 0 / 100.0% |
| 2.1 s | 2 | 2 | 0 | 100.0% | 2 / 2 / 0 / 100.0% |

And on the **energy population** — the same population as the pause table in
§3, where "did the speaker come back" is exactly defined by the segmentation
(a silence run that is not the last segment of the file is followed by
speech; the last one ran to the end of the recording and is ambiguous):

| threshold | silence runs | speaker resumed | at file end | would have interrupted |
|---|---|---|---|---|
| 0.6 s | 96 | 95 | 1 | 98.96% |
| 0.9 s | 51 | 50 | 1 | 98.04% |
| 1.2 s | 28 | 27 | 1 | 96.43% |
| 1.5 s | 16 | 15 | 1 | 93.75% |
| 1.8 s | 9 | 8 | 1 | 88.89% |
| 2.1 s | 6 | 5 | 1 | 83.33% |

The single run "at file end" at every threshold of the energy table is the
poetry cell's 5.100 s trailing silence — the recording stopped, so it is
neither a turn end nor an interruption.

**The conclusion of Addendum 2 still holds, and is now verified on a
population that can be re-derived:** in recorded speech, silence is
essentially never a turn end. At every threshold from 0.6 s up, at least
98% of the silence runs were followed by the speaker coming back, and the
one exception is the end of a recording.

---

## 6. Provenance

Every number above is traceable to its inputs. The result file carries, for
every input wav: the absolute path, the SHA-256, the byte size, the file's
**own** header (sample rate, frame count, duration, channels, sample width
— read from the WAV header, not from the reader, because
`wavio.read_wav` resamples to 16 kHz and the narration cell is 44.1 kHz on
disk), and the analysis-rate sample count. It also carries the sample rate
and the exact gap-detection parameters, the SHA-256 of the script that
produced it, and the source file and line range of the method. The gap list
itself is persisted, together with the SHA-256 of its exact bytes, so the
counts can be re-derived from the list without re-running the segmentation.

```json
{
  "script": {
    "path": "/home/proart/projects/communicator/experiments/pause_threshold.py",
    "sha256": "<recorded in experiments/pause_threshold_results.json>"
  },
  "method": {
    "source_file": "tests/test_pause_threshold.py",
    "gaps_helper_lines": "90-99",
    "cells_lines": "77-87",
    "copied_verbatim": true,
    "copy_check": {"gaps_helper_identical": true, "cells_identical": true, "ok": true},
    "parameters": {
      "sample_rate_hz": 16000, "frame_len_samples": 800, "hop_samples": 160,
      "hop_s": 0.01, "min_silence_s": 0.15, "min_speech_s": 0.06,
      "peak_rms": "max(f.rms for f in ef) -- the cell's own peak frame",
      "adaptive_threshold": false
    }
  },
  "cells": {
    "<cell>": {
      "input": {
        "path": "/home/proart/projects/communicator/examples/realaudio/...",
        "sha256": "<sha256 of that wav>",
        "bytes": 0,
        "file_sample_rate": 0, "file_samples": 0, "file_duration_s": 0.0,
        "file_channels": 1, "file_sample_width_bytes": 2, "resampled": false,
        "analysis_sample_rate": 16000, "analysis_samples": 0,
        "analysis_duration_s": 0.0
      },
      "gaps": {
        "n": 0, "shortest_s": 0.0, "p05_s_nearest_rank": 0.0,
        "p05_s_linear": 0.0, "median_s_standard": 0.0,
        "median_s_published_convention": 0.0, "max_s": 0.0,
        "durations_s": [0.0], "durations_sha256": "<sha256 of the exact bytes>"
      },
      "trigger": {"at_0_150": 0, "at_0_200": 0, "lost": 0, "lost_pct": 0.0}
    }
  }
}
```

Regenerate with:

```
/home/proart/projects/glasses/.venv/bin/python experiments/pause_threshold.py
```

The measurement is deterministic; only `generated_utc` and the elapsed times
change between runs.

---

## 7. Stale numbers — what has to be corrected, and where

**These files were not edited.** The corrections below are for the parent
agent to apply; a note cannot fix a docstring, and `docs/research/ADOPTION-PLAN.md`
and `docs/research/TURN-ROUTES.md` are outside this lane.

### `docs/research/ADOPTION-PLAN.md`

| line | what it says | what the measurement says | action |
|---|---|---|---|
| **:121** | "the shortest pause a human voice actually takes is **110 ms**, in dramatic speech" | 0.110 s is the **leading silence before the first word** of the shakespeare and children cells, an artefact of `src/energy.py:290`. The shortest *inter-utterance* pause in the corpus is **0.150 s** (poetry, essay, shakespeare, children). | **stale — replace.** Suggested: "the shortest *inter-utterance* pause in the corpus is **150 ms**, exactly at the shipped trigger" |
| **:129** | dialogue "shortest 0.130" | 0.130 s is the leading silence; the shortest inter-utterance pause in dialogue is **0.160** | **stale** (or relabel the column "shortest reported gap") |
| **:130** | shakespeare "shortest **0.110**" | leading silence; shortest inter-utterance pause is **0.150** | **stale** |
| **:131** | children "shortest **0.110**" | leading silence; shortest inter-utterance pause is **0.150** | **stale** |
| **:126–132** | "Real inter-utterance gaps by cell" — the *gaps* column (45/7/33/**35**/**62**/**78**/2) | the verbatim method's output is exactly this, but three of the counts include the leading artefact; the inter-utterance counts are 45/7/33/**34**/**61**/**77**/2 | **reword or footnote** the column heading; the counts themselves are a correct measurement of the copied method |
| **:145** | TOTAL **259 / 213 / 46 (17.8%)** | confirmed exactly | **not stale** — but state the denominator: 46 of the **259** pauses the 150 ms trigger keeps |
| **:147** | "**17.8% of all real pauses disappear.**" | 17.8% is 46/259 — the pauses the **150 ms trigger** reports. Over all 262 silence runs it is 17.6%. | **denominator is wrong — needs a stated population** |
| **:154** | "150 ms is measured to sit *below* the shortest real pause in every cell" | on the artefact-free population 150 ms sits **AT** the shortest real pause (0.150 s) in poetry, essay, shakespeare and children, and above it elsewhere. It is never above, but it is not below. | **stale wording — "at or below"** |
| **:337** | "the 200 ms trigger half costs 17.8% of real pauses" | the number survives (46/259) | **not stale** — add the denominator |
| — | no corpus length is quoted in §3 | the corpus is **485.011 s** | nothing to correct here (the 465 s was in `TURN-ROUTES.md`) |

### `docs/research/TURN-ROUTES.md`

| line | what it says | what the measurement says | action |
|---|---|---|---|
| **:12** | "**680 measured pause runs**" | the 680 was a **run** count; the repo's 1671 is an **event** count (the watcher reports an open pause every 0.25 s). Today's distinct run count is **673** — 1.03% from 680. The 680 is not exactly reproducible on this corpus. | **stale — replace with "673 distinct silence runs (1671 pause events)" and say which is which** |
| **:12** | "485 s" | 485.011125 s | **correct** (the correction block at :24-33 already says so) |
| **:17** | poetry "20.0 s" | 20.011125 s | **stale in the third decimal** — write 20.011 s |
| **:16–22** | "longest pause run" per cell: 1.810 / 1.180 / 1.230 / 0.980 / 1.430 / 1.580 / 2.440 s | measured on the current watcher: narration **1.830**, poetry **1.200**, essay **1.560**, dialogue **1.530**, shakespeare **1.450**, children **1.600**, oration **2.540** | **stale — all seven are on the 680 population** |
| **:287–290** | "narration 1.830 s / oration 2.540 s" | confirmed | **not stale** |
| **:345–352** | the Addendum-2 table: 54 / 38 / 24 / 15 / 6 / 2 silence runs and 98.1% / 100.0% ×5, "at file end" 0 throughout | re-measured on the 673-run population: 170 / 71 / 35 / **15 / 6 / 2** and 99.41% / 98.59% / 100.0% ×4, "at file end" **1** at 0.6 s and 0.9 s. The 1.5/1.8/2.1 rows reproduce exactly; the 0.6/0.9/1.2 rows do not. | **stale — replace the three low rows and the "at file end" column** (numbers in §5 of this report) |
| **:354–366** | the correction block saying the Addendum-2 rates "are rates on a different population" and "what has to be re-measured" | **now re-measured** — see §5 above | **update the block to point at this report** |
| **:373–374** | "the longest silence in 485 s of seven cells is 2.54 s" | confirmed (oration, 2.540 s) | **not stale** |
| **:38–40** | "**1671 pause runs** … min 0.160 s / median 0.380 s / max 2.540 s" | the 1671 and the distribution reproduce exactly, but 1671 is an **event** count, not a run count | **reword: "1671 pause EVENTS"** |

### Also stale, outside this lane's write access

`NOTES-AUDIT.md` F7 lists six code and test sites that still hard-code the
wrong 465 s corpus length: `src/turn.py:13`, `src/speculate.py:6`,
`src/watcher.py:661`, `tests/test_turn.py:25`, `tests/test_speculate.py:7`,
`experiments/turn_routes.py:8`. None of them were touched here.

`experiments/turn_routes.py:8` also says "across 680 measured pause runs the
longest is 2.1 s" — both the population and the 2.1 s figure are stale; the
longest run on the current corpus is 2.540 s (oration), as
`TURN-ROUTES.md:289` already records.

`experiments/turn_routes.py:125` chooses its silence-length sweep from
"the MEASURED pause distribution in tests/test_pause_threshold.py: real
inter-utterance gaps run 0.11-0.64 s". The 0.11 s end of that range is the
leading-silence artefact; the inter-utterance range starts at 0.15 s.

---

## 8. What was NOT changed

* `tests/test_pause_threshold.py` — the load-bearing regression test, owned
  by another agent. It was read and copied from, never modified.
* `docs/research/ADOPTION-PLAN.md` and `docs/research/TURN-ROUTES.md` — read
  and audited, never modified. §7 lists what they need.
* `src/energy.py` — the `kept = [segs[0]]` defect is **reported, not
  patched**. The measurement is done both ways and the headline does not
  depend on it. Patching `src/energy.py` would change the numbers in
  `tests/test_pause_threshold.py`, which is another agent's tree.
* Nothing was committed, staged, or pushed.
