# The leading-silence artefact: what it would cost to fix

*Written after `experiments/silence_leading_artefact.py`, which measures the
blast radius of the defect and changes nothing. `src/energy.py` is exactly as
it was before this file existed.*

## 1. The defect, precisely

`src/energy.py`, `_merge_short_runs`, lines 284-306:

```python
def _merge_short_runs(segs: list[Segment], min_silence_s: float,
                      min_speech_s: float) -> list[Segment]:
    """Drop silences shorter than min_silence_s and speech shorter than
    min_speech_s by merging them into their neighbours, then re-join."""
    if not segs:
        return []
    kept = [segs[0]]                    # <-- src/energy.py:290
    for s in segs[1:]:
        prev = kept[-1]
        if s.speech and s.duration() < min_speech_s:
            continue  # blip disappears into surrounding silence
        if (not s.speech) and s.duration() < min_silence_s:
            prev.end = s.end  # gap disappears into surrounding speech
            continue
        kept.append(s)
    # re-join adjacent segments of the same state created by the merges
    joined = [kept[0]]
    for s in kept[1:]:
        if s.speech == joined[-1].speech:
            joined[-1].end = s.end
        else:
            joined.append(s)
    return joined
```

`kept = [segs[0]]` (line 290) keeps the **first** segment of the list
unconditionally. Every other segment passes through two filters — a silence
shorter than `min_silence_s` (0.150 s) is merged into the segment before it, a
speech segment shorter than `min_speech_s` (0.060 s) is dropped — and the first
segment passes through neither.

On a whole file, "the first segment" is the silence before the first word.
`docs/research/PAUSE-THRESHOLD.md` already established that this is where the
published "the shortest pause a human voice actually takes is 110 ms" comes
from, and that the artefact-free shortest inter-utterance pause is 0.150 s —
*at* the shipped trigger, not below it.

What had NOT been established is what else the line is holding up. That is
what this document is for.

## 2. Method

Two populations of segments, from the same audio, the same parameters and the
same code path, differing in exactly one rule.

**`current`** — the shipped `src/energy.py:_merge_short_runs`, unmodified.

**`artefact_free`** — the same function with the leading-segment rule
changed: a first segment that is a silence shorter than `min_silence_s` is
merged **forward** into the segment that follows it (that segment's start
moves back to the silence's start), exactly as a short silence *with* a
predecessor is merged into that predecessor at `src/energy.py:296`. The
variant lives in `experiments/silence_leading_artefact.py` and is installed
only while a measurement runs, by replacing `energy._merge_short_runs` for the
duration of the call — `src/energy.py` is not edited, and the shipped function
is restored afterwards. The variant's own line-by-line difference from the
shipped function is:

| | shipped | artefact-free |
|---|---|---|
| first segment | kept unconditionally | merged forward if it is a short silence |
| every other segment | filtered, then merged into its predecessor | unchanged |
| re-join step | unchanged | unchanged |

One edge case is recorded rather than hidden: the segment that absorbs the
leading silence becomes the first segment of the list, and the shipped
algorithm keeps the first segment unconditionally, so the absorber inherits
the exemption from the `min_speech_s` filter. On this corpus the exemption is
**not load-bearing** — the absorber is 1.06 s (dialogue), 0.13 s
(shakespeare) and 1.21 s (children) of speech, and after absorption 1.19 s,
0.24 s and 1.32 s, all of them above `min_speech_s = 0.060 s`
(`absorber_would_be_dropped_by_min_speech_s: false` for every cell in
`experiments/silence_leading_artefact_results.json`).

Everything else is copied rather than reinvented: `CELLS` is copied verbatim
from `tests/test_pause_threshold.py:77-87` and re-verified byte-for-byte at
run time (`--check-copy`), and the statistics helpers, the watcher probe, the
run reconstruction and the resume flags are **imported** from
`experiments/pause_threshold.py`, so this measurement cannot drift from the
one that produced the committed 673 runs / 1671 events.

Parameters: analysis rate 16 000 Hz, frame 800 samples, hop 160 samples,
`min_silence_s = 0.150`, `min_speech_s = 0.060`, whole-file population
non-adaptive, watcher adaptive with 1600-sample chunks, pause events
`run_s > 0.15`, turn threshold 3.0 s.

### The watcher makes the defect fire far more often than the file does

This is the single most important thing in this document, and it was not
obvious in advance.

The whole-file population calls `energy.segment` ONCE per file, so "the first
segment" is the head of the file: three cells, three artefact silences.

The watcher calls `energy.segment` on a **moving buffer** on every push
(`src/watcher.py:385`), and the buffer is trimmed back to the last emitted
segment. So "the first segment" in the watcher is *the segment at the buffer
origin* — and the buffer origin lands in a short silence constantly. Measured
over the corpus (`mechanism` in the result file):

| | buffers | current: open with a SHORT silence | artefact-free: open with a SHORT silence | pushes whose first segment differs |
|---|---|---|---|---|
| narration | 900 | 101 | 0 | 404 |
| poetry | 201 | 23 | 0 | 85 |
| essay | 750 | 102 | 1 | 465 |
| dialogue | 750 | 107 | 1 | 403 |
| shakespeare | 750 | 110 | 0 | 418 |
| children | 750 | 88 | 1 | 414 |
| oration | 750 | 45 | 1 | 198 |
| **total** | **4851** | **576** | **4** | **2387** |

So the defect fires **576 times** in the watcher, not three times. The four
residual cases in the artefact-free column are buffers whose *only* segment
is that short silence, so there is nothing to merge it into.

Two consequences follow, and both are measured rather than argued:

1. The leading short silence becomes a segment of its own, and a silence
   segment is exactly what the watcher reports as a pause (`src/watcher.py:646`
   onward).
2. Because the buffer is trimmed to the last emitted segment, keeping that
   silence as its own segment moves the trim point, so the two runs' buffer
   origins diverge and their analysis grids move apart. That is why the
   difference is larger than "one extra event per artefact firing": the first
   segment differs on 2387 of 4851 pushes.

## 3. The per-cell leading-silence counts

| cell | first segment of the file | duration | current code | artefact-free code |
|---|---|---|---|---|
| narration | speech | – | no leading silence | no leading silence |
| poetry | speech | – | no leading silence | no leading silence |
| essay | silence 0.025 → 0.785 | **0.760 s** | kept — a REAL leading silence | kept |
| dialogue | silence 0.025 → 0.155 | **0.130 s** | kept — the artefact | **dropped** |
| shakespeare | silence 0.025 → 0.135 | **0.110 s** | kept — the artefact | **dropped** |
| children | silence 0.025 → 0.135 | **0.110 s** | kept — the artefact | **dropped** |
| oration | speech | – | no leading silence | no leading silence |

**Leading silences shorter than the 0.150 s trigger: 3 kept by the current
code, 0 kept by the artefact-free code** (dialogue, shakespeare, children —
one each). The essay cell's 0.760 s leading silence is longer than the
trigger, so it is a real silence in the recording and survives in both
populations.

## 4. Current versus artefact-free

Every number below is in `experiments/silence_leading_artefact_results.json`,
produced by `experiments/silence_leading_artefact.py` in one run
(188.6 s of measurement).

### 4.1 The pause table (whole-file energy population)

| cell | gaps cur | gaps art | shortest cur | shortest art | p05nr cur | p05nr art | median std cur | median std art | median pub cur | median pub art | @0.150 | @0.200 | lost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| narration | 45 | 45 | 0.170 | 0.170 | 0.210 | 0.210 | 0.640 | 0.640 | 0.640 | 0.640 | 45 / 45 | 43 / 43 | 2 / 2 |
| poetry | 7 | 7 | 0.150 | 0.150 | 0.150 | 0.150 | 0.510 | 0.510 | 0.510 | 0.510 | 7 / 7 | 6 / 6 | 1 / 1 |
| essay | 33 | 33 | 0.150 | 0.150 | 0.160 | 0.160 | 0.760 | 0.760 | 0.760 | 0.760 | 33 / 33 | 24 / 24 | 9 / 9 |
| dialogue | 35 | **34** | **0.130** | **0.160** | **0.160** | **0.170** | **0.450** | **0.460** | **0.450** | **0.470** | 34 / 34 | 30 / 30 | 4 / 4 |
| shakespeare | 62 | **61** | **0.110** | **0.150** | 0.160 | 0.160 | **0.375** | **0.380** | 0.380 | 0.380 | 61 / 61 | 46 / 46 | 15 / 15 |
| children | 78 | **77** | **0.110** | **0.150** | 0.160 | 0.160 | 0.300 | 0.300 | 0.300 | 0.300 | 77 / 77 | 62 / 62 | 15 / 15 |
| oration | 2 | 2 | 0.310 | 0.310 | 0.310 | 0.310 | 0.345 | 0.345 | 0.380 | 0.380 | 2 / 2 | 2 / 2 | 0 / 0 |
| **corpus** | **262** | **259** | 0.110 | **0.150** | | | | | | | **259 / 259** | **213 / 213** | **46 / 46** |

**The headline does not move.** 46 of the 259 pauses the 150 ms trigger keeps
are lost at 200 ms — **17.760618 %**, in both populations, to the last digit.

### 4.2 The pause-run population (the watcher)

Per cell: `events>0.15 / events unfiltered / distinct silence runs`, then the
longest, shortest and median run.

| cell | events >0.15 | events all | runs | max run | min run | median run |
|---|---|---|---|---|---|---|
| narration | 299 → **222** | 321 → **227** | 126 → **113** | **1.8301 → 1.7900** | **0.01 → 0.15** | 0.355 → 0.340 |
| poetry | 59 → **46** | 66 → **46** | 29 → **22** | 1.2000 → 1.2000 | **0.13 → 0.17** | 0.450 → 0.420 |
| essay | 263 → **173** | 281 → **178** | 109 → **97** | 1.5600 → 1.5600 | **0.08 → 0.15** | 0.330 → 0.320 |
| dialogue | 269 → **196** | 292 → **201** | 115 → **99** | 1.5300 → 1.5300 | **0.03 → 0.15** | 0.330 → 0.350 |
| shakespeare | 286 → **209** | 311 → **212** | 124 → **107** | **1.4500 → 1.3100** | **0.03 → 0.15** | 0.340 → 0.350 |
| children | 214 → **151** | 229 → **151** | 89 → **75** | **1.6000 → 1.5500** | **0.07 → 0.16** | 0.300 → 0.360 |
| oration | 281 → **244** | 289 → **247** | 81 → **75** | 2.5400 → 2.5400 | **0.07 → 0.15** | 0.530 → 0.540 |
| **corpus** | **1671 → 1241** | **1789 → 1262** | **673 → 588** | 2.5400 → 2.5400 | **0.01 → 0.15** | 0.350 → 0.360 |

The corpus loses **430 pause events (−25.7 %)** and **85 silence runs
(−12.6 %)**.

A new invariant appears: **artefact-free, the shortest silence run the
watcher reports is 0.150 s** — exactly `MIN_SILENCE_S`. Today it is 0.010 s.
The constant the segmenter claims to enforce is not enforced at the head of a
buffer.

### 4.3 The speculation figures

The population `tests/test_speculate.py::_corpus_pause_runs()` builds: pause
events with `run_s > 0.15` across the seven cells.

| | current | artefact-free |
|---|---|---|
| events | 1671 | 1241 |
| reach at `speculate_s = 0.30` | 1258 → **75.2843 %** | 935 → **75.3425 %** |
| share below the rejected 1.20 s | 1578 → **94.4345 %** | 1167 → **94.0371 %** |

Both still clear the floors the test asserts (reach > 0.35, below 1.20 s >
0.90). The published docstring figures (75.3 % and 94.4 %) become 75.3 % and
**94.0 %**.

### 4.4 Turn boundaries

| | current | artefact-free |
|---|---|---|
| silence runs ≥ 3.0 s | **0** | **0** |
| pause events ≥ 3.0 s | **0** | **0** |
| turn boundaries from `turn.TurnTracker` | **0** | **0** |
| longest run | 2.540 s | 2.540 s |
| whole-file gaps ≥ 3.0 s (essay / poetry) | 2 / 1 | 2 / 1 |

The 3 s turn threshold fires on **zero** of them in both populations, and the
longest run is 2.540 s in both. This conclusion does not move.

### 4.5 The F0 speaker-change population

`experiments/f0_speaker_change.py` measures the gaps between consecutive
utterance spans in (0.20 s, 4.0 s). That population is derived from the same
watcher run, so it was measured without a second pass over the corpus. (The
derivation was checked against the F0 script's own `utterances()` on the
poetry cell: 30 spans, identical to the last decimal.)

| | current | artefact-free |
|---|---|---|
| utterance spans | 684 | 671 |
| gaps in (0.20 s, 4.0 s) | **398** | **353** |

The F0 **numbers** would change, because the population they are computed on
changes. The expensive YIN pass was **not** re-run — see §7.

Note that this is the population the F0 script *selects*, not the number of
records it ends up with: `experiments/f0_speaker_change.py` also drops a gap
whose two F0 windows cannot be measured, so the `n pauses` column of
`docs/research/F0-SPEAKER-CHANGE.md:42-49` (children 52, essay 45, narration
72, dialogue 62, oration 56, shakespeare 66 = 353 records) is a different
population from the 398 gaps selected here, and the two numbers are not
comparable. The 353 in both places is a coincidence of arithmetic, not a
reproduction.

### 4.6 Addendum 2 of TURN-ROUTES.md (re-measured)

Runs / speaker resumed / at file end, on the watcher run population.

| threshold | current | artefact-free | published |
|---|---|---|---|
| 0.6 s | 170 / 169 / 1 | **153 / 152 / 1** | 54 / 53 / 0 |
| 0.9 s | 71 / 70 / 1 | **72 / 71 / 1** | 38 / 38 / 0 |
| 1.2 s | 35 / 35 / 0 | **32 / 32 / 0** | 24 / 24 / 0 |
| 1.5 s | 15 / 15 / 0 | **14 / 14 / 0** | 15 / 15 / 0 |
| 1.8 s | 6 / 6 / 0 | **5 / 5 / 0** | 6 / 6 / 0 |
| 2.1 s | 2 / 2 / 0 | 2 / 2 / 0 | 2 / 2 / 0 |

### 4.7 The summary table

| measurement | current | artefact-free | changes? | where the current number lives |
|---|---|---|---|---|
| whole-file gap count | 262 | 259 | **yes (−3)** | `experiments/pause_threshold_results.json` `totals.gaps`; `docs/research/PAUSE-THRESHOLD.md:247` |
| gaps ≥ 0.150 s | 259 | 259 | no | same file `totals.at_0_150`; `PAUSE-THRESHOLD.md:156` |
| gaps ≥ 0.200 s | 213 | 213 | no | same file `totals.at_0_200` |
| lost at 200 ms | 46 | 46 | no | same file `totals.lost` |
| lost % | 17.760618 % | 17.760618 % | no | same file `totals.lost_pct`; `PAUSE-THRESHOLD.md:22` |
| shortest gap, corpus | 0.110 s | 0.150 s | **yes** | `PAUSE-THRESHOLD.md:130-131`, `:205` |
| shortest gap, dialogue | 0.130 s | 0.160 s | **yes** | `PAUSE-THRESHOLD.md:181-182` |
| p05 / median, dialogue | 0.160 / 0.450 | 0.170 / 0.460 | **yes** | `PAUSE-THRESHOLD.md:124-131` |
| median (published convention), dialogue | 0.450 | 0.470 | **yes** | same table |
| watcher pause events > 0.15 s | 1671 | 1241 | **yes (−430)** | `docs/research/TURN-ROUTES.md:12`, `:48`; `PAUSE-THRESHOLD.md:241` |
| watcher pause events, unfiltered | 1789 | 1262 | **yes (−527)** | `PAUSE-THRESHOLD.md:242` |
| watcher distinct silence runs | 673 | 588 | **yes (−85)** | `TURN-ROUTES.md:12`, `:48`; `PAUSE-THRESHOLD.md:243` |
| watcher shortest run | 0.010 s | 0.150 s | **yes** | `pause_threshold_results.json` `watcher_pause_run_population.run_min_s` |
| watcher longest run | 2.540 s | 2.540 s | no | `TURN-ROUTES.md:23`; `run_max_s` |
| longest run, narration | 1.8301 s | 1.7900 s | **yes** | `TURN-ROUTES.md:17` |
| longest run, shakespeare | 1.4500 s | 1.3100 s | **yes** | `TURN-ROUTES.md:21` |
| longest run, children | 1.6000 s | 1.5500 s | **yes** | `TURN-ROUTES.md:22` |
| longest run, poetry / essay / dialogue / oration | 1.200 / 1.560 / 1.530 / 2.540 | unchanged | no | `TURN-ROUTES.md:18-20`, `:23` |
| speculation reach rate | 75.2843 % | 75.3425 % | **yes (value)** | `tests/test_speculate.py:103` |
| share below 1.20 s | 94.4345 % | 94.0371 % | **yes (value)** | `tests/test_speculate.py:104` |
| silence runs ≥ 3 s | 0 | 0 | no | `TURN-ROUTES.md:15-23`; `src/turn.py:33-41` |
| turn boundaries at 3 s | 0 | 0 | no | `TURN-ROUTES.md:15-23` |
| whole-file gaps ≥ 3 s | 3 (essay 2, poetry 1) | 3 | no | `src/turn.py:33-41` |
| F0 gap population | 398 | 353 | **yes** | `docs/research/F0-SPEAKER-CHANGE.md:42-49` |
| Addendum-2 rows (0.6 / 0.9 / 1.2 / 1.5 / 1.8 s) | 170 / 71 / 35 / 15 / 6 | 153 / 72 / 32 / 14 / 5 | **yes** | `PAUSE-THRESHOLD.md:270-277`; `TURN-ROUTES.md` addendum 2 |

## 5. The verdict, on conclusions rather than numbers

**What does NOT change.**

1. **The headline survives.** "Raising the trigger from 150 ms to 200 ms
   destroys 17.8 % of real pauses" is 46 of 259 = 17.760618 % in both
   populations, digit for digit. The argument in `src/turn.py:12-16`,
   `docs/research/ADOPTION-PLAN.md` §3 and `docs/research/PAUSE-THRESHOLD.md`
   stands unchanged.
2. **The 3 s turn threshold still fires on nothing.** Zero runs ≥ 3 s, zero
   turn boundaries, longest run 2.540 s, in both populations. The conclusion
   "people recorded monologue do not pause for three seconds" is untouched.
3. **The speculation conclusions hold.** Reach 75.28 % → 75.34 % and the
   share below 1.20 s 94.43 % → 94.04 %; both still clear the asserted floors
   by a wide margin, and the *rejected* 1.20 s threshold is still rejected for
   the same reason (it would void ~94 % of the speculations it caught).
4. **The "shortest real pause" claim is already falsified and stays
   falsified.** Artefact-free, the shortest inter-utterance pause is 0.150 s —
   *at* the trigger, not below it. The fix does not rescue the 110 ms claim;
   it removes the one piece of evidence that made the claim look
   survivable.

**What DOES change.**

5. **The watcher's pause population is the casualty, and it is large.** 673
   silence runs become 588, 1671 pause events become 1241. Every cell moves,
   including the four cells whose *files* have no leading artefact, because
   the watcher re-segments a moving buffer and the defect fires on 576
   buffers rather than three files. Anyone who quotes "673 runs / 1671
   events" as the corpus's pause population would be quoting a population
   that contains the artefact: 85 runs and 430 events disappear under the
   fix.
6. **Three committed tables would be stale**: `TURN-ROUTES.md`'s per-cell
   longest-run table (narration, shakespeare, children), its "673 / 1671"
   population sentence, and `PAUSE-THRESHOLD.md`'s Addendum-2 rows for
   0.6 / 0.9 / 1.2 / 1.5 / 1.8 s. `tests/test_speculate.py`'s docstring
   figures (75.3 % / 94.4 %) would become 75.3 % / 94.0 %.
7. **A new, better invariant appears.** Artefact-free, no silence run the
   watcher reports is shorter than `MIN_SILENCE_S = 0.150 s`; today the
   shortest is 0.010 s. The fix makes the shipped constant true.
8. **The F0 speaker-change numbers would have to be re-measured**, because
   the population they are computed on shrinks from 398 to 353 gaps.

**One thing this lane did NOT establish.** The difference between the two
watcher populations is not only the artefact's own events: keeping the
leading silence as a segment of its own moves the trim point, so the two runs'
analysis grids diverge (2387 of 4851 pushes have a different first segment).
The end-to-end numbers above are the numbers that a fix would actually
produce; they are not "the artefact's events removed from a fixed
population", and they should not be quoted as such.

## 6. The recommended fix — written out, NOT applied

The minimal change to `src/energy.py`, in `src/energy.py:284-306`:

```python
def _merge_short_runs(segs: list[Segment], min_silence_s: float,
                      min_speech_s: float) -> list[Segment]:
    """Drop silences shorter than min_silence_s and speech shorter than
    min_speech_s by merging them into their neighbours, then re-join."""
    if not segs:
        return []
    # The first segment has no predecessor, so it used to escape BOTH
    # filters: `kept = [segs[0]]` kept a silence shorter than
    # min_silence_s, which is how the silence before the first word of a
    # file survived. A short silence at the head of the list is merged
    # FORWARD into the segment that follows it, exactly as a short silence
    # with a predecessor is merged into that predecessor below.
    first = segs[0]
    if (not first.speech) and first.duration() < min_silence_s \
            and len(segs) > 1:
        nxt = segs[1]
        nxt.start = first.start      # the leading silence disappears
        segs = [nxt] + segs[2:]
    kept = [segs[0]]
    for s in segs[1:]:
        prev = kept[-1]
        if s.speech and s.duration() < min_speech_s:
            continue  # blip disappears into surrounding silence
        if (not s.speech) and s.duration() < min_silence_s:
            prev.end = s.end  # gap disappears into surrounding speech
            continue
        kept.append(s)
    # re-join adjacent segments of the same state created by the merges
    joined = [kept[0]]
    for s in kept[1:]:
        if s.speech == joined[-1].speech:
            joined[-1].end = s.end
        else:
            joined.append(s)
    return joined
```

This is the variant measured in `experiments/silence_leading_artefact.py`,
byte-for-byte in behaviour. The mutation of `segs[1].start` follows the
existing convention in this function (`prev.end = s.end` at
`src/energy.py:296` already mutates the caller's segments in place), and
`energy.segment` builds fresh `Segment` objects on every call, so nothing
outside the call is affected.

### What this fix would break

**It breaks `tests/test_pause_threshold.py::test_the_shortest_real_pause_is_under_the_trigger`
(line 103).** That test asserts

```python
assert min(shortest.values()) < energy.MIN_SILENCE_S
```

and it passes today **only because of the artefact**: the shortest per-cell
gap is 0.130 s (dialogue) / 0.110 s (shakespeare) / 0.110 s (children), and
those are the leading silences. Artefact-free the shortest gap is 0.150 s
(shakespeare, children, poetry, essay) and 0.160 s (dialogue) — so
`min(shortest.values())` becomes 0.150, which is **not** `< 0.150`, and the
test fails:

```
AssertionError: the shortest real pause is now 0.150s, at or above
MIN_SILENCE_S=0.15; per cell: children=0.150, dialogue=0.160, essay=0.150,
narration=0.170, oration=0.310, poetry=0.150, shakespeare=0.150
```

That test is asserting something that is not true of speech: it asserts that
a real pause is shorter than the trigger, using the silence before the first
word as its evidence. The fix does not invalidate the test's *purpose* — "the
trigger is on the correct side of the distribution rather than inside it" —
it invalidates its *evidence*. The replacement has to be re-derived, not
deleted: the honest statement is that the shortest inter-utterance pause in
the corpus is 0.150 s, exactly AT the trigger, which means the shipped
0.150 s value is on the boundary and any lowering of it starts eating real
pauses. (The whole-file population still contains gaps of 0.150 s in poetry,
essay, shakespeare and children — at the trigger, not below it.)

**Verified, not predicted.** The fix was applied *in memory* — the module
attribute `energy._merge_short_runs` was replaced with the variant above, with
no file edited — and the whole suite was run. Exactly two test failures
result, and they are the two expected ones:

```
FAILED tests/test_pause_threshold.py::TestRealPauseLength::
       test_the_shortest_real_pause_is_under_the_trigger
FAILED tests/test_pause_threshold_stats.py::TestTheArtefactMatchesTheCorpus::
       test_a_fresh_measurement_agrees_with_the_persisted_result
3 failed, 606 passed, 5 skipped
```

(the third failure was `tests/test_silence_leading_artefact.py`'s own
`test_the_variant_restores_the_shipped_function_afterwards`, which asserts
that the shipped function is the module attribute — it failed only because
the in-memory harness had replaced that attribute globally, not because of
the fix.)

**It does NOT break these, which is worth saying out loud:**

- `tests/test_pause_threshold.py::test_no_cell_has_its_pauses_eaten_at_the_shipped_value`
  (line 115) still passes: the number inside it moves from 1/35 = 2.9 % to
  0/34 = 0 %, and `0 < 0.05` still holds.
- `tests/test_pause_threshold.py::test_dramatic_speech_has_the_shortest_pauses`
  (line 131) still passes — but its margin collapses from 0.110 s vs
  0.310/2 = 0.155 s to **0.150 s vs 0.155 s, a 5 ms margin**. It is the test
  that would break next, on any further change to the trigger or the
  corpus.
- `tests/test_pause_threshold.py::TestTheRejectedChange` (lines 147-195):
  `lost/total` goes from 46/262 to 46/259 (17.76 %, unchanged to four
  figures), and the per-cell concentration tests still pass (15/61 = 24.6 %
  shakespeare, 15/77 = 19.5 % children).

**It also invalidates these artefacts (not tests, but things that hold a
number that would become stale):**

- `experiments/pause_threshold_results.json` — the `gaps` blocks of the
  dialogue / shakespeare / children cells (n, shortest, p05, median), the
  `leading_silence_artefact` blocks (they become null), the
  `corrected_without_leading_artefact` blocks (they become identical to the
  main blocks), and the entire `watcher_pause_run_population` block.
- `tests/test_pause_threshold_stats.py` — it re-derives the per-cell counts
  from the corpus and compares them with the persisted file, so it goes red
  until the result file is re-measured (verified above).
- `tests/test_speculate.py` — the two docstring figures (75.3 % / 94.4 % →
  75.3 % / 94.0 %). The assertions themselves (`reach > 0.35`,
  `below 1.20 s > 0.90`) still pass.
- `docs/research/TURN-ROUTES.md` — the population sentence, the three
  per-cell longest-run rows, and Addendum 2.
- `docs/research/PAUSE-THRESHOLD.md` — the Addendum-2 rows, and the §3 table's
  dialogue row (the corrected block and the main block would become the same
  thing, so the sensitivity section would have to be rewritten rather than
  deleted).
- `docs/research/F0-SPEAKER-CHANGE.md` and
  `experiments/f0_speaker_change_results.json` — the population shrinks from
  398 to 353 gaps, so every F0 band count changes.
- Any code that depends on the watcher's event count. I did not find one
  inside `src/`; the counts live in documentation, test docstrings and
  experiment result files.

**Nothing in `src/` would change behaviourally except the segmentation
itself.** `src/turn.py`, `src/speculate.py` and `src/vad.py` consume the
watcher's events and would see the new population; their constants and their
logic are unaffected.

## 7. What would need re-running, and re-checking

**Re-run:**

| artefact | command | why |
|---|---|---|
| `experiments/pause_threshold_results.json` | `python experiments/pause_threshold.py` | the gap blocks, the corrected blocks and the whole watcher population |
| `experiments/f0_speaker_change_results.json` | `python experiments/f0_speaker_change.py` | the F0 population changes (398 → 353 gaps); the YIN pass is the expensive part |
| `experiments/turn_routes_results.json` | `python experiments/turn_routes.py` | it consumes the corpus; its scoring is on manufactured cases, but its corpus-derived inputs must be re-checked |
| `experiments/silence_leading_artefact_results.json` | `python experiments/silence_leading_artefact.py` | it would then measure the SHIPPED code, which would be the fixed code — the current-vs-artefact-free comparison in it becomes a no-op and the file should be retired or re-scoped |

**Re-check:**

| artefact | what to re-check |
|---|---|
| `docs/research/PAUSE-THRESHOLD.md` | §3 table (dialogue row), the corrected/sensitivity section, Addendum 2, and the §"stale claims" table at :381-384 — the 110 ms claims are already marked stale, and the fix would make the corrected block and the main block identical |
| `docs/research/TURN-ROUTES.md` | the population sentence at :12 and :48, the per-cell table at :15-23, Addendum 2 |
| `docs/research/F0-SPEAKER-CHANGE.md` | the per-cell pause counts at :42-49 |
| `tests/test_speculate.py` | the docstring figures at :103-104 |
| `tests/test_pause_threshold.py` | `test_the_shortest_real_pause_is_under_the_trigger` (line 103) **fails** and has to be re-derived; `test_dramatic_speech_has_the_shortest_pauses` (line 131) survives on a 5 ms margin; `test_no_cell_has_its_pauses_eaten_at_the_shipped_value` (line 115) survives with its number moved to zero |
| `tests/test_pause_threshold_stats.py` | it re-derives the corpus and compares it with the persisted file, so it goes red until `experiments/pause_threshold_results.json` is re-measured |
| `src/turn.py` | the docstring table at :33-41 and the `17.8%` paragraph at :12-16 — the 46-of-259 figure survives, but the table's `max_gap` column is the whole-file population and should be checked against the new one |
| `src/watcher.py` | the corpus-length and longest-pause figures quoted in its docstring (:660-665) |
| `docs/research/ADOPTION-PLAN.md` §3 | the published pause table |

**One stale number found on the way, outside this lane and NOT changed by it:**
the `src/turn.py:37` docstring row for poetry read `max_gap "-"` and `gaps >= 3 s
"-"`, and the whole-file energy population has a **5.100 s** gap in poetry
(1 gap ≥ 3 s), which is what `experiments/pause_threshold_results.json`
records (`max_s: 5.1`). The row appeared to predate the poetry cell being
measured.

> **CORRECTION (2026-10-19, `NOTES-AUDIT-3.md` F10).** This was true when it
> was written and it was **fixed in the same commit that added this file**
> (`d4a49c1`), so the "it is wrong today" is no longer true. The row now reads
> `20.011 / 1.740 / 0`.
>
> The fix turned out to be a **population** question rather than a number
> question, and this note got the first half of it wrong. The 5.100 s silence
> in poetry runs from **14.895 s to 19.995 s of a 20.011 s file** — it is the
> recording stopping, not a speaker pausing. Poetry's longest
> **inter-utterance** gap is **1.740 s**, so poetry produces no turn boundary
> at 3 s, like the other five cells, and the "-" was right for the
> inter-utterance population the turn table counts. `src/turn.py` also now
> says "six of seven cells", because poetry is no longer an unmeasured cell.
> `docs/research/PAUSE-THRESHOLD.md` §4, "The other end of the file:
> trailing silence".

## 8. Provenance

Every number in this document comes from
`experiments/silence_leading_artefact_results.json`, written by
`experiments/silence_leading_artefact.py`. That file records, for every
result:

- the **absolute path** and **SHA-256** of every input wav, its size, and its
  own header (sample rate, frame count, duration, channels, sample width) as
  well as the length **after** the resampling read at the analysis rate —
  because `wavio.read_wav` resamples to 16 kHz and a provenance block that
  reports the analysis rate as the file rate is lying about its input (the
  narration cell is 44 100 Hz on disk and 16 kHz after the read);
- the **exact parameters** (analysis rate, frame, hop, `min_silence_s`,
  `min_speech_s`, adaptive on/off, chunk size, the event filter, the turn
  threshold, the speculation threshold);
- the **SHA-256 of the script itself**;
- the **source file and line of the logic that was varied** —
  `src/energy.py`, `_merge_short_runs`, lines 284-306, defect line 290 —
  together with the text of that line, checked against the file on disk at
  measurement time (`defect_line_verified`);
- the **SHA-256 of `src/energy.py`**, of the committed
  `experiments/pause_threshold_results.json`, and of
  `tests/test_pause_threshold.py` (the source of the copied `CELLS`).

The test file `tests/test_silence_leading_artefact.py` (23 tests) keeps this
honest: it re-derives both populations from the audio, checks them against
the committed `gaps` and `corrected_without_leading_artefact` blocks key by
key, asserts that the two populations actually differ, attacks the checker
with a perturbed count, shows that removing the variant would be caught, and
checks every recorded SHA-256 against the file on disk.
