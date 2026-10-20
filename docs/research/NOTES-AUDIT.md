# NOTES AUDIT — every standing numeric claim checked against its result file

**Measured 2026-10-19. 14 findings: 10 confirmed, 4 candidates. 22 claims
checked and found sound.**

This is an audit of `docs/research/` against `experiments/*.json`, run as a
read-only pass over 64 markdown files and 55 result files. Every numeric
claim in the six failure families was traced to the file that produced it.

**Why this exists.** The project has four documented instances of the same
failure — a comparison drawn between quantities measured over different
intervals or populations (`HOW-WE-KEEP-FOOLING-OURSELVES.md`). The apparatus
is producing the right answers; the notes are carrying forward numbers a
later run superseded. **Six of the fourteen findings are numbers a result
file already contradicted, and four more are numbers the repository's own
test docstring already contradicted** (`tests/test_speculate.py:91-97`).

---

## 1. The confirmed findings

### F1 — The "speech is visible" factor uses the WRONG condition's control

`ROW-BAND-REMOVAL.md:137-139`, and the same pair at `AV-SYNC.md:152`,
`TEACHER-PIPELINE.md:21`, and inside `experiments/pose_test_results.json`.

> *"Facing-and-speaking vs facing-and-silent differ by 4.914 against a
> within-condition split of 1.917 — 2.6×"*

The contrast is **A-vs-C**. The denominator 1.917 is
`within_B_turned_speaking` — the within-split of condition **B** — not of A.
`pose_test_results.json` holds all three numbers:

| field | value |
|---|---:|
| `within_A_facing_speaking.mean_abs_diff` | **2.878** |
| `within_B_turned_speaking.mean_abs_diff` | 1.917 |
| `within_C_facing_silent` | **absent** |

4.914 / 1.917 = 2.564 (the published 2.6×). 4.914 / **2.878** = **1.708**.
The published factor is **1.5× larger** than the matched control produces, and
the control it used belongs to the condition with the *largest* cross-condition
difference in the set (B-vs-C = 9.411).

The note prints the correct denominator 25 lines earlier, in its own table.
It picks the other one.

**Settling measurement.** Recompute against A's own within-split (1.71×) and
add a `within_C_facing_silent` split so the control is matched to the
contrast. **The qualitative conclusion survives** — `frac_above_null` for
A-vs-C is 12.1 % against 34.9 % for A-vs-B, and the top-5 % share is 79.1 %.
It is the factor that is wrong.

### F2 — "Four of six" inverts a cell count

`F0-SPEAKER-CHANGE.md:52-55`, `ADOPTION-PLAN.md:348-351`.

> *"**Four of six** have a long-pause band that does not exceed pure intonation
> inside a single speaker's own speech"*

`f0_speaker_change_results.json` → `verdict`: `cells_long_exceed_control: 4`,
`cells_long_under_control: 2`. **Two of six** are below the control, and the
note's own table agrees with the JSON. The sentence is also self-contradictory:
it says four do not exceed, then names the two that do.

### F3 — "Leave-one-out 0.000 for every feature set" is false

`TEACHER-PIPELINE.md:75-76`, and adjudicated TRUE at `LLM-DIRECTOR.md:38`.

| feature set | LOO A | LOO B |
|---|---:|---:|
| summary | 0.0 | 0.0 |
| region | 0.0 | 0.0 |
| **map** | 0.0 | **1.000** |
| both | 0.0 | 0.0 |
| **region+map** | 0.0 | **0.200** |

Two of five are not 0.000 — **map** and **region+map**. Separately,
`TEACHER-PIPELINE.md:71-73`'s `both` row read `A 0.000, C 0.650`, and **no
result file contains a held-out session "C"** for this task; `0.650` matched
neither candidate source (B 0.675 or B 0.200).

> **AMENDED (2026-10-19) — F3's own table had a bad cell, and it is the same
> defect F3 is about.** This table listed `both` as **B 0.675**, taken from the
> committed artefact `experiments/teacher_conventional_speaking_both.json` at
> `7a4dff5`. That number **does not reproduce.** Re-running the committed
> script against the committed corpus reproduces every other column of that
> artefact exactly — test 1.000, scrambled 1.000, blank 0.667, margin +0.000 —
> and returns LOO A 0.000, B 0.000.
>
> It is not a code problem and it is not a data problem:
>
> - The **script** at `7a4dff5` and at `HEAD` are **not** the same file — it
>   gained the provenance API in `d6baa6d` (blob `459e23c0…` → `f8f89c96…`) —
>   and **both** versions return 0.000, so the conclusion does not depend on
>   which one is run. (`NOTES-AUDIT-3.md` F6; this sentence was true when
>   written, because `HEAD` was then `7a4dff5`'s neighbour, and it stopped
>   being true when the provenance API landed. A claim about "the script"
>   needs the script's hash next to it.)
> - The **corpus** rebuilds byte for byte from the captures
>   (`seed_corpus_from_captures.py --filter row+median3` produces three
>   SHA-256-identical `.npz` files), so the input is not the variable either.
> - The artefact **never recorded** which corpus produced it, nor its stride,
>   max-per, frame count or task — so the run cannot even be *specified* from
>   what was committed. A sweep of 1280 configurations over those unrecorded
>   parameters -- committed as `experiments/retraction_sweep.py` and
>   `experiments/retraction_sweep_results.json` -- never produces 0.675
>   together with the artefact's other columns.
>
> The other four feature sets' artefacts reproduce bit-exactly from
> `/mnt/c/proart_capture/corpus`, so the **verdict** ("0.000 for every feature
> set" is false) stands on numbers that do reproduce: map **1.000** and
> region+map **0.200**. The **0.675 is withdrawn.** Result files now carry the
> identity of their corpus, so an untraceable number cannot be quoted again.
> `COMBINED-FILTER.md`, `ARTIFACT-REPRODUCIBILITY.md`.
>
> The generalisable lesson, and it is F3's own lesson one level up: **an audit
> that trusts a result file inherits that file's provenance defects.** The audit
> was right that the claim was false; it was wrong about which number made it
> false.

### F4 — The "empty saturation" column holds the OCCUPIED flip rate

`COMBINED-FILTER.md:25-31`, and the consequences at `:33-35` and `:83-86`.

`experiments/combined_filter.py:185-186` prints
`empty flip | empty sat | occ flip | occ sat | A-vs-C gap | loc`.
`combined_filter_results.json` for `raw` holds `empty_flip: null`,
`empty_saturation: 0.41725`, `A_flip: 0.066344`, `A_saturation: 0.99729`,
`A_vs_C.mean_abs_diff: 3.94146`. The note's table headers are shifted against
the script's own output.

Consequences:

  - **"an empty-room saturation of 0.0014" is FALSE.** The real empty-room
    saturation for row+median3 is **0.034787 (3.48 %)**, and for
    row+median5 it is **0.0**. 0.0014 is the occupied-A flip rate.
  - **"a 21× improvement over either filter on its own"** — 0.0299/0.0014 =
    21.4× against `row`, and 0.0409/0.0014 = **29.2×** against `median3`. The
    21× uses the *better* of the two singles, so it is not "over either".
  - On the actual `empty_saturation`, row+median3 is **14.0×** better than
    `row` alone and **1.12×** better than `median3` alone — the composition
    buys almost nothing over the temporal median on the empty room.

### F5 — A retracted claim and a wrong cause still standing in the body

`GATE-PEAK-COVERAGE.md:74-75, :79, :88, :102-106`. The retraction banner is
at lines 3-39; **the body is unamended**. It asserts a 14× factor (the
retracted quantity) at two places, *"Not the LSTM, and not a warm-up
artefact"* — which the retraction reverses, since the peak it points at **is**
the t = 0.064 s LSTM transient — and *"The cause is buffer trimming"*, a
competing explanation the retraction rejects.

**Settling measurement.** None needed. Amend or strike the body; a banner
alone does not catch a reader who arrives at line 74.

### F6 — A quoted probe verdict the probe cannot produce

`ROW-BAND-REMOVAL.md:172-177` quotes

```
drift   : after row-band removal 9.909 counts   OK - below ~5
```

but `syzygy_capture_win.py:203-208` prints `OK - below ~5` **only** when the
value is below 5, and `9.909 >= 5`. `git log -p` confirms the threshold has
never been other than 5. **The quoted block is not an output of this script at
any revision**, and the note then sets a standing target ("under ~5") that its
own quoted measurement fails, under an OK verdict.

### F7 — "465 s" against its own table

`TURN-ROUTES.md:12` and `:336-337`. The seven-cell table sums to **485 s**;
the prose says **465 s** twice. Measured on disk the cells total **485.01 s**.
The 20 s gap is exactly the poetry cell. The same 465 s is hard-coded in
`src/turn.py:13`, `src/speculate.py:6`, `src/watcher.py:661`,
`tests/test_turn.py:25`, `tests/test_speculate.py:7`,
`experiments/turn_routes.py:8` — **the wrong total is in the code and the
test suite, not only in the note.**

### F8 — "680 measured pause runs" against the repo's own measurement: 1671

`TURN-ROUTES.md:12` and the whole Addendum 2 table at `:322-332`.
`tests/test_speculate.py:91-97` already records the disagreement in the
repository, with the numbers: **1671 pause runs**, min 0.160 s / median
0.380 s / max 2.540 s, against the prose's 680 / 0.320 / 2.440. The Addendum-2
rates (98.1 %, 100.0 % ×5) are computed on the 680-run population.

### F9 — "A factor of 13" against "a factor of 10" for the same comparison

`alwayson/JEV-AS-INSTRUMENT.md:310-311` says 13 (0.940 vs 0.070);
`alwayson/README.md:218-220` says 10 (0.940 vs 0.090). The results file holds
**one** run: holistic 0.09, max_battery 0.94, ratio **10.44×**. The value
0.070 appears only in a run-to-run paragraph of the same note, so the "13"
pairs a run-1 battery score with a run-2 holistic score.

### F10 — The "28×" and "5.4×" separation figures are not derivable

`alwayson/README.md:234`, `alwayson/SILERO-VAD.md:40-42`,
`alwayson/SYNTHESIS.md:44-46`. From `hotpath_measurements.json`:

| candidate | value |
|---|---:|
| worst speech mean ÷ largest non-speech mean (0.7530/0.0081) | **93.0×** |
| speech max ÷ non-speech max (1.0000/0.1736) | **5.76×** |
| the parenthetical the note supplies (0.7530/0.1736) | 4.34× |

**Nothing yields 28 or 5.4.** The underlying separation is not wrong — the
file supports 93× on the mean, which is *larger* than 28.

### F11 — "39.1 % of a core" has no denominator

`ADOPTION-PLAN.md:36, :47-49, :57-59`, `alwayson/SYNTHESIS.md:48-56`.
39.13 ms is the mean over the **six non-speech** fixtures. On real speech the
watcher is **21.9–36.1 ms** (2.77–4.56× realtime). The rate is real and the
trade-off is sound — "a gate that costs 0.23 % of a core to delete a 39 %-of-a-
core failure mode" is correct precisely because 39.1 % is the non-speech cost.
What is missing is the population, and the 8× Pi projection (313 % of a core)
is the non-speech worst case stated as the general case.

## 2. The candidates

  - **F12** — `CAMERA-FLICKER-BANDING.md:100-107`'s "~70 % of the frame-to-frame
    variation is the lamps" divides a spatial standard deviation (31.312) by a
    temporal MAD (43.686). For a Gaussian, MAD = 0.6745σ, so the share is
    ≈48 %, not 71.7 %. **Settling:** decompose each frame pair into
    row-constant and column-varying parts and report the fraction of the
    *temporal residual* explained by the row term.
  - **F13** — `jev-play-findings.md:395-396`'s "24/24, perfectly stable" is 3
    cases × 8 re-asks of the same input; the effective n is 3, and the 24
    repeats are not in the results file. **Settling:** store the per-repeat
    outcomes and report the number of distinct cases beside the repeat count.
  - **F14** — `DEVICE-GENERALISATION.md:119-123`'s "YIN's 1.074" is actually
    the **whole ungated watcher** (YIN plus segmenter plus buffer
    management); YIN was never measured alone. The relative cache penalty
    (1.107) is computed correctly. **Settling:** measure YIN in isolation
    under the same flush protocol, or rename the number.

## 3. Claims checked and found sound (22)

`WATCHER-COST.md` — **every** figure matches `watcher_cost_results.json`:
72 grid cells confirmed, max threads ratio 1.003, speech_30s 29.87/30.02, max
cpu 49.105, 46.735/3.31 = 14.1×, 1.585/0.044 = 36.0×, all six growth
exponents in both columns to 3 decimals.

`EMPTY-ROOM-CONTROL.md` — every number matches. Occupied luma 111.175, MAD
43.686, flip 48.190 %, spread 33.93–56.82 %; empty luma 49.6986, MAD 39.3436,
flip 50.00 %, spread 25.00–50.00 %; ratio 1.0376.

`ROW-BAND-REMOVAL.md` — the 159× row-constant reduction is the same measure
on the same capture, so it is well formed; the flip-rate table matches to 4
decimals; and the A-vs-C localisation figures (12.1 %, 79.14 %) are sound.
Only the factor built on 1.917 is defective (F1).

`DEVICE-GENERALISATION.md` — every ratio matches, and the flush control is
itself verified (`control_slowdown_x` 19.285, `flush_is_real: true`).

`IN-THE-WILD-AUDIO.md` — matches `inthewild_probe_results.json`, including
the 1.68 % relative loss.

`NOTCH-AB.md`, `MIC-HARDWARE.md` — every derived factor checks out (4.07×,
3.54×, 613×, 2.83 %), with the sample sizes stated next to the means.

`SPECTROGRAM-TURN.md` — the 0.5000 / 0.6225 figures are read correctly, and
the note treats them as the **negative** result they are.

`F0-SPEAKER-CHANGE.md` — everything except the cell count matches, to 4
decimals, including the ρ and p columns and the Bonferroni correction.

`TWO-SPEAKER-TURNS.md` — AUC 0.529 over 1396 pauses, best bar 0.5709, base
rate 0.6225, all consistent.

`TURN-ROUTES.md` — the statistics and the self-criticism are sound. The
permutation result is reported with **its own null mean (0.5079)** rather
than assuming 0.5, and two of three controls are reported as at-or-below
chance. The document carries the house rule explicitly: *"A measurement that
returns a suspiciously round 100 % or 0 % is usually a harness bug, not a
result."*

`ADOPTION-PLAN.md` — the retracted pause table is internally consistent
(45+7+33+35+62+78+2 = 262 gaps; 46 lost = 17.76 %).

`alwayson/SILERO-VAD.md` — every gate score matches
`hotpath_measurements.json` exactly, and "0.23 % of a core" = 74.9/32000.

`alwayson/JEV-AS-INSTRUMENT.md` — every battery number matches.

`jev-play-findings.md` — the symbolic-audio table matches
`symbolic_audio_results.json`.

`CLOUD-TEACHER.md` — the arithmetic is sound and the guards are in place,
including its own statement of the blank-control trap.

## 4. Unknowns (not findings)

  - `EXPERIMENTS-AUDIO.md` §3f/§3h (E7) and `TIMING.md`'s Day-3 correction
    are backed only by `/tmp/e7/`, which is not in the repository. The two
    notes agree with each other; neither is checkable against a result file.
  - `turn_routes_results.json` stores **no summary** — the headline
    permutation result and the pair-concordance values are printed by the
    script but not persisted, so the headline statistic cannot be reproduced
    from the repository.
  - `combined_filter.py` has no `drift` field, so "a drift of 9.4 counts after
    composition, against 32.1 for row removal alone" has no counterpart in any
    result file.
  - The subdirectory lanes (audio-dsp, audio-engines, audio-llms, voiceloop)
    quote third-party numbers with their sources named, and no instance of the
    six failure shapes was found in them.

## 5. The process note, which is the real finding

**Six of the fourteen findings are numbers that a result file already
contradicted, and four more are numbers the repository's own test docstring
already contradicted.** The measurement apparatus is producing the right
answer; the notes are carrying forward numbers that a later run superseded.

So the fix is not only to correct fourteen sentences. It is to **persist the
summary statistics** — the pair-concordance and permutation output from
`turn_routes.py`, the LOO table from `teacher_conventional.py`, the column
labels from `combined_filter.py` — so that these disagreements become
visible without a line-by-line audit.
