# NOTES AUDIT 2 — the second pass over `docs/research/`

**Measured 2026-10-19. 16 findings: 10 confirmed, 6 candidates. 49 documents
audited, ~1,000 measurement-shaped numeric claims traced.**

This is round two of the audit begun in [`NOTES-AUDIT.md`](NOTES-AUDIT.md). Round
one covered 14 documents and found 14 findings (10 confirmed). This pass covers
the documents round one did **not** audit, plus the four subdirectory lanes
round one recorded as "quote third-party numbers with their sources named".

**Why this pass was worth running.** Round one ended by writing the subdirectory
lanes off as uncheckable: *"The subdirectory lanes (audio-dsp, audio-engines,
audio-llms, voiceloop) quote third-party numbers with their sources named, and
no instance of the six failure shapes was found in them."* That is half true.
Three of the four lanes are indeed third-party lit reviews with no local
measurement. **`docs/research/voiceloop/` is not** — it ships two result files
of its own (`jev_voiceloop_results.json`, `smartturn_bench.json`) and a
benchmark script that writes them. Auditing it against those files, plus
`alwayson/OPENWAKEWORD.md` and `alwayson/PYANNOTE.md` (the two `alwayson`
documents round one did not open), produced every confirmed finding below.

**Method.** Same as round one. For each numeric claim: find the file that
produced it; check the number; then check the **interval** — same population,
same split, same control, same normalisation. Result files live in
`experiments/*_results.json`, `experiments/teacher_conventional_*.json`,
`experiments/*_observations.json`, `docs/research/alwayson/*_results.json`,
`docs/research/alwayson/hotpath_measurements.json`, and — newly discovered in
this pass — `docs/research/voiceloop/*.json`. Claims that name an upstream
repository's published table (PAPERS.md, the audio-engines lane, funasr.md,
voxcpm.md) are outside the reach of this method and are listed in §4, not
scored.

**The six established traps were checked against every in-scope document and
none of them is repeated.** No in-scope document quotes 2.6× for the
speech-visibility factor, 465 s for the corpus, the "two of five are 0.000"
conventional-teacher LOO, the blank control as a task control, +0.0088 for the
audio-only pose channel, or the llama-4-scout-17b +0.167 margin. (The one
`2.6×` that appears — `alwayson/PYANNOTE.md:188` — is an unrelated ratio and is
wrong for a different reason: F1.)

---

## Summary of findings

| id | doc | line | claim | verdict | severity |
|---|---|---|---|---|---|
| F1 | `alwayson/PYANNOTE.md` | 188 | "2.6× openWakeWord's Python" | actual **3.67×** (798,995 ÷ 217,800) | CONFIRMED |
| F2 | `alwayson/PYANNOTE.md` | 188 | "6.6× silero's" — but silero's `.py` is a *different population* | bases are not comparable | CONFIRMED |
| F3 | `alwayson/PYANNOTE.md` | 188–189 | "the only one whose *notebooks* exceed its source" | openWakeWord's do too (3.96×) | CONFIRMED |
| F4 | `alwayson/OPENWAKEWORD.md` | 184 | notebooks "more than 4× the entire Python source" | actual **3.96×** | CONFIRMED |
| F5 | `voiceloop/JEV-AS-INSTRUMENT.md` | 238–239, 243 | per-axis spread table + "the substantive axes are *bit-identical*" | file gives spreads 0.020 / 0.010 / 0.010 | CONFIRMED |
| F6 | `voiceloop/README.md` | 266–268 | "spread across knobs of exactly **0.000** on both substantive axes" | prosody spread is 0.020 | CONFIRMED |
| F7 | `voiceloop/JEV-AS-INSTRUMENT.md` | 155 | "Reversed order: **6/9 first option**" | file's own `first_option` field gives **3/9** | CONFIRMED |
| F8 | `voiceloop/README.md` | 254–255 | "reversed order only 6/9" | same | CONFIRMED |
| F9 | `voiceloop/README.md` | 157–167 | Coqui licence table, "I parsed all 95 entries" | rows sum to **93**; a row is missing | CONFIRMED |
| F10 | `voiceloop/README.md` | 282 | "93 of 95 models permissive" | manifest gives **89** | CONFIRMED |
| C1 | `voiceloop/pipecat.md` | 162–164 | run-1 benchmark column | no file on disk holds it | CANDIDATE |
| C2 | `TIMING.md` | 75 | "distilled separates 0.19/0.81" | no source found | CANDIDATE |
| C3 | `voiceloop/JEV-AS-INSTRUMENT.md` | 241 | "`knobs_are_separate` flag fired `True` on the 0.02–0.03 differences" | the flag has no threshold | CANDIDATE |
| C4 | `voiceloop/JEV-AS-INSTRUMENT.md` | 268 | "short ~0.02" | file says 0.01 | CANDIDATE |
| C5 | `ASCII-MOTION.md` | 192–193 | "beyond ~20 counts a dot is already 1.6× more stable" | the note's own table gives 1.95× | CANDIDATE |
| C6 | `ASCII-MOTION.md` | 102–104 | "static-repeat control of exactly +1.000 **throughout**" | defined at lags 1–3; null at 4–6 | CANDIDATE |

---

## 1. The confirmed findings

### F1 — "2.6× openWakeWord's Python" is 3.67×

`alwayson/PYANNOTE.md:188`:

> three by a wide margin (**2.6× openWakeWord's Python**, 6.6× silero's)

The two byte counts are in the same lane, in the same tables:

| repo | `.py` bytes | source |
|---|---|---|
| pyannote | 798,995 | `PYANNOTE.md:183` |
| openWakeWord | 217,800 | `OPENWAKEWORD.md:176` |
| silero-vad | 120,777 | `SILERO-VAD.md:250` |

798,995 ÷ 217,800 = **3.668**. The document says 2.6. The sibling ratio in
the same parenthesis is right (798,995 ÷ 120,777 = 6.615 → "6.6×"), which is
what makes the 2.6 look like a computed figure rather than a guess — but no
arithmetic on these three numbers produces 2.6, and no other quantity in the
repository does either (798,995 ÷ 2.6 = 307,305, which is not any repo's file
total).

**Correction:** replace `2.6×` with `3.7×`.

### F2 — the cross-repo comparison mixes two different populations of `.py`

The same parenthesis compares pyannote's whole-tree `.py` against openWakeWord's
whole-tree `.py` against silero's `.py`, but those are not the same population.
`alwayson/SILERO-VAD.md:245-246` states its own basis:

> Whole tree at the pinned commit (`.py` counts **only the package + tests**, so it
> is deliberately small)

So silero's 120,777 bytes is a *package + tests* count while pyannote's
798,995 and openWakeWord's 217,800 are whole-tree counts (both documents say
"Whole tree at the pinned commit"). The "6.6× silero's" therefore divides a
whole-tree number by a subset number and is inflated by whatever the silero
repo's examples/scripts/docs contribute.

This is the round-one failure shape exactly — *the same statistic summarised
over two different intervals* — and it is the only place in the repo where
these three source-vs-prose tables are compared side by side, so it is the
place a reader will use.

**Correction:** state the basis in the comparison — e.g. "3.7× openWakeWord's
whole-tree Python, 6.6× silero's **package + tests only** (not whole-tree)" —
or recompute silero on a whole-tree basis. Do not present the three ratios as
if they share a denominator.

### F3 — "the only one whose *notebooks* exceed its source" is false

`alwayson/PYANNOTE.md:188-189`:

> and it is the only one whose *notebooks* exceed its source by 6.6×

pyannote's ratio is right: 5,264,103 ÷ 798,995 = 6.588 → "6.6×". But openWakeWord's
notebooks also exceed its Python source: 862,476 ÷ 217,800 = **3.960**. silero
lists no `.ipynb` row at all, so silero is the only one of the three that
does *not* exceed. The exclusivity is exactly inverted.

**Correction:** "the only one whose *notebooks* exceed its source by **6.6×**"
(drop "the only one whose … exceed its source", keep the factor), or "it
exceeds its source by 6.6×, the largest of the three (openWakeWord's
notebooks are 3.96× its source)".

### F4 — openWakeWord's notebooks are 3.96× its source, not "more than 4×"

`alwayson/OPENWAKEWORD.md:184`:

> the 4 notebooks total 862 KB, **more than 4× the entire Python source**

862,476 ÷ 217,800 = **3.960**. The claim is off by 1% and lands on the wrong
side of its own threshold: it says "more than 4×" for a number that is less than
4×. (The neighbouring claim, "3.3× more code than prose", is correct:
217,800 ÷ 67,006 = 3.250 → 3.3×.)

**Correction:** "the 4 notebooks total 862 KB, **3.96×** the entire Python
source."

### F5 — the E6 spread table is not in the result file, and the file contradicts it

`voiceloop/JEV-AS-INSTRUMENT.md:238-244`:

> Spread across the three knobs: `timing` **0.000**, `prosody` **0.000**,
> `orthogonal` 0.020, `realtime_safe` 0.030.
>
> My script's `knobs_are_separate` flag fired `True` on the 0.02–0.03
> differences — and **that flag is a false positive I should not trust.** The
> substantive axes are *bit-identical*.

`docs/research/voiceloop/jev_voiceloop_results.json` holds E6 as:

```json
"rows": {
 "noise_scale":     {"timing": 0.91, "prosody": 0.85, "realtime_safe": 0.38, "orthogonal": 0.77},
 "noise_scale_w":   {"timing": 0.91, "prosody": 0.84, "realtime_safe": 0.38, "orthogonal": 0.78},
 "sdp_ratio":       {"timing": 0.91, "prosody": 0.86, "realtime_safe": 0.39, "orthogonal": 0.78}
}
```

max − min per axis, from those numbers:

| axis | doc's "spread" | actual spread | |
|---|---|---|---|
| `timing` | 0.000 | **0.000** | ✓ |
| `prosody` | 0.000 | **0.020** | ✗ |
| `orthogonal` | 0.020 | **0.010** | ✗ |
| `realtime_safe` | 0.030 | **0.010** | ✗ |

Three of the four entries disagree with the file, and there is no "spread"
field anywhere in it — the script computes `distinct_control_vectors` and
`knobs_are_separate` and nothing else (`jev_voiceloop.py:528-534`). The
document's own table three lines above the spread line shows
`prosody` as 0.85 / 0.84 / 0.86, so the document contradicts itself before it
contradicts the file.

The load-bearing sentence — "The substantive axes are *bit-identical*" — is
false against the file: only `timing` is identical. The three knobs were rated
*distinguishable* on `prosody`, `orthogonal` and `realtime_safe`; the
differences are small (≤ 0.02 on a 0–1 scale), which is a weaker and less
dramatic claim than the one the document makes. The document's conclusion
("Jev rated three semantically different knobs identically") survives only in
the weaker form "rated them nearly identically", and the strong form is what
the text asserts.

**Correction:** replace the spread line with the file's own numbers —

> Spread across the three knobs: `timing` **0.000**, `prosody` **0.020**,
> `orthogonal` 0.010, `realtime_safe` 0.010.

and change "The substantive axes are *bit-identical*" to something like "The
substantive axes differ by at most 0.02 — near-identical, not bit-identical."
The script's flag is best described by what it actually computes (see C3).

### F6 — the same spread claim in the lane's README

`voiceloop/README.md:266-268`:

> it scored all three *identically* — spread across knobs of exactly **0.000** on
> both substantive axes

Same defect as F5, in the lane's summary document rather than its detail
document (the README is the journey log; `JEV-AS-INSTRUMENT.md` is the
detail). "Both substantive axes" = `timing` and `prosody`. `timing` is 0.000;
`prosody` is 0.020. Half the claim is right, and it is the half that happens
to be the one the file supports.

**Correction:** "spread across knobs of 0.000 on `timing` and 0.020 on
`prosody`", or "of at most 0.020 on every axis".

### F7 — "Reversed order: 6/9 first option" is 3/9 under the file's own definition

`voiceloop/JEV-AS-INSTRUMENT.md:153-157`:

> - **Forward order: 9/9 first option.** That is *exactly* the reported "picks the
>   first 9/9" bias, reproduced precisely.
> - **Reversed order: 6/9 first option.** The bias **breaks** under reversal.

`jev_voiceloop_results.json` E8 records, for each of six order-blocks, a
`first_option` field (the option listed first *in that order*) and the three
picks. Counting calls where the pick equals that block's `first_option`:

| pair | order | `first_option` | picks | calls on `first_option` |
|---|---|---|---|---|
| A/B | forward | variant_A | A A A | 3 / 3 |
| A/B | reversed | variant_B | A A A | **0 / 3** |
| C/D | forward | variant_C | C C C | 3 / 3 |
| C/D | reversed | variant_D | D D C | **2 / 3** |
| E/F | forward | variant_E | E E E | 3 / 3 |
| E/F | reversed | variant_F | E F E | **1 / 3** |

Forward **9/9** ✓ (the document is right). Reversed **3/9**, not 6/9.

The number 6 does exist in the data, but it counts something else: 6 of the 9
reversed calls landed on the pair's *forward-declared* label (A/A/A, C, E/E) —
i.e. the **content**-anchored count, which is the opposite quantity from
"first option". Quoting it under the label "first option" while the next
sentence says "The bias breaks under reversal" mixes two denominators inside
one bullet, and the sentence that follows it ("Per-pair content stability
**contradicts itself**") is right about the pairs while the headline count
above it is not.

A second denominator problem sits in the code block the same claim is quoted
from: `picked_first_option_rate = 0.5`. That number is **not** a call-level
rate. `jev_voiceloop.py:659` computes `sum(r["all_first"] for r in rows) / n`
over the six *order-blocks*, so 0.5 = "3 of 6 blocks were all-first". It
coincides with 9/18 by arithmetic accident and invites the reader to read it
as a call rate.

**Correction:** replace "**Reversed order: 6/9 first option.**" with
"**Reversed order: 3/9 first option** — the bias breaks under reversal
(6 of the 9 reversed calls stayed with the pair's content label instead)."
And annotate the printed metric: `picked_first_option_rate = 0.5` (fraction of
the six order-blocks that were all-first, not a per-call rate).

### F8 — the same "6/9" in the lane's README

`voiceloop/README.md:253-256`:

> With genuinely indistinguishable options, forward order picked the
> first option **9/9** (reproducing the reported bias exactly) but reversed order
> only 6/9, and the per-pair readings contradicted each other

Same defect as F7, in the lane's summary form. Item 6 in the same document's
"Where I was wrong, collected" list — `voiceloop/README.md:289`, "**'Picks the
first 9/9' is order-specific.** Real forward, broken reversed" — is the
correct statement, and it is inconsistent with the 6/9 twenty lines above it.

**Correction:** "but reversed order only 3/9".

### F9 — the Coqui licence table sums to 93, and the document says 95

`voiceloop/README.md:155-171`:

> I parsed all 95 entries:
>
> | licence | count |
> |---|---|
> | apache 2.0 | 34 |
> | bsd-3-clause | 24 |
> | MIT | 11 |
> | CC-BY-SA 4.0 | 8 |
> | MPL | 7 |
> | CC-BY-4.0 | 3 |
> | **CPML (non-commercial)** | **2** |
> | **CC BY-NC-ND 4.0** | **2** |
> | openrail / custom | 2 |
>
> **Only 2 of 95 are the non-commercial CPML everyone remembers** … `tos_required` is
> `True` for exactly those 2 and `False` for the other 93.

34 + 24 + 11 + 8 + 7 + 3 + 2 + 2 + 2 = **93**. The table claims 95 entries two
sentences apart.

The lane's own detail file has the complete table —
`voiceloop/coqui-tts.md:24-36` — which has **eleven** rows, including a second
Apache row the README dropped:

| licence | count | coqui-tts.md |
|---|---|---|
| apache 2.0 | 34 | `:26` |
| bsd-3-clause | 24 | `:27` |
| MIT | 11 | `:28` |
| CC-BY-SA 4.0 | 8 | `:29` |
| MPL | 7 | `:30` |
| CC-BY-4.0 | 3 | `:31` |
| CPML | 2 | `:32` |
| CC BY-NC-ND 4.0 | 2 | `:33` |
| **Apache 2.0** | **2** | `:34` |
| openrail | 1 | `:35` |
| custom (jenny) | 1 | `:36` |
| | **95** | ✓ |

Two different spellings of the same licence ("apache 2.0" / "Apache 2.0") are
two rows in the source manifest, and the README collapsed the openrail and
custom rows while dropping the second Apache row entirely. The lane's
`SYNTHESIS.md:281` gets it right — "Apache-2.0 ×36" = 34 + 2 — which is what
the dropped row was for.

**Correction:** add the missing row `| Apache 2.0 | 2 |` before
`| openrail / custom | 2 |`, so the table sums to 95. (Optionally split
"openrail / custom" into the two rows the manifest has, as `coqui-tts.md`
does.)

### F10 — "93 of 95 models permissive" is 89

`voiceloop/README.md:282-284`:

> **Coqui is not "non-commercial."** MPL-2.0 code; **93 of 95 models permissive**;
> the non-commercial ones are XTTS and two CC BY-NC-ND models.

The sentence contradicts its own clause. XTTS is two models (`xtts_v2`,
`xtts_v1.1`, per `coqui-tts.md:45-46`) and there are two CC BY-NC-ND models,
so four are non-commercial; `openrail` (1) and `custom (jenny)` (1) are
restricted or unverified. 95 − 4 − 2 = **89**.

The lane's own detail file states the correct set — `coqui-tts.md:208-210`:

> 34 Apache-2.0 + 24 BSD-3 + 11 MIT + 7 MPL + 8 CC-BY-SA + 3 CC-BY + 2 Apache-2.0
> are all fine

34 + 24 + 11 + 7 + 8 + 3 + 2 = **89**. The README's "93" is the number of
models that are *not CPML* (95 − 2), used where the number of *permissive*
models belongs.

**Correction:** replace "93 of 95 models permissive" with "**89 of 95** models
commercially permissive (4 non-commercial, 2 restricted/unverified)".

---

## 2. The candidates

  - **C1 — the run-1 benchmark column has no file behind it.**
    `voiceloop/pipecat.md:160-164` presents a two-run table:

    | stage | run 1 median | run 2 median | p95 (runs 1/2) |
    |---|---|---|---|
    | log-mel features (pure numpy) | 7.78 ms | 6.52 ms | 13.38 / 10.98 ms |
    | ONNX inference | 60.88 ms | 72.87 ms | 89.75 / 99.19 ms |
    | **total** | **68.66 ms** | **79.39 ms** | — |

    Run 2 matches `docs/research/voiceloop/smartturn_bench.json` exactly
    (`feature_ms.median` 6.52, `feature_ms.p95` 10.98, `infer_ms.median` 72.87,
    `infer_ms.p95` 99.19, `total_ms_median` 79.39). **Run 1 has no counterpart
    on disk**: `bench_smartturn.py:128` writes to the single path
    `smartturn_bench.json`, so the second run overwrote the first, and
    7.78 / 13.38 / 60.88 / 89.75 / 68.66 appear nowhere in the repository
    except in the markdown that reports them. The same two-run summary is
    repeated at `voiceloop/README.md:186-188` and `voiceloop/SYNTHESIS.md:170-171`.
    *Suggested correction:* have the benchmark append runs (or write
    `smartturn_bench_run1.json`), or state in the table that run 1 was
    overwritten and is no longer recoverable from the repository.

  - **C2 — "distilled separates 0.19/0.81" has no source.**
    `TIMING.md:75` attributes the difference between naive and distilled gating
    to "0.19/0.81". The closest thing in the repository is the garbage→real
    column of `PROGRESS.md:765-770`, which reads "19% → 70%" (INTENTION),
    "27% → 89%" (GESTURE), "19% → 41%" (ATTENTION), "19% → 2%" (TIMBRE) — the
    0.19 appears, an 0.81 does not, and those columns are *percentages of two
    classes*, not a single separation. The claim needs its source named before
    it can be checked.

  - **C3 — the `knobs_are_separate` flag is described as a threshold test it
    is not.** `voiceloop/JEV-AS-INSTRUMENT.md:241` and `voiceloop/README.md:270`
    say the flag "fired `True` on the 0.02–0.03 differences" and that it was
    "fooled by a 0.02 difference on a flat field". The script's flag is
    `distinct == len(PROSODY_KNOBS)` (`jev_voiceloop.py:529-534`) — it fires
    because the three knob *vectors* are distinct, and it has no spread
    threshold to be fooled by. The file's `knobs_are_separate: true` and
    `distinct_control_vectors: 3` are correct; the narrative misdescribes the
    mechanism it is disowning, and the disowning is the part that makes the
    document useful.

  - **C4 — "short ~0.02" against a file value of 0.01.**
    `voiceloop/JEV-AS-INSTRUMENT.md:268` prints `all p: adaptive ~0.13, short
    ~0.02`. `jev_voiceloop_results.json` E7 `verdict_all_p` is
    `{"short_150ms": 0.01, "long_600ms": 0.85, "adaptive": 0.13, "none": 0.01}`.
    The adaptive value is right; the short value is off by a factor of two in
    its leading digit. Low impact (the verdict is unaffected) but it is a
    quoted number that does not match.

  - **C5 — the "1.6×" is quoted from the result file but is not derivable
    from it.** `ASCII-MOTION.md:192-193`:

    > beyond ~20 counts a dot is already **1.6×** more stable than one on the
    > threshold

    This reproduces `ascii_motion_threshold_results.json`'s own `consequence`
    string verbatim, so the note traces to its source faithfully. But the
    number does not follow from that file's own threshold table: a dot *more
    than ~20 counts* from the threshold is the `21-40` band at 34.48 % flip,
    and 67.25 ÷ 34.48 = **1.95×** (the 6–10 band gives 1.64×, which is the
    only band that lands on 1.6). The file understates its own effect, and the
    note repeats it in §8 "What follows" as if it were the lane's own
    conclusion, thirty lines below the table that does not produce it.
    *Suggested correction:* replace "1.6×" with "1.95×", or with the band's own
    flip rate ("34.5 % flip at 21–40 counts against 67.3 % on the threshold").

  - **C6 — "exactly +1.000 throughout" covers three lags, not six.**
    `ASCII-MOTION.md:102-104`:

    > Mean luma autocorrelation: lag 1 **+0.566**, lag 2 **−0.450**, lag 3
    > **−0.678**, lag 5 **+0.436**, against a static-repeat control of exactly
    > +1.000 throughout.

    `ascii_motion_cause_results.json` records
    `autocorr_static_control` as `{"1": 1.0, "2": 1.0, "3": 1.0, "4": null,
    "5": null, "6": null}`. The control is exactly 1.000 at lags 1–3 and
    *undefined* at lags 4–6 — and lag 5, one of the four lags the sentence
    names, is one of the undefined ones. "Throughout" states a coverage the
    file does not support. *Suggested correction:* "against a static-repeat
    control of exactly +1.000 at lags 1–3 (undefined beyond)".

---

## 3. Claims checked and found sound

**`docs/research/voiceloop/` (everything else).** `JEV-AS-INSTRUMENT.md` E1
matches exactly (15 probes, errors `0.0 ×10, 0.003333 ×3, 0.005 ×2`,
`max_abs_err` 0.0050000000000000044, `mean_abs_err` 0.0013333333333333522,
and the claim that the five non-zero errors are exactly 1/300 and 1/200);
E2's six-row table matches `holistic` / `max_field` / `argmax_field` for all
six inputs, including the 0.910 − 0.630 = **0.28** margin over a correct
complete sentence and the "max saturates toward 1.0" reading; E3 matches
(`first_option_forward` `fixed_energy_silence`, `first_option_reversed`
`speculative_gate`, pick `semantic_waveform_model` in both orders,
`picked_first_option_both_orders: false`); E4's latency set 40/150/180/620 ms
and its `correct: true` match, including the document's own admission that the
criteria restated the numbers so it was retrieval, not derivation; E5's
two-policy table matches `per_criterion` exactly and all three rankings
(`max`, `mean`, `holistic`) match `rank_by_max_of_criteria`,
`rank_by_mean_of_criteria`, `rank_by_holistic` in order, with
`max_and_mean_agree_at_top: true` and `holistic_and_max_agree_at_top: false`;
E7's short/long maxima, argmax fields and the `long_600ms` verdict at p=0.85
match. `pipecat.md` §2's behaviour table matches `smartturn_bench.json` row
for row (0.0495 / 0.8972 / 0.9619 / 0.9574 / 0.987) and its 8,679,182-byte
model size matches. `SYNTHESIS.md`'s 200 ms trigger / 3 s cap / 15× ratio and
its 8.3 MB model row are consistent with the same file.

**`docs/research/alwayson/OPENWAKEWORD.md` and `PYANNOTE.md`, everything
else.** The §2 size table matches `hotpath_measurements.json` byte for byte
(1,087,958 / 1,326,578 / 854,246 / 857,691 / 1,271,370 / 204,081 / 1,742,475 /
1,149,158 / 855,312), the 2.303 MB shared front and 0.195–1.662 MB per head
are the correct MB conversions of those bytes, and the §3 latency table matches
`openwakeword_latency` exactly (0.0641 / 0.8494 / 0.9199 / 0.0103 / 1.1269 /
1.6281 against the 80 ms budget). In `PYANNOTE.md`, the model sizes are right
(31.301 MB for community-1, 2.220 MB for the silero hot path, 14× larger,
81 % of it the embedding network), "16.9× more code than prose" is
798,995 ÷ 47,166 = 16.94 ✓, and the same document's silero ratio (6.6×) is
correct as far as it goes (F2 is about the basis, not the arithmetic).

**The ASCII-MOTION lane.** Every *measurement* in `ASCII-MOTION.md` matches
`ascii_motion_*_results.json`: the §2 frame statistics (median 1074/1920 =
55.9 %, min/max 890/1323, 24/24 frames needing downscale), the §3 lag sweep
(55.94 / 57.55 / 57.45 / 11.46 / 15.16 / 18.96 %, with the nulls at 0.00 %
and 100.0 %), the §4 autocorrelations (+0.566 / −0.450 / −0.678 / +0.436),
the 14.01-count mean-luma span against a 0.00 control, and the +0.702
cross-test correlation; the §5 leave-one-out table (all six drops) and the
7.71 % global minimum; and the §6 threshold table (67.25 / 49.27 / 48.39 /
40.96 / 37.71 / 34.48 / 21.61 %, the 3.11× end-to-end ratio, the 77.73 % dot
density and the 2p(1−p) = 34.62 % independence baseline). The §7 tone-channel
control (70.91 %) matches, and the banner's `A.nv12` gradient (0.94×) matches
`emptyroom_null_results.json`'s occupied `gradient_on_to_far` 0.9449, as do
the 50.00 % / 48.19 % flip rates. The only two defects are the derived
factors quoted from the file's prose rather than its table (C5) and the
"throughout" over a control that is undefined at three of six lags (C6).

**The cross-modal lane.** `CROSS-MODAL-FUSION.md` §8's ceiling row (n=34,
chance 0.25, AUC 0.334 — file 0.3338) and far-field row (n=34, LOO id 0.0,
AUC 0.451) match `crossmodal_probe_results.json`, as do the 90 s paired window,
the 34 labelled segments, the 20-band log-LTAS feature description, and the
F0 medians (98.5 Hz overall; 200 / 208 / 109 / 103 Hz per speaker). The
duration sub-claim (AUC 0.33 → 0.13) and corr(cos, |Δt|) = −0.05 match the
file's own interpretation note. The 111-of-314 wholly-overlapped segments and
the 1396-pause / 0.6225 base rate match `twospeaker_verify_results.json`
(two slices of 420.0 s each = **840 s** of meeting audio, which is what
`OTHER-REALM-IDEAS.md:131` claims) and `turn_eval_results.json`.

**The turn/route lanes.** `TWO-SPEAKER-TURNS.md` (AUC 0.529 over 1396 pauses,
best bar 0.5709, trivial baseline 0.6225, 0.647 and 0.685 at the adjacent
thresholds, 325) and `SPECTROGRAM-TURN.md` (0.5000 on the balanced sample,
0.6225 reweighted, pooled AUC 0.4567 with CI [0.3564, 0.5589], p_diff mean
0.6124 / std 0.0172, the 120-image sample with cross 39 / within 21, the
1/3 positive control with floor 1/3) both match their result files, and the
spectrogram note correctly treats the result as the negative it is.

**The gate/room lanes.** `EMPTY-ROOM-CONTROL.md` matches (occupied luma
111.18, MAD 43.686, flip 48.19 %, spread 33.93–56.82 %; empty luma 49.70, MAD
39.344, flip 50.00 %, spread 25.00–50.00 %; the 1.038× ratio; the
`empty1.nv12` 39.11 % / 1.22× row). `NOTCH-AB.md` matches `notch_ab_results.json`
on the R column (0.410 / 0.500 / 0.449 / 0.370), the attenuation table, the
S3 thresholds (50.73 / 1333.49 / 428.62), and the sub-threshold miss
(175.87 → 181.10 Hz). `MIC-HARDWARE.md` and `IN-THE-WILD-AUDIO.md` match
`mic_filter_sim_results.json` and `inthewild_probe_results.json`; the 22.04
in `IN-THE-WILD-AUDIO.md:33` is "Ubuntu 22.04.3", not a measurement.

**`TIMING.md`, `FLEET-MUSIC.md`, `GOLD.md`, `HOW-WE-KEEP-FOOLING-OURSELVES.md`,
`PAPERS.md`.** The judged-track latency range (214–554 ms, and the rounded
200–550 ms in the addendum) matches `PROGRESS.md:131`; the Day-3 correction
(56 annotated moments, p = 0.28) matches `EXPERIMENTS-AUDIO.md` §3h; the
~500 ms speculative-execution figure matches `PAPERS.md`'s 505 ms (Udupa et
al. 2026, arXiv:2606.13450); `FLEET-MUSIC.md`'s 500 ms cognitive beats and
9-track MIDI lineage are design facts. `HOW-WE-KEEP-FOOLING-OURSELVES.md`
reproduces the round-one retractions correctly and currently: the 14.5×
GATE_FIELD understatement, the 0.7293 peak at t = 0.064 s / frame 2, the
3.1× → 0.94× dither gradient, the 50.00 % / 48.19 % flip rates, and the
49.1 % watcher-cost maximum. `GOLD.md` is a proposal document — its thresholds
(AUC ≥ 0.65, ≥ 2 of 3, 90 % coverage) are targets, not claims; the one
recalled measurement it carries, "whole-line intent×RMS was +0.038", matches
`EXPERIMENTS-AUDIO.md` §3f.

**`CROSS-MODAL-FUSION.md` §7, `PAPERS.md`, `OTHER-REALM-IDEAS.md`,
`audio-dsp/`, `audio-engines/`, `audio-llms/`.** These quote upstream
repositories' published tables with the source named (OmniVLA's 84 % vs 25 %
RGB-only, TTS-AR's 88 % vs 40 %, FunASR's RTFx/CER table, VoxCPM's WER/SIM
table, aubio's `tol = 0.15` / `0.85`, the `pts.min.js` 200.3 KB tree listing).
They are outside this method's reach and are listed in §4. The one
locally-measured table in those lanes, `audio-dsp/SYNTHESIS.md:29-32`
(90.00 s clip, 32.72 s wall, 2.8× realtime, 3.64 ms/frame; 0.72 s clip,
0.18 s wall, 4.0×, 2.68 ms) is internally consistent — 90.00 ÷ 32.72 = 2.75 →
2.8×, 32.72 s ÷ 8,996 frames = 3.64 ms ✓, 0.72 ÷ 0.18 = 4.0 ✓ — and its
"3.6 ms per 10 ms hop … about 36 % of one core" follows from the same table.

**Root `README.md`, `CALIBRATION.md`, `LANE-FIXTURES.md`.** `CALIBRATION.md`
is the literal output format of `src/calibration.py`'s `main()`, and its
sample sizes are self-consistent with that script's corpus (GESTURE 5×3 = 15,
ATTENTION 4×3 = 12, INTENTION 4×3 = 12, TIMBRE 3+3 = 6, total 45 = the "45
calls" in the header). `README.md` repeats those calibration figures
(1.000/0.001, 0.917/0.097, Spearman 0.978, 1.4 vs 0.02) and the INTENTION
"CC74 = 121, gate 0.95" line, and `LANE-FIXTURES.md`'s fixture table is
consistent with it. `LANE-FIXTURES.md` is a task brief rather than a findings
note; its two operative numbers — "a low voice near FMIN (70 Hz) and a high one
near FMAX (400 Hz)" — match `src/pitch.py:53-54` (`FMIN = 70.0`, `FMAX =
400.0`). Note for the record that `judged_calibration_results.json`
is **not** the source of these per-track figures — it stores pooled ECE
(`ece_equal_width.mean` 0.05649, `ece_equal_mass.mean` 0.04927, `n_labelled`
192, `n_ambiguous` 10) for the later Phase 2.5 corpus, which is a different
measurement from the 45-call by-construction corpus `CALIBRATION.md` reports.

---

## 4. Claims I could not resolve

  - **Upstream-repository claims.** `PAPERS.md` (all ~40 arXiv entries and
    their quoted numbers), `prior-art-audit.md` (every claim is about another
    repository's source, licence or GitHub metadata), `audio-engines/*.md`,
    `audio-dsp/aubio.md` / `figaro.md` / `mechvibes.md` / `noisecraft.md` /
    `torch-audiomentations.md`, `audio-llms/audio-gpt.md` / `funasr.md` /
    `voxcpm.md`, `voiceloop/MeloTTS.md` / `coqui-tts.md`, and
    `OTHER-REALM-IDEAS.md` §3–§7. These name their source file/line in a
    repository that is not cloned here. They are not wrong as far as I can
    tell; they are simply outside a result-file audit. `prior-art-audit.md`
    also references `experiments/outputs`, a path that does not exist.
  - **`EXPERIMENTS-AUDIO.md` §3f / §3h and `TIMING.md`'s Day-3 correction.**
    Backed only by `/tmp/e7/`, which is not in the repository. Round one
    reached the same conclusion; I re-checked the numbers that appear in
    in-scope documents quoting them (`GOLD.md` H1's "+0.038", `TIMING.md`'s
    "56 annotated moments, p = 0.28") and found them mutually consistent,
    but neither is checkable against a result file. One internal oddity worth
    a future look: §3f reports "626 sentence-lines scored" while also
    reporting "160 top-quartile vs 328 mid-control lines" (160 + 328 = 488)
    and calling the pipeline a "629-line pipeline".
  - **The `audio-dsp` local measurement table** (`audio-dsp/SYNTHESIS.md:29-32`).
    Internally consistent, but no script or result file in the repository
    produced it; it is a one-off measurement recorded only in prose.
  - **`alwayson/OPENWAKEWORD.md` §8 and `PYANNOTE.md` §8 source-vs-prose
    tables.** The byte counts are used here as given; the repositories they
    were measured from are not cloned here, so only the *ratios* between the
    three tables could be checked, not the byte counts themselves.
  - **`CALIBRATION.md`'s per-track figures.** They trace to
    `src/calibration.py`'s own output format and to
    `docs/agents/zcode/HANDOFF.md:18`, not to a result file — the labels are
    by-construction and the run was live against the Jev API.

---

## 5. The process note

**The voiceloop lane is the one in this repository that measures things and
then does not check its own arithmetic against the file it just wrote.** Every
other lane's finding in this repo has been a number that drifted away from its
result file. Here the result file is present, correct, and *quoted* — and the
prose around it drifts anyway: a spread table that is in no file and that
contradicts the file's own numbers (F5, F6), a position-bias count that
silently changes denominator between the bullet that states it and the
sentence that interprets it (F7, F8), and a licence table that lost a row on
the way from the detail file to the summary file (F9, F10).

**Every one of them is a summarisation artefact, not a measurement
artefact.** The numbers in `jev_voiceloop_results.json` and
`smartturn_bench.json` are right. What failed is the step where a lane's
detail file is compressed into a summary: the spread row was recomputed by
hand and got wrong; the "6/9" was read off the wrong column of the E8 block;
the licence table was retyped and lost a row. The same is true of the two
`alwayson` source-vs-prose tables, where one ratio was mistyped and one
comparison was made across two different populations of source files.

**The check that would have caught all of them is the cheapest one in the
repo:** read the claim off the file rather than off the file you wrote
earlier. Five of this round's ten confirmed findings (F1, F5, F7, F9, F10) are
one transcription step away from being caught by `grep`.

---

*Round two: 49 documents audited (18 in `docs/research/`, 2 in
`docs/research/alwayson/`, 26 across `audio-dsp/` `audio-engines/`
`audio-llms/` `voiceloop/`, and 3 at the repo root); ~1,000 measurement-shaped
numeric claims traced to a result file, a script, or a named upstream table;
16 findings (10 confirmed, 6 candidates). Read with
[`NOTES-AUDIT.md`](NOTES-AUDIT.md), which covers the other 14 documents.*
