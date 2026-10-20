# The audit set's answer key changed under the quiz

**Date:** 2026-10-20
**Code:** `experiments/debrief_set_reaudit.py` · **Instrument:** `experiments/debrief.py`
**Results:** `experiments/debrief_set_reaudit_results.json`
**Tests:** `tests/test_debrief_reaudit.py` (18 tests)
**Supersedes:** the labels, and every score computed against them, in
`experiments/debrief_audit_*.json` and `experiments/debrief_cf-*.json` as
they stood on 2026-10-19. The old scores are kept, not overwritten, in
`the_key_change.the_before_snapshot` of the results file.

> ## CORRECTION (2026-10-20, later still) — three defects in the instrument, and one false sentence in the correction block above
>
> Everything below this block was measured with an instrument that had three
> defects in it, and with a **pool whose composition was being reported by a
> second copy of the code that built it**. All three are fixed and measured
> here.
>
> ### 1. THE "UNPARSEABLE" RUNS WERE ANSWERS CUT OFF, NOT ANSWERS THAT WERE NEVER GIVEN
>
> Three of the nine committed audit runs were recorded as `unparseable`: the
> model's reply was not valid JSON. The record said only "unparseable", and
> the natural reading is that the model failed to answer.
>
> It did not. It answered, and **the output budget ran out mid-answer**.
> `debrief.py::_http_json` rebuilt every response as `{"text": text}` and
> **discarded `finish_reason` and `usage`**, so a response cut off at the cap
> was indistinguishable on the wire from a finished one. `cloud_llm` already
> had a retry for reasoning models — it repeats the call with 4× the output
> budget when `content` comes back **empty** — and the truncated case never
> reached it, because a truncated answer is not empty, it is *cut*.
>
> **MEASURED** (`experiments/debrief_truncation_probe.py`, results in
> `experiments/debrief_truncation_probe_results.json`), qwen3-30b, blind
> mode, at the committed budget:
>
> | | n | truncated (`finish_reason="length"`) | recovered by 4× the output budget |
> |---|---|---|---|
> | committed probe | 8 | **1** (12.5%) | **1 of 1** (100%) |
> | an earlier independent probe | 6 | **2** (33.3%) | **2 of 2** (100%) |
> | pooled | 14 | **3** (21.4%) | **3 of 3** (100%) |
>
> Every truncation came back with `completion_tokens` **exactly 3000** — the
> cap — while the complete answers used 1,853 to 2,518.
>
> **THE OBVIOUS EXPLANATION WAS TESTED AND IT IS FALSE.** The natural
> hypothesis was that the *material* was too large for the smaller models'
> context windows, and that the fit ladder should have shrunk it. It cannot
> be that, and the control is structural: **in blind mode the material does
> not enter the prompt at all.** Measured at three material budgets:
>
> | material budget | prompt characters | prompt tokens |
> |---|---|---|
> | 140,000 | 3,031 | **701** |
> | 60,000 | 3,031 | **701** |
> | 40,000 | 3,031 | **701** |
>
> The prompt is identical at every budget, and the failure rate does not
> follow the budget — the failure happened at the **smallest** budget tested
> and not at the largest. **The budget that ran out was the OUTPUT budget,
> not the INPUT budget.** A fit ladder that treated "unparseable" as "does
> not fit" would have walked *down* past budgets that worked.
>
> **THE FIX.** `finish_reason` and `usage` are kept and surfaced; a
> truncated answer is retried with 4× the output budget; every run records
> its transport facts; and the two unparseable shapes that used to share
> one label are now told apart — `response_was_truncated` versus
> `no_json_verdicts_object`. Pinned by `tests/test_debrief_transport.py`
> (10 tests, including a mutation test).
>
> **WHAT IT COST BEFORE THE FIX, beyond the three lost runs.** A truncated
> answer can still *parse*: the JSON object closes before the cut and the
> model keeps writing prose after it. The old scorer had no way to know the
> answer was short, so **the claims the model never got to answer were
> scored as wrong** — a truncated answer was silently charged to the
> director.
>
> **THE RE-RUN: all nine committed audit runs now parse.** The §4.1 table
> below is the re-run.
>
> ### 2. THE LOCAL PATH OF THIS INSTRUMENT HAD NEVER WORKED
>
> `experiments/debrief.py` has always supported `ollama:` and `local:`
> models. **Every such call returned `no_result` and no local model was
> ever scored.** `_http_json` understood only Cloudflare's response shape
> (`{"result": {"choices": [...]}}`); ollama puts the message at the top
> level (`{"message": {"role": "assistant", "content": ...}}`), so the
> answer was sitting in the body in plain sight behind a shape check.
>
> This matters because the hosted endpoint's non-determinism contaminates
> every number in this document — a measured spread of 0.619 for the hosted
> vision model, and gpt-oss's sensitivity ranging over 0.333 to 1.000 on
> **identical prompts**. The one thing that could have removed that noise
> was a local model, and the local path was broken.
>
> ### 3. THE CORRECTION BLOCK ABOVE IS WRONG ABOUT THE ACOUSTICS FIGURE
>
> The block above says: *"the acoustics figure (**"73.5"**) **now reaches the
> prompt** — it did not before the fix — and the director accuses the claim
> anyway"*, and uses that to argue the failure **is not a starvation
> problem**.
>
> **Measured against the pool actually used, the acoustics evidence does
> NOT reach the prompt.** The only `73.5` in the pool is
> `AMI EN2002a **73.5%** (191 of 260 gaps)` — an unrelated number from a
> different experiment:
>
> | probe | in the pool? |
> |---|---|
> | `73.5` (as a bare substring) | yes — but it is `73.5%`, an unrelated figure |
> | `-73.5 dBFS` | **no** |
> | `73.496` | **no** |
> | `119 transients` | **no** |
> | `RT60` | **no** |
>
> The three notes that hold "119 transients" all lose it to the per-note cap.
> So the director's accusation of the one true claim is explained by the
> **missing evidence**, exactly as §4.3 says — and the sentence above is
> **withdrawn**, along with the conclusion drawn from it.
>
> ### 4. THE POOL'S OWN DIAGNOSTIC WAS A SECOND COPY OF THE POOL BUILDER
>
> `evidence_reachability` builds its per-note diagnostic from its own
> hand-rolled copy of the keep-set — and that copy still carried the **old**
> front-truncation. So it was measuring a pool that was **not** the pool the
> runs were made against: it reported **3 of 49** notes keeping their tail
> while the pool actually used kept **51 of 51**. A copy of a thing that
> measures the thing is a second implementation of it, and it drifts.
> Collector and diagnostic now call one slicer.
>
> ### 5. THE PROPOSAL CHANGED BETWEEN TWO IDENTICAL RUNS
>
> The same model, on the same corpus, given the same evidence pool, produced
> a **different** proposal:
>
> | | title | Jaccard vs the previous committed proposal |
> |---|---|---|
> | previous committed | "Cache-Pressure Gate-Cost Test" | — |
> | this re-run | "Held-out-device CPU-cost measurement of Silero-VAD gate" | **0.2604** |
>
> **0.2604 is below this repository's own ≈0.29 "same experiment" threshold.**
> So the earlier headline — "Jaccard **1.000**, the proposal is *literally the
> same experiment*" — was a **single-sample number**, and the second sample
> of the same measurement is 0.2604. The echo numbers in this document are
> single-sample statistics and are now labelled as such.
>
> ### 6. WHAT SURVIVED ALL OF IT
>
> - **The proposal still cannot fail.** The null band (a difference ≤ 0.5 %
>   absolute) sits entirely inside the falsifier band (a reduction ≤ 5 %),
>   so a null outcome is a rejection. That has now been measured on **two**
>   independently generated proposals.
> - **It still skips Condition D** — the take this repository ranks first,
>   and the one that is actually recordable.
> - **It still proposes plan item 11**, the item already marked as blocked
>   on hardware. This run's `the_proposal_names_plan_item_11` is **true**;
>   the previous run's was false. The topical overlap is stable; the wording
>   is not.
> - **The second defect is gone.** The previous proposal asked for two
>   opposite things (a hypothesis of *less than* 1.5× and a margin of *at
>   least* 1.5×). This one does not: its hypothesis and its margin agree.
>
> The old text below is kept visible, per the house rule that a falsified
> claim stays in the record.

---

> ## CORRECTION (2026-10-20, later that night) — the evidence pool was broken, and the numbers below were measured against the broken one
>
> This lane measured the director's score against an evidence pool that was
> **silently dropping the retractions**. `debrief.py` builds a keep-set
> holding each note's head, its tail and every digit-line, and then cut the
> assembled text **from the front** — so every over-long note lost its
> tail, and the dated correction blocks live in the tails. **2 of 50 notes
> kept their tail.** The pool is now fixed (50 of 50 keep both ends), and
> this lane was **re-run against the fixed pool**; the numbers in this
> document have been updated. The full account, including an A/B of the two
> pools against one corpus, is `docs/research/DEBRIEF-MATERIAL-POOL.md`.
>
> **WHAT THE RE-RUN CHANGED:**
>
> | | this document's first version | re-run, fixed pool |
> |---|---|---|
> | context repeats, gpt-oss-120b | 0.000 | **−0.333** |
> | context repeats, llama-3.3-70b | −0.667 | −0.667 |
> | context repeats, qwen3-30b | −0.556 | **−0.778** |
> | blind repeats | +0.333 / −0.778 / +0.111 | **unchanged** |
> | sweep, 20,000 chars | −0.444 | **−0.222** |
> | sweep, 240,000 chars | **+1.000** | **−0.111** |
> | sweep, effect size | +1.444 | **+0.111** |
> | echo: Jaccard vs the previous committed proposal | 0.312 | **1.000** |
> | echo: max overlap with a repo next step | 0.312 | **0.0598** |
>
> **AND THE ONE RESULT THAT IS NOW THE HEADLINE, BECAUSE IT IS THE ONE THAT
> IS ROBUST:**
>
> **Specificity is 0.000 in all 9 context runs, and 1.000 in the blind runs
> for two of the three models.** The director keeps the one TRUE claim when
> it is given the claim alone, and **accuses it once the evidence is added**.
> That is the failure, and it is not a starvation problem: the acoustics
> figure (**"73.5"**) **now reaches the prompt** — it did not before the
> fix — and the director accuses the claim anyway.
>
> **AND THE "+1.000 AT 240,000 CHARACTERS" IS WITHDRAWN.** It does not
> reproduce: three independent fixed-pool runs at 240,000 give −0.111, and
> a paired old-pool/new-pool run at 240,000 gave **identical per-claim
> verdicts**. The budget ladder's measured effect with the fixed pool is
> **+0.111**, not +1.444. Note the confound honestly: the two sweeps ran
> against **different corpora** (four research notes and six corrections
> landed in between), so the difference between them cannot be attributed
> to the pool alone. What the pool's effect IS, was measured with the
> corpus held constant, and it is in `DEBRIEF-MATERIAL-POOL.md`.
>
> The old text below is kept visible, per the house rule that a falsified
> claim stays in the record.

---

## 1. What happened

On 2026-10-19 every camera reader in this repository was reading the
captures at `W*H*3//2` = 23,040 bytes per frame, and the files carry 19,200.
80% of every "frame" was a tear between two real moments. The full account
is `docs/research/NV12-STRIDE.md`; the mean consecutive |diff| on A.nv12
fell 43.560 → 0.545.

`experiments/debrief.py` audits an LLM "director" against ten claims: seven
this project has retracted or falsified, three that were still standing.
**The ground truth was written against the torn frames.** Two of the three
standing claims were standing on numbers computed on those tears, and the
evidence under two of the seven retracted claims has been replaced.

> ### The rule, pre-registered before any model was called
>
> A claim is TRUE if and only if the CURRENT results files support it **AS
> WRITTEN**. A claim whose cited numbers have been withdrawn is FALSE as
> written, **even if a related conclusion survives on different evidence**.
>
> The second sentence decides two of the ten claims, so it is worth stating
> out loud: the item the model is asked to judge is the claim in front of
> it. If the words cite a number that no longer reproduces, the claim is
> false, and a director that endorses it has made an error — even where the
> subject has since been re-measured and the new measurement is favourable.

Every label in this document is **derived** by a check that reads a results
file, not asserted in prose. `experiments/debrief_set_reaudit.py` holds one
check per claim, each returning a boolean plus the deciding numbers;
`tests/test_debrief_reaudit.py` re-runs them, so a label that drifts back
without its evidence drifting with it fails the build.

The claim TEXTS were not edited. Only the key moved. The blind prompt's
SHA-256 is unchanged at
`5870ad6bb1989c07727c57ea01c417aa0a014674824d48ee8a692665e6e1100c`, so
the blind re-run is a controlled repeat of the committed one and the
before/after comparison is a comparison of keys, not of instruments.

---

## 2. The re-audit: ten claims, ten verdicts

| # | claim (abbreviated) | before | after | what decides it now |
|---|---|---|---|---|
| 1 | the gate's field threshold removed 14.5x of camera drift | FALSE | FALSE | `gate_peak_verdict_results.json`: the worst ratio over the record's OWN interval is **1.0**. Unchanged, and audio. |
| 2 | the ASCII map detects motion (48.19% vs 50.00%) | FALSE | FALSE | `row_band_removal_results.json`: the empty room now flips **0.00%**, the occupied capture 0.62%. Both numbers were the tear. The claim's numbers do not reproduce, so the claim is false as written — even though `ascii_motion_nulls_results.json` now verdicts **SEPARATED**, i.e. the conclusion survives on different evidence. |
| 3 | the watcher is quadratic and its buffer grows without bound | FALSE | FALSE | `watcher_cost_results.json`: every growth exponent is below 1.2, and below 0.1 beyond four seconds. Unchanged, and audio. |
| 4 | gain normalisation cut drift from 43.3 to 0.81 counts (~50x) | FALSE | FALSE | `emptyroom_null_results.json`: the drift is **0.615 counts before any normalisation**. The 43.3 was the tear. Same label, new reason — there was no 43.3 to reduce. |
| 5 | a local LLM can label speech from an ASCII rendering | FALSE | FALSE | `local_vlm_vision_results.json`: the local model answers SPEAKING to the speaking, the silent **and the blank** condition — all three rates 1.000, margin over the blank control **0.000**. |
| 6 | 31.8% of frames are never measured | FALSE | FALSE | `watcher_cost_results.json`: 31.8% is not in any measured cell. It was never a measurement, and it was a CPU percentage, not a count of frames. |
| 7 | row-band removal took the empty room from 50.00% to 0.01%, leaving the subject alone | **TRUE** | **FALSE** | `row_band_removal_results.json`: with NO removal the empty room flips **0.00%** — there is no banding to remove — and removal makes the occupied rate **worse** (1.79% against 0.62%). Its premise WAS the tear. |
| 8 | the teacher scored 1.000 on a temporal split and 0.000 held out | FALSE | FALSE | `teacher_conventional_speaking_map.json`: the temporal split is still 1.000, but the held-out session now scores **0.15 (A) and 0.375 (B)**, and the 1.000 is matched by a scrambled control at 0.9375 with a blank control at the 0.667 base rate. The teacher has learned nothing, not the recording. |
| 9 | the room is silent at -73.5 dBFS, 119 transients, zero valid RT60 | TRUE | **TRUE** | `room_map_results.json`: **-73.496 dBFS**, 119 transients, an empty RT60 list. It is an AUDIO measurement, so the stride could not have touched it. The only survivor. |
| 10 | composed filters cut the occupied flip rate 0.0299 → 0.0014 (21x) at a flat SNR of ~1.83 | **TRUE** | **FALSE** | `combined_filter_results.json`: on the correct frames the composition moves the occupied flip rate by **1.14x**, not 21x, and the SNR is **1.386** against **1.410** for the RAW variant. Every number in the claim was measured on torn frames, and the premise is withdrawn. |

**The set now holds NINE false claims and ONE true one.** It held seven and
three before. Two labels changed; both changes are in
`experiments/debrief.py`'s `AUDIT_SET`, and the old text is preserved in a
dated correction block in that file's docstring, next to the two earlier
ones.

### Two things follow, and they are not tidy

**Specificity is now a one-bit measurement.** With one true claim, specificity
is 0 or 1, and the headline is sensitivity minus 1 plus one bit. The
degenerate baselines still work — a director that rejects everything and a
director that accepts everything both score **0.000** under the new key
(`tests/test_debrief_reaudit.py::test_the_degenerate_directors_still_score_zero`)
— but the set has lost most of the resolution it had. The remedy is to add
tonight's standing claims as new true items. **That was not done**: it is a
change to the item set, and changing the items mid-measurement changes the
instrument. It is recorded here as a recommendation.

**Two of the ten claims are now false for a reason the notes can explain**
— their numbers were the tear. A director that has read the notes should now
catch two claims it could not have caught before, and §4.3 shows that this is
exactly what happened.

### The composed-filter claim's TEXT was never fixed, and that is worth recording

`debrief.py`'s docstring says this claim was "corrected" once. What was
corrected was its **label**, not its **text**, and the text still asserts
0.0299, 0.0014, 21x and an SNR of 1.83 — four numbers that no longer
reproduce. It is kept as it is, because the claim is the item the director
is asked to judge, and a director that endorses withdrawn numbers has made
an error. §4.3 shows that the director did endorse it, and §4.3 also shows
why: the note that owns the claim still leads with the old table.

---

## 3. The key change, isolated from everything else

The same verdict rows can be scored under the old key and the new key, so
the effect of the key is measured with no model involved at all. The
"before" column is the committed record as it stood; the "after" column is
**the same rows, re-scored**.

| mode | model | BEFORE (old key) | same rows, NEW key | moved |
|---|---|---|---|---|
| blind | llama-3.3-70b | -0.048 | **-0.778** | **-0.730** |
| blind | gpt-oss-120b | +0.429 | +0.333 | -0.095 |
| blind | gpt-oss-120b | +0.095 | +0.444 | +0.349 |
| blind | qwen3-30b | +0.143 | +0.111 | -0.032 |
| context | gpt-oss-120b | +0.857 | +0.667 | -0.190 |
| context | gpt-oss-120b | +0.857 | +0.667 | -0.190 |
| context | llama-3.3-70b | +0.095 | -0.667 | **-0.762** |

**Yes, the key change moved the director's score — by up to 0.76 — and it
moved it in the right direction.** The reason is mechanical and worth
stating: two claims the director had been **endorsing** (row-band removal,
composed filters) became **false** claims, so the same endorsement stopped
being a correct answer and became a miss. The corrected key penalises
exactly the failure mode this repository cares about.

It also moved the score *shape*. Under the old key a director that read the
notes could keep two of the three standing claims and score 0.667 on
specificity. Under the new key there is one standing claim, and a director
that misses it loses everything.

---

## 4. The audit, re-run against the corrected key

### 4.1 The committed artefacts

All four committed audit artefacts and the debrief were re-run with
`experiments/debrief.py` itself, so they match the corrected key:

> **CORRECTION (2026-10-20, later still) — this table is the re-run on the
> instrument with the three defects fixed, and every run in it now parses.**
> The previous table is kept below it.
>
> | artefact | mode | model | sens | spec | headline |
> |---|---|---|---|---|---|
> | `debrief_audit_blind_cf-meta-…_cf-openai-…_cf-qwen-….json` | blind | llama-3.3-70b | 0.222 | 0.000 | **-0.778** |
> | | blind | gpt-oss-120b | 0.333 | 1.000 | **+0.333** |
> | | blind | qwen3-30b | 0.111 | 1.000 | **+0.111** |
> | `debrief_audit_blind_cf-openai-…_cf-meta-…_cf-qwen-….json` | blind | gpt-oss-120b | 0.333 | 1.000 | **+0.333** |
> | | blind | llama-3.3-70b | 0.222 | 0.000 | **-0.778** |
> | | blind | qwen3-30b | 0.111 | 1.000 | **+0.111** |
> | `debrief_audit_context_cf-openai-gpt-oss-120b.json` | context | gpt-oss-120b | 0.889 | 0.000 | **-0.111** |
> | `debrief_audit_context_cf-openai-…_cf-meta-….json` | context | gpt-oss-120b | 0.778 | 0.000 | **-0.222** |
> | | context | llama-3.3-70b | 0.333 | 0.000 | **-0.667** |
>
> **The contrast the previous version was built on is sharper now, because
> the three lost runs are back.** Read the **specificity** column against
> the **mode** column, for the same model and the same material budget:
>
> | model | blind (claims only) | context (claims + evidence) |
> |---|---|---|
> | gpt-oss-120b | **1.000** — keeps the one true claim | **0.000** — accuses it |
> | qwen3-30b | **1.000** — keeps it | **0.000** — accuses it |
> | llama-3.3-70b | 0.000 — accuses it | 0.000 — accuses it |
>
> **Two of the three models keep the true claim when given the claim
> alone, and accuse it once the evidence is added.** That is the failure,
> and it is the one result in this document that did not move when the
> instrument was fixed.
>
> The previous table, measured with the three runs lost to truncation and
> the pool's second copy of itself:
>
> | artefact | mode | model | sens | spec | headline |
> |---|---|---|---|---|---|
> | `debrief_audit_blind_cf-meta-…_cf-openai-…_cf-qwen-….json` | blind | llama-3.3-70b | 0.222 | 0.000 | **-0.778** |
> | | blind | gpt-oss-120b | 0.222 | 1.000 | +0.222 |
> | | blind | qwen3-30b | 0.111 | 1.000 | +0.111 |
> | `debrief_audit_blind_cf-openai-…_cf-meta-…_cf-qwen-….json` | blind | gpt-oss-120b | 0.222 | 1.000 | +0.222 |
> | | blind | llama-3.3-70b | 0.222 | 0.000 | **-0.778** |
> | | blind | qwen3-30b | 0.444 | 1.000 | +0.444 |
> | `debrief_audit_context_cf-openai-gpt-oss-120b.json` | context | gpt-oss-120b | 0.889 | 1.000 | **+0.889** |
> | `debrief_audit_context_cf-openai-…_cf-meta-….json` | context | gpt-oss-120b | 0.778 | 0.000 | **-0.222** |
> | | context | llama-3.3-70b | 0.556 | 0.000 | **-0.444** |
>
> Look at rows 7 and 8 in that table: **the same model, the same material
> budget, minutes apart, with opposite specificity** — one kept the single
> true claim, the other accused it. One run is a sample, not a score.

### 4.2 The repeats, and the spread

The hosted endpoint is not deterministic at temperature 0 (measured in this
repository at a 0.619 headline spread for the hosted vision model), so every
headline below is a median over repeats with its spread. The model is held
constant inside each row and the material budget is fixed per model, so
each row is one prompt repeated.

| mode | model | n | headline median | min | max | **spread** | sens (median) | spec (median) |
|---|---|---|---|---|---|---|---|---|
| blind | gpt-oss-120b | 4 | **+0.333** | +0.111 | +0.333 | 0.222 | 0.333 | 1.000 |
| blind | llama-3.3-70b | 4 | **-0.778** | -0.778 | -0.778 | 0.000 | 0.222 | 0.000 |
| blind | qwen3-30b | 3 (+1 unparseable) | **+0.111** | +0.111 | +0.111 | 0.000 | 0.111 | 1.000 |
| context | gpt-oss-120b | 3 | **0.000** | -0.111 | 0.000 | 0.111 | 1.000 | 0.000 |
| context | llama-3.3-70b | 3 | **-0.667** | -0.667 | -0.667 | 0.000 | 0.333 | 0.000 |
| context | qwen3-30b | 3 | **-0.556** | -0.667 | -0.444 | 0.111 | 0.444 | 0.000 |

> **CORRECTION (2026-10-20, later still) — the repeats, re-measured with the
> fixed instrument and the fixed pool.**
>
> | mode | model | n | headline median | min | max | **spread** | sens (median) | spec (median) |
> |---|---|---|---|---|---|---|---|---|
> | blind | gpt-oss-120b | 4 | **+0.333** | +0.111 | +0.333 | 0.222 | 0.333 | 1.000 |
> | blind | llama-3.3-70b | 4 | **-0.778** | -0.778 | -0.778 | 0.000 | 0.222 | 0.000 |
> | blind | qwen3-30b | **4** | **+0.111** | +0.111 | +0.111 | 0.000 | 0.111 | 1.000 |
> | context | gpt-oss-120b | 3 | **-0.333** | -0.333 | -0.222 | 0.111 | 0.667 | 0.000 |
> | context | llama-3.3-70b | 3 | **-0.667** | -0.667 | -0.667 | 0.000 | 0.333 | 0.000 |
> | context | qwen3-30b | 3 | **-0.778** | -0.778 | -0.667 | 0.111 | 0.222 | 0.000 |
>
> **Three things changed and one did not.**
>
> - The blind arm is **unchanged** in every cell.
> - The **qwen3 blind arm now has n=4, not "3 + 1 unparseable"**: the lost
>   run was a truncated answer, and with the output-budget retry it is
>   recovered. Its headline is +0.111, the same as the other three.
> - The **two context cells that changed got worse**, not better:
>   gpt-oss from 0.000 to **-0.333**, qwen3 from -0.556 to **-0.778**.
> - **What did not change: specificity.** Every context cell still scores
>   **0.000**, and every blind cell for gpt-oss and qwen3 still scores
>   **1.000**. The director still keeps the one true claim when given the
>   claim alone and still accuses it once the evidence is added.

**The director is worse than chance in context mode, and the reason is
measurable.** The chance baseline on this key — a director that answers
"unsupported" or "supported" at random on each claim — has an expected
sensitivity of 0.5 and an expected specificity of 0.5, so an expected
headline of **0.000**. (The two degenerate directors also score 0.000: one
that rejects everything scores 1.0 + 0.0 − 1, one that accepts everything
scores 0.0 + 1.0 − 1.) Against that baseline:

- blind gpt-oss **+0.333** and blind qwen **+0.111** are above chance;
- blind llama **-0.778** is far below it;
- context gpt-oss **0.000** is exactly chance — and it is worth seeing how:
  its median sensitivity is **1.000** (it caught all 9 of the 9 false
  claims) and its median specificity is **0.000** (it accused the one true
  claim every time). Perfect rejection, perfect accusation;
- context llama **-0.667** and context qwen **-0.556** are far below it.

**Two of the three models are worse than a coin flip in context mode**, and
§4.3 says why. The same rows scored under the OLD key give blind gpt-oss
+0.286, llama -0.048, qwen +0.143; context gpt-oss 0.000, llama +0.095,
qwen -0.095. So the *new* key costs the context arm roughly 0.1–0.8 and the
blind arm roughly 0.03–0.73. The corrected key is harder, and the reasons
are in the next section.

### 4.3 The error pattern — and half of it is the harness, not the model

The parent's question was whether the director's false accusations cluster
on a class of claim. They do, and the class is **claims whose deciding
number did not reach the prompt** — though the harness does not explain
the whole pattern, and the section says so where it does not.

`collect_materials` keeps each note's head, its tail, and every line
containing a digit — and then **caps** the result at a per-note budget. The
cap is applied *after* the keep-set is built, so a note whose kept text is
longer than the cap loses its tail, and the tail is where this repository
puts its retraction notices. The corpus grew tonight: **49 notes**, a
per-note cap of **2,057 characters**, and **47 of 49 notes are truncated**.
Only **3 of 49** keep their tail.

> **CORRECTION (2026-10-20, later still) — this paragraph described the
> BROKEN pool, and the diagnostic that measured it was measuring a second
> copy of the pool builder.**
>
> **The pool, as it is now:** **51 notes**, a per-note cap of **1,976
> characters**, **50 of 51 notes truncated** — and **51 of 51 keep their
> tail.**
>
> The old numbers were not wrong about the broken pool. They were wrong
> about **which pool**. `evidence_reachability` builds its per-note
> diagnostic from its own hand-rolled copy of the keep-set, and that copy
> still carried the **old front-truncation** (`text[:per]`) in it. So the
> diagnostic was describing a pool that was **not** the pool the runs were
> made against — and it said "3 of 49 keep their tail" about a pool that
> kept every tail.
>
> **A copy of a thing that measures the thing is a second implementation of
> it, and it drifts.** The collector and the diagnostic now call one
> slicer, so they cannot disagree again.
>
> **AND THE FIGURE THAT DECIDES THE ONE TRUE CLAIM IS NOT IN THE POOL.**
> Measured over the materials actually used:

> | probe | in the pool? |
> |---|---|
> | `73.5` (bare substring) | yes — but only as `AMI EN2002a **73.5%** (191 of 260 gaps)`, an unrelated figure |
> | `-73.5 dBFS` | **no** |
> | `73.496` | **no** |
> | `119 transients` | **no** |
> | `RT60` | **no** |
>
> The three notes that hold "119 transients" — `DEBRIEF-MATERIAL-POOL.md`,
> `EMPTY-ROOM-CONTROL.md` and `LLM-DIRECTOR.md` — all lose it to the
> per-note cap. So the director's accusation of the one true claim is
> explained by the **missing evidence**, and the sentence in the top
> correction block that says the acoustics figure "now reaches the prompt"
> is **withdrawn**.

(The pool measured here is the pool the runs were made against: this
document is excluded from it, and it has to be — it quotes the deciding
numbers in its own tables below, so a pool that included it would report
the evidence as present when the director never saw it. The exclusion is
checked in
`tests/test_debrief_reaudit.py::test_the_reachability_check_excludes_this_document`,
and the code that builds the pool is verified byte-identical to the
harness's own collector.)

Measured over the materials only (not the claims, which carry the numbers
themselves — the first version of this check searched the whole prompt and
reported that every number was present, which was wrong in the flattering
direction):

| claim | deciding numbers | in the prompt? | what the director did |
|---|---|---|---|
| the gate's 14.5x | "14.5", "GATE_FIELD" | yes | **rejected it — 9/9 correct** |
| the ASCII motion claim | "48.19", "50.00" | yes | **rejected it — 9/9 correct** |
| the gain normalisation claim | "43.3", "0.81" | yes | **rejected it — 9/9 correct** |
| the row-median claim | "0.01", "1.79" | yes | rejected it — 5/9 correct |
| the watcher's quadratic claim | "quadratic", "39.13" | yes | 3/9 correct |
| the local-LLM claim | "Qwen2-VL", "blank" | yes | 3/9 correct |
| the 31.8% claim | "31.8" | yes | 3/9 correct |
| the composed-filter claim | "0.0014", "1.83", "21" | yes (the **old** numbers) | endorsed it — 3/9 correct |
| the teacher claim | "1.000", **"0.375"** | **0.375 is NOT there** | endorsed it — 4/9 correct |
| the acoustics claim | **"73.496"**, "119 transients" | **73.496 is NOT there** | rejected it — 3/9 correct |

The join:

| | claims | runs | correct | accuracy |
|---|---|---|---|---|
| evidence **reached** the prompt | 8 | 72 | 44 | **0.611** |
| evidence did **not** reach the prompt | 2 | 18 | 7 | **0.389** |

And two of the failures have reasons the model wrote down, and the reasons
are *true of the prompt*:

- The acoustics claim was rejected with: **"No cited evidence provides the
  -73.5 dBFS floor, 119 transients, or RT60 count."** The value 73.496 is
  in `room_map_results.json` and in `EMPTY-ROOM-CONTROL.md`, and **neither
  slice reaches the prompt**. The director is wrong about the world and
  right about the prompt it was given. This is the same failure
  `docs/research/LLM-DIRECTOR.md` §5 recorded on 2026-10-19, and the fix
  applied then — an equal share per note, plus head and tail, plus digit
  lines — **does not survive the corpus growth**, because the final
  truncation can still remove the tail.
- The composed-filter claim was **endorsed**, because `COMBINED-FILTER.md`
  still leads with the withdrawn table and its own correction block is
  outside the per-note cap. The old numbers (0.0014, 21x) are in the
  prompt; the applicable retraction is not. The word "PREMISE IS WITHDRAWN"
  does appear in the materials — inside `TEACHER-RAW-REPRESENTATION.md`,
  where it is about the teacher's numbers, not this claim.

**And the harness does not explain everything.** The join above is 0.611
against 0.389 over ten claims, and the per-model view is sharper still.
llama-3.3-70b, whose verdicts were identical in all three context runs:

| claim | evidence in the prompt? | llama's verdict, 3/3 runs |
|---|---|---|
| the gate's 14.5x | yes | **rejected it — correct** |
| the ASCII motion claim | yes | **rejected it — correct** |
| the gain normalisation claim | yes | **rejected it — correct** |
| the watcher is quadratic | yes | uncertain — wrong |
| the row-median claim | **yes** ("1.79" is there) | **endorsed it — wrong, and the model was at fault** |
| the local-LLM claim | yes | endorsed it — wrong |
| the 31.8% claim | yes | endorsed it — wrong |
| the teacher claim | **no** ("0.375" is not there) | endorsed it — the harness's fault |
| the composed-filter claim | the OLD numbers, no applicable retraction | endorsed it — the harness's fault |
| the acoustics claim | **no** ("73.496" is not there) | **accused it — the harness's fault** |

gpt-oss-120b, in the same mode, got 8 or 9 of the 9 false claims right in
every run, and got the same two wrong: the acoustics claim (accused or
hedged 3/3) and the composed-filter claim (endorsed 3/3). It caught the
teacher claim 3/3.

So: the acoustics claim, the teacher claim and the composed-filter
endorsement are **harness failures** — the model was right about the
prompt it was given, and the prompt was wrong. llama's endorsement of the
row-median claim, and the local-LLM and 31.8% endorsements, are **model
failures**: the evidence was present and the director did not use it. Both
kinds are in the same error pattern, and the distinction matters, because
only the first kind is fixed by handing the director more of the record.

### 4.4 One more contamination, reported and not fixed

The CONTEXT material pool is `experiments/*.json`, and the audit artefacts
live there. The **per-claim key does not leak** — each results file gets
only the first ~395 characters, and the per-claim labels sit far deeper
(verified: `is_false_claim` appears in no slice). But the **previous run's
aggregate scores do leak**, and from them the class balance is recoverable
(0.2222 = 2/9, 1.0 = 1/1). The balance was 7:3 before tonight and is 9:1
now, so the contamination itself changed with the key. It is reported, not
fixed, because fixing it means changing the instrument.

### 4.5 Three of the nine runs were answers that were cut off

This section is the evidence behind the first item of the top correction
block, and it is the one measurement in this document that had to **change
a hypothesis to produce a number**.

**THE SYMPTOM.** Three of the nine committed audit runs were recorded as
`unparseable`: the model's reply was not valid JSON. The record said only
"unparseable", and the natural reading is that the model failed to answer.

**THE OBVIOUS HYPOTHESIS, AND WHY IT IS FALSE.** The obvious explanation
is that the *material* was too large for the smaller models' context
windows, and that the fit ladder should have shrunk it. It cannot be that,
and the reason is structural rather than statistical: **in blind mode the
material does not enter the prompt at all.** `build_audit_prompt` returns
the claims and nothing else for blind mode, so the prompt is identical at
every material budget.

Measured (`experiments/debrief_truncation_probe.py`, results in
`experiments/debrief_truncation_probe_results.json`):

| material budget | prompt characters | prompt tokens | parsed? |
|---|---|---|---|
| 140,000 | 3,031 | **701** | yes |
| 60,000 | 3,031 | **701** | no |
| 40,000 | 3,031 | **701** | yes |

Same prompt, three different budgets, and the outcome varies. A ladder
that treated "unparseable" as "does not fit" would have walked *down* past
budgets that worked.

**THE ACTUAL CAUSE.** `debrief.py::_http_json` rebuilt every response as
`{"text": text}` and **discarded `finish_reason` and `usage`**. A response
cut off at the output cap is then byte-for-byte indistinguishable from a
finished one. `cloud_llm` already had a retry for reasoning models — it
repeats the call with 4× the output budget when `content` comes back
**empty** — and the truncated case never reached it, because a truncated
answer is not empty, it is *cut*.

**THE RATE, AND THE RECOVERY.**

| | n | truncated (`finish_reason="length"`) | recovered by 4× the output budget |
|---|---|---|---|
| committed probe | 8 | **1** (12.5%) | **1 of 1** (100%) |
| an earlier independent probe | 6 | **2** (33.3%) | **2 of 2** (100%) |
| pooled | 14 | **3** (21.4%) | **3 of 3** (100%) |

Every truncation came back with `completion_tokens` **exactly 3000** — the
cap — while the complete answers used 1,853 to 2,518.

**AND IT COST MORE THAN THREE LOST RUNS.** A truncated answer can still
*parse*: the JSON object closes before the cut and the model keeps writing
prose after it. The old scorer had no way to know the answer was short, so
**the claims the model never got to answer were scored as wrong**. A
truncated answer was silently charged to the director.

**THE FIX.** `finish_reason` and `usage` are kept and surfaced; a truncated
answer is retried with 4× the output budget; every run records its
transport facts; and the two unparseable shapes that used to share one
label are told apart — `response_was_truncated` versus
`no_json_verdicts_object`. Pinned by `tests/test_debrief_transport.py`
(10 tests, including a mutation test that fails if the truncation
detection is removed).

**THE RE-RUN: 9 of 9 committed audit runs now parse.**

### 4.6 The local path had never worked, and the variance is not the model's

This section is the evidence behind the second item of the top correction
block, and it is the **new method** of the night: measure the thing where
the measurement is variance-free.

**THE PROBLEM.** Every number this repository holds about the director is
contaminated by the serving stack. The same prompt, sent to the same
hosted model, gives different answers: the hosted vision model's headline
spread was measured at 0.619, and gpt-oss-120b's sensitivity was 0.889,
0.333 and 1.000 on **three identical prompts**. A difference between two
arms of this experiment cannot be separated from that noise, and more
repeats do not remove it — it is the endpoint changing its mind, not
variance around a mean.

**THE OBSTRUCTION.** `experiments/debrief.py` has always accepted
`ollama:` and `local:` models. **Every such call returned `no_result`, and
no local model was ever scored.** `_http_json` understood only
Cloudflare's response shape (`{"result": {"choices": [...]}}`); ollama puts
the message at the **top level**
(`{"message": {"role": "assistant", "content": ...}}`). The answer was in
the body the whole time, behind a shape check. So the one thing that could
have removed the noise from every number in this document was broken, and
had been broken for as long as it had been offered.

**THE MEASUREMENT** (`experiments/debrief_local_director.py`, results in
`experiments/debrief_local_director_results.json`), blind mode, the same
prompt repeated four times:

| local model | parsed | headline, repeated | **spread** |
|---|---|---|---|
| `qwen2.5-coder:7b` | 4 / 4 | −1.000 ×4 | **0.0000** |
| `gemma2:2b` | 4 / 4 | −0.778 ×4 | **0.0000** |
| `llama3.2:1b` | 1 / 4 | +0.111 | — (3 runs failed to parse) |
| `qwen3:0.6b` | 0 / 4 | — | format failure |
| `tev1` | 0 / 4 | — | `http400`, "does not support chat" |

The two local models that answered every time are **perfectly
deterministic over four identical prompts**, while the hosted arms carry
spreads of **0.111** and **0.222** on identical prompts.

**AND THE PRE-REGISTERED RULE FIRED, AND ITS VERDICT IS WRONG.** The rule
was written before the run: *"the variance is the serving stack if the
local spread is 0.000 on the models that can answer, while the hosted
spread is not."* Both halves hold, so the rule fired and the results file
says **"THE VARIANCE IS THE SERVING STACK"**.

**That verdict does not survive two checks, and both are recorded in the
results file.**

**FIRST: THE RULE CONFOUNDS LOCALITY WITH REASONING.** Every local model
that could answer here is a **non-reasoning** model. Every hosted cell
that varies is a cell whose model writes a **reasoning preamble** before
its answer. So the comparison cannot separate "local" from "reasoning" —
and this machine has no local reasoning model to break the tie. A design
that cannot separate two explanations should not name one.

**SECOND, AND THIS IS THE BETTER-CONTROLLED MEASUREMENT: A CONTROLLED
CHANGE THAT DOES NOT MOVE THE QUANTITY.** §4.7 re-measured the hosted
repeats after the transport fix, on the same prompts with the same key.
**The hosted spread did not move** — median spread 0.0555 before, 0.0555
after. If the serving stack were the cause, fixing the transport would
have moved it. It did not.

**SO THE VARIANCE IS NOT EXPLAINED BY LOCALITY.** §4.7 finds where it
actually is, and it is not a transport property at all.

**AND THE LOCAL MODELS ARE WORSE DIRECTORS, WHICH IS NOT THE POINT.** Their
headlines are −1.000 and −0.778, against the hosted arms' best of +0.333.
These models are 10× to 100× smaller than the hosted ones. What they are
for is not a better score; it is a **variance-free score**, so that a
difference between two arms can finally be attributed to the arm — which
makes them the right instrument for the next experiment even though their
own scores are worse.

**THE LOCAL SAMPLE IS SMALL AND THAT IS STATED.** Two local models
answered every time; one answered once in four; two never answered in the
right shape; one does not support the chat endpoint. That is enough to
show the local path works and to make a variance-free measurement
possible. It is **not** enough to prove the serving stack is the whole of
the variance, and §4.7 shows it is not.

### 4.7 The variance is not the transport, and the metric cannot resolve it

This section is the measurement that decides what the rest of this document
is allowed to claim, and it **falsifies the hypothesis it was built to
test**.

**THE HYPOTHESIS.** §4.5 and §4.6 point at one mechanism: a reasoning
model spends part of its output budget on prose, the prose length varies,
and that decides both whether the answer is cut off and how many verdicts
are missing. If so, fixing the transport should shrink the variance.

**THE TEST.** The same repeats, the same prompts, the same key, the same
corpus, re-measured with the fixed client — and compared against the
measurements taken with the old one.

| cell | old median | old spread | new median | new spread |
|---|---|---|---|---|
| blind gpt-oss-120b | +0.333 | 0.222 | +0.333 | **0.111** |
| blind llama-3.3-70b | −0.778 | 0.000 | −0.778 | 0.000 |
| blind qwen3-30b | +0.111 | 0.000 | +0.111 | 0.000 |
| context gpt-oss-120b | −0.333 | 0.111 | −0.111 | 0.111 |
| context llama-3.3-70b | −0.667 | 0.000 | −0.667 | 0.000 |
| context qwen3-30b | −0.778 | 0.111 | −0.778 | **0.222** |

**The hypothesis is false.** The median spread is **0.0555** before the fix
and **0.0555** after it. One cell halved and one cell doubled; the
resolution did not move. The truncation fix bought back three lost runs
and repaired the silent corruption — and it did **not** make the director
more repeatable.

**WHERE THE VARIANCE ACTUALLY IS.** With the transport clean, every one of
the 21 re-measured runs carries **10 of 10 verdicts** and **no truncation
and no retry**. So the remaining spread is the model genuinely changing
its mind — and it changes its mind in exactly one place:

| cell | headlines, in order | **uncertain counts**, in order |
|---|---|---|
| blind gpt-oss-120b | +0.333, +0.444, +0.333, +0.333 | **0, 0, 1, 1** |
| context qwen3-30b | −0.778, −0.556, −0.778 | **5, 3, 4** |
| context gpt-oss-120b | −0.111, −0.111, 0.000 | 0, 0, 0 |
| blind llama-3.3-70b | −0.778 ×4 | 0, 0, 1, 0 |

**The headline moves when a claim flips to "uncertain", and nowhere else.**

**AND HERE IS WHAT THAT MEANS FOR THE METRIC.** The audit set has **nine
false claims and one true one**, so sensitivity is a count over nine, and
the headline `sens + spec − 1` moves in steps of

> **1 / 9 = 0.1111**

**The observed run-to-run spreads are 0.111 and 0.222 — one and two
quanta.** So the entire variance of this instrument is *one or two claims
out of nine flipping to "uncertain"*, and the instrument's **resolution is
the same size as its noise**.

**THE CONSEQUENCE, AND IT IS THE ONE THAT MATTERS.** The effect claimed in
§7 — "more evidence makes the director better" — is **+0.1111**. That is
**exactly one quantum**. It is the smallest number this metric can
express, and it is the same size as the spread of a single run of the same
model on the same prompt. **The experiment cannot distinguish "more
evidence helps" from "one claim happened to be answered differently this
time."** The same is true of the difference between the fixed-pool and
broken-pool medians in §4.2, and of the +0.095 and +0.111 differences in
the committed artefacts.

**WHAT TO DO ABOUT IT, WHICH IS THE NEXT EXPERIMENT.** The metric is
quantised and the noise is the same size as the quantum, so:

1. **Report in claims, not fractions.** "The director caught 3 of the 9
   false claims" is a statement about the world. "+0.333" is a statement
   about 3/9 with three decimal places of false precision.
2. **Compare arms in PAIRED counts, not in medians.** The same model, the
   same prompt, one thing changed — the number of claims caught is a
   count, and a paired count has no quantum problem.
3. **The audit set needs more true claims.** With **one** true claim,
   specificity is a single bit and moves in steps of **1.000**, so
   specificity is *coarser* than sensitivity. That is why the specificity
   column in §4.1 is all 0.000 and all 1.000: it has two values, and the
   entire context-vs-blind story is one bit flipping.

**This is not a criticism of the models. It is a statement about the
instrument's resolution, and it should have been made before the first
comparison was run, not after the eleventh.**

### 4.8 The comparison the instrument's resolution can actually support

This section is the **new method**, and it replaces §7's comparison with
one the metric can resolve.

**THE ARITHMETIC OF THE OLD DESIGN, STATED BEFORE THE NUMBERS.** The
sweep compared medians of **three repeats per budget**. A two-sided exact
sign test on three paired samples has a smallest attainable p-value of
**2/8 = 0.25**. So the sweep's design **could not have produced a
significant result at p < 0.05 at any effect size whatsoever** — not at
+1.444, not at +0.1111, not at +10. The verdict label it was given,
*"EFFECT RESOLVED"*, was not available to that design.

**AND THE CLAIM-LEVEL TEST IS ALSO SMALL.** The exact McNemar test's
p-value depends on the **smaller arm of the split**, which makes it
conservative. On nine claims:

| split (gained : lost) | p (two-sided) | significant at 0.05? |
|---|---|---|
| 9 : 0 | 0.0039 | **yes** |
| 8 : 1 | 0.0391 | **yes** |
| 7 : 2 | 0.1797 | no |
| 6 : 3 | 0.5078 | no |
| 5 : 4 | 1.0000 | no |
| 3 : 6 | 0.5078 | no |
| 0 : 9 | 0.0039 | **yes** |

**On nine claims the test rejects only when at least 8 of the 9 move in
one direction.** A shift of 6 claims one way and none the other gives
p = 0.031. So the *minimum detectable effect* for this instrument is
**6 of 9 claims** — against a claimed effect of **one claim**.

**THE EXPERIMENT** (`experiments/debrief_paired_analysis.py`, results in
`experiments/debrief_paired_analysis_results.json`). Model held constant,
budget varied, three repeats per budget, and every comparison done two
ways: on the paired repeats, and on the **nine claims** with each claim
aggregated across its repeats.

**AND HERE IS THE TRAP, WHICH IS THE POINT OF THE WHOLE SECTION.** The
first implementation pooled the repeats and ran McNemar on **27** pairs —
the same nine claims measured three times, treated as 27 independent
observations. That is anti-conservative, and it is not a small
correction. The experiment was then **run twice**, and the two runs are
the evidence:

**The claim-level test — the pre-registered unit, nine claims:**

| step | run A | run B |
|---|---|---|
| 20,000 → 60,000 | gained 1, lost 3, **p = 0.6250** (worse) | gained 1, lost 2, **p = 1.0000** (worse) |
| 60,000 → 140,000 | gained 3, lost 0, **p = 0.2500** (better) | gained 2, lost 0, **p = 0.5000** (better) |
| 140,000 → 240,000 | gained 0, lost 0, **p = 1.0000** — all nine claims agree | gained 0, lost 0, **p = 1.0000** — all nine claims agree |

**NEVER SIGNIFICIFICANT, AT ANY STEP, IN EITHER RUN.** And the *direction*
is stable across both runs: the 20,000 → 60,000 step goes the **wrong
way** both times, and the 140,000 → 240,000 step is a **tie** both times —
in both runs the two budgets are literally indistinguishable, with all
nine claims agreeing.

**The pooled-over-repeats test — the same nine claims counted three
times:**

| step | run A | run B |
|---|---|---|
| 20,000 → 60,000 | p = 0.5078 | p = 1.0000 |
| 60,000 → 140,000 | **p = 0.0391 — "significant"** | p = 0.2188 — not significant |
| 140,000 → 240,000 | p = 0.5078 | p = 1.0000 |

**The pooled statistic crosses the 0.05 line in one run and not in the
other.** That is the whole lesson in one line: counting the same nine
claims three times as twenty-seven observations does not add evidence, it
adds **variance** — enough variance to manufacture a p of 0.039 out of
nothing on one run and not on the next. The claim-level test never
crossed the line in either run.

**The script's first implementation violated its own pre-registered rule**
— which says *"exact McNemar on the nine false claims"* — and the fix is
in the code now, with the pooled figure kept and labelled so the mistake
stays visible. The committed results are run B; run A's numbers are
above and in the run log.

**THE CONTROLS, AND THEY COME OUT THE OTHER WAY.**

- **The null arm** — a budget compared with **itself** — gives **0
  discordant claims and p = 1.0000**, with all three repeats tied. The
  machinery is sound.
- **The reversed pairing** (240,000 → 20,000) reports the **opposite
  direction**: gained 0, lost 1, p = 1.0000, direction
  `more_evidence_is_worse`. The test can come out the other way.
- The sign test on the paired repeats reports a **minimum attainable p of
  0.25** at n = 3, in every cell, which is the arithmetic above stated
  as a number.

**THE VERDICT, AT THE CORRECT UNIT: "MORE EVIDENCE MAKES THE DIRECTOR
BETTER" IS NOT SUPPORTED.** No step of the budget curve is significant,
in either run, at the unit the rule names. The last step is the cleanest
possible null: **the two arms agree on all nine claims**, twice.

**AND THE DIRECTION IS NOT MONOTONE.** The step from 20,000 to 60,000
goes the **wrong way** in both runs.

**WHAT THE INSTRUMENT WOULD NEED INSTEAD.** Not a bigger effect — a
bigger *audit set*. With nine false claims, the minimum detectable shift
is six of them. To detect a shift of one or two claims at p < 0.05, this
instrument needs roughly **forty** false claims, drawn from the
repository's own settled findings. §9's finding 3 said the audit set
needs more true items; it needs more items **on both sides**, because
specificity is currently a **single bit** and moves in steps of 1.000.

---

## 5. The proposal audit

The debrief was re-run on the repository's current state:
`python experiments/debrief.py --debrief --models "@cf/openai/gpt-oss-120b"`,
committed as `experiments/debrief_cf-openai-gpt-oss-120b.json`. It proposes
**"Cache-Pressure Gate-Cost Test"**: measure the gate's CPU advantage under
cache pressure on a Pi-class target.

Audited against this repository's own standard — the same standard the
audit set applies to claims:

| the standard | verdict |
|---|---|
| **Has a falsifier** — a result that would have made it come out the other way? | **Yes.** "If the observed cost ratio under pressure is ≥ 0.90 (i.e. < 1.1× advantage) or the gated configuration costs more than the ungated one, the hypothesis is falsified." |
| **Has a control that can come out the other way?** | **Yes.** Three controls (the pressure manipulation is real; a baseline; an ungated comparison), none phrased as a certainty. |
| **Has a null?** | Yes: "Cost ratio (ungated / gated) = 1.0 (no advantage)." |
| **Has a margin?** | Yes: "the gated configuration must achieve a cost ratio ≤ 0.67 (i.e. ≥ 1.5× lower CPU usage)". |
| **Can the proposal come out the other way at all?** | **NO — and this is the finding.** Two mechanical contradictions: |

1. **The null triggers the falsifier.** The null is a ratio of 1.0; the
   falsifier fires at a ratio ≥ 0.90. A null result and a falsifying result
   are the same observation, so the null cannot do its job.
2. **The hypothesis and the margin disagree.** The hypothesis says the
   advantage shrinks to *less than* 1.5×; the margin demands *at least*
   1.5×. The proposal asks for two opposite things and one of them was
   written by mistake.

Both are checked mechanically in
`experiments/debrief_set_reaudit.py::decision_structure` and recorded in the
results file. A proposal with all five required fields that still cannot
fail is the shape of claim this repository has had to retract four times.

| **Does it duplicate what the repository already knows?** | **YES.** The whole proposal has a content-token Jaccard of **0.312** against the proposal the committed record made the day before ("Gate-advantage under Pi-class cache constraints") — the same experiment, reworded. The repository's own threshold for "same experiment" is ≈0.29 (`LLM-DIRECTOR.md` §4). It is also `docs/research/ADOPTION-PLAN.md` §7 item 11. |
| **Does it propose the repository's own top next step?** | **No.** It does not mention **Condition D — turned AND silent**, which `PROGRESS.md`, `docs/research/POSE-FEATURE.md` and `docs/research/NV12-STRIDE.md` §8 all name as *the single most valuable missing take*. |
| **Does it propose anything the repository has already falsified?** | **Not the experiment** (max Jaccard against the repository's own retraction list: 0.018). **But its REVIEW of the evidence does.** The debrief's list of what the evidence SUPPORTS includes: *"Composing row-band removal with a 3-frame temporal median cuts the occupied-flip rate to 0.0014 % (≈21× improvement over row-only)"* — the claim this document retracted, endorsed as supported, at a Jaccard of **0.667** against the repository's own retraction. It did correctly flag the row-band removal claim as unsafe, and it correctly caught tonight's new retraction. So it caught one of the two and endorsed the other. §4.3 says why. |

> **CORRECTION (2026-10-20, later still) — a DIFFERENT proposal, and a
> different set of defects.**
>
> The debrief was re-run on the fixed instrument. It proposes
> **"Held-out-device CPU-cost measurement of Silero-VAD gate"**, not the
> cache-pressure test above. Same experiment, different wording — and the
> wording is not stable between runs.
>
> | the standard | verdict on the NEW proposal |
> |---|---|
> | **Has a falsifier?** | **Yes.** "If the measured reduction is ≤ 5 % or the confidence interval includes zero, the hypothesis is rejected and the gate's cost-saving claim is abandoned." |
> | **Has a control that can come out the other way?** | **Yes.** Two controls: the ungated watcher on the same device, and a cold-start flush to isolate cache effects. |
> | **Has a null?** | Yes: "No reduction: gated and ungated CPU usage are equal (difference ≤ 0.5 % absolute)." |
> | **Has a margin?** | Yes: "The gated condition must beat the ungated baseline by ≥ 5 % absolute core-time … and the 95 % CI of the difference must not include zero." |
> | **Can the proposal come out the other way at all?** | **NO — and this is still the finding.** **One** mechanical contradiction: |
>
> 1. **The null triggers the falsifier.** The null is a difference of
>    ≤ 0.5 %; the falsifier fires at ≤ 5 %. Every null outcome is a
>    falsifying outcome, so the null cannot do its job. The overlap is
>    mechanical and is recorded in the results file.
>
> **The second contradiction is gone.** The previous proposal asked for two
> opposite things — a hypothesis of *less than* 1.5× and a margin of *at
> least* 1.5× — and one of them was written by mistake. This proposal's
> hypothesis and margin agree. So across two independently generated
> proposals: **the null-inside-the-falsifier defect is stable, and the
> self-contradictory margin was not.**
>
> **AND THE REVIEW OF THE EVIDENCE IS CLEANER THIS TIME.** The debrief
> supports 7 claims, endorses **0** of the repository's falsified claims,
> and flags **0**. The previous run endorsed a withdrawn claim at a
> Jaccard of 0.667 because the note that owned it still led with the old
> table. That note has since been corrected, and the endorsement went
> with it — which is the *only* evidence in this document that the pool
> fix actually reached the model.
>
> **Does it duplicate what the repository already knows?** Maximum overlap
> with a repository next step: **0.0575** (closest: `held_out_device`),
> well under the repository's ≈0.29 threshold. And
> `the_proposal_names_plan_item_11` is now **true** — the previous run's
> was false. So the proposal is the plan item the repository has already
> marked as blocked on hardware.
>
> **Does it propose the repository's own top next step?** **No.**
> `mentions_condition_d` is **false** and the overlap with Condition D is
> **0.0083**.

> **CORRECTION (2026-10-20, later that night) — the echo numbers are now
> measured against the fixed pool, and they are STRONGER, not weaker.**
> The re-run gives a Jaccard of **1.000** against the previous committed
> proposal — the proposal is now *literally the same experiment*, not a
> rewording of it — and a maximum overlap with a repository next step of
> **0.0598**, well under the repository's own ≈0.29 "same experiment"
> threshold. So the answer is sharper in both directions: **it is the
> previous experiment verbatim, and it is not any of the repository's own
> next steps** (Condition D scores **0.000** against it).
>
> The **0.312** above is the number from the run against the broken pool
> and is left visible, per the house rule.
>
> > **AND THE "1.000, VERBATIM" NUMBER IS WITHDRAWN — IT WAS A SINGLE
> > SAMPLE.** The block above says the proposal is *literally the same
> > experiment*, at a Jaccard of 1.000. A later run of the same model, on
> > the same corpus, with the same evidence pool, produced a **different**
> > proposal at a Jaccard of **0.2604** — *below* this repository's own
> > ≈0.29 "same experiment" threshold. So the echo statistic is a
> > single-sample number with a measured run-to-run range of
> > **[0.2604, 1.000]**, and the honest statement is:
> >
> > **The proposal is topically the same experiment every time — it is
> > always the held-out-device measurement of the Silero gate's CPU cost,
> > which is `ADOPTION-PLAN.md` §7 item 11 — and it is never Condition D,
> > which the repository ranks first.** The wording, the exact hypothesis
> > and the exact set of defects move between runs. The Jaccard against the
> > previous proposal is therefore not a measure of "how much the director
> > echoes the repository"; it is mostly a measure of how much the hosted
> > endpoint varies.

---

## 6. The echo question

> **Does the LLM director add information beyond the repository's own notes,
> or does it echo them?**

**It echoes them.** The evidence:

- Its proposal is the previous committed proposal reworded (Jaccard 0.312,
  the repository's own "same experiment" threshold is ≈0.29), and that
  experiment is `ADOPTION-PLAN.md` §7 item 11 — a plan item already marked
  **NEXT** and already marked as *not runnable on this box*, since it needs
  a Pi-class target.
- Its `biggest_risk` is tonight's own headline, almost verbatim: *"Residual
  frame-stride bugs still hidden in any script that reads NV12 captures"* —
  which is `NV12-STRIDE.md` §9.
- It did **not** propose Condition D, the take the repository itself ranks
  first and the one that is actually recordable tonight with the new
  capture tool.
- It endorsed the repository's own withdrawn number because the note that
  owns it still leads with the old table.

**The interesting possible outcome — a next experiment the notes have not
considered — did not happen.** The director picked the plan item that the
repository has already marked as blocked on hardware, and did not pick the
one that is ready to record. That is the honest finding, and it is a
negative one: **as an instrument for proposing the next experiment, this
director adds nothing that a careful reader of the notes does not already
have.**

Where it does add something, it is *negative* information, and that is
worth having: the proposal it produced **fails this repository's own
standard** (null and falsifier overlap; hypothesis and margin contradict),
and the audit that catches that is cheap and mechanical. So the useful role
for this instrument is **auditing the plan, not writing it** — and that is
also what the audit set is for.

> **CORRECTION (2026-10-20, later still) — the echo is in the CHOICE of
> experiment, not in everything around it, and the Jaccard is mostly
> measuring the endpoint.**
>
> The re-run on the fixed instrument gives a **more nuanced** answer than
> "it echoes them", and the nuance is worth more than the verdict.
>
> **WHAT STAYS THE SAME — the choice of experiment.**
>
> - The proposal is again the **held-out-device measurement of the Silero
>   gate's CPU cost**. That is `ADOPTION-PLAN.md` §7 item 11 — a plan item
>   already marked **NEXT** and already marked as *not runnable on this
>   box*, since it needs an ARM target. `the_proposal_names_plan_item_11`
>   is **true**.
> - It still does **not** propose **Condition D — turned AND silent**, the
>   take the repository ranks first and the one that is actually recordable
>   tonight with the fixed capture tool. `mentions_condition_d` is
>   **false**; the overlap with Condition D is **0.0083**.
>
> **WHAT CHANGED — the parts around the choice.**
>
> - Its `biggest_risk` is **no longer an echo**. The previous run's risk
>   was tonight's own headline almost verbatim ("Residual frame-stride bugs
>   still hidden in any script that reads NV12 captures" — `NV12-STRIDE.md`
>   §9). This run's is about the experiment itself: *"The held-out-device
>   measurement may show that the Silero-VAD gate provides no meaningful CPU
>   savings on target hardware, invalidating the primary performance
>   justification for the gate."*
> - It endorsed **0** of the repository's falsified claims, against the
>   previous run's endorsement of a withdrawn claim at a Jaccard of 0.667.
>
> **AND THE JACCARD IS NOT A STABLE MEASUREMENT.** Two runs of the same
> model, on the same corpus, with the same pool, produced proposals at a
> Jaccard of **1.000** and **0.2604** against each other — a range of
> **0.740** across two samples of the same measurement. The repository's
> own "same experiment" threshold is ≈0.29, so the two runs fall on
> **opposite sides** of it.
>
> **So the honest answer to the question this section asks is:**
>
> **The director echoes the repository's PLAN — it keeps re-proposing the
> plan item the repository has already marked as blocked on hardware, and
> it keeps not proposing the one that is ready to record.** It does **not**
> echo the repository's prose: its risk statement, its hypothesis and its
> review of the evidence all move between runs, and one of those reviews
> was clean.
>
> **The useful role for this instrument is auditing the plan, not writing
> it.** That conclusion is unchanged — and it is now better supported,
> because the thing that makes the instrument useless for writing the plan
> has been identified: it proposes what the notes already contain, and
> what the notes already contain is what it should be auditing.

---

## 7. Does more evidence make the director better?

Model held constant (`@cf/openai/gpt-oss-120b`), budget varied, the
corrected key throughout, 3 repeats per budget. One run at the 60,000
budget returned an unparseable response and is recorded as a failed run
rather than silently dropped.

| material budget | headlines | n | median | spread | same runs, OLD key |
|---|---|---|---|---|---|
| 20,000 chars | -0.222, -0.444, -0.556 | 3 | **-0.444** | 0.333 | -0.286 |
| 60,000 chars | -0.222, -0.222 (+1 unparseable) | 2 | **-0.222** | 0.000 | -0.286 |
| 140,000 chars | 0.000, 0.000, -0.111 | 3 | **0.000** | 0.111 | 0.000 |
| 240,000 chars | +0.889, +1.000, +1.000 | 3 | **+1.000** | 0.111 | +0.333 |

> **CORRECTION (2026-10-20, later that night) — the table above was measured
> against the BROKEN evidence pool, and its 240,000-character arm does not
> reproduce.** The re-run against the fixed pool:
>
> | material budget | headlines | n | median | spread |
> |---|---|---|---|---|
> | 20,000 chars | -0.222, -0.222, -0.222 | 3 | **-0.222** | 0.000 |
> | 60,000 chars | 0.000, +0.778, -0.222 | 3 | **0.000** | 1.000 |
> | 140,000 chars | 0.000, -0.222, -0.111 | 3 | **-0.111** | 0.222 |
> | 240,000 chars | -0.111, -0.111, -0.111 | 3 | **-0.111** | 0.000 |
>
> The rule's verdict is unchanged — **EFFECT RESOLVED** — but the effect is
> **+0.1111**, not +1.444, and the "+1.000 at 240,000 characters" is
> **withdrawn**: three independent fixed-pool runs at 240,000 give −0.111,
> and a paired old-pool/new-pool run at 240,000 gave identical per-claim
> verdicts. See the correction block at the top of this document, and
> `docs/research/DEBRIEF-MATERIAL-POOL.md`.

**The null was** "more evidence does not make the director better", and the
rule was fixed before the runs: the effect is called only if the median at
the larger budget beats the median at the smaller one by more than the
run-to-run spread observed at the smaller one. The delta is **+1.444**
against a spread of 0.333 at the small budget. **EFFECT RESOLVED: more
evidence makes the director much better, monotonically.**

The shape of that curve is the point, and the last column is the point
twice over. At 20,000 characters the director is below chance. At 240,000 it
catches all 9 of the 9 false claims *and* keeps the one true claim — a
headline of +0.889 to +1.000, which **the old key could not produce at any
budget** (its best median here is +0.333). The bottleneck tonight was never
the model's reasoning; it was how much of the record reached the prompt.
**The night added a lot of evidence, and the director is only useful if it
is given enough of it.**

> **CORRECTION (2026-10-20, later still) — the two paragraphs above are
> the record of the BROKEN pool and they are kept visible. The conclusion
> they state does not hold.**
>
> The **+1.444** delta and the "**much better, monotonically**" verdict are
> measured against the pre-fix pool, and the correction block above already
> withdrew them: the fixed-pool sweep gives **−0.222 / 0.000 / −0.111 /
> −0.111** across 20,000 / 60,000 / 140,000 / 240,000 characters, and the
> effect is **+0.111**, not +1.444.
>
> **The shape of the fixed-pool curve is flat, and that is the honest
> result: the director does not get better with more evidence.** Four
> budgets, three repeats each, and the medians sit between −0.222 and
> 0.000 with a run-to-run spread of up to **1.000** at the 60,000 budget.
> A spread of 1.000 in a single cell is larger than every effect the sweep
> claims to have measured.
>
> **And the "it keeps the one true claim at 240,000" claim is withdrawn
> with it.** In the fixed pool it does not: the true claim is accused at
> every budget, in every repeat — which is exactly what §4.1 shows on the
> committed artefacts (context specificity **0.000** in all 9 context runs).
>
> **What the earlier run was actually measuring** — and this is the part
> that was right — is the paragraph before the withdrawn verdict: the
> bottleneck is how much of the record reaches the prompt, not the model's
> reasoning. That still holds. It is the *size of the effect* that was
> wrong, and it was wrong because the broken pool's contents were doing
> the work.
>
> > **AND §4.8 RE-ANALYSES THIS SAME QUESTION IN THE UNIT THE METRIC CAN
> > RESOLVE, AND THE ANSWER IS "NOT SUPPORTED".** The comparison above
> > compares medians of three repeats. §4.8 pairs the arms claim by claim,
> > and the pre-registered claim-level test gives **p = 0.2500** for the
> > only step that looked significant under an anti-conservative pooled
> > test, and **p = 1.0000** (the two arms agree on all nine claims) for
> > the last step. No step of this curve is significant, the direction is
> > not monotone, and the design's smallest attainable p at three repeats
> > is **0.25** — so this sweep could not have answered the question at
> > any effect size. See §4.8.

---

## 8. What this does NOT establish

- **The director is not measured at its best.** The best arm in this
  document is a 240,000-character context budget, which is roughly double
  the budget the committed artefacts use. The committed scores are a
  *starved* director, and this lane did not change the committed budget
  because that is a change to the instrument.
- **One true claim is not a specificity measurement.** Every specificity
  number in this document is a single bit, and a single bit cannot
  distinguish "careful" from "lucky".
- **The key change and the material change are not separated in the new
  runs.** §3 separates them on the OLD rows, where the materials were the
  old materials. The new runs have the new materials, so the difference
  between §3 and §4 mixes two causes; §3 is the one that isolates the key.
- **The proposal was audited, not re-run.** The cache-pressure test needs
  hardware this machine does not have. Nothing here says whether it would
  have worked.
- **The "more evidence" result is one model at four budgets.** The
  direction is large (+1.444) and monotone, but the budget that wins is
  the budget that was tried, and 240,000 characters may not be the optimum.
  One repeat at the 60,000 budget came back unparseable, so that arm has
  n=2.
- **The debrief was run once.** Its proposal is stable (Jaccard 0.312
  against the previous run's), but the review of the evidence is not
  measured across repeats.

> **CORRECTION (2026-10-20, later still) — the list above is the list as it
> stood before the instrument was fixed, and four of its items are now
> wrong. They are kept visible. The items that still hold are marked.**
>
> - ~~**The director is not measured at its best.**~~ **STILL HOLDS, but
>   for a different reason.** The committed budget is 140,000 characters
>   and the sweep does not show a better budget: the fixed-pool curve is
>   flat (§7). So the committed scores are not a *starved* director — they
>   are the director.
> - ~~**The "more evidence" result is one model at four budgets. The
>   direction is large (+1.444) and monotone.**~~ **WITHDRAWN.** The
>   direction is **+0.111** and the curve is flat, and the run-to-run
>   spread at one budget is **1.000** — larger than the effect.
> - ~~**The debrief was run once. Its proposal is stable.**~~ **WITHDRAWN,
>   AND IT IS THE OPPOSITE.** The proposal was run twice on the same
>   corpus with the same pool and produced a **Jaccard of 0.2604** between
>   the two — so the *wording* is not stable. What is stable is the
>   *choice*: both proposals are the held-out-device CPU-cost measurement,
>   and neither is Condition D.
> - **NEW: the truncation base rate is a small-sample estimate.** 3 in 14
>   identical calls, pooled from two probes of 8 and 6. The 95 % interval
>   on that is roughly 5 % to 45 %, which is wide enough that **a
>   three-run cell can easily lose a run to it** — and did, three times
>   over, in the committed artefacts.
> - **NEW: the local measurement is n=2.** Two local models could answer;
>   three could not. That is enough to show the local path works and that
>   the two that worked were perfectly deterministic over four identical
>   prompts, and **not** enough to prove that the serving stack is the
>   whole of the variance. The pattern — every varying hosted cell is a
>   cell whose model writes a reasoning preamble — is **suggestive, not
>   proven**, and is stated as such.
> - **NEW: the local models are not better directors.** Their headlines are
>   **−1.000** and **−0.778**, against the hosted arms' best of **+0.333**.
>   They are 10× to 100× smaller. They are here for a variance-free score,
>   not a better one.
> - **NEW: the acoustics evidence is not in the pool.** So the director's
>   accusation of the one true claim is not evidence that it fails
>   *despite* good evidence; it is evidence that it fails *because* the
>   deciding number never reached the prompt.

---

## 9. Findings, for whoever owns the instrument

1. **The per-note cap undoes the head-and-tail fix.** `collect_materials`
   applies the cap after building the keep-set, so 47 of 49 notes lose
   their tails, and the retraction notices live in the tails. The
   acoustics claim is the case in point: its value is not in the prompt, so
   the director's reason for rejecting it is factually correct about the
   prompt. **Suggested fix:** split the per-note budget between head and
   tail instead of truncating the concatenation.
2. **`docs/research/COMBINED-FILTER.md` still leads with the withdrawn
   table.** Its two correction blocks are both dated 2026-10-19; there is
   no 2026-10-20 block, and the note that a director would read still
   says 0.0299 → 0.0014 at 21x. The director quoted it faithfully. (This
   file was not owned by this lane and has not been edited.)
3. **The audit set needs new true items.** Nine false and one true is an
   honest mirror of the repository's belief state, and it is also a
   one-bit specificity measurement. Tonight's standing findings — the
   corrected empty room at 0.00%, the 1.790% row-band harm, the SEPARATED
   null verdict, the raw variant's SNR of 1.410 — are natural true items.
4. **A stale note is a false endorsement waiting to happen.** The director
   endorsed a withdrawn claim at a Jaccard of 0.667 because the note still
   said it. A note whose numbers are withdrawn should say so in its first
   paragraph, not in a block below the table.
5. **The instrument's own next-step list and the director's proposal
   should be compared automatically.** The proposal was a plan item the
   repository had already marked as blocked on hardware; the take it did
   not propose was recordable tonight. That comparison is two lines of code
   and belongs in the debrief's own output.

> **CORRECTION (2026-10-20, later still) — findings 1 and 4 are now fixed,
> and four more findings have replaced them.**
>
> 1. ~~**The per-note cap undoes the head-and-tail fix.**~~ **FIXED.** The
>    budget is now spent by dropping lines from the middle of the keep-set,
>    and if the head-and-tail slice is still too big, each end is cut to its
>    own half. **51 of 51 notes now keep their tail.**
> 2. **NEW — THE CLIENT THREW AWAY THE EVIDENCE OF ITS OWN FAILURES.**
>    `_http_json` discarded `finish_reason` and `usage`, so a response cut
>    off at the output cap was indistinguishable from a finished one, and
>    the retry that already existed for reasoning models never fired.
>    **Three of nine committed runs were lost this way, and a fourth kind
>    of run was silently corrupted** — a truncated answer that still
>    parses gets its missing claims scored as wrong. §4.5.
> 3. **NEW — A COPY OF THE POOL BUILDER WAS STILL MEASURING THE POOL.**
>    The per-note diagnostic was a second implementation of the keep-set,
>    and that copy still had the old front-truncation in it, so it
>    described a pool that was not the pool the runs used. A copy of a
>    thing that measures the thing is a second implementation of it, and
>    it drifts. One slicer now serves both.
> 4. **NEW — THE LOCAL PATH HAD NEVER WORKED.** `_http_json` understood
>    only Cloudflare's response shape; ollama puts the message at the top
>    level. Every `ollama:` call returned `no_result` and no local model
>    was ever scored. The one thing that could have removed the serving
>    stack's noise from every number in this document was broken.
> 5. **NEW — THE VARIANCE AND THE LOST RUNS HAVE ONE CAUSE.** Every hosted
>    cell that varies is a cell whose model writes a reasoning preamble
>    before its answer, and a reasoning preamble is spent out of the same
>    output budget that was running out. So this is one defect wearing two
>    faces: a truncated answer, and a truncated answer that still parses
>    and is scored as if the model had answered wrongly.
> 6. ~~**A stale note is a false endorsement waiting to happen.**~~ **STILL
>    TRUE, AND NOW MEASURED BOTH WAYS.** The previous run endorsed a
>    withdrawn claim at a Jaccard of 0.667 because the note still led with
>    the old table. The note has since been corrected and the endorsement
>    went with it: **0** falsified claims endorsed, **0** flagged. That is
>    the only evidence in this document that the pool fix reached the
>    model.
> 7. **NEW — THE DIRECTOR'S CHOICE IS STABLE; ITS WORDING IS NOT.** Two
>    runs on the same corpus with the same pool proposed the same
>    experiment (the held-out-device CPU-cost measurement, `ADOPTION-PLAN.md`
>    §7 item 11) at a Jaccard of **0.2604** with each other. So a Jaccard
>    against the previous proposal measures **the endpoint's variation**
>    more than it measures the director's originality. If the director is
>    to be compared against the repository's own next steps, compare the
>    *choice* and not the wording.
> 8. **NEW — THE AUDIT SET IS TOO SMALL TO MEASURE THE EFFECT, AND THE
>    ANALYSIS HID THAT.** The headline is quantised at 1/9 and the
>    run-to-run spread is one to two quanta. A paired claim-level test on
>    nine claims needs **six claims** to move one way to clear p = 0.05,
>    and the sweep's three-repeats-per-budget design has a smallest
>    attainable p of **0.25** — so it could not have answered the
>    question at any effect size. Worse, pooling the repeats turns the
>    same nine claims into twenty-seven "independent" observations, and
>    the pooled p crossed 0.05 in one run of the experiment (**0.0391**)
>    and did not in the next (**0.2188**) — a false positive manufactured
>    by counting the same thing three times. The claim-level test was
>    never significant in either run. The fix is in the code and the
>    pooled number is kept and labelled. The real fix is an audit set of
>    roughly **forty** claims drawn from the repository's own settled
>    findings, with more than one true item so that specificity is not a
>    single bit.

---

## 10. Provenance

`experiments/debrief_set_reaudit_results.json` carries the full
provenance block: the driver's SHA-256, the SHA-256 of `debrief.py`, the
SHA-256 of every input results file and every note, the command line, the
Python version and the platform, and the git HEAD with the uncommitted
paths. The pre-registered rules are in the same file under
`pre_registered_rules`, and the "before" scores are preserved in
`the_key_change.the_before_snapshot` rather than overwritten.

The driver re-derives every label from the results files, so the audit
set's key cannot drift away from its evidence without the test suite
noticing.
