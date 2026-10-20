# The audit set's answer key changed under the quiz

**Date:** 2026-10-20
**Code:** `experiments/debrief_set_reaudit.py` · **Instrument:** `experiments/debrief.py`
**Results:** `experiments/debrief_set_reaudit_results.json`
**Tests:** `tests/test_debrief_reaudit.py` (18 tests)
**Supersedes:** the labels, and every score computed against them, in
`experiments/debrief_audit_*.json` and `experiments/debrief_cf-*.json` as
they stood on 2026-10-19. The old scores are kept, not overwritten, in
`the_key_change.the_before_snapshot` of the results file.

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

| artefact | mode | model | sens | spec | headline |
|---|---|---|---|---|---|
| `debrief_audit_blind_cf-meta-…_cf-openai-…_cf-qwen-….json` | blind | llama-3.3-70b | 0.222 | 0.000 | **-0.778** |
| | blind | gpt-oss-120b | 0.222 | 1.000 | +0.222 |
| | blind | qwen3-30b | 0.111 | 1.000 | +0.111 |
| `debrief_audit_blind_cf-openai-…_cf-meta-…_cf-qwen-….json` | blind | gpt-oss-120b | 0.222 | 1.000 | +0.222 |
| | blind | llama-3.3-70b | 0.222 | 0.000 | **-0.778** |
| | blind | qwen3-30b | 0.444 | 1.000 | +0.444 |
| `debrief_audit_context_cf-openai-gpt-oss-120b.json` | context | gpt-oss-120b | 0.889 | 1.000 | **+0.889** |
| `debrief_audit_context_cf-openai-…_cf-meta-….json` | context | gpt-oss-120b | 0.778 | 0.000 | **-0.222** |
| | context | llama-3.3-70b | 0.556 | 0.000 | **-0.444** |

Look at rows 7 and 8: **the same model, the same material budget, minutes
apart, with opposite specificity** — one kept the single true claim, the
other accused it. One run is a sample, not a score.

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
