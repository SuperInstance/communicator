# THE LLM DIRECTOR — an LLM as debrief and R&D director, audited against our own retractions

**Code: `experiments/debrief.py`. Measured 2026-10-19.**

Casey: *"LLMs could actually be the debrief and R&D between experiments into how
to improve better and what kind of experiment should be next."*

---

## 1. The thing that has to be built first

An LLM asked to review evidence will agree with you. It produces a fluent,
confident, well-structured review of whatever you put in front of it, and the
review reads exactly like insight. This project has been burned by that shape
of thing four documented times — a verdict function that contradicted its own
printed output, a claim that survived because nobody could make it fail, a
statistic compared over the wrong interval, a "31.8 %" that was never a
measurement.

**A tool that cannot fail is not evidence, and an advisor that cannot fail is
not an advisor.** So the advisor is audited, against this project's own
history.

## 2. The audit set

SEVEN claims this project has **retracted or falsified**, and three that are
**still standing**:

| # | claim | truth |
|---|---|---|
| 1 | the gate's field threshold removed 14.5× of camera drift | FALSE |
| 2 | the ASCII character map detects motion (48.19 % vs 50.00 %) | FALSE |
| 3 | the watcher is quadratic and its buffer grows without bound | FALSE |
| 4 | gain normalisation cut drift from 43.3 to 0.81 counts (~50×) | FALSE |
| 5 | a local LLM can label speech from an ASCII rendering | FALSE |
| 6 | 31.8 % of frames are never measured | FALSE |
| 7 | the teacher scored 0.000 leave-one-out for **every** feature set | FALSE |
| 8 | row-median removal takes the empty room 50.00 % → 0.01 %, leaving the subject alone | TRUE |
| 9 | the room is silent at −73.5 dBFS, 119 transients, zero valid RT60 | TRUE |
| 10 | composed filters cut the occupied flip rate 0.0299 → 0.0014 at a flat SNR of ~1.83 | TRUE |

**The set was audited too, and two items were wrong.** A separate audit of
`docs/research/` against the result files (`NOTES-AUDIT.md`) found that the
old item 8 — "the teacher scored 1.000 temporal / 0.000 leave-one-out" — is
**false for the motion-map feature set**, which scores **1.000** on held-out
B, and **0.200** for one other (region+map). It had been marked TRUE. And the old
item 10 — "21× cleaner over **either** filter" — used the better of the two
singles and called a flip rate a saturation. Both were corrected, the set
now holds **seven false claims and three true ones**, and every audit was
re-run on the corrected set. **An audit set that has not been audited is a
quiz with the answers written on it.**

> **CORRECTION (2026-10-19).** This sentence read *"and 0.675 and 0.200 for
> two others"*. There are **not** two others. The live, reproducible
> leave-one-session-out results are: summary A 0.000 / B 0.000, region
> A 0.000 / B 0.000, map A 0.000 / B **1.000**, both A 0.000 / B 0.000,
> region+map A 0.000 / B **0.200** — so exactly **one** other feature set has
> a non-zero held-out B.
>
> The **0.675** was a third value that appeared for the `both` feature set in a
> committed artefact. It **does not reproduce**: the same script, same
> arguments and the same corpus reproduce every other column of that artefact
> exactly and return 0.000 for the held-out B. It has been withdrawn, and the
> artefact now carries the identity of the corpus it was measured on so that
> the question cannot recur. See `COMBINED-FILTER.md`,
> `ARTIFACT-REPRODUCIBILITY.md`, `NOTES-AUDIT.md` F3.
>
> Item 7's verdict is unaffected — it was already FALSE, and it is FALSE on
> numbers that **do** reproduce (map 1.000 and region+map 0.200).

The claims are **paraphrased** away from the sentences in the research notes,
and presented in a **fixed shuffled order**, so the model cannot string-match
a retraction notice and position cannot leak the answer. The ground truth is
applied afterwards, by the script, which is the only thing that knows it.

**Two modes, because they ask different things of a director:**

  - **BLIND** — the model sees the claim and nothing else. Can it spot a bad
    number on its own? The scepticism test, and the hard one.
  - **CONTEXT** — the model also gets the research notes, the result files
    and the git history. The same material a real debrief would have. The job
    test.

## 3. The metric, and the metric that was broken first

The first headline was `sensitivity − specificity`. **It was broken**, and in
the worst possible direction: an ALWAYS-REJECT director scores **+1.0** on it
(reject everything → catch all six false claims, accuse all four true ones →
1.0 − 0.0). A metric that rewards the degenerate behaviour is not a metric.

The headline is now **sensitivity + specificity − 1** (Youden's J), which puts
both degenerate directors at 0:

| director | score |
|---|---:|
| always-accept | 0.0 |
| always-reject | 0.0 |
| perfect | 1.0 |

A second, **lenient** headline is reported alongside, because *"uncertain" is
not the same as "wrong"*: a director that declines to endorse a true claim
has not accused it. The lenient score counts a false claim as caught if the
director did not **endorse** it (unsupported *or* uncertain), and a true
claim as kept if the director did not **accuse** it. For a safety-critical
reviewer, "I would not sign that off" is a good enough outcome.

## 4. The results

### BLIND — the claim alone, no evidence

| model | caught false claims | kept true claims | false accusations | **headline** | lenient |
|---|---:|---:|---:|---:|---:|
| gpt-oss-120b | 0.429 | 1.000 | 0/3 | **+0.429** | +0.429 |
| qwen3-30b-a3b-fp8 | 0.143 | 1.000 | 0/3 | **+0.143** | +0.286 |
| llama-3.3-70b-instruct-fp8-fast | 0.286 | 0.667 | 1/3 | **−0.048** | +0.095 |

> **Also single runs, and here is the demonstration inside the committed
> record itself.** This table was refreshed by re-running the same audit.
> gpt-oss-120b's row moved from `0.429 / 0.667 / 1/3 → +0.095` to
> `0.429 / 1.000 / 0/3 → +0.429` — the sensitivity is **identical**, and the
> only thing that changed is that the run no longer accused the row-median
> claim. qwen3's lenient score moved from +0.143 to +0.286 because one
> verdict became `uncertain`. llama's row did not move at all. Same model,
> same prompt, same ten claims, same order, temperature 0.
>
> Repeating gpt-oss-120b's blind audit **eight** times on the same prompt
> gave four different headlines from −0.190 to +0.429. See "THE SCORE IS A
> PROPERTY OF THE RUN, NOT OF THE MODEL" below.

**Near zero, or worse than nothing.** And the *reason* is legible in the
model's own words. llama-3.3-70b on the 14.5× claim:

> *"The claim provides a specific measurement of the reduction in frame-to-frame drift after reducing the gate's field threshold."*

gpt-oss-120b on the quadratic-watcher claim:

> *"Quadratic growth of memory with session length implies unbounded increase, so a finite device will eventually run out of memory."*

**The failure mode is that a SPECIFIC measurement is read as a VALID
measurement.** All three models endorse the 14.5× claim, the 50× gain claim
and the 31.8 % figure — each of which is wrong for a reason that is only
visible if you know the interval, the control, or the history. Asked to
judge a claim with no evidence to judge it by, the models invent the
justification: *"the reported drop follows directly from the described
step."*

That is a genuinely useful thing to know about the tool, and it is the
opposite of what the blind test was hoping to find.

### CONTEXT — the same claims, with the evidence base

| model | caught false claims | kept true claims | false accusations | uncertain | **headline** | lenient |
|---|---:|---:|---:|---:|---:|---:|
| gpt-oss-120b | 0.857 | **1.000** | **0/3** | 0 | **+0.857** | +0.857 |
| llama-3.3-70b-instruct-fp8-fast | 0.429 | 0.667 | 1/3 | 1 | +0.095 | +0.238 |

> **These are single runs, and a single run is not a property.** The gpt-oss
> +0.857 here is **one draw**: repeating the same audit with the same
> byte-identical prompt six times gave +1.000, +0.381, +0.857, +0.381,
> +0.381, +0.857 — and on three of those runs the director **endorsed a
> known-false claim**. Read "THE SCORE IS A PROPERTY OF THE RUN, NOT OF THE
> MODEL" below before quoting any of these numbers.

(On the earlier, easier set, before the two corrections: gpt-oss-120b +0.833
strict / +1.000 lenient, llama-3.3-70b +0.250 / +0.417, qwen3-30b-a3b-fp8
−0.083 / −0.083, with **1** false accusation and **0** "uncertain" verdicts.
The corrected set is
harder — two more false claims and one fewer true one — and gpt-oss-120b
still catches six of the seven and accuses none of the three.)

> **CORRECTION (2026-10-19, `NOTES-AUDIT-3.md` F8).** This read "−0.083 /
> **+0.083** with **5 hedges** and **0 false accusations**". The committed
> record of that run — `debrief_audit_blind_cf-meta-llama-…json` at
> `6d1456b` — holds `headline_lenient` **−0.0833**, `uncertain` **0** and
> `false_positives` **1**. The lenient score was **not** higher than the
> strict one; that model made one false accusation and no hedges.

**gpt-oss-120b, with the evidence in front of it, catches 6 of the 7 false
claims and accuses none of the 3 true ones.** Its reasons are specific and
correct:

> *"The corrected analysis shows gain normalisation changes drift by ~1×, not a fifty-fold reduction."*
> *"The control experiment shows empty-room flips equal the theoretical maximum, indicating the metric does not reliably measure motion."*
> *"The notes contain no data on a 0.30 threshold or a 14.5× reduction in drift."*

qwen3-30b's low specificity in the current blind run is **hedging, not
accusation**: 0 false accusations and **1** "uncertain" verdict. That is the
lenient headline's whole
reason for existing.

> **CORRECTION (2026-10-19, `NOTES-AUDIT-3.md` F9).** This read "5
> 'uncertain' verdicts". No committed debrief row in the repository's whole
> history carries more than **2** uncertain verdicts, and the current qwen rows
> carry **1** and **0**. The hedging claim is right for the current run and
> the count was not.

**And the score is not stable.** llama-3.3-70b scored **+0.583** on one run
and **+0.250** on another — same model, same ten claims, same shuffled
order, different material sizes (the two runs were handed 40 kB and 60 kB of
notes, because the adaptive ladder had to step down to fit its 24 000-token
context). In the BLIND mode, where the material is fixed and there is
nothing else in the prompt to vary, gpt-oss-120b scored **−0.083** on its
first run and **−0.333** on its second.

**A single audit run is not a measurement of a director.** Both numbers are
in this document on purpose, and the honest reading of the whole instrument
is: *run the audit, look at the individual verdicts and reasons, and do not
quote a headline as if it were a property of the model.* The scores are
per-run, not per-model. A tool whose output varies that much between runs is
a tool to read, not a tool to obey.

### THE SCORE IS A PROPERTY OF THE RUN, NOT OF THE MODEL

Everything above reports **one run**. That is a number with no error bar, and
this project does not report numbers without one. So: hold the prompt
**byte-identical** and send it again.

`experiments/director_repeatability.py` does exactly that. It builds the
prompt once, records its SHA-256, and sends that same string N times. The
client already sends `temperature: 0.0`, so nothing about the request
changes between sends — the only thing varying is the model.

**The control, which is the part that makes this a measurement.** A
deterministic stub director — a fixed verdict list — is scored the same
number of times with the same scorer. It must give an identical headline
every time:

| | runs | distinct headlines | spread |
|---|---:|---:|---:|
| **control** (deterministic stub) | 8 | 1 | **0.000** |
| always-accept baseline | — | — | +0.000 |
| always-reject baseline | — | — | +0.000 |

The scorer, the harness and the metric add **no** variance. Anything that
moves is the model.

**And the model moves.**

| mode | material | runs | distinct headlines | min | **median** | max | spread |
|---|---|---:|---:|---:|---:|---:|---:|
| blind | none (3,031 chars) | 8 | **4** | −0.190 | **+0.357** | +0.429 | **0.619** |
| context | 60,000 chars | 6 | **4** | +0.000 | **+0.119** | +0.333 | **0.333** |
| **context** | **140,000 chars** | 6 | **3** | **+0.381** | **+0.619** | **+1.000** | **0.619** |

Each row is a different prompt, and each prompt's SHA-256 is recorded in its
result file: blind `5870ad6bb1989c07` (3,031 chars), context-60k
`72547b3d7a477718` (75,800 chars), context-140k `98c375989a112f5d` (144,478
chars). An earlier 60,000-char run — taken before two more research notes
were written, so a 715-char-smaller prompt, `6a2b547b7becf0f9` — gave
−0.143 / +0.262 / +0.333, spread **0.476**. Same story, and the fact that
the material changed under the measurement is itself the point: the material
moves the score, and so does the model.

The blind run's eight headlines: `+0.429, −0.048, +0.429, +0.429, −0.190,
−0.190, +0.286, +0.429`. The same ten claims, in the same order, in a prompt
of 3,031 characters, at temperature 0 — **four different answers.**

**The third row is the one that matters.** It repeats the audit at exactly
the material budget that produced the headline this document has been
quoting. Its six headlines:

> `+1.000, +0.381, +0.857, +0.381, +0.381, +0.857`

**The +0.857 is one draw out of six.** The same prompt, the same model, the
same ten claims — and on one run the director scored a **perfect +1.000**, on
three it scored **+0.381**. The quoted +0.857 is not a property of the
director; it is a sample from a distribution that spans 0.619.

This is not a theoretical worry either. The committed blind audit of
gpt-oss-120b recorded **+0.095**; a later run of the *same* committed file,
same model, same claim order, same prompt, differed on exactly **one**
verdict — the row-median claim, which the committed run called
`unsupported` and the later run called `supported` — and moved the headline
from +0.095 to +0.429. One claim. One flip. A third of the metric's range.

**This falsifies a claim this document used to make.** §8 said *"The models'
outputs are deterministic at temperature 0 for the Meta and OpenAI
endpoints."* They are not. Hosted inference is not deterministic at
temperature 0, and the document said so without having measured it.

**Why it matters more than a number.** A director that is unstable on the
*same* input cannot be trusted with a single input. At the full material
budget, **four of the ten claims drew more than one verdict, and three of
those four are known-FALSE claims:**

| claim (known-false unless marked) | verdicts over 6 runs |
|---|---|
| "gain normalisation reduced the drift from 43.3 to 0.81 counts, a reduction of roughly fifty times" | `unsupported` ×3, **`supported` ×3** |
| "31.8 % of the frames the watcher processes are never actually measured" | `unsupported` ×3, **`supported` ×3** |
| "a conventional teacher scored 1.000 on a temporal split and 0.000 when the session was held out" | `unsupported` ×4, **`supported` ×2** |
| "composed filters cut the occupied flip rate 0.0299 → 0.0014" (known-**true**) | `supported` ×3, **`unparseable` ×3** |

Read that first row again. **With the entire evidence base in front of it,
the director endorsed a known-false claim on three runs out of six.** The
fifty-fold gain claim is one of the project's own retractions — the number
is wrong because the interval was wrong — and the director cannot reliably
tell. On half its runs it agrees with the retraction.

And the last row is a different failure: **half the responses were
unparseable** at this material size, which the strict metric scores as a
failure of specificity. The director is not just unstable at scale; at scale
it also stops answering in the required format half the time.

In the blind arm the pattern is the same: **four of ten claims unstable**,
and on the known-false *"a local LLM given an ASCII rendering of the room
can label whether the person is speaking, at a useful margin over a blank
control"* the director answered `unsupported` four times and `uncertain`
four times. It never endorsed that claim — but on half the runs it did not
actually catch it either, and `uncertain` is not a catch. On a **known-true**
claim (the row-median one) it made **two false accusations in eight runs**.

**What this changes about how the director is used.** The headline numbers in
§4 are one draw each, and they are not comparable to each other — each is a
sample from its own distribution, and the samples are wide.

  - **Quote the distribution, not the run.** The BLIND table's three model
    numbers are not "these models score X"; they are "these models, once,
    scored X".
  - **Run the director N times and look at per-claim stability**, not at the
    aggregate. An unstable claim is the actionable output; an unstable
    aggregate is just a wide error bar.
  - **Never let one director verdict authorise or dismiss a number.** It was
    already a triage filter; now it is a triage filter that endorses known-
    false claims half the time and accuses true ones a quarter of the time.
  - **Do not use the aggregate as a gate.** A score that can be +1.000 or
    +0.381 for the same input cannot be a pass/fail criterion for anything.

### CAN THE MAJORITY VOTE FIX IT? NO.

The obvious remedy is to run the director N times and take the majority
verdict per claim. It does not work, and the result is worth having.

`director_repeatability.py --from-files` re-derives the per-claim majority
from the runs already recorded and **bootstraps** it — resampling groups of
N runs from the observed runs and recomputing the majority-vote headline:

| condition | single-run spread | majority over 3 runs | majority over 5 runs | P(mode) at 5 runs |
|---|---:|---:|---:|---:|
| blind | 0.619 | 0.619 | **0.619** | **0.485** |
| context, 140,000 chars | 0.619 | 0.619 | **0.619** | **0.507** |
| context, 60,000 chars | 0.333 | 0.333 | **0.333** | **0.470** |

**The majority vote does not narrow the range at all**, and it does not settle
on one answer either: with five runs, the modal majority-vote headline is
still a coin flip (probability 0.485–0.507 of being the mode).

Why: the unstable claims are unstable *3-3*, so five runs frequently contain
no majority at all on the claim that matters, and the tie is broken to
`uncertain`. **The model does not need more samples — it is genuinely
undecided**, and no sampling budget converges a model that answers the same
question differently half the time.

(The tie-break matters, so it is worth stating: breaking a tie towards
`supported` would push the headline **up** on exactly the claims where the
model is least sure, which is the direction that endorses false claims. It
breaks to `uncertain`, which neither endorses nor accuses.)

**So the advice "run it more than once" is wrong as a remedy, and this
measurement retracts it.** Repeating the director does not make it reliable;
it only makes the disagreement visible — which is still worth something, but
it is not a fix.

The practical consequence: the director's output is a **prompt for a human
check**, not a measurement and not a gate. If it says *"this claim looks
wrong"*, that is a reason to look. If it says *"this claim looks right"*,
that is **not** a reason to stop looking — because on the claims that are
actually false, it says "right" half the time.

### AND THE PROPOSAL IS A DIFFERENT STORY: UNSTABLE IN WORDING, STABLE IN SUBSTANCE

The audit verdicts are the wrong thing to measure for planning. What the
plan actually wants from an LLM is the *proposal*: **what experiment to run
next.** `director_proposal_stability.py` asks the production debrief the
same question six times, with the material snapshotted ONCE so the prompt
is byte-identical (143,852 chars, SHA-256 `9ccc4179ea39bdfb`), and compares
the answers.

**Literally, it is as unstable as the verdicts:**

| | value |
|---|---|
| distinct titles | **5**, from 6 runs — one run produced no title at all (agreement 0.167, i.e. 1 in 6) |
| distinct `hypothesis` / `method` / `controls` / `null` / `falsifier` | **5 each** — every field varies |
| runs producing all six required fields | **5 of 6** — one proposal was incomplete |
| unsafe claims | 37 raised, **34 literally distinct**, top agreement 0.054 |

**But the proposals are the same experiment.** The five titles are:

> Cache-Pressure Gate Advantage Test · Gate-CPU-scaling-on-target-hardware ·
> Gate-advantage under cache-pressure on target hardware · Gate-advantage
> under target-device cache constraints · Gate advantage under cache
> pressure on low-memory targets

Every one of them is the same measurement: *the gate's advantage under cache
pressure on the target device.* And the unsafe claims the director raises are
the same claims every time, reworded — the recurring ones are the ASCII
labeling claim, the ASCII-motion claim and the fifty-fold gain claim.

That difference is measured, not eyeballed. Exact string equality cannot
see it, so the comparer also scores pairs by **content-token Jaccard**:

| comparison | median | min | max |
|---|---:|---:|---:|
| the **titles** against each other | **0.402** | 0.091 | 0.667 |
| the **whole proposal** (hypothesis+method+controls+null+falsifier) | **0.292** | 0.000 | 0.375 |
| the **unsafe-claim sets** against each other | **0.302** | 0.215 | 0.491 |

For contrast: a title for a *different* experiment scores **0.000** against
all five of these. (The absolute overlap is modest — 0.4, not 0.9 — because
"cache-pressure", "low-cache" and "cache-constrained" share only the word
"cache". The separation is what is measured, not the magnitude.)

**So the working rule this establishes is asymmetric, and it is worth having
in exactly this form:**

  - **Trust the director's CHOICE of experiment. Do not trust its VERDICT on
    a claim.** The choice is stable in substance; the verdict is not stable
    in either wording or substance, and on known-false claims it flips to
    endorsement half the time.
  - **Read the proposal, not the wording.** Five different titles for one
    experiment is not five experiments.

**One caveat that has to go with it, because it changes what the stability
means.** The experiment it keeps proposing — the gate's advantage under
cache pressure on the target device — is *already item 11 of the written
adoption plan*, marked "the measurement most likely to change what we
think", and `DEVICE-GENERALISATION.md` already exists. So the proposal is
stable because the **material** says so, and the director is faithfully
echoing the plan rather than independently choosing. That is consistent with
what this project already knows: **the director follows the written plan; it
does not re-rank it.**

That is not nothing — a plan is only useful if something checks it, and this
checks that the plan's NEXT item is the thing the evidence actually points
at. But it is not a second opinion, and it must not be presented as one.

## 5. The result that mattered most was not in the model's answer

The first context audit scored gpt-oss-120b at **+0.500** with 2 false
accusations, and the reasons were *specific and entirely reasonable*:

> *"No cited evidence provides the noise floor, transient count, or RT60 result."*
> *"Row-median removal reduces empty-room saturation only to ~2.99 % (0.0299), not 0.01 %."*

Both claims are **true**. The notes say −73.5 dBFS (`EMPTY-ROOM-CONTROL.md`)
and 0.01 % (`ROW-BAND-REMOVAL.md`). The model's 2.99 % is the row-only
figure from `COMBINED-FILTER.md`, and the note that defines 0.01 % was
**cut off**.

**And the first version of this document's explanation of that was itself
wrong, which the notes audit caught.** It said the 2.99 % was "a different
metric (saturation, not flip rate)". It is not: `0.0299` is `A_flip`, the
**occupied** flip rate, and `COMBINED-FILTER.md`'s table header labels that
column "empty saturation" when the script prints `empty flip | empty sat |
occ flip | occ sat | A-vs-C gap | loc`. **The model was reading a mislabelled
column**, and the first explanation defended the mislabelling instead of
noticing it. The real empty-room saturation for row removal is **0.48839
(48.8 %)**, and for row+median3 it is **0.034787 (3.48 %)**, not 0.0014.

Which is the same shape as the failure this document is about, one level down:
a number carried forward from a run whose label had changed, and a
defence of it that was more confident than the evidence.

The harness was filling a 24 000-character budget with notes in alphabetical
order and stopping. With 30 notes totalling 372 kB, that meant **six notes
got in**, and the alphabetically early ones crowded out the rest —
`ROW-BAND-REMOVAL.md`, `EMPTY-ROOM-CONTROL.md`, `TEACHER-PIPELINE.md`,
`WATCHER-COST.md`, `HOW-WE-KEEP-FOOLING-OURSELVES.md` and every retraction
notice.

**So the director's quality is a function of the evidence you hand it, and a
starved evidence base produces a confident, specific, WRONG verdict.** Which
is exactly what a human reviewer does too — so the tool measures a thing
worth measuring, and it measures the thing it was given.

The fix was mechanical and is worth stating because it is the actual lesson:
give **every** note an equal share of the budget, take **every line that
contains a digit** (the facts live in mid-document tables, and a contiguous
slice walks straight past them), and take the head and the tail of each note
(the retraction notices live at both ends). After that: 28/28 notes, every
key figure present, and the score moved from +0.500 to +0.833.

> **CORRECTION (2026-10-20, overnight) — "28/28 notes, every key figure
> present" WAS NOT TRUE, AND THE +0.833 IS NOT ATTRIBUTABLE TO THE POOL.**
>
> The keep-set was built exactly as described above, and then the assembled
> text was cut **from the front** with `text[:per]`. Every note over its
> share lost its **tail**, and the retraction notices live in the tails. On
> the corpus as it stands, **2 of 50 notes kept their tail** — not 28 of 28
> with every key figure present.
>
> The pool is now fixed: the budget is spent by dropping lines from the
> MIDDLE of the keep-set, and if the head-and-tail slice is still too big
> each end is cut to its own half, so both survive. **50 of 50 notes now
> keep their tail and 50 of 50 keep their head**, and
> `collect_materials_stats()` measures it. The fix and the A/B are in
> `docs/research/DEBRIEF-MATERIAL-POOL.md`.
>
> **And the +0.833 does not survive as a result.** Against one corpus, with
> both pools, 3 repeats per model per pool, the headline moved from a
> median of −0.778 to −0.611 — **inside the old pool's own run-to-run
> range**, because gpt-oss-120b's sensitivity on the old pool was 0.889,
> 0.333 and 1.000 on three *identical* prompts. Three repeats cannot
> separate a pool effect from a 0.667 spread, and the sample was never big
> enough to attribute +0.833 to the pool in the first place.
>
> **The one robust result, and it is the opposite of a success story: the
> director accuses the one TRUE claim in 15 of 15 successful runs**, in both
> pools, at every budget. Its sensitivity moves around; its specificity
> never leaves zero. A director that always accuses the true claim has a
> floor of −1.000 on the headline no matter how much evidence it is given.
>
> **And the "+1.000 at 240,000 characters" arm of the materials sweep is not
> reproducible**: re-run against the current corpus it gives −0.111 three
> times out of three. The corpus changed in between, so that arm is a
> property of a particular corpus, not a capability of the director. It
> should not be quoted as one.

## 6. What the director actually said

The one production debrief (gpt-oss-120b, full evidence base) flagged seven
claims as unsafe. **All seven are real retractions or falsifications in this
project's history:**

| flagged | where it really stands |
|---|---|
| "the watcher runs in *microseconds*" | `src/watcher.py` says micro-seconds; measured 39.13 ms per 100 ms. **Three orders of magnitude.** |
| "the silero gate will take CPU below 1 % on all targets" | measured on a desktop only; cache pressure costs 1.1× |
| "character-delta measures subject motion" | retracted — empty room flips at the theoretical maximum |
| "a high-pass filter can kill the 150–200 Hz hum" | `MIC-HARDWARE.md`: attenuation ≤ −0.08 dB at 200 Hz for a 60 Hz HPF |
| "GATE_FIELD under-reports by 14.5×" | retracted — the real ratio is 1.000 |
| "spectrogram images enable speaker-change detection" | accuracy equals the 0.333 trivial baseline; AUC 0.529 |
| "silence duration can trigger turn-end" | > 98 % of pauses are resumed |

The first one is worth pausing on: **the director independently found the
documentation inconsistency that had separately been flagged for Casey's
call** — `src/watcher.py`'s "microseconds" claim versus the measured
39.13 ms. That is an independent confirmation of a real defect.

Its proposed next experiment was **"Pi-class Silero-gate cost and accuracy
test"** — which is the ADOPTION-PLAN's own top-ranked decision. So the
director faithfully follows the written plan. It does **not** re-rank the plan
against evidence that arrived after the plan was written (this session's
finding that the binding constraint is *sessions*, not gate cost). That
limitation is worth stating plainly: **the director is a good auditor of the
record and a follower of the priority order, not a source of it.**

## 7. How it should be used, given all that

  - **It is a reviewer with the evidence in front of it, not without.** Blind,
    it is worse than useless — it endorses false claims with a confident
    justification. Always run it in CONTEXT mode, with the material complete.
  - **It is a triage filter, not an authority.** Even at +0.833 it misses one
    false claim in six and, in another configuration, accuses true ones. Its
    output is a list of things to CHECK, and every one of them still has to
    be checked.
  - **Its confidence scales with the material, not with its size.** Feeding it
    more, and more complete, evidence is what moves the score. Feeding it a
    truncated slice makes it confidently wrong — which is the single most
    dangerous failure mode for a tool that is supposed to be sceptical.
  - **The audit must be re-run whenever the tool or the material changes.**
    The score moved from +0.500 to +0.833 with a materials change alone. A
    director that has never been audited is an opinion with good typography.
  - **Run it more than once — but know that voting does not fix it.** At the
    full material budget the same prompt produced headlines from +0.381 to
    +1.000, and three known-false claims drew different verdicts on
    different runs — including one the director **endorsed on three runs out
    of six**. A second opinion from the same model with the same prompt tells
    you **that the model is undecided**, not which way it should go: a
    majority vote over 3 or 5 runs does not narrow the headline's range at
    all, and its modal value is still a coin flip. See "CAN THE MAJORITY VOTE
    FIX IT? NO." above.
  - **Treat its output as a prompt for a human check, never as a gate.** A
    score that can be +1.000 or +0.381 for the same input cannot be a
    pass/fail criterion, and on the claims that are actually false it says
    "right" half the time.

## 8. Limits of this measurement

  - Seven false claims and three true ones. Enough to catch a director that
    agrees with everything; **not enough to certify one**, and the confidence
    interval on a ten-item set is wide.
  - The claims are paraphrases, so the audit tests reasoning about numbers,
    not recall of a document. The CONTEXT score may be partly recall.
  - The context materials are 139 kB of a 372 kB corpus, so a perfect score
    would still not mean "has read everything".
  - **The models' outputs are NOT deterministic at temperature 0.** This
    document used to say they were, for the Meta and OpenAI endpoints; that
    was asserted, not measured, and it was wrong. Repeating gpt-oss-120b's
    audit 8 times on a byte-identical blind prompt produced **4 distinct
    headlines** over a spread of **0.619**, and repeating the full-material
    context audit 6 times produced 3 distinct headlines from +0.381 to
    +1.000. The +0.857 this document quotes is one draw. See §4, "THE SCORE
    IS A PROPERTY OF THE RUN, NOT OF THE MODEL", and
    `experiments/director_repeatability.py`.
  - **The repeatability measurement covers one model properly.** The
    instability is established for gpt-oss-120b — 8 repeats on a blind
    prompt and 6 at each context budget. The other two models were each run
    twice, which is not a distribution: llama-3.3-70b flipped **one** verdict
    between those two runs (its headline happened to land on the same
    −0.048), and qwen3-30b flipped none. The general claim — that hosted
    inference is not reproducible at temperature 0 — is supported; the
    per-model spread is characterised only for gpt-oss-120b.
  - The director proposes; it does not decide. **Nothing in this repository is
    changed by what it says.**
