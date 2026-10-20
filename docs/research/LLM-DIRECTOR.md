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

Six claims this project has **retracted or falsified**, and four that are
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
B, and 0.675 and 0.200 for two others. It had been marked TRUE. And the old
item 10 — "21× cleaner over **either** filter" — used the better of the two
singles and called a flip rate a saturation. Both were corrected, the set
now holds **seven false claims and three true ones**, and every audit was
re-run on the corrected set. **An audit set that has not been audited is a
quiz with the answers written on it.**

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
| gpt-oss-120b | 0.429 | 0.667 | 1/3 | **+0.095** | +0.095 |
| qwen3-30b-a3b-fp8 | 0.143 | 1.000 | 0/3 | **+0.143** | +0.143 |
| llama-3.3-70b-instruct-fp8-fast | 0.286 | 0.667 | 1/3 | **−0.048** | +0.095 |

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

(On the earlier, easier set, before the two corrections: gpt-oss-120b +0.833
strict / +1.000 lenient, llama-3.3-70b +0.250 / +0.417, qwen3-30b-a3b-fp8
−0.083 / +0.083 with 5 hedges and 0 false accusations. The corrected set is
harder — two more false claims and one fewer true one — and gpt-oss-120b
still catches six of the seven and accuses none of the three.)

**gpt-oss-120b, with the evidence in front of it, catches 5 of the 6 false
claims and accuses none of the 4 true ones.** Its reasons are specific and
correct:

> *"The corrected analysis shows gain normalisation changes drift by ~1×, not a fifty-fold reduction."*
> *"The control experiment shows empty-room flips equal the theoretical maximum, indicating the metric does not reliably measure motion."*
> *"The notes contain no data on a 0.30 threshold or a 14.5× reduction in drift."*

qwen3-30b's low specificity is **hedging, not accusation**: 0 false
accusations and 5 "uncertain" verdicts. That is the lenient headline's whole
reason for existing.

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

## 8. Limits of this measurement

  - Six false claims and four true ones. Enough to catch a director that
    agrees with everything; **not enough to certify one**, and the confidence
    interval on a six-item set is wide.
  - The claims are paraphrases, so the audit tests reasoning about numbers,
    not recall of a document. The CONTEXT score may be partly recall.
  - The context materials are 139 kB of a 372 kB corpus, so a perfect score
    would still not mean "has read everything".
  - Three models, one audit each. The models' outputs are deterministic at
    temperature 0 for the Meta and OpenAI endpoints, but qwen3's reasoning
    makes it less so.
  - The director proposes; it does not decide. **Nothing in this repository is
    changed by what it says.**
