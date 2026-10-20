# NIGHT REPORT — 2026-10-20

**Repository:** `SuperInstance/communicator`
**Author:** Casey Digennaro, with an AI agent (MiniMax Code)
**Window:** 2026-10-19 night → 2026-10-20 08:50 local
**Status at hand-over:** 24 commits of the night's work pushed (138 in the
repository), `master` in sync with `origin/master`, full suite green
(**1002 passed, 5 skipped**), artifact gate green
(**REPRODUCIBLE=37, NOT_REPRODUCIBLE=0**).

This document is the wrap-up of one night of work on **Communicator**, the
local-first voice/peripheral system. It is written to be read by whoever
picks the work up next — including by an agent that has none of the
context. It records **what was achieved**, **what was struggled with**, and
**what is still open**, with numbers rather than adjectives.

Read the house-style documents first if you are new here:
`docs/SCIENCE.md`, `docs/research/HOW-WE-KEEP-FOOLING-OURSELVES.md`, and
`docs/research/NV12-STRIDE.md`.

---

## 0. THE ONE-PARAGRAPH VERSION

Every video experiment in this repository read the camera captures at the
wrong byte stride, so **80 % of every "frame" this repo had ever measured
was a tear between two different moments**. Fixing that exposed a real,
mouth-localised speaking signal that had been invisible, and forced the
correction of roughly a dozen committed results. Later in the same night,
three defects were found in the *measurement instrument itself* — a client
that discarded the evidence of its own failures, a local-model path that
had never worked, and a diagnostic that was measuring a different object
than the one it claimed to measure — and fixing them led to the night's
most uncomfortable result: **the instrument's resolution is the same size
as its noise**, so most of the comparisons the lane had been making could
not have resolved what they claimed.

---

## 1. THE ACHIEVEMENTS

### 1.1 The frame stride — the root cause of the night

**Every video reader in the repo hard-coded `W*H*3//2` = 23,040 bytes per
frame.** The captures on disk are **19,200 bytes per frame**: a 160×96 luma
plane plus *interleaved* U,V chroma at 80×24 — half the vertical
resolution of standard NV12's chroma.

Read at 23,040, every "frame" is rows 0..71 of one real moment spliced to
rows 72..95 of the next, and only every sixth frame survives intact.

**Three independent lines of evidence were required**, because any one
alone would have been arguable:

| line | evidence |
|---|---|
| the file must agree with the audio | at 19,200 the captures are 900 frames = **30.00 s** against A/B/C's 29.98 s of audio, and empty1's 4,500 frames = **150.00 s** against 149.91 s. At 23,040 they are 25 s and 125 s. Every capture, every time. |
| the frames must stop being garbage | mean consecutive \|diff\|: A **43.560 → 0.545**, B 37.740 → 0.433, C 42.935 → 0.367, empty1 39.295 → **0.005** |
| a human eye | the correctly-read frames render as a sharp, stable photograph; a filmstrip of the mouth region across the take shows the mouth **visibly opening and closing** |

**The reader is now `src/nv12.py`**, which *detects* the layout from the
content and reports the margin over the runner-up (80× on A, 117× on C,
**7,446×** on the empty room). **An ambiguous layout is an error, not a
guess.** Three layouts are described: `standard_nv12` (23,040),
`y_uv_halfres` (19,200 — the captures), `luma_only` (15,360 — described
because it is realisable, though **no script in the repo has ever written
it**; the earlier attribution of it to `corpus_capture_win.py` was false
and was corrected).

**What the fix bought.** A speaking signal, at session level, localised to
the mouth:

| quantity | value |
|---|---|
| mouth box (rows 46–58, cols 62–86) | **+0.733σ**, z **+10.9** |
| cheek control (rows 46–58, cols 92–116) | +0.477σ |
| excess | +0.256σ (ratio 1.54×) |
| empty room | **0.015 %** of A's signal |

**Everything it cost.** The "LED flicker banding" (a perfect 50.00 % in an
empty room), the teacher's two USABLE verdicts, the pose features, the
motion maps, the composed-filter claim, and a dozen other numbers were
artefacts of reading tears. Corrected: `row_band_removal`'s empty-room
figure **50.0000 % → 0.0000 %**; the teacher's margins **+0.208 / +0.312
→ +0.000 / +0.062**; the pose feature **1.71× → 1.249×**; the cloud
multimodal lane **+0.455 (5/11) → +0.667 (8/12)** with FPR 0.000.

The old corpus is **left on disk** so every withdrawn number can be
re-run. The corrected one is `/mnt/c/proart_capture/corpus_v2/` (900
frames, 30.0 s, `y_uv_halfres`, byte-identical across re-seeds).

### 1.2 Four research lanes, two of which falsified their own hypotheses

| lane | question | verdict |
|---|---|---|
| **4 — teacher on 4 representations** | is any representation better than raw? | **No representation beats the teacher's own controls.** All four NOT USABLE. The rule's largest gap is +0.0833 = 4.0 of 48 test windows. |
| **5 — cross-validated mouth detector** | is the mouth signal a mouth detector? | **It survives head motion and is NOT mouth-specific.** Cheek 0.7443 beats mouth 0.7086 (paired z −1.70); the blank control scores 0.6667, above the detector's 0.6655. |
| **6 — dither characterisation** | is the character map measuring dither? | **NOT DITHER.** Dither fraction **0.0 %** (empty room 0.0000 %, frozen null 0.0000 %, real capture 8.6315 %). 76.2 % of the change is local scene change. Recommendation reverses back to the **mask**. |
| **7 — re-audit the LLM director** | is the director's audit set right? | **The answer key changed under the quiz.** The set holds **9 FALSE / 1 TRUE**. The director's proposal **echoes** the notes and **cannot fail**. |

### 1.3 Lane 8 — does retaining the image beat describing it?

The user pointed at <https://huggingface.co/papers/2610.26860> —
*OpenVLHarness: Real World Visual Agentic Tool-Use with Multimodal Memory*
(Yang et al., UCLA/Apple). Its Appendix H states that *"text descriptions of
images are a weak substitute for retaining the images themselves"*.

This repository is the case that claim is about, and had never tested it:
the project has spent its life converting camera frames into **text/ASCII**
representations and asking a model to judge them, and had never handed a
model the frame itself.

**Method.** Qwen2.5-VL-3B-Instruct locally on the RTX 4050 (4.62 GB peak
of 6.14 GB), one forward pass per frame, the score being the logit of the
first token of `SPEAKING` minus the first token of `NOT SPEAKING` — one
token each, so no length bias and no dependence on JSON formatting. 120
frames per main condition, evenly spaced, no selected pair byte-identical.
**The prompt is identical in every arm except its content.**

| arm | AUC (speaking vs silent) | blank vs silent | margin over blank |
|---|---|---|---|
| **image** | 0.7784 (z +7.43) | **0.0034** | **+0.7750** |
| **text** (ASCII) | **0.9562** (z +12.26) | **0.9970** | −0.0408 |
| text_global (fixed 0–255 ramp) | 0.5265 | — | −0.4733 |
| image_scrambled | **0.4062** | — | −0.1378 |
| text_scrambled | 0.6438 | — | −0.0878 |

**Verdict: `no_difference_at_the_preregistered_threshold`.** D = +0.8159
clears the ±0.05 margin, but the paired permutation z is **−0.41** against
a required +2.0.

**Two component facts point in opposite directions, and the pre-registered
statistic cannot adjudicate them** (the permutation preserves the very
blank-control term that dominates D; the decomposition is written out in
`docs/research/VLM-IMAGE-RETENTION.md` §3.4):

* The **character rendering separated better** than the picture
  (0.9562 vs 0.7784).
* The **picture was the only one of the two that still knew the difference
  between a room and a man**: the text arm ranks the *empty room* above
  the silent person (0.9970); the image arm ranks it below (0.0034).

**The scramble control says something the headline does not.** Destroying
spatial arrangement drops the **image** arm below chance (0.7784 →
0.4062) but the **text** arm only to 0.6438. At 160×96 the character
rendering retains **more** usable spatial information than the upscaled
picture does.

**A correction to the lane's own brief:** capture **B is a speaking take**
(`speak=True, pose=turned` in `corpus_v2/B.json`), not a silent one. C is
therefore the pose-matched negative class and B is reported as a secondary
condition. The lane's brief said otherwise; the captures' own metadata is
right, and the doc says so rather than quietly diverging.

### 1.4 The night's three instrument defects, and what fixing them cost

**Defect 1 — the client was discarding the evidence of its own failures.**
`debrief.py::_http_json` rebuilt every response as `{"text": text}` and
**discarded `finish_reason` and `usage`**, so a response cut off at the
output cap was indistinguishable on the wire from a finished one.
`cloud_llm` already had a retry for reasoning models — it repeats the call
with 4× the output budget when `content` comes back **empty** — and the
truncated case never reached it, because a truncated answer is not empty,
it is *cut*.

Three of the nine committed audit runs were exactly this. And it cost more
than three lost runs: **a truncated answer can still parse** (the JSON
closes before the cut and the model keeps writing prose), and the scorer
had no way to know the answer was short, so **the claims the model never
got to answer were scored as wrong**.

**Defect 2 — the local path had never worked.** `debrief.py` has always
accepted `ollama:` and `local:` models. **Every such call returned
`no_result` and no local model was ever scored**, because the response
parser understood only Cloudflare's shape
(`{"result": {"choices": [...]}}`) while ollama puts the message at the
**top level**. The answer was in the body the whole time, behind a shape
check. The one thing that could have removed the hosted endpoint's
non-determinism from every number in the lane was broken.

**Defect 3 — a copy of the thing that measures the thing.** The pool's
per-note diagnostic was a hand-rolled second copy of the keep-set builder,
and that copy still carried the **old front-truncation** in it. So it was
measuring a pool that was **not** the pool the runs were made against: it
reported **3 of 49** notes keeping their tail while the pool actually used
kept **51 of 51**.

**The measurement that settled defect 1.** The obvious hypothesis — too
much *material* for the smaller models' context windows — is **false**,
and the control is structural rather than statistical: **in blind mode the
material does not enter the prompt at all.**

| material budget | prompt characters | prompt tokens |
|---|---|---|
| 140,000 | 3,031 | **701** |
| 60,000 | 3,031 | **701** |
| 40,000 | 3,031 | **701** |

Same prompt, three budgets, varying outcome — and the failure happened at
the **smallest** budget and not the largest. The budget that ran out was
the **output** budget. *A fit ladder that treated "unparseable" as "does
not fit" would have walked down past budgets that worked.*

Truncation rate and recovery, pooled over two probes: **3 in 14**
identical calls truncated (21.4 %), **3 of 3 recovered** by asking for 4×
the output budget. Every truncation came back with `completion_tokens`
**exactly 3000** — the cap — while complete answers used 1,853 to 2,518.

**After the fix, all nine committed audit runs parse.** The three lost runs
are back.

### 1.5 The uncomfortable result: the metric's resolution is the size of its noise

This is the finding that should change what the next person does, so it is
stated plainly.

* The audit set has **nine false claims and one true one**, so the
  headline `sens + spec − 1` moves in steps of **1/9 = 0.1111**. It is
  **quantised**.
* The observed run-to-run spread is **0.111 and 0.222** — **one and two
  quanta**.
* With the transport clean, all 21 re-measured runs carry **10 of 10**
  verdicts, and the headline moves **only when a claim flips to
  "uncertain"** (blind gpt-oss: uncertain counts 0, 0, 1, 1; context
  qwen3: 5, 3, 4).

**So the instrument's noise is the same size as its resolution.** And the
effect the lane's sweep claimed — **+0.1111** — is *exactly one quantum*.
**The experiment cannot distinguish "more evidence helps" from "one claim
happened to be answered differently this time."**

**The new method that replaces it** (`experiments/debrief_paired_analysis.py`)
compares the arms **claim by claim**, which is the unit the metric can
resolve:

| step | claim-level McNemar (n = 9, the pre-registered unit) | pooled over repeats (n = 27, anti-conservative) |
|---|---|---|
| 20,000 → 60,000 | run A: p = 0.6250 · run B: p = 1.0000 | run A: 0.5078 · run B: 1.0000 |
| 60,000 → 140,000 | run A: p = 0.2500 · run B: p = 0.5000 | run A: **0.0391 "significant"** · run B: 0.2188 |
| 140,000 → 240,000 | run A: p = 1.0000 · run B: p = 1.0000 — **all nine claims agree, both runs** | 0.5078 · 1.0000 |

**Never significant at the claim level, at any step, in either run.** The
pooled statistic crosses 0.05 in one run and not in the next — counting the
same nine claims three times adds **variance**, not evidence.

**And the arithmetic that should have been stated before the first
comparison:** a two-sided exact sign test on **three** paired samples has a
smallest attainable p of **2/8 = 0.25**, so the sweep's design **could not
have produced a significant result at any effect size whatsoever**. Its
verdict label, *"EFFECT RESOLVED"*, was not available to that design.

**Controls, which come out the other way:** the null arm (a budget
compared with itself) gives **0 discordant claims, p = 1.0000**; the
reversed pairing reports the **opposite direction**.

**What the instrument needs instead is not a bigger effect but a bigger
audit set:** with nine false claims the minimum detectable shift is **six
of them**, so a one- or two-claim shift at p < 0.05 needs roughly **forty**
false claims, and **more than one true item**, because specificity is
currently a single bit moving in steps of 1.000.

### 1.6 A variance-free instrument, and the confound in it

`experiments/debrief_local_director.py` measures the director on **local**
models, where the endpoint has no freedom to change its mind:

| local model | parsed | headline, repeated | spread |
|---|---|---|---|
| `qwen2.5-coder:7b` | 4 / 4 | −1.000 ×4 | **0.0000** |
| `gemma2:2b` | 4 / 4 | −0.778 ×4 | **0.0000** |
| `llama3.2:1b` | 1 / 4 | +0.111 | — (3 runs failed to parse) |
| `qwen3:0.6b` | 0 / 4 | — | wrong output shape |
| `tev1` | 0 / 4 | — | `http400`, "does not support chat" |

**Its pre-registered rule fired** ("local spread 0.000 and hosted spread
not 0.000 ⇒ the variance is the serving stack") and **that verdict is
wrong**, for two reasons recorded in the results file:

1. **The rule confounds locality with reasoning.** Every local model that
   could answer is a *non-reasoning* model; every hosted cell that varies
   is a cell whose model writes a *reasoning preamble*. This machine has
   no local reasoning model to break the tie.
2. **A better-controlled measurement says otherwise:** after the transport
   fix, the hosted spread **did not move** — median spread 0.0555 before
   and 0.0555 after.

The corrected conclusion is in the file: **the variance is not explained by
locality.** It is one or two claims out of nine flipping to "uncertain".

### 1.7 The house-style record

`docs/research/HOW-WE-KEEP-FOOLING-OURSELVES.md` now carries **eight**
retractions. The first five were claims about the *world*; **#6, #7 and #8
are claims about the measurement**, and each was invisible to the check
that was supposed to catch it. The standing practice gained three lines:

> An instrument records the reason for its own failures, not only their
> occurrence.
> A copy of a thing that measures the thing is a second implementation of
> it, and it drifts.
> An effect smaller than the metric's quantum — or smaller than the
> metric's own noise — is not an effect.

---

## 2. THE STRUGGLES

This section is the one that matters for whoever picks the work up. Every
item below is a thing that went wrong, cost time, or was wrong and had to
be undone.

### 2.1 The stride was found by noticing a number that looked too clean

The stride defect was not found by a failing test. It was found because
**50.00 %** — the theoretical maximum for a binary flip rate — appeared in
an *empty room*. A perfect number in a physical measurement is a smell.
It then took three independent lines of evidence to settle, because any
one alone was arguable, and because every *internal* check in the repo had
been computed with the same wrong constant and had therefore **confirmed**
the wrong answer.

### 2.2 Four lanes; two falsified their own hypotheses

That is a healthy rate and it is the point of the house style, but it does
mean **two of the night's four lanes produced a negative result.** Lane 4
found that no representation beats the teacher's own controls; Lane 6
found the dither fraction is 0.0 % and that the "dither" recommendation
had to be **reversed**. Neither is a failure of the work; both are the
work.

### 2.3 A real defect was fixed, and the fix did not help

The material-pool defect was real: `debrief.py` cut the assembled evidence
**from the front**, so every over-long note lost its tail, and the tails
are where the retraction notices live. **2 of 50 notes kept their tail;
after the fix, 50 of 50 do.**

**And the fix did not make the director better.** Specificity stayed at
**0.000 in 15 of 15** successful runs, before and after. The defect was
kept and documented anyway, because the pool *should not* silently drop
the retractions — but the honest conclusion is that **the director's
failure is not a starvation problem**, and the fix's benefit is to the
record, not to the score.

### 2.4 A headline number was withdrawn as not reproducible

The "+1.000 at 240,000 characters" arm of the materials sweep did not
reproduce: three fixed-pool runs at 240,000 give **−0.111**, and a paired
old-pool/new-pool run at 240,000 gave identical per-claim verdicts. The
effect is **+0.1111**, not +1.444, and the verdict label was changed from
"much better, monotonically" to **"EFFECT RESOLVED"** with the size of
the effect stated beside it rather than inside it.

### 2.5 The director's proposal changed between two identical runs

The same model, on the same corpus, with the same evidence pool, produced
a **different** proposal: a Jaccard of **0.2604** against the previous
committed proposal, against the repository's own ≈0.29 "same experiment"
threshold. So the earlier headline — *"Jaccard 1.000, literally the same
experiment"* — was a **single-sample number**, and the two samples fall on
**opposite sides** of the threshold.

The surviving finding is more precise and more useful: **the director's
*choice* is stable** (it keeps proposing the plan item already marked as
blocked on hardware, and it keeps **not** proposing Condition D) and its
*wording* is not. A Jaccard against the previous proposal measures **the
endpoint's variation** more than it measures the director's originality.

### 2.6 My own hypothesis about the truncation was refuted by my own probe

The natural reading of "three runs came back unparseable" is "too much
material". I designed and ran a two-point probe to check it — same model,
same mode, same key, only the budget changed — and **it refuted the
hypothesis**: qwen3 parsed at 140,000 and at 60,000 characters and failed
at 40,000. I then found the structural reason (in blind mode the material
never enters the prompt) and went looking for the real cause.

Had I not probed, I would have "fixed" the fit ladder to shrink the
material on truncation — and that fix would have walked **down past
budgets that worked**.

### 2.7 My second hypothesis was also refuted, by a controlled re-measurement

Having found the truncation, the natural next hypothesis was that the
truncation explained **the variance** too. I tested it the right way: fix
the transport, re-measure the same repeats on the same prompts with the
same key, compare. **The hosted spread did not move** — median 0.0555
before, 0.0555 after. One cell halved and one doubled.

A controlled change that does not move the quantity is evidence that the
quantity is not what the hypothesis was about. The variance is the model's
own indecision, and §1.5 found it.

### 2.8 A pre-registered rule fired and produced a verdict I had to retract

`debrief_local_director.py`'s rule fired and the results file says
**"THE VARIANCE IS THE SERVING STACK"**. That verdict was then
contradicted by the confound (§1.6) and by the controlled re-measurement
(§2.7). The rule, its firing, and the retraction are **all three kept
visible** in the results file, in a block that says who added the
annotation and when. **The measurements were not touched; only the
analysis was annotated.**

### 2.9 The paired-analysis script violated its own pre-registered rule

The first implementation of the claim-level test **pooled the repeats**
and ran McNemar on 27 pairs — the same nine claims measured three times,
treated as 27 independent observations. That is anti-conservative, and it
produced a **p = 0.0391** on the one step that looked significant, where
the pre-registered unit gives **p = 0.2500**.

The script's own rule said *"exact McNemar on the nine false claims"*, so
the implementation **did not do what the rule said**. The fix is in the
code; the pooled figure is kept and labelled; and a test now reproduces
the pooling and asserts that it disagrees with the pre-registered
analysis. **The pooled p then turned out to be 0.0391 in one run and
0.2188 in the next** — so it was not even reliably a false positive; it
was noise.

### 2.10 A correction block in the repo's own record was false

The previous top correction block said the acoustics figure **"73.5"**
*"now reaches the prompt — it did not before the fix — and the director
accuses the claim anyway"*, and used that to argue the failure is **not** a
starvation problem.

**The only `73.5` in the pool is `AMI EN2002a 73.5 % (191 of 260 gaps)`** —
an unrelated number from a different experiment. `"-73.5 dBFS"`,
`"73.496"`, `"119 transients"` and `"RT60"` are **all absent**. The
acoustics evidence does **not** reach the prompt, so the director accuses
the one true claim **because the deciding number never arrived**, and the
sentence — and the conclusion drawn from it — is **withdrawn**.

That is a correction to a correction, in a document whose whole subject is
corrections. It is also the single most useful thing the night produced
about the pool: **the pool's composition is a moving target, and a claim's
evidence can be present in one run and absent in the next.**

### 2.11 The artifact gate was canceled once and its output was lost

The gate re-runs ~37 scripts and takes half an hour. It was started with
buffered output into a file; the background job was canceled; the buffer
went with it and the log was zero bytes. **Lesson: a long verification run
whose output is buffered and then killed leaves you with nothing.** It was
re-run detached (`setsid`) and **unbuffered** (`python -u`) so that a
partial run is still readable.

### 2.11a The gate also reported two crashes that were not code defects

A later gate run reported `row_band_removal_results.json` as
`IndexError: list index out of range` and
`teacher_raw_representation_results.json` as `EOFError: No data left in
file` — both of which had been **REPRODUCIBLE** in an earlier gate run.

**Re-run directly, both succeed and their outputs are byte-identical to
what is committed** (`git status` clean afterwards): the empty room still
flips 0.00 %, the occupied 0.62 %, the composed filter still reads "NO
REPRESENTATION BEATS THE TEACHER'S OWN CONTROLS".

The most likely cause is the **`/mnt/c` Windows mount returning short
reads after the machine had been suspended and resumed** — the gate run in
question spanned a five-hour gap in which the box was asleep. Both
failures are read errors on the capture mount, not arithmetic errors.

**The lesson is the same one as §2.11 in the other direction: a gate
failure is a claim, and the claim has to be checked before it is acted
on.** Re-running the script directly is the cheapest way to tell an
environmental failure from a real one, and it is cheaper than a
revert.

> ## CORRECTION (2026-10-20, last thing before the reboot) — the two crashes were confirmed environmental, and the full gate is clean
>
> The paragraph above attributes the two crashes to the `/mnt/c` mount
> returning short reads after the machine was suspended. That was a
> **hypothesis**, supported by the fact that both scripts succeed in
> isolation. It is now **measured**.
>
> A fourth gate run — same command, same `--source worktree`, detached
> with `setsid` and unbuffered with `python -u`, log at
> `/tmp/gate4.log`, kept beside the earlier runs at
> `.scratch/fitladder/gate4.log` — was run against the worktree at
> `96090ec` and finished with:
>
> ```
> REPRODUCIBLE=37  NOT_REPRODUCIBLE=0  NO_SCRIPT=4  NOT_RE_RUN=37  LEGACY_NO_SCRIPT=1
> ```
>
> `row_band_removal_results.json` and `teacher_raw_representation_results.json`
> are **both REPRODUCIBLE** in this run. They are the two artefacts the
> previous run reported as `IndexError` and `EOFError`.
>
> **What this establishes:** the crashes were not code defects, and no
> code was changed to make them go away. The same worktree, the same
> scripts and the same commands produce 37/0 when the machine has not
> been suspended underneath them. The `NOT_RE_RUN` count rose from 35
> to 37 because the two scripts added late in the night —
> `debrief_truncation_probe.py` and `debrief_paired_analysis.py` — have
> artefact files now, and both are skipped for a stated reason (they need
> the Cloudflare API over the network), not for a failure.
>
> **What this does not establish:** it does not identify *which* read
> went short, and it does not prove the suspend/resume mechanism. What
> it does establish is the thing that actually matters for the
> repository: **the committed artefacts still reproduce from the
> committed code, 37 of 37, with zero not-reproducible.** The
> worktree was clean before and after the run.

### 2.11b `zcode` could not be run, and the reason is an account entitlement

The instruction to have `zcode` do the same work could not be carried
out: `zcode` fails headlessly with

```
[project-access-token] { family: 'zai', stage: 'resolve', code: 'project_token_login_failed' }
RuntimeHeadersRefreshError: project_token_login_failed
  cause: CodingPlanApiKeyError
  providerId: account:zai-individual-coding-plan   modelId: GLM-5.3
  baseURL: https://api.z.ai/api/anthropic
```

**It is not a network problem** — `api.z.ai` and `open.bigmodel.cn` were
both reachable (HTTP 301 and 200) from this box. The stored credentials
in `~/.zcode/v2/credentials.json` are present, but the **coding-plan
project token** will not mint, and the request is marked
`retryable: false` and is not retried.

**The fix is one command and it needs a browser:**

```bash
zcode login zai          # or: zcode login bigmodel
```

Headless, it prints a fallback URL and then fails with `Browser open
failed: spawn xdg-open EACCES`. **That is physical authorisation for the
account owner, and it was deliberately not worked around** — the
entitlement is the account's to grant.

**What was done instead:** the complete brief for the next session was
written and committed as **`docs/research/NEXT-SESSION-BRIEF.md`**, with
the exact command to run `zcode` on it once it is authorised. The brief
carries the house rules, the state being inherited, the highest-value
open item, the open items that need a human, the corrected reader, and
the commit-and-push discipline.

### 2.12 Lane 8's remaining gap

The lane's artefact stores per-condition **summaries**, not the raw
per-frame scores, so the **0.178 AUC difference** between the image and
text arms has **no paired test of its own** — the lane reports each arm's
AUC against its own permutation null and states the limitation. Closing
it needs a re-run that stores the scores. **Left open deliberately**,
because a re-run would invalidate the lane's committed numbers and require
a cascade through its document.

### 2.13 Things that were never fixable tonight

* **The captures cannot give frame-level audio↔video labels.** The audio
  and the video are **independent takes**. No lag (±20 s) and no time base
  (10–60 Hz) recovers frame-level alignment. The session-level signal is
  real and is enough for the peripheral system's first job; it is not
  enough for frame labels.
* **The teacher's blocker is data, not modelling.** Two labelled sessions
  per class cannot tell "speaking" from "this recording". The blank
  control scores **1.000 on every fold**.
* **Clock drift between two recorders** is unmeasurable from one clap.
* **Cloudflare's vision endpoint is not deterministic at temperature 0**
  (hosted spread 0.619 vs local 0.000); gpt-oss's sensitivity ranged
  0.333–1.000 on identical prompts.

---

## 3. WHAT IS STILL OPEN, AND WHAT NEEDS A HUMAN

### 3.1 Needs Casey, and it is worth doing

1. **RECORD THE CORPUS — Condition D is the highest-value take.** The
   capture tool is fixed and synced to the Windows side
   (`/mnt/c/proart_capture/corpus_capture_win.py`). Condition D — *turned
   AND silent* — is now the highest-value missing take for **two
   independent reasons**: it makes every pose fold identifiable, and it
   separates "the head moved" from "the mouth opened". Command:
   `C:\Python314\python.exe C:\proart_capture\corpus_capture_win.py --plan --out C:\proart_capture\corpus`
   (~16 min; A1–A3/B1–B3 poem 50 s, C1–C3/D1–D3 silent 30 s, F1–F3 60 s,
   FREE 5 min. **Glasses off for B and D.**)
2. **RECORD THE TWO-MICROPHONE DIALOGUE** (adoption-plan item 10). Now
   safe: the intake fixes have landed.
3. **`src/vad.py:41` and `tests/test_vad.py`'s 39.13 ms citations** —
   deliberately untouched, awaiting Casey's standing decision.

### 3.2 Needs an agent, and can be done unattended

4. **BUILD A ~40-CLAIM AUDIT SET** whose labels are **derived** from the
   results files rather than asserted, with **more than one true item**
   so specificity is not a single bit. This is the direct consequence of
   §1.5 and the single highest-value piece of remaining work: **until the
   instrument can resolve a one-claim difference, every comparison of the
   director is uninterpretable.**
5. **Store the raw per-frame scores in Lane 8's artefact** and add the
   paired test of the 0.178 AUC difference.
6. **`src/energy.py:290`'s leading-silence defect** remains unpatched;
   blast radius in `docs/research/SILENCE-LEADING-ARTEFACT.md`.
7. **The remaining known defects are recorded, not fixed**, in
   `known_defects_not_fixed`: `ascii_motion_threshold.py`'s frame-pair
   loop still starts at `f = 0`, counting one spurious last-vs-first pair;
   `ascii_motion_threshold.py` and `ascii_motion_nulls.py` still disagree
   on the dither gradient between `small.nv12` (52.6×) and `A.nv12`
   (flat 0.99×).
8. **Cloudflare account limits**: llama-3.3-70b and qwen3-30b cannot
   accept 240,000 characters (HTTP 413); `llava-1.5-7b-hf` accepts no
   image shape on this account; `llama-4-scout-17b-vision-instruct` does
   not exist on it.

### 3.3 The one-sentence version of where this leaves the project

> **The camera pipeline now has a verified speaking signal at session
> level, and the LLM director has been shown not to be the thing that
> should be trusted to plan the next step — but the instrument used to
> judge the director is too coarse to tell a one-claim difference from
> noise, and the record needs forty claims before it can.**

---

## 4. HOW TO REPRODUCE ANY OF IT

```bash
cd /home/proart/projects/communicator

# the whole suite
/home/proart/projects/glasses/.venv/bin/python -m pytest tests/ -q

# the artifact gate (half an hour; use -u, it is long)
python -u experiments/artifact_repro.py --source worktree

# the night's three new measurements
set -a && . ~/.config/cloudflare/env && set +a
python experiments/debrief_truncation_probe.py --model qwen3 --mode blind --trials 8
python experiments/debrief_local_director.py --mode blind --repeats 4   # needs ollama
python experiments/debrief_paired_analysis.py --repeats 3

# GPU work lives in a separate venv, outside the repository
/home/proart/projects/gpu-venv/bin/python    # torch 2.6.0+cu124, RTX 4050, mediapipe 0.10.21
```

**The corrected reader** — use this, do not re-derive it:

```python
from src import nv12
luma, layout = nv12.read_luma(path, width=160, height=96)   # (n, 96, 160) uint8
layout.as_dict()
# {"layout": "y_uv_halfres", "stride_bytes": 19200, "n_frames": 900,
#  "mean_consecutive_frame_diff": 0.544891, "margin_over_runner_up": 79.932,
#  "chroma": "interleaved U,V, 80x24 samples"}
```

Strides: `standard_nv12` 23,040 · `y_uv_halfres` 19,200 (the captures) ·
`luma_only` 15,360 (no script writes it). The alias `y_u_v_planar` →
`y_uv_halfres` is deprecated.

---

## 5. WHERE THE RECORDS LIVE

| document | what it holds |
|---|---|
| `docs/research/NV12-STRIDE.md` | the stride defect, in full |
| `docs/research/DEBRIEF-REAUDIT.md` | the director re-audit; **eight** corrections, three of them about the instrument |
| `docs/research/DEBRIEF-MATERIAL-POOL.md` | the evidence-pool defect and its honest limits |
| `docs/research/VLM-IMAGE-RETENTION.md` | Lane 8 — image vs description, a null |
| `docs/research/MOUTH-DETECTOR.md` | the mouth signal, and why it is not a mouth detector |
| `docs/research/ASCII-DITHER-CHARACTERISATION.md` | the dither characterisation |
| `docs/research/TEACHER-RAW-REPRESENTATION.md` | Lane 4 — no representation beats the controls |
| `docs/research/HOW-WE-KEEP-FOOLING-OURSELVES.md` | the eight retractions and the standing practice |
| `docs/research/LOCAL-VLM-DETERMINISM.md` | hosted vs local variance |
| `PROGRESS.md` | the running log, in date order |
| `docs/research/ADOPTION-PLAN.md` | the plan this work serves |

**Commits on `master`, in order** (each message carries its own detail):

```
2b59268  the camera captures were being read at the wrong frame stride
b59d2a8  every camera experiment now reads the captures at the layout they have
5c7a854  the teacher's two USABLE verdicts were false positives
deea7fb  three lanes, two of which falsified their own hypotheses
bc166f2  the hosted model's instability is the serving stack, not the task
8f987ea  the progress log for the night the stride was found
7d07560  the capture tool was baking a withdrawn filter into every take
907f448  the capture instructions said three things that were withdrawn overnight
d274ead  the representation was never the problem, and this file said it was
c5336e3  the mouth signal survives head motion, and is still not a mouth detector
234d129  the teacher was measured on a representation the repo had rejected
a5862f0  the braille representation is not measuring dither, and it is not the tone channel either
873b29a  the LLM director's answer key changed under the quiz, and it echoes the notes
4b74128  the progress log for the four lanes and the auditor's own audit
6014534  the director was being starved, and the fix for it did not help
106ba1c  the progress log for the material-pool defect and the auditor's own audit
5fcc5f8  the auditor's own evidence pool was dropping the retractions
5b75899  the instrument was discarding the evidence of its own failures
ac2d839  the audit is re-run on the instrument that now knows why it failed
0e65e57  two new measurements of the instrument itself, and the gate learns them
cbeb1a6  the record, reconciled: eight retractions, three about the instrument
88ac89c  does retaining the image beat describing it? On this repo's captures, no.
5c32a44  the comparison the instrument's resolution can support, and the trap it fell into
745f791  the provenance section names the three new measurements
```

**Key numeric anchors** (corrected corpus, for a quick sanity check):

* stride margins: A 79.9×, B 87.1×, C 117.1×, empty1 7,446×
* mean \|diff\|: A 0.544891, B 0.433207, C 0.366688, empty1 0.005277
* mouth +0.733σ (z +10.9), cheek +0.477σ, excess +0.256σ; empty room 0.015 % of A
* teacher margins: summary +0.000, region +0.062, map +0.062; blank control 1.000 on every fold
* cloud multimodal: +0.667 (8 of 12), FPR 0.000
* dither fraction 0.0 %; 76.2 % local scene change
* braille pairing: 0 / 24,894 / 42,680 mismatches (correct / identity / column-major)
* director's audit set: 9 FALSE / 1 TRUE; context specificity 0.000 in 15 of 15
* truncation: 3 of 14 identical calls; 3 of 3 recovered by 4× the output budget
* local director spread: 0.0000 on the two models that could answer
* the metric's quantum: 1/9 = 0.1111; the observed spread: 0.111 and 0.222
