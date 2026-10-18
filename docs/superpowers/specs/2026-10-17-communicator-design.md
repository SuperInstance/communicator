# Communicator — design

*Date: 2026-10-17. Status: design for approval, nothing built.*

A person's frontend computer. You speak, it understands quickly, answers simple
things immediately, escalates the rest to a real agent, and matches the shape of
the conversation you're actually having.

---

## 1. Authority, and what it obliges

**Full desktop control.** The agent may act on the machine. This is the point and
it is not being softened in this document.

That choice has one consequence worth stating plainly, because it governs every
other section: with full authority, the permission model is not a feature of the
system, it *is* the system. A voice agent additionally has a failure mode a typed
agent does not — **a transcription error is an action error**. So:

- No action is ever taken directly from a raw transcript. Every action passes
  through a confirmation that renders **what was understood**, not what was heard.
- The confirm step is speech-shaped, not a prompt. It reads back the resolved
  intent and the exact command, and waits for a distinct affirmative. "Yes" to one
  thing is never consent to another.
- **Agitation narrows authority, never widens it.** High volume, panic, or a
  fast pace dial pushes the nudge prior toward *ask first*. A user having a bad
  moment is the moment least likely to catch a mishearing.
- Refusal is a first-class outcome with its own voice, not an error path.

`tool-judge/LIMITS.md` is blunt about its own reach: an allowlist advises, it does
not enforce — the real boundary lives in the credential layer. That is adopted as a
design constraint, not a caveat.

## 2. Three planes, two clocks, one seam

Latency *is* the product in a voice system. The project's own quilt-vs-tangermeme
thesis says so directly: *"Latency is not a detail, it is the product… a ledgered,
witnessable patch operation costs a git write plus protocol ceremony."* So the
ledger never sits in the audio path, and the boundary is drawn on that principle.

| Plane | Lives in | Clock | Ledgered? |
|---|---|---|---|
| **Fast** | browser, ephemeral | microseconds–ms | **Never** |
| **Durable** | Quilt sheet, portable | seconds | Yes, witnessed |
| **Delegated** | tmux, out of process | minutes | Yes, via the agent's own log |

**The seam is the turn boundary.** When the user stops speaking, the fast plane
emits one turn summary into the durable sheet. That is the only crossing. Cost is
paid per *turn*, not per *frame*.

## 3. The cell chain

Inherited from `voice-games/COLOR SQUAD`, which already runs this in the browser
with 15/15 headless logic tests.

```
audio ─► dials(field) ─► nudge prior ─┬─► reflex          (3 clean runs → no LLM)
                                      └─► Jev Gate ── act / review / block
                                               │
                                    answer in-line│
                                               └─► tmux → mcode/zcode
                                                        │
                                              grant broker (REQUEST only)
                                                        │
                                                  curator shapes result
                                                        │
                              turn edge ──► Durable Object (chain-hashed)
```

| Cell | Job | Provenance |
|---|---|---|
| `Ear` | capture audio, VAD drops silence before anything sees it | COLOR SQUAD |
| `Field` | dials read the moment — pace, volume, verbosity, coherence, agitation | **new** (replaces `Prism`) |
| `Nudge` | dials → signed prior over behaviours; zero means no opinion | **new** (replaces `Decompose`) |
| `Router` | reflex vs gate | COLOR SQUAD |
| `Jev Gate` | triage into act / review / block, fails closed | `gitpp-gate` |
| `Act` | execute, or request a grant | tool-judge broker |
| `Memory` | turn summaries, preference state | Quilt sensor cells |
| `Sleep` | distil; dial weights drift and the drift is measured | `elephant-sim-worker` |

COLOR SQUAD's actual chain is `Ear → Router → Jev Gate → Prism → Decompose →
Act → Memory → Sleep`. **`Prism` and `Decompose` are deliberately replaced, not
dropped by omission.** In the game they decomposed a fixed verb grammar into unit
orders; here the shape is "read the moment, then steer by prior," which is what
`Field` and `Nudge` do. Their names and roles are preserved in the new cells so the
diff against the game stays legible — this is a generalisation of the chain, not a
rewrite of it.

## 4. The field: dials and the nudge prior

`elephant`'s central claim is *"a room is a field, not a stream"* — the unit of
perception is the ambient state, not the ordered feed. Adopted directly.

A **dial** is one reading on one dimension. A `DialBank` reads the moment
simultaneously. The **nudge prior** maps dials to behaviours with signs and
weights; each behaviour gets a nudge in `[-1, 1]`, and **zero means no opinion**.

For a communicator the dials are: pacing, volume, verbosity tolerance,
interruption tolerance, topic coherence, agitation. The behaviours they steer:
*speak shorter*, *don't escalate*, *ask first*, *offer options*.

**Contrast is the training signal**, per `elephant/contrast.py`: positives from the
same conversation, negatives from other conversations, and no centroid is ever an
anchor. The interesting signal is *how style shifts between contexts*, never an
absolute verbosity score. Two guards are carried over:

- **Anti-collapse** — a hinge penalty if within-conversation spread falls below a
  fraction of a frozen baseline, measured before training and frozen for the run.
- **Condition-level fitting** — fit at the conversation level with message-level
  bootstrap. The deadband lesson: signal lives at condition level, not message grain.

**Standing constraint: no self-feedback.** The preference model must never consume
its own prior verdicts as features. A model that reads its own past opinions learns
to agree with itself — the same defect as `clean_pushes` depending on how tidily
the backlog was worked. Inherited from `minigate/train.py:59`.

Dial implementations are adapted from `elephant/dials/` (mood, panic, presence,
volume, earnestness, cynicism, joke-landing) and the prior mapping from
`elephant/nudge.py`. The dial set is replaced wholesale for voice — a room's
vibe is not a conversation's vibe — but the mechanism is the mechanism.

## 5. Triage

Three rules, all inherited from measured work rather than invented here.

**Decidable predicates belong in code.** From `gitpp-gate`: an earlier draft asked
Jev something the caller already knew — *"the prompt contained its own answer."*
A parent check that is a string comparison must not be a model question. Only
genuinely underdetermined questions reach the model.

**Jev is asked whether a decision is safe, never what the right value is.** The
code knows the answer; the model supplies calibrated confidence that acting on it
now is appropriate.

**Route three ways, fail closed.**

| Route | Condition |
|---|---|
| `act` | every gate clear of threshold |
| `review` | decision is close, or a gate is weak |
| `block` | a gate is confidently no, **or no gate answered** |

**A gate that does not separate may not block.** `gitpp-gate` measured this: the
intent gate separated on the wrong scale, so it was demoted to `review` and only
the gate that actually separates kept blocking authority. Any new gate must
demonstrate separation before it is allowed to block.

**The threshold is a knob, not a constant.** `DEFAULT_THRESHOLD = 0.80` is a
starting point to be tuned against real data. Tuning it until a test's assertions
pass is the false-green move this project has already made repeatedly.

**Reflexes skip the model entirely.** After 3 clean runs a verb pattern becomes a
local reflex with no LLM call. Counters track LLM calls against local reflexes.

**The pass detector distinguishes two failure modes.** `decision-ledger`'s
`pass_detector()` plus `question_hash()` tells us whether a triage prompt is
ill-posed rather than the model being unsure — the instrument that says *"rewrite
the question"*, and a hash that proves the rewrite fixed it. Nothing else in the
existing repos measures this.

## 6. Permission

| Mechanism | Source | Role |
|---|---|---|
| Grant broker | `tool-judge/mcp_server.py:242` | agent may REQUEST, human may GRANT |
| Scope + rails | `tool-judge/scope.py`, `rails.py:139` | deterministic, no model, fails closed |
| Posture + kill switch | `tool-judge/modes.py:122,184` | graded authority, one lever |
| Autonomy ceiling | `trust-ledger/trust.py:234` | `may_act_autonomously()` by stake |

**The broker invariant is mechanically tested** in `tool-judge/tests/test_broker.py`:
no agent-facing tool may write a grant, and the suite fails if one ever appears. A
voice agent can be socially engineered into granting itself anything, so this
invariant is adopted whole and is the single most important thing not to reinvent.

Grant storage is **not** tamper-evident (`tool-judge/LIMITS.md`), so grants live in
the credential layer, not in a file the agent can write.

The goal-fit judge is **measurably weak** — two in-scope commands scored 0.58 and
0.52 at n=5, *"the judge as configured does not separate well."* It is advisory
only and is not permitted to block.

## 7. Escalation

Complex requests announce intent in one short sentence, then hand off to a **clean
tmux session running mcode or zcode**, shaped to the task. The delegated agent is
also the vibe-coding surface: it may rewrite the **durable sheet and the behaviour
config**.

**It may not rewrite the fast plane.** An agent editing its own audio loop is a
bad idea, and drawing that line now is cheaper than discovering it at 2am. This is
the one place where approach B gives up Quilt's uniformity on purpose.

Results return to the curator, which shapes them into the conversational form
rather than reading out raw agent output. The two-view pattern from
`bomberman-commander/CONCEPT.md` carries over: the conversation, plus the live
substrate beside it, so the operator can see the machine's state without reading
logs.

## 8. Durable plane at the edge

`elephant-sim-worker` is the shape: a **Cloudflare Durable Object** whose loop reads
state, runs steps, records every transition as a **chain-hashed edge**, and returns
a summary. Nights persist mid-flight. `dial_weights` drift across runs — *that
drift is the experiment.*

Cloudflare's role, chosen against measured limits rather than vibes:

| Need | Product | Note |
|---|---|---|
| durable conversation state | Durable Objects | stateful, survives mid-turn |
| preference + grant records | D1 | relational, no egress charges |
| long transcripts, artefacts | R2 | S3-compatible, no egress fees |
| embeddings + semantic recall | Workers AI + Vectorize | ASR also available: 720 req/min |

The UNO Q later needs only a microphone, a triage model, and a thin voice. Its state
lives at the edge and is reachable from anywhere with a connection — a terminal for
the ecosystem rather than a box that owns the data.

**This is also the test of the two-plane split.** If the durable sheet runs unchanged
from Worker to UNO Q, the split was correct. If it needs laptop-specific code, it
wasn't.

## 9. What is reused, and from where

| Asset | Repo | Reuse |
|---|---|---|
| Cell chain, reflex loop, Web Speech capture | `voice-games/color-squad` | extend |
| Triage gate, act/review/block, fail-closed | `gitpp-gate` | adopt whole |
| Grant broker, scope, modes, anomaly detection | `tool-judge` | adopt whole |
| Autonomy ceiling, anti-inflation, decay | `trust-ledger` | adopt whole |
| Dials, nudge prior, contrast objective | `elephant` | adapt |
| Durable Object loop, chain-hashed edges | `elephant-sim-worker` | adapt |
| Reflex gate, blindness veto, no-self-feedback law | `minigate` | adopt |
| Triage-threshold harness, adversarial probes | `gate-replay` | reuse harness |
| Pass detector, question hash | `decision-ledger` | adopt |
| Local inference, three-tier permission stub | `zero-poc` | reuse patterns |
| Two-view UI, vibe-code confirmation | `bomberman-commander` | adapt |
| UNO Q bridge, staleness, heartbeat | `vision/research/introception-node.md` | follow |

## 10. What is genuinely new

Everything else is assembly. These four do not exist anywhere in the surveyed work:

1. **Verbs → tasks.** COLOR SQUAD's grammar is game commands. Arbitrary computer
   tasks need decomposition, and that is a real design problem.
2. **Escalation to a larger agent.** Every existing gate decides *skip vs run* on
   an agent digest, never *answer vs escalate* on a user request.
3. **Conversational preference learning.** The `elephant` nudge is a *prior over
   attention*; it has never been wired to a user's spoken corrections over time.
4. **The durable plane at the edge.** No repo touches Vectorize or Workers AI.

## 11. Honest limits

Carried forward rather than discovered later:

- `elephant`'s dials are **keyword heuristics** and its JEPA backbone is a stub —
  its own README says the aspiration is not yet an implementation. The field layer
  is scaffolding, and perception quality is unproven.
- `elephant-sim-worker` is **scaffold, never deployed**.
- `cycle-judge`'s eval numbers are **refuted by its own `EVAL.md`** — 5/14 met
  expectation, InjectionScreen 0/6 attacks annotated. Take its fail-closed /
  fail-open law; leave its accuracy claims.
- Platt scaling **hurts** an already-calibrated model (ECE 0.033 → 0.053).
- All existing eval numbers come from n≈49 idle-dominated cycles with 3–4
  positives. None of them is a real accuracy estimate.
- Quilt's engine has **not been benchmarked at audio push rates**. The source says
  backpressure is *"reserved for future use."* At game cadence this is proven in
  production; at per-token cadence it is unknown.
- Web Speech API is Chrome-only today. A Cloudflare STT fallback is referenced but
  was not located in the repo tree and must be found or rebuilt.
- `SharedArrayBuffer` requires cross-origin isolation (`COOP`/`COEP` headers), or
  it fails in a way that looks like nothing at all.

## 12. Sequence

This spec covers a whole product, so the implementation plan decomposes it. The
order is dictated by risk, not by feature appeal — each phase is only worth
starting if the previous one survived.

**Phase 0 — Break the broker.** Adversarial tests against the grant broker using
voice-shaped social engineering. Nothing else ships first. If a spoken request can
obtain a grant it should not, the design is wrong regardless of how good the
conversation is.

**Phase 1 — Ear and turn.** Capture, VAD, transcription, turn detection, speech
back. No triage, no authority. This is where the latency claim gets measured
rather than asserted — a voice agent that answers in four seconds is broken in a
way a terminal is not.

**Phase 2 — Reflex before gate.** Port the reflex loop and the Jev gate, and
establish the triage threshold against `gate-replay`'s corpus. The pass detector
comes here too, because until the question is known-good, a threshold on it is
meaningless.

**Phase 3 — Field.** Dials and the nudge prior. **Explicitly allowed to fail** —
if the dials do not yield a usable signal, that is a finding, and the agent simply
runs with a zero prior rather than a decorative one.

**Phase 4 — Authority.** Grant broker wired in, escalation to tmux, curator. Only
once phases 0–3 are real does the agent get to touch the machine.

**Phase 5 — Edge.** Durable Object, D1, R2, Vectorize. Portable only after the
laptop version works end to end locally.

Phases 0–2 are laptop-only and need nothing from Cloudflare. That is deliberate:
the edge is an optimisation and a portability test, not a dependency.

### mcode or zcode

Selection is not yet decided and this spec does not pretend otherwise. The
decision belongs with Phase 4, where the escalation contract is written. What the
design fixes now is the *interface* — a clean tmux session with a task-shaped
prompt and a summarisable result — so either can be swapped without touching the
gate.

## 13. Open risks, in priority order

1. **The grant broker has never been adversarially tested against voice-shaped
   social engineering.** This is the one failure that is not survivable, and it is
   testable before anything else ships.
2. **Triage quality is unmeasured.** The threshold has no real corpus behind it.
   `gate-replay` has 55 real + 24 adversarial rows to start from.
3. **The field layer may not work.** If dials do not produce a usable signal, the
   nudge prior is decoration and the agent reads as tone-deaf.
4. **Quilt at audio rates is unknown** and the mitigation is to keep audio out of
   the engine, which the two-plane split already does.