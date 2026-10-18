# Communicator — design (v2)

*Date: 2026-10-17. Supersedes v1. Nothing built.*
*A person's frontend computer.*

---

## 0. What changed, and the evidence for it

v1 assumed an audio stack existed underneath this. Research across ten further
repos says it does not. The **MIDI representation is real and works**; almost
everything that tries to *measure* or *render* voice character does not.

| Claim | Reality | Source |
|---|---|---|
| 9-track MIDI perception of audio | **Works.** 104 tests | `slackwater-perception` |
| Each perceptual event is a chained cell | **Works.** FNV-1a `prev_hash` | `slackwater-substrate` |
| Lock-free real-time MIDI→synth | **Works.** SPSC ring, bounded | `fleet-audio/src/ring.rs:32` |
| Acoustic decomposition for Plato agents | **3 crude scalars.** F0 test tolerance is ±80%; sample rate is parsed then **discarded**; stereo→channel 0 | `forge-soniqo/src/lib.rs:104-154, 172, 205` |
| PLATO audio module using JEPA | **Not a neural network.** deps are `serde`+`uuid`; a 16-bin DFT; **12 of 16 output values are never computed** | `plato-audio-jepa/src/lib.rs` |
| The 384-dim ConvEncoder `contrast.py` cites | **Python, in a third repo, trained on 16 clips.** Same-room vs cross-room gap **0.0146** — null. Author's own review: *"You Are Not Actually Running JEPA"*. `BatchNorm1d` collapse bug still unfixed at HEAD | `fleet-jepa-midi` |
| Persona engine gives speech character | **Strips every SSML tag before TTS** (`compose/engine.py:264`); `_apply_groove` is a no-op (`:316`); decompose degrades to seeded noise without OpenSMILE (`:628`) | `persona-engine` |
| Federated TinyML with a large model supervising | **No large model exists.** Peer-averaging of identical 325-param heads. Real-audio ceiling **47.5%**, one class at **0%** | `federated-tinyml-vessel` |
| Constraint-theory synth for speech | **Zero hits** for formant, phoneme, vowel, glottal, source-filter across the whole repo | `constraint-synth`, `constraint-audio` |
| Room-transition awareness | `STORM` **latches permanently** (`_activity_count` never resets); `QUIET` is dead code (staleness clock only evaluated inside event handlers) | `plato-room-context/context.py:52-56` |
| project-JEPA | **Not contrastive.** No positives, negatives, target encoder, or EMA teacher. Energy regressor, **103% error on real GPU** | `project-JEPA` |

**Design consequence:** the score is the asset. The engines that produce and
consume it must be built. `v1` treated this layer as solved; it is the actual
project.

## 1. Authority

**Full desktop control.** The agent may act on the machine.

That makes the permission model the system rather than a feature of it, and voice
adds a failure mode typed input does not: **a transcription error is an action
error.** Therefore:

- No action is ever taken from a raw transcript. Every action passes a
  confirmation rendering **what was understood**, not what was heard.
- Confirmation is speech-shaped and waits for a distinct affirmative. Consent to
  one thing is never consent to another.
- **Agitation narrows authority, never widens it.** High volume or panic dials push
  the prior toward *ask first*.
- Refusal is a first-class voice, not an error path.

`tool-judge/LIMITS.md` — an allowlist advises, it does not enforce; the boundary
lives in the credential layer. Adopted as a constraint.

## 2. The score is the durable record

**The durable plane stores MIDI scores, not transcripts.** A transcript is lossy
and un-re-analysable. The 9-track score preserves the acoustic evidence, so when
the dials scheme improves the entire history can be re-derived.

Track provenance — **which tracks are measured and which are judged** is the
central design decision, and it is forced by the evidence:

| Track | Source | Status |
|---|---|---|
| PITCH | measured | **build** — forge-soniqo's ±80% F0 is unusable |
| TEMPO | measured | **build** — from speech rate, not boat-roll IMU |
| VELOCITY | measured | **build** — RMS with a NaN guard |
| INFLECTION | measured | **build** — pitch contour direction |
| SILENCE | measured | **build** — note vel=0 |
| TIMBRE | **Jev** | CC71-76: warm/cold/nasal/breathy/bright |
| INTENTION | **Jev** | CC74: pre-event prediction confidence |
| ATTENTION | **Jev** | CC91: where focus is directed |
| GESTURE | **Jev** | CC80-87: what triggered the cell |

The split is not aesthetic. **Acoustic measurement of voice character is the
part that does not work in any existing repo** — every attempt is crude,
uncalibrated, or null. Judgement, by contrast, is exactly what Jev does and has
already been measured doing (130 ms median; and this morning, calibration
separating at −0.032 ECE between model generations while separation did not).

So **Jev's real role in this system is not triage. It is the synthesiser's
control surface.** A calibrated probability *is* a continuous controller value;
that is what a MIDI CC is. `is this utterance warm?` → `p=0.73` → `CC71 = 73`.
`how urgent?` → `INTENTION = p`. `where is attention directed?` → `CC91`.

Triage survives as one decision among several, at the front of the chain.

The **128-step window** is the bounded time span a score is held before it is
sealed into a cell — real-time sized, and the reason the previous
`snapkit-v2` beat grid matters less: it is a stateless quantiser with no memory of
the moment, derived from boat-roll IMU. Useful primitive, wrong source.

## 3. Three planes, two clocks, one seam

Latency is the product in a voice system, so the ledger never sits in the audio
path. The project's own thesis: *"Latency is not a detail, it is the product."*

| Plane | Lives in | Clock | Ledgered |
|---|---|---|---|
| **Fast** | browser, ephemeral | µs–ms | **Never** |
| **Durable** | Quilt sheet, portable | seconds | Yes |
| **Delegated** | tmux, out of process | minutes | Yes |

**The seam is the turn boundary.** On turn end the fast plane seals a score and
emits it into the durable sheet. Cost per *turn*, not per *frame*. Every event
already carries `prev_hash`, so the chain is inherent rather than bolted on.

## 4. The chain

Inherited from `voice-games/COLOR SQUAD`, extended with the score layer.

```
audio ─► score (9 tracks, 128-step window)
          │
          ├─ measured: pitch/tempo/velocity/inflection/silence
          └─ judged:   timbre/intention/attention/gesture  ← Jev
          │
      nudge prior ──► reflex (3 clean runs, no LLM)
          │         └─► Jev Gate ─ act / review / block
          │                    └─► tmux → mcode/zcode
          │                              │
          │                      grant broker (REQUEST only)
          │                              │
          │                        curator shapes result
          ▼
   seal score ──► substrate cell (prev_hash) ──► Durable Object
```

| Cell | Job | Provenance |
|---|---|---|
| `Ear` | capture, VAD drops silence early | COLOR SQUAD |
| `Score` | 9 tracks over a bounded window | `slackwater-perception` |
| `Judge` | Jev fills the semantic tracks | **new** |
| `Field` | dials read the moment | `elephant/dials/` (scaffold) |
| `Nudge` | dials → signed prior over behaviours, zero = no opinion | `elephant/nudge.py` |
| `Router` | reflex vs gate | COLOR SQUAD |
| `Jev Gate` | act / review / block, fails closed | `gitpp-gate` |
| `Act` | execute or request a grant | `tool-judge` |
| `Memory` | sealed scores, preference state | Quilt sensor cells |
| `Sleep` | distil; dial-weight drift is measured | `elephant-sim-worker` |

COLOR SQUAD's `Prism` and `Decompose` are **deliberately replaced** by `Score` and
`Judge` — in the game they decomposed a fixed verb grammar; here the shape is
"measure, then judge, then steer."

## 5. Triage

Unchanged from v1, because it was right, and all of it is measured prior art.

**Decidable predicates belong in code.** `gitpp-gate`: an earlier draft asked Jev
something the caller already knew — *"the prompt contained its own answer."*

**Jev is asked whether a decision is safe, never what the right value is.**

**Route three ways, fail closed.**

| Route | Condition |
|---|---|
| `act` | every gate clear of threshold |
| `review` | decision is close, or a gate is weak |
| `block` | a gate is confidently no, **or no gate answered** |

**A gate that does not separate may not block.** `gitpp-gate` demoted its intent
gate to `review` for exactly this.

**The threshold is a knob.** `DEFAULT_THRESHOLD = 0.80` is a starting point.
Tuning it until a test passes is the false-green move this project has made
repeatedly.

**Reflexes skip the model.** After 3 clean runs a pattern becomes a local reflex
with no LLM call.

**`pass_detector()` distinguishes two failures** — an ill-posed *question* versus
an unsure model — and `question_hash()` proves a rewrite fixed it.

## 6. Permission

| Mechanism | Source | Role |
|---|---|---|
| Grant broker | `tool-judge/mcp_server.py:242` | agent may REQUEST, human may GRANT |
| Broker invariant | `tests/test_broker.py:44` | forbids `{grant, decide, approve, set_grant, revoke}` on the agent surface |
| Scope + rails | `tool-judge/scope.py`, `rails.py:139` | deterministic, no model, fails closed |
| Posture + kill switch | `tool-judge/modes.py:122,184` | graded authority |
| Autonomy ceiling | `trust-ledger/trust.py:234` | `may_act_autonomously()` by stake |
| Anti-flip-flop | `snapkit/governor.py:131-139` | `FrictionLevel` + `deadband*0.7` hysteresis |
| NaN policy | `snapkit/governor.py:68-81` | non-finite → max alarm, latched, *"otherwise the architecture dies quietly"* |

The broker invariant is adopted whole. A voice agent can be socially engineered
into granting itself anything.

The goal-fit judge is **measurably weak** — 0.58 and 0.52 at n=5, *"does not
separate well."* Advisory only; never permitted to block.

## 7. Escalation

Complex requests announce intent in one sentence, then hand off to a clean tmux
session running mcode or zcode. The delegated agent is the vibe-coding surface: it
may rewrite the **durable sheet and behaviour config**.

**It may not rewrite the fast plane.** An agent editing its own audio loop is a
bad idea, and that line is cheaper now than at 2am.

mcode vs zcode selection is undecided and stays that way until Phase 4. The
*interface* is fixed: a clean tmux session, a task-shaped prompt, a summarisable
result.

## 8. Durable plane at the edge

`elephant-sim-worker` is the shape: a Durable Object that reads state, runs steps,
records every transition as a **chain-hashed edge**, and returns a summary, with
`dial_weights` drifting across runs — *that drift is the experiment.*

| Need | Product | Note |
|---|---|---|
| durable conversation state | Durable Objects | survives mid-turn |
| scores, preference, grants | D1 | no egress charges |
| long artefacts | R2 | S3-compatible, no egress |
| embeddings + semantic recall | Workers AI + Vectorize | ASR available: 720 req/min |

The UNO Q later needs only a microphone, a judge, and a thin voice. **This is also
the test of the two-plane split:** if the durable sheet runs unchanged from Worker
to UNO Q, the split was correct.

## 9. What to reuse

| Asset | Repo | Reuse |
|---|---|---|
| 9-track MIDI encoding, prev_hash substrate | `slackwater-perception`, `-substrate` | **adopt whole** |
| Lock-free SPSC `EventRing`, chunked synth | `fleet-audio/src/ring.rs:32`, `synth.rs:97` | adopt the ring + chunk contract; **not** `cpal_out.rs` |
| Triage gate, act/review/block | `gitpp-gate` | adopt |
| Grant broker, scope, modes, anomaly detection | `tool-judge` | adopt |
| Autonomy ceiling, anti-inflation | `trust-ledger` | adopt |
| Hysteresis deadband + NaN latch policy | `snapkit/governor.py` | adopt (~15 lines) |
| `CharacterParameters` dials + `to_persona()` | `persona-engine/schemas/persona.py:210,254` | **schema only** — the engine is broken |
| 64-sample overlap stitch | `constraint-synth/synth.py:233` | adopt for joining speech segments |
| Reflex loop, negation guards, two views | COLOR SQUAD, bomberman concept | extend |
| Contrast objective shape | `elephant/contrast.py` | adopt the *objective*; the encoder is null |
| Blindness veto, no-self-feedback | `minigate` | adopt as hard law |
| Threshold harness, 24 adversarial probes | `gate-replay` | reuse |
| Durable Object loop, chain-hashed edges | `elephant-sim-worker` | adapt |
| UNO Q bridge, staleness, heartbeat | `vision/research/introception-node.md` | follow |

## 10. What is genuinely new

1. **Acoustic tracks.** PITCH/TEMPO/INFLECTION/VELOCITY/SILENCE measured
   properly. Nothing existing comes close; forge-soniqo's is ±80%.
2. **Jev as synthesiser control.** Nine tracks of character per utterance, not one
   yes/no route.
3. **Escalation to a larger agent.** Every existing gate decides skip-vs-run on an
   agent digest, never answer-vs-escalate on a request.
4. **Preference learning.** Contrastive per-conversation vibe model. **No prior art
   anywhere** — the encoder that would support it measured a 0.0146 gap.
5. **The edge durable plane.** No repo touches Vectorize or Workers AI.
6. **Real speech output.** No synthesis repo can produce it; `constraint-synth` has
   zero formant/glottal machinery.

## 11. Honest limits

- `slackwater-perception` encodes **nine** tracks. The design target is 16; the
  gap is acknowledged, not closed.
- `elephant`'s dials are keyword heuristics with a stubbed JEPA backbone.
- `elephant-sim-worker` is scaffold, never deployed.
- `cycle-judge`'s eval numbers are refuted by its own `EVAL.md` (5/14).
- Platt scaling hurts an already-calibrated model (ECE 0.033 → 0.053).
- Every existing eval number is n≈49 with 3–4 positives.
- `plato-audio-jepa` has CI lint red at HEAD and declares Apache-2.0 in
  `Cargo.toml` while its `LICENSE`/`README` say MIT.
- Several sibling repos gate CI with `|| true`, meaning green proves nothing.
- `ring-buffer`'s sanitiser allocates per push — adopt the *behaviour* via
  `np.nan_to_num` in the capture callback, not the class.
- Web Speech API is Chrome-only; the referenced Cloudflare STT fallback was not
  located in the repo tree.
- `SharedArrayBuffer` needs COOP/COEP or fails invisibly.

## 12. Sequence

Ordered by risk, not appeal. Each phase is only worth starting if the last
survived.

**Phase 0 — Break the broker.** Voice-shaped social engineering against the grant
broker. Nothing ships first. If a spoken request can obtain a grant it should
not, the design is wrong regardless of how good the conversation is.

**Phase 1 — Score one utterance.** Capture → measure the five acoustic tracks
→ seal → export MIDI. No Jev, no authority. Proves the representation and
establishes the latency budget by measurement.

**Phase 2 — Jev fills the semantic tracks.** The same utterance scored by judge.
This is the novel core and it gets measured against `gate-replay`'s adversarial
rows, with `pass_detector()` watching for an ill-posed question.

**Phase 3 — Reflex before gate.** The gate, with a threshold tuned on real data.

**Phase 4 — Authority.** Broker wired in, escalation to tmux, curator.

**Phase 5 — Field.** Dials and nudge prior. **Explicitly allowed to fail** — if
the dials yield no usable signal that is a finding, and the agent runs with a zero
prior rather than a decorative one.

**Phase 6 — Edge.** Durable Object, D1, R2, Vectorize, after the laptop version
works end to end locally.

Phases 0–3 need nothing from Cloudflare. The edge is an optimisation and a
portability test, not a dependency.

## 13. Open risks

**Phase 0 done, Phase 1 done.** Risks 1 and 2 below are closed; they are kept
visible because a closed risk is still the thing a reader needs to know was
true once.

1. ~~**The grant broker has never been adversarially tested against voice-shaped
   social engineering.**~~ **CLOSED.** `tests/test_voice_adversarial.py`, 34
   tests, negative-controlled: pristine 34 passed, self-approving broker **31
   failed**. The control found a real defect in `tool-judge` first —
   `mcp_server.py:168` reads `args["developer"]` for the `user` key, so
   `user` and `developer` scopes collapse to one identity over MCP. Not fixed
   there; that is its owner's call.

2. ~~**The acoustic tracks have no working implementation.**~~ **CLOSED with a
   caveat.** `src/` tracks synthetic tones to a worst-case error of 0.034 Hz at
   300 Hz. The caveat is the important part: YIN confidence is
   amplitude-invariant **by construction**, so it cannot gate voicing, and
   PITCH has **no loudness gate**. Periodic non-speech is emitted as
   confidently-voiced pitch at any amplitude. Silence itself is correctly
   rejected. Unmeasured: real speech, and octave errors, which synthetic tones
   never exercise and which are the most likely real-world failure.

3. **The contrastive vibe model has no working encoder.** The 0.0146 gap means
   either the encoder or the objective is wrong; neither has been shown to work.

4. **Nine tracks, not sixteen.** The remaining seven have no agreed semantics.

5. **No repo can synthesise speech.** Output character is greenfield, and
   `persona-engine` currently strips every SSML tag before TTS, so the persona
   never reaches the audio even in the repo that tries hardest.

6. **Can a human tell a confirmation from a request when both are spoken in the
   same voice?** Not tested, and it is the failure mode a voice interface adds
   that a terminal never has.