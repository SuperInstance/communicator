# Communicator — design v3

**Status:** proposed, not built. Supersedes v2
(`2026-10-17-communicator-design.md`), which is kept unedited for the record.

---

## 0. What changed, and the measurement that forced it

v2 designed a **voice request interface**: speech in, a triage decision,
speech out, at a turn boundary. That is the right shape for a *first* UI and
the wrong shape for the thing being built.

v3 designs a **peripheral nervous system**. It watches continuously, learns
what normal looks like, and builds shorter paths into the work. It is one
member of a family — a database interface, a log interface, a robotics
interface — and what they share is not the medium but the position: they sit
alongside the primary work surface and accumulate context it does not have.

Three measurements forced the change rather than taste:

1. **Judged tracks cannot run continuously.** Eight questions cost 0.13 s in
   the cloud because they evaluate in parallel, but 2.45 s through llama.cpp
   because it scores them sequentially (1 gate 0.267 s, 4 gates 0.967 s, 4
   scales 1.404 s). A layer that costs a second per call is a per-turn layer,
   full stop. A watching system cannot be built on it.
2. **The measured acoustic tracks are ~10⁵ times cheaper** — pure arithmetic,
   microseconds, no model — and they run on an UNO Q with the network off.
   They are the only layer that can watch.
3. **Shorter utterances are more dangerous, not less.** The baseline INTENTION
   question scored *higher* on `qwop zzz brxt` than on real speech. Ambiguity
   is what drives confident invention, and a shortened path is more ambiguous
   by construction.

So the architecture is not "voice in, answer out". It is:

> a cheap continuous sensor that produces a record, a derivation step that
> learns from the record, and an expensive judged layer summoned only when the
> cheap layer says something changed.

---

## 1. The record is the nine-track score

This survives v2 unchanged, and the reframing strengthens it. Five tracks
measured, four judged, provenance written into the file:

| Track | Source | Encoding |
|---|---|---|
| PITCH | measured | note numbers from YIN F0, `hz=None` when unvoiced |
| TEMPO | measured | set_tempo = the clock; CC120 = measured speech rate |
| VELOCITY | measured | CC7, normalised to the clip's own peak RMS |
| INFLECTION | measured | CC11, direction as a controller value |
| SILENCE | measured | slackwater's velocity-0 notes |
| TIMBRE | judged | CC71 |
| INTENTION | judged | CC74 |
| ATTENTION | judged | CC91 |
| GESTURE | judged | CC80-87, winning class drives its own slot |

Two encoding decisions that cost something and are kept:

**One tick is one millisecond** (ppq 1000, set_tempo 1,000,000 µs/beat), and
the clock event is written *unconditionally*. A reader that finds no tempo
meta event assumes 500,000 µs/beat, which at ppq 1000 makes a tick 0.5 ms —
so "1 tick = 1 ms" would be true of the module and false of the file, and
every time re-derived from the score would be out by a factor of two. The
clock is a property of the encoding; the *measured* tempo is a separate fact
on CC120 and is omitted when unmeasurable.

**TIMBRE is null from text, by three independent instruments.** Jev's
presence gate reads 0.13 on real speech; `calibration.py` reached the same
conclusion from the corpus side; `tev1`, a local model never told about
either, reads 0.02. Its CC in a score is a measurement of its own absence.
Warmth lives in the measured acoustic tracks, or nowhere.

---

## 2. The seam is the EVENT, not the turn

**This is the correction that matters most, and it invalidates v2 §3.**

v2: *"The seam is the turn boundary. On turn end the fast plane seals a score
and emits it into the durable sheet. Cost per turn, not per frame."*

A watching system has no turns. It has **events**: a change in the acoustic
stream worth recording or worth escalating. The cost model is therefore
per-event, and the *event* is what gets sealed into the record.

Concretely, an event is any of:

* a **voiced onset** after ≥150 ms of silence (energy.MIN_SILENCE_S)
* an **utterance boundary** — a run of speech separated by a rest
* a **tempo excursion** outside the session baseline (§4)
* a **gate crossing** — the judged layer's presence gate rising above
  threshold on something the record has not seen
* a **human decision**, which is always an event even when acoustically mute

The last one is load-bearing. A refusal, a correction, or an approval from the
operator is an event with no acoustic signature, and a system that only
listens cannot see the most important events it will ever receive.

**Sealing granularity.** Scores are sealed per utterance, not per event and
not per frame. An utterance is the unit the judged layer can actually
characterise, and the measured tracks are all defined on a 10 ms frame grid
inside one. The event marks *when* attention is spent; the utterance is *what*
gets written down.

---

## 3. Three planes become four steps

v2 had three planes (fast / durable / delegated) with the seam between fast
and durable. That sketch has no room for learning, so v3 adds the missing
step and names where it sits:

```
  audio ─► [1] WATCHER        no model, microseconds, always on
              │  event stream
              ▼
  record ──► [2] SCORE         the nine tracks, sealed per utterance
              │  score history
              ▼
         [3] TABLES            what is normal; what recurs   ◄── NEW
              │  baselines, motifs
              ▼
         [4] JUDGED            presence gates + character, event-triggered
              │
              ▼
         [5] DELEGATED         tmux, escalation — unchanged
```

Step 3 is new and is the part v2 had no concept of. Steps 1 and 2 are what
Phases 1 and 3 built. Step 4 is what Phases 2.7-2.9 measured. Step 5 is
unchanged and still untested.

The crucial property: **step 3 is where "watches and learns" lives, and it is
pure arithmetic over step 2.** It needs no model, so it runs on the UNO Q, it
costs nothing per utterance, and it is fully unit-testable. Learning is not
the expensive part. Judgement is.

---

## 4. The tables

Two tables, both derived from score history, both answering a question that
only counts can answer.

### 4.1 Baselines — "is this normal?"

Per session and per context, robust aggregates over the measured tracks:

| field | from | why |
|---|---|---|
| `tempo_bpm` | TEMPO CC120 | rhythm of the session |
| `energy_p50`, `energy_p95` | VELOCITY CC7 | loudness distribution |
| `pause_p50_s`, `pause_p95_s` | SILENCE segmentation | rest structure |
| `utterance_p50_s` | SILENCE segmentation | phrasing length |
| `pitch_p05`, `pitch_p95` | PITCH notes | vocal range in use |
| `inflection_up_rate` | INFLECTION CC11 | pitch-activity rate |

**Deviation** is then a real number: each new utterance scored against the
baseline, aggregated into a scalar in 0..1. That scalar is what makes this a
second UI rather than a query box — it flags the moment something is *not*
normal, which is exactly the thing you want told to you and exactly the thing
you would never think to ask for.

Robust aggregates, not means: a single dropped microphone frame must not move
a baseline, and a median/percentile family is chosen so it cannot.

### 4.2 Motifs — "what recurs, so what can be shortened?"

A motif is a recurring figure across the nine tracks: the tuple of
`(gesture, intention-band, energy-band, tempo-band, duration-band, pitch-shape)`
for an utterance, quantised. Quantisation is what makes this tractable and
what makes a habit recognisable at all — an unquantised tuple never repeats
twice.

For each motif: occurrence count, the distinct actions it has resolved to, and
the spread of the judged tracks when it occurred. **A motif with more than
one distinct resolution is not compressible** — that is the rule, and it is
what stops the system from learning a shortcut that means different things on
different days.

This is where the MIDI framing earns its keep rather than merely being tidy: a
habit is *literally* a repeated figure in the score. Not a metaphor for one.
Because the record is the score and not the transcript, the motif is
re-analysable — when the perception layer improves, every historical motif is
re-derivable and the compression decisions are re-made against better data.

---

## 5. Shorter paths

> "this second system builds its shorter paths little by little"

A path shortens by **borrowing from context, not by understanding less**:

```
  "could you please run the test suite in the glasses repo"   (long)
  "run the glasses tests"                                     (referent named)
  "the tests"                                                 (referent = cwd)
  "tests"                                                     (referent + verb borrowed)
```

The length leaves the utterance and gets paid for in context. That is the
entire economy of a peripheral system: you stop carrying what your
surroundings already carry.

Two mechanisms, both local:

* **Referential borrowing** — the working directory supplies the repo; the
  active window supplies the target. Free, because the context is already
  known.
* **Lexical compression** — once a motif has support, the long form is no
  longer required. This one is *earned*, not assumed, and §4.2 is what earns
  it.

**"Shorter" may mean shorter to say. It may never mean fewer gates.** See §6.

---

## 6. Safety: shorter paths are more dangerous, and Phase 0 is the wall

This is the part that inverts the intuition, so it is stated as a law:

> **A shortened path carries more authority per syllable, therefore must pass
> more gates, never fewer.**

The evidence is in the session's own measurements. The baseline INTENTION
question placed `qwop zzz brxt` *higher* on its scale than real speech. A
compressed path is a maximally ambiguous utterance, and ambiguity is precisely
what provokes confident invention. The failure mode of compression is not that
it breaks — it is that it *works confidently while meaning something else*.

Therefore the load-bearing constraint is the Phase 0 invariant, unchanged and
now load-bearing in a second place:

> **No transcript may result in a grant. Only a human decision grants.**

Applied to compression:

* a motif may **propose** a shorter path; it may never **enable** one
* a proposed path is shown with the action it would run and the gate values
  that triggered it — the operator sees what was *understood*, never what was
  *heard*
* no motif, however well-supported, may ever execute without a decision at
  the point of use
* agitation narrows authority further, as in v2

The `tests/test_voice_adversarial.py` suite (34 tests, three rounds of real
defects found by negative control) was written for voice-shaped social
engineering. It becomes, in this design, the suite that keeps compression
honest — and it must be extended, because **a compressed path is the hardest
attack surface in the system**: it is short enough that a social engineer
does not have to sound urgent, only familiar.

That extension is Phase 4 work and is listed as such in §9.

---

## 7. What runs where, as measured

Not a preference. A table of numbers, each one from a named run.

| work | where | latency | evidence |
|---|---|---|---|
| measured tracks (1, 2) | **local, always** | µs | pure arithmetic |
| deviation vs baseline | **local, always** | µs | pure arithmetic |
| 4 presence gates | **local** | **0.967 s** | tev1, `/v1/systemone` |
| GESTURE, INTENTION | **local** | 0.967 s | +0.386 / +0.425 |
| ATTENTION | **cloud** | 0.13 s | local only +0.189 vs cloud +0.420 |
| TIMBRE from text | **nowhere** | — | absent per three instruments |

And the hybrid mechanism is now concrete rather than aspirational: **`tev1`
speaks the TypeSafe SDK's own primitives** — `score`, `choice`, `noul` — over
an undocumented first-party Ollama route, `POST /v1/systemone`. The SDK's
`model_dump()` on `judged.build_questions()` *is* the wire format, and
`judged.read_tracks` already accepts a `client=`. So the judged stack runs
locally with `judged.py` unmodified.

Two caveats that must travel with this table:

* An **UNO Q is slower than this laptop.** 6 GB of RAM, no GPU. The honest
  claim is "local-first means the gates and the tables", not "the whole judged
  half runs locally".
* **ATTENTION is the track that needs the cloud.** It is not a matter of taste
  — it is the one track where local is measurably weak, and if the cloud is
  unreachable then ATTENTION is simply absent rather than guessed.

---

## 8. Phases

| phase | content | status |
|---|---|---|
| 0 | voice-shaped adversarial suite, negative-controlled | done, 34 tests |
| 1 | five measured acoustic tracks | done, worst F0 error 0.034 Hz |
| 2 | four judged tracks + presence gates | done — see §0 |
| 2.5-2.9 | calibration, question health, rewrite, local judge | done |
| 3 | the nine-track score, rendered and parsed back | done, suite now 229 |
| **4** | **the watcher** — continuous, no model, event stream | **next** |
| **5** | **the tables** — baselines then motifs, with deviation | after 4 |
| 6 | shorter paths — propose only, never enable | after 5, gated on §6 |
| 6b | soft-spot instrument (§11) + adversarial coverage | alongside 6 |
| 7 | delegation, tmux escalation — untested since v1 | last |

Phase 4 is next because it is the only component with no measured number
attached and the only one everything else consumes.

---

## 9. Honest limits

* **The conserved budget does not exist, and cannot yet.** η is
  surprise-against-baseline, and there is no baseline until Phase 5, so C
  cannot be chosen honestly (§15). Until both exist the system must report
  the budget as unmeasured rather than as satisfied.
* **No prior-art mechanism in this family has been shown to bind under load.**
  One had a law that could not fail; one had a law that never fired. Any
  conservation adopted here inherits that burden of proof.
* **Federation across cloud/workstation/ESP32 does not exist in any repo
  studied**, despite being described. The claimed implementation is an
  in-process write-through whose `query()` never consults the upstream at
  all. Resource harmony is currently manual deployment.
* **The latent layer does not exist.** §13 is an argument for a tiny JEPA
  over the score history, and the org's three JEPA repositories were measured
  this session and are empty or null (§13). Building one is new work with no
  working prior art to port.
* **Soft-spot detection has never been run.** §11 specifies the divergence
  instrument; it has not been implemented, and its false-positive rate is
  unknown. An instrument that flags everything is as useless as the decision
  tree it was meant to replace.
* **No real speech has been tested.** Every acoustic test runs on synthetic
  tones. Octave errors are the classic failure of a YIN-style estimator on
  human voice and **none of this repo can currently catch one**. The measured
  accuracy (0.034 Hz worst case) is against signals no human produces.
* **The deviation number does not exist yet.** §4.1 is specified, not
  measured, and a deviation score nobody has validated is a number nobody
  should trust.
* **No table has been built from a real session.** Everything in §4 rests on
  synthetic material.
* **Compression is entirely unbuilt and entirely unvalidated.** §5 and §6 are
  an argument, not a result.
* **Labels remain the author's judgements, not Casey's.** The calibration
  figure (ECE 0.0565 ± 0.0027) measures agreement with that labelling, not
  correctness.
* **Delegation has never been exercised.** Step 5 is unchanged from v1 and is
  still a sketch.
* **The SILENCE track does not survive `parse_tracks`.** A note-on with
  velocity 0 is a note-off in MIDI and the parser normalises it, so the mask
  lives in the bytes and not in the reader. Kept because a synth hears it
  correctly; asserted in the tests so it cannot rot silently.

---

## 10. Falsified along the way

Kept visible per the house rule. Each of these was believed, measured, and
found wrong:

* ~~Re-wording the ill-posed questions will fix them.~~ **Measured: it made
  them worse.** INTENTION CC separation +82 → **−59**. Adding a "not an
  utterance" escape level gave the model somewhere to put garbage, at full
  tilt. The cause is structural — `Score` and `Choice` cannot decline.
* ~~Three of four questions are ill-posed.~~ **Half wrong.** ATTENTION really
  was inverted; INTENTION separated +82 on the CC and was never broken. The
  verdict came from spread-confidence, a different quantity from the one a
  synthesiser plays.
* ~~A local judge is 55× too slow to be useful.~~ **Wrong.** `tev1` answers in
  0.34 s per question over `/v1/systemone`.
* ~~`tev1` is unservable.~~ **Wrong.** It refuses the two text endpoints
  because it is not a chat model. The capability is System One.
* ~~Small models cannot do this.~~ **Wrong about the cause.** `qwen2.5:0.5b`
  answered every gate exactly 0.0 because *my prompt* said "report a LOW
  number near 0"; remove the example and it answers 0.9. The sweep measured
  prompt-following.
* ~~Written is working.~~ **Twice.** `question_health.py` (161 lines) raised
  `AttributeError` on first run. `score.py` (248 lines, committed, published)
  raised `TypeError` on first run. Both had tests written after, never before.
* ~~"One tick is one millisecond."~~ **False in the file** until the clock
  event was written unconditionally.
---

## 11. The system's actual job: finding soft spots

> a cashier's job reduced to basic math and robotics, plus the emotional
> intelligence of how to be with different people as they banter, or don't
> want to take something, or seem touchy that day — things an LLM in a job
> could do but a decision tree struggles with

This names the operation precisely, and it is not "replace the job with a
model". It is **decompose a skilled job into its decidable part and its
undecidable part, and put a model only where the logic is insufficient.**

So the interesting capability is meta: the system finds where *its own* logic
needs softening, and how the logic of the larger situation interconnects. A
decision tree fails on "seem touchy that day" not because the tree is badly
written but because that input does not localise — the answer depends on the
whole situation, and the tree has no state. That is what the tables (§4) are
for.

### Soft spots are measurable, not a judgement call

"Place a model here" is only credible if the placement is evidence. The
instrument is **divergence between the deterministic component and the model
component on the same inputs**:

| observation | meaning |
|---|---|
| both agree always | the rule is sufficient; **delete the model** |
| they diverge persistently on a cluster | **soft spot**, and the cluster names it |
| they diverge randomly | noise, not a soft spot — do not spend a model here |

The middle row is the whole idea, and it is the same discipline as everything
else in this repo: an instrument that cannot fail proves nothing, so the
divergence detector must be shown failing on a rule that *is* sufficient.
A soft-spot map that flags every boundary would be as useless as the decision
tree was.

The benefit is symmetric and worth stating: **most rules should win.** A
system that discovers its soft spots will conclude it needs very few models,
and that conclusion is the valuable output.

---

## 12. Large calls calibrating small ones

> larger JEV calls can calibrate smaller models stuck into answers for a
> specific component

This is supervision by a stronger teacher, and most of the plumbing already
exists in this repo:

* `judged.read_tracks(client=...)` — a client is already the swap point, so a
  small local model and the cloud model are the same call site. `tev1` is
  wired in with `judged.py` unmodified.
* `verdict_cache` keys on a question fingerprint, so a small model given a
  *different* question set cannot silently inherit another's numbers.
* `calib.py` measures ECE against ground truth built by construction, which is
  the quantity a teacher can actually improve.
* The `question_fingerprint` makes the teacher/student question set a
  versioned input — retrain on a reword and the old verdicts are invalid by
  construction.

The shape of the loop:

    idle capacity ──► JEV labels / checks a component's answers
                          │
                          ├─ disagree ──► the small model's question is
                          │               soft, or the small model is
                          │               wrong. Either way: not shipped.
                          └─ agree   ──► the small model may stand alone

This is the "local system self-trains under large-system supervision during
idle capacity" idea, and it has a hard property worth keeping: **the teacher
is only consulted off the latency path.** Nothing in a turn waits on a cloud
call, which is what lets the small model be the thing that actually runs.

Measured, so the sizing is not guesswork: `tev1` answers one question in
0.267 s and four gates in 0.967 s locally; JEV answers eight questions in
0.13 s in the cloud. A component that needs one yes/no per event is a
candidate for the small model. A component that needs four is a candidate for
the teacher.

---

## 13. Momentum: tiny models in latent space

> tiny-JEPA models can help find the momentum over more and more data

This fits a specific gap that the measurements left open. Today the
continuous layer (§1 step 1) is **acoustic only** — pitch, energy, tempo,
inflection, silence. There is no continuous *semantic* layer, because nothing
that runs continuously may cost a second. A latent predictive model is the
shape that fits:

* it predicts the **next embedding**, not the next token, so "momentum" is its
  native output rather than something decoded back out of text;
* it needs no calibrated probability, which side-steps the entire calibration
  problem that made the judged tracks expensive;
* it is small enough to run always-on, which the judged layer demonstrably
  cannot be.

Deviation detection is then trivially available: *is this embedding where I
expect it?* — which is §4.1's deviation number expressed in a space where
"where I expect it" is meaningful rather than a hand-picked percentile.

### Ground state, verified: this org's JEPA prior art is empty

Checked against the repositories this session, not recalled:

| repo | source files | finding |
|---|---|---|
| `project-JEPA` | **0** of 43 blobs | documentation only; no implementation exists |
| `plato-audio-jepa` | 2 of 11 | a 16-bin DFT with serde and uuid — not a neural net |
| `fleet-jepa-midi` | 26 of 53 | same/cross-room gap **0.0146**; its own README: *"You Are Not Actually Running JEPA"* |

That is not a reason to skip this. It is the same pattern as forge-soniqo's
±80% F0 and persona-engine's stripped SSML: the *idea* is consistently right
and the *implementations* are consistently null. A correctly built tiny JEPA
over the score history would be new work, not a port.

It also means §4's tables cannot be leaned on to arrive at this — they must be
built, and a latent model is the honest way to build the momentum part. The
baseline half (§4.1) does not need a neural net at all and should be built
first precisely because it is arithmetic.

---

## 14. The attack surface this creates, and it is a new one

Self-modifying logic has a hazard that the rest of this design does not have.

If the system decides where its own logic is insufficient, then an adversary
who can influence its inputs can try to **induce a soft spot** — to push a
safety-relevant action off the deterministic path and onto the model path,
where the grant law is weaker by construction. That is not social engineering
of the *user*; it is social engineering of the *system's own architecture*.

Constraints, in the same spirit as §6:

* soft-spot declarations are **appended-only and witnessed**; the system may
  learn that a rule is insufficient and may never learn that one is
  unnecessary
* a declared soft spot is **logged with its evidence** — the divergence
  cluster, its size, its rate — so it can be audited like any other decision
* **no soft spot may be declared on a single observation.** The instrument in
  §11 requires a persistent cluster, and that threshold is a security
  parameter, not a tuning knob
* a soft spot may **propose**; only a human decision **grants** — unchanged
  from Phase 0, and now covering a path the system chose for itself

Extending the adversarial suite to cover induced soft spots belongs in Phase 4
alongside compression, because both are ways of getting authority without
saying much.

---

## 15. The conserved budget, and the requirement that it bind

> *growth is redistribution under a conserved budget, not creation*

Two repositories in this family attempted a conservation law, and they are
opposite failure modes worth learning from together (`docs/research/prior-art-audit.md`):

* **A law that cannot fail.** `exoJ` defines η ≡ 1−γ, so γ+η=1 is an arithmetic
  identity. No input violates it. It is a safety property that enforces
  nothing.
* **A law that never fired.** `quilt-dba` implements γ+η ≤ C=log₂3 with a real
  predicate and a rollback, and its own recorded verdict is
  `NEVER-BOUND (law untested in vivo)`. The code is correct; the budget was
  simply never reached.

So the design requirement is not "have a conservation law". It is:

> **A conserved quantity must be demonstrated to bind in the measured regime,
> or it is decoration.**

Concretely, for this system:

| term | meaning | how it is measured | measured so far |
|---|---|---|---|
| **γ** | compute spent on a turn | wall clock, tokens, whether the cloud was called | 0.13 s cloud / 2.44 s local / µs measured-only |
| **η** | surprise | prediction error against the §4.1 baseline | **not yet measurable — η does not exist until Phase 5** |
| **C** | the budget | must be *reachable* | must be set from data, not chosen |

Two consequences, both awkward and both load-bearing:

1. **The budget cannot be written until the tables exist**, because η needs a
   baseline to be surprised *about*. C is not a design parameter; it is a
   measurement.
2. Until then **no law is enforced, and the system must say so.** A budget
   that reads 0% utilisation on every turn is `quilt-dba`'s trap with better
   manners.

The *shape* is adopted from `quilt-dba` because it is right: fail closed,
restore the snapshot, log the refusal, **learn nothing from a refused
transition.**

---

## 16. The reflex tier, and an embedding that runs without a model

`quilt-pincher` is the one prior-art mechanism that survives this audit intact,
and it answers the question §7 left open — *what can run continuously?*

**Three tiers, with the expensive one unreachable on the common path:**

```
score >= 0.80          → EXECUTE        no model touched
0.55 <= score < 0.80   → CONFIRM        surface to the operator, no action
score <  0.55          → COMPILE        spend a model
```

This is the presence gate of §0–2 generalised, and it is the mechanism that
makes "decide, don't steer" true rather than aspirational: the common case
never consults a model at all.

**The embedding is hyperdimensional computing, not a neural net.** Tokens hash
to ±1 hypervectors, position binds by permutation, superposition bundles by
sign. Deterministic, seeded, no GPU, no API. That matters more than elegance:
*this is what runs on an UNO Q*, and it makes the continuous layer of §3 step 3
possible at all, because a baseline needs an embedding to compare against and
a transformer embedding costs a forward pass per utterance.

Requirements this project adopts from their implementation, with the two
defects they shipped:

* a **frozen seed and a hash-pinned regression test** — they have this, and it
  is why their HDC is trustworthy
* **never a silent fallback in the similarity path.** Their
  `CloudflareAIEmbedder` falls back to a character histogram when the API is
  unavailable, and the comment admits it. A silent degradation there is the
  least visible and most dangerous kind, because the system keeps producing
  numbers that merely get worse. Fallback in the embedder must be **loud**.

---

## 17. Recognition may be learned; approval may not

> *give the pilot and the autopilot the decisions, not the hand-steering*

The prior art resolves this cleanly and it collides with §5, so the collision
is named here rather than discovered later.

`lever-runner` is described as a *"post-inference command executor"* running
*"pre-approved shell commands by intent, not by tool schemas."* Three ideas
stack in one sentence:

* **decide, not steer** — an intent says *what*; the executor owns *how*. Tool
  schemas are the hand-steering equivalent: reaching for the right tool with
  the right arguments is a chance to be wrong at every step.
* **token-lean** — if the model decides once and a deterministic executor
  carries it out, the expensive thing stops being consulted continuously.
* **pre-approved** — and here is the collision. If the approval set is static
  and immutable at runtime, then "tests" can never *become* runnable: the
  system recognises it, offers, and a human decides. But §5 asks the system to
  **learn** that a phrase maps to an action, which is unavoidably a mutable
  approval surface.

So:

> **Recognition may be learned, mutable, and may only ever propose.
> Approval may be explicit, human-decided, and immutable while the system runs.**

If those two are one surface, then every shorter path is a slow grant and §6's
law becomes unsatisfiable — the system would be teaching itself new authority
while believing it was learning vocabulary.

This also disposes of §14: induced soft spots are only dangerous if *learning*
and *authority* share a surface. Split them, and an adversary manipulating
recognition gains the ability to propose something it was already forbidden to
do, which is a nuisance, rather than the ability to do it, which is not.

Two properties are non-negotiable for the approval surface, both taken from
the prior art's intent rather than its implementation:

* **no runtime path may add to the approval set** — not the model, not a
  learned table, not an induced soft spot
* **the executor is fail-closed** — if the approval check cannot be performed,
  the action does not run. `fastloop-guard`'s stated purpose is
  sub-millisecond validation and sandbox termination; whether it fails open or
  closed is the fact to verify before this is designed further.

---

## 18. Per-utterance trust is a SET, not a scalar

This corrects an open question from v2's own limits list, which said per-
utterance reliability was *"unmeasured"*. It is not unmeasured. It is
**provably unavailable in the form assumed.**

> Distribution-free instance-conditional calibration is **impossible except
> trivially** — Gupta, Podkopaev & Ramdas, *NeurIPS* 2020, arXiv:2006.10564,
> Thm. 3.

So the Phase 2.5 figure (ECE 0.0565 ± 0.0027) is **aggregate** calibration and
is not evidence about any individual utterance. A controller cannot ask JEV
"how confident are you about this one" and get a guaranteed answer, and any
design that pretends otherwise is promising something the literature rules
out.

### What is available instead

Per-item guarantees exist only as **prediction-set coverage**: conformal
prediction (Vovk/Gammerman/Shafer 2005; Campos et al., *TACL* 2024 survey,
arXiv:2405.01976; Jiang et al., *NeurIPS* 2018 arXiv:1805.11783 for the
per-instance trust score). Under a coverage guarantee the model returns a
**set** of plausible readings and is right if the truth is anywhere in it.

> **The set size is the per-utterance trust signal.**

This is not a consolation prize. For a control surface it is better-shaped
than a scalar, because it is the honest geometry of the uncertainty:

| set | reading | controller behaviour |
|---|---|---|
| singleton | certain | drive the CC directly |
| 2–3 | ambiguous but bounded | drive, and flag the spread |
| large | the question does not localise | **hold back** (§6's law) |

And it composes with the presence gate rather than replacing it. Two
independent kinds of uncertainty:

* **spread** — how many readings survive, *given* that something is here
* **gate** — whether anything is here at all

They fail independently, which is exactly what you want. A high gate with a
large set means the utterance is definitely speech and the system cannot say
what kind. A high spread with a low gate means near-nothing was said at all.
Only the conjunction licenses driving hard.

### Consequences, which are not small

1. **The `confidence` field is not a trust signal.** §1 already established
   it is distribution spread and is never driven; this section explains *why*
   that rule was right rather than merely conservative.
2. **A conformal set is the natural encoding for the record.** The score's
   judged tracks would carry a modal reading plus a set-size CC, and both are
   re-analysable when the question set changes — which the fingerprint
   already forces.
3. **Calibration work moves.** Effort stops going into making scalar
   probabilities more accurate and goes into (a) choosing an exchangeable
   calibration set, and (b) emitting sets. Neither is built.
4. **The obstacle is data, not method.** Split-conformal needs a calibration
   set drawn from the deployment distribution. This repo has 58 probes and
   **no real speech at all**, so any set produced today would be conformal
   with respect to a distribution that does not exist in production.
