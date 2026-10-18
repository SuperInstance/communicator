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
| 7 | delegation, tmux escalation — untested since v1 | last |

Phase 4 is next because it is the only component with no measured number
attached and the only one everything else consumes.

---

## 9. Honest limits

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