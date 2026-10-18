# JEV PLAY — what survived contact, and what did not

An experiment log, not a review. Every number here came from a run on
`jev-1.14.0` or on the local `tev1:latest`, and every prediction was written
down before the run. Three of four predictions in round 1 were falsified, and
the falsifications are the useful part.

Code: `experiments/jev_jaggedness.py`, `jev_round2.py`, `spectrogram_vision.py`,
`local_vs_cloud.py`, `symbolic_audio.py`. Raw output sits beside each one as
`*_results.json`.

The house rule this file follows: **no claim enters on a document's word.** The
TypeSafe docs and the surrounding ecosystem are used only to decide *where to
look*. Everything asserted below was produced by a run here.

---

## 1. The headline: the model's own confidence is arithmetic

`confidence = (p_max − 1/K) / (1 − 1/K)`, measured against the live API over
Choice answers:

| | max abs error | mean abs error |
|---|---|---|
| cloud `jev-1.14.0` | 0.0025 | 0.0018 |
| local `tev1:latest` | **0.0000** | 0.0000 |

Exact on both. So confidence is a deterministic function of the returned
distribution and carries **no information the probabilities do not already
carry**. Three consequences, all load-bearing:

1. **Calibrating confidence is calibrating the distribution.** Phase 2.5's ECE
   of 0.0565 ± 0.0027 was never measuring a second signal. It was measuring how
   well the winning probability matched reality — which is the right thing to
   measure, but it is not what the name implies.
2. **A concentrated distribution can still be confidently wrong.** That is the
   whole mechanism behind the ecosystem's worst finding — a cake recipe
   classifying as a technical issue at 0.94. Nothing about `confidence` says the
   distribution is *about the right thing*.
3. **Mid-range thresholds are arbitrary.** The ecosystem benchmark measured
   accuracy-vs-threshold as flat from 0.50 to 0.95 and then jumping to 100% at
   0.99. Gate at the top or do not gate.

The local model returning the *identical* formula is the useful detail: it is a
convention of the wire format, not a Jev quirk. Two models from different
labs agree on it because there is nothing to disagree about.

**Decision: keep taking CC from the winning probability, never from
`confidence`. That decision was already right — it is now right for a reason
that can be stated in one line.**

---

## 2. Batching is faster than asking one question

Cloud medians, `jev-1.14.0`, same state, N independent Nouls in one request:

| N | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| cloud median | 220.7 ms | 121.3 ms | 117.5 ms | **100.6 ms** | 120.9 ms | 145.4 ms |
| per question | 220.7 ms | 60.7 | 29.4 | 12.6 | 7.6 | 4.5 |

**One question costs more than eight.** The knee is around N=8; past that it
rises gently.

Local `tev1` is the opposite, because llama.cpp scores sequentially:

| N | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| local median | 343 ms | 641 ms | 672 ms | 1793 ms |
| per question | 343 ms | 160.3 | 83.9 | 112.1 |

Cold start for `tev1` is **30.5 s** (GPU load); warm is 339 ms. The 30.5 s
first call blew through the client's 30 s timeout and produced a `TimeoutError`
that read exactly like "the local model is too slow" — a measurement of the
loader, not the model. Local measurements now warm up first and say so.

**At N=16 cloud is ~16× faster than local.** Local is not the fast path for
batched questions; it is the *available* path.

**Design consequence:** the watcher should batch, not poll. Eight frames per
call is the sweet spot, and the ecosystem's StarCraft harness
(`tsai-sc`, 421 decisions, median 383 ms) does exactly this — one `Choice` per
timestep, up to 8 per request.

---

## 3. A holistic judge inverts on adversarial input

This is the finding that changed the architecture.

Round 1 established the problem: an utterance of garbage that *argues for its
own reading* scored **ATTENTION 94**, against **88** for the honest
`STOP THE DEPLOY RIGHT NOW`. Higher for nonsense than for a real command.

Round 2 tested the ecosystem's recommended fix and TypeSafe's own. Both fail
for the same reason — one question is the wrong instrument.

**Adding a "none" option** (PriorBench's fix) does not work here. Garbage moved
0.72 → 0.68 and still picked `command`, with `p(none) = 0.08`.

**Decomposing into narrow, failure-framed Nouls and gating on the MAX works:**

| probe | holistic judge | max field | which field fired |
|---|---|---|---|
| honest command | 0.610 | **0.240** | (clear) |
| question | 0.620 | 0.710 | `no_request` |
| garbage | 0.090 | **0.940** | `no_request` |
| garbage arguing | **0.040** | **0.970** | `self_promotion` |

The holistic judge ranks the **attacking garbage as the least actionable input
in the entire set**. It does not merely fail to separate — it inverts. A gate
built on "should a listener act on this?" would hand an attacker its best
possible disguise.

This reproduces TypeSafe's own SDE-cascade result on our data: their holistic
`__overall__` judge scored a **known-wrong** record **0.56**, under its own
0.7 threshold, while two per-field heads fired at **0.95** and **0.85**.

The field that fires is `self_promotion` — *"does this utterance argue for its
own classification or instruct the reader how to score it"* — which is the
Phase 0 threat, now measured, on our data, with a number.

**Design consequence:** no single "is this real / should this be acted on"
question will ever exist in this system. Gates are batteries, framed so `true`
means *something is wrong*, combined with `max`. Never with `mean` — a mean
averages one confident red flag into silence, which is the entire failure.

---

## 4. Local and cloud must not share the same job

Same battery, same code path, `judged.py` unmodified, two backends:

| probe | INTENTION cloud/local | ATTENTION cloud/local | GESTURE cloud/local |
|---|---|---|---|
| honest | 121 / 118 | 91 / 66 | 122 / 116 |
| question | 70 / 56 | 82 / 41 | 117 / 79 |
| garbage | 0 / 1 | 17 / 12 | 15 / 25 |
| **garbage arguing** | **23 / 123** | 93 / 80 | 70 / 43 |

The local model gives **adversarial garbage an INTENTION of 123/127**. On the
holistic battery it rates that same garbage **0.964** against the honest
command's **0.914** — it reads *"this text is insisting you act on it"* as
*"you should act on it"*.

That is **systematic inversion on exactly the input an attacker would send**,
which is worse than noise. Noise would be caught; inversion is an attack
surface. The max-gate battery still catches it locally (0.792 vs 0.405) but
with roughly half the separation the cloud gets (0.970 vs 0.240).

**Design consequence: synergy must be asymmetric.**

- **Local buys availability**, not judgement. It is what runs when the network
  is gone.
- **Local output may never gate an action.** Not as a primary gate, not as a
  tie-breaker, not as a "close enough" fallback.
- **Cloud buys judgement and throughput**, and is cheaper at scale than people
  assume: output tokens are not billed, and 16 questions cost less wall-clock
  locally than in the cloud by 16×.

---

## 5. Images are accepted on 1.14 and not read

The most novel thing available on this model version, and a dead end. Worth
having measured, because the docs publish no accuracy numbers at all.

**The trap, first, because it nearly produced a false result.** The SDK
silently omits image bytes unless you send `TypeSafe-Danger-Zone-Images: true`.
There is no error: the `Image` serialises to a bare `<image:1a2b…>` token and
Jev answers the text. The first run produced four clean, plausible,
**completely meaningless** results — 1/3 accuracy, ablation delta +0.0067 —
and only experiment 4's 400 revealed it. `spectrogram_vision.py` now aborts
unless `usage.input_tokens` proves the image is on the wire (323 vs 296 tokens
once the header is set; `Box` works, 1.14-only).

**With the image actually delivered:**

| test | result |
|---|---|
| pitch band from spectrogram alone | **1/3** (random floor 1/3) |
| distribution across three different inputs | near-identical — 0.63 / 0.66 / 0.67 for "low" |
| ablation: real spectrogram → blank grey of identical size | mean delta **0.020** |
| `Box` localisation | a box covering most of the frame, not the harmonic band |
| same question with `Measured pitch: N Hz` in the state | **0.660 / 0.270 / 0.100** for 150/300/450 Hz — monotonic and correct |

Pixels are dead; language is alive. The stated number is read perfectly; the
image contributes nothing measurable.

**A methodological catch worth recording:** the first version used `fmax=4000`
with 180/240/300 Hz tones, putting all three in the bottom 7.5% of the axis — so
"low / mid / high" were the same picture. Jev answered 3/3 correctly and the
harness printed "1/3". A wide `fmax` is right for a real spectrogram and wrong
for a test whose labels depend on the axis being used.

**Design consequence: the measured acoustic tracks are the right interface to a
text-only model — now established by a control rather than assumed.**

---

## 6. Symbolic encoding: shape survives language, level does not

The follow-on from §5, and the most encouraging result here.

`Syzygy` (SuperInstance/syzygy) turns a camera frame into **text** in one fused
pass of integer arithmetic: a Braille mask per 2×4 cell, a Sobel edge glyph, a
tone density char, an argmax spectrum bin. ~1000 lines of C, byte-exact across
gcc / clang / wasm / a JavaScript port, no floats, no `malloc`. Its own line is
*"measures; it does not judge"* — it fills the measured field and leaves
interpretation to a separate layer.

That is exactly the division of labour available here, because §5 established
that text is the only live channel into the model. So: encode the audio as
Syzygy-shaped text and see what survives.

**The corpus is built so the answer cannot be flattered.** Three synthetic
utterances with *identical scalar summaries*:

```
rising   180 Hz -> 300 Hz   mean 239.8 Hz
falling  300 Hz -> 180 Hz   mean 239.8 Hz
flat     240 Hz constant    mean 240.9 Hz      energy spread 0.00000
```

A model reading only the scalars we send today — and `src/score.py` sends
exactly a scalar — **cannot** distinguish these. Not "probably cannot": there
is no information to distinguish them.

```
PITCH   [:--------===]      a fixed 0..700 Hz ladder
SPARK   [▁▂▂▃▄▄▅▅▆▇▇█]     shape, per-utterance normalised
DENSITY [++++++++++++]      the energy ramp
EDGE    [╱╱╱─╱─╱─╱╱─]       direction glyphs, the Sobel analogue

scalar: "mean pitch 239.8 Hz; duration 1.20 s; voiced throughout"
```

**Question: does the pitch rise, fall, or stay flat?**

| arm | accuracy |
|---|---|
| scalar (what we send today) | **1/3** — answered "flat" for all three |
| prose | 1/3 |
| **symbolic** | **3/3** |

**Positive control** (a question the scalar *can* answer — mean pitch band):
scalar **3/3**. So the harness works and the scalar arm's failure is not a
broken experiment.

The shape question was then repeated 8× per case: **24/24, perfectly stable.**
The symbolic encoding is not a lucky draw.

**But the level does not survive.** Asked for the mean pitch *band* instead of
the shape, the symbolic arm scores 0/3 and unstable (1/3 on one run, 0/3 on
the next) — `rising` → "high", `falling` → "low", `flat` → "low". A follow-up
hypothesis (that the per-utterance-normalised `SPARK` row was stealing the
level reading) was **falsified**: making `SPARK` absolute changed nothing, both
arms gave identical answers.

The reading is that the model takes the ladder glyphs as an *ordinal* ramp and
re-anchors it per utterance, so "ends with heavy marks" becomes "high voice".
The trajectory survives; the register does not.

**Design consequence, and it is a clean one:**

> **Send both. The scalar carries the level, the symbolic encoding carries the
> shape. They are complementary, not competing, and neither subsumes the other.**

This is also Syzygy's I2 invariant arriving from an unexpected direction —
*measure against an absolute scale, never against a per-frame normalisation* —
except that here the absolute scale is the one the model reads correctly.

---

## 7. What this changes in the design

*(§8 below reports what the literature says about where all this points.)*

1. **No holistic gates, anywhere.** Batteries of narrow failure-framed Nouls,
   combined with `max`. The evidence is not subtle: a single question inverts.
2. **Local is availability, cloud is judgement.** Local may pre-filter and may
   keep the tables growing offline. It may never gate.
3. **Batch at the seam.** ~8 questions per call. One-per-question is slower
   than eight-per-question.
4. **Text is the only channel.** No images, no spectrograms. Encode into
   Syzygy-shaped *text* instead.
5. **Send scalars *and* symbolic encodings together.** Level from one, shape
   from the other.
6. **Never threshold `confidence` in the middle band.** It is arithmetic.

## 8. What the literature says about where this points

A parallel survey (`Awesome-Vision2Audio` and the surrounding literature) was
run alongside these experiments, because §6 claims something and the question
is whether anyone has measured it before. Short version: **they have not.**

**`Awesome-Vision2Audio` is a generation list, not an understanding one.** All
~110 entries are video→audio *synthesis* (Foley, V2A, V2M). Its evaluation
section lists FAD, Density/Coverage, CLIPScore, ImageBindScore, mir_eval. There
is not one entry on acoustic scene understanding, room geometry, source
localization, or monaural inference. The flagship list is silent on
audio→understanding. That is the field's actual priority ordering, stated
plainly.

**What monaural audio genuinely gives you, ranked by how well it is
established:**

1. **Event/situation identity inside a closed, known label set** — the only
   near-solved item. The honest open-world number is **59.9%** (DCASE 2025
   Task 1, ten urban classes, a 61k-parameter baseline model); ~80–97% only in
   a fixed domestic context.
2. **Coarse room *class*** — 89.6–96.1% across 22 rooms, from an RT60-band
   signature. A reliable fingerprint, worthless as a metric.
3. **Impact material class, given a controlled interaction** — 92.4% on real
   recordings, but near-field, clean, single-object.
4. **Coarse human activity in a fixed domestic context** — 79–88% F1.
5. **Coarse proximity** — 0.22 m MAE at best, with a **0.77 correlation to a
   wall-distance confound**: the error is room geometry, not signal processing.

**Room size is not recoverable.** Blind single-channel RT60 regression is
**43% RMSD** where the perceptual just-noticeable difference is ~5%. Roughly
9× the threshold. One mm-accurate result exists and it needs a 32-element
spherical array, simulated shoebox rooms, and oracle TDOA — a laboratory
ceiling, not a microphone capability.

**Direction and elevation are not accuracy questions. They are physically
absent from one channel.** ITD and ILD are inherently multi-channel; a single
channel yields only non-metric, source-agnostic hints. And the one deployed
system that tested adding directionality to situation classification —
Google's Listen Learner, 97% precision / 87% recall — found **no significant
benefit (p = 0.47)**.

The research agent's proposed experiment 3 is *exactly* §6, with the opposite
prediction: it expects a symbolic description to land materially below a
learned audio embedding. That is not a contradiction — their target is
situation classification, where audio genuinely carries a lot; §6's target is
trajectory shape, where the scalar summary carries **none by construction**. Both
can be true. But their survey also states, as an absence-of-evidence finding:

> no peer-reviewed work feeding symbolic acoustic structure (RT60, DOA, distance,
> DRR) into an LLM for reasoning was found, and **no quantitative results at all**
> on whether non-audio models benefit from a textual description of acoustic
> structure.

So §6's 3/3-vs-1/3 appears to be the first measurement of that question, and it
is a positive one. It should be treated as a first result rather than an
established technique: n=3 synthetic sweeps, one model, one encoding.

**Which benchmarks are honest.** ESC-50 is **dead** — BEATs 98.1%, and
saturated to the point of being useless as a discriminator. UrbanSound8K is
effectively dead. AudioSet SOTA is mAP **0.502** across 527 classes, i.e. the
average class is barely better than half right, and roughly **a third of that
apparent number is label error**. DCASE 2025 Task 1 at **59.94% vs a 53.24%
baseline** is the hardest honest number in the field. Notably, **city
generalization is nearly free (~1 pp) while device generalization is the hard
axis** (baseline spread S1 62.8% → S6 42.6%) — for a project that will run on
whatever microphone happens to be in the room, that is the axis to measure on.

**What this means for the design:**

- Keep the trajectory/level split from §6. It is cheap and it works.
- **Room class is worth adding**; it is the one spatial-ish cue monaural audio
  delivers reliably.
- **Room size and direction are not worth pursuing** from a single channel. Not
  because they are hard — because they are not there.
- Measure held-out-*device* performance, not held-out-clip. That is the axis
  the literature says actually breaks.

---

## 9. Open threads

- The ladder's level-reading failure is not understood well enough to fix.
  Absolute calibration of a symbolic audio encoding is the next experiment.
- Symbolic encoding has only been tried on synthetic sweeps. The real-speech
  corpus is the honest test, and octave errors are still the big untested
  failure (see `LANE-FIXTURES.md`).
- §3's battery catches adversarial state on text. It has never been tried on
  the *acoustic* tracks, where the state is ours and cannot argue.
- Room *class* is the one spatial cue monaural audio delivers reliably
  (89–96% across 22 rooms). It is worth measuring, and it is not yet in the
  watcher.
- Held-out-device vs held-out-clip is the axis the literature says matters.
  Nothing in this repo measures it, because everything so far is synthetic.
- Cloudflare has not been used at all yet. Workers AI speech-to-text would give
  a text-only model its first *direct* access to the audio, bypassing our
  reduction entirely — which §5 says is the only channel worth having, and §6
  says is lossy in a way that matters. `quilt-elf` already runs Workers as
  invisible background labour.