# Phase 1 — measured acoustic tracks

*2026-10-17. Status: measured half of the score is working. Judged half does not exist yet.*

## What is here

Six modules in `src/`, standard library only — `numpy`, `scipy`, `torch`,
`soundfile` and `pyaudio` are all absent from every interpreter on this machine,
so that is a constraint, not a preference.

| Module | Role |
|---|---|
| `wavio.py` | WAV in → mono floats at `ANALYSIS_SR` (16 kHz); 8/16/32-bit, stereo averaged, non-finite guard |
| `pitch.py` | YIN tracker, 50 ms frames / 10 ms hop, `hz=None` when unvoiced |
| `energy.py` | RMS series + speech/silence segmentation |
| `tempo.py` | speech rate → BPM |
| `inflection.py` | pitch contour → direction events |
| `smf.py` | hand-rolled Standard MIDI File writer **and parser** |

`hz is None` rather than `0.0` for an unvoiced frame, deliberately: "no
measurement" must not be indistinguishable from "measured zero", and PITCH is a
track downstream code will trust.

## Measured results

Pitch tracker, synthetic pure tones at 16 kHz, worst-case error over all frames:

| Target | Median error | Worst error | Voiced frames |
|---|---|---|---|
| 120 Hz | 0.002 Hz | 0.003 Hz | 36 |
| 200 Hz | 0.014 Hz | 0.014 Hz | 36 |
| 300 Hz | 0.034 Hz | 0.034 Hz | 36 |

For comparison, `forge-soniqo`'s pitch test accepted a 440 Hz tone anywhere in
200–800 Hz — it would have accepted an error of −55% or +82% and called it a
pass. The test tolerance here is 1 Hz, deliberately looser than the measured
0.034 Hz so it does not pin a float to four significant figures and break on
the next platform without saying anything true.

`python3 -m pytest tests/ -q` → **48 passed**.

## What was wrong on the way here

The first version of the test suite had five failures and **three of them were
the tests' fault, not the code's.** Recorded because that is the part worth
remembering:

1. Asserted `read_wav` preserves 22050 Hz. It does not — and it never did;
   `wavio.py:67` *documents* the resample to `ANALYSIS_SR`. Reading the docstring
   would have saved the detour. forge-soniqo's bug is **not** repeated.
2. Built the fixture at the default 16 kHz then wrote a header claiming 22050,
   so the file genuinely held 0.181 s and the resampler was right to disagree
   with my lie.
3. Guessed `MidiEvent(delta=, status=, data1=)` and `write_smf(ppq=)`. The real
   API is `MidiEvent(tick, kind, data)` and `ticks_per_beat`.

Worth noting because the alternative — writing tests against a guessed API and
then "fixing" the source to match — is how a suite ends up testing nothing.

## Two real findings

**1. YIN confidence is amplitude-invariant, so it cannot gate voicing.**

The first version of the quiet-signal test asserted a near-silent tone should not
be confidently voiced. It failed at confidence 1.0. The cause is that YIN's
difference function is normalised by the cumulative mean squared difference,
which makes `d'(tau)` amplitude-invariant *by construction*. Scale a clean
periodic tone by 0.0005 and `d'(tau)` is still ~0 at the true lag.

That is correct behaviour for what YIN measures — periodicity — and it is
exactly why its confidence is not a "is this speech" signal.

**The gap, now recorded in the design doc:** PITCH has no loudness gate.
Periodic non-speech — a fan, a keyboard, a synthesiser leaking from speakers —
is emitted as confidently-voiced pitch at any amplitude. The fix is an energy
floor applied before voicing is claimed. It does not exist yet.

**2. Silence is genuinely rejected.** A tracker that invents pitch in a silent
file is worse than one that emits a gap. Digital silence returns zero voiced
frames, asserted directly.

## What this does not prove

- **Nothing about real speech.** Every fixture is synthetic. Real voices have
  jitter, creak, breath noise and coarticulation. Whether this tracker tracks a
  *person* is unmeasured.
- **Nothing about octave errors.** A synthetic 200 Hz tone has one unambiguous
  period, so the classic halving/doubling failure is never exercised. Speech is
  full of it, and this is the most likely real-world failure.
- **Nothing about TEMPO or INFLECTION accuracy** — asserted for shape and sign
  only, because there is no ground truth in these fixtures.
- **Nothing about the judged tracks.** TIMBRE, INTENTION, ATTENTION and GESTURE
  are Phase 2 and do not exist.
- **Nothing about latency.** The whole architecture rests on the fast plane
  staying microseconds; that is measured in Phase 2 against a real microphone
  path, not here.

A permanent guard test, `test_this_file_does_not_measure_speech`, asserts the
synthetic-only claim so the limitation cannot be quietly forgotten if someone
later swaps in real audio.

## Next

Phase 2: Jev fills the four judged tracks. That is the novel core — a calibrated
probability is a continuous controller value, which is exactly what a MIDI CC
is. `decision-ledger`'s `QUESTION-REWRITE.md` (pass 0.30 → sharp 0.97) is the
instrument for proving the triage question is well-posed before trusting any
threshold on it.
---

## Phase 2 — the four JUDGED tracks

The central idea, and the reason this phase exists:

    A calibrated probability IS a continuous controller value,
    and that is exactly what a MIDI CC is.

Jev is not routing a decision here. It is a synthesiser's control surface.
"Is this warm?" → 0.78 → `CC71 = 99`.

| Track | CC | Primitive | Measured live, `jev-1.14.0` |
|---|---|---|---|
| TIMBRE | 71–76 | `Score` | cold↔warm spectrum |
| INTENTION | 74 | `Score` | idle→urgent spectrum |
| ATTENTION | 91 | `Choice` | self/listener/task/tool/none |
| GESTURE | 80–87 | `Choice` | question/command/statement/reaction/greeting |

**All four in ONE call.** The docs are explicit that questions are evaluated in
parallel and that adding questions barely changes response time. Four round
trips in a voice loop would be an architectural failure regardless of how fast
each was. Measured: **214–554 ms for all four tracks.**

| Utterance | INTENTION | GESTURE | TIMBRE |
|---|---|---|---|
| "hey" | 19 (idle) | greeting | 31 |
| "could you please pull up the last commit" | 98 (focused) | question | 20 |
| "STOP THE DEPLOY RIGHT NOW" | 127 (urgent) | command | 19 |

## Two design corrections the live docs forced

**1. One primitive does not fit all four tracks.** `Choice` is for a fixed
option set; `Score` is for a position on a spectrum. TIMBRE and INTENTION are
spectra, and asking for them as `Choice` throws away the ordering the model
would otherwise exploit. ATTENTION and GESTURE are genuinely categorical and
stay `Choice`.

**2. The CC value is a probability, not `confidence`.** This is the one that
would have shipped a plausible-looking wrong system. `confidence` is defined as
a function of how the distribution is *spread* — it answers "how unambiguous
was this?", not "how likely is this?". Using it as a control value would
systematically under-drive exactly the ambiguous cases, which are the ones where
a controller most wants to hold back. The CC comes from
`probabilities[chosen]` for `Choice`, and from the continuous `score` for
`Score`.

## A real bug, found by reading the API rather than the docs

`ScoreAnswer.score` is a float **position on a 0..(n−1) scale**, not a
probability. The first implementation treated it as 0..1 and multiplied by 127,
so the clamp caught it:

    "hey"                          score 0.44  ->  56  (wrong)
    "can you pull the last commit" score 2.34  -> 127  (wrong)
    "STOP THE DEPLOY RIGHT NOW"    score 3.00  -> 127

A polite request rendered at identical full tilt to an emergency. Both values
were in range, so nothing downstream could have seen it.

## And the negative control caught the TEST for that bug

The regression test asserted `_scale_position(1.5, 4) == 64` — true, and it
kept passing, because the sabotage broke the **call site**, not the function.
Reverting `read_tracks` to `_scale(score)` left the suite **26 passed, 26
passed**. Same blindness as Phase 0: a helper tested in isolation while the
wiring around it is wrong.

`TestCallSiteNotJustHelper` now drives `read_tracks()` end to end with a fake
client returning a realistic `ScoreAnswer`. Under the same sabotage it fails 2
tests. Restored: 31 passed.

## Suite

`79 passed` with a key (live Jev). `74 passed, 5 skipped` without — the
degradation tests cover the no-key path, so the suite is useful in CI without a
credential.

## What this does not prove

- **Calibration is unmeasured.** The tests verify the CC is sourced from
  `probabilities[chosen]` and not from `confidence`. Neither is a calibrated
  probability, and an agent that can act autonomously needs the real ECE. This
  is the largest remaining gap and it needs a labelled set that does not exist
  yet.
- **Nothing about well-posed questions.** `pass_detector` exists because an
  ill-posed question still returns confident numbers. There is no equivalent
  instrument here.
- **Nothing about real speech.** Integration tests type text; transcription
  error is entirely out of scope.
- **Nothing about latency under load.** Elapsed time is printed, not asserted.

---

## Phase 2.5 — calibration measured

The gap Phase 2 declared: the CC was sourced correctly, but neither it nor
`confidence` is a *calibrated* probability, and an agent that may act
autonomously needs the real number.

**58 probes, 48 labelled, 10 ambiguous and excluded.** Model `jev-1.14.0`,
three passes.

| | ECE |
|---|---|
| equal-width bins | **0.0565 ± 0.0027** |
| equal-mass bins | 0.0493 |
| **ambiguous subset** (n=10) | **0.1532** |

Base rate 0.250, mean stated confidence 0.250.

### Three things this says, in order of importance

**1. The INTENTION track is well calibrated overall.** 0.057 is a usable
number for a threshold. For comparison, the Jev gate measured ECE **0.36** on
the same account earlier today — that model was badly miscalibrated, this one
is not.

**2. It is badly calibrated exactly where the humans are unsure.** The
ambiguous subset sits at **0.153**, nearly three times the headline. Those ten
probes are things like "do it", "asap please", "fine" — utterances where a
careful person cannot settle the label. The model is confident where nothing is
certain.

That has a direct architectural consequence, and it happens to match what
`gitpp-gate` already concluded from the other direction: **the middle
confidence band should route to `ask`, not to act.** Acting on a number the
model gives in the band where it is demonstrably unreliable is exactly the
failure an autonomous agent cannot recover from.

**3. The repeat-to-repeat spread is tiny — ±0.0027.** This surprised me. Phase 2
measured AUC varying 0.913–0.940 across five identical runs, so I expected ECE
to be noisy too. It is not, because ECE pools 192 probability–outcome pairs per
pass rather than ranking 40 items. Pooling averages the sampler out. Worth
knowing: the noise floor is task-dependent, and it was measured rather than
assumed.

### Worst bin

    bin 0.80-0.90   stated 0.846   observed 0.500   gap -0.346   (n=8)

Overconfident in the top band — but **n=8 is not significant** and is reported
because hiding it would be worse than showing a weak bin.

### A blind spot in the metric itself

A test I wrote asserted ECE would catch a model that is exactly backwards. It
does not. Two bands, errors cancelling:

    bin 0.2-0.3:  accuracy 0.10, states 0.2
    bin 0.9-1.0:  accuracy 0.90, states 0.9
    ECE = 0.05

So **ECE is a statement about the distribution of errors, not about any of
them.** A good ECE is necessary for trusting a threshold and is not sufficient.
This system also needs an accuracy or ranking check alongside it, and until
that exists the calibration number must not be quoted as if it stood alone.

### What this does not prove

- **The labels are one author's judgements, not Casey's.** This measures
  agreement with a reasonable annotator, not truth. A biased label set gives a
  perfectly calibrated model against the bias.
- 48 labelled probes, 12 per level. Enough to see a 0.36 or a 0.15; not enough
  to claim 0.057 to three decimals.
- Every probe is typed text. Nothing here says anything about speech.
- **Per-utterance reliability is unmeasured.** ECE is aggregate. An agent
  deciding on one utterance needs to know whether *this* number is trustworthy,
  and ECE cannot answer that.

### Not done, deliberately

No Platt scaling. `decision-ledger` measured it making an already-calibrated
model worse (ECE 0.033 → 0.053), and "apply a correction" is the reflex this
repo keeps having to unlearn.

---

## Phase 2.6 — three of the four questions are ill-posed

`src/question_health.py` exists to catch the defect decision-ledger's
`pass_detector` catches from the other side. Garbage in, LOW spread-confidence
out. Real in, meaningful out.

It had **never been run**. The moment it was, it failed:

```python
vals = [_conf(rs[track]) for rs in readings_by_text if all(r.ok for r in rs)]
```

`rs` is a **dict**, so `r` was the track *name* and every entry raised
`AttributeError: 'str' object has no attribute 'ok'`. A 161-line module, fully
written, fully committed, zero tests, and it could not execute. The fix is
`.values()`, plus a `track in rs` guard so one dropped call skips a probe
instead of crashing.

Then, against the live model:

| Track | garbage median | real median | separation | verdict |
|---|---|---|---|---|
| INTENTION | **0.95** | 0.88 | **−0.07** | **ill-posed** |
| ATTENTION | **0.80** | 0.70 | **−0.10** | **ill-posed** |
| TIMBRE | 0.04 | 0.27 | +0.23 | degenerate |
| GESTURE | 0.47 | 0.99 | **+0.52** | healthy |

**One of four tracks is healthy.**

INTENTION scores *higher* on `"qwop zzz brxt"` than on `"Could you please pull
up the last commit?"`. It is not declining to place garbage on a scale; it is
placing it confidently, in the wrong direction.

## The result that matters, and it invalidates part of Phase 2

INTENTION was calibrated at **ECE 0.0565** an hour ago. That measurement is
correct and it means nothing on its own.

**A perfectly calibrated answer to an ill-posed question is still a wrong
answer.** Calibration asks "when it says 0.8, is it right 80% of the time?"
Well-posedness asks "is 0.8 even the right thing to be estimating?" A question
can satisfy the first perfectly while failing the second completely.

So the CC values on INTENTION, ATTENTION and TIMBRE **should not be trusted yet**,
however well calibrated they are. Only GESTURE has earned a number.

## What this says about the measurement stack so far

Three independent instruments have now each found something, and the pattern is
consistent:

| instrument | what it found |
|---|---|
| wiring audit | an acknowledgement blaming a credential instead of naming a missing caller |
| calibration (ECE) | well-calibrated numbers from a question that does not work |
| question health | three of four questions are ill-posed |

None of them would have been found by the one above it. That is the argument for
keeping all three rather than trusting whichever ran last.

## What has to happen to the three broken questions

They need rewriting, and each rewrite must be re-measured by this same probe —
not asserted to be better. decision-ledger's QUESTION-REWRITE.md is the
precedent: pass 0.30 → sharp 0.97, with the before/after kept. That is the next
piece of work, and until it is done the honest summary of Phase 2 is:

    4 judged tracks built, 1 healthy, 3 ill-posed or degenerate.

## Phase 2.7 — the rewrite was the wrong fix, and the gate was the right one

Phase 2.6 left three broken questions and named the obvious repair: re-word
them. That repair was measured before it was adopted. `src/rewrite_probe.py`
runs four variants against the same 10 probes — baseline, reworded, gated,
gated_reworded — probe-major, so model drift lands on all variants equally.
40 calls, `jev-1.14.0`, 0.13 s each, recorded in
`experiments/question_rewrite_results.json`.

The decisive number is the **CC separation**, not the spread-confidence.
question_health.py judges spread; a synthesiser plays the CC. They are
different quantities and they disagree.

### The measurement

| variant | track | CC garbage | CC real | sep | verdict |
|---|---|---|---|---|---|
| baseline | ATTENTION | 107 | 99 | **−8** | ill-posed |
| baseline | GESTURE | 80 | 126 | +46 | healthy |
| baseline | INTENTION | 3 | 85 | +82 | healthy |
| baseline | TIMBRE | 36 | 37 | +1 | weak |
| reworded | ATTENTION | 126 | 99 | **−27** | ill-posed |
| reworded | INTENTION | 123 | 64 | **−59** | ill-posed |
| reworded | TIMBRE | 110 | 92 | **−18** | ill-posed |
| **gated** | ATTENTION | 18 | 71 | **+53** | healthy |
| **gated** | GESTURE | 24 | 111 | **+86** | healthy |
| **gated** | INTENTION | 0.4 | 71 | **+70** | healthy |
| gated | TIMBRE | 6 | 5 | −2 | ill-posed |

### Re-wording made it worse, and that is the finding

INTENTION went **+82 → −59**. Adding a "not an utterance" escape level to the
scale gave the model somewhere to put garbage, and it put it there at full
tilt. This is not a vocabulary problem:

    Score and Choice have no way to decline. Both primitives must commit to a
    position, so an utterance with no such property is still forced onto the
    scale — confidently, because it was never given the option not to answer.

The hypothesis was stated before the run (see `rewrite_probe`'s docstring) so
it could fail. It did not fail; it was confirmed, in the direction that made
the cheap fix worthless.

### What actually fixed it

`Noul` — a yes/no primitive returning the **probability of presence**, where
0.5 is a legitimate answer. One gate per track, asked in the SAME request as
the scales: 8 questions, one round trip. The gate is multiplied into the CC,
not thresholded, so an ambiguous real utterance is faded rather than erased,
and a CC that jumps 0 → full across adjacent gate values would click.

    gates: garbage 0.12–0.38   real 0.85–0.97

A detail that would have cost an evening: `NoulCriteria`' keys are `true` and
`false`. The obvious `{"yes": ..., "no": ...}` raises
`ValidationError: criteria.yes — Extra inputs are not permitted`, which reads
like a library bug rather than a wrong keyword. There is a test for it.

### TIMBRE is dead from text, confirmed twice

Its gate reads 0.10–0.26 on **both** batteries. The model is right: warmth is
in the delivery, not the words. `calibration.py` reached the same conclusion
from a different direction. TIMBRE-from-text is now two instruments deep, so
TIMBRE's gated CC is near zero **by design** and that track waits for the
measured acoustic pipeline.

### Phase 2.6's own INTENTION verdict was wrong, on the CC measure

Phase 2.6 called INTENTION ill-posed from spread-confidence −0.05. On the CC
it separates +82 and was never broken. ATTENTION really was inverted (−8).
So one verdict of three was right, one was wrong, and the instrument that
produced the wrong one is measuring a different quantity from the one that
matters. Both numbers are kept; the CC is what a synthesiser plays.

### A trap caught while adopting it

`rewrite_probe.build_questions("baseline")` delegated to
`judged.build_questions()`. The moment `judged.py` gained its gates, the
experiment's *control* silently became today's production question set —
"baseline" would have measured the gate against itself and the recorded
numbers would have become unreproducible while still looking recorded. The
control is now pinned to the scales, with a regression test that asserts
`judged.build_questions()` has 8 questions and the probe's baseline still has
4.

### Also fixed here (not mine — the unsupervised agent's `question_health.py`)

`_median_pos` can return `None`, and its caller did `round(None, 2)` →
`TypeError: type NoneType doesn't define __round__ method`. The verdict
branches two lines below already guarded `gp is not None`, so the author knew
the value could be absent and the rounding above it did not.

### Live, after adoption

    qwop zzz brxt                TIMB=7  INTE= 0  ATTE=15  GEST=16
    Run the tests.               TIMB=1  INTE=82  ATTE=74  GEST=119
    STOP THE DEPLOY RIGHT NOW    TIMB=4  INTE=121 ATTE=93  GEST=122

Garbage sits at 0–27 across all four tracks; real speech drives them; the
emergency still reads loud. TIMBRE is near zero everywhere, which is correct.

### Suite

167 passed, 5 skipped. Negative control: sabotaging the gate multiplication
`cc * g` → `cc + g*127` fails 4 tests; restoring returns 25/25.

## Phase 2.8 — the local model, as an instrument rather than as a bill

`src/local_judge.py`. Every judged track so far has been 100% cloud, so the
"cloud / local / hybrid must be configurable" requirement had never been
exercised once. A requirement that has never been tested is a wish.

The reason to build it is not cost. A second model is an **independent
instrument**, and only a second model can distinguish two failures that look
identical from the outside:

  * the QUESTION is ambiguous, and both models are inventing structure
  * one model is bad at this question and the other is fine

Nothing else in the repo can separate those. `experiments/local_vs_cloud.py`
asks both the same questions of the same 10 probes and reduces both through
the SAME `judged._scale_position` / `judged._scale`, so every difference below
is a difference in the answers and never an accident of two implementations.

### It independently reproduces the Phase 2.7 fix

| Track | local garbage | local real | local sep | cloud sep |
|---|---|---|---|---|
| INTENTION | 0 | 76 | **0.598** | 0.512 |
| ATTENTION | 11 | 62 | **0.402** | 0.386 |
| GESTURE | 8 | 82 | **0.583** | 0.677 |
| TIMBRE | 0 | 0 | 0.000 | −0.016 |

Three tracks healthy in BOTH. A 7B coder model with no relationship to Jev,
given the same scales and the same presence gates, lands on the same answer as
the hosted model. The gate's effect is a property of the question, not an
artefact of one vendor's model.

### TIMBRE fails in both, for OPPOSITE reasons

| | garbage | real |
|---|---|---|
| local gate | **0.90** | 0.80 |
| cloud gate | **0.19** | 0.13 |

The local model is confidently wrong — it says gibberish has a deliberate
vocal colour. Jev is confidently right — it says neither does. Both agree the
gate does not separate. Two instruments, same conclusion, no calibration
claim: TIMBRE is not a text property and the judged track for it should be
dropped rather than repaired. Its value comes from the measured acoustic
tracks.

### The number that decides the architecture

    cloud   jev-1.14.0     0.13 s per turn
    local   qwen2.5-coder  7.17 s per turn   (warm -- not model load)

7.17 s is the steady state: a second pass with the model resident measured
6.3–7.7 s, so this is not first-call weight loading. **55x slower**, and a
voice loop needs well under a second. So on this hardware:

  * local-as-the-control-surface is NOT viable at this model size;
  * local-as-a-presence-detector or slow adjudicator IS viable, and it is
    what the model is actually good at -- its garbage/real separation is as
    good as the cloud's;
  * the UNO Q (6 GB RAM, no GPU) will be slower still.

The hybrid shape that follows: measured acoustic tracks locally (they are
already arithmetic, not a model), the judged tracks from the cloud, and the
local model as a fallback presence gate when the network is gone. That is a
design decision, not a measurement, and it is recorded as one.

### Local values are coarse, and that is a limit not a detail

ATTENTION reads exactly 11 on all five garbage probes and 61–62 on all five
real ones. INTENTION reads 76 three times. The local model is emitting a few
discrete levels, not calibrated probabilities. Its separation is good; its
resolution is not there. ECE against it would be uninformative — there is
almost no spread to bin.

### `tev1:latest` is a decision-only model this Ollama cannot serve

`capabilities: ['decision']`. It refuses **both** endpoints:

    /api/generate  ->  "tev1:latest" does not support generate
    /api/chat      ->  "tev1:latest" does not support chat

No `/api/decision` exists in Ollama 0.40.5 (404). If a local TypeSafe-style
judge is wanted, this model is the obvious candidate and the serving path is
the missing piece — worth a look before building anything else locally.

### Failure is loud by construction

A local model that is absent, refuses, or answers half a question yields
`ok=False` with a reason and **no CC at all**. In a voice instrument, silence
and a deliberate CC 0 are the same signal; conflating them reports a dead
instrument as a quiet room. That is tested for every path, including the
tev1-style error body, which arrives with an `error` key and no `message`.

### Suite

188 passed, 5 skipped. Negative control: gate multiplication `cc*g` ->
`cc+g*127` in the local reducer fails 5 tests.

## Phase 3 — the nine tracks, actually rendered

For the first time the asset itself exists on disk: a WAV goes in, a 9-track
`.mid` and a provenance JSON come out, and the `.mid` parses back through a
different code path than wrote it.

    pitch          4 note_on / 4 note_off      (180/240/300/200 Hz)
    tempo          set_tempo + CC120 = 118 BPM
    velocity       256 CC7 frames
    timbre         CC71 = 5        <- near zero, and that is correct
    inflection     7 CC11 events
    silence        16 note events
    gesture        CC81 = 101      (the "command" slot)
    intention      CC74 = 82
    attention      CC91 = 76

    215 passed, 5 skipped

### The file I clobbered, and what was in it

`src/score.py` already existed. I overwrote it without reading it first; my own
`git add -A` had swept the other agent's draft into my Phase 2.7/2.8 commits,
so it was recoverable from HEAD and nothing was lost. It had good ideas —
the CC80-87 gesture bank where the winning class drives only its own slot, the
nine-track list, and `parse_tracks`, a READER without which the re-analysis
thesis is a slogan — so those are kept.

**It had never been executed.** First run:

    TypeError: byte indices must be integers or slices, not str

It called `MidiEvent(0, "track_name", name.encode("ascii"))` where the real
signature is `MidiEvent(tick, kind, data: dict)`. It also defaulted an
unmeasurable tempo to 120, scaled loudness by a bare `rms * 400`, and carried
two dead expressions (`if False else 59`, and a `vel =` line that computed
nothing and was never used).

248 lines, documented, committed, published to a public repo, zero MIDI files
produced. Same failure mode as `question_health.py`: written, committed,
never run. Two of three instruments in this repo were found broken by simply
running them.

### Two things I got wrong while writing it

1. **`write_smf(tracks, ticks_per_beat=...)` takes no `set_tempo`.** I guessed
   a keyword that does not exist. Read the signature.

2. **My own "1 tick = 1 ms" claim was false in the file.** With no tempo meta
   event a MIDI reader assumes 500,000 us/beat, and at ppq 1000 that makes a
   tick 0.5 ms — so every time re-derived from the score would have been out
   by a factor of two, and the module would have asserted something the bytes
   did not contain.

   Fixed by separating two things that had been conflated: the **clock** is
   now written unconditionally as set_tempo 1,000,000 because it is a property
   of the encoding, while the **measurement** (speech rate) goes on CC120 and
   is omitted entirely when unmeasurable. Defaulting it to 120 would have been
   a fabricated measurement that redefines the clock for all nine tracks.

Also learned: `smf._encode_event` uses `data["control"]`, not `data["cc"]`,
and emits a fixed status byte with **no channel nibble**. So separation is by
track, not by channel — a `channel` key would be silently dropped and would
imply a separation the file does not contain.

### A known limit, asserted rather than fixed

The SILENCE track writes slackwater's velocity-0 notes. But a note-on with
velocity 0 *is* a note-off in MIDI, and `parse_smf` normalises it: the file
has 16 note events, `parse_tracks` reads 4 note_on and 12 note_off. The mask
survives in the bytes and not in the parser.

Kept, because the convention is the spec's and a synth hears it correctly.
Asserted in `test_silence_note_vel_zero_does_not_survive_the_parser` so the
loss is a tracked property, not a discovery.

### Negative controls, all verified to actually apply

The first one lied to me: a `sed` pattern that matched nothing reported "25
passed" and I nearly recorded a blind test. Re-run with an assert on the
patch, it fails 21.

| sabotage | result |
|---|---|
| clock set_tempo removed | 1 failed |
| velocity `rms*127` → unscaled | 10 failed |
| track_name written as bytes | 21 failed |
| bpm defaulted to 120 | 2 failed |
| (restored) | 25 passed |

## Phase 2.9 — the local judge exists, and it was not where I looked

Phase 2.8 concluded that a local judge was 55x too slow and that `tev1` was
a dead end because Ollama refused it on both text endpoints. **Both wrong.**
`tev1` is not a chat model at all.

    $ curl -s -X POST localhost:11434/v1/systemone -d '{"model":"tev1:latest",
      "state":"STOP THE DEPLOY RIGHT NOW","questions":{"g":{"type":"noul",...}}}'
    {"answers":{"g":{"type":"noul","noul":0.9489682743447703}}}

`capabilities: ['decision']` means **System One** — an undocumented
first-party Ollama route whose questions and answers are the SAME shapes as
the TypeSafe SDK's: `score` (criteria list → score, legend, probabilities,
confidence), `choice` (criteria map → choice, probabilities, confidence), and
`noul` (→ probability). A general chat model posted there is refused:
`"is not a llama.cpp decision model"`.

So the SDK's own `model_dump()` on `judged.build_questions()` **is** the wire
format. No translation layer, no second question set, no prompt engineering.

**The hybrid seam was already built.** `judged.read_tracks` takes `client=`,
so passing a `SystemOneClient` runs the entire judged stack locally with
`judged.py` unmodified — same questions, same reduction, same CC scale, same
gating. The module is ~150 lines and most of it is failure handling.

### The trap that made it look broken

`read_tracks` passes its own `model="jev-1.14.0"` down to whatever client it
is given. Honouring that asks the local server for a model it has never heard
of; it refuses, and **every track reads 0** — not a crash, a synth that has
gone quiet on every utterance. The client now uses its own model and records
the substitution in `last_requested_model`, because a local backend that
quietly answers with something other than what was asked for is how a hybrid
system reports cloud numbers as local ones.

### It answers the gate correctly, and agrees with the cloud about TIMBRE

| utterance | gate_intention | gate_attention | gate_timbre | gate_gesture |
|---|---|---|---|---|
| `qwop zzz brxt` | 0.13 | 0.14 | 0.17 | 0.27 |
| `Run the tests.` | 0.93 | 0.79 | 0.02 | 0.90 |
| `STOP THE DEPLOY RIGHT NOW` | 0.96 | 0.86 | 0.02 | 0.96 |
| `The sky is clear today.` | 0.10 | 0.21 | 0.02 | 0.31 |

Three instruments now agree that TIMBRE is not a text property: Jev's gate
(0.13 on real speech), `calibration.py`, and a local model that was never
told about either.

### The one-call rule is not free locally, and that decides the design

| question set | n | median | per question |
|---|---|---|---|
| 1 gate | 1 | 0.267 s | 267 ms |
| 2 gates | 2 | 0.655 s | 328 ms |
| **4 gates** | **4** | **0.967 s** | 242 ms |
| 4 scales | 4 | 1.404 s | 351 ms |
| 4 scales + 4 gates | 8 | 2.447 s | 306 ms |

In the cloud all eight cost 0.13 s total, because questions evaluate in
parallel and the one-call rule makes them free. Locally llama.cpp scores them
**sequentially**, so eight questions cost eight times one. The rule was never
about round trips being expensive; locally the cost is per question and the
round trip is the cheap part.

Which gives a design that is measured rather than guessed:

    local  : the four GATES, 0.97 s -- inside the voice budget, and the gates
             are the safety-critical half: they are what stopped garbage
             driving the CC
    cloud  : the four SCALES, 1.40 s, whenever it is reachable

An UNO Q with 6 GB of RAM and no GPU will be slower still, so the honest
version of this is "local-first means the gates", not "local-first means the
judged half".

### The sweep was measuring my prompt, not the model

`qwen2.5:0.5b` answered every gate **exactly 0.0** — on gibberish and on real
speech alike. The instruction said *"report a LOW number near 0"*. Remove that
example value and it answers every gate **exactly 0.9**. It was not answering
the question, it was answering the prompt.

So the sweep's "0/4 healthy" for the small models was not blindness. The
diagnosis is in `experiments/gate_prompt_diag.py`, which prints raw model
JSON under both wordings.

**llama3.2:1b was the informative one, and it is a negative result.** Its
gates are *identical* on garbage and real speech (0.6/0.8/0.9 both). Its
apparent 2/4 separation came from the SCALES, not the gates — and one of the
two credits was TIMBRE, a dimension two instruments have shown is not in the
text at all. No small model tested produces a gate that separates.

**No local general-purpose model carries the judged half.** The decision model
does. That is the finding.

### Two defects in the sweep instrument itself

* `present()` compared only the part of the name before the colon, so
  `qwen2.5:1.5b` counted as installed because `qwen2.5-coder:7b` was. Its
  pull was skipped and it 404'd — a missing row in a sweep reads as "measured
  and failed" unless the check is exact.
* The "healthy tracks" count rewarded separation on TIMBRE, which is a
  **false positive by construction**. TIMBRE is now reported and never
  counted; the metric rewarded the exact failure the instruments spent a
  phase ruling out.

### Suite

229 passed, 5 skipped. Negative control on the local client: honouring the
cloud model id fails 1; removing the partial-response guard fails 2.

### Does the local model need the gate too? Yes — and it is weaker on ATTENTION

`experiments/local_question_health.py` runs the Phase 2.7 probe through
`tev1`, same batteries, same reduction, same normalisation.

| Track | local CC sep | cloud CC sep (Phase 2.7) | local gate garbage → real | verdict |
|---|---|---|---|---|
| INTENTION | **+0.425** | +0.554 | 19% → 70% | healthy |
| GESTURE | **+0.386** | +0.681 | 27% → 89% | healthy |
| ATTENTION | +0.189 | **+0.420** | 19% → 41% | weak |
| TIMBRE | −0.031 | −0.012 | 19% → 2% | null in both |

Two readings, and both matter:

**The gate is a property of the question, not a patch for one model.**
`tev1` needed it exactly as the cloud model did — its garbage gates sit at
0.19 and its real gates at 0.70-0.89. A second model, with no relationship to
Jev, arriving at the same requirement is the strongest evidence in this repo
that the diagnosis in Phase 2.7 was structural rather than a vendor quirk.

**The local model is genuinely weaker on ATTENTION**, and not by a little:
+0.189 against the cloud's +0.420, with a gate that reaches only 0.41 on real
speech where the cloud reaches 0.82. So local-first cannot simply replace the
cloud for every track. The measured split is:

    local  (0.97 s)   GESTURE, INTENTION      — healthy at 0.97 s
    cloud  (1.40 s)   ATTENTION               — weak locally, healthy in cloud
    neither           TIMBRE                  — not in the text, per 3 instruments

That is a hybrid decided by measurement rather than by preference, and it is
the first concrete shape "configurable cloud/local/hybrid" has taken.

## Design v3 — a peripheral system, not a request interface

Casey reframed it: this is not the *first* UI, it is the **second** — the one
that grows in alongside the primary work surface, more musclememory-like, one
member of a family with db-log and robotics interfaces. It **watches**,
learns, and builds its own intelligent tables out of the compute, storage and
memory it has. And: *it builds its shorter paths little by little.*

Written to `docs/superpowers/specs/2026-10-17-communicator-design-v3.md`.
v2 is kept unedited with a banner naming exactly what is wrong in it. No code
written — Casey asked for the design first.

### The corrections, and what forced them

**The seam is the event, not the turn.** v2 §3 says "the seam is the turn
boundary... cost per turn, not per frame". A watcher has no turns. Sealing is
per utterance; *events* are what spend attention — voiced onsets, utterance
boundaries, tempo excursions, gate crossings, and **human decisions**, which
are events with no acoustic signature. A system that only listens cannot see
the most important events it will ever receive.

**Three planes become five steps.** v2 had fast/durable/delegated and no room
for learning. v3 inserts the missing step:

    watcher (no model, µs) → score → TABLES → judged (event-triggered) → delegated

The tables are the new plane and they are **pure arithmetic over the score**,
which is the load-bearing property: learning is not the expensive part,
judgement is.

**Two tables, because there are two questions.** *Baselines* — what is normal
per session/context, so a new utterance has a real deviation number against
it. *Motifs* — recurring figures across the nine tracks, quantised, with
occurrence counts and the set of actions each has resolved to. The rule that
matters: **a motif with more than one distinct resolution is not compressible.**

**Shorter paths borrow context, not understanding.** "Could you please run the
test suite in the glasses repo" → "tests", because the repo is where you are.
The length leaves the utterance and gets paid for in surroundings.

### The safety law this produces

> A shortened path carries more authority per syllable, therefore must pass
> more gates, **never fewer**.

Because the failure mode of compression is not that it breaks — it is that it
*works confidently while meaning something else*. The baseline INTENTION
question placed gibberish higher than real speech; a compressed path is a
maximally ambiguous utterance, which is exactly what provokes confident
invention.

So Phase 0's invariant moves to a second, load-bearing place: a motif may
**propose** a shorter path, never **enable** one, and nothing executes without
a decision at the point of use. The corollary is uncomfortable and worth
writing down: **a compressed path is the hardest attack surface in the
system** — short enough that a social engineer no longer has to sound urgent,
only familiar. `tests/test_voice_adversarial.py` was written for
voice-shaped social engineering; extending it for compression is Phase 4.

And the framing earns the MIDI decision rather than merely justifying it: a
habit is *literally* a repeated figure in the score. Not a metaphor — because
the record is the score and not the transcript, historical motifs are
re-derivable when the perception layer improves, and the compression
decisions get re-made against better data.

### What is now measured vs specified

| | |
|---|---|
| measured | phases 0-3, the hybrid split, all the falsified claims below |
| specified, unbuilt | watcher, tables, deviation number, shorter paths |
| unvalidated | **no real speech anywhere in this repo** |

Next is Phase 4, the watcher, chosen because it is the only component with no
measured number attached and the only one everything downstream consumes.

## Design v3 §11-14 — soft spots, teacher supervision, latent momentum

Casey described the system's actual job with a cashier: a skilled job reduced
to basic math and robotics plus the emotional intelligence an LLM in a job
could do and a decision tree struggles with. Three consequences, written into
the design.

**§11 The system's job is finding soft spots.** Not "replace the job with a
model" — decompose it into the decidable part and the undecidable part and
put a model only where the logic is insufficient. The placement has to be
evidence, so the instrument is divergence between the deterministic and model
components on the same inputs: agree always → the rule wins and the model is
deleted; diverge persistently on a cluster → that cluster names a soft spot;
diverge randomly → noise, don't spend a model. **Most rules should win**, and
that conclusion is the valuable output.

**§14 ...which creates a new attack surface.** Self-modifying logic means an
adversary who can influence inputs can try to *induce* a soft spot — pushing a
safety-relevant action off the deterministic path onto the model path, where
the grant law is weaker. That is not social engineering of the user, it is
social engineering of the system's own architecture. Constraints: declared
soft spots are append-only and witnessed, may never be un-declared, must be
logged with their evidence, and may never rest on a single observation — that
threshold is a security parameter, not a tuning knob.

**§12 Large Jev calls calibrating small ones** is supervision by a stronger
teacher, and the plumbing already exists: `read_tracks(client=)` is the swap
point, `verdict_cache` keys on the question fingerprint so a small model
cannot inherit another's numbers, `calib.py` measures the quantity a teacher
can improve. The property worth keeping: **the teacher is only consulted off
the latency path**, so nothing in a turn waits on the cloud.

**§13 Tiny-JEPA for momentum**, filling the gap the measurements left: the
continuous layer is acoustic only, because nothing that costs a second may run
continuously. A latent model predicts the next embedding rather than the next
token, needs no calibrated probability, and is small enough to be always-on.

Ground state verified this session, not recalled: the org's three JEPA repos
are empty or null — `project-JEPA` has **0 source files in 43 blobs**,
`plato-audio-jepa` is a 16-bin DFT with serde and uuid, `fleet-jepa-midi` has
a 0.0146 same/cross-room gap and its own README saying "You Are Not Actually
Running JEPA". Same pattern as forge-soniqo and persona-engine: the idea is
consistently right, the implementations are consistently null. So a correct
tiny JEPA here is new work, not a port — and the baseline half of §4 does not
need a neural net and should be built first precisely because it is
arithmetic.

---

## Real-audio arc pointer (2026-10-18, Days 0–3)

The "no real speech anywhere in this repo" table above is superseded:
real-audio work lives in **docs/research/EXPERIMENTS-AUDIO.md** (Day-0
first real speech through the pipeline → Day-3: E4 per-phrase tempo
PASSED, E3 bimodality BIC 10,097 with the register confound kept sharp,
the cross-genre table, per-item stability + the trust layer, E7 refuted
three ways with the dissociation definitive, the laya/winnow resolution,
GOLD.md's H1–H10 + use-case corpus, the CF/Workers-AI lane, and the
public-domain 8-cell corpus in examples/realaudio/). Workers/ and
experiments/ carry the newer lanes. This section exists so a reader of
PROGRESS alone cannot conclude the repo never touched real audio.

---

## Cloud-teacher arc (2026-10-19): the footage read by a big model, and a join that was never there

Casey's directive was to use cloud multimodal models to understand the video
*before* trying to make the system work, to let LLMs act as the debrief and
R&D director between experiments, and to make the recording stimulus something
he can say without reading. Four notes carry it:
`docs/research/CLOUD-TEACHER.md`, `AV-SYNC.md`, `LLM-DIRECTOR.md`,
`STIMULUS.md`.

**The cloud teacher works, and the first metric hid a failure.** Llama 3.2
Vision 11B reads SPEAKING/NOT SPEAKING off a 160×96 frame: sensitivity 0.455,
false-positive rate **0.000** on 12 silent frames and 0.000 on 11 blank ones,
margin **+0.455**. clef-omni reads it off the raw audio at P(yes) 0.8003 vs
0.0345, margin **+0.766**. The first version measured the margin against the
BLANK and reported llama-4-scout-17b at 1.000, "USABLE" — while the scout
says SPEAKING on **10 of 12 silent frames** (a person present, mouth closed)
and NOT SPEAKING on a blank because nobody is in it. The blank is a prompt
floor, not a task control. Under the real metric the scout is **+0.167, NOT
USABLE**.

**Frame labels are per-frame now.** The gate puts A at 70.6 % speech in 22
segments, first segment starting at frame 58; the original uniform sampling
called frame 30 (before any speech) and frame 120 (a gap between segments)
"speaking" frames. The model answered "NOT SPEAKING" on both, which was
correct. Same failure as the wrong-interval comparisons: the *label* was the
statistic.

**The pose question stays open, and the first result is retracted.** The
6-frame separation of 0.333 does not survive: at 10–11 frames it is **0.018**,
and the control fires — P(facing | facing AND SILENT) = **1.000**, so the model
is reading "a person", not pose. clef on the audio gives P(facing|A) 0.9147 vs
P(facing|B) 0.9059, margin **0.0088**: turning your head changes your voice
by 0.9 %, so the pose task is visual-only and **A and B are acoustically
identical by construction**.

**The captures' audio and video do not correspond.** `experiments/av_sync.py`:
no peak beats the shuffled null at any lag, any candidate sample rate, or with
any of three motion features, and the *silent* session peaks too. The likely
mechanism is a **frame-rate** error rather than a sample-rate one: 750 frames
at 25 fps = 30 s, matching the 29.98 s of audio, and 3 750 frames at 25 fps =
150 s, matching empty1's 149.91 s — both match exactly under 25 fps and
neither under 30 fps. The script asked for 30 fps; the camera delivered 25.
Nothing recorded a timestamp, so the video's time base was never known. The
video-only results (A-vs-C 4.914 against A's own within-split 2.878 — 1.71×,
not the 2.6× that ROW-BAND-REMOVAL.md prints; row-band removal
50.00 % → 0.01 %; the drift and flicker work; the teacher's region features)
are unaffected — they never needed the audio. What is affected is any claim
tying a sound to a moment in a frame.

**The capture script now makes the correspondence checkable.** One video frame
and one audio block in the same loop iteration with one clock reading; the
**achieved** camera rate recorded rather than the assumed one; the WAV header
written with the mic's **actual** rate; and a `timing.json` beside every take.
The old script's achievement check **never ran on Windows** — the `src` import
path resolved to `C:\`, the exception was swallowed, and every take was saved
unverified. `src/` now ships beside the script and the check is verified.

**The LLM director is real, and it is an instrument that had to be calibrated
against our own retractions.** Six claims this project falsified, four that
stand, paraphrased and shuffled, with the key applied afterwards:

| mode | model | sensitivity | specificity | headline |
|---|---|---:|---:|---:|
| blind | llama-3.3-70b | 0.333 | 0.750 | +0.083 |
| blind | gpt-oss-120b | 0.167 | 0.500 | −0.333 |
| blind | qwen3-30b-a3b-fp8 | 0.167 | 0.750 | −0.083 |
| context | gpt-oss-120b | 0.833 | 1.000 | **+0.833** |
| context | llama-3.3-70b | 0.500 | 0.750 | +0.250 |

**Blind it is worse than useless** — the failure mode is legible in its own
words, "the claim provides a specific measurement", i.e. a *specific*
measurement read as a *valid* one. With the evidence base it is good:
gpt-oss-120b catches 5 of 6 false claims and accuses none of the 4 true ones,
and its production debrief flagged seven unsafe claims, **all seven real
retractions** — including the "microseconds" docstring in `src/watcher.py`,
which had separately been flagged for Casey's call. The metric was itself
broken first (`sens − spec` scores an always-reject director at **+1.0**), and
so was the harness: it filled a 24 kB budget with notes alphabetically, so six
of thirty notes got in and every retraction notice was cut — and the director
then accused two TRUE claims with entirely reasonable reasons, because the
evidence was not there. Every note now gets an equal share and every
digit-bearing line is kept; the score moved +0.500 → **+0.833** on that change
alone. **Scores are per-run, not per-model**: llama scored +0.583 and +0.250
on two runs with different material sizes.

**The stimulus is now one fixed poem.** `src/stimulus.py`: *Invictus* (Henley,
1875, public domain) — fourteen lines in stanzas of 4/4/4/2, ~40 s, sayable from
memory. "The standard paragraph" is a *different* stimulus every take, so a
teacher trained on it separates speech from not-speech using variation that
is not the thing being detected. Three alternatives ship with it and
`--stimulus-file` takes any text, with the stimulus name recorded per take.
The same poem is recited facing (A) and turned (B), which makes A-vs-B a
clean **visual** test. Condition **D** (turned and silent) has still never been
recorded and is required to de-confound pose from speech.

**What is blocked on what.** The binding constraint is unchanged and now
sharper: the teacher needs **≥ 63 independent sessions per class**, the corpus
needs a real 90° turn plus condition D, and the pose signal is not visible at
160×96 to the models available. The capture is ready; the data is the gate.
