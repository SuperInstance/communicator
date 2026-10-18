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
