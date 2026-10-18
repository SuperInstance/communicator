# The backend science

Why each piece of this system is built the way it is, at the level of the
mechanics underneath the code. Three layers: **signal** (the measured
tracks), **decision** (the judged tracks), and **measurement** (how we know
either layer works). Full curated bibliography with verified citations:
`docs/research/PAPERS.md` — every classical result named here has an entry
there.

House rule applied throughout: a claim is either (a) derived here, with the
derivation stated, (b) measured here, with the number and model named, or
(c) attributed to a citation. Nothing else.

---

## 1. Signal: the measured tracks

### 1.1 Why YIN and not plain autocorrelation

Autocorrelation answers "how correlated is the signal with itself at lag τ"
and is satisfied by ANY multiple of the true period — it happily locks onto
a harmonic (τ/2) or a slow subharmonic (2τ). YIN (de Cheveigné & Kawahara,
2002) fixes two things:

1. The **cumulative-mean-normalized difference function**

       d'(τ) = d(τ) · τ / Σ_{j=1..τ} d(j)

   makes every dip's depth comparable across lags, instead of favoring short
   lags the way raw difference does.

2. The **absolute-threshold + first-dip** rule: take the FIRST τ whose d'
   dips below a fixed threshold (0.15 here) rather than the global minimum.
   The true period produces the first deep dip; its multiples produce later,
   equally deep ones. Preferring the first is what breaks the octave-error
   symmetry — and why, when nothing dips under threshold, we fall back to
   the global dip and let confidence decide emission rather than guessing.

forge-soniqo's ±80% F0 tolerance (design doc §0) is what not doing this
costs. Successors (pYIN, SWIPE', CREPE — see PAPERS.md) improve robustness,
not the principle; none is affordable under this repo's pure-stdlib
constraint, and the measured worst-case error here (0.034 Hz at 300 Hz on
synthetic tones) is far inside what the 7-bit CC scale can represent anyway.

### 1.2 Amplitude invariance is a theorem, not a defect

Because d'(τ) is normalized by a running mean of d, it is **scale-invariant
by construction**: multiply the input by 0.0005 and d' at the true lag is
still ≈ 0. So YIN confidence ≈ 1.0 at ANY amplitude. This was first
"discovered" here as a failing test, then derived: YIN measures
*periodicity*, and periodicity has no loudness in it.

Consequence (measured, then encoded): **YIN confidence can never gate
voicing.** A periodic non-speech signal — fan, keyboard, synth bleed — is
confidently periodic at any level. The fix is an external **energy floor**:
a frame with RMS < 1e-3 (≈ −60 dBFS) is unvoiced before YIN runs. The floor
is 10× stricter than segmentation's ABS_FLOOR (1e-4, ≈ −80 dBFS) because
claiming a pitch is a stronger statement than "not silent." dBFS anchor:
RMS r corresponds to 20·log₁₀(r) dBFS.

### 1.3 Frame geometry and the equal-terms window

50 ms windows / 10 ms hop at 16 kHz (800/160 samples) — fine enough to
follow syllable-rate F0 movement, coarse enough to contain ≥ 2 periods of
anything in the 70–400 Hz speaking range (a 70 Hz period is 229 samples;
800-sample windows hold ≥ 3).

The difference function is summed over a FIXED span W = frame_len − τ_max
for every lag. Summing fewer terms at longer lags is the classic
normalization bug that biases toward slow periods: a long-lag d(τ) computed
over a short effective window is noisier, and noise in d translates
directly into spurious dips. Equal terms for every lag, always.

Sub-sample refinement: a parabola through d'(τ−1), d'(τ), d'(τ+1); the
vertex offset is clamped to ±1 sample. This is what takes synthetic-tone
error from ~sample-quantization (~1 Hz) to the measured 0.002–0.034 Hz.

Out-of-range content (outside 70–400 Hz) is reported unvoiced, not
force-fit to the nearest in-range F0 — a gap is better than a wrong number,
the same law as `hz=None` vs `0.0`.

### 1.4 RMS, hysteresis, and two thresholds

Loudness is frame RMS with a hard finite-guard: non-finite inputs are
zeroed and counted, and the output is asserted finite before return — one
NaN must not poison every later frame (the ring-buffer defect class).

Speech/silence segmentation uses **dual thresholds with hysteresis**: enter
speech above `on`, exit below `off = on/2`, hold state in between. A single
threshold chatters at the boundary — a decaying syllable tail would flip
speech↔silence per frame. The deadband is snapkit's governor idea (design
doc §9), and the same principle as Schmitt triggers in electronics.

`on = max(ABS_FLOOR, 0.10 · peak_frame_RMS)` — peak, not percentile, because
Phase-1 clips are single utterances where the loudest frame is the voice by
construction. Stated limit: a long recording with applause would need a
percentile.

Merges: silences < 150 ms are inside a phrase (a stop consonant gap is part
of a word, not punctuation); speech blips < 60 ms are not speech. Both
durations are downstream-consumable choices, not physics.

### 1.5 Tempo and inflection

Tempo: speech-rate → BPM via voiced-speech duration per segment; asserted
for shape and positivity only — synthetic fixtures carry no ground-truth
rhythm (PROGRESS.md Phase 1's stated limit).

Inflection: a **ratio test** with slackwater's semantics — rising if
hz_now/hz_prev > 1 + θ, falling if < 1 − θ, else flat. A ratio, not a
difference, because pitch perception is roughly logarithmic (a semitone is
a constant ratio ≈ 1.0595); equal Hz steps are not equal perceptual steps.

### 1.6 MIDI encoding decisions

Format-1 SMF; **one tick = one millisecond** (480 ppq at the written tempo)
so wall-clock time survives the write→parse round trip exactly — a musical
tick grid would quantize utterance timing to the beat.

CC values are 7-bit (0..127); anything outside is clamped and that clamp is
treated as an encoding error to investigate, not a loud signal. A
probability p maps to CC = round(127·p).

Wire-format quirk, asserted not fixed: **a note-on with velocity 0 IS a
note-off** in MIDI. The SILENCE track's velocity-0 rests survive in the
bytes but are normalized away by parsers; tracked by a named test so the
loss is a property, not a surprise.

---

## 2. Decision: the judged tracks

### 2.1 The central identity

    A calibrated probability is a continuous controller value,
    and a MIDI CC is a continuous controller value in [0, 127].

CC = round(127·p). That is the whole bridge between a decision model and a
synthesiser, and it is why this system uses a decision model rather than a
chat model: no generation, one forward pass, probabilities out.

### 2.2 Probability ≠ confidence (the load-bearing distinction)

`confidence` in the TypeSafe SDK is a function of how **spread** the answer
distribution is — it answers "how unambiguous was this?", not "how likely?".
A flat three-way distribution {0.34, 0.40, 0.24} is maximally ambiguous
(confidence ≈ 0.20) while containing the most probable state. Driving the
CC off confidence would systematically under-drive exactly the ambiguous
cases — which are precisely where a controller wants to hold back. So:

- Choice tracks: CC ← probabilities[chosen]
- Score tracks: CC ← normalized position (see 2.3)
- never `confidence`

For two options, confidence reduces to |2·p_max − 1| — the normalization
zero-poc's dual-provider gate uses to make one threshold mean the same
thing on both backends (see that repo's docs/decision-science.md).

### 2.3 Score's `score` is a position, not a probability

`ScoreAnswer.score` is a float **position on a 0..(n−1) level scale**. The
first implementation multiplied it by 127 and clamped: every score ≥ 1.0
rendered as CC 127, so a polite request and an emergency both came out at
full tilt — plausible-looking, in range, invisible to any consumer.
Normalize by (n − 1) first. This bug shipped silently and was caught by
reading the API, not by the suite (the regression test's helper kept
passing while the call site was wrong — PROGRESS.md Phase 2).

### 2.4 Presence gates: why Noul, why multiply

Score and Choice **must commit** — both primitives always return a position
or a winner. Measured consequence (Phase 2.7): adding abstain wording or an
"not an utterance" level made garbage handling WORSE (INTENTION CC
separation +82 → −59) because the model then had somewhere confident to put
nonsense.

Noul is the one primitive whose semantics include "not present" — it
returns a presence probability where 0.5 is a legitimate middle. So each
track gained a gate question ("is this property present in this text?"),
asked in the SAME call, and the gate is **multiplied into the CC, not
thresholded**: at gate 0.5 the synthesiser receives half the value. A voice
the system is unsure it characterized fades, rather than clicking off —
ambiguous real speech is held back, not erased. Measured: ATTENTION garbage
CC separation −8 → +53.

### 2.5 Safe-parking: controller safety ≠ epistemic health

Garbage probes score INTENTION at median position 0.06/4 with spread-
confidence 0.95. Epistemically that is overconfidence (the model is SURE
nonsense is idle). But for the controller it is harmless: position 0.06
drives CC ≈ 2. The health verdicts split accordingly:

- **ill-posed** — garbage drives the controller somewhere wrong (Choice
  tracks: p(chosen) always asserts SOME option won)
- **safe-parked** — garbage parks a Score at the scale's zero end; the dial
  rests however confident the model claims to be
- **degenerate** — the question sees nothing even in real input
- **healthy** — separates real from garbage

INTENTION was never controller-dangerous; it was never broken. The
distinction cost one measurement to find and is why "confidently wrong"
and "dangerously wrong" are different failure classes here.

### 2.6 Non-determinism and content-addressed stability

Jev is not bit-deterministic: jev-lattice measured 7% of gates flipping
GO↔OPEN at threshold 0.9 on a pure re-ask against identical evidence. A
synthesiser fed a controller that moves between readings of the SAME
utterance is an instrument nobody can play — the jitter is audible and an
operator tuning against it is tuning against noise.

The verdict cache keys on sha1(model + question-fingerprint + text):
model is named by explicit build (floating aliases make numbers
unattributable — on this account `jev-latest` = 1.13.0 ≠ `jev-1.14.0`);
question wording is fingerprinted, so any rewording invalidates its cached
verdicts by construction; the text never changes after it is spoken. No
TTL — staleness here would mean "the model changed its mind about identical
evidence," which is exactly the defect being suppressed.

### 2.7 One call: fan-out economics

The API evaluates questions in parallel: four track questions plus four
presence gates return in ONE round trip (measured 200–550 ms, jev-1.14.0).
In a voice loop, latency is the product — four round trips would be an
architectural failure regardless of per-call speed. Cost corroboration from
gate-replay's bench: a four-question call ≈ $0.05/month at 96 calls/day.

---

## 3. Measurement: how we know

### 3.1 Ground truth by construction

No annotated corpus of utterance character exists here, and a model cannot
grade itself. So corpus labels are true BY CONSTRUCTION — each text written
to embody its label, the way a synthetic 200 Hz tone is ground-truth pitch
(PROGRESS.md Phase 1). Valid where the property is carried by grammar or
wording (speech act, addressee, designed urgency); invalid for delivery
properties (timbre) — those get a wording-separation probe and an explicit
no-calibration-claim, with the presence gate fading them until the acoustic
pipeline exists.

### 3.2 ECE is a bin property, and it is not correctness

Expected calibration error: bin predictions by p, then

    ECE = Σ_bins (share of rows) · |mean predicted p − mean outcome rate|.

Two consequences, both learned here by writing wrong test expectations
first (commit 4082e9f's message):

1. A single row cannot be "perfectly calibrated" — p = 0.8 predicts a RATE,
   and one observation of y = 1 does not make 0.8 right. Calibration lives
   in bins.
2. **ECE is a statement about the distribution of errors, not their
   direction.** A model that is exactly backwards — 10% accuracy where it
   says 0.2–0.3, 90% where it says 0.9–1.0 — scores ECE 0.05. Good ECE is
   necessary for trusting a threshold and insufficient alone; pair it with
   accuracy or a ranking measure.

### 3.3 Brier, Spearman, and honest ties

Brier = mean (p − y)²: proper scoring rule, punishes confident wrongness
quadratically — a useful second opinion because it is decomposable into
calibration + resolution + uncertainty.

Spearman ρ = Pearson on ranks, with ties averaged. Tie-averaging loses
information: monotone data with ties in one variable and none in the other
legitimately scores BELOW 1.0 (measured example: 0.9487). A constant vector
has no denominator: NaN, never a fake 1.0. Both behaviors are asserted,
because a ranking metric that flatters its input is worse than none.

### 3.4 Platt scaling is repair, not upgrade

Platt scaling fits (a, b) mapping scores to calibrated probabilities via
sigmoid(ap + b). The fleet negative that governs reuse: decision-ledger
measured it making an ALREADY-calibrated model worse (winnow ECE
0.033 → 0.053) while repairing a miscalibrated one (laya 0.374 → 0.077).
The classical literature agrees on the condition (see Niculescu-Mizil &
Caruana in PAPERS.md): stretched/sigmoid-distorted models are repaired;
already-regular or overconfident-everywhere models are not improved and can
be hurt. Rule: measure ECE first; apply scaling only on evidence of the
specific miscalibration it fixes.

### 3.5 Question health: separation as the health metric

Two batteries — engineered garbage (non-sentences carrying no timbre,
urgency, addressee, or speech act) and plain real utterances — with median
spread-confidence per track. Health = the gap: confident-on-real AND
unconfident-on-garbage. Either collapse is a finding (invents structure vs
sees nothing); a gate that does not separate may not steer (minigate's
law, adopted). Fingerprinted with decision-ledger's question_hash method,
so a rewrite is evidenced before/after, never asserted.

### 3.6 Negative controls and the never-executed class

Two defect classes recur across this repo's history and the fleet's:

- **Never-executed drafts**: modules fully written, documented, committed,
  and incapable of running (question_health's dict-iteration AttributeError;
  score.py's MidiEvent-shape TypeError; render()'s tuple-unpack). The
  countermeasure is executing the real path — which is why the demo files
  live in examples/ — and end-to-end call-site tests rather than
  helper-in-isolation tests (a helper can pass while its only caller is
  wrong; that exact blindness hid the _scale_position bug at 26/26 green).
- **Sabotage controls that do not apply**: a negative-control patch that
  matched nothing once reported "25 passed" and nearly got recorded as a
  blind test. Every recorded sabotage now asserts the patch landed.

---

## 4. What this stack still does not know

Carried forward, not hidden: nothing about real speech (jitter, creak,
coarticulation) or octave errors — every acoustic fixture is synthetic;
latency asserted nowhere under load; the extra seven tracks to sixteen have
no agreed semantics; TIMBRE calibration waits on acoustics; per-utterance
reliability (as opposed to aggregate ECE) is unmeasured.
