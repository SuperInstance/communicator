# AWARENESS CALCULATORS — the addendum's four claims, measured against the corpus

**Source brief:** `docs/research/ADDENDUM-AWARENESS-CALCULATORS.md` (Casey, 2026-10-21)
**Measured from:** `/mnt/c/proart_capture/corpus/free_sessions` — 97 bundles
**Device:** CPU, numpy, the repository's own `src/energy.py` envelope and `src/pitch.py` tracker
**Status of the brief:** present in the main checkout on disk, **not yet in `origin/master` or on this branch** — filed here because it is readable and it is the next brief in the queue.

> **CORRECTION — 2026-10-21, filed after `ARRIVAL-NULL.md`.**
> **§1's central claim was wrong and this note corrects it.**
> This report said the double-hearing technique was blocked by the
> fleet's **114 ms** arrival-time uncertainty — 39.1 m of range, "missed
> by 5.7×". **That reasoning mistook one measurement's domain for
> another's.** The 114 ms is the offset between the **audio and the
> video within one capture**, measured through the wall clock. A
> time-difference-of-arrival measurement compares the two audio signals
> **to each other**, so the clock offset cancels and no clock
> relationship is needed at all.
>
> The estimator, measured: **null bias 0.0002 ms**, **delay recovery
> error 0.004 ms** across 0–80 ms — **0.0014 m of range**. The
> technique is **ready**, and the remaining obstacle is a *calibration*
> constant (the two devices' internal latency difference), not a
> precision problem.
>
> **§1 below is left as filed, unedited, and its §1.2 and §1.4's
> framing is superseded by this note.** The 114 ms remains correct *for
> the question it answers* — how well the audio and video in one
> capture align — which is what governs the walk-by envelope's
> alignment to a crossing (§3.7 of `LENS-AND-WALK-BY.md`).
> Full measurement: `docs/research/ARRIVAL-NULL.md`.

---

## 0. THE HEADLINE

**The addendum's capstone is right and its instruments are not yet
available. Three of its four claims are measurable on this corpus, and
two of the three fail for reasons that are specific and fixable, and one
is untested because the corpus has never heard two of anything.**

| claim | verdict | the number |
|---|---|---|
| **Double-hearing is a sensor** — the delay between two arrivals is geometry | **untested on this corpus, but the instrument is ready** *(corrected — see the note above)* | the estimator's **null bias is 0.0002 ms** and its **delay-recovery error is 0.004 ms** — **0.0014 m of range**. The clock offset cancels; what is left is a calibration constant |
| **Acoustic scene learning** — a space learns its own sound | **the signature is 83 % the device's gain policy** | gain model **R² = 0.8328**; after gain removal the within-room spread is **114.9 Hz**; the between-room spread is **unmeasured** |
| **Screen is a mode** — screen reading is a first-class input | **not measurable: no screen capture in the corpus** | 0 of 97 bundles carry a screen source |
| **The calculator doctrine** — the fleet's real product is calculators | **the metric exists and reads 1.0** | teacher-dependence **1.000** in the tone game; **0 graders** exist fleet-wide |

**The one number that decides the acoustic claim is 114.9 Hz.** Two rooms
can only be told apart if their signatures differ by more than the
within-room spread of one of them. The corpus holds **one** room, so it
cannot say whether real rooms differ by more — and the honest statement
is that this is a half-day capture away.

---

## 1. DOUBLE-HEARING IS A SENSOR

### 1.1 The claim

> Casey talks to Muse on the phone while the laptop hears the same
> voice. The system KNOWS it is hearing one voice twice — and the DELAY
> between the arrivals is geometry: range and place from time difference,
> the phone-node beacon made concrete. Any source captured on two devices
> is a localisation instrument, free.

This is the most interesting claim in the addendum, because it is the
one-clock doctrine taken one node further out: the fixed cameras share
a clock and therefore cannot see each other's geometry at all, but two
devices with two clocks can, because the difference between two clocks
is a delay and a delay against a known signal is a distance.

### 1.2 The precondition, and it is measured

**A delay is only a distance if the delay is bigger than the uncertainty
in measuring it.** The fleet's arrival-time uncertainty is measurable
today, because the audio and the video are two arrivals of the same
session measured against each other on two different clocks — the WAV's
first sample against the session's first frame, derived from
`wall_clock_end − wall_clock_start − audio_length`.

Across the **53 sessions on the real microphone**:

| | |
|---|---|
| the audio's lead over the video | **−105.7 ms to +8.3 ms** |
| the **spread** between sessions | **114.0 ms** |
| the mean lead | −69.6 ms |

**114.0 ms of timing uncertainty is 39.1 m of range.** The offset between
two clocks is not a constant: it drifts with the two devices' own
timebases and with when the capture started, so the fleet cannot
currently point at a phone and say how far away it is to better than
**thirty-nine metres**.

That is the precondition failing, and it fails by a lot.

### 1.3 The bandwidth, and why the marker has to be a voice

The marker has to be in a band the chain carries **honestly**, or the
delay measures the device rather than the path.

- The corpus's resampler (`src/wavio.py`) has **no anti-aliasing**:
  tones at or below **8 kHz** land where they were put (0.0 cents of
  error), a **10 kHz** tone arrives at 6 kHz (**−884.4 cents**), and a
  **14 kHz** tone arrives at 2 kHz (**−3,368.8 cents**). Anything above
  8 kHz in these sessions is not what it appears to be.
- And the ultrasonic idea is already dead on this hardware, measured: a
  **16–22 kHz** sweep came back **87–94 dB below the stimulus**, sitting
  **0.26 dB above the microphone's own floor**.

**So the beacon must live in the voice band, which means it is a VOXEL,
not a chirp.** That is a design consequence, not a preference.

### 1.4 What a voxel inherits

The voice-band detectability, measured over the sessions with real
speech (14,380 frames of 20 ms):

| | |
|---|---|
| voice band (300–3400 Hz) against the band above it, median | **7.61 dB** |
| p90 | **28.82 dB** |
| the frame a marker is found in | **20 ms** |
| **so the range resolution** | **6.86 m** |

A voxel is a signal in noise, and its arrival time is only as good as
the frame it is found in. So the *frame cost* is the *range resolution*:
a 20 ms frame is **6.86 m** of range, and the envelope cross-correlation
that finds the arrival has its own bias to subtract.

### 1.5 The test, specified so it can be run

| step | what it is |
|---|---|
| **setup** | one source, two devices, both recording, on clocks related by a measured sync event |
| **the measurement** | the delay between the two arrivals, by envelope cross-correlation in a band both devices carry honestly (below 8 kHz) |
| **the geometry** | `range = delay × 343 m/s`, after the sync offset and the devices' own separation are subtracted |
| **the NULL** | **the same device on two channels, which must give zero delay.** Any non-zero reading is the cross-correlation's own bias and sets the floor for the resolution |
| **the control** | the same two devices with the source **moved** — the delay must change by the distance the source moved. A delay that does not track the source is a device artefact |
| **the acceptance** | the delay must be measurable to better than the timing uncertainty — **114 ms** today, **39.1 m** of range |

**The null is the important one**, and it is the thing that can be run
*without* a second device: two channels of one device must give zero
delay. That is a free measurement that establishes the floor on the
whole technique, and it is available tonight.

### 1.6 The verdict

**The claim is untested on this corpus — the corpus holds ONE real input
device**, so there is no second arrival of anything and the delay has
nothing to be measured against. That is not refutation.

**But the precondition is measured and it is far off**: 114 ms of timing
uncertainty against a 20 ms detection frame. The technique needs the
uncertainty below the frame cost, and it is currently **5.7× above** it.
The phone node has to fix the *clock* before it is a *sensor*, and the
phone-node brief's own first build — "the two-device time base, and
nothing else" — is the right order for exactly that reason.

---

## 2. ACOUSTIC SCENE LEARNING

### 2.1 The claim

> Over time the system learns a space's sound: kitchen vs desk vs
> paper-in-hand are recognisable from the nature of the sound alone.
> Learned per space, per person. Place and activity become shading-layer
> facts without a camera seeing anything.

### 2.2 The measurement, and two definitions of "floor"

The room's signature is what the room sounds like when **nobody is
speaking**. There are two ways to take it and they disagree, and the
disagreement is itself a finding.

**Take 1 — every non-speech frame.** The floor centroid moves **2.81×**
between sessions in the same room (639.9 Hz to 1798.9 Hz).

**Take 2 — the quietest 10 % of frames only.** The floor centroid moves
**1.495×** (1626.1 Hz to 2430.3 Hz).

**The difference between the two numbers is the size of the contamination
a person adds to a space's own sound**, and it is large: 2.81× against
1.495×. Take 1 measured *the room with speech in it*; take 2 measured
*the room quiet*. Since the claim is explicitly **per space, per
person**, and the addendum wants them separable, take 2 is the right
definition and take 1 is the thing that makes the claim look true.

### 2.3 The confounds, and the gain one is fatal

**The floor's centroid correlates 0.913 with the session's own SNR.** The
apparent room signature is mostly the microphone's **gain policy**.

Fitting it:

```
floor_centroid = 1542.5 + 16.7 × snr_db          R² = 0.8328
```

**83 % of the apparent room signature is the device's gain.** After the
gain is removed, the residual is the candidate room signature:

| session | residual (Hz) |
|---|---|
| 1792560758 | +84.3 |
| 1792560773 | −0.1 |
| 1792596296 | −127.2 |
| 1792596354 | +166.1 |
| 1792596369 | −123.0 |

**Residual spread: 114.9 Hz.** (Raw: 281.0 Hz.)

So a room signature exists, it is measurable, and it has to be
**gain-normalised** to be visible at all. The device's gain policy is
**2.4× larger** than the thing it was hiding.

### 2.4 The control: is the "room" actually the person?

The floor is the place and the speech is the person. If the two
fingerprints track each other, the signature is the person's voice and
the space is not being learned at all.

| | |
|---|---|
| floor centroid (Hz) | 2353.5, 1626.1, 2160.5, 2430.3, 2121.1 |
| speech centroid (Hz) | 967.6, 1301.9, 1107.4, 1155.3, 1150.0 |
| **correlation** | **0.59** |

**0.59 is not zero, and it is not one.** The place and the person are
partly independent — good, the shading layer can carry them separately —
but they are not cleanly separable at n = 5. The honest reading is that
**the correlation cannot be resolved with five sessions**, and the
control is inconclusive rather than passed.

### 2.5 The decision number, and what would close it

**Two spaces can only be told apart if their signatures differ by MORE
than the within-room spread of one of them.**

| | |
|---|---|
| the within-room spread, after gain normalisation | **114.9 Hz** |
| the between-room spread | **UNMEASURED — the corpus holds one room** |
| **what would close it** | **one second room.** Two rooms, one capture protocol each, the between-room spread against this within-room spread. A half-day capture and one number. |

This is the shape of the whole lane, and it is the same shape the
quintessential study found: **the claim is not refuted, it is under-
determined, and the thing that would decide it is a specific capture
rather than more analysis.**

### 2.6 The synthetic rig and the probes are out

**27 of the 97 bundles** are an end-to-end test rig — Chrome's "Fake
Default Audio Input" and a static `.y4m` test pattern. Their audio is a
synthetic **33–39 Hz** burst. A scene classifier fitted on it would
learn the test harness.

And **20 more carry `trigger: "probe"`** with the detection rule reading
*"probe artifact, no detection"* and **no device label at all** (`?`).
Their content is the sonar stimulus — the four originals are the 16–22
kHz exponential sweeps, and the newer ones carry 260 Hz and 83 Hz tones.
A probe's "floor" is the tail of a stimulus, not a room.

So **47 of the 97 bundles are not rooms** and **5 are** — and the 5 are
the entire acoustic-scene population. That is the real constraint on
this lane, and it is worth stating as a count rather than a caveat.

---

## 3. SCREEN IS A MODE

> Screen capture/reading is a first-class input mode, same class as a
> camera view: with screen access the system reads WITH the person, so
> they can talk as if it already knows what they are looking at. Deixis
> ("this", "here", "put it on the screen") resolves against whichever
> modes are live.

**Not measurable on this corpus: 0 of the 97 bundles carry a screen
source.** The capture page records the camera's luma, the microphone's
WAV and the clock, and nothing else.

There is one piece of indirect evidence and it points the right way. The
corpus's own **deixis** problem is already visible in the pause study:
Casey's speech carries a **terminal** pause as "a falling pitch into a
completed sentence", and the terminal class's *syntax* separates it
**6/6** with zero misses — a pre-word function word in **0/6** terminals
against **23/32** internal. A terminal pause is what "this" sounds like
when the thing it refers to is on a screen the system cannot see.

**So the screen mode's absence is not a missing feature; it is a
measured blind spot.** With the screen mode live, the deixis that
currently has to be inferred from prosody alone would have a referent.

---

## 4. THE CALCULATOR DOCTRINE

> The fleet's real product is not answers. It is calculators.

**The metric exists and it reads 1.000.** The teacher-dependence ratio
is the fraction of judgments still needing a teacher, and the tone game's
own is **1.000**: no referee has graded anything, so every judgment in
the repository is uncertified.

**And the calculators this session produced are real, which is the
doctrine's evidence.** Each is a small old inspectable stage that takes
numbers in and gives numbers out:

| calculator | what it is | its state |
|---|---|---|
| the pause grid | the endpointer's gate and vetoes | **46 script-labelled pauses**, AUCs 0.78–0.91 |
| the motion rate policy | the byte budget follows the change | **8,885.5 bytes/s**, 93.4 % still |
| the crossing detector | a blob that translates | **574 crossings** across 80 sessions |
| the lens files | each sensor's self-model | **14 behaviours**, each with a number |
| the RT60 fitter | the room's decay, honestly bounded | **0.254–0.368 s** over 0.10–0.12 s of data |
| the material-cluster fitter | the room's glyph alphabet | **specified, not built** |

**The doctrine's shadow is the teacher-dependence ratio, and the ratio
cannot move until a grader exists.** Every calculator above removes work
from the pipeline; none of them removes the need for someone to say
whether the output is right. That is the honest reading of the capstone:
the fleet is building the library, and the library's *value* is still
only as good as the referee it does not yet have.

---

## 5. WHAT KILLS EACH CLAIM

1. **Double-hearing dies if the timing uncertainty stays at 114 ms.** The
   delay is 39.1 m of range, and no classifier recovers that. The
   phone node's two-device time base has to land the uncertainty below
   the 20 ms detection frame — a **5.7×** improvement.
2. **Double-hearing dies if the null is not run first.** Two channels of
   one device must give zero delay; without that floor, a
   cross-correlation bias becomes a fake position.
3. **Acoustic scene learning dies if the gain is not normalised.** The
   uncorrected signature is **83 %** the device's gain policy, and the
   ratio between a false room and a real one is **2.4×** — large enough
   to invent a kitchen out of a desk.
4. **Acoustic scene learning is undecidable on this corpus**, and no
   amount of analysis fixes it: the between-room spread is unmeasured
   and the corpus holds one room.
5. **The place/person control is inconclusive at n = 5.** A correlation
   of 0.59 is neither independence nor dependence.
6. **Screen mode is not optional for deixis.** The terminal pause
   separates **6/6** on syntax alone; without a referent, that syntax is
   inference rather than resolution.
7. **The calculator doctrine's metric cannot move while the referee
   does not exist.** Teacher-dependence is **1.000** and every
   calculator in the table above is still uncertified.

---

## 6. WHAT TO BUILD FIRST

**One calculator, and it is the one that can run tonight:**

> **`arrival_null(a, b) -> {lag_s, peak, band, bias_hz}`** — the delay
> between two arrivals of the same signal, by envelope cross-correlation
> in a band the chain carries honestly, with the **null** case measured:
> two channels of **one** device, which must return zero.

It needs one device, not two. It establishes the floor on the whole
double-hearing technique, and every later measurement of a real delay is
only as good as that floor. Its acceptance is a number: **the null's
bias must be below 1 ms**, or the technique's resolution is 0.34 m of
range before it has seen a phone.

**And the second, which needs a room rather than a device:**

> **`room_signature(session) -> {centroid_hz, gain_normalised_centroid_hz, residual_hz, floor_db, snr_db}`** — the room's own sound, with the gain removed and the contamination stated.

Acceptance: it must return `None` for the synthetic rig's 27 sessions
and for the 20 probe bundles; it must report the gain-explained fraction
alongside the signature; and the **first two rooms** must give a
between-room spread **larger than 114.9 Hz**, or the space is not
learnable from sound alone on this hardware.

---

## 7. WHAT THIS DOES NOT SAY

- **It does not say double-hearing does not work.** It says the corpus
  has one device, and the precondition — timing uncertainty below the
  detection frame — is currently missed by **5.7×**.
- **It does not say the room has no signature.** It says 83 % of the
  apparent signature is the device's gain, and that after removing it the
  residual spread is **114.9 Hz** and the between-room spread is
  unmeasured.
- **It does not say screen mode is unimportant.** It says the corpus
  cannot see a screen, and that deixis currently has to be inferred from
  prosody.
- **It does not grade the calculators listed above.** The
  teacher-dependence ratio is **1.000** and no referee exists.

---

*Measured 2026-10-21 from the corpus as it stood: 97 bundles, 53 on the
real microphone, 27 synthetic, 20 probe bundles with no device label.
The corpus is live and grew to 106 during the work; every number here is
a snapshot.*

*Committed for Muse to review; not merged.*
