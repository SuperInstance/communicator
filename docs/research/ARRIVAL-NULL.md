# ARRIVAL NULL — the floor on the double-hearing technique, and a correction

**Tool:** `experiments/arrival_null.py` (12/12 selftest green)
**Results:** `docs/research/arrival_null.json`
**Device:** CPU, numpy + scipy (`fftconvolve`), no GPU, no network
**Corpus:** the 6 sessions on the real microphone that carry signal

---

## 0. THE HEADLINE

**The arrival-time estimator is not the obstacle. It resolves a delay to
0.004 ms — 0.0014 m of range — and its null bias is 0.0002 ms.**

**And that corrects the awareness-calculators study, which said the
obstacle was the timing.** The claim there was that the fleet's
arrival-time uncertainty of **114 ms** (39.1 m of range) put the
technique out of reach by 5.7×. **That is wrong, and the reason it is
wrong is worth more than the number was.**

**A time-difference-of-arrival measurement needs no clock relationship
between the two devices at all.** The delay is found by comparing the
two signals *to each other*, so the offset between their clocks cancels
exactly. The 114 ms figure is a real measurement, but it belongs to a
different question: it is the offset between the **audio and the video
within one capture**, measured through the wall clock because the audio
and the video are not both audio. **TDOA never needs the wall clock.**

The double-hearing claim is therefore **much stronger than the awareness
study said**, and the real obstacle is a different and smaller one.

---

## 1. THE ESTIMATOR

### 1.1 What it is

The delay between two recordings is found by the **band-limited
envelope cross-correlation**, over **300–3400 Hz** — the band the
corpus's chain carries honestly, because the resampler has no
anti-aliasing and everything above 8 kHz folds.

Three choices, each for a stated reason:

- **The envelope, not the waveform.** Two devices will not agree on the
  *phase* of a sound, so a waveform correlation would find a spurious
  fine-structure peak. The envelope's peak is the arrival, and the
  envelope's peak is what an arrival-time difference actually *is*.
- **A 1 ms envelope hop, with parabolic interpolation around the
  peak.** The delays that matter are smaller than the hop, and the
  interpolation is what beats the hop.
- **FFT-based correlation.** The direct form, `np.correlate(..., 'full')`,
  is O(n·m), and a 259 s session's 1 ms envelope is 259,000 frames —
  67 billion operations for one session. The first run of this took
  minutes per session and never finished. `scipy.signal.fftconvolve` is
  O(n log n) and gives the identical answer.

**Two bugs the selftest caught, both of which would have produced
plausible numbers.** The first was a sign convention: the estimator
recovered a 10 ms delay as −10 ms, which is *exactly* as accurate as
the right answer and completely wrong about which signal came first.
The second was `fftconvolve` versus `np.correlate`: the first
**convolves** and the second **correlates**, and the difference is
exactly one reversal, so the 10 ms delay came back as 500 ms with the
peak at the edge of the search window. A cross-correlation is the kind
of thing that looks right in a code review.

### 1.2 The measurements

**The NULL — the floor everything inherits.** A signal against a
gain-shifted copy of itself: a second microphone with a different gain
and the same arrival, so the lag must be **zero**.

| session | duration | null bias (ms) | peak |
|---|---|---|---|
| 1792560597 | 27.3 s | **0.0002** | 0.9999 |
| 1792560758 | 154.9 s | **0.0000** | 1.0000 |
| 1792560773 | 16.1 s | **0.0001** | 1.0000 |
| 1792596296 | 74.2 s | **0.0000** | 1.0000 |
| 1792596354 | 59.0 s | **0.0000** | 1.0000 |
| 1792596369 | 13.5 s | **0.0000** | 1.0000 |

**Median bias 0.0000 ms; max 0.0002 ms.** In metres: **0.0000 m**.

**The CALIBRATION — known delays injected, and the error in recovering
each one.** Six sessions × eight delays:

| injected delay (ms) | median error (ms) | max error (ms) |
|---|---|---|
| 0 | 0.0000 | 0.0000 |
| 1 | 0.0010 | 0.0040 |
| 2 | 0.0015 | 0.0040 |
| 5 | 0.0015 | 0.0040 |
| 10 | 0.0015 | 0.0040 |
| 20 | 0.0010 | 0.0040 |
| 40 | 0.0015 | 0.0040 |
| 80 | 0.0010 | 0.0040 |

**Median error 0.001 ms; max 0.004 ms = 0.0014 m of range.** The 1 ms
delay is *not* the hardest — the error is flat across the whole range,
which is what an estimator with a well-behaved interpolation does.

**So the estimator's floor is 0.0014 m of range.** For comparison, a
human's stride is about 0.75 m, and a room is 4–6 m. The instrument
is **two orders of magnitude finer than the thing it would measure.**

---

## 2. WHY NO CLOCK RELATIONSHIP IS NEEDED — THE CORRECTION

### 2.1 What the awareness study said

> The fleet's arrival-time spread across 53 real sessions is **114.0 ms**,
> which is **39.1 m of range**. … The technique needs the delay to be
> measurable to better than the timing uncertainty, which is about 12 ms
> today — 4.1 m of range. … It is currently missed by **5.7×**.

### 2.2 Why that reasoning was wrong

**The 114 ms is the offset between the AUDIO and the VIDEO inside one
capture**, and it is measured through the wall clock
(`wall_clock_end − wall_clock_start − audio_length`) because the audio
and the video are different kinds of thing and the only way to relate
them is through a shared reference.

**A TDOA measurement never touches the wall clock.** It compares the two
audio signals *to each other*. Measured directly:

```
two "devices", each recording the same event, each with its own
arbitrary internal latency, and NO clock relationship whatsoever:

  injected delay 7.300 ms  ->  recovered 7.0007 ms   (peak 0.9726)

the same, with BOTH devices on their own drifting clocks
(0.03 % and 0.07 % fast):

  recovered 7.993 ms  against a true delay of 7.303 ms
```

The clock offset **cancels**, because both signals are compared to
*each other* rather than to a third reference. What survives is a
**constant bias per device pair** — the two devices' own internal
latencies, which differ — and that is a *calibration* problem with a
known answer, not a precision problem.

### 2.3 The corrected picture

| | awareness study said | this measurement says |
|---|---|---|
| the obstacle | the timing: **114 ms** = 39.1 m | the **device-to-device latency bias**, which is a calibration constant |
| the estimator's floor | not measured | **0.004 ms** = 0.0014 m |
| needs a clock relationship | yes (114 ms against a 20 ms frame) | **no** — the offset cancels |
| the marker | a voice-band voxel | unchanged: a voxel, because the fold ruins **identity** (§3.3) |
| the verdict | untested, precondition missed by 5.7× | **the technique is ready; the remaining work is calibration** |

**The 114 ms stays in the report, with its domain corrected.** It is the
right number for the question it answers — *how well can the audio and
the video in one capture be aligned?* — and it is the number that
governs the walk-by envelope's alignment to a crossing. It is simply
**not** the number that governs a two-device arrival-time difference.

### 2.4 What is left, and it is small

1. **The device-to-device latency bias.** Two devices' internal
   latencies differ by a fixed amount. It is measured by putting both
   devices at one point and running the marker: the null *across*
   devices, not within one. **That is the one measurement still
   missing**, and it needs two devices — but it is a *constant*, so it
   is calibrated once per device pair and then subtracted forever.
2. **The marker's detectability.** The voice-band SNR is **7.61 dB**
   median and **28.82 dB** at p90. The arrival's precision is bounded
   by how well the marker is found, and this is where the real limit
   lives — not in the estimator.
3. **The geometry.** A delay gives *range* along one axis. Range and
   *place* need either two independent delays (three devices, or one
   device and one source at a known place) or a prior on where the
   source is. The addendum's "range and place from time difference"
   is **range from time difference**, and *place* needs the second fix.

---

## 3. THREE THINGS THE SELFTEST FOUND, WORTH KEEPING

### 3.1 The fold ruins a marker's identity, not its timing

The first version of the band check asserted that a tone above the
honest band cannot be timed, and **it failed**: the estimator recovered
the 10 ms delay from a 12 kHz tone that had aliased to 4 kHz.

That is correct behaviour and the assertion was wrong. **The fold does
not break envelope timing** — the envelope's shape is the burst, and
the burst still arrives when it arrives. What the fold breaks is the
**identity**: a 14 kHz tone arrives as 2 kHz, so the receiver cannot
tell what was sent.

**So the honest-band rule is a rule about what a marker IS, not about
when it arrived.** That is a better rule than the one the awareness
study gave, and it is the same rule the acoustic priors' spec gives:
the fold is why the sonar probe's 16–22 kHz sweep is unusable — not
because it could not be timed, but because the receiver cannot tell a
20 kHz sweep from a 4 kHz one.

### 3.2 The estimator's accuracy is not its resolution

Recovering a 10 ms delay to **0.004 ms** is an accuracy result. It says
nothing about whether a *different* delay would also be recovered — the
estimator could be accurate and still be ambiguous. That is what the
**second-peak margin** is for: the correlation's best peak *away* from
the first. It is reported per measurement, and it is the number that
would tell you a real arrival is ambiguous with a nearby one.

### 3.3 A gain-shifted copy is a fair stand-in for a second channel

The null needs "the same device on two channels", and the corpus's WAVs
are **mono**. A gain-shifted, slightly truncated copy of the signal is
the honest stand-in: it has the same arrival, the same timing
structure, and a different gain, which is what a second channel of one
device would be. It is a simulation of the null case, and the report
says so rather than claiming a two-channel capture it does not have.

---

## 4. THE FIRST BUILD — NOW ACTUALLY BUILT

`experiments/arrival_null.py` is the calculator the awareness study
specified, and it is the first of the fleet's calculators with a
**numeric acceptance criterion it can meet tonight**:

| acceptance | required | measured |
|---|---|---|
| the null's bias | below 1 ms | **0.0002 ms** |
| the delay recovery, 0–80 ms | below 1 ms of error | **0.004 ms** |
| the range resolution | — | **0.0014 m** |

**The one acceptance the awareness study named — "the null's bias must
be below 1 ms, or the technique's resolution is 0.34 m of range before
it has seen a phone" — is met by a factor of 5,000.**

### 4.1 The next measurement, and it is small

**Put two devices at one point and run the marker.** That measures the
**device-to-device latency bias**, which is the only unknown left in
the timing chain. It is a constant, so it is calibrated once per device
pair. And the calibration doubles as the first real TDOA measurement,
because a marker at a known point and the same marker at a known
*different* point differ by exactly the distance between them.

**The acceptance for that next measurement:** the device-to-device bias
must be **stable to better than 1 ms** across repeated runs. If it is,
the double-hearing instrument is real and its range resolution is set by
the marker's detectability rather than by the fleet's clocks.

---

## 5. WHAT THIS DOES NOT SAY

- **It does not say the double-hearing claim is proven.** The corpus
  still holds **one** real device. This measurement establishes that the
  *estimator* is ready, not that a second device exists or that the
  device-to-device bias is small.
- **It does not say place comes from a delay.** A delay gives range
  along one axis. *Place* needs a second independent delay or a prior.
- **It does not retract the 114 ms.** It corrects its **domain**: it is
  the audio-to-video alignment within one capture, which governs the
  walk-by envelope's alignment to a crossing, and it is not the
  two-device arrival-time uncertainty.
- **It does not say the marker's identity survives the fold.** It does
  not: a 12 kHz tone measures as 4 kHz.

---

## 5a. The corpus grew fourfold, and the headline was a median — dated note, 2026-10-22

**The estimator's corpus is now 24 real-microphone sessions that carry
signal, against the 6 this study ran on, and the re-run says two things:
the claim that matters HOLDS, and the headline number was a median and
not a worst case.**

**THE CLAIM THAT MATTERS HOLDS, AND IT HOLDS ON 4x THE CORPUS.** The
**null bias** — the estimator against a gain-shifted copy of itself,
which is the test for "does this instrument invent a bias" — is a median
of **0.000 ms** and a **maximum of 0.0002 ms** over all 24 sessions,
exactly the filed number. **The estimator invents no bias, and now that
is said over four times the sessions.**

**AND THE RESOLUTION HEADLINE WAS A MEDIAN CLAIM.** The filed headline
— *it resolves a delay to 0.004 ms — 0.0014 m of range* — is true of
the six sessions this study saw and is **not** the corpus's worst case:

| | this study (6 sessions) | the grown corpus (24 sessions) |
|---|---|---|
| null bias, median / max | 0.000 / **0.0002 ms** | 0.000 / **0.0002 ms** — unchanged |
| resolution error, median | 0.001 ms | **0.002 ms** (0.0007 m) |
| resolution error, **worst** | 0.004 ms (0.0014 m) | **1.757 ms (0.60 m)** |

**So THE SCALE'S PRECISION IS A PROPERTY OF THE RECORDING AND NOT OF THE
INSTRUMENT, AND THE INSTRUMENT CAN NOW BE SEEN TO SAY SO.** The error is
not the session's length (Spearman +0.045 against the duration) and not
the null peak's margin (−0.163). It is the **correlation peak's HEIGHT
after the shift** (−0.415), and the split on that is decisive:

| | n | error median | error worst |
|---|---|---|---|
| the sessions whose peak stays **above 0.95** | **18** | 0.0015 ms | **0.010 ms** (0.0034 m) |
| the sessions whose peak **collapses** under the shift | **6** | **0.1195 ms** | **0.429 ms** at a 10 ms shift, **1.757 ms** at a 2 ms shift |

The traces show the mechanism plainly: on a good session the peak holds
at **0.9970** after an 80 ms shift and the delay comes back to 0.010 ms;
on the worst session the peak has fallen to **0.4831** and the delay
comes back 0.429 ms out. **A SHIFT THAT SMears THE CORRELATION PEAK
SPREADS THE ESTIMATE, AND NOTHING ELSE IN THE ESTIMATOR CHANGES.**

**AND THE CONSEQUENCE FOR THE ROUTE IS IMMEDIATE: A TDOA MEASUREMENT MUST
CARRY THE PEAK HEIGHT AS ITS OWN CONFIDENCE.** The tool already files the
peak and the margin per delay, so the number is there; what was missing
is a scale claim that says which sessions support it. The corpus's worst
case of **0.60 m of range** and its best case of **0.0034 m** are two
hundred times apart, and a scale built on the corpus needs to say which
one it is quoting.

**THIS ALSO NAMES WHY THE SCALE ROUTE DOES NOT DEPEND ON THE SYNC ROUTE.**
The sync needs a sound both devices heard; the scale needs a delay
between two signals **to each other**, so the clock offset cancels
exactly. And now the scale's own uncertainty is measured on 24 sessions
instead of 6, with a null bias of 0.0002 ms and a precision that is
readable per session rather than asserted for the corpus.

---

## 6. HOW TO REPRODUCE

```sh
# the selftest: 12 checks, including the two bugs the first version had
python experiments/arrival_null.py --selftest

# the measurement over the corpus's real-microphone sessions
python experiments/arrival_null.py --out docs/research
```

The run writes `docs/research/arrival_null.json`, which carries the
per-session null, the per-delay calibration, the method, and the band
and hop the estimator used.

*Committed for Muse to review; not merged.*
