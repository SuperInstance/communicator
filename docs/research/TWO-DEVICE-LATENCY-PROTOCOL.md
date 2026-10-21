# THE TWO-DEVICE LATENCY PROTOCOL — the one constant the corpus cannot measure

**Tool:** `experiments/two_device_null.py` (17/17 selftest green)
**Results:** `docs/research/two_device_null.json`
**State:** **the hardware measurement has NOT been run.** The corpus
holds one real input device, so this file is the protocol and the
calculator, and the numbers in it are the calculator's own accuracy
against a synthetic truth — not a measurement of any hardware.

---

## 0. WHAT THIS MEASURES, AND WHY IT IS THE LAST CONSTANT

The arrival-null study (`docs/research/ARRIVAL-NULL.md`) measured the
floor of the arrival estimator on **one** device: a signal against a
gain-shifted copy of itself must return zero, and it returns **0.0002 ms
max**; injected delays from 0 to 80 ms come back to **0.001 ms median,
0.004 ms max**, which is **0.00034 m of range**.

That is the floor for a delay measured between the samples of one
device. What it does not touch is the constant a TDOA between two
**different** devices carries: the difference between the two input
chains' own latencies. A microphone that wakes 2 ms later than the
other will report every arrival 2 ms late, and **2 ms of systematic
error is 0.34 m of range** — three orders of magnitude worse than the
estimator's precision.

**So this is a calibration constant, not a precision problem.** The
estimator is already precise enough; it is the hardware's offset that
has to be measured, once, per device pair.

**And it cannot be measured from the corpus.** There is no paired
two-device capture anywhere: `mic.wav` is silent, `probe.wav` is a 6 s
stimulus tail, and no bundle holds two microphones recording the same
event. What can be done *now* is to make the measurement runnable the
moment two devices exist — which is what this protocol and the
calculator beside it are for.

---

## 1. WHAT YOU NEED

- **Two devices that can record audio at the same time** — a laptop and
  a phone is enough. They do **not** need a shared clock, and they must
  **not** be given one: the whole point of a TDOA is that the offset
  cancels because the two signals are compared to each other.
- **A loudspeaker or a clapper** that both devices can hear clearly.
  A voice is fine and is closer to the corpus's own content than a
  synthetic tone.
- **A tape measure**, and a floor to mark positions on.
- **A quiet room** — quiet enough that the direct sound dominates the
  reverberant tail. The estimator band-limits to 300–3400 Hz, and above
  8 kHz the corpus's resampler folds, so the marker must live in the
  voice band.
- **About twenty minutes.**

---

## 2. THE PROCEDURE

### 2.1 Set the geometry

1. Put the two devices **1.50 m apart**, facing each other across the
   room, and mark both positions on the floor. Measure from the
   **acoustic centre** of each device — the diaphragm of the mic, not
   the centre of the box — and say in the notes which point you used.
2. Put the speaker on the **line between them**, and mark five
   positions along it:

   | position `x` | what it is |
   |---|---|
   | 0.000 m | at device A |
   | 0.375 m | a quarter along |
   | 0.750 m | **the midpoint — the null position** |
   | 1.125 m | three quarters along |
   | 1.500 m | at device B |

3. The midpoint is the **null**: at `x = D/2` the geometric delay is
   zero by symmetry, so the measured delay there **is** the device
   bias, with the geometry divided out. It is a check on the fit and
   it costs nothing.

### 2.2 Record

For **each of the five positions**, record **five repetitions**: move
the speaker, wait for the room to settle, play the marker, and let it
ring out. Five positions × five repetitions = **25 trials**.

Record on **both devices at once**. Whatever each device's own clock
says is irrelevant and must not be used to align anything.

### 2.3 Write the sidecar

Each trial needs a `trial.json` saying where the speaker was:

```json
{ "trial_id": "p02_r03", "position_x_m": 0.75,
  "device_a": "laptop mic", "device_b": "phone mic" }
```

And the bundle needs one `geometry.json`:

```json
{ "separation_m": 1.5, "sample_rate": 48000,
  "device_a": "laptop mic", "device_b": "phone mic" }
```

**The separation is the only number that must come from a tape
measure**, and it enters the bias directly, so its uncertainty
(± 1 cm) is folded into the reported error bar.

---

## 3. THE FILE FORMAT THE CALCULATOR READS

```
<capture>/
  geometry.json
  p00_r00/
    device_a.wav
    device_b.wav
    trial.json
  p00_r01/
    ...
```

Any sample rate the devices record at is fine: the corpus's own reader
resamples to 16 kHz, which is the only rate the analysis stages see.

To see the format without hardware, the tool writes a **synthetic**
capture with a known injected bias:

```sh
python experiments/two_device_null.py --synthetic --bias-ms 3.0
```

---

## 4. THE BAR

| arm | bar | what failing it would mean |
|---|---|---|
| the fitted slope implies the speed of sound | within **5 %** of 343 m/s | the geometry, the estimator or the sign convention is wrong, and the whole fit is worthless |
| the bias's standard error | **≤ 0.5 ms** (≤ 0.086 m of range) | the capture is too small to call the bias |
| the capture's size | **≥ 5 positions × ≥ 5 repetitions** | one position confounds a geometry error with the bias, and one position has no error bar |

**The slope arm was first written at 2 %, and the synthetic sweep
showed that a 2 % bar is a coin flip at this protocol's own trial
count: it was met in 1 of the 5 injected biases.** The arithmetic says
why — the implied speed of sound inherits the slope's relative standard
error, which with a 0.25 ms per-trial residual, 5 positions and a
1.5 m baseline comes out near **2.6 %**, so a 2 % bar is a 0.8-sigma
test. A 5 % bar still catches every error the arm exists to catch (a
wrong sign, a factor of two, a geometry that was taped wrong) and is
met by every synthetic case. **A 2 % bar would need about a 5 m
baseline, or 10 repetitions per position.** The first number is kept
in the tool so the revision is on the record rather than buried.

---

## 5. WHAT THE CALCULATOR HAS ALREADY PROVEN

The hardware measurement is unrun, so what is measured here is **the
measurement**: the calculator's own accuracy against a truth nobody had
to be trusted for.

### 5.1 The design: a swept source

The measured delay between the devices is a straight line in the
speaker's position:

```
the arrival at A        = x/c
the arrival at B        = (D - x)/c
the measured TDOA       = (D - 2x)/c + (lat_B - lat_A)
```

so the line carries **two** numbers: the **slope** is −2/c, which
validates the entire chain at once, and the **intercept** is D/c plus
the **device-to-device bias**, which is the number wanted.

One fixed position would confound a geometry error with the bias, and
a bias measured at one position has no error bar. The sweep separates
them and gives the fit's own standard error.

### 5.2 The accuracy

Five synthetic captures, five injected biases, 25 trials each:

| injected bias | recovered | error | the bias's standard error | implied speed of sound |
|---|---|---|---|---|
| −4.00 ms | −3.9469 ms | 0.0531 ms | 0.1017 ms | 338.03 m/s |
| −2.00 ms | −2.0728 ms | 0.0728 ms | 0.0715 ms | 355.51 m/s |
| 0.00 ms | 0.1070 ms | 0.1070 ms | 0.1249 ms | 333.71 m/s |
| +2.00 ms | 1.9440 ms | 0.0560 ms | 0.1127 ms | 350.23 m/s |
| +4.00 ms | 3.7934 ms | 0.2066 ms | 0.1113 ms | 357.49 m/s |

**Max error 0.2066 ms — 0.0354 m of range. Mean error 0.0991 ms.** The
sign survives a negative bias, and the zero case comes back at
0.107 ms.

### 5.3 The demonstration

The shipped run injected **+3.0 ms** and the calculator returned:

| | |
|---|---|
| the bias | **2.9926 ms** (error 0.0074 ms) |
| its standard error | 0.0970 ms → **0.0166 m of range** |
| the implied speed of sound | 346.75 m/s (1.09 % from 343) |
| the null position's measured delay | 3.166 ms against the fitted 2.993 ms |
| the within-device null, in that very capture | **9.1 × 10⁻¹⁶ ms** |

### 5.4 One bad arrival must not be able to tilt the fit

The sweep found the failure mode, and it is worth recording because
**it is the whole reason the slope arm exists.** At the +4 ms bias, one
trial of 25 read **16.93 ms** where its five siblings read 7.96–8.29
— the estimator had found a weak peak (its margin over the runner-up
was 0.094, where the good trials were near 0.20). Because that trial
sat at the **end of the baseline**, it tilted the slope enough to drag
the implied speed of sound from 343 m/s to **306 m/s**, and it moved
the bias by 0.845 ms.

The fix is a **robust refit**: a trial is dropped only if it is more
than 5 robust standard deviations (MAD) from the line, every dropped
trial is **reported with its residual and its peak margin**, and if
more than a tenth of the trials want dropping then the capture is
**refused** rather than rescued — a capture that needed a lot of
rescuing would be reporting the rescue, not the device. With the
robust fit the +4 ms case recovers 3.7934 ms and the implied speed of
sound comes back to 357.49 m/s.

### 5.5 The refusals

The calculator refuses, with a stated reason and **no bias number**:

- a bundle with **no `geometry.json`** — the separation is unknown, so
  the bias cannot be told from the geometry;
- a bundle with **one position** — the geometry and the bias would be
  the same number;
- a bundle with **too few repetitions**;
- a bundle where **more than a tenth of the trials are outliers**.

---

## 6. HOW TO RUN IT

```sh
# the selftest: 17 checks
python experiments/two_device_null.py --selftest

# a demonstration on a synthetic capture with a known bias
python experiments/two_device_null.py --synthetic --bias-ms 3.0 --out docs/research

# the real thing, on a capture made by the protocol above
python experiments/two_device_null.py --capture /path/to/capture --out docs/research
```

The run writes `docs/research/two_device_null.json`, which carries the
fit, the bias in milliseconds **and in metres of range**, the standard
errors, the null-position check, the within-device null, every rejected
trial, and the calculator's own accuracy sweep.

---

## 7. WHAT THIS DOES NOT DO

- **It is not a measurement of any hardware.** Every number here comes
  from a synthetic capture. The hardware number does not exist yet.
- **It does not certify the estimator's other claims.** The
  within-device null (0.0002 ms) and the resolution (0.004 ms) were
  measured by the arrival-null study and are unaffected; what this adds
  is the *pair* constant, which is new.
- **A lens entry is deliberately NOT filed.** The microphone lens
  (`docs/research/lens/microphone.json`) carries measured sensor
  behaviours with a number, a unit and a provenance, and its
  `ARRIVAL_ESTIMATOR` entry says so plainly: *"the null is a
  WITHIN-DEVICE bias; the device-to-device bias is a different constant
  and is NOT measured."* That sentence stays true until a capture
  lands, and the entry gets written when it does.
- **The calibration is per device PAIR.** Swap either device and the
  constant changes. It is a property of a rig, not of a sensor.

---

*Committed for Muse to review; not merged.*
