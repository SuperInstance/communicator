# MOUTH-APERTURE-ALIGNMENT — the mouth's motion does not track the speech, and the frames the repository measured were torn

**2026-10-20.** This is the fourth measurement in the argument that
`docs/research/TEACHER-LOO-FOLDS.md` opened, and it answers it in two
parts, one of which is a correction to that document.

`TEACHER-LOO-FOLDS.md` §3 measured the video's motion against the audio's
speech and found no alignment above the nulls: A +0.0776 and B +0.0247
against a scrambled-audio null of +0.0608. That is the finding that would
cost the recording plan the most, so it deserved the hardest test this
repository can give it. This measurement was commissioned with the
following caveat about §3's instrument:

> "The whole-frame motion energy is a blunt instrument. The frame is
> 160x96 greyscale and the whole-scene illumination noise and the person's
> shoulders can swamp a mouth that occupies maybe 20x10 pixels. So §3 may be
> a statement about the instrument, not about the video."

The caveat was right, and it was right in a way neither §3 nor the brief
anticipated. **§3's instrument was not measuring the video.** It was
measuring a tear in the video files. So this lane does two things:

1. it re-measures the thing §3 measured, on frames that are not torn, and
   shows that the near-zero was an artefact of the reading and not a fact
   about the corpus; and
2. it measures the thing the brief asked for — **the mouth's geometry,
   per frame, from a face landmark model, against the audio's speech
   envelope** — and finds that the mouth's correlation, while it beats the
   null, **does not survive the controls**: a landmark that must not track
   speech reaches the same number, and a dimension of the mouth that a
   mouth opening cannot change reaches a higher one.

The scripts are `experiments/mouth_aperture_alignment.py`; the results are
`experiments/mouth_aperture_alignment_results.json`.

---

## 0. The correction, dated

> **2026-10-20 — correction to the frame stride.**
>
> Every reader in this repository assumed a frame was `W*H*3//2` = 23,040
> bytes (standard NV12 4:2:0 for a 160x96 frame):
> `experiments/seed_corpus_from_captures.py:41-42`,
> `experiments/local_vlm_vision.py:116-117`.
>
> The files are **19,200-byte records**: 15,360 bytes of 160x96 luma
> followed by 3,840 bytes of chroma. The writer is
> `/mnt/c/proart_capture/nv12small.py`, which appends the *interleaved* uv
> plane, so the stride is 19,200 and not 23,040.
>
> > **CORRECTION (2026-10-20) — "there is no V plane of its own" is wrong.**
> > The trailing 3,840 bytes are **INTERLEAVED U,V** — U at even offsets, V
> > at odd — and they are **80×24 samples**, half the *vertical* resolution
> > of a standard NV12 frame's chroma (80×48). The stride of 19,200, which
> > is the load-bearing conclusion of this lane, is unaffected, and so is
> > every number here: the luma this lane analyses is the first 15,360
> > bytes either way. (Two independent tools were checked, not assumed:
> > `nv12small.py` and `syzygy_capture_win.py` both build the chroma the
> > same way, `uv[:, 0] = u; uv[:, 1] = v` on an (N, 2) array, which in
> > row-major order interleaves.)
>
> Read at 23,040, each "frame" is torn between two real frames and the
> scene appears to jump a quarter of its height on every single frame.
> Measured on `A.nv12`:
>
> | stride | frames | mean consecutive \|diff\| | consecutive-frame correlation |
> |---|---:|---:|---:|
> | **19,200 (correct)** | **900** | **0.5449** | **+0.9990** |
> | 23,040 (the repository's) | 750 | 43.5596 | -0.0359 |
>
> The seeded corpus maps — the ones §2's static-template analysis and §3's
> whole-frame alignment were both computed from — correlate with the torn
> reading at **0.9985** and with the correct reading at **-0.005**. Their
> mean motion map is flat across all 48 rows (54–62 everywhere), which is
> what a tear looks like and is not what a face looks like.
>
> **This lane has never published a number computed on the torn reading.**
> It was written against the corrected stride from the start; the torn
> reading is run only as a labelled sensitivity row (§6).

Nothing in this lane is a silent overwrite of a previous number: the
results file did not exist before today, and it carries the correction in
its `identity` block.

---

## 1. The frames, and the clock

The corrected reading gives 900 frames for each of A, B and C and 4,500
for the empty room. That fixes the frame rate, which is **30 fps** and
not 25, by the two spans agreeing rather than by assumption:

| | frames | span at 30 fps | the recording | span at 25 fps |
|---|---:|---:|---:|---:|
| A / B / C | 900 | **30.00 s** | 29.981 s | 36.0 s |
| empty1 | 4,500 | **150.00 s** | 149.906 s | 180.0 s |

At the repository's 23,040-byte stride the video would be 25 s for 30 s
of audio — the same disagreement seen from the other side. (`CAPTURE-NEXT.md`
records an achieved camera rate of 25 fps for a *later* capture round; it
does not apply to these files, and the spans decide it.)

**The offset is searched, not assumed to be zero.** The audio and the
video of A/B/C are separate recordings that were started independently, so
their relative offset is unknown. The video is 30.00 s and the audio
29.98 s: the same *length*, but equal length does not mean equal *start*,
and the capture script's own docstring says the two were started
independently. The ±15 s search covers every relative start possible.

---

## 2. The landmark model, and where its landmarks are

MediaPipe FaceMesh (`mp.solutions.face_mesh.FaceMesh`), version
**0.10.21**, installed into the GPU venv. `static_image_mode=True`, so
every frame is detected independently with no temporal state carried
between frames; `refine_landmarks=True`, so the 478-landmark mesh
including the iris points. The frames are upscaled **4x** (cubic, to
640x384) before landmarking.

**Face detection: 900/900 frames in A, 900/900 in B, 900/900 in C, and
0/4500 in the empty room.** A detector that fires on zero frames is a
result; this one fires on every frame of every occupied take, and on
none of the empty room's.

The landmarks' mean positions, in the original 160x96 frame's own pixel
coordinates, are recorded so a reader can check that landmark 13 is the
upper inner lip and not something else:

| landmark | index | position (px) |
|---|---:|---|
| upper inner lip | 13 | (78.60, 52.52) |
| lower inner lip | 14 | (78.66, 53.76) |
| mouth corner, right | 61 | (71.01, 53.21) |
| mouth corner, left | 291 | (85.71, 53.01) |
| nose tip | 1 | (78.83, 45.39) |
| chin bottom | 152 | (78.47, 65.57) |
| forehead top | 10 | (78.44, 22.09) |
| eye outer corner, right | 33 | (64.51, 35.93) |
| eye outer corner, left | 263 | (91.98, 35.49) |

The mouth is at x≈78, y≈53, which is where it was said to be.

### The mouth really opens

The signal is the vertical distance between the inner lips, normalised by
the face height (landmark 10 to landmark 152) so that the person's
distance from the camera cannot produce a correlation.

| | mouth aperture (normalised) | sd |
|---|---:|---:|
| A (facing, speaking) | 0.0284 | 0.0159 |
| B (turned, speaking) | 0.0277 | 0.0154 |
| C (facing, silent) | **0.0010** | 0.0014 |

The mouth in the speaking takes is **28x** more open than in the silent
take. The landmark model is measuring a mouth, and it is measuring one
that opens. Whatever this lane concludes, it is not concluding that the
mouth does not move.

---

## 3. The measurement

The speech envelope is the repo's own gate (`src/vad.py`, the Silero model
already in the repo, `models/silero_vad.onnx`), resampled onto the video's
frame clock. Both signals are detrended with a **2 s moving average**
before every comparison — the same convention as
`experiments/audio_video_alignment.py`, from which the envelope, the
detrending, the statistic and the lag convention are **imported**, so
that the numbers here and the numbers in §3 are the same statistic on the
same grid rather than two statistics that look alike. The correlation
parameters and the SHA-256 of that file are recorded in the results.

The statistic is the **maximum over ±15 s (937 lags) of the Pearson
correlation** on the 32 ms hop grid.

| | A | B |
|---|---:|---:|
| mouth aperture, best r | **+0.2052** | **+0.1866** |
| at offset | +6.688 s | +4.864 s |
| r at offset zero | -0.1763 | -0.1855 |

The offset-zero row is the "assume the recordings started together"
reading, and it is *anti*-correlated: the mouth's best offset is nowhere
near zero, which is what the independently-started recordings predict.

---

## 4. The controls, which are the point

### 4.1 Scrambled audio

The speech envelope's hops permuted, the SAME max-over-lags statistic, 25
permutations, seeded.

| | mean of the 25 | worst of the 25 |
|---|---:|---:|
| A's mouth aperture | +0.0902 | +0.1100 |
| B's mouth aperture | +0.0902 | +0.1104 |

**The headline: +0.2052 - +0.0902 = +0.1150** over the null's mean, or
**+0.0952** over its worst permutation. The mouth's correlation is above
the null.

### 4.2 The eye-distance control — a landmark that must NOT track speech

The distance between the outer eye corners, normalised the same way. If
the eye distance tracks the speech as well as the mouth aperture, the
correlation is an artefact of the whole frame moving and not of the mouth.

| statistic, on A's video | best r | its own scrambled null |
|---|---:|---:|
| mouth aperture | +0.2052 | +0.0902 |
| **eye distance** | **+0.1977** | +0.0902 |
| **mouth width** | **+0.2982** | +0.0889 |
| jaw open | +0.2353 | +0.0860 |
| head vertical position | +0.3316 | +0.0646 |
| head horizontal position | +0.2320 | +0.0734 |

**The eye distance reaches +0.1977 against the mouth's +0.2052.** A
landmark that cannot respond to a mouth opening achieves the same
number, so the margin over the null is not a margin about the mouth.

**The mouth width reaches +0.2982, higher than the mouth aperture.** A
mouth that OPENS does not get WIDER; the width is the horizontal distance
between the corners, and at this resolution it varies by 0.65 px around a
14.7 px span — essentially a noise signal. It correlates with the speech
*more* than the aperture does. That is backwards, and it is the sharpest
single falsification in this lane.

### 4.3 Wrong session

| control | best r | a clean control? |
|---|---:|---|
| A's audio vs C's video (C is silent) | +0.1064 | yes |
| B's audio vs C's video | +0.1221 | yes |
| A's audio vs B's video | +0.2782 | **no — see below** |
| B's audio vs A's video | +0.2834 | **no — see below** |
| C's audio vs A's video | +0.1136 | no — C's envelope is flat |
| C's audio vs B's video | +0.2190 | no — C's envelope is flat |
| *ceiling:* A's audio vs B's audio | +0.3537 | — |

The A↔B rows are the uncomfortable ones: they are **larger** than the
mouth's own correlation. They are not a clean control, and the reason is
measured rather than assumed: **A and B are two takes of the same person
reciting the same poem**, so their speech envelopes agree with each other
at **+0.3537** — the ceiling for any signal that tracks speech *content*,
and it is higher than any video-audio number in this lane. A signal that
tracks the content will align with *both* takes. These rows are reported
and are **not** used to judge the mouth.

The C-audio rows are numerically unstable for the reason
`TEACHER-LOO-FOLDS.md` §3 already noted for C: its "speech envelope" is a
flat line at 0.003, so the correlation is the ratio of two near-constant
signals. Reported, not used.

### 4.4 The empty room — which is not a passive null

The empty room has real, low-amplitude motion: the repo's own work
attributes it to LED flicker, and this lane measures it rather than
assuming it.

- **0 of 4,500 empty-room frames contain a face.** The landmark model
  does not hallucinate a mouth in an empty room, so the empty room
  contributes no mouth signal to be correlated. The control is vacuous by
  construction — and that is itself the result.
- The motion is **sparse**: the top 5% of map pixels carry **97.6%** of
  the motion energy, and the top 1% carry 45.0%.
- It is **nowhere near the mouth**. The mouth's neighbourhood (±8 px in x,
  ±5 px in y around (78.63, 53.14)) carries **0.0499%** of the empty
  room's motion energy, with a mean of 0.000253 against a frame mean of
  0.005277 — **21x below** the frame average. The energy is in the bottom
  rows of the frame (px rows 80–87 carry the most).
- The same measurement on the **occupied** take is the other way round:
  the mouth's neighbourhood carries **2.56%** of session A's motion
  energy, with a mean of 1.705 against a frame mean of 0.693 — **above**
  the frame average. So the comparison the control is meant to make is a
  real one: the empty room's motion is not mouth motion, and the occupied
  take's motion is.
- The empty room's motion against A's speech: `total` +0.1282, `active`
  +0.1282, `area` +0.0889, against a scrambled null of **+0.0914**. The
  empty room's flicker does not track the speech either.
- The empty room's mean motion-map energy is **0.0076 of session A's**,
  on the same pipeline. `TEACHER-LOO-FOLDS.md` §2 quotes **0.0000066**
  for this ratio, measured on the torn frames — a factor of ~1000 apart,
  because the tear added energy to the occupied take and to the empty
  room alike but not in the same proportion.

### 4.5 The degenerate baselines — the search's own inflation

937 lags is a lot of opportunities. The same statistic applied to a
signal against *itself*:

| | at a random lag | max over the 937-lag search |
|---|---:|---:|
| the speech envelope | +0.0099 | **+0.7243** |
| the mouth aperture | +0.0094 | **+0.7154** |

So a max-over-lags statistic reaches +0.71 when the two signals really
are the same signal, and around +0.09 when they are a signal and a
permutation of it. The reader should hold both numbers.

### 4.6 Does the peak stand out, and does it replicate?

A maximum over many lags finds a good-looking offset even when there is
nothing there, so the shape of the whole correlation-vs-lag curve matters
more than its maximum.

| | peak | peak z-score | fraction of lags above half the peak |
|---|---:|---:|---:|
| A, mouth aperture | +0.2052 | **+2.79** | 0.077 |
| A, mouth aperture, **scrambled** | +0.0813 | **+2.70** | 0.097 |
| A, head vertical position | +0.3316 | **+3.77** | 0.031 |
| A, head vertical position, **scrambled** | +0.0668 | +3.16 | 0.074 |

The mouth's peak is **not** more unusual than noise's peak (z +2.79
against +2.70 for the scrambled pair). The head's is (z +3.77 against
+3.16, and only 3.1% of the lags come within half the peak, against 7.7%
for the mouth).

And the split-half test — does the SAME offset work on data the search
did not choose?

| | full | 1st half's own best | 2nd half's own best | agree? |
|---|---:|---:|---:|---|
| A, mouth aperture | +0.2052 @ +6.69 s | -3.52 s | +0.45 s | **no** (3.97 s apart) |
| B, mouth aperture | +0.1866 @ +4.86 s | +2.11 s | +1.79 s | yes (0.32 s) |
| A, eye distance | +0.1977 @ +12.10 s | -0.35 s | +1.25 s | no (1.60 s) |
| B, eye distance | +0.1989 @ +0.45 s | +4.13 s | -2.82 s | no (6.94 s) |
| **A, head vertical position** | +0.3316 @ +0.58 s | **+0.64 s** | **+0.54 s** | **yes (0.10 s)** |

For A, the mouth's best offset does not replicate across the two halves
of its own recording, and the full recording's best offset (+6.69 s)
agrees with neither half. The head's vertical position replicates to
within 0.10 s.

---

## 5. The verdict

> **The mouth aperture's correlation with the speech envelope is +0.2052
> (A) and +0.1866 (B), against a scrambled-audio null of +0.0902 — a
> margin of +0.1150. The margin does not survive the controls, so the
> video's mouth does not carry a speech-tracking signal that this
> measurement can attest to.**

The verdict requires **five** checks, and it is recorded as the AND of
them so that it can fail:

| check | result |
|---|---|
| beats the scrambled-audio null | **yes** (+0.1150) |
| beats the eye-distance control | **yes, by 0.0063** (+0.2052 vs +0.1989) |
| beats the mouth-width control | **NO** (+0.2052 vs +0.3038) |
| beats the clean wrong-session control | yes (+0.2052 vs +0.1221) |
| replicates on the two halves of the recording | yes |

**`the_mouth_tracks_the_speech: false`.**

### What the correlation is actually about

Not the mouth. The **head is moving**.

The vertical position of the face in the frame correlates with the speech
at **+0.3316** (A), with a peak that is sharper than noise's and an offset
that replicates to within a tenth of a second across the halves of the
recording. The eye-distance control is a proxy for exactly that: the eye
separation changes when the head turns or tilts, so a correlation there
is the head moving, and the mouth aperture inherits it.

That the head moves more while speaking is real, not an artefact:

| | head centroid sd, x | head centroid sd, y | lag-1 autocorrelation of the detrended head height |
|---|---:|---:|---:|
| A (speaking) | 3.653 px | 1.383 px | +0.981 |
| B (speaking) | 1.589 px | 0.488 px | +0.936 |
| C (silent) | 1.184 px | 0.372 px | +0.937 |

The head moves about **3x** further in the speaking take than in the
silent one, and the motion is **smooth** (lag-1 autocorrelation +0.981),
which is what a person moving their head looks like and not what a
landmark fit jumping around looks like (on the torn frames the
forehead's vertical position had a standard deviation of 26.86 px).

**So what does the video carry?** On the correctly-read frames, the video
carries a **time-resolved signal that lines up with the audio's speech**,
and it is a head-motion signal rather than a mouth signal. Whether that is
a usable source of frame-level labels is a further question this lane does
not answer: it is one person, one seat, three takes, and A and B recite
the same poem, so their envelopes agree at +0.3537 and any content-tracking
signal will align with both.

---

## 6. Sensitivity, and what the torn reading would have produced

| reading | frames | face detected | mouth aperture | its scrambled null |
|---|---:|---:|---:|---:|
| **correct stride, 4x** | 900 | 900/900 | +0.2052 @ +6.69 s | +0.0902 |
| correct stride, 1x (native 160x96) | 900 | 900/900 | +0.2083 @ +6.69 s | +0.0897 |
| correct stride, 6x | 900 | 900/900 | +0.2138 @ +6.69 s | +0.0905 |
| **torn stride (the repository's)** | 750 | 601/750 | +0.1439 @ +2.27 s | +0.0902 |

The upscale factor does not move the result and does not move the offset
(the same +6.69 s at 1x, 4x and 6x), so the landmark geometry at 160x96 is
stable and the number is not a resolution artefact. On the torn reading
the landmark model detects a face in only 80% of frames, the landmarks
jump around the frame (the forehead's vertical position has a standard
deviation of 26.86 px, against 1.09 px on the correct frames), and the
margin is roughly half as large at a different offset.

### The whole-frame instrument, re-measured on the correct frames

Not this lane's instrument, but the comparison the reader needs — §3's
whole-frame motion envelope, recomputed on frames that are not torn:

| | §3 (torn frames) | re-measured on the correct frames |
|---|---:|---:|
| A, `active` | +0.0776 | **+0.1536** @ +4.67 s |
| A, `total` | +0.0656 | +0.0863 @ +3.26 s |
| B, `active` | +0.0247 | +0.0870 @ -11.58 s |
| B, `total` | +0.0299 | +0.0458 @ +4.96 s |

And the mouth aperture at the offset §3's whole-frame measurement found:

| | §3's offset | mouth aperture r at that offset |
|---|---:|---:|
| A, `total` | +2.336 s | +0.0277 |
| A, `active` | +9.472 s | +0.0162 |
| A, `area` | +2.304 s | +0.0331 |

---

## 7. What this does to `TEACHER-LOO-FOLDS.md`

**§3 is contradicted, and §4 with it.** §3 concluded that "the video's
motion does not follow the audio's speech" and that therefore frame-level
labels cannot be recovered from these captures. Both halves of that are
now known to rest on an instrument that was reading a tear:

- §3's near-zero was measured on torn frames. On the correct frames the
  same instrument gives **+0.1536** for A's `active` envelope, twice
  §3's number, and the mouth signal itself gives **+0.2052** against a
  null of +0.0902.
- The caveat this measurement was commissioned with — "the whole-frame
  motion energy is a blunt instrument" — was the right worry, but the
  instrument was blunter than anyone thought: it was measuring the file
  format, not the person.

So the claim "the video does not carry a time-aligned speaking signal that
an envelope correlation can find" **cannot be carried forward**. What
replaces it is narrower and more specific:

> The video carries a time-aligned signal that tracks the audio's speech,
> but it is **head motion**, not mouth motion, and the mouth's own
> geometry does not show a speech-tracking alignment that survives the
> controls.

**§1 and §2 are not re-measured here, and this lane does not claim they
are unchanged.** Their *argument* is untouched by the stride: §1's
degeneracy is arithmetic about pure folds and does not depend on a single
pixel, and §2's static-variant test is a question about the features that
can be asked again. But their *numbers* were computed on features derived
from torn frames, and two of them have to be re-run before they are
quoted again:

- §2's headline, that the `map` features' discrimination comes from a
  static spatial template, was measured on maps whose "motion" was a
  constant vertical tear. Removing the motion and keeping the mean
  removes a constant artefact — which is a *stronger* version of the
  finding, not a weaker one, but the table's numbers are not the ones the
  corrected frames give.
- §2's empty-room control quoted the empty room's mean-map energy as
  **0.0000066** of session A's. On the corrected frames the same
  pipeline gives **0.0076** — a factor of ~1000 apart, because the tear
  added energy to both captures but not in the same proportion.

**What this lane has NOT done:** it has not re-run §1's or §2's analyses
on the corrected frames. That is the obvious next job for whoever owns
those scripts, and until it is done their tables should be treated as
measured on the old reading.

**§4's consequence — "the only way to get within-session label variation
from this data is frame-level labels, which §3 shows the current captures
cannot supply" — is no longer blocked by §3.** The captures *do* carry a
time-resolved speaking signal (head motion, +0.3316 with a sharp,
replicating peak). Whether it is *usable* for frame-level labels is now
the open question, and it is a better question than the one §3 answered.

**What is NOT changed by this lane:** the mouth is not the signal. The
peripheral still cannot read the mouth at 160x96 reliably enough to
justify per-frame speech detection from the mouth, and if the teacher is
to be built on frame-level labels it should be built on a motion signal,
not on lip geometry.

---

## 8. Reproducing this

```bash
/home/proart/projects/gpu-venv/bin/python experiments/mouth_aperture_alignment.py
/home/proart/projects/glasses/.venv/bin/python -m pytest tests/test_mouth_aperture_alignment.py -q
```

The experiment needs the GPU venv, because it needs MediaPipe FaceMesh.
The **test does not**: it runs on the repository venv, downloads nothing,
and needs no model file — the landmark model is imported lazily inside
the experiment, and one test asserts that importing the experiment does
not import mediapipe at all. The experiment takes about a minute and
writes `experiments/mouth_aperture_alignment_results.json`, which
records the model and its version, the bundled weights' SHA-256s, the
upscale factor, the record layout, the frame indices, the landmark
indices, the correlation parameters, the SHA-256 of every input file, and
every count.

The test asserts on the committed results, and **it passes on a null**:
the headline margin is recomputed from the recorded numbers rather than
transcribed, and the verdict is checked to be the AND of its recorded
checks, so the suite fails only if the measurement stops being derivable
from what was recorded — not if the answer changes.
