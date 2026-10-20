# A/V SYNC — the captures' audio and video do not correspond

**Status: a retraction of an assumption the project had never tested.**
Measured 2026-10-19. Code: `experiments/av_sync.py`.

---

## 1. How this came up

Preparing the cloud teacher (`CLOUD-TEACHER.md`), two facts about the
existing captures turned up, and they do not go together:

| file | what it says | duration |
|---|---|---:|
| `A.wav` | 479 700 samples at the header's 16 000 Hz | **29.98 s** |
| `A.nv12` | 750 frames at 30 fps | **25.00 s** |

**The two streams are not the same length.** Every statement of the form "the
audio says X and the video shows X at time t" had been assuming they were.

The project has been failing at the audio–vision join for a long time.
`TWO-SPEAKER-TURNS.md` found that at the moment of decision, END and RESUME
are byte-identical audio. The spectrogram-turn model was falsified. The
"join" has failed repeatedly, and always in the same direction: the audio
never resolves what the video cannot.

So the obvious suspicion was that the join was never there, for reasons of
analysis. The obvious alternative was that the join was there and the
analysis was wrong. Both deserve a number.

## 2. The measurement

Cross-correlate the **video motion envelope** against the **audio energy
envelope** over a range of lags.

```
video motion m[t] = mean |frame(t+1) - frame(t)| over the frame,
                    on ROW-BAND-REMOVED frames
audio energy e[t] = RMS of the audio over video frame t's time span
```

Row-band removal is applied first because the LED flicker banding
(`CAMERA-FLICKER-BANDING.md`) otherwise dominates the motion signal, and a
correlation against a drift-dominated signal measures the drift.

The lag at which the correlation peaks is the A/V offset. If a peak exists,
well above the null, the two streams show the same thing.

### The nulls

A peak is easy to produce by accident, so:

  - **SHUFFLED AUDIO** — the audio envelope's 10-frame blocks are shuffled in
    time and the whole statistic is redone, 60 draws. This is the null
    distribution of the peak under *no* relationship between sound and
    picture.
  - **OTHER SESSION** — A's video against B's audio and vice versa. Two
    different recordings of the same room in the same conditions. If a peak
    survives *that*, the measurement is picking up the room.
  - **SILENT SESSION** — C, which has no speech at all, must not produce a
    peak. It is the cleanest possible null, because nothing is there to
    correlate.

### The result

| session | peak lag | peak r | null max \|r\| | ratio | other-session control |
|---|---:|---:|---:|---:|---:|
| A (speaking) | +7167 ms | 0.0513 | 0.1452 | 0.35 | 0.0739 |
| B (turned, speaking) | −3667 ms | 0.1181 | 0.1637 | 0.72 | 0.0678 |
| C (**silent**) | +7567 ms | 0.0724 | 0.1335 | 0.54 | 0.0517 |

**No peak beats its null. In every session the peak is smaller than what
noise alone produces.** The lags are large and inconsistent in sign, and the
silent session C produces a peak too — which is exactly what noise does.

## 3. Three ways it might still have been a clock fault, all checked

The result above says "no correspondence". Before believing that, three
time-base explanations were tested, because all three were plausible and
two of them turned out to be *real*.

### 3.1 The rates do not agree under the 30 fps assumption — but that is the wrong thing to measure

If the video really is 30 fps, then 479 700 audio samples over 25.00 s of
video implies **19 188 Hz**, not the 16 000 the WAV header claims. And it is
consistent:

| capture | samples | video frames | implied rate at 30 fps |
|---|---:|---:|---:|
| A / B / C | 479 700 | 750 | 19 188 Hz |
| empty1 | 2 398 500 | 3 750 | 19 188 Hz |

That looks like a real 1.2× error in every audio timestamp the project has
computed. **It is not, and the reason matters.** The 19 188 figure is derived
from an assumption — that the camera delivers 30 fps — and that assumption is
exactly what is in doubt.

`nv12small.py` loops `int(SECS * 30)` times. If the camera delivers **25 fps**,
then 750 frames span **30 s**, and the audio is 29.98 s. And for empty1:
3 750 frames at 25 fps is 150 s, and the audio is 149.91 s. **Both captures
match exactly under 25 fps**, and neither matches under 30 fps.

So the likely story is the reverse of the first reading: **the audio header
is right and the video's frame rate is wrong.** The script asks for 30 fps,
the camera delivers 25, and the video's time base is stretched by 1.2×. That
is a real defect, and it is the same defect the new capture script fixes by
measuring the achieved rate instead of assuming it.

It is also **not the reason there is no correspondence.** At 25 fps the spans
agree to within 40 ms and the cross-correlation still finds nothing:

| session | fps 25: peak r | null max \|r\| | ratio |
|---|---:|---:|---:|
| A | 0.0765 | 0.0905 | 0.85 |
| B | 0.1047 | 0.1127 | 0.93 |
| C (**silent**) | 0.0491 | 0.2656 | 0.18 |

And the audio rate itself is not independently verifiable from the data: the
strongest narrowband lines in the recordings are at 139 Hz (A, speaking),
399 Hz (C, silent) and 244 Hz (empty1) — none of them a mains fundamental, so
there is no hum line to calibrate the rate against. **The one measurement
that would have settled it — a recorded timestamp per frame — was never
taken.**

### 3.2 The video's frame rate is wrong — and this one is probably real too

`nv12small.py` loops `int(SECS * 30)` times, so if the camera delivers 25 fps
then 750 frames span 30 s of wall clock and the two spans match exactly
(29.96 s vs 29.98 s). Tested at 24, 25, 26 and 30 fps: best ratio 0.97, and
the silent session still peaks.

### 3.3 The motion feature is drift-dominated

Retested with the motion summed over the region where A and C differ, rather
than over the whole frame — to remove the drift from the feature itself.

| session | fps 25 | fps 30 |
|---|---:|---:|
| A | r 0.0716, null 0.1298, ratio 0.55 | r 0.0439, null 0.1520, ratio 0.29 |
| B | r 0.0945, null 0.1173, ratio 0.81 | r 0.0607, null 0.2201, ratio 0.28 |
| C | r 0.0494, null 0.2444, ratio 0.20 | r 0.0232, null 0.2431, ratio 0.10 |

Still nothing above the null.

## 4. What this does and does not touch

It is worth being precise, because "the audio and video do not correspond"
could be read as "everything is wrong".

**Not touched — every video-only result stands:**

  - A-vs-C differing by 4.914 counts against a within-split of 2.878, with
    79 % of the excess in the top 5 % of pixels. Video only. (The factor is
    **1.71×**, not the 2.6× that `ROW-BAND-REMOVAL.md` printed — that one
    divides by the within-split of condition **B**, 1.917, rather than of the
    condition in the contrast. `ROW-BAND-REMOVAL.md` has since been corrected
    to 1.71× as well. See `NOTES-AUDIT.md` F1.)
  - Row-band removal: empty 50.00 % → 0.01 %, occupied 48.19 % → 48.57 %.
    Video only.
  - The drift characterisation, the gain-invariance result, the flicker
    banding analysis, the teacher's region/concentration/persistence
    features. All video only.

**Touched:**

  - Any statement that ties a sound to a moment in a frame.
  - The per-frame speech labels this session's cloud teacher was first built
    on. The label now comes from the gate and is checked against the video's
    own evidence, and the gate is applied to the SAME file the frames come
    from — but with no correspondence between the streams, that labelling is
    provisional.
  - The acoustic half of `TWO-SPEAKER-TURNS.md` and any other audio/video
    fusion claim, if it relied on these files.

## 5. Why the files are like this

`rec.py` is a standalone winmm recorder: it records a fixed number of
seconds of audio to a WAV. `nv12small.py` is a standalone camera recorder.
**They are two separate programs.** The A/B/C files were produced by running
them at different times with different lengths, and nothing recorded that
fact, so the pairing was inferred from the filenames for as long as it was
convenient to infer it.

## 6. The fix is structural, not analytical

No amount of re-analysis recovers a correspondence that was never captured.
The capture script now (`experiments/corpus_capture_win.py`):

  - takes **one video frame and one audio block in the same loop iteration**,
    with **one clock reading** per iteration;
  - records the **achieved** camera frame rate and warns if it differs from
    the assumed one;
  - writes the WAV header with the microphone's **actual** sample rate
    rather than the requested one;
  - writes a `timing.json` beside every take holding the per-frame
    timestamps, so the correspondence is **verifiable after the fact** rather
    than assumed.

## 7. The lesson, which is the same one as always

**An assumption that has never been measured is not an assumption; it is a
claim that has been silently accruing interest.** This one sat under the
project's headline capability — the join — and every failure of the join was
attributed to the analysis rather than to the data.

The instrumentation that found it was not clever. It was: take the thing
everyone believed, write down the number that would show it, and check. And
the check had to be able to come out the other way, which is why there is a
shuffled null, an other-session control, and a silent session in there. A
correlation that cannot lose is not a measurement.
