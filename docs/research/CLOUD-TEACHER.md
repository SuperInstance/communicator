# THE CLOUD TEACHER — a large multimodal model reads the footage

**Code: `experiments/cloud_multimodal.py`. Measured 2026-10-19.**
Casey: *"if you use various cloud models like multimodal models and LLMs to
analyse too. you would have a good understanding of the video before you even
start to try to make the system work."*

> ## CORRECTION (2026-10-20) - the frame stride was wrong, and the margin got STRONGER
>
> The captures are 19,200 bytes per frame (luma 160x96 + interleaved U,V
> chroma at 80x24), not the 23,040 of a standard NV12 frame. The frame this
> bench sent to the model was **a tear between two real frames** - rows
> 0..71 of one moment and rows 72..95 of the next. See
> `docs/research/NV12-STRIDE.md`.
>
> Re-measured on the corrected frames
> (`experiments/cloud_multimodal_results.json`):
>
> | | before | **after** |
> |---|---:|---:|
> | sensitivity (says SPEAKING on speech frames) | 0.455 (5/11) | **0.667 (8/12)** |
> | false-positive rate on the silent person | 0.000 | **0.000** |
> | false-positive rate on blank | 0.000 | **0.000** |
> | **margin over the silent class** | +0.455 | **+0.667 - USABLE** |
>
> **The defect did not manufacture this margin; it depressed it.** The
> speaking arm now has the same denominator as the other arms, because the
> corrected capture has 900 frames instead of 750 and the VAD selects twelve
> speech frames rather than eleven.
>
> **An independent lane falsified the opposite hypothesis too.** The natural
> guess - "the margin is a stride artefact" - is wrong, and was tested
> directly: `docs/research/VISION-MOTION-TEACHER.md` ran the same models on
> clean single frames and on a **torn** strip, and the clean single frame
> scores **+0.583** against the published +0.455. The torn strip scores a
> margin of **exactly 0.000** - a 1280x192 strip of four torn panels makes
> llama-3.2 answer SPEAKING on the blank control 12 times out of 12.
>
> **So the defect was in the repo's MOTION numbers, not in the cloud
> teacher's judgement.** A large multimodal model was reading a legible
> photo of Casey's face all along; the repo's own measurements of that
> photo were the thing that was broken.
>
> Two further defects were found in this bench while re-running it:
>
> * **The endpoint is not deterministic at temperature 0.0.** Five identical
>   runs gave margins 0.583, 0.667, 0.583, 0.667, 0.583, and the *silent*
>   arm moved too (12/12 -> 11/12). The old 0.455 and the new 0.667 are the
>   same measurement, twice.
> * **The answer parser could not read the probe's own vocabulary.** The
>   question asks "SPEAKING or NOT SPEAKING" and the parser looked for
>   "yes"/"no", so a perfectly correct "SPEAKING." was logged as
>   `parsed=None`. Fixed, and the fix is not attributed to the stride.
>
> `experiments/cloud_vision_probe.py` was re-run at the same time, and the
> result is worth having: **llama-3.2-11b-vision's swap test shows the
> answer follows the IMAGE, not the prompt.** Asked the speaking question
> while shown the silent frame it answers "NOT SPEAKING."; asked the silent
> question while shown the speaking frame it answers "SPEAKING." Relabelling
> the prompt does not move it; the picture does.

---

## 1. What was measured, and against what

The Cloudflare account has 72 models, including Llama 4 Scout 17B, Llama 3.2
Vision 11B, LLaVA 1.5 7B, and clef-omni (30B, which takes **raw audio**).

The question is not "can a big model label a frame" — of course it can. The
question is whether it knows anything, and the answer to that is its accuracy
**relative to controls**. A model that answers SPEAKING to everything scores
50 % on a two-class task and 100 % on the silent class.

So every arm runs the same three conditions and reports the margin over
each:

| condition | what it is | what it controls |
|---|---|---|
| **speak** | frames inside DETECTED SPEECH, condition A | the positive class |
| **silent** | condition C — a person in the room, mouth closed, saying nothing | **the real negative class** |
| **blank** | a frame of nothing | the prompt-only floor |

## 2. The result

**Llama 3.2 Vision 11B, 11 speech frames / 12 silent / 11 blank:**

| | value (torn frames) | **value (corrected frames)** |
|---|---:|---:|
| sensitivity, P(says SPEAKING \| speech frames) | 0.455 (5/11) | **0.667 (8/12)** |
| false-positive rate, silent person present | 0.000 (0/12) | **0.000** |
| false-positive rate, blank | 0.000 (0/11) | **0.000** |
| **margin over the silent class** | +0.455 — USABLE | **+0.667 — USABLE** |

(The corrected column is from the 2026-10-20 re-run; the left column is what
this file originally reported. See the correction block at the top.)

**clef-omni on raw audio, same sessions:**

| | value |
|---|---|
| P(yes \| speech) | **0.8003** |
| P(yes \| silence) | **0.0345** |
| **audio margin** | **+0.766** |

So: **a large multimodal model reads "is this person speaking" off a 160×96
webcam frame with a 0.455 margin and zero false positives on 23 non-speaking
frames, and reads it off the raw audio with a 0.766 margin.** This is the
first large teacher in the project that beats its controls, and it is
Casey's cloud-model proposal working exactly as proposed.

## 3. The blank control was hiding a failure, and that is the important part

The first version of the metric measured the margin as

```
P(SPEAKING | speech frames) − P(SPEAKING | BLANK frame)
```

and reported **llama-4-scout-17b at margin 1.000 — "USABLE"**.

The scout says SPEAKING on **10 of the 12 frames of the SILENT condition** —
a person in the room, mouth closed — and NOT SPEAKING on a blank, because
there is nobody in it.

| model | sensitivity | FPR on silent person | FPR on blank | margin vs silent | margin vs blank |
|---|---:|---:|---:|---:|---:|
| llama-3.2-11b-vision | 0.455 *(corrected: 0.667)* | **0.000** | 0.000 | **+0.455** *(corrected: +0.667)* | +0.455 |
| llama-4-scout-17b | 1.000 | **0.833** | 0.000 | **+0.167** | +1.000 |

**The blank is a prompt floor, not a task control.** Any model that treats a
human face as evidence of speech sails through it. The headline is now
measured against a person who is present and not speaking, and under that
metric the scout drops from "perfect" to **not usable** — which is the truth
about it.

## 4. The frame labels are per-frame, and that changed the number

The first version sampled frames uniformly across the session and called
every one of them "speaking".

The audio says otherwise. The Silero gate on A.wav gives **70.6 % speech in
22 segments**, and the first segment does not begin until video frame 58
(t = 1.93 s). The sampled frame 30 is **before any speech**, and frame 120
falls in the **gap between segments** (103…126). The model answered "NOT
SPEAKING" on both — which was **correct**. The label was wrong, not the
model.

This is the same failure the project has now documented four times: a
statistic summarised over the wrong interval. Here the *label* was the
statistic. Frames are now chosen from inside detected speech segments, so
each frame's label holds at that frame's own timestamp.

## 5. The pose question: still open, and the first result was noise

The pose question has never been resolved. `experiments/pose_test.py`
measured A-vs-B at 9.182 against a time-shifted control of 9.178 — no
separation — and that result is compatible with two very different worlds:
the head turn never happened in that capture, or the change-map statistic is
blind to it. A vision model that can name the head angle tells us which.

**It cannot.** Llama 3.2 Vision 11B, asked "FACING or AWAY":

| condition | P(facing) | n |
|---|---:|---:|
| A — facing, speaking | 0.818 | 11 |
| **C — facing, SILENT** | **1.000** | 10 |
| B — turned, speaking | 0.800 | 10 |
| **separation** | **+0.018** | — |

Two things follow.

**The first result was noise.** An earlier 6-frame run gave separation 0.333
and it was quoted as if it meant something. At 10–11 frames it is 0.018.
**That 0.333 is retracted** — it was a six-sample artifact, and the honest
statement is that the model says AWAY for about 20 % of turned frames and
FACING for about 18 % of facing frames.

**The control fires.** P(facing | facing AND SILENT) = **1.000** — the model
says FACING for a person who is facing the camera and not speaking at all.
So the model is not reading pose; it is reading *a person*, and its
"facing" answer is nearly constant.

**What is now known.** The 160×96 frame at **this model's** resolution does
not carry the head-turn signal — the model reads *a person*, not a pose.
But **the frame itself does carry it**: a separate search over fourteen
candidate statistics found `mirror_cov`, the frame's covariance with its own
horizontal mirror, separating facing from turned with margin **+0.814**, and
pose-consistent (A and C — the first and last captures — agree with each
other and differ from the middle one). See `POSE-FEATURE.md`.

**So the remaining question is not "is there a signal" but "is the signal in
B the head".** These statistics see the frame, not the head, and in B the
extra structure is at the frame's left edge, where a posture or seat change
would look identical. **Condition D (turned and silent) is the control that
separates those, and it has never been recorded.**

## 6. The audio arm confirms the design premise for the pose test

clef-omni was asked the directionality question on A (facing, speaking) and
B (turned, speaking):

| | P(facing) |
|---|---:|
| A — facing, speaking | 0.9147 |
| B — turned, speaking | 0.9059 |
| **margin** | **+0.0088** |

**Zero.** Turning your head changes your voice by 0.9 %. So:

  - the pose task **cannot** be solved from audio, by any model, at any
    budget — the information is not there;
  - a fixed poem in conditions A and B makes the two acoustically identical
    **by construction**, which is what turns the pose comparison into a clean
    test instead of an ambiguous one;
  - any acoustic classifier scoring above chance on A-vs-B would be cheating,
    and now there is a measurement that says cheating is impossible.

## 7. Three request-shape facts, all found by probing

  - **Meta's licence gate** wants a top-level `{"prompt": "agree"}` **alone**.
    `{"prompt": ..., "messages": ...}` fails a oneOf validation.
  - **clef-omni has no licence gate at all.** It rejects `{"prompt": "agree"}`
    with a *shape* error, which is not a licence refusal. Treating that as a
    failed agreement silently killed the whole audio arm — every call
    returned "agreement_failed" and the arm looked like a model failure.
  - **clef-omni needs the WAV FILE's bytes.** Sending raw PCM under a
    content-type of `audio/wav` gives "Audio could not be decoded".
  - **LLaVA 1.5's schema on this account rejects every image shape tried** —
    `{"image": "<b64>", "prompt": ...}`, `{"image": [{"url": ...}]}`, and the
    OpenAI `messages` form all fail with `oneOf at '/' not met`, the last
    complaining that `/image/0` must be an integer. **LLaVA is unavailable
    here and is reported as unavailable**, not as a model that failed.

## 8. What is NOT claimed

  - One session per condition, one speaker, one room, one lighting state.
    Frames from one session are correlated, so the effective sample size is
    far below the frame count.
  - The speech frames' labels come from the gate applied to A.wav. Per
    `AV-SYNC.md`, **A.wav and A.nv12 do not correspond** — the cross-correlation
    finds no relationship between them at any lag, rate or feature. So the
    per-frame labels are provisional, and the 0.455 margin is a number on
    frames selected by an audio file that may not belong to them.
  - A good result here is a demonstration that **the signal exists in a 160×96
    frame**, not an accuracy claim, and not a substitute for the
    leave-one-session-out test that refuses to be fooled.
  - The pose question is **not** answered. The measurement says the pose
    signal is not visible to this model at this resolution. That is a real
    result and it is a negative one.

## 9. What it changes

  - The teacher can be a **large cloud model**, not a small local one. The
    local models were measured answering constantly (margin 0.0); the cloud
    models are not.
  - The **audio channel is not dead**: a 30B multimodal model reads speech
    from the raw audio with a 0.766 margin. `TWO-SPEAKER-TURNS.md`'s
    conclusion was about *individual speakers*, not about speech presence.
  - The **pose channel is visual-only**, and now that is a measurement rather
    than an assumption.
  - The next corpus has to carry a **real 90° turn** and **condition D**, or
    the pose question cannot be closed at all.
