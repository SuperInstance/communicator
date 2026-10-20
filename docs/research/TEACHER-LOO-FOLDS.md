# TEACHER-LOO-FOLDS — the teacher's leave-one-session-out split cannot tell a detector from a constant

**2026-10-19.** This is an audit of the evaluation behind
`docs/research/TEACHER-PIPELINE.md`, not of the teacher itself. It has three
parts, and each one is a measurement with a control:

1. the leave-one-session-out metric is **degenerate**;
2. the `map` feature set's headline number **needs no motion at all**;
3. the video's motion **does not track the audio's speech**, so the teacher
   cannot be rescued by frame-level labels derived from the recordings.

The scripts are `experiments/teacher_loo_control.py` (the controls and the
identifiability analysis) and `experiments/teacher_static_control.py` (the
static variant and the scale control). Their results are
`experiments/teacher_loo_control_results.json` and
`experiments/teacher_static_control_results.json`.

The data is the same three seeded takes the teacher has always used:
`/mnt/c/proart_capture/corpus/` — A (facing, speaking), B (turned, speaking),
C (facing, silent), 600 frames each at 30 fps, motion maps 48×80, from
160×96 greyscale video.

---

## 1. The metric is degenerate: a constant predictor scores 1.000

The teacher labels whole SESSIONS. A is "speaking", B is "speaking", C is
"silent". So **every window of a held-out session carries the same label**,
and the leave-one-session-out accuracy of a held-out session can only be
near 0.000 or near 1.000. It is a one-bit statistic.

Measured, with the controls run on the SAME fold (`teacher_loo_control.py`):

| feature set | fold A: real | fold A: **blank** | fold B: real | fold B: **blank** |
|---|---:|---:|---:|---:|
| summary | 0.000 | **1.000** | 0.000 | **1.000** |
| map | 0.000 | **1.000** | **1.000** | **1.000** |
| both | 0.000 | **1.000** | 0.000 | **1.000** |
| region | 0.000 | **1.000** | 0.000 | **1.000** |
| region+map | 0.000 | **1.000** | 0.200 | **1.000** |

**The blank control — the model applied to constant input, which predicts one
class for everything — scores 1.000 on every fold of every feature set.**
Because the held-out session is pure, a constant predictor that happens to
answer "speaking" is exactly right.

So the `map` feature set's **1.000 on held-out B is worth exactly what a
constant predictor's 1.000 is worth: nothing.** The LOO rows cannot tell the
two apart. This is not a criticism of the feature set; it is a statement
that **the metric has no room to measure anything**.

The repository already knows the half of this that is about the *split*: the
`leave_one_session_out` function reports a fold as `skipped` when holding out
a session leaves one class in the training set, precisely so a table cell
cannot be transcribed for a fold that does not exist. The same reasoning
applies one step further, to the held-out set, and it had not been applied.

### And the LOO fold has no control of its own

`experiments/teacher_conventional.py` computes two controls — a spatially
scrambled input and a blank input — for the **temporal** split. The
leave-one-session-out fold is computed on the real features with **no
control beside it**.

That matters, because the report used one number to dismiss another:

> "`map` does score 1.000 on held-out B, but its scrambled control is 0.938,
> so it is not beating its own control either."

**The 0.938 is the TEMPORAL split's scrambled control.** The LOO fold's own
scrambled control, measured with the same fold, is **0.000**:

| fold | real | spatially scrambled (same fold) | blank |
|---|---:|---:|---:|
| A (unidentified) | 0.000 | 0.000 | 1.000 |
| B (identified) | **1.000** | **0.000** | 1.000 |

So the two numbers were compared across two different splits. Against its
own control, the LOO fold's margin is **+1.000**, not −0.062.

**Both readings are wrong, in opposite directions, and the honest summary is
the degenerate one: on a pure fold, a control that answers constantly
scores 1.000, so no margin computed on this fold is interpretable.** The
scrambled control is the informative one here (0.000) precisely because it
is NOT constant — it carries the session's energy statistics and therefore
answers differently per session.

---

## 2. The `map` feature set does not need the motion

`TEACHER-PIPELINE.md` argues that the map features are the way to express
"the mouth moved":

> "mode `map` — the window's MEAN motion map, downsampled to a coarse grid,
> concatenated with the temporal summary. This puts the spatial pattern of the
> change directly in the feature vector, which is the only way a linear model
> can express 'the mouth moved'."

The `map` feature set is the mean motion map over the window (48 numbers, a
6×8 grid) plus `vel` (5 numbers: the temporal derivative of the per-frame
summary). So the map is a **static** quantity by construction, and averaging
over six frames also suppresses the per-frame noise.

**The test: replace every motion frame with the session's temporal mean map.**
There is then no motion, no order and no change anywhere in the input — only
a picture of the session. If the teacher still separates the sessions, the
motion was not what it was reading.

| feature set | real A | real B | **static A** | **static B** | time-shuffled A | time-shuffled B |
|---|---:|---:|---:|---:|---:|---:|
| summary | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.400 |
| map | 0.000 | **1.000** | **1.000** | **1.000** | 0.000 | 0.725 |
| both | 0.000 | 0.000 | **1.000** | **1.000** | 0.000 | 0.300 |
| region | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| region+map | 0.000 | 0.200 | **1.000** | **1.000** | 0.000 | 0.275 |

**The three feature sets that contain the 6×8 grid — `map`, `both` and
`region+map` — all score 1.000 on BOTH folds with the motion removed.** The
two that do not contain it (`summary`, `region`) do not.

And the static variant is *better* than the real maps on fold A (1.000
against 0.000): **adding the temporal information to the static template
makes the classifier worse.** The temporal features are not carrying the
discrimination; they are adding noise to it.

So the discriminative feature is a **static spatial template of the session**,
and the motion is at best irrelevant and at worst misleading.

### A control on the model, not the data: the scale

`ascii_motion_render.load_session` returns `maps * scale`, where `scale` is a
per-session display constant (143.27 for A, 140.05 for C) that recovers the
original count scale. The logistic regression is **not scale-invariant** — its
regularisation is `l2 * w / n` against a fixed learning rate, so multiplying
a feature by 143 changes that feature's effective step size by 143.

The `unscaled` variant divides every session by the same constant, so the
between-session ratios are untouched and only the absolute scale moves:

| feature set | real A | real B | unscaled A | unscaled B |
|---|---:|---:|---:|---:|
| summary | 0.000 | 0.000 | 0.000 | 0.000 |
| map | 0.000 | 1.000 | 0.000 | 1.000 |
| both | 0.000 | 0.000 | 0.000 | 0.000 |
| region+map | 0.000 | 0.200 | 0.000 | 0.200 |

**The results do not move.** The teacher's numbers are not a scale artefact.
That is a clean null, and it is recorded here because a null that could have
been a finding is worth having.

### What the static template actually is

| quantity | value |
|---|---|
| correlation between the speaking and the silent take's average motion map | **0.904** |
| amplitude ratio (speaking / silent) | **0.960** |
| fraction of the pixel-wise difference explained by one global scale factor | **R² = 0.817** |
| effect size of A vs C on the total-motion feature | **0.59σ** (within-session) |
| pixelwise temporal std / temporal mean of the motion maps | 0.64 (A), 0.61 (C) |
| frame-to-frame correlation of consecutive motion maps | +0.365 (A), +0.346 (C) |
| mean-map energy of the EMPTY ROOM capture, relative to session A | **0.0000066** |

The two takes' average motion pictures are nearly the same picture, differing
by about 4% in amplitude, and 82% of that difference is one global scale
factor. The residual is the scene's fixed vertical structure at a different
amplitude — not a face, not a mouth.

The empty room settles what the picture is made of: with the same
row-removal and median filters, an empty room's average motion map has
**0.0000066** of session A's energy. So the picture is the person and the
scene, and it is present in both takes; what differs between them is how
much of it moves, by 4%.

---

## 3. The video does not track the audio

If the motion maps' average picture is "how much the scene moved, on
average", the question becomes: **does the motion follow the speech in
time?** The teacher cannot be rescued by frame-level labels unless it can.

`experiments/audio_video_alignment.py` measures it. The speech envelope comes
from the repo's own gate (`src/vad.py`, the Silero model already in the
repo), and it is a good one on this audio:

| capture | audio duration | mean speech probability | fraction of hops ≥ 0.5 |
|---|---:|---:|---:|
| A (speaking) | 30.0 s | 0.695 | **0.706** |
| B (speaking) | 30.0 s | 0.657 | **0.662** |
| C (silent) | 30.0 s | 0.002 | **0.000** |
| empty1 (room tone) | 149.9 s | 0.003 | **0.000** |

**The audio and the video of A/B/C are SEPARATE recordings** — the capture
script's own docstring says so — so their relative offset is unknown and has
to be searched for. The search is over ±15 s, which covers the whole
possibility (the video is 20 s and the audio is 30 s).

The comparison is between the speech envelope and three motion envelopes:
`total` (all pixels), `active` (the top 5% of pixels, where
`ROW-BAND-REMOVAL.md` measured 79% of the speaking/silent difference to
live) and `area`. Both envelopes are **detrended** before the comparison,
because the first version of this script did not detrend and produced a
+0.26 correlation for a *wrong-session control* — the largest number in the
run. Two drifting signals correlate whether or not they have anything to do
with each other, and the corpus's own metadata records a drift of 32 counts
after row removal.

| | A | B | C (silent) | scrambled audio | A's audio vs C's video |
|---|---:|---:|---:|---:|---:|
| `active`, detrended, best r | +0.0776 | +0.0247 | +0.1130 | +0.0608 | +0.0677 |
| `total`, detrended, best r | +0.0656 | +0.0299 | +0.1032 | +0.0554 | +0.0547 |

**The speaking sessions correlate with their own speech at +0.078 and +0.025,
against a scrambled-audio null of +0.061 and a wrong-session control of
+0.068.** There is no alignment above the nulls.

(The C row is not evidence of anything: C's "speech envelope" is a flat line
at 0.002 with a maximum of 0.099, so its correlation is the ratio of two
near-constant signals and is numerically unstable. It is reported because
leaving it out would look like cherry-picking, and because it shows how
large a number a pure artefact can produce.)

**So the video's motion does not follow the audio's speech.** Not weakly —
not at all above the nulls. And that is consistent with §2: the thing the
teacher is reading is a session-level average, not a time-resolved one.

---

## 4. What this means, and what it does not

**What it means.**

- The teacher's headline LOO number cannot be compared with anything,
  because the metric's best constant baseline is 1.000.
- The `map` features' discrimination does not come from motion.
- The video does not carry a time-aligned speaking signal that an envelope
  correlation can find.

**What it does not mean.**

- It does NOT say the video contains no information about speech. It says
  the *session-level* label and the *motion maps* do not have a measurable
  time-aligned relationship, and that the features in use do not use the
  temporal information.
- It does NOT say the teacher is useless. It says the teacher is currently a
  **session-identity detector**, and the evaluation cannot tell it from a
  constant.
- It is one person, one seat, three takes, 160×96. A different capture
  could behave differently; the measurement is about this corpus.

**The consequence for the recording plan, which is the part that costs
money.** `experiments/teacher_data_requirement.py` asks for **63 sessions
per class**. That number comes from a precision requirement (estimate the
accuracy to ±0.37% at 95% confidence, z = 2, s = 0.01022, d = 0.00366) —
and it is about the precision of an estimate, not about whether the claim is
identifiable. **Recording 63 sessions per class would not change any of the
three findings above.** The fold would still be pure, the constant
predictor would still score 1.000, and the static template would still be
what the grid feature is reading.

What would change it is a **different task**:

1. **A task with within-session label variation**, so the held-out set is
   not pure and a constant predictor scores the base rate. The only way to
   get that from this data is frame-level labels — which §3 shows the
   current captures cannot supply, because the motion does not follow the
   speech.
2. **A stimulus that contains the motion.** The repository's own cloud
   vision bench asks a vision model "is the mouth moving" from a **single
   160×96 still**, which cannot show movement; the strip-of-frames
   experiment (`docs/research/VISION-MOTION-TEACHER.md`) is the direct test
   of whether giving the model the frames in order helps.
3. **Condition D** (turned AND silent) still matters — but for the POSE
   question, not this one. It makes the pose fold identifiable (the training
   set would contain a turned-and-silent session, so pose would no longer
   determine the label). It does not add within-session variation, and it
   does not make the motion follow the speech.

**So the honest position on the teacher's data requirement is: the number
63 is not the blocker, and the blocker is not a number of sessions.** It is
that the task as constructed cannot be measured, and the fix is a change of
task, not a change of sample size.

---

## Reproducing this

```bash
/home/proart/projects/glasses/.venv/bin/python experiments/teacher_loo_control.py
/home/proart/projects/glasses/.venv/bin/python experiments/teacher_static_control.py
/home/proart/projects/glasses/.venv/bin/python experiments/audio_video_alignment.py
/home/proart/projects/glasses/.venv/bin/python -m pytest tests/test_teacher_loo_folds.py -q
```

The three scripts take about a minute each and use only numpy and the
repo's own `src/`. The audio-video script needs the repo's Silero gate
(`models/silero_vad.onnx`, already in the repository).
