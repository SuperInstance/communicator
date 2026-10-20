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

> ## CORRECTION (2026-10-20) — the corpus was read at the wrong frame stride
>
> **Every number in this file was computed on torn frames.** The captures are
> 19,200 bytes per frame (luma 160×96, then a U plane and a V plane at
> 80×24), written by `C:\proart_capture\nv12small.py`. The repo read them at
> `W*H*3//2` = 23,040 — the size of a standard NV12 frame — so every
> "frame" was rows 0..71 from one moment spliced to rows 72..95 from the
> next, and only every sixth frame was intact.
>
> The full account, with the evidence and the fix, is
> **`docs/research/NV12-STRIDE.md`**. The reader is now `src/nv12.py`, which
> **detects** the layout and reports the margin over the runner-up; the
> corrected corpus is `/mnt/c/proart_capture/corpus_v2/` — **900 frames at
> 30 fps = 30.0 s**, not 600 frames = 20.0 s.
>
> **What survives, what does not, and what is now known that was not:**
>
> | claim | verdict |
> |---|---|
> | "the blank control scores 1.000 on every fold of every feature set" | **SURVIVES.** Still 1.000 on every fold. |
> | "the `map` features need no motion" | **SURVIVES**, and strengthens: the static variant still scores 1.000 on both folds for `map`, `both` and `region+map`. |
> | "the video's motion does not follow the audio's speech" | **SURVIVES**, with a stronger control (see §3). |
> | "the LOO accuracy can only be near 0.000 or near 1.000 — it is a one-bit statistic" | **REFUTED.** On the corrected corpus the accuracies are gradated: 0.150, 0.375, 0.950, 0.100, 0.275. The fold has room. The blank control's 1.000 is a statement about the *labels* being session-level, not about the metric being binary. |
> | §1's and §2's tables | **WITHDRAWN.** Corrected tables are in §1 and §2 below. |
> | §3's correlation figures (+0.0776, +0.0247, +0.1130) | **WITHDRAWN.** They were measured on torn frames. The corrected figures are roughly twice as large — and the conclusion is unchanged, because the controls got stronger too. |
> | the teacher's `summary` margin of **+0.208** and `region` margin of **+0.312**, both marked USABLE | **WITHDRAWN.** Both were false positives created by the tear. On the corrected corpus the margins are **+0.000** and **+0.062** — NOT USABLE, like every other feature set. |
>
> The old corpus has been left on disk, untouched, so the comparison can be
> re-run: `experiments/teacher_conventional.py --corpus
> /mnt/c/proart_capture/corpus` reproduces the withdrawn numbers exactly.
>
> **The lesson is at the end of `docs/research/NV12-STRIDE.md`, and it
> is not about statistics.** A number that is
> derived from the assumption it is supposed to test is not a check. The
> frame count was "verified" against a frame count computed from the same
> wrong stride.

---

## 1. The metric is degenerate: a constant predictor scores 1.000

The teacher labels whole SESSIONS. A is "speaking", B is "speaking", C is
"silent". So **every window of a held-out session carries the same label**,
and a constant predictor — one that answers "speaking" for everything — is
exactly right on a held-out speaking session.

Measured, with the controls run on the SAME fold (`teacher_loo_control.py`),
on the **corrected** corpus:

| feature set | fold A: real | fold A: scrambled | fold A: **blank** | fold B: real | fold B: scrambled | fold B: **blank** |
|---|---:|---:|---:|---:|---:|---:|
| summary | 1.000 | 1.000 | **1.000** | 0.000 | 0.000 | **1.000** |
| map | 0.150 | 0.450 | **1.000** | 0.375 | 0.100 | **1.000** |
| both | 0.950 | 0.975 | **1.000** | 0.100 | 0.000 | **1.000** |
| region | 1.000 | 1.000 | **1.000** | 0.000 | 0.000 | **1.000** |
| region+map | 0.150 | 0.450 | **1.000** | 0.275 | 0.025 | **1.000** |

> **CORRECTION (2026-10-20).** This table was originally

> | feature set | fold A: real | fold A: **blank** | fold B: real | fold B: **blank** |
> |---|---:|---:|---:|---:|
> | summary | 0.000 | **1.000** | 0.000 | **1.000** |
> | map | 0.000 | **1.000** | **1.000** | **1.000** |
> | both | 0.000 | **1.000** | 0.000 | **1.000** |
> | region | 0.000 | **1.000** | 0.000 | **1.000** |
> | region+map | 0.000 | **1.000** | 0.200 | **1.000** |
>
> measured on the torn corpus. The **blank column is unchanged** — 1.000 on
> every fold of every feature set, which is the finding. The real column
> changed, and one thing that was believed about it is now false.

**The blank control — the model applied to constant input, which predicts one
class for everything — scores 1.000 on every fold of every feature set.**
Because the held-out session is pure, a constant predictor that happens to
answer "speaking" is exactly right.

So the `map` feature set's **1.000 on held-out B is worth exactly what a
constant predictor's 1.000 is worth: nothing.** The LOO rows cannot tell the
two apart. This is not a criticism of the feature set; it is a statement
that **the metric has no room to measure anything**.

> **CORRECTION (2026-10-20) — "it is a one-bit statistic" is REFUTED.**
> The text above used to say the leave-one-session-out accuracy "can only be
> near 0.000 or near 1.000. It is a one-bit statistic." On the corrected
> corpus that is **false**: the real accuracies are **0.150, 0.375, 0.950,
> 0.100 and 0.275** — gradated, with room to move.
>
> The argument for the correction is worth keeping, because the distinction
> matters: **the blank control's 1.000 is a property of the LABELS, not of
> the metric.** The labels are session-level, so a constant predictor is
> perfect on a pure held-out session; that much is unchanged and still
> fatal. But the metric itself can discriminate, and it does. What the
> teacher's real features score against that 1.000 is the number that
> matters — and on every fold but `summary`/region` on fold A, they score
> **below** it.
>
> The torn corpus made the model all-or-nothing. It looked like the metric
> was binary; it was the *inputs* that were.

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
| summary | 1.000 | 0.000 | 1.000 | 0.000 | 0.100 | 0.000 |
| map | 0.150 | 0.375 | **1.000** | **1.000** | 0.350 | 0.025 |
| both | 0.950 | 0.100 | **1.000** | 0.000 | 0.100 | 0.000 |
| region | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| region+map | 0.150 | 0.275 | **1.000** | **1.000** | 0.775 | 0.000 |

> **CORRECTION (2026-10-20).** This table was originally

> | feature set | real A | real B | **static A** | **static B** | time-shuffled A | time-shuffled B |
> |---|---:|---:|---:|---:|---:|---:|
> | summary | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.400 |
> | map | 0.000 | **1.000** | **1.000** | **1.000** | 0.000 | 0.725 |
> | both | 0.000 | 0.000 | **1.000** | **1.000** | 0.000 | 0.300 |
> | region | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
> | region+map | 0.000 | 0.200 | **1.000** | **1.000** | 0.000 | 0.275 |
>
> measured on the torn corpus. **The finding survives, and it is sharper.**
> The three feature sets that contain the 6×8 grid — `map`, `both` and
> `region+map` — still score 1.000 on BOTH folds with the motion removed,
> while the real features score 0.150/0.375, 0.950/0.100 and 0.150/0.275.

**The three feature sets that contain the 6×8 grid — `map`, `both` and
`region+map` — all score 1.000 on BOTH folds with the motion removed.** The
two that do not contain it (`summary`, `region`) do not.

And the static variant is *better* than the real maps on fold A (1.000
against 0.000): **adding the temporal information to the static template
makes the classifier worse.** The temporal features are not carrying the
discrimination; they are adding noise to it.

So the discriminative feature is a **static spatial template of the session**,
and the motion is at best irrelevant and at worst misleading.

> **CORRECTION (2026-10-20).** On the corrected corpus the static variant is
> still better on fold A, and by a much larger margin: for `map`,
> **1.000 against 0.150**. The same words hold, and they now mean more,
> because the real features are no longer the all-or-nothing 0.000 they were
> on torn frames.
>
> Note what the static variant's 1.000 **is**, given §1: it is the same
> degenerate score the blank control gets. The static template is a picture
> of the session, and a picture of the session is enough to classify a pure
> held-out session perfectly. So "the features need no motion" and "the
> metric cannot tell a detector from a constant" are two views of one
> failure, not two independent findings.

### A control on the model, not the data: the scale

`ascii_motion_render.load_session` returns `maps * scale`, where `scale` is a
per-session display constant (143.27 for A, 140.05 for C on the torn
corpus; **8.00 for A and 2.50 for C** on the corrected one) that recovers the
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

> **CORRECTION (2026-10-20).** The null **survives**, and it is now exact
> rather than approximate: on the corrected corpus the `unscaled` variant
> reproduces the `real` variant to the digit on every fold of every feature
> set (`map`: real A 0.150 / B 0.375, unscaled A 0.150 / B 0.375).
>
> One defect was found and fixed on the way. The divisor was a **hard-coded
> constant, 143.2708282470703** — session A's scale in the *torn* corpus. On
> the corrected corpus A's scale is **8.00**, so the "scale" control had
> been dividing by a number that no longer meant anything: the control was
> testing a scale 17.9× the one it claimed to be undoing. The divisor is now
> **read from the corpus** (`experiments/teacher_static_control.py`), and
> the fallback is named for what it is rather than being a bare constant.
>
> A stale constant is the same failure as the wrong stride, one level down:
> a number that was correct once, copied forward into a world where it was
> not, and never re-derived.

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

> **CORRECTION (2026-10-20) — this table is WITHDRAWN, and the conclusion
> with it.**
>
> Every figure above was computed on the torn corpus. Re-measured on the
> corrected corpus, by `experiments/teacher_static_control.py` (the
> `static_template` block of its results file):
>
> | quantity | torn corpus | **corrected corpus** |
> |---|---:|---:|
> | correlation between the two takes' average motion map | 0.9037 | **0.7046** |
> | amplitude ratio, raw motion (A / C) | 1.4011 | **1.4011** |
> | amplitude ratio, stored (quantised) maps | 0.9380 | **0.4202** |
> | fraction explained by one global scale factor | R² = 0.8112 | **R² = 0.4724** |
> | pixelwise temporal std / temporal mean | 0.620 (A), 0.585 (C) | **1.156 (A), 0.957 (C)** |
> | frame-to-frame correlation of consecutive maps | +0.3571 (A), +0.3299 (C) | **+0.5987 (A), +0.3973 (C)** |
> | empty room's mean-map energy relative to A | 0.0000935 | **0.0002549** |
>
> **The conclusion does not survive.** The old text said the two takes'
> average motion pictures were "nearly the same picture, differing by about
> 4% in amplitude, and 82% of that difference is one global scale factor.
> The residual is the scene's fixed vertical structure at a different
> amplitude — not a face, not a mouth."
>
> On correct frames the two takes' motion pictures are **markedly less
> alike** — they correlate 0.705, not 0.904 — and only **47%** of their
> difference is one global scale factor, not 82%. And the amplitude
> difference is not 4%: on the raw motion, the speaking take moves **40%
> more** than the silent take (ratio **1.4011**). The stored maps' ratio
> reads 0.4202 because each session's maps are quantised against its own
> 99.5th percentile, which normalises the amplitude away; the raw-motion
> ratio is the scene's.
>
> So the old reading — *"the motion maps are a picture of the room, and the
> mouth is not in it"* — **is withdrawn.** The corrected frames put a
> face-shaped difference between the two takes, and a session-level
> speaking signal that is measurable in the mouth (**+0.733 sd**, against a
> cheek control at +0.477 sd; `docs/research/NV12-STRIDE.md` §5).
>
> What still stands is §2's finding, which is about the *teacher* rather
> than about the picture: the static variant still scores 1.000 on both
> folds for `map`, `both` and `region+map`. The teacher reads a session
> template. On correct frames that template now visibly contains more than
> the room — but the teacher still does not beat its own controls.

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
possibility (the video is 30 s and the audio is 30 s).

The comparison is between the speech envelope and three motion envelopes:
`total` (all pixels), `active` (the top 5% of pixels, where
`ROW-BAND-REMOVAL.md` measured 79% of the speaking/silent difference to
live) and `area`. Both envelopes are **detrended** before the comparison,
because the first version of this script did not detrend and produced a
+0.26 correlation for a *wrong-session control* — the largest number in the
run. Two drifting signals correlate whether or not they have anything to do
with each other, and the corpus's own metadata records a drift of 32 counts
after row removal.

| | A | B | C (silent) | scrambled audio | A's audio vs C's video | **blank room** |
|---|---:|---:|---:|---:|---:|---:|
| `active`, detrended, best r | +0.1536 | +0.0870 | +0.1492 | +0.1199 | +0.0954 | **+0.5451** |
| `total`, detrended, best r | +0.0863 | +0.0458 | +0.2121 | +0.0985 | +0.0393 | **+0.5451** |

> **CORRECTION (2026-10-20).** This table was originally
>
> | | A | B | C (silent) | scrambled audio | A's audio vs C's video |
> |---|---:|---:|---:|---:|---:|
> | `active`, detrended, best r | +0.0776 | +0.0247 | +0.1130 | +0.0608 | +0.0677 |
> | `total`, detrended, best r | +0.0656 | +0.0299 | +0.1032 | +0.0554 | +0.0547 |
>
> measured on torn frames. **The corrected figures are about twice as
> large** — A goes from +0.0776 to +0.1536 — so the published numbers were
> not merely imprecise, they were measuring the defect.
>
> Two things also changed that make the conclusion **stronger**, not weaker:
>
> 1. **The blank-room control now runs at all.** The committed version
>    looked for `corpus/empty1_seed.npz`, which was never built, so the
>    control was **silently skipped** and the file did not say so. It now
>    reads the empty-room capture directly, and it is the **largest number in
>    the table: +0.5451.**
> 2. **A wrong-session control now beats the real pairing**: C's audio
>    against A's video is +0.3357, against A's own +0.1536.
>
> So the empty room's motion correlates with the empty room's audio more
> strongly than Casey's speaking correlates with his speaking. That is what
> a correlation that is about drift — or about the empty room's own sparse
> residual motion, which is 103× quieter than the occupied frame and whose
> envelope is short-lag-correlated (r = 0.716 at lag 1) rather than periodic —
> looks like.

**The speaking sessions correlate with their own speech at +0.154 and +0.087,
against a scrambled-audio null of +0.120 and a wrong-session control of
+0.095 — and both controls are beaten by the blank room.** There is no
alignment above the nulls.

(The C row is not evidence of anything: C's "speech envelope" is a flat line
at 0.002 with a maximum of 0.099, so its correlation is the ratio of two
near-constant signals and is numerically unstable. It is reported because
leaving it out would look like cherry-picking, and because it shows how
large a number a pure artefact can produce.)

**So the video's motion does not follow the audio's speech.** Not weakly —
not at all above the nulls. And that is consistent with §2: the thing the
teacher is reading is a session-level average, not a time-resolved one.

> **CORRECTION (2026-10-20) — one more control, from an independent lane.**
> `docs/research/MOUTH-APERTURE-ALIGNMENT.md` measured the same question
> with MediaPipe face landmarks on the **corrected** frames, which is a
> sharper instrument than a motion map: it can ask whether the MOUTH opens.
> Its result is the same verdict with more detail:
>
> * mouth aperture vs speech: **+0.2052**, against a scrambled-audio null of
>   **+0.0902** — so it *does* beat the null;
> * but mouth **width** scores **+0.3038** (higher than the aperture; a
>   mouth that opens does not get wider) and head **height** scores
>   **+0.3316** with a sharper peak (z +3.77 vs +2.70);
> * the wrong-session control (A's audio vs C's video) is +0.1221, below
>   the mouth.
>
> The correlation is about **head motion**, not about speech. And with the
> stride corrected, the session-level signal is real and measurable: the
> mouth region moves **+0.733 sd** more when Casey speaks than when he is
> silent, against a cheek control at +0.477 sd, with a permutation z of
> **+10.9** and the empty room 6,600× quieter (`docs/research/NV12-STRIDE.md`
> §5). **There is a speaking signal in the video. It is a session-level
> signal, and this data cannot give it a time axis.**

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
