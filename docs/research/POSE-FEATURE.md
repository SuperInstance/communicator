# POSE — one statistic separates facing from turned, and it was the statistic that was blind

**Code: `experiments/pose_feature_search.py`. Measured 2026-10-19.**

---

## 1. The question, and the two things it could have meant

`experiments/pose_test.py` concluded **NO POSE SIGNAL**: A-vs-B is 9.182 and
the time-shifted control is 9.178 — indistinguishable. And its own verdict
string explained the null away:

> *"the head turn was too small to register (Casey reported the glasses
> restricting it), so a null here is the expected outcome and not evidence
> against the idea."*

That null was compatible with two very different worlds:

  1. the head turn **never happened** (or was too small), and
  2. the head turn **did happen**, and the statistic used — the magnitude of
     a frame-to-frame change map — is **blind** to head pose.

The second reading was never tested, because the explanation in (1) was
assumed rather than measured. So the question was re-asked properly: **can
ANY statistic separate facing from turned in a 160×96 frame?**

## 2. The panel, and the controls

Fourteen candidate features, each measured against its own controls. The
headline is always the **margin over the largest control**, never the raw
separation.

  - **within-A** and **within-B**: the within-condition split of each
    condition (contiguous halves).
  - **EMPTY ROOM**: the max pairwise separation over five disjoint
    750-frame blocks of `empty1.nv12`. The mandatory control — a feature
    that "separates" A from B in an empty room is reading the banding.
  - **permutation null** and a **seam-free crop scramble** (a random
    120-column crop). The cyclic pixel-roll is reported but not gated: its
    seam is a differential artefact.
  - a **synthetic positive control** runs on every invocation: 12 of the 14
    features detect a known frontal-vs-turned difference, so the panel's
    negative results are meaningful rather than a broken instrument.

All windows are in **frames**, never seconds, because the frame rate was
never recorded (`AV-SYNC.md`).

## 3. The result

```
feature                  A-vs-B     wA     wB    empty  perm95  cropnl  margin  verdict
mirror_resid_static       0.451  0.253  0.080  0.001   0.067   1.040  +0.198  no signal
active_width              0.090  0.045  0.045  0.000   0.045   0.135  +0.045  no signal
lr_motion_ratio           0.454  0.054  0.050  0.437   0.017   0.539  +0.017  no signal
mirror_cov                1.097  0.282  0.275  0.165   0.114   0.768  +0.814  SIGNAL
grad_vertical_share       0.000  0.014  0.031  0.399   0.025   0.137  -0.398  no signal
column_energy_share       0.094  0.040  0.011  0.396   0.035   0.355  -0.302  no signal
mirror_axis_offset        0.209  0.030  0.000  0.596   0.000   0.596  -0.387  no signal
(asym_static_centre, asym_motion_centre, mirror_resid_motion,
 active_sharpness, motion_centroid_x, motion_centroid_x_spread,
 profile_skew  -- all "no signal")
```

**Exactly one feature separates facing from turned: `mirror_cov`** — the
frame's covariance with its own **horizontal mirror**. Margin **+0.814** over
the largest control, ratio 3.88×.

That is the intuitive one. A face looking at the camera is roughly
left-right symmetric; a head turned 90° is not.

> **CORRECTION (2026-10-20) — the captures were read at the wrong frame
> stride, and the one "signal" is WITHDRAWN.**
>
> The captures are 19,200 bytes per frame (luma 160×96 + interleaved U,V
> chroma at 80×24), not the 23,040 of a standard NV12 frame. Read at
> 23,040, every "frame" is a tear between two real frames, and the row
> statistics that every one of these features is built from are then
> bimodal. The full account is `docs/research/NV12-STRIDE.md`.
>
> Re-measured on the corrected frames
> (`experiments/pose_feature_search_results.json`):
>
> | | before (torn) | **after (corrected)** |
> |---|---:|---:|
> | best feature | `mirror_cov` | `mirror_cov` |
> | best margin | +0.814 | **+1.083** |
> | verdict | "1 feature separates facing from turned beyond their controls" | **"NO feature is a clean pose signal"** |
> | `mirror_cov`: A-vs-B / B-vs-C / A-vs-C | 1.097 / 0.851 / **0.246** | 1.097 / 0.851 / **A and C DO NOT AGREE** |
>
> **The margin got BIGGER and the result got WORSE, and that is the whole
> point.** `mirror_cov` now separates facing from turned by +1.083, more
> than before. But it also separates **A from C** — and A and C are *both
> facing*. A feature that cannot tell two facing takes apart cannot be a
> pose feature, whatever its margin against the controls.
>
> So the verdict flips from "SIGNAL" to **"NO feature is a clean pose
> signal"**, and `mirror_cov` is listed under `confounded` in the results
> file. The §4 argument below — *A and C agree with each other, so B's
> difference is a manipulation and not a clock* — **does not survive**: A
> and C do not agree any more.
>
> The reason is the same reason everything else moved. On torn frames every
> take's frames carried the same seam, so the two facing takes looked alike
> for a reason that had nothing to do with facing. On correct frames they
> do not, and the feature that was separating B was separating the **seam**.
>
> **What this means for the project.** The pose channel has no validated
> feature, and it never had one — it had a seam. The one measurement that
> survived the correction and got *stronger* is the mouth: the mouth
> region moves **+0.733 sd** more when Casey speaks than when he is silent,
> against a cheek control at +0.477 sd, with a permutation z of **+10.9**
> and the empty room 6,600× quieter. That is the camera channel's real
> signal, and it is a **speaking** signal, not a pose signal. See
> `docs/research/NV12-STRIDE.md` §5.
>
> **And condition D.** The identifiability analysis
> (`docs/research/TEACHER-LOO-FOLDS.md`) says what would fix the pose
> question: **D is turned AND silent**, which is the one condition whose
> absence lets pose ≡ speaking in a training fold. D has still never been
> recorded. It is now the highest-value missing take in the project, for
> two independent reasons.

## 4. The argument that it is pose and not a clock

This is the part that makes the result mean anything, and it is a property
of the capture order rather than of the statistic.

| | captured | mirror_cov vs C |
|---|---|---:|
| **A** facing, speaking | 19:52 — the FIRST | 0.246 |
| **B** turned, speaking | 19:54 — the MIDDLE | **0.851** |
| **C** facing, silent | 19:55 — the LAST | — |

**A and C are the two ends of the capture run, and they agree with each
other (0.246, which is *under* the control 0.282), while only the middle
capture differs (0.851).**

A monotonic drift — the light changing over the session, the camera settling,
exposure walking — would separate A from C **at least as much** as it
separates either from B. It does not. So the difference in B is a
**manipulation and not a clock.**

That is the strongest argument available from these three files, and it is
not a statistical one; it is about which capture came when.

## 5. What this does NOT establish

**These statistics see the FRAME, not the HEAD.**

In B there is extra bright, moving structure at the frame's **left edge**
(x ≈ 0–24). A change of posture, a shift in the seat, or anything else that
moved between the B capture and the facing captures would look identical. So:

  - the B condition is now **DISTINGUISHABLE** from the facing captures, but
  - it is **not yet ATTRIBUTABLE** to head pose.

And the confound cannot be resolved from the files that exist, because
**condition D (turned AND silent) has never been recorded.** D is exactly the
control that separates "the head turned" from "something else moved": if D
differs from C in `mirror_cov` by the same margin B does, it is pose; if D
looks like C, then B's difference was posture, seat or lighting.

Until D exists, the honest statement is: **the B capture contains a real,
pose-consistent difference, and the most likely explanation is still the
head turn — but "most likely" is not a measurement.**

## 6. What this overturns, and what it leaves standing

**OVERTURNED.** `pose_test.py`'s "NO POSE SIGNAL" was a property of the
**change-map magnitude statistic**, not a property of the capture. Its
explanation — "the head turn was too small to register" — was an
**assumption resting on a blind instrument**, and it is now the weaker
hypothesis: the frames differ, the difference is not a clock, and a
mirror-symmetry statistic finds it.

**STILL STANDING.** The **cloud vision model** cannot read the pose. Llama
3.2 Vision 11B says FACING for 91.8 % of the A frames, 80.0 % of the B
frames and 100 % of the C frames — separation 0.018, and the control fires
(P(facing | facing AND silent) = 1.000), so it is reading *a person*, not
pose. **A 160×96 frame at that model's resolution does not carry the
head-turn signal** — but the frame does, and a statistic does. See
`CLOUD-TEACHER.md` §5.

Also still standing: the audio carries no pose information whatsoever.
clef-omni gives P(facing|A) 0.9147 against P(facing|B) 0.9059 — margin
**0.0088**. Turning your head changes your voice by 0.9 %.

## 7. What to record next, in priority order

  1. **Condition D, turned and silent.** It is the only thing that
     separates pose from posture, and it does not exist. Three repetitions
     of D are worth more than anything else in the plan.
  2. **A real 90° turn, with the glasses off if they restrict it**, and the
     turn recorded *before and after* within the same session if possible —
     that turns the pose measure into a within-session control with no
     session-to-session confound at all.
  3. Everything else the corpus needs, at the volume the teacher needs
     (≥ 63 independent sessions per class).

The capture script now writes `timing.json` with per-frame timestamps, so
the A and B conditions of the next corpus can be compared **within one
session**, which is the design that removes the session-to-session confound
entirely.

## 8. Limits

  - One session per condition, one person, one room, one lighting state.
  - Frames from one session are **not independent samples**. The separations
    above are descriptive, not inferential.
  - A and B were captured minutes apart; their difference includes
    session-to-session change that the empty room cannot rule out.
  - The empty room is a different session in different light.
  - The spatial null uses only 8 trials, so its median is noisy.
  - A one-dimensional column profile is blind to a head that is **shifted but
    still symmetric**.
  - `mirror_cov` was found by searching a panel of 14. One feature out of 14
    passing is a multiple-comparison exposure; it needs the D condition, or a
    second session, to be believed.
