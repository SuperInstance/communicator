# NEXT — record the corpus

**One command. About 16 minutes. Everything else is blocked on it.**

```
C:\Python314\python.exe C:\proart_capture\corpus_capture_win.py --plan --out C:\proart_capture\corpus
```

The script is already in `C:\proart_capture\`, along with the `src\` package
it needs (`C:\proart_capture\src\`). Nothing to install, nothing to copy.

> ## CORRECTION (2026-10-20, overnight) — four things in this file changed, and one of them changes the reason D matters
>
> **1. THE MIRROR_COV REASON IS WITHDRAWN, AND D MATTERS FOR A STRONGER
> REASON.** This file said `mirror_cov` separates the turned capture from
> both facing captures with margin **+0.814**, and that D settles "is the
> difference the HEAD?". It does not: on the corrected frames the margin
> *rises* to +1.083 but the feature **also separates the two FACING
> conditions** (A from C, which differ only in speech), so it cannot be a
> pose signal. The verdict is now **"NO feature is a clean pose signal"**.
>
> **D is still the most valuable take, for a better reason.** The night's
> largest result is a speaking signal in the **mouth** (+0.733 sd against a
> +0.477 sd cheek control, z = +10.9), and the open question is whether it
> is really the MOUTH or just the HEAD MOVING — the MediaPipe lane found
> that head height moves more sharply (+0.3316, z +3.77) than the mouth
> aperture itself (+0.2052, z +2.70). **D (turned AND silent) is the control
> that separates "the head moved" from "the mouth opened",** and it also
> makes every pose fold identifiable. Two independent reasons, one take.
>
> **2. THE OLD CORPUS'S DRIFT FAILURE WAS THE TORN FRAMES.** This file said
> the corpus on disk has "3 takes, all failing the drift check". The
> recorded drift was 32.08 / 32.71 / 33.44 against a threshold of 5 — and
> those numbers are the TEAR, not the lighting: the same captures drift
> **0.545** mean |diff| when read at the correct stride. The corrected
> values are **0.7 / 0.6 / 0.5**, so the takes **PASS** the drift check by
> a factor of about seven. Nothing about the room's lighting was ever
> wrong.
>
> **3. THE CLOUD TEACHER'S MARGIN IS +0.667, NOT +0.455** (8 of 12, false
> positive rate 0.000). The stride defect was *depressing* it.
>
> **4. THE CAPTURE NOW ALSO STORES THE RAW LUMA AND THE UNFILTERED MAPS.**
> See "What the script prints" below. The row-removal filter the old
> version applied at capture time is **no longer justified** — the banding
> it removed was the tear — so the tool now stores `maps` (row-removed,
> as before), `maps_raw` (unfiltered) and `luma` (the raw frames), and the
> representation is a post-hoc decision rather than something frozen into
> the recording.

---

## What it will ask you for

| | what to do | how long |
|---|---|---:|
| **A1 A2 A3** | facing the camera, reciting *Invictus* from memory | 50 s each |
| **B1 B2 B3** | head turned fully away (90°, at the wall), reciting the **same** poem | 50 s each |
| **C1 C2 C3** | facing the camera, completely silent | 30 s each |
| **D1 D2 D3** | head turned fully away, completely silent | 30 s each |
| **F1 F2 F3** | free session — just be in the room, talk, read, move | 60 s each |
| **FREE** | one long free session at the end | 5 min |

There is a 5-second countdown before each one. The poem is printed on
screen so you can check it, **but you should not read it — recite it**.

### The poem

*Invictus* — William Ernest Henley, 1875 (public domain)

```
Out of the night that covers me,
  Black as the pit from pole to pole,
I thank whatever gods may be
  For my unconquerable soul.

Under the bludgeon of the world,
  My head is bloody, but unbowed.
Beyond this place of wrath and tears
  Looms but the Horror of the shade.

And yet of fortune's merriest lot,
  My name is written on the page...
And all the while I have remained
  The master of my fate.

I am the master of my fate:
  I am the captain of my soul!
```

Fourteen lines, four stanzas (4 / 4 / 4 / 2 — the poem ends on a couplet),
about 40 seconds. If you know a different poem better, use yours:

```
C:\Python314\python.exe C:\proart_capture\corpus_capture_win.py --plan --out C:\proart_capture\corpus --stimulus-file C:\proart_capture\mypoem.txt
```

Either way the stimulus name is recorded in every take's metadata, so
different poems can never be silently pooled.

---

## THE ONE THING THAT MATTERS MOST

**Condition D — turned AND silent — has never been recorded, and it is now
the single most valuable missing take in the plan.**

Here is why, because it changed during this session. The previous capture
said "NO POSE SIGNAL" and explained it away:

> *"the head turn was too small to register (Casey reported the glasses
> restricting it)"*

That explanation was an **assumption resting on a blind instrument**. A
later search over fourteen candidate statistics found one that does see it:
`mirror_cov`, the frame's covariance with its own horizontal mirror. It
separates the turned capture from both facing captures with margin **+0.814**
— and the two facing captures (first and last of the run) agree with each
other while only the middle one differs, so the difference is a manipulation
and not a clock.

**But nobody can yet say the difference is the HEAD.** In the turned capture
the extra structure is at the frame's left edge, where a change of posture
or a shift in the seat would look exactly the same. **Condition D settles
it**: D is turned and silent, C is facing and silent. If D differs from C by
the same margin B does, it is the head turn. If D looks like C, then B's
difference was posture, seat or lighting, and the pose channel does not
exist in this capture at all.

> **CORRECTION (2026-10-20, overnight) — the `mirror_cov` claim is
> WITHDRAWN as a pose signal, and it is withdrawn because it is CONFOUNDED,
> not because it is small.** On the corrected frames its margin *rises* to
> **+1.083** — but it also separates **A from C**, and A and C are both
> FACING the camera, differing only in whether Casey is speaking. A feature
> that separates two facing conditions cannot be a head-direction feature.
> The verdict is now **"NO feature is a clean pose signal"**.
>
> **So D is needed even more than this file said, and for a reason this
> file did not have.** The repo's biggest result is a speaking signal in
> the **mouth** (+0.733 sd against a +0.477 sd cheek control, permutation
> z +10.9, and 6,600× quieter in an empty room). The open question is
> whether that is the mouth or the head: an independent MediaPipe lane
> found that **head height moves more sharply than the mouth aperture
> itself** (+0.3316 at z +3.77, against +0.2052 at z +2.70), so the
> correlation may be about the head moving. **D is turned and silent; C is
> facing and silent.** If D differs from C, the head turn has a measurable
> footprint. If D looks like C, then the signal in the mouth box was the
> head, and the whole mouth result has to be re-read as head-motion.
>
> **And keep the turn a real 90°, and stay still otherwise.** The statistic
> sees the frame, not the head, so a posture shift is indistinguishable from
> a head turn. It is the one confound that cannot be measured away.

**So: three repetitions of D are worth more than everything else in this
plan.** And while you are at it —

  - **Make the turn a real 90°.** If the glasses restrict your head, take
    them off for B and D, or turn your shoulders as well, or move further
    from the camera. Nothing in a 160×96 frame can tell a 20° turn from a
    90° one, and the whole pose question turns on this.
  - **Stay as still as you can otherwise.** The statistic sees the frame, not
    the head, so a posture shift is indistinguishable from a head turn. It
    is the one confound that cannot be measured away.

---

## What the capture now measures that it did not before

The previous capture script had three defects, all now fixed:

1. **The audio and the video were never verified to correspond.** They were
   two separate programs (`rec.py` and `nv12small.py`) run at different
   times, and nothing recorded it. The cross-correlation of the old
   captures finds **no relationship at any lag, at any sample rate, with any
   motion feature** — and the silent session peaks too, which is what noise
   does. The new script takes **one video frame and one audio block in the
   same loop iteration with one clock reading**, and writes a `timing.json`
   beside every take with the per-frame timestamps. The correspondence is
   now **verifiable after the fact** instead of assumed.

2. **The frame rate was assumed, never measured.** The script asks for 30
   fps; the camera appears to deliver **25**, and nothing checked. That
   stretched the video's time base by 1.2× — which is exactly what would
   make an audio/video join fail. The new script records the **achieved**
   rate and warns if it differs from the assumed one.

3. **The achievement check never ran.** The old script's `src` import
   resolved to `C:\`, the exception was swallowed, and every take was saved
   with "speech check unavailable" and **no verification at all**. `src\`
   now ships beside the script and the check works, so a take where you were
   supposed to speak but didn't is **flagged as VOICE_MISSING** rather than
   quietly entering the corpus.

You will also see the mic's **actual** sample rate in the output, and the
WAV is written with that rate rather than the requested one — the old
headers claimed 16 000 Hz when the data was not.

---

## What the script prints, and what to look at

```
  frames 750   video span 25.00s (assumed 30 fps)
  ACHIEVED camera rate 25.00 fps   mic rate 16000 Hz (requested 16000)   audio span 29.98s
  drift raw 32.71   after row-band removal 1.44   OK
  speech 71.3% of frames
```

> **CORRECTION (2026-10-20, overnight) — those are the TORN numbers, and
> they are what a correct take looks like now.** The drift figures above
> (32.71) are the tear, not the room: the same captures drift **0.545**
> mean |diff| at the correct frame stride. The example a correct take
> should produce is
>
> ```
>   frames 900   video span 30.00s (assumed 30 fps)
>   ACHIEVED camera rate 30.02 fps   mic rate 19188 Hz (requested 16000)   audio span 29.98s
>   drift raw 0.54   after row-band removal 0.70   OK
>   speech 70.6% of frames
> ```
>
> The take also writes three arrays now: `maps` (row-removed change maps,
> the same key and the same meaning as before), `maps_raw` (the
> **unfiltered** change maps) and `luma` (the **raw frames**). The luma is
> what the mouth, pose, room-map and MediaPipe work needs, and the
> unfiltered maps are there because the filter's justification was
> withdrawn overnight — the banding it removed was the tear. Keeping both
> means the representation is a decision made AFTER the recording, not
> something frozen into it.

Two lines matter:

  - **ACHIEVED camera rate** — if it is not ~30, the warning below it
    explains what that means for the timing.
  - **drift after row-band removal** — must be **under 5**. If it is high,
    the note says to change the lighting, and that is worth doing before
    recording the whole set rather than after.
    > **CORRECTION (2026-10-20, overnight).** The threshold of 5 is ~9× the
    > measured **occupied** baseline (A.nv12 drifts 0.545) and ~1000× the
    > measured **empty-room** baseline (empty1.nv12 drifts 0.005), on the
    > corrected stride. It was set when the tear made the number 43.560, so
    > it has been re-derived against the real baselines. The old corpus's
    > recorded 32.08 / 32.71 / 33.44 was the tear, and on the corrected
    > frames the same takes measure **0.7 / 0.6 / 0.5** and **PASS**.

And **VOICE_MISSING** / **NOT_SILENT** warnings, which mean that take's
label does not match what actually happened. The take is still saved, but
flagged, so it can be excluded without re-recording blind.

---

## After the recording

```
python experiments/corpus_integrity.py --json /tmp/ci.json
```

This is the gate. It checks every take's label, speech plausibility, frame
rate, mic rate, A/V span agreement, drift, file integrity and stimulus
consistency, and **exits non-zero if anything fails**. It also prints the
one number that measures progress:

```
COVERAGE  (target 63 independent sessions per class)
    A             3 distinct sessions  (3 usable)  gap 60
    ...
    SPEAK         6 distinct sessions  (6 usable)  gap 57
```

**Sixty-three is not a round number someone invented.** It is
`ceil(2*(z*s/d)^2)` with z=2, s=0.01022 (the between-session spread within
a label) and d=0.00366 (the between-label separation), from
`experiments/teacher_data_requirement.py`. Two sessions per class give
leave-one-session-out **0.000**, because the teacher learns the recording
instead of learning speech. Sixty-three is the floor at which the label
becomes resolvable at all.

Right now the corpus on disk has **3 takes, all failing the drift check, and
62 sessions short of every class.** That number is the thing to beat.

> **CORRECTION (2026-10-20, overnight) — "all failing the drift check" was
> the tear, and it is no longer true.** The recorded drift was 32.08 /
> 32.71 / 33.44 against a threshold of 5; on the corrected frames the same
> three takes measure **0.7 / 0.6 / 0.5** and **PASS**. The room's
> lighting was never the problem. So the honest state of the corpus is
> **3 takes, all usable, and still 62 sessions short of every class** — the
> coverage number is the whole remaining gap, and it is unchanged.

---

## What happens to the data afterwards

1. The **cloud teacher** labels the FREE sessions. Llama 3.2 Vision 11B
   reads "is this person speaking" off a 160×96 frame with a margin of
   **+0.455** and a **zero** false-positive rate; clef-omni reads it off
   the raw audio with a margin of **+0.766**. The teacher is only allowed
   to label FREE once it beats chance on A–D.
   > **CORRECTION (2026-10-20, overnight): the vision margin is +0.667
   > (8 of 12), not +0.455 (5 of 11).** The stride defect was *depressing*
   > the margin — it was making the teacher's job harder by feeding it torn
   > frames. A separate check now shows the model's answer follows the
   > **image** rather than the prompt (swap test: shown=speak → SPEAKING,
   > shown=silent → NOT SPEAKING), so it is reading something.
2. The **conventional teacher** is benchmarked leave-one-session-out
   against the new corpus. That is the number that decides whether a
   student can be trained at all.
3. The **pose question** gets its answer, once D exists — and so does the
   **head-versus-mouth** question, because D is the control for it.

Two of those three depend on data that does not exist yet. That is the
whole remaining plan.
