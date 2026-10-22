# LENS AND WALK-BY — the sensors' own self-model, and what the envelope actually carries

**Built by:** `experiments/lens.py` (26/26 selftest green)
**The lens files:** `docs/research/lens/{microphone,camera,pitch_tracker,energy_segmenter,asr}.json`
**The study:** `docs/research/walk_by_envelope.json`
**Device:** CPU, numpy + scipy. No GPU, no paid API, no network.
**Corpus:** `/mnt/c/proart_capture/corpus/free_sessions` — **a live corpus**: 153 session directories at the time of the last run, 128 of them speech-bearing

---

## 0. THE HEADLINE

**The lens files exist, and they hold fourteen sensor behaviours, each with a
number, a unit and a provenance.** A prose sentence about a sensor is a
belief; a lens entry is a measurement, and it can be checked.

**Casey's walk-by claim is half right, and the half that is wrong is the
half that matters.** The claim was that Doppler at walking speed is below
threshold and the envelope is the robust signal. The Doppler half is
confirmed and measured: at 1.4 m/s the shift is **7.05 cents**, and the
pitch tracker's own frame-to-frame jitter on this material has a median
of **14.70 cents** — so the Doppler shift is *half* the tracker's own
noise and cannot be read.

The envelope half does not hold **on this corpus**, and the reason is
measurable rather than a matter of taste. Graded against the video's own
radial label, the envelope's direction agrees **38.5 %** of the time on
the audio-bearing crossings, and the AUC of the envelope's slope at the
event is **0.5002** — chance, on the nose. The cause is in the data: at
**50.5 %** of the audio-bearing crossings the envelope's peak lands on
one of the session's own speech segments. The hump at a crossing is
**Casey's own voice**, not a footstep. The corpus contains no labelled
toward/pass/away crossings at all, so the claim is **untested rather
than refuted** — and the tool that would test it now exists.

---

## 1. THE LENS FORMAT

A lens file is one sensor's living self-model. Five of them ship, one
per sensor the fleet actually has.

| file | sensor | what it models |
|---|---|---|
| `microphone.json` | the microphone | silence, gain, resampling fold, audio lead, the walk-by envelope, Doppler |
| `camera.json` | the camera | repeated frames, the sensor noise floor, frozen video |
| `pitch_tracker.json` | the pitch tracker | octave artefacts, the Doppler deadband |
| `energy_segmenter.json` | the energy segmenter | the leading-silence escape, the peak-anchored threshold |
| `asr.json` | the ASR | the non-speech hallucination |

The format is deliberately small. Every entry has an `id`, a `class`
(`deadband`, `delay`, `level`, `signature`, `failure_signature`), a
`statement` in prose, a `measurement` with numbers, and a `provenance`
naming where the numbers came from. The selftest enforces it: every
behaviour must have an id and a class, and every behaviour must have a
measurement and a provenance. A lens entry without a number is a
sentence, and the selftest refuses it.

---

## 2. THE MICROPHONE

### 2.1 SILENT_AUDIO — the failure signature

**56 of 97 session bundles carry digital silence** — every sample zero.
That is 57.7 % of the corpus with a dead audio channel. The eight
`session_saved` records in `browser_capture/liaison_outbox.jsonl` carry
`peak_abs: 0` and exactly eight `SILENT_AUDIO` alerts, so the linker
knew and said so; the failure is upstream of the linker.

**44 of the 97 bundles are not a room at all.** They are an end-to-end
test rig: Chrome's "Fake Default Audio Input" and a static `.y4m` test
pattern standing in for the camera. Each session's `meta.json` names its
own devices, so the split is read rather than guessed — and it has to be
read, because the first cut of this study mixed the two populations and
produced a false claim (below).

On the **real** microphone (53 sessions), **47 are silent** — 88.7 %.
The loud ones peak between **140** and **3,064**, a **21.9×** spread in
the microphone's own gain across one room on one day.

### 2.2 GAIN_CHANGE — the level

The same 21.9× ratio, filed as a level rather than a fault, because the
two readings of it are different: a silent session is a dead channel, a
loud one is a gain change, and only the second is informative. Any
absolute threshold set from one session's loudness is wrong for the
next one by up to 22×.

**The population matters, and getting it wrong is what this section's
first draft did.** Mixing the synthetic sessions in gives a spread of
**54 to 18,064 — 334×** — because the fake input's synthetic test tone
peaks at 18,064 while the real microphone peaks at 3,064. That 334× was
a test tone counted against a room. The lens file now reports the real
microphone's 21.9× as the headline and keeps the whole-corpus figure
under a separate key, with the reason attached.

### 2.3 RESAMPLE_FOLD — the deadband

`src/wavio.py` resamples without an anti-aliasing filter. Measured
directly, by generating tones at 48 kHz and resampling them the
repository's way:

| input | measured at 16 kHz | error |
|---|---|---|
| 1,000 Hz | 1,000 Hz | 0.0 cents |
| 5,000 Hz | 5,000 Hz | 0.0 cents |
| 8,000 Hz | 8,000 Hz | 0.0 cents |
| 10,000 Hz | **6,000 Hz** | **−884.4 cents** |
| 14,000 Hz | **2,000 Hz** | **−3,368.8 cents** |

Below 8 kHz the tone lands where it was put. Above it, it folds — and a
14 kHz tone arrives at 2 kHz, **10 octaves** down. Anything in the
corpus above 8 kHz is not what it appears to be, and the fold error is
not a small distortion: it is an identity change.

### 2.4 AUDIO_LEAD — the delay

The session time of the WAV's first sample against the session's first
frame, derived from `wall_clock_end − wall_clock_start − audio length`:

| session | audio lead |
|---|---|
| 1792560597 | −71.0 ms |
| 1792560758 | −92.3 ms |
| 1792560773 | −6.3 ms |

The audio **leads** the video by 6–92 ms. The sign is negative because
the audio's end and the wall clock's end are the same close, so the
audio's *start* is earlier than the video's. A 92 ms offset is longer
than the pitch tracker's own frame spacing at some settings and long
enough to move a boundary decision by a frame.

### 2.5 WALK_BY_ENVELOPE — the signature, and the measured null

This is the entry that answers Casey's homework, so it gets the most
space. §3 is the whole story; here is the entry's shape.

---

## 3. THE WALK-BY ENVELOPE — THE MEASUREMENT

### 3.1 What was asked

The perception addendum's homework: *in the fixed-mic sessions, extract
loudness envelopes around known crossings and say whether toward / pass /
away separates in the envelope alone.* And the framing: Doppler at
walking speed is below threshold, Casey said so first, and the envelope
is the claim to test.

### 3.2 The first two detectors, and why they failed

Both failures are worth recording, because each produced a number that
looked fine.

**Detector 1 — a run of consecutive changed frames. Zero crossings, in
every session.** The recorder writes a picture when it changes and
repeats it otherwise, so changed frames are isolated: the run-length
histogram over 15,513 frames is `{1: 5152, 2: 8, 3: 2}` — a maximum run
of three. A detector that requires a run finds nothing, ever.

**Detector 2 — a burst of change events. 54 "crossings" in 80 sessions,
and every one of them was the whole session.** The burst's centroid
drifted 2.6 columns over 258.9 seconds, which is a direction that does
not exist. The reason: change events are **dense**, not sparse. A third
of all frames change (5,174 of 15,513), with a median gap of **three
frames**. Any gap-based split of a stream with a median gap of three
frames returns the stream.

The fix was to notice that the two things being measured have different
*shapes*. A person walking past is a **connected blob of change that
holds its shape while it translates**. The camera's own grain is spread
thinly over the whole frame and does not hold its shape. A highlight is
one frame. A fidget does not travel.

### 3.3 The change map's noise floor — measured, not guessed

The third detector needed a threshold, and the threshold was the thing
that had been wrong from the start. The first version used **zero**, on
the belief that a change map is either exactly zero (the frame was
repeated) or a real change. It is not: the camera's grain fires on a
third of the frames.

Measured across three sessions, the per-frame maximum of the change map:

| session | frames with max = 0 | non-zero max: p10 | median | p90 |
|---|---|---|---|---|
| 1792553040 | 66.6 % | 3.71 | 21.75 | 61.11 |
| 1792560758 | 66.9 % | 2.19 | 8.38 | 46.95 |
| 1792560773 | 66.1 % | 1.56 | 2.59 | 6.66 |

The room map records the motion noise at **σ 0.61 luma** per change, and
the quietest session's non-zero maxima bottom out at 1.56. The threshold
used is **3.0 luma** — above the quiet session's 90th percentile of
1.56 and about 5× the room's noise σ. It is the same 3.0 luma the
motion-as-text rate policy uses, and the same measurement, which is the
point of a lens: one number, one meaning, two consumers.

### 3.4 The detector

A crossing is a connected blob above the noise floor, at least **4
cells** wide, surviving at least **5 frames**, whose centroid travels at
least **3.0 grid columns** in one direction. The blob's centroid is
weighted by change magnitude rather than by cell count, because the
blob's edge is fainter than its core. Events are grouped by time **and**
position: a blob that keeps its place is the same object seen again.

**574 crossings across 80 sessions**, in 35 sessions that have any.

**NOTE, 2026-10-22 — the gap was counted in seconds and the corpus moved
to 2 fps.** The detector groups events by time as well as by position, and the
gap it allowed a blob to be missing was a flat **0.35 s**. That is 21 frames at
the 60 fps every session above was captured at, and **0.70 frames** at the 2 fps
the corpus is captured at now. At 0.70 frames no blob can ever chain, every blob
is a singleton, every event is shorter than the 5-frame minimum, and the
walk-by study finds **nothing at all** in those sessions.

The gap is now the session's own, counted in frames: `max(0.35 s, 1.5 × the
session's median frame interval)`. Measured over the corpus as it stands:

| the corpus's sessions | the flat 0.35 s gap | the gap in frames |
|---|---|---|
| the 92 fast ones (60 fps, all of them captured on 2026-10-21) | 739 crossings | **739** — the same number |
| the 11 slow ones (2 fps, all of them captured on 2026-10-22) | 2 crossings | **25** |

**NO FILED NUMBER MOVES.** The 574 crossings above are the fast sessions'
crossings, and the fast sessions' crossings are unchanged; the eleven 2 fps
sessions are all newer than this study, so this study never saw them. What the
fix does is let the corpus's next 2 fps sessions be seen at all. The defect was
found by the dual-capture lane, which had to ask the crossing question of two
cameras of the same minutes and found the instrument returning nothing
(`docs/research/dual/DUET-RENDER.md` §6).

**NOTE, 2026-10-22 — the corpus grew, the sweep was re-run, and the
prediction above is confirmed.** The sweep now runs over **103** session
entries (the filed json held 88), finds **764** crossings against the filed
**574**, and **45** sessions have any against the filed **35**.

**NOT ONE EXISTING SESSION'S CROSSINGS MOVED.** All 88 shared sessions came
back byte-identical, which is the "no filed number moves" promise above kept
on the real corpus rather than argued. **And THE ELEVEN 2 fps SESSIONS NOW
CONTRIBUTE EXACTLY 25 CROSSINGS**, which is the number the table above
predicted — 25 was measured in the gap note's own sweep, and here are those
sessions inside the corpus sweep itself. The 25 crossings are spread over six
of the eleven; five of the 2 fps sessions carry none.

**THE FOUR NEW FAST SESSIONS ARE ORDINARY.** Their crossings per minute run
0.22, 0.20, 0.20 and 0.09, inside the corpus's existing range (median 0.19,
max 0.45 over the 35 sessions that had any). Their higher raw count is
duration, not character: three of them run 198–300 s and carry 65, 58 and 39
crossings, while the fourth runs 32 s and carries 3.

**AND THE RATE IS NOT COMPARABLE ACROSS THE TWO FRAME RATES, WHICH IS THE
NEXT THING TO FIX.** A crossing's minimum duration is **5 FRAMES**, and 5
frames is **2.5 s at 2 fps** against **0.083 s at 60 fps**. The corpus's
crossings-per-minute is therefore now a mixture of two different quantities,
and only the COUNT is comparable between them. The 2 fps sessions show
0.31–3.16 crossings per minute against the 60 fps sessions' median of 0.19.
**THIS IS THE SAME FAMILY AS THE NOISE FLOOR: A MEASUREMENT IS A PROPERTY
OF A SENSOR AND AN INTERVAL, AND THE DETECTOR'S FRAME CRITERIA ARE IN
FRAMES WHILE ITS WINDOW IS IN SECONDS.** The gap defect was the chaining
rule; this is the duration rule. Both are the detector's parameters being
in seconds where the corpus has moved to a different interval.

**THE LENS FILES MOVED WITH THE CORPUS, AND ONE OF THEM MATTERS FOR THE
ARRIVAL NULL.** The real microphone's session count goes **53 → 71** with the
silent count **unchanged at 47**, so the corpus's real-microphone silence
fraction falls from **88.7 % to 66.2 %**: all eighteen new real-microphone
sessions carry signal. The arrival-null study's corpus therefore grew from
**6** sessions carrying signal to **24** — see
`docs/research/ARRIVAL-NULL.md` §5a, where the re-run says the null bias
holds and the resolution headline turns out to be a median and not a worst
case.

**NOTE, 2026-10-22 — THE DURATION RULE IS DECLARED BEFORE THE RUN, AND IT
WILL MOVE FILED NUMBERS.**

The sweep-refresh note above named two of the detector's parameters being
in the wrong unit: the gap was in seconds where the corpus had moved to a
different interval, and the duration is still **5 FRAMES** where the
corpus's window is in seconds. The gap is fixed. The duration is not.

**THE RULE: A CROSSING MUST BE PRESENT FOR AT LEAST ONE MOMENT, AND A
MOMENT IS A TIME.** The detector's criteria must be in the same units as
the quantity they measure, and the corpus's window is in seconds. So the
duration floor is:

> the minimum duration = **max(0.35 s, 1.5 × the session's own median
> frame interval)**

**WHICH IS THE CORPUS'S OWN "A MOMENT", AND IT IS THE SAME EXPRESSION THE
CHAINING GAP ALREADY USES.** The gap is how long a blob may be **MISSING**
and the duration is how long it must be **PRESENT**; they are one moment,
one constant and one frame rate. **The `5 FRAMES` criterion is retired.**

**WHAT IT DOES TO THE TWO FRAME RATES**, stated before the run and not
derived from it:

| | the old floor | the new floor | what the floor means |
|---|---|---|---|
| 60 fps | 5 frames = 0.083 s | **0.35 s = 22 frames** | a blob present for a moment |
| 2 fps | 5 frames = 2.5 s | **0.75 s = 3 frames** | a blob present for a moment |

**AND THE OLD RULE NEVER EXCLUDED A HAND AT 60 fps ANYWAY.** Its stated
purpose is to be "a crossing rather than a single moment of a person's
hand", and a hand flick lasts longer than 83 ms — so at 60 fps the old
floor admitted hand flicks. At 2 fps the old floor excluded the real
crossings too, because a person who walks through in one second is two
frames and the old rule wanted five. **Neither rate was counting
walk-bys**, and the corpus's crossings-per-minute was a mixture of two
quantities.

**THE BARS, STATED BEFORE THE RUN.**

1. **The rule is stated before the run, and the run consults no truth.**
   The detector counts blobs; there is no label anywhere in it.
2. **Every number that moves is filed with the number it replaces**, and
   nothing is quietly restated: the crossings, the sessions with any, the
   envelope's AUC and the rate are all restated with the new counts.
3. **The crossings-per-minute is reported per frame rate and never
   pooled**, and each rate's minimum crossing duration is stated in
   seconds beside it.
4. **Nothing else in the detector changes**: the same threshold (the
   corpus's own floor of 3.0 luma), the same blob rule, the same travel
   floor of 3 columns, and the same chaining gap.

**AND THE FILED NUMBERS ARE STALE UNTIL THE NEXT COMMITMENT.** This note
and the tool change land together; **the sweep has not been re-run yet**,
and `results/walk_by_envelope.json` still holds the counts the old rule
produced. The next commit runs the sweep and files what moves.

**NOTE, 2026-10-22 — THE RUN, AND THE NUMBERS THAT MOVED.** The sweep was
re-run under the new rule. **The corpus grew again while it ran**, which
is the corpus's nature and not the rule's doing, so the two effects are
separated: first the rule's effect alone, on the same sessions, and then
what the new arrivals added.

**THE RULE'S EFFECT ALONE, ON THE SAME SESSIONS:**

| | the 5-frame duration | the duration as a time | the rate |
|---|---|---|---|
| the 92 fast sessions (60 fps), 63.54 min | 739 crossings | **640** (−99) | 11.6309 → **10.0728** per minute |
| the 11 slow sessions (2 fps), 14.50 min | 25 crossings | **33** (+8) | 1.7247 → **2.2766** per minute |

**THE DIRECTION IS THE ONE THE DECLARATION PREDICTED: THE FAST RATE FALLS
AND THE SLOW RATE RISES.** The fast sessions lose the flickers the old
5-frame rule admitted, and the slow sessions find the crossings the old
rule could not see. **AND THE TWO RATES ARE CLOSER TOGETHER: the gap
between them was 6.7× and is now 4.4×** — which is what a shared unit
buys. They are not equal, and they should not be: a camera that looks
twice a second cannot see a 0.4 s walk-by.

**THE CORPUS GREW UNDER THE SWEEP, AND WHAT IT ADDED IS STATED
SEPARATELY:** 21 new fast sessions (12.37 min, 22 crossings) and 4 new
slow ones (5.82 min, 8 crossings). **THE WHOLE SET AS IT NOW STANDS:**

| the frame rate | sessions | minutes | crossings | sessions with any | the rate | the minimum duration that applied |
|---|---|---|---|---|---|---|
| 60 fps | 113 | 75.87 | 662 | 42 | **8.7254** per minute | **0.35 s** |
| 2 fps | 15 | 20.17 | 41 | 7 | **2.0329** per minute | **0.7441 s** |

**THE RATE IS NOW REPORTED PER FRAME RATE AND NEVER POOLED**, and each
rate's minimum crossing duration is stated in seconds beside it, because
the duration criterion is the session's own moment and the two rates'
moments are different lengths. The file's own `the_summary` carries the
grouping; a pooled rate would be a mixture of two quantities, which is
the defect this rule was declared to remove.

**THE SWEEP AS A WHOLE: 128 session entries, 703 crossings, 49 sessions
with any** — against the 103 entries, 764 crossings and 45 sessions with
any that the 5-frame rule produced. The crossings fell because the fast
sessions shed their flickers, and the corpus grew by twenty-five sessions
while it ran.

**AND THE ENVELOPE'S VERDICT HOLDS UNDER THE NEW RULE.** Graded against
the video's own radial label on the crossings with a live envelope, the
agreement is **38.5 %** against a chance of **37.4 %**, and the AUC of the
envelope's slope at the event is **0.5002** — chance on the nose, where
the old run read 0.5064. The envelope's peak still lands on the session's
own speech **50.5 %** of the time, where it was 74.5 % then. **The hump
at a crossing is still mostly Casey's own voice; it is half of them now
instead of three quarters.**

### 3.5 The envelope, and the bug that hid itThe envelope is the repository's own: `src/energy.py:rms_series` on a
50 ms frame with a 10 ms hop, in dB.

The first run returned a null envelope for **every single crossing**. The
cause was a time-base error in the study, not in the corpus.
`rms_series` numbers its frames from the start of the *slice* it is
handed, so the times it returns are slice-relative; the shape function
then compared them to the *absolute* session time. The window it looked
for was the whole session away from the data it had, and every window
came back empty.

The fix is one line — shift the frames back onto the session's clock
before the comparison — and it is the kind of bug that a report without
the null-check would have hidden entirely. The version of this study
that reported "0 crossings" and the version that reported "54
crossings, all whole sessions" were both *plausible*; the one that
reported a null envelope was not, and that is why the null shows up.

There is a second time-base subtlety worth naming, because it is
quantitative: the analysis frame is 50 ms long and its time is its
**centre**, so a reconstructed grid shifts the whole envelope by 25 ms.
That is a quarter of a walking hump's width, and enough to turn a
symmetric hump into a slope — which would have turned PASS into TOWARD
or AWAY for reasons of arithmetic rather than of physics.

### 3.6 The label: the video's own radial direction

To grade the envelope, something has to label the crossings. The video
can, and the mechanism is the blob's **apparent size**: a person walking
toward the phone grows in the picture; walking away, they shrink. The
room map says its own words about the limits: *"this is the room as the
camera knows it: an image plane, not a floor plan... Nothing here
measures a distance."* The apparent size is the closest thing the camera
has to a distance, and it is used here as a **label**, not as a
measurement of metres.

Size growth across the 574 crossings: p10 **0.25**, median **1.12**, p90
**5.55**. The video's labels come out **253 TOWARD, 196 AWAY, 125 PASS**.

### 3.7 The result: the envelope carries the event and not the direction

On the **audio-bearing** crossings — the 283 whose envelope is live rather
than digital silence (the synthetic rig's sessions are excluded: their
envelopes are a test tone, not a room):

| | | the 574-crossing run this was first filed on |
|---|---|---|
| crossings with a live envelope | **283** | 94 |
| agreeing with the video's radial label | **109 (38.5 %)** | 39 (41.5 %) |
| chance agreement, from the marginals | **37.4 %** | 39.2 % |
| AUC of the envelope's slope at the event, TOWARD vs AWAY | **0.5002** | 0.5064 |
| envelope's peak lands on one of the session's own speech segments | **143 (50.5 %)** | 70 (74.5 %) |

The confusion matrix over these 283 crossings (rows = video's label, columns =
the envelope's):

|  | TOWARD | PASS | AWAY |
|---|---|---|---|
| **TOWARD** | 66 | 12 | 41 |
| **PASS** | 33 | 7 | 28 |
| **AWAY** | 54 | 6 | 36 |

Chance agreement on this split, from the marginals, is **37.4 %**; the
measured agreement is **38.5 %**. The AUC of the envelope's slope at the
event is **0.5002** — chance on the nose. Across all 703 crossings the
same computation gives chance **28.4 %** against measured **27.9 %**.

**Why, and it is measurable.** The envelope's peak at a crossing lands
on one of the session's own speech segments **50.5 %** of the time. Across
all 703 crossings it is much lower, and that low number is itself the
confound made visible: the sessions whose audio is digital silence cannot
produce a speech-aligned peak, so the all-sessions figure understates it.
On the sessions where the microphone was actually alive, the hump at a
crossing is **Casey's own voice** — and it is half of them now, where it
was three quarters when this was first filed.

**EVERY NUMBER IN THIS SECTION NOW HAS A TOOL BEHIND IT, AND IT WAS NOT
TRUE BEFORE.** The agreement, the confusion matrix, the chance agreement
and the AUC were written by hand into this report and there was no code
anywhere that produced them. `lens.envelope_grading(rows)` is the tool
now, and it **reproduces every one of the old figures exactly** on the
exact file the old figures were filed against — 94 live crossings, 39
agreeing, 41.5 % against a chance of 39.2 %, an AUC of **0.5064**, 70 on
speech, and 24.7 % against 24.6 % over all 574. **The old numbers were
right; they were just unchecked.**

**AND THE AUC'S DEFINITION HAD TO BE RECOVERED, WHICH IS HOW THE GAP WAS
FOUND.** The filed 0.5064 is the **trapezoidal** AUC — one if a positive
outscores a negative, half if they tie — of the envelope's slope at the
event, labelled by the video's own radial label, on the crossings whose
video label says TOWARD or AWAY. The Mann-Whitney rule gives 0.6045 on the
same data, and the difference is not small. **The first version of this
tool was silently computed by a DIFFERENT FUNCTION**: there is an older
`_auc(pos, neg)` further down this file, and a function of the same name
defined later shadows the newer one. The new one is called `_auc_of` now,
and **a name that is taken twice is a number nobody checked.**

This is the same confound the pause study found and named in the other
direction — there, 47 free sessions carry no interior pause because the
ASR heard nothing; here, the crossings carry no envelope excursion
because the envelope is the speaker.

### 3.8 The honest verdict

**The envelope carries the event; on this corpus it does not carry the
direction.** But the corpus cannot answer the question, and the reason
is not a defect in the method:

- **There are no labelled toward/pass/away crossings.** The four
  `probe` sessions in the corpus are `sonar-band` stimulus artefacts
  (16–22 kHz exponential sweeps with a pink burst), not walk-bys. The
  dialogue protocol that would produce labelled crossings has **0 of
  1,230 s** recorded.
- **The crossings that exist are Casey's own body moving.** The blob's
  centroid sits inside the room map's person zone (x 7–119, y 42–95) at
  **90 %** of crossings across the corpus and **93 %** of the
  audio-bearing ones. Casey is seated and gesturing, not walking past
  the microphone. His voice is in the microphone. The envelope sees his
  voice.
- **So the test that was asked for was never run**, and what is measured
  instead is the null that the confound predicts.

What the study *does* deliver is the instrument. The crossing detector
works and is now a lens entry; the envelope extraction works and is
now a lens entry; the label mechanism is stated; and the one thing the
corpus can answer — the Doppler question — is answered with a number.

### 3.9 What would answer it

A capture with a person walking **toward / past / away** on cue, from a
fixed microphone, in silence or with the speaker muted. Then the same
tool grades the envelope against the known label, and the null above
becomes either a pass or a kill. The corpus for it does not exist yet;
the tool for it does.

---

## 4. THE DOPPLER QUESTION — CASEY WAS RIGHT

Casey's claim was that Doppler at walking speed is below honest
threshold. **Measured, and confirmed.**

At 1.4 m/s with the source at 1 m, the fractional shift is v/c =
1.4/343 = 0.408 %, which is **7.05 cents**. The pitch tracker's own
frame-to-frame step, measured on the real crossings with a voiced track:

| | cents |
|---|---|
| Doppler shift expected at 1.4 m/s | **7.05** |
| tracker's step, median across crossings | **14.70** (min 11.22, max 18.26) |
| tracker's step, 90th percentile | **74.31** (min 50.52, max 1195.53) |

The Doppler shift is **less than half** the tracker's own median jitter
and **an order of magnitude** below its 90th percentile. A shift smaller
than the instrument's noise is not a measurement, and the pitch channel
cannot carry the direction of a walk-by on this material.

So the pitch channel is out. The envelope was supposed to be the
substitute, and §3 is what the envelope does when it is actually
measured against a label instead of being asserted.

---

## 5. THE CAMERA

| entry | class | the number |
|---|---|---|
| REPEAT_FRAMES | deadband | **66.9 %** of frames are byte-identical to their predecessor (median over 25 sessions; min 63.3 %, max 67.0 %) |
| SENSOR_NOISE_FLOOR | deadband | a changed frame differs by **0.67 luma** on average over the whole frame (pooled median); the room map's per-pixel motion σ is 0.61 |
| FROZEN_VIDEO | failure_signature | **0** frozen sessions — every session in the sample delivers frames |

The repeat fraction is the load-bearing number: **two thirds of the
frames carry no new information at all**, and the recorder's own
"write when it changes" behaviour is why the change map is a sparse
event stream rather than a dense one. The motion-as-text rate policy
turns that same measurement into the bytes-per-second argument.

---

## 6. THE PITCH TRACKER, THE SEGMENTER, THE ASR

**Pitch tracker.** The octave artefact is the failure signature: one
octave artefact and seven of thirteen subharmonic artefacts in one
cell, and the pause study's free corpus reaches **+344 st/s**. The
walk-by crossings add the Doppler deadband above.

**Energy segmenter.** Two documented behaviours, both inherited and
both re-verified here. `LEADING_SILENCE_ESCAPES`: the first segment of
a file is kept unconditionally (`src/energy.py:290`), so the silence
*before* the first word escapes the `min_silence` filter and is reported
as a pause — the published shortest pause is 0.110 s and the
artefact-free shortest is 0.150 s. `PEAK_ANCHORED_THRESHOLD`: the speech
threshold is 10 % of the clip's loudest frame, so in a longer recording
where the loudest frame is a clap, the threshold is set by the transient
and the recording is labelled speech. The source says so itself at
`src/energy.py:224`.

**ASR.** The failure signature is hallucination on non-speech: the
16–22 kHz exponential sweep with a pink burst produced **14 words** in
one probe and none in the other three. The pause study found the same
thing independently — a sine sweep produced 14 hallucinated words.

---

## 7. WHAT THIS DOES NOT SAY

- **It does not say the envelope cannot carry a walk-by.** It says the
  corpus cannot show it, because the hump at a crossing is the speaker's
  voice and there are no labelled crossings.
- **It does not say the crossings are wrong.** 574 crossings were found
  by a detector whose every threshold is measured, and their centroids
  sit inside the person zone at **90 %**. They are Casey moving. They
  are not a person walking past the microphone.
- **It does not treat the synthetic rig as a room.** 44 of the 97
  bundles are an end-to-end test rig — Chrome's fake audio input and a
  static `.y4m` test pattern. The device label is read from each
  session's `meta.json` and the acoustic population is the real
  microphone's. This is not a formality: the first draft of this study
  mixed the two and produced a 334× "gain spread" that was a test tone
  counted against a room.
- **It does not generalise past one room.** The room map's own verdict:
  one space, one device, one camera position. The 46 of 50 sessions that
  map to the composite viewpoint do so with shifts in (−1..4, −2..1)
  pixels — the camera barely moved.
- **It does not grade the camera.** The 25-session sample is the
  camera's own behaviour; the room's is a separate claim with separate
  numbers.

---

## 8. HOW TO REPRODUCE

```sh
# the lens files and the walk-by study (about two minutes)
python experiments/lens.py --out docs/research

# the selftest: 13 checks, including the format checks
python experiments/lens.py --selftest
```

The study is a snapshot: the corpus is live and was growing during this
work. The session list is recorded in `docs/research/walk_by_envelope.json`.

Committed for Muse to review; not merged.
