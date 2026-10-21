# SHADOW WORK — what the system had to keep, and re-analyse, to not repeat last night

**What this is.** The retention brief's addendum makes the garbage
collector a **student, not a static rule**. Its curriculum is the
system's own error history. Any segment beneath a conversation where the
system got something wrong is **pinned** — exempt from collection at
every layer — until *shadow work* completes: background re-analysis of
the pinned material with the critique as the question, producing an
extracted lesson filed to the person portrait. The debt clears when the
lesson exists, not when time passes.

Casey named three real corrections from the night of 2026-10-20 and
asked the question that turns the addendum into a method: **what would
the system have had to KEEP, and re-analyse, to not repeat each one?**

**The short version.**

1. **The three corrections live in three different layers**, and that is
   the whole finding. The silent-audio miss is an **AUDIO** failure. The
   biggest-peak mislabel is a failure of a **label** — of the evidence
   the label was derived from, and of its controls. The stalled seats
   are a failure of an **END** — the material existed and was thrown
   away because the wait never ended.
2. **The law the three share is one sentence: the layer that failed is
   the layer that must be kept.** Not the cheapest layer, not the
   richest, and not the transcript because the transcript is nice. The
   layer in which the failure happened, because that is the only layer
   whose re-analysis can detect a repeat.
3. **For each correction the answer has two halves, and they are
   different halves.** What had to be KEPT is a question about *which
   layer's material must survive collection*. What had to be
   RE-ANALYSED is a question about *what to ask that material*.
4. **The pins fall out of the corpus tonight**: 51 sessions whose WAV
   peaks at exactly zero, one performed take, one frozen video layer,
   one sonar stimulus that made the ASR hallucinate fourteen words.
   §6 is the list.
5. **The shadow work graduates into a calculator**, which is the
   doctrine's end-state: a cheap mechanical detector, fitted per user,
   that needs less energy every generation. §7.

---

## 1. The three corrections, and where each one lives

| correction | what happened | the layer it lives in | the evidence |
|---|---|---|---|
| **the silent-audio miss** | the input device delivered literal zeros while motion kept opening segments; **82.4M samples of silence accrued across 34 folders** with no signal anywhere in the UI that the mic was dead | **AUDIO** — and only AUDIO can tell a dead device from a quiet room | `browser_capture/DONE.md`, SICK LANE section |
| **the biggest-peak mislabel** | a label was taken from the largest peak of some signal, and that peak was not what the label was about | the **DERIVATION** — the evidence a label was computed from, and the controls that say whether the label is about the signal or about the session | three measured instances, §3.1 |
| **the stalled seats** | `await recorder.onstop` after `recorder.stop()` never fired, leaving a **finished take hung forever with no POST** — "the observed failure" | the **END** — the partial result and the liveness record | `browser_capture/DONE.md`, the stall audit |

The three are not three bugs. They are three shapes of the same
question, and the question is the one the GC has to answer every night:
**what is the least material that, if it were kept, would have caught
this?**

---

## 2. The silent-audio miss

### 2.1 What happened

On the bad night the Realtek array delivered zeros — literal zeros, not
a quiet room — while the motion trigger kept opening segments. Nothing
in the UI said the microphone was dead. The number is in the repo's own
record: **82.4M samples of silence across 34 folders**.

The bug became visible only when somebody went and measured the files
afterwards. That is the shape of the failure: **the failure was in the
layer nobody was reading.**

### 2.2 What the corpus says now

Measured over the corpus as it stands (71 sessions carrying an
`audio.wav`):

| | count |
|---|---:|
| sessions with an `audio.wav` | **71** |
| of those, the WAV's peak is **exactly 0** — digital silence | **51** |
| sessions carrying signal | **20** |
| of those 20, the free-speech sessions | **6** |
| the rest are sonar probe stimuli | 4 |
| the newest sessions, captured during this study | 10 |

**The class is not an anecdote. It is three quarters of the corpus.**

The repair is already in place on the capture side and is the right
one: the deafness gate, the measured `peak_rms` stamped into every
session's `meta.json`, and the liaison's `SILENT_AUDIO` self-alert. The
outbox shows the self-alert working: **8 `session_saved` events with
`peak_abs: 0` and exactly 8 matching `SILENT_AUDIO` alerts**, with no
miss among the sessions that went through the liaison's path.

### 2.3 What had to be KEPT

**The raw WAV, until the health measurement has been made on it.**

The reason is specific and it is the reason a derived layer cannot
substitute. A silent WAV produces an **empty derivation**: no notes, no
words, no score. An empty derivation is *indistinguishable from a quiet
moment* — a person sitting still saying nothing, and a dead microphone,
produce the same transcript and the same score. The one artefact that
separates them is the waveform, and it is the one artefact the
derivation chain was built to replace.

So the retention rule that would have prevented the miss is:

> **A segment's audio may not be collected until the audio's own health
> has been measured, and a segment whose derivation is empty is a PIN,
> not a collection candidate.**

The second clause is the important one. A quiet session and a dead
device produce the same *derived* material, and the derived material is
what the GC looks at. Only a rule that treats "no derivation" as
*suspicious* rather than as *empty* catches it.

### 2.4 What had to be RE-ANALYSED, and with what question

**The WAV's own peak and nonzero count — once, at save time.** One
measurement, and it is the measurement the liaison already makes
(`/api/sessions` computes it from the files with `array`/`math.sumprod`,
memoized on `(size, mtime)`, never asserted).

For the **51 sessions already collected**, the re-analysis is the same
measurement over the same files, and the question it asks is narrow:

> **Does this segment have audio at all?**

That question is answerable today, on all 51, and it does not need a
judge, a teacher or a referee. It is arithmetic on a file that is
already on disk.

### 2.5 The lesson, and the pin

**Lesson:** *a segment with no derivation is not a quiet segment; it is
an unmeasured segment.* Filed to the portrait as a property of the
**capture chain**, not of Casey — this is a lesson about the instrument,
and putting it in the person portrait would be wrong.

**Pin:** the 51 zero-peak sessions, with the re-analysis above as the
shadow work. The debt clears when the lesson exists.

**What this creates if the lesson is not applied:** the GC will collect
51 of 71 sessions as "routine, nothing happened" — and one day it will
collect the one session where the microphone died during the only thing
Casey said that mattered. The silence is not evidence of nothing; it is
evidence of nothing *about the microphone*.

---

## 3. The biggest-peak mislabel

### 3.1 The failure class

**A label was decided by the largest peak of some signal, and that peak
was not what the label was about.**

The repository contains three measured instances of this class, and they
are worth naming separately because they are different instruments.

**(a) The threshold anchored on the clip's loudest frame.**
`src/energy.py:224`:

> The speech-on threshold is 10% of the peak frame RMS (never below
> ABS_FLOOR). Peak, not percentile, because Phase 1 clips are single
> utterances: **the loudest frame is the voice by construction**; a
> longer recording with applause would need a percentile instead —
> **stated limit**.

The limitation is stated honestly in the source. The mislabel follows
from it: in a longer recording where the loudest frame is a clap, a
knock, a door or an applause, the speech threshold is set by the
transient, and the whole recording is labelled speech.

**(b) The gate's peak, understated by up to 14.5×.**
`experiments/gate_peak_coverage_results.json`: the question asked was
whether `GATE_FIELD` reports the segment's peak as documented, and the
answer is **no — `max_understatement_x: 14.5`, and every row is
understated.** A peak that is the anchor of a label, and is wrong by an
order of magnitude, mislabels in one direction only: the label is set
too low, and the silence around the sound is called sound.

**(c) The mouth detector's fold C, which is the sharpest instance.**
`docs/research/MOUTH-DETECTOR.md` §5: the detector's threshold was
fitted against the **empty room** as the negative class. The empty
room's mouth motion is **10,748× quieter** than the speaking take's, so
the fitted threshold landed at zero, and the detector **labelled the
silent wearer "speaking"**. The numbers:

| | value |
|---|---:|
| macro frame recall of the mouth detector | **0.6655** |
| macro frame recall of a **blank** that always says "speaking" | **0.6667** |
| the margin over the blank | **−0.0011** |
| change in the metric when A's and C's labels are **swapped** | **−0.0006** |

**The detector scored exactly what a constant predictor scores**, and
the metric cannot tell the right labelling from the wrong one. That is
the mislabel, measured: the label was about the *session*, and it was
being read as a statement about the *speaking state*.

**(d) And the F0 version.** `docs/research/F0-SPEAKER-CHANGE.md`: the
biggest F0 jumps in this repository's material are **tracker
artifacts**, not voices — one octave artefact and seven of the thirteen
subharmonic artefacts sit in a single cell. A speaker-change label
taken from the biggest jump is a label taken from an artefact.

### 3.2 What had to been KEPT

**The per-frame evidence a label was derived from, and the controls that
say whether the label is about the signal or about the session.**

Concretely, for each of the three instances:

| instance | what had to be kept |
|---|---|
| the peak-anchored threshold | the **whole frame series**, not a summary: the peak can only be judged against the rest of the recording, and a clip is not a recording |
| the understated gate peak | the **gate's own per-frame output**, not the scalar it reports: an understatement of 14.5× is invisible in the scalar and obvious in the series |
| the mouth detector's fold C | the **mouth box's per-pixel temporal deviation**, the empty room, the blank, and the **wrong-session pairing** — the controls ARE the evidence |

The last row is the one that matters for the method. **The controls are
not a nicety added to a measurement; for a label, the controls ARE the
measurement.** The wrong-session pairing cost nothing to keep — it is
the same frames with the labels swapped — and it is the only thing that
exposed the mislabel.

### 3.3 What had to be RE-ANALYSED, and with what question

**Re-run the label's own derivation against the controls, and ask
whether the label moves when the labels are swapped.**

> Is this label a property of the **signal**, or a property of the
> **session**?

For the mouth detector the answer is measured and it is the answer that
should have stopped the lane: the metric moved by **−0.0006** when A's
and C's labels were exchanged. A label that does not move when the
labels move is not measuring the label.

This is also, note, exactly the shape of the audit in the retention
study: an independent instrument, with its own controls, grading
something the selection already decided.

### 3.4 The lesson, and the pin

**Lesson:** *a label that does not move when the labels are swapped is
not measuring the label; it is measuring the session.* Filed to the
portrait as a property of the **detector**, not of Casey.

**Pin:** every segment whose label was decided by a peak, together with
its controls. On this corpus that is the seed A/B/C triple and the
derived takes — the only material in the corpus that carries a speaking
label at all — plus every free session whose trigger label was decided
by the audio's peak.

---

## 4. The stalled seats

### 4.1 What happened

`await recorder.onstop` after `recorder.stop()` is a wait on a driver
event. When the driver never fires it, the take is **finished** — the
frames are recorded, the state machine says RECORDING is over — and the
system **hangs forever with no POST**. The repository calls it "the
observed failure".

The repair was the **stall audit**: every end bounded on the wall
clock.

| the wait | now bounded at | what survives |
|---|---:|---|
| `recorder.onstop` after `stop()` | **3 s** | whatever was buffered |
| no frame has arrived for 5 s | **5 s** | ≥10 frames saved, flagged `short_take` with the reason; fewer discarded, with the reason |
| the upload fetch | **120 s** | the take stays in memory, with the Retry path |
| the Jetson seat's liveness | **heartbeat fresher than 10 s** | the sidecar ships only while the page is alive |

### 4.2 What had to be KEPT

**The partial result, and the liveness record.**

The stall is not a measurement failure. The work was **done** — the
frames existed, the take was finished — and the material was destroyed
by the *absence of an end*. So what had to be kept is:

- the **buffered frames of a take whose end never arrived**, kept with
  the reason it ended (`short_take`), rather than discarded because the
  normal end did not happen;
- the **heartbeat**, which is the only thing that distinguishes "the
  seat is busy" from "the seat is dead" — and the Jetson seat's rule is
  exactly that: it ships only while a heartbeat is fresher than 10 s.

Note what did **not** have to be kept: the audio, the score, the
transcript. None of them existed. The stall happened before any
derivation, and the material it destroyed was raw frames — the
cheapest-to-re-derive layer, which is exactly the layer you can afford
to keep when an end is uncertain.

### 4.3 What had to be RE-ANALYSED, and with what question

For the stall itself, the re-analysis is nearly free, because the
question is not about the content at all:

> **Did the end arrive? And if it did not, what did we have when it
> did not?**

That is a wall-clock question answered by a timer, plus the partial
take finished by hand. The one thing that does need re-analysis is the
**partial take's own health**: a take that stopped after 5 s of
silence is not a take, and the reason has to be recorded on it or the
next night's GC cannot tell a short take from a broken one.

### 4.4 The lesson, and the pin

**Lesson:** *an unbounded wait is a deletion with no notice.* Filed to
the portrait as a property of the **pipeline**, not of Casey.

**Pin:** every take that ended by timeout rather than by an end event,
because those are the takes whose provenance is weakest and whose
lesson has not been extracted.

---

## 5. The law the three share

> **The layer that failed is the layer that must be kept.**

Three readings of that, one per correction:

| correction | the layer that failed | the layer that must be kept | the layer that must NOT be kept |
|---|---|---|---|
| the silent-audio miss | AUDIO | the WAV, until its health is measured | the empty derivation, which cannot tell dead from quiet |
| the biggest-peak mislabel | the DERIVATION and its controls | the per-frame evidence and the controls | the label, which is the thing under suspicion |
| the stalled seats | the END | the partial result and the heartbeat | nothing — the stall destroyed the cheapest layer, and that is the one you can afford to keep |

**This is not "keep everything."** It is narrower than that and it
pays for itself. Each correction names one layer, and in two of the
three the layer is *already being kept* by the retention ladder for
other reasons. The WAV is kept because the derivation is not finished.
The per-frame evidence is kept because the score is derived from it.
The partial result is kept because the stall audit says to keep it. The
law does not add a retention class; **it decides which existing class
a failure pins.**

And it has an uncomfortable consequence, which is the point of stating
it: **the layer that failed is the layer whose re-analysis is
expensive.** The AUDIO layer is the most expensive to re-analyse. The
GC's budget is finite. So the law is a budget rule, not a slogan —
which is why the pins in §6 are the *whole* of it and not the beginning
of a new pile.

---

## 6. The pins, tonight, from this corpus

The addendum's rule is error-driven: pin the segments beneath a
conversation where the system got something wrong. Applied to what the
corpus actually contains, the pin set is this:

| # | pinned | why | the shadow work's question |
|---:|---|---|---|
| 1 | the **51 sessions whose WAV peaks at exactly 0** | the silent-audio class; the derivation is empty and emptiness is the tell | *does this segment have audio at all?* |
| 2 | **1792560597**, the only DRILLED session | the only performed take in the corpus; first-of-class by the data-class doctrine, and its class has one member | *what does a deliberate delivery carry that a first-take does not?* |
| 3 | the **sessions whose video is byte-frozen** while the speech trigger fires | the mirror image of the silent-audio class, and a different bug with a different fix: the body cannot be read, so the **mode** cannot be tagged | *is the body present, and if not, what may not be concluded about the mode?* |
| 4 | the **sonar probe** sessions, one of which returned **14 ASR words from a sine sweep** | a false transcription is a false lesson; the ASR produced words out of a stimulus | *which layer is allowed to say that speech happened?* |
| 5 | the **4 nudged sessions** the room map found | the camera moved; the map is a map of one viewpoint and the other four are a different picture | *what does a change of viewpoint do to everything derived from it?* |

**The pin count is 5 classes, 51 + 1 + 6 + 4 + 4 sessions.** In the
retention ladder's terms: the audio for pins 1, 2 and 4 is retained at
every tier; the *per-frame evidence* for pin 3 and pin 5 is retained,
not the summary.

**What is NOT pinned, and this is the point of the exercise:** the 20
sessions that carry signal and are not in a correction. Those are the
routine pile. They are what the GC collects, and §3 of
`docs/research/QUINTESSENTIAL-SELECTION.md` is the audit of what the
GC would be throwing away.

---

## 7. The shadow-work method

### 7.1 The loop

1. **A correction arrives.** It is a real failure of the system, with
   an artefact that shows it: a zero-peak WAV, a swapped-label metric
   that does not move, a take that ended by timeout.
2. **The segment beneath it is pinned**, at every layer, immediately.
   The pin is on the **LAYER that failed**, not on the segment as a
   whole — a pin on the whole segment is a pin on the cheap layers too,
   and the cheap layers are what the GC is for.
3. **Shadow work runs on the pinned material**, in the background, with
   the critique as the question. Not "re-analyse this segment" but the
   question from §6: *does this segment have audio at all?* / *is this
   label about the signal or about the session?* / *did the end
   arrive?*
4. **A lesson is extracted and filed to the person portrait.** A lesson
   is a file. It names the failure class, the question that exposed it,
   the measurement, and the instrument it belongs to (the capture chain,
   a detector, the pipeline — **not** the person).
5. **The debt clears when the lesson exists**, not when time passes.

### 7.2 What a lesson is, and where it goes

A lesson is **about an instrument**, and it is filed to the person
portrait because the portrait is where "what I have learned about this
person and about the instruments I use on them" lives. But it must be
filed with its instrument named, because:

- the silent-audio lesson belongs to the **capture chain**;
- the biggest-peak lesson belongs to the **detector**;
- the stall lesson belongs to the **pipeline**.

A lesson filed without its instrument becomes a lesson about Casey,
and that is how a system learns that a person is quiet.

### 7.3 The metrics, and the one that matters

The addendum names four: regret rate (audits), pin count, lessons
extracted, post-lesson re-ask performance (Casey spot-checks as
referee). Three of them are countable today. The fourth is the one the
whole doctrine is built on, and it is the one with no instrument in this
corpus: **no referee has graded anything yet**, which is why the tone
game's teacher-dependence ratio is **1.0** — *"the reader's training
labels all come from the performer's own priors, and no referee has
graded anything yet, so not one of its judgments is certified."*

So the honest position is:

> **The metrics that can be computed now are the internal ones (pin
> count, lessons extracted, audit yield). The metric that decides
> whether the GC is allowed to run at all — the post-lesson re-ask
> performance, at held referee-graded quality — has no referee, and
> therefore has no value yet.**

That is not a reason to delay the shadow work. It is a reason to be
explicit that the shadow work's own output is currently certified by
nobody, exactly like every other judgment in the repository.

### 7.4 The calculator this graduates into

The awareness addendum's doctrine says the fleet's real product is
calculators: cheap mechanical detectors, fitted per user, needing less
energy every generation.

Each of the three corrections has a calculator in it:

| the correction | the calculator it graduates into | what it costs |
|---|---|---|
| the silent-audio miss | a **health check on every saved WAV** — one pass, peak and nonzero count, at save time | one arithmetic pass over a file that is already being written |
| the biggest-peak mislabel | a **control on every label** — the swapped-label pairing, run with the label, every time | the same frames with the labels swapped: no new data |
| the stalled seats | a **bounded end** — every wait raced against a timer, with the partial result kept | one timer |

All three are cheap, all three are mechanical, and all three are
**per-instrument rather than per-person** — which is the point. The
library grows by one calculator per corrected failure class, and the
energy cost of each generation falls because the corpus of pinned
material grows.

---

## 8. What this does not establish

- **It does not establish that the three corrections are the three
  corrections that matter.** They are the ones Casey named. The night
  report lists others — the client's discarded `finish_reason`, the
  local path that had never worked, the copy of the thing that measures
  the thing — and the same law applies to them, but they are not
  worked through here.
- **It does not establish the miss rate of the GC.** The shadow work
  prevents a repeat by construction; whether it *catches* a new one is
  the audit's job, and the audit's measured miss rate is in
  `docs/research/QUINTESSENTIAL-SELECTION.md`.
- **It does not establish that the post-lesson re-ask performance
  improves.** There is no referee, so there is nothing to measure it
  against.
- **It does not put any of these lessons into the portrait.** The
  portrait is Casey's; three lessons about instruments are not about
  him, and the file format for a lesson that is about an instrument is
  not written yet.
- **It does not change the GC's behaviour.** ZCode owns the ladder
  plumbing and the dry-run GC report. This document is the method the
  report should implement, not the implementation.

Committed for Muse to review; not merged.
