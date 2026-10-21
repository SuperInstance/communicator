# SEED-ASCII — the seed's four movements are four ways of deciding what the score has to carry

**Seat:** mcode (tiers and decomposition)
**Written:** 2026-10-21, overnight, after the room map and the sonar.
**Status:** written before the mail landed. See section 0.

---

## 0. WHAT THIS IS WRITTEN AGAINST, AND WHAT IS MISSING

The overnight queue says the mail lands at
`docs/research/ideation/SEED-EMAIL.md` in the main checkout, and that the
mail runs *vision-identity landscape, conversions, acoustics, then ASCII
crossovers*, with my lens being **how that tech feeds the score designs**.

**That file has not landed.** The ideation directory in the main checkout
is empty, as it was when this was written. So this riff is written against
four section headings and against the state of the programme as it stands
this morning — not against the mail itself. Everything below is an
argument about where those four movements *land* in the score designs, and
it should be read against the mail when it arrives: where the mail says
something different, the mail is right and this is wrong.

What this riff is not is a summary of the mail. It is a claim about the
score designs that the mail's four movements, read in order, keep pointing
at.

---

## 1. THE QUESTION UNDER ALL FOUR MOVEMENTS

The design law for this programme is already written down, and it is
short: **the quantized score — MIDI plus parameters — is the portable
currency between tiers.** A weak node ships scores up when a link appears;
a strong node ships nothing. Decompose over monoliths; every stage small,
old, inspectable, replaceable; visible seams. And the corollary that makes
the whole thing a design rather than a pipeline: **quantization is the
feature, not the loss.** Once prosody is parameters, what sticks out
becomes findable — and exaggerable — and the parameters that stick out
*are* the identity.

So the score designs are not the middle of the pipeline. They are the
**specification of what crosses the boundary**, and every piece of
technology in this programme is really a candidate answer to one question:

> which parameters belong in the score, and at what resolution do they
> survive the trip?

Read in that order, the mail's four movements are four attacks on that
question from four directions. The vision-identity landscape says what the
parameters are *for*. The conversions say what the parameters *cost*. The
acoustics say what the parameters *omit*. The ASCII crossovers say what
the parameters *survive*.

---

## 2. THE VISION-IDENTITY LANDSCAPE, AND THE PARAMETER IT NAMES

The programme already has the instrument for this, and it was built last
night: the person profile, with its per-class baselines and its drift
measurement. It keeps a baseline for DRILLED data and a baseline for
AMBIENT data separately, measures each session against its own class, and
never averages the two together.

**That discipline is already a parameter-selection rule, and it has not
been stated as one.** A parameter is worth carrying if the difference
*between* classes beats the difference *within* a class. That ratio —
between-class variance over within-class variance — is a number the
profile already computes, per parameter, per class. It is the natural
criterion for choosing which parameters the score carries, and it is
strictly better than "the parameter that moved the most", which is what a
design that has not thought about classes would use.

The landscape's contribution is that identity is not one axis but a
*space*, and the caricature mechanism in the tiers brief says what to do
with a space: give the performer a knob per parameter and report accuracy
against exaggeration per intent, so that the reader reads a caricatured
intent *more* reliably. Measure the curve; do not assume it.

The feed into the score designs is therefore concrete. **The caricature
knob belongs on the parameters the profile says separate the classes, not
on the parameters that are merely large.** A knob on an occasion parameter
is a noise amplifier with a dial.

And there is a second feed, which is the one I think is the sharpest thing
in this riff. The room map made a discovery last night that belongs to
this section and not to any other: over the capture, the person is in the
most common view of the room, as a **smudge** — eleven luma levels
darker than the empty room where his head and shoulders are, and brighter
than the empty room everywhere else. The identity is legible in the
picture only as a residue of itself. A portrait built from the median
frame would be a portrait of a smudge.

**That is what a quantized score is too.** The score's parameters are the
smudge of the reading. The design law's "the parameters that stick out
ARE the identity" is the same statement as the smudge: what survives the
quantization is what was in front of the camera all along. The caricature
knob and the dither ladder are the same instrument pointed at two
carriers.

---

## 3. THE CONVERSIONS, AND THE NULLS THEY NEED

Every step in this programme that turns one representation into another
is lossy, and the house rule is that the loss gets **measured, not
assumed**. The dither characterisation is the exemplar, and it is worth
reading as a template for everything else in the programme because it was
run honestly:

| the rule | the outcome |
|---|---|
| the calibration must reproduce the observed near bands within 0.5 pp | **FAILS** — and the failure is reported as the finding |
| the excess is motion only if the empty room does not reproduce it | the empty room's far band is 0.0000% against the capture's 0.31% |
| the channel with the larger excess over the null is the one to build on | the mask — so "the tone channel, not the mask" is **withdrawn** |

A plausible fix, killed by the control built specifically to test it, is
called the cleanest negative result in the lane. That is the standard this
programme holds itself to, and the score designs are not exempt from it.

**The feed is that the conversion into the quantized score is itself a
conversion, and it has never been given a null.** The score is built by
taking a continuous reading and quantizing it to notes and parameters.
Nobody has asked what a *null* reading quantizes to, in the same way the
dither lane asked what an *empty room* renders as. The empty-room null is
the one that made the dither result trustworthy, and the score pipeline
does not have its equivalent.

What the null would be is easy to say and worth saying precisely: **the
same quantized score, drawn from a session in which the person was not
there.** The corpus now has one, by accident. Session `1792560773` is
silent — sixteen seconds, 934 frames, a measured peak RMS and no words at
all. It is a reading with no content in it, and its quantized score is the
programme's first natural null.

There is a second, sharper feed here, and it is about what the conversion
is allowed to conclude. The ASCII lane ended with the sentence that
should be the programme's motto: *a demonstration that motion is visible
is not a demonstration that motion is sufficient.* That is the
teacher-dependence ratio's own claim in another language. A score that
carries identity, visibly, in its parameters, is not yet a score that
carries identity *for the judge*. The conversion tells you the first; only
labelled data and a held referee tell you the second.

---

## 4. THE ACOUSTICS, AND THE ROOM AS A SCORE PARAMETER

The sonar closed last night with a number instead of an answer. The
channel above 15 kHz is not measurable on this hardware: the return sits
87 to 94 dB below the stimulus, the probe's own "22-24 kHz" verdict was an
artifact of dividing by a stimulus that never played there, and a room
measurement does not need ultrasound, because the microphone hears the
room perfectly well below 8 kHz.

**The acoustic parameter of a reading therefore has to be measured in the
voice band, and the room map says the same room from the picture side.**
The room's life is held in one zone of the frame and that zone is a
person; thirty-six percent of the picture is a place where nothing ever
happened for two hours; and the camera is not quite fixed, because the
laptop was nudged once and then twice during the night.

Put the two together and the score designs have a hole in them. The
quantized score carries notes and prosody parameters, and it carries
**nothing about where the reading happened**. But the same words, read
with the same prosody, in an empty hard room and in a soft furnished one,
are two different performances, and the difference is audible to anyone
who was in both rooms. A score that cannot say which room it was read in
cannot be compared across rooms, and the programme's ambition is to
compare readings across nights, across devices, across tiers.

The honest form of the fix is a constraint, not a feature: **the room is a
context on the score, not a property of the note.** The same note in the
same room at five in the morning and at two in the afternoon is the same
note; a score that folded the room into the note would make the two
incomparable in a way that is harder to undo than to add. A context
travels with the score and does not pretend to be part of it.

And the first context field is measurable tonight, because the room map
measured it: the occupancy of the room at the moment of the reading — the
fraction of frames in which something stood in front of the camera, and
how much of the frame it stood in. Not the room's identity; the room's
*state*. That is a number the camera can produce, it is already in the
tool, and it is the thing that distinguishes a reading made at a desk
from a reading made at a desk with somebody in it.

---

## 5. THE ASCII CROSSOVERS, AND WHY ASCII IS THE RIGHT TEST

The ASCII crossovers are the last movement, and they are the one that
closes the argument, because ASCII is the only carrier in this programme
with **no semantic content at all**.

That is exactly why it is the right instrument for testing a score
design. Everything else in the programme — audio, video, text — is
*about* something, and a converter that understands its input can smuggle
meaning across the boundary that the parameter set never carried. ASCII
cannot smuggle. It is a grid of characters. Whatever survives the
conversion to ASCII is exactly what was in the score.

**So the ASCII crossover is a compression test for the score design, and
it is the cheapest possible one.** It needs no fleet, no teacher, no
labelled data, no GPU and no internet. It runs on the artifacts the
corpus night's work already produced.

The feed into the score designs is therefore a test with a shape:

> **If the caricature knob does not survive the ASCII crossover, then the
> identity is not in the score, and the knob is decorative.**

That is a falsifiable claim about the design law, made with the cheapest
instrument available, and it can be run on the three real sessions
tonight. `1792560597` quantized to 17 words, 19 syllables, 29 notes, over
frames 1158 to 1623 of 1626. `1792560758` to 229 words, 294 syllables, 288
notes. There are the two score objects, one DRILLED and one AMBIENT, and
there is the question of whether the thing that makes them different is
in the score or only in the room they were read in.

The ASCII lane's own findings are the reason this test is sharp rather
than decorative, and they should be carried into it. Roughly half of the
raw delta in the ASCII carrier is threshold dither — dots on the threshold
flip at 67.25% — and the residual below the threshold is structured, at
21.61% against a 34.62% independence baseline, which means content moving
coherently is recoverable. But the tone channel is a dead end: 70.91% on
the control, worse than the mask, and the proposed fix was withdrawn. Any
ASCII crossover of a score must therefore be built with **hysteresis
sized by the measured gradient** — hold a bit until the value crosses the
threshold by a margin — or it will measure its own dither and call it
signal.

---

## 6. THE ONE FACT THAT BELONGS TO BOTH LANES

Two lanes, two nights, one fact.

The ASCII lane ended with: *the capture needs per-frame checksums. Every
stage of this investigation was contaminated by not knowing which frames
were real.* And last night's room map hit the same wall from the other
side, and found the reason: **two thirds of the frames in the corpus are
byte-identical repeats**, because the recorder writes a frame only when
the picture changed. 42,221 of the 127,527 frames at the mapped viewpoint
were a new picture; the other 85,306 were the frame before, again.

The same defect, found twice, in two lanes, is not a coincidence. It is a
corpus-level fact: **the corpus does not record which frames are new.**
And the consequence is uniform — every motion-shaped measurement taken
over this corpus is diluted by the duplicates unless its author happens to
notice and divide by the number of changes instead of the number of
frames. The room map had to make that correction itself, and it can only
be a correction, because the underlying record is missing.

The fix belongs at capture time and it is small: write the frame index and
a checksum with each frame, and the deduplication stops being a question
every lane has to re-ask. This is the single highest-value change to the
recorder that I know of from this week's work, and it is worth saying so
plainly in the mail's own language: it is a null, and the programme's rule
about nulls is that they get built before the measurement, not after it.

---

## 7. WHAT I WOULD BUILD FIRST

**The score-parameter survival test.** One script, local, no GPU, no
labelled data:

| step | what it does | what it measures |
|---|---|---|
| 1 | take the quantized scores of the three real sessions | the score objects that already exist |
| 2 | render each score's parameter set to the ASCII carrier, with hysteresis sized by the measured gradient | what the score actually carries |
| 3 | round-trip: reconstruct the parameters from the ASCII and compare | the survival fraction, per parameter |
| 4 | compare the DRILLED session's survival against the AMBIENT ones' | whether identity is in the set that survives |

Step 4 is the one that matters and it is the one with a real chance of
coming out negative, which is what makes it worth running. If the
parameters that separate DRILLED from AMBIENT are precisely the ones the
ASCII carrier cannot carry, then the score designs need to change, and
that is a finding worth more than a demo.

---

## 8. WHAT MEASUREMENT WOULD PROVE IT

The programme's headline metric is the **teacher-dependence ratio**: the
fraction of judgments that still need a teacher, at held referee-graded
quality. It has to be the metric here too, and it has to be paired with
the exaggeration curve, because the caricature knob's claim is specific:
a caricatured intent should be read *more* reliably, and the curve should
have a rising limb before it falls.

| the claim | the measurement | what would falsify it |
|---|---|---|
| identity is in the score's parameters | the between-class / within-class variance ratio, per parameter | a parameter set where every axis is occasion |
| the caricature knob amplifies identity | teacher's reading accuracy against exaggeration, per parameter | a flat or falling limb |
| the score survives the tier boundary | survival fraction per parameter through the ASCII carrier | survival below the point where the knob is readable |
| the whole thing reduces teacher dependence | teacher-dependence ratio at held quality, before and after | a ratio that does not fall |

The held referee grading is not optional in any of the four rows. The
judge cannot certify itself, and the same applies to a measurement of the
judge's dependence on a teacher.

---

## 9. WHAT KILLS IT

* **The parameters that stick out are the occasion, not the identity.** If
  the profile's variance ratios say that every parameter is within-class
  noise with a class mean that is really the speaker's mood, then the
  caricature knob amplifies nothing but mood, and the teacher-dependence
  ratio will not move.
* **The ASCII crossover kills the same parameters the knob needs.** Then
  the phone tier kills them too, and the score designs have to be
  redesigned around what survives rather than around what is expressive.
* **The conversion's null keeps failing.** The dither calibration still
  fails to reproduce the observed near bands, and that failure is
  reported rather than resolved. Until it is resolved, no parameter
  measurement built on that conversion is trustworthy, and the survival
  test inherits the doubt.
* **One reading is not a class.** The profile has one DRILLED session and
  two AMBIENT ones. A between-class variance computed over one member of
  a class is not a variance, and the criterion in section 2 cannot be
  trusted until there is a second drill-mode delivery. **A potato-drill
  delivery is the blocking input for this whole riff.**
* **The room turns out to be the whole story.** If the DRILLED reading
  differs from the AMBIENT ones because of where it was read rather than
  how, then the score designs are measuring the wrong thing, and the
  right fix is the context field of section 4, not a better parameter.

---

## 10. THE ONE SENTENCE

The seed's four movements are four ways of specifying what the score has
to carry across the tier boundary, and the cheapest way to test that
specification is to convert the score into a carrier that cannot
understand it and see what is left.

---

Committed for Muse to review; not merged.
