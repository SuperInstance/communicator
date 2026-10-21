# PHONE-NODE — the second viewpoint is a clock problem wearing a sensor costume

**Seat:** mcode (tiers and decomposition)
**Source brief:** `docs/research/BRIEF-PHONE-NODE.md`
**Written:** after the corpus night's work — the clock fix, the room map,
and the sonar measurement — because all three bear on this directly.

---

## 0. THE READING I WOULD START FROM

Everything in this programme is built on one rule: **one clock**. The
session's `timestamps.json` is the spine; the syllables, the notes, the
face segments and the motion events all carry frame indices on it; and
the architecture exists so that nothing ever needs a second clock. The
free-recorder was *born* with one clock, and the old project's blocker
was that the audio and the video ran on independent ones.

A phone is a second clock. That is the whole difficulty, and it is not
a sensor difficulty wearing a costume — it is the costume.

But here is the thing I did not expect to find while reading the brief:
**the phone's second clock is the only instrument in the fleet that can
measure the fleet's geometry.** The fixed cameras all share one clock and
therefore cannot see each other's geometry at all. Two devices with two
clocks *can* — because the difference between two clocks is a delay, and
a delay against a known signal is a distance. Muse's "timing as geometry"
is not an idea that sits beside the one-clock doctrine. It is the
one-clock doctrine, taken one node further out.

So the phone node is not "another sensor". It is the fleet's first
**spatial** node, and the thing it contributes is not more pixels. It is
a second origin for the same timeline.

---

## 1. WHAT IT MAKES POSSIBLE THAT NOTHING ELSE IN THE FLEET DOES

**It converts image-plane truth into metric truth.** Tonight's room map
had to carry this clause in writing: *"this is the room as the camera
knows it — image-plane truth, not a floor plan."* Every coordinate in
it is a place in a picture. It cannot tell a door from a picture of a
door, it cannot measure a distance, and it cannot see behind anything.
One view is a projection and nothing in it can be inverted.

Two views of the same point is triangulation, and triangulation is the
whole difference between a picture of a room and a measurement of one.
The fixed cameras cannot do this with each other: they share a clock, so
their images of the same moment differ only by parallax and they have
no delay to measure. The phone, with its own clock, does.

**It gives the fleet a calibration anchor that moves.** A fixed camera's
own geometry is known only up to the camera's own uncertainty. A phone
is a beacon that the fixed microphones can localise by arrival-time
difference *and* the fixed cameras can localise by image — two
independent fixes of one point, which is the only kind of calibration
that can find out whether the calibration is right.

**It changes the teacher-dependence ratio's denominator.** The headline
metric is the fraction of judgments still needing a teacher. Today a
judgment is made on one view and one listener's reading. With a phone
node, the same moment can be judged from two views, and the
**disagreement between the two judgments is a free, unsupervised
estimate of how much of the judgment was the annotator.** That is a
measurement the programme currently cannot make at all, because it has
never had two opinions about the same thing.

---

## 2. WHAT I WOULD BUILD FIRST

Not the cameras. The cameras are the easy part and they are worth
nothing until the timing works.

**First build: the two-device time base, and nothing else.**

One page on the phone. One page on the laptop. Both record, on their
own clocks:

- a **sync event** — a flash on the phone's screen and a click from the
  same page, captured by the laptop's camera and the laptop's mic;
- a **chirped marker** in a band the chain actually carries.

And then, on the laptop, the measurement I already know how to do: the
offset between the two clocks, measured from the recordings, with the
uncertainty stated. This is the same shape as tonight's audio lead — I
measured the WAV's edge against the first frame to within 12 ms, by
comparing the two records against each other, and the same move applies
here with a much bigger prize.

The point of building this first is that it is **falsifiable in an
afternoon** and it is the load-bearing piece. Every other idea in the
brief — the dual cameras, the chirp train, the motion trial — is
meaningless until two devices agree on when something happened.

And I would build it with the same discipline the corpus already has:
**the arrival-time measurement and the image measurement must be two
independent fixes of the same point, and their disagreement is the
error bar.** If the phone's position from arrival times and its
position from the fixed camera's image of the phone disagree by more
than the stated uncertainty, the whole thing is wrong, and the
disagreement says which half is.

**Second build, and only if the first holds: the phone as a moving
beacon in the room.** The phone plays a marker in a band the chain
carries and the fixed microphones localise it by arrival-time
difference. The phone's two cameras are a *bonus* on top of this — they
give the phone's own view of the room, which is the first-person view
nobody in the fleet has.

**Third: the dual-camera capture.** Simultaneous front+rear capture in
a mobile browser is the least certain thing in the brief, so it goes
last and it is gated on the first two working without it.

---

## 3. WHAT MEASUREMENT WOULD PROVE IT

The proof is not "the phone node works". The proof is one number.

**THE CLOSED LOOP.** Pick a moment when the phone is visible to a fixed
camera and audible to a fixed microphone. From the audio alone, compute
the phone's position by arrival-time difference. From the image alone,
compute the phone's position by the camera's own geometry. Compare.

| quantity | what it is |
|---|---|
| the position from arrival times | metric, from the delay between the phone's mic and the room's mics |
| the position from the image | metric, from the phone's visible position in a fixed frame |
| **the disagreement** | **the error bar of the whole spatial claim** |
| the agreement over many such moments | whether the error is a fixed bias or noise |

Two independent measurements of one point is the only honest test. If
they agree over a range of positions, the spatial node is real. If they
disagree by a constant offset, the clocks are offset and that is
fixable and interesting. If they disagree randomly, the timing claim is
noise and the lane is dead.

**And the measurement that decides whether the beacon is possible at
all.** Tonight's sonar result says the ultrasonic idea is dead on this
hardware: a 16-22 kHz sweep came back **87 to 94 dB below the
stimulus**, sitting 0.26 dB above the microphone's own floor, and the
"22-24 kHz usable band" in the probe's own manifest was an artifact of
dividing the floor by a near-zero reference. The brief hoped the chain
would carry near-ultrasound; measured, it does not.

So the beacon has to live in a band the chain carries, and the chain
carries **voice**. That means the beacon is a **voxels** — a chirped
voice-band marker — not a chirp at 20 kHz. The probe's own pink burst
came back at 0.17 dB above the floor in the ultrasonic band; the same
burst in the voice band is in the middle of the spectrum the microphone
hears perfectly well. This is not a compromise. It is the measurement
choosing the band, which is what the brief asked for ("the usable band
on this hardware is whatever survives the chain — measured, never
assumed").

**A second proof, and it is the one that would earn the money: does the
phone node lower the teacher-dependence ratio?** The programme's
headline metric is the fraction of judgments still needing a teacher, at
held referee-graded quality. If a second viewpoint makes the machine's
judgment match the teacher's more often — and the disagreement between
the two views is what tells you *which* judgments are shaky — then the
ratio falls for a reason that can be named. If it does not fall, the
node is carrying pixels and no new information, and that should be
said plainly.

---

## 4. WHAT KILLS IT

**The clock, and it is the real one.** Two devices, two clocks, and the
sync event has to be sharp enough to discipline them. Tonight's audio
lead work says how sharp: the WAV's edge sat 71 ms from the first frame
and the best estimator reached 12 ms. A phone node needs milliseconds
across two devices, and if the sync event cannot be made sharp — a
flash is a few ms of exposure, a click is a few ms of audio, and both
have to be seen by a camera at an unknown distance — then the position
error follows the clock error straight back and the whole thing
degrades into a picture with extra steps.

**Simultaneous dual-camera capture.** The brief flags it and the flag is
right. Two cameras at once on a phone browser is a capability question
with an OS-dependent answer, and if the answer is "not on this phone",
the front+rear idea quietly becomes one camera, and the node's value
drops to what the audio already gives.

**The phone moves, so its fixes are a track, not a point.** A position
from arrival times is a position at an instant. The phone's own motion
between the emission and the arrival is an error term that grows with
speed and distance, and it is not a constant, so it cannot be calibrated
away once. The fix is a phone that is still, or a clock disciplined
enough to interpolate across the motion — which brings the clock
difficulty back with a second copy.

**The acoustic channel may not support it even where the ultrasound
did not.** A voice-band beacon is audible, and it is in a room with a
person in it, and it is a 20-second stimulus. The brief's own safety
rule ("modest playback level, short duty cycle") is there because
near-ultrasound is not inaudible to everyone and not to dogs; a
voice-band beacon is inaudible to nobody. If the beacon has to be heard,
the duty cycle has to be short, and a short burst gives a
shorter integration and a noisier position — the trade that closed the
ultrasonic lane closes this one too, just less obviously.

**And the one that is most likely, if I am honest: the fixed cameras
may already be enough.** The room map needed a second angle to become a
floor plan, but the room map's job is to say where the room's life
happens, and the presence map does that in one view. If the phone node's
first contribution is a better picture of one room, it is competing
against a camera that costs nothing and already runs.

---

## 5. THE ONE SENTENCE

The phone node's value is not that it sees more; it is that it is the
first node in the fleet whose **clock disagrees**, and a clock that
disagrees is a ruler — and the whole programme has been built, night
after night, on the rule that there must be only one clock. This node
breaks that rule on purpose, and in exchange it measures the geometry
that the one-clock fleet has never been able to see.

The build order is the proof order: two clocks first, a disagreement
second, and only then a room.

---

*Written for the overnight ideation round. Muse synthesises at dawn.*
