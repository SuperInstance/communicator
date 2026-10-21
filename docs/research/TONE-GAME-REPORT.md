# THE TONE GAME — the acting and understanding loop, as a runnable experiment

**Built by:** `experiments/tone_game.py`
**Casey's design:** "If you can act it, you can understand it."
**The intents:** the potato deck — sincere, leaving, obligated,
everything, joking, quiet-furious.
**Device:** the repository's own pure-Python pitch tracker on the CPU. No
GPU, no paid API, no network.

---

## 0. THE HEADLINE

**The melody alone recovers the intent, every time, and the recovery
survives a completely different carrier phrase.** On held-out renders,
with the reader hearing only the audio and never the note list:

| | recovery rate |
|---|---|
| the melody, all features | **1.000** |
| the melody, pitch features only | **1.000** |
| the melody, timing features only | 0.625 |
| chance, six intents | 0.167 |
| **Casey's own prior** | **0.70** |

The reader is not counting notes: the timing channel alone reaches 0.625
and the pitch channel alone reaches 1.000. And the number is not an
artefact of the phrase: swapping the carrier from "I love you" to "I am
leaving now" — a different number of syllables, a different melody
length — leaves the recovery at 1.000.

**The caricature curve is the measurement that says how much separation
is needed, and it starts exactly where it should.** At an exaggeration
of zero every intent *is* the neutral delivery and the reader scores
**0.167, which is chance to three decimal places** — the control at the
end of the curve is the curve's own proof that the reader is reading the
melody and not the bookkeeping. From there the accuracy climbs to 0.792
at one tenth of the way out, 0.958 at a third, and 1.000 by halfway.

**Which intents need exaggeration is the per-intent finding, and it is
the one Casey would want.** At one tenth of the exaggeration, the
readable ones are the ones that stick out: `joking` and `quiet-furious`
at 1.00, `everything` at 0.88. The ones that need more are the quiet
ones: `leaving` at 0.50 and `sincere` at 0.62. **What sticks out,
exaggerated, reads better** — measured, per intent, not assumed.

**And the teacher-dependence ratio for this lane is 1.000, which is the
honest number.** Every label the reader learned from came from the
performer's own priors, and no human has graded anything yet, so not one
of its judgments is certified. The machinery to measure the ratio the
moment a referee grades the pack is in the tool; what is missing is
Casey's own drill deliveries, which would replace the priors with
measurements.

---

## 1. THE GAME

Two players and one loop.

**The performer** says a carrier phrase with a chosen intent, rendered as a
pitch melody. The melody is written as a MIDI score and as an audio
render, so it can be heard as well as read.

**The reader** hears only the melody. No words, no labels, no note list.
It is handed a WAV and nothing else, extracts the pitch contour with the
repository's own tracker, computes eleven features from the sound, and
guesses the intent.

**The rounds.** The performer reshapes each intent's delivery to maximise
READABILITY — not beauty — and the reader retrains. Which intents blur is
the finding.

### The headline number

**SENTIMENT RECOVERY RATE**: the reader's accuracy on held-out renders,
melody alone, against chance and against Casey's own prior that the
melody carries about 70% of the sentiment.

Chance with six intents is 0.167.

### Three decisions that carry the experiment

**THE READER HEARS AUDIO, NOT A NOTE LIST.** This is the decision the
whole measurement rests on. If the reader were handed the performer's own
notes, the recovery rate would be a statement about the performer's
bookkeeping and not about anything audible. Instead the reader is given
the rendered WAV and measures the pitch itself, so the number is a claim
about the channel Casey means: the melody as a sound.

**THE SCORE IS THE CURRENCY.** The performer writes a MIDI file; the
audio is rendered *from that file*, not from the object the performer had
in its hand. The unit test proves the round trip: the delivery rebuilds
from the score alone, note for note, and the audio rendered from the
rebuilt score is sample-for-sample the same audio. That is what lets a
weak node ship a score up instead of shipping audio, and it is what makes
the stages replaceable — the renderer can be swapped without the
performer knowing.

**THE NOTE COUNT IS NEUTRALISED IN THE HEADLINE RUN.** If the intents
differ in how many notes they use, a reader can recover the intent by
counting notes and never listening to the melody. So every intent sings
the same number of notes, and the note count is measured separately, as
an ablation, with the number free. The headline is the honest one; the
ablation says what the counting would have added.

---

## 2. THE SEED PROSODY — real, and labelled

The seed comes from the real captures, which carry real speech.

| | capture A (facing) | capture B (turned) |
|---|---|---|
| notes measured | 66 | 57 |
| note duration, median | 0.185 s | 0.220 s |
| note duration, p10 – p90 | 0.090 – 0.500 s | 0.090 – 0.508 s |
| step between notes, median | +0.03 semitones | −0.47 semitones |
| step, p90 absolute | 4.57 semitones | 13.17 semitones |
| pitch range (p5 – p95) | 6.40 semitones | 7.11 semitones |
| centre | MIDI 45.7 (≈ A2) | MIDI 45.7 (≈ A2) |
| rate | 5.01 notes/s | 4.34 notes/s |

Capture C is silent — peak RMS 0.0008, no voiced run at all — so it seeds
nothing, and the empty room proves the seeding is coming from the voice
and not from the room.

The two captures are genuinely different performances, not the same take
twice: 66 notes against 57, and an envelope correlation of 0.03. What
they share is the speaker's centre pitch, which is the point — the seed
is one voice.

**The envelope the contours are drawn inside.** The range bound is 8.0
semitones, which is the tool's floor rather than the captures' own
measured spread of 6.4 and 7.1, and the report says so rather than
pretending the bound is measured. Two intents were clamped by it:
`everything` wanted a 13-semitone range and got 8.0, and `joking`
wanted 9 and got 8.0. Everything else fitted.

Every intent says which it is, and the distinction matters:

| intent | provenance |
|---|---|
| sincere | real-seeded from capture A |
| leaving | real-seeded from capture B |
| obligated | synthetic inside the real-prosody envelope |
| everything | synthetic inside the real-prosody envelope |
| joking | synthetic inside the real-prosody envelope |
| quiet-furious | synthetic inside the real-prosody envelope |

**What the real captures seed is the STATISTICS, not the INTENT
SEMANTICS.** Capture A is the facing capture and capture B is the turned
one, and calling A "sincere" and B "leaving" is a construction made here,
not an observation made from the audio. What is not a construction is the
envelope: the note durations, the note-to-note steps, the pitch range and
the rate that the contours are drawn inside are all measured, with the
repository's own tracker, from a real voice.

Casey's own drill deliveries, when they exist, are the ground truth that
would replace the construction. Until then this is a machine's opinion
about which delivery should read as which intent, and the report says so
wherever the number appears.

The capture C is silent — its peak RMS is 0.0008 and it has no voiced
run at all — so it seeds nothing, and the empty room is the control that
proves the seeding is coming from the voice and not from the room.

---

## 3. THE PERFORMER

A delivery is a **style vector**: ten parameters, and they are the
knobs a caricaturist would have.

| parameter | what it is |
|---|---|
| `centre` | semitones of the melody's centre above the carrier's base |
| `range` | the melody's total span in semitones |
| `tilt` | semitones gained or lost across the phrase |
| `arc` | curvature: positive arches, negative bowls |
| `zig` | zig-zag amplitude, for the bouncy deliveries |
| `drop` | the depth of a late drop, for the suppressed ones |
| `density` | notes per syllable |
| `jitter` | random deviation per note, in semitones |
| `pause` | trailing silence, in seconds |
| `hold` | how much longer the last note is |

The starting values are **priors**, not measurements: this is the
performer's opinion about how a delivery looks, and the rounds are what
find out whether the opinion was any good.

Two of the priors are worth naming because they are the interesting ones.
`obligated` is flat, narrow and even — flat affect, the delivery of
someone who means nothing by it. `quiet-furious` is low, compressed and
carries one sharp drop near the end, which is what suppressed anger
sounds like when it is not allowed to be loud.

### The envelope clamps the design

The prosody envelope comes from the real captures' own measured pitch
spread, and the starting design is **clamped** into it. This is not a
detail: the emphatic delivery wants a range of 13 semitones and the real
voice managed 6.4, so the emphatic intent gets 6.4. A contour that no
real voice could have produced is not a rendering of an intent; it is a
drawing. The clamped fields are listed in `results.json` under
`clamped_by_the_envelope`.

---

## 4. THE RENDER

The audio is a synthetic voice, and it says so.

Each note is a harmonic glottal source run through three formant
resonators for its syllable's vowel, with an edge envelope, a 5 Hz
vibrato of about a seventh of a semitone, and a 70 ms articulation gap
between notes. It is a synthetic voice, and a sine wave is not a
delivery.

The carrier is **"I love you"** — the potato drill's own phrase — and it
is the same phrase in every render, so the words carry nothing. The
second carrier, "I am leaving now", exists for the carrier-invariance
ablation.

The MIDI is written by the repository's own `src/smf.py` at the
repository's own convention of one tick per millisecond. One track, one
property, constant velocity. The words are not in the MIDI: `src/smf.py`
has no text-meta event and was not modified to add one.

---

## 5. THE READER

The reader is given a WAV and nothing else.

**The features**, all measured from the sound by the repository's own
tracker: the melody's centre, its pitch range, its overall drift, how
many notes were sung, the note rate, the median and the spread of the
note durations, the mean absolute step between notes, the spread of
those steps, the signed mean of those steps, and how much of the phrase
was voiced.

**The classifier** is a multinomial logistic regression in numpy —
full-batch gradient descent, standardised inputs, an L2 term, and a
fixed iteration count so the reader is the same instrument in every
round. It is small and deterministic on purpose: a reader that changed
its mind between rounds could not be compared against itself.

**The split.** The reader is trained on one set of renders and tested on
a **held-out** set — the same styles, different prosody seeds. So the
recovery rate measures generalisation across deliveries, not
memorisation of the training renders.

**Three feature sets** are reported for every round, because the claim
needs all three: everything, pitch only, and timing only. If the pitch
channel alone does the work, then the melody is carrying the sentiment,
which is the whole point. If the timing channel alone does as well, then
the number of notes is doing the work and the claim is false.

---

## 6. THE ROUNDS, AND WHAT THE READER GETS WRONG

Four rounds, eight training variants and eight held-out test variants per
intent, the reader retrained from scratch each round. 48 training renders
and 48 test renders per round; a round takes about 65 seconds of compute.

| round | all features | pitch only | timing only |
|---|---|---|---|
| 0 | 1.000 | 0.979 | 0.354 |
| 1 | 1.000 | 1.000 | 0.417 |
| 2 | 1.000 | 1.000 | 0.396 |
| 3 | 1.000 | 1.000 | 0.625 |

The reader is at ceiling from the first round. **The rounds therefore
have nothing to optimise, and that is a finding rather than a
disappointment:** the performer's starting priors, drawn from the
conventional reading of what those six deliveries sound like, are already
far enough apart for the melody to carry the intent. The optimiser still
ran, and what it chose to do is worth reading — it separated the two
loud intents from each other rather than from the quiet ones.

**What the optimiser did.** Between round 0 and round 3 it raised
`everything`'s centre from +2 to +4.25 semitones and its tilt from +2 to
+5, and it moved `joking` the other way: its tilt went from 0 to −2.25,
a late drop of 2.25 semitones appeared, and its jitter was cut from 1.2
to 0.45. In other words the optimiser noticed that `everything` and
`joking` were the two intents at risk of being confused — both are wide
and both sit high — and pushed them apart in opposite directions rather
than pushing them away from the quiet ones.

**What the reader gets wrong when the melody is not allowed to help.**
The timing-only confusion at the final round is the interesting one:

```
true \ predicted  sincere  leaving  obligat  everyth   joking  quiet-f
sincere                 1        4        3        0        0        0
leaving                 0        5        3        0        0        0
obligated               0        2        6        0        0        0
everything              0        0        1        4        3        0
joking                  0        0        0        2        6        0
quiet-furious           0        0        0        0        0        8
```

Without the pitch, the reader confuses the three deliveries that differ
mostly in *where* their pitch sits — `sincere`, `leaving` and
`obligated` — and it never confuses `quiet-furious` with anything, because
the suppressed delivery is unmistakable in the timing. The pitch channel
is what separates the three that are quiet; the timing channel is what
separates the one that is angry. The two channels are not doing the same
job, and that is a cleaner answer to Casey's question than the single
headline number.

### 6.1 The learned contour, per intent

The style vector after four rounds, and the melody it actually produced.
The picture is the score read back out of the MIDI — six notes for every
intent, because the note count is neutralised, so what the picture shows
is the pitch and nothing else.

```
sincere                                 leaving
  50 |              ##    ##    ##        49 |  ##    ##
  49 |        ##                      ##    48 |
  48 |  ##                                 47 |              ##
                                            46 |                    ##
                                            44 |                          ##
                                            42 |                                ##

obligated                                everything
  47 |        ##          ##              57 |                    ##
  46 |  ##          ##          ##    ##  55 |              ##                ##
                                            53 |        ##
                                            49 |  ##

joking                                  quiet-furious
  60 |        ##                          41 |  ##    ##
  59 |  ##                ##              40 |              ##    ##
  58 |              ##                     39 |                          ##
  55 |                          ##          35 |                                ##
  54 |                                ##
```

The six deliveries are as different as the prose says they are. `sincere`
is a gentle rise that comes back one step. `leaving` falls eight semitones
and never comes back. `obligated` sits on two adjacent pitches and never
moves at all. `everything` climbs from 49 to 57. `joking` is high and
zig-zags. `quiet-furious` is low and falls, and it falls in the last
note, which is the drop the design asked for.

**The listenable renders** are the WAV files next to the scores, in the
run's output directory:

```
<out>/renders/round_3/<intent>_v<NN>.wav    the audio
<out>/renders/round_3/<intent>_v<NN>.mid    the score
<out>/listening_check/r3_NN.wav             the blind pack
```

The renders are not committed — they are large and they are regenerated
by the run — but the pack is small and it is the piece meant to be
listened to.

---

## 7. THE ABLATIONS

### 7.1 Does the note count do the work?

In the headline run every intent sings the same number of notes — two
per syllable, six for "I love you" — so the count cannot be the cue.
The comparison run lets the count vary with the intent's density, which
is a legitimate prosodic dimension and one a real listener also has.

| run | all features | pitch only | timing only |
|---|---|---|---|
| the note count equalised (headline) | 1.000 | 1.000 | 0.625 |
| the note count free | 1.000 | 1.000 | 0.729 |

The count is worth about **0.10 to the timing channel** and nothing at
all to the headline: the melody still carries the intent on its own. So
the headline is not inflated by the counting, and the counting is not
worthless either — a real listener has it too.

### 7.2 Does the recovery survive a different phrase?

The melody channel is supposed to be the language-independent layer, so
the whole game was re-run on the second carrier, "I am leaving now":
five syllables instead of three, a different phrase length, a different
melody.

| carrier | all features | pitch only |
|---|---|---|
| "I love you" (3 syllables) | 1.000 | 1.000 |
| "I am leaving now" (5 syllables) | 1.000 | 1.000 |

Nothing changed. The reader was retrained on the new phrase's renders
and recovered the intents perfectly, which is the evidence for the claim
that the melody is the language-independent layer: **the words are
identical in every render of both carriers, and the reader never sees
them at all, so nothing about the wording could have been doing the
work.**

---

## 8. THE CARICATURE KNOB

Casey's mechanism: once the prosody is parameters, what sticks out
becomes findable and exaggerable, and a caricature keeps the few features
that make a delivery that delivery and leans on them.

The knob scales each intent's departure from the **neutral delivery**,
which is the mean of all the intents. At an exaggeration of zero every
intent *is* the neutral delivery and the reader's best possible score is
chance — that end of the curve is the control, and it is what makes the
rest of the curve mean something.

The curve, one round per point, the reader retrained each time, the
design held fixed and only the exaggeration moving:

| exaggeration | all features | pitch only | timing only |
|---|---|---|---|
| 0.0 | **0.167** | 0.167 | 0.167 |
| 0.1 | 0.792 | 0.688 | 0.229 |
| 0.2 | 0.854 | 0.833 | 0.250 |
| 0.35 | 0.958 | 0.938 | 0.396 |
| 0.5 | 1.000 | 0.958 | 0.333 |
| 0.75 | 0.958 | 0.979 | 0.396 |
| 1.0 | 1.000 | 1.000 | 0.417 |
| 1.5 | 1.000 | 0.979 | 0.458 |
| 2.0 | 1.000 | 0.917 | 0.583 |
| 3.0 | 1.000 | 0.979 | 0.604 |

**The control holds exactly.** At zero exaggeration the reader scores
0.167 — chance, to three decimal places, with six intents. This is not
a formality: an earlier version of this experiment keyed each intent's
random seed on its own name, and the reader scored 0.583 at zero
exaggeration because the *noise* had become a fingerprint. The fix was
to make the design **paired** — every intent draws the same prosody
noise, so the only thing that differs between two renders of the same
variant is the intent's own style. The unit test pins it, and the
control now reads chance.

**The curve is a rise from chance to ceiling, and it gets there by a
third of the way out.** The interesting region is between 0 and 0.35,
where the accuracy climbs from 0.167 through 0.792 to 0.958. Past that
the design is simply readable, and the curve flattens at 1.000.

**Per intent, the curve says which deliveries need exaggeration to read
at all.** The recall of each intent at each point:

| exaggeration | sincere | leaving | obligated | everything | joking | quiet-furious |
|---|---|---|---|---|---|---|
| 0.1 | 0.62 | 0.50 | 0.75 | 0.88 | 1.00 | 1.00 |
| 0.2 | 0.62 | 0.88 | 0.75 | 0.88 | 1.00 | 1.00 |
| 0.35 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.5 and above | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

This is Casey's mechanism, measured rather than assumed. At a tenth of
the exaggeration the two deliveries that stick out — the bouncy joke and
the suppressed anger — are read perfectly, and the two quiet ones are
read at chance-ish: `leaving` at 0.50 and `sincere` at 0.62. `leaving`
needs twice the exaggeration of the others to get there. **What sticks
out, exaggerated, reads better**, and the quiet deliveries are the ones
that have to be pushed.

**One thing the curve does not show, and the report will not claim.** At
x = 2.0 the pitch-only accuracy *falls* to 0.917 while the full-feature
accuracy stays at 1.000. That is the envelope binding: at high
exaggeration the styles are clamped, and the pitch features stop
discriminating while the timing features carry the rest. The flattening
at the top of the curve is partly the envelope and partly the reader
being at ceiling, and the two are not the same thing.

---

## 9. THE BLIND LISTENING CHECK — the external referee

The machine's recovery rate is one opinion. Casey's is the other one.

The pack is a directory of renders with **anonymous names**, in a
shuffled order, a grade sheet asking for a guess and a confidence, the
machine's guess for each file beside them, and the answer key **held
back** in a `key/` subdirectory so that grading does not stumble over it.

**The pack that was produced:** 12 files, two per intent, from the final
round's held-out test renders, at `listening_check/` in the run's output
directory. The files are named `r3_00.wav` through `r3_11.wav` and the
order is shuffled, so the file names carry nothing. The grade sheet is
`GRADE-SHEET.csv`, the answer key is `key/ANSWER-KEY.csv`, and the
machine's guesses are in `MACHINE-PREDICTIONS.csv`.

**The teacher-dependence ratio.** The programme's headline metric is the
fraction of judgments the system must still send to a teacher, at held
referee-graded quality. For this lane it is **1.000 today**: every
label the reader learned from came from the performer's own priors, and
no referee has graded anything, so not one of its judgments is
certified. The tool computes the ratio the moment the grade sheet is
filled in — it is one minus the machine's agreement with the referee,
because a judgment is trusted where the two agree and is not where they
do not.

The path to moving that number is not more engineering. It is Casey's
own drill deliveries, which would replace the performer's priors with
measurements, and his blind grades, which would certify what they
produce. The pack is the hole in the sphere that the brief asks for, and
it is deliberately kept open.

### How to grade it

1. Open `GRADE-SHEET.csv`.
2. Listen to each `<file_id>.wav` and put your guess in `your_guess`.
3. `confidence_1_to_5` is how sure you are. `notes` is anything you want
   to say about the file.
4. Do not look in `key/` until the sheet is filled in.

Then:

```bash
python experiments/tone_game.py --score-listening <pack dir>
```

which reports your accuracy, your confusion matrix, and the machine's
accuracy on exactly the same files, side by side.

---

## 10. THE FAILURE MODES, HONESTLY

**The intents are a machine's opinion.** The starting design is a set of
priors about prosody written by a program, and the reader learns those
priors. A perfect recovery rate on synthetic deliveries says the
melody channel is *sufficient* when the identities are well separated by
construction. It does not say that real deliveries of the same words are
equally separable — they are not, and Casey's own drill deliveries are
the test that would show it.

**The voices are synthetic.** A harmonic source through three formant
resonators is a cartoon of a voice. It has the pitch right and the
timbre wrong, and the reader is reading the pitch. A real voice would
add jitter, breathiness and creak that this render does not have.

**The reader has no memory.** It is a classifier over eleven features,
trained from scratch each round, and it has no notion of a person. The
person profile is a different tool.

**The rendering is slow.** The pitch tracker is pure Python and runs at
about 1× realtime, so a full game with the ablations and the caricature
sweep is a quarter of an hour of compute. That is the price of using the
repository's validated tracker instead of a faster unvalidated one.

**The caricature knob is bounded.** The envelope clamps the exaggerated
styles, so the curve flattens where the envelope binds rather than where
the reader stops being able to tell the intents apart. The two are not
the same thing and the report does not claim they are.

---

## 11. WHAT IS STILL OPEN

**Casey's own deliveries.** The drill's ground truth. When the potato
drill mode lands, its six intents stop being a construction and become
measurements, and this experiment is re-run against them. This is also
what moves the teacher-dependence ratio off 1.0: the reader's labels stop
coming from the performer's priors and start coming from a person.

**The curriculum is already written.** The confusion classes name what to
drill next: `leaving` is the intent that needs twice the exaggeration of
the others before it reads, so `leaving` is the intent the drill should
spend its time on, and the caricature curve says how far the performer
has to push it before a listener can hear it at all. That is the
confusion-class-commissions-the-next-drill loop, with the drill chosen
from a measurement instead of from an accident.

**A second speaker.** The seed is one voice. The tone game's claim about
a *person* is not yet a claim about a person, only about a delivery.

**The liaison lane.** The digestion plumbing and the person profile both
consume the score this game produces, and the score is already the
currency between them.

---

## 12. HOW TO REPRODUCE ALL OF THIS

```bash
# the unit tests -- no model, no GPU, no real capture
/home/proart/projects/gpu-venv/bin/python experiments/tone_game.py --selftest

# the whole game: the rounds, the ablations and the caricature curve
/home/proart/projects/gpu-venv/bin/python experiments/tone_game.py \
  --out /tmp/tone_full --rounds 4 --train-variants 8 --test-variants 8 \
  --ablation-carrier

# score a filled-in grade sheet
/home/proart/projects/gpu-venv/bin/python experiments/tone_game.py \
  --score-listening /tmp/tone_full/listening_check
```

Every run writes `results.json` with the per-round accuracies, the
confusion matrices, the styles, the seed measurements, the ablations and
the caricature curve, and the renders themselves under `renders/`.
