# Real dialogue: recording protocol

`examples/realtwospeaker/` unblocked the turn question but, measured, it cannot
settle it. Three limits, all recorded in
[../../docs/research/TWO-SPEAKER-TURNS.md](../../docs/research/TWO-SPEAKER-TURNS.md):

1. **The gaps are not acoustic.** AMI's audio is a reconstruction: each
   utterance is painted at its annotated `begin_time`, unannotated stretches are
   digital silence. 83% of frames carry real signal; the other 11% are holes
   placed by the annotation timeline, so their *position* is a convention, not a
   measurement. Every silence-duration number we have is therefore a number
   about AMI's file layout.
2. **The audible annotation gap is structurally 0.000 s.** Consecutive audible
   turns tile to the millisecond. The labels carry who spoke; they carry
   nothing about how long anyone waited.
3. **One far-field microphone destroys the interesting part.** 111 of 314
   segments are wholly overlapped and inaudible. Barge-in — the event that
   actually decides who holds the floor — is partly gone before we see it.

Real recordings fix all three, and only real recordings do. Everything below
exists because of a specific number above.

---

## The one decision that matters: **two microphones, two files**

Not one stereo file. **Two separate mono WAVs, one per speaker, started from the
same start signal.**

This is what makes the corpus worth building:

| what you get | what it fixes |
|---|---|
| speaker attribution is *exact* | no diarizer, no ear-labelling, no ground-truth error |
| overlaps stay separable | the AMI far-field channel loses overlapping voices entirely |
| gaps are real | silence between two files is silence in the room, not a hole in a file |
| two voices can be swapped | lets one recording test *which* of two hypotheses was right |

One stereo file is the trap. It looks like two channels and behaves like one
far-field mic, which is exactly the corpus we are trying to replace.

---

## Recording spec

- **Format:** WAV, PCM 16-bit.
- **Rate:** anything 16 kHz and up. The pipeline resamples to 16 kHz itself
  (`src/wavio.read_wav`), so do not pre-convert.
- **Channels:** one per file. One microphone per person, roughly 20–50 cm away,
  at mouth height. Not a laptop's built-in array — separate mics are the point.
- **Room:** an ordinary lived-in room. Do **not** soundproof. Room tone is a
  measurement (`src/vad.py` has a gate calibrated on exactly that), and a silent
  studio would throw away the hum case the gate exists for.
- **Start:** clap once, loudly, before each scene. It gives an unambiguous
  alignment point across the two files without needing a shared clock.
- **Also record 30 s of nobody talking**, at the end of any session. This is the
  noise floor every other number is measured against.

---

## Scenes

Eight scenes. Each exists because a specific measurement needs a case it cannot
get from anywhere else. Total about 25 minutes of recording.

### 1. Free conversation — 5 min
Just talk. Ask each other things, disagree, agree, trail off.

The hardest and the most valuable. Real hand-offs land wherever they land, and
it is the only scene with genuinely unpredictable gaps. **Do not try to perform
the other scenes here.**

### 2. Question, then answer — 3 min
A asks a question and stops. B answers, at whatever delay feels natural.

This is the clean positive class: a real hand-off with a real, uncontrolled gap.
Repeat the same question ten times with different natural pauses — that is what
turns a duration into a distribution.

### 3. Thinking pause — 3 min
Ask B a genuinely hard question (something neither of you knows offhand). B
must **think silently** and answer. Aim for 2–10 s of real silence before any
word.

This scene is the single most important one in the list. Measured on AMI, every
threshold at or above 3.0 s fired **zero** times with recall 0.000, and raising
the bar monotonically destroyed accuracy — because long silences are where
*nobody* speaks, not where someone hands over. This scene is the only way to
confirm that, because nobody scripts a 2 s think on purpose in a meeting.

### 4. Turn-holding fillers — 2 min
A starts answering, then stalls: *"umm… so… you know what I mean is…"* — and
continues without stopping.

The negative class. Same speaker, long in practice, **no silence at all**. Any
rule that fires here is interrupting someone mid-thought.

### 5. Backchannel — 3 min
A talks at length. B says *"yeah"*, *"mm-hmm"*, *"right"* while A is still
going, without taking the floor.

The classic hard case and the one F0 cannot help with: a different speaker, a
very short utterance, near-zero gap. Measured, median F0 separates two voice
*classes* but not two speakers within one class (0.5σ and 0.3σ apart), so pitch
is blind to this case by construction.

### 6. Deliberate interruption — 2 min
B cuts A off mid-sentence, on purpose, several times. Include some where B
starts *before* A finishes and some a beat after.

Overlap with ground truth. AMI lost 111 of 314 segments to exactly this, so the
existing corpus cannot tell overlap from silence. Here the two files can.

### 7. One speaker alone, long pauses — 2 min
A talks or reads a passage aloud, pausing over 2 s between sentences. B is
silent throughout.

The pure negative control: same speaker, silence, no hand-off. If a threshold
fires here it is interrupting, which is the failure that is not recoverable.

### 8. Room tone — 0.5 min
Nobody speaks. Just the room, with whatever is running in it.

---

## What not to do

- **Do not read from a script.** Read speech is what the old corpus was, and it
  is why it could not answer the question. Improvise.
- **Do not merge to one file.** See above.
- **Do not label by ear afterwards.** Per-file channels are the labels. Ear
  labelling is where silent ground-truth errors come from, and this project has
  already been caught by one.
- **Do not use the same microphone.** One mic is one far-field channel, which is
  the thing being replaced.
- **Do not do only scenes 2 and 7.** Those are the easy pair, and on their own
  they would let a trivial rule look good.

---

## Dropping files in

Put them in `examples/realdialogue/raw/` — create it if it is not there. Keep the
two channels of one scene adjacent and named as a matched pair:

```
raw/scene03_qa_A.wav      raw/scene03_qa_B.wav
raw/scene03_think_A.wav   raw/scene03_think_B.wav
raw/scene07_alone_A.wav   raw/scene07_alone_B.wav
```

Same basename prefix, `_A` / `_B` suffix. Whatever tool you use, exporting the
pair twice (once as A, once as B) from the same take is fine.

Then:

```bash
/home/proart/projects/glasses/.venv/bin/python experiments/dialogue_intake.py
```

It checks the files actually exist as matched pairs, reports rate / duration /
level per channel, and measures how many silences, hand-offs and overlaps the
corpus now contains. It reports what is **still missing** — a suite that cannot
say "this is not yet enough" is not a useful intake check.

It also does three things this protocol needs it to do, each of which depends
on something above:

- **it aligns the two channels on the clap.** Two recorders started by hand
  are not synchronised, and without the correction every gap in the corpus
  carries the offset between them. That is why the clap is asked for *once
  per scene, loudly, before either person speaks* — it is the alignment
  point, not decoration.
- **it measures the noise floor from the room-tone scene** and measures every
  other level against it. That is why scene 8 is asked for: "the noise floor
  every other number is measured against".
- **it exits non-zero when the corpus is not ready** (0 = usable,
  1 = not yet usable, 2 = nothing recorded), so a script can act on the
  verdict.

If it reports a channel B that is silent where B is supposed to speak, a
cross-talk number above 0.60 (channel B hearing channel A), or a pair whose
two claps do not match — those are faults in the recording, and the tool
names the scene and the check.