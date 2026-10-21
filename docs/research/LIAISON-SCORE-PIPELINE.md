# THE LIAISON SCORE PIPELINE — the segment becomes a score, and the score becomes a paragraph

**Built by:** `experiments/digest_segment.py` and `experiments/person_profile.py`
**The brief:** `BRIEF-LIAISON-CONVERSATION-PAGE.md` — "everything is saved,
saved as a score not flat text, and the analysis grows over time very
uniquely to the person"
**The design law:** `BRIEF-TIERS-AND-DECOMPOSITION.md` — decompose into
small old tools, the quantized score is the currency between tiers, and
a judge cannot certify itself
**The north star:** `BRIEF-SYNERGY-LEARNING-LOOP.md` — the headline
metric is the teacher-dependence ratio falling at held quality

---

## 0. THE HEADLINE

Two tools, and they are two tools because the design law says to
decompose and because the seam between them is the interesting part.

**`digest_segment.py` takes a closed segment and produces a score bundle
and one paragraph.** The score bundle is the transcript with timestamps,
the MIDI melody, and the alignment table with per-syllable frame indices
— the same archive format the speech-to-score work established, on the
session's own clock. The paragraph is the digestion, and it is made *of*
the score: there is no second pass over the audio and no separate
description of what happened.

**`person_profile.py` folds sessions into a living document.** It carries
the contour statistics, the vocabulary that is growing, the contour
motif vocabulary, and the drift of each session against everything before
it. It is descriptive and it says so: a drift says a thing moved, not
what the speaker was doing.

**The one clock is the spine, and the frame indices are in every row.** A
syllable inside a segment that began at 4.23 s carries the SESSION's
frame indices, not the segment's, so the page can link straight to the
video without re-deriving anything.

**It has been run on real sessions, and the proof is in the bundles.**
Sixteen segments across four sessions — the facing capture, the turned
capture, the silent capture and the empty room — were digested end to
end. Eight produced a score and a paragraph; eight produced an honest
"there was no speech in this" and nothing else. The four sessions are
folded into `docs/research/PERSON-PROFILE.md`, which is the profile
growing.

---

## 1. THE SCORE BUNDLE — the currency

A segment's bundle is a directory, and every part of it is on the
session's timeline.

| file | what it is |
|---|---|
| `transcript.txt` | the words with their times, readable |
| `transcript.json` | the same, with the frame indices attached |
| `segment.mid` | the melody, one track, one property, 1 tick = 1 ms |
| `alignment.json` | the whole report: words, notes, every syllable row, the digest |
| `digest.json` | the paragraph, and which generator made it |

The alignment row is the unit the rest of the system joins on:

```
syllable  17   4.31 -  4.39 s   frames 130..131   A2   facing speaking  'and'
syllable  18   4.39 -  4.51 s   frames 132..135   A2   facing speaking  'the'
syllable  19   4.51 -  4.81 s   frames 136..144   A2   facing speaking  'light'
syllable  20   4.81 -  5.17 s   frames 145..155   C#3  facing speaking  'came'
```

Those are the frames in the session's video, and the note that carries
the syllable, and the face and speech state the analyzer already
measured for that stretch of frames. Nothing is recomputed: the face
state is a join on the frame index, and the melody is the score the
speech-to-score pipeline wrote.

**The score is the currency, and it is provably a complete description
of the delivery.** The tone game already proved the round trip — a
delivery rebuilds from the MIDI alone, note for note, and the audio
rendered from the rebuilt score is the same audio — and that property is
what lets a weak node ship a score up instead of shipping audio. The
digest pipeline inherits it: the melody in the bundle is a MIDI file on
the session's clock, and a node that cannot run the tracker can still
ship it.

---

## 2. THE DIGESTION

One paragraph: what was said, in essence.

**The generator is a pluggable stage, because the tiers differ.** A
laptop may visit the fleet; the boat must not. So the tool takes a
`--digest-command`: it hands the command the transcript as JSON on
stdin, and takes one paragraph back on stdout. The command is the fleet
hook, and the boat's is a local model.

**The fallback is a real paragraph, and it is honestly labelled.** With no
command, the digest is built from the measurement: the duration, the
word count, what the melody did, what the face and speech state was,
and the words themselves.

> You spoke for 4.8 seconds, 15 words, on a melody that falls from A2 to
> G2, spanning 6 semitones around A#2, in 18 notes whose longest lasted
> 0.59 seconds, while facing and speaking. You said: "and the light
> came through the east window in the long bar across the floor."

It does not interpret, and it does not pretend to be generated. The
`digest.json` says which generator made it:

```json
{
  "text": "...",
  "generator": "extractive",
  "honesty": "this paragraph was built by the fallback from the
              measurement, not generated; it says what was said and what
              the delivery did, and it does not interpret"
}
```

**A command that fails does not lose the segment.** The tool falls back
to the extractive digest and says so. A digestion is a small thing to
lose and a segment is not.

---

## 3. THE PERSON PROFILE — a living document

Casey's framing: the corpus is a portrait of ONE speaker, deepening over
time. The profile is the portrait, and it is regenerated whole each time
so it never carries a stale paragraph.

**Three things accumulate.**

*Contour statistics*, measured the same way for every session: the
melody's centre, its range, its slope, the note rate, the note
durations and their spread, the step sizes, the jitter, the voiced
fraction.

*The vocabulary*: the words, the cumulative count, and which of them are
new in this session.

*The contour motif vocabulary*: the coarse shape of the melody —
rising, falling, level, arched, dipped, mixed — and how often each is
used. This is a vocabulary of shapes and not of meanings, because the
intent labels need labelled data and the drill supplies that when it
lands. When the drill does, the intent labels join the same table.

**The drift is measured against the running baseline**, in standard
deviations. The first session has no drift: it *is* the baseline. The
second carries a delta and no standard deviations, because a baseline of
one point has no spread to measure against; from the third on the
comparison is in standard deviations.

**The honesty rule is the point.** The profile does not say what the
speaker was feeling and it does not label a drift with an emotion. A
large drift is a measurement, not a diagnosis. The interpretation is
earned slowly: it needs sessions, it needs a second opinion, and it
needs Casey's own drill deliveries as the ground truth that would make
it mean anything.

**What the four real sessions showed.** The facing session contributed
27 new words and the turned session 8; the silent capture and the empty
room contributed none, because there was nothing to contribute. The
silence shows up in the portrait as a hole rather than being averaged
away: the silent session's `voiced_fraction` sits 9.0 standard
deviations below the baseline of the sessions before it, and the empty
room 1.4 below. The contour baseline over the two sessions that carried
voice is a centre of MIDI 46.1 (≈ A♯2), a range of 9.5 semitones, a
slope of −1.26 semitones per second — a voice that falls as it talks —
and 4.39 notes per second.

---

## 4. WHO OWNS WHAT

**Mine.** The score pipeline that runs per segment; the digestion; the
profile's accumulation and its rendering; the failure classes that
become detectors.

**ZCode's.** The liaison's API and the page. The page needs to do three
things and none of them are mine: take the score bundle the digest
writes, show the paragraph, and link every element back to the saved
bundle. The segment-closed event is the trigger.

**The shared contract** is the bundle's shape, and it is the same shape
the speech-to-score work established: transcript, MIDI, alignment rows
with frame indices, on the session's clock.

**Casey's.** The drill deliveries that become the ground truth, and the
blind grades that certify what the system produces. The tone game's
referee step is the same hole in the sphere, and the teacher-dependence
ratio is measured through it.

---

## 5. THE TIERS — degrade up, never down

| tier | what it does |
|---|---|
| phone / Uno-class | record and meter; ship the score when a link appears |
| laptop (RTX 4050) | decompose locally — VAD, pitch, MIDI, the digest may visit the fleet |
| Jetson 8 GB | bigger local speech, the profile analysis, batch passes |
| Thor 128 GB (boat) | everything local, including the digestion; autonomous offline |

The score is what crosses the tiers, and it is small, inspectable and
complete. The digestion is the only stage that needs a big model, and
it is the only stage with a fallback — so the boat loses charm, not
truth.

---

## 6. THE FAILURE MODES, HONESTLY

**The silent audio.** Tonight's ten free sessions are digital silence,
and the digestion says so: an empty transcript, no notes, and a
paragraph that says there was no speech in it. This is the
`SILENT_AUDIO` failure class, and the profile records the session as a
session with no contour statistics — a hole in the portrait, visible
rather than averaged away.

**The pitch tracker is slow.** About 1× realtime in pure Python, so a
long segment takes as long as it is long. The transcription is fast by
comparison. This is the price of using the repository's validated
tracker.

**The resampler does not anti-alias.** `src/wavio.py` says so in its own
docstring. The time base is exact — from 48 kHz the decimation is 3:1
with no shift — but content near the old Nyquist folds.

**The fallback digest is not a generated digest.** It reports; it does
not summarise in prose, and it does not know what the words meant. The
label in `digest.json` is the difference between a digestion and a
description.

**The profile has one speaker.** It is a portrait of one voice, and the
baseline is that voice's. A different speaker's session folded into the
same profile would be a measurement error, not a finding, and the
profile's shape assumes one speaker because Casey's corpus is one
speaker.

---

## 7. HOW TO REPRODUCE ALL OF THIS

```bash
# the unit tests -- no model, no session
/home/proart/projects/speech-venv/bin/python experiments/digest_segment.py --selftest
/home/proart/projects/gpu-venv/bin/python experiments/person_profile.py --selftest

# digest one segment of a real session
/home/proart/projects/speech-venv/bin/python experiments/digest_segment.py \
  /mnt/c/proart_capture/corpus/free_sessions/<session> \
  --segments <the analyzer's segments_<session>.json> --segment 4 \
  --out /tmp/digest

# with the fleet as the digest generator
/home/proart/projects/speech-venv/bin/python experiments/digest_segment.py \
  /mnt/c/proart_capture/corpus/free_sessions/<session> --segment 4 \
  --digest-command "python3 /path/to/fleet_digest.py"

# fold the sessions into the profile
/home/proart/projects/gpu-venv/bin/python experiments/person_profile.py \
  --session /tmp/digest_batches
/home/proart/projects/gpu-venv/bin/python experiments/person_profile.py --report
```

The profile lives in `experiments/person_profile.json` and the living
document in `docs/research/PERSON-PROFILE.md`.
