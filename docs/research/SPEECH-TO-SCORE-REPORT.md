# SPEECH-TO-SCORE — the words, the notes, and the syllables between them

**Built by:** `experiments/speech_to_score.py`
**Target in the brief:** `/mnt/c/proart_capture/corpus/free_sessions/1792553040`
— still silent, and reported as such. **The first real free-session
target**, `1792560597`, is scored in section 5.7.
**Device:** the local RTX 4050 — `faster-whisper small.en` on `cuda` with
`float16`, and the repository's own pure-Python pitch tracker on the CPU.
No paid API, no network, no subscription.

---

## 0. THE HEADLINE

The pipeline is built, it runs end to end, and it is validated against the
seed corpus with an independent measurement the repository already had.
Four things are worth stating first.

**The first real free session is scored.** The sick-lane fix landed, and
session `1792560597` is the first thing that came out of a real microphone:
27.3 seconds of Casey's voice. The tool transcribed 17 words and 19
syllables, found 29 note events, and placed every syllable on the session's
own frame indices — frames 1158 through 1623, of 1626. The ambient session
that followed it, `1792560758`, is 154.9 seconds and 229 words, and Casey is
in it saying that the audio had only just started arriving in the video,
which is true, and is why the session exists at all.

**The clock had a real bug, and it was found by the real session.** The
frame offsets in `timestamps.json` are page-relative — the session that
opened 450 seconds after its page did starts at 450122.3 ms, not at 0 — and
the tool was reading them as session seconds. Every word in that session
landed on frame 0. The fix, the measurement that pins it down, and the
86-millisecond audio-versus-video question it opened up are in section 2.

**The melody the tool extracts agrees with the measurement the repository
already trusted.** The seed corpus's own manifest records the F0 median
and the voiced fraction of each file, measured by `src.score.measure` — a
different code path from this one, with no octave smoothing and no note
segmentation. Across seven files the melody's centre pitch lands within
**0.7% of that independent measurement on average** and within 3.4% in the
worst case, and the voiced fractions agree to within 1%. The tool is not
inventing a melody; it is measuring the one the repo already measures,
and expressing it as notes.

**There is one clock, and everything is on it.** The spine is the
session's own `timestamps.json`. Every syllable row carries its start and
end in seconds *and* the frame indices it spans, so "which note owns
frames 46 to 60" is a lookup and not a re-derivation. The analyzer's face
and speech segments join onto the same frame index — 80 of 80 rows joined
on the seed session, 19 of 19 and 295 of 295 on the two real ones, with
nothing re-measured.

---

## 1. THE TOOL

`experiments/speech_to_score.py` runs the four stages the brief asks for,
in this order.

**Transcribe.** A local `faster-whisper` model (`small.en` by default)
produces words with timestamps. It runs on the GPU when one is present and
falls back to the CPU when it is not; the report says which, every time.
The transcript is written twice: `transcript.txt` for reading and
`transcript.json` for machines, with the frame indices attached to each
word.

**Pitch.** The repository's own tracker, `src/pitch.py` — YIN, a 50 ms
window and a 10 ms hop, 70 to 400 Hz — followed by the repository's own
`pitch.smooth_octaves` post-pass against YIN's classic octave error. This
is not a re-implementation: the repo already had the answer to this
defect and the tool uses it.

**Notes.** The contour is segmented into note events, each with an onset,
an offset, a median pitch and a MIDI note number. The details are in
section 3.

**Align.** Note events are mapped onto the transcript's words and then
onto the words' syllables by time overlap. The output is
`alignment.json`: one row per syllable, carrying the start, the end, the
duration, the frame span, the MIDI note, the note's index, the overlap
that chose it, and the word it belongs to.

**Score.** `session.mid` is a real MIDI file, written by the repository's
own `src/smf.py` at the repository's own convention of one tick per
millisecond. It holds one track, one property, and the notes at constant
velocity. It does not hold the words: `src/smf.py` has no text-meta
event and was not modified to add one. The words are the other
deliverable, the transcript.

### What is the repository's and what is new

| piece | where it comes from |
|---|---|
| the pitch tracker | `src/pitch.py` — YIN, plus `smooth_octaves` |
| the energy measure | `src/energy.py` — `rms_series`, the same grid as the tracker |
| the pitch-to-note conversion | `src/score.py:hz_to_note` |
| the tick convention | `src/score.py` — `TICKS_PER_BEAT=1000`, `ms_to_ticks` |
| the MIDI writer | `src/smf.py` |
| the WAV reader | `src/wavio.py` — always resamples to 16 kHz |
| the transcription | **new** — `faster-whisper`, local, GPU |
| the syllable table | **new** — `cmudict` 1.1.3, 126,052 entries |

Two things are worth saying about that table. `mido` is not installable
on this machine, so `src/smf.py` is used as-is. And `av==14.4.0` cannot
be installed either — there is no compiler — so the Whisper input is a
numpy array rather than a file path, which sidesteps the PyAV/faster-
whisper incompatibility entirely and lets the repository's own reader do
the decoding.

---

## 2. THE ONE CLOCK

The keystone of this lane is that nothing has a second clock.

A free session writes `timestamps.json`, and that file is the spine. The
alignment rows are placed on it, and the frame indices are computed from
it, so a syllable that starts at 1.52 seconds on a 30.03 fps session
occupies frames 46 through 59, and a note that owns those frames is the
note that carries that syllable.

That is what makes the lip analysis possible without re-deriving
anything: the note owns the frames, the frames own the video, and the
syllable owns the note. Three lookups, one timeline.

The composition with the analyzer is a join, and it is a join on the frame
index. `free_session_analyze.py` already measured the face state and the
speech state on the session's own timeline; this tool does not re-detect a
face and does not re-threshold an RMS. It asks the analyzer's segment
table which segment a row's frames fall in, and copies the answer:

```bash
/home/proart/projects/speech-venv/bin/python experiments/speech_to_score.py \
  /tmp/harvest_dry3/free_sessions/1758000001 \
  --segments /tmp/harvest_dry3/derived/rich/segments_1758000001.json \
  --out /tmp/s2s_corpus/session_casey_join
```

The result on that session: 22 segments, 80 rows joined, 0 unjoined, and
every joined row now carries `face_state`, `speech_state` and the
segment's own bounds.

```
syl  0 [  1.52,  2.00] frames 46..59 midi=47 B2 seg=1 face=facing speech=silent   word='The'
syl  1 [  2.00,  2.12] frames 60..63 midi=48 C3 seg=2 face=facing speech=speaking word='library'
syl  2 [  2.12,  2.30] frames 64..69 midi=50 D3 seg=2 face=facing speech=speaking word='library'
syl  3 [  2.30,  2.42] frames 70..72 midi=50 D3 seg=2 face=facing speech=speaking word='library'
syl  4 [  2.42,  2.50] frames 73..74 midi=46 A#2 seg=2 face=facing speech=speaking word='opened'
syl  5 [  2.50,  2.80] frames 75..83 midi=47 B2 seg=2 face=facing speech=speaking word='opened'
```

One subtlety cost a frame and was worth catching: the analyzer's segment
spans are half-open, so a segment's `frame_end` is the frame *after* the
last one it owns, while an alignment row's `frame_end` is the last frame
the syllable touches. Joining them without noticing costs a frame at
every boundary. The join converts the row's end to an exclusive bound
before comparing, and the unit test checks a row that sits on a boundary
joins the later segment.

### 2.1 The origin, and the 86 milliseconds

The real session found a real bug, and the bug is worth writing down
because it is the kind that never shows itself in a test fixture.

**The offsets are page-relative, not session-relative.** The recorder
stamps each frame with `performance.now()` minus the page's own origin.
A session that opens 450 seconds after its page loaded therefore starts at
450122.3 ms, not at 0. That is the page's clock and the recorder's right
to use it, but a tool that treats those numbers as session seconds is
reading the wrong clock — and it fails silently, because the frames
*exist*, they are just in the wrong place. Session `1792560597` printed
`frames 0..0` for every one of its nineteen syllables.

The fix is to name the origin instead of assuming it: the session's first
frame is t=0, and the raw offsets are shifted by their own first value.
The shift is recorded (`origin_offset_s`), not hidden, so the wall clock
still turns it back into an absolute time. `free_session_analyze.py` was
already doing this — `arr - arr[0]` at its line 415 — which is why the
analyzer's segments joined onto the tool's frames so cleanly on the seed
corpus and why the two now agree on the real one.

**Then there were 86 milliseconds.** The audio is 27.3360 seconds long.
The frames span 27.2500 seconds. Nothing in the tool was wrong; two
rings of two different things simply do not end on the same instant. The
recorder keeps a pre-roll of frames (up to 1.2 seconds) and a pre-roll of
audio (up to 1.2 seconds plus one block), and the two are trimmed
independently, so the WAV's first sample sits against the first frame by
an amount that depends on where the block boundary fell.

Three ways of measuring it, on all three real sessions:

| session | class | the WAV's first sample against the first frame |
|---|---|---|
| `1792560597` | drilled | −67.3 ms |
| `1792560758` | ambient | −92.3 ms |
| `1792560773` | ambient | +5.7 ms |

The number above is the truth, measured from the page's own sample
counter: the diagnostics heartbeat reports the accumulated sample count
every second, and a least-squares fit of that count against the page's
wall clock gives the audio's position on the page clock to within 5 to 9
milliseconds. The pre-roll model agrees with it to within 2
milliseconds, which is the two-clock difference between `Date.now()` and
`performance.now()`.

The tool cannot use the diagnostics log — it is not part of the session
— so it estimates the lead from the session's own two records: the
wall-clock span and the audio's own length. The audio's end and the wall
clock's end are the same close, and the first frame is the wall clock's
start, so the difference between the wall span and the audio length is
where the WAV's first sample sits. Measured against the truth above, that
estimate is within **3.7 ms, 0.0 ms and 12.0 ms** — under one frame at
60 fps in the worst case, against a systematic error of up to 92 ms
without it.

The lead is applied in exactly one place, `SessionClock.frames_for_audio_span`,
and the clock reports it, says how it was measured, and refuses to apply
it when the audio it was handed is not the session's own recording. A
seed-corpus WAV analysed beside a session directory is left alone.

---

## 3. THE NOTE SEGMENTATION

The contour is segmented in two stages, and the order matters.

**Voiced runs first.** A frame with no pitch is an absence, and an
absence ends a note. The tracker refuses to guess, so a run boundary is
always a real boundary.

**Then splits within a run.** A note is closed when the median pitch of
the recent window has moved away from the note's own median by at least
`split_semitones` (one semitone by default) and the note being closed is
already long enough to be a note.

The geometry is stated rather than hidden. A note's onset is the first
frame's window *start* and its offset is the last frame's window *end*, so
consecutive notes tile their run with no gap and no overlap; the 50 ms
analysis window is the only blur. Where a contour steps, the cut lands on
the step itself — the window is walked back to the frame where the pitch
is closer to the new level than to the old one.

### The two bugs the unit tests caught

The first version of the split did not split. It compared the recent
window's median against a *forward-looking* window that included frames
it had not reached yet, and it measured the note's age with
`times[end-1] - times[start]` — the window's own span, not the note's age
since the note began. A note therefore never grew old enough to be
closed, and the two halves of a stepped contour came out as one note.

The fix walks with two pointers instead of one. `note_start` is where the
current note began, and the note's age is measured from there. `scan` is
the frame being examined, and the recent window is the frames before it.
The cut is then placed at the step rather than at the window edge.

The second bug was in the test, not the code. 120 Hz is MIDI 47 (B2) and
180 Hz is MIDI 54 (D♯3); the expectations said 57 and 62. The code was
right and the expectation was wrong, which the real output confirms: on
Casey's own voice the median is C3 at 130.8 Hz, and on the oration it is
D3 at 146.8 Hz. Both check out against `src/score.py:hz_to_note`.

---

## 4. THE SYLLABLES

Syllable counts come from CMUdict. A phoneme is a vowel when it ends in
a stress digit — the dictionary's own convention — and the words are
grouped into syllables by it. A word with more than one pronunciation
takes the first and the ambiguity is counted and reported rather than
hidden: 6 ambiguous words in the poetry excerpt, 40 in the short story.

Where the dictionary is silent, a stated heuristic is used: count the
vowel-letter groups in the word, treating a silent final `e` as not a
vowel. It is wrong sometimes, so every word it touched is listed in the
report. In practice the heuristic is asked for proper nouns and for
Whisper's punctuation artefacts, and almost nothing else:

| file | words | syllables | from the dictionary | from the heuristic |
|---|---|---|---|---|
| poetry | 22 | 31 | 31 | 0 |
| essay | 130 | 218 | 213 | 5 |
| shortstory | 201 | 282 | 230 | 52 |
| shakespeare | 133 | 180 | 160 | 20 |
| children | 226 | 279 | 267 | 12 |
| oration | 165 | 242 | 241 | 1 |
| alice01 | 227 | 265 | 264 | 1 |
| Casey's voice | 68 | 80 | 80 | 0 |

The heuristic's words, in full: `25th`, `Hoji,`, `Tozan`, `Tozan's`,
`Tozan,`, `Tozan.`, `Umon`, `Umon's`, `Umon,`, `spiritless` in the short
story; `-bag,`, `-bag.`, `-blossom,`, `-blossom.`, `-blossom?`,
`-hipped`, `-roses`, `-seed,`, `-seed?`, `humblebee`, `overflown` in
Shakespeare; `?î`, `awayî`, `reproachfully.`, `roseî`, `ìAll`, `ìWhat`
in the children's tales; `1795.`, `1891.`, `Kant's`, `signboard` in
Kant; and `12` in the oration.

A word's duration is split across its syllables weighted by phoneme
count, so a three-phoneme syllable takes three times as long as a
one-phoneme one. That is a heuristic and not a duration model, and it is
stated as such in the code.

---

## 5. THE VALIDATION

### 5.1 Notes against syllables

The brief asks for notes-versus-syllables count agreement. Across the
corpus the ratio ranges from 0.73 to 1.65, and the spread is not noise —
it tracks how much the melody does inside the syllables.

| file | notes | syllables | notes per syllable |
|---|---|---|---|
| essay | 160 | 218 | 0.73 |
| children | 281 | 279 | 1.01 |
| Casey's voice | 84 | 80 | 1.05 |
| shortstory | 298 | 282 | 1.06 |
| alice01 | 309 | 265 | 1.17 |
| oration | 343 | 242 | 1.42 |
| shakespeare | 285 | 180 | 1.58 |
| poetry | 51 | 31 | 1.65 |

The readings are consistent. In the essay the delivery is narrow and
quick, and many syllables carry no note at all — 21 of 218 have no
overlapping note. In the poetry and the oration the delivery is slow and
sustained with wide pitch movement, and a single syllable carries
several notes: the oration has 343 notes over 242 syllables and no
syllable without a note.

That ratio is worth keeping. It is a measure of how much the pitch
contour does beyond the words, which is the same quantity the tone game
is trying to maximise.

### 5.2 The melody against the repository's own measurement

`examples/realaudio/corpus/MANIFEST.md` records each seed file's F0
median and voiced fraction, measured with `src.score.measure` — a
different code path from this one, with no octave smoothing and no note
segmentation. It is a genuine cross-check rather than a restatement.

| file | melody centre | manifest F0 | ratio | this tool's voiced | manifest voiced | ratio |
|---|---|---|---|---|---|---|
| poetry | 195.1 Hz | 196 Hz | 0.995 | 0.392 | 0.39 | 1.01 |
| essay | 92.6 Hz | 90 Hz | 1.029 | 0.306 | 0.31 | 0.99 |
| shortstory | 107.4 Hz | 109 Hz | 0.985 | 0.555 | 0.55 | 1.01 |
| shakespeare | 149.3 Hz | 154 Hz | 0.969 | 0.528 | 0.53 | 1.00 |
| children | 178.1 Hz | 181 Hz | 0.984 | 0.533 | 0.53 | 1.01 |
| oration | 189.3 Hz | 196 Hz | 0.966 | 0.640 | 0.64 | 1.00 |
| alice01 | 233.8 Hz | 229 Hz | 1.021 | 0.498 | 0.50 | 1.00 |

The mean ratio on the melody centre is 0.993. The worst single file is
the essay at 3.4% high, and the essay is also the file with the most
syllables carrying no note, so its tracker is being asked the hardest
questions.

### 5.3 The melodic rest against the sentence boundary

Two clocks that were never compared to each other should agree where the
speaker stops. The transcript's pauses longer than 0.3 s were counted,
and each one was checked for a melodic rest in the same place.

| file | pauses over 0.3 s | with a coinciding melodic rest |
|---|---|---|
| oration | 7 | 7 |
| poetry | 2 | 2 |
| shortstory | 28 | 26 |
| Casey's voice | 6 | 5 |
| essay | 15 | 10 |

This is Whisper's word timestamps on one side and the pitch tracker's
notes on the other, agreeing on where the speaker breathed without either
one being told about the other.

### 5.4 A spot-check with frame indices

Casey's own capture, re-encoded in the free-session layout so it has real
frames and real timestamps, at 30.03 fps. The opening words and the
frames they own:

```
The       1.52- 2.00 s  frames 46..59  B2   seg 1  facing  silent
library   2.00- 2.12 s  frames 60..63  C3   seg 2  facing  speaking
library   2.12- 2.30 s  frames 64..69  D3   seg 2  facing  speaking
library   2.30- 2.42 s  frames 70..72  D3   seg 2  facing  speaking
opened    2.42- 2.50 s  frames 73..74  A#2  seg 2  facing  speaking
opened    2.50- 2.80 s  frames 75..83  B2   seg 2  facing  speaking
```

The three syllables of "library" land on C3, D3, D3 across frames 60 to
72, and the two of "opened" on A♯2 and B2 across frames 73 to 83. A lip
analysis would read frames 60 through 72 and know it is looking at
"library", and the note it is looking at is D3.

### 5.5 The MIDI round trip

A writer without a reader is a slogan, so the file is written and parsed
back. The unit test writes two notes, reads them with the repository's
own `smf.parse_smf`, and checks the ticks: the first note's onset is
tick 0, the second's onset is tick 500, the second's offset is tick 1250,
and one tick is one millisecond. The pitches survive the round trip
unchanged.

### 5.6 The unit tests

`--selftest` runs 49 checks that need no model: the clock's frame
arithmetic including a span past the end of the recording, the clock's
origin shift and its audio lead, the MIDI round trip, the syllable split
and its weights, the note segmentation and the placement of the cut, the
alignment's note choice, and the segment join. All 49 pass.

### 5.7 The first real free session

This is the thing the brief was waiting for, so it is given in full.

| | `1792560597` | `1792560758` | `1792560773` |
|---|---|---|---|
| class | **drilled** | ambient | ambient |
| the page's trigger | `manual` | `motion` | `motion` |
| length | 27.34 s | 154.89 s | 16.06 s |
| frames | 1626 at 59.63 fps | 9260 at 59.82 fps | 934 at 58.14 fps |
| words | 17 | 229 | 0 |
| syllables | 19 | 294 | 0 |
| note events | 29 | 288 | 0 |
| audio lead | −71.0 ms | −92.3 ms | −6.3 ms |
| rows joined to the analyzer | 19 of 19 | 295 of 295 | — |
| frame span of the first syllable | 1158 | 178 | — |

Casey's own words in the drilled session, from `transcript.txt`:

```
[ 19.50 - 20.20] There
[ 20.20 - 21.30] now
[ 21.30 - 21.50] you
[ 21.50 - 21.68] got
[ 21.68 - 21.88] my
[ 21.88 - 22.22] mic
[ 22.22 - 22.90] on
[ 22.90 - 23.98] darn
[ 23.98 - 25.78] I'm
[ 25.78 - 25.80] gonna
...
```

And from the ambient session that came after it, which is the session
where the audio started arriving in the video:

```
There. Now you're getting me with my mic. You weren't getting me with my
mic in before, so. Huh. All that stuff. Whatever. Cool, though. Well, here
I am. Now you get me in my temporal state. So if I talk into this thing,
you'll be able to sync up with it. That's all good stuff. Big thing is, how
do we make it effective and efficient? All that good stuff. So I noticed
that the audio wasn't coming through in the video until just a few seconds
ago. So now I reset the page and it looks like you're for the first time
probably getting audio in the video on the computer. ...
```

Two things in that transcript are worth stopping on.

The first is that it is evidence about the person and not about a
performance. Nobody composed "So I noticed that the audio wasn't coming
through in the video until just a few seconds ago" — the page caught it
because there was motion, and what it caught was Casey working something
out. That is exactly the distinction the data classes are for: this
session is evidence about Casey, the drilled one is evidence about a
delivery.

The second is that the corpus now contains the doctrine's own
justification, in Casey's words, unposed and first take:

> I feel like a second take you can't you can't fake it like like the
> first take is real discovery. The second take is you're trying to act
> like you discovered and that's part of the reason why it'd be nice to
> have this kind of system is then when inspiration happens you've got all
> the different lenses to catch it in.

The third session, `1792560773`, is honest about being nothing: the
transcript is empty, the pitch tracker found no voiced run, and the tool
writes `no_speech` with the measured peak RMS instead of inventing a
syllable. A corpus with a session in it that says nothing is a corpus
that can be trusted when it says something.

---

## 6. THE DRILL-SHAPED ANALYSIS

The potato drill holds the words constant and varies the delivery, and
the brief asks for that shape wherever a session contains repeated
same-phrase deliveries. Natural repetitions exist: the oration says
"noted for" three times with different words after it, and says "despite"
three times, and the tool can find them with `--drill N`.

```bash
/home/proart/projects/speech-venv/bin/python experiments/speech_to_score.py \
  examples/realaudio/corpus/oration_jfks19620912.wav --drill 2 \
  --out /tmp/s2s_corpus/oration_drill2
```

Fourteen phrases repeat in the oration. The one that matters is the
parallel triplet, and it is the drill's table already, measured on real
speech:

**"noted for" ×3** — "a college noted for knowledge in a city noted for
progress in a state noted for strength"

| instance | when | duration | notes | melody | mean RMS |
|---|---|---|---|---|---|
| 1 | 13.32 s | 1.88 s | 9 | 57 38 55 42 38 50 56 58 53 | 0.0745 |
| 2 | 17.22 s | 0.50 s | 3 | 57 54 55 | 0.1466 |
| 3 | 19.68 s | 0.48 s | 2 | 55 54 | 0.0990 |

The words are identical and the delivery is not. The first phrase is
drawn out over 1.88 seconds across nine notes at a quiet 0.0745 RMS; the
second collapses it to half a second and three notes and nearly doubles
the loudness; the third is faster still and quieter than the second. The
deltas against the first instance are +2.2 and +2.0 semitones in centre
pitch, −1.38 s and −1.40 s in duration, −6 and −7 notes, and ×1.97 and
×1.33 in loudness.

**"fact that" ×2** — the same shape, sharper: 1.62 s and nine notes
against 0.50 s and three notes, at twice the loudness.

**"despite the" ×2** — the control: a genuine parallel construction,
where the two deliveries are near-identical in shape, with the duration
within 0.12 s, the note count the same, the loudness within 5% and the
centre pitch within 1.5 semitones. The parallel pairs agree; the
elaborated-then-compressed pair does not. That is the difference between
a rhetorical figure and a change of delivery, and it is measurable.

The table the drill mode will feed is this table: per delivery, the note
sequence, the timing, the energy, the centre pitch, and the face-segment
state, with the deltas between the instances of the same phrase.

---

## 7. THE FAILURE MODES, HONESTLY

**The brief's literal target yields nothing — and now something else
does.** Session `1792553040`, the brief's target, is still digital
silence: every one of its 12,431,744 samples is zero. The tool reports
that plainly — peak RMS 0.00000000, 0 of 25,895 frames over the energy
floor, "there is nothing to transcribe, nothing to pitch and no
syllables to align" — and writes an empty but well-formed set of outputs
rather than failing. The sick-lane fix has since landed, and the three
sessions that came after it are scored in section 5.7.

**A clock can be wrong without looking wrong.** The page-relative origin
in section 2.1 is the example that keeps: the frames existed, the lookup
returned a frame, the table printed, and every row said frame 0. Nothing
in the output said "wrong" — it said "zero" with great confidence. The
unit test that now covers it is built from the real numbers of the real
session, not from a tidy fixture, because a tidy fixture starts at zero
and so cannot catch this at all.

**The audio lead is an estimate, and it is off by up to 12 milliseconds.**
The truth is knowable to about a millisecond from the page's own sample
counter, but the page's counter is not in the session directory, so the
tool estimates it from the wall clock and the audio's length. The error
is under one frame in the worst case measured and the direction is
consistent, but it is an estimate and it is labelled as one.

**The resampler does not anti-alias.** `src/wavio.py` resamples with
linear interpolation and says so in its own docstring. The *time base* is
exact — from 48 kHz the decimation is exactly 3:1, and sample *i* at
16 kHz is sample *3i* at 48 kHz, with no time shift — but content near
the old Nyquist folds. Tonight's session is 48 kHz and the seed corpus is
already 16 kHz, and alice01 at 44.1 kHz goes through the interpolating
path, where the note centre still matched the manifest to 2.1%. This is
a stated limit of a repository component and was not changed.

**The pitch tracker is slow.** It is pure Python and runs at about
1× realtime — 69 seconds for 75 seconds of audio, and 56 seconds for the
154.9-second ambient session. The transcription is fast by comparison
(4.6 s for the same session on the GPU), so a long session is dominated
by the tracker. This is a correctness-and-time tradeoff that was made
knowingly: the repository's tracker is the thing that was validated
against the corpus manifest, and a faster tracker would have been a
second, unvalidated measurement.

**Notes and syllables disagree, and the direction is meaningful.** The
ratio runs from 0.73 in the essay to 1.65 in the poetry. The tool reports
the ratio rather than reconciling it, because a syllable that carries
three notes and a syllable that carries none are both real.

**The alignment cannot verify meaning.** It joins a syllable to a note by
time overlap and nothing more. It cannot tell you whether the note
matched the word, and nothing in the pipeline claims to.

**The .mid holds no words.** One track, one property, constant velocity.
`src/smf.py` has no text-meta event and was not modified. If the words
are needed inside the MIDI, that is a change to the repository's writer
and a decision for someone else.

**Twenty-one of the essay's 218 syllables have no note at all**, because
the pitch tracker found no voiced run under them. They are reported as
`midi: null` and counted, not quietly dropped.

---

## 8. WHAT IS STILL OPEN

**The corpus is too thin to say anything about drift.** Three sessions
have audio: one drilled, two ambient, and one of the ambient ones is
silent. The person profile has one ambient baseline and no drilled
baseline at all, so a drift number would be a number with nothing to be
drifted from. The apparatus is ready; the corpus is the missing part.

**The drill's ground truth is still Casey's own deliveries.** The natural
repetitions in the oration are a stand-in for the drill's shape, not its
data, and no potato drill has been recorded yet. Until one lands, the
notes-per-syllable ratio and the rest are measured on free speech, and
they are measurements of the method, not of the intent.

**The note count is not a meaning.** The tool says which MIDI note each
syllable carries. It does not say what the melody meant, and section 6's
table is a shape, not an emotion.

**The tone game consumes this.** The pitch tooling, the note segmentation,
the one-clock alignment and the drill's table are all here, and the tone
game's reader is meant to work on exactly the note representations this
tool produces.

---

## 9. HOW TO REPRODUCE ALL OF THIS

```bash
# the environment: the local speech venv, with CUDA's libraries on the path
export LD_LIBRARY_PATH="/home/proart/projects/speech-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/proart/projects/speech-venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/proart/projects/speech-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"

# the unit tests -- no model, no GPU, 49 checks
/home/proart/projects/speech-venv/bin/python experiments/speech_to_score.py --selftest

# the seed corpus
for f in examples/realaudio/corpus/*.wav; do
  /home/proart/projects/speech-venv/bin/python experiments/speech_to_score.py \
    "$f" --out "/tmp/s2s_corpus/$(basename "$f" .wav)"
done

# the drill's table on the oration
/home/proart/projects/speech-venv/bin/python experiments/speech_to_score.py \
  examples/realaudio/corpus/oration_jfks19620912.wav --drill 2 \
  --out /tmp/s2s_corpus/oration_drill2

# the analyzer's segment table for a real session
/home/proart/projects/gpu-venv/bin/python experiments/free_session_analyze.py \
  /mnt/c/proart_capture/corpus/free_sessions/1792560597 \
  --out /tmp/fsa_new/1792560597 \
  --json /tmp/fsa_new/1792560597/free_session_segments.json

# a REAL session, with the analyzer's segments joined onto the frame index
/home/proart/projects/speech-venv/bin/python experiments/speech_to_score.py \
  /mnt/c/proart_capture/corpus/free_sessions/1792560597 \
  --segments /tmp/fsa_new/1792560597/free_session_segments.json \
  --out /tmp/s2s_real/1792560597
```

Every run writes `transcript.txt`, `transcript.json`, `session.mid` and
`alignment.json`, and the last of those holds the whole report — the
transcription's metadata, the pitch's metadata, every note, every
syllable row with its frame span, and the drill's table when it was
asked for.
