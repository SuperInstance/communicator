# PAUSE STUDY — what a pause is, measured, and how far a thinking-vs-terminal split can be pushed today

**What this is.** The wake-and-turn brief
(`docs/research/BRIEF-WAKE-AND-TURN.md`) asks the endpointer to stop
treating silence as absence. It names three kinds of pause — THINKING
(mid-thought, the speaker will continue), BEAT or BREATHING (a pause
inside a phrase), and TERMINAL (the speaker has finished and the floor
may be yielded) — and it asks for the pause statistics before any
endpointer is built.

This is those statistics. The measurements are in
`experiments/pause_study_results.json`, produced by
`experiments/pause_study.py`, which carries its own self-test
(`--selftest`, 23 checks).

**The short version.**

1. **The corpus has exactly two kinds of pause evidence, and they are
   not interchangeable.** The script pair (`A.wav`, `B.wav`) carries
   ground truth: two independent readings of the same written passage,
   so the script itself says where a sentence ends and where a clause
   ends. The free sessions carry none. The two are reported apart and
   never averaged.
2. **A terminal pause is longer, and it is preceded by a FALLING
   pitch.** Terminal median **0.90 s**, beat **0.645 s**, internal
   **0.31 s**. The pitch slope over the 300 ms before the pause is
   **−20.1 st/s** at a terminal pause, **−0.4** at a beat, and **+4.0**
   at an internal one. That is declination, measured.
3. **A terminal pause is followed by a pitch RESET.** The pitch step
   across the pause is **+4.5 semitones** at a terminal pause, **+1.0**
   at a beat, and **−0.1** at an internal one. After an internal pause
   the speaker comes back at the same pitch; after a terminal pause the
   speaker comes back somewhere else.
4. **Syntax separates the classes perfectly on this corpus.** The word
   before a terminal pause is **never** a function word (0 of 6) and the
   word after it is **never** lowercase (0 of 6): the utterance is
   finished and a new sentence starts. At an internal pause the
   pre-word is a function word **23 times out of 32** and the post-word
   is lowercase **32 times out of 32**: the utterance was left hanging
   and then continued.
5. **The prosody of the terminal class REPRODUCES across independent
   takes; the prosody of the internal class does not.** At the same
   script position, A and B agree on the terminal pause's falling slope
   (tok 19: −21.0 vs −19.3 st/s; tok 38: −22.9 vs −31.0; tok 58:
   −22.8 vs −29.5) and DISAGREE on the internal pause's slope, which
   flips sign between takes (tok 17: +11.1 vs −6.7; tok 10: −10.9 vs
   +6.5). The declination is a property of the position, not of the
   speaker's mood. That is what makes it usable.
6. **Duration alone is the strongest single observable** (AUC **0.908**
   for terminal against the rest) but **6 terminal events is a
   direction, not a measurement.** The power statement is in §6 and it
   is not favourable: the four-leg conjunction the brief asks for, built
   and graded as a whole, **misses 4 of the 6 terminal pauses** while
   holding 39 of the 40 non-terminal ones. The leg that actually
   carries is **syntax** — it fires on **6 of 6** terminals — so the
   shape the endpointer should take is a **gate with a veto**, not four
   conditions multiplied.
7. **The dialogue protocol has produced nothing.** `examples/
   realdialogue/raw/` is empty: **0 of 1230 s** of the eight-scene
   design is recorded. Scene 3 is the labelled thinking pause and it
   does not exist. The design is in §7 because it is the plan, and it
   is the corpus that would settle the power problem.

---

## 1. The corpus, and the one thing about it that decides everything

The brief names three corpora: tonight's free sessions, the A/B/C
recitations, and the dialogue protocol design. Only two of the three
carry audio.

| corpus | what it is | audio | ground truth |
|---|---|---|---|
| script pair | `A.wav`, `B.wav` — two independent readings of the same written passage | 30 s each | **yes**: the script |
| script nulls | `C.wav` (facing, silent), `empty1.wav` (nobody talking) | 30 s, 150 s | n/a |
| free sessions | 57 `browser_free_record` sessions | **6 carry speech**; 51 are digital silence | **no** |
| protocol design | `examples/realdialogue/README.md`, eight scenes | **0 s recorded** | by design |

### 1.1 A and B are the same script, read twice

This is the whole reason the study has ground truth. The passage is:

> The library opened at nine, and the light came through the east
> window in the long bar across the floor. A man in a gray coat sat
> near the window with a cup of coffee he did not drink. He turned a
> page, then another, and he did not look up. Outside, a bus pulled
> away from the curb, and the sound of it faded down the street.

`A` is the facing take, `B` is the turned-away take. Two independent
readings, so the *script* — not a rater's judgement — labels every
pause:

| class | definition | n |
|---|---|---:|
| `terminal` | a sentence boundary falls in the gap | **6** |
| `beat` | a clause boundary (comma) falls in the gap | **8** |
| `internal` | the reader stopped where the script did not tell them to | **32** |

**The labels come from the alignment, and the alignment is checked.**
The script is aligned to the ASR words by `difflib.SequenceMatcher`
over normalised tokens; the ratio is **0.985** on A and **0.865** on B.
The words are assigned to the energy-measured speech segments by a
**monotonic sweep**, because whisper's word timestamps are off by up to
a third of a second while its word ORDER is exact. A word whose
timestamp lands inside a pause therefore joins the speech that came
before it, which is what repairs a late timestamp.

This classification is the first thing to get wrong in this kind of
work, and it was wrong in the first draft of the tool: selecting "the
last word before the pause" by timestamp put `drink.` *after* the
pause that followed it, and the terminal class came out with n = 1
instead of n = 6.

### 1.2 The free sessions are mostly silence, and that is measured

The seed corpus's sessions carry a WAV with a **peak of exactly
0.0000** — digital silence, not a quiet room. Of the 57 free
sessions, **6 carry speech** and **47 carry no interior pause at
all**; the remaining four are sonar probe artefacts. The corpus is a
**video** corpus with a small speech corpus inside it.

That is a finding about the corpus, and it constrains every conclusion
below: the free-speech statistics are computed on **103 interior
pauses** drawn from **5 sessions**, not on 57 sessions.

---

## 2. The method, stated so it can be checked against the rest of the repo

Pause finding uses the repository's own segmentation, byte-identical to
`experiments/pause_threshold.py` and therefore comparable with the
table in `docs/research/PAUSE-THRESHOLD.md`:

```
energy.rms_series(samples, 800, 160, 16000)
energy.segment(frames, 0.01, peak_rms=max(f.rms for f in frames))
```

`min_silence_s` 0.150, `min_speech_s` 0.060, adaptive threshold off.
Percentiles are **nearest rank**, the same convention
`PAUSE-THRESHOLD.md` uses, so the two documents' numbers can be read
side by side.

Prosody is the repository's YIN tracker (`src/pitch.py`), run on the
speech either side of each pause in two windows of one second. The
silence itself is never tracked — a long pause would otherwise cost
minutes of YIN to learn nothing, because silence has no pitch.

Syntax is a **local** faster-whisper `small.en` on **CUDA, float16**
(RTX 4050) with word timestamps. No paid API is called. The device is
recorded in the results file.

---

## 3. The measured table

All 46 script-labelled interior pauses, A and B together.

| class | n | duration median | p25 | p75 | pre-pause slope (st/s) | pitch step across the pause (st) |
|---|---:|---:|---:|---:|---:|---:|
| **terminal** | 6 | **0.900** | 0.800 | 0.960 | **−20.1** | **+4.5** |
| **beat** | 8 | 0.645 | 0.580 | 0.720 | −0.4 | +1.0 |
| **internal** | 32 | **0.310** | 0.190 | 0.540 | **+4.0** | **−0.1** |

Read the third column of the two right-hand columns together. A
terminal pause is **preceded by a fall and followed by a step**; an
internal pause is **preceded by a rise and followed by no step at
all** — the speaker comes back at the pitch they left.

### 3.1 The syntax side, which separates more cleanly than the prosody

| class | pre-word is a function word | post-word is lowercase |
|---|---:|---:|
| terminal | **0 / 6** | **0 / 6** |
| beat | 2 / 8 | 8 / 8 |
| internal | **23 / 32** | **32 / 32** |

The rule that does the work is short: **an utterance that ends on a
function word was not finished.** The pre-word of an internal pause is
"the", "of", "a", "in", "with", "and", "he" — the speaker stopped
mid-phrase and the next word continues the same clause. The pre-word
of a terminal pause is "floor.", "drink.", "up." — the sentence ended
and a capitalised word begins a new one.

Whisper's capitalisation is doing real work here and it is worth being
explicit about the assumption: whisper capitalises what it believes is
a sentence start, so "the post-word is lowercase" is whisper's opinion
about sentence structure, not a fact about the recording. On the
script corpus that opinion is **checkable against the script** — and
where the two disagree, the script wins.

---

## 4. How far the separation goes, honestly

Each observable was graded on its ability to tell a terminal pause
from every other labelled pause. The AUC is rank-based with ties at
half; the interval is a 2000-draw bootstrap; the accuracy is
**leave-one-out**, because the in-sample accuracy of a single cut
fitted on the very pauses it is graded on is not a measurement.

| observable | AUC | 90 % interval | LOO accuracy | n pos / n neg |
|---|---:|---|---:|---|
| **duration** | **0.908** | 0.796 – 0.983 | 0.848 | 6 / 40 |
| **pitch step across the pause** | **0.855** | 0.726 – 0.957 | 0.778 | 6 / 39 |
| pre-pause pitch slope | **0.175** | 0.067 – 0.308 | 0.848 | 6 / 40 |
| post-pause pitch slope | 0.650 | 0.470 – 0.808 | 0.844 | 6 / 39 |
| pre-pause pitch range | 0.646 | 0.475 – 0.804 | 0.848 | 6 / 40 |
| pre-pause mean pitch | 0.117 | 0.025 – 0.238 | 0.913 | 6 / 40 |

Two entries need reading carefully.

**The pre-pause slope's AUC of 0.175 is an AUC of 0.825 with the sign
flipped.** The terminal pause is the one with the **most negative**
slope. A grader that reported "0.175" without saying that would be
reporting a number that looks like evidence against its own
conclusion. The tool records the sign it chose in
`separation.<feature>.sign`.

**The pre-pause mean pitch's AUC of 0.117 is the same story** — the
terminal pause is preceded by *lower* pitch. That is declination:
pitch falls across the sentence and the sentence-final pause sits at
the bottom of the fall. It is the expected direction, and it is
measured rather than assumed.

**The best leave-one-out accuracy is 0.913, on the weakest-sounding
observable.** Mean pitch before the pause is not a cue anyone would
design on purpose; it works here because the script corpus is a single
reader reading four sentences, so declination and session identity are
entangled. This is exactly the degeneracy
`docs/research/MOUTH-DETECTOR.md` §5.1 names — "a detector that
separates sessions is not a speaking detector" — and it is why the
free-corpus transfer in §5 matters more than the numbers above.

### 4.1 The thresholds these AUCs would give you are overfitted, and here is the proof

An AUC says how well an observable *could* rank the two classes. It
says nothing about where to put the line. So the tool also fits a
single cut, and reports **two** of them:

| observable | in-sample best cut | its accuracy | **robust cut** (midpoint of the class medians) | its accuracy |
|---|---:|---:|---:|---:|
| duration | **0.800 s** | 0.913 | **0.650 s** | 0.826 |
| pre-pause slope | **−81.8 st/s** | 0.870 | **−8.19 st/s** | 0.761 |
| pitch step | 5.88 st | 0.889 | 2.29 st | 0.733 |
| pre-pause mean pitch | 100.5 Hz | 0.935 | 107.8 Hz | 0.739 |

**The in-sample cut for the pitch slope is −81.8 st/s.** That is not a
threshold; it is one internal pause at −81.8 st/s being mistaken for a
boundary between the classes. It separates the data perfectly and
separates *nothing* in the world. Every in-sample cut in the table
above is at least that contaminated, and the leave-one-out accuracies
in §4 are the ones to believe — they are consistently 4 to 8 points
lower than the in-sample numbers.

**Every threshold used from here on is the robust cut.** The in-sample
cut is reported because the overfit is worth seeing, not because it is
usable.

---

## 5. The free corpus: the same observables, no labels, and the transfer

The 103 interior pauses of the five speech sessions have no ground
truth, so no AUC is computed on them. What can be said is what the
distributions look like, and what the rule calibrated on the script
does when it is pointed at them.

| quantity | script corpus (labelled) | free corpus (103 pauses) |
|---|---|---|
| duration median | 0.31 s (internal) / 0.90 s (terminal) | **0.65 s** |
| duration p75 / p95 | — | **1.14 s / 3.72 s** |
| duration max | — | **31.71 s** |
| pre-pause slope median | −20.1 st/s (terminal) | **−2.5 st/s** |
| post-pause slope median | +1.0 st/s (terminal) | **+0.8 st/s** |

The free corpus's pauses are **twice as long** as the script corpus's
internal pauses, which is what one would expect of informal speech
with real hesitation in it. The free corpus's median pre-pause slope
is slightly falling where the script corpus's terminal slope is
strongly falling.

**The transfer is not established, and the reason is specific.** The
script corpus is a single reader reciting four finished sentences; the
free corpus is a person talking to himself mid-thought. The rule that
separates "the reader reached the end of a sentence" from "the reader
stopped for a breath" was calibrated on the first population and is
being asked about the second. **A pause of 0.9 s in free speech is
usually a hesitation, and in the script corpus it was usually a
sentence end.** Duration alone does not carry across.

### 5.1 The tail is where the endpointer will be hurt

The free corpus's pause distribution has a long tail: p95 is **3.72 s**
and the longest interior pause is **31.71 s**. A 31 s pause in a
free-talk recording is not a turn boundary, it is a person thinking.
Any rule that yields the floor on a pause longer than N seconds will
fire on the thinking pauses, which is precisely the failure the brief
names when it says the thinking pause must hold the floor.

### 5.2 Two artefacts worth knowing about before building on this

**The pitch tracker produced octave errors.** The free corpus's
pre-pause slope distribution runs to **+344 st/s**, which is not a
slope, it is a half-doubling inside one 300 ms window. `docs/
research/F0-SPEAKER-CHANGE.md` already records that the biggest F0
jumps in this repository's material are tracker artefacts rather than
voices. Any slope feature needs an octave-error guard before it is
used on a live stream.

**The ASR hallucinates on the sonar stimulus.** The probe session
`1792596245` is a 16–22 kHz exponential sweep with a pink burst, and
faster-whisper returned **14 words** from it. The other three probes
returned none. The probe artefacts are excluded from every free-corpus
statistic in this document, and the count is reported in the results
file under `probe_artifacts` — but it is a standing warning: on a
corpus with non-speech stimuli in it, an ASR word count is not a
speech detector.

---

## 6. The power statement, which is the weakest part of this document

**Six terminal events.** Every AUC in §4 is computed on six positive
cases. The bootstrap intervals are wide and two of them come close to
0.5. The class medians in §3 are ordered in the expected direction on
every observable, which is encouraging, but a consistent ordering on
six events is also what you get from a systematic confound — and §4
names one: declination and session identity are entangled in a
four-sentence reading.

The only way out is more terminal events, and the corpus has three
sources of them and no more:

1. **More of the same script.** Each new reading of the passage adds
   four terminal pauses and thirty internal ones. It does not fix the
   confound — it repeats it, in the same four sentences.
2. **The dialogue protocol, scene 3.** Two microphones, a hard
   question, 2–10 s of real silence before any word. That is the
   labelled THINKING pause, and it is the class the brief actually
   needs and the script corpus cannot supply: a script cannot tell you
   what a reader was thinking, because a script has already decided.
3. **Free speech, labelled by a referee.** Expensive, and the referee
   cannot certify itself.

**So the honest position is: the first thinking-vs-terminal separation
exists and is measured, it rests on six events, and the free corpus
does not confirm it.** What it does give the endpointer design is a
*conjunction*, not a threshold — and the conjunction, graded as a
whole, is **worse than the AUC table above suggests**.

### 6.1 The conjunction, graded as a whole, and it is not good enough

The brief wants a conjunction. So the tool builds one and grades it as
one, at the robust cuts from §4.1. "Fired" means the rule would
**yield the floor** at that pause.

| leg | fires on terminals | fires on non-terminals |
|---|---:|---:|
| duration (> 0.65 s) | 5 / 6 | 7 / 40 |
| pre-pause slope (< −8.19 st/s) | 4 / 6 | 9 / 40 |
| pitch step (> 2.29 st) | 4 / 6 | 10 / 40 |
| syntax (pre-word is not a function word) | **6 / 6** | 15 / 40 |
| **all four together** | **2 / 6** | **1 / 40** |

**The four-leg conjunction misses four of the six terminal pauses and
holds 39 of the 40 non-terminal ones.** A 67 % miss rate on the class
the endpointer exists to detect, at a 2.5 % false-fire rate. That is
the honest measurement and it is the most important number in this
document, because the AUC table in §4 reads like a success and this
reads like a failure. Both are the same data.

The four misses, and why:

| pause | duration | pre-slope | pitch step | which leg refused it |
|---|---:|---:|---:|---|
| A `floor.` | 0.98 ✓ | −21.0 ✓ | 2.03 ✗ | the pitch step was 2.03, just under the 2.29 cut |
| B `floor.` | 0.56 ✗ | −19.3 ✓ | 3.15 ✓ | it was 0.09 s under the duration cut |
| A `up.` | 0.96 ✓ | **+2.87** ✗ | 0.83 ✗ | the pitch *rose* into it |
| B `up.` | 0.95 ✓ | −7.13 ✗ | 5.88 ✓ | the slope missed the cut by 1.06 st/s |

**Three of the four misses are near-misses on a cut fitted to six
points.** That is the shape of the problem: with six positives, a
conjunction of four legs each with its own threshold multiplies the
uncertainty four times. The rule is not badly designed; the *evidence*
is six events, and a four-leg conjunction needs more.

**The syntax leg is the one that carries.** It fires on **6 of 6**
terminal pauses and **15 of 40** non-terminal ones: zero misses, a
37.5 % false-fire rate. Its false fires are the internal pauses that
happen to end on a content word — and the fix for those is not a
better threshold, it is the duration leg, which catches exactly that
kind of pause. Which is the point: **the legs are not independent and
should not be multiplied.** The practical rule from this data is the
**syntax leg as a gate with the duration leg as a veto**, not four
conditions ANDed.

### 6.2 What this says the endpointer should actually be

Not a threshold and not a four-way conjunction. A **gate**:

> **Yield the floor only when the utterance is syntactically complete.**
> A pause after a function word holds the floor, whatever its length.
> A pause after a content word yields the floor only if it is long
> relative to this speaker's own pause distribution.

That ordering puts the leg with zero misses where it can do the most
work and lets the noisier legs only *veto*. On this corpus it would
fire on 6 of 6 terminals, and its false fires would be reduced by the
duration veto from 15 to 7. **Every number in this paragraph is a
measurement on six positives and forty negatives, and none of it is a
claim about free speech.**

---

## 7. The dialogue protocol: the design, and the fact that it is empty

`examples/realdialogue/README.md` specifies eight scenes totalling
**1230 s**. `examples/realdialogue/raw/` contains **no files**. The
tool records this in `protocol_design.n_files_recorded` so the claim
cannot rot.

| scene | s | channels | what it would supply |
|---|---:|---|---|
| 1 free conversation | 300 | both | the unpredictable gaps |
| 2 question, then answer | 180 | both | a real hand-off with a real gap |
| **3 thinking pause** | **180** | **both** | **the labelled thinking pause: 2–10 s of real silence before any word** |
| 4 turn-holding fillers | 120 | A only | the negative class — long in practice, **no silence at all** |
| 5 backchannel | 180 | both | a different speaker, near-zero gap |
| 6 deliberate interruption | 120 | both | overlap with ground truth |
| 7 one speaker alone, long pauses | 120 | A only | the pure negative control — silence, no hand-off |
| 8 room tone | 30 | neither | the noise floor |

**Scene 3 is the corpus this study most needs and does not have.** The
brief says of it: *"This scene is the single most important one in the
list… nobody scripts a 2 s think on purpose in a meeting."* A script
cannot manufacture a thinking pause; a script can only manufacture a
pause that the script decided on. The difference between the terminal
pause and the thinking pause is exactly the difference the corpus
cannot currently show.

**Scene 4 is the control that would keep the study honest.** A
turn-holding filler is a long, awkward, *silence-free* stretch: if the
endpointer's pause rule fires there, it is interrupting someone
mid-thought, which is the failure the brief says is not recoverable.

One more thing the protocol says that bears on the endpointer, and it
is a constraint rather than a suggestion: **"Do not read from a
script."** The script corpus used here is exactly the material the
protocol says cannot settle the turn question. It is used here for a
different job — it labels pause *position* without a rater — and the
report never claims it settles the turn question.

---

## 8. The cross-take test, which is the most transferable thing here

The script pair allows a test that free speech cannot: **does a
feature measured at a script position reproduce when the same position
is read again, independently?** Fifteen positions were paused in both
takes.

| script position | class | A duration | B duration | A pre-slope | B pre-slope |
|---|---|---:|---:|---:|---:|
| 19 `floor.` | terminal | 0.98 | 0.56 | **−21.0** | **−19.3** |
| 38 `drink.` | terminal | 0.85 | 0.80 | **−22.9** | **−31.0** |
| 50 `up.` | terminal | 0.96 | 0.95 | +2.9 | −7.1 |
| 58 `curb,` | beat | 0.63 | 0.66 | **−22.8** | **−29.5** |
| 17 `across` | internal | 0.34 | 0.18 | +11.1 | **−6.7** |
| 10 `the` | internal | 0.26 | 0.62 | **−10.9** | **+6.5** |

The terminal pause's falling slope reproduces across independent takes
— same sign, same order of magnitude — while the internal pause's
slope **flips sign between takes**. The declination at a sentence end
is a property of the *position*; the slope at a mid-phrase hesitation
is a property of the *moment*, and it will not transfer to the next
take of the same sentence.

The duration ratios run from 0.25 to 2.38 — a pause's length is much
the least reproducible observable of the four.

**This is the finding the endpointer design should lean on.** A
feature that reproduces across two independent deliveries of the same
material is a feature of the material. A feature that does not is a
feature of the occasion, and it will not generalise to the next
occasion. The pause's *length* does not transfer. The pitch's *fall
into* it does.

---

## 9. What this hands the endpointer design

| the endpointer needs | what this study gives it | how firm |
|---|---|---|
| a floor threshold | the internal-pause distribution is **0.19 / 0.31 / 0.54 s** at p25 / median / p75 | firm, on 32 events |
| a yield threshold | terminal median **0.90 s**, p25 **0.80 s**; robust cut **0.65 s** | 6 events, directional |
| a prosodic leg | pre-pause slope **−20.1 st/s** vs **+4.0 st/s**; pitch step **+4.5** vs **−0.1** st | 6 events, directional, and it reproduces across takes |
| a syntactic leg | pre-word is a function word: **0/6** terminal, **23/32** internal | firm on the script, and it is the leg with the largest margin — **6/6 terminals, zero misses** |
| the shape of the rule | **gate + veto**, not four conditions multiplied: the four-leg conjunction misses **4/6** terminals | firm, on 46 labelled events |
| the free-corpus priors | pause duration median **0.65 s**, p75 **1.14 s**, p95 **3.72 s**, max **31.7 s** | firm, on 103 pauses |
| a guard | the tracker produced **+344 st/s** slopes; the ASR produced **14 words** from a sine sweep | firm |
| a corpus that would settle it | protocol scene 3, **0 s recorded** | the blocker |

**The recommendation is a gate, and the gate is syntax.** The brief
already says the OVER word must be "a conjunction of terminal
position + its own prosody + trailing silence". This study says the
*pause* rule needs a different shape: **the syntactic leg as the
gate** — yield the floor only when the utterance is syntactically
complete — **with the duration and prosody legs as vetoes** that can
only hold the floor, never release it. Multiplying the legs is what
produces the 4-of-6 miss rate in §6.1, because each leg's threshold
carries its own uncertainty and four of them multiply.

**And the fourth clause is the one this corpus cannot supply**: a
duration that is long *relative to the population the pause is drawn
from*. In the script corpus a 0.9 s pause was usually a sentence end;
in the free corpus a 0.9 s pause is usually a hesitation. A threshold
that is not per-speaker and per-population is the wrong threshold.

---

## 10. Provenance

Every number in this document is in
`docs/research/pause_study_results.json`. The file carries the
absolute path of every input clip, the source and analysis sample
rates, the resample note (`src/wavio.py` resamples without
anti-aliasing: the time base is exact, content near the old Nyquist
folds), the exact segmentation parameters, the ASR device and model,
the full pause list with every feature on it, the per-class rows, the
bootstrap intervals, the cross-take pairs, the nulls and the probe
artefacts, and the SHA-256 of the script that produced it.

The nulls are the check that the method does not invent pauses:
`C.wav` (facing, silent) and `empty1.wav` (nobody talking) each
produce **one** silence segment, covering the whole file, and **zero**
interior pauses.

Regenerate with:

```
export LD_LIBRARY_PATH="/home/proart/projects/speech-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/proart/projects/speech-venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/proart/projects/speech-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
/home/proart/projects/speech-venv/bin/python experiments/pause_study.py --out docs/research
/home/proart/projects/speech-venv/bin/python experiments/pause_study.py --selftest
```

The ASR words are cached in `docs/research/pause_study_asr_cache.json`,
so a re-run does not re-transcribe unless `--no-cache` is passed.

---

## 11. What this does not establish

- **It does not establish a thinking-vs-terminal classifier.** It
  establishes six terminal events and the observables that separate
  them from internal pauses in a scripted reading. The THINKING class
  in the brief's sense — a pause inside a thought the speaker was
  still having — is not in the corpus at all.
- **It does not establish that the separation transfers to free
  speech.** §5 shows the duration prior moves by a factor of two
  between the two populations, and the free corpus has no labels with
  which to check.
- **It does not establish anything about the OVER word.** The OVER
  detector is ZCode's lane; this study supplies the pause side of its
  conjunction and nothing else.
- **It does not use the video.** Every pause here is an audio event.
  The corpus's motion layer is available and untouched by this study,
  and the room map already knows where the person is. A pause whose
  video shows movement and a pause whose video does not are different
  events, and nothing in this document can tell them apart.
- **Its statistics rest on one reader.** Casey's voice, one passage,
  two takes. The free corpus widens the population and loses the
  labels at the same time, which is the trade the next corpus has to
  break.

Committed for Muse to review; not merged.
