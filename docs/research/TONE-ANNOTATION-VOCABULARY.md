# TONE-ANNOTATION VOCABULARY — the marks that survive the loss of the MIDI

**What this is.** The retention brief's addendum says that before a
MIDI stretch is let go, **its essentials graduate into transcript
annotations** — "the sarcasm, the question-rise, the earnest weight
become marks on the words (a small versioned annotation vocabulary:
bracket/footnote/hash conventions to be designed from evidence)". The
annotations are judgements: they ride as a revisable derived layer with
provenance, never baked into the descriptive transcript.

This is that vocabulary, designed from the evidence that exists. The
evidence is two studies' worth:

- the **tone game's six measured intents**, each with a ten-field style
  vector, which IS the MIDI control surface — so the delta between two
  intents is a delta in the score's own parameter space
  (`experiments/tone_game_results.json`);
- the **pause study's taxonomy**, which gives the marks that are about
  the *boundary* rather than the delivery (the terminal/internal split,
  the fall-or-rise into the pause).

**The short version.**

1. **Eight marks in three families**: six DELIVERY marks in brackets,
   one DELIVERY mark in brackets that comes from the pause study, one
   STRUCTURE mark as a hash, and a PROVENANCE footnote.
2. **Every mark has a one-of-six anchor**, and the anchor is a field
   exactly one intent owns: `zig` belongs to `joking` alone, a `range`
   of 3.00 belongs to `obligated` alone, a `drop` above 3 belongs to
   `quiet-furious` alone. A mark that cannot name the feature that
   makes its class unique is not a mark, it is a label.
3. **Four of the ten style fields cannot carry a mark at all.** Their
   range across the six intents is 0.35 to 0.70 units (`density`,
   `jitter`, `pause`, `hold`), while the four that can carry one are
   `centre`, `tilt`, `arc` and the two rare flags `zig` and `drop`.
4. **The pause study's two axes are the vocabulary's backbone**: is the
   utterance *complete* or *continued*, and did the pitch *fall* or
   *rise* into the pause. Four cells, all measured.
5. **Which marks survive the MIDI's loss is a measured question, and the
   answer is not "all of them".** `#` and `[q]` survive; `[e]` and `[o]`
   degrade; `[s]`, `[f]`, `[l]` and `[x]` do not survive at all. That
   ordering IS the graduation schedule: graduate the un-survivable
   ones first.
6. **The honest limits**: `[s]` is anchored on a **proxy** (the tone game
   has no sarcasm intent), `[q]` rests on **one event** in the corpus,
   and the tone game's own teacher-dependence ratio is **1.0** — no
   referee has graded anything, so none of these marks is certified.

---

## 1. Where the vocabulary comes from, and what it is not

The addendum's phrase is "the sarcasm, the question-rise, the earnest
weight". Three examples, and they do not all come from the same place:

| the example | what carries it | where the evidence is |
|---|---|---|
| the earnest weight | the delivery's shape: a small rise, no zig, no drop | the tone game's `sincere` intent |
| the sarcasm | a high centre with a **falling** tilt and a zig | the tone game's `joking` intent — a **proxy** |
| the question-rise | a completed utterance whose pitch **rose** into the pause | the pause study's terminal class, rising cell |

The third is not a tone-game intent at all. It is a **boundary**
property, and it is measured in the pause study: a terminal pause is
preceded by a **falling** pitch (median **−20.1 st/s**) and an internal
pause by a **rising** one (median **+4.0 st/s**), and the single
terminal pause in the corpus that broke that rule — A's "up.", at
**+2.87 st/s** — is exactly the shape of a question.

**What the vocabulary is not:** it is not a sentiment scale, and it is
not a set of adjectives. Each mark names **a measured, reproducible
difference in the score's own parameters**, and each carries the
provenance of the measurement that made it. A vocabulary of adjectives
would be a transcription of the model's opinion; a vocabulary of
one-of-six anchors is a transcription of the measurement.

---

## 2. The style space, and which of its fields can carry a mark

The tone game's six intents, with the style vector each one owns. These
are the **MIDI control surface**, so every number here is a number in
the score.

| intent | centre | range | tilt | arc | zig | drop | density | jitter | pause | hold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sincere | +1.50 | +7.00 | +1.50 | +2.50 | 0.00 | 0.00 | 1.00 | 0.50 | 0.25 | 1.30 |
| leaving | −1.00 | +8.00 | −4.00 | +1.00 | 0.00 | 0.00 | 1.00 | 0.60 | 0.45 | 1.55 |
| obligated | 0.00 | **3.00** | 0.00 | +0.40 | 0.00 | 0.00 | 1.00 | 0.30 | 0.15 | 1.05 |
| everything | +4.25 | +8.00 | +5.00 | **5.00** | 0.00 | 0.00 | 1.70 | 1.00 | 0.20 | 1.20 |
| joking | **+7.75** | +8.00 | −2.25 | +2.25 | **5.00** | +2.25 | 1.50 | 0.45 | 0.10 | 0.95 |
| quiet-furious | −6.00 | +6.00 | −1.00 | 0.00 | 0.00 | **5.00** | 1.10 | 0.40 | 0.35 | 1.45 |

And which fields can carry a mark at all:

| field | range across the intents | mean step between neighbours | intents with any non-zero value |
|---|---:|---:|---:|
| **centre** | **13.75** | 5.00 | 5 / 6 |
| **tilt** | **9.00** | 4.60 | 5 / 6 |
| **arc** | 5.00 | 2.34 | 5 / 6 |
| range | 5.00 | 2.60 | 6 / 6 |
| **zig** | 5.00 | 2.00 | **1 / 6** |
| **drop** | 5.00 | 1.00 | **2 / 6** |
| density | 0.70 | 0.26 | 6 / 6 |
| jitter | 0.70 | 0.34 | 6 / 6 |
| hold | 0.60 | 0.33 | 6 / 6 |
| pause | 0.35 | 0.18 | 6 / 6 |

**The bottom four cannot carry a mark.** `density`, `jitter`, `pause` and
`hold` move by less than a unit across the whole intent set, and
`pause` — the field that sounds like it should carry the most — is the
weakest of the ten. A mark anchored on `pause` would be noise dressed
as a judgement.

**The two rare flags are the best anchors there are**, and that is
counter-intuitive until you see it: a feature that exactly one intent
has cannot be confused with anything.

---

## 3. The two axes, and the backbone they make

The pause study's measurements give the vocabulary a second axis, and
the two together are the backbone.

|  | pitch **falls** into the pause | pitch **rises** into the pause |
|---|---|---|
| the utterance **completed** (terminal) | the ordinary end of a sentence: median **−20.1 st/s**, 6 events | **a question**: the one terminal pause in the corpus that broke the rule, A's "up.", **+2.87 st/s** |
| the speaker **continued** (internal) | a beat or a breath: median **−0.4 st/s**, 8 events | **a hold**: median **+4.0 st/s**, 32 events, and the word before it is a function word **23 times out of 32** |

Both rows of the left column and both rows of the right column are
measured; the top-right cell is measured **n = 1**, and the report says
so.

---

## 4. The vocabulary

### 4.1 The marks

| mark | family | asserts | the measured anchor | one-of-six? |
|---|---|---|---|---|
| `[e]` | DELIVERY | delivered earnest: a small rise, no zig, no drop | `sincere` — tilt +1.50, arc +2.50, zig 0, drop 0, hold 1.30 | no — it is Casey's **dominant** mode (prior 0.7), so it is the default and the least informative |
| `[l]` | DELIVERY | a sign-off: a falling delivery with a long hold | `leaving` — tilt **−4.00** (the steepest fall), hold **1.55** (the longest), pause **0.45** (the longest) | the steepest fall in the set |
| `[o]` | DELIVERY | flat and constrained: a narrow, level delivery | `obligated` — range **3.00**, the only intent below 6.00 | **yes** |
| `[x]` | DELIVERY | exuberant: a big rise with a big arc | `everything` — arc **5.00** (the only one), tilt +5.00, density 1.70 (the highest) | **yes** on arc |
| `[s]` | DELIVERY | sarcastic: a high centre with a falling tilt and a zig | `joking` — centre **+7.75** (the highest), tilt **−2.25** (negative), zig **5.00**, drop 2.25, hold 0.95 (the shortest) | **yes** on zig — but see §6 |
| `[f]` | DELIVERY | cold anger: a low centre with a drop | `quiet-furious` — centre **−6.00** (the lowest), drop **5.00** (the only drop above 3) | **yes** |
| `[q]` | DELIVERY | a question end: the utterance completed **and** the pitch rose into the pause | the pause study's terminal+rising cell — **n = 1** | a different source |
| `#` | STRUCTURE | the speaker **held the floor** here and continued the same thought | the pause study's internal class — the pre-word is a function word 23/32, the post-word is lowercase 32/32, the pitch rose into it | — |
| `[^n]` | PROVENANCE | which measurement made this judgement | the run and the vocabulary version | — |

### 4.2 The shapes, and why they differ

The three families have three shapes because they make three different
kinds of claim:

```
"the delivery was quiet[^tg1]"          a DELIVERY mark, on a word
"and I was thinking about it#            a STRUCTURE mark, on a word
"really?[q][^ps1]                       two DELIVERY marks and a PROVENANCE note
```

- **Square brackets** are for **judgements about a span of delivery**.
  They are openable — a span may be marked at its end, and the bracket
  says "this judgement is about how these words were said". They are
  also the shape the addendum named.
- **A hash** is for a **judgement about a boundary**. A hash attaches
  to one word and asserts what happened *after* it: the speaker held
  the floor here. A boundary is a point, so the mark is a point. Using
  brackets for it would claim a span where there is a point.
- **A footnote** is for **provenance**, and it is the reason the
  annotations are revisable. `[^tg1]` says "this was measured by the
  tone game, vocabulary version 1"; `[^ps1]` says "this was measured by
  the pause study, version 1". A judgement with no footnote is not
  admissible, because a judgement that cannot be traced cannot be
  revised.

### 4.3 The placement rules

1. A DELIVERY mark sits on **the last word of the span it judges** —
   for `[q]`, the word before the pause the pitch rose into.
2. A STRUCTURE mark sits on **the word the speaker continued from** —
   in the measured corpus, the function word.
3. A mark is **never** placed on a word the layer that produced it does
   not cover. If the score is missing, the DELIVERY marks are missing.
   Not guessed, not inferred from the words: absent.
4. Every mark carries a footnote, and the footnote carries the
   vocabulary version. A later version of the vocabulary may re-mark
   the same word; both marks stay, and the later one wins.

---

## 5. Which marks graduate, and which do not

The addendum's rule is that a MIDI stretch's essentials graduate into
the transcript annotations *before* the MIDI is let go. So the question
is: **which marks can still be read once the MIDI is gone?**

| mark | what it needs | survives the MIDI's loss? |
|---|---|---|
| `#` | the pre-word is a function word and the speaker continued | **YES — entirely.** It is linguistic. |
| `[q]` | the utterance completed | **YES**, but degraded: the *terminal* part is linguistic, the *rising* part is not, so the mark survives as "a question end" without the rise. |
| `[e]` | the **absence** of zig and drop | **PARTLY.** The absence is inferable from the transcript; the small rise is not. |
| `[o]` | range 3.00 against 6.00–8.00 | **PARTLY.** A flat, unvaried delivery is partly audible in a monotonous transcript, but "range 3.00" is a number in the score. |
| `[l]` | tilt −4.00, hold 1.55, pause 0.45 | **NO** |
| `[x]` | arc 5.00, density 1.70 | **NO** |
| `[s]` | centre +7.75, tilt −2.25, zig 5.00 | **NO** |
| `[f]` | centre −6.00, drop 5.00 | **NO** |

**That ordering is the graduation schedule.** Four marks cannot be read
off a transcript and must be graduated *before* their MIDI stretch is
let go: `[l]`, `[x]`, `[s]`, `[f]`. Two can be re-derived and need not
be: `#` and `[q]`.

**And the direct measurement that a delivery mark cannot be recovered
from the words** is the tone game's `other_carrier` ablation: the same
six styles rendered on a **different phrase**, graded by the same
reader, scored **1.0 and 1.0** — and the experiment's own note is
that **"the words carry nothing in either case."** The intents are
carried by the delivery. A transcript without the delivery carries
nothing about them, which is precisely why the marks have to exist.

One supporting number: the `free_note_count` ablation, where the reader
may use how many notes were sung, scored **0.9583** in round 0 and
**1.0** thereafter. So the note count carries a *little* information —
one sincere/joking confusion in the first round and none after. It is
not nothing, and it is not much.

---

## 6. The limits, stated before anyone uses this

- **`[s]` is anchored on a proxy.** The tone game has no sarcasm
  intent; `joking` is the nearest measured neighbour, and a joke
  delivered straight is not a sarcasm delivered dry. The mark `[s]`
  asserts something the corpus cannot measure directly, and it should
  be filed with that caveat until a sarcastic exemplar exists.
- **`[q]` rests on one event.** The terminal+rising cell has **n = 1**:
  A's "up.", at +2.87 st/s, which the pause study's conjunction missed
  because a rising slope contradicts its own falling-slope leg. The
  classification of that pause as a *question* is the study's reading,
  not a measurement of one.
- **`[e]` is the default.** Casey's prior over the intents is **0.7**
  for `sincere`, so `[e]` is the mark a listener will want to apply
  most and the one that says least. A vocabulary whose most-used mark is
  its least informative mark is a vocabulary that will be applied
  without thought.
- **No referee has graded any of it.** The tone game's
  teacher-dependence ratio is **1.0**: *"the reader's training labels
  all come from the performer's own priors, and no referee has graded
  anything yet, so not one of its judgments is certified."* The
  listening pack exists (12 items, the key held back for blind
  grading), but the grading has not happened. **The vocabulary is
  therefore a set of marks with measured anchors and no certified
  judgements**, and it must not be used as if it were the latter.
- **The envelope is measured from two readings of one passage.** The
  style envelope — note duration median 0.20 s, step median −0.239
  semitones, pitch range 6.757 semitones, centre MIDI 45.71, 4.673
  notes/s — was pooled from **A and B**. The deltas are therefore
  calibrated against one person's reading of one text.

---

## 7. What the vocabulary becomes, and what it does not

The vocabulary is a **derived layer**, and the addendum is explicit
about what that means: revisable, with provenance, never baked into the
descriptive transcript. Concretely:

- the **descriptive transcript** stays clean. It is what the
  transcript layer is for.
- the **annotated transcript** is a second artefact, derived from the
  descriptive transcript plus the score, carrying the marks and their
  footnotes.
- when the MIDI stretch is let go, the **annotated transcript survives**
  and the score does not. The mark is the thing that was graduated.
- when the vocabulary is revised, **the annotations are re-derived from
  the score if the score is still there**, and **left alone if it is
  not** — with the older version's marks still in place and their
  footnotes still naming the version that made them. A revision never
  silently overwrites a judgement.

**The calculator this graduates into** is a **tone annotator**: a cheap
mechanical detector that reads the score's style vector and writes the
marks, at the cost of one pass over the score and no model at all. Its
precision is bounded by the tone game's own confusion matrix — which is
perfect in the synthetic game and **certified by nobody** in the real
one. That is the next measurement, and it is the one the listening
pack was built for.

---

## 8. What this does not establish

- **It does not establish that the six tone-game intents are the six
  ways Casey sounds.** They are the six ways the *tool* was built to
  sound, and the tool's author chose them. A vocabulary designed from
  them inherits that choice, and the choice has not been tested against
  Casey's own sense of what he sounds like.
- **It does not establish a sarcasm detector.** It establishes that
  `joking` has a one-of-six anchor, and that no sarcasm exemplar
  exists.
- **It does not establish that the marks improve anything.** They are
  ungraded. The teacher-dependence ratio of 1.0 is the honest number.
- **It does not choose where the marks are written down.** The
  convention here is a proposal: the addendum asked for a
  bracket/footnote/hash vocabulary "to be designed from evidence", and
  the evidence has been used to design the marks and their anchors. The
  *file format* is ZCode's, and this document does not prescribe it.
- **It does not resolve the conflict between the tone game's `tilt` and
  the pause study's fall.** `everything` has the steepest rise
  (+5.00) and the pause study found that a terminal pause is
  preceded by a fall (−20.1 st/s). These are not necessarily in
  conflict — one is a phrase-level tilt, the other a 300 ms window at
  the boundary — but nothing here has measured them on the same
  material, and the vocabulary's `[q]` mark sits exactly where the two
  would disagree if they were.

Committed for Muse to review; not merged.
