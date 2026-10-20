# A speaker embedding resolves one of the two speaker pairs pitch cannot

**Status: one pair resolved, one pair not.** Written after measuring, not
before. The pair that failed is a clean null and is reported as one.

`docs/research/F0-SPEAKER-CHANGE.md` closed a door and said which one:

> "A large F0 shift is intonation inside a phrase, not a speaker change. F0
> cannot rescue this corpus as a speaker-change proxy, and no threshold on
> pitch [can]."
> "across four speakers, median F0 separates two *voice classes* cleanly --
> men at 104-111 Hz against women at 197-208 Hz -- but cannot resolve the two
> speakers *within* each class. FEO070 and FEO072 sit **0.5 sigma** apart;
> MEE071 and MEE073 **0.3 sigma**. Pitch proves there are two voices, not who
> is speaking."

And it named the door out:

> "It would take a speaker embedding, and the far-field single microphone
> would still have to resolve two simultaneous voices, which 203 audible
> segments across 19,305 voiced frames say is marginal at best."

This document is that experiment: `microsoft/wavlm-base-plus-sv`, on the same
AMI turns, with the pitch measurement rerun on the same turns so the two
instruments are compared on the same data rather than in prose.

**The answer in one line.** On the same turns, in the same units, the
embedding separates **MEE071 from MEE073 at 2.6326 sigma** where pitch
manages 0.5229, and it separates **FEO070 from FEO072 at 0.1577 sigma**
where pitch manages -0.0057. One pair is rescued at the operating point. The
other is not, and it stays unresolvable for as long as the turns are long
enough to be measured at all: it only reaches 1.2806 sigma at 3.0 s turns,
where FEO070 is down to 14 turns and the enrolment/test split is no longer
sound.

---

## The instrument, and what it needs

| | |
|---|---|
| model | `microsoft/wavlm-base-plus-sv` (WavLM + x-vector head) |
| revision | `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e` |
| class | `transformers.AutoModelForAudioXVector`, 512-d embedding |
| audio | `src/wavio.py:read_wav`, mono, 16 kHz, no resampling |
| preprocessor | the model's own `preprocessor_config.json`: `do_normalize=false`, `return_attention_mask=true` |
| embedding unit | the WHOLE annotated turn, one embedding per turn |
| distance | cosine distance, `1 - cos` |
| corpus | `examples/realtwospeaker/`, EN2002a + EN2002b far-field, 361 audible turns |

**The instrument was checked against its own model card first.** The card's
example is three utterances of one Librispeech speaker, with a published
same-speaker threshold of cosine similarity 0.86 (distance 0.14). Measured
here: **0.0272**. The model does what its card says.

**And then it had to be given enough audio.** This is the largest thing
learned while building the experiment, and it is the part most likely to be
repeated by anyone else who tries this.

The first version of this experiment embedded fixed 1.0-second windows, and
it produced a cross-gender control (FEO070 against MEE071 — a man against a
woman) of the *same order* as the same-gender separation it was trying to
measure. That is impossible for an instrument that works, and it is what
said the instrument was being starved rather than the corpus being hard.

The committed result file measures the starvation directly, on the model
card's own three utterances of one speaker:

| clip | same-speaker cosine distance | against the card's 0.14 threshold |
|---|---:|---|
| full utterances (4.8-12.5 s) | 0.0272 | comfortably inside |
| first 3.0 s | 0.0600 | inside, but 2.2x the full-utterance value |
| first 1.0 s | 0.2047 | **outside** |

One second of speech is about 49 frames, and the x-vector head pools statistics
over all of them: at that length the embedding is dominated by *what was said*
rather than *who said it*. The model card's 0.86 threshold is meaningless on a
one-second clip, and the same instrument on the same speaker gets 7.5x worse
when the audio is cut to one second.

So the embedding unit is the turn, and the measurement is reported as a
**sweep over the minimum turn length**, because the separation is a function
of how much audio the model was given. Reporting one point of that curve
without the curve would be exactly the mistake
`docs/research/HOW-WE-KEEP-FOOLING-OURSELVES.md` is about.

**A second defect, found the same way.** The x-vector head's TDNN needs a
minimum input length. Below roughly 2 s the forward pass returns NaN (or
raises `Kernel size can't be greater than actual input size`). Five of 394
boundary clips came back non-finite before this was found. Every utterance is
now padded to at least 2.0 s with the attention mask keeping the padding out
of the pooling, and the run records **0** non-finite embeddings.

This is not a rare corner: **209 of the 394 boundary clips are shorter than
2.0 s.** The padding rule is what makes the boundary task measurable at all.

---

## The measurement, in pitch's units

Both instruments are reduced to the same shape so the numbers are comparable:

* a **within-speaker distance distribution**: distances between two turns of
  the SAME speaker;
* a **between-speaker distance distribution**: distances between two turns of
  DIFFERENT speakers;
* the **separation**: `(centre of between - centre of within) / sigma of
  within`.

The F0 document's 0.5 / 0.3 sigma were a difference of centres over the
within-speaker spread, so the median-based form above is the directly
comparable one. The pitch baseline is computed here with the *identical*
arithmetic on the *identical* turns: the "distance" is `|delta f0|` between
two turns.

**The F0 document's own number reproduces exactly.** On the document's own
population (EN2002a, every audible segment not overlapping another speaker),
with the document's own arithmetic:

| pair | this file | the F0 document |
|---|---:|---:|
| pooled within-speaker MAD sigma | 22.28 Hz | 22.28 Hz |
| FEO070 vs FEO072 | **0.4756 sigma** | 0.48 |
| MEE071 vs MEE073 | **0.3288 sigma** | 0.33 |

The two documents cannot have drifted apart silently.

### The sweep

Separation in sigma, far-field, both meetings pooled. "F0 (same turns)" is
the pitch instrument in the identical arithmetic on the identical turns.
"F0 (doc arithmetic)" is the document's statistic on that turn subset.

| min turn | n turns | FEO070\|FEO072 emb | FEO070\|FEO072 F0 (same turns) | MEE071\|MEE073 emb | MEE071\|MEE073 F0 (same turns) | FEO070\|MEE071 emb (cross-gender control) |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 s | 281 | 0.0254 | -0.0322 | 0.3927 | 0.3232 | 0.4742 |
| 1.0 s | 218 | 0.1044 | -0.0435 | 0.9781 | 0.3634 | 0.9363 |
| 1.5 s | 177 | 0.1347 | -0.0448 | 2.0275 | 0.5501 | 1.5073 |
| **2.0 s** | **145** | **0.1577** | **-0.0057** | **2.6326** | **0.5229** | **1.5412** |
| 3.0 s | 89 | 1.2806 | 0.0197 | 3.9100 | 0.5925 | 2.8534 |

The headline operating point is **2.0 s**, chosen by a rule and not by the
answer: it is the longest minimum in the sweep that still leaves every
speaker with at least 20 turns (FEO070 23, FEO072 39, MEE071 31, MEE073 52),
so the enrolment/test split is not degenerate. It is *not* the point with the
largest separation. The 3.0 s row is in the table precisely so a reader can
see what a different rule would have bought, and what it would have cost
(FEO070 down to 14 turns).

**The pitch baseline is negative for FEO070 against FEO072 at every length
except the longest.** A negative separation means the two speakers'
turn-to-turn pitch differences are *smaller* than one speaker's differences
with themselves. On this corpus, pitch has no signal at all for that pair.

### The headline

Far-field, min turn length 2.0 s, 145 turns. 2-speaker verification: enrol on
one speaker's turns, test on that speaker's held-out turns and on the other
speaker's turns.

| | FEO070 vs FEO072 | MEE071 vs MEE073 | pitch's number, for comparison |
|---|---:|---:|---|
| within-speaker distances | 994, mean 0.1961 | 1791, mean 0.1669 | -- |
| between-speaker distances | 897, mean 0.2140 | 1612, mean 0.3610 | -- |
| **separation (sigma)** | **0.1577** | **2.6326** | 0.5 / 0.3 (the document) |
| ratio of the means | 1.0916 | 2.1625 | -- |
| ratio of the medians | 1.1481 | 2.7074 | -- |
| **2-speaker EER** | **0.333333** | **0.121951** | -- |
| 4-speaker EER (all four speakers) | -- | 0.169014 | -- |
| F0, same turns, same arithmetic | -0.0057 | 0.5229 | -- |
| F0, the document's arithmetic | 0.1260 | 0.8244 | 0.48 / 0.33 |
| shuffled-label null (median) | -0.00715 | -0.00555 | -- |
| permutation p | 0.024876 | 0.004975 | -- |
| **verdict** | **NOT resolved** | **RESOLVED** | -- |

The verdict rule is written down as numbers, not adjectives, and can come out
the other way: a pair is RESOLVED if the separation is at least 1.0 sigma,
the 2-speaker EER is at most 0.25, **and** the shuffled-label null is beaten
at p <= 0.05. MEE071/MEE073 passes all three. FEO070/FEO072 fails two.

**The two women's voices are not separable by this method.** 0.1577 sigma is
about a sixth of the within-speaker spread, and an EER of 0.333 against a
chance level of 0.5 is not a speaker-change detector. This is a clean null.

**The two men's voices are.** 2.6326 sigma, a 2-speaker EER of 0.122, a
between-speaker mean distance of 0.361 against a within-speaker mean of
0.167, and it beats every one of 200 label permutations (p = 0.004975).
Pitch, on the same turns in the same arithmetic, reaches 0.5229.

---

## The controls

**The same-speaker control.** 2785 distances between two turns of the *same*
speaker: mean 0.1773, median 0.1342, MAD sigma 0.1001. A method must not
fire on this population, and the headline is a *margin over it*: for the MEE
pair the between-speaker mean is 0.3610 against that 0.1773; for the FEO pair
it is 0.2140 against the same 0.1773, which is the null showing itself.

**The shuffled-label null.** 200 permutations of the speaker labels over the
turns, with the per-speaker turn counts preserved, separation recomputed from
scratch each time. The null distribution is centred on zero (medians
-0.00715 and -0.00555) and the observed separations sit above all but 4
(FEO070/FEO072) and **0** (MEE071/MEE073) of the 200 permutations
respectively. The separation is about speakers.

**Time reversal.** The waveform reversed in time: the MEE pair keeps
**1.7569** of its 2.6326, the FEO pair 0.0989 of its 0.1577. So the
measurement is about the voice and not only about the temporal detail of the
utterance -- but not entirely: about a third of the MEE separation is
reversal-dependent, and saying otherwise would be over-claiming.

**Gain.** Each turn multiplied by a log-uniform random gain in [0.1, 10].
This model's preprocessor has `do_normalize: false`, so gain is *not*
normalised away by construction. Measured: the MEE pair keeps **2.3542** of
2.6326, the FEO pair goes to -0.0230. The instrument is largely but not
perfectly gain-robust, and the residual is reported rather than assumed away.

**No speaker at all.** Each turn replaced by noise of the same length and
RMS: the separations collapse to **0.0244** and **-0.0032**. There is no
speaker in that audio and there is no separation in the number. This is the
control that matters most: it is the one that cannot be argued around.

**The degenerate baselines.** Always predicting "speaker changed" scores
**EER 0.5** by construction; so does always predicting "same". Both are in
the result file so the headline cannot be read without them. The
F0-only baseline on the same turns is in the table above: 0.5229 sigma and an
EER of 0.346154 at the boundary task.

**Cross-check on a clean channel.** The per-speaker headset channels, where a
speaker's identity is the *recording* and not an inference: MEE071/MEE073
separates at **4.3096** sigma and FEO070/FEO072 at **0.3413**. Same order,
same verdict, on 4 and 7 turns for the first pair and 2 and 6 for the second
-- far too few to carry weight on their own, but they do not contradict the
far-field result.

**The overlap population.** The F0 experiment excludes far-field segments that
overlap another speaker, because "a median f0 over them describes neither
speaker". An x-vector over cross-talk is a different object, and the
contamination should show up as an inflated *within*-speaker spread. It does:
on the non-overlapping subset alone, MEE071/MEE073 separates at **6.8695**
sigma against 2.6326 on the full population, and FEO070/FEO072 at 0.3049
against 0.1577 -- the same order, in both pairs. But that subset has only 4
and 24 turns for the first pair and 4 and 15 for the second, so it is
corroboration, not a second measurement.

---

## The turn-change task: the number the project actually needs

The project's need is not "who are these two people" but "did the speaker
change at this pause". So: consecutive annotated turns that do not overlap,
scored by the embedding distance between the last clip before the boundary and
the first clip after it. 197 boundaries -- 78 speaker changes, 119 same
speaker -- plus 57 within-turn controls (one turn cut in half, no change at
all).

| | EER |
|---|---:|
| **embedding, against same-speaker boundaries** | **0.260504** |
| embedding, against same-speaker + within-turn controls | 0.221591 |
| embedding, restricted to boundaries with >= 1.5 s on both sides | 0.175439 (8 positive, 57 negative) |
| F0 baseline, same boundaries | 0.346154 |
| constant decision | 0.5 |
| **shuffled-label null (median of 200)** | **0.5** |

The embedding beats pitch on the same trials (0.260504 against 0.346154) and
beats **every one of 200** label permutations (p = 0.004975; the null's median
is 0.5 and its best permutation still scored 0.402878).

The caveat is in the clip lengths: the median boundary has only **0.88 s** of
audio on its shorter side, and the shortest has 0.02 s. The model is being
asked the same question it answers badly -- and it still answers better than
pitch. That is the most encouraging single number here, and it is also the
one that would improve first if the boundary detector ran on longer context.

---

## The single-channel corpus: a null, and an exploratory one

`examples/realaudio/` has no per-turn speaker ground truth, so **nothing in
this section is evidence.** It is reported because the question was asked:
does an embedding-based speaker-change statistic behave sanely on the
seven single-channel cells, and does it separate the `dialogue` cell, which
contains two speakers?

A label-free two-cluster split of the pairwise embedding distances was
computed per cell, and the separation of the two cluster centres in units of
the within-cluster spread is the statistic:

| cell | windows | mean pairwise distance | 2-cluster separation (pooled sigma) | median F0 |
|---|---:|---:|---:|---:|
| narration | 59 | 0.1416 | 1.7158 | 229.4 Hz |
| poetry | 12 | 0.1393 | 2.1402 | 195.6 Hz |
| essay | 49 | 0.1343 | 3.1762 | 89.6 Hz |
| **dialogue** | 49 | **0.0805** | **1.7071** | 109.2 Hz |
| shakespeare | 49 | 0.3180 | 1.8963 | 154.5 Hz |
| children | 49 | 0.0580 | 1.7656 | 180.8 Hz |
| oration | 49 | 0.1189 | 1.8312 | 196.1 Hz |

**The `dialogue` cell does not stand out.** Its two-cluster separation
(1.7071) is indistinguishable from `narration` (1.7158) and `children`
(1.7656) -- both single readers -- and its mean pairwise distance (0.0805) is
the *second lowest* of the seven, below the single-reader `narration`. The
cell with two speakers is not the cell the embedding finds unusual.

The one cell that does look unusual is `shakespeare`, at 0.3180 mean
pairwise distance, nearly four times `dialogue` -- and `examples/realaudio/corpus/MANIFEST.md`
records it as a "full-cast dramatic" reading, i.e. *also* more than one
voice. So the statistic is behaving plausibly, and the cell that was expected
to stand out is not the one that does.

This is a null for the exploratory hypothesis, and it is reported as one.

---

## What was falsified

* **FEO070 against FEO072 is NOT resolved by the embedding.** 0.1577 sigma,
  2-speaker EER 0.333333. It fails the separation criterion and the EER
  criterion. Pitch could not resolve this pair either -- and the pitch
  baseline on the same turns in the same arithmetic is *negative*
  (-0.0057), meaning pitch had no signal at all -- but the embedding does not
  rescue it. **The escape route named by the F0 document does not work for
  this pair.**
* **The exploratory claim that the embedding separates the `dialogue` cell is
  falsified.** Its two-cluster separation is the same as the single-reader
  cells'.
* **My own first implementation was falsified by the model card's own
  example.** The 1.0-second-window version of this experiment produced a
  cross-gender control of the same order as the same-gender separation it
  was supposed to beat, and the model card's own same-speaker distance
  climbed from 0.0272 to 0.2047 when the audio was cut to one second. The
  control — a male against a female — is what made it impossible to keep
  believing. The measurement is reported as a sweep over turn length because
  of it.

And one defect was found in the *reporting* rather than the instrument: the
permutation test on the turn-change EER compared in the wrong direction
(higher-is-better, borrowed from the separation statistic) and reported
p = 1.0 for a result that beat every permutation. The direction is now a
parameter with the tail it uses written into the result file.

## What this does not show

* **It does not show that the two women are separable.** It shows the
  opposite, for this model, on this room, with these four people. A different
  model might do better; nothing here says one would.
* **It does not show a general speaker-change detector.** 197 boundaries from
  one meeting series, one room, four speakers, and an EER of 0.26 is not a
  deployable accuracy. The number is a *margin over three controls*, not a
  performance claim.
* **It does not validate the far-field channel.** The far-field microphone
  carries two voices at once in 24-30% of the annotated turns; the solo-only
  numbers are the ones that describe one voice at a time, and they are
  measured on 4-to-24 turns per speaker. The far-field numbers are the ones
  that describe the actual recording channel.
* **It does not say anything about the single-channel corpus.** That
  section is exploratory and its answer is null.
* **It does not separate the two men because of their pitch.** It does not
  use pitch at all. The relevant comparison is the F0 baseline on the same
  turns, and the embedding beats it by 5x on that pair and loses to it
  (marginally, and both at zero) on the other.
* **A separation measured one way is one measurement.** The sweep table is
  the honest form of this result: the separation moves by a factor of 20
  between a 0.5 s and a 3.0 s turn, and nothing here should be quoted
  without the length it was measured at.

## Reproduce

```bash
/home/proart/projects/gpu-venv/bin/python experiments/speaker_embedding_f0.py
```

Writes `experiments/speaker_embedding_f0_results.json`, which records the
model id and revision, the SHA-256 of every audio and segment file, the
preprocessing, the length sweep with its counts, every distance distribution
as a summary plus the raw score lists behind every EER, all four nulls, the
degenerate baselines, and the falsified claims.

The result file is checked by
`tests/test_speaker_embedding_f0.py`, which runs in the repo's own `.venv`,
needs no GPU, downloads nothing, and recomputes the headline separation and
every EER from the recorded numbers rather than asserting them. A null
result with its controls recorded passes that suite.
