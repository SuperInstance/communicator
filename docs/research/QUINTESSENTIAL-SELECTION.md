# QUINTESSENTIAL SELECTION — which clips are worth keeping, and how you find the ones that are hiding

**What this is.** The retention brief
(`docs/research/BRIEF-RETENTION-GC.md`) says the garbage collector is
really a **curation** system, and that its product is the
**QUINTESSENTIAL SET**: a small set of clips per day that are
quintessential of THIS person — "meaning carried beyond words and
beyond tone". It says the selection criteria are **measured, not
vibes**, and that the catches are the truth: keep the oddities, collect
the routine.

It also names the guardrail: collection is the one sanctioned deletion,
so it runs **dry first**, with audit samples re-checked for
quintessentials hiding in the routine pile, and the **miss rate**
reported. Live collection starts only when the miss rate is measured and
near zero, and Casey rules on it.

This is the criteria, the first candidate set, and the audit. The
measurements are in `docs/research/quintessential_results.json`,
produced by `experiments/quintessential.py` (self-test: 16 checks).

**The short version.**

1. **The corpus is 140 clips from 7 sessions** — 45 SCRIPT, 5 DRILLED,
   90 AMBIENT. Three classes, and they are never averaged: one is a
   performed reading, one is a performed take, one is first-take speech.
2. **The criterion is distance from Casey's own baseline, in baseline
   sigmas, weighted by each feature's between-session/within-session
   ratio.** And the weights turned out to say something important:
   **every ratio in the corpus is below 1** (0.35 to 0.99). No feature
   in this corpus separates this person's sessions more than it varies
   inside a session. **There is no identity-carrying feature here yet.**
3. **The first candidate set is 14 clips** (the top 10 %). The leader is
   `1792560758#009` — the word "state." — and it is the one clip that
   passes **both** exclusion tests: it scores 4.54 overall, **4.92
   without the word features** and **4.88 without the register
   features**. It is unusual in something that is neither the words nor
   the tone.
4. **The reference set needs no rater.** The DRILLED session is the only
   performed take in the corpus, so it is first-of-class by the
   data-class doctrine, and that is a fact about the capture, not a
   judgement about a take.
5. **The measured miss rate is 0.6** — three of the five reference
   clips are caught by neither the selection nor the audit. The audit
   reads the VIDEO, which the selection never reads, and its recall on
   the reference set is **0.0**.
6. **The budget curve is the deliverable Casey actually asked for.**
   How many quintessentials the GC may miss is a **policy** decision,
   not a measurement: at a 5 % budget the miss rate is 0.6, at 20 % it
   is 0.4, at 30 % it is 0.2. §7.
7. **And the audit's own control bites: three of its four flagged clips
   are session artefacts, not unusual moments.** That is the mouth
   detector's degeneracy, measured on the video layer.
8. **No session in this corpus can be defended as DEEP-THOUGHT.** The
   mode the addendum wants a different digestion policy for — the one
   where flattening destroys the layering — cannot be identified here.
   §8.

---

## 1. The corpus, and the three classes

| class | what it is | clips | sessions |
|---|---|---:|---:|
| **SCRIPT** | A and B, two independent readings of the same written passage | 45 | 2 |
| **DRILLED** | 1792560597, the only manually-triggered, deliberately performed take | **5** | 1 |
| **AMBIENT** | the free sessions, auto-caught, first-take | 90 | 4 |

A clip is **one speech segment** — a contiguous utterance — which is the
unit a trainer learns from and the unit the retention ladder would
graduate. The segmentation is the pause study's, byte-identical to the
repository's own `src/energy.py` method, so the two studies share one
spine and one clock.

**The classes are never averaged together.** The script pair is a
performed reading of a finished text; the drilled take is a person
performing into a machine; the ambient sessions are first talk. A
baseline computed over all three would be a baseline of nothing.

---

## 2. The criterion, and what the weights turned out to be

Each clip is scored by how far it sits from **Casey's own baseline**,
in units of that baseline's own spread:

- the **baseline** is the MEDIAN over his own clips and the spread is
  the **MAD**, scaled to be comparable with a sigma. One wild clip
  cannot move the centre and one quiet stretch cannot inflate the
  scale. This is the room map's lesson applied to a person.
- the **distance** is the weighted mean of the per-feature sigma
  distances.
- the **weight** is the feature's **between-session / within-session sd
  ratio**, the shape `experiments/teacher_data_requirement_results.json`
  already uses. A feature that is nearly constant inside a session and
  different between sessions is carrying the person's identity; one
  that moves as much inside a session as between sessions is carrying
  the moment.

### 2.1 The weights, and what they say

| feature | layer | between / within |
|---|---|---:|
| energy_mean | SCORE | **0.99** |
| mean_word_dur_s | TRANSCRIPT | 0.92 |
| f0_std_st | SCORE | 0.86 |
| f0_range_st | SCORE | 0.82 |
| motion_mean | VIDEO | 0.76 |
| speech_rate_wps | TRANSCRIPT | 0.68 |
| n_words | TRANSCRIPT | 0.61 |
| f0_mean_hz | SCORE | 0.57 |
| energy_range_db | SCORE | 0.45 |
| voiced_fraction | SCORE | 0.37 |
| f0_slope_st_s | SCORE | 0.35 |
| motion_max | VIDEO | 0.73 |

**Every ratio is below 1.** A ratio of 1 would mean the feature
distinguishes this person's sessions as much as it distinguishes his
moments; below 1, the moment wins. The best feature in the corpus —
the energy level — reaches 0.99, which is a tie.

**So the corpus contains no identity-carrying feature, and the study
cannot claim to have found one.** The reason is structural and worth
saying plainly: there is **one person**, on **one day**, with **one
room**. The between-session variation that a between/within ratio
needs does not exist yet. The machinery is built and the numbers are
right; the population is too small to populate it.

**What that means for the selection:** the weighted score is currently a
**moment** score, not an **identity** score. It finds unusual moments.
It does not yet find "quintessential of Casey" in the sense the brief
means, because nothing in the corpus yet distinguishes Casey from a
baseline of nobody.

---

## 3. The first candidate set

The candidate rule is the top **10 %** of the clips by weighted score —
**14 clips** of 140.

| rank | clip | class | score | beyond words | beyond tone | the words |
|---:|---|---|---:|---:|---:|---|
| 1 | 1792560758#009 | AMBIENT | **4.54** | **4.92** | **4.88** | "state." |
| 2 | 1792596369#003 | AMBIENT | 2.70 | 2.56 | 3.11 | "give them the" |
| 3 | 1792560758#008 | AMBIENT | 2.31 | 2.82 | 2.12 | "Now you get me in my temporal state." |
| 4 | 1792596354#003 | AMBIENT | 2.24 | 2.17 | 1.69 | "seconds" |
| 5 | 1792596369#001 | AMBIENT | 1.96 | 2.36 | 1.49 | "That's about all" |
| 6 | 1792560597#003 | **DRILLED** | 1.94 | 1.25 | 1.82 | "darn I'm" |
| 7 | 1792560597#000 | **DRILLED** | 1.90 | 1.91 | 1.67 | "There now" |
| 8 | 1792596369#000 | AMBIENT | 1.84 | 2.07 | 1.10 | "Well, that was it." |

**The leader is a one-word clip.** `1792560758#009` is the utterance
"state." — and its motion z is **16.11**, against a pile whose median
is 0.62. The person moved a great deal during it. It scores 4.54
overall, and it scores **4.92 when every word-derived feature is
removed** and **4.88 when the register features are removed**. So
neither the words nor the tone are carrying it.

Two of the three scripted clips are in the set, which is a **rate of
2 of 45** — the script class is under-represented in the candidates
against its 32 % share of the corpus. That is a fact about the corpus
and it is not evidence about the criteria.

---

## 4. Beyond words, and beyond tone, are two different tests

The brief's phrase is "meaning carried **beyond words** and **beyond
tone**". Those are two claims and they have to be tested separately, so
the score is recomputed twice with a leg removed:

| variant | the word features | the register features |
|---|---|---|
| **beyond words** | removed: n_words, speech_rate_wps, mean_word_dur_s | kept |
| **beyond tone** | kept | removed: f0_mean_hz, energy_mean |

A clip that survives both removals is unusual in something that is
neither its words nor its voice's register. On this corpus the two
removals barely move the scores, because **the word features carry
almost none of the variance** — their between/within ratios (0.61,
0.68, 0.92) are not far from the others, and the word-derived z
distances are small.

**That is the finding, and it cuts both ways.** It says the words are
not what distinguishes the unusual clips in this corpus — which
supports the brief's premise that the words are not the carrier. But
it also says the test is not yet sharp, because removing the words
changes almost nothing.

---

## 5. The reference set, and why it needs no rater

The brief's ground truth is human: "teacher/student disagreements,
referee-marked moments". The corpus has **no referee**, and the tone
game's own teacher-dependence ratio is **1.0**.

So the reference set is built from a fact about the **capture** rather
than a judgement about a **take**:

> **The DRILLED session is the only manually-triggered, deliberately
> performed take in the corpus.** It is first-of-class by the
> data-class doctrine — one member of a class, so there is no variance
> within it — and that is checkable from `meta.json` by anyone.

Its **5 clips** are the reference set. The `SCRIPT` pair is *not* in
it: a scripted reading is not a delivery of this person's, it is a
delivery of a text.

**The power statement is in §7 and it is not favourable.**

---

## 6. The audit: an independent instrument, with controls

The selection reads the AUDIO, the SCORE and the TRANSCRIPT. The audit
must not, because **a judge cannot certify itself** — the repository's
own law. The only other layer the corpus has is the **VIDEO**, so the
audit reads the motion of the person zone (the room map's region R0).

**The audit has a budget, not a threshold.** The first version used
three sigma over the pile's own motion and flagged **nothing** — and
the control agreed with it, because a threshold nobody can reach fires
on nobody. An audit that cannot fire is not an audit. The budget is
**5 % of the routine pile**: 4 clips of the 82 routine clips that carry
video.

| | the audit's flagged clips | the routine pile |
|---|---|---|
| motion z, median | **2.53** | 0.62 |
| motion z, max | 2.88 | 2.88 |
| the ratio | **3.52×** | — |

So the flagged clips ARE at the tail of the pile's motion. That is the
audit's first claim and it holds.

### 6.1 And the control that bites

**Three of the four flagged clips (75 %) are also outliers WITHIN their
own session.** They are unusual against the whole corpus and ordinary
inside the session they belong to.

That is `docs/research/MOUTH-DETECTOR.md` §5.1's degeneracy, measured
on the video layer: *a detector that separates sessions is not a
speaking detector*. The audit's hits are mostly "this clip is in a
busy session", and **that is not the same claim as "this clip is an
unusual moment"**.

**And the audit's recall on the reference set is 0.0.** The DRILLED
session's clips are not visually unusual at all — their motion z values
are 1.11, 0.94, 0.81, 0.07 and 0.62. The performance of a deliberately
performed take is not visible in the person zone's motion.

**So the video layer is blind to the thing this corpus calls
quintessential.** That is a measured finding about the instrument, and
it is the strongest argument in this document for keeping the audio.

---

## 7. The miss rate, and the budget curve

The guardrail wants a miss rate, and the miss rate depends on a choice
that is not a measurement: **how many clips may the GC keep?** That is
a budget, and the study's job is the curve.

| candidate budget | clips kept | reference clips caught | **miss rate** |
|---:|---:|---:|---:|
| 2 % | 3 | 0 / 5 | **1.00** |
| 5 % | 7 | 2 / 5 | **0.60** |
| 8 % | 12 | 2 / 5 | **0.60** |
| 10 % | 14 | 2 / 5 | **0.60** |
| 15 % | 21 | 2 / 5 | **0.60** |
| 20 % | 28 | 3 / 5 | **0.40** |
| 30 % | 42 | 4 / 5 | **0.20** |

(The audit adds nothing to this table: its recall on the reference set
is 0.0, so the union's miss rate is the selection's.)

**The headline number is 0.60 at the top-10 % budget** — three of the
five reference quintessentials are missed by both the selection and the
audit. The curve says what buying them costs: **a 20 % budget catches
three, a 30 % budget catches four.** A 30 % budget is 42 of 140 clips,
which is not "a small set" and is not what the brief describes as the
GC's product.

**The power statement.** Five reference events. The miss rate's
interval, at n = 5, runs from roughly 0.15 to 0.95 on any reasonable
reading. **This is a count, not a rate.** The brief's condition — "the
miss rate is measured and near zero" — is **not** met, and cannot be
met on this corpus, because the reference set has five members and the
selection needs 30 % of the corpus to find four of them.

**And the honest reason the curve is so flat is in §2.** Every feature
is a moment feature, so the ranking is dominated by how unusual the
moment was, and a deliberately performed take of two words does not
look unusual. The criteria would improve far more from **more
reference quintessentials** than from better features.

---

## 8. The mode, tagged only as far as the evidence reaches

The modes addendum makes MODE a first-class session field, read off the
body — posture, motion, gaze, phone handling — with the digestion
policy indexed by it.

**What this corpus can read:** posture as motion inside the room map's
person zone, over every session with a live video layer (**53
sessions**), and the pause structure of the speech.

**What this corpus cannot read at all:** gaze (no eye tracker) and
phone handling (no phone detector). No claim is made about either.

| session | mode | why |
|---|---|---|
| 1792596296 | **RAMBLE** | person-zone motion 0.6185, at or above the corpus's 75th percentile (0.5375) |
| 1792596354 | **RAMBLE** | 0.5655, at or above the 75th percentile |
| 1792596369 | **RAMBLE** | 1.2780, far above |
| 1792560597 | AMBIGUOUS | 0.3900, between the 25th (0.2119) and the 75th |
| 1792560758 | AMBIGUOUS | 0.4108, between them |
| A, B | NO-VIDEO | the script pair has no frame clock |

**There is no defensible DEEP-THOUGHT tag anywhere in this corpus.**
The rule for it requires the body to be *still* — at or below the 25th
percentile — **and** the speech to carry at least two pauses of 1.5 s
or more. No session satisfies both. 1792560773 is the stillest of the
speech sessions (0.1707) and it carries **no speech at all**: it closed
on `close_reason: "pause"` and it is the corpus's natural null.

**That is a blocker for mode-aware digestion, not a footnote.** The
policy the addendum wants most — do not brush-stroke, because the
thought accrues in layers — is indexed by a mode this corpus cannot
identify. The next capture has to be a still, long-thinking session, and
until it exists the mode index has one populated cell and one empty one.

---

## 9. The acoustic scene: what the corpus can and cannot separate

The awareness addendum asks what in last night's sessions already
separates **kitchen** from **desk** from **paper-in-hand**.

**The corpus has one space.** The room map is the evidence: 46 sessions
at one viewpoint and 4 at a nudged one, all in the same room, and the
furniture read off the composite (the desk, the shelf, the rail with
hanging bags, the lamp). There is no second space in the corpus to
separate from the first.

**There is one device.** Every browser session used the same Realtek
array at 48 kHz.

**There is no double-hearing.** No two-device capture of one voice
exists, so the delay-is-place instrument has nothing to measure.

**What the acoustic measurement does show**, per session:

| session | noise floor (RMS) | noise centroid | decay T20 | SNR |
|---|---:|---:|---:|---:|
| 1792560758 | 0.000229 | 2198 Hz | 0.23 s | 41.0 dB |
| 1792596296 | 0.000241 | 2281 Hz | 0.38 s | 41.7 dB |
| 1792596354 | 0.000208 | 3514 Hz | 0.145 s | 39.6 dB |
| 1792596369 | 0.000328 | 2984 Hz | 0.11 s | 40.6 dB |

**The spectral centroid of the room's own noise varies by a factor of
1.6 across sessions in the same room**, and the level by a factor of
6 (the same microphone produced peaks near 3 000 and near 18 000 within
one day — a lens entry in `docs/research/lens/microphone.json`).

**That is not a scene measurement, and the difference is the whole
point.** A scene classifier fitted on this corpus would learn the
**microphone's gain**, because that is what varies; the room does not
vary, because there is only one of it. **The acoustic-scene calculator
cannot be fitted from this corpus at all**, and the first thing to
build is a capture in two spaces with the same device — which is the
phone node's two-device time base, and it is the same blocker.

---

## 10. The calculator this graduates into

The awareness addendum's doctrine says the fleet's real product is
calculators. The selection rule this study produces is one:

> **A quintessential clip is one that sits far from this person's own
> baseline in the features that separate his sessions rather than his
> moments, and it stays there when the words are removed and when the
> register is removed.**

It is cheap: a median, a MAD, and a weighted mean over twelve
features. It needs no model at all. And its cost per correct judgment
falls as the corpus grows, because the baseline gets better.

**But on this corpus it is a moment detector wearing an identity
detector's clothes** (§2), and the audit that would certify it is
three-quarters session artefacts (§6.1). The calculator is built; it
is not yet fitted to the right thing.

---

## 11. What this does not establish

- **It does not establish a miss rate near zero.** The measured miss
  rate is **0.60** at the top-10 % budget, on five reference events, and
  the brief's condition for starting live collection is not met.
- **It does not establish that the criteria find "quintessential of
  Casey".** Every between/within ratio in the corpus is below 1. The
  corpus contains no identity-carrying feature, and the criteria find
  unusual moments.
- **It does not establish that the audit works.** Its recall on the
  reference set is 0.0, and three of its four hits are session
  artefacts.
- **It does not establish a mode tag for DEEP-THOUGHT.** No session in
  the corpus can be defended as one.
- **It does not establish anything about acoustic scenes.** There is one
  space and one device; what varies is the gain.
- **It does not touch the GC.** ZCode owns the ladder plumbing and the
  dry-run GC report. This is the criteria and the audit method the
  report should implement, and the budget curve that lets Casey choose
  the budget knowingly.

Committed for Muse to review; not merged.
