# F0 cannot identify a speaker change

**Status:** hypothesis falsified. Written after measuring, not before.

`docs/research/TURN-ROUTES.md` ends at a wall. Every file in the corpus is one
person reading alone, so after any silence the same speaker resumes 98.1–100%
of the time (`17ae46a`). The positive class — *a speaker yields the floor and
someone else takes it* — does not occur in the data.

The obvious rescue is the cheapest second signal in an audio stream: pitch.

> **H.** A speaker change produces a large F0 jump, so longer pauses, where a
> hand-off is more likely, should jump more than short same-speaker ones.

## The measurement, and the two controls it needed

Two failure modes had to be closed before any number meant anything.

**Ratio alone is not a voice change.** `|f2−f1|/f1` is dimensionless, and a
tracker slipping one octave moves 110 Hz → 220 Hz — 100% — from the same
throat. So every observation carries both frequencies, and jumps are classified.
`jump` is a *relative magnitude*: an octave is 1.00, a subharmonic 0.50. An
earlier version of this experiment used bands `[1.8, 2.2]` and `[0.45, 0.55]`,
which would have filed every real octave error as ordinary movement. That was
caught before the run, not after.

**A hypothesis with no null cannot fail.** The decisive control is
**within-utterance**: the last second of a stretch of continuous speech against
its own first second. The speaker is identical by construction.

That control could not be built the obvious way. Measured on this corpus, the
median watcher utterance is **0.270 s** and the longest is **1.220 s**, so not
one utterance in six files holds two non-overlapping 1-second windows. The first
version returned **zero** control samples and printed `control_median: null`,
which is indistinguishable at a glance from "the control found no difference".
It is now built from the silero gate, which detects speech without chopping on
breaths, and it returns 50 comparisons.

## Result

Median `|f2−f1|/f1`, by pause band, against the same-speaker control:

| cell | n pauses | short <0.30s | mid 0.30–0.60s | long >0.60s | **control** | Spearman ρ | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| children | 52 | 0.102 | 0.081 | **0.235** | 0.179 | +0.341 | **0.006** |
| essay | 45 | 0.089 | 0.063 | 0.169 | 0.066 | +0.105 | 0.245 |
| narration | 72 | 0.151 | 0.116 | 0.174 | **0.204** | +0.021 | 0.431 |
| dialogue | 62 | 0.113 | 0.167 | 0.138 | 0.098 | −0.018 | 0.561 |
| oration | 56 | 0.163 | 0.063 | 0.136 | 0.092 | +0.080 | 0.276 |
| shakespeare | 66 | 0.184 | 0.144 | **0.086** | **0.173** | −0.179 | 0.925 |

**Zero of six cells** show the monotone increase the hypothesis predicts.
**Two of six** have a long-pause band that does not exceed pure intonation
inside a single speaker's own speech; in **narration and shakespeare the long
band is below the control** — the "hand-off" signal is smaller than phrasing.

> **CORRECTION (2026-10-19).** This said *"**Four of six** have a long-pause
> band that does not exceed pure intonation inside a single speaker's own
> speech"* — which inverted the count and contradicted its own second clause
> (it named two cells). `experiments/f0_speaker_change_results.json` holds
> `cells_long_exceed_control: 4`, `cells_long_under_control: 2`, and the table
> above agrees with it: **narration** (0.174 long vs 0.204 control) and
> **shakespeare** (0.086 vs 0.173) are the two under the control; the other
> four exceed it. Settled by that result file; `NOTES-AUDIT.md` F2.

Only `children` is significant, at ρ = +0.341, p = 0.0061. Across six cells
that is a Bonferroni-corrected p of 0.037: it survives correction, and it is
still one cell in six on a corpus with no hand-offs in it at all. It is the
weakest available evidence for the hypothesis, not the strongest.

Shakespeare runs the other way: ρ = −0.179, one-sided p = 0.92, and its large
jumps sit in the *short* and *mid* bands, not the long ones. It contains the
single octave artifact in the whole run (p90 = 1.0454 — a near-doubling) and
seven of the thirteen subharmonic artifacts, so its biggest jumps are tracker
artifacts rather than voices.

## Consequence

**A large F0 shift is intonation inside a phrase, not a speaker change.** F0
cannot rescue this corpus as a speaker-change proxy, and no threshold on pitch
jump size can be given a meaning.

This is not only this experiment's conclusion. The AMI two-speaker corpus
(`examples/realtwospeaker/`) reaches the same wall from the other direction:
across four speakers, median F0 separates two *voice classes* cleanly — men at
104–111 Hz against women at 197–208 Hz — but cannot resolve the two speakers
*within* each class. FEO070 and FEO072 sit **0.5σ** apart; MEE071 and MEE073
**0.3σ**. Pitch proves there are two voices, not who is speaking. A hand-off
between two men produces no reliable jump at all.

Two independent measurements, made with different instruments on different
audio, agree. That is the strongest form this claim can take here.

## What would change this

Not a better pitch estimator. The information is not in the pitch contour: for
a same-gender hand-off it is absent by construction. It would take a speaker
embedding, and the far-field single microphone would still have to resolve two
simultaneous voices, which 203 audible segments across 19,305 voiced frames
say is marginal at best.

## Reproduce

```bash
/home/proart/projects/glasses/.venv/bin/python experiments/f0_speaker_change.py
```

Per-pause observations are cached in `experiments/f0_observations.json`, so
`--reuse` re-derives every statistic above without re-running YIN.