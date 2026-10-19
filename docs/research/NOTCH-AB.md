# Does a mains notch fix the emitted F0? The §8 experiment, run

Lane: audio DSP. Executes the experiment proposed at
`docs/research/IN-THE-WILD-AUDIO.md` §8 and never run. Code
`experiments/notch_ab.py`, results `experiments/notch_ab_results.json`.

**The answer is no, and the reason it is no is more useful than a yes would
have been.** The notch fixes exactly the frequencies whose *fundamental*
carries the false lock (50–120 Hz) and does nothing at all at 150/200 Hz,
because YIN locks onto the **period**, and deleting the fundamental of a
harmonic stack leaves the remaining harmonics with an unchanged period. The
band this project has spent three documents arguing about is the band the
notch cannot touch.

**Nothing ships.** No arm met the decision rule, which was written before any
arm was run (§1). The run is valid: all three void conditions held clear (§2).

## 0. The bug, reproduced before anything was changed

Arm `none`, unmodified pipeline, same fixtures as
`experiments/inthewild_probe.py`:

| signal | emitted `hz_p05` |
|---|---|
| clean speech | 194.49, 189.28, 163.61, 179.04 Hz |
| 150 Hz testvad stack, gate PASSES | 150.0, 150.0, 150.0 Hz |
| 200 Hz testvad stack, gate PASSES | 200.01, 200.01, 200.01 Hz |
| speech + 150 Hz @ 10 dB | 74.97, 79.86, 74.99 Hz |

Every figure matches `IN-THE-WILD-AUDIO.md` §4.3. The harness is the harness
that found the bug.

## 1. The decision rule, as written before the run

Reproduced verbatim from the script docstring and the `pre_registered_decision_rule`
block of the results JSON. `CLEAN_MED` = median emitted `hz_p05` of unfiltered
clean 6 s speech = **184.16 Hz**. For arm A:

- **R(A)** = fraction of mixture `hz_p05` values within **100 cents** of `CLEAN_MED`
- **SHIFT(A)** = median |cents| shift of clean-speech `hz_p05`, A vs none
- **MAXSHIFT(A)** = max of the same (this is the no-information control)
- **dUTT / dVOX** = clean utterances with a pitch, and median `voiced_ratio`, A − none
- **dCELLS(A)** = mixture cells emitting ≥1 pitch, A − none

> **SHIP iff** S1 `R ≥ 0.90`; S2 `SHIFT < 10.0` cents on **both** clean clips;
> S3 `MAXSHIFT < 50.0` cents on both; S4 `dUTT == 0` and `|dVOX| < 0.02`;
> S5 `dCELLS ≥ 0`.
>
> **VOID the run if** V1 `R(hpf60) ≥ 0.90` (the deliberately-losing control
> passed); V2 the no-op arm is not bit-identical to `none`; V3 any arm emits
> zero clean-speech utterances.

Clean speech is measured on **two** clips: the 6 s `inthewild_probe` clip
(n = 4 utterances) and the full 90 s `alice01_excerpt.wav` (n = 52). S2–S4 must
hold on both.

## 2. Validity: all three controls behaved

| control | required | measured |
|---|---|---|
| V1 `hpf60` must **fail** | R < 0.90 | **R = 0.370** — it lost, and lost *to the baseline*: no filter at all scored R = 0.410. A 4th-order 60 Hz high-pass makes the emitted pitch worse than leaving the pipeline alone. |
| V2 no-op == `none` | bit-identical | **true**, on both clean clips, all 52 + 4 utterances |
| V3 no arm silent | > 0 utterances | **held** (minimum 4) |

The no-information control proper — the notch applied to a signal containing no
hum — is S3, and it is reported in §4.

Filter attenuation, measured by pushing unit sines through the same
`mic_filter_sim` code (dB):

| arm | 50 | 60 | 100 | 120 | 150 | 200 |
|---|---|---|---|---|---|---|
| `notch_mains` (50/60/100/120) | −15.50 | −16.44 | −18.63 | −19.45 | **−0.30** | **−0.07** |
| `notch_mains_plus_speech` (+150/200) | −15.51 | −16.46 | −18.75 | −19.88 | −20.58 | −21.44 |
| `hpf60` (control) | −9.79 | −6.06 | −1.08 | −0.54 | −0.23 | **−0.08** |

`hpf60` reproduces `MIC-HARDWARE.md`'s published −9.79 dB @ 50 Hz and −0.08 dB
@ 200 Hz exactly. 175 Hz is in no arm's notch bank and therefore acts as a
built-in negative control: the residual hum probe inside the actual mixture
reads **0.00 dB** for `none`, **−0.12 dB** for `notch_mains`, **−0.12 dB** for
`hpf60` and **−1.12 dB** for the 6-notch — the last being the small leakage
through the 200 Hz section that is an octave below it in Q terms. None of the
arms does anything to 175 Hz, and none of them changes behaviour there
(R = 0.70 / 0.70 / 0.72 / 0.70).

## 3. Primary metric: recovery on mixtures

`R(A)`, pooled over all 34 mixture cells (12-harmonic family ×
{50,60,100,120,150,175,200} Hz × {20,10,0} dB, plus the 3-harmonic testvad
family × {150,175,200} Hz × {20,10,0} dB):

| arm | R | values within 100 c | p10 / median / p90 cents |
|---|---|---|---|
| `none` | 0.410 | 41 / 100 | −1564.0 / −175.5 / +89.8 |
| `notch_mains` | **0.500** | 50 / 100 | −1555.4 / −91.3 / +89.3 |
| `notch_mains_plus_speech` | 0.449 | 44 / **98** | −1541.4 / −113.1 / +78.9 |
| `hpf60` | 0.370 | 37 / 100 | −1555.7 / −527.2 / +90.7 |

Broken out by hum frequency — this is the whole result:

| hum f0 | `none` | `notch_mains` | `notch_mains_plus_speech` | `hpf60` |
|---|---|---|---|---|
| 50 Hz | 0.70 | **0.80** | 0.70 | 0.20 |
| 60 Hz | 0.30 | **0.70** | 0.70 | 0.40 |
| 100 Hz | 0.00 | **0.30** | 0.30 | 0.00 |
| 120 Hz | 0.20 | **0.30** | 0.30 | 0.20 |
| **150 Hz** | **0.20** | **0.20** | **0.20** | 0.20 |
| 175 Hz | 0.70 | 0.70 | 0.72 | 0.70 |
| **200 Hz** | 0.55 | 0.55 | **0.35** | 0.55 |
| **mean, 50/60/100/120** | **0.300** | **0.525** | 0.500 | 0.200 |

Two facts, and the second one is the headline.

**First: the notch does work, on the frequencies it notches.** Mean recovery
over 50/60/100/120 Hz rises **0.300 → 0.525**. The 60 Hz case — the actual
mains case per `IN-THE-WILD-AUDIO.md` §4.2 — goes **0.30 → 0.70**. That is a
real effect, measured, and it is worth having.

**Second: at 150 Hz and 200 Hz it does nothing, and at 200 Hz it makes things
worse.** Recovery at 150 Hz is 0.20 for *every* arm, including the arm with a
−20.58 dB notch at 150 Hz. And the 6-notch arm *drops* 200 Hz recovery from
0.55 to 0.35.

### Why: the lock is to the period, not to the fundamental

A Goertzel probe at the hum frequency inside the actual mixture shows the
6-notch removing **49.42–49.50 dB** of the 150 Hz line and **50.32–51.73 dB** of
the 200 Hz line. The emitted pitch does not move:

| cell | `none` | 6-notch | hum removed |
|---|---|---|---|
| speech + 150 Hz @ 10 dB | 74.97, 79.86, 74.99 | 74.84, **75.02**, 74.96 | −49.49 dB |
| speech + 150 Hz @ 0 dB | 72.70, 72.89, 148.81 | 74.96, 75.13, 149.77 | −49.50 dB |
| speech + 200 Hz @ 10 dB (testvad) | 193.87, 186.73, 185.57 | **100.04, 100.54**, 188.87 | −51.38 dB |
| 150 Hz hum alone | 150.0, 150.0, 150.0 | **150.0** | −20.99 dB design |
| 200 Hz hum alone | 200.01, 200.01, 200.01 | **200.01** | −21.44 dB design |

A 12-harmonic stack at 150 Hz has lines at 150, 300, 450, 600 … Delete the
150 Hz line and the surviving 300/450/600 lines still repeat every 1/150 s.
**The periodicity is unchanged, so YIN's answer is unchanged.** The testvad
3-harmonic fixture shows it most plainly: remove the 150 Hz line and the
300/450 Hz remainder still reports exactly 150.0 Hz. And where the 6-notch
does change a reading it moves it *down* an octave to the subharmonic
(200 → 100.02), which is the same failure mode arriving by a different route,
made more deterministic because the strongest line was deleted first.

This is the prediction at `IN-THE-WILD-AUDIO.md` §8 confirmed by measurement:

> "If it does not, the octave error is a YIN-configuration problem rather than
> a spectral one, and §5.1's periodicity-consistency test becomes the next
> thing to try."

## 4. Regression metric: what it costs on clean real speech

The no-information control required by this experiment — the notch applied to a
signal with **no hum in it at all**. Paired per utterance against `none`.

**Full 90 s clip, 52 clean utterances (81.73–280.75 Hz):**

| arm | median |p90 | max | >10 c | >50 c | >100 c | >1 octave | dVOX |
|---|---|---|---|---|---|---|---|---|
| `notch_mains` | 0.09 | 0.72 | **50.73** | 2 | 1 | 0 | 0 | 0.0 |
| `notch_mains_plus_speech` | 2.50 | 80.88 | **1333.49** | 16 | 8 | 6 | **1** | −0.0165 |
| `hpf60` | 0.19 | 4.22 | 428.62 | 4 | 1 | 1 | 0 | −0.0074 |

**6 s clip, 4 clean utterances:**

| arm | median | max | dUTT | dVOX |
|---|---|---|---|---|
| `notch_mains` | 0.10 | 0.18 | 0 | 0.0 |
| `notch_mains_plus_speech` | 13.31 | 37.27 | 0 | +0.0006 |
| `hpf60` | 0.27 | 0.77 | 0 | 0.0 |

**The 150/200 Hz notches destroy real speech**, exactly as
`IN-THE-WILD-AUDIO.md` §7.5 and `MIC-HARDWARE.md` warned. Utterance 15 of the
90 s clip is reported at **94.44 Hz unfiltered and 204.02 Hz notched — 1333.5
cents, more than an octave.** Utterance 22 moves +920.0 cents, utterance 18
−854.0 cents, utterance 20 −805.4 cents. 16 of 52 utterances move by more than
10 cents. `MIC-HARDWARE.md` measured this filter's median cents error at 0.24
on the *median* and called the cost small; the emitted `hz_p05` metric shows
the median cost is indeed 2.50 cents, but it hides that individual utterances
are destroyed outright. **That is the case for judging a filter on its
distribution and not its median.**

## 5. Verdict against the pre-registered rule

| arm | R | S1 | S2 | S3 | S4 | S5 | **SHIPS** |
|---|---|---|---|---|---|---|---|
| `none` | 0.410 | ✗ | ✓ | ✓ | ✓ | ✓ | **no** |
| `notch_mains` | 0.500 | ✗ | ✓ | **✗** (50.73 ≥ 50.0) | ✓ | ✓ | **no** |
| `notch_mains_plus_speech` | 0.449 | ✗ | **✗** | **✗** (1333.49) | ✓ | ✓ | **no** |
| `hpf60` | 0.370 | ✗ | ✓ | **✗** (428.62) | ✓ | ✓ | **no** |

`notch_mains` misses S3 by **0.73 cents** on one utterance out of 52
(index 44: 175.87 → 181.10 Hz) and passes everything else. I am reporting that
as a **fail** because the threshold was fixed in advance and I am not moving it
after seeing the number. It is the only arm whose cost profile is genuinely
small, and if someone wants to argue for it, that is a decision to re-open the
rule, not one to reinterpret.

## 6. Recommendation

**Do not ship a notch.** Specifically:

1. **The 150/200 Hz notches must not ship.** They buy no recovery at 150 Hz
   (0.20 → 0.20 with 49.5 dB removed), actively lose recovery at 200 Hz
   (0.55 → 0.35), and move one clean utterance by 1333 cents. This retires
   the last row of the `MIC-HARDWARE.md` intervention table that the other
   rows do not already retire.
2. **The 50/60/100/120 Hz notch is the one thing that measurably works** —
   recovery 0.300 → 0.525 over its own four frequencies, 60 Hz 0.30 → 0.70, at
   a median cost of 0.09 cents and 0 lost utterances. It fails the
   pre-registered S3 by 0.73 cents on a single utterance. That is a real
   result and it is a near-miss, not a pass. If the project wants it, the next
   move is to re-open the 50-cent criterion with this measurement in hand —
   not to ship it under this one.
3. **The correctness fix is not spectral.** §8's own fallback applies: the
   octave error survives a 49.5 dB removal of the interfering fundamental, so
   it is a YIN-configuration problem. `IN-THE-WILD-AUDIO.md` §5.1's
   periodicity-consistency test on the F0 track — real voice F0 moves
   continuously, a mains-locked estimate does not — is now the primary
   candidate, not a fallback.

## 7. Limits

Synthetic harmonic stacks; no audio hardware on this box. One 6 s clip and one
90 s clip, one narrator (female, `alice01_excerpt.wav`, median F0 184 Hz), one
machine, one gate instance per signal. The 90 s clean clip's median F0 is well
above 150 Hz, so it is the **friendliest** possible case for a 150 Hz notch —
`MIC-HARDWARE.md` §"the tension, stated precisely" warned that the small
numbers there are a property of this narrator, and §4 shows what happens when
the notches land on a 94 Hz utterance. A male speaker would be worse, not
better. SNR is speech-RMS to hum-RMS. The filter is applied to the sample array
ahead of the gate rather than between the gate and `PitchTracker` as §8
specified, because `src/` was not editable; the tracker sees the same audio
either way, and the gate/segmenter side effects are reported per cell
(`utterances`, `voiced_ratio`, `gate_speech_prob`) rather than hidden.