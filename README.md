# communicator

**A person's frontend computer.** Voice character rendered as a nine-track
MIDI score: five tracks **measured** from the waveform, four **judged** by a
decision model that plays the synthesiser's control surface instead of
routing a decision.

    A calibrated probability IS a continuous controller value,
    and that is exactly what a MIDI CC is.

"STOP THE DEPLOY RIGHT NOW" → INTENTION CC74 = 121, gate 0.95.
A fan humming near the mic → PITCH gap (energy floor), no note.
Warmth in the *words* with no voice yet → TIMBRE faded toward 0 by its
presence gate, because warmth is a delivery property and the file says so
instead of inventing it.

## The nine tracks

| # | track | source | in the file |
|---|---|---|---|
| 1 | pitch | measured (YIN, 0.03 Hz worst-case on synthetic tones) | notes, velocity from co-timed RMS |
| 2 | tempo | measured (speech rate → BPM) | set_tempo + CC120 |
| 3 | velocity | measured (RMS per frame) | CC7 |
| 4 | timbre | judged (Score, warm↔cold) | CC71, gated |
| 5 | inflection | measured (pitch contour direction) | CC11 per event |
| 6 | silence | measured (hysteresis segmentation) | velocity-0 notes (rests) |
| 7 | gesture | judged (Choice: speech act) | one CC in the 80–87 bank, on the winning class's slot |
| 8 | intention | judged (Score, idle→urgent) | CC74, gated |
| 9 | attention | judged (Choice: addressee) | CC91, gated |

Layout follows slackwater-perception's MultiTrackEncoder, with one stated
reconciliation: `gesture` is a **speech act** here (question/command/...),
not slackwater's physical cue.

## What has been measured (see PROGRESS.md for all of it)

- **Question health**: garbage in must not drive the controller. Three of
  four judged questions failed the first health probe; the fix that worked
  was not re-wording (measured: made it worse) but **presence gates** — a
  Noul "is this property here?" per track, in the same call, multiplied into
  the CC. ATTENTION garbage separation −8 → +53.
- **Calibration** (ground truth by construction, `CALIBRATION.md`):
  GESTURE accuracy 1.000 / ECE 0.001; ATTENTION 0.917 / 0.097; INTENTION
  Spearman 0.978 with a +0.6-level mid-scale urgency bias. TIMBRE carries no
  calibration claim — wording separates (1.4 vs 0.02) but delivery warmth is
  unmeasurable from text.
- **End-to-end**: `examples/demo-utterance.mid` — WAV + text in, gated
  nine-track file out, 553 ms, one model call (four questions + four gates).

## Repo map

```
src/wavio.py         WAV in → mono floats at 16 kHz, non-finite guard
src/pitch.py         YIN tracker; hz=None when unvoiced; energy floor
src/energy.py        RMS + speech/silence segmentation (hysteresis)
src/tempo.py         speech rate → BPM
src/inflection.py    contour direction events
src/smf.py           Standard MIDI File writer AND parser, stdlib only
src/judged.py        the four judged tracks; one call; CC = probability,
                     never `confidence`; presence gates; model named
src/verdict_cache.py same utterance, same questions, same model → same CC
                     (Jev is not bit-deterministic; a jittering controller
                     is an instrument nobody can play)
src/question_health.py  ill-posed question detector (garbage vs real
                     batteries); safe-parked verdict for Score tracks
src/calibration.py   label-by-construction corpus + ECE/Brier/Spearman
src/score.py         merge + export; tick = 1 ms; provenance sidecar
docs/superpowers/specs/   the design doc (v2) — read §0 before extending
```

## Running

```bash
# tests: 215 passed offline (live tests skip without a key)
python3 -m pytest tests/ -q

# end-to-end against live Jev (needs TYPESAFE_API_KEY):
python3 -c "from src import score; score.render('examples/demo-utterance.wav', \
    'STOP THE DEPLOY RIGHT NOW', 'examples/demo-utterance')"
```

`typesafe_sdk` is provided by the shared venv on this machine; a key is
needed only for live calls — every suite is useful without one.

## House rules (violated once each, then written down)

1. **One call per utterance.** Four questions + four gates ride together;
   latency is the product.
2. **CC is a probability, never `confidence`.** Confidence measures
   distribution spread; using it would under-drive exactly the ambiguous
   cases.
3. **A number with no model attached is not a measurement.** Models are
   named by explicit build; question wording is fingerprinted, so a reworded
   question invalidates every cached verdict by construction.
4. **Absent ≠ zero.** `hz=None`, ok=False tracks export no event, garbage
   parks the controller — never a confident 0 for an unmeasured thing.
5. **Honest negatives are results.** Re-wording made the questions worse;
   Platt scaling hurts calibrated models; TIMBRE-from-text is dead until the
   acoustic pipeline exists. All kept, with numbers.
