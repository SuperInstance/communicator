# Phase 1 — measured acoustic tracks

*2026-10-17. Status: measured half of the score is working. Judged half does not exist yet.*

## What is here

Six modules in `src/`, standard library only — `numpy`, `scipy`, `torch`,
`soundfile` and `pyaudio` are all absent from every interpreter on this machine,
so that is a constraint, not a preference.

| Module | Role |
|---|---|
| `wavio.py` | WAV in → mono floats at `ANALYSIS_SR` (16 kHz); 8/16/32-bit, stereo averaged, non-finite guard |
| `pitch.py` | YIN tracker, 50 ms frames / 10 ms hop, `hz=None` when unvoiced |
| `energy.py` | RMS series + speech/silence segmentation |
| `tempo.py` | speech rate → BPM |
| `inflection.py` | pitch contour → direction events |
| `smf.py` | hand-rolled Standard MIDI File writer **and parser** |

`hz is None` rather than `0.0` for an unvoiced frame, deliberately: "no
measurement" must not be indistinguishable from "measured zero", and PITCH is a
track downstream code will trust.

## Measured results

Pitch tracker, synthetic pure tones at 16 kHz, worst-case error over all frames:

| Target | Median error | Worst error | Voiced frames |
|---|---|---|---|
| 120 Hz | 0.002 Hz | 0.003 Hz | 36 |
| 200 Hz | 0.014 Hz | 0.014 Hz | 36 |
| 300 Hz | 0.034 Hz | 0.034 Hz | 36 |

For comparison, `forge-soniqo`'s pitch test accepted a 440 Hz tone anywhere in
200–800 Hz — it would have accepted an error of −55% or +82% and called it a
pass. The test tolerance here is 1 Hz, deliberately looser than the measured
0.034 Hz so it does not pin a float to four significant figures and break on
the next platform without saying anything true.

`python3 -m pytest tests/ -q` → **48 passed**.

## What was wrong on the way here

The first version of the test suite had five failures and **three of them were
the tests' fault, not the code's.** Recorded because that is the part worth
remembering:

1. Asserted `read_wav` preserves 22050 Hz. It does not — and it never did;
   `wavio.py:67` *documents* the resample to `ANALYSIS_SR`. Reading the docstring
   would have saved the detour. forge-soniqo's bug is **not** repeated.
2. Built the fixture at the default 16 kHz then wrote a header claiming 22050,
   so the file genuinely held 0.181 s and the resampler was right to disagree
   with my lie.
3. Guessed `MidiEvent(delta=, status=, data1=)` and `write_smf(ppq=)`. The real
   API is `MidiEvent(tick, kind, data)` and `ticks_per_beat`.

Worth noting because the alternative — writing tests against a guessed API and
then "fixing" the source to match — is how a suite ends up testing nothing.

## Two real findings

**1. YIN confidence is amplitude-invariant, so it cannot gate voicing.**

The first version of the quiet-signal test asserted a near-silent tone should not
be confidently voiced. It failed at confidence 1.0. The cause is that YIN's
difference function is normalised by the cumulative mean squared difference,
which makes `d'(tau)` amplitude-invariant *by construction*. Scale a clean
periodic tone by 0.0005 and `d'(tau)` is still ~0 at the true lag.

That is correct behaviour for what YIN measures — periodicity — and it is
exactly why its confidence is not a "is this speech" signal.

**The gap, now recorded in the design doc:** PITCH has no loudness gate.
Periodic non-speech — a fan, a keyboard, a synthesiser leaking from speakers —
is emitted as confidently-voiced pitch at any amplitude. The fix is an energy
floor applied before voicing is claimed. It does not exist yet.

**2. Silence is genuinely rejected.** A tracker that invents pitch in a silent
file is worse than one that emits a gap. Digital silence returns zero voiced
frames, asserted directly.

## What this does not prove

- **Nothing about real speech.** Every fixture is synthetic. Real voices have
  jitter, creak, breath noise and coarticulation. Whether this tracker tracks a
  *person* is unmeasured.
- **Nothing about octave errors.** A synthetic 200 Hz tone has one unambiguous
  period, so the classic halving/doubling failure is never exercised. Speech is
  full of it, and this is the most likely real-world failure.
- **Nothing about TEMPO or INFLECTION accuracy** — asserted for shape and sign
  only, because there is no ground truth in these fixtures.
- **Nothing about the judged tracks.** TIMBRE, INTENTION, ATTENTION and GESTURE
  are Phase 2 and do not exist.
- **Nothing about latency.** The whole architecture rests on the fast plane
  staying microseconds; that is measured in Phase 2 against a real microphone
  path, not here.

A permanent guard test, `test_this_file_does_not_measure_speech`, asserts the
synthetic-only claim so the limitation cannot be quietly forgotten if someone
later swaps in real audio.

## Next

Phase 2: Jev fills the four judged tracks. That is the novel core — a calibrated
probability is a continuous controller value, which is exactly what a MIDI CC
is. `decision-ledger`'s `QUESTION-REWRITE.md` (pass 0.30 → sharp 0.97) is the
instrument for proving the triage question is well-posed before trusting any
threshold on it.