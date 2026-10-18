# aubio — C/Python audio analysis

Investigated 2026-10-18. Source: shallow clone of `aubio/aubio` @ `ad5cf97` (2026-04-10),
read from source, not from the docs.

## Shape of the thing

| fact | value | source |
|---|---|---|
| license | **GPL-3.0** | `COPYING`, GitHub `licenseInfo.spdxId = GPL-3.0` |
| stars / forks | 3,773 / 430 | API |
| created / last push | 2009-12-04 / 2026-04-10 | API |
| default branch | `master` | API |
| languages | 139 `.c`, 65 `.h`, 86 `.py` | `git ls-files` |
| docs | 24 `.rst` | `git ls-files` |
| CI | `.travis.yml`, `.circleci/config.yml`, `azure-pipelines.yml`, `.appveyor.yml` | `git ls-files` |
| tests | 52 C test files (`tests/src/test-*.c`), 31 Python (`python/tests/test_*.py`) | `git ls-files` |

**Not a toy, and not abandoned.** Sixteen years old, 3.7k stars, commits in April 2026.

**CI actually runs.** `.circleci/config.yml:17-39` builds a Python wheel, installs it, then
`make create_test_sounds` and runs the Python suite against the installed wheel. I grepped all
four CI configs for `|| true`, `allow_failure`, and `continue-on-error`: **zero hits**. This is
real gating CI, not decorative.

## What it implements

### Pitch — 7 methods, all in `src/pitch/`

`pitchyin.c`, `pitchyinfast.c`, `pitchyinfft.c`, `pitchfcomb.c`, `pitchmcomb.c`,
`pitchschmitt.c`, `pitchspecacf.c`, behind a `pitch.c` (530 lines) dispatcher.

- **yin** — direct transcription of YIN (de Cheveigné & Kawahara 2002)
- **yinfast** — time-domain, trades resolution for speed
- **yinfft** — FFT/Wiener form of YIN, plus a psychoacoustic weighting curve
- **fcomb / mcomb** — McLeod Pitch Method (MPM) autocorrelation + filtered comb
- **schmitt** — fixed-threshold Schmitt trigger on zero crossings
- **specacf** — spectral autocorrelation

**There is no pYIN.** That is the single most important negative finding here. aubio has no
probabilistic-HMM pitch tracker, i.e. none of Mauch & Dixon 2014. The octave-error machinery
that pYIN solves is absent from aubio's pitch module entirely.

### Onset — `src/onset/onset.c` (355 lines)

Four modes, dispatch at `onset.c:297-320`: `energy`, `hfc` (default), `complexdomain`/`complex`,
`specflux`. Pipeline per frame is `onset.c:64-71`:

```c
if (o->apply_awhitening) { aubio_spectral_whitening_do(...); }      // :64-66
if (o->apply_compression) { cvec_logmag(o->fftgrain, o->lambda_compression); }  // :67-69
aubio_specdesc_do (o->od, o->fftgrain, o->desc);                   // :70
aubio_peakpicker_do(o->pp, o->desc, onset);                         // :71
```

18 spectral descriptors in `src/spectral/specdesc.c`: `complex`, `complexdomain`, `energy`,
`hfc`, `kl`, `mkl`, `specflux`, `specdiff`, `centroid`, `spread`, `skewness`, `kurtosis`,
`slope`, `decrease`, `phase`, `wphase`, `rolloff`.

### Tempo / beat — `src/tempo/`

`tempo.c` (304), `beattracking.c` (449). `beattracking.c:274-288` picks 3- vs 4-beat periodicity
by comparing autocorrelation energy at multiples of the estimated beat period.

### Spectral

`src/spectral/`: `mfcc.c` (179), `dct.c`, `fft.c` (+ `ooura_fft8g.c`), `filterbank.c`,
`filterbank_mel.c`, `phasevoc.c`, `awhitening.c`, `tss.c`, `statistics.c`.

**No CQT.** `git ls-files | grep -ci cqt` → **0**. No constant-Q transform anywhere in the tree.
If we ever need log-frequency resolution, aubio simply does not have it.

### Also present, easy to miss

- `src/effects/` — `pitchshift_rubberband.c`, `timestretch_rubberband.c` (plus `_dummy` stubs)
- `src/synth/` — `sampler.c`, `wavetable.c`
- `src/notes/notes.c` — MIDI note parsing
- `src/io/` — a dozen sinks/sources (wavwrite, sndfile, vorbis, flac, avcodec, Apple Audio Unit)

## The head-to-head with our YIN

This is the part that matters, so I read `src/pitch/pitchyin.c:137-171` against our
`src/pitch.py:155-187` line by line.

**We independently reimplemented plain YIN, correctly.** Specifically, the thing that is easy to
get wrong and that most naive implementations get wrong — summing the difference function over a
*fixed* window so every lag sees the same number of terms:

```c
// aubio/src/pitch/pitchyin.c:149-154
for (tau = 1; tau < length; tau++) {
  yin_data[tau] = 0.;
  for (j = 0; j < length; j++) {              // fixed `length`, not shrinking
    tmp = input_data[j] - input_data[j + tau];
    yin_data[tau] += SQR (tmp);
```

```python
# our src/pitch.py:105, 157-169
self.win = frame_len - self.tau_max          # fixed comparison window
head = frame[:w]
for tau in range(1, self.tau_max + 1):
    seg = frame[tau:tau + w]
    c = math.fsum(map(mul, head, seg))
```

Identical construction. Our docstring calls this "the classic normalization bug that biases
toward slow periods" — aubio gets it right too.

**Our threshold is aubio's threshold.** `aubio_pitchyin_do` uses `o->tol`, and
`new_aubio_pitchyin` sets `o->tol = 0.15` at `src/pitch/pitchyin.c:71`. Our
`PitchTracker.__init__` defaults `threshold=0.15` at `src/pitch.py:85`. Same number, arrived at
independently — 0.15 is simply the conventional YIN setting.

**Shared behaviour, confirmed:**

| behaviour | aubio | ours |
|---|---|---|
| fixed-window difference function | `pitchyin.c:149-154` | `src/pitch.py:157-169` |
| cumulative mean normalization | `pitchyin.c:155-160` | `src/pitch.py:171-176` |
| threshold on d'(tau) | `pitchyin.c:162` | `src/pitch.py:196-200` |
| fall back to global min if no dip | `pitchyin.c:169` | `src/pitch.py:201-205` |
| parabolic sub-sample refine | `pitchyin.c:165,170` (`fvec_quadratic_peak_pos`) | `src/pitch.py:207-215` |

**Where we are ahead of aubio:**

1. **Amplitude gate before periodicity** — `src/pitch.py:140-146` refuses frames below
   `ENERGY_FLOOR = 1e-3`. aubio's `pitchyin_do` has no such gate. Our docstring at
   `src/pitch.py:29-36` makes the correct argument: d'(tau) is amplitude-invariant by
   construction, so a periodic fan hum scores confidence 1.0. aubio will happily report f0 for
   a refrigerator. For a peripheral sitting in one room, that is not a theoretical concern.
2. **Non-finite rejection** — `src/pitch.py:138`, `149-153`. aubio has no such guard.
3. **Confidence-gated emission** — `min_confidence` (`src/pitch.py:86`, `181-183`) converts weak
   candidates into explicit gaps rather than numbers.
4. **Octave post-pass** — `smooth_octaves` (`src/pitch.py:218-286`) is a pYIN-lite median repair.
   Since aubio ships no pYIN and no octave-correction post-pass, **we are ahead here**.

**Where aubio is ahead of us:**

1. **yinfft** — `src/pitch/pitchyinfft.c:111-177`. YIN computed via Wiener–Khinchin:
   `yin->data[tau] = sum - fftout->data[tau]` (`:143`) after two FFTs, i.e. O(N log N) instead of
   our O(W·τmax). At 50 ms / 16 kHz that is 800 samples × 229 lags ≈ 183k multiply-adds per frame
   in pure Python; the FFT form is a couple of 1024-point transforms.
2. **Psychoacoustic spectral weighting** — `pitchyinfft.c:43-55` ships a 34-point weighting
   curve in dB (peaks near 1–2 kHz, −75.8 dB at DC, −17.8 dB above 12.5 kHz), interpolated in
   log-frequency at `:73-99` and applied to the squared magnitude at `:124-132`. **We have no
   spectral weighting at all.** This directly attacks the octave errors we patch up afterwards.
3. **Explicit in-frame octave-halving check** — `pitchyinfft.c:162-172`: when the winning lag is
   below `short_period = samplerate/1300` (`:101`), it re-examines `tau/2` and prefers it if that
   dip also clears tolerance. A cheaper, earlier line of defence than our median post-pass.
4. **A real onset detector.** Our onset work is *energy-envelope* based (`src/tempo.py:31`
   `_peak_times` works off `EnergyFrame`), not spectral. aubio's is a genuine
   complex-domain spectral-flux onset with whitening. Different class of thing entirely.

**Note the tolerance scales differ and must not be copied across.** `yinfft` defaults
`p->tol = 0.85` (`pitchyinfft.c:69`) while `yin` uses `0.15` (`pitchyin.c:71`), because the
Wiener formulation normalizes differently. Blindly porting 0.15 into a yinfft port would break it.

## The most stealable single mechanism

**Sliding local median as an adaptive detection threshold** —
`src/onset/peakpicker.c:110-117`:

```c
median = p->thresholdfn (scratch);
thresholded->data[0] =
    onset_proc->data[p->win_post] - median - mean * p->threshold;
```

Defaults `win_post = 5`, `win_pre = 1`, `threshold = 0.1` (`peakpicker.c:165-167`).

Our segmentation is calibrated *globally per file*: `RELATIVE_ON = 0.10` of the file's peak RMS
and `ABS_FLOOR = 1e-4` (`src/energy.py:28-29`). That works on a finished 90-second clip. It is
the wrong shape for a peripheral that runs continuously in a room whose noise floor drifts — a
threshold set from a file's peak will mis-fire as soon as the acoustic conditions change.

aubio's local median is **re-anchored every 7 frames** to the recent local distribution. That is
exactly the drift-robust property we lack, and it is 6 lines. **Caveat: GPL-3.0 — idea only,
we cannot copy the code.** The mechanism is generic enough to reimplement from the description.

Also worth naming, both GPL-free to reimplement from first principles:

- **Minimum inter-onset interval** as a refractory period — `src/onset/onset.c:81`
  (`if (o->last_onset + o->minioi < new_onset)`), plus a silence gate at `:74`.
- **Spectral whitening before flux** — `src/onset/onset.c:64-66` → `src/spectral/awhitening.c`.
  Removes the spectral tilt so the detector responds to *changes*, not to overall brightness.

## Can we actually use it here?

I tried. This is the practical finding, and it is mostly negative.

`pip3 download aubio` succeeds but yields **only `aubio-0.4.9.tar.gz`** — there is no wheel on
PyPI. Installing it means compiling the C library.

Attempt 1, plain `pip install aubio` → **failed**:

```
python/ext/aubio-types.h:17:10: fatal error: numpy/arrayobject.h: No such file or directory
```

The Python extension needs numpy's headers at build time. **numpy is not installed in this
environment.** (The brief assumed numpy was present — it is not; `import numpy` fails. Same for
scipy, torch, torchaudio, librosa, soundfile, aubio: all missing.) Compounding it, `pkg-config`
is not installed, so aubio's system-library detection step raises `FileNotFoundError` at several
points (benign individually) before the numpy header kills the actual compile.

Attempt 2, numpy present in a venv + `--no-build-isolation` → failed on
`BackendUnavailable: Cannot import 'setuptools.build_meta'`. Attempt 3 added setuptools and is
recorded in the synthesis; see there for the final outcome.

**The governance point regardless of whether attempt 3 succeeds:** aubio is **GPL-3.0**. Our
repo must stay permissively licensed. We can *read* aubio, *benchmark* against it, and
reimplement its ideas from scratch, but **no aubio source line can enter our codebase**. That is
a hard wall, and it is why "just link aubio" is not on the table even if it compiled cleanly.

## Verdict

**Do not adopt. Do mine.** Our YIN is a faithful, correct, dependency-free reimplementation of
the part of aubio that matters for us, and it is *better* on the axes that matter for a
perpetually-running peripheral (amplitude gating, non-finite rejection, honest gaps, octave
repair). aubio's genuine advantages are algorithmic, not architectural: FFT-accelerated YIN and
psychoacoustic spectral weighting.

The single highest-value takeaway is not aubio's pitch at all — it is that our **detection
thresholds are file-global where they should be locally adaptive**. aubio's local-median
thresholding (`peakpicker.c:110-117`) is the thing to copy conceptually.

## Unverified

- No accuracy benchmark against ground-truth f0 (no MIR/FME corpus locally). Everything about
  *relative* accuracy between our YIN and aubio's variants is reasoning from source, not
  measurement. This is the biggest gap in this file.
- `pitchyinfast.c` (201 lines) read only by name; not analyzed.
- `mcomb`/`fcomb` (MPM) not analyzed — potentially interesting as an octave-error alternative
  to the median post-pass, since MPM's NSDF has a clearer peak structure than YIN's d'.