# torch-audiomentations — PyTorch waveform augmentation

Investigated 2026-10-18. Shallow clone of `iver56/torch-audiomentations` @ `f738dda` (2025-11-24).

| fact | value |
|---|---|
| license | **MIT** (asteroid-team) |
| language / size | 69 `.py` files |
| tests | **30** `tests/test_*.py`, incl. `test_differentiable.py` |
| CI | `.github/workflows/ci.yml` — real, runs `coverage run -a -m py.test tests` |
| CI matrix | **python 3.9 only** (`.github/workflows/ci.yml:13`) — single version |
| last push | 2025-11-24 |
| deps | `torch>=1.7.0`, `torchaudio>=0.9.0,<2.9`, `torch-pitch-shift>=1.2.2` (`setup.py:41-43`) |

**MIT — code is reusable.** This is the only repo in our set with a permissive license that also
does real audio work, which makes it the main one we can actually borrow from.

## The 19 transforms

`torch_audiomentations/augmentations/`:

`background_noise`, `band_pass_filter`, `band_stop_filter`, `colored_noise`, `gain`,
`high_pass_filter`, `identity`, `impulse_response`, `low_pass_filter`, `mix`, `padding`,
`peak_normalization`, `pitch_shift`, `polarity_inversion`, `random_crop`, `shift`,
`shuffle_channels`, `splice_out`, `time_inversion`.

Every transform takes `mode` ∈ {`per_batch`, `per_example`, `per_channel`}, `p`, and `p_mode`
(`core/transforms_interface.py`), so one `Compose` can express "vary gain per example but
filters per batch".

**The brief asked about time stretch. There is none.** `time_inversion` exists (reverse the
waveform); `shift` exists (translate in time); `pitch_shift` exists. A duration change is absent.
This is the one augmentation we would most want for a speech corpus and it is not here.

## Which ones are actually well-built

The best three, in order of usefulness to us:

### 1. `ApplyImpulseResponse` — `augmentations/impulse_response.py`

Room simulation, the thing the brief specifically cared about. Convolves the waveform with real
recorded impulse responses:

- **Requires IR audio files on disk.** `impulse_response.py:68` calls `find_audio_files_in_paths`
  and raises `EmptyPathException` if none are found (`:73-74`). It does **not** synthesize RIRs.
  So this needs a corpus of measured room responses to be useful at all.
- **`compensate_for_propagation_delay` (`:116-128`) is the genuinely stealable part:**

```python
propagation_delays = ir.abs().argmax(dim=2, keepdim=False)[:, 0]   # :117
convolved_samples = torch.stack([
    s[:, d : d + num_samples]
    for s, d in zip(convolved_samples, propagation_delays)], dim=0)
```

After convolving, it finds the peak-|amplitude| index of the IR and rolls the result back by
exactly that much, restoring sample alignment. One line of real insight: an RIR convolution
silently *time-shifts* your audio by the IR's propagation delay unless you undo it.

For us this matters more than it would for a classifier. Our audio events are timestamped — we
judge timing, onset positions, tempo. An augmentation that quietly moves every timestamp by a
different per-example offset would poison the exact quantity we measure. **This is the single
most valuable thing in this repo for us, and it is MIT.**

### 2. `SpliceOut` — `augmentations/splice_out.py`

Mixup for audio (`splice_out.py:15` cites arXiv 2110.00046). Cuts N random-width windows out of a
clip and splices in N windows taken from elsewhere, **cross-faded with a Hann window** at each seam
(`:85-86` builds `torch.hann_window(hann_window_len)`). Unrelated clips get blended into one
plausible utterance, which is precisely the anti-overfitting move for a single-speaker corpus.

Also forces splice lengths to even samples (`:82-83`) so a seam cannot land mid-cycle and inject
a DC step — a small correctness detail that most naive implementations miss.

### 3. `ColoredNoise` + `Gain`

`ColoredNoise` (spectral-slope-tilted background noise, `colored_noise.py`) and
`Gain` (`augmentations/gain.py:30-31`, random amplitude in dB, default −18…+6 dB). `Gain`'s own
docstring warns (`:15`) it can return samples outside [−1, 1] and clip. `PeakNormalization`
(`peak_normalization.py:59-60`) is the companion that renormalizes via a global max.

## What is weak

**`PitchShift` is not implemented here.** `augmentations/pitch_shift.py:5`:

```python
from torch_pitch_shift import pitch_shift, get_fast_shifts, semitones_to_ratio
```

It delegates to `KentoNishi/torch-pitch-shift` (MIT, 139★, last push 2024-09-25) — a third-party
package. So the pitch-shift *mechanism* is not in this repo at all, though the dependency is
permissive so the idea is still reusable.

Worse, the maintainers document that it does not work well:

> `PitchShift` does not support small pitch shifts, especially for low sample rates
> ([#151](https://github.com/asteroid-team/torch-audiomentations/issues/151))

(`README.md:63`). At 16 kHz with 70–400 Hz content — our range — small pitch shifts are exactly
what we would want, and this is exactly the case they disown. Treat `PitchShift` as unusable for
our operating point.

**Version churn.** The HEAD commit is literally "Set upper limit on torchaudio version, because of
breaking changes", and `setup.py:42` now reads `torchaudio>=0.9.0,<2.9`. An upstream torchaudio
release will break this package.

**Known operational warts, from the README itself:** multiprocessing memory leaks
(`README.md:61`, [#132]), and multi-GPU/DDP unsupported (`README.md:62`, [#136]). Both are
irrelevant to us (we'd run CPU single-process) but they are the authors documenting their own
rough edges, which is a good sign of honesty.

## Does this solve our one-room problem?

No, and it is worth being blunt about why.

The problem the brief poses — *a peripheral in one room with one person's voice will overfit to
that room and that voice* — is a **real** problem, and augmentation is the right instinct. But
this library only helps if we are training a model. Its value is in diversifying training
inputs. We have no trained model; we have hand-written DSP in `src/`.

So the honest position: **torch-audiomentations is a well-built, MIT-licensed, genuinely useful
library that we currently have nothing to plug it into.** It is a good answer to a question we
have not yet started answering. If we ever train a classifier or a small neural voice model,
this is a strong option and the MIT license makes it a clean one.

What is worth taking *today*, without torch:

- the propagation-delay compensation idea (`impulse_response.py:117`) — reimplementable in ~3
  lines of stdlib;
- the even-splice-length guard (`splice_out.py:82-83`);
- the `mode` × `p` × `p_mode` parameterization shape (`core/transforms_interface.py`) — a nice
  way to express "apply this with 50% probability per example" uniformly.

## Unverified

- No transform was executed. torch/torchaudio are not installed here, so nothing was benchmarked.
  All statements about behaviour are from source and docstrings.
- `utils/fft.py`, `utils/dsp.py`, `utils/mel_scale.py` were not read in depth; they are plausible
  places for further borrowing (FFT, mel scaling, decibels↔amplitude).
- `BackgroundNoise` and the filter transforms (`band_pass`/`band_stop`/`high_pass`/`low_pass`)
  were not analyzed; the filters are thin `torchaudio` wrappers and likely unremarkable.