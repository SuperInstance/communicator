# Synthesis — audio DSP and audio-ML for a local-first voice peripheral

2026-10-18. Companion to the five repo files in this directory. Everything below is grounded in
source that was cloned and read; `file:line` refers to the upstream repos, and to our own
`src/` where noted.

## TL;DR

- **aubio: do not adopt, do mine.** We already reimplemented its YIN correctly. It cannot even be
  built in this environment. It is GPL-3.0, so no line of it can enter our codebase regardless.
- **The most valuable thing in this entire investigation is not a pitch algorithm.** It is that our
  detection thresholds are **file-global** where a continuously-running peripheral needs them
  **locally adaptive**.
- **figaro and noisecraft are both dead ends**, for unrelated reasons, and one of them is dead
  because the repository *no longer exists*.
- **torch-audiomentations is good and MIT but has nothing to plug into** — we have no model to
  augment.
- **mechvibes-dx contributes two small, clean, reusable mechanisms** and nothing else.

---

## 1. The aubio head-to-head

### Our baseline, measured

Benchmarked `src/pitch.py` on real speech in this repo (`examples/realaudio/`), Python 3.14,
single core, no numpy:

| file | sr | duration | wall | rate | per frame |
|---|---|---|---|---|---|
| `alice01_excerpt.wav` | 16 kHz | 90.00 s | 32.72 s | **2.8× realtime** | 3.64 ms |
| `demo-utterance.wav` | 16 kHz | 0.72 s | 0.18 s | 4.0× realtime | 2.68 ms |

On the 90 s clip: 8,996 frames, 4,478 voiced (50%), 2,350 rejected by the energy floor,
**0 poisoned frames**. Octave post-pass adds 1.5 ms total — negligible.

So our YIN is ~3.6 ms per 10 ms hop, i.e. about **36% of one core while processing audio in
realtime on one thread**. That is real but not comfortable, and it is the honest baseline for any
speed argument.

### Are we worse, better, or the same?

**Same, and I can prove it structurally.** Reading `src/pitch/pitchyin.c:137-171` against our
`src/pitch.py:155-187`, every load-bearing step matches:

- fixed-window difference function (each lag sees the same number of terms) —
  `pitchyin.c:149-154` vs `src/pitch.py:157-169`. Both correctly avoid the classic
  shrinking-window bias toward slow periods.
- cumulative mean normalization — `pitchyin.c:155-160` vs `src/pitch.py:171-176`
- absolute threshold on d′(τ) — `pitchyin.c:162` vs `src/pitch.py:196-200`
- fallback to global minimum when nothing dips — `pitchyin.c:169` vs `src/pitch.py:201-205`
- parabolic sub-sample refinement — `pitchyin.c:165,170` vs `src/pitch.py:207-215`

And the defaults converged independently: aubio `o->tol = 0.15` (`pitchyin.c:71`), ours
`threshold=0.15` (`src/pitch.py:85`).

**We are strictly better on four axes**, all of which matter more for a perpetual peripheral than
raw accuracy does:

1. **Amplitude gate before periodicity** (`src/pitch.py:140-146`). aubio's `pitchyin_do` has no
   loudness gate. Our own docstring (`src/pitch.py:29-36`) gets the argument exactly right: d′(τ)
   is amplitude-invariant *by construction*, so a periodic fan or a synthesizer bleeding through
   speakers scores confidence 1.0. aubio will report a confident f0 for a refrigerator.
2. **Non-finite rejection** (`src/pitch.py:138`, `149-153`), which aubio has no equivalent of.
3. **Confidence-gated emission** (`min_confidence`, `src/pitch.py:86`, `181-183`) — weak candidates
   become explicit gaps rather than numbers that downstream code trusts.
4. **Octave repair** — `smooth_octaves` (`src/pitch.py:218-286`), 10 passing tests
   (`tests/test_octave_smoothing.py`). **aubio ships no pYIN and no octave post-pass at all.**
   On this specific axis we are ahead of the most battle-tested library in the set.

**aubio is better on two axes, both algorithmic:**

1. **FFT-accelerated YIN** — `pitchyinfft.c:111-177`. Uses Wiener–Khinchin:
   `yin->data[tau] = sum - fftout->data[tau]` (`:143`), two FFTs instead of an O(W·τmax) inner
   loop. Our 800 samples × 229 lags ≈ 183k multiply-adds per frame, in pure Python, is exactly
   the cost this removes.
2. **Psychoacoustic spectral weighting** — the 34-point dB curve at `pitchyinfft.c:43-55`,
   interpolated in log-frequency at `:73-99` and applied at `:124-132`. **We have no spectral
   weighting at all.**

### Concrete recommendations, in priority order

**R1 — Make detection thresholds locally adaptive. (Highest value, smallest change.)**
This is the one finding I would act on regardless of everything else. Our segmentation is
calibrated per *file*: `RELATIVE_ON = 0.10` of the file's peak RMS, `ABS_FLOOR = 1e-4`
(`src/energy.py:28-29`). aubio re-anchors its detection threshold to the **local** distribution
every few frames — `peakpicker.c:110-117`:

```c
median = p->thresholdfn (scratch);
thresholded->data[0] = onset_proc->data[p->win_post] - median - mean * p->threshold;
```

with `win_post = 5`, `win_pre = 1`, `threshold = 0.1` (`peakpicker.c:165-167`). The median gives
robust location; the mean term scales the threshold to local energy. Ours is computed once from
the whole clip. A file-peak-relative threshold is fine for a finished 90-second recording and
wrong for a box that runs all night in a room whose noise floor drifts — HVAC cycling, a fan
spinning up, traffic. Reimplement from the description (GPL-3.0: **idea only, no code copied**);
it is a few lines plus a median over a 7-frame window, pure stdlib, no numpy.

**R2 — Add a minimum-inter-onset-interval refractory period.** `onset.c:81`:
`if (o->last_onset + o->minioi < new_onset)`. Onset functions like energy-envelope peaks produce
a cluster of near-duplicate peaks across one syllable; a refractory period collapses them. We
have no equivalent. Trivially reimplementable.

**R3 — Adopt psychoacoustic weighting only if we first move to yinfft, and do not copy its
tolerance.** This is a real but conditional recommendation. The weighting curve only exists in
the FFT formulation, and the two aubio variants have *different* tolerance scales because the
Wiener normalization differs: `yin` uses `tol = 0.15` (`pitchyin.c:71`), `yinfft` uses
`tol = 0.85` (`pitchyinfft.c:69`). Porting the 0.15 into a yinfft port would silently break it.
Also: yinfast/yinfft use an FFT, and **we have no FFT and no numpy in this environment** — so
this is explicitly a "if we ever add numpy" item, not now.

**R4 — Do not adopt aubio. Three independent reasons:**
- **License: GPL-3.0.** Hard wall. Not "read and adapt", not "vendor with attribution".
- **It does not build here.** See §3.
- **It solves a problem we already solved.** The only two things it genuinely has that we lack
  (FFT speed, spectral weighting) both require a dependency we do not have, to fix a cost we
  currently pay at 36% of one core.

**R5 — If our onset work ever needs to be spectral, that is a genuine gap.** Ours is
energy-envelope based (`src/tempo.py:31`, `_peak_times` off `EnergyFrame`), which is a reasonable
way to get *speech syllable rate*. aubio's is complex-domain spectral flux with whitening
(`onset.c:64-71`). Different class of tool. For measuring how fast someone is talking, envelope
peaks are fine and cheaper. For locating precise articulation instants, they are not. Worth
knowing which of the two problems we actually have before borrowing anything.

---

## 2. The five most stealable mechanisms

Ordered by value to us. License column is decisive: **only #1 and #3 are in permissively-licensed
code.**

| # | mechanism | where | license | verdict |
|---|---|---|---|---|
| 1 | **RIR propagation-delay compensation** — after convolution, roll the result back by `argmax|ir|` so sample alignment survives | `torch-audiomentations/augmentations/impulse_response.py:117` | **MIT** | **Take it** |
| 2 | **Locally adaptive median threshold** for detection | `aubio/src/onset/peakpicker.c:110-117` | **GPL-3.0** | Reimplement from description |
| 3 | **Click-free voice eviction** — ramp a stolen voice to zero over ~10 steps instead of cutting | `mechvibes-dx/src/libs/audio/engine.rs:420-435` | **MIT** | **Take it** |
| 4 | **Even-splice-length guard** — force random splice windows to an even sample count so a crossfade seam cannot land mid-cycle and inject a DC step | `torch-audiomentations/augmentations/splice_out.py:82-83` | **MIT** | Take when we mix |
| 5 | **Symmetric edge fade clamped to half the buffer** — fade in/out, clamped so short samples don't double-multiply the overlap | `mechvibes-dx/src/libs/audio/engine.rs:382-416` | **MIT** | **Take it** |

**On #1, which is my top pick.** `propagation_delays = ir.abs().argmax(dim=2, keepdim=False)[:, 0]`
then slicing from that offset. Convolution with a room impulse response silently shifts your audio
by the IR's propagation delay. For a classifier that is often invisible. For *us* it would be
actively harmful, because our whole product judges **timing** — onset positions, tempo,
inter-onset intervals. An augmentation that moves every timestamp by a different per-example
offset would poison precisely the quantity we exist to measure. The whole line is a warning about
what convolving a signal with an IR does to its timestamps. It is MIT and it is ~3 lines of stdlib
to reimplement.

**On #2, the most valuable and the one we cannot take.** GPL-3.0. The mechanism is generic —
robust location (median) plus scale (mean) over a short sliding window — and describable in one
sentence without looking at aubio's source. Reimplement from the description. **Do not copy the
code.**

---

## 3. Can we even use these? (the environment reality)

The brief said "our environment already has numpy". **It does not.** Checked directly:

```
python 3.14.4 — numpy MISSING, scipy MISSING, torch MISSING,
                torchaudio MISSING, librosa MISSING, soundfile MISSING, aubio MISSING
```

There *is* network access and `gcc`/`make`/`pip3` are present, and numpy installs cleanly
(`numpy-2.5.4-cp314`), so this is a policy choice rather than a hard limit. But `BORROWING.md`
records a deliberate "pure stdlib (no numpy/scipy in the runtime)" constraint, so I have assumed
it holds and flagged every recommendation that would break it.

**I tried to install aubio, three times, and it failed all three.**

1. `pip install aubio` → `python/ext/aubio-types.h:17:10: fatal error: numpy/arrayobject.h: No
   such file or directory`. The extension needs numpy headers at build time. Also, `pkg-config`
   is not installed, so aubio's system-library detection raises `FileNotFoundError` at several
   points.
2. numpy in a venv + `--no-build-isolation` → `BackendUnavailable: Cannot import
   'setuptools.build_meta'`.
3. numpy + setuptools + `--no-build-isolation` + numpy headers on `CFLAGS` → **hard compile
   failure**, and this one is not fixable by configuration:

```
python/ext/ufuncs.c:48:3: error: initialization of 'void (*)(char **, const npy_intp *, ...)'
  from incompatible pointer type 'void (*)(char **, npy_intp *, ...)' [-Wincompatible-pointer-types]
```

aubio 0.4.9 predates **NumPy 2.0's C-API change** and its `ufunc` signatures are incompatible with
the NumPy 2.x headers we have. There is no wheel on PyPI either — `pip download aubio` returns
only the source tarball `aubio-0.4.9.tar.gz`. Patching a GPL-3.0 library's C source to make it
build is not a route we should take.

**Net: aubio is unusable here on both governance and engineering grounds.**

**torch-dependent paths are all currently blocked**, since torch/torchaudio are absent. Anything
I recommend from torch-audiomentations is either (a) a mechanism small enough to reimplement in
stdlib, which is how I've framed all of them above, or (b) blocked pending a deliberate decision
to add torch.

---

## 4. Honest verdicts

### figaro — dead end, cleanly

**It is not a VAE library.** The brief said "a VAE library in PyTorch". It is a real-time
voice-modification app and soundboard: an `asciimatics` TUI, a websockets server, yapsy plugins.
`grep -rn "import torch\|from torch\|VAE\|vae" lib/ --include=*.py` → **zero hits**. No ML
framework appears in `requirements-unix.txt` at all. The entire audio layer is 199 lines across
three files, and `lib/filters/filter.py` (54 lines) is an abstract base class, not an
implementation.

It also 404s at the URL we were given (`MattMooney` vs the real **`MattMoony`**), is **GPL-3.0**,
last committed **2022-05-29**, has **0 tests and no CI**, and requires **PyAudio** — a live audio
input device, which we do not have.

Four independent disqualifiers. This was the cheapest negative result in the set and it is a
*complete* one: the VAE idea we were sent to evaluate does not exist in that repository.

One live thread: the same author maintains **`MattMoony/cortex`**, a PyTorch ML library. Outside
our scope, uninvestigated, but it is the more likely referent for whatever the brief half-remembered.

### noisecraft — dead end, and actively misleading

**The generative model repository no longer exists.** `maximecb/mincraft` returns *Repository not
found*. I verified via the GraphQL API that the author has **39 public repositories and none is
named `mincraft`** — so it was deleted or made private, **not archived** (which would have left
it readable). I searched for surviving forks; every hit is the video game. And the noisecraft IDE
never references it — `grep -rniE "mincraft"` across that tree returns zero hits.

**The repo we were given has no ML in it.** `maximecb/noisecraft` is a browser visual programming
language (Web Audio / Web MIDI, plain JS/HTML/CSS). **0 Python files.** The 30 `.ncft` files I
initially took for model weights are JSON node graphs — all 20 of a sample set parse as JSON, e.g.
`examples/x_fm_guitar.ncft` contains `{"title":"X.FM v.1.0","nodes":{"0":{"type":"MonoSeq", ...}}}`.

**So the brief's central question cannot be answered.** "What are its measured results and its real
capabilities?" — the artifact is not obtainable. I have deliberately **not** reconstructed
published results from memory and presented them as findings. If anyone cites Noisecraft numbers
to justify a plan, they are citing something we cannot verify from primary sources.

Status is **unknown**, not "validated". One more reason to distrust the framing: the same author's
`atari-representation-learning` (unsupervised state representation learning) is public, live, and
genuinely on-topic for "learn a compact representation of acoustic states" — the thing the brief
speculated figaro might help with. Worth someone's time; not mine.

### mechvibes — mostly dead end, two clean wins

The assigned repo is an **Electron/JS app**, not Rust (`git ls-files | grep -c '\.rs$'` → 0). The
Rust is `hainguyents13/mechvibes-dx` (MIT, 332★, actively developed). Even there, the "modular
synth" is a **voice-pool mixer over pre-recorded samples** — `rodio`/`cpal` sinks, a command
channel, JSON sound packs mapping keycode → filename. No oscillators, no filter graph. It needs a
live audio device *and* a global keyboard hook, so it cannot run in headless WSL.

But mechanisms #3 and #5 above are good, MIT, and directly applicable to bounded-resource audio
on our side. Take those two, skip the project.

### torch-audiomentations — good library, wrong moment

MIT, 30 test files, real CI, honest README that documents its own known warts. The RIR transform
is the best of the 19 and its propagation-delay compensation is our top stealable mechanism.

But it is an **augmentation library**, and we have **no model to augment** — `src/` is
hand-written DSP. It is the right answer to a question we have not started asking. Two specific
gaps if we ever do: **there is no time-stretch transform** (the brief assumed one), and
`PitchShift` is delegated to a third-party package (`torch_pitch_shift.py:5`) and is documented as
unreliable for small shifts at low sample rates — which is exactly our 16 kHz / 70–400 Hz
operating point.

---

## 5. What I did not verify

Stated plainly, because these bound how much weight the above deserves:

- **No accuracy measurement of any kind.** There is no ground-truth f0 corpus locally (no MIR/FME).
  Every claim about relative *accuracy* between our YIN and aubio's variants is reasoning from
  source, not measurement. **This is the biggest gap in this document.** The structural
  equivalence findings (fixed window, threshold, parabolic refinement) are solid because they are
  code-level. The claims "yinfft would be more accurate" and "yinfft is faster" are **not
  measured** — the first is a well-supported expectation from the psychoacoustic weighting, the
  second follows from O(N log N) vs O(N·τ) but was not benchmarked because aubio will not build.
- **No torch-dependent code was executed** (torch absent). All torch-audiomentations statements
  are from source and docstrings.
- **No Rust was compiled** (no audio device; `cargo build` not attempted).
- `aubio/src/pitch/pitchyinfast.c`, `pitchmcomb.c` (McLeod Pitch Method) and `pitchfcomb.c` were
  not analyzed. **MPM is the one algorithmic thread I would pull next**: its NSDF gives a cleaner
  peak structure than YIN's d′, which makes it a candidate for a better octave answer than our
  median post-pass. Unread, so this is a hunch, not a finding.
- `mechvibes-dx/src/libs/audio/soundpack_loader.rs` (915 lines) and `resampler.rs` unread.
- I did not investigate `MattMoony/cortex` or `maximecb/atari-representation-learning`, both
  out of scope, both flagged above as the better leads if someone wants to follow the
  representation-learning thread properly.