# Learning journey — audio DSP and audio-ML for a voice peripheral

Written as I went, 2026-10-18. Five repos: aubio, torch-audiomentations, figaro, mechvibes,
noisecraft. Per-repo detail in the sibling files; conclusions in `SYNTHESIS.md`.

This is the part where I record what I expected, what surprised me, and the three times the brief
was wrong and I had to correct course.

---

## What I expected

A fairly specific and, it turns out, largely fictional starting picture:

- **aubio** would confirm our YIN was naive and hand us something sharper. I expected to find
  pYIN in there — I had mentally filed aubio as "the YIN reference implementation" and attached
  the whole YIN research lineage to it.
- **torch-audiomentations** would be the augmentation answer, with a time-stretch transform in it
  (I listed one in the brief's own framing, so that was self-inflicted).
- **figaro** would be a small positive or a small negative; either way a library.
- **mechvibes** would be a Rust modular synth with a node graph, comparable in spirit to
  noisecraft's.
- **noisecraft** would be the exciting one: a model that learns a synthesizer from raw audio, with
  published numbers I could check.

---

## Surprise 1 — aubio has no pYIN, and our YIN was already correct

I expected to be corrected by aubio. I was not. Reading `src/pitch/pitchyin.c:137-171` against
our `src/pitch.py:155-187` line by line, the implementations match on every load-bearing step —
including the two things that are easy to get wrong and that I had assumed were our own careful
work: the fixed-window difference function (each lag sees the same number of terms, avoiding the
bias toward slow periods) and falling back to the global minimum when no dip clears threshold.

Even the default threshold matches: aubio `o->tol = 0.15` (`pitchyin.c:71`), ours
`threshold=0.15` (`src/pitch.py:85`). Two independent implementations converging on the same
conventional constant is a good sign, not a coincidence.

Then I went looking for pYIN to be sure I had found the best method in there. It is not there.
`src/pitch/` has seven methods and none of them is probabilistic. Which means our octave post-pass
(`smooth_octaves`, `src/pitch.py:218-286`, 10 passing tests) puts us *ahead* of aubio on that axis,
because aubio has neither pYIN nor a substitute.

**What this corrected:** my assumption that "battle-tested reference library" means "strictly
better at everything". It meant "better at some things, and we'd have to find out which". It turned
out to be slower to install than to reimplement.

**What surprised me more:** aubio's YIN has *no loudness gate at all*. Our docstring at
`src/pitch.py:29-36` makes the argument that YIN confidence is amplitude-invariant by construction,
so a fan scores confidence 1.0. aubio will happily report a confident f0 for a periodic appliance.
For a peripheral that sits in a room all day, our gate is not a nice-to-have — it is the
difference between a usable pitch track and nonsense.

---

## Surprise 2 — three of the five URLs pointed somewhere else

This was the big one, and it happened three separate times.

1. **`MattMooney/figaro` does not exist.** The owner is **`MattMoony`** — one letter different.
   And the repo is not a VAE library at all. It is a real-time voice-modification app and
   soundboard: an asciimatics TUI, a websockets server, yapsy plugins, PyAudio. I grepped the
   whole `lib/` tree for `import torch` and got **zero hits**. There is no machine learning in it.
   The entire audio layer is 199 lines across three files, and the file that looks like the DSP
   (`lib/filters/filter.py`, 54 lines) is an abstract base class.

2. **`hainguyents13/mechvibes` is an Electron app, not Rust.** Zero `.rs` files. The Rust is in a
   separate repo, `hainguyents13/mechvibes-dx`, which I found by searching and then cloned. Even
   there, the "modular synth" is a voice-pool mixer over pre-recorded keyboard samples — no
   oscillators, no filter graph.

3. **`maximecb/noisecraft` is a browser IDE, and the model repo is deleted.** The generative audio
   model is `maximecb/mincraft`, which returns *Repository not found*. I did not accept the 404:
   I checked the author's account through the GraphQL API and found 39 public repositories, none
   named `mincraft` — so it was deleted or made private, **not archived**, which would have left
   it readable. I searched for surviving forks; every hit was the video game. And the noisecraft
   IDE never mentions it — zero references in the entire tree.

**What this corrected:** the entire premise of three of my five assignments. Two of them are dead
ends, and one of them is dead because the thing I was asked to evaluate *no longer exists*. That
is a much stronger negative than "turned out not to be useful", and it took real work to
establish rather than assume.

**A methodological lesson I want to keep:** the `mincraft` result is only trustworthy because I
went past the first 404. A single `git clone` failure would have produced a much weaker and
wrongly-confident claim ("repo unavailable, skipping"). The account-wide repo listing and the
fork search are what turned "404" into "deleted, and here's the evidence". Cheap checks, and they
were the difference between a hunch and a finding.

**A related self-correction:** partway through I noted that the 30 `.ncft` files in noisecraft
looked like model weights — `.ncft` reads like a model extension, and the brief was talking about
a model. I was wrong. They are JSON node graphs; I verified all 20 of a sample set parse as JSON.
`examples/x_fm_guitar.ncft` opens `{"title":"X.FM v.1.0","nodes":{"0":{"type":"MonoSeq",...}}}`.
The extension had fooled me into an assumption I had not checked, which is exactly the failure
mode I should be most alert to when reading a repo against a pre-formed story.

---

## Surprise 3 — the brief's environment assumption was wrong

The brief states "our environment already has numpy". I checked before recommending anything
torch- or numpy-dependent:

```
python 3.14.4 — numpy MISSING, scipy MISSING, torch MISSING,
                torchaudio MISSING, librosa MISSING, soundfile MISSING, aubio MISSING
```

**numpy is not installed.** `BORROWING.md` records a deliberate pure-stdlib constraint, which is
consistent with this — the brief's assumption appears to be stale rather than the constraint
being wrong. There *is* network access and `gcc`/`make`/`pip3` are present, so numpy is available
if we ever want it. But I have assumed the pure-stdlib posture holds and flagged every
recommendation that would break it.

This changed the shape of every conclusion. It is also what led me to actually try installing
aubio rather than reasoning about it.

---

## Surprise 4 — aubio will not build here, and it is not a configuration problem

I expected "GPL, so don't use it" to be the whole answer. It is not even the main finding.

I tried three times:

1. `pip install aubio` → `fatal error: numpy/arrayobject.h: No such file or directory`. Needs
   numpy headers at build time; also `pkg-config` is absent.
2. numpy + `--no-build-isolation` → `BackendUnavailable: Cannot import 'setuptools.build_meta'`.
3. numpy + setuptools + numpy headers on `CFLAGS` → **hard compile error**:

```
python/ext/ufuncs.c:48:3: error: initialization of 'void (*)(char **, const npy_intp *, ...)'
  from incompatible pointer type 'void (*)(char **, npy_intp *, ...)' [-Wincompatible-pointer-types]
```

aubio 0.4.9 predates the **NumPy 2.0 C-API change**. This is not fixable by flags. There is no
wheel on PyPI either — `pip download aubio` returns only `aubio-0.4.9.tar.gz`.

**What this corrected:** my mental model of "mature C library ⇒ easy to depend on". The library is
fine; the packaging is stale against a NumPy major version and a Python 3.14 interpreter. I would
not have predicted this from reading the repo, and it took an actual build to find out. The lesson
is that installability is a separate question from quality, and for a GPL dependency we were never
going to adopt anyway, I spent real time confirming it.

The GPL-3.0 licence remains the decisive fact, and it is why recommendation R1 in the synthesis is
worded as *reimplement from the description* rather than *port this*.

---

## Surprise 5 — the most valuable finding was not an algorithm

I went in looking for a better pitch tracker. The finding I care most about is about thresholds.

Our segmentation is calibrated **per file**: `RELATIVE_ON = 0.10` of the file's peak RMS,
`ABS_FLOOR = 1e-4` (`src/energy.py:28-29`). That is correct for a finished 90-second clip, and
wrong for a box that runs continuously in a room whose noise floor drifts. aubio re-anchors its
detection threshold to the **local** distribution every few frames — sliding median plus a
mean-scaled term, `peakpicker.c:110-117`.

A file-peak-relative threshold is a property of the recording, not of the room. The moment the
recording is a live stream, it stops being the right shape. This is a few lines of stdlib and it
addresses the actual deployment condition — a peripheral in one room — that the whole project is
named for.

**What this corrected:** my search was for "what computes pitch better", and the answer that
mattered was "our notion of 'loud enough' is anchored to the wrong thing". Worth remembering that
the most valuable findings often come from the mismatch between an artifact and *our* operating
condition, not from the artifact being sophisticated.

---

## Smaller things that stuck

- **Measured our own baseline** rather than asserting it: 2.8× realtime, 3.64 ms per frame on 90 s
  of real speech, ~36% of one core. Useful as an honest anchor for any future speed claim.
- **`PitchShift` in torch-audiomentations is not implemented there** — it delegates to
  `torch_pitch_shift` (`pitch_shift.py:5`), and the maintainers document that it fails for small
  shifts at low sample rates, which is exactly our 16 kHz / 70–400 Hz case. Also: **there is no
  time-stretch transform at all**, which I had assumed there was.
- **The best mechanism in torch-audiomentations is a warning, not a feature.**
  `impulse_response.py:117` compensates for the propagation delay that convolution with a room
  impulse introduces. For a classifier that is nearly invisible. For us it is a live hazard: our
  product judges onset positions and tempo, so anything that shifts timestamps by a
  per-example offset would poison the exact quantity we measure. It is MIT and about three lines.
- **mechvibes-dx tests an architectural invariant** — `engine.rs:663-664` asserts the audio loop never
  polls the audio device. Cheap and worth copying when we split our worker/UI boundary.

---

## What I would tell the next person

1. **Check the URL before building a plan on it.** Three of five were wrong. `MattMoony` not
   `MattMooney`; mechvibes is Electron, the Rust is `mechvibes-dx`; noisecraft is an IDE, the
   model is deleted.
2. **A 404 is not a conclusion.** `mincraft` required an account-wide repo listing and a fork
   search before "deleted" became a defensible claim instead of a guess.
3. **Check the environment before recommending a dependency.** The brief said numpy was present.
   It was not, and that single fact reshaped every recommendation.
4. **Reference implementations are peers, not authorities.** Our YIN matches aubio's
   structurally, beats it on four operational axes, and loses to it on two algorithmic ones.
   Assuming the library wins on everything would have made us worse.
5. **The best finding was about our own thresholds, not their algorithms.** I went looking for a
   better pitch tracker and the useful thing was noticing that a file-global threshold is the
   wrong shape for a device that never stops running.
6. **Two honest negatives beat a hedged maybe.** figaro and noisecraft are dead ends, cleanly, and
   saying so with evidence is worth more than a soft "might be worth revisiting".

---

## Files

- `aubio.md` — deep dive, algorithm inventory, the YIN comparison, three failed build attempts
- `torch-audiomentations.md` — the 19 transforms, which are well-built, and where they break
- `figaro.md` — the clean negative: not a VAE library, GPL, unrunnable, and 199 lines of plumbing
- `mechvibes.md` — the Electron/Rust split, and the two MIT mechanisms worth taking
- `noisecraft.md` — the model repo is deleted; what remains is a browser IDE with no ML
- `SYNTHESIS.md` — the head-to-head, the five stealable mechanisms, verdicts, and what I did not verify