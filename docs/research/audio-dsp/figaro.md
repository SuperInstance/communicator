# figaro — investigated, and it is a clean dead end

Investigated 2026-10-18.

## First: the URL in the brief was wrong

The brief gave `MattMooney/figaro`. That 404s. The real owner is **`MattMoony`** (one fewer
letter): `MattMoony/figaro`, 1,056★, not archived. `MattMoony` also owns `zincware/`-adjacent
audio work and `cortex` (a PyTorch ML library) — worth noting for a different reason, see below.

Cloned from `https://github.com/MattMoony/figaro.git`.

| fact | value |
|---|---|
| license | **GPL-3.0** |
| stars | 1,056 |
| last commit | **2022-05-29** — over four years stale |
| size | 27 `.py` files |
| tests | **0** |
| CI | **none** (`git ls-files` finds no workflow/travis/circleci config) |
| docs | 6 `.md` |

## Second: it is not a VAE library

The brief describes it as "a VAE library in PyTorch". It is not. `README.md:17` describes it as:

> Real-time open-source voice modification program & sound board. Can be useful for many things,
> especially when used in combination with virtual sound i/o devices.

It is an Electron-style desktop/TUI app: an `asciimatics` terminal UI (`lib/gui/`), a
websockets HTTP server (`lib/server/`), YAML plugin loading (`yapsy`), a soundboard
(`lib/sounds/sound.py`), and a filter chain (`lib/filters/`, `lib/transformer.py`).

**There is no machine learning in it at all.** I grepped the entire `lib/` tree:

```
grep -rn "import torch\|from torch\|VAE\|vae" lib/ --include=*.py   → 0 hits
```

Not one `import torch`. Not one VAE. `requirements-unix.txt` lists `numpy`, `PyAudio`, `pydub`,
`websockets`, `asciamatics`, `Pillow`, `PyJWT`, `pynput` — no ML framework whatsoever.

## How thin the DSP actually is

The entire audio-processing layer is **199 lines across three files**:

```
wc -l lib/filters/filter.py lib/transformer.py lib/sounds/sound.py
  54 lib/filters/filter.py     ← an abstract base class, not an implementation
  53 lib/transformer.py
  92 lib/sounds/sound.py
```

`lib/filters/filter.py` is a 54-line abstract base holding a list of `Filter` objects behind a
`Lock`; `lib/transformer.py` is a 53-line `apply_all()` loop over that list. The actual filter
implementations are not in the library — they live in plugins. So there is essentially no DSP to
read, evaluate, or borrow. This is a thin application shell around user-supplied plugins.

## Third: it cannot run here at all

`requirements-unix.txt` pins `PyAudio==0.2.11`. That is a hard dependency on a live audio input
device. Our constraint is WSL with no audio hardware, file/WAV-based testing only. Beyond that,
the app wants an interactive terminal UI event loop and a network server — architecturally
incompatible with our batch, file-driven style.

The pins are also ancient (`numpy==1.21.0`, `websockets==9.1`, `six==1.15`, `pynput==1.6.8`) and
predate Python 3.14.

## Verdict: dead end, on four independent grounds

1. **No ML.** Zero torch, zero VAE. The premise for including it does not hold.
2. **GPL-3.0.** Even if there were something, we could not take code.
3. **Unrunnable here.** PyAudio needs audio hardware; we have none.
4. **Nothing to borrow anyway.** 199 lines of plumbing, most of it an abstract base class.

Any one of these would be enough. Together they make this the clearest negative result in the set,
and it cost one clone to establish.

## One loose thread, deliberately not pulled

The same author (`MattMoony`) maintains **`cortex`**, described as "a machine learning library for
PyTorch". That is a *different* repo from the one I was assigned, and it may actually be the
VAE-adjacent thing the brief was half-remembering. I have not investigated it, and it is outside
this task's scope — flagging it so someone can decide whether a follow-up is warranted.

Also worth noting: the real `noisecraft` ML repo (`maximecb/mincraft`) turns out to be **deleted**
entirely — see `noisecraft.md`. figaro and that story are the same failure mode: a repo that was
widely described, is no longer there under that description, and whose surviving artifact is a
shell rather than the thing everyone remembers.

## Unverified

- Plugin code (where the real filters live) was not fetched — out of scope, and irrelevant given
  the four disqualifiers above.
- No claim is made about the quality or license of any third-party figaro plugin.