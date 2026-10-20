# Always-on voice detection — a learning journey

Three repos, one question: **what should run in the hot path of a peripheral that must
watch audio continuously and cheaply?**

- [SILERO-VAD.md](SILERO-VAD.md) — the answer
- [OPENWAKEWORD.md](OPENWAKEWORD.md) — rejected on licensing, worth copying
- [PYANNOTE.md](PYANNOTE.md) — rejected, structurally
- [SYNTHESIS.md](SYNTHESIS.md) — the ranked comparison and the recommendation
- [JEV-AS-INSTRUMENT.md](JEV-AS-INSTRUMENT.md) — the required decision-model writeup

Reproduce everything:

```bash
# measurements (needs a python with onnxruntime; I used a throwaway venv)
/tmp/ortenv/bin/python docs/research/alwayson/measure_hotpath.py --watcher

# the Jev suite
set -a && . /home/proart/projects/glasses/.env && set +a
/home/proart/projects/glasses/.venv/bin/python docs/research/alwayson/jev_alwayson.py
```

Pinned commits: openWakeWord `368c037`, silero-vad `1e261b0`, pyannote-audio `b749285`.

---

## The headline, before the journey

**Put a silero-vad ONNX model in front of the existing `src/watcher.py`. One new
dependency (`onnxruntime`). 2.220 MB, MIT, 0.23% of a core — and it scores 0.0036 on the
steady hum that our watcher calls speech.**

## What I expected

Honestly, I expected this to be a licensing-and-tradeoffs survey with a boring answer,
and I expected two specific things to go wrong in my own favour:

1. **I expected `onnxruntime` to be installed.** The brief said the runtime has no
   `torch`, `librosa`, `scipy`, `soundfile` or `aubio` and implied ONNX was the escape
   hatch. So I assumed an ONNX-only path existed and cost nothing extra.
2. **I expected "several seconds of hum reads as speech" to be a documented caveat rather
   than a measured 89%.** `src/watcher.py:40-47` declares the failure in prose. I expected
   to cite that prose, not to go measure it.

## What surprised me

### 1. `onnxruntime` is ALSO missing — and that reframes the whole question

The brief listed the absent packages and `onnxruntime` was not among them. It is absent:

```
numpy 2.5.4   OK
onnxruntime   MISSING
torch, torchaudio, librosa, scipy, soundfile, aubio, sklearn,
tflite_runtime, ai_edge_litert, speexdsp_ns, requests, joblib   all MISSING
```

This is the most consequential thing I learned, and it cuts both ways:

- **"ONNX-only path" is not a thing you get for free.** Every candidate needs
  `onnxruntime`, a ~50 MB binary wheel. It is still *vastly* cheaper than torch — a
  handful of MB of runtime versus a research ecosystem — but "no torch" and "no new
  dependency" are different claims and I had been conflating them.
- It converted "silero-vad needs no torch" from a **yes** into a **yes, if you reimplement
  30 lines of its wrapper.** See §3.

### 2. openWakeWord's models are CC BY-NC-SA 4.0 — non-commercial

The repo-level licence badge says Apache-2.0, and that is true of the *code*. The README's
own License section says the **pre-trained models** are
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0**. I expected this to be the
story, and it is: it disqualifies the only part of openWakeWord that makes it
openWakeWord. Retraining your way out needs torch + speechbrain + tensorflow-cpu +
deep-phonemizer **and ~30,000 hours of negative data** — the data is the real blocker.

This is also a good example of why you read the licence paragraph and not the badge: an
automated `licenseInfo` check returns `Apache-2.0` and is wrong about the artifact you
actually load.

### 3. silero-vad's Python package cannot be installed in our runtime, but its model can

This is the finding that shaped the implementation, and it took three steps to see:

- `pyproject.toml:52-55` advertises an `onnx-cpu` extra with just `onnxruntime` + `numpy`,
  which reads like an official torch-free path.
- But `pyproject.toml:30-33` puts `torch>=1.12.0` in **base** `dependencies`. Extras only
  *add*. So `pip install silero-vad[onnx-cpu]` **installs torch anyway.**
- And `src/silero_vad/model.py:2` does `import torch` at **module scope**, reached from
  `__init__.py:7`. `OnnxWrapper` itself uses `torch.zeros`/`torch.cat`/`from_numpy`
  (`utils_vad.py:104-137`). It is an *ONNX-inference* wrapper, not a *numpy-only* wrapper.

So the package is unavailable to us. **But the model is not.** The working path is
`onnxruntime` + `numpy` directly — and **openWakeWord already wrote exactly that, MIT,
with silero's licence header preserved** (`openwakeword/vad.py:15-39`). We take the
technique and the copy from openWakeWord, the weights from silero.

The irony worth writing down: the piece of openWakeWord we may actually use is the
third-party piece it vendored.

### 4. The hum failure is much worse than "declared"

I measured it rather than citing it. Over 4-second clips, our watcher reports:

| signal | seconds called "speech" | onsets/min |
|---|---:|---:|
| steady 60 Hz hum | **3.550 / 4.000** | 15 |
| keyboard clicks | 3.550 | 15 |
| flat white noise | 3.550 | 15 |
| **near-silence** | **3.550** | 15 |

**Our watcher calls near-silence speech at exactly the same rate it calls hum speech.**
The six non-speech cases all land on 3.550 s — that is saturation, not measurement: the
segmenter is marking essentially the whole clip. On a room with a fan, **we get no pause
events at all**, so every downstream table built on pauses is wrong in the same direction.

silero-vad on the same clips: 0.0036–0.0081 mean, max 0.1736, never above 0.5. Speech:
0.75–0.88.

### 5. Our own hot path is 39% of a core, and its docstring says "microseconds"

`src/watcher.py:1` — *"No model, no network, microseconds."*

Measured in the numpy-only venv, no onnxruntime involved: **39.13 ms per 100 ms of audio**.
That is 39.1% of a core and a realtime ratio of **2.56×**. **That 39.13 ms is the mean
over the SIX NON-SPEECH fixtures** (hum 31.913, fan 41.095, clicks 48.387, music 33.877,
white noise 39.307, silence 40.153 ms). **On real speech the watcher is 21.9–36.1 ms**
(speech_jane 21.945, speech_mycroft 24.666, speech_alexa 36.124; mean 27.578 ms) —
a realtime ratio of **2.77–4.56×**.

It is still affordable on this CPU and still nothing like a model. But the docstring is
wrong by ~1000×, and it reframes the economics: the hot path is the second-largest cost in
the system, so anything removable from it is worth removing. A 0.23% gate that deletes a
39% failure mode is the best trade in this lane — where 39% is the **non-speech** cost —
and gated, the YIN tracker would only run on real speech (21.9 ms per push, measured)
instead of on every periodic hum (48.4 ms per push, measured, worst case).

> **CORRECTION (2026-10-19) — "39.1% of a core" had no denominator.** From
> `hotpath_measurements.json` → `watcher`: the six non-speech fixtures average **39.122 ms**;
> the three speech fixtures average **27.578 ms**. The 8× Pi projection (313% of a core)
> further down is the **non-speech worst case stated as the general case**. The rate is real
> and the trade-off is sound; the population was missing. `NOTES-AUDIT.md` F11.

**The lesson I did not expect:** I set out to evaluate three external libraries and ended up
with the most actionable finding about our own code.

### 6. Three of their headline numbers were wrong or unverifiable

| Claim | Reality |
|---|---|
| silero README: chunk takes "less than **1 ms**" | Their own wiki table says **189 μs** (V5 ONNX, 1 thread). ~5× conservative. I measured **74.9 μs**. |
| silero README: ONNX "up to 4-5× faster" than JIT | Their own table: 325/189 = **1.72×** |
| "silero-vad is ~1.8M params" (in the brief) | The shipped 16 kHz ONNX is **309,633** params; the opset-18 export is 545,689. **No artifact supports 1.8M.** |
| openWakeWord README: "15-20 models on a Raspberry Pi 3 core" | **No runnable benchmark exists in the repo.** `benchmark/benchmark.py:29` passes `input_sizes=[16]`, which appears **nowhere** in v0.6.0 — it would raise `TypeError`. `examples/benchmark_efficiency.py` needs `pyaudio` and a live mic. |

The parameter-count trap is worth its own line: a naive `graph.initializer` count on
`silero_vad.onnx` returns **0**, because the whole body is inside `If` subgraphs. You have
to walk recursively or you will "prove" the model is empty.

### 7. silero-vad's CI does not run automatically

`.github/workflows/test.yml:3-4`:
```yaml
on:
  workflow_dispatch:       # запуск вручную
```
Manual trigger only. The green README badge is for a workflow that gates nothing, and
`python-publish.yml` fires on tag push — so a release can publish without tests having run
on that commit.

For contrast: **openWakeWord's CI is the cleanest of the three** — bare `pytest` on every
push to `main`, on Linux *and* Windows, with no `|| true` and no `continue-on-error`.
pyannote's is also real (PR → `develop`, push → `develop`/`master`/`release/*`). silero
has the most thorough *tests* (17, including 6 that specifically cover the optional-
dependency matrix) and the worst *triggers*.

The lesson: count the tests **and** read the trigger. A repo with 17 tests that only runs
them on demand is weaker evidence than one with 16 tests that runs on every push.

### 8. pyannote's weights are gated — but that is not the real objection

I expected to find a proprietary licence and to have to warn you off it. Instead:

- Code: **MIT** (`LICENSE:1-3`, CNRS).
- Weights: **CC-BY-4.0** (`speaker-diarization-community-1`) or **MIT**
  (`segmentation-3.0`, `speaker-diarization-3.1`) — both **commercially usable**.
- But `gated: auto` on the Hub: an account, a token, and a **click-through licence
  acceptance** (`README.md:28-29`). And the 25.4 MB embedding inside the pipeline carries
  **WeSpeaker's own** `LICENSE.WeSpeaker`, which a repo-level check will not catch.

So the licence is *fine* and the objection is different: **a licence acceptance by a third
party should not be a boot-time dependency of an always-on peripheral.** That is a more
honest argument than "the licence is bad", and I would rather make it.

The objection that actually decides it is structural: **diarization's clustering step is
order-dependent and non-incremental**, so it cannot be a bounded-memory always-on loop —
which is exactly the property `src/watcher.py:126-128` is built around. Plus
`grep -rn -i onnx` over the entire repo returns **zero** matches, and the only published
speed figure is on an **NVIDIA H100 80GB** (`README.md:110`). No CPU number exists, so
"does it fit a peripheral" is not a question anyone has answered.

## What I got wrong in my own work

Two real errors, both caught by instrumenting rather than trusting.

**1. My position-bias control was confounded.** I built the reversed option set as
`{v: k for k, v in FAMS.items()}` — swapping keys and values but **preserving insertion
order**, so the same option was still first in both orders. "The pick travelled with the
content" was then uninterpretable: equally consistent with reasoning and with a position
preference. Fixed with `reversed(list(...))`, and the harness now *prints* the first option
in each order so it cannot recur. Details in
[JEV-AS-INSTRUMENT.md](JEV-AS-INSTRUMENT.md) §E3b.

**2. My silero wrapper aliased its own context buffer.** I wrote
`feed = ctx = feed[:, -64:]`, which truncates `feed` to 64 samples and blows up inside the
model's `Pad` node. The error surfaced as a confusing
`Pad reflect: post-pad (64) exceeds maximum allowed (63)` — three levels away from the real
cause. Found by re-reading silero's `utils_vad.py:130-137` rather than by reading my own
code.

I also chased a false lead: I tried to time a 1536-sample chunk as a "faster path", and
the model rejected it. The cap is real — `max_frames=512` at `model.py:7`,
`num_samples = 512` at `utils_vad.py:112`. Chunk size is not a performance lever.

## What Jev was good and bad at

Full detail in [JEV-AS-INSTRUMENT.md](JEV-AS-INSTRUMENT.md). The short version:

**Good:** a failure-framed battery of narrow Nouls combined with `max` **survived an
adversarial input that defeated a holistic judge by a factor of 10.4** (0.940 vs 0.090 on a
hum that insisted it was an emergency — the single stored run in
`jev_alwayson_results.json`). It also screened my four candidates correctly and
independently agreed with my own measurements.

**Bad:** it cannot hear audio, so every "result" is its reading of a description *I* wrote
— it can tell me whether a rule is *sayable*, not whether it is *true*. `max` flattened
three candidates into a 0.92/0.96/0.96 tie that `mean` separated cleanly
(0.44/0.62/0.84) — so the combinator has to match the question. And on the derive/count
probe its free-form `Score` came back at **1.4–1.9 out of 100** on a scale whose bottom
means "no such quantity can be stated": **it refused to fabricate**, while the `Choice`
version I gave was "right" 3/3 — because *I* had done the arithmetic when writing the
options. That trap is the reason the rule is "never ask it to derive; ask it to choose."

## The part I still do not know

- **My non-speech signals are synthetic and clean.** The **93.0×** mean separation
  (0.7530 worst speech mean ÷ 0.0081 worst non-speech mean) is measured on
  3 real speech clips and 6 clean synthetic signals. Real rooms have reverb, competing
  speech and music beds this battery does not model. **The recommendation is gated on
  re-measuring on the real-room fixture lane** with the actual fan that is currently
  breaking us.

  > **CORRECTION (2026-10-19).** This said "The 28× separation". **No derivation in
  > `hotpath_measurements.json` yields 28.** From
  > `silero_v5_scores.current_v5_opset16`: worst speech mean **0.7530** (speech_jane),
  > worst speech max **1.0000**; worst non-speech mean **0.0081** (white noise), worst
  > non-speech max **0.1736** (music). Derivable: **93.0× on the mean**
  > (0.7530 / 0.0081) and **5.76× on the max** (1.0000 / 0.1736); the parenthetical
  > pair (0.7530 / 0.1736) is **4.34×**. **The separation is not wrong — the file
  > supports a LARGER mean gap than 28×.** `NOTES-AUDIT.md` F10.
- **Everything is a 24-core x86 desktop CPU.** At a pessimistic 8× per-core penalty the
  watcher needs 313% of a core — **saturated** — while silero needs 1.8%. That 313% is the
  **non-speech** projection (39.122 ms mean, 48.387 ms worst case); on speech the same 8×
  gives 176–289% of a core (221% at the 27.578 ms speech mean), so it is saturated on both
  populations. So on a Pi-class target, profiling the YIN pitch tracker becomes the top
  priority, not a nice-to-have.
- **silero's own benchmark is a wiki page, not a repo artifact**, so their 189 μs is not
  reproducible from source.
- I did not measure pyannote at all — I concluded it is structurally unfit before spending
  a CPU benchmark on it.

## Files

| File | What |
|---|---|
| [SYNTHESIS.md](SYNTHESIS.md) | the ranked answer, the comparison table, the 5 stealable mechanisms |
| [SILERO-VAD.md](SILERO-VAD.md) | the candidate we are adopting, with the hum battery |
| [OPENWAKEWORD.md](OPENWAKEWORD.md) | rejected on licensing; what to copy and why |
| [PYANNOTE.md](PYANNOTE.md) | rejected; the licence question answered properly |
| [JEV-AS-INSTRUMENT.md](JEV-AS-INSTRUMENT.md) | the decision-model writeup, including my failures |
| `measure_hotpath.py` | the measurement harness (all numbers above come from it) |
| `hotpath_measurements.json` | its raw output |
| `jev_alwayson.py` | the Jev suite |
| `jev_alwayson_results.json` | its raw output |