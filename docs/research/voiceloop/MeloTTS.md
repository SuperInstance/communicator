# MeloTTS — multilingual TTS

- Repo: `myshell-ai/MeloTTS`, branch **`main`**
- License: **MIT** — verified twice. `LICENSE:1-3` is the standard MIT grant
  ("Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software... `LICENSE`), GitHub `licenseInfo` reports `MIT`, and
  `README.md:58` states it plainly:
  > This library is under MIT License, which means it is free for both
  > commercial and non-commercial use.
- 7,679 stars, **last push 2024-12-24** — ~22 months stale as of 2026-10-18.
  Not archived, but not developed either. Treat as a frozen reference, not a
  dependency to track.

## Verdict up front

**The licensing question is a clean yes. The streaming question is a clean no,
and the interesting content is a single mechanism: three scalar prosody knobs
that are not orthogonal and not real-time safe.**

For a peripheral we do not want the speech. We want the *shape*.

---

## 1. There is no streaming synthesis

I checked this properly rather than trusting the word "streaming":

```
$ grep -rn "yield\|stream" api.py models.py
(no matches)
```

Zero occurrences of `yield` or `stream` in either file. The public API is
`tts_to_file` (`api.py:83`), which:

1. splits the text into sentences (`split_sentences_into_pieces`, `:75-80`),
2. synthesizes **each sentence to completion** in a loop (`:96-125`),
3. concatenates the numpy arrays with a fixed gap (`:66-72`):

```python
audio_segments += segment_data.reshape(-1).tolist()
audio_segments += [0] * int((sr * 0.05) / speed)
```

So the granularity is *sentence*, and even within a sentence you wait for the
whole waveform. There is no chunk callback, no partial result, no
vocoder-streaming. The 50 ms gap is inter-sentence padding, not streaming.

**This is the opposite of what I expected and it matters:** both TTS repos in
this lane are whole-utterance synthesizers. Neither offers a technique to lift
for incremental audio. MeloTTS's `README.md:25` claim of "Fast enough for CPU
real-time inference" is an RTF statement (faster than playback), not a
time-to-first-audio statement, and no figure is given anywhere.

---

## 2. The stealable thing: three prosody scalars

`api.py:83`:

```python
def tts_to_file(self, text, speaker_id, output_path=None,
                sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, ...)
```

passed through to `model.infer(...)` (`api.py:110-122`), whose own defaults
(`models.py:966-977`) are `noise_scale=0.667`, `length_scale=1`,
`noise_scale_w=0.8`.

Read from the VITS/VITS2 machinery these mean:

| knob | what it actually controls |
|---|---|
| **`noise_scale`** (0.6–0.667) | scale on the noise injected in the flow that generates the hidden representation. **How much prosody varies between takes of the same text.** Low → flat, repeatable. High → expressive, variable. |
| **`noise_scale_w`** (0.8) | scale on the noise in the **duration** predictor. **How much timing jitter** varies between takes. |
| **`sdp_ratio`** (0.2) | scale on the **Stochastic Duration Predictor**. At 0, durations are the deterministic argmax prediction; raising it makes each token's duration a stochastic draw. |

Supporting source: `StochasticDurationPredictor.forward(..., noise_scale=1.0)`
at `models.py:206`, built from `n_flows` alternating `ConvFlow` + `Flip`
(`models.py:177-186`) — a normalising-flow model of the duration distribution.

**Why this matters for a peripheral.** These are exactly the degrees of freedom
we want over MIDI CC output, and they are named in a way that transfers
directly:

- `noise_scale` → **how much the output varies event to event** (our
  "expressiveness" / jitter CC)
- `noise_scale_w` → **how much the timing varies** (our rhythmic instability)
- `sdp_ratio` → **how stochastic vs deterministic the timing is** — at 0 the
  system is metronomic and predictable, which is a legitimate musical mode, not
  just a degenerate one.

That last one is the interesting insight. `sdp_ratio=0` gives a *deterministic
prosody* mode — the model still predicts durations, it just stops sampling them.
A peripheral that wants to feel mechanical and intentional rather than random
has a single scalar for it, and it is the same scalar a TTS model exposes for
free.

### The blunt caveat (measured, not assumed)

I asked Jev whether these three knobs are separable and real-time safe. **It
could not answer the question** — it scored all three *nearly identically* on the
substantive axes (spread across knobs: `timing` **0.000**, `prosody` **0.020**;
see [JEV-AS-INSTRUMENT.md](JEV-AS-INSTRUMENT.md) §E6). What it did say, and
what I accept, is that **`realtime_safe` ≈ 0.38 for all three**.

> **CORRECTION (2026-10-19).** This said the three knobs scored *identically*,
> with "spread across knobs: `timing` **0.000**, `prosody` **0.000**".
> `docs/research/voiceloop/jev_voiceloop_results.json` gives a `prosody` spread
> of **0.020** (and 0.010 on `orthogonal` and `realtime_safe`) — near-identical
> rather than identical. `NOTES-AUDIT-2.md` F5.

That is consistent with the source: each knob is a per-utterance draw through a
flow with internal state, set once at `infer()` entry. You could re-draw them
between utterances, but **not at audio-frame rate within one**, and the
`noise_scale=0.667` default implies you *should* randomise them per utterance —
so consecutive utterances will differ unless you deliberately pin them.

**So: the knobs are a good model of musical parameters and a poor real-time
modulator.** Reuse the *vocabulary*, not the runtime.

---

## 3. Language coverage and cost

11 languages/accent variants in the README table (`README.md:10-21`): EN-US,
EN-BR, EN-IN (Indian), EN-AU, EN-Default, ES, FR, ZH (mixed EN), JP, KR. ZH
supports code-switched Chinese/English (`:24`).

The cost of that coverage is visible in `requirements.txt`: `txtsplit`, plus a
per-language front-end — `unidic`/`unidic_lite`/`fugashi`/`mecab-python3`/
`pykakasi` (JA/KO), `pypinyin`/`cn2an`/`jieba` (ZH), `g2p_en`/`eng_to_ipa`
(EN), `gruut[de,es,fr]` (EU), `g2pkk` (KO), `num2words`, `transformers==4.27.4`.
A multilingual TTS front-end is a multilingual NLP dependency tree.

---

## 4. Environment feasibility

`api.py:4-9` imports at module scope:

```python
import torch, librosa, soundfile, torchaudio, numpy, torch.nn
```

| library | present in our env | required |
|---|---|---|
| `numpy` | **yes** | yes |
| `torch` | **no** | **yes, hard** |
| `torchaudio` | **no** | **yes, hard** |
| `librosa` | **no** | **yes, hard** (`librosa==0.9.1` pinned) |
| `soundfile` | **no** | yes |
| `transformers==4.27.4` | **no** | yes (BERT features) |

**numpy-only path: not viable.** There is no ONNX export, no numpy inference,
and the import graph fails at line 4 before anything else runs. MeloTTS is
PyTorch-only in practice.

Note `librosa==0.9.1` and `transformers==4.27.4` are **exact pins** — this is a
2023-era dependency set. Combined with the 2024-12 last push, adopting MeloTTS
means adopting a frozen dependency closure.

---

## 5. Provenance

`README.md:62`: derived from Coqui `TTS`, VITS, VITS2, Bert-VITS2. Authors at
Tsinghua and MIT/MyShell (`README.md:40-44`). MIT license is Coqui's to grant
for the inherited portions, which is consistent and unremarkable.

## Bottom line

**License: use it freely, MIT, including commercially.**
**Streaming: there is none — do not look here for incremental synthesis.**
**Steal the three-knob prosody vocabulary (`noise_scale` / `noise_scale_w` /
`sdp_ratio`), and note that `sdp_ratio=0` is a useful deterministic mode.**
**Runtime: no numpy/ONNX path exists; adopting it means adopting PyTorch.**