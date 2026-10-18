# Coqui TTS (`coqui-ai/TTS`) — licensing, maintenance, and what actually runs

- Repo: `coqui-ai/TTS`, default branch **`dev`** (not `main`/`master`)
- Code license: **MPL-2.0** — `LICENSE.txt:1-2` reads verbatim
  "Mozilla Public License Version 2.0", GitHub `licenseInfo` reports
  `MPL-2.0`, `README.md:27` carries the MPL 2.0 badge. All three agree.
- 46,132 stars, **last push 2024-08-16** — ~26 months stale as of 2026-10-18.
  Not archived, but dormant.

## Verdict up front — the brief's premise needs correcting

The brief said *"Coqui's license has historically been non-commercial — verify
the CURRENT license, we need to know if we may use it."*

**Half right, and the half that is wrong matters more than the half that is
right.**

The repo code is **MPL-2.0** — weak copyleft, commercially fine. The *model
weights* are licensed **per-model**, recorded in a manifest inside the repo. I
parsed all of them:

**95 models carry a `license` field:**

| license | count | commercial? |
|---|---|---|
| apache 2.0 | 34 | **yes** |
| bsd-3-clause | 24 | **yes** |
| MIT | 11 | **yes** |
| CC-BY-SA 4.0 | 8 | yes, share-alike |
| MPL | 7 | yes |
| CC-BY-4.0 | 3 | yes |
| **CPML** | **2** | **NO — non-commercial** |
| **CC BY-NC-ND 4.0** | **2** | **NO — non-commercial + no-derivatives** |
| Apache 2.0 | 2 | yes |
| openrail | 1 | restricted (OpenRAIL) |
| custom (jenny) | 1 | see dataset |

`tos_required` is `True` for exactly the 2 CPML models and `False` for the other
93.

**Only 2 of 95 models are the non-commercial CPML everyone remembers** — and
they are the flagship:

```
CPML            tos=True   tts_models/multilingual/multi-dataset/xtts_v2
CPML            tos=True   tts_models/multilingual/multi-dataset/xtts_v1.1
CC BY-NC-ND 4.0           tts_models/multilingual/multi-dataset/your_tts
CC BY-NC-ND 4.0           tts_models/en/vctk/fast_pitch
custom                    tts_models/en/jenny/jenny
openrail                  tts_models/fa/custom/vits-female
```

So: **the classic English Coqui models (VITS on LJSpeech, Tacotron, the
multi-speaker sets) are Apache-2.0 / BSD-3 / MIT and free to use
commercially. XTTS-v2 — the famous one — is not.**

The mechanism is `TTS/utils/manage.py:39-48`, which maps licence strings to
URLs:

```python
LICENSE_URLS = {
    "cc by-nc-nd 4.0": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "mit":  "https://choosealicense.com/licenses/mit/",
    "apache 2.0": "https://choosealicense.com/licenses/apache-2.0/",
    "cc-by-sa 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cpml": "https://tts-hub.github.io/cpml",
}
```

`manage.py:221-222` has a `print_model_license(model_item)` — the project
prints the licence when you download, which is the right design and the reason
this is knowable at all.

### What CPML actually says

`https://tts-hub.github.io/cpml`, verbatim:

> This license allows only **non-commercial use** of a machine learning model
> **and its outputs**.

With the definition of non-commercial purpose including:

> Use by commercial or for-profit entities for testing, evaluation, or
> non-commercial research and development. **Use of the model to train other
> models for commercial use is not a non-commercial purpose.**

**"and its outputs" is the clause to watch for us.** It is not only training and
distribution that is restricted — the *generated audio* inherits the
restriction. That matters differently for a peripheral than for a product: we
would not be shipping generated speech, but if we ever train on Coqui output the
restriction follows the training.

For a local, non-commercial, personal peripheral, CPML models are *permitted*.
For anything commercial, they are not, and no amount of MPL-2.0 on the code
changes that. **Check `TTS/.models.json` for the specific model before use —
that is the authoritative per-model answer, not the repo licence.**

---

## Has it moved? **Yes — Idiap maintains the continuation.**

| repo | license | last push | stars |
|---|---|---|---|
| `coqui-ai/TTS` | MPL-2.0 | 2024-08-16 | 46,132 |
| **`idiap/coqui-ai-TTS`** | MPL-2.0 | **2026-10-18** (today) | 2,337 |

`idiap/coqui-ai-TTS` is the same project (identical description "🐸💬 - a deep
learning toolkit for Text-to-Speech, battle-tested in research and production",
same `dev` default branch, same MPL-2.0 licence) and it is **actively
developed**. This is the one to use. Coqui AI the company is gone; the toolkit
lived on at Idiap Research Institute.

Note both README badges still link `github.com/idiap/coqui-ai-TTS` in the
model links (`idiap_README.md:59`) — the fork relationship is acknowledged.

---

## 1. Streaming: also no

`TTS/utils/synthesizer.py:257-296`:

```python
def tts(self, text=None, speaker_name=None, language=None, speaker_wav=None,
        ..., split_sentences: bool = True, ...):
    sens = [text]
    if split_sentences:
        sens = self.split_into_sentences(text)
```

Same shape as MeloTTS: sentence-split, then synthesize each to completion. No
`stream()` method, no partial-audio callback, no chunked vocoder in the Python
API. The server (`TTS/server/server.py`) does stream over websockets, but that
is chunked *transport*, not incremental *synthesis* — the model still produces
whole segments.

**Across all three repos in this lane, incremental audio synthesis is absent.**
If we ever need it, the field to look at is neural vocoder streaming
(HiFi-GAN chunked inference, Vocos) rather than these text-to-waveform stacks.

---

## 2. Dependency / environment feasibility

| library | present in our env | status |
|---|---|---|
| `numpy` | **yes** | — |
| `torch` / `torchaudio` | **no** | required |
| `librosa` / `scipy` | **no** | required |
| `soundfile` | **no** | required |

**numpy/ONNX-only path: not viable for the Python package.** There is no ONNX
export for the model zoo and the import graph is PyTorch-rooted.

**The one partial exception is worth naming:** Coqui ships **ONNX models for
inference in the browser** (for XTTS and others). Those are separately
distributed artefacts under their own terms — not in this repo, not covered by
the `.models.json` manifest I parsed, and I did **not** verify their licence or
download them. Flagged as unverified below.

For us the practical consequence is the same as MeloTTS: adopting a Coqui model
means adopting PyTorch.

---

## 3. What is actually worth taking from this repo

Honestly, little for the peripheral problem — and that is a legitimate finding
rather than a failure to look hard. Two things:

1. **The per-model licence manifest pattern** (`TTS/.models.json` +
   `manage.py:print_model_license`). We are going to depend on third-party
   model weights; making the licence a *machine-readable field checked at
   download time* is a pattern worth copying directly into our own model
   fetcher. It costs one field and it is the only reason the Coqui licensing
   question is answerable at all.

2. **The explicit statement that generated *outputs* inherit the licence.** The
   sentence "non-commercial use of a machine learning model **and its outputs**"
   in CPML is broader than most people assume, and it is a real constraint for a
   system that trains on generated data. Worth writing into our own licence
   checklist even though we may never use Coqui weights.

What I did **not** find: any streaming technique, any prosody-controllability
API beyond the VITS-family knobs already documented in
[MeloTTS.md](MeloTTS.md), and any latency benchmark in the repo.

---

## 4. Unverified / caveats

- **Not verified:** licences of the ONNX browser-inference artefacts (separate
  download channel, outside `.models.json`).
- **Not verified:** whether the `.models.json` licence fields have kept pace with
  the actual files on the HF hub at `tts-hub/*`. The manifest is the repo's own
  statement of intent; I treated it as authoritative because
  `print_model_license` exists, but I did not diff 95 entries against HuggingFace.
- **Not verified:** `openrail` terms — I recorded it as restricted but did not
  read the OpenRAIL variant, and it has use restrictions plus use-based
  redistribution conditions.
- **Not read:** `jenny`'s custom dataset licence
  (`https://github.com/dioco-group/jenny-tts-dataset#important`), which the
  manifest itself flags as an "important" custom case.
- **Not inspected:** CI configuration, in either repo.

## Bottom line

**Code: MPL-2.0, commercially usable. Models: check `.models.json` per model —
34 Apache-2.0 + 24 BSD-3 + 11 MIT + 7 MPL + 8 CC-BY-SA + 3 CC-BY + 2 Apache-2.0
are all fine; XTTS-v2/v1.1 (CPML) and your_tts/fast_pitch (CC BY-NC-ND) are
non-commercial, and CPML extends to outputs.**
**Maintenance: use `idiap/coqui-ai-TTS`, which is alive today; the original is
dormant.**
**Streaming: none. Prosody knobs: same VITS family as MeloTTS.**
**Runtime: PyTorch required; no numpy/ONNX path in the Python package.**