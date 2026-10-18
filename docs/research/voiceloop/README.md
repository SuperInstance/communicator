# Voice-loop research lane — learning journey

Research log for `pipecat`, `MeloTTS`, and `Coqui TTS`, plus the use of **Jev**
(TypeSafe System One) as a research instrument.

**Read `SYNTHESIS.md` first if you want the answer.** This file is the journey,
including the parts where I was wrong.

Deliverables in this directory:

| file | what |
|---|---|
| `SYNTHESIS.md` | **the ranked answer** on end-of-turn + interruption, top 5 stealable mechanisms, licences |
| `pipecat.md` | pipecat turn-taking, interruption, classifiers, latency metrics |
| `MeloTTS.md` | streaming (none), prosody knobs, MIT, PyTorch-only |
| `coqui-tts.md` | the licence investigation, the Idiap move, model-by-model |
| `JEV-AS-INSTRUMENT.md` | Task B — what Jev is and is not good for |
| `jev_voiceloop.py` + `jev_voiceloop_results.json` | the instrument itself |
| `bench_smartturn.py` + `smartturn_bench.json` | my latency/behaviour benchmark (fetches the model + extractor on demand) |
| `_whisper_features.py` | vendored upstream extractor the benchmark imports |

---

## What I expected going in

Four predictions, all written down before I read source:

1. **pipecat's end-of-turn is VAD-based with a configurable silence knob.** I
   expected a phone bot to threshold energy and move on. This was the *most*
   wrong assumption and the most valuable one to have written down.
2. **TTS repos contain clever streaming synthesis** worth stealing for real-time
   output.
3. **Coqui is unusable** because its licence is non-commercial.
4. **Jev will behave as briefed** — confidence is arithmetic, decompose-and-`max`
   fixes holistic inversion, position preference is real, it cannot count.

---

## Surprise 1 — pipecat's turn-taking is not a VAD knob (the big one)

I went looking for a silence threshold and found a **two-stage detector**:
a cheap VAD decides *whether there is speech to look at*, then a **binary
classifier over the raw waveform** decides *whether the thought is complete* —
**with no transcription anywhere in the loop**.

The abstract signature gave it away (`audio/turn/base_turn_analyzer.py:97`):
`append_audio(self, buffer: bytes, is_speech: bool)` — the VAD boolean arrives
from *outside*. Feeding and deciding are separate calls, so the model only ever
runs after something has already established that speech exists.

Then the number that reorganised the whole lane:

```python
# audio/vad/vad_analyzer.py:27
VAD_STOP_SECS = 0.2
# audio/turn/smart_turn/base_smart_turn.py:27
STOP_SECS = 3
```

**~200 ms to trigger, 3 s to give up.** A ~15× ratio between "start looking"
and "stop looking". Our single 150 ms has been asked to be both. That reframed
the entire task: the problem was never "150 ms is the wrong number", it was
**"150 ms is one number doing two jobs, and the second job is not a threshold
at all."**

**Correction to my own framing.** I had assumed this was a phone-bot framework
whose turn logic I would have to fight to adapt. It is a **peripheral-friendly
architecture that I only had to point the right way** — waveform in, binary
decision out, no transcript. Our peripheral has no transcript and needs none.
pipecat already has the switch (`wait_for_transcript=False`, documented as
putting "transcripts off the latency critical path").

## Surprise 2 — they ship a TypeSafe/Jev classifier

I had assumed our unusual integration shape (probabilities out, no text
generation) would require a bespoke adapter. pipecat ships
`classifiers/typesafe/jev/` as a **first-class backend** alongside
`cloudflare/clef`, `openai/decisions`, `llm`, and `jev`, behind one interface:

```python
class YesNoQuestion(BaseModel):  instructions; yes; no
class ChoiceQuestion(BaseModel): instructions; options: dict[str, str|...]
class ScoreQuestion(BaseModel):  instructions; levels: list
```

Typed decision models are a **vendor category**, not a workaround. And the
interface pushes the threshold back to the caller — "compare `probability` with
a threshold of their own" (`base_classifier.py:144-151`).

One detail I liked: the classifier is deliberately *not* a pipeline stage. "It
does not sit in a pipeline and no frames flow into it"
(`base_classifier.py:10`). Two shapes, cleanly separated — classifiers for
decisions, frame processors for audio.

## Surprise 3 — a silence subtraction bug they documented

`turn_analyzer_user_turn_stop_strategy.py:236`:

```python
stt_deadline = frame.timestamp - frame.stop_secs + self._stt_timeout
```

The VAD only *reports* "stopped speaking" `stop_secs` after the speech actually
stopped. A naive deadline double-counts the silence window. They subtract it,
and at `:247-262` they warn at runtime if `stop_secs` drifts from the tuned
default — because their latency numbers are conditional on it.

**This is a bug I would have shipped.** The moment we add any wait after the
energy gate — a second classifier stage, a hysteresis check — we pay the silence
window twice unless we do the same subtraction. Cheap to inherit, expensive to
debug later.

## Surprise 4 — the failure direction is backwards for us

```python
# base_smart_turn.py:267-271
except SmartTurnTimeoutException:
    state = EndOfTurnState.COMPLETE
```

They fail **open** — a model timeout means "act". Defensible for a phone bot,
where silence is expensive. For a peripheral it is exactly backwards: an unwanted
musical gesture is intrusive and cannot be recalled. This is a one-line
inversion and it is the most important adaptation in the synthesis.

## Surprise 5 — no streaming synthesis anywhere

Prediction 2 was wrong, and it was wrong in all three repos. I checked rather
than trusting the word "streaming":

```
$ grep -rn "yield\|stream" melo/api.py melo/models.py
(no matches)
```

MeloTTS synthesizes each sentence to completion and concatenates with a 50 ms
gap (`api.py:66-72`). Coqui sentence-splits then does the same
(`synthesizer.py:257-296`). Coqui's *server* streams over websockets, but that
is chunked transport, not incremental synthesis.

**No repo in this lane offers a technique to lift for incremental audio.** If
we ever need it, the field is neural vocoders (HiFi-GAN chunked inference,
Vocos), not text-to-waveform stacks. Recording this as a negative so nobody
spends another day in these repos looking.

## Surprise 6 — Coqui is not the licence I was told it was

Prediction 3 was half wrong, and **the half that was wrong mattered more**.

The code is **MPL-2.0**, not non-commercial — verified three ways
(`LICENSE.txt:1-2` verbatim, GitHub `licenseInfo`, README badge). Fine
commercially.

But weights are licensed **per-model**, recorded in a manifest *inside the
repo*: `TTS/.models.json`. I parsed all 95 entries:

| licence | count |
|---|---|
| apache 2.0 | 34 |
| bsd-3-clause | 24 |
| MIT | 11 |
| CC-BY-SA 4.0 | 8 |
| MPL | 7 |
| CC-BY-4.0 | 3 |
| **CPML (non-commercial)** | **2** |
| **CC BY-NC-ND 4.0** | **2** |
| openrail / custom | 2 |

**Only 2 of 95 are the non-commercial CPML everyone remembers** — and they are
the flagship XTTS v1.1/v2. The classic English models are Apache-2.0/BSD-3/MIT.
`tos_required` is `True` for exactly those 2 and `False` for the other 93.

The clause I would have missed without reading the actual licence rather than
the badge: CPML restricts "non-commercial use of a machine learning model **and
its outputs**". Not just training and distribution — the *generated audio*
inherits it.

And the project has **moved**: `idiap/coqui-ai-TTS` was pushed **today**
(2026-10-18) with the same `dev` branch and identical description, while
`coqui-ai/TTS` has been dormant since 2024-08-16. Idiap is the live
continuation. The company is gone; the toolkit is not.

## Surprise 7 — the model says "finished" about silence

I ran the detector rather than only reading about it
(`bench_smartturn.py`). Measured here across two runs: **6.5–7.8 ms features +
61–73 ms inference = 69–79 ms total** (p95 to 99 ms), single-threaded ONNX,
model **8,679,182 bytes** shipped inside the pip package. Behaviour numbers were
bit-identical across runs; only the timings varied, by ~15% on an idle machine.

The behaviour result is the one I keep coming back to:

| input | P(complete) |
|---|---|
| speech-like, 4 syllables | 0.0495 |
| digital silence | **0.987** |
| white noise | 0.9619 |
| 220 Hz tone | 0.9574 |

**The model is not a silence detector.** Fed silence it is near-maximally
confident the turn is over. That is exactly why the architecture is VAD-gated —
and it means anyone lifting this model without the gate gets a system that
agrees with itself constantly.

**I had to build the benchmark because there was nothing to find.** pipecat
publishes no latency figures anywhere; the README says "Ultra-low latency"
(`README.md:29`) and stops. Per the brief's own instruction — *find the
benchmark, not the summary* — the benchmark did not exist, so I wrote one.

---

## Prediction 4 — Jev: right about the arithmetic, wrong about the remedy

Full detail in `JEV-AS-INSTRUMENT.md`. The short version:

**Confirmed:** `confidence == (p_max − 1/K)/(1 − 1/K)`. 10 of 15 probes were
*exactly* zero; the other 5 were exactly `1/300` or `1/200` — pure output
rounding.

**Corrected:** it applies to `Choice`/`Score` only. **`Noul` returns no
`confidence` and no `probabilities` at all** (`NoulAnswer` has exactly one
field, `noul`) — so the distribution diagnostics that would let me check whether
a question is flat are unavailable on the question type I most wanted to use.
Several of my experiments are weaker for this reason.

**The finding that inverted me:** the brief says decompose and combine with
`max`. I tested it on the actual end-of-turn problem:

| input | holistic | max-of-battery |
|---|---|---|
| complete sentence | 0.600 | **0.630** |
| trailing off | 0.300 | 0.820 |
| **garbage** | 0.280 | **0.910** |
| mid-thought | 0.090 | 0.920 |

**`max` makes it worse.** Garbage outscores a correct complete sentence by 0.28
under the recommended aggregation. The reason is structural:
failure-framed questions measure *"does this have a defect?"*, not *"is this
actionable?"* — and every real utterance has some defect, so `max` saturates on
good input and only reserves 1.0 for input that trips several defects at once.

The inversion that worked: ask **positive capability** questions and gate on
**`min`**. A peripheral that must almost never interrupt should be governed by
the weakest evidence, not the strongest.

**And in the policy ranking** the same thing happened again: `max` crowned
`fixed_energy_silence` — the policy the brief already told us was naive — with
`robust_to_trailing_off = 0.37`, promoted by a `cpu_and_offline = 0.83`. The
holistic question, which the brief says is the *unsafe* one, got it right.

**Position preference: half right.** With six *distinguishable* policies there
was **no** position bias at all — the pick travelled with the content in both
orders. With genuinely indistinguishable options, forward order picked the
first option **9/9** (reproducing the reported bias exactly) but reversed order
only 6/9, and the per-pair readings contradicted each other: one pair held the
content, one flipped to position, one flip-flopped. The honest finding is
**instability**, not a clean positional bias.

**The methodological lesson, which is the real result:** *every time I built an
aggregator on top of Jev's outputs, the aggregator was worse than the rows
underneath it.* Jev is a good instrument for getting a decomposed table of
judgements in one call — E5's per-criterion table took one call and was
genuinely informative. It is a bad instrument for producing a verdict.

**And Jev was simply the wrong tool for one question.** Asked whether the VITS
prosody knobs are separable and real-time safe, it scored all three
*identically* — spread across knobs of exactly **0.000** on both substantive
axes — because their descriptions are structurally similar prose. It has no
access to the mechanism, only the vocabulary. Confident, well-structured, and
worthless. My script's own `knobs_are_separate` flag fired `True` on a 0.02
difference; I did not trust my own metric.

---

## Where I was wrong, collected

1. **Pipecat's turn-taking is not VAD-with-a-knob.** It is a waveform
   classifier with no transcript. I expected a phone bot; found a peripheral
   architecture.
2. **There is no streaming synthesis to steal** from either TTS repo. Zero
   `yield`/`stream` in MeloTTS.
3. **Coqui is not "non-commercial."** MPL-2.0 code; 93 of 95 models permissive;
   the non-commercial ones are XTTS and two CC BY-NC-ND models. Checked the
   manifest instead of the badge.
4. **`max` is not the fix for holistic inversion** — it is worse than the
   holistic question on the actual end-of-turn data.
5. **`Noul` has no `confidence`**, so the brief's arithmetic claim does not
   cover the question type I most wanted to use.
6. **"Picks the first 9/9" is order-specific.** Real forward, broken reversed.
7. **"Cannot count" survived** — my test was too easy (the criteria restated
   each number, making it retrieval, not derivation) and does not refute it.
8. **My own `knobs_are_separate` metric was a false positive**, fooled by a
   0.02 difference on a flat field.

---

## Honest gaps

- **My speech-like test signal is a synthetic formant stack, not real speech.**
  The 0.05-vs-0.90 discrimination is directionally right and justifies the
  architecture; the model's real accuracy is unmeasured.
- **Latency is one machine, single-threaded, two runs** (69–79 ms total, ±15%
  run-to-run on an idle box). There is no upstream number to check it against,
  because none is published.
- **The 600 ms adaptive floor is my synthesis.** pipecat's actual numbers are
  200 ms trigger / 3 s cap. The adaptive floor is my reasoning, not their
  measurement.
- **Not verified:** whether pipecat's 18 turn tests run in CI on every push (I
  did not read `.github/workflows/`); Coqui's ONNX browser artefacts (licensed
  outside `.models.json`); whether `.models.json` licences still match the files
  on HuggingFace; `openrail` terms; jenny's custom dataset licence.
- **Coqui ONNX browser artefacts** and **MeloTTS streaming** were both checked
  and are genuinely absent — those negatives are solid.