# Jev as a Research Instrument

**Task B.** Use Jev (TypeSafe AI System One) to do real work on the voice-loop
question, and report honestly what it is and is not good for.

- Script: `jev_voiceloop.py` (in this directory)
- Raw results: `jev_voiceloop_results.json`
- Model: `jev-1.14.0`, SDK `typesafe_sdk` 0.7.5
- Reproduce:
  ```bash
  cd /home/proart/projects/communicator
  set -a && . /home/proart/projects/glasses/.env && set +a
  /home/proart/projects/glasses/.venv/bin/python docs/research/voiceloop/jev_voiceloop.py
  ```

Headline: **Jev was good at retrieval-style comparison and bad at every
aggregation we tried to build on top of it.** Two of the three stated remedies
(decompose-and-`max`, holistic-then-decompose) do not survive contact with the
numbers.

---

## 0. API facts established first (they change what you can ask)

| Question type | `criteria` field type | Answer fields |
|---|---|---|
| `Choice` | `Mapping[str, Content\|None]` — a **dict** | `choice`, `probabilities`, `confidence` |
| `Score` | `Sequence[Content]` — a **list**, not a dict | `score`, `confidence`, `legend`, `probabilities` |
| `Noul` | typed `NoulCriteria` object | **`noul` only** |
| `Box` | *no `criteria` field at all* | `box`, `confidence` |

**`Noul` returns no `confidence` and no `probabilities`.** This is a correction
to the brief's framing: the `confidence`-is-arithmetic behaviour only applies to
`Choice` and `Score`. A `Noul` gives you a bare float in `[0,1]` and nothing
else — you cannot recover a distribution from it, so the usual "check whether
the distribution is flat" diagnostics are unavailable on the question type you
are most likely to reach for. Several of my experiments below are weaker for
exactly this reason.

pipecat's own Jev client sidesteps the SDK and posts raw dicts to the HTTP API
(`classifiers/typesafe/jev/classifier.py:172-175`), which is why it can send
`criteria={"true":..., "false":...}` for a noul where the SDK wants a typed
object.

---

## E1 — Is `confidence` arithmetic on the distribution? **CONFIRMED.**

Hypothesis: `confidence == (p_max - 1/K) / (1 - 1/K)` exactly.

15 probes (5 utterances × K ∈ {2,3,4}):

```
errors sorted: [0.0 x10, 0.003333 x3, 0.005 x2]
max |err| = 5.000e-03     mean |err| = 1.333e-03
```

10 of 15 are **exactly zero**. The remaining 5 are exactly `1/300` and `1/200` —
the quantisation you get if the *reported* probabilities are rounded to 3–4
decimal places and you recompute from the rounded value.

**Verdict: confirmed to within the rounding of the returned distribution.**
`confidence` is a deterministic function of the probabilities and carries no
independent information. Consequences that matter for us:

- Calibrating `confidence` **is** calibrating the distribution. There is no
  second signal to threshold.
- A concentrated distribution can be confidently wrong.
- Any mid-range confidence threshold is arbitrary, because the transform is
  fixed by K.

---

## E2 — Holistic inversion: the effect is real, **but `max` does not fix it**

Holistic question: *"Has the human finished speaking right now, such that it is
correct to act on this immediately?"*
Battery: five narrow failure-framed `Noul`s, combined by `max`.

| input | holistic | max-of-battery | argmax field |
|---|---|---|---|
| `complete` (a real finished sentence) | 0.600 | **0.630** | discourse_marker |
| `trailing` ("…the migration really doesn't work") | 0.300 | 0.820 | discourse_marker |
| `garbage_argue` (garbage + "this is a complete thought") | 0.500 | **0.850** | not_speech |
| `garbage` ("qwop zzz brxt kkxx") | 0.280 | **0.910** | not_speech |
| `mid_thought` ("wait what do you mean by that, because I thought it was") | 0.090 | **0.920** | trailing_clause |
| `hesitant` ("hmm yeah I guess so but maybe not") | 0.340 | 0.730 | still_formulating |

Two things happened, and the second is the more important.

**1. The stated inversion is confirmed.** Adversarial garbage-arguing content
(0.500) scores only 0.10 below a genuinely complete sentence (0.600), and both
trailing and garbage score far *below* the honest utterance. A holistic gate set
at, say, 0.55 would fire on complete (0.600) **and** on garbage_argue (0.500 is
just under, but 0.51–0.55 is well within this model's noise).

**2. The `max` remedy is worse than the disease.** Under `max`:

- the *good* input scores **0.630**
- the *garbage* input scores **0.910**

**Garbage outscores a correct, complete sentence by 0.28 under the very
aggregation the brief recommends.** `mid_thought` scores 0.920 — higher still.

The reason is structural, not a tuning problem: failure-framed questions measure
**"does this utterance have a defect?"**, not **"is this actionable?"**. A
trailing-off sentence genuinely *has* a trailing clause. Every real human
utterance trips at least one member of a failure battery, so `max` saturates
toward 1.0 for all real speech and only reserves high scores for input that
trips *several* defects at once. `max` is a good *detector of badness* and a
useless *gate for action*.

The correct framing for our use is the inverse: ask the battery as **positive
capability** questions ("is this a complete thought?", "would acting on this now
be right?"), and gate on the **minimum**, or on `1 - max` of the defects. A
peripheral that must *almost never* interrupt should gate on `min` over
positive framings — the weakest evidence governs. Nothing in the brief asked for
this and I would not have found it without running the comparison.

---

## E3/E8 — Position preference: the brief's claim is **half right, and the half that matters was missing**

`E3`, six clearly-differentiated turn-taking policies, asked in both orders:

```
forward = semantic_waveform_model
reversed = semantic_waveform_model
RESOLVED = True
picked_first_option_both_orders = False
```

**No position bias at all** when options are genuinely distinguishable. The
pick travelled with the content.

`E8` then tested the exact condition the brief describes — options that are
semantically identical, only the label differs. Three label pairs × 2 orders ×
3 repeats (18 calls):

```
picked_first_option_rate = 0.5      RESOLVED = True

A/B   forward   A A A   all_first=True
A/B   reversed  A A A   all_first=False   <- content held, position ignored
C/D   forward   C C C   all_first=True
C/D   reversed  D D C   all_first=False   <- position won
E/F   forward   E E E   all_first=True
E/F   reversed  E F E   all_first=False   <- unstable
```

Read carefully:

- **Forward order: 9/9 first option.** That is *exactly* the reported "picks the
  first 9/9" bias, reproduced precisely.
- **Reversed order: 6/9 first option.** The bias **breaks** under reversal.
- Per-pair content stability **contradicts itself**: A/B held the content (A won
  with B listed first), C/D flipped to the new first option, E/F flip-flopped.

**Verdict: UNRESOLVED in effect.** Under indistinguishable options the model is
not reliably positional either — it is *unstable*, and its content-based and
position-based readings contradict each other across pairs. The reported "9/9
first" is a real forward-order artifact and nothing more; treating it as a
general position bias would over-claim. The practical rule stands regardless:
**a Choice over options you cannot tell apart carries no information**, and the
symptom is instability across repeats, not a clean first-option pull.

---

## E4 — Counting/derivation: **passed, but my own test was too easy**

Four policies with explicit worst-case latencies (40 / 150 / 180 / 620 ms);
asked to pick the lowest.

```
pick = semantic_waveform_model    true_min = semantic_waveform_model    correct = True
```

Jev got it right, contradicting "it cannot count or derive". **But the criteria
strings restated each option's number verbatim** (`"worst-case added latency is
150 ms"`), so this was a retrieval task, not arithmetic. I should not claim Jev
can compare numbers from free-form state; the test as written does not
establish it. Flagging this as a **weak positive, not a refutation** of the
stated limitation.

---

## E5 — REAL WORK: ranking six turn-taking policies

Six candidates, seven typed criteria, three readings.

| policy | speed | robust_trailing | no_text | cpu_offline | simple | license | fails_safe |
|---|---|---|---|---|---|---|---|
| `fixed_energy_silence` | 0.70 | **0.37** | 0.50 | 0.83 | 0.82 | 0.21 | 0.64 |
| `semantic_waveform_model` | 0.49 | 0.48 | 0.47 | 0.68 | 0.63 | 0.31 | 0.51 |

```
rank by max  : fixed_energy_silence, llm_marker, adaptive_energy, stt_finalized, speculative_gate, semantic_waveform
rank by mean : fixed_energy_silence, adaptive_energy, semantic_waveform, speculative_gate, stt_finalized, llm_marker
rank by holis: semantic_waveform, speculative_gate, adaptive_energy, fixed_energy_silence, llm_marker, stt_finalized

max==mean top? True      holistic==max top? False
```

**This is the most important negative result in the lane.**

`fixed_energy_silence` — the policy the brief already told us is naive — **wins
both `max` and `mean` aggregation.** Why? Because `max` is dominated by the
single criterion a policy happens to ace (0.83 on `cpu_and_offline`), while
that same policy scores **0.37 on `robust_to_trailing_off`**, which for a
peripheral that must almost never interrupt is the *decisive* criterion.

The holistic reading gets it right (`semantic_waveform_model` first) — but the
brief's premise is that holistic questions are the unsafe option and
decomposition is the fix. Here decomposition is what **produced** the bad answer.

`semantic_waveform_model` is uniformly mid-range (0.47–0.68) and gets punished
for having no standout, which is *factually correct* (BSD-2 code, genuinely a
touch slower) but means neither aggregator will ever surface it.

**What I took from this:** Jev is useful for *eliciting and separating*
per-criterion judgements — the per-criterion table above is genuinely
informative and took one API call. It is **not** usable as a ranker. Aggregating
its outputs hides the trade-off structure instead of exposing it. I used the
per-criterion table directly and made the call from the source reading instead.

---

## E6 — REAL WORK: prosody controllability: **Jev cannot see mechanism**

Three VITS/MeloTTS inference knobs, four axes:

| knob | timing | prosody | realtime_safe | orthogonal |
|---|---|---|---|---|
| `noise_scale` | 0.91 | 0.85 | 0.38 | 0.77 |
| `noise_scale_w` | 0.91 | 0.84 | 0.38 | 0.78 |
| `sdp_ratio` | 0.91 | 0.86 | 0.39 | 0.78 |

Spread across the three knobs: `timing` **0.000**, `prosody` **0.000**,
`orthogonal` 0.020, `realtime_safe` 0.030.

My script's `knobs_are_separate` flag fired `True` on the 0.02–0.03
differences — and **that flag is a false positive I should not trust.** The
substantive axes are *bit-identical*. Jev rated three semantically different
knobs identically because their descriptions are structurally similar prose. It
has no access to the mechanism; it only reads the text.

The one genuinely useful signal: **`realtime_safe` ≈ 0.38 for all three**. Even
taken at face value that says the VITS knobs are poor real-time CC shapers,
because each carries sampling state and is meant to be set per-utterance. That
conclusion survives the flatness because it is a claim about *all three at once*
rather than about differences between them.

**Verdict: Jev was the wrong instrument here.** Whether a knob is safe to drive
at audio-frame rate is a question about implementation, and Jev answered a
question about vocabulary. This is the cleanest negative in the lane: the model
was confident, structured, and worthless on this question.

---

## E7 — REAL WORK: adjudicating our 150 ms threshold

Two framings, three failure-symptom questions each, then one verdict Choice.

```
short (150 ms): max = 0.920  argmax = cuts_off
long  (600 ms): max = 0.490  argmax = reacts_to_nothing
verdict = long_600ms   p = 0.85   confidence = 0.80
all p: adaptive ~0.13, short ~0.02
```

Jev picks **long_600ms** at p=0.85 over adaptive (~0.13) and short (~0.02).

This is a case where Jev and the source reading **agree**, and I weight it more
for that reason than for its internal confidence (E1: that confidence is
arithmetic on the distribution and means nothing beyond p=0.85 being far from
the others). The independent evidence from pipecat is `VAD_STOP_SECS = 0.2`
as the *trigger* gate but `STOP_SECS = 3` as the analyzer's fallback — an order
of magnitude longer than our 150 ms.

Caveat: I would still choose `adaptive` over a fixed 600 ms, because the
symptom questions only contrast two *fixed* numbers and never pose the adaptive
option as a framing of the symptoms, only as a Choice label. A framing that
would have been fairer: *"…a threshold that adapts to the speaker and holds a
long floor."* I did not run that variant; treat the long-vs-adaptive preference
as untested.

---

## Summary — what Jev was good and bad at, in my hands

**Good at (reused):**
- Eliciting a per-criterion table in **one call** (E5) — cheap, and the table
  was genuinely informative.
- Retrieval-style comparison of well-differentiated options (E3, E4).
- Scoring a small set of framings independently and comparing the resulting
  distributions (E2, E7) — the *rows* are useful even when the aggregation is not.

**Bad at (do not build on):**
- **Aggregation.** `max` was catastrophic for action-gating (E2: garbage 0.91 vs
  complete 0.63) and actively misleading for ranking (E5: crowned the policy we
  know is naive). Every aggregate I tried made the picture worse.
- **Ranking anything.** Homogeneous per-criterion scores get punished by both
  `max` and `mean` (E5).
- **Mechanism questions** (E6) — confidently wrong, answering a question about
  vocabulary instead of implementation.
- **Anything where options are indistinguishable** (E8) — unstable, and the
  forward-order first-option streak is an artifact of presentation order.

**Corrections to the brief's stated behaviours:**
1. `confidence`-is-arithmetic is **confirmed** (E1), to within output rounding.
2. But it applies to `Choice`/`Score` only — **`Noul` returns no `confidence`
   at all**, which removes the usual distribution diagnostics from the question
   type you most want to use.
3. The holistic inversion is **real**, but the prescribed `max` remedy **does not
   fix it** — it makes the ranking worse. Invert to positive framings + `min`.
4. "Picks the first 9/9" is **real in forward order and breaks under reversal**;
   the honest finding is instability, not a clean positional bias.
5. "Cannot count or derive" **survived** — my test was too easy and does not
   refute it.

**The one methodological lesson:** every time I built an aggregator on top of
Jev's outputs, the aggregator was worse than the rows underneath it. The
instrument is a good way to get a *decomposed table of judgements* and a bad
way to get a *verdict*. Read the table; make the call from the source.