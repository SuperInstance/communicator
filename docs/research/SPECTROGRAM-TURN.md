# Spectrogram → vision model: can an image carry the hand-off?

**Status: measured write-lane result. One experiment, real AMI audio, one
vision model (the only one that exists here), all controls run.**

**Headline, stated up front because it decides how to read everything below:**
the only vision-capable model reachable from this machine answers **"a
different speaker takes over" for every image it is shown — including a
uniform gray rectangle with no spectrogram at all.** Accuracy lands exactly on
the trivial baseline, AUC is chance, and a positive control shows the model
cannot read even a gross, visibly obvious cue (which third of the frequency
axis a tone sits in) from a spectrogram image. This is a clean negative for
**our spectrogram route with this reader**, and it is **not** a fair test of
image-native fusion, and it **cannot adjudicate** whether phase is the missing
ingredient. Those two limits were given as constraints in the lane brief and
the evidence below is why they bind.

---

## 1. What was asked

For a labelled pause at a turn boundary on the real two-speaker AMI corpus
(`examples/realtwospeaker/`), render the boundary as a 4.0 s grayscale
spectrogram image and ask a vision model: does the voice after the boundary
belong to the same speaker (SAME) or a different one (CHANGE)? Ground truth is
AMI `speaker_id`, the same labels `experiments/turn_eval.py` uses.

The measured dead ends this was allowed to improve on: silence duration
(pooled AUC 0.529 over 1396 pauses) and pitch (separates voice classes, not
speakers within one). The bar any method must beat: **always-predict-change =
0.6225 pooled**.

## 2. What `experiments/spectrogram_vision.py` actually does (required reading)

Restated because the brief says report it, not assume it. It tests whether the
cloud model **jev-1.14.0** (via `typesafe_sdk`, images enabled by the opt-in
header `TypeSafe-Danger-Zone-Images: true`) can read a spectrogram at all. It
renders hand-written PNGs (no Pillow) of three synthetic tones at 150/300/450
Hz on a 600 Hz axis — i.e. at roughly 25%/50%/75% of the visible frequency
axis — and asks which pitch band the image shows. **It measured 1/3: the
random floor**, unchanged when the numeric answer was added as text
(image ignored), ablation delta vs a blank image ≈ 0.02, and its Box probe
returned a box covering most of the image rather than the harmonic band. Its
most valuable contribution is the documented trap: **without the opt-in header
the SDK silently drops the image bytes and every result comes back plausible
and meaningless** — which is why this lane's harness re-runs the token
preflight and records per-call token counts.

## 3. What is actually available here (measured, not assumed)

| check | result |
|---|---|
| numpy in `/home/proart/projects/glasses/.venv` | present (2.5.4) |
| torch / transformers / timm / open_clip / PIL / scipy | **MISSING** — not added, per the brief |
| onnxruntime | present, but there is no vision model to run in it |
| ollama | present; all four local models (`gemma2:2b`, `llama3.2:1b`, `qwen3:0.6b`, `qwen2.5:0.5b`) **reject image input with HTTP 400** — text-only, every one |
| typesafe `jev-1.14.0` | accepts images; the **only** vision model available |

So "can a vision model do this" could only mean "can *this* model do this".
No heavy dependency was added to the project.

## 4. Design, construction, weaknesses

- **Population** — exactly the one behind 0.6225: watcher pause events on both
  far-field meetings, each labelled by the AMI turns before and after it
  (`turn_eval.pairs_from_audio` logic, re-implemented locally so the pause
  times survive to centre the image). Reproduced here: **1396 pooled pairs,
  base rate 0.6225** — identical to `turn_eval_results.json`.
- **Sample** — 120 boundaries, 60 CHANGE / 60 SAME, seed 11, no two windows
  overlapping by more than half a window. Balanced by design, so the on-sample
  trivial baseline is exactly 0.500; the pooled 0.6225 and an accuracy
  reweighted to the pooled base rate are reported alongside so nothing is
  hidden by the sampling choice. The CHANGE class keeps its natural hard mix:
  39 hand-offs across the two F0 groups (FEO↔MEE, spectrally obvious), 21
  within one group (the measured hard case).
- **Image** — fixed 4.0 s window centred on the pause midpoint, so geometry is
  identical for both classes and no shape or duration cue exists; 0–4000 Hz
  (formants live there); per-window dB normalisation (removes absolute level
  as a trivial cue); 640×480 8-bit grayscale; a 2 px bright column marks the
  boundary. Decoded back and checked: speech structure is present on both
  halves of every window (22–38% of pixels above −37 dB per side).
- **Weaknesses, stated not hidden** — one far-field microphone; real overlap
  means extra voices inside some windows; ~11% of frames are digital-silence
  holes placed by the annotation timeline; 8-bit grayscale compresses dynamic
  range; and the representation **discards phase by construction**.

## 5. Result

120/120 calls succeeded, preflight confirmed the image was on the wire
(324 vs 296 input tokens), no call's token count ever fell to the text-only
level.

| quantity | value |
|---|---|
| accuracy (balanced sample) | **0.5000** — exactly the on-sample trivial baseline |
| accuracy reweighted to pooled base rate | **0.6225** — exactly the pooled baseline (TPR 1.0, TNR 0.0) |
| pooled corpus baseline (1396 pairs) | **0.6225** — not beaten |
| AUC of P(different) | **0.4567**, bootstrap 95% CI [0.356, 0.559] — chance |
| distinct choices emitted | **1** ("different", 120/120) |
| P(different) range across all 120 real images | 0.57 – 0.67 (std 0.017) |

Even the spectrally obvious hand-offs did not move it: mean P(different) =
0.614 (cross-F0-group CHANGE, n=39) vs 0.605 (within-group CHANGE, n=21) vs
**0.614 for SAME (n=60)** — the SAME class sits between the two CHANGE
subclasses. That ordering is noise; there is no signal.

## 6. The two controls — one passed, one reframes everything

**No-information control (passed):** the identical question over a uniform
gray image of identical dimensions, same 120 labels. AUC **exactly 0.500**,
accuracy 0.500 = base rate, std of P(different) = 0.000. So the experiment is
**not rigged** — no artefact, no label leak; given nothing, the model scores
chance. But the sharper fact is the comparison: on a blank gray rectangle the
model answers "different" at 0.620, and on real spectrograms it answers
"different" at 0.612 ± 0.017. **Real and blank are indistinguishable.** The
model is not reading the spectrogram; it is reading the question.

**Positive control (failed, replicating prior art):** three tones rendered at
25%/50%/75% of a 600 Hz axis — a cue that is visibly present in the image, no
phase needed. Accuracy **1/3**, the random floor, the same number
`spectrogram_vision.py` measured before this lane existed. A reader that
cannot tell which third of the axis is bright was never going to read a
within-gender formant difference.

## 7. What this negative does and does not license

1. **It kills our spectrogram route as built**: magnitude-spectrogram image →
   jev-1.14.0 → same/different. Constant answer, baseline accuracy, chance
   AUC, blank-indistinguishable. Do not spend another call on this shape of
   experiment with this model.
2. **It is NOT a fair test of image-native fusion.** OmniVLA trains
   per-sensor projectors and spatialises cues; nothing in that work licenses
   "render it as an image and ask a frozen cloud VLM". Our null is evidence
   about our route, not about the pattern. (This was constraint #1 in the
   brief; the positive control is the proof that it binds — the reader, not
   the idea, failed first.)
3. **Phase: explicitly, as required.** A negative result here is *consistent*
   with "phase is the missing ingredient" — a magnitude spectrogram throws
   phase away, and the route failed — but it is **not evidence for it**, and
   this experiment had **no power to test it**: the only available reader
   fails on phase-free cues that are visibly present (the tone control). The
   failure is upstream of the representation question. Phase remains exactly
   what `CROSS-MODAL-FUSION.md` §10.4 called it: untested, not disproven.
   Note also the accumulating pattern: every magnitude-only route tried on
   this corpus — silence duration (0.529), trivial LTAS statistic (0.33–0.45,
   inverted), and now the full image of the magnitude spectrogram read by a
   VLM — has carried no measurable speaker-identity signal. The live
   hypotheses are now concentrated in two places: **learned embeddings**
   (wav2vec 2.0 SCD / USM-SCD solve this task well above chance) and
   **phase-carrying representations**.

## 8. If this lane is ever reopened

The gate is the positive control, and it is cheap: any candidate reader must
first pass the 3-tone pitch-band test (floor 1/3) — and ideally localise the
band — before it is worth spending boundary-classification calls on. A
reader that passes that gate would finally make the phase question testable:
render the same boundaries with a phase-preserving visualisation (e.g.
reassigned/complex-domain imagery) and compare against the magnitude-only
images with the reader held fixed. Until a reader passes the gate, a phase
experiment would only measure the reader's blindness again.

## Reproduce

```bash
set -a && . /home/proart/projects/glasses/.env && set +a
/home/proart/projects/glasses/.venv/bin/python \
    experiments/spectrogram_turn_eval.py --n-per-class 60 --seed 11
```

Writes `experiments/spectrogram_turn_eval_results.json` (all 240 per-call
rows, probabilities, token counts, latencies). Responses are cached under
`experiments/_spec/turn/` keyed by image hash + model + question, so the
re-run reproduces these numbers without re-paying the API; delete that
directory to force a fresh pass, which may move probabilities in the third
decimal but — given the constant answer and the blank-control signature —
will not move any conclusion.

## Uncertainty

High confidence: the constant answer, the blank-control equivalence, the
1/3 positive control (measured twice, this lane and the prior one), the
availability survey (every MISSING verified, every ollama model probed with
an actual image). Medium: nothing — there is no positive result to be
confident about. The one thing this experiment cannot tell you is whether a
*capable* reader would find speaker identity in these magnitude spectrograms;
see §7.3 and §8.
