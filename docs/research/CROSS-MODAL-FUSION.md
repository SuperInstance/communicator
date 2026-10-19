# Cross-modal fusion and cross-modal distillation: what actually transfers

**Status: read-only research lane. One measured result (ours), everything else read.**

**Headline, stated up front because it decides how to read everything below:**
of the three works assigned to this lane, **one does not exist**, one is about a
different problem than ours, and one contributes a training technique that is
genuinely useful but small. The most valuable outcome is a set of things that
should *not* be built.

Our own dead ends stand and are not re-litigated here: silence duration (pooled
AUC 0.529) and pitch (0.5σ / 0.3σ within class) cannot identify a speaker
hand-off (`docs/research/TWO-SPEAKER-TURNS.md`,
`docs/research/F0-SPEAKER-CHANGE.md`).

---

## 1. Work 1 — `theguy21/openvla-micro`: NOT FOUND. Do not plan around it.

The assignment describes a drop-in OmniVLA replacement with a learned
hidden-state shim (896→2048→4096), an unchanged action head, action cosine
0.997, and ~14× speedup. **I could not find any artifact matching this, and
several checks came back negative.**

| check | result |
|---|---|
| `https://huggingface.co/theguy21/openvla-micro` | **404** |
| `https://huggingface.co/api/models/theguy21/openvla-micro` | `{"error":"Invalid username or password."}` |
| `.../resolve/main/config.json` | **401** |
| HF model search `?search=openvla-micro` | no such model; only unrelated `Auryal/openvla-oft-...` |
| HF listing for author `theguy21` | exactly **one** model: `theguy21/octo-base-1.5-pytorch` (Octo, not OpenVLA) |
| `theguy21` spaces / datasets | none |
| GitHub repo search `q=openvla-micro` | **0 results** |
| GitHub repo search `q=OmniVLA micro` | **0 results** |

The user account exists (profile 200) and owns an Octo checkpoint, so this is
not a typo in a URL — the described model is simply not published. The 401 on
`resolve` is consistent with a private or nonexistent repo; either way it is
not readable.

**Consequence for the lane.** The "hidden-state distillation with an unchanged
head" idea was the most interesting item assigned, and it is the one thing here
with no verifiable source. I have **not** substituted a similar-sounding paper
and written it up as if it said this. Section 5 assesses the *idea* against the
distillation literature that does exist, and flags clearly which parts are
supported and which are extrapolation.

> If someone has a local copy or a private link, the assessment in §5 stands or
> falls on its own evidence — but the 0.997-cosine / 14× numbers should be
> treated as **unverified** until the artifact can be read.

## 2. Work 2 — OmniVLA (multisensory), arXiv 2511.01210, ICRA'26

Source: <https://arxiv.org/abs/2511.01210> (PDF referenced by the infovaya
summary). Everything below is from §III-B/III-C and §IV of that paper.

**What it does.** Infrared, mmWave radar and a **six-microphone circular array**
are rendered as *sensor-masked images*: each sensor is beamformed into an
azimuth–elevation heatmap, spatially calibrated to the RGB frame, and blended
under a SAM2/Grounded-SAM mask selected by GPT-4o from the task text (α=1 by
default). These become ordinary vision tokens, projected by per-sensor MLPs
(initialised from the base model's RGB projector), concatenated with language
tokens and fed to a VLA backbone. Vision encoder frozen; everything else
trained. SmolVLA base.

**Headline numbers (Table I).** Average success 84%, vs 25% RGB-only and 56%
raw-sensor-input. Learning-curve claim: similar success with ~50% of the
episodes (Fig. 6).

**What the paper claims is gained by image-native unification:** the sensor cue
is *spatially co-registered with the thing it is about*, so a pretrained
vision encoder processes it with its existing machinery. §IV-C supports the
spatial claim specifically: on unseen tasks, Stage 1 ("select the correct target
to interact with") improves +17% over no-pretraining and +32% over raw-sensor
input, which the authors attribute to the overlay making cues spatially aligned
with RGB. Stage 2 gains are attributed to pretraining priors, not the overlay.

**What it costs** — and this is the part that matters for us:

- **A cloud VLM in the data path.** Segmentation prompts come from GPT-4o.
  §IV-A mitigates this ("invoked at the beginning of the task and only updated
  infrequently") but it is a network call in the loop.
- **Alignment is approximate.** "High-precision pixel matching is not strictly
  required" (§III-B). The overlay is a *spatial prior*, not a calibrated
  measurement.
- **Compute**: 8×A100 for ~14 h; inference 15 Hz on an RTX 4090, with both the
  VLA and the segmentation model resident.
- **Calibration** is a one-time rotation/crop per sensor (§III-B).

**Does the phase loss matter for us?** The paper never says, and I will not
extrapolate. What can be said: beamforming is inherently phase-based — it
estimates direction-of-arrival from inter-element phase differences across
microphones. The paper's own acoustic pipeline *deliberately uses* the
quantity a magnitude spectrogram throws away. So for OmniVLA, phase was an
**asset to be exploited**, not information sacrificed. Our spectrogram
(`experiments/spectrogram_vision.py`) discards phase for a different reason —
not because it was shown to be safe, but because a spectrogram is the cheapest
thing to render. That is an unvalidated choice, and the only paper in this lane
points the other way.

## 3. Work 3 — OmniVLA (navigation), arXiv 2509.19480

Source: <https://arxiv.org/abs/2509.19480>, `NHirose/OmniVLA` (Hirose, Glossop,
Shah, Levine). Mechanism from §III.

**What it does.** Goal conditioning in three modalities — 2D pose, egocentric
goal image, natural language — with **randomised modality dropout**: for each
training sample it independently samples which modalities are present, then
builds an attention mask so unavailable modalities are **filled with random
values** and excluded from attention. At inference, modality masking lets it
run on any subset. Stated purpose: "modality imbalance and scarcity" (§I).

**The mechanism, precisely.** Dropout is not voting. It is a *data-augmentation
of the conditioning set* that forces one model to be correct under many input
subsets. The model learns "given whatever I can see right now, do the right
thing" instead of learning "combine my three experts".

**Does this fit our situation** — four routes, all individually weak, ensemble
dropped because disagreement was noise? **Yes in mechanism, and this is the
strongest transferable thing in the lane.** Concretely, it says:

1. Stop treating routes as voters. A route that is right 55% of the time is
   still a *conditioning signal*, not a vote.
2. Train one model on **randomly masked subsets** of routes, so that "route A
   alone", "A+B", "all four" are all training cases.
3. At runtime the device picks the **cheapest subset it can afford**, and the
   mask mechanism means nothing breaks when a route is unavailable.
4. Crucially, it makes the always-on budget explicit and *trainable*: a route
   you can only afford sometimes is dropped from the input, not from the label.

**The cost, stated honestly:** this needs a trainable model and a labelled
corpus where routes can be masked independently. Our corpus is 34 labelled
segments in the paired window and ~1396 pauses pooled across two meetings
(`TWO-SPEAKER-TURNS.md`). That is a thin base for learning a fusion model, and
thin enough that overfitting would be indistinguishable from success. So the
*idea* transfers cleanly; the *experiment* does not fit our current data.

## 4. Microphone array as a first-class modality

Both OmniVLA papers treat acoustic sensing as worth fusing. For an always-on
wearable, is there a result about what acoustics contributes that vision
cannot?

**Honest answer: not for our problem, and the "see_hear_feel" framing does not
apply.** The AV literature that fuses audio with vision is
*grounding/locator* work — given an image, find the pixels making the sound
(AVSBench, <https://arxiv.org/abs/2207.05042>; AV-SAM,
<https://arxiv.org/abs/2305.01836>). Audio's contribution there is **spatial**,
and vision is the modality that is semantically rich. Our question is not
"where is the sound" but "did the identity of the voice change". That is
*already* carried entirely by the audio channel. Adding vision would add a
camera to a glasses-class device to help a question vision cannot address.

The one genuinely transferable piece: OmniVLA's acoustic task gets 88% success
vs 40% RGB-only — i.e. audio found a **concealed** target under opaque
coverings (§IV, Table I). Concealment is the transferable concept, not audio
per se. For us the analogue would be *self-masking / barge-in*: 111 of 314
segments in our corpus are wholly overlapped and inaudible
(`TWO-SPEAKER-TURNS.md`), and real barge-in is partly destroyed by the channel
before we see it. **An array is a route to seeing through overlap.** That is
the honest version of the claim — and it is a reason to care about
`MIC-HARDWARE.md`'s lane, not a result we have.

## 5. Hidden-state distillation — assessing the *idea*, since the artifact is missing

Given §1, this is inference from the general literature, not from the assigned
work. **The loss:** match student hidden states to teacher hidden states
through a learned projection (a regression/MSE or cosine term on projected
representations), keep the teacher's decision head so the output space is
unchanged.

Does it apply to **classification**, not just action regression? In principle
yes — nothing in the mechanism is action-specific; a classifier's logit layer
is a head like any other. In practice, three caveats:

1. **The teacher's behaviour is inherited, warts included.** If our teacher is
   also near chance on this task, the student inherits near-chance. Distillation
   transfers a function; it cannot create information that is not in the
   teacher. Given silence (0.529) and pitch (dead), **we have not yet
   established a teacher that is good at this task.** That is the blocking
   issue, and it is upstream of the architecture.
2. **It needs a teacher you can query.** Ours is a cloud VLM reached over the
   network (`experiments/local_vs_cloud.py`) — and that same file already
   measures its turn-related behaviour as unreliable. Distilling an unreliable
   teacher is a known way to spend a lot of compute reproducing a bad function.
3. **Unknown whether it survives a fine-tuned teacher** when the shim only sees
   hidden states. **I found no evidence either way** and am not going to
   assert one.

**The most transferable idea in this lane is therefore not this one.** It is §3's
modality dropout, and only after we have a teacher worth copying.

---

## 6. Transferable ideas, ranked

| # | Idea | Mechanism | What it buys us | Cost / risk |
|---|---|---|---|---|
| 1 | **Modality dropout over routes** (§3) | random subset mask at train, mask at inference | turns weak routes into conditioning signals; makes the always-on budget a *trainable* choice | needs a trainable model + enough labelled data; our corpus is thin |
| 2 | **Beamformed spatial cues for overlap** (§2, §4) | azimuth–elevation heatmaps from array phase | the one route that could see through self-masking/barge-in | needs real array hardware — none here; explicitly deferred to the MIC-HARDWARE lane |
| 3 | **Overlay/spatial-alignment idea** (§2) | co-register sensor cue with its referent | sound in general, but our signal is not spatial | needs a second modality; no analogue in our problem |
| 4 | **Hidden-state distillation** (§5) | projected hidden-state match, teacher's head | possibly large cost saving if a good teacher exists | **blocking prerequisite: no good teacher yet** |

## 7. Explicitly does NOT transfer

Do not spend time on these; they were bulk, not load-bearing.

- **Diffusion / flow-matching action heads.** Irrelevant to a classification decision.
- **7B VLAs, OpenVLA/Octo/SmolVLA/π₀ backbones.** Orders of magnitude past an always-on budget.
- **SLAM, sim-to-real, robot deployment, multi-GPU training.** Not our problem.
- **SAM2 / Grounded-SAM / GPT-4o in the loop.** Requires a cloud VLM and multiple GPU-resident models; incompatible with local-first by construction.
- **RGB-anything.** A glasses-class peripheral with no camera in this lane.
- **The 84%/25%/56% numbers.** Robotic manipulation success on 25 trials/task. Not comparable to an AUC on a hand-off detection task, and not transferable as a target.

## 8. The experiment I ran (measured — separate from everything above)

`experiments/crossmodal_probe.py` → `experiments/crossmodal_probe_results.json`.
numpy only, four 90 s WAVs already in the tree, no network/model/GPU.

**Question published sources cannot answer for us:** is speaker-identity
information present in *our* audio at all, for a cheap spectral front end?

**Design.** `examples/realtwospeaker/` contains, for the same 90 s window,
both the far-field mic **and** four clean per-speaker headset channels, with 34
audible labelled segments. That allows a genuine oracle-ceiling test. Feature:
20-band log LTAS, 25/10 ms, 100–4000 Hz, L2-normalised, cosine similarity.
Label = different speaker. Positive control: F0 must separate MEE vs FEO in the
same code path.

| slice | n | LOO id (chance 0.25) | AUC different-speaker |
|---|---:|---:|---:|
| **ceiling:** own clean headset channel | 34 | 0.0 | **0.334** |
| **far-field:** single mic, same window | 34 | 0.0 | **0.451** |

Positive control **passed**: F0 separates the voice classes in this harness,
median |ΔF0| = 98.5 Hz (FEO070 200 / FEO072 208 / MEE071 109 / MEE073 103 Hz).
So the harness resolves a signal known to be present.

**This failed, and I am reporting it as a failure.** A deliberately trivial
magnitude-only spectral statistic does **not** carry usable speaker identity
here — not even on clean, single-speaker, un-overlapped audio. That is the
third independent dead end, and it is the one that bites hardest: it argues the
information is not sitting in the cheap spectral front end at all.

**Integrity notes, because both numbers are inverted rather than flat:**
- An inversion is a harness signature — the same shape as the 0.289 overlap
  inversion caught in `TWO-SPEAKER-TURNS.md`. I chased it: it is **not**
  explained by duration (AUC gets *more* inverted with longer segments,
  0.33→0.13), **not** by temporal drift (corr(cos,|Δt|) = −0.05), and **not**
  by shared backchannel text (only 1–2% of pairs share opening words). I did
  not find the cause. **The weak reading is the defensible one** — "this
  statistic does not work" — and I do not claim the inversion as a result.
- Channel↔speaker mapping was independently verified via per-channel F0.
- The overlap slice is **empty by construction** — AMI audible turns tile
  (0.0% overlap, `TWO-SPEAKER-TURNS.md`) — so this probe says **nothing**
  about whether overlap destroys speaker identity. That remains untested.
- n=34 segments in the paired window. Small. It is a floor, not a verdict.

## 9. Proposed next experiment (cheap, and it can fail)

**Do not build a fusion model yet. Build the teacher.**

Take `experiments/spectrogram_vision.py`'s working cloud-VLM harness and ask it
the one question that blocks everything: **for each of the 1396 labelled pauses,
does the post-pause voice belong to a different speaker?** Score against the
AMI `speaker_id` ground truth as a probability, compute AUC against the 0.6225
trivial baseline.

- **Cost:** one pass over existing WAVs + sidecars. No new data, no training.
- **It can fail, and that is the point.** If the teacher's AUC is ≈0.5, then
  item #4 (distillation) is not merely hard — it is pointless, because there is
  nothing to inherit — and we should stop pursuing architecture work on this
  question entirely and say so. If it is high, we finally have a teacher, and
  #4 plus #1 become buildable in that order.
- This is the cheapest experiment that can invalidate the whole lane.

**Second, if hardware ever exists:** beamform the overlap case. That is the only
idea here targeting the phenomenon (barge-in) that actually decides floors, and
it is the one our current corpus cannot test at all.

## 10. Claims that contradict or complicate our assumptions

1. **The assigned model does not exist.** Any plan citing `openvla-micro`,
   its 0.997 cosine, or 14× speedup is citing something unverifiable.
2. **The best available evidence contradicts our own VLM measurement.**
   `experiments/spectrogram_vision.py` found the VLM reads a spectrogram at the
   1/3 random floor. OmniVLA's result is that rendering sensor data as an image
   makes a pretrained encoder *better*, not that any image suffices. We measured
   "no signal"; they measured "large gain over an RGB baseline". These are not
   the same experiment and do not contradict each other — but neither licenses
   "just render it as an image".
3. **Our own spectrogram-vision result is not a fair test of the idea.** It used
   synthetic tones and a ~300-token image, no training, no overlay, no
   co-registration. OmniVLA trains the projectors and *spatialises* the cue. We
   should not cite our 1/3 as evidence the trick is dead.
4. **Phase may matter more than we assumed.** Our spectrogram discards it;
   OmniVLA's beamforming is built on inter-microphone phase (§III-B). We have
   never tested phase. Untested, not disproven.
5. **The ensemble we dropped may have been dropped for the wrong reason.** We
   dropped it because disagreement looked like noise. Modality dropout's claim
   (§3) is that weak-but-complementary routes are exactly what random-masking
   is *for*. Our data is probably too thin to prove it — but "the routes
   disagreed" is not, by itself, evidence they are complementary.
6. **The literature says our exact task is learnable** — speaker change
   detection is solved well above chance (wav2vec 2.0 multitask SCD,
   <https://arxiv.org/abs/2210.14755>; USM-SCD, 85.8% F1 on American English,
   <https://arxiv.org/abs/2309.08023>). Those systems use **learned
   representations**, not pitch and not silence duration. This is the strongest
   positive signal in the whole lane and it points at a learned embedding —
   which is precisely the expensive thing an always-on budget resists.
7. **Nothing here shows a mic array solves this.** It shows arrays reveal
   *concealed* sources. Overlap as an information-destroyer is untested by
   anyone in this lane, and our corpus cannot test it.

## Reproduce

```bash
/home/proart/projects/glasses/.venv/bin/python experiments/crossmodal_probe.py
```

## Uncertainty

High confidence: §1 (absence is well-evidenced), §2/§3 mechanisms (read directly
from the papers), §7. Medium: the ranking in §6 — it reflects my judgement, not
a measurement. Low / explicitly unresolved: the §5 distillation assessment
(no source found), and the cause of the inversion in §8 (unexplained).
