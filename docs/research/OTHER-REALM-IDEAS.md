# Other-realm material: room acoustics, cross-modal learning, and the JEV substrate

A synthesis of ideas brought in from adjacent domains, scored against what this
project has **measured**. Written to record what is worth chasing and what is
already known not to work, so the same proposals are not re-litigated later.

The house rule here is that a plan written from other people's numbers fails on
contact with real audio. Three of the eight original recommendations in
[ADOPTION-PLAN.md](ADOPTION-PLAN.md) did exactly that. This file applies the same
suspicion to material that arrives with confident framing.

---

## 1. Measured here first, before anything else

The shared snippets open with `sd.playrec(chirp, ...)` capturing a room
impulse response from inside WSL, and assert that WSL "handles the laptop's
built-in array microphones and speakers without requiring audio server bridge
configuration."

**Measured on this machine, that is false.**

| probe | result |
|---|---|
| `/dev/snd` | contains only `timer` — no `pcmC*`, no capture device |
| PulseAudio sources | `RDPSink.monitor`, `RDPSource` only — an RDP bridge, not a host mic |
| `arecord` / `parec` / `pactl` / `ffmpeg` | all absent |
| `sounddevice`, `pyaudio`, `soundfile`, `cv2` | not installed |
| `pulsectl` (installed to test) | connects, but this build ships **no recording API at all** |
| `qemu-*` / `docker` / `podman` | unavailable or integration-disabled |

So the self-pinging room scanner cannot run here as written, and neither can any
other live-capture idea. **Everything below is therefore either file-based
measurement or reasoning from published work** — which is why the recording
protocol in `examples/realdialogue/README.md` exists.

A second problem, independent of hardware: `playrec` measures the **laptop's own
loudspeaker and microphone in a room**. That is a real and useful measurement,
but it tells you about the laptop. It says nothing about a wearable, and it is
circular if used to validate a voice pipeline — the signal being measured is
the signal the test injected.

---

## 2. The one idea that genuinely transfers

**Cross-modal teacher–student with a contrastive alignment loss.** Vision is
the teacher when it can see; audio is the student; the loss pulls the audio
embedding toward the visual one (InfoNCE or similar). The audio model then
carries information about the world that the camera would have supplied.

This is the same mechanism as teacher–student **self-training**, which this
project already uses in a much smaller form: `src/judge.py` / `experiments/`
score local candidates and a stronger model is consulted only occasionally.

What the adjacent literature adds is the specific recipe in
**OpenVLA-Micro**: replace a 7B stack with a 0.6B one, insert a *learned hidden
state shim* (896→2048→4096), keep the teacher's pretrained head **unchanged**,
and recover **action cosine 0.997 at ~14× the speed**. The claim worth stealing
is not the architecture — it is that **you do not need to re-learn the head.**

That maps onto an open problem here: we may want the semantics of a large model
we can only query intermittently, inside a loop that must run always-on. If the
head is frozen and only a projection is learned, the expensive part is paid once.

**Open question, not yet answered:** the published result is action regression.
Whether hidden-state transfer survives into **classification** — which is what a
turn decision is — is not established by that result. Do not assume it does.

---

## 3. The second idea, and the cost that comes with it

**Sensor-native image unification** (OmniVLA, multisensory): render infrared,
mmWave and a microphone array as coloured masks over an RGB frame instead of
fusing raw tensors. Reported 84% task success against 25% RGB-only.

Applied to us, this is *already half-built*: `experiments/spectrogram_vision.py`
renders audio as a spectrogram image for a vision model. The lane running now
tests it against real hand-offs on the two-speaker corpus.

The cost is not obvious and is decisive here: **a spectrogram discards phase.**
Phase is precisely what distinguishes two simultaneously speaking voices. For
every other sensor in the paper (infrared, mmWave) that loss is irrelevant. For
the specific problem we have — who holds the floor when two people overlap — it
may be fatal. Render-the-sensor-as-an-image is a strong general prior; this is
one of its known-bad cases, and it should be treated as a hypothesis that can
fail rather than a pattern to apply.

---

## 4. The third idea, and why it is more relevant than it looks

**Modality dropout.** OmniVLA masks goal modalities randomly during training
and at inference, so no single modality becomes load-bearing.

This is the honest response to our situation. `docs/research/TURN-ROUTES.md`
tried four turn-detection routes, measured that their *disagreement was noise*
rather than signal, and dropped the ensemble. Voting is the wrong tool when
every route is individually weak.

But there is a catch specific to us: we have four routes and **not one dataset on
which they can be trained**, because the only corpus with real hand-offs is 840
seconds of AMI. Dropout is a training-time technique. With no training set,
there is nothing to apply it to. This makes it a bet on a corpus we do not have
— which is another argument for the recording protocol.

---

## 5. The JEV substrate idea — and a category error to avoid

The material proposes decomposing a perceptual loop into `BIND`, `EFFECT`,
`VIEW`, `TICK`, `FORGET` cell mechanics with **hash-chained receipts**, a
geometric `JEV-lattice` of decision nodes, `quilt-dba` for longitudinal baseline
and drift tracking, and `exoj` to hold probabilities open instead of forcing a
hard label.

The genuinely good parts:

- **Hold probabilities open across time** rather than snapping to a label.
  Directly relevant: our own `src/turn.py` already returns `COMPLETE` / `OPEN` /
  `UNRESOLVED` for exactly this reason, and the speculation gate exists because
  committing early is the dangerous direction.
- **Drift tracking against a historical baseline.** A device's own noise floor
  is not constant — a fan spins up, a charger engages. A per-device calibration
  baseline is the natural answer, and it is the same pattern as ManiWAV's
  `robot-noise-calib` set.
- **Receipts on every decision** — a decision that leaves no trace is not
  auditable. That is already written into `src/watcher.py`'s `GATE_FIELD`.

The category error: the snippets' `evaluate_acoustic_lattice()` is

```python
if decay_time > 0.8: ...
if spectral_centroid > 4000.0: ...
```

Two `if` statements. Calling that a lattice does not make it one, and presenting
it as a substitute for a learned model would repeat the exact failure this
project documents — mistaking a hand-tuned threshold for structure. The
*discipline* (bounded, auditable, receipted decisions) is real and worth
keeping. The claim that it replaces a model is not supported.

---

## 6. Ranked, and what to do about each

| # | idea | verdict | next step |
|---|---|---|---|
| 1 | hidden-state distillation, frozen head | **strongest** | evaluate on a classification task, not action |
| 2 | probabilities held open over time | already ours | keep; `src/turn.py` does this |
| 3 | per-device noise baseline + drift | **new and cheap** | record the device's own noise floor as a calibration artifact |
| 4 | sensor-native image unification | plausible, **known-bad for overlap** | lane running now; may fail on phase |
| 5 | modality dropout | right answer, no training set | blocked on the recording protocol |
| 6 | room RIR fingerprinting | real technique, wrong target | measures the laptop, not a wearable |
| 7 | "lattice" as a model substitute | not supported | keep the discipline, drop the claim |

---

## 7. Sources

- ManiWAV — <https://github.com/real-stanford/maniwav>, arXiv 2406.19464 (MIT)
- UMI — <https://github.com/real-stanford/universal_manipulation_interface>
- See, Hear, and Feel (CoRL 2022) — <https://github.com/JunzheJosephZhu/see_hear_feel>
- OmniVLA, navigation — arXiv 2509.19480, <https://github.com/NHirose/OmniVLA>
- OmniVLA, multisensory — "Physically-Grounded Multimodal VLA with Unified
  Sensor Integration" (infovaya PDF)
- OpenVLA-Micro — <https://huggingface.co/theguy21/openvla-micro>

Related, in this repo: [TWO-SPEAKER-TURNS.md](TWO-SPEAKER-TURNS.md),
[F0-SPEAKER-CHANGE.md](F0-SPEAKER-CHANGE.md),
[DEVICE-GENERALISATION.md](DEVICE-GENERALISATION.md),
[../examples/realdialogue/README.md](../examples/realdialogue/README.md).