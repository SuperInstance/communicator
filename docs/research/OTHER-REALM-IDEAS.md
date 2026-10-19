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

## 2. Scope correction, and a retraction

**Scope.** The source material is framed around boat and workshop systems. This
project is a **general-purpose voice peripheral**; a boat is one deployment
domain, not the target. So the room-fingerprinting, the hull-resonance features
and the `confined_metal_hull` environmental label are all *domain plugins*. The
architecture underneath them is the part worth stealing, and this file is scored
on that basis. An earlier version of this file read the boat framing as if it
were our problem and over-weighted it.

### 2a. RETRACTED: the OpenVLA-Micro citation was fabricated

**An earlier version ranked "hidden-state distillation with a frozen head" as the
single most transferable idea and cited `theguy21/openvla-micro`. That model
does not exist.**

Verified four ways: the model page 404s, `resolve` returns 401, the author
`theguy21` owns exactly one unrelated Octo checkpoint, and a GitHub repository
search returns zero results. The account is real (profile 200), so this is not a
typo for a similarly-named project.

The figures quoted with it — action cosine 0.997, ~14× faster, a
896→2048→4096 shim — came from a search snippet and were never checked against
a primary source. That is precisely the failure this project keeps recording: a
confident number from an unverified artifact, wearing the costume of a citation.

Hidden-state distillation remains a real technique, and it remains a reasonable
thing to try. But **the specific claim and the specific source are withdrawn**,
and it is demoted to "inference from general literature, not something we read
in a paper".

### 2b. What survives, on the general-purpose framing

**Cross-modal teacher–student with a contrastive alignment loss.** Vision is
the teacher when it can see; audio is the student. This is the same mechanism as
teacher–student self-training, which this project already uses in a smaller
form: local models score candidates and a stronger model is consulted only
occasionally.

For a *general* voice peripheral the honest version is narrower than the boat
framing suggests. The transferable claim is not "audio predicts vision" — it is
that **a perceptual front-end can be trained against a privileged signal that
is available at training time and absent at run time.** For us the privileged
signal is not a camera; it is the *two-microphone separation* we can record once
and cannot have in production. That is a cleaner and more available teacher than
vision, and it is already sitting in
`examples/realtwospeaker/ihm_EN2002a_channels_90s/` as paired clean channels
alongside the far-field mix.

**Open question, not answered:** whether any of this transfers to
**classification** — which is what a turn decision is — rather than to action
regression. Do not assume it does.

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

Re-ranked after the retraction in §2a and the scope correction in §2.

| # | idea | verdict | next step |
|---|---|---|---|
| 1 | probabilities held open over time | **already ours, and now justified** | `src/turn.py` returns COMPLETE / OPEN / UNRESOLVED; silence AUC 0.529 says it *must* |
| 2 | trajectory, not snapshot | **explains our dead end** | at the decision instant END and RESUME are byte-identical, so only history can separate them |
| 3 | per-device noise baseline + drift | new and cheap | record the device's own noise floor as a calibration artifact |
| 4 | teacher = two-microphone separation | reframed and more available | clean channels already exist alongside the far-field mix in our corpus |
| 5 | modality dropout | right answer, no training set | blocked on the recording protocol, not on the idea |
| 6 | hidden-state distillation | **source withdrawn**, inference only | do not cite; try only after a teacher exists |
| 7 | room RIR fingerprinting | real technique, **domain plugin** | boat/workshop feature, not the general core |
| 8 | "lattice" as a model substitute | not supported | keep the discipline, drop the claim |

**On #2**, which is the most important line in this table: we measured that at
the moment of decision, END and RESUME are byte-identical audio. The
information is not in the frame. Every route we have tried — duration, pitch,
route voting — reads a single instant, which is why all of them failed for the
same reason. A trajectory representation is not a stylistic upgrade here; it is
the only place the information can physically be.

**On #7**, the scope correction matters practically: a room-impulse-response
feature is a plugin for one domain. Building it before the general path would be
optimising a case study.

---

## 7. Sources

- ManiWAV — <https://github.com/real-stanford/maniwav>, arXiv 2406.19464 (MIT)
- UMI — <https://github.com/real-stanford/universal_manipulation_interface>
- See, Hear, and Feel (CoRL 2022) — <https://github.com/JunzheJosephZhu/see_hear_feel>
- OmniVLA, navigation — arXiv 2509.19480, <https://github.com/NHirose/OmniVLA>
- OmniVLA, multisensory — arXiv 2511.01210 (ICRA'26), "Physically-Grounded
  Multimodal VLA with Unified Sensor Integration"
- Speaker-change detection, for calibration of our own dead end —
  wav2vec 2.0 SCD <https://arxiv.org/abs/2210.14755>; USM-SCD
  <https://arxiv.org/abs/2309.08023>

**WITHDRAWN:** `theguy21/openvla-micro` — cited in an earlier revision of this
file, **does not exist** (404 / 401, author owns one unrelated checkpoint, zero
repository results). See §2a. Do not cite it.

Related, in this repo: [TWO-SPEAKER-TURNS.md](TWO-SPEAKER-TURNS.md),
[F0-SPEAKER-CHANGE.md](F0-SPEAKER-CHANGE.md),
[DEVICE-GENERALISATION.md](DEVICE-GENERALISATION.md),
[../examples/realdialogue/README.md](../examples/realdialogue/README.md).