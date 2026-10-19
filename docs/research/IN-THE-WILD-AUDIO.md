# In-the-wild audio: what transfers from ManiWAV and See-Har-Feel

Lane: deep research on `real-stanford/maniwav` and `JunzheJosephZhu/see_hear_feel`,
for the local-first voice peripheral.

**Read this first.** The load-bearing result of this lane is *negative* and it
contradicts a stated project assumption. The project's documented "150–200 Hz
gate weakness" is a property of a **test fixture**, not of 150–200 Hz. The
frequency band where the gate actually lets mains hum through is **60–120 Hz**.
And the harm that actually materialises is not the gate at all — it is F0
corruption inside segments that real speech has already opened, which no gate
threshold can see. Details and numbers in §6 and §7.

**Measured vs read.** Everything attributed to `experiments/inthewild_probe.py`
was measured on this box, this run, on synthetic signals (no audio hardware
exists here — `/dev/snd` has only a timer). Everything attributed to a paper
was read in the paper or repo, with a section or file reference. Where the
papers do not support a claim, §3 and §5 say so rather than filling the gap.

---

## 1. What ManiWAV actually does

Source: [arxiv.org/abs/2406.19464](https://arxiv.org/abs/2406.19464) (v2, 04 Nov
2024), code at [github.com/real-stanford/maniwav](https://github.com/real-stanford/maniwav),
built on UMI ([github.com/real-stanford/universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface)).

**Hardware (paper §3.1).** "we redesign one gripper finger to embed a
piezoelectric contact microphone that senses audio vibrations through contact
with solid objects. The audio signals can be easily streamed to the GoPro
camera through a mic port and stored synchronously with vision data in MP4
files." The capture card is named in §3.2: an Elgato HD60 X into an Ubuntu
22.04.3 desktop.

**Audio encoder (paper §3.2, "Audio Encoder").** AST (spectrogram
transformer). Input is the last 2–3 s, resampled **48 kHz → 16 kHz** "which
helps filter high-frequency noises and increase the frequency resolution of
task-relevant signals on the spectrogram." Then log-mel with FFT/window 400,
hop 160, 64 mel banks, linearly normalised to [-1, 1].

**Noise augmentation (paper §3.2, "Audio Data Augmentation"; Fig. 3 caption).**
This is the load-bearing recipe and it is three parts:

> "we randomly sample audio as background noises from ESC-50. The sounds are
> normalized to the same scale as the collected sound in the training dataset.
> We also record **10 samples of robot motor noises under randomly sampled
> trajectories with the same contact microphone location as deployment time**.
> The background noises and robot noises are overlayed to the original audio
> signal, each with a probability of 0.5."

Note what the second clause is: a **per-device, per-mount-position calibration
set**, recorded at the place the sensor will actually sit. Not a generic noise
library.

**Audio latency calibration (paper App. A.1).** The clapperboard method:

> "Similar to using a clapperboard in film production, we tap on the contact
> microphone and match the time between the frame when the finger is observed
> to be in contact with the microphone and the corresponding audio signal
> captured by the contact microphone. The audio is received about 0.06s after
> the image is received. As a result, the total audio latency is 0.17 + 0.06 =
> 0.23s, where 0.17s is the calibrated image latency following the approach in
> [1]."

The constant is wired in at
[`scripts_real/eval_real_umi.py`](https://github.com/real-stanford/maniwav/blob/main/scripts_real/eval_real_umi.py)
as `--mic_obs_latency` (default `0.23`) beside `--camera_obs_latency` (`0.17`).
In [`umi/real_world/umi_env.py`](https://github.com/real-stanford/maniwav/blob/main/umi/real_world/umi_env.py)
that value becomes `receive_latency` for the `MultiMicrophone`, so audio and
image are resampled onto the same clock before the policy ever sees them.

**The result that matters most to us (paper §4.4 and App. B.2).** They compared
three ways of closing the audio domain gap:

| method | outcome |
|---|---|
| train-time noise augmentation | best |
| test-time noise reduction (spectral mask) | worse — see quote below |
| no noise augmentation | worst; does not generalise |

On the contact-vs-environment comparison (§4.4, "Taping Task"): the baseline
using "a Rode VideoMic GO II directional microphone on the GoPro camera" "fails
to pick up the subtle differences between the surface material", scoring like
the vision-only policy, while the contact microphone picks up hook-vs-loop
velcro reliably.

And the quote that constrains everything downstream of it (App. B.2, Fig. 12):

> "Even though the noise reduction seems to successfully remove most of the
> robot noises and some other background noises, **it does not preserve the
> original signal well and still results in a domain gap** as compared to the
> training signal."

Their own §4.5 limits: the contact microphone "is robust against environment
noises in the background by design" but "can easily become imperceptible due to
robot motor noises during deployment."

---

## 2. What See-Har-Feel actually does

Source: [arxiv.org/abs/2212.03858](https://arxiv.org/abs/2212.03858) (CoRL 2022),
code at [github.com/JunzheJosephZhu/see_hear_feel](https://github.com/JunzheJosephZhu/see_hear_feel).

**Hardware (paper §3.1).** "We attach a piezo contact microphone to the object
of interest, and receive acoustic signals through vibration. The microphone is
sensitive to any contact with the surface that the microphone is attached to and
**captures minimal environmental noise**." Hardware list in §4.1: HOYUJI TD-11
piezo contact mic at 44.1 kHz into a TASCAM US-4x4HR; RealSense D435 at
320×240/60 Hz; GelSight tactile at 400×300/30 Hz.

**Fusion (paper §3.2; `src/models/imi_models.py`).** MulSA: encode each
modality separately, stack the three embeddings as **three tokens**, run one
`nn.MultiheadAttention` layer over them with a residual add, concatenate, then
a linear bottleneck and an MLP head. It is a modality-level attention over three
vectors — cheap in *parameters*, but see §5 for why it still does not transfer.

**The ablation that is actually worth reading (paper Table 1, dense packing,
3D-printed bases).**

| | hard slant | soft slant | back flat | left flat | avg |
|---|---|---|---|---|---|
| MulSA (V) | 0.00 | 0.00 | 0.00 | 1.00 | **0.25** |
| MulSA (V+A) | 0.80 | 1.00 | 0.10 | 0.50 | **0.60** |
| MulSA (V+T) | 0.00 | 1.00 | 1.00 | 1.00 | **0.75** |
| MulSA (A+T) | 0.20 | 1.00 | 1.00 | 1.00 | **0.80** |
| MulSA (V+A+T) | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |

The load-bearing sentence: "the model cannot tell the difference between the
hard and the soft slanted bumps if not using audio, because the two bumps
produce **very similar visual and tactile feedbacks, while the sound** [is
different]." That is the occlusion claim, stated as a measurement.

**The limit that should have been in the abstract (paper §5).** "Audio provides
immediate feedback and also indicates the characteristics of the objects such as
material properties, **but is less informative when there is no contact.**"

---

## 3. Contact microphone as a physical noise filter — the precise answer

**What the papers support.** Both papers assert a contact mic rejects
environmental noise. ManiWAV backs that with one comparison (§4.4: env-mic
baseline fails on the velcro task). See-Har-Feel asserts it (§3.1). **Neither
paper publishes a frequency response, a transfer function, an SNR figure, or a
spectrum for the transducer.** I searched both full texts for resonance,
frequency response and coupling discussion; there is none. ManiWAV's only
frequency-domain remarks are the 48→16 kHz resampling rationale (§3.2). So:

> **Any claim about a contact mic's behaviour at 50/60 Hz is NOT supported by
> either paper. What follows is my own reasoning from acoustics, and it is
> labelled as such.**

The reasoning, three steps:

1. **There is no spatial term to exploit at mains frequency.** λ = c/f gives
   6.9 m at 50 Hz, 5.7 m at 60 Hz, 2.3 m at 150 Hz, 1.7 m at 200 Hz. A
   glasses-class device (~0.15 m across) is 1–10% of a wavelength across that
   entire band, so the incident acoustic pressure is essentially **uniform over
   the whole device**. No shadowing, no spatial gradient, no near-field effect.

2. **A contact mic's rejection of airborne sound is therefore a level ratio,
   not a frequency-selective law.** A piezo contact mic has no diaphragm and no
   air cavity, so it responds to structural motion, not to pressure variation in
   the air. What it rejects is the *coupling efficiency of the housing* versus
   the *coupling efficiency of the contact event*. That ratio applies roughly
   equally to every airborne frequency. It is a broadband, event-proximity
   signal — which is precisely what ManiWAV's velcro result measures, and why
   that result is real.

3. **The mic's own roll-off cannot separate hum from voice.** A piezo contact
   mic's transfer function is roughly high-pass (no low-frequency
   pressure–displacement coupling; sensitivity rises toward the structure and
   piezo resonances, typically kHz). That roll-off attenuates 150 Hz mains
   harmonics **and** the 85–255 Hz speech F0 range at the same rate, because a
   150 Hz mains harmonic and a 150 Hz male vowel fundamental are *the same
   frequency*. No linear filter in the world separates them. Only the amplitude
   ratio and the harmonic structure differ.

**Verdict.** A contact mic is a **principled fix for broadband airborne
ambience** (room tone, construction noise, HVAC) and a **partial fix for
narrowband tonal hum**. Against our specific 150–200 Hz weakness it buys
nothing distinguishable, because that band is inside the speech fundamental
range the transducer also attenuates.

**And the hardware does not transfer at all for us.** Our signal is airborne: a
person talking to the glasses. A contact mic on a temple measures jaw and frame
vibration, so it would reject the user's voice as thoroughly as it rejects the
hum. The measurement that decides this is See-Har-Feel's own §5 limit — audio
"is less informative when there is no contact" — read in reverse: our task *is*
non-contact.

What survives is the **method**, not the transducer: the idea that a sensor's
own noise is knowable, cheap to characterise, and worth shipping as an artefact.
§5.4 covers why the runtime version of that idea is already falsified for us.

---

## 4. The one experiment I ran (prerequisite to the proposed one)

`experiments/inthewild_probe.py` → `experiments/inthewild_probe_results.json`.
Run on the shared venv interpreter. 6 s clips at 16 kHz, one silero ONNX
session reset between signals, `GATE_SPEECH = 0.5`. Signal generation uses the
same 12-harmonic 1/k peak-normalised stack as the existing `hum_sweep.py`, and
additionally reproduces `tests/test_vad.py`'s 3-harmonic unnormalised fixture so
the project's own regression case can be re-run beside it. **This does not
duplicate `hum_sweep.py`** — that file has an empty results JSON and covers
hum-alone and speech-alone; this file adds the speech+hum mixture, the emitted
F0, and the conditioning arms.

### 4.1 The documented weakness is a fixture artefact

| signal | gate mean | frac ≥ 0.5 |
|---|---|---|
| `testvad_stack_150Hz_amp0.3` (3 harm, unnormalised) | 0.943 | **1.000** |
| `testvad_stack_175Hz_amp0.3` | 0.940 | **1.000** |
| `testvad_stack_200Hz_amp0.3` | 0.894 | **1.000** |
| `hum_stack_150Hz_amp0.05` (12 harm, peak-norm) | 0.021 | **0.000** |
| `hum_stack_150Hz_amp0.3` | 0.019 | **0.000** |
| `hum_stack_150Hz_amp0.5` | 0.008 | **0.000** |
| `hum_stack_200Hz_amp0.05` | 0.041 | **0.000** |

Same nominal frequency, two fixtures, opposite verdicts. `hum_sweep.py` uses the
second; `tests/test_vad.py::TestDeclaredLimits` uses the first. The "0.66–0.97
at 150–200 Hz" number is real *for its fixture* and does not describe a
mains-harmonic-shaped hum.

### 4.2 The real boundary is 60–120 Hz

| hum stack @ 0.05 | gate mean | frac ≥ 0.5 | watcher CPU **with** gate |
|---|---|---|---|
| 50 Hz | 0.0253 | 0.000 | 6.8% of a core |
| **60 Hz** | 0.0856 | **0.037** | **38.7%** |
| **100 Hz** | 0.0484 | **0.005** | **39.0%** |
| **120 Hz** | 0.0492 | **0.005** | **43.0%** |
| 150 Hz | 0.0213 | 0.000 | 6.9% |
| 175 Hz | 0.0327 | 0.000 | 12.0% |
| 200 Hz | 0.0410 | 0.000 | 7.2% |

60/100/120 Hz is where the mitigation actually leaks, and 60 Hz *is* the mains
case. Two mechanisms compound: (a) a low fundamental puts its 1/k harmonics into
the speech band (60×12 = 720 Hz, 100×12 = 1200 Hz) where a 150 Hz stack's
harmonics climb out of it fast; (b) a continuous hum means a 0.5–4% frame pass
rate still opens one long speech segment, so YIN runs over the whole clip.
**CPU saved is not proportional to the pass rate.**

### 4.3 The harm that actually materialises is F0 corruption, and the gate is blind to it

| signal | emitted `hz_p05` | voiced_ratio |
|---|---|---|
| real speech alone | 163.61, 179.04, 189.28, 194.49 Hz | 1.00 |
| speech + 150 Hz @ 10 dB | **74.97, 74.99, 79.86 Hz** | 0.98 |
| speech + 150 Hz @ 0 dB | **72.70, 72.89, 148.81 Hz** | 1.00 |
| 150 Hz hum alone (gate rejects it, frac 0.0) | *none — no pitch at all* | 0.00 |

Read the last two rows together. **The gate rejects the 150 Hz hum perfectly in
isolation — and that does not help.** The moment real speech is present the
segment opens, YIN runs over speech *and* hum, and the reported F0 collapses by
roughly an octave to ~75 Hz = 150/2. YIN is autocorrelation; a strong periodic
interferer gives it a narrow peak at the hum period and at its subharmonics,
and the gate's per-segment decision never inspects the interior. This is the
concrete version of what `tests/test_vad.py::test_which_is_why_the_gate_may_not_decide_what_is_reported`
asserts as a principle, with a mechanism attached.

### 4.4 Per-device noise calibration, applied to our frozen gate: falsified

ManiWAV's mechanism needs a *trainable* model. Silero is frozen, so our only
lever is the one they measured as worse. Spectral subtraction against a profile
calibrated on 60 Hz @ 0.05, α ∈ {1, 2}:

| arm | gate mean | frac ≥ 0.5 |
|---|---|---|
| real speech, unconditioned | 0.6428 | 0.6364 |
| real speech, α=1 | 0.6253 | 0.6257 |
| real speech, α=2 | 0.6250 | 0.6257 |
| 150 Hz hum @ 0.3, **unconditioned** | 0.019 | **0.000** |
| 150 Hz hum @ 0.3, profile α=1 | 0.0404 | **0.0107** |

Two things, both negative. It buys nothing on frequencies it was not calibrated
for (150/175/200 Hz stay at frac 0.0 — they were already rejected), and at
matched amplitude it makes the gate's job **worse** (0.0 → 0.0107), because
subtracting a mismatched profile leaves artefacts the learned front-end likes.
Meanwhile every α tested costs ~1.7% relative real-speech detection
(0.6364 → 0.6257). **This is ManiWAV's "does not preserve the original signal
well → domain gap" result, reproduced on our gate.** A runtime per-device
denoiser should be treated as already falsified, not as an open option.

### 4.5 Caveats on all of the above

Synthetic signals, not recordings of equipment — this measures the gate's
response to a frequency structure, not to a fan bearing or a coil whine. One
6 s clip, one machine (x86 desktop), one gate instance per signal. My SNR is
speech-RMS-to-hum-RMS; a real room's ratio was not swept independently of
speech loudness. Hum-alone CPU numbers on continuous signals are dominated by
one long segment and should be read as such.

---

## 5. Transferable ideas, ranked

### 5.1 The gate is a CPU filter, not a correctness filter — measure the emitted decision
**The most valuable idea in this lane.** Ranked first because it redirects the
project's remaining work rather than adding to it.

Both source papers are built on the same structural commitment: audio is an
*input feature to a learned model*, and the evaluation is on the emitted
decision, not on any intermediate statistic. ManiWAV's whole negative result
(§4.4) is a statement about signal fidelity feeding a model. See-Har-Feel's
Table 1 measures success rates, not gate outputs. Our gate is the only stage in
our pipeline whose correctness is currently argued from a probability.

The discipline: **when a stage exists to make a downstream stage affordable,
evaluate it on what the downstream stage emits** — in our case `hz_p05`,
`voiced_ratio`, `inflections`, the tables built from them — not on the gate
probability. `experiments/inthewild_probe.py` §4.3 is the first time this
project has done that for hum.

**Mechanism of the fix it points at.** Since a gate threshold cannot see inside
an open segment, the correction belongs where the corruption happens: either
(a) a periodicity-consistency test on the F0 track — real voice F0 moves
continuously, a mains-locked estimate does not — or (b) notching the mains
lines before YIN. Both are O(mains lines), not O(model).

### 5.2 The clapperboard discipline for latency
**Real, cheap, and directly applicable.** ManiWAV App. A.1 measures a sensor
offset with a physical event both sensors can see — a finger tap on the
contact mic, matched to the frame where contact is visible — and carries the
result as a single constant (`0.23 = 0.17 image + 0.06 audio`) into the runtime
plumbing. The transferable content is not the number; it is **measure the
relative offset of a sensor pair with a shared physical event, once, rather than
inferring it from an average**.

For us that means our 214–554 ms should not be one global figure. We have at
least two capture paths (local mic vs cloud ASR) and they do not share a clock
event. The discipline says: pick a per-path offset, measure it against something
both paths can see, and carry the constant — instead of treating the range as
one distribution and designing around it.

### 5.3 Train-time augmentation beats test-time denoising — and we cannot do the first
This is a **constraint**, not a tool, and it is the reason §4.4 came out
negative. ManiWAV's recipe works because they train. We hold a frozen 2.2 MB
ONNX gate, so conditioning the input is exactly the "test-time noise reduction"
they measured as losing to augmentation and as damaging the signal. Anyone
proposing a runtime denoiser for this project should be shown §4.4 first.

### 5.4 Ship the per-device noise *characterisation* — as a measurement, not a filter
What survives ManiWAV's calibration set is the **artefact**, not the technique:
10 clips of the robot's own motor noise, recorded *at the deployment sensor
position*. The wearable analogue is one short idle-noise clip per unit — its
fan, frame resonance, charger buzz — stored as a regression fixture.

**Failure mode, and it is the same one they hit.** Their §4.5: contact-mic
audio "can easily become imperceptible due to robot motor noises during
deployment." Noise is not stationary over a unit's life. Ours has a fan that
spins up. §4.4 measured what staleness does: a profile that helps nothing
starts costing speech the moment the device changes. So the honest version of
this idea is a *re-measurement trigger*, not a one-shot profile.

Note the tie to ADOPTION-PLAN item 11: a per-unit noise clip is the only thing on
that list that makes held-out-device generalisation falsifiable on one device,
even with no second device.

### 5.5 Resample early to buy low-frequency resolution — already ours, no action
ManiWAV resamples 48 kHz → 16 kHz specifically to "filter high-frequency noises
and increase the frequency resolution of task-relevant signals" (§3.2). We
capture at 16 kHz natively, so this decision is already made and already the
cheap side. Recorded here so nobody re-litigates it.

### 5.6 Fusion: take the ablation discipline, not the architecture
See-Har-Feel's Table 1 is worth one habit: **measure each modality alone before
believing the fusion.** V alone = 0.25; adding audio took it to 0.60 and the
full stack to 1.00, and the paper names the specific case where only audio
separated the classes. That habit is cheap and we can apply it to our own
camera.

What is *not* worth taking is the architecture, and the reasons are structural,
not stylistic: their policy runs at **10 Hz with a 200 Hz low-level control
loop**, three synchronised sensors (60 Hz vision, 30 Hz tactile, 44.1 kHz
audio), offline workstation inference, and a 0.5 s audio window with N=6
history frames (≈3 s of audio). See §6 for the accounting.

**Sceptical note on fusion for a voice peripheral.** The occlusion argument —
audio reports events vision cannot see — does not port, because our occlusion
problem is different in kind. Our microphone is not occluded by a hand; it is
*masked*, and no amount of vision resolves a masked mic. What a forward-facing
camera on a glasses-class device does see is the *interlocutor*, and the
question worth asking is therefore not "does fusion help the gate" but "can
seeing the speaker's mouth movement disambiguate a decision the audio gate
cannot make." That is a real possibility and it is untested — but it is a
different experiment from porting MulSA.

---

## 6. What does NOT transfer

Stated explicitly so it is not re-litigated.

1. **Contact-microphone hardware.** §3. Our signal is airborne. Full stop.
2. **AST / spectrogram transformer, CLIP ViT-B/16, diffusion policy head,
   10-DoF action output, behaviour cloning from teleop.** All assume a
   GPU-class trainer and a closed-loop controller. Our entire hot path is a
   2.2 MB ONNX gate in front of a stdlib YIN, sized against "a small fraction
   of one core on Pi-class hardware."
3. **Noise augmentation as a technique.** It is unavailable to us (we do not
   train the gate) and its test-time stand-in is measured worse (§4.4).
4. **The in-the-wild capture pipeline** — GoPro + Elgato + redesigned gripper
   finger + MP4 muxing. That is capture-scale engineering for a robot
   demonstrator.
5. **GelSight tactile sensing** (400×300 @ 30 Hz, vision-based). No role in a
   voice peripheral.
6. **See-Har-Feel's audio windowing** (0.5 s per step, N=6 ≈ 3 s). That is a
   10 Hz control-loop budget. An always-on peripheral's budget is the opposite:
   the cheapest possible number on the vast majority of frames.
7. **Their latency *numbers*** (0.23 s) — different hardware, different pipeline.
   The clapperboard *method* transfers (§5.2); the constant does not.
8. **The claim that contact mics are "robust against environment noises by
   design."** It is an assertion in ManiWAV §4.5 with no published spectrum
   behind it. Do not inherit it as fact.
9. **The transformer-fusion rationale** ("shift invariance … less suitable to
   audio spectrograms", §3.2). Interesting, and irrelevant to a binary gate.

---

## 7. Contradictions and complications to current design assumptions

Flagged because they change plans, not because they are tidiness notes.

1. **"The gate fails at 150–200 Hz" is a fixture claim, not a frequency
   boundary.** §4.1. Two fixtures, same nominal frequency, frac 1.000 vs 0.000.
   Any plan that budgets work against "150–200 Hz" may be fixing a synthetic
   artefact. The two numbers in circulation (`src/vad.py`'s 0.0205 mean over
   90 s, `tests/test_vad.py`'s 0.66–0.97 max over the first 0.5 s) are not
   comparable statistics and were never taken at the same amplitude — 0.05 vs
   0.3, a 15 dB gap.

2. **The real leak is 60–120 Hz — the actual mains case.** §4.2. 60 Hz @ 0.05
   leaks at frac 0.037 and costs 38.7% of a core *with the gate installed*,
   against 6.8% for 50 Hz. And the saving is non-linear in the pass rate,
   because a continuous signal turns a small frame pass rate into one long
   YIN run.

3. **Fixing the gate would not fix the F0 corruption.** §4.3. The gate already
   rejects the 150 Hz hum perfectly; the damage happens inside segments real
   speech opened. This inverts the priority: the gate's failure mode is a *cost*
   problem, and the *correctness* problem is downstream in the pitch tracker.
   If that holds, §4.2 is a performance item and §4.3 is the correctness item.

4. **An unreconciled CPU discrepancy.** My ungated watcher on a continuous
   150 Hz stack measured **116.9% of a core**, against the 31.8% recorded in
   `src/vad.py`. Both cannot be the number for "watcher, no gate". Likely
   contributors: segment length (a continuous hum is one long segment), clip
   length, machine load. I did not chase it and it should not be quoted until
   it is resolved — it is the denominator of any "the gate saves N%" claim.

5. **The 150–200 Hz band is inside the speech F0 range (85–255 Hz).** Any
   spectral fix aimed at mains harmonics in that band is also attenuating voice
   fundamentals. This constrains the notch design in §8 and is a reason not to
   expect a free lunch there.

---

## 8. Proposed next experiment (hours, files only, no audio hardware)

**Notch the mains lines before YIN and measure the emitted F0 — not the gate
probability.**

Rationale: §4.3 localises the correctness failure inside the pitch tracker, and
§4.1/§4.2 show the gate is not the place to fix it. A mains notch is a handful
of IIR/FIR bands at 50/60/100/120/150/200 Hz (±3 Hz), applied to the audio
*after* the gate has opened a segment and *before* `pitch.PitchTracker`. It
costs nothing on the frames the gate rejects, which is the entire point of the
architecture.

Run it against the existing fixtures, as a controlled A/B:

- **Arms:** (a) current pipeline, (b) notch before YIN. Same signals, same clip.
- **Signal:** the speech+hum mixtures at 50/60/100/120/150/200 Hz × {20, 10, 0}
  dB, plus `examples/realaudio/alice01_excerpt.wav` alone.
- **Primary metric:** emitted `hz_p05` on mixtures — does the 75 Hz octave
  collapse come back to the clean-speech 164–194 Hz band?
- **Regression metric:** `hz_p05` on clean speech, which must not move.
- **Cost metric:** watcher CPU with the gate, to confirm the notch does not eat
  the 31.8% → 2.4% saving.

**Decision rule, fixed before the run.** If the notch restores the F0 band on
mixtures without moving clean-speech F0, take it. If it does not, the octave
error is a YIN-configuration problem rather than a spectral one, and §5.1's
periodicity-consistency test becomes the next thing to try. Either way the run
is decisive, which is the point: this is a few hours against a 2.2 MB model on
files already in the repo.

**Why not the alternatives.** Per-device denoising is already falsified
(§4.4). Contact-mic hardware is excluded on mechanism (§3). Fusion needs a
camera and a latency budget we do not have (§5.6). The notch is the only
proposal in this lane that is both cheap and pointed at the failure that was
actually measured.

---

## 9. Unknowns and risks

- Every signal here is synthetic. Nothing in this lane is evidence about any
  real fan, charger, or frame.
- One clip, one machine, one gate instance. The gate's response to a given
  frequency structure is stable within a run; across machines and silero
  versions it is not verified.
- The device-notion of "60 Hz @ 0.05" is a waveform, not a level. Amplitude
  here is peak amplitude, which is not how anybody reports a noise floor.
- §4.2's CPU attribution between "harmonic content in the speech band" and "one
  long segment" is not separated; both are plausible and they have different
  fixes.
- Whether the octave error is 150/2 or a genuine subharmonic of the mixed signal
  is inferred from the 74.97–79.86 Hz cluster, not proven.
- See-Har-Feel's fusion claim is about *contact* events. Any port to a voice
  peripheral rests on the assumption that the analogous event — someone
  speaking — is one the camera can independently observe. That assumption is
  named here and **has not been tested**.

## 10. Suggested next step

Run §8. Then, if the notch works, re-run this probe against it and fold the
result back into `src/vad.py`'s docstring and `tests/test_vad.py`, because both
currently state the 150–200 Hz weakness as a property of the frequency rather
than of the fixture, and that is the specific sentence most likely to send the
next person down the wrong path.
