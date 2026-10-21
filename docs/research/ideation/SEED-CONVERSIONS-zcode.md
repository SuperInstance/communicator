# SEED-CONVERSIONS — zcode's lens on Casey's overnight seed

**Provenance, stated first.** The seed is Casey's mailed-to-himself
conversation: open-source real-time vision identity → conversion
technologies → acoustics → crossovers with our ASCII designs. The full
text is to land at `SEED-EMAIL.md` beside this file; **it had not landed
when this first edition was written** (waited ~3.5 h past 2026-10-21
00:00 AKDT). A watcher is running: when it lands, this document gets
re-read against it and re-aimed (second edition, changes marked). What
stands regardless of the aim — because it is measured, tonight, on this
hardware — is §2; the catalog classes in §3 are the aimable part.

The lens I was given: **CONVERSIONS — what conversions between these
technologies the page tier can actually perform.**

---

## 0. What "conversion" means here

Not format-shuffling. A conversion, in this document, is a
**representation-to-representation transform the PAGE can perform live
on the box**, whose output lands on the session's one clock as a file.
The framing gift is `EXPERIMENTS-AUDIO.md`'s sharpened thesis: the raw
waveform contains all the information; every derived representation is
a **lossy projection whose loss is chosen**. A conversion is good
exactly when its chosen loss is the loss meaning lives outside of. The
page tier's special property — the reason this lens is MINE — is that
the page is the only tier that sits **inside the room with all three
physical transducers** (camera, mic, speakers) while also holding a
full DSP environment (WebAudio graph, AudioWorklet, canvas, typed
arrays, an in-page FFT proven at 5.86 Hz resolution). Everything the
page converts, it converts **at the moment of capture, on one clock** —
no other tier can say that; every other tier converts after the fact,
from files we shipped it.

The tiers law (`BRIEF-TIERS-AND-DECOMPOSITION.md`) governs the catalog:
decomposition over monoliths, fleet as upgrade never dependency, the
boat the forcing case. A page-tier conversion that needs a cloud model
is not a page-tier conversion.

## 1. The page as it stands: the converter's bench

What the page already holds, all proven in production tonight:

| holds | proven by |
|---|---|
| camera → 160×96 luma at 30 fps + per-frame ms timestamps | the session corpus; every bundle |
| mic → full-rate float32 (AudioWorklet tap), no processing | the WAV in every bundle; the deafness telemetry |
| speakers → railed stimulus output (gain ≤ 0.3, duty ≤ 5 s/60 s) | the evolve lab's probe rails, live |
| in-page radix-2 FFT, 8192-point Hann, 5.86 Hz bins | the sonar probe, measured |
| canvas 2D up/downscaling, BT.601 luma math | the capture path itself |
| the one-clock session bundle (WAV + luma + timestamps + meta) as files | the whole archive; the forwarder ships these shapes |
| hot-swappable probe modules that can run any of the above on schedule | sonar-band v1→v2, swapped live without reload |

This bench IS the conversion surface. The catalog below asks, for each
pair of representations the seed's conversation touches: can the page
perform it, at what class?

**Classes:** PROVEN (measured, this repo, numbers below or cited) ·
CHEAP (pure composition of proven pieces, hours not nights) · DEEP
(real engineering, page-tier possible, needs design against the laws) ·
OTHER-TIER (the page feeds it, another tier performs it) · CLOSED
(measured dead on this hardware, with the number).

## 2. Tonight's measured proofs (the part no re-aim touches)

Three conversions were built and measured tonight, on real data from
the real corpus, to ground this document. Numbers are citable.

### 2.1 luma → braille (the ASCII-crossover conversion) — PROVEN

A real mid-session frame from `free_sessions/1792560773` (the
quiet-speech era; frame luma mean 77.4) rendered as a 2×4-dot braille
field: **160×96 luma → exactly 80×24 = 1,920 characters.** The
geometry is not a coincidence to squint at — the corpus's own NV12
chroma is 80×24 interleaved; the braille field and the chroma plane
share dimensions because both halve/quarter the same 160×96 lattice.

- Bytes: 5,783 B UTF-8 vs 15,360 B luma = **2.66:1**.
- Content: 76.6% non-empty cells at threshold 96 — on a DARK-era frame,
  which is the caveat wearing its own sign: `ASCII-MOTION.md` retracted
  its threshold-dither mechanism precisely because an empty room
  flipped 50.00% of dots. **Braille conversion is
  threshold-dependent, and the repo already knows the threshold
  measures the camera unless row-banded** (`ROW-BAND-REMOVAL.md`).
- Compression honesty: at 30 fps, braille is ~173 kB/s UTF-8 while the
  WebM of the same scene runs ~16 kB/s. **Braille is not a compression
  win and must never be sold as one; its value is that TEXT-ONLY models
  can consume the room directly** — Casey's original
  `ascii_motion_render.py` idea, now measured on the page-tier side.

### 2.2 WAV → F0 in-page (the score conversion's spine) — PROVEN, bit-equal

`src/pitch.py`'s YIN (50 ms/10 ms framing at 16 kHz, τ 40–229,
threshold 0.15, min-confidence 0.5) ported faithfully to JS and run on
**identical float32 samples** (16.1 s of real quiet speech, 48k→16k
linear resample, the capture whose RMS is 0.0003):

- 1,602 frames on both sides; **all 71 voiced frames agree: median Δ
  0.000 cents, max |0.00| cents, 71/71 within one cent.**
- The port runs in seconds in node for 16 s of audio — page-viable,
  and the live case would hop along in real time next to the worklet
  that already feeds the meter.
- Caveat carried, not hidden: this capture is so quiet that only 4.4%
  of frames clear the confidence gate, and the energy floor had to drop
  to 1e-5 — which is `audio-dsp/SYNTHESIS.md`'s exact finding (our
  gates are file-global where a live peripheral needs adaptive ones)
  answered by the page's own live rule (the 10×-floor detector the
  AUTO page already runs).

**The page tier can produce the score's pitch track exactly.** This is
the load-bearing proof for the conversation page's "local scoring" leg
and for the tiers law's laptop rung ("decompose locally: STT, pitch,
MIDI").

### 2.3 F0 → note events (score quantization) — PROVEN, arithmetic

The same 71 voiced frames quantize to **47 note events** (semitone
rounding, gap-merge), in 0 ms. The single-frame-event dominance is the
honest signature of marginal-SNR input, matching Day-0 expectations in
`EXPERIMENTS-AUDIO.md` — a property of the capture, not the
conversion. Chain complete: **WAV → F0 → notes, entirely in-page, on
the capture clock.**

## 3. The conversion catalog (the aimable part)

Ordered by the seed's own arc: vision identity → conversion techs →
acoustics → ASCII crossovers.

| # | conversion | class | evidence / what feeds it |
|---|---|---|---|
| 1 | camera photons → 160×96 luma | **PROVEN** | the corpus; the currency every tier reads |
| 2 | mic pressure → full-rate WAV | **PROVEN** | every bundle; measured-not-asserted health |
| 3 | luma → braille / ASCII text | **PROVEN** | §2.1 tonight; feeds text-only judges (the `local_judge` lineage) |
| 4 | WAV → F0 track | **PROVEN** | §2.2 tonight, bit-equal port |
| 5 | F0 → note events / MIDI-track skeleton | **PROVEN** | §2.3; feeds `src/score.py`'s nine-track asset |
| 6 | WAV → spectrum / sonagram bands | **PROVEN** | the sonar probe's FFT; speech-band version is CHEAP (same code, 80–8k bins) |
| 7 | luma → change-maps / motion maps | **PROVEN** | the corpus npz maps; the browser's own math (row-median removal lives in the shipped page) |
| 8 | score/notes → sound (the room hears the score) | **CHEAP** | WebAudio oscillator/buffer scheduling through the railed stimulus path; unproven only because unneeded until now — the playback side of the probe rails already exercises the identical path with sweeps |
| 9 | text → sound (TTS) | **OTHER-TIER / CLOSED in-page** | no offline in-page TTS worth the tiers law's weight; `voiceloop/MeloTTS.md` is the local candidate on the python side |
| 10 | sound → text (STT) | **OTHER-TIER (DEEP on laptop)** | tiers law puts STT on the laptop's python side (4050), fed by the page's WAV; the page's job is the one-clock feed, not the model |
| 11 | vision → identity (the seed's first leg: real-time person/face identity) | **OTHER-TIER** | no in-page ML by the decomposition law; the page converts photons → the luma/motion representations identity models consume — the page is the FRONT of that conversion, not the middle. What the page CAN own: the identity-probe loop (present/absent, same/different room) as stimulus+capture experiments, sonar-probe-shaped |
| 12 | ultrasonic room acoustics (sonar) | **CLOSED, with the number** | `ROOM-SONAR-REPORT.md`: return 12–29 dB under the room floor in every excited band at rail levels; this hardware's chain does not carry it |
| 13 | speech-band room acoustics (RT60-class, level, onset stats) | **CHEAP→DEEP** | the FFT is proven; deconvolution math is MiniMax's step-2 lane per the sonar brief — the page CONVERTS (stimulus out, return in, one clock), the fleet computes |
| 14 | ASCII ↔ score crossovers (melody rendered as character fields; braille motion with note tracks) | **DEEP / speculative** | both halves exist separately (§2.1, §2.3); the crossover the seed gestures at — one clock carrying BOTH the braille room and the note events of the speech in it — is a design question, not a feasibility one. This is the one item I most expect the seed text to re-aim |

## 4. The one-clock spine (the architectural claim)

Every conversion above lands as a file on the session's timestamps
clock (`timestamps.json` + ISO wall clock). That is the whole
architecture: **the bundle is the conversion bus.** A conversion
upgrades (fleet, offline, next year) without redesign because its input
and output are files with clocks, per the tiers law's visible seams.
The page's monopoly is the CONCURRENT conversions — it is the only tier
that can convert camera, mic, and speaker events onto one clock while
they happen. Everything else is downstream projection.

## 5. What I could not decide without the seed (the re-aim slots)

1. **"Conversion technologies"** in the relayed arc could mean format
   conversion (container/codec), modality conversion (this document's
   reading), or model-conversion (distillation/pruning — the
   cross-modal-fusion lane's territory). I bet on modality; the bet is
   marked.
2. **The vision-identity leg**: whether Casey's interest is the
   open-source landscape survey (MediaPipe-class face/identity models —
   an OTHER-TIER catalog entry) or identity AS a conversion target (a
   person → a stable text token the score can carry as a track). If
   the latter, §3 row 11 becomes a first-class design item: the page
   supplies presence+luma, the Jetson supplies the embedding, the
   score carries `who` as ninth-track-adjacent events.
3. **Which crossover the ASCII designs actually want**: consumability
   by text models (my §2.1 reading), human legibility on a terminal
   bench, or aesthetic unification with the score. The three imply
   different conversions next.

## 6. Appendix: tonight's artifacts

- braille: `/tmp/proof_braille.txt` (80×24), frame
  `/tmp/proof_frame.npy`; pitch cross-check `/tmp/pitch_ref.json` +
  `/tmp/speech16k.f32`; notes inline in §2.3.
- Working notes with the full context bank (ASCII-MOTION retractions,
  audio-dsp synthesis findings, FLEET-MUSIC map) in
  `.scratch/zcode-seed-proof-notes.md` — scratch, not citable, but the
  citations above are.
