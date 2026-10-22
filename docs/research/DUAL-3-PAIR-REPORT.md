# DUAL-3 PAIR REPORT — the duet take, measured (2026-10-22, zcode)

Phone session 07:52:12–08:03:25Z (236 s walkabout, 513 frames, 103
keyframes, 5 audio parts, 7 RIR sweeps). Witness chain: 8 sessions,
754.2 s at 48 kHz from 07:51:07Z. **Headline: cross-device offset and
drift are UNMEASURABLE on this take** — four independent detectors, all
at their noise floors, numbers below. The take's audio also exposed a
real page bug (the parts race), now fixed and self-tested. The TDOA
position track is blocked by the same cause (no sync), not attempted on
 guesses.

## The detectors and their floors

| detector | best evidence | verdict |
|---|---|---|
| envelope correlation (300–2000 Hz, 20 ms hops) | r ∈ [0.765, 0.809], but sweeps 3–6 all "arrive" at the same witness instant (t≈542 s) = template self-match artifact | no signal |
| TF-path matched filter (log-frequency glide, 64 bands) | pathSNR = 1.00 on all 7 (peak = background; scores dominated by speech energy) | no signal |
| spectrogram correlation (phone loopback as template, 1–5 kHz glide, Pearson) | r ∈ [0.074, 0.082] across 7 sweeps, ±90 s search | noise floor |
| 8–16 kHz band energy at the 7 expected windows | mean 0.01 vs global p99 0.04 (uniform) | sweeps absent above witness floor |

Consistent with dual-1: **the phone speaker at gain 0.4 does not reach
the Realtek witness across the room.** The clap channel was also dead:
the witness transient scan found only a periodic artifact (below), and
the phone's ambient audio — the only place its claps would live — is
4.4 s of the ~3.5 min (see the defect).

## The measured artifact (noted, unexplained)

The witness chain carries a strictly periodic transient: exact 20.000 s
spacing, uniform strength (91× local floor), continuing across session
boundaries (07:51:12.53, :32.53, :52.53 … for the whole chain). Not a
clap signature, not the sweeps, not page-tied (page events are 1 s /
300 s / 600 s periods). Benign for this analysis; flagged for
identification.

## The defect this take caught: the parts race (FIXED)

The phone banked 5 parts but only the last decodes (4.4 s; parts 1–4
= 0 samples via ffmpeg). Root cause, from the source: my serialization
fix called `stopMicRecorder(true)` then cleared `micChunks`
synchronously — racing the async `onstop` blob build, so the banked
parts carried empty/broken bytes. Fixed (room-lens `474d6c0`): the
measure now awaits the bank; the self-test reports per-part sizes.
**Self-test v3: `audioParts: 2, partKB: [40, 88], rirMeasures: 1`** —
both parts carry real bytes with a measure mid-walk. Commit chain for
the whole audio-fix arc: `2a0564c` → `fd903e7` → `474d6c0`.

## Why motion-envelope sync also failed (an honest negative)

Cross-modal fallback tried: phone ASCII character-change rate vs
witness luma |Δ| — best r = 0.170 at the ±90 s search edge = noise.
The reason is structural, not bad luck: the phone's frame-change signal
is dominated by SELF-MOTION (mean 39% of characters change per 400 ms —
hand-shake, not room events), while the witness's motion signal is
Casey-in-frame. Roving-camera change and fixed-camera change are
different observables; correlating them needs pose correction first.

## Protocol for the next take (each item kills a measured failure)

1. **CLAPS, loud, near BOTH devices for sync moments** — stand within
   arm's reach of the laptop for the three claps, then walk. The sweeps
   (phone-speaker, gain 0.4) are proven inaudible at the witness twice;
   do not spend sync on them again unless the rail rises (Casey's
   decision) or the phone lies next to the witness for one sync sweep.
2. **Reload the phone page before the take** (`/lens` serves HEAD;
   parts-race fix is live only after reload).
3. Keep the witness page's PAUSE off; AUTO armed. The 20 s artifact is
   worth one look at the bench while setting up.

The analysis script (`browser_capture/pair_analysis.py`) is staged with
the clap finder; on a take with audible claps it produces offset, drift
(sub-ms resolution at 5 ms hop onset estimates), and the TDOA track in
one run.

---

## ADDENDUM — the research verdict, applied (2026-10-22)

Survey verdict received (VM research notes, two-ears-two-eyes
20261022-0915). Three points, each answered:

**1. Alignment method confirmed**: windowed GCC-PHAT + robust affine
fit (drift dominates; resample after). Adopted in `pair_analysis.py`'s
design for any take with a live sync channel.

**2. CORRECTION ACCEPTED — no position point from two mics.** One TDOA
= one hyperbola branch, and the clock offset is perfectly confounded
with range difference: the honest target is the TDOA *curve* + floor-
plane prior + motion filtering + the cameras (my script's note already
said "hyperbola, not a point"; the fuller target is now the stated
goal). **New lane item for the next take: BeepBeep-style two-way
ranging** — laptop and phone exchange the same chirp both directions;
the round-trip cancels the clock offset entirely and pins metric scale
for the views (literature: ~1–2 cm over 10 m). Implementation sketch:
the witness page plays a chirp when the phone's sweep fires (both
record both); round-trip time from the two recordings' emission and
arrival stamps gives range at each sync moment — a ranging track
along the walk, no hyperbola ambiguity.

**3. The 7 sweeps, deconvolved (Farinas ESS, FFT-domain; honest band
250 Hz–4 kHz).** T20 from Schroeder backward integration (-5→-25 dB
× 3), per sweep, with distrust levels from peak-vs-pre-roll SNR:

| sweep | at | peak SNR | T20 250–1000 Hz | T20 1–4 kHz | distrust |
|---|---|---|---|---|---|
| 1 | 07:52:12 | 3.6 dB | 2.56 s | 3.56 s | high |
| 2 | 07:58:13 | 3.3 dB | 2.72 s | 3.97 s | high |
| 3 | 07:59:00 | 3.9 dB | 2.69 s | 3.91 s | high |
| 4 | 08:00:07 | 2.6 dB | 2.61 s | 3.70 s | high |
| 5 | 08:00:53 | 3.7 dB | 2.60 s | 3.70 s | high |
| 6 | 08:01:30 | 6.6 dB | 2.68 s | 3.85 s | high |
| 7 | 08:02:56 | 4.3 dB | 2.65 s | 3.84 s | high |

The consistency is real (250–1000 Hz: 2.56–2.72 s across seven
positions; 1–4 kHz: 3.56–3.97 s) but **every sweep's peak SNR is 3–7
dB — the deconvolved tails sit barely above the phone's own noise
floor, so these T20s are TREND-TRUSTWORTHY, VALUE-DISTRUSTWORTHY**
(high distrust on all seven: the -25 dB crossing may be noise arrival,
making measured T20 an upper bound on the true decay). The phone's
honest band tops out ~4 kHz (transducer); above that its loopback
spectrum dies, so no 4–8 kHz T20 is reported at all. Room verdict the
numbers whisper: a liveable, moderately live room (domestic T20
~2.5–4 s is long — consistent with the bare-floors spare room the
videos show), pending a louder sweep or a bench-side measurement for
trustworthy values.
