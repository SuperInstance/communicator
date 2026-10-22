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
