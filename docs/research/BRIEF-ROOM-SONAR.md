# BRIEF - Room sonar (Casey via Muse, 2026-10-20) - can the page hear the rooms shape?

## The question
From the browser: emit sound (chirps, sweeps, pink noise - ideally
near-ultrasonic, above comfortable hearing), record the returns, and
understand the room - its shape from reflections, and Casey moving in
it (toward / away) from how the echoes change. The sleep-app sonar he
remembers is a real technique; commodity hardware is the open question.

## Step 1 - MEASURE THE CHANNEL (before any claims)
Probe page mode (ZCode, on the liaison page, additive): play an
exponential sine sweep 16k to 22k plus a pink-noise burst through the
laptop speakers with audio processing OFF; record the mic return.
Report the per-frequency return level: the usable band ON THIS HARDWARE
is whatever survives the speaker+mic chain - measured, never assumed.
Sample rate is 48k, so Nyquist is 24k; expect the honest band to be
somewhere in 17-21k. If the chain is deaf above 16k, say so and the
lane closes with a number, not a vibe.

## Step 2 - ROOM IMPULSE RESPONSE
Deconvolve a clean sweep capture (MiniMax, Python): RT60, and the
strongest early-reflection delays converted to distances. Compare
against the rooms known rough dimensions if Casey offers them.

## Step 3 - MOTION TRIAL
Chirp train at low duty cycle in the measured band while Casey walks
toward and away from the laptop. Report: which delay taps move, whether
approach vs retreat separates, and whether still-presence (breathing-
class micro-motion) is visible above noise. Verdicts measured only.

## Rules
Modest playback level, short duty cycle (near-ultrasonic is inaudible
to most adults, not to dogs, and not to everyone). All returns land as
another channel on the session clock if this graduates - the room gets
a score too. File results in docs/research/ROOM-SONAR-REPORT.md.
