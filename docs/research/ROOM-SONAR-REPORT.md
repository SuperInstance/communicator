# ROOM-SONAR-REPORT — step 1: the channel, measured (2026-10-21)

Per `BRIEF-ROOM-SONAR.md`: measure the usable near-ultrasonic band on
THIS hardware before any claims. Probe: `browser_capture/probes/sonar-band.js`,
run by the evolve lab (`/evolve`) against the real laptop — ASUS FHD
webcam's machine, "Default — Microphone Array (Realtek(R) Audio)" as the
mic, the laptop's own speakers, all browser audio processing off.

## Method

Exponential sine sweep 16→22 kHz (3 s, 10 ms fades) + Voss pink burst
(0.5 s), gain 0.25 (the shell's rail ceiling is 0.3), through the real
speakers; the mic tapped around them by the same AudioWorklet the
capture path uses; 8192-point Hann FFT; transfer = |return| / |stimulus
reference| per 250-Hz band; SNR against a 0.4 s pre-stimulus room floor.
Verdict rule (stated, not tuned): a band is USABLE when sweep SNR ≥ 10 dB
and transfer within 40 dB of the best band; the usable band is the widest
contiguous run in 15–24 kHz. Raw captures + reports saved as probe
session folders (mode `probe`) — e.g. `free_sessions/1792561987` — for
the step-2 deconvolution, should a band ever open.

## The v1 lesson (kept, it is the report's own error bar)

v1's verdict said "22–24 kHz usable" — an ARTIFACT: bands the sweep never
excited were scored through a normalization floor (dividing by a ~zero
reference). v2 (hot-swapped into the already-open lab page, no reload —
the evolve loop's first live proof) scores only bands the reference
actually drove (within 60 dB of the reference's own peak); everything
else is marked NO EXCITATION.

## The measurement (v2, real hardware, honest)

| band (kHz) | sweep SNR (dB) | pink SNR (dB) | verdict |
|---|---|---|---|
| 16–17 | −19 to −28 | −13 to −20 | dead |
| 17–18 | −19 to −25 | −13 to −19 | dead |
| 18–19 | −18 to −26 | −15 | dead |
| 19–20 | −16 to −24 | −14 to −15 | dead |
| 20–21 | −13 to −25 | −14 | dead |
| 21–22 | −12 to −27 | −14 to −16 | dead |

**The usable band on this hardware is NONE.** In every band the sweep
drove, the return measures 12–29 dB BELOW the room's noise floor; the
pink burst agrees (−12 to −20 dB everywhere). At the modest level the
safety rails enforce, this laptop's speaker→air→mic chain does not carry
a measurable near-ultrasonic return.

## What that closes, and the one decision that could reopen it

The lane closes WITH A NUMBER on this hardware at rail-compliant levels:
the best in-band SNR is ≈ −12 dB (21 kHz), i.e. the return would need
≥ 12 dB more level (or a quieter room, or a stiffer mic gain) just to
reach the floor, and ≥ 22 dB to clear the 10 dB rule. Physically this
is the expected commodity-laptop story: small drivers roll off hard
above ~15 kHz. Whether an EXTERNAL speaker (or a raised level ceiling —
a policy change for Casey, not a probe tweak) opens a band is the one
honest reopen; steps 2–3 (impulse response, motion) are NOT justified
on this chain until it is.

Artifacts: probe session folders under
`/mnt/c/proart_capture/corpus/free_sessions/` (mode `probe`, `meta.sonar`
carries the full band table; `audio.wav` is the raw return).
