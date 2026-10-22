# BRIEF: DUAL CAPTURE TAKE 2 -- sync redesigned from the duets failure (2026-10-22)

What take 1 taught (DUAL-3-PAIR-REPORT): the pair existed but alignment was
UNMEASURABLE -- phone chirps at gain 0.4 never reached the witness ear and
the clap channel carried nothing usable. A sync event nobody hears is not
a sync event. Also open: the unexplained 20.000 s transient; RT60 from the
7 sweeps was trend-only/high-distrust; the page-gone watchdog does not exist.

TAKE-2 DESIGN RULES (zcode builds, room-lens next build):
1. LOUD AND NEAR: the protocol puts the sync events where the WITNESS can
   hear them -- three claps NEAR the laptop at start and end (the phone
   hears them too, quieter; cross-correlation does not care about the
   asymmetry). Chirps stay as the fine instrument INSIDE the phone track.
2. TWO-WAY WHEN READY (research shortlist #3): the witness answers --
   laptop speaker chirps back on a liaison cue, BeepBeep-style. Cancels
   clock offset AND yields the inter-device range. Build behind the loud-
   clap fix, not instead of it: evidence sooner beats elegance later.
3. LEVEL FROM DATA: before Casey is asked for take 2, a bench pass sets the
   chirp level from measurement (what the witness actually receives at
   room distance), not from a constants file.

OPEN INVESTIGATIONS (before take 2 is called):
- The 20.000 s transient: present in which tracks, at what aligned time?
  Round numbers are artefacts until proven innocent.
- Recompute the 7 sweeps the Farina ESS way with per-band SNR and r2;
  report slopes (T20) only, with the distrust graded per band.
- Watchdog: the /evolve page heartbeats; an alert must fire when the
  heartbeat goes stale -- the 2026-10-21 page-gone gap (3 h unalerted)
  is the failure this closes.
