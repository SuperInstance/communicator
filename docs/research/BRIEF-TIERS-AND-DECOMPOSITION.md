# BRIEF - Tiers and decomposition (Casey via Muse, 2026-10-20) - design law for all liaison work

## General purpose, hardware-tiered
This system is not a Casey-only build. The local node does MORE or LESS
depending on the hardware it finds itself on, and the design must stay
portable across tiers. The forcing case is the BOAT: a 128 GB Thor on the
boat, no dependable internet - the local voice agent there must be
self-sufficient. Home, with fast internet, is the easy mode; the fleet is
an upgrade there, never a dependency. Design for the boat, degrade up.

## Decomposition over monoliths
Do NOT reach for a sophisticated end-to-end model that hears sentiment.
DECOMPOSE into the easy tools that have existed for 20 years: VAD, pitch
tracking, quantize pitch to notes, envelopes, dynamics. A single voice is
a single melody. Every stage small, old, inspectable, replaceable -
visible seams. The quantized SCORE (MIDI + parameters) is the portable
currency between tiers: a weak node ships scores up when a link appears;
a strong node ships nothing.

Tier sketch (earn each rung, do not build ahead):
- phone / Uno-class: record + meter; score elsewhere.
- laptop (RTX 4050 6 GB): decompose locally (STT, pitch, MIDI);
  digestion may visit the fleet.
- Jetson 8 GB: bigger local STT + profile analysis.
- Thor 128 GB (boat): everything local incl. digestion and learning;
  autonomous offline.

## Caricature (Casey, the mechanism under the Tone Game)
Quantization is the feature, not the loss: once prosody is parameters,
what STICKS OUT becomes findable - and exaggerable. A caricature keeps
the few features that make a face that face and leans on them. Same
here: the parameters that stick out ARE the identity (person portrait)
and the sentiment (readability). Tone Game addendum: give the performer
a CARICATURE KNOB per parameter (pitch range, onset sharpness, dynamics)
and report accuracy vs exaggeration per intent. The reader should read
a caricatured intent more reliably - measure the curve, do not assume it.
