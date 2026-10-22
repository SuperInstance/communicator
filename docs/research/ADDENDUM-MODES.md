# ADDENDUM - Modal agnosticism and mode-aware response (Casey, 2026-10-21)

## The architecture does not care what the sense is
The guided-RL front-end design (capture, describe, shade, guide,
consolidate, forget wisely) is MODAL AGNOSTIC. Voice is one channel.
Equally first-class: video gestures and posture (OpenCV-class, the
Kinect idea on cheap cameras), gait and movement around the room,
and the PHONE AS A GESTURING DEVICE - how it is held, how it swings:
its motion is part of the conversation. Every input has meaning once
the system and the person start responding to each other through it;
the vocabulary grows because the person performs INTO a responsive
system and learns what it hears (the potato-drill logic, generalised
to the body). Per-modality students differ; the architecture is one.

## Mode-aware digestion (the operational consequence)
The shading layer reads the persons MODE off the body - posture,
motion, gaze, phone handling - and the mode conditions how words
are treated, before any word is judged:
- RAMBLE MODE (walking around, talking): compress freely - key
  points, brush strokes.
- DEEP-THOUGHT MODE (still, staring out the window, architecting):
  do NOT brush-stroke. Point by point, outline form, everything
  preserved - the thought accrues in layers, like paint, and
  flattening it destroys the layering.
Same words, different handling, because the body declared the intent.
Mode is per-person and learned; sessions are tagged with their mode
in the portrait so the calibration is auditable. Digestion policy is
a mode-indexed table, owned by the conversation surface (ZCode);
mode inference quality is a student metric (Jetson tier).
