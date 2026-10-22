# BRIEF - The Liaison conversation page (Casey via Muse, 2026-10-20)

## The experience Casey specced (his reference: Gemini voice mode)
You ramble. It listens. When you pause, it comes in with a DIGESTION of
what you said. Back out, and the full transcript is there. Recreate the
local page as that experience - with two differences that are the point:

1. EVERYTHING IS SAVED. No ephemeral sessions. Every ramble lands on disk
   in full: audio, video, transcript.
2. SAVED AS A SCORE, NOT FLAT TEXT. The archive format is the annotated
   one from the speech-to-score work: words + MIDI melody + face/motion
   segments on ONE clock (timestamps.json spine), per-syllable frame
   indices. The purpose: an ANALYSIS THAT GROWS OVER TIME VERY UNIQUELY
   TO THE PERSON - the corpus accumulates and the system learns THIS
   speaker: his contours, his intent vocabulary, his tells. A per-person
   prosody portrait that deepens with every session.

## Build order (do not invert it)
1. Audio sick-lane fix (the ears must be true) - in flight.
2. Liaison v1 API (BRIEF-ROOM-LIAISON.md) - in queue.
3. Speech-to-score pipeline (MiniMax) - in queue; supplies transcript +
   melody for every segment automatically.
4. THIS page (ZCode): ambient listening surface; on pause/segment close,
   the segment is scored and a DIGESTION appears (one short paragraph:
   what was said, in essence - generated via the fleet, MiniMax lane
   supplies the digestion plumbing); scroll-back shows the full
   transcript with the score attached; every element links to the saved
   annotated bundle.
5. Person-profile analysis (MiniMax): a living profile doc updated per
   session - contour statistics, intent-vocabulary growth, drift over
   time. Starts descriptive; earns inference slowly.

## v1 honesty
Digestion v1 is text on screen, not a spoken voice reply - the loop must
be TRUE before it is charming. The annotation and the accumulation are
NOT v2: a conversation not captured as a score tonight can never become
one later. Capture format is destiny; polish iterates, prosody does not
wait.
