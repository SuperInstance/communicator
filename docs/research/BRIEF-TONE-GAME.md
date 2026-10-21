# BRIEF - The Tone Game (Casey via Muse, 2026-10-20) - the acting/understanding loop

## Caseys design
"If you can act it, you can understand it." Two players, one loop:
- THE PERFORMER (generator) says a carrier phrase with a chosen INTENT,
  rendered as a pitch melody (MIDI first; audio render too - make noise).
  Intents start as the potato deck: sincere / leaving / obligated /
  everything / joking / quiet-furious.
- THE READER (discriminator) hears ONLY the melody (no words, no labels)
  and guesses the intent.
Rounds: the performer reshapes each intents contour to maximise
READABILITY - not beauty - and the reader retrains. Report accuracy and
the confusion matrix per round: which intents blur is the finding.

## Anchors (a judge cannot certify itself)
- NOW: seed it from REAL prosody - the seed corpus captures (A/B/C) have
  true audio; extract their pitch behaviour as the starting material, plus
  synthetic contours where data is missing (label which is which).
- LATER: Caseys potato-drill deliveries become the ground truth, and
  Casey himself is the external referee: periodic LISTENING CHECKS where
  he grades renders blind. The loop is graded from outside the dojo or it
  is graded by nobody. That is the point of the design (his words: with a
  reason outside, it stops being a GAN loop and becomes excellence).

## Build (this worktree): experiments/tone_game.py
Pipeline: contour representation -> performer -> MIDI + WAV renders per
intent -> reader (features + a small classifier) -> rounds loop ->
docs/research/TONE-GAME-REPORT.md with per-round accuracy, confusions,
the learned contour per intent, and links to listenable renders.
Queue: after speech-to-score validation (it consumes pitch tooling);
audio-dependent parts wait for ZCodes sick-lane fix; seed-prosody and
synthetic parts can start now. Commit, push, iterate without waiting.
