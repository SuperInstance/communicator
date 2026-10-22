# BRIEF - Wake and turn protocol (Casey, 2026-10-21) - how a prompt begins and ends

Casey dictating the design, in his terms, after living with Gemini Live
(endpointing that fires on a thinking pause, forcing ums and ramble to
hold the floor) versus chat dictation (dead air tolerated, everything
accumulates into one transcript). Three mechanisms:

## 1. The pre-wake buffer (listening before the prompt)
The system records ALWAYS and, while ambient, keeps transcribing in
score format (transcript + MIDI melody) for the last few minutes.
When a prompt begins, the human may use pronouns and references to
that pre-wake talk ("it", "that thing I said"). On wake, the resolver
looks BACK over the buffer (Jev-style ranking: recency x salience from
the shading layer) to ground the prompt. The buffer is description
only - no judgments banked against it; resolution happens at wake,
with more information, per doctrine.

## 2. The pause grammar (pauses are cues, not absence)
Endpointing must be pause-TYPE aware, never a bare silence timer:
- thinking pause (incomplete syntax, mid-thought prosody, level or
  held pitch): floor HELD - the person is about to say the thing they
  had to think about; the pause itself is a deliberation cue.
- breathing/beat pauses: floor held.
- terminal pause (falling prosody, complete syntax, longer): floor
  may yield.
Labelled seed data already exists: the realdialogue protocol scene 3
(the thinking pause, marked most important) and tonights free
sessions. MiniMax owns the descriptive study: pause distributions,
durations, what precedes/follows them.

## 3. OVER - the explicit transmit word (radio protocol)
A trained turn-yield signal. It fires ONLY as a conjunction:
- the word "over" in terminal position,
- spoken with its own prosody (a distinct melody shape, the way a
  question has one),
- followed by ~1.5-2s of silence.
Mid-sentence "over" never fires (no terminal prosody, no trailing
pause). Reliability is a convention both sides learn, like radio
operators: the human does not pause after "over" unless they mean
it. The drill deck gains a class: OVER meant vs OVER incidental,
potato-style, so the prosody of meant-OVER becomes measurable.

## Build order
ZCode (page/liaison): buffer plumbing + wake assembly (prompt =
wake point + resolved buffer window), OVER detector as a probe on
the evolve page first (measure false-fire rate on ambient talk),
pause-aware endpointer behind it. MiniMax: the pause study.
Report false-fires and missed yields as the headline numbers.
