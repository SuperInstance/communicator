Caseys name for this architecture (2026-10-21): GUIDED REINFORCEMENT LEARNING, FRONT-END CENTRIC - the persons corrections at the interaction surface are the reward signal; the back end only consolidates.

# BRIEF - The synergy loop (Casey via Muse, 2026-10-20) - how the fleet gets better all the time

## The question Casey posed
How do the captures, the APIs, and the different compute tiers synergize
so the whole system improves continuously? Answer, as architecture:

## The loop
1. CAPTURE - the liaison records automatically; everything annotated on
   one clock (audio, video, score). Data accrues by the owner living.
2. SCORE LOCALLY - decomposition on the box (VAD, pitch, MIDI, face,
   motion-text). Cheap, private, immediate.
3. TEACHERS LABEL VIA API - the big models (fleet subscriptions, cloud
   multimodal) grade SAMPLES: intent, sentiment, quality, what happened.
   Sampling is a curriculum decision, not an accident - label what the
   students are worst at. Every cloud sample is a deliberate, logged
   choice (consent axis); local-first is the default.
4. STUDENTS LEARN AT THE EDGE - small models (intent-from-melody reader,
   floor decider, segment triage) train and evaluate on teacher labels,
   on the Jetson batch tier, against the growing corpus.
5. REALITY GRADES - held-out REAL data the students never train on, plus
   Casey as referee (blind listening grades). A judge cannot certify
   itself: this step is the hole in the sphere, and it is not optional.
6. GAPS WRITE THE CURRICULUM - confusion classes commission the next
   drills (the potato deck adapts to the intents the student confuses)
   and the harvest report names what the corpus lacks.
7. DEPLOY BACK - improved students ship to the liaison tier and run
   live; the loop returns to 1.

## The headline metric
TEACHER-DEPENDENCE RATIO: the fraction of judgments the system must
send to a teacher, over time, at held referee-graded quality. The
programme is improving iff that ratio FALLS while quality holds or
rises. Every digest reports it. Improvement you cannot plot is a rumor.

## Failure classes are curriculum too
Every failure becomes (a) a detector in the liaison, (b) a test in the
suite, (c) a line in the digest. Prototype: SILENT_AUDIO - caught by
hand once, self-reported forever after. The archive teaches the crashes.

## Who runs what
- ProArt: capture + live decomposition + digestion (interactive tier).
- Jetson: student training/eval, embeddings, batch passes (patient tier).
- Oracle: rendezvous, scheduling, the nightly FLEET DIGEST (captures
  gained, teacher labels spent, student deltas, failure classes,
  curriculum proposed).
- Muse: maintenance + curriculum director - reads digests, spots
  failure classes, commissions drills, verifies by measurement.
- Casey: performer of the drills, referee of record, owner. The
  non-shell anchor; his grades are the ground truth the loop cannot game.
