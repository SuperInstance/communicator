# BRIEF - The retention ladder and the garbage collector (Casey, 2026-10-21)

## The problem Casey named
Saving everything in every form is unaffordable: an hour of raw audio
is ~350 MB; its MIDI score plus transcript is a few hundred KB. Three
layers, orders of magnitude apart. The system needs a sophisticated
garbage collector - which on inspection is a CURATION system.

## The layers (what each is FOR)
- AUDIO (waveform): ground truth. Complete at capture, expensive.
- MIDI (the score): a MECHANICAL summary - pitch, dynamics, timing -
  not linguistic. The middle point: keeps how it was said, drops
  the bulk.
- TRANSCRIPT: the linguistic residue. Cheapest, often sufficient.

## Two ladders
RESOLUTION (at wake/query time): answer from the cheapest layer that
settles the question. Clear words -> transcript suffices. Ambiguity
classes (sarcasm? question? earnest?) -> reach into the MIDI. Recent
material (last hours, still HOT) -> re-analyse the raw audio itself.
Escalate only on ambiguity; the layers are a ladder, not a pile.
RETENTION (the GC, run in the reflection phase, nightly):
- HOT (hours): full audio + all derivations; re-analysis allowed.
- WARM (after derivation completes): MIDI + transcript carry the day.
- KEPT FOREVER: transcript + MIDI + the QUINTESSENTIAL SET (below).
- Audio for the routine bulk is collected ONLY when derivation is
  complete, the segment was not selected, and audits pass.

## The quintessential set (the GCs real product)
Deletion is the side effect; curation is the point. From each days
audio, keep a small set of clips that are quintessential of THIS
person: meaning carried beyond words and beyond tone - first-of-class
deliveries, emotional extremes, moments the shading layer flagged as
surprise, teacher/student disagreements, referee-marked moments.
The set exists so the system can TRAIN ON THE USER during idle time:
back-end models re-fit on the quintessentials, a better per-person
agent by accumulation. Selection criteria are measured, not vibes:
MiniMax owns the mining study; the catches are the truth - keep the
oddities, collect the routine.

## The guardrail (non-negotiable)
Collection is the one sanctioned deletion in a never-delete house,
so it runs DRY FIRST: nightly report of what WOULD be collected,
with audit samples re-checked for quintessentials hiding in the
routine pile (report the miss rate). Live collection starts only
when the miss rate is measured and near zero, and Casey rules on it.

## Ownership
ZCode: ladder plumbing in the liaison (per-session tier state,
derivation status, dry-run GC report wired to the digest).
MiniMax: quintessential selection study over the current corpus
(criteria + first candidate set + audit method).

## Addendum (Casey, 2026-10-21) - forgetting is a learned skill

### Resolution decays with age, every layer (schedule is per-person)
- Audio: collected first (hours to days).
- MIDI: thins next (a week or two). Before a MIDI stretch is let go,
  its ESSENTIALS GRADUATE INTO TRANSCRIPT ANNOTATIONS - the sarcasm,
  the question-rise, the earnest weight become marks on the words
  (a small versioned annotation vocabulary: bracket/footnote/hash
  conventions to be designed from evidence). Annotations are
  judgments: they ride as a revisable derived layer with provenance,
  never baked into the descriptive transcript.
- Oldest routine material: annotated transcript alone may suffice.
- MIDI QUINTESSENTIALS survive like audio ones: cadence exemplars -
  segments kept because the cadence itself carries meaning and the
  phrasing patterns must be learnable over time.

### The garbage collector is a student, not a static rule
Its curriculum is the systems own error history:
- ERROR-DRIVEN RETENTION: any segment beneath a conversation where
  the system got something wrong (corrections, negative feedback in
  the back-and-forth) is PINNED - exempt from collection at every
  layer - until shadow work completes.
- SHADOW WORK: background re-analysis of the pinned material with
  the critique as the question (reach down the resolution ladder as
  far as needed), producing an extracted lesson filed to the person
  portrait. The debt clears when the lesson exists, not when time
  passes.
- The GC tracks what it understands vs does not: collection is
  aggressive exactly where understanding is VERIFIED, conservative
  where it is not. Digest metrics: regret rate (audits), pin count,
  lessons extracted, post-lesson re-ask performance (Casey spot-
  checks as referee).
