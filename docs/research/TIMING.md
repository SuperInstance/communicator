# The timing theory — capture point, upstream calibration, priming

*2026-10-18, from Casey's frame: "MIDI captures the signal between intended
music and the sound, at a different point in the chain than a microphone...
the kick pedal has to have been in swing with full force before the down
beat... prime them for the rugged landing, and a bumpy landing is exciting
instead of scary." Sharpened here into three laws and two experiments.*

## 1. The capture point: our score is an inversion, and honesty about that

A microphone captures at the END of the chain — the rendered sound. MIDI
captures in the MIDDLE — the gesture as it becomes a control signal, before
the synthesizer decides what to sound like. Same performance, different
tapping point:

    MIDI:      intent ──gesture──► CONTROL SIGNAL ──synth──► sound
    mic:       intent ──gesture──► control signal ──synth──► SOUND ──mic──► recording

Speech gives us only the microphone. So the 9-track score of a SPEAKER is
an **inversion**: recovering the performance-MIDI from the render —
inferring the key presses from the recording. That is why the judged
tracks are hard, why presence gates exist (do not claim a control signal
that is not detectable in the render), and why calibration against
by-construction truth is non-negotiable.

The OUTPUT direction needs no inversion. When this system speaks, it can
go score-first: intent stream → control events → synthesis last. Then the
decision model is literally what Casey called it — the last mile through
our trigger — and the synthesizer renders what the score says. Inversion
is only forced on the listening side.

## 2. The kick-drum law: upstream calibration

A trigger has latency. The drummer does not fix this at the speaker; the
pedal is in full swing BEFORE the downbeat so the transient LANDS on it.
The compensation lives upstream of the render, keyed to a target moment:

    T-zero  = the moment the event must land (the downbeat)
    T-minus = T-zero − (latency of every stage downstream of you)

Everything upstream schedules against T-minus, not T-zero. Sequencers do
this (events dispatched ahead by the synth's latency); DAWs do it (plugin
lookahead, automatic delay compensation). Our stack's measured latencies
make the voice loop's T-minus table concrete: judged tracks 200–550 ms
per call, acoustic measurement ~tens of ms, synthesis TBD. A response
that must land when the user finishes speaking must START before they
finish — endpointing is not an optimization, it is the pedal swing.

Fleet instances of the same law, previously unlabeled: the pacemaker
schedules the agent's day against deadlines, not wakeups; the duty-cycle
forced probe is upstream calibration against attention decay; the
digest-not-document finding is runway preparation for the model (§3).

## 3. Priming is receiver-side calibration: why bumpy can be exciting

The cross-examination frame: the case-winning question lands on a jury
only if every story told on the way WRAPPED IN TIME to prepare the
landing. The tour must be LED to the airfield — the rugged landing
announced before it happens.

Formally, priming is **calibrating the receiver's prior to the payload's
actual variance**. A bumpy landing is scary exactly when unpredicted —
when promised variance ≠ delivered variance. That is miscalibration, the
same defect ECE measures in models, now measured in communication: the
speaker's implicit promise ("smooth approach") was not calibrated to the
delivery. Prime correctly and the identical physical event — same
turbulence, same question — lands as excitement, because surprisal stays
inside the promised band. **A bumpy landing is exciting instead of scary
iff the bump was in the promise.**

This system already implements the law twice, unrecognized:
- The **gate digest** (10-line distilled state, never raw markdown) is
  the stories on the way; the gate question is the landing. Distill
  wrong and the model meets the payload unprepared — the measured
  difference (naive gating fails, distilled separates 0.19/0.81) is the
  priming effect with numbers.
- The **presence gate** multiplies instead of thresholding — it fades a
  voice the system is not sure it characterized. Fading IS priming the
  listener's variance expectation: "hold this loosely."

The receiver-side resource being scheduled is ATTENTION — which is
already a track (CC91). Priming makes attention a quantity you SPEND
before the payload, not just detect during it.

## 4. The two experiments this adds

**E7 — The pedal swing, observable: pre-event rises in real speech.**
The score's INTENTION track is, in slackwater's own semantics, "what is
ABOUT to happen — pre-event propagation." If the kick-drum law holds for
speakers, significant moments (punchlines, interruptions, the turn in an
argument) are PRECEDED by measurable rises: energy/tempo/velocity
climbing into the moment, a beat of rest right before it (the swing),
then the event. Protocol: annotate ±3 s windows around 30 known beats
(punchlines in stand-up, the reveal in storytelling podcasts); test
whether VELOCITY/TEMPO in the pre-window exceed the clip's baseline, and
whether a pre-event rest occurs, more often than chance. Pass: pre-event
rise in ≥ 2 tracks before ≥ 60% of annotated moments. This would be the
first evidence that the score captures not just what happened but the
INTENT ARRIVING — the control signal, one tap upstream of the render.

**E8 — Prime-then-payload in agent output: landing quality.**
When the agent delivers a high-stakes message (an alarm, a disagreement,
a correction), does announcing the variance first ("this is a rough
landing") improve the operator's response — faster acknowledgment, fewer
clarifying rounds, better recall of the content — versus delivering the
identical payload unprimed? Protocol: within-subject, 2 conditions × 20
messages via the Telegram leg; measure acknowledgment latency, follow-up
question count, and a 1–5 "how did that land" rating. Pass: primed
condition better on ≥ 2 of 3. This turns "bumpy is exciting when primed"
from metaphor into a measurable property of the agent's voice.

## 5. Three laws, stated for the wall

1. **Tap where the intent is.** Listening is inversion (recover the
   control signal from the render — and admit when it isn't there).
   Speaking is forward (score first, synthesis last mile).
2. **Schedule against T-minus, not T-zero.** Every stage starts early by
   the sum of the latencies below it. The pedal is in swing before the
   downbeat, or the beat is lost no matter how good the drum.
3. **Calibrate the receiver's variance promise.** Prime the landing so
   delivered surprisal stays inside promised surprisal — then bumps are
   excitement. Same event, different promise, different world.

---

**Addendum 2026-10-18 — the law already runs, and it has numbers.**
`tminus-dispatcher`/`tminus-client` (this fleet, npm) implement cue-
primed-fire with 500 ms cognitive beats and negative-n pre-cues — §2's
T-minus scheduling, deployed. The published precedents now in PAPERS.md
§13: pro drummers strike ahead of the click by measurable margins (Fujii
et al. 2011), negative mean asynchrony is the normal regime of skilled
performance (Repp & Su 2013), and end-of-turn anticipation with
speculative execution cuts dialogue latency by ~500 ms in published work
(Udupa et al. 2026, arXiv:2606.13450) — that paper is T-minus scheduling
for speech, independently invented. What the fleet adds over all of them:
the score, so the offset can be scheduled PER MODALITY (judged tracks
200–550 ms vs acoustic tens of ms) rather than one global lookahead.
