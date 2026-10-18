# The music/time fleet — synergy map for meaning, intent, timing

2026-10-18 inventory of the fleet's audio/MIDI/timing repos and how they
combine. Read with `docs/research/TIMING.md` (the theory) — several of
these repos implement its laws without ever having named them.

## The repos

**tminus-dispatcher + tminus-client** (npm, live) — the timing theory as
RUNNING INFRASTRUCTURE. A cue-primed-fire protocol: agents get T-minus
countdown cues (in 500 ms "cognitive beats"), PRIME (acknowledge ready),
then FIRE together — explicitly because "you can't send go and hope."
Positive n = countdown, zero = now, NEGATIVE n = pre-cue (already primed).
This is TIMING.md's kick-drum law and T-minus scheduling, deployed. What
communicator adds back: measured latencies (judged tracks 200–550 ms) are
the per-agent T-minus offsets the dispatcher needs to schedule real
cognitive work, not just action gates.

**midi-temporal** (seed, 148 LOC) — the SAME thesis as communicator,
stated independently: "MIDI is a protocol for triggers with parameters,
fired in time... velocity is emphasis, pitch bend is doubt, urgency,
warmth... 16 tracks = 16 characters in a room," plus the training-system
idea (grab open-source lectures, learn the cadence of great speakers —
which is EXPERIMENTS-AUDIO.md's corpus plan). Convergence is evidence the
thesis is load-bearing. Merge path: midi-temporal's 16-dimension framing
is communicator's nine-plus-seven problem (design doc risk 4) — the extra
seven tracks should be designed there, against its "commentary as
first-class MIDI events" idea.

**voice-games** (live on Pages) — COLOR SQUAD's reflex loop, the
voice-commanded-game lineage the design doc inherits (Router, Prism→Score
substitution). The player-facing end of the same representation.

**slackwater-tools / slack-water** — CORRECTED (audit): the local
slackwater-tools repo is the roundtable/ideation extraction from
lucineer-system, NOT the perception/substrate code. The
slackwater-perception 9-track lineage our CC assignments adopt lives in
the design doc's source table and stays archived on the lucineer-system
branch — no local checkout exists.

**Design-doc siblings** (fleet-audio's SPSC ring, constraint-synth's
64-sample stitch, snapkit's deadband+NaN-latch, persona-engine's
CharacterParameters schema, elephant's dials objective) — catalogued with
verdicts in the design doc §9; nothing here changes those.

## The combination, one sentence

communicator renders a person as a score; tminus schedules the fleet
against the moment; midi-temporal wants dimensions beyond music;
voice-games is the play surface; the research corpus (PAPERS.md §13)
supplies the published precedents — VAP and endpoint-anticipation for the
intention track, MIDI-DDSP for the hierarchy, Repp/Fujii for the offsets.

## Concrete next welds

1. tminus-dispatcher consuming MEASURED latencies per agent (ours are in
   PROGRESS.md Phase 2 (214–554 ms, jev-1.14.0)) instead of fixed beats — the calibration loop
   between the two repos.
2. midi-temporal's lecture-cadence training system run on the
   EXPERIMENTS-AUDIO corpus — its idea, our data.
3. voice-games as E5's rendering surface candidate (score → game audio
   feedback → human recognition).
