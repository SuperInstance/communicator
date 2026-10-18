"""
turn -- when did the speaker stop, as distinct from when did the voice stop.

THIS IS A CONSUMER, NOT A FILTER, AND THAT IS THE WHOLE DESIGN.

Two different questions, with different evidence and different costs of
being wrong:

    "did the voice stop?"        energy segmentation, `MIN_SILENCE_S` = 150 ms
    "is the turn over?"          this module, `TURN_END_S` = 3 s

The adoption plan wanted to merge them into one constant at 200 ms. Measured
across seven cells of real speech (465 s, 465 gaps) that costs 46 pauses,
17.8% of them all, concentrated in dramatic and children's speech where the
pause structure IS the prosody. See `tests/test_pause_threshold.py` and the
retraction in `docs/research/ADOPTION-PLAN.md` §3.

So the 3 s decision sits ON TOP of the pause stream and never removes
anything from it. A pause that does not reach the turn threshold is still a
pause: it still counts, it still divides TEMPO, it still appears in the §4.1
baseline table. The turn decision only adds information, which is the only
safe direction for a stage that sits downstream of everything.

THE MEASUREMENT THAT SHAPED THIS MODULE, AND IT IS UNCOMFORTABLE.

Over the whole corpus, pauses long enough to end a turn at 3 s:

    cell          audio_s   max_gap   gaps >= 3 s
    narration        90.0     1.980              0
    poetry           20.0       -                -
    essay            75.0     4.530              2
    dialogue         75.0     1.130              0
    shakespeare       75.0     2.100              0
    children          75.0     1.960              0
    oration           75.0     0.380              0

**Five of six cells produce ZERO turn boundaries at 3 s.** People recorded
monologue simply do not pause for three seconds; the longest gap anywhere in
the corpus outside the essay is 2.1 s.

That is not a defect in the constant. A 3 s give-up is an INTERACTIVE rule --
"nobody has spoken for three seconds, I may speak now" -- and it is correct
for that. It is wrong as a claim about where turns END in a recording, and
this corpus is recordings. So:

  * `TURN_YIELD_FLOOR_S` records how long the stream must run before a turn
    decision is expected to be available at all. Below it, `TurnTracker` says
    `UNRESOLVED` rather than `COMPLETE`. A peripheral that never speaks is
    also a failure, and the honest answer here is "not enough evidence", not
    "the turn ended".
  * `note_external()` still works. A refusal, an approval or a correction has
    no acoustic signature and still closes a turn immediately. A system that
    can only see silence will misread every human decision as a long pause.

WHAT "COMPLETE" IS NOT.

It is not "the speaker is silent". `vad.py` measures that gate separately,
and it measures that P(complete) from a waveform classifier is 0.987 on
digital silence -- so a silence-based completion test is a test that always
passes. The difference this module makes is that it requires sustained
silence AND an absence of human decisions AND enough elapsed time to be
credible, and it reports which of those was the binding constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# How long the voice must be absent before the turn is treated as over.
#
# pipecat's measured practice is ~200 ms to trigger and 3 s to give up; the
# 3 s is this constant and the 150 ms stays where it is. The gap between them
# is the design: the trigger is about segmentation and is measured to be
# correct against real speech, and this one is about when to stop waiting.
TURN_END_S = 3.0

# Below this much elapsed time, no turn decision is reported.
#
# Not a tuning knob: the corpus's longest inter-utterance gap outside a single
# essay is 2.1 s, so on 3 s of audio there is no evidence that could support
# either answer. Reporting COMPLETE there would be a coin flip presented as a
# measurement, which is the failure mode this repo keeps re-learning.
TURN_YIELD_FLOOR_S = 5.0

COMPLETE = "complete"
OPEN = "open"
UNRESOLVED = "unresolved"

STATES = (COMPLETE, OPEN, UNRESOLVED)


@dataclass
class TurnVerdict:
    """
    What the tracker currently believes, and why.

    `reason` is not decoration. "Complete" for four different reasons is four
    different situations, and a caller that acts on the state without reading
    the reason will eventually act on the wrong one.
    """

    state: str
    reason: str
    since: float = 0.0          # absolute time the state was reached
    elapsed_s: float = 0.0

    def __post_init__(self):
        if self.state not in STATES:
            raise ValueError(f"unknown turn state {self.state!r}; "
                             f"expected one of {STATES}")

    @property
    def settled(self) -> bool:
        """True only when a decision is actually supported by evidence."""
        return self.state == COMPLETE


@dataclass
class TurnTracker:
    """
    Consumes PAUSE and human-decision events; emits turn boundaries.

    It never removes, filters, or rewrites an event. `feed` returns a list of
    newly-decided turn boundaries and nothing else.

        t = TurnTracker()
        for ev in watcher_events:
            for turn in t.feed(ev, elapsed_s=watcher.elapsed_s):
                ...                       # a turn ended here

    `speaking` is set by an UTTERANCE or ONSET and cleared by a PAUSE, so the
    tracker knows whether a long silence followed speech (a turn ending) or
    merely followed nothing (idle time, which ends nothing).
    """

    turn_end_s: float = TURN_END_S
    yield_floor_s: float = TURN_YIELD_FLOOR_S
    # Time a turn has been open with no human decision in it.
    _silence_since: float | None = None
    _speaking: bool = False
    _human_decision_at: float | None = None
    turns: list = field(default_factory=list)

    # -- input ---------------------------------------------------------------

    def feed(self, ev, elapsed_s: float) -> list:
        """
        One event in, any newly-decided turns out.

        Returns an empty list almost always. That is the normal case and it is
        not a failure: a tracker that fires often is a tracker guessing.

        `elapsed_s` is the stream clock, and it is what lets the tracker know
        silence is accumulating even when no PAUSE event exists yet. A pause
        is only emitted once its segment CLOSES behind the tail cut, so a
        still-open trailing silence produces no event at all -- 1.0 s of
        speech followed by 1.5 s of silence emits an onset and an utterance
        and nothing else. A tracker that only learned from pause events would
        therefore believe the speaker was still talking forever, which is the
        failure mode that matters most here: a peripheral that never knows
        when to speak.
        """
        kind = getattr(ev, "kind", None)
        if kind in ("onset", "utterance"):
            self._speaking = True
            # Sound was heard until this event ENDED, so silence has been
            # accumulating since `ev.t1` -- not since a pause event that may
            # never arrive. A 1.0 s utterance followed by 1.5 s of silence
            # emits an onset and an utterance and nothing else, so a tracker
            # that only learned from pause events would believe the speaker
            # was still talking forever.
            self._silence_since = getattr(ev, "t1", elapsed_s)
        elif kind == "pause":
            # Only a pause that FOLLOWS speech can end a turn. A long silence
            # in an empty room is not the end of anyone's turn, and treating it
            # as one is how a peripheral decides to talk to nobody.
            #
            # The `_silence_since is None` guard belongs on the ASSIGNMENT, not
            # on the call. It used to guard both, which meant the first pause
            # set the start and every LATER pause of the same run skipped
            # `_decide` entirely -- so the run that finally exceeded 3 s was
            # exactly the one that could not be seen. A 6 s silence produced
            # run_s values of 2.755, 5.555 and 5.940 and zero turns, because
            # the 5.555 arrived when `_silence_since` was already set.
            if not self._speaking:
                return []
            if self._silence_since is None:
                self._silence_since = ev.t0
            return self._decide(ev)
        elif kind == "decision":
            self._human_decision_at = getattr(ev, "t0", elapsed_s)

        # Silence is accumulating even if no pause has closed yet. Deciding
        # from the clock here is what makes a trailing silence visible; the
        # turn is emitted with `open=True` because the speaker may yet resume.
        self._observe_clock(elapsed_s)
        return []

    def _observe_clock(self, elapsed_s: float) -> list:
        """
        Decide from elapsed time alone. Used for silence that has not yet been
        reported as an event.

        Distinct from `_decide`, which trusts the pause's own measured run.
        This one is a fallback and is marked `source="clock"` so a caller can
        tell a clock-inferred boundary from a measured one.
        """
        if not self._speaking or self._silence_since is None:
            return []
        silence_for = elapsed_s - self._silence_since
        if silence_for < self.turn_end_s:
            return []
        if elapsed_s < self.yield_floor_s:
            return []
        turn = {
            "t0": self._silence_since,
            "t1": elapsed_s,
            "silence_s": round(silence_for, 4),
            "reason": "sustained_silence",
            "human_decision": False,
            "open": True,
            "source": "clock",
        }
        self.turns.append(turn)
        self._speaking = False
        self._silence_since = None
        return [turn]

    def _decide(self, ev) -> list:
        """
        Called when a pause arrives. The silence measured is the PAUSE RUN's
        length -- not the time since it was seen, and not this slice's length.

        Two earlier bugs, both of which the real corpus found immediately.

        1. It compared `elapsed_s - _silence_since`, where `elapsed_s` was the
           watcher's clock at push time, so every pause was measured against
           the moment it was observed rather than its own extent. On 75 s of
           the essay that produced 94 turn boundaries instead of 2, all with
           `t1` equal to the end of the stream.

        2. It used the pause EVENT's own duration, which is never long: a pause
           is only emitted once its segment closes behind the tail cut, and
           `MAX_PAUSE_S` flushes an open one at 3.0 s, so a 6 s silence
           arrives as 2.755 / 2.775 / 0.360 s and every slice is below any
           useful threshold. The watcher's `run_s` field is the accumulated
           length of the whole silence run, which is the quantity a turn
           decision is about.

        3. It waited for the run to CLOSE. A 6 s silence is 2.755 / 2.775 /
           0.360 s and the last of those is short, so a tracker that only
           fires on a closed pause sees no pause over 3 s at all -- the run
           crossed the threshold and then kept going. An OPEN checkpoint whose
           `run_s` already clears the threshold IS the turn ending: the
           evidence exists, and waiting for it to be re-reported in a
           shorter final slice is waiting for information that has already
           arrived.

        `run_s` is used when present and falls back to the event's own extent,
        so the tracker also works on events from a watcher that predates it.
        """
        if self._silence_since is None:
            return []
        own = getattr(ev, "t1", self._silence_since) - getattr(ev, "t0", 0.0)
        silence_for = float(ev.data.get("run_s", own)) if hasattr(
            ev, "data") else own
        if silence_for < self.turn_end_s:
            return []
        if getattr(ev, "t1", 0.0) < self.yield_floor_s:
            # Enough silence to look like an end, not enough elapsed time for
            # the answer to mean anything. Said out loud rather than guessed.
            return []
        turn = {
            "t0": self._silence_since,
            "t1": ev.t1,
            "silence_s": round(silence_for, 4),
            "reason": "sustained_silence",
            "human_decision": False,
            # Whether the silence had finished when this was decided. An open
            # turn is a real state: the speaker may resume, and a caller
            # reacting to it has to be able to void the reaction.
            "open": bool(getattr(ev, "data", {}).get("open", False)),
        }
        self.turns.append(turn)
        # The turn is decided either way. If the speaker resumes, the next
        # utterance opens a new turn; if they do not, `verdict()` keeps
        # reporting the state without re-emitting.
        self._speaking = False
        self._silence_since = None
        return [turn]

    def note_external(self, t0: float, elapsed_s: float) -> list:
        """
        A human decision -- a refusal, an approval, a correction.

        It has no acoustic signature, so no amount of silence analysis will
        ever find it. This is the only way a turn can end for a reason other
        than silence, and a tracker that cannot represent it will misread
        every one of the most consequential moments in a session as a long
        pause that happened to land at the same time.
        """
        self._human_decision_at = t0
        if self._speaking:
            turn = {
                "t0": self._silence_since if self._silence_since is not None
                      else t0,
                "t1": t0,
                "silence_s": round(max(0.0, t0 - (self._silence_since
                                                   or t0)), 4),
                "reason": "human_decision",
                "human_decision": True,
            }
            self.turns.append(turn)
            self._speaking = False
            self._silence_since = None
            return [turn]
        return []

    # -- state ---------------------------------------------------------------

    def verdict(self, elapsed_s: float) -> TurnVerdict:
        """
        What is believed right now, and which constraint is binding.

        The three reasons are not interchangeable:

          * `too_little_time`   -- below the yield floor. Not enough evidence.
          * `idle`              -- nobody has spoken yet. An empty room.
          * `still_speaking`    -- sound is still arriving. Nothing to decide.
          * `silence_short_of_threshold` -- waiting. The expected state most
            of the time, and not an error.
        """
        if elapsed_s < self.yield_floor_s:
            return TurnVerdict(UNRESOLVED, "too_little_time", elapsed_s=elapsed_s)
        if self._silence_since is None:
            # No speech has been seen at all: an idle room, not an open turn.
            return TurnVerdict(OPEN, "idle", elapsed_s=elapsed_s)
        silence_for = elapsed_s - self._silence_since
        if self._speaking and silence_for <= 0:
            # Sound is still coming; `ev.t1` is at or ahead of the clock.
            return TurnVerdict(OPEN, "still_speaking", elapsed_s=elapsed_s)
        if silence_for >= self.turn_end_s:
            return TurnVerdict(COMPLETE, "sustained_silence",
                               since=self._silence_since, elapsed_s=elapsed_s)
        return TurnVerdict(OPEN, "silence_short_of_threshold",
                           since=self._silence_since, elapsed_s=elapsed_s)


def end_to_end_s(turn: dict) -> float:
    """
    The number a latency budget actually wants: how long from the last sound
    to the decision.

    NOT the turn's own duration. A turn that lasted 20 s of speech and ended
    on a 3 s pause has a 3 s end-to-end cost, and a system measuring "turn
    length" to bound its own responsiveness would be measuring the wrong
    quantity by an order of magnitude.
    """
    return float(turn["silence_s"])