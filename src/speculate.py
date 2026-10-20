"""
speculate -- act before you are sure, and be able to take it back.

WHY THIS EXISTS, WITH THE NUMBER THAT FORCED IT.

Across the seven-cell real corpus the 3 s turn decision in `turn.py` fires
on ZERO of the 673 distinct silence runs measured on the population this
number was taken from — or on ZERO of the 1671 pause EVENTS the watcher
reports over them, since it re-reports an open pause every 0.25 s. The
longest gap anywhere in that corpus outside a single essay is
2.4 s. So a peripheral built on it does not respond slowly — it does not
respond at all, and "wait for certainty" has a measured cost of "never".

TWO NUMBERS HERE ARE STALE, and they are stale in the corpus, not in the
mechanism. The cells on disk total 485.01 s, not the 465 s quoted in earlier
revisions, and the code now measures 673 silence runs (1671 pause events)
over those cells rather than the 680 runs this statistic was taken from.
The 680 is not exactly reproducible, because the population behind it no
longer exists: the same measurement gives 673 runs adaptively, 647 events
energy-only, and 665 runs on a hypothetical 465 s corpus — so truncating
poetry to 15 s moves the count AWAY from 680, not towards it. The
conclusion -- the 3 s
threshold fires on none of them -- is unaffected, because 3 s exceeds the
longest gap either way. `docs/research/NOTES-AUDIT.md` F7, F8.

At the same time, `docs/research/TURN-ROUTES.md` established that the
decision cannot be made better by looking harder:

    our duration rule   end 0.0064   resume 0.0067   <- byte-identical inputs
    smart-turn v3.2     end 0.6142   resume 0.4784   <- p = 0.0197, uncalibrated

A waveform classifier separates "finished" from "about to continue" weakly and
inconsistently. No route available today can tell them apart reliably at the
moment of decision, because at that moment **the audio is identical** -- the
only difference is speech that has not happened yet.

So the honest response is not a better classifier. It is to accept that every
turn decision is a BET, and to make the bet cheap and reversible.

THE THREE STATES, AND WHY EACH IS SEPARATE.

    OPEN       silence is accumulating, nothing claimed yet. Free to stop.
    HOLDING    a reaction has STARTED on the bet -- audio rendered, a plan
               built, tokens spent. This is the state that costs money, and
               the only state where being wrong is expensive.
    DROPPING   the bet lost: the speaker resumed. The reaction must be torn
               down and whatever it produced must be marked unusable.

Committing is a separate, later event. The lifecycle is:

    predict -> void -> commit

`void` must be possible at any point up to `commit`, and after `commit` the
speculation becomes a real turn with `voided=False` and no way to pretend
otherwise. Pretending a committed action was speculative is how a peripheral
lies to its own tables.

WHY THE THRESHOLDS ARE TWO AND NOT ONE.

    speculate_s   act on the bet this early. Costs money when wrong.
    commit_s      treat it as settled this late. Measured from the corpus,
                  never below the longest real pause in any cell.

The gap between them is the entire design. If they are equal there is no
speculation, only delay. If `speculate_s` is below the shortest real pause
(0.11 s measured) then EVERY pause is a bet and the mechanism is theatre.

`speculate_s` is NOT a constant this module gets to invent. It is measured
from data, and the first value I wrote -- 1.20 s -- was theatre:

    676 real pause runs, min 0.160 s, median 0.320 s, max 2.440 s

    speculate_s   reach rate   void rate
        0.30 s       53.8%       46.2%
        0.60 s       21.9%       78.1%
        0.90 s        9.9%       90.1%
        1.20 s        3.1%       96.9%   <- the default I first wrote
        1.80 s        0.6%       99.4%
        2.40 s        0.1%       99.9%

At 1.20 s, **96.9% of speculations would be voided**. The gate would fire
almost never and be interrupted almost immediately when it did. It would look
like it worked, because nothing would go wrong -- there would be nothing to
go wrong.

0.30 s is where speculation actually does something: it sits just under the
0.320 s median run, so roughly half the pauses in real speech produce a bet
and half void. That is not a comfortable number. It is the number, and the
mechanism only earns its place at values where it fires.

WHAT "VOID" ACTUALLY MEANS HERE, PRECISELY.

Not "we were wrong". It means: a reaction that was started on a bet has been
stopped, and everything it produced is marked `speculative=True, voided=True`.
Anything the reaction produced that has left the process -- speech already
audible, a message already sent -- CANNOT be recalled, and this module says
so rather than pretending otherwise. A void is a claim about internal state,
not about the world.

WHAT THIS DOES NOT FIX.

It does not make the decision more accurate; `speculation_accuracy()` reports
the measured rate and it will be well below 1.0, because the evidence says it
must be. It trades a bounded, retractable cost for latency, and the trade is
only worth making if the reaction is genuinely retractable. Acting
irreversibly on a speculation is the failure this module exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

OPEN = "open"
HOLDING = "holding"
DROPPING = "dropping"
COMMITTED = "committed"
VOIDED = "voided"

STATES = (OPEN, HOLDING, DROPPING, COMMITTED, VOIDED)


@dataclass
class Speculation:
    """
    One bet on one turn.

    `state` is the only thing a caller needs to branch on, and the transitions
    are one-way except for HOLDING -> DROPPING -> VOIDED. There is no path
    back from VOIDED to COMMITTED: a voided speculation cannot be un-voided,
    and a system that can do that has a bug in its state machine.
    """

    id: int
    t0: float                       # when the silence began
    state: str = OPEN
    speculated_at: float | None = None
    committed_at: float | None = None
    voided_at: float | None = None
    # Whether the action taken on this bet touched the outside world. Set by
    # the caller, because only the caller knows. A speculation that already
    # spoke cannot be retracted, and pretending otherwise is the lie this
    # whole module exists to avoid.
    externally_visible: bool = False

    def __post_init__(self):
        if self.state not in STATES:
            raise ValueError(f"unknown state {self.state!r}; "
                             f"expected one of {STATES}")

    @property
    def live(self) -> bool:
        """True while the bet can still be changed."""
        return self.state in (OPEN, HOLDING)

    @property
    def settled(self) -> bool:
        """True once it can no longer change."""
        return self.state in (COMMITTED, VOIDED)


@dataclass
class SpeculationGate:
    """
    Turns silence into a reversible bet.

        g = SpeculationGate()
        for ev in watcher_events:
            for s in g.feed(ev, elapsed_s=watcher.elapsed_s):
                ...                # a bet opened / holding / committed / voided

    Nothing here reaches the outside world. `feed` returns state changes and
    the caller decides what to do with them -- which is deliberate, because a
    gate that could speak for you would be a gate that could interrupt
    someone irreversibly on a coin flip.
    """

    speculate_s: float = 0.30
    commit_s: float = 3.0
    next_id: int = 0
    live: dict = field(default_factory=dict)      # id -> Speculation
    history: list = field(default_factory=list)
    # Counters, exposed because a gate nobody audits is a gate nobody trusts.
    opened: int = 0
    held: int = 0
    committed: int = 0
    voided: int = 0
    # The silence each bet was opened on, kept so speculation_accuracy() can be
    # computed without re-running the session.
    _opened_at_silence: dict = field(default_factory=dict)

    # -- input ---------------------------------------------------------------

    def feed(self, ev, elapsed_s: float) -> list:
        kind = getattr(ev, "kind", None)

        if kind in ("onset", "utterance"):
            return self._on_speech(ev, elapsed_s)
        if kind == "pause":
            return self._on_pause(ev)
        if kind == "decision":
            return self._on_decision(ev)
        return []

    def _on_speech(self, ev, elapsed_s: float) -> list:
        """
        The speaker came back. Every live bet is now wrong.

        This is the whole point: a bet that cannot be voided is not a bet, it
        is a guess with consequences, and guessing whether someone has
        finished speaking has the worst possible consequence -- interrupting
        them.
        """
        out = []
        for sid in list(self.live):
            s = self.live[sid]
            if s.state in (OPEN, HOLDING):
                s.state = DROPPING
                s.voided_at = ev.t0
                out.append(self._settle(s))
        return out

    def _on_pause(self, ev) -> list:
        run = float(ev.data.get("run_s", 0.0)) if hasattr(ev, "data") else 0.0
        if run <= 0:
            return []
        out = []
        had_live = bool(self.live)
        # commit anything already speculating that has now cleared the bar
        for sid in list(self.live):
            s = self.live[sid]
            if run >= self.commit_s:
                s.state = COMMITTED
                s.committed_at = ev.t1
                out.append(self._settle(s))
        # Open a new bet only if this pause did NOT just settle one.
        #
        # The check used to be `not self.live`, evaluated AFTER the commit loop
        # had emptied it -- so the very event that committed a bet also opened
        # a second one ON THE SAME SILENCE, which was then voided by the speech
        # that ended the first. The gate reported a void on a pause it had just
        # committed. The silence has already been bet on; betting on it again
        # is not a second opinion, it is a double charge.
        if run >= self.speculate_s and not had_live:
            self.next_id += 1
            s = Speculation(id=self.next_id, t0=ev.t0)
            s.speculated_at = ev.t1
            # A silence that already clears the COMMIT bar is not a
            # speculation -- it is a decision, and it opens already settled.
            # Left in HOLDING it could never commit, because nothing longer is
            # coming, and the next utterance would void a turn that had in
            # fact been reached.
            settled = run >= self.commit_s
            self.opened += 1
            self._opened_at_silence[self.next_id] = run
            if settled:
                s.state = COMMITTED
                s.committed_at = ev.t1
                self.committed += 1
                self.history.append(s)
            else:
                s.state = HOLDING
                self.live[self.next_id] = s
                self.held += 1
            out.append(self._transition(s))
        return out

    def _on_decision(self, ev) -> list:
        """
        A human decision settles the turn regardless of silence.

        It is also the case where a speculation is most valuable to void: the
        person answered, so the system about to react to a pause was wrong
        about who had the floor.
        """
        return self._on_speech(ev, ev.t1)

    def _transition(self, s: Speculation) -> dict:
        return {"id": s.id, "state": s.state, "t0": s.t0,
                "speculated_at": s.speculated_at,
                "externally_visible": s.externally_visible}

    def _settle(self, s: Speculation) -> dict:
        self.live.pop(s.id, None)
        if s.state == COMMITTED:
            self.committed += 1
        elif s.state == DROPPING:
            s.state = VOIDED
            self.voided += 1
        self.history.append(s)
        return self._transition(s)

    # -- accounting ----------------------------------------------------------

    def speculation_accuracy(self) -> float | None:
        """
        Of the bets that reached a decision, how many were right.

        Returns None when nothing settled, because a ratio over zero is a
        number that means nothing and reporting 0.0 or 1.0 would both be lies.
        """
        decided = [s for s in self.history if s.settled]
        if not decided:
            return None
        return self.committed / len(decided)

    def irreversible(self) -> list:
        """
        Bets that were acted on externally and then voided.

        These are the ones that cost something real, and a caller that wants
        to know how bad its speculation actually was looks here rather than
        at the accuracy ratio.
        """
        return [s for s in self.history
                if s.state == VOIDED and s.externally_visible]

    def summary(self) -> dict:
        acc = self.speculation_accuracy()
        return {
            "opened": self.opened,
            "committed": self.committed,
            "voided": self.voided,
            "accuracy": None if acc is None else round(acc, 4),
            "irreversible_voids": len(self.irreversible()),
            "speculate_s": self.speculate_s,
            "commit_s": self.commit_s,
        }


def yield_curve(pause_runs, speculate_grid=(0.30, 0.60, 0.90, 1.20, 1.50,
                                            1.80, 2.10, 2.40)):
    """
    Measure what speculation would have cost, from real pause runs alone.

    `pause_runs` is a list of silence-run durations in seconds, taken from the
    corpus. This cannot know whether a speaker resumed -- that is exactly the
    information nobody has -- so it reports the two things that ARE knowable:

      void rate     fraction of pauses below the threshold, i.e. bets that
                    would have been voided
      reach rate    fraction at or above it, i.e. bets that would have stuck

    A threshold is only worth choosing if it sits where the reach rate is
    already high. Below the shortest real pause -- 0.11 s, measured across
    seven cells -- every threshold voids everything and the mechanism is
    theatre. That is the check this function exists to make.
    """
    if not pause_runs:
        return []
    out = []
    for t in speculate_grid:
        below = sum(1 for r in pause_runs if r < t)
        out.append({
            "speculate_s": t,
            "n": len(pause_runs),
            "would_void": below,
            "would_reach": len(pause_runs) - below,
            "void_rate": round(below / len(pause_runs), 4),
            "reach_rate": round((len(pause_runs) - below) / len(pause_runs), 4),
        })
    return out