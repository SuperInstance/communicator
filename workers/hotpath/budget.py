"""
Neuron budget accounting. Pure Python, no Cloudflare dependency, no I/O.

This is the piece that makes it safe to run unattended. Cloudflare's free
allowance does not degrade gracefully when you cross it: on Workers Free,
"if you exceed any one of the above limits, further operations will fail with
an error" (pricing page). And on Paid the overage is simply billed. Neither
behaviour tells you WHICH call pushed you over. A ledger that knows, refuses,
and says so is therefore not bookkeeping -- it is the safety property.

Three deliberate design choices, each load-bearing:

1.  RESERVE-THEN-SETTLE, never pay-as-you-go.
    Cost is known before a clef-omni call only by ESTIMATING tokens, because
    tokens are billed on the way back. If we charged on response, N concurrent
    calls would all observe the same "affordable" balance and collectively
    overdraw. So every expensive call reserves worst-case cost up front and
    reconciles against the reported `usage` afterwards. Over-reserving costs
    us a little throughput; under-reserving is the failure we exist to prevent.

2.  ACCOUNT-WIDE, NOT PER-SESSION.
    Neurons are billed to the ACCOUNT, so a ledger kept per session lets N
    sessions each believe they hold the entire 10,000/day allowance and
    overdraw by a factor of N. The ledger therefore has exactly one home: a
    single Durable Object that every session reports to. See budget_do.js.

3.  FAIL LOUD, AND ON OVERRUN FAIL CLOSED.
    If a settled call cost more than it reserved, that means our constants
    are wrong -- a model repriced, or a clip was bigger than believed. The
    ledger records the overrun and then REFUSES further expensive calls until
    the daily rollover. Silently re-baselining is the behaviour that turns a
    wrong constant into an unbounded bill.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

COSTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "costs.json")


class BudgetError(RuntimeError):
    """Base class. Anything raised here is a refusal, never a soft warning."""


class BudgetExhausted(BudgetError):
    """
    The ledger will not pay for another call of this model right now.

    Carries the numbers that caused the refusal so the refusal is actionable
    rather than opaque: a caller that logs only "BudgetExhausted" has learned
    nothing it could not have guessed.
    """

    def __init__(self, model: str, needed: float, available: float, reason: str):
        self.model = model
        self.needed = needed
        self.available = available
        self.reason = reason
        super().__init__(
            f"refusing {model}: needs {needed:.2f} neurons, "
            f"{available:.2f} available ({reason})"
        )


class UnknownModel(BudgetError):
    """A cost was quoted for a model absent from costs.json.

    Deliberately fatal. Defaulting an unknown model to zero cost would be
    exactly the fail-open behaviour this module exists to prevent.
    """


class LedgerCorrupt(BudgetError):
    """Persisted ledger state could not be understood."""


# --------------------------------------------------------------------------
# Costs: the one place prices become numbers
# --------------------------------------------------------------------------

class Costs:
    """
    Typed view over costs.json.

    The JSON is the single source of truth and is read by the JavaScript
    Workers too, so a price change is one edit rather than four.
    """

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.neuron_usd: float = raw["neuron"]["usd_per_neuron"]
        self.free_neurons_per_day: float = raw["neuron"]["free_neurons_per_day"]
        self.models: dict[str, Any] = raw["models"]
        self._ours: dict[str, Any] = raw["OURS"]
        # Pre-resolve the effective rate per model once; see `_rate`.
        self._rates: dict[str, float] = {
            name: self._rate(name) for name in self.models
        }

    @classmethod
    def load(cls, path: str = COSTS_PATH) -> "Costs":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def _rate(self, model: str) -> float:
        """
        Neurons per billed unit, taking the MORE EXPENSIVE of the two numbers
        Cloudflare publishes.

        Every model on the pricing page states both a USD figure and a neuron
        figure, and the page claims they are equivalent. They are equivalent
        for eight of the nine priced models but disagree by 60x for
        smart-turn-v2. Taking the max means that when the two disagree we plan
        against the worse case. Under-estimating by 60x is the difference
        between "the ledger refuses" and "the ledger believes it has 60x the
        money it has", so the direction of the error is the whole decision.

        The rule is uniform across models rather than a special case for the
        one slug that disagrees, so it keeps meaning if Cloudflare re-prices
        something else tomorrow.

        Note there is no per-million scaling here. Both published figures are
        quoted against the SAME unit -- "$0.15 per M input tokens" and
        "13,636 neurons per M input tokens" -- so the "per million" cancels in
        the ratio and the conversion is simply usd / usd_per_neuron. Dividing
        by a million again here would understate a token-priced model by 10^6,
        and the `max()` would silently hide it by always picking the published
        figure. The unit name matters for interpreting how many `units` a
        caller passes, not for converting the rate.
        """
        spec = self.models.get(model)
        if spec is None:
            raise UnknownModel(f"no cost entry for model {model!r}")
        if spec["usd_per_unit_unit"] not in ("audio-minute", "M-input-tokens"):
            raise LedgerCorrupt(
                f"unknown pricing unit {spec['usd_per_unit_unit']!r} for {model}")
        implied = float(spec["usd_per_unit"]) / self.neuron_usd
        published = float(spec["published_neurons_per_unit"])
        return max(published, implied)

    def neurons_per_unit(self, model: str) -> float:
        """Effective neurons per billed unit (see `_rate`)."""
        if model not in self._rates:
            # Explicitly UnknownModel, not an incidental KeyError: a caller
            # asking about an unpriced model needs to be told that the model is
            # unknown, which is a different (and much more serious) problem
            # than a malformed lookup.
            raise UnknownModel(f"no cost entry for model {model!r}")
        return self._rates[model]

    def usd(self, neurons: float) -> float:
        return neurons * self.neuron_usd

    def ours(self, key: str) -> Any:
        """One of our own tunables, by name."""
        return self._ours[key]["value"]

    def note(self, key: str) -> str:
        return self._ours[key]["rationale"]


@dataclass(frozen=True)
class Quote:
    """What a call will cost. Immutable, so it can be logged verbatim."""

    model: str
    units: float
    neurons: float
    unit_name: str
    usd: float

    def __str__(self) -> str:
        return (f"{self.model}: {self.units:.4f} {self.unit_name} -> "
                f"{self.neurons:.2f} neurons (${self.usd:.6f})")


@dataclass(frozen=True)
class Reservation:
    """
    A claim on the ledger, held until settled or released.

    Frozen on purpose: the reserved amount is what the ledger will reconcile
    against, and a caller that could mutate it after the fact could talk the
    ledger into releasing capacity it never gave away.
    """

    model: str
    neurons: float
    day: str
    reservation_id: str
    units: float = 0.0


@dataclass
class Ledger:
    """
    Mutable spend state for ONE UTC day.

    Split out from NeuronBudget so it can be serialised to Durable Object
    storage verbatim. It is deliberately small and flat: a Durable Object
    round-trips are billed, so the state must stay cheap to read and write.
    """

    day: str
    settled_neurons: float = 0.0      # measured, from reported usage
    reserved_neurons: float = 0.0     # claimed, awaiting settle
    calls: dict[str, int] = field(default_factory=dict)
    audio_minutes_measured: float = 0.0
    audio_minutes_reserved: float = 0.0
    refused: int = 0
    overruns: int = 0
    # Set when a settled call exceeded its reservation. While this is True the
    # ledger refuses all expensive calls -- see NeuronBudget.reserve.
    poisoned: bool = False
    poison_reason: str = ""

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "settled_neurons": self.settled_neurons,
            "reserved_neurons": self.reserved_neurons,
            "calls": dict(self.calls),
            "audio_minutes_measured": self.audio_minutes_measured,
            "audio_minutes_reserved": self.audio_minutes_reserved,
            "refused": self.refused,
            "overruns": self.overruns,
            "poisoned": self.poisoned,
            "poison_reason": self.poison_reason,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Ledger":
        try:
            led = cls(day=str(raw["day"]))
            led.settled_neurons = float(raw.get("settled_neurons", 0.0))
            led.reserved_neurons = float(raw.get("reserved_neurons", 0.0))
            led.calls = {str(k): int(v) for k, v in (raw.get("calls") or {}).items()}
            led.audio_minutes_measured = float(raw.get("audio_minutes_measured", 0.0))
            led.audio_minutes_reserved = float(raw.get("audio_minutes_reserved", 0.0))
            led.refused = int(raw.get("refused", 0))
            led.overruns = int(raw.get("overruns", 0))
            led.poisoned = bool(raw.get("poisoned", False))
            led.poison_reason = str(raw.get("poison_reason", ""))
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerCorrupt(f"unreadable ledger state: {exc}") from exc
        return led

    # -- capacity -----------------------------------------------------------

    @property
    def committed(self) -> float:
        """Everything the ledger has already promised, settled or not."""
        return self.settled_neurons + self.reserved_neurons


class NeuronBudget:
    """
    Daily neuron accounting against Cloudflare's allowance.

    One instance corresponds to one ACCOUNT for one run of time. In production
    exactly one of these exists, inside a Durable Object, because the allowance
    is per-account. Constructing one per session would be the bug this class
    exists to prevent.
    """

    def __init__(
        self,
        costs: Costs,
        *,
        paid: bool = False,
        now: datetime | None = None,
        max_paid_overage_neurons: float | None = None,
    ):
        self.costs = costs
        self.paid = paid
        self._now = now or datetime.now(timezone.utc)
        if max_paid_overage_neurons is None:
            max_paid_overage_neurons = float(
                costs.ours("max_paid_overage_neurons_per_day")
            )
        self.max_paid_overage_neurons = max_paid_overage_neurons
        self.ledger = Ledger(day=self.day_key(self._now))
        self._seq = 0

    # -- clock --------------------------------------------------------------

    def _utcnow(self) -> datetime:
        return self._now

    def set_now(self, moment: datetime) -> None:
        """Move the clock. Used by the rollover tests; production passes real time."""
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        self._now = moment.astimezone(timezone.utc)

    @staticmethod
    def day_key(moment: datetime) -> str:
        """
        The UTC day this instant falls in.

        Cloudflare resets at "00:00 UTC" (pricing page), so the key MUST be the
        UTC day. Keying on local time would double-count or double-refill
        depending on where the process happens to run, which is precisely the
        kind of bug that only appears in production.
        """
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).strftime("%Y-%m-%d")

    def rollover_if_due(self) -> bool:
        """
        Refill when the UTC day has turned over. Returns True if it did.

        Note what this does NOT carry over: reservations from yesterday. A
        reservation belongs to the day it was made; if a call settles after
        midnight its cost lands on the new day, and the old reservation simply
        ceases to exist with the old ledger. That is the honest reading --
        Cloudflare bills by when the call is made, and we cannot know the
        bill's own attribution rule for a straddling call.
        """
        today = self.day_key(self._utcnow())
        if today != self.ledger.day:
            self.ledger = Ledger(day=today)
            return True
        return False

    # -- costing ------------------------------------------------------------

    def quote(self, model: str, units: float) -> Quote:
        """Cost of `units` billed units of `model`, at the conservative rate."""
        if units < 0:
            raise BudgetError(f"negative units for {model}: {units}")
        rate = self.costs.neurons_per_unit(model)
        neurons = rate * units
        return Quote(
            model=model,
            units=units,
            neurons=neurons,
            unit_name=self.costs.models[model]["usd_per_unit_unit"],
            usd=self.costs.usd(neurons),
        )

    def quote_judgement(self, input_tokens: int) -> Quote:
        """Cost of one clef-omni judgement over `input_tokens` tokens."""
        return self.quote("clef-omni", input_tokens / 1_000_000.0)

    def quote_audio(self, audio_minutes: float) -> Quote:
        """Cost of endpointing `audio_minutes` of audio."""
        return self.quote("smart-turn-v2", audio_minutes)

    # -- capacity -----------------------------------------------------------

    def allowance(self) -> float:
        """Neurons payable today before overage begins."""
        return self.costs.free_neurons_per_day

    def headroom(self) -> float:
        """Neurons still available for new reservations."""
        self.rollover_if_due()
        return max(0.0, self.allowance() - self.ledger.committed)

    def would_overdraw(self, neurons: float) -> tuple[bool, str]:
        """
        Would spending `neurons` take us past what we permit today?

        Returns (would_overdraw, reason). The reason is always populated, so a
        refusal can be logged with its cause instead of a bare boolean.

        The two plans permit different things, and conflating them is a bug:

          * FREE  -- the 10,000/day allowance is the WHOLE budget. Beyond it
            there is no overage to bill, only the documented hard stop: "If you
            exceed any one of the above limits, further operations will fail
            with an error." So free must refuse at the allowance.

          * PAID  -- the same 10,000/day is INCLUDED and spending above it is
            legitimate, billed at the per-neuron rate. The only ceiling is our
            own overage cap. Refusing at the allowance on Paid would make the
            cap unreachable and Paid would buy nothing -- the system would stop
            at exactly the same point either way, at the cost of a plan.
        """
        if self.ledger.poisoned:
            return True, (f"ledger poisoned: {self.ledger.poison_reason}")
        projected = self.ledger.committed + neurons
        if not self.paid:
            if projected > self.allowance():
                return True, (
                    f"free-tier daily allowance {self.allowance():.0f} exhausted "
                    f"(committed {self.ledger.committed:.2f}); Cloudflare fails "
                    f"further calls past this point"
                )
            return False, "within free allowance"
        overage = max(0.0, projected - self.allowance())
        if overage > self.max_paid_overage_neurons:
            return True, (
                f"paid overage cap {self.max_paid_overage_neurons:.0f} neurons "
                f"(${self.costs.usd(self.max_paid_overage_neurons):.2f}) would be "
                f"exceeded (overage so far {max(0.0, self.ledger.committed - self.allowance()):.2f})"
            )
        return False, "within paid allowance plus overage cap"

    def can_afford(self, model: str, units: float) -> bool:
        """Non-mutating affordability check, for planning and for tests."""
        try:
            q = self.quote(model, units)
        except UnknownModel:
            raise
        over, _ = self.would_overdraw(q.neurons)
        return not over

    # -- committing ---------------------------------------------------------

    def reserve(self, model: str, units: float) -> Reservation:
        """
        Claim `units` of budget, or raise BudgetExhausted.

        Raises rather than returning a sentinel: a caller that forgets to check
        a boolean is exactly how a ledger gets overrun. Reservation is
        pessimistic -- it counts against the day immediately -- so that
        concurrent callers cannot each pass a check the others then invalidate.
        """
        self.rollover_if_due()
        q = self.quote(model, units)
        over, reason = self.would_overdraw(q.neurons)
        if over:
            self.ledger.refused += 1
            raise BudgetExhausted(model, q.neurons, self.headroom(), reason)

        self.ledger.reserved_neurons += q.neurons
        if self.costs.models[model]["usd_per_unit_unit"] == "audio-minute":
            self.ledger.audio_minutes_reserved += units
        self._seq += 1
        return Reservation(
            model=model,
            neurons=q.neurons,
            day=self.ledger.day,
            reservation_id=f"{self.ledger.day}-{self._seq}",
            units=units,
        )

    def reserve_judgement(self, input_tokens: int) -> Reservation:
        """Reserve one clef-omni judgement, sized by the OURS token estimate."""
        return self.reserve("clef-omni", input_tokens / 1_000_000.0)

    def settle(self, reservation: Reservation, actual_neurons: float) -> None:
        """
        Replace a reservation with what the call actually cost.

        `actual_neurons` should come from the model's reported usage, not from
        our own arithmetic -- that is the whole reason the measurement is fed
        back in. When the two disagree beyond floating-point noise the ledger
        records the overrun, sets `poisoned`, and from then on refuses
        expensive calls until rollover.
        """
        if reservation.day != self.ledger.day:
            # The reservation was made on a day that has since rolled over.
            # Drop it rather than corrupt the new day with a stale claim.
            return
        if actual_neurons < 0:
            raise BudgetError(f"negative actual cost: {actual_neurons}")

        self.ledger.reserved_neurons -= reservation.neurons
        self.ledger.settled_neurons += actual_neurons
        if self.costs.models[reservation.model]["usd_per_unit_unit"] == "audio-minute":
            # Minutes are the caller's own measurement of the audio it sent, so
            # they are recorded as settled here; the reservation's share is
            # dropped rather than carried over into the next call.
            self.ledger.audio_minutes_measured += reservation.units
            self.ledger.audio_minutes_reserved = max(
                0.0, self.ledger.audio_minutes_reserved - reservation.units
            )

        self.ledger.calls[reservation.model] = (
            self.ledger.calls.get(reservation.model, 0) + 1
        )

        tolerance = max(1e-9, 1e-6 * reservation.neurons)
        if actual_neurons > reservation.neurons + tolerance:
            self.ledger.overruns += 1
            self.ledger.poisoned = True
            self.ledger.poison_reason = (
                f"{reservation.model} settled at {actual_neurons:.2f} neurons vs "
                f"{reservation.neurons:.2f} reserved; cost constants are stale"
            )

    def release(self, reservation: Reservation) -> None:
        """
        Give a reservation back without charging.

        For a call that failed before the model was invoked. A failed call that
        still held its reservation would slowly starve the ledger for no
        benefit; one that was never made must not be billed.
        """
        if reservation.day != self.ledger.day:
            return
        self.ledger.reserved_neurons = max(
            0.0, self.ledger.reserved_neurons - reservation.neurons
        )

    # -- reporting ----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Serialisable state for Durable Object storage."""
        led = self.ledger
        return {
            "day": led.day,
            "settled_neurons": round(led.settled_neurons, 4),
            "reserved_neurons": round(led.reserved_neurons, 4),
            "committed_neurons": round(led.committed, 4),
            "allowance_neurons": self.allowance(),
            "headroom_neurons": round(self.headroom(), 4),
            "fraction_of_day_used": round(
                led.committed / self.allowance(), 6) if self.allowance() else None,
            "calls": dict(led.calls),
            "refused": led.refused,
            "overruns": led.overruns,
            "poisoned": led.poisoned,
            "poison_reason": led.poison_reason,
            "paid": self.paid,
            "usd_settled": round(self.costs.usd(led.settled_neurons), 6),
            "max_paid_overage_neurons": self.max_paid_overage_neurons,
        }

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any], costs: Costs) -> "NeuronBudget":
        """Rebuild from Durable Object storage, preserving the day key."""
        budget = cls(costs, paid=bool(snap.get("paid", False)),
                     max_paid_overage_neurons=snap.get(
                         "max_paid_overage_neurons"))
        try:
            led = Ledger.from_dict(snap)
        except LedgerCorrupt as exc:
            # A corrupt ledger is not something to paper over by starting at
            # zero: that reads as a full budget and invites exactly the
            # overrun we are guarding against. Refuse loudly.
            raise LedgerCorrupt(
                f"refusing to resume from unreadable ledger: {exc}"
            ) from exc
        budget.ledger = led
        return budget