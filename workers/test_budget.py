"""
Tests for the neuron budget accountant.

Pure Python. No Cloudflare dependency, no network, no credentials. These run
offline and must keep doing so: the whole point of the accountant is that it is
checkable without spending anything.

What these tests are really guarding is a direction of error. Every assertion
here is about the ledger being too STINGY rather than too generous, because the
two failures are not symmetric:

  * too generous  -> silent overdraw, an unbounded bill, and a system that
                     looks healthy while it spends. Invisible until it hurts.
  * too stingy    -> the system declines to work, visibly, and someone fixes
                     the constant.

Given the choice, refusing is correct. These tests exist to make sure we never
quietly drift toward the first failure.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from batching import BatchPlanner, PlanningError
from budget import (
    BudgetError,
    BudgetExhausted,
    Costs,
    Ledger,
    LedgerCorrupt,
    NeuronBudget,
    UnknownModel,
)

COSTS = Costs.load()
UTC = timezone.utc


def at(hour: int, minute: int = 0, day: int = 18) -> datetime:
    """A fixed UTC instant, so rollover tests never depend on when they run."""
    return datetime(2026, 10, day, hour, minute, tzinfo=UTC)


def fresh(*, paid: bool = False, now: datetime | None = None) -> NeuronBudget:
    return NeuronBudget(COSTS, paid=paid, now=now or at(12))


# --------------------------------------------------------------------------
# The constants themselves
# --------------------------------------------------------------------------

class TestCosts:
    def test_costs_json_is_the_only_place_numbers_live(self):
        """Guard against a price being hard-coded somewhere and drifting."""
        raw = COSTS.raw
        assert raw["neuron"]["usd_per_neuron"] == 0.000011
        assert raw["neuron"]["free_neurons_per_day"] == 10_000
        assert raw["plans"]["free"]["requests_per_day"] == 100_000
        assert raw["plans"]["max_simultaneous_open_connections"] == 6

    def test_every_neuron_and_price_claim_carries_a_source(self):
        """A limit without a citation cannot be defended, so make it fail."""
        assert COSTS.raw["neuron"]["usd_per_neuron_source"]
        assert COSTS.raw["neuron"]["free_neurons_per_day_source"]
        assert COSTS.raw["neuron"]["reset_source"]
        for name, spec in COSTS.models.items():
            assert spec.get("usd_source"), f"{name} missing usd_source"
            assert spec.get("published_neurons_source"), \
                f"{name} missing published_neurons_source"

    def test_rate_takes_the_more_expensive_of_two_published_numbers(self):
        """
        The conservative rule, and the case that motivates it.

        smart-turn-v2 is priced at $0.000338/audio-minute and its docs also
        claim 0.51 neurons/audio-minute. Those disagree by 60x. We bill the max.
        """
        effective = COSTS.neurons_per_unit("smart-turn-v2")
        published = 0.51
        implied = 0.000338 / COSTS.neuron_usd
        assert implied > published, "the conflict this test encodes has gone away"
        assert effective == pytest.approx(implied, rel=1e-9)
        assert effective > published

    def test_models_whose_two_prices_agree_are_unaffected(self):
        """The rule must be uniform, not a special case for one slug."""
        for name in ("clef-omni", "whisper", "whisper-large-v3-turbo"):
            spec = COSTS.models[name]
            # Both published figures share a unit, so the "per million"
            # cancels and the conversion is just usd / usd_per_neuron.
            implied = spec["usd_per_unit"] / COSTS.neuron_usd
            assert COSTS.neurons_per_unit(name) == pytest.approx(
                max(float(spec["published_neurons_per_unit"]), implied), rel=1e-9)
            assert abs(COSTS.neurons_per_unit(name) - implied) / implied < 0.10, \
                f"{name}: published and implied prices should roughly agree"

    def test_unknown_model_is_fatal_not_free(self):
        """Defaulting an unknown model to zero cost would be the worst bug here."""
        b = fresh()
        with pytest.raises(UnknownModel):
            b.quote("some-unpriced-model", 1.0)
        with pytest.raises(UnknownModel):
            b.reserve("some-unpriced-model", 1.0)


# --------------------------------------------------------------------------
# Quoting
# --------------------------------------------------------------------------

class TestQuoting:
    def test_audio_quote_uses_the_conservative_rate(self):
        b = fresh()
        q = b.quote_audio(1.0)
        assert q.neurons == pytest.approx(30.7272727, rel=1e-6)
        assert q.unit_name == "audio-minute"

    def test_judgement_quote_scales_with_millionths_of_a_token_unit(self):
        b = fresh()
        one_million = b.quote_judgement(1_000_000)
        assert one_million.neurons == pytest.approx(13636.0, rel=1e-4)
        eight_k = b.quote_judgement(8000)
        assert eight_k.neurons == pytest.approx(one_million.neurons / 125.0,
                                                rel=1e-9)

    def test_quote_does_not_mutate_the_ledger(self):
        b = fresh()
        b.quote_audio(600.0)
        b.quote_judgement(50_000)
        assert b.ledger.committed == 0.0
        assert b.ledger.calls == {}

    def test_negative_units_are_rejected(self):
        with pytest.raises(BudgetError):
            fresh().quote_audio(-1.0)


# --------------------------------------------------------------------------
# THE REQUIRED TEST: refuses to overdraw
# --------------------------------------------------------------------------

class TestRefusesToOverdraw:
    def test_repeated_audio_reservations_stop_at_the_allowance(self):
        """
        Spend the day down in realistic clip-sized steps and prove the ledger
        refuses before it crosses 10,000, not at or after it.

        Clip size is the watcher's own maximum speech run, 2.0 s
        (src/watcher.py MAX_UTTERANCE_S), because that is the largest unit of
        audio the gate will ever actually send.
        """
        b = fresh()
        allowance = COSTS.free_neurons_per_day
        clip_minutes = 2.0 / 60.0
        per_clip = b.quote_audio(clip_minutes).neurons
        # How many clips must fit before the day is spent. Derived, so the
        # test does not silently pass with a wrong mental model.
        expected = int(allowance / per_clip)
        assert 9000 < expected < 10_500, "clip cost / count drifted"

        granted = []
        for _ in range(expected + 100):
            try:
                r = b.reserve("smart-turn-v2", clip_minutes)
            except BudgetExhausted:
                break
            granted.append(r)
            b.settle(r, r.neurons)
        else:
            pytest.fail("ledger never refused; it is not enforcing anything")

        assert len(granted) == expected
        # The hard requirement: committed never exceeds the allowance.
        assert b.ledger.committed <= allowance, (
            f"overdrew: committed {b.ledger.committed} > allowance {allowance}")
        # And it refused for the right reason.
        assert b.ledger.refused >= 1
        assert b.ledger.overruns == 0, "refusal is not an overrun"
        assert b.ledger.audio_minutes_measured == pytest.approx(
            expected * clip_minutes, rel=1e-9)

    def test_exhaustion_raises_rather_than_returning_a_sentinel(self):
        """
        An exception cannot be ignored by accident.

        A bool return invites `if not budget.reserve(...): pass`; a raise does
        not. This is the whole mechanism, so it is worth pinning.
        """
        b = fresh()
        # One reservation big enough to blow the whole day.
        with pytest.raises(BudgetExhausted) as exc:
            b.reserve("smart-turn-v2", 10_000.0)
        assert exc.value.model == "smart-turn-v2"
        assert exc.value.needed > exc.value.available
        assert exc.value.reason
        # A refusal must not have moved the ledger.
        assert b.ledger.committed == 0.0
        assert b.ledger.reserved_neurons == 0.0

    def test_concurrent_reservations_cannot_all_pass(self):
        """
        N callers checking a balance that does not yet include each other is
        the classic overdraw race.

        Reserving is pessimistic -- it counts immediately -- so the second
        caller sees the first's claim. Proven here by reserving without
        settling: an unsettled reservation must still block.
        """
        b = fresh()
        half = COSTS.free_neurons_per_day / 2.0 / COSTS.neurons_per_unit("smart-turn-v2")
        b.reserve("smart-turn-v2", half)          # claim half, do NOT settle
        assert b.can_afford("smart-turn-v2", half), "second half should fit"
        b.reserve("smart-turn-v2", half)          # second half now committed
        assert b.headroom() == pytest.approx(0.0, abs=1e-6)
        with pytest.raises(BudgetExhausted):
            b.reserve("smart-turn-v2", half)      # a third half cannot

    def test_free_tier_never_exceeds_the_allowance(self):
        """On free there is nothing to bill, so an overrun is a hard error."""
        b = fresh(paid=False)
        step_minutes = 60.0
        step_neurons = b.quote_audio(step_minutes).neurons
        while b.can_afford("smart-turn-v2", step_minutes):
            r = b.reserve("smart-turn-v2", step_minutes)
            b.settle(r, r.neurons)
        assert b.ledger.committed <= COSTS.free_neurons_per_day
        # Discrete steps mean the day is not necessarily spent to the last
        # neuron -- what matters is that not one more step fits.
        assert b.headroom() < step_neurons
        with pytest.raises(BudgetExhausted):
            b.reserve("smart-turn-v2", step_minutes)

    def test_paid_tier_is_still_capped(self):
        """
        Paid removes Cloudflare's ceiling, not ours -- but it does NOT stop us
        at the allowance either.

        This is the distinction the ledger must get right, and getting it wrong
        is silent: an implementation that refuses at the 10,000 allowance on
        Paid makes the overage cap unreachable, so the Paid plan costs money and
        buys exactly nothing. Both halves are asserted -- it DOES overage, and it
        stops at the cap.
        """
        b = fresh(paid=True)
        allowance = COSTS.free_neurons_per_day
        cap = b.max_paid_overage_neurons
        assert cap == float(COSTS.ours("max_paid_overage_neurons_per_day"))

        step_minutes = 120.0
        step = b.quote_audio(step_minutes).neurons
        while True:
            try:
                r = b.reserve("smart-turn-v2", step_minutes)
            except BudgetExhausted:
                break
            b.settle(r, r.neurons)

        # It went over the included allowance -- otherwise Paid is pointless.
        assert b.ledger.committed > allowance, (
            "paid tier must be allowed to exceed the included allowance")
        # And it stopped before the cap.
        assert b.ledger.committed - allowance <= cap + 1e-6, (
            f"overage {b.ledger.committed - allowance} exceeded cap {cap}")
        # Discrete steps leave headroom, but not a whole step of it.
        assert b.would_overdraw(step)[0] is True

    def test_can_afford_does_not_consume(self):
        b = fresh()
        for _ in range(50):
            assert b.can_afford("smart-turn-v2", 1.0)
        assert b.ledger.committed == 0.0


# --------------------------------------------------------------------------
# THE REQUIRED TEST: recovers correctly at daily rollover
# --------------------------------------------------------------------------

class TestDailyRollover:
    def test_refill_at_0000_utc(self):
        """A full day of spend, then midnight UTC: capacity returns."""
        b = fresh(now=at(23, 59))
        step_minutes = 120.0
        while b.can_afford("smart-turn-v2", step_minutes):
            r = b.reserve("smart-turn-v2", step_minutes)
            b.settle(r, r.neurons)
        assert b.headroom() < b.quote_audio(step_minutes).neurons, "precondition"
        spent_yesterday = b.ledger.settled_neurons
        assert spent_yesterday > 0

        b.set_now(at(0, 0, day=19))
        assert b.rollover_if_due() is True
        assert b.headroom() == pytest.approx(COSTS.free_neurons_per_day)
        assert b.ledger.settled_neurons == 0.0
        assert b.ledger.day == "2026-10-19"
        # And it genuinely spends again.
        r = b.reserve("smart-turn-v2", 60.0)
        assert r.neurons > 0
        assert b.ledger.committed > 0

    def test_no_rollover_within_a_day(self):
        b = fresh(now=at(0, 0))
        r = b.reserve("smart-turn-v2", 1.0)
        b.settle(r, r.neurons)
        spent = b.ledger.settled_neurons
        for hour in (1, 6, 12, 23, 23):
            b.set_now(at(hour))
            assert b.rollover_if_due() is False
            assert b.ledger.settled_neurons == pytest.approx(spent)
        assert b.ledger.day == "2026-10-18"

    def test_rollover_is_keyed_on_utc_not_local_time(self):
        """
        Cloudflare resets at 00:00 UTC. A ledger keyed on local time would
        double-count or double-refill depending on where it ran.
        """
        b = fresh(now=at(12))
        # 23:30 UTC is 13:30 in Alaska (UTC-9) -- same UTC day.
        b.set_now(datetime(2026, 10, 18, 23, 30, tzinfo=UTC))
        assert b.rollover_if_due() is False
        # 00:30 UTC is still the 18th in Alaska? No: 14:30 on the 18th there.
        b.set_now(datetime(2026, 10, 19, 0, 30, tzinfo=UTC))
        assert b.rollover_if_due() is True
        assert b.ledger.day == "2026-10-19"

    def test_stale_reservation_does_not_corrupt_the_new_day(self):
        """
        A reservation made before midnight, settled after.

        Cloudflare bills when the call is made, and we cannot know its
        attribution rule for a call that straddles the boundary. The honest
        behaviour is to drop the stale claim rather than debit a day we never
        charged.
        """
        b = fresh(now=at(23, 59))
        r = b.reserve("smart-turn-v2", 30.0)
        assert b.ledger.reserved_neurons == pytest.approx(r.neurons)

        b.set_now(at(0, 1, day=19))
        b.rollover_if_due()
        assert b.ledger.reserved_neurons == 0.0

        b.settle(r, r.neurons)   # late arrival, wrong day
        assert b.ledger.committed == 0.0
        assert b.ledger.settled_neurons == 0.0
        assert b.ledger.day == "2026-10-19"

    def test_release_does_not_leak_capacity(self):
        """A failed call must give its reservation back."""
        b = fresh()
        r = b.reserve("clef-omni", 0.008)
        assert b.ledger.committed > 0
        b.release(r)
        assert b.ledger.committed == pytest.approx(0.0)
        assert b.headroom() == pytest.approx(COSTS.free_neurons_per_day)

    def test_poison_resets_at_rollover(self):
        """
        A poisoned ledger stops everything until the day turns.

        It must then recover on its own -- an operator should not have to
        intervene daily just to keep the system running, or it will not be
        left running.
        """
        b = fresh()
        small = b.reserve_judgement(1000)
        b.settle(small, small.neurons * 10)          # cost 10x the estimate
        assert b.ledger.poisoned is True
        with pytest.raises(BudgetExhausted) as exc:
            b.reserve_judgement(1000)
        assert "poisoned" in exc.value.reason

        b.set_now(at(0, 0, day=19))
        b.rollover_if_due()
        assert b.ledger.poisoned is False
        r = b.reserve_judgement(1000)
        assert r.neurons > 0


# --------------------------------------------------------------------------
# Settling against measured usage
# --------------------------------------------------------------------------

class TestSettlement:
    def test_reservation_is_replaced_by_measured_cost(self):
        b = fresh()
        r = b.reserve_judgement(8000)
        assert b.ledger.reserved_neurons == pytest.approx(r.neurons)
        b.settle(r, 123.45)
        assert b.ledger.reserved_neurons == pytest.approx(0.0)
        assert b.ledger.settled_neurons == pytest.approx(123.45)
        assert b.ledger.calls["clef-omni"] == 1

    def test_under_spend_is_not_an_overrun(self):
        """Estimates run high on purpose; that must not trip the alarm."""
        b = fresh()
        r = b.reserve_judgement(8000)
        b.settle(r, r.neurons / 2.0)
        assert b.ledger.overruns == 0
        assert b.ledger.poisoned is False

    def test_over_spend_poisons_the_ledger(self):
        """
        If a call costs more than reserved, our constants are wrong.

        Re-baselining silently would turn a stale price into an unbounded bill,
        so the ledger refuses expensive calls until the next day instead.
        """
        b = fresh()
        r = b.reserve_judgement(8000)
        b.settle(r, r.neurons * 1.5)
        assert b.ledger.overruns == 1
        assert b.ledger.poisoned is True
        assert "stale" in b.ledger.poison_reason

    def test_floating_point_noise_is_not_an_overrun(self):
        """A 1-ULP wobble must not halt the system."""
        b = fresh()
        r = b.reserve_judgement(8000)
        b.settle(r, r.neurons * (1 + 1e-15))
        assert b.ledger.overruns == 0
        assert b.ledger.poisoned is False

    def test_measured_usage_replaces_the_estimate_end_to_end(self):
        """
        The intended production loop: reserve from an estimate, settle from the
        model's reported usage, and end up with a ledger that reflects reality.

        Actuals stay within the estimate, which is the designed direction of the
        error: the reservation is an over-estimate on purpose, so ordinary calls
        under-spend and leave the ledger un-poisoned.
        """
        b = fresh()
        estimate = int(COSTS.ours("expected_input_tokens_per_judgement"))
        total = 0.0
        for actual_tokens in (4000, 6000, 8000):
            r = b.reserve_judgement(estimate)
            b.settle(r, b.quote_judgement(actual_tokens).neurons)
            total += b.quote_judgement(actual_tokens).neurons
        assert b.ledger.settled_neurons == pytest.approx(total)
        assert b.ledger.calls["clef-omni"] == 3
        assert b.ledger.poisoned is False
        # The ledger tracks MEASURED cost, not the sum of the reservations.
        reserved_total = 3 * b.quote_judgement(estimate).neurons
        assert b.ledger.settled_neurons < reserved_total


# --------------------------------------------------------------------------
# Serialisation round-trip (Durable Object storage)
# --------------------------------------------------------------------------

class TestSnapshotRoundTrip:
    def test_round_trip_preserves_the_ledger(self):
        b = fresh()
        r = b.reserve("smart-turn-v2", 2.0)
        b.settle(r, r.neurons)
        open_r = b.reserve_judgement(5000)          # left unsettled
        snap = json.loads(json.dumps(b.snapshot()))  # as DO storage would

        restored = NeuronBudget.from_snapshot(snap, COSTS)
        assert restored.ledger.day == b.ledger.day
        assert restored.ledger.settled_neurons == pytest.approx(
            b.ledger.settled_neurons)
        assert restored.ledger.reserved_neurons == pytest.approx(
            b.ledger.reserved_neurons)
        assert restored.ledger.calls == b.ledger.calls
        assert restored.snapshot()["fraction_of_day_used"] == \
            b.snapshot()["fraction_of_day_used"]

    def test_poison_survives_a_round_trip(self):
        b = fresh()
        r = b.reserve_judgement(1000)
        b.settle(r, r.neurons * 3)
        restored = NeuronBudget.from_snapshot(b.snapshot(), COSTS)
        assert restored.ledger.poisoned is True
        with pytest.raises(BudgetExhausted):
            restored.reserve_judgement(1000)

    def test_corrupt_state_is_refused_not_reset_to_zero(self):
        """
        The tempting recovery for unreadable state is "start fresh", which reads
        as a full budget and invites precisely the overdraw being guarded
        against. Refuse instead.
        """
        with pytest.raises(LedgerCorrupt):
            Ledger.from_dict({"settled_neurons": "not-a-number"})
        with pytest.raises(LedgerCorrupt):
            NeuronBudget.from_snapshot({}, COSTS)

    def test_snapshot_reports_the_numbers_a_human_needs(self):
        b = fresh()
        r = b.reserve_judgement(8000)
        b.settle(r, 100.0)
        snap = b.snapshot()
        for key in ("day", "settled_neurons", "reserved_neurons",
                    "allowance_neurons", "headroom_neurons",
                    "fraction_of_day_used", "usd_settled", "paid"):
            assert key in snap
        assert snap["usd_settled"] == pytest.approx(100.0 * COSTS.neuron_usd)


# --------------------------------------------------------------------------
# Batching: the request budget, and why it is separate from the neuron budget
# --------------------------------------------------------------------------

class TestBatchPlanning:
    def test_request_allowance_is_shared_not_whole(self):
        p = BatchPlanner(COSTS)
        assert p.requests_per_day() == pytest.approx(10_000.0)
        assert p.requests_per_day() < COSTS.raw["plans"]["free"]["requests_per_day"], \
            "the gate must leave room for the other Workers on the account"

    def test_cadence_follows_from_the_request_allowance(self):
        p = BatchPlanner(COSTS)
        assert p.min_batch_interval_s() == pytest.approx(8.64)
        assert p.requests_per_second() == pytest.approx(10_000 / 86400.0)

    def test_parallelism_is_capped_by_the_connection_limit(self):
        """Six is a platform limit, not a preference."""
        p = BatchPlanner(COSTS)
        assert p.max_parallel_model_calls() == 6

    def test_24x7_is_not_affordable_at_the_effective_rate(self):
        """
        The headline finding, pinned as a test so it cannot quietly regress.

        At the conservative rate, a day of audio needs ~44,247 neurons against a
        10,000 allowance: 4.4x over. Continuous endpointing does not fit in the
        free tier, and no amount of batching changes that, because batching does
        not reduce audio minutes.
        """
        p = BatchPlanner(COSTS)
        assert p.neurons_per_day_for_24x7("effective") > \
            COSTS.free_neurons_per_day
        assert p.neurons_per_day_for_24x7("effective") / \
            COSTS.free_neurons_per_day == pytest.approx(4.42, abs=0.02)
        assert p.days_of_24x7_affordable("effective") < 1.0
        assert p.headroom_minutes_per_day("effective") == pytest.approx(325.4,
                                                                        abs=0.5)

    def test_the_two_documented_rates_disagree_by_60x(self):
        """
        Record both readings. This is a live disagreement in Cloudflare's own
        pricing page; if it is ever resolved, this test is what should be
        revisited first.
        """
        p = BatchPlanner(COSTS)
        effective = p.headroom_minutes_per_day("effective")
        published = p.headroom_minutes_per_day("published")
        assert published / effective == pytest.approx(60.2, rel=0.01)
        assert p.days_of_24x7_affordable("published") == pytest.approx(
            13.6, abs=0.1)

    def test_batching_does_not_buy_neurons(self):
        """
        The load-bearing insight, stated as arithmetic.

        smart-turn is billed per audio minute, so N calls of 1 minute cost the
        same as 1 call of N minutes. Batching reduces invocations, which is the
        request budget, and leaves the neuron budget untouched. Any plan that
        claims batching makes 24/7 affordable is wrong.
        """
        b = fresh()
        one_big = b.quote_audio(60.0).neurons
        sixty_small = sum(
            b.quote_audio(1.0).neurons for _ in range(60))
        assert one_big == pytest.approx(sixty_small, rel=1e-9)

    def test_clip_filter_drops_sub_threshold_and_oversized_clips(self):
        p = BatchPlanner(COSTS)
        assert p.clip_is_worth_endpointing(2.0) is True
        assert p.clip_is_worth_endpointing(0.05) is False   # below OURS floor
        assert p.clip_is_worth_endpointing(0.0) is False
        assert p.clip_is_worth_endpointing(30.0) is False   # beyond a speech run
        # The boundary is inclusive of the watcher's own maximum.
        assert p.clip_is_worth_endpointing(2.0000001) is False

    def test_unknown_rate_basis_is_rejected(self):
        with pytest.raises(PlanningError):
            BatchPlanner(COSTS).headroom_minutes_per_day("wishful")

    def test_plan_is_self_describing(self):
        text = BatchPlanner(COSTS).plan().describe()
        assert "req/day" in text and "parallel" in text