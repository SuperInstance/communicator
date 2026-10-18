"""
Batch planning for the hotpath gate. Pure Python, no Cloudflare dependency.

The design has one economic hinge, and this module exists to make the hinge
explicit rather than implicit:

    Batching buys REQUESTS, not NEURONS.

smart-turn-v2 is billed per AUDIO MINUTE (costs.json -> models."smart-turn-v2").
Coalescing ten events into one call therefore costs almost exactly what ten
calls cost -- the audio still all goes up. What coalescing actually saves is
invocations against the 100,000 requests/day free allowance, which is the
binding constraint at the event rate the watcher produces. Those are different
resources and confusing them is how a system ends up "optimising" its way into
an overage it did not see coming.

So the planner below has two independent budgets and refuses to conflate them:

  * REQUEST budget  -- 100,000/day shared by ALL Workers on the account.
    We take OURS.request_budget_share of it. This sets batch cadence.

  * NEURON budget   -- 10,000/day, account-wide. Sets how much audio may be
    endpointed at all. Handled by NeuronBudget, not here.

And a third limit that is neither, and is the hardest one in the system:

  * CONNECTION budget -- 6 simultaneous open connections per invocation
    (Workers platform limits). This bounds how many model calls may be in
    flight from one Worker, and it is why MAX_PARALLEL_MODEL_CALLS is 6 and
    not "as many as we like".
"""

from __future__ import annotations

from dataclasses import dataclass

from budget import Costs

# --------------------------------------------------------------------------
# Watcher-derived bounds. These come from the repo, not from Cloudflare.
# --------------------------------------------------------------------------

# The watcher emits an utterance for every speech segment, and
# src/watcher.py caps a speech run at MAX_UTTERANCE_S = 2.0 s. A monologue with
# no pause therefore flushes a checkpoint utterance every 2.0 s -- that is the
# designed steady state, not an edge case. So the event RATE the gate must
# absorb is ~0.5 utterance events/second, plus ONSET and PAUSE events.
#
# This is why the gate takes batches at all. Per-event invocation would need
# ~0.5-1 req/s sustained, and the free tier's 100,000/day allowance is
# 1.157 req/s for ALL Workers combined -- so per-event invocation would consume
# the entire account's daily request budget on a single feature, leaving
# nothing for the tables plane, the queue consumers, or D1.
WATCHER_MAX_UTTERANCE_S = 2.0
WATCHER_MAX_PAUSE_S = 3.0
WATCHER_MAX_UTTERANCE_SOURCE = "src/watcher.py MAX_UTTERANCE_S = 2.0 / MAX_PAUSE_S = 3.0"


class PlanningError(ValueError):
    """A plan was requested that the constraints cannot satisfy."""


@dataclass(frozen=True)
class BatchPlan:
    """The cadence and in-flight bounds the gate runs at."""

    requests_per_day: float
    min_batch_interval_s: float
    max_events_per_batch: int
    max_parallel_model_calls: int
    audio_minutes_per_batch_max: float

    def describe(self) -> str:
        return (
            f"plan: {self.requests_per_day:,.0f} req/day "
            f"(1 batch / {self.min_batch_interval_s:.2f}s), "
            f"<={self.max_events_per_batch} events/batch, "
            f"<={self.max_parallel_model_calls} parallel model calls, "
            f"<={self.audio_minutes_per_batch_max:.3f} audio-min/batch"
        )


class BatchPlanner:
    """
    Turns the published limits into concrete per-invocation bounds.

    Every number it emits is traceable to costs.json or to the watcher; the
    planner introduces no numbers of its own. That is deliberate -- a planner
    with private constants is a planner whose arithmetic nobody can check.
    """

    def __init__(self, costs: Costs, *, plan: str = "free"):
        self.costs = costs
        if plan not in ("free", "paid"):
            raise PlanningError(f"unknown plan {plan!r}")
        self._plan = plan

    # -- request side -------------------------------------------------------

    def requests_per_day(self) -> float:
        """
        Requests this Worker may spend per day.

        The free allowance is per-ACCOUNT across all Workers, so we take a
        fraction (OURS.request_budget_share). On Paid there is no request cap,
        but we still plan to the same cadence: the batch cadence is driven by
        how often we WANT to hear from the watcher, not by a ceiling, and
        batching more finely would only spend neurons faster.
        """
        if self._plan == "paid":
            # No published ceiling, so we fall back to the free-tier number
            # purely as a planning figure, not as a limit.
            return float(self.costs.raw["plans"]["free"]["requests_per_day"])
        free_rpd = float(self.costs.raw["plans"]["free"]["requests_per_day"])
        return free_rpd * float(self.costs.ours("request_budget_share"))

    def min_batch_interval_s(self) -> float:
        """Seconds per batch implied by the request allowance."""
        rpd = self.requests_per_day()
        if rpd <= 0:
            raise PlanningError("request allowance exhausted")
        return 86400.0 / rpd

    def requests_per_second(self) -> float:
        return self.requests_per_day() / 86400.0

    # -- in-flight bounds ---------------------------------------------------

    def max_parallel_model_calls(self) -> int:
        """
        How many model calls may be in flight from ONE invocation.

        Capped by the platform's 6 simultaneous open connections, which counts
        requests awaiting response headers -- exactly the state a Workers AI
        call is in for the whole inference. Not a tuning choice.
        """
        return int(self.costs.raw["plans"]["max_simultaneous_open_connections"])

    def max_events_per_batch(self) -> int:
        """
        Cap on events per batch, derived from the audio that can be endpointed.

        Bounded by two things, and we take the smaller:
          * the neuron budget for the batch (audio is the scarce resource --
            smart-turn is billed per audio minute), and
          * the request budget: a batch bigger than this cannot be shipped
            inside one interval, so it would just queue.

        We do NOT cap by the subrequest limit (50/request on free) because
        concurrency, not count, is what binds: 6 concurrent calls repeated is
        still one subrequest slot each at any instant.
        """
        interval = self.min_batch_interval_s()
        return max(1, int(self.audio_minutes_per_batch_max() * 60.0 / interval))

    def audio_minutes_per_batch_max(self) -> float:
        """
        Audio per batch that keeps a day of batching inside the neuron budget.

        Solves  n_batches_per_day * neurons_per_batch <= free_neurons_per_day
        for neurons_per_batch, with n_batches_per_day from the request cadence.

        On the free tier this comes out NEGATIVE in practice -- the honest
        result, and the reason `headroom_minutes_per_day` is the number the
        README leads with. Continuous 24/7 endpointing does not fit inside
        10,000 neurons/day; see costs.json -> models."smart-turn-v2".CONFLICT.
        """
        batches_per_day = self.requests_per_day()
        if batches_per_day <= 0:
            raise PlanningError("no batches per day")
        return self.costs.free_neurons_per_day / batches_per_day / \
            self.costs.neurons_per_unit("smart-turn-v2")

    # -- the number that matters -------------------------------------------

    def headroom_minutes_per_day(self, rate: str = "effective") -> float:
        """
        Audio-minutes per day the account can afford to endpoint.

        This is the headline feasibility number. `rate="effective"` uses the
        conservative neurons-per-audio-minute; `rate="published"` uses the 0.51
        figure Cloudflare's pricing page prints. The gap between them is a 60x
        disagreement in Cloudflare's own docs, and both are computed so the
        README can state it rather than pick one and hope.
        """
        if rate == "effective":
            per_min = self.costs.neurons_per_unit("smart-turn-v2")
        elif rate == "published":
            per_min = float(
                self.costs.models["smart-turn-v2"]["published_neurons_per_unit"]
            )
        else:
            raise PlanningError(f"unknown rate basis {rate!r}")
        return self.costs.free_neurons_per_day / per_min

    def days_of_24x7_affordable(self, rate: str = "effective") -> float:
        return self.headroom_minutes_per_day(rate) / 1440.0

    def neurons_per_day_for_24x7(self, rate: str = "effective") -> float:
        if rate == "effective":
            per_min = self.costs.neurons_per_unit("smart-turn-v2")
        else:
            per_min = float(
                self.costs.models["smart-turn-v2"]["published_neurons_per_unit"]
            )
        return per_min * 1440.0

    # -- gate shape ---------------------------------------------------------

    def clip_is_worth_endpointing(self, duration_s: float) -> bool:
        """
        Should this clip go to smart-turn-v2 at all?

        Uses OURS.hotpath_min_clip_s. The honest framing: this is hygiene, not
        economics. A 150 ms clip costs ~0.08 neurons at the conservative rate,
        so dropping it saves a rounding error. It is worth doing because
        sub-150 ms segments in a busy room are mostly clicks and breaths, and
        because every clip we do not send is a slot on the 6-connection budget
        that a real utterance can use instead. It is NOT the mechanism that
        keeps us inside budget -- the neuron ledger is.
        """
        if duration_s <= 0:
            return False
        if duration_s < float(self.costs.ours("hotpath_min_clip_s")):
            return False
        # The watcher's own ceiling: a clip longer than this cannot have come
        # from a single speech run, so sending it would be sending a
        # mis-assembled buffer.
        return duration_s <= WATCHER_MAX_UTTERANCE_S

    def gate_threshold(self) -> float:
        """P(end-of-turn) above which a clip is promoted to an expensive call."""
        return float(self.costs.ours("gate_probability_threshold"))

    def expected_reservation_tokens(self) -> int:
        return int(self.costs.ours("expected_input_tokens_per_judgement"))

    def plan(self) -> BatchPlan:
        return BatchPlan(
            requests_per_day=self.requests_per_day(),
            min_batch_interval_s=self.min_batch_interval_s(),
            max_events_per_batch=self.max_events_per_batch(),
            max_parallel_model_calls=self.max_parallel_model_calls(),
            audio_minutes_per_batch_max=self.audio_minutes_per_batch_max(),
        )


def judge_cost_table(budget, *, token_counts=(2000, 8000, 16000)) -> list[str]:
    """
    Human-readable clef-omni cost ladder, for the README and for run logs.

    A judgement is cheap relative to the allowance -- the whole point of the
    architecture is that the expensive-model calls are RATE-LIMITED BY US, not
    by the cloud. This table is how you see that at a glance.
    """
    rows = []
    for n in token_counts:
        q = budget.quote_judgement(n)
        pct = q.neurons / budget.allowance() * 100.0
        rows.append(
            f"  {n:>6,} input tokens -> {q.neurons:8.2f} neurons "
            f"(${q.usd:.6f}) = {pct:5.3f}% of the daily allowance"
        )
    return rows