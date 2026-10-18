/**
 * The neuron ledger, as a Durable Object.
 *
 * This is the single most important structural decision in the directory, and
 * it is not a matter of taste.
 *
 * Neurons are billed to the ACCOUNT, not to a session. So a ledger kept per
 * session would hand every session the full 10,000/day allowance, and N
 * sessions would overdraw by a factor of N while each one honestly believed it
 * was within budget. The failure is invisible from any single session's point
 * of view -- which is exactly what makes it dangerous.
 *
 * Therefore: ONE Durable Object for the whole account, and every session's gate
 * calls into it. Durable Objects are single-threaded and serialise requests per
 * instance (developers.cloudflare.com/durable-objects/), which supplies the
 * one thing needed for a correct read-modify-write on a shared balance:
 * mutual exclusion. KV cannot supply it -- see costs.json -> kv.verdict.
 *
 * Throughput is a non-issue. The gate's request cadence is ~1 batch per 8.64 s
 * (costs.json -> OURS.request_budget_share), so this object handles well under
 * one request per second. Choosing to serialise correctness behind a queue is
 * free here.
 *
 * Storage is read once and cached in memory, then written back on mutation. DO
 * storage operations are not free, and the ledger is read on every reserve.
 */

const STATE_KEY = "ledger";

/**
 * Durable Object stub. Wrangler injects the real base class; this exists so the
 * file is importable and testable outside the Workers runtime.
 *
 * DECLARED BEFORE NeuronLedger on purpose. Class declarations are hoisted but
 * not initialised, so `extends` against a class defined further down the file
 * throws "Cannot access ... before initialization" -- which means the file only
 * imports cleanly inside a bundler that happens to tolerate it, and not in
 * plain Node. Since being importable is the entire point of the stub, it has to
 * come first.
 */
export class ServerDurableObjectStub {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }
}

export class BudgetError extends Error {}
export class BudgetExhausted extends BudgetError {}
export class UnknownModel extends BudgetError {}
export class LedgerCorrupt extends BudgetError {}

/**
 * Convenience accessor so the Worker calls read as intent
 * (`budgetDO(env.BUDGET).reserve(...)`) rather than binding plumbing.
 */
export function budgetDO(namespace) {
  return namespace;
}

/**
 * Mirror of workers/hotpath/budget.py.
 *
 * HONEST SCOPE NOTE, because it matters: the Python module is the reference
 * implementation and is covered by workers/test_budget.py (39 tests). This is a
 * mirror, and the Python tests do NOT execute this file -- there is no JS test
 * harness in this repo and adding one was out of scope. What IS genuinely
 * shared is the arithmetic's input: both sides read the same costs.json and
 * both apply the same max(published, implied) rate rule, so the tests do guard
 * the numbers and the rule. What is NOT machine-verified here is this
 * translation of them. Treat any divergence as a bug in this file, not in the
 * constants.
 */
export class NeuronLedger extends ServerDurableObjectStub {
  constructor(state, env) {
    super(state, env);
    this.state = state;
    this.env = env;
    this.costs = env.COSTS;
    this.paid = env.PAID_PLAN === "true";
    this.ledger = null;
  }

  /**
   * Effective neurons per billed unit: the MORE EXPENSIVE of the two figures
   * Cloudflare publishes. Mirrors Costs._rate in budget.py.
   */
  neuronsPerUnit(model) {
    const spec = this.costs.models[model];
    if (!spec) throw new UnknownModel(`no cost entry for model ${model}`);
    if (spec.usd_per_unit_unit !== "audio-minute" &&
        spec.usd_per_unit_unit !== "M-input-tokens") {
      throw new LedgerCorrupt(`unknown pricing unit for ${model}`);
    }
    // The "per million" cancels: both published figures share the same unit.
    const implied = spec.usd_per_unit / this.costs.neuron.usd_per_neuron;
    return Math.max(spec.published_neurons_per_unit, implied);
  }

  async load() {
    if (this.ledger) return this.ledger;
    const stored = await this.state.storage.get(STATE_KEY);
    if (stored) {
      this.ledger = stored;
    } else {
      this.ledger = freshLedger(this.dayKey(new Date()));
    }
    // If the stored day is not today, the allowance has reset. Dropping the
    // stale ledger rather than merging into it is the whole rollover rule:
    // Cloudflare's allowance resets at 00:00 UTC and carries nothing over.
    if (this.ledger.day !== this.dayKey(new Date())) {
      this.ledger = freshLedger(this.dayKey(new Date()));
    }
    return this.ledger;
  }

  async save() {
    await this.state.storage.put(STATE_KEY, this.ledger);
  }

  /**
   * The UTC day this instant falls in. Must be UTC because Cloudflare resets at
   * "00:00 UTC"; keying on local time would double-count or double-refill
   * depending on where this happens to run.
   */
  dayKey(moment) {
    return moment.toISOString().slice(0, 10);
  }

  allowance() {
    return this.costs.neuron.free_neurons_per_day;
  }

  maxOverage() {
    return this.costs.OURS.max_paid_overage_neurons_per_day.value;
  }

  committed() {
    return this.ledger.settled_neurons + this.ledger.reserved_neurons;
  }

  headroom() {
    return Math.max(0, this.allowance() - this.committed());
  }

  /**
   * Claim budget, or refuse. Mirrors NeuronBudget.reserve.
   *
   * The reservation counts against the day IMMEDIATELY, unsettled. That is what
   * stops two concurrent callers from each passing a check the other then
   * invalidates -- and single-threaded execution is what makes "immediately"
   * actually immediate.
   */
  async reserve({ model, units }) {
    const led = await this.load();
    const rate = this.neuronsPerUnit(model);
    const neurons = rate * units;
    const over = this.wouldOverdraw(neurons);
    if (over) {
      led.refused += 1;
      await this.save();
      throw new BudgetExhausted(
        `refusing ${model}: needs ${neurons.toFixed(2)} neurons, ` +
        `${this.headroom().toFixed(2)} available (${over})`);
    }
    led.reserved_neurons += neurons;
    if (this.costs.models[model].usd_per_unit_unit === "audio-minute") {
      led.audio_minutes_reserved += units;
    }
    const id = `${led.day}-${++led.seq}`;
    await this.save();
    return { reservation_id: id, model, units, neurons, day: led.day };
  }

  wouldOverdraw(neurons) {
    if (this.ledger.poisoned) {
      return `ledger poisoned: ${this.ledger.poison_reason}`;
    }
    const projected = this.committed() + neurons;
    if (!this.paid) {
      // Free: the allowance IS the whole budget. Past it Cloudflare's
      // documented behaviour is that "further operations will fail with an
      // error", so there is nothing graceful to allow.
      if (projected > this.allowance()) {
        return `free-tier daily allowance ${this.allowance()} exhausted ` +
               `(committed ${this.committed().toFixed(2)})`;
      }
      return null;
    }
    // Paid: the same allowance is INCLUDED and going over is legitimate; the
    // only ceiling is our own overage cap. Refusing at the allowance here
    // would make the cap unreachable and Paid would buy nothing.
    const overage = Math.max(0, projected - this.allowance());
    if (overage > this.maxOverage()) {
      return `paid overage cap ${this.maxOverage()} neurons would be exceeded`;
    }
    return null;
  }

  /**
   * Replace a reservation with measured cost. Mirrors NeuronBudget.settle.
   *
   * Over the reservation is treated as a signal that our constants are stale,
   * and poisons the ledger until the next daily rollover. Re-baselining silently
   * would convert a wrong constant into an unbounded bill.
   */
  async settle({ reservation_id, actual_neurons, units, reservation_neurons }) {
    const led = await this.load();
    // Settle the reservation we can actually find. The caller echoes the
    // reserved amount because the DO does not keep a reservation table; a
    // mismatch is reported rather than trusted.
    const reserved = Number(reservation_neurons);
    if (!Number.isFinite(reserved) || reserved < 0) {
      throw new BudgetError(`settle called without a reservation_neurons for ${reservation_id}`);
    }
    led.reserved_neurons = Math.max(0, led.reserved_neurons - reserved);
    led.settled_neurons += actual_neurons;
    if (Number.isFinite(units)) led.audio_minutes_measured += units;

    const tol = Math.max(1e-9, 1e-6 * reserved);
    if (actual_neurons > reserved + tol) {
      led.overruns += 1;
      led.poisoned = true;
      led.poison_reason =
        `settled at ${actual_neurons.toFixed(2)} neurons vs ${reserved.toFixed(2)} ` +
        `reserved; cost constants are stale`;
    }
    await this.save();
    return { ok: true, poisoned: led.poisoned };
  }

  /** Give an unspent reservation back, e.g. a call that never was made. */
  async release({ reservation_neurons }) {
    const led = await this.load();
    led.reserved_neurons = Math.max(0, led.reserved_neurons - Number(reservation_neurons));
    await this.save();
    return { ok: true };
  }

  /**
   * Settle a clef-omni reservation from the call's REPORTED token usage.
   *
   * This is the measurement path that makes a wrong constant self-correcting.
   * clef-omni's output schema requires `usage: {input_tokens, output_tokens}`,
   * so the real quantity is observable rather than guessed. Converting tokens
   * to neurons happens HERE rather than in the caller so there is exactly one
   * place that knows the rate.
   */
  async settleMeasured({ reservation_id, reservation_neurons, input_tokens, output_tokens }) {
    if (!Number.isFinite(input_tokens)) {
      // No reported usage. Do NOT silently settle at the reserved estimate --
      // that is the fail-open path, and it is how an overcharge becomes
      // invisible. Report it so the caller can mark the row unknown.
      return { ok: false, reason: "no usage reported", neurons: null };
    }
    const rate = this.neuronsPerUnit("clef-omni");
    // Published pricing is per INPUT token; output tokens are reported for
    // completeness and logged, but not billed, because costs.json carries no
    // separate output rate for clef-omni. Stated rather than assumed silently.
    const neurons = (Number(input_tokens) / 1_000_000.0) * rate;
    await this.settle({
      reservation_id,
      reservation_neurons,
      actual_neurons: neurons,
      units: Number(input_tokens),
    });
    return { ok: true, neurons, input_tokens, output_tokens: output_tokens ?? null };
  }

  async snapshot() {
    const led = await this.load();
    return {
      day: led.day,
      settled_neurons: round4(led.settled_neurons),
      reserved_neurons: round4(led.reserved_neurons),
      committed_neurons: round4(this.committed()),
      allowance_neurons: this.allowance(),
      headroom_neurons: round4(this.headroom()),
      fraction_of_day_used: round6(this.committed() / this.allowance()),
      refused: led.refused,
      overruns: led.overruns,
      poisoned: led.poisoned,
      poison_reason: led.poison_reason,
      paid: this.paid,
      usd_settled: round6(led.settled_neurons * this.costs.neuron.usd_per_neuron),
      max_paid_overage_neurons: this.maxOverage(),
    };
  }
}

function freshLedger(day) {
  return {
    day,
    settled_neurons: 0,
    reserved_neurons: 0,
    audio_minutes_measured: 0,
    audio_minutes_reserved: 0,
    refused: 0,
    overruns: 0,
    seq: 0,
    poisoned: false,
    poison_reason: "",
  };
}

const round4 = (n) => Math.round(n * 1e4) / 1e4;
const round6 = (n) => Math.round(n * 1e6) / 1e6;