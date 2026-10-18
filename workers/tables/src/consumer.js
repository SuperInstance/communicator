/**
 * Queue consumer: run the expensive judgement for clips the gate promoted.
 *
 * This is the at-least-once end of the pipeline, and the entire design
 * question here is: what happens when a message arrives twice?
 *
 * Queues delivery is at-least-once. A message can be delivered up to 100 times
 * (queues/platform/limits: "Message retries | 100"), and every redelivery is
 * another consumer invocation -- another billable request against the
 * 100,000/day free allowance, and another clef-omni call against the neurons.
 * So idempotency here is not hygiene, it is cost control: a consumer that
 * double-charges on every retry turns a transient provider blip into a
 * multiplied bill.
 *
 * IDEMPOTENCY STRATEGY, in the order it is checked:
 *   1. `INSERT OR IGNORE` on a deterministic job_id. If the row already exists,
 *      this delivery is a duplicate and we ack without calling any model.
 *   2. The reservation is settled exactly once, because settle() is keyed on
 *      the reservation_id carried in the message.
 *
 * A retry therefore costs one D1 read and one write, and zero model calls.
 * That is the cheapest correct outcome available, and it is why the expensive
 * call happens AFTER the dedupe insert rather than before it.
 *
 * Consumer budget, for the record: 15 min wall clock and up to 5 min CPU
 * (queues/platform/limits). Awaiting model calls is wall-clock, not CPU, so a
 * batch of judgements fits comfortably.
 */

import { CLEF_MODEL_SELECTOR, clefQuestions, clefState } from "./questions.js";

export default {
  async fetch(request, env) {
    // Wrangler's dev-mode poke.
    return new Response("judge consumer", { status: 200 });
  },

  /**
   * Called once per batch (up to 100 messages, queues/platform/limits).
   * `batch.ackAll()` on success and `batch.retryAll()` on failure -- never
   * ackAll on a partial failure, which would silently drop the failures.
   */
  async queue(batch, env) {
    const results = [];
    for (const message of batch.messages) {
      try {
        results.push(await handle(message, env));
      } catch (err) {
        // Let the platform retry this one. A retry costs a read op (budget
        // item 1 above) but not a model call, because the dedupe insert will
        // short-circuit if we already did the work.
        results.push({ ok: false, error: String(err?.message ?? err) });
        message.retry();
      }
    }
    return results;
  },
};

async function handle(message, env) {
  const job = JSON.parse(message.body);

  // -- 1. Dedupe BEFORE spending anything. ---------------------------------
  // Claim the job_id first. A duplicate delivery stops here, having cost one
  // D1 write and zero neurons.
  const claim = await env.DB.prepare(
    `INSERT OR IGNORE INTO verdicts
       (job_id, session_id, event_id, t0, t1, duration_s, gate_probability,
        gate_is_complete, model, status, created_at, updated_at)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'clef-omni', 'pending', ?9, ?9)`
  )
    .bind(
      job.job_id,
      job.session_id,
      job.event_id ?? null,
      job.t0,
      job.t1,
      Math.max(0, job.t1 - job.t0),
      job.gate_probability ?? null,
      job.gate_is_complete ? 1 : 0,
      new Date().toISOString()
    )
    .run();

  if (!claim.meta?.changes) {
    // Already judged. Acknowledge without calling a model.
    return { ok: true, duplicate: true, job_id: job.job_id };
  }

  // -- 2. The expensive call, with the reservation the gate already took. ---
  try {
    const result = await env.AI.run("@cf/cloudflare/clef-omni", {
      model: CLEF_MODEL_SELECTOR,
      state: clefState({
        sample_rate: job.sample_rate,
        duration_s: job.t1 - job.t0,
        origin: job.origin,
      }),
      questions: clefQuestions(),
    });

    // -- 3. Settle against MEASURED usage, not the estimate. ----------------
    // clef-omni's output schema requires `usage: {input_tokens,
    // output_tokens}`. Feeding the real numbers back to the budget DO is what
    // lets a stale cost constant self-correct from measurement -- and if the
    // call turns out to have cost more than the gate reserved, the ledger
    // poisons itself instead of quietly carrying the loss forward.
    const usage = result?.usage ?? {};
    const neurons = await env.BUDGET.settleMeasured({
      reservation_id: job.reservation_id,
      reservation_neurons: job.reserved_neurons,
      input_tokens: usage.input_tokens ?? null,
      output_tokens: usage.output_tokens ?? null,
    });

    const answers = result?.answers ?? {};
    await env.DB.prepare(
      `UPDATE verdicts SET
         is_speech = ?2,
         is_directive = ?3,
         is_directive_conf = ?4,
         usage_input_tokens = ?5,
         usage_output_tokens = ?6,
         neurons = ?7,
         status = 'judged',
         updated_at = ?8
       WHERE job_id = ?1`
    )
      .bind(
        job.job_id,
        answers.is_speech?.noul ?? null,
        answers.is_directive?.choice ?? null,
        answers.is_directive?.confidence ?? null,
        usage.input_tokens ?? null,
        usage.output_tokens ?? null,
        neurons?.neurons ?? null,
        new Date().toISOString()
      )
      .run();

    return { ok: true, job_id: job.job_id, neurons: neurons?.neurons ?? null };
  } catch (err) {
    // Record the failure rather than only retrying. A row stuck at 'pending'
    // forever is indistinguishable from a judgement that was never attempted.
    await env.DB.prepare(
      `UPDATE verdicts SET status = 'failed', updated_at = ?2 WHERE job_id = ?1`
    )
      .bind(job.job_id, new Date().toISOString())
      .run();

    // Release the reservation: the call did not complete, so holding its budget
    // for the whole day would starve the ledger for nothing.
    await env.BUDGET.release({
      reservation_neurons: job.reserved_neurons,
      reservation_id: job.reservation_id,
    });
    throw err;
  }
}