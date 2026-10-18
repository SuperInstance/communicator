/**
 * The hotpath gate: a Worker that decides which watcher events are worth
 * paying an expensive model for.
 *
 * The economic shape, in one paragraph. smart-turn-v2 (endpointing) is billed
 * per AUDIO MINUTE and is cheap enough per minute to consider running on
 * everything. clef-omni and whisper are the expensive tier and must not be.
 * So cheap-first, and let the cheap tier do the selecting: smart-turn returns
 * P(end-of-turn) and only clips above a threshold get promoted to an
 * expensive judgement. The gate never decides anything itself -- it spends the
 * budget DO's decision and the model's probability, nothing else.
 *
 * Why this is a batched HTTP Worker and not per-event:
 *   Workers Free allows 100,000 requests/day for the ACCOUNT
 *   (developers.cloudflare.com/workers/platform/limits/#account-plan-limits),
 *   which is 1.157 req/s shared across every Worker on it. The watcher emits
 *   an utterance every MAX_UTTERANCE_S = 2.0 s during continuous speech, so
 *   per-event invocation would spend the account's entire daily request budget
 *   on one feature. Batching is not an optimisation here; it is the difference
 *   between working and not.
 *
 * Why awaiting model calls is safe on the free tier's 10 ms CPU budget:
 *   "Waiting on network requests (such as fetch() calls, KV reads, or database
 *   queries) does not count toward CPU time" (#cpu-time), and HTTP-triggered
 *   Workers have NO wall-clock limit (#duration). So a Worker may await several
 *   inference calls inside one invocation almost for free in CPU terms. That is
 *   what makes the whole design affordable rather than merely legal.
 *
 * Concurrency is capped at 6 in-flight model calls, which is a platform limit
 * ("up to six connections simultaneously waiting for response headers",
 * #simultaneous-open-connections), not a preference. A Workers AI call sits in
 * exactly that state for the whole inference.
 */

import { budgetDO } from "./budget_do.js";
import { buildSmartTurnPayload, buildClefPayload } from "./media.js";
import { mapLimit } from "./pool.js";

/** Watcher event kinds, mirrored from src/watcher.py. */
const ONSET = "onset";
const UTTERANCE = "utterance";
const PAUSE = "pause";

/**
 * Kinds that carry no audio and need no model at all.
 *
 * ONSET and PAUSE exist to spend attention and to record a duration; there is
 * nothing to judge about them. Routing them straight to D1 rather than through
 * the gate is what keeps the request cost proportional to SPEECH rather than to
 * event volume.
 */
const ROUTING_ONLY = new Set([ONSET, PAUSE]);

export default {
  /**
   * POST /gate
   *
   * Body:
   *   {
   *     session_id: string,
   *     events: [{ kind, t0, t1, data?, origin? }],          // absolute seconds
   *     clips:   [{ event_id, audio_b64, sample_rate }]       // speech only
   *   }
   *
   * `t0`/`t1` are absolute seconds since the watcher started -- deliberately
   * NOT chunk-local. src/watcher.py calls this out explicitly: a chunk-local
   * timestamp makes every downstream table wrong the moment the stream is
   * chunked differently, and it fails silently.
   */
  async fetch(request, env) {
    if (request.method !== "POST") {
      return json({ error: "POST only" }, 405);
    }

    const batch = await request.json().catch(() => null);
    const problems = validate(batch);
    if (problems.length) {
      // Fail loudly on a malformed batch. Silently dropping events would leave
      // a hole in the baseline table that looks exactly like "nothing happened",
      // which is the failure mode this project keeps getting bitten by.
      return json({ error: "invalid batch", problems }, 400);
    }

    const costs = env.COSTS;
    const threshold = costs.OURS.gate_probability_threshold.value;
    const minClipS = costs.OURS.hotpath_min_clip_s.value;
    const concurrency = costs.plans.max_simultaneous_open_connections;
    const reserveTokens = costs.OURS.expected_input_tokens_per_judgement.value;

    const events = batch.events;
    const report = {
      session_id: batch.session_id,
      received: events.length,
      clips_received: batch.clips.length,
      routed_without_model: 0,
      skipped: [],
      endpointed: 0,
      promoted: [],
      budget: null,
    };

    // -- 1. Routing-only events go straight to the tables plane. ------------
    const routingOnly = events.filter((e) => ROUTING_ONLY.has(e.kind));
    report.routed_without_model = routingOnly.length;
    if (routingOnly.length) {
      await env.DB.prepare(
        `INSERT OR IGNORE INTO events
           (session_id, event_id, kind, t0, t1, duration_s, origin, data_json, recorded_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)`
      ).bind(
        batch.session_id,
        deterministicId(batch.session_id, "event", routingOnly[0].t0, routingOnly[0].t1, routingOnly[0].kind),
        "batch",
        routingOnly[0].t0,
        routingOnly[routingOnly.length - 1].t1,
        0,
        "acoustic",
        JSON.stringify(routingOnly),
        new Date().toISOString()
      ).run();
    }

    // -- 2. Speech clips -> the always-on gate (smart-turn-v2). --------------
    const clipByEvent = new Map(batch.clips.map((c) => [c.event_id, c]));
    const speech = events.filter((e) => e.kind === UTTERANCE);

    const candidates = [];
    for (const ev of speech) {
      const clip = clipByEvent.get(ev.event_id);
      const durationS = Math.max(0, ev.t1 - ev.t0);
      if (!clip) {
        report.skipped.push({ event_id: ev.event_id, why: "no audio supplied" });
        continue;
      }
      // Clip filter mirrors batching.py:clip_is_worth_endpointing. Sub-threshold
      // clips are mostly clicks and breaths, and every clip not sent frees a
      // slot on the 6-connection budget. Not the mechanism that saves neurons.
      if (durationS < minClipS || durationS > costs.OURS.hotpath_max_clip_s.value) {
        report.skipped.push({ event_id: ev.event_id, why: "duration out of range" });
        continue;
      }
      candidates.push({ ev, clip, durationS });
    }

    // Reserve smart-turn budget for the whole batch up front. Pessimistic on
    // purpose: per-clip reservations interleaved with inference would let a
    // burst of clips each observe a balance the others had not yet claimed.
    let reservation = null;
    const audioMinutes = candidates.reduce((s, c) => s + c.durationS, 0) / 60.0;
    try {
      reservation = await budgetDO(env.BUDGET).reserve({
        model: "smart-turn-v2",
        units: audioMinutes,
      });
      report.endpointed = candidates.length;
    } catch (err) {
      // The budget said no. That is a normal, expected outcome, not an error --
      // it is the system working. Record it and stop; do NOT fall through to an
      // unbudgeted call.
      report.skipped.push({ why: "budget_refused", detail: String(err.message || err) });
      report.budget = await budgetDO(env.BUDGET).snapshot();
      return json({ ...report, gate: "budget_refused" }, 200);
    }

    // Endpoint every candidate, 6 at a time.
    const endpointed = await mapLimit(candidates, concurrency, async (c) => {
      try {
        const payload = buildSmartTurnPayload(c.clip);
        const result = await env.AI.run("@cf/pipecat-ai/smart-turn-v2", payload);
        return {
          ...c,
          is_complete: Boolean(result?.is_complete),
          probability: Number(result?.probability ?? 0),
          ok: true,
        };
      } catch (err) {
        // A failed inference still consumed wall-clock and probably some
        // neurons. Release the slice so a flaky provider cannot silently eat
        // the day's allowance through retries.
        return { ...c, ok: false, error: String(err.message || err) };
      }
    });

    // Settle smart-turn against what we actually sent. smart-turn-v2's schema
    // reports no `usage`, so the measured quantity is our own audio duration --
    // which is the billed quantity anyway, since it is billed per audio minute.
    await budgetDO(env.BUDGET).settle({
      reservation_id: reservation.reservation_id,
      actual_units: audioMinutes,
      actual_neurons: reservation.neurons,
      units: audioMinutes,
    });

    // -- 3. Promote the clips smart-turn liked, budget permitting. -----------
    const promoted = endpointed.filter((r) => r.ok && r.probability >= threshold);
    for (const cand of promoted) {
      let clefReservation = null;
      try {
        clefReservation = await budgetDO(env.BUDGET).reserve({
          model: "clef-omni",
          units: reserveTokens / 1_000_000.0,
        });
      } catch (err) {
        report.skipped.push({
          event_id: cand.ev.event_id,
          why: "judgement_budget_refused",
          detail: String(err.message || err),
        });
        continue;
      }

      // Queue a DESCRIPTOR, not audio. A Queues message is capped at 128 KB
      // (queues/platform/limits), and 16-bit 16 kHz mono is 32,000 bytes/s --
      // about 4 s, or 3 s after base64. Audio cannot travel in the message, so
      // the job carries the session, the event id and the verdict that earned
      // it; the consumer re-reads audio from wherever the watcher stored it.
      const job = {
        job_id: deterministicId(batch.session_id, "job", cand.ev.t0, cand.ev.t1, cand.ev.kind),
        session_id: batch.session_id,
        event_id: cand.ev.event_id,
        t0: cand.ev.t0,
        t1: cand.ev.t1,
        gate_probability: cand.probability,
        gate_is_complete: cand.is_complete,
        questions: buildClefPayload(cand.clip).questions,
        reservation_id: clefReservation.reservation_id,
        reserved_neurons: clefReservation.neurons,
        queued_at: new Date().toISOString(),
      };
      await env.JUDGE_QUEUE.send(job);
      report.promoted.push({
        event_id: cand.ev.event_id,
        probability: cand.probability,
        job_id: job.job_id,
      });
    }

    report.budget = await budgetDO(env.BUDGET).snapshot();
    return json({ ...report, gate: "ok" }, 200);
  },
};

/**
 * Stable id for a (session, kind, t0, t1) tuple.
 *
 * This is the idempotency key for the whole plane. Queues is at-least-once and
 * a message can be redelivered up to 100 times, each redelivery costing another
 * billable request, so consumers key on this and INSERT OR IGNORE on it.
 *
 * Derived from the event's own coordinates rather than generated fresh, so a
 * retry of the SAME batch produces the SAME ids. A random uuid per attempt
 * would make every retry a new row and defeat the point.
 */
function deterministicId(sessionId, kind, t0, t1, eventKind) {
  const raw = `${sessionId}|${kind}|${t0}|${t1}|${eventKind}`;
  // FNV-1a 32-bit: short, dependency-free, and adequate for a deduplication
  // key. NOT a security primitive and not used as one.
  let h = 0x811c9dc5;
  for (let i = 0; i < raw.length; i++) {
    h ^= raw.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return `${kind}_${h.toString(16).padStart(8, "0")}`;
}

function validate(batch) {
  const problems = [];
  if (!batch || typeof batch !== "object") return ["body must be a JSON object"];
  if (typeof batch.session_id !== "string" || !batch.session_id) {
    problems.push("session_id is required");
  }
  if (!Array.isArray(batch.events)) {
    problems.push("events must be an array");
  } else if (batch.events.length) {
    for (const e of batch.events) {
      if (typeof e.kind !== "string") problems.push("event.kind must be a string");
      if (!Number.isFinite(e.t0) || !Number.isFinite(e.t1)) {
        problems.push(`event t0/t1 must be finite numbers (got ${e.t0}, ${e.t1})`);
      }
      if (e.t1 < e.t0) problems.push("event t1 precedes t0");
    }
  }
  if (!Array.isArray(batch.clips)) problems.push("clips must be an array");
  return problems;
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body, null, 2), {
    status,
    headers: { "content-type": "application/json" },
  });
}