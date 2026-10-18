/**
 * Per-session hot state, as a Durable Object.
 *
 * One instance per session_id. Durable Objects are single-threaded and serialise
 * requests for a given key (developers.cloudflare.com/durable-objects/), so a
 * read-modify-write here needs no lock and no transaction. That single property
 * is why this exists.
 *
 * WHY NOT KV -- the honest version, because "we use a DO" is not an argument:
 *
 *  1. Not atomic. KV's own docs: "KV is not ideal for applications where you
 *     need support for atomic operations or where values must be read and
 *     written in a single transaction. If you need stronger consistency
 *     guarantees, consider using Durable Objects."
 *     (kv/concepts/how-kv-works)
 *
 *  2. Eventually consistent, by up to 60 s. Same page: "changes may take up to
 *     60 seconds or more to be visible in other global network locations", and
 *     it is "not advised to rely on" immediate visibility even at the writer's
 *     own edge. A session counter that is a minute stale is not a session
 *     counter.
 *
 *  3. Write rate. KV Free allows 1,000 writes/day to different keys and 1
 *     write/second to the SAME key (kv/platform/limits). One write/second to
 *     the same key is below the event rate this system must absorb: the watcher
 *     caps a speech run at MAX_UTTERANCE_S = 2.0 s and emits a checkpoint every
 *     2 s during continuous speech (src/watcher.py). Sustained monologue would
 *     hit the cap and then start dropping.
 *
 *  4. Wrong shape. KV is a globally replicated read-through cache optimised for
 *     read-heavy, highly cacheable workloads -- static assets, config, feature
 *     flags. Per-session mutable hot state is none of those things.
 *
 * The summary: KV is not a per-session state store, and reaching for it here
 * would trade a correctness property we need for a cache we do not.
 *
 * Storage is bounded on purpose. This object is meant to hold RECENT state, not
 * history -- durable rows belong in D1. An unbounded session state would grow
 * with session length, which is the same mistake the watcher explicitly avoids
 * with its trimming rule.
 */

/** Keep only this many recent events in hot state. */
const HOT_WINDOW = 200;

/** Keep only this many recent judged outcomes. */
const VERDICT_WINDOW = 50;

/**
 * Durable Object stub. Wrangler injects the real base class; this exists so the
 * file is importable outside the Workers runtime.
 *
 * Declared BEFORE SessionState on purpose -- see the identical note in
 * hotpath/src/budget_do.js. A class declared after the class that extends it
 * throws "Cannot access ... before initialization" in plain Node ESM.
 */
export class ServerDurableObjectStub {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }
}

export class SessionState extends ServerDurableObjectStub {
  constructor(state, env) {
    super(state, env);
    this.state = state;
    this.env = env;
  }

  async fetch(request) {
    const url = new URL(request.url);
    const sessionId = url.searchParams.get("session_id");
    if (!sessionId) return json({ error: "session_id required" }, 400);

    switch (url.pathname) {
      case "/state":
        return json(await this.getState());
      case "/record":
        return json(await this.record(request));
      default:
        return json({ error: "not found" }, 404);
    }
  }

  async getState() {
    const s = (await this.state.storage.get("session")) ?? {
      session_id: null,
      events_seen: 0,
      onsets_seen: 0,
      speech_seconds: 0,
      last_event_t1: 0,
      recent_events: [],
      recent_verdicts: [],
      updated_at: null,
    };
    return s;
  }

  /**
   * Fold one batch of events into hot state.
   *
   * `last_event_t1` is monotonic by design. src/watcher.py keeps an explicit
   * `_emitted_until` for the same reason: segment boundaries jitter by a frame
   * or two as the analysis window realigns, which without an explicit watermark
   * caused a 2.4 s session to emit 71 events. A hot-state counter that counts
   * re-reports is a table of noise that looks exactly like data.
   */
  async record(request) {
    const body = await request.json();
    const sessionId = body.session_id;
    const events = Array.isArray(body.events) ? body.events : [];
    const s = await this.getState();
    s.session_id = sessionId;

    let accepted = 0;
    let stale = 0;
    for (const e of events) {
      if (!Number.isFinite(e.t1)) continue;
      if (e.t1 <= s.last_event_t1) {
        stale += 1;
        continue;
      }
      s.events_seen += 1;
      accepted += 1;
      if (e.kind === "onset") s.onsets_seen += 1;
      if (e.kind === "utterance") {
        s.speech_seconds += Math.max(0, e.t1 - e.t0);
      }
      s.last_event_t1 = Math.max(s.last_event_t1, e.t1);
      s.recent_events.push({
        kind: e.kind,
        t0: e.t0,
        t1: e.t1,
        at: new Date().toISOString(),
      });
    }

    s.recent_events = s.recent_events.slice(-HOT_WINDOW);
    s.updated_at = new Date().toISOString();
    await this.state.storage.put("session", s);
    return { ok: true, accepted, stale, state: await this.getState() };
  }

  /** Record a judged outcome. Called by the queue consumer. */
  async recordVerdict(body) {
    const s = await this.getState();
    s.session_id = body.session_id ?? s.session_id;
    s.recent_verdicts.push({
      job_id: body.job_id,
      gate_probability: body.gate_probability,
      is_directive: body.answers?.is_directive?.choice ?? null,
      is_speech_probability: body.answers?.is_speech?.noul ?? null,
      at: new Date().toISOString(),
    });
    s.recent_verdicts = s.recent_verdicts.slice(-VERDICT_WINDOW);
    await this.state.storage.put("session", s);
    return { ok: true };
  }
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body, null, 2), {
    status,
    headers: { "content-type": "application/json" },
  });
}