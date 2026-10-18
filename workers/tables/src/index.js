/**
 * The tables plane: durable storage for watcher events and judged outcomes.
 *
 * Two stores, because they answer different questions and have opposite shapes:
 *
 *   Durable Object (per session) -> "what is the session doing RIGHT NOW?"
 *     Hot, mutable, bounded, read-modify-write. Needs single-threaded
 *     serialisation; that is exactly what a DO gives and exactly what KV cannot
 *     (see session_do.js for the full argument against KV).
 *
 *   D1 -> "what happened, and what did we decide?"
 *     Append-only, queryable, durable, aggregateable. Needs ORDER BY and time
 *     ranges. D1 gives that; KV has no query surface at all.
 *
 * Neither replaces the other. Using D1 for hot state would mean a read and a
 * write per event with no atomicity between them; using a DO for history would
 * mean unbounded per-session storage, which is the failure the watcher
 * explicitly designs against with its trimming rule.
 *
 * This Worker is also the only place events are written before any model runs,
 * so a judgement failure can never lose the underlying observation. That
 * ordering is deliberate: the raw record is the expensive thing to recreate,
 * not the verdict.
 */

import { SessionState } from "./session_do.js";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method !== "POST") {
      return json({ error: "POST only" }, 405);
    }

    const body = await request.json().catch(() => null);
    if (!body || typeof body.session_id !== "string" || !body.session_id) {
      // Refuse rather than accept an unusable batch. A session_id-less row is
      // unjoinable with any judgement that follows it, and it would look like a
      // legitimate record of a session that never existed.
      return json({ error: "session_id is required" }, 400);
    }
    const events = Array.isArray(body.events) ? body.events : [];

    // -- 1. Durable rows first. --------------------------------------------
    // Written BEFORE the DO update so a DO failure cannot lose the record.
    if (events.length) {
      const stmt = env.DB.prepare(
        `INSERT OR IGNORE INTO events
           (dedupe_key, session_id, event_id, kind, t0, t1, duration_s,
            origin, data_json, recorded_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)`
      );
      const bindings = events
        .filter((e) => Number.isFinite(e.t0) && Number.isFinite(e.t1))
        .map((e) => [
          dedupeKey(body.session_id, e),
          body.session_id,
          e.event_id ?? null,
          e.kind,
          e.t0,
          e.t1,
          Math.max(0, e.t1 - e.t0),
          e.origin ?? "acoustic",
          e.data ? JSON.stringify(e.data) : null,
          new Date().toISOString(),
        ]);
      // D1's parameter-count ceiling is well above what a single statement can
      // carry; 50 events per statement keeps each under it with room to spare.
      for (const chunk of chunked(bindings, 50)) {
        await env.DB.batch(chunk.map((b) => stmt.bind(...b)));
      }
    }

    // -- 2. Hot state, via the per-session Durable Object. -----------------
    // The DO id is the session id, so every write for a session lands on one
    // instance and is serialised there. No lock, no transaction, no race.
    const stub = getSessionStub(env, body.session_id);
    const hot = await stub.fetch(
      `https://do.internal/record?session_id=${encodeURIComponent(body.session_id)}`,
      {
        method: "POST",
        body: JSON.stringify({ session_id: body.session_id, events }),
      }
    );
    const hotResult = await hot.json();

    return json({
      ok: true,
      session_id: body.session_id,
      events_written: events.length,
      events_accepted: hotResult?.accepted ?? null,
      events_stale: hotResult?.stale ?? null,
    });
  },
};

/**
 * Deterministic dedupe key.
 *
 * `INSERT OR IGNORE` on a UNIQUE column is what makes redelivery safe. The key
 * is derived from the event's own coordinates rather than generated fresh,
 * because a fresh id per attempt turns every retry into a duplicate row while
 * still appearing to work.
 */
function dedupeKey(sessionId, e) {
  const raw = `${sessionId}|${e.kind}|${e.t0}|${e.t1}`;
  let h = 0x811c9dc5;
  for (let i = 0; i < raw.length; i++) {
    h ^= raw.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return `evt_${h.toString(16).padStart(8, "0")}`;
}

/**
 * Resolve the Durable Object for a session.
 *
 * A fixed namespace id (from wrangler.toml) with the session as the DO id is
 * what gives "one instance per session". Chosen over idFromName() hashing
 * because the session id is already a unique opaque string and using it
 * verbatim keeps the mapping inspectable -- you can look up an object's id and
 * know exactly which session it is.
 */
function getSessionStub(env, sessionId) {
  const id = env.SESSIONS.idFromName(sessionId);
  return env.SESSIONS.get(id);
}

function* chunked(iterable, size) {
  let buf = [];
  for (const item of iterable) {
    buf.push(item);
    if (buf.length === size) {
      yield buf;
      buf = [];
    }
  }
  if (buf.length) yield buf;
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body, null, 2), {
    status,
    headers: { "content-type": "application/json" },
  });
}

export { SessionState };