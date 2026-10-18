-- Durable event rows for the tables plane.
--
-- Applied with:  wrangler d1 execute communicator --file=workers/tables/schema.sql
-- (NOT run as part of this task -- see README, "what was deliberately not built".)
--
-- WHY D1 AND NOT KV: these are append-only rows that must be queryable and
-- aggregateable. KV is an eventually-consistent read-through cache with no
-- query surface and a 1-write/second-per-key ceiling (kv/platform/limits);
-- a baseline table needs ORDER BY, WHERE on time ranges, and COUNT. The two
-- have no overlap in purpose.

PRAGMA foreign_keys = ON;

-- One row per judged outcome.
--
-- job_id is the PRIMARY KEY, and that is the whole idempotency mechanism.
-- Queues delivery is at-least-once and a message may be redelivered up to 100
-- times (queues/platform/limits: "Message retries | 100"), with each
-- redelivery costing another billable request against the 100,000/day free
-- allowance. Consumers therefore do `INSERT OR IGNORE` on a deterministic
-- job_id, so a redelivery is a no-op instead of a duplicate row.
--
-- The key must be DETERMINISTIC (derived from session + event coordinates, not
-- generated per attempt). A fresh uuid on each attempt would make every retry a
-- new row, which defeats idempotency entirely while looking correct.
CREATE TABLE IF NOT EXISTS verdicts (
  job_id             TEXT PRIMARY KEY,
  session_id         TEXT NOT NULL,
  event_id           TEXT,
  t0                 REAL NOT NULL,   -- absolute seconds, NOT chunk-local
  t1                 REAL NOT NULL,
  duration_s         REAL,
  gate_probability   REAL,            -- smart-turn-v2 P(end-of-turn)
  gate_is_complete   INTEGER,         -- smart-turn-v2 is_complete
  model              TEXT NOT NULL,   -- which model judged it
  is_speech          REAL,            -- clef noul: P(yes)
  is_directive       TEXT,            -- clef choice: chosen option
  is_directive_conf  REAL,            -- clef choice: confidence
  usage_input_tokens INTEGER,
  usage_output_tokens INTEGER,
  neurons            REAL,            -- measured from usage, not estimated
  status             TEXT NOT NULL DEFAULT 'judged'
                       CHECK (status IN ('judged','failed','skipped','refused')),
  created_at         TEXT NOT NULL,
  updated_at         TEXT NOT NULL
);

-- Time-range queries are the only access pattern that matters for a baseline
-- table ("what happened between t and t"), so the index is on exactly that.
CREATE INDEX IF NOT EXISTS idx_verdicts_session_time
  ON verdicts (session_id, t0);

CREATE INDEX IF NOT EXISTS idx_verdicts_created
  ON verdicts (created_at);

-- Raw watcher events, including the ones that never reached a model.
--
-- This table is deliberately NOT filtered to judged events. Recording the
-- events that were considered and skipped is what makes the gate's selectivity
-- measurable later; a table of only successes cannot answer "what did we throw
-- away and was that right?".
CREATE TABLE IF NOT EXISTS events (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  dedupe_key    TEXT NOT NULL UNIQUE, -- deterministic; see note above
  session_id    TEXT NOT NULL,
  event_id      TEXT,
  kind          TEXT NOT NULL,
  t0            REAL NOT NULL,
  t1            REAL NOT NULL,
  duration_s    REAL,
  origin        TEXT,
  data_json     TEXT,
  recorded_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_session_time
  ON events (session_id, t0);

-- Every budget decision, for after the fact.
--
-- A refusal is a first-class outcome, not an absence. If the gate starts
-- refusing 40% of judgements because the day ran out, that must be visible in
-- the data -- otherwise the only symptom is that answers quietly stopped
-- appearing and nobody can tell a budget stop from a bug.
CREATE TABLE IF NOT EXISTS budget_events (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  day           TEXT NOT NULL,        -- UTC day, matching the reset boundary
  session_id    TEXT,
  model         TEXT NOT NULL,
  units         REAL,
  neurons       REAL,
  decision      TEXT NOT NULL
                  CHECK (decision IN ('reserved','refused','settled','released')),
  reservation_id TEXT,
  reason        TEXT,                 -- populated on refusal
  created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_budget_day
  ON budget_events (day, decision);