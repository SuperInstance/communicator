# workers/ — Cloudflare as an invisible backend

This directory makes Cloudflare an invisible backend for the watcher's event
stream: batched in, rate-limited and budgeted internally, durable rows and
judged work out.

**Nothing here has been deployed.** No `wrangler deploy` was run, no Cloudflare
resource was created or mutated, and no inference was invoked. See
[What was verified, and how](#what-was-verified-and-how) — including one
finding that changes the headline answer.

---

## The short version

The watcher emits ONSET / UTTERANCE / PAUSE far faster than any request budget
allows. So:

- Events arrive **in batches**, not individually.
- Everything cheap runs on everything. `smart-turn-v2` (endpointing) is billed
  per audio-minute and is the only model cheap enough to consider running
  continuously, so it is the **always-on gate**.
- Everything expensive (`clef-omni`, `whisper`) sits **behind** that gate, and
  only for clips the gate likes *and* the budget permits.
- A **neuron accountant** decides whether an expensive call can be afforded,
  and refuses when it cannot.

The economic hinge, stated once so it is not confused later:

> **Batching buys REQUESTS, not NEURONS.**
> `smart-turn-v2` is billed per audio-minute, so ten 2-second clips cost the
> same neurons sent as one batch as sent as ten calls. What batching saves is
> invocations against the 100,000 req/day allowance. Two different resources;
> conflating them is how you "optimise" into an overage.

---

## Layout

| Path | What it is |
| --- | --- |
| `hotpath/costs.json` | **Single source of truth** for every limit and price. Read by the Python modules, the JS Workers, and the tests. |
| `hotpath/budget.py` | `NeuronBudget` — the accountant. Pure Python, the reference implementation, covered by the test suite. |
| `hotpath/batching.py` | `BatchPlanner` — request cadence, concurrency bounds, and the feasibility arithmetic. |
| `hotpath/src/index.js` | The gate Worker: batches in, routed rows + queued jobs out. |
| `hotpath/src/budget_do.js` | The account-wide ledger, as a Durable Object. |
| `hotpath/src/media.js` | Model payload builders. Isolates the one **unverified** assumption. |
| `hotpath/src/pool.js` | Bounded-concurrency map (the 6-connection limit). |
| `tables/src/index.js` | Durable event rows → D1, hot state → per-session DO. |
| `tables/src/session_do.js` | Per-session hot state, bounded. Includes the argument **against** KV. |
| `tables/src/consumer.js` | Queue consumer: the expensive judgement, idempotent. |
| `tables/src/questions.js` | The clef question set. |
| `tables/schema.sql` | D1 schema. Not applied. |
| `test_budget.py` | 39 tests. Offline, no credentials. |
| `wrangler.toml.example` | Configuration, placeholders only. |

---

## The neuron accountant

`hotpath/budget.py`. Three decisions in it matter more than the rest.

**1. Reserve, then settle.** Token cost is only knowable on the way *back*, so
charging on response lets N concurrent calls each observe a balance the others
had not yet claimed. Every expensive call therefore **reserves worst-case cost
up front** and reconciles against the model's reported `usage` afterwards.
Over-reserving costs throughput; under-reserving is the failure this exists to
prevent.

**2. Account-wide, not per-session.** Neurons are billed to the **account**. A
per-session ledger would hand every session the full 10,000/day allowance and
overdraw by a factor of the session count, while each session honestly believed
it was fine. That failure is invisible from any one session's point of view.
So there is exactly **one** ledger, in one Durable Object, that every session
calls into. Single-threaded DO execution is the mutual-exclusion primitive that
makes a shared balance correct.

**3. Fail loud, and on surprise fail closed.** If a settled call costs more than
it reserved, our constants are wrong. The ledger records the overrun, marks
itself **poisoned**, and refuses expensive calls until the daily rollover.
Re-baselining silently would convert one stale constant into an unbounded bill.
A ledger that quietly absorbs a surprise is the exact failure mode this
directory exists to prevent.

Refusal is an **exception**, not a boolean. A caller cannot forget to check it.

### Daily arithmetic

Allowance resets at 00:00 UTC (so the ledger keys on the UTC day — local time
would double-count or double-refill depending on where it runs).

| | Free | Paid |
| --- | --- | --- |
| Included neurons/day | 10,000 | 10,000 |
| Beyond that | calls **fail** — documented hard stop | billed at $0.011/1,000 |
| Our ceiling | the allowance | + 40,000 overage/day ($0.44/day) |

The 10,000-neuron allowance is worth **$0.11/day**. That number is the most
useful one in this file.

---

## ⚠️ The finding that changes the answer

**Cloudflare's own pricing page contradicts itself about `smart-turn-v2`, by
~60×.** Both figures are current:

- **0.51 neurons per audio-minute** (pricing page, audio table)
- **$0.000338 per audio-minute** → **30.73 neurons** at the page's own stated
  $0.011/1,000 unit price

The page asserts the two columns are equivalent. Recomputing
price ÷ published-neurons for **every** model on that page yields exactly
$0.000011 — the stated unit price — for whisper, whisper-large-v3-turbo,
melotts, deepgram nova-3, deepgram flux, clef, clef-omni and clef-flash.
`smart-turn-v2` alone yields $0.0006627, **60.2× the stated rate**. One outlier
out of nine is the signature of a typo, not of a different billing unit. The
live catalog endpoint independently confirms the **USD** figure
(`price: 0.000338`, `unit: "per audio minute"`).

| | at published 0.51 | at the conservative 30.73 |
| --- | --- | --- |
| 24/7 audio | 734 neurons/day | **44,247 neurons/day** |
| vs 10,000 allowance | 0.07× | **4.42× over** |
| Affordable audio/day | 19,608 min (326.8 h) | **325 min (5.4 h)** |
| Days of 24/7 inside the allowance | 13.62 | **0.23** |

**So: continuous 24/7 endpointing does not fit in the free allowance — it is
4.4× over every single day.** Under the more forgiving reading it would fit for
13.6 days, which is the figure circulating. This directory plans for the
expensive one.

The rule applied, in `_rate()` and in both implementations:

```
neurons_per_unit = max(published_neurons, usd_per_unit / usd_per_neuron)
```

Taking the max is the whole point: under-estimating by 60× is the difference
between "the ledger refuses" and "the ledger believes it has 60× the money it
has". The rule is uniform across models rather than a special case for one
slug, so it keeps meaning if Cloudflare re-prices something else tomorrow.

**Not verified:** I did not run a `smart-turn-v2` inference to measure the real
charge, because that spends neurons and this task forbids mutating spend. The
real per-call charge for clef-omni *is* reported back in the response `usage`,
and `settle()` reconciles the ledger against reported usage — so a wrong
constant self-corrects from measurement rather than from assumption. **This
discrepancy should be resolved with one cheap smart-turn call before anyone
depends on the free tier.**

---

## What is NOT viable, and why

**Per-event invocation on the free tier — not viable.** Workers Free allows
100,000 requests/day **for the account** (all Workers combined) = 1.157 req/s.
The watcher caps a speech run at `MAX_UTTERANCE_S = 2.0 s` and emits a
checkpoint every 2 s during continuous speech, so per-event invocation would
spend the account's entire daily request budget on one feature, leaving nothing
for the tables plane, the consumers, or D1. Batching is the difference between
working and not.

**Whisper as a streaming path — not viable, and the pattern is a
misnomer.** `@cf/openai/whisper` and `whisper-large-v3-turbo` are **batch-only:
no streaming, no partial results, no WebSocket path.** The "streaming Whisper"
pattern in circulation is *client-side chunking*, which is not streaming. So
there is no incremental-transcript architecture to build here, and
`experiments/clef_audio_native.py`'s Whisper arm (`arm_c_via_stt`) is a
batch comparison only. Neither model is wired into the gate.

**Continuous 24/7 endpointing inside the free allowance — not viable.** 4.42×
over, per the table above. Paid, it costs ~$0.49/day (~$178/year) at the
conservative rate, which is affordable but is a *deliberate purchase*, not a
free-tier default.

**KV for per-session state — wrong tool.** Not "imperfect", wrong:
- **Not atomic.** KV's own docs: *"KV is not ideal for applications where you
  need support for atomic operations or where values must be read and written in
  a single transaction. If you need stronger consistency guarantees, consider
  using Durable Objects."*
- **Eventually consistent by up to 60 s**, and immediate visibility is *"not
  advised to rely on"* even at the writer's own edge.
- **Write-rate capped** at 1 write/second per key and 1,000 writes/day on Free
  — *below* the watcher's sustained event rate.
- **Wrong shape.** It is a read-heavy, globally-replicated read-through cache for
  static assets, config and flags. A mutable per-session counter is none of
  those.

**Audio in a Queues message — does not fit.** Messages cap at 128 KB. 16-bit
mono at 16 kHz is 32,000 bytes/s, so ~4 s, or ~3 s after base64. Messages carry
**job descriptors**; audio never travels in one.

**Cron triggers — no failure to defend against.** The watcher POSTs, the gate
reacts, the queue delivers. A poller would be mechanism without justification.

---

## Measured constraints, and what each one forced

| Constraint | Value | Source | What it forced |
| --- | --- | --- | --- |
| Free requests/day | 100,000 (account-wide) | Workers platform limits | Batching. Our share is 10% = 1 batch/8.64 s. |
| Free CPU/request | 10 ms | Workers limits | Batching, *and* not doing real work per request. |
| HTTP wall-clock | **no limit** | Workers limits | Awaiting inference inside a request is fine. |
| Awaiting network I/O | **not CPU** | Workers limits | The gate awaits several model calls ~free in CPU. |
| Open connections | 6/invocation | Workers limits | `pool.js` caps in-flight model calls at 6. Not a preference. |
| Neuron allowance | 10,000/day @ $0.011/1000 | Workers AI pricing | The accountant; the `$0.11/day` framing. |
| Reset boundary | 00:00 UTC | Workers AI pricing | Ledger keys on the **UTC** day. |
| Free overage | calls fail with an error | Workers AI pricing | Free refuses at the allowance. |
| ASR rate limit | 720 req/min | Workers AI limits | Whisper (not wired) would fit. |
| Text Generation limit | 300 req/min | Workers AI limits | clef-omni's ceiling. |
| smart-turn-v2 limit | **undocumented** | Workers AI limits | Its task type ("Dumb Pipe") is absent from the by-task-type table; planned against 300/min. |
| clef-omni media | ≤4 clips, 8 MiB, 300 s each; **no URLs** | live input schema | Clips arrive inline as base64. |
| clef-omni usage | `usage` is **required** in output | live output schema | Measured settlement, not estimation alone. |
| Queues | at-least-once, 100 retries | Queues limits | Idempotent consumers: a retry costs 1 D1 write, **0** model calls. |
| Queues message | 128 KB | Queues limits | Descriptors, not audio. |
| KV | eventually consistent, 1 write/s/key | KV limits + how-kv-works | DO per session, **not** KV. |
| DO wall clock | unlimited (RPC/HTTP) | Workers limits | Long-lived ledger object is fine. |
| Queue consumer | 15 min wall, 5 min CPU | Queues limits | Batch judgement fits. |

### Two more decisions worth stating

**Idempotency keys are deterministic, not random.** Derived from
`(session, kind, t0, t1)`, never a fresh uuid per attempt. A fresh id turns every
retry into a duplicate row while still appearing to work — Queues retries cost a
read op each, so this is cost control, not hygiene.

**Events are deduplicated on the watcher's own coordinates.** `t0`/`t1` are
absolute seconds, never chunk-local (`src/watcher.py` is explicit: chunk-local
timestamps make every downstream table wrong the moment the stream is chunked
differently, and fail silently).

---

## Tests

```
$ /home/proart/projects/glasses/.venv/bin/python -m pytest workers/ -q
39 passed
```

Pure Python, offline, no credentials. The two required behaviours:

- **Refuses to overdraw** — `test_repeated_audio_reservations_stop_at_the_allowance`
  spends the day in realistic 2.0 s clips (the watcher's own `MAX_UTTERANCE_S`)
  and asserts `committed <= 10,000`, that it refused *before* crossing, and that
  a refusal is not recorded as an overrun. Also covered: concurrent reservations
  cannot all pass, free never exceeds, and paid still stops at its own cap.
- **Recovers at rollover** — `test_refill_at_0000_utc` fills the day, crosses
  midnight, and asserts the full allowance returns and spending resumes. Plus:
  no rollover *within* a day, UTC-not-local keying, a stale reservation not
  corrupting the new day, and a poisoned ledger self-recovering.

The suite also pins the numbers themselves (`test_every_neuron_and_price_claim_carries_a_source`)
so a price cannot be hard-coded somewhere and drift.

**Scope of coverage, honestly:** these 39 tests execute `budget.py` and
`batching.py`. They do **not** execute the JavaScript. What is genuinely shared
is the arithmetic's input — both sides read the same `costs.json` and apply the
same `max(published, implied)` rule, so the tests do guard the constants. What is
**not** machine-verified by pytest is the JS translation of them
(`budget_do.js`), which is a mirror. I validated the mirror separately against a
fake DO storage and the real `costs.json` (22 checks: reserve/settle netting,
poisoning, exhaustion at 9,763 clips with `committed` 9,999.68, paid cap, day
rollover, `mapLimit` peak concurrency ≤ 6, payload shapes). That harness lives in
the gitignored `.scratch/` — adding a JS test runner to a Python repo was out of
scope. **Treat any divergence between `budget.py` and `budget_do.js` as a bug in
the latter.**

---

## What was verified, and how

**Verified live, read-only** (catalog listing only — no inference, no writes),
2026-10-18:
- All four slugs exist: `@cf/cloudflare/clef`, `@cf/cloudflare/clef-omni`,
  `@cf/pipecat-ai/smart-turn-v2`, `@cf/openai/whisper-large-v3-turbo`.
  (`@cf/cloudflare/clef-flash` also exists, is 4× cheaper than clef, and is
  **not** wired in — the System-One primitives are the same, but nothing
  justifies the switch yet.)
- Per-unit prices, from `properties[].value` in the catalog.
- clef-omni's live input schema: `audio` is an **array** (max 4, 8 MiB, 300 s
  each), remote URLs rejected, 1–64 questions, `model`/`state`/`questions`
  required.
- clef-omni's output schema **requires** `usage`.
- smart-turn-v2's output is exactly `{is_complete, probability}` — **no
  transcript**, which is why the gate cannot use it for text.

**Taken from docs:** neuron unit price, free/paid allowances, reset time, ASR and
Text-Generation rate limits, Workers request/CPU/wall-clock/connection limits,
KV and Queues limits, DO wall clock.

**Could NOT verify:**
- **The real smart-turn-v2 charge** (would require spending neurons). Hence the
  conservative rate and the loud conflict note above. Resolve with one cheap call.
- **`smart-turn-v2`'s rate limit** — its task type is absent from the docs table.
- **smart-turn-v2's exact PCM container** (raw vs WAV) and sample-rate
  expectation. The input schema documents base64 and a `dtype`, but not the
  container. I did not resolve this by inference. It is **isolated in
  `hotpath/src/media.js`** so fixing it is a change to one file — and until it is
  measured, **treat the smart-turn audio path as unverified**: the plumbing and
  budget accounting are sound, but do not assume `probability` is calibrated.
- **`clef`'s confidence scale.** `experiments/clef_audio_native.py` records that
  clef's confidence does *not* reliably follow Jev's
  `(p_max - 1/K)/(1 - 1/K)` formula. So the dead-band rule inherited from the
  Jev ecosystem **must be re-measured** for clef rather than assumed — which is
  why `gate_probability_threshold` is documented as an uncalibrated knob.

## What was deliberately NOT built

- **No deployment.** No `wrangler deploy`, no Worker/D1/Queue/DO created, no D1
  schema applied, no token written anywhere. `wrangler.toml.example` contains no
  account id and no credential; tokens belong in `wrangler secret put`.
- **No R2 integration.** Audio fits comfortably inline over HTTP for 2 s clips
  (Free body limit is 100 MB); where audio is *persisted* is a decision for
  whoever wires the watcher in. The gate does not care.
- **No KV anywhere** — see the table above.
- **No cron trigger** — nothing to poll.
- **No streaming-transcription architecture** — it does not exist on Workers AI
  (see above).
- **No JS test runner.** The repo is Python with no JS toolchain; adding one was
  out of scope. The mirror was validated out-of-tree instead.
- **No wiring to `src/watcher.py`.** The watcher is unmodified, as instructed.
  The batch contract is `POST /gate` with `{session_id, events, clips}`;
  `audio_b64` is supplied by the caller.