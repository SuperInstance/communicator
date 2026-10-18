# Prior-art audit — what is built, what is described

**Date:** 2026-10-18 · **Method:** read the source, not the README · **Scope:** the
repositories this project should learn from rather than re-derive.

Every entry was verified by reading code and counting files. Where a
repository's own documentation claims something its source does not do, both
are recorded, because **that gap is the most reliable thing this audit found.**

---

## The finding that matters more than any single repo

**In this codebase family, scaffolding consistently outruns execution.** The
recurring failure is never "the idea is wrong" — it is a claim with no run
behind it, or a guard that cannot fail, or a file that was committed and
never imported.

| instance | what it is | how it was found |
|---|---|---|
| `project-JEPA` | 43 blobs, **0 source files** — documentation only | file-extension count |
| `plato-audio-jepa` | 2 source files; a 16-bin DFT with `serde`+`uuid` | reading the deps |
| `fleet-jepa-midi` | 0.0146 same/cross-room gap; own README says *"You Are Not Actually Running JEPA"* | the repo's own text |
| `exoJ` smoke test | asserts sha256 of the **empty string**; says so in a comment | `smoke.mjs:9-13` |
| `quilt-dba` | two files import symbols a module does not export — ESM **link-time SyntaxError**, referenced by no CI target | import/export mismatch |
| `quilt-pincher` | `allReflexes()` returns `[]` unconditionally; `memoryUsage()` returns a hardcoded `n*4096` | reading the functions |
| `exoJ` conservation | `eta` is *defined* as `1-gamma`, so `gamma+eta=1` **cannot be violated** | `jev_backends.mjs:18` |
| `quilt-dba` β₁ | `V`/`E` hardcoded to 24/26 and 22/26 against a real 32-cell graph | `skills.graph` |
| this repo, Phase 0–3 | `question_health.py` 161 lines → `AttributeError`; `score.py` 248 lines, published → `TypeError` | first run |

**The operational rule this implies, for this project:** a mechanism is not
adopted because a document describes it. It is adopted when (a) a run produced
a number, and (b) a sabotage makes the guarding test go red. Anything else is
a hypothesis to test, not a foundation to build on.

---

## 1. `quilt-pincher` — the one mechanism worth adopting whole

**Verified real:** 21 source files, ~1,300 LOC TypeScript, CI green 6× on
2026-10-15 (typecheck, test, build, `npm pack`).

### The reflex path is genuinely LLM-free

`src/core/engine.ts:36-68` is the whole hot path:

```
embed(trigger) → store.query(query, 5) → score vs threshold → executeDirectly
```

The LLM is reachable **only** at `engine.ts:53`, and only when
`score < confirmThreshold`. Thresholds: `hitThreshold = 0.80`,
`confirmThreshold = 0.55`. Three tiers — hit, confirm, compile — with the
expensive tier unreachable on the common path.

### HDC: a model-free embedding that is actually deterministic

`src/hdc/hdc-embedder.ts` hashes tokens to ±1 hypervectors, binds position by
permutation, bundles by sign. Deterministic, zero marginal cost, no API, no
model. It is CI-pinned by a hash regression test (`test/hdc-spine.mjs`).

This is the answer to "what runs continuously." It is not a neural
embedding — it is hyperdimensional computing, which needs no training and no
GPU, and therefore runs on an UNO Q. **Adoptable as-is.**

### What does not exist

**Federation is a naming choice, not an implementation.** Zero network
primitives in `src/` — no mqtt, grpc, websocket, fetch, http or socket. The
whole of "federates across cloud, workstation, and ESP32" is one constructor
argument: `MemoryReflexStore({upstream})` writes through to an **in-process
object reference**. And `query()` never consults `upstream` at all, so **reads
do not federate even in principle.**

**ESP32 is aspirational.** `esp32.ts:4` says the Rust port lives in
`src/port/no_std/`; that directory does not exist. `docs/ESP32_PORT.md` is
non-compiling pseudo-Rust, its 128 KB budget assumes 64-dim f32 while the CLI
runs 1024-dim, and it claims `buildNail()` emits a flatbuffer while it emits a
plain JS object.

**There is no scheduler.** Nothing measures RAM, GPU, latency or cost;
the platform is chosen by the caller at construction time. So
"resource harmony" is manual deployment.

**Also:** `CloudflareAIEmbedder.embed()` silently falls back to a character
histogram when the API is unavailable (`embedding-adapter.ts:15-19`, the
comment admits it) — a silent degradation in the similarity path, which is the
one path where degradation is least visible and most dangerous.

### Adoptable, with caveats

| idea | caveat |
|---|---|
| threshold-tiered reflex (hit/confirm/compile) | requires the LLM tier be *genuinely* optional — here it is |
| HDC deterministic embedding | needs a frozen seed and a hash-pinned test; they have both |
| hash-pinned JSONL audit ledger of hashes + exit codes | sound, cheap, portable |
| precompiled bundle on the constrained node | right instinct; they did **not** build the binary encoder |

---

## 2. `quilt-dba` — the conservation law, done correctly and left inert

**Verified real:** 40 hand-written `.mjs` vs 8 markdown, green CI, 8
experiment runs completed. The vendored `@quilt/core` API genuinely matches
how it is used (`loadSheet`, `register`, `get/set/call`, `boundRuntime`) —
not a facade.

### The law

```
kernels.mjs:169   gammaScaled + etaScaled <= C_SCALED + C_TOL     C = log2(3) ≈ 1.585 bits
core.mjs §7       breach → restore snapshot, mark aversion, log refusal, learn nothing
```

γ = compute-cost proxy for one transition; η = surprise (scaled prediction
error). Conserved across one action. Fail-closed, with rollback, and nothing
learned from a refused action. **This is the correct shape for a resource
law** — and it is exactly what `exoJ` does not have.

### …and it has never fired

The boundary *is* executed (1585 → ok, 1586 → refuse, receipted in
`experiments/outputs/e_d1_summary.json` with `boundary.pass: true`). But the
recorded runs show **0 refusals and 0 debt**, and the repo's own verdict is
`NEVER-BOUND (law untested in vivo at these parameters)`.

> **A conservation law that never binds is untested.** This is the single most
> transferable finding in the audit, and it is a direct warning about writing
> γ+η≤C into this project: the budget must be shown to bind in the *measured*
> regime, or it is decoration.

### Growth gated on surprise decline, and the negative control

```js
gain = max(0, prevSurprise - surprise)
pos += clamp(floor(gain / 50), 0, 10)
if (!grown && pos >= 1000) { engine.loadSheet(buildSheet(2)); /* cells.clear() */ }
```

Growth is a **competence counter, not a reward counter**. The negative control
is in their own data: reward-driven growth (arm D) ran 5,000 evals and reached
position 10. So the transferable rule is:

> **Something becomes compressible when the system has stopped being surprised
> by it — not when it has been rewarded.**

Growth does a real graph rebuild (`cells.clear()`) followed by a state
restore, which is the snapshot→rebuild→restore pattern worth copying.

### Two false claims, and two dead files

* **β₁ is fabricated** — `skills.graph` hardcodes `V = 24/26, E = 22/26`; the
  README's "β₁ moves −1 → 1" is two literals against a real 32/33-cell graph.
  This sits directly under `quilt`'s own evolution spec, which reasons about
  Betti numbers evolving.
* **`kernels.mjs:1-9` claims "the engine program cells call THESE kernels (no
  drift)" — false.** `sheet.mjs` imports only `shared/kit.mjs`; the
  conservation kernels are not the ones the cells use.
* `driver.mjs` and the root `smoke.mjs` import `{growthCells, TASKS,
  SEED_DECLARATION}` from `sheet.mjs`, which exports only `buildSheet` — an
  ESM link-time **SyntaxError**. Neither is referenced by any script or CI
  target, so the "second implementation" has never run.
* `npm test` covers spec-seal hygiene and Surface-C durability only — **no CI
  coverage of the agent, the law, or growth.**
* Mocks carry explicit and honest flags, which deserves credit:
  `teacher.jev_gate` is `0.5+0.5·tanh((success−0.75)·8)` marked `mock:true`;
  `qrc.features` is a hashed PRNG marked `live:false`. Real JEV exists in
  `jev_live.mjs`, but the README's claims do not rest on it.

---

## 3. `exoJ` — real infrastructure, and a conservation law that cannot fail

**Verified real:** 34 `.mjs`, 491 lines of tests, CI green across 44 runs,
85 committed output artifacts. It genuinely ran.

* **The offline "JEV" is a PRNG** — `mulberry32`, at
  `jev_backends.mjs:14-42`. Only `EXOJ_LIVE=1` reaches the real API. Every
  offline headline number is therefore measuring plumbing.
* **The conservation law is vacuous.** η is definitionally `1-γ` in both
  backends, so `γ+η=1` is an arithmetic identity that no input can violate.
  The architecture's central safety property enforces nothing.
* **No calibrated probability anywhere.** `prob_mass` is binary 1.0/0.0, and
  the README's `prob_open=0.9412` is the fraction of *unobserved* cells.
* **"Non-collapsing"** is real in exactly one place — the `'ledger'` policy
  makes `accumulate()` permutation-invariant to ~1e-15 (`core.mjs:94-99`).
  That is the number in this repo worth trusting.
* **Growth is not implemented.** `decompose` is a hardcoded `SUBS` table;
  `atlas.mjs` sweeps pre-registered rules `R1_OPEN…R6_THIN` over a
  *hand-written* corpus of 29 sibling repos — re-analysis, not self-extension.
* The entire category theory in the README (functors, Kan extensions,
  discrete fibrations) has zero implementation.

### Two mechanisms worth taking

* **`attendWeakest`** (`selflocal.mjs:44-58`) pools cells with `|γ-0.5| < τ`
  and picks the lowest. A deterministic rule that *reads* the model to find
  where the system is least sure. This is the soft-spot detector (§11) in
  miniature — and it composes with a divergence-based detector, which is
  stricter because it needs two independent components to disagree rather than
  one model's distance from its own midpoint.
* **`noulToEmit`** (`live.mjs:70-73`) maps maximum uncertainty to maximum
  `Δ`: `Δ = 0.4 + 0.2(1-|2n-1|)`, peaking at n=0.5. **Be most curious where
  you are least sure** — a defensible policy for a watcher, and the opposite
  of the under-driving the Phase 2 rewrite had to correct.

---

## 4. `quilt` — 249 markdown files describing a system with none of it in code

The evolution spec (`docs/specs/spec-0010-evolution.md`, **Draft**) is
genuinely good theory, and it is the source of two ideas worth carrying:

* **Growth is redistribution under a conserved budget.** Evolution can move γ
  and η between cells but cannot create or destroy them.
* **"Evolution is not observable without perturbation — the act of looking at
  evolution changes evolution."** The measurement problem, stated as an axiom.
  It is why every instrument in this repo must declare that it perturbed what
  it measured.
* **The watch is the slowest integrator.** Attention is the rate-limiting step
  of evolution, so the watch sets `dt`, so the watch sets the rate.

**But:** `packages/core/src/engine.ts` (851 lines) and
`packages/sdk/src/index.ts` (1,738 lines) contain **zero** occurrences of
`gamma`, `eta`, `conserv`, `integrat`, `evolution` or `Betti`. The 9th
primitive, the evolution kernel, the conservation contract and the integrator
are entirely documentary.

---

## 5. `pincher` — a well-tested fuzzy lookup table in front of a SQL executor

**Verified real:** 296 files, 99 `.rs` (75 real source + 24 research
prototypes), a real Cargo workspace (`pincher-core`, `pincher-cli`,
`hybrid-bridge`) and a Python sidecar. CI is unusually honest with itself: its
workflow file carries a comment admitting a sandbox test *"has been red since
2026-06-06."*

### "Vector Database as runtime" is a slogan

Real KNN exists (`db/schema.rs:112` `vec0(embedding float[384])`;
`search_nearest` uses `WHERE v.embedding MATCH ?1 AND k = ?2`). But the index
only **selects a row**, and behaviour is then 100% determined by that row's
`action_sql` text. Nothing about the nearest-neighbour geometry affects
execution beyond argmax. That is storage with code around it.

**And the default build has no semantics.** `pincher-core/Cargo.toml:6` is
`default = []`, so the ONNX embedder is off and the default is a deterministic
hash (`embed/onnx.rs:266`, "ONNX feature not enabled, using deterministic hash
embeddings"). The repo's own benchmark states the hash has **no semantic
similarity** — *"turn on lights" ≠ "switch on lights"* — so the README's
headline claim that two differently-worded requests map to the same reflex is
**false in the default build**.

### The reflex path is real

`pincher do "<input>"` → `reflex/engine.rs:222` → `matcher.rs`: exact string
equality **first**, then sqlite-vec KNN k=5, cosine re-rank, thresholds
**0.80 exact / 0.55 similar**. Below 0.55 it returns
`"[NOVEL INTENT] No matching reflex found"` — no LLM, no write.

Between 0.55 and 0.80 it executes anyway and prepends
`"This response may need LLM refinement."` — a string concatenation. The
refinement is **not implemented**. So the middle tier does nothing different,
which makes the dead zone decorative.

### No learning loop exists

Only `ReflexEngine::teach()` via interactive `pincher teach` or RPC. **No
counter anywhere.** `pincher-infer/distiller.py`'s docstrings claim *"after the
third time you ask the agent to do something, it compiles a reflex"* — there
is no counter, threshold, or promotion code, and that file is **dead**: nothing
imports `ReflexDistiller`; `server.py` uses `distill.py` instead. The repo's
own `docs/REFLEX_PROMOTION.md` is titled **"Gap κ Design"** — the promotion
pipeline is a markdown procedure.

Confidence *is* updated (`confidence.rs:19-25`, +5% gap / −10%, clamped
0.05–0.95) but triggers nothing, and `ReflexPath::Direct/Confirm/LlmRoute` is
defined and never wired into `do_command`.

### Conservation, again un-enforced

`oxide-pipeline`'s `verify_conservation` is `|sum(in) − sum(out)| <=
in.len()` — tolerance equal to the element count. It is computed, stored in
`PipelineResult.conserved`, and **never enforced**: no branch, no error, only a
test asserting it is true. `layers_used: 5` is a hardcoded literal and
`execution_us = len*4` is fabricated.

### The one thing worth copying from any repo in this family

**`benchmarks/embed-benchmark-rtx4050.md` is genuinely honest** — real numbers,
a real negative result, and it *publishes* that the default embedding cannot do
semantics. It is the model for how retrieval quality should be reported here:
> you cannot learn habits if you cannot measure whether retrieval works.

**Falsifier for the whole repo:** build default features, teach "turn on the
lights", query "switch on the lights". It should fail.

---

## 6. `lever-runner` — the decide-not-steer loop, and two gates that fail OPEN

This is the most consequential entry in the audit, because it is the system
whose *description* this project has been built against, and it contains
exactly the failure modes §14 and §17 predicted.

**Verified real:** 4,612 LOC across 20 modules, 42 `.py` vs 27 `.md`. Three
front-ends (CLI, Telegram bot, HTTP) → `orchestrator.do()` → LanceDB →
`executor.run_command()`.

### The loop is genuinely the shape §17 wants

```
orchestrator.py:59  fast-loop
          :73       ONE LLM call — compress the utterance to 3-8 words, nothing else
          :91       embed + vector search
          :107      trust score
          :113      MATCH_SIMILARITY_FLOOR = 0.55
          :140      {{param}} substitution
          :185      run_command
          :186      update_trust
```

Deterministic to completion after a single inference. **The LLM never emits a
command and never sees the shell** (`intent_extractor.py:87-104` is a ~58-token
prompt whose whole output is a phrase). Token counts are real and modest
(~58 in, ~8 out, ~6 embed) — though counted by `_approx_tokens = len//4`, and
`BENCHMARKS.md` **contradicts its own `results-rtx4050.json`**: it claims
orchestrator p50 **7.6 ms** where the raw file says **4168 ms**.

### The approval set is mutable, unsigned, and open by default

Not a static committed file. It is a **LanceDB table, unsigned, unhashed,
mutable at runtime**, seeded with 57 commands at trust 50.

Three things add to it: `/teach` (Telegram, `bot.py:256`), the CLI, and HTTP
`/teach`. A human types the command — so there *is* a human decision — but the
authorisation is **opt-in and defaults to open**:

```python
bot.py:99        if not ALLOWED_USER_ID: return True   # anyone can teach
http_api.py:107  if not API_TOKEN:        return True   # anyone can teach
```

### Gate 1: the trust floor fails open

```python
orchestrator.py:107
eligible = [m for m in matches if m.trust_score >= min_trust] or matches[:1]
```

If **nothing** clears the trust floor, it runs the best match anyway. The gate
is decorative. And the trust floor itself is untested — `tests/smoke.py:610`
only tests `list_all(min_trust=)`, a listing filter, not execution gating.

### Gate 2: a latent path where an LLM authors approved commands

`auto_promote.py:63-124` asks a remote LLM to **rewrite a failing shell
command**, and `rewrite_losers()` would `teach()` that rewrite at trust 40 and
delete the original.

It is currently dead **by bug**: `store.TRUST_REWRITTEN` (`auto_promote.py:145`)
is a module constant rather than an attribute, so `AttributeError` is swallowed
at line 146.

> **Fix that one reference and a remote LLM begins authoring the approved
> command set.** This is §14's induced-soft-spot hazard and §17's collapsed
> surface, realised in a real codebase and one line from being live.

### The guard is a cache, not a validator, and it is unreachable

`fastloop-guard` is a **semantic response cache** — BLAKE2b exact → MinHash
fuzzy → miss. Grepping its `src/*.rs` for validation, rate-limit or terminate
returns nothing; its own README calls it a cache. *"Sub-millisecond validation,
rate limiting, and sandbox termination"* is a **docs claim only**.

It is also **unreachable**: Rust binds `/tmp/fastloop.sock` (`main.rs:13`) and
Python dials `/tmp/fastloop_guard.sock` (`fastloop_bridge.py:34`). And
`fastloop_bridge.py:55-61` drops to the Python path on **any** exception —
**fail-open, silently, with no log**. `SANDBOX_TERMINATED` appears only in a
comment; it is implemented nowhere.

Actual enforcement is that Python fallback: 10 req/1s per chat, 500-char cap,
metacharacter block.

### No confidence gating, and the seed set contradicts the executor

Zero occurrences of confidence/logprob/entropy. The only gate is geometric —
a vector-distance similarity floor, which is a retrieval heuristic and not
model confidence.

The executor blocks `$ ; | & < >` (`executor.py:37`), so **5 of the 57 seeds
contain pipes and can never execute**. The pre-approved set contradicts the
thing enforcing it.

### And CI cannot fail

`.github/workflows/ci.yml:19` is `pytest || true`. The only real gate,
`smoke.yml`, has been **red since 2026-06-04**.

---

## 7. The verdict across all six repositories

| repo | real code | claimed intelligence | verdict |
|---|---|---|---|
| `exoJ` | yes, CI green 44× | conservation (vacuous), growth (not implemented) | infrastructure only |
| `quilt-dba` | yes, CI green, 8 runs | conservation **implemented but never fired**; growth real | best of the six |
| `quilt-pincher` | yes, CI green 6× | federation (absent), ESP32 (aspirational) | **one adoptable mechanism** |
| `pincher` | yes, 99 `.rs`, honest CI | "vector runtime" (a slogan), learning loop (absent) | a fuzzy lookup table |
| `lever-runner` | yes, 4.6k LOC | approval gate (**fails open**), LLM authoring (**latent**) | **the cautionary tale** |
| `quilt` | SDK + engine, no evolution code | 249 docs of unimplemented theory | a paper, not a system |

Five of six have the same shape: **real infrastructure, aspirational
intelligence, and at least one gate that cannot fail.**

---

## What this project takes from all of it

| from | mechanism | status here |
|---|---|---|
| `quilt-pincher` | HDC deterministic embedding | **adopt** — it is what runs continuously |
| `quilt-pincher` | hit / confirm / compile tiering | **adopt** — maps onto the presence gates |
| `quilt-dba` | fail-closed conservation with rollback and no learning | **adopt the shape**, must be shown to bind |
| `quilt-dba` | growth gated on surprise decline | **adopt** — the compressibility rule for §4.2 |
| `quilt-dba` | snapshot → rebuild → restore on growth | adopt at growth time |
| `exoJ` | attend-weakest uncertainty clustering | **adopt** as the cheap half of the soft-spot instrument |
| `exoJ` | uncertainty drives exploration | adopt as policy |
| `lever-runner` | decide-not-steer; immutable approval | see design §18 |
| everywhere | the scaffolding-outruns-execution pattern | **adopted as a rule**: no mechanism enters this project on a document's word |