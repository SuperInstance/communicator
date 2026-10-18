# Fleet knowledge — what the sibling repos know that applies here

communicator is one of ~12 repos under SuperInstance. Several sibling
instruments produced findings this project already depends on. If you are
about to re-measure something, check this list first: the fleet has been
burned by re-deriving a known negative result more than once.

## decision-ledger (private)

The question-quality instrument shop.

- `pass_detector()` + `question_hash()`: questions whose answers cluster at
  ~0.5 sharpness are INVALID (the model is passing), distinct from unsure.
  Communicator's question_health.py is the mirror image (catches questions
  that invent structure instead of questions that pass).
- QUESTION-REWRITE.md: a rewrite moved a question from 0.30-pass to 0.97-sharp.
  The method — hash before/after, keep both — is how any question change here
  must be evidenced.
- Platt scaling HURTS an already-calibrated model (winnow ECE 0.033 → 0.053).
  Measure first; repair only on evidence. Scaling is repair, not upgrade.
- Teacher labels: Jev-as-teacher agreed with ground truth 96.4%, so
  Jev-teaching-Jev calibration loops are valid.

## jev-lattice (private, another agent's project — read-only, never push)

The deterministic decision graph. Three findings this repo borrowed:

- **Jev is not bit-deterministic**: 7% of gates flip GO↔OPEN at threshold 0.9
  on a pure re-ask. Their proof cache fixed it (0 flips, 66–73% fewer
  calls); communicator's verdict_cache.py is the same fix for a control
  surface.
- **Field-question technique**: one Choice/Score per field, derive yes/no by
  probability mass — ECE 0.033 → 0.0004 with half the questions.
- **Compiled code can say no, not yes**: refute-only deterministic
  predicates gave 0 wrong GOs at half Jev cost; trusted to say yes they
  produced 5–24 wrong GOs after a format shift. Same law as this repo's
  "blind veto in front of any model gate" and minigate's blindness veto.

## gate-replay (private)

The idle-gate bench. Live Jev results: 95% idle-skip at 10.5% false-idle,
$0.05/month at 96 beats/day; weasel-claim catch p=1.00 (beats winnow 0.99,
laya 0.31); injection 7/8. The false-idle class was `llm_error` retry rows —
deterministic blind veto stays in front of ANY model gate. `jev_bench.py`
there is the reusable harness shape.

## zero-poc (PUBLIC — the live agent)

The production consumer of gate thinking: dual-provider `decide()`
(Ollaya local ↔ TypeSafe cloud), live-validated `gated_idle`, duty-cycle
forced probe. Its gate is dormant in Actions until the repo secret
`TYPESAFE_API_KEY` is set (user action, still pending).

## glasses (g1-ble-backend, private)

Owns the venv this repo tests under, and `jev.py` — the adapter wisdom:
`KEY_ENVS` accepts only the two names above; values stripped not de-quoted;
**use `choice`, never N independent nouls** for exclusive option sets
(measured: independent nouls cleared 0.5 on two options at once); **never
ask Jev to count characters** (competence map scored counting 2/5;
misspellings came back 0.82–0.89 TRUE).

## tool-judge (private)

The permission layer this project's Phase 4 adopts: grant broker (agent may
REQUEST, human GRANTs), scope/rails fail-closed, postures + kill switch,
anomaly detection. The broker invariant — no `{grant, decide, approve,
set_grant, revoke}` on the agent surface — is adopted whole by the design
doc. Known defect found from here (fixed nowhere yet, owner's call):
`mcp_server.py:168` read `args["developer"]` for the `user` key, collapsing
user and developer scopes into one identity over MCP.

## Crash-recovery log (2026-10-18, by ZCode)

For audit: the 2026-10-17 evening session crashed mid-"going public" pass.
Recovered and pushed: trust-ledger `d16be30` (public README+LICENSE),
gate-replay `613c73a` (README rewrite + key-path scrub), vision `13e1154`
(path scrub), tool-judge `8fe9f38`+`3f627b8` (detector false-positive fix +
audit/telegram/doctor, 243 tests green; egg-info artifacts removed),
communicator given its GitHub repo (private). hermit-bench is deliberately
not a repo.
