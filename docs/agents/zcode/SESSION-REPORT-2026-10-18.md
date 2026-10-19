# ZCode session report — 2026-10-18 (stand-down)

One continuous session, ~20 hours, from crash recovery to outside-
playtested consolidation. HEAD at stand-down: see git log; every unit
committed and pushed as it landed. The definitive detailed records:
PROGRESS.md, docs/research/EXPERIMENTS-AUDIO.md (Day-0→3), GOLD.md,
SCIENCE.md, PAPERS.md, TIMING.md, FLEET-MUSIC.md, BORROWING.md,
docs/agents/zcode/{HANDOFF,OPERATIONS,COLLABORATION,FLEET-KNOWLEDGE}.md.

## Shipped

**Recovered from the crashed session**: 4 repos' interrupted work
committed/pushed; communicator given its GitHub repo.

**Phases 2.5–3 complete** (co-written with the Claude agent in the same
tree): verdict cache, PITCH energy floor, question-health instrument →
presence gates, by-construction calibration (GESTURE 1.000/0.001,
ATTENTION 0.917/0.097, INTENTION ρ=0.978), nine-track .mid export with
provenance (553 ms, one call), README + SCIENCE + the verified 30+-paper
corpus with its corrections ledger.

**Real-audio program, Days 0–3**: first real speech through the pipeline
(octave errors quantified 3.0% → smoother −34% + the cross-gap artifact
found); E4 per-phrase tempo PASSED (SD 19% of mean); E3 bimodality BIC
10,097 with the register confound kept honest; 8-cell public-domain
corpus + cross-genre table (phrase length 35× spread); ASR leg (11×
realtime CPU); judged tracks on real dialogue; per-item stability
measured (Choice deterministic, Score jitters) → trust layer designed
AND implemented (src/stability.py); **E7 refuted three ways with
symmetric controls — the dissociation (intent ⊥ within-line delivery,
n=626) is the definitive finding**; provider matrix complete (Jev-1.14
dominates; tev1 = free deterministic fallback; winnow = local gesture
model; laya "inversion" resolved as polarity bias; Workers AI mid-tier).

**The theory layer**: TIMING.md (capture point, T-minus, priming) with
law 2 honestly corrected after E7; the fleet music map (tminus-
dispatcher = the law deployed; midi-temporal = convergent thesis);
GOLD.md (H1–H10 falsifiable hypotheses + 10 use-cases; E8 = the pick).

**Consolidation**: four audit lanes + fresh-clone playtests → fresh-
clone blockers fixed, a real trust-inflation bug fixed with regression
test, all stale claims corrected with sources, and
tools/guard_test_deps.py making the bare-interpreter contract
self-maintaining. Final: fresh clone, bare python3, zero failures.

## The two things worth remembering

1. **Symmetric controls or it didn't happen.** E7's "signal" (p=0.022)
   died the moment controls got the same +1.5 s shift the hypothesis
   demanded. Three suggestive results this session survived that test;
   one didn't. Run the falsifier first (GOLD.md's standing rule).
2. **This box's orchestration shape**: subagents are the
   research/ideation/annotation layer (superb at reading, citing,
   proposing — 19 of them ran today); the main session is the only
   execution layer; approval outages are waited out, never fought, and
   every blocked agent still delivered a methodology that executed
   later in minutes.

## Standing down

Open threads, in value order: GOLD.md H1 (dissociation sub-line test,
0 API), E8 prime-then-payload (2-week pick, protocol ready), H3 (gate
evidence-accumulation = the judged modality's T-minus), the colleague's
in-flight lanes (speculate/test_speculate order-flake flagged to them).
User actions outstanding (fleet-wide, not this repo): zero-poc repo
secret TYPESAFE_API_KEY, pacemaker deploy. The keys live where
OPERATIONS.md says; never in repos.
