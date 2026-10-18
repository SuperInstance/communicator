# ZCode handoff — 2026-10-18

Written at session end. If you are another agent starting here: read
PROGRESS.md and README.md first — they are the primary record. This file is
what I knew that is not convenient to put anywhere else.

## Where things stand

Phases 0–3 of the design doc are complete and live on `master`. The units,
in the order they landed this session:

| unit | one-line state |
|---|---|
| verdict cache | `src/verdict_cache.py`, fingerprint-keyed incl. gates; jitter killed |
| PITCH energy floor | 1e-3 RMS pre-YIN; risk 2 closed; floor is a knob (`energy_floor=`) |
| question health | garbage-vs-real batteries; verdicts healthy / ill-posed / degenerate / safe-parked |
| presence gates | fleet-designed after re-wording measured WORSE; multiplied into CC |
| calibration | `CALIBRATION.md`, labels by construction; GESTURE 1.000/0.001, ATTENTION 0.917/0.097, INTENTION Spearman 0.978, TIMBRE wording-only |
| Stage 3 export | `src/score.py`, tick=1 ms, provenance sidecar; `examples/demo-utterance.mid` = 553 ms, 1 Jev call |
| README | the front door, with the five house rules |

Suite at handoff: **215 passed, 5 skipped** (skips = live-key tests) under
`/home/proart/projects/glasses/.venv/bin/python`.

## Open threads, in the order I would take them

1. **Reflex before gate** (design doc §12, the actual "Phase 3"): threshold
   tuned on real data, not asserted. The gate machinery now exists; the
   threshold does not.
2. **Real-speech fixtures.** Everything acoustic is synthetic. Octave errors
   are the most likely real-world failure and the suite never exercises
   them. A handful of recorded syllables would be the first honest input.
3. **Phase 4 authority** — grant broker + tmux escalation. tool-judge's
   broker invariant is adopted whole by the design doc; do not redesign it.
4. **Latency under load** — printed everywhere, asserted nowhere.
5. **Nine vs sixteen tracks** — no agreed semantics for the extra seven.
   Don't invent them to fill a table.

## Concurrency warning

At least one other agent (Claude, Opus 4.8, GitHub-noreply identity) works
this same working tree. Commits can appear between your commands; files you
just wrote can be fixed or rewritten underneath you. That is normal here and
so far it has been productive. See COLLABORATION.md for the protocol.

## Fleet context (outside this repo)

User actions still pending, not this repo's to do: zero-poc repo secret
`TYPESAFE_API_KEY` (activates its cloud gate), pacemaker `wrangler deploy`.
The TypeSafe key never enters any repo — see OPERATIONS.md for where it
lives and how to load it.
