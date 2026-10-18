# Operations — how to actually run things in this repo

Everything here was learned by hitting it. None of it is in the vendor docs.

## The interpreter

`typesafe_sdk` (0.7.5) is installed in the **glasses repo's venv**, not the
system python:

```bash
/home/proart/projects/glasses/.venv/bin/python -m pytest tests/ -q
```

System `python3` will `ModuleNotFoundError: typesafe_sdk` and the live tests
will skip or error. All numbers in PROGRESS.md were produced under the venv
interpreter.

## The API key

Two sources, same key:

- `/home/proart/projects/glasses/.env` — `TYPESAFE_API_KEY=...` (load with
  `set -a && . /home/proart/projects/glasses/.env && set +a`; that repo has
  no dotenv loader, everything reads os.environ).
- `/mnt/c/Users/casey/type.txt` — Windows side. The line is
  `TYPESAFE_AI_TOKEN="apikey_<...>"`. Extract **between the quotes** and
  **keep the `apikey_` prefix** — the bare tail 401s, and sending the whole
  line including quotes also 401s. Those two failure modes cost the fleet an
  evening once.

Env var names accepted: `TYPESAFE_API_KEY` or `TYPESAFEAI_KEY`, `.strip()`ed,
never de-quoted. Never print it, never write it to any file, never into a
ledger or log line.

## Live runs

```bash
# health probe (10 calls): garbage vs real batteries
/home/proart/projects/glasses/.venv/bin/python -c \
  "from src import question_health, judged; \
   print(question_health.format_report(question_health.probe_questions(), model=judged.DEFAULT_MODEL))"

# calibration corpus (45 calls, ~11 s) -> CALIBRATION.md + calibration-ledger.jsonl
python -m src.calibration

# end-to-end demo -> examples/*.mid + provenance
python -c "from src import score; score.render('examples/demo-utterance.wav', 'STOP THE DEPLOY RIGHT NOW', 'examples/demo-utterance')"
```

Measured costs (jev-1.14.0, 2026-10-18): four questions + four gates in ONE
call, 200–550 ms; full demo 553 ms end-to-end; whole calibration corpus 45
calls / 11 s. Call counts are a first-class output — report them.

## Model discipline

- Name the build explicitly: `jev-1.14.0`. `jev-latest` resolves to 1.13.0
  on this account — a floating alias makes every number unattributable.
- `judged.question_fingerprint()` versions the question wording. Any record
  tying numbers to a question (cache, CALIBRATION.md) must cite it; a
  reworded question invalidates old numbers BY CONSTRUCTION.
- `confidence` on any answer is distribution spread, never a probability.
  The CC comes from `probabilities[chosen]` (Choice) or the normalized
  `score` position (Score). Mixing these up shipped a plausible wrong system
  once already (PROGRESS.md, the `_scale_position` regression).

## Box quirks

- **Approval-handler outages recur** (three in one day, minutes to hours):
  shell `echo` passes while `python3`/`git`/file writes fail. Recovery so
  far: wait, retry; record what failed. Don't conclude your command was
  wrong because one attempt died.
- The GPU is shared with other agents' models — check VRAM before pulling
  or loading anything (Ollama is on :11434, Ollaya on :11435).
- A tmux session (`quiltprobe`) and possibly other agents share this tree —
  see COLLABORATION.md before assuming a surprise is corruption.

## Cloudflare (2026-10-18)

- Token: `/mnt/c/Users/casey/CF.txt` — `CF_TOKEN="..."` (extract between
  quotes). Verified active. Account:
  `049ff5e84ecf636b53b162cbb580aae6`.
- Workers AI REST: `POST /accounts/<id>/ai/run/@cf/meta/llama-3.1-8b-instruct`
  with `{"messages":[...]}` — battery probe 3/3 correct incl. garbage
  decline (better than jev-lattice's smaller-model overconfidence finding;
  treat as a second path, never sole authority, per the fleet law).
- Never write either key (TypeSafe or CF) into any repo or log line.

## Jev continual-experiment recipes (the trails)

- Per-item stability (semantic-entropy lite): resample `read_tracks` N=10
  UNCACHED on fixed texts → per-track mode-share. Script pattern in
  /tmp/jev_stability.py (recreate from git history of this section if
  lost); results → EXPERIMENTS-AUDIO.md.
- Cross-build: `read_tracks(text, model="jev-1.13.0")` etc. — always
  fingerprint + model-name the numbers.
- The corpus in examples/realaudio/corpus/ is the standing battery
  surface for new questions; whisper (audio venv) makes more of it.
