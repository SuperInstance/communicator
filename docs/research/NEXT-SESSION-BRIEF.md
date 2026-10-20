# NEXT SESSION BRIEF — hand this to the next agent

**Purpose:** a self-contained brief for whoever picks up `Communicator`
next. It was written by the agent that ended the 2026-10-20 night, so
that the next one starts from the findings rather than from zero.

**For `zcode` specifically:** this file is a complete task brief. Once
`zcode` is authorised (see §0), it can be run with it directly:

```bash
cd /home/proart/projects/communicator
zcode -p "$(cat docs/research/NEXT-SESSION-BRIEF.md)" --cwd /home/proart/projects/communicator
```

or, preferably, with the file attached so the agent can read it as a
document rather than as one enormous prompt:

```bash
zcode -p "Read docs/research/NEXT-SESSION-BRIEF.md and do the work it describes." \
   --attach /home/proart/projects/communicator/docs/research/NEXT-SESSION-BRIEF.md \
   --cwd /home/proart/projects/communicator
```

---

## 0. BEFORE YOU START — ZCODE'S OWN AUTH

On 2026-10-20 `zcode` could not run headlessly on this machine. The
failure is exact and it is **not** a network problem:

```
[project-access-token] { family: 'zai', stage: 'resolve', code: 'project_token_login_failed' }
RuntimeHeadersRefreshError: project_token_login_failed
  cause: CodingPlanApiKeyError
  providerId: account:zai-individual-coding-plan   modelId: GLM-5.3
  baseURL: https://api.z.ai/api/anthropic
```

`api.z.ai` and `open.bigmodel.cn` were both **reachable** from the box
(HTTP 301 and 200). The stored credentials in
`~/.zcode/v2/credentials.json` are present but the **coding-plan project
token** will not mint.

**The fix is one command, and it needs a browser:**

```bash
zcode login zai          # or: zcode login bigmodel
```

Headless on this box it prints a fallback URL and then fails with
`Browser open failed: spawn xdg-open EACCES`. **Run it somewhere with a
browser**, or forward the printed URL to a machine that has one. After
that, `zcode -p ...` works.

**Do not try to work around this by copying tokens around.** The
entitlement is the account's to grant.

---

## 1. WHO YOU ARE AND WHAT THE REPO IS

You are an AI agent working in `/home/proart/projects/communicator`, a
Python research repo called **Communicator**: Casey's local-first
voice/peripheral system. The camera is a small always-on peripheral; the
model is a glasses-style assistant; the system should notice that the
person is speaking and respond, on a device that is not a phone and not a
laptop.

**Read these first, in this order, and obey them:**

1. `docs/SCIENCE.md` — the standard.
2. `docs/research/HOW-WE-KEEP-FOOLING-OURSELVES.md` — the eight
   retractions and the standing practice.
3. `docs/research/NIGHT-REPORT-2026-10-20.md` — **the night's work in
   full: achievements AND struggles.** Start here; it is written for you.
4. `docs/research/NV12-STRIDE.md` — the root-cause defect.
5. `docs/research/DEBRIEF-REAUDIT.md` — the instrument's own corrections.

**The house style, which is not optional:**

* Every measurement names its **condition**, its **interval**, and its
  **null**.
* Every verdict **can come out the other way**, and is shown failing when
  it is built.
* A **control that cannot come out the other way is not a control.**
* A number derived from the assumption it is supposed to test **is not a
  check**. The check comes from outside: the file size, the script on
  disk, the audio's own duration, a human eye.
* **Falsified results stay visible.** A retracted claim is left in the
  document with a **dated correction block**, not silently overwritten.
* **A suite that cannot fail is not evidence.**
* The headline is a **margin over controls**, not a raw score.
* Dated correction blocks, house style: declarative title, ALL-CAPS
  section headers, exact numbers, a "what this does not establish"
  section, and the old text left visible.
* **Never `git add -A`.** Stage explicit paths. Other agents share this
  tree.
* **`sudo` is blocked. `rm -f` / `rm -rf` are blocked** — use
  `/home/proart/.minimax/bin/mavis-trash -- <paths>` or `mv` it aside.
* **Never push a key.** `.env` is gitignored; the Cloudflare token lives
  at `~/.config/cloudflare/env` (`CF_TOKEN`).

**Commands:**

```bash
cd /home/proart/projects/communicator
/home/proart/projects/glasses/.venv/bin/python -m pytest tests/ -q
/home/proart/projects/glasses/.venv/bin/python -u experiments/artifact_repro.py --source worktree
set -a && . ~/.config/cloudflare/env && set +a
/home/proart/projects/gpu-venv/bin/python   # torch 2.6.0+cu124, RTX 4050, mediapipe 0.10.21
```

---

## 2. THE STATE YOU ARE INHERITING

**Everything is pushed.** `master` is in sync with `origin/master`, the
suite is green (**1002 passed, 5 skipped**), and the artifact gate reports
**REPRODUCIBLE=37, NOT_REPRODUCIBLE=0** — every committed artefact
reproduces from the committed code.

**The camera pipeline now has a verified speaking signal at session
level**: mouth box rows 46–58, cols 62–86, **+0.733σ** against the cheek
control's **+0.477σ**, z **+10.9**, with the empty room at **0.015 %** of
that. It is real, and it is what the night's biggest fix bought.

**Three things about that signal are NOT established, and you will need
them to be:**

1. It is **not mouth-specific** — the cheek box scores *higher* (0.7443
   vs 0.7086), because the "mouth" box straddles the face's silhouette
   edge.
2. It is **not frame-level** — the audio and video in these captures are
   **independent takes**. No lag (±20 s) and no time base (10–60 Hz)
   recovers frame alignment. The label is **session level**, and that is
   the only label these captures can carry.
3. The captures are **19.83–19.87 Hz distinct frames, not 30** — 596 of
   900 frames in A are distinct.

**The LLM "director" has been shown not to be the thing that should be
trusted to plan the next step.** It accuses the repository's one true
claim in **15 of 15** successful runs. Its proposal keeps re-proposing
the plan item already marked as blocked on hardware, and it keeps **not**
proposing Condition D.

**And the instrument used to judge the director is currently too coarse
to answer the questions being asked of it.** The audit set has nine false
claims and one true one, so the headline is **quantised at 1/9 = 0.1111**,
the run-to-run spread is **0.111 and 0.222** — one and two quanta — and
the effect the lane claimed was **+0.1111**, exactly one quantum. **A
comparison of the director cannot currently resolve a one-claim
difference.** See `docs/research/DEBRIEF-REAUDIT.md` §4.7 and §4.8.

---

## 3. THE WORK: BUILD A ~40-CLAIM AUDIT SET WITH DERIVED LABELS

**This is the single highest-value piece of remaining work, and it is not
optional.** Until the instrument can resolve a one-claim difference,
every comparison of the director is uninterpretable.

### 3.1 What to build

A new, larger audit set — target **~40 claims, with roughly half true and
half false** — in the style of `experiments/debrief.py`'s `AUDIT_SET`,
where:

* **every label is DERIVED from the repository's own results files**, not
  asserted in code. The pattern already exists in
  `experiments/debrief_set_reaudit.py`: each claim has an evidence
  function that reads a deciding number out of a results JSON and
  compares it with the number the claim asserts.
* **the label is a function of the file, and the test proves it.** For
  every claim, a **mutation test** must perturb the deciding number in
  the results file and require the label to flip. A label that does not
  move when its evidence moves is asserted, not derived.
* **there is more than one true claim**, so specificity is no longer a
  single bit. Today it moves in steps of **1.000**, which is why the
  context-vs-blind story is one bit flipping.

### 3.2 The arithmetic that sets your target

* The exact McNemar test's p depends on the **smaller arm of the split**,
  so it is conservative. On **nine** claims it rejects at p < 0.05 only
  when at least **8 of the 9** move in one direction. A shift of **6
  claims one way** gives p = 0.031.
* So the minimum detectable effect on nine claims is **six**. To detect a
  one- or two-claim shift you need roughly **forty** claims.
* With **n** false claims the metric's quantum is **1/n**, and the
  run-to-run spread stays at one to two quanta. **Make n large enough
  that a quantum is smaller than the effect you care about.**

### 3.3 Where the claims come from

The repository is full of them, and they are already machine-readable:

* **FALSE claims** — the withdrawn numbers. The `supersedes` blocks in
  the results JSONs carry the old values: `row_band_removal_results.json`
  has `supersedes.previous_results.*` (48.19 %, 50.00 %, 0.0130 %),
  `combined_filter_results.json` has `supersedes.previous_variants.*`
  (0.0299 → 0.0014, 21×, SNR 1.83), and so on. A claim asserting a
  superseded number is false, and the current file says so.
* **TRUE claims** — the current verified numbers: `room_map_results.json`'s
  −73.5 dBFS / 119 transients / zero RT60; `row_band_removal_results.json`'s
  0.0000 % empty and 0.6250 % occupied; the stride margins (79.9×, 87.1×,
  117.1×, 7,446×); the mouth signal's +0.733σ; the dither fraction's 0.0 %;
  the cloud multimodal margin's +0.667.

**A claim must be checkable.** If you cannot point at a deciding number in
a results file, it does not belong in the set. That rule is what keeps
the set from becoming a pile of assertions — which is what the old
ten-claim set was.

### 3.4 The shape of the deliverable

* `experiments/debrief_audit_set_v2.py` — the set, the derivation, the
  scoring, and a `--out` that writes a results JSON with
  `pre_registered_rules`, per-claim labels, the deciding numbers, the
  files they came from, and a `provenance` block.
* `tests/test_vlm…` no — `tests/test_debrief_audit_set_v2.py` — including
  **one mutation test per claim class** that proves the label moves when
  the evidence moves, and a test that the set's **true and false sides are
  both large enough** to be worth having.
* `docs/research/DEBRIEF-AUDIT-SET-V2.md` — the write-up in the house
  style, with the exact numbers, a "what this does not establish"
  section, and the provenance.

**Then re-measure the director on it**, with
`experiments/debrief_paired_analysis.py`'s machinery, and report whether
the instrument's resolution finally beats its noise.

### 3.5 Do not

* Do not change `debrief.AUDIT_SET`. The ten-claim set is the historical
  record and the lane's committed numbers depend on it. **Add** a v2.
* Do not weaken a label to make the director look better. That is the
  one thing this repo must never do.
* Do not re-run the ten-claim set and overwrite its numbers.

---

## 4. THE OTHER OPEN ITEMS, IN ORDER

1. **Store the raw per-frame scores in Lane 8's artefact**
   (`experiments/vlm_image_retention_results.json`) and add the paired
   test of the **0.178 AUC difference** between the image and text arms.
   The lane stores per-condition summaries only, so the difference has no
   test of its own. This needs a re-run, which changes the lane's
   committed numbers — so it needs a dated correction block in
   `docs/research/VLM-IMAGE-RETENTION.md`.
2. **`src/energy.py:290`'s leading-silence defect** is still unpatched.
   Blast radius in `docs/research/SILENCE-LEADING-ARTEFACT.md`.
3. **`src/vad.py:41` and `tests/test_vad.py`'s 39.13 ms citations** are
   deliberately untouched, awaiting Casey's standing decision.
4. **Known defects recorded, not fixed** (in `known_defects_not_fixed`):
   `ascii_motion_threshold.py`'s frame-pair loop still starts at `f = 0`,
   counting one spurious last-vs-first pair; and
   `ascii_motion_threshold.py` and `ascii_motion_nulls.py` still disagree
   on the dither gradient between `small.nv12` (52.6×) and `A.nv12`
   (flat 0.99×) — **unchcharacterised**, and worth a lane.

---

## 5. WHAT NEEDS CASEY, AND CANNOT BE DONE BY AN AGENT

1. **RECORD THE CORPUS — Condition D is the highest-value take.** The
   capture tool is fixed and synced to the Windows side
   (`/mnt/c/proart_capture/corpus_capture_win.py`).

   ```
   C:\Python314\python.exe C:\proart_capture\corpus_capture_win.py --plan --out C:\proart_capture\corpus
   ```

   ~16 minutes: A1–A3 / B1–B3 read the poem for 50 s, C1–C3 / D1–D3
   silent for 30 s, F1–F3 for 60 s, then 5 minutes free. **Glasses off for
   B and D.**

   **Condition D — turned AND silent — is now the highest-value missing
   take for two independent reasons:** it makes every pose fold
   identifiable, and it separates "the head moved" from "the mouth
   opened". Everything else on the list is secondary to it.
2. **RECORD THE TWO-MICROPHONE DIALOGUE** (adoption-plan item 10). Now
   safe: the intake fixes have landed.
3. **Re-authorise zcode** — `zcode login zai` in a browser. See §0.

---

## 6. THE CORRECTED READER — use this, do not re-derive it

```python
from src import nv12
luma, layout = nv12.read_luma(path, width=160, height=96)   # (n, 96, 160) uint8
layout.as_dict()
# {"layout": "y_uv_halfres", "stride_bytes": 19200, "n_frames": 900,
#  "mean_consecutive_frame_diff": 0.544891, "margin_over_runner_up": 79.932,
#  "chroma": "interleaved U,V, 80x24 samples"}
```

Strides: `standard_nv12` 23,040 · `y_uv_halfres` 19,200 (the captures) ·
`luma_only` 15,360 (no script in the repo writes it). The alias
`y_u_v_planar` → `y_uv_halfres` is deprecated.

**A layout that is ambiguous is an error, not a guess.** The old captures
stay on disk so every withdrawn number can be re-run.

---

## 7. HOW TO COMMIT AND PUSH

* Commit **explicit paths**. Never `git add -A` — another agent may be
  working in the tree.
* The commit message is the documentation. Say **what was measured, what
  the number was, what was falsified, and what was wrong on the way**.
  The repo's commits are read as the research log; a commit message that
  only says "fixed stuff" is a wasted commit.
* Push often. The night was 24 commits.
* After pushing, `git rev-list --left-right --count origin/master...HEAD`
  must print `0	0`.
* Run the suite before you push: **1002 passed, 5 skipped** is the
  baseline. Anything less is a regression.
* Run the gate before you push:
  `python -u experiments/artifact_repro.py --source worktree`. **Use `-u`**,
  and run it detached (`setsid … > /tmp/gate.log 2>&1 &`) — a long
  verification run whose output is buffered and then killed leaves you
  with nothing, which happened once on 2026-10-20.
* **If the gate reports NOT_REPRODUCIBLE > 0, do not paper over it.**
  Re-run the script directly. On 2026-10-20 two scripts crashed *inside
  the gate* (`IndexError`, `EOFError`) and reproduced perfectly when run
  directly, byte-identically — the failures came from the `/mnt/c`
  Windows mount after the machine had been suspended and resumed, not
  from the code. **Prove which it was before deciding.** This was
  confirmed, not assumed: a later full run of the same worktree gave
  **REPRODUCIBLE=37, NOT_REPRODUCIBLE=0** with both scripts reproducing
  and no code changed.

---

## 8. THE ONE-SENTENCE VERSION

> **The camera pipeline now has a verified speaking signal at session
> level, and the LLM director has been shown not to be the thing that
> should be trusted to plan the next step — but the instrument used to
> judge the director is too coarse to tell a one-claim difference from
> noise, and the record needs forty claims before it can.**
