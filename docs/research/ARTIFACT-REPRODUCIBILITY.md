# ARTIFACT REPRODUCIBILITY

*Which committed numbers in `experiments/` can still be produced by the script
that sits next to them?*

**Tool:** `experiments/artifact_repro.py` · **Tests:** `tests/test_artifact_repro.py`
**Audited commit:** `9c77f77` (`9c77f777b41bbf00467beb957b114b11de98542f`, 2026-10-20)

---

## 1. The question this answers

`experiments/` holds 60 committed result files at `9c77f77`: the
`*_results.json` family and the `teacher_conventional_*.json` family. Each was
written by a script that sits next to it — but *the script that sits next to it*
is not *the script that wrote it*. Scripts get edited. Corpora get re-captured.
Inputs get deleted. Once any of that has happened, a note that quotes a number
cannot be checked, and a number that cannot be checked is a rumour.

The tool answers one question per artefact:

> **Can this committed number still be produced?**

It is a **commit gate**: it exits 1 when a committed number has stopped being
reachable, so that a number cannot quietly rot behind a note that still quotes
it.

---

## 2. What "reproducible" means here

Precisely:

> A committed artefact is **REPRODUCIBLE** if, re-running the companion script
> with the **same inputs** and the **same arguments**, every **headline number**
> in the committed file is reproduced.

Three qualifications, all of them load-bearing.

### 2.1 A committed script, not a working-tree script

The script that is actually run is the one committed at the audited commit. A
dirty working tree would otherwise let an in-progress edit change a verdict,
and the same answer would not be available to the next person who checks out
the commit. `--source worktree` opts into the working tree instead; the two
are reported separately because they can legitimately differ.

### 2.2 Same inputs, same arguments

Where the artefact records its inputs, the tool uses them. The newer artefacts
carry a `provenance` block with the exact `cli_args` the script was called
with — including the corpus path — and the tool re-runs with those arguments,
minus `--out`, which is redirected into the scratch tree. That is how "the
same corpus" is honoured: the corpus that produced the number is *named in the
file*, not guessed.

Where an artefact records nothing, the script's own declared defaults are used,
and the manifest records which corpus that was. That is a weaker guarantee and
the manifest says so.

### 2.3 Headline numbers, not byte equality

This repo is mid-way through adding provenance keys to its result files. A file
that gained three keys is **not** a falsification; a file whose `acc_test`
changed **is**. So the comparison runs over the numeric and boolean leaves the
file presents as *measurements*, and it reports **the fields that differ, by
name**, with the old and the new value. Key-count differences are reported
separately and never decide a verdict.

**A headline number** is a scalar whose key names a measurement — an accuracy, a
margin, a rate, a score. Depth does not matter: the number this repo is in
trouble about, a leave-one-session-out accuracy of 0.675, lives two levels down
inside a list of records. Rows in a list are addressed by **the identity they
carry** (`held_out`, `name`, `cell`, `condition`, …), not by their position, so
a row that moved does not read as a changed number.

**Numbers that are a property of the machine or the clock are excluded** —
`elapsed*`, `cpu_*`, `wall_*`, `latency_s`, `per_question_ms`, token counts.
They change on every run for reasons that have nothing to do with the claim;
including them would make every artefact look irreproducible.

---

## 3. Method

1. **Discover.** `experiments/*_results.json` and
   `experiments/teacher_conventional_*.json`, as tracked at the audited commit.
   (60 at `9c77f77`.)
2. **Map to a script by name convention.** `foo_results.json` → `foo.py`;
   `teacher_conventional_{task}_{features}.json` → `teacher_conventional.py`.
   For the teacher family, the task **and** the feature set must both match:
   `speaking_both` and `facing_both` share one script and are different
   experiments. (An earlier version of this tool matched only the feature set
   and silently ran the `speaking` task against the `facing` artefacts; the
   bug is fixed and the `facing_*` artefacts now reproduce.)
3. **Record the provenance of the pair**: the script's path, whether it exists,
   its mtime and SHA-256, the SHA-256 of the script that was *actually
   audited*, the artefact's mtime, SHA-256 and **git blob hash**
   (`git rev-parse <ref>:experiments/<name>`), and the SHA-256 of the baseline
   bytes that were compared against.
4. **Re-run** the script in a scratch tree.
5. **Extract** the headline numbers from the committed artefact and from the
   re-run's output, and **compare the shared paths**.
6. **Classify** the artefact and record the reason.
7. **Report** a human table to stdout and, with `--json`, a machine manifest.

### 3.1 The scratch tree, and why the repo is never mutated

Every re-run runs in a directory under `/tmp/artifact_repro/` that mirrors the
**committed tree**: a real `experiments/`, `src/` and `tests/` holding the
audited code *and* the committed result files, and symlinks for the data
directories (`examples/`, `models/`, `.scratch/`, `tools/`, `workers/`,
`docs/`). A script that writes next to itself therefore writes into the
scratch tree and not into the repo; and a script that reads a sibling
committed artefact — `silence_leading_artefact.py` reads
`pause_threshold_results.json` — finds it.

`tests/` is mirrored as **real files**, not symlinked, because
`pause_threshold.py` checks its copied method against
`tests/test_pause_threshold.py`; a symlink would hand the script a path into
the repo.

The tool never deletes anything — it only ever adds files under the scratch
root, with one residual risk recorded in §12, item 8. The fixture test
asserts the repository's `experiments/` directory is byte-for-byte and
mtime-for-mtime unchanged after a full run, with a script that deliberately
writes next to itself.

### 3.2 The commit is pinned

The working tree of this repo moves while people work in it: another agent can
commit a regenerated artefact between two runs of this tool, and an audit that
silently changed its own subject is worse than no audit. `--ref <commit>`
audits that commit. The default, `HEAD`, is resolved and printed at the top of
every run and recorded in the manifest.

### 3.3 Exit code

**1 if and only if** some artefact is `NOT_REPRODUCIBLE`; 0 otherwise.
`NO_SCRIPT`, `NOT_RE_RUN` and `LEGACY_NO_SCRIPT` never fail the gate — they
are statements about the repository, not about a number.

---

## 4. The five verdicts

| verdict | meaning | fails the gate? |
|---|---|---|
| `REPRODUCIBLE` | re-ran it; every shared headline number matches | no |
| `NOT_REPRODUCIBLE` | re-ran it; a shared headline number differs. The differing field names are reported | **yes** |
| `NO_SCRIPT` | no companion script exists next to the artefact | no |
| `NOT_RE_RUN` | no measurement could be taken: network, model, missing input, uninferable arguments, timeout, or an unparseable artefact. The reason is always stated | no |
| `LEGACY_NO_SCRIPT` | the companion script exists, but **no invocation of it can write this filename**, and the file carries no provenance | no |

`NOT_RE_RUN` is **not** a softer `NOT_REPRODUCIBLE`. It means *"no measurement
was taken"*. It must never be reported as a disagreement, and it must never
fail the gate — otherwise an offline machine would look like a repository full
of lies.

`LEGACY_NO_SCRIPT` is a refinement of `NO_SCRIPT`, for one specific case (see
§8): "the script next to it exists but cannot write it" and "there is no
script next to it at all" are different situations, and merging them hides a
real defect.

---

## 5. Results at `9c77f77`

**60 committed artefacts at `9c77f77`.** Every headline number in every
artefact that could be re-run was reproduced.

| verdict | count |
|---|---:|
| `REPRODUCIBLE` | **38** |
| `NOT_REPRODUCIBLE` | **0** |
| `NO_SCRIPT` | 4 |
| `NOT_RE_RUN` | 17 |
| `LEGACY_NO_SCRIPT` | 1 |
| **total** | **60** |

The gate's exit code at this commit is **0**: nothing in the current tree is
falsified. That is a statement about *this* commit, not a licence — §7 is
about the commit that was.

Run: `python experiments/artifact_repro.py --ref 9c77f77 --timeout 900`
(interpreter `/home/proart/projects/glasses/.venv/bin/python`, numpy 2.5.4).

The timeout is 900 s rather than the 120 s default because four of the scripts
are slow on this machine, and the 120 s default turned them into `NOT_RE_RUN`
for a reason that had nothing to do with their numbers:
`speculation_payoff.py` takes 389 s, `f0_speaker_change.py` 350 s,
`inthewild_probe.py` 225 s, and `notch_ab.py` did not finish at 900 s at all
(§12, item 2). At the 120 s default the same audit reports four fewer
`REPRODUCIBLE` and four more `NOT_RE_RUN`; raising the timeout converts
**three** of those into real measurements and changes **no** verdict.

### 5.1 The full table

| artefact | script | verdict | differing fields |
|---|---|---|---|
| `ascii_motion_cause_results.json` | `ascii_motion_cause.py` | **REPRODUCIBLE** | — |
| `ascii_motion_nulls_results.json` | `ascii_motion_nulls.py` | **REPRODUCIBLE** | — |
| `ascii_motion_results.json` | `ascii_motion.py` | **REPRODUCIBLE** | — |
| `ascii_motion_robustness_results.json` | `ascii_motion_robustness.py` | **REPRODUCIBLE** | — |
| `ascii_motion_threshold_results.json` | `ascii_motion_threshold.py` | **REPRODUCIBLE** | — |
| `av_sync_results.json` | `av_sync.py` | **REPRODUCIBLE** | — |
| `combined_filter_results.json` | `combined_filter.py` | **REPRODUCIBLE** | — |
| `crossmodal_probe_results.json` | `crossmodal_probe.py` | **REPRODUCIBLE** | — |
| `dialogue_intake_selftest_results.json` | `dialogue_intake_selftest.py` | **REPRODUCIBLE** | — |
| `drift_structure_results.json` | `drift_structure.py` | **REPRODUCIBLE** | — |
| `emptyroom_null_results.json` | `emptyroom_null.py` | **REPRODUCIBLE** | — |
| `f0_speaker_change_results.json` | `f0_speaker_change.py` | **REPRODUCIBLE** | — |
| `gain_invariance_results.json` | `gain_invariance.py` | **REPRODUCIBLE** | — |
| `gate_peak_coverage_results.json` | `gate_peak_coverage.py` | **REPRODUCIBLE** | — |
| `gate_peak_verdict_results.json` | `gate_peak_verdict.py` | **REPRODUCIBLE** | — |
| `inthewild_probe_results.json` | `inthewild_probe.py` | **REPRODUCIBLE** | — |
| `mic_filter_sim_results.json` | `mic_filter_sim.py` | **REPRODUCIBLE** | — |
| `pause_threshold_results.json` | `pause_threshold.py` | **REPRODUCIBLE** | — |
| `pose_test_results.json` | `pose_test.py` | **REPRODUCIBLE** | — |
| `retraction_sweep_results.json` | `retraction_sweep.py` | **REPRODUCIBLE** | — |
| `room_map_results.json` | `room_map.py` | **REPRODUCIBLE** | — |
| `row_band_removal_results.json` | `row_band_removal.py` | **REPRODUCIBLE** | — |
| `silence_leading_artefact_results.json` | `silence_leading_artefact.py` | **REPRODUCIBLE** | — |
| `speculation_payoff_results.json` | `speculation_payoff.py` | **REPRODUCIBLE** | — |
| `teacher_conventional_facing_both.json` | `teacher_conventional.py` | **REPRODUCIBLE** | — |
| `teacher_conventional_facing_map.json` | `teacher_conventional.py` | **REPRODUCIBLE** | — |
| `teacher_conventional_facing_region+map.json` | `teacher_conventional.py` | **REPRODUCIBLE** | — |
| `teacher_conventional_facing_region.json` | `teacher_conventional.py` | **REPRODUCIBLE** | — |
| `teacher_conventional_facing_summary.json` | `teacher_conventional.py` | **REPRODUCIBLE** | — |
| `teacher_conventional_speaking_both.json` | `teacher_conventional.py` | **REPRODUCIBLE** | — |
| `teacher_conventional_speaking_map.json` | `teacher_conventional.py` | **REPRODUCIBLE** | — |
| `teacher_conventional_speaking_region+map.json` | `teacher_conventional.py` | **REPRODUCIBLE** | — |
| `teacher_conventional_speaking_region.json` | `teacher_conventional.py` | **REPRODUCIBLE** | — |
| `teacher_conventional_speaking_summary.json` | `teacher_conventional.py` | **REPRODUCIBLE** | — |
| `teacher_data_requirement_results.json` | `teacher_data_requirement.py` | **REPRODUCIBLE** | — |
| `temporal_median_fix_results.json` | `temporal_median_fix.py` | **REPRODUCIBLE** | — |
| `turn_eval_results.json` | `turn_eval.py` | **REPRODUCIBLE** | — |
| `twospeaker_verify_results.json` | `twospeaker_verify.py` | **REPRODUCIBLE** | — |
| `teacher_conventional_speaking.json` | `teacher_conventional.py` | **LEGACY_NO_SCRIPT** | — (no invocation of the companion script can produce this filename (the script writes teacher_conventional_{task}_{features}.json, and this file has no feature set); the file also carries no provenance block) |
| `real_interrupt_rate_results.json` | `real_interrupt_rate.py` | **NO_SCRIPT** | — (no companion script next to this artefact) |
| `turn_routes_causal_results.json` | `turn_routes_causal.py` | **NO_SCRIPT** | — (no companion script next to this artefact) |
| `turn_routes_clef_results.json` | `turn_routes_clef.py` | **NO_SCRIPT** | — (no companion script next to this artefact) |
| `turn_routes_jev_results.json` | `turn_routes_jev.py` | **NO_SCRIPT** | — (no companion script next to this artefact) |
| `clef_audio_native_results.json` | `clef_audio_native.py` | **NOT_RE_RUN** | — (not re-run: needs the Cloudflare Workers AI API (network) and a TYPESAFE_API_KEY) |
| `cloud_multimodal_results.json` | `cloud_multimodal.py` | **NOT_RE_RUN** | — (not re-run: needs the Cloudflare Workers AI API (network) and a TYPESAFE_API_KEY) |
| `cloud_vision_probe_results.json` | `cloud_vision_probe.py` | **NOT_RE_RUN** | — (not re-run: needs the Cloudflare Workers AI API (network) and a TYPESAFE_API_KEY) |
| `device_generalisation_results.json` | `device_generalisation.py` | **NOT_RE_RUN** | — (not re-run: flushes CPU caches and measures them (needs privileges and ~11 min); its numbers are machine-specific by construction) |
| `hum_sweep_results.json` | `hum_sweep.py` | **NOT_RE_RUN** | — (not re-run: the committed artefact is not valid JSON: Extra data: line 319 column 1 (char 6874)) |
| `jev_jaggedness_results.json` | `jev_jaggedness.py` | **NOT_RE_RUN** | — (not re-run: calls the live jev model over the network) |
| `jev_round2_results.json` | `jev_round2.py` | **NOT_RE_RUN** | — (not re-run: calls the live jev model over the network) |
| `judged_calibration_results.json` | `judged_calibration.py` | **NOT_RE_RUN** | — (not re-run: calls the live jev model over the network) |
| `local_vs_cloud_results.json` | `local_vs_cloud.py` | **NOT_RE_RUN** | — (not re-run: needs both a local Ollama server and the cloud model) |
| `notch_ab_results.json` | `notch_ab.py` | **NOT_RE_RUN** | — (not re-run: timed out after 900s) |
| `question_rewrite_results.json` | `question_rewrite.py` | **NOT_RE_RUN** | — (not re-run: calls the live jev model over the network) |
| `small_model_sweep_results.json` | `small_model_sweep.py` | **NOT_RE_RUN** | — (not re-run: needs a local Ollama model server, and the committed script does not even parse (SyntaxError at line 148)) |
| `spectrogram_turn_eval_results.json` | `spectrogram_turn_eval.py` | **NOT_RE_RUN** | — (not re-run: calls the live jev model over the network) |
| `spectrogram_vision_results.json` | `spectrogram_vision.py` | **NOT_RE_RUN** | — (not re-run: calls the live jev model over the network) |
| `symbolic_audio_results.json` | `symbolic_audio.py` | **NOT_RE_RUN** | — (not re-run: calls the live jev model over the network) |
| `turn_routes_results.json` | `turn_routes.py` | **NOT_RE_RUN** | — (not re-run: the smartturn stage needs the Cloudflare API (network)) |
| `watcher_cost_results.json` | `watcher_cost.py` | **NOT_RE_RUN** | — (not re-run: measures CPU cost on this machine; the numbers are a property of the host, not of the code) |

---

## 6. What was not re-run, and why

Nothing in this list is a falsified number. Each is a case where **no
measurement was taken**, and the reason is stated so that a reader can tell
"we did not check this" from "we checked and it did not hold". The tool
records the same reasons in the manifest under `results[].reason`.

**7** — calls the live jev model over the network

- `jev_jaggedness_results.json`
- `jev_round2_results.json`
- `judged_calibration_results.json`
- `question_rewrite_results.json`
- `spectrogram_turn_eval_results.json`
- `spectrogram_vision_results.json`
- `symbolic_audio_results.json`

**3** — needs the Cloudflare Workers AI API (network) and a TYPESAFE_API_KEY

- `clef_audio_native_results.json`
- `cloud_multimodal_results.json`
- `cloud_vision_probe_results.json`

**1** — flushes CPU caches and measures them (needs privileges and ~11 min); its numbers are machine-specific by construction

- `device_generalisation_results.json`

**1** — measures CPU cost on this machine; the numbers are a property of the host, not of the code

- `watcher_cost_results.json`

**1** — needs a local Ollama model server, and the committed script does not even parse (SyntaxError at line 148)

- `small_model_sweep_results.json`

**1** — needs both a local Ollama server and the cloud model

- `local_vs_cloud_results.json`

**1** — the committed artefact is not valid JSON: Extra data: line 319 column 1 (char 6874)

- `hum_sweep_results.json`

**1** — the smartturn stage needs the Cloudflare API (network)

- `turn_routes_results.json`

**1** — timed out after 900s

- `notch_ab_results.json`


### 6.1 Artefacts with no companion script at all

- `real_interrupt_rate_results.json`
- `turn_routes_causal_results.json`
- `turn_routes_clef_results.json`
- `turn_routes_jev_results.json`

These are not "the script is broken"; the script simply is not in the
repository. `turn_routes.py` writes `turn_routes_results.json`; the three
`turn_routes_{causal,clef,jev}_results.json` files were written by
`turn_routes.py` under three different flags, and no such script exists.
`real_interrupt_rate.py` does not exist either. Until a script is committed
next to them, the numbers in those four files cannot be checked by anything.

### 6.2 One artefact that is not valid JSON

`hum_sweep_results.json` is two JSON documents concatenated into one file
(`Extra data: line 319 column 1`). The tool reports it as `NOT_RE_RUN` with
that parse error rather than silently reading the first document. The file
itself is committed and is left exactly as it is — repairing it would be
editing a committed artefact to make an audit pass, which is the opposite of
the point.

---

## 7. The 0.675

### 7.1 What the number is

`0.675` is the **leave-one-session-out accuracy for held-out session B**, in
`experiments/teacher_conventional_speaking_both.json` — the conventional
teacher, `speaking` task, `both` feature set. It is one cell of one table in
one artefact. It was written by commit **`7a4dff5`** (*"Re-run the conventional
teacher on the 'both' feature set: margin over the best control is now 0.0"*,
2026-10-19).

It exists in exactly one commit. The commit before it (`415ecb2`) had **0.65**
in that cell; the commit after it (`d6baa6d`) regenerated the file with
**0.000** in that cell, and the current tree (`9c77f77`) holds **0.000**.

### 7.2 How this was established

By the tool, against the commit that wrote it, which is the only way the
question can be asked honestly:

```bash
python experiments/artifact_repro.py --ref 7a4dff5 --only teacher_conventional
```

**Exit code 1.** The verdict for `teacher_conventional_speaking_both.json` at
`7a4dff5` is **`NOT_REPRODUCIBLE`**, with three differing fields:

| field | committed | re-run |
|---|---:|---:|
| `leave_one_session_out[held_out=B].acc` | 0.675 | 0.0 |
| `acc_test_spatially_scrambled` | 1.0 | 0.9791666666666666 |
| `margin_over_best_control` | 0.0 | 0.02083333333333337 |

Everything else in that artefact reproduces, and the other four feature sets
in the same family reproduce **exactly**:

| artefact (at `7a4dff5`) | verdict |
|---|---|
| `teacher_conventional_speaking_summary.json` | REPRODUCIBLE |
| `teacher_conventional_speaking_map.json` | REPRODUCIBLE |
| `teacher_conventional_speaking_region.json` | REPRODUCIBLE |
| `teacher_conventional_speaking_region+map.json` | REPRODUCIBLE |
| `teacher_conventional_facing_both.json` | REPRODUCIBLE |
| **`teacher_conventional_speaking_both.json`** | **NOT_REPRODUCIBLE** |

**One correction to the way the finding is usually stated.** The tool's run
above uses the script's **default** corpus, `/mnt/c/proart_capture/corpus`,
because the artefact at `7a4dff5` records no corpus at all. Under that corpus
the *controls* differ too, which is why three fields are listed above. The
artefact's own control columns (scrambled 1.000, margin 0.000) match
`/mnt/c/proart_capture/corpus_rm3` **exactly**, and re-running the `7a4dff5`
script against `corpus_rm3` reproduces every other column bit-exactly and
differs in **exactly one** field:

```
committed 7a4dff5 artefact vs re-run with corpus_rm3:
    leave_one_session_out[held_out=B].acc : 0.675 -> 0.0
```

So the statement *"re-running the committed script against the committed corpus
reproduces every other column exactly and returns LOO A 0.000, B 0.000"*
(`TEACHER-PIPELINE.md:115-122`, `NOTES-AUDIT.md:84-101`,
`COMBINED-FILTER.md:99-110`) is **true for `corpus_rm3`** and **not true for
the script's default corpus**. That difference is not a quibble: it is the
whole point. The artefact did not record which corpus produced it, so "the
committed corpus" is not a well-defined phrase, and the two corpora on disk
give different control values. The number that is *not* reproducible is
`0.675`, under both corpora.

The current tree's script is not the script that wrote the number (it gained
the provenance API in `d6baa6d`: blob `459e23c0…` at `7a4dff5`, `f8f89c96…`
at `d6baa6d` and after), so the finding was checked against **both**:
the `7a4dff5` script and the current script (blob `f8f89c96…`, unchanged
since `d6baa6d`), against both corpora. All
four combinations return LOO B = 0.0. The conclusion does not depend on which
script version is used.

> A correction to `NOTES-AUDIT.md:93-94`, which said *"the script at `7a4dff5`
> and at `HEAD` are identical"*. That was true when it was written — `HEAD`
> was then `7a4dff5`'s neighbour — and it is **no longer true**: the script
> gained the provenance API in `d6baa6d` and its blob changed from
> `459e23c0…` to `f8f89c96…`. The substantive claim survives, because both
> versions return LOO B = 0.000. It is a good illustration of why a claim
> about "the script" needs the script's hash next to it.
>
> **This correction has landed.** `9c77f77` rewrote that bullet of
> `NOTES-AUDIT.md` to say the two scripts *"are **not** the same file"* and
> that *"both* versions return 0.000" — the wording this report proposed. The
> outstanding item is closed; it is recorded here because this report was
> written against `c71a6c5` and the reader needs to know the correction is in
> the tree, not just in this file.

### 7.3 The sweep is committed, and the gate can check it too

The docs' claim that *"a sweep of 352 configurations over the parameters the
artefact never recorded never produces 0.675 together with the artefact's
other columns"* used to rest on an uncommitted script. **That script is now
committed**: `experiments/retraction_sweep.py`, with its result in
`experiments/retraction_sweep_results.json` (`9c77f77`).

What the committed sweep says, read from the artefact:

| | |
|---|---|
| question | is the withdrawn leave-one-session-out accuracy of 0.675 reachable in a sweep over the parameters the artefact never recorded? |
| answer | **NO** |
| configurations planned / run | 1280 / 1280 |
| the grid | 4 filter variants × 4 frame counts × 8 strides × 5 max-pers × 2 tasks, features `both` |
| reproductions | **0** |
| distinct LOO-B values seen | 53 (none of them 0.675) |
| corpus rebuild check | **3 of 3 session files byte-identical** (`A_rm3`, `B_rm3`, `C_rm3`) |
| elapsed | 186.6 s |

The earlier "352 configurations" figure was an earlier, smaller sweep; the
committed one is 1280, and it is the one to quote.

**And the sweep is itself checkable.** `retraction_sweep_results.json` is an
artefact in the gate's scope, and the gate re-ran it at `9c77f77`:
**`REPRODUCIBLE`**. So the claim "0.675 is unreachable" is no longer an
assertion in a note — it is a committed script, a committed result file, and a
verdict that a third party can re-derive with one command.

### 7.4 What this means for quoting documents

1. **`0.675` must not be quoted as a measurement.** It is not a number the
   committed code and the committed inputs can produce. It is now quoted in
   four documents (§11), and in every one of them it appears **only inside a
   withdrawal** — that is the correct treatment, and it should stay that way.
2. **A number whose artefact records no inputs cannot be re-derived; it can
   only be withdrawn.** This is the generalisable lesson, and it is why the
   provenance blocks now exist (§10).
3. **The tool makes this checkable rather than a matter of memory.** Before
   this tool, "is that number still reachable?" was answered by whoever
   happened to remember. Now it is a command with an exit code.

### 7.5 What this does NOT mean

- **It does not mean the teacher pipeline is broken.** Four of the five
  feature sets in the same family reproduce bit-exactly, at the same commit,
  with the same script. The pipeline, its controls and its margins all
  reproduce; one cell of one table does not.
- **It does not mean the controls were wrong.** The spatially-scrambled
  control reproduces at exactly 1.000 from `corpus_rm3`.
- **It does not mean `0.000` is "the true value" either.** It is the value
  the committed script and the corpus on disk produce. The artefact's own
  `limits` say the leave-one-session-out table is *"a sanity check, not a
  measurement"* — two labelled sessions, one speaker, one room, one lighting
  state, and no error bars. Neither 0.675 nor 0.000 is a claim about the
  world.
- **It does not mean the corpus is the variable.** The artefact's control
  columns identify `corpus_rm3`, and 0.675 still does not appear from it.
  `NOTES-AUDIT.md:95-97` adds that the corpus rebuilds byte-for-byte from the
  captures (`seed_corpus_from_captures.py --filter row+median3` produces three
  SHA-256-identical `.npz` files); that rebuild was **not re-run here**, and
  the conclusion above does not depend on it.
- **It does not mean the file is wrong now.** `d6baa6d` regenerated
  `teacher_conventional_speaking_both.json` with the provenance block, and
  the current file is `REPRODUCIBLE`: it carries `corpus_path`, the full
  `cli_args`, the session SHA-256s and a `corpus_fingerprint`, and it
  reproduces every headline number. The current tree is clean; what the
  current tree cannot do is produce the *quoted* 0.675, because the quoted
  0.675 was never reproducible.

---

## 8. The legacy orphan: `teacher_conventional_speaking.json`

`experiments/teacher_conventional_speaking.json` is committed, and it is
**not a current measurement**.

It has a companion script — `teacher_conventional.py` — but **no invocation of
that script can write that filename**: the script always writes
`teacher_conventional_{task}_{features}.json`, and this file names no feature
set. The file is also visibly older than the script: it carries
`acc_test_shuffled` and `acc_test_shuffled_train` (the *pre-spatial-scramble*
control) where the current script writes `acc_test_spatially_scrambled`, it
has no `features` key, and it carries **no provenance block**. It is a
leftover from a script version that predates the spatially-scrambled control.

The tool therefore reports it as **`LEGACY_NO_SCRIPT`** — a refinement of
`NO_SCRIPT` — rather than folding it into `NO_SCRIPT` ("no script at all").
The two situations are different: one is a file with nothing to check it
against, the other is a file whose companion script *exists* and is
*actively unable* to produce it. Merging them hides a real defect.

**What it holds:** `acc_test` 1.0, `acc_test_shuffled` 0.9375,
`acc_test_shuffled_train` 0.986, `acc_test_blank` 0.667, `margin_over_best_control`
0.0625, `usable` false, LOO A 0.0 and B 0.0.

**What to do with it:** nothing, yet. It is **not** to be read as a current
measurement of anything, and it is **not** to be deleted or hand-edited as
part of this work — editing or removing a committed artefact to make an audit
pass is the failure mode this whole tool exists to prevent. The correct
retirement is a separate, explicit decision: re-run the experiment under the
current script and let the new artefact supersede it, or record why the old
file is kept. The tool reports the file as a defect to be *decided about*, not
as a number that was falsified, so it does not fail the gate.

(No `docs/research/` document references this file.)

---

## 9. A filename that is a function of CLI order

Two committed artefacts are the same experiment with the same three models
and the same mode, and their filenames differ only by the **order of the model
list**:

- `experiments/debrief_audit_blind_cf-meta-llama-3-3-70b-instruct-fp8-fast_cf-openai-gpt-oss-120b_cf-qwen-qwen3-30b-a3b-fp8.json`
- `experiments/debrief_audit_blind_cf-openai-gpt-oss-120b_cf-meta-llama-3-3-70b-instruct-fp8-fast_cf-qwen-qwen3-30b-a3b-fp8.json`

The cause is one line in `experiments/debrief.py:876`:

```python
p = outdir / f"debrief_audit_{a.mode}_{'_'.join(slug(m) for m in models)}.json"
```

`models` is the `--models` list **in the order it was given on the command
line**, so the filename is a function of the *invocation*, not of the *model
set*. The same three models, passed in a different order, land in a different
file.

That is not cosmetic. The history shows the two files have held **different
claim sets**:

- the `meta-llama…`-first file was created at `6d1456b` with the old
  **6 false / 4 true** claim set;
- the `openai…`-first file was created at `1c564c5` with the corrected
  **7 false / 3 true** set;
- `d6baa6d` then **overwrote** the first file (same order → same name) with
  the 7/3 set.

So today both files hold the 7/3 set, but they are still two different runs
of the same nominal experiment, and the only thing distinguishing them in the
filename is a permutation. In the two files the `@cf/openai/gpt-oss-120b` row
reads `specificity_kept_true_claims` **1.0** in one and **0.666…** in the
other, with `false_positives` 0 and 1. A reader who assumes the
`meta-llama…`-first file is the old 6/4 run is reading 7/3 numbers from a
filename that suggests otherwise.

**A canonical filename would look like this** — the model list sorted, plus a
date and the claim-set identity, so the name describes the *experiment* and
not the *command line*:

```
debrief_audit_blind_2026-10-19_7f3_cf-meta-llama-3-3-70b-instruct-fp8-fast_cf-openai-gpt-oss-120b_cf-qwen-qwen3-30b-a3b-fp8.json
```

The one-line fix is `"_".join(sorted(slug(m) for m in models))` at
`experiments/debrief.py:876`, plus recording the claim-set identity and the
date in the file's `provenance` block.

**No committed file was renamed.** Renaming would break every reference to
the current names, and the fix belongs with whoever owns `debrief.py`. This
report states the defect; it does not act on it.

**Scope note:** these two files are *outside* this tool's discovery scope,
which is `experiments/*_results.json` plus `experiments/teacher_conventional_*.json`.
The `debrief_*.json` family uses a different naming convention, and a
convention rule for it would have to be written before it could be audited.

---

## 10. The provenance blocks, and what they bought

`teacher_conventional.py`, `combined_filter.py` and `turn_routes.py` now
write a `provenance` block into their result files, and a top-level
`reproducible: bool`. The block records:

`generated_by`, `script_path`, `script_sha256`, `corpus_path`, `cli_args[]`,
`command_line`, `parameters{}`, `sessions[]`, `corpus_fingerprint`, `problems[]`.

What that bought, measured:

1. **The corpus is named.** `teacher_conventional_*.json` records
   `corpus_path: /mnt/c/proart_capture/corpus` and
   `corpus_fingerprint: 5f6327786ad0713f…`. The fingerprint is a hash of the
   session `.npz` files, so it identifies the *input data* independently of
   where the corpus happens to live on disk. That is what makes "same inputs"
   checkable rather than a path comparison.
2. **The exact invocation is recorded.** `cli_args` lists the arguments the
   script was actually called with, including `--corpus`, `--k`, `--stride`,
   `--max-per`, `--train-frac`. The tool uses those recorded arguments for
   the re-run, which is why the current `teacher_conventional_*` artefacts
   reproduce with the corpus the file itself names.
3. **The script is pinned by hash.** `script_sha256` ties the file to the
   code that wrote it.

The expected verdicts hold. At `9c77f77` all ten
`teacher_conventional_{speaking,facing}_{summary,region,map,region+map,both}.json`
artefacts are **`REPRODUCIBLE`**, including `speaking_both.json` — the file
that now holds 0.000, with the corpus and the arguments recorded next to it.

The provenance blocks do not retroactively fix `7a4dff5`'s artefact. They make
the *next* one checkable.

---

## 11. Documents that quote a number from a `NOT_REPRODUCIBLE` artefact

At `9c77f77` there is **no** `NOT_REPRODUCIBLE` artefact, so no document
quotes a number from one. The list below is the historical one: the artefacts
that are `NOT_REPRODUCIBLE` when the audit is pinned to the commit that wrote
them, and every `docs/research/` line that quotes a number from them.

### 11.1 The 0.675 (`teacher_conventional_speaking_both.json` at `7a4dff5`)

The artefact is `NOT_REPRODUCIBLE` (§7.2). Every document that quotes a
number from it, with line numbers:

| document | lines | what it quotes |
|---|---|---|
| `docs/research/TEACHER-PIPELINE.md` | 102, 109, 113, 114, 115, 122 | the 0.675, as a **withdrawn** number and as a "candidate source" that is then withdrawn |
| `docs/research/NOTES-AUDIT.md` | 81, 84, 101, 106 | the 0.675, in the F3 amendment that withdraws it |
| `docs/research/COMBINED-FILTER.md` | 99, 109, 110 | the 0.675, in the correction that withdraws it |
| `docs/research/LLM-DIRECTOR.md` | 53, 60 | the 0.675, in the correction that withdraws it |

**In all four documents the 0.675 appears only inside a withdrawal.** No
document presents it as a live measurement, and none should.

For contrast, the *replacement* values from the regenerated artefact
(scrambled 0.979, margin 0.0208) appear at
`docs/research/TEACHER-PIPELINE.md:70, 82, 104` and
`docs/research/COMBINED-FILTER.md:87`.

### 11.2 The rule this implies

A document that quotes a number from an artefact is making a claim that the
artefact supports. When the artefact stops being reproducible, the document
that quotes it is making a claim that nothing supports — and the reader has
no way to know, because a note looks exactly the same whether its number is
checkable or not. That is the gap `artifact_repro.py` closes: it turns "is
this number still reachable?" from a question about memory into a command
with an exit code.

---

## 12. What could not be measured

Stated plainly, so that the table above is not read as more than it is.

1. **17 of 57 artefacts were not measured at all** (§6). They are
   `NOT_RE_RUN`, with the reason recorded: the network, a model server, a
   missing input, an unparseable file, or a timeout. **No claim in this
   report extends to them.** They are not "reproducible" and they are not
   "falsified"; nobody checked.
2. **`notch_ab.py` did not finish.** It ran for the full 900 s timeout and was
   killed. This machine is slow for these audio scripts:
   `speculation_payoff.py` took 389 s, `f0_speaker_change.py` 350 s,
   `inthewild_probe.py` 225 s. Whether `notch_ab` reproduces is **unknown**,
   and the report does not guess.
3. **`device_generalisation_results.json` and `watcher_cost_results.json` were
   skipped deliberately.** They measure CPU on the host; the numbers are a
   property of the machine, not of the code, so re-running them would produce
   a number that differs for reasons that say nothing about the claim. The
   tool's exclusion list says so rather than pretending to check them.
4. **The 57 artefacts are the ones this tool can name.** The discovery rule
   is `experiments/*_results.json` plus `experiments/teacher_conventional_*.json`.
   The `debrief_*.json`, `director_*.json` and other families are outside it
   and were **not** audited (§9 covers one of them by hand). **This report says
   nothing about them.**
5. **The verdict depends on the inputs that are on disk today.** A corpus
   that has been re-captured since the artefact was written would show up as
   `NOT_REPRODUCIBLE`, correctly, but the tool cannot tell "the input changed"
   from "the code changed" on its own — that is what the `provenance` block's
   `corpus_fingerprint` is for, and only artefacts that carry one get that
   answer for free.
6. **The audit is pinned to a commit.** The working tree of this repo moves
   while people work in it — another agent committed repeatedly while this
   audit was running, and the tree has since moved on again. The tool audited
   the **committed** code and the **committed** artefacts at the named
   revision, which is why the verdict is the same for everybody who checks
   that commit out, and why the earlier `c71a6c5` results and the current
   ones can be compared like for like.

   > **CORRECTION (2026-10-20).** An earlier draft of this item read
   > *"another agent has uncommitted changes to `experiments/pause_threshold.py`
   > and `experiments/pause_threshold_results.json` right now"*, and a still
   > earlier draft of this report was audited at `c71a6c5`.
   >
   > **What is true now:** both files are committed and the working tree is
   > clean. `experiments/pause_threshold.py` and
   > `experiments/pause_threshold_results.json` were changed in `d4a49c1`
   > (*"The director's PROPOSAL is stable in substance even though its
   > verdicts are not"*) and have been unchanged since, and `9c77f77` carries
   > them forward. The results file was **regenerated** with a new
   > `edge_silences` block per cell, and its committed SHA-256 changed with
   > it:
   >
   > | revision | SHA-256 of `experiments/pause_threshold_results.json` | `edge_silences` per cell |
   > |---|---|---|
   > | `c71a6c5` | `4fc69b5ac12d99c1e3846a9a50228a26605243798b5c2714717d0bf74abbbe3c` | absent |
   > | `d4a49c1` … `9c77f77` | `168d2c48e2540f21ee73dfbe8ccf0fccfc3add6d06294bca3b1186df86568e95` | present (7 cells) |
   >
   > The file is still `REPRODUCIBLE`: the audit re-ran the committed
   > `pause_threshold.py` and reproduced every headline number of the
   > regenerated file. **The point of the change is that a result file now
   > carries its own hash history** — which is the same discipline this whole
   > report is about, applied to the artefact rather than to the note.
7. **The `hum_sweep_results.json` parse error is not diagnosed here.** The
   tool reports that the committed file is two JSON documents; which of them
   is authoritative is a question for whoever owns `hum_sweep.py`.
8. **The symlinked data directories are writable.** `examples/`, `models/`,
   `.scratch/`, `tools/`, `workers/` and `docs/` are symlinked into the
   scratch tree, so a script that *wrote* into one of them would write into
   the repository. **No audited script does** — the only mentions of
   `docs/` in the audited scripts are in docstrings — and the test asserts
   the repository's `experiments/` is untouched. But the guarantee is
   currently behavioural rather than structural, and it should be made
   structural (drop the `docs` symlink, or mount the data directories
   read-only) before a script that writes there is audited.

---

## 13. What the gate says about the new artefacts

This report was first audited against `c71a6c5`; the tree has since moved to
`9c77f77`. Stated plainly, so that "the gate says nothing about it" is not
left ambiguous:

### 13.1 New artefacts the gate **does** pick up

Three result files have entered the discovery scope since `c71a6c5`, because
they follow the `*_results.json` convention:

| artefact | companion script | verdict at `9c77f77` |
|---|---|---|
| `dialogue_intake_selftest_results.json` | `dialogue_intake_selftest.py` | **REPRODUCIBLE** |
| `silence_leading_artefact_results.json` | `silence_leading_artefact.py` | **REPRODUCIBLE** |
| `retraction_sweep_results.json` | `retraction_sweep.py` | **REPRODUCIBLE** |

A fourth, `pause_threshold_results.json`, was **regenerated** with a new
`edge_silences` block per cell and is still **REPRODUCIBLE** (§12, item 6).

The retraction sweep is the interesting one: §7.3 is the claim that 0.675 is
unreachable, and the claim is now an artefact the gate re-derives.

### 13.2 New artefacts the gate does **not** pick up

These are committed result files that the discovery rule does not name,
because they do not end in `_results.json` and are not in the teacher family:

| artefact | why it is outside the scope |
|---|---|
| `director_majority_analysis.json` | name does not end in `_results.json` |
| `director_proposal_stability.json` | name does not end in `_results.json` |
| `director_repeatability_blind.json` | name does not end in `_results.json` |
| `director_repeatability_context_140000.json` | name does not end in `_results.json` |
| `director_repeatability_context_60000.json` | name does not end in `_results.json` |
| `debrief_audit_blind_cf-meta-llama-3-3-70b-instruct-fp8-fast_cf-openai-gpt-oss-120b_cf-qwen-qwen3-30b-a3b-fp8.json` | name does not end in `_results.json`; see §9 |
| `debrief_audit_blind_cf-openai-gpt-oss-120b_cf-meta-llama-3-3-70b-instruct-fp8-fast_cf-qwen-qwen3-30b-a3b-fp8.json` | name does not end in `_results.json`; see §9 |
| `debrief_audit_context_cf-openai-gpt-oss-120b.json` | name does not end in `_results.json` |
| `debrief_audit_context_cf-openai-gpt-oss-120b_cf-meta-llama-3-3-70b-instruct-fp8-fast.json` | name does not end in `_results.json` |
| `debrief_cf-openai-gpt-oss-120b.json` | name does not end in `_results.json` |

**The gate says nothing about these ten files.** That is a limitation of the
naming convention the gate reads, not a statement that they are in order. Two
of them (`debrief_audit_blind_*`) are the order-dependent-filename pair of §9,
and one of them (`director_proposal_stability.json`) would need a cloud model
to re-run in any case.

**The fix is a convention, not a code change:** either rename these files to
end in `_results.json`, or extend the discovery rule to name the families
explicitly. Either way it is a decision for whoever owns the naming, and it
is recorded here rather than acted on.

---

## 14. Running it

```bash
# the whole tree, at the current commit
python experiments/artifact_repro.py

# a machine-readable manifest, and a pinned commit
python experiments/artifact_repro.py --ref 9c77f77 --json /tmp/manifest.json

# the plan, without running anything
python experiments/artifact_repro.py --list

# one family
python experiments/artifact_repro.py --only teacher_conventional
```

Options: `--json PATH`, `--timeout SECONDS` (per script, default 120),
`--scratch DIR` (default `/tmp/artifact_repro`), `--only SUBSTRING`,
`--source {head,worktree}`, `--ref REV`, `--list`.

Exit code 1 iff some artefact is `NOT_REPRODUCIBLE`.

The tests:

```bash
python -m pytest tests/test_artifact_repro.py -q
```

They use a throwaway git repository in `tmp_path` as the repo under audit, and
an **injected fake runner** for the classification tests, so they neither
touch the real tree nor need the `/mnt/c` corpus, the network, or a model.
**17 tests, all passing.**

They cover: exit 0 on a reproducing fixture *and* the verdict behind it;
exit 1 on a non-reproducing fixture; `NO_SCRIPT` for an artefact with no
companion script; `NOT_REPRODUCIBLE` with the **differing field names**
reported; `LEGACY_NO_SCRIPT` for the legacy orphan; `NOT_RE_RUN` for a failed
run (and that it does not fail the gate); key-count differences not counting
as falsifications; list rows addressed by identity rather than position;
volatile fields excluded; the repository's `experiments/` directory unchanged
in both bytes and mtime after a full run, with a script that deliberately
writes next to itself; the scratch run landing in the scratch tree; the
manifest recording the audited commit, the interpreter and the numpy version;
the git blob hash being recorded; and importing the tool running nothing.

That the tests can actually fail was checked by mutation: making `compare()`
never report a difference turns 2 of them red, and making the audited script
run inside the repository instead of the scratch tree turns 4 of them red,
including the one that asserts the repository is untouched.
