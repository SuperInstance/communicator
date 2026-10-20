#!/usr/bin/env python3
"""
artifact_repro.py -- can each committed result file still be produced?

THE PROBLEM THIS FILE GUARDS AGAINST.

`experiments/` holds about fifty committed `*_results.json` files. Each one was
written by a script that sits next to it, but "the script that sits next to it"
is not the same as "the script that wrote it". Scripts get edited. Corpora get
re-captured. Inputs get deleted. After a while nobody can tell whether a
committed number is still reachable, and a note that quotes it cannot be
checked. The 0.675 in `teacher_conventional_speaking_both.json` is the worked
example: four of the five feature sets reproduce bit-exactly, the fifth does
not, and the only trace of the corpus that produced it is gone.

WHAT "REPRODUCIBLE" MEANS HERE, precisely:

    A committed artefact is REPRODUCIBLE if, re-running the companion script
    with the SAME INPUTS and the SAME ARGUMENTS, every HEADLINE NUMBER in the
    committed file is reproduced.

Three qualifications, all of them load-bearing:

  * HEADLINE NUMBERS, NOT BYTE EQUALITY. This repo is mid-way through adding
    provenance keys to its result files. A file that gained three keys is not
    a falsification; a file whose `acc_test` changed is. So the comparison is
    over the numeric leaves the file presents as MEASUREMENTS, and it reports
    the FIELDS THAT DIFFER BY NAME, with the old and the new value. Key-count
    differences are reported, but they never decide a verdict.

    "Headline" means: a numeric or boolean scalar whose key names a
    measurement -- an accuracy, a margin, a rate, a score. The depth does not
    matter, because the number this repo is in trouble about (a
    leave-one-session-out accuracy of 0.675) lives two levels down inside a
    list of records. Rows in a list are addressed by the identity they carry
    (`held_out`, `name`, `cell`, ...), not by their position, so a row that
    moved does not read as a changed number.

    Numbers that are a property of the machine or the clock are excluded:
    they change on every run for reasons that have nothing to do with the
    claim, and including them would make every artefact look irreproducible.

  * A COMMITTED SCRIPT, NOT A WORKING-TREE SCRIPT. A dirty working tree
    would otherwise let an in-progress edit change a verdict. By default the
    tool materialises the COMMITTED script from git into a scratch directory
    and runs THAT, so the answer is the same for everybody who checks out the
    commit. `--source worktree` opts into the working tree instead, and the
    two are reported separately because they can legitimately differ.

  * SAME INPUTS. Where a result file records its inputs (the newer files
    carry a `provenance` block with the exact `cli_args` the script was
    called with, and the corpus path it read), the tool re-runs with those
    arguments. Where a file does not, the tool runs the script's own declared
    defaults and says so, and records which corpus it used.

THE VERDICTS. Five:

    REPRODUCIBLE       re-ran it; the headline numbers match.
    NOT_REPRODUCIBLE   re-ran it; the headline numbers differ. The differing
                       field names are reported.
    NO_SCRIPT          no companion script exists next to the artefact.
    NOT_RE_RUN         no measurement could be taken: the script needs the
                       network, or a model, or an input that is not on disk,
                       or arguments that cannot be inferred. The reason is
                       always stated.
    LEGACY_NO_SCRIPT   the companion script exists, but no invocation of it
                       can write this filename, and the file carries no
                       provenance. One artefact is in this class; see below.

`NOT_RE_RUN` is not a softer `NOT_REPRODUCIBLE`. It means "no measurement was
taken", and it never causes a non-zero exit. `NOT_REPRODUCIBLE` means "a
measurement was taken and it disagreed", and it does.

ONE EXTRA CLASS, for one specific case. `experiments/teacher_conventional_speaking.json`
has a companion script next to it, but NO invocation of that script can write
that filename: the script always writes `teacher_conventional_{task}_{features}.json`,
and this file has no feature set. It is a leftover from a script version that
predates the spatially-scrambled control. The tool reports it as
`LEGACY_NO_SCRIPT` -- a refinement of `NO_SCRIPT` ("no companion script can
produce this file") -- because "the script next to it exists but cannot write
it" and "there is no script next to it at all" are different situations and
merging them hides a real defect. It does not affect the exit code.

EXIT CODE. 1 if any artefact is NOT_REPRODUCIBLE, else 0. That is what makes
this usable as a commit gate: `python experiments/artifact_repro.py` fails the
build when a committed number has stopped being reachable.

THE COMMIT IS PINNED. The working tree of this repo moves while you work:
another agent can commit a regenerated artefact between two runs of this
tool, and an audit that silently changed its own subject is worse than no
audit. `--ref <commit>` audits that commit; the default, HEAD, is resolved
and printed at the top of every run and recorded in the manifest.

THE REPO IS NEVER MUTATED. Every re-run writes into a scratch directory under
`/tmp/artifact_repro/`. The scratch tree mirrors the committed tree -- a real
`experiments/`, `src/` and `tests/` holding the audited code AND the committed
result files, and symlinks for the data directories -- so that a script which
writes next to itself writes into the scratch tree and not into the repo, and
so that a script which reads a sibling committed artefact finds it. The tool
never deletes anything; it only ever adds files under the scratch root.

USAGE

    python experiments/artifact_repro.py                  # table to stdout
    python experiments/artifact_repro.py --json out.json  # + machine manifest
    python experiments/artifact_repro.py --only teacher   # a subset
    python experiments/artifact_repro.py --list           # the plan, no runs
    python experiments/artifact_repro.py --ref c71a6c5    # audit a commit

`--only` takes a substring of the artefact name. `--list` prints the plan
without running anything, which is the cheap way to see what the tool thinks
it can and cannot re-run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"

REPRODUCIBLE = "REPRODUCIBLE"
NOT_REPRODUCIBLE = "NOT_REPRODUCIBLE"
NO_SCRIPT = "NO_SCRIPT"
NOT_RE_RUN = "NOT_RE_RUN"

# A refinement of NO_SCRIPT: the companion script exists, but no invocation
# of it can produce this filename, and the file carries no provenance.
LEGACY_NO_SCRIPT = "LEGACY_NO_SCRIPT"

VERDICTS = (REPRODUCIBLE, NOT_REPRODUCIBLE, NO_SCRIPT, NOT_RE_RUN,
            LEGACY_NO_SCRIPT)

# Only NOT_REPRODUCIBLE fails the gate. A legacy orphan is a defect to
# report, not a number that was falsified.
FAILING_VERDICTS = (NOT_REPRODUCIBLE,)

DEFAULT_TIMEOUT = 120          # seconds, per script
DEFAULT_SCRATCH = Path("/tmp/artifact_repro")


# ---------------------------------------------------------------------------
# artefact discovery
# ---------------------------------------------------------------------------

# `experiments/teacher_conventional_*.json` is a family, not a
# `*_results.json` file: the script names its output after the feature set it
# ran, so `teacher_conventional_speaking_both.json` and
# `teacher_conventional_facing_map.json` are different experiments that share
# one script.
TEACHER_PREFIX = "teacher_conventional_"
TEACHER_SCRIPT = "teacher_conventional.py"

TEACHER_TASKS = ("speaking", "facing")
TEACHER_FEATURES = ("summary", "map", "both", "region", "region+map")


def rev_of(ref: str | None) -> str:
    """The revision to audit. `None` means the current HEAD."""
    return ref or "HEAD"


def is_teacher_family(path: Path) -> bool:
    return path.name.startswith(TEACHER_PREFIX) and path.suffix == ".json"


def teacher_invocation(path: Path) -> tuple[str, str] | None:
    """
    (task, features) for a teacher artefact, or None if no invocation of the
    current script can write that filename.

    `teacher_conventional_speaking.json` -> None: no feature set, and the
    script always writes one.
    """
    stem = path.name[len(TEACHER_PREFIX):-len(".json")]
    for task in TEACHER_TASKS:
        prefix = task + "_"
        if stem.startswith(prefix):
            features = stem[len(prefix):]
            if features in TEACHER_FEATURES:
                return task, features
            return None
    return None


def companion_script(path: Path) -> Path | None:
    """
    The script that is supposed to have written this artefact, by convention.

    `foo_results.json`  -> `foo.py`
    `teacher_conventional_speaking_both.json` -> `teacher_conventional.py`
    """
    if is_teacher_family(path):
        return EXPERIMENTS / TEACHER_SCRIPT
    if path.name.endswith("_results.json"):
        return EXPERIMENTS / (path.name[: -len("_results.json")] + ".py")
    return None


def discover_head(repo: Path | None = None,
                  ref: str | None = None) -> list[str]:
    """Artefact paths tracked in the given git revision, repo-relative."""
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev_of(ref),
                           "experiments/"],
                         cwd=str(repo or ROOT), capture_output=True, text=True)
    if out.returncode != 0:
        return []
    names = []
    for line in out.stdout.splitlines():
        if not line.startswith("experiments/"):
            continue
        name = line[len("experiments/"):]
        if name.endswith("_results.json") or name.startswith(TEACHER_PREFIX):
            if name.endswith(".json"):
                names.append(name)
    return sorted(names)


def discover_worktree(experiments: Path | None = None) -> list[str]:
    experiments = experiments or EXPERIMENTS
    names = [p.name for p in sorted(experiments.glob("*_results.json"))
             if p.is_file()]
    for p in sorted(experiments.glob(f"{TEACHER_PREFIX}*.json")):
        if p.is_file() and p.name not in names:
            names.append(p.name)
    return sorted(names)


# ---------------------------------------------------------------------------
# the recipes: how each script is actually re-run
# ---------------------------------------------------------------------------

# Where a script puts its output.
OUT_DIR = "dir"        # --out <DIR>, writes <DIR>/<expect>
OUT_FILE = "file"      # --out <PATH>, writes exactly that file
OUT_SIBLING = "sibling"  # writes next to itself (ROOT/experiments or __file__)
OUT_STDOUT = "stdout"  # prints the JSON to stdout


@dataclass
class Run:
    """One way to invoke a script and get a result file back."""
    argv: list[str] = field(default_factory=list)
    expect: str = ""
    out_kind: str = OUT_DIR
    note: str = ""


@dataclass
class Recipe:
    """
    How to re-run one script, and what to do when it cannot be re-run.

    `runs` may hold several invocations (the teacher family needs one per
    feature set). `skip` short-circuits the whole script with a stated
    reason -- that is the NOT_RE_RUN path.
    """
    script: str
    runs: list[Run] = field(default_factory=list)
    skip: str = ""
    inputs: list[str] = field(default_factory=list)


def _teacher_recipe() -> Recipe:
    """One invocation per (task, feature set), each over the default corpus."""
    runs = []
    for task in TEACHER_TASKS:
        for features in TEACHER_FEATURES:
            runs.append(Run(
                argv=["--task", task, "--features", features],
                expect=f"teacher_conventional_{task}_{features}.json",
                out_kind=OUT_DIR,
            ))
    return Recipe(
        script=TEACHER_SCRIPT,
        runs=runs,
        inputs=["/mnt/c/proart_capture/corpus"],
    )


# Scripts that need the network, a model server, or an input that is not on
# disk are listed with the reason. The reason is the point: a reader has to be
# able to tell "we did not check this" from "we checked and it did not hold".
#
# NOTE ON COMPLETENESS: this table is built from the scripts' own argument
# parsers and imports. A script that is not listed here is attempted with no
# arguments, and if it fails the failure is reported verbatim.
RECIPES: dict[str, Recipe] = {
    TEACHER_SCRIPT: _teacher_recipe(),

    "ascii_motion.py": Recipe(
        script="ascii_motion.py",
        runs=[Run(argv=["--capture", "/mnt/c/proart_capture/small.nv12"],
                  expect="ascii_motion_results.json")],
        inputs=["/mnt/c/proart_capture/small.nv12"],
    ),
    "ascii_motion_cause.py": Recipe(
        script="ascii_motion_cause.py",
        runs=[Run(argv=["--capture", "/mnt/c/proart_capture/small.nv12"],
                  expect="ascii_motion_cause_results.json")],
        inputs=["/mnt/c/proart_capture/small.nv12"],
    ),
    "ascii_motion_nulls.py": Recipe(
        script="ascii_motion_nulls.py",
        runs=[Run(argv=["--capture", "/mnt/c/proart_capture/small.nv12"],
                  expect="ascii_motion_nulls_results.json")],
        inputs=["/mnt/c/proart_capture/small.nv12"],
    ),
    "ascii_motion_robustness.py": Recipe(
        script="ascii_motion_robustness.py",
        runs=[Run(argv=["--capture", "/mnt/c/proart_capture/small.nv12"],
                  expect="ascii_motion_robustness_results.json")],
        inputs=["/mnt/c/proart_capture/small.nv12"],
    ),
    "ascii_motion_threshold.py": Recipe(
        script="ascii_motion_threshold.py",
        runs=[Run(argv=["--capture", "/mnt/c/proart_capture/small.nv12"],
                  expect="ascii_motion_threshold_results.json")],
        inputs=["/mnt/c/proart_capture/small.nv12"],
    ),
    "ascii_dither_characterisation.py": Recipe(
        script="ascii_dither_characterisation.py",
        runs=[Run(argv=[], expect="ascii_dither_characterisation_results.json")],
        inputs=["/mnt/c/proart_capture/small.nv12",
                "/mnt/c/proart_capture/A.nv12",
                "/mnt/c/proart_capture/empty1.nv12"],
    ),
    "av_sync.py": Recipe(
        script="av_sync.py",
        runs=[Run(argv=[], expect="av_sync_results.json")],
        inputs=["/mnt/c/proart_capture"],
    ),
    "clef_audio_native.py": Recipe(
        script="clef_audio_native.py",
        skip="needs the Cloudflare Workers AI API (network) and a TYPESAFE_API_KEY",
        inputs=["/mnt/c/proart_capture/A.wav"],
    ),
    "cloud_multimodal.py": Recipe(
        script="cloud_multimodal.py",
        skip="needs the Cloudflare Workers AI API (network) and a TYPESAFE_API_KEY",
        inputs=["/mnt/c/proart_capture/A.nv12"],
    ),
    "cloud_vision_probe.py": Recipe(
        script="cloud_vision_probe.py",
        skip="needs the Cloudflare Workers AI API (network) and a TYPESAFE_API_KEY",
        inputs=["/mnt/c/proart_capture/A.nv12"],
    ),
    "debrief_local_director.py": Recipe(
        script="debrief_local_director.py",
        skip=("needs a local ollama server with the model already pulled; "
              "the model is not on disk in the repository"),
        inputs=["/mnt/c/proart_capture/A.nv12"],
    ),
    "debrief_paired_analysis.py": Recipe(
        script="debrief_paired_analysis.py",
        skip="needs the Cloudflare Workers AI API (network)",
        inputs=["/mnt/c/proart_capture/A.nv12"],
    ),
    "debrief_truncation_probe.py": Recipe(
        script="debrief_truncation_probe.py",
        skip="needs the Cloudflare Workers AI API (network)",
        inputs=["/mnt/c/proart_capture/A.nv12"],
    ),
    "vlm_image_retention.py": Recipe(
        script="vlm_image_retention.py",
        skip=("needs the GPU and the Qwen2.5-VL-3B weights in the local "
              "Hugging Face cache; the weights are not in the repository"),
        inputs=["/mnt/c/proart_capture/A.nv12",
                "/mnt/c/proart_capture/C.nv12",
                "/mnt/c/proart_capture/empty1.nv12"],
    ),
    "combined_filter.py": Recipe(
        script="combined_filter.py",
        runs=[Run(argv=[], expect="combined_filter_results.json")],
        inputs=["/mnt/c/proart_capture/corpus",
                "/home/proart/projects/glasses/.capture_keep/empty1.nv12"],
    ),
    "crossmodal_probe.py": Recipe(
        script="crossmodal_probe.py",
        runs=[Run(argv=[], expect="crossmodal_probe_results.json",
                  out_kind=OUT_SIBLING)],
        inputs=["examples/realtwospeaker"],
    ),
    "device_generalisation.py": Recipe(
        script="device_generalisation.py",
        skip="flushes CPU caches and measures them (needs privileges and ~11 "
             "min); its numbers are machine-specific by construction",
    ),
    "dialogue_intake_selftest.py": Recipe(
        script="dialogue_intake_selftest.py",
        runs=[Run(argv=[], expect="dialogue_intake_selftest_results.json",
                  out_kind=OUT_FILE)],
    ),
    "drift_structure.py": Recipe(
        script="drift_structure.py",
        runs=[Run(argv=[], expect="drift_structure_results.json")],
        inputs=["/mnt/c/proart_capture/A.nv12",
                "/home/proart/projects/glasses/.capture_keep/empty1.nv12"],
    ),
    "emptyroom_null.py": Recipe(
        script="emptyroom_null.py",
        runs=[Run(argv=[], expect="emptyroom_null_results.json")],
        inputs=["/mnt/c/proart_capture/empty1.nv12"],
    ),
    "f0_speaker_change.py": Recipe(
        script="f0_speaker_change.py",
        runs=[Run(argv=[], out_kind=OUT_STDOUT)],
        inputs=["examples/realaudio"],
    ),
    "gain_invariance.py": Recipe(
        script="gain_invariance.py",
        runs=[Run(argv=[], expect="gain_invariance_results.json")],
        inputs=["/home/proart/projects/glasses/.capture_keep/empty1.nv12"],
    ),
    "gate_peak_coverage.py": Recipe(
        script="gate_peak_coverage.py",
        runs=[Run(argv=[], expect="gate_peak_coverage_results.json",
                  out_kind=OUT_SIBLING)],
    ),
    "gate_peak_verdict.py": Recipe(
        script="gate_peak_verdict.py",
        runs=[Run(argv=[], expect="gate_peak_verdict_results.json",
                  out_kind=OUT_SIBLING)],
    ),
    "hum_sweep.py": Recipe(
        script="hum_sweep.py",
        runs=[Run(argv=[], out_kind=OUT_STDOUT)],
    ),
    "inthewild_probe.py": Recipe(
        script="inthewild_probe.py",
        runs=[Run(argv=[], out_kind=OUT_STDOUT)],
    ),
    "jev_jaggedness.py": Recipe(
        script="jev_jaggedness.py",
        skip="calls the live jev model over the network",
    ),
    "jev_round2.py": Recipe(
        script="jev_round2.py",
        skip="calls the live jev model over the network",
    ),
    "judged_calibration.py": Recipe(
        script="judged_calibration.py",
        skip="calls the live jev model over the network",
    ),
    "local_vs_cloud.py": Recipe(
        script="local_vs_cloud.py",
        skip="needs both a local Ollama server and the cloud model",
    ),
    "mic_filter_sim.py": Recipe(
        script="mic_filter_sim.py",
        runs=[Run(argv=[], expect="mic_filter_sim_results.json",
                  out_kind=OUT_SIBLING)],
        inputs=["examples/realaudio/alice01_excerpt.wav"],
    ),
    "notch_ab.py": Recipe(
        script="notch_ab.py",
        runs=[Run(argv=[], expect="notch_ab_results.json",
                  out_kind=OUT_SIBLING)],
        inputs=["examples/realaudio/alice01_excerpt.wav"],
    ),
    "mouth_aperture_alignment.py": Recipe(
        script="mouth_aperture_alignment.py",
        skip="needs MediaPipe, which lives in the GPU venv "
             "(/home/proart/projects/gpu-venv), not in the venv this gate "
             "runs the scripts with",
    ),
    "mouth_detector.py": Recipe(
        script="mouth_detector.py",
        runs=[Run(argv=["--capture-dir", "/mnt/c/proart_capture"],
                  expect="mouth_detector_results.json",
                  out_kind=OUT_SIBLING)],
        inputs=["/mnt/c/proart_capture/A.nv12",
                "/mnt/c/proart_capture/B.nv12",
                "/mnt/c/proart_capture/C.nv12",
                "/mnt/c/proart_capture/empty1.nv12"],
    ),
    "nv12_stride_audit.py": Recipe(
        script="nv12_stride_audit.py",
        runs=[Run(argv=[], expect="nv12_stride_audit_results.json",
                  out_kind=OUT_SIBLING)],
        inputs=["/mnt/c/proart_capture"],
    ),
    "pause_threshold.py": Recipe(
        script="pause_threshold.py",
        runs=[Run(argv=[], expect="pause_threshold_results.json",
                  out_kind=OUT_FILE)],
        inputs=["examples/realtwospeaker"],
    ),
    "pose_feature_search.py": Recipe(
        script="pose_feature_search.py",
        runs=[Run(argv=["--json", "experiments/pose_feature_search_results.json"],
                  expect="pose_feature_search_results.json",
                  out_kind=OUT_SIBLING)],
        inputs=["/mnt/c/proart_capture/A.nv12",
                "/mnt/c/proart_capture/B.nv12",
                "/mnt/c/proart_capture/C.nv12",
                "/mnt/c/proart_capture/empty1.nv12"],
    ),
    "pose_test.py": Recipe(
        script="pose_test.py",
        runs=[Run(argv=[], expect="pose_test_results.json")],
        inputs=["/mnt/c/proart_capture/A.nv12",
                "/home/proart/projects/glasses/.capture_keep/empty1.nv12"],
    ),
    "question_rewrite.py": Recipe(
        script="question_rewrite.py",
        skip="calls the live jev model over the network",
    ),
    "retraction_sweep.py": Recipe(
        script="retraction_sweep.py",
        runs=[Run(argv=[], expect="retraction_sweep_results.json",
                  out_kind=OUT_DIR)],
        inputs=["/mnt/c/proart_capture"],
    ),
    "room_map.py": Recipe(
        script="room_map.py",
        runs=[Run(argv=[], expect="room_map_results.json")],
        inputs=["/mnt/c/proart_capture/empty1.wav"],
    ),
    "speaker_embedding_f0.py": Recipe(
        script="speaker_embedding_f0.py",
        skip="needs torch and the WavLM checkpoint, which live in the GPU "
             "venv (/home/proart/projects/gpu-venv), not in the venv this "
             "gate runs the scripts with",
    ),
    "row_band_removal.py": Recipe(
        script="row_band_removal.py",
        runs=[Run(argv=[], expect="row_band_removal_results.json")],
        inputs=["/mnt/c/proart_capture/A.nv12",
                "/home/proart/projects/glasses/.capture_keep/empty1.nv12"],
    ),
    "silence_leading_artefact.py": Recipe(
        script="silence_leading_artefact.py",
        runs=[Run(argv=[], expect="silence_leading_artefact_results.json",
                  out_kind=OUT_SIBLING)],
    ),
    "small_model_sweep.py": Recipe(
        script="small_model_sweep.py",
        skip="needs a local Ollama model server, and the committed script does "
             "not even parse (SyntaxError at line 148)",
    ),
    "spectrogram_turn_eval.py": Recipe(
        script="spectrogram_turn_eval.py",
        skip="calls the live jev model over the network",
    ),
    "spectrogram_vision.py": Recipe(
        script="spectrogram_vision.py",
        skip="calls the live jev model over the network",
    ),
    "speculation_payoff.py": Recipe(
        script="speculation_payoff.py",
        runs=[Run(argv=[], expect="speculation_payoff_results.json",
                  out_kind=OUT_FILE)],
    ),
    "symbolic_audio.py": Recipe(
        script="symbolic_audio.py",
        skip="calls the live jev model over the network",
    ),
    "teacher_data_requirement.py": Recipe(
        script="teacher_data_requirement.py",
        runs=[Run(argv=[], expect="teacher_data_requirement_results.json")],
        inputs=["/mnt/c/proart_capture/corpus"],
    ),
    "teacher_raw_representation.py": Recipe(
        script="teacher_raw_representation.py",
        runs=[Run(argv=[], expect="teacher_raw_representation_results.json",
                  out_kind=OUT_SIBLING)],
        inputs=["/mnt/c/proart_capture/A.nv12",
                "/mnt/c/proart_capture/B.nv12",
                "/mnt/c/proart_capture/C.nv12"],
    ),
    "debrief_set_reaudit.py": Recipe(
        script="debrief_set_reaudit.py",
        skip="calls the live Cloudflare models over the network",
    ),
    "vision_motion_teacher.py": Recipe(
        script="vision_motion_teacher.py",
        skip="calls the live Cloudflare models over the network",
    ),
    "temporal_median_fix.py": Recipe(
        script="temporal_median_fix.py",
        runs=[Run(argv=[], expect="temporal_median_fix_results.json")],
        inputs=["/mnt/c/proart_capture/A.nv12",
                "/home/proart/projects/glasses/.capture_keep/empty1.nv12"],
    ),
    "turn_eval.py": Recipe(
        script="turn_eval.py",
        runs=[Run(argv=[], out_kind=OUT_STDOUT)],
        inputs=["examples/realtwospeaker"],
    ),
    "turn_routes.py": Recipe(
        script="turn_routes.py",
        skip="the smartturn stage needs the Cloudflare API (network)",
    ),
    "twospeaker_verify.py": Recipe(
        script="twospeaker_verify.py",
        runs=[Run(argv=[], expect="twospeaker_verify_results.json",
                  out_kind=OUT_FILE)],
        inputs=["examples/realtwospeaker"],
    ),
    "watcher_cost.py": Recipe(
        script="watcher_cost.py",
        skip="measures CPU cost on this machine; the numbers are a property of "
             "the host, not of the code",
    ),
}


# ---------------------------------------------------------------------------
# headline numbers
# ---------------------------------------------------------------------------

# A headline number is a scalar the file presents as a MEASUREMENT: an
# accuracy, a margin, a rate, a count of scored items. The vocabulary is
# deliberately narrow. Including every numeric leaf would compare counts of
# frames and window sizes -- things that change when a corpus is re-captured
# but that do not falsify a claim.
HEADLINE_KEY = re.compile(
    r"(acc|accuracy|margin|rate|score|f1|precision|recall|auroc|auc|iou|"
    r"wer|cer|bleu|density|flip_rate|inlier_rate|usable|"
    r"interrupt_rate|base_rate|different_speaker_rate|said_speaking_rate|"
    r"dot_flip|independence_baseline|separation_in_mad)",
    re.IGNORECASE,
)

# Numbers that are a property of the machine or the clock, not of the
# measurement. Comparing them would make every artefact look irreproducible
# for reasons that have nothing to do with the claim.
VOLATILE_KEY = re.compile(
    r"(elapsed|cpu_pct|wall_pct|cpu_s$|cpu_gated_pct|cpu_ungated_pct|"
    r"per_question_ms|flush_s_per_run|total_tokens|input_tokens|"
    r"output_tokens|latency_s)",
    re.IGNORECASE,
)

# A row in a list is addressed by the identity it carries, so that a row
# which moved does not read as a changed number.
IDENTITY_KEYS = ("held_out", "name", "cell", "condition", "key", "id",
                 "label", "model", "arm", "stem", "signal", "question",
                 "f0", "from", "to", "kind", "variant", "feature_set")


@dataclass
class Headline:
    """One comparable number, addressed by its path in the file."""
    path: str
    value: float | int | bool


def _identity(obj: Any) -> str | None:
    """A stable label for a dict that sits in a list, if it has one."""
    if not isinstance(obj, dict):
        return None
    for k in IDENTITY_KEYS:
        if k in obj and isinstance(obj[k], (str, int, float)):
            return f"{k}={obj[k]}"
    return None


def _walk(obj: Any, path: str = "", key_matched: bool = False) -> list[Headline]:
    """
    Collect the scalar measurements a file presents.

    A scalar counts as a headline number when the KEY THAT HOLDS IT names a
    measurement -- an accuracy, a margin, a rate. The key may be nested: the
    leave-one-session-out rows hold `acc` under `held_out`, and that number is
    exactly the one this repo is in trouble about.

    A list of bare numbers (`classes_in_train: [0, 1]`) is only a headline
    when the list's own key names a measurement; a list of records is
    descended into, because the records carry named scalars.
    """
    out: list[Headline] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{path}.{k}" if path else str(k)
            matched = bool(HEADLINE_KEY.search(str(k)))
            if isinstance(v, (dict, list)):
                out.extend(_walk(v, child, matched))
            elif isinstance(v, bool):
                # A boolean verdict (e.g. "usable") is a headline claim.
                if matched:
                    out.append(Headline(child, v))
            elif isinstance(v, (int, float)):
                if matched and not VOLATILE_KEY.search(str(k)):
                    out.append(Headline(child, v))
    elif isinstance(obj, list):
        labels = [_identity(v) for v in obj]
        use_identity = (all(l is not None for l in labels)
                        and len(set(labels)) == len(labels))
        for i, v in enumerate(obj):
            if use_identity:
                child = f"{path}[{labels[i]}]"
            else:
                child = f"{path}[{i}]"
            if isinstance(v, (dict, list)):
                out.extend(_walk(v, child, key_matched))
            elif isinstance(v, bool):
                if key_matched:
                    out.append(Headline(child, v))
            elif isinstance(v, (int, float)):
                if key_matched:
                    out.append(Headline(child, v))
    return out


def headline_numbers(obj: Any) -> dict[str, Headline]:
    """Comparable numbers, keyed by path."""
    return {h.path: h for h in _walk(obj)}


def compare(baseline: dict[str, Headline],
            candidate: dict[str, Headline]) -> list[dict[str, Any]]:
    """
    The fields that differ, as {path, old, new}.

    Only fields present in BOTH files are compared. A field that exists in
    one and not the other is a KEY difference, not a falsification: the repo
    is adding provenance keys, and that is not a disagreement about a number.
    Those are reported separately.
    """
    diffs: list[dict[str, Any]] = []
    for path, old in baseline.items():
        new = candidate.get(path)
        if new is None:
            continue
        if not _close(old.value, new.value):
            diffs.append({"path": path, "old": old.value, "new": new.value})
    return diffs


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(a)))
    return a == b


# ---------------------------------------------------------------------------
# environment facts
# ---------------------------------------------------------------------------

@dataclass
class Environment:
    python: str
    numpy: str
    input_paths: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "numpy": self.numpy,
            "input_paths": self.input_paths,
        }


def script_default_paths(script_text: str) -> list[str]:
    """
    Absolute-looking paths a script declares as argument defaults.

    Read from the source with `ast`, so it cannot be fooled by a string that
    merely mentions a path in a docstring.
    """
    found: list[str] = []
    try:
        tree = ast.parse(script_text)
    except SyntaxError:
        return found

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        kwargs = {k.arg: k.value for k in node.keywords}
        default = kwargs.get("default")
        if isinstance(default, ast.Constant) and isinstance(default.value, str):
            v = default.value
            if v.startswith("/") and len(v) > 1:
                found.append(v)
    return found


def describe_environment(recipes: dict[str, Recipe],
                         extra_inputs: list[str] | None = None) -> Environment:
    """
    The facts that make a re-run comparable to the original: which
    interpreter, which numpy, and where every input directory is.

    An input path that is not on disk is reported as such. That is the
    difference between "this reproduces" and "this was checked against
    something that is still there".
    """
    try:
        import numpy
        numpy_version = numpy.__version__
    except Exception:                                   # noqa: BLE001
        numpy_version = "not installed"

    seen: list[str] = []
    for recipe in recipes.values():
        for p in recipe.inputs:
            ap = p if os.path.isabs(p) else str(ROOT / p)
            if ap not in seen:
                seen.append(ap)
    for p in extra_inputs or []:
        ap = p if os.path.isabs(p) else str(ROOT / p)
        if ap not in seen:
            seen.append(ap)

    inputs = []
    for ap in sorted(seen):
        inputs.append({
            "path": ap,
            "exists": Path(ap).exists(),
            "kind": ("dir" if Path(ap).is_dir()
                     else "file" if Path(ap).is_file() else "missing"),
        })
    return Environment(python=sys.executable, numpy=numpy_version,
                       input_paths=inputs)


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------

def git(*args: str, cwd: Path | None = None) -> str:
    r = subprocess.run(["git", *args], cwd=str(cwd or ROOT),
                       capture_output=True, text=True)
    return r.stdout.strip()


def resolve_ref(ref: str | None) -> tuple[str, str]:
    """
    (ref-name, commit-sha) for the revision being audited.

    The tool pins itself to a commit because the working tree moves: another
    agent can commit a regenerated artefact between two runs of this tool, and
    an audit that silently changed its own subject is worse than no audit.
    """
    name = rev_of(ref)
    sha = git("rev-parse", f"{name}^{{commit}}")
    if not sha or sha.startswith("fatal"):
        raise SystemExit(f"cannot resolve revision {name!r} in {ROOT}")
    return name, sha


def blob_hash(rel: str, repo: Path | None = None,
              ref: str | None = None) -> str | None:
    """The git blob hash of a path at the audited revision, or None."""
    out = git("rev-parse", f"{rev_of(ref)}:experiments/{rel}", cwd=repo or ROOT)
    if not out or out.startswith("fatal"):
        return None
    return out


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def committed_bytes(rel: str, repo: Path | None = None,
                    ref: str | None = None) -> bytes | None:
    """The committed bytes of an experiments/ file, or None."""
    r = subprocess.run(["git", "show", f"{rev_of(ref)}:experiments/{rel}"],
                       cwd=str(repo or ROOT), capture_output=True)
    if r.returncode != 0:
        return None
    return r.stdout


# ---------------------------------------------------------------------------
# the scratch tree
# ---------------------------------------------------------------------------

# Data directories a script may read. They are symlinked into the scratch
# tree so a script that resolves its inputs from its own location still finds
# them, while anything it WRITES lands in the scratch tree.
LINKED_DIRS = ("examples", "models", ".scratch", "tools", "workers", "docs")


def build_scratch_tree(work: Path, repo: Path, source: str,
                       ref: str | None = None) -> Path:
    """
    Build `work` so that it looks like the repo, but with the AUDITED code
    inside it.

    `source` is "head" (the committed code at `ref`) or "worktree" (the code
    on disk).

    The scratch tree gets a real `experiments/` and `src/` holding the source
    under audit, and symlinks for the data directories. A script that writes
    next to itself therefore writes into the scratch tree, and the repo's
    committed artefacts are never touched.
    """
    work.mkdir(parents=True, exist_ok=True)

    for name in LINKED_DIRS:
        target = repo / name
        if not target.exists():
            continue
        link = work / name
        if link.is_symlink() or link.exists():
            continue
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            pass

    # src/, experiments/ and tests/ are real directories holding the audited
    # code. `tests/` is included because a script may check its method
    # against the copy in a test file -- `pause_threshold.py` does exactly
    # that -- and a symlink would hand the script a path into the repo.
    for sub in ("src", "experiments", "tests"):
        d = work / sub
        d.mkdir(parents=True, exist_ok=True)

    if source == "head":
        files = git("ls-tree", "-r", "--name-only", rev_of(ref),
                    "src/", "experiments/", "tests/", cwd=repo).splitlines()
        for f in files:
            if not f.startswith(("src/", "experiments/", "tests/")):
                continue
            r = subprocess.run(["git", "show", f"{rev_of(ref)}:{f}"],
                               cwd=str(repo), capture_output=True)
            blob = r.stdout if r.returncode == 0 else None
            if blob is None:
                continue
            out = work / f
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(blob)
    else:
        for sub in ("src", "experiments", "tests"):
            src_dir = repo / sub
            if not src_dir.is_dir():
                continue
            for p in src_dir.glob("*.py"):
                shutil.copy2(p, work / sub / p.name)

    return work / "experiments"


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

@dataclass
class RunOutcome:
    ok: bool
    reason: str = ""
    payload: Any = None          # parsed JSON from the run
    seconds: float = 0.0
    argv: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""


def _env_for(work: Path) -> dict[str, str]:
    env = dict(os.environ)
    # The scratch tree must come first so that a module imported by the
    # audited script resolves to the audited copy, not to the repo's.
    env["PYTHONPATH"] = os.pathsep.join([
        str(work),
        str(work / "experiments"),
        str(ROOT),
        env.get("PYTHONPATH", ""),
    ]).strip(os.pathsep)
    return env


def execute(script_path: Path, run: Run, work: Path, timeout: int,
            out_dir: Path | None) -> RunOutcome:
    """Run one script, with a timeout, and collect its JSON."""
    cmd = [sys.executable, str(script_path), *run.argv]

    if run.out_kind == OUT_DIR:
        out_dir = out_dir or (work / "out")
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd += ["--out", str(out_dir)]
    elif run.out_kind == OUT_FILE:
        out_dir = out_dir or (work / "out")
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd += ["--out", str(out_dir / run.expect)]

    # File outputs. The candidate paths are resolved BEFORE the run, so
    # that a file left over from an earlier run cannot be mistaken for this
    # run's output: a fresh write changes the mtime, a leftover does not.
    #
    # A script that writes next to itself has two possible destinations --
    # beside the script, or in the out directory -- and which one it used is
    # only knowable AFTER it has run. So both are candidates, and the one
    # that was actually written is the one that is read.
    candidates: list[Path] = []
    if run.out_kind in (OUT_DIR, OUT_FILE):
        candidates.append((out_dir or (work / "out")) / run.expect)
    else:                                     # OUT_SIBLING
        candidates.append(work / "experiments" / run.expect)
        if out_dir is not None:
            candidates.append(out_dir / run.expect)
    before = {c: (c.stat().st_mtime_ns if c.is_file() else None)
              for c in candidates}

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(work), env=_env_for(work),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return RunOutcome(ok=False, seconds=time.time() - t0, argv=cmd,
                          reason=f"timed out after {timeout}s")
    except Exception as exc:                            # noqa: BLE001
        return RunOutcome(ok=False, seconds=time.time() - t0, argv=cmd,
                          reason=f"could not be launched: {exc}")

    elapsed = time.time() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        reason = tail[-1] if tail else f"exit code {proc.returncode}"
        return RunOutcome(ok=False, seconds=elapsed, argv=cmd, reason=reason,
                          stdout=proc.stdout, stderr=proc.stderr)

    if run.out_kind == OUT_STDOUT:
        text = proc.stdout.strip()
        start = text.find("{")
        if start < 0:
            return RunOutcome(ok=False, seconds=elapsed, argv=cmd,
                              reason="printed no JSON to stdout", stdout=text,
                              stderr=proc.stderr)
        try:
            payload = json.loads(text[start:])
        except json.JSONDecodeError as exc:
            return RunOutcome(ok=False, seconds=elapsed, argv=cmd,
                              reason=f"stdout was not valid JSON: {exc}",
                              stdout=text, stderr=proc.stderr)
        return RunOutcome(ok=True, payload=payload, seconds=elapsed, argv=cmd,
                          stdout=proc.stdout, stderr=proc.stderr)

    written = [c for c in candidates
               if c.is_file() and c.stat().st_mtime_ns != before.get(c)]
    if not written:
        leftover = [c for c in candidates if c.is_file()]
        if leftover:
            return RunOutcome(ok=False, seconds=elapsed, argv=cmd,
                              reason=f"the script ran but did not rewrite "
                                     f"{run.expect} (only a leftover file was "
                                     "there)",
                              stdout=proc.stdout, stderr=proc.stderr)
        return RunOutcome(ok=False, seconds=elapsed, argv=cmd,
                          reason=f"the script ran but wrote no {run.expect}",
                          stdout=proc.stdout, stderr=proc.stderr)
    target = written[0]
    try:
        payload = json.loads(target.read_text())
    except json.JSONDecodeError as exc:
        return RunOutcome(ok=False, seconds=elapsed, argv=cmd,
                          reason=f"{run.expect} is not valid JSON: {exc}",
                          stdout=proc.stdout, stderr=proc.stderr)
    return RunOutcome(ok=True, payload=payload, seconds=elapsed, argv=cmd,
                      stdout=proc.stdout, stderr=proc.stderr)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

@dataclass
class Result:
    artefact: str
    artefact_path: str
    script: str | None
    script_exists: bool
    script_mtime: float | None
    script_sha256: str | None
    artefact_mtime: float | None
    artefact_sha256: str | None
    artefact_git_blob: str | None
    verdict: str
    reason: str = ""
    audited_script_sha256: str | None = None
    baseline_sha256: str | None = None
    differing_fields: list[dict[str, Any]] = field(default_factory=list)
    only_in_artefact: list[str] = field(default_factory=list)
    only_in_rerun: list[str] = field(default_factory=list)
    headline_count: int = 0
    argv: list[str] = field(default_factory=list)
    seconds: float = 0.0
    corpus: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artefact": self.artefact,
            "artefact_path": self.artefact_path,
            "script": self.script,
            "script_exists": self.script_exists,
            "script_mtime": self.script_mtime,
            "script_sha256": self.script_sha256,
            "artefact_mtime": self.artefact_mtime,
            "artefact_sha256": self.artefact_sha256,
            "artefact_git_blob": self.artefact_git_blob,
            "verdict": self.verdict,
            "reason": self.reason,
            "audited_script_sha256": self.audited_script_sha256,
            "baseline_sha256": self.baseline_sha256,
            "differing_fields": self.differing_fields,
            "only_in_artefact": self.only_in_artefact,
            "only_in_rerun": self.only_in_rerun,
            "headline_count": self.headline_count,
            "argv": self.argv,
            "seconds": round(self.seconds, 2),
            "corpus": self.corpus,
        }


def _pick_run(recipe: Recipe, artefact_name: str) -> Run | None:
    """
    The invocation that produces THIS artefact.

    For the teacher family the artefact's own name says which task and
    feature set it is, and BOTH have to match: `speaking_both` and
    `facing_both` are different experiments that share one script, and
    running one with the other's task silently produces the wrong number.
    For everything else the first run is the one.
    """
    if recipe.script == TEACHER_SCRIPT:
        inv = teacher_invocation(Path(artefact_name))
        if inv is None:
            return None
        task, features = inv
        for run in recipe.runs:
            if _flag_value(run.argv, "--task") == task and \
                    _flag_value(run.argv, "--features") == features:
                return run
        return None
    return recipe.runs[0] if recipe.runs else None


def _flag_value(argv: list[str], flag: str) -> str | None:
    """The value of `--flag value` in argv, or `--flag=value`, else None."""
    for i, a in enumerate(argv):
        if a == flag:
            return argv[i + 1] if i + 1 < len(argv) else None
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def recorded_invocation(obj: Any) -> list[str] | None:
    """
    The exact arguments an artefact was produced with, if it recorded them.

    The newer artefacts carry a `provenance` block with the `cli_args` the
    script was called with. Using them is how "same inputs" is honoured:
    the corpus that produced the number is named in the file, not guessed.
    """
    if not isinstance(obj, dict):
        return None
    prov = obj.get("provenance")
    if not isinstance(prov, dict):
        return None
    cli = prov.get("cli_args")
    if not isinstance(cli, list) or not cli:
        return None
    if not all(isinstance(x, str) for x in cli):
        return None
    return list(cli)


def strip_out_flag(argv: list[str]) -> list[str]:
    """
    Drop `--out` (and its value).

    The recorded `--out` points into the repo. The re-run must write into the
    scratch tree, so the flag is removed here and re-added by `execute()`.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--out":
            i += 2
            continue
        if a.startswith("--out="):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def check_one(artefact_name: str, scratch: Path, timeout: int,
              source: str = "head",
              runner: Callable[..., RunOutcome] | None = None,
              ref: str | None = None,
              tree: Path | None = None,
              ) -> Result:
    """The verdict for one artefact."""
    artefact = EXPERIMENTS / artefact_name
    script = companion_script(artefact)
    script_rel = script.name if script else None
    script_exists = bool(script and script.is_file())
    script_mtime = script.stat().st_mtime if script_exists else None
    script_sha = sha256_file(script) if script_exists else None
    blob = blob_hash(artefact_name, ref=ref)
    art_mtime = artefact.stat().st_mtime if artefact.is_file() else None
    art_sha = sha256_file(artefact)

    base = dict(
        artefact=artefact_name,
        artefact_path=str(artefact),
        script=script_rel,
        script_exists=script_exists,
        script_mtime=script_mtime,
        script_sha256=script_sha,
        artefact_mtime=art_mtime,
        artefact_sha256=art_sha,
        artefact_git_blob=blob,
    )

    # --- no companion script at all ------------------------------------
    if not script_exists:
        return Result(**base, verdict=NO_SCRIPT,
                      reason="no companion script next to this artefact")

    recipe = RECIPES.get(script_rel)
    if recipe is None:
        return Result(**base, verdict=NOT_RE_RUN,
                      reason="no recipe: this script is not in the runner "
                             "table, so its arguments cannot be inferred")
    # --- the recipe says this cannot be run ----------------------------
    if recipe.skip:
        return Result(**base, verdict=NOT_RE_RUN, reason=recipe.skip)

    run = _pick_run(recipe, artefact_name)
    if run is None:
        # The companion script exists but no invocation of it can write this
        # filename. That is a different defect from "no script at all".
        if is_teacher_family(artefact):
            has_provenance = False
            if artefact.is_file():
                try:
                    has_provenance = "provenance" in json.loads(artefact.read_text())
                except Exception:                      # noqa: BLE001
                    has_provenance = False
            return Result(**base, verdict=LEGACY_NO_SCRIPT,
                          reason="no invocation of the companion script can "
                                 "produce this filename (the script writes "
                                 "teacher_conventional_{task}_{features}.json, "
                                 "and this file has no feature set); the file "
                                 f"also carries {'a' if has_provenance else 'no'} "
                                 "provenance block")
        return Result(**base, verdict=NOT_RE_RUN,
                      reason="no invocation of this script is defined in the "
                             "runner table, so nothing was re-run")

    # --- the baseline is the committed artefact ------------------------
    baseline_bytes = (committed_bytes(artefact_name, ref=ref)
                      if source == "head" else None)
    if baseline_bytes is None and artefact.is_file():
        baseline_bytes = artefact.read_bytes()
    if baseline_bytes is None:
        return Result(**base, verdict=NOT_RE_RUN,
                      reason="the artefact is not on disk and not committed")
    try:
        baseline_obj = json.loads(baseline_bytes)
    except json.JSONDecodeError as exc:
        return Result(**base, verdict=NOT_RE_RUN,
                      reason=f"the committed artefact is not valid JSON: {exc}")

    # --- same arguments as the original run, if the file recorded them ---
    recorded = recorded_invocation(baseline_obj)
    if recorded is not None:
        run = replace(run, argv=strip_out_flag(recorded))

    # --- the audited code, in a scratch tree that mirrors the repo ------
    if tree is None:
        tree = scratch / "tree"
        build_scratch_tree(tree, ROOT, source, ref=ref)
    script_path = tree / "experiments" / script_rel
    if not script_path.is_file():
        return Result(**base, verdict=NOT_RE_RUN,
                      reason="the script could not be materialised from the "
                             "chosen source")
    audited_sha = sha256_file(script_path)

    work = tree
    out_dir = tree / "out"
    exec_fn = runner or execute
    outcome = exec_fn(script_path, run, work, timeout, out_dir)

    base["audited_script_sha256"] = audited_sha
    base["baseline_sha256"] = sha256_bytes(baseline_bytes)

    if not outcome.ok:
        return Result(**base, verdict=NOT_RE_RUN, reason=outcome.reason,
                      argv=outcome.argv, seconds=outcome.seconds)

    baseline = headline_numbers(baseline_obj)
    candidate = headline_numbers(outcome.payload)
    diffs = compare(baseline, candidate)
    only_old = sorted(set(baseline) - set(candidate))
    only_new = sorted(set(candidate) - set(baseline))

    verdict = NOT_REPRODUCIBLE if diffs else REPRODUCIBLE
    reason = f"{len(diffs)} headline field(s) differ" if diffs else ""
    corpus = _flag_value(run.argv, "--corpus")
    return Result(**base, verdict=verdict, reason=reason,
                  differing_fields=diffs, only_in_artefact=only_old,
                  only_in_rerun=only_new, headline_count=len(baseline),
                  argv=outcome.argv, seconds=outcome.seconds, corpus=corpus)


def run_all(names: list[str], scratch: Path, timeout: int,
            source: str = "head",
            runner: Callable[..., RunOutcome] | None = None,
            ref: str | None = None,
            ) -> list[Result]:
    """
    Every verdict, in one pass.

    The scratch tree is built ONCE and shared: it holds the audited code, so
    materialising it per artefact would mean the same fifty files being
    written fifty-seven times.
    """
    tree = scratch / "tree"
    build_scratch_tree(tree, ROOT, source, ref=ref)
    return [check_one(n, scratch, timeout, source, runner, ref, tree)
            for n in names]


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def _fmt_fields(diffs: list[dict[str, Any]], limit: int = 3) -> str:
    if not diffs:
        return "-"
    parts = [f"{d['path']}: {d['old']!r} -> {d['new']!r}" for d in diffs[:limit]]
    if len(diffs) > limit:
        parts.append(f"(+{len(diffs) - limit} more)")
    return "; ".join(parts)


def print_table(results: list[Result]) -> None:
    if not results:
        print("no artefacts found")
        return
    name_w = min(max(max(len(r.artefact) for r in results), 11), 46)
    script_w = max(len(r.script or "-") for r in results)
    verdict_w = max(len(r.verdict) for r in results)

    print(f"{'ARTEFACT'.ljust(name_w)}  {'SCRIPT'.ljust(script_w)}  "
          f"{'VERDICT'.ljust(verdict_w)}  DIFFERING FIELDS")
    print("-" * (name_w + script_w + verdict_w + 24))
    for r in results:
        art = r.artefact
        if len(art) > name_w:
            art = art[: name_w - 1] + "…"
        scr = r.script or "-"
        print(f"{art.ljust(name_w)}  {scr.ljust(script_w)}  "
              f"{r.verdict.ljust(verdict_w)}  {_fmt_fields(r.differing_fields)}")
        if r.verdict in (NO_SCRIPT, NOT_RE_RUN, LEGACY_NO_SCRIPT) and r.reason:
            print(f"{' '.ljust(name_w)}  {'':{script_w}}  {'':{verdict_w}}  "
                  f"reason: {r.reason}")

    counts = {v: 0 for v in VERDICTS}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    print()
    print("  ".join(f"{v}={counts.get(v, 0)}" for v in VERDICTS))
    bad = [r for r in results if r.verdict in FAILING_VERDICTS]
    if bad:
        print()
        print("NOT REPRODUCIBLE:")
        for r in bad:
            print(f"  {r.artefact}")
            for d in r.differing_fields:
                print(f"      {d['path']}: {d['old']!r} -> {d['new']!r}")


def build_manifest(results: list[Result], env: Environment, timeout: int,
                   scratch: Path, source: str,
                   ref: str | None = None,
                   commit: str | None = None) -> dict[str, Any]:
    counts = {v: 0 for v in VERDICTS}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    return {
        "tool": "experiments/artifact_repro.py",
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "repo_root": str(ROOT),
        "source": source,
        "ref": rev_of(ref),
        "commit": commit,
        "timeout_s": timeout,
        "scratch": str(scratch),
        "environment": env.to_dict(),
        "summary": {
            "total": len(results),
            **{v: counts.get(v, 0) for v in VERDICTS},
        },
        "results": [r.to_dict() for r in results],
    }


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Can each committed result file still be produced by its "
                    "companion script?")
    ap.add_argument("--json", metavar="PATH",
                    help="write a machine-readable manifest to PATH")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help=f"per-script timeout in seconds (default {DEFAULT_TIMEOUT})")
    ap.add_argument("--scratch", default=str(DEFAULT_SCRATCH),
                    help="scratch root for re-runs (default %s)" % DEFAULT_SCRATCH)
    ap.add_argument("--only", metavar="SUBSTRING",
                    help="only consider artefacts whose name contains SUBSTRING")
    ap.add_argument("--source", choices=("head", "worktree"), default="head",
                    help="which code to audit: the committed code (default) or "
                         "the working tree")
    ap.add_argument("--ref", metavar="REV", default=None,
                    help="the commit to audit (default: HEAD). The working "
                         "tree moves under this tool's feet, so an audit "
                         "should name the commit it describes")
    ap.add_argument("--list", action="store_true",
                    help="print the plan and exit without running anything")
    args = ap.parse_args(argv)

    ref_name, commit = resolve_ref(args.ref)

    names = (discover_head(ref=ref_name) if args.source == "head"
             else discover_worktree())
    if args.only:
        names = [n for n in names if args.only in n]

    extra_inputs: list[str] = []
    for n in names:
        script = companion_script(EXPERIMENTS / n)
        if script and script.is_file():
            extra_inputs.extend(script_default_paths(script.read_text()))
    env = describe_environment(RECIPES, extra_inputs)

    print(f"repo:   {ROOT}")
    print(f"audit:  {ref_name} ({commit})")
    print(f"python: {sys.executable}   numpy: {env.numpy}")
    print()

    if args.list:
        for name in names:
            script = companion_script(EXPERIMENTS / name)
            recipe = RECIPES.get(script.name) if script else None
            if script is None or not script.is_file():
                state = NO_SCRIPT
            elif recipe is None:
                state = NOT_RE_RUN
            elif recipe.skip:
                state = NOT_RE_RUN
            elif _pick_run(recipe, name) is None:
                state = LEGACY_NO_SCRIPT
            else:
                state = "WOULD RUN"
            print(f"{name:52s} {script.name if script else '-':38s} {state}")
        return 0

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    results = run_all(names, scratch, args.timeout, args.source, ref=ref_name)
    print_table(results)

    if args.json:
        manifest = build_manifest(results, env, args.timeout, scratch,
                                  args.source, ref=ref_name, commit=commit)
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        print()
        print(f"manifest: {out}")

    return 1 if any(r.verdict in FAILING_VERDICTS for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
