"""
test_artifact_repro.py -- a committed number has to stay reachable.

THE PROBLEM THIS FILE GUARDS AGAINST.

`experiments/` holds about fifty committed `*_results.json` files. Each was
written by a script that sits next to it, but "the script that sits next to
it" is not the same as "the script that wrote it". Scripts get edited,
corpora get re-captured, inputs get deleted. After a while nobody can tell
whether a committed number is still reachable, and a note that quotes it
cannot be checked.

The worked example is the leave-one-session-out accuracy of 0.675 in
`teacher_conventional_speaking_both.json`: the commit that wrote it
(`7a4dff5`) cannot be reproduced by the script next to it, while the other
four feature sets in the same family reproduce exactly.

WHAT IS BEING TESTED HERE.

The gate itself is `experiments/artifact_repro.py`. It is a COMMIT GATE: it
exits 1 when a committed number has stopped being reachable, so that a
number cannot quietly rot behind a note that still quotes it. The tests
below pin the parts of that gate that a broken comparison would silently
break:

  * a re-run that agrees is REPRODUCIBLE and the gate exits 0;
  * a re-run that disagrees is NOT_REPRODUCIBLE, the gate exits 1, and the
    DIFFERING FIELD NAMES are reported by name (not just "it differs");
  * an artefact with no companion script is NO_SCRIPT, which is a fact
    about the repo, not a falsified number;
  * the tool NEVER WRITES INTO THE REPO. The fixture script deliberately
    writes its result file next to itself -- the dangerous case -- and the
    test asserts that the committed artefact's bytes and mtime are
    untouched afterwards.

The comparison logic is tested directly, with an INJECTED FAKE RUNNER, so
that a verdict can be produced without running anything: that is how the
tests stay fast, and it is also how the "this must be able to fail"
requirement is met -- if `check_one` ever returned REPRODUCIBLE for a
payload that disagrees, the classification tests would go red.

Nothing here touches the real `/mnt/c` corpus, the network, or a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import artifact_repro                              # noqa: E402
from artifact_repro import (                        # noqa: E402
    LEGACY_NO_SCRIPT,
    NO_SCRIPT,
    NOT_REPRODUCIBLE,
    NOT_RE_RUN,
    REPRODUCIBLE,
    Recipe,
    Run,
    RunOutcome,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

# The fixture script writes its result file NEXT TO ITSELF. That is the
# dangerous case: a script that writes where it lives would overwrite the
# committed artefact if the tool ever ran it inside the repo.
FIXTURE_SCRIPT = '''#!/usr/bin/env python3
"""A fixture companion script. It writes next to itself, on purpose."""
import argparse
import json
import pathlib

ACC = 0.5
MARGIN = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    payload = {
        "acc_test": ACC,
        "margin_over_best_control": MARGIN,
        "leave_one_session_out": [
            {"held_out": "A", "acc": 1.0, "n": 40},
            {"held_out": "B", "acc": 0.75, "n": 40},
        ],
    }
    here = pathlib.Path(__file__).resolve().parent
    (here / "foo_results.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
'''

# What the fixture script produces. A committed artefact that matches this
# exactly is REPRODUCIBLE.
PRODUCED = {
    "acc_test": 0.5,
    "margin_over_best_control": 0.25,
    "leave_one_session_out": [
        {"held_out": "A", "acc": 1.0, "n": 40},
        {"held_out": "B", "acc": 0.75, "n": 40},
    ],
}


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update({
        "GIT_AUTHOR_NAME": "artifact_repro tests",
        "GIT_AUTHOR_EMAIL": "tests@example.invalid",
        "GIT_COMMITTER_NAME": "artifact_repro tests",
        "GIT_COMMITTER_EMAIL": "tests@example.invalid",
    })
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                          text=True, env=env)


def _snapshot(directory: Path) -> dict[str, tuple[int, str]]:
    """(mtime_ns, sha256) for every file in a directory, recursively."""
    out: dict[str, tuple[int, str]] = {}
    if not directory.is_dir():
        return out
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            out[str(p.relative_to(directory))] = (p.stat().st_mtime_ns, digest)
    return out


def _make_repo(root: Path, artefacts: dict[str, object]) -> Path:
    """
    A throwaway git repo shaped like this one: `experiments/` holding the
    artefacts and their companion scripts, committed once.
    """
    repo = root / "repo"
    experiments = repo / "experiments"
    experiments.mkdir(parents=True)
    (repo / "src").mkdir()

    (experiments / "foo.py").write_text(FIXTURE_SCRIPT)
    (experiments / "teacher_conventional.py").write_text(FIXTURE_SCRIPT)

    for name, payload in artefacts.items():
        (experiments / name).write_text(json.dumps(payload, indent=2))

    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    proc = _git(repo, "commit", "-q", "-m", "fixture")
    assert proc.returncode == 0, proc.stderr
    return repo


@pytest.fixture
def fixture_repo(tmp_path, monkeypatch):
    """
    A committed fixture repo, wired into the tool as THE repo.

    `artifact_repro` reads its repo root from module globals, so pointing
    those at the fixture is what makes the tool audit the fixture instead
    of the real tree.
    """
    repo = _make_repo(tmp_path, {
        "foo_results.json": PRODUCED,
        "orphan_results.json": {"acc_test": 0.1},
        "teacher_conventional_speaking.json": {"acc_test": 0.2},
    })
    monkeypatch.setattr(artifact_repro, "ROOT", repo)
    monkeypatch.setattr(artifact_repro, "EXPERIMENTS", repo / "experiments")

    recipes = dict(artifact_repro.RECIPES)
    recipes["foo.py"] = Recipe(
        script="foo.py",
        runs=[Run(argv=[], expect="foo_results.json",
                  out_kind=artifact_repro.OUT_SIBLING)],
    )
    monkeypatch.setattr(artifact_repro, "RECIPES", recipes)
    return repo


def _fake_runner(payload: object, ok: bool = True, reason: str = ""):
    """A runner that returns a canned outcome instead of running anything."""
    def runner(script_path, run, work, timeout, out_dir):
        return RunOutcome(ok=ok, payload=payload, reason=reason,
                          argv=["fake", str(script_path)])
    return runner


# ---------------------------------------------------------------------------
# the gate: exit code
# ---------------------------------------------------------------------------

def test_gate_exits_zero_when_the_rerun_agrees(fixture_repo, tmp_path):
    """
    A committed artefact that the script still reproduces is REPRODUCIBLE,
    and the gate exits 0 so it can sit in a commit hook.

    The verdict is asserted as well as the exit code: a gate that exited 0
    because nothing was measured would be useless.
    """
    manifest_path = tmp_path / "manifest.json"
    code = artifact_repro.main([
        "--only", "foo_results",
        "--scratch", str(tmp_path / "scratch"),
        "--json", str(manifest_path),
    ])
    assert code == 0, "the gate failed a re-run that agreed with the artefact"

    manifest = json.loads(manifest_path.read_text())
    verdicts = {r["artefact"]: r["verdict"] for r in manifest["results"]}
    assert verdicts["foo_results.json"] == REPRODUCIBLE, verdicts


def test_gate_exits_one_when_the_rerun_disagrees(tmp_path, monkeypatch):
    """
    A committed artefact the script no longer reproduces is NOT_REPRODUCIBLE
    and the gate exits 1.

    The committed artefact claims an accuracy of 0.9; the script next to it
    produces 0.5. That is the 0.675 case in miniature, and it has to fail
    the build.
    """
    stale = json.loads(json.dumps(PRODUCED))
    stale["acc_test"] = 0.9
    repo = _make_repo(tmp_path, {"foo_results.json": stale})
    monkeypatch.setattr(artifact_repro, "ROOT", repo)
    monkeypatch.setattr(artifact_repro, "EXPERIMENTS", repo / "experiments")
    recipes = dict(artifact_repro.RECIPES)
    recipes["foo.py"] = Recipe(
        script="foo.py",
        runs=[Run(argv=[], expect="foo_results.json",
                  out_kind=artifact_repro.OUT_SIBLING)],
    )
    monkeypatch.setattr(artifact_repro, "RECIPES", recipes)

    code = artifact_repro.main([
        "--only", "foo_results",
        "--scratch", str(tmp_path / "scratch"),
    ])
    assert code == 1, "the gate passed a committed number that no longer reproduces"


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def test_missing_companion_script_is_no_script(fixture_repo, tmp_path):
    """
    An artefact with no script next to it is NO_SCRIPT.

    This is a fact about the repository, not a falsified number: nothing was
    measured, so it must not fail the gate and must not be reported as a
    disagreement.
    """
    result = artifact_repro.check_one("orphan_results.json", tmp_path / "s",
                                      timeout=30, source="head")
    assert result.verdict == NO_SCRIPT, result.reason
    assert result.script == "orphan.py", "the expected companion path is wrong"
    assert result.script_exists is False
    assert not result.differing_fields
    assert result.verdict not in artifact_repro.FAILING_VERDICTS


def test_legacy_orphan_is_reported_as_its_own_class(fixture_repo, tmp_path):
    """
    `teacher_conventional_speaking.json` has a companion script, but no
    invocation of that script can write that filename -- the script always
    writes `teacher_conventional_{task}_{features}.json`.

    That is a different defect from "no script at all", so it gets its own
    class, and it does not fail the gate: it is a file to retire, not a
    number that was falsified.
    """
    result = artifact_repro.check_one("teacher_conventional_speaking.json",
                                      tmp_path / "s", timeout=30, source="head")
    assert result.verdict == LEGACY_NO_SCRIPT, result.reason
    assert result.verdict not in artifact_repro.FAILING_VERDICTS


def test_differing_numbers_are_reported_by_field_name(fixture_repo, tmp_path):
    """
    The verdict is NOT_REPRODUCIBLE and the DIFFERING FIELDS ARE NAMED.

    Naming the field is the whole point: "it differs" cannot be acted on,
    but `leave_one_session_out[held_out=B].acc: 0.5 -> 0.75` can. The
    field name has to be the path, because the number in question lives two
    levels down inside a list of rows.
    """
    payload = json.loads(json.dumps(PRODUCED))
    payload["acc_test"] = 0.42
    payload["leave_one_session_out"][1]["acc"] = 0.11

    result = artifact_repro.check_one(
        "foo_results.json", tmp_path / "s", timeout=30, source="head",
        runner=_fake_runner(payload),
    )

    assert result.verdict == NOT_REPRODUCIBLE, result.reason
    assert result.verdict in artifact_repro.FAILING_VERDICTS

    paths = {d["path"] for d in result.differing_fields}
    assert "acc_test" in paths, sorted(paths)
    assert "leave_one_session_out[held_out=B].acc" in paths, sorted(paths)

    by_path = {d["path"]: d for d in result.differing_fields}
    assert by_path["acc_test"]["old"] == 0.5
    assert by_path["acc_test"]["new"] == 0.42
    assert by_path["leave_one_session_out[held_out=B].acc"]["old"] == 0.75
    assert by_path["leave_one_session_out[held_out=B].acc"]["new"] == 0.11

    # the field that did NOT change is not reported
    assert "margin_over_best_control" not in paths


def test_matching_payload_is_reproducible(fixture_repo, tmp_path):
    """The other half of the classification test: agreement is REPRODUCIBLE."""
    result = artifact_repro.check_one(
        "foo_results.json", tmp_path / "s", timeout=30, source="head",
        runner=_fake_runner(json.loads(json.dumps(PRODUCED))),
    )
    assert result.verdict == REPRODUCIBLE, result.reason
    assert not result.differing_fields
    assert result.verdict not in artifact_repro.FAILING_VERDICTS


def test_a_failed_run_is_not_re_run_and_not_a_disagreement(fixture_repo, tmp_path):
    """
    A script that could not be run is NOT_RE_RUN, with the reason.

    NOT_RE_RUN means "no measurement was taken". It must never be reported
    as a disagreement, and it must never fail the gate -- otherwise an
    offline machine would look like a repository full of lies.
    """
    result = artifact_repro.check_one(
        "foo_results.json", tmp_path / "s", timeout=30, source="head",
        runner=_fake_runner(None, ok=False, reason="the corpus is not on disk"),
    )
    assert result.verdict == NOT_RE_RUN, result.reason
    assert "not on disk" in result.reason
    assert result.verdict not in artifact_repro.FAILING_VERDICTS


def test_only_shared_fields_are_compared(fixture_repo, tmp_path):
    """
    A key that exists in one file and not the other is a KEY DIFFERENCE, not
    a falsification.

    This repo is mid-way through adding provenance keys to its result
    files. A file that gained three keys is not a disagreement about a
    number, so the verdict stays REPRODUCIBLE and the extra keys are
    reported separately.
    """
    payload = json.loads(json.dumps(PRODUCED))
    payload["provenance"] = {"corpus_path": "/somewhere", "acc_probe": 0.5}
    payload["acc_test_new"] = 0.99

    result = artifact_repro.check_one(
        "foo_results.json", tmp_path / "s", timeout=30, source="head",
        runner=_fake_runner(payload),
    )
    assert result.verdict == REPRODUCIBLE, result.reason
    assert not result.differing_fields
    assert "provenance.acc_probe" in result.only_in_rerun
    assert "acc_test_new" in result.only_in_rerun


def test_rows_are_addressed_by_identity_not_by_position(fixture_repo, tmp_path):
    """
    A row that MOVED in the list is not a changed number.

    The leave-one-session-out rows are addressed by the session they hold
    out, so a file that lists B before A reads the same as one that lists A
    before B.
    """
    payload = json.loads(json.dumps(PRODUCED))
    payload["leave_one_session_out"] = list(
        reversed(payload["leave_one_session_out"]))

    result = artifact_repro.check_one(
        "foo_results.json", tmp_path / "s", timeout=30, source="head",
        runner=_fake_runner(payload),
    )
    assert result.verdict == REPRODUCIBLE, result.differing_fields
    assert not result.differing_fields


def test_volatile_numbers_are_not_headline_numbers():
    """
    A number that is a property of the machine or the clock is not a
    measurement, and including it would make every artefact look
    irreproducible for reasons that have nothing to do with the claim.
    """
    numbers = artifact_repro.headline_numbers({
        "acc_test": 0.9,
        "elapsed_s": 12.5,
        "cpu_pct": 87.0,
        "per_question_ms": 431.0,
        "total_tokens": 51234,
        "latency_s": 0.4,
    })
    assert set(numbers) == {"acc_test"}, sorted(numbers)


# ---------------------------------------------------------------------------
# the repo is never mutated
# ---------------------------------------------------------------------------

def test_the_tool_never_writes_into_the_repo(fixture_repo, tmp_path):
    """
    Running the gate must leave the repository exactly as it was.

    The fixture script writes its result file next to itself, which is the
    case that would do damage: the tool has to run the script inside a
    scratch tree, not inside the repo. The assertion is on BOTH the bytes
    and the mtime, because a tool that rewrites a file with identical
    content has still touched a committed artefact.
    """
    experiments = fixture_repo / "experiments"
    before = _snapshot(experiments)

    code = artifact_repro.main([
        "--only", "foo_results",
        "--scratch", str(tmp_path / "scratch"),
    ])
    assert code == 0

    after = _snapshot(experiments)
    assert after == before, (
        "the tool changed the repo's experiments/ directory: "
        f"{sorted(set(after) ^ set(before))}"
    )


def test_the_scratch_run_does_not_land_in_the_repo(fixture_repo, tmp_path):
    """
    The re-run's output goes to the scratch tree, not to the repo.

    The script writes `foo_results.json` next to itself; the tool has to
    have copied it somewhere else first.
    """
    scratch = tmp_path / "scratch"
    artifact_repro.main(["--only", "foo_results", "--scratch", str(scratch)])
    tree = scratch / "tree" / "experiments"
    assert (tree / "foo_results.json").is_file(), \
        "the re-run did not happen inside the scratch tree"


# ---------------------------------------------------------------------------
# the tool's own bookkeeping
# ---------------------------------------------------------------------------

def test_the_manifest_records_the_commit_it_audited(fixture_repo, tmp_path):
    """
    An audit that does not say which commit it audited is not an audit.

    The working tree of this repo moves while people work in it, so the
    commit has to be pinned into the manifest.
    """
    manifest_path = tmp_path / "manifest.json"
    artifact_repro.main([
        "--only", "foo_results",
        "--scratch", str(tmp_path / "scratch"),
        "--json", str(manifest_path),
    ])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["ref"] == "HEAD"
    assert len(manifest["commit"]) == 40
    assert manifest["environment"]["python"] == sys.executable
    assert manifest["summary"]["total"] >= 1
    assert manifest["environment"]["numpy"] != "not installed"


def test_the_artefact_blob_hash_is_recorded(fixture_repo, tmp_path):
    """The git blob hash ties the verdict to a specific committed file."""
    result = artifact_repro.check_one("foo_results.json", tmp_path / "s",
                                      timeout=30, source="head")
    assert result.artefact_git_blob, "no git blob hash was recorded"
    assert len(result.artefact_git_blob) == 40


def test_a_pinned_ref_selects_that_commit(fixture_repo, tmp_path):
    """
    `--ref` audits the named commit, not the moving working tree.

    The fixture repo has one commit, so pinning to it and pinning to HEAD
    have to agree -- and the tool has to accept the flag and report the
    commit it resolved.
    """
    ref_name, commit = artifact_repro.resolve_ref("HEAD")
    assert ref_name == "HEAD"
    assert len(commit) == 40

    names = artifact_repro.discover_head(ref="HEAD")
    assert "foo_results.json" in names
    assert "orphan_results.json" in names
    assert "teacher_conventional_speaking.json" in names


def test_the_real_repo_is_never_rewritten_by_the_tool():
    """
    The tool's own file lives in `experiments/` and must not be mistaken
    for an artefact.

    `artifact_repro.py` is a `.py` file, so it is not discovered; and the
    discovery rule is the one the repo's convention implies.
    """
    names = artifact_repro.discover_worktree()
    assert "artifact_repro.py" not in names
    assert all(n.endswith(".json") for n in names)


def test_importing_the_tool_does_not_run_anything(tmp_path):
    """Importing must not re-run fifty scripts as a side effect."""
    code = (
        "import sys; sys.path[:0] = [%r, %r];"
        "import artifact_repro;"
        "print('ok')" % (str(ROOT), str(ROOT / "experiments"))
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("ok")
    assert not list(tmp_path.iterdir()), "importing wrote a file"
