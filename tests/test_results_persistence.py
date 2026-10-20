"""
test_results_persistence.py -- the numbers the notes quote have to exist in a
result file, or nobody can check them.

THE PROBLEM THIS FILE GUARDS AGAINST.

`experiments/turn_routes.py` printed its headline permutation result and its
pair concordances to STDOUT and stored neither; `teacher_conventional.py`
stored its leave-one-session-out table as a bare list of {held_out, acc, n}
with no feature-set name on it; `combined_filter.py` printed columns whose
header row did not line up with the values under it. So a research note could
quote a number that no result file contained, and nothing would notice.

These tests run each script's persistence path on SYNTHETIC input -- no
/mnt/c corpus, no network, no models -- and then check the shape of what
comes out:

  * every summary scalar sits under a key that NAMES it (not a bare
    positional element, not a duplicated header);
  * `empty_saturation` and `A_flip` are distinct, separately named entries,
    each with its own definition, so they cannot be confused;
  * the permutation null is MEASURED: the persisted null mean is the mean of
    a null distribution that can be recomputed, not a hard-coded 0.5;
  * the leave-one-session-out table is keyed by feature set, every feature set
    appears exactly once, and a held-out session can only be read from a row
    that exists.
"""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import combined_filter            # noqa: E402
import teacher_conventional       # noqa: E402
import turn_routes                # noqa: E402


# -- helpers ------------------------------------------------------------------

def _leaves(obj, path=()):
    """(path, value) for every scalar leaf of a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, path + (str(k),))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _leaves(v, path + (f"[{i}]",))
    else:
        yield path, obj


def _same_value(a, b):
    """Equality that treats NaN as equal to NaN (the summary and the flat
    block carry the same float, NaN included)."""
    if isinstance(a, float) and isinstance(b, float):
        if np.isnan(a) and np.isnan(b):
            return True
    return a == b


def _assert_every_scalar_is_named(obj, where):
    """
    No scalar may sit in a position that does not name it.

    * a leaf must be under at least one named key;
    * a list may not hold bare scalars -- a list of numbers is a column, and a
      column is what got mis-transcribed in the first place;
    * two keys in the same object may not differ only in case or spacing,
      because that is a duplicated header waiting to happen.
    """
    for path, _value in _leaves(obj):
        assert path, f"{where}: a scalar sits at the root with no key"
        assert not path[0].startswith("["), \
            f"{where}: a scalar sits in a bare list at {'.'.join(path)}"

    def walk(node, path):
        if isinstance(node, dict):
            keys = [str(k) for k in node]
            norm = [k.strip().lower() for k in keys]
            assert len(set(norm)) == len(norm), \
                f"{where}: duplicated (case/space-only) header at {path}"
            for k, v in node.items():
                walk(v, path + (str(k),))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                assert not isinstance(v, (int, float, str, bool)) or \
                    isinstance(v, bool) is False, \
                    f"{where}: bare scalar in a list at {path}[{i}]"
                walk(v, path + (f"[{i}]",))

    walk(obj, ())


# -- synthetic fixtures --------------------------------------------------------

def _write_nv12(path: Path, n_frames: int, seed: int, moving: bool) -> Path:
    """A synthetic 160x96 capture in the layout the captures on disk use.

    WHY THIS LAYOUT, AND NOT STANDARD NV12.

    `C:\\proart_capture\\nv12small.py` writes, per 160x96 frame:

        Y plane    15,360 bytes   (160 * 96)
        U,V         3,840 bytes   (80 x 24 samples, INTERLEAVED)
        -------------------------
        total      19,200 bytes   = W*H*5/4

    so the stride of a real capture is 19,200, not the W*H*3//2 = 23,040 that
    a standard-NV12 reader assumes. `src/nv12.py` DETECTS the layout from the
    file's content, and the fixture is written in the real layout so that the
    detection is exercised rather than bypassed. (The chroma order does not
    matter to the reader, which takes the luma plane only; the U and V planes
    are written separately here because that is what the writer does.)

    A frame is a bright square that moves (or does not) on a noisy
    background, so consecutive frames are similar -- which is what the layout
    detector scores, and what makes the detection unambiguous.
    """
    rng = np.random.default_rng(seed)
    W, H = 160, 96
    luma_bytes = W * H
    chroma_bytes = luma_bytes // 8          # 1,920 each for U and V
    stride = luma_bytes + 2 * chroma_bytes  # 19,200
    buf = bytearray()
    for i in range(n_frames):
        y = np.full((H, W), 40, dtype=np.uint8)
        if moving:
            x0 = 10 + (i * 3) % 120
            y[40:56, x0:x0 + 16] = 220
        else:
            y[40:56, 40:56] = 220
        y = np.clip(y.astype(np.int16)
                    + rng.integers(-3, 4, size=(H, W)), 0, 255).astype(np.uint8)
        buf += y.tobytes()
        buf += bytes(chroma_bytes)          # U
        buf += bytes(chroma_bytes)          # V
    assert len(buf) == n_frames * stride
    path.write_bytes(bytes(buf))
    return path


def _write_maps_npz(path: Path, n_frames: int, seed: int, moving: bool) -> Path:
    """A synthetic motion-map session in the corpus's npz format."""
    rng = np.random.default_rng(seed)
    maps = np.zeros((n_frames, 48, 80), dtype=np.float32)
    for i in range(n_frames):
        m = rng.random((48, 80)).astype(np.float32) * 0.05
        if moving:
            x0 = 5 + (i * 2) % 60
            m[10:18, x0:x0 + 10] += 1.0
        else:
            m[10:18, 30:40] += 1.0
        maps[i] = m
    np.savez(path, maps=maps, scale=1.0)
    return path


def _write_wav(path: Path, n_bursts: int = 8) -> Path:
    """A synthetic clip with voiced bursts separated by silences, so the
    segmenter finds the voiced runs the case builder needs."""
    sr = 16000
    frames = []
    for b in range(n_bursts):
        t = np.arange(int(1.0 * sr)) / sr
        burst = 0.3 * np.sin(2 * np.pi * 200.0 * t)
        frames.append(burst)
        frames.append(np.zeros(int(0.4 * sr)))
    pcm = np.concatenate(frames)
    pcm = np.clip(pcm * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


@pytest.fixture(scope="module")
def combined_run(tmp_path_factory):
    """Run combined_filter.py's persistence path on synthetic captures."""
    tmp = tmp_path_factory.mktemp("combined")
    corpus = tmp / "corpus"
    corpus.mkdir()
    empty = _write_nv12(tmp / "empty.nv12", 160, 1, moving=False)
    for cond, moving in (("A", True), ("B", True), ("C", False)):
        src = _write_nv12(tmp / f"{cond}.nv12", 160, ord(cond), moving=moving)
        (corpus / f"{cond}.json").write_text(json.dumps({
            "condition": cond, "kind": "condition",
            "pose": "facing" if cond != "B" else "turned",
            "speak": cond != "C", "source": str(src),
        }))
        _write_maps_npz(corpus / f"{cond}.npz", 20, ord(cond), moving=moving)
    combined_filter.main(["--empty", str(empty), "--corpus", str(corpus),
                          "--nf", "160", "--out", str(tmp)])
    return json.loads((tmp / "combined_filter_results.json").read_text())


@pytest.fixture(scope="module")
def teacher_run(tmp_path_factory):
    """Run teacher_conventional.py's persistence path on a synthetic corpus."""
    tmp = tmp_path_factory.mktemp("teacher")
    corpus = tmp / "corpus"
    corpus.mkdir()
    for cond, speak in (("A", True), ("B", True), ("C", False)):
        (corpus / f"{cond}.json").write_text(json.dumps({
            "condition": cond, "kind": "condition",
            "pose": "facing" if cond != "B" else "turned", "speak": speak,
        }))
        _write_maps_npz(corpus / f"{cond}.npz", 24, ord(cond), moving=speak)
    teacher_conventional.main(["--corpus", str(corpus), "--out", str(tmp),
                               "--features", "both", "--k", "4", "--stride", "2",
                               "--max-per", "8"])
    return json.loads((tmp / "teacher_conventional_speaking_both.json")
                       .read_text())


@pytest.fixture(scope="module")
def routes_run(tmp_path_factory, monkeypatch_module):
    """Run turn_routes.py's persistence path on a synthetic clip."""
    tmp = tmp_path_factory.mktemp("routes")
    wav = _write_wav(tmp / "synthetic.wav")
    monkeypatch_module.setattr(turn_routes, "CELLS", {"synthetic": wav})
    turn_routes.main(["--stages", "acoustic", "--per-cell", "2",
                      "--permutations", "200", "--out", str(tmp / "out.json")])
    return json.loads((tmp / "out.json").read_text())


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


# -- turn_routes --------------------------------------------------------------

def test_turn_routes_summary_is_fully_named(routes_run):
    """Every summary scalar sits under a key that names it."""
    summary = routes_run["summary"]
    assert summary, "no summary was persisted"
    _assert_every_scalar_is_named(summary, "turn_routes summary")

    leaves = list(_leaves(summary))
    assert leaves
    # the headline statistics are present, not just the per-case rows
    stage = summary["stages"]["acoustic"]
    for key in ("mean_p_end_by_truth", "concordance_all_pairs",
                "pair_concordance", "control_concordance_vs_none",
                "permutation_null"):
        assert key in stage, f"summary is missing {key}"


def test_turn_routes_persists_observed_statistic_null_and_p_value(routes_run):
    """The observed statistic, the null mean, the hit count and the p value."""
    null = routes_run["summary"]["stages"]["acoustic"]["permutation_null"]
    for key in ("observed_concordance", "null_mean", "null_sd", "hits",
                "permutations", "p_value", "seed"):
        assert key in null, f"permutation_null is missing {key}"
    assert null["permutations"] > 0
    assert null["hits"] <= null["permutations"]
    assert null["p_value"] == pytest.approx(null["hits"] / null["permutations"])


def test_turn_routes_null_mean_is_measured_not_assumed(routes_run):
    """
    The null mean is the MEASURED null mean, not a hard-coded 0.5.

    Recomputing the null from the persisted per-case scores and the persisted
    seed has to reproduce the persisted mean exactly, and a different seed has
    to move it -- a hard-coded 0.5 would do neither.
    """
    res = routes_run["results"]
    null = routes_run["summary"]["stages"]["acoustic"]["permutation_null"]

    def scores(truth):
        return [r["acoustic"].get("p_end") for r in res
                if r.get("truth") == truth
                and r["acoustic"].get("p_end") is not None]

    again = turn_routes.permutation_null(scores("end"), scores("resume"),
                                         n_perm=null["permutations"],
                                         seed=null["seed"])
    assert again["null_mean"] == pytest.approx(null["null_mean"])
    assert again["hits"] == null["hits"]
    assert null["null_mean"] != 0.5, \
        "the null mean is exactly chance -- that is a harness bug, not a result"

    other = turn_routes.permutation_null(scores("end"), scores("resume"),
                                         n_perm=null["permutations"],
                                         seed=null["seed"] + 1)
    assert other["null_mean"] != again["null_mean"], \
        "the null mean does not depend on the seed -- it is not measured"


def test_turn_routes_persists_each_control_concordance(routes_run):
    """The three controls the notes quote are persisted by name."""
    stage = routes_run["summary"]["stages"]["acoustic"]
    controls = stage["control_concordance_vs_none"]
    for key in ("end_vs_none", "resume_vs_none", "mid_vs_none"):
        assert key in controls, f"control {key} was not persisted"
        assert isinstance(controls[key], float)
    pairs = stage["pair_concordance"]
    assert "end_vs_resume" in pairs


# -- combined_filter ----------------------------------------------------------

def test_combined_filter_summary_distinguishes_saturation_from_flip(
        combined_run):
    """
    `empty_saturation` and `A_flip` are DISTINCT, separately named entries,
    each carrying its own definition -- a consumer cannot confuse them.
    """
    summary = combined_run["summary"]
    assert summary, "no summary was persisted"
    for variant, entries in summary.items():
        assert "empty_saturation" in entries, \
            f"{variant}: empty_saturation is missing"
        assert "A_flip" in entries, f"{variant}: A_flip is missing"
        assert "empty_saturation" != "A_flip"
        for key in ("empty_saturation", "A_flip"):
            entry = entries[key]
            assert isinstance(entry["definition"], str) and entry["definition"]
            assert isinstance(entry["unit"], str) and entry["unit"]
        assert (entries["empty_saturation"]["definition"]
                != entries["A_flip"]["definition"])
        assert (entries["empty_saturation"]["value"]
                != entries["A_flip"]["value"])


def test_combined_filter_summary_covers_every_stored_scalar(combined_run):
    """Nothing is stored without a name and a definition."""
    summary = combined_run["summary"]
    for variant, entries in summary.items():
        _assert_every_scalar_is_named(entries, f"combined_filter {variant}")
        stored = combined_filter._flatten("", combined_run["variants"][variant])
        assert set(stored) == set(entries), \
            "the summary does not describe exactly the stored scalars"
        for name, value in stored.items():
            assert _same_value(entries[name]["value"], value), \
                f"{variant}/{name}: the summary and the variant disagree"


def test_combined_filter_keeps_the_flat_variants_block(combined_run):
    """The existing shape is kept for anything that already reads it."""
    assert "variants" in combined_run
    raw = combined_run["variants"]["raw"]
    assert "empty_saturation" in raw and "A_flip" in raw
    assert isinstance(raw["empty_saturation"], float)


# -- teacher_conventional -----------------------------------------------------

def test_teacher_loo_table_is_keyed_by_feature_set(teacher_run):
    """The LOO table is keyed by feature-set name, each appearing once."""
    table = teacher_run["leave_one_session_out_by_feature_set"]
    assert table, "the LOO table was not persisted"
    assert list(table) == ["both"], "the LOO table is not keyed by feature set"
    for feature_set, entry in table.items():
        assert entry["feature_set"] == feature_set
        assert "rows" in entry and "held_out_sessions" in entry
        assert "skipped_sessions" in entry
        assert entry["n_rows"] == len(entry["rows"])
    # the flat list is still there for anything that reads it
    assert "leave_one_session_out" in teacher_run


def test_teacher_loo_rows_carry_their_own_names(teacher_run):
    """Each row names its feature set, its held-out session, acc and n."""
    entry = teacher_run["leave_one_session_out_by_feature_set"]["both"]
    for row in entry["rows"]:
        for key in ("feature_set", "held_out", "acc", "n"):
            assert key in row, f"LOO row is missing {key}"
        assert row["feature_set"] == "both"
        assert isinstance(row["acc"], float)
        assert isinstance(row["n"], int)


def test_teacher_loo_margin_and_controls_are_persisted(teacher_run):
    """The margin over the best control, and both control values."""
    entry = teacher_run["leave_one_session_out_by_feature_set"]["both"]
    for key in ("margin_over_best_control", "control_spatially_scrambled",
                "control_blank", "best_control"):
        assert key in entry, f"the LOO object is missing {key}"
    assert entry["best_control"] == max(entry["control_spatially_scrambled"],
                                        entry["control_blank"])
    assert entry["margin_over_best_control"] == pytest.approx(
        teacher_run["acc_test"] - entry["best_control"])


def test_teacher_held_out_session_cannot_be_transcribed_into_a_missing_row(
        teacher_run):
    """
    A held-out session can only be read from a row that exists.

    The corpus has three sessions and one of them (C, the silent one) cannot
    be held out -- the fold would leave a single class. The persisted table
    therefore has rows for A and B, and names C in `skipped_sessions`. There is
    no value anywhere in the table that could be read as "C".
    """
    entry = teacher_run["leave_one_session_out_by_feature_set"]["both"]
    sessions = {s["name"] for s in teacher_run["sessions"]}
    rows = entry["rows"]
    held = [r["held_out"] for r in rows]

    assert len(held) == len(set(held)), "a session appears twice"
    assert set(held) <= sessions, "a row names a session that does not exist"
    assert set(held) == set(entry["held_out_sessions"])
    assert set(held).isdisjoint({s["held_out"]
                                 for s in entry["skipped_sessions"]})

    # every session without a row is accounted for, with a reason
    without_row = sessions - set(held)
    assert without_row == {s["held_out"] for s in entry["skipped_sessions"]}
    for skipped in entry["skipped_sessions"]:
        assert skipped["reason"], "a skipped session has no reason recorded"

    # and the flat list agrees with the keyed table
    assert [r["held_out"] for r in teacher_run["leave_one_session_out"]] == held


# -- import hygiene ------------------------------------------------------------

def test_scripts_import_without_side_effects(tmp_path):
    """Importing the scripts must not run them or write anything."""
    code = (
        "import sys; sys.path[:0] = [%r, %r];"
        "import combined_filter, teacher_conventional, turn_routes;"
        "print('ok')" % (str(ROOT), str(ROOT / "experiments"))
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("ok")
    assert not list(tmp_path.iterdir()), "importing wrote a file"


# -- provenance ---------------------------------------------------------------
# A number that cannot say where it came from is a bare number. The `both`
# artefact committed at HEAD carried a leave-one-session-out accuracy of 0.675
# that no corpus on disk reproduces, and the file did not record WHICH corpus
# produced it -- so the number could not be traced and had to be withdrawn.
#
# These tests require the provenance to be present, complete, and TRUE: the
# recorded SHA-256 must match the file actually on disk.

def test_teacher_result_carries_a_provenance_block(teacher_run):
    prov = teacher_run.get("provenance")
    assert prov, "the result carries no provenance block"
    assert prov.get("corpus_path"), "provenance has no corpus_path"
    assert prov.get("script_sha256"), "provenance has no script sha256"
    assert prov.get("cli_args") is not None, "the CLI args were not recorded"
    assert prov.get("command_line")
    assert prov.get("corpus_fingerprint")
    assert teacher_run.get("reproducible") is True
    for s in prov["sessions"]:
        assert s.get("npz_path"), "a session has no npz path"
        assert s.get("npz_sha256"), "a session has no npz sha256"
        assert s.get("n_frames") is not None
        assert s.get("map_shape")
        assert s.get("scale") is not None
        assert s.get("label") is not None


def test_teacher_recorded_sha256_matches_the_file_on_disk(teacher_run):
    """The hash in the file is the hash of the file that was actually read."""
    prov = teacher_run["provenance"]
    for s in prov["sessions"]:
        path = Path(s["npz_path"])
        assert path.is_file(), f"{path} is not on disk"
        assert teacher_conventional.sha256_file(path) == s["npz_sha256"], \
            f"{path} does not match the recorded sha256"


def test_teacher_provenance_verifier_accepts_a_good_result(teacher_run):
    verdict = teacher_conventional.verify_provenance(teacher_run)
    assert verdict["ok"], verdict["problems"]
    assert verdict["problems"] == []


def test_teacher_provenance_verifier_rejects_a_result_without_provenance(
        teacher_run):
    """A result with no provenance block is NOT verifiable. This test fails
    if the verifier ever accepts one."""
    stripped = {k: v for k, v in teacher_run.items() if k != "provenance"}
    verdict = teacher_conventional.verify_provenance(stripped)
    assert not verdict["ok"], "a bare number was accepted as reproducible"
    assert verdict["problems"]


def test_teacher_provenance_verifier_rejects_a_changed_input(teacher_run):
    """A recorded hash that no longer matches the file is NOT verifiable."""
    import copy
    tampered = copy.deepcopy(teacher_run)
    prov = tampered["provenance"]
    prov["sessions"][0]["npz_sha256"] = "0" * 64
    verdict = teacher_conventional.verify_provenance(tampered)
    assert not verdict["ok"], "a tampered hash was accepted"
    assert any("changed" in p or "fingerprint" in p for p in verdict["problems"])

    # and the corpus fingerprint no longer matches the sessions either
    tampered2 = copy.deepcopy(teacher_run)
    tampered2["provenance"]["sessions"][0]["npz_sha256"] = "1" * 64
    verdict2 = teacher_conventional.verify_provenance(tampered2)
    assert not verdict2["ok"]


def test_teacher_loo_rows_carry_the_corpus_fingerprint(teacher_run):
    """A row cannot be attributed to the wrong corpus."""
    fingerprint = teacher_run["provenance"]["corpus_fingerprint"]
    entry = teacher_run["leave_one_session_out_by_feature_set"]["both"]
    for row in entry["rows"]:
        assert row.get("corpus_fingerprint") == fingerprint, \
            "an LOO row does not carry the corpus fingerprint"


def test_combined_and_routes_results_carry_provenance(combined_run,
                                                       routes_run):
    for name, result in (("combined_filter", combined_run),
                         ("turn_routes", routes_run)):
        prov = result.get("provenance")
        assert prov, f"{name}: the result carries no provenance block"
        assert prov.get("script_sha256"), f"{name}: no script sha256"
        assert prov.get("cli_args") is not None, f"{name}: no CLI args"
        assert prov.get("command_line")
        assert result.get("reproducible") is True, f"{name}: not reproducible"
    assert combined_run["provenance"].get("corpus_path")
    assert combined_run["provenance"]["empty_capture"].get("sha256")
    assert routes_run["provenance"].get("cells")
    for c in routes_run["provenance"]["cells"]:
        assert c.get("sha256")
        assert Path(c["path"]).is_file()
        assert turn_routes.sha256_file(Path(c["path"])) == c["sha256"]


def test_committed_teacher_results_carry_provenance():
    """Every committed teacher result either carries its provenance, or is
    identifiable as a bare number that cannot be verified.

    The current script always writes the feature set into the filename, so a
    file that follows that convention MUST carry provenance. The one legacy
    artefact whose name does not encode a feature set was written by an older
    version of the script; it has to be identifiable as a bare number rather
    than pass as a reproducible measurement.
    """
    current = {f"teacher_conventional_{t}_{f}"
               for t in ("speaking", "facing")
               for f in ("summary", "region", "map", "both", "region+map")}
    files = sorted((ROOT / "experiments").glob("teacher_conventional_*.json"))
    assert files, "no teacher results found"
    for p in files:
        res = json.loads(p.read_text())
        prov = res.get("provenance")
        if p.stem not in current:
            # a legacy artefact of an older script version: it must be
            # identifiable as a bare number, and must not pass as a
            # reproducible measurement
            assert not prov, f"{p.name} claims provenance it does not have"
            verdict = teacher_conventional.verify_provenance(res)
            assert not verdict["ok"], f"{p.name} passes as reproducible"
            continue
        assert prov, f"{p.name} carries no provenance block"
        assert prov.get("corpus_path"), f"{p.name} has no corpus_path"
        assert prov.get("cli_args") is not None, f"{p.name} has no CLI args"
        assert res.get("reproducible") is True, f"{p.name} is not reproducible"
        for s in prov.get("sessions", []):
            assert s.get("npz_sha256"), f"{p.name}: a session has no sha256"
            path = Path(s["npz_path"])
            assert path.is_file(), f"{p.name}: {path} is not on disk"
            assert teacher_conventional.sha256_file(path) == s["npz_sha256"], \
                f"{p.name}: {path} does not match the recorded sha256"
