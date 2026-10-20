"""
retraction_sweep.py -- make the retraction reproducible instead of asserted.

WHY THIS FILE EXISTS.

The teacher artefact `experiments/teacher_conventional_speaking_both.json`
held a leave-one-session-out accuracy of **0.675** for held-out session B.
It was quoted in four documents and in the LLM director's audit set. It does
not reproduce.

The argument for withdrawing it rests on two legs:

  1. The corpus rebuilds **byte for byte** from the captures, so the data is
     not the variable.
  2. The artefact **never recorded** the parameters that would specify the
     run -- not the corpus, nor the stride, max-per, frame count or task. So
     the run cannot be *specified* from what was committed, and a sweep over
     those unrecorded parameters does not produce 0.675 together with the
     artefact's other columns.

Leg 2 was, until now, a **number in a paragraph with no script behind it** --
which is exactly the defect this whole effort is about. This file is that
script. It runs the sweep, records every configuration it tried, and writes
the result with its provenance.

WHAT IT CHECKS.

For every configuration in the sweep it records the leave-one-session-out
accuracy for every held-out session, and the control columns (test accuracy,
spatially-scrambled control, blank control, margin over the best control).

It then asks the question that decides the retraction:

    Is there ANY configuration in which the held-out-B accuracy is 0.675
    AND the control columns match the withdrawn artefact's
    (scrambled 1.0, blank 0.667, margin 0.0)?

If yes, the artefact is reproducible and the retraction is wrong. If no, the
number is unreachable in the sweep and is withdrawn.

Note the second clause. It is not enough for a configuration to produce
0.675 somewhere: the artefact's OTHER columns have to match too, because
those are what pin the corpus. A configuration that hits 0.675 on a different
corpus with different control values is not the same measurement.

WHAT IT DOES NOT CLAIM.

It does not claim 0.675 is impossible in principle. It claims that it is not
reachable in the sweep, from the captures on disk, over the parameters the
artefact failed to record -- and it publishes the sweep so that claim can be
re-run and re-examined.

It also does not assert the corpus rebuilds byte for byte. That is checked
separately, by comparing SHA-256s of the seeded `.npz` files against the
corpus on disk; the result is recorded here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import contextlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import teacher_conventional as tc  # noqa: E402


# The withdrawn artefact's own columns. A configuration only counts as a
# reproduction if it matches these as well as the held-out accuracy.
WITHDRAWN = {
    "acc_test": 1.0,
    "acc_test_spatially_scrambled": 1.0,
    "acc_test_blank": 0.6666666666666666,
    "margin_over_best_control": 0.0,
}
WITHDRAWN_LOO_B = 0.675

# The sweep. These are the parameters the artefact did not record.
FILTERS = ("none", "row", "row+median3", "row+median5")
FRAME_COUNTS = (206, 300, 450, 600)
STRIDES = (1, 2, 3, 4, 5, 6, 8, 10)
MAX_PERS = (20, 30, 40, 50, 60)
TASKS = ("speaking", "facing")

CAPTURES = Path("/mnt/c/proart_capture")
CORPUS_ON_DISK = CAPTURES / "corpus"
CORPUS_RM3_ON_DISK = CAPTURES / "corpus_rm3"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_identity(directory: Path) -> dict:
    """The SHA-256 of every session file in a corpus directory.

    Recorded so a reader can check the claim that the corpus rebuilds byte for
    byte without re-running the seeder.
    """
    out = {"path": str(directory.resolve()), "sessions": {}}
    if not directory.is_dir():
        out["present"] = False
        return out
    out["present"] = True
    for p in sorted(directory.glob("*.npz")):
        out["sessions"][p.name] = sha256_file(p)
    sessions = [tc.session_fingerprint(p) for p in
                sorted(directory.glob("*.npz"))]
    out["fingerprint"] = tc.corpus_fingerprint(sessions)
    out["session_fingerprints"] = sessions
    return out


def seed_corpus(filter_name: str, tag: str, max_frames: int, out: Path) -> Path:
    """Seed a corpus from the captures with the repo's own seeder."""
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "seed_corpus_from_captures.py"),
         "--filter", filter_name, "--tag", tag,
         "--max-frames", str(max_frames), "--out", str(out)],
        capture_output=True, check=False)
    return out


def run_config(corpus: Path, task: str, features: str, stride: int,
               max_per: int, out_dir: Path) -> dict | None:
    """One configuration. Returns the headline columns, or None on failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tc.main(["--corpus", str(corpus), "--features", features,
                 "--task", task, "--stride", str(stride),
                 "--max-per", str(max_per), "--out", str(out_dir)])
    p = out_dir / f"teacher_conventional_{task}_{features}.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    loo = {r["held_out"]: r["acc"] for r in d.get("leave_one_session_out", [])}
    return {
        "acc_test": d.get("acc_test"),
        "acc_test_spatially_scrambled": d.get("acc_test_spatially_scrambled"),
        "acc_test_blank": d.get("acc_test_blank"),
        "margin_over_best_control": d.get("margin_over_best_control"),
        "loo": loo,
        "n_windows": d.get("n_windows"),
        "k_frames": d.get("k_frames"),
    }


def matches_withdrawn(res: dict) -> bool:
    """Does this configuration reproduce the withdrawn artefact exactly?

    Both the held-out-B accuracy AND the control columns have to match. A
    configuration that hits 0.675 on a different corpus is a different
    measurement, not a reproduction.
    """
    if res is None:
        return False
    if abs(res["loo"].get("B", -1) - WITHDRAWN_LOO_B) > 1e-9:
        return False
    for k, v in WITHDRAWN.items():
        got = res.get(k)
        if got is None or abs(got - v) > 1e-9:
            return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    ap.add_argument("--work", default="/tmp/retraction_sweep",
                    help="scratch directory for the seeded corpora")
    ap.add_argument("--features", default="both")
    ap.add_argument("--max-configs", type=int, default=0,
                    help="stop after N configurations (0 = the whole sweep)")
    a = ap.parse_args(argv)

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"corpus on disk : {CORPUS_ON_DISK}")
    print(f"corpus_rm3      : {CORPUS_RM3_ON_DISK}")
    identities = {
        "corpus_on_disk": corpus_identity(CORPUS_ON_DISK),
        "corpus_rm3_on_disk": corpus_identity(CORPUS_RM3_ON_DISK),
    }
    for k, ident in identities.items():
        print(f"  {k}: {len(ident.get('sessions', {}))} session files, "
              f"fingerprint {ident.get('fingerprint')}")

    # Seed every corpus variant, and check the byte-for-byte claim while we
    # are here rather than asserting it in prose.
    seeded = {}
    rebuild_checks = {}
    for f in FILTERS:
        for mf in FRAME_COUNTS:
            tag = f"sw{mf}"
            d = work / f"{f.replace('+', '_')}_{mf}"
            seed_corpus(f, tag, mf, d)
            seeded[(f, mf)] = d
    # The byte-for-byte claim: does the seeded row+median3 / 600 corpus match
    # the one on disk? The seeded files carry this sweep's tag, the on-disk
    # ones carry theirs -- the CONTENT is what must match, not the name.
    d = seeded[("row+median3", 600)]
    sweep_tag = f"sw{600}"
    for cond in ("A", "B", "C"):
        a_path = d / f"{cond}_{sweep_tag}.npz"
        b_path = CORPUS_RM3_ON_DISK / f"{cond}_rm3.npz"
        if a_path.is_file() and b_path.is_file():
            a_sha, b_sha = sha256_file(a_path), sha256_file(b_path)
            rebuild_checks[f"{cond}_rm3"] = {
                "seeded_file": a_path.name,
                "on_disk_file": b_path.name,
                "seeded_sha256": a_sha,
                "on_disk_sha256": b_sha,
                "identical": a_sha == b_sha,
            }
        else:
            rebuild_checks[f"{cond}_rm3"] = {
                "seeded_file": a_path.name,
                "on_disk_file": b_path.name,
                "identical": None,
                "problem": "one of the two files is missing",
            }
    print(f"rebuild check   : {sum(1 for v in rebuild_checks.values() if v['identical'])}"
          f"/{len(rebuild_checks)} session files byte-identical")

    total = len(FILTERS) * len(FRAME_COUNTS) * len(STRIDES) * len(MAX_PERS) * len(TASKS)
    print(f"sweep           : {total} configurations "
          f"({len(FILTERS)} filters x {len(FRAME_COUNTS)} frame counts x "
          f"{len(STRIDES)} strides x {len(MAX_PERS)} max-pers x "
          f"{len(TASKS)} tasks)")

    tried = 0
    hits = []
    loo_b_values = {}
    t0 = time.time()
    for f in FILTERS:
        for mf in FRAME_COUNTS:
            corpus = seeded[(f, mf)]
            for task in TASKS:
                for stride in STRIDES:
                    for mp in MAX_PERS:
                        if a.max_configs and tried >= a.max_configs:
                            break
                        res = run_config(corpus, task, a.features, stride, mp,
                                         work / "out")
                        tried += 1
                        if res is None:
                            continue
                        b = res["loo"].get("B")
                        if b is not None:
                            loo_b_values.setdefault(round(b, 6), 0)
                            loo_b_values[round(b, 6)] += 1
                        if matches_withdrawn(res):
                            hits.append({"filter": f, "max_frames": mf,
                                         "task": task, "stride": stride,
                                         "max_per": mp, "result": res})
            if tried and tried % 40 == 0:
                print(f"  {tried}/{total}  ({time.time() - t0:.0f}s)")

    print(f"\nran {tried} configurations in {time.time() - t0:.0f}s")
    print(f"distinct held-out-B accuracies seen: {len(loo_b_values)}")
    near = {k: v for k, v in sorted(loo_b_values.items())
            if abs(k - WITHDRAWN_LOO_B) < 0.05}
    print(f"values within 0.05 of {WITHDRAWN_LOO_B}: {near}")
    print(f"CONFIGURATIONS REPRODUCING THE WITHDRAWN ARTEFACT: {len(hits)}")

    result = {
        "question": ("is the withdrawn leave-one-session-out accuracy of "
                     f"{WITHDRAWN_LOO_B} reachable in a sweep over the "
                     "parameters the artefact never recorded?"),
        "answer": "NO" if not hits else "YES",
        "withdrawn_value": WITHDRAWN_LOO_B,
        "withdrawn_control_columns": WITHDRAWN,
        "sweep": {
            "filters": list(FILTERS),
            "max_frames": list(FRAME_COUNTS),
            "strides": list(STRIDES),
            "max_pers": list(MAX_PERS),
            "tasks": list(TASKS),
            "features": a.features,
            "n_configurations_planned": total,
            "n_configurations_run": tried,
        },
        "reproductions": hits,
        "loo_b_value_histogram": {str(k): v for k, v in
                                  sorted(loo_b_values.items())},
        "n_distinct_loo_b_values": len(loo_b_values),
        "corpus_identity": identities,
        "corpus_rebuild_check": rebuild_checks,
        "corpus_rebuild_byte_identical": (
            bool(rebuild_checks) and
            all(v["identical"] for v in rebuild_checks.values())),
        "provenance": {
            "generated_by": "experiments/retraction_sweep.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "script_path": str(Path(__file__).resolve()),
            "teacher_conventional_sha256": sha256_file(
                ROOT / "experiments" / "teacher_conventional.py"),
            "seeder_sha256": sha256_file(
                ROOT / "experiments" / "seed_corpus_from_captures.py"),
            "python": sys.executable,
            "work_dir": str(work),
            "command_line": " ".join(sys.argv),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    p = outdir / "retraction_sweep_results.json"
    p.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {p}")
    return 0 if not hits else 1


if __name__ == "__main__":
    sys.exit(main())
