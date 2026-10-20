"""
Tests for the real-dialogue intake gate.

WHY THIS FILE EXISTS.

`examples/realdialogue/raw/` is empty and `experiments/dialogue_intake.py`
has never been run against a real file. The project has already paid for
exactly that once, on the other corpus: the captures in
`docs/research/AV-SYNC.md` were wrong in three ways at once and every one of
them was invisible, because nothing had ever checked (AV correspondence
unverified, frame rate assumed rather than measured, and the achievement
check's import resolving to the wrong path with the exception swallowed).

So the intake needs a test suite that can FAIL, built the way the repo's
other suites are built: synthetic corpora in a temp directory, a gate that
must ACCEPT a good one and REJECT a bad one, and a separate suite that
attempts to show the tool cannot detect what it claims to detect.

WHAT IS PINNED HERE.

  - the synthetic generator is sound: the two channels of a pair are
    genuinely different files with genuinely different content;
  - a good corpus is not rejected (nothing but the protocol-length gate
    fires), and an EMPTY directory cannot satisfy the intake at all;
  - the intake's gate logic is actually REACHED -- a corpus that must trip a
    gate does trip it, so the suite cannot pass by the intake no-opping;
  - one test per perturbation: a missing channel, a digitally silent
    channel on each side, a very low but non-zero channel, cross-talk, no
    overlap, no hand-off, a sample-rate mismatch, an unreadable file;
  - the alignment question, measured: an inter-channel offset of D moves
    the measured hand-off gap by D, and the clap the README calls the
    alignment point recovers D exactly (and is used by NOBODY today);
  - the hand-off semantics: a canonical hand-off -- A stops, B takes the
    floor, A stays quiet -- is counted ZERO times by the intake;
  - the exit code does not reflect the verdict;
  - the intake fails LOUDLY when its `src` import cannot resolve, instead
    of quietly reporting success.

WHAT IS CHARACTERISATION, NOT ENFORCEMENT.

Several of the tests below assert that a defect is STILL PRESENT. Those
assertions are written so that they FAIL when the defect is fixed, with a
message saying the report needs updating. That is deliberate: a test that
asserts "the tool cannot do X" must be able to notice the day the tool
starts doing X, and a suite that quietly passes forever is not evidence.

COST.

Each intake run is a full pass over the corpus, and the intake is pure
Python: roughly 0.3 s of CPU per second of audio. The corpus here is three
8 s scenes, so a run is a few seconds and the file takes a couple of
minutes. Nothing here touches `examples/realdialogue/raw/`.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import dialogue_intake_selftest as st  # noqa: E402

INTAKE = ROOT / "experiments" / "dialogue_intake.py"
PY = sys.executable
SCENE_SECONDS = 8.0
TONE_RMS = 1e-5          # below the intake's own 1e-4 floor
TEST_SCENES = ("scene01_free", "scene06_interrupt", "scene08_roomtone")


# ---------------------------------------------------------------- helpers --

def scene_specs(names=TEST_SCENES) -> list:
    return [s for s in st.SCENES if s["name"] in names]


def code_strings(path: Path) -> list[str]:
    """Every string literal in a module that is not a docstring.

    A grep for a word in a file's text finds the prose as well as the code,
    and prose is where the intent is documented rather than implemented. The
    AST separates the two.
    """
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            d = ast.get_docstring(node, clean=False)
            if d is not None:
                docstrings.add(d)
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value not in docstrings]


def build(tmp_path: Path, name: str, specs=None) -> Path:
    d = tmp_path / name
    st.build_corpus(d, SCENE_SECONDS, TONE_RMS,
                    specs=specs if specs is not None else scene_specs())
    return d


def run_intake(raw: Path) -> tuple[int, dict]:
    """Run the REAL intake, as a subprocess, and parse its report."""
    proc = subprocess.run([PY, str(INTAKE), "--raw", str(raw)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.stdout.strip(), (
        f"the intake printed nothing; stderr:\n{proc.stderr[-1500:]}")
    return proc.returncode, json.loads(proc.stdout)


def pair_of(report: dict, scene: str) -> dict:
    for p in report.get("pairs", []):
        if p.get("scene") == scene:
            return p
    return {}


def read_channel(path: Path) -> np.ndarray:
    return np.array(st.wavio.read_wav(str(path)).samples)


def header_rate(path: Path) -> int:
    with wave.open(str(path), "rb") as w:
        return w.getframerate()


# ------------------------------------------------------------ the baseline --

def test_the_generator_makes_two_genuinely_different_channels(tmp_path):
    """The corpus is only evidence if its two channels are actually different.

    If the generator wrote the same signal to both files, every
    cross-talk, overlap and hand-off number below would be an artefact of
    the test data rather than a measurement of the intake.
    """
    d = build(tmp_path, "gen")
    a_path = d / "scene01_free_A.wav"
    b_path = d / "scene01_free_B.wav"

    assert a_path.read_bytes() != b_path.read_bytes(), \
        "the two channels are byte-identical"

    a = read_channel(a_path)
    b = read_channel(b_path)
    assert a.shape == b.shape
    assert not np.allclose(a, b), "the two channels have identical samples"
    assert float(np.abs(a - b).max()) > 0.01, \
        "the two channels differ by less than 1% of full scale"

    # They are not the same signal time-shifted either: the correlation of
    # their ENVELOPES must be well below 1.
    env_a = np.abs(a)
    env_b = np.abs(b)
    c = float(np.corrcoef(env_a, env_b)[0, 1])
    assert c < 0.9, f"the two channels' envelopes correlate at {c:.3f}"


def test_a_good_corpus_is_not_rejected_for_anything_but_its_length(tmp_path):
    """The gate must accept a corpus that satisfies the protocol.

    The corpus here is three scenes rather than eight, so the intake's
    protocol-LENGTH gate ("only 24s of audio; the protocol asks for ~25
    min") is expected to fire. Nothing else may.
    """
    code, report = run_intake(build(tmp_path, "good"))
    gates = report["gates"]
    non_length = [u for u in gates["unmet"] if not u.startswith("only ")]
    assert non_length == [], f"a good corpus was rejected for: {non_length}"
    assert gates["matched_pairs"] == len(TEST_SCENES)
    assert gates["total_audio_s"] > 0
    assert gates["hand_offs_seen"] >= 1, \
        "the good corpus contains a barge-back and must produce a hand-off"
    assert code == 0


def test_an_empty_directory_cannot_satisfy_the_intake(tmp_path):
    """The trivial case: nothing recorded must not read as a corpus."""
    empty = tmp_path / "empty"
    empty.mkdir()
    code, report = run_intake(empty)
    gates = report["gates"]
    assert gates["status"] != "USABLE"
    assert gates["unmet"], "an empty directory produced no unmet requirements"
    assert gates["matched_pairs"] == 0
    assert gates["hand_offs_seen"] == 0
    assert gates["total_audio_s"] == 0


def test_the_gate_logic_is_actually_reached(tmp_path):
    """A suite that passes because the tool did nothing is not evidence.

    This asserts the intake really walked the corpus: it found the pairs,
    measured their duration, and produced the per-channel statistics. If the
    intake silently no-opped (an import that resolved to the wrong path and
    an exception that was swallowed -- the defect this project has already
    paid for once), every one of these would be zero or absent.
    """
    code, report = run_intake(build(tmp_path, "reached"))
    gates = report["gates"]
    assert gates["matched_pairs"] == len(TEST_SCENES), \
        "the intake did not find the matched pairs"
    assert gates["total_audio_s"] > 0, \
        "the intake measured no audio at all"
    assert gates["silences_seen"] > 0, \
        "the intake found no silences in a corpus full of them"
    for scene in TEST_SCENES:
        p = pair_of(report, scene)
        assert p, f"no entry for {scene}"
        assert p["A"]["duration_s"] > 0
        assert p["B"]["duration_s"] > 0
        assert p["A"]["raw_rate_hz"] == 16000
        assert p["crosstalk"] is not None
        assert p["levels"]["frames"] > 0


def test_the_intake_fails_loudly_when_its_import_cannot_resolve(tmp_path):
    """The corpus capture's third defect: a broken import that was swallowed.

    The intake builds its sys.path from `__file__`, so it resolves `src`
    correctly wherever it is run from. This copies it somewhere with no `src`
    beside it: the import must FAIL LOUDLY (non-zero exit, no report),
    not quietly produce a report with nothing in it.
    """
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    copy = lonely / "dialogue_intake.py"
    copy.write_text(INTAKE.read_text())
    proc = subprocess.run([PY, str(copy), "--raw", str(tmp_path)],
                          capture_output=True, text=True, cwd=str(tmp_path))
    assert proc.returncode != 0, \
        "the intake exited 0 with its `src` import unresolvable"
    assert "USABLE" not in proc.stdout, \
        "the intake reported a usable corpus while its import was broken"


def test_the_intake_code_never_reads_the_clap(tmp_path):
    """The README's alignment point is unused by the code that needs it.

    `examples/realdialogue/README.md` says the clap "gives an unambiguous
    alignment point across the two files without needing a shared clock".
    Nothing in the intake's CODE reads it -- the word appears only in the
    docstring, which says the intake does not attempt to align. If someone
    wires the clap in, this test fails and says so -- and
    `docs/research/DIALOGUE-INTAKE.md` needs updating, because the whole
    alignment section changes.

    (The behavioural guard is `test_the_intake_does_not_align_the_channels`
    below; this one pins the absence of the mechanism itself.)
    """
    strings = code_strings(INTAKE)
    assert not any("clap" in s.lower() for s in strings), \
        "the intake's code now mentions the clap: re-measure the alignment " \
        "and update docs/research/DIALOGUE-INTAKE.md"


def test_the_selftest_never_writes_into_the_raw_directory():
    """The synthetic corpora must not land where Casey's recording goes."""
    strings = code_strings(ROOT / "experiments" / "dialogue_intake_selftest.py")
    assert not any("realdialogue/raw" in s for s in strings), \
        "the selftest's code references the real recording directory"


# ------------------------------------------------------------ perturbations --

def test_a_missing_channel_is_reported_as_an_unmatched_pair(tmp_path):
    """One channel of a pair missing: the intake must say so."""
    d = build(tmp_path, "missing")
    (d / "scene01_free_B.wav").unlink()
    code, report = run_intake(d)
    p = pair_of(report, "scene01_free")
    assert p.get("problem") == "unmatched pair", \
        f"expected 'unmatched pair', got {p.get('problem')!r}"
    # CHARACTERISATION: this is reported but NOT gated -- the pair still
    # counts towards `matched_pairs`, and nothing in `unmet` mentions it.
    assert report["gates"]["matched_pairs"] == len(TEST_SCENES), \
        "the intake counted an unmatched pair as a matched pair"
    non_length = [u for u in report["gates"]["unmet"]
                  if not u.startswith("only ")]
    assert not any("unmatched" in u or "not matching" in u
                   for u in non_length), \
        "the intake now GATES an unmatched pair: update " \
        "docs/research/DIALOGUE-INTAKE.md"


def test_a_digitally_silent_channel_A_is_rejected(tmp_path):
    """A dead channel on side A: the one level failure that IS gated."""
    d = build(tmp_path, "silent_a")
    st.write_wav(d / "scene01_free_A.wav", np.zeros(int(SCENE_SECONDS * st.SR)))
    code, report = run_intake(d)
    p = pair_of(report, "scene01_free")
    assert p["A"]["silent"] is True
    assert p["A"]["peak"] == 0.0
    assert any("digitally silent" in u for u in report["gates"]["unmet"]), \
        "a digitally silent channel A was not gated"
    assert report["gates"]["status"] != "USABLE"


def test_a_digitally_silent_channel_B_is_reported_but_not_gated(tmp_path):
    """A dead channel on side B is reported and NOT gated.

    The intake's silent check only ever looks at channel A. That is not
    merely an oversight -- scene 7 of the protocol is SPECIFIED to have a
    silent channel B -- so a blanket "B is silent = error" rule would be
    wrong. The intake cannot tell a dead microphone from a deliberately
    unused one, and this test pins that it does not claim to.
    """
    d = build(tmp_path, "silent_b")
    st.write_wav(d / "scene01_free_B.wav", np.zeros(int(SCENE_SECONDS * st.SR)))
    code, report = run_intake(d)
    p = pair_of(report, "scene01_free")
    assert p["B"]["silent"] is True, "the intake did not report the silent B"
    assert p["A"]["silent"] is False
    non_length = [u for u in report["gates"]["unmet"]
                  if not u.startswith("only ")]
    assert not any("digitally silent" in u for u in non_length), \
        "the intake now GATES a silent channel B: update " \
        "docs/research/DIALOGUE-INTAKE.md"


def test_a_very_low_but_non_zero_channel_is_not_silently_accepted(tmp_path):
    """A channel that is quiet but alive must be distinguishable from a dead
    one -- and the intake reports no number that distinguishes them."""
    d = build(tmp_path, "low")
    b = read_channel(d / "scene01_free_B.wav")
    rms = float(np.sqrt(np.mean(b ** 2)))
    st.write_wav(d / "scene01_free_B.wav", b / rms * 2e-4)     # RMS 2e-4
    code, report = run_intake(d)
    p = pair_of(report, "scene01_free")
    assert p["B"]["silent"] is False, "a 2e-4 RMS channel was called silent"
    assert p["B"]["rms"] < 1e-3, "the reported RMS does not look low"
    assert p["B"]["peak"] > 0.0, "the channel is digitally silent, not low"
    # The defect: the report says "not silent" and gives no level verdict.
    non_length = [u for u in report["gates"]["unmet"]
                  if not u.startswith("only ")]
    assert not any("quiet" in u or "low" in u or "level" in u
                   for u in non_length), \
        "the intake now GATES a low-level channel: update " \
        "docs/research/DIALOGUE-INTAKE.md"


def test_a_very_low_channel_below_the_floor_is_not_gated_either(tmp_path):
    """Below the floor the channel reports as signal-free but is still not
    rejected, and it is indistinguishable from a dead mic in the report."""
    d = build(tmp_path, "vlow")
    b = read_channel(d / "scene01_free_B.wav")
    rms = float(np.sqrt(np.mean(b ** 2)))
    st.write_wav(d / "scene01_free_B.wav", b / rms * 5e-5)     # RMS 5e-5
    code, report = run_intake(d)
    p = pair_of(report, "scene01_free")
    assert p["B"]["silent"] is False
    assert p["B"]["signal_frame_frac"] < 0.1, \
        "a channel below the floor should report almost no signal frames"
    non_length = [u for u in report["gates"]["unmet"]
                  if not u.startswith("only ")]
    assert not any("digitally silent" in u or "quiet" in u
                   for u in non_length), \
        "the intake now GATES a sub-floor channel: update " \
        "docs/research/DIALOGUE-INTAKE.md"


def test_crosstalk_is_measured_but_not_gated(tmp_path):
    """Channel B clearly hearing channel A.

    The intake computes a cross-talk number per pair. This asserts the
    number responds to the corruption and that nothing is gated on it.
    """
    clean = build(tmp_path, "xt_clean")
    bad = build(tmp_path, "xt_bad")
    for spec in scene_specs():
        name = spec["name"]
        a = read_channel(bad / f"{name}_A.wav")
        b = read_channel(bad / f"{name}_B.wav")
        st.write_wav(bad / f"{name}_B.wav", 0.6 * a + b)

    _, clean_report = run_intake(clean)
    _, bad_report = run_intake(bad)
    clean_corrs = [p["crosstalk"]["corr"] for p in clean_report["pairs"]
                   if p["crosstalk"]["corr"] is not None]
    bad_corrs = [p["crosstalk"]["corr"] for p in bad_report["pairs"]
                 if p["crosstalk"]["corr"] is not None]
    assert bad_corrs, "no cross-talk number was produced at all"
    assert min(bad_corrs) > max(clean_corrs), \
        "the cross-talk number did not rise when B heard A"
    # CHARACTERISATION: the separation is tiny, and nothing is gated on it.
    separation = min(bad_corrs) - max(clean_corrs)
    assert separation < 0.10, \
        f"the cross-talk metric now separates cleanly ({separation:.3f}): " \
        "re-measure and update docs/research/DIALOGUE-INTAKE.md"
    non_length = [u for u in bad_report["gates"]["unmet"]
                  if not u.startswith("only ")]
    assert not any("cross" in u.lower() for u in non_length), \
        "the intake now GATES cross-talk: update " \
        "docs/research/DIALOGUE-INTAKE.md"


def test_a_corpus_with_no_overlap_is_not_rejected(tmp_path):
    """The barge-in scenes did not happen -- and the intake cannot tell.

    The corpus is re-placed so that the two speakers strictly alternate,
    except for one barge-back in scene 6, which is kept so that the
    hand-off count (and therefore the hand-off gate) does not move with it.
    """
    good = build(tmp_path, "ov_good")
    specs = [s for s in st.alternate_specs(keep_overlap=("scene06_interrupt",))
             if s["name"] in TEST_SCENES]
    alt = tmp_path / "ov_alt"
    st.build_corpus(alt, SCENE_SECONDS, TONE_RMS, specs=specs)

    _, good_report = run_intake(good)
    _, alt_report = run_intake(alt)

    def overlapping(report):
        return {p["scene"]: p["levels"]["both_loud_frac"]
                for p in report["pairs"] if p.get("levels")}

    good_ov = overlapping(good_report)
    alt_ov = overlapping(alt_report)
    assert good_ov["scene01_free"] > 0.0, "the good corpus has no overlap"
    assert alt_ov["scene01_free"] < 0.02, \
        "the alternating corpus still has overlap in scene 1"
    assert alt_ov["scene06_interrupt"] > 0.0, \
        "the kept barge-back should still overlap"
    # The hand-off count is preserved, so the corpus is NOT rejected --
    # the missing overlap changes nothing about the verdict.
    assert alt_report["gates"]["hand_offs_seen"] == \
        good_report["gates"]["hand_offs_seen"]
    assert alt_report["gates"]["status"] == good_report["gates"]["status"]
    non_length = [u for u in alt_report["gates"]["unmet"]
                  if not u.startswith("only ")]
    assert not any("overlap" in u.lower() for u in non_length), \
        "the intake now GATES a missing overlap: update " \
        "docs/research/DIALOGUE-INTAKE.md"


def test_a_corpus_with_no_handoff_is_rejected(tmp_path):
    """One speaker only: the positive class is absent and the intake says so."""
    d = build(tmp_path, "no_handoff")
    for spec in scene_specs():
        path = d / f"{spec['name']}_B.wav"
        st.write_wav(path, np.zeros(len(read_channel(path))))
    code, report = run_intake(d)
    assert report["gates"]["hand_offs_seen"] == 0
    assert any("zero hand-offs" in u for u in report["gates"]["unmet"]), \
        "a corpus with no hand-off at all was not rejected"
    assert report["gates"]["status"] != "USABLE"


def test_a_sample_rate_mismatch_is_not_detected(tmp_path):
    """Two channels of one pair recorded at different rates.

    `src.wavio.read_wav` resamples to the analysis rate and returns an
    Audio whose `sample_rate` is 16000, so the intake's `raw_rate_hz`
    field is 16000 whatever the file says. The duration survives, because
    it is computed after the resample; the rate does not.
    """
    d = build(tmp_path, "rate")
    b = read_channel(d / "scene01_free_B.wav")
    st.write_wav(d / "scene01_free_B.wav", b, sr=22050)
    assert header_rate(d / "scene01_free_B.wav") == 22050

    code, report = run_intake(d)
    p = pair_of(report, "scene01_free")
    assert p["A"]["raw_rate_hz"] == 16000
    # CHARACTERISATION: the real rate is invisible in the intake's report.
    assert p["B"]["raw_rate_hz"] == 16000, \
        "the intake now reports the file's real sample rate: re-measure " \
        "and update docs/research/DIALOGUE-INTAKE.md"
    non_length = [u for u in report["gates"]["unmet"]
                  if not u.startswith("only ")]
    assert not any("rate" in u.lower() for u in non_length), \
        "the intake now GATES a sample-rate mismatch: update " \
        "docs/research/DIALOGUE-INTAKE.md"


def test_an_unreadable_channel_is_reported_but_not_gated(tmp_path):
    """A file that is named .wav and is not one."""
    d = build(tmp_path, "unreadable")
    (d / "scene01_free_B.wav").write_bytes(b"NOT A WAV FILE" * 64)
    code, report = run_intake(d)
    p = pair_of(report, "scene01_free")
    assert p.get("problem") == "unreadable"
    non_length = [u for u in report["gates"]["unmet"]
                  if not u.startswith("only ")]
    assert not any("unreadable" in u.lower() for u in non_length), \
        "the intake now GATES an unreadable channel: update " \
        "docs/research/DIALOGUE-INTAKE.md"


def test_the_exit_code_does_not_reflect_the_verdict(tmp_path):
    """The corpus capture's third defect again: the check cannot fail the
    pipeline. `main()` returns None; nothing calls sys.exit(1)."""
    d = build(tmp_path, "exitcode")
    code, report = run_intake(d)
    assert report["gates"]["status"] == "NOT YET USABLE"
    assert code == 0, \
        "the intake now exits non-zero on a rejected corpus: update " \
        "docs/research/DIALOGUE-INTAKE.md"

    missing = tmp_path / "no_such_dir"
    proc = subprocess.run([PY, str(INTAKE), "--raw", str(missing)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 0, \
        "the intake now exits non-zero when nothing is recorded: update " \
        "docs/research/DIALOGUE-INTAKE.md"


# --------------------------------------------------------------- alignment --

def _offset_pair(tmp_path, offset_ms: float, name: str) -> tuple:
    """A pair with a KNOWN inter-channel offset, and its true gap."""
    spec = dict(name="probe", protocol_s=8, kind="dialogue", events=[
        ("A", 0.60, 2.00, 81), ("B", 2.95, 3.60, 82), ("A", 3.20, 4.20, 83),
    ])
    a, b, _ = st.build_scene(spec, 8.0, TONE_RMS)
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    st.write_wav(d / "probe_A.wav", a)
    st.write_wav(d / "probe_B.wav", st.shift(b, offset_ms))
    return d, a, b


def test_an_inter_channel_offset_moves_the_measured_gap_by_the_whole_offset(
        tmp_path):
    """The headline measurement.

    Two recorders started by hand are not synchronised. The intake does
    not align the channels, so an offset of D between the files moves the
    measured hand-off gap by D -- the whole offset, with no correction.
    """
    truth = 0.95                     # B's onset (2.95) - A's offset (2.00)
    gaps = {}
    for offset in (0, 40, 120, 300):
        d, a, b = _offset_pair(tmp_path, offset, f"off{offset}")
        aa = read_channel(d / "probe_A.wav")
        bb = read_channel(d / "probe_B.wav")
        measured = st.measured_gap_samples(aa, bb)
        assert measured, f"no gap was measurable at offset {offset} ms"
        gaps[offset] = measured[0]
        assert abs(gaps[offset] - (truth + offset / 1000.0)) < 0.01, \
            (f"at offset {offset} ms the measured gap was {gaps[offset]:.3f} s, "
             f"expected about {truth + offset / 1000.0:.3f} s")
    for offset in (40, 120, 300):
        shift = gaps[offset] - gaps[0]
        assert abs(shift - offset / 1000.0) < 0.01, \
            (f"the offset of {offset} ms moved the measured gap by "
             f"{shift * 1000:.0f} ms")


def test_the_clap_can_recover_the_offset_that_nobody_uses(tmp_path):
    """The README's alignment point works -- it is simply never read.

    The clap is the loudest transient in the first seconds of both files.
    Cross-correlating the two 1 ms energy envelopes finds the offset
    exactly, and correcting it removes the whole error.
    """
    truth = 0.95
    for offset in (40, 120, 300, 600):
        d, a, b = _offset_pair(tmp_path, offset, f"clap{offset}")
        aa = read_channel(d / "probe_A.wav")
        bb = read_channel(d / "probe_B.wav")
        det = st.clap_offset_ms(aa, bb)
        assert det["lag_ms"] == -offset, \
            (f"the clap offset was detected as {det['lag_ms']} ms, "
             f"expected {-offset} ms")
        assert det["peak_r"] > 0.3, \
            f"the clap correlation peak was only {det['peak_r']}"
        aligned = st.shift(bb, det["lag_ms"])
        gaps = st.measured_gap_samples(aa, aligned)
        assert gaps, "no gap was measurable after alignment"
        assert abs(gaps[0] - truth) < 0.01, \
            (f"after aligning by the clap the gap was {gaps[0]:.3f} s, "
             f"expected {truth:.3f} s")


def test_the_intake_does_not_align_the_channels(tmp_path):
    """Behavioural proof that no alignment is applied by the tool.

    The same pair, offset by 300 ms, changes the intake's own hand-off
    count from one to zero: the hand-off stops being counted because
    channel B has moved away from the instant the intake samples.
    """
    spec = dict(name="probe", protocol_s=8, kind="dialogue", events=[
        ("A", 0.60, 2.00, 81), ("B", 2.95, 3.60, 82), ("A", 3.20, 4.20, 83),
    ])
    a, b, _ = st.build_scene(spec, 8.0, TONE_RMS)
    d = tmp_path / "align"
    d.mkdir(parents=True, exist_ok=True)
    counts = {}
    for offset in (0, 300):
        st.write_wav(d / "probe_A.wav", a)
        st.write_wav(d / "probe_B.wav", st.shift(b, offset))
        _, report = run_intake(d)
        counts[offset] = report["gates"]["hand_offs_seen"]
    assert counts[0] >= 1, "the aligned pair should produce a hand-off"
    assert counts[300] == 0, (
        f"a 300 ms offset did not change the hand-off count ({counts}); "
        "the intake may have gained an alignment step -- re-measure and "
        "update docs/research/DIALOGUE-INTAKE.md")


def test_a_canonical_handoff_is_not_counted_by_the_intake(tmp_path):
    """The measurement the corpus exists to make, and the one the intake
    does not make.

    A canonical hand-off is: A stops speaking, B takes the floor, A stays
    quiet. The intake counts a hand-off only when A's silence ENDS while B
    is still talking -- so a canonical hand-off is counted ZERO times.
    """
    spec = dict(name="probe", protocol_s=8, kind="dialogue", events=[
        ("A", 1.00, 2.00, 101), ("B", 2.50, 3.50, 102),
    ])
    a, b, _ = st.build_scene(spec, 8.0, TONE_RMS)
    d = tmp_path / "canon"
    d.mkdir(parents=True, exist_ok=True)
    st.write_wav(d / "probe_A.wav", a)
    st.write_wav(d / "probe_B.wav", b)
    _, report = run_intake(d)
    assert report["gates"]["hand_offs_seen"] == 0, \
        ("the intake now counts a canonical hand-off: re-measure and "
         "update docs/research/DIALOGUE-INTAKE.md")

    # The barge-back shape, where A comes back while B is still talking.
    spec2 = dict(name="probe", protocol_s=8, kind="dialogue", events=[
        ("A", 1.00, 2.00, 101), ("B", 2.20, 4.00, 102), ("A", 3.00, 4.20, 103),
    ])
    a2, b2, _ = st.build_scene(spec2, 8.0, TONE_RMS)
    st.write_wav(d / "probe_A.wav", a2)
    st.write_wav(d / "probe_B.wav", b2)
    _, report2 = run_intake(d)
    assert report2["gates"]["hand_offs_seen"] >= 1, \
        "the barge-back shape should produce a hand-off"


def test_the_proposed_hand_off_rule_counts_what_the_intake_misses(tmp_path):
    """The fix proposed in docs/research/DIALOGUE-INTAKE.md, measured.

    A hand-off should be "channel B loud ANYWHERE during channel A's
    silence", not "channel B loud at the instant that silence ends". The
    proposed rule is implemented in the selftest (the intake itself is
    untouched) and run over the same protocol-strict corpus.
    """
    specs = [s for s in st.strict_specs() if s["name"] in TEST_SCENES]
    d = tmp_path / "strict_prop"
    st.build_corpus(d, SCENE_SECONDS, TONE_RMS, specs=specs)

    _, report = run_intake(d)
    assert report["gates"]["hand_offs_seen"] == 0, \
        "the intake now counts a canonical hand-off: re-measure and update " \
        "docs/research/DIALOGUE-INTAKE.md"

    proposed = 0
    for spec in specs:
        a = read_channel(d / f"{spec['name']}_A.wav")
        b = read_channel(d / f"{spec['name']}_B.wav")
        proposed += st.hand_offs_proposed_rule(a, b, skip_before_s=0.6)
    assert proposed >= 2, \
        (f"the proposed rule found only {proposed} hand-offs in a corpus "
         "full of canonical ones; the fix no longer works as described")
    assert proposed > report["gates"]["hand_offs_seen"], \
        "the proposed rule is no better than the current one"


def test_a_corpus_of_canonical_handoffs_is_rejected(tmp_path):
    """The corpus-level consequence.

    The protocol corpus, with every turn canonical, contains eight real
    hand-offs and the intake counts none of them -- so it rejects the
    corpus for "zero hand-offs".
    """
    specs = [s for s in st.strict_specs() if s["name"] in TEST_SCENES]
    d = tmp_path / "strict"
    st.build_corpus(d, SCENE_SECONDS, TONE_RMS, specs=specs)
    _, report = run_intake(d)
    assert report["gates"]["hand_offs_seen"] == 0
    assert any("zero hand-offs" in u for u in report["gates"]["unmet"]), \
        ("the intake no longer rejects a canonical-only corpus: "
         "re-measure and update docs/research/DIALOGUE-INTAKE.md")
