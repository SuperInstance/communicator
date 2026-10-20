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
    alignment point recovers D exactly;
  - the hand-off semantics: a canonical hand-off -- A stops, B takes the
    floor, A stays quiet -- IS counted;
  - the exit code carries the verdict;
  - the intake fails LOUDLY when its `src` import cannot resolve, instead
    of quietly reporting success.

WHAT CHANGED ON 2026-10-19, AND WHY THIS FILE WAS EDITED.

`docs/research/DIALOGUE-INTAKE.md` audited the intake and found seven
defects. The fixes landed, and this file's assertions were INVERTED, one by
one, as each defect was fixed. The inversions were deliberate and are
recorded here rather than quietly rewritten, because a reader of the old
file deserves to know that the tests once asserted the opposite.

  - the intake's code never mentioned the clap -> it now READS the clap and
    aligns the two channels on it;
  - an unmatched pair was reported but not gated -> it is gated;
  - a silent channel B was reported but not gated -> it is gated, and
    SCOPED to the scenes where the protocol has B speaking (scene 7 is
    specified to have a silent B, so a blanket rule would be wrong);
  - a low-level channel was reported but not gated -> it is gated, by the
    balance between the two channels of a pair;
  - cross-talk was measured by a statistic that separated clean from
    contaminated by 0.0067 and was not gated -> the statistic was replaced
    (it now separates by 0.6939) and it IS gated;
  - a sample-rate mismatch was invisible -> the intake now reports the
    file's REAL rate, which `src/wavio.read_wav` used to discard;
  - an unreadable channel was reported but not gated -> it is gated;
  - the exit code did not reflect the verdict -> it does now;
  - a canonical hand-off was counted ZERO times -> it is counted once;
  - a corpus of canonical hand-offs was REJECTED for having none -> it is
    accepted.

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


def non_length_unmet(report: dict) -> list:
    """The unmet requirements, minus the protocol-LENGTH one.

    Every corpus in this file is three 8 s scenes and the protocol asks for
    1230 s, so the length gate fires on all of them. It is the gate that is
    under test everywhere else, so it is filtered out here -- but it is
    never filtered out silently: `test_a_good_corpus_is_not_rejected_for_
    anything_but_its_length` asserts that it is the ONLY thing that fires.
    """
    return [u for u in report["gates"]["unmet"]
            if "the protocol's eight scenes are" not in u]


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
    protocol-LENGTH gate is expected to fire. Nothing else may.
    """
    code, report = run_intake(build(tmp_path, "good"))
    gates = report["gates"]
    assert non_length_unmet(report) == [], \
        f"a good corpus was rejected for: {non_length_unmet(report)}"
    assert gates["matched_pairs"] == len(TEST_SCENES)
    assert gates["total_audio_s"] > 0
    assert gates["hand_offs_seen"] >= 1, \
        "the good corpus contains a barge-back and must produce a hand-off"
    # The exit code carries the verdict: this corpus is rejected, for its
    # length, and says so in the exit code.
    assert code == 1, \
        f"the intake exited {code} on a corpus it reports as NOT YET USABLE"
    assert gates["status"] == "NOT YET USABLE"


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
        assert p["A"]["source_rate_hz"] == 16000
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


def test_the_intake_code_reads_the_clap(tmp_path):
    """The README's alignment point is now USED by the code that needs it.

    `examples/realdialogue/README.md` says the clap "gives an unambiguous
    alignment point across the two files without needing a shared clock".

    This assertion was INVERTED on 2026-10-19. It used to read:

        assert not any("clap" in s.lower() for s in strings)

    -- pinning the ABSENCE of the mechanism. The audit found the absence was
    the defect, so the test now pins the presence. If the clap is ever
    removed from the code again, this fails.
    """
    strings = code_strings(INTAKE)
    assert any("clap" in s.lower() for s in strings), \
        "the intake's code no longer mentions the clap: the alignment step " \
        "has gone; re-measure and update docs/research/DIALOGUE-INTAKE.md"


def test_the_selftest_never_writes_into_the_raw_directory():
    """The synthetic corpora must not land where Casey's recording goes."""
    strings = code_strings(ROOT / "experiments" / "dialogue_intake_selftest.py")
    assert not any("realdialogue/raw" in s for s in strings), \
        "the selftest's code references the real recording directory"


# ------------------------------------------------------------ perturbations --

def test_a_missing_channel_is_reported_and_gated(tmp_path):
    """One channel of a pair missing: the intake must say so AND reject it.

    The GATE half of this assertion was INVERTED on 2026-10-19: the defect
    was that the pair was reported and then ignored by the verdict.
    """
    d = build(tmp_path, "missing")
    (d / "scene01_free_B.wav").unlink()
    code, report = run_intake(d)
    p = pair_of(report, "scene01_free")
    assert p.get("problem") == "unmatched pair", \
        f"expected 'unmatched pair', got {p.get('problem')!r}"
    assert report["gates"]["matched_pairs"] == len(TEST_SCENES) - 1, \
        "the intake counted an unmatched pair as a matched pair"
    assert any("unmatched" in u for u in report["gates"]["unmet"]), \
        "an unmatched pair was not gated"
    assert code != 0


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


def test_a_digitally_silent_channel_B_is_gated_where_B_speaks(tmp_path):
    """A dead channel on side B is gated -- in the scenes where B speaks.

    The old behaviour was to report it and never gate it, on the grounds
    that scene 7 of the protocol is SPECIFIED to have a silent channel B,
    so a blanket "B is silent = error" rule would be wrong.

    The fix keeps that reasoning and scopes the rule: in scene 1 the
    protocol has B speaking, so a silent B there IS a fault. The
    companion test below covers the other half -- scene 7, where a silent B
    is correct.

    This assertion was INVERTED on 2026-10-19.
    """
    d = build(tmp_path, "silent_b")
    st.write_wav(d / "scene01_free_B.wav", np.zeros(int(SCENE_SECONDS * st.SR)))
    code, report = run_intake(d)
    p = pair_of(report, "scene01_free")
    assert p["B"]["silent"] is True, "the intake did not report the silent B"
    assert p["A"]["silent"] is False
    assert any("digitally silent" in u for u in report["gates"]["unmet"]), \
        "a silent channel B was not gated in a scene where B speaks"
    assert code != 0


def test_a_low_level_channel_is_gated(tmp_path):
    """A channel that is quiet but alive must be distinguishable from a dead
    one -- and now it is.

    The gate is the BALANCE between the two channels of a pair: a microphone
    on the table instead of at the mouth is tens of dB down, and that is a
    fault in the recording, not a stylistic choice.

    This assertion was INVERTED on 2026-10-19.
    """
    d = build(tmp_path, "low")
    b = read_channel(d / "scene01_free_B.wav")
    rms = float(np.sqrt(np.mean(b ** 2)))
    st.write_wav(d / "scene01_free_B.wav", b / rms * 2e-4)     # RMS 2e-4
    code, report = run_intake(d)
    p = pair_of(report, "scene01_free")
    assert p["B"]["silent"] is False, "a 2e-4 RMS channel was called silent"
    assert p["B"]["rms"] < 1e-3, "the reported RMS does not look low"
    assert p["B"]["peak"] > 0.0, "the channel is digitally silent, not low"
    assert any("differ by" in u for u in report["gates"]["unmet"]), \
        "a low-level channel was not gated"
    assert code != 0


def test_a_very_low_channel_below_the_floor_is_gated_too(tmp_path):
    """Below the floor the channel carries nothing, and the gate says so.

    This assertion was INVERTED on 2026-10-19.
    """
    d = build(tmp_path, "vlow")
    b = read_channel(d / "scene01_free_B.wav")
    rms = float(np.sqrt(np.mean(b ** 2)))
    st.write_wav(d / "scene01_free_B.wav", b / rms * 5e-5)     # RMS 5e-5
    code, report = run_intake(d)
    p = pair_of(report, "scene01_free")
    assert p["B"]["silent"] is False
    assert p["B"]["signal_frame_frac"] < 0.1, \
        "a channel below the floor should report almost no signal frames"
    assert non_length_unmet(report), \
        "a sub-floor channel was accepted"
    assert code != 0


def test_crosstalk_is_measured_by_a_statistic_that_separates_and_is_gated(
        tmp_path):
    """Channel B clearly hearing channel A: separated widely, and gated.

    THE STATISTIC WAS REPLACED on 2026-10-19. The old one correlated the
    MEANS of the ten loudest windows and separated clean from contaminated
    by 0.0067 -- a margin too small to gate on. The new one asks "of the
    time A is loud, how much of it is B loud too?", which separates by
    0.6939 on the same signals: clean 0.000-0.306, contaminated 1.000.
    """
    clean = build(tmp_path, "xt_clean")
    bad = build(tmp_path, "xt_bad")
    for spec in scene_specs():
        name = spec["name"]
        a = read_channel(bad / f"{name}_A.wav")
        b = read_channel(bad / f"{name}_B.wav")
        st.write_wav(bad / f"{name}_B.wav", 0.6 * a + b)

    _, clean_report = run_intake(clean)
    code, bad_report = run_intake(bad)

    def stat(report):
        return [p["crosstalk"]["b_loud_when_a_loud"] for p in report["pairs"]
                if p["crosstalk"]["b_loud_when_a_loud"] is not None]

    clean_vals, bad_vals = stat(clean_report), stat(bad_report)
    assert bad_vals, "no cross-talk number was produced at all"
    assert min(bad_vals) > max(clean_vals), \
        "the cross-talk number did not rise when B heard A"
    separation = min(bad_vals) - max(clean_vals)
    assert separation > 0.30, \
        (f"the cross-talk statistic separates by only {separation:.3f}: "
         "re-measure and update docs/research/DIALOGUE-INTAKE.md")
    assert any("hears channel A" in u for u in bad_report["gates"]["unmet"]), \
        "the intake does not GATE cross-talk"
    assert code != 0
    # And a clean corpus is not gated on it.
    assert not any("hears channel A" in u
                   for u in non_length_unmet(clean_report)), \
        "the intake gates cross-talk on a clean corpus"


def test_a_corpus_with_no_overlap_is_not_rejected(tmp_path):
    """The barge-in scenes did not happen -- and that is not a fault.

    The corpus is re-placed so that the two speakers strictly alternate,
    except for one barge-back in scene 6, which is kept so that the
    hand-off count (and therefore the hand-off gate) does not move with it.

    NOTE WHERE THE OVERLAP IS. It is in scene 6, not scene 1: scene 1's
    turns are strictly alternating even in the good corpus. The old
    segmentation reported an overlap in scene 1 anyway, because it counted
    the CLAP -- which is loud in both channels by construction and is
    40 ms, shorter than `energy.MIN_SPEECH_S`, so the new segmentation
    correctly leaves it out of the conversation and out of the overlap
    budget.
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
    assert good_ov["scene06_interrupt"] > 0.0, \
        "the good corpus has no barge-in to overlap"
    assert alt_ov["scene06_interrupt"] > 0.0, \
        "the kept barge-back should still overlap"
    assert alt_ov["scene01_free"] < 0.02, \
        "the alternating corpus still has overlap in scene 1"
    # The hand-off count is preserved, so the corpus is NOT rejected --
    # the missing overlap changes nothing about the verdict.
    assert alt_report["gates"]["hand_offs_seen"] == \
        good_report["gates"]["hand_offs_seen"]
    assert alt_report["gates"]["status"] == good_report["gates"]["status"]
    assert not any("overlap" in u.lower() for u in non_length_unmet(alt_report)), \
        "the intake GATES a missing overlap: that is a legitimate variation"


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
    assert code != 0


def test_a_sample_rate_mismatch_is_reported(tmp_path):
    """Two channels of one pair recorded at different rates.

    `src/wavio.read_wav` resamples to the analysis rate, so the file's real
    rate used to be unrecoverable -- `raw_rate_hz` was 16000 whatever the
    file said. `src/wavio.py` now carries the source rate on the Audio
    object, and the intake reports it.

    A mismatch is REPORTED and not GATED, because the protocol explicitly
    allows it: "Rate: anything 16 kHz and up."

    This assertion was INVERTED on 2026-10-19.
    """
    d = build(tmp_path, "rate")
    b = read_channel(d / "scene01_free_B.wav")
    st.write_wav(d / "scene01_free_B.wav", b, sr=22050)
    assert header_rate(d / "scene01_free_B.wav") == 22050

    code, report = run_intake(d)
    p = pair_of(report, "scene01_free")
    assert p["A"]["source_rate_hz"] == 16000
    assert p["B"]["source_rate_hz"] == 22050, \
        "the intake does not report the file's real sample rate"
    assert p["source_rates_hz"]["B"] == 22050
    assert not any("rate" in u.lower() for u in non_length_unmet(report)), \
        "the intake GATES a sample-rate mismatch: the protocol allows it"


def test_an_unreadable_channel_is_reported_and_gated(tmp_path):
    """A file that is named .wav and is not one.

    The GATE half of this assertion was INVERTED on 2026-10-19.
    """
    d = build(tmp_path, "unreadable")
    (d / "scene01_free_B.wav").write_bytes(b"NOT A WAV FILE" * 64)
    code, report = run_intake(d)
    p = pair_of(report, "scene01_free")
    assert p.get("problem", "").startswith("unreadable")
    assert any("unreadable" in u.lower() for u in report["gates"]["unmet"]), \
        "an unreadable channel was not gated"
    assert code != 0


def test_the_exit_code_carries_the_verdict(tmp_path):
    """The corpus capture's third defect again: the check must be able to
    fail the pipeline.

    This assertion was INVERTED on 2026-10-19: it used to assert that the
    exit code was 0 on a rejected corpus.
    """
    d = build(tmp_path, "exitcode")
    code, report = run_intake(d)
    assert report["gates"]["status"] == "NOT YET USABLE"
    assert code != 0, \
        "the intake exited 0 on a corpus it reports as NOT YET USABLE"

    missing = tmp_path / "no_such_dir"
    proc = subprocess.run([PY, str(INTAKE), "--raw", str(missing)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode != 0, \
        "the intake exited 0 when nothing was recorded"
    assert json.loads(proc.stdout)["gates"]["status"] == "NOTHING RECORDED YET"


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


def test_the_clap_recovers_the_offset_and_the_intake_applies_it(tmp_path):
    """The README's alignment point is now read, and used.

    The clap is the loudest transient in the first seconds of both files.
    Cross-correlating the two 1 ms energy envelopes finds the offset
    exactly, and the intake now corrects with it BEFORE it computes any
    cross-channel number.
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

        # And the INTAKE does it: the report carries the offset it found.
        _, report = run_intake(d)
        p = pair_of(report, "probe")
        assert p["alignment"]["clap_offset_ms"] == -offset, \
            ("the intake did not detect the inter-channel offset: "
             f"{p.get('alignment')}")
        assert p["alignment"]["applied"] is True


def test_the_hand_off_count_does_not_flip_with_a_300ms_offset(tmp_path):
    """The alignment step is BEHAVIOURAL, not decorative.

    Before the fix, a 300 ms offset moved the intake's hand-off count from
    one to zero: the hand-off stopped being counted because channel B had
    moved away from the instant the intake sampled.

    This assertion was INVERTED on 2026-10-19.
    """
    spec = dict(name="probe", protocol_s=8, kind="dialogue", events=[
        ("A", 0.60, 2.00, 81), ("B", 2.95, 3.60, 82), ("A", 3.20, 4.20, 83),
    ])
    a, b, _ = st.build_scene(spec, 8.0, TONE_RMS)
    d = tmp_path / "align"
    d.mkdir(parents=True, exist_ok=True)
    counts = {}
    for offset in (0, 40, 120, 300, 600):
        st.write_wav(d / "probe_A.wav", a)
        st.write_wav(d / "probe_B.wav", st.shift(b, offset))
        _, report = run_intake(d)
        counts[offset] = report["gates"]["hand_offs_seen"]
    assert counts[0] >= 1, "the aligned pair should produce a hand-off"
    for offset, count in counts.items():
        assert count == counts[0], \
            (f"a {offset} ms offset changed the hand-off count from "
             f"{counts[0]} to {count}: the alignment is not being applied")


def test_a_canonical_handoff_is_counted(tmp_path):
    """The measurement the corpus exists to make, now made.

    A canonical hand-off is: A stops speaking, B takes the floor, A stays
    quiet.

    This assertion was INVERTED on 2026-10-19: the intake used to count it
    ZERO times.
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
    assert report["gates"]["hand_offs_seen"] == 1, \
        (f"the intake counted {report['gates']['hand_offs_seen']} canonical "
         "hand-offs, expected 1")

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


def test_a_corpus_of_canonical_handoffs_is_not_rejected(tmp_path):
    """The corpus-level consequence of the fix.

    The protocol corpus, with every turn canonical, contains real
    hand-offs. The intake used to count NONE of them and reject the corpus
    for "zero hand-offs".

    This assertion was INVERTED on 2026-10-19.
    """
    specs = [s for s in st.strict_specs() if s["name"] in TEST_SCENES]
    d = tmp_path / "strict"
    st.build_corpus(d, SCENE_SECONDS, TONE_RMS, specs=specs)
    _, report = run_intake(d)
    assert report["gates"]["hand_offs_seen"] >= 2, \
        (f"the intake counted {report['gates']['hand_offs_seen']} hand-offs "
         "in a corpus full of canonical ones")
    assert not any("zero hand-offs" in u for u in report["gates"]["unmet"]), \
        ("the intake rejects a canonical-only corpus for having no "
         "hand-offs")
    assert not any("hand-off" in u for u in non_length_unmet(report)), \
        "the intake rejects a canonical-only corpus for a hand-off reason"


# ------------------------------------------------------------- the silence --

def test_a_short_dip_is_not_a_hand_off(tmp_path):
    """A 20 ms dip inside a word is not a rest, and is not a hand-off.

    The old rule had no duration filter at all between "a pause" and "a
    hand-off", so a 20 ms envelope dip counted. The fix applies the repo's
    own `energy.MIN_SILENCE_S` (0.150 s).
    """
    spec = dict(name="probe", protocol_s=8, kind="dialogue", events=[
        ("A", 0.60, 2.00, 101), ("B", 2.95, 3.60, 102),
    ])
    a, b, _ = st.build_scene(spec, 8.0, TONE_RMS)
    # A 20 ms hole in the middle of A's speech.
    hole = a.copy()
    start = int(1.20 * st.SR)
    hole[start:start + int(0.020 * st.SR)] = 0.0
    d = tmp_path / "dip"
    d.mkdir(parents=True, exist_ok=True)
    st.write_wav(d / "probe_A.wav", hole)
    st.write_wav(d / "probe_B.wav", b)
    _, report = run_intake(d)
    assert report["gates"]["hand_offs_seen"] == 1, \
        ("the 20 ms dip changed the hand-off count: the minimum-duration "
         "filter is not applied")


def test_a_silent_channel_B_in_scene_7_is_not_an_error(tmp_path):
    """Scene 7 is SPECIFIED to have a silent channel B.

    So "B is silent" cannot be an error on its own, and the gate has to be
    scoped to the scenes where the protocol has B speaking. This is the
    constraint the audit named, and the fix has to keep it.
    """
    specs = [s for s in st.SCENES if s["name"] == "scene07_alone"]
    d = tmp_path / "scene7"
    st.build_corpus(d, SCENE_SECONDS, TONE_RMS, specs=specs)
    code, report = run_intake(d)
    p = pair_of(report, "scene07_alone")
    assert p["B"]["signal_frame_frac"] < 0.02, \
        "scene 7's channel B should carry almost nothing"
    assert not any("silent" in u.lower() for u in non_length_unmet(report)), \
        ("the intake rejects a scene 7 whose channel B is silent, which is "
         "what the protocol asks for")
    assert p["B_expected_to_speak"] is False


def test_the_room_tone_scene_is_not_reported_as_two_people_talking(tmp_path):
    """The §7 failure: an ordinary room, reported as 100% overlap.

    The old floor was `energy.ABS_FLOOR` (1e-4 RMS, about -80 dBFS). An
    ordinary lived-in room is 23.5 dB above that, so the old intake
    reported the empty room-tone scene as 100% signal and 100% overlap --
    "a full minute of two people talking over each other" out of an empty
    room.
    """
    quiet = tmp_path / "quiet"
    ordinary = tmp_path / "ordinary"
    st.build_corpus(quiet, SCENE_SECONDS, TONE_RMS,
                    specs=scene_specs(("scene08_roomtone",)))
    st.build_corpus(ordinary, SCENE_SECONDS, 1.5e-3,
                    specs=scene_specs(("scene08_roomtone",)))

    _, q_report = run_intake(quiet)
    _, o_report = run_intake(ordinary)
    q = pair_of(q_report, "scene08_roomtone")
    o = pair_of(o_report, "scene08_roomtone")
    assert q["levels"]["both_loud_frac"] < 0.02, \
        f"the quiet room tone reads {q['levels']['both_loud_frac']} overlap"
    assert o["levels"]["both_loud_frac"] < 0.02, \
        (f"the ordinary room tone reads {o['levels']['both_loud_frac']} "
         "overlap: the level floor is still absolute")
    assert o["A"]["signal_frame_frac"] < 0.02
    assert o["B"]["signal_frame_frac"] < 0.02
