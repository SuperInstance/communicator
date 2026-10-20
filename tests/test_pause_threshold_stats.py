"""
Tests for the PERSISTED pause-threshold statistics.

WHY THIS FILE EXISTS.

The pause statistics in `docs/research/ADOPTION-PLAN.md` §3 and
`docs/research/TURN-ROUTES.md` used to live ONLY in a test docstring and in
prose. There was no result file, so the numbers could not be traced to the
audio they claimed to describe, and they could not be re-derived by anyone
else. `experiments/pause_threshold_results.json` is that result file, and
these tests are what keep it honest.

WHAT IS ASSERTED HERE, AND WHAT IS DELIBERATELY NOT.

  * The corpus length is asserted against the value MEASURED ON DISK
    (485.011 s), never against the 465 s the prose used to claim.
  * The per-cell gap counts and the lost-table counts are re-derived from the
    corpus and compared with the persisted result file, so the artefact and
    the document cannot drift apart.
  * The provenance block is asserted to exist, and every recorded SHA-256 is
    checked against the file on disk.
  * The checker is attacked: a result file edited to hold a wrong number must
    be REJECTED. A suite that cannot fail is not evidence.

The OLD numbers are not asserted. The published table's counts were
re-measured on the corpus that is actually on disk and they reproduce
exactly, so the counts asserted here are the measured ones; where a
published number was an artefact of a method defect it is recorded as such in
`docs/research/PAUSE-THRESHOLD.md` and in the result file, not asserted as
if it were a property of the corpus.

The corpus WAVs are gitignored. The house pattern (tests/test_vad.py's
`needs_weights`, tests/test_speculate.py's `needs_corpus`) is to SKIP with a
real, stated reason when the prerequisite is genuinely absent, rather than to
fail for want of a file.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import pause_threshold as pt              # noqa: E402

RESULTS = ROOT / "experiments" / "pause_threshold_results.json"
REPORT = ROOT / "docs" / "research" / "PAUSE-THRESHOLD.md"

# The corpus length, MEASURED ON DISK. This is the number the test asserts;
# the 465 s the prose used to claim is 20.011 s short and is asserted NOT to
# be the corpus (see TestTheCorpusIsTheLengthOnDisk).
MEASURED_CORPUS_S = 485.011
MEASURED_CELL_S = {
    "narration": 90.000,
    "poetry": 20.011,
    "essay": 75.000,
    "dialogue": 75.000,
    "shakespeare": 75.000,
    "children": 75.000,
    "oration": 75.000,
}

MISSING_CELLS = tuple(n for n, p in pt.CELLS.items() if not p.is_file())
needs_corpus = pytest.mark.skipif(
    bool(MISSING_CELLS),
    reason="corpus not built, missing cell(s): " + ", ".join(MISSING_CELLS)
           + " -- audio is gitignored; regenerate with"
             " examples/realaudio/corpus/_downloads/make_excerpt.py")


def load_result() -> dict:
    assert RESULTS.is_file(), (
        f"missing result artefact: {RESULTS}. Regenerate it with "
        "/home/proart/projects/glasses/.venv/bin/python "
        "experiments/pause_threshold.py")
    return json.loads(RESULTS.read_text())


# ---------------------------------------------------------------------------
# The checker. This is the function that has to be able to fail, so it is
# written once and used by every test that needs a verdict.
# ---------------------------------------------------------------------------

def verify(result: dict) -> None:
    """
    Re-derive the measurement from the corpus ON DISK and assert the
    persisted result agrees with it, number by number.

    Raises AssertionError on any disagreement. The perturbation tests below
    attack THIS function, so a defect here is a defect in the evidence.
    """
    for name, path in pt.CELLS.items():
        assert name in result["cells"], f"result file has no cell {name!r}"
        cell = result["cells"][name]
        fresh = pt.measure_cell(name, path)

        assert cell["gaps"]["n"] == fresh["gaps"]["n"], (
            f"{name}: persisted gap count {cell['gaps']['n']} != measured "
            f"{fresh['gaps']['n']}")
        assert cell["gaps"]["durations_s"] == fresh["gaps"]["durations_s"], (
            f"{name}: persisted gap list differs from the measured one")
        assert cell["gaps"]["durations_sha256"] == fresh["gaps"]["durations_sha256"], (
            f"{name}: persisted gap-list hash differs from the measured one")
        assert cell["gaps"]["shortest_s"] == fresh["gaps"]["shortest_s"], (
            f"{name}: persisted shortest gap differs from the measured one")
        assert (cell["gaps"]["p05_s_nearest_rank"]
                == fresh["gaps"]["p05_s_nearest_rank"]), (
            f"{name}: persisted p05 differs from the measured one")
        assert (cell["gaps"]["median_s_published_convention"]
                == fresh["gaps"]["median_s_published_convention"]), (
            f"{name}: persisted median differs from the measured one")

        for key in ("at_0_150", "at_0_200", "lost", "lost_pct"):
            assert cell["trigger"][key] == fresh["trigger"][key], (
                f"{name}: persisted trigger figure {key}="
                f"{cell['trigger'][key]!r} != measured {fresh['trigger'][key]!r}")

        assert cell["input"]["sha256"] == fresh["input"]["sha256"], (
            f"{name}: persisted input SHA-256 differs from the file on disk")
        assert (cell["input"]["analysis_duration_s"]
                == fresh["input"]["analysis_duration_s"]), (
            f"{name}: persisted duration differs from the file on disk")

    total = sum(c["gaps"]["n"] for c in result["cells"].values())
    assert result["totals"]["gaps"] == total, (
        f"persisted total {result['totals']['gaps']} != the sum of the cells "
        f"{total}")
    at150 = sum(c["trigger"]["at_0_150"] for c in result["cells"].values())
    at200 = sum(c["trigger"]["at_0_200"] for c in result["cells"].values())
    assert result["totals"]["at_0_150"] == at150, "persisted @0.150 total is wrong"
    assert result["totals"]["at_0_200"] == at200, "persisted @0.200 total is wrong"
    assert result["totals"]["lost"] == at150 - at200, (
        f"persisted lost total {result['totals']['lost']} != "
        f"{at150} - {at200}")
    # The persisted percentage is rounded to four decimals, so the tolerance
    # is 1e-4 rather than exact equality.
    assert abs(result["totals"]["lost_pct"]
               - 100.0 * (at150 - at200) / at150) < 1e-4, (
        "persisted lost%% is not the loss over the @0.150 population")


# ---------------------------------------------------------------------------
# The corpus length.
# ---------------------------------------------------------------------------

class TestTheCorpusIsTheLengthOnDisk:
    @needs_corpus
    def test_the_seven_cells_total_485_011_seconds(self):
        """
        THE corpus length, read from the WAV headers rather than from any
        document.

        465 s was the number the prose carried and it was 20.011 s short --
        exactly the poetry cell, which is 20.011 s on disk and was implicitly
        assumed to be 15 s.
        """
        total = 0.0
        for name, path in pt.CELLS.items():
            with wave.open(str(path), "rb") as w:
                frames, sr = w.getnframes(), w.getframerate()
            cell_s = frames / sr
            assert abs(cell_s - MEASURED_CELL_S[name]) < 0.01, (
                f"{name}: {cell_s:.6f} s on disk, expected "
                f"{MEASURED_CELL_S[name]:.6f} s")
            total += cell_s
        assert abs(total - MEASURED_CORPUS_S) < 0.01, (
            f"corpus is {total:.6f} s on disk, expected "
            f"{MEASURED_CORPUS_S:.6f} s")

    @needs_corpus
    def test_the_published_465_seconds_is_not_this_corpus(self):
        """
        The falsification, stated as a test. If the corpus ever really were
        465 s this would fail, which is the point: the published figure must
        not come back by accident.
        """
        total = 0.0
        for path in pt.CELLS.values():
            with wave.open(str(path), "rb") as w:
                total += w.getnframes() / w.getframerate()
        assert abs(total - 465.0) > 10.0, (
            f"the corpus is {total:.3f} s, which is within 10 s of the "
            "published 465 s -- the F7 misstatement may have been right about "
            "this corpus, or the corpus changed; re-derive before assuming")

    def test_the_result_file_records_the_measured_length(self):
        """
        The artefact must carry the number that was measured. A hand-edited
        result file that still said 465 s is rejected here.
        """
        result = load_result()
        assert abs(result["corpus"]["total_duration_s_on_disk"]
                   - MEASURED_CORPUS_S) < 0.01, (
            f"the result file records "
            f"{result['corpus']['total_duration_s_on_disk']} s, expected "
            f"{MEASURED_CORPUS_S} s")
        assert result["corpus"]["matches_published_465_s"] is False


# ---------------------------------------------------------------------------
# The artefact agrees with the corpus, and with the document.
# ---------------------------------------------------------------------------

@needs_corpus
class TestTheArtefactMatchesTheCorpus:
    def test_a_fresh_measurement_agrees_with_the_persisted_result(self):
        """
        The whole point: the number in the file can be re-derived from the
        audio. Every count, every gap list, every hash.
        """
        verify(load_result())

    def test_the_per_cell_gap_counts_are_the_measured_ones(self):
        """
        The gap counts, per cell, as measured. These are the numbers the
        document and the artefact both have to carry.
        """
        result = load_result()
        expected = {
            "narration": 45, "poetry": 7, "essay": 33, "dialogue": 35,
            "shakespeare": 62, "children": 78, "oration": 2,
        }
        for name, n in expected.items():
            assert result["cells"][name]["gaps"]["n"] == n, (
                f"{name}: {result['cells'][name]['gaps']['n']} gaps measured, "
                f"{n} expected")

    def test_the_lost_table_counts_are_the_measured_ones(self):
        """
        The @0.150 / @0.200 / lost counts per cell, as measured.
        """
        result = load_result()
        expected = {
            # cell: (at_0_150, at_0_200, lost)
            "narration":   (45, 43,  2),
            "poetry":      ( 7,  6,  1),
            "essay":       (33, 24,  9),
            "dialogue":    (34, 30,  4),
            "shakespeare": (61, 46, 15),
            "children":    (77, 62, 15),
            "oration":     ( 2,  2,  0),
        }
        for name, (a, b, lost) in expected.items():
            trig = result["cells"][name]["trigger"]
            assert (trig["at_0_150"], trig["at_0_200"], trig["lost"]) == \
                (a, b, lost), (
                f"{name}: trigger figures {trig} != expected {(a, b, lost)}")

    def test_the_headline_is_the_loss_over_the_150ms_population(self):
        """
        The headline, with its denominator stated. 46 of the 259 pauses the
        150 ms trigger keeps are shorter than 200 ms.
        """
        result = load_result()
        assert result["totals"]["at_0_150"] == 259
        assert result["totals"]["at_0_200"] == 213
        assert result["totals"]["lost"] == 46
        assert abs(result["totals"]["lost_pct"] - 17.76) < 0.01

    def test_the_headline_survives_the_leading_silence_defect(self):
        """
        A defect was found in the copied method: `src/energy.py:290` keeps the
        first segment of a file regardless of `min_silence_s`, so a file's
        LEADING silence is reported as a gap. The headline must not depend on
        it, and this says so with numbers rather than with an assurance.
        """
        result = load_result()
        corrected = result["totals"]["corrected_without_leading_artefact"]
        assert corrected["gaps"] == 259
        assert corrected["at_0_150"] == 259
        assert corrected["at_0_200"] == 213
        assert corrected["lost"] == 46
        assert abs(corrected["lost_pct"] - result["totals"]["lost_pct"]) < 1e-6
        # And the artefact is real: three cells have a leading silence below
        # the trigger, and it is the reason the published table's "shortest"
        # column reads 0.110 s.
        artefact = result["known_defect"]["effect_here"]
        assert set(artefact) == {"dialogue", "shakespeare", "children"}
        for cell, info in artefact.items():
            assert info["duration_s"] < 0.150, (
                f"{cell}: the leading artefact {info} is not below the trigger")


# ---------------------------------------------------------------------------
# The document and the artefact cannot drift apart.
# ---------------------------------------------------------------------------

def _doc_table() -> dict:
    """Parse the measured table out of docs/research/PAUSE-THRESHOLD.md."""
    assert REPORT.is_file(), f"missing report: {REPORT}"
    lines = REPORT.read_text().splitlines()
    header = None
    rows = {}
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if header is not None:
                break            # the measured table ended
            continue
        cells = [c.strip().replace("*", "") for c in stripped.strip("|").split("|")]
        if header is None:
            if cells and cells[0] == "cell" and "gaps" in cells:
                header = cells
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        name = cells[0].strip()
        if name in pt.CELLS:
            rows[name] = cells
        elif name.upper().startswith("TOTAL"):
            rows["TOTAL"] = cells
    assert header is not None, (
        f"no measured table found in {REPORT} (expected a header row "
        "starting '| cell |' and containing 'gaps')")
    return {"header": header, "rows": rows}


class TestTheDocumentAndTheArtefactAgree:
    def test_the_document_table_matches_the_result_file(self):
        """
        The document is a rendering of the artefact. If someone edits one
        without the other, this fails.
        """
        result = load_result()
        table = _doc_table()
        for name, cell in result["cells"].items():
            assert name in table["rows"], f"the report has no row for {name}"
            row = table["rows"][name]
            g, t = cell["gaps"], cell["trigger"]
            assert int(row[1]) == g["n"], (
                f"{name}: report says {row[1]} gaps, artefact says {g['n']}")
            assert abs(float(row[2]) - g["shortest_s"]) < 5e-4, (
                f"{name}: report shortest {row[2]} != artefact {g['shortest_s']}")
            assert abs(float(row[3]) - g["p05_s_nearest_rank"]) < 5e-4, (
                f"{name}: report p05 {row[3]} != artefact "
                f"{g['p05_s_nearest_rank']}")
            assert abs(float(row[4]) - g["median_s_published_convention"]) < 5e-4, (
                f"{name}: report median {row[4]} != artefact "
                f"{g['median_s_published_convention']}")
            assert int(row[5]) == t["at_0_150"], (
                f"{name}: report @0.150 {row[5]} != artefact {t['at_0_150']}")
            assert int(row[6]) == t["at_0_200"], (
                f"{name}: report @0.200 {row[6]} != artefact {t['at_0_200']}")
            assert int(row[7]) == t["lost"], (
                f"{name}: report lost {row[7]} != artefact {t['lost']}")
            assert abs(float(row[8].rstrip("%")) - t["lost_pct"]) < 0.05, (
                f"{name}: report lost% {row[8]} != artefact {t['lost_pct']}")

    def test_the_document_total_row_matches_the_result_file(self):
        result = load_result()
        table = _doc_table()
        assert "TOTAL" in table["rows"], "the report has no TOTAL row"
        row = table["rows"]["TOTAL"]
        t = result["totals"]
        assert int(row[1]) == t["gaps"], (
            f"report TOTAL gaps {row[1]} != artefact {t['gaps']}")
        assert int(row[5]) == t["at_0_150"], (
            f"report TOTAL @0.150 {row[5]} != artefact {t['at_0_150']}")
        assert int(row[6]) == t["at_0_200"], (
            f"report TOTAL @0.200 {row[6]} != artefact {t['at_0_200']}")
        assert int(row[7]) == t["lost"], (
            f"report TOTAL lost {row[7]} != artefact {t['lost']}")
        assert abs(float(row[8].rstrip("%")) - t["lost_pct"]) < 0.05, (
            f"report TOTAL lost% {row[8]} != artefact {t['lost_pct']}")

    def test_the_report_states_the_measured_corpus_length(self):
        text = REPORT.read_text()
        assert "485.011" in text, (
            "the report does not state the measured corpus length 485.011 s")


# ---------------------------------------------------------------------------
# Provenance. A number that cannot be traced to its inputs is the defect this
# repository is trying to eliminate.
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_the_provenance_block_is_present(self):
        result = load_result()
        assert result["method"]["source_file"] == "tests/test_pause_threshold.py"
        assert result["method"]["gaps_helper_lines"] == "90-99"
        assert result["method"]["cells_lines"] == "77-87"
        assert result["method"]["parameters"]["sample_rate_hz"] == 16000
        assert result["method"]["parameters"]["frame_len_samples"] == 800
        assert result["method"]["parameters"]["hop_samples"] == 160
        assert result["method"]["parameters"]["min_silence_s"] == 0.150
        assert result["method"]["parameters"]["adaptive_threshold"] is False
        assert result["script"]["sha256"]
        assert result["script"]["path"].endswith("experiments/pause_threshold.py")
        for name in pt.CELLS:
            cell = result["cells"][name]
            assert cell["input"]["sha256"]
            assert Path(cell["input"]["path"]).is_absolute()

    def test_the_method_here_is_the_method_in_the_test_file(self):
        """
        The copy is checked byte for byte, at run time, so the two cannot
        drift apart silently.
        """
        report = pt.check_copy()
        assert report["gaps_helper_identical"], (
            "the _gaps() helper here is no longer byte-identical to "
            f"{report['source_file']}:{report['gaps_helper_lines']}")
        assert report["cells_identical"], (
            "the CELLS mapping here is no longer byte-identical to "
            f"{report['source_file']}:{report['cells_lines']}")
        assert load_result()["method"]["copy_check"]["ok"], (
            "the result file was produced by a script whose method copy did "
            "not match the source -- regenerate it")

    def test_the_recorded_script_hash_is_this_script(self):
        result = load_result()
        assert result["script"]["sha256"] == pt.sha256_file(Path(pt.__file__)), (
            "the result file's script SHA-256 does not match the script on "
            "disk -- regenerate the result file")

    @needs_corpus
    def test_every_recorded_input_hash_matches_the_file_on_disk(self):
        """
        The whole point of a provenance block: every input wav is pinned by
        content, and the pin still holds.
        """
        result = load_result()
        for name, path in pt.CELLS.items():
            recorded = result["cells"][name]["input"]["sha256"]
            assert recorded == pt.sha256_file(path), (
                f"{name}: recorded SHA-256 {recorded} does not match the file "
                f"on disk ({path})")
            assert result["cells"][name]["input"]["path"] == str(path.resolve())

    @needs_corpus
    def test_the_recorded_header_is_the_file_on_disk(self):
        """
        The provenance records the file's OWN header, not the reader's.

        `wavio.read_wav` resamples to 16 kHz, so a provenance block that
        took its sample rate from the reader would report 16 kHz for the
        44.1 kHz narration cell and the block would be lying about its
        input.
        """
        result = load_result()
        for name, path in pt.CELLS.items():
            with wave.open(str(path), "rb") as w:
                frames, rate = w.getnframes(), w.getframerate()
            cell = result["cells"][name]["input"]
            assert cell["file_sample_rate"] == rate, (
                f"{name}: recorded file rate {cell['file_sample_rate']} != "
                f"the header's {rate}")
            assert cell["file_samples"] == frames, (
                f"{name}: recorded frame count {cell['file_samples']} != "
                f"the header's {frames}")
            assert abs(cell["file_duration_s"] - frames / rate) < 1e-9
            assert cell["resampled"] == (rate != 16000)

    @needs_corpus
    def test_the_narration_cell_is_recorded_at_its_own_rate(self):
        """
        The one cell that is resampled, pinned. A provenance block that
        quietly reported 16 kHz for a 44.1 kHz file would pass every other
        check here.
        """
        result = load_result()
        narration = result["cells"]["narration"]["input"]
        assert narration["file_sample_rate"] == 44100
        assert narration["resampled"] is True
        assert narration["analysis_sample_rate"] == 16000
        assert abs(narration["file_duration_s"] - 90.0) < 1e-6
        assert abs(narration["analysis_duration_s"] - 90.0) < 1e-6

    def test_the_gap_lists_are_pinned_by_their_own_bytes(self):
        """
        The gap list is persisted with the SHA-256 of its exact bytes, so a
        reader can re-derive the counts from the list without re-running the
        segmentation.
        """
        result = load_result()
        for name, cell in result["cells"].items():
            blob = pt.canonical_gap_bytes(cell["gaps"]["durations_s"])
            assert cell["gaps"]["durations_sha256"] == pt.sha256_bytes(blob), (
                f"{name}: the persisted gap-list hash does not match the "
                "persisted gap list")


# ---------------------------------------------------------------------------
# The checker must be able to fail.
# ---------------------------------------------------------------------------

@needs_corpus
class TestTheCheckerCanFail:
    def test_a_perturbed_gap_count_is_rejected(self):
        """
        Edit the result file to hold a wrong number and the checker must
        say so. Without this, every other test in this file is decoration.
        """
        result = load_result()
        result["cells"]["essay"]["gaps"]["n"] = 32
        with pytest.raises(AssertionError):
            verify(result)

    def test_a_perturbed_lost_count_is_rejected(self):
        result = load_result()
        result["cells"]["shakespeare"]["trigger"]["lost"] = 3
        with pytest.raises(AssertionError):
            verify(result)

    def test_a_perturbed_trigger_population_is_rejected(self):
        result = load_result()
        result["cells"]["children"]["trigger"]["at_0_200"] = 99
        with pytest.raises(AssertionError):
            verify(result)

    def test_a_perturbed_gap_list_is_rejected(self):
        result = load_result()
        result["cells"]["narration"]["gaps"]["durations_s"][0] = 0.001
        with pytest.raises(AssertionError):
            verify(result)

    def test_a_perturbed_duration_is_rejected(self):
        result = load_result()
        result["cells"]["oration"]["input"]["analysis_duration_s"] = 60.0
        with pytest.raises(AssertionError):
            verify(result)

    def test_a_perturbed_input_hash_is_rejected(self):
        result = load_result()
        result["cells"]["poetry"]["input"]["sha256"] = "0" * 64
        with pytest.raises(AssertionError):
            verify(result)

    def test_a_perturbed_total_is_rejected(self):
        result = load_result()
        result["totals"]["gaps"] = 999
        with pytest.raises(AssertionError):
            verify(result)

    def test_a_perturbed_headline_percentage_is_rejected(self):
        result = load_result()
        result["totals"]["lost_pct"] = 3.5
        with pytest.raises(AssertionError):
            verify(result)
