"""
Tests for the LEADING-SILENCE-ARTEFACT measurement.

WHY THIS FILE EXISTS.

`src/energy.py:_merge_short_runs` keeps the first segment of a segment list
unconditionally:

    kept = [segs[0]]                       # src/energy.py:290

so the first segment escapes both filters. On a whole file that is the
silence before the first word -- 0.130 s in `dialogue`, 0.110 s in
`shakespeare`, 0.110 s in `children` -- and `docs/research/PAUSE-THRESHOLD.md`
shows that the published "the shortest pause a human voice actually takes is
110 ms" is an artefact of that line rather than a property of speech.

The defect was deliberately NOT patched, because nobody had measured which
numbers would move. `experiments/silence_leading_artefact.py` measures them,
and this file keeps that measurement honest.

WHAT IS ASSERTED HERE, AND WHAT IS DELIBERATELY NOT.

  * The ARTEFACT-FREE population reproduces the
    `corrected_without_leading_artefact` block of
    `experiments/pause_threshold_results.json` EXACTLY, on every key. That
    block is the committed authority for the artefact-free numbers; if this
    file's computation disagreed with it, that disagreement would be the
    finding, not something to paper over.
  * The CURRENT population reproduces the main `gaps` block of the same file
    EXACTLY, so the two populations are measured with the shipped method.
  * The two populations ACTUALLY DIFFER. A test that passes when the fix is
    a no-op is not a test, so the difference is asserted in both directions:
    the populations differ, AND a checker that would have been used to
    validate the fix is shown to REJECT the current population.
  * The checker is attacked: a perturbed count must be REJECTED.
  * The provenance block is asserted to exist, and every recorded SHA-256 is
    checked against the file on disk.

WHAT IS NOT ASSERTED.

The OLD numbers are not asserted as properties of the corpus. Where a
committed number turns out to be an artefact of this defect, that is recorded
in `docs/research/SILENCE-LEADING-ARTEFACT.md` and in the result file, not
asserted as if it were true.

The corpus WAVs are gitignored. The house pattern (tests/test_vad.py's
`needs_weights`, tests/test_speculate.py's `needs_corpus`) is to SKIP with a
real, stated reason when the prerequisite is genuinely absent.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import silence_leading_artefact as sla           # noqa: E402

RESULTS = ROOT / "experiments" / "silence_leading_artefact_results.json"
COMMITTED = ROOT / "experiments" / "pause_threshold_results.json"
SCRIPT = ROOT / "experiments" / "silence_leading_artefact.py"
ENERGY = ROOT / "src" / "energy.py"

# The cells whose leading silence IS the artefact: the first segment of the
# file is a silence shorter than MIN_SILENCE_S, so it survives only because
# `kept = [segs[0]]` skipped the filter.
ARTEFACT_CELLS = ("dialogue", "shakespeare", "children")
CLEAN_CELLS = ("narration", "poetry", "essay", "oration")

MISSING_CELLS = tuple(n for n, p in sla.CELLS.items() if not p.is_file())
needs_corpus = pytest.mark.skipif(
    bool(MISSING_CELLS),
    reason="corpus not built, missing cell(s): " + ", ".join(MISSING_CELLS)
           + " -- audio is gitignored")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_committed() -> dict:
    assert COMMITTED.is_file(), f"missing committed result file: {COMMITTED}"
    return json.loads(COMMITTED.read_text())


def load_result() -> dict:
    assert RESULTS.is_file(), (
        f"missing result artefact: {RESULTS}. Regenerate it with "
        "`python experiments/silence_leading_artefact.py`")
    return json.loads(RESULTS.read_text())


@pytest.fixture(scope="module")
def live() -> dict:
    """
    Both populations, measured LIVE from the corpus on disk.

    Not read from the result file: the result file is the thing under test
    elsewhere, and a test that checks a file against itself proves nothing.
    """
    from src import wavio
    out = {}
    for name, path in sla.CELLS.items():
        if not path.is_file():
            continue
        samples = wavio.read_wav(str(path)).samples
        out[name] = {
            "current": sla.measure_energy_population(samples, sla.CURRENT),
            "artefact_free": sla.measure_energy_population(samples,
                                                           sla.ARTEFACT_FREE),
        }
    return out


def _disagreements(got: dict, want: dict) -> list:
    """The keys of `want` that `got` does not reproduce."""
    return [k for k in want if got.get(k) != want[k]]


class TestTheEnergyPopulations:
    """The whole-file pause table, both populations, against the committed file."""

    @needs_corpus
    def test_the_current_population_reproduces_the_committed_gaps_block(self, live):
        """
        The CURRENT population must reproduce the main `gaps` block of
        `experiments/pause_threshold_results.json` exactly -- every key,
        including the gap list and its digest.

        If this fails, this file's method is not the committed method, and
        every comparison below is meaningless.
        """
        committed = load_committed()
        for name, cell in live.items():
            want = committed["cells"][name]["gaps"]
            got = cell["current"]["gap_block"]
            assert _disagreements(got, want) == [], (
                f"{name}: the current population does not reproduce the "
                f"committed gaps block on {_disagreements(got, want)}")

    @needs_corpus
    def test_the_artefact_free_population_reproduces_the_committed_corrected_block(self, live):
        """
        The ARTEFACT-FREE population must reproduce the
        `corrected_without_leading_artefact` block exactly -- all eight keys.

        That block is the committed authority for the artefact-free numbers.
        A disagreement here is a FINDING to report, not something to adjust
        until it goes away.
        """
        committed = load_committed()
        for name, cell in live.items():
            want = committed["cells"][name]["corrected_without_leading_artefact"]
            got = sla._flat_gap_block(cell["artefact_free"]["gap_block"])
            assert _disagreements(got, want) == [], (
                f"{name}: the artefact-free population does not reproduce the "
                f"committed corrected block on {_disagreements(got, want)}; "
                f"got {got}, want {want}")

    @needs_corpus
    def test_the_two_populations_actually_differ(self, live):
        """
        The measurement must not be a no-op.

        The three cells with a leading artefact lose exactly one gap each; the
        four cells without one are untouched. A run of this file in which the
        variant did nothing would fail here.
        """
        for name in ARTEFACT_CELLS:
            cur = live[name]["current"]["gap_block"]
            art = live[name]["artefact_free"]["gap_block"]
            assert art["n"] == cur["n"] - 1, (
                f"{name}: expected the artefact-free population to drop one "
                f"gap, got {cur['n']} -> {art['n']}")
            assert cur["durations_sha256"] != art["durations_sha256"], (
                f"{name}: the two populations have the same gap list")
            assert art["durations_s"] == cur["durations_s"][1:], (
                f"{name}: the artefact-free list is not the current list "
                f"without its first gap")

        for name in CLEAN_CELLS:
            cur = live[name]["current"]["gap_block"]
            art = live[name]["artefact_free"]["gap_block"]
            assert art["n"] == cur["n"], (
                f"{name}: a cell with no leading artefact lost a gap")
            assert cur["durations_sha256"] == art["durations_sha256"], (
                f"{name}: a cell with no leading artefact changed its gap list")

        total_cur = sum(live[n]["current"]["gap_block"]["n"] for n in live)
        total_art = sum(live[n]["artefact_free"]["gap_block"]["n"] for n in live)
        assert total_cur == 262, f"current gap total is {total_cur}, not 262"
        assert total_art == 259, f"artefact-free gap total is {total_art}, not 259"

    @needs_corpus
    def test_the_shortest_real_pause_is_at_the_trigger_not_below_it(self, live):
        """
        The artefact-free shortest inter-utterance pause, per cell.

        This is the number the published "110 ms" claim rests on. Artefact-free
        it is 0.150 s -- AT the shipped trigger, not below it.
        """
        shortest = {n: live[n]["artefact_free"]["gap_block"]["shortest_s"]
                    for n in live}
        assert shortest["dialogue"] == 0.16, shortest
        assert shortest["shakespeare"] == 0.15, shortest
        assert shortest["children"] == 0.15, shortest
        assert min(shortest.values()) >= 0.15, shortest


class TestTheVariant:
    """
    The variant itself, on synthetic segments.

    These are the tests that FAIL if the variant is removed, reverted to the
    shipped function, or subtly broken.
    """

    def _segments(self):
        from src import energy
        return [energy.Segment(0.025, 0.155, False),      # a 0.130 s silence
                energy.Segment(0.155, 1.155, True)]        # 1.000 s of speech

    def test_the_shipped_function_keeps_the_short_leading_silence(self):
        """The defect, as it stands in src/energy.py."""
        segs = self._segments()
        kept = sla.CURRENT(segs, 0.150, 0.060)
        assert len(kept) == 2, (
            "the shipped _merge_short_runs no longer keeps the leading "
            "silence; the defect this file measures is gone (or moved)")
        assert not kept[0].speech and kept[0].duration() == 0.130

    def test_the_variant_merges_the_short_leading_silence_forward(self):
        """The artefact-free rule: the leading silence disappears."""
        segs = self._segments()
        kept = sla.ARTEFACT_FREE(segs, 0.150, 0.060)
        assert len(kept) == 1, f"expected one segment, got {kept}"
        assert kept[0].speech, "the absorber should be the speech segment"
        assert kept[0].duration() == pytest.approx(1.130, abs=1e-9), (
            "the leading silence should be absorbed into the speech that "
            f"follows it; duration is {kept[0].duration()}")

    def test_the_variant_is_a_no_op_when_the_leading_segment_survives(self):
        """
        A leading segment that is NOT a short silence must come through
        untouched -- the variant changes one rule, not the segmenter.
        """
        from src import energy
        segs = [energy.Segment(0.025, 0.785, False),      # 0.760 s of silence
                energy.Segment(0.785, 1.785, True)]
        shipped = sla.CURRENT([energy.Segment(s.start, s.end, s.speech)
                               for s in segs], 0.150, 0.060)
        variant = sla.ARTEFACT_FREE([energy.Segment(s.start, s.end, s.speech)
                                     for s in segs], 0.150, 0.060)
        assert len(shipped) == len(variant) == 2
        for a, b in zip(shipped, variant):
            assert (a.start, a.end, a.speech) == (b.start, b.end, b.speech)

    def test_the_variant_does_not_mutate_the_segments_it_is_given(self):
        """The caller's segments must survive the call."""
        from src import energy
        segs = self._segments()
        before = [(s.start, s.end, s.speech) for s in segs]
        sla.ARTEFACT_FREE(segs, 0.150, 0.060)
        assert [(s.start, s.end, s.speech) for s in segs] == before

    def test_the_variant_restores_the_shipped_function_afterwards(self):
        """
        The variant is installed only for the duration of a measurement, so
        `src/energy.py` keeps behaving exactly as it does now.
        """
        from src import energy
        original = energy._merge_short_runs
        with sla._variant_active(sla.ARTEFACT_FREE):
            assert energy._merge_short_runs is sla.ARTEFACT_FREE
        assert energy._merge_short_runs is original
        assert energy._merge_short_runs is sla.CURRENT


class TestTheCheckerIsAttacked:
    """A checker that cannot fail is not evidence."""

    @needs_corpus
    def test_a_perturbed_count_is_rejected(self, live):
        """
        Perturb one cell's artefact-free count by one and the check against
        the committed block must FAIL.
        """
        committed = load_committed()
        want = committed["cells"]["dialogue"]["corrected_without_leading_artefact"]
        got = sla._flat_gap_block(live["dialogue"]["artefact_free"]["gap_block"])
        assert _disagreements(got, want) == []

        perturbed = dict(got)
        perturbed["n"] = got["n"] + 1
        assert _disagreements(perturbed, want) == ["n"], (
            "the checker accepted a perturbed gap count")

        perturbed = dict(got)
        perturbed["shortest_s"] = round(got["shortest_s"] + 0.001, 6)
        assert _disagreements(perturbed, want) == ["shortest_s"], (
            "the checker accepted a perturbed shortest gap")

    @needs_corpus
    def test_removing_the_variant_would_be_caught(self, live):
        """
        If the variant were removed -- that is, if the artefact-free
        population were measured with the shipped function -- the check
        against the committed corrected block would FAIL for exactly the
        three cells that carry the artefact.

        This is the test that would go red if someone "fixed" the artefact
        by deleting the measurement instead of applying the fix.
        """
        committed = load_committed()
        for name in ARTEFACT_CELLS:
            want = committed["cells"][name]["corrected_without_leading_artefact"]
            got = sla._flat_gap_block(live[name]["current"]["gap_block"])
            assert _disagreements(got, want), (
                f"{name}: the shipped population was accepted as the "
                "artefact-free one; the variant is not doing anything")

        for name in CLEAN_CELLS:
            want = committed["cells"][name]["corrected_without_leading_artefact"]
            got = sla._flat_gap_block(live[name]["current"]["gap_block"])
            assert _disagreements(got, want) == [], (
                f"{name}: the two populations should agree here")


class TestTheWatcherPopulation:
    """
    The watcher numbers, read from the persisted result file.

    The watcher is the streaming layer: `energy.segment` is called on a MOVING
    buffer, so "the first segment" there is "the segment at the buffer
    origin". That is why the defect fires far more often in the watcher than
    in the whole-file population.
    """

    def test_the_current_watcher_population_reproduces_the_committed_block(self):
        """
        The CURRENT watcher population must reproduce the committed per-cell
        block -- 673 runs and 1671 events across the corpus -- exactly.
        """
        committed = load_committed()
        result = load_result()
        want = committed["watcher_pause_run_population"]["per_cell"]
        got = result["watcher"]["current"]["per_cell"]
        for name, block in want.items():
            assert {k: got[name].get(k) for k in block} == block, (
                f"{name}: the current watcher population does not reproduce "
                "the committed per-cell block")

        assert result["watcher"]["current"]["distinct_silence_runs"] == 673
        assert result["watcher"]["current"]["events_gt_0_15"] == 1671

    def test_the_artefact_free_watcher_population_is_different(self):
        """
        The fix is NOT a no-op for the watcher. If it were, the whole premise
        of this lane would be false.
        """
        result = load_result()
        cur = result["watcher"]["current"]
        art = result["watcher"]["artefact_free"]
        assert art["distinct_silence_runs"] != cur["distinct_silence_runs"], (
            "the artefact-free watcher population has the same number of "
            "silence runs as the current one")
        assert art["events_gt_0_15"] != cur["events_gt_0_15"], (
            "the artefact-free watcher population has the same number of "
            "pause events as the current one")
        for name in ARTEFACT_CELLS:
            assert (art["per_cell"][name]["distinct_silence_runs"] !=
                    cur["per_cell"][name]["distinct_silence_runs"]), (
                f"{name}: the watcher population did not change for a cell "
                "that carries the artefact")

    def test_the_speculation_figures_are_reproduced_and_challenged(self):
        """
        The two figures `tests/test_speculate.py` asserts on: the reach rate
        at the shipped speculation threshold, and the share below the
        rejected 1.20 s.
        """
        result = load_result()
        cur = result["watcher"]["current"]["specification"]
        assert cur["n"] == 1671
        # The committed docstring quotes 75.3% and 94.4%, i.e. one decimal.
        assert round(cur["reach_rate_pct"], 1) == 75.3, cur
        assert round(cur["below_1_20_pct"], 1) == 94.4, cur
        assert cur["reach"] == 1258, cur
        assert cur["below_1_20"] == 1578, cur

        art = result["watcher"]["artefact_free"]["specification"]
        assert art["n"] != cur["n"], (
            "the artefact-free speculation population is the same size")
        # The floors `tests/test_speculate.py` asserts on.
        assert art["reach_rate_pct"] / 100.0 > 0.35, art
        assert art["below_1_20_pct"] / 100.0 > 0.90, art

    def test_the_three_second_turn_threshold_fires_on_nothing(self):
        """
        The 3 s turn threshold fires on zero silence runs, in BOTH
        populations -- the longest run is below it either way.
        """
        result = load_result()
        for label in ("current", "artefact_free"):
            w = result["watcher"][label]
            assert w["runs_at_or_above_3s"] == 0, w
            assert w["events_at_or_above_3s"] == 0, w
            assert w["run_max_s"] < 3.0, w

    def test_the_f0_population_is_recorded(self):
        """
        The population `experiments/f0_speaker_change.py` consumes is
        recorded for both populations, so the F0 numbers can be checked
        without re-running the expensive pitch pass.
        """
        result = load_result()
        f0 = result["f0_population"]
        assert f0["current"]["pause_population"] > 0
        assert f0["artefact_free"]["pause_population"] > 0
        assert "identical" in f0 or "population_identical" in f0


class TestTheTotalsAreTheSumOfTheCells:
    """
    The corpus totals must be the sum of the per-cell numbers.

    A total that is not the sum of its parts is a number nobody can trace, and
    this file exists to keep numbers traceable.
    """

    def test_the_energy_totals_are_the_sum_of_the_per_cell_gap_blocks(self):
        result = load_result()
        for label in ("current", "artefact_free"):
            blocks = [c[label]["gap_block"] for c in result["cells"].values()]
            n = sum(b["n"] for b in blocks)
            at150 = sum(b["trigger"]["at_0_150"] for b in blocks)
            at200 = sum(b["trigger"]["at_0_200"] for b in blocks)
            total = result["totals"][label]
            assert total["gaps"] == n, (label, total["gaps"], n)
            assert total["at_0_150"] == at150, (label, total["at_0_150"], at150)
            assert total["at_0_200"] == at200, (label, total["at_0_200"], at200)
            assert total["lost"] == at150 - at200, label

    def test_the_watcher_totals_are_the_sum_of_the_per_cell_blocks(self):
        result = load_result()
        for label in ("current", "artefact_free"):
            cells = result["watcher"][label]["per_cell"]
            events = sum(c["pause_events_gt_0_15"] for c in cells.values())
            runs = sum(c["distinct_silence_runs"] for c in cells.values())
            assert result["watcher"][label]["events_gt_0_15"] == events, label
            assert result["watcher"][label]["distinct_silence_runs"] == runs, \
                label


class TestProvenance:
    """A result that cannot be traced is the defect this project is fixing."""

    def test_the_provenance_block_is_present(self):
        result = load_result()
        prov = result["provenance"]
        assert prov, "no provenance block"
        for key in ("cells", "committed_results", "method_source",
                    "energy_module", "script"):
            assert key in prov, f"provenance has no {key!r}"
        assert result["method"]["varied_source"]["defect_line_verified"] is True

    def test_every_recorded_sha256_matches_the_file_on_disk(self):
        result = load_result()
        prov = result["provenance"]
        for name, block in prov["cells"].items():
            path = Path(block["path"])
            assert path.is_file(), f"{name}: {path} is not on disk"
            assert sha256_file(path) == block["sha256"], (
                f"{name}: the recorded SHA-256 does not match the file on "
                "disk; the input changed since the measurement")
        assert sha256_file(SCRIPT) == prov["script"]["sha256"], (
            "the measured script has changed since the result file was "
            "written")
        assert sha256_file(ENERGY) == prov["energy_module"]["sha256"], (
            "src/energy.py has changed since the measurement; the defect may "
            "have moved")
        assert sha256_file(COMMITTED) == prov["committed_results"]["sha256"]
        assert sha256_file(ROOT / "tests" / "test_pause_threshold.py") == \
            prov["method_source"]["sha256"]

    def test_the_defect_is_where_the_report_says_it_is(self):
        """src/energy.py:290 is the line the report names."""
        lines = ENERGY.read_text().splitlines()
        assert lines[sla.DEFECT_LINE_NO - 1] == sla.DEFECT_LINE_TEXT, (
            f"src/energy.py:{sla.DEFECT_LINE_NO} is "
            f"{lines[sla.DEFECT_LINE_NO - 1]!r}, not {sla.DEFECT_LINE_TEXT!r}")

    def test_the_inputs_are_recorded_with_their_own_headers(self):
        result = load_result()
        for name, block in result["provenance"]["cells"].items():
            assert block["analysis_sample_rate"] == 16000, name
            assert block["file_sample_rate"] > 0, name
            assert block["sha256"] and len(block["sha256"]) == 64, name
            assert block["analysis_samples"] > 0, name

    def test_the_result_file_records_the_script_that_wrote_it(self):
        result = load_result()
        assert result["script"]["sha256"] == sha256_file(SCRIPT), (
            "the result file was not written by the script on disk")
        assert result["schema"] == "silence-leading-artefact/1"
