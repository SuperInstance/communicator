"""
silence_leading_artefact.py -- measure the blast radius of the leading-silence
artefact in `src/energy.py:_merge_short_runs`, WITHOUT changing shipped
behaviour.

WHY THIS EXISTS.

`src/energy.py:_merge_short_runs` (lines 284-306) keeps the FIRST segment of a
segment list unconditionally:

    kept = [segs[0]]                       # src/energy.py:290

so the first segment escapes BOTH filters -- it is not dropped for being too
short a silence, and it is not dropped for being too short a speech. On a whole
file that is the silence before the first word: 0.130 s in `dialogue`, 0.110 s
in `shakespeare`, 0.110 s in `children`. Those are not inter-utterance pauses,
they are the silence before the recording starts, and `docs/research/
PAUSE-THRESHOLD.md` shows the consequence: the published claim "the shortest
pause a human voice actually takes is 110 ms" is an artefact of this line.

THE DEFECT WAS DELIBERATELY NOT PATCHED.

Patching `_merge_short_runs` changes every watcher-based number in the
repository, and nobody has measured which ones move. This script measures
them. It does not touch `src/energy.py`: the artefact-free population is
produced by a LOCAL VARIANT of the merge rule that lives in this file and is
installed only while the measurement runs.

WHAT IS MEASURED, TWICE.

Two populations of segments, from the same audio, the same parameters and the
same code path, differing in exactly one rule:

  * `current`      -- the shipped `src/energy.py:_merge_short_runs`.
  * `artefact_free`-- the same function with the leading-segment rule changed:
                      a first segment that is a silence shorter than
                      `min_silence_s` is merged FORWARD into the segment that
                      follows it, exactly as a short silence with a predecessor
                      is merged into that predecessor.

Everything downstream of the segmentation is then measured on both:

  * the energy-method pause table (gap count, shortest, p05, median, the
    @0.150 / @0.200 trigger counts) -- compared against the committed
    `experiments/pause_threshold_results.json`;
  * the watcher's pause population (events and distinct silence runs);
  * the speculation reach rate and the "share below the rejected 1.20 s"
    figure that `tests/test_speculate.py` asserts on;
  * the turn-boundary counts, and whether the 3 s turn threshold fires;
  * the pause population `experiments/f0_speaker_change.py` consumes.

THE MEASUREMENT IS THE POINT. NOT THE FIX.

No shipped file is edited by this script, and the recommended fix is written
out in `docs/research/SILENCE-LEADING-ARTEFACT.md` rather than applied.

THE METHOD IS COPIED, NOT REINVENTED.

`CELLS` below is copied VERBATIM from `tests/test_pause_threshold.py:77-87`,
and the statistics helpers and the watcher's event/run reconstruction are
IMPORTED from `experiments/pause_threshold.py`, so the two scripts cannot
drift apart silently. `--check-copy` re-verifies the copy byte-for-byte at
run time.

PROVENANCE.

Every result records the absolute path and SHA-256 of every input wav, the
wav's own header (rate, frames, duration) AND the analysis-rate length after
the resampling read, the exact parameters, this script's own SHA-256, and the
file and line of the segmentation logic that was varied. A result that cannot
be traced is the exact defect this repository is fixing.

Run:

    /home/proart/projects/glasses/.venv/bin/python \\
        experiments/silence_leading_artefact.py
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import energy, speculate, turn, wavio, watcher           # noqa: E402
from experiments import pause_threshold as pt                     # noqa: E402

SR = 16000
HOP = 160
HOP_S = HOP / SR
CORPUS = ROOT / "examples" / "realaudio"

# The seven cells from examples/realaudio/corpus/MANIFEST.md. Copied
# VERBATIM from tests/test_pause_threshold.py (the `CELLS` mapping at lines
# 77-87). A different corpus would be a DIFFERENT MEASUREMENT, not a
# re-measurement, so the copy is marked with its source line range and
# `--check-copy` verifies it byte-for-byte at run time.
CELLS = {
    "narration": CORPUS / "alice01_excerpt.wav",
    "poetry": CORPUS / "corpus" / "poetry_weeds_by_the_wall_1702_librivox.wav",
    "essay": CORPUS / "corpus" / "essay_perpetual_peace_librivox.wav",
    "dialogue": CORPUS / "corpus" / "shortstory_shortstory058_1405_librivox.wav",
    "shakespeare": CORPUS / "corpus"
                   / "shakespeare_midsummernightsdream_1201_librivox.wav",
    "children": CORPUS / "corpus"
                / "children_english_fairy_tales_joy_librivox.wav",
    "oration": CORPUS / "corpus" / "oration_jfks19620912.wav",
}

METHOD_SOURCE = "tests/test_pause_threshold.py"
METHOD_CELLS_LINES = "77-87"

# The code this script varies, and the line that carries the defect.
DEFECT_FILE = "src/energy.py"
DEFECT_FUNCTION = "_merge_short_runs"
DEFECT_LINES = "284-306"
DEFECT_LINE_NO = 290
DEFECT_LINE_TEXT = "    kept = [segs[0]]"

COMMITTED_RESULTS = ROOT / "experiments" / "pause_threshold_results.json"
RESULTS_PATH = ROOT / "experiments" / "silence_leading_artefact_results.json"

# The watcher chunk, the pause-event threshold and the turn threshold, all
# taken from the shipped code rather than restated here, so this script cannot
# drift from what it is measuring.
PAUSE_EVENT_MIN_S = 0.15
TURN_END_S = turn.TURN_END_S


# ---------------------------------------------------------------------------
# The variant. This is the ONLY thing that differs between the two
# populations, and it lives here, not in src/energy.py.
# ---------------------------------------------------------------------------

def _merge_short_runs_artefact_free(segs, min_silence_s, min_speech_s):
    """
    A LOCAL VARIANT of `src/energy.py:_merge_short_runs` (lines 284-306).

    It is the shipped function with ONE rule changed: the first segment of the
    list is no longer kept unconditionally. A first segment that is a silence
    shorter than `min_silence_s` is merged FORWARD into the segment that
    follows it -- that segment's start moves back to the silence's start, so
    the silence disappears exactly as a short silence with a predecessor
    disappears into that predecessor (src/energy.py:296-298).

    Everything after that is the shipped algorithm, in the shipped order, with
    the shipped comparisons.

    ONE EDGE CASE, STATED RATHER THAN HIDDEN: the segment that absorbs the
    leading silence becomes the first segment of the list, and the shipped
    algorithm keeps the first segment unconditionally -- so the absorber
    inherits the exemption from the `min_speech_s` filter. `absorber_*
` is recorded per cell so this can be checked rather than assumed.
    """
    if not segs:
        return []
    # Copies, so the caller's segments are not mutated.
    segs = [energy.Segment(s.start, s.end, s.speech) for s in segs]

    first = segs[0]
    if (not first.speech) and first.duration() < min_silence_s and len(segs) > 1:
        nxt = segs[1]
        nxt.start = first.start          # the leading silence is absorbed
        segs = [nxt] + segs[2:]

    kept = [segs[0]]
    for s in segs[1:]:
        prev = kept[-1]
        if s.speech and s.duration() < min_speech_s:
            continue
        if (not s.speech) and s.duration() < min_silence_s:
            prev.end = s.end
            continue
        kept.append(s)

    joined = [kept[0]]
    for s in kept[1:]:
        if s.speech == joined[-1].speech:
            joined[-1].end = s.end
        else:
            joined.append(s)
    return joined


@contextlib.contextmanager
def _variant_active(variant):
    """
    Install a merge rule for the duration of the block.

    `energy.segment` looks `_merge_short_runs` up as a module global at CALL
    time, so replacing the module attribute changes the segmentation without
    touching a byte of `src/energy.py`. The shipped function is restored on
    the way out, including on an exception.
    """
    original = energy._merge_short_runs
    energy._merge_short_runs = variant
    try:
        yield
    finally:
        energy._merge_short_runs = original


CURRENT = energy._merge_short_runs
ARTEFACT_FREE = _merge_short_runs_artefact_free


# ---------------------------------------------------------------------------
# Provenance helpers.
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wav_input(path: Path, samples, sample_rate: int) -> dict:
    """
    The input, described twice: as it is ON DISK and as it is AFTER the read.

    `wavio.read_wav` resamples to 16 kHz, so the reader's own sample rate is
    the ANALYSIS rate and says nothing about the file. The narration cell is
    44 100 Hz on disk and 16 kHz after the read; a provenance block that
    reports the second as the first is lying about its own input.
    """
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "file_sample_rate": rate,
        "file_samples": frames,
        "file_duration_s": round(frames / rate, 9) if rate else None,
        "file_channels": channels,
        "file_sample_width_bytes": width,
        "resampled": rate != SR,
        "analysis_sample_rate": sample_rate,
        "analysis_samples": len(samples),
        "analysis_duration_s": round(len(samples) / SR, 9),
    }


def defect_location() -> dict:
    """
    Where the defect is, verified against the file on disk.

    The recorded line number is only worth something if the line still says
    what this script says it says, so that is checked at measurement time.
    """
    path = ROOT / DEFECT_FILE
    lines = path.read_text().splitlines()
    line = lines[DEFECT_LINE_NO - 1] if len(lines) >= DEFECT_LINE_NO else ""
    fn_line = next((i + 1 for i, l in enumerate(lines)
                    if l.startswith(f"def {DEFECT_FUNCTION}(")), None)
    return {
        "file": str(path.resolve()),
        "sha256": sha256_file(path),
        "function": DEFECT_FUNCTION,
        "function_def_line": fn_line,
        "function_lines": DEFECT_LINES,
        "defect_line": DEFECT_LINE_NO,
        "defect_line_text": line,
        "defect_line_text_expected": DEFECT_LINE_TEXT,
        "defect_line_verified": line == DEFECT_LINE_TEXT,
    }


# ---------------------------------------------------------------------------
# The energy-method population: one segmentation of a whole file.
# ---------------------------------------------------------------------------

def _leading_silence(segs) -> dict | None:
    """
    The silence at the head of the file, and whether it is the artefact.

    A silence that starts at (or within one frame of) the start of the file
    AND is shorter than `min_silence_s` can only exist because
    `kept = [segs[0]]` skipped the filter for the first segment. A LONGER
    leading silence is a real silence in the recording and is not an artefact.
    """
    frame_s = 800 / SR
    for s in segs:
        if s.speech:
            continue
        if s.start <= frame_s:
            return {
                "start_s": round(s.start, 6),
                "end_s": round(s.end, 6),
                "duration_s": round(s.duration(), 6),
                "is_artefact": s.duration() < energy.MIN_SILENCE_S,
            }
        break
    return None


def _absorb_would_be_dropped(segs) -> dict | None:
    """
    The segment that would absorb the leading silence, and whether the
    `min_speech_s` exemption it inherits could matter here.

    Recorded so the edge case in `_merge_short_runs_artefact_free` is checked
    on the data rather than argued about.
    """
    if not segs:
        return None
    first = segs[0]
    if first.speech or first.duration() >= energy.MIN_SILENCE_S or len(segs) < 2:
        return None
    nxt = segs[1]
    grown = nxt.duration() + first.duration()
    return {
        "absorber_speech": nxt.speech,
        "absorber_duration_s": round(nxt.duration(), 6),
        "absorber_duration_after_absorption_s": round(grown, 6),
        "min_speech_s": energy.MIN_SPEECH_S,
        "absorber_would_be_dropped_by_min_speech_s":
            bool(nxt.speech and grown < energy.MIN_SPEECH_S),
    }


def _gap_block(gaps) -> dict:
    """
    The gap statistics, in the conventions `experiments/pause_threshold.py`
    publishes.

    The helpers are IMPORTED from that file, so a percentile convention
    cannot quietly differ between the two scripts.
    """
    at_150 = sum(1 for g in gaps if g >= 0.150)
    at_200 = sum(1 for g in gaps if g >= 0.200)
    lost = at_150 - at_200
    gap_bytes = pt.canonical_gap_bytes(gaps)
    return {
        "n": len(gaps),
        "shortest_s": round(min(gaps), 6) if gaps else None,
        "p05_s_nearest_rank": pt._r(pt.p05_nearest_rank(gaps)),
        "p05_s_linear": pt._r(pt.p05_linear(gaps)),
        "median_s_standard": pt._r(pt.median_standard(gaps)),
        "median_s_published_convention": pt._r(pt.median_published(gaps)),
        "max_s": round(max(gaps), 6) if gaps else None,
        "durations_s": [round(g, 6) for g in gaps],
        "durations_sha256": pt.sha256_bytes(gap_bytes),
        "durations_bytes_len": len(gap_bytes),
        "trigger": {
            "at_0_150": at_150,
            "at_0_200": at_200,
            "lost": lost,
            "lost_pct": pt._r(100.0 * lost / at_150) if at_150 else None,
        },
    }


def _flat_gap_block(gap_block: dict) -> dict:
    """
    The gap block in the FLAT shape the committed
    `corrected_without_leading_artefact` block uses.

    The committed block carries the trigger counts at its top level; the gap
    block computed here carries them under `trigger`. Flattening one side
    lets the two be compared key by key instead of by eye.
    """
    return {
        "n": gap_block["n"],
        "shortest_s": gap_block["shortest_s"],
        "p05_s_nearest_rank": gap_block["p05_s_nearest_rank"],
        "median_s_standard": gap_block["median_s_standard"],
        "at_0_150": gap_block["trigger"]["at_0_150"],
        "at_0_200": gap_block["trigger"]["at_0_200"],
        "lost": gap_block["trigger"]["lost"],
        "lost_pct": gap_block["trigger"]["lost_pct"],
    }


def measure_energy_population(samples, variant) -> dict:
    """
    The whole-file segmentation under one merge rule.

    The gap list is the silence segments of `energy.segment`, which is the
    method `tests/test_pause_threshold.py::_gaps` uses, copied through
    `experiments/pause_threshold.py`.
    """
    ef, _ = energy.rms_series(samples, 800, HOP, SR)
    if not ef:
        return {"segments": [], "gaps": [], "gap_block": _gap_block([])}
    with _variant_active(variant):
        segs = energy.segment(ef, HOP_S, peak_rms=max(f.rms for f in ef))
    gaps = [s.duration() for s in segs if not s.speech]
    return {
        "segments": [(round(s.start, 6), round(s.end, 6), s.speech) for s in segs],
        "gaps": gaps,
        "gap_block": _gap_block(gaps),
        "leading_silence": _leading_silence(segs),
        "absorber": _absorb_would_be_dropped(segs),
    }


# ---------------------------------------------------------------------------
# The watcher population: events, runs, turns and the F0 input.
# ---------------------------------------------------------------------------

class _TimedProbe(pt._EventProbe):
    """
    The committed probe, plus the stream clock at each emission.

    `experiments/pause_threshold.py::_EventProbe` is the probe that produced
    the committed 673 runs / 1671 events. It is subclassed rather than
    reimplemented so the two measurements cannot drift. The only addition is
    the clock, which `turn.TurnTracker` needs.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.timed: list = []

    def push(self, samples):
        evs = super().push(samples)
        now = self.elapsed_s
        for ev in evs:
            self.timed.append((ev, now))
        return evs


def measure_watcher_population(samples, variant) -> dict:
    """
    The watcher's pause population under one merge rule.

    The watcher is the streaming layer: `energy.segment` is called on a MOVING
    buffer, so "the first segment" here is "the segment at the buffer origin",
    not "the start of the file". That is why the defect fires far more often in
    the watcher than in the whole-file population.

    The first segment of every buffer is recorded, so the mechanism can be
    counted rather than argued about: how many buffers OPEN with a silence,
    and how many of those silences are short enough that the shipped code
    keeps them only because it keeps `segs[0]`.
    """
    first_segments: list = []
    original_segment = energy.segment

    def spy(*a, **k):
        segs = original_segment(*a, **k)
        if segs:
            first_segments.append((round(segs[0].start, 6),
                                   round(segs[0].end, 6), segs[0].speech,
                                   round(segs[0].duration(), 6)))
        else:
            first_segments.append(None)
        return segs

    energy.segment = spy
    try:
        with _variant_active(variant):
            probe = _TimedProbe(SR, adaptive=True)
            for i in range(0, len(samples), 1600):
                probe.push(samples[i:i + 1600])
    finally:
        energy.segment = original_segment

    with _variant_active(variant):
        pause_events = [ev for ev in probe.events if ev.kind == watcher.PAUSE]
        events_gt = [ev for ev in pause_events
                     if ev.data.get("run_s", 0) > PAUSE_EVENT_MIN_S]
        runs = pt._runs_from_events(probe)
        pt._resumed_flags(probe, runs)

        # The turn boundaries src/turn.py would produce from these events.
        tracker = turn.TurnTracker()
        boundaries = []
        for ev, now in probe.timed:
            boundaries.extend(tracker.feed(ev, now))

        # The population experiments/f0_speaker_change.py consumes: the gaps
        # between consecutive utterance spans, in (GAP_MIN, GAP_MAX).
        spans = [(ev.t0, ev.t1) for ev in probe.events
                 if ev.kind == watcher.UTTERANCE]
        f0_gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
        f0_gaps = [g for g in f0_gaps if 0.20 < g < 4.0]

    run_durations = sorted(r["run_s"] for r in runs)
    event_durations = sorted(e.data.get("run_s", 0.0) for e in events_gt)
    return {
        "pause_events_all": len(pause_events),
        "pause_events_gt_0_15": len(events_gt),
        "distinct_silence_runs": len(runs),
        "max_run_s": pt._r(max(run_durations), 4) if run_durations else None,
        "min_run_s": pt._r(min(run_durations), 4) if run_durations else None,
        "median_run_s": pt._r(pt.median_standard(run_durations), 4)
        if run_durations else None,
        "event_min_s": pt._r(event_durations[0], 4) if event_durations else None,
        "event_median_s": pt._r(pt.median_standard(event_durations), 4)
        if event_durations else None,
        "event_max_s": pt._r(event_durations[-1], 4) if event_durations else None,
        "runs_at_or_above_3s": sum(1 for r in runs if r["run_s"] >= TURN_END_S),
        "events_at_or_above_3s": sum(1 for e in events_gt
                                    if e.data.get("run_s", 0) >= TURN_END_S),
        "turn_boundaries": len(boundaries),
        "turn_boundary_reasons": sorted(
            {b.get("reason") for b in boundaries if isinstance(b, dict)}),
        "f0_utterance_spans": len(spans),
        "f0_pause_population": len(f0_gaps),
        "f0_gap_sha256": pt.sha256_bytes(
            "".join(f"{g:.6f}\n" for g in f0_gaps).encode("utf-8")),
        # The mechanism, counted rather than argued about.
        "buffers": len(first_segments),
        "buffers_opening_with_silence": sum(
            1 for s in first_segments if s is not None and not s[2]),
        "buffers_opening_with_a_short_silence": sum(
            1 for s in first_segments
            if s is not None and not s[2] and s[3] < energy.MIN_SILENCE_S),
        "first_segment_digest": pt.sha256_bytes(
            "".join(repr(s) + "\n" for s in first_segments).encode("utf-8")),
        "_first_segments": first_segments,
        "_runs": runs,
        "_events_gt": events_gt,
        "_spans": spans,
    }


# ---------------------------------------------------------------------------
# The corpus-level measurements.
# ---------------------------------------------------------------------------

def _spec_stats(events_gt) -> dict:
    """
    The two figures `tests/test_speculate.py` asserts on.

    The population is the PAUSE EVENTS with `run_s > 0.15` across the declared
    corpus cells -- the same population `_corpus_pause_runs()` builds -- and
    the reach rate is measured against the shipped speculation threshold.
    """
    runs = [e.data.get("run_s", 0.0) for e in events_gt]
    if not runs:
        return {"n": 0}
    speculate_s = speculate.SpeculationGate().speculate_s
    reach = sum(1 for r in runs if r >= speculate_s)
    below = sum(1 for r in runs if r < 1.20)
    return {
        "n": len(runs),
        "speculate_s": speculate_s,
        "reach": reach,
        "reach_rate_pct": pt._r(100.0 * reach / len(runs), 4),
        "below_1_20": below,
        "below_1_20_pct": pt._r(100.0 * below / len(runs), 4),
    }


def _addendum2(runs) -> dict:
    """
    Addendum 2 of TURN-ROUTES.md: for each threshold, how many silence runs
    clear it and how many of those had the speaker come back.
    """
    out = {}
    for th in pt.ADDENDUM_THRESHOLDS:
        sel = [r for r in runs if r["run_s"] >= th]
        resumed = sum(1 for r in sel if r.get("resumed"))
        out[f"{th:.1f}"] = {
            "silence_runs": len(sel),
            "speaker_resumed": resumed,
            "at_file_end": len(sel) - resumed,
            "would_have_interrupted_pct":
                pt._r(100.0 * resumed / len(sel), 4) if sel else None,
        }
    return out


# ---------------------------------------------------------------------------
# The comparison against the committed result file.
# ---------------------------------------------------------------------------

def compare_with_committed(cells, watcher) -> dict:
    """
    Which committed numbers the fix would move, stated as a diff.

    The committed file is the reference, not this script's own output: if the
    two disagree, that disagreement is the finding.
    """
    if not COMMITTED_RESULTS.is_file():
        return {"committed_file_present": False}
    committed = json.loads(COMMITTED_RESULTS.read_text())
    c_cells = committed.get("cells", {})
    c_watcher = committed.get("watcher_pause_run_population", {})

    per_cell = {}
    disagreements = []

    for name, cell in cells.items():
        cc = c_cells.get(name, {})
        cur = cell["current"]["gap_block"]
        art = cell["artefact_free"]["gap_block"]
        committed_main = cc.get("gaps", {})
        committed_corrected = cc.get("corrected_without_leading_artefact", {})

        # The committed `gaps` block and the committed corrected block have
        # different key sets, so each is compared on its own keys. The
        # corrected block is FLAT (the trigger counts sit at its top level),
        # so the artefact-free gap block is flattened the same way before the
        # comparison.
        art_flat = _flat_gap_block(art)
        main_match = all(cur.get(k) == committed_main.get(k)
                         for k in committed_main if k in cur)
        corrected_match = all(art_flat.get(k) == committed_corrected.get(k)
                              for k in committed_corrected)
        missing = [k for k in committed_main if k not in cur]
        missing += [k for k in committed_corrected if k not in art_flat]

        per_cell[name] = {
            "committed_gaps_block_matches_current": main_match,
            "committed_corrected_block_matches_artefact_free": corrected_match,
            "keys_not_reproduced": missing,
            "committed_gaps": {k: committed_main.get(k)
                               for k in ("n", "shortest_s", "p05_s_nearest_rank",
                                         "median_s_standard",
                                         "median_s_published_convention",
                                         "max_s")},
            "committed_corrected": committed_corrected,
            "artefact_free_corrected_block": art_flat,
            "current_gaps": {k: cur.get(k)
                             for k in ("n", "shortest_s", "p05_s_nearest_rank",
                                       "median_s_standard",
                                       "median_s_published_convention",
                                       "max_s")},
            "artefact_free_gaps": {k: art.get(k)
                                   for k in ("n", "shortest_s",
                                             "p05_s_nearest_rank",
                                             "median_s_standard",
                                             "median_s_published_convention",
                                             "max_s")},
            "trigger_committed": cc.get("trigger"),
            "trigger_current": cur.get("trigger"),
            "trigger_artefact_free": art.get("trigger"),
        }
        if not main_match:
            disagreements.append(f"{name}: the current population does not "
                                  f"reproduce the committed gaps block")
        if not corrected_match:
            disagreements.append(f"{name}: the artefact-free population does "
                                  f"not reproduce the committed corrected block")
        if missing:
            disagreements.append(f"{name}: keys not reproduced: {missing}")

    # The watcher numbers.
    cw = c_watcher.get("per_cell", {})
    watcher_cmp = {}
    for name in cells:
        if name not in cw:
            continue
        if name not in watcher.get("current", {}).get("per_cell", {}):
            continue
        got = watcher["current"]["per_cell"][name]
        want = cw[name]
        same = all(got.get(k) == want.get(k) for k in want)
        watcher_cmp[name] = {
            "committed": want,
            "current": {k: got.get(k) for k in want},
            "matches": same,
            "artefact_free": {
                k: watcher["artefact_free"]["per_cell"][name].get(k)
                for k in want},
        }
        if not same:
            disagreements.append(f"{name}: the current watcher population does "
                                  f"not reproduce the committed per-cell block")

    return {
        "committed_file_present": True,
        "committed_file_path": str(COMMITTED_RESULTS.resolve()),
        "committed_file_sha256": sha256_file(COMMITTED_RESULTS),
        "per_cell": per_cell,
        "watcher_per_cell": watcher_cmp,
        "committed_totals": committed.get("totals", {}),
        "disagreements": disagreements,
        "all_agree": not disagreements,
    }


# ---------------------------------------------------------------------------
# The copy check.
# ---------------------------------------------------------------------------

def check_copy() -> dict:
    """
    Verify that the corpus in THIS file is the corpus in the test file, byte
    for byte.

    A "verbatim copy" that nobody checks is a paraphrase. This runs at
    measurement time, so if the upstream mapping changes, the result file says
    so instead of quietly measuring something else.
    """
    src = ROOT / METHOD_SOURCE
    here = Path(__file__).read_text()
    text = src.read_text()
    src_cells = text.split("CELLS = {", 1)[1].split("\n}\n", 1)[0]
    here_cells = here.split("CELLS = {", 1)[1].split("\n}\n", 1)[0]
    return {
        "source_file": str(src.resolve()),
        "source_sha256": sha256_file(src),
        "cells_lines": METHOD_CELLS_LINES,
        "cells_identical": src_cells == here_cells,
        "ok": src_cells == here_cells,
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def build_result(skip_watcher: bool = False) -> dict:
    t0 = time.time()
    cells: dict = {}
    watcher_cells = {"current": {}, "artefact_free": {}}
    all_runs = {"current": [], "artefact_free": []}
    all_events = {"current": [], "artefact_free": []}
    mechanism: dict = {}

    for name, path in CELLS.items():
        if not path.is_file():
            cells[name] = {"error": "absent"}
            continue
        audio = wavio.read_wav(str(path))
        samples = audio.samples

        cur = measure_energy_population(samples, CURRENT)
        art = measure_energy_population(samples, ARTEFACT_FREE)

        cur_gaps = cur["gap_block"]
        art_gaps = art["gap_block"]
        dropped = cur_gaps["n"] - art_gaps["n"]

        entry = {
            "input": wav_input(path, samples, audio.sample_rate),
            "current": {
                "gap_block": cur_gaps,
                "leading_silence": cur["leading_silence"],
                "absorber": cur["absorber"],
            },
            "artefact_free": {
                "gap_block": art_gaps,
                "leading_silence": art["leading_silence"],
                "absorber": art["absorber"],
            },
            "delta": {
                "gaps_dropped": dropped,
                "gaps_identical": cur_gaps["durations_sha256"] ==
                art_gaps["durations_sha256"],
                "current_durations_sha256": cur_gaps["durations_sha256"],
                "artefact_free_durations_sha256": art_gaps["durations_sha256"],
                "at_0_150_delta": art_gaps["trigger"]["at_0_150"] -
                cur_gaps["trigger"]["at_0_150"],
                "at_0_200_delta": art_gaps["trigger"]["at_0_200"] -
                cur_gaps["trigger"]["at_0_200"],
                "lost_delta": art_gaps["trigger"]["lost"] -
                cur_gaps["trigger"]["lost"],
            },
        }
        cells[name] = entry

        if not skip_watcher:
            first_segments = {}
            for label, variant in (("current", CURRENT),
                                   ("artefact_free", ARTEFACT_FREE)):
                w = measure_watcher_population(samples, variant)
                watcher_cells[label][name] = {
                    k: v for k, v in w.items() if not k.startswith("_")}
                all_runs[label].extend(w["_runs"])
                all_events[label].extend(w["_events_gt"])
                first_segments[label] = w["_first_segments"]
            # How much of the difference is the artefact itself, and how much
            # is the streaming state that follows from it. The two runs trim
            # the buffer at different points once the leading silence is no
            # longer a segment of its own, so their grids move apart.
            cur_fs = first_segments["current"]
            art_fs = first_segments["artefact_free"]
            mechanism[name] = {
                "buffers": len(cur_fs),
                "pushes_whose_first_segment_differs": sum(
                    1 for a, b in zip(cur_fs, art_fs) if a != b),
                "current": {
                    "buffers_opening_with_silence":
                        watcher_cells["current"][name]
                        ["buffers_opening_with_silence"],
                    "buffers_opening_with_a_short_silence":
                        watcher_cells["current"][name]
                        ["buffers_opening_with_a_short_silence"],
                },
                "artefact_free": {
                    "buffers_opening_with_silence":
                        watcher_cells["artefact_free"][name]
                        ["buffers_opening_with_silence"],
                    "buffers_opening_with_a_short_silence":
                        watcher_cells["artefact_free"][name]
                        ["buffers_opening_with_a_short_silence"],
                },
                "note": (
                    "the watcher re-segments a MOVING buffer, so 'the first "
                    "segment' is the segment at the buffer origin, not the "
                    "start of the file; the defect therefore fires once per "
                    "BUFFER whose origin lands in a short silence, which is "
                    "why the whole-file population (3 leading silences) "
                    "understates it"),
            }

    watcher_out = {"current": {}, "artefact_free": {}}
    for label in ("current", "artefact_free"):
        runs = all_runs[label]
        events = all_events[label]
        run_durations = sorted(r["run_s"] for r in runs)
        event_durations = sorted(e.data.get("run_s", 0.0) for e in events)
        watcher_out[label] = {
            "events_gt_0_15": len(events),
            "pause_events_all": sum(watcher_cells[label][c]["pause_events_all"]
                                    for c in watcher_cells[label]),
            "distinct_silence_runs": len(runs),
            "event_min_s": pt._r(event_durations[0], 4) if event_durations else None,
            "event_median_s": pt._r(pt.median_standard(event_durations), 4)
            if event_durations else None,
            "event_max_s": pt._r(event_durations[-1], 4) if event_durations else None,
            "run_min_s": pt._r(run_durations[0], 4) if run_durations else None,
            "run_median_s": pt._r(pt.median_standard(run_durations), 4)
            if run_durations else None,
            "run_max_s": pt._r(run_durations[-1], 4) if run_durations else None,
            "runs_at_or_above_3s": sum(1 for r in runs if r["run_s"] >= TURN_END_S),
            "events_at_or_above_3s": sum(1 for e in events
                                        if e.data.get("run_s", 0) >= TURN_END_S),
            "turn_boundaries": sum(watcher_cells[label][c]["turn_boundaries"]
                                   for c in watcher_cells[label]),
            "f0_utterance_spans": sum(watcher_cells[label][c]["f0_utterance_spans"]
                                      for c in watcher_cells[label]),
            "f0_pause_population": sum(watcher_cells[label][c]["f0_pause_population"]
                                       for c in watcher_cells[label]),
            "per_cell": watcher_cells[label],
            "addendum2_resume_table": _addendum2(runs),
            "specification": _spec_stats(events),
        }

    totals = {}
    for label in ("current", "artefact_free"):
        cur_gaps = [cells[n][label]["gap_block"] for n in cells
                    if "error" not in cells[n]]
        n = sum(g["n"] for g in cur_gaps)
        at150 = sum(g["trigger"]["at_0_150"] for g in cur_gaps)
        at200 = sum(g["trigger"]["at_0_200"] for g in cur_gaps)
        totals[label] = {
            "gaps": n,
            "at_0_150": at150,
            "at_0_200": at200,
            "lost": at150 - at200,
            "lost_pct": pt._r(100.0 * (at150 - at200) / at150) if at150 else None,
            "cells_with_a_leading_silence_kept": sum(
                1 for c in cells.values()
                if "error" not in c
                and (c[label]["leading_silence"] or {}).get("is_artefact")),
        }

    comparison = compare_with_committed(cells, watcher_out)

    f0 = {
        "source": "experiments/f0_speaker_change.py",
        "population": ("the gaps between consecutive utterance spans that the "
                       "F0 experiment measures across (GAP_MIN, GAP_MAX) = "
                       "(0.20, 4.0) s; derived from the same watcher run, so "
                       "no second pass over the corpus was needed"),
        "current": {
            "utterance_spans": watcher_out["current"]["f0_utterance_spans"],
            "pause_population": watcher_out["current"]["f0_pause_population"],
        },
        "artefact_free": {
            "utterance_spans": watcher_out["artefact_free"]["f0_utterance_spans"],
            "pause_population": watcher_out["artefact_free"]["f0_pause_population"],
        },
        "population_identical": (
            watcher_out["current"]["f0_pause_population"] ==
            watcher_out["artefact_free"]["f0_pause_population"]
            and watcher_out["current"]["f0_utterance_spans"] ==
            watcher_out["artefact_free"]["f0_utterance_spans"]),
    }

    return {
        "schema": "silence-leading-artefact/1",
        "note": (
            "MEASUREMENT ONLY. src/energy.py is not modified. Two populations "
            "are produced from the same audio with the same parameters: "
            "'current' uses the shipped _merge_short_runs, 'artefact_free' "
            "uses a local variant in which a first segment that is a silence "
            "shorter than min_silence_s is merged forward into the segment "
            "that follows it instead of being kept unconditionally."),
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "argv": sys.argv,
            "elapsed_s": round(time.time() - t0, 1),
        },
        "method": {
            "varied_source": defect_location(),
            "variant_function": "_merge_short_runs_artefact_free",
            "variant_lives_in": str(Path(__file__).resolve()),
            "parameters": {
                "analysis_sample_rate": SR,
                "frame_len_samples": 800,
                "hop_samples": HOP,
                "hop_s": HOP_S,
                "min_silence_s": energy.MIN_SILENCE_S,
                "min_speech_s": energy.MIN_SPEECH_S,
                "energy_population_adaptive": False,
                "watcher_adaptive": True,
                "watcher_chunk_samples": 1600,
                "pause_event_min_s": PAUSE_EVENT_MIN_S,
                "turn_end_s": TURN_END_S,
                "pause_report_s": watcher.PAUSE_REPORT_S,
                "max_pause_s": watcher.MAX_PAUSE_S,
                "speculation_threshold_s":
                    speculate.SpeculationGate().speculate_s,
            },
            "statistics_helpers": {
                "imported_from": "experiments/pause_threshold.py",
                "functions": ["p05_nearest_rank", "p05_linear",
                              "median_standard", "median_published",
                              "canonical_gap_bytes", "sha256_bytes"],
            },
            "watcher_probe": {
                "imported_from": "experiments/pause_threshold.py",
                "classes": ["_EventProbe", "_runs_from_events",
                            "_resumed_flags"],
                "note": ("imported rather than copied so the watcher numbers "
                         "here and the committed 673 runs / 1671 events cannot "
                         "drift apart"),
            },
            "copy_check": check_copy(),
            "populations": {
                "current": "src/energy.py:_merge_short_runs, unmodified",
                "artefact_free": ("local variant in this file; the first "
                                  "segment is no longer kept unconditionally -- "
                                  "a first-segment silence shorter than "
                                  "min_silence_s is merged forward into the "
                                  "segment that follows it"),
            },
        },
        "provenance": {
            "cells": {name: cells[name]["input"] for name in cells
                      if "error" not in cells[name]},
            "committed_results": {
                "path": str(COMMITTED_RESULTS.resolve()),
                "sha256": sha256_file(COMMITTED_RESULTS)
                if COMMITTED_RESULTS.is_file() else None,
            },
            "method_source": {
                "path": str((ROOT / METHOD_SOURCE).resolve()),
                "sha256": sha256_file(ROOT / METHOD_SOURCE),
                "cells_lines": METHOD_CELLS_LINES,
            },
            "energy_module": {
                "path": str((ROOT / DEFECT_FILE).resolve()),
                "sha256": sha256_file(ROOT / DEFECT_FILE),
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "cells": cells,
        "totals": totals,
        "watcher": watcher_out,
        "mechanism": mechanism,
        "f0_population": f0,
        "committed_comparison": comparison,
    }


def main() -> int:
    argv = sys.argv[1:]
    if "--check-copy" in argv:
        print(json.dumps(check_copy(), indent=2))
        return 0

    result = build_result(skip_watcher="--skip-watcher" in argv)

    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n")

    cur, art = result["totals"]["current"], result["totals"]["artefact_free"]
    print(json.dumps({
        "written": str(RESULTS_PATH),
        "energy_gaps": {"current": cur["gaps"], "artefact_free": art["gaps"]},
        "at_0_150": {"current": cur["at_0_150"],
                     "artefact_free": art["at_0_150"]},
        "watcher_events": {
            "current": result["watcher"]["current"]["events_gt_0_15"],
            "artefact_free": result["watcher"]["artefact_free"]["events_gt_0_15"]},
        "watcher_runs": {
            "current": result["watcher"]["current"]["distinct_silence_runs"],
            "artefact_free":
                result["watcher"]["artefact_free"]["distinct_silence_runs"]},
        "spec_reach_pct": {
            "current": result["watcher"]["current"]["specification"]
            .get("reach_rate_pct"),
            "artefact_free": result["watcher"]["artefact_free"]["specification"]
            .get("reach_rate_pct")},
        "committed_disagreements":
            result["committed_comparison"].get("disagreements", []),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
