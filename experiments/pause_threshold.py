"""
pause_threshold.py -- re-measure the inter-utterance pause statistics on the
corpus that is actually on disk, and persist them with provenance.

WHY THIS EXISTS.

`docs/research/ADOPTION-PLAN.md` §3 and `docs/research/TURN-ROUTES.md` carry a
pause-statistics table used to argue that raising the VAD trigger from 150 ms
to 200 ms destroys 17.8% of real pauses. Those numbers lived ONLY in a test
docstring (`tests/test_pause_threshold.py`) and in prose. There was no result
file. A number that cannot be traced to its inputs is the exact defect this
repository is trying to eliminate, so this script exists to make the number
traceable, and to record whether the number is still true.

THE PROSE SAID THE CORPUS WAS 465 s. IT IS NOT.

    measured on disk, seven declared cells: 485.011125 s
      narration    90.000000   poetry     20.011125
      essay        75.000000   dialogue   75.000000
      shakespeare  75.000000   children   75.000000
      oration      75.000000

The 20 s discrepancy is exactly the poetry cell, which is 20.011 s on disk and
not 15 s (`NOTES-AUDIT.md` F7). So the published table was quoted against a
population whose SIZE was misstated. Whether the table's COUNTS were measured
on that population is a separate question, and it is the one this script
answers by re-measuring rather than by reasoning about it.

THE METHOD IS COPIED, NOT REINVENTED.

`_gaps()` and `CELLS` below are copied VERBATIM from
`tests/test_pause_threshold.py` (the `_gaps` helper at lines 90-99, the
`CELLS` mapping at lines 77-87). A different gap definition would produce a
different number, and a different number would be a DIFFERENT MEASUREMENT, not
a re-measurement. The copy is marked with its source line range so a reader
can check the two files against each other.

`--check-copy` verifies the copy byte-for-byte against the source file at
run time, so the two cannot drift apart silently.

ONE DEFECT WAS FOUND IN THE COPIED METHOD, AND IT IS MEASURED BOTH WAYS.

`src/energy.py:_merge_short_runs` keeps `segs[0]` unconditionally:

    kept = [segs[0]]                       # src/energy.py:290

so the FIRST segment of a file escapes the `min_silence_s` filter entirely. On
this corpus that is the leading silence before the first speech, and it is
what produces the published table's "shortest gap 0.110 s" for shakespeare and
children and 0.130 s for dialogue. Those are not inter-utterance pauses; they
are the silence before the recording starts.

This script does NOT change the definition. It measures the copied method
verbatim (authoritative), and ALSO measures the same method with the leading
artefact removed (sensitivity), and reports both. The headline question --
does 17.8% survive? -- is answered on the verbatim population and checked
against the corrected one.

PROVENANCE.

Every number in `experiments/pause_threshold_results.json` carries the
absolute path and SHA-256 of every input wav, the sample rate, the exact
gap-detection parameters, the SHA-256 of this script, and the source file and
line of the method. The result file is regenerated, never hand-edited:

    /home/proart/projects/glasses/.venv/bin/python experiments/pause_threshold.py

    # or, to verify the copy against its source:
    /home/proart/projects/glasses/.venv/bin/python experiments/pause_threshold.py --check-copy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import energy, watcher, wavio                    # noqa: E402

# ---------------------------------------------------------------------------
# The corpus, and the method. COPIED VERBATIM from tests/test_pause_threshold.py
# (lines 69-99). Do not "improve" these: a different definition is a different
# measurement, and the whole point of this script is that the number is
# traceable to one specific definition.
# ---------------------------------------------------------------------------

SR = 16000
HOP = 160
HOP_S = HOP / SR
CORPUS = ROOT / "examples" / "realaudio"

# The seven cells from examples/realaudio/corpus/MANIFEST.md. Held out from
# every threshold decision made before the corpus existed, which is what makes
# them a test rather than a demonstration.
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
METHOD_GAPS_LINES = "90-99"
METHOD_CELLS_LINES = "77-87"


class _PytestShim:
    """
    A stand-in for the `pytest` module, so the copied `_gaps()` below can be
    BYTE-IDENTICAL to the one in `tests/test_pause_threshold.py` without this
    script taking a dependency on a test framework.

    `skip` in a test means "do not run this"; in a script the honest
    equivalent is to stop with the reason attached, so that is what it does.
    """

    @staticmethod
    def skip(reason: str):
        raise FileNotFoundError(reason)


pytest = _PytestShim()


def _gaps(path: Path):
    """Every silence run the segmenter finds, in seconds."""
    if not path.is_file():
        pytest.skip(f"corpus cell absent: {path.name}")
    samples = wavio.read_wav(str(path)).samples
    ef, _ = energy.rms_series(samples, 800, HOP, SR)
    if not ef:
        pytest.skip(f"{path.name} produced no frames")
    segs = energy.segment(ef, HOP_S, peak_rms=max(f.rms for f in ef))
    return [s.duration() for s in segs if not s.speech]


def _gap_segments(path: Path):
    """
    The same segmentation as `_gaps()`, but returning ALL the segments (with
    their start and end) instead of bare durations.

    DIAGNOSTIC ONLY. Every published number in this script comes from
    `_gaps()`; this exists so the leading-silence artefact and the
    "did the speaker resume" question can be answered rather than assumed.
    The durations of the silence segments are asserted to equal the ones
    `_gaps()` returns, so the diagnostic cannot drift from the measurement.
    """
    audio = wavio.read_wav(str(path))
    ef, _ = energy.rms_series(audio.samples, 800, HOP, SR)
    if not ef:
        raise ValueError(f"{path.name} produced no frames")
    segs = energy.segment(ef, HOP_S, peak_rms=max(f.rms for f in ef))
    return audio, segs


# ---------------------------------------------------------------------------
# The published table, transcribed so the diff is mechanical rather than
# editorial. Source: docs/research/ADOPTION-PLAN.md:124-145 (also
# docs/research/TURN-ROUTES.md, and the docstring of
# tests/test_pause_threshold.py:13-36).
# ---------------------------------------------------------------------------

PUBLISHED = {
    # cell: (gaps, shortest, p05, median, at_0_150, at_0_200, lost, lost_pct)
    "narration":   (45, 0.170, 0.210, 0.640, 45, 43,  2,  4.4),
    "poetry":      ( 7, 0.150, 0.150, 0.510,  7,  6,  1, 14.3),
    "essay":       (33, 0.150, 0.160, 0.760, 33, 24,  9, 27.3),
    "dialogue":    (35, 0.130, 0.160, 0.450, 34, 30,  4, 11.8),
    "shakespeare": (62, 0.110, 0.160, 0.380, 61, 46, 15, 24.6),
    "children":    (78, 0.110, 0.160, 0.300, 77, 62, 15, 19.5),
    "oration":     ( 2, 0.310, 0.310, 0.380,  2,  2,  0,  0.0),
}
PUBLISHED_TOTAL = (259, 213, 46, 17.8)
PUBLISHED_CORPUS_S = 465.0
AUDIT_CORPUS_S = 485.01

# The pause-run population quoted in prose, and the one the repo measures.
PROSE_PAUSE_RUNS = 680
REPO_PAUSE_RUNS = 1671

# The thresholds of Addendum 2 in docs/research/TURN-ROUTES.md, re-measured
# here on a stated population rather than inherited.
ADDENDUM_THRESHOLDS = (0.6, 0.9, 1.2, 1.5, 1.8, 2.1)


# ---------------------------------------------------------------------------
# Statistics. The conventions are STATED, because a percentile is a choice and
# an unstated choice is how a table stops meaning anything.
# ---------------------------------------------------------------------------

def p05_nearest_rank(values):
    """
    p05 as the PUBLISHED table computes it: nearest-rank, index
    ceil(q*n) - 1. No interpolation, so the result is a value that occurred.

    This is the convention that reproduces the published table exactly: on
    poetry (n=7) linear interpolation gives 0.255 where the published table
    says 0.150, and on essay (n=33) 0.166 against the published 0.160. It is
    recorded in the result file so the number can be recomputed.
    """
    if not values:
        return None
    s = sorted(values)
    k = min(len(s) - 1, max(0, math.ceil(0.05 * len(s)) - 1))
    return s[k]


def p05_linear(values):
    """p05 by linear interpolation between order statistics (numpy default)."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = 0.05 * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(len(s) - 1, lo + 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def median_standard(values):
    """The conventional median: the mean of the two middle values if n is even."""
    if not values:
        return None
    return statistics.median(values)


def median_published(values):
    """
    The median as the PUBLISHED table computes it: sorted[n // 2], i.e. the
    UPPER middle value for an even n.

    It is identical to the standard median for every odd n, which is six of
    the seven cells. It differs on shakespeare (0.380 published against 0.375
    standard) and on oration (0.380 published against 0.345 standard) -- and
    oration is the only cell with two gaps, so the two conventions cannot
    disagree anywhere else. Recorded, not silently adopted.
    """
    if not values:
        return None
    s = sorted(values)
    return s[len(s) // 2]


def canonical_gap_bytes(values) -> bytes:
    """
    The EXACT bytes of a gap list, so a reader can hash a gap list and get
    the same digest the result file records.

    Format: one gap per line, "%.6f", UTF-8, LF separated, with a trailing
    LF. Six decimals is 1 microsecond, which is finer than the 10 ms frame
    grid the gaps are measured on, so no information is lost.
    """
    return "".join(f"{v:.6f}\n" for v in values).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wav_header(path: Path) -> dict:
    """
    The WAV file's OWN header: the rate, frame count and duration as they
    are ON DISK.

    This is not the same as what `wavio.read_wav` returns, because that
    reader resamples to the analysis rate. The narration cell is 44 100 Hz
    and 3 969 000 frames on disk and 16 000 Hz and 1 440 000 samples after
    the read; a provenance block that reports the second as the first is
    lying about its input.
    """
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
    return {
        "sample_rate": rate,
        "frames": frames,
        "duration_s": round(frames / rate, 9) if rate else None,
        "channels": channels,
        "sample_width_bytes": width,
    }


# ---------------------------------------------------------------------------
# Per-cell measurement.
# ---------------------------------------------------------------------------

def measure_cell(name: str, path: Path) -> dict:
    gaps = _gaps(path)
    audio, segs = _gap_segments(path)

    # The gap list from the verbatim helper and from the diagnostic must
    # agree, or the diagnostic is measuring something else.
    diag_durations = [s.duration() for s in segs if not s.speech]
    if [round(g, 9) for g in gaps] != [round(d, 9) for d in diag_durations]:
        raise AssertionError(
            f"{name}: _gaps() and _gap_segments() disagree on the gap list; "
            "the diagnostic and the measurement have diverged")

    n = len(gaps)
    at_150 = sum(1 for g in gaps if g >= 0.150)
    at_200 = sum(1 for g in gaps if g >= 0.200)
    lost = at_150 - at_200

    # The leading-silence artefact: a silence segment that starts at (or
    # within one frame of) the start of the file AND is shorter than
    # min_silence_s can only exist because `kept = [segs[0]]` in
    # src/energy.py:290 skipped the filter for the first segment.
    frame_s = 800 / SR
    silences = [(i, s) for i, s in enumerate(segs) if not s.speech]
    leading = None
    for _, s in silences:
        if s.start <= frame_s and s.duration() < energy.MIN_SILENCE_S:
            leading = {"start_s": round(s.start, 6), "end_s": round(s.end, 6),
                       "duration_s": round(s.duration(), 6)}
            break

    # "Did the speaker come back?" on the ENERGY population. The segments
    # alternate speech/silence, so a silence that is not the last segment of
    # the file is followed by speech; the last one ran to the end of the
    # recording and is ambiguous (the recording simply stopped).
    resumed = []
    at_file_end = []
    for i, s in silences:
        (resumed if i + 1 < len(segs) else at_file_end).append(s.duration())

    addendum = {}
    for th in ADDENDUM_THRESHOLDS:
        sel = [d for d in resumed if d >= th]
        end = [d for d in at_file_end if d >= th]
        n_runs = len(sel) + len(end)
        addendum[f"{th:.1f}"] = {
            "silence_runs": n_runs,
            "speaker_resumed": len(sel),
            "at_file_end": len(end),
            "would_have_interrupted_pct":
                _r(100.0 * len(sel) / n_runs, 4) if n_runs else None,
        }

    # Sensitivity: the same measurement with the leading artefact removed.
    corrected = list(gaps)
    if leading is not None and gaps and abs(gaps[0] - leading["duration_s"]) < 1e-9:
        corrected = gaps[1:]

    c_at_150 = sum(1 for g in corrected if g >= 0.150)
    c_at_200 = sum(1 for g in corrected if g >= 0.200)
    c_lost = c_at_150 - c_at_200

    # A sub-threshold gap anywhere else would be a second instance of the
    # same defect and is recorded loudly if it ever appears.
    sub = [g for g in gaps if g < energy.MIN_SILENCE_S]
    unexpected_sub = [g for g in sub
                      if leading is None
                      or abs(g - leading["duration_s"]) > 1e-9]

    gap_bytes = canonical_gap_bytes(gaps)
    header = wav_header(path)
    return {
        "cell": name,
        "input": {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            # ON DISK, from the WAV header. `wavio.read_wav` resamples to
            # 16 kHz, so the reader's own sample rate is the ANALYSIS rate
            # and says nothing about the file. The narration cell is 44.1 kHz
            # on disk and 16 kHz after the read, and conflating the two
            # would make the provenance block lie about its own input.
            "file_sample_rate": header["sample_rate"],
            "file_samples": header["frames"],
            "file_duration_s": header["duration_s"],
            "file_channels": header["channels"],
            "file_sample_width_bytes": header["sample_width_bytes"],
            "resampled": header["sample_rate"] != SR,
            # AFTER the read, at the analysis rate.
            "analysis_sample_rate": audio.sample_rate,
            "analysis_samples": len(audio.samples),
            "analysis_duration_s": round(len(audio.samples) / SR, 9),
        },
        "gaps": {
            "n": n,
            "shortest_s": round(min(gaps), 6) if gaps else None,
            "p05_s_nearest_rank": _r(p05_nearest_rank(gaps)),
            "p05_s_linear": _r(p05_linear(gaps)),
            "median_s_standard": _r(median_standard(gaps)),
            "median_s_published_convention": _r(median_published(gaps)),
            "max_s": round(max(gaps), 6) if gaps else None,
            "durations_s": [round(g, 6) for g in gaps],
            "durations_sha256": sha256_bytes(gap_bytes),
            "durations_bytes_len": len(gap_bytes),
        },
        "trigger": {
            "at_0_150": at_150,
            "at_0_200": at_200,
            "lost": lost,
            "lost_pct": _r(100.0 * lost / at_150) if at_150 else None,
        },
        "leading_silence_artefact": leading,
        "sub_threshold_gaps_s": [round(g, 6) for g in sub],
        "unexpected_sub_threshold_gaps_s": [round(g, 6) for g in unexpected_sub],
        "resume_table": addendum,
        "corrected_without_leading_artefact": {
            "n": len(corrected),
            "shortest_s": round(min(corrected), 6) if corrected else None,
            "p05_s_nearest_rank": _r(p05_nearest_rank(corrected)),
            "median_s_standard": _r(median_standard(corrected)),
            "at_0_150": c_at_150,
            "at_0_200": c_at_200,
            "lost": c_lost,
            "lost_pct": _r(100.0 * c_lost / c_at_150) if c_at_150 else None,
        },
    }


def _r(x, nd=6):
    return None if x is None else round(x, nd)


# ---------------------------------------------------------------------------
# The pause-run population (the "680 vs 1671" question, NOTES-AUDIT.md F8).
#
# `tests/test_speculate.py::_corpus_pause_runs` counts PAUSE EVENTS from the
# streaming Watcher with run_s > 0.15. The watcher reports a still-open pause
# on a fixed cadence (`PAUSE_REPORT_S = 0.25 s`, src/watcher.py:202), so ONE
# silence run produces SEVERAL events. This measures both: the events (what
# the repo counts) and the distinct silence runs (what the prose says it
# counts).
# ---------------------------------------------------------------------------

class _EventProbe(watcher.Watcher):
    """
    A Watcher that records every event it emits, and the identity of the
    silence run each pause event belongs to.

    Two things have to be captured, and they have to be captured at the
    moment of emission:

    * EVERY event, including the OPEN-pause checkpoints. Those are emitted
      straight from `push()` (src/watcher.py:434-456) and never pass through
      `_events_for()`. Capturing only `_events_for()` sees the closed pauses
      and misses every checkpoint -- which is how a 2.5 s run, reported every
      0.25 s, collapses from eleven events to one.
    * the run START. `t1 - run_s` recovers it exactly for a checkpoint
      (src/watcher.py:443) but NOT for a closed pause: the closed event's
      interval is re-derived from the trimmed buffer, so its `t1` can be
      EARLIER than the run it closes, and its `run_s` is then clamped
      upward (src/watcher.py:679). For a closed event the run start is
      `self._pause_run_start` as it stood when the event was built, so that
      is what is captured.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.events: list = []
        self.closed_starts: list = []

    def _events_for(self, seg, ef, is_open=False):
        start_before = self._pause_run_start
        evs = super()._events_for(seg, ef, is_open=is_open)
        for ev in evs:
            if ev.kind == watcher.PAUSE:
                self.closed_starts.append(
                    start_before if start_before is not None else ev.t0)
        return evs

    def push(self, samples):
        evs = super().push(samples)
        self.events.extend(evs)
        return evs


def _runs_from_events(probe) -> list:
    """
    Group a probe's pause events into distinct silence RUNS.

    A run is one stretch of silence, however many times it was reported.
    The run's duration is the largest `run_s` reported for it, because the
    report is monotone within a run by construction.
    """
    runs: list = []
    closed = iter(probe.closed_starts)
    for index, ev in enumerate(probe.events):
        if ev.kind != watcher.PAUSE:
            continue
        run_s = float(ev.data.get("run_s", 0.0))
        if ev.data.get("checkpoint"):
            start = ev.t1 - run_s
        else:
            start = next(closed)
        if runs and abs(runs[-1]["start"] - start) <= 5e-3:
            r = runs[-1]
        else:
            r = {"start": start, "run_s": run_s, "last_index": index}
            runs.append(r)
        r["run_s"] = max(r["run_s"], run_s)
        r["last_index"] = index
    return runs


def _resumed_flags(probe, runs) -> None:
    """
    Did the speaker come back after each run?

    Answered by EVENT ORDER, not by comparing timestamps: the watcher's
    reported intervals jitter (a closed pause's `t1` can precede the run it
    closes), so timestamps are not a reliable ordering. The order of
    emission is exact. A speech segment emitted after the run's last pause
    event means the voice returned.
    """
    speech_after = [i for i, ev in enumerate(probe.events)
                    if ev.kind == watcher.UTTERANCE]
    for r in runs:
        r["resumed"] = any(i > r["last_index"] for i in speech_after)


def measure_watcher_population(skip: bool = False) -> dict:
    if skip:
        return {"skipped": True,
                "reason": "--skip-watcher was passed; the pause-run "
                           "population was not measured"}

    per_cell = {}
    all_events = []
    all_runs = []
    t0 = time.time()
    for name, path in CELLS.items():
        samples = wavio.read_wav(str(path)).samples
        w = _EventProbe(SR, adaptive=True)
        for i in range(0, len(samples), 1600):
            w.push(samples[i:i + 1600])

        pause_events = [ev for ev in w.events if ev.kind == watcher.PAUSE]
        events = [ev for ev in pause_events
                  if ev.data.get("run_s", 0) > 0.15]
        all_events.extend(events)

        runs = _runs_from_events(w)
        _resumed_flags(w, runs)
        all_runs.extend(runs)

        per_cell[name] = {
            "pause_events_gt_0_15": len(events),
            "pause_events_all": len(pause_events),
            "distinct_silence_runs": len(runs),
            "max_run_s": round(max((r["run_s"] for r in runs), default=0.0), 4),
            "min_run_s": round(min((r["run_s"] for r in runs), default=0.0), 4),
            "median_run_s": _r(median_standard([r["run_s"] for r in runs]), 4),
        }

    # Addendum 2 of TURN-ROUTES.md, re-measured on THIS population: for each
    # threshold, how many silence runs clear it, how many of those had the
    # speaker come back, and how many ran to the end of the file.
    #
    # "would have interrupted" is the RESUMED fraction: a turn decision that
    # fired on those runs would have cut the speaker off. It is the
    # complement of "the silence was a genuine turn end".
    addendum2 = {}
    for th in ADDENDUM_THRESHOLDS:
        sel = [r for r in all_runs if r["run_s"] >= th]
        resumed = sum(1 for r in sel if r["resumed"])
        at_end = len(sel) - resumed
        addendum2[f"{th:.1f}"] = {
            "silence_runs": len(sel),
            "speaker_resumed": resumed,
            "at_file_end": at_end,
            "would_have_interrupted_pct":
                _r(100.0 * resumed / len(sel), 4) if sel else None,
        }

    run_durations = sorted(r["run_s"] for r in all_runs)
    event_durations = sorted(e.data.get("run_s", 0.0) for e in all_events)
    longest = run_durations[-1] if run_durations else 0.0
    return {
        "skipped": False,
        "method": {
            "source": "tests/test_speculate.py::_corpus_pause_runs (lines 409-426)",
            "watcher": "src/watcher.py",
            "adaptive": True,
            "chunk_samples": 1600,
            "filter": "PAUSE events with run_s > 0.15",
            "note": "the watcher reports an OPEN pause every "
                    "PAUSE_REPORT_S = 0.25 s (src/watcher.py:202), so one "
                    "silence run yields several events; distinct runs are "
                    "reconstructed from t1 - run_s, which is the run start by "
                    "construction (src/watcher.py:443 and :668)",
        },
        "events_gt_0_15": len(all_events),
        "distinct_silence_runs": len(all_runs),
        "event_min_s": _r(event_durations[0], 4) if event_durations else None,
        "event_median_s": _r(median_standard(event_durations), 4),
        "event_max_s": _r(event_durations[-1], 4) if event_durations else None,
        "run_min_s": _r(run_durations[0], 4) if run_durations else None,
        "run_median_s": _r(median_standard(run_durations), 4),
        "run_max_s": _r(longest, 4),
        "max_pause_s_constant": watcher.MAX_PAUSE_S,
        "longest_run_below_max_pause_s": longest < watcher.MAX_PAUSE_S,
        "flush_clamp_note": (
            "the longest run is below MAX_PAUSE_S, so no run was flushed and "
            "the flush clamp (src/watcher.py:679) did not shift any "
            "reconstructed run start"
            if longest < watcher.MAX_PAUSE_S else
            "at least one run reached MAX_PAUSE_S, so the flush clamp may "
            "have split a run in the reconstruction; treat the distinct-run "
            "count as a lower bound"),
        "per_cell": per_cell,
        "addendum2_resume_table": addendum2,
        "elapsed_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# The copy check: the method in this file must be the method in the test file.
# ---------------------------------------------------------------------------

def check_copy() -> dict:
    """
    Verify that the method in THIS file is the method in the test file,
    byte for byte.

    A "verbatim copy" that nobody checks is a paraphrase. This runs the
    comparison at measurement time, so if the upstream helper changes, the
    result file says so instead of quietly measuring something else.
    """
    src = ROOT / METHOD_SOURCE
    here = Path(__file__).read_text()
    text = src.read_text()

    src_gaps = text.split("def _gaps(path: Path):", 1)[1].split("\n\n\n", 1)[0]
    here_gaps = here.split("def _gaps(path: Path):", 1)[1].split("\n\n\n", 1)[0]
    src_cells = text.split("CELLS = {", 1)[1].split("\n}\n", 1)[0]
    here_cells = here.split("CELLS = {", 1)[1].split("\n}\n", 1)[0]

    gaps_match = src_gaps == here_gaps
    cells_match = src_cells == here_cells
    return {
        "source_file": str(src.resolve()),
        "gaps_helper_lines": METHOD_GAPS_LINES,
        "cells_lines": METHOD_CELLS_LINES,
        "gaps_helper_identical": gaps_match,
        "cells_identical": cells_match,
        "ok": gaps_match and cells_match,
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def build_result(skip_watcher: bool = False) -> dict:
    t0 = time.time()
    cells = {name: measure_cell(name, path) for name, path in CELLS.items()}

    total_gaps = sum(c["gaps"]["n"] for c in cells.values())
    total_150 = sum(c["trigger"]["at_0_150"] for c in cells.values())
    total_200 = sum(c["trigger"]["at_0_200"] for c in cells.values())
    total_lost = total_150 - total_200

    corr_gaps = sum(c["corrected_without_leading_artefact"]["n"]
                    for c in cells.values())
    corr_150 = sum(c["corrected_without_leading_artefact"]["at_0_150"]
                    for c in cells.values())
    corr_200 = sum(c["corrected_without_leading_artefact"]["at_0_200"]
                    for c in cells.values())
    corr_lost = corr_150 - corr_200

    corpus_s = sum(c["input"]["analysis_duration_s"] for c in cells.values())

    # The mechanical diff against the published table.
    diff = {}
    for name, (g_n, g_min, g_p05, g_med, g_150, g_200, g_lost, g_pct) in \
            PUBLISHED.items():
        c = cells[name]
        d = {}
        if c["gaps"]["n"] != g_n:
            d["gaps"] = {"published": g_n, "measured": c["gaps"]["n"],
                         "delta": c["gaps"]["n"] - g_n}
        if abs((c["gaps"]["shortest_s"] or 0) - g_min) > 5e-4:
            d["shortest_s"] = {"published": g_min,
                               "measured": c["gaps"]["shortest_s"],
                               "delta": round(c["gaps"]["shortest_s"] - g_min, 6)}
        if abs((c["gaps"]["p05_s_nearest_rank"] or 0) - g_p05) > 5e-4:
            d["p05_s"] = {"published": g_p05,
                          "measured": c["gaps"]["p05_s_nearest_rank"],
                          "delta": round(c["gaps"]["p05_s_nearest_rank"] - g_p05, 6)}
        if abs((c["gaps"]["median_s_published_convention"] or 0) - g_med) > 5e-4:
            d["median_s"] = {"published": g_med,
                             "measured": c["gaps"]["median_s_published_convention"],
                             "delta": round(c["gaps"]["median_s_published_convention"]
                                            - g_med, 6)}
        if c["trigger"]["at_0_150"] != g_150:
            d["at_0_150"] = {"published": g_150,
                             "measured": c["trigger"]["at_0_150"],
                             "delta": c["trigger"]["at_0_150"] - g_150}
        if c["trigger"]["at_0_200"] != g_200:
            d["at_0_200"] = {"published": g_200,
                             "measured": c["trigger"]["at_0_200"],
                             "delta": c["trigger"]["at_0_200"] - g_200}
        if c["trigger"]["lost"] != g_lost:
            d["lost"] = {"published": g_lost,
                         "measured": c["trigger"]["lost"],
                         "delta": c["trigger"]["lost"] - g_lost}
        diff[name] = d
        if not d:
            diff.pop(name)          # only cells that actually differ

    total_diff = {}
    if total_gaps != sum(PUBLISHED[c][0] for c in PUBLISHED):
        total_diff["gaps"] = {"published": sum(PUBLISHED[c][0] for c in PUBLISHED),
                              "measured": total_gaps}
    if total_150 != PUBLISHED_TOTAL[0]:
        total_diff["at_0_150"] = {"published": PUBLISHED_TOTAL[0],
                                  "measured": total_150}
    if total_200 != PUBLISHED_TOTAL[1]:
        total_diff["at_0_200"] = {"published": PUBLISHED_TOTAL[1],
                                  "measured": total_200}
    if total_lost != PUBLISHED_TOTAL[2]:
        total_diff["lost"] = {"published": PUBLISHED_TOTAL[2],
                              "measured": total_lost}
    measured_total_pct = 100.0 * total_lost / total_150 if total_150 else 0.0
    if abs(measured_total_pct - PUBLISHED_TOTAL[3]) > 0.05:
        total_diff["lost_pct"] = {"published": PUBLISHED_TOTAL[3],
                                  "measured": round(measured_total_pct, 4)}

    script_path = Path(__file__).resolve()
    result = {
        "schema": "pause-threshold/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
            "elapsed_s": None,
        },
        "method": {
            "source_file": METHOD_SOURCE,
            "gaps_helper_lines": METHOD_GAPS_LINES,
            "cells_lines": METHOD_CELLS_LINES,
            "copied_verbatim": True,
            "copy_check": check_copy(),
            "parameters": {
                "sample_rate_hz": SR,
                "frame_len_samples": 800,
                "hop_samples": HOP,
                "hop_s": HOP_S,
                "frame_grid": "energy.rms_series(samples, 800, 160, 16000)",
                "segmenter": "src/energy.py:segment()",
                "peak_rms": "max(f.rms for f in ef) -- the cell's own peak frame",
                "min_silence_s": energy.MIN_SILENCE_S,
                "min_speech_s": energy.MIN_SPEECH_S,
                "adaptive_threshold": False,
                "gap_definition": "durations of the non-speech segments "
                                  "returned by energy.segment()",
            },
            "conventions": {
                "p05": "nearest-rank, sorted[ceil(0.05*n)-1] -- the "
                       "convention that reproduces the published table",
                "p05_alternative_linear": "linear interpolation between order "
                                          "statistics (recorded alongside)",
                "median_standard": "mean of the two middle values for even n",
                "median_published": "sorted[n//2] -- the upper middle value "
                                    "for even n; what the published table used",
            },
        },
        "corpus": {
            "total_duration_s_on_disk": round(corpus_s, 9),
            "total_duration_s_published": PUBLISHED_CORPUS_S,
            "total_duration_s_audit": AUDIT_CORPUS_S,
            "matches_published_465_s": abs(corpus_s - PUBLISHED_CORPUS_S) < 0.5,
            "matches_audit_485_01_s": abs(corpus_s - AUDIT_CORPUS_S) < 0.01,
            "per_cell_duration_s": {n: c["input"]["analysis_duration_s"]
                                    for n, c in cells.items()},
        },
        "cells": cells,
        "totals": {
            "gaps": total_gaps,
            "at_0_150": total_150,
            "at_0_200": total_200,
            "lost": total_lost,
            "lost_pct": round(measured_total_pct, 4),
            "corrected_without_leading_artefact": {
                "gaps": corr_gaps,
                "at_0_150": corr_150,
                "at_0_200": corr_200,
                "lost": corr_lost,
                "lost_pct": round(100.0 * corr_lost / corr_150, 4)
                if corr_150 else None,
            },
        },
        "published_table_comparison": {
            "published_source": "docs/research/ADOPTION-PLAN.md:124-145",
            "per_cell": diff,
            "total": total_diff,
            "identical": (not diff) and (not total_diff),
        },
        "known_defect": {
            "where": "src/energy.py:290 -- `kept = [segs[0]]`",
            "what": "the first segment of a file escapes the min_silence_s "
                    "filter, so a file's LEADING silence is reported as a gap",
            "effect_here": {
                name: c["leading_silence_artefact"]
                for name, c in cells.items()
                if c["leading_silence_artefact"] is not None
            },
            "affects_headline": (
                abs(100.0 * corr_lost / corr_150 - measured_total_pct) < 1e-9
                if corr_150 else None),
            "note": "the artefact is below 0.150 s, so it is already excluded "
                    "by the 0.150 trigger; the @0.150 / @0.200 / lost columns "
                    "and the 17.8% are identical with and without it. It "
                    "does change the 'gaps' and 'shortest' columns, and the "
                    "claim that the shortest real pause is 0.110 s.",
        },
        "watcher_pause_run_population": measure_watcher_population(skip_watcher),
        "energy_population_resume_table": {
            "population": "the silence runs of the verbatim _gaps() method, "
                          "i.e. the SAME population as the pause table above",
            "thresholds": {
                f"{th:.1f}": {
                    "silence_runs": sum(
                        c["resume_table"][f"{th:.1f}"]["silence_runs"]
                        for c in cells.values()),
                    "speaker_resumed": sum(
                        c["resume_table"][f"{th:.1f}"]["speaker_resumed"]
                        for c in cells.values()),
                    "at_file_end": sum(
                        c["resume_table"][f"{th:.1f}"]["at_file_end"]
                        for c in cells.values()),
                    "would_have_interrupted_pct": _r(
                        100.0 * sum(
                            c["resume_table"][f"{th:.1f}"]["speaker_resumed"]
                            for c in cells.values())
                        / sum(c["resume_table"][f"{th:.1f}"]["silence_runs"]
                              for c in cells.values()), 4)
                    if sum(c["resume_table"][f"{th:.1f}"]["silence_runs"]
                           for c in cells.values()) else None,
                }
                for th in ADDENDUM_THRESHOLDS
            },
        },
        "prose_claims": {
            "pause_runs_prose": PROSE_PAUSE_RUNS,
            "pause_runs_repository": REPO_PAUSE_RUNS,
        },
    }
    result["script"]["elapsed_s"] = round(time.time() - t0, 1)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "experiments"
                                          / "pause_threshold_results.json"))
    ap.add_argument("--check-copy", action="store_true",
                    help="verify the copied method against its source and exit")
    ap.add_argument("--skip-watcher", action="store_true",
                    help="skip the (slow) streaming pause-run population")
    args = ap.parse_args(argv)

    if args.check_copy:
        report = check_copy()
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    result = build_result(skip_watcher=args.skip_watcher)
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")
    print(f"wrote {out}")
    print(f"corpus duration on disk: {result['corpus']['total_duration_s_on_disk']:.6f} s")
    print(f"gaps {result['totals']['gaps']} "
          f"@0.150 {result['totals']['at_0_150']} "
          f"@0.200 {result['totals']['at_0_200']} "
          f"lost {result['totals']['lost']} "
          f"({result['totals']['lost_pct']:.1f}%)")
    print("published table identical: "
          f"{result['published_table_comparison']['identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
