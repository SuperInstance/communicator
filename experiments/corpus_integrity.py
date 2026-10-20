"""
corpus_integrity.py -- is the corpus trustworthy enough to train a teacher on?

WHY THIS FILE EXISTS.

The project has ONE binding constraint: the teacher needs at least 63
INDEPENDENT sessions per class before any accuracy number it produces means
anything (`experiments/teacher_data_requirement.py`; the 63 comes from
n = ceil(2*(z*s/d)^2) with z=2, s=0.01022 the between-session sd within a
label and d=0.00366 the between-label separation, on the `conc_mean` region
feature -- see `experiments/teacher_data_requirement_results.json`).

A take counts toward that 63 only if three things are true at once:

  (a) THE LABEL IS RIGHT.  The condition the take claims must be the
      condition the file says it is, and `speak` must be consistent with the
      kind of the condition.  A mislabelled take is worse than a missing one,
      because it poisons the class it was filed under.
  (b) THE SPEECH ACTUALLY HAPPENED.  A speaking condition with a measured
      speech fraction of 0.02 did not contain the poem, and a silent
      condition at 0.40 was not silent.  The capture script measures this
      with the Silero gate and writes `speech_frac`; this tool compares that
      measurement against what the LABEL PROMISES.
  (c) THE AUDIO AND THE VIDEO ACTUALLY CORRESPOND.  This is the one that
      bites.  The earlier captures were made by a script that recorded 30 s
      of audio and 25 s of video separately, wrote a 16000 Hz WAV header over
      audio that implies ~19188 Hz (a 1.2x error in every audio timestamp
      ever computed from them), and assumed 30 fps when the camera was
      delivering something else.  `experiments/av_sync.py` found NO
      correspondence at any lag, at any candidate sample rate, with any of
      three motion features: the audio and the video of A/B/C were two
      separate recordings.  A take whose time base is wrong cannot be
      pooled with a take whose time base is right, so the time base is
      CHECKED here rather than assumed.

WHAT IS CHECKED, per take.

  label     the file's condition prefix matches the metadata's condition;
            `speak` is true/false for a `condition` and null for a `free`.
  speech    speak=True needs speech_frac >= 0.15; speak=False needs <= 0.05.
            (The capture script's own thresholds.)
  rate      |achieved_fps - assumed_fps| / assumed_fps <= 0.05, and
            audio_sample_rate == requested_sample_rate.  This is the check
            that catches the 1.2x time-base defect.
  span      |video_span_s - audio_span_s| / video_span_s <= 0.02.
  drift     drift_after_row_removal < 5.0, the capture script's own
            usability threshold.
  file      the npz exists, loads, and holds exactly n_frames - 1 change
            maps.
  stimulus  the stimulus matches the corpus majority.  A mixed stimulus
            cannot be pooled, so a minority take is a failure of the POOL,
            not only of the take.

WHAT IS REPORTED, and what the number is FOR.

  Per condition, the number of DISTINCT SESSIONS -- distinct timestamps,
  not distinct files, because one session recorded twice is one session --
  and the gap to 63.  That gap is the only number in this file that
  measures progress, and it is stated together with where 63 comes from, so
  the denominator can be argued with rather than inherited.

WHAT IS NOT CLAIMED.

  A take that passes every check here has a trustworthy LABEL, a plausible
  amount of speech, a self-consistent time base and a readable change-map
  file.  That is a precondition for pooling, not a validation of the
  content, and it says nothing about whether the teacher can be trained on
  the result.  One speaker, one room, one lighting state: the usual limits
  of this project still apply.

USAGE:

    python experiments/corpus_integrity.py /mnt/c/proart_capture/corpus
    python experiments/corpus_integrity.py --json corpus_integrity.json

EXIT CODES: 0 clean, 1 at least one take failed a hard check, 2 the report
could not be produced (missing corpus, no takes, unreadable output path).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_CORPUS = "/mnt/c/proart_capture/corpus"

# -- the thresholds, and where each one comes from -------------------------

# The capture script's own achievement thresholds (corpus_capture_win.py:
# `want is True and speech_frac < 0.15` -> VOICE_MISSING; `want is False
# and speech_frac > 0.05` -> NOT_SILENT).
SPEECH_MIN = 0.15
SPEECH_MAX = 0.05

# The capture script's own camera-rate warning threshold:
# `abs(achieved_fps - FPS) / FPS > 0.05`.
FPS_REL_TOL = 0.05

# The A/V span tolerance.  The two streams are read in the same loop
# iteration, so their spans must agree to within 2% for the correspondence
# to be believable.
SPAN_REL_TOL = 0.02

# The capture script's own usability threshold: `if row_mad >= 5` -> "TOO
# HIGH -- change the lighting".
DRIFT_MAX = 5.0

# The binding constraint, and where the number comes from.
TARGET_SESSIONS_PER_CLASS = 63
TARGET_SOURCE = (
    "experiments/teacher_data_requirement.py -- n = ceil(2*(z*s/d)^2) with "
    "z=2, s=0.01022 (between-session sd within a label) and d=0.00366 "
    "(between-label separation) on the conc_mean region feature; recorded in "
    "experiments/teacher_data_requirement_results.json"
)

# Fallbacks for old-format takes that predate timing.json.  The capture
# script's constants: it assumed 30 fps and requested 16000 Hz.
ASSUMED_FPS_FALLBACK = 30.0
REQUESTED_RATE_FALLBACK = 16000

CHECKS = ("label", "speech", "rate", "span", "drift", "file", "stimulus")


# -- helpers ----------------------------------------------------------------

def _prefix(name: str) -> str:
    """The leading alphabetic run of a condition name: 'A1' -> 'A'."""
    m = re.match(r"^[A-Za-z]+", (name or "").strip())
    return m.group(0).upper() if m else ""


def _num(value):
    """A float, or None if the value is missing or not a number."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rel_diff(a: float, b: float):
    """Relative difference against b, or None if b is zero."""
    if b == 0:
        return None
    return abs(a - b) / abs(b)


# -- the per-take check -----------------------------------------------------

def check_take(meta_path: Path, majority_stimulus=None) -> dict:
    """Check one take and return its record.

    `majority_stimulus` is the corpus-wide majority stimulus name, or None
    before the majority is known (first pass).  A take whose stimulus
    differs from it fails the `stimulus` check, because a corpus with two
    stimuli in it cannot be pooled.
    """
    stem = meta_path.stem
    parts = stem.split("_")
    file_name = parts[0]
    file_stamp = parts[-1] if len(parts) > 1 else ""
    file_stamp = file_stamp if file_stamp.isdigit() else None

    rec = {
        "file": meta_path.name,
        "condition": None,
        "kind": None,
        "speak": None,
        "pose": None,
        "stimulus": None,
        "speech_frac": None,
        "n_frames": None,
        "timestamp": None,
        "session_key": stem,
        "achieved_fps": None,
        "assumed_fps": None,
        "audio_sample_rate": None,
        "requested_sample_rate": None,
        "video_span_s": None,
        "audio_span_s": None,
        "drift_after_row_removal": None,
        "has_timing": False,
        "failures": [],
        "unverified": [],
    }

    def fail(check, message):
        rec["failures"].append({"check": check, "message": message})

    def unver(check, message):
        rec["unverified"].append({"check": check, "message": message})

    # -- the metadata itself ------------------------------------------------
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:  # noqa: BLE001
        fail("file", f"metadata is not readable JSON: {exc}")
        return rec
    if not isinstance(meta, dict):
        fail("file", "metadata is not a JSON object")
        return rec

    # -- the timing file, if this is a new-format take ----------------------
    timing_path = meta_path.with_name(f"{stem}_timing.json")
    timing = None
    if timing_path.is_file():
        rec["has_timing"] = True
        try:
            timing = json.loads(timing_path.read_text())
        except Exception as exc:  # noqa: BLE001
            unver("rate", f"timing.json is not readable JSON: {exc}")
            timing = None
        if not isinstance(timing, dict):
            timing = None
    if timing is None:
        rec["has_timing"] = False

    # -- 1. LABEL CONSISTENCY ----------------------------------------------
    condition = meta.get("condition")
    rec["condition"] = condition if isinstance(condition, str) else None
    rec["kind"] = meta.get("kind")
    speak = meta.get("speak", None)
    rec["speak"] = speak if isinstance(speak, bool) else None
    rec["pose"] = meta.get("pose")

    if not isinstance(condition, str) or not condition:
        fail("label", "metadata has no condition field")
    else:
        if _prefix(condition) != _prefix(file_name):
            fail("label", f"file names condition {_prefix(file_name)!r} "
                          f"but the metadata says {condition!r}")
        kind = meta.get("kind")
        if kind == "condition":
            if not isinstance(speak, bool):
                fail("label", f"kind 'condition' requires speak true/false, "
                              f"got {speak!r}")
        elif kind == "free":
            if speak is not None:
                fail("label", f"kind 'free' requires speak null, "
                              f"got {speak!r}")
        else:
            fail("label", f"unknown condition kind {kind!r}")

    # -- 2. SPEECH PLAUSIBILITY --------------------------------------------
    speech_frac = _num(meta.get("speech_frac"))
    rec["speech_frac"] = speech_frac
    if isinstance(speak, bool):
        if speech_frac is None:
            unver("speech", "speech_frac was never measured (the gate was "
                            "unavailable at capture time)")
        elif speak and speech_frac < SPEECH_MIN:
            fail("speech", f"speech_frac {speech_frac:.4f} < {SPEECH_MIN} "
                            f"for a condition labelled speak=True")
        elif (not speak) and speech_frac > SPEECH_MAX:
            fail("speech", f"speech_frac {speech_frac:.4f} > {SPEECH_MAX} "
                            f"for a condition labelled speak=False")

    # -- 3. RATE CONSISTENCY -----------------------------------------------
    if timing:
        assumed = _num(timing.get("assumed_fps"))
        achieved = _num(timing.get("achieved_fps"))
        got_rate = timing.get("audio_sample_rate")
        want_rate = timing.get("requested_sample_rate")
    else:
        assumed = None
        achieved = _num(meta.get("achieved_fps"))
        got_rate = meta.get("audio_sample_rate")
        want_rate = None
    if achieved is None:
        unver("rate", "no achieved_fps was recorded")
    else:
        if achieved <= 0:
            fail("rate", f"achieved_fps {achieved:g} -- the camera delivered "
                         f"no measurable rate")
        else:
            if assumed is None:
                assumed = ASSUMED_FPS_FALLBACK
            rel = _rel_diff(achieved, assumed)
            if rel is not None and rel > FPS_REL_TOL:
                fail("rate", f"camera delivered {achieved:.3f} fps against an "
                             f"assumed {assumed:.3f} (relative error "
                             f"{rel:.4f} > {FPS_REL_TOL}) -- the video's time "
                             f"base is not the frame index's time base")
    if got_rate is None:
        unver("rate", "no audio_sample_rate was recorded")
    else:
        if want_rate is None:
            want_rate = REQUESTED_RATE_FALLBACK
        if got_rate != want_rate:
            fail("rate", f"audio was captured at {got_rate} Hz but "
                         f"{want_rate} Hz was requested -- every audio "
                         f"timestamp computed from this take is wrong")

    rec["achieved_fps"] = achieved
    rec["assumed_fps"] = assumed
    rec["audio_sample_rate"] = got_rate
    rec["requested_sample_rate"] = want_rate

    # -- 4. A/V SPAN AGREEMENT ---------------------------------------------
    if timing:
        video_span = _num(timing.get("video_span_s"))
        audio_span = _num(timing.get("audio_span_s"))
    else:
        video_span = _num(meta.get("seconds"))
        audio_span = _num(meta.get("audio_seconds"))
    rec["video_span_s"] = video_span
    rec["audio_span_s"] = audio_span
    if video_span is None or audio_span is None:
        unver("span", "no video/audio span was recorded")
    elif video_span <= 0:
        fail("span", f"video_span_s {video_span:g} is not positive")
    else:
        rel = _rel_diff(audio_span, video_span)
        if rel is not None and rel > SPAN_REL_TOL:
            fail("span", f"video span {video_span:.3f} s against audio span "
                         f"{audio_span:.3f} s (relative difference "
                         f"{rel:.4f} > {SPAN_REL_TOL}) -- the two streams are "
                         f"not the same length")

    # -- 5. DRIFT -----------------------------------------------------------
    drift = _num(meta.get("drift_after_row_removal"))
    rec["drift_after_row_removal"] = drift
    if drift is None:
        unver("drift", "no drift_after_row_removal was recorded")
    elif drift >= DRIFT_MAX:
        fail("drift", f"drift after row removal {drift:.3f} >= {DRIFT_MAX} "
                      f"-- the banding is not the only problem")

    # -- 6. FILE INTEGRITY --------------------------------------------------
    n_frames = meta.get("n_frames")
    rec["n_frames"] = int(n_frames) if isinstance(n_frames, (int, float)) \
        and not isinstance(n_frames, bool) else None
    npz_path = meta_path.with_suffix(".npz")
    if not npz_path.is_file():
        fail("file", f"{npz_path.name} is missing")
    else:
        try:
            with np.load(npz_path) as z:
                if "maps" not in z:
                    fail("file", f"{npz_path.name} has no 'maps' array")
                else:
                    maps = z["maps"]
                    if rec["n_frames"] is None:
                        unver("file", "no n_frames in the metadata, so the "
                                      "map count could not be checked")
                    elif maps.shape[0] != rec["n_frames"] - 1:
                        fail("file", f"{npz_path.name} holds "
                                     f"{maps.shape[0]} change maps but "
                                     f"n_frames-1 is {rec['n_frames'] - 1}")
                if "scale" not in z:
                    unver("file", f"{npz_path.name} has no 'scale'")
        except Exception as exc:  # noqa: BLE001
            fail("file", f"{npz_path.name} did not load: {exc}")

    # -- 7. STIMULUS --------------------------------------------------------
    stimulus = meta.get("stimulus")
    rec["stimulus"] = stimulus if isinstance(stimulus, str) else None
    if rec["stimulus"] is None:
        unver("stimulus", "no stimulus was recorded")
    elif majority_stimulus and rec["stimulus"] != majority_stimulus:
        fail("stimulus", f"stimulus {rec['stimulus']!r} differs from the "
                         f"corpus majority {majority_stimulus!r} -- a mixed "
                         f"stimulus cannot be pooled")

    # -- the session identity ----------------------------------------------
    timestamp = meta.get("timestamp")
    timestamp = int(timestamp) if isinstance(timestamp, (int, float)) \
        and not isinstance(timestamp, bool) else None
    rec["timestamp"] = timestamp
    if timestamp is not None:
        rec["session_key"] = str(timestamp)
    elif file_stamp is not None:
        rec["session_key"] = file_stamp

    rec["ok"] = not rec["failures"]
    return rec


# -- the corpus-level analysis ---------------------------------------------

def find_takes(corpus: Path):
    """Every take's metadata file, excluding the timing files."""
    return sorted(p for p in corpus.glob("*.json")
                  if not p.name.endswith("_timing.json"))


def analyse(corpus: Path) -> dict:
    """Check every take in `corpus` and return the report as a dict."""
    corpus = Path(corpus)
    meta_paths = find_takes(corpus)

    # Pass 1: the stimulus majority, so a minority take can be caught.
    stimuli = []
    for p in meta_paths:
        try:
            meta = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(meta, dict) and isinstance(meta.get("stimulus"), str):
            stimuli.append(meta["stimulus"])
    counts = Counter(stimuli)
    majority = counts.most_common(1)[0][0] if counts else None

    # Pass 2: the checks.
    takes = [check_take(p, majority) for p in meta_paths]

    failure_counts = {c: 0 for c in CHECKS}
    unverified_counts = {c: 0 for c in CHECKS}
    for t in takes:
        for f in t["failures"]:
            failure_counts[f["check"]] = failure_counts.get(f["check"], 0) + 1
        for u in t["unverified"]:
            unverified_counts[u["check"]] = unverified_counts.get(u["check"], 0) + 1

    # -- the coverage that actually matters ---------------------------------
    per_condition = {}
    for t in takes:
        cond = t["condition"] or "<unlabelled>"
        c = per_condition.setdefault(cond, {"sessions": set(), "usable": set(),
                                            "takes": 0})
        c["takes"] += 1
        c["sessions"].add(t["session_key"])
        if t["ok"]:
            c["usable"].add(t["session_key"])

    per_class = {}
    for t in takes:
        speak = t["speak"]
        label = "SPEAK" if speak is True else ("SILENT" if speak is False
                                                else "UNLABELLED")
        c = per_class.setdefault(label, {"sessions": set(), "usable": set(),
                                         "takes": 0})
        c["takes"] += 1
        c["sessions"].add(t["session_key"])
        if t["ok"]:
            c["usable"].add(t["session_key"])

    coverage = {
        "target_sessions_per_class": TARGET_SESSIONS_PER_CLASS,
        "target_source": TARGET_SOURCE,
        "per_condition": {},
        "per_class": {},
    }
    for cond, c in sorted(per_condition.items()):
        n = len(c["sessions"])
        coverage["per_condition"][cond] = {
            "takes": c["takes"],
            "distinct_sessions": n,
            "usable_sessions": len(c["usable"]),
            "gap_to_target": TARGET_SESSIONS_PER_CLASS - n,
            "target_met": n >= TARGET_SESSIONS_PER_CLASS,
        }
    for label, c in sorted(per_class.items()):
        n = len(c["sessions"])
        entry = {
            "takes": c["takes"],
            "distinct_sessions": n,
            "usable_sessions": len(c["usable"]),
        }
        if label in ("SPEAK", "SILENT"):
            entry["gap_to_target"] = TARGET_SESSIONS_PER_CLASS - n
            entry["target_met"] = n >= TARGET_SESSIONS_PER_CLASS
        else:
            # Free sessions carry no label until the teacher is trusted, so
            # they are not pooled into a class and have no class target.
            entry["gap_to_target"] = None
            entry["target_met"] = None
            entry["note"] = ("unlabelled: not pooled into a class until the "
                             "teacher is trusted")
        coverage["per_class"][label] = entry

    n_failed = sum(1 for t in takes if not t["ok"])
    n_missing_timing = sum(1 for t in takes if not t["has_timing"])
    n_unverified = sum(len(t["unverified"]) for t in takes)

    return {
        "corpus": str(corpus),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_takes": len(takes),
        "n_failed": n_failed,
        "n_missing_timing": n_missing_timing,
        "n_unverified_checks": n_unverified,
        "failure_counts": failure_counts,
        "unverified_counts": unverified_counts,
        "ok": n_failed == 0,
        "stimulus": {
            "majority": majority,
            "counts": dict(counts),
            "mixed": len(counts) > 1,
        },
        "coverage": coverage,
        "takes": takes,
    }


# -- the report -------------------------------------------------------------

def format_report(report: dict) -> str:
    """The human-readable report."""
    out = []
    out.append("CORPUS INTEGRITY REPORT")
    out.append("=" * 62)
    out.append(f"corpus            {report['corpus']}")
    out.append(f"takes             {report['n_takes']}")
    out.append(f"failed            {report['n_failed']}")
    out.append(f"missing timing    {report['n_missing_timing']}  "
               f"(old-format takes, counted separately from failures)")
    out.append(f"unverified checks {report['n_unverified_checks']}")

    out.append("")
    out.append("TAKES")
    if not report["takes"]:
        out.append("  (none)")
    for t in report["takes"]:
        mark = "ok  " if t["ok"] else "FAIL"
        stim = t["stimulus"] if t["stimulus"] is not None else "-"
        out.append(f"  [{mark}] {t['file']}  condition={t['condition']}  "
                   f"speak={t['speak']}  stimulus={stim}")
        for f in t["failures"]:
            out.append(f"           {f['check']:<8}: {f['message']}")
        for u in t["unverified"]:
            out.append(f"           {u['check']:<8}: (unverified) {u['message']}")

    stim = report["stimulus"]
    out.append("")
    out.append("STIMULUS")
    if stim["majority"] is None:
        out.append("  no stimulus was recorded by any take")
    else:
        out.append(f"  majority {stim['majority']!r}  "
                   + ", ".join(f"{k!r}: {v}" for k, v in
                               sorted(stim["counts"].items())))
        if stim["mixed"]:
            out.append("  MIXED: takes with a different stimulus cannot be "
                       "pooled with the majority.")

    cov = report["coverage"]
    out.append("")
    out.append(f"COVERAGE  (target {cov['target_sessions_per_class']} "
               f"independent sessions per class)")
    out.append(f"  source: {cov['target_source']}")
    out.append("  per condition:")
    for cond, c in cov["per_condition"].items():
        out.append(f"    {cond:<10} {c['distinct_sessions']:>4} distinct "
                   f"sessions  ({c['usable_sessions']} usable)  "
                   f"gap {c['gap_to_target']}")
    out.append("  per class:")
    for label, c in cov["per_class"].items():
        if c["gap_to_target"] is None:
            out.append(f"    {label:<10} {c['distinct_sessions']:>4} distinct "
                       f"sessions  ({c['usable_sessions']} usable)  "
                       f"no class target")
        else:
            out.append(f"    {label:<10} {c['distinct_sessions']:>4} distinct "
                       f"sessions  ({c['usable_sessions']} usable)  "
                       f"gap {c['gap_to_target']}")

    counts = report["failure_counts"]
    detail = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
    out.append("")
    out.append(f"SUMMARY: {report['n_failed']} of {report['n_takes']} takes "
               f"failed" + (f" ({detail})" if detail else ""))
    out.append(f"         {report['n_missing_timing']} takes have no "
               f"timing.json (old format) -- not counted as failures")
    return "\n".join(out)


# -- the CLI ----------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Check a captured corpus for label, speech, rate, "
                    "span, drift, file and stimulus integrity, and report "
                    "the gap to the sessions-per-class the teacher needs.")
    ap.add_argument("corpus", nargs="?", default=DEFAULT_CORPUS,
                    help=f"corpus directory (default: {DEFAULT_CORPUS})")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="write the JSON summary to this path")
    a = ap.parse_args(argv)

    corpus = Path(a.corpus)
    if not corpus.is_dir():
        print(f"corpus directory not found: {corpus}", file=sys.stderr)
        print("nothing was checked; pass the directory the capture script "
              "wrote to (its --out).", file=sys.stderr)
        return 2

    report = analyse(corpus)
    if report["n_takes"] == 0:
        print(f"no takes found in {corpus} (expected <NAME>_<STAMP>.json)",
              file=sys.stderr)
        return 2

    print(format_report(report))

    if a.json_path:
        out_path = Path(a.json_path)
        try:
            if out_path.parent and not out_path.parent.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2))
        except OSError as exc:
            print(f"could not write {out_path}: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {out_path}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
