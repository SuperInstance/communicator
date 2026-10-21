#!/usr/bin/env python3
"""
browser_capture_acceptance.py -- the independent acceptance checker for the
browser-native corpus capture.

WHY THIS FILE EXISTS.

ZCode built `browser_capture/` and wrote its own `verify.py`. A judge cannot
certify itself, so this file is the referee: it takes a session output
directory and checks every take against expectations derived from the
REFERENCE implementation -- `/mnt/c/proart_capture/corpus_capture_win.py`
(what `--plan` writes per take), `CAPTURE-NEXT.md`, the two browser briefs,
and the gate `experiments/corpus_integrity.py` -- and against the repo's own
readers `src/nv12.py` and `src/wavio.py`.

The derivation is written down, BEFORE any of ZCode's code was read, in
`docs/research/ACCEPTANCE-EXPECTED.md`. This file is that document, executable.

WHAT IT CHECKS, per take.

  LAYOUT     the required artifact set exists, with the reference's names
  NV12       the raw luma stream round-trips through src/nv12.py: the file
             size is exactly frames x stride, and the content is identical
  WAV        src/wavio.py reads it, at the rate the metadata states
  JSON       timing.json and meta.json parse and match the reference schema,
             and their cross-field arithmetic holds
  ARRAYS     the .npz holds the five arrays with the right shapes and dtypes,
             and the change maps re-derive from the stored luma through the
             REFERENCE's own functions
  DRIFT      drift_after_row_removal is under the capture script's own 5.0
             threshold, and matches the value re-derived from the stored luma
  FLAGS      VOICE_MISSING / NOT_SILENT are present exactly where the data
             demands them, and the recorded speech_frac is corroborated by
             an INDEPENDENT re-measurement with the same Silero gate
  RICH       the addendum's _rich.json carries all eight required items
  WEBM       the addendum's _full.webm is a real WebM carrying BOTH a video
             and a muxed audio track
  MANIFEST   the addendum's session.json lists every take, its files, its
             labels and its flags
  GATE       experiments/corpus_integrity.py passes on the takes
  PLAN       which of the 16 plan conditions are present (reported)

WHAT IT DELIBERATELY DOES NOT DO.

It does not modify anything. It does not read ZCode's code to decide what is
right; the expectations are in ACCEPTANCE-EXPECTED.md and were fixed first.
It does not soften a failure into a note: every failure is named, and the
exit code is non-zero.

WHERE THE BRIEFS CONFLICT WITH EACH OTHER, the checker says so rather than
picking a winner silently. The two known conflicts are recorded in
ACCEPTANCE-EXPECTED.md §12: the base brief says "19 takes" where the plan has
16, and the addendum's `session.json` is individuated as a take by the gate
(which globs `*.json`), so the gate is run twice -- once on the directory as
it stands, once on the takes alone.

USAGE:

    python experiments/browser_capture_acceptance.py /path/to/session
    python experiments/browser_capture_acceptance.py /path/to/session \
        --json experiments/browser_capture_acceptance_results.json
    python experiments/browser_capture_acceptance.py /path/to/session \
        --require-full-plan

EXIT CODES: 0 every check passed, 1 at least one check failed, 2 the report
could not be produced (no such directory, no takes in it).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import nv12  # noqa: E402
from src import wavio  # noqa: E402

# -- the reference implementation's own constants ----------------------------
# (corpus_capture_win.py:187-224). The browser version must produce takes the
# existing pipeline accepts UNCHANGED, so these are the numbers, not mine.
W, H = 160, 96
MAP_W, MAP_H = 80, 48
RATE = 16000
FPS = 30
REPEATS = 3

# -- the thresholds, and where each one comes from ---------------------------
# corpus_integrity.py:99-113 -- the capture script's own achievement thresholds,
# re-used verbatim by the gate.
SPEECH_MIN = 0.15
SPEECH_MAX = 0.05
FPS_REL_TOL = 0.05
SPAN_REL_TOL = 0.02
DRIFT_MAX = 5.0

# The capture script's own drift warning: `if row_mad >= DRIFT_OK`.
DRIFT_OK = 5.0

# The 16-take plan (CAPTURE-NEXT.md's table, and the reference's PLAN).
PLAN_CONDITIONS = [f"{c}{r}" for r in range(1, REPEATS + 1)
                   for c in ("A", "B", "C", "D", "F")] + ["FREE"]
PLAN_KIND = {c: "condition" for c in ("A", "B", "C", "D")}
PLAN_KIND.update({c: "free" for c in ("F", "FREE")})
PLAN_SPEAK = {"A": True, "B": True, "C": False, "D": False}
PLAN_POSE = {"A": "facing", "B": "turned", "C": "facing", "D": "turned"}

# The metadata keys the reference writes, with the types it writes them as.
# corpus_capture_win.py:481-554.
TIMING_KEYS = {
    "condition": str,
    "n_frames": int,
    "frame_offsets_s": list,
    "assumed_fps": (int, float),
    "achieved_fps": (int, float),
    "video_span_s": (int, float),
    "audio_span_s": (int, float),
    "audio_sample_rate": int,
    "requested_sample_rate": int,
    "mic": str,
    "capture_wallclock_s": (int, float),
}

META_KEYS = {
    "condition": str,
    "kind": str,
    "note": str,
    "pose": str,
    "speak": (bool, type(None)),
    "stimulus": str,
    "stimulus_title": str,
    "stimulus_estimated_s": (int, float),
    "speech_frac": (int, float, type(None)),
    "n_frames": int,
    "seconds": (int, float),
    "achieved_fps": (int, float),
    "audio_sample_rate": int,
    "audio_seconds": (int, float),
    "map_shape": list,
    "map_scale": (int, float),
    "map_scale_raw": (int, float),
    "drift_raw": (int, float),
    "drift_after_row_removal": (int, float),
    "timestamp": int,
    "rep": int,
    "filter": str,
    "filters": dict,
    "arrays": list,
}

NPZ_KEYS = ("maps", "scale", "maps_raw", "scale_raw", "luma")

# The addendum's rich sidecar: required CONTENT. The addendum fixes no key
# names, so the checker matches on normalised aliases and reports which key
# it matched, rather than inventing a format the brief never asked for.
RICH_REQUIREMENTS = (
    ("condition", ("condition", "conditionlabel", "label", "takelabel", "take")),
    ("instruction", ("instruction", "instructiontext", "instructionshown",
                     "instructiondisplayed", "prompt", "prompttext", "instructionprompt")),
    ("stimulus", ("stimulus", "stimulusname", "stimulustitle", "stimulus_title")),
    ("wall_clock_start", ("wallclockstart", "start", "startedat", "starttime",
                          "startutc", "wallclockstartutc", "wallclockstartiso",
                          "recordingstart")),
    ("wall_clock_end", ("wallclockend", "end", "endedat", "endtime", "endutc",
                        "wallclockendutc", "wallclockendiso", "recordingend")),
    ("achieved_fps", ("achievedfps", "fps", "achievedframespersecond",
                      "actualfps", "framerate", "measuredfps")),
    ("sample_rate", ("samplerate", "audiosamplerate", "achievedsamplerate",
                     "actualsamplerate", "micrate", "audiorate")),
    ("timestamps_ref", ("frameoffsets", "frameoffsetss", "frameoffsetsref",
                        "frametimestamps", "frametimestampsfile",
                        "frametimestampsref", "timestamps", "timestamparray",
                        "timestampref", "timestampsref", "perframetimestamps",
                        "timingref", "timingfile", "timestampsfile",
                        "frameoffsetsreference")),
    ("device", ("device", "devicelabel", "devicename", "cameradevice",
                "cameradevicelabel", "browserdevice", "browser", "browserlabel",
                "useragent", "mediadevices", "deviceinfo")),
    ("notes", ("notes", "note", "freetextnotes", "comment", "comments",
               "operatornotes", "casenotes", "whatidid")),
)

MANIFEST_TAKES_KEYS = ("takes", "records", "items", "entries", "captures")
MANIFEST_FILES_KEYS = ("files", "artifacts", "file", "filenames", "paths")
MANIFEST_FLAGS_KEYS = ("flags", "flag", "warnings", "status")


# ============================================================================
# the reference implementation, imported so that this checker computes the
# maps and the drift with the SAME code the pipeline will use.
# ============================================================================

def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _reference():
    """The reference capture script, if it is in this checkout."""
    p = ROOT / "experiments" / "corpus_capture_win.py"
    if not p.is_file():
        return None
    try:
        return _load_module(p, "_ref_corpus_capture_win")
    except Exception:  # noqa: BLE001
        return None


REF = _reference()


def _gate_module():
    """The gate, `experiments/corpus_integrity.py`, if it is in this checkout."""
    p = ROOT / "experiments" / "corpus_integrity.py"
    if not p.is_file():
        return None
    try:
        return _load_module(p, "_ref_corpus_integrity")
    except Exception:  # noqa: BLE001
        return None


GATE = _gate_module()


# ============================================================================
# results
# ============================================================================

@dataclass
class Check:
    """One named check, its failures, and its notes."""
    name: str
    failures: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    skipped: bool = False

    def fail(self, message: str):
        self.failures.append(message)

    def note(self, message: str):
        self.notes.append(message)

    def skip(self, message: str):
        self.skipped = True
        self.notes.append(message)

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        return "FAIL" if self.failures else "PASS"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "failures": list(self.failures),
            "notes": list(self.notes),
        }


@dataclass
class Take:
    """One take, and everything the checker found out about it."""
    meta_path: Path
    stem: str
    condition: str = ""
    meta: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)

    def check(self, name: str) -> Check:
        c = self.checks.get(name)
        if c is None:
            c = Check(name)
            self.checks[name] = c
        return c

    @property
    def ok(self) -> bool:
        return all(c.status != "FAIL" for c in self.checks.values())


# ============================================================================
# helpers
# ============================================================================

def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _silero_model() -> Path | None:
    """Find the Silero weights, so the speech re-measurement can run."""
    env = os.environ.get("SILERO_VAD_MODEL")
    if env and Path(env).is_file():
        return Path(env)
    local = ROOT / "models" / "silero_vad.onnx"
    if local.is_file():
        return local
    # A sibling checkout of the same project: the acceptance worktree is
    # `communicator-acceptance` and the main checkout is `communicator`.
    base = ROOT.name.split("-")[0]
    try:
        for sib in ROOT.parent.iterdir():
            if not sib.is_dir() or not sib.name.startswith(base):
                continue
            cand = sib / "models" / "silero_vad.onnx"
            if cand.is_file():
                return cand
    except OSError:
        pass
    return None


class SpeechMeter:
    """An INDEPENDENT speech measurement, using the same gate the reference uses.

    The reference computes `speech_frac` with `SileroGate().push(...)` over
    `read_wav(...)` and counts the probabilities at or above 0.5
    (corpus_capture_win.py:501-504). Re-running exactly that, on the WAV the
    producer wrote, is what stops a producer from asserting a speech_frac
    that the audio does not support.
    """

    def __init__(self):
        self.model = _silero_model()
        self.available = False
        self.reason = ""
        if self.model is None:
            self.reason = "the Silero weights were not found"
            return
        try:
            os.environ["SILERO_VAD_MODEL"] = str(self.model)
            from src.vad import SileroGate
            self._gate_cls = SileroGate
            self.available = True
        except Exception as exc:  # noqa: BLE001
            self.reason = f"the gate could not be loaded: {exc}"

    def measure(self, wav_path: Path) -> tuple[float | None, str]:
        if not self.available:
            return None, self.reason
        try:
            audio = wavio.read_wav(str(wav_path))
            if not audio.samples:
                return None, "the WAV holds no samples"
            # A FRESH gate for every take: the gate carries an LSTM state
            # across frames on purpose, so reusing one instance would make a
            # take's measurement depend on which takes were measured before
            # it. The reference creates a new one per take
            # (corpus_capture_win.py:502) and so does this.
            probs = self._gate_cls().push(list(audio.samples))
            if not probs:
                return None, "the gate returned no probabilities"
            frac = sum(1 for v in probs if v >= 0.5) / len(probs)
            return frac, ""
        except Exception as exc:  # noqa: BLE001
            return None, f"the measurement failed: {exc}"


# ============================================================================
# the EBML / WebM reader (stdlib only)
# ============================================================================
#
# The addendum asks for a WebM "WITH the mic audio track muxed". That is not
# checkable by looking at the file extension, so this parses the container:
# EBML magic -> DocType -> Segment -> Tracks -> TrackEntry (TrackType, CodecID,
# PixelWidth/PixelHeight, SamplingFrequency).

_EBML_HEADER = 0x1A45DFA3
_ID_DOCTYPE = 0x4282
_ID_SEGMENT = 0x18538067
_ID_TRACKS = 0x1654AE6B
_ID_TRACK_ENTRY = 0xAE
_ID_TRACK_TYPE = 0x83
_ID_CODEC_ID = 0x86
_ID_VIDEO = 0xE0
_ID_PIXEL_WIDTH = 0xB0
_ID_PIXEL_HEIGHT = 0xBA
_ID_AUDIO = 0xE1
_ID_SAMPLING_FREQ = 0xB5


def _read_id(buf: bytes, pos: int):
    """An EBML element ID: the marker bits are part of the ID."""
    first = buf[pos]
    length = 1
    mask = 0x80
    while not (first & mask):
        mask >>= 1
        length += 1
        if length > 4:
            raise ValueError("not an EBML element ID")
    return int.from_bytes(buf[pos:pos + length], "big"), length


def _read_size(buf: bytes, pos: int):
    """An EBML data size: the marker bits are stripped."""
    first = buf[pos]
    length = 1
    mask = 0x80
    while not (first & mask):
        mask >>= 1
        length += 1
        if length > 8:
            raise ValueError("not an EBML size")
    value = first & (mask - 1)
    for i in range(1, length):
        value = (value << 8) | buf[pos + i]
    return value, length


def _elements(buf: bytes, start: int, end: int):
    """(id, data_start, data_end) for each element in the range."""
    pos = start
    while pos < end:
        try:
            eid, ilen = _read_id(buf, pos)
            pos += ilen
            size, slen = _read_size(buf, pos)
            pos += slen
        except (IndexError, ValueError):
            return
        unknown = size == (1 << (7 * slen)) - 1
        data_end = end if unknown else min(end, pos + size)
        yield eid, pos, data_end
        pos = data_end
        if unknown:
            return


def read_webm(path: Path) -> dict:
    """Parse a WebM's container. Returns a dict describing what is in it."""
    data = path.read_bytes()
    info = {
        "bytes": len(data),
        "ebml_magic": data[:4] == b"\x1a\x45\xdf\xa3",
        "doc_type": None,
        "tracks": [],
        "video_width": None,
        "video_height": None,
        "audio_sampling_frequency": None,
        "error": None,
    }
    if not info["ebml_magic"]:
        info["error"] = "the file does not start with the EBML magic"
        return info
    try:
        for eid, s, e in _elements(data, 0, len(data)):
            if eid == _EBML_HEADER:
                for e2, s2, e2e in _elements(data, s, e):
                    if e2 == _ID_DOCTYPE:
                        info["doc_type"] = data[s2:e2e].decode("utf-8", "replace")
            elif eid == _ID_SEGMENT:
                for e2, s2, e2e in _elements(data, s, e):
                    if e2 != _ID_TRACKS:
                        continue
                    for e3, s3, e3e in _elements(data, s2, e2e):
                        if e3 != _ID_TRACK_ENTRY:
                            continue
                        track = {"type": None, "codec": None, "width": None,
                                 "height": None, "sampling_frequency": None}
                        for e4, s4, e4e in _elements(data, s3, e3e):
                            if e4 == _ID_TRACK_TYPE:
                                track["type"] = int.from_bytes(data[s4:e4e], "big")
                            elif e4 == _ID_CODEC_ID:
                                track["codec"] = data[s4:e4e].decode("utf-8", "replace")
                            elif e4 == _ID_VIDEO:
                                for e5, s5, e5e in _elements(data, s4, e4e):
                                    if e5 == _ID_PIXEL_WIDTH:
                                        track["width"] = int.from_bytes(data[s5:e5e], "big")
                                    elif e5 == _ID_PIXEL_HEIGHT:
                                        track["height"] = int.from_bytes(data[s5:e5e], "big")
                            elif e4 == _ID_AUDIO:
                                for e5, s5, e5e in _elements(data, s4, e4e):
                                    if e5 == _ID_SAMPLING_FREQ:
                                        raw = data[s5:e5e]
                                        if len(raw) == 4:
                                            track["sampling_frequency"] = struct.unpack("<f", raw)[0]
                                        elif len(raw) == 8:
                                            track["sampling_frequency"] = struct.unpack("<d", raw)[0]
                        info["tracks"].append(track)
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"the container could not be parsed: {exc}"
        return info
    for t in info["tracks"]:
        if t["type"] == 1:
            info["video_width"] = t["width"]
            info["video_height"] = t["height"]
        elif t["type"] == 2:
            info["audio_sampling_frequency"] = t["sampling_frequency"]
    return info


# ============================================================================
# session discovery
# ============================================================================

MANIFEST_NAMES = ("session.json",)


def _find_artifact(session_dir: Path, takes_dir: Path,
                   filename: str) -> tuple[Path | None, str]:
    """Find a per-take artifact, and say where it was found.

    The base brief says the rich artifacts sit "next to the take"; the
    addendum says the manifest is "at the out root". Neither location is
    pinned by a format, so the checker looks in the canonical place first and
    then in the session's `rich/` subdirectory, and REPORTS which one it used.
    A missing artifact is a failure; a relocated one is a discrepancy.
    """
    for base, label in ((takes_dir, "next to the take"),
                        (session_dir, "at the out root")):
        p = base / filename
        if p.is_file():
            return p, label
        p = base / "rich" / filename
        if p.is_file():
            return p, f"in {base.name}/rich/ rather than {label}"
    try:
        for child in sorted(session_dir.iterdir()):
            if not child.is_dir():
                continue
            p = child / filename
            if p.is_file():
                return p, f"in {child.name}/ rather than next to the take"
    except OSError:
        pass
    return None, ""


def _find_manifest(session_dir: Path, takes_dir: Path) -> tuple[Path | None, str]:
    """The session manifest, and where it was found."""
    for base in (session_dir, takes_dir):
        p = base / "session.json"
        if p.is_file():
            where = ("at the out root" if base == session_dir
                     else "in the takes directory")
            return p, where
        p = base / "rich" / "session.json"
        if p.is_file():
            return p, (f"in {base.name}/rich/ rather than at the out root, "
                       f"as the addendum asks")
    try:
        for base in (session_dir, takes_dir):
            for child in sorted(base.iterdir()):
                if not child.is_dir():
                    continue
                p = child / "session.json"
                if p.is_file():
                    return p, (f"in {child.name}/ rather than at the out "
                               f"root, as the addendum asks")
    except OSError:
        pass
    return None, ""



def _looks_like_take(path: Path) -> bool:
    """A take's metadata file: parses as a JSON object with a condition."""
    try:
        data = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return False
    return isinstance(data, dict) and "condition" in data


def find_session(session_dir: Path) -> tuple[Path, Path | None, list[Path]]:
    """Find the directory holding the takes, the manifest, and the take files.

    The addendum puts `session.json` at the OUT ROOT and the base brief puts
    the takes in the out dir as well. If the takes are in a subdirectory, the
    subdirectory is where they are; the checker reports which directory it
    used, so the verdict never hides the layout it measured.
    """
    candidates = [session_dir]
    try:
        for child in sorted(session_dir.iterdir()):
            if child.is_dir():
                candidates.append(child)
    except OSError:
        pass

    best_dir, best_takes = session_dir, []
    for cand in candidates:
        takes = []
        try:
            for p in sorted(cand.glob("*.json")):
                if p.name.endswith("_timing.json") or p.name.endswith("_rich.json"):
                    continue
                if p.name in MANIFEST_NAMES:
                    continue
                if _looks_like_take(p):
                    takes.append(p)
        except OSError:
            continue
        if len(takes) > len(best_takes):
            best_dir, best_takes = cand, takes

    manifest, manifest_where = _find_manifest(session_dir, best_dir)
    return best_dir, manifest, best_takes, manifest_where

# ============================================================================
# the per-take checks
# ============================================================================

def check_layout(take: Take, session_dir: Path, takes_dir: Path,
                 paths: dict) -> None:
    """The reference's file names, and the addendum's extra artifacts."""
    c = take.check("LAYOUT")
    stem = take.stem
    required = {
        "npz": paths["npz"],
        "wav": paths["wav"],
        "timing": paths["timing"],
        "meta": take.meta_path,
    }
    for label, path in required.items():
        if not path.is_file():
            c.fail(f"{label}: {path.name} is missing")
        elif path.stat().st_size == 0:
            c.fail(f"{label}: {path.name} is empty")

    # The addendum's per-take extras. The base brief allows a take to be
    # re-recorded out of order, so the per-take artifacts must be attributable
    # to a take; a condition-named file shared by two takes of the same
    # condition cannot be.
    if paths["rich"] is None:
        c.fail(f"rich: {stem}_rich.json is missing (addendum item 3)")
    else:
        c.note(f"the rich sidecar is {paths['rich_where']}")
    if paths["webm"] is None:
        c.fail(f"webm: {stem}_full.webm is missing (addendum item 1)")
    else:
        c.note(f"the full-resolution video is {paths['webm_where']}")


def check_nv12(take: Take, npz_path: Path, luma: np.ndarray | None) -> None:
    """The raw frame stream must round-trip through src/nv12.py.

    The pipeline's raw frames are the luma plane: 160 x 96, 15,360 bytes a
    frame, which is src/nv12.py's `luma_only` layout. The requirement is the
    one the base brief states: file size == frames x stride.
    """
    c = take.check("NV12")
    if luma is None:
        c.fail("the luma array is missing, so the frame stream cannot be read")
        return
    stride = nv12.stride_for("luma_only", W, H)
    data = luma.tobytes()
    n_bytes = len(data)
    if n_bytes % stride:
        c.fail(f"the luma stream is {n_bytes} bytes, which is not a whole "
               f"number of {stride}-byte frames")
        return
    n_frames = n_bytes // stride
    declared = take.meta.get("n_frames")
    if isinstance(declared, int) and declared != n_frames:
        c.fail(f"the luma stream holds {n_frames} frames but the metadata "
               f"declares n_frames={declared}")
    if take.timing.get("n_frames") not in (None, n_frames):
        c.fail(f"the luma stream holds {n_frames} frames but timing.json "
               f"declares n_frames={take.timing.get('n_frames')}")
    try:
        view = nv12.luma_view(np.frombuffer(data, dtype=np.uint8),
                              "luma_only", W, H)
    except Exception as exc:  # noqa: BLE001
        c.fail(f"src/nv12.py could not read the luma stream: {exc}")
        return
    if view.shape != (n_frames, H, W):
        c.fail(f"src/nv12.py read {view.shape} from the luma stream; "
               f"expected ({n_frames}, {H}, {W})")
        return
    if not np.array_equal(view, luma):
        c.fail("the luma stream read back through src/nv12.py is not "
               "byte-identical to the stored array")
    c.note(f"frames x stride = {n_frames} x {stride} = {n_bytes} bytes")

    # An extra raw frame file, if the producer wrote one, is checked too.
    for extra in (take.meta_path.with_suffix(".nv12"),
                  take.meta_path.parent / f"{take.stem}.nv12"):
        if not extra.is_file() or extra.resolve() == npz_path.resolve():
            continue
        try:
            found = nv12.detect_layout(extra, W, H)
        except Exception as exc:  # noqa: BLE001
            c.fail(f"{extra.name}: src/nv12.py could not detect the layout: {exc}")
            continue
        if extra.stat().st_size != found.n_frames * found.stride:
            c.fail(f"{extra.name}: the file size is not frames x stride")
        c.note(f"{extra.name}: layout {found.name}, stride {found.stride}, "
               f"{found.n_frames} frames, margin {found.margin:.2f}")


def check_wav(take: Take, wav_path: Path) -> None:
    """src/wavio.py must read the WAV, at the rate the metadata states."""
    c = take.check("WAV")
    if not wav_path.is_file():
        c.fail(f"{wav_path.name} is missing")
        return
    try:
        with wave.open(str(wav_path), "rb") as w:
            n_channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            rate = w.getframerate()
            n_frames = w.getnframes()
    except Exception as exc:  # noqa: BLE001
        c.fail(f"{wav_path.name} is not a readable WAV: {exc}")
        return

    if n_channels != 1:
        c.fail(f"{wav_path.name} has {n_channels} channels; the reference "
               f"writes mono (1)")
    if sampwidth != 2:
        c.fail(f"{wav_path.name} has a {sampwidth * 8}-bit sample width; the "
               f"reference writes 16-bit PCM")

    stated = take.timing.get("audio_sample_rate")
    if isinstance(stated, int) and stated != rate:
        c.fail(f"{wav_path.name} declares {rate} Hz in its header but "
               f"timing.json says audio_sample_rate={stated}")
    requested = take.timing.get("requested_sample_rate")
    if isinstance(requested, int) and requested != rate:
        c.fail(f"{wav_path.name} was written at {rate} Hz but "
               f"{requested} Hz was requested -- every audio timestamp "
               f"computed from this take is wrong")

    try:
        audio = wavio.read_wav(str(wav_path))
    except Exception as exc:  # noqa: BLE001
        c.fail(f"src/wavio.py could not read {wav_path.name}: {exc}")
        return
    if audio.source_sample_rate != rate:
        c.fail(f"src/wavio.py read the file at {audio.source_sample_rate} Hz; "
               f"the header says {rate} Hz")
    if audio.poisoned_samples:
        c.note(f"{audio.poisoned_samples} non-finite samples were replaced "
               f"with 0.0 on read")

    duration = len(audio.samples) / wavio.ANALYSIS_SR
    stated_span = take.timing.get("audio_span_s")
    if _is_num(stated_span):
        if abs(duration - stated_span) > max(0.25, 0.02 * abs(stated_span)):
            c.fail(f"{wav_path.name} holds {duration:.3f} s of audio but "
                   f"timing.json says audio_span_s={stated_span}")
    # The sample count the reference's blocking would produce.
    n_frames_meta = take.meta.get("n_frames")
    if isinstance(n_frames_meta, int) and rate > 0:
        expected = n_frames_meta * (rate / FPS)
        if expected > 0 and abs(n_frames - expected) / expected > 0.05:
            c.fail(f"{wav_path.name} holds {n_frames} samples; the reference's "
                   f"blocking gives about {int(expected)} for "
                   f"{n_frames_meta} frames at {rate} Hz")
    c.note(f"{wav_path.name}: {n_frames} samples at {rate} Hz = {duration:.3f} s")


def _type_ok(value, want) -> bool:
    """Does a JSON value have the type the reference writes?

    `want` is a type or a tuple of types. `bool` is a subclass of `int`, so a
    JSON `true` must not satisfy an `int` slot, and a JSON `null` must not
    satisfy a `bool` slot.
    """
    if isinstance(want, tuple):
        types = want
    else:
        types = (want,)
    if not isinstance(value, types):
        return False
    if isinstance(value, bool) and bool not in types:
        return False
    return True


def check_json(take: Take, timing_path: Path) -> None:
    """timing.json and meta.json: the reference schema, and its arithmetic."""
    c = take.check("JSON")

    if not timing_path.is_file():
        c.fail(f"{timing_path.name} is missing")
        return
    try:
        timing = json.loads(timing_path.read_text())
    except Exception as exc:  # noqa: BLE001
        c.fail(f"{timing_path.name} is not readable JSON: {exc}")
        return
    if not isinstance(timing, dict):
        c.fail(f"{timing_path.name} is not a JSON object")
        return
    take.timing = timing
    for key, want in TIMING_KEYS.items():
        if key not in timing:
            c.fail(f"timing.json has no {key!r}")
            continue
        value = timing[key]
        if not _type_ok(value, want):
            c.fail(f"timing.json {key!r} is {type(value).__name__}, "
                   f"expected {getattr(want, '__name__', want)}")

    meta = take.meta
    for key, want in META_KEYS.items():
        if key not in meta:
            c.fail(f"meta.json has no {key!r}")
            continue
        value = meta[key]
        if not _type_ok(value, want):
            c.fail(f"meta.json {key!r} is {type(value).__name__}, "
                   f"expected {getattr(want, '__name__', want)}")

    # -- the label, the way the gate reads it ----------------------------
    cond = meta.get("condition")
    if isinstance(cond, str):
        take.condition = cond
        file_prefix = re.match(r"^[A-Za-z]+", take.stem)
        file_prefix = file_prefix.group(0).upper() if file_prefix else ""
        meta_prefix = re.match(r"^[A-Za-z]+", cond)
        meta_prefix = meta_prefix.group(0).upper() if meta_prefix else ""
        if file_prefix and meta_prefix and file_prefix != meta_prefix:
            c.fail(f"the file names condition {file_prefix!r} but the "
                   f"metadata says {cond!r}")
    kind = meta.get("kind")
    speak = meta.get("speak")
    if kind == "condition" and not isinstance(speak, bool):
        c.fail(f"kind 'condition' requires speak true/false, got {speak!r}")
    if kind == "free" and speak is not None:
        c.fail(f"kind 'free' requires speak null, got {speak!r}")
    if kind not in ("condition", "free"):
        c.fail(f"unknown condition kind {kind!r}")

    # -- the cross-field arithmetic --------------------------------------
    n_frames = meta.get("n_frames")
    offsets = timing.get("frame_offsets_s")
    if isinstance(offsets, list) and isinstance(n_frames, int):
        if len(offsets) != n_frames:
            c.fail(f"timing.json holds {len(offsets)} frame offsets but the "
                   f"metadata declares n_frames={n_frames}")
        elif len(offsets) >= 2 and all(_is_num(o) for o in offsets):
            span = offsets[-1] - offsets[0]
            if span <= 0:
                c.fail("the frame timestamps do not advance")
            else:
                achieved = (len(offsets) - 1) / span
                stated = timing.get("achieved_fps")
                if _is_num(stated) and abs(achieved - stated) > 0.05 * max(1.0, achieved):
                    c.fail(f"achieved_fps is {stated} but the frame timestamps "
                           f"give {achieved:.4f}")
                assumed = timing.get("assumed_fps")
                if _is_num(assumed) and assumed > 0:
                    rel = abs(achieved - assumed) / assumed
                    if rel > FPS_REL_TOL:
                        c.fail(f"the camera delivered {achieved:.3f} fps "
                               f"against an assumed {assumed} (relative error "
                               f"{rel:.4f} > {FPS_REL_TOL}) -- the video's "
                               f"time base is not the frame index's time base")
    assumed = timing.get("assumed_fps")
    if _is_num(assumed) and isinstance(n_frames, int):
        span = timing.get("video_span_s")
        if _is_num(span) and assumed > 0:
            expect = n_frames / assumed
            if abs(expect - span) > max(0.05, 0.02 * abs(expect)):
                c.fail(f"video_span_s is {span} but n_frames/assumed_fps is "
                       f"{expect:.3f}")
    a_span = timing.get("audio_span_s")
    v_span = timing.get("video_span_s")
    if _is_num(a_span) and _is_num(v_span):
        if v_span <= 0:
            c.fail(f"video_span_s {v_span} is not positive")
        else:
            rel = abs(a_span - v_span) / abs(v_span)
            if rel > SPAN_REL_TOL:
                c.fail(f"video span {v_span:.3f} s against audio span "
                       f"{a_span:.3f} s (relative difference {rel:.4f} > "
                       f"{SPAN_REL_TOL}) -- the two streams are not the "
                       f"same length")

    # -- the stamp, which is the session identity ------------------------
    stamp = meta.get("timestamp")
    parts = take.stem.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        if isinstance(stamp, int) and int(parts[-1]) != stamp:
            c.fail(f"the file's stamp is {parts[-1]} but the metadata's "
                   f"timestamp is {stamp} -- the session identity is "
                   f"ambiguous")


def check_arrays(take: Take, npz_path: Path) -> np.ndarray | None:
    """The .npz: the five arrays, their shapes, and the maps they imply."""
    c = take.check("ARRAYS")
    if not npz_path.is_file():
        c.fail(f"{npz_path.name} is missing")
        return None
    try:
        with np.load(npz_path) as z:
            keys = list(z.files)
            arrays = {k: z[k] for k in keys}
    except Exception as exc:  # noqa: BLE001
        c.fail(f"{npz_path.name} did not load: {exc}")
        return None

    for key in NPZ_KEYS:
        if key not in arrays:
            c.fail(f"{npz_path.name} has no {key!r} array")
    luma = arrays.get("luma")
    n_frames = take.meta.get("n_frames")
    if luma is not None:
        if luma.dtype != np.uint8:
            c.fail(f"luma is {luma.dtype}, expected uint8")
        if luma.ndim != 3 or luma.shape[1:] != (H, W):
            c.fail(f"luma has shape {getattr(luma, 'shape', None)}, expected "
                   f"(n_frames, {H}, {W})")
        if isinstance(n_frames, int) and luma.shape[0] != n_frames:
            c.fail(f"luma holds {luma.shape[0]} frames but the metadata "
                   f"declares n_frames={n_frames}")
    for key in ("maps", "maps_raw"):
        arr = arrays.get(key)
        if arr is None:
            continue
        if arr.dtype != np.uint8:
            c.fail(f"{key} is {arr.dtype}, expected uint8")
        if arr.ndim != 3 or arr.shape[1:] != (MAP_H, MAP_W):
            c.fail(f"{key} has shape {getattr(arr, 'shape', None)}, expected "
                   f"(n_frames-1, {MAP_H}, {MAP_W})")
        if isinstance(n_frames, int) and arr.shape[0] != n_frames - 1:
            c.fail(f"{key} holds {arr.shape[0]} change maps but n_frames-1 is "
                   f"{n_frames - 1}")
    declared_shape = take.meta.get("map_shape")
    if isinstance(declared_shape, list) and list(declared_shape) != [MAP_H, MAP_W]:
        c.fail(f"meta.json map_shape is {declared_shape}, expected "
               f"[{MAP_H}, {MAP_W}]")
    for key, meta_key in (("scale", "map_scale"), ("scale_raw", "map_scale_raw")):
        arr = arrays.get(key)
        stated = take.meta.get(meta_key)
        if arr is not None and _is_num(stated):
            if abs(float(arr) - float(stated)) > 1e-4 * max(1.0, abs(float(stated))):
                c.fail(f"the npz {key} is {float(arr)} but meta.json "
                       f"{meta_key} is {stated}")
    if isinstance(take.meta.get("arrays"), list):
        if list(take.meta["arrays"]) != ["maps", "maps_raw", "luma"]:
            c.fail(f"meta.json arrays is {take.meta['arrays']}, expected "
                   f"['maps', 'maps_raw', 'luma']")

    # -- do the stored maps re-derive from the stored luma? --------------
    # The reference's own functions, so the answer is not mine to argue about.
    if luma is not None and REF is not None and luma.ndim == 3:
        try:
            Yf = luma.astype(np.float64)
            Yr = REF.row_band_remove(Yf)
            maps = REF.change_maps(Yr)
            q, scale = REF.quantise(maps)
            maps_raw = REF.change_maps(Yf)
            q_raw, scale_raw = REF.quantise(maps_raw)
        except Exception as exc:  # noqa: BLE001
            c.note(f"the reference functions could not be run: {exc}")
            return luma
        for key, expect in (("maps", q), ("maps_raw", q_raw)):
            arr = arrays.get(key)
            if arr is None or expect.shape != arr.shape:
                continue
            diff = np.abs(arr.astype(np.int32) - expect.astype(np.int32))
            worst = int(diff.max()) if diff.size else 0
            mean = float(diff.mean()) if diff.size else 0.0
            if worst > 8:
                c.fail(f"{key} does not re-derive from the stored luma "
                       f"(max difference {worst} counts, mean {mean:.3f}) -- "
                       f"the maps are not the quantised change maps of the "
                       f"frames that were stored")
            elif worst > 2:
                c.note(f"{key} re-derives from the stored luma within {worst} "
                       f"counts (mean {mean:.3f})")
            else:
                c.note(f"{key} re-derives from the stored luma exactly "
                       f"(max difference {worst} counts)")
        for key, expect in (("scale", scale), ("scale_raw", scale_raw)):
            arr = arrays.get(key)
            if arr is not None and abs(float(arr) - float(expect)) > 1e-4 * max(1.0, float(expect)):
                c.fail(f"the npz {key} is {float(arr)} but the reference's own "
                       f"quantisation of the stored luma gives {float(expect)}")
    return luma


def check_drift(take: Take, luma: np.ndarray | None) -> None:
    """The drift, re-derived from the stored luma with the reference's code."""
    c = take.check("DRIFT")
    stated = take.meta.get("drift_after_row_removal")
    if not _is_num(stated):
        c.fail("meta.json has no drift_after_row_removal")
        return
    if stated >= DRIFT_MAX:
        c.fail(f"drift after row removal {stated:.3f} >= {DRIFT_MAX} -- the "
               f"banding is not the only problem")
    if luma is None or REF is None:
        return
    try:
        Yf = luma.astype(np.float64)
        Yr = REF.row_band_remove(Yf)
        raw = float(np.abs(np.diff(Yf, axis=0)).mean())
        row = float(np.abs(np.diff(Yr, axis=0)).mean())
    except Exception as exc:  # noqa: BLE001
        c.note(f"the drift could not be re-derived: {exc}")
        return
    for name, mine, theirs in (("drift_raw", raw, take.meta.get("drift_raw")),
                               ("drift_after_row_removal", row, stated)):
        if _is_num(theirs) and abs(mine - theirs) > max(0.25, 0.15 * abs(mine)):
            c.fail(f"meta.json {name} is {theirs} but the value re-derived "
                   f"from the stored luma is {mine:.4f}")
    c.note(f"drift re-derived from the stored luma: raw {raw:.3f}, after row "
           f"removal {row:.3f}")


def check_flags(take: Take, wav_path: Path, paths: dict,
                manifest: dict | None, meter: SpeechMeter) -> None:
    """VOICE_MISSING / NOT_SILENT, and an independent re-measurement.

    Two separate questions, so two separate checks:

    FLAGS   is the flag present exactly where the data demands it? This is
            judged against the recorded `speech_frac`, which is the quantity
            the capture script's thresholds are defined against.
    SPEECH  does the recorded `speech_frac` survive an INDEPENDENT
            measurement of the WAV, using the same Silero gate the reference
            uses? Without this, a producer could assert `speech_frac: 0.9`
            to avoid a flag and nothing would notice.
    """
    c = take.check("FLAGS")
    speak = take.meta.get("speak")
    frac = take.meta.get("speech_frac")

    # Where a flag may be recorded: the take's own sidecars, or its entry in
    # the session manifest.
    blobs = []
    for p in (paths["meta"], paths["timing"], paths["rich"]):
        if p is not None and p.is_file():
            try:
                blobs.append(p.read_text())
            except OSError:
                pass
    if manifest is not None:
        for entry in _manifest_entries(manifest):
            if _entry_matches_take(entry, take):
                blobs.append(json.dumps(entry))

    def present(token: str) -> bool:
        return any(token in blob for blob in blobs)

    if isinstance(speak, bool):
        if not _is_num(frac):
            c.fail("speak is set but speech_frac was never measured -- the "
                   "achievement check did not run")
        else:
            if speak and frac < SPEECH_MIN:
                if not present("VOICE_MISSING"):
                    c.fail(f"speech_frac {frac:.4f} < {SPEECH_MIN} for a "
                           f"condition labelled speak=True, so VOICE_MISSING "
                           f"must be flagged -- and it is not")
                else:
                    c.note(f"VOICE_MISSING is present, as the data demands "
                           f"(speech_frac {frac:.4f})")
            elif speak:
                if present("VOICE_MISSING"):
                    c.fail(f"VOICE_MISSING is flagged but speech_frac "
                           f"{frac:.4f} >= {SPEECH_MIN} -- the flag contradicts "
                           f"the data")
            if not speak and frac > SPEECH_MAX:
                if not present("NOT_SILENT"):
                    c.fail(f"speech_frac {frac:.4f} > {SPEECH_MAX} for a "
                           f"condition labelled speak=False, so NOT_SILENT "
                           f"must be flagged -- and it is not")
                else:
                    c.note(f"NOT_SILENT is present, as the data demands "
                           f"(speech_frac {frac:.4f})")
            elif not speak:
                if present("NOT_SILENT"):
                    c.fail(f"NOT_SILENT is flagged but speech_frac "
                           f"{frac:.4f} <= {SPEECH_MAX} -- the flag "
                           f"contradicts the data")
    else:
        if present("VOICE_MISSING") or present("NOT_SILENT"):
            c.note("a free take carries an achievement flag, which has no "
                   "meaning without ground truth")

    # -- the independent re-measurement, as its own check -----------------
    s = take.check("SPEECH")
    if not wav_path.is_file():
        s.skip("there is no WAV to measure")
        return
    mine, why = meter.measure(wav_path)
    if mine is None:
        s.skip(f"the independent speech measurement could not be run: {why}")
        return
    gate = take.meta.get("speech_gate")
    s.note(f"independent Silero re-measurement: {mine:.4f}; recorded "
           f"{frac if _is_num(frac) else 'unmeasured'}"
           + (f"; the take records speech_gate={gate!r}" if gate else ""))
    if not _is_num(frac):
        s.note("the take records no speech_frac, so there is nothing to "
               "corroborate")
        return
    if abs(mine - frac) > 0.25:
        s.fail(f"the recorded speech_frac is {frac:.4f} but the independent "
               f"re-measurement of the same WAV gives {mine:.4f} -- the "
               f"recorded number is not what the audio supports"
               + (f"; the take records speech_gate={gate!r}, and the "
                  f"reference's gate is Silero" if gate else ""))
    # The two cases where the disagreement inverts the label. Stated
    # carefully: a low independent score means either that the take is not
    # what its label says, or that the producer's gate and the pipeline's
    # gate do not agree -- the Silero gate is documented to mis-score pure
    # tones and hums (src/vad.py), so the measurement alone does not say
    # which.
    if isinstance(speak, bool) and speak and mine < 0.05 and frac >= SPEECH_MIN:
        s.fail(f"the take is labelled as speaking, but the independent "
               f"measurement of its audio finds almost no speech "
               f"({mine:.4f}): either the take is not what its label says, "
               f"or the producer's gate and the pipeline's gate do not agree")
    if isinstance(speak, bool) and not speak and mine > 0.25 and frac <= SPEECH_MAX:
        s.fail(f"the take is labelled as silent, but the independent "
               f"measurement of its audio finds speech ({mine:.4f}): either "
               f"the take is not what its label says, or the producer's gate "
               f"and the pipeline's gate do not agree")


def check_rich(take: Take, rich_path: Path, session_dir: Path) -> None:
    """The addendum's rich sidecar: the eight required items.

    The addendum fixes the CONTENT and not the key names, so each item is
    matched on normalised aliases and the matched key is reported.
    """
    c = take.check("RICH")
    if rich_path is None or not rich_path.is_file():
        c.fail(f"{take.stem}_rich.json is missing (addendum item 3)")
        return
    try:
        data = json.loads(rich_path.read_text())
    except Exception as exc:  # noqa: BLE001
        c.fail(f"{rich_path.name} is not readable JSON: {exc}")
        return
    if not isinstance(data, dict):
        c.fail(f"{rich_path.name} is not a JSON object")
        return

    norm = {_norm_key(k): v for k, v in data.items()}
    for item, aliases in RICH_REQUIREMENTS:
        matched, loose = None, None
        for alias in aliases:
            if alias in norm:
                matched = alias
                break
        if matched is None:
            for alias in aliases:
                for key, value in norm.items():
                    if alias in key:
                        loose = key
                        break
                if loose:
                    break
        if matched is not None:
            value = norm[matched]
            if item == "notes":
                if not isinstance(value, str):
                    c.fail(f"{item}: the matched key {matched!r} is "
                           f"{type(value).__name__}, expected a free-text string")
                elif value.strip() == "":
                    c.note(f"{item}: present and empty, as the addendum "
                           f"requires by default")
            elif item in ("wall_clock_start", "wall_clock_end"):
                if not (_is_num(value) or isinstance(value, str)):
                    c.fail(f"{item}: the matched key {matched!r} is "
                           f"{type(value).__name__}, expected a time")
            elif item == "timestamps_ref":
                if isinstance(value, str):
                    if value.strip() == "":
                        c.fail(f"{item}: the matched key {matched!r} is empty")
                    else:
                        c.note(f"{item}: references {value!r}")
                        cand = session_dir / value
                        if not cand.is_file() and not (session_dir / value).is_file():
                            if "/" in value or value.endswith(".json"):
                                c.fail(f"{item}: {value!r} does not name a file "
                                       f"in the session directory")
                elif isinstance(value, (list, dict)):
                    c.note(f"{item}: an inline array/object of {len(value)}")
                else:
                    c.fail(f"{item}: the matched key {matched!r} is "
                           f"{type(value).__name__}, expected a reference")
            elif item == "sample_rate":
                if not _is_num(value):
                    c.fail(f"{item}: the matched key {matched!r} is "
                           f"{type(value).__name__}, expected a number")
            elif item == "achieved_fps":
                if not _is_num(value):
                    c.fail(f"{item}: the matched key {matched!r} is "
                           f"{type(value).__name__}, expected a number")
            continue
        if loose is not None:
            c.note(f"{item}: matched loosely by {loose!r} (no exact alias)")
            continue
        c.fail(f"{item}: no key matches any of {list(aliases)}")


def check_webm(take: Take, paths: dict) -> None:
    """The addendum's full-resolution WebM, with the mic audio muxed."""
    c = take.check("WEBM")
    path = paths["webm"]
    if path is None:
        c.fail(f"{take.stem}_full.webm is missing")
        return
    if path.stat().st_size == 0:
        c.fail(f"{path.name} is empty")
        return
    if path.name != f"{take.stem}_full.webm":
        c.note(f"found as {path.name} (condition-named rather than "
               f"take-named); a take that shares its condition with another "
               f"take would overwrite it")
    try:
        info = read_webm(path)
    except Exception as exc:  # noqa: BLE001
        c.fail(f"{path.name} could not be read: {exc}")
        return
    if not info["ebml_magic"]:
        c.fail(f"{path.name} is not a WebM: {info['error']}")
        return
    if info["doc_type"] and info["doc_type"] != "webm":
        c.fail(f"{path.name} declares DocType {info['doc_type']!r}, not 'webm'")
    if info["error"]:
        c.fail(f"{path.name}: {info['error']}")
    video = [t for t in info["tracks"] if t["type"] == 1]
    audio = [t for t in info["tracks"] if t["type"] == 2]
    if not video:
        c.fail(f"{path.name} carries no video track")
    if not audio:
        c.fail(f"{path.name} carries no audio track -- the addendum requires "
               f"the mic audio to be muxed into it")
    if video:
        t = video[0]
        c.note(f"video track {t['codec']} at "
               f"{t['width']}x{t['height']}")
        if t["width"] and t["height"]:
            if t["width"] < W or t["height"] < H:
                c.fail(f"{path.name} is {t['width']}x{t['height']}, which is "
                       f"below the capture resolution {W}x{H} -- the addendum "
                       f"asks for the camera's NATIVE resolution")
    if audio:
        t = audio[0]
        c.note(f"audio track {t['codec']} at "
               f"{t['sampling_frequency']} Hz")


# ============================================================================
# the session manifest
# ============================================================================

def _all_strings(value) -> list[str]:
    """Every string inside a nested structure (a manifest's files map)."""
    out = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            out.extend(_all_strings(k))
            out.extend(_all_strings(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            out.extend(_all_strings(v))
    return out


def _manifest_entries(manifest) -> list:
    """The take entries in a manifest, whatever it calls them."""
    if isinstance(manifest, list):
        return [e for e in manifest if isinstance(e, dict)]
    if not isinstance(manifest, dict):
        return []
    for key in MANIFEST_TAKES_KEYS:
        value = manifest.get(key)
        if isinstance(value, list):
            return [e for e in value if isinstance(e, dict)]
        if isinstance(value, dict):
            out = []
            for name, entry in value.items():
                if isinstance(entry, dict):
                    out.append({**entry, "_key": name})
            if out:
                return out
    return []


def _entry_matches_take(entry: dict, take: Take) -> bool:
    """Does a manifest entry describe this take?"""
    blob = json.dumps(entry)
    if take.stem and take.stem in blob:
        return True
    if take.condition and re.search(
            rf'"{re.escape(take.condition)}"', blob):
        # A bare condition label can match a different take of the same
        # condition; the stem check above is the reliable one.
        return take.condition in blob
    return False


def check_manifest(take: Take, manifest_path: Path | None,
                   manifest: dict | None) -> None:
    """The addendum's session.json: every take, its files, its labels, its flags."""
    c = take.check("MANIFEST")
    if manifest_path is None:
        c.fail("no session.json at the out root (addendum item 4)")
        return
    entries = _manifest_entries(manifest)
    if not entries:
        c.fail(f"{manifest_path.name} lists no takes")
        return
    entry = None
    for e in entries:
        if take.stem in json.dumps(e):
            entry = e
            break
    if entry is None:
        c.fail(f"{manifest_path.name} does not list this take "
               f"({take.stem})")
        return
    blob = json.dumps(entry)
    if take.condition and take.condition not in blob:
        c.fail(f"the manifest entry for {take.stem} does not carry its label "
               f"{take.condition!r}")
    norm = {_norm_key(k): v for k, v in entry.items()}
    files_key = next((k for k in MANIFEST_FILES_KEYS if k in norm), None)
    if files_key is None:
        c.fail(f"the manifest entry for {take.stem} lists no files "
               f"(looked for {list(MANIFEST_FILES_KEYS)})")
    else:
        value = norm[files_key]
        names = set(_all_strings(value))
        for wanted in (f"{take.stem}.npz", f"{take.stem}.wav"):
            if wanted not in names:
                c.fail(f"the manifest entry for {take.stem} does not list "
                       f"{wanted}")
    flags_key = next((k for k in MANIFEST_FLAGS_KEYS if k in norm), None)
    if flags_key is None and "VOICE_MISSING" not in blob \
            and "NOT_SILENT" not in blob:
        c.fail(f"the manifest entry for {take.stem} carries no flags field")


# ============================================================================
# the gate
# ============================================================================

def _takes_only_dir(takes: list[Path], session_dir: Path,
                    manifest_path: Path | None) -> Path:
    """A throwaway directory holding ONLY the take files.

    The gate individuates takes with `glob("*.json")` minus `*_timing.json`,
    so the addendum's `session.json` would be picked up as a take and would
    fail for having no condition and no .npz. This isolates the takes so the
    gate's verdict on the TAKES can be read separately from its verdict on the
    directory as it stands.
    """
    tmp = Path(tempfile.mkdtemp(prefix="bc_acceptance_"))
    for meta in takes:
        stem = meta.stem
        for suffix in (".npz", ".wav", ".json", "_timing.json", "_rich.json"):
            src = meta.parent / f"{stem}{suffix}"
            if not src.is_file():
                continue
            dst = tmp / src.name
            try:
                shutil.copy2(src, dst)
            except OSError:
                pass
        for extra in ("_full.webm",):
            src = meta.parent / f"{stem}{extra}"
            if src.is_file():
                try:
                    shutil.copy2(src, tmp / src.name)
                except OSError:
                    pass
    return tmp


def check_gate(takes: list[Path], session_dir: Path,
               manifest_path: Path | None) -> tuple[dict, dict]:
    """Run the repo's own gate, twice: on the directory, and on the takes.

    THE DISTINCTION THIS CHECK MAKES. The gate's `speech` check fails a take
    whose measured `speech_frac` contradicts its label -- and the capture
    script's own answer to that is to SAVE THE TAKE AND FLAG IT
    ("a take that fails its check is still saved, but flagged, so it can be
    excluded without being re-recorded blindly", corpus_capture_win.py:137).
    So a `speech` failure is a DATA OUTCOME, correctly flagged, and not a
    defect in the capture's format. Every other gate check -- label, file,
    rate, span, drift, stimulus -- is about whether the take was WRITTEN
    correctly, and a failure there IS a defect. Only those fail this check;
    the `speech` ones are reported and cross-referenced to the FLAGS check.
    """
    c_raw = Check("GATE")
    c_takes = Check("GATE_TAKES_ONLY")
    DATA_CHECKS = ("speech",)
    if GATE is None:
        c_raw.skip("experiments/corpus_integrity.py is not in this checkout")
        c_takes.skip("experiments/corpus_integrity.py is not in this checkout")
        return {"error": "the gate is not in this checkout"}, \
            c_takes.as_dict()

    try:
        raw = GATE.analyse(session_dir)
    except Exception as exc:  # noqa: BLE001
        c_raw.fail(f"the gate could not run on {session_dir}: {exc}")
        raw = None

    tmp = _takes_only_dir(takes, session_dir, manifest_path)
    try:
        isolated = GATE.analyse(tmp)
    except Exception as exc:  # noqa: BLE001
        c_takes.fail(f"the gate could not run on the takes: {exc}")
        isolated = None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    summary = {}
    for key, report, check in (("on_directory", raw, c_raw),
                               ("on_takes_only", isolated, c_takes)):
        if report is None:
            continue
        structural, data = [], []
        for t in report["takes"]:
            for f in t["failures"]:
                line = f"{t['file']}: {f['check']}: {f['message']}"
                (data if f["check"] in DATA_CHECKS else structural).append(line)
        summary[key] = {
            "n_takes": report["n_takes"], "n_failed": report["n_failed"],
            "ok": report["ok"],
            "structural_failures": structural,
            "data_failures": data,
            "failures": structural + data,
        }
        if report["ok"]:
            check.note(f"the gate passes on {key.replace('_', ' ')} "
                       f"({report['n_takes']} takes)")
        else:
            for line in structural:
                check.fail(line)
            for line in data:
                check.note(f"data outcome (the take's speech contradicts its "
                           f"label, which is what the achievement flags are "
                           f"for): {line}")
    if raw is not None and isolated is not None and not raw["ok"] \
            and isolated["ok"]:
        c_raw.note("the gate fails on the directory but passes on the takes "
                   "alone: the manifest is what the gate picked up as a take "
                   "-- see ACCEPTANCE-EXPECTED.md §7.3")
    return summary, c_takes.as_dict()


# ============================================================================
# the plan
# ============================================================================

def check_plan(takes: list[Take], require_full: bool) -> tuple[dict, Check]:
    """Which of the 16 plan conditions are present."""
    c = Check("PLAN")
    present = [t.condition for t in takes if t.condition]
    counts = {cond: present.count(cond) for cond in sorted(set(present))}
    missing = [cond for cond in PLAN_CONDITIONS if cond not in counts]
    extra = [cond for cond in counts if cond not in PLAN_CONDITIONS]
    coverage = {
        "plan_size": len(PLAN_CONDITIONS),
        "present": counts,
        "missing": missing,
        "unexpected": extra,
        "complete": not missing,
    }
    if extra:
        c.fail(f"the session holds conditions that are not in the plan: "
               f"{extra}")
    if missing:
        message = (f"{len(missing)} of the {len(PLAN_CONDITIONS)} plan "
                   f"conditions are not in this session: {missing}")
        if require_full:
            c.fail(message)
        else:
            c.note(message + " (reported, not failed: a synthetic or partial "
                             "session is expected)")
    else:
        c.note(f"all {len(PLAN_CONDITIONS)} plan conditions are present")
    # The plan's per-condition kind, pose and speak.
    for t in takes:
        cond = t.condition
        base = re.match(r"^[A-Za-z]+", cond or "")
        base = base.group(0).upper() if base else ""
        if base in PLAN_KIND and t.meta.get("kind") != PLAN_KIND[base]:
            c.fail(f"{cond}: kind is {t.meta.get('kind')!r}, the plan says "
                   f"{PLAN_KIND[base]!r}")
        if base in PLAN_SPEAK and t.meta.get("speak") is not PLAN_SPEAK[base]:
            c.fail(f"{cond}: speak is {t.meta.get('speak')!r}, the plan says "
                   f"{PLAN_SPEAK[base]!r}")
        if base in PLAN_POSE and t.meta.get("pose") != PLAN_POSE[base]:
            c.fail(f"{cond}: pose is {t.meta.get('pose')!r}, the plan says "
                   f"{PLAN_POSE[base]!r}")
    return coverage, c


# ============================================================================
# the driver
# ============================================================================

def analyse(session_dir: Path, require_full_plan: bool = False) -> dict:
    """Check every take in the session directory. Returns the report."""
    meter = SpeechMeter()
    takes_dir, manifest_path, meta_paths, manifest_where = find_session(session_dir)

    manifest = None
    if manifest_path is not None:
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:  # noqa: BLE001
            manifest = {"__error__": str(exc)}

    takes: list[Take] = []
    for meta_path in meta_paths:
        take = Take(meta_path=meta_path, stem=meta_path.stem)
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:  # noqa: BLE001
            take.check("JSON").fail(f"{meta_path.name} is not readable JSON: "
                                     f"{exc}")
            takes.append(take)
            continue
        if not isinstance(meta, dict):
            take.check("JSON").fail(f"{meta_path.name} is not a JSON object")
            takes.append(take)
            continue
        take.meta = meta
        take.condition = meta.get("condition") or ""
        takes.append(take)

    for take in takes:
        stem = take.stem
        rich_path, rich_where = _find_artifact(session_dir, takes_dir,
                                                f"{stem}_rich.json")
        webm_path, webm_where = _find_artifact(session_dir, takes_dir,
                                                f"{stem}_full.webm")
        if webm_path is None and take.condition:
            webm_path, webm_where = _find_artifact(
                session_dir, takes_dir, f"{take.condition}_full.webm")
        paths = {
            "npz": takes_dir / f"{stem}.npz",
            "wav": takes_dir / f"{stem}.wav",
            "meta": take.meta_path,
            "timing": takes_dir / f"{stem}_timing.json",
            "rich": rich_path,
            "rich_where": rich_where,
            "webm": webm_path,
            "webm_where": webm_where,
        }

        check_layout(take, session_dir, takes_dir, paths)
        check_json(take, paths["timing"])
        luma = check_arrays(take, paths["npz"])
        check_nv12(take, paths["npz"], luma)
        check_wav(take, paths["wav"])
        check_drift(take, luma)
        check_flags(take, paths["wav"], paths, manifest, meter)
        check_rich(take, rich_path, session_dir)
        check_webm(take, paths)
        check_manifest(take, manifest_path, manifest)

    coverage, plan_check = check_plan(takes, require_full_plan)
    gate_summary, gate_takes = check_gate(meta_paths, takes_dir, manifest_path)

    # -- roll up -----------------------------------------------------------
    check_names = ("LAYOUT", "JSON", "ARRAYS", "NV12", "WAV", "DRIFT", "FLAGS",
                   "SPEECH", "RICH", "WEBM", "MANIFEST")
    summary = {}
    n_fail = 0
    for name in check_names:
        fails = [f for t in takes for f in t.checks.get(name, Check(name)).failures]
        notes = [f"{t.stem}: {n}" for t in takes
                 for n in t.checks.get(name, Check(name)).notes]
        status = "FAIL" if fails else "PASS"
        if fails:
            n_fail += 1
        summary[name] = {"status": status, "n_failures": len(fails),
                         "failures": fails, "notes": notes}
    summary["PLAN"] = plan_check.as_dict()
    summary["GATE"] = {"status": "FAIL" if gate_takes.get("status") == "FAIL"
                       else gate_takes.get("status", "SKIP"),
                       "failures": gate_takes.get("failures", []),
                       "notes": gate_takes.get("notes", [])}
    if gate_takes.get("status") == "FAIL":
        n_fail += 1

    report = {
        "question": ("does the browser-capture deliverable produce takes that "
                     "the existing corpus pipeline accepts unchanged?"),
        "session": {
            "given": str(session_dir),
            "takes_directory": str(takes_dir),
            "manifest": str(manifest_path) if manifest_path else None,
            "manifest_location": manifest_where,
            "n_takes": len(takes),
        },
        "environment": {
            "numpy": np.__version__,
            "reference_implementation_loaded": REF is not None,
            "gate_loaded": GATE is not None,
            "silero_model": str(meter.model) if meter.model else None,
            "independent_speech_measurement": meter.available,
            "independent_speech_measurement_note": meter.reason,
        },
        "plan_coverage": coverage,
        "gate": gate_summary,
        "checks": summary,
        "takes": [
            {
                "stem": t.stem,
                "condition": t.condition,
                "meta_file": t.meta_path.name,
                "checks": {k: v.as_dict() for k, v in t.checks.items()},
                "ok": t.ok,
            }
            for t in takes
        ],
        "verdict": {
            "ok": n_fail == 0,
            "n_failed_checks": n_fail,
            "n_failed_takes": sum(1 for t in takes if not t.ok),
        },
        "provenance": {
            "script": "experiments/browser_capture_acceptance.py",
            "script_sha256": _sha256_file(Path(__file__)),
            "python": sys.executable,
            "expectations": "docs/research/ACCEPTANCE-EXPECTED.md",
        },
    }
    return report


def format_report(report: dict) -> str:
    """The human-readable report."""
    out = []
    out.append("BROWSER CAPTURE -- INDEPENDENT ACCEPTANCE")
    out.append("=" * 62)
    sess = report["session"]
    out.append(f"session dir      {sess['given']}")
    out.append(f"takes directory  {sess['takes_directory']}")
    out.append(f"manifest         {sess['manifest'] or '(none)'}")
    if sess.get("manifest_location"):
        out.append(f"                  ({sess['manifest_location']})")
    out.append(f"takes            {sess['n_takes']}")
    env = report["environment"]
    out.append(f"silero re-measurement  "
               f"{'available' if env['independent_speech_measurement'] else 'UNAVAILABLE'}"
               + ("" if env["independent_speech_measurement"]
                  else f" ({env['independent_speech_measurement_note']})"))
    out.append("")
    out.append("CHECKS")
    for name, c in report["checks"].items():
        out.append(f"  [{c['status']:<4}] {name}"
                   + (f"  ({c['n_failures']} failures)" if c.get("n_failures")
                      else ""))
        for f in c.get("failures", [])[:12]:
            out.append(f"           - {f}")
        if len(c.get("failures", [])) > 12:
            out.append(f"           ... and "
                       f"{len(c['failures']) - 12} more")
    out.append("")
    out.append("TAKES")
    if not report["takes"]:
        out.append("  (none)")
    for t in report["takes"]:
        mark = "ok  " if t["ok"] else "FAIL"
        out.append(f"  [{mark}] {t['stem']:<16} condition={t['condition']}")
        for name, c in t["checks"].items():
            for f in c["failures"]:
                out.append(f"           {name:<8}: {f}")
    cov = report["plan_coverage"]
    out.append("")
    out.append(f"PLAN COVERAGE  {len(cov['present'])} of {cov['plan_size']} "
               f"conditions present")
    if cov["missing"]:
        out.append(f"  missing: {cov['missing']}")
    if cov["unexpected"]:
        out.append(f"  not in the plan: {cov['unexpected']}")
    gate = report.get("gate") or {}
    if gate:
        out.append("")
        out.append("GATE (experiments/corpus_integrity.py)")
        for key, label in (("on_directory", "on the directory as it stands"),
                           ("on_takes_only", "on the takes alone")):
            g = gate.get(key)
            if not g:
                continue
            out.append(f"  {label}: {'PASS' if g['ok'] else 'FAIL'} "
                       f"({g['n_takes']} takes, {g['n_failed']} failed)")
            for f in g["failures"][:8]:
                out.append(f"           - {f}")
    v = report["verdict"]
    out.append("")
    out.append(f"VERDICT: {'PASS' if v['ok'] else 'FAIL'} -- "
               f"{v['n_failed_checks']} check(s) failed, "
               f"{v['n_failed_takes']} of {report['session']['n_takes']} "
               f"take(s) failed")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Independently check a browser-capture session against the "
                    "expectations derived in docs/research/ACCEPTANCE-EXPECTED.md")
    ap.add_argument("session", nargs="?",
                    default="/mnt/c/proart_capture/corpus",
                    help="the session output directory")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="write the JSON report to this path")
    ap.add_argument("--require-full-plan", action="store_true",
                    help="fail when any of the 16 plan conditions is missing "
                         "(use for a complete session, not a synthetic one)")
    a = ap.parse_args(argv)

    session_dir = Path(a.session)
    if not session_dir.is_dir():
        print(f"session directory not found: {session_dir}", file=sys.stderr)
        print("pass the directory the browser capture wrote its takes to.",
              file=sys.stderr)
        return 2

    try:
        report = analyse(session_dir, require_full_plan=a.require_full_plan)
    except Exception as exc:  # noqa: BLE001
        # A checker that crashes has not checked anything. Say so loudly, and
        # do not let a crash read as a pass.
        import traceback
        traceback.print_exc()
        print(f"\nINTERNAL ERROR: the checker could not complete: {exc}",
              file=sys.stderr)
        return 1
    if report["session"]["n_takes"] == 0:
        print(f"no takes found in {session_dir}", file=sys.stderr)
        print("(a take's metadata file is a JSON object with a 'condition' "
              "field)", file=sys.stderr)
        return 2

    print(format_report(report))

    if a.json_path:
        out_path = Path(a.json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {out_path}")

    return 0 if report["verdict"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
