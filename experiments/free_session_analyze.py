#!/usr/bin/env python3
"""
free_session_analyze.py -- derive the corpus labels AFTERWARDS, from a free
session, and slice it into takes the integrity gate eats.

WHY THIS EXISTS.

Casey: "make it so I can click record and just talk freely and turn and
things, and use OpenCV and other things to see what I was doing -- you just
need me to do a bunch of stuff in front of the camera."

He is right, and the choreographed plan was the wrong instrument: a
manufactured label only has the value of the moment it was manufactured, and
in the browser it broke. So the page only RECORDS, and the labels are
DERIVED here, offline, from the recording.

WHAT IT DOES.

1. Reads a free-session folder (see INPUT below).
2. Derives, per frame, a FACE STATE (facing / turned / absent) and a
   SPEECH STATE (speaking / silent).
3. Groups the frames into contiguous segments and prints/writes a segment
   table with a confidence per label.
4. Slices each usable segment into a take in EXACTLY the corpus take format
   `corpus_capture_win.py --plan` writes -- the same npz layout, the same
   WAV, the same timing/meta schemas -- labelled by the derived condition,
   with the provenance and the confidence in the metadata.

THE TWO MEASURES, AND WHY THEY ARE THE ONES.

FACE STATE -- MediaPipe FaceMesh landmarks, and the nose-to-eye asymmetry:

    sym = (|nose - eye_outer_left| - |nose - eye_outer_right|)
          / (|nose - eye_outer_left| + |nose - eye_outer_right|)

A face square to the camera puts the nose equidistant from the two outer
eyes, so sym is near zero; a turned head breaks that symmetry. Measured on
every frame of the captures that exist today (2700 frames):

    A (facing)  sym -0.021 +/- 0.060   range -0.177 .. +0.105
    B (turned)  sym +0.350 +/- 0.034   range +0.246 .. +0.423
    C (facing)  sym +0.017 +/- 0.014   range -0.037 .. +0.059

A threshold of 0.18 -- the midpoint of the measured gap between A's maximum
(+0.105) and B's minimum (+0.246) -- misclassifies NONE of the 2700 frames.

The three-dimensional cue (the nose's depth against the eyes') was measured
too and REJECTED: it reads -13.4 on A, -9.3 on B and -11.5 on C, so it
confounds head direction with whether the mouth is open. That is the same
trap `docs/research/MOUTH-APERTURE-ALIGNMENT.md` documents for the mouth
aperture, and it is why the asymmetry is used instead.

SPEECH STATE -- the repository's own energy VAD, `src/energy.py`
(`rms_series` + `segment(adaptive=True)`), on the session's WAV, mapped onto
the video frames through the per-frame timestamps. The adaptive threshold is
the repo's measured fix for a transient that would otherwise silence the
rest of a session. Measured:

    A (speaking)  0.670    B (speaking)  0.627
    C (silent)    0.007    empty1 0.046

The SAME thresholds the integrity gate uses (0.15 to call it speech, 0.05 to
call it silence) decide the derived label, so an emitted take and the gate
agree by construction rather than by luck. A segment whose speech fraction
lands between the two is AMBIGUOUS, and is dropped and reported rather than
mislabelled.

INPUT -- a free-session folder, as the free-record page writes it. Read
from `browser_capture/server.py:save_free_session` and
`browser_capture/free_record.html`, NOT guessed:

    out/free_sessions/<stamp>/
        meta.json        tool "browser_free_record"; n_frames; the luma
                        geometry; achieved_video_fps; the audio's rate
        luma.raw         RAW LUMA: uint8, row-major, 160x96, 15,360 bytes
                        a frame, frames concatenated
        timestamps.json  {"wall_clock_start": ..., "frame_offsets_ms": [...]}
                        -- MILLISECONDS, performance.now() minus frame 0
        audio.wav        mono 16-bit at the AudioContext's own rate (48 kHz
                        typically), NOT resampled
        video.webm       optional, the full-resolution recording

The script capture's NV12 files are also accepted, and there the layout is
DETECTED from the bytes by `src/nv12.py`, exactly as the rest of this
repository does -- three strides divide those files evenly, so only the
content can say which one is real.

The reader is deliberately liberal about FILE NAMES and strict about
CONTENT: the declared luma geometry selects the raw-luma path, and
everything else falls through to layout detection. The conventions the page
already follows (the same BT.601 luma, the same WAV header, the same
per-frame timestamps) are the ones the reference implementation uses, so a
derived take is a take.

OUTPUT --

    <out dir>/A1_<stamp>.npz        the slice, in corpus take format
                   .wav
                   _timing.json
                   .json
    <out dir>/rich/free_session_segments.json
                                      the segment table -- in a
                                      SUBDIRECTORY, because the gate
                                      individuates every root-level *.json
                                      as a take
    printed: the segment table and the summary

USAGE:

    /home/proart/projects/gpu-venv/bin/python experiments/free_session_analyze.py \\
        /mnt/c/proart_capture/corpus/free_sessions/<stamp> --out <dir> --gate

    # the recovery report over the captures that exist today:
    /home/proart/projects/gpu-venv/bin/python experiments/free_session_analyze.py \\
        --recover /mnt/c/proart_capture --slices /tmp/derived

    # the slicer's unit tests, on synthetic arrays:
    /home/proart/projects/gpu-venv/bin/python experiments/free_session_analyze.py \\
        --selftest

INTERPRETER. MediaPipe is needed for the face state, and it lives in the
GPU venv (`/home/proart/projects/gpu-venv`), which also has numpy, cv2,
onnxruntime and scipy. Without it the module still imports and the speech,
segmentation and slicing halves still run -- the face state is reported as
unavailable rather than guessed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import nv12  # noqa: E402
from src import wavio  # noqa: E402

# -- the capture script's constants (corpus_capture_win.py:221-224) ----------
FRAME_W, FRAME_H = 160, 96
LUMA_BYTES = FRAME_W * FRAME_H
MAP_W, MAP_H = 80, 48
RATE = 16000
FPS = 30

# -- the thresholds ----------------------------------------------------------
# The head-direction threshold: the midpoint of the measured gap between the
# facing captures' maximum (+0.105) and the turned capture's minimum
# (+0.246). Zero errors on all 2700 frames of A, B and C.
TURNED_THRESHOLD = 0.18
# The measured range of the turned capture, used to scale the confidence.
TURNED_FULL = 0.42
# The gate's own speech thresholds (corpus_integrity.py:99-100). The derived
# label uses the SAME numbers, so an emitted take and the gate agree.
SPEECH_MIN = 0.15
SPEECH_MAX = 0.05

# -- the segmenter's own defaults -------------------------------------------
MIN_LABEL_S = 0.5       # a state shorter than this is noise, not a state
MIN_SEGMENT_S = 3.0     # a segment shorter than this is not a take
UPSCALE = 4             # the FaceMesh input scale (measured: 100% detection)

PROVENANCE = "derived: free_session_analyze"

# The derived condition for each (face, speech) pair.
CONDITION_FOR = {
    ("facing", "speaking"): "A",
    ("turned", "speaking"): "B",
    ("facing", "silent"): "C",
    ("turned", "silent"): "D",
}
POSE_FOR = {"facing": "facing", "turned": "turned"}


# ============================================================================
# the reference implementation, imported so that the derived maps are the
# pipeline's own maps, not a re-implementation of them
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
    p = ROOT / "experiments" / "corpus_capture_win.py"
    if not p.is_file():
        return None
    try:
        return _load_module(p, "_ref_capture_win")
    except Exception:  # noqa: BLE001
        return None


REF = _reference()


def _gate():
    """The integrity gate, imported so the derived takes are checked by the
    SAME code that checks the script's takes."""
    p = ROOT / "experiments" / "corpus_integrity.py"
    if not p.is_file():
        return None
    try:
        return _load_module(p, "_ref_corpus_integrity")
    except Exception:  # noqa: BLE001
        return None


def run_gate(corpus_dir: Path) -> dict:
    """Run the integrity gate over a directory of takes."""
    g = _gate()
    if g is None:
        return {"error": "experiments/corpus_integrity.py is not in this "
                         "checkout, so the gate could not be run"}
    report = g.analyse(Path(corpus_dir))
    return {
        "n_takes": report.get("n_takes"),
        "n_failed": report.get("n_failed"),
        "ok": report.get("ok"),
        "text": g.format_report(report),
        "takes": [
            {"condition": t.get("condition"), "ok": t.get("ok"),
             "failures": t.get("failures"), "unverified": t.get("unverified")}
            for t in report.get("takes", [])],
    }


# ============================================================================
# the session
# ============================================================================

@dataclass
class Session:
    """What was read, and where each piece came from."""
    path: Path
    luma: np.ndarray = None                # (n, 96, 160) uint8
    offsets_s: np.ndarray = None           # (n,) float64, relative seconds
    audio: np.ndarray = None               # (m,) float32 at RATE
    rate: int = RATE
    meta: dict = field(default_factory=dict)
    webm: Path | None = None
    sources: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return 0 if self.luma is None else int(self.luma.shape[0])

    @property
    def duration_s(self) -> float:
        if self.offsets_s is None or len(self.offsets_s) < 2:
            return 0.0
        return float(self.offsets_s[-1] - self.offsets_s[0])


def _read_luma_stream(session_dir: Path, meta: dict | None = None):
    """The 160x96 luma stream, found by CONTENT rather than by filename.

    Two stream formats are in play, and the difference matters:

    * the free-record page writes `luma.raw` -- RAW LUMA, uint8, row-major,
      15,360 bytes a frame, frames concatenated. Its metadata SAYS so
      (`luma.width`, `luma.height`, `luma.frame_bytes`, and a format string
      that reads `np.fromfile(path, np.uint8).reshape(-1, 96, 160)`), and
      when the metadata says so the file is read that way.
    * the script capture writes an NV12 file, whose layout is DETECTED from
      the bytes by `src/nv12.py`, because three strides divide these files
      evenly and only the content can say which one is real.

    Neither is assumed: the raw-luma path is entered only on a declared
    geometry, and the NV12 path is entered only when the file does not
    declare one.
    """
    meta = meta if isinstance(meta, dict) else {}

    # 1. a take's npz carries the luma array
    for npz in sorted(session_dir.glob("*.npz")):
        try:
            with np.load(npz) as z:
                if "luma" in z.files:
                    arr = z["luma"]
                    if arr.ndim == 3 and arr.shape[1:] == (FRAME_H, FRAME_W):
                        return arr.astype(np.uint8), f"npz:{npz.name}"
        except Exception:  # noqa: BLE001
            continue

    # 2. the declared raw-luma geometry
    lmeta = meta.get("luma")
    declared_n = meta.get("n_frames")
    force = str(meta.get("luma_layout", "")).lower()
    if force != "nv12":
        w, h = FRAME_W, FRAME_H
        fb = FRAME_W * FRAME_H
        declared_geom = isinstance(lmeta, dict)
        if declared_geom:
            w = int(lmeta.get("width", FRAME_W) or FRAME_W)
            h = int(lmeta.get("height", FRAME_H) or FRAME_H)
            fb = int(lmeta.get("frame_bytes", w * h) or w * h)
        # The metadata's own frame count gives the file's expected size
        # EXACTLY, which is a content check that needs no filename: n frames
        # of raw luma are n x 15,360 bytes and nothing else. A session whose
        # metadata omits the luma block is still readable this way.
        expected = None
        if (isinstance(declared_n, int) and not isinstance(declared_n, bool)
                and declared_n > 0 and fb > 0):
            expected = declared_n * fb
        if declared_geom or expected is not None:
            # a file whose NAME says luma first, then the rest
            cands = [c for c in sorted(session_dir.iterdir())
                     if c.is_file() and "luma" in c.name.lower()]
            cands += [c for c in sorted(session_dir.iterdir())
                      if c.is_file() and "luma" not in c.name.lower()]
            for cand in cands:
                if cand.suffix.lower() in (".json", ".wav", ".webm", ".npz"):
                    continue
                try:
                    size = cand.stat().st_size
                except OSError:
                    continue
                if expected is not None:
                    if size != expected:
                        # it may still be the luma stream, TRUNCATED against
                        # the metadata -- read it, and let the frame-count
                        # check say so, rather than reporting nothing found
                        if size < fb or size % fb:
                            continue
                else:
                    if size < fb * 2 or size % fb:
                        continue
                data = np.frombuffer(cand.read_bytes(), dtype=np.uint8)
                n = len(data) // fb
                arr = data[: n * fb].reshape(n, h, w)
                return arr, f"raw-luma:{cand.name} ({n} x {w}x{h})"

    # 3. a raw frame file -- the layout is DETECTED, never assumed
    for cand in sorted(session_dir.iterdir()):
        if not cand.is_file() or cand.suffix.lower() in (".json", ".wav",
                                                          ".webm", ".npz"):
            continue
        try:
            size = cand.stat().st_size
        except OSError:
            continue
        if size < LUMA_BYTES:
            continue
        try:
            lay = nv12.detect_layout(cand, FRAME_W, FRAME_H)
        except Exception:  # noqa: BLE001
            continue
        if lay.n_frames < 1:
            continue
        try:
            luma, _ = nv12.read_luma(cand, width=FRAME_W, height=FRAME_H)
        except Exception:  # noqa: BLE001
            continue
        return luma.astype(np.uint8), f"nv12:{cand.name} ({lay.name})"
    return None, "none"


def _read_offsets(session_dir: Path, meta: dict, n_frames: int):
    """The per-frame timestamps, in seconds relative to the first frame.

    The free-record page writes `timestamps.json` with `frame_offsets_ms` --
    MILLISECONDS, `performance.now()` minus the first frame. The script
    capture writes `frame_offsets_s` -- SECONDS. Both are accepted, and the
    unit is read from the key, never guessed.
    """
    candidates = []
    for pat in ("*_timing.json", "timing.json", "timestamps.json",
                "meta.json", "*.json"):
        candidates.extend(sorted(session_dir.glob(pat)))
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        try:
            data = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        offsets = None
        scale = 1.0
        if isinstance(data, dict):
            offsets = data.get("frame_offsets_s")
            if offsets is None:
                offsets = data.get("frame_offsets_ms")
                scale = 0.001
            if offsets is None and isinstance(data.get("timing"), dict):
                offsets = data["timing"].get("frame_offsets_s")
                if offsets is None:
                    offsets = data["timing"].get("frame_offsets_ms")
                    scale = 0.001
        elif isinstance(data, list) and data and isinstance(data[0], (int, float)):
            offsets = data
        if offsets is None or not isinstance(offsets, list) or len(offsets) < 2:
            continue
        arr = np.asarray(offsets, dtype=np.float64) * scale
        if n_frames and len(arr) != n_frames:
            continue
        unit = "ms" if scale != 1.0 else "s"
        return arr - arr[0], f"{path.name} (frame_offsets_{unit})"
    # no timestamps: assume the script's rate and say so
    return (np.arange(n_frames, dtype=np.float64) / FPS,
            f"assumed {FPS} fps (no timestamps found)")


def read_session(session_dir: Path) -> Session:
    """Read a free-session folder. Liberal about names, strict about content."""
    session_dir = Path(session_dir)
    s = Session(path=session_dir)

    # the metadata. The free-record page writes `meta.json` with
    # tool "browser_free_record"; the script capture writes a take's meta.
    for path in sorted(session_dir.glob("*.json")):
        if path.name.endswith("_timing.json"):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict) and ("kind" in data or "condition" in data
                                       or "stimulus" in data or "tool" in data
                                       or "wall_clock_start" in data
                                       or "n_frames" in data):
            s.meta = data
            s.sources["meta"] = path.name
            break

    # the luma stream
    s.luma, s.sources["luma"] = _read_luma_stream(session_dir, s.meta)

    # the video
    webms = sorted(session_dir.glob("*.webm"))
    if webms:
        s.webm = webms[0]
        s.sources["webm"] = webms[0].name

    # the timestamps
    s.offsets_s, s.sources["timestamps"] = _read_offsets(
        session_dir, s.meta, s.n_frames)

    # the audio. The free-record page writes the WAV at the AudioContext's
    # own rate (no resampling); `src/wavio.py` resamples it to the analysis
    # rate, which is the rate every take in this repo carries.
    wavs = sorted(session_dir.glob("*.wav"))
    if wavs:
        try:
            audio = wavio.read_wav(str(wavs[0]))
            s.audio = np.asarray(audio.samples, dtype=np.float32)
            s.rate = wavio.ANALYSIS_SR
            s.sources["audio"] = f"{wavs[0].name} ({audio.source_sample_rate} Hz)"
            s.sources["audio_rate"] = int(audio.source_sample_rate or s.rate)
        except Exception as exc:  # noqa: BLE001
            s.sources["audio"] = f"{wavs[0].name} (unreadable: {exc})"
    else:
        s.sources["audio"] = "none"

    # -- the metadata is CHECKED against what was actually read -------------
    # The server checks the five sections on the way in; this checks the
    # same things on the way out, because a session folder that has been
    # copied, truncated or half-written must not be analysed silently.
    declared_n = s.meta.get("n_frames")
    if isinstance(declared_n, int) and not isinstance(declared_n, bool):
        if s.luma is not None and declared_n != s.n_frames:
            s.warnings.append(
                f"meta.n_frames is {declared_n} but the luma holds "
                f"{s.n_frames} frames")
    declared_rate = (s.meta.get("audio") or {}).get("sample_rate") \
        if isinstance(s.meta.get("audio"), dict) else None
    got_rate = s.sources.get("audio_rate")
    if (isinstance(declared_rate, int) and isinstance(got_rate, int)
            and declared_rate != got_rate):
        s.warnings.append(
            f"meta.audio.sample_rate is {declared_rate} but the WAV is "
            f"{got_rate} Hz")
    lmeta = s.meta.get("luma")
    if isinstance(lmeta, dict) and s.luma is not None:
        w = int(lmeta.get("width", FRAME_W) or FRAME_W)
        h = int(lmeta.get("height", FRAME_H) or FRAME_H)
        if (w, h) != (FRAME_W, FRAME_H):
            s.warnings.append(
                f"the luma geometry is {w}x{h}, not the pipeline's "
                f"{FRAME_W}x{FRAME_H}")
    if s.luma is not None and s.offsets_s is not None \
            and len(s.offsets_s) != s.n_frames:
        s.warnings.append(
            f"the timestamps hold {len(s.offsets_s)} entries for "
            f"{s.n_frames} frames")
    if s.warnings:
        s.sources["warnings"] = "; ".join(s.warnings)
    return s


# ============================================================================
# 1. the face state
# ============================================================================

def face_symmetry(luma: np.ndarray, upscale: int = UPSCALE):
    """Per-frame nose-to-eye asymmetry, and whether a face was found.

    Returns (sym, detected). sym is NaN where no face was found.

    The landmark model is imported INSIDE the function, so this module
    imports (and the speech half runs) on an interpreter without mediapipe.
    """
    sym = np.full(len(luma), np.nan)
    detected = np.zeros(len(luma), dtype=bool)
    try:
        import cv2
        import mediapipe as mp
    except Exception as exc:  # noqa: BLE001
        return sym, detected, f"mediapipe unavailable: {exc}"

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.05, min_tracking_confidence=0.05)
    try:
        for i in range(len(luma)):
            img = luma[i]
            if upscale != 1:
                img = cv2.resize(img, (FRAME_W * upscale, FRAME_H * upscale),
                                 interpolation=cv2.INTER_CUBIC)
            rgb = np.repeat(img[:, :, None], 3, axis=2)
            res = mesh.process(rgb)
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            nose = np.array([lm[1].x * FRAME_W, lm[1].y * FRAME_H])
            er = np.array([lm[33].x * FRAME_W, lm[33].y * FRAME_H])
            el = np.array([lm[263].x * FRAME_W, lm[263].y * FRAME_H])
            d_r = float(np.linalg.norm(nose - er))
            d_l = float(np.linalg.norm(nose - el))
            sym[i] = (d_l - d_r) / max(1e-6, d_l + d_r)
            detected[i] = True
    finally:
        mesh.close()
    return sym, detected, ""


def face_states(sym: np.ndarray, detected: np.ndarray) -> list[str]:
    """Per-frame face state: facing / turned / absent."""
    states = []
    for s, d in zip(sym, detected):
        if not d or not np.isfinite(s):
            states.append("absent")
        elif abs(s) >= TURNED_THRESHOLD:
            states.append("turned")
        else:
            states.append("facing")
    return states


# ============================================================================
# 2. the speech state
# ============================================================================

def speech_intervals(audio: np.ndarray, rate: int = RATE):
    """Speech/silence intervals, from the repository's own energy VAD."""
    from src import energy
    frame_len = int(rate * 0.030)
    hop = int(rate * 0.010)
    frames, _ = energy.rms_series(list(audio), frame_len, hop, rate)
    segs = energy.segment(frames, hop_s=hop / float(rate), adaptive=True)
    return [(s.start, s.end, s.speech) for s in segs]


def per_frame_speech(offsets_s: np.ndarray, intervals, rate: int = RATE):
    """The fraction of each video frame's window that is speech."""
    n = len(offsets_s)
    out = np.zeros(n)
    if n == 0:
        return out
    edges = np.empty(n + 1)
    edges[:n] = offsets_s
    edges[n] = offsets_s[-1] + (offsets_s[-1] - offsets_s[-2]
                                 if n > 1 else 1.0 / FPS)
    for i in range(n):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        speech = 0.0
        for s, e, is_speech in intervals:
            if not is_speech:
                continue
            lo, hi = max(a, s), min(b, e)
            if hi > lo:
                speech += hi - lo
        out[i] = speech / (b - a)
    return out


def speech_states(frac: np.ndarray) -> list[str]:
    """Per-frame speech state: speaking / silent.

    A frame is speaking when more than half of its window is speech.
    """
    return ["speaking" if f > 0.5 else "silent" for f in frac]


# ============================================================================
# 3. the segments
# ============================================================================

def _merge_short_runs(labels: list[str], min_s: float,
                      offsets_s: np.ndarray) -> list[str]:
    """A state shorter than min_s is noise: give it to its longer neighbour."""
    labels = list(labels)
    n = len(labels)
    if n == 0:
        return labels
    edges = np.empty(n + 1)
    edges[:n] = offsets_s
    edges[n] = offsets_s[-1] + (offsets_s[-1] - offsets_s[-2]
                                 if n > 1 else 1.0 / FPS)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < n:
            j = i
            while j + 1 < n and labels[j + 1] == labels[i]:
                j += 1
            run_s = edges[j + 1] - edges[i]
            if run_s < min_s and (i > 0 or j + 1 < n):
                # the neighbouring run that is longer takes it
                left = (edges[i] - edges[i - 1]) if i > 0 else -1.0
                right = (edges[j + 2] - edges[j + 1]) if j + 1 < n else -1.0
                take = "left" if left > right else "right"
                if take == "left" and i > 0:
                    for k in range(i, j + 1):
                        labels[k] = labels[i - 1]
                elif take == "right" and j + 1 < n:
                    for k in range(i, j + 1):
                        labels[k] = labels[j + 1]
                else:
                    i = j + 1
                    changed = True
                    continue
                changed = True
            i = j + 1
    return labels


def build_segments(face: list[str], speech: list[str],
                   offsets_s: np.ndarray, sym: np.ndarray,
                   detected: np.ndarray, speech_frac: np.ndarray,
                   min_segment_s: float = MIN_SEGMENT_S,
                   min_label_s: float = MIN_LABEL_S,
                   audio_total_s: float | None = None) -> list[dict]:
    """Contiguous segments with a face state, a speech state and confidences.

    `audio_total_s` is the extent of the session's audio. When it is given,
    the AUDIO BOUNDS THE SEGMENTS: a segment that runs past the end of the
    audio is cut at the audio's end, so the segment's duration, its speech
    fraction and its usability are all judged on the part that has audio.
    Without this the tail of a session whose video outran its audio is
    scored as a long silence -- and a take cut from it would fail the gate's
    A/V span check.
    """
    n = len(face)
    if n == 0:
        return []
    face = _merge_short_runs(face, min_label_s, offsets_s)
    speech = _merge_short_runs(speech, min_label_s, offsets_s)

    edges = np.empty(n + 1)
    edges[:n] = offsets_s
    edges[n] = offsets_s[-1] + (offsets_s[-1] - offsets_s[-2]
                                 if n > 1 else 1.0 / FPS)

    segments = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and face[j + 1] == face[i] \
                and speech[j + 1] == speech[i]:
            j += 1
        if audio_total_s is not None:
            keep = np.nonzero(offsets_s[i:j + 1]
                              <= audio_total_s + 1e-9)[0]
            if len(keep) == 0:
                # the whole segment is past the end of the audio
                i = j + 1
                continue
            j = i + int(keep[-1])
        seg = {
            "frame_start": i,
            "frame_end": j + 1,
            "start_s": float(edges[i]),
            "end_s": float(edges[j + 1]),
            "duration_s": float(edges[j + 1] - edges[i]),
            "face_state": face[i],
            "speech_state": speech[i],
            "n_frames": j + 1 - i,
            "face_detection_rate": float(detected[i:j + 1].mean()),
            "mean_sym": (float(np.nanmean(np.abs(sym[i:j + 1])))
                         if detected[i:j + 1].any() else None),
            "speech_fraction": float(speech_frac[i:j + 1].mean()),
        }
        segments.append(seg)
        i = j + 1

    # The confidence per label
    for seg in segments:
        # THE LABEL COMES FROM THE SEGMENT'S AGGREGATE, NOT FROM ITS FRAMES.
        # The per-frame states decide the BOUNDARIES; the gate's thresholds
        # decide the LABEL, because the gate reads `speech_frac` -- the
        # segment's aggregate -- and compares it with 0.15 and 0.05. A
        # segment whose frames are mostly silent but which carries a few
        # audible ones would be labelled "silent" by the majority and FAIL
        # the gate's 0.05. So the aggregate decides, and a segment between
        # the two thresholds is AMBIGUOUS: it is dropped and reported, never
        # mislabelled.
        f = seg["speech_fraction"]
        if f >= SPEECH_MIN:
            seg["speech_state"] = "speaking"
        elif f <= SPEECH_MAX:
            seg["speech_state"] = "silent"
        else:
            seg["speech_state"] = "ambiguous"
        seg["speech_margin"] = round(min(abs(f - SPEECH_MIN),
                                          abs(f - SPEECH_MAX)), 6)
        seg["face_confidence"] = _face_confidence(seg)
        seg["speech_confidence"] = _speech_confidence(seg)
        seg["confidence"] = min(seg["face_confidence"], seg["speech_confidence"])
        seg["condition"] = CONDITION_FOR.get((seg["face_state"],
                                              seg["speech_state"]))
        seg["usable"] = (seg["duration_s"] >= min_segment_s
                         and seg["condition"] is not None)
        if seg["condition"] is None:
            seg["reason"] = ("no face in this segment"
                             if seg["face_state"] == "absent"
                             else "ambiguous speech state")
        elif not seg["usable"]:
            seg["reason"] = f"shorter than {min_segment_s:g} s"
    return segments


def _face_confidence(seg: dict) -> float:
    if seg["face_state"] == "absent":
        return float(1.0 - seg["face_detection_rate"])
    m = seg["mean_sym"]
    if m is None:
        return 0.0
    if seg["face_state"] == "facing":
        return float(max(0.0, min(1.0, 1.0 - m / TURNED_THRESHOLD)))
    return float(max(0.0, min(1.0, (m - TURNED_THRESHOLD)
                              / max(1e-6, TURNED_FULL - TURNED_THRESHOLD))))


def _speech_confidence(seg: dict) -> float:
    f = seg["speech_fraction"]
    state = seg.get("speech_state", "")
    if state == "ambiguous":
        # There is no label, so there is no confidence in one; the margin to
        # the nearest threshold is reported separately.
        return 0.0
    if state == "speaking":
        return float(max(0.0, min(1.0, (f - SPEECH_MIN) / (1.0 - SPEECH_MIN))))
    return float(max(0.0, min(1.0, (SPEECH_MAX - f) / SPEECH_MAX)))


# ============================================================================
# 4. the slices
# ============================================================================

def _maps_for(luma: np.ndarray):
    """The pipeline's own maps, by the reference implementation's functions."""
    if REF is None:
        raise RuntimeError("experiments/corpus_capture_win.py is not in this "
                           "checkout, so the reference maps cannot be built")
    Yf = luma.astype(np.float64)
    Yr = REF.row_band_remove(Yf)
    maps = REF.change_maps(Yr)
    q, scale = REF.quantise(maps)
    maps_raw = REF.change_maps(Yf)
    q_raw, scale_raw = REF.quantise(maps_raw)
    drift_raw = float(np.abs(np.diff(Yf, axis=0)).mean()) if len(Yf) > 1 else 0.0
    drift_row = float(np.abs(np.diff(Yr, axis=0)).mean()) if len(Yr) > 1 else 0.0
    return q, scale, q_raw, scale_raw, drift_raw, drift_row


def _session_fps(session: Session) -> float:
    """The session's OWN frame rate, asked once.

    The free-record page records `achieved_video_fps`; the script capture
    records `achieved_fps`. Neither is trusted blindly: the rate is checked
    against the timestamps, and the measured one wins.
    """
    declared = session.meta.get("achieved_video_fps",
                                 session.meta.get("achieved_fps"))
    measured = 0.0
    if session.offsets_s is not None and len(session.offsets_s) > 1:
        span = float(session.offsets_s[-1] - session.offsets_s[0])
        if span > 0:
            measured = (len(session.offsets_s) - 1) / span
    if measured > 0:
        return round(measured, 4)
    try:
        return round(float(declared), 4)
    except (TypeError, ValueError):
        return 0.0


def _mic_label(meta: dict) -> str:
    """The microphone's name, however this session records it."""
    mic = meta.get("mic")
    if isinstance(mic, dict):
        return str(mic.get("label", "unknown"))
    if isinstance(mic, str) and mic:
        return mic
    return "unknown"


def write_take(out_dir: Path, seg: dict, session: Session,
               stamp: int, name: str) -> dict:
    """Write one slice in EXACTLY the corpus take format."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    i0, i1 = seg["frame_start"], seg["frame_end"]

    offsets = session.offsets_s[i0:i1]
    t0 = float(offsets[0])
    rate = session.rate

    if session.audio is None or len(session.audio) == 0:
        raise ValueError("the session has no audio, so the take cannot carry "
                         "the speech half of its label")

    # The audio and the video are SEPARATE CLOCKS, and the gate checks that
    # the two spans agree within 2%. So the SHORTER of the two bounds the
    # slice: if the audio ran out before the last frame, the tail frames are
    # dropped; if the video ran out, the audio is trimmed to it. A derived
    # take that disagrees with itself is worse than one frame fewer.
    #
    # `audio_total` is the audio's absolute extent in session-relative
    # seconds, so it is the limit -- NOT the segment's start plus it.
    audio_total = len(session.audio) / float(rate)
    keep = np.nonzero(offsets <= audio_total + 1e-9)[0]
    if len(keep) < 2:
        raise ValueError("the audio ends before the segment has two frames")
    i1 = i0 + int(keep[-1]) + 1
    offsets = session.offsets_s[i0:i1]
    rel_offsets = offsets - t0
    video_span = float(rel_offsets[-1])

    luma = session.luma[i0:i1]
    n = luma.shape[0]
    if n < 2:
        raise ValueError("a take needs at least two frames")

    # the audio slice, by the wall clock the frames cover
    a0 = int(round(t0 * rate))
    a1 = int(round((t0 + video_span) * rate))
    audio = session.audio[a0:a1]
    audio_span = len(audio) / float(rate)
    av_error = (abs(audio_span - video_span) / video_span
                if video_span > 0 else 1.0)

    achieved = (n - 1) / video_span if video_span > 0 else float(FPS)

    q, scale, q_raw, scale_raw, drift_raw, drift_row = _maps_for(luma)

    stem = f"{name}_{stamp}"
    np.savez_compressed(out_dir / f"{stem}.npz", maps=q,
                        scale=np.float32(scale), maps_raw=q_raw,
                        scale_raw=np.float32(scale_raw), luma=luma)
    wavio.write_wav(out_dir / f"{stem}.wav", list(audio), rate)

    timing = {
        "condition": name,
        "n_frames": n,
        "frame_offsets_s": [round(float(v), 6) for v in rel_offsets],
        # The slice's frame index IS this time base, so the assumed rate is
        # the measured one. The meaningful rate question is asked once, at the
        # session level, and recorded in the metadata.
        "assumed_fps": round(float(achieved), 4),
        "achieved_fps": round(float(achieved), 4),
        "video_span_s": round(video_span, 3),
        "audio_span_s": round(audio_span, 3),
        "audio_sample_rate": int(rate),
        "requested_sample_rate": int(rate),
        "mic": str(_mic_label(session.meta)),
        "capture_wallclock_s": 0.0,
    }
    (out_dir / f"{stem}_timing.json").write_text(json.dumps(timing))

    speak = seg["speech_state"] == "speaking"
    meta = {
        "condition": name,
        "kind": "condition",
        "note": (f"derived from a free session: {seg['face_state']}, "
                 f"{seg['speech_state']}"),
        "pose": POSE_FOR.get(seg["face_state"], "any"),
        "speak": bool(speak),
        "stimulus": session.meta.get("stimulus", "free-session"),
        "stimulus_title": session.meta.get("stimulus_title", "free session"),
        "stimulus_estimated_s": float(session.meta.get(
            "stimulus_estimated_s", 0.0)),
        "speech_frac": round(float(seg["speech_fraction"]), 6),
        "n_frames": n,
        "seconds": round(video_span, 3),
        "achieved_fps": round(float(achieved), 4),
        "audio_sample_rate": int(rate),
        "audio_seconds": round(audio_span, 3),
        "map_shape": [MAP_H, MAP_W],
        "map_scale": float(scale),
        "map_scale_raw": float(scale_raw),
        "drift_raw": drift_raw,
        "drift_after_row_removal": drift_row,
        "av_span_error": round(float(av_error), 6),
        "timestamp": int(stamp),
        "rep": int(name[1:]) if name[1:].isdigit() else 1,
        "filter": "row",
        "filters": {"maps": "row", "maps_raw": "none", "luma": "none"},
        "arrays": ["maps", "maps_raw", "luma"],
        # -- the provenance of a DERIVED take --------------------------
        "provenance": PROVENANCE,
        "derived": {
            "session": str(session.path),
            "segment_start_s": round(seg["start_s"], 3),
            "segment_end_s": round(seg["end_s"], 3),
            "face_state": seg["face_state"],
            "speech_state": seg["speech_state"],
            "confidence": round(float(seg["confidence"]), 4),
            "face_confidence": round(float(seg["face_confidence"]), 4),
            "speech_confidence": round(float(seg["speech_confidence"]), 4),
            "mean_sym": (round(seg["mean_sym"], 5)
                         if seg["mean_sym"] is not None else None),
            "face_detection_rate": round(seg["face_detection_rate"], 4),
            "session_achieved_fps": _session_fps(session),
            "session_source_audio_rate": int(
                session.sources.get("audio_rate", session.rate) or session.rate),
            "tool": "experiments/free_session_analyze.py",
        },
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return {"stem": stem, "condition": name, "files": [f"{stem}.npz",
                                                       f"{stem}.wav",
                                                       f"{stem}.json",
                                                       f"{stem}_timing.json"]}


# ============================================================================
# the driver
# ============================================================================

def analyse_session(session: Session, out_dir: Path | None = None,
                    min_segment_s: float = MIN_SEGMENT_S,
                    min_label_s: float = MIN_LABEL_S,
                    upscale: int = UPSCALE,
                    stamp: int | None = None,
                    include_frames: bool = False) -> dict:
    """Analyse one session and, if out_dir is given, write the slices."""
    report = {
        "session": str(session.path),
        "sources": session.sources,
        "n_frames": session.n_frames,
        "duration_s": session.duration_s,
        "segments": [],
        "takes": [],
        "skipped": [],
    }
    if session.warnings:
        report["warnings"] = list(session.warnings)
    if session.luma is None or session.n_frames < 2:
        report["error"] = "no luma stream was found in the session folder"
        return report
    if session.offsets_s is None or len(session.offsets_s) != session.n_frames:
        report["error"] = "the timestamps do not match the frame count"
        return report

    sym, detected, face_note = face_symmetry(session.luma, upscale)
    report["face_note"] = face_note or "mediapipe FaceMesh"
    face = face_states(sym, detected)

    if session.audio is not None and len(session.audio) > 0:
        intervals = speech_intervals(session.audio, session.rate)
        sfrac = per_frame_speech(session.offsets_s, intervals, session.rate)
    else:
        intervals = []
        sfrac = np.zeros(session.n_frames)
        report["speech_note"] = "no audio in the session folder"
    speech = speech_states(sfrac)

    # the head-direction statistic the threshold was set from, so the report
    # carries the evidence for its own decision rule
    finite = sym[np.isfinite(sym)]
    report["symmetry"] = {
        "n": int(finite.size),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "threshold": TURNED_THRESHOLD,
        "n_at_or_above_threshold": int(np.count_nonzero(
            np.abs(finite) >= TURNED_THRESHOLD)),
    }

    segments = build_segments(face, speech, session.offsets_s, sym, detected,
                              sfrac, min_segment_s=min_segment_s,
                              min_label_s=min_label_s,
                              audio_total_s=(len(session.audio)
                                             / float(session.rate)
                                             if session.audio is not None
                                             and len(session.audio) else None))
    report["segments"] = segments
    report["speech_fraction_overall"] = float(sfrac.mean()) if len(sfrac) else 0.0
    report["face_detection_rate"] = float(detected.mean()) if len(detected) else 0.0
    if include_frames:
        report["frames"] = {"face": face, "speech": speech}

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(stamp if stamp is not None else time.time())
        counters = {}
        for seg in segments:
            if not seg["usable"]:
                report["skipped"].append({
                    "start_s": seg["start_s"], "end_s": seg["end_s"],
                    "face_state": seg["face_state"],
                    "speech_state": seg["speech_state"],
                    "reason": seg.get("reason", "not usable")})
                continue
            cond = seg["condition"]
            counters[cond] = counters.get(cond, 0) + 1
            name = f"{cond}{counters[cond]}"
            try:
                written = write_take(out_dir, seg, session, stamp, name)
            except Exception as exc:  # noqa: BLE001
                report["skipped"].append({
                    "start_s": seg["start_s"], "end_s": seg["end_s"],
                    "reason": f"could not be written: {exc}"})
                continue
            written["condition"] = cond
            written["start_s"] = seg["start_s"]
            written["end_s"] = seg["end_s"]
            written["confidence"] = seg["confidence"]
            report["takes"].append(written)
        report["out_dir"] = str(out_dir)
        # The gate individuates EVERY root-level *.json as a take
        # (`find_takes` is a non-recursive glob that excludes only
        # `*_timing.json`), so the segment table goes in a SUBDIRECTORY --
        # the same reasoning that put ZCode's rich copies in `out/rich/`.
        rich = out_dir / "rich"
        rich.mkdir(parents=True, exist_ok=True)
        (rich / "free_session_segments.json").write_text(
            json.dumps(report, indent=2))
        report["segments_json"] = str(rich / "free_session_segments.json")
    return report


def format_segments(report: dict) -> str:
    lines = []
    lines.append("SEGMENT TABLE")
    lines.append("-" * 78)
    lines.append(f"{'start':>7} {'end':>7} {'dur':>6} {'face':>8} {'speech':>8} "
                 f"{'conf':>6} {'cond':>5}  speechfrac  sym")
    for s in report.get("segments", []):
        sym = s.get("mean_sym")
        sym_s = f"{sym:+.3f}" if sym is not None else "  n/a"
        lines.append(f"{s['start_s']:7.2f} {s['end_s']:7.2f} "
                     f"{s['duration_s']:6.2f} {s['face_state']:>8} "
                     f"{s['speech_state']:>8} {s['confidence']:6.3f} "
                     f"{str(s.get('condition') or '-'):>5}  "
                     f"{s['speech_fraction']:10.3f}  {sym_s}")
    lines.append("")
    lines.append(f"takes written: {len(report.get('takes', []))}   "
                 f"segments skipped: {len(report.get('skipped', []))}")
    for t in report.get("takes", []):
        lines.append(f"  {t['stem']:<24} {t.get('condition')}  "
                     f"{t['start_s']:.2f}-{t['end_s']:.2f}s  "
                     f"conf {t.get('confidence', 0):.3f}")
    for s in report.get("skipped", []):
        lines.append(f"  skipped {s['start_s']:.2f}-{s['end_s']:.2f}s  "
                     f"{s.get('reason')}")
    return "\n".join(lines)


# ============================================================================
# the recovery report -- does the machinery recover the labels the captures
# are KNOWN to carry?
# ============================================================================

RECOVERY_CAPTURES = [
    ("A", "facing", "speaking"),
    ("B", "turned", "speaking"),
    ("C", "facing", "silent"),
    ("empty1", "absent", "silent"),
]

# Where the labelled takes live, and which file in each carries the label.
# The ground truth is read from the take's OWN metadata, so it is the
# artifact's label and not this script's opinion of it.
TAKE_DIRS = ("corpus", "corpus_v2")


def _find_takes(capture_dir: Path, name: str) -> list[tuple[dict, Path | None, str]]:
    """Every labelled take for a capture: (metadata, npz, where it came from).

    There is more than one, and they are NOT the same artifact: the `corpus/`
    seed takes were built from the TORN 23,040-byte read, and the `corpus_v2/`
    takes were built from the corrected 19,200-byte read of the same
    capture. The maps of the two differ, and only one of them is what the
    pipeline produces today -- which is exactly what the check below finds.
    """
    found = []
    for sub in TAKE_DIRS:
        d = Path(capture_dir) / sub
        for meta_name in (f"{name}_seed.json", f"{name}.json"):
            p = d / meta_name
            if not p.is_file():
                continue
            try:
                meta = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            stem = meta_name[:-len(".json")]
            npz = d / f"{stem}.npz"
            found.append((meta, (npz if npz.is_file() else None),
                          f"{sub}/{meta_name}"))
    return found


def verify_take_maps(luma: np.ndarray, take_npz: Path) -> dict:
    """Do the maps the analyzer derives REPRODUCE the stored take's maps?

    This is the check that makes a derived take a take: the seeded corpus was
    written by `experiments/seed_corpus_from_captures.py`, and the analyzer
    builds its maps with the reference implementation's own `row_band_remove`,
    `change_maps` and `quantise`. If the two agree cell for cell, a slice the
    analyzer writes carries exactly the maps the pipeline writes.
    """
    with np.load(take_npz) as z:
        want = z["maps"]
        want_scale = float(z["scale"])
    n = int(want.shape[0]) + 1
    if n > luma.shape[0]:
        return {"error": "the take has more frames than the capture"}
    q, scale, _, _, _, _ = _maps_for(luma[:n])
    if tuple(q.shape) != tuple(want.shape):
        return {"error": f"shape {q.shape} against the stored {want.shape}"}
    diff = int(np.count_nonzero(q != want))
    return {
        "n_maps": int(want.shape[0]),
        "differing_cells": diff,
        "total_cells": int(want.size),
        "max_abs_diff": int(np.abs(q.astype(np.int16)
                                   - want.astype(np.int16)).max()),
        "scale_derived": float(scale),
        "scale_stored": want_scale,
        "scale_matches": bool(abs(float(scale) - want_scale)
                              <= 1e-4 * max(1.0, abs(want_scale))),
    }


def recovery_report(capture_dir: Path, slices_dir: Path | None = None) -> dict:
    """Run the machinery over the captures that exist and check the labels.

    When `slices_dir` is given, the derived TAKES are written as well, and
    the integrity gate is run over them -- which is the point of the whole
    exercise: a derived take is only worth anything if the gate eats it.
    """
    import tempfile

    capture_dir = Path(capture_dir)
    results = []
    work = Path(tempfile.mkdtemp(prefix="fsa_recovery_"))
    for name, fallback_face, fallback_speech in RECOVERY_CAPTURES:
        nv12_path = capture_dir / f"{name}.nv12"
        wav_path = capture_dir / f"{name}.wav"
        if not nv12_path.is_file() or not wav_path.is_file():
            results.append({"capture": name, "error": "missing"})
            continue

        # -- the ground truth, read from the take artifact itself ------------
        takes = _find_takes(capture_dir, name)
        if takes:
            take_meta = takes[0][0]
            truth_face = str(take_meta.get("pose", "")).lower()
            truth_speech = ("speaking" if take_meta.get("speak") is True
                            else "silent" if take_meta.get("speak") is False
                            else "")
            truth_src = ", ".join(t[2] for t in takes)
        else:
            truth_face, truth_speech = fallback_face, fallback_speech
            truth_src = "fallback (no labelled take exists)"

        # -- the session folder, assembled from the capture -----------------
        tmp = work / name
        tmp.mkdir(parents=True, exist_ok=True)
        raw = np.frombuffer(nv12_path.read_bytes(), np.uint8)
        lay = nv12.detect_layout(nv12_path, FRAME_W, FRAME_H)
        luma, _ = nv12.read_luma(nv12_path, width=FRAME_W, height=FRAME_H)
        (tmp / "luma.nv12").write_bytes(raw.tobytes())
        (tmp / "audio.wav").write_bytes(wav_path.read_bytes())
        offsets = np.arange(luma.shape[0], dtype=np.float64) / FPS
        (tmp / "timing.json").write_text(json.dumps({
            "frame_offsets_s": [round(float(v), 6) for v in offsets],
            "assumed_fps": FPS, "achieved_fps": FPS,
            "note": ("the capture script recorded no per-frame timestamps, so "
                     "the nominal 30 fps grid is used; the WALL-CLOCK rate of "
                     "this capture was lower (the driver repeated frames)")}))
        (tmp / "meta.json").write_text(json.dumps({
            "condition": name, "kind": "condition", "stimulus": "invictus",
            "achieved_fps": FPS, "mic": "recovery"}))

        session = read_session(tmp)
        rep = analyse_session(session, out_dir=None, min_segment_s=1.0,
                              include_frames=True)
        segs = rep.get("segments", [])        # the label the machinery gives the WHOLE capture, by the frames'
        # majority face state and the measured speech fraction
        face_counts = {}
        for s in segs:
            face_counts[s["face_state"]] = face_counts.get(s["face_state"], 0) \
                + s["n_frames"]
        got_face = max(face_counts, key=face_counts.get) if face_counts else "?"
        total_speech = (sum(s["speech_fraction"] * s["n_frames"] for s in segs)
                        / max(1, sum(s["n_frames"] for s in segs)))
        got_speech = ("speaking" if total_speech >= SPEECH_MIN
                      else "silent" if total_speech <= SPEECH_MAX
                      else "ambiguous")
        ok = (got_face == truth_face and got_speech == truth_speech)

        # -- the per-frame face errors, against the take's label ------------
        states = rep.get("frames", {}).get("face", [])
        if truth_face == "absent":
            face_errors = sum(1 for s in states if s != "absent")
        else:
            face_errors = sum(1 for s in states if s != truth_face)

        entry = {
            "capture": name,
            "known": {"face": truth_face, "speech": truth_speech},
            "known_from": truth_src,
            "recovered": {"face": got_face, "speech": got_speech},
            "speech_fraction": round(float(total_speech), 4),
            "face_detection_rate": round(float(rep.get("face_detection_rate",
                                                       0.0)), 4),
            "n_segments": len(segs),
            "n_frames": int(len(states)),
            "face_errors": int(face_errors),
            "symmetry": rep.get("symmetry"),
            "recovered_correctly": bool(ok),
            "layout": f"{lay.name} ({lay.stride} B/frame)",
        }
        # -- and the maps, against EVERY take that was seeded from this
        #    capture. The seed takes were built on the torn stride and the
        #    v2 takes on the corrected one; saying so is the point.
        entry["maps_vs_takes"] = []
        for _, take_npz, take_src in takes:
            if take_npz is None:
                continue
            try:
                chk = verify_take_maps(luma, take_npz)
            except Exception as exc:  # noqa: BLE001
                chk = {"error": str(exc)}
            chk["take"] = take_src
            entry["maps_vs_takes"].append(chk)
        results.append(entry)

        # -- and the derived TAKES, with the gate run over them -------------
        if slices_dir is not None:
            out = Path(slices_dir) / name
            written = analyse_session(session, out_dir=out,
                                      min_segment_s=1.0)
            entry["derived_takes"] = [t["stem"] for t in written.get("takes", [])]
            entry["derived_skipped"] = len(written.get("skipped", []))
            gate = run_gate(out)
            entry["gate"] = {
                "n_takes": gate.get("n_takes"),
                "n_failed": gate.get("n_failed"),
                "ok": gate.get("ok"),
                "error": gate.get("error"),
                "failures": [
                    f"{t['condition']}: {f}"
                    for t in gate.get("takes", []) if t.get("ok") is False
                    for f in (t.get("failures") or [])],
            }

    n_ok = sum(1 for r in results if r.get("recovered_correctly"))
    n_maps = sum(1 for r in results
                 for m in r.get("maps_vs_takes", [])
                 if m.get("differing_cells") == 0 and "error" not in m)
    n_maps_total = sum(len(r.get("maps_vs_takes", [])) for r in results)
    out = {"results": results, "recovered": n_ok, "total": len(results),
           "maps_reproduced": n_maps, "maps_total": n_maps_total}
    if slices_dir is not None:
        gated = [r for r in results if "gate" in r]
        out["gate"] = {
            "n_takes": sum(r["gate"].get("n_takes") or 0 for r in gated),
            "n_failed": sum(r["gate"].get("n_failed") or 0 for r in gated),
            "ok": all(r["gate"].get("ok") for r in gated) if gated else None,
            "per_capture": {r["capture"]: r["gate"] for r in gated},
        }
    return out


def format_recovery(rep: dict) -> str:
    lines = []
    lines.append("RECOVERY REPORT -- does the machinery recover the known "
                 "labels?")
    lines.append("-" * 92)
    lines.append(f"{'capture':<9} {'known':<20} {'recovered':<20} "
                 f"{'spfrac':>7} {'det':>5} {'err':>4}  {'sym range':<17} "
                 f"{'maps':>10}")
    for r in rep["results"]:
        if "error" in r:
            lines.append(f"{r['capture']:<9} ERROR: {r['error']}")
            continue
        k = f"{r['known']['face']}/{r['known']['speech']}"
        g = f"{r['recovered']['face']}/{r['recovered']['speech']}"
        sym = r.get("symmetry") or {}
        if sym.get("min") is not None:
            sym_s = f"{sym['min']:+.3f}..{sym['max']:+.3f}"
        else:
            sym_s = "n/a"
        m = r.get("maps_vs_takes") or []
        if not m:
            maps_s = "-"
        else:
            parts = []
            for chk in m:
                if "error" in chk:
                    parts.append("ERR")
                elif chk.get("differing_cells") == 0:
                    parts.append("exact")
                else:
                    parts.append(f"{chk['differing_cells']} differ")
            maps_s = ", ".join(parts)
        lines.append(f"{r['capture']:<9} {k:<20} {g:<20} "
                     f"{r['speech_fraction']:7.4f} "
                     f"{r['face_detection_rate']:5.3f} "
                     f"{r.get('face_errors', 0):4d}  {sym_s:<17} "
                     f"{maps_s:>10}")
    lines.append("")
    lines.append(f"recovered {rep['recovered']} of {rep['total']}")
    if "maps_reproduced" in rep:
        lines.append(f"maps reproduced exactly against the stored takes: "
                     f"{rep['maps_reproduced']} of {rep.get('maps_total', 0)}")
        lines.append("  (the corpus/ seed takes were built on the TORN "
                     "23,040-byte stride; the corpus_v2/ takes were built on "
                     "the corrected 19,200-byte read)")
    lines.append("")
    lines.append("the ground truth comes from each take's own metadata:")
    for r in rep["results"]:
        if "error" in r:
            continue
        lines.append(f"  {r['capture']:<9} {r.get('known_from', '?')}")
    return "\n".join(lines)


# ============================================================================
# the self-test -- the slicer on synthetic arrays
# ============================================================================

def selftest() -> int:
    """Unit-test the slicer, the segmenter and the take writer, on arrays."""
    import tempfile

    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  [{'ok' if got == want else 'FAIL'}] {name}: {got!r}")

    print("SELFTEST -- the slicer on synthetic arrays")
    print("-" * 60)

    # 1. the segmenter on a synthetic label sequence.
    #    The four states in the order the conditions name them:
    #    facing/speaking (A), turned/speaking (B), turned/silent (D),
    #    facing/silent (C) -- 180 frames, two seconds each.
    n = 180
    offsets = np.arange(n, dtype=np.float64) / FPS
    face = ["facing"] * 60 + ["turned"] * 60 + ["facing"] * 60
    speech = ["speaking"] * 90 + ["silent"] * 90
    sym = np.where(np.array(face) == "turned", 0.35, 0.02)
    det = np.ones(n, dtype=bool)
    sfrac = np.where(np.array(speech) == "speaking", 0.7, 0.01)
    segs = build_segments(face, speech, offsets, sym, det, sfrac,
                          min_segment_s=1.0, min_label_s=0.5)
    check("four segments", len(segs), 4)
    check("first is facing/speaking",
          (segs[0]["face_state"], segs[0]["speech_state"]), ("facing", "speaking"))
    check("second is turned/speaking",
          (segs[1]["face_state"], segs[1]["speech_state"]), ("turned", "speaking"))
    check("third is turned/silent",
          (segs[2]["face_state"], segs[2]["speech_state"]), ("turned", "silent"))
    check("fourth is facing/silent",
          (segs[3]["face_state"], segs[3]["speech_state"]), ("facing", "silent"))
    check("conditions", [s["condition"] for s in segs], ["A", "B", "D", "C"])

    # 2. the boundaries
    check("first segment starts at 0", segs[0]["frame_start"], 0)
    check("first segment ends at 60", segs[0]["frame_end"], 60)
    check("second segment ends at 90", segs[1]["frame_end"], 90)
    check("third segment ends at 120", segs[2]["frame_end"], 120)

    # 3. a short run is merged into its neighbour
    m = 120
    offsets2 = np.arange(m, dtype=np.float64) / FPS
    face2 = ["facing"] * 50 + ["turned"] * 3 + ["facing"] * 67
    speech2 = ["speaking"] * m
    sym2 = np.where(np.array(face2) == "turned", 0.35, 0.02)
    segs2 = build_segments(face2, speech2, offsets2, sym2,
                            np.ones(m, dtype=bool), np.full(m, 0.7),
                            min_segment_s=1.0, min_label_s=0.5)
    check("a 3-frame run is merged away", len(segs2), 1)
    check("and the merged segment is facing", segs2[0]["face_state"], "facing")

    # 4. a segment too short to be a take is not usable
    segs3 = build_segments(face, speech, offsets, sym, det, sfrac,
                            min_segment_s=100.0, min_label_s=0.5)
    check("nothing is usable at a 100 s minimum",
          sum(1 for s in segs3 if s["usable"]), 0)

    # 5. an absent face has no condition
    face4 = ["absent"] * m
    sym4 = np.full(m, np.nan)
    det4 = np.zeros(m, dtype=bool)
    segs4 = build_segments(face4, ["speaking"] * m, offsets2, sym4, det4,
                            np.full(m, 0.7), min_segment_s=1.0)
    check("absent has no condition", segs4[0]["condition"], None)
    check("absent is not usable", segs4[0]["usable"], False)

    # 5b. THE AGGREGATE DECIDES THE LABEL, NOT THE FRAMES. Here 100 frames
    #     are silent and 10 carry a loud blip: the frames are 91% silent,
    #     so the MAJORITY says "silent", but the aggregate is 0.082 --
    #     between the gate's 0.05 and 0.15, so labelling it "silent" would
    #     FAIL the gate. It must be dropped as ambiguous.
    n6 = 110
    offsets6 = np.arange(n6, dtype=np.float64) / FPS
    sfrac6 = np.zeros(n6)
    sfrac6[100:] = 0.9
    sym6 = np.full(n6, 0.02)
    det6 = np.ones(n6, dtype=bool)
    segs6 = build_segments(["facing"] * n6, ["silent"] * n6, offsets6, sym6,
                            det6, sfrac6, min_segment_s=1.0, min_label_s=0.5)
    check("the blip is merged into its neighbour", len(segs6), 1)
    check("its aggregate sits in the ambiguous band",
          round(segs6[0]["speech_fraction"], 3), 0.082)
    check("so it has no condition", segs6[0]["condition"], None)
    check("and it is not usable", segs6[0]["usable"], False)
    check("and it says why", segs6[0]["reason"], "ambiguous speech state")
    check("and the margin to the nearest threshold is reported",
          round(segs6[0]["speech_margin"], 3), 0.032)

    # 6. the take writer, on a synthetic session
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "takes"
        luma = np.full((40, FRAME_H, FRAME_W), 100, dtype=np.uint8)
        luma[:, 40:50, 60:90] = 200          # something that moves
        luma[20:, 40:50, 60:90] = 210
        audio = np.zeros(int(RATE * 2), dtype=np.float32)
        sess = Session(path=Path(td), luma=luma,
                       offsets_s=np.arange(40, dtype=np.float64) / FPS,
                       audio=audio, rate=RATE,
                       meta={"stimulus": "synthetic", "stimulus_title": "s"})
        seg = {"frame_start": 0, "frame_end": 40, "start_s": 0.0,
               "end_s": 40 / FPS, "duration_s": 40 / FPS,
               "face_state": "facing", "speech_state": "speaking",
               "n_frames": 40, "face_detection_rate": 1.0, "mean_sym": 0.02,
               "speech_fraction": 0.7, "face_confidence": 0.9,
               "speech_confidence": 1.0, "confidence": 0.9, "condition": "A",
               "usable": True}
        written = write_take(out, seg, sess, stamp=1758000000, name="A1")
        stem = written["stem"]
        with np.load(out / f"{stem}.npz") as z:
            check("the npz holds five arrays", sorted(z.files),
                  ["luma", "maps", "maps_raw", "scale", "scale_raw"])
            check("luma is the slice", z["luma"].shape, (40, FRAME_H, FRAME_W))
            check("maps are n-1 of them", z["maps"].shape, (39, MAP_H, MAP_W))
            check("maps_raw is there too", z["maps_raw"].shape,
                  (39, MAP_H, MAP_W))
        meta = json.loads((out / f"{stem}.json").read_text())
        timing = json.loads((out / f"{stem}_timing.json").read_text())
        check("the meta carries the provenance", meta["provenance"], PROVENANCE)
        check("the meta carries the confidence",
              meta["derived"]["confidence"], 0.9)
        check("the condition is a label the gate reads", meta["condition"], "A1")
        check("speak is a bool for a condition", meta["speak"], True)
        check("the timestamps are relative to the slice",
              timing["frame_offsets_s"][0], 0.0)
        check("the spans agree within the gate's 2%",
              abs(timing["audio_span_s"] - timing["video_span_s"])
              / timing["video_span_s"] < 0.02, True)
        check("the rates agree within the gate's 5%",
              abs(timing["achieved_fps"] - timing["assumed_fps"])
              / timing["assumed_fps"] < 0.05, True)
        a = wavio.read_wav(str(out / f"{stem}.wav"))
        check("the wav is mono 16-bit at 16 kHz",
              (a.source_sample_rate, len(a.samples)),
              (RATE, int(round(((40 - 1) / FPS) * RATE))))

    # 7. the REAL free-session layout, as the free-record page writes it:
    #    raw luma (not NV12), MILLISECOND timestamps, a full-rate WAV, and
    #    the page's own metadata. This is the production path.
    with tempfile.TemporaryDirectory() as td:
        sdir = Path(td) / "free_sessions" / "1758000000"
        sdir.mkdir(parents=True)
        n2 = 90
        luma2 = np.full((n2, FRAME_H, FRAME_W), 90, dtype=np.uint8)
        luma2[:, 30:60, 40:100] = 180
        luma2[45:, 30:60, 40:100] = 190
        (sdir / "luma.raw").write_bytes(luma2.tobytes())
        offs_ms = [round(i * 1000.0 / 30.0, 1) for i in range(n2)]
        (sdir / "timestamps.json").write_text(json.dumps({
            "wall_clock_start": "2026-10-20T00:00:00Z",
            "frame_offsets_ms": offs_ms}))
        (sdir / "meta.json").write_text(json.dumps({
            "tool": "browser_free_record", "n_frames": n2,
            "luma": {"width": FRAME_W, "height": FRAME_H,
                     "frame_bytes": FRAME_W * FRAME_H,
                     "format": "uint8 BT.601 luma, row-major"},
            "achieved_video_fps": 30.0,
            "audio": {"sample_rate": 48000, "samples": int(n2 * 1600)},
            "mic": {"label": "test mic"}}))
        rate48 = 48000
        audio48 = np.zeros(int(n2 * 1000.0 / 30.0 * rate48), dtype=np.float32)
        wavio.write_wav(sdir / "audio.wav", list(audio48), rate48)

        sess2 = read_session(sdir)
        check("the free session's luma is read",
              sess2.luma.shape, (n2, FRAME_H, FRAME_W))
        check("and the source says it is raw luma",
              "raw-luma" in sess2.sources["luma"], True)
        check("the millisecond timestamps become seconds",
              round(float(sess2.offsets_s[-1]), 3), 2.967)
        check("the audio is resampled to the analysis rate", sess2.rate, RATE)
        check("the source rate is recorded",
              sess2.sources.get("audio_rate"), 48000)
        check("the mic label is read", _mic_label(sess2.meta), "test mic")
        check("a consistent session raises no warnings",
              sess2.warnings, [])

        # and a slice of it, written in the take format
        out2 = Path(td) / "takes"
        seg2 = {"frame_start": 0, "frame_end": n2, "start_s": 0.0,
                "end_s": float(sess2.offsets_s[-1]),
                "duration_s": float(sess2.offsets_s[-1]),
                "face_state": "facing", "speech_state": "speaking",
                "n_frames": n2, "face_detection_rate": 1.0, "mean_sym": 0.02,
                "speech_fraction": 0.7, "face_confidence": 0.9,
                "speech_confidence": 1.0, "confidence": 0.9, "condition": "A",
                "usable": True}
        w2 = write_take(out2, seg2, sess2, stamp=1758000000, name="A1")
        t2 = json.loads((out2 / f"{w2['stem']}_timing.json").read_text())
        m2 = json.loads((out2 / f"{w2['stem']}.json").read_text())
        check("the slice's spans agree within the gate's 2%",
              abs(t2["audio_span_s"] - t2["video_span_s"])
              / t2["video_span_s"] < 0.02, True)
        check("the slice carries the whole session's frames",
              t2["n_frames"], n2)
        check("the sample rate is the analysis rate on both sides",
              (t2["audio_sample_rate"], t2["requested_sample_rate"]),
              (RATE, RATE))
        check("the mic label reaches the timing",
              t2["mic"], "test mic")
        check("the A/V span error is recorded",
              m2["av_span_error"] < 0.02, True)
        check("the session's own rate is measured, not assumed",
              abs(m2["derived"]["session_achieved_fps"] - 30.0) < 0.01, True)
        check("the source audio rate is recorded",
              m2["derived"]["session_source_audio_rate"], 48000)

    # 8. the audio and the video are SEPARATE CLOCKS: when the audio runs
    #    out first, the slice must be TRIMMED to it, or the gate's 2% A/V
    #    span check fails. (This is a bug the gate caught, not a theory.)
    with tempfile.TemporaryDirectory() as td:
        sess3 = Session(path=Path(td),
                        luma=np.full((100, FRAME_H, FRAME_W), 80, np.uint8),
                        offsets_s=np.arange(100, dtype=np.float64) / 10.0,
                        audio=np.zeros(int(RATE * 3), dtype=np.float32),
                        rate=RATE, meta={"stimulus": "s"})
        seg3 = {"frame_start": 20, "frame_end": 100, "start_s": 2.0,
                "end_s": 10.0, "duration_s": 8.0, "face_state": "facing",
                "speech_state": "speaking", "n_frames": 80,
                "face_detection_rate": 1.0, "mean_sym": 0.02,
                "speech_fraction": 0.7, "face_confidence": 0.9,
                "speech_confidence": 1.0, "confidence": 0.9, "condition": "A",
                "usable": True}
        out3 = Path(td) / "takes"
        w3 = write_take(out3, seg3, sess3, stamp=1758000000, name="A1")
        t3 = json.loads((out3 / f"{w3['stem']}_timing.json").read_text())
        check("the slice is trimmed to the audio",
              t3["video_span_s"] <= 3.0 + 0.01, True)
        check("and the two spans then agree",
              abs(t3["audio_span_s"] - t3["video_span_s"])
              / t3["video_span_s"] < 0.02, True)
        check("and the frames match the trimmed span",
              t3["n_frames"], int(round(t3["video_span_s"] * 10.0)) + 1)

    # 9. the metadata is CHECKED against what was read: a session whose
    #    metadata disagrees with its files must say so, not analyse quietly
    with tempfile.TemporaryDirectory() as td:
        sdir = Path(td) / "bad"
        sdir.mkdir()
        n2 = 40
        (sdir / "luma.raw").write_bytes(
            np.full((n2, FRAME_H, FRAME_W), 70, dtype=np.uint8).tobytes())
        (sdir / "timestamps.json").write_text(json.dumps({
            "frame_offsets_ms": [round(i * 1000.0 / 30.0, 1)
                                 for i in range(n2)]}))
        (sdir / "meta.json").write_text(json.dumps({
            "tool": "browser_free_record", "n_frames": n2 + 5,
            "luma": {"width": FRAME_W, "height": FRAME_H,
                     "frame_bytes": FRAME_W * FRAME_H},
            "audio": {"sample_rate": 48000, "samples": 2000}}))
        wavio.write_wav(sdir / "audio.wav", [0.0] * 2000, 44100)
        sess4 = read_session(sdir)
        check("a wrong frame count is reported",
              any("n_frames is 45" in w for w in sess4.warnings), True)
        check("and a wrong sample rate is reported too",
              any("48000" in w and "44100" in w for w in sess4.warnings), True)

    # 10. the HARVESTER: a batch pass over a corpus's free sessions
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td) / "corpus"
        sessions = corpus / "free_sessions"
        sessions.mkdir(parents=True)
        report_path = Path(td) / "HARVEST-REPORT.md"

        def make_session(name: str, n: int, frames: int = 60):
            d = sessions / name
            d.mkdir(parents=True)
            luma = np.full((frames, FRAME_H, FRAME_W), 80, dtype=np.uint8)
            luma[:, 30:60, 40:100] = 180
            luma[frames // 2:, 30:60, 40:100] = 190
            (d / "luma.raw").write_bytes(luma.tobytes())
            (d / "timestamps.json").write_text(json.dumps({
                "frame_offsets_ms": [round(i * 1000.0 / 30.0, 1)
                                     for i in range(frames)]}))
            (d / "meta.json").write_text(json.dumps({
                "tool": "browser_free_record", "n_frames": frames,
                "luma": {"width": FRAME_W, "height": FRAME_H,
                         "frame_bytes": FRAME_W * FRAME_H},
                "audio": {"sample_rate": 48000, "samples": int(n * 1600)}}))
            wavio.write_wav(d / "audio.wav", [0.0] * int(n * 1600), 48000)
            return d

        make_session("1758000001", 2 * 16000)
        # A session built from the REAL capture, when it is on this machine.
        # The synthetic luma has no face in it -- MediaPipe finds no face in
        # a rectangle -- so only a real frame can prove that the harvester
        # writes TAKES and not merely sessions. Where the capture is not
        # present, the take-writing checks are skipped rather than faked.
        real_capture = Path("/mnt/c/proart_capture/A.nv12")
        have_real = real_capture.is_file()
        if have_real:
            d = sessions / "1758000002"
            d.mkdir(parents=True)
            luma, _ = nv12.read_luma(real_capture, width=FRAME_W, height=FRAME_H)
            frames = len(luma)
            (d / "luma.raw").write_bytes(luma.astype(np.uint8).tobytes())
            (d / "timestamps.json").write_text(json.dumps({
                "frame_offsets_ms": [round(i * 1000.0 / 20.0, 1)
                                     for i in range(frames)]}))
            (d / "meta.json").write_text(json.dumps({
                "tool": "browser_free_record", "n_frames": frames,
                "luma": {"width": FRAME_W, "height": FRAME_H,
                         "frame_bytes": FRAME_W * FRAME_H},
                "audio": {"sample_rate": 48000, "samples": 480000}}))
            wavio.write_wav(d / "audio.wav", [0.0] * 480000, 48000)
        else:
            make_session("1758000002", 2 * 16000)
        # and one that is being written right now: its metadata promises more
        # frames than its luma holds
        broken = sessions / "1758000003"
        broken.mkdir()
        (broken / "luma.raw").write_bytes(
            np.full((10, FRAME_H, FRAME_W), 80, dtype=np.uint8).tobytes())
        (broken / "timestamps.json").write_text(json.dumps({
            "frame_offsets_ms": [round(i * 1000.0 / 30.0, 1)
                                 for i in range(10)]}))
        (broken / "meta.json").write_text(json.dumps({
            "tool": "browser_free_record", "n_frames": 900,
            "luma": {"width": FRAME_W, "height": FRAME_H,
                     "frame_bytes": FRAME_W * FRAME_H},
            "audio": {"sample_rate": 48000, "samples": 1000}}))
        wavio.write_wav(broken / "audio.wav", [0.0] * 1000, 48000)

        run1 = harvest(corpus, report_path=report_path, min_segment_s=1.0)
        check("three sessions are found", run1["sessions_found"], 3)
        check("two are harvested", run1["sessions_new"], 2)
        check("one is left for the next pass", run1["sessions_failed"], 1)
        check("the report is written", report_path.is_file(), True)
        check("and it has a run section",
              report_path.read_text().count("## Run "), 1)

        # the second pass must not re-harvest what is already done
        run2 = harvest(corpus, report_path=report_path, min_segment_s=1.0)
        check("the second pass harvests nothing new",
              run2["sessions_new"], 0)
        check("and skips the ones already done",
              run2["sessions_skipped"], 2)
        check("and still retries the broken one",
              run2["sessions_failed"], 1)

        # the derived takes are in their own corpus root, with the SESSION's
        # stamp on every one of them
        derived = corpus / DERIVED_DIR
        check("the derived corpus root exists", derived.is_dir(), True)
        stems = sorted(p.name for p in derived.glob("*.json")
                       if not p.name.endswith("_timing.json"))
        if have_real:
            check("and it holds take metadata", len(stems) > 0, True)
            stamps = set()
            for s in stems:
                m = json.loads((derived / s).read_text())
                stamps.add(m["timestamp"])
            check("every take carries its session's stamp",
                  stamps, {1758000002})
        else:
            print("  [skip] the take-writing checks: /mnt/c/proart_capture "
                  "is not on this machine")
        check("the state file is out of the gate's way",
              (corpus / HARVEST_DIR / HARVEST_STATE).is_file(), True)
        check("and the gate does not see it as a take",
              len(list(corpus.glob("*.json"))), 0)
        gate = run_gate(derived)
        check("the derived corpus passes the gate",
              gate.get("n_failed"), 0)

    print("-" * 60)
    if failures:
        print(f"SELFTEST FAILED: {len(failures)} check(s)")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SELFTEST PASSED")
    return 0


# ============================================================================
# the HARVESTER -- the batch pass that turns a room into a corpus
# ============================================================================

# The state file lives in its OWN subdirectory of the corpus, because the
# gate individuates every root-level *.json as a take.
HARVEST_DIR = "harvest"
HARVEST_STATE = "state.json"
# The derived takes live in their OWN corpus root. They cannot share one
# with the script's takes: the gate's stimulus check fails any take whose
# stimulus differs from the corpus majority, so a derived take ("free
# session") beside a script take ("invictus") would fail one of them.
DERIVED_DIR = "derived"


def _harvest_state_path(corpus_dir: Path) -> Path:
    return Path(corpus_dir) / HARVEST_DIR / HARVEST_STATE


def _load_harvest_state(corpus_dir: Path) -> dict:
    p = _harvest_state_path(corpus_dir)
    if not p.is_file():
        return {"sessions": {}, "runs": 0, "passes_with_nothing_new": 0}
    try:
        data = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {"sessions": {}, "runs": 0, "passes_with_nothing_new": 0}
    if not isinstance(data, dict):
        return {"sessions": {}, "runs": 0, "passes_with_nothing_new": 0}
    data.setdefault("sessions", {})
    data.setdefault("runs", 0)
    data.setdefault("passes_with_nothing_new", 0)
    return data


def _save_harvest_state(corpus_dir: Path, state: dict) -> None:
    p = _harvest_state_path(corpus_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def _session_stamp(session_dir: Path) -> int | None:
    """The session's own stamp -- its folder name, when it is an integer.

    EVERY take derived from one session carries this stamp, and that is what
    makes the gate count one session once. The gate's session identity is
    the take's `timestamp`; if each slice carried its own, one recording
    would count as a dozen independent sessions, which is exactly the
    over-counting `corpus_integrity.py` exists to prevent.
    """
    try:
        return int(session_dir.name)
    except ValueError:
        return None


def harvest_session(session_dir: Path, takes_dir: Path,
                    min_segment_s: float = MIN_SEGMENT_S,
                    min_label_s: float = MIN_LABEL_S,
                    upscale: int = UPSCALE) -> dict:
    """Process ONE free session into derived takes.

    Returns a record of what happened, for the harvest report.
    """
    session_dir = Path(session_dir)
    record = {
        "session": str(session_dir),
        "stamp": _session_stamp(session_dir),
        "takes": [],
        "skipped": [],
        "error": None,
        "notes": [],
    }
    stamp = record["stamp"]
    if stamp is None:
        record["error"] = ("the folder name is not an integer stamp, so the "
                            "derived takes cannot carry a session identity")
        return record

    session = read_session(session_dir)
    record["sources"] = session.sources
    record["n_frames"] = session.n_frames
    record["duration_s"] = session.duration_s
    if session.warnings:
        record["notes"] = list(session.warnings)

    # A session folder being WRITTEN RIGHT NOW looks like an inconsistent
    # one. Skip it, do not mark it processed, and let the next pass retry.
    if any("n_frames is" in w for w in session.warnings):
        record["error"] = ("the session is incomplete or inconsistent with "
                           "its own metadata; it will be retried")
        return record
    if session.luma is None or session.n_frames < 2:
        record["error"] = "no luma stream was found in the session folder"
        return record

    report = analyse_session(session, out_dir=None,
                             min_segment_s=min_segment_s,
                             min_label_s=min_label_s, upscale=upscale)
    if report.get("error"):
        record["error"] = report["error"]
        return record

    # write the takes, with the SESSION's stamp on every one of them
    counters = {}
    for seg in report.get("segments", []):
        if not seg.get("usable"):
            record["skipped"].append({
                "start_s": seg.get("start_s"), "end_s": seg.get("end_s"),
                "face_state": seg.get("face_state"),
                "speech_state": seg.get("speech_state"),
                "reason": seg.get("reason", "not usable")})
            continue
        cond = seg["condition"]
        counters[cond] = counters.get(cond, 0) + 1
        name = f"{cond}{counters[cond]}"
        try:
            written = write_take(takes_dir, seg, session, stamp, name)
        except Exception as exc:  # noqa: BLE001
            record["skipped"].append({"start_s": seg.get("start_s"),
                                      "end_s": seg.get("end_s"),
                                      "reason": f"could not be written: {exc}"})
            continue
        written.update({"condition": cond,
                        "start_s": seg["start_s"], "end_s": seg["end_s"],
                        "seconds": seg["duration_s"],
                        "confidence": seg["confidence"],
                        "face_confidence": seg["face_confidence"],
                        "speech_confidence": seg["speech_confidence"],
                        "face_state": seg["face_state"],
                        "speech_state": seg["speech_state"]})
        record["takes"].append(written)

    # the session's own segment table, beside the takes
    rich = Path(takes_dir) / "rich"
    rich.mkdir(parents=True, exist_ok=True)
    (rich / f"segments_{stamp}.json").write_text(json.dumps(report, indent=2))
    record["segments_json"] = str(rich / f"segments_{stamp}.json")
    record["segments"] = len(report.get("segments", []))
    return record


def harvest(corpus_dir: Path, takes_dir: Path | None = None,
            report_path: Path | None = None,
            min_segment_s: float = MIN_SEGMENT_S,
            min_label_s: float = MIN_LABEL_S,
            upscale: int = UPSCALE,
            reharvest: bool = False,
            append: bool = True) -> dict:
    """One batch pass over a corpus's free sessions.

    Every session folder under `<corpus>/free_sessions/` that has not been
    harvested before is processed, its derived takes are written, and a
    section is appended to the harvest report.
    """
    corpus_dir = Path(corpus_dir)
    sessions_root = corpus_dir / "free_sessions"
    takes_dir = Path(takes_dir) if takes_dir else corpus_dir / DERIVED_DIR
    state = _load_harvest_state(corpus_dir)

    found = sorted([p for p in sessions_root.iterdir() if p.is_dir()]) \
        if sessions_root.is_dir() else []

    run = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus": str(corpus_dir),
        "takes_dir": str(takes_dir),
        "sessions_found": len(found),
        "sessions_new": 0,
        "sessions_skipped": 0,
        "sessions_failed": 0,
        "records": [],
        "takes": [],
        "gate": None,
    }

    for session_dir in found:
        stamp = _session_stamp(session_dir)
        key = str(stamp) if stamp is not None else session_dir.name
        if key in state["sessions"] and not reharvest:
            run["sessions_skipped"] += 1
            continue
        rec = harvest_session(session_dir, takes_dir,
                              min_segment_s=min_segment_s,
                              min_label_s=min_label_s, upscale=upscale)
        run["records"].append(rec)
        if rec["error"]:
            run["sessions_failed"] += 1
            # NOT recorded in the state: a failed session is retried.
            continue
        run["sessions_new"] += 1
        state["sessions"][key] = {
            "session": str(session_dir),
            "stamp": stamp,
            "n_frames": rec.get("n_frames"),
            "duration_s": rec.get("duration_s"),
            "takes": [t["stem"] for t in rec["takes"]],
            "harvested_at": run["started_at"],
        }
        run["takes"].extend(rec["takes"])

    # the gate over the derived corpus -- the coverage numbers are the
    # repository's own, not a summary of my own
    if takes_dir.is_dir():
        gate = run_gate(takes_dir)
        run["gate"] = {"n_takes": gate.get("n_takes"),
                       "n_failed": gate.get("n_failed"),
                       "ok": gate.get("ok"),
                       "error": gate.get("error")}
        try:
            greport = _gate().analyse(takes_dir)
            run["coverage"] = greport.get("coverage")
            run["stimulus"] = greport.get("stimulus")
        except Exception:  # noqa: BLE001
            run["coverage"] = None
        run["cumulative"] = _cumulative(takes_dir)

    state["runs"] = int(state.get("runs", 0)) + 1
    if run["sessions_new"] == 0:
        state["passes_with_nothing_new"] = int(
            state.get("passes_with_nothing_new", 0)) + 1
    else:
        state["passes_with_nothing_new"] = 0
    _save_harvest_state(corpus_dir, state)

    run["passes_with_nothing_new"] = state["passes_with_nothing_new"]
    run["total_sessions_known"] = len(state["sessions"])
    if report_path is not None and append and (run["sessions_new"] > 0
                                               or run["sessions_failed"] > 0):
        append_harvest_report(Path(report_path), run)
        run["report_appended"] = str(report_path)
    return run


# -- the harvest report ------------------------------------------------------

def _condition_table(takes: list[dict], key: str = "condition") -> str:
    """Minutes, takes and confidences per derived condition."""
    per: dict = {}
    for t in takes:
        c = t.get(key) or "-"
        e = per.setdefault(c, {"takes": 0, "seconds": 0.0, "confs": []})
        e["takes"] += 1
        e["seconds"] += float(t.get("seconds", 0.0) or 0.0)
        e["confs"].append(float(t.get("confidence", 0.0) or 0.0))
    lines = ["| condition | takes | minutes | mean conf | min conf | max conf |",
             "|---|---|---|---|---|---|"]
    for c in sorted(per):
        e = per[c]
        confs = e["confs"] or [0.0]
        lines.append(
            f"| {c} | {e['takes']} | {e['seconds'] / 60.0:.2f} | "
            f"{sum(confs) / len(confs):.3f} | {min(confs):.3f} | {max(confs):.3f} |")
    if not per:
        lines.append("| (none) | 0 | 0.00 | - | - | - |")
    return "\n".join(lines)


def _cumulative(takes_dir: Path) -> dict:
    """What the derived corpus holds in TOTAL, read from the takes on disk.

    The report is appended, so a reader needs to know the standing as well
    as the run's delta -- and the standing is read from the corpus itself,
    not from a running total that could drift.
    """
    per: dict = {}
    sessions: set = set()
    total_s = 0.0
    for p in sorted(Path(takes_dir).glob("*.json")):
        if p.name.endswith("_timing.json"):
            continue
        try:
            meta = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(meta, dict) or "condition" not in meta:
            continue
        cond = str(meta.get("condition"))
        e = per.setdefault(cond, {"takes": 0, "seconds": 0.0, "confs": [],
                                   "sessions": set()})
        e["takes"] += 1
        e["seconds"] += float(meta.get("seconds", 0.0) or 0.0)
        e["confs"].append(float((meta.get("derived") or {})
                                .get("confidence", 0.0) or 0.0))
        ts = meta.get("timestamp")
        if ts is not None:
            e["sessions"].add(str(ts))
            sessions.add(str(ts))
        total_s += float(meta.get("seconds", 0.0) or 0.0)
    return {"per_condition": per, "sessions": len(sessions),
            "minutes": total_s / 60.0}


def format_harvest(run: dict) -> str:
    """The section appended to the harvest report."""
    lines = []
    lines.append(f"## Run {run['started_at']} — {run['sessions_new']} new "
                 f"session(s), {run['sessions_failed']} failed")
    lines.append("")
    lines.append(f"**corpus** `{run['corpus']}`  ")
    lines.append(f"**derived takes** `{run['takes_dir']}`  ")
    lines.append(f"**sessions found** {run['sessions_found']}, "
                 f"**new** {run['sessions_new']}, "
                 f"**already harvested** {run['sessions_skipped']}, "
                 f"**failed** {run['sessions_failed']}  ")
    lines.append(f"**takes written** {len(run['takes'])}  ")
    lines.append(f"**sessions known to the harvester** "
                 f"{run.get('total_sessions_known', 0)}")
    lines.append("")

    lines.append("### Minutes harvested this run")
    lines.append("")
    lines.append(_condition_table(run["takes"]))
    lines.append("")

    cum = run.get("cumulative")
    if cum and cum.get("per_condition"):
        lines.append("### Minutes harvested in total, from the derived corpus")
        lines.append("")
        rows = []
        for c, e in sorted(cum["per_condition"].items()):
            confs = e["confs"] or [0.0]
            rows.append(f"| {c} | {e['takes']} | {e['seconds'] / 60.0:.2f} | "
                        f"{sum(confs) / len(confs):.3f} | {min(confs):.3f} | "
                        f"{max(confs):.3f} | {len(e['sessions'])} |")
        lines.append("| condition | takes | minutes | mean conf | min conf | "
                     "max conf | sessions |")
        lines.append("|---|---|---|---|---|---|---|")
        lines.extend(rows)
        lines.append("")
        lines.append(f"**{cum['minutes']:.2f} minutes in total, from "
                     f"{cum['sessions']} independent session(s).**")
        lines.append("")

    gate = run.get("gate") or {}
    if gate:
        lines.append(f"### The gate over the derived corpus")
        lines.append("")
        lines.append(f"**{gate.get('n_takes')} takes, {gate.get('n_failed')} "
                     f"failed, ok={gate.get('ok')}**")
        lines.append("")

    cov = run.get("coverage")
    if cov:
        lines.append("### Coverage against the target")
        lines.append("")
        lines.append(f"Target: **{cov.get('target_sessions_per_class')} "
                     f"independent sessions per class** "
                     f"({cov.get('target_source')}).")
        lines.append("")
        lines.append("| class | takes | distinct sessions | usable sessions | "
                     "gap to target |")
        lines.append("|---|---|---|---|---|")
        for cls, c in sorted((cov.get("per_class") or {}).items()):
            gap = c.get("gap_to_target")
            gap_s = "-" if gap is None else str(gap)
            lines.append(f"| {cls} | {c.get('takes')} | "
                         f"{c.get('distinct_sessions')} | "
                         f"{c.get('usable_sessions')} | {gap_s} |")
        lines.append("")
        per_cond = cov.get("per_condition") or {}
        if per_cond:
            lines.append("| condition | takes | distinct sessions | "
                         "usable sessions | gap to target |")
            lines.append("|---|---|---|---|---|")
            for c, e in sorted(per_cond.items()):
                gap = e.get("gap_to_target")
                gap_s = "-" if gap is None else str(gap)
                lines.append(f"| {c} | {e.get('takes')} | "
                             f"{e.get('distinct_sessions')} | "
                             f"{e.get('usable_sessions')} | {gap_s} |")
            lines.append("")

    lines.append("### What is still missing")
    lines.append("")
    lines.extend(_what_is_missing(run))
    lines.append("")

    failed = [r for r in run.get("records", []) if r.get("error")]
    if failed:
        lines.append("### Sessions that failed")
        lines.append("")
        for r in failed:
            lines.append(f"- `{r['session']}` — {r['error']}")
        lines.append("")

    notes = [r for r in run.get("records", []) if r.get("notes")]
    if notes:
        lines.append("### Sessions with inconsistent metadata")
        lines.append("")
        for r in notes:
            lines.append(f"- `{r['session']}` — {'; '.join(r['notes'])}")
        lines.append("")
    return "\n".join(lines)


def _what_is_missing(run: dict) -> list[str]:
    """What the room still has to give, said plainly."""
    out = []
    cov = run.get("coverage")
    if not cov:
        out.append("- No coverage is available yet: no derived corpus has "
                   "been written.")
        return out
    target = cov.get("target_sessions_per_class", 0)
    per_class = cov.get("per_class") or {}
    speak = per_class.get("SPEAK", {})
    silent = per_class.get("SILENT", {})
    s_gap = speak.get("gap_to_target")
    l_gap = silent.get("gap_to_target")
    if s_gap is not None and s_gap > 0:
        out.append(f"- **SPEAK is {s_gap} sessions short of the {target} "
                   f"target** ({speak.get('distinct_sessions')} recorded). "
                   f"Each free session that contains talking fills it.")
    elif s_gap is not None:
        out.append(f"- SPEAK has met the {target}-session target.")
    if l_gap is not None and l_gap > 0:
        out.append(f"- **SILENT is {l_gap} sessions short of the {target} "
                   f"target** ({silent.get('distinct_sessions')} recorded). "
                   f"It is the harder half: the room has to be quiet on "
                   f"purpose.")
    elif l_gap is not None:
        out.append(f"- SILENT has met the {target}-session target.")
    per_cond = cov.get("per_condition") or {}
    prefixes: dict = {}
    for c, e in per_cond.items():
        if (e.get("gap_to_target") or 0) > 0:
            p = c.rstrip("0123456789") or c
            prefixes[p] = prefixes.get(p, 0) + 1
    if prefixes:
        out.append("- The conditions still short of the target: "
                   + ", ".join(f"{p} ({prefixes[p]} take"
                              + ("s)" if prefixes[p] != 1 else ")")
                              for p in sorted(prefixes))
                   + ". The target is per CLASS, so these are a record of "
                     "which behaviour the room has been asked for, not a "
                     "separate quota.")
    if not run.get("takes"):
        out.append("- **No takes were harvested in this run.** A session "
                   "yields nothing when the face is not found (an empty "
                   "room), when no segment is long enough, or when the "
                   "speech sits between the two thresholds and is "
                   "therefore ambiguous.")
    if run.get("passes_with_nothing_new"):
        out.append(f"- {run['passes_with_nothing_new']} consecutive pass(es) "
                   f"found nothing new. The corpus is not growing.")
    return out


HARVEST_REPORT_HEADER = """# HARVEST REPORT — what the room gave us

**Appended by:** `experiments/free_session_analyze.py --harvest`
**Corpus:** `/mnt/c/proart_capture/corpus` (the capture page's output
directory; its free sessions arrive under `free_sessions/`)
**Derived takes:** `<corpus>/derived/` — its own corpus root, because the
gate's stimulus check fails any take whose stimulus differs from the
corpus majority, so a derived take cannot sit beside a script take.

This report is APPENDED, once per run. Each section is one pass of the
harvester over the corpus: the minutes it harvested per condition, the
confidence of those labels, the coverage against the target of 63
independent sessions per class, and what is still missing.

The steady state this buys: **Casey does nothing.** The page records, the
harvester derives, and this file says what the room produced.

**The first section below is a DRY RUN**, over the captures that exist
today: the script's A, B, C and empty1 recordings, re-encoded in the
free-record layout so the harvester could be exercised over the room that
is already on disk. Its numbers are real measurements of real footage. The
minutes are small because the captures are thirty seconds each.

**To run it:**

```bash
# one pass (this is the cron line)
/home/proart/projects/gpu-venv/bin/python experiments/free_session_analyze.py \\
    --harvest /mnt/c/proart_capture/corpus

# or continuously
/home/proart/projects/gpu-venv/bin/python experiments/free_session_analyze.py \\
    --harvest /mnt/c/proart_capture/corpus --watch 60
```

---
"""


def append_harvest_report(path: Path, run: dict) -> None:
    """Append one run's section to the harvest report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(HARVEST_REPORT_HEADER)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + format_harvest(run) + "\n")


# ============================================================================
# main
# ============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", nargs="?", default=None,
                    help="the free-session folder to analyse")
    ap.add_argument("--out", default=None,
                    help="write the sliced takes into this directory")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="write the report to this path")
    ap.add_argument("--min-segment-s", type=float, default=MIN_SEGMENT_S)
    ap.add_argument("--min-label-s", type=float, default=MIN_LABEL_S)
    ap.add_argument("--upscale", type=int, default=UPSCALE)
    ap.add_argument("--recover", default=None, metavar="DIR",
                    help="run the recovery report over a capture directory")
    ap.add_argument("--slices", default=None, metavar="DIR",
                    help="with --recover, also write the derived takes there "
                         "and run the integrity gate over them")
    ap.add_argument("--gate", action="store_true",
                    help="with --out, run the integrity gate over the "
                         "derived takes and report its verdict")
    ap.add_argument("--selftest", action="store_true",
                    help="unit-test the slicer on synthetic arrays")
    ap.add_argument("--harvest", default=None, metavar="DIR",
                    help="batch pass over DIR/free_sessions/: derive the "
                         "takes, write them, append the harvest report")
    ap.add_argument("--watch", type=float, default=0.0, metavar="SECONDS",
                    help="with --harvest, repeat the pass every SECONDS "
                         "instead of once")
    ap.add_argument("--takes-dir", default=None, metavar="DIR",
                    help="where the harvested takes go "
                         "(default: <corpus>/derived)")
    ap.add_argument("--report", default=None, metavar="PATH",
                    help="the harvest report to append to "
                         "(default: docs/research/HARVEST-REPORT.md)")
    ap.add_argument("--reharvest", action="store_true",
                    help="process sessions that were harvested before")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    if a.harvest:
        corpus = Path(a.harvest)
        report_path = (Path(a.report) if a.report
                       else ROOT / "docs" / "research" / "HARVEST-REPORT.md")
        while True:
            run = harvest(corpus,
                          takes_dir=Path(a.takes_dir) if a.takes_dir else None,
                          report_path=report_path,
                          min_segment_s=a.min_segment_s,
                          min_label_s=a.min_label_s,
                          upscale=a.upscale,
                          reharvest=a.reharvest)
            print(f"harvest: {run['sessions_found']} session(s) found, "
                  f"{run['sessions_new']} new, "
                  f"{run['sessions_skipped']} already harvested, "
                  f"{run['sessions_failed']} failed, "
                  f"{len(run['takes'])} take(s) written")
            for rec in run.get("records", []):
                for t in rec.get("takes", []):
                    print(f"  {t['stem']:<24} {t['condition']}  "
                          f"{t['start_s']:.1f}-{t['end_s']:.1f}s  "
                          f"{t['seconds'] / 60.0:.2f} min  "
                          f"conf {t['confidence']:.3f}")
                if rec.get("error"):
                    print(f"  FAILED {rec['session']}: {rec['error']}")
            if run.get("report_appended"):
                print(f"  report appended to {run['report_appended']}")
            if a.watch and a.watch > 0:
                time.sleep(a.watch)
                continue
            return 0 if run["sessions_failed"] == 0 else 1

    if a.recover:
        rep = recovery_report(Path(a.recover),
                              slices_dir=Path(a.slices) if a.slices else None)
        print(format_recovery(rep))
        if rep.get("gate"):
            g = rep["gate"]
            print("")
            print(f"THE GATE over the derived takes: {g['n_takes']} takes, "
                  f"{g['n_failed']} failed, ok={g['ok']}")
            for cap, res in g.get("per_capture", {}).items():
                fails = res.get("failures") or []
                print(f"  {cap:<9} {res.get('n_takes')} takes  "
                      f"{res.get('n_failed')} failed"
                      + ("" if not fails else f"  {fails}"))
        if a.json_path:
            Path(a.json_path).write_text(json.dumps(rep, indent=2))
            print(f"wrote {a.json_path}")
        ok = rep["recovered"] == rep["total"]
        if rep.get("gate"):
            ok = ok and bool(rep["gate"].get("ok"))
        return 0 if ok else 1

    if not a.session:
        ap.error("give a session directory, or --recover, or --selftest")

    session = read_session(Path(a.session))
    print(f"session   {session.path}")
    for k, v in session.sources.items():
        print(f"  {k:<11} {v}")
    for w in session.warnings:
        print(f"  WARNING    {w}")
    report = analyse_session(session,
                             out_dir=Path(a.out) if a.out else None,
                             min_segment_s=a.min_segment_s,
                             min_label_s=a.min_label_s,
                             upscale=a.upscale)
    if report.get("error"):
        print(f"ERROR: {report['error']}", file=sys.stderr)
        return 2
    print()
    print(format_segments(report))
    if a.gate and a.out:
        gate = run_gate(Path(a.out))
        print()
        if gate.get("error"):
            print(f"the gate could not be run: {gate['error']}")
            return 2
        print(gate["text"])
    if a.json_path:
        Path(a.json_path).write_text(json.dumps(report, indent=2))
        print(f"wrote {a.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
