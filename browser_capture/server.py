"""
server.py -- the localhost half of the browser-native corpus capture.

WHY THIS EXISTS (docs/research/BRIEF-BROWSER-CAPTURE.md). The Windows-terminal
capture flow works but is clunky. This server serves two pages:
free_record.html at / -- the FRONT DOOR, a FREE RECORD mode with one
RECORD button and one STOP button that saves the whole raw session for
offline analysis -- and index.html at /plan, the condition-plan surface
(live self-preview, big prompts, one click per plan). The plan page's
DATA PRODUCT IS UNCHANGED -- every take it accepts is written in EXACTLY
the layout `corpus_capture_win.py --plan` writes, so the existing
pipeline (src/nv12.py, src/wavio.py, experiments/corpus_integrity.py)
accepts the takes without modification.

PYTHON STDLIB ONLY. No numpy, no pip installs, no external calls. The one
place that stings is the .npz: the script writes it with
np.savez_compressed, so this file carries a hand-rolled NPY-format writer
inside a deflate zip (see _npy_* below) that emits what numpy emits for the
arrays the script stores -- uint8 maps, float32 scalar scales, uint8 luma.
browser_capture/verify.py proves np.load reads the result back identically.

WHO COMPUTES WHAT. The browser does the frame math (row medians, change maps,
quantisation, drift): its typed arrays make that a sub-second job, while
stdlib Python would take minutes on a 5-minute take. So one POST per finished
take carries the raw luma frames, BOTH quantised map arrays with their
scales, the finished WAV bytes, and the timing/meta JSON; this server
validates everything and writes the four files.

THE POST BODY (Content-Type: application/octet-stream), eight sections, each
an unsigned 64-bit little-endian length followed by that many bytes:

    [u64 len][meta JSON]   -- the take's metadata, WITHOUT "timestamp"
                             (the server assigns it, as the script's stamp)
    [u64 len][timing JSON] -- timing.json content verbatim
    [u64 len][WAV bytes]   -- 16-bit PCM mono at the ACHIEVED rate
    [u64 len][luma]        -- n_frames * 160 * 96 raw luma bytes
    [u64 len][maps]        -- (n_frames - 1) * 48 * 80 quantised uint8
    [u64 len][maps_raw]    -- same shape, unfiltered variant
    [u64 len][rich JSON]   -- the per-take rich sidecar body (addendum)
    [u64 len][WebM bytes]  -- native-res video of the take, or empty if the
                             browser could not record one

The meta/timing JSON key sets must match the script's schema exactly; extra
provenance keys (e.g. "capture_tool") are allowed and passed through,
because the integrity gate ignores keys it does not read.

THE RICH COPIES (docs/research/BRIEF-BROWSER-CAPTURE-ADDENDUM.md). Every
take is also preserved at full fidelity for external analysis: a
<name>_<stamp>_full.webm (native res + mic audio, recorded in parallel --
an ADDITIONAL artifact, never a substitute for the raw frames), and a
<name>_<stamp>_rich.json sidecar (label, instruction shown, stimulus, wall
clocks, rates, device labels, a free-text "notes" field). A session.json
manifest at the out root lists every take with its files, labels and flags.

WHERE THE RICH FILES LIVE, AND WHY NOT LITERALLY "AT THE OUT ROOT": the
integrity gate (experiments/corpus_integrity.py, find_takes) treats EVERY
*.json IN THE OUT ROOT as a take metadata file, and the gate is not ours to
modify. A session.json or *_rich.json at the root would make every future
gate run fail. So the WebM sits next to its take (the gate only reads
.json), and the two JSON artifacts live in out/rich/ -- one subdirectory,
invisible to the gate's non-recursive glob, one hop from everything they
reference. verify.py asserts the gate stays green with all of this present.

ENDPOINTS:
    GET  /          -> free_record.html, the FREE RECORD front door (no-store,
                       so a refresh always gets latest)
    GET  /plan      -> index.html, the condition-plan page (no-store)
    GET  /evolve    -> evolve.html, THE LAB: the stable shell (capture +
                       probe engine + safety rails) whose probes hot-swap
                       without reloads
    GET  /probes/*  -> the versioned probe modules and manifest.json,
                       re-read from disk on every request (an agent's
                       edit reaches already-open pages on the next cycle)
    GET  /info      -> {"out": ..., "port": ...}
    GET  /takes     -> every take already in the out dir (script-made ones
                       included), newest first, with file presence and the
                       fields the UI needs to mirror the gate's checks
    POST /take      -> write one take, respond {"ok": true, "files": [...]}
                       or 400 with a reason
    POST /notes     -> {"condition", "stamp", "notes"}: update a take's
                       rich sidecar notes field ("what I actually did"),
                       then regenerate session.json
    POST /free      -> write one FREE RECORD session folder under
                       out/free_sessions/<stamp>/ (webm, luma.raw,
                       timestamps.json, audio.wav, meta.json); the same
                       length-prefix framing as /take but FIVE sections
    POST /log       -> append one JSONL diagnostics line from a page to
                       browser_capture/diag.log, stamped on receipt
    GET  /live      -> live.html, the LIVE MONITOR (read-only window on
                       the capture page: heartbeat state, thumbnail,
                       events, saved segments, harvest verdicts)
    GET  /live-state -> everything /live polls: the tail of the diag
                       heartbeat feed, the latest thumbnail, the session
                       folders with sizes, and any verdict.json
    POST /thumb     -> the capture page's latest 160x96 luma thumbnail
                       (kept in memory only, replaced every second)
    POST /luma      -> the capture page's newest live luma batch (the
                       last ~90 frames + per-frame offsets, binary);
                       kept in memory only, replaced every couple of
                       seconds -- the LIVE FORWARDER's source
    GET  /luma-latest -> that batch back as JSON {n, offsets, luma_b64},
                       for the forwarder sidecar

THE AGENT API (docs/research/BRIEF-ROOM-LIAISON.md) -- the liaison in
the room. JSON on localhost, meant for agents, additive to the pages:
    GET  /api/status   -> uptime, page state from the last heartbeat,
                          sessions saved, rendezvous ship health
    GET  /api/sessions -> every free session, meta PLUS MEASURED audio
                          stats (peak, nonzero, RMS) computed from the
                          WAV files -- artifact health is measured,
                          never asserted
    POST /api/inbox    -> {from, text} appended to liaison_inbox.jsonl;
                          local agents message the room through this
    GET  /api/inbox    -> the recent inbox lines

SELF-CHECKS: liaison events (session_saved, segment_closed, deafness,
device failures) append to liaison_outbox.jsonl and are shipped to a
rendezvous on the Oracle; a saved WAV whose MEASURED peak is 0 raises a
SILENT_AUDIO alert immediately -- the liaison reports its own sickness.

USAGE (from the repo root, inside WSL):

    python3 browser_capture/server.py                 # default out dir below
    python3 browser_capture/server.py --out /tmp/ct   # somewhere else
    python3 browser_capture/server.py --port 8799     # verify.py uses this

Then open http://localhost:8765 in the Windows browser (WSL2's localhost
forwarding reaches it; with mirrored networking it also just works).
"""

from __future__ import annotations

import argparse
import array
import io
import json
import math
import os
import re
import struct
import subprocess
import sys
import threading
import time
import wave
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The script's constants (corpus_capture_win.py). The take format is defined
# by these numbers; they are repeated here rather than imported, because the
# script lives on the Windows side (/mnt/c/proart_capture) and this server
# must run anywhere with nothing but the repo.
W, H = 160, 96
MAP_W, MAP_H = 80, 48
FPS = 30
RATE = 16000
LUMA_BYTES = W * H                 # 15,360 -- also src/nv12.py's luma_only stride
MAP_BYTES = MAP_W * MAP_H          # 3,840

# The plan names the script's --plan mode can produce (REPEATS=3 over
# A,B,C,D,F, then FREE). A POST naming anything else is rejected: the corpus
# is not a place to improvise identifiers.
PLAN_NAMES = {f"{c}{rep}" for rep in (1, 2, 3) for c in "ABCDF"} | {"FREE"}

NAME_RE = re.compile(r"^[A-Za-z]{1,8}[0-9]{0,3}$")

# The script's timing.json keys, verbatim (corpus_capture_win.py, `timing`).
TIMING_KEYS = [
    "condition", "n_frames", "frame_offsets_s", "assumed_fps", "achieved_fps",
    "video_span_s", "audio_span_s", "audio_sample_rate",
    "requested_sample_rate", "mic", "capture_wallclock_s",
]

# The script's metadata keys, verbatim (`meta`). "timestamp" is assigned by
# THIS server at write time, exactly as the script assigns its own stamp, so
# the filename and the metadata always agree.
META_KEYS = [
    "condition", "kind", "note", "pose", "speak", "stimulus",
    "stimulus_title", "stimulus_estimated_s", "speech_frac", "n_frames",
    "seconds", "achieved_fps", "audio_sample_rate", "audio_seconds",
    "map_shape", "map_scale", "map_scale_raw", "drift_raw",
    "drift_after_row_removal", "timestamp", "rep", "filter", "filters",
    "arrays",
]
META_KEYS_FROM_CLIENT = [k for k in META_KEYS if k != "timestamp"]

MAX_BODY = 512 * 1024 * 1024  # a 5-minute FREE take posts ~220 MB

# A free session posts luma + full-rate WAV + native-res WebM in one body;
# at ~4 Mbps webm + 30 fps luma that is roughly 35 minutes of headroom.
MAX_FREE_BODY = 2 * 1024 * 1024 * 1024

# The diagnostics heartbeat (see DONE.md): the pages POST /log while a
# take/session runs, and every line lands here as JSONL, stamped on
# receipt. Appends are serialized -- ThreadingHTTPServer can serve /log
# concurrently with /take and /free.
DIAG_LOCK = threading.Lock()
DIAG_PATH = Path(__file__).resolve().parent / "diag.log"

# The LIVE MONITOR's thumbnail: the capture page POSTs a small JPEG of
# the 160x96 luma canvas once a second; only the LATEST is kept, in
# memory -- it is a window, not a record.
THUMB_LOCK = threading.Lock()
THUMB = {"data": None, "ts": None}
THUMB_MAX = 262144

# The live luma batch (the LIVE FORWARDER's source): the page posts the
# newest ~90 frames every couple of seconds while AUTO runs; only the
# LATEST is kept, in memory.
LUMA_LOCK = threading.Lock()
LUMA_LATEST = {"n": 0, "offsets": [], "luma": b"", "ts": None}
LUMA_MAX = 8 * 1024 * 1024

# The luma frame the FREE RECORD page captures, same geometry as the plan
# page's data view.
FREE_W, FREE_H = 160, 96
FREE_LUMA_BYTES = FREE_W * FREE_H

# -- the LIAISON (docs/research/BRIEF-ROOM-LIAISON.md) -----------------------
# The local host is the agent in the room: an agent API on localhost, an
# outbox of liaison events, self-reported health alerts, and a shipper to
# a rendezvous on the Oracle. The outbox file is the durable record; the
# ship is best-effort with honest failure reporting and retry.
LIAISON_INBOX = Path(__file__).resolve().parent / "liaison_inbox.jsonl"
LIAISON_OUTBOX = Path(__file__).resolve().parent / "liaison_outbox.jsonl"
LIAISON_PENDING = Path(__file__).resolve().parent / "liaison_outbox.pending.jsonl"
LIAISON_STARTED_AT = time.time()
LIAISON_LOCKS = {name: threading.Lock() for name in
                 ("inbox", "outbox", "pending", "audio")}
LIAISON_AUDIO_CACHE = {}   # (path, size, mtime) -> measured stats
LIAISON_SHIP_Q = None      # queue.Queue, created in main()
RENDEZVOUS = {
    "user": "wsl-tunnel",
    "host": "147.224.38.131",      # the Oracle the reverse tunnel ends on
    "key": str(Path.home() / ".ssh" / "wsl-tunnel"),
    "remote": "rendezvous/room-liaison/events.jsonl",
    # v1 reality, stated: the wsl-tunnel account is nologin for shells
    # AND sftp, so every ship attempt refuses until a writable path is
    # granted; failures are self-reported in /api/status and retried.
    # Muse can read the outbox meanwhile through the reverse tunnel
    # (Oracle loopback :2222 -> this host's sshd).
}


def wav_audio_stats(path: Path) -> dict:
    """MEASURED audio health of a saved WAV: peak abs sample, nonzero
    count, RMS. Computed from the file, never asserted; memoized on
    (size, mtime) so repeat API calls are instant.
    """
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime)
    with LIAISON_LOCKS["audio"]:
        if key in LIAISON_AUDIO_CACHE:
            return dict(LIAISON_AUDIO_CACHE[key])
    with wave.open(str(path), "rb") as w:
        rate, nframes = w.getframerate(), w.getnframes()
        frames = w.readframes(nframes)
    a = array.array("h")
    a.frombytes(frames[:len(frames) // 2 * 2])
    if sys.byteorder == "big":
        a.byteswap()
    if len(a):
        peak = max(max(a), -min(a))
        nonzero = len(a) - a.count(0)
        acc = 0
        CH = 1 << 20
        for i in range(0, len(a), CH):
            c = a[i:i + CH]
            acc += math.sumprod(c, c)
        rms = math.sqrt(acc / len(a)) / 32768.0
    else:
        peak, nonzero, rms = 0, 0, 0.0
    stats = {"samples": len(a), "sample_rate": rate,
             "peak_abs": peak, "nonzero": nonzero,
             "rms": round(rms, 6)}
    with LIAISON_LOCKS["audio"]:
        LIAISON_AUDIO_CACHE[key] = stats
    return dict(stats)


def liaison_event(ev: dict) -> None:
    """Append one liaison event to the durable outbox and queue it for
    the rendezvous ship."""
    ev = dict(ev)
    ev.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    line = json.dumps(ev) + "\n"
    with LIAISON_LOCKS["outbox"]:
        with LIAISON_OUTBOX.open("a", encoding="utf-8") as f:
            f.write(line)
    with LIAISON_LOCKS["pending"]:
        with LIAISON_PENDING.open("a", encoding="utf-8") as f:
            f.write(line)
    if LIAISON_SHIP_Q is not None:
        LIAISON_SHIP_Q.put(line)
    if ev.get("class") == "alert":
        print(f"[liaison][ALERT] {ev.get('kind')}: {ev.get('dir')}",
              flush=True)


def _ship_line(line: str) -> bool:
    r = RENDEZVOUS
    cmd = ["ssh", "-i", r["key"], "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=accept-new",
           f"{r['user']}@{r['host']}",
           "mkdir -p ~/rendezvous/room-liaison && cat >> ~/" + r["remote"]]
    try:
        p = subprocess.run(cmd, input=line.encode(), capture_output=True,
                           timeout=15)
        return p.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _mark_shipped(line: str) -> None:
    with LIAISON_LOCKS["pending"]:
        if LIAISON_PENDING.is_file():
            lines = LIAISON_PENDING.read_text().splitlines(True)
            kept = [l for l in lines if l != line]
            LIAISON_PENDING.write_text("".join(kept))


def _liaison_shipper(q) -> None:
    """Ships outbox lines to the Oracle rendezvous; failures stay in the
    pending file and retry every 5 minutes and on every new event."""
    import queue as queue_mod
    last_fail_log = 0.0
    while True:
        try:
            item = q.get(timeout=300)
            batch = [item]
            while True:
                try:
                    batch.append(q.get_nowait())
                except queue_mod.Empty:
                    break
        except queue_mod.Empty:
            with LIAISON_LOCKS["pending"]:
                batch = (LIAISON_PENDING.read_text().splitlines(True)
                         if LIAISON_PENDING.is_file() else [])
            batch = [b for b in batch if b.strip()]
        for line in batch:
            if _ship_line(line):
                _mark_shipped(line)
            elif time.time() - last_fail_log > 300:
                last_fail_log = time.time()
                print("[liaison] rendezvous ship refused (reported in "
                      "/api/status, retrying)", flush=True)
                break   # keep the rest pending; no point hammering


def _last_page_heartbeat() -> dict:
    """The newest page heartbeat from diag.log, with its age."""
    try:
        with DIAG_PATH.open("rb") as f:
            size = f.seek(0, 2)
            f.seek(max(0, size - 65536))
            tail = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return {"present": False}
    for ln in reversed(tail):
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and d.get("kind") == "heartbeat":
            age = max(0, int(time.time() - time.mktime(
                time.strptime(d["received_at"][:19], "%Y-%m-%dT%H:%M:%S"))))
            return {"present": True, "age_s": age, "fields": d}
    return {"present": False}


# page diag events the liaison forwards to its outbox (the page's own
# SEGMENT_SAVED is superseded by the server's measured session_saved)
LIAISON_FORWARD_KINDS = {"BOOT", "DEAF_ON", "DEAF_OFF", "DEVICE_FAILURE",
                         "SEGMENT_CLOSE", "MIC_SWITCH", "CAM_SWITCH"}


# -- the NPY format, by hand (numpy is not stdlib) --------------------------
#
# One .npy member is: b"\x93NUMPY" + version(1,0) + u16 header length + a
# literal Python dict + space padding + "\n", padded so the total header is
# a multiple of 16. np.load is indifferent to the padding, but matching
# numpy's layout keeps the archive byte-comparable in verification.

def _npy_header(descr: str, shape: tuple) -> bytes:
    shape_repr = repr(tuple(shape)) if shape else "()"
    body = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (
        descr, shape_repr)
    while (10 + len(body) + 1) % 16 != 0:
        body += " "
    body += "\n"
    if len(body) > 0xFFFF:
        raise ValueError("npy header too long")
    return b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(body)) \
        + body.encode("ascii")


def _npy_u8(shape: tuple, data: bytes) -> bytes:
    expected = 1
    for dim in shape:
        expected *= dim
    if len(data) != expected:
        raise ValueError(f"u8 array payload {len(data)} != {expected} bytes")
    return _npy_header("|u1", shape) + data


def _npy_f4_scalar(value: float) -> bytes:
    return _npy_header("<f4", ()) + struct.pack("<f", value)


def write_npz(path: Path, n_frames: int, maps: bytes, scale: float,
              maps_raw: bytes, scale_raw: float, luma: bytes) -> None:
    """The take archive, in the script's array order and compression.

    np.savez_compressed(npz, maps=q, scale=np.float32(scale), maps_raw=q_raw,
    scale_raw=np.float32(scale_raw), luma=Y) -- same members, same deflation,
    built without numpy. Written to a temp name and renamed into place so a
    killed POST can never leave a half-written archive in the corpus.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("maps.npy", _npy_u8((n_frames - 1, MAP_H, MAP_W), maps))
        z.writestr("scale.npy", _npy_f4_scalar(scale))
        z.writestr("maps_raw.npy", _npy_u8((n_frames - 1, MAP_H, MAP_W),
                                           maps_raw))
        z.writestr("scale_raw.npy", _npy_f4_scalar(scale_raw))
        z.writestr("luma.npy", _npy_u8((n_frames, H, W), luma))
    tmp.replace(path)


# -- POST body parsing -------------------------------------------------------

def parse_take_body(body: bytes) -> dict:
    """Split the six length-prefixed sections; parse and type-check them.

    Raises ValueError with a human-readable reason on any mismatch. Every
    check here mirrors a property the capture script guarantees by
    construction, so a take that cannot satisfy it is not a take.
    """
    sections, off = [], 0
    for _ in range(8):
        if off + 8 > len(body):
            raise ValueError("body ended inside a length prefix")
        (n,) = struct.unpack_from("<Q", body, off)
        off += 8
        if off + n > len(body):
            raise ValueError("body ended inside a section payload")
        sections.append(body[off:off + n])
        off += n
    if off != len(body):
        raise ValueError(f"{len(body) - off} trailing bytes after last section")
    meta_raw, timing_raw, wav, luma, maps, maps_raw, rich_raw, webm = sections

    try:
        meta = json.loads(meta_raw.decode("utf-8"))
        timing = json.loads(timing_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"meta/timing is not UTF-8 JSON: {exc}") from exc
    if not isinstance(meta, dict) or not isinstance(timing, dict):
        raise ValueError("meta and timing must be JSON objects")
    try:
        rich = json.loads(rich_raw.decode("utf-8")) if rich_raw else None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"rich sidecar is not UTF-8 JSON: {exc}") from exc
    if rich is not None and not isinstance(rich, dict):
        raise ValueError("rich sidecar must be a JSON object")

    missing = [k for k in META_KEYS_FROM_CLIENT if k not in meta]
    if missing:
        raise ValueError(f"meta is missing the script's keys: {missing}")
    missing = [k for k in TIMING_KEYS if k not in timing]
    if missing:
        raise ValueError(f"timing is missing the script's keys: {missing}")

    name = meta["condition"]
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError(f"bad condition name {name!r}")
    if name not in PLAN_NAMES:
        raise ValueError(f"condition {name!r} is not in the plan; "
                         f"known: {sorted(PLAN_NAMES)}")
    kind = meta["kind"]
    speak = meta["speak"]
    if kind == "condition" and not isinstance(speak, bool):
        raise ValueError("kind 'condition' requires speak true/false")
    if kind == "free" and speak is not None:
        raise ValueError("kind 'free' requires speak null")
    if kind not in ("condition", "free"):
        raise ValueError(f"unknown kind {kind!r}")

    for key in ("n_frames",):
        if not isinstance(meta[key], int) or isinstance(meta[key], bool):
            raise ValueError(f"meta.{key} must be an integer")
    n = meta["n_frames"]
    if n < 10:  # the script skips takes under 10 frames
        raise ValueError(f"n_frames {n} < 10; the script would skip this take")
    if timing["condition"] != name or timing["n_frames"] != n:
        raise ValueError("timing disagrees with meta (condition/n_frames)")
    if meta["map_shape"] != [MAP_H, MAP_W]:
        raise ValueError(f"map_shape {meta['map_shape']!r} != {[MAP_H, MAP_W]}")
    if meta["filter"] != "row" or meta["arrays"] != ["maps", "maps_raw",
                                                     "luma"]:
        raise ValueError("meta filter/arrays do not match the take format")

    if len(luma) != n * LUMA_BYTES:
        raise ValueError(f"luma is {len(luma)} bytes; n_frames {n} needs "
                         f"{n * LUMA_BYTES} (160x96 per frame)")
    want_maps = (n - 1) * MAP_BYTES
    if len(maps) != want_maps or len(maps_raw) != want_maps:
        raise ValueError(f"map arrays must be {want_maps} bytes each "
                         f"(n_frames-1 x 48 x 80)")

    for key in ("map_scale", "map_scale_raw"):
        v = meta[key]
        if not isinstance(v, (int, float)) or isinstance(v, bool) \
                or not v > 0:
            raise ValueError(f"meta.{key} must be a positive number")
    for key in ("drift_raw", "drift_after_row_removal"):
        v = meta[key]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
            raise ValueError(f"meta.{key} must be a number >= 0")

    # The WAV must be an honest 16-bit mono file at the rate the timing
    # claims -- the defect the whole timing.json design exists to prevent.
    try:
        with wave.open(io.BytesIO(wav), "rb") as wv:
            wav_info = (wv.getnchannels(), wv.getsampwidth(),
                        wv.getframerate(), wv.getnframes())
    except wave.Error as exc:
        raise ValueError(f"WAV does not parse: {exc}") from exc
    if wav_info[0] != 1 or wav_info[1] != 2:
        raise ValueError(f"WAV must be mono 16-bit, got ch={wav_info[0]} "
                         f"width={wav_info[1] * 8}bit")
    got_rate = timing["audio_sample_rate"]
    if not isinstance(got_rate, int) or got_rate <= 0:
        raise ValueError("timing.audio_sample_rate must be a positive int")
    if wav_info[2] != got_rate:
        raise ValueError(f"WAV header says {wav_info[2]} Hz but timing says "
                         f"{got_rate} Hz -- the header must be honest")
    n_samples = wav_info[3]
    if n_samples <= 0:
        raise ValueError("WAV holds no samples")
    audio_span = n_samples / got_rate
    if abs(audio_span - timing["audio_span_s"]) > 0.01:
        raise ValueError(f"timing.audio_span_s {timing['audio_span_s']} "
                         f"disagrees with the WAV ({audio_span:.3f}s)")
    if meta["audio_sample_rate"] != got_rate:
        raise ValueError("meta.audio_sample_rate disagrees with timing")
    if abs(meta["audio_seconds"] - audio_span) > 0.01:
        raise ValueError("meta.audio_seconds disagrees with the WAV")
    if abs(meta["seconds"] - n / FPS) > 0.01:
        raise ValueError(f"meta.seconds {meta['seconds']} != n_frames/"
                         f"{FPS} = {n / FPS:.3f}")

    offsets = timing["frame_offsets_s"]
    if not isinstance(offsets, list) or len(offsets) != n:
        raise ValueError("frame_offsets_s must hold n_frames entries")
    achieved = timing["achieved_fps"]
    if not isinstance(achieved, (int, float)) or isinstance(achieved, bool) \
            or achieved <= 0:
        raise ValueError("timing.achieved_fps must be a positive number")
    if timing["assumed_fps"] != FPS:
        raise ValueError(f"timing.assumed_fps must be {FPS}")
    if timing["requested_sample_rate"] != RATE:
        raise ValueError(f"timing.requested_sample_rate must be {RATE}")
    if abs(timing["video_span_s"] - n / FPS) > 0.01:
        raise ValueError("timing.video_span_s != n_frames / assumed_fps")

    # The rich sidecar body (addendum). The server injects the stamp-dependent
    # references; the client sends everything it is the sole source of.
    if rich is not None:
        for key in ("condition", "instruction_text", "stimulus",
                    "wall_clock_start", "wall_clock_end", "notes"):
            if key not in rich:
                raise ValueError(f"rich sidecar is missing {key!r}")
        if rich["condition"] != name:
            raise ValueError("rich sidecar condition disagrees with meta")
        if not isinstance(rich["notes"], str) or len(rich["notes"]) > 4000:
            raise ValueError("rich notes must be a string up to 4000 chars")
    if webm and webm[:4] != b"\x1a\x45\xdf\xa3":
        raise ValueError("WebM section does not start with the EBML magic")

    return {"name": name, "meta": meta, "timing": timing, "wav": wav,
            "luma": luma, "maps": maps, "maps_raw": maps_raw,
            "rich": rich, "webm": webm, "n_samples": n_samples}


def unique_stamp(out: Path, name: str) -> int:
    """A wall-clock stamp no other take in the dir is using, like the
    script's -- plus one guarantee the script gets for free from its
    30-second takes: no two takes EVER share a stamp. The integrity gate
    identifies sessions BY TIMESTAMP, so a collision between two fast
    back-to-back takes would silently merge them into one session in the
    coverage count.
    """
    stamps = set()
    for p in out.glob("*.json"):
        m = re.search(r"_(\d+)(?:_timing)?\.json$", p.name)
        if m:
            stamps.add(int(m.group(1)))
    stamp = int(time.time())
    while stamp in stamps or any((out / f"{name}_{stamp}{suffix}").exists()
                                 for suffix in (".npz", ".wav", ".json",
                                                "_timing.json")):
        stamp += 1
    return stamp


def save_take(out: Path, take: dict) -> dict:
    """Write the take's files, exactly as the script names and formats them,
    plus the addendum's rich copies (out/rich/<stem>_rich.json, full webm).
    """
    name = take["name"]
    stamp = unique_stamp(out, name)
    meta = dict(take["meta"])
    meta["timestamp"] = stamp

    npz_path = out / f"{name}_{stamp}.npz"
    write_npz(npz_path, meta["n_frames"], take["maps"],
              float(meta["map_scale"]), take["maps_raw"],
              float(meta["map_scale_raw"]), take["luma"])

    wav_path = out / f"{name}_{stamp}.wav"
    wav_path.write_bytes(take["wav"])

    # The script writes timing compact and meta indented; match both.
    (out / f"{name}_{stamp}_timing.json").write_text(json.dumps(take["timing"]))
    (out / f"{name}_{stamp}.json").write_text(json.dumps(meta, indent=2))

    files = [p.name for p in
             (npz_path, wav_path, out / f"{name}_{stamp}.json",
              out / f"{name}_{stamp}_timing.json")]

    # -- the addendum's rich copies ------------------------------------
    webm_name = None
    if take["webm"]:
        webm_name = f"{name}_{stamp}_full.webm"
        (out / webm_name).write_bytes(take["webm"])
        files.append(webm_name)

    rich_name = None
    if take["rich"] is not None:
        rich_dir = out / "rich"
        rich_dir.mkdir(parents=True, exist_ok=True)
        rich = dict(take["rich"])
        rich["stamp"] = stamp
        rich["frame_timestamps_file"] = f"{name}_{stamp}_timing.json"
        rich["full_video"] = webm_name
        rich["pipeline_files"] = {
            "npz": f"{name}_{stamp}.npz",
            "wav": f"{name}_{stamp}.wav",
            "meta": f"{name}_{stamp}.json",
            "timing": rich["frame_timestamps_file"],
        }
        rich_name = f"{name}_{stamp}_rich.json"
        (rich_dir / rich_name).write_text(json.dumps(rich, indent=2))
        files.append(f"rich/{rich_name}")

    write_session(out)

    return {
        "stamp": stamp,
        "condition": name,
        "files": files,
        "rich": f"rich/{rich_name}" if rich_name else None,
        "full_video": webm_name,
        "npz_kib": npz_path.stat().st_size // 1024,
    }


# -- the FREE RECORD session (the front door's artifact) ---------------------

def parse_free_body(body: bytes) -> dict:
    """Split the FIVE length-prefixed sections of a free session: meta
    JSON, WAV bytes, luma bytes, timestamps JSON, WebM bytes.

    The checks here are about honest files, not the take schema -- a free
    session is a different artifact class (no conditions, no plan, no
    math pass; analysis happens offline), so what must hold is only that
    each file is what it claims to be.
    """
    sections, off = [], 0
    for _ in range(5):
        if off + 8 > len(body):
            raise ValueError("body ended inside a length prefix")
        (n,) = struct.unpack_from("<Q", body, off)
        off += 8
        if off + n > len(body):
            raise ValueError("body ended inside a section payload")
        sections.append(body[off:off + n])
        off += n
    if off != len(body):
        raise ValueError(f"{len(body) - off} trailing bytes after last section")
    meta_raw, wav, luma, ts_raw, webm = sections

    try:
        meta = json.loads(meta_raw.decode("utf-8"))
        ts = json.loads(ts_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"meta/timestamps is not UTF-8 JSON: {exc}") from exc
    if not isinstance(meta, dict) or not isinstance(ts, dict):
        raise ValueError("meta and timestamps must be JSON objects")
    if meta.get("tool") != "browser_free_record":
        raise ValueError(f"meta.tool {meta.get('tool')!r} is not this "
                         "endpoint's")
    for key in ("wall_clock_start", "wall_clock_end", "n_frames", "audio"):
        if key not in meta:
            raise ValueError(f"meta is missing {key!r}")
    n = meta["n_frames"]
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("meta.n_frames must be a non-negative integer")
    if len(luma) != n * FREE_LUMA_BYTES:
        raise ValueError(f"luma is {len(luma)} bytes; n_frames {n} needs "
                         f"{n * FREE_LUMA_BYTES} (160x96 per frame)")
    offs = ts.get("frame_offsets_ms")
    if not isinstance(offs, list) or len(offs) != n:
        raise ValueError("timestamps.frame_offsets_ms must hold n_frames "
                         "entries")
    audio = meta["audio"]
    if not isinstance(audio, dict):
        raise ValueError("meta.audio must be an object")
    rate = audio.get("sample_rate")
    if not isinstance(rate, int) or rate <= 0:
        raise ValueError("meta.audio.sample_rate must be a positive int")
    # the WAV must be an honest mono 16-bit file at the claimed rate
    try:
        with wave.open(io.BytesIO(wav), "rb") as wv:
            info = (wv.getnchannels(), wv.getsampwidth(), wv.getframerate(),
                    wv.getnframes())
    except wave.Error as exc:
        raise ValueError(f"WAV does not parse: {exc}") from exc
    if info[0] != 1 or info[1] != 2:
        raise ValueError(f"WAV must be mono 16-bit, got ch={info[0]} "
                         f"width={info[1] * 8}bit")
    if info[2] != rate:
        raise ValueError(f"WAV header says {info[2]} Hz but meta says "
                         f"{rate} Hz -- the header must be honest")
    if webm and webm[:4] != b"\x1a\x45\xdf\xa3":
        raise ValueError("WebM section does not start with the EBML magic")

    return {"meta": meta, "wav": wav, "luma": luma, "ts": ts, "webm": webm,
            "wav_frames": info[3]}


def save_free_session(out: Path, sess: dict) -> dict:
    """One folder, out/free_sessions/<stamp>/, with the five files. NOT
    the take layout, deliberately: no conditions, no plan, no math pass
    -- the browser recorded, analysis happens offline. The subfolder also
    keeps every json here away from the gate's out-root glob, the same
    reasoning that puts the rich copies in out/rich/.
    """
    root = out / "free_sessions"
    root.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    while (root / str(stamp)).exists():
        stamp += 1
    d = root / str(stamp)
    d.mkdir()
    files = ["meta.json", "audio.wav", "luma.raw", "timestamps.json"]
    (d / "meta.json").write_text(json.dumps(sess["meta"], indent=2))
    (d / "audio.wav").write_bytes(sess["wav"])
    (d / "luma.raw").write_bytes(sess["luma"])
    (d / "timestamps.json").write_text(json.dumps(sess["ts"]))
    if sess["webm"]:
        (d / "video.webm").write_bytes(sess["webm"])
        files.append("video.webm")
    meta = sess["meta"]
    rate = meta["audio"]["sample_rate"]
    print(f"[free] free_sessions/{stamp}: {meta['n_frames']} frames, "
          f"audio {sess['wav_frames'] / rate:.2f}s @ {rate} Hz"
          + (", webm" if sess["webm"] else ""), flush=True)

    # the liaison's self-check: MEASURE the WAV that just landed; a peak
    # of 0 is a SILENT_AUDIO alert the room files about itself
    wavp = d / "audio.wav"
    stats = wav_audio_stats(wavp) if wavp.is_file() else None
    liaison_event({"class": "session_saved",
                   "dir": f"free_sessions/{stamp}", "stamp": stamp,
                   "mode": meta.get("mode"), "trigger": meta.get("trigger"),
                   "close_reason": meta.get("close_reason"),
                   "n_frames": meta.get("n_frames"),
                   "duration_s": round(sess["wav_frames"] / rate, 2),
                   "audio_measured": stats,
                   "audio_silent": bool(stats and stats["peak_abs"] == 0)})
    if stats and stats["peak_abs"] == 0:
        liaison_event({"class": "alert", "kind": "SILENT_AUDIO",
                       "dir": f"free_sessions/{stamp}", "stamp": stamp,
                       "detail": "the saved WAV measures zero: peak_abs=0, "
                                 f"nonzero={stats['nonzero']} of "
                                 f"{stats['samples']} -- the input device "
                                 "delivered silence",
                       "audio_measured": stats})
    return {"stamp": stamp, "dir": f"free_sessions/{stamp}", "files": files}


# -- the LIVE MONITOR's data --------------------------------------------------

def live_state(out: Path) -> dict:
    """Everything /live polls, once a second. Read-only: the tail of the
    heartbeat feed, the latest thumbnail, the session folders on disk
    (newest first, with sizes), and any harvester verdicts -- a
    verdict.json next to a session's meta.json is the harvester's to
    write and this server's to surface, nothing more.
    """
    events = []
    try:
        with DIAG_PATH.open("rb") as f:
            size = f.seek(0, 2)
            f.seek(max(0, size - 131072))
            tail = f.read().decode("utf-8", "replace").splitlines()
        for ln in tail[-60:]:
            try:
                d = json.loads(ln)
                if isinstance(d, dict):
                    events.append(d)
            except json.JSONDecodeError:
                pass
    except OSError:
        pass
    events = events[-30:]

    segments = []
    root = out / "free_sessions"
    if root.is_dir():
        for d in sorted(root.iterdir(), key=lambda p: p.name,
                        reverse=True)[:50]:
            try:
                m = json.loads((d / "meta.json").read_text())
            except (OSError, json.JSONDecodeError):
                continue
            audio = m.get("audio") or {}
            entry = {
                "dir": f"free_sessions/{d.name}", "stamp": d.name,
                "mode": m.get("mode"), "trigger": m.get("trigger"),
                "close_reason": m.get("close_reason"),
                "n_frames": m.get("n_frames"),
                "duration_s": round(
                    audio.get("samples", 0) /
                    max(audio.get("sample_rate", 1), 1), 1),
                "bytes": {p.name: p.stat().st_size for p in d.iterdir()
                          if p.name in ("luma.raw", "audio.wav",
                                        "video.webm")},
            }
            v = d / "verdict.json"
            if v.is_file():
                try:
                    entry["verdict"] = json.loads(v.read_text())
                except (OSError, json.JSONDecodeError):
                    entry["verdict"] = {"error": "verdict.json is not JSON"}
            segments.append(entry)

    with THUMB_LOCK:
        thumb = dict(THUMB) if THUMB["data"] else None
    return {"ok": True,
            "now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "events": events, "thumb": thumb, "segments": segments}


# -- the flags, mirrored from the gate so session.json can carry them -------

def take_flags(meta: dict, timing: dict, has_npz: bool, has_wav: bool,
               has_timing: bool) -> list:
    """experiments/corpus_integrity.py's per-take verdicts, mirrored so the
    session manifest can tell an external tool which takes are trustworthy.
    Display/manifest only -- the gate itself remains the authority.
    """
    flags = []
    if not has_npz:
        flags.append("FILES_MISSING")
    if not has_timing:
        flags.append("NO_TIMING")
    speak = meta.get("speak")
    frac = meta.get("speech_frac")
    if isinstance(speak, bool):
        if not isinstance(frac, (int, float)):
            flags.append("SPEECH_UNMEASURED")
        elif speak and frac < 0.15:
            flags.append("VOICE_MISSING")
        elif not speak and frac > 0.05:
            flags.append("NOT_SILENT")
    achieved = timing.get("achieved_fps", meta.get("achieved_fps"))
    assumed = timing.get("assumed_fps", FPS)
    if isinstance(achieved, (int, float)) and not isinstance(achieved, bool) \
            and achieved > 0:
        if abs(achieved - assumed) / assumed > 0.05:
            flags.append("FPS_OFF")
    got = timing.get("audio_sample_rate", meta.get("audio_sample_rate"))
    want = timing.get("requested_sample_rate", RATE)
    if got is not None and got != want:
        flags.append("MIC_RATE")
    v = timing.get("video_span_s", meta.get("seconds"))
    a = timing.get("audio_span_s", meta.get("audio_seconds"))
    if isinstance(v, (int, float)) and isinstance(a, (int, float)) and v > 0:
        if abs(a - v) / v > 0.02:
            flags.append("SPAN_MISMATCH")
    drift = meta.get("drift_after_row_removal")
    if isinstance(drift, (int, float)) and not isinstance(drift, bool) \
            and drift >= 5.0:
        flags.append("DRIFT_HIGH")
    return flags


def write_session(out: Path) -> Path:
    """Regenerate rich/session.json: every take in the out dir, its files,
    labels, flags and notes -- one manifest an external tool can ingest
    without knowing this repo's layout. Lives in rich/ because a *.json at
    the out root would be misread as take metadata by the integrity gate.
    """
    rich_dir = out / "rich"
    rich_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for t in list_takes(out):
        stem = t["stem"]
        rich_path = rich_dir / f"{stem}_rich.json"
        rich = {}
        if rich_path.is_file():
            try:
                loaded = json.loads(rich_path.read_text())
                if isinstance(loaded, dict):
                    rich = loaded
            except json.JSONDecodeError:
                pass
        meta_path = out / f"{stem}.json"
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        timing = {}
        timing_path = out / f"{stem}_timing.json"
        if timing_path.is_file():
            try:
                timing = json.loads(timing_path.read_text()) or {}
            except json.JSONDecodeError:
                timing = {}
        webm = f"{stem}_full.webm"
        entries.append({
            "label": t["condition"],
            "stamp": t["timestamp"],
            "kind": t["kind"],
            "pose": t["pose"],
            "speak": t["speak"],
            "instruction_text": rich.get("instruction_text"),
            "stimulus": t["stimulus"],
            "flags": take_flags(meta, timing, t["has_npz"], t["has_wav"],
                                t["has_timing"]),
            "notes": rich.get("notes", ""),
            "files": {
                "npz": f"{stem}.npz" if t["has_npz"] else None,
                "wav": f"{stem}.wav" if t["has_wav"] else None,
                "timing": f"{stem}_timing.json" if t["has_timing"] else None,
                "rich": f"rich/{stem}_rich.json" if rich else None,
                "full_video": webm if (out / webm).is_file() else None,
            },
        })
    session = {
        "manifest": "session.json -- every take in this out dir, for external "
                    "ingest; see browser_capture/DONE.md for the format",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out": str(out),
        "n_takes": len(entries),
        "takes": entries,
    }
    path = rich_dir / "session.json"
    path.write_text(json.dumps(session, indent=2))
    return path


def list_takes(out: Path) -> list:
    """Every take in the out dir (script-made takes included), newest first.

    Returns exactly the fields the UI needs to mirror the integrity gate's
    per-take checks; nothing here judges pass/fail, it just reports.
    """
    takes = []
    for meta_path in sorted(out.glob("*.json")):
        if meta_path.name.endswith("_timing.json"):
            continue
        stem = meta_path.stem
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict) or "condition" not in meta:
            continue
        timing = {}
        timing_path = meta_path.with_name(f"{stem}_timing.json")
        if timing_path.is_file():
            try:
                t = json.loads(timing_path.read_text())
                if isinstance(t, dict):
                    timing = t
            except json.JSONDecodeError:
                pass
        npz_path = meta_path.with_suffix(".npz")
        wav_path = meta_path.with_suffix(".wav")
        takes.append({
            "stem": stem,
            "condition": meta.get("condition"),
            "kind": meta.get("kind"),
            "speak": meta.get("speak"),
            "pose": meta.get("pose"),
            "n_frames": meta.get("n_frames"),
            "seconds": meta.get("seconds"),
            "achieved_fps": timing.get("achieved_fps", meta.get("achieved_fps")),
            "assumed_fps": timing.get("assumed_fps", FPS),
            "audio_sample_rate": timing.get("audio_sample_rate",
                                            meta.get("audio_sample_rate")),
            "requested_sample_rate": timing.get("requested_sample_rate",
                                                RATE),
            "video_span_s": timing.get("video_span_s", meta.get("seconds")),
            "audio_span_s": timing.get("audio_span_s",
                                       meta.get("audio_seconds")),
            "speech_frac": meta.get("speech_frac"),
            "drift_after_row_removal": meta.get("drift_after_row_removal"),
            "timestamp": meta.get("timestamp"),
            "stimulus": meta.get("stimulus"),
            "has_timing": bool(timing),
            "has_npz": npz_path.is_file(),
            "has_wav": wav_path.is_file(),
            "mtime": meta_path.stat().st_mtime,
        })
    takes.sort(key=lambda t: (t["timestamp"] if isinstance(t["timestamp"], int)
                              else 0, t["mtime"]), reverse=True)
    return takes


# -- the HTTP half -----------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "browser_capture/1.0"
    protocol_version = "HTTP/1.1"

    # class-level wiring set by main()
    out_dir: Path = None
    index_path: Path = None
    free_path: Path = None
    live_path: Path = None
    evolve_path: Path = None
    probes_dir: Path = None

    def log_message(self, fmt, *args):  # keep the console signal-heavy
        if self.command in ("POST",) or " 40" in (fmt % args) \
                or " 50" in (fmt % args):
            sys.stderr.write("%s - %s\n" % (self.address_string(),
                                            fmt % args))

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"),
                   "application/json")

    def _handle_notes(self):
        """Update a take's rich-sidecar notes ("what I actually did")."""
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if n <= 0 or n > 65536:
            self._json(400, {"ok": False, "error": "bad Content-Length"})
            return
        try:
            req = json.loads(self.rfile.read(n).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._json(400, {"ok": False, "error": f"not JSON: {exc}"})
            return
        if not isinstance(req, dict):
            self._json(400, {"ok": False, "error": "expected a JSON object"})
            return
        cond, stamp, notes = req.get("condition"), req.get("stamp"), \
            req.get("notes", "")
        if not isinstance(cond, str) or not NAME_RE.match(cond) or \
                cond not in PLAN_NAMES:
            self._json(400, {"ok": False, "error": f"bad condition {cond!r}"})
            return
        if not isinstance(stamp, int) or isinstance(stamp, bool) or stamp < 0:
            self._json(400, {"ok": False, "error": "bad stamp"})
            return
        if not isinstance(notes, str) or len(notes) > 4000:
            self._json(400, {"ok": False,
                             "error": "notes must be a string up to 4000 chars"})
            return
        rich_path = self.out_dir / "rich" / f"{cond}_{stamp}_rich.json"
        if not rich_path.is_file():
            self._json(404, {"ok": False,
                             "error": f"no rich sidecar for {cond}_{stamp}"})
            return
        try:
            rich = json.loads(rich_path.read_text())
        except json.JSONDecodeError:
            self._json(500, {"ok": False, "error": "sidecar is not JSON"})
            return
        rich["notes"] = notes
        rich_path.write_text(json.dumps(rich, indent=2))
        write_session(self.out_dir)
        print(f"[notes] {cond}_{stamp}: {len(notes)} chars", flush=True)
        self._json(200, {"ok": True, "condition": cond, "stamp": stamp})

    def _handle_log(self):
        """Append one diagnostics line from a page as JSONL, stamped on
        receipt. Any JSON object is accepted: the heartbeat exists to be
        watched live (tail -f browser_capture/diag.log), not judged.
        """
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if n <= 0 or n > 65536:
            self._json(400, {"ok": False, "error": "bad Content-Length"})
            return
        try:
            evt = json.loads(self.rfile.read(n).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._json(400, {"ok": False, "error": f"not JSON: {exc}"})
            return
        if not isinstance(evt, dict):
            self._json(400, {"ok": False, "error": "expected a JSON object"})
            return
        now = time.time()
        line = {"received_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                             time.gmtime(now)) +
                f".{int((now % 1) * 1000):03d}Z"}
        line.update(evt)
        with DIAG_LOCK:
            with DIAG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
        # page events of liaison interest ride the outbox too
        if evt.get("kind") in LIAISON_FORWARD_KINDS:
            liaison_event({"class": "page_" + str(evt.get("kind")).lower(),
                           "source": "capture_page", "page": evt.get("page"),
                           **{k: v for k, v in evt.items()
                              if k not in ("kind", "page", "ts",
                                           "speech_level", "motion",
                                           "speech_thr", "motion_thr",
                                           "visibility", "floors_ready")}})
        self._json(200, {"ok": True})

    def _handle_thumb(self):
        """The capture page's latest thumbnail: {data: dataURL, ts: ISO}.
        In-memory only, latest-wins."""
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if n <= 0 or n > THUMB_MAX:
            self._json(400, {"ok": False, "error": "bad Content-Length"})
            return
        try:
            req = json.loads(self.rfile.read(n).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._json(400, {"ok": False, "error": f"not JSON: {exc}"})
            return
        if not isinstance(req, dict) or not isinstance(req.get("data"), str) \
                or not req["data"].startswith("data:image/"):
            self._json(400, {"ok": False, "error":
                             "expected {data: data:image/... , ts}"})
            return
        with THUMB_LOCK:
            THUMB["data"] = req["data"]
            THUMB["ts"] = req.get("ts") or time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._json(200, {"ok": True})

    def _handle_luma(self):
        """The page's newest live luma batch: [u64 n][n x u64 offset_ms]
        [n x 15360 luma bytes]. In-memory only, latest-wins."""
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if n <= 0 or n > LUMA_MAX:
            self._json(400, {"ok": False, "error": "bad Content-Length"})
            return
        body = self.rfile.read(n)
        if len(body) != n:
            self._json(400, {"ok": False, "error": "truncated body"})
            return
        if len(body) < 8:
            self._json(400, {"ok": False, "error": "no frame count"})
            return
        (count,) = struct.unpack_from("<Q", body, 0)
        if count > 10000 or len(body) < 8 + count * 8:
            self._json(400, {"ok": False, "error": "bad frame count"})
            return
        offsets = list(struct.unpack_from("<" + "Q" * count, body, 8)) \
            if count else []
        luma = body[8 + count * 8:]
        if len(luma) != count * FREE_LUMA_BYTES:
            self._json(400, {"ok": False,
                             "error": f"luma is {len(luma)} bytes; "
                                      f"{count} frames need "
                                      f"{count * FREE_LUMA_BYTES}"})
            return
        with LUMA_LOCK:
            LUMA_LATEST.update(
                {"n": count, "offsets": offsets, "luma": luma,
                 "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self._json(200, {"ok": True, "n": count})

    def _handle_free(self):
        """Write one FREE RECORD session folder (the front door's artifact)."""
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if n <= 0:
            self._json(400, {"ok": False, "error": "Content-Length required"})
            return
        if n > MAX_FREE_BODY:
            self._json(400, {"ok": False, "error":
                             f"body {n} > {MAX_FREE_BODY} bytes -- keep a "
                             "free session under ~35 minutes, or it cannot "
                             "be uploaded in one piece"})
            return
        body = self.rfile.read(n)
        if len(body) != n:
            self._json(400, {"ok": False, "error": "truncated body"})
            return
        try:
            sess = parse_free_body(body)
            result = save_free_session(self.out_dir, sess)
        except ValueError as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        except OSError as exc:
            self._json(500, {"ok": False, "error": f"write failed: {exc}"})
            return
        self._json(200, {"ok": True, **result})

    # -- the agent API: the liaison in the room -------------------------

    def _api_status(self):
        hb = _last_page_heartbeat()
        page = None
        if hb.get("present"):
            f = hb["fields"]
            page = {k: f.get(k) for k in (
                "state", "deaf", "seg_open", "seg_age_s", "n_frames",
                "n_samples", "speech_level", "speech_thr", "speech_now",
                "saved", "pending_uploads", "mic", "cam", "visibility")}
            page["heartbeat_age_s"] = hb["age_s"]
            page["received_at"] = f.get("received_at")
        n_sessions = 0
        root = self.out_dir / "free_sessions"
        if root.is_dir():
            n_sessions = len(list(root.iterdir()))
        with LIAISON_LOCKS["pending"]:
            pending_ship = (len(LIAISON_PENDING.read_text().splitlines())
                            if LIAISON_PENDING.is_file() else 0)
        self._json(200, {
            "ok": True,
            "now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "uptime_s": round(time.time() - LIAISON_STARTED_AT, 1),
            "out": str(self.out_dir),
            "port": self.server.server_address[1],
            "sessions_saved": n_sessions,
            "page": page,
            "rendezvous": {
                "target": f"{RENDEZVOUS['user']}@{RENDEZVOUS['host']}"
                          f":~/{RENDEZVOUS['remote']}",
                "pending_ship": pending_ship,
                "note": "ships are attempted per event and retried every "
                        "5 min; failures are honest, not silent",
            },
        })

    def _api_sessions(self):
        """Every free session, meta PLUS measured audio health."""
        segments = []
        root = self.out_dir / "free_sessions"
        if root.is_dir():
            for d in sorted(root.iterdir(), key=lambda p: p.name,
                            reverse=True):
                try:
                    meta = json.loads((d / "meta.json").read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                audio = meta.get("audio") or {}
                entry = {
                    "dir": f"free_sessions/{d.name}", "stamp": d.name,
                    "mode": meta.get("mode"), "trigger": meta.get("trigger"),
                    "close_reason": meta.get("close_reason"),
                    "n_frames": meta.get("n_frames"),
                    "duration_s": round(
                        audio.get("samples", 0) /
                        max(audio.get("sample_rate", 1), 1), 2),
                    "bytes": {p.name: p.stat().st_size for p in d.iterdir()
                              if p.name in ("luma.raw", "audio.wav",
                                            "video.webm")},
                }
                wavp = d / "audio.wav"
                if wavp.is_file():
                    entry["audio_measured"] = wav_audio_stats(wavp)
                    entry["audio_silent"] = \
                        entry["audio_measured"]["peak_abs"] == 0
                    entry["meta_asserted_audio_silent"] = \
                        audio.get("audio_silent")
                v = d / "verdict.json"
                if v.is_file():
                    try:
                        entry["verdict"] = json.loads(v.read_text())
                    except (OSError, json.JSONDecodeError):
                        pass
                segments.append(entry)
        self._json(200, {"ok": True, "n": len(segments),
                         "sessions": segments})

    def _api_inbox_get(self):
        lines = []
        if LIAISON_INBOX.is_file():
            with LIAISON_LOCKS["inbox"]:
                text = LIAISON_INBOX.read_text()
            for ln in text.splitlines()[-100:]:
                try:
                    lines.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
        self._json(200, {"ok": True, "n": len(lines), "messages": lines})

    def _api_inbox_post(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if n <= 0 or n > 65536:
            self._json(400, {"ok": False, "error": "bad Content-Length"})
            return
        try:
            req = json.loads(self.rfile.read(n).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._json(400, {"ok": False, "error": f"not JSON: {exc}"})
            return
        if not isinstance(req, dict) or \
                not isinstance(req.get("from"), str) or \
                not isinstance(req.get("text"), str) or \
                not req["from"].strip() or not req["text"].strip():
            self._json(400, {"ok": False,
                             "error": "expected {from: str, text: str}"})
            return
        if len(req["from"]) > 64 or len(req["text"]) > 8000:
            self._json(400, {"ok": False,
                             "error": "from <= 64 chars, text <= 8000"})
            return
        msg = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "from": req["from"], "text": req["text"]}
        with LIAISON_LOCKS["inbox"]:
            with LIAISON_INBOX.open("a", encoding="utf-8") as f:
                f.write(json.dumps(msg) + "\n")
        liaison_event({"class": "inbox_message", "from": req["from"],
                       "chars": len(req["text"])})
        self._json(200, {"ok": True, **msg})

    def do_GET(self):
        # match on the path portion only -- query strings (/ ?maxseg=8)
        # and trailing slashes (/evolve/) must not turn a route into a
        # 404 (the OVER-probe E2E caught the slash class)
        route = self.path.split("?", 1)[0]
        if len(route) > 1:
            route = route.rstrip("/")
        if route in ("/", "/free", "/free_record.html"):
            try:
                page = self.free_path.read_bytes()
            except OSError as exc:
                self._json(500, {"ok": False,
                                 "error": f"free_record.html: {exc}"})
                return
            self._send(200, page, "text/html; charset=utf-8")
        elif route in ("/plan", "/index.html"):
            try:
                page = self.index_path.read_bytes()
            except OSError as exc:
                self._json(500, {"ok": False, "error": f"index.html: {exc}"})
                return
            self._send(200, page, "text/html; charset=utf-8")
        elif route == "/info":
            self._json(200, {"ok": True, "out": str(self.out_dir),
                             "port": self.server.server_address[1]})
        elif route == "/takes":
            self._json(200, {"ok": True, "takes": list_takes(self.out_dir)})
        elif route == "/live":
            try:
                page = self.live_path.read_bytes()
            except OSError as exc:
                self._json(500, {"ok": False, "error": f"live.html: {exc}"})
                return
            self._send(200, page, "text/html; charset=utf-8")
        elif route == "/live-state":
            self._json(200, live_state(self.out_dir))
        elif route == "/evolve":
            try:
                page = self.evolve_path.read_bytes()
            except OSError as exc:
                self._json(500, {"ok": False,
                                 "error": f"evolve.html: {exc}"})
                return
            self._send(200, page, "text/html; charset=utf-8")
        elif route.startswith("/probes/"):
            name = route[len("/probes/"):]
            if "/" in name or ".." in name or not re.fullmatch(
                    r"[A-Za-z0-9._-]+", name):
                self._json(404, {"ok": False, "error": "no such probe"})
                return
            p = self.probes_dir / name
            try:
                body = p.read_bytes()
            except OSError:
                self._json(404, {"ok": False, "error": f"no {name!r}"})
                return
            ctype = "application/javascript" if name.endswith(".js") \
                else "application/json"
            self._send(200, body, ctype + "; charset=utf-8")
        elif route == "/luma-latest":
            with LUMA_LOCK:
                batch = dict(LUMA_LATEST)
            import base64
            self._json(200, {"ok": True, "n": batch["n"],
                             "offsets": batch["offsets"],
                             "ts": batch["ts"],
                             "luma_b64": base64.b64encode(
                                 batch["luma"]).decode("ascii")})
        elif route == "/api/status":
            self._api_status()
        elif route == "/api/sessions":
            self._api_sessions()
        elif route == "/api/inbox":
            self._api_inbox_get()
        else:
            self._json(404, {"ok": False, "error": f"no {self.path!r}"})

    def do_POST(self):
        route = self.path.split("?", 1)[0]   # query strings are not routing
        if len(route) > 1:
            route = route.rstrip("/")
        if route == "/notes":
            self._handle_notes()
            return
        if route == "/log":
            self._handle_log()
            return
        if route == "/free":
            self._handle_free()
            return
        if route == "/thumb":
            self._handle_thumb()
            return
        if route == "/luma":
            self._handle_luma()
            return
        if route == "/api/inbox":
            self._api_inbox_post()
            return
        if route != "/take":
            self._json(404, {"ok": False, "error": f"no {self.path!r}"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if n <= 0:
            self._json(400, {"ok": False, "error": "Content-Length required"})
            return
        if n > MAX_BODY:
            self._json(400, {"ok": False,
                             "error": f"body {n} > {MAX_BODY} bytes"})
            return
        body = self.rfile.read(n)
        if len(body) != n:
            self._json(400, {"ok": False, "error": "truncated body"})
            return
        try:
            take = parse_take_body(body)
            result = save_take(self.out_dir, take)
        except ValueError as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        except OSError as exc:
            self._json(500, {"ok": False, "error": f"write failed: {exc}"})
            return

        # Echo the script's console block, so the operator watching the
        # terminal sees the same numbers the browser UI shows.
        meta, timing = take["meta"], take["timing"]
        print(f"[take] {result['condition']}_{result['stamp']}: "
              f"{meta['n_frames']} frames, "
              f"video {timing['video_span_s']:.2f}s, "
              f"audio {timing['audio_span_s']:.2f}s, "
              f"achieved {timing['achieved_fps']:.2f} fps "
              f"(assumed {FPS}), mic {timing['audio_sample_rate']} Hz "
              f"(requested {RATE}), "
              f"drift raw {meta['drift_raw']:.2f} / "
              f"row-removed {meta['drift_after_row_removal']:.2f}, "
              f"speech {meta['speech_frac']}"
              f" -> {result['npz_kib']} kB npz", flush=True)
        self._json(200, {"ok": True, **result})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/mnt/c/proart_capture/corpus",
                    help="take output directory (the script's --out)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="localhost only, per the brief")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Handler.out_dir = out
    Handler.index_path = Path(__file__).resolve().parent / "index.html"
    Handler.free_path = Path(__file__).resolve().parent / "free_record.html"
    Handler.live_path = Path(__file__).resolve().parent / "live.html"
    Handler.evolve_path = Path(__file__).resolve().parent / "evolve.html"
    Handler.probes_dir = Path(__file__).resolve().parent / "probes"

    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    httpd.daemon_threads = True

    # the liaison: shipper thread + a start event; ship failures are
    # reported, never fatal
    import queue as queue_mod
    global LIAISON_SHIP_Q
    LIAISON_SHIP_Q = queue_mod.Queue()
    t = threading.Thread(target=_liaison_shipper,
                         args=(LIAISON_SHIP_Q,), daemon=True)
    t.start()
    liaison_event({"class": "liaison_started", "port": a.port,
                   "out": str(out), "pid": os.getpid()})
    print(f"browser_capture serving on http://{a.host}:{a.port}")
    print(f"takes will be written to   {out}")
    print("open the URL in the browser, allow camera + microphone, "
          "and click START PLAN. Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
