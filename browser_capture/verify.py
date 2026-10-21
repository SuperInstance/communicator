"""
verify.py -- prove the browser-capture POST path writes takes the existing
pipeline accepts, without needing a camera.

WHAT THIS DOES (docs/research/BRIEF-BROWSER-CAPTURE.md, "Verify before you
call it done"):

  1. Starts server.py for real, on a scratch port and a TEMP out dir. The
     real corpus (/mnt/c/proart_capture/corpus) is never touched.
  2. Generates synthetic takes: deterministic pseudo-random 160x96 luma
     frames with real frame-to-frame motion, a synthetic 16 kHz WAV, honest
     timing/meta/rich JSON, and a minimal (header-only) WebM. The change
     maps, scales and drift values are computed by the ACTUAL JAVASCRIPT
     shipped in index.html -- the MATH block is extracted from the file the
     server serves and run under node -- so the POST body is byte-for-byte
     the shape the browser will send.
  3. POSTs the takes through the real HTTP path and checks everything
     stdlib can check: file names, WAV sizes, JSON schemas (the script's
     exact key sets), the hand-rolled NPY members inside the npz, /takes,
     /notes, session.json, and the duplicate-POST stamp bump.
  4. If numpy is importable (e.g. the audio venv), additionally proves the
     round-trip the brief demands: np.load reads the npz with the right
     dtypes/shapes; the map arrays match an INDEPENDENT numpy
     re-implementation of corpus_capture_win.py's math (maps_raw
     byte-identical, maps within quantisation tolerance); src/nv12.py
     detects the luma stream as luma_only with n frames at the 15,360-byte
     stride; src/wavio.py reads the WAV at its stated rate; and -- the
     binding check -- experiments/corpus_integrity.py (THE GATE) exits 0 on
     the synthetic corpus with all the addendum's rich artifacts present.

USAGE, from the repo root:

    python3 browser_capture/verify.py                     # stdlib checks
    /home/proart/.local/audio-venv/bin/python \\
        browser_capture/verify.py                        # + numpy + gate

Requires node on PATH for the take generation (the shipped JS is the
generator, not a Python re-implementation). Exit 0 = everything that ran
passed; failures and skips are itemised.
"""

from __future__ import annotations

import json
import math
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from io import BytesIO
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HERE = REPO / "browser_capture"
PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"

W, H = 160, 96
FPS, RATE = 30, 16000
LUMA_BYTES = W * H

# The script's exact key sets (corpus_capture_win.py), for schema equality.
TIMING_KEYS = [
    "condition", "n_frames", "frame_offsets_s", "assumed_fps", "achieved_fps",
    "video_span_s", "audio_span_s", "audio_sample_rate",
    "requested_sample_rate", "mic", "capture_wallclock_s",
]
META_KEYS = [
    "condition", "kind", "note", "pose", "speak", "stimulus",
    "stimulus_title", "stimulus_estimated_s", "speech_frac", "n_frames",
    "seconds", "achieved_fps", "audio_sample_rate", "audio_seconds",
    "map_shape", "map_scale", "map_scale_raw", "drift_raw",
    "drift_after_row_removal", "timestamp", "rep", "filter", "filters",
    "arrays",
]

# A syntactically valid EBML/WebM header (empty body): enough for the
# server's magic check and honest about being synthetic, not playable.
WEBM_HEADER = (b"\x1a\x45\xdf\xa3\x40\x1f"
               b"\x42\x86\x81\x01" b"\x42\xf7\x81\x01" b"\x42\xf2\x81\x04"
               b"\x42\xf3\x81\x08" b"\x42\x82\x84webm"
               b"\x42\x87\x81\x02" b"\x42\x85\x81\x02")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" +
          (f"  -- {detail}" if detail and not ok else ""))
    return bool(ok)


def skip(name, why):
    RESULTS.append((name, None, why))
    print(f"  [SKIP] {name}  -- {why}")


# -- synthetic take generation ------------------------------------------------

def synth_luma(n: int, seed: int) -> bytes:
    """n frames of 160x96 luma: a random base, small per-pixel motion, and a
    bright block that moves -- so consecutive frames are similar (low drift,
    a real video's property) while a mis-strided read of the byte stream
    mixes unrelated frames (src/nv12.py's detection statistic)."""
    rng = random.Random(seed)
    base = [rng.randrange(256) for _ in range(LUMA_BYTES)]
    out = bytearray(n * LUMA_BYTES)
    for t in range(n):
        off = t * LUMA_BYTES
        for i in range(LUMA_BYTES):
            v = base[i] + rng.gauss(0, 1.5)
            out[off + i] = 0 if v < 0 else 255 if v > 255 else int(v + 0.5)
        bx = (t * 7) % (W - 20)
        for y in range(30, 50):
            row = off + y * W
            for x in range(bx, bx + 20):
                out[row + x] = 210
    return bytes(out)


def synth_wav(n_samples: int, rate: int = RATE) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        frames = bytearray()
        for i in range(n_samples):
            # bursts of tone separated by quiet, like speech pauses
            env = 0.55 if (i // rate) % 2 == 0 else 0.03
            v = int(math.sin(2 * math.pi * 220 * i / rate) * env * 20000)
            frames += struct.pack("<h", v)
        f.writeframes(bytes(frames))
    return buf.getvalue()


def js_maps_for(luma_path: Path, out_dir: Path) -> dict:
    """Run the SHIPPED math block from index.html under node on this luma."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node not on PATH")
    harness = out_dir / "harness.js"
    harness.write_text(
        "const fs = require('fs');\n"
        "const html = fs.readFileSync(process.argv[2], 'utf8');\n"
        "const math = html.split('// === MATH-BEGIN ===')[1]"
        ".split('// === MATH-END ===')[0];\n"
        "const computeTake = new Function(math + '\\nreturn computeTake;')();\n"
        "const luma = new Uint8Array(fs.readFileSync(process.argv[3]));\n"
        "const n = luma.length / (160 * 96);\n"
        "const r = computeTake(luma, n);\n"
        "fs.writeFileSync(process.argv[4], Buffer.from(r.maps));\n"
        "fs.writeFileSync(process.argv[5], Buffer.from(r.mapsRaw));\n"
        "fs.writeFileSync(process.argv[6], JSON.stringify({\n"
        "  scale: r.scale, scaleRaw: r.scaleRaw,\n"
        "  driftRaw: r.driftRaw, driftRow: r.driftRow, n: n}));\n")
    maps_bin = out_dir / "js_maps.bin"
    maps_raw_bin = out_dir / "js_maps_raw.bin"
    stats = out_dir / "js_stats.json"
    subprocess.run([node, str(harness), str(HERE / "index.html"),
                    str(luma_path), str(maps_bin), str(maps_raw_bin),
                    str(stats)], check=True, capture_output=True, timeout=300)
    r = json.loads(stats.read_text())
    r["maps"] = maps_bin.read_bytes()
    r["maps_raw"] = maps_raw_bin.read_bytes()
    return r


def build_take(name, kind, speak, pose, note, rep, n, seed, speech_frac,
               with_webm, js, rate=RATE):
    blk = rate // FPS
    n_samples = n * blk
    meta = {
        "condition": name, "kind": kind, "note": note, "pose": pose,
        "speak": speak, "stimulus": "invictus", "stimulus_title": "Invictus",
        "stimulus_estimated_s": 40.0, "speech_frac": speech_frac,
        "n_frames": n, "seconds": n / FPS, "achieved_fps": 30.0,
        "audio_sample_rate": rate,
        "audio_seconds": round(n_samples / rate, 3),
        "map_shape": [48, 80], "map_scale": js["scale"],
        "map_scale_raw": js["scaleRaw"], "drift_raw": js["driftRaw"],
        "drift_after_row_removal": js["driftRow"], "rep": rep,
        "filter": "row", "filters": {"maps": "row", "maps_raw": "none",
                                     "luma": "none"},
        "arrays": ["maps", "maps_raw", "luma"],
        "capture_tool": "browser_capture",
        "speech_gate": "browser_rms_adaptive",
    }
    timing = {
        "condition": name, "n_frames": n,
        "frame_offsets_s": [round(i / FPS, 6) for i in range(n)],
        "assumed_fps": FPS, "achieved_fps": 30.0,
        "video_span_s": round(n / FPS, 3),
        "audio_span_s": round(n_samples / rate, 3),
        "audio_sample_rate": rate, "requested_sample_rate": rate,
        "mic": "verify-synth-mic", "capture_wallclock_s": 0.25,
    }
    rich = {
        "condition": name, "kind": kind, "pose": pose, "speak": speak,
        "instruction_text": "SYNTHETIC VERIFY TAKE",
        "instruction_note": note, "stimulus": "invictus",
        "stimulus_title": "Invictus",
        "wall_clock_start": "2026-10-20T12:00:00.000Z",
        "wall_clock_end": "2026-10-20T12:00:05.000Z",
        "achieved_fps": 30.0, "audio_sample_rate": rate,
        "requested_sample_rate": rate, "mic_device": "verify-synth-mic",
        "camera_device": "verify-synth-cam", "browser": "verify.py",
        "frame_count": n, "video_span_s": timing["video_span_s"],
        "audio_span_s": timing["audio_span_s"],
        "full_video_recorded": with_webm, "notes": "",
    }
    return {
        "meta": meta, "timing": timing, "rich": rich,
        "wav": synth_wav(n_samples, rate), "luma": js["luma"],
        "maps": js["maps"], "maps_raw": js["maps_raw"],
        "webm": WEBM_HEADER if with_webm else b"",
        "n_samples": n_samples, "js": js,
    }


def frame_body(t: dict) -> bytes:
    enc = lambda obj: json.dumps(obj).encode("utf-8")
    body = b""
    for part in (enc(t["meta"]), enc(t["timing"]), t["wav"], t["luma"],
                 t["maps"], t["maps_raw"], enc(t["rich"]), t["webm"]):
        body += struct.pack("<Q", len(part)) + part
    return body


# -- HTTP helpers --------------------------------------------------------------

def http(method, path, body=None, ctype="application/octet-stream"):
    req = urllib.request.Request(BASE + path, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        # 4xx is a legitimate answer here (the rejection tests want it)
        return e.code, e.read()


# -- stdlib NPY inspection ------------------------------------------------------

def read_npy_member(data: bytes):
    """Parse one .npy: (descr, shape, fortran_order, payload_bytes)."""
    if data[:6] != b"\x93NUMPY":
        raise ValueError("bad magic")
    ver = data[6:8]
    (hlen,) = struct.unpack("<H" if ver[0] == 1 else "<I", data[8:8 + (2 if ver[0] == 1 else 4)])
    import ast
    off = 10 if ver[0] == 1 else 12
    header = ast.literal_eval(data[off:off + hlen].decode("ascii").strip())
    return header["descr"], tuple(header["shape"]), \
        header["fortran_order"], data[off + hlen:]


# -- the numpy reference (the script's math, verbatim) --------------------------

def numpy_reference(luma: bytes, n: int):
    import numpy as np
    Y = np.frombuffer(luma, np.uint8).reshape(n, H, W)
    Yf = Y.astype(np.float64)
    prof = np.median(Yf, axis=2)
    g = prof.mean(axis=1, keepdims=True)
    Yr = Yf - (prof - g)[:, :, None]

    def change_maps(ys):
        d = np.abs(np.diff(ys, axis=0))
        return d.reshape(d.shape[0], H // 2, 2, W // 2, 2).mean(axis=(2, 4))

    def quantise(maps):
        hi = float(np.percentile(maps, 99.5)) if maps.size else 1.0
        hi = max(hi, 1e-6)
        return np.clip(maps / hi * 255.0, 0, 255).astype(np.uint8), hi

    maps, hi = quantise(change_maps(Yr))
    maps_raw, hi_raw = quantise(change_maps(Yf))
    drift_raw = float(np.abs(np.diff(Yf, axis=0)).mean())
    drift_row = float(np.abs(np.diff(Yr, axis=0)).mean())
    return {"maps": maps, "scale": hi, "maps_raw": maps_raw,
            "scale_raw": hi_raw, "drift_raw": drift_raw,
            "drift_row": drift_row, "Y": Y}


def compare_maps(label, ref_q, got_bytes, hi_ref, hi_got, exact):
    """Compare the stored uint8 maps against the numpy reference.
    maps_raw must be byte-identical (integer arithmetic all the way);
    maps may differ by 1 quantisation count on a small fraction of cells
    (row medians are float; summation order differs between JS and numpy).
    The npz stores the scale as float32 -- the script's own choice
    (np.float32(scale)) -- so the round-trip is exact AT float32 precision,
    and the float64 the client computed must equal numpy's percentile."""
    import numpy as np
    got = np.frombuffer(got_bytes, np.uint8)
    if got.shape != ref_q.reshape(-1).shape:
        return False, f"shape {got.shape} != {ref_q.size}"
    if exact:
        bad = int((got != ref_q.reshape(-1)).sum())
        return bad == 0, f"{bad} bytes differ (expected byte-identical)"
    diff = np.abs(got.astype(np.int32) - ref_q.reshape(-1).astype(np.int32))
    nbad = int((diff > 1).sum())
    frac = float((diff > 0).mean())
    scale_ok = hi_got == float(np.float32(hi_ref))  # exact float32 round-trip
    ok = nbad == 0 and frac <= 0.005 and scale_ok
    return ok, (f"{nbad} cells differ by >1, {frac * 100:.3f}% differ at all, "
                f"npz scale {hi_got!r} vs float32({hi_ref!r}) = "
                f"{float(np.float32(hi_ref))!r}")


# -- main ------------------------------------------------------------------------

def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="bc_verify_"))
    out_dir = tmp / "corpus"
    work = tmp / "work"
    work.mkdir()
    server_log = (tmp / "server.log").open("wb")
    failures = 0
    try:
        proc = subprocess.Popen(
            [sys.executable, str(HERE / "server.py"),
             "--out", str(out_dir), "--port", str(PORT)],
            stdout=server_log, stderr=server_log)
        up = False
        for _ in range(50):
            try:
                http("GET", "/info")
                up = True
                break
            except Exception:
                time.sleep(0.2)
        print(f"server on :{PORT} up={up}  out={out_dir}")

        print("\n== GET /, /plan and /info ==")
        status, page = http("GET", "/")
        check("GET / returns 200", status == 200)
        page_s = page.decode("utf-8")
        check("GET / is the repo's free_record.html (the front door)",
              page == (HERE / "free_record.html").read_bytes())
        check("free_record.html has DOCTYPE", page_s.lstrip().lower()
              .startswith("<!doctype html>"))
        ext = re.findall(r"https?://[^\s'\"<>]+", page_s)
        check("free_record.html references no external URLs", not ext,
              str(ext[:3]))
        status, plan = http("GET", "/plan")
        check("GET /plan is the repo's index.html (plan mode intact)",
              status == 200 and
              plan == (HERE / "index.html").read_bytes())
        status, qpage = http("GET", "/?tuned=maxseg")
        check("a query string does not break routing (the 404 class)",
              status == 200 and
              qpage == (HERE / "free_record.html").read_bytes())
        status, body = http("GET", "/evolve")
        check("GET /evolve is the repo's evolve.html (the lab shell)",
              status == 200 and
              body == (HERE / "evolve.html").read_bytes())
        status, body = http("GET", "/probes/manifest.json")
        man = json.loads(body)
        check("/probes/manifest.json lists sonar-band and audio-health",
              status == 200 and
              {p.get("name") for p in man.get("probes", [])} >=
              {"sonar-band", "audio-health"})
        status, body = http("GET", "/probes/sonar-band.js")
        check("probe modules are served from disk",
              status == 200 and b"sonar-band" in body)
        status, body = http("GET", "/probes/../server.py")
        check("probe path traversal is refused",
              status == 404 and not json.loads(body).get("ok"))
        status, info = http("GET", "/info")
        check("GET /info", status == 200 and
              json.loads(info)["out"] == str(out_dir))

        print("\n== generating synthetic takes (shipped JS under node) ==")
        specs = [
            ("A1", "condition", True, "facing",
             "facing the camera, reciting the poem from memory", 1, 0.72, True),
            ("C1", "condition", False, "facing",
             "facing the camera, completely silent", 1, 0.01, True),
            ("D1", "condition", False, "turned",
             "head turned fully away, completely silent", 1, 0.02, True),
            ("F1", "free", None, "any",
             "free session: just be in the room, talk, read, move", 1,
             None, False),
        ]
        takes = {}
        for i, (name, kind, speak, pose, note, rep, frac, webm) in \
                enumerate(specs):
            n = 150
            luma_path = work / f"luma_{name}.bin"
            luma_path.write_bytes(synth_luma(n, seed=1000 + i))
            js = js_maps_for(luma_path, work)
            js["luma"] = luma_path.read_bytes()
            takes[name] = build_take(name, kind, speak, pose, note, rep, n,
                                     1000 + i, frac, webm, js)
            print(f"  generated {name}: {n} frames, "
                  f"drift {js['driftRow']:.3f}, scale {js['scale']:.3f}")

        print("\n== POST /take (the real HTTP path) ==")
        stamps = {}
        for name, t in takes.items():
            status, body = http("POST", "/take", frame_body(t))
            j = json.loads(body)
            check(f"POST {name} accepted", status == 200 and j.get("ok"),
                  j.get("error", ""))
            stamps[name] = j.get("stamp")
        check("stamps are unique per take",
              len(set(stamps.values())) == len(stamps))

        print("\n== written files: names, sizes, JSON schema ==")
        for name, t in takes.items():
            stamp = stamps[name]
            npz = out_dir / f"{name}_{stamp}.npz"
            wav = out_dir / f"{name}_{stamp}.wav"
            meta_p = out_dir / f"{name}_{stamp}.json"
            timing_p = out_dir / f"{name}_{stamp}_timing.json"
            check(f"{name}: four pipeline files exist",
                  all(p.is_file() for p in (npz, wav, meta_p, timing_p)))
            n, ns = t["meta"]["n_frames"], t["n_samples"]
            check(f"{name}: wav size == 44 + samples*2",
                  wav.stat().st_size == 44 + ns * 2,
                  f"{wav.stat().st_size} vs {44 + ns * 2}")
            check(f"{name}: luma in npz source == frames x stride",
                  len(t["luma"]) == n * LUMA_BYTES)
            meta = json.loads(meta_p.read_text())
            # The script's keys must all be present. Two EXTRA provenance
            # keys are deliberate (capture_tool, speech_gate) -- the gate
            # ignores keys it does not read, and an honest record of what
            # produced the take is worth more than a bit-exact key set.
            check(f"{name}: meta carries the script's exact key set "
                  "(+ documented provenance extras)",
                  set(META_KEYS) <= set(meta) and
                  set(meta) - set(META_KEYS) <=
                  {"capture_tool", "speech_gate"},
                  f"extra={set(meta) - set(META_KEYS)} "
                  f"missing={set(META_KEYS) - set(meta)}")
            check(f"{name}: meta.timestamp == filename stamp",
                  meta["timestamp"] == stamp)
            check(f"{name}: meta values survived the round trip",
                  meta["speech_frac"] == t["meta"]["speech_frac"] and
                  meta["map_scale"] == t["meta"]["map_scale"] and
                  meta["drift_after_row_removal"] ==
                  t["meta"]["drift_after_row_removal"])
            timing = json.loads(timing_p.read_text())
            check(f"{name}: timing has the script's exact key set",
                  set(timing) == set(TIMING_KEYS),
                  f"{set(timing) ^ set(TIMING_KEYS)}")
            check(f"{name}: timing.json is compact (script style)",
                  "\n" not in timing_p.read_text())
            check(f"{name}: meta.json is indented (script style)",
                  meta_p.read_text().startswith('{\n  "'))
            check(f"{name}: frame_offsets_s has n_frames entries",
                  len(timing["frame_offsets_s"]) == n and
                  timing["frame_offsets_s"][0] == 0)

        print("\n== the npz, member by member (stdlib NPY parse) ==")
        import zipfile
        for name, t in takes.items():
            stamp = stamps[name]
            with zipfile.ZipFile(out_dir / f"{name}_{stamp}.npz") as z:
                names = z.namelist()
                check(f"{name}: npz members exact",
                      names == ["maps.npy", "scale.npy", "maps_raw.npy",
                                "scale_raw.npy", "luma.npy"], str(names))
                n = t["meta"]["n_frames"]
                want = {
                    "maps.npy": ("|u1", (n - 1, 48, 80), n != 0),
                    "maps_raw.npy": ("|u1", (n - 1, 48, 80), True),
                    "luma.npy": ("|u1", (n, 96, 160), True),
                }
                for member, (descr, shape, _) in want.items():
                    d, hdr, fortran, payload = read_npy_member(
                        z.read(member))
                    okh = d == descr and hdr == shape and not fortran
                    size = 1
                    for dim in shape:
                        size *= dim
                    check(f"{name}: {member} header+size",
                          okh and len(payload) == size,
                          f"descr={d} shape={hdr} len={len(payload)}")
                for member in ("scale.npy", "scale_raw.npy"):
                    d, hdr, fortran, payload = read_npy_member(
                        z.read(member))
                    check(f"{name}: {member} is a float32 scalar",
                          d == "<f4" and hdr == () and len(payload) == 4)

        print("\n== rich artifacts (addendum) ==")
        sess = out_dir / "rich" / "session.json"
        check("rich/session.json exists", sess.is_file())
        session = json.loads(sess.read_text())
        check("session.json lists every take",
              session["n_takes"] == len(takes) and
              {t["label"] for t in session["takes"]} == set(takes))
        a1_entry = next(t for t in session["takes"] if t["label"] == "A1")
        check("session A1 entry has files+flags+empty notes",
              a1_entry["files"]["npz"] and a1_entry["files"]["wav"] and
              a1_entry["files"]["rich"] and a1_entry["files"]["timing"] and
              a1_entry["flags"] == [] and a1_entry["notes"] == "")
        a1_stamp = stamps["A1"]
        a1_rich = out_dir / "rich" / f"A1_{a1_stamp}_rich.json"
        rich = json.loads(a1_rich.read_text())
        check("A1 rich sidecar carries the addendum's fields",
              rich["instruction_text"] == "SYNTHETIC VERIFY TAKE" and
              rich["wall_clock_start"].endswith("Z") and
              rich["full_video"] == f"A1_{a1_stamp}_full.webm" and
              rich["frame_timestamps_file"] == f"A1_{a1_stamp}_timing.json")
        check("A1 full.webm sits next to the take",
              (out_dir / f"A1_{a1_stamp}_full.webm").read_bytes()[:4] ==
              b"\x1a\x45\xdf\xa3")
        f1_stamp = stamps["F1"]
        check("F1 (no webm posted) has no _full.webm",
              not (out_dir / f"F1_{f1_stamp}_full.webm").exists())
        f1_rich = json.loads(
            (out_dir / "rich" / f"F1_{f1_stamp}_rich.json").read_text())
        check("F1 rich sidecar records full_video: null", f1_rich["full_video"]
              is None)

        root_jsons = sorted(p.name for p in out_dir.glob("*.json"))
        check("out root holds ONLY meta+timing JSON (gate safety)",
              len(root_jsons) == 2 * len(takes) and
              all(re.match(r"^[A-Z]+\d*_\d+(_timing)?\.json$", n)
                  for n in root_jsons), str(root_jsons))

        print("\n== /notes ==")
        status, body = http("POST", "/notes",
                            json.dumps({"condition": "A1",
                                        "stamp": a1_stamp,
                                        "notes": "turned slightly late"}).encode(),
                            ctype="application/json")
        check("/notes accepts an update", status == 200 and
              json.loads(body)["ok"])
        rich = json.loads(a1_rich.read_text())
        check("rich sidecar notes updated on disk",
              rich["notes"] == "turned slightly late")
        session = json.loads(sess.read_text())
        a1_entry = next(t for t in session["takes"] if t["label"] == "A1")
        check("session.json notes propagated",
              a1_entry["notes"] == "turned slightly late")
        status, body = http("POST", "/notes",
                            json.dumps({"condition": "A1", "stamp": a1_stamp,
                                        "notes": "x" * 5000}).encode(),
                            ctype="application/json")
        check("/notes rejects oversized notes", status == 400)

        print("\n== /takes and duplicate POST stamp bump ==")
        status, body = http("GET", "/takes")
        listed = json.loads(body)["takes"]
        check("/takes lists every take",
              {t["condition"] for t in listed} == set(takes))
        status, body = http("POST", "/take", frame_body(takes["A1"]))
        j = json.loads(body)
        check("duplicate A1 POST accepted with a distinct stamp",
              status == 200 and j["ok"] and
              j["stamp"] not in stamps.values())
        status, body = http("GET", "/takes")
        listed = json.loads(body)["takes"]
        check("/takes now shows two A1 takes",
              sum(1 for t in listed if t["condition"] == "A1") == 2)

        bad = dict(takes["A1"])
        bad["meta"] = dict(takes["A1"]["meta"], n_frames=999)
        status, body = http("POST", "/take", frame_body(bad))
        check("inconsistent POST rejected with 400", status == 400 and
              any(k in json.loads(body)["error"] for k in
                  ("luma", "n_frames", "timing", "map")),
                  body.decode()[:200])
        bad2 = dict(takes["A1"])
        bad2["meta"] = dict(takes["A1"]["meta"], condition="../etc")
        status, body = http("POST", "/take", frame_body(bad2))
        check("path-traversal condition rejected", status == 400)

        # -- numpy round-trip ------------------------------------------------
        try:
            import numpy  # noqa: F401
        except ImportError:
            skip("numpy round-trip (npz/nv12/wavio/gate)",
                 "numpy not importable in this interpreter; re-run with the "
                 "audio venv python")
        else:
            print("\n== numpy round-trip: np.load, script math, nv12, wavio ==")
            sys.path.insert(0, str(REPO))
            from src.nv12 import detect_layout, read_luma, stride_for
            from src.wavio import read_wav

            for name in takes:
                t, stamp = takes[name], stamps[name]
                npz = out_dir / f"{name}_{stamp}.npz"
                ref = numpy_reference(t["luma"], t["meta"]["n_frames"])
                with numpy_import(npz) as z:
                    check(f"{name}: np.load keys",
                          set(z.files) == {"maps", "scale", "maps_raw",
                                           "scale_raw", "luma"}, str(z.files))
                    check(f"{name}: luma array identical",
                          numpy.array_equal(z["luma"], ref["Y"]))
                    check(f"{name}: scale dtype float32 0-d",
                          z["scale"].dtype == numpy.float32 and
                          z["scale"].shape == ())
                    ok, why = compare_maps(
                        name, ref["maps_raw"], z["maps_raw"].tobytes(),
                        ref["scale_raw"], float(z["scale_raw"]),
                        exact=True)
                    check(f"{name}: maps_raw byte-identical to script math",
                          ok, why)
                    ok, why = compare_maps(
                        name, ref["maps"], z["maps"].tobytes(),
                        ref["scale"], float(z["scale"]), exact=False)
                    check(f"{name}: maps match script math (<=1 count)", ok,
                          why)
                    js, meta = t["js"], t["meta"]
                    check(f"{name}: js scale == numpy percentile (float64)",
                          abs(js["scale"] - ref["scale"]) <=
                          1e-12 * ref["scale"],
                          f"{js['scale']!r} vs {ref['scale']!r}")
                    check(f"{name}: js scaleRaw == numpy (float64)",
                          abs(js["scaleRaw"] - ref["scale_raw"]) <=
                          1e-12 * ref["scale_raw"],
                          f"{js['scaleRaw']!r} vs {ref['scale_raw']!r}")
                    check(f"{name}: drift_raw exact", js["driftRaw"] ==
                          ref["drift_raw"],
                          f"{js['driftRaw']} vs {ref['drift_raw']}")
                    check(f"{name}: drift_row within 1e-9",
                          abs(js["driftRow"] - ref["drift_row"]) <=
                          1e-9 * ref["drift_row"])

                # nv12: file size == frames x stride, layout detected
                blob = t["luma"]
                check(f"{name}: luma stream size == frames x 15360",
                      len(blob) == t["meta"]["n_frames"] * LUMA_BYTES and
                      len(blob) % stride_for("luma_only") == 0)
                lay = detect_layout(blob)
                check(f"{name}: nv12 detects luma_only, {lay.n_frames} frames",
                      lay.name == "luma_only" and
                      lay.n_frames == t["meta"]["n_frames"] and
                      lay.margin >= 5,
                      f"{lay.name} margin={lay.margin}")
                luma_arr, _ = read_luma(blob, layout="luma_only")
                check(f"{name}: read_luma returns n 160x96 frames",
                      luma_arr.shape == (t["meta"]["n_frames"], 96, 160))

                # wavio: honest rate, full length
                a = read_wav(out_dir / f"{name}_{stamp}.wav")
                check(f"{name}: wavio reads {t['n_samples']} samples "
                      f"at {RATE} Hz",
                      a.source_sample_rate == RATE and a.sample_rate == RATE
                      and len(a.samples) == t["n_samples"],
                      f"{len(a.samples)} @ {a.source_sample_rate}")

            print("\n== THE GATE (experiments/corpus_integrity.py) ==")
            g = subprocess.run(
                [sys.executable, str(REPO / "experiments" / "corpus_integrity.py"),
                 str(out_dir)], capture_output=True, text=True, timeout=600)
            check("gate exits 0 with rich artifacts present",
                  g.returncode == 0,
                  g.stdout[-800:] + g.stderr[-400:])
            summary = [ln for ln in g.stdout.splitlines()
                       if ln.startswith("SUMMARY:")]
            check("gate summary says 0 takes failed",
                  summary and summary[0].startswith("SUMMARY: 0 of"),
                  str(summary))

        print("\n== POST /free (the FREE RECORD session path) ==")
        n_f = 5
        fluma = bytes(range(256)) * (n_f * LUMA_BYTES // 256)
        buf = BytesIO()
        with wave.open(buf, "wb") as wv:
            wv.setnchannels(1)
            wv.setsampwidth(2)
            wv.setframerate(RATE)
            wv.writeframes(b"\x01\x00" * 800)
        fwav = buf.getvalue()
        fmeta = {"tool": "browser_free_record",
                 "wall_clock_start": "2026-10-20T12:00:00.000Z",
                 "wall_clock_end": "2026-10-20T12:00:05.000Z",
                 "n_frames": n_f,
                 "audio": {"sample_rate": RATE, "samples": 800}}
        fts = {"wall_clock_start": fmeta["wall_clock_start"],
               "frame_offsets_ms": [round(i * 33.4, 1)
                                    for i in range(n_f)]}
        fwebm = b"\x1a\x45\xdf\xa3" + b"\x00" * 16

        def free_body(meta, wav, luma, ts, webm):
            parts = []
            for chunk in (json.dumps(meta).encode(), wav, luma,
                          json.dumps(ts).encode(), webm):
                parts.append(struct.pack("<Q", len(chunk)))
                parts.append(chunk)
            return b"".join(parts)

        root_before = {p.name for p in out_dir.glob("*.json")}
        status, body = http("POST", "/free", free_body(
            fmeta, fwav, fluma, fts, fwebm))
        j = json.loads(body)
        check("POST /free accepted", status == 200 and j.get("ok"),
              j.get("error", ""))
        free_stamp = j.get("stamp")
        fdir = out_dir / "free_sessions" / str(j.get("stamp"))
        check("session folder holds the five files",
              j.get("ok") and all((fdir / f).is_file() for f in
                                  ("meta.json", "audio.wav", "luma.raw",
                                   "timestamps.json", "video.webm")))
        check("luma.raw is frames x 15360",
              j.get("ok") and
              (fdir / "luma.raw").stat().st_size == n_f * LUMA_BYTES)
        jmeta = json.loads((fdir / "meta.json").read_text())
        check("meta.json round-trips",
              jmeta.get("tool") == "browser_free_record" and
              jmeta.get("n_frames") == n_f)
        jts = json.loads((fdir / "timestamps.json").read_text())
        check("timestamps.json holds one offset per frame",
              len(jts.get("frame_offsets_ms", [])) == n_f)
        root_after = {p.name for p in out_dir.glob("*.json")}
        check("free session adds NO json to the out root (gate-safe)",
              root_before == root_after)
        status, body = http("POST", "/free", free_body(
            dict(fmeta, n_frames=99), fwav, fluma, fts, fwebm))
        check("luma/n_frames mismatch rejected with 400",
              status == 400 and not json.loads(body).get("ok"))
        status, body = http("POST", "/log",
                            json.dumps({"kind": "verify-heartbeat",
                                        "state": "idle"}).encode())
        diag_line = ""
        if (HERE / "diag.log").is_file():
            diag_line = (HERE / "diag.log").read_text().strip().splitlines()[-1]
        check("POST /log appends a JSONL line to diag.log",
              status == 200 and json.loads(diag_line)
              .get("kind") == "verify-heartbeat" and
              "received_at" in json.loads(diag_line))

        print("\n== the LIVE MONITOR endpoints ==")
        status, body = http("GET", "/live")
        check("GET /live is the repo's live.html",
              status == 200 and body == (HERE / "live.html").read_bytes())
        status, body = http("GET", "/live-state")
        j = json.loads(body)
        check("/live-state returns events and segments",
              status == 200 and j.get("ok") and
              isinstance(j.get("events"), list) and
              isinstance(j.get("segments"), list))
        status, body = http("POST", "/thumb", json.dumps(
            {"data": "data:image/jpeg;base64,dmVyaWZ5",
             "ts": "2026-10-21T00:00:00Z"}).encode())
        check("POST /thumb accepted", status == 200 and
              json.loads(body).get("ok"))
        status, body = http("GET", "/live-state")
        j = json.loads(body)
        check("the thumbnail surfaces in /live-state",
              status == 200 and j.get("thumb", {}).get("data") ==
              "data:image/jpeg;base64,dmVyaWZ5")

        # the live luma batch (the forwarder's source): post frames,
        # read them back, same offsets, same bytes
        import base64 as b64mod
        n_b = 3
        offs_b = [0, 33, 67]
        luma_b = bytes(range(256)) * (n_b * LUMA_BYTES // 256)
        luma_post = struct.pack("<Q", n_b) + struct.pack("<QQQ", *offs_b) \
            + luma_b
        status, body = http("POST", "/luma", luma_post)
        check("POST /luma accepted", status == 200 and
              json.loads(body).get("ok") and
              json.loads(body).get("n") == n_b)
        status, body = http("GET", "/luma-latest")
        jl = json.loads(body)
        check("the luma batch round-trips through /luma-latest",
              status == 200 and jl.get("n") == n_b and
              jl.get("offsets") == offs_b and
              b64mod.b64decode(jl.get("luma_b64", "")) == luma_b)
        seg_dirs = {s["dir"] for s in j.get("segments", [])}
        check("the free session shows up in /live-state segments",
              f"free_sessions/{free_stamp}" in seg_dirs,
              str(sorted(seg_dirs)))

        print("\n== the LIAISON agent API ==")
        status, body = http("GET", "/api/status")
        j = json.loads(body)
        check("/api/status answers with uptime, page, rendezvous",
              status == 200 and j.get("ok") and "uptime_s" in j and
              "page" in j and "rendezvous" in j and
              "sessions_saved" in j)
        status, body = http("GET", "/api/sessions")
        j = json.loads(body)
        mine = [s for s in j.get("sessions", [])
                if s.get("stamp") == str(free_stamp)]
        check("/api/sessions lists the saved session",
              status == 200 and len(mine) == 1)
        # the synthetic wav is 800 samples of value 1: measured, not asserted
        am = mine[0].get("audio_measured", {}) if mine else {}
        check("audio stats are MEASURED from the file",
              am.get("peak_abs") == 1 and am.get("nonzero") == 800 and
              am.get("samples") == 800, str(am))
        status, body = http("POST", "/api/inbox",
                            json.dumps({"from": "verify.py",
                                        "text": "inbox round trip"}).encode())
        check("POST /api/inbox accepted",
              status == 200 and json.loads(body).get("ok"))
        status, body = http("GET", "/api/inbox")
        j = json.loads(body)
        check("inbox round-trips (GET reads the POSTed message)",
              any(m.get("from") == "verify.py" and
                  m.get("text") == "inbox round trip"
                  for m in j.get("messages", [])))
        # a FORCED SILENT WAV must self-report as a SILENT_AUDIO alert
        swav = BytesIO()
        with wave.open(swav, "wb") as wv:
            wv.setnchannels(1)
            wv.setsampwidth(2)
            wv.setframerate(RATE)
            wv.writeframes(b"\x00\x00" * 800)
        status, body = http("POST", "/free", free_body(
            fmeta, swav.getvalue(), fluma, fts, fwebm))
        j = json.loads(body)
        check("forced silent session accepted for the alert test",
              status == 200 and j.get("ok"))
        time.sleep(0.3)
        alerts = []
        if (HERE / "liaison_outbox.jsonl").is_file():
            outbox_tail = (HERE / "liaison_outbox.jsonl").read_text() \
                .splitlines()[-30:]
            for ln in outbox_tail:
                d = json.loads(ln)
                if d.get("kind") == "SILENT_AUDIO":
                    alerts.append(d)
        check("a silent WAV raised its own SILENT_AUDIO alert in the outbox",
              any(a.get("stamp") == j.get("stamp") for a in alerts),
              str(alerts[-1] if alerts else "no alerts"))

        failures = sum(1 for _, ok, _ in RESULTS if ok is False)
        skipped = sum(1 for _, ok, _ in RESULTS if ok is None)
        print(f"\n==== {len(RESULTS) - failures - skipped} passed, "
              f"{failures} failed, {skipped} skipped ====")
        return 1 if failures else 0
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        server_log.close()
        shutil.rmtree(tmp, ignore_errors=True)


def numpy_import(npz):
    import numpy as np
    return np.load(npz)


if __name__ == "__main__":
    sys.exit(main())
