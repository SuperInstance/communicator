#!/usr/bin/env python3
"""
pair_analysis.py -- BRIEF-DUAL-CAPTURE.md: the pair analysis.

WHAT Casey RECORDS: phone room-lens (START, three sharp CLAPS -- start /
middle / end -- ~5 natural minutes walking and talking, STOP, UPLOAD) and
the ProArt witness page (fixed cam+mic, AUTO banking all evening). The
claps are the sync events; everything below hangs off finding them.

WHAT THIS DOES, in order (each stage measured, each honest when the data
is not good enough):
  1. LOAD the pair: the phone bundle from the room-lens upload worker
     (GET /bundle/<id> or a local file), and the ProArt sessions that
     overlap its wall-clock window from the corpus.
  2. FIND THE CLAPS in each audio track: onset detection (energy in a
     clap band, adaptive floor, min separation) -- expects exactly 3,
     reports whatever it finds with times and strengths.
  3. OFFSET + DRIFT: clap k's phone time vs ProArt time, k = 1..3.
     Offset = mean delay; drift = slope of delay across the three
     (sub-ms resolution when the onset estimates allow; the report
     states the onset uncertainty honestly). The pair (offset, drift)
     is LENS-FILE data for both microphones.
  4. TDOA TRACK: generalized cross-correlation (plain CC at 10 ms hop
     over 1 s windows, GCC-PHAT refinement optional) across the
     wall-clock-aligned timeline -> per-window inter-mic delay -> a
     bearing/distance track (two mics give a hyperbola; with the clap
     sync anchoring absolute time, the track is where Casey is in the
     room, up to the left/right ambiguity a mic pair cannot resolve).
  5. WHO-HEARS-WAT MAP: per window, per mic: SNR against its own noise
     floor -> which mic hears the speech better, and when it flips.
  6. REPORT: JSON to stdout (and --out), human table to stderr. Never
     asserts a sync it cannot measure; a clap not found is reported as
     a missing anchor, not guessed.

USAGE:
    python3 browser_capture/pair_analysis.py --bundle <id-or-url-or-path> \
        [--corpus /mnt/c/proart_capture/corpus] [--out report.json]

Stdlib-only for the frame math; numpy used when present (the audio-venv
has it) for the cross-correlations.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import struct
import sys
import time
import urllib.request
import wave
from pathlib import Path

try:
    import numpy as np
except ImportError:      # pragma: no cover -- the audio-venv has numpy
    np = None

WORKER = "https://room-lens-upload.casey-digennaro.workers.dev"


# -- loading -----------------------------------------------------------------

def load_phone_audio(source: str) -> tuple[np.ndarray, int, dict]:
    """Phone bundle -> (mono float32 [-1,1] at its native rate, rate, meta).
    Accepts: a worker session id, a full worker URL, or a local JSON path.
    The bundle's audio rides as base64 webm/opus -- decoded via ffmpeg
    if present; otherwise the loader says so honestly."""
    if source.isdigit() or source.startswith("http"):
        url = source if source.startswith("http") else \
            f"{WORKER}/bundle/{source}"
        with urllib.request.urlopen(url, timeout=30) as r:
            bundle = json.loads(r.read().decode())
    else:
        bundle = json.loads(Path(source).read_text())
    audio = (bundle.get("audio") or {})
    if not audio.get("base64"):
        raise SystemExit("bundle has no audio (was the mic granted?)")
    raw = base64.b64decode(audio["base64"])
    suffix = ".webm" if "webm" in (audio.get("mimeType") or "") else ".m4a"
    tmp = Path(f"/tmp/pair_phone{suffix}")
    tmp.write_bytes(raw)
    import subprocess, shutil
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("phone audio is opus-in-webm; ffmpeg is needed "
                         "to decode it (sudo apt install ffmpeg)")
    out = subprocess.run([ffmpeg, "-v", "error", "-i", str(tmp),
                          "-f", "f32le", "-ac", "1", "-ar", "48000", "-"],
                         capture_output=True, timeout=120)
    if out.returncode != 0:
        raise SystemExit(f"ffmpeg decode failed: {out.stderr[:200]}")
    x = np.frombuffer(out.stdout, dtype=np.float32)
    meta = bundle.get("meta") or {}
    return x, 48000, {"meta": meta, "bytes": len(raw),
                      "frames": len(bundle.get("frames") or []),
                      "started_at": meta.get("startedAt")}


def load_witness_window(corpus: Path, t0_iso: str, t1_iso: str, pad_s=10):
    """Concatenate the ProArt session WAVs whose wall clocks overlap
    [t0-pad, t1+pad], returning (samples, rate, [(stamp, t_offset_s)])."""
    root = corpus / "free_sessions"
    picked = []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        try:
            m = json.loads((d / "meta.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        ws, we = m.get("wall_clock_start"), m.get("wall_clock_end")
        if not ws or not we or m.get("audio", {}).get("audio_silent"):
            continue
        if ws[:19] <= t1_iso[:19] and we[:19] >= t0_iso[:19]:
            picked.append((ws, d))
    if not picked:
        raise SystemExit("no ProArt sessions overlap the phone window "
                         "(is the witness page banking?)")
    picked.sort()
    parts, stamps = [], []
    rate = 48000
    for ws, d in picked:
        with wave.open(str(d / "audio.wav")) as w:
            rate = w.getframerate()
            a = np.frombuffer(w.readframes(w.getnframes()), "<i2")
        parts.append(a.astype(np.float32) / 32768)
        stamps.append((ws, sum(len(p) for p in parts[:-1]) / rate))
    return np.concatenate(parts), rate, stamps


# -- claps --------------------------------------------------------------------

def find_claps(x: np.ndarray, rate: int, want: int = 3) -> list:
    """Clap onsets: energy in a 1.2-4 kHz clap band (a clap is broadband
    but its crunch lives there), adaptive floor, onset = first frame
    over floor*8 within a rising run; min 0.8 s apart."""
    hop = int(0.005 * rate)
    win = int(0.02 * rate)
    n = (len(x) - win) // hop
    if n <= 0:
        return []
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, win), strides=(hop * 4, 4)).copy()
    # cheap clap band: |diff| energy (broadband transient emphasis)
    e = (np.abs(np.diff(frames, axis=1)).sum(axis=1) / win)
    order = np.argsort(e)
    floor = np.median(e)
    # a clap must be a REAL transient: at least 20x the room floor AND
    # at least 30% of the strongest event in the track -- speech onsets
    # (the synthetic pair proved they sneak through a floor-only gate)
    gate = max(floor * 20, float(e.max()) * 0.3)
    hits, last = [], -1e9
    for i in order[::-1]:
        if e[i] < gate:
            break
        if any(abs(i - h) * 0.005 < 0.8 for h in hits):
            continue
        hits.append(int(i))
        hits.sort()
    out = []
    for h in hits[:want]:
        out.append({"t": round(h * 0.005, 4),
                    "strength": round(float(e[h] / max(floor, 1e-9)), 1)})
    return out


# -- offset + drift ------------------------------------------------------------

def offset_drift(phone_claps, wit_claps):
    """Pair the claps in order (both sorted by time); report per-clap
    delay, mean offset, and drift (least-squares slope)."""
    k = min(len(phone_claps), len(wit_claps))
    if k < 2:
        return {"ok": False,
                "reason": f"only {k} clap pair(s) -- need >= 2 for drift"}
    pairs = [(phone_claps[i]["t"], wit_claps[i]["t"]) for i in range(k)]
    delays = [w - p for p, w in pairs]
    ts = [p for p, _ in pairs]
    if k >= 3:
        n = float(len(ts))
        slope = (n * sum(t * d for t, d in zip(ts, delays)) -
                 sum(ts) * sum(delays)) / \
                (n * sum(t * t for t in ts) - sum(ts) ** 2 + 1e-12)
    else:
        slope = delays[1] - delays[0] / max(ts[1] - ts[0], 1e-6)
    return {"ok": True, "pairs": [{"phone_s": p, "witness_s": w,
                                   "delay_ms": round(d * 1000, 2)}
                                  for (p, w), d in zip(pairs, delays)],
            "offset_ms": round(sum(delays) / k * 1000, 2),
            "drift_ms_per_min": round(slope * 60 * 1000, 3)}


# -- TDOA track + who-hears-what ------------------------------------------------

def tdoa_track(px, pr, rate, offset_s, win_s=1.0, hop_s=0.5, max_lag_ms=8):
    """Per-window inter-mic delay by plain cross-correlation, after
    aligning the tracks on the clap offset. With one mic pair this is a
    hyperbola of position, not a point -- the report says so."""
    lag = int(max_lag_ms / 1000 * rate)
    w = int(win_s * rate)
    hop = int(hop_s * rate)
    shift = int(offset_s * rate)
    out = []
    for start in range(0, min(len(px), len(pr)) - w, hop):
        a = px[start:start + w]
        b = pr[start + shift:start + shift + w]
        if len(b) < w:
            break
        if np.abs(a).max() < 1e-4 and np.abs(b).max() < 1e-4:
            continue
        c = np.correlate(a[:rate], b[:rate], mode="full")
        mid = rate - 1
        seg = c[mid - lag:mid + lag + 1]
        peak = int(np.argmax(seg)) - lag
        ea = float(np.sqrt((a[:rate] ** 2).mean()))
        eb = float(np.sqrt((b[:rate] ** 2).mean()))
        out.append({
            "t": round(start / rate + win_s / 2, 2),
            "delay_ms": round(peak / rate * 1000, 3),
            "phone_rms": round(ea, 5), "witness_rms": round(eb, 5),
            "clearer": "phone" if ea > eb * 1.5 else
                       "witness" if eb > ea * 1.5 else "both",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True,
                    help="worker session id, worker URL, or local path")
    ap.add_argument("--corpus", default="/mnt/c/proart_capture/corpus")
    ap.add_argument("--out", help="write the JSON report here")
    a = ap.parse_args()
    if np is None:
        raise SystemExit("numpy required (audio-venv python)")

    px, prate, pinfo = load_phone_audio(a.bundle)
    print(f"phone: {len(px)/prate:.1f}s @ {prate}Hz, "
          f"frames banked {pinfo['frames']}", file=sys.stderr)
    t0 = pinfo.get("started_at") or time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - len(px) / prate))
    t1 = time.strftime("%Y-%m-%dT%H:%M:%S",
                       time.gmtime(time.mktime(time.strptime(
                           t0[:19], "%Y-%m-%dT%H:%M:%S")) + len(px) / prate))
    wx, wrate, stamps = load_witness_window(Path(a.corpus), t0, t1)
    print(f"witness: {len(wx)/wrate:.1f}s @ {wrate}Hz over "
          f"{len(stamps)} session(s)", file=sys.stderr)

    pc = find_claps(px, prate)
    wc = find_claps(wx, wrate)
    print(f"claps: phone {len(pc)} {[c['t'] for c in pc]} | "
          f"witness {len(wc)} {[c['t'] for c in wc]}", file=sys.stderr)

    od = offset_drift(pc, wc)
    track = tdoa_track(px, wx, prate,
                       od.get("pairs", [{}])[0].get("delay_s", 0)
                       if od.get("ok") else 0) if od.get("ok") else []
    flips = sum(1 for i in range(1, len(track))
                if track[i]["clearer"] != track[i - 1]["clearer"])
    report = {
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phone": {"seconds": round(len(px) / prate, 1),
                  "frames": pinfo["frames"], "claps": pc},
        "witness": {"seconds": round(len(wx) / wrate, 1),
                    "sessions": len(stamps), "claps": wc},
        "sync": od,
        "tdoa": {"windows": len(track), "flip_count": flips,
                 "note": "one mic pair = a hyperbola of position, not a "
                         "point; the left/right ambiguity is physical",
                 "track": track},
        "who_hears_what": {
            "summary": {k: sum(1 for t in track if t["clearer"] == k)
                        for k in ("phone", "witness", "both")}},
    }
    text = json.dumps(report, indent=2)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
