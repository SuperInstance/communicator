"""
room_map.py -- characterise the room itself, with nobody in it.

WHY THIS EXPERIMENT EXISTS.

Two of Casey's asks, folded into one pass over a single long capture taken with
the room empty:

  1. "capture your own sound" -- what does this room SOUND like, on its own,
     with no one speaking into it?
  2. "mapping the room" -- what does this room LOOK like, and can the existing
     pipeline hold a stable map of it?

Both are controls the rest of the project has been missing. Every audio number
so far has been measured on a signal with a person in it; every vision number on
a frame with motion in it. Neither has ever characterised the substrate.

SOUND.

  * ROOM TONE. The level distribution and its spectrum with no speech present.
    This is the floor any detector has to beat, measured rather than assumed.
  * HARMONIC PEAKS. A closed room with electronics in it usually hums. Peaks in
    the spectrum that survive with nobody speaking are the building, not the
    speaker -- and they are what `docs/research/MIC-HARDWARE.md` was fighting.
  * ROOM TAIL. Not a synthesised impulse response -- the measured decay after
    REAL onsets. The capture spans Casey leaving, the door, and the room
    settling, so there are genuine transients in it. Each detected onset gets
    its own decay slope; the median slope across onsets is a crude RT60. It is
    crude, and it is real data rather than a synthetic tail.

  The control that makes the hum finding trustworthy: if a peak is present in the
  empty room AND in speech-containing captures, it is the room, not the speaker.

LOOK.

  * MEDIAN COMPOSITE. Over a few thousand frames of a scene nobody is moving,
    the per-pixel MEDIAN is a far cleaner map than any single frame. Sensor
    noise and the exposure hunt are zero-mean-ish, so the median removes them and
    keeps the scene. This is the cheapest denoiser in the project and it is
    completely standard -- but it has never been run here on a long capture,
    because every capture so far has been 30 to 900 frames of a moving subject.
  * TEMPORAL STABILITY OF THE MAP. How much does the median composite move as
    the window slides? If a room map is stable, the substrate is static and
    differences between it and any future capture are content, not camera.

THE FRAME LAYOUT.

The captures on disk are NOT standard NV12: `C:\\proart_capture\\nv12small.py`
writes 19,200 bytes per 160x96 frame (15,360 luma + 1,920 U + 1,920 V), not
the `W*H*3//2` = 23,040 a standard-NV12 reader assumes. Read at 23,040, all
but every sixth frame is torn. The layout is DETECTED from the file's content
by `src/nv12.py` -- the candidate whose luma has the smallest mean consecutive
|diff| wins, and an ambiguous file is an error rather than a guess. See
`src/nv12.py` and `docs/research/NV12-STRIDE.md`.

LIMITS, STATED BEFORE THE NUMBERS.

  * Onset detection will fire on non-room events -- a door, a chair, Casey
    walking off. That is fine and intended; the decay after a door closing is
    still the room. But the report labels these as "transients", not as
    "reverberation events", and a small number of them is a weak sample.
  * One room, one night, one machine. Nothing here generalises to other rooms.
  * The median composite assumes the scene is static over the window. If Casey
    is still moving at the start, the median silently smears him. The stability
    series is reported precisely so that failure is visible rather than hidden.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import wave
from array import array
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FRAME_W, FRAME_H = 160, 96
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.nv12 import read_luma  # noqa: E402


def sha256_file(path) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_wav_mono(path: Path):
    with wave.open(str(path), "rb") as f:
        assert f.getnchannels() == 1, f"expected mono, got {f.getnchannels()}"
        sr = f.getframerate()
        a = array("h")
        a.frombytes(f.readframes(f.getnframes()))
    return sr, a


def goertzel_power(x, sr, freq):
    """Power at one frequency, without an FFT dependency.

    The angular frequency MUST stay a float. An earlier version wrote
    `k = int(2 * math.pi * freq / sr)`, which truncates to 0 for every
    frequency below sr/pi (~2.5 kHz at 16 kHz), making coeff = 2*cos(0) = 2
    identically and returning the SAME power for 50, 100, 200 and 1000 Hz.
    The tell was eight harmonics all reporting the same value to one decimal
    place, which real noise cannot do. Kept as a comment because the symptom --
    a plausible-looking spectrum that is entirely fabricated -- is exactly the
    kind of thing that silently survives review.
    """
    k = 2 * math.pi * freq / sr
    n = len(x)
    if n == 0:
        return 0.0
    coeff = 2 * math.cos(k)
    s1 = s2 = 0.0
    for v in x:
        s0 = v + coeff * s1 - s2
        s2, s1 = s1, s0
    return s1 * s1 + s2 * s2 - coeff * s1 * s2


def room_tail(sr, a, win_s=0.4):
    """Decay slope after each detected onset. Returns (n_onsets, list of ms60)."""
    hop = int(sr * 0.02)
    n_win = int(sr * win_s)
    n = len(a) // hop
    if n < n_win // hop + 1:
        return 0, []
    env = []
    for i in range(n):
        s = i * hop
        seg = a[s:s + hop]
        env.append(math.sqrt(sum(float(v) * v for v in seg) / max(len(seg), 1)))
    thr = max(env) * 0.12
    above = [e > thr for e in env]
    onsets = [i for i in range(1, len(above)) if above[i] and not above[i - 1]]

    out = []
    for o in onsets:
        seg = env[o:o + n_win // hop]
        if len(seg) < 8:
            continue
        peak = max(seg)
        if peak <= 0:
            continue
        # first crossing of -5 dB and -25 dB relative to the local peak
        t5 = t25 = None
        for i, e in enumerate(seg):
            db = 20 * math.log10(max(e, 1e-9) / peak)
            if t5 is None and db <= -5:
                t5 = i
            if db <= -25:
                t25 = i
                break
        if t5 is not None and t25 is not None and t25 > t5:
            t60 = (t25 - t5) * 3 * 0.02
            if 0.02 < t60 < 3.0:
                out.append(t60 * 1000.0)
    return len(onsets), out


def audio_report(path: Path):
    sr, a = read_wav_mono(path)
    dur = len(a) / sr
    n = len(a)
    rms = math.sqrt(sum(float(v) * v for v in a) / max(n, 1))
    p95 = sorted(abs(v) for v in a)[int(n * 0.95)]

    # DFT magnitudes over a 1 s window, subsampled for speed
    win = a[: sr] if len(a) >= sr else a
    freqs = [(i, goertzel_power(win, sr, i)) for i in range(20, 400, 5)]
    peak_f, peak_p = max(freqs, key=lambda t: t[1])
    top = sorted(freqs, key=lambda t: -t[1])[:12]

    # 50/60 Hz and harmonics -- the hum the project has been fighting
    hum = []
    for base in (50, 60):
        for h in range(1, 5):
            f = base * h
            if f < sr / 2:
                hum.append((f, goertzel_power(win, sr, f)))

    n_on, rt = room_tail(sr, a)
    return {
        "path": str(path), "sample_rate": sr, "duration_s": dur,
        "rms_amplitude": rms, "peak_95pct": p95,
        "rms_dbfs": 20 * math.log10(max(rms, 1e-9) / 32768),
        "loudest_bin_hz": peak_f,
        "top_bins_hz": [f for f, _ in top],
        "hum_harmonics_db": [(f, 10 * math.log10(max(p, 1e-9))) for f, p in hum],
        "n_transients": n_on,
        "rt60_estimates_ms": rt,
        "rt60_median_ms": statistics.median(rt) if rt else None,
    }


def median_composite(frames, n):
    import numpy as np
    F = frames[:n].astype(np.uint8)
    return np.median(F, axis=0).astype(np.uint8)


def vision_report(path: Path):
    import numpy as np
    # The layout is DETECTED from the file's content, not assumed -- see
    # `src/nv12.py`. The captures on disk are 19,200 bytes per frame, not the
    # 23,040 a standard-NV12 reader assumes.
    luma, layout = read_luma(path, width=FRAME_W, height=FRAME_H)
    nf = layout.n_frames
    F = luma
    n = min(nf, 2000)
    F = F[:n].astype(np.float64)
    means = F.reshape(n, -1).mean(axis=1)[:, None, None]

    raw_mad = float(np.abs(np.diff(F, axis=0)).mean())
    norm_mad = float(np.abs(np.diff(F / means, axis=0)).mean())

    # sliding median composite: does the map move as the window advances?
    drift = []
    comps = []
    step = max(n // 10, 1)
    for s in range(0, max(n - 200, 1), step):
        w = F[s:s + 200]
        comps.append(np.median(w, axis=0))
    for i in range(1, len(comps)):
        drift.append(float(np.abs(comps[i] - comps[i - 1]).mean()))

    med = median_composite(F.astype(np.uint8), n)
    out = ROOT / "experiments" / "room_median_map.pgm"
    Path(out).write_bytes(f"P5\n{FRAME_W} {FRAME_H}\n255\n".encode() + med.tobytes())
    return {
        "path": str(path), "n_frames": n,
        "capture_n_frames": int(nf),
        "layout": layout.name,
        "stride_bytes": layout.stride,
        "layout_evidence": layout.as_dict(),
        "mean_luma": float(F.mean()),
        "adjacent_mad_raw": raw_mad,
        "adjacent_mad_gain_normalised": norm_mad,
        "median_map_windows": len(comps),
        "median_map_step_drift_mean": statistics.fmean(drift) if drift else None,
        "median_map_step_drift_max": max(drift) if drift else None,
        "median_map_pgm": str(out),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="/mnt/c/proart_capture/empty1.wav")
    ap.add_argument("--nv12", default="/mnt/c/proart_capture/empty1.nv12")
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    report = {"question": "what does this room sound and look like with nobody in it?",
              "confounds": [
                  "one room, one night, one machine -- nothing here generalises",
                  "onsets in the tail analysis include door and chair events; "
                  "they are transients, not calibrated reverberation events",
                  "the median composite assumes a static scene; its sliding "
                  "window drift is reported so that assumption can be checked",
              ],
              "provenance": {
                  "generated_by": "experiments/room_map.py",
                  "script_path": str(Path(__file__).resolve()),
                  "script_sha256": sha256_file(Path(__file__).resolve()),
                  "python": sys.executable,
                  "command_line": " ".join(sys.argv),
                  "cli_args": list(sys.argv[1:]),
                  "inputs": {},
              }}

    if Path(a.wav).is_file():
        ar = audio_report(Path(a.wav))
        report["audio"] = ar
        report["provenance"]["inputs"]["wav"] = {
            "path": str(Path(a.wav).resolve()),
            "sha256": sha256_file(a.wav),
            "bytes": Path(a.wav).stat().st_size,
        }
        print(f"AUDIO  {ar['duration_s']:.1f}s  rms {ar['rms_dbfs']:.1f} dBFS  "
              f"p95 {ar['peak_95pct']}")
        print(f"       loudest bin {ar['loudest_bin_hz']} Hz")
        print("       top bins:", ", ".join(f"{f}Hz" for f in ar["top_bins_hz"][:8]))
        print("       hum harmonics (dB, arbitrary ref):")
        for f, db in ar["hum_harmonics_db"]:
            print(f"           {f:6.0f} Hz  {db:7.1f}")
        print(f"       transients: {ar['n_transients']}   "
              f"RT60 median: {ar['rt60_median_ms']} ms"
              if ar["rt60_median_ms"] else
              f"       transients: {ar['n_transients']}   RT60 median: n/a")

    if Path(a.nv12).is_file():
        vr = vision_report(Path(a.nv12))
        report["vision"] = vr
        report["provenance"]["inputs"]["nv12"] = {
            "path": str(Path(a.nv12).resolve()),
            "sha256": sha256_file(a.nv12),
            "bytes": Path(a.nv12).stat().st_size,
            "layout": vr["layout"],
            "stride_bytes": vr["stride_bytes"],
            "n_frames": vr["capture_n_frames"],
        }
        report["supersedes"] = {
            "why": ("Re-measured on 2026-10-20. The previous run read the "
                    "capture at W*H*3//2 = 23,040 bytes per frame; the "
                    "capture on disk carries 19,200 bytes per frame, so all "
                    "but every sixth frame was torn. The previous numbers are "
                    "recorded here rather than silently overwritten."),
            "previous_stride_bytes": 23040,
            "previous_n_frames": 3750,
            "previous_input_paths": {
                "nv12": "/home/proart/projects/glasses/.capture_keep/empty1.nv12",
                "wav": "/home/proart/projects/glasses/.capture_keep/empty1.wav",
            },
            "previous_input_paths_note": (
                "The previous run read the same BYTES from a different path "
                "(the .capture_keep copy). The SHA-256 is identical, so this "
                "is a path change, not a data change; the audio section is "
                "unch apart from the path field."),
            "previous_vision": {
                "n_frames": 2000,
                "mean_luma": 49.70077926432292,
                "adjacent_mad_raw": 39.315094363588045,
                "adjacent_mad_gain_normalised": 0.808906145981632,
                "median_map_windows": 9,
                "median_map_step_drift_mean": 0.0001708984375,
                "median_map_step_drift_max": 0.00048828125,
            },
            "previous_verdict": None,
        }
        print(f"\nVISION {vr['n_frames']} frames  mean luma {vr['mean_luma']:.2f}")
        print(f"       adjacent MAD raw {vr['adjacent_mad_raw']:.3f} "
              f"-> gain-normalised {vr['adjacent_mad_gain_normalised']:.4f}")
        print(f"       median-map step drift: mean "
              f"{vr['median_map_step_drift_mean']:.4f}  max "
              f"{vr['median_map_step_drift_max']:.4f}")
        print(f"       median map -> {vr['median_map_pgm']}")

    p = Path(a.out) / "room_map_results.json"
    p.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()