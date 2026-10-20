"""
av_sync.py -- do the video frames and the audio of one capture actually
correspond, and does the visible motion track the audio?

THE FRAME LAYOUT, FIRST, because it decides whether any of this is true.

The captures on disk are NOT standard NV12. `C:\\proart_capture\\nv12small.py`
writes, per 160x96 frame:

    Y plane   15,360 bytes   (160 * 96)
    U,V        3,840 bytes   (80 * 24 samples, INTERLEAVED)
    ---------------------------
    total     19,200 bytes   = W*H*5/4

so the stride is 19,200 bytes, not the `W*H*3//2` = 23,040 this file used to
assume. Read at 23,040, all but every sixth frame is a tear -- rows 0..71 from
frame k, rows 72..95 from frame k+1 -- and the mean |diff| between adjacent
frames is tens of counts instead of well under one.

The layout is DETECTED from the file's content by `src/nv12.py` (the candidate
whose luma has the smallest mean consecutive |diff| wins; an ambiguous file is
an error, not a guess). See `src/nv12.py` and `docs/research/NV12-STRIDE.md`.

WHY THIS FILE EXISTS.

Preparing the cloud-teacher measurement, two facts about the existing
captures turned up:

  1. `A.wav` is 29.98 s of audio. `A.nv12` is 900 frames at 30 fps = 30.00 s
     of video. THE TWO ARE THE SAME LENGTH, to within one frame.

     This file used to say the opposite. It said `A.nv12` was 750 frames =
     25.0 s and that "THE TWO ARE NOT THE SAME LENGTH", with a 4.98 s
     mismatch. The 750 came from dividing 17,280,000 bytes by the wrong
     stride. At the right stride the mismatch is 0.019 s -- less than half a
     video frame -- and the premise of this experiment changes with it.

  2. The Silero gate puts A's first speech at block 61 = 1.95 s, which is
     video frame 58 at 30 fps.

Those two facts are compatible with very different worlds. If the audio and
video were captured by the same loop, then time-to-frame conversion is valid
and the frame labels are sound. If they were separate recordings, or the audio
loop ran at a different rate, then every "frame t corresponds to audio time
t/30" claim is an assumption with no evidence behind it -- and the whole
per-frame labelling scheme rests on it.

So it gets measured. This is the same discipline as the rest of the project:
the alignment is a claim, and a claim needs a number behind it.

THE MEASUREMENT.

Cross-correlate the VIDEO MOTION envelope against the AUDIO ENERGY envelope
over a range of lags.

    video motion m[t]  = mean |frame(t+1) - frame(t)| over the frame,
                         computed on ROW-BAND-REMOVED frames so the LED
                         flicker banding does not dominate the signal
    audio energy e[t]  = RMS of the audio over video frame t's time span

The lag at which the correlation peaks is the A/V offset. If the peak sits at
lag 0 with a correlation well above the null, the two streams correspond and
the speech in the audio is the speech in the pixels.

The row-band removal was introduced to fight an artefact that was, in large
part, the torn frame itself. It is kept for the primary statistic so the
method is unchanged, and a RAW-motion statistic -- no row-band removal at all
-- is reported alongside it as the control that says whether the filter is
doing any work.

THE NULLS, because a peak is easy to produce by accident.

  SHUFFLED AUDIO   the audio envelope's blocks are shuffled in time and the
                   whole cross-correlation is redone. This is the null
                   distribution of the peak statistic under NO relationship
                   between sound and picture. The real peak has to beat it.
  OTHER SESSION    A's video against B's audio, and vice versa. Two
                   different recordings of the same room in the same
                   conditions. If the peak survives THAT, the measurement is
                   picking up the room, not the person.
  WITHIN SESSION  A's video against its own audio at a SHIFTED lag, which is
                   how long the speech segments are, in seconds. If the
                   correlation is flat across all lags, the video has no
                   time-varying structure at the speech timescale at all.

WHAT IS REPORTED.

The peak lag in frames and milliseconds, the correlation there, the null
distribution's maximum, and the ratio. And the durations, because that is the
number that started this.

WHAT IS NOT CLAIMED.

One session per condition, one room, one person, one laptop camera. A
cross-correlation peak is evidence that the two streams correspond; it is
not a statement about a general system, and a peak of 0.3 does not become 0.3
"accuracy" for anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.nv12 import read_luma  # noqa: E402

W, H = 160, 96
YS = W * H
FPS = 30
RATE = 16000


def sha256_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_y(path: Path):
    """The luma of every frame, and the layout that was detected.

    The stride is DETECTED from the file's content by `src/nv12.py`, not
    assumed. See `src/nv12.py`.
    """
    luma, layout = read_luma(path, width=W, height=H)
    return luma.astype(np.float64), layout


def row_band_remove(ys: np.ndarray) -> np.ndarray:
    """Subtract each row's median across the columns -- the LED flicker
    banding measured in ROW-BAND-REMOVAL.md."""
    prof = np.median(ys, axis=2)
    g = prof.mean(axis=1, keepdims=True)
    return ys - (prof - g)[:, :, None]


def motion_envelope(Y: np.ndarray, rows: slice | None = None):
    """Per-frame motion energy, on row-band-removed frames."""
    Yr = row_band_remove(Y)
    if rows is not None:
        Yr = Yr[:, rows, :]
    d = np.abs(np.diff(Yr, axis=0)).mean(axis=(1, 2))
    return d


def raw_motion_envelope(Y: np.ndarray, rows: slice | None = None):
    """Per-frame motion energy with NO row-band removal.

    The control for the row-band filter: if the two agree, the filter is not
    what is producing the result.
    """
    Yr = Y if rows is None else Y[:, rows, :]
    return np.abs(np.diff(Yr, axis=0)).mean(axis=(1, 2))


def audio_envelope(wav_path: Path, n_frames: int | None = None):
    """RMS energy per video frame, plus the audio's true duration.

    The envelope is built over the AUDIO's own length, not the video's, so
    the duration mismatch between the two streams is visible rather than
    hidden by a shared frame count.
    """
    sys.path.insert(0, str(ROOT))
    from src.wavio import read_wav
    a = read_wav(str(wav_path))
    x = np.asarray(a.samples, dtype=np.float64)
    dur = len(x) / RATE
    if n_frames is None:
        n_frames = int(np.ceil(dur * FPS))
    out = np.zeros(n_frames)
    for t in range(n_frames):
        i0 = int(t * RATE / FPS)
        i1 = max(i0 + 1, int((t + 1) * RATE / FPS))
        if i0 >= len(x):
            break
        out[t] = float(np.sqrt((x[i0:min(i1, len(x))] ** 2).mean()))
    return out, dur, len(x)


def zc(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.std() < 1e-12:
        return np.zeros_like(v)
    return (v - v.mean()) / v.std()


def best_lag(m: np.ndarray, e: np.ndarray, max_lag: int = 300):
    """Cross-correlate motion against audio over lags, return (lag, r, curve).

    The lag is in video frames: positive means the motion happens AFTER the
    audio, which is what a camera-and-mic capture on the same clock shows.
    """
    n = min(len(m), len(e))
    m, e = zc(m[:n]), zc(e[:n])
    lags = np.arange(-max_lag, max_lag + 1)
    curve = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag >= 0:
            a, b = m[lag:], e[:n - lag]
        else:
            a, b = m[:n + lag], e[-lag:]
        if len(a) < 30:
            curve[i] = 0.0
            continue
        curve[i] = float((a * b).mean())
    k = int(np.argmax(curve))
    return int(lags[k]), float(curve[k]), curve, lags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="/mnt/c/proart_capture")
    ap.add_argument("--tags", default="A,B,C")
    ap.add_argument("--max-lag-s", type=float, default=8.0)
    ap.add_argument("--shuffles", type=int, default=200,
                    help="null distribution: block-shuffled audio")
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    cap = Path(a.captures)
    max_lag = int(a.max_lag_s * FPS)
    rng = np.random.default_rng(20261019)
    results = {}
    provenance = {}
    conclusion = {}

    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        nv = cap / f"{tag}.nv12"
        wav = cap / f"{tag}.wav"
        if not nv.is_file() or not wav.is_file():
            print(f"{tag}: missing")
            continue
        Y, layout = load_y(nv)
        n_vid = len(Y)
        e, dur, n_samp = audio_envelope(wav)
        m = motion_envelope(Y)
        m_head = motion_envelope(Y, rows=slice(0, 40))
        m_raw = raw_motion_envelope(Y)

        print(f"\n===== {tag} =====")
        print(f"  video {n_vid} frames = {n_vid / FPS:.2f}s    "
              f"audio {n_samp} samples = {dur:.2f}s    "
              f"LENGTH MISMATCH = {dur - n_vid / FPS:+.3f}s")
        print(f"  layout {layout.name} stride {layout.stride}  "
              f"mean|d| {layout.mean_frame_diff:.4f}  "
              f"margin {layout.margin:.1f}x")

        lag, r, curve, lags = best_lag(m, e, max_lag)
        lag_h, r_h, _, _ = best_lag(m_head, e, max_lag)
        lag_raw, r_raw, _, _ = best_lag(m_raw, e, max_lag)

        # -- the null: block-shuffled audio ------------------------------
        # Blocks of 10 frames are shuffled, so the envelope keeps its local
        # structure but loses its alignment with the picture. A peak of the
        # real statistic that the null can also produce is not evidence.
        block = 10
        nb = len(e) // block
        null_peaks = []
        null_peaks_raw = []
        for _ in range(a.shuffles):
            order = rng.permutation(nb)
            shuffled = np.concatenate([e[i * block:(i + 1) * block]
                                       for i in order])
            shuffled = np.concatenate([shuffled, e[nb * block:]])
            _, rn, _, _ = best_lag(m, shuffled[:len(e)], max_lag)
            null_peaks.append(abs(rn))
            _, rr, _, _ = best_lag(m_raw, shuffled[:len(e)], max_lag)
            null_peaks_raw.append(abs(rr))
        null_peaks = np.array(null_peaks)
        null_max = float(null_peaks.max())
        null_mean = float(null_peaks.mean())
        null_peaks_raw = np.array(null_peaks_raw)
        null_max_raw = float(null_peaks_raw.max())

        # -- the other-session control ------------------------------------
        other = None
        for ot in ("A", "B", "C"):
            if ot == tag:
                continue
            if not (cap / f"{ot}.wav").is_file():
                continue
            eo, _, _ = audio_envelope(cap / f"{ot}.wav", int(np.ceil(dur * FPS)))
            _, ro, _, _ = best_lag(m, eo[:len(e)], max_lag)
            other = {"audio_from": ot, "peak_r": ro, "abs_r": abs(ro)}
            break

        ratio = abs(r) / null_max if null_max > 0 else None
        ratio_raw = abs(r_raw) / null_max_raw if null_max_raw > 0 else None
        res = {
            "video_frames": int(n_vid),
            "video_seconds": round(n_vid / FPS, 3),
            "audio_seconds": round(dur, 3),
            "duration_mismatch_s": round(dur - n_vid / FPS, 3),
            "peak_lag_frames": lag,
            "peak_lag_ms": round(lag * 1000 / FPS, 1),
            "peak_r": round(r, 4),
            "peak_r_top40rows": round(r_h, 4),
            "peak_lag_frames_top40rows": lag_h,
            "null_shuffled_max_abs_r": round(null_max, 4),
            "null_shuffled_mean_abs_r": round(null_mean, 4),
            "ratio_peak_over_null_max": (round(ratio, 2) if ratio else None),
            "other_session_control": other,
            "motion_mean": float(m.mean()),
            "motion_std": float(m.std()),
            "audio_env_mean": float(e[:len(m)].mean()),
            "raw_motion_control": {
                "note": ("the same statistic with NO row-band removal: the "
                         "row-band filter was introduced to fight an artefact "
                         "that was largely the torn frame, so this control says "
                         "whether the filter is doing any work"),
                "peak_lag_frames": lag_raw,
                "peak_lag_ms": round(lag_raw * 1000 / FPS, 1),
                "peak_r": round(r_raw, 4),
                "null_shuffled_max_abs_r": round(null_max_raw, 4),
                "ratio_peak_over_null_max": (round(ratio_raw, 2)
                                             if ratio_raw else None),
                "motion_mean": float(m_raw.mean()),
                "motion_std": float(m_raw.std()),
            },
        }
        print(f"  motion envelope: mean {m.mean():.3f} std {m.std():.3f}")
        print(f"  peak at lag {lag} frames ({lag * 1000 / FPS:+.0f} ms), "
              f"r = {r:.4f}   [top 40 rows: lag {lag_h}, r = {r_h:.4f}]")
        print(f"  null (block-shuffled audio, {a.shuffles} draws): "
              f"max |r| = {null_max:.4f}, mean |r| = {null_mean:.4f}")
        print(f"  ratio peak/null_max = {ratio}")
        print(f"  raw-motion control: lag {lag_raw}, r = {r_raw:.4f}, "
              f"null max {null_max_raw:.4f}, ratio {ratio_raw}")
        if other:
            print(f"  other-session control: audio from {other['audio_from']} "
                  f"-> peak |r| = {other['abs_r']:.4f}")
        results[tag] = res
        beats_null = (ratio is not None and abs(r) > null_max)
        other_beats = (other is not None and other["abs_r"] >= abs(r))
        conclusion[tag] = {
            "peak_beats_shuffled_null": bool(beats_null),
            "other_session_control_at_least_as_strong": bool(other_beats),
            "raw_motion_control_agrees": bool(
                lag_raw == lag and abs(r_raw - r) < 0.02),
            "verdict": (
                "NO EVIDENCE OF A/V CORRESPONDENCE: the peak correlation does "
                "not beat the block-shuffled null"
                if not beats_null else
                "EVIDENCE OF A/V CORRESPONDENCE: the peak correlation beats "
                "the block-shuffled null")
            + (", and the OTHER-SESSION control is at least as strong as the "
               "real one, so the statistic is picking up the room rather than "
               "the person" if other_beats else "")
            + (". The raw-motion control (no row-band removal) lands on the "
               "same lag and nearly the same r, so the row-band filter is not "
               "what is producing the result"
               if lag_raw == lag and abs(r_raw - r) < 0.02 else ""),
        }
        provenance[tag] = {
            "capture": {
                "path": str(nv.resolve()),
                "sha256": sha256_file(nv),
                "bytes": nv.stat().st_size,
                "layout": layout.name,
                "stride_bytes": layout.stride,
                "n_frames": layout.n_frames,
                "frames_used": int(n_vid),
                "layout_evidence": layout.as_dict(),
            },
            "audio": {
                "path": str(wav.resolve()),
                "sha256": sha256_file(wav),
                "bytes": wav.stat().st_size,
                "seconds": round(dur, 4),
                "samples": int(n_samp),
                "sample_rate": RATE,
            },
        }

    out = {
        "method": ("cross-correlation of the row-band-removed video motion "
                   "envelope against the per-frame audio RMS envelope; the "
                   "null is the peak of the same statistic on block-shuffled "
                   "audio. A raw-motion control (no row-band removal) is "
                   "reported alongside it."),
        "fps": FPS,
        "results": results,
        "conclusion": conclusion,
        "limits": [
            "one session per condition, one room, one person",
            "a correlation peak is evidence that the streams correspond; it is "
            "not an accuracy for any task",
            "the null distribution uses 200 block shuffles of ONE session, so "
            "it describes that session's variability, not the population's",
        ],
        "provenance": {
            "generated_by": "experiments/av_sync.py",
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.executable,
            "command_line": " ".join(sys.argv),
            "cli_args": list(sys.argv[1:]),
            "captures": provenance,
        },
        "supersedes": {
            "why": ("Re-measured on 2026-10-20. The previous run read the "
                    "captures at W*H*3//2 = 23,040 bytes per frame; the "
                    "captures on disk carry 19,200 bytes per frame, so all but "
                    "every sixth frame was torn (rows 0..71 from frame k, rows "
                    "72..95 from frame k+1), and the frame COUNT was wrong too "
                    "(750 instead of 900). The previous numbers are recorded "
                    "here rather than silently overwritten."),
            "previous_stride_bytes": 23040,
            "previous_video_frames": 750,
            "previous_video_seconds": 25.0,
            "previous_duration_mismatch_s": 4.981,
            "previous_results": {
                "A": {"peak_lag_frames": 215, "peak_lag_ms": 7166.7,
                      "peak_r": 0.0513, "peak_r_top40rows": 0.0582,
                      "peak_lag_frames_top40rows": 222,
                      "null_shuffled_max_abs_r": 0.1452,
                      "null_shuffled_mean_abs_r": 0.0707,
                      "ratio_peak_over_null_max": 0.35,
                      "other_session_control": {"audio_from": "B", "peak_r": 0.07394518016877037,
                                                "abs_r": 0.07394518016877037},
                      "motion_mean": 32.0347239678947, "motion_std": 2.3671607035062903,
                      "audio_env_mean": 0.0019325569076717336},
                "B": {"peak_lag_frames": -110, "peak_lag_ms": -3666.7,
                      "peak_r": 0.1181, "peak_r_top40rows": 0.0574,
                      "peak_lag_frames_top40rows": -54,
                      "null_shuffled_max_abs_r": 0.1637,
                      "null_shuffled_mean_abs_r": 0.0811,
                      "ratio_peak_over_null_max": 0.72,
                      "other_session_control": {"audio_from": "A", "peak_r": 0.0677791851306449,
                                                "abs_r": 0.0677791851306449},
                      "motion_mean": 32.63491468985557, "motion_std": 1.6157476373794213,
                      "audio_env_mean": 0.0017740670491771472},
                "C": {"peak_lag_frames": 227, "peak_lag_ms": 7566.7,
                      "peak_r": 0.0724, "peak_r_top40rows": 0.0547,
                      "peak_lag_frames_top40rows": 239,
                      "null_shuffled_max_abs_r": 0.1335,
                      "null_shuffled_mean_abs_r": 0.066,
                      "ratio_peak_over_null_max": 0.54,
                      "other_session_control": {"audio_from": "A", "peak_r": 0.051667944197120286,
                                                "abs_r": 0.051667944197120286},
                      "motion_mean": 33.441730783912696, "motion_std": 2.196553078485423,
                      "audio_env_mean": 0.00031837303993601077},
            },
            "verdict_changed": (
                "The headline changes and it is the point of this file. The "
                "previous run reported video_seconds 25.0 against audio "
                "29.981, a 4.981 s mismatch, and treated the two streams as "
                "possibly unrelated. The 4.981 s was the stride: 17,280,000 "
                "bytes divided by 23,040 gives 750 frames, and divided by the "
                "true 19,200 gives 900. At the correct stride the mismatch is "
                "0.019 s, less than half a video frame, and the video and the "
                "audio are the same length."),
        },
    }
    p = Path(a.out) / "av_sync_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
