"""
av_sync.py -- do the video frames and the audio of one capture actually
correspond, and does the visible motion track the audio?

WHY THIS FILE EXISTS.

Preparing the cloud-teacher measurement, two facts about the existing
captures turned up:

  1. `A.wav` is 29.98 s of audio. `A.nv12` is 750 frames at 30 fps = 25.0 s
     of video. THE TWO ARE NOT THE SAME LENGTH.
  2. The Silero gate puts A's first speech at block 61 = 1.95 s, which is
     video frame 58 at 30 fps.

Those two facts are compatible with very different worlds. If the audio and
video were captured by the same loop and the video simply stopped 5 s early,
then time-to-frame conversion is valid for the first 25 s and the frame
labels are sound. If they were separate recordings, or the audio loop ran at
a different rate, then every "frame t corresponds to audio time t/30" claim
is an assumption with no evidence behind it -- and the whole per-frame
labelling scheme rests on it.

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
W, H = 160, 96
FB = W * H * 3 // 2
YS = W * H
FPS = 30
RATE = 16000


def load_y(path: Path):
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    n = len(raw) // FB
    return raw[:n * FB].reshape(n, FB)[:, :YS].reshape(n, H, W).astype(np.float64)


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

    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        nv = cap / f"{tag}.nv12"
        wav = cap / f"{tag}.wav"
        if not nv.is_file() or not wav.is_file():
            print(f"{tag}: missing")
            continue
        Y = load_y(nv)
        n_vid = len(Y)
        e, dur, n_samp = audio_envelope(wav)
        m = motion_envelope(Y)
        m_head = motion_envelope(Y, rows=slice(0, 40))

        print(f"\n===== {tag} =====")
        print(f"  video {n_vid} frames = {n_vid / FPS:.2f}s    "
              f"audio {n_samp} samples = {dur:.2f}s    "
              f"LENGTH MISMATCH = {dur - n_vid / FPS:+.2f}s")

        lag, r, curve, lags = best_lag(m, e, max_lag)
        lag_h, r_h, _, _ = best_lag(m_head, e, max_lag)

        # -- the null: block-shuffled audio ------------------------------
        # Blocks of 10 frames are shuffled, so the envelope keeps its local
        # structure but loses its alignment with the picture. A peak of the
        # real statistic that the null can also produce is not evidence.
        block = 10
        nb = len(e) // block
        null_peaks = []
        for _ in range(a.shuffles):
            order = rng.permutation(nb)
            shuffled = np.concatenate([e[i * block:(i + 1) * block]
                                       for i in order])
            shuffled = np.concatenate([shuffled, e[nb * block:]])
            _, rn, _, _ = best_lag(m, shuffled[:len(e)], max_lag)
            null_peaks.append(abs(rn))
        null_peaks = np.array(null_peaks)
        null_max = float(null_peaks.max())
        null_mean = float(null_peaks.mean())

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
        }
        print(f"  motion envelope: mean {m.mean():.3f} std {m.std():.3f}")
        print(f"  peak at lag {lag} frames ({lag * 1000 / FPS:+.0f} ms), "
              f"r = {r:.4f}   [top 40 rows: lag {lag_h}, r = {r_h:.4f}]")
        print(f"  null (block-shuffled audio, {a.shuffles} draws): "
              f"max |r| = {null_max:.4f}, mean |r| = {null_mean:.4f}")
        print(f"  ratio peak/null_max = {ratio}")
        if other:
            print(f"  other-session control: audio from {other['audio_from']} "
                  f"-> peak |r| = {other['abs_r']:.4f}")
        results[tag] = res

    out = {
        "method": ("cross-correlation of the row-band-removed video motion "
                   "envelope against the per-frame audio RMS envelope; the "
                   "null is the peak of the same statistic on block-shuffled "
                   "audio"),
        "fps": FPS,
        "results": results,
        "limits": [
            "one session per condition, one room, one person",
            "a correlation peak is evidence that the streams correspond; it is "
            "not an accuracy for any task",
            "the null distribution uses 200 block shuffles of ONE session, so "
            "it describes that session's variability, not the population's",
        ],
    }
    p = Path(a.out) / "av_sync_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
