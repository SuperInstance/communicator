"""
audio_video_alignment.py -- does the video carry the speaking signal, and can
the two recordings be put back in step?

WHY THIS FILE EXISTS.

Two facts about the captures, both of which the repo knows and neither of
which has been measured together:

  1. "the audio and the video of A/B/C are two separate recordings. The old
     captures were made by a script that recorded 30 s of audio and 25 s of
     video, separately, and started them independently" -- the capture
     script's own docstring, `experiments/corpus_capture_win.py`. So the
     audio timeline and the video timeline have an UNKNOWN RELATIVE OFFSET.

  2. The teacher labels whole SESSIONS: A is "speaking", C is "silent". That
     makes the leave-one-session-out fold a ONE-BIT statistic -- the held-out
     session is pure, so the accuracy is 0.000 or 1.000 and nothing in
     between, and a constant "always NOT SPEAKING" predictor scores 1.000 on
     a held-out speaking session. Measured: the blank control scores 1.000
     on every fold of the LOO.

This file measures the thing that fixes both, or reports that it does not
exist:

  - cross-correlate the SPEECH ENVELOPE (from the repo's own VAD, run on the
    audio) against the MOTION ENERGY (from the video's motion maps), over a
    range of relative offsets;
  - report the best offset and the correlation at that offset.

If the correlation is real, then the video does carry the speaking signal
AND the offset is measurable, and the teacher can be given FRAME-LEVEL labels
from the audio -- which turns the LOO fold from a one-bit statistic into a
gradated one that a constant predictor cannot beat.

If the correlation is not real, that is a much more important finding: the
video would carry no usable speaking signal at all, and every teacher result
built on it would be about something else.

THE CONTROLS, WHICH ARE THE POINT.

  scrambled_audio   the speech envelope with its frames permuted. The offset
                    search must collapse to nothing: this is the null for
                    "the speech pattern aligns with the motion".
  wrong_session     A's audio against C's video, and C's audio against A's
                    video. The two recordings are of the same person in the
                    same session, so if the correlation is about the person
                    and not about the session, it survives; if it is about
                    the moment, it does not.
  blank_room        the empty-room capture (`empty1.nv12` / `empty1.wav`),
                    where there is no person to speak and nothing to move.

A statistic that survives all three is not measuring speech.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from src import wavio                            # noqa: E402

SR = 16000
HOP = 512                       # 32 ms: the VAD's own frame size
CAPTURES = Path("/mnt/c/proart_capture")

# The capture frame size. The captures are 160x96; see `maps_from_capture`
# for why the stride cannot be assumed.
FRAME_W, FRAME_H = 160, 96

# The corpus metadata records 600 frames as 20.0 s, so the capture ran at
# 30 fps. The frames and the audio are separate recordings, so this is the
# VIDEO's clock, not a shared one.
VIDEO_FPS = 30.0

# How far to search for the relative offset. The video is 20 s and the audio
# is 30 s, so the video could start anywhere inside the audio.
OFFSET_SEARCH_S = 15.0


def speech_envelope(wav_path: Path, hop: int = HOP) -> tuple:
    """Per-hop speech probability from the repo's own gate.

    Returns (envelope, sr). The gate is `src/vad.py`'s Silero model, which the
    repo already ships at `models/silero_vad.onnx`. It is used here because
    it is the repo's own definition of "speech", not because it is the best
    available.
    """
    from src import vad
    audio = wavio.read_wav(str(wav_path))
    gate = vad.SileroGate()
    probs = gate.push(audio.samples)
    return np.asarray(probs, dtype=np.float64), audio.sample_rate


def motion_envelopes(maps: np.ndarray, n_hops: int, hop: int = HOP,
                     top_frac: float = 0.05) -> dict:
    """Per-frame motion statistics, placed on the 32 ms hop grid.

    Returns a dict of envelopes:

      total    the sum of every pixel's change. The repo's own measurement
               says this DOES NOT distinguish speaking from silent
               ("total change is 3128 / 3209 / 3268 for speaking / speaking /
               SILENT"), so it is included as the NEGATIVE case.
      active   the sum of the top `top_frac` of pixels' change -- the region
               where the change concentrates, and where ROW-BAND-REMOVAL.md
               measured 79% of the speaking/silent difference to live. This
               is the candidate for carrying the speaking signal.
      area     how many pixels are above half the frame's maximum change: a
               measure of how SPREAD OUT the change is.

    THE TWO CLOCKS ARE DIFFERENT, and mixing them up is the obvious way to
    get a correlation of exactly zero. The speech envelope is on 32 ms hops
    (the VAD's own frame size). The motion maps are on the VIDEO's clock,
    30 fps. So each per-frame value is placed at its own timestamp
    (t = i / 30 s) and interpolated onto the hop grid.
    """
    names = ("total", "active", "area")
    if len(maps) == 0 or n_hops < 2:
        return {k: np.zeros(max(0, n_hops)) for k in names}
    flat = maps.reshape(len(maps), -1).astype(np.float64)
    total = flat.sum(axis=1)
    k = max(1, int(flat.shape[1] * top_frac))
    part = np.partition(flat, -k, axis=1)[:, -k:]
    active = part.sum(axis=1)
    area = (flat > 0.5 * flat.max(axis=1, keepdims=True)).sum(axis=1)
    t_frame = np.arange(len(maps)) / VIDEO_FPS
    t_hop = np.arange(n_hops) * (hop / SR)
    out = {}
    for name, vals in (("total", total), ("active", active), ("area", area)):
        out[name] = np.interp(t_hop, t_frame, vals, left=vals[0], right=vals[-1])
    return out


def detrend(env: np.ndarray, window_s: float = 2.0) -> np.ndarray:
    """Remove the slow trend from an envelope.

    WITHOUT THIS, THE MEASUREMENT IS WORTHLESS, and the first version of this
    script proved it: the raw correlation of the speech envelope against the
    motion envelope was +0.26 for a WRONG-session control, which is the
    largest number in the run. Both envelopes drift -- the corpus's own
    metadata records a drift of 32 counts after row removal, and the speech
    envelope has a slow rise and fall over a take -- and two drifting signals
    correlate whether or not they have anything to do with each other.

    So the slow trend is removed from both before they are compared, and the
    detrended correlation is the one that means anything. The window is
    2 s, which is longer than a pause and shorter than a take.
    """
    if len(env) < 10:
        return env - env.mean() if len(env) else env
    w = max(3, int(window_s / (HOP / SR)))
    kernel = np.ones(w) / w
    trend = np.convolve(env, kernel, mode="same")
    # Fix the edges: 'same' convolution zero-pads, which drags the edges down.
    half = w // 2
    trend[:half] = trend[half]
    trend[-half:] = trend[-half - 1]
    return env - trend


def correlate(env_a: np.ndarray, env_b: np.ndarray, max_lag: int,
             do_detrend: bool = True) -> dict:
    """The correlation of two envelopes at every relative offset.

    A POSITIVE lag means the motion envelope leads the speech envelope: the
    motion happens first. The sign convention is stated because getting it
    backwards turns a real offset into a wrong one.

    The envelopes are detrended before the comparison unless `do_detrend` is
    False, and BOTH numbers are reported, because the raw one is the number
    that gets quoted by accident.
    """
    m = min(len(env_a), len(env_b))
    if m < 50:
        return {"best_lag": 0, "best_r": 0.0, "lags_tried": 0}
    a = env_a[:m]
    b = env_b[:m]
    if do_detrend:
        a, b = detrend(a), detrend(b)
    a = a - a.mean()
    b = b - b.mean()
    na = float(np.linalg.norm(a)) or 1.0
    nb = float(np.linalg.norm(b)) or 1.0
    a = a / na
    b = b / nb
    best_lag, best_r, tried = 0, -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[:m - lag]
        else:
            x, y = a[:m + lag], b[-lag:]
        if len(x) < 50:
            continue
        r = float(np.dot(x, y))
        tried += 1
        if r > best_r:
            best_r, best_lag = r, lag
    return {"best_lag": int(best_lag), "best_r": round(best_r, 4),
            "lags_tried": tried}


def at_offset(env_a: np.ndarray, env_b: np.ndarray, lag: int) -> float:
    """The correlation at one specific offset."""
    m = min(len(env_a), len(env_b))
    if m < 50:
        return 0.0
    if lag >= 0:
        x, y = env_a[lag:m], env_b[:m - lag]
    else:
        x, y = env_a[:m + lag], env_b[-lag:m]
    if len(x) < 50:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    dx = float(np.linalg.norm(x))
    dy = float(np.linalg.norm(y))
    if dx == 0 or dy == 0:
        return 0.0
    return float(np.dot(x, y) / (dx * dy))




def speech_envelope_hop(wav_path: Path) -> np.ndarray:
    """The speech probability envelope on the 32 ms grid.

    One value per HOP samples of audio, straight from the gate. The gate's
    own frame size is 512 samples at 16 kHz, which is exactly HOP.
    """
    probs, _sr = speech_envelope(wav_path)
    return np.asarray(probs, dtype=np.float64)


def load_maps(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path)
    return d["maps"]


def maps_from_capture(nv12_path: Path) -> tuple[np.ndarray, dict]:
    """The motion maps of a capture, read at the layout the file actually has.

    READ THIS BEFORE TRUSTING ANY NUMBER IN THIS FILE.

    The captures are NOT standard NV12 frames. They are 19,200 bytes each
    (luma 160x96, then a U plane and a V plane at 80x24), written by
    `nv12small.py`. Reading them at the usual 23,040 bytes produces torn
    frames -- rows 0..71 from one moment and rows 72..95 from the next --
    and 80% of every "frame" is a splice. The committed version of this
    measurement was made on exactly those torn frames, and its numbers are
    withdrawn in `docs/research/NV12-STRIDE.md`.

    `src.nv12.read_luma` detects the layout from the content and reports the
    margin over the runner-up, so the frame count and the stride are
    recorded rather than assumed.
    """
    from src import nv12
    luma, layout = nv12.read_luma(nv12_path, width=FRAME_W, height=FRAME_H)
    Y = luma.astype(np.float64)
    prof = np.median(Y, axis=2)
    g = prof.mean(axis=1, keepdims=True)
    Yr = Y - (prof - g)[:, :, None]
    d = np.abs(np.diff(Yr, axis=0))
    maps = d.reshape(d.shape[0], FRAME_H // 2, 2, FRAME_W // 2, 2).mean(axis=(2, 4))
    return maps, layout.as_dict()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--captures", default=str(CAPTURES))
    ap.add_argument("--out", default=str(ROOT / "experiments" /
                                         "audio_video_alignment_results.json"))
    args = ap.parse_args(argv)
    caps = Path(args.captures)

    max_lag = int(OFFSET_SEARCH_S * SR / HOP)

    speech, motion, layouts = {}, {}, {}
    for name in ("A", "B", "C"):
        wav = caps / f"{name}.wav"
        nv = caps / f"{name}.nv12"
        if not (wav.is_file() and nv.is_file()):
            continue
        s_env = speech_envelope_hop(wav)
        speech[name] = s_env
        maps, layouts[name] = maps_from_capture(nv)
        motion[name] = motion_envelopes(maps, len(s_env))

    results = {"video_fps": VIDEO_FPS, "hop_samples": HOP,
               "hop_ms": int(HOP / SR * 1000),
               "offset_search_s": OFFSET_SEARCH_S,
               "envelopes": ("total", "active", "area"),
               "frame_layouts": layouts,
               "sessions": {}, "controls": {}}

    def compare(spk, mov, label, bucket):
        row = {}
        for name in ("total", "active", "area"):
            raw = correlate(spk, mov[name], max_lag, do_detrend=False)
            det = correlate(spk, mov[name], max_lag, do_detrend=True)
            row[name] = {
                "raw_best_r": raw["best_r"],
                "raw_best_lag_s": round(raw["best_lag"] * HOP / SR, 3),
                "detrended_best_r": det["best_r"],
                "detrended_best_lag_s": round(det["best_lag"] * HOP / SR, 3),
            }
        bucket[label] = row
        return row

    for name in speech:
        if name not in motion:
            continue
        row = compare(speech[name], motion[name], name, results["sessions"])
        row["speech_mean"] = round(float(speech[name].mean()), 4)
        row["n_speech_hops"] = int(len(speech[name]))

    # --- the controls ----------------------------------------------------
    rng = np.random.default_rng(7)
    if "A" in speech:
        perm = rng.permutation(len(speech["A"]))
        compare(speech["A"][perm], motion["A"], "scrambled_audio_A",
                results["controls"])
    if "A" in speech and "C" in motion:
        compare(speech["A"], motion["C"], "wrong_session_A_audio_C_video",
                results["controls"])
        compare(speech["C"], motion["A"], "wrong_session_C_audio_A_video",
                results["controls"])
    empty_wav = caps / "empty1.wav"
    empty_nv = caps / "empty1.nv12"
    if empty_wav.is_file() and empty_nv.is_file():
        s_env = speech_envelope_hop(empty_wav)
        maps, layouts["empty1"] = maps_from_capture(empty_nv)
        mov = motion_envelopes(maps, len(s_env))
        compare(s_env, mov, "blank_room", results["controls"])

    # --- the headline ----------------------------------------------------
    # The candidate signal is the ACTIVE envelope (the top 5% of pixels,
    # where 79% of the speaking/silent difference lives). The TOTAL envelope
    # is the negative case: the repo has already measured that total change
    # does not distinguish speaking from silent, so a correlation there would
    # mean the measurement is about drift and not about speech.
    real = [v["active"]["detrended_best_r"] for v in results["sessions"].values()]
    ctrl = [v["active"]["detrended_best_r"]
            for v in results["controls"].values()]
    total_real = [v["total"]["detrended_best_r"]
                  for v in results["sessions"].values()]
    results["headline"] = {
        "active_real_sessions": real,
        "active_controls": ctrl,
        "total_real_sessions": total_real,
        "margin_over_the_worst_control": (
            round(min(real) - max(ctrl), 4) if (real and ctrl) else None),
        "the_video_carries_a_time_aligned_speaking_signal": (
            None if not (real and ctrl) else bool(min(real) > max(ctrl))),
        "reading": ("the ACTIVE envelope is the candidate: the top 5% of "
                    "pixels, where ROW-BAND-REMOVAL.md measured 79% of the "
                    "speaking/silent difference. The TOTAL envelope is the "
                    "negative case: total change does not distinguish "
                    "speaking from silent, so a correlation there would be "
                    "about drift."),
    }

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2) + "\n")
    for name, row in results["sessions"].items():
        print(f"{name}:")
        for env in ("total", "active", "area"):
            d = row[env]
            print(f"   {env:7s} raw {d['raw_best_r']:+.4f}  detrended "
                  f"{d['detrended_best_r']:+.4f} at "
                  f"{d['detrended_best_lag_s']:+.3f} s")
    print("controls:")
    for name, row in results["controls"].items():
        print(f"   {name}: active detrended "
              f"{row['active']['detrended_best_r']:+.4f}, total detrended "
              f"{row['total']['detrended_best_r']:+.4f}")
    print(json.dumps(results["headline"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
