"""
mouth_aperture_alignment.py -- does the MOUTH's motion, measured properly,
track the AUDIO's speech in time?

WHY THIS FILE EXISTS.

`docs/research/TEACHER-LOO-FOLDS.md` §3 measured the video's motion against
the audio's speech and found no alignment above the nulls: A +0.0776 and
B +0.0247 against a scrambled-audio null of +0.0608. That is the finding that
would cost the recording plan the most, so it deserves the hardest test this
repository can give it -- and §3's own caveat names the weakness:

    "The caveat on (3) is that the whole-frame motion energy is a blunt
     instrument. The frame is 160x96 greyscale and the whole-scene
     illumination noise and the person's shoulders can swamp a mouth that
     occupies maybe 20x10 pixels. So (3) may be a statement about the
     instrument, not about the video."

So this file replaces the blunt instrument with a specific one. A face
landmark model gives the MOUTH's geometry per frame; the mouth aperture
(the vertical distance between the inner lips) is then compared against the
same speech envelope, with the same statistic, over the same lag search, with
the same controls.

THE ANSWER, AND IT IS NOT THE OBVIOUS ONE.

The capture files are NOT the frames the repository thinks they are.

    /mnt/c/proart_capture/{A,B,C}.nv12 is NOT 750 NV12 frames of 160x96.
    Each record is 19200 bytes: 15360 bytes of 160x96 luma, followed by
    3840 bytes of INTERLEAVED U,V chroma (80x24 samples).

    > CORRECTION (2026-10-20). This paragraph used to say the trailing
    > 3,840 bytes were "the U chroma plane (80x48). There is no V plane."
    > That is wrong on both counts. The chroma is INTERLEAVED -- U at even
    > offsets, V at odd -- and it is 80x24 samples, half the VERTICAL
    > resolution of a standard NV12 frame's chroma (80x48), which is exactly
    > why the frame is 20% shorter. The writer builds it as
    > `uv[:, 0] = u; uv[:, 1] = v` on an (N, 2) array, which in row-major
    > order is U, V, U, V. The stride of 19,200 -- the load-bearing
    > conclusion of this lane -- is unaffected, and so is every number
    > below, because the luma is the first 15,360 bytes either way.

The repository reads these files with a 23040-byte stride -- the full NV12
frame size -- which is `experiments/seed_corpus_from_captures.py:41-42`,
`experiments/local_vlm_vision.py:116-117` and the brief this lane was given.
That stride slides 24 rows per record, so every "frame" is TORN between two
real frames and the scene appears to jump vertically by a quarter of its
height on every single frame.

Measured, not asserted. Note that the 19,200 row correlates the WHOLE
record (luma + chroma) reshaped as 160x120 -- it is a stride DETECTOR, not
a luma reading, and the luma this lane analyses is always the first 15,360
bytes:

    stride 15360 (H=96)   frame-to-frame correlation  -0.011
    stride 19200 (H=120)  frame-to-frame correlation  +0.999   <-- the truth
    stride 23040 (H=96)   frame-to-frame correlation  -0.025   <-- the repo's

And the seeded corpus maps -- the ones §2's static-template analysis and §3's
whole-frame alignment were both computed from -- correlate with the torn
reading at 0.9985 and with the correct reading at -0.005. Their mean motion
map is FLAT across all 48 rows (54-62 everywhere), which is what a tear
artefact looks like and is not what a face looks like.

The consequences for THIS lane, which is why it matters here:

  1. The frames are 900, not 750. 900 frames at 30 fps is 30.00 s, and the
     audio is 29.98 s. The empty room is 4500 frames = 150.00 s against
     149.91 s of audio. The frame rate is therefore 30 fps for THESE
     captures, which is what `audio_video_alignment.py` assumed, and what
     `CAPTURE-NEXT.md`'s 25 fps note (about a later round) does not apply to.
  2. On the correctly-read frames the landmark model detects a face in
     900/900 frames and the landmarks stop jittering (forehead-y standard
     deviation 1.09 px, against 27 px on the torn frames). On the torn
     frames the detection rate was 62-88% and the "face" was jumping around
     the frame. Every landmark number in this file is from the correct
     reading.
  3. The whole-frame motion instrument of §3 was measuring the tear, not the
     person. So §3's conclusion cannot be carried forward as it stands.

THE CONTROLS, WHICH ARE THE POINT.

  scrambled_audio   the speech envelope with its hops permuted, run through
                    the SAME max-over-lags statistic. This is the null for
                    "the mouth's pattern aligns with the speech".
  wrong_session     A's audio against C's video and C's audio against A's
                    video. If the number is about the person rather than the
                    moment, it survives; if it is about the moment, it does
                    not.
  eye_distance      THE SAME STATISTIC on a landmark that must NOT track
                    speech: the distance between the outer eye corners,
                    normalised the same way. If the eye distance tracks the
                    speech as well as the mouth does, the correlation is an
                    artefact of the whole frame moving and not of the mouth.
  empty_room        the empty-room capture, where there is no person and
                    therefore no mouth to speak.
  degenerate        the search's own inflation: the same max-over-lags
                    statistic applied to a signal against ITSELF, so the
                    reader can see what the statistic returns when the answer
                    is "yes", and how much the 937-lag search inflates a
                    number by chance.

A statistic that survives all of them is not measuring speech.

WHAT IS IMPORTED, AND WHY.

The speech envelope, the detrending, the max-over-lags statistic and the
lag convention are IMPORTED from `experiments/audio_video_alignment.py`, so
that the numbers here and the numbers in TEACHER-LOO-FOLDS.md §3 are the
same statistic on the same grid, not two statistics that look alike. That
file's SHA-256 is recorded in the results, because a result that depends on
another file must say which version of that file it depended on.

A number measured one way is one measurement, not a property. So the
torn-frame reading is ALSO run, as a labelled sensitivity row: it is the
number the repository's own frame reader would produce, and it is recorded
so the reader can see the difference rather than take it on trust.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

# The statistic and the speech envelope, imported so that this lane and
# TEACHER-LOO-FOLDS.md §3 are literally the same measurement.
from audio_video_alignment import (  # noqa: E402
    HOP,
    OFFSET_SEARCH_S,
    SR,
    correlate,
    detrend,
    speech_envelope_hop,
)

CAPTURES = Path("/mnt/c/proart_capture")

# ---------------------------------------------------------------- the frame
#
# THE RECORD LAYOUT, MEASURED. See the docstring.
#
#     A record is 19200 bytes:
#         15360 bytes of luma, 160 wide by 96 tall, greyscale
#         3840 bytes of INTERLEAVED U,V chroma, 80 wide by 24 tall
#
#     There is no SEPARATE V plane -- the V samples are interleaved with
#     the U samples, one byte each -- so the file is NOT standard NV12 and
#     the stride is NOT 23040. The 4,080 extra bytes that a 23040-byte
#     stride exposes are a mix of luma and chroma, which is why they read
#     as a constant bright band (neutral chroma is 128; measured mean
#     129.9).

RECORD_BYTES = 19200
LUMA_BYTES = 15360
CHROMA_BYTES = 3840
FRAME_W = 160
FRAME_H = 96

# The stride the REPOSITORY uses, and the one the brief for this lane
# specified. Kept so the sensitivity row can be produced from the same code.
TORN_RECORD_BYTES = 23040
TORN_FRAME_H = 96

# The achieved frame rate for THESE captures. Not assumed: 900 frames and a
# 29.981 s recording agree at 30 fps (30.00 s) and disagree by 6 s at
# 25 fps (36.0 s); the empty room agrees at 30 fps (150.00 s against
# 149.906 s) and disagrees by 30 s at 25 fps.
VIDEO_FPS = 30.0

# ------------------------------------------------------------- the landmark
#
# MediaPipe FaceMesh's canonical 468-landmark mesh, plus the 10 iris
# landmarks that `refine_landmarks=True` adds. The indices used here are
# recorded in the results file and their mean positions in ORIGINAL frame
# coordinates are recorded too, so a reader can check that landmark 13 is
# the upper inner lip and not something else.
LANDMARKS = {
    "upper_inner_lip": 13,
    "lower_inner_lip": 14,
    "mouth_corner_right": 61,
    "mouth_corner_left": 291,
    "nose_tip": 1,
    "chin_bottom": 152,
    "forehead_top": 10,
    "eye_outer_right": 33,
    "eye_outer_left": 263,
}

# The landmark model's own settings. Recorded because they are part of the
# measurement: `static_image_mode=True` means every frame is detected
# independently, with no temporal state carried between frames.
FACE_MESH = {
    "model": "MediaPipe FaceMesh (mp.solutions.face_mesh.FaceMesh)",
    "static_image_mode": True,
    "max_num_faces": 1,
    "refine_landmarks": True,
    "min_detection_confidence": 0.05,
    "min_tracking_confidence": 0.05,
    "n_landmarks": 478,
}

# The upscale factor the frames are fed to the model at. The face is
# detected in every frame at 1x, 4x and 6x (recorded as a sensitivity
# row), so this is a resolution choice, not a necessity.
UPSCALE = 4
UPSCALE_SENSITIVITY = (1, 6)

# The seed for the scrambled-audio permutations.
SCRAMBLE_SEED = 20261020
N_SCRAMBLES = 25

# The number of permutations used for the scrambled-audio null, and the
# windows used for the degenerate self-correlation baselines.
N_RANDOM_LAGS = 20


# ------------------------------------------------------------------- helpers

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_luma(path: Path) -> np.ndarray:
    """The luma plane of every record, as (n_frames, 96, 160) uint8.

    The record stride is 19200 bytes, NOT the 23040 that a full NV12 frame
    would be. Read with 23040 and every frame is torn between two real
    frames, with the scene jumping 24 rows per frame; the frame-to-frame
    correlation of the torn reading is -0.025 against +0.999 for this one.
    """
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    n = len(raw) // RECORD_BYTES
    rec = raw[:n * RECORD_BYTES].reshape(n, RECORD_BYTES)
    return rec[:, :LUMA_BYTES].reshape(n, FRAME_H, FRAME_W).copy()


def load_luma_torn(path: Path) -> np.ndarray:
    """The reading the REPOSITORY's frame reader produces. Do not trust it.

    Kept so the sensitivity row is generated by the same code as the main
    one, and so the reader can see the difference rather than take it on
    trust. This is `experiments/seed_corpus_from_captures.py`'s geometry.
    """
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    n = len(raw) // TORN_RECORD_BYTES
    rec = raw[:n * TORN_RECORD_BYTES].reshape(n, TORN_RECORD_BYTES)
    return rec[:, :LUMA_BYTES].reshape(n, TORN_FRAME_H, FRAME_W).copy()


def face_mesh_landmarks(frames: np.ndarray, upscale: int = UPSCALE):
    """Per-frame FaceMesh landmarks in ORIGINAL frame coordinates.

    Returns (landmarks, detected) where `landmarks` is (n, 478, 2) float64
    in the 160x96 frame's own pixel coordinates and `detected` is a boolean
    array saying which frames produced a face.

    The model is imported INSIDE this function so that this module can be
    imported by a test that has no mediapipe and no model files.
    """
    import cv2
    import mediapipe as mp

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=FACE_MESH["static_image_mode"],
        max_num_faces=FACE_MESH["max_num_faces"],
        refine_landmarks=FACE_MESH["refine_landmarks"],
        min_detection_confidence=FACE_MESH["min_detection_confidence"],
        min_tracking_confidence=FACE_MESH["min_tracking_confidence"],
    )
    out = np.full((len(frames), FACE_MESH["n_landmarks"], 2), np.nan)
    detected = np.zeros(len(frames), dtype=bool)
    try:
        for i in range(len(frames)):
            img = frames[i]
            if upscale != 1:
                img = cv2.resize(img, (FRAME_W * upscale, FRAME_H * upscale),
                                 interpolation=cv2.INTER_CUBIC)
            rgb = np.repeat(img[:, :, None], 3, axis=2)
            res = mesh.process(rgb)
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            for k in range(len(lm)):
                out[i, k, 0] = lm[k].x * FRAME_W
                out[i, k, 1] = lm[k].y * FRAME_H
            detected[i] = True
    finally:
        mesh.close()
    return out, detected


def landmark_signals(lm: np.ndarray, detected: np.ndarray) -> dict:
    """The per-frame mouth geometry, normalised by the face height.

    Every signal is divided by the face height (forehead landmark 10 to
    chin landmark 152) so that it is scale-free: the person's distance from
    the camera must not be able to produce a correlation.

    mouth_aperture   the vertical distance between the inner lips (13, 14).
                     THIS IS THE HEADLINE SIGNAL.
    mouth_width      the horizontal distance between the mouth corners
                     (61, 291).
    jaw_open         the distance from the nose tip to the chin (1, 152),
                     which grows when the jaw drops.
    eye_distance     the distance between the outer eye corners (33, 263).
                     THE CONTROL: this landmark must NOT track speech.

    A signal that is undefined for every frame comes back as an array of
    NaN, and the caller records that fact rather than a number.
    """
    idx = LANDMARKS

    def pt(k):
        return lm[:, idx[k], :]

    face_height = np.linalg.norm(pt("forehead_top") - pt("chin_bottom"), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "mouth_aperture": np.linalg.norm(pt("upper_inner_lip")
                                             - pt("lower_inner_lip"), axis=1)
                                / face_height,
            "mouth_width": np.linalg.norm(pt("mouth_corner_right")
                                          - pt("mouth_corner_left"), axis=1)
                           / face_height,
            "jaw_open": np.linalg.norm(pt("nose_tip") - pt("chin_bottom"),
                                       axis=1) / face_height,
            "eye_distance": np.linalg.norm(pt("eye_outer_right")
                                           - pt("eye_outer_left"), axis=1)
                            / face_height,
        }
    # The raw pixel aperture, because a normalised ratio can hide a signal
    # that the normalisation itself is compressing. Recorded, and correlated
    # as a robustness row.
    out["mouth_aperture_px"] = np.linalg.norm(pt("upper_inner_lip")
                                              - pt("lower_inner_lip"), axis=1)
    out["face_height_px"] = face_height
    for k, v in out.items():
        v[~detected] = np.nan
    return out


def fill_gaps(sig: np.ndarray):
    """Fill the frames with no face by interpolating over the ones that have
    one, and report how much was filled.

    A long gap would be a fabrication, so the caller records the gap
    length. On the correctly-read frames there are no gaps at all: the
    detector fires in 900/900.
    """
    good = np.isfinite(sig)
    n = len(sig)
    if not good.any():
        return np.full(n, np.nan), {"n_detected": 0, "n_filled": n,
                                     "max_gap_frames": n, "n_frames": n}
    idx = np.arange(n)
    filled = np.interp(idx, idx[good], sig[good])
    # the longest run of frames that had to be filled
    gaps, run = [], 0
    for g in good:
        if g:
            if run:
                gaps.append(run)
            run = 0
        else:
            run += 1
    if run:
        gaps.append(run)
    return filled, {"n_detected": int(good.sum()), "n_filled": int(n - good.sum()),
                    "max_gap_frames": int(max(gaps)) if gaps else 0,
                    "n_frames": int(n)}


def to_hop_grid(sig: np.ndarray, fps: float, n_hops: int) -> np.ndarray:
    """Put a per-frame signal onto the 32 ms hop grid.

    THE TWO CLOCKS ARE DIFFERENT and mixing them up is the obvious way to
    get a correlation of exactly zero. The video is on its own frame clock
    (30 fps here); the speech envelope is on the gate's own 32 ms hop grid.
    Each per-frame value is placed at its own timestamp and interpolated
    onto the hop grid. The same convention `audio_video_alignment.py` uses
    for the motion envelopes, in the other direction.
    """
    t_frame = np.arange(len(sig)) / fps
    t_hop = np.arange(n_hops) * (HOP / SR)
    return np.interp(t_hop, t_frame, sig, left=sig[0], right=sig[-1])


def at_offset_detrended(env_a: np.ndarray, env_b: np.ndarray, lag: int) -> float:
    """The correlation at ONE offset, with the same detrending as the search.

    `audio_video_alignment.at_offset` does NOT detrend, and the numbers in
    TEACHER-LOO-FOLDS.md §3 are the detrended ones, so the comparison
    against them has to be detrended too.
    """
    m = min(len(env_a), len(env_b))
    if m < 50:
        return 0.0
    a, b = detrend(env_a[:m]), detrend(env_b[:m])
    if lag >= 0:
        x, y = a[lag:m], b[:m - lag]
    else:
        x, y = a[:m + lag], b[-lag:m]
    if len(x) < 50:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    dx = float(np.linalg.norm(x))
    dy = float(np.linalg.norm(y))
    if dx == 0 or dy == 0:
        return 0.0
    return float(np.dot(x, y) / (dx * dy))


def degenerate_baselines(env: np.ndarray, label: str) -> dict:
    """The search's own inflation, on a signal against ITSELF.

    TWO NUMBERS, and they answer different questions.

    self_vs_self_random_lag_r   the correlation of the signal against a
        copy of itself shifted by a random lag, at that lag. It is the
        signal's own autocorrelation at a random separation -- the ordinary
        number, with no search over it.

    self_vs_self_max_over_lags_r   the SAME max-over-lags search applied
        to (signal, that shifted copy). It finds the alignment back, so it
        approaches 1.0. THIS is the scale of the search's inflation: 937
        lags is 937 chances to find a good-looking offset, and this is
        what the statistic returns when the two signals really are the same
        signal.
    """
    rng = np.random.default_rng(SCRAMBLE_SEED)
    m = len(env)
    max_lag = int(OFFSET_SEARCH_S * SR / HOP)
    point, searched = [], []
    for _ in range(N_RANDOM_LAGS):
        lag0 = int(rng.integers(-max_lag, max_lag + 1))
        rolled = np.roll(env, lag0)
        point.append(at_offset_detrended(env, rolled, lag0))
        c = correlate(env, rolled, max_lag)
        searched.append(c["best_r"])
    return {
        "signal": label,
        "n_random_lags": N_RANDOM_LAGS,
        "self_vs_self_random_lag_r_mean": round(float(np.mean(point)), 4),
        "self_vs_self_random_lag_r_max": round(float(np.max(point)), 4),
        "self_vs_self_max_over_lags_r_mean": round(float(np.mean(searched)), 4),
        "self_vs_self_max_over_lags_r_min": round(float(np.min(searched)), 4),
    }


def scrambled_null(env_a: np.ndarray, env_b: np.ndarray, max_lag: int,
                   n: int = N_SCRAMBLES) -> dict:
    """The null: the speech envelope's hops permuted, same statistic.

    The SAME max-over-lags statistic, on the same grid, over the same lag
    range. The permutation keeps every marginal statistic of the envelope
    (its distribution, its mean, its total speech) and destroys only its
    TIMING, which is exactly what the alignment claim is about.
    """
    rng = np.random.default_rng(SCRAMBLE_SEED)
    rs = []
    for _ in range(n):
        perm = rng.permutation(len(env_a))
        rs.append(correlate(env_a[perm], env_b, max_lag)["best_r"])
    rs = np.asarray(rs, dtype=np.float64)
    return {
        "n_permutations": int(n),
        "seed": SCRAMBLE_SEED,
        "best_r_mean": round(float(rs.mean()), 4),
        "best_r_median": round(float(np.median(rs)), 4),
        "best_r_min": round(float(rs.min()), 4),
        "best_r_max": round(float(rs.max()), 4),
        "best_r_values": [round(float(v), 4) for v in rs],
    }


def row(sig_name: str, env_signal: np.ndarray, env_speech: np.ndarray,
        max_lag: int) -> dict:
    """One comparison: the signal against the speech, over the lag search."""
    c = correlate(env_speech, env_signal, max_lag)
    return {
        "signal": sig_name,
        "mean": round(float(np.mean(env_signal)), 6),
        "sd": round(float(np.std(env_signal)), 6),
        "best_r": c["best_r"],
        "best_lag_hops": c["best_lag"],
        "best_lag_s": round(c["best_lag"] * HOP / SR, 3),
        "lags_tried": c["lags_tried"],
    }


def landmark_identity(lm: np.ndarray, detected: np.ndarray) -> dict:
    """Where the landmarks actually are, in ORIGINAL frame coordinates.

    A number measured off landmark 13 is only meaningful if landmark 13 IS
    the upper inner lip. This records the mean position of every landmark
    used, so a reader can check it.
    """
    out = {}
    for name, k in LANDMARKS.items():
        pts = lm[detected, k, :]
        if len(pts) == 0:
            out[name] = {"index": k, "n": 0}
            continue
        out[name] = {
            "index": k,
            "n": int(len(pts)),
            "mean_x_px": round(float(pts[:, 0].mean()), 2),
            "mean_y_px": round(float(pts[:, 1].mean()), 2),
            "sd_x_px": round(float(pts[:, 0].std()), 2),
            "sd_y_px": round(float(pts[:, 1].std()), 2),
        }
    return out


def row_band_remove(ys: np.ndarray) -> np.ndarray:
    """Remove the row-constant component (the LED flicker banding).

    Copied from `experiments/corpus_capture_win.py` (which is the capture
    script) rather than imported, because that module runs a capture when
    imported. The algorithm is three lines and is the same one the corpus
    was seeded with.
    """
    prof = np.median(ys, axis=2)
    g = prof.mean(axis=1, keepdims=True)
    return ys - (prof - g)[:, :, None]


def change_maps(ys: np.ndarray, height: int, width: int) -> np.ndarray:
    """Per-frame |frame(t+1) - frame(t)|, row-band-removed, downsampled 2x.

    The same map the corpus was seeded with, so that the whole-frame
    motion instrument of TEACHER-LOO-FOLDS.md §3 can be RE-MEASURED on the
    correct frames and compared like with like.
    """
    d = np.abs(np.diff(ys, axis=0))
    return d.reshape(d.shape[0], height // 2, 2, width // 2, 2).mean(axis=(2, 4))


def motion_envelopes(maps: np.ndarray, n_hops: int) -> dict:
    """The whole-frame motion envelopes, on the hop grid.

    Same three envelopes as `audio_video_alignment.motion_envelopes`:
    `total` (every pixel), `active` (the top 5% of pixels) and `area`.
    Recomputed here on the CORRECT frames, because §3's version was
    computed on torn ones.
    """
    flat = maps.reshape(len(maps), -1).astype(np.float64)
    total = flat.sum(axis=1)
    k = max(1, int(flat.shape[1] * 0.05))
    active = np.partition(flat, -k, axis=1)[:, -k:].sum(axis=1)
    area = (flat > 0.5 * flat.max(axis=1, keepdims=True)).sum(axis=1)
    t_frame = np.arange(len(maps)) / VIDEO_FPS
    t_hop = np.arange(n_hops) * (HOP / SR)
    out = {}
    for name, vals in (("total", total), ("active", active), ("area", area)):
        out[name] = np.interp(t_hop, t_frame, vals, left=vals[0], right=vals[-1])
    return out


def stride_evidence(path: Path) -> dict:
    """Evidence for the stride, measured on this file rather than asserted.

    Two numbers per candidate stride: the correlation between consecutive
    frames, and the mean absolute difference between them. A stride that
    slices the frame boundary makes the scene jump a quarter of its height
    every frame, and both numbers say so loudly.
    """
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    out = {}
    for name, stride, height in (
            ("correct_19200_luma160x96", RECORD_BYTES, FRAME_H),
            ("repo_assumed_23040_luma160x96", TORN_RECORD_BYTES, TORN_FRAME_H)):
        n = len(raw) // stride
        rec = raw[:n * stride].reshape(n, stride)
        Y = rec[:, :LUMA_BYTES].reshape(n, height, FRAME_W).astype(np.float64)
        if n < 10:
            out[name] = {"n_frames": int(n)}
            continue
        corr = [float(np.corrcoef(Y[i - 1].ravel(), Y[i].ravel())[0, 1])
                for i in range(1, min(n, 300))]
        mad = float(np.abs(np.diff(Y, axis=0)).mean())
        out[name] = {
            "stride_bytes": stride,
            "n_frames": int(n),
            "mean_consecutive_abs_diff": round(mad, 4),
            "mean_consecutive_correlation": round(float(np.mean(corr)), 4),
        }
    out["reading"] = (
        "the correct stride makes consecutive frames nearly identical "
        "(correlation near 1, mean consecutive difference near zero). The "
        "23040-byte stride that the repository uses tears every frame "
        "between two real frames, so its consecutive difference is two "
        "orders of magnitude larger and its correlation is near zero.")
    return out


def head_motion_signals(lm: np.ndarray, detected: np.ndarray) -> dict:
    """Where the face is in the frame, in pixels.

    head_cx, head_cy   the centroid of every landmark, i.e. where the face
                       IS. A translation of the whole head.
    head_scale         the face height in pixels, i.e. how big the face is.

    These are recorded because the eye-distance control needs an
    explanation: if the eye distance tracks the speech, something is moving
    the whole head, and these say what and how much.
    """
    cen = np.nanmean(lm, axis=1)
    out = {"head_cx": cen[:, 0].copy(), "head_cy": cen[:, 1].copy()}
    for k, v in out.items():
        v[~detected] = np.nan
    return out


def partial_residual(y: np.ndarray, controls: list) -> np.ndarray:
    """The part of `y` that the control signals do not explain.

    A least-squares fit of `y` on the controls, subtracted off. If the
    mouth aperture's correlation with the speech survives this, the
    correlation is not merely the head moving.
    """
    m = min(len(y), *(len(c) for c in controls))
    yy = detrend(y[:m])
    A = np.column_stack([np.ones(m)] + [detrend(c[:m]) for c in controls])
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A @ beta
    total = float(np.var(yy))
    return resid, {"variance_explained": round(1.0 - float(np.var(resid)) / total, 4)
                   if total > 0 else None}


def split_half_check(env_speech: np.ndarray, env_signal: np.ndarray,
                     max_lag: int) -> dict:
    """Does the alignment REPLICATE on the two halves of the recording?

    A max-over-lags statistic over 937 lags will find a good-looking
    offset even when there is nothing there. The way to tell a real
    alignment from a lucky one is to ask whether the SAME offset works on
    data the search did not choose: the correlation is computed at the
    full recording's best lag, separately on each half, and each half is
    given its OWN scrambled null to beat.
    """
    n = len(env_speech)
    h = n // 2
    full = correlate(env_speech, env_signal, max_lag)
    lag = full["best_lag"]
    halves = {}
    for label, sp_h, sig_h in (("first_half", env_speech[:h], env_signal[:h]),
                               ("second_half", env_speech[h:], env_signal[h:])):
        r_at = at_offset_detrended(sp_h, sig_h, lag)
        own = correlate(sp_h, sig_h, max_lag)
        null = scrambled_null(sp_h, sig_h, max_lag, n=N_SCRAMBLES)
        halves[label] = {
            "r_at_the_full_best_lag": round(r_at, 4),
            "own_best_r": own["best_r"],
            "own_best_lag_s": round(own["best_lag"] * HOP / SR, 3),
            "scrambled_null_mean": null["best_r_mean"],
            "scrambled_null_max": null["best_r_max"],
            "beats_its_own_scrambled_null_at_the_full_lag": bool(
                r_at > null["best_r_mean"]),
        }
    agree = abs(halves["first_half"]["own_best_lag_s"]
                - halves["second_half"]["own_best_lag_s"])
    return {
        "full_best_r": full["best_r"],
        "full_best_lag_s": round(lag * HOP / SR, 3),
        "half_best_lags_agree_within_s": round(agree, 3),
        "halves_agree_on_the_offset": bool(agree <= 1.0),
        "both_halves_beat_their_own_null_at_the_full_lag": bool(
            halves["first_half"]["beats_its_own_scrambled_null_at_the_full_lag"]
            and halves["second_half"]
            ["beats_its_own_scrambled_null_at_the_full_lag"]),
        "halves": halves,
    }


def flicker_localisation(maps: np.ndarray, mouth_xy=None) -> dict:
    """Where a capture's motion energy actually is, in ORIGINAL pixels.

    The empty room is not still: it has real, low-amplitude motion that
    the repo's own work attributes to LED flicker. This localises it, so
    that the empty-room control can say whether the motion is anywhere
    near the mouth.

    The bounding box of the top-few-percent pixels is too coarse for a
    SPARSE region -- the flicker is scattered, so its bounding box is
    nearly the whole frame while the energy is in a handful of cells.
    So the mouth's own neighbourhood is measured directly: what share of
    the motion energy falls inside it.
    """
    mm = maps.mean(axis=0)
    total = float(mm.sum())
    flat = mm.ravel()
    order = np.argsort(flat)[::-1]
    out = {"mean_map_energy": round(float(maps.mean()), 6),
           "total_map_energy": round(total, 4)}
    for frac in (0.01, 0.05, 0.20):
        k = max(1, int(len(flat) * frac))
        idx = order[:k]
        rr, cc = idx // mm.shape[1], idx % mm.shape[1]
        out[f"top_{int(frac * 100)}_pct_of_map_pixels"] = {
            "pixel_fraction": frac,
            "energy_fraction": round(float(flat[idx].sum() / total), 4)
            if total > 0 else None,
            "row_range_px": [int(rr.min() * 2), int(rr.max() * 2 + 1)],
            "col_range_px": [int(cc.min() * 2), int(cc.max() * 2 + 1)],
        }
    if mouth_xy is not None:
        ys, xs = np.mgrid[0:mm.shape[0], 0:mm.shape[1]]
        px_y = ys * 2 + 1
        px_x = xs * 2 + 1
        inbox = (np.abs(px_x - mouth_xy[0]) <= 8) & \
                (np.abs(px_y - mouth_xy[1]) <= 5)
        share = float(mm[inbox].sum() / total) if total > 0 else 0.0
        out["mouth_neighbourhood"] = {
            "centre_px": [round(mouth_xy[0], 2), round(mouth_xy[1], 2)],
            "half_width_px": [8, 5],
            "n_map_cells": int(inbox.sum()),
            "energy_fraction": round(share, 6),
            "mean_energy_in_the_box": round(float(mm[inbox].mean()), 6),
            "mean_energy_overall": round(float(mm.mean()), 6),
            "box_is_above_the_frame_mean": bool(mm[inbox].mean() > mm.mean()),
        }
    # WHERE, vertically. The mean energy per 8-pixel band of the frame, so
    # that "the flicker is in the bottom of the frame" is a recorded
    # number rather than an impression.
    bands = []
    for r0 in range(0, mm.shape[0], 4):
        band = mm[r0:r0 + 4]
        bands.append({
            "px_rows": [r0 * 2, min(r0 + 4, mm.shape[0]) * 2 - 1],
            "mean_energy": round(float(band.mean()), 6),
        })
    hottest = max(bands, key=lambda b: b["mean_energy"])
    out["row_band_profile"] = bands
    out["hottest_row_band"] = hottest
    return out


def lag_profile(env_speech: np.ndarray, env_signal: np.ndarray,
                max_lag: int) -> dict:
    """The whole correlation-vs-lag curve, and how special its peak is.

    A max-over-lags statistic over 937 lags finds a good-looking offset
    even when there is nothing there. The peak of 937 roughly-normal
    draws sits about 3 sigma above the mean, so a peak at z = 3 is what
    NOISE gives. A real alignment is a peak that stands out from the rest
    of the curve, so the peak's z-score and the fraction of the curve
    that comes within half the peak are recorded.
    """
    m = min(len(env_speech), len(env_signal))
    a, b = detrend(env_speech[:m]), detrend(env_signal[:m])
    a = a - a.mean()
    b = b - b.mean()
    na = float(np.linalg.norm(a)) or 1.0
    nb = float(np.linalg.norm(b)) or 1.0
    a, b = a / na, b / nb
    prof = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[:m - lag]
        else:
            x, y = a[:m + lag], b[-lag:]
        if len(x) < 50:
            continue
        # the SAME statistic as `correlate`, so the peak's z-score is
        # comparable with the headline's r
        prof.append(float(np.dot(x, y)))
    prof = np.asarray(prof, dtype=np.float64)
    sd = float(prof.std())
    return {
        "peak": round(float(prof.max()), 4),
        "mean": round(float(prof.mean()), 4),
        "sd": round(sd, 4),
        "peak_z": round(float((prof.max() - prof.mean()) / sd), 3) if sd > 0 else None,
        "n_lags": int(len(prof)),
        "frac_of_lags_above_half_the_peak": round(
            float((prof > 0.5 * prof.max()).mean()), 4),
    }


def session_matrix(video_env: dict, speech: dict, signals, max_lag: int) -> dict:
    """Every video signal against every session's speech, in one matrix.

    A signal that tracks its OWN session's speech should be the largest
    entry in its row, and larger than the speech-to-speech agreement
    between the two sessions -- which is the ceiling, because if two takes
    of the same person reciting the same poem have similar envelopes then
    any signal that tracks speech content will align with both.
    """
    out = {}
    for sig_name in signals:
        if not any(sig_name in video_env[v] for v in video_env):
            continue
        cells = {}
        for v in video_env:
            for s in speech:
                if sig_name not in video_env[v]:
                    continue
                c = correlate(speech[s], video_env[v][sig_name], max_lag)
                cells[f"video_{v}_vs_speech_{s}"] = {
                    "best_r": c["best_r"],
                    "best_lag_s": round(c["best_lag"] * HOP / SR, 3),
                }
        out[sig_name] = cells
    # the ceiling: how similar are the two takes' SPEECH envelopes to each
    # other? A and B recite the same poem, so this is not a small number.
    ceiling = {}
    for a in speech:
        for b in speech:
            if a >= b:
                continue
            c = correlate(speech[a], speech[b], max_lag)
            ceiling[f"speech_{a}_vs_speech_{b}"] = {
                "best_r": c["best_r"],
                "best_lag_s": round(c["best_lag"] * HOP / SR, 3),
            }
    out["_speech_to_speech_ceiling"] = ceiling
    return out


def head_motion_statistics(lm: np.ndarray, detected: np.ndarray) -> dict:
    """How much the head moves, and whether the motion is smooth.

    A landmark-fit artefact is spiky; a person moving their head is
    smooth. The lag-1 autocorrelation of the detrended head position
    separates the two, and it is recorded because the eye-distance
    control needs to be explained rather than just reported.
    """
    cen = np.nanmean(lm, axis=1)
    out = {
        "centroid_sd_x_px": round(float(np.nanstd(cen[:, 0])), 4),
        "centroid_sd_y_px": round(float(np.nanstd(cen[:, 1])), 4),
    }
    d = detrend(cen[:, 1])
    d = d - d.mean()
    if len(d) > 2 and np.std(d) > 0:
        out["head_cy_lag1_autocorrelation"] = round(
            float(np.corrcoef(d[:-1], d[1:])[0, 1]), 4)
        out["head_cy_detrended_sd_px"] = round(float(np.std(d)), 4)
    return out


# ----------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--captures", default=str(CAPTURES))
    ap.add_argument("--out", default=str(ROOT / "experiments" /
                                         "mouth_aperture_alignment_results.json"))
    ap.add_argument("--upscale", type=int, default=UPSCALE)
    args = ap.parse_args(argv)
    caps = Path(args.captures)
    max_lag = int(OFFSET_SEARCH_S * SR / HOP)

    import mediapipe
    import cv2

    results = {
        "identity": {
            "script": "experiments/mouth_aperture_alignment.py",
            "question": ("does the MOUTH's motion, measured with a face "
                         "landmark model, track the AUDIO's speech in time?"),
            "run_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(
                    timespec="seconds"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "mediapipe": mediapipe.__version__,
            "landmark_model": dict(FACE_MESH),
            "upscale_factor": args.upscale,
            "upscale_interpolation": "cv2.INTER_CUBIC",
            "upscale_sensitivity_factors": list(UPSCALE_SENSITIVITY),
            "landmark_indices": dict(LANDMARKS),
            "video_fps": VIDEO_FPS,
            "fps_evidence": (
                "900 frames = 30.00 s at 30 fps against a 29.981 s "
                "recording (36.0 s at 25 fps); the empty room is 4500 frames "
                "= 150.00 s at 30 fps against 149.906 s (180 s at 25 fps). "
                "The frame rate is DERIVED from the two spans agreeing, "
                "not assumed. Read at the repository's 23040-byte stride "
                "the video would be 25 s for 30 s of audio, which is the "
                "same disagreement seen from the other side."),
            "record_layout": {
                "record_bytes": RECORD_BYTES,
                "luma_bytes": LUMA_BYTES,
                "chroma_bytes": CHROMA_BYTES,
                "frame_w": FRAME_W,
                "frame_h": FRAME_H,
                "note": ("each record is 15360 bytes of 160x96 luma "
                         "followed by 3840 bytes of INTERLEAVED U,V chroma "
                         "(80x24 samples). The file is NOT standard NV12: "
                         "the writer "
                         "('/mnt/c/proart_capture/nv12small.py') appends "
                         "the INTERLEAVED uv plane, so the stride is 19200 "
                         "and not 23040."),
                "writer": "/mnt/c/proart_capture/nv12small.py",
                "torn_stride_for_comparison": TORN_RECORD_BYTES,
            },
            "correction": {
                "date": "2026-10-20",
                "what": ("the stride. Every reader in this repository "
                         "assumed a 23040-byte NV12 frame; the files are "
                         "19200-byte records. Read at 23040, every frame "
                         "is torn between two real frames. This lane was "
                         "written against the CORRECTED stride from the "
                         "start and has never published a number computed "
                         "on the torn reading; the torn reading is run "
                         "only as a labelled sensitivity row."),
                "affects": ("the seeded corpus maps, and therefore "
                            "TEACHER-LOO-FOLDS.md §2 and §3, which were "
                            "computed on torn frames. Those two sections "
                            "are measured on an instrument that was "
                            "reading a tear, not the person."),
            },
            "correlation_params": {
                "hop_samples": HOP,
                "hop_s": HOP / SR,
                "offset_search_s": OFFSET_SEARCH_S,
                "max_lag_hops": max_lag,
                "n_lags_searched": 2 * max_lag + 1,
                "detrend_window_s": 2.0,
                "detrend_window_hops": int(2.0 / (HOP / SR)),
                "detrend": ("2 s moving average, from "
                            "audio_video_alignment.detrend; both signals "
                            "are detrended before every comparison"),
                "statistic": ("max over lags of the Pearson correlation on "
                              "the 32 ms hop grid"),
                "lag_sign": ("a POSITIVE lag means the mouth signal leads "
                             "the speech envelope, the same convention as "
                             "audio_video_alignment.correlate"),
                "imported_from": "experiments/audio_video_alignment.py",
            },
            "audio_gate": {
                "module": "src/vad.py",
                "model": "models/silero_vad.onnx",
                "frame_samples": 512,
                "sample_rate": 16000,
                "note": ("the repo's own gate, so 'speech' means the same "
                         "thing here as it does everywhere else"),
            },
        },
        "inputs": {},
        "detection": {},
        "sessions": {},
        "controls": {},
        "sensitivity": {},
    }

    # ---- the inputs, by identity ---------------------------------------
    input_paths = []
    for name in ("A", "B", "C"):
        input_paths += [caps / f"{name}.nv12", caps / f"{name}.wav"]
    input_paths += [caps / "empty1.nv12", caps / "empty1.wav",
                    ROOT / "models" / "silero_vad.onnx",
                    ROOT / "experiments" / "audio_video_alignment.py",
                    ROOT / "experiments" / "audio_video_alignment_results.json"]
    for p in input_paths:
        if p.is_file():
            results["inputs"][str(p)] = {
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
            }

    # The landmark model's own bundled weights, by hash. They ship inside
    # the mediapipe wheel, so the wheel's version plus these hashes are the
    # identity of the model.
    import mediapipe as _mp
    mp_dir = Path(_mp.__file__).resolve().parent
    for rel in ("modules/face_landmark/face_landmark_with_attention.tflite",
                "modules/face_landmark/face_landmark.tflite",
                "modules/palm_detection/palm_detection_full.tflite",
                "modules/palm_detection/palm_detection_lite.tflite",
                "modules/iris_landmark/iris_landmark.tflite"):
        f = mp_dir / rel
        if f.is_file():
            results["identity"]["landmark_model"].setdefault(
                "bundled_model_files", {})[str(f)] = {
                    "sha256": sha256(f), "bytes": f.stat().st_size}

    # ---- the speech envelopes -------------------------------------------
    speech = {}
    for name in ("A", "B", "C"):
        wav = caps / f"{name}.wav"
        if wav.is_file():
            e = speech_envelope_hop(wav)
            speech[name] = e
            results["sessions"].setdefault(name, {})["speech"] = {
                "mean": round(float(e.mean()), 4),
                "max": round(float(e.max()), 4),
                "frac_over_0.5": round(float((e >= 0.5).mean()), 4),
                "n_hops": int(len(e)),
                "duration_s": round(len(e) * HOP / SR, 3),
            }

    # ---- the mouth signals ----------------------------------------------
    mouth_env = {}
    for name in ("A", "B", "C"):
        nv = caps / f"{name}.nv12"
        if not (nv.is_file() and name in speech):
            continue
        frames = load_luma(nv)
        lm, det = face_mesh_landmarks(frames, args.upscale)
        sig = landmark_signals(lm, det)
        sig.update(head_motion_signals(lm, det))
        n_hops = len(speech[name])
        rowenv = {}
        for sname, s in sig.items():
            filled, info = fill_gaps(s)
            if not np.isfinite(filled).any():
                rowenv[sname] = {"defined": False, "gap_fill": info}
                continue
            env = to_hop_grid(filled, VIDEO_FPS, n_hops)
            r = row(sname, env, speech[name], max_lag)
            r["gap_fill"] = info
            r["defined"] = True
            r["r_at_lag_zero"] = round(at_offset_detrended(
                speech[name], env, 0), 4)
            rowenv[sname] = r
        results["sessions"][name]["signals"] = rowenv
        results["sessions"][name]["landmarks"] = landmark_identity(lm, det)
        results["sessions"][name]["head_motion"] = head_motion_statistics(
            lm, det)
        results["detection"][name] = {
            "n_frames": int(len(frames)),
            "n_faces_detected": int(det.sum()),
            "detection_fraction": round(float(det.mean()), 4),
            "frame_indices_used": "all",
        }
        results["sessions"][name]["stride_evidence"] = stride_evidence(nv)
        mouth_env[name] = {k: to_hop_grid(fill_gaps(v)[0], VIDEO_FPS, n_hops)
                           for k, v in sig.items() if np.isfinite(v).any()}

    # ---- the controls ----------------------------------------------------
    ctrl = results["controls"]

    # 1. scrambled audio, for EVERY signal. The null is only meaningful
    #    if it is computed for the signal it is compared against.
    for name in ("A", "B", "C"):
        if name not in mouth_env or name not in speech:
            continue
        for sig_name in ("mouth_aperture", "eye_distance", "mouth_width",
                         "jaw_open", "head_cx", "head_cy"):
            if sig_name not in mouth_env[name]:
                continue
            ctrl[f"scrambled_audio_{name}_{sig_name}"] = scrambled_null(
                speech[name], mouth_env[name][sig_name], max_lag)

    # 2. wrong session. The rows are grouped, because the three groups
    #    answer three different questions and only one of them is a clean
    #    control:
    #      content DIFFERS (A's audio vs C's video) -- the clean control.
    #        C is the same person, silent, so a correlation here would be
    #        about the person rather than the moment.
    #      same CONTENT (A's audio vs B's video) -- NOT a clean control, and
    #        the reason is measured and recorded: A and B are two takes of
    #        the same person reciting the SAME poem, so their speech
    #        envelopes agree with each other at a correlation far above
    #        this lane's null. A signal that tracks speech CONTENT will
    #        therefore align with both takes. Reported, and not used to
    #        judge the mouth.
    #      C's audio (C's audio vs A's video) -- numerically unstable: C's
    #        speech envelope is a flat line at 0.003, so the correlation is
    #        the ratio of two near-constant signals.
    for a_name, v_name in (("A", "C"), ("C", "A"), ("B", "C"), ("C", "B"),
                            ("A", "B"), ("B", "A")):
        if a_name not in speech or v_name not in mouth_env:
            continue
        if "mouth_aperture" not in mouth_env[v_name]:
            continue
        r = row("mouth_aperture", mouth_env[v_name]["mouth_aperture"],
                speech[a_name], max_lag)
        r["audio_session"] = a_name
        r["video_session"] = v_name
        r["audio_is_a_real_speech_envelope"] = bool(a_name in ("A", "B"))
        r["content_differs"] = bool({a_name, v_name} != {"A", "B"})
        r["is_a_clean_control"] = bool(r["audio_is_a_real_speech_envelope"]
                                       and r["content_differs"])
        if not r["is_a_clean_control"]:
            if not r["content_differs"]:
                r["note"] = ("A and B recite the SAME poem, so their "
                             "speech envelopes agree with each other. This "
                             "row measures content match, not session "
                             "identity, and it is NOT used to judge the "
                             "mouth.")
            else:
                r["note"] = ("C's speech envelope is flat at 0.003, so this "
                             "correlation is numerically unstable. "
                             "Reported, not used to judge anything.")
        key = (f"wrong_session_{a_name}_audio_{v_name}_video")
        ctrl[key] = r

    # 3. the eye-distance control, the same statistic on a landmark that
    #    must not track speech
    for name in ("A", "B", "C"):
        if name not in mouth_env or "eye_distance" not in mouth_env[name]:
            continue
        ctrl[f"eye_distance_{name}"] = row(
            "eye_distance", mouth_env[name]["eye_distance"], speech[name],
            max_lag)

    # 4. is the mouth's alignment MOUTH-specific, or is it the head
    #    moving? The mouth aperture with the head motion and the eye
    #    distance regressed out, and then the SAME statistic on it.
    for name in ("A", "B"):
        if name not in mouth_env:
            continue
        need = ("eye_distance", "head_cx", "head_cy", "face_height_px")
        if not all(k in mouth_env[name] for k in need):
            continue
        resid, info = partial_residual(
            mouth_env[name]["mouth_aperture"],
            [mouth_env[name][k] for k in need])
        r = row("mouth_aperture|head+eye removed", resid, speech[name], max_lag)
        r["variance_explained_by_the_controls"] = info["variance_explained"]
        r["controls_regressed_out"] = list(need)
        r["scrambled_audio_null"] = scrambled_null(speech[name], resid, max_lag)
        r["margin_over_scrambled_audio_null"] = round(
            r["best_r"] - r["scrambled_audio_null"]["best_r_mean"], 4)
        ctrl[f"mouth_aperture_partial_{name}"] = r

    # 5. does the alignment REPLICATE on the two halves of the recording?
    repl = {}
    for name in ("A", "B"):
        if name not in mouth_env:
            continue
        for sig_name in ("mouth_aperture", "eye_distance", "head_cy"):
            if sig_name not in mouth_env[name]:
                continue
            repl[f"{name}_{sig_name}"] = split_half_check(
                speech[name], mouth_env[name][sig_name], max_lag)
    ctrl["split_half_replication"] = repl

    # 6. the full matrix (every video signal against every session's
    #    speech), the speech-to-speech ceiling, and how special the peak
    #    of the correlation curve actually is.
    ctrl["session_matrix"] = session_matrix(
        mouth_env, speech,
        ("mouth_aperture", "eye_distance", "mouth_width", "head_cy"), max_lag)
    peaks = {}
    for name in ("A", "B"):
        if name not in mouth_env:
            continue
        for sig_name in ("mouth_aperture", "eye_distance", "head_cy"):
            if sig_name not in mouth_env[name]:
                continue
            peaks[f"{name}_{sig_name}"] = lag_profile(
                speech[name], mouth_env[name][sig_name], max_lag)
    # the same statistic on a SCRAMBLED pair, so the reader can see what
    # noise's peak looks like
    rng = np.random.default_rng(SCRAMBLE_SEED)
    for name in ("A",):
        if name not in mouth_env:
            continue
        for sig_name in ("mouth_aperture", "head_cy"):
            if sig_name not in mouth_env[name]:
                continue
            perm = rng.permutation(len(speech[name]))
            peaks[f"{name}_{sig_name}_SCRAMBLED"] = lag_profile(
                speech[name][perm], mouth_env[name][sig_name], max_lag)
    ctrl["lag_profile_peak"] = peaks

    # 4. the empty room
    empty_nv = caps / "empty1.nv12"
    empty_wav = caps / "empty1.wav"
    if empty_nv.is_file() and empty_wav.is_file():
        eframes = load_luma(empty_nv)
        elm, edet = face_mesh_landmarks(eframes, args.upscale)
        esig = landmark_signals(elm, edet)
        results["detection"]["empty1"] = {
            "n_frames": int(len(eframes)),
            "n_faces_detected": int(edet.sum()),
            "detection_fraction": round(float(edet.mean()), 4),
        }
        room_tone = speech_envelope_hop(empty_wav)
        erow = {
            "n_frames": int(len(eframes)),
            "n_faces_detected": int(edet.sum()),
            "room_tone_mean": round(float(room_tone.mean()), 4),
            "room_tone_max": round(float(room_tone.max()), 4),
            "room_tone_frac_over_0.5": round(float((room_tone >= 0.5).mean()), 4),
            "room_tone_n_hops": int(len(room_tone)),
        }
        if edet.any():
            filled, info = fill_gaps(esig["mouth_aperture"])
            env = to_hop_grid(filled, VIDEO_FPS, len(room_tone))
            erow["mouth_signal"] = row("mouth_aperture", env, room_tone, max_lag)
            erow["mouth_signal"]["gap_fill"] = info
            # the informative version: the empty room's mouth signal against
            # a real speech envelope, with the matched scrambled null
            n_slice = min(len(eframes), int(round(30.0 * VIDEO_FPS)))
            sl = to_hop_grid(filled[:n_slice], VIDEO_FPS, len(speech["A"]))
            erow["empty_room_vs_A_speech"] = row(
                "mouth_aperture", sl, speech["A"], max_lag)
            erow["empty_room_vs_A_speech"]["frame_indices"] = \
                f"0-{n_slice - 1}"
            erow["empty_room_vs_A_speech_scrambled"] = scrambled_null(
                speech["A"], sl, max_lag)
        else:
            erow["mouth_signal"] = {
                "defined": False,
                "note": ("the landmark model detected NO face in any of the "
                         f"{len(eframes)} empty-room frames, so there is no "
                         "mouth signal to correlate. The control is VACUOUS "
                         "by construction, and that is itself the result: "
                         "the model does not hallucinate a mouth in an "
                         "empty room, so the empty room's motion cannot be "
                         "a mouth."),
            }
        # The empty room is NOT still. It has real, low-amplitude motion
        # (the repo's own work attributes it to LED flicker), and that
        # motion is confined to a small region. Localise it, and measure
        # what share of it lands in the MOUTH's neighbourhood.
        emaps = change_maps(row_band_remove(eframes.astype(np.float64)),
                            FRAME_H, FRAME_W)
        # the mouth's position comes from session A, so it is available
        # even though the empty room has no face at all
        mouth_xy = None
        _lm = results["sessions"].get("A", {}).get("landmarks", {})
        _u = _lm.get("upper_inner_lip", {})
        _l = _lm.get("lower_inner_lip", {})
        if _u.get("mean_x_px") is not None and _l.get("mean_x_px") is not None:
            mouth_xy = ((_u["mean_x_px"] + _l["mean_x_px"]) / 2.0,
                        (_u["mean_y_px"] + _l["mean_y_px"]) / 2.0)
        erow["motion_localisation"] = flicker_localisation(emaps, mouth_xy)
        erow["mouth_region_is_dominant"] = bool(
            erow["motion_localisation"]["mouth_neighbourhood"]
            ["box_is_above_the_frame_mean"])
        erow["mouth_landmark_position_px"] = (
            [round(mouth_xy[0], 2), round(mouth_xy[1], 2)]
            if mouth_xy else None)
        # for comparison, the same measurement on the occupied take, on the
        # SAME pipeline, so the energy ratio is comparable
        if "A" in speech:
            amaps = change_maps(row_band_remove(
                load_luma(caps / "A.nv12").astype(np.float64)), FRAME_H, FRAME_W)
            erow["occupied_take_motion_localisation"] = \
                flicker_localisation(amaps, mouth_xy)
            erow["occupied_take_mean_map_energy"] = round(
                float(amaps.mean()), 6)
            erow["empty_room_mean_map_energy"] = round(
                float(emaps.mean()), 6)
            erow["empty_room_energy_relative_to_the_occupied_take"] = round(
                float(emaps.mean() / amaps.mean()), 6) if amaps.mean() > 0 else None
            erow["energy_ratio_note"] = (
                "the empty room's mean motion-map energy as a fraction of "
                "session A's, both from the SAME pipeline (row-band removal "
                "then 2x2 downsample). TEACHER-LOO-FOLDS.md §2 quotes "
                "0.0000066 for this ratio, measured on the TORN frames.")
        # the empty room's motion against a real speech envelope: the
        # scene's own motion must not track the speech either
        eenv = motion_envelopes(emaps, len(speech["A"]))
        erow["empty_room_motion_vs_A_speech"] = {
            e: row(e, eenv[e], speech["A"], max_lag)
            for e in ("total", "active", "area")}
        erow["empty_room_motion_vs_A_speech"]["active_scrambled_null"] = \
            scrambled_null(speech["A"], eenv["active"], max_lag)
        ctrl["empty_room"] = erow

    # 5. the degenerate baselines: the search's own inflation
    deg = {}
    if "A" in speech:
        deg["speech_envelope"] = degenerate_baselines(speech["A"], "speech")
    if "A" in mouth_env and "mouth_aperture" in mouth_env["A"]:
        deg["mouth_aperture_A"] = degenerate_baselines(
            mouth_env["A"]["mouth_aperture"], "mouth_aperture")
    ctrl["degenerate_baselines"] = deg

    # ---- the offset the WHOLE-FRAME measurement found --------------------
    av_path = ROOT / "experiments" / "audio_video_alignment_results.json"
    at_whole = {}
    if av_path.is_file():
        av = json.loads(av_path.read_text())
        for name in ("A", "B"):
            if name not in mouth_env or "mouth_aperture" not in mouth_env[name]:
                continue
            srow = av.get("sessions", {}).get(name, {})
            entry = {"source": str(av_path),
                     "note": ("TEACHER-LOO-FOLDS.md §3's offsets, computed "
                              "on the TORN frames; the mouth correlation is "
                              "evaluated at the same offset in seconds on "
                              "the correct frames")}
            for env_name in ("total", "active", "area"):
                if env_name not in srow:
                    continue
                lag_s = srow[env_name]["detrended_best_lag_s"]
                lag_hops = int(round(lag_s / (HOP / SR)))
                entry[env_name] = {
                    "whole_frame_lag_s": lag_s,
                    "mouth_aperture_r_at_that_offset": round(
                        at_offset_detrended(speech[name],
                                            mouth_env[name]["mouth_aperture"],
                                            lag_hops), 4),
                }
            at_whole[name] = entry
    results["controls"]["at_the_whole_frame_offset"] = at_whole

    # ---- the whole-frame instrument, RE-MEASURED on the correct frames ---
    # Not this lane's instrument, but the comparison the reader needs: the
    # question is whether the mouth is a better instrument than the whole
    # frame, and §3's whole-frame numbers were computed on torn frames.
    whole = {}
    for name in ("A", "B", "C"):
        nv = caps / f"{name}.nv12"
        if not (nv.is_file() and name in speech):
            continue
        fr = load_luma(nv)
        maps = change_maps(row_band_remove(fr.astype(np.float64)),
                           FRAME_H, FRAME_W)
        envs = motion_envelopes(maps, len(speech[name]))
        whole[name] = {e: row(e, envs[e], speech[name], max_lag)
                       for e in ("total", "active", "area")}
        whole[name]["mean_map_energy"] = round(float(maps.mean()), 4)
    results["controls"]["whole_frame_motion_on_correct_frames"] = whole

    # ---- the sensitivity rows --------------------------------------------
    # (a) the TORN reading, which is what the repository's frame reader
    #     produces, run through the same pipeline.
    for name in ("A",):
        nv = caps / f"{name}.nv12"
        if not (nv.is_file() and name in speech):
            continue
        tframes = load_luma_torn(nv)
        tlm, tdet = face_mesh_landmarks(tframes, args.upscale)
        tsig = landmark_signals(tlm, tdet)
        filled, info = fill_gaps(tsig["mouth_aperture"])
        tenv = to_hop_grid(filled, VIDEO_FPS, len(speech[name]))
        results["sensitivity"]["torn_frames"] = {
            "note": ("the SAME pipeline on the 23040-byte stride the "
                     "repository uses. Recorded so the difference between "
                     "the two readings is visible rather than asserted."),
            "n_frames": int(len(tframes)),
            "n_faces_detected": int(tdet.sum()),
            "detection_fraction": round(float(tdet.mean()), 4),
            "gap_fill": info,
            "mouth_aperture": row("mouth_aperture", tenv, speech[name], max_lag),
            "scrambled_audio_null": scrambled_null(
                speech[name], tenv, max_lag),
            "landmarks": landmark_identity(tlm, tdet),
        }
    # (b) the upscale factor: a number measured one way is one measurement
    for up in UPSCALE_SENSITIVITY:
        if up == args.upscale:
            continue
        nv = caps / "A.nv12"
        if not nv.is_file():
            continue
        fr = load_luma(nv)
        lm, det = face_mesh_landmarks(fr, up)
        sig = landmark_signals(lm, det)
        filled, info = fill_gaps(sig["mouth_aperture"])
        env = to_hop_grid(filled, VIDEO_FPS, len(speech["A"]))
        results["sensitivity"][f"upscale_{up}x"] = {
            "n_faces_detected": int(det.sum()),
            "detection_fraction": round(float(det.mean()), 4),
            "gap_fill": info,
            "mouth_aperture": row("mouth_aperture", env, speech["A"], max_lag),
            "scrambled_audio_null": scrambled_null(speech["A"], env, max_lag),
            "landmarks": landmark_identity(lm, det),
        }

    # ---- the headline ----------------------------------------------------
    # The mouth-aperture correlation at its best offset, MINUS the same
    # statistic with the scrambled audio. A margin over a control, never a
    # correlation alone.
    head = {
        "signal": "mouth_aperture",
        "statistic": ("max over +/-15 s of the Pearson correlation of the "
                      "detrended signals on the 32 ms hop grid"),
        "offset_policy": (
            "SEARCHED, not assumed to be zero. The audio and the video of "
            "A/B/C are separate recordings that were started independently, "
            "so their relative offset is unknown. The video is 30.00 s and "
            "the audio 29.98 s -- the same LENGTH, but equal length does "
            "not mean equal START, and the capture script's own docstring "
            "says the two were started independently. The +/-15 s search "
            "covers every relative start possible, because the video is "
            "30 s long and the audio 30 s long."),
    }
    real_r = []
    for name in ("A", "B"):
        s = results["sessions"].get(name, {}).get("signals", {})
        if "mouth_aperture" not in s or not s["mouth_aperture"].get("defined"):
            continue
        head[f"{name}_mouth_aperture_best_r"] = s["mouth_aperture"]["best_r"]
        head[f"{name}_mouth_aperture_best_lag_s"] = s["mouth_aperture"]["best_lag_s"]
        head[f"{name}_mouth_aperture_r_at_lag_zero"] = s["mouth_aperture"].get(
            "r_at_lag_zero")
        real_r.append(s["mouth_aperture"]["best_r"])
    null_r = None
    if "A" in speech and "A" in mouth_env and "mouth_aperture" in mouth_env["A"]:
        null_r = scrambled_null(speech["A"], mouth_env["A"]["mouth_aperture"],
                                 max_lag)
    if real_r and null_r is not None:
        head["mouth_aperture_best_r"] = max(real_r)
        head["scrambled_audio_null_r"] = null_r["best_r_mean"]
        head["margin_over_scrambled_audio_null"] = round(
            max(real_r) - null_r["best_r_mean"], 4)
        head["scrambled_audio_null_max_r"] = null_r["best_r_max"]
        head["margin_over_scrambled_audio_null_max"] = round(
            max(real_r) - null_r["best_r_max"], 4)

    # the controls, each as its own number
    eye_r = []
    for name in ("A", "B"):
        c = ctrl.get(f"eye_distance_{name}")
        if c:
            eye_r.append(c["best_r"])
    if eye_r:
        head["eye_distance_best_r"] = max(eye_r)
    stable_wrong = [v["best_r"] for k, v in ctrl.items()
                    if k.startswith("wrong_session_")
                    and v.get("is_a_clean_control")]
    if stable_wrong:
        head["wrong_session_best_r"] = max(stable_wrong)
    same_content = [v["best_r"] for k, v in ctrl.items()
                    if k.startswith("wrong_session_")
                    and v.get("audio_is_a_real_speech_envelope")
                    and not v.get("content_differs")]
    if same_content:
        head["wrong_session_same_content_best_r"] = max(same_content)
        head["wrong_session_same_content_note"] = (
            "A's audio against B's video and B's audio against A's video. "
            "A and B are two takes of the same person reciting the SAME "
            "poem, so their speech envelopes agree with each other at a "
            "correlation far above this lane's null -- which is the CEILING "
            "for any signal that tracks speech content. These rows are "
            "reported and NOT used to judge the mouth.")
    unstable_wrong = [v["best_r"] for k, v in ctrl.items()
                      if k.startswith("wrong_session_")
                      and not v.get("audio_is_a_real_speech_envelope")]
    if unstable_wrong:
        head["wrong_session_c_audio_best_r"] = max(unstable_wrong)
        head["wrong_session_c_audio_note"] = (
            "C's speech envelope is a flat line at 0.003, so a correlation "
            "against it is the ratio of two near-constant signals and is "
            "numerically unstable. Reported, not used to judge anything.")
    width_r = []
    for name in ("A", "B"):
        s = results["sessions"].get(name, {}).get("signals", {})
        if "mouth_width" in s and s["mouth_width"].get("defined"):
            width_r.append(s["mouth_width"]["best_r"])
    if width_r:
        head["mouth_width_best_r"] = max(width_r)
        head["mouth_width_note"] = (
            "the horizontal distance between the mouth corners. A mouth "
            "that OPENS does not get wider, so this is a control on the "
            "same lips: if the mouth width correlates MORE than the mouth "
            "aperture, the correlation cannot be about the mouth opening.")
    head_cy_r = []
    for name in ("A", "B"):
        s = results["sessions"].get(name, {}).get("signals", {})
        if "head_cy" in s and s["head_cy"].get("defined"):
            head_cy_r.append(s["head_cy"]["best_r"])
    if head_cy_r:
        head["head_vertical_motion_best_r"] = max(head_cy_r)
        head["head_motion_note"] = (
            "the vertical position of the face in the frame, which is the "
            "signal the eye-distance control is really measuring. It is "
            "recorded because it says what the correlation is ABOUT.")

    # ---- the verdict, which has to be able to fail -----------------------
    if real_r and null_r is not None:
        margin = max(real_r) - null_r["best_r_mean"]
        best = max(real_r)
        repl = ctrl.get("split_half_replication", {})
        mouth_repl = repl.get("A_mouth_aperture") or repl.get("B_mouth_aperture")
        checks = {
            "beats_the_scrambled_audio_null": bool(margin > 0),
            "beats_the_eye_distance_control": bool(
                eye_r and best > max(eye_r)),
            "beats_the_mouth_width_control": bool(
                width_r and best > max(width_r)),
            "beats_the_wrong_session_control": bool(
                stable_wrong and best > max(stable_wrong)),
            "replicates_on_the_halves_of_the_recording": bool(
                mouth_repl and mouth_repl.get(
                    "both_halves_beat_their_own_null_at_the_full_lag")),
        }
        head["checks"] = checks
        head["the_mouth_tracks_the_speech"] = bool(all(checks.values()))
        head["reading"] = (
            "The mouth-aperture correlation at its best offset, MINUS the "
            "same max-over-lags statistic on the permuted speech. The "
            "verdict requires ALL FIVE checks: beating the scrambled-audio "
            "null; beating the eye-distance control (a landmark that must "
            "not track speech); beating the mouth-width control (a mouth "
            "that OPENS does not get WIDER); beating the clean "
            "wrong-session control; and replicating on the two halves of "
            "the recording. A correlation that beats the null but not the "
            "other controls is a correlation about the whole frame moving, "
            "not about a mouth.")
    results["headline"] = head

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2) + "\n")

    # ---- the printout ----------------------------------------------------
    print("landmark model: mediapipe", results["identity"]["mediapipe"],
          "FaceMesh, upscale", args.upscale)
    for name, d in results["detection"].items():
        print(f"  {name}: face detected in {d['n_faces_detected']}/"
              f"{d['n_frames']} frames")
    print("\nsessions (detrended, best over +/-15 s):")
    for name in ("A", "B", "C"):
        s = results["sessions"].get(name, {}).get("signals", {})
        if not s:
            continue
        for sig_name in ("mouth_aperture", "mouth_width", "jaw_open",
                         "eye_distance"):
            if sig_name in s and s[sig_name].get("defined"):
                r = s[sig_name]
                print(f"  {name} {sig_name:15s} best r {r['best_r']:+.4f} "
                      f"at {r['best_lag_s']:+.3f} s")
    print("\ncontrols:")
    for k, v in ctrl.items():
        if k in ("degenerate_baselines", "at_the_whole_frame_offset",
                 "empty_room", "whole_frame_motion_on_correct_frames"):
            continue
        if isinstance(v, dict) and "best_r" in v:
            print(f"  {k}: best r {v['best_r']:+.4f} at {v['best_lag_s']:+.3f} s")
        elif isinstance(v, dict) and "best_r_mean" in v:
            print(f"  {k}: scrambled null mean {v['best_r_mean']:+.4f} "
                  f"(max {v['best_r_max']:+.4f})")
    if "degenerate_baselines" in ctrl:
        for k, v in ctrl["degenerate_baselines"].items():
            print(f"  degenerate {k}: random-lag r "
                  f"{v['self_vs_self_random_lag_r_mean']:+.4f}, "
                  f"max-over-lags {v['self_vs_self_max_over_lags_r_mean']:+.4f}")
    print("\nheadline:")
    print(json.dumps(results["headline"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
