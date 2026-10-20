"""
vision_motion_teacher.py -- does a SEQUENCE of frames beat a single frame?

THE QUESTION, and the conceptual hole it pokes.

`experiments/cloud_multimodal.py` measures whether a vision-language model can
tell "this person is SPEAKING" from "this person is not speaking" off a single
160x96 greyscale webcam frame. Llama 3.2 Vision 11B came out at sensitivity
0.4545, false-positive rate 0.0000 on a person-present-but-silent session, and
a headline margin of +0.4545.

But look at the question that was actually ASKED:

    "Is the person in this image SPEAKING (their mouth is moving or open, as if
     talking) or NOT SPEAKING (mouth closed, still)?"

"their mouth is MOVING" is a claim about TIME. A single frame cannot show
motion. So the current stimulus gives the model one still image and asks it a
question about movement, and then scores whatever it says. Whether the
0.4545 is the model reading motion, or the model reading a single frame's
Mouth-Open cue and calling it "speaking", is not distinguishable by that
experiment. The hypothesis this file tests is that the extra frames are doing
nothing.

THE FIX, and the control that makes it measurable.

Give the model what it is being asked about: several CONSECUTIVE frames, in
time order, in one image. Then run the control that separates "motion" from
"appearance":

    strip           K consecutive frames, in time order, tiled into one image.
    strip_shuffled  THE SAME K FRAMES, in a scrambled order.

Both stimuli contain the identical pixels. The only difference is the ORDER,
so the only thing that can separate them is temporal structure. If the model is
using motion, `strip` must score materially higher than `strip_shuffled`. If
they come out the same, the extra frames carried no motion information for this
model and the hypothesis is false. A second control, `strip_repeated` (K copies
of ONE frame), shows what the model does with no temporal information at all.

This is the same shape as the OpenVLHarness result Casey pointed at
(https://huggingface.co/papers/2610.26860): a text description of an image is a
weak substitute for retaining the image, and the gain comes from the
capability layer that renders tool output into model-readable evidence. Here
the capability layer is MOTION EVIDENCE, and the evidence is frames.

WHAT IS HELD FIXED.

The model, the frames, the question, the request shape and the frame COUNT are
all identical across the stimuli. Only the stimulus ARRANGEMENT changes. The
baseline `single` arm is the current published setup, re-measured under this
file's own anchor rule so that every stimulus refers to the SAME moments.

THE DEGENERATE BASELINES ARE MANDATORY.

A model that always answers SPEAKING scores FPR 1.000; a model that always
answers NOT SPEAKING scores sensitivity 0.000. Both score a margin of exactly
0.000, because the margin is sensitivity minus FPR. Every margin in this file
is read against those two rows, and both are written into the results file.

And the base-rate trap: the published sensitivity of 0.4545 is NOT "45% of the
way to 1.0". It is a number that a constant-answer model could have produced
in either direction, and it only means something next to the FPR on a person
who is present and NOT speaking.

WHAT THIS DOES NOT SHOW.

One session per condition, one speaker, one room, twelve moments per
condition, and frames from one session are correlated, so the effective sample
size is far below twelve. A null here is a null about ONE MODEL on ONE CAPTURE
at ONE RESOLUTION, and the whole point of the shuffled control is that a null
can be detected rather than argued about.

THE VIDEO IS 160x96 GREYSCALE, AND THE FRAME STRIDE IS NOT THE OBVIOUS ONE.

This is the correction that invalidates the first version of this file, and
it invalidates the published baseline too.

`experiments/cloud_multimodal.py` reads an NV12 frame as `W*H*3//2` = 23,040
bytes: luma then INTERLEAVED chroma, the standard 4:2:0 layout. The captures
on disk are not that. `/mnt/c/proart_capture/{A,B,C}.nv12` were written by
`/mnt/c/proart_capture/nv12small.py`, which writes

    Y plane   15,360 bytes   (160x96)
    U plane    1,920 bytes   (80x48, PLANAR -- all of U, then all of V)
    V plane    1,920 bytes
    --------------------------
    STRIDE    19,200 bytes   900 frames = 30.00 s at 30 fps

and 19,200 x 900 = 17,280,000 = exactly the file size. The WAV beside each
capture is 29.981 s, so 900 frames at 30.02 fps matches the audio to within
0.02 s. Read at 23,040 instead, the same file yields 750 frames for 30 s of
audio, and 80% of them are TORN: rows 0..71 from frame k, rows 72..95 from
frame k+1.

THE TELL, and it is a cheap one. The mean absolute difference between
CONSECUTIVE frames of a static 30 fps webcam scene is a fraction of a grey
level. Read at the right stride, A gives 0.545, B 0.433, C 0.367, and the
EMPTY ROOM gives 0.005. Read at the wrong stride, A gives 43.560, B 37.740,
C 42.935, empty1 39.295 -- all of them the same number, because they are all
measuring the tear and not the scene.

So the motion that the repo has been measuring at 160x96 was, to about 99%,
stride artefact. And the published headline -- Llama 3.2 Vision 11B, margin
+0.4545, "USABLE" -- was measured on torn frames. This file re-measures
everything on CLEAN frames, and it also runs the torn frames as a CONTROL,
because the torn strip is the sharpest available null for "this strip carries
motion evidence": it is a strip of the same size, in the same layout, holding
the same number of panels, made of frames that do not exist.

The stride and the mean |diff| are recorded in the results file, so every
number here carries the identity of the pixels it was computed from.

CROSS-CHECKED. `src/nv12.py` (landed by another agent on the same day)
DETECTS the frame layout from the file's content rather than assuming one,
and it independently reports `y_u_v_planar`, stride 19,200, 900 frames for
all three captures. Its luma output is byte-identical to this file's
`load_y_strided(path, 19200)`. This file keeps its own explicit-stride
reader anyway, for two reasons: the stride has to be a PARAMETER here,
because the torn control needs the wrong one on purpose, and a results file
that says "19,200" is worth more than a results file that says "whatever the
detector decided".
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# READ-ONLY REUSE. `cloud_multimodal.py` is owned by another agent; nothing in
# this file modifies it. What is borrowed, and WHAT IS NOT:
#   speech_frame_indices -- the DETECTED-SPEECH alignment. `speak_frame_indices`
#                        with a huge n returns EVERY frame inside a speech
#                        segment, which is how this file recovers the segment
#                        boundaries it needs to place a K-frame window. This is
#                        an AUDIO-side function and the stride defect does not
#                        touch it.
#   write_png         -- the greyscale PNG writer (this env has no PIL).
#   parse_speaking    -- the NEGATIVE-FIRST parser. "NOT SPEAKING" contains
#                        "SPEAKING", so a substring check would score every
#                        negative as a positive.
#   cf_raw            -- the Cloudflare call.
#   Q_YESNO           -- the question, verbatim, so the question is provably
#                        the same one the published baseline used.
#
# NOT BORROWED: `load_y`. It reads at 23,040 bytes per frame, which is the
# stride defect this file exists partly to correct. The corrected reader is
# `load_y_strided` below, and the reader is a local copy precisely so that
# the stride is stated in THIS file, next to the numbers it produced.
from experiments.cloud_multimodal import (  # noqa: E402
    Q_YESNO,
    cf_raw,
    parse_speaking,
    speech_frame_indices,
    write_png,
)

ACCOUNT = os.environ.get("CF_ACCOUNT_ID", "049ff5e84ecf636b53b162cbb580aae6")
W, H = 160, 96
YS = W * H                 # 15,360 luma bytes per frame
CH = YS // 4               # 3,840 chroma bytes per frame, PLANAR
STRIDE_CORRECT = YS + CH   # 19,200 -- the real stride of these captures
STRIDE_TORN = YS + CH * 2  # 23,040 -- the standard 4:2:0 stride, and the
                           # control: read at this stride these files tear.
FPS = 30.0

# THE MODELS.
#
# `@cf/meta/llama-3.2-11b-vision-instruct` is the published baseline
# (margin +0.4545) and is the primary.
#
# The brief asked for `@cf/meta/llama-4-scout-17b-vision-instruct` as the
# second, weaker model. THAT MODEL NAME DOES NOT EXIST ON THIS ACCOUNT. The
# catalogue contains `@cf/meta-llama-4-scout-17b-16e-instruct` (the name the
# repo's own `experiments/cloud_vision_probe.py` uses, and the model the
# published "+0.167, NOT usable" number was measured on), and that is what is
# run here. The substitution is recorded in the results file.
#
# The other two are vision models on the same account that were found by
# sweeping all 73 catalogue entries; see `models_probed` in the results file for
# what was tried and what failed.
MODELS = [
    "@cf/meta/llama-3.2-11b-vision-instruct",
    "@cf/meta-llama-4-scout-17b-16e-instruct",
    "@cf/zai-org/glm-5.3-flash",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
]

# THE STIMULI. Every one of them is built from the SAME frames, so the
# comparison is like-for-like.
#
# `k`     how many source frames go into the stimulus.
# `scale` the bilinear upscale applied to EACH TILE before tiling.
# `order` "identity" = time order; "shuffled" = scrambled time order;
#         "repeated" = the same frame k times.
#
# `single_up` at scale 4 is 640x384, and `strip_k4` at scale 2 is 1280x192 --
# the SAME total pixel count (245760). So `single_up` vs `strip_k4` compares
# "one big frame" against "four small frames in a row" with the total image
# area held fixed. `strip_k8` is 2560x192, twice the area of `strip_k4`; that
# doubling is a confound of the K comparison and is recorded as such. The
# HEADLINE comparison (strip vs strip_shuffled at the same K) has identical
# dimensions and identical pixels, so it is free of it.
STIMULUS_SPECS = [
    {"name": "single", "k": 1, "scale": 1, "order": "identity",
     "layout": "horizontal", "source": "clean"},
    {"name": "single_up", "k": 1, "scale": 4, "order": "identity",
     "layout": "horizontal", "source": "clean"},
    {"name": "strip_k4", "k": 4, "scale": 2, "order": "identity",
     "layout": "horizontal", "source": "clean"},
    {"name": "strip_k4_shuffled", "k": 4, "scale": 2, "order": "shuffled",
     "layout": "horizontal", "source": "clean"},
    {"name": "strip_k4_repeated", "k": 4, "scale": 2, "order": "repeated",
     "layout": "horizontal", "source": "clean"},
    {"name": "strip_k8", "k": 8, "scale": 2, "order": "identity",
     "layout": "horizontal", "source": "clean"},
    {"name": "strip_k8_shuffled", "k": 8, "scale": 2, "order": "shuffled",
     "layout": "horizontal", "source": "clean"},
    {"name": "strip_k8_repeated", "k": 8, "scale": 2, "order": "repeated",
     "layout": "horizontal", "source": "clean"},
    # THE GRID ARM, and why it is here.
    #
    # A 1280x192 strip is 6.67:1. Llama 3.2 Vision does not have a
    # multi-image interface -- it tiles ONE image -- so a wide strip arrives
    # as a wide band divided into arbitrary crops, and the pilot run showed
    # the model answering SPEAKING to a strip of four BLANK tiles. That is a
    # property of the SHAPE, not of motion.
    #
    # The grid arm puts the SAME four frames in a 2x2 arrangement at the same
    # tile scale, which makes it 640x384 -- EXACTLY the dimensions and the
    # pixel count of `single_up`. So `single_up` against `grid_k4` is "one big
    # frame" against "four frames in a grid" with the total image area held
    # identical, and `grid_k4` against `grid_k4_shuffled` is the same
    # motion control on a shape the model can actually read.
    {"name": "grid_k4", "k": 4, "scale": 2, "order": "identity",
     "layout": "grid2x2", "source": "clean"},
    {"name": "grid_k4_shuffled", "k": 4, "scale": 2, "order": "shuffled",
     "layout": "grid2x2", "source": "clean"},
    # THE TORN-STRIP CONTROL, which is the sharpest null available here.
    #
    # The same anchors, the same layout, the same dimensions, the same number
    # of panels -- but every frame is read at the WRONG stride, so every
    # frame is a tear of two real frames. If the model scores the torn strip
    # the same as the clean one, then the strip is not carrying per-frame
    # evidence; it is carrying FORMAT. And if the torn ORDERED strip beats
    # the torn SHUFFLED strip, then even a garbage strip shows a "motion
    # effect", which would mean the headline measure is not measuring motion.
    {"name": "strip_k4_torn", "k": 4, "scale": 2, "order": "identity",
     "layout": "horizontal", "source": "torn"},
    {"name": "strip_k4_torn_shuffled", "k": 4, "scale": 2, "order": "shuffled",
     "layout": "horizontal", "source": "torn"},
    {"name": "strip_k8_torn", "k": 8, "scale": 2, "order": "identity",
     "layout": "horizontal", "source": "torn"},
    {"name": "strip_k8_torn_shuffled", "k": 8, "scale": 2, "order": "shuffled",
     "layout": "horizontal", "source": "torn"},
]

CONDITIONS = ("speak", "silent", "turned", "blank")

# The longest window any stimulus uses. The torn control is clamped against
# this so that the K=4 and K=8 torn strips start at the same frame.
KMAX = max(s["k"] for s in STIMULUS_SPECS)

# THE DEGENERATE BASELINES. A model that answers one word to everything.
# Both score a margin of 0.000, and that is the point: the margin separates a
# model that discriminates from a model that does not, and a raw sensitivity
# does not.
DEGENERATE_BASELINES = {
    "always_SPEAKING": {
        "sensitivity_on_speak": 1.0,
        "fpr_on_silent": 1.0,
        "fpr_on_blank": 1.0,
        "margin_over_silent": 0.0,
        "margin_over_blank": 0.0,
        "note": "a model that answers SPEAKING to everything, including the "
                "blank; scores FPR 1.000",
    },
    "always_NOT_SPEAKING": {
        "sensitivity_on_speak": 0.0,
        "fpr_on_silent": 0.0,
        "fpr_on_blank": 0.0,
        "margin_over_silent": 0.0,
        "margin_over_blank": 0.0,
        "note": "a model that answers NOT SPEAKING to everything; scores "
                "sensitivity 0.000 and FPR 0.000",
    },
}

# The threshold a margin has to clear to be called a separation, and the
# threshold the HEADLINE (strip over shuffled) has to clear to be called
# motion. Both are recorded with the numbers that produced them.
USABLE_MARGIN = 0.20
MATERIAL_HEADLINE = 0.20


# ---------------------------------------------------------------------------
# IMAGE CONSTRUCTION. No network, no API, no account. These are the functions
# the unit tests exercise directly.
# ---------------------------------------------------------------------------

def load_y_strided(path: Path, stride: int) -> np.ndarray:
    """Read the LUMA plane of an NV12 file at an EXPLICIT stride.

    `stride` is the number of bytes in one frame of the file on disk. The
    luma plane is the first W*H bytes of every frame, so the frames are
    recovered by reshaping and slicing; the chroma is not needed, and is not
    used.

    THE STRIDE IS A PARAMETER, not a constant, on purpose. The captures this
    file measures are 19,200 bytes per frame; the standard 4:2:0 layout is
    23,040; and the newer corpus tool writes luma only, 15,360. Reading any
    of them with any of the others produces frames that do not exist, and
    the failure is SILENT: the image looks like a plausible webcam frame and
    the "motion" between frames is a tear. So the stride is stated, recorded
    in the results file, and checked against the file size.
    """
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    n = len(raw) // stride
    if n == 0:
        raise ValueError(f"{path}: no complete frame at stride {stride}")
    return raw[:n * stride].reshape(n, stride)[:, :YS].reshape(n, H, W)


def frame_stats(Y: np.ndarray) -> dict:
    """The identity of a frame set: how many, and how much they move.

    `mean_abs_consecutive_diff` is the diagnostic that caught the stride
    defect. A real 30 fps webcam scene of a seated person changes by a
    fraction of a grey level between consecutive frames; a TORN read changes
    by tens of grey levels, and by the SAME amount in every session, because
    it is measuring the tear and not the scene.
    """
    if len(Y) < 2:
        return {"n_frames": int(len(Y)), "mean_abs_consecutive_diff": None}
    d = np.abs(np.diff(Y.astype(np.int16), axis=0))
    return {"n_frames": int(len(Y)),
            "mean_abs_consecutive_diff": round(float(d.mean()), 4),
            "max_abs_consecutive_diff": int(d.max())}


def upscale(img: np.ndarray, scale: int, mode: str = "bilinear") -> np.ndarray:
    """Upscale a (H, W) uint8 image by an integer `scale`.

    BILINEAR by default, and the choice is recorded in the results file. Why
    bilinear and not nearest: nearest is exact but it turns a 160x96 face into
    a 640x384 mosaic of 4x4 blocks, and the block edges are an artefact the
    model can see. Bilinear invents no new structure but softens the aliasing
    that the 4x4 blocks would otherwise add. Bilinear also cannot invent
    structure that was not in the source pixels, because it is a weighted sum
    of neighbouring real pixels -- so the upscaled frame carries exactly the
    information of the original, which is what makes `single_up` a fair test
    of RESOLUTION rather than of added detail.
    """
    if scale == 1:
        return img
    if mode == "nearest":
        return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    if mode != "bilinear":
        raise ValueError(f"unknown upscale mode {mode!r}")
    h, w = img.shape
    # Output pixel centres mapped back to source coordinates.
    ys = np.clip((np.arange(h * scale) + 0.5) / scale - 0.5, 0, h - 1)
    xs = np.clip((np.arange(w * scale) + 0.5) / scale - 0.5, 0, w - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bot = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    out = top * (1 - wy) + bot * wy
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def tile_horizontal(tiles: list[np.ndarray]) -> np.ndarray:
    """Tile images left-to-right into one image. Time runs left to right."""
    if not tiles:
        raise ValueError("no tiles")
    h = tiles[0].shape[0]
    for t in tiles:
        if t.shape[0] != h:
            raise ValueError("tiles must share a height")
    return np.concatenate(tiles, axis=1)


def tile_grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    """Tile images into a grid, ROW-MAJOR: left to right, then down.

    Time still runs left to right, then wraps to the next row.
    """
    if not tiles:
        raise ValueError("no tiles")
    if len(tiles) % cols:
        raise ValueError(f"{len(tiles)} tiles do not fill a grid of {cols}")
    h, w = tiles[0].shape
    rows = len(tiles) // cols
    out = np.zeros((rows * h, cols * w), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return out


def assemble(tiles: list[np.ndarray], layout: str) -> np.ndarray:
    """Lay out the tiles according to the stimulus's layout."""
    if layout == "horizontal":
        return tile_horizontal(tiles)
    if layout == "grid2x2":
        if len(tiles) != 4:
            raise ValueError("grid2x2 needs exactly 4 tiles")
        return tile_grid(tiles, 2)
    raise ValueError(f"unknown layout {layout!r}")


def build_stimulus(frames: list[np.ndarray], k: int, scale: int,
                   order: str, perm: list[int] | None = None,
                   layout: str = "horizontal") -> np.ndarray:
    """Build one stimulus image from `k` source frames.

    `frames` are the k source frames IN SOURCE (time) ORDER. `order`:
      "identity" -- tiles in time order.
      "shuffled" -- tiles in `perm` order. `perm` must be a permutation of
                    range(k) and must NOT be the identity (see
                    `shuffle_order`).
      "repeated" -- every tile is frames[0]: k copies of ONE frame.

    The returned image is greyscale uint8.
    """
    if len(frames) != k:
        raise ValueError(f"expected {k} frames, got {len(frames)}")
    if order == "identity":
        tiles = list(frames)
    elif order == "shuffled":
        if perm is None:
            raise ValueError("shuffled order needs a permutation")
        if sorted(perm) != list(range(k)):
            raise ValueError("permutation is not a permutation of range(k)")
        if perm == list(range(k)):
            raise ValueError("the identity order is not a control")
        tiles = [frames[i] for i in perm]
    elif order == "repeated":
        tiles = [frames[0]] * k
    else:
        raise ValueError(f"unknown order {order!r}")
    return assemble([upscale(t, scale) for t in tiles], layout)


def stimulus_size(spec: dict) -> tuple[int, int]:
    """The (width, height) of the image a stimulus produces."""
    tw = 160 * spec["scale"]
    th = 96 * spec["scale"]
    layout = spec.get("layout", "horizontal")
    if layout == "horizontal":
        return (tw * spec["k"], th)
    if layout == "grid2x2":
        return (tw * 2, th * 2)
    raise ValueError(f"unknown layout {layout!r}")


def make_tiles(frames: list[np.ndarray], spec: dict,
               perm: list[int] | None = None) -> tuple[np.ndarray, list[int]]:
    """Build one stimulus image and the ORDER its tiles are in.

    Returns (image, order) where `order[j]` is the index into `frames` of the
    source frame shown in tile j. `order` is what goes into the results file
    as the tile order, so the identity of every tile is recorded.
    """
    k = spec["k"]
    order_kind = spec["order"]
    if order_kind == "identity":
        order = list(range(k))
    elif order_kind == "shuffled":
        if perm is None:
            raise ValueError("shuffled order needs a permutation")
        order = list(perm)
    elif order_kind == "repeated":
        order = [0] * k
    else:
        raise ValueError(f"unknown order {order_kind!r}")
    tiles = [upscale(frames[j], spec["scale"]) for j in order]
    return assemble(tiles, spec.get("layout", "horizontal")), order


def shuffle_order(k: int, rng: np.random.Generator) -> list[int]:
    """A permutation of range(k) that DESTROYS motion.

    The identity is rejected, and so is every cyclic rotation. A rotation
    such as (1,2,3,0) preserves k-1 of the k adjacencies, so it is NOT a
    motion-destroyed control -- it is nearly the ordered strip wearing a
    disguise, and using one would let the headline be won by adjacency rather
    than destroyed. Rejecting rotations makes the control strictly harder to
    beat, which is the right direction for a control.
    """
    rotations = {tuple((i + r) % k for i in range(k)) for r in range(k)}
    for _ in range(1000):
        p = rng.permutation(k)
        if tuple(p) not in rotations:
            return [int(x) for x in p]
    raise RuntimeError(f"no non-rotational permutation found for k={k}")


def tile_multiset_hash(tiles: list[np.ndarray]) -> str:
    """Order-INDEPENDENT hash of a set of tiles.

    Two stimuli built from the same frames in a different order have the same
    multiset hash and different orders. That is the machine-checkable form of
    "appearance preserved, motion destroyed", and it is what the results file
    records per call so the claim can be audited rather than believed.
    """
    parts = sorted(hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest()
                   for t in tiles)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def tile_order_hash(tiles: list[np.ndarray]) -> str:
    """Order-DEPENDENT hash of a list of tiles.

    The pixels, in order, hashed as bytes. NOT `str(tile)`: numpy truncates
    the repr of a large array with "...", so two DIFFERENT tiles can produce
    the SAME string, and a hash of that string would call them identical. The
    whole point of this hash is to prove that two stimuli differ, so it has
    to be a hash of the pixels.
    """
    h = hashlib.sha256()
    for t in tiles:
        h.update(hashlib.sha256(np.ascontiguousarray(t).tobytes()).digest())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# FRAME SELECTION. The label of a frame comes from the AUDIO, at that frame's
# own timestamp -- the lesson `cloud_multimodal.py` already learned the hard
# way.
# ---------------------------------------------------------------------------

def speech_runs(wav_path: Path, n_video: int | None = None) -> list[tuple[int, int]]:
    """Contiguous runs of video-frame indices that fall inside DETECTED SPEECH.

    `speech_frame_indices` with a huge n returns every speech frame, because
    the stride collapses to 1. Grouping those into runs recovers the segment
    boundaries, which is what a K-frame window has to fit inside.

    `n_video` CLIPS the runs to the frames that actually exist in the video.
    This is not a detail: the WAV of session A is longer than its NV12. The
    audio-derived frame indices run to 897 while the video has 750 frames, so
    without the clip a K=8 window would run off the end of the video.
    `cloud_multimodal.py` hit the same wall and dealt with it by dropping
    out-of-range indices (`idx_speak = [i for i in idx_speak if i < len(YA)]`),
    which silently changed how many moments the speaking condition had.

    Per `docs/research/AV-SYNC.md` A.wav and A.nv12 do not correspond anyway,
    so these labels are provisional in either case; the clip is about not
    running off the end of the video, not about fixing the alignment.
    """
    frames = speech_frame_indices(wav_path, 10 ** 6)
    if n_video is not None:
        frames = [f for f in frames if f < n_video]
    runs: list[tuple[int, int]] = []
    for f in frames:
        if runs and f == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], f)
        else:
            runs.append((f, f))
    return runs


def pick_speech_anchors(runs: list[tuple[int, int]], n: int,
                         kmax: int) -> list[int]:
    """`n` anchors, each the start of a kmax-frame window INSIDE one speech run.

    Only runs at least kmax frames long are eligible, so the whole window of
    consecutive frames is speech -- not speech plus silence, which would be a
    mislabelled stimulus. The eligible runs are then thinned evenly over the
    session so the moments are spread in time rather than bunched at the
    start, and each anchor sits in the middle of its run's eligible range so
    it is not on a speech boundary.
    """
    elig = [(s, e) for (s, e) in runs if e - s + 1 >= kmax]
    if not elig or n <= 0:
        return []
    if n == 1:
        picks = [0]
    else:
        picks = sorted({int(round(i * (len(elig) - 1) / (n - 1)))
                        for i in range(n)})
    anchors = []
    for i in picks:
        s, e = elig[i]
        anchors.append(s + (e - s + 1 - kmax) // 2)
    return anchors


def pick_uniform_anchors(n_total: int, n: int, kmax: int,
                          lo: int = 30) -> list[int]:
    """`n` anchors spread evenly over a session that is silent throughout.

    `lo` skips the first second of the session (encoder warm-up). The last
    anchor is placed so its whole kmax-frame window is inside the session.
    The anchors are far enough apart that no two windows overlap.
    """
    hi = n_total - kmax
    if hi <= lo or n <= 0:
        return []
    stride = (hi - lo) / n
    return [lo + int(i * stride) for i in range(n)]


# ---------------------------------------------------------------------------
# THE CALL.
# ---------------------------------------------------------------------------

_agree_lock = threading.Lock()
_agreed: set[str] = set()
_no_gate: set[str] = set()

# The three classes of answer the licence probe can give.
#
# SHAPE: the endpoint wants a different BODY. Not a refusal -- the model is
# simply not the kind that takes this shape. The Llama 4 Scout endpoint
# answers the `{"prompt": "agree"}` shape with Cloudflare error code 7000,
# "No route for that URI", while accepting the `messages` shape and answering
# normally. That is a request-shape rejection.
#
# TRANSIENT: the account is rate limiting, or the route is briefly gone. NOT
# a refusal, and NOT a statement about the model. This MUST NOT be cached as
# a verdict: an earlier version of this file treated a rate-limit error as a
# licence refusal, and that single misclassification killed an entire model
# arm -- 96 calls, every one of them "agreement_failed", which reads exactly
# like a model that cannot see.
#
# LICENCE: an explicit refusal to agree. The only thing that may stop the run.
LICENCE_MARKERS = ("licen", "cannot agree", "must agree", "agreement required",
                   "terms of use", "not permitted")
SHAPE_MARKERS = ("oneof", "required properties", "type mismatch", "not met",
                  "no route for that uri")
TRANSIENT_MARKERS = ("rate limiting", "rate limit", "too many requests",
                     "timeout", "timed out", "internal server error",
                     "service unavailable", "bad gateway", "no route for that uri")


def ensure_agreement(model: str) -> tuple[bool, str]:
    """Meta's licence gate, with the three answer classes kept apart.

    The standing rule of this project, learned the hard way in
    `cloud_multimodal.py`: treating a shape error as a failed agreement
    silently kills a whole arm, and every one of its calls then reads
    "agreement_failed" -- which looks like a model failure and is not one.
    The opposite mistake is just as bad: treating a TRANSIENT error as a
    refusal. So:

      - a SHAPE error means the endpoint has no gate -> proceed;
      - a TRANSIENT error is retried, and if it never clears, the probe
        gives up and PROCEEDS, because the real call has its own retry and
        will report the real error if there is one;
      - only an explicit LICENCE refusal stops the arm;
      - a call that succeeds at all means proceed, even if the model answers
        with something empty. An empty answer to "agree" is not a refusal.

    `cloud_multimodal.cf_agree` gets the shape case right but not the
    transient case, and its "did it say anything?" test rejects a model
    whose `{"prompt": "agree"}` simply returns an empty completion -- which
    is what several non-Meta models do, and it is not a refusal.
    """
    with _agree_lock:
        if model in _agreed:
            return True, "agreed"
        if model in _no_gate:
            return True, "no gate on this endpoint"
    last = ""
    for attempt in range(5):
        r = cf_raw(model, {"prompt": "agree"}, timeout=60.0)
        if "error" not in r:
            with _agree_lock:
                _agreed.add(model)
            return True, "agreed"
        detail = (r.get("detail") or "").lower()
        last = detail
        if any(m in detail for m in LICENCE_MARKERS):
            return False, "licence refusal: " + detail[:200]
        if any(m in detail for m in SHAPE_MARKERS):
            with _agree_lock:
                _no_gate.add(model)
            return True, "no gate on this endpoint (request-shape rejection)"
        if any(m in detail for m in TRANSIENT_MARKERS) or \
                str(r.get("error")).startswith("http5"):
            time.sleep(2.0 * (attempt + 1))
            continue
        # An error we do not recognise is NOT a licence refusal, and refusing
        # to proceed on it would kill an arm on the strength of a guess.
        with _agree_lock:
            _no_gate.add(model)
        return True, "no gate on this endpoint (unrecognised error: " + \
            detail[:120] + ")"
    # The probe never got a clean answer. Proceed: the real call retries and
    # records its own error, so a genuinely unusable model still shows up as
    # an error on every call rather than as a false verdict here.
    return True, "agreement probe inconclusive (" + last[:120] + ")"


# Per-model pacing, with congestion control.
#
# The account rate-limits inference PER MODEL PER MINUTE, and the limits
# differ wildly between models: Llama 3.2 Vision and Mistral Small served
# every request in a probe while glm-5.3-flash answered "rate limiting:
# inference request per min rate reached" to every one of four calls made
# while the other two models were fine.
#
# A fixed interval cannot be right for all of them, and a fixed interval that
# is too fast for one model turns that model's calls into a retry storm: the
# run gets slow and the arm comes back full of errors, which reads like a
# model failure and is really a scheduling failure.
#
# So this is AIMD -- additive increase, multiplicative decrease. Every model
# starts at MIN_INTERVAL_S. A rate-limit error DOUBLES that model's
# interval; a success walks it back down by 5%. The interval is recorded, so
# the results file says what the scheduler was doing.
MIN_INTERVAL_S = 0.5
MAX_INTERVAL_S = 20.0
_rate_lock = threading.Lock()
_pace_state: dict[str, list] = {}      # model -> [interval, last_call_time]


def _pace(model: str) -> None:
    """Wait out this model's interval.

    The wait happens OUTSIDE the lock. Holding the lock across a sleep would
    serialise every model's pacing behind the slowest one, which is the
    opposite of what a per-model limiter is for.
    """
    with _rate_lock:
        st = _pace_state.get(model)
        if st is None:
            st = [MIN_INTERVAL_S, 0.0]
            _pace_state[model] = st
        interval, last = st
        wait = interval - (time.time() - last)
    if wait > 0:
        time.sleep(wait)
    with _rate_lock:
        _pace_state[model][1] = time.time()


def _note_rate_limited(model: str) -> None:
    with _rate_lock:
        st = _pace_state.setdefault(model, [MIN_INTERVAL_S, 0.0])
        st[0] = min(MAX_INTERVAL_S, st[0] * 2.0)


def _note_success(model: str) -> None:
    with _rate_lock:
        st = _pace_state.setdefault(model, [MIN_INTERVAL_S, 0.0])
        st[0] = max(MIN_INTERVAL_S, st[0] * 0.95)


def pacing_report() -> dict:
    with _rate_lock:
        return {m: {"final_interval_s": round(st[0], 3),
                    "calls_per_minute_at_final_interval": round(
                        60.0 / max(st[0], 1e-6), 1)}
                for m, st in _pace_state.items()}


def vision_call(model: str, png_b64: str, question: str,
                retries: int = 6) -> dict:
    """The request shape `cloud_multimodal.py` already proves works."""
    ok, why = ensure_agreement(model)
    if not ok:
        return {"error": "agreement_failed", "detail": why, "elapsed_s": 0.0}
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{png_b64}"}}]}],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    last = {}
    for attempt in range(retries):
        _pace(model)
        last = cf_raw(model, body, timeout=180.0)
        if "error" not in last:
            _note_success(model)
            last["agreement"] = why
            last["attempts"] = attempt + 1
            return last
        det = (last.get("detail") or "")
        if "rate limiting" in det or "rate limit" in det:
            # The account is rate limiting THIS model. Back that model off
            # and wait it out rather than burning retries.
            _note_rate_limited(model)
            time.sleep(min(30.0, 2.0 * (2 ** attempt)))
            continue
        # A hard shape error will not fix itself on a retry.
        if "oneOf" in det or "required properties" in det:
            break
        # The Llama 4 Scout route answers "No route for that URI" for a
        # while and then comes back. Transient, and worth a retry with a
        # backoff that grows.
        time.sleep(min(30.0, 2.0 * (2 ** attempt)))
    last["agreement"] = why
    last["attempts"] = retries
    return last


def preflight_model(model: str, png_b64: str, attempts: int = 4,
                    gap: float = 12.0) -> tuple[bool, str]:
    """Is this model actually reachable right now?

    The Llama 4 Scout endpoint is in the account catalogue but its RUN route
    answers "No route for that URI" (Cloudflare code 7000) some of the time
    and answers normally some of the other time. A model in the catalogue is
    not a model you can call.

    So every model is probed once, before the run, with a real image and the
    real question. A model that cannot be reached is RECORDED AS UNAVAILABLE
    and dropped from the run -- which is a different statement from "the
    model failed the task", and the two must not be confused.
    """
    last = ""
    for i in range(attempts):
        r = vision_call(model, png_b64, Q_YESNO, retries=1)
        if "error" not in r:
            return True, "reachable: " + (r.get("text") or "")[:60]
        last = f"{r.get('error')}: {(r.get('detail') or '')[:160]}"
        if "no route for that uri" in (r.get("detail") or "").lower():
            # the route may come back; wait and try again
            time.sleep(gap)
            continue
        return False, last
    return False, last


# ---------------------------------------------------------------------------
# THE MEASUREMENT.
# ---------------------------------------------------------------------------

def _condition_stats(records: list[dict]) -> dict:
    """Rates for one condition. The DENOMINATOR IS EXPLICIT and is the number
    of stimuli SCORED, with unparsed and errored stimuli broken out so a
    short denominator cannot hide inside a percentage."""
    n = len(records)
    said = sum(1 for r in records if r["label"] is True)
    parsed = sum(1 for r in records if r["label"] is not None)
    unparsed = sum(1 for r in records if r["label"] is None and not r["error"])
    errors = sum(1 for r in records if r["error"])
    return {
        "n_stimuli": n,
        "n_parsed": parsed,
        "n_unparsed": unparsed,
        "n_errors": errors,
        "said_speaking": said,
        "rate_says_speaking": (said / n) if n else None,
        "denominator": n,
    }


def score_model(model: str, records: list[dict]) -> dict:
    """Per-stimulus rates and margins for one model, from the call records."""
    by_stim: dict[str, dict[str, list[dict]]] = {}
    for r in records:
        by_stim.setdefault(r["stimulus"], {}).setdefault(r["condition"], []).append(r)

    out: dict[str, dict] = {}
    for spec in STIMULUS_SPECS:
        name = spec["name"]
        cond_stats = {}
        for c in CONDITIONS:
            rs = by_stim.get(name, {}).get(c, [])
            cond_stats[c] = _condition_stats(rs)
        sp = cond_stats["speak"]
        si = cond_stats["silent"]
        bl = cond_stats["blank"]
        sens = sp["rate_says_speaking"]
        fpr_silent = si["rate_says_speaking"]
        fpr_blank = bl["rate_says_speaking"]
        margin = (None if (sens is None or fpr_silent is None)
                  else sens - fpr_silent)
        margin_blank = (None if (sens is None or fpr_blank is None)
                        else sens - fpr_blank)
        out[name] = {
            "stimulus": dict(spec),
            "conditions": cond_stats,
            "sensitivity_on_speak": sens,
            "fpr_on_silent_person_present": fpr_silent,
            "fpr_on_blank": fpr_blank,
            "turned_rate_says_speaking": cond_stats["turned"]["rate_says_speaking"],
            "margin_over_silent": margin,
            "margin_over_blank": margin_blank,
            "usable": bool(margin is not None and margin >= USABLE_MARGIN),
        }
    return out


# THE HEADLINE ARMS: an ordered stimulus and the shuffled control that
# shares its frames.
HEADLINE_ARMS = [
    ("k4", "strip_k4", "strip_k4_shuffled"),
    ("k8", "strip_k8", "strip_k8_shuffled"),
    ("grid_k4", "grid_k4", "grid_k4_shuffled"),
    ("torn_k4", "strip_k4_torn", "strip_k4_torn_shuffled"),
    ("torn_k8", "strip_k8_torn", "strip_k8_torn_shuffled"),
]


def headline_for(model_scores: dict) -> dict:
    """THE HEADLINE: the margin of the ORDERED strip over the SHUFFLED strip.

    Computed from the recorded per-condition numbers, never hardcoded.

    The FPR is reported alongside, and `fpr_matched` records whether the two
    strips were even compared at a comparable false-positive rate. If the
    shuffled strip had a wildly different FPR, the headline would be a
    difference in prompt-scale response rather than a difference in motion
    use, and it would have to be read that way.
    """
    heads = {}
    for label, ordered_name, shuffled_name in HEADLINE_ARMS:
        s = model_scores.get(ordered_name)
        sh = model_scores.get(shuffled_name)
        repeated_name = ordered_name + "_repeated"
        rp = (model_scores.get(repeated_name)
              if repeated_name in model_scores else None)
        if not (s and sh):
            continue
        f_s = s["fpr_on_silent_person_present"]
        f_sh = sh["fpr_on_silent_person_present"]
        sens_s = s["sensitivity_on_speak"]
        sens_sh = sh["sensitivity_on_speak"]
        m_s = None if (sens_s is None or f_s is None) else sens_s - f_s
        m_sh = None if (sens_sh is None or f_sh is None) else sens_sh - f_sh
        sens_delta = (None if (sens_s is None or sens_sh is None)
                      else sens_s - sens_sh)
        margin_delta = (None if (m_s is None or m_sh is None) else m_s - m_sh)
        # "matched FPR": the two strips have to be running at a comparable
        # false-positive rate for the sensitivity difference to be a motion
        # difference. Tolerance is ONE STIMULUS at this denominator.
        #
        # The comparison is done in COUNTS, not in rates, because a rate
        # comparison at this denominator is knife-edged: 7/12 - 6/12 is
        # 1/12 in exact arithmetic and 0.08333333333333337 in floating
        # point, so a rate comparison calls that "not matched" and a count
        # comparison calls it "matched" -- which is the right answer, since
        # one stimulus out of twelve IS the resolution of this measurement.
        #
        # The counts are recovered from the SAME rates the margin is computed
        # from, so the matching test and the margin can never disagree about
        # the same number.
        n_den = s["conditions"]["silent"]["denominator"]
        c_s = (None if f_s is None else int(round(f_s * n_den)))
        c_sh = (None if f_sh is None else int(round(f_sh * n_den)))
        matched = (None if (c_s is None or c_sh is None)
                   else abs(c_s - c_sh) <= 1)
        tol = 1.0 / max(1, n_den)
        head = {
            "arm": label,
            "ordered_stimulus": ordered_name,
            "shuffled_control": shuffled_name,
            "k": s["stimulus"]["k"],
            "layout": s["stimulus"].get("layout", "horizontal"),
            "strip": {"sensitivity_on_speak": sens_s,
                      "fpr_on_silent": f_s,
                      "fpr_on_blank": s["fpr_on_blank"],
                      "margin_over_silent": m_s},
            "strip_shuffled": {"sensitivity_on_speak": sens_sh,
                                "fpr_on_silent": f_sh,
                                "fpr_on_blank": sh["fpr_on_blank"],
                                "margin_over_silent": m_sh},
            "sensitivity_delta_strip_minus_shuffled": sens_delta,
            "margin_delta_strip_minus_shuffled": margin_delta,
            "fpr_tolerance": tol,
            "fpr_tolerance_in_stimuli": 1,
            "fpr_matched": matched,
            "materiality_threshold": MATERIAL_HEADLINE,
            # THE DEGENERATE READ. A stimulus on which the model answers
            # SPEAKING to everything, including the blank, is not a
            # classifier at all -- it is a constant. The margin of a
            # constant is 0.000, and the margin of `always_SPEAKING` in the
            # degenerate baselines is also 0.000, so this is the same
            # comparison the baselines are there to make.
            "strip_is_degenerate_on_blank": (
                s["fpr_on_blank"] == 1.0
                and s["fpr_on_silent_person_present"] == 1.0),
        }
        if rp is not None:
            head["strip_repeated"] = {
                "sensitivity_on_speak": rp["sensitivity_on_speak"],
                "fpr_on_silent": rp["fpr_on_silent_person_present"],
                "fpr_on_blank": rp["fpr_on_blank"],
                "margin_over_silent": rp["margin_over_silent"],
            }
            head["margin_delta_strip_minus_repeated"] = (
                None if (m_s is None or rp["margin_over_silent"] is None)
                else m_s - rp["margin_over_silent"])
        # THE VERDICT, and it is able to fail.
        if margin_delta is None:
            head["verdict"] = "UNDETERMINED (no numbers)"
        elif matched is False:
            head["verdict"] = (
                "INCONCLUSIVE: the ordered and shuffled strips did not run at "
                "a matched false-positive rate, so the difference is not "
                "cleanly a motion difference")
        elif margin_delta >= MATERIAL_HEADLINE:
            head["verdict"] = (
                f"MOTION HELPS: the ordered strip beats the shuffled strip by "
                f"{margin_delta:+.4f} in margin, above the {MATERIAL_HEADLINE} "
                f"threshold")
        elif margin_delta > 0:
            head["verdict"] = (
                f"WEAK, BELOW THRESHOLD: the ordered strip beats the shuffled "
                f"strip by only {margin_delta:+.4f} in margin, under the "
                f"{MATERIAL_HEADLINE} threshold")
        else:
            head["verdict"] = (
                f"NO MOTION EFFECT: the ordered strip does NOT beat the "
                f"shuffled strip (margin delta {margin_delta:+.4f}). The extra "
                f"frames carried no motion information for this model.")
        heads[label] = head
    return heads


# ---------------------------------------------------------------------------
# MAIN.
# ---------------------------------------------------------------------------

def build_moments(a) -> dict:
    """The frame moments each condition is built from.

    One anchor per moment. Every stimulus for that condition uses the SAME
    anchor, so `single`, `single_up`, `strip_k4` and `strip_k8` all refer to
    the same instant and the comparison is like-for-like.

    TWO READS OF EVERY SESSION: the CLEAN one at 19,200 bytes per frame,
    which is what these files actually are, and the TORN one at 23,040, which
    is what every reader in this repo used to do. The torn read is kept as a
    CONTROL and never as a measurement.
    """
    YA = load_y_strided(Path(a.speaking), STRIDE_CORRECT)
    YC = load_y_strided(Path(a.silent), STRIDE_CORRECT)
    YB = load_y_strided(Path(a.turned), STRIDE_CORRECT) \
        if Path(a.turned).is_file() else None
    TORN = load_y_strided(Path(a.speaking), STRIDE_TORN)
    TORN_C = load_y_strided(Path(a.silent), STRIDE_TORN)
    TORN_B = (load_y_strided(Path(a.turned), STRIDE_TORN)
              if Path(a.turned).is_file() else None)
    kmax = max(s["k"] for s in STIMULUS_SPECS)

    runs_speak = speech_runs(Path(a.speaking_wav), len(YA))
    anchors_speak = pick_speech_anchors(runs_speak, a.frames, kmax)
    anchors_silent = pick_uniform_anchors(len(YC), a.frames, kmax)

    moments = {
        "speak": {
            "source_video": a.speaking,
            "label_source": a.speaking_wav,
            "Y": YA,
            "Y_torn": TORN,
            "anchors": anchors_speak,
            "rule": ("anchors are the middles of DETECTED-SPEECH runs long "
                     "enough to contain the whole K=8 window, thinned evenly "
                     "over the session; the label of every frame in the "
                     "window comes from the gate on the WAV at that frame's "
                     "own timestamp"),
        },
        "silent": {
            "source_video": a.silent,
            "label_source": a.silent_wav,
            "Y": YC,
            "Y_torn": TORN_C,
            "anchors": anchors_silent,
            "rule": ("anchors spread evenly over the session (first second "
                     "skipped for encoder warm-up); the gate on the WAV "
                     "detects NO speech anywhere in this session, so every "
                     "frame of every window is a no-speech frame"),
        },
    }
    if YB is not None:
        runs_turned = speech_runs(Path(a.turned_wav), len(YB))
        moments["turned"] = {
            "source_video": a.turned,
            "label_source": a.turned_wav,
            "Y": YB,
            "Y_torn": TORN_B,
            "anchors": pick_speech_anchors(runs_turned, a.frames, kmax),
            "rule": ("the TURNED condition: head turned away, speaking; "
                     "anchors chosen by the same rule as `speak`, from "
                     "DETECTED-SPEECH runs on the turned session's WAV. "
                     "Reported separately and NEVER pooled with `speak`."),
        }
    return moments


def main():
    global MIN_INTERVAL_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaking", default="/mnt/c/proart_capture/A.nv12")
    ap.add_argument("--silent", default="/mnt/c/proart_capture/C.nv12")
    ap.add_argument("--turned", default="/mnt/c/proart_capture/B.nv12")
    ap.add_argument("--speaking-wav", default="/mnt/c/proart_capture/A.wav")
    ap.add_argument("--silent-wav", default="/mnt/c/proart_capture/C.wav")
    ap.add_argument("--turned-wav", default="/mnt/c/proart_capture/B.wav")
    ap.add_argument("--frames", type=int, default=12,
                    help="moments (stimuli) per condition; the DENOMINATOR")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--min-interval", type=float, default=MIN_INTERVAL_S,
                    help="starting per-model call interval, in seconds; the "
                         "scheduler backs a model off when it is rate limited")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--records", default=None,
                    help="JSONL checkpoint of completed calls; a re-run "
                         "resumes from it")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing checkpoint and start over")
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    if not os.environ.get("CF_TOKEN"):
        sys.exit("no CF_TOKEN (source ~/.config/cloudflare/env)")

    MIN_INTERVAL_S = max(0.05, a.min_interval)

    models = [m.strip() for m in a.models.split(",") if m.strip()]

    # THE PRE-FLIGHT. A model that is in the catalogue but whose run route is
    # down is not a model, and calling it 600 times to collect 600 identical
    # errors is waste. Each model is probed once, here, with a real image and
    # the real question; the ones that answer are the ones that run.
    probe_png = base64.b64encode(write_png(np.full((H, W), 128, np.uint8))).decode()
    preflight = {}
    runnable = []
    for m in models:
        ok, why = preflight_model(m, probe_png)
        preflight[m] = {"reachable": ok, "detail": why}
        print(f"preflight {m}: {'OK' if ok else 'UNAVAILABLE'} -- {why[:110]}",
              flush=True)
        if ok:
            runnable.append(m)
    if not runnable:
        sys.exit("no model is reachable; refusing to write an empty result")
    models = runnable

    moments = build_moments(a)

    # THE PERMUTATIONS. Fixed by the seed, and recorded, so the shuffled
    # strips are reproducible and so a reader can check that the ordered and
    # shuffled strips really do hold the same frames.
    #
    # Only the K values that a `shuffled` stimulus actually uses get a
    # permutation. For K=1 there is no non-identity permutation at all, and
    # `shuffle_order` would (correctly) refuse to produce one.
    rng = np.random.default_rng(a.seed)
    shuffled_ks = sorted({s["k"] for s in STIMULUS_SPECS
                           if s["order"] == "shuffled"})
    perms = {k: shuffle_order(k, rng) for k in shuffled_ks}

    # THE STIMULUS IMAGES. Built ONCE and reused for every model, so every
    # model is shown byte-identical stimuli. The PNG is encoded once per
    # stimulus too, because the blank condition's stimuli are byte-identical
    # to each other and re-encoding them per model per moment is pure waste.
    blank = np.full((H, W), 128, np.uint8)
    stimuli: dict[tuple, dict] = {}
    for cond in CONDITIONS:
        if cond == "blank":
            # The blank condition has no moments: its stimuli are K copies of
            # one uniform frame. They are still RUN at the full count, so the
            # denominators stay equal, but the images are identical.
            for spec in STIMULUS_SPECS:
                k = spec["k"]
                frames = [blank] * k
                img, order = make_tiles(frames, spec, perms.get(k))
                tiles = [upscale(blank, spec["scale"])] * k
                stimuli[(cond, spec["name"], 0)] = {
                    "img": img,
                    "tile_source_frames": ["generated: uniform 128 grey"] * k,
                    "tile_multiset_sha256": tile_multiset_hash(tiles),
                    "tile_order_sha256": tile_order_hash(tiles),
                }
            continue
        m = moments[cond]
        n_torn = len(m["Y_torn"])
        for i, anchor in enumerate(m["anchors"]):
            for spec in STIMULUS_SPECS:
                k = spec["k"]
                if spec.get("source") == "torn":
                    # THE TORN CONTROL, MADE LIKE-FOR-LIKE.
                    #
                    # The wrong-stride read has 750 frames where the right
                    # one has 900, so the index of a frame in one is not the
                    # index of the same WALL-CLOCK INSTANT in the other.
                    # Clean frame a starts at byte 19200*a; the wrong-stride
                    # frame that contains that byte is the one starting at
                    # byte 23040*floor(19200*a / 23040) = floor(a/1.2). So
                    # the torn counterpart of clean frame a is torn frame
                    # floor(a * 19200 / 23040) -- the same instant, torn.
                    #
                    # Without this mapping the torn control would be built
                    # from DIFFERENT moments than the clean strip it is
                    # supposed to control, and it would control for nothing.
                    tidx = int(anchor * STRIDE_CORRECT // STRIDE_TORN)
                    # Clamp ONCE, against the LONGEST window, so the K=4 and
                    # K=8 torn strips start at the SAME frame and the two are
                    # nested. Clamping per-K would slide the K=4 window one
                    # frame later than the K=8 one and the nesting would be
                    # an accident of the clamp.
                    tidx = min(tidx, n_torn - KMAX)
                    Y = m["Y_torn"]
                    base = tidx
                    src = [f"torn@{tidx + j}" for j in range(k)]
                else:
                    Y = m["Y"]
                    base = anchor
                    src = [str(anchor + j) for j in range(k)]
                frames = [Y[base + j] for j in range(k)]
                img, order = make_tiles(frames, spec, perms.get(k))
                tiles = [upscale(frames[j], spec["scale"]) for j in order]
                stimuli[(cond, spec["name"], i)] = {
                    "img": img,
                    "tile_source_frames": [src[j] for j in order],
                    "tile_multiset_sha256": tile_multiset_hash(tiles),
                    "tile_order_sha256": tile_order_hash(tiles),
                }

    # THE CALLS. One per (model, condition, stimulus, moment).
    #
    # The PNG is encoded ONCE per stimulus, not once per model, because every
    # model must be shown byte-identical stimuli and the encoding is the
    # expensive part.
    #
    # EQUAL DENOMINATORS. Every condition is scored with the same number of
    # moments. A previous bug in this project came from two conditions having
    # different frame counts, so the count is CHECKED here and the run stops
    # rather than reporting a rate over a denominator nobody can see.
    n_moments = {c: (len(moments[c]["anchors"]) if c != "blank"
                     else len(moments["speak"]["anchors"]))
                 for c in CONDITIONS}
    distinct = set(n_moments.values())
    if len(distinct) != 1:
        sys.exit(f"UNEQUAL DENOMINATORS, refusing to run: {n_moments}. "
                 f"Every condition must be scored with the same number of "
                 f"moments or the rates are not comparable.")
    n = distinct.pop()

    # THE CHECKPOINT. A long run that is killed at minute 40 must not lose 40
    # minutes, so every completed call is appended to a JSONL as it finishes
    # and a re-run resumes from it. The key is (model, condition, stimulus,
    # moment), which is unique per call.
    #
    # The checkpoint lives in /tmp, NOT in the repo: this file's write lane
    # is four paths, and a scratch file is not a deliverable.
    records_path = Path(a.records) if a.records else \
        Path("/tmp/vision_motion_teacher_records.jsonl")
    records_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if records_path.is_file() and not a.no_resume:
        for line in records_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec.get("model"), rec.get("condition"),
                   rec.get("stimulus"), rec.get("moment"))
            if key not in done_keys:
                done_keys.add(key)
        print(f"resuming: {len(done_keys)} calls already recorded in "
              f"{records_path}", flush=True)

    png_cache: dict[tuple, bytes] = {}
    jobs = []
    for model in models:
        for cond in CONDITIONS:
            for spec in STIMULUS_SPECS:
                k = spec["k"]
                for i in range(n):
                    key = (cond, spec["name"], 0 if cond == "blank" else i)
                    st = stimuli[key]
                    if key not in png_cache:
                        png_cache[key] = write_png(st["img"])
                    png = png_cache[key]
                    jobs.append({
                        "model": model,
                        "condition": cond,
                        "stimulus": spec["name"],
                        "k": k,
                        "scale": spec["scale"],
                        "order": spec["order"],
                        "moment": i,
                        "anchor": (moments[cond]["anchors"][i]
                                   if cond != "blank" else None),
                        "png_b64": base64.b64encode(png).decode(),
                        "png_sha256": hashlib.sha256(png).hexdigest(),
                        "image_width": st["img"].shape[1],
                        "image_height": st["img"].shape[0],
                        "tile_source_frames": st["tile_source_frames"],
                        "tile_multiset_sha256": st["tile_multiset_sha256"],
                        "tile_order_sha256": st["tile_order_sha256"],
                        "source_video": (moments[cond]["source_video"]
                                         if cond != "blank" else
                                         "generated: uniform 128 grey"),
                    })

    pending = [j for j in jobs
               if (j["model"], j["condition"], j["stimulus"], j["moment"])
               not in done_keys]
    print(f"{len(pending)} calls to make "
          f"({len(models)} models x {len(CONDITIONS)} conditions x "
          f"{len(STIMULUS_SPECS)} stimuli x {n} moments; "
          f"{len(jobs) - len(pending)} already done)", flush=True)

    records: list[dict] = []
    done = 0
    t0 = time.time()

    def run(job):
        r = vision_call(job["model"], job["png_b64"], Q_YESNO)
        text = (r.get("text") or r.get("response") or "")
        label = parse_speaking(text)
        rec = {k: v for k, v in job.items() if k != "png_b64"}
        rec["text"] = text[:300]
        rec["label"] = label
        rec["error"] = r.get("error")
        rec["detail"] = (r.get("detail") or "")[:300]
        rec["elapsed_s"] = r.get("elapsed_s")
        rec["agreement"] = r.get("agreement")
        rec["attempts"] = r.get("attempts")
        return rec

    if pending:
        with open(records_path, "a") as sink, \
                ThreadPoolExecutor(a.workers) as ex:
            for rec in ex.map(run, pending):
                records.append(rec)
                sink.write(json.dumps(rec) + "\n")
                sink.flush()
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(pending)}  {time.time() - t0:.0f}s  "
                          f"{pacing_report()}", flush=True)

    # EVERYTHING, including the calls carried over from the checkpoint.
    all_records = []
    seen = set()
    for line in records_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (rec.get("model"), rec.get("condition"),
               rec.get("stimulus"), rec.get("moment"))
        if key in seen:
            continue
        seen.add(key)
        all_records.append(rec)
    for rec in records:
        key = (rec["model"], rec["condition"], rec["stimulus"], rec["moment"])
        if key not in seen:
            seen.add(key)
            all_records.append(rec)
    records = all_records

    records.sort(key=lambda r: (r["model"], r["stimulus"], r["condition"],
                                 r["moment"]))

    models_out = {}
    for model in models:
        mr = [r for r in records if r["model"] == model]
        models_out[model] = score_model(model, mr)

    heads = {m: headline_for(models_out[m]) for m in models}

    out = {
        "experiment": "vision_motion_teacher",
        "question": (
            "Does a SEQUENCE of consecutive frames tell a vision-language "
            "model more about 'is this person speaking' than a SINGLE frame "
            "does? The control is the same frames in a scrambled order, which "
            "keeps appearance and destroys motion."),
        "hypothesis": (
            "A single frame cannot show motion, and the question asked of the "
            "model is about movement ('their mouth is moving'). So the "
            "published +0.4545 margin may be reading a single frame's "
            "mouth-open cue rather than motion. If it is reading motion, then "
            "an ORDERED strip of consecutive frames must beat the SAME frames "
            "in a SHUFFLED order."),
        "headline_definition": (
            "per model and per K: margin_over_silent(strip) - "
            "margin_over_silent(strip_shuffled), where margin_over_silent = "
            "sensitivity_on_speak - fpr_on_silent_person_present. Reported "
            "alongside the FPRs so it can be checked that the two strips ran "
            "at a matched false-positive rate."),
        "code": "experiments/vision_motion_teacher.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question_text": Q_YESNO,
        "request_shape": {
            "method": "POST",
            "pacing": pacing_report(),
            "pacing_note": (
                "The scheduler's per-model call interval at the end of the "
                "run, for the record. A rate-limit error doubles a model's "
                "interval and a success walks it back down; the interval is "
                "a scheduling detail and changes no stimulus and no answer."),
            "url": (f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}"
                    f"/ai/run/<MODEL>"),
            "headers": ["Authorization: Bearer <CF_TOKEN>",
                        "Content-Type: application/json"],
            "body_template": {
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "<question_text>"},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64,<PNG_B64>"}}]}],
                "max_tokens": 200,
                "temperature": 0.0,
            },
            "image_content_type": "image/png",
            "png_encoder": ("experiments/cloud_multimodal.write_png -- "
                            "greyscale (colour type 0), 8 bits per pixel"),
        },
        "image_construction": {
            "layout": ("horizontal strips: tiles side by side, TIME RUNS LEFT "
                       "TO RIGHT. The grid arm: 2x2, ROW-MAJOR, so time runs "
                       "left to right and then wraps to the next row."),
            "upscale_method": "bilinear",
            "upscale_note": (
                "Each TILE is upscaled by its stimulus's `scale` before "
                "tiling. Bilinear is a weighted sum of neighbouring REAL "
                "source pixels, so an upscaled tile carries exactly the "
                "information of the original frame and adds no structure that "
                "was not there. That is what makes `single_up` a test of "
                "RESOLUTION rather than of added detail."),
            "strip_definition": (
                "K CONSECUTIVE frames of one session, in time order, tiled "
                "into one image. The K=4 window is the first 4 frames of the "
                "K=8 window, so the two K arms are nested."),
            "shuffle_definition": (
                "the same K frames in a permuted order. The identity and "
                "every cyclic rotation are rejected, because a rotation "
                "preserves k-1 of the k adjacencies and so is not a "
                "motion-destroyed control."),
            "repeat_definition": (
                "K copies of ONE frame from that moment: the same pixel "
                "content k times, no temporal information at all."),
            "permutations": {str(k): v for k, v in perms.items()},
            "seed": a.seed,
            "stimuli": STIMULUS_SPECS,
            "image_sizes": {
                s["name"]: (f"{stimulus_size(s)[0]}x{stimulus_size(s)[1]}")
                for s in STIMULUS_SPECS},
            "confounds": [
                ("`strip_k8` has TWICE the total pixels of `strip_k4` "
                 "(2560x192 vs 1280x192) because the tiles are tiled, so the "
                 "K comparison is confounded with total image area. The "
                 "HEADLINE comparison (strip vs strip_shuffled at the SAME K) "
                 "has identical dimensions and identical pixels, so it is "
                 "free of this."),
                ("`single_up` (640x384) and `grid_k4` (640x384) have the "
                 "SAME dimensions and the SAME total pixel count, 245760, so "
                 "those two differ only in arrangement and multiplicity: one "
                 "big frame against four frames in a 2x2 grid."),
                ("`strip_k4` (1280x192) is 6.67:1. Llama 3.2 Vision has no "
                 "multi-image interface -- it tiles ONE image -- so a wide "
                 "strip arrives as a wide band divided into arbitrary crops. "
                 "The `grid_k4` arm exists to separate 'the frames do not "
                 "help' from 'the 6.67:1 shape does not help'."),
            ],
        },
        "frame_geometry": {
            "frame_size": [W, H],
            "measurement_stride_bytes": STRIDE_CORRECT,
            "measurement_stride_why": (
                "These captures are written by /mnt/c/proart_capture/"
                "nv12small.py as Y (15360) + U plane (1920) + V plane (1920) = "
                "19200 bytes per frame, PLANAR chroma, not the interleaved "
                "4:2:0 that W*H*3//2 = 23040 assumes. 19200 x 900 = "
                "17280000 = the exact file size, and 900 frames at 30.02 fps "
                "matches the 29.981 s WAV."),
            "control_stride_bytes": STRIDE_TORN,
            "control_stride_why": (
                "The standard 4:2:0 stride every reader in this repo used. "
                "Read at this stride the same files yield 750 frames and 80% "
                "of them are TORN: rows from frame k mixed with rows from "
                "frame k+1. The `*_torn` stimuli are built from that read, "
                "and are a CONTROL, never a measurement."),
            "fps": FPS,
            "sessions": {
                cond: {
                    "source_video": moments[cond]["source_video"],
                    "bytes": os.path.getsize(moments[cond]["source_video"]),
                    "clean_read": frame_stats(moments[cond]["Y"]),
                    "torn_read": frame_stats(moments[cond]["Y_torn"]),
                }
                for cond in moments
            },
            "mean_abs_consecutive_diff_note": (
                "The identity of a frame set. A real 30 fps webcam scene "
                "changes by a fraction of a grey level between consecutive "
                "frames; a torn read changes by tens of grey levels and by "
                "the SAME amount in every session, because it measures the "
                "tear and not the scene."),
        },
        "conditions": {
            cond: {
                "source_video": (moments[cond]["source_video"]
                                 if cond != "blank"
                                 else "generated: uniform 128 grey"),
                "label_source": (moments[cond]["label_source"]
                                 if cond != "blank"
                                 else "none (no person, no speech)"),
                "anchors": (moments[cond]["anchors"] if cond != "blank"
                            else []),
                "n_moments": n_moments[cond],
                "selection_rule": (moments[cond]["rule"] if cond != "blank"
                                   else "K copies of one uniform 128 grey "
                                        "frame; run at the same count as the "
                                        "real conditions so the denominators "
                                        "stay equal"),
            }
            for cond in CONDITIONS
        },
        "denominators": {
            "moments_per_condition": n_moments,
            "equal_across_conditions": True,
            "note": ("Every condition is scored with the SAME number of "
                     "stimuli, so every rate has the same denominator. The "
                     "blank condition's stimuli are byte-identical to each "
                     "other, so its rate is a per-call rate on ONE image; it "
                     "is run at the full count to keep the denominators "
                     "equal, as `cloud_multimodal.py` did."),
        },
        "degenerate_baselines": DEGENERATE_BASELINES,
        "base_rate_note": (
            "The published sensitivity of 0.4545 must be read against a "
            "constant-answer baseline, NOT against 0. `always_NOT_SPEAKING` "
            "has sensitivity 0.000 and `always_SPEAKING` has sensitivity "
            "1.000; both score a margin of exactly 0.000. A margin is the "
            "only number here that a constant answer cannot fake."),
        "models_probed": {
            "swept": ("all 73 catalogue entries of account "
                      "049ff5e84ecf636b53b162cbb580aae6 were called once with "
                      "this exact request shape; the catalogue is the "
                      "account's, not a guess"),
            "usable": models,
            "preflight": preflight,
            "failed": {
                "@cf/meta/llama-4-scout-17b-vision-instruct":
                    "the model named in the brief does NOT exist on this "
                    "account; the catalogue entry is "
                    "@cf/meta-llama-4-scout-17b-16e-instruct and that is "
                    "what was asked for. Its RUN ROUTE is INTERMITTENT: it "
                    "answers normally sometimes and returns Cloudflare error "
                    "code 7000 'No route for that URI' at other times, so it "
                    "is pre-flighted before the run and dropped if it does "
                    "not answer. See `preflight` above for what happened on "
                    "this run.",
                "@cf/llava-hf/llava-1.5-7b-hf":
                    "HTTP 400, oneOf at '/' not met -- the known shape "
                    "failure, unchanged",
                "@cf/moondream/moondream3.1-9B-A2B":
                    "HTTP 200 but the result object is EMPTY ({}); no text to "
                    "parse",
                "@cf/zai-org/glm-5.3":
                    "HTTP 200 but empty content; the model's own reasoning "
                    "content says it has no multimodal input ability",
                "@cf/cloudflare/clairic": "HTTP 400: needs {model, fields}",
                "@cf/cloudflare/clef": "HTTP 400: needs {model, state, questions}",
                "@cf/cloudflare/clef-flash":
                    "HTTP 400: needs {model, state, questions}",
                "@cf/cloudflare/clef-omni":
                    "HTTP 400: needs {model, state, questions}",
                "@cf/qwen/qwq-32b": "HTTP 400: 'not a multimodal model'",
                "@cf/zai-org/glm-4.7-flash":
                    "HTTP 400: 'not a multimodal model'",
                "@cf/zai-org/glm-5.2":
                    "HTTP 400: 'not a multimodal model'",
                "@cf/deepseek-ai/deepseek-v4-flash-0731":
                    "HTTP 400: model-only-supports-text",
                "@cf/swiss-ai/apertus-v1.5-8b":
                    "HTTP 403: account not allowed to access",
                "all remaining catalogue entries":
                    "HTTP 400 on this request shape -- text-only models "
                    "reject the image content type",
            },
            "licence_gate_note": (
                "Meta's gate wants a TOP-LEVEL {\"prompt\": \"agree\"} alone. "
                "The Llama 4 Scout endpoint rejects that shape with "
                "Cloudflare code 7000 'No route for that URI' while "
                "accepting the messages shape and answering normally, so it "
                "is recorded as 'no gate on this endpoint' rather than as a "
                "failed agreement. Treating it as a failed agreement would "
                "have silently killed the whole scout arm."),
        },
        "models": models_out,
        "headlines": heads,
        "calls": records,
        "limits": [
            "one session per condition, one speaker, one room, one lighting "
            "state",
            f"{a.frames} moments per condition and frames from one session "
            "are correlated, so the effective sample size is far below the "
            "frame count; the resolution of a rate is one divided by the "
            "denominator",
            "the speech labels come from the Silero gate applied to the WAV; "
            "per AV-SYNC.md A.wav and A.nv12 do not correspond, so the "
            "per-frame labels are provisional",
            "the model's own internal image resizing is not observable from "
            "here, so what the model actually sees after resizing is unknown; "
            "the ORDER of the tiles survives any whole-image resize, which "
            "is why the shuffled control is valid and the resolution "
            "question is not settled by it",
            "one measurement of one model on one capture is one measurement, "
            "not a property",
        ],
    }

    p = Path(a.out) / "vision_motion_teacher_results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    for model in models:
        print(f"\n=== {model} ===")
        for spec in STIMULUS_SPECS:
            s = models_out[model][spec["name"]]
            errs = sum(s["conditions"][c]["n_errors"] for c in CONDITIONS)
            print(f"  {spec['name']:20s} sens={s['sensitivity_on_speak']} "
                  f"fpr_silent={s['fpr_on_silent_person_present']} "
                  f"fpr_blank={s['fpr_on_blank']} "
                  f"margin={s['margin_over_silent']} errors={errs}")
        for kk, hh in heads[model].items():
            print(f"  HEADLINE {kk}: {hh['verdict']}")


if __name__ == "__main__":
    main()
