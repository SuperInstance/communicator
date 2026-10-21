#!/usr/bin/env python3
"""
room_map.py -- what the camera knows about the room.

WHAT THIS IS.

The free-session corpus holds 139,469 luma frames of one room, seen by
one fixed laptop webcam. That is a lot of evidence about a space, and
this tool turns it into three things:

  1. THE COMPOSITE. The static background of the room, as a per-pixel
     median. The median is the point: a person moves, a room does not,
     so the median is the room with the moving things taken out.

  2. THE MOTION MAP. How much each pixel changes, frame to frame,
     averaged over the sessions. This is where the room's life is.

  3. THE PRESENCE MAP. How often each pixel holds something other than
     the room's own value. This is where the room's life is *held*.

Then the regions: the connected pieces of the presence map, each with
its bounds in frame coordinates, how much of the time it was occupied,
and how it behaves.

THE EMPTY ROOM IS FOUND, NOT ASSUMED.

The corpus's most common view is NOT an empty room. Over the two hours
of capture, Casey was at the desk more often than not, so the plain
median over every frame has him in it. The empty room has to be found.

The tool finds it without being told what a person looks like:

  * the frame is divided into a grid of blocks;
  * for each block, the per-frame mean luma is gathered across the whole
    corpus, and the block whose distribution splits most cleanly into
    two populations is chosen (Otsu's between-class variance);
  * the two populations are the frames with something in that block and
    the frames without;
  * the population with the TIGHTER spread is the empty one, because a
    room does not change and a person does. That is the only assumption
    in the chain, and it is an assumption about the physics rather than
    about what Casey looks like.

The block that wins and the numbers it split on are reported, so the
choice can be checked rather than trusted.

THE HONESTY CLAUSE IS THE POINT.

This is the room AS THE CAMERA KNOWS IT. It is an image plane, not a
floor plan. The camera sees a wall, a ceiling and a desk from one
angle; it cannot tell a door from a picture of a door, it cannot measure
a distance, and it cannot see behind anything. Every number here is in
FRAME COORDINATES -- x from 0 to 159, y from 0 to 95 -- and a frame
coordinate is a place in the picture, not a place in the world.

THE PARTS ARE THE REPOSITORY'S.

The corpus format is the recorder's own: `luma.raw` is uint8 BT.601
luma, 160x96, frames concatenated (`meta.json` says so). The
frame-to-frame difference is the same quantity `motion_as_text.py` and
`corpus_capture_win.py:change_maps` already use, at full resolution
here because a heatmap wants the pixels and not a 2x2 average.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

FRAME_W = 160
FRAME_H = 96
FRAME_BYTES = FRAME_W * FRAME_H

#: The default corpus: the recorder's own output directory.
DEFAULT_CORPUS = Path("/mnt/c/proart_capture/corpus/free_sessions")

#: The grid the blocks are chosen from: 8 across by 6 down.
GRID_X = 8
GRID_Y = 6

#: A pixel counts as occupied when it differs from the empty room by more
#: than this, after a 3x3 box blur so a single hot pixel is not a
#: presence and a person's edge counts over its neighbourhood. This is a
#: FLOOR in luma levels, for the pixels whose own measured noise is
#: essentially zero: without it, one step of quantisation would read as
#: a person.
PRESENCE_FLOOR = 4.0

#: ... and by more than this many of its own noise sigma, measured at
#: that pixel from the empty frames. A quiet wall and a textured shelf
#: do not have the same noise, and one threshold for both of them says
#: the whole frame is occupied.
NOISE_SIGMAS = 6.0

#: The motion map is measured in noise sigmas per frame rather than in
#: luma levels, so that a textured shelf and a blank wall can be
#: compared. This is the floor on the per-pixel difference noise; below
#: it the divisor would turn quantisation into movement.
MOTION_FLOOR = 0.75

#: Frames per session kept for the coarse statistics.
SAMPLE_FRAMES = 400

#: A stretch of the room is called EMPTY when nothing held it for more
#: than one frame in twenty. That is the floor the quiet regions are
#: found at, and it is a plain-language number on purpose.
QUIET_FLOOR = 0.05


# ============================================================================
# 1. READING THE CORPUS
# ============================================================================

def session_frames(session_dir: Path) -> np.ndarray:
    """The session's luma, as a memmap of (frames, 96, 160)."""
    p = session_dir / "luma.raw"
    if not p.is_file():
        return np.zeros((0, FRAME_H, FRAME_W), dtype=np.uint8)
    n = os.path.getsize(p) // FRAME_BYTES
    if n < 1:
        return np.zeros((0, FRAME_H, FRAME_W), dtype=np.uint8)
    return np.memmap(p, dtype=np.uint8, mode="r").reshape(-1, FRAME_H, FRAME_W)


def survey(corpus: Path) -> list[dict]:
    """Every session in the corpus, and what it holds."""
    out = []
    if not corpus.is_dir():
        return out
    for d in sorted(corpus.iterdir(), key=lambda p: (not p.name.isdigit(),
                                                      p.name)):
        if not d.is_dir():
            continue
        luma = d / "luma.raw"
        meta_path = d / "meta.json"
        n = 0
        if luma.is_file():
            n = os.path.getsize(luma) // FRAME_BYTES
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:  # noqa: BLE001
                meta = {}
        out.append({
            "session": d.name,
            "path": str(d),
            "frames": int(n),
            "trigger": meta.get("trigger"),
            "mode": meta.get("mode"),
            "close_reason": meta.get("close_reason"),
            "probe": (meta.get("probe") or {}).get("name"),
        })
    return out


def usable_sessions(sessions: list[dict]) -> list[dict]:
    """The sessions that have frames and are not a probe artifact."""
    return [s for s in sessions if s["frames"] >= 2 and not s["probe"]]


# ============================================================================
# 2. THE HISTOGRAM, AND THE EXACT MEDIAN
# ============================================================================

def accumulate_histogram(hist: np.ndarray, frames: np.ndarray) -> None:
    """Add a chunk of frames into a per-pixel value histogram.

    The histogram is (pixels, 256) of counts. It is built with one
    `bincount` over a flat index of (value, pixel), which is exact and
    fast, and it is what makes the median exact rather than sampled.
    """
    if frames.size == 0:
        return
    f = frames.reshape(frames.shape[0], -1)
    npix = f.shape[1]
    pix = np.arange(npix, dtype=np.int64)
    idx = (f.astype(np.int64) * npix + pix[None, :]).ravel()
    counts = np.bincount(idx, minlength=256 * npix)
    hist += counts.reshape(256, npix).T.astype(hist.dtype)


def median_from_histogram(hist: np.ndarray) -> np.ndarray:
    """The per-pixel median of a histogram, as uint8.

    The median is the lowest value whose cumulative count reaches half
    the frames. A histogram of an even count has two middle values; the
    lower one is taken, and the rule is stated rather than left to be
    inferred. (It matters: it is the reason the plain median over the
    whole corpus has Casey in it, and the reason the empty room has to
    be found rather than assumed.)
    """
    total = hist.sum(axis=1)
    half = total / 2.0
    cum = np.cumsum(hist, axis=1)
    idx = (cum >= half[:, None]).argmax(axis=1)
    return idx.astype(np.uint8)


# ============================================================================
# 3. FINDING THE EMPTY ROOM
# ============================================================================

def block_means(frames: np.ndarray) -> np.ndarray:
    """The mean luma of every grid block, per frame.

    (frames, GRID_Y, GRID_X). The blocks are the whole frame divided
    evenly, and the remainder is spread over the last blocks rather than
    dropped.
    """
    if frames.shape[0] == 0:
        return np.zeros((0, GRID_Y, GRID_X), dtype=np.float32)
    ys = np.linspace(0, FRAME_H, GRID_Y + 1).astype(int)
    xs = np.linspace(0, FRAME_W, GRID_X + 1).astype(int)
    out = np.zeros((frames.shape[0], GRID_Y, GRID_X), dtype=np.float32)
    for by in range(GRID_Y):
        y0, y1 = ys[by], max(ys[by] + 1, ys[by + 1])
        for bx in range(GRID_X):
            x0, x1 = xs[bx], max(xs[bx] + 1, xs[bx + 1])
            out[:, by, bx] = frames[:, y0:y1, x0:x1].reshape(frames.shape[0], -1).mean(axis=1)
    return out


def otsu(values: np.ndarray, bins: int = 256) -> tuple[float, float]:
    """Otsu's threshold and the between-class variance ratio.

    The ratio is 1.0 when the two classes are as far apart as the data
    allows and 0.0 when the split explains nothing. It is the score
    used to choose which block of the frame best separates "something
    is there" from "nothing is there".
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size < 4:
        return 0.0, 0.0
    hist, edges = np.histogram(v, bins=bins)
    total = v.size
    centres = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)
    w1 = total - w0
    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        return 0.0, 0.0
    s0 = np.cumsum(hist * centres)
    s1 = s0[-1] - s0
    m0 = np.where(w0 > 0, s0 / np.maximum(w0, 1), 0.0)
    m1 = np.where(w1 > 0, s1 / np.maximum(w1, 1), 0.0)
    between = w0 * w1 * (m0 - m1) ** 2
    between[~valid] = 0.0
    var_total = float(((v - v.mean()) ** 2).mean())
    ratio = float(between.max() / (total * var_total)) if var_total > 0 else 0.0
    idx = int(between.argmax())
    return float(centres[idx]), ratio


def refine_threshold(values: np.ndarray, thr: float,
                     significant: float = 0.2) -> float:
    """Put the split in the GAP between the two populations.

    Otsu returns a bin centre, and with a well-separated pair of
    populations that centre can land just inside one of them, leaking a
    few frames into the wrong half. The gap between the two populations
    is the real boundary. Among the gaps in the sorted values that are
    big enough to be a class boundary -- a fifth of the whole range --
    the one closest to where Otsu already pointed is the boundary.
    """
    v = np.sort(np.asarray(values, dtype=np.float64))
    if v.size < 4:
        return thr
    mid = 0.5 * (v[:-1] + v[1:])
    gaps = np.diff(v)
    span = (v[-1] - v[0]) * significant
    big = gaps >= span
    if not big.any():
        return thr
    idx = np.where(big)[0]
    i = int(idx[np.argmin(np.abs(mid[idx] - thr))])
    return float(mid[i])


def population_tightness(frames: np.ndarray, sel: np.ndarray,
                         limit: int = 200) -> float | None:
    """How alike a population's frames are, as images.

    The empty room's frames are the SAME room, so they agree with each
    other closely. The occupied frames are not the same image at all --
    the person is somewhere different in each one -- so they agree with
    each other less well. The mean distance from each frame to its own
    population's median frame is the measure, and the smaller one is the
    empty room.

    This is the physical statement, and it does not assume that a person
    is darker than the room or smaller than it.
    """
    take = frames[sel]
    if take.shape[0] < 4:
        return None
    if take.shape[0] > limit:
        idx = np.linspace(0, take.shape[0] - 1, limit).astype(int)
        take = take[idx]
    centre = np.median(take.astype(np.float32), axis=0)
    d = np.abs(take.astype(np.float32) - centre).mean(axis=(1, 2))
    return float(d.mean())


def find_empty_population(block_stats: np.ndarray,
                          sample_frames: np.ndarray) -> dict:
    """Choose the block that best separates the two populations, and
    split the corpus on it.

    `block_stats` is (frames, GRID_Y, GRID_X) over EVERY frame;
    `sample_frames` is (frames, 96, 160) over a SAMPLE, aligned with
    the first part of `block_stats`. The winner is the block with the
    highest between-class variance ratio. The two populations are then
    split by Otsu's threshold, and the one whose frames agree with each
    other most closely is the empty one: a room does not change and a
    person does.
    """
    best = None
    for by in range(GRID_Y):
        for bx in range(GRID_X):
            vals = block_stats[:, by, bx]
            thr, ratio = otsu(vals)
            if best is None or ratio > best["ratio"]:
                best = {"block": [bx, by], "ratio": ratio, "threshold": thr,
                        "values": vals}
    vals = best["values"]
    thr = refine_threshold(vals, best["threshold"])
    low = vals <= thr
    high = ~low
    stats = {}
    for name, sel in (("low", low), ("high", high)):
        if sel.sum() < 2:
            stats[name] = {"n": int(sel.sum()), "sd": None, "mean": None}
            continue
        stats[name] = {"n": int(sel.sum()), "sd": float(vals[sel].std()),
                       "mean": float(vals[sel].mean())}
    # the population whose FRAMES agree with each other is the room
    tight = {}
    for name, sel in (("low", low), ("high", high)):
        tight[name] = population_tightness(sample_frames, sel)
    empty_name = None
    ambiguous = False
    if tight["low"] is not None and tight["high"] is not None:
        lo, hi = tight["low"], tight["high"]
        # a person who sits perfectly still agrees with themselves as much
        # as a room does, and then the split cannot be made on this
        # evidence. That is reported rather than hidden.
        spread = max(abs(lo - hi) / max(1e-9, max(lo, hi)), 1e-9)
        ambiguous = spread < 0.10
        empty_name = "low" if lo <= hi else "high"
    empty_mask = low if empty_name == "low" else high
    return {
        "block": best["block"],
        "ratio": round(best["ratio"], 5),
        "otsu_threshold": round(best["threshold"], 3),
        "threshold": round(thr, 3),
        "populations": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                            for kk, vv in v.items()}
                        for k, v in stats.items()},
        "frame_agreement": {k: (None if v is None else round(v, 4))
                            for k, v in tight.items()},
        "empty_population": empty_name,
        "ambiguous": ambiguous,
        "empty_mask": empty_mask,
        "reason": ("the population whose frames agree with each other most "
                   "closely is the empty room: a room does not change and a "
                   "person does"),
    }


# ============================================================================
# 3b. FITTING THE EXPOSURE OUT
# ============================================================================

def _median(x: np.ndarray, stride: int = 4) -> float:
    """A median taken over a strided sample of the pixels.

    The median is the robust centre: a person covering part of the
    frame must not be allowed to drag the frame's own centre with it.
    Every pixel is not needed for that -- a quarter of them is enough,
    and it keeps the fit cheap enough to run on every frame of the
    corpus.
    """
    v = np.asarray(x).ravel()[::stride]
    if v.size == 0:
        return 0.0
    return float(np.median(v))


def _median_rows(x: np.ndarray, stride: int = 4) -> np.ndarray:
    """The per-row median of a (k, P) stack, over a strided sample."""
    v = np.asarray(x)[:, ::stride]
    if v.shape[1] == 0:
        return np.zeros(v.shape[0], dtype=np.float32)
    return np.median(v, axis=1).astype(np.float32)


def fit_gain_offset(ref: np.ndarray, img: np.ndarray) -> tuple[float, float]:
    """The gain and the offset that carry `ref` onto `img`.

    The camera's auto-exposure changes the level AND the gain of the
    whole frame, and it changes them between sessions. The empty frames
    -- the ones that decide what the room looks like -- came out at a
    different exposure from the rest of the corpus, so a straight
    subtraction would read the exposure change as occupancy.

    The fit is a least squares over the whole frame, on MEDIAN-centred
    values. The median and not the mean, because the thing being fitted
    out is often the very thing the measurement is about: a person
    covering a quarter of the frame would drag a mean-centred fit with
    it, and the person would then be partly absorbed into the exposure
    and partly read as a change in the room.

    The gain is clamped to a sane range. A fit that has gone wrong must
    not be allowed to invent contrast out of nothing, and a negative
    gain through two luma levels fits a bright patch on a dark
    background exactly -- which would make the room read as empty.
    """
    f = np.asarray(img, dtype=np.float32).ravel()
    r = np.asarray(ref, dtype=np.float32).ravel()
    fm = _median(f)
    rm = _median(r)
    fmad = float(np.median(np.abs(f[::4] - fm)))
    rmad = float(np.median(np.abs(r[::4] - rm)))
    a = (fmad / rmad) if rmad > 1e-6 else 1.0
    a = min(max(a, 0.4), 2.5)
    return a, fm - a * rm


def residual_frame(ref: np.ndarray, img: np.ndarray) -> np.ndarray:
    """The frame with the room's exposure fitted out of it."""
    a, b = fit_gain_offset(ref, img)
    return (np.asarray(img, dtype=np.float32)
            - (a * np.asarray(ref, dtype=np.float32) + b))


def robust_sigma(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """A per-pixel scale that ignores the outliers.

    The median absolute deviation, scaled to be comparable with a
    standard deviation for a normal distribution. The empty frames are
    not perfectly empty -- a person in another part of the room would
    still be in them -- so the scale is measured robustly rather than
    with a standard deviation, which would be dragged up by exactly the
    frames that matter.
    """
    v = np.asarray(values, dtype=np.float32)
    med = np.median(v, axis=axis, keepdims=True)
    mad = np.median(np.abs(v - med), axis=axis)
    return 1.4826 * mad


# ============================================================================
# 3c. THE VIEWPOINTS
# ============================================================================

def shift_image(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """The image moved by (dy, dx), with the vacated edge filled."""
    out = np.full_like(a, 255.0)
    ys0, ys1 = max(0, dy), min(a.shape[0], a.shape[0] + dy)
    xs0, xs1 = max(0, dx), min(a.shape[1], a.shape[1] + dx)
    out[ys0:ys1, xs0:xs1] = a[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def find_viewpoints(sessions: list[dict], corpus: Path, composite: np.ndarray,
                    frames_by_session: dict, band: tuple[int, int] = (0, 40),
                    reach: int = 4, limit: int = 40) -> list[dict]:
    """Where each session was shot from, and how well it agrees.

    THE CAMERA IS NOT QUITE FIXED. The room is one room and the laptop
    stays on the same desk for the whole night, but the laptop was
    nudged a couple of pixels at some point in the night, and a map
    built across the nudge is a map of two slightly different pictures
    averaged into one. The nudge is also invisible in a median: it
    moves the edges of everything by two pixels and leaves the middle
    of every surface exactly where it was.

    So the corpus is SPLIT BY WHERE IT WAS SHOT FROM, and the split is
    measured rather than guessed:

      * each session is compared against the composite at every shift
        in a small window, and the shift that agrees best is kept;
      * the comparison is made on the TOP band of the frame, which is
        ceiling and beam and never a person. A person in the frame
        would drag the score and make the measurement about occupancy
        instead of about place.

    A session whose best shift is (0, 0) and whose score is at the
    floor is at the same place as the composite. A session whose best
    shift is (0, -3) was shot three pixels lower.
    """
    comp = np.asarray(composite, dtype=np.float32)
    y0, y1 = band
    out = []
    for s in sessions:
        arr = frames_by_session.get(s["session"])
        if arr is None or arr.shape[0] < 2:
            continue
        step = max(1, arr.shape[0] // limit)
        idx = np.linspace(0, arr.shape[0] - 1, min(limit, arr.shape[0])).astype(int)
        take = np.asarray(arr[idx], dtype=np.float32)[:, y0:y1, :]
        best = None
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                cs = shift_image(comp, dy, dx)[y0:y1, :]
                d = np.abs(take - cs)
                d = d - np.median(d, axis=(1, 2), keepdims=True)
                score = float(np.median(np.abs(d)))
                if best is None or score < best[0]:
                    best = (score, dy, dx)
        out.append({"session": s["session"], "frames": int(arr.shape[0]),
                    "shift_y": best[1], "shift_x": best[2],
                    "score": round(best[0], 4)})
    out.sort(key=lambda r: (r["score"], -r["frames"]))
    return out


# ============================================================================
# 4. THE HEATMAPS
# ============================================================================

def blur3(a: np.ndarray) -> np.ndarray:
    """A 3x3 box blur, edge-padded. Works on one frame or on a stack."""
    single = a.ndim == 2
    if single:
        a = a[None]
    p = np.pad(a.astype(np.float32), ((0, 0), (1, 1), (1, 1)), mode="edge")
    out = np.zeros_like(a, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += p[:, dy:dy + a.shape[1], dx:dx + a.shape[2]]
    out /= 9.0
    return out[0] if single else out


def _mad_rows(x: np.ndarray, stride: int = 4) -> np.ndarray:
    """The per-row median absolute deviation -- a robust spread."""
    v = np.asarray(x)[:, ::stride]
    if v.shape[1] == 0:
        return np.zeros(v.shape[0], dtype=np.float32)
    med = np.median(v, axis=1, keepdims=True)
    return np.median(np.abs(v - med), axis=1).astype(np.float32)


def residual_batch(ref: np.ndarray, img: np.ndarray) -> np.ndarray:
    """A stack of frames with the reference's exposure fitted out.

    `ref` is (P,) or (k, P); `img` is (k, P). The gain and the offset are
    fitted per frame, so a frame that is brighter than the room -- or
    brighter in its dark parts than in its light parts -- is not mistaken
    for a change in the room.

    THE FIT IS ROBUST, AND IT USES TWO STATISTICS THAT A PERSON IN THE
    FRAME CANNOT MOVE.

    The thing being fitted out is often the very thing the measurement
    is about. Casey at the desk is a big dark mass covering a quarter
    of the frame. A plain least squares fit lets that mass pull the
    slope down: the fit comes out with a gain of 0.45, and every pixel
    of the frame is then left sitting thirty luma levels away from the
    room -- which reads as the whole frame being occupied, everywhere,
    all the time.

    So the fit is started from statistics that a quarter of the frame
    going dark cannot move:

      * the GAIN starts as the ratio of the two pictures' robust
        spreads. If the frame is the room at twice the brightness, its
        spread is twice the spread; if a quarter of it is covered by
        something dark, the spread of the rest is unchanged. The median
        absolute deviation is used for the spread, because it ignores
        the tail the occluder is in;
      * the OFFSET starts as the difference of the two pictures'
        medians, for the same reason.

    That start is close but not exact, so the fit is then REFINED on
    the pixels that agree with the room: the ones whose residual is
    small are the background, and the background alone can say what the
    exposure really was. The occluder cannot vote, because the occluder
    is exactly what did not agree.

    What is left over is the residual: the room, minus the room, with
    whatever was in front of it still in it.
    """
    img = np.asarray(img, dtype=np.float32)
    k, P = img.shape
    if ref.ndim == 1:
        rm = float(_median(ref))
        fm = _median_rows(img)
        rmad = float(np.median(np.abs((ref - rm)[::4])))
        imad = _mad_rows(img)
        a = np.where(imad > 1e-6, imad / max(rmad, 1e-6), 1.0)
        a = np.clip(a, 0.4, 2.5)
        b = fm - a * rm
        rc = (ref - rm).astype(np.float32)
        for _ in range(2):
            resid = img - (a[:, None] * ref.astype(np.float32)[None, :] + b[:, None])
            scale = _mad_rows(resid)
            keep = (np.abs(resid) <= np.maximum(3.0 * scale[:, None], 1.0)).astype(np.float32)
            sxx = (keep * rc * rc).sum(axis=1)
            step = (keep * resid * rc).sum(axis=1) / np.maximum(sxx, 1e-6)
            a = np.clip(a + step, 0.4, 2.5)
            b = _median_rows(img - a[:, None] * ref.astype(np.float32)[None, :])
        return img - (a[:, None] * ref.astype(np.float32)[None, :] + b[:, None])
    ref = np.asarray(ref, dtype=np.float32)
    rm = _median_rows(ref)
    im = _median_rows(img)
    rmad = _mad_rows(ref)
    imad = _mad_rows(img)
    a = np.where(imad > 1e-6, imad / np.maximum(rmad, 1e-6), 1.0)
    a = np.clip(a, 0.4, 2.5)
    b = im - a * rm
    rc = ref - rm[:, None]
    for _ in range(2):
        resid = img - (a[:, None] * ref + b[:, None])
        scale = _mad_rows(resid)
        keep = (np.abs(resid) <= np.maximum(3.0 * scale[:, None], 1.0)).astype(np.float32)
        sxx = (keep * rc * rc).sum(axis=1)
        step = (keep * resid * rc).sum(axis=1) / np.maximum(sxx, 1e-6)
        a = np.clip(a + step, 0.4, 2.5)
        b = _median_rows(img - a[:, None] * ref)
    return img - (a[:, None] * ref + b[:, None])


def measure_noise(empty_frames: np.ndarray, composite: np.ndarray,
                  diff_pairs: list[np.ndarray] | None = None) -> dict:
    """The noise of this room, measured where nothing is happening.

    TWO NOISES, AND THEY HAVE TO BE MEASURED IN THE SAME UNITS AS THE
    TEST THAT USES THEM. The first version of this tool measured the
    noise on RAW luma and then applied it to an exposure-corrected
    residual, so the threshold came out at 182 to 298 luma levels --
    above the top of the 8-bit range, which is why nothing in the frame
    was ever occupied and the presence map came out flat and black. A
    threshold that is not in the units of the quantity it thresholds is
    not a threshold.

      * the presence noise: how far a pixel moves when the room is
        empty and the person is nowhere near it. This is the scale the
        occupancy test is measured against.
      * the motion noise: how much a pixel changes BETWEEN two frames
        of an empty room. This is the divisor that turns the motion map
        from a texture map into an activity map.

    The empty frames are not perfectly empty -- a person standing
    somewhere else in the room would still be in them -- so both scales
    are measured with a median absolute deviation, which ignores the
    outliers that a standard deviation would be dragged up by.
    """
    n = empty_frames.shape[0]
    if n < 2:
        z = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        return {"presence_sigma": z, "motion_sigma": z, "n_empty": int(n),
                "n_pairs": 0}
    comp = np.asarray(composite, dtype=np.float32).ravel()
    resid = residual_batch(comp, empty_frames.reshape(n, -1))
    sigma_presence = robust_sigma(resid, axis=0).reshape(FRAME_H, FRAME_W)

    # the motion noise: the exposure-fitted difference between ADJACENT
    # frames of an empty room. A jump between two sessions is a cut, not
    # a jitter, so the pairs are collected within a session.
    if diff_pairs:
        d = np.concatenate(diff_pairs, axis=0)
        sigma_motion = robust_sigma(d, axis=0).reshape(FRAME_H, FRAME_W)
        n_pairs = int(d.shape[0])
    else:
        sigma_motion = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        n_pairs = 0
    return {"presence_sigma": sigma_presence,
            "motion_sigma": sigma_motion,
            "n_empty": int(n), "n_pairs": n_pairs}


def heatmaps(sessions: list[dict], corpus: Path, composite: np.ndarray,
             noise: dict, frames_by_session: dict | None = None) -> dict:
    """The motion map and the presence map, over every frame of every
    session.

    The maps are averaged PER SESSION and then across the sessions, so
    a 15,514-frame session does not drown out a 122-frame one.

    THE PRESENCE TEST IS PER PIXEL AND IN THE PIXEL'S OWN UNITS. For
    two reasons that the first version of this tool got wrong:

      * the camera's auto-exposure changes the level and the gain of
        the whole frame between frames, so an un-corrected comparison
        says every pixel is occupied all the time -- the map came out
        flat at 0.37 with no structure in it at all. Each frame's own
        gain and offset are therefore fitted out before the comparison;
      * a pixel's own noise is not the same as another's, so the test
        is against the noise MEASURED at that pixel from the empty
        frames, not against one number for the whole image.

    THE MOTION MAP IS IN NOISE SIGMAS, NOT IN LUMA. The raw map --
    mean |dframe| per pixel -- is a TEXTURE map: a bookshelf is all
    edges and noise, so it scores high, and a blank wall scores low,
    whether or not anything moves in front of either. Dividing by the
    noise each pixel makes when nothing happens turns it into what it
    was asked for: where the room's life happens.
    """
    motion_sum = np.zeros((FRAME_H, FRAME_W), dtype=np.float64)
    luma_sum = np.zeros((FRAME_H, FRAME_W), dtype=np.float64)
    presence_sum = np.zeros((FRAME_H, FRAME_W), dtype=np.float64)
    n_sessions = 0
    n_frames = 0
    n_changes_total = 0
    comp = np.asarray(composite, dtype=np.float32).ravel()
    thr = np.maximum(PRESENCE_FLOOR,
                     NOISE_SIGMAS * noise["presence_sigma"].ravel())
    sigma_m = np.maximum(MOTION_FLOOR, noise["motion_sigma"].ravel())
    for s in sessions:
        arr = (frames_by_session or {}).get(s["session"])
        if arr is None:
            arr = session_frames(corpus / s["session"])
        if arr.shape[0] < 2:
            continue
        n = arr.shape[0]
        # each session's own sums, divided by that session's own length
        # BEFORE they are added in, so that a long session and a short
        # one each count once
        motion = np.zeros((FRAME_H, FRAME_W), dtype=np.float64)
        luma = np.zeros((FRAME_H, FRAME_W), dtype=np.float64)
        occupied = np.zeros((FRAME_H, FRAME_W), dtype=np.float64)
        # THE MOTION MAP IS PER CHANGE, NOT PER FRAME. The recorder only
        # writes a frame when the image has actually changed, so two
        # thirds of the consecutive frames in this corpus are the same
        # bytes twice. Dividing by the number of frames would mix two
        # different things -- how OFTEN the picture changes, and how
        # MUCH it changes when it does -- and a room where nothing ever
        # moves would score the same as a room where something is
        # always moving. The divisor here is the number of changes.
        n_changes = 0
        for i in range(1, n, 400):
            j = min(i + 400, n)
            cur = np.asarray(arr[i:j], dtype=np.float32).reshape(j - i, -1)
            prev = np.asarray(arr[i - 1:j - 1], dtype=np.float32).reshape(j - i, -1)
            d = np.abs(residual_batch(prev, cur))
            motion += (d / sigma_m).sum(axis=0).reshape(FRAME_H, FRAME_W)
            luma += d.sum(axis=0).reshape(FRAME_H, FRAME_W)
            n_changes += int((d.sum(axis=1) > 0).sum())
        # PRESENCE: how often the pixel holds something other than the room.
        for i in range(0, n, 400):
            chunk = np.asarray(arr[i:i + 400], dtype=np.float32)
            r = np.abs(residual_batch(comp, chunk.reshape(chunk.shape[0], -1)))
            r = blur3(r.reshape(chunk.shape[0], FRAME_H, FRAME_W))
            occupied += (r.reshape(chunk.shape[0], -1)
                         > thr[None, :]).sum(axis=0).reshape(FRAME_H, FRAME_W)
        motion_sum += motion / max(1, n_changes)
        luma_sum += luma / max(1, n_changes)
        presence_sum += occupied / n
        n_sessions += 1
        n_frames += int(n)
        n_changes_total += n_changes
    return {
        "motion": motion_sum / max(1, n_sessions),
        "motion_luma": luma_sum / max(1, n_sessions),
        "presence": presence_sum / max(1, n_sessions),
        "n_sessions": n_sessions,
        "n_frames": n_frames,
        "n_changes": n_changes_total,
    }


# ============================================================================
# 5. THE REGIONS
# ============================================================================

def label_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """The connected components of a mask, 4-connected.

    A small flood fill rather than a dependency: scipy is not guaranteed
    here and the image is 160x96.
    """
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    h, w = mask.shape
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w \
                            and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(comp)
    return comps


def region_name(box: dict) -> str:
    """A GEOMETRIC name for a region, from where it is in the frame.

    This is not a claim about what the region IS. The desk and the
    person are named in the report, where the reasoning is given; the
    tool only says where in the picture the region is, so a reader can
    check the name against the picture.
    """
    cx = (box["x0"] + box["x1"]) / 2.0
    cy = (box["y0"] + box["y1"]) / 2.0
    horiz = "left" if cx < FRAME_W / 3 else ("right" if cx > 2 * FRAME_W / 3
                                             else "centre")
    vert = ("top" if cy < FRAME_H / 3 else
            ("bottom" if cy > 2 * FRAME_H / 3 else "middle"))
    return f"{vert}-{horiz}"


def find_regions(presence: np.ndarray, motion: np.ndarray,
                 threshold: float, composite: np.ndarray | None = None
                 ) -> list[dict]:
    """The occupied regions, measured."""
    mask = presence >= threshold
    regions = []
    for comp in label_components(mask):
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        box = {"x0": int(min(xs)), "x1": int(max(xs)),
               "y0": int(min(ys)), "y1": int(max(ys))}
        box["area_px"] = len(comp)
        box["area_pct"] = round(100.0 * len(comp) / (FRAME_H * FRAME_W), 3)
        box["presence"] = round(float(np.mean([presence[p] for p in comp])), 4)
        if motion is not None:
            box["motion"] = round(float(np.mean([motion[p] for p in comp])), 4)
        if composite is not None:
            # what the ROOM looks like there -- the evidence a reader
            # checks a name against
            box["composite_luma"] = round(
                float(np.mean([composite[p] for p in comp])), 1)
        box["centre_x"] = round(float(np.mean(xs)), 1)
        box["centre_y"] = round(float(np.mean(ys)), 1)
        regions.append(box)
    regions.sort(key=lambda b: -b["area_px"])
    for i, b in enumerate(regions):
        b["region_id"] = i
        b["geometric_name"] = region_name(b)
    return regions


def find_quiet_regions(presence: np.ndarray, threshold: float,
                        composite: np.ndarray | None = None,
                        min_pct: float = 0.5) -> list[dict]:
    """The stretches the room's life never reached.

    The brief asked for the empty stretches as well as the busy ones,
    and an empty stretch is a measurable thing: the connected pieces of
    the frame where the presence never rose above the threshold over
    the whole corpus. A room that is never disturbed is not the same
    evidence as a room that is sometimes disturbed, and the difference
    is worth drawing.
    """
    mask = presence < threshold
    out = []
    for comp in label_components(mask):
        if 100.0 * len(comp) / (FRAME_H * FRAME_W) < min_pct:
            continue
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        box = {"x0": int(min(xs)), "x1": int(max(xs)),
               "y0": int(min(ys)), "y1": int(max(ys)),
               "area_px": len(comp),
               "area_pct": round(100.0 * len(comp) / (FRAME_H * FRAME_W), 3)}
        box["presence"] = round(float(np.mean([presence[p] for p in comp])), 5)
        if composite is not None:
            box["composite_luma"] = round(
                float(np.mean([composite[p] for p in comp])), 1)
        box["centre_x"] = round(float(np.mean(xs)), 1)
        box["centre_y"] = round(float(np.mean(ys)), 1)
        out.append(box)
    out.sort(key=lambda b: -b["area_px"])
    for i, b in enumerate(out):
        b["region_id"] = i
        b["geometric_name"] = region_name(b)
    return out


def choose_region_threshold(presence: np.ndarray,
                            quantile: float = 0.75) -> dict:
    """The presence a pixel needs before it counts as a region.

    DERIVED, NOT CHOSEN -- but derived the only way this map allows.

    The presence map is GRADED, not bimodal. A person's edge fades into
    the room: the shoulder is in front of the desk in one frame and
    beside it in the next, and the pixels between the two are occupied
    a third of the time rather than never. The map therefore has no
    valley to put a threshold in -- its histogram is a broad shoulder
    with a long tail, and Otsu's between-class variance splits it at
    0.14, which selects a third of the picture and fuses the person
    with every stray pixel on the walls.

    So the threshold is a QUANTILE of the map, stated in the same words
    the map can be read in: **the occupied zone is the quarter of the
    picture that holds the most occupancy.** It is reproducible, it is
    one number, and the tool takes --threshold so that a reader who
    wants a different quarter can have one.

    The first version of this tool carried a hand-set threshold of 0.05
    against a map whose largest value was 0.09, which is a number
    picked to look tidy rather than to mean anything. The evidence for
    the threshold is in the JSON, next to the number.
    """
    vals = np.asarray(presence, dtype=np.float64).ravel()
    thr = float(np.quantile(vals, quantile))
    note = ""
    if thr <= 0:
        # The quantile landed on the map's zero mass: more than a
        # quarter of the picture was never held at all. The words
        # "the quarter that holds the most occupancy" would then select
        # the whole picture, which is true and useless. The threshold
        # becomes the middle of the pixels that WERE held instead, and
        # the JSON says so.
        nz = vals[vals > 0]
        thr = float(np.median(nz)) if nz.size else 1.0
        note = ("the stated quantile landed on the map's zero mass, so the "
                "threshold was taken as the median of the pixels that were "
                "held at all")
    otsu_thr, ratio = otsu(vals)
    hist, edges = np.histogram(vals, bins=30, range=(0.0, max(1e-9, vals.max())))
    return {
        "threshold": thr,
        "quantile": quantile,
        "rule": ("the occupied zone is the quarter of the picture that holds "
                 "the most occupancy: the presence value at the stated "
                 "quantile of the map"),
        "note": note,
        "otsu_threshold": float(otsu_thr),
        "otsu_ratio": float(ratio),
        "otsu_note": ("Otsu is reported but not used: the map is graded "
                      "rather than bimodal, and its Otsu split fuses the "
                      "person with every stray pixel on the walls"),
        "histogram": {"edges": [round(float(e), 5) for e in edges],
                      "counts": [int(c) for c in hist]},
        "pixels_above": int((vals >= thr).sum()),
        "pct_above": round(100.0 * int((vals >= thr).sum()) / vals.size, 3),
        "max_presence": float(vals.max()),
    }


# ============================================================================
# 6. THE IMAGES
# ============================================================================

def draw(doc: dict, out_dir: Path) -> list[str]:
    """The pictures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    composite = np.array(doc["composite"], dtype=np.float32)
    median = np.array(doc["global_median"], dtype=np.float32)
    motion = np.array(doc["motion_map"], dtype=np.float32)
    presence = np.array(doc["presence_map"], dtype=np.float32)
    written = []

    def save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=110, bbox_inches="tight", facecolor="#101014")
        plt.close(fig)
        written.append(str(p))

    def grid(ax):
        for x in range(0, FRAME_W + 1, 20):
            ax.axvline(x - 0.5, color="#3a3a46", lw=0.6)
        for y in range(0, FRAME_H + 1, 16):
            ax.axhline(y - 0.5, color="#3a3a46", lw=0.6)
        ax.set_xlabel("frame x  (0-159)", color="#b8b8c4")
        ax.set_ylabel("frame y  (0-95)", color="#b8b8c4")
        ax.tick_params(colors="#8a8a98")

    # 1. the composite
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.imshow(composite, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    grid(ax)
    ax.set_title("THE EMPTY ROOM -- the per-pixel median of the frames with\n"
                 "nothing in the seat, found by splitting the corpus on the\n"
                 "block whose luma separates most cleanly",
                 color="#e8e8ee", fontsize=11)
    save(fig, "room-map-composite.png")

    # 2. the corpus's own median, for the comparison
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.imshow(median, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    grid(ax)
    ax.set_title("THE CORPUS'S OWN MEDIAN -- the most common view of the room\n"
                 "which is not quite empty: the person is in it, but he is in\n"
                 "fewer frames than the room is, so he is a ghost in it",
                 color="#e8e8ee", fontsize=11)
    save(fig, "room-map-median.png")

    # 3. the motion map
    fig, ax = plt.subplots(figsize=(9, 5.6))
    hi = max(1e-6, float(np.percentile(motion, 99.5)))
    im = ax.imshow(motion, cmap="inferno", vmin=0, vmax=hi,
                   interpolation="nearest")
    grid(ax)
    ax.set_title("MOTION -- where the room's life happens, measured in NOISE\n"
                 f"SIGMAS per frame (not luma: a shelf is all texture), over "
                 f"{doc['n_sessions']} sessions and {doc['n_frames']} frames",
                 color="#e8e8ee", fontsize=11)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("mean |dframe| in that pixel's own noise sigmas",
                 color="#b8b8c4")
    cb.ax.tick_params(colors="#8a8a98")
    save(fig, "room-map-motion.png")

    # 4. the presence map
    fig, ax = plt.subplots(figsize=(9, 5.6))
    im = ax.imshow(presence, cmap="magma", vmin=0, vmax=1,
                   interpolation="nearest")
    grid(ax)
    ax.set_title("PRESENCE -- how often each pixel holds something other than\n"
                 "the room's own value (fraction of frames, per session averaged)",
                 color="#e8e8ee", fontsize=11)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("fraction of frames occupied", color="#b8b8c4")
    cb.ax.tick_params(colors="#8a8a98")
    save(fig, "room-map-presence.png")

    # 5. the overlay: the occupied zones and the empty stretches
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.imshow(composite, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    grid(ax)
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(doc["regions"]):
        colour = cmap(i % 10)
        ax.add_patch(plt.Rectangle((r["x0"] - 0.5, r["y0"] - 0.5),
                                   r["x1"] - r["x0"] + 1, r["y1"] - r["y0"] + 1,
                                   fill=False, edgecolor=colour, lw=1.8))
        ax.text(r["x0"] + 1, max(9, r["y0"] - 2), f"R{i}",
                color=colour, fontsize=12, fontweight="bold")
    for i, r in enumerate(doc.get("quiet_regions", [])[:6]):
        ax.add_patch(plt.Rectangle((r["x0"] - 0.5, r["y0"] - 0.5),
                                   r["x1"] - r["x0"] + 1, r["y1"] - r["y0"] + 1,
                                   fill=False, edgecolor="#4de08a", lw=1.2,
                                   linestyle="--"))
        ax.text(r["x0"] + 1, min(93, r["y1"] + 6), f"Q{i}",
                color="#4de08a", fontsize=10, fontweight="bold")
    ax.set_title("THE REGIONS -- the empty room with the occupied zones drawn on\n"
                 f"R = held by something, threshold {doc['region_threshold']:.3f}. "
                 f"Q = never held (empty stretches). These are places in the PICTURE.",
                 color="#e8e8ee", fontsize=11)
    save(fig, "room-map-regions.png")
    return written


# ============================================================================
# 7. THE DRIVER
# ============================================================================

def load_corpus(sessions: list[dict], corpus: Path,
                cache: bool = True) -> dict[str, np.ndarray]:
    """Every session's frames, in memory.

    The corpus is 2.7 GB of uint8 and it is read three times: once for
    the histogram and the split, once for the empty room, once for the
    heatmaps. Reading it from the Windows mount three times costs about
    four minutes of the run; holding it costs the same bytes of RAM
    once. The cache is on by default and can be turned off with
    --no-cache on a machine that would rather not hold it.
    """
    out = {}
    for s in sessions:
        arr = session_frames(corpus / s["session"])
        if arr.shape[0] < 2:
            continue
        out[s["session"]] = np.ascontiguousarray(arr) if cache else arr
    return out


def build(corpus: Path, out_dir: Path, threshold: float | None = None,
          cache: bool = True) -> dict:
    """The whole measurement."""
    sessions = usable_sessions(survey(corpus))
    frames_by_session = load_corpus(sessions, corpus, cache)

    # PASS 1: the per-frame block means and a SAMPLE of the frames
    # themselves, for the split. The corpus's own median is taken later,
    # over the mapped viewpoint only.
    block_stats = []
    sample_frames = []
    n_sampled = 0
    SAMPLE_LIMIT = 6000
    for s in sessions:
        arr = frames_by_session.get(s["session"])
        if arr is None:
            continue
        n = arr.shape[0]
        step = max(1, n // SAMPLE_FRAMES)
        idx = np.arange(0, n, step)
        # the block means for this session's sampled frames, in one go
        block_stats.append(block_means(np.asarray(arr[idx], dtype=np.uint8)))
        # a bounded sample of the frames themselves, for the agreement test
        take = idx[:max(0, SAMPLE_LIMIT - n_sampled)]
        if len(take):
            sample_frames.append(np.asarray(arr[take], dtype=np.uint8))
            n_sampled += len(take)
    block_stats = np.concatenate(block_stats, axis=0)
    sample_frames = np.concatenate(sample_frames, axis=0)
    # the sample must line up with the START of block_stats
    block_stats = block_stats[:sample_frames.shape[0]]

    # the split
    split = find_empty_population(block_stats, sample_frames)
    empty_mask = split.pop("empty_mask")
    split["n_frames"] = int(empty_mask.sum())
    split["n_frames_total"] = int(empty_mask.size)
    split["block_bounds"] = block_bounds(split["block"])

    # the same split, over every frame
    bx, by = split["block"]
    all_block_session = []
    for s in sessions:
        arr = frames_by_session.get(s["session"])
        if arr is None:
            continue
        ys = np.linspace(0, FRAME_H, GRID_Y + 1).astype(int)
        xs = np.linspace(0, FRAME_W, GRID_X + 1).astype(int)
        y0, y1 = ys[by], ys[by + 1]
        x0, x1 = xs[bx], xs[bx + 1]
        vals = np.asarray(arr, dtype=np.float32)[:, y0:y1, x0:x1] \
            .reshape(arr.shape[0], -1).mean(axis=1)
        all_block_session.append((s["session"], vals))
    empty_by_session = {}
    for name, vals in all_block_session:
        if split["empty_population"] == "low":
            empty_by_session[name] = vals <= split["threshold"]
        else:
            empty_by_session[name] = vals > split["threshold"]
    n_empty_all = int(sum(int(v.sum()) for v in empty_by_session.values()))
    split["n_frames_all"] = n_empty_all
    split["n_frames_total_all"] = int(sum(len(v) for _, v in all_block_session))

    # PASS 2: a PROVISIONAL empty room, over every session -- the empty
    # frames and the noise of the empty room, measured in the units the
    # tests will use. This composite is only used to ask the next
    # question: where was each session shot from?
    def collect_empty(sel_list, want_pairs):
        hist = np.zeros((FRAME_H * FRAME_W, 256), dtype=np.uint64)
        frames_out = []
        pairs_out = []
        n_empty = 0
        for s, sel in sel_list:
            arr = frames_by_session.get(s["session"])
            if arr is None or sel is None:
                continue
            step = max(1, arr.shape[0] // SAMPLE_FRAMES)
            idx = np.arange(0, arr.shape[0], step)
            take = sel[idx]
            if take.any():
                picked = np.asarray(arr[idx][take], dtype=np.uint8)
                accumulate_histogram(hist, picked)
                frames_out.append(picked)
                n_empty += picked.shape[0]
            if not want_pairs:
                continue
            # the ADJACENT empty pairs, for the difference noise: two
            # frames of the same session, both empty, one frame apart
            both = sel[:-1] & sel[1:]
            pi = np.where(both)[0]
            if len(pi):
                if len(pi) > MAX_PAIRS:
                    pi = pi[np.linspace(0, len(pi) - 1, MAX_PAIRS).astype(int)]
                a = np.asarray(arr[pi], dtype=np.float32).reshape(len(pi), -1)
                b = np.asarray(arr[pi + 1], dtype=np.float32).reshape(len(pi), -1)
                # only the pairs where the picture actually changed: a
                # pair of identical frames says nothing about the noise
                # of an update, and there are a great many of them
                changed = np.abs(b - a).sum(axis=1) > 0
                if not changed.any():
                    continue
                a, b = a[changed], b[changed]
                pairs_out.append(residual_batch(a, b))
        return (median_from_histogram(hist).reshape(FRAME_H, FRAME_W),
                frames_out, pairs_out, n_empty)

    MAX_PAIRS = 3000
    sel_list = [(s, empty_by_session.get(s["session"])) for s in sessions]
    provisional, _, _, _ = collect_empty(sel_list, False)

    # WHERE WAS EACH SESSION SHOT FROM? The laptop was nudged during the
    # night, and a map built across the nudge is a map of two slightly
    # different pictures averaged into one.
    viewpoints = find_viewpoints(sessions, corpus, provisional, frames_by_session)
    vp_scores = sorted(v["score"] for v in viewpoints)
    vp_gap = 0.0
    vp_cut = len(vp_scores)
    if len(vp_scores) > 1:
        gaps = [(vp_scores[i + 1] - vp_scores[i], i) for i in range(len(vp_scores) - 1)]
        vp_gap, i = max(gaps)
        vp_cut = i + 1
    vp_floor = vp_scores[vp_cut - 1] if vp_cut else 0.0
    dominant = {v["session"] for v in viewpoints if v["score"] <= vp_floor}
    others = [v for v in viewpoints if v["session"] not in dominant]

    # the map is built over the DOMINANT VIEWPOINT only
    mapped_sessions = [s for s in sessions if s["session"] in dominant]
    sel_list = [(s, empty_by_session.get(s["session"])) for s in mapped_sessions]

    # the corpus's own median, over the mapped viewpoint
    hist = np.zeros((FRAME_H * FRAME_W, 256), dtype=np.uint64)
    for s in mapped_sessions:
        arr = frames_by_session.get(s["session"])
        if arr is None:
            continue
        for i in range(0, arr.shape[0], 400):
            accumulate_histogram(hist, np.asarray(arr[i:i + 400], dtype=np.uint8))
    global_median = median_from_histogram(hist).reshape(FRAME_H, FRAME_W)

    # the empty room, and the noise of the empty room
    composite, empty_frames, empty_pairs, n_empty = collect_empty(sel_list, True)
    empty_arr = (np.concatenate(empty_frames, axis=0) if empty_frames
                 else np.zeros((0, FRAME_H, FRAME_W), dtype=np.uint8))
    noise = measure_noise(empty_arr, composite, empty_pairs)

    # PASS 3: the heatmaps, and then the regions.
    heat = heatmaps(mapped_sessions, corpus, composite, noise, frames_by_session)
    presence = heat["presence"]
    derived = choose_region_threshold(presence)
    used_threshold = derived["threshold"] if threshold is None else threshold
    regions = find_regions(presence, heat["motion"], used_threshold,
                           composite=composite)
    quiet = find_quiet_regions(presence, QUIET_FLOOR, composite=composite)

    doc = {
        "corpus": str(corpus),
        "frame_size": [FRAME_W, FRAME_H],
        "n_sessions_used": len(sessions),
        "n_sessions_mapped": len(mapped_sessions),
        "n_frames_sampled": int(block_stats.shape[0]),
        "n_frames_used": heat["n_frames"],
        "sessions_used": [s["session"] for s in mapped_sessions],
        "viewpoints": {
            "rule": ("each session is compared against the composite at "
                     "every shift in a small window, on the top band of the "
                     "frame (ceiling, never a person); the sessions are split "
                     "at the largest GAP in the sorted scores"),
            "score_gap": round(float(vp_gap), 4),
            "score_floor": round(float(vp_floor), 4),
            "n_mapped": len(dominant),
            "n_other": len(others),
            "mapped_sessions": sorted(dominant),
            "other_viewpoints": others,
            "per_session": viewpoints,
        },
        "split": split,
        "presence_floor": PRESENCE_FLOOR,
        "presence_noise_sigmas": NOISE_SIGMAS,
        "motion_floor": MOTION_FLOOR,
        "n_empty_frames_measured": noise["n_empty"],
        "n_empty_pairs_measured": noise["n_pairs"],
        "presence_sigma": [[round(float(v), 4) for v in row]
                           for row in noise["presence_sigma"]],
        "motion_sigma": [[round(float(v), 4) for v in row]
                         for row in noise["motion_sigma"]],
        "region_threshold": used_threshold,
        "region_threshold_rule": derived,
        "quiet_floor": QUIET_FLOOR,
        "n_sessions": heat["n_sessions"],
        "n_frames": heat["n_frames"],
        "n_changes": heat["n_changes"],
        "composite": composite.astype(int).tolist(),
        "global_median": global_median.astype(int).tolist(),
        "motion_map": [[round(float(v), 5) for v in row]
                       for row in heat["motion"]],
        "motion_luma_map": [[round(float(v), 5) for v in row]
                            for row in heat["motion_luma"]],
        "presence_map": [[round(float(v), 5) for v in row]
                         for row in presence],
        "regions": regions,
        "quiet_regions": quiet,
        "honesty": ("this is the room as the camera knows it: an image plane, "
                    "not a floor plan. Every coordinate is a place in the "
                    "picture. Nothing here measures a distance, and nothing "
                    "here can tell a door from a picture of a door."),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "room_map.json").write_text(json.dumps(doc, indent=2))
    return doc


def block_bounds(block: list[int]) -> dict:
    """The frame coordinates of a grid block."""
    bx, by = block
    ys = np.linspace(0, FRAME_H, GRID_Y + 1).astype(int)
    xs = np.linspace(0, FRAME_W, GRID_X + 1).astype(int)
    return {"x0": int(xs[bx]), "x1": int(xs[bx + 1]) - 1,
            "y0": int(ys[by]), "y1": int(ys[by + 1]) - 1}



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS),
                    help="the free-sessions folder")
    ap.add_argument("--out", default=str(REPO / "docs" / "research"),
                    help="where the images and the json go")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the region threshold; by default it is "
                         "derived from the presence map's own distribution")
    ap.add_argument("--no-cache", action="store_true",
                    help="do not hold the corpus in memory (it is 2.7 GB)")
    ap.add_argument("--redraw", default=None,
                    help="re-draw the images from a room_map.json, without "
                         "reading the corpus again")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    out_dir = Path(a.out)
    if a.redraw:
        doc = json.loads(Path(a.redraw).read_text())
        for p in draw(doc, out_dir):
            print(f"wrote      {p}")
        return 0

    corpus = Path(a.corpus)
    doc = build(corpus, out_dir, a.threshold, cache=not a.no_cache)
    sp = doc["split"]
    rule = doc["region_threshold_rule"]
    print(f"corpus     {corpus}")
    print(f"sessions   {doc['n_sessions_used']} used, "
          f"{doc['n_frames_used']} frames")
    print(f"split      block {sp['block']} (x {sp['block_bounds']['x0']}-"
          f"{sp['block_bounds']['x1']}, y {sp['block_bounds']['y0']}-"
          f"{sp['block_bounds']['y1']})  ratio {sp['ratio']}")
    print(f"           threshold {sp['threshold']}  empty = "
          f"{sp['empty_population']}  {sp['n_frames']} of "
          f"{sp['n_frames_total']} frames")
    print(f"noise      {doc['n_empty_frames_measured']} empty frames, "
          f"{doc['n_empty_pairs_measured']} empty pairs")
    print(f"changes    {doc['n_changes']} of {doc['n_frames']} frames were "
          f"a new picture")
    print(f"threshold  {doc['region_threshold']:.4f}  "
          f"(otsu {rule['otsu_threshold']:.4f}, ratio {rule['otsu_ratio']:.3f})  "
          f"{rule['pct_above']:.1f}% of the frame")
    print(f"regions    {len(doc['regions'])}")
    for r in doc["regions"]:
        print(f"  R{r['region_id']} {r['geometric_name']:>14} "
              f"x {r['x0']:>3}-{r['x1']:<3} y {r['y0']:>2}-{r['y1']:<2} "
              f"area {r['area_pct']:>5.2f}%  presence {r['presence']:.3f}  "
              f"motion {r['motion']:.2f}")
    print(f"quiet      {len(doc['quiet_regions'])} stretches")
    for r in doc["quiet_regions"][:6]:
        print(f"  Q{r['region_id']} {r['geometric_name']:>14} "
              f"x {r['x0']:>3}-{r['x1']:<3} y {r['y0']:>2}-{r['y1']:<2} "
              f"area {r['area_pct']:>5.2f}%  presence {r['presence']:.4f}")
    if not a.no_images:
        for p in draw(doc, out_dir):
            print(f"wrote      {p}")
    print(f"json       {out_dir / 'room_map.json'}")
    return 0


# ============================================================================
# 8. THE SELFTEST
# ============================================================================

def selftest() -> int:
    """The checks that need no corpus."""
    import tempfile

    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  [{'ok' if got == want else 'FAIL'}] {name}: {got!r}")

    print("SELFTEST -- the histogram, the median, the split, the regions")
    print("-" * 60)

    # 1. THE HISTOGRAM. The median must be exact, not sampled.
    hist = np.zeros((4, 256), dtype=np.uint64)
    frames = np.array([[[0, 0], [10, 10]], [[0, 0], [20, 20]],
                       [[0, 0], [30, 30]], [[0, 0], [40, 40]]],
                      dtype=np.uint8)
    accumulate_histogram(hist, frames.reshape(4, 4))
    med = median_from_histogram(hist).reshape(2, 2)
    check("a constant pixel's median is itself", int(med[0, 0]), 0)
    check("the middle of four is the lower one", int(med[1, 0]), 20)
    check("the histogram counted every sample", int(hist.sum()), 16)

    # 2. THE MEDIAN REMOVES A MOVING THING.
    bg = np.full((FRAME_H, FRAME_W), 100, dtype=np.uint8)
    arr = np.stack([bg, bg, bg])
    arr[1, 40:60, 70:90] = 200          # something happened in one frame
    hist = np.zeros((FRAME_H * FRAME_W, 256), dtype=np.uint64)
    accumulate_histogram(hist, arr)
    med = median_from_histogram(hist).reshape(FRAME_H, FRAME_W)
    check("the room survives the median", int(med[50, 80]), 100)
    check("and so does everything around it", int(med[10, 10]), 100)

    # 3. OTSU FINDS THE SPLIT.
    vals = np.concatenate([np.random.RandomState(0).normal(20, 3, 400),
                            np.random.RandomState(1).normal(80, 3, 600)])
    thr, ratio = otsu(vals)
    check("the threshold lands between the two populations",
          20 < thr < 80, True)
    check("and the split explains most of the variance", ratio > 0.8, True)

    # 4. THE EMPTY POPULATION IS THE ONE WHOSE FRAMES AGREE.
    n = 300
    bs = np.zeros((n, GRID_Y, GRID_X), dtype=np.float32)
    rs = np.random.RandomState(7)
    bs[:n // 2, 2, 3] = 20 + rs.normal(0, 2, n // 2)   # tight: the room
    bs[n // 2:, 2, 3] = 80 + rs.normal(0, 12, n - n // 2)  # spread: a person
    # the frames: the first half is the SAME room every time, the second
    # half is the room with a person somewhere different in it
    fr = np.full((n, FRAME_H, FRAME_W), 90, dtype=np.uint8)
    for i in range(n // 2):
        fr[i] = 90
    for i in range(n // 2, n):
        f = np.full((FRAME_H, FRAME_W), 90, dtype=np.uint8)
        f[0:40, 0:40] = 20 + (i % 30)
        fr[i] = f
    split = find_empty_population(bs, fr)
    check("the best block is the one that splits", split["block"], [3, 2])
    check("the population whose frames agree is the empty one",
          split["empty_population"], "low")
    check("the empty half was found",
          int(split["empty_mask"].sum()), n // 2)
    check("and the agreement is reported",
          split["frame_agreement"]["low"] < split["frame_agreement"]["high"],
          True)

    # 5. THE WHOLE THING ON A SYNTHETIC CORPUS.
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td) / "corpus"
        sess = corpus / "1"
        sess.mkdir(parents=True)
        base = np.full((FRAME_H, FRAME_W), 80, dtype=np.uint8)
        base[60:90, 40:80] = 150          # a desk
        frames = []
        for i in range(60):
            f = base.copy()
            if i % 2 == 0:
                # a person, half the time, and MOVING -- the criterion is
                # that the occupied frames do not agree with each other
                f[10 + (i % 5):50 + (i % 5), 20:60] = 20 + (i % 7)
            frames.append(f)
        arr = np.stack(frames)
        (sess / "luma.raw").write_bytes(arr.tobytes())
        (sess / "meta.json").write_text(json.dumps(
            {"trigger": "motion", "mode": "auto", "n_frames": len(arr)}))
        doc = build(corpus, corpus, None)
        comp = np.array(doc["composite"], dtype=int)
        check("the session was counted", doc["n_sessions_used"], 1)
        check("the desk is in the composite", int(comp[75, 60]), 150)
        check("the wall is in the composite", int(comp[5, 5]), 80)
        check("a person who comes and goes does not survive it",
              int(comp[30, 40]), 80)
        check("a region was found", len(doc["regions"]) > 0, True)
        if doc["regions"]:
            r = doc["regions"][0]
            check("the region is where the person was",
                  r["x0"] <= 22 and r["x1"] >= 57, True)

    # 6. THE REGION LABELLING IS GEOMETRIC, NOT SEMANTIC.
    check("a region at the top left is named so",
          region_name({"x0": 0, "x1": 20, "y0": 0, "y1": 10}), "top-left")
    check("a region at the bottom right is named so",
          region_name({"x0": 140, "x1": 159, "y0": 80, "y1": 95}),
          "bottom-right")

    # 7. THE COMPONENTS.
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:4, 1:4] = True
    mask[7:9, 7:9] = True
    comps = label_components(mask)
    check("two separate blobs", len(comps), 2)
    check("the bigger one is first", len(comps[0]), 9)

    # 8. THE EXPOSURE IS FITTED OUT, AND IT IS FITTED IN THE RIGHT UNITS.
    #     The room has to have real structure, because the fit needs
    #     something to fit on: a gradient across the frame and a desk
    #     in it, so that the background alone can determine the gain.
    ramp = np.tile(np.linspace(20, 120, FRAME_W, dtype=np.float32), (FRAME_H, 1))
    room = ramp.astype(np.uint8)
    room[60:80, 60:80] = 40
    brighter = np.clip(room.astype(np.float32) * 1.5 + 20, 0, 255).astype(np.uint8)
    r = residual_batch(room.ravel(), brighter.ravel().reshape(1, -1))
    check("a brighter copy of the room leaves almost nothing behind",
          float(np.abs(r).max()) < 2.0, True)
    darker = np.clip(room.astype(np.float32) * 0.6 + 5, 0, 255).astype(np.uint8)
    r = residual_batch(room.ravel(), darker.ravel().reshape(1, -1))
    check("and so does a darker one", float(np.abs(r).max()) < 2.0, True)
    # a real change must SURVIVE the fit
    moved = brighter.copy()
    moved[60:80, 60:80] = 255
    r = residual_batch(room.ravel(), moved.ravel().reshape(1, -1))
    r = r.reshape(FRAME_H, FRAME_W)
    check("but a thing that moved is still there",
          float(np.abs(r).max()) > 100, True)
    check("and it is only there", float(np.abs(r[0:60, 0:60]).max()) < 3.0, True)

    # 9. THE NOISE SCALE IS IN THE UNITS OF THE TEST.
    #     The first version of this tool measured the noise on RAW luma
    #     and applied it to a corrected residual, and the threshold came
    #     out above the top of the 8-bit range. This checks that a room
    #     with two levels of exposure has a SMALL measured noise.
    frames = []
    for i in range(40):
        f = room.astype(np.float32) * (0.8 + 0.4 * (i % 2)) + (i % 3)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    empty = np.stack(frames)
    noise = measure_noise(empty, room)
    check("the noise of a still room is small",
          float(noise["presence_sigma"].max()) < 12.0, True)
    thr = np.maximum(PRESENCE_FLOOR, NOISE_SIGMAS * noise["presence_sigma"])
    check("and the threshold it produces is inside the 8-bit range",
          float(thr.max()) < 255, True)

    # 10. THE REGION THRESHOLD IS DERIVED, NOT HAND-SET.
    #     The map is graded, the way a real presence map is: a busy
    #     zone, a halo around it, and a graded edge.
    pres = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
    pres[30:95, 20:140] = 0.3      # its halo
    pres[40:90, 50:110] = 0.8      # a busy zone inside it
    pres[10:20, 10:20] = 0.1      # a smaller one
    rule = choose_region_threshold(pres)
    check("the threshold is derived from the map",
          rule["quantile"] == 0.75, True)
    check("and it is the stated quantile of the map",
          abs(rule["threshold"] - float(np.quantile(pres, 0.75))) < 1e-9, True)
    check("and it is inside the map's own range",
          0.0 < rule["threshold"] < float(pres.max()), True)
    regions = find_regions(pres, np.zeros_like(pres), rule["threshold"])
    check("the busy zone is found", len(regions) >= 1, True)
    check("and it is where it was put", regions[0]["x0"] <= 20, True)
    quiet = find_quiet_regions(pres, QUIET_FLOOR)
    check("the empty stretches are found too", len(quiet) >= 1, True)

    # 10b. ... AND WHEN THE MAP IS MOSTLY EMPTY, THE RULE SAYS SO.
    sparse = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
    sparse[70:95, 70:110] = 0.8
    rule2 = choose_region_threshold(sparse)
    check("a map that is mostly empty falls back, and says so",
          bool(rule2["note"]) and rule2["threshold"] > 0, True)
    check("and the fallback is the middle of the pixels that were held",
          abs(rule2["threshold"] - 0.8) < 1e-6, True)

    # 11. A FRAME THAT IS NOT THE ROOM IS NOT CALLED THE ROOM.
    room2 = np.full((FRAME_H, FRAME_W), 100, dtype=np.uint8)
    with_person = room2.copy()
    with_person[70:95, 70:110] = 20
    r = residual_batch(room2.ravel(),
                       with_person.ravel().reshape(1, -1)).reshape(FRAME_H, FRAME_W)
    check("the person is found",
          float(np.abs(r)[70:95, 70:110].mean()) > 50, True)
    check("and the rest of the room is not",
          float(np.abs(r)[0:60, 0:60].mean()) < 2.0, True)

    print("-" * 60)
    if failures:
        print(f"SELFTEST FAILED: {len(failures)} check(s)")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
