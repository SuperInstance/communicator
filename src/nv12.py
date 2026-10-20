"""
nv12.py -- read the camera captures, with the frame layout DETECTED, not assumed.

WHY THIS FILE EXISTS.

Every experiment in this repo that touches video used to assume the frame
stride was `W*H*3//2` -- 23,040 bytes for a 160x96 frame, which is standard
NV12 (a luma plane followed by INTERLEAVED chroma). That assumption is wrong
for the captures that are actually on disk, and it was wrong in a way that
destroyed the signal rather than merely shifting it.

The old captures were written by `C:\\proart_capture\\nv12small.py`, which
writes, per frame:

    Y plane    15,360 bytes   (160 * 96)
    U,V plane   3,840 bytes   (80 * 24 samples, INTERLEAVED)
    ------------------------
    total      19,200 bytes   = W*H*5/4

So the real stride is 19,200, not 23,040 -- 20% longer -- and the chroma is
subsampled 2:1 across but 4:1 DOWN, at half the vertical resolution of a
standard NV12 frame's chroma. Reading a 19,200-byte file at 23,040 produces
750 frames of which only every sixth is intact; the rest are torn, with rows
0..71 from frame k and rows 72..95 from frame k+1. The measured consequence
on A.nv12:

    mean consecutive |diff|   wrong stride 23,040:  43.560
                             right stride 19,200:   0.545

which is an 80-fold reduction. The "motion" the repo measured, the "row
banding" it wrote a removal pass for, and the pose features it trained on
were ~99% stride artifact. See `docs/research/NV12-STRIDE.md`.

A NOTE ON THE CHROMA ORDER, because it was wrong here first.

This module originally called the layout `y_u_v_planar` and said the chroma
was "all the U bytes, then all the V bytes". That is FALSE, and an
independent lane measured it false: in A.nv12's chroma region the even-indexed
bytes average 125.05 and the odd-indexed bytes 135.88, with no step at the
midpoint (1.16 counts). The chroma is INTERLEAVED -- U at even offsets, V at
odd -- exactly as `nv12small.py` writes it (`uv[:, 0] = u; uv[:, 1] = v` on
an (N, 2) array is row-major, which interleaves). The stride and the luma
are unaffected either way, so no measurement changes; but a layout identifier
that asserts a falsehood is a landmine, so the layout is now named for what
it actually is, and `y_u_v_planar` survives only as a deprecated alias.

THE THIRD FORMAT.

> CORRECTION (2026-10-20). This paragraph used to say that
> `experiments/corpus_capture_win.py` writes the LUMA PLANE ONLY -- 15,360
> bytes per frame, no chroma at all. That is FALSE, and it was worth
> checking rather than assuming: **no script in this repo has ever written
> luma-only frames.** `corpus_capture_win.py` writes numpy change-map
> archives (`.npz`), and the two tools that DO write frames --
> `C:\proart_capture\nv12small.py` and `syzygy_capture_win.py` -- both write
> 19,200-byte frames with INTERLEAVED chroma, and both build the chroma the
> same way (`uv[:, 0] = u; uv[:, 1] = v` on an (N, 2) array is row-major,
> which interleaves). So the reader's claim about the chroma being
> interleaved holds for every writer in play.
>
> The `luma_only` layout is KEPT as something the detector can recognise --
> a luma-only stream is a real thing -- but nothing here produces one, and
> a reader that assumes one will appear is a landmine. It was the
> ATTRIBUTION that was wrong, not the layout.

So there are three layouts the reader knows, and they must not be confused:

    standard_nv12     W*H*3//2   luma + interleaved UV      (the ASSUMPTION)
    y_uv_halfres      W*H*5//4   luma + interleaved UV      (the captures)
    luma_only         W*H        luma only                   (nothing writes it)

Note the chroma: `y_uv_halfres` is INTERLEAVED and HALF-RESOLUTION, not
planar. The name is the correction of a name that used to be `y_u_v_planar`
and assert "U plane + V plane"; that assertion was FALSE (see the note on
the chroma order above) and the old name survives only as a deprecated
alias so that older results files still parse.

HOW THE LAYOUT IS DETECTED.

By the only thing that distinguishes them: a real video's consecutive frames
are similar, and a mis-strided read mixes unrelated bytes together. So the
detector scores each candidate layout by the mean consecutive |diff| of the
luma plane it produces, and takes the smallest. The margin over the
runner-up is reported, because a detector that cannot say how sure it is has
no business being trusted. A file whose layout is ambiguous (a static scene,
say, where every layout scores ~0) is an ERROR, not a guess.

Reading the layout from the FILE SIZE alone is impossible: 17,280,000 bytes
divides by all three strides. The content is what decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DEFAULT_WIDTH = 160
DEFAULT_HEIGHT = 96

#: The layouts this reader knows about, by name.
LAYOUT_NAMES = ("standard_nv12", "y_uv_halfres", "luma_only")

#: Deprecated names, kept so that a script written against the old (wrong)
#: description still reads the file correctly. `y_u_v_planar` claimed the
#: chroma was two separate planes; it is interleaved. The name is kept only
#: so that nothing breaks -- it is wrong, and nothing should use it.
LAYOUT_ALIASES = {"y_u_v_planar": "y_uv_halfres"}

#: The chroma geometry of each layout, as (samples across, samples down).
#: `y_uv_halfres` is 80x24 -- half the VERTICAL resolution of a standard
#: NV12 frame's chroma (80x48) -- which is what makes its frame 20%
#: shorter.
CHROMA_GEOMETRY = {
    "standard_nv12": (None, None),      # 80 x 48 for 160x96
    "y_uv_halfres": (None, None),       # 80 x 24 for 160x96
    "luma_only": None,
}


@dataclass(frozen=True)
class Layout:
    """A frame layout, and the evidence for believing in it."""

    name: str
    stride: int
    n_frames: int
    luma_bytes: int
    mean_frame_diff: float
    runner_up_name: str
    runner_up_diff: float
    chroma_order: str = "none"
    chroma_samples: str = ""

    @property
    def margin(self) -> float:
        """How much better the winner is than the runner-up, as a ratio.

        A ratio of 1.0 means "no better than the alternative" -- the layout is
        ambiguous and the caller must not act on it. The degenerate 0-over-0
        case (a file with no content at all) counts as ambiguous, not as a
        perfect score.
        """
        if self.mean_frame_diff <= 0 and self.runner_up_diff <= 0:
            return 1.0
        if self.runner_up_diff <= 0:
            return float("inf")
        return self.runner_up_diff / max(self.mean_frame_diff, 1e-12)

    def as_dict(self) -> dict:
        margin = self.margin
        return {
            "layout": self.name,
            "stride_bytes": self.stride,
            "n_frames": self.n_frames,
            "luma_bytes": self.luma_bytes,
            "mean_consecutive_frame_diff": round(self.mean_frame_diff, 6),
            "runner_up_layout": self.runner_up_name,
            "runner_up_mean_consecutive_frame_diff": (
                None if self.runner_up_diff == float("inf")
                else round(self.runner_up_diff, 6)),
            "margin_over_runner_up": (None if margin == float("inf")
                                      else round(margin, 3)),
            "chroma": "none (luma only)" if self.chroma_order == "none"
                      else f"{self.chroma_order}, {self.chroma_samples}",
        }


def canonical_layout(layout: str) -> str:
    """Resolve a deprecated layout name to its current one."""
    return LAYOUT_ALIASES.get(layout, layout)


def stride_for(layout: str, width: int = DEFAULT_WIDTH,
               height: int = DEFAULT_HEIGHT) -> int:
    """The byte stride of one frame in the named layout."""
    layout = canonical_layout(layout)
    luma = width * height
    if layout == "standard_nv12":
        return luma * 3 // 2
    if layout == "y_uv_halfres":
        return luma + luma // 4
    if layout == "luma_only":
        return luma
    raise ValueError(f"unknown layout {layout!r}; known: {LAYOUT_NAMES}")


def chroma_geometry(layout: str, width: int = DEFAULT_WIDTH,
                    height: int = DEFAULT_HEIGHT):
    """The chroma plane's shape in samples, or None if there is no chroma.

    `y_uv_halfres` is 80 x 24 -- the same width as a standard NV12 frame's
    chroma but HALF the height, which is why its frame is 20% shorter.
    """
    layout = canonical_layout(layout)
    if layout == "standard_nv12":
        return (width // 2, height // 2)
    if layout == "y_uv_halfres":
        return (width // 2, height // 4)
    return None


def luma_view(data: np.ndarray, layout: str, width: int = DEFAULT_WIDTH,
              height: int = DEFAULT_HEIGHT) -> np.ndarray:
    """The luma plane of every frame, as (n_frames, height, width) uint8.

    `data` is the whole file as a flat uint8 array.
    """
    stride = stride_for(layout, width, height)
    luma = width * height
    n = len(data) // stride
    if n == 0:
        return np.zeros((0, height, width), dtype=np.uint8)
    frames = data[: n * stride].reshape(n, stride)
    return frames[:, :luma].reshape(n, height, width)


def mean_consecutive_diff(luma: np.ndarray) -> float:
    """Mean absolute difference between consecutive frames.

    The detection statistic. A real video scores low; a mis-strided read
    scores high because it mixes two different frames into one.
    """
    if len(luma) < 2:
        return float("nan")
    d = np.abs(np.diff(luma.astype(np.int32), axis=0))
    return float(d.mean())


def candidate_layouts(width: int = DEFAULT_WIDTH,
                       height: int = DEFAULT_HEIGHT) -> list[str]:
    """The layouts worth testing for this frame size."""
    return list(LAYOUT_NAMES)


def _chroma_fields(layout: str, width: int = DEFAULT_WIDTH,
                   height: int = DEFAULT_HEIGHT) -> tuple[str, str]:
    """(order, samples) for a layout's chroma."""
    geo = chroma_geometry(layout, width, height)
    if geo is None:
        return "none", ""
    return "interleaved U,V", f"{geo[0]}x{geo[1]} samples"


def detect_layout(path_or_bytes, width: int = DEFAULT_WIDTH,
                  height: int = DEFAULT_HEIGHT,
                  layouts: Optional[Sequence[str]] = None) -> Layout:
    """Work out how the frames in this file are laid out.

    Returns the winning Layout, with the margin over the runner-up.

    Raises `ValueError` if the file does not divide evenly into the candidate
    strides, or if the best margin is below `min_margin` -- an ambiguous file
    is an error, not a guess.
    """
    data = _as_array(path_or_bytes)
    names = list(layouts) if layouts else candidate_layouts(width, height)
    scored: list[tuple[str, int, int, float]] = []
    for name in names:
        stride = stride_for(name, width, height)
        if len(data) % stride:
            continue
        n = len(data) // stride
        if n < 2:
            continue
        luma = luma_view(data, name, width, height)
        scored.append((name, stride, n, mean_consecutive_diff(luma)))
    if not scored:
        raise ValueError(
            f"no candidate layout divides {len(data)} bytes evenly "
            f"(tried {names} for {width}x{height})")
    scored.sort(key=lambda t: t[3])
    name, stride, n, diff = scored[0]
    if len(scored) == 1:
        # Only one candidate stride divides the file, so the file size
        # ALONE determines the layout and the content confirms it. That is
        # certainty, not ambiguity. (This used to be NaN, which made
        # `read_luma` refuse a file that could not possibly be ambiguous.)
        runner_name, runner_diff = "(none: only one candidate divides)", float("inf")
    else:
        runner_name, _, _, runner_diff = scored[1]
    order, samples = _chroma_fields(name, width, height)
    return Layout(name=name, stride=stride, n_frames=n, luma_bytes=width * height,
                  mean_frame_diff=diff, runner_up_name=runner_name,
                  runner_up_diff=runner_diff, chroma_order=order,
                  chroma_samples=samples)


def read_luma(path_or_bytes, width: int = DEFAULT_WIDTH,
              height: int = DEFAULT_HEIGHT, layout: Optional[str] = None,
              min_margin: float = 5.0) -> tuple[np.ndarray, Layout]:
    """Read the luma plane of every frame, and report the layout that was used.

    `layout` may be given to force a layout; otherwise it is detected and the
    margin must be at least `min_margin` or the file is rejected as
    ambiguous.
    """
    data = _as_array(path_or_bytes)
    if layout is None:
        found = detect_layout(data, width, height)
        if not (found.margin >= min_margin):
            raise ValueError(
                f"ambiguous frame layout: {found.name} scores "
                f"{found.mean_frame_diff:.4f} vs {found.runner_up_name} "
                f"{found.runner_up_diff:.4f} (margin {found.margin:.2f} < "
                f"{min_margin}). Pass layout= explicitly.")
        layout = found.name
    stride = stride_for(layout, width, height)
    n = len(data) // stride
    luma = luma_view(data, layout, width, height)
    diff = mean_consecutive_diff(luma)
    runner_name, runner_diff = "(none: only one candidate divides)", float("inf")
    for other in candidate_layouts(width, height):
        if other == layout:
            continue
        st = stride_for(other, width, height)
        if len(data) % st or len(data) // st < 2:
            continue
        d = mean_consecutive_diff(luma_view(data, other, width, height))
        if runner_diff == float("inf") or d < runner_diff:
            runner_name, runner_diff = other, d
    order, samples = _chroma_fields(layout, width, height)
    found = Layout(name=layout, stride=stride, n_frames=n,
                   luma_bytes=width * height, mean_frame_diff=diff,
                   runner_up_name=runner_name, runner_up_diff=runner_diff,
                   chroma_order=order, chroma_samples=samples)
    return luma, found


def _as_array(path_or_bytes) -> np.ndarray:
    """The whole file as a flat uint8 array."""
    if isinstance(path_or_bytes, np.ndarray):
        return path_or_bytes.astype(np.uint8, copy=False).ravel()
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(path_or_bytes), dtype=np.uint8)
    return np.frombuffer(Path(path_or_bytes).read_bytes(), dtype=np.uint8)
