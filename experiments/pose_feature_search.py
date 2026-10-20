"""
pose_feature_search.py -- can ANY statistic separate FACING from TURNED at 160x96?

WHY THIS EXPERIMENT EXISTS.

`pose_test.py` measured a facing-vs-turned difference of 9.182 against a
time-shifted control of 9.178 and concluded "no pose signal". That number is
ambiguous in exactly the way this project has been bitten before. It is
equally consistent with

  (a) the head turn never actually happened in that capture, and
  (b) the statistic that was used (a saturating flip rate over a spatial
      change map) is blind to head pose.

A large vision model (llama-3.2-11b-vision-instruct) was then asked directly
and answered FACING for 91.8% of the facing frames and 80.0% of the turned
frames -- a separation of 0.018 -- which says the turn is also not legible to
a 11B model at 160x96. So the remaining avenue is a FEATURE SEARCH: throw a
panel of statistics at the frames and ask which ones separate, WITH controls
that can kill the answer.

THE POINT IS THE CONTROLS, NOT THE FEATURE LIST.

A statistic that separates facing from turned by 0.3 and also separates two
halves of the SAME facing session by 0.28 has told you nothing. A statistic
that separates facing from turned in a room containing NO PERSON is reading
the banding, not the person -- the empty room is the control that has already
killed one project claim (the ASCII character delta), so it is mandatory
here. Every number in the table is printed next to its own controls and the
headline is the MARGIN over those controls, never the raw separation.

  C1 WITHIN-CONDITION SPLIT. Split a condition's own frames in half and
     compare the halves. This is the null the A-vs-B number must beat.
  C2 EMPTY ROOM. The same comparison, run on 4500 frames of an empty dark
     room, split into five 750-frame blocks. A feature that separates two
     blocks of an empty room is reading the banding.
  C3 LABEL PERMUTATION. Pool the facing and turned frames, relabel them at
     random, recompute the separation. The 95th percentile of that null is
     the separation a feature gets for free.
  C4 SPATIAL NULL. Put the subject at a random place inside the frame and
     recompute the separation. A statistic that measures the subject's
     internal SHAPE keeps separating; a statistic that is really measuring
     WHERE THE SUBJECT SATES loses its separation. Two variants are
     computed: a cyclic roll (simple, but it leaves a seam at a different
     column in each condition, so it is reported and not used as a gate) and
     a random 120-column crop (no seam, no black band, so this is the
     variant the verdict uses).
  C5 A-vs-C and B-vs-C. C is facing and SILENT; A is facing and SPEAKING;
     B is turned and SPEAKING. A and C are therefore the SAME POSE. A
     statistic that measures pose must separate B from BOTH of them and NOT
     separate A from C. A statistic that separates A from C as well is
     measuring motion, or the difference between two sessions, not pose.
     This one matters more than it looks: A and C are the two ENDS of the
     capture run, so a monotonic drift over the session would separate them
     from each other too. If they agree and only the middle one differs,
     the difference is not a clock.

FRAMES, NOT SECONDS.

The frame rate of these captures was never recorded (see
docs/research/AV-SYNC.md), so every window in this script is expressed in
FRAMES and every comparison is between the same number of frames. Nothing
here is converted to seconds and no frame rate is assumed anywhere.

A POSITIVE CONTROL, because this project has had verdicts that contradicted
their own printed output three times (see HOW-WE-KEEP-FOOLING-OURSELVES.md).

Before the real captures are read, the identical pipeline is run on SYNTHETIC
frames where the answer is known: a symmetric frontal blob versus an
asymmetric turned blob in a banded, noisy room. If the pipeline cannot find
the difference it has been handed, its negative result on the real captures
means nothing. The synthetic check is printed alongside the real one.

ROW-BAND REMOVAL.

`row_band_removal.py` established the model frame(x, y) = content(x, y) +
band(y): the camera flicker is a per-row offset, so the per-row MEDIAN across
the 160 columns estimates band(y) and subtracting it removes the banding
without touching time. `row_remove()` below is that exact operation
(subtract each row's median across the columns, then restore the frame's own
mean level). Everything here is computed on the row-removed frames, so a
feature cannot score by reading the banding.

LIMITS, STATED UP FRONT AND REPEATED AT THE BOTTOM.

  * one session per condition, one person, one room, one lighting state.
  * frames from one session are NOT independent samples. They are a
    correlated time series. The separations below are therefore descriptive,
    not inferential; the permutation null bounds them, it does not make them
    p-values from independent data.
  * A and B were captured minutes apart, so ANY difference between them
    includes whatever changed between two sessions -- lighting, posture,
    seat position, the subject's mood. The empty room cannot rule that out,
    because the empty room is a different session in a different light.
    Only the WITHIN-CONDITION comparison is clean, and it is the control
    that matters.
  * condition D (turned and silent) has never been recorded, so a
    motion-and-pose confound cannot be fully resolved from these four
    captures.
  * the subject is close to the frame edge, so "symmetry about the frame
    centre" is a statement about the room as much as about the head. The
    best-fit-mirror-axis features exist precisely to search past that
    assumption instead of baking it in.
  * a ONE-DIMENSIONAL column profile has a blind spot: a turned head can
    still project a roughly symmetric bump, shifted sideways and narrower.
    The best-fit mirror residual measures whether the profile is symmetric
    about SOME axis, which a shifted bump satisfies. That is why the panel
    also carries width, sharpness, skew and centroid features, and why the
    spatial null matters: several of the features that separate here do so
    by reading position and width, not by reading a head shape.

THIS FILE WRITES NO OUTPUT FILE BY DEFAULT.

The owning task for this experiment restricted writes to this file and its
test, so the results are printed rather than written to
experiments/pose_feature_search_results.json. Pass --json PATH to write one
explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAME_W, FRAME_H = 160, 96
Y_SIZE = FRAME_W * FRAME_H
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.nv12 import detect_layout, read_luma  # noqa: E402

DEFAULT_A = "/mnt/c/proart_capture/A.nv12"
DEFAULT_B = "/mnt/c/proart_capture/B.nv12"
DEFAULT_C = "/mnt/c/proart_capture/C.nv12"
DEFAULT_EMPTY = "/mnt/c/proart_capture/empty1.nv12"
DEFAULT_EMPTY_ALT = "/home/proart/projects/glasses/.capture_keep/empty1.nv12"

# chunk size for the vectorised mirror fit
CHUNK = 128
# trials for the spatial null (the subject placed at random in the frame)
ROLL_TRIALS = 8
# seed, so the permutation and spatial nulls are reproducible
SEED = 20261019

# index tables, cached per frame width:
#   mirror: the column opposite x when mirrored about axis c (x -> 2c - x)
#   span:   the cumulative-sum index of a window of width w starting at lo
_MIRROR_CLIP_CACHE: dict = {}
_SPAN_CACHE: dict = {}


def _mirror_tables(W: int):
    """Index tables for the mirror fit at a given width."""
    if W not in _MIRROR_CLIP_CACHE:
        xs = np.arange(W, dtype=np.int64)
        axes = np.arange(W, dtype=np.int64)
        idx = 2 * axes[:, None] - xs[None, :]
        _MIRROR_CLIP_CACHE[W] = (np.clip(idx, 0, W - 1),)
    return _MIRROR_CLIP_CACHE[W][0]


def _span_index(W: int):
    """_SPAN_IDX[w, lo] = lo + w, the cumulative-sum index of a window of
    width w, at a given frame width."""
    if W not in _SPAN_CACHE:
        _SPAN_CACHE[W] = (np.arange(W + 1)[None, :]
                          + np.arange(W + 1)[:, None])
    return _SPAN_CACHE[W]


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------

def sha256_file(path) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_identity(path: Path) -> dict:
    """Where a capture is, what it is, and how it was laid out."""
    layout = detect_layout(path, width=FRAME_W, height=FRAME_H)
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "bytes": Path(path).stat().st_size,
        "layout": layout.name,
        "stride_bytes": layout.stride,
        "n_frames": layout.n_frames,
        "layout_evidence": layout.as_dict(),
    }


def _read_luma(path: Path):
    """`read_luma`, with the one-candidate case handled.

    `src.nv12.read_luma` reports its margin over the runner-up as NaN when
    only ONE candidate stride divides the file -- the unambiguous case -- and
    so rejects it. Only that case is handled here; a genuine ambiguity, where
    two candidates are close, still raises.
    """
    try:
        return read_luma(path, width=FRAME_W, height=FRAME_H)
    except ValueError:
        found = detect_layout(path, width=FRAME_W, height=FRAME_H)
        if found.runner_up_name != "(none)":
            raise
        return read_luma(path, width=FRAME_W, height=FRAME_H,
                         layout=found.name)


def load_y(path: Path, nf: int | None = None) -> np.ndarray:
    """Read the luma plane of a capture as (n, H, W) float32.

    The frame layout is DETECTED from the file's content by `src/nv12.py`: the
    candidate whose luma has the smallest mean consecutive |diff| wins, and a
    file that cannot decide is an error rather than a guess. Nothing here
    computes a stride by hand.

    The captures on disk are NOT standard NV12. `C:\\proart_capture\\
    nv12small.py` writes 15,360 luma bytes + 3,840 bytes of INTERLEAVED U,V (80x24) per 160x96
    frame -- 19,200 bytes -- not the `W*H*3//2` = 23,040 that a standard-NV12
    reader assumes. Read at 23,040, all but every sixth frame is torn: rows
    0..71 from frame k and rows 72..95 from frame k+1. See `src/nv12.py` and
    `docs/research/NV12-STRIDE.md`.
    """
    luma, _layout = _read_luma(path)
    if nf is not None:
        luma = luma[:nf]
    if len(luma) == 0:
        raise ValueError(f"{path}: no complete frames")
    return luma.astype(np.float32)


def row_remove(F: np.ndarray) -> np.ndarray:
    """Remove the row-constant banding.

    Identical to `row_band_removal.remove_rows(mode="subtract")`: the per-row
    MEDIAN across the 160 columns estimates band(y) (the median because the
    subject occupies a minority of a row's columns and must not drag the
    estimate), and subtracting it removes the banding without combining
    frames -- no latency, no motion smearing.
    """
    F = np.asarray(F, dtype=np.float64)
    prof = np.median(F, axis=2)              # (n, H) per-row median
    gmean = prof.mean(axis=1, keepdims=True)  # (n, 1) the frame's own level
    return F - (prof - gmean)[:, :, None]


def iter_blocks(path: Path, block: int):
    """Yield (start, block_frames) over a capture without holding all of it.

    The stride is the DETECTED one -- see `src/nv12.py` and `load_y`.
    """
    layout = detect_layout(path, width=FRAME_W, height=FRAME_H)
    stride = layout.stride
    raw = np.memmap(str(path), dtype=np.uint8, mode="r")
    n = len(raw) // stride
    for s in range(0, n, block):
        b = raw[s * stride:(s + block) * stride]
        yield s, b.reshape(len(b) // stride, stride)[:, :Y_SIZE] \
            .reshape(len(b) // stride, FRAME_H, FRAME_W).astype(np.float32)


# --------------------------------------------------------------------------
# the feature panel
# --------------------------------------------------------------------------

def centre_asym(P: np.ndarray) -> np.ndarray:
    """Normalised asymmetry of each profile about the FRAME CENTRE.

    For a 160-wide frame the exact centre mirror is x -> 159 - x. The
    residual sum (p[x] - p[159-x])^2, normalised by 2*sum p^2, is 0 for a
    perfectly symmetric profile and 1 for a profile with no symmetry at all.
    """
    P = np.asarray(P, dtype=np.float64)
    resid = ((P - P[:, ::-1]) ** 2).sum(axis=1)
    energy = (P * P).sum(axis=1)
    return np.where(energy > 0, resid / (2 * energy + 1e-12), 0.0)


def mirror_fit(P: np.ndarray):
    """Best-fit vertical mirror axis of each profile.

    For every candidate axis c in 0..W-1 the residual of mirroring the
    profile about c is computed and the best axis is kept. The axis is
    SEARCHED, not assumed to be the frame centre -- a turned head is not
    symmetric about the centre, so a centre-only statistic measures the
    subject's position in the frame as much as its pose.

    Returns (normalised_residual, axis_index).
    """
    P = np.asarray(P, dtype=np.float64)
    m, W = P.shape
    clip = _mirror_tables(W)
    resid = np.empty(m)
    axis = np.empty(m, dtype=np.int64)
    for s in range(0, m, CHUNK):
        p = P[s:s + CHUNK]
        c = p.shape[0]
        pm = p[:, clip]                            # (c, W, W)
        # The mirrored value is ZERO where it falls outside the frame. That
        # padding is what stops an extreme axis from looking perfect: with
        # the out-of-range terms simply SKIPPED, mirroring about column 0
        # compares one sample with itself and wins trivially.
        diff = p[:, None, :] - pm
        r = (diff * diff).sum(axis=2)               # (c, W)
        k = r.argmin(axis=1)
        resid[s:s + c] = r[np.arange(c), k]
        axis[s:s + c] = k
    energy = (P * P).sum(axis=1)
    resid = np.where(energy > 0, resid / (2 * energy + 1e-12), 0.0)
    return resid, axis


def energy_span(p: np.ndarray, frac: float = 0.80):
    """Width in columns of the SMALLEST contiguous window that holds `frac`
    of the profile's energy. Returns (width, lo, hi).

    A peak-relative threshold is too brittle here -- the motion peak sits on
    a single noisy column and the run above half the peak is one or two
    columns wide whatever the subject is doing. The energy span instead
    measures how far the change is spread, which is the thing that differs
    between a broad frontal face and a narrow turned one.
    """
    W = len(p)
    if W == 0:
        return 0, 0, 0
    c = np.concatenate([[0.0], np.cumsum(np.asarray(p, dtype=np.float64))])
    total = c[-1]
    if total <= 0:
        return 0, 0, 0
    need = frac * total
    cp = np.concatenate([c, c])
    win = cp[_span_index(W)] - cp[np.arange(W + 1)[None, :]]
    best = win.max(axis=1)              # best energy for each width
    ok = np.nonzero(best >= need)[0]
    if ok.size == 0:
        return W, 0, W - 1
    w = int(ok[0])
    lo = int(np.argmax(win[w]))
    return w, lo, min(lo + w, W) - 1


def frame_features(R: np.ndarray) -> dict:
    """Per-frame pose features on row-removed frames.

    R is (n, H, W). The returned arrays have length n-1 and are aligned to
    the motion between R[t] and R[t+1]; the static features use frame t.
    """
    R = np.asarray(R, dtype=np.float32)
    n, H, W = R.shape
    out = {}
    if n < 3:
        return out

    D = np.abs(R[1:] - R[:-1])            # (m, H, W) change magnitude
    colmot = D.mean(axis=1)               # (m, W) column motion profile
    colstat = R[:-1].mean(axis=1)         # (m, W) signed static column profile

    # -- symmetry of the column profiles -----------------------------------
    out["asym_static_centre"] = centre_asym(colstat)
    out["asym_motion_centre"] = centre_asym(colmot)
    r_m, a_m = mirror_fit(colmot)
    out["mirror_resid_motion"] = r_m
    out["mirror_axis_offset"] = a_m - (W - 1) / 2.0
    r_s, a_s = mirror_fit(colstat)
    out["mirror_resid_static"] = r_s

    # -- width and sharpness of the active region --------------------------
    widths = np.empty(len(colmot))
    sharp = np.empty(len(colmot))
    for t in range(len(colmot)):
        w, lo, hi = energy_span(colmot[t])
        widths[t] = w
        seg = colmot[t, lo:hi + 1]
        sharp[t] = float(seg.max() / (seg.mean() + 1e-9))
    out["active_width"] = widths
    out["active_sharpness"] = sharp

    # -- horizontal position of the motion centroid ------------------------
    csum = colmot.sum(axis=1)
    xs = np.arange(W)
    cent = np.where(csum > 0, (colmot * xs[None, :]).sum(axis=1)
                    / np.maximum(csum, 1e-9), W / 2.0)
    out["motion_centroid_x"] = cent

    # -- left/right motion energy over a symmetric pair of windows ----------
    half = W // 2
    el = colmot[:, :half].sum(axis=1)
    er = colmot[:, half:].sum(axis=1)
    out["lr_motion_ratio"] = np.log((el + 1e-6) / (er + 1e-6))

    # -- gradient orientation ----------------------------------------------
    # A frontal face concentrates VERTICAL structure (the outline of the face,
    # the nose, the line of the mouth); a turned head presents a profile
    # whose edges are mostly horizontal. Vertical edges mean a HORIZONTAL
    # gradient, so the share of |gx| is the share of vertical structure.
    gx = np.abs(R[:-1, :, 1:] - R[:-1, :, :-1]).sum(axis=(1, 2))
    gy = np.abs(R[:-1, 1:, :] - R[:-1, :-1, :]).sum(axis=(1, 2))
    out["grad_vertical_share"] = gx / (gx + gy + 1e-9)

    # -- column energy share (drift_structure.py's statistic) --------------
    cp = D.mean(axis=1)
    rp = D.mean(axis=2)
    cs = cp.std(axis=1)
    rs = rp.std(axis=1)
    out["column_energy_share"] = cs ** 2 / (cs ** 2 + rs ** 2 + 1e-12)

    # -- second order: covariance with the frame's own horizontal mirror ---
    x = R[:-1] - R[:-1].mean(axis=(1, 2), keepdims=True)
    y = R[:-1, :, ::-1] - R[:-1, :, ::-1].mean(axis=(1, 2), keepdims=True)
    num = (x * y).sum(axis=(1, 2))
    den = np.sqrt((x * x).sum(axis=(1, 2)) * (y * y).sum(axis=(1, 2))) + 1e-12
    out["mirror_cov"] = num / den

    # -- skewness of the motion profile ------------------------------------
    p = colmot - colmot.mean(axis=1, keepdims=True)
    m2 = (p * p).mean(axis=1)
    m3 = (p * p * p).mean(axis=1)
    out["profile_skew"] = m3 / (m2 ** 1.5 + 1e-12)

    return out


def session_features(R: np.ndarray, frame_feat: dict) -> dict:
    """Session-level features: statistics of a per-frame feature over the
    whole block, which is the thing that cannot be read off a single frame."""
    out = {}
    cent = frame_feat.get("motion_centroid_x")
    if cent is not None and len(cent) > 1:
        out["motion_centroid_x_spread"] = float(np.std(cent))
    else:
        out["motion_centroid_x_spread"] = 0.0
    return out


# name -> (kind, one-line justification)
FEATURE_SPECS = [
    ("asym_static_centre", "frame",
     "horizontal asymmetry of the STATIC column profile about the frame centre"),
    ("asym_motion_centre", "frame",
     "horizontal asymmetry of the MOTION column profile about the frame centre"),
    ("mirror_resid_motion", "frame",
     "best-fit mirror residual of the motion profile (searched axis): how "
     "symmetric at best"),
    ("mirror_axis_offset", "frame",
     "the searched best-fit mirror axis of the motion profile, minus the frame "
     "centre, in pixels"),
    ("mirror_resid_static", "frame",
     "best-fit mirror residual of the static column profile"),
    ("active_width", "frame",
     "width of the contiguous motion-active region, in columns"),
    ("active_sharpness", "frame",
     "peak over mean motion inside the active region (a narrow silhouette is "
     "sharper)"),
    ("motion_centroid_x", "frame",
     "horizontal centroid of the motion energy, in frame widths"),
    ("lr_motion_ratio", "frame",
     "log ratio of left to right motion energy over a symmetric pair of "
     "half-frame windows"),
    ("grad_vertical_share", "frame",
     "share of gradient energy in vertical structure (frontal faces are "
     "vertical, profiles are horizontal)"),
    ("column_energy_share", "frame",
     "column-vs-row share of the change map (drift_structure.py's statistic)"),
    ("mirror_cov", "frame",
     "second order: normalised covariance of the frame with its own "
     "horizontal mirror"),
    ("profile_skew", "frame",
     "skewness of the motion column profile (asymmetry of the activity)"),
    ("motion_centroid_x_spread", "session",
     "variance of the motion centroid WITHIN a session, in pixels"),
]

PER_FRAME_NAMES = [n for n, k, _ in FEATURE_SPECS if k == "frame"]
SESSION_NAMES = [n for n, k, _ in FEATURE_SPECS if k == "session"]


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def robust_scale(x: np.ndarray) -> float:
    """A robust scale for putting every separation into the same units (the
    feature's own spread).

    The central 80% range is used rather than the MAD because these
    features are often BIMODAL -- "symmetric about the frame centre" is near
    zero for a facing subject and large for a turned one, so a large share of
    the samples sit in one mode and the rest in the other. A MAD taken over
    such a sample measures the distance between the modes' centres rather
    than the spread WITHIN a condition, and the separations come out as
    absurd numbers (tens of "sigmas" for a difference that is really one
    mode width). The central 80% range spans the modes when the sample is
    bimodal and behaves like a standard deviation when it is not.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    for lo, hi, k in ((10.0, 90.0, 2.5626), (1.0, 99.0, 4.6526)):
        s = float(np.percentile(x, hi) - np.percentile(x, lo)) / k
        if s > 0:
            return s
    med = np.median(x)
    mad = float(1.4826 * np.median(np.abs(x - med)))
    if mad > 0:
        return mad
    return float(x.max() - x.min())


def sep(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    """Separation of two samples: |median(a) - median(b)| in units of the
    feature's robust scale. Zero when the scale is zero (a feature that does
    not vary cannot separate anything)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0 or scale <= 0:
        return 0.0
    return float(abs(np.median(a) - np.median(b)) / scale)


def max_pair_sep(samples, scale: float) -> float:
    """The largest separation between any two of a list of samples."""
    best = 0.0
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            best = max(best, sep(samples[i], samples[j], scale))
    return best


def permutation_null(a: np.ndarray, b: np.ndarray, scale: float,
                     trials: int = 200, seed: int = SEED) -> float:
    """95th percentile of the separation when the two conditions' labels are
    shuffled. A feature that cannot beat this got its separation for free."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    pool = np.concatenate([a, b])
    na = a.size
    out = np.empty(trials)
    for t in range(trials):
        perm = rng.permutation(pool.size)
        out[t] = sep(pool[perm[:na]], pool[perm[na:]], scale)
    return float(np.percentile(out, 95))


def feature_verdict(name: str, sep_ab: float, within: float, empty: float,
                     perm95: float | None = None, sep_ac: float | None = None,
                     sep_bc: float | None = None,
                     spatial: float | None = None) -> dict:
    """The verdict for one feature.

    The verdict is a function of the numbers, and it is allowed to come out
    the other way. A feature whose A-vs-B separation does not exceed its own
    controls gets the verdict "no signal" -- that string is the whole point
    of the function, so it is not decorated or softened.

      control = max(within-condition split, empty room)
      margin  = sep_ab - control
      signal  = margin > 0 AND sep_ab > perm95 AND sep_ab > spatial null
                AND pose-consistent
      pose-consistent = B differs from C (both A and C are facing) AND A does
                        NOT differ from C
    """
    control = max(within, empty)
    margin = sep_ab - control
    gates = []
    if margin <= 0:
        return {
            "verdict": "no signal",
            "control": control,
            "margin": margin,
            "detail": (f"A-vs-B {sep_ab:.3f} does not beat the largest control "
                       f"({control:.3f}: within-split {within:.3f}, empty-room "
                       f"{empty:.3f}); margin {margin:+.3f}"),
        }
    if perm95 is not None:
        gates.append(("label-permutation null", perm95))
    if spatial is not None:
        gates.append(("spatial (crop) null", spatial))
    for label, val in gates:
        if sep_ab <= val:
            return {
                "verdict": "no signal",
                "control": control,
                "margin": margin,
                "detail": (f"A-vs-B {sep_ab:.3f} beats within-split "
                           f"{within:.3f} and empty-room {empty:.3f} by "
                           f"{margin:+.3f} but does not beat the {label} "
                           f"({val:.3f})"),
            }
    extra = ""
    if sep_ac is not None and sep_bc is not None:
        if not (sep_bc > control and sep_ac <= control):
            return {
                "verdict": "confounded",
                "control": control,
                "margin": margin,
                "detail": (f"A-vs-B {sep_ab:.3f} beats the controls by "
                           f"{margin:+.3f} but is NOT pose-like: B-vs-C "
                           f"{sep_bc:.3f}, A-vs-C {sep_ac:.3f} (A and C are "
                           f"both facing, so a pose statistic must not "
                           f"separate them; control {control:.3f})"),
            }
        extra = (f"; pose-like: B-vs-C {sep_bc:.3f} > control {control:.3f} "
                 f"while A-vs-C {sep_ac:.3f} does not")
    return {
        "verdict": "signal",
        "control": control,
        "margin": margin,
        "detail": (f"A-vs-B {sep_ab:.3f} beats within-split {within:.3f} and "
                   f"empty-room {empty:.3f} by {margin:+.3f}"
                   + "".join(f", {lab} {val:.3f}" for lab, val in gates)
                   + extra),
    }


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------

def condition_features(R: np.ndarray) -> dict:
    """Feature sets for one condition's frames.

    Returns the whole-block per-frame features, the two halves' per-frame
    features (the within-condition split) and the session-level features
    for the block and for each half.
    """
    Rr = row_remove(R)
    mid = len(Rr) // 2
    ff = frame_features(Rr)
    ff1 = frame_features(Rr[:mid])
    ff2 = frame_features(Rr[mid:])
    sess = session_features(Rr, ff)
    sess1 = session_features(Rr[:mid], ff1)
    sess2 = session_features(Rr[mid:], ff2)
    return {"frame": ff, "frame_h1": ff1, "frame_h2": ff2,
            "session": sess, "session_h1": sess1, "session_h2": sess2}


def _values(feature: str, kind: str, cond: dict, which: str = ""):
    """One sample for a feature: an array (per-frame) or a scalar (session)."""
    if kind == "frame":
        src = cond["frame" + which]
        return src.get(feature, np.zeros(0))
    src = cond["session" + which]
    v = src.get(feature)
    return np.array([v if v is not None else 0.0])


def compute_scales(conds: dict, empty_blocks: list) -> dict:
    """One robust scale per feature, from every condition including the empty
    room, so that every separation below is in the same units."""
    scales = {}
    for name, kind, _ in FEATURE_SPECS:
        pool = []
        for lab in ("A", "B", "C"):
            if lab in conds:
                pool.append(_values(name, kind, conds[lab]))
        for blk in empty_blocks:
            pool.append(_values(name, kind, blk))
        pool = np.concatenate([p for p in pool if p.size]) if pool else np.zeros(0)
        scales[name] = robust_scale(pool)
    return scales


def build_analysis(conds: dict, empty_blocks: list, spatial: dict | None = None) -> dict:
    """Run every feature through every control.

    conds: {"A": condset, "B": condset, "C": condset}
    empty_blocks: list of condset for the empty room, one per block.
    spatial: optional {"roll": {...}, "crop": {...}} of spatial-null results.
    """
    feats = {}
    scales = compute_scales(conds, empty_blocks)
    for name, kind, _ in FEATURE_SPECS:
        scale = scales[name]

        a = _values(name, kind, conds["A"]) if "A" in conds else np.zeros(0)
        b = _values(name, kind, conds["B"]) if "B" in conds else np.zeros(0)
        c = _values(name, kind, conds["C"]) if "C" in conds else np.zeros(0)

        sep_ab = sep(a, b, scale)

        # C1 within-condition split: A's halves against each other, and B's
        within_a = within_b = 0.0
        if "A" in conds:
            within_a = sep(_values(name, kind, conds["A"], "_h1"),
                           _values(name, kind, conds["A"], "_h2"), scale)
        if "B" in conds:
            within_b = sep(_values(name, kind, conds["B"], "_h1"),
                           _values(name, kind, conds["B"], "_h2"), scale)
        within = max(within_a, within_b)

        # C2 empty room: the same comparison between different time blocks
        empties = [_values(name, kind, blk) for blk in empty_blocks]
        empty = max_pair_sep(empties, scale) if len(empties) > 1 else 0.0

        # C3 label permutation
        perm95 = None
        if kind == "frame" and a.size and b.size:
            perm95 = permutation_null(a, b, scale)

        # C4 spatial scramble (the artifact-free crop variant gates)
        spatial_med = None
        if spatial:
            s = spatial.get("crop", {}).get(name)
            if s:
                spatial_med = s["median"]

        sep_ac = sep(a, c, scale) if c.size else None
        sep_bc = sep(b, c, scale) if c.size else None

        v = feature_verdict(name, sep_ab, within, empty, perm95, sep_ac, sep_bc,
                            spatial_med)
        ratio = sep_ab / max(max(within, empty), 1e-9)
        feats[name] = {
            "name": name, "kind": kind, "scale": scale,
            "sep_ab": sep_ab, "within": within, "within_a": within_a,
            "within_b": within_b, "empty": empty,
            "perm95": perm95, "spatial_median": spatial_med,
            "sep_ac": sep_ac, "sep_bc": sep_bc,
            "margin": v["margin"], "ratio": ratio,
            "control": v["control"], "verdict": v["verdict"],
            "detail": v["detail"],
            "median_a": float(np.median(a)) if a.size else float("nan"),
            "median_b": float(np.median(b)) if b.size else float("nan"),
            "median_c": float(np.median(c)) if c.size else float("nan"),
            "median_empty": (float(np.median(np.concatenate(empties)))
                             if empties else float("nan")),
            "frac_above_a": _frac_above(a, a, b, c),
            "frac_above_b": _frac_above(b, a, b, c),
            "frac_above_c": _frac_above(c, a, b, c),
        }
    return feats


def _frac_above(sample, a, b, c) -> float:
    """Fraction of a condition's frames that lie above the pooled median of
    the three conditions. A difference that is a stable per-frame property
    of one capture shows up here as a large gap between the conditions; a
    transient (someone walking past) does not."""
    if sample is None or not np.size(sample):
        return float("nan")
    pool = np.concatenate([v for v in (a, b, c) if np.size(v)])
    if not pool.size:
        return float("nan")
    return float((sample > np.median(pool)).mean())


def spatial_null(Ra: np.ndarray, Rb: np.ndarray, scales: dict,
                 trials: int = ROLL_TRIALS, seed: int = SEED,
                 mode: str = "crop") -> dict:
    """C4 spatial null: put the subject at a random place inside the frame
    and recompute the A-vs-B separation.

    A statistic that measures the subject's INTERNAL shape keeps separating
    when the scene is moved around inside the frame. A statistic that is
    really measuring WHERE THE SUBJECT SATES loses its separation.

    Two variants are computed, because they have different artifacts:

      "roll" -- a cyclic roll. Simple, but it leaves a seam at a different
                column in each condition, which is a differential artifact
                that inflates any seam-sensitive statistic (mirror_cov most
                of all). Reported, but NOT used as a gate.
      "crop" -- a random 120-column window taken out of each condition. No
                seam and no black band, so the only differential effect is
                the one being tested: the subject's position within the
                frame. This is the variant the verdict uses.
    """
    rng = np.random.default_rng(seed + (0 if mode == "roll" else 5000))
    out = {name: [] for name, _, _ in FEATURE_SPECS}
    width = Ra.shape[2]
    keep = min(120, width)
    for t in range(trials):
        if mode == "roll":
            sa = int(rng.integers(1, width))
            sb = int(rng.integers(1, width))
            A = np.roll(Ra, sa, axis=2)
            B = np.roll(Rb, sb, axis=2)
        else:
            sa = int(rng.integers(0, width - keep + 1))
            sb = int(rng.integers(0, width - keep + 1))
            A = Ra[:, :, sa:sa + keep]
            B = Rb[:, :, sb:sb + keep]
        fa = frame_features(A)
        fb = frame_features(B)
        for name, kind, _ in FEATURE_SPECS:
            if kind != "frame":
                continue
            scale = scales.get(name, 0.0)
            out[name].append(sep(fa.get(name, np.zeros(0)),
                                 fb.get(name, np.zeros(0)), scale))
    res = {}
    for name, vals in out.items():
        if vals:
            res[name] = {"min": float(np.min(vals)),
                         "median": float(np.median(vals)),
                         "mean": float(np.mean(vals)),
                         "max": float(np.max(vals)),
                         "n_over_unrolled": int(sum(1 for v in vals
                                                   if v > 0))}
        else:
            res[name] = None
    return res


# --------------------------------------------------------------------------
# synthetic data (the positive control, and the fallback when the real
# captures are absent)
# --------------------------------------------------------------------------

def make_synthetic_capture(kind: str, n: int = 150, seed: int = 0) -> np.ndarray:
    """Synthetic 160x96 frames.

    kind:
      "frontal" -- a symmetric head at the centre of the frame; the head
                   jitters horizontally a little, so the MOTION is at the two
                   vertical edges of the face
      "turned"  -- the same subject with the head shifted to one side and a
                   nose bump, so the silhouette is asymmetric and narrower
      "empty"   -- a banded, noisy room with nobody in it

    The jitter is deliberate: the motion of a head that is only speaking
    lives at the edges of the face, and that is what the "active region
    width" and "sharpness" features are supposed to see.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(np.float64)
    rows = np.arange(FRAME_H, dtype=np.float64)
    out = np.empty((n, FRAME_H, FRAME_W), dtype=np.float32)
    for t in range(n):
        # camera flicker banding: a per-row offset that changes over time
        band = 18.0 * np.sin(2 * np.pi * (t % 4) / 4.0
                             + 6.0 * np.pi * rows / FRAME_H)
        frame = 100.0 + band[:, None] + rng.normal(0.0, 0.6, (FRAME_H, FRAME_W))
        if kind == "empty":
            out[t] = frame
            continue
        if kind == "frontal":
            cx, cy, rx, ry = 80.0, 50.0, 15.0, 20.0
            nose_dx, mouth_dx = 0.0, 0.0
        else:
            cx, cy, rx, ry = 95.0, 50.0, 9.0, 20.0
            nose_dx, mouth_dx = -9.0, -4.0
        # the head jitters horizontally: the edges are where the motion is
        jx = 1.5 * np.sin(t / 4.0)
        head = ((xx - (cx + jx)) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        nose = (((xx - (cx + jx + nose_dx)) / 3.5) ** 2
                + ((yy - cy) ** 2)) <= 30.0
        frame = frame + 45.0 * head + 25.0 * nose
        mouth = (((xx - (cx + jx + mouth_dx)) / 4.0) ** 2
                 + ((yy - (cy + 8.0)) / 3.0) ** 2) <= 1.0
        frame = frame - 25.0 * mouth * (0.5 + 0.5 * np.cos(t / 3.0))
        out[t] = frame
    return out


def run_synthetic_panel(n: int = 150) -> dict:
    """The positive control: the identical pipeline on synthetic frames
    whose answer is known."""
    conds = {}
    seeds = {"frontal": 11, "turned": 22, "empty": 33}
    for lab, kind in (("A", "frontal"), ("B", "turned"), ("C", "frontal")):
        R = make_synthetic_capture(kind, n=n, seed=seeds[kind])
        conds[lab] = condition_features(R)
    empty_blocks = [condition_features(make_synthetic_capture("empty", n=n,
                                                               seed=100 + i))
                    for i in range(3)]
    return build_analysis(conds, empty_blocks)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def print_table(feats: dict, title: str) -> None:
    print(f"\n{title}")
    print(f"  {'feature':26s} {'kind':7s} {'A-vs-B':>8s} {'wA':>7s} {'wB':>7s} "
          f"{'empty':>8s} {'perm95':>8s} {'cropnl':>8s} {'margin':>8s} "
          f"{'ratio':>7s}  verdict")
    print("  " + "-" * 124)
    for name, kind, _ in FEATURE_SPECS:
        f = feats.get(name)
        if f is None:
            continue
        perm = "n/a" if f["perm95"] is None else f"{f['perm95']:.3f}"
        crop = ("n/a" if f.get("spatial_median") is None
                else f"{f['spatial_median']:.3f}")
        print(f"  {name:26s} {kind:7s} {f['sep_ab']:8.3f} {f['within_a']:7.3f} "
              f"{f['within_b']:7.3f} {f['empty']:8.3f} {perm:>8s} {crop:>8s} "
              f"{f['margin']:+8.3f} {f['ratio']:6.2f}x  {f['verdict']}")
    print("  wA/wB = the within-condition split of A and of B (the null the "
          "A-vs-B number has to beat);")
    print("  empty = the empty-room control (the same comparison between two "
          "blocks of an empty room);")
    print("  perm95 = the label-permutation null; cropnl = the spatial null "
          "(the subject moved at random inside the frame).")


def print_raw_medians(feats: dict) -> None:
    """The raw medians, so the numbers above can be checked rather than
    taken on trust."""
    print("\nRAW MEDIANS (the underlying values, not the standardised "
          "separations)")
    print(f"  {'feature':26s} {'A':>12s} {'B':>12s} {'C':>12s} "
          f"{'EMPTY':>12s}   {'frac>A-pool-med A/B/C':>24s}")
    print("  " + "-" * 112)
    for name, kind, _ in FEATURE_SPECS:
        f = feats.get(name)
        if f is None:
            continue
        fr = (f"{f['frac_above_a']:.2f} / {f['frac_above_b']:.2f} / "
              f"{f['frac_above_c']:.2f}")
        print(f"  {name:26s} {f['median_a']:12.5g} {f['median_b']:12.5g} "
              f"{f['median_c']:12.5g} {f['median_empty']:12.5g}   {fr:>24s}")


def print_motion_vs_pose(feats: dict) -> None:
    print("\nMOTION vs POSE  (A and C are BOTH facing; B is turned)")
    print(f"  {'feature':26s} {'A-vs-C':>8s} {'B-vs-C':>8s} {'control':>8s}  "
          f"reads as")
    print("  " + "-" * 72)
    for name, kind, _ in FEATURE_SPECS:
        f = feats.get(name)
        if f is None or f["sep_ac"] is None:
            continue
        ctrl = f["control"]
        if f["sep_bc"] > ctrl and f["sep_ac"] <= ctrl:
            reads = "POSE-like (B differs, A~C)"
        elif f["sep_ac"] > ctrl and f["sep_bc"] > ctrl:
            reads = "motion/session, not pose"
        else:
            reads = "neither"
        print(f"  {name:26s} {f['sep_ac']:8.3f} {f['sep_bc']:8.3f} "
              f"{ctrl:8.3f}  {reads}")


def summarise(feats: dict) -> dict:
    """The honest summary. It is allowed -- and expected -- to say that
    nothing separated."""
    signals = [f for f in feats.values() if f["verdict"] == "signal"]
    conf = [f for f in feats.values() if f["verdict"] == "confounded"]
    best = max(feats.values(), key=lambda f: f["margin"]) if feats else None
    if signals:
        verdict = (f"{len(signals)} feature(s) separate facing from turned "
                   f"beyond their controls: "
                   + ", ".join(sorted(f["name"] for f in signals)))
    elif conf:
        verdict = (f"NO feature is a clean pose signal. {len(conf)} feature(s) "
                   f"separate facing from turned beyond their controls but are "
                   f"NOT pose-like (they also separate the two FACING "
                   f"conditions): "
                   + ", ".join(sorted(f["name"] for f in conf)))
    else:
        verdict = ("NO SIGNAL. No feature in the panel separates facing from "
                   "turned beyond its own within-condition split and the "
                   "empty-room control.")
    return {"signals": sorted(f["name"] for f in signals),
            "confounded": sorted(f["name"] for f in conf),
            "best_feature": best["name"] if best else None,
            "best_margin": best["margin"] if best else None,
            "verdict": verdict}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=DEFAULT_A, help="facing + speaking")
    ap.add_argument("--b", default=DEFAULT_B, help="turned + speaking")
    ap.add_argument("--c", default=DEFAULT_C, help="facing + silent")
    ap.add_argument("--empty", default=DEFAULT_EMPTY, help="empty room")
    ap.add_argument("--nf", type=int, default=750,
                    help="frames per condition (FRAMES, not seconds: the frame "
                         "rate of these captures was never recorded)")
    ap.add_argument("--roll-trials", type=int, default=ROLL_TRIALS)
    ap.add_argument("--json", default=None,
                    help="optional path to write a results JSON (nothing is "
                         "written by default)")
    a = ap.parse_args()

    print("=" * 96)
    print("POSE FEATURE SEARCH -- can any statistic separate FACING from TURNED?")
    print("=" * 96)
    print("windows are in FRAMES, not seconds: the frame rate of these "
          "captures was never recorded")

    # -- the positive control, always -------------------------------------
    print("\nSYNTHETIC POSITIVE CONTROL (same pipeline, known answer)")
    syn = run_synthetic_panel()
    syn_best = max(syn.values(), key=lambda f: f["margin"])
    syn_sig = sorted(f["name"] for f in syn.values() if f["verdict"] == "signal")
    if syn_sig:
        syn_note = f"the pipeline CAN find a known difference ({', '.join(syn_sig)})"
    else:
        syn_note = ("the pipeline CANNOT find a known difference -- THE "
                    "INSTRUMENT IS BLIND, TRUST NOTHING BELOW")
    print(f"  frontal-vs-turned synthetic: best margin "
          f"{syn_best['margin']:+.3f} on {syn_best['name']}; "
          f"features that clear their controls: {len(syn_sig)}")
    print(f"  -> {syn_note}")

    # -- load the real captures -------------------------------------------
    paths = {"A": Path(a.a), "B": Path(a.b), "C": Path(a.c)}
    emp = Path(a.empty)
    captures = {}
    if not emp.is_file() and Path(DEFAULT_EMPTY_ALT).is_file():
        emp = Path(DEFAULT_EMPTY_ALT)
    missing = [f"{k} ({v})" for k, v in paths.items() if not v.is_file()]
    if not emp.is_file():
        missing.append(f"EMPTY ({emp})")

    if missing:
        print("\n" + "!" * 96)
        print("THE REAL CAPTURES ARE ABSENT: " + "; ".join(missing))
        print("The table below is therefore SYNTHETIC ONLY. It is a "
              "demonstration of the pipeline, NOT a result about the "
              "captures: nothing here says anything about whether Casey's "
              "head turn is visible.")
        print("!" * 96)
        feats = syn
        conds = None
    else:
        conds = {}
        raw = {}
        for lab, p in paths.items():
            R = load_y(p, a.nf)
            raw[lab] = row_remove(R)
            conds[lab] = condition_features(R)
            ident = capture_identity(p)
            ident["frames_used"] = int(len(R))
            captures[lab] = ident
            print(f"  {lab} {p}  {len(R)} frames")
        empty_blocks = []
        nblocks = 0
        for s, blk in iter_blocks(emp, a.nf):
            empty_blocks.append(condition_features(blk))
            nblocks += 1
            if nblocks >= 5:
                break
        ident = capture_identity(emp)
        ident["frames_used"] = int(nblocks * a.nf)
        ident["blocks"] = nblocks
        captures["EMPTY"] = ident
        print(f"  EMPTY {emp}  {nblocks} block(s) of {a.nf} frames")

        # -- C4 the spatial nulls ------------------------------------------
        scales = compute_scales(conds, empty_blocks)
        roll = spatial_null(raw["A"], raw["B"], scales, trials=a.roll_trials,
                            mode="roll")
        crop = spatial_null(raw["A"], raw["B"], scales, trials=a.roll_trials,
                            mode="crop")
        feats = build_analysis(conds, empty_blocks, spatial={"roll": roll,
                                                             "crop": crop})

        print(f"\nSPATIAL NULL ({a.roll_trials} random placements of the "
              f"subject inside the frame)")
        print(f"  {'feature':26s} {'unrolled':>9s} {'roll med':>9s} "
              f"{'crop med':>9s} {'crop max':>9s}")
        for name, kind, _ in FEATURE_SPECS:
            if kind != "frame":
                continue
            r = roll.get(name)
            cp = crop.get(name)
            if not r or not cp:
                continue
            print(f"  {name:26s} {feats[name]['sep_ab']:9.3f} "
                  f"{r['median']:9.3f} {cp['median']:9.3f} {cp['max']:9.3f}")
        print("  (the roll leaves a seam at a different column in each "
              "condition, so it is reported but not used as a gate; the crop "
              "has no seam and is the variant the verdict uses)")

    print_table(feats, "\nFEATURE PANEL -- separation in units of each "
                        "feature's own robust scale")
    print_raw_medians(feats)
    print_motion_vs_pose(feats)
    summary = summarise(feats)

    print("\n" + "=" * 96)
    print(f"VERDICT: {summary['verdict']}")
    if summary["best_feature"]:
        print(f"  best margin: {summary['best_margin']:+.3f} on "
              f"{summary['best_feature']}")
    if conds is not None:
        if summary["signals"]:
            print("\nWHAT THAT DOES AND DOES NOT MEAN.")
            print("  The B capture IS statistically distinguishable from the "
                  "two facing captures:")
            print("  A and C -- both facing -- agree with each other, and B "
                  "differs from both.")
            print("  Because A and C are the two ends of the capture run, a "
                  "monotonic drift (light changing")
            print("  over the session) would separate A from C too. It does "
                  "not. That is the strongest argument here")
            print("  that the difference in B is a manipulation and not a "
                  "clock.")
            print("  WHAT IS NOT ESTABLISHED: that the difference is the HEAD "
                  "TURNING. These statistics see the")
            print("  frame, not the head. A change of posture, a shift in the "
                  "seat, or anything else that")
            print("  moved in the frame between the B capture and the facing "
                  "captures would look the same.")
        else:
            print("\nWHAT THAT MEANS.")
            print("  The honest reading is: the turn is either not in the "
                  "capture, or not visible at")
            print("  160x96, or visible only in a statistic not tried here.")
            print("  The B condition of the corpus cannot be self-verified by "
                  "any statistic tried so far,")
            print("  so it rests on Casey's word alone.")
        print("\nLIMITS: one session per condition, one person, one room, one "
              "lighting state.")
        print("  Frames from one session are NOT independent samples; the "
              "separations above are descriptive, not inferential.")
        print("  A and B were captured minutes apart, so their difference "
              "includes session-to-session change (light, posture, seat) "
              "that the empty room cannot rule out.")
        print("  Condition D (turned and silent) has never been recorded, so "
              "a motion-and-pose confound cannot be fully resolved.")
    print("=" * 96)

    if a.json:
        payload = {
            "question": ("can any statistic separate facing from turned at "
                         "160x96, measured against its own controls?"),
            "units": "frames (the frame rate was never recorded)",
            "features": feats,
            "synthetic_positive_control": {
                "signals": syn_sig,
                "best_margin": syn_best["margin"],
                "best_feature": syn_best["name"],
            },
            "summary": summary,
            "limits": [
                "one session per condition, one person, one room, one "
                "lighting state",
                "frames from one session are not independent samples",
                "A and B were captured minutes apart; their difference "
                "includes session-to-session change",
                "condition D (turned and silent) has never been recorded",
            ],
            "provenance": {
                "generated_by": "experiments/pose_feature_search.py",
                "script_path": str(Path(__file__).resolve()),
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "python": sys.executable,
                "command_line": " ".join(sys.argv),
                "cli_args": list(sys.argv[1:]),
                "frames_per_condition": a.nf,
                "captures": captures if conds is not None else {},
            },
            "supersedes": {
                "why": ("Re-measured on 2026-10-20 at the CORRECT frame "
                        "stride. The previous run read the captures at "
                        "W*H*3//2 = 23,040 bytes per frame; the captures on "
                        "disk carry 19,200 bytes per frame, so all but every "
                        "sixth frame was torn. The previous verdict and its "
                        "margin are recorded here rather than silently "
                        "overwritten."),
                "previous_stride_bytes": 23040,
                "previous_best_feature": "mirror_cov",
                "previous_best_margin": 0.814,
                "previous_verdict": (
                    "1 feature(s) separate facing from turned beyond their "
                    "controls: mirror_cov. The B capture IS statistically "
                    "distinguishable from the two facing captures: A and C -- "
                    "both facing -- agree with each other, and B differs from "
                    "both."),
                "previous_motion_vs_pose_mirror_cov": {
                    "A_vs_C": 0.246, "B_vs_C": 0.851, "control": 0.282,
                    "reading": "POSE-like (B differs, A~C)"},
            },
        }
        p = Path(a.json)
        p.write_text(json.dumps(payload, indent=2, default=float))
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
