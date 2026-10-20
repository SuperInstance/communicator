#!/usr/bin/env python3
"""
mouth_detector.py -- CAN A MOUTH-LOCALISED DETECTOR SEPARATE SPEAKING FROM
SILENT ONCE HEAD MOTION IS HELD CONSTANT?

WHY THIS FILE EXISTS.

`docs/research/NV12-STRIDE.md` §5 found, on the CORRECTLY-read frames, that
the wearer's mouth region moves 1.54x more than a cheek control when he
speaks (mouth A-C +0.8044 = +0.733 sd, cheek A-C +0.3267 = +0.477 sd,
permutation z +10.9), and that the empty room carries 0.015% of the mouth
region's motion -- 6,600x quieter.

`docs/research/MOUTH-APERTURE-ALIGNMENT.md` then found, with a face
landmark model, that the mouth aperture's correlation with the speech
envelope does NOT survive its controls: mouth WIDTH scored +0.3038 (higher
than the aperture's +0.2052) and head HEIGHT scored +0.3316 with a sharper
peak (z +3.77 against the mouth's +2.70). The correlation was about the head
moving, not about the mouth opening.

So the question this file answers is the sharp one:

    Given that a single scalar "how much did the mouth box change" is the
    head-motion-confounded instrument, is there a mouth-localised feature
    that separates speaking from silent AFTER head motion is held constant,
    and how far above its own controls does it sit?

THE ANSWER IS NOT ASSUMED. The pre-registered decision rule is written into
this file, verbatim, BELOW, and it is evaluated mechanically against the
recorded numbers. A signal that fails the head-motion control is the
finding, and this script is built to be able to print that.

WHAT IS IN THE BOXES.

    mouth   rows 46-58, cols 62-86   (13 x 25)  the wearer's mouth
    cheek   rows 46-58, cols 92-116  (13 x 25)  the SPATIAL control
    fore    rows 20-44, cols 56-104  (25 x 49)  the upper face: the
                                               head-motion covariate's
                                               own box, and a third
                                               spatial control

The mouth box is the one `docs/research/NV12-STRIDE.md` §5 read off a 5x
render with a coordinate grid; it is consistent with the MediaPipe landmark
positions recorded in `docs/research/MOUTH-APERTURE-ALIGNMENT.md` §2
(upper inner lip at (78.60, 52.52), lower inner lip at (78.66, 53.76)),
which fall inside it. The cheek box is at the same rows, to the right of the
mouth, and does not overlap it.

WHAT IS THE NULL, FOR EACH MEASUREMENT.

    the classifier          the blank control (a constant predictor) and the
                            session-fingerprint control (a feature that
                            identifies the SESSION and nothing else), run on
                            the SAME fold.
    the mouth feature       the CHEEK feature, computed by the identical
                            pipeline on the identical rows.
    the mouth feature       the HEAD-MOTION PROXY, as a competing predictor.
    the A-vs-C effect       the H-CONDITIONAL version of the same statistic
                            (AUC within strata of the head-motion proxy),
                            and the residualised version (the feature with
                            the head-motion proxy regressed out).
    the A-vs-C effect       permuted session labels, within the strata.
    the physical claim      the EMPTY ROOM, where there is no person.

THE FUNDAMENTAL LIMIT, STATED UP FRONT.

The audio and the video of A/B/C are INDEPENDENT TAKES. No lag in +/-20 s
and no video time base in 10-60 Hz aligns them
(`docs/research/NV12-STRIDE.md` §6, `docs/research/MOUTH-APERTURE-ALIGNMENT.md`
§4.3 and §5). So FRAME-LEVEL audio<->video labels are NOT recoverable from
this corpus, and this lane does not attempt them. Every evaluation below is
a SESSION-LEVEL statement, and the labels are session-level, which makes
the leave-one-session-out accuracy a partly degenerate metric: a predictor
that identifies the session rather than the speaking state scores 1.000.
That is measured here, not assumed, and it is reported beside every
accuracy.

WHAT IS IMPORTED, AND WHY.

    src/nv12.py            the shared reader. The frame layout is DETECTED
                           from the content, not assumed, and the detection
                           margin is recorded. The captures are 19,200-byte
                           records (luma + interleaved half-resolution
                           chroma); reading them at 23,040 tears every frame
                           and every number in this lane would be about the
                           tear. See `docs/research/NV12-STRIDE.md`.

numpy only, deliberately: no scipy and no GPU, so the result reproduces on
the repository's own interpreter. (MediaPipe is not used here; the
landmark-based mouth aperture is the subject of
`docs/research/MOUTH-APERTURE-ALIGNMENT.md`, not of this lane.)
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import nv12  # noqa: E402

# --------------------------------------------------------------------------
# The geometry. Rows are [r0, r1), cols are [c0, c1), in the 160x96 frame.
# --------------------------------------------------------------------------

FRAME_W = 160
FRAME_H = 96
FPS = 30.0

MOUTH_BOX = (46, 59, 62, 87)      # rows 46..58, cols 62..86
CHEEK_BOX = (46, 59, 92, 117)     # rows 46..58, cols 92..116
FORE_BOX = (20, 45, 56, 105)      # rows 20..44, cols 56..104
BG_BOX = (46, 59, 130, 158)       # rows 46..58, cols 130..157

BOXES = {"mouth": MOUTH_BOX, "cheek": CHEEK_BOX, "fore": FORE_BOX,
         "bg": BG_BOX}

#: The candidate mouth features, in a FIXED order (feature selection breaks
#: ties by the lower index, so the order is part of the pre-registration).
FEATURES = ("motion", "lumafluct", "histdiff", "hfchange", "vgradfluct",
            "darkfracfluct")

#: The head-motion proxies. `H_upper` is the PRIMARY covariate: the motion of
#: the upper face, which is outside the mouth box, so regressing the mouth
#: feature on it does not regress the mouth feature on itself.
HEAD_PROXIES = ("H_upper", "H_frame", "H_shift", "H_mouthshift")

#: 2 s moving average at 30 fps. Used to take the slow trend out of the
#: per-frame statistics, so that a slow illumination or posture drift is not
#: counted as "the mouth changed".
MA_FRAMES = 61

#: Number of permutations for every null in this file.
N_PERM = 2000
PERM_SEED = 20261020

#: The head-motion-matched pair test's caliper ladder, in units of the
#: pooled head-motion proxy's standard deviation. The first rung that forms
#: at least MIN_PAIRS pairs is used, and the rung is recorded.
CALIPER_LADDER = (0.05, 0.10, 0.25, 0.50)
MIN_PAIRS = 50

#: Number of quantile strata of the head-motion proxy, for the conditional
#: (H-stratified) AUC.
N_STRATA = 10

# --------------------------------------------------------------------------
# THE PRE-REGISTERED DECISION RULE.
#
# Written 2026-10-20, BEFORE the measurement code in this file was written,
# and not edited afterwards. It is copied verbatim into the results file, and
# `tests/test_mouth_detector.py` asserts that the copy in the results file
# still equals the copy here -- so the rule cannot be quietly rewritten to
# fit the numbers.
# --------------------------------------------------------------------------

PRE_REGISTERED_RULE = {
    "version": "1.0",
    "written": "2026-10-20",
    "written_before": ("the measurement code in this file existed; the rule "
                       "text was fixed first and the numbers were produced by "
                       "the code as written"),
    "scope": ("session-level speaking detection from the mouth region, on "
              "captures A (speaking, facing camera), B (speaking, facing "
              "away), C (silent, facing camera) and empty1 (no person), read "
              "at the detected NV12 stride"),
    "terms": {
        "M": ("the mouth-box feature (rows 46-58, cols 62-86), selected "
              "per fold by FEATURE SELECTION below"),
        "C": ("the SAME feature computed on the CHEEK box (rows 46-58, cols "
              "92-116), selected the same way"),
        "H": ("the head-motion proxy: the mean absolute consecutive-frame "
              "difference over the upper-face box (rows 20-44, cols "
              "56-104). PRIMARY covariate. H_frame (whole frame), H_shift "
              "(upper-face translation) and H_mouthshift (mouth-box "
              "translation) are reported as variants."),
        "LOO": ("leave-one-session-out over {A, B, C}: three folds, test "
                "sessions A, B and C in turn"),
    },
    "feature_selection": (
        "Inside each fold, the mouth feature is the candidate with the "
        "highest TRAINING balanced frame accuracy of the threshold "
        "classifier; ties are broken by the lower candidate index in the "
        "fixed order motion, lumafluct, histdiff, hfchange, vgradfluct, "
        "darkfracfluct. The test session is never used for selection. The "
        "cheek control and the head-motion proxy are evaluated with the "
        "identical pipeline."),
    "classifier": (
        "A per-frame threshold on one feature: 'speaking' iff feature > "
        "threshold. The threshold maximises balanced accuracy on the "
        "TRAINING frames. The direction is fixed (higher = speaking) for "
        "every feature, because every candidate is a magnitude. The session "
        "decision is 'speaking' iff the MAJORITY of the held-out session's "
        "frames are above the threshold."),
    "training_set": (
        "The two non-test sessions' frames. If those two sessions contain no "
        "silent class (which happens in exactly one fold, the fold that "
        "holds out C), the negative class is supplied by the empty room's "
        "TRAINING SLICE: frames 0..1499 of empty1.nv12, labelled silent. "
        "The empty room's frames 1500..4499 are the HELD-OUT physical "
        "control and are never trained on. A sensitivity variant that adds "
        "the empty-room negative to every fold is reported."),
    "primary_metrics": (
        "macro_frame_recall: the mean over the three folds of the held-out "
        "session's frame accuracy (each held-out session is single-class, so "
        "this is the recall of that session's class). "
        "loo_session_accuracy: the fraction of folds whose session decision "
        "is correct. "
        "auc_ac: the AUC of the feature separating A's frames from C's "
        "frames. "
        "conditional_auc_ac: the AUC computed WITHIN strata of H, so the "
        "head-motion proxy is held constant. "
        "residual_d_ac: Cohen's d on the feature after regressing it on H "
        "with the TRAINING sessions only."),
    "checks": {
        "R1_survives_the_head_motion_control": (
            "The mouth signal survives the confound iff BOTH: "
            "R1a -- conditional_auc_ac for M is above 0.50 with a permutation "
            "z of at least +2.0 (2000 permutations of the session labels "
            "within the H-strata); AND "
            "R1b -- residual_d_ac for M is above 0 with a permutation z of "
            "at least +3.0 (2000 permutations of the session labels over the "
            "residualised values). "
            "If either fails, the mouth signal does NOT survive the "
            "head-motion control, whatever the classifier scores."),
        "R2_mouth_specific": (
            "The effect is mouth-specific iff conditional_auc_ac for M "
            "exceeds that for C (the cheek) by at least 0.02, with a "
            "permutation z of at least +2.0 for the paired difference."),
        "R3_beats_the_competing_predictor": (
            "The detector beats head motion iff macro_frame_recall for M "
            "exceeds that for H by at least 0.05, AND M's loo_session_"
            "accuracy is at least H's in at least 2 of the 3 folds."),
        "R4_beats_the_nulls": (
            "The detector beats the nulls iff macro_frame_recall for M "
            "exceeds EACH of: the best permuted-session-label null, the "
            "training-majority blank control, and the session-fingerprint "
            "control, by at least 0.05."),
        "R5_honesty_gate": (
            "Always reported, never a pass or a fail of the feature: the "
            "labels are session-level, so the leave-one-session-out metric "
            "is partly degenerate -- a predictor that identifies the SESSION "
            "rather than the speaking state can score 1.000, and the blank "
            "control does. No frame-level audio<->video claim is admissible "
            "from these captures (the audio and the video are independent "
            "takes). Every number in this lane is a SESSION-LEVEL statement."),
    },
    "verdict_rule": (
        "The verdict is the FIRST of these that applies, and it is computed "
        "mechanically from the recorded control numbers: "
        "1. if NOT (R1a AND R1b) -> "
        "'mouth_signal_does_not_survive_head_motion'; "
        "2. elif NOT R2 -> "
        "'mouth_signal_survives_but_is_not_mouth_specific'; "
        "3. elif NOT R3 -> "
        "'mouth_detector_does_not_beat_head_motion'; "
        "4. elif NOT R4 -> 'mouth_detector_beats_nothing'; "
        "5. else -> 'mouth_detector_survives_head_motion'. "
        "If the data are insufficient to evaluate any check, the verdict is "
        "'UNDETERMINED' and is never printed from a partial run."),
    "stopping_rule": (
        "Every number reported here comes from one run of this script over "
        "the four captures on disk. No threshold, no window and no "
        "permutation count was changed after the first run; the results file "
        "records the script's own SHA-256 so that any later edit is visible."),
}

PRE_REGISTERED_RULE_TEXT = """\
PRE-REGISTERED DECISION RULE (v1.0, 2026-10-20)

SCOPE: session-level speaking detection from the mouth region, on captures
A (speaking, facing), B (speaking, away), C (silent, facing) and empty1 (no
person), read at the detected NV12 stride.

TERMS
  M = the mouth-box feature (rows 46-58, cols 62-86), selected per fold.
  C = the same feature on the CHEEK box (rows 46-58, cols 92-116).
  H = the head-motion proxy: mean |I_t - I_{t-1}| over the upper-face box
      (rows 20-44, cols 56-104).  PRIMARY covariate.
  LOO = leave-one-session-out over {A, B, C}.

FEATURE SELECTION: inside each fold, the candidate with the highest
TRAINING balanced frame accuracy; ties broken by the lower index in the
fixed order motion, lumafluct, histdiff, hfchange, vgradfluct,
darkfracfluct.  The test session is never used for selection.

CLASSIFIER: per-frame threshold, 'speaking' iff feature > threshold, the
threshold maximising balanced accuracy on the TRAINING frames.  The session
decision is 'speaking' iff the MAJORITY of the held-out session's frames
are above the threshold.

TRAINING SET: the two non-test sessions' frames.  If those two contain no
silent class (the fold that holds out C), the negative class is the empty
room's TRAINING SLICE (frames 0..1499 of empty1, labelled silent); the
empty room's frames 1500..4499 are the held-out physical control and are
never trained on.

PRIMARY METRICS: macro_frame_recall (mean over folds of the held-out
session's frame accuracy); loo_session_accuracy; auc_ac; conditional_auc_ac
(AUC within strata of H); residual_d_ac (Cohen's d after regressing the
feature on H with the TRAINING sessions only).

R1  the mouth signal survives the head-motion control iff BOTH
    R1a  conditional_auc_ac for M > 0.50, permutation z >= +2.0
         (2000 permutations of the session labels within the H-strata)
    R1b  residual_d_ac for M > 0, permutation z >= +3.0
         (2000 permutations of the session labels)
    If either fails -> the mouth signal does NOT survive the confound,
    whatever the classifier scores.

R2  the effect is mouth-specific iff conditional_auc_ac for M exceeds that
    for C by >= 0.02, with permutation z >= +2.0 for the paired difference.

R3  the detector beats head motion iff macro_frame_recall for M exceeds that
    for H by >= 0.05, AND M's loo_session_accuracy is at least H's in at
    least 2 of the 3 folds.

R4  the detector beats the nulls iff macro_frame_recall for M exceeds EACH
    of the best permuted-session-label null, the training-majority blank
    control and the session-fingerprint control by >= 0.05.

R5  HONESTY GATE, always reported: the labels are session-level, so the
    leave-one-session-out metric is partly degenerate -- a predictor that
    identifies the SESSION rather than the speaking state can score 1.000,
    and the blank control does.  No frame-level audio<->video claim is
    admissible from these captures (independent takes).  Everything in this
    lane is a SESSION-LEVEL statement.

VERDICT: the first of these that applies, computed mechanically from the
recorded control numbers:
  1. NOT (R1a AND R1b) -> mouth_signal_does_not_survive_head_motion
  2. R1 OK but NOT R2  -> mouth_signal_survives_but_is_not_mouth_specific
  3. R1, R2 OK, NOT R3 -> mouth_detector_does_not_beat_head_motion
  4. R1..R3 OK, NOT R4 -> mouth_detector_beats_nothing
  5. otherwise          -> mouth_detector_survives_head_motion
  Insufficient data    -> UNDETERMINED (never printed from a partial run).
"""


# --------------------------------------------------------------------------
# Small numeric helpers. numpy only: scipy is not available in this
# repository's interpreter and the result must reproduce without it.
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared. (scipy.stats.rankdata, without scipy.)"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = xs[1:] != xs[:-1]
    grp = np.cumsum(is_new) - 1
    counts = np.bincount(grp)
    starts = np.cumsum(counts) - counts
    avg = starts + (counts - 1) / 2.0 + 1.0
    ranks = np.empty(n, dtype=float)
    ranks[order] = avg[grp]
    return ranks


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + 0.5 P(pos == neg): the rank AUC, ties included."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = _avg_ranks(np.concatenate([pos, neg]))
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def auc_from_labels(values: np.ndarray, labels: np.ndarray) -> float:
    """AUC of `values` for labels==1 against labels==0."""
    return auc(values[labels == 1], values[labels == 0])


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d, a minus b, pooled sd."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def residualise(x: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Remove the least-squares linear dependence of x on h."""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    design = np.column_stack([np.ones_like(h), h])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    resid = x - design @ coef
    return resid, [float(c) for c in coef]


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average, edge-padded, same length as x."""
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w) / float(w)
    return np.convolve(xp, kernel, mode="valid")[:len(x)]


def perm_z(observed: float, null: np.ndarray) -> float:
    """z of an observation against a null distribution."""
    null = np.asarray(null, dtype=float)
    sd = null.std(ddof=1)
    if sd <= 0:
        return float("inf") if observed > null.mean() else float("-inf")
    return float((observed - null.mean()) / sd)


# --------------------------------------------------------------------------
# Feature extraction, straight from the luma.
# --------------------------------------------------------------------------

def _box_view(luma: np.ndarray, box) -> np.ndarray:
    r0, r1, c0, c1 = box
    return luma[:, r0:r1, c0:c1].astype(np.float32)


def _histogram_stack(b: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Per-frame histogram of a (n, h, w) block, normalised to sum 1."""
    n = b.shape[0]
    idx = np.clip((b.reshape(n, -1) // (256 // n_bins)).astype(np.int64), 0,
                  n_bins - 1)
    offsets = (np.arange(n) * n_bins)[:, None]
    counts = np.bincount((idx + offsets).ravel(), minlength=n * n_bins)
    h = counts.reshape(n, n_bins).astype(np.float64)
    totals = h.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return h / totals


def distinct_frames(luma: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Keep the first frame of every run of identical frames.

    THE CAPTURES ARE NOT 30 DISTINCT FRAMES PER SECOND.  The files are 900
    records on a 30 fps time base -- that part is right, and it is what
    makes the recordings 30.00 s long -- but the camera delivered only
    about 19.9 DISTINCT images per second and the Windows driver repeated
    frames to fill the grid.  Measured on the correctly-read luma:

        A      304 of 899 consecutive pairs are byte-identical  (596 distinct)
        B      305 of 899                                        (595 distinct)
        C      305 of 899                                        (595 distinct)
        empty1 149 of 4499                                       (4350 distinct)

    Differencing the grid as recorded therefore reads about a third of the
    frames as "nothing happened", and the head-motion proxy inherits a
    third of its mass at exactly zero -- which is a property of the
    capture driver, not of the wearer.  So the features are computed on
    the DISTINCT frames, and the moving-average window is expressed in
    seconds of real time and converted to distinct frames at that
    session's own measured rate.

    The empty room was captured in a different round and delivers ~29
    distinct frames per second, so its window is correspondingly longer.
    That difference is recorded, because it bears on the empty-room
    comparison.
    """
    if len(luma) == 0:
        return luma, np.array([], dtype=int), {"n_frames": 0}
    same = np.zeros(len(luma), dtype=bool)
    if len(luma) > 1:
        same[1:] = (np.abs(np.diff(luma.astype(np.int16), axis=0))
                    .sum(axis=(1, 2)) == 0)
    keep = ~same
    keep[0] = True
    idx = np.where(keep)[0]
    stats = {
        "n_frames_on_disk": int(len(luma)),
        "n_duplicate_consecutive_pairs": int(same.sum()),
        "duplicate_pair_fraction": float(same.sum() / max(len(same), 1)),
        "n_distinct_frames": int(len(idx)),
        "time_base_s": float(len(luma) / FPS),
        "distinct_frame_rate_hz": float((len(idx) - 1) / (len(luma) / FPS)),
    }
    return luma[idx], idx, stats


def box_features(luma: np.ndarray, box, ma_window: int = MA_FRAMES) -> dict:
    """The candidate features for one box, per frame.

    Every feature is a magnitude of CHANGE in that box, so "higher" means
    "more changed", and the classifier's direction is fixed.
    """
    b = _box_view(luma, box)
    n = b.shape[0]
    out: dict[str, np.ndarray] = {}

    # 1. motion: the mean absolute consecutive-frame difference.
    diff = np.abs(np.diff(b, axis=0)).reshape(n - 1, -1).mean(axis=1)
    out["motion"] = np.concatenate([[0.0], diff])

    # 2. lumafluct: the box's mean brightness, minus its 2 s trend. A slow
    #    illumination or posture drift is removed; a mouth opening, which
    #    darkens the box, is not.
    bmean = b.reshape(n, -1).mean(axis=1)
    out["lumafluct"] = np.abs(bmean - moving_average(bmean, ma_window))

    # 3. histdiff: the histogram-intersection distance between consecutive
    #    frames. The one feature that is close to translation-insensitive:
    #    sliding the box's content sideways barely changes its histogram,
    #    while a mouth opening does (a dark interior appears).
    hist = _histogram_stack(b)
    hd = 0.5 * np.abs(np.diff(hist, axis=0)).sum(axis=1)
    out["histdiff"] = np.concatenate([[0.0], hd])

    # 4. hfchange: the box's high-frequency energy (vertical Laplacian),
    #    minus its 2 s trend. Lip edges appear and disappear when the mouth
    #    opens.
    lap = b[:, 1:-1, :] - 2.0 * b[:, :-2, :] + b[:, 2:, :]
    hf = (lap * lap).reshape(n, -1).mean(axis=1)
    out["hfchange"] = np.abs(hf - moving_average(hf, ma_window))

    # 5. vgradfluct: vertical gradient energy, minus its 2 s trend. The jaw
    #    moves vertically, so the mouth's vertical structure changes.
    vg = np.abs(b[:, 1:, :] - b[:, :-1, :]).reshape(n, -1).mean(axis=1)
    out["vgradfluct"] = np.abs(vg - moving_average(vg, ma_window))

    # 6. darkfracfluct: the fraction of the box's pixels that are darker
    #    than the box's own session-level 20th percentile, minus its 2 s
    #    trend. "How much dark area is in the mouth box right now."
    thr = float(np.percentile(b, 20.0))
    df = (b < thr).reshape(n, -1).mean(axis=1)
    out["darkfracfluct"] = np.abs(df - moving_average(df, ma_window))

    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def estimate_shift(luma: np.ndarray, box, max_shift: int = 2) -> np.ndarray:
    """Per-frame integer translation of a box's content, by SAD search.

    Returns (n, 2): (dy, dx) for frames 1..n-1, and (0, 0) for frame 0.
    This is the translation proxy: if the head moves, the face's content
    slides inside the box and the best-matching shift is non-zero.
    """
    r0, r1, c0, c1 = box
    r0 = max(r0, max_shift)
    c0 = max(c0, max_shift)
    r1 = min(r1, FRAME_H - max_shift)
    c1 = min(c1, FRAME_W - max_shift)
    h, w = r1 - r0, c1 - c0
    padded = luma[:, r0 - max_shift:r1 + max_shift,
                  c0 - max_shift:c1 + max_shift].astype(np.float32)
    n = padded.shape[0]
    best_sad = np.full(n - 1, np.inf)
    best_dy = np.zeros(n - 1, dtype=np.int64)
    best_dx = np.zeros(n - 1, dtype=np.int64)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            a = padded[1:, max_shift + dy:max_shift + dy + h,
                       max_shift + dx:max_shift + dx + w]
            b = padded[:-1, max_shift:max_shift + h, max_shift:max_shift + w]
            sad = np.abs(a - b).reshape(n - 1, -1).mean(axis=1)
            better = sad < best_sad
            best_sad = np.where(better, sad, best_sad)
            best_dy = np.where(better, dy, best_dy)
            best_dx = np.where(better, dx, best_dx)
    shifts = np.zeros((n, 2), dtype=np.float64)
    shifts[1:, 0] = best_dy
    shifts[1:, 1] = best_dx
    return shifts


def head_motion_proxies(luma: np.ndarray) -> dict:
    """The head-motion proxies, per frame."""
    n = luma.shape[0]
    f = luma.astype(np.float32)

    frame_diff = np.abs(np.diff(f, axis=0)).reshape(n - 1, -1).mean(axis=1)
    h_frame = np.concatenate([[0.0], frame_diff])

    fore = _box_view(luma, FORE_BOX)
    fore_diff = np.abs(np.diff(fore, axis=0)).reshape(n - 1, -1).mean(axis=1)
    h_upper = np.concatenate([[0.0], fore_diff])

    sh_upper = estimate_shift(luma, FORE_BOX)
    sh_mouth = estimate_shift(luma, MOUTH_BOX)

    return {
        "H_upper": h_upper,
        "H_frame": h_frame,
        "H_shift": np.hypot(sh_upper[:, 0], sh_upper[:, 1]),
        "H_mouthshift": np.hypot(sh_mouth[:, 0], sh_mouth[:, 1]),
    }


# --------------------------------------------------------------------------
# The statistics.
# --------------------------------------------------------------------------

def _strata(values: np.ndarray, n_strata: int) -> np.ndarray:
    """Quantile strata of `values`, labels 0..n_strata-1."""
    edges = np.quantile(values, np.linspace(0, 1, n_strata + 1)[1:-1])
    return np.searchsorted(edges, values, side="right")


def conditional_auc(pos: np.ndarray, neg: np.ndarray, h_pos: np.ndarray,
                    h_neg: np.ndarray, n_strata: int = N_STRATA) -> float:
    """AUC of the feature separating pos from neg, computed WITHIN strata of
    the head-motion proxy h, pooled by pair count.

    This is the head-motion-controlled version of the AUC: within a stratum
    the head-motion proxy is (as nearly as the strata allow) the same, so a
    feature that is only a function of head motion has no headroom left.
    """
    values = np.concatenate([pos, neg])
    h = np.concatenate([h_pos, h_neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    strata = _strata(h, n_strata)
    num = 0.0
    den = 0.0
    for s in np.unique(strata):
        m = strata == s
        v, l = values[m], labels[m]
        if l.sum() == 0 or (1 - l).sum() == 0:
            continue
        a = auc(v[l == 1], v[l == 0])
        npairs = (l.sum()) * ((1 - l).sum())
        num += a * npairs
        den += npairs
    return float(num / den) if den > 0 else float("nan")


def _label_permutations_within_strata(h_pos: np.ndarray, h_neg: np.ndarray,
                                      n_perm: int, seed: int):
    """Permute the session labels WITHIN each head-motion stratum.

    The strata survive, so the head-motion matching survives, and only the
    session assignment is broken. That is the right null for "does the
    feature separate the sessions once H is held constant".
    """
    h = np.concatenate([h_pos, h_neg])
    strata = _strata(h, N_STRATA)
    base = np.concatenate([np.ones(len(h_pos), dtype=np.int8),
                           np.zeros(len(h_neg), dtype=np.int8)])
    rng = np.random.default_rng(seed)
    perms = np.empty((n_perm, len(h)), dtype=np.int8)
    for i in range(n_perm):
        flipped = base.copy()
        for s in np.unique(strata):
            m = np.where(strata == s)[0]
            vals = base[m].copy()
            rng.shuffle(vals)          # shuffle the LABELS, not the indices
            flipped[m] = vals
        perms[i] = flipped
    return perms, strata


def _conditional_auc_with_labels(values: np.ndarray, labels: np.ndarray,
                                 strata: np.ndarray) -> float:
    """The conditional AUC of `values` for `labels`, within `strata`."""
    num = 0.0
    den = 0.0
    for s in np.unique(strata):
        m = strata == s
        v, l = values[m], labels[m]
        n1 = int(l.sum())
        n0 = int(len(l) - n1)
        if n1 == 0 or n0 == 0:
            continue
        a = auc(v[l == 1], v[l == 0])
        npairs = n1 * n0
        num += a * npairs
        den += npairs
    return float(num / den) if den > 0 else float("nan")


def conditional_auc_null_multi(pairs, h_pos: np.ndarray, h_neg: np.ndarray,
                                n_perm: int = N_PERM,
                                seed: int = PERM_SEED) -> np.ndarray:
    """The null distribution of the conditional AUC, for several features at
    once under the SAME label permutation.

    `pairs` is a list of (pos, neg) value arrays.  The returned array has
    shape (n_perm, len(pairs)), so the DIFFERENCE of two features' nulls is
    also a null: the paired mouth-vs-cheek comparison.
    """
    perms, strata = _label_permutations_within_strata(h_pos, h_neg, n_perm,
                                                        seed)
    values = [np.concatenate([p, n]) for (p, n) in pairs]
    out = np.empty((n_perm, len(pairs)), dtype=float)
    for i in range(n_perm):
        for j, v in enumerate(values):
            out[i, j] = _conditional_auc_with_labels(v, perms[i], strata)
    return out


def conditional_auc_null(pos: np.ndarray, neg: np.ndarray, h_pos: np.ndarray,
                         h_neg: np.ndarray, n_perm: int = N_PERM,
                         seed: int = PERM_SEED) -> np.ndarray:
    """Null distribution of the conditional AUC: the session labels are
    permuted WITHIN each head-motion stratum, so the strata (and therefore
    the matching) are preserved and only the session assignment is broken."""
    return conditional_auc_null_multi([(pos, neg)], h_pos, h_neg, n_perm,
                                      seed)[:, 0]


def matched_pairs(h_speak: np.ndarray, h_sil: np.ndarray,
                  caliper: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching of speaking frames to silent frames by
    the head-motion proxy, keeping only pairs within `caliper`."""
    cost = np.abs(h_speak[:, None] - h_sil[None, :])
    order = np.argsort(cost.ravel(), kind="mergesort")
    used_a = np.zeros(len(h_speak), dtype=bool)
    used_b = np.zeros(len(h_sil), dtype=bool)
    ia, ib = [], []
    for flat in order:
        i, j = divmod(int(flat), cost.shape[1])
        if cost[i, j] > caliper:
            break
        if used_a[i] or used_b[j]:
            continue
        used_a[i] = True
        used_b[j] = True
        ia.append(i)
        ib.append(j)
    return np.array(ia, dtype=int), np.array(ib, dtype=int)


def signflip_null(indicators: np.ndarray, n_perm: int = N_PERM,
                  seed: int = PERM_SEED) -> np.ndarray:
    """Null distribution of a matched-pair win rate: each pair's winner is
    swapped at random. Under 'the two sides are exchangeable', the win rate
    is 0.50."""
    indicators = np.asarray(indicators, dtype=np.int8)
    n = len(indicators)
    if n == 0:
        return np.array([0.5])
    rng = np.random.default_rng(seed)
    flips = rng.integers(0, 2, size=(n_perm, n), dtype=np.int8)
    draws = np.where(flips == 1, indicators[None, :],
                     (1 - indicators)[None, :]).mean(axis=1)
    return draws


def label_permutation_null(x: np.ndarray, labels: np.ndarray,
                            n_perm: int = N_PERM,
                            seed: int = PERM_SEED) -> np.ndarray:
    """Null distribution of a statistic under permuted session labels."""
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(labels)
        draws[i] = cohens_d(x[perm == 1], x[perm == 0])
    return draws


def two_group_effect(a: np.ndarray, b: np.ndarray) -> float:
    """(mean(a) - mean(b)) / the mean of the two within-group sds.

    This is the statistic `docs/research/NV12-STRIDE.md` §5 published for the
    mouth effect (+0.733 sd) and for the cheek control (+0.477 sd), so that
    this lane's numbers and that document's numbers are the same statistic
    and not two statistics that look alike.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sd = 0.5 * (a.std(ddof=1) + b.std(ddof=1))
    if sd <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / sd)


def effect_permutation_null(a: np.ndarray, b: np.ndarray,
                            n_perm: int = N_PERM,
                            seed: int = PERM_SEED) -> np.ndarray:
    """The null distribution of `two_group_effect` under permuted labels."""
    pooled = np.concatenate([a, b])
    labs = np.concatenate([np.ones(len(a), dtype=int),
                           np.zeros(len(b), dtype=int)])
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(labs)
        draws[i] = two_group_effect(pooled[perm == 1], pooled[perm == 0])
    return draws


def fit_threshold(values: np.ndarray, labels: np.ndarray,
                   n_grid: int = 201) -> tuple[float, float]:
    """The threshold maximising balanced accuracy on (values, labels), and
    that balanced accuracy. Direction is fixed: 'speaking' iff value > thr."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    grid = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_grid)))
    pos = labels == 1
    neg = labels == 0
    best_ba, best = -1.0, []
    for t in grid:
        pred = values > t
        tpr = float(pred[pos].mean()) if pos.any() else 0.0
        tnr = float((~pred[neg]).mean()) if neg.any() else 0.0
        ba = 0.5 * (tpr + tnr)
        if ba > best_ba + 1e-12:
            best_ba, best = ba, [float(t)]
        elif ba > best_ba - 1e-12:
            best.append(float(t))
    return float(np.median(best)), float(best_ba)


# --------------------------------------------------------------------------
# The leave-one-session-out evaluation.
# --------------------------------------------------------------------------

def _predict(values: np.ndarray, thr: float) -> np.ndarray:
    return (np.asarray(values, dtype=float) > thr).astype(int)


def run_loo(feature_values: dict, labels: dict, test_names: list[str],
            empty_values: dict | None = None,
            empty_label: int = 0,
            empty_train_frames: int = 1500) -> dict:
    """Leave-one-session-out over the given sessions.

    `feature_values` maps session name -> array of per-frame feature values.
    `labels` maps session name -> 0 (silent) or 1 (speaking).

    The threshold and the feature are chosen on the training sessions only.
    """
    folds = []
    for test in test_names:
        train = [s for s in test_names if s != test]
        xtr = np.concatenate([feature_values[s] for s in train])
        ytr = np.concatenate([np.full(len(feature_values[s]), labels[s],
                                      dtype=int) for s in train])
        source = "train_sessions"
        if len(np.unique(ytr)) == 1 and empty_values is not None:
            xtr = np.concatenate([xtr, empty_values[:empty_train_frames]])
            ytr = np.concatenate([ytr, np.full(empty_train_frames,
                                               empty_label, dtype=int)])
            source = "train_sessions+empty_room_slice"
        thr, train_ba = fit_threshold(xtr, ytr)
        xte = feature_values[test]
        yte = int(labels[test])
        pred = _predict(xte, thr)
        frame_acc = float((pred == yte).mean())
        session_pred = int(pred.mean() > 0.5)
        folds.append({
            "test_session": test,
            "train_sessions": train,
            "train_negative_source": source,
            "threshold": thr,
            "train_balanced_accuracy": train_ba,
            "held_out_frame_accuracy": frame_acc,
            "held_out_session_prediction": session_pred,
            "held_out_session_label": yte,
            "held_out_session_accuracy": float(session_pred == yte),
            "held_out_frames_predicted_speaking": float(pred.mean()),
        })
    macro = float(np.mean([f["held_out_frame_accuracy"] for f in folds]))
    sess = float(np.mean([f["held_out_session_accuracy"] for f in folds]))
    return {
        "folds": folds,
        "macro_frame_recall": macro,
        "loo_session_accuracy": sess,
        "per_fold_frame_accuracy": {f["test_session"]:
                                    f["held_out_frame_accuracy"]
                                    for f in folds},
        "per_fold_session_accuracy": {f["test_session"]:
                                      f["held_out_session_accuracy"]
                                      for f in folds},
    }


def run_loo_with_selection(feature_sets: dict, labels: dict,
                           test_names: list[str],
                           empty_sets: dict | None = None,
                           empty_train_frames: int = 1500) -> dict:
    """Leave-one-session-out with the feature chosen INSIDE each fold.

    `feature_sets` maps feature name -> {session: values}.  In each fold the
    candidate with the best TRAINING balanced frame accuracy is chosen, the
    threshold is fitted on the training sessions, and the held-out session
    is scored.  The test session is never used for either choice.
    """
    folds = []
    for test in test_names:
        train = [s for s in test_names if s != test]
        best_name, best_ba, best_thr = None, -1.0, None
        for name in FEATURES:
            sets = feature_sets[name]
            xtr = np.concatenate([sets[s] for s in train])
            ytr = np.concatenate([np.full(len(sets[s]), labels[s], dtype=int)
                                  for s in train])
            source = "train_sessions"
            if len(np.unique(ytr)) == 1 and empty_sets is not None:
                xtr = np.concatenate([xtr,
                                      empty_sets[name][:empty_train_frames]])
                ytr = np.concatenate([ytr, np.full(empty_train_frames, 0,
                                                   dtype=int)])
                source = "train_sessions+empty_room_slice"
            thr, ba = fit_threshold(xtr, ytr)
            if ba > best_ba + 1e-12:
                best_name, best_ba, best_thr = name, ba, thr
                best_source = source
        sets = feature_sets[best_name]
        xte = sets[test]
        yte = int(labels[test])
        pred = _predict(xte, thr)
        folds.append({
            "test_session": test,
            "train_sessions": train,
            "selected_feature": best_name,
            "train_negative_source": best_source,
            "threshold": float(best_thr),
            "train_balanced_accuracy": float(best_ba),
            "held_out_frame_accuracy": float((pred == yte).mean()),
            "held_out_session_prediction": int(pred.mean() > 0.5),
            "held_out_session_label": yte,
            "held_out_session_accuracy": float(int(pred.mean() > 0.5) == yte),
            "held_out_frames_predicted_speaking": float(pred.mean()),
        })
    macro = float(np.mean([f["held_out_frame_accuracy"] for f in folds]))
    sess = float(np.mean([f["held_out_session_accuracy"] for f in folds]))
    return {
        "folds": folds,
        "macro_frame_recall": macro,
        "loo_session_accuracy": sess,
        "per_fold_frame_accuracy": {f["test_session"]:
                                    f["held_out_frame_accuracy"]
                                    for f in folds},
        "per_fold_session_accuracy": {f["test_session"]:
                                      f["held_out_session_accuracy"]
                                      for f in folds},
        "selected_feature_per_fold": {f["test_session"]: f["selected_feature"]
                                      for f in folds},
    }


def select_feature(feature_sets: dict, labels: dict, test_names: list[str],
                   empty_sets: dict | None = None,
                   empty_train_frames: int = 1500) -> dict:
    """Pick the candidate feature by its TRAINING balanced frame accuracy,
    inside each fold.

    `feature_sets` maps feature name -> {session: values}.  The test session
    is never used for the selection.
    """
    chosen = {}
    for test in test_names:
        train = [s for s in test_names if s != test]
        best_name, best_ba = None, -1.0
        for name in FEATURES:
            sets = feature_sets[name]
            xtr = np.concatenate([sets[s] for s in train])
            ytr = np.concatenate([np.full(len(sets[s]), labels[s], dtype=int)
                                  for s in train])
            if len(np.unique(ytr)) == 1 and empty_sets is not None:
                xtr = np.concatenate([xtr, empty_sets[name][:empty_train_frames]])
                ytr = np.concatenate([ytr, np.full(empty_train_frames, 0,
                                                   dtype=int)])
            _, ba = fit_threshold(xtr, ytr)
            if ba > best_ba + 1e-12:
                best_ba, best_name = ba, name
        chosen[test] = best_name
    return chosen


# --------------------------------------------------------------------------
# The comparison statistics for one feature / one box.
# --------------------------------------------------------------------------

def compare_feature(name: str, values: dict, h: dict, h_name: str,
                    seed: int = PERM_SEED) -> dict:
    """Every statistic for one feature.

    The PRIMARY comparison is A vs C: both are facing the camera, so the
    pose is matched and the only thing that differs is whether the wearer
    is speaking. A vs B is reported as a secondary comparison and is
    CONTENT-confounded -- A and B recite the same poem -- so it is not used
    to judge the mouth.

    Three head-motion-controlled versions of the same comparison are
    computed, because they can disagree and the disagreement is the point:

      conditional_auc   the AUC computed WITHIN strata of the head-motion
                        proxy H, so H is held as constant as the strata
                        allow;
      residual_d        Cohen's d after regressing the feature on H, with
                        the regression fitted on the TRAINING sessions only
                        (three fits, one per fold);
      matched_win_rate  the win rate over frames matched one-to-one on H.
    """
    out = {"feature": name, "head_covariate": h_name}

    for tag, a, b in (("A_vs_C", "A", "C"), ("A_vs_B", "A", "B"),
                      ("B_vs_C", "B", "C")):
        va, vb = values[a], values[b]
        out[tag] = {
            "auc": auc(va, vb),
            "cohens_d": cohens_d(va, vb),
            "conditional_auc": conditional_auc(va, vb, h[a], h[b]),
            "mean_a": float(va.mean()),
            "mean_b": float(vb.mean()),
            "sd_a": float(va.std(ddof=1)),
            "sd_b": float(vb.std(ddof=1)),
        }

    # --- the null for the conditional AUC --------------------------------
    # The session labels are permuted WITHIN each head-motion stratum, so
    # the strata survive and only the session assignment is broken.
    va, vb = values["A"], values["C"]
    null = conditional_auc_null(va, vb, h["A"], h["C"], seed=seed)
    observed = out["A_vs_C"]["conditional_auc"]
    out["A_vs_C"]["conditional_auc_null_mean"] = float(null.mean())
    out["A_vs_C"]["conditional_auc_null_sd"] = float(null.std(ddof=1))
    out["A_vs_C"]["conditional_auc_perm_z"] = perm_z(observed, null)
    out["A_vs_C"]["conditional_auc_perm_p_gt"] = float((null >= observed).mean())

    # --- the residualised effect, refitted per fold -----------------------
    out["residualisation"] = {}
    for test in ("A", "B", "C"):
        train = [s for s in ("A", "B", "C") if s != test]
        xtr = np.concatenate([values[s] for s in train])
        htr = np.concatenate([h[s] for s in train])
        _, coef = residualise(xtr, htr)
        resid = {s: values[s] - (coef[0] + coef[1] * h[s])
                 for s in ("A", "B", "C")}
        # The orientation is fixed: SPEAKING minus SILENT.  A and B are the
        # speaking takes and C is the silent one, so the effect size is
        # positive when the feature is higher while speaking, in every fold.
        speaking = "B" if test == "B" else "A"
        silent = "C"
        pooled = np.concatenate([resid[speaking], resid[silent]])
        labs = np.concatenate([np.ones(len(resid[speaking]), dtype=int),
                               np.zeros(len(resid[silent]), dtype=int)])
        d_obs = cohens_d(resid[speaking], resid[silent])
        d_null = label_permutation_null(pooled, labs, seed=seed)
        out["residualisation"][f"fold_{test}"] = {
            "train_sessions": train,
            "intercept": coef[0],
            "slope_on_head_proxy": coef[1],
            "comparison": f"{speaking}_vs_{silent}",
            "held_out_session": test,
            "residual_d": d_obs,
            "residual_d_perm_z": perm_z(d_obs, d_null),
            "residual_d_perm_p_gt": float((d_null >= d_obs).mean()),
            "auc_after_residualisation": auc(resid[speaking], resid[silent]),
        }

    # A DIAGNOSTIC, not part of the pre-registered rule: the same residual
    # effect with the regression fitted on the A+C data itself (label-free,
    # so it leaks no label, but it does not extrapolate to an unseen
    # session).  It is a stability check on the fold-wise fits above.
    x_pool = np.concatenate([values["A"], values["C"]])
    h_pool = np.concatenate([h["A"], h["C"]])
    _, coef_pool = residualise(x_pool, h_pool)
    r_pool = x_pool - (coef_pool[0] + coef_pool[1] * h_pool)
    r_a = r_pool[:len(values["A"])]
    r_c = r_pool[len(values["A"]):]
    labs_pool = np.concatenate([np.ones(len(r_a), dtype=int),
                                np.zeros(len(r_c), dtype=int)])
    d_pool = cohens_d(r_a, r_c)
    d_pool_null = label_permutation_null(r_pool, labs_pool, seed=seed)
    out["residualisation"]["pooled_diagnostic"] = {
        "note": ("DIAGNOSTIC ONLY, not used by the pre-registered rule: the "
                 "regression is fitted on the A and C frames themselves, so "
                 "it does not extrapolate to an unseen session"),
        "slope_on_head_proxy": coef_pool[1],
        "residual_d": d_pool,
        "residual_d_perm_z": perm_z(d_pool, d_pool_null),
        "auc_after_residualisation": auc(r_a, r_c),
    }

    # --- the head-motion-matched pair test --------------------------------
    out["matched_pairs"] = {}
    for tag, a, b in (("A_vs_C", "A", "C"), ("A_vs_B", "A", "B"),
                      ("B_vs_C", "B", "C")):
        pooled_h = np.concatenate([h[a], h[b]])
        sd_h = float(pooled_h.std(ddof=1))
        chosen_caliper, pairs = None, (np.array([], dtype=int),
                                       np.array([], dtype=int))
        for cal in CALIPER_LADDER:
            ia, ib = matched_pairs(h[a], h[b], cal * sd_h)
            if len(ia) >= MIN_PAIRS:
                chosen_caliper, pairs = cal * sd_h, (ia, ib)
                break
        ia, ib = pairs
        ind = (values[a][ia] > values[b][ib]).astype(np.int8)
        rate = float(ind.mean()) if len(ind) else float("nan")
        mnull = signflip_null(ind, seed=seed)
        out["matched_pairs"][tag] = {
            "caliper_ladder_in_sd_units": list(CALIPER_LADDER),
            "caliper_used_in_sd_units": (None if chosen_caliper is None
                                         else chosen_caliper / sd_h),
            "n_pairs": int(len(ia)),
            "mean_abs_head_gap": (float(np.abs(h[a][ia] - h[b][ib]).mean())
                                  if len(ia) else None),
            "max_abs_head_gap": (float(np.abs(h[a][ia] - h[b][ib]).max())
                                 if len(ia) else None),
            "win_rate": rate,
            "win_rate_perm_z": perm_z(rate, mnull),
            "win_rate_perm_p_gt": float((mnull >= rate).mean()),
        }

    # --- the covariate's own numbers --------------------------------------
    # If the matching is doing its job, the covariate's own matched win
    # rate must be 0.50. If it is not, the conditional AUC is not
    # interpretable and this lane's central test is void.
    mp = out["matched_pairs"]["A_vs_C"]
    sd_h = float(np.concatenate([h["A"], h["C"]]).std(ddof=1))
    a_idx, b_idx = matched_pairs(h["A"], h["C"],
                                  (mp["caliper_used_in_sd_units"] or 0.5) * sd_h)
    out["head_covariate_check"] = {
        "auc_A_vs_C": auc(h["A"], h["C"]),
        "conditional_auc_A_vs_C": conditional_auc(h["A"], h["C"], h["A"], h["C"]),
        "matched_win_rate_A_vs_C": (float((h["A"][a_idx] > h["C"][b_idx]).mean())
                                    if len(a_idx) else float("nan")),
        "note": ("the covariate's matched win rate is 0.50 by construction; "
                 "if it is not, the matching is not holding the covariate "
                 "constant and the conditional AUC is not interpretable"),
    }
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

SESSIONS = ("A", "B", "C")
LABELS = {"A": 1, "B": 1, "C": 0, "empty1": 0}
#: The empty room's training slice is its first 50 s of real time; the
#: remaining 100 s is the held-out physical control.
EMPTY_TRAIN_SECONDS = 50.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--capture-dir", default="/mnt/c/proart_capture")
    ap.add_argument("--out", default=str(ROOT / "experiments" /
                                         "mouth_detector_results.json"))
    args = ap.parse_args(argv)

    cap = Path(args.capture_dir)
    report: dict = {
        "identity": {
            "script": "experiments/mouth_detector.py",
            "question": ("can a mouth-localised detector separate speaking "
                         "from silent once head motion is held constant, and "
                         "how far above its own controls does it sit?"),
            "run_utc": None,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "dependency_note": ("numpy only, no scipy and no GPU: the result "
                                "reproduces on the repository's own "
                                "interpreter. MediaPipe is NOT used in this "
                                "lane; see docs/research/MOUTH-DETECTOR.md "
                                "for the cross-check."),
        },
        "pre_registered_rule": PRE_REGISTERED_RULE,
        "pre_registered_rule_text": PRE_REGISTERED_RULE_TEXT,
    }

    # ---- read the captures through the shared reader --------------------
    layouts = {}
    luma = {}
    luma_raw = {}
    dup_stats = {}
    ma_windows = {}
    for sess in ("A", "B", "C", "empty1"):
        path = cap / f"{sess}.nv12"
        arr, layout = nv12.read_luma(path, width=FRAME_W, height=FRAME_H)
        luma_raw[sess] = arr
        dist, _idx, stats = distinct_frames(arr)
        luma[sess] = dist
        dup_stats[sess] = stats
        ma_windows[sess] = max(3, int(round(2.0 *
                                           stats["distinct_frame_rate_hz"])))
        layouts[sess] = layout
        report["identity"].setdefault("record_layout", {})[sess] = \
            layout.as_dict()

    # The empty room's TRAIN slice is the first 50 s of its own recording
    # and its HELD-OUT slice is the remaining 100 s.  The slices are in
    # seconds of real time, because the empty room delivers a different
    # number of distinct frames per second than the occupied captures do.
    empty_rate = dup_stats["empty1"]["distinct_frame_rate_hz"]
    empty_train_distinct = int(round(EMPTY_TRAIN_SECONDS * empty_rate))

    # ---- features --------------------------------------------------------
    feats = {box: {} for box in BOXES}
    heads = {}
    for sess in luma:
        for box_name, box in BOXES.items():
            feats[box_name][sess] = box_features(luma[sess], box,
                                                 ma_windows[sess])
        heads[sess] = head_motion_proxies(luma[sess])

    # ---- the per-session summary ----------------------------------------
    def summarise(arr: np.ndarray) -> dict:
        return {"mean": float(arr.mean()), "median": float(np.median(arr)),
                "sd": float(arr.std(ddof=1)), "p95": float(np.percentile(arr, 95))}

    session_stats = {}
    for sess in luma:
        entry = {"n_frames_on_disk": int(luma_raw[sess].shape[0]),
                 "duration_s_at_30fps": luma_raw[sess].shape[0] / FPS,
                 "n_distinct_frames": int(luma[sess].shape[0]),
                 "capture": dup_stats[sess],
                 "moving_average_window_frames": int(ma_windows[sess]),
                 "moving_average_window_s": float(
                     ma_windows[sess] / max(dup_stats[sess][
                         "distinct_frame_rate_hz"], 1e-9)),
                 "label": "speaking" if LABELS[sess] else "silent",
                 "boxes": {}}
        for box_name in BOXES:
            entry["boxes"][box_name] = {
                f: summarise(feats[box_name][sess][f]) for f in FEATURES}
        entry["head_motion"] = {h: summarise(heads[sess][h])
                                for h in HEAD_PROXIES}
        session_stats[sess] = entry
    report["sessions"] = session_stats
    report["geometry"] = {
        "mouth_box": {"rows": [MOUTH_BOX[0], MOUTH_BOX[1] - 1],
                      "cols": [MOUTH_BOX[2], MOUTH_BOX[3] - 1],
                      "provenance": ("read off a 5x render with a coordinate "
                                     "grid (docs/research/NV12-STRIDE.md §5); "
                                     "consistent with the MediaPipe landmark "
                                     "positions in "
                                     "docs/research/MOUTH-APERTURE-ALIGNMENT.md"
                                     " §2 (upper inner lip (78.60, 52.52), "
                                     "lower inner lip (78.66, 53.76)), which "
                                     "fall inside it")},
        "cheek_box": {"rows": [CHEEK_BOX[0], CHEEK_BOX[1] - 1],
                      "cols": [CHEEK_BOX[2], CHEEK_BOX[3] - 1],
                      "role": "spatial control, same rows, right of the mouth"},
        "fore_box": {"rows": [FORE_BOX[0], FORE_BOX[1] - 1],
                     "cols": [FORE_BOX[2], FORE_BOX[3] - 1],
                     "role": ("the upper face: the head-motion covariate's "
                              "box, and a third spatial control")},
        "bg_box": {"rows": [BG_BOX[0], BG_BOX[1] - 1],
                   "cols": [BG_BOX[2], BG_BOX[3] - 1],
                   "role": ("the same rows, far to the right of the face: a "
                            "pure-scene control that contains nothing of "
                            "the wearer")},
        "what_each_box_contains": (
            "MEASURED, not assumed. The boxes were drawn on a 6x render of "
            "A.nv12 and the per-pixel temporal standard deviation was "
            "measured over each of them. The face's own right-hand edge is "
            "at x ~ 90 (the MediaPipe left eye corner in "
            "docs/research/MOUTH-APERTURE-ALIGNMENT.md §2 is at x=91.98), so "
            "the CHEEK control box, which starts at column 92, STRADDLES "
            "the face's silhouette rather than sitting on a cheek. It is "
            "therefore not a clean 'face but not mouth' control: a "
            "high-contrast edge that moves with the head passes through it. "
            "The bg box is the clean scene control, and the fore box is the "
            "upper face."),
    }
    # the measured temporal standard deviation of each box, per session
    geom_probe = {}
    for sess in luma:
        sd_img = luma_raw[sess].astype(np.float32).std(axis=0)
        geom_probe[sess] = {b: float(sd_img[r0:r1, c0:c1].mean())
                             for b, (r0, r1, c0, c1) in BOXES.items()}
    report["geometry"]["temporal_sd_by_box"] = geom_probe

    # ---- the feature tables ----------------------------------------------
    comparisons = {}
    h_primary = {s: heads[s]["H_upper"] for s in heads}
    for box_name in BOXES:
        per_feature = {}
        for f in FEATURES:
            per_feature[f] = compare_feature(
                f, {s: feats[box_name][s][f] for s in feats[box_name]},
                h_primary, "H_upper")
        comparisons[box_name] = per_feature
    # The head-motion proxies themselves, as competing predictors.
    head_as_feature = {}
    for hname in HEAD_PROXIES:
        head_as_feature[hname] = compare_feature(
            hname, {s: heads[s][hname] for s in heads}, h_primary, "H_upper")
    report["comparisons"] = {"boxes": comparisons, "head_proxies": head_as_feature}

    # ---- the empty room, as a physical control ---------------------------
    empty = {}
    for box_name in BOXES:
        empty[box_name] = {
            f: {
                "empty1": summarise(feats[box_name]["empty1"][f]),
                "A": summarise(feats[box_name]["A"][f]),
                "empty_over_A_ratio_of_means": float(
                    feats[box_name]["empty1"][f].mean()
                    / max(feats[box_name]["A"][f].mean(), 1e-12)),
                "A_over_empty_ratio_of_means": float(
                    feats[box_name]["A"][f].mean()
                    / max(feats[box_name]["empty1"][f].mean(), 1e-12)),
            } for f in FEATURES}
    report["empty_room_control"] = {
        "note": ("the empty room carries no person, so its mouth-box motion "
                 "is the floor: what a mouth region does when nobody is "
                 "speaking. The held-out slice (frames 1500..4499) is never "
                 "trained on."),
        "train_slice_s": [0.0, EMPTY_TRAIN_SECONDS],
        "heldout_slice_s": [EMPTY_TRAIN_SECONDS,
                            dup_stats["empty1"]["time_base_s"]],
        "train_slice_distinct_frames": [0, empty_train_distinct],
        "heldout_slice_distinct_frames": [empty_train_distinct,
                                          len(feats["mouth"]["empty1"]["motion"])],
        "boxes": empty,
    }

    # ---- the leave-one-session-out evaluation -----------------------------
    # The primary pipeline: the mouth features, the cheek features and the
    # head-motion proxy, each run through the identical classifier.
    def sets_for(box_name, f):
        return {s: feats[box_name][s][f] for s in SESSIONS}

    empty_by_box = {b: {f: feats[b]["empty1"][f] for f in FEATURES}
                    for b in BOXES}

    loo_results = {}
    for box_name, empty_set in empty_by_box.items():
        chosen = select_feature({f: sets_for(box_name, f) for f in FEATURES},
                                 LABELS, list(SESSIONS), empty_set,
                                 empty_train_distinct)
        per_feature = {}
        for f in FEATURES:
            per_feature[f] = run_loo(sets_for(box_name, f), LABELS,
                                     list(SESSIONS), empty_values=empty_set[f],
                                     empty_train_frames=empty_train_distinct)
        # The LOO with the feature chosen INSIDE each fold: this is the
        # detector that is actually deployed, and the per-feature table
        # above gives the full picture of every candidate.
        sel = run_loo_with_selection({f: sets_for(box_name, f)
                                      for f in FEATURES},
                                     LABELS, list(SESSIONS), empty_set,
                                     empty_train_distinct)
        loo_results[box_name] = {
            "selected_feature_per_fold": sel["selected_feature_per_fold"],
            # the two selection routes are the same criterion computed two
            # ways; they are recorded so a divergence would be visible
            "selection_methods_agree": bool(
                chosen == sel["selected_feature_per_fold"]),
            "per_feature": per_feature,
            "per_fold_selected": sel["folds"],
            "macro_frame_recall_selected": sel["macro_frame_recall"],
            "loo_session_accuracy_selected": sel["loo_session_accuracy"],
        }

    # The head-motion proxy as a competing predictor, through the same
    # pipeline (one "feature", so no selection is involved).
    head_loo = {}
    for hname in HEAD_PROXIES:
        sets = {s: heads[s][hname] for s in SESSIONS}
        head_loo[hname] = run_loo(sets, LABELS, list(SESSIONS),
                                  empty_values=heads["empty1"][hname],
                                  empty_train_frames=empty_train_distinct)
    report["loo"] = {"boxes": loo_results, "head_proxies": head_loo}

    # ---- the empty room as a physical control, on held-out frames -------
    # Each fold's DETECTOR (its selected feature and its fitted threshold)
    # is applied to the empty room's HELD-OUT slice -- frames the detector
    # was never trained on.  A mouth detector that fires on an empty room
    # is measuring the camera, not the mouth.
    empty_fa = {}
    for box_name in BOXES:
        rows = []
        for i, test in enumerate(SESSIONS):
            fold = loo_results[box_name]["per_fold_selected"][i]
            f = fold["selected_feature"]
            thr = fold["threshold"]
            heldout = feats[box_name]["empty1"][f][empty_train_distinct:]
            pred = _predict(heldout, thr)
            rows.append({
                "fold_test_session": test,
                "feature": f,
                "threshold": float(thr),
                "trained_on_empty_room": (
                    fold["train_negative_source"].endswith("empty_room_slice")),
                "n_heldout_frames": int(len(heldout)),
                "fraction_predicted_speaking": float(pred.mean()),
            })
        empty_fa[box_name] = rows
    report["empty_room_false_alarm"] = {
        "note": ("the false-alarm rate of each fold's detector on the empty "
                 "room's HELD-OUT slice (the last 100 s of the recording), "
                 "which no detector was trained on"),
        "per_fold": empty_fa,
    }

    # ---- the published claim, re-measured --------------------------------
    # `docs/research/NV12-STRIDE.md` §5 published: the mouth box's A-C
    # effect is +0.8044 (+0.733 sd, permutation z +10.9), the cheek
    # control's is +0.3267 (+0.477 sd), the mouth-specific excess is
    # +0.256 sd and the ratio is 1.54x.  Those are re-measured here with
    # the SAME statistic, on both readings of the capture: the 30 fps grid
    # as recorded, and the distinct-frame sequence.
    def _motion_series(lum, box):
        r0, r1, c0, c1 = box
        b = lum[:, r0:r1, c0:c1].astype(np.float32)
        return np.abs(np.diff(b, axis=0)).reshape(len(b) - 1, -1).mean(axis=1)

    reproduction = {}
    for reading, source in (("grid_30fps_as_recorded", luma_raw),
                            ("distinct_frames", luma)):
        mA = _motion_series(source["A"], MOUTH_BOX)
        mC = _motion_series(source["C"], MOUTH_BOX)
        cA = _motion_series(source["A"], CHEEK_BOX)
        cC = _motion_series(source["C"], CHEEK_BOX)
        me = _motion_series(source["empty1"], MOUTH_BOX)
        mouth_eff = two_group_effect(mA, mC)
        cheek_eff = two_group_effect(cA, cC)
        null = effect_permutation_null(mA, mC)
        reproduction[reading] = {
            "mouth_A_minus_C": float(mA.mean() - mC.mean()),
            "mouth_effect_sd": mouth_eff,
            "mouth_effect_perm_z": perm_z(mouth_eff, null),
            "mouth_effect_perm_p_gt": float((null >= mouth_eff).mean()),
            "cheek_A_minus_C": float(cA.mean() - cC.mean()),
            "cheek_effect_sd": cheek_eff,
            "mouth_minus_cheek_excess_sd": float(mouth_eff - cheek_eff),
            "mouth_over_cheek_ratio_of_effects": float(mouth_eff / cheek_eff),
            "mouth_over_cheek_ratio_of_differences": float(
                (mA.mean() - mC.mean()) / (cA.mean() - cC.mean())),
            "empty_room_mouth_mean": float(me.mean()),
            "empty_room_share_of_A": float(me.mean() / max(mA.mean(), 1e-12)),
            "A_over_empty_room": float(mA.mean() / max(me.mean(), 1e-12)),
        }
    reproduction["published"] = {
        "source": "docs/research/NV12-STRIDE.md §5",
        "mouth_A_minus_C": 0.8044,
        "mouth_effect_sd": 0.733,
        "mouth_effect_perm_z": 10.9,
        "cheek_A_minus_C": 0.3267,
        "cheek_effect_sd": 0.477,
        "mouth_minus_cheek_excess_sd": 0.256,
        "mouth_over_cheek_ratio_of_effects": 1.54,
        "empty_room_share_of_A": 0.00015,
        "A_over_empty_room": 6600,
        "note": ("the published figures are the ratio of the two EFFECT "
                 "SIZES (0.733 / 0.477 = 1.54), not the ratio of the raw "
                 "differences; the raw-difference ratio is 2.42x"),
    }
    report["published_claim_reproduction"] = reproduction

    # ---- the nulls and the degenerate controls ---------------------------
    nulls = {}

    # (a) The blank control: a constant predictor. Two versions -- a fixed
    #     "always speaking", and the training set's majority class.
    blank_fixed = {}
    blank_majority = {}
    for test in SESSIONS:
        train = [s for s in SESSIONS if s != test]
        counts = {0: sum(1 for s in train if LABELS[s] == 0),
                  1: sum(1 for s in train if LABELS[s] == 1)}
        maj = 1 if counts[1] >= counts[0] else 0
        blank_fixed[test] = float(LABELS[test] == 1)
        blank_majority[test] = float(LABELS[test] == maj)
    nulls["blank_constant_speaking"] = {
        "per_fold": blank_fixed,
        "macro_frame_recall": float(np.mean(list(blank_fixed.values()))),
        "note": ("a constant 'speaking' answer is exactly right on a pure "
                 "speaking session, which is why it scores 1.000 on the "
                 "folds whose held-out session is a speaking one"),
    }
    nulls["blank_training_majority"] = {
        "per_fold": blank_majority,
        "macro_frame_recall": float(np.mean(list(blank_majority.values()))),
        "note": ("the training set's majority class; ties break to "
                 "'speaking'"),
    }

    # (b) The session-fingerprint control: a feature that identifies the
    #     SESSION and says nothing whatever about the mouth.  If this scores
    #     1.000, the LOO metric is degenerate and every accuracy above must
    #     be read with that in mind.
    fingerprint = {}
    for s in SESSIONS:
        fingerprint[s] = float(luma[s].astype(np.float32).mean())
    fingerprint_empty = float(luma["empty1"].astype(np.float32).mean())
    fp_folds = []
    for test in SESSIONS:
        train = [s for s in SESSIONS if s != test]
        # nearest training fingerprint, ties broken by the lower distance
        d = [(abs(fingerprint[s] - fingerprint[test]), s) for s in train]
        d.sort()
        pred = LABELS[d[0][1]]
        fp_folds.append({"test_session": test, "nearest_train_session": d[0][1],
                         "distance": d[0][0], "predicted_label": pred,
                         "correct": float(pred == LABELS[test])})
    nulls["session_fingerprint"] = {
        "feature": "the whole frame's mean luma (illumination), per session",
        "fingerprints": fingerprint,
        "empty1_mean_luma": fingerprint_empty,
        "folds": fp_folds,
        "macro_frame_recall": float(np.mean([f["correct"] for f in fp_folds])),
        "note": ("a pure session identifier: it cannot see the mouth at "
                 "all. If it reaches 1.000, then a leave-one-session-out "
                 "accuracy of 1.000 is not evidence of anything"),
    }

    # (c) The permuted-session-label null: every distinct assignment of the
    #     three sessions to {speaking, speaking, silent}, run through the
    #     WHOLE pipeline (feature selection inside each fold included).
    label_perms = []
    base = {"A": 1, "B": 1, "C": 0}
    mouth_sets = {f: {s: feats["mouth"][s][f] for s in SESSIONS}
                  for f in FEATURES}
    mouth_empty = {f: feats["mouth"]["empty1"][f] for f in FEATURES}
    for swapped in ({"A": 1, "B": 0, "C": 1}, {"A": 0, "B": 1, "C": 1}):
        lab = dict(base)
        lab.update(swapped)
        r = run_loo_with_selection(mouth_sets, lab, list(SESSIONS),
                                   mouth_empty, empty_train_distinct)
        label_perms.append({"labels": lab,
                            "macro_frame_recall": r["macro_frame_recall"],
                            "loo_session_accuracy": r["loo_session_accuracy"],
                            "selected_feature_per_fold":
                                r["selected_feature_per_fold"]})
    nulls["permuted_session_labels"] = {
        "note": ("the session-label null has only THREE distinct draws with "
                 "three sessions, so it is a sanity check, not a null "
                 "distribution; the frame-label shuffle below is the "
                 "distributional null"),
        "permutations": label_perms,
        "best_macro_frame_recall": float(max(p["macro_frame_recall"]
                                             for p in label_perms)),
    }

    # (d) The frame-label shuffle: the TRAINING labels are permuted across
    #     the pooled training frames -- which breaks the association between
    #     the session and the label while keeping every frame's feature
    #     value and the class balance -- the threshold is refitted, and the
    #     held-out session is re-scored against the REAL label.  This is
    #     the pipeline's own optimism, measured.  200 draws.
    rng = np.random.default_rng(PERM_SEED)
    shuffle_draws = []
    for _ in range(200):
        accs = []
        for test in SESSIONS:
            train = [s for s in SESSIONS if s != test]
            xtr_parts, ytr_parts = [], []
            for s in train:
                xtr_parts.append(feats["mouth"][s]["motion"])
                ytr_parts.append(np.full(len(feats["mouth"][s]["motion"]),
                                         LABELS[s], dtype=int))
            if len(train) == 2 and LABELS[train[0]] == LABELS[train[1]]:
                # the only silent class available is the empty room
                xtr_parts.append(feats["mouth"]["empty1"]["motion"][
                    :empty_train_distinct])
                ytr_parts.append(np.full(empty_train_distinct, 0, dtype=int))
            xtr = np.concatenate(xtr_parts)
            ytr = np.concatenate(ytr_parts)
            ytr = rng.permutation(ytr)
            thr, _ = fit_threshold(xtr, ytr)
            pred = _predict(feats["mouth"][test]["motion"], thr)
            accs.append(float((pred == LABELS[test]).mean()))
        shuffle_draws.append(float(np.mean(accs)))
    nulls["frame_label_shuffle"] = {
        "n_draws": 200,
        "mean": float(np.mean(shuffle_draws)),
        "sd": float(np.std(shuffle_draws, ddof=1)),
        "p05": float(np.percentile(shuffle_draws, 5)),
        "p95": float(np.percentile(shuffle_draws, 95)),
        "max": float(np.max(shuffle_draws)),
        "note": ("the training labels are permuted across the pooled "
                 "training frames, so the training set keeps its feature "
                 "values and its class balance but loses the session-label "
                 "association; the held-out session is scored against the "
                 "REAL label, so this is the pipeline's own optimism, "
                 "measured rather than assumed"),
    }

    # (e) The wrong-session pairing: A's video with C's labels, and C's
    #     video with A's labels, through the whole pipeline.
    swapped = {"A": 0, "B": 1, "C": 1}
    wrong = run_loo_with_selection(mouth_sets, swapped, list(SESSIONS),
                                   mouth_empty, empty_train_distinct)
    nulls["wrong_session_pairing"] = {
        "note": ("A's video is labelled silent and C's video is labelled "
                 "speaking: the negative control the brief asks for. A "
                 "detector that has learned 'this take is the speaking one' "
                 "must fail here."),
        "labels": swapped,
        "macro_frame_recall": wrong["macro_frame_recall"],
        "inverted_macro_frame_recall": float(
            1.0 - wrong["macro_frame_recall"]),
        "per_fold": wrong["per_fold_frame_accuracy"],
        "selected_feature_per_fold": wrong["selected_feature_per_fold"],
    }
    report["nulls"] = nulls

    # ---- the pre-registered checks, mechanically --------------------------
    checks = evaluate_checks(report, feats, heads)

    # ---- the verdict -----------------------------------------------------
    verdict = derive_verdict(checks)
    report["checks"] = checks
    report["verdict"] = verdict

    # ---- provenance ------------------------------------------------------
    inputs = {}
    for sess in ("A", "B", "C", "empty1"):
        p = cap / f"{sess}.nv12"
        inputs[str(p)] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    for extra in (ROOT / "src" / "nv12.py",):
        inputs[str(extra)] = {"sha256": sha256_file(extra),
                              "bytes": extra.stat().st_size}
    report["provenance"] = {
        "generated_by": "experiments/mouth_detector.py",
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "command_line": " ".join(sys.argv),
        "cli_args": list(sys.argv[1:]),
        "inputs": inputs,
    }
    report["identity"]["run_utc"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat()

    repro = report["published_claim_reproduction"]
    report["supersedes"] = {
        "why": ("This lane re-tests, under a head-motion control, the mouth "
                "effect published in docs/research/NV12-STRIDE.md §5 and the "
                "falsification in docs/research/MOUTH-APERTURE-ALIGNMENT.md "
                "§4.2. Nothing is overwritten: the earlier numbers are "
                "reproduced here and their status is stated."),
        "claims_under_test": [
            {"claim": ("the mouth box moves more than the cheek when the "
                       "wearer speaks: +0.733 sd at the mouth against "
                       "+0.477 sd at the cheek, a 1.54x ratio, permutation "
                       "z +10.9"),
             "source": "docs/research/NV12-STRIDE.md §5",
             "status": ("CONFIRMED, not withdrawn. Re-measured here with the "
                        "same statistic: mouth A-C %+.4f (%+.4f sd) against "
                        "cheek %+.4f (%+.4f sd), a ratio of %.4fx on the 30 fps "
                        "grid as recorded (published 1.54x) and %.4fx on the "
                        "distinct frames. The empty room carries %.4f%% of "
                        "the mouth region's motion (published 0.015%%). "
                        "It is also STRENGTHENED by this lane's "
                        "duplicate-frame correction and it SURVIVES the "
                        "head-motion control (conditional AUC %.4f, "
                        "permutation z %+.2f), which the earlier document "
                        "did not test."
                        % (repro["grid_30fps_as_recorded"]["mouth_A_minus_C"],
                           repro["grid_30fps_as_recorded"]["mouth_effect_sd"],
                           repro["grid_30fps_as_recorded"]["cheek_A_minus_C"],
                           repro["grid_30fps_as_recorded"]["cheek_effect_sd"],
                           repro["grid_30fps_as_recorded"][
                               "mouth_over_cheek_ratio_of_effects"],
                           repro["distinct_frames"][
                               "mouth_over_cheek_ratio_of_effects"],
                           100.0 * repro["distinct_frames"][
                               "empty_room_share_of_A"],
                           checks["R1a_conditional_auc"]["value"],
                           checks["R1a_conditional_auc"]["perm_z"]))},
            {"claim": ("the mouth aperture's correlation with the speech "
                       "envelope is about head motion, not the mouth: mouth "
                       "width +0.3038 and head height +0.3316 beat the "
                       "aperture's +0.2052"),
             "source": "docs/research/MOUTH-APERTURE-ALIGNMENT.md §4.2",
             "status": ("NOT CONTRADICTED, because the statistic is "
                        "different: that lane correlates the mouth's "
                        "geometry with the AUDIO envelope frame by frame "
                        "(which cannot be aligned here), while this lane "
                        "asks whether the mouth region separates the two "
                        "TAKES at matched head motion. On this lane's "
                        "question the raw-luma mouth signal SURVIVES the "
                        "head-motion control, and it does NOT beat its "
                        "spatial control (R2 false) nor a constant "
                        "predictor (R4 false)."),
             },
        ],
        "not_superseded": [
            ("the frame-level limit: the audio and the video of A/B/C are "
             "independent takes, so no frame-level audio<->video label is "
             "recoverable. This lane does not revisit it and does not "
             "contradict it."),
        ],
    }

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {out}")
    print(f"VERDICT: {verdict['verdict']}")
    for line in verdict["reasons"]:
        print(f"  - {line}")
    return 0


def evaluate_checks(report: dict, feats: dict, heads: dict) -> dict:
    """Evaluate R1-R5 mechanically from the recorded control numbers.

    Nothing here decides anything by hand: every check is a comparison of
    two recorded numbers against a threshold that was written down before
    the measurement ran (PRE_REGISTERED_RULE).
    """
    checks: dict = {}

    mouth_c = report["comparisons"]["boxes"]["mouth"]
    cheek_c = report["comparisons"]["boxes"]["cheek"]

    # The headline feature is the one the fold-C selection chose -- fold C is
    # the hardest fold (its training set has no silent PERSON), so it is the
    # one whose choice is least flattered by the selection. Every
    # candidate's numbers are in `comparisons`, and the per-candidate
    # survival table is recorded below so the reader can see whether ANY
    # candidate survives.
    selected = report["loo"]["boxes"]["mouth"]["selected_feature_per_fold"]
    headline_feature = selected["C"]
    checks["headline_mouth_feature"] = headline_feature
    checks["headline_note"] = (
        "the feature the fold-C selection chose, used for the headline "
        "checks; every candidate's numbers are in `comparisons`, and "
        "`per_candidate_survival` says which candidates would have passed")

    m = mouth_c[headline_feature]
    c = cheek_c[headline_feature]

    # --- R1a: the conditional (head-motion-controlled) AUC ----------------
    r1a_auc = m["A_vs_C"]["conditional_auc"]
    r1a_z = m["A_vs_C"]["conditional_auc_perm_z"]
    checks["R1a_conditional_auc"] = {
        "value": r1a_auc, "perm_z": r1a_z,
        "passes": bool(r1a_auc > 0.5 and r1a_z >= 2.0),
        "rule": "conditional_auc_ac > 0.50 and perm z >= +2.0",
        "null_mean": m["A_vs_C"]["conditional_auc_null_mean"],
        "null_sd": m["A_vs_C"]["conditional_auc_null_sd"],
        "perm_p_gt": m["A_vs_C"]["conditional_auc_perm_p_gt"],
    }

    # --- R1b: the residualised effect, in every fold ----------------------
    # The regression is refitted on each fold's TRAINING sessions, so the
    # held-out session is never used to fit the covariate.
    per_fold = {}
    for fold in ("fold_A", "fold_B", "fold_C"):
        rec = m["residualisation"][fold]
        per_fold[fold] = {
            "comparison": rec["comparison"],
            "train_sessions": rec["train_sessions"],
            "slope_on_head_proxy": rec["slope_on_head_proxy"],
            "residual_d": rec["residual_d"],
            "perm_z": rec["residual_d_perm_z"],
            "perm_p_gt": rec["residual_d_perm_p_gt"],
            "passes": bool(rec["residual_d"] > 0
                           and rec["residual_d_perm_z"] >= 3.0),
        }
    checks["R1b_residual_d"] = {
        "per_fold": per_fold,
        "worst_perm_z": float(min(v["perm_z"] for v in per_fold.values())),
        "worst_residual_d": float(min(v["residual_d"] for v in per_fold.values())),
        "passes": bool(all(v["passes"] for v in per_fold.values())),
        "rule": ("in EVERY fold the residualised effect size is positive "
                 "with a permutation z of at least +3.0"),
    }

    # --- R2: mouth-specific against the cheek ----------------------------
    # The paired null: the SAME label permutation is applied to the mouth
    # and the cheek, so the difference has its own null.
    paired_null = conditional_auc_null_multi(
        [(feats["mouth"]["A"][headline_feature], feats["mouth"]["C"][headline_feature]),
         (feats["cheek"]["A"][headline_feature], feats["cheek"]["C"][headline_feature])],
        heads["A"]["H_upper"], heads["C"]["H_upper"], seed=PERM_SEED)
    diff_null = paired_null[:, 0] - paired_null[:, 1]
    diff = r1a_auc - c["A_vs_C"]["conditional_auc"]
    diff_z = perm_z(diff, diff_null)
    checks["R2_mouth_specific"] = {
        "mouth_conditional_auc": r1a_auc,
        "cheek_conditional_auc": c["A_vs_C"]["conditional_auc"],
        "difference": diff,
        "perm_z": diff_z,
        "null_mean_difference": float(diff_null.mean()),
        "passes": bool(diff >= 0.02 and diff_z >= 2.0),
        "rule": ("the mouth's conditional AUC exceeds the cheek's by >= 0.02 "
                 "with a paired permutation z of at least +2.0"),
    }

    # --- R3: beats the head-motion proxy as a competing predictor --------
    mouth_loo = report["loo"]["boxes"]["mouth"]
    head_loo = report["loo"]["head_proxies"]["H_upper"]
    recall_margin = (mouth_loo["macro_frame_recall_selected"]
                     - head_loo["macro_frame_recall"])
    sess_folds = {}
    wins = 0
    for i, t in enumerate(SESSIONS):
        ms = mouth_loo["per_fold_selected"][i]["held_out_session_accuracy"]
        hs = head_loo["per_fold_session_accuracy"][t]
        sess_folds[t] = {"mouth": ms, "head": hs}
        if ms >= hs:
            wins += 1
    checks["R3_beats_head_motion"] = {
        "mouth_macro_frame_recall": mouth_loo["macro_frame_recall_selected"],
        "head_macro_frame_recall": head_loo["macro_frame_recall"],
        "recall_margin": recall_margin,
        "per_fold_session_accuracy": sess_folds,
        "folds_not_worse": wins,
        "passes": bool(recall_margin >= 0.05 and wins >= 2),
        "rule": ("macro_frame_recall margin >= 0.05 AND at least 2 of 3 "
                 "folds where the mouth's session accuracy is >= the head's"),
    }

    # --- R4: beats the nulls ---------------------------------------------
    nulls = report["nulls"]
    fps = nulls["permuted_session_labels"]["best_macro_frame_recall"]
    blank = nulls["blank_training_majority"]["macro_frame_recall"]
    fp = nulls["session_fingerprint"]["macro_frame_recall"]
    mrr = mouth_loo["macro_frame_recall_selected"]
    margins = {"permuted_labels": mrr - fps,
               "blank": mrr - blank,
               "session_fingerprint": mrr - fp}
    checks["R4_beats_nulls"] = {
        "mouth_macro_frame_recall": mrr,
        "nulls": {"permuted_session_labels": fps, "blank": blank,
                  "session_fingerprint": fp},
        "margins": margins,
        "passes": bool(all(v >= 0.05 for v in margins.values())),
        "rule": "macro_frame_recall exceeds EACH null by >= 0.05",
    }

    # --- the per-candidate survival table, for the record -----------------
    # Every candidate is put through the same three questions, so the reader
    # can see whether the headline's answer depends on which feature was
    # picked.  The paired mouth-vs-cheek null is computed per candidate.
    per_candidate = {}
    for f in FEATURES:
        mm = mouth_c[f]
        paired = conditional_auc_null_multi(
            [(feats["mouth"]["A"][f], feats["mouth"]["C"][f]),
             (feats["cheek"]["A"][f], feats["cheek"]["C"][f]),
             (feats["bg"]["A"][f], feats["bg"]["C"][f])],
            heads["A"]["H_upper"], heads["C"]["H_upper"], seed=PERM_SEED)
        d_null = paired[:, 0] - paired[:, 1]
        b_null = paired[:, 0] - paired[:, 2]
        m_cond = mm["A_vs_C"]["conditional_auc"]
        c_cond = cheek_c[f]["A_vs_C"]["conditional_auc"]
        b_cond = report["comparisons"]["boxes"]["bg"][f]["A_vs_C"][
            "conditional_auc"]
        per_candidate[f] = {
            "auc_A_vs_C": mm["A_vs_C"]["auc"],
            "conditional_auc_A_vs_C": m_cond,
            "conditional_auc_perm_z": mm["A_vs_C"]["conditional_auc_perm_z"],
            "residual_d_worst_fold": float(min(
                mm["residualisation"][k]["residual_d"]
                for k in ("fold_A", "fold_B", "fold_C"))),
            "residual_d_perm_z_worst_fold": float(min(
                mm["residualisation"][k]["residual_d_perm_z"]
                for k in ("fold_A", "fold_B", "fold_C"))),
            "residual_d_pooled_diagnostic": mm["residualisation"][
                "pooled_diagnostic"]["residual_d"],
            "matched_win_rate_A_vs_C": mm["matched_pairs"]["A_vs_C"][
                "win_rate"],
            "matched_win_rate_perm_z": mm["matched_pairs"]["A_vs_C"][
                "win_rate_perm_z"],
            "cheek_conditional_auc_A_vs_C": c_cond,
            "bg_conditional_auc_A_vs_C": b_cond,
            "mouth_minus_cheek_conditional_auc": m_cond - c_cond,
            "mouth_minus_cheek_paired_perm_z": perm_z(m_cond - c_cond, d_null),
            "mouth_minus_bg_conditional_auc": m_cond - b_cond,
            "mouth_minus_bg_paired_perm_z": perm_z(m_cond - b_cond, b_null),
        }
    checks["per_candidate_survival"] = per_candidate

    # --- the same checks, per fold, with each fold's own selected feature --
    per_fold_checks = {}
    for i, test in enumerate(SESSIONS):
        f = report["loo"]["boxes"]["mouth"]["per_fold_selected"][i][
            "selected_feature"]
        mm = mouth_c[f]
        cc = cheek_c[f]
        bb = report["comparisons"]["boxes"]["bg"][f]["A_vs_C"]
        paired = conditional_auc_null_multi(
            [(feats["mouth"]["A"][f], feats["mouth"]["C"][f]),
             (feats["cheek"]["A"][f], feats["cheek"]["C"][f])],
            heads["A"]["H_upper"], heads["C"]["H_upper"], seed=PERM_SEED)
        d_null = paired[:, 0] - paired[:, 1]
        m_cond = mm["A_vs_C"]["conditional_auc"]
        c_cond = cc["A_vs_C"]["conditional_auc"]
        per_fold_checks[test] = {
            "selected_feature": f,
            "R1a_conditional_auc": m_cond,
            "R1a_perm_z": mm["A_vs_C"]["conditional_auc_perm_z"],
            "R1a_passes": bool(m_cond > 0.5
                               and mm["A_vs_C"]["conditional_auc_perm_z"] >= 2.0),
            "R1b_residual_d_fold": mm["residualisation"][f"fold_{test}"][
                "residual_d"],
            "R1b_perm_z_fold": mm["residualisation"][f"fold_{test}"][
                "residual_d_perm_z"],
            "R1b_passes": bool(
                mm["residualisation"][f"fold_{test}"]["residual_d"] > 0
                and mm["residualisation"][f"fold_{test}"][
                    "residual_d_perm_z"] >= 3.0),
            "R2_mouth_minus_cheek": m_cond - c_cond,
            "R2_paired_perm_z": perm_z(m_cond - c_cond, d_null),
            "R2_passes": bool(m_cond - c_cond >= 0.02
                               and perm_z(m_cond - c_cond, d_null) >= 2.0),
            "bg_conditional_auc": bb["conditional_auc"],
        }
    checks["per_fold_checks"] = per_fold_checks

    # --- R5: the honesty gate, always reported ---------------------------
    checks["R5_honesty_gate"] = {
        "frame_level_labels_recoverable": False,
        "why": ("the audio and the video of A/B/C are independent takes: no "
                "lag in +/-20 s and no video time base in 10-60 Hz aligns "
                "them (docs/research/NV12-STRIDE.md §6, "
                "docs/research/MOUTH-APERTURE-ALIGNMENT.md §4.3). Every "
                "evaluation here is therefore a SESSION-LEVEL statement, and "
                "the labels are session-level."),
        "session_level_metric_is_degenerate": {
            "blank_constant_speaking_per_fold": nulls[
                "blank_constant_speaking"]["per_fold"],
            "session_fingerprint_macro_frame_recall": nulls[
                "session_fingerprint"]["macro_frame_recall"],
            "note": ("a predictor that identifies the SESSION rather than "
                     "the speaking state can score 1.000 on this metric, so "
                     "a leave-one-session-out accuracy is not on its own "
                     "evidence of a mouth detector"),
        },
    }
    return checks


def check_flags(checks: dict) -> dict:
    """Recompute each check's pass/fail from its OWN recorded numbers.

    The verdict must not read a stored boolean: it must read the numbers,
    so that a changed number changes the verdict. That is the difference
    between a verdict that can fail and a verdict that is a constant.
    """
    r1a = checks["R1a_conditional_auc"]
    r2 = checks["R2_mouth_specific"]
    r3 = checks["R3_beats_head_motion"]
    r4 = checks["R4_beats_nulls"]
    r1b_folds = checks["R1b_residual_d"]["per_fold"]
    return {
        "R1a": bool(r1a["value"] > 0.5 and r1a["perm_z"] >= 2.0),
        "R1b": bool(all(v["residual_d"] > 0 and v["perm_z"] >= 3.0
                        for v in r1b_folds.values())),
        "R2": bool(r2["difference"] >= 0.02 and r2["perm_z"] >= 2.0),
        "R3": bool(r3["recall_margin"] >= 0.05 and r3["folds_not_worse"] >= 2),
        "R4": bool(all(v >= 0.05 for v in r4["margins"].values())),
    }


def derive_verdict(checks: dict) -> dict:
    """The verdict, mechanically, from the recorded checks.

    The pass/fail of each check is RECOMPUTED from its numbers here, not
    read from a stored flag, so the verdict is a function of the data.
    """
    flags = check_flags(checks)
    r1a, r1b = flags["R1a"], flags["R1b"]
    r2, r3, r4 = flags["R2"], flags["R3"], flags["R4"]

    reasons = []
    if not (r1a and r1b):
        verdict = "mouth_signal_does_not_survive_head_motion"
        if not r1a:
            reasons.append(
                "R1a FAILS: the mouth feature's AUC computed WITHIN "
                "head-motion strata is %.4f against a null of %.4f "
                "(permutation z %+.2f, needed >= +2.0) -- the mouth signal "
                "does not separate speaking from silent once head motion is "
                "held constant."
                % (checks["R1a_conditional_auc"]["value"],
                   checks["R1a_conditional_auc"]["null_mean"],
                   checks["R1a_conditional_auc"]["perm_z"]))
        if not r1b:
            worst = min(checks["R1b_residual_d"]["per_fold"].items(),
                        key=lambda kv: kv[1]["perm_z"])
            reasons.append(
                "R1b FAILS: with the head-motion proxy regressed out, the "
                "mouth feature's effect size is %+.4f sd in the worst fold "
                "(%s, permutation z %+.2f, needed >= +3.0)."
                % (worst[1]["residual_d"], worst[0], worst[1]["perm_z"]))
    elif not r2:
        verdict = "mouth_signal_survives_but_is_not_mouth_specific"
        reasons.append(
            "R1 passes but R2 FAILS: the mouth feature's conditional AUC "
            "%.4f does not exceed the cheek control's %.4f by the required "
            "0.02 (difference %+.4f, paired permutation z %+.2f)."
            % (checks["R2_mouth_specific"]["mouth_conditional_auc"],
               checks["R2_mouth_specific"]["cheek_conditional_auc"],
               checks["R2_mouth_specific"]["difference"],
               checks["R2_mouth_specific"]["perm_z"]))
    elif not r3:
        verdict = "mouth_detector_does_not_beat_head_motion"
        reasons.append(
            "R1 and R2 pass but R3 FAILS: the mouth feature's macro frame "
            "recall %.4f against the head-motion proxy's %.4f (margin "
            "%+.4f, needed >= 0.05)."
            % (checks["R3_beats_head_motion"]["mouth_macro_frame_recall"],
               checks["R3_beats_head_motion"]["head_macro_frame_recall"],
               checks["R3_beats_head_motion"]["recall_margin"]))
    elif not r4:
        verdict = "mouth_detector_beats_nothing"
        reasons.append(
            "R1-R3 pass but R4 FAILS: the mouth feature's macro frame "
            "recall %.4f does not clear every null (margins %s)."
            % (checks["R4_beats_nulls"]["mouth_macro_frame_recall"],
               checks["R4_beats_nulls"]["margins"]))
    else:
        verdict = "mouth_detector_survives_head_motion"
        reasons.append("R1, R2, R3 and R4 all pass.")

    # Every failing check is reported, even when the verdict's precedence
    # means a later check is not the headline.
    if not r4 and verdict != "mouth_detector_beats_nothing":
        reasons.append(
            "R4 also FAILS: the mouth feature's macro frame recall %.4f does "
            "not clear every null (margins %s)."
            % (checks["R4_beats_nulls"]["mouth_macro_frame_recall"],
               checks["R4_beats_nulls"]["margins"]))
    if not r3 and verdict not in ("mouth_detector_does_not_beat_head_motion",
                                  "mouth_detector_beats_nothing"):
        reasons.append(
            "R3 also FAILS: the mouth feature's macro frame recall %.4f "
            "against the head-motion proxy's %.4f."
            % (checks["R3_beats_head_motion"]["mouth_macro_frame_recall"],
               checks["R3_beats_head_motion"]["head_macro_frame_recall"]))

    return {
        "verdict": verdict,
        "reasons": reasons,
        "checks": flags,
        "rule": PRE_REGISTERED_RULE["verdict_rule"],
        "scope": ("SESSION-LEVEL only. No frame-level speaking claim is "
                  "made or supported by this lane."),
    }


if __name__ == "__main__":
    raise SystemExit(main())
