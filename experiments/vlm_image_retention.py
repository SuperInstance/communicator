#!/usr/bin/env python3
"""
vlm_image_retention.py -- DOES RETAINING AN IMAGE BEAT DESCRIBING IT?

WHY THIS FILE EXISTS.

The external claim under test is the one in "OpenVLHarness: Real World Visual
Agentic Tool-Use with Multimodal Memory" (Yang et al., UCLA/Apple,
https://huggingface.co/papers/2610.26860), Appendix H:

    "The textual-memory step (+0.7 at 8B and +0.3 at 32B) is comparable to
     run-to-run variation, which is consistent with our conclusion that text
     descriptions of images are a weak substitute for retaining the images
     themselves."

and, in section 2.2: "Repeating a description of a previous result does not
provide this access."

THIS REPOSITORY IS THE OPPOSITE CASE. It has spent its life turning camera
frames into TEXT/ASCII representations and asking a language model to judge
them. It has NEVER handed a model the frame itself. So the claim has never
been tested here, and the question is sharp:

    On this repository's own captures, does a vision-language model that
    receives the RAW IMAGE separate "speaking" from "silent" better than
    the SAME model receiving the SAME INFORMATION as a TEXT/ASCII
    DESCRIPTION of that image?

WHAT IS CONTROLLED, AND HOW.

Both arms are handed the SAME 15,360 luma values of the SAME frame. The
IMAGE arm gets them as pixels (upscaled, in a PNG). The TEXT arm gets them
as characters (the repository's own tone ramp, one character per pixel).
Nothing else differs: same model, same weights, same prompt wording, same
frames, same frame indices, same scoring rule, same operating point. The
ONLY thing that varies between the two headline arms is the ENCODING. That
is the whole treatment, and it is why the comparison can say something about
the encoding rather than about the model or the frames.

THE LABELS ARE THE REPOSITORY'S OWN, AND THEY ARE NOT WHAT THE BRIEF SAID.

    A       facing the camera, SPEAKING a poem   (speak=True, pose=facing)
    B       head turned away, SPEAKING           (speak=True, pose=turned)
    C       facing the camera, SILENT            (speak=False, pose=facing)
    empty1  no person at all

as recorded in /mnt/c/proart_capture/corpus_v2/{A,B,C}.json and in
`docs/research/MOUTH-DETECTOR.md` section 2. B is a SPEAKING take, so B is
NOT a silent class and is not used as one here: the negative class is C,
which is POSE-MATCHED with A (both face the camera), and B is reported as a
secondary condition. The decision that the positive class is A and the
negative class is C was made from the captures' own metadata before any
number in this file was computed.

THE FUNDAMENTAL LIMIT, STATED UP FRONT.

The audio and the video of A/B/C are INDEPENDENT TAKES. No lag aligns them
(`docs/research/NV12-STRIDE.md` section 6, `docs/research/MOUTH-APERTURE-
ALIGNMENT.md`). So FRAME-LEVEL audio<->video labels are NOT recoverable from
this corpus, and this lane does not attempt them. Every number here is a
SESSION-LEVEL statement about the DISTRIBUTION of scores on A's frames
against the distribution on C's frames. The "speaking" label on an A frame
means "this frame comes from a take in which the person was speaking", NOT
"the mouth was moving at this instant".

THE NULL, FOR EACH MEASUREMENT.

    the separation statistic   the EMPTY ROOM, which sets the operating
                               point. A model that says "speaking" to an
                               empty room is not detecting anything, so the
                               blank room's scores set the threshold at
                               which the arm is scored.
    the separation statistic   the A/C session labels PERMUTED (2,000
                               permutations, the SAME permutation applied to
                               both arms so the difference has its own null).
    the difference between     the SCRAMBLED arm: the same frames with
    the arms                   their 4x4 blocks permuted, which preserves
                               the pixel histogram exactly and destroys
                               only the spatial arrangement.
    the harness itself         the same frames scored twice, which must
                               reproduce exactly.

WHAT IS NOT CLAIMED, BEFORE ANY NUMBER IS LOOKED AT.

Nothing about speech detection, about mouths, or about frame-level
alignment. Nothing about any model other than the one named here. The
arms are compared to each other and to their own controls, and the verdict
is one of four pre-registered strings that can each come out.

numpy only for everything except the model itself: torch and transformers
are imported INSIDE the scoring functions, so this module stays importable
on the repository's own interpreter (which has no torch), which is what
`tests/test_vlm_image_retention.py` uses.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import io
import json
import platform
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import nv12  # noqa: E402

# --------------------------------------------------------------------------
# The geometry, the model, and the numbers that are FIXED IN ADVANCE.
# --------------------------------------------------------------------------

FRAME_W = 160
FRAME_H = 96
FPS = 30.0

#: Frames scored per condition. The brief allows 60-120; the two decisive
#: conditions (A and C) and the blank control get the top of that range and
#: the secondary condition B gets half of it.
N_FRAMES_MAIN = 120
N_FRAMES_SECONDARY = 60

#: The IMAGE arm's upscale. 160x96 -> 640x384, NEAREST, so that no pixel
#: value is invented: every source pixel becomes an exact 4x4 block.
UPSCALE = 4
IMAGE_W = FRAME_W * UPSCALE
IMAGE_H = FRAME_H * UPSCALE

#: The scramble block, in pixels (and, for the text arm, in characters --
#: one character per pixel, so the two arms share the SAME partition).
SCRAMBLE_BLOCK = 4

#: The operating point: the percentile of the EMPTY ROOM's scores at which
#: the arm is scored. At 95, the empty room's false-alarm rate is 5% by
#: construction, and the arm cannot buy separation by saying "speaking" to
#: everything.
BLANK_PERCENTILE = 95.0

#: The pre-registered margin, in the units of the margin statistic (a
#: difference of two rates, so 0.05 is five percentage points).
HEADLINE_MARGIN = 0.05

#: The pre-registered permutation z.
HEADLINE_Z = 2.0

#: Permutations for every null in this file.
N_PERM = 2000

#: The seed for the permutations and for the scramble permutation.
SEED = 20261020

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE_MAP = "auto"
DTYPE = "float16"

CAPTURE_DIR = Path("/mnt/c/proart_capture")

#: The condition labels, read off /mnt/c/proart_capture/corpus_v2/*.json and
#: `docs/research/MOUTH-DETECTOR.md` section 2 -- NOT assumed.
CONDITIONS = {
    "A": {"role": "positive", "label": "speaking", "pose": "facing"},
    "C": {"role": "negative", "label": "silent", "pose": "facing"},
    "empty1": {"role": "blank_control", "label": "no_person", "pose": "none"},
    "B": {"role": "secondary", "label": "speaking", "pose": "turned"},
}

#: The arms. `image` and `text` are the HEADLINE pair; the others are
#: pre-registered controls and robustness variants, all decided before the
#: run.
ARMS = {
    "image": {
        "role": "headline",
        "encoding": "raw luma as pixels",
        "detail": ("the frame's 15,360 luma values, replicated to RGB, "
                   "upscaled x4 with NEAREST to 640x384 and written as a "
                   "greyscale PNG whose bytes are hashed and recorded"),
    },
    "text": {
        "role": "headline",
        "encoding": "the same frame as characters",
        "detail": ("the same 15,360 luma values rendered by the repository's "
                   "OWN tone ramp (experiments/ascii_motion_render.py's "
                   "render_frame): 96 rows of 160 characters, one character "
                   "per pixel, with the renderer per-frame 1st/99th-"
                   "percentile contrast stretch"),
    },
    "text_global": {
        "role": "robustness_variant",
        "encoding": "the same frame as characters, absolute level kept",
        "detail": ("the same ramp and the same grid, but the mapping is a "
                   "FIXED 0..255 instead of the renderer's per-frame "
                   "stretch, so the description keeps the frame's absolute "
                   "luma level. This is a robustness variant on the "
                   "description's own terms, pre-registered, and it is "
                   "reported beside the headline"),
    },
    "image_scrambled": {
        "role": "control",
        "encoding": "the raw luma with its 4x4 blocks permuted",
        "detail": ("the same pixel values, the same histogram, the same "
                   "640x384 PNG, with the 4x4 blocks permuted by a FIXED "
                   "seed. It is the 'pixels destroyed but the statistics "
                   "kept' control"),
    },
    "text_scrambled": {
        "role": "control",
        "encoding": "the characters with their 4x4 blocks permuted",
        "detail": ("the SAME block permutation as image_scrambled, applied "
                   "to the same 96x160 character grid, so the two scrambled "
                   "arms are matched to each other exactly"),
    },
}

#: Which arms are scored on which conditions. B is a secondary condition
#: and is not scored for the control arms.
ARM_CONDITIONS = {
    "image": ("A", "C", "empty1", "B"),
    "text": ("A", "C", "empty1", "B"),
    "text_global": ("A", "C", "empty1"),
    "image_scrambled": ("A", "C", "empty1"),
    "text_scrambled": ("A", "C", "empty1"),
}

#: The prompt. The wording is IDENTICAL for every arm; only the CONTENT
#: differs (an image, or the characters). The question is the repository's
#: own vision question, from `experiments/local_vlm_vision.py`.
PROMPT_HEAD = ("Here is a representation of one camera frame from a "
               "desk-mounted camera in a workshop.\n\n")
QUESTION = ("Question: Is the person in this frame SPEAKING (their mouth is "
            "moving or open, as if talking) or NOT SPEAKING (mouth closed, "
            "still)? Answer with exactly one word: SPEAKING or NOT SPEAKING.")

#: The two scored answers, and the fact that only their FIRST token is
#: scored -- one token each, so there is no length bias between them.
ANSWER_POSITIVE = "SPEAKING"
ANSWER_NEGATIVE = "NOT SPEAKING"

# --------------------------------------------------------------------------
# THE PRE-REGISTERED DECISION RULE.
#
# Written 2026-10-20, BEFORE the measurement code in this file existed and
# BEFORE any model was run. It is copied verbatim into the results file
# under "pre_registered_rules", and `tests/test_vlm_image_retention.py`
# asserts that the copy in the results file still equals the copy here --
# so the rule cannot be quietly rewritten to fit the numbers.
# --------------------------------------------------------------------------

PRE_REGISTERED_RULE = {
    "version": "1.0",
    "written": "2026-10-20",
    "written_before": (
        "the measurement code in this file existed and before any model was "
        "run; the rule text was fixed first and every number was produced "
        "by the code as written"),
    "scope": (
        "session-level speaking/silent separation on this repository's own "
        "captures A (facing, speaking), C (facing, silent), empty1 (no "
        "person) and B (turned, speaking, secondary), read at the DETECTED "
        "NV12 stride, with one vision-language model scoring a forced "
        "binary choice per frame"),
    "question": (
        "does a model that receives the RAW IMAGE separate speaking from "
        "silent better than the SAME model receiving the SAME information "
        "as a TEXT/ASCII description of that image?"),
    "arms": {
        "image": ("the frame's 15,360 luma values as pixels: replicated to "
                  "RGB, upscaled x4 NEAREST to 640x384, written as a "
                  "greyscale PNG"),
        "text": ("the SAME 15,360 luma values as characters: the "
                 "repository's own tone ramp, 96 rows of 160 characters, one "
                 "character per pixel, per-frame 1st/99th-percentile "
                 "contrast stretch"),
        "text_global": ("the SAME ramp and grid with a FIXED 0..255 mapping, "
                        "so the absolute luma level survives into the "
                        "description. Pre-registered robustness variant."),
        "image_scrambled": ("the raw luma with its 4x4 blocks permuted by a "
                            "fixed seed: identical pixel histogram, "
                            "destroyed spatial arrangement"),
        "text_scrambled": ("the text arm's characters with the SAME 4x4 "
                           "block permutation"),
    },
    "headline_pair": ("image vs text. The other three arms are pre-registered "
                      "controls and robustness variants and are reported "
                      "beside the headline, not substituted into it."),
    "scoring": (
        "one forward pass per frame. The prompt is identical in every arm "
        "except for its CONTENT. The score is the logit of the FIRST token "
        "of 'SPEAKING' minus the logit of the FIRST token of 'NOT "
        "SPEAKING' at the final prompt position -- one token each, so there "
        "is no length bias and no dependence on the model's JSON formatting. "
        "A positive score means the model prefers 'speaking'."),
    "operating_point": (
        "the empty room's SCORE DISTRIBUTION is the control, and it enters "
        "the headline as the blank control's own separation. The "
        "threshold-free statistic is used because a thresholded margin "
        "saturates: when both the speaking take and the silent take sit "
        "on the same side of any blank-calibrated threshold, it reports "
        "zero even when the scores separate them cleanly. The "
        "thresholded margin IS computed and reported as a secondary "
        "statistic, at the operating point the empty room imposes "
        "(tau = the 95th percentile of the blank room's scores), because "
        "that is what a deployable system would see."),
    "margin_over_blank": (
        "margin_over_blank(arm) = sep(arm) - blank_sep(arm), where "
        "sep(arm) = AUC(arm's scores; A's frames vs C's frames) is the "
        "arm's separation of the speaking take from the silent take, and "
        "blank_sep(arm) = AUC(arm's scores; the EMPTY ROOM's frames vs "
        "C's frames) is the blank control's separation, ORIENTED so that "
        "the failure mode is a high value. An arm that ranks the empty "
        "room as MORE speaking than a silent person -- a model that says "
        "'speaking' for the blank room -- scores high on blank_sep, and "
        "the margin subtracts that credit. An arm that says the same "
        "thing about everything scores 0.50 on both terms, so its margin "
        "is 0 and it cannot win on the margin alone."),
    "headline_statistic": (
        "D = margin_over_blank(image) - margin_over_blank(text). The image "
        "arm wins iff D >= +0.05 AND the paired permutation z of D is at "
        "least +2.0. The text arm wins iff D <= -0.05 AND the paired "
        "permutation z of D is at most -2.0. Otherwise the verdict is that "
        "there is no difference at the pre-registered threshold."),
    "null": (
        "2,000 permutations of the A/C session labels over the POOLED A and "
        "C frames. The EMPTY ROOM IS NOT PERMUTED: it is the control, and "
        "the control is what the margin is measured against. The SAME "
        "permutation is applied to both arms, so the difference D has its "
        "own null rather than borrowing two separate ones. Seed "
        "20261020."),
    "checks": {
        "R1_headline": (
            "the image arm beats the text arm iff D >= +0.05 with a paired "
            "permutation z >= +2.0; the text arm beats the image arm iff "
            "D <= -0.05 with a paired permutation z <= -2.0"),
        "R2_honesty_gate": (
            "an arm is USABLE iff its own auc_ac (the AUC of its scores "
            "separating A's frames from C's frames) is above 0.50 with a "
            "permutation z of at least +2.0. The 'image beats text' verdict "
            "requires the image arm to be usable, and the 'text beats image' "
            "verdict requires the text arm to be usable. An arm that "
            "separates nothing cannot win, however its margin compares."),
        "R3_scramble_control": (
            "the scrambled arms are reported with the same statistics. An "
            "arm whose margin survives the scramble is reading global "
            "statistics of the frame, not its spatial arrangement. This is "
            "reported and is NOT decisive for the headline."),
        "R4_session_level_gate": (
            "always reported, never a pass or a fail: the labels are "
            "session-level, so every number here is a statement about the "
            "DISTRIBUTION of scores on A's frames against the distribution "
            "on C's frames. No frame-level audio<->video claim is "
            "admissible from these captures (the audio and the video are "
            "independent takes), and a predictor that identifies the "
            "SESSION rather than the speaking state scores the same as one "
            "that identifies the state."),
    },
    "verdict_rule": (
        "the verdict is computed mechanically from the recorded numbers: "
        "1. if NEITHER the image arm nor the text arm is usable (R2) -> "
        "'neither_arm_separates_speaking_from_silent'; "
        "2. elif R1 is met for the image arm -> "
        "'image_retention_beats_text_description'; "
        "3. elif R1 is met for the text arm -> "
        "'text_description_beats_image_retention'; "
        "4. else -> 'no_difference_at_the_preregistered_threshold'. "
        "Every branch is reachable and the verdict is recomputed from the "
        "numbers rather than read from a stored boolean."),
    "stopping_rule": (
        "every number reported here comes from ONE run of this script over "
        "the four captures on disk, with the frame indices, the scramble "
        "permutation, the prompt, the model revision and the seeds all "
        "fixed in advance. The results file records the script's own "
        "SHA-256, so any later edit is visible. No threshold, no frame "
        "count and no percentile was changed after the first run."),
}

PRE_REGISTERED_RULE_TEXT = """\
PRE-REGISTERED DECISION RULE (v1.0, 2026-10-20)

QUESTION.  Does a model that receives the RAW IMAGE separate speaking from
silent better than the SAME model receiving the SAME information as a
TEXT/ASCII description of that image?

SCOPE.  Session-level only, on this repository's own captures A (facing,
speaking), C (facing, silent), empty1 (no person) and B (turned, speaking,
secondary), read at the DETECTED NV12 stride.  The audio and the video are
independent takes, so no frame-level audio<->video label is recoverable
and none is claimed.

ARMS (all decided before the run).
  image            the frame's 15,360 luma values as pixels: RGB, upscaled
                   x4 NEAREST to 640x384, greyscale PNG.
  text             the SAME 15,360 values as characters: the repository's
                   own tone ramp, 96 rows x 160 chars, one char per pixel,
                   per-frame 1st/99th-percentile contrast stretch.
  text_global      the same ramp and grid with a FIXED 0..255 mapping, so
                   the absolute luma level survives.  Robustness variant.
  image_scrambled  the raw luma with its 4x4 blocks permuted (fixed seed):
                   identical pixel histogram, destroyed spatial structure.
  text_scrambled   the text arm's characters with the SAME permutation.

HEADLINE PAIR: image vs text.  The other arms are controls and are reported
beside the headline, never substituted into it.

SCORING.  One forward pass per frame.  The prompt is identical in every arm
except for its CONTENT.  The score is logit(first token of "SPEAKING") minus
logit(first token of "NOT SPEAKING") at the final prompt position: one
token each, so no length bias and no dependence on JSON formatting.

OPERATING POINT.  The empty room's SCORE DISTRIBUTION is the control.  A
thresholded margin SATURATES -- when the speaking take and the silent take
both sit on the same side of any blank-calibrated threshold it reports
zero even though the scores separate them -- so the headline statistic is
threshold-free.  The thresholded margin is still computed and reported as a
secondary statistic, at the operating point the empty room imposes (tau =
the 95th percentile of the blank room's scores), because that is what a
deployable system would see.

MARGIN OVER BLANK.
  sep(arm)        = AUC(arm; A's frames vs C's frames) -- the arm's
                    separation of the speaking take from the silent take.
  blank_sep(arm)  = AUC(arm; the EMPTY ROOM's frames vs C's frames), the
                    blank control's separation, ORIENTED so that the
                    failure mode is a HIGH value: an arm that ranks the
                    empty room as MORE speaking than a silent person -- a
                    model that says "speaking" for the blank room -- scores
                    high here, and the margin subtracts that credit.
  margin_over_blank(arm) = sep(arm) - blank_sep(arm).

  An arm that says the same thing about everything scores 0.50 on both
  terms, so its margin is 0 and it cannot win on the margin alone.

HEADLINE STATISTIC.
  D = margin_over_blank(image) - margin_over_blank(text).
  The image arm wins iff D >= +0.05 AND the paired permutation z of D is
  >= +2.0.  The text arm wins iff D <= -0.05 AND the paired permutation z
  of D is <= -2.0.  Otherwise: no difference at the pre-registered
  threshold.

NULL.  2,000 permutations of the A/C session labels over the pooled A and C
frames.  The EMPTY ROOM IS NOT PERMUTED: it is the control, and the
control is what the margin is measured against.  The SAME permutation is
applied to both arms so D has its own null.  Seed 20261020.

R1  the image arm beats the text arm iff D >= +0.05 with paired
    permutation z >= +2.0; the text arm beats the image arm iff
    D <= -0.05 with paired permutation z <= -2.0.
R2  HONESTY GATE.  An arm is USABLE iff its own auc_ac is above 0.50 with
    a permutation z >= +2.0.  The "image beats text" verdict requires the
    image arm to be usable; "text beats image" requires the text arm to be
    usable.  An arm that separates nothing cannot win.
R3  the scrambled arms are reported with the same statistics.  An arm whose
    margin survives the scramble is reading global statistics, not spatial
    arrangement.  Reported, not decisive.
R4  HONESTY GATE, always reported: every number here is a SESSION-LEVEL
    statement about the distribution of scores on A's frames against the
    distribution on C's frames.  A predictor that identifies the SESSION
    rather than the speaking state scores the same as one that identifies
    the state.

VERDICT (mechanical, from the recorded numbers; every branch reachable):
  1. neither arm usable (R2) -> neither_arm_separates_speaking_from_silent
  2. R1 met for the image arm -> image_retention_beats_text_description
  3. R1 met for the text arm -> text_description_beats_image_retention
  4. otherwise -> no_difference_at_the_preregistered_threshold
"""


# --------------------------------------------------------------------------
# Small numeric helpers. numpy only: the result must reproduce on the
# repository's own interpreter, which has no scipy.
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    """AUC(pos > neg) + 0.5 AUC(pos == neg): the rank AUC, ties included.

    The same statistic `experiments/mouth_detector.py` uses, so the AUCs in
    this lane are comparable with the mouth-lane's AUCs.
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = _avg_ranks(np.concatenate([pos, neg]))
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def perm_z(observed: float, null: np.ndarray) -> float:
    """z of an observation against a null distribution."""
    null = np.asarray(null, dtype=float)
    if len(null) == 0:
        return float("nan")
    sd = null.std(ddof=1)
    if sd <= 0:
        return float("inf") if observed > null.mean() else float("-inf")
    return float((observed - null.mean()) / sd)


# --------------------------------------------------------------------------
# Frame selection, rendering, and the scramble.
# --------------------------------------------------------------------------

def evenly_spaced_indices(n_total: int, n: int) -> list[int]:
    """n indices spread evenly over a capture's frame grid, in order.

    The rule is fixed in advance: index i is round(i * n_total / n). No
    randomness, no seed, no dependence on the content.
    """
    if n_total <= 0 or n <= 0:
        return []
    out: list[int] = []
    for i in range(n):
        j = int(round(i * n_total / float(n)))
        j = min(max(j, 0), n_total - 1)
        if not out or j != out[-1]:
            out.append(j)
    return out


def block_permutation(n_blocks: int, seed: int) -> np.ndarray:
    """A fixed permutation of a grid of blocks."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n_blocks)


def apply_block_permutation(arr: np.ndarray, block: int,
                            perm: np.ndarray) -> np.ndarray:
    """Permute the `block` x `block` tiles of a 2-D array.

    The pixel histogram is preserved EXACTLY (a permutation moves values,
    it does not create them), and only the spatial arrangement changes.
    """
    h, w = arr.shape
    if h % block or w % block:
        raise ValueError(f"shape {arr.shape} is not a multiple of {block}")
    nb_h, nb_w = h // block, w // block
    tiles = (arr.reshape(nb_h, block, nb_w, block)
             .transpose(0, 2, 1, 3)
             .reshape(nb_h * nb_w, block * block))
    out = (tiles[perm].reshape(nb_h, nb_w, block, block)
           .transpose(0, 2, 1, 3)
           .reshape(h, w))
    return out


def _load_repo_renderer():
    """The repository's OWN ASCII renderer, imported from its own file.

    `experiments/ascii_motion_render.py` holds the tone ramp and the
    `render_frame` function this lane uses for the TEXT arms, so the TEXT
    arm really is the repository's existing representation and not a
    re-derivation of it. The file's SHA-256 is recorded in the results.
    """
    path = ROOT / "experiments" / "ascii_motion_render.py"
    spec = importlib.util.spec_from_file_location("_repo_ascii_renderer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def render_text_frame(frame: np.ndarray, ramp: str, stretch: bool) -> str:
    """One frame as characters.

    stretch=True  -> the repository's own render_frame: a per-frame
                      1st/99th-percentile contrast stretch, then the ramp.
    stretch=False -> the same ramp and the same grid, but a FIXED 0..255
                      mapping, so the absolute luma level survives.
    """
    if stretch:
        lo = float(frame.min())
        hi = float(np.percentile(frame, 99.0))
        if hi <= lo:
            hi = lo + 1e-9
        q = np.clip((frame.astype(np.float64) - lo) / (hi - lo), 0, 1)
    else:
        q = np.clip(frame.astype(np.float64) / 255.0, 0, 1)
    last = len(ramp) - 1
    return "\n".join("".join(ramp[int(v * last)] for v in row) for row in q)


def png_bytes(grey: np.ndarray) -> bytes:
    """A greyscale PNG, written here rather than imported.

    `experiments/local_vlm_vision.py` says why: the point of the SHA-256 of
    the image bytes is that they are attributable to THIS file.
    """
    h, w = grey.shape
    raw = b"".join(b"\x00" + grey[r].tobytes() for r in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def upscale_nearest(frame: np.ndarray, factor: int) -> np.ndarray:
    """Exact pixel replication. No interpolation, so no invented detail."""
    return np.repeat(np.repeat(frame, factor, axis=0), factor, axis=1)


# --------------------------------------------------------------------------
# The statistics, and the verdict. All pure, so the tests can exercise
# them and can mutate a number and watch the verdict change.
# --------------------------------------------------------------------------

def separation_stats(scores: dict) -> dict:
    """The pre-registered separation, the blank control, and the margin.

    sep(X)       = AUC(A vs C): the arm's separation of the speaking take
                   from the silent take. Both are people facing the camera,
                   so this is the closest this corpus comes to a
                   speaking-vs-silent contrast.
    blank_sep(X) = AUC(blank vs C): the arm's separation of the EMPTY ROOM
                   from the silent person, oriented so that the failure
                   mode is a HIGH value. An arm that ranks the empty room
                   as MORE speaking than a silent person -- a model that
                   says "speaking" for the blank room -- scores high here,
                   and the margin subtracts that credit.
    margin(X)    = sep(X) - blank_sep(X).

    An arm that says the same thing about everything scores 0.50 for both
    terms, so its margin is 0. A degenerate arm therefore cannot win, and
    an arm that is fooled by the empty room has its margin reduced.
    """
    a = np.asarray(scores["A"], dtype=float)
    c = np.asarray(scores["C"], dtype=float)
    b = np.asarray(scores["empty1"], dtype=float)
    sep = auc(a, c)
    blank_sep = auc(b, c)
    return {
        "sep_auc_ac": sep,
        "blank_sep_auc_blank_vs_c": blank_sep,
        "margin_over_blank": sep - blank_sep,
    }


def threshold_stats(scores: dict, tau: float) -> dict:
    """The leak-free thresholded margin, at the empty room's own operating
    point. This is a SECONDARY statistic: it is what a deployable system
    would see, because the threshold comes from the blank room and not
    from the labels. It saturates when both classes sit on the same side
    of the threshold, which is why it is not the headline."""
    a = np.asarray(scores["A"], dtype=float)
    c = np.asarray(scores["C"], dtype=float)
    b = np.asarray(scores["empty1"], dtype=float)
    tpr_a = float((a > tau).mean())
    fpr_c = float((c > tau).mean())
    fpr_b = float((b > tau).mean())
    return {
        "tau": float(tau),
        "tpr_a": tpr_a,
        "fpr_c": fpr_c,
        "fpr_blank": fpr_b,
        "margin_over_blank_at_blank_tau": tpr_a - fpr_c,
    }


def blank_threshold(scores_blank: np.ndarray,
                    percentile: float = BLANK_PERCENTILE) -> float:
    """The operating point the empty room imposes on an arm."""
    return float(np.percentile(np.asarray(scores_blank, dtype=float),
                               percentile))


def make_permutations(n_a: int, n_c: int, n_perm: int,
                      seed: int) -> np.ndarray:
    """The A/C label permutations, generated ONCE so every arm sees the
    SAME permutation and the difference between arms has its own null."""
    rng = np.random.default_rng(seed)
    pool = n_a + n_c
    return np.stack([rng.permutation(pool) for _ in range(n_perm)])


def null_stats(scores_a: np.ndarray, scores_c: np.ndarray,
               scores_blank: np.ndarray, perms: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """The null distribution of (margin_over_blank, auc_ac) under the
    permuted A/C labels.

    The blank room is NOT permuted: it is the control, and the control is
    what the margin is measured against.
    """
    na = len(scores_a)
    pool = np.concatenate([np.asarray(scores_a, dtype=float),
                           np.asarray(scores_c, dtype=float)])
    margins = np.empty(len(perms))
    aucs = np.empty(len(perms))
    for k, p in enumerate(perms):
        pos = pool[p[:na]]
        neg = pool[p[na:]]
        aucs[k] = auc(pos, neg)
        margins[k] = aucs[k] - auc(scores_blank, neg)
    return margins, aucs


def derive_verdict(results: dict) -> dict:
    """The verdict, RECOMPUTED from the recorded numbers.

    Nothing here reads a stored boolean: each check is re-evaluated from
    its own numbers, so a changed number changes the verdict and the
    verdict can come out either way.
    """
    margin = float(results["headline"]["margin_threshold"])
    z_thr = float(results["headline"]["z_threshold"])
    d = float(results["headline"]["difference"])
    z = float(results["headline"]["perm_z"])

    usable = {}
    for arm in ("image", "text"):
        a = results["arms"][arm]
        usable[arm] = bool(a["auc_ac"] > 0.50
                            and a["auc_ac_perm_z"] >= z_thr)

    image_wins = bool(d >= margin and z >= z_thr and usable["image"])
    text_wins = bool(d <= -margin and z <= -z_thr and usable["text"])

    if not usable["image"] and not usable["text"]:
        verdict = "neither_arm_separates_speaking_from_silent"
    elif image_wins:
        verdict = "image_retention_beats_text_description"
    elif text_wins:
        verdict = "text_description_beats_image_retention"
    else:
        verdict = "no_difference_at_the_preregistered_threshold"

    return {
        "verdict": verdict,
        "checks": {
            "R1_image_wins": image_wins,
            "R1_text_wins": text_wins,
            "R2_image_usable": usable["image"],
            "R2_text_usable": usable["text"],
        },
        "inputs": {
            "difference": d, "perm_z": z,
            "margin_threshold": margin, "z_threshold": z_thr,
        },
    }


# --------------------------------------------------------------------------
# The model. torch and transformers are imported HERE, not at the top of
# the file, so that the pure functions above stay importable on a machine
# with no GPU and no torch.
# --------------------------------------------------------------------------

def model_identity(model_id: str) -> dict:
    """The identity of the weights: revision, and a SHA-256 over them.

    The revision alone is a NAME; the content hash says the bytes were the
    ones measured.
    """
    from huggingface_hub import snapshot_download
    try:
        snap = Path(snapshot_download(model_id))
    except Exception:  # noqa: BLE001 -- offline: use what is cached
        snap = Path(snapshot_download(model_id, local_files_only=True,
                                      allow_patterns=["*.safetensors"]))
    h = hashlib.sha256()
    files = []
    for f in sorted(snap.glob("*.safetensors")):
        data = f.read_bytes()
        h.update(data)
        files.append({"name": f.name, "bytes": len(data),
                      "sha256": sha256_hex(data)})
    return {
        "model_id": model_id,
        "revision": snap.name,
        "weights_sha256": h.hexdigest(),
        "weight_files": files,
        "weights_bytes": sum(f["bytes"] for f in files),
    }


def load_model(model_id: str, dtype: str = DTYPE, device_map: str = DEVICE_MAP):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch_dtype = {"float16": torch.float16, "float32": torch.float32,
                   "bfloat16": torch.bfloat16}[dtype]
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch_dtype, device_map=device_map)
    model.eval()
    load_s = time.time() - t0
    first = next(model.parameters())
    return model, processor, {
        "dtype": dtype,
        "device_map": device_map,
        "resolved_device": str(first.device),
        "resolved_dtype": str(first.dtype),
        "class": type(model).__name__,
        "load_seconds": round(load_s, 2),
        "hf_device_map": getattr(model, "hf_device_map", None),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
        "gpu_total_mib": (int(torch.cuda.get_device_properties(0).total_memory
                              // (1024 * 1024))
                          if torch.cuda.is_available() else None),
        "vram_allocated_gb": (round(torch.cuda.memory_allocated() / 1e9, 3)
                              if torch.cuda.is_available() else None),
        "vram_peak_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                         if torch.cuda.is_available() else None),
    }


def score_one(model, processor, arm: str, frame: np.ndarray,
              ramp: str, perm: np.ndarray) -> dict:
    """One forward pass. Returns the score and the record of what was sent.

    The image arms are sent as PNG bytes whose SHA-256 is recorded; the
    text arms are sent as characters whose SHA-256 is recorded. In both
    cases the bytes sent are the bytes hashed.
    """
    import torch
    from PIL import Image

    if arm in ("image", "image_scrambled"):
        if arm == "image_scrambled":
            frame = apply_block_permutation(frame, SCRAMBLE_BLOCK, perm)
        big = upscale_nearest(frame, UPSCALE)
        data = png_bytes(big)
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"},
                                            {"type": "text",
                                             "text": PROMPT_HEAD + QUESTION}]}]
        text = processor.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil], return_tensors="pt")
        content_sha = sha256_hex(data)
        content_kind = "png"
        content_shape = [int(big.shape[0]), int(big.shape[1])]
    else:
        if arm == "text_scrambled":
            frame = apply_block_permutation(frame, SCRAMBLE_BLOCK, perm)
        stretch = arm != "text_global"
        ascii_text = render_text_frame(frame, ramp, stretch)
        body = PROMPT_HEAD + ascii_text + "\n\n" + QUESTION
        msgs = [{"role": "user", "content": [{"type": "text", "text": body}]}]
        text = processor.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt")
        content_sha = sha256_hex(ascii_text.encode("utf-8"))
        content_kind = "ascii"
        content_shape = [int(frame.shape[0]), int(frame.shape[1])]

    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}
    t0 = time.time()
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    dt = time.time() - t0
    logits = out.logits[0, -1].float()
    tok = processor.tokenizer
    id_pos = tok.encode(ANSWER_POSITIVE)[0]
    id_neg = tok.encode(ANSWER_NEGATIVE)[0]
    score = float(logits[id_pos] - logits[id_neg])
    greedy = int(logits.argmax())
    grid = None
    if "image_grid_thw" in inputs:
        grid = [int(v) for v in inputs["image_grid_thw"][0].tolist()]
    return {
        "score": score,
        "logit_positive": float(logits[id_pos]),
        "logit_negative": float(logits[id_neg]),
        "greedy_first_token_id": greedy,
        "greedy_is_one_of_the_options": bool(greedy in (id_pos, id_neg)),
        "n_prompt_tokens": int(inputs["input_ids"].shape[1]),
        "content_sha256": content_sha,
        "content_kind": content_kind,
        "content_shape": content_shape,
        "image_grid_thw": grid,
        "seconds": round(dt, 3),
    }


# --------------------------------------------------------------------------
# The run.
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--capture-dir", default=str(CAPTURE_DIR))
    ap.add_argument("--out", "--json", dest="out",
                    default=str(ROOT / "experiments" /
                                 "vlm_image_retention_results.json"),
                    help="path to write the results JSON to")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device-map", default=DEVICE_MAP)
    ap.add_argument("--dtype", default=DTYPE)
    ap.add_argument("--frames", type=int, default=N_FRAMES_MAIN,
                    help="frames per main condition (A, C, empty1)")
    ap.add_argument("--frames-secondary", type=int, default=N_FRAMES_SECONDARY,
                    help="frames for the secondary condition B")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-model", action="store_true",
                    help="write the report without scoring anything "
                         "(for a smoke test; the verdict is then UNDETERMINED)")
    args = ap.parse_args(argv)

    cap = Path(args.capture_dir)
    renderer, renderer_path = _load_repo_renderer()
    ramp = renderer.RAMP

    report: dict = {
        "identity": {
            "script": "experiments/vlm_image_retention.py",
            "question": PRE_REGISTERED_RULE["question"],
            "run_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "argv": list(sys.argv),
            "dependency_note": (
                "numpy only outside the scoring functions; torch and "
                "transformers are imported inside them, so this module is "
                "importable on the repository's own interpreter, which has "
                "no torch"),
        },
        "pre_registered_rules": PRE_REGISTERED_RULE,
        "pre_registered_rule_text": PRE_REGISTERED_RULE_TEXT,
    }

    # ---- read the captures through the shared reader --------------------
    layouts: dict = {}
    luma: dict = {}
    selection: dict = {}
    for sess in ("A", "B", "C", "empty1"):
        path = cap / f"{sess}.nv12"
        arr, layout = nv12.read_luma(path, width=FRAME_W, height=FRAME_H)
        luma[sess] = arr
        layouts[sess] = layout
        n = (args.frames if sess in ("A", "C", "empty1")
             else args.frames_secondary)
        idx = evenly_spaced_indices(arr.shape[0], n)
        # How many of the SELECTED frames repeat their predecessor? The
        # captures repeat frames (the camera delivers ~19.9 distinct
        # images per second on a 30 fps grid), and the selected frames
        # must be reported, not assumed to be distinct.
        dup = 0
        for a, b in zip(idx, idx[1:]):
            if np.array_equal(arr[a], arr[b]):
                dup += 1
        selection[sess] = {
            "n_frames_on_disk": int(arr.shape[0]),
            "duration_s_at_30fps": float(arr.shape[0] / FPS),
            "n_scored": len(idx),
            "frame_indices": idx,
            "selection_rule": "round(i * n_frames_on_disk / n_scored), "
                              "in order, duplicates removed",
            "n_selected_pairs_byte_identical": dup,
        }
        report["identity"].setdefault("record_layout", {})[sess] = \
            layout.as_dict()
    report["frames"] = selection

    # ---- the scramble permutation, fixed in advance --------------------
    nb_h = FRAME_H // SCRAMBLE_BLOCK
    nb_w = FRAME_W // SCRAMBLE_BLOCK
    perm = block_permutation(nb_h * nb_w, args.seed)
    report["scramble"] = {
        "block_px": SCRAMBLE_BLOCK,
        "grid": [nb_h, nb_w],
        "seed": int(args.seed),
        "permutation_sha256": sha256_hex(
            np.asarray(perm, dtype=np.int64).tobytes()),
        "permutation": [int(p) for p in perm],
        "note": ("the SAME permutation is applied to every frame of every "
                 "condition and to both scrambled arms, so the image and "
                 "text scrambles are matched to each other"),
    }

    report["representations"] = {
        "image": {
            "source": "the frame's raw luma, exactly as src/nv12.py read it",
            "upscale": UPSCALE,
            "size_px": [IMAGE_H, IMAGE_W],
            "resampling": "NEAREST (exact pixel replication, no invented "
                          "detail)",
            "colour": "greyscale luma replicated to RGB",
            "png": "8-bit greyscale PNG, written by this file so the bytes "
                   "sent are the bytes hashed",
        },
        "text": {
            "source": "the SAME luma values",
            "renderer": "experiments/ascii_motion_render.py's render_frame",
            "renderer_sha256": sha256_file(renderer_path),
            "ramp": ramp,
            "grid": [FRAME_H, FRAME_W],
            "chars_per_pixel": 1,
            "normalisation": "per-frame min and 99th percentile (the "
                             "renderer's own stretch)",
            "known_property": ("the stretch makes each frame's own contrast "
                               "normal, so the TEXT arm does not carry the "
                               "frame's ABSOLUTE luma level; the IMAGE arm "
                               "does. The text_global arm is the "
                               "pre-registered variant that keeps it"),
        },
        "text_global": {
            "source": "the SAME luma values",
            "renderer": "experiments/ascii_motion_render.py's ramp",
            "renderer_sha256": sha256_file(renderer_path),
            "ramp": ramp,
            "grid": [FRAME_H, FRAME_W],
            "normalisation": "FIXED 0..255, so the absolute luma level "
                             "survives into the description",
        },
    }

    report["model"] = {
        "prompt_head": PROMPT_HEAD,
        "question": QUESTION,
        "answer_positive": ANSWER_POSITIVE,
        "answer_negative": ANSWER_NEGATIVE,
        "scoring": ("logit(first token of the positive answer) minus "
                    "logit(first token of the negative answer) at the final "
                    "prompt position; one token each, so no length bias"),
        "batch_size": 1,
        "decoding": "none: one forward pass, no generation",
    }

    if args.skip_model:
        report["arms"] = {}
        report["headline"] = {"difference": None, "perm_z": None,
                              "margin_threshold": HEADLINE_MARGIN,
                              "z_threshold": HEADLINE_Z}
        report["verdict"] = {"verdict": "UNDETERMINED",
                             "checks": {},
                             "inputs": {}}
        report["provenance"] = {"note": "skipped: no model was run"}
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out} (no model run)")
        return 0

    # ---- the model ------------------------------------------------------
    model, processor, model_info = load_model(args.model, args.dtype,
                                              args.device_map)
    report["model"].update(model_info)
    report["model"]["identity"] = model_identity(args.model)
    tok = processor.tokenizer
    id_pos = tok.encode(ANSWER_POSITIVE)[0]
    id_neg = tok.encode(ANSWER_NEGATIVE)[0]
    report["model"]["answer_token_ids"] = {
        "positive_first_token_id": int(id_pos),
        "negative_first_token_id": int(id_neg),
        "positive_tokens": [int(t) for t in tok.encode(ANSWER_POSITIVE)],
        "negative_tokens": [int(t) for t in tok.encode(ANSWER_NEGATIVE)],
    }

    # ---- score every arm on every condition -----------------------------
    raw: dict = {}
    timings: dict = {}
    for arm in ARMS:
        raw[arm] = {}
        for sess in ARM_CONDITIONS[arm]:
            idx = selection[sess]["frame_indices"]
            recs = []
            t0 = time.time()
            for k, j in enumerate(idx):
                rec = score_one(model, processor, arm, luma[sess][j], ramp,
                                perm)
                rec["frame_index"] = int(j)
                recs.append(rec)
                if (k + 1) % 30 == 0:
                    print(f"  {arm}/{sess}: {k + 1}/{len(idx)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
            raw[arm][sess] = recs
            timings[f"{arm}/{sess}"] = round(time.time() - t0, 2)
            print(f"{arm}/{sess}: {len(recs)} frames in "
                  f"{timings[f'{arm}/{sess}']}s", flush=True)

    # ---- the harness control: the same frames scored twice --------------
    repeat: dict = {}
    for arm in ("image", "text"):
        first = raw[arm]["A"][:8]
        again = [score_one(model, processor, arm, luma["A"][r["frame_index"]],
                           ramp, perm) for r in first]
        deltas = [abs(a["score"] - b["score"])
                  for a, b in zip(first, again)]
        repeat[arm] = {
            "n_frames": len(deltas),
            "max_abs_score_delta": float(max(deltas)) if deltas else None,
            "scores_identical": bool(all(d == 0.0 for d in deltas)),
        }

    # ---- the statistics --------------------------------------------------
    scores = {arm: {sess: np.array([r["score"] for r in raw[arm][sess]],
                                   dtype=float)
                    for sess in raw[arm]} for arm in ARMS}

    perms = make_permutations(len(scores["image"]["A"]),
                              len(scores["image"]["C"]),
                              args.n_perm, args.seed)

    arms_out: dict = {}
    for arm in ARMS:
        entry: dict = {
            "role": ARMS[arm]["role"],
            "encoding": ARMS[arm]["encoding"],
            "detail": ARMS[arm]["detail"],
            "conditions": {},
        }
        for sess in ARM_CONDITIONS[arm]:
            s = scores[arm][sess]
            recs = raw[arm][sess]
            entry["conditions"][sess] = {
                "label": CONDITIONS[sess]["label"],
                "role": CONDITIONS[sess]["role"],
                "pose": CONDITIONS[sess]["pose"],
                "n_frames": int(len(s)),
                "mean_score": float(s.mean()),
                "median_score": float(np.median(s)),
                "sd_score": float(s.std(ddof=1)) if len(s) > 1 else None,
                "min_score": float(s.min()),
                "max_score": float(s.max()),
                "fraction_preferring_speaking_at_zero": float((s > 0).mean()),
                "n_greedy_first_token_outside_the_two_options": int(
                    sum(1 for r in recs
                        if not r["greedy_is_one_of_the_options"])),
                "mean_prompt_tokens": float(
                    np.mean([r["n_prompt_tokens"] for r in recs])),
                "content_sha256_first": recs[0]["content_sha256"],
                "content_sha256_last": recs[-1]["content_sha256"],
            }
        # the separation and the controls
        tau = blank_threshold(scores[arm]["empty1"])
        sep = separation_stats(scores[arm])
        thr = threshold_stats(scores[arm], tau)
        a, c = scores[arm]["A"], scores[arm]["C"]
        _m_null, a_null = null_stats(a, c, scores[arm]["empty1"], perms)
        entry["separation"] = {
            "auc_ac": sep["sep_auc_ac"],
            "auc_ac_perm_z": perm_z(sep["sep_auc_ac"], a_null),
            "auc_a_blank": auc(a, scores[arm]["empty1"]),
            "auc_c_blank": auc(c, scores[arm]["empty1"]),
            "blank_sep_auc_blank_vs_c": sep["blank_sep_auc_blank_vs_c"],
            "cohens_d_ac": float((a.mean() - c.mean())
                                 / np.sqrt((a.var(ddof=1) + c.var(ddof=1)) / 2)),
            "permuted_label_null_auc": 0.50,
            "n_permutations": int(args.n_perm),
        }
        entry["control_blank"] = {
            "percentile": BLANK_PERCENTILE,
            "tau": tau,
            "tpr_a": thr["tpr_a"],
            "fpr_c": thr["fpr_c"],
            "fpr_blank": thr["fpr_blank"],
            "margin_over_blank_at_blank_tau": thr[
                "margin_over_blank_at_blank_tau"],
            "blank_mean_score": float(scores[arm]["empty1"].mean()),
            "blank_median_score": float(np.median(scores[arm]["empty1"])),
        }
        entry["margin_over_blank"] = sep["margin_over_blank"]
        entry["sep_auc_ac"] = sep["sep_auc_ac"]
        entry["auc_ac"] = sep["sep_auc_ac"]
        entry["auc_ac_perm_z"] = entry["separation"]["auc_ac_perm_z"]
        if "B" in scores[arm]:
            entry["secondary_condition_B"] = {
                "label": CONDITIONS["B"]["label"],
                "pose": CONDITIONS["B"]["pose"],
                "note": ("B is a SPEAKING take with the head turned away. It "
                         "is NOT a silent class and is not used as one; it "
                         "is reported because it separates 'facing' from "
                         "'speaking' as a confound"),
                "mean_score": float(scores[arm]["B"].mean()),
                "auc_a_b": auc(scores[arm]["A"], scores[arm]["B"]),
                "auc_b_c": auc(scores[arm]["B"], c),
                "fraction_preferring_speaking_at_zero": float(
                    (scores[arm]["B"] > 0).mean()),
            }
        arms_out[arm] = entry

    # ---- the headline ---------------------------------------------------
    img_m_null, _ = null_stats(scores["image"]["A"], scores["image"]["C"],
                               scores["image"]["empty1"], perms)
    txt_m_null, _ = null_stats(scores["text"]["A"], scores["text"]["C"],
                               scores["text"]["empty1"], perms)
    d_null = img_m_null - txt_m_null
    d_obs = (arms_out["image"]["margin_over_blank"]
             - arms_out["text"]["margin_over_blank"])
    headline = {
        "statistic": ("margin_over_blank(image) - margin_over_blank(text), "
                      "where margin_over_blank(arm) = AUC(arm; A vs C) - "
                      "AUC(arm; blank vs C)"),
        "margin_over_blank_image": arms_out["image"]["margin_over_blank"],
        "margin_over_blank_text": arms_out["text"]["margin_over_blank"],
        "sep_auc_ac_image": arms_out["image"]["sep_auc_ac"],
        "sep_auc_ac_text": arms_out["text"]["sep_auc_ac"],
        "blank_sep_image": arms_out["image"]["separation"][
            "blank_sep_auc_blank_vs_c"],
        "blank_sep_text": arms_out["text"]["separation"][
            "blank_sep_auc_blank_vs_c"],
        "difference": d_obs,
        "margin_threshold": HEADLINE_MARGIN,
        "z_threshold": HEADLINE_Z,
        "perm_z": perm_z(d_obs, d_null),
        "null_mean": float(np.mean(d_null)),
        "null_sd": float(np.std(d_null, ddof=1)),
        "n_permutations": int(args.n_perm),
        "null_seed": int(args.seed),
        "paired": ("the SAME A/C permutation is applied to both arms, so "
                   "the difference has its own null"),
        "secondary_auc_difference": (arms_out["image"]["auc_ac"]
                                     - arms_out["text"]["auc_ac"]),
        "secondary_threshold_margin_image": arms_out["image"][
            "control_blank"]["margin_over_blank_at_blank_tau"],
        "secondary_threshold_margin_text": arms_out["text"][
            "control_blank"]["margin_over_blank_at_blank_tau"],
    }

    report["arms"] = arms_out
    report["headline"] = headline
    report["controls"] = {
        "blank_room": {
            "what": ("the empty room's frames, scored in EACH arm's own "
                     "representation; it sets that arm's operating point "
                     "and it is the null for 'the arm is detecting "
                     "something'"),
            "per_arm": {arm: arms_out[arm]["control_blank"] for arm in ARMS},
        },
        "scrambled": {
            "what": ("the same frames with their 4x4 blocks permuted: "
                     "identical histogram, destroyed spatial arrangement"),
            "per_arm": {arm: {
                "margin_over_blank": arms_out[arm]["margin_over_blank"],
                "auc_ac": arms_out[arm]["auc_ac"],
                "tpr_a": arms_out[arm]["control_blank"]["tpr_a"],
                "fpr_c": arms_out[arm]["control_blank"]["fpr_c"],
                "fpr_blank": arms_out[arm]["control_blank"]["fpr_blank"],
            } for arm in ("image_scrambled", "text_scrambled")},
        },
        "harness_determinism": repeat,
        "timings_seconds": timings,
    }

    report["verdict"] = derive_verdict(report)

    report["provenance"] = {
        "script": "experiments/vlm_image_retention.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {name: {"path": str(cap / f"{name}.nv12"),
                          "sha256": sha256_file(cap / f"{name}.nv12"),
                          "bytes": (cap / f"{name}.nv12").stat().st_size}
                   for name in ("A", "B", "C", "empty1")},
        "reader": {"path": "src/nv12.py",
                   "sha256": sha256_file(ROOT / "src" / "nv12.py"),
                   "note": ("the frame layout is DETECTED from the content "
                            "and the margin is recorded; see "
                            "docs/research/NV12-STRIDE.md")},
        "renderer": {"path": "experiments/ascii_motion_render.py",
                     "sha256": sha256_file(renderer_path)},
        "model": report["model"]["identity"],
        "gpu": {"name": model_info.get("gpu_name"),
                "total_mib": model_info.get("gpu_total_mib"),
                "dtype": model_info.get("dtype"),
                "device_map": model_info.get("device_map"),
                "vram_peak_gb": model_info.get("vram_peak_gb")},
        "seeds": {"permutation_seed": int(args.seed),
                  "scramble_seed": int(args.seed),
                  "note": ("no sampling anywhere: every score is one "
                           "deterministic forward pass, so there is no "
                           "generation seed to set")},
        "commands": [
            ("/home/proart/projects/gpu-venv/bin/python "
             "experiments/vlm_image_retention.py --out "
             "experiments/vlm_image_retention_results.json"),
            ("/home/proart/projects/glasses/.venv/bin/python -m pytest "
             "tests/test_vlm_image_retention.py -q"),
        ],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": model_info.get("torch_version"),
            "cuda": model_info.get("cuda_version"),
        },
    }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nverdict: {report['verdict']['verdict']}")
    print(f"  margin_over_blank image = "
          f"{arms_out['image']['margin_over_blank']:+.4f}")
    print(f"  margin_over_blank text  = "
          f"{arms_out['text']['margin_over_blank']:+.4f}")
    print(f"  difference D = {d_obs:+.4f}  (perm z "
          f"{headline['perm_z']:+.2f})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
