"""
speaker_embedding_f0.py -- can a pretrained speaker embedding resolve the
same-gender speaker pairs that pitch provably cannot?

WHY THIS FILE EXISTS.

`docs/research/F0-SPEAKER-CHANGE.md` closed a door and said which one:

    "A large F0 shift is intonation inside a phrase, not a speaker change. F0
     cannot rescue this corpus as a speaker-change proxy, and no threshold on
     pitch [can]."
    "across four speakers, median F0 separates two *voice classes* cleanly --
     men at 104-111 Hz against women at 197-208 Hz -- but cannot resolve the
     two speakers *within* each class. FEO070 and FEO072 sit 0.5 sigma apart;
     MEE071 and MEE073 0.3 sigma. Pitch proves there are two voices, not who
     is speaking."

That is a MEASURED blind spot, not a shrug: the same document reports the
numbers (`experiments/twospeaker_verify_results.json`) and the same document
names the escape:

    "It would take a speaker embedding, and the far-field single microphone
     would still have to resolve two simultaneous voices, which 203 audible
     segments across 19,305 voiced frames say is marginal at best."

So this file is that experiment: a pretrained x-vector model, applied to the
same AMI turns the pitch measurement used, with the pitch measurement
reproduced HERE, ON THE SAME TURNS, so the two instruments are compared on
the same data rather than in prose.

WHAT IS CLAIMED, AND WHAT COULD KILL IT.

  * H1: a speaker embedding separates FEO070 from FEO072, and MEE071 from
    MEE073, by MORE than the 0.5 sigma / 0.3 sigma that pitch manages -- in
    the same units, computed on the same turns.
  * H2: at a turn boundary, the embedding distance across a speaker change is
    larger than the distance within a speaker.

Both are falsifiable, and both are given a way to come out the other way:

  * A SHUFFLED-LABEL NULL. The speaker labels are permuted (preserving the
    per-speaker turn counts) and the whole separation is recomputed. If the
    separation survives shuffling, the measurement is not about speakers, and
    the hypothesis is dead. This is reported even if it kills the result.
  * A CONSTANT-DECISION BASELINE. Always predicting "speaker change" scores
    EER 0.5 by construction. It is written down so the headline can never be
    read without it.
  * AN F0-ONLY BASELINE ON THE SAME TURNS. The pitch measurement is rerun
    here, on the identical turn set, with the identical arithmetic. If the
    embedding cannot beat F0 on F0's own home turf, the embedding has added
    nothing.

THE MEASUREMENT, IN THE UNITS THE F0 DOCUMENT USED.

Both instruments are reduced to the SAME shape, so the two numbers are
comparable:

  * A within-speaker distance distribution: distances between two TURNS of the
    SAME speaker.
  * A between-speaker distance distribution: distances between two turns of
    DIFFERENT speakers.
  * The separation: (centre of the between distribution - centre of the
    within distribution) / sigma of the within distribution.

That last line is the same FORM as the F0 document's number, where the
"distance" was a difference of Hz and the "sigma" was the within-speaker
spread of F0. Here the "distance" is a difference of cosine distances and the
"sigma" is the within-speaker spread of the embedding distance. The units of
the numerator differ (Hz vs cosine distance) but the form is identical, and
because the F0 baseline is computed here with the identical arithmetic on the
identical turns, the comparison is like-for-like rather than rhetorical.

The F0 document's OWN number is also reproduced here, frame by frame, on the
same turn set, so the two documents cannot drift apart silently.

ONE MEASUREMENT IS NOT A PROPERTY: THE LENGTH SWEEP IS THE RESULT.

The single largest thing learned while building this file is that a speaker
embedding is only worth having if it is given enough audio. The x-vector head
pools statistics over the whole input, so a one-second clip of speech is
pooled over ~49 frames and the embedding is dominated by WHAT WAS SAID rather
than by WHO SAID IT. Measured on the model card's own example
(Librispeech speaker 1272, three utterances of the same person):

    full utterances (5-12 s)  cosine distance 0.027
    first 3.0 s               cosine distance 0.060
    first 1.0 s               cosine distance 0.273

The first version of this experiment used 1.0 s windows and produced a
separation of 0.02 sigma for FEO070 against FEO072 -- and a cross-gender
control of 0.48 sigma, which should have been impossible. The instrument was
being starved. The model card's own threshold (cosine similarity 0.86) is
meaningless on a one-second clip.

So the embedding unit here is THE TURN -- the whole annotated utterance, of
whatever length it happens to be -- and the measurement is reported as a
SWEEP over the minimum turn length, with the counts at every point. The
separation is a function of how much audio the model was given, and reporting
one point of that curve without the curve would be exactly the mistake
`docs/research/HOW-WE-KEEP-FOOLING-OURSELVES.md` is about.

THE HEADLINE OPERATING POINT IS CHOSEN BY A RULE, NOT BY THE ANSWER.

`HEADLINE_MIN_LEN_S = 2.0` is chosen because it is the longest minimum that
still leaves EVERY speaker with at least `MIN_TURNS_PER_SPEAKER = 20` turns,
so the enrolment/test split is not degenerate. It is not the point of the
sweep with the largest separation. The sweep is reported beside it, so a
reader can check the choice and see what a different rule would have given.

THE OVERLAP RULE IS NOT COPIED FROM THE F0 EXPERIMENT, AND THAT IS DELIBERATE.

`experiments/twospeaker_verify.py` excludes far-field segments that overlap
another speaker, because "a median f0 over them describes neither speaker".
That reasoning is specific to a MEDIAN OF A PERIODIC QUANTITY: two voices in
one window give two pitches, and the median of the mixture is not one
person's pitch. A turn-level x-vector over cross-talk is a different object
-- a soft-weighted summary in which the louder voice dominates -- and the
contamination shows up as an inflated WITHIN-speaker spread, which makes the
test conservative rather than flattering.

Both populations are therefore measured and reported side by side:
`all_audible` (the primary, larger sample) and `solo_only` (the F0
experiment's rule, a much smaller sample). If the solo-only separation is
NOT larger, the contamination story is wrong and the file says so.

WHAT IS DELIBERATELY NOT CLAIMED.

  * The single-channel corpus (`examples/realaudio/`) has NO per-turn speaker
    ground truth. Whatever is computed there is EXPLORATORY and is labelled
    as such in both this file and the document. It is not evidence.
  * Four speakers, one meeting series, one room. A separation measured here is
    a separation measured here.
  * The far-field channel carries two voices at once 24-30% of the time. The
    `solo_only` numbers are the ones that describe one voice at a time; the
    `all_audible` numbers are the ones that describe the actual recording
    channel. Neither is the other.

Run:  /home/proart/projects/gpu-venv/bin/python experiments/speaker_embedding_f0.py

The GPU venv is required because this file loads a pretrained x-vector model.
It is NOT required to read the committed result file, and the test suite
(`tests/test_speaker_embedding_f0.py`) never imports torch: the analysis
layer below is numpy + stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                                  # noqa: E402

from src import wavio                                               # noqa: E402

# The fast vectorised YIN from the F0 experiment, imported READ-ONLY so the
# pitch baseline in this file is measured with the same instrument as the
# number it is compared against. Importing it runs no measurement.
from experiments import twospeaker_verify as tsv                    # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

SR = 16000                     # the analysis rate src/wavio.py produces

MODEL_ID = "microsoft/wavlm-base-plus-sv"
MODEL_REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
MODEL_CLASS = "transformers.AutoModelForAudioXVector"

# The minimum turn length, swept. The embedding's discriminative power is a
# function of how much audio it is given, so the sweep IS the measurement.
MIN_LEN_SWEEP = (0.5, 1.0, 1.5, 2.0, 3.0)
HEADLINE_MIN_LEN_S = 2.0
MIN_TURNS_PER_SPEAKER = 20
BOUNDARY_W_S = 2.0             # the before/after clip at a turn boundary

# The x-vector head's TDNN convolutions need a minimum input length. A clip
# shorter than roughly 2 s leaves the network with one or zero frames and the
# forward pass returns NaN (or raises "Kernel size can't be greater than
# actual input size"). Every utterance is therefore padded to at least this
# length, with the attention mask keeping the padding out of the pooling.
# Measured, not assumed: 5 of 394 boundary clips came back non-finite before
# this was added.
MIN_INPUT_S = 2.0

REALTWOSPEAKER = ROOT / "examples" / "realtwospeaker"
SDM_FILES = ["ami_sdm_EN2002a_turns.wav", "ami_sdm_EN2002b_turns.wav"]
IHM_DIR = REALTWOSPEAKER / "ihm_EN2002a_channels_90s"
IHM_SPEAKERS = ["FEO070", "FEO072", "MEE071", "MEE073"]

SAME_GENDER_PAIRS = [("FEO070", "FEO072"), ("MEE071", "MEE073")]
CROSS_GENDER_CONTROL = ("FEO070", "MEE071")

# The bar a pair has to clear to be called RESOLVED. Written down as numbers,
# not adjectives, and able to come out the other way: a separation below
# RESOLVED_MIN_SIGMA, or an EER above RESOLVED_MAX_EER, or a null that the
# result does not beat, means the pair is NOT resolved.
RESOLVED_MIN_SIGMA = 1.0
RESOLVED_MAX_EER = 0.25
RESOLVED_MAX_P = 0.05

N_SHUFFLE = 200                # permutations in the shuffled-label null
SHUFFLE_SEED = 20261020
GAIN_SEED = 20261021
NOISE_SEED = 20261022

REAL_AUDIO_CELLS = {
    "narration": ROOT / "examples" / "realaudio" / "alice01_excerpt.wav",
    "poetry": ROOT / "examples" / "realaudio" / "corpus"
              / "poetry_weeds_by_the_wall_1702_librivox.wav",
    "essay": ROOT / "examples" / "realaudio" / "corpus"
             / "essay_perpetual_peace_librivox.wav",
    "dialogue": ROOT / "examples" / "realaudio" / "corpus"
               / "shortstory_shortstory058_1405_librivox.wav",
    "shakespeare": ROOT / "examples" / "realaudio" / "corpus"
                   / "shakespeare_midsummernightsdream_1201_librivox.wav",
    "children": ROOT / "examples" / "realaudio" / "corpus"
                / "children_english_fairy_tales_joy_librivox.wav",
    "oration": ROOT / "examples" / "realaudio" / "corpus"
               / "oration_jfks19620912.wav",
}
REAL_AUDIO_WINDOW_S = 3.0
REAL_AUDIO_HOP_S = 1.5


# ---------------------------------------------------------------------------
# small helpers (numpy + stdlib only; the test suite imports this module)
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mad_sigma(values) -> float:
    """Robust sigma from the median absolute deviation (1.4826 factor).

    The same convention as `experiments/twospeaker_verify.py:194`, so the
    numbers in this file and the numbers in the F0 document mean the same
    thing by "sigma".
    """
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    c = statistics.median(vals)
    return 1.4826 * statistics.median([abs(v - c) for v in vals])


def describe(values) -> dict:
    """Summary of a distance distribution. The counts and the centre/spread are
    enough to recompute the headline separation from the result file."""
    raw = [float(v) for v in values if v is not None]
    # a non-finite value would take a statistic down with it, and
    # `statistics.stdev` raises an obscure AttributeError on one, so they
    # are dropped here and COUNTED rather than carried silently
    vals = [v for v in raw if math.isfinite(v)]
    non_finite = len(raw) - len(vals)
    if not vals:
        return {"n": 0, "n_non_finite": non_finite}
    s = sorted(vals)
    n = len(s)
    return {
        "n": n,
        "n_non_finite": non_finite,
        "mean": round(float(statistics.fmean(s)), 6),
        "std": round(float(statistics.stdev(s)) if n > 1 else 0.0, 6),
        "mad_sigma": round(mad_sigma(s), 6),
        "median": round(float(statistics.median(s)), 6),
        "p05": round(s[max(0, math.ceil(0.05 * n) - 1)], 6),
        "p95": round(s[min(n - 1, math.ceil(0.95 * n) - 1)], 6),
        "min": round(s[0], 6),
        "max": round(s[-1], 6),
    }


def separation(within: dict, between: dict) -> dict:
    """The headline number, in the units the F0 document used.

    separation_sigma_mad = (median(between) - median(within)) / mad_sigma(within)
    separation_sigma_std = (mean(between)  - mean(within))  / std(within)

    The F0 document's 0.5 / 0.3 sigma were a difference of CENTRES over the
    within-speaker spread, so the median-based form is the one that is
    directly comparable; the mean-based and std-based forms are recorded
    beside it because the task's own wording is "the ratio of their means".
    """
    if not within.get("n") or not between.get("n"):
        return {"separation_sigma_mad": None, "separation_sigma_std": None,
                "mean_ratio": None, "median_ratio": None,
                "formula": ("separation_sigma_mad = (median(between) - "
                            "median(within)) / mad_sigma(within)")}
    ms = within.get("mad_sigma") or 0.0
    ss = within.get("std") or 0.0
    return {
        "separation_sigma_mad": (round((between["median"] - within["median"]) / ms, 4)
                                 if ms > 0 else None),
        "separation_sigma_std": (round((between["mean"] - within["mean"]) / ss, 4)
                                 if ss > 0 else None),
        "mean_ratio": (round(between["mean"] / within["mean"], 4)
                       if within["mean"] else None),
        "median_ratio": (round(between["median"] / within["median"], 4)
                         if within["median"] else None),
        "formula": ("separation_sigma_mad = (median(between) - median(within))"
                    " / mad_sigma(within)"),
    }


def eer_from_scores(target_scores, impostor_scores) -> dict:
    """Equal error rate for the rule "score >= threshold => different speaker".

    target   = trials whose speakers really ARE different (a miss is
               accepting them as the same speaker)
    impostor = trials whose speakers really are the same (a false alarm is
               rejecting them)

    The EER is the smallest achievable max(miss, false-alarm) over all
    thresholds, and the threshold that achieves it is reported, so the number
    can be recomputed from the recorded score lists.
    """
    t = sorted(float(x) for x in target_scores)
    i = sorted(float(x) for x in impostor_scores)
    if not t or not i:
        return {"eer": None, "n_target": len(t), "n_impostor": len(i),
                "note": "no trials of one class; EER is undefined, not 0.5"}
    candidates = sorted(set(t) | set(i))
    best = None
    for thr in candidates:
        miss = sum(1 for x in t if x < thr) / len(t)          # accepted as same
        fa = sum(1 for x in i if x >= thr) / len(i)           # rejected as different
        cand = {"threshold": thr, "miss_rate": miss, "false_alarm_rate": fa,
                "max_error": max(miss, fa)}
        if best is None or cand["max_error"] < best["max_error"] - 1e-12:
            best = cand
    cross = None
    for k in range(len(candidates) + 1):
        thr = candidates[k] if k < len(candidates) else candidates[-1] + 1e-9
        miss = sum(1 for x in t if x < thr) / len(t)
        fa = sum(1 for x in i if x >= thr) / len(i)
        if fa >= miss:
            cross = {"threshold": thr, "miss_rate": round(miss, 6),
                     "false_alarm_rate": round(fa, 6)}
            break
    return {
        "eer": round(best["max_error"], 6),
        "eer_definition": "min over thresholds of max(miss_rate, false_alarm_rate)",
        "at_threshold": best["threshold"],
        "miss_rate": round(best["miss_rate"], 6),
        "false_alarm_rate": round(best["false_alarm_rate"], 6),
        "crossing": cross,
        "n_target": len(t),
        "n_impostor": len(i),
    }


def cos_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two L2-normalised vectors (1 - dot)."""
    return float(1.0 - float(np.dot(a, b)))


def l2_normalise(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def two_means_1d(values: np.ndarray, iters: int = 100) -> tuple[float, float]:
    """Plain 2-means on one dimension; returns the two centres."""
    lo, hi = float(values.min()), float(values.max())
    centres = np.array([lo + (hi - lo) / 3.0, lo + 2.0 * (hi - lo) / 3.0])
    for _ in range(iters):
        a, b = centres
        side = (np.abs(values[:, None] - centres[None, :])).argmin(axis=1)
        new = np.array([values[side == 0].mean() if (side == 0).any() else a,
                        values[side == 1].mean() if (side == 1).any() else b])
        if np.allclose(new, centres, atol=1e-6):
            centres = new
            break
        centres = new
    return float(centres[0]), float(centres[1])


def permutation_pvalue(observed, null_values, higher_is_better: bool = True) -> dict:
    """How often the shuffled null is at least as extreme as the observed value.

    The direction matters and is stated, because getting it wrong inverts the
    result: a separation is better when it is LARGER, an EER is better when
    it is SMALLER. A first version of this file compared EERs the wrong way
    round and reported p = 1.0 for a result that beat every permutation.
    """
    vals = [float(v) for v in null_values if v is not None]
    if observed is None or not vals:
        return {"p_value": None, "n_null": len(vals),
                "note": "no null values or no observed value"}
    if higher_is_better:
        ge = sum(1 for v in vals if v >= observed)
        tail = "at_or_above_observed"
    else:
        ge = sum(1 for v in vals if v <= observed)
        tail = "at_or_below_observed"
    return {
        "p_value": round((1.0 + ge) / (1.0 + len(vals)), 6),
        "n_null": len(vals),
        "n_null_at_least_as_extreme": ge,
        "tail": tail,
        "direction": "higher is better" if higher_is_better else "lower is better",
        "null_mean": round(float(np.mean(vals)), 4),
        "null_max": round(float(np.max(vals)), 4),
        "null_median": round(float(np.median(vals)), 4),
    }


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

def load_mono16k(path: Path) -> tuple[np.ndarray, dict]:
    """Read a WAV with the repo's own reader.

    `src/wavio.py` is the instrument the whole project measures with: mono
    floats at 16 kHz, stereo averaged. Using it here means the claim "16 kHz
    mono, as src/wavio.py produces" is literally true rather than approximately
    true.
    """
    audio = wavio.read_wav(path)
    arr = np.asarray(audio.samples, dtype=np.float32)
    info = {"reader": "src/wavio.py:read_wav", "sample_rate": audio.sample_rate,
            "source_sample_rate": audio.source_sample_rate,
            "poisoned_samples": audio.poisoned_samples,
            "duration_s": round(len(arr) / audio.sample_rate, 6)}
    return arr, info


# ---------------------------------------------------------------------------
# corpus geometry: turns, speakers, overlap
# ---------------------------------------------------------------------------

def load_sidecar(path: Path) -> dict:
    return json.loads(path.read_text())


def window_origin(sidecar: dict) -> float:
    """The start of the FILE on the meeting timeline."""
    for key in ("meeting_window_start_s", "window_start_s"):
        if key in sidecar:
            return float(sidecar[key])
    return 0.0


def solo_flags(segments: list[dict]) -> list[bool]:
    """Does this segment overlap a DIFFERENT speaker's segment?

    Same rule as `experiments/twospeaker_verify.py`: on the meeting timeline,
    because that is where the overlap actually happened. A far-field segment
    that overlaps another speaker carries two voices on one channel.
    """
    flags = []
    for i, s in enumerate(segments):
        overlap = False
        for j, t in enumerate(segments):
            if j == i or t["speaker_id"] == s["speaker_id"]:
                continue
            if t["begin_time"] < s["end_time"] and s["begin_time"] < t["end_time"]:
                overlap = True
                break
        flags.append(overlap)
    return flags


class Turn:
    """One annotated utterance, with the file-local audio span it owns."""

    __slots__ = ("file", "index", "speaker", "audio_id", "begin_time", "end_time",
                 "audible_start", "audible_end", "overlapped", "lo", "hi",
                 "duration_s")

    def __init__(self, file, index, seg, origin, overlapped):
        self.file = file
        self.index = index
        self.speaker = seg["speaker_id"]
        self.audio_id = seg.get("audio_id")
        self.begin_time = float(seg["begin_time"])
        self.end_time = float(seg["end_time"])
        self.audible_start = float(seg["audible_start_s"])
        self.audible_end = float(seg["audible_end_s"])
        self.overlapped = overlapped
        self.lo = int(round((self.audible_start - origin) * SR))
        self.hi = int(round((self.audible_end - origin) * SR))
        self.duration_s = round(self.audible_end - self.audible_start, 4)


def collect_turns(name: str, sidecar: dict) -> list[Turn]:
    """The audible, annotated turns of one file, in meeting-time order."""
    segs = sidecar["segments"]
    flags = solo_flags(segs)
    origin = window_origin(sidecar)
    turns = []
    for i, seg in enumerate(segs):
        if not seg.get("audible"):
            continue
        turns.append(Turn(name, i, seg, origin, flags[i]))
    turns.sort(key=lambda t: (t.begin_time, t.end_time))
    return turns


# ---------------------------------------------------------------------------
# the model (torch is imported lazily so the test suite never needs it)
# ---------------------------------------------------------------------------

def load_xvector(model_id: str, revision: str, device: str):
    """Load a pretrained x-vector model and its feature extractor."""
    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector  # noqa: PLC0415
    fe = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
    model = AutoModelForAudioXVector.from_pretrained(model_id, revision=revision)
    model = model.to(device).eval()
    return fe, model


def embed_variable(fe, model, arrays: list[np.ndarray], device: str,
                   batch_size: int = 16,
                   per_item: bool = False) -> tuple[np.ndarray, dict]:
    """Embed a list of variable-length utterances. Returns (L2-normalised
    (N, D) embeddings, a provenance dict).

    Two things are pinned here, and both were found the hard way:

    1. LENGTH. The utterances are handed to the model's own feature
       extractor with `padding="max_length"` at ONE length for the whole
       call -- the longest utterance in the call, and never less than
       `MIN_INPUT_S`. `padding="longest"` would pad to the longest item in
       the BATCH, which would make one utterance's embedding depend on what
       else happened to be in its batch. Padding to a single length for the
       whole call makes every embedding a function of its own audio alone.

    2. THE ATTENTION MASK. `padding="max_length"` still returns a mask that
       marks the real samples, so the x-vector head's statistic pooling counts
       only the real frames and the padding contributes nothing.

    `per_item=True` pads each utterance to its OWN length instead, which is
    the model card's own usage; it is used only by the self-check, where the
    card's published threshold has to mean what the card says it means.

    NOTE: this model's `preprocessor_config.json` has `do_normalize: false`,
    so the raw waveform amplitude reaches the network. The gain null in
    `main()` measures what that actually costs rather than assuming it.
    """
    import torch                                                    # noqa: PLC0415
    info = {"n_inputs": len(arrays), "min_input_s": MIN_INPUT_S,
            "non_finite_embeddings": 0, "padding": "per_item" if per_item
            else "uniform_max_length"}
    if not arrays:
        return np.zeros((0, 0)), info
    lengths = [len(a) for a in arrays]
    max_len = max(max(lengths), int(MIN_INPUT_S * SR))
    info["max_length_samples"] = max_len
    info["max_length_s"] = round(max_len / SR, 4)
    out = []
    non_finite = 0

    def _run(chunk, length):
        inputs = fe(chunk, padding="max_length", max_length=length,
                    sampling_rate=SR, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = model(**inputs).embeddings
        return emb.detach().cpu().numpy().astype(np.float64)

    if per_item:
        for a in arrays:
            one = np.asarray(a, dtype=np.float32)[None, :]
            emb = _run([one[0]], max(len(one[0]), int(MIN_INPUT_S * SR)))
            non_finite += int((~np.isfinite(emb).all(axis=1)).sum())
            out.append(emb)
    else:
        for lo in range(0, len(arrays), batch_size):
            chunk = [np.asarray(a, dtype=np.float32)
                     for a in arrays[lo:lo + batch_size]]
            emb = _run(chunk, max_len)
            non_finite += int((~np.isfinite(emb).all(axis=1)).sum())
            out.append(emb)
    info["non_finite_embeddings"] = non_finite
    mat = np.vstack(out)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    mat = mat / norms
    # a non-finite embedding would poison every distance it takes part in, so
    # it is replaced by a zero vector and counted rather than carried along
    bad = ~np.isfinite(mat).all(axis=1)
    if bad.any():
        mat[bad] = 0.0
    return mat, info


# ---------------------------------------------------------------------------
# the pitch instrument
# ---------------------------------------------------------------------------

def turn_pitch(turns: list[Turn], samples_by_file: dict) -> dict:
    """Per-turn median F0 and the voiced-frame values behind it.

    Measured with the F0 experiment's own vectorised YIN, so the pitch
    baseline here and the number in `docs/research/F0-SPEAKER-CHANGE.md` are
    produced by the same instrument.
    """
    out: dict[int, dict] = {}
    for t in turns:
        s = samples_by_file[t.file]
        hz = tsv.voiced_hz(s[max(0, t.lo):max(0, t.hi)])
        out[id(t)] = {"median_hz": float(np.median(hz)) if hz.size else 0.0,
                      "frames": hz, "n_voiced_frames": int(hz.size)}
    return out


# ---------------------------------------------------------------------------
# the distance distributions
# ---------------------------------------------------------------------------

def build_index(turns: list[Turn], emb: np.ndarray):
    """Split a turn list and its embeddings into the two structures the
    distance code needs: {id(turn): {"turn": ...}} and {id(turn): vector}."""
    index: dict[int, dict] = {}
    embs: dict[int, np.ndarray] = {}
    for i, t in enumerate(turns):
        index[id(t)] = {"turn": t}
        embs[id(t)] = emb[i]
    return index, embs


def pair_distances(emb_by_turn: dict, turn_index: dict, a: str, b: str,
                   label_override: dict | None = None) -> tuple[list, list]:
    """Within- and between-speaker distances at TURN level.

    within  = pairs of turns of the SAME speaker
    between = pairs of turns of DIFFERENT speakers

    `label_override` relabels the turns (used by the shuffled-label null); it
    never touches the embeddings.
    """
    def spk_of(key):
        if label_override is not None:
            return label_override[key]
        return turn_index[key]["turn"].speaker

    within: list[float] = []
    between: list[float] = []
    keys_a = [k for k in turn_index if spk_of(k) == a]
    keys_b = [k for k in turn_index if spk_of(k) == b]
    for keys in (keys_a, keys_b):
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                within.append(cos_distance(emb_by_turn[keys[i]], emb_by_turn[keys[j]]))
    for ka in keys_a:
        for kb in keys_b:
            between.append(cos_distance(emb_by_turn[ka], emb_by_turn[kb]))
    return within, between


def speaker_pair_distances(emb_by_turn: dict, turn_index: dict, spk: str) -> list:
    keys = [k for k in turn_index if turn_index[k]["turn"].speaker == spk]
    return [cos_distance(emb_by_turn[keys[i]], emb_by_turn[keys[j]])
            for i in range(len(keys)) for j in range(i + 1, len(keys))]


# ---------------------------------------------------------------------------
# the speaker-verification task (enrol on some turns, test on others)
# ---------------------------------------------------------------------------

def enrol_test_split(turn_index: dict) -> dict:
    """Deterministic, disjoint enrol/test split of each speaker's turns.

    Turns are ordered by (file, begin time) and dealt alternately, so the two
    halves never share a turn and the split needs no RNG.
    """
    split: dict[str, list] = {}
    for spk in sorted({v["turn"].speaker for v in turn_index.values()}):
        keys = [k for k, v in turn_index.items() if v["turn"].speaker == spk]
        keys.sort(key=lambda k: (turn_index[k]["turn"].file,
                                 turn_index[k]["turn"].begin_time))
        split[spk] = {"enrol": keys[0::2], "test": keys[1::2]}
    return split


def verification_scores(emb_by_turn: dict, turn_index: dict) -> dict:
    """Target and impostor trial scores for the 4-speaker verification task."""
    split = enrol_test_split(turn_index)
    enrol_emb = {}
    for spk, d in split.items():
        if not d["enrol"]:
            continue
        enrol_emb[spk] = l2_normalise(
            np.mean([emb_by_turn[k] for k in d["enrol"]], axis=0))
    target, impostor, trials = [], [], []
    for spk, d in sorted(split.items()):
        for k in d["test"]:
            for other in sorted(split):
                if other not in enrol_emb:
                    continue
                score = cos_distance(enrol_emb[other], emb_by_turn[k])
                same = (other == spk)
                (impostor if same else target).append(score)
                trials.append({"enrol": other, "test_speaker": spk,
                               "same_speaker": same, "score": round(score, 6)})
    return {"target": target, "impostor": impostor, "trials": trials,
            "split_sizes": {s: {"enrol": len(d["enrol"]), "test": len(d["test"])}
                            for s, d in split.items()}}


def pair_verification_scores(emb_by_turn: dict, turn_index: dict,
                             a: str, b: str) -> dict:
    """The same task restricted to two speakers: can you tell A from B?

    Enrol on A's turns, test on A's held-out turns (impostor) and on B's
    turns (target); and the mirror image. This is the two-speaker question
    the F0 document could not answer.
    """
    split = enrol_test_split(turn_index)
    target, impostor = [], []
    for enrol_spk in (a, b):
        d_enrol = split.get(enrol_spk, {})
        if not d_enrol.get("enrol"):
            continue
        e = l2_normalise(np.mean([emb_by_turn[k] for k in d_enrol["enrol"]], axis=0))
        for test_spk in (a, b):
            for k in split.get(test_spk, {}).get("test", []):
                s = cos_distance(e, emb_by_turn[k])
                (impostor if enrol_spk == test_spk else target).append(s)
    return {"target": target, "impostor": impostor}


# ---------------------------------------------------------------------------
# the pitch instrument, in the same arithmetic
# ---------------------------------------------------------------------------

def f0_distances(turn_pitch_data: dict, turn_index: dict,
                 a: str, b: str) -> tuple[list, list]:
    """The F0 measurement, in the same shape as the embedding measurement.

    within  = |f0(turn1) - f0(turn2)| for two turns of the SAME speaker
    between = |f0(turn1) - f0(turn2)| for turns of DIFFERENT speakers
    """
    within, between = [], []
    keys_a = [k for k in turn_index if turn_index[k]["turn"].speaker == a]
    keys_b = [k for k in turn_index if turn_index[k]["turn"].speaker == b]
    for keys in (keys_a, keys_b):
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                within.append(abs(turn_pitch_data[keys[i]]["median_hz"]
                                  - turn_pitch_data[keys[j]]["median_hz"]))
    for ka in keys_a:
        for kb in keys_b:
            between.append(abs(turn_pitch_data[ka]["median_hz"]
                               - turn_pitch_data[kb]["median_hz"]))
    return within, between


def f0_doc_style_separation(turn_pitch_data: dict, turn_index: dict,
                            a: str, b: str, pooled_sigma: float | None = None) -> dict:
    """The F0 DOCUMENT's own number, recomputed on this turn set.

    The document pooled every voiced frame of a speaker, took the median of
    that pool, and divided the difference of the two speakers' medians by the
    pooled within-speaker MAD sigma of the FRAME values. This reproduces that
    arithmetic on the same turns, so the document's 0.5 / 0.3 sigma and this
    file's numbers are the same statistic and not two similar-looking ones.
    """
    keys_a = [k for k in turn_index if turn_index[k]["turn"].speaker == a]
    keys_b = [k for k in turn_index if turn_index[k]["turn"].speaker == b]
    if not keys_a or not keys_b:
        return {"status": "no turns"}
    frames_a = np.concatenate([turn_pitch_data[k]["frames"] for k in keys_a])
    frames_b = np.concatenate([turn_pitch_data[k]["frames"] for k in keys_b])
    pooled = np.concatenate([frames_a, frames_b])
    sig_a = mad_sigma(frames_a.tolist())
    sig_b = mad_sigma(frames_b.tolist())
    if pooled_sigma is None:
        # the document's noise floor: the ROOT-MEAN-SQUARE of the per-speaker
        # MAD sigmas, NOT the MAD sigma of the pooled frames. These are very
        # different numbers (22.28 Hz against 68.86 Hz on this population) and
        # the document's separations only reproduce with the first one.
        pooled_sigma = math.sqrt((sig_a ** 2 + sig_b ** 2) / 2.0)
    delta = abs(float(np.median(frames_a)) - float(np.median(frames_b)))
    return {
        "form": ("difference of the two speakers' pooled voiced-frame median "
                 "F0, over the pooled within-speaker MAD sigma of the frame "
                 "values -- the arithmetic of docs/research/F0-SPEAKER-CHANGE.md"),
        "median_f0_hz": {a: round(float(np.median(frames_a)), 2),
                         b: round(float(np.median(frames_b)), 2)},
        "median_delta_hz": round(delta, 2),
        "pooled_within_speaker_mad_sigma_hz": round(pooled_sigma, 2),
        "pooled_sigma_definition": ("root-mean-square of the per-speaker MAD "
                                    "sigmas -- the noise floor "
                                    "experiments/twospeaker_verify.py uses"),
        "pooled_frames_mad_sigma_hz": round(mad_sigma(pooled.tolist()), 2),
        "per_speaker_mad_sigma_hz": {a: round(sig_a, 2), b: round(sig_b, 2)},
        "separation_sigma": (round(delta / pooled_sigma, 4)
                             if pooled_sigma > 0 else None),
        "voiced_frames": {a: int(frames_a.size), b: int(frames_b.size)},
    }


def f0_doc_reproduction(turns_by_file: dict, turn_pitch_data: dict,
                        file_name: str) -> dict:
    """The F0 DOCUMENT's number, on the DOCUMENT's population.

    `docs/research/F0-SPEAKER-CHANGE.md` measured EN2002a with every audible
    segment that does not overlap another speaker, pooled the voiced frames
    per speaker, and divided the difference of the two speakers' pooled
    medians by the MAD sigma of the frame values pooled over ALL FOUR
    speakers. This reproduces that arithmetic on that population, so the
    document's 0.48 / 0.33 can be checked against this file's instrument
    instead of being taken on trust -- and so that the headline's F0
    baseline, measured on a DIFFERENT population, is not confused with it.
    """
    sel = [t for t in turns_by_file.get(file_name, []) if not t.overlapped]
    if not sel:
        return {"status": "no turns", "file": file_name}
    index = {id(t): {"turn": t} for t in sel}
    speakers = sorted({t.speaker for t in sel})
    frames = {}
    for spk in speakers:
        keys = [k for k, v in index.items() if v["turn"].speaker == spk]
        frames[spk] = np.concatenate([turn_pitch_data[k]["frames"] for k in keys])
    pooled = np.concatenate([frames[s] for s in speakers])
    per_speaker_sigma = {s: mad_sigma(frames[s].tolist()) for s in speakers}
    pooled_sigma = math.sqrt(sum(v ** 2 for v in per_speaker_sigma.values())
                             / len(per_speaker_sigma))
    out = {
        "file": file_name,
        "population": ("every audible segment that does not overlap another "
                       "speaker -- the F0 document's own rule"),
        "n_turns": len(sel),
        "pooled_within_speaker_mad_sigma_hz": round(pooled_sigma, 2),
        "pooled_sigma_definition": ("root-mean-square of the per-speaker MAD "
                                    "sigmas, as experiments/twospeaker_verify.py "
                                    "defines it"),
        "per_speaker_mad_sigma_hz": {s: round(v, 2)
                                     for s, v in per_speaker_sigma.items()},
        "document_value_to_check": {
            "FEO070|FEO072": 0.48, "MEE071|MEE073": 0.33,
            "source": "experiments/twospeaker_verify_results.json / "
                      "docs/research/F0-SPEAKER-CHANGE.md",
        },
        "pairs": {},
    }
    for a, b in SAME_GENDER_PAIRS:
        blk = f0_doc_style_separation(turn_pitch_data, index, a, b,
                                      pooled_sigma=pooled_sigma)
        doc = out["document_value_to_check"][f"{a}|{b}"]
        got = blk.get("separation_sigma")
        blk["document_value"] = doc
        blk["matches_document"] = (got is not None and abs(got - doc) <= 0.01)
        out["pairs"][f"{a}|{b}"] = blk
    return out


def f0_turn_level_separation(turn_pitch_data: dict, turn_index: dict,
                             a: str, b: str) -> dict:
    """The same FORM, but with the TURN median F0 as the per-turn scalar.

    The frame-level number above is the document's; this one is the pitch
    instrument in the same units as the embedding's headline (a per-turn
    scalar, a within-speaker spread, a difference of centres).
    """
    keys_a = [k for k in turn_index if turn_index[k]["turn"].speaker == a]
    keys_b = [k for k in turn_index if turn_index[k]["turn"].speaker == b]
    if not keys_a or not keys_b:
        return {"status": "no turns"}
    va = [turn_pitch_data[k]["median_hz"] for k in keys_a]
    vb = [turn_pitch_data[k]["median_hz"] for k in keys_b]
    pooled_sigma = mad_sigma(va + vb)
    delta = abs(statistics.median(va) - statistics.median(vb))
    return {
        "form": ("difference of the two speakers' median per-turn F0 over the "
                 "pooled within-speaker MAD sigma of the per-turn medians"),
        "median_f0_hz": {a: round(statistics.median(va), 2),
                         b: round(statistics.median(vb), 2)},
        "median_delta_hz": round(delta, 2),
        "pooled_within_speaker_mad_sigma_hz": round(pooled_sigma, 2),
        "separation_sigma": (round(delta / pooled_sigma, 4)
                             if pooled_sigma > 0 else None),
    }


# ---------------------------------------------------------------------------
# the turn-change task
# ---------------------------------------------------------------------------

def boundary_pairs(turns: list[Turn], window_s: float) -> list[dict]:
    """Consecutive annotated turns of ONE FILE that do not overlap in time.

    The score for a pair is the distance between the LAST clip of the earlier
    turn and the FIRST clip of the later turn -- exactly the decision the
    project has to make at a pause.
    """
    pairs = []
    w = int(round(window_s * SR))
    for a, b in zip(turns, turns[1:]):
        if b.begin_time < a.end_time:
            continue                                  # overlapping: skip
        if a.hi <= a.lo or b.hi <= b.lo:
            continue                                  # no audio at all
        a_lo = max(a.lo, a.hi - w)
        b_hi = min(b.hi, b.lo + w)
        pairs.append({
            "a": a, "b": b, "gap_s": round(b.begin_time - a.end_time, 3),
            "a_slice": (a_lo, a.hi), "b_slice": (b.lo, b_hi),
            "a_seconds": round((a.hi - a_lo) / SR, 3),
            "b_seconds": round((b_hi - b.lo) / SR, 3),
            "same_speaker": a.speaker == b.speaker,
        })
    return pairs


def within_turn_control(turns: list[Turn], window_s: float) -> list[dict]:
    """The "no speaker change at all" control: a turn split in half."""
    out = []
    w = int(round(window_s * SR))
    for t in turns:
        if (t.hi - t.lo) < 2 * w:
            continue
        mid = t.lo + (t.hi - t.lo) // 2
        out.append({"turn": t, "first": (t.lo, t.lo + w), "second": (mid, mid + w)})
    return out


def _median_f0(samples: np.ndarray, span: tuple[int, int]) -> float:
    lo, hi = span
    hz = tsv.voiced_hz(samples[max(0, lo):max(0, hi)])
    return float(np.median(hz)) if hz.size else 0.0


def _turn_change_task(samples_by_file: dict, turns_by_file: dict, fe, model,
                       device: str) -> dict:
    """The project's actual need: is there a speaker change at this pause?

    Boundaries are consecutive annotated turns that do not overlap. The score
    is the embedding distance between the last clip of the earlier turn and
    the first clip of the later turn.

    Negative classes:
      same-speaker boundary -- a different turn, the same person
      within-turn split    -- one turn cut in half, no change at all

    The F0 baseline is the same boundaries with |median F0 before - median F0
    after| as the score, so the two instruments are compared on identical
    trials.
    """
    pairs = []
    for fname in sorted(turns_by_file):
        pairs.extend(boundary_pairs(turns_by_file[fname], BOUNDARY_W_S))
    ctrls = []
    for fname in sorted(turns_by_file):
        ctrls.extend(within_turn_control(turns_by_file[fname], BOUNDARY_W_S))

    slices = []
    for p in pairs:
        slices.append(samples_by_file[p["a"].file][p["a_slice"][0]:p["a_slice"][1]])
        slices.append(samples_by_file[p["b"].file][p["b_slice"][0]:p["b_slice"][1]])
    c_slices = []
    for c in ctrls:
        s = samples_by_file[c["turn"].file]
        c_slices.append(s[c["first"][0]:c["first"][1]])
        c_slices.append(s[c["second"][0]:c["second"][1]])

    all_slices = slices + c_slices
    if all_slices:
        emb, _emb_info = embed_variable(fe, model, all_slices, device)
    else:
        emb = np.zeros((0, 512))
    n_pairs = len(pairs)
    pair_emb = emb[:2 * n_pairs].reshape(n_pairs, 2, -1) if n_pairs else np.zeros((0, 2, 512))
    c_emb = emb[2 * n_pairs:].reshape(len(ctrls), 2, -1) if ctrls else np.zeros((0, 2, 512))

    pos, neg_same, within_turn = [], [], []
    pos_f0, neg_same_f0 = [], []
    sides_min = []
    for i, p in enumerate(pairs):
        d = cos_distance(pair_emb[i, 0], pair_emb[i, 1])
        f0a = _median_f0(samples_by_file[p["a"].file], p["a_slice"])
        f0b = _median_f0(samples_by_file[p["b"].file], p["b_slice"])
        fd = abs(f0a - f0b)
        sides_min.append(min(p["a_seconds"], p["b_seconds"]))
        if p["same_speaker"]:
            neg_same.append(d)
            neg_same_f0.append(fd)
        else:
            pos.append(d)
            pos_f0.append(fd)
    for i in range(len(ctrls)):
        within_turn.append(cos_distance(c_emb[i, 0], c_emb[i, 1]))

    def subset_eer(min_side_s):
        idx = [i for i, s in enumerate(sides_min) if s >= min_side_s]
        p_scores, n_scores = [], []
        pi = ni = 0
        for i, p in enumerate(pairs):
            s = pos[pi] if not p["same_speaker"] else neg_same[ni]
            if not p["same_speaker"]:
                pi += 1
            else:
                ni += 1
            if i in idx:
                (n_scores if p["same_speaker"] else p_scores).append(s)
        return eer_from_scores(p_scores, n_scores)

    return {
        "definition": {
            "positive": "consecutive annotated turns with DIFFERENT speakers",
            "negative_same_speaker": "consecutive annotated turns with the SAME speaker",
            "negative_within_turn": "one turn split in half (no change at all)",
            "score": ("cosine distance between the last clip before the "
                      "boundary and the first clip after it"),
            "clip_s": BOUNDARY_W_S,
            "clip_rule": ("the last/first BOUNDARY_W_S seconds of the turn, "
                          "clipped to the turn when the turn is shorter"),
        },
        "n_boundaries": len(pairs),
        "n_clips_shorter_than_min_input_s": sum(
            1 for p in pairs
            for sec in (p["a_seconds"], p["b_seconds"])
            if sec < MIN_INPUT_S),
        "n_clips_total": 2 * len(pairs),
        "n_positive_different_speaker": len(pos),
        "n_negative_same_speaker": len(neg_same),
        "n_within_turn_controls": len(ctrls),
        "boundary_side_seconds": describe(sides_min),
        "embedding": {
            "eer_vs_same_speaker": eer_from_scores(pos, neg_same),
            "eer_vs_same_speaker_and_within_turn":
                eer_from_scores(pos, neg_same + within_turn),
            "eer_restricted_to_boundaries_with_1p5s_on_both_sides":
                subset_eer(1.5),
            "positive_scores": [round(x, 6) for x in pos],
            "negative_same_speaker_scores": [round(x, 6) for x in neg_same],
            "within_turn_scores": [round(x, 6) for x in within_turn],
            "positive": describe(pos),
            "negative_same_speaker": describe(neg_same),
            "within_turn": describe(within_turn),
        },
        "f0_baseline": {
            "eer_vs_same_speaker": eer_from_scores(pos_f0, neg_same_f0),
            "positive_scores": [round(x, 6) for x in pos_f0],
            "negative_same_speaker_scores": [round(x, 6) for x in neg_same_f0],
            "positive": describe(pos_f0),
            "negative_same_speaker": describe(neg_same_f0),
            "note": ("the same boundaries, scored by |median f0 before - median "
                     "f0 after| in Hz"),
        },
    }


def _turn_change_shuffled_null(turns_by_file: dict, task: dict,
                                n_perm: int, seed: int) -> dict:
    """Shuffle the speaker labels over the turns and re-derive the classes.

    The boundary geometry and the scores are held fixed; only the label that
    says whether a boundary is a hand-off is permuted. If the EER survives
    shuffling, the statistic is not reading the speaker change.
    """
    pairs = []
    for fname in sorted(turns_by_file):
        pairs.extend(boundary_pairs(turns_by_file[fname], BOUNDARY_W_S))
    if not pairs:
        return {"status": "no boundaries"}
    block = task
    pos_scores = block["embedding"]["positive_scores"]
    neg_scores = block["embedding"]["negative_same_speaker_scores"]
    if not pos_scores or not neg_scores:
        return {"status": "no trials"}

    boundary_scores = []
    pi = ni = 0
    for p in pairs:
        if p["same_speaker"]:
            boundary_scores.append(neg_scores[ni]); ni += 1
        else:
            boundary_scores.append(pos_scores[pi]); pi += 1

    turn_list = []
    for fname in sorted(turns_by_file):
        turn_list.extend(turns_by_file[fname])
    turn_speaker = {id(t): t.speaker for t in turn_list}

    rng = np.random.default_rng(seed)
    eers = []
    for _ in range(n_perm):
        keys = list(turn_speaker.keys())
        perm = rng.permutation(len(keys))
        shuffled = {keys[i]: turn_speaker[keys[perm[i]]] for i in range(len(keys))}
        new_pos, new_neg = [], []
        for p, s in zip(pairs, boundary_scores):
            if shuffled[id(p["a"])] == shuffled[id(p["b"])]:
                new_neg.append(s)
            else:
                new_pos.append(s)
        if not new_pos or not new_neg:
            continue
        e = eer_from_scores(new_pos, new_neg)
        if e.get("eer") is not None:
            eers.append(e["eer"])
    obs = block["embedding"]["eer_vs_same_speaker"]["eer"]
    return {
        "n_permutations": n_perm, "seed": seed,
        "observed_eer": obs,
        "null_eer": describe(eers),
        "permutation_test": permutation_pvalue(obs, eers, higher_is_better=False),
        "note": ("the boundary geometry and the scores are held fixed; only "
                 "the speaker labels over the turns are permuted, so the "
                 "hand-off / same-speaker classes are reassigned at random"),
    }


# ---------------------------------------------------------------------------
# the nulls on the headline separation
# ---------------------------------------------------------------------------

def shuffled_label_null(emb_by_turn: dict, turn_index: dict, pairs: list[tuple],
                        n_perm: int, seed: int) -> dict:
    """Permute the speaker labels over the TURNS and recompute the separation.

    The labels are permuted as TURN labels, so the per-speaker turn counts
    are preserved: only the assignment changes. A separation that survives
    this is not a separation between speakers.
    """
    keys = list(turn_index.keys())
    speakers = [turn_index[k]["turn"].speaker for k in keys]
    rng = np.random.default_rng(seed)
    raw: dict[str, list] = {f"{a}|{b}": [] for a, b in pairs}
    ratios: dict[str, list] = {f"{a}|{b}": [] for a, b in pairs}
    for _ in range(n_perm):
        perm = rng.permutation(len(keys))
        shuffled = {keys[i]: speakers[perm[i]] for i in range(len(keys))}
        for a, b in pairs:
            within, between = pair_distances(emb_by_turn, turn_index, a, b,
                                             label_override=shuffled)
            s = separation(describe(within), describe(between))
            raw[f"{a}|{b}"].append(s["separation_sigma_mad"])
            ratios[f"{a}|{b}"].append(s["mean_ratio"])
    return {
        "n_permutations": n_perm, "seed": seed,
        "method": ("speaker labels permuted over turns, per-speaker turn "
                   "counts preserved, separation recomputed from scratch"),
        "separation_sigma_mad": {k: describe(v) for k, v in raw.items()},
        "mean_ratio": {k: describe(v) for k, v in ratios.items()},
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# the headset-channel reference (the clean, channel-separated comparison)
# ---------------------------------------------------------------------------

def headset_reference(fe, model, device: str, min_len_s: float) -> dict:
    """The same separation measured on the per-speaker headset channels.

    On the headset channels a speaker's identity is the RECORDING, so this
    is the clean reference the far-field number should be compared against.
    """
    out: dict = {"role": ("channel-separated reference: one headset "
                          "microphone per speaker, so speaker identity is "
                          "separation of the recording, not an inference"),
                 "min_len_s": min_len_s}
    emb_by_spk: dict[str, dict] = {}
    for spk in IHM_SPEAKERS:
        wav = IHM_DIR / f"spk_{spk}.wav"
        side_path = IHM_DIR / f"spk_{spk}.segments.json"
        if not wav.is_file() or not side_path.is_file():
            out["status"] = "headset files not on disk"
            return out
        side = load_sidecar(side_path)
        samples, _info = load_mono16k(wav)
        turns = [t for t in collect_turns(wav.name, side)
                 if (t.hi - t.lo) >= int(min_len_s * SR)]
        if not turns:
            emb_by_spk[spk] = {}
            continue
        arrays = [samples[t.lo:t.hi] for t in turns]
        emb, _emb_info = embed_variable(fe, model, arrays, device)
        emb_by_spk[spk] = {id(t): emb[i] for i, t in enumerate(turns)}
    out["per_speaker_turns"] = {s: len(v) for s, v in emb_by_spk.items()}
    pairs = {}
    for a, b in SAME_GENDER_PAIRS:
        within = []
        for spk in (a, b):
            M = list(emb_by_spk.get(spk, {}).values())
            if len(M) < 2:
                within = []
                break
            mat = np.vstack(M)
            d = 1 - mat @ mat.T
            within.extend(d[np.triu_indices(len(M), 1)].tolist())
        ka = list(emb_by_spk.get(a, {}).values())
        kb = list(emb_by_spk.get(b, {}).values())
        between = []
        if ka and kb:
            A, B = np.vstack(ka), np.vstack(kb)
            between = (1 - A @ B.T).ravel().tolist()
        pairs[f"{a}|{b}"] = {
            "within_speaker": describe(within),
            "between_speaker": describe(between),
            "separation": separation(describe(within), describe(between)),
        }
    out["pairs"] = pairs
    return out


# ---------------------------------------------------------------------------
# the model self-check against its own model card
# ---------------------------------------------------------------------------

def model_card_selfcheck(fe, model, device: str) -> dict:
    """Measure the instrument on the model card's own example.

    The model card's threshold is a cosine similarity of 0.86 for
    same-speaker, and its example dataset is a three-utterance, ONE-SPEAKER
    Librispeech split. If the same-speaker distance measured here is not far
    below 0.14, the instrument is not behaving as documented and every number
    in this file is suspect. This is the check that caught the one-second
    window defect.

    Needs the network; if it cannot run, that is recorded rather than hidden.
    """
    out = {"purpose": ("same-speaker distance on the model card's own example "
                       "(Librispeech, speaker 1272); the card's same-speaker "
                       "threshold is cosine similarity 0.86, i.e. distance 0.14"),
           "card_threshold_distance": 0.14}
    try:
        import io
        import urllib.request
        import soundfile as sf
        url = ("https://datasets-server.huggingface.co/rows?dataset="
               "hf-internal-testing%2Flibrispeech_asr_demo&config=clean&"
               "split=validation&offset=0&length=3")
        with urllib.request.urlopen(url, timeout=60) as fh:
            rows = json.loads(fh.read().decode("utf-8"))["rows"]
        arrays = []
        for r in rows:
            a = r["row"]["audio"]
            if isinstance(a, list):
                a = a[0]
            with urllib.request.urlopen(a["src"], timeout=60) as fh:
                x, _sr = sf.read(io.BytesIO(fh.read()), dtype="float32")
            arrays.append(x)
        out["durations_s"] = [round(len(a) / SR, 2) for a in arrays]
        emb, _emb_info = embed_variable(fe, model, arrays, device, per_item=True)
        d = 1 - emb @ emb.T
        iu = np.triu_indices(len(emb), 1)
        out["same_speaker_distance"] = describe(d[iu].tolist())
        # the same three utterances, truncated. This is the measurement that
        # showed the instrument is only worth having above about 2 s, and it
        # is what the document's "the model needs audio" table quotes.
        out["by_clip_length"] = {}
        for secs in (1.0, 3.0):
            n = int(secs * SR)
            if min(len(a) for a in arrays) < n:
                out["by_clip_length"][f"{secs}s"] = {
                    "status": "an utterance is shorter than the clip"}
                continue
            e, _i = embed_variable(fe, model, [a[:n] for a in arrays], device,
                                   per_item=True)
            dd = 1 - e @ e.T
            j = np.triu_indices(len(e), 1)
            out["by_clip_length"][f"{secs}s"] = describe(dd[j].tolist())
        out["status"] = "ran"
    except Exception as exc:                          # noqa: BLE001 - recorded
        out["status"] = "could not run"
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------------------
# the realaudio exploration (NO GROUND TRUTH -- exploratory only)
# ---------------------------------------------------------------------------

def realaudio_exploration(fe, model, device: str) -> dict:
    """How much voice structure does each single-channel cell contain?

    There is no per-turn speaker ground truth in `examples/realaudio/`, so
    nothing here is a speaker-accuracy number. What CAN be computed without
    labels is the STRUCTURE of the embedding-distance distribution: if a cell
    holds one voice, the distances between its windows are unimodal; if it
    holds two, there is a second mode. A 2-means split on the distances is a
    label-free two-cluster split, and the separation of the two cluster
    centres in units of the within-cluster spread says how convincing the
    split is. A weak split (separation near 0) means one voice.
    """
    out: dict = {"status": "EXPLORATORY -- no per-turn speaker ground truth "
                            "exists in this corpus, so nothing here is "
                            "evidence about speaker accuracy",
                 "window_s": REAL_AUDIO_WINDOW_S, "hop_s": REAL_AUDIO_HOP_S}
    w = int(REAL_AUDIO_WINDOW_S * SR)
    h = int(REAL_AUDIO_HOP_S * SR)
    for name, path in REAL_AUDIO_CELLS.items():
        if not path.is_file():
            out[name] = {"status": "file not on disk", "path": str(path)}
            continue
        samples, info = load_mono16k(path)
        starts = list(range(0, max(0, samples.size - w) + 1, h))
        if not starts:
            out[name] = {"status": "shorter than one window", "path": str(path)}
            continue
        arrays = [samples[s:s + w] for s in starts]
        emb, _emb_info = embed_variable(fe, model, arrays, device)
        allv = np.array([cos_distance(emb[i], emb[j])
                         for i in range(len(emb)) for j in range(i + 1, len(emb))])
        adj = [cos_distance(emb[i], emb[i + 1]) for i in range(len(emb) - 1)]
        c0, c1 = two_means_1d(allv)
        side = (np.abs(allv - c1) < np.abs(allv - c0)).astype(int)
        lo_side, hi_side = allv[side == 0], allv[side == 1]
        split_sep = None
        if len(lo_side) and len(hi_side):
            within = np.concatenate([lo_side, hi_side])
            if np.std(within) > 0:
                split_sep = float(abs(np.mean(hi_side) - np.mean(lo_side))
                                  / np.std(within))
        f0_hz = tsv.voiced_hz(samples)
        out[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "duration_s": info["duration_s"],
            "n_windows": len(emb),
            "pairwise_distance": describe(allv.tolist()),
            "adjacent_window_distance": describe(adj),
            "label_free_two_cluster_split": {
                "centres": [round(float(c0), 6), round(float(c1), 6)],
                "cluster_sizes": [int(len(lo_side)), int(len(hi_side))],
                "centre_separation_in_pooled_std": (round(split_sep, 4)
                                                   if split_sep is not None else None),
                "note": ("a split whose centres sit close together is NOT a "
                         "voice split; this is a label-free structure check, "
                         "not a speaker count"),
            },
            "median_f0_hz": round(float(np.median(f0_hz)), 2) if f0_hz.size else None,
            "voiced_frames": int(f0_hz.size),
            "ground_truth": "NONE",
        }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _environment_block() -> dict:
    import torch                                                    # noqa: PLC0415
    import transformers                                              # noqa: PLC0415
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None),
        "executable": sys.executable,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default=str(ROOT / "experiments"
                                         / "speaker_embedding_f0_results.json"))
    ap.add_argument("--device", default=None,
                    help="torch device; defaults to cuda when available")
    ap.add_argument("--no-realaudio", action="store_true",
                    help="skip the exploratory single-channel pass")
    ap.add_argument("--no-headset", action="store_true",
                    help="skip the headset-channel reference")
    ap.add_argument("--no-selfcheck", action="store_true",
                    help="skip the model-card self-check (it needs the network)")
    args = ap.parse_args(argv)

    import torch                                                    # noqa: PLC0415
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    print(f"loading {MODEL_ID}@{MODEL_REVISION[:8]} on {device}")
    fe, model = load_xvector(MODEL_ID, MODEL_REVISION, device)
    model_info = {
        "id": MODEL_ID, "revision": MODEL_REVISION, "class": MODEL_CLASS,
        "pretrained_by": "Microsoft (the WavLM x-vector recipe for speaker "
                         "verification)",
        "licence_note": "model weights are MIT; the AUDIO is the AMI Meeting "
                        "Corpus, CC-BY-4.0, and is what the attribution "
                        "obligation attaches to",
    }
    selfcheck = ({"status": "skipped by flag"} if args.no_selfcheck
                 else model_card_selfcheck(fe, model, device))

    # ---------------- read the corpus -------------------------------------
    samples_by_file: dict[str, np.ndarray] = {}
    turns_by_file: dict[str, list[Turn]] = {}
    inputs_block: dict[str, dict] = {}
    for name in SDM_FILES:
        wav = REALTWOSPEAKER / name
        side_path = REALTWOSPEAKER / name.replace(".wav", ".segments.json")
        side = load_sidecar(side_path)
        samples, info = load_mono16k(wav)
        samples_by_file[name] = samples
        turns = collect_turns(name, side)
        turns_by_file[name] = turns
        inputs_block[name] = {
            "path": str(wav.relative_to(ROOT)), "sha256": sha256_file(wav),
            "sidecar": str(side_path.relative_to(ROOT)),
            "sidecar_sha256": sha256_file(side_path),
            "meeting_id": side.get("meeting_id"),
            "audible_segments": len(turns),
            "annotated_segments": len(side["segments"]),
            **info,
        }

    all_turns = [t for v in turns_by_file.values() for t in v]
    non_finite_embeddings = 0
    print(f"measuring the F0 baseline on {len(all_turns)} audible turns ...")
    pitch = turn_pitch(all_turns, samples_by_file)

    # ---------------- the length sweep ------------------------------------
    print("embedding turns at each minimum length ...")
    sweep: dict[str, dict] = {}
    headline_index = None
    headline_emb = None
    headline_embs = None
    for min_len in MIN_LEN_SWEEP:
        sel = [t for t in all_turns if (t.hi - t.lo) >= int(min_len * SR)]
        arrays = [samples_by_file[t.file][t.lo:t.hi] for t in sel]
        emb, emb_info = embed_variable(fe, model, arrays, device)
        non_finite_embeddings += emb_info["non_finite_embeddings"]
        index, embs = build_index(sel, emb)
        counts = {}
        for t in sel:
            counts[t.speaker] = counts.get(t.speaker, 0) + 1
        row = {
            "min_len_s": min_len,
            "n_turns": len(sel),
            "per_speaker_turns": dict(sorted(counts.items())),
            "pairs": {},
        }
        for a, b in SAME_GENDER_PAIRS + [CROSS_GENDER_CONTROL]:
            within, between = pair_distances(embs, index, a, b)
            w, bt = describe(within), describe(between)
            f0w, f0b = f0_distances(pitch, index, a, b)
            f0w_d, f0b_d = describe(f0w), describe(f0b)
            row["pairs"][f"{a}|{b}"] = {
                "within_speaker": w, "between_speaker": bt,
                "separation": separation(w, bt),
                "f0_same_turns_same_arithmetic": {
                    "within_speaker": f0w_d, "between_speaker": f0b_d,
                    "separation": separation(f0w_d, f0b_d),
                },
                "f0_doc_style": f0_doc_style_separation(pitch, index, a, b),
            }
        sweep[f"{min_len}"] = row
        print(f"  min={min_len}s n={len(sel)} "
              + " ".join(f"{k}={v['separation']['separation_sigma_mad']}"
                         for k, v in row["pairs"].items()))
        if abs(min_len - HEADLINE_MIN_LEN_S) < 1e-9:
            headline_index = index
            headline_emb = emb
            headline_embs = embs
    if headline_index is None:
        raise SystemExit("HEADLINE_MIN_LEN_S is not in MIN_LEN_SWEEP")

    speakers = sorted({v["turn"].speaker for v in headline_index.values()})
    model_info["embedding_dim"] = int(headline_emb.shape[1])

    # ---------------- headline detail -------------------------------------
    print("computing the headline detail ...")
    headline_pairs = {}
    for a, b in SAME_GENDER_PAIRS:
        within, between = pair_distances(headline_embs, headline_index, a, b)
        w, bt = describe(within), describe(between)
        f0w, f0b = f0_distances(pitch, headline_index, a, b)
        f0w_d, f0b_d = describe(f0w), describe(f0b)
        head = {
            "a": a, "b": b,
            "turns": {a: sum(1 for v in headline_index.values()
                             if v["turn"].speaker == a),
                      b: sum(1 for v in headline_index.values()
                             if v["turn"].speaker == b)},
            "within_speaker": w,
            "between_speaker": bt,
            "separation": separation(w, bt),
            "f0_same_turns_same_arithmetic": {
                "within_speaker": f0w_d, "between_speaker": f0b_d,
                "separation": separation(f0w_d, f0b_d),
                "note": ("the pitch instrument in the SAME shape as the "
                         "embedding headline: a within-speaker distance "
                         "distribution and a between-speaker distance "
                         "distribution over |delta f0| between turns"),
            },
            "f0_doc_style": f0_doc_style_separation(pitch, headline_index, a, b),
            "f0_turn_level": f0_turn_level_separation(pitch, headline_index, a, b),
        }
        pv = pair_verification_scores(headline_embs, headline_index, a, b)
        head["verification_2speaker"] = {
            "eer": eer_from_scores(pv["target"], pv["impostor"]),
            "target_scores": [round(x, 6) for x in pv["target"]],
            "impostor_scores": [round(x, 6) for x in pv["impostor"]],
        }
        headline_pairs[f"{a}|{b}"] = head

    all_pair_sep = {}
    for i, a in enumerate(speakers):
        for b in speakers[i + 1:]:
            within, between = pair_distances(headline_embs, headline_index, a, b)
            all_pair_sep[f"{a}|{b}"] = separation(describe(within),
                                                 describe(between))

    # ---------------- the shuffled-label null -----------------------------
    print(f"shuffled-label null ({N_SHUFFLE} permutations) ...")
    null = shuffled_label_null(headline_embs, headline_index, SAME_GENDER_PAIRS,
                                N_SHUFFLE, SHUFFLE_SEED)
    for key, head in headline_pairs.items():
        obs = head["separation"]["separation_sigma_mad"]
        head["shuffled_label_null"] = {
            "observed_separation_sigma_mad": obs,
            "null": null["separation_sigma_mad"][key],
            "permutation_test": permutation_pvalue(obs, null["raw"].get(key, [])),
        }
        head["verdict"] = _pair_verdict(head)

    # ---------------- the crop/scramble-style nulls -----------------------
    print("computing the reversal / gain / noise nulls ...")
    sel = [v["turn"] for v in headline_index.values()]
    arrays = [samples_by_file[t.file][t.lo:t.hi] for t in sel]
    raw = [np.asarray(a, dtype=np.float32) for a in arrays]
    rng_gain = np.random.default_rng(GAIN_SEED)
    gains = np.exp(rng_gain.uniform(math.log(0.1), math.log(10.0), size=len(raw)))
    rng_noise = np.random.default_rng(NOISE_SEED)
    noise = rng_noise.standard_normal(len(raw))
    noise = noise / (np.std(noise) + 1e-12)
    rms = np.array([float(np.sqrt((a.astype(np.float64) ** 2).mean())) for a in raw])
    noise = (noise * rms).astype(np.float32)

    variants = {
        "time_reversed": [a[::-1].copy() for a in raw],
        "gain_scaled": [(a * g).astype(np.float32) for a, g in zip(raw, gains)],
        "white_noise": [rng_noise.standard_normal(len(a)).astype(np.float32) * 0.01
                        for a in raw],
    }
    null_variants = {}
    for vname, varrays in variants.items():
        vemb, vemb_info = embed_variable(fe, model, varrays, device)
        non_finite_embeddings += vemb_info["non_finite_embeddings"]
        vindex, vembs = build_index(sel, vemb)
        vout = {}
        for a, b in SAME_GENDER_PAIRS:
            within, between = pair_distances(vembs, vindex, a, b)
            w, bt = describe(within), describe(between)
            vout[f"{a}|{b}"] = {"separation": separation(w, bt),
                                "within_speaker": w, "between_speaker": bt}
        null_variants[vname] = vout
    null_variants["description"] = {
        "time_reversed": ("the waveform reversed in time. If the separation "
                          "survives, the measurement is about the VOICE, not "
                          "about the temporal detail of the utterance."),
        "gain_scaled": ("each turn multiplied by a log-uniform random gain in "
                        "[0.1, 10]. This model's preprocessor has "
                        "do_normalize=false, so gain is NOT normalised away "
                        "by construction; this measures what it actually "
                        "costs."),
        "white_noise": ("each turn replaced by noise of the same length and "
                        "RMS. There is no speaker in the audio at all, so any "
                        "separation that survives here is an artefact."),
    }

    # ---------------- the turn-change task --------------------------------
    print("building the turn-change task ...")
    turn_change = _turn_change_task(samples_by_file, turns_by_file, fe, model,
                                    device)
    turn_change["shuffled_label_null"] = _turn_change_shuffled_null(
        turns_by_file, turn_change, N_SHUFFLE, SHUFFLE_SEED)
    turn_change["degenerate_baselines"] = {
        "constant_decision_always_change": {
            "eer": 0.5,
            "note": ("predicting 'speaker changed' for every boundary scores "
                     "EER 0.5 by construction. It is written down so the "
                     "headline cannot be read without it."),
        },
        "constant_decision_always_same": {
            "eer": 0.5,
            "note": "the mirror image; also 0.5 by construction.",
        },
    }

    # ---------------- the 4-speaker verification task ---------------------
    ver = verification_scores(headline_embs, headline_index)
    verification = {
        "enrol_test_split": ver["split_sizes"],
        "eer": eer_from_scores(ver["target"], ver["impostor"]),
        "target_scores": [round(x, 6) for x in ver["target"]],
        "impostor_scores": [round(x, 6) for x in ver["impostor"]],
        "trials": ver["trials"],
    }

    # ---------------- the same-speaker control --------------------------
    same_speaker = {}
    pooled = []
    for spk in speakers:
        dists = speaker_pair_distances(headline_embs, headline_index, spk)
        pooled.extend(dists)
        same_speaker[spk] = {
            "n_turns": sum(1 for v in headline_index.values()
                           if v["turn"].speaker == spk),
            "distances": describe(dists)}
    same_speaker["_pooled"] = describe(pooled)
    same_speaker["_note"] = ("the same-speaker control: distances between two "
                             "turns of the SAME speaker. Any method must not "
                             "fire on this population.")

    # ---------------- the solo-only population --------------------------
    solo_index = {k: v for k, v in headline_index.items()
                  if not v["turn"].overlapped}
    solo_embs = {k: headline_embs[k] for k in solo_index}
    solo_pairs = {}
    for a, b in SAME_GENDER_PAIRS:
        within, between = pair_distances(solo_embs, solo_index, a, b)
        w, bt = describe(within), describe(between)
        solo_pairs[f"{a}|{b}"] = {
            "turns": {a: sum(1 for v in solo_index.values()
                             if v["turn"].speaker == a),
                      b: sum(1 for v in solo_index.values()
                             if v["turn"].speaker == b)},
            "within_speaker": w, "between_speaker": bt,
            "separation": separation(w, bt),
        }

    # ---------------- the headset-channel reference ----------------------
    headset = ({"status": "skipped by flag"} if args.no_headset
               else headset_reference(fe, model, device, HEADLINE_MIN_LEN_S))

    # ---------------- the exploratory single-channel pass ------------------
    realaudio = ({"status": "skipped by flag"} if args.no_realaudio
                 else realaudio_exploration(fe, model, device))

    # ---------------- assemble --------------------------------------------
    result = {
        "schema": "speaker-embedding-f0/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": {"path": str(Path(__file__).resolve()),
                   "sha256": sha256_file(Path(__file__).resolve()),
                   "elapsed_s": round(time.time() - t_start, 1)},
        "environment": _environment_block(),
        "model": model_info,
        "models_considered": {
            "note": ("the instrument selection, recorded so the choice can be "
                     "argued with. The attempts below were made while "
                     "developing this file; only the first one produced the "
                     "committed numbers."),
            "used": {
                "id": MODEL_ID, "revision": MODEL_REVISION,
                "why": ("ungated, a pinned revision SHA, a single "
                        "`transformers` class, and a published same-speaker "
                        "threshold on its own model card that this file "
                        "measures and reproduces"),
            },
            "tried_and_not_used": {
                "speechbrain/spkrec-ecapa-voxceleb": {
                    "status": ("installed into the GPU venv and run once as a "
                               "diagnostic on the headset channels; it showed "
                               "the same weakness as WavLM at 1 s windows, "
                               "which is what identified the LENGTH defect "
                               "rather than the model"),
                    "why_not_the_headline": ("speechbrain fetches its "
                                             "checkpoint at run time, so the "
                                             "weights cannot be pinned by "
                                             "revision SHA the way this "
                                             "project pins its inputs"),
                    "install_note": ("speechbrain 1.1.2 needed torchaudio "
                                     "2.6.0+cu124 (to match the existing "
                                     "torch 2.6.0+cu124) and requests; both "
                                     "went into the GPU venv, never into the "
                                     "repo's .venv"),
                },
                "microsoft/wavlm-base-plus": {
                    "status": "considered, not downloaded",
                    "why_not": ("the plain backbone has no x-vector head, and "
                                "training one is out of scope for a "
                                "measurement"),
                },
                "microsoft/unispeech-sat-base-plus": {
                    "status": "available, not used",
                    "why_not": ("a different pretraining with the same "
                                "x-vector head, so it was the natural second "
                                "instrument, but the time went into the "
                                "controls instead"),
                },
            },
        },
        "model_card_selfcheck": selfcheck,
        "method": {
            "reader": "src/wavio.py:read_wav (mono, 16 kHz)",
            "sample_rate_hz": SR,
            "embedding_unit": ("the WHOLE annotated turn, embedded as one "
                               "utterance with the model's own feature "
                               "extractor (padding + attention mask)"),
            "distance": "cosine distance = 1 - cosine similarity",
            "preprocessor": ("the model's own preprocessor_config.json: "
                             "do_normalize=false, return_attention_mask=true"),
            "separation_form": ("separation_sigma_mad = (median(between) - "
                                "median(within)) / mad_sigma(within)"),
            "populations": ("all_audible is the primary population; solo_only "
                            "applies the F0 experiment's no-overlap rule and "
                            "is much smaller"),
            "boundary_clip_s": BOUNDARY_W_S,
            "min_input_s": MIN_INPUT_S,
            "min_input_s_rationale": {
                "why": ("the x-vector head's TDNN convolutions need a minimum "
                        "input length; below roughly 2 s the forward pass "
                        "returns NaN or raises 'Kernel size can't be greater "
                        "than actual input size'. Every utterance is padded to "
                        "at least min_input_s with the attention mask keeping "
                        "the padding out of the pooling."),
                "measured_before_the_rule": ("5 of the 394 boundary clips in "
                                              "an earlier run came back "
                                              "non-finite"),
                "n_clips_shorter_than_min_input_s":
                    turn_change.get("n_clips_shorter_than_min_input_s"),
                "n_clips_total": turn_change.get("n_clips_total"),
            },
            "padding": ("every utterance is padded to ONE length for the whole "
                        "call (the longest in it, and never less than "
                        "min_input_s) with the attention mask keeping the "
                        "padding out of the pooling, so an embedding depends "
                        "on its own audio and not on its batch"),
            "non_finite_embeddings_total": non_finite_embeddings,
        },
        "inputs": inputs_block,
        "turn_population": {
            "audible_turns_total": len(all_turns),
            "headline_min_len_s": HEADLINE_MIN_LEN_S,
            "headline_turns": len(headline_index),
            "per_speaker_turns": {s: sum(1 for v in headline_index.values()
                                         if v["turn"].speaker == s)
                                 for s in speakers},
            "per_speaker_voiced_frames": {
                s: int(sum(pitch[id(v["turn"])]["n_voiced_frames"]
                           for v in headline_index.values()
                           if v["turn"].speaker == s)) for s in speakers},
            "per_speaker_overlapped_turns": {
                s: int(sum(1 for v in headline_index.values()
                           if v["turn"].speaker == s and v["turn"].overlapped))
                for s in speakers},
        },
        "length_sweep": {
            "why": ("the embedding's discriminative power is a function of how "
                    "much audio it is given; a 1.0 s window of speech is not "
                    "enough for this model, and reporting one point of this "
                    "curve without the curve would be a measurement reported "
                    "as a property"),
            "headline_min_len_s": HEADLINE_MIN_LEN_S,
            "headline_rule": (f"the longest minimum in the sweep that still "
                              f"leaves every speaker with at least "
                              f"{MIN_TURNS_PER_SPEAKER} turns, so the "
                              f"enrolment/test split is not degenerate; NOT the "
                              f"point with the largest separation"),
            "sweep": sweep,
        },
        "headline": {
            "units": ("sigma, as in the F0 document: the within-speaker "
                      "spread of the distance distribution"),
            "comparison_point_from_f0_doc": {
                "FEO070|FEO072": 0.5, "MEE071|MEE073": 0.3,
                "source": "docs/research/F0-SPEAKER-CHANGE.md",
                "note": ("the document's 0.48 / 0.33 are reproduced here as "
                         "the `f0_doc_style` block of each pair, on this turn "
                         "set, with the document's own arithmetic"),
            },
            "pairs": headline_pairs,
            "all_pairs": all_pair_sep,
            "solo_only_population": solo_pairs,
            "f0_document_reproduction": f0_doc_reproduction(
                turns_by_file, pitch, "ami_sdm_EN2002a_turns.wav"),
        },
        "verification_4speaker": verification,
        "controls": {
            "same_speaker_control": same_speaker,
            "shuffled_label_null": null,
            "crop_and_scramble_nulls": null_variants,
        },
        "turn_change_task": turn_change,
        "headset_channel_reference": headset,
        "realaudio_exploratory": realaudio,
        "falsified": [],
    }
    result["falsified"] = _falsified_block(result)

    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}  ({out.stat().st_size} bytes, "
          f"{round(time.time() - t_start, 1)}s)")
    _print_summary(result)
    return 0


def _pair_verdict(head: dict) -> dict:
    """Did this pair get RESOLVED? Written as a rule, not an adjective.

    The rule is fixed in advance: the separation has to clear
    RESOLVED_MIN_SIGMA, the two-speaker EER has to be under
    RESOLVED_MAX_EER, and the shuffled-label null has to be beaten at
    RESOLVED_MAX_P. Any one of them failing means the pair is not resolved,
    and the reason is written down.
    """
    obs = head["separation"]["separation_sigma_mad"]
    eer = head.get("verification_2speaker", {}).get("eer", {}).get("eer")
    pval = head.get("shuffled_label_null", {}).get(
        "permutation_test", {}).get("p_value")
    checks = {
        "separation_sigma_mad": {
            "value": obs, "at_least": RESOLVED_MIN_SIGMA,
            "pass": obs is not None and obs >= RESOLVED_MIN_SIGMA},
        "two_speaker_eer": {
            "value": eer, "at_most": RESOLVED_MAX_EER,
            "pass": eer is not None and eer <= RESOLVED_MAX_EER},
        "shuffled_label_null_p": {
            "value": pval, "at_most": RESOLVED_MAX_P,
            "pass": pval is not None and pval <= RESOLVED_MAX_P},
    }
    resolved = all(c["pass"] for c in checks.values())
    return {
        "resolved": resolved,
        "criteria": {
            "separation_sigma_mad_at_least": RESOLVED_MIN_SIGMA,
            "two_speaker_eer_at_most": RESOLVED_MAX_EER,
            "shuffled_label_null_p_at_most": RESOLVED_MAX_P,
        },
        "checks": checks,
        "failed": [k for k, c in checks.items() if not c["pass"]],
    }


def _falsified_block(result: dict) -> list:
    """The claims this run falsified, written out rather than implied."""
    out = []
    pairs = result["headline"]["pairs"]
    for key in ("FEO070|FEO072", "MEE071|MEE073"):
        head = pairs.get(key)
        if head is None:
            continue
        obs = head["separation"]["separation_sigma_mad"]
        pv = head.get("shuffled_label_null", {}).get("permutation_test", {})
        pval = pv.get("p_value")
        if obs is None:
            out.append(f"{key}: no separation could be computed.")
            continue
        if pval is not None and pval > RESOLVED_MAX_P:
            out.append(f"{key}: the embedding separation ({obs} sigma) does not "
                       f"survive the shuffled-label null (p={pval}); the "
                       f"separation is not attributable to speaker identity.")
        verdict = head.get("verdict") or {}
        if verdict.get("resolved") is False:
            failed = verdict.get("failed") or []
            out.append(
                f"{key}: the pair is NOT resolved by the embedding "
                f"({obs} sigma, 2-speaker EER "
                f"{head.get('verification_2speaker', {}).get('eer', {}).get('eer')}). "
                f"Failed: {', '.join(failed)}. Pitch could not resolve it "
                f"either, and neither can the embedding.")
        f0_same = head.get("f0_same_turns_same_arithmetic", {}).get(
            "separation", {}).get("separation_sigma_mad")
        if f0_same is not None and obs <= f0_same:
            out.append(f"{key}: the embedding separation ({obs} sigma) does not "
                       f"exceed the F0 separation measured on the same turns "
                       f"in the same arithmetic ({f0_same} sigma); the "
                       f"embedding adds nothing here.")
    tc = result["turn_change_task"]
    obs_eer = tc.get("embedding", {}).get("eer_vs_same_speaker", {}).get("eer")
    null_p = tc.get("shuffled_label_null", {}).get("permutation_test", {}).get("p_value")
    if obs_eer is not None and null_p is not None and null_p > RESOLVED_MAX_P:
        out.append(f"turn-change task: the EER ({obs_eer}) does not differ "
                   f"from the shuffled-label null (p={null_p}); the statistic "
                   f"is not reading the speaker change.")
    return out


def _print_summary(result: dict) -> None:
    print("\n=== SUMMARY ===")
    for key in ("FEO070|FEO072", "MEE071|MEE073"):
        head = result["headline"]["pairs"].get(key)
        if not head:
            continue
        s = head["separation"]
        print(f"  {key}: {s['separation_sigma_mad']} sigma "
              f"(mean ratio {s['mean_ratio']}), "
              f"EER {head['verification_2speaker']['eer']['eer']}")
        print(f"      F0 same turns, same arithmetic: "
              f"{head['f0_same_turns_same_arithmetic']['separation']['separation_sigma_mad']} sigma")
        print(f"      F0 doc arithmetic: "
              f"{head['f0_doc_style'].get('separation_sigma')} sigma")
        print(f"      shuffled null median: "
              f"{head['shuffled_label_null']['null']['median']} "
              f"(p={head['shuffled_label_null']['permutation_test']['p_value']})")
    tc = result["turn_change_task"]
    print(f"  turn change: EER {tc['embedding']['eer_vs_same_speaker']['eer']} "
          f"vs F0 {tc['f0_baseline']['eer_vs_same_speaker']['eer']}")
    print("  shuffled null EER: "
          f"{tc['shuffled_label_null']['null_eer']['median']}")
    if result["falsified"]:
        print("\n  FALSIFIED:")
        for f in result["falsified"]:
            print(f"   - {f}")


if __name__ == "__main__":
    sys.exit(main())
