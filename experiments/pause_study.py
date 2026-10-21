#!/usr/bin/env python3
"""Pause study: what a pause is, measured on the corpus that is on disk.

The wake-and-turn brief (`docs/research/BRIEF-WAKE-AND-TURN.md`) asks the
endpointer to stop treating silence as absence and start treating it as a
cue.  The brief names three kinds of pause -- THINKING (the speaker is
mid-thought and will continue), BEAT or BREATHING (a pause inside a phrase,
no thought change), and TERMINAL (the speaker has finished and the floor
may be yielded) -- and it names an explicit transmit word, OVER.

This script measures the pauses.  It does not build an endpointer and it
does not claim a classifier; it produces the descriptive statistics the
endpointer design needs, and it measures how far a *first* separation of
thinking from terminal can be pushed with the evidence that exists.

THE CORPUS, and what each part can and cannot do:

| corpus | what it is | what it gives |
|---|---|---|
| script pair | `/mnt/c/proart_capture/A.wav` and `B.wav`: two INDEPENDENT readings of the SAME written passage, plus `C.wav` (silent) and `empty1.wav` (empty room) | the only GROUND TRUTH available: the script says where a sentence ends and where a clause ends, so a pause can be labelled TERMINAL, BEAT or INTERNAL without a human raters' judgement |
| free sessions | tonight's `browser_free_record` sessions with real audio | free speech: long, informal, full of incomplete syntax -- the population the endpointer actually runs on, with NO ground truth |
| protocol design | `examples/realdialogue/README.md`: eight scenes, 1230 s | the DESIGN of the corpus that would give a labelled thinking pause (scene 3) and a labelled negative control (scene 7).  NOT RECORDED: `examples/realdialogue/raw/` is empty |

The script corpus is the calibration set and the free corpus is the
transfer set.  The two are never averaged together, for the same reason the
DRILLED and AMBIENT classes are never averaged together: one is a
performed reading, the other is first-take speech.

THE METHOD for finding pauses is the repository's own, byte-identical to
`experiments/pause_threshold.py` and therefore comparable with the table in
`docs/research/PAUSE-THRESHOLD.md`:

    energy.rms_series(samples, 800, 160, 16000)
    energy.segment(frames, 0.01, peak_rms=max(f.rms for f in frames))

Prosody is the repository's own YIN tracker (`src/pitch.py`), run on a
window around each pause.  Syntax comes from a LOCAL faster-whisper
`small.en` on CPU with word timestamps; the words are assigned to the
energy-measured speech segments by a MONOTONIC sweep, because whisper's
word timestamps are far too coarse to place a pause inside a word (they
are off by up to a third of a second) while its word ORDER is exact.

Nothing here calls a paid API.  The device is reported in the results.

    python experiments/pause_study.py --out docs/research
    python experiments/pause_study.py --selftest
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import energy, wavio  # noqa: E402

# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------

CAPTURE_ROOT = Path("/mnt/c/proart_capture")
FREE_ROOT = CAPTURE_ROOT / "corpus" / "free_sessions"

# The written passage both A and B read.  It is transcribed here from the
# two ASR passes, which agree with each other; where they differ in
# wording ("9" vs "nine", "the long bar" vs "a long bar") the difference is
# in the ASR, not in the script, and the normaliser below folds the
# numeral away.
SCRIPT_TEXT = (
    "The library opened at nine, and the light came through the east window "
    "in the long bar across the floor. "
    "A man in a gray coat sat near the window with a cup of coffee he did not "
    "drink. "
    "He turned a page, then another, and he did not look up. "
    "Outside, a bus pulled away from the curb, and the sound of it faded down "
    "the street."
)

SCRIPT_TOKENS = SCRIPT_TEXT.split()

SENTENCE_END = frozenset(i for i, t in enumerate(SCRIPT_TOKENS)
                          if t.endswith((".", "!", "?")))
CLAUSE_END = frozenset(i for i, t in enumerate(SCRIPT_TOKENS)
                       if t.endswith((",", ";", ":")))

# The clips, and what each one is for.
SCRIPT_CLIPS = [
    ("A", CAPTURE_ROOT / "A.wav", "script", "facing the camera, speaking"),
    ("B", CAPTURE_ROOT / "B.wav", "script", "head turned away, speaking"),
    ("C", CAPTURE_ROOT / "C.wav", "silent", "facing the camera, silent"),
    ("empty1", CAPTURE_ROOT / "empty1.wav", "silent", "nobody talking"),
]

# Words that leave an utterance HANGING.  A pause after one of these is
# the speaker mid-clause by construction, whatever the prosody does.
# The list is a closed class: determiners, prepositions, conjunctions,
# auxiliaries, pronouns and common degree words.
FUNCTION_WORDS = frozenset("""
a an the this that these those there here
and or but nor so yet for if then than because although while whereas
of in on at by to from with without within into onto over under about
into as per via
am is are was were be been being being
have has had having do does did doing
will would shall should can could may might must
i me my mine you your yours he him his she her hers it its we us our ours
they them their theirs who whom whose which what
not no nor very too also just only even still yet ever never always
some any all both each every many much more most other another such
one two three
""".split())

# A pause is "interior" when speech exists on both sides.
# The edge silences are kept but labelled: a leading silence is the
# recording starting, a trailing silence is the recording stopping.
PRE_WINDOW_S = 1.0     # prosody measured over this much speech before a pause
POST_WINDOW_S = 1.0    # ... and this much after it
PROSODY_SPAN_S = 0.30  # the slope is fitted over this much at the edge

ENERGY_FRAME = 800
ENERGY_HOP = 160
ANALYSIS_SR = 16000


def _norm(tok: str) -> str:
    """Normalise a token for alignment: lowercase, letters and digits only."""
    t = re.sub(r"[^a-z0-9']", "", tok.lower())
    return {"9": "nine"}.get(t, t)


SCRIPT_NORM = [_norm(t) for t in SCRIPT_TOKENS]


# --------------------------------------------------------------------------
# Small statistics, with the conventions written down
# --------------------------------------------------------------------------

def pctl(values, q: float) -> float | None:
    """Nearest-rank percentile, the convention `PAUSE-THRESHOLD.md` uses."""
    if not values:
        return None
    s = sorted(values)
    k = max(1, math.ceil(q * len(s)))
    return s[k - 1]


def median(values) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def describe(values) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "p05": pctl(values, 0.05),
        "p25": pctl(values, 0.25),
        "median": median(values),
        "p75": pctl(values, 0.75),
        "p95": pctl(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Rank AUC with ties counted as halves.  None if a class is empty."""
    if not pos or not neg:
        return None
    vals = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    vals.sort(key=lambda t: t[0])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1][0] == vals[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r_pos = sum(r for r, (_, lab) in zip(ranks, vals) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def auc_ci(pos: list[float], neg: list[float], n_boot: int = 2000,
           seed: int = 20261021) -> dict:
    """Bootstrap interval for an AUC.  The point estimate is the AUC."""
    point = auc(pos, neg)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        p = [rng.choice(pos) for _ in pos]
        q = [rng.choice(neg) for _ in neg]
        a = auc(p, q)
        if a is not None:
            draws.append(a)
    draws.sort()
    return {
        "auc": point,
        "ci90_lo": pctl(draws, 0.05),
        "ci90_hi": pctl(draws, 0.95),
        "n_boot": len(draws),
    }


def best_threshold_accuracy(pos: list[float], neg: list[float]) -> dict:
    """The best single cut, and what a single cut scores when it is chosen
    on all the data.  The leave-one-out number is the honest one: the
    in-sample number is what a rule would look like if it had been fit on
    the very pauses it is graded on.

    TWO thresholds are reported, because they are different instruments and
    only one of them is usable:

    * `threshold` is the in-sample best cut.  On a small positive class it
      routinely lands on an OUTLIER -- on this corpus the pre-pause slope's
      in-sample cut is -81.8 st/s, which separates one extreme internal
      pause and no terminal one.  It is reported because the overfit is
      worth seeing, and it must not be used as a design threshold.
    * `robust_threshold` is the midpoint between the two class medians.
      It is stable, it is what a designer would actually ship, and its
      accuracy is reported next to it."""
    if not pos or not neg:
        return {}
    vals = sorted(set(pos) | set(neg))
    cuts = [v - 1e-9 for v in vals] + [vals[-1] + 1e-9]

    def acc_at(cut, sign):
        ok = 0
        for v in pos:
            ok += 1 if (sign * v > sign * cut) else 0
        for v in neg:
            ok += 1 if (sign * v <= sign * cut) else 0
        return ok / (len(pos) + len(neg))

    best = None
    for cut in cuts:
        for sign in (1, -1):
            a = acc_at(cut, sign)
            if best is None or a > best["accuracy_in_sample"]:
                best = {"threshold": cut, "sign": sign, "accuracy_in_sample": a}

    # leave-one-out: choose the cut on all but one pause, grade the one left out
    correct = 0
    for i in range(len(pos) + len(neg)):
        p2 = [v for j, v in enumerate(pos) if j != i] if i < len(pos) else pos
        n2 = ([v for j, v in enumerate(neg) if j != i - len(pos)]
              if i >= len(pos) else neg)
        if not p2 or not n2:
            continue
        bp = None
        for cut in sorted(set(p2) | set(n2)):
            for sign in (1, -1):
                a = acc_at_cut(cut, sign, p2, n2)
                if bp is None or a > bp[0]:
                    bp = (a, cut, sign)
        if bp is None:
            continue
        _, cut, sign = bp
        v = pos[i] if i < len(pos) else neg[i - len(pos)]
        is_pos = i < len(pos)
        pred_pos = (sign * v > sign * cut)
        if pred_pos == is_pos:
            correct += 1
    total = len(pos) + len(neg)

    mp = median(pos)
    mn = median(neg)
    robust_cut = 0.5 * (mp + mn)
    robust_sign = 1 if mp > mn else -1
    return {
        "threshold": best["threshold"],
        "sign": best["sign"],
        "accuracy_in_sample": best["accuracy_in_sample"],
        "accuracy_leave_one_out": correct / total if total else None,
        "robust_threshold": robust_cut,
        "robust_sign": robust_sign,
        "accuracy_at_robust": acc_at(robust_cut, robust_sign),
        "pos_median": mp,
        "neg_median": mn,
    }


def acc_at_cut(cut, sign, pos, neg) -> float:
    ok = sum(1 for v in pos if sign * v > sign * cut)
    ok += sum(1 for v in neg if sign * v <= sign * cut)
    return ok / (len(pos) + len(neg))


def semitones(hz: float | None) -> float | None:
    if hz is None or hz <= 0:
        return None
    return 12.0 * math.log2(hz)


def slope_semitones_per_s(times, hz) -> float | None:
    """Least-squares slope of f0 in semitones against time."""
    pts = [(t, semitones(h)) for t, h in zip(times, hz)
           if h is not None and h > 0 and semitones(h) is not None]
    if len(pts) < 3:
        return None
    n = len(pts)
    mt = statistics.fmean(p[0] for p in pts)
    ms = statistics.fmean(p[1] for p in pts)
    num = sum((p[0] - mt) * (p[1] - ms) for p in pts)
    den = sum((p[0] - mt) ** 2 for p in pts)
    if den <= 0:
        return None
    return num / den


# --------------------------------------------------------------------------
# Audio in, pauses out
# --------------------------------------------------------------------------

@dataclass
class Pause:
    index: int
    start_s: float
    end_s: float
    duration_s: float
    position: str = "interior"     # leading | interior | trailing
    script_class: str | None = None  # terminal | beat | internal | None
    pre_text: str = ""
    post_text: str = ""
    pre_punct: str = ""
    pre_function_word: bool | None = None
    post_function_word: bool | None = None
    post_lowercase: bool | None = None
    f0_pre_last: float | None = None
    f0_pre_slope_st_s: float | None = None
    f0_pre_mean: float | None = None
    f0_pre_range_st: float | None = None
    f0_post_first: float | None = None
    f0_post_slope_st_s: float | None = None
    f0_jump_st: float | None = None
    f0_pre_phrase_mean: float | None = None
    n_words_pre: int = 0
    n_words_post: int = 0
    script_pre_tok: int | None = None
    script_post_tok: int | None = None


def read_clip(path: Path):
    audio = wavio.read_wav(str(path))
    return audio, list(audio.samples)


def segment_audio(samples, hop_s: float = 0.01):
    """The repository's own segmentation, byte-identical to the method
    `docs/research/PAUSE-THRESHOLD.md` publishes."""
    ef, _ = energy.rms_series(samples, ENERGY_FRAME, ENERGY_HOP, ANALYSIS_SR)
    if not ef:
        return [], []
    peak = max(f.rms for f in ef)
    segs = energy.segment(ef, hop_s, peak_rms=peak)
    return segs, ef


def assign_words_to_speech(words, speech_segs):
    """Monotonic assignment of ASR words to the energy-measured speech
    segments.

    Whisper's word timestamps are too coarse to place a pause inside a
    word -- they are off by up to a third of a second -- but its word
    ORDER is exact.  So the words are swept onto the speech segments in
    order: a word belongs to the first segment that has not yet ended
    before the word starts.  A word whose timestamp lands inside a pause
    therefore joins the speech that came before it, which is the
    behaviour that repairs a late timestamp.

    Returns the word INDEX in each segment, so the caller can map a word
    back onto the script without searching for an equal dict."""
    out = {i: [] for i in range(len(speech_segs))}
    seg_idx = 0
    for wi, w in enumerate(words):
        while (seg_idx < len(speech_segs)
               and speech_segs[seg_idx][1] < w["start"] - 0.05):
            seg_idx += 1
        if seg_idx >= len(speech_segs):
            seg_idx = len(speech_segs) - 1
        if seg_idx < 0:
            continue
        out[seg_idx].append(wi)
    return out


def align_to_script(words):
    """Map ASR words onto script token indices.  Returns (mapping, ratio)."""
    seq = [_norm(w["text"]) for w in words]
    keep = [(i, s) for i, s in enumerate(seq) if s]
    if not keep:
        return {}, 0.0
    sm = difflib.SequenceMatcher(a=[s for _, s in keep], b=SCRIPT_NORM,
                                  autojunk=False)
    mapping = {}
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            mapping[keep[i + k][0]] = j + k
    return mapping, sm.ratio()


def script_class_for(pre_tok: int | None, post_tok: int | None) -> str | None:
    """TERMINAL when a sentence boundary falls in the gap between the last
    word before the pause and the first word after it; BEAT when a clause
    boundary falls there; INTERNAL otherwise (the reader stopped where the
    script did not tell them to)."""
    if pre_tok is None or post_tok is None or post_tok <= pre_tok:
        return None
    window = range(pre_tok, post_tok)
    if any(k in SENTENCE_END for k in window):
        return "terminal"
    if any(k in CLAUSE_END for k in window):
        return "beat"
    return "internal"


def track_pitch_window(samples, start_s: float, end_s: float, sr: int):
    """Run the repository's YIN tracker over a window and return frames
    with times rebased to the clip."""
    from src.pitch import PitchTracker
    i0 = max(0, int((start_s) * sr))
    i1 = min(len(samples), int((end_s) * sr) + 1)
    if i1 - i0 < 800:
        return []
    tracker = PitchTracker(sr)
    frames = tracker.track(samples[i0:i1])
    return [{"t": f.time + start_s, "hz": f.hz, "conf": f.confidence}
            for f in frames]


def pitch_features(frames, pause_start: float, pause_end: float) -> dict:
    """Prosody at the two edges of the pause."""
    out = {k: None for k in (
        "f0_pre_last", "f0_pre_slope_st_s", "f0_pre_mean", "f0_pre_range_st",
        "f0_post_first", "f0_post_slope_st_s", "f0_jump_st",
        "f0_pre_phrase_mean")}

    def window(t0, t1):
        return [f for f in frames if t0 <= f["t"] < t1 and f["hz"]]

    pre = window(pause_start - PROSODY_SPAN_S, pause_start)
    post = window(pause_end, pause_end + PROSODY_SPAN_S)
    phrase = window(pause_start - PRE_WINDOW_S, pause_start)

    if pre:
        out["f0_pre_last"] = pre[-1]["hz"]
        out["f0_pre_mean"] = statistics.fmean(f["hz"] for f in pre)
        st = [semitones(f["hz"]) for f in pre]
        out["f0_pre_range_st"] = max(st) - min(st)
        sl = slope_semitones_per_s([f["t"] for f in pre], [f["hz"] for f in pre])
        out["f0_pre_slope_st_s"] = sl
    if post:
        out["f0_post_first"] = post[0]["hz"]
        sl = slope_semitones_per_s([f["t"] for f in post], [f["hz"] for f in post])
        out["f0_post_slope_st_s"] = sl
    if out["f0_pre_last"] and out["f0_post_first"]:
        out["f0_jump_st"] = (semitones(out["f0_post_first"])
                             - semitones(out["f0_pre_last"]))
    if phrase:
        out["f0_pre_phrase_mean"] = statistics.fmean(f["hz"] for f in phrase)
    return out


def punct_of(word_text: str) -> str:
    m = re.search(r"[.,!?;:]+$", word_text.strip())
    return m.group(0) if m else ""


def base_of(word_text: str) -> str:
    return re.sub(r"[^A-Za-z']", "", word_text).lower()


def analyse_clip(name: str, path: Path, role: str, note: str,
                 words: list[dict], meta: dict | None = None) -> dict:
    """Measure every pause in one clip."""
    audio, samples = read_clip(path)
    sr = audio.sample_rate
    dur = len(samples) / float(sr)
    segs, ef = segment_audio(samples)
    speech_segs = [(s.start, s.end) for s in segs if s.speech]
    silences = [(s.start, s.end) for s in segs if not s.speech]

    words = sorted(words, key=lambda w: w["start"])
    mapping, ratio = align_to_script(words)
    assigned = assign_words_to_speech(words, speech_segs)

    # which speech segment does each pause sit between?
    pauses: list[Pause] = []
    for i, (t0, t1) in enumerate(silences):
        before_idx = None
        after_idx = None
        for j, (s0, s1) in enumerate(speech_segs):
            if s1 <= t0 + 1e-6:
                before_idx = j
            if s0 >= t1 - 1e-6 and after_idx is None:
                after_idx = j
        if before_idx is None:
            position = "leading"
        elif after_idx is None:
            position = "trailing"
        else:
            position = "interior"

        pre_idx = assigned.get(before_idx, []) if before_idx is not None else []
        post_idx = assigned.get(after_idx, []) if after_idx is not None else []

        p = Pause(index=i, start_s=t0, end_s=t1, duration_s=t1 - t0,
                  position=position)
        p.n_words_pre = len(pre_idx)
        p.n_words_post = len(post_idx)
        if pre_idx:
            last = words[pre_idx[-1]]
            p.pre_text = last["text"].strip()
            p.pre_punct = punct_of(last["text"])
            base = base_of(last["text"])
            p.pre_function_word = base in FUNCTION_WORDS if base else None
            p.script_pre_tok = mapping.get(pre_idx[-1])
        if post_idx:
            first = words[post_idx[0]]
            p.post_text = first["text"].strip()
            base = base_of(first["text"])
            p.post_function_word = base in FUNCTION_WORDS if base else None
            stripped = first["text"].strip()
            p.post_lowercase = bool(stripped) and stripped[0].islower()
            p.script_post_tok = mapping.get(post_idx[0])

        if role == "script":
            p.script_class = script_class_for(p.script_pre_tok, p.script_post_tok)
            if p.script_class is None and position == "trailing" \
                    and p.script_pre_tok is not None:
                # the recording stopped at a sentence end: still terminal,
                # but it is an edge pause and the tables keep it apart
                p.script_class = "terminal" if p.script_pre_tok in SENTENCE_END \
                    else "internal"
        pauses.append(p)

    # prosody: the YIN tracker is run on the speech either side of the
    # pause, in two windows of PRE/POST seconds.  The silence itself is
    # never tracked -- a long pause would otherwise cost minutes of YIN
    # to learn nothing, because silence has no pitch.
    for p in pauses:
        frames = []
        w0 = max(0.0, p.start_s - PRE_WINDOW_S)
        frames += track_pitch_window(samples, w0, p.start_s, sr)
        frames += track_pitch_window(samples, p.end_s,
                                     min(dur, p.end_s + POST_WINDOW_S), sr)
        feats = pitch_features(frames, p.start_s, p.end_s)
        for k, v in feats.items():
            setattr(p, k, v)

    return {
        "name": name,
        "path": str(path),
        "role": role,
        "note": note,
        "meta": meta or {},
        "duration_s": dur,
        "source_sample_rate": audio.source_sample_rate,
        "resampled": audio.source_sample_rate != sr,
        "n_frames_energy": len(ef),
        "peak_rms": max((f.rms for f in ef), default=0.0),
        "n_speech_segments": len(speech_segs),
        "n_silences": len(silences),
        "script_align_ratio": ratio,
        "n_words": len(words),
        "words": words,
        "pauses": [asdict(p) for p in pauses],
    }


# --------------------------------------------------------------------------
# ASR, cached
# --------------------------------------------------------------------------

def transcribe_clip(path: Path, model_name: str = "small.en") -> tuple[list[dict], dict]:
    from faster_whisper import WhisperModel
    audio, samples = read_clip(path)
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size < 1600:          # under a tenth of a second: nothing to say
        return [], {"engine": "faster-whisper", "model": model_name,
                    "device": "none", "n_words": 0,
                    "note": "clip shorter than 0.1 s"}
    try:
        import ctranslate2
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:  # noqa: BLE001
        device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    t0 = time.time()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    load_s = time.time() - t0
    t0 = time.time()
    segments, info = model.transcribe(arr, word_timestamps=True, beam_size=5,
                                      language="en", vad_filter=True)
    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({"text": w.word, "start": float(w.start),
                          "end": float(w.end), "p": float(w.probability)})
    meta = {
        "engine": "faster-whisper",
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "model_load_s": round(load_s, 2),
        "transcribe_s": round(time.time() - t0, 2),
        "n_words": len(words),
    }
    return words, meta


def numpy_array(samples):
    return np.asarray(samples, dtype=np.float32)


# --------------------------------------------------------------------------
# Corpus discovery
# --------------------------------------------------------------------------

def free_sessions() -> list[dict]:
    """Tonight's free sessions, with the class read from the page's own
    trigger in meta.json -- never guessed from the folder name."""
    out = []
    if not FREE_ROOT.is_dir():
        return out
    for d in sorted(FREE_ROOT.iterdir()):
        if not d.is_dir():
            continue
        wav = d / "audio.wav"
        meta_path = d / "meta.json"
        if not wav.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        except Exception:  # noqa: BLE001
            meta = {}
        mode = meta.get("mode")
        trigger = meta.get("trigger")
        if mode == "probe":
            klass = "PROBE"
        elif mode == "manual" or trigger == "manual":
            klass = "DRILLED"
        else:
            klass = "AMBIENT"
        out.append({
            "name": d.name,
            "path": wav,
            "role": "free",
            "note": f"{mode}/{trigger}",
            "meta": {
                "mode": mode,
                "trigger": trigger,
                "close_reason": meta.get("close_reason"),
                "wall_clock_start": meta.get("wall_clock_start"),
                "wall_clock_end": meta.get("wall_clock_end"),
                "n_frames": meta.get("n_frames"),
                "data_class": klass,
            },
        })
    return out


def protocol_design() -> dict:
    """The dialogue protocol as DESIGN: the eight scenes, their durations,
    and the fact that none of it is recorded."""
    scenes = [
        (1, "free conversation", 300, "both", "unpredictable gaps"),
        (2, "question, then answer", 180, "both", "a real hand-off with a real gap"),
        (3, "thinking pause", 180, "both",
         "B thinks silently, 2-10 s of real silence before any word"),
        (4, "turn-holding fillers", 120, "A only",
         "the negative class: long in practice, no silence at all"),
        (5, "backchannel", 180, "both", "a different speaker, near-zero gap"),
        (6, "deliberate interruption", 120, "both", "overlap with ground truth"),
        (7, "one speaker alone, long pauses", 120, "A only",
         "the pure negative control: silence, no hand-off"),
        (8, "room tone", 30, "neither", "the noise floor everything else is measured against"),
    ]
    raw_dir = ROOT.parent / "communicator" / "examples" / "realdialogue" / "raw"
    if not raw_dir.is_dir():
        raw_dir = Path("/home/proart/projects/communicator/examples/realdialogue/raw")
    recorded = []
    if raw_dir.is_dir():
        recorded = sorted(p.name for p in raw_dir.iterdir() if p.is_file())
    return {
        "source": "examples/realdialogue/README.md",
        "raw_dir": str(raw_dir),
        "raw_dir_exists": raw_dir.is_dir(),
        "files_recorded": recorded,
        "n_files_recorded": len(recorded),
        "total_s": sum(s[2] for s in scenes),
        "scenes": [{"n": n, "name": nm, "seconds": sec, "channels": ch, "purpose": why}
                   for n, nm, sec, ch, why in scenes],
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def pause_rows(clips: list[dict]) -> list[dict]:
    rows = []
    for clip in clips:
        for p in clip["pauses"]:
            row = dict(p)
            row["clip"] = clip["name"]
            row["role"] = clip["role"]
            row["data_class"] = clip.get("meta", {}).get("data_class")
            rows.append(row)
    return rows


def feature_table(rows: list[dict], key: str) -> dict:
    """positives / negatives / AUC for one feature."""
    pos, neg = [], []
    for r in rows:
        v = r.get(key)
        if v is None or r.get("script_class") is None:
            continue
        if r["script_class"] == "terminal":
            pos.append(v)
        else:
            neg.append(v)
    if not pos or not neg:
        return {"n_pos": len(pos), "n_neg": len(neg), "auc": None}
    out = {"n_pos": len(pos), "n_neg": len(neg)}
    out.update(auc_ci(pos, neg))
    out.update(best_threshold_accuracy(pos, neg))
    out["pos_median"] = median(pos)
    out["neg_median"] = median(neg)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/research",
                    help="where to write the results JSON")
    ap.add_argument("--results", default=None,
                    help="path of the results JSON (default <out>/pause_study_results.json)")
    ap.add_argument("--no-cache", action="store_true",
                    help="re-transcribe even when the results file has the words")
    ap.add_argument("--reaggregate", action="store_true",
                    help="recompute the tables from the stored pause rows "
                         "without re-running the analysis")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    out_dir = ROOT / args.out if not os.path.isabs(args.out) else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results) if args.results else out_dir / "pause_study_results.json"

    cache = {}
    cache_path = out_dir / "pause_study_asr_cache.json"
    if results_path.is_file() and not args.no_cache:
        try:
            cache = json.loads(results_path.read_text()).get("asr_cache", {})
        except Exception:  # noqa: BLE001
            cache = {}
    if cache_path.is_file() and not args.no_cache:
        try:
            cache.update(json.loads(cache_path.read_text()))
        except Exception:  # noqa: BLE001
            pass

    if args.reaggregate:
        stored = json.loads(results_path.read_text())
        clips = stored["clips"]
        rows = stored["pause_rows"]
        asr_meta = stored.get("asr_meta", {})
        asr_cache = dict(cache)
        print(f"  re-aggregating {len(rows)} stored pauses")
    else:
        # ---- build the clip list -------------------------------------------
        clips_meta = []
        for name, path, role, note in SCRIPT_CLIPS:
            clips_meta.append({"name": name, "path": path, "role": role, "note": note,
                               "meta": {}})
        clips_meta.extend(free_sessions())

        # ---- transcribe (cached) -------------------------------------------
        asr_cache = dict(cache)
        asr_meta = {}
        cache_path = out_dir / "pause_study_asr_cache.json"
        for c in clips_meta:
            key = str(c["path"])
            if key in asr_cache and not args.no_cache:
                continue
            try:
                t0 = time.time()
                words, m = transcribe_clip(c["path"])
                asr_cache[key] = words
                m["total_s"] = round(time.time() - t0, 2)
                asr_meta[c["name"]] = m
                # written as it goes: a failure later must not cost the ASR again
                cache_path.write_text(json.dumps(asr_cache))
                print(f"  transcribed {c['name']}: {len(words)} words "
                      f"({m['device']}, {m['total_s']}s)", flush=True)
            except Exception as exc:  # noqa: BLE001
                asr_meta[c["name"]] = {"error": repr(exc)}
                print(f"  ASR FAILED {c['name']}: {exc}", flush=True)

        # ---- analyse --------------------------------------------------------
        clips = []
        for c in clips_meta:
            words = asr_cache.get(str(c["path"]), [])
            try:
                clips.append(analyse_clip(c["name"], c["path"], c["role"], c["note"],
                                          words, c["meta"]))
            except Exception as exc:  # noqa: BLE001
                print(f"  ANALYSIS FAILED {c['name']}: {exc}", flush=True)
                clips.append({"name": c["name"], "path": str(c["path"]),
                              "role": c["role"], "note": c["note"], "meta": c["meta"],
                              "error": repr(exc), "duration_s": 0.0,
                              "pauses": [], "words": [], "n_words": 0})

        rows = pause_rows(clips)

    results = aggregate(clips, rows, asr_meta, asr_cache)
    # the ASR words live in their own cache file, not duplicated here
    results["asr_cache_file"] = "docs/research/pause_study_asr_cache.json"
    results_path.write_text(json.dumps(results, indent=1))
    print(f"wrote {results_path}")
    print(f"  pauses: {len(rows)} total, "
          f"{len([r for r in rows if r['position'] == 'interior'])} interior, "
          f"{results['honesty']['labelled_events']} script-labelled")
    for cls in ("terminal", "beat", "internal"):
        print(f"  {cls}: {results['by_class'][cls]['n']}")
    return 0


def aggregate(clips, rows, asr_meta, asr_cache):
    """Everything the results file reports, computed from the pause rows."""
    rows = pause_rows(clips)
    interior = [r for r in rows if r["position"] == "interior"]

    # ---- the measured tables --------------------------------------------
    labelled = [r for r in interior if r.get("script_class")]
    free_rows = [r for r in interior if r["role"] == "free"
                 and r.get("data_class") != "PROBE"]

    def dist(rows_, key, filt=None):
        vals = [r[key] for r in rows_
                if r.get(key) is not None and (filt is None or filt(r))]
        return describe(vals)

    by_class = {}
    for cls in ("terminal", "beat", "internal"):
        sub = [r for r in labelled if r["script_class"] == cls]
        by_class[cls] = {
            "n": len(sub),
            "duration_s": describe([r["duration_s"] for r in sub]),
            "f0_pre_slope_st_s": dist(sub, "f0_pre_slope_st_s"),
            "f0_post_slope_st_s": dist(sub, "f0_post_slope_st_s"),
            "f0_jump_st": dist(sub, "f0_jump_st"),
            "f0_pre_range_st": dist(sub, "f0_pre_range_st"),
            "pre_function_word": {
                "n": len([r for r in sub if r.get("pre_function_word") is not None]),
                "n_function": len([r for r in sub if r.get("pre_function_word")]),
            },
            "post_lowercase": {
                "n": len([r for r in sub if r.get("post_lowercase") is not None]),
                "n_lowercase": len([r for r in sub if r.get("post_lowercase")]),
            },
            "rows": [{k: r[k] for k in ("clip", "start_s", "duration_s",
                                        "pre_text", "post_text",
                                        "script_class", "f0_pre_slope_st_s",
                                        "f0_post_slope_st_s", "f0_jump_st",
                                        "f0_pre_range_st")}
                     for r in sub],
        }

    features = ["duration_s", "f0_pre_slope_st_s", "f0_post_slope_st_s",
                "f0_jump_st", "f0_pre_range_st", "f0_pre_mean"]
    separation = {f: feature_table(labelled, f) for f in features}
    conjunction = conjunction_report(labelled, separation)

    # the proxy separation on the free corpus: how the same observables
    # distribute when there is no script to label them
    free_dist = {
        "n_interior": len(free_rows),
        "duration_s": describe([r["duration_s"] for r in free_rows]),
        "f0_pre_slope_st_s": dist(free_rows, "f0_pre_slope_st_s"),
        "f0_post_slope_st_s": dist(free_rows, "f0_post_slope_st_s"),
        "f0_jump_st": dist(free_rows, "f0_jump_st"),
        "n_pre_function_word": len([r for r in free_rows
                                    if r.get("pre_function_word")]),
        "n_post_lowercase": len([r for r in free_rows
                                 if r.get("post_lowercase")]),
    }

    # cross-take consistency: the SAME script position, read twice.
    # A and B read the same passage, so a pause in A and a pause in B
    # that sit at the same script token are the same event told twice.
    by_pos_a = {r["script_pre_tok"]: r for r in labelled
                if r["clip"] == "A" and r.get("script_pre_tok") is not None}
    by_pos_b = {r["script_pre_tok"]: r for r in labelled
                if r["clip"] == "B" and r.get("script_pre_tok") is not None}
    cross = []
    for tok in sorted(set(by_pos_a) & set(by_pos_b)):
        a, b = by_pos_a[tok], by_pos_b[tok]
        cross.append({
            "script_tok": tok,
            "class": a["script_class"],
            "a_start_s": a["start_s"], "b_start_s": b["start_s"],
            "a_duration_s": a["duration_s"], "b_duration_s": b["duration_s"],
            "duration_ratio": (b["duration_s"] / a["duration_s"]
                               if a["duration_s"] else None),
            "a_pre": a["pre_text"], "b_pre": b["pre_text"],
            "a_f0_pre_slope": a["f0_pre_slope_st_s"],
            "b_f0_pre_slope": b["f0_pre_slope_st_s"],
        })

    # the nulls: clips that should hold no speech at all
    nulls = []
    for clip in clips:
        if clip["role"] != "silent":
            continue
        inner = [p for p in clip["pauses"] if p["position"] == "interior"]
        nulls.append({
            "name": clip["name"],
            "duration_s": clip["duration_s"],
            "peak_rms": clip.get("peak_rms"),
            "n_silences": clip.get("n_silences"),
            "n_interior_silences": len(inner),
            "interior_durations_s": [p["duration_s"] for p in inner],
        })
    # the free sessions whose audio carried no speech at all: the corpus's
    # own null population, and the reason the corpus is a video corpus
    silent_free = []
    for clip in clips:
        if clip["role"] != "free":
            continue
        inner = [p for p in clip["pauses"] if p["position"] == "interior"]
        if len(inner) == 0 and clip["duration_s"] > 1.0:
            silent_free.append({"name": clip["name"], "duration_s": clip["duration_s"],
                                 "peak_rms": clip.get("peak_rms"),
                                 "n_words": clip.get("n_words"),
                                 "data_class": clip.get("meta", {}).get("data_class")})

    # the probe artefacts: a sonar sweep is not speech, and it is the one
    # clip in the corpus that is neither ambient nor drilled
    probes = []
    for clip in clips:
        if clip.get("meta", {}).get("data_class") != "PROBE":
            continue
        inner = [p for p in clip["pauses"] if p["position"] == "interior"]
        probes.append({
            "name": clip["name"],
            "duration_s": clip["duration_s"],
            "peak_rms": clip.get("peak_rms"),
            "n_words": clip.get("n_words"),
            "n_interior_silences": len(inner),
            "interior_durations_s": [p["duration_s"] for p in inner],
        })

    results = {
        "study": "pause study",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script_sha256": _sha256(Path(__file__)),
        "method": {
            "segmentation": ("src/energy.py rms_series(800, 160, 16000) + "
                             "segment(peak_rms=max frame rms); identical to "
                             "experiments/pause_threshold.py"),
            "min_silence_s": energy.MIN_SILENCE_S,
            "min_speech_s": energy.MIN_SPEECH_S,
            "pitch": "src/pitch.py YIN, window +/-1.0 s around each pause",
            "asr": "faster-whisper small.en, CPU int8, word timestamps",
            "word_assignment": ("monotonic sweep onto the energy speech segments; "
                                "whisper's word times are too coarse to place a "
                                "pause inside a word but its word order is exact"),
            "percentile_convention": "nearest rank",
        },
        "asr_meta": asr_meta,
        "clips": [{k: v for k, v in c.items() if k != "words"} for c in clips],
        "pause_rows": rows,
        "by_class": by_class,
        "separation": separation,
        "conjunction": conjunction,
        "free_distribution": free_dist,
        "cross_take": cross,
        "nulls": nulls,
        "probe_artifacts": probes,
        "silent_free_sessions": silent_free,
        "protocol_design": protocol_design(),
        "honesty": {
            "ground_truth": ("only the script pair carries ground truth; the free "
                             "sessions have none and are never counted as if they did"),
            "terminal_events": len([r for r in labelled
                                    if r["script_class"] == "terminal"]),
            "labelled_events": len(labelled),
            "power": ("the terminal class is a handful of events; an AUC on this "
                      "many points is a direction, not a measurement"),
            "classes_never_averaged": ("the script pair (performed reading) and the "
                                       "free sessions (first-take speech) are reported "
                                       "separately"),
        },
    }
    return results



def conjunction_report(labelled: list[dict], separation: dict) -> dict:
    """What the WHOLE rule does, not what one leg does.

    The brief asks for a conjunction -- terminal position AND prosody AND
    trailing silence -- and a conjunction has to be graded as one.  The
    thresholds are the ROBUST ones (the midpoint between the two class
    medians), NOT the in-sample best cut: on six positive events the
    in-sample cut lands on an outlier, and a rule built on an outlier is
    a rule that will not survive the next day.

    The legs, and which side of the cut is the terminal side:
      duration      -- longer
      pre slope     -- FALLING into the pause (the terminal side is the
                       LOWER one, which is why the sign is recorded)
      pitch step    -- a RESET across the pause (larger)
      syntax        -- the utterance was COMPLETE: the last word before
                       the pause is not a function word
    """
    legs = ("duration_s", "f0_pre_slope_st_s", "f0_jump_st")
    cuts = {}
    for f in legs:
        s = separation.get(f, {})
        cuts[f] = (s.get("robust_threshold"), s.get("robust_sign"))

    def on_terminal_side(f, v):
        thr, sign = cuts[f]
        if thr is None or sign is None or v is None:
            return False
        return sign * v > sign * thr

    def syntax_complete(r):
        # explicitly not a function word.  None (unknown) does not count
        # as complete: an unknown is not evidence.
        return r.get("pre_function_word") is False

    pos = [r for r in labelled if r["script_class"] == "terminal"]
    neg = [r for r in labelled if r["script_class"] != "terminal"]

    def grade(use_legs, use_syntax):
        fired_pos = [r for r in pos
                     if all(on_terminal_side(f, r.get(f)) for f in use_legs)
                     and (syntax_complete(r) if use_syntax else True)]
        fired_neg = [r for r in neg
                     if all(on_terminal_side(f, r.get(f)) for f in use_legs)
                     and (syntax_complete(r) if use_syntax else True)]
        return {
            "n_pos": len(pos), "n_neg": len(neg),
            "terminal_fired": len(fired_pos),
            "terminal_missed": len(pos) - len(fired_pos),
            "nonterminal_fired": len(fired_neg),
            "nonterminal_held": len(neg) - len(fired_neg),
            "missed_rows": [
                {"clip": r["clip"], "start_s": r["start_s"],
                 "duration_s": r["duration_s"], "pre_text": r["pre_text"],
                 "f0_pre_slope_st_s": r["f0_pre_slope_st_s"],
                 "f0_jump_st": r["f0_jump_st"]}
                for r in pos if not all(
                    on_terminal_side(f, r.get(f)) for f in use_legs)
                or (use_syntax and not syntax_complete(r))],
            "false_fire_rows": [
                {"clip": r["clip"], "start_s": r["start_s"],
                 "duration_s": r["duration_s"], "pre_text": r["pre_text"],
                 "script_class": r["script_class"]} for r in fired_neg],
        }

    out = {
        "note": ("a rule is graded as a whole: 'fired' means the rule would "
                 "YIELD THE FLOOR at that pause"),
        "cuts": {f: {"robust_threshold": cuts[f][0], "robust_sign": cuts[f][1]}
                 for f in legs},
        "syntax_leg": "the last word before the pause is not a function word",
        "each_leg_alone": {},
        "conjunction_prosody_only": grade(legs, False),
        "conjunction_all_four": grade(legs, True),
    }
    for f in legs:
        out["each_leg_alone"][f] = {
            "terminal_fired": len([r for r in pos if on_terminal_side(f, r.get(f))]),
            "n_pos": len(pos),
            "nonterminal_fired": len([r for r in neg if on_terminal_side(f, r.get(f))]),
            "n_neg": len(neg),
        }
    out["each_leg_alone"]["syntax"] = {
        "terminal_fired": len([r for r in pos if syntax_complete(r)]),
        "n_pos": len(pos),
        "nonterminal_fired": len([r for r in neg if syntax_complete(r)]),
        "n_neg": len(neg),
    }
    return out


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def selftest() -> int:
    import numpy as np
    checks = []

    def ck(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    # 1. a synthetic clip with two known silences
    sr = ANALYSIS_SR
    tone = [0.3 * math.sin(2 * math.pi * 200 * i / sr) for i in range(sr * 2)]
    silence = [0.0] * int(sr * 0.80)
    clip = tone + silence + tone
    segs, ef = segment_audio(clip)
    sils = [(s.start, s.end) for s in segs if not s.speech]
    ck("synthetic: one silence found", len(sils) == 1, f"got {sils}")
    if sils:
        ck("synthetic: silence length ~0.80 s",
           abs((sils[0][1] - sils[0][0]) - 0.80) < 0.08,
           f"got {sils[0][1]-sils[0][0]:.3f}")

    # 2. the same clip with two silences
    clip2 = tone + silence + tone + silence + tone
    segs2, _ = segment_audio(clip2)
    sils2 = [(s.start, s.end) for s in segs2 if not s.speech]
    ck("synthetic: two silences found", len(sils2) == 2, f"got {len(sils2)}")
    ck("synthetic: both silences ~0.80 s",
       all(abs((b - a) - 0.80) < 0.08 for a, b in sils2),
       str([round(b - a, 3) for a, b in sils2]))

    # 3. pitch tracker on a synthetic tone
    frames = track_pitch_window(clip, 0.0, 1.0, sr)
    voiced = [f["hz"] for f in frames if f["hz"]]
    ck("pitch: tracker fires on a 200 Hz tone", len(voiced) > 20, f"n={len(voiced)}")
    if voiced:
        est = statistics.fmean(voiced)
        ck("pitch: 200 Hz within 2 %", abs(est - 200) / 200 < 0.02, f"got {est:.2f}")

    # 4. script classification
    ck("script: sentence end is terminal",
       script_class_for(19, 20) == "terminal", str(script_class_for(19, 20)))
    ck("script: comma is a beat",
       script_class_for(4, 5) == "beat", str(script_class_for(4, 5)))
    ck("script: unmarked position is internal",
       script_class_for(48, 49) == "internal", str(script_class_for(48, 49)))
    ck("script: nothing to classify",
       script_class_for(None, 3) is None)

    # 5. normalisation folds the numeral the ASR produced
    ck("normalise: 9 folds to nine", _norm("9,") == "nine")
    ck("normalise: punctuation stripped", _norm("floor.") == "floor")

    # 6. alignment: a synthetic ASR of the script maps back onto itself
    fake = [{"text": t, "start": i * 0.3, "end": i * 0.3 + 0.25}
            for i, t in enumerate(SCRIPT_TOKENS)]
    mapping, ratio = align_to_script(fake)
    ck("align: ratio is 1.0 on the script itself", ratio == 1.0, f"{ratio}")
    ck("align: every token maps", all(mapping.get(i) == i for i in
                                       range(len(SCRIPT_TOKENS))))

    # 7. the monotonic word assignment repairs a late timestamp
    speech = [(0.0, 1.0), (2.0, 3.0)]
    words = [{"text": "a", "start": 0.5, "end": 0.9},
             {"text": "b", "start": 1.05, "end": 1.4},   # timestamp inside the pause
             {"text": "c", "start": 2.2, "end": 2.5}]
    assigned = assign_words_to_speech(words, speech)
    ck("assign: a word inside the pause joins the speech before it",
       [words[i]["text"] for i in assigned[0]] == ["a", "b"],
       str([words[i]["text"] for i in assigned[0]]))
    ck("assign: the word after the pause joins the next speech",
       [words[i]["text"] for i in assigned[1]] == ["c"],
       str([words[i]["text"] for i in assigned[1]]))

    # 8. statistics
    ck("auc: separated sets give 1.0", auc([3, 4, 5], [0, 1, 2]) == 1.0)
    ck("auc: identical sets give 0.5", auc([1, 2, 3], [1, 2, 3]) == 0.5)
    ck("percentile: nearest rank", pctl([1, 2, 3, 4, 5], 0.05) == 1)
    ck("median: even count averages", median([1, 2, 3, 4]) == 2.5)

    # 9. the function-word list
    ck("function words: 'the' hangs an utterance", "the" in FUNCTION_WORDS)
    ck("function words: 'coffee' does not", "coffee" not in FUNCTION_WORDS)

    # 10. a clip with no speech holds no interior pause
    noise = [(random.Random(7).random() - 0.5) * 0.001 for _ in range(sr * 3)]
    segs3, _ = segment_audio(noise)
    inner3 = [s for s in segs3 if not s.speech and s.start > 0.06
              and s.end < len(noise) / sr - 0.06]
    ck("null: quiet noise holds no interior pause", len(inner3) == 0,
       f"got {len(inner3)}")

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"   [{detail}]" if detail and not ok else ""))
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
