"""
turn_routes.py -- four independent routes to "did the turn end here?", scored
against ground truth that is KNOWN because we cut it ourselves.

WHY THIS EXPERIMENT EXISTS.

`turn.py` uses duration and it produces ZERO boundaries on six of seven real
corpus cells: across 673 measured silence runs (1671 pause events) the
longest is 2.54 s, so a 3 s
threshold never fires. A peripheral that never speaks is also a failure, so
"the threshold is correct and the material is atypical" is not a place to
stop.

Duration alone cannot do this job. The question is whether ANYTHING can, and
whether several things that each fail differently can be combined into
something better than any of them.

GROUND TRUTH BY CONSTRUCTION.

There are no turn annotations in the corpus and getting them would mean
labelling hours of audio by hand. So the truth is manufactured instead: take
continuous real speech and INSERT a silence at a controlled place. Two cases,
matched on everything except where the silence falls:

  * END   the silence goes after the audio stops, and nothing follows.
           Truth: the turn ended.
  * MID   the same silence goes in the MIDDLE of continuous speech, and the
           same voice resumes immediately afterwards. Truth: the turn did not
           end; the speaker has simply paused.

Same speaker, same words, same silence length. The only variable is whether
anyone speaks again. Any route that can tell MID from END is measuring
something real, because the acoustic difference between the two is precisely
the resumption.

A third case, NO-SILENCE, is a control: the untouched original. Truth: never
ended. A route that says "ended" here is reading the silence, not the speech.

THE FOUR ROUTES, EACH FAILING IN A DIFFERENT DIRECTION.

  1 ours     our own acoustic reduction (duration, pitch contour, energy).
             Deterministic, free, and blind to syntax -- which is the point:
             it is the route that cannot know a sentence finished.
  2 smartturn  smart-turn v3.2, BSD-2, 8,679,182 bytes, bundled in the
             pipecat wheel. Waveform only, no transcript. Previously measured
             at P(complete)=0.987 on DIGITAL SILENCE, which is a model that
             agrees with you about nothing.
  3 jev     cloud decision model via typesafe.ai. The only route that can
             reason about intent, and the only one that has never heard this
             voice or this conversation.
  4 clef     clef-omni on Cloudflare Workers. Audio-native, so it sees
             things our reduction throws away.

WHAT THE EXPERIMENT IS ACTUALLY LOOKING FOR.

Not a winner. An ensemble whose consensus is the product is the obvious move
and it is the wrong one here, because none of these four can be trusted to
cast a deciding vote.

The claim worth testing is that their DISAGREEMENT is the signal: that the
routes part company exactly where the answer is genuinely uncertain, and agree
where it is not. If so, variance is the product and attention can be spent
where they differ -- which is cheaper than asking a model about every pause,
and matches design v3 §2, where the ONSET is the event that spends attention.

If instead they disagree randomly, disagreement is noise and the honest
conclusion is that this whole ensemble should be dropped.

Run:  python experiments/turn_routes.py --stages acoustic,smartturn
      python experiments/turn_routes.py --stages jev,clef --limit 12
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import struct
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                             # noqa: E402

from src import wavio                                          # noqa: E402

SR = 16000
PIECAT_PKG = ROOT / ".scratch" / "st2" / "pkg"
SMARTTURN_ONNX = (PIECAT_PKG / "pipecat/audio/turn/smart_turn/data/"
                  "smart-turn-v3.2-cpu.onnx")
SMARTTURN_FEATURES = (PIECAT_PKG / "pipecat/audio/turn/smart_turn/"
                      "_whisper_features.py")

JEV_MODEL = os.environ.get("JEV_MODEL", "jev-1.14.0")
_JEV = None


def _jev_client(cli=None):
    global _JEV
    if cli is not None:
        return cli
    if _JEV is None:
        from typesafe_sdk import TypeSafeClient
        _JEV = TypeSafeClient()
    return _JEV


CORPUS = ROOT / "examples" / "realaudio"
CELLS = {
    "narration": CORPUS / "alice01_excerpt.wav",
    "essay": CORPUS / "corpus" / "essay_perpetual_peace_librivox.wav",
    "dialogue": CORPUS / "corpus" / "shortstory_shortstory058_1405_librivox.wav",
    "children": CORPUS / "corpus"
                / "children_english_fairy_tales_joy_librivox.wav",
}

# Silence lengths swept. Chosen from the MEASURED pause distribution in
# tests/test_pause_threshold.py: real inter-utterance gaps run 0.15-0.64 s
# with a median around 0.30, so this spans "breath" to "long enough that a
# human would start talking". (The 0.11 s end that earlier revisions of this
# comment quoted was the silence BEFORE THE FIRST WORD, which `energy.py`
# lets escape the min_silence_s filter -- it is not an inter-utterance gap.
# `docs/research/PAUSE-THRESHOLD.md`.)
SILENCE_LENGTHS = (0.15, 0.30, 0.60, 1.00, 1.50)


# -- provenance ---------------------------------------------------------------
# A concordance is only a claim about the audio it was computed from, so the
# result records which audio that was -- path and SHA-256 per cell -- plus the
# model file for the route that uses one, the script's own SHA-256, and the
# exact command line.

def sha256_file(path) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path) -> dict:
    """Where an input file is, and what it is."""
    p = Path(path)
    out = {"path": str(p.resolve())}
    if not p.is_file():
        out["problem"] = "the file is not on disk"
        return out
    out["sha256"] = sha256_file(p)
    out["bytes"] = p.stat().st_size
    return out


def build_provenance(cells, stages, cli_args, parameters,
                     skipped_cells=None) -> dict:
    """The provenance block, and whether it is complete."""
    script = Path(__file__).resolve()
    prov = {
        "generated_by": "experiments/turn_routes.py",
        "script_path": str(script),
        "script_sha256": sha256_file(script) if script.is_file() else None,
        "cells": list(cells),
        "skipped_cells": list(skipped_cells or []),
        "stages": list(stages),
        "cli_args": list(cli_args),
        "command_line": " ".join(str(x) for x in cli_args),
        "parameters": dict(parameters),
        "problems": [],
    }
    for c in prov["cells"]:
        if c.get("problem"):
            prov["problems"].append(f"{c.get('path')}: {c['problem']}")
    if "smartturn" in stages:
        prov["smartturn_model"] = file_identity(SMARTTURN_ONNX)
        prov["smartturn_features"] = file_identity(SMARTTURN_FEATURES)
        if prov["smartturn_model"].get("problem"):
            prov["problems"].append("the smart-turn model is not on disk")
    return prov


# -- the manufactured corpus -----------------------------------------------

def voiced_runs(samples, min_s=0.8):
    """
    Contiguous voiced stretches, from our own segmenter.

    Using the segmenter to build the ground truth is not circular here: the
    question under test is where a turn ENDS, and the cuts only need to be
    places where sound is currently present or absent. Whether a 0.3 s gap
    should count as a pause is a different question that this experiment does
    not ask.
    """
    from src import energy
    ef, _ = energy.rms_series(samples, 800, 160, SR)
    if not ef:
        return []
    segs = energy.segment(ef, 160 / SR,
                          peak_rms=max(f.rms for f in ef))
    runs = []
    for s in segs:
        if not s.speech:
            continue
        a, b = int(s.start * SR), int(s.end * SR)
        if (b - a) / SR >= min_s:
            runs.append((a, b))
    return runs


def build_cases(samples, cell, n_cases):
    """
    Manufacture matched END / MID / NONE cases.

    MID is the load-bearing one: the same audio with the same silence, but
    somebody speaks again. Anything that separates MID from END is reacting to
    the resumption rather than to the silence, which is the only thing a turn
    detector should be reacting to.
    """
    runs = voiced_runs(samples)
    if len(runs) < 4:
        return []
    cases = []
    # spread the cases across the file so we do not sample one speaker mood
    step = max(1, len(runs) // max(1, n_cases))
    chosen = runs[::step][:n_cases]
    for k, (a, b) in enumerate(chosen):
        span = b - a
        for sil in SILENCE_LENGTHS:
            gap = [0.0] * int(sil * SR)

            # END: silence after the audio, nothing follows.
            cases.append({
                "cell": cell, "case": f"END/{k}/{sil:.2f}", "truth": "end",
                "audio": samples[a:b] + gap, "silence_s": sil,
                "speech_after": False,
            })

            # MID: silence inside continuous speech, same voice resumes.
            # Cut at ~60% through so there is always speech on both sides.
            cut = a + int(span * 0.6)
            cases.append({
                "cell": cell, "case": f"MID/{k}/{sil:.2f}", "truth": "mid",
                "audio": samples[a:cut] + gap + samples[cut:b],
                "silence_s": sil, "speech_after": True,
            })

    # NONE: the untouched original, as a control. A route that says "ended"
    # on audio with no silence in it at all is reading nothing.
    for k, (a, b) in enumerate(chosen):
        cases.append({
            "cell": cell, "case": f"NONE/{k}", "truth": "none",
            "audio": samples[a:b], "silence_s": 0.0, "speech_after": True,
        })
    return cases


def build_resume_cases(samples, cell, n_cases):
    """
    THE CASE THAT MAKES THE WHOLE EXPERIMENT NON-CIRCULAR.

    The first version of this experiment was circular and I did not notice
    until the numbers came back: our own acoustic route scored a perfect
    concordance of 1.0, which looked like a triumph and was actually a
    giveaway that the question had been rigged.

    END is built as speech-then-silence. MID is speech-silence-speech. So
    "is there silence at the END of the audio" separates them perfectly BY
    DEFINITION, and a route that computes trailing silence is being handed
    the label. It would score 1.0 on any corpus built this way and tells us
    nothing about whether it could tell a speaker finishing from a speaker
    breathing.

    This construction removes the free pass:

      RESUME   speech, then a LONG silence, then the same voice resumes.

    Trailing silence is present at the moment of decision -- it is even longer
    than anything in END -- and yet the turn has NOT ended, because the
    speaker comes back.

    This is also the case that matters operationally. A peripheral that
    answers "the turn ended" whenever it hears silence will interrupt every
    pause a person ever takes, and the interruption happens BEFORE the
    evidence that it was wrong arrives. So a route that scores well on END
    and badly here is the exact failure this project is built to avoid.
    """
    runs = voiced_runs(samples)
    if len(runs) < 4:
        return []
    cases = []
    step = max(1, len(runs) // max(1, n_cases))
    for k, (a, b) in enumerate(runs[::step][:n_cases]):
        for sil in (0.60, 1.00, 1.50, 2.00):
            gap = [0.0] * int(sil * SR)
            cut = a + int((b - a) * 0.45)
            cases.append({
                "cell": cell, "case": f"RESUME/{k}/{sil:.2f}",
                "truth": "resume",
                "audio": samples[a:cut] + gap + samples[cut:b],
                "silence_s": sil, "speech_after": True,
            })
    return cases


# -- the causal cut: what a live system actually has ------------------------

def truncate_at_decision(case):
    """
    Cut the audio at the moment the decision would have to be made.

    THIS IS THE EXPERIMENT. The previous version scored each route on the
    complete clip, and that quietly handed our own route the answer:

    `trailing_silence` is computed over the whole audio, so it can see the
    speech that comes AFTER the pause. In END there is none and in RESUME
    there is 0.45-2.0 s of it, so a trailing-silence rule separates them
    perfectly -- a perfect concordance of 1.0 that was an artefact of the
    harness rather than a property of anything.

    A live peripheral stands at the END of the silence with no idea what
    follows. So the honest score is computed on `audio[:decision]`, where
    `decision` is where the trailing silence begins -- and at that instant
    END and RESUME are BYTE-IDENTICAL prefixes. They differ only in what a
    live system cannot yet know.

    Which means the experiment stops being a ranking exercise and becomes the
    question it should have been: is there ANY acoustic signal at the moment
    of decision that separates "finished" from "about to continue", or is
    waiting the only honest option?
    """
    audio = case["audio"]
    # the decision point is the start of the LAST silence in the clip
    from src import energy
    ef, _ = energy.rms_series(audio, 800, 160, SR)
    cut_idx = len(audio)
    if ef:
        peak = max(f.rms for f in ef)
        segs = energy.segment(ef, 160 / SR, peak_rms=peak)
        silences = [s for s in segs if not s.speech]
        if silences:
            cut_idx = int(silences[-1].start * SR)
    return dict(case, audio=audio[:cut_idx],
                decision_at_s=round(cut_idx / SR, 3))


# -- the causal cut: what a live system actually has ------------------------

def causal_agreement(cases):
    """
    Are the decision-time prefixes of END and RESUME actually identical?

    If they are, no acoustic route can separate them and the entire design
    question becomes "how long do you wait", not "what do you measure". This
    checks the premise rather than assuming it.
    """
    import hashlib
    ends = {hashlib.sha1(_fingerprint(c["audio"])).hexdigest()
            for c in cases if c["truth"] == "end"}
    res = {hashlib.sha1(_fingerprint(c["audio"])).hexdigest()
           for c in cases if c["truth"] == "resume"}
    if not ends or not res:
        return None
    return len(ends & res) / len(ends | res)


def _fingerprint(audio):
    a = np.asarray(audio, dtype=np.float32)
    return a[: min(len(a), SR * 30)].tobytes()


# -- route 1: ours ----------------------------------------------------------

def score_acoustic(case):
    """
    Our own reduction. Duration only, on purpose.

    Route 1 in the experiment is the route that CANNOT know a sentence
    finished. If it did as well as the others, the others would be buying
    nothing for 8.7 MB and a network call.
    """
    audio = case["audio"]
    from src import energy
    ef, _ = energy.rms_series(audio, 800, 160, SR)
    if not ef:
        return {"route": "acoustic", "p_end": 0.0, "detail": "no frames"}
    peak = max(f.rms for f in ef)
    segs = energy.segment(ef, 160 / SR, peak_rms=peak)
    longest_silence = max((s.duration() for s in segs if not s.speech),
                          default=0.0)
    last_speech_end = max((s.end for s in segs if s.speech), default=0.0)
    total = len(audio) / SR
    trailing = total - last_speech_end

    # The honest form of "our route": a soft function of trailing silence.
    # A 0.30 s breath and a 2 s gap are different but neither is 3 s.
    p = min(1.0, trailing / 3.0)
    return {
        "route": "acoustic",
        "p_end": round(p, 4),
        "longest_silence_s": round(longest_silence, 4),
        "trailing_silence_s": round(trailing, 4),
        "detail": f"trailing {trailing:.2f}s",
    }


# -- route 2: smart-turn v3.2 ----------------------------------------------

_sess = None
_feats = None


def _load_smartturn():
    """Load pipecat's vendored feature code by path, bypassing its package.

    `pipecat/__init__.py` imports loguru and aiohttp to expose ONE function
    that needs neither. Importing the file directly keeps this experiment at
    numpy + onnxruntime + soxr, with no web framework in the way.
    """
    global _sess, _feats
    if _sess is not None:
        return _sess, _feats
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "st_features", str(SMARTTURN_FEATURES))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    _sess = ort.InferenceSession(str(SMARTTURN_ONNX), sess_options=so,
                                 providers=["CPUExecutionProvider"])
    _feats = mod
    return _sess, _feats


def score_smartturn(case):
    sess, feats = _load_smartturn()
    a = np.asarray(case["audio"], dtype=np.float32)
    n = 8 * SR
    if len(a) > n:
        a = a[-n:]                       # keep the END: that is the question
    elif len(a) < n:
        a = np.pad(a, (n - len(a), 0))   # pipecat pads at the FRONT
    mel = feats.compute_whisper_log_mel_features(a, do_normalize=True)
    out = sess.run(None, {"input_features": mel[None, :, :]})
    p = float(np.ravel(out[0])[0])
    return {"route": "smartturn", "p_end": round(p, 6),
            "detail": f"p={p:.4f}"}


# -- route 3: Jev / typesafe.ai -------------------------------------------

JEV_NEUTRAL = (
    "Audio measurement. The waveform is 16 kHz mono. "
    "F0 median 214 Hz, F0 p05-p95 96-301 Hz, voiced fraction 0.62, "
    "RMS mean 0.041, RMS peak 0.118. The final 0.30 s before the decision "
    "point contains no voiced frames. Speaker identity unknown."
)


def score_jev(case, mode="leak"):
    """
    Cloud decision model. The ONLY route that can reason about intent.

    AND IT HAS A METHODOLOGICAL PROBLEM I AM MEASURING RATHER THAN HIDING.

    A text model cannot hear. Its entire input is whatever I type. So the
    first version of this route described the structure -- "and then RESUMES
    in the same voice" versus "does not resume" -- and asked the model to
    judge. The description CONTAINS THE ANSWER. A route that scores well
    there has learned to read my phrasing, not the speaker.

    That is not a hypothetical worry: it is the same failure the symbolic
    audio experiment found, where two encodings of identical audio produced
    1/3 and 3/3 correct depending only on how the shape was written down.

    So both modes are run:

      leak    the structural description, which names the resumption.
              The flattering number, and the dishonest one.
      neutral the same acoustic scalars for every case, with no mention of
              whether anyone speaks again.

    The GAP between them is the measurement. It is a reusable instrument: any
    time a language model is pointed at a signal, the honest baseline is
    "the same model, same question, no answer in the prompt".
    """
    from typesafe_sdk import Choice
    cli = _jev_client()
    state = (case["structural_description"] if mode == "leak"
             else JEV_NEUTRAL)
    question = Choice(
        instructions=(
            "Decide whether the speaker has finished their turn at the end "
            "of this measurement. A finished turn is one the speaker is not "
            "going to continue."),
        criteria={
            "yes": "The speaker has finished speaking and the turn is over.",
            "no": "The speaker may still continue, or the turn is not over.",
        })
    try:
        ans = cli.system_one(state=state,
                             questions={"turn_ended": question},
                             model=JEV_MODEL).answers
    except Exception as exc:                                # noqa: BLE001
        return {"route": f"jev_{mode}", "p_end": None,
                "detail": f"error: {type(exc).__name__}: {exc}"}
    cc = _cc_from_answer(ans.get("turn_ended"))
    return {"route": f"jev_{mode}", "p_end": cc, "detail": f"cc={cc}"}


def _cc_from_answer(ans):
    """
    CC = winning probability. NEVER `confidence` -- see src/policy.py.

    The transform is (p_max - 1/K) / (1 - 1/K), verified exact to 0.0025
    cloud / 0.0000 local in experiments/jev_round2.py. Two of the three
    attribute spellings are accepted because the SDK has used both.
    """
    if ans is None:
        return None
    dist = (getattr(ans, "probabilities", None)
            or getattr(ans, "distribution", None)
            or getattr(ans, "probs", None))
    if not dist:
        return None
    vals = list(dist.values()) if isinstance(dist, dict) else list(dist)
    if not vals:
        return None
    k = len(vals)
    return round((max(vals) - 1.0 / k) / (1.0 - 1.0 / k), 4)


# -- route 4: clef-omni on Cloudflare -------------------------------------

CF_ACCT = None


def _cf_account():
    global CF_ACCT
    if CF_ACCT:
        return CF_ACCT
    tok = os.environ.get("CF_TOKEN") or os.environ.get(
        "CLOUDFLARE_API_TOKEN")
    if not tok:
        raise RuntimeError("no CF token in the environment")
    import urllib.request
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/accounts",
        headers={"Authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        import json as _j
        CF_ACCT = _j.load(r)["result"][0]["id"]
    return CF_ACCT


def score_clef(case):
    """
    clef-omni on Cloudflare Workers: audio-native, so it sees the waveform
    rather than our reduction of it.
    """
    import urllib.request
    acct = _cf_account()
    pcm = np.asarray(case["audio"], dtype=np.float32)
    pcm = np.clip(pcm, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype("<i2")
    buf = io_wav_bytes(pcm16, SR)
    body = {
        "model": "clef-omni",
        "state": f"turn-end probe: {'resumes' if case['speech_after'] else 'stops'}",
        "questions": {
            "turn_complete": case["clef_question"],
        },
        "audio": [{"content_type": "audio/wav",
                   "base64": base64.b64encode(buf).decode()}],
    }
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/"
        f"@cf/cloudflare/clef-omni",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ.get('CF_TOKEN')}",
            "Content-Type": "application/json",
        })
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.load(r)
        ans = out.get("result", {}).get("answers", {}).get("turn_complete")
        p = _cc_from_answer(ans)
        usage = out.get("usage", {})
        return {"route": "clef", "p_end": p,
                "detail": f"cc={p}",
                "usage_tokens": usage.get("input_tokens")}
    except Exception as exc:                                # noqa: BLE001
        return {"route": "clef", "p_end": None,
                "detail": f"error: {type(exc).__name__}: {exc}"}


def io_wav_bytes(pcm16, sr):
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


# -- the questions ----------------------------------------------------------

def annotate(case):
    """The two descriptions the cloud routes are asked about."""
    truth = case["truth"]
    if truth == "end":
        structural = ("The speaker's audio ends with silence and does not "
                      "resume. There is no speech after the pause.")
    elif truth == "mid":
        structural = ("The speaker pauses briefly and then RESUMES in the "
                      "same voice. Speech follows the pause.")
    elif truth == "resume":
        structural = ("The speaker pauses for a long moment -- the longest "
                      "silence in this audio -- and then RESUMES in the same "
                      "voice. Speech follows the pause.")
    else:
        structural = ("The audio is continuous speech with no pause "
                      "inserted at all.")
    clef_q = (
        "Does the speaker STOP at the end of this audio (silence, nobody "
        "speaking after), or does the speech simply continue? Answer with "
        "the speaker stopping at the end or the speech continuing.")
    case["structural_description"] = structural
    case["clef_question"] = clef_q
    return case


# -- analysis --------------------------------------------------------------

def aucc(scores_by_truth):
    """
    Area under the correctness curve, from ranked scores.

    Not AUC-ROC on a binary label -- there is no single positive class here.
    The question is whether a route RANKS the three truth conditions
    end > mid > none, which is what a usable score has to do for the
    disagreement structure to mean anything.

    Computed as concordance within pairs, which is exact for the small
    samples this produces and needs no smoothing.
    """
    conc = disc = 0
    for hi, lo in (("end", "resume"), ("end", "mid"), ("end", "none"),
                   ("resume", "mid"), ("mid", "none")):
        a = [s for s in scores_by_truth.get(hi, []) if s is not None]
        b = [s for s in scores_by_truth.get(lo, []) if s is not None]
        for x in a:
            for y in b:
                if x > y:
                    conc += 1
                elif x == y:
                    conc += 0.5
                else:
                    disc += 1
    total = conc + disc
    return (conc / total) if total else None


def spread(pvals):
    """Standard deviation across routes for one case. The disagreement metric."""
    v = [p for p in pvals if p is not None]
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


# -- the summary, PERSISTED ----------------------------------------------------
# The headline statistics were printed to STDOUT and stored nowhere, so a note
# could quote a number that no result file held and nothing would notice the
# disagreement. The same numbers are computed here and written into the
# results JSON, each under a key that NAMES it.
#
# THE TIE CONVENTION is `aucc`'s and is recorded in the output itself: a tie
# contributes 0.5 to the numerator and NOTHING to the denominator, so the
# denominator counts only the decided pairs. The concordance figures quoted in
# docs/research/TURN-ROUTES.md were computed this way, so the persisted
# numbers have to use it and not the more common "ties count in the
# denominator" convention.

PAIR_CONTRASTS = (("end", "resume"), ("end", "mid"), ("end", "none"),
                   ("resume", "mid"), ("resume", "none"), ("mid", "none"))
CONTROL_CONTRASTS = (("end", "none"), ("resume", "none"), ("mid", "none"))
PRIMARY_CONTRAST = ("end", "resume")
TIE_CONVENTION = ("ties count 0.5 toward the concordance and are excluded from "
                  "the denominator, so the denominator counts only the decided "
                  "pairs (the `aucc` convention)")


def pair_concordance(a, b):
    """
    Concordance between two groups of scores, in `aucc`'s tie convention.

    Returns the fraction of decided pairs on which the first group scores
    higher, or None when there is nothing to compare.
    """
    xs = [v for v in a if v is not None]
    ys = [v for v in b if v is not None]
    if not xs or not ys:
        return None
    conc = disc = 0.0
    for x in xs:
        for y in ys:
            if x > y:
                conc += 1.0
            elif x == y:
                conc += 0.5
            else:
                disc += 1.0
    total = conc + disc
    return (conc / total) if total else None


def _concordance_np(xs, ys):
    """`pair_concordance` for numpy arrays -- the permutation null calls this
    20000 times, so the inner loop cannot be in Python."""
    diff = xs[:, None] - ys[None, :]
    conc = float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
    disc = float((diff < 0).sum())
    total = conc + disc
    return (conc / total) if total else None


def permutation_null(a, b, n_perm=20000, seed=7):
    """
    Label-permutation null for one contrast.

    The two groups' labels are shuffled among the cases that carry either of
    them and the concordance is recomputed. The null mean is MEASURED and not
    assumed to be 0.5: under this tie convention a null draw is biased
    slightly upward by its ties, and the figure the notes quote (0.5079) is a
    measured number, not a definition of chance.

    Returns None when there is nothing to permute.
    """
    xs = np.asarray([v for v in a if v is not None], dtype=np.float64)
    ys = np.asarray([v for v in b if v is not None], dtype=np.float64)
    if len(xs) == 0 or len(ys) == 0:
        return None
    observed = _concordance_np(xs, ys)
    pooled = np.concatenate([xs, ys])
    n_hi = len(xs)
    labels = np.array(["hi"] * n_hi + ["lo"] * len(ys))
    rng = np.random.default_rng(seed)
    null = np.zeros(int(n_perm), dtype=np.float64)
    for i in range(int(n_perm)):
        lab = labels[rng.permutation(len(pooled))]
        v = _concordance_np(pooled[lab == "hi"], pooled[lab == "lo"])
        null[i] = v if v is not None else 0.0
    hits = int((null >= observed).sum())
    return {
        "contrast": f"{PRIMARY_CONTRAST[0]}_vs_{PRIMARY_CONTRAST[1]}",
        "observed_concordance": float(observed),
        "null_mean": float(null.mean()) if n_perm else None,
        "null_sd": float(null.std(ddof=1)) if n_perm > 1 else None,
        "null_min": float(null.min()) if n_perm else None,
        "null_max": float(null.max()) if n_perm else None,
        "null_p95": float(np.percentile(null, 95)) if n_perm else None,
        "permutations": int(n_perm),
        "hits": hits,
        "p_value": (hits / n_perm) if n_perm else None,
        "seed": int(seed),
    }


def build_summary(results, stages, permutations=20000, seed=7, causal=False):
    """
    The headline statistics, as a dict that can be persisted.

    Every scalar sits under a key that names it: the observed concordance, the
    measured null mean and its spread, the hit count, the permutation count,
    the p value, and each control's concordance.
    """
    by_truth = {}
    by_cell = {}
    for r in results:
        by_truth.setdefault(r.get("truth"), []).append(r)
        cell = r.get("cell")
        by_cell[cell] = by_cell.get(cell, 0) + 1

    def scores(stage, truth):
        return [r[stage].get("p_end") for r in by_truth.get(truth, [])
                if r.get(stage) and r[stage].get("p_end") is not None]

    summary = {
        "n_cases": len(results),
        "cases_by_truth": {t: len(v) for t, v in sorted(by_truth.items())},
        "cases_by_cell": {c: n for c, n in sorted(by_cell.items())},
        "causal": bool(causal),
        "tie_convention": TIE_CONVENTION,
        "primary_contrast": f"{PRIMARY_CONTRAST[0]}_vs_{PRIMARY_CONTRAST[1]}",
        "stages": {},
    }

    for stage in stages:
        means = {}
        for truth in ("end", "resume", "mid", "none"):
            v = scores(stage, truth)
            means[truth] = (sum(v) / len(v)) if v else None
        pairs = {f"{hi}_vs_{lo}": pair_concordance(scores(stage, hi),
                                                   scores(stage, lo))
                 for hi, lo in PAIR_CONTRASTS}
        by_t = {t: scores(stage, t)
                for t in ("end", "resume", "mid", "none")}
        summary["stages"][stage] = {
            "n_scored": sum(1 for r in results if r.get(stage)),
            "mean_p_end_by_truth": means,
            "concordance_all_pairs": aucc(by_t),
            "pair_concordance": pairs,
            "control_concordance_vs_none": {
                f"{hi}_vs_none": pairs[f"{hi}_vs_none"]
                for hi, _ in CONTROL_CONTRASTS},
            "permutation_null": permutation_null(
                scores(stage, PRIMARY_CONTRAST[0]),
                scores(stage, PRIMARY_CONTRAST[1]),
                n_perm=permutations, seed=seed),
        }

    dis = {}
    for truth in ("end", "resume", "mid", "none"):
        v = sorted(r["disagreement"] for r in by_truth.get(truth, [])
                   if r.get("disagreement") is not None)
        if v:
            dis[truth] = {"mean": sum(v) / len(v), "median": v[len(v) // 2],
                          "max": max(v), "n": len(v)}
    summary["disagreement_by_truth"] = dis

    circ = {}
    for stage in stages:
        ends = scores(stage, "end")
        res = scores(stage, "resume")
        if not ends or not res:
            continue
        fe = sum(ends) / len(ends)
        fr = sum(res) / len(res)
        circ[stage] = {
            "mean_p_end_end": fe,
            "mean_p_end_resume": fr,
            "verdict": ("INTERRUPTS PEOPLE" if fr >= fe * 0.8
                        else "separates them"),
        }
    summary["circularity_check"] = circ
    return summary


def print_summary(summary):
    """Print the persisted summary, so stdout and the results file agree."""
    print("\nSUMMARY (the same numbers are persisted in the results file)")
    for stage, s in summary["stages"].items():
        print(f"  {stage}:")
        print(f"    concordance, {summary['primary_contrast']}: "
              f"{s['pair_concordance'].get(summary['primary_contrast'])}")
        ctl = s.get("control_concordance_vs_none", {})
        print("    controls vs none: " + ", ".join(
            f"{k} {v}" for k, v in ctl.items()))
        n = s.get("permutation_null")
        if n:
            print(f"    permutation null: mean {n['null_mean']:.4f} "
                  f"sd {n['null_sd']:.4f}  hits {n['hits']}/{n['permutations']}"
                  f"  p = {n['p_value']:.4f}")
    print(f"  tie convention: {summary['tie_convention']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="acoustic,smartturn")
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap cases for the paid stages")
    ap.add_argument("--causal", action="store_true",
                    help="score only the audio a live system would have at "
                         "the moment of decision (no future)")
    ap.add_argument("--out", default="experiments/turn_routes_results.json")
    ap.add_argument("--permutations", type=int, default=20000,
                    help="permutations for the measured null distribution")
    ap.add_argument("--perm-seed", type=int, default=7,
                    help="seed for the permutation null")
    args = ap.parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    runners = {"acoustic": score_acoustic, "smartturn": score_smartturn,
               "jev_leak": lambda c: score_jev(c, "leak"),
               "jev_neutral": lambda c: score_jev(c, "neutral"),
               "clef": score_clef}
    for s in stages:
        if s not in runners:
            ap.error(f"unknown stage {s!r}; pick from {sorted(runners)}")

    cases = []
    cell_ids = []
    skipped_cells = []
    for cell, path in CELLS.items():
        if not path.is_file():
            print(f"skip {cell}: absent")
            ident = file_identity(path)
            ident["name"] = cell
            skipped_cells.append(ident)
            continue
        ident = file_identity(path)
        ident["name"] = cell
        cell_ids.append(ident)
        samples = wavio.read_wav(str(path)).samples
        built = (build_cases(samples, cell, args.per_cell)
                 + build_resume_cases(samples, cell, args.per_cell))
        print(f"{cell:12s} {len(built):4d} cases")
        cases.extend(built)
    print(f"total {len(cases)} cases, stages={stages}\n")

    paid = [c for c in stages if c.startswith("jev") or c == "clef"]
    if paid and args.limit:
        cases = cases[:args.limit]
        print(f"CAPPED to {len(cases)} cases for paid stages\n")

    if args.causal:
        trunc = [truncate_at_decision(c) for c in cases]
        overlap = causal_agreement(trunc)
        print(f"CAUSAL MODE: scoring {len(trunc)} decision-time prefixes")
        if overlap is not None:
            print(f"  END/RESUME prefix overlap: {overlap:.1%} of the union")
            print("  100% would mean the two are indistinguishable at the")
            print("  instant of decision, and no acoustic route can help.\n")
        cases = trunc

    results = []
    t0 = time.perf_counter()
    for i, c in enumerate(cases):
        row = {"case": c["case"], "cell": c["cell"], "truth": c["truth"],
               "silence_s": c["silence_s"],
               "speech_after": c["speech_after"],
               "audio_s": round(len(c["audio"]) / SR, 3)}
        for s in stages:
            annotate(c)
            row[s] = runners[s](c)
        # the leak/neutral gap is the headline for the Jev route
        if "jev_leak" in row and "jev_neutral" in row:
            a, b = row["jev_leak"].get("p_end"), row["jev_neutral"].get("p_end")
            row["jev_leak_gap"] = (None if a is None or b is None
                                   else round(abs(a - b), 4))
        row["disagreement"] = round(spread(
            [row[s].get("p_end") for s in stages]), 4)
        results.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(cases):
            print(f"  {i + 1}/{len(cases)}  ({time.perf_counter() - t0:.0f}s)")

    # -- report ----------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'truth':8s} {'n':>4s} " +
          " ".join(f"{s[:9]:>9s}" for s in stages))
    print("-" * 72)
    by_truth = {}
    for truth in ("end", "resume", "mid", "none"):
        rows = [r for r in results if r["truth"] == truth]
        by_truth[truth] = rows
        cells = []
        for s in stages:
            v = [r[s]["p_end"] for r in rows if r[s].get("p_end") is not None]
            cells.append(f"{(sum(v) / len(v)):9.3f}" if v else f"{'-':>9s}")
        print(f"{truth:8s} {len(rows):4d} " + " ".join(cells))

    print("\nRANKS end > mid > none ?  (concordance within pairs)")
    for s in stages:
        by_t = {t: [r[s].get("p_end") for r in by_truth[t]
                    if r[s].get("p_end") is not None]
                for t in ("end", "resume", "mid", "none")}
        print(f"  {s:12s} {aucc(by_t)}")

    print("\nDISAGREEMENT across routes, by truth  <-- the actual claim")
    for truth in ("end", "resume", "mid", "none"):
        v = [r["disagreement"] for r in by_truth[truth]]
        if v:
            v = sorted(v)
            print(f"  {truth:6s} mean={sum(v) / len(v):.4f} "
                  f"median={v[len(v) // 2]:.4f} max={max(v):.4f}")
    print("\nTHE CIRCULARITY CHECK -- does trailing silence predict RESUME?")
    print("  RESUME has the LONGEST silence in the set and the speaker still")
    print("  comes back. A route that reads 'silence at the end' will score it")
    print("  as an end, and a peripheral doing that interrupts every pause.")
    for s_ in stages:
        ends = [r[s_].get("p_end") for r in by_truth["end"]
                if r[s_].get("p_end") is not None]
        res = [r[s_].get("p_end") for r in by_truth["resume"]
               if r[s_].get("p_end") is not None]
        if not ends or not res:
            continue
        fe = sum(ends) / len(ends)
        fr = sum(res) / len(res)
        verdict = "INTERRUPTS PEOPLE" if fr >= fe * 0.8 else "separates them"
        print(f"  {s_:12s} end={fe:.3f}  resume={fr:.3f}  -> {verdict}")

    ends = [r["disagreement"] for r in by_truth["end"]]
    res = [r["disagreement"] for r in by_truth["resume"]]
    mids = [r["disagreement"] for r in by_truth["mid"]]
    if ends and mids:
        se = sum(ends) / len(ends)
        sr = sum(res) / len(res) if res else 0.0
        sm = sum(mids) / len(mids)
        print(f"\n  mean disagreement: end={se:.4f} resume={sr:.4f} mid={sm:.4f}")
        print("  The claim under test is that routes part company where the")
        print("  answer is genuinely uncertain. If every truth condition")
        print("  disagrees by the same amount, disagreement is noise and the")
        print("  ensemble should be dropped.")

    if any(r.get("jev_leak_gap") is not None for r in results):
        gaps = [r["jev_leak_gap"] for r in results
                if r.get("jev_leak_gap") is not None]
        by_t = {}
        for truth in ("end", "resume", "mid", "none"):
            v = [r["jev_leak_gap"] for r in results
                 if r["truth"] == truth and r.get("jev_leak_gap") is not None]
            by_t[truth] = sum(v) / len(v) if v else None
        print("\nTHE FRAMING GAP -- how much of Jev's score is my phrasing?")
        print("  |CC(leak) - CC(neutral)| per case, by ground truth")
        for truth, v in by_t.items():
            if v is not None:
                print(f"    {truth:8s} {v:.4f}")
        print(f"  overall mean gap {sum(gaps) / len(gaps):.4f}")
        print("  A large gap means the model was reading my description of")
        print("  the answer rather than anything about the speaker.")

    summary = build_summary(results, stages,
                            permutations=args.permutations,
                            seed=args.perm_seed, causal=args.causal)
    print_summary(summary)

    used_argv = list(sys.argv[1:]) if argv is None else list(argv)
    provenance = build_provenance(
        cell_ids, stages, used_argv,
        {"per_cell": args.per_cell, "limit": args.limit, "causal": args.causal,
         "permutations": args.permutations, "perm_seed": args.perm_seed,
         "out": str(args.out),
         "silence_lengths": list(SILENCE_LENGTHS)},
        skipped_cells=skipped_cells)

    out = Path(args.out)
    out.write_text(json.dumps({
        "stages": stages, "causal": args.causal, "results": results,
        "summary": summary,
        "provenance": provenance,
        "reproducible": bool(
            provenance.get("script_sha256")
            and provenance.get("cli_args") is not None
            and all(c.get("sha256") for c in provenance.get("cells", []))
            and not provenance.get("problems")),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()