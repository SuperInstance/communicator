"""
turn_routes.py -- four independent routes to "did the turn end here?", scored
against ground truth that is KNOWN because we cut it ourselves.

WHY THIS EXPERIMENT EXISTS.

`turn.py` uses duration and it produces ZERO boundaries on six of seven real
corpus cells: across 680 measured pause runs the longest is 2.1 s, so a 3 s
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
# tests/test_pause_threshold.py: real inter-utterance gaps run 0.11-0.64 s
# with a median around 0.30, so this spans "breath" to "long enough that a
# human would start talking".
SILENCE_LENGTHS = (0.15, 0.30, 0.60, 1.00, 1.50)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="acoustic,smartturn")
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap cases for the paid stages")
    ap.add_argument("--causal", action="store_true",
                    help="score only the audio a live system would have at "
                         "the moment of decision (no future)")
    ap.add_argument("--out", default="experiments/turn_routes_results.json")
    args = ap.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    runners = {"acoustic": score_acoustic, "smartturn": score_smartturn,
               "jev_leak": lambda c: score_jev(c, "leak"),
               "jev_neutral": lambda c: score_jev(c, "neutral"),
               "clef": score_clef}
    for s in stages:
        if s not in runners:
            ap.error(f"unknown stage {s!r}; pick from {sorted(runners)}")

    cases = []
    for cell, path in CELLS.items():
        if not path.is_file():
            print(f"skip {cell}: absent")
            continue
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

    out = Path(args.out)
    out.write_text(json.dumps({
        "stages": stages, "causal": args.causal, "results": results,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()