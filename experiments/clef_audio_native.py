#!/usr/bin/env python
"""
Does the "reduce audio to numeric tracks, then send text" step survive contact
with a model that reads audio directly?

WHY THIS EXPERIMENT EXISTS.

Every measurement in this project has gone through the same bottleneck: Jev
reads TEXT ONLY. So audio became PITCH / TEMPO / VELOCITY / INFLECTION /
SILENCE -- five scalars per frame -- and those went into a score, and the score
went to the model as a sentence. That reduction was not a design choice, it was
a workaround, and its cost has never been measured.

Meanwhile the folklore said images were dead (measured: true -- swapping a
spectrogram for blank grey moved the answer by 0.020) and that no path to a
text-only model could carry more than scalars. The folklore did not consider a
third option.

`@cf/cloudflare/clef-omni` IS a System One implementation -- same typed
Choice/Score/Noul primitives, same wire shape as the TypeSafe SDK's
`model_dump()` -- and it accepts RAW AUDIO. That is not a novelty; it is a
direct attack on the reduction step this whole pipeline is built around.

THE THREE ARMS, on the same bytes:

  A. ours       acoustic tracks -> numeric sentence -> Jev (text-only)
  B. direct     raw audio -> clef-omni
  C. via STT    raw audio -> Whisper -> text -> Jev

If B agrees with A, the reduction is redundant and could be deleted, which would
be a large simplification. If B disagrees materially, the reduction is
LOAD-BEARING -- it is throwing away something, and arm C measures whether
throwing it away was necessary.

PREDICTIONS, stated before the run:

  P1  clef-omni answers typed questions about raw audio. If it does not, the
      "System One accepts audio" claim does not hold and arm B is dead.
  P2  clef's `confidence` FOLLOWS the same arithmetic as Jev's,
      (p_max - 1/K)/(1 - 1/K). A first probe returned 0.9604 where that
      formula predicts 0.98, which is a RED FLAG: if confirmed, the two
      implementations disagree about what confidence means, and a confidence
      threshold tuned on one is meaningless on the other.
  P3  clef handles the adversarial text probes at least as well as Jev. Same
      battery, same framing.

    set -a && . ~/.config/cloudflare/env && set +a
    /home/proart/projects/glasses/.venv/bin/python experiments/clef_audio_native.py
"""

from __future__ import annotations

import base64
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import judged, wavio                                 # noqa: E402
from experiments.jev_round2 import (                            # noqa: E402
    ADVERSARIAL_BATTERY, GARBAGE, GARBAGE_ARGUE, HONEST, QUESTION)

ACCOUNT = os.environ.get("CF_ACCOUNT_ID", "049ff5e84ecf636b53b162cbb580aae6")
CLEF = "@cf/cloudflare/clef"
CLEF_OMNI = "@cf/cloudflare/clef-omni"
WHISPER = "@cf/openai/whisper-large-v3-turbo"
CLIP_SECONDS = 6.0


# -- plumbing ----------------------------------------------------------------

def wav_bytes(samples, sr=16000) -> bytes:
    """16-bit mono PCM WAV. ~15 lines; nothing in the repo writes one yet."""
    pcm = b"".join(struct.pack("<h", max(-32768, min(32767, int(x * 32767))))
                   for x in samples)
    hdr = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
           + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
           + b"data" + struct.pack("<I", len(pcm)))
    return hdr + pcm


def cf_run(model: str, body: dict, timeout: float = 180.0):
    token = os.environ.get("CF_TOKEN")
    if not token:
        raise SystemExit("no CF_TOKEN (source ~/.config/cloudflare/env)")
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/{model}",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode()[:300]
        except Exception:                                  # noqa: BLE001
            pass
        return {"error": f"http{exc.code}", "detail": detail,
                "elapsed_s": round(time.perf_counter() - t0, 3)}
    out["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return out


# -- arms --------------------------------------------------------------------

def arm_b_direct(audio_wav_b64: str, questions: dict,
                  state: str = "A recording follows."):
    """
    Raw audio straight to a model that accepts audio.

    The schema was found by probing, not by reading it anywhere. Four wrong
    shapes first: `audio` must be an ARRAY (not a base64 string), each element
    an OBJECT with `content_type` and `base64` (not `audio`), and the root also
    requires `model` and `state` alongside `questions`. The 400s say all of
    that precisely, which is better than most documentation.

    `content_type` is the literal string "audio/wav" -- verified working.
    """
    payload = {"model": "clef-omni", "state": state, "questions": questions,
               "audio": [{"content_type": "audio/wav",
                          "base64": audio_wav_b64}]}
    return cf_run(CLEF_OMNI, payload)


def arm_c_via_stt(audio_wav_b64: str):
    """Audio -> text -> Jev, i.e. the hop the STT folklore assumes is required."""
    out = cf_run(WHISPER, {"audio": audio_wav_b64})
    if out.get("error") or not out.get("result"):
        return None, out
    return str(out.get("result", {}).get("text", "")).strip(), out


def plain(answers) -> dict:
    """
    SDK answer objects -> plain JSON-able dicts, AT THE SOURCE.

    Reading the results file back and finding every arm A and C field `None`
    cost a full debugging round. Cause: `json.dump(..., default=str)` renders
    an SDK `NoulAnswer` as the string `"0.58"`, so after the round trip the
    comparison was doing getattr() on strings. The numbers were never missing;
    they were stringified on the way to disk. Anything read back for
    comparison has to be plain BEFORE it is written.
    """
    out = {}
    for key, a in (answers or {}).items():
        out[key] = {
            "type": getattr(a, "type", None),
            "noul": getattr(a, "noul", None),
            "choice": getattr(a, "choice", None),
            "probabilities": getattr(a, "probabilities", None),
            "confidence": getattr(a, "confidence", None),
        }
    return out


def as_sdk(questions: dict) -> dict:
    """
    Plain specs -> SDK objects.

    The SDK does NOT accept the raw dicts Cloudflare's REST endpoint takes.
    Passing them silently produces answers with every field None -- which
    looked exactly like "the model declined to answer" and was in fact "the
    question was never built". A plain-dict backend and a typed-client backend
    are the same wire format with different front doors.
    """
    from typesafe_sdk import Choice, Noul
    out = {}
    for key, spec in questions.items():
        if spec["type"] == "choice":
            out[key] = Choice(instructions=spec["instructions"],
                              criteria=spec["criteria"])
        else:
            out[key] = Noul(instructions=spec["instructions"],
                            criteria=spec["criteria"])
    return out


def jev(client, state: str, questions: dict, model="jev-1.14.0"):
    t0 = time.perf_counter()
    answers = client.system_one(state=state, questions=as_sdk(questions),
                                model=model).answers
    return plain(answers), round(time.perf_counter() - t0, 3)


# -- the questions asked of every arm ---------------------------------------

def descriptive_questions() -> dict:
    """Plain textual probes about the utterance. Every arm can answer these."""
    return {
        "is_speech": {
            "type": "noul",
            "instructions": "Is this recognisable human speech?",
            "criteria": {"false": "It is not recognisable human speech."}},
        "is_directive": {
            "type": "choice",
            "instructions": "Is the speaker asking for something to be done?",
            "criteria": {"yes": "The speaker asks for something to be done.",
                         "no": "The speaker is not asking for anything."}},
    }


def battery_questions() -> dict:
    out = {}
    for key, (text,) in ((k, v) for k, v in ADVERSARIAL_BATTERY.items()):
        out[key] = {"type": "noul", "instructions": text,
                    "criteria": {"false": "No such defect is present."}}
    return out


# -- experiments -------------------------------------------------------------

def exp_p1_audio_is_accepted(wav_b64: str, clip_label: str):
    """
    PREDICTION: clef-omni answers typed questions about raw audio.

    Without this, arms B and C are meaningless, so it runs first and reports
    the raw response rather than a summary -- including the model-reported
    usage, because tokens-per-second-of-audio is the number that decides whether
    this is affordable as a continuous path or only a batch one.
    """
    out = arm_b_direct(wav_b64, descriptive_questions())
    return {
        "clip": clip_label,
        "ok": "error" not in out,
        "error": out.get("error"),
        "detail": out.get("detail"),
        "elapsed_s": out.get("elapsed_s"),
        "usage": out.get("usage"),
        "answers": (out.get("result") or {}).get("answers")
        if isinstance(out.get("result"), dict) else None,
    }


def exp_p2_confidence_formula(questions_results: list):
    """
    PREDICTION: clef's confidence follows (p_max - 1/K)/(1 - 1/K), exactly as
    Jev's does.

    Jev measured exact: local tev1 max abs error 0.0000, cloud 0.0025. A first
    clef probe returned 0.9604 for probabilities {yes: 0.99, no: 0.01}, where
    the formula predicts 0.98. If that is systematic, the two System One
    implementations do NOT share a confidence scale -- and the first
    consequence would be that the dead-band rule in policy.py, which came from
    the Jev ecosystem, cannot be assumed to hold here.
    """
    rows = []
    for item in questions_results:
        answers = item.get("answers") or {}
        for key, ans in answers.items():
            if not isinstance(ans, dict):
                continue
            probs = ans.get("probabilities") or {}
            conf = ans.get("confidence")
            if not probs or conf is None or len(probs) < 2:
                continue
            k = len(probs)
            predicted = (max(probs.values()) - 1.0 / k) / (1.0 - 1.0 / k)
            rows.append({"clip": item.get("clip"), "question": key,
                         "k": k, "reported": round(conf, 4),
                         "predicted": round(predicted, 4),
                         "abs_error": round(abs(predicted - conf), 4)})
    errs = [r["abs_error"] for r in rows]
    return {"rows": rows, "n": len(rows),
            "max_abs_error": max(errs) if errs else None,
            "mean_abs_error": round(sum(errs) / len(errs), 4) if errs else None}


def exp_p3_adversarial(text_questions: dict):
    """
    PREDICTION: clef handles the adversarial probes at least as well as Jev.

    Same battery, same failure-framed phrasing, same max combination. Jev
    measured: honest 0.240, attacking garbage 0.970.
    """
    out = {}
    for label, text in (("honest", HONEST), ("question", QUESTION),
                        ("garbage", GARBAGE),
                        ("garbage_arguing", GARBAGE_ARGUE)):
        res = cf_run(CLEF, {"state": text, "questions": text_questions})
        answers = res.get("result", {}).get("answers") or {}
        values = {k: float(v.get("noul", 0.0))
                  for k, v in answers.items() if isinstance(v, dict)}
        out[label] = {"elapsed_s": res.get("elapsed_s"),
                      "error": res.get("error"),
                      "fields": {k: round(v, 4) for k, v in values.items()},
                      "max": round(max(values.values()), 4) if values else None,
                      "argmax": max(values, key=values.get) if values else None}
    return out


def exp_three_way(clip_path: str, clip_label: str, client):
    """
    The actual comparison. Same bytes, three routes.

    Arm A is what this project ships: the acoustic tracks become a sentence and
    the sentence goes to a text-only model. Arm B is the same decisions with no
    reduction step at all. Arm C is the reduction everyone assumes is necessary.
    """
    audio = wavio.read_wav(clip_path)
    n = int(CLIP_SECONDS * audio.sample_rate)
    clip = audio.samples[:n]
    b64 = base64.b64encode(wav_bytes(clip, audio.sample_rate)).decode()

    tracks_text = judge_with_tracks(clip, audio.sample_rate, client)
    direct = arm_b_direct(b64, descriptive_questions())
    transcript, stt_meta = arm_c_via_stt(b64)
    via_stt = (jev(client, transcript, descriptive_questions())
               if transcript else (None, None))

    return {
        "clip": clip_label,
        "duration_s": round(n / audio.sample_rate, 2),
        "wav_bytes": len(b64) * 3 // 4,
        "arm_A_ours_tracks_to_jev": tracks_text,
        "arm_B_direct_audio_to_clef": {
            "elapsed_s": direct.get("elapsed_s"),
            "error": direct.get("error"),
            "usage": direct.get("usage"),
            "answers": (direct.get("result") or {}).get("answers")
            if isinstance(direct.get("result"), dict) else None,
        },
        "arm_C_stt_to_jev": {
            "transcript_head": (transcript or "")[:160],
            "transcript_len": len(transcript or ""),
            "stt_elapsed_s": stt_meta.get("elapsed_s") if stt_meta else None,
            "elapsed_s": via_stt[1],
            "answers": via_stt[0],
        },
    }


def judge_with_tracks(clip, sr, client):
    """Arm A: the pipeline as it ships. Acoustic tracks -> a sentence -> Jev."""
    from src import energy, inflection, pitch, tempo
    ef, _poisoned = energy.rms_series(clip, 400, 160, sr)
    segs = energy.segment(ef, 160 / sr, peak_rms=max((f.rms for f in ef),
                                                     default=0.0))
    voiced = [f for f in ef if f.rms > energy.ABS_FLOOR]
    text = (
        f"Measured from a {len(clip)/sr:.2f} s audio clip. "
        f"Peak RMS {max((f.rms for f in ef), default=0.0):.4f}, "
        f"mean RMS {sum(f.rms for f in ef)/max(1,len(ef)):.4f}. "
        f"{len(segs)} segments, "
        f"{sum(1 for s in segs if s.speech)} of them speech. "
        f"Voiced proportion {len(voiced)/max(1,len(ef)):.2f}."
    )
    answers, elapsed = jev(client, text, descriptive_questions())
    return {"state_sent": text, "elapsed_s": elapsed, "answers": answers}


# -- run ---------------------------------------------------------------------

def main() -> int:
    if not os.environ.get("CF_TOKEN"):
        print("source ~/.config/cloudflare/env first")
        return 2
    from typesafe_sdk import TypeSafeClient
    client = TypeSafeClient()

    corpus_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples", "realaudio", "corpus")
    clips = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".wav"))
    if not clips:
        print("no real-audio corpus; nothing to compare")
        return 2

    report = {"account": ACCOUNT[:6] + "...", "models": [CLEF_OMNI, CLEF]}
    print("== P1. does clef-omni accept raw audio? ==")
    first = clips[0]
    audio = wavio.read_wav(os.path.join(corpus_dir, first))
    n = int(CLIP_SECONDS * audio.sample_rate)
    b64 = base64.b64encode(
        wav_bytes(audio.samples[:n], audio.sample_rate)).decode()
    report["p1"] = exp_p1_audio_is_accepted(b64, first)
    p1 = report["p1"]
    print(f"   {first}: ok={p1['ok']} {p1['elapsed_s']}s "
          f"usage={p1['usage']}")
    if p1["error"]:
        print(f"   ERROR {p1['error']}: {p1['detail']}")
        print("   arms B and C are dead; stopping.")
        _dump(report)
        return 1
    print(f"   answers: {p1['answers']}")

    print("\n== P2. does clef follow the same confidence formula as Jev? ==")
    probes = []
    for clip in clips[:3]:
        a = wavio.read_wav(os.path.join(corpus_dir, clip))
        bb = base64.b64encode(
            wav_bytes(a.samples[:n], a.sample_rate)).decode()
        probes.append(exp_p1_audio_is_accepted(bb, clip))
    report["p2"] = exp_p2_confidence_formula(probes)
    p2 = report["p2"]
    print(f"   n={p2['n']}  max|err|={p2['max_abs_error']}  "
          f"mean|err|={p2['mean_abs_error']}")
    print("   (Jev cloud measured 0.0025, local tev1 measured 0.0000)")
    for r in p2["rows"][:6]:
        print(f"     {r['question']:14s} k={r['k']} reported={r['reported']:.4f} "
              f"predicted={r['predicted']:.4f} err={r['abs_error']:.4f}")

    print("\n== P3. adversarial battery on clef ==")
    print("   (Jev: honest 0.240, attacking garbage 0.970)")
    report["p3"] = exp_p3_adversarial(battery_questions())
    for label, r in report["p3"].items():
        if r["max"] is None:
            print(f"   {label:16s} ERROR {r['error']}")
        else:
            print(f"   {label:16s} max={r['max']:.3f} ({r['argmax']}) "
                  f"{r['elapsed_s']}s")

    print("\n== THREE-WAY on real speech ==")
    report["three_way"] = [exp_three_way(os.path.join(corpus_dir, c), c, client)
                           for c in clips[:2]]
    for row in report["three_way"]:
        print(f"   {row['clip']}  {row['duration_s']}s  "
              f"{row['wav_bytes']} bytes")
        print(f"     A tracks->Jev : {row['arm_A_ours_tracks_to_jev']['elapsed_s']}s")
        print(f"     B audio->clef : {row['arm_B_direct_audio_to_clef']['elapsed_s']}s "
              f"{row['arm_B_direct_audio_to_clef']['usage']}")
        c = row["arm_C_stt_to_jev"]
        print(f"     C stt->Jev    : {c['stt_elapsed_s']}s STT + "
              f"{c['elapsed_s']}s Jev  transcript={c['transcript_len']} chars")

    _dump(report)
    return 0


def _dump(report):
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "clef_audio_native_results.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())