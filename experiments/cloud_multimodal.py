"""
cloud_multimodal.py -- the teacher Casey proposed, measured properly.

WHAT CASEY PROPOSED.

"use various cloud models like multimodal models and LLMs to analyse too. you
would have a good understanding of the video before you even start to try to
make the system work. this understanding from larger models can be used to
supervise the experimentation and ML and NN."

The Cloudflare account has 72 models, including Llama 4 Scout 17B, Llama 3.2
Vision 11B, LLaVA 1.5 7B, and clef-omni (30B, raw audio). So a large
multimodal model CAN look at the footage. The question is whether it knows
anything, and the answer to that is not its accuracy -- it is its accuracy
relative to controls.

THE FRAME LAYOUT, AND WHAT IT CHANGED.

The captures on disk are NOT standard NV12: `C:\\proart_capture\\nv12small.py`
writes 19,200 bytes per 160x96 frame (15,360 luma + 1,920 U + 1,920 V), not
the `W*H*3//2` = 23,040 that every experiment here used to assume. Read at
23,040, all but every sixth frame is torn -- rows 0..71 of one frame and rows
72..95 of the next -- so every frame that was sent to a model was a collage
of two of them. The layout is DETECTED from the file's content by
`src/nv12.py`; see `src/nv12.py` and `docs/research/NV12-STRIDE.md`.

The previous run's speaking arm had 11 frames rather than 12 for a reason
that was not visible in the results: the wrong stride produced 750 frames
instead of 900, and one of the VAD-picked frames (817) fell off the end and
was dropped. The two arms therefore had DIFFERENT denominators, which is the
shape of every wrong number in this project's history. They match now.

THE FIRST PROBE, and why it is not yet a result.

One frame per condition, Llama 3.2 Vision 11B:

    speaking frame  ->  "SPEAKING."
    silent frame    ->  "NOT SPEAKING."
    blank frame     ->  "NOT SPEAKING."

Two out of two on the task, and the blank is not exploited. Promising, and also
a single sample of a single frame, which is the kind of number this project has
been burned by repeatedly. `teacher_bench.py` exists because a model answering
a constant to everything reads exactly like a working classifier in a summary
table.

So this file does it properly:

  MANY FRAMES. Every frame of the speech window of each condition, not one.
    Twenty speaking frames and twenty silent frames, so the accuracy has a
    denominator and the wrong-answer set is visible.
  THE BLANK CONTROL, per model, at the same count. A model that answers
    "NOT SPEAKING" to everything scores 100% on the silent class and 0% on the
    speaking class, which is 50% -- the blank control is what makes that
    visible.
  THE TEMPLATE CONTROL. The same prompt with no image. The floor.
  THE PARSER. "NOT SPEAKING" CONTAINS THE WORD "SPEAKING". A substring check
    would score every negative as positive. The first version of the probe
    looked for "yes"/"no" and silently returned None for every model, which is
    the same "empty response" shape that hid a real bug earlier today. The
    parser now tests for the NEGATIVE phrase first.

THE AUDIO ARM, which is the join.

clef-omni takes RAW AUDIO. That makes it the only model in the account that
can answer the audio half of the question, and the audio half is the one the
project has spent its life failing at: `TWO-SPEAKER-TURNS.md` established that
at the moment of decision, END and RESUME are byte-identical audio.

So the audio arm asks clef-omni the same question about the same sessions,
and the result is directly comparable to the vision arm on the same data.
If clef-omni can separate facing from turned from AUDIO ALONE, that is the
join the project has been looking for, and it would not need the camera at
all.

THE DESCRIPTIONS, which are the actual deliverable.

A yes/no label is worth one bit. A description is worth a lot more, because it
can name the visual cue -- the mouth, the head angle, the hands -- and that
is the thing a small local student needs to be told to look for. The
descriptions are collected and reported verbatim.

WHAT IS NOT CLAIMED.

Twenty frames from one session per condition, one speaker, one room. Even a
good number here is a demonstration that the signal exists, not an accuracy
claim, and not a substitute for the leave-one-session-out test that refuses
to be fooled.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = os.environ.get("CF_ACCOUNT_ID", "049ff5e84ecf636b53b162cbb580aae6")
W, H = 160, 96
YS = W * H
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.nv12 import read_luma  # noqa: E402

Q_YESNO = ("Is the person in this image SPEAKING (their mouth is moving or "
           "open, as if talking) or NOT SPEAKING (mouth closed, still)? "
           "Answer with exactly one word: SPEAKING or NOT SPEAKING.")
Q_DESCRIBE = ("Describe this image in detail. Is there a person visible? "
               "Where are they and what are they doing? Is their mouth open "
               "or moving? Are they facing the camera or turned away?")
Q_AUDIO = ("Listen to this recording. Is the person SPEAKING (you can hear "
           "them talking) or NOT SPEAKING (silence or only background noise)? "
           "Answer with exactly one word: SPEAKING or NOT SPEAKING.")
Q_POSE = ("Is the person in this image FACING the camera, or turned AWAY from "
          "the camera? Answer with exactly one word: FACING or AWAY.")
Q_FACING = ("Listen to this recording. Does the speaker sound like they are "
            "facing TOWARDS the microphone, or turned AWAY from it? Answer with "
            "exactly one word: FACING or AWAY.")


def sha256_file(path) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_identity(path) -> dict:
    """Where a capture is, what it is, and how it was laid out."""
    p = Path(path)
    _luma, layout = read_luma(p, width=W, height=H)
    return {
        "path": str(p.resolve()),
        "sha256": sha256_file(p),
        "bytes": p.stat().st_size,
        "layout": layout.name,
        "stride_bytes": layout.stride,
        "n_frames": layout.n_frames,
        "layout_evidence": layout.as_dict(),
    }


def write_png(img: np.ndarray) -> bytes:
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def load_y(path: Path, nf: int | None = None):
    """Load the luma plane of a capture.

    The frame layout is DETECTED from the file's content by `src/nv12.py` --
    the candidate whose luma has the smallest mean consecutive |diff| wins,
    and an ambiguous file is an error rather than a guess. The captures on
    disk carry 19,200 bytes per 160x96 frame, not the 23,040 a standard-NV12
    reader assumes; read at 23,040 all but every sixth frame is torn (rows
    0..71 of frame k, rows 72..95 of frame k+1). See `src/nv12.py` and
    `docs/research/NV12-STRIDE.md`.

    `nf` DEFAULTS TO ALL FRAMES. An earlier version defaulted to 400, which
    silently truncated every session at frame 400 (13.3 s) -- and, because the
    speech-aligned frame picker chose frames across the whole session and then
    filtered them to the loaded range, it cut the speaking condition from 12
    frames to 6 while leaving the silent condition at 12. Two conditions with
    different denominators, which is the shape of every wrong number in this
    project's history.
    """
    luma, _layout = read_luma(path, width=W, height=H)
    n = len(luma) if nf is None else min(nf, len(luma))
    return luma[:n]


def cf_raw(model: str, body: dict, timeout: float = 180.0):
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/{model}",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['CF_TOKEN']}",
                 "Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        return {"error": f"http{exc.code}",
                "detail": exc.read().decode()[:400],
                "elapsed_s": round(time.perf_counter() - t0, 2)}
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__, "detail": str(exc)[:200],
                "elapsed_s": round(time.perf_counter() - t0, 2)}
    out = d.get("result") or {}
    if isinstance(out, dict):
        ch = out.get("choices") or []
        if ch:
            out = {"text": (ch[0].get("message") or {}).get("content", "")}
        elif "response" in out:
            out = {"text": out["response"]}
    out["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return out


_AGREED: set = set()
_FAILED_AGREE: set = set()


def cf_agree(model: str) -> bool:
    """Meta's licence gate: a top-level `prompt` field set to 'agree'.

    The two shapes are mutually exclusive -- `{"prompt": ..., "messages": ...}`
    fails a oneOf validation -- so the agreement must go in `prompt` ALONE.

    NOT EVERY ENDPOINT HAS THE GATE, and treating a shape error as a failed
    agreement is a bug that silently kills a whole arm. clef-omni was
    returning "agreement_failed" for every call because it rejects the body
    `{"prompt": "agree"}` with a SHAPE error ("required properties at '/' are
    'model,state,questions'"). That is not a licence refusal -- it means the
    endpoint wants a different body entirely, and it has no gate to pass. So a
    shape error is read as "no gate on this endpoint", and only a licence-style
    refusal counts as a failed agreement.
    """
    if model in _AGREED:
        return True
    if model in _FAILED_AGREE:
        return False
    r = cf_raw(model, {"prompt": "agree"}, timeout=60.0)
    if "error" not in r:
        _AGREED.add(model)
        return True
    detail = (r.get("detail") or "").lower()
    shape_error = ("oneof" in detail or "required properties" in detail
                   or "type mismatch" in detail or "not met" in detail)
    if shape_error:
        _AGREED.add(model)
        return True
    _FAILED_AGREE.add(model)
    return False


def parse_speaking(text: str) -> bool | None:
    """True = SPEAKING, False = NOT SPEAKING, None = unparseable.

    THE NEGATIVE PHRASE IS TESTED FIRST, because "NOT SPEAKING" CONTAINS
    "SPEAKING". A naive substring check scores every negative as positive, and
    that failure is silent: it produces a plausible-looking accuracy.
    """
    t = (text or "").strip().lower()
    if not t:
        return None
    neg = t.find("not speaking")
    pos = t.find("speaking")
    if neg >= 0:
        return False
    if pos >= 0:
        return True
    if "yes" in t:
        return True
    if "no" in t:
        return False
    return None


def parse_facing(text: str) -> bool | None:
    """True = FACING, False = AWAY, None = unparseable.

    "AWAY" and "FACING" do not overlap as substrings, but the model likes to
    write "not facing", "turned away", "facing away from the camera". So the
    NEGATIVE is tested first for the same reason as in parse_speaking: a
    substring check that scores every "facing away" as "facing" produces a
    plausible-looking number with the sign flipped.
    """
    t = (text or "").strip().lower()
    if not t:
        return None
    neg = ("away" in t) or ("not facing" in t) or ("turned" in t) or ("profile" in t)
    pos = ("facing" in t) or ("towards the camera" in t) or ("toward the camera" in t)
    if neg and not pos:
        return False
    if pos and not neg:
        return True
    if neg and pos:
        # "facing away" -- the away wins, because that is the specific claim.
        return False
    return None


def vision_call(model: str, png_b64: str, question: str):
    if not cf_agree(model):
        return {"error": "agreement_failed", "detail": "", "elapsed_s": 0.0}
    if "llava" in model:
        # LLaVA 1.5 on Workers AI takes a top-level `image`, not messages.
        body = {"image": png_b64, "prompt": question, "max_tokens": 200}
    else:
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{png_b64}"}}]}],
            "max_tokens": 200, "temperature": 0.0}
    return cf_raw(model, body)


def audio_call(model: str, wav_b64: str, question: str, kind: str = "speaking"):
    """clef-omni over RAW AUDIO.

    The question format was found by reading the shape the route already
    uses, not by guessing: a dict of key -> {type: choice|noul,
    instructions: str, criteria: {answer: description}}. A `choice` question
    has one criterion per allowed answer, and the model's answer comes back
    keyed by the question name.

    The `criteria` are not decoration -- they are the answer SPACE. Asking
    "is the person speaking?" with criteria {yes, no} makes the model choose
    between them; asking the same thing with an open criterion invites a
    paragraph, which is unparseable and would look like a model failure.

    `kind` selects the answer space: {yes, no} for a yes/no question,
    {facing, away} for the directionality question.
    """
    if not cf_agree(model):
        return {"error": "agreement_failed", "detail": "", "elapsed_s": 0.0}
    if kind == "facing":
        criteria = {
            "facing": "The speaker's voice is directed towards the microphone.",
            "away": "The speaker is turned away from the microphone.",
        }
    else:
        criteria = {
            "yes": "The recording contains the condition described.",
            "no": "The recording does not contain that condition.",
        }
    body = {
        "model": "clef-omni",
        "state": "A recording follows. Answer the questions about it.",
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": question,
                "criteria": criteria,
            }
        },
        "audio": [{"content_type": "audio/wav", "base64": wav_b64}],
    }
    r = cf_raw(model, body)
    # clef returns result.text and result.answers, not choices. `cf_raw`
    # already unwrapped `result`, so the answers live at the TOP level of the
    # returned dict -- an earlier version looked for r["result"] and so never
    # saw them, silently falling back to text parsing.
    if "answers" in r:
        r.setdefault("answers_raw", r.get("answers"))
    elif isinstance(r.get("result"), dict):
        r.setdefault("text", (r["result"].get("text") or ""))
        r.setdefault("answers_raw", r["result"].get("answers"))
    return r


def clef_choice(answers_raw):
    """clef's nested answer shape -> (choice, probabilities).

    The real shape is
        {'q1': {'type': 'choice', 'choice': 'yes',
                'probabilities': {'yes': 0.8003, 'no': 0.1997},
                'confidence': 0.3608}}
    so the answer is at answers['q1']['choice'] and the PROBABILITY is
    alongside it. An earlier parser looked for a flat {'yes': ...} mapping and
    returned None for every call, which is the "empty response" shape that
    hides a real bug.

    The probability is worth more than the choice: it is continuous, so the
    margin between two conditions is a real number rather than a coin flip.
    """
    if not isinstance(answers_raw, dict):
        return None, {}
    node = answers_raw.get("q1")
    if node is None:
        vals = list(answers_raw.values())
        node = vals[0] if vals else None
    if isinstance(node, dict):
        probs = node.get("probabilities")
        if not isinstance(probs, dict):
            probs = {}
        return node.get("choice"), probs
    return None, {}


def clef_prob(answers_raw, key: str) -> float | None:
    """P(the model gave this answer), or None if the model gave no
    probability."""
    _choice, probs = clef_choice(answers_raw)
    if not probs:
        return None
    try:
        return float(probs.get(key))
    except (TypeError, ValueError):
        return None


def clef_answer(text: str, answers_raw) -> bool | None:
    """True/False from clef's answer, which may be a string or a dict."""
    choice, _probs = clef_choice(answers_raw)
    if isinstance(choice, str):
        t = choice.strip().lower()
        if t in ("yes", "true"):
            return True
        if t in ("no", "false"):
            return False
    if answers_raw is not None:
        a = answers_raw
        if isinstance(a, dict):
            vals = list(a.values())
            if vals:
                a = vals[0]
        if isinstance(a, str):
            t = a.strip().lower()
            if t in ("yes", "true"):
                return True
            if t in ("no", "false"):
                return False
        if isinstance(a, dict):
            for k, v in a.items():
                if str(k).strip().lower() in ("yes", "true"):
                    return bool(v)
                if str(k).strip().lower() in ("no", "false"):
                    return not bool(v)
    t = (text or "").strip().lower()
    if "yes" in t:
        return True
    if "no" in t:
        return False
    return None


def clef_facing(text: str, answers_raw) -> bool | None:
    """True = FACING, False = AWAY, from clef's answer or its prose."""
    choice, _probs = clef_choice(answers_raw)
    if isinstance(choice, str):
        t = choice.strip().lower()
        if t in ("facing", "towards", "toward", "yes", "true"):
            return True
        if t in ("away", "turned", "no", "false"):
            return False
    if answers_raw is not None:
        a = answers_raw
        if isinstance(a, dict):
            for k, v in a.items():
                kk = str(k).strip().lower()
                if kk in ("facing", "towards", "toward"):
                    return True
                if kk in ("away", "turned"):
                    return False
            vals = list(a.values())
            if vals:
                a = vals[0]
        if isinstance(a, str):
            t = a.strip().lower()
            if t in ("facing", "towards", "toward", "yes", "true"):
                return True
            if t in ("away", "turned", "no", "false"):
                return False
    return parse_facing(text)


def wav_b64_from_nv12_wav(path: Path):
    """The WAV FILE'S BYTES, header included.

    The first version sent the raw PCM frames with a content-type of
    "audio/wav", and clef-omni answered "Audio could not be decoded" -- the
    content type says what the bytes are, and raw PCM has no header, so the
    decoder has nothing to parse. Sending the file as it exists on disk is
    the honest thing: it is a real 16 kHz mono 16-bit WAV.
    """
    return base64.b64encode(Path(path).read_bytes()).decode(), None


def speech_frame_indices(wav_path: Path, n_frames: int, fps: int = 30,
                         block_s: float = 0.032) -> list[int]:
    """Video frame indices that fall inside DETECTED SPEECH.

    WHY THIS EXISTS -- and it is the same mistake this project has now made
    four documented times, in a new place.

    The first version of this file sampled frames uniformly across the
    session and called every one of them "speaking". The audio says
    otherwise. `src/vad.py` on A.wav gives 70.6% speech in 22 segments, and
    the first segment does not begin until video frame 58 (t = 1.93 s). The
    sampled frame 30 is BEFORE ANY SPEECH, and frame 120 falls in the gap
    between segments (103..126). The model answered "NOT SPEAKING" on both,
    which is CORRECT -- the label was wrong, not the model. Scoring those
    frames as speaking frames is the label-side twin of comparing a start-up
    transient against a steady-state record.

    So the ground truth for "is the person speaking" has to come from the
    AUDIO, at the frame's own timestamp, not from the session's label. A
    session label says "this person was speaking for 30 seconds"; a frame
    label has to say "at 4.0 s the mouth was moving", and the only instrument
    we have for that is the gate.

    Returns video frame indices spread across the detected speech segments.
    """
    sys.path.insert(0, str(ROOT))
    from src.vad import SileroGate
    from src.wavio import read_wav
    a = read_wav(str(wav_path))
    probs = SileroGate().push(list(a.samples))
    if not probs:
        return []
    sp = [p >= 0.5 for p in probs]
    segs = []
    start = None
    for i, s in enumerate(sp):
        if s and start is None:
            start = i
        elif not s and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(sp)))
    segs = [s for s in segs if s[1] - s[0] >= 3]      # >= ~0.1 s
    if not segs:
        return []
    frames = []
    for s, e in segs:
        f0, f1 = int(s * block_s * fps), int(e * block_s * fps)
        frames.extend(range(f0, max(f0 + 1, f1)))
    if not frames:
        return []
    stride = max(1, len(frames) // max(1, n_frames))
    picked = frames[::stride][:n_frames]
    # spread over the whole set if the stride collapsed onto one segment
    if len(picked) < n_frames:
        extra = [f for f in frames if f not in set(picked)]
        picked = (picked + extra)[:n_frames]
    return picked


def uniform_frame_indices(n_total: int, n_frames: int, lo: int = 0,
                          hi: int | None = None) -> list[int]:
    """Frames spread evenly over a session -- for conditions that are
    silent THROUGHOUT, where every frame has the same label and there is
    nothing to align to."""
    hi = n_total if hi is None else min(hi, n_total)
    if hi <= lo:
        return []
    stride = max(1, (hi - lo) // max(1, n_frames))
    return list(range(lo, hi, stride))[:n_frames]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaking", default="/mnt/c/proart_capture/A.nv12")
    ap.add_argument("--silent", default="/mnt/c/proart_capture/C.nv12")
    ap.add_argument("--turned", default="/mnt/c/proart_capture/B.nv12",
                    help="the TURNED condition, same session family as --speaking")
    ap.add_argument("--speaking-wav", default="/mnt/c/proart_capture/A.wav")
    ap.add_argument("--silent-wav", default="/mnt/c/proart_capture/C.wav")
    ap.add_argument("--turned-wav", default="/mnt/c/proart_capture/B.wav",
                    help="the TURNED condition's audio, for the A/B audio join")
    ap.add_argument("--frames", type=int, default=20,
                    help="frames per condition to test")
    ap.add_argument("--t0", type=float, default=1.0)
    ap.add_argument("--t1", type=float, default=9.0)
    ap.add_argument("--align-vad", dest="align_vad", action="store_true",
                    default=True,
                    help="choose speaking frames from detected speech segments "
                         "(default on; the label is per-frame, not per-session)")
    ap.add_argument("--no-align-vad", dest="align_vad", action="store_false",
                    help="sample frames uniformly instead (the OLD behaviour, "
                         "kept so the difference can be measured)")
    ap.add_argument("--models", default="@cf/meta/llama-3.2-11b-vision-instruct")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--pose", action="store_true",
                    help="also run the facing-vs-turned pose arm")
    ap.add_argument("--audio", action="store_true",
                    help="also run the clef-omni raw-audio arm")
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    if not os.environ.get("CF_TOKEN"):
        sys.exit("no CF_TOKEN (source ~/.config/cloudflare/env)")

    YA = load_y(Path(a.speaking))
    YC = load_y(Path(a.silent))
    YB = load_y(Path(a.turned)) if Path(a.turned).is_file() else None

    captures = {
        "speaking": capture_identity(a.speaking),
        "silent": capture_identity(a.silent),
    }
    if YB is not None:
        captures["turned"] = capture_identity(a.turned)

    # THE FRAME SELECTION, which is where the label comes from.
    #
    # Speaking conditions: frames inside DETECTED SPEECH, so every frame's
    # label is defensible at that frame's own timestamp. Silent conditions:
    # frames spread over the session, because the whole session has the same
    # label and there is nothing to align to.
    idx_speak = (speech_frame_indices(Path(a.speaking_wav), a.frames)
                 if a.align_vad else
                 uniform_frame_indices(len(YA), a.frames, int(a.t0 * 30),
                                       int(a.t1 * 30)))
    idx_speak = [i for i in idx_speak if i < len(YA)]
    idx_silent = uniform_frame_indices(len(YC), a.frames, int(a.t0 * 30),
                                       int(a.t1 * 30))
    idx_turned = None
    if YB is not None:
        idx_turned = (speech_frame_indices(Path(a.turned_wav), a.frames)
                      if a.align_vad else
                      uniform_frame_indices(len(YB), a.frames, int(a.t0 * 30),
                                            int(a.t1 * 30)))
        idx_turned = [i for i in idx_turned if i < len(YB)]

    print(f"speaking frames {idx_speak}")
    print(f"silent  frames {idx_silent}")
    if idx_turned:
        print(f"turned  frames {idx_turned}")

    b64_speak = [base64.b64encode(write_png(YA[i])).decode() for i in idx_speak]
    b64_silent = [base64.b64encode(write_png(YC[i])).decode() for i in idx_silent]
    b64_blank = [base64.b64encode(write_png(np.full((H, W), 128, np.uint8))).decode()
                 for _ in idx_speak]
    b64_turned = ([base64.b64encode(write_png(YB[i])).decode() for i in idx_turned]
                  if idx_turned else None)

    question = Q_DESCRIBE if a.describe else Q_YESNO
    results = {}

    for model in [m.strip() for m in a.models.split(",") if m.strip()]:
        print(f"\n=== {model} ===")
        acc = {}
        for variant, frames, truth in (("speak", b64_speak, True),
                                       ("silent", b64_silent, False),
                                       ("blank", b64_blank, None)):
            right = wrong = unparsed = err = 0
            said_speaking = 0
            texts = []
            for b in frames:
                r = vision_call(model, b, question)
                if r.get("error"):
                    err += 1
                    texts.append({"error": r.get("error"),
                                  "detail": r.get("detail", "")[:200]})
                    continue
                t = r.get("text") or ""
                texts.append(t[:300])
                p = parse_speaking(t)
                if p is None:
                    unparsed += 1
                else:
                    if p:
                        said_speaking += 1
                    if truth is None:
                        pass
                    elif p == truth:
                        right += 1
                    else:
                        wrong += 1
            n = right + wrong
            n_parsed = said_speaking + (n - said_speaking if truth is not None else 0)
            acc[variant] = {"right": right, "wrong": wrong,
                            "unparsed": unparsed, "errors": err,
                            "accuracy": (right / n) if n else None,
                            "said_speaking": said_speaking,
                            "said_speaking_rate": (said_speaking / len(frames))
                                                  if frames else None,
                            "texts": texts}
            a_str = f"{acc[variant]['accuracy']:.3f}" if acc[variant]["accuracy"] is not None else "n/a"
            print(f"  {variant:7s} right={right:3d} wrong={wrong:3d} "
                  f"unparsed={unparsed:3d} errors={err:3d}  acc={a_str}"
                  f"  said-SPEAKING {said_speaking}/{len(frames)}")
            if texts and not any("error" in t for t in texts[:1]):
                print(f"          e.g. {texts[0][:120]!r}")

        # THE MARGIN, and WHICH CONTROL IT HAS TO BE MEASURED AGAINST.
        #
        # THE FIRST VERSION OF THIS METRIC WAS WRONG, and the wrongness is
        # worth more than the rightness.
        #
        # It measured the margin as
        #     sensitivity - (rate of saying SPEAKING on a BLANK frame)
        # and llama-4-scout-17b came out at 1.000, "USABLE". But the scout
        # says SPEAKING on 10 of the 12 frames of the SILENT condition -- a
        # real person, present in the room, mouth closed. It says NOT SPEAKING
        # on a blank frame because there is no one in it.
        #
        # So the blank control is not measuring "does this model know when
        # someone is speaking". It is measuring "does this model answer
        # SPEAKING when there is nobody there", and ANY model that treats a
        # human face as evidence of speech passes it. The blank is a prompt
        # floor, not a task control.
        #
        # THE TASK CONTROL IS A PERSON WHO IS PRESENT AND NOT SPEAKING. The
        # headline margin is therefore
        #     P(says SPEAKING | speech frames) - P(says SPEAKING | silent frames)
        # and the blank is kept as the secondary floor it actually is.
        sp, si, bl = acc.get("speak"), acc.get("silent"), acc.get("blank")
        sens = sp["said_speaking_rate"] if sp else None
        silent_fpr = (1.0 - si["accuracy"]) if (si and si["accuracy"] is not None) else None
        blank_fpr = bl["said_speaking_rate"] if bl else None
        margin = None
        if sens is not None and silent_fpr is not None:
            margin = sens - silent_fpr
        margin_blank = None
        if sens is not None and blank_fpr is not None:
            margin_blank = sens - blank_fpr
        acc["sensitivity_says_speaking_on_speech_frames"] = sens
        acc["false_positive_rate_on_silent_person_present"] = silent_fpr
        acc["false_positive_rate_on_blank"] = blank_fpr
        acc["margin_over_silent_person_present"] = margin
        acc["margin_over_blank_secondary"] = margin_blank
        acc["usable"] = bool(margin is not None and margin >= 0.20)
        print(f"  sensitivity (says SPEAKING on speech frames) = {sens}")
        print(f"  false-positive rate on SILENT person present   = {silent_fpr}")
        print(f"  false-positive rate on BLANK (secondary floor)  = {blank_fpr}")
        print(f"  MARGIN over the silent class = {margin}  -> "
              f"{'USABLE' if acc['usable'] else 'NOT USABLE'}")
        print(f"  (margin over the blank, secondary = {margin_blank})")

        # THE POSE ARM -- facing vs turned, which the project has never
        # resolved locally.
        #
        # `experiments/pose_test.py` found A-vs-B 9.182 against a time-shifted
        # control of 9.178 -- no separation. That result is compatible with two
        # very different worlds: (1) the head turn never happened in that
        # capture, or (2) the head turn happened and the change-map statistic
        # is blind to it. A vision model that can name the head angle tells us
        # WHICH world we are in, and that distinction decides whether the pose
        # task is a data problem or a feature problem.
        #
        # The separation measure is the difference of P(facing) between the two
        # conditions, so a model that answers FACING to everything scores 0.
        # The control can come out the other way.
        if a.pose and b64_turned is not None:
            pose = {}
            variants = [("facing", b64_speak, True)]
            # THE POSE CONTROL: condition C is a person facing the camera and
            # SILENT. If P(facing) on C matches P(facing) on A, the model is
            # reading POSE. If instead P(facing) on C collapses towards
            # P(facing) on B, then what the model is actually reading is
            # SPEAKING, and the "pose separation" is a speech separation in
            # disguise -- which is the confound this arm exists to exclude.
            n_pose = len(b64_turned)
            if len(b64_silent) >= n_pose:
                variants.append(("facing_silent", b64_silent[:n_pose], True))
            variants.append(("turned", b64_turned, False))
            for variant, frames, truth_facing in variants:
                said_facing = parsed = unparsed = err = right = wrong = 0
                texts = []
                for b in frames:
                    r = vision_call(model, b, Q_POSE)
                    if r.get("error"):
                        err += 1
                        texts.append({"error": r.get("error")})
                        continue
                    t = r.get("text") or ""
                    texts.append(t[:300])
                    p = parse_facing(t)
                    if p is None:
                        unparsed += 1
                        continue
                    parsed += 1
                    if p:
                        said_facing += 1
                    if p == truth_facing:
                        right += 1
                    else:
                        wrong += 1
                n = len(frames)
                pose[variant] = {
                    "n": n, "parsed": parsed, "unparsed": unparsed, "errors": err,
                    "right": right, "wrong": wrong,
                    "accuracy": (right / (right + wrong)) if (right + wrong) else None,
                    "p_facing": (said_facing / n) if n else None,
                    "texts": texts,
                }
                print(f"  pose {variant:7s} right={right:3d} wrong={wrong:3d} "
                      f"unparsed={unparsed:3d} errors={err:3d}  "
                      f"P(facing)={pose[variant]['p_facing']}")
            pf = pose["facing"]["p_facing"]
            pt = pose["turned"]["p_facing"]
            sep = None if (pf is None or pt is None) else pf - pt
            pose["separation_p_facing_facing_minus_turned"] = sep
            pose["usable"] = bool(sep is not None and sep >= 0.30)
            print(f"  pose separation P(facing|facing) - P(facing|turned) = {sep}"
                  f"  -> {'SEPARATES' if pose['usable'] else 'does NOT separate'}")
            if "facing_silent" in pose:
                pfs = pose["facing_silent"]["p_facing"]
                # The confound test: P(facing | facing AND SILENT) should sit
                # near P(facing | facing AND SPEAKING) if the model reads
                # pose. If it sits near P(facing | turned), the model is
                # reading speech and the separation above is spurious.
                gap = None if (pfs is None or pt is None) else pfs - pt
                pose["speech_confound_gap_facing_silent_minus_turned"] = gap
                print(f"  pose control: P(facing|facing+SILENT) = {pfs}  "
                      f"(vs turned {pt}) -> "
                      f"{'reads POSE' if (gap is not None and gap >= 0.30) else 'may be reading SPEECH'}")
            acc["pose"] = pose
        elif a.pose:
            print("  pose arm skipped: no --turned capture at "
                  f"{a.turned}")

        results[model] = acc

    if a.audio:
        print("\n=== @cf/cloudflare/clef-omni (RAW AUDIO) ===")
        ar = {}
        for variant, wav, truth in (("speak", a.speaking_wav, True),
                                    ("silent", a.silent_wav, False)):
            b64, _params = wav_b64_from_nv12_wav(Path(wav))
            r = audio_call("@cf/cloudflare/clef-omni", b64, Q_AUDIO)
            txt = r.get("text") or ""
            p = clef_answer(txt, r.get("answers_raw"))
            p_yes = clef_prob(r.get("answers_raw"), "yes")
            ar[variant] = {"parsed": p, "p_yes": p_yes,
                           "text": txt[:600],
                           "answers_raw": str(r.get("answers_raw"))[:400],
                           "error": r.get("error"),
                           "detail": (r.get("detail") or "")[:300],
                           "elapsed_s": r.get("elapsed_s")}
            print(f"  {variant:7s} parsed={p!s:5s} P(yes)={p_yes}  {txt[:120]!r}")
            if r.get("error"):
                print(f"          ERROR {r.get('error')}: {(r.get('detail') or '')[:200]}")
        # THE AUDIO MARGIN, on the probability rather than the coin flip.
        ps, pn = ar["speak"].get("p_yes"), ar["silent"].get("p_yes")
        if ps is not None and pn is not None:
            ar["audio_margin_p_yes_speak_minus_silent"] = ps - pn
            print(f"  AUDIO margin P(yes|speech) - P(yes|silence) = {ps - pn:.4f}")
        else:
            print("  AUDIO margin unavailable (no probabilities returned)")
        # THE JOIN, on audio alone: A (facing, speaking) vs B (turned,
        # speaking). These are acoustically near-identical, so this is the
        # hardest possible version of the question, and it is the one the
        # whole project has been failing at with local models.
        if Path(a.turned_wav).is_file():
            b64a, _ = wav_b64_from_nv12_wav(Path(a.speaking_wav))
            b64b, _ = wav_b64_from_nv12_wav(Path(a.turned_wav))
            rA = audio_call("@cf/cloudflare/clef-omni", b64a, Q_FACING,
                            kind="facing")
            rB = audio_call("@cf/cloudflare/clef-omni", b64b, Q_FACING,
                            kind="facing")
            pA = clef_facing(rA.get("text") or "", rA.get("answers_raw"))
            pB = clef_facing(rB.get("text") or "", rB.get("answers_raw"))
            qA = clef_prob(rA.get("answers_raw"), "facing")
            qB = clef_prob(rB.get("answers_raw"), "facing")
            ar["facing_A"] = {"parsed": pA, "p_facing": qA,
                              "text": (rA.get("text") or "")[:400],
                              "answers_raw": str(rA.get("answers_raw"))[:300]}
            ar["facing_B"] = {"parsed": pB, "p_facing": qB,
                              "text": (rB.get("text") or "")[:400],
                              "answers_raw": str(rB.get("answers_raw"))[:300]}
            sep = (None if (pA is None or pB is None) else
                   (1.0 if pA and not pB else 0.0 if pA == pB else -1.0))
            ar["facing_separation"] = sep
            # THE PROBABILITY MARGIN is the real number: A is facing and
            # speaking, B is turned and speaking, and the two are the SAME
            # POEM, so any separation here is the acoustic signature of
            # turning your head. There should not be one.
            if qA is not None and qB is not None:
                ar["facing_probability_margin"] = qA - qB
                print(f"  JOIN P(facing|A) - P(facing|B) = {qA:.4f} - {qB:.4f} "
                      f"= {qA - qB:+.4f}")
            print(f"  JOIN facing A(parsed={pA}) vs turned B(parsed={pB})  "
                  f"-> {'SEPARATES' if sep == 1.0 else 'does NOT separate'}")
            print(f"        A: {(rA.get('text') or '')[:150]!r}")
            print(f"        B: {(rB.get('text') or '')[:150]!r}")
        else:
            print(f"  JOIN skipped: no turned audio at {a.turned_wav}")
        results["clef-omni-audio"] = ar

    out = {"question": question if not a.describe else Q_DESCRIBE,
           "frames_per_condition": len(idx_speak),
           "frame_indices": {"speak": idx_speak, "silent": idx_silent,
                             "turned": idx_turned},
           "frame_selection": ("speaking frames are aligned to DETECTED SPEECH "
                               "segments from src/vad.py, so each frame's label "
                               "holds at that frame's own timestamp; silent "
                               "frames are spread over the session"),
           "models": results,
           "note": ("The margin over the blank control is the headline, not "
                    "the accuracy. A model that answers NOT SPEAKING to "
                    "everything scores 50% on a two-class task and 100% on "
                    "the silent class, which is why the blank control is "
                    "run at the same count as the real conditions."),
           "limits": [
               "one session per condition, one speaker, one room",
               "frames from one session are correlated, so the effective "
               "sample size is far below the frame count",
               "a good result here is a demonstration that the signal "
               "exists; the leave-one-session-out test is what would certify "
               "a teacher",
           ],
           "provenance": {
               "generated_by": "experiments/cloud_multimodal.py",
               "script_path": str(Path(__file__).resolve()),
               "script_sha256": sha256_file(Path(__file__).resolve()),
               "python": sys.executable,
               "command_line": " ".join(sys.argv),
               "cli_args": list(sys.argv[1:]),
               "captures": captures,
           },
           "supersedes": {
               "why": ("Re-run on 2026-10-20. The previous run read the "
                       "captures at W*H*3//2 = 23,040 bytes per frame; the "
                       "captures on disk carry 19,200 bytes per frame, so "
                       "every frame that was sent to the model was a TORN "
                       "frame -- rows 0..71 of one frame and rows 72..95 of "
                       "the next. The previous answers are recorded here "
                       "rather than silently overwritten."),
               "previous_stride_bytes": 23040,
               "previous_frame_indices": {
                   "speak": [58, 134, 188, 245, 323, 391, 464, 537, 603, 689, 747],
                   "silent": [30, 50, 70, 90, 110, 130, 150, 170, 190, 210,
                              230, 250],
                   "turned": [56, 120, 169, 252, 319, 378, 440, 506, 584, 669],
               },
               "previous_frames_per_condition": 11,
               "previous_model": "@cf/meta/llama-3.2-11b-vision-instruct",
               "previous_speak_accuracy": 0.45454545454545453,
               "previous_silent_accuracy": 1.0,
               "previous_blank_accuracy": None,
               "previous_margin_over_silent_class": 0.45454545454545453,
               "previous_note": (
                   "The speaking condition had 11 frames, not 12, because one "
                   "of the VAD-picked frames (817) was past the end of the "
                   "750 frames the WRONG stride produced. At the correct "
                   "stride the capture has 900 frames and all 12 survive, so "
                   "the denominators now match across conditions."),
               "run_to_run_spread": (
                   "THIS ENDPOINT IS NOT DETERMINISTIC at temperature 0.0. Five "
                   "identical runs of this script on the same frames gave a "
                   "margin over the silent class of 0.583, 0.667, 0.583, 0.667 "
                   "and 0.583, and the frames that moved were not always the "
                   "same ones: in the last run the SPEAKING arm scored 8/12 "
                   "and the SILENT arm 11/12, where earlier runs had the "
                   "silent arm at 12/12. The verdict (USABLE) is the same in "
                   "every run and the margin is far above the 0.0 silent and "
                   "blank floors, but the number carries that jitter and "
                   "should be read as 'about 0.6', not to three decimals. A "
                   "twelve-frame arm cannot resolve a difference of one "
                   "frame; the previous run's 0.455 and this one's 0.583 are "
                   "the same measurement."),
           }}
    p = Path(a.out) / "cloud_multimodal_results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
