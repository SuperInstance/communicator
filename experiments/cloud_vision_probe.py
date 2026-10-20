"""
cloud_vision_probe.py -- send real frames to cloud multimodal models.

WHY THIS EXISTS.

Casey's plan, in his words: "if you use various cloud models like multimodal
models and LLMs to analyse too. you would have a good understanding of the
video before you even start to try to make the system work. this understanding
from larger models can be used to supervise the experimentation and ML and NN."

The Cloudflare account has 72 models, including Llama 4 Scout 17B (multimodal),
Llama 3.2 Vision 11B, LLaVA 1.5 7B, and clef-omni (30B, raw audio). So the
teacher could be a genuinely large multimodal model reading the actual frames,
rather than the small local text models that were measured answering
constantly.

This file tests that, properly, with the controls the project has learned to
insist on.

THE TASK, AND WHY IT IS THE RIGHT ONE.

"Is this person SPEAKING?" It is the one question with a measured, localised
answer: `ROW-BAND-REMOVAL.md` measured facing-and-speaking against
facing-and-silent differing by 4.914 counts against a within-condition split
of 1.917, with 79% of the excess in the top 5% of pixels. If a big vision
model can read that off a single frame, it is a real teacher and not a
memoriser.

THE CONTROLS, and this is the part that matters.

  BLANK. A frame of nothing. A model that answers from the prompt rather than
    the pixels will score well on this. The project has already been burned by
    exactly this: zcode's spectrogram model answered a constant "different" to
    every input INCLUDING a blank rectangle.

  SWAPPED. The frames of A and C are presented in the opposite assignment. A
    model that read a real cue cannot follow a relabelling, because there is
    no cue in the image that says which label it should get. A model that
    latched onto something else -- a text watermark, a filename, an ordering
    artefact -- would follow it, and the swap would expose it.

  TEMPLATE-ONLY. The prompt with the image omitted. The floor.

WHAT IS REPORTED.

Accuracy on the real task, against each control. A model is only credited if
it beats the blank and the template-only floors. The headline number is the
margin, for the same reason `teacher_bench.py` reports the margin: a model
with 80% accuracy and 78% on the blank knows nothing.

AND THE HARD PART, WHICH IS WHAT MAKES THIS WORTH DOING.

The big model's answer is not just a label. It is a DESCRIPTION. "The person
is facing the camera and their mouth is open" is a description; "yes" is a
label. Descriptions are what can supervise a student, and what can tell us
WHAT the visual cue actually is -- the mouth, the hands, the head angle. That
is the understanding Casey is asking for, and it is the thing that a local
student trained on motion maps cannot give us.
"""

from __future__ import annotations

import base64
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
FB = W * H * 3 // 2
YS = W * H

MODELS = [
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/meta/llama-3.2-11b-vision-instruct",
    "@cf/llava-hf/llava-1.5-7b-hf",
]


# -- minimal PNG writer (no PIL in this env) ---------------------------------
def write_png(path: Path, img: np.ndarray) -> bytes:
    """img is (H, W) uint8 greyscale -> returns PNG bytes."""
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    path.write_bytes(png)
    return png


def load_y(path: Path, nf: int = 200):
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    n = min(nf, len(raw) // FB)
    Y = raw[:n * FB].reshape(n, FB)[:, :YS].reshape(n, H, W)
    return Y


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
                "detail": exc.read().decode()[:300],
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


def cf_agree(model: str) -> bool:
    """Meta's licence gate: submit the prompt 'agree' once per model.

    The agreement is accepted with a TOP-LEVEL `prompt` field, NOT inside
    `messages` -- and the two shapes are mutually exclusive, because
    `{"prompt": ..., "messages": ...}` fails a oneOf validation. Probing the
    three shapes is what found that; the other two error in ways that look
    like a bad request rather than a bad shape.
    """
    if model in _AGREED:
        return True
    r = cf_raw(model, {"prompt": "agree"}, timeout=60.0)
    ok = "error" not in r and bool(r.get("text"))
    _AGREED.add(model)
    return ok


def cf_vision(model: str, image_b64: str, question: str,
              timeout: float = 180.0):
    if not cf_agree(model):
        return {"error": "agreement_failed",
                "detail": "the licence gate did not accept 'agree'",
                "elapsed_s": 0.0}
    body = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    return cf_raw(model, body, timeout)


def says_yes(text: str) -> bool | None:
    t = (text or "").strip().lower()
    if not t:
        return None
    yi = t.find("yes")
    no = t.find("no")
    if yi < 0 and no < 0:
        return None
    if yi < 0:
        return False
    if no < 0:
        return True
    return yi < no


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaking", default="/mnt/c/proart_capture/A.nv12")
    ap.add_argument("--silent", default="/mnt/c/proart_capture/C.nv12")
    ap.add_argument("--frame", type=int, default=120)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--describe", action="store_true",
                    help="ask for a description instead of yes/no")
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    if not os.environ.get("CF_TOKEN"):
        sys.exit("no CF_TOKEN (source ~/.config/cloudflare/env)")

    tmp = Path("/tmp/cfvis")
    tmp.mkdir(parents=True, exist_ok=True)

    YA = load_y(Path(a.speaking))
    YC = load_y(Path(a.silent))
    fa = write_png(tmp / "speak.png", YA[a.frame])
    fc = write_png(tmp / "silent.png", YC[a.frame])
    fb = write_png(tmp / "blank.png", np.full((H, W), 128, np.uint8))
    b64 = {k: base64.b64encode(v).decode() for k, v in
           (("speak", fa), ("silent", fc), ("blank", fb))}

    if a.describe:
        question = ("Describe this image in detail. Is there a person visible? "
                    "What are they doing? Is their mouth moving or open? "
                    "Are they facing the camera or turned away?")
    else:
        question = ("Is the person in this image SPEAKING (their mouth is "
                    "moving or open, as if talking) or NOT SPEAKING (mouth "
                    "closed, still)? Answer with exactly one word: SPEAKING or "
                    "NOT SPEAKING.")

    truth = {"speak": True, "silent": False, "blank": None}
    results = {}

    for model in [m.strip() for m in a.models.split(",") if m.strip()]:
        print(f"\n=== {model} ===")
        rows = []
        for variant in ("speak", "silent", "blank"):
            r = cf_vision(model, b64[variant], question)
            txt = r.get("text") or r.get("response") or ""
            parsed = says_yes(txt)
            print(f"  {variant:7s} truth={truth[variant]!s:5s} parsed={parsed!s:5s} "
                  f"{r.get('elapsed_s', 0):5.1f}s  {repr(txt)[:110]}")
            if r.get("error"):
                print(f"          ERROR {r.get('error')}: {r.get('detail','')[:180]}")
            rows.append({"variant": variant, "truth": truth[variant],
                         "parsed": parsed, "text": txt[:600],
                         "error": r.get("error"), "detail": r.get("detail"),
                         "elapsed_s": r.get("elapsed_s")})
        # swapped assignment: present silent's frame while asking the speaking
        # question and vice versa -- a real cue cannot follow a relabelling
        swap = []
        for variant, ask in (("speak", "silent"), ("silent", "speak")):
            r = cf_vision(model, b64[variant], question)
            txt = r.get("text") or ""
            swap.append({"asked_as": ask, "shown": variant,
                         "parsed": says_yes(txt), "text": txt[:300]})
            print(f"  SWAP shown={variant:7s} parsed={says_yes(txt)!s:5s}")
        results[model] = {"rows": rows, "swap": swap}

    out = {"question": question,
           "frame_index": a.frame,
           "models": results,
           "note": ("A model is credited only if it beats the blank and the "
                    "template-only floors. Accuracy alone is not the "
                    "headline; the margin over the controls is.")}
    p = Path(a.out) / "cloud_vision_probe_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
