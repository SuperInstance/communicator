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

THOSE NUMBERS HAVE MOVED, AND THE REASON MATTERS.

They were measured on a torn signal. Re-measured at the correct frame stride
on 2026-10-20 (`experiments/pose_test.py`), facing-and-speaking differs from
facing-and-silent by 0.2506 counts against A's own within-condition split of
0.2006 -- 1.25x, which does not clear the 1.25x margin the test was given
before the numbers were seen -- with 31% of the excess in the top 5% of
pixels. So the camera-channel speech signal is no longer established, and the
question this probe asks of a cloud model is open rather than confirmed. The
probe is still worth running: the point of it is whether a large vision model
can read a cue that the camera channel has not been shown to carry.

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

MODELS = [
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/meta/llama-3.2-11b-vision-instruct",
    "@cf/llava-hf/llava-1.5-7b-hf",
]


def sha256_file(path) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    """The luma plane of the first `nf` frames.

    The frame layout is DETECTED from the file's content by `src/nv12.py` --
    the candidate whose luma has the smallest mean consecutive |diff| wins,
    and an ambiguous file is an error rather than a guess. The captures on
    disk carry 19,200 bytes per 160x96 frame, not the 23,040 a standard-NV12
    reader assumes; read at 23,040 all but every sixth frame is torn. See
    `src/nv12.py` and `docs/research/NV12-STRIDE.md`.
    """
    luma, _layout = read_luma(path, width=W, height=H)
    n = min(nf, len(luma))
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
    """Parse the probe's answer.

    THE PROBE'S OWN VOCABULARY IS "SPEAKING" / "NOT SPEAKING", and the
    parser used to look for "yes"/"no" -- so a perfectly correct
    "SPEAKING." was logged as `parsed=None` and the primary metric of the
    primary probe was measuring nothing. The vocabulary is checked first;
    the yes/no logic is kept as a fallback.

    This is NOT a stride fix and the numbers it changes are not attributed
    to one: it is a separate defect in the measurement, found while
    re-running the probe on the corrected frames.
    """
    t = (text or "").strip().lower()
    if not t:
        return None
    if "not speaking" in t:
        return False
    if "speaking" in t:
        return True
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
    layout_a, layout_c = None, None
    for p, store in ((a.speaking, "a"), (a.silent, "c")):
        _l, lay = read_luma(Path(p), width=W, height=H)
        if store == "a":
            layout_a = lay
        else:
            layout_c = lay
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
           "provenance": {
               "generated_by": "experiments/cloud_vision_probe.py",
               "script_path": str(Path(__file__).resolve()),
               "script_sha256": sha256_file(Path(__file__).resolve()),
               "python": sys.executable,
               "command_line": " ".join(sys.argv),
               "cli_args": list(sys.argv[1:]),
               "captures": {
                   "speaking": {
                       "path": str(Path(a.speaking).resolve()),
                       "sha256": sha256_file(a.speaking),
                       "bytes": Path(a.speaking).stat().st_size,
                       "layout": layout_a.name,
                       "stride_bytes": layout_a.stride,
                       "n_frames": layout_a.n_frames,
                       "layout_evidence": layout_a.as_dict(),
                   },
                   "silent": {
                       "path": str(Path(a.silent).resolve()),
                       "sha256": sha256_file(a.silent),
                       "bytes": Path(a.silent).stat().st_size,
                       "layout": layout_c.name,
                       "stride_bytes": layout_c.stride,
                       "n_frames": layout_c.n_frames,
                       "layout_evidence": layout_c.as_dict(),
                   },
               },
           },
           "supersedes": {
               "why": ("Re-run on 2026-10-20. The previous run read the "
                       "captures at W*H*3//2 = 23,040 bytes per frame; the "
                       "captures on disk carry 19,200 bytes per frame, so the "
                       "frame that was sent to the model was a TORN frame: "
                       "rows 0..71 of one frame and rows 72..95 of the next. "
                       "The previous answers are recorded here rather than "
                       "silently overwritten."),
               "previous_stride_bytes": 23040,
               "previous_frame_index": 120,
               "previous_note": (
                   "A model is credited only if it beats the blank and the "
                   "template-only floors. Accuracy alone is not the headline; "
                   "the margin over the controls is."),
           },
           "note": ("A model is credited only if it beats the blank and the "
                    "template-only floors. Accuracy alone is not the "
                    "headline; the margin over the controls is.")}
    p = Path(a.out) / "cloud_vision_probe_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
