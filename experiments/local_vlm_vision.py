"""
local_vlm_vision.py -- is the LLM director's run-to-run instability the
SERVING STACK's, or is the TASK ambiguous?

WHY THIS FILE EXISTS.

`docs/research/LLM-DIRECTOR.md` reports, from
`experiments/director_repeatability.py`:

    hosted @cf/openai/gpt-oss-120b, BLIND audit, 8 runs, temperature 0
        4 distinct headlines, spread 0.619
    deterministic stub director, same scorer, same 8 runs
        spread 0.000

`docs/research/LLM-DIRECTOR.md` had already claimed determinism at
temperature 0. The measurement said otherwise, and the note says so. But the
note ALSO says "this is hosted-inference non-determinism", and that sentence
was an inference, not a measurement. Two explanations fit the same eight
numbers:

    (A) the serving stack returns different answers for identical requests
    (B) the task is ambiguous -- any model asked it would answer differently,
        because the question admits several defensible readings

They make DIFFERENT predictions. A local model with greedy decoding is
deterministic by construction: same weights, same input, same arithmetic,
same answer. If a local model at temperature 0 gives a spread of 0.000 on the
same material, then (B) is not the explanation and the instability belongs to
the serving stack. If the local model ALSO gives a large spread at
temperature 0, then the task itself is the problem -- and the hosted number
was never about the hosting.

That is the whole experiment. The vision arm is here too because the same
question has a second face: this project has a cloud vision bench
(`experiments/cloud_multimodal.py`) and no local one, and a local vision
model that answers "SPEAKING" to a closed mouth is a fact about the LOCAL
model worth having regardless of how the determinism question lands.

WHAT THIS FILE MEASURES.

PART 1 -- the vision task, locally, on the GPU.
    The same question the cloud bench used, on the luma plane of the project's
    own captures, at the same frame counts:

        "Is the person in this image SPEAKING (their mouth is moving or
         open, as if talking) or NOT SPEAKING (mouth closed, still)? Answer
         with exactly one word: SPEAKING or NOT SPEAKING."

    Three conditions, EQUAL denominators: `speak` (A.nv12, VAD-aligned),
    `silent` (C.nv12, person present and not speaking), `blank` (flat grey).
    Reported: sensitivity, FPR on silent, FPR on blank, and the margin
    (sens - FPR_silent), because the margin over a real person who is NOT
    speaking is the only control that can fail.

    THE DEGENERATE BASELINES ARE RECORDED, COMPUTED, NOT ASSERTED:
        always-SPEAKING      -> FPR 1.000 on silent and on blank
        always-NOT-SPEAKING  -> sensitivity 0.000

    AT BOTH FRAME STRIDES. The captures on disk are 19,200 bytes per frame
    (luma + interleaved U,V), 900 frames, 30.02 fps. The repo's readers
    assume the standard NV12 stride of 23,040 bytes, 750 frames, and at
    that stride 80% of every frame is a SPLICE of two frames. So the
    stride is DETECTED per file and the vision arm is run at the clean
    stride AND at the repo's assumed stride. The two are not the same
    measurement and the difference between them is a result.

PART 2 -- the determinism question, which is the point of the lane.
    The IDENTICAL request, N=8 times, in two regimes:
        greedy   do_sample=False                      (temperature 0)
        sampled  do_sample=True, temperature 0.8, seed fixed
    on BOTH tasks: the vision task above, and the director's own BLIND audit
    prompt (the one that produced the hosted spread of 0.619). The audit
    prompt is BYTE-IDENTICAL to the hosted one -- its SHA-256 is checked
    against the committed artefact, not merely asserted.

    THE AUDIT ARM IS TEXT-ONLY, AND THAT MATTERS FOR WHAT IT CAN SETTLE.
    The BLIND prompt is the ten claims and nothing else: no research note,
    no result file, and certainly no video frame. So the frame-stride
    defect cannot reach it, and a finding about the audit cannot be
    explained away by the frames being torn. The vision arm is where the
    stride question lives.

THE CONTROL, and it is not optional.
A deterministic stub director -- the repo's own `stub_verdicts`, scored by the
repo's own `debrief.score_audit` -- must produce a spread of exactly 0.000
when scored N times. A harness that cannot produce 0.000 on a deterministic
input is not measuring anything. The same control is run for the vision task,
where the stub answer is a constant.

WHAT THIS DOES NOT CLAIM.

It does not claim the hosted model is broken. It measures whether the
instability is a property of the MODEL or of the SERVING STACK, using a model
whose decoding is deterministic by construction as the instrument. If the
local model is ALSO unstable at temperature 0, that is a finding and it kills
hypothesis (A) -- it is reported as such, not as a failure of the experiment.

It also does not claim the local model is a good director or a good vision
model. A 2B model on one 30-second session of one person is a demonstration,
not a measurement of the task.

THE SECONDARY MODEL, AND HOW THE ARTEFACT IS ASSEMBLED.

A second local model (`Qwen/Qwen2.5-VL-3B-Instruct`) is measured as a
control on the model rather than on the task: it is larger, and it needs 21
of 41 modules offloaded to the CPU because its fp16 weights are 7.51 GB and
the GPU has 6.14 GB. It is ~13x slower per token -- 234 s for one 549-token
audit answer against 24 s -- so it is measured in a SEPARATE invocation and
its block is MERGED into the artefact by `--merge-secondary-from`, which
copies the block whole and recomputes every summary from it. The merge is a
scripted step with a recorded source path, not a hand edit.

`--recompute` does the summary recomputation alone, from an artefact's OWN
recorded runs. That is how a change in the ANALYSIS code is applied to a
measurement that has already been taken, without re-running the model.

THE IDENTITY OF EVERYTHING.

Every number below is recorded with the model id and revision, the dtype,
the device, the exact prompt text, the generation parameters, the seed, the
SHA-256 of every image's bytes, the frame indices, and the per-condition
counts. A result whose inputs cannot be named cannot be re-checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

# The repo's own audit builder and scorer, imported read-only. `debrief` is
# standard-library only, so this stays importable on a machine with no GPU
# and no torch -- which is what lets `tests/test_local_vlm_vision.py` run
# against the committed artefact.
import debrief  # noqa: E402

# ------------------------------------------------------------------ constants

W, H = 160, 96
YS = W * H                  # 15360 luma bytes per frame
CH = YS // 4                # 3840 chroma bytes (interleaved U,V) in the
                            # legacy capture format
STRIDE_LEGACY = YS + CH     # 19200 -- what the captures on disk ACTUALLY use
STRIDE_REPO_ASSUMED = YS + YS // 2   # 23040 -- what the repo's readers assume
STRIDE_LUMA_ONLY = YS       # 15360 -- a LUMA-ONLY stride. Nothing in this
                            # repo writes one (CORRECTION 2026-10-20: the
                            # old comment here claimed corpus_capture_win.py
                            # did, which is false -- it writes .npz change
                            # maps). Kept as a measured third arm.

# The strides this file will try, and the one the DETECTOR picks.
STRIDE_CANDIDATES = (STRIDE_LEGACY, STRIDE_REPO_ASSUMED, STRIDE_LUMA_ONLY)
STRIDE_NAMES = {
    STRIDE_LEGACY: "clean_19200",
    STRIDE_REPO_ASSUMED: "torn_23040",
    STRIDE_LUMA_ONLY: "luma_only_15360",
}

FPS_NOMINAL = 30.0          # the camera rate the capture script asked for
GATE_BLOCK_S = 0.032        # the Silero frame, 32 ms
GATE_THRESHOLD = 0.5        # src/vad.py DEFAULT_THRESHOLD

CAPTURE = Path("/mnt/c/proart_capture")

# THE PROMPT, byte for byte, the one `experiments/cloud_multimodal.py` sends
# as Q_YESNO. Not "the same question" -- the same question.
VISION_PROMPT = (
    "Is the person in this image SPEAKING (their mouth is moving or "
    "open, as if talking) or NOT SPEAKING (mouth closed, still)? "
    "Answer with exactly one word: SPEAKING or NOT SPEAKING.")

# The two decoding regimes. Both are recorded in the results file verbatim.
REGIMES = {
    "greedy": {"do_sample": False, "temperature": None,
               "top_p": None, "top_k": None,
               "note": "greedy: no sampling, so no temperature is applied"},
    "sampled": {"do_sample": True, "temperature": 0.8,
                "top_p": 1.0, "top_k": 0,
                "note": ("pure temperature sampling; the seed is set ONCE per "
                         "regime so the N runs are N independent draws from "
                         "one seeded stream, the local analogue of N separate "
                         "requests")},
}

# The models this lane tried, in the order the lane says to try them.
MODEL_ORDER = ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2-VL-2B-Instruct"]

AUDIT_SEED = 20261019     # the seed the committed hosted measurement used


# ------------------------------------------------------------------ stimulus

def load_luma(path: Path, stride: int) -> np.ndarray:
    """The LUMA PLANE of a raw frame stream, as (n_frames, H, W) uint8.

    THE STRIDE IS AN ARGUMENT, NOT A CONSTANT, and that is the whole point.

    Three frame formats exist in this project and they do not agree:

      19200  the legacy captures (`/mnt/c/proart_capture/*.nv12`, written by
             `nv12small.py`: 15360 luma bytes + 3840 bytes of interleaved
             U,V). 900 frames, 30.02 fps.
      23040  what the repo's readers ASSUME: standard NV12 4:2:0, luma then
             interleaved chroma. Reading a legacy capture at this stride
             gives 750 frames of which 80% are TORN -- rows 0..71 from one
             real frame and rows 72..95 from the next.
      15360  luma only. NO SCRIPT IN THIS REPO WRITES A LUMA-ONLY FRAME
             STREAM -- the old comment here said `corpus_capture_win.py`
             did, which is false (it writes .npz change maps). This stride is
             kept as a measured third arm.

    So the stride is DETECTED (`detect_stride`), not assumed, and whichever
    stride is used is recorded in the results file.
    """
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    total = len(raw) // stride
    return raw[:total * stride].reshape(total, stride)[:, :YS].reshape(total, H, W)


def detect_stride(path: Path, candidates=STRIDE_CANDIDATES) -> dict:
    """Which stride reads this file as a SEQUENCE OF FRAMES.

    The test is physical, not documentary: at the right stride consecutive
    frames are nearly identical, and at the wrong stride every frame is a
    splice of two frames and the mean absolute difference between
    consecutive frames is enormous. Measured on the captures in this repo:

        A.nv12   stride 23040 -> 43.560    stride 19200 -> 0.545
        B.nv12   stride 23040 -> 37.740    stride 19200 -> 0.433
        C.nv12   stride 23040 -> 42.935    stride 19200 -> 0.367

    Every candidate's score is returned, so the decision is auditable and
    not a black box.
    """
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    scores = {}
    for c in candidates:
        n = len(raw) // c
        if n < 2:
            scores[c] = None
            continue
        Y = raw[:n * c].reshape(n, c)[:, :YS].reshape(n, H, W).astype(np.int16)
        scores[c] = float(np.abs(np.diff(Y, axis=0)).mean())
    usable = {c: s for c, s in scores.items() if s is not None}
    best = min(usable, key=usable.get) if usable else None
    return {"path": str(path), "bytes": int(len(raw)),
            "mean_abs_frame_diff_by_stride": {str(c): scores[c] for c in candidates},
            "detected_stride": best,
            "detected_stride_name": STRIDE_NAMES.get(best),
            "n_frames_at_detected_stride": (len(raw) // best) if best else None}


def write_png(img: np.ndarray) -> bytes:
    """A greyscale PNG, written here rather than imported so that the bytes
    this experiment sends are recorded with their own SHA-256.

    `experiments/cloud_multimodal.py` has a copy of this writer; it is
    duplicated, not shared, because the point of the SHA-256 is that the
    image bytes are attributable to THIS file.
    """
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ the gate

def gate_probabilities(wav_path: Path) -> list[float]:
    """One speech probability per 32 ms, from the project's OWN gate.

    `src/vad.py` + `models/silero_vad.onnx`. The rule this project uses is
    that a frame label comes from the gate at that frame's OWN timestamp, and
    not from the session's label: "this person was speaking for 30 seconds" is
    not the same claim as "at 4.0 s the mouth was moving".
    """
    from src.vad import SileroGate
    from src.wavio import read_wav
    a = read_wav(str(wav_path))
    return list(SileroGate().push(list(a.samples)))


def frame_is_speech(frame_index: int, probs: list[float], fps: float) -> bool:
    """Is the gate saying SPEECH at THIS frame's own timestamp?

    The frame's centre is `(i + 0.5)/fps`. The gate block containing that
    instant decides. The fps is the one the READING implies -- the frame
    count over the audio duration -- not a nominal constant, because a
    wrong stride means a wrong frame count and therefore a wrong clock.
    """
    t = (frame_index + 0.5) / fps
    b = int(t // GATE_BLOCK_S)
    if b >= len(probs):
        b = len(probs) - 1
    if b < 0:
        return False
    return bool(probs[b] >= GATE_THRESHOLD)


def speech_frame_indices(probs: list[float], n_frames: int, n_total: int,
                         fps: float) -> list[int]:
    """Speech frames, spread over the detected speech.

    Stride over the frames the gate labels as speech, so the condition
    covers the whole session rather than the first utterance.
    """
    frames = [i for i in range(n_total) if frame_is_speech(i, probs, fps)]
    if not frames:
        return []
    stride = max(1, len(frames) // max(1, n_frames))
    picked = frames[::stride][:n_frames]
    if len(picked) < n_frames:
        extra = [f for f in frames if f not in set(picked)]
        picked = (picked + extra)[:n_frames]
    return picked


def uniform_frame_indices(n_total: int, n_frames: int) -> list[int]:
    """Frames spread evenly over the whole session.

    For a condition that is silent THROUGHOUT, every frame has the same
    label and there is nothing to align to.
    """
    stride = max(1, n_total // max(1, n_frames))
    return list(range(0, n_total, stride))[:n_frames]


# ------------------------------------------------------------------ the parser

def parse_speaking(text: str) -> bool | None:
    """True = SPEAKING, False = NOT SPEAKING, None = unparseable.

    THE NEGATIVE PHRASE IS TESTED FIRST, because "NOT SPEAKING" CONTAINS THE
    WORD "SPEAKING". A substring check scores every negative as a positive,
    and it fails silently: the result looks like an accuracy.
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
    return None


def parse_answer_word(text: str) -> str:
    """The parsed answer as a STRING, for the determinism runs.

    "SPEAKING", "NOT SPEAKING", or "UNPARSEABLE". The string is what is
    compared across runs; the numeric score below is only the encoding of
    it, so that a spread can be computed.
    """
    p = parse_speaking(text)
    if p is None:
        return "UNPARSEABLE"
    return "SPEAKING" if p else "NOT SPEAKING"


# ------------------------------------------------------------------ verdicts

# One verdict object: {"index": 3, "verdict": "supported", "reason": "..."}
VERDICT_RE = re.compile(
    r'\{\s*"index"\s*:\s*(\d+)\s*,\s*"verdict"\s*:\s*"([^"]*)"', re.S)


def extract_verdicts(text: str):
    """The verdicts in a model response, and WHICH parser produced them.

    `debrief.extract_json` returns the FIRST balanced `{...}` pair it finds.
    When a model emits an unescaped double quote inside a `reason` string,
    the OUTER object is not valid JSON, the first pair the scanner finds is
    an INNER verdict object, and the repo's caller correctly rejects that as
    unparseable -- but the response actually CONTAINS all ten verdicts, and
    the strict parser throws them away.

    So this falls back to scanning for every object carrying an "index" and
    a "verdict". The repo's strict parser is ALWAYS tried first, so a
    response the repo can read is read by the repo, and which parser each
    run used is recorded. The fallback is applied to every run, greedy and
    sampled alike, so it cannot favour one regime over the other.

    Returns (verdict_object_or_None, parser_name, n_verdicts_recovered).
    """
    obj = debrief.extract_json(text)
    if isinstance(obj, dict) and isinstance(obj.get("verdicts"), list):
        return obj, "repo_extract_json", len(obj["verdicts"])
    found = {}
    for m in VERDICT_RE.finditer(text or ""):
        idx = int(m.group(1))
        if idx not in found:
            found[idx] = {"index": idx, "verdict": m.group(2), "reason": ""}
    if not found:
        return None, "none", 0
    verdicts = [found[i] for i in sorted(found)]
    return {"verdicts": verdicts}, "fallback_scan", len(verdicts)


# ------------------------------------------------------------------ statistics

def spread(values) -> float | None:
    """max - min. The repo's headline spread, and the vision analogue.

    `director_repeatability.summarise_runs` computes exactly this over the
    per-run headlines; the vision arm uses the same definition over the
    per-run parsed answers, which is what makes the two rows of the
    comparison table commensurable.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return max(vals) - min(vals)


def answer_score(word: str) -> float | None:
    """SPEAKING -> 1.0, NOT SPEAKING -> 0.0, UNPARSEABLE -> None."""
    if word == "SPEAKING":
        return 1.0
    if word == "NOT SPEAKING":
        return 0.0
    return None


def vision_degenerate_baselines(n: int) -> dict:
    """The two answers that carry no information, COMPUTED not asserted.

    always-SPEAKING and always-NOT-SPEAKING, scored through the same
    metric as the model, so the margin can be read against them.
    """
    out = {}
    for name, said in (("always_speaking", True), ("always_not_speaking", False)):
        sens = 1.0 if said else 0.0
        # always-SPEAKING says SPEAKING on a silent frame TOO, so its false
        # positive rate is 1.000; always-NOT-SPEAKING says nothing on a
        # silent frame, so its rate is 0.000.
        fpr = 1.0 if said else 0.0
        out[name] = {
            "n_per_condition": n,
            "sensitivity": sens,
            "fpr_silent": fpr,
            "fpr_blank": fpr,
            "margin_sens_minus_fpr_silent": sens - fpr,
        }
    return out


# ------------------------------------------------------------------ the model

def model_identity(model_id: str) -> dict:
    """The identity of the weights: revision, and a SHA-256 over them.

    The revision alone is a NAME; the content hash is what says the bytes
    were the ones measured.
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


def load_model(model_id: str, dtype: str = "float16", device_map: str = "cuda:0"):
    """Load the model and the processor. torch is imported HERE, not at the
    top of the file, so that the pure functions above stay importable on a
    machine with no GPU and no torch."""
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
    info = {
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
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32)
        if torch.cuda.is_available() else None,
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)
        if torch.cuda.is_available() else None,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic)
        if torch.cuda.is_available() else None,
    }
    return model, processor, info


def _generate(model, processor, inputs, regime: str, max_new_tokens: int,
              seed: int | None):
    """One generation, under the named regime.

    The seed is applied ONCE per regime by the caller, so the N runs inside
    a regime are N draws from one seeded stream. Re-seeding before every run
    would make the sampled regime identical to itself and the measurement
    vacuous.
    """
    import torch
    reg = REGIMES[regime]
    kwargs = {"max_new_tokens": max_new_tokens,
              "do_sample": bool(reg["do_sample"])}
    if reg["do_sample"]:
        kwargs["temperature"] = reg["temperature"]
        kwargs["top_p"] = reg["top_p"]
        kwargs["top_k"] = reg["top_k"]
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    gen = out[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(gen, skip_special_tokens=True)
    return {"text": text, "n_new_tokens": int(len(gen)),
            "elapsed_s": round(time.time() - t0, 3)}


def vision_inputs(processor, png_bytes: bytes):
    """The processor's inputs for one PNG and the prompt."""
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": VISION_PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    return processor(text=[text], images=[img], return_tensors="pt")


def text_inputs(processor, prompt: str):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    return processor(text=[text], return_tensors="pt")


def move_to_device(inputs, device: str):
    out = {}
    for k, v in inputs.items():
        try:
            out[k] = v.to(device)
        except Exception:  # noqa: BLE001 -- non-tensor entries
            out[k] = v
    return out


# ------------------------------------------------------------------ the arms

def run_vision_task(model, processor, conditions: dict, device: str) -> dict:
    """The vision task: every frame of every condition, one question each."""
    out = {}
    for name, cond in conditions.items():
        said_speaking = 0
        unparsed = 0
        errors = 0
        texts = []
        for png in cond["pngs"]:
            try:
                inputs = move_to_device(vision_inputs(processor, png), device)
                r = _generate(model, processor, inputs, "greedy", 16, None)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                texts.append({"error": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            p = parse_speaking(r["text"])
            if p is None:
                unparsed += 1
            elif p:
                said_speaking += 1
            texts.append({"text": r["text"][:120], "parsed": p,
                          "n_new_tokens": r["n_new_tokens"]})
        n = len(cond["pngs"])
        out[name] = {
            "n": n,
            "said_speaking": said_speaking,
            "unparsed": unparsed,
            "errors": errors,
            "said_speaking_rate": (said_speaking / n) if n else None,
            "texts": texts,
        }
    speak = out.get("speak", {})
    silent = out.get("silent", {})
    blank = out.get("blank", {})
    sens = speak.get("said_speaking_rate")
    fpr_silent = silent.get("said_speaking_rate")
    fpr_blank = blank.get("said_speaking_rate")
    margin = (None if (sens is None or fpr_silent is None)
              else sens - fpr_silent)
    margin_blank = (None if (sens is None or fpr_blank is None)
                    else sens - fpr_blank)
    return {
        "conditions": out,
        "sensitivity_says_speaking_on_speech_frames": sens,
        "false_positive_rate_on_silent_person_present": fpr_silent,
        "false_positive_rate_on_blank": fpr_blank,
        "margin_over_silent_person_present": margin,
        "margin_over_blank_secondary": margin_blank,
        "usable_margin_ge_0_20": bool(margin is not None and margin >= 0.20),
        "degenerate_baselines": vision_degenerate_baselines(
            len(conditions.get("speak", {}).get("pngs", []))),
    }


def run_vision_determinism(model, processor, conditions: dict, device: str,
                            runs: int, seed: int) -> dict:
    """The IDENTICAL request, N times, per condition, in both regimes.

    One fixed image per condition, the same prompt, and the same bytes every
    time -- the local analogue of the hosted bench sending the same request N
    times.
    """
    import torch
    out = {}
    for name, cond in conditions.items():
        png = cond["pngs"][0]
        inputs = move_to_device(vision_inputs(processor, png), device)
        for regime in ("greedy", "sampled"):
            if REGIMES[regime]["do_sample"]:
                torch.manual_seed(seed)
            per_run = []
            for _ in range(runs):
                r = _generate(model, processor, inputs, regime, 16, seed)
                word = parse_answer_word(r["text"])
                per_run.append({"text": r["text"][:80], "answer": word,
                                "score": answer_score(word)})
            scores = [p["score"] for p in per_run]
            words = [p["answer"] for p in per_run]
            n_speaking = sum(1 for w in words if w == "SPEAKING")
            out.setdefault(name, {})[regime] = {
                "n_runs": runs,
                "per_run": per_run,
                "answers": words,
                "n_unparseable": sum(1 for w in words if w == "UNPARSEABLE"),
                "n_distinct_answers": len(set(words)),
                "n_saying_speaking": n_speaking,
                "said_speaking_rate": (n_speaking / len(words)) if words else None,
                "spread": spread(scores),
                "image_sha256": cond["image_sha256"][0],
            }
    return out


def run_audit_determinism(model, processor, device: str, runs: int,
                           seed: int, mode: str = "blind",
                           max_new_tokens: int = 3000) -> dict:
    """The director's own audit, N times, in both regimes.

    The prompt is built by `debrief.build_audit_prompt` -- the repo's own
    function, so the local model is asked EXACTLY what the hosted model was
    asked -- and its SHA-256 is compared with the committed hosted artefact.
    The score is `debrief.score_audit`, and the spread is
    `director_repeatability.summarise_runs`, so the number in the comparison
    table is produced by the same code that produced the hosted number.
    """
    import torch
    import debrief
    import director_repeatability as dr

    pairs = debrief.audit_pairs(AUDIT_SEED)
    prompt, lim = debrief.build_audit_prompt(mode, pairs, None)
    prompt_sha = sha256_hex(prompt.encode())
    inputs = move_to_device(text_inputs(processor, prompt), device)

    out = {"mode": mode, "prompt_sha256": prompt_sha,
           "prompt_chars": len(prompt), "materials_chars": lim,
           "n_claims": len(pairs), "audit_seed": AUDIT_SEED,
           "max_new_tokens": max_new_tokens}

    for regime in ("greedy", "sampled"):
        if REGIMES[regime]["do_sample"]:
            torch.manual_seed(seed)
        scored = []
        runs_detail = []
        for i in range(runs):
            r = _generate(model, processor, inputs, regime, max_new_tokens, seed)
            obj, parser, n_recovered = extract_verdicts(r["text"] or "")
            if not obj or "verdicts" not in obj:
                scored.append({"error": "unparseable"})
                runs_detail.append({"run": i, "error": "unparseable",
                                    "parser": parser,
                                    "n_verdicts_recovered": n_recovered,
                                    "raw": (r["text"] or "")[:300],
                                    "n_new_tokens": r["n_new_tokens"],
                                    "elapsed_s": r["elapsed_s"]})
                continue
            res = debrief.score_audit(pairs, obj, model="local",
                                      mode=mode, lim=lim,
                                      elapsed=r["elapsed_s"], raw=r["text"] or "")
            scored.append(res)
            runs_detail.append({
                "run": i,
                "parser": parser,
                "n_verdicts_recovered": n_recovered,
                "headline": res["headline_sens_plus_spec_minus_1"],
                "sensitivity": res["sensitivity_rejected_false_claims"],
                "specificity": res["specificity_kept_true_claims"],
                "uncertain": res["uncertain"],
                "unparseable": res["unparseable"],
                "verdicts": [row["verdict"] for row in res["rows"]],
                "n_new_tokens": r["n_new_tokens"],
                "elapsed_s": r["elapsed_s"],
                "raw_head": (r["text"] or "")[:300],
            })
        summary = dr.summarise_runs(scored)
        out[regime] = {
            "n_runs": runs,
            "headlines": summary["headlines"],
            "n_distinct_headlines": summary["n_distinct_headlines"],
            "headline_min": summary["headline_min"],
            "headline_max": summary["headline_max"],
            "headline_median": summary["headline_median"],
            "headline_mean": summary["headline_mean"],
            "headline_spread": summary["headline_spread"],
            "n_scored": summary["n_scored"],
            "n_unparseable": summary["n_unparseable"],
            "n_unstable_claims": summary["n_unstable_claims"],
            "n_runs_parsed_by_fallback": sum(
                1 for r in runs_detail if r.get("parser") == "fallback_scan"),
            "per_claim": summary["per_claim"],
            "runs": runs_detail,
        }
    return out


def run_controls(n: int) -> dict:
    """THE CONTROL. Both deterministic stubs, scored N times.

    A deterministic stub director must give an identical headline every
    time, and a deterministic stub vision answer must give an identical
    answer every time. If either moves, the harness is adding variance and
    every number in this file is worthless.
    """
    import debrief
    import director_repeatability as dr
    pairs = debrief.audit_pairs(AUDIT_SEED)
    stub = {"verdicts": [{"index": i, "verdict": "supported", "reason": "stub"}
                         for i in range(len(pairs))]}
    runs = [debrief.score_audit(pairs, stub) for _ in range(n)]
    summary = dr.summarise_runs(runs)
    words = ["SPEAKING"] * n
    return {
        "stub_director": {
            "what": ("the repo's deterministic stub director, scored "
                     f"{n} times with the repo's own scorer "
                     "(debrief.score_audit)"),
            "headlines": summary["headlines"],
            "n_distinct_headlines": summary["n_distinct_headlines"],
            "headline_spread": summary["headline_spread"],
        },
        "stub_vision": {
            "what": ("a constant vision answer, scored by the same spread "
                     "statistic the model runs use"),
            "answers": words,
            "n_distinct_answers": len(set(words)),
            "spread": spread([answer_score(w) for w in words]),
        },
        "degenerate_directors": dr.degenerate_baselines(pairs),
    }


# ------------------------------------------------------------------ the runner

def hosted_reference() -> dict:
    """The hosted row of the comparison table, COPIED from the repo's own
    committed artefact.

    Not re-measured here, and not a new claim: this project already
    measured it, and the number belongs to
    `experiments/director_repeatability_blind.json`. It is copied so the
    table can be read against THIS file, and so a reader can see that the
    two rows came from the same prompt (the prompt SHA-256 is recorded on
    both sides and the test checks they are equal).

    The hosted model at temperature > 0 was never run by this project. It
    cannot be measured from this machine either, so that cell is LEFT EMPTY.
    """
    p = ROOT / "experiments" / "director_repeatability_blind.json"
    if not p.is_file():
        return {"error": f"missing {p}"}
    data = json.loads(p.read_text())
    e = data["results"][0]
    r = e["result"]
    return {
        "source": "experiments/director_repeatability_blind.json",
        "note": ("the repo's COMMITTED measurement, copied here. It was NOT "
                 "re-measured by this file and no Cloudflare token was used."),
        "model": e["model"],
        "mode": e["mode"],
        "n_runs": e["n_requested"],
        "prompt_sha256": e["prompt_sha256"],
        "prompt_chars": e["prompt_chars"],
        "headlines": r["headlines"],
        "n_distinct_headlines": r["n_distinct_headlines"],
        "headline_spread": r["headline_spread"],
        "headline_median": r["headline_median"],
        "control_stub_headline_spread": e["control"]["headline_spread"],
        "sampled": {
            "headline_spread": None,
            "note": ("NOT MEASURED. The hosted model at temperature > 0 was "
                     "never run by this project and cannot be measured from "
                     "this machine. The cell is left empty rather than "
                     "guessed."),
        },
    }


def compute_verdict(out: dict) -> dict:
    """The verdict, DERIVED from the recorded numbers.

    Not asserted. The rule this lane was built to apply:

        local greedy spread == 0.000  ->  (B) is not the explanation, and the
                                         instability belongs to the serving
                                         stack
        local greedy spread >  0      ->  (A) is refuted and the task itself
                                         is ambiguous

    Two guards, because a verdict that cannot come out the other way is not
    a verdict:

      * if the local model produced no scorable output, the comparison is
        UNDETERMINED and no hypothesis is supported;
      * if the local model is DEGENERATE -- the same answer to every claim,
        so its headline is constant for a reason that has nothing to do
        with determinism -- then its stability is not evidence about the
        task's ambiguity, and that is said plainly.
    """
    au = (out.get("determinism") or {}).get("audit") or {}
    g = au.get("greedy") or {}
    s = au.get("sampled") or {}
    hosted = out.get("hosted_reference") or {}

    local_greedy = g.get("headline_spread")
    local_sampled = s.get("headline_spread")
    n_scored = g.get("n_scored", 0)

    verdicts_used = set()
    greedy_verdicts = set()
    for regime in ("greedy", "sampled"):
        for r in ((au.get(regime) or {}).get("runs") or []):
            for v in (r.get("verdicts") or []):
                verdicts_used.add(v)
                if regime == "greedy":
                    greedy_verdicts.add(v)
    # The claim rests on the GREEDY arm, so the question "does this model
    # discriminate at all?" is asked of the greedy arm alone: a model that
    # answers every claim the same way has a constant headline for a
    # reason that has nothing to do with determinism.
    discriminates = len(greedy_verdicts) > 1

    undetermined = local_greedy is None or n_scored == 0
    hosted_spread = hosted.get("headline_spread")
    hosted_txt = ("none" if hosted_spread is None
                  else f"{hosted_spread:.3f}")
    if undetermined:
        a = b = None
        reason = ("the local model produced no scorable audit output, so the "
                  "comparison is UNDETERMINED and neither hypothesis is "
                  "supported by this measurement")
    elif local_greedy == 0.0 and discriminates:
        a, b = True, False
        reason = (f"the local model at temperature 0 gave a spread of "
                  f"{local_greedy:.3f} over {n_scored} scorable runs of the "
                  f"BYTE-IDENTICAL audit prompt, while the hosted model at "
                  f"temperature 0 gave {hosted_txt}. The task is therefore "
                  f"answerable stably by at least one model, so hypothesis "
                  f"(B) -- that the task is so ambiguous that any model "
                  f"differs -- does not explain the hosted spread. The "
                  f"instability is a property of the serving stack.")
    elif local_greedy == 0.0 and not discriminates:
        a = b = None
        reason = (f"the local model at temperature 0 gave a spread of "
                  f"{local_greedy:.3f}, but it answered EVERY claim with the "
                  f"same verdict, so its headline is constant for a reason "
                  f"that has nothing to do with determinism. The comparison "
                  f"is UNDETERMINED: a degenerate answerer cannot show that "
                  f"the task is not ambiguous.")
    else:
        a, b = False, True
        reason = (f"the local model at temperature 0 gave a spread of "
                  f"{local_greedy:.3f} over {n_scored} scorable runs of the "
                  f"BYTE-IDENTICAL audit prompt. A local model with greedy "
                  f"decoding is deterministic by construction, so this is "
                  f"not serving-stack non-determinism: the TASK is what "
                  f"admits several defensible answers, and hypothesis (A) is "
                  f"refuted.")

    vision_spread = {}
    for sname, per_cond in (out.get("determinism_vision_by_stride") or {}).items():
        vision_spread[sname] = {
            c: {"greedy_spread": r["greedy"]["spread"],
                "sampled_spread": r["sampled"]["spread"],
                "n_distinct_greedy_answers": r["greedy"]["n_distinct_answers"]}
            for c, r in per_cond.items()}

    return {
        "local_greedy_audit_spread": local_greedy,
        "local_sampled_audit_spread": local_sampled,
        "hosted_greedy_audit_spread": hosted_spread,
        "hosted_sampled_audit_spread": (hosted.get("sampled") or {}).get(
            "headline_spread"),
        "local_model_n_scored_greedy": n_scored,
        "local_model_verdicts_used": sorted(verdicts_used),
        "local_model_greedy_verdicts_used": sorted(greedy_verdicts),
        "local_model_discriminates": discriminates,
        "hypothesis_A_hosting_not_task": a,
        "hypothesis_B_task_ambiguous": b,
        "reason": reason,
        "vision_spread_by_stride": vision_spread,
        "vision_note": ("the vision arm cannot decide hypothesis (A) versus "
                        "(B) about the DIRECTOR, because the director's "
                        "audit is text-only and no frame enters it. What it "
                        "can show is whether a torn frame is what makes a "
                        "vision answer unstable."),
    }


def build_conditions(n_frames: int, stride: int) -> tuple[dict, dict]:
    """The three conditions at ONE stride, with equal denominators.

    The fps is derived from the READING: the frame count the stride
    implies, over the duration of the matching audio. A wrong stride means
    a wrong frame count and therefore a wrong clock, so the clock is
    derived, never assumed.
    """
    from src.wavio import read_wav
    probs = gate_probabilities(CAPTURE / "A.wav")
    probs_c = gate_probabilities(CAPTURE / "C.wav")
    dur = len(read_wav(str(CAPTURE / "A.wav")).samples) / 16000.0
    YA = load_luma(CAPTURE / "A.nv12", stride)
    YC = load_luma(CAPTURE / "C.nv12", stride)
    n_total = YA.shape[0]
    fps = n_total / dur

    idx_speak = speech_frame_indices(probs, n_frames, n_total, fps)
    idx_silent = uniform_frame_indices(YC.shape[0], n_frames)
    blank = np.full((H, W), 128, np.uint8)

    conds = {
        "speak": {"source": "A.nv12", "truth": "speaking",
                  "pngs": [write_png(YA[i]) for i in idx_speak]},
        "silent": {"source": "C.nv12", "truth": "not speaking",
                   "pngs": [write_png(YC[i]) for i in idx_silent]},
        "blank": {"source": "(synthetic flat grey 128)", "truth": "no person",
                  "pngs": [write_png(blank) for _ in range(n_frames)]},
    }
    for name, c in conds.items():
        c["image_sha256"] = [sha256_hex(p) for p in c["pngs"]]
        c["n"] = len(c["pngs"])

    meta = {
        "capture_dir": str(CAPTURE),
        "width": W, "height": H,
        "stride": stride,
        "stride_name": STRIDE_NAMES.get(stride, str(stride)),
        "n_frames": int(n_total),
        "audio_duration_s": round(dur, 4),
        "fps_implied_by_this_stride": round(fps, 4),
        "fps_nominal": FPS_NOMINAL,
        "luma_plane_bytes": YS,
        "vad": {
            "source": "src/vad.py + models/silero_vad.onnx",
            "block_s": GATE_BLOCK_S,
            "threshold": GATE_THRESHOLD,
            "rule": ("a frame is labelled speaking iff the gate block "
                     "containing the frame's centre timestamp (i+0.5)/fps "
                     "has probability >= 0.5, at the fps THIS READING "
                     "implies"),
            "A.wav_speech_fraction": round(
                sum(1 for p in probs if p >= GATE_THRESHOLD) / max(1, len(probs)), 4),
            "C.wav_speech_fraction": round(
                sum(1 for p in probs_c if p >= GATE_THRESHOLD) / max(1, len(probs_c)), 4),
            "A.nv12_frames_labelled_speaking": int(sum(
                1 for i in range(n_total) if frame_is_speech(i, probs, fps))),
            "gate_prob_at_chosen_speak_frames": [
                round(probs[min(len(probs) - 1,
                               int(((i + 0.5) / fps) // GATE_BLOCK_S))], 4)
                for i in idx_speak],
        },
        "conditions": {
            name: {
                "source": c["source"], "truth": c["truth"], "n": c["n"],
                "frame_indices": (idx_speak if name == "speak"
                                  else (idx_silent if name == "silent" else None)),
                "image_sha256": c["image_sha256"],
            } for name, c in conds.items()
        },
    }
    return conds, meta


# ------------------------------------------------------------------ the record

CORRECTIONS = [
    {
        "date": "2026-10-20",
        "what": "the frame stride of the raw captures",
        "wrong": ("this file's FIRST run read /mnt/c/proart_capture/*.nv12 at "
                  "23,040 bytes per frame -- the standard NV12 4:2:0 stride "
                  "the repo's readers assume -- and used 25.00 fps."),
        "right": ("the captures on disk are 19,200 bytes per frame: 15,360 "
                  "of luma followed by 3,840 bytes of interleaved U,V, "
                  "written by /mnt/c/proart_capture/nv12small.py. That is 900 "
                  "frames over the 29.981 s of the matching audio, i.e. "
                  "30.02 fps. At the 23,040 stride 80% of every frame is a "
                  "SPLICE of two frames -- rows 0..71 from one and rows "
                  "72..95 from the next."),
        "evidence": ("mean |difference| between consecutive frames. "
                     "A.nv12 43.560 at 23,040 against 0.545 at 19,200; "
                     "B.nv12 37.740 against 0.433; C.nv12 42.935 against "
                     "0.367. The stride is now DETECTED per file, by that "
                     "physical test, and recorded."),
        "effect": ("the VISION numbers of the first run are SUPERSEDED and "
                   "are re-measured here at BOTH strides, so the clean and "
                   "the torn readings are two cells of one table rather "
                   "than one silent replacement of the other. The AUDIT "
                   "numbers are NOT affected: the audit prompt is text-only "
                   "and no frame enters it."),
    },
]

SUPERSEDED_FIRST_RUN = {
    "date": "2026-10-20",
    "status": "SUPERSEDED -- kept visible, not deleted",
    "model": "Qwen/Qwen2-VL-2B-Instruct",
    "stride": 23040,
    "fps": 25.0,
    "why_superseded": ("the captures were read at the wrong stride, so the "
                       "frames were torn; and the audit runs were parsed "
                       "with the repo's strict parser alone, which rejects "
                       "the local model's malformed JSON."),
    "vision_task": {
        "sensitivity": 1.0,
        "fpr_silent": 1.0,
        "fpr_blank": 1.0,
        "margin": 0.0,
    },
    "vision_determinism_spread": {
        "greedy": {"speak": 0.0, "silent": 0.0, "blank": 0.0},
        "sampled": {"speak": 1.0, "silent": 1.0, "blank": 1.0},
    },
    "audit_strict_parser": {
        "greedy": {"n_scored": 0, "n_unparseable": 8, "headlines": [],
                   "spread": None},
        "sampled": {"n_scored": 5, "n_unparseable": 3,
                    "headlines": [-1.0, -0.23809523809523814, -1.0,
                                  -0.04761904761904767,
                                  -0.09523809509523834],
                    "spread": 0.9523809523809523},
    },
    "controls": {"stub_director_headline_spread": 0.0,
                 "stub_vision_spread": 0.0},
}


def model_attempts(out: dict) -> list:
    """Every model this lane tried, what happened to it, and why.

    The task order was the 3B first and the 2B second. Both ran. The
    3B is the weaker PRIMARY because its fp16 weights are 7.51 GB and this
    GPU has 6.14 GB, so 17 of its 36 language-model layers run on the CPU:
    a model with half its weights in system RAM is a weaker demonstration
    of "deterministic by construction" than one that is wholly on the GPU.
    """
    attempts = []
    prim = out.get("model_identity") or {}
    prim_rt = out.get("model_runtime") or {}
    attempts.append({
        "model_id": prim.get("model_id"),
        "revision": prim.get("revision"),
        "weights_sha256": prim.get("weights_sha256"),
        "weights_bytes": prim.get("weights_bytes"),
        "role": "PRIMARY",
        "outcome": "ran",
        "failure": None,
        "why": ("fp16 weights are 4.42 GB and the GPU has 6.14 GB, so it "
                "loads WHOLLY onto the GPU. That is the clean demonstration "
                "of a model whose decoding is deterministic by construction."),
        "resolved_device": prim_rt.get("resolved_device"),
        "hf_device_map": prim_rt.get("hf_device_map"),
        "vram_allocated_gb": prim_rt.get("vram_allocated_gb"),
    })
    sec = out.get("secondary_model") or {}
    sec_id = sec.get("model_identity") or {}
    sec_rt = sec.get("model_runtime") or {}
    attempts.append({
        "model_id": sec_id.get("model_id"),
        "revision": sec_id.get("revision"),
        "weights_sha256": sec_id.get("weights_sha256"),
        "weights_bytes": sec_id.get("weights_bytes"),
        "role": "SECONDARY",
        "outcome": "ran" if sec_id else "not run",
        "failure": None,
        "why_not_primary": ("fp16 weights are 7.51 GB and the GPU has "
                            "6.14 GB, so device_map='auto' offloads 17 of "
                            "the 36 language-model layers to the CPU. It "
                            "works, and it is the stronger model, but a "
                            "model that is partly in system RAM is a weaker "
                            "demonstration of deterministic decoding than "
                            "one that is wholly on the GPU, and it is about "
                            "13x slower per token."),
        "resolved_device": sec_rt.get("resolved_device"),
        "hf_device_map": sec_rt.get("hf_device_map"),
        "vram_allocated_gb": sec_rt.get("vram_allocated_gb"),
    })
    attempts.append({
        "model_id": None,
        "role": "SETUP FAILURE",
        "outcome": "fixed",
        "failure": ("transformers 5.20.2 could not build the Qwen2-VL "
                    "processor: 'Qwen2VLVideoProcessor requires the "
                    "Torchvision library but it was not found'."),
        "fix": ("installed torchvision==0.21.0+cu124 into "
                "/home/proart/projects/gpu-venv, which is OUTSIDE the repo. "
                "The repo's own .venv was not touched."),
    })
    return attempts


def comparison_table(out: dict) -> dict:
    """The decisive table, in the artefact, so it cannot be quoted wrong.

    Two arms, because they are different questions:

      AUDIT   the director's own task, text-only, and the SAME prompt
              byte-for-byte as the hosted one. This is the row that decides
              hosted-non-determinism versus task ambiguity.
      VISION  the speaking/silent task, at both strides. This is where the
              clean-versus-torn question lives, because a torn frame is an
              ambiguous picture.
    """
    hosted = out.get("hosted_reference") or {}
    au = (out.get("determinism") or {}).get("audit") or {}
    vt_by = out.get("vision_task_by_stride") or {}
    vd_by = out.get("determinism_vision_by_stride") or {}
    table = {
        "audit_headline_spread": {
            "note": ("the audit is TEXT-ONLY: no frame enters the audit "
                     "prompt, so the stride defect cannot reach this row. "
                     "The prompt is byte-identical to the hosted one -- the "
                     "SHA-256 is recorded on both sides and checked."),
            "greedy": {
                "hosted": hosted.get("headline_spread"),
                "local": (au.get("greedy") or {}).get("headline_spread"),
            },
            "sampled": {
                "hosted": (hosted.get("sampled") or {}).get("headline_spread"),
                "local": (au.get("sampled") or {}).get("headline_spread"),
                "hosted_note": ("NOT MEASURED. The hosted model at "
                                "temperature > 0 was never run by this "
                                "project and cannot be measured from this "
                                "machine. Left empty rather than guessed."),
            },
        },
        "vision_answer_spread": {
            "note": ("the hosted vision bench measured ACCURACY, never "
                     "run-to-run spread, and it read the frames at the "
                     "23,040 stride, so its frames were torn. There is no "
                     "hosted vision spread to compare against."),
            "by_stride": {},
        },
        "vision_task_by_stride": {},
    }
    for sname, vd in vd_by.items():
        table["vision_answer_spread"]["by_stride"][sname] = {
            c: {"greedy_spread": r["greedy"]["spread"],
                "sampled_spread": r["sampled"]["spread"],
                "greedy_answers": r["greedy"]["answers"],
                "sampled_answers": r["sampled"]["answers"],
                "n_runs": r["greedy"]["n_runs"]}
            for c, r in vd.items()}
    for sname, vt in vt_by.items():
        table["vision_task_by_stride"][sname] = {
            "sensitivity": vt["sensitivity_says_speaking_on_speech_frames"],
            "fpr_silent": vt["false_positive_rate_on_silent_person_present"],
            "fpr_blank": vt["false_positive_rate_on_blank"],
            "margin": vt["margin_over_silent_person_present"],
            "degenerate_baselines": vt["degenerate_baselines"],
        }
    sec = out.get("secondary_model") or {}
    sec_au = (sec.get("determinism") or {}).get("audit") or {}
    if sec_au:
        table["audit_headline_spread"]["secondary_model"] = {
            "model_id": (sec.get("model_identity") or {}).get("model_id"),
            "greedy": {"local": (sec_au.get("greedy") or {}).get(
                "headline_spread")},
            "sampled": {"local": (sec_au.get("sampled") or {}).get(
                "headline_spread")},
        }
    if sec:
        table["vision_answer_spread"]["secondary_by_stride"] = {
            sname: {c: {"greedy_spread": r["greedy"]["spread"],
                        "sampled_spread": r["sampled"]["spread"],
                        "n_runs": r["greedy"]["n_runs"]}
                    for c, r in per.items()}
            for sname, per in (sec.get("determinism_vision_by_stride")
                               or {}).items()}
    return table


def merge_secondary(out_path: Path, src_path: Path | None = None) -> dict:
    """Merge a SECONDARY model's block into an existing artefact.

    The secondary model is measured by an earlier run of THIS file, with
    that model as its primary, and the result is merged in here. The merge
    is a scripted step with a recorded source path, not a hand edit: the
    block is copied whole, and the summary blocks are recomputed from it.

    With `src_path` None this only RECOMPUTES the summary blocks from the
    artefact's own recorded runs, which is how a change in the analysis
    code is applied to a measurement that has already been taken.
    """
    out = json.loads(Path(out_path).read_text())
    if src_path is not None:
        src = json.loads(Path(src_path).read_text())
        for key in ("model_identity", "model_runtime", "vision_task",
                    "vision_task_by_stride", "determinism",
                    "determinism_vision_by_stride"):
            if key not in src:
                raise SystemExit(f"{src_path} has no {key}")
        sec = {k: src[k] for k in ("model_identity", "model_runtime",
                                   "vision_task", "vision_task_by_stride",
                                   "determinism", "determinism_vision_by_stride")}
        sec["source_artefact"] = str(src_path)
        sec["source_command"] = src.get("command")
        sec["verdict"] = compute_verdict(sec)
        out["secondary_model"] = sec
        cmds = [c for c in (out.get("commands") or [out.get("command")]) if c]
        if src.get("command") and src.get("command") not in cmds:
            cmds.append(src.get("command"))
        out["commands"] = cmds
    out["hosted_reference"] = hosted_reference()
    out["model_attempts"] = model_attempts(out)
    out["comparison_table"] = comparison_table(out)
    out["verdict"] = compute_verdict(out)
    out["commands_note"] = (
        "TWO INVOCATIONS, because the secondary model is ~13x slower per "
        "token: the first command measured the primary, the second measured "
        "the secondary, and --merge-secondary-from put the secondary's block "
        "into this artefact. The primary invocation was originally launched "
        "with --secondary-model as well; that invocation's secondary arm was "
        "abandoned and the secondary was measured standalone instead. Both "
        "commands are recorded verbatim above.")
    Path(out_path).write_text(json.dumps(out, indent=2, default=str))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct",
                    help="the PRIMARY local model")
    ap.add_argument("--secondary-model", dest="secondary_model", default=None,
                    help="an optional second local model, measured the same "
                         "way, to show the conclusion is not a quirk of one "
                         "model")
    ap.add_argument("--secondary-device-map", dest="secondary_device_map",
                    default="auto")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device-map", dest="device_map", default=None,
                    help="override --device, e.g. 'auto' for CPU offload")
    ap.add_argument("--frames", type=int, default=11,
                    help="frames per condition; EQUAL across conditions")
    ap.add_argument("--runs", type=int, default=8,
                    help="identical requests per determinism regime")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int,
                    default=3000,
                    help="audit generation budget (the hosted call used 3000)")
    ap.add_argument("--strides", default="19200,23040",
                    help="comma-separated frame strides to measure. The first "
                         "is the primary; 19200 is the stride the captures "
                         "ACTUALLY use and 23040 is the one the repo's "
                         "readers assume, which tears 80%% of every frame.")
    ap.add_argument("--vision-only", action="store_true")
    ap.add_argument("--merge-secondary-from", dest="merge_secondary_from",
                    default=None,
                    help="merge the secondary-model block of the artefact at "
                         "PATH into the artefact at --out, and exit. The "
                         "merge is scripted: the block is copied whole and "
                         "the summaries are recomputed from it.")
    ap.add_argument("--recompute", action="store_true",
                    help="recompute the summary blocks of the artefact at "
                         "--out from its own recorded runs, and exit. This "
                         "is how a change in the ANALYSIS code is applied to "
                         "a measurement that has already been taken.")
    ap.add_argument("--out", default=str(ROOT / "experiments"
                                         / "local_vlm_vision_results.json"))
    a = ap.parse_args(argv)

    if a.merge_secondary_from or a.recompute:
        src = Path(a.merge_secondary_from) if a.merge_secondary_from else None
        merged = merge_secondary(Path(a.out), src)
        print(f"{'merged the secondary model from ' + str(src) if src else 'recomputed the summaries of'} "
              f"{a.out}")
        v = merged["verdict"]
        print(f"verdict: A(hosting not task)="
              f"{v['hypothesis_A_hosting_not_task']} "
              f"B(task ambiguous)={v['hypothesis_B_task_ambiguous']}")
        return

    device_map = a.device_map or a.device
    strides = [int(s) for s in str(a.strides).split(",") if s.strip()]
    primary_stride = strides[0]

    print(f"=== LOCAL VLM VISION :: {a.model} ===")
    print("detecting the frame stride of the captures")
    detections = {f"{n}.nv12": detect_stride(CAPTURE / f"{n}.nv12")
                  for n in ("A", "B", "C")}
    for name, d in detections.items():
        print(f"  {name}: detected stride {d['detected_stride']} "
              f"({d['detected_stride_name']})  "
              f"mean|diff| {d['mean_abs_frame_diff_by_stride']}")

    print(f"building the stimulus ({a.frames} frames per condition)")
    conds_by_stride = {}
    stim_by_stride = {}
    for stride in strides:
        conds, meta = build_conditions(a.frames, stride)
        conds_by_stride[stride] = conds
        stim_by_stride[stride] = meta
        print(f"  stride {stride} ({meta['stride_name']}): "
              f"{meta['n_frames']} frames at {meta['fps_implied_by_this_stride']} fps")

    ident = model_identity(a.model)
    print(f"model revision {ident['revision']}  weights sha256 "
          f"{ident['weights_sha256'][:16]}")

    model, processor, minfo = load_model(a.model, a.dtype, device_map)
    print(f"loaded {minfo['class']} on {minfo['resolved_device']} "
          f"vram {minfo['vram_allocated_gb']} GB")

    out = {
        "experiment": "local_vlm_vision",
        "date": time.strftime("%Y-%m-%d"),
        "command": " ".join(sys.argv),
        "question": ("is the director's run-to-run instability a property of "
                     "the hosted serving stack, or of the task?"),
        "corrections": CORRECTIONS,
        "superseded_first_run": SUPERSEDED_FIRST_RUN,
        "model_identity": ident,
        "model_runtime": minfo,
        "prompt": {
            "vision": VISION_PROMPT,
            "vision_sha256": sha256_hex(VISION_PROMPT.encode()),
            "audit": "built by debrief.build_audit_prompt('blind', pairs, None)",
        },
        "regimes": REGIMES,
        "seed": a.seed,
        "n_runs": a.runs,
        "n_frames_per_condition": a.frames,
        "primary_model_id": a.model,
        "primary_stride": STRIDE_NAMES.get(primary_stride, str(primary_stride)),
        "stride_detection": detections,
        "stimulus": stim_by_stride[primary_stride],
        "stimulus_by_stride": {STRIDE_NAMES.get(s, str(s)): m
                               for s, m in stim_by_stride.items()},
    }

    print("running the vision task at every stride")
    vision_task_by_stride = {}
    vision_det_by_stride = {}
    for stride in strides:
        conds = conds_by_stride[stride]
        sname = STRIDE_NAMES.get(stride, str(stride))
        vt = run_vision_task(model, processor, conds, device_map)
        vision_task_by_stride[sname] = vt
        print(f"  {sname}: sens {vt['sensitivity_says_speaking_on_speech_frames']} "
              f"fpr_silent {vt['false_positive_rate_on_silent_person_present']} "
              f"fpr_blank {vt['false_positive_rate_on_blank']} "
              f"margin {vt['margin_over_silent_person_present']}")
        vd = run_vision_determinism(model, processor, conds, device_map,
                                    a.runs, a.seed)
        vision_det_by_stride[sname] = vd
        for cname, r in vd.items():
            print(f"    {sname:16s} {cname:7s} greedy spread {r['greedy']['spread']}"
                  f"  sampled spread {r['sampled']['spread']}")

    pname = STRIDE_NAMES.get(primary_stride, str(primary_stride))
    out["vision_task"] = vision_task_by_stride[pname]
    out["vision_task_by_stride"] = vision_task_by_stride
    out["determinism"] = {"vision": vision_det_by_stride[pname]}
    out["determinism_vision_by_stride"] = vision_det_by_stride

    # CHECKPOINT 1: the vision arms are on disk before the audit starts, so
    # a slow audit cannot take the vision measurement with it.
    p = Path(a.out)
    out["hosted_reference"] = hosted_reference()
    out["model_attempts"] = model_attempts(out)
    out["comparison_table"] = comparison_table(out)
    out["verdict"] = compute_verdict(out)
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"checkpointed the vision arms to {p}")

    print("running the controls")
    out["controls"] = run_controls(a.runs)

    if not a.vision_only:
        print("running the audit determinism (this takes a while)")
        print("  NOTE: the audit is TEXT-ONLY. No frame enters the audit "
              "prompt, so the stride defect cannot reach this arm.")
        out["determinism"]["audit"] = run_audit_determinism(
            model, processor, device_map, a.runs, a.seed,
            max_new_tokens=a.max_new_tokens)
        au = out["determinism"]["audit"]
        print(f"  prompt sha256 {au['prompt_sha256']}")
        print(f"  greedy  headlines {au['greedy']['headlines']}")
        print(f"  greedy  spread {au['greedy']['headline_spread']}")
        print(f"  sampled headlines {au['sampled']['headlines']}")
        print(f"  sampled spread {au['sampled']['headline_spread']}")

    # CHECKPOINT: the primary model's results are on disk before the
    # secondary model starts, so a slow or failing secondary cannot take
    # the primary measurement with it.
    p = Path(a.out)
    out["hosted_reference"] = hosted_reference()
    out["model_attempts"] = model_attempts(out)
    out["comparison_table"] = comparison_table(out)
    out["verdict"] = compute_verdict(out)
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"checkpointed the primary measurement to {p}")

    if a.secondary_model:
        print(f"=== SECONDARY MODEL: {a.secondary_model} ===")
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        sident = model_identity(a.secondary_model)
        smodel, sprocessor, sminfo = load_model(a.secondary_model, a.dtype,
                                                a.secondary_device_map)
        sec = {"model_identity": sident, "model_runtime": sminfo}
        sec["vision_task_by_stride"] = {}
        sec["determinism_vision_by_stride"] = {}
        for stride in strides:
            conds = conds_by_stride[stride]
            sname = STRIDE_NAMES.get(stride, str(stride))
            vt = run_vision_task(smodel, sprocessor, conds, a.secondary_device_map)
            sec["vision_task_by_stride"][sname] = vt
            print(f"  {sname}: sens {vt['sensitivity_says_speaking_on_speech_frames']} "
                  f"fpr_silent {vt['false_positive_rate_on_silent_person_present']} "
                  f"margin {vt['margin_over_silent_person_present']}")
            vd = run_vision_determinism(smodel, sprocessor, conds,
                                        a.secondary_device_map, a.runs, a.seed)
            sec["determinism_vision_by_stride"][sname] = vd
        sec["vision_task"] = sec["vision_task_by_stride"][pname]
        sec["determinism"] = {"vision": sec["determinism_vision_by_stride"][pname]}
        if not a.vision_only:
            sec["determinism"]["audit"] = run_audit_determinism(
                smodel, sprocessor, a.secondary_device_map, a.runs, a.seed,
                max_new_tokens=a.max_new_tokens)
            au2 = sec["determinism"]["audit"]
            print(f"  secondary greedy  headlines {au2['greedy']['headlines']}")
            print(f"  secondary greedy  spread {au2['greedy']['headline_spread']}")
            print(f"  secondary sampled headlines {au2['sampled']['headlines']}")
            print(f"  secondary sampled spread {au2['sampled']['headline_spread']}")
        sec["verdict"] = compute_verdict(sec)
        out["secondary_model"] = sec

    p = Path(a.out)
    out["hosted_reference"] = hosted_reference()
    out["model_attempts"] = model_attempts(out)
    out["comparison_table"] = comparison_table(out)
    out["verdict"] = compute_verdict(out)
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {p}")
    v = out["verdict"]
    print(f"verdict: A(hosting not task)={v['hypothesis_A_hosting_not_task']} "
          f"B(task ambiguous)={v['hypothesis_B_task_ambiguous']}")


if __name__ == "__main__":
    main()
