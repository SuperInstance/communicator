#!/usr/bin/env python
"""
Measured latency + behaviour of pipecat's bundled smart-turn v3.2 CPU model.

Everything here is measured on this machine, not quoted from a README.

    cd /home/proart/projects/communicator/docs/research/voiceloop
    /home/proart/projects/glasses/.venv/bin/python bench_smartturn.py

Requires: numpy + onnxruntime (pip install onnxruntime). torch/torchaudio/
librosa/scipy are NOT needed -- pipecat vendored a numpy-only log-mel
extractor precisely so this path stays dependency-free.

Findings written up in SYNTHESIS.md.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import numpy as np
import onnxruntime as ort

REPO = "pipecat-ai/pipecat"
BRANCH = "main"
MODEL_PATH = "src/pipecat/audio/turn/smart_turn/data/smart-turn-v3.2-cpu.onnx"
FEATURES_URL = (
    f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
    "src/pipecat/audio/turn/smart_turn/_whisper_features.py"
)

SR = 16000
WINDOW_S = 8
N = 128000  # 8 s @ 16 kHz -- the model's fixed input window


def fetch() -> tuple[str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    mp = os.path.join(here, "smart-turn-v3.2-cpu.onnx")
    fp = os.path.join(here, "_whisper_features.py")
    if not os.path.exists(mp):
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{MODEL_PATH}", mp)
    if not os.path.exists(fp):
        urllib.request.urlretrieve(FEATURES_URL, fp)
    return mp, fp


def speechlike(f0: float = 120.0, nsyll: int = 4, dur: float = 8.0) -> np.ndarray:
    """A crude speech-shaped signal: formant stack under a syllable envelope."""
    t = np.arange(int(SR * dur)) / SR
    x = np.zeros_like(t)
    for k in range(nsyll):
        st = int(k * (dur / max(nsyll, 1)) * SR)
        en = int((k + 0.6) * (dur / max(nsyll, 1)) * SR)
        if en > len(t):
            break
        env = np.sin(np.linspace(0, np.pi, en - st))
        seg = (np.sin(2 * np.pi * f0 * t[st:en]) * 0.5
               + np.sin(2 * np.pi * f0 * 3 * t[st:en]) * 0.3)
        x[st:en] += env * seg
    x = x / max(np.max(np.abs(x)), 1e-9)
    return (x * 0.4).astype(np.float32)


def main() -> int:
    mp, fp = fetch()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from _whisper_features import compute_whisper_log_mel_features

    size = os.path.getsize(mp)
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(mp, sess_options=so)

    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(N) * 0.05).astype(np.float32)

    feat_ms, infer_ms = [], []
    for _ in range(30):
        t0 = time.perf_counter()
        feat = compute_whisper_log_mel_features(noise, do_normalize=True)
        t1 = time.perf_counter()
        sess.run(None, {"input_features": np.expand_dims(feat, 0)})
        t2 = time.perf_counter()
        feat_ms.append((t1 - t0) * 1000)
        infer_ms.append((t2 - t1) * 1000)
    feat_ms.sort()
    infer_ms.sort()
    med = lambda a: a[len(a) // 2]
    p95 = lambda a: a[int(0.95 * len(a))]

    def score(x: np.ndarray) -> float:
        f = compute_whisper_log_mel_features(
            np.pad(x, (0, max(0, N - x.size)))[:N].astype(np.float32),
            do_normalize=True)
        return float(sess.run(None, {"input_features": np.expand_dims(f, 0)})[0].ravel()[0])

    tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(N) / SR)).astype(np.float32)
    behaviour = {
        "speechlike_4_syllables": round(score(speechlike(nsyll=4)), 4),
        "speechlike_1_syllable": round(score(speechlike(nsyll=1)), 4),
        "white_noise": round(score(noise), 4),
        "sine_220hz_tone": round(score(tone), 4),
        "digital_silence": round(score(np.zeros(N, dtype=np.float32)), 4),
    }

    out = {
        "model_bytes": size,
        "input_shape": [1, 80, 800],
        "output_node_name": sess.get_outputs()[0].name,
        "feature_ms": {"median": round(med(feat_ms), 2), "p95": round(p95(feat_ms), 2)},
        "infer_ms": {"median": round(med(infer_ms), 2), "p95": round(p95(infer_ms), 2)},
        "total_ms_median": round(med(feat_ms) + med(infer_ms), 2),
        "behaviour_prob_complete": behaviour,
        "note": "pipecat thresholds prediction at probability > 0.5 "
                "(local_smart_turn_v3.py:174)",
    }
    print(json.dumps(out, indent=2))
    with open(os.path.join(here, "smartturn_bench.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())