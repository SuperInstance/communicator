#!/usr/bin/env python3
"""
nv12_stride_audit.py -- the captures were being read at the wrong stride, and
that is where most of this repo's video evidence went.

WHY THIS FILE EXISTS.

Every video experiment in this repo read a capture with a hard-coded frame
stride of `W*H*3//2` = 23,040 bytes for a 160x96 frame. That is the size of a
standard NV12 or I420 frame. The captures on disk are NOT standard frames.
They were written by `C:\\proart_capture\\nv12small.py`, which writes, per
frame:

    Y plane   15,360 bytes   (160 x 96)
    U,V        3,840 bytes   (80 x 24 samples, INTERLEAVED)
    -----------------------
    total     19,200 bytes   = W*H*5/4

The chroma is subsampled 2:1 across but 4:1 down, so the frame is 20%
shorter than a standard one. Reading it at 23,040 therefore splices two
different frames into one: rows 0..71 come from frame k and rows 72..95 from
frame k+1. Only every sixth frame survives intact.

This file measures the whole blast radius, and it measures it in a way that
cannot be argued with: the video's own length must agree with the audio's.

THE THREE LAYOUTS. There are three, and they must not be confused:

    standard_nv12   W*H*3//2 = 23,040   luma + interleaved UV
    y_uv_halfres    W*H*5//4 = 19,200   luma + interleaved UV  <- the captures
    luma_only       W*H     = 15,360   luma only   <- the new capture tool

WHAT IT MEASURES.

  1. For every capture: the file size, the frame count and frame rate each
     candidate stride implies, the mean consecutive |diff| each candidate
     produces, and whether the resulting duration matches the WAV's.

  2. The mouth-signal payoff: whether the video carries the speaking signal
     at all, with a CHEEK control, because "the video moves" is not a finding
     -- "the MOUTH moves more than the cheek does" is.

THE CONTROLS, WHICH ARE THE POINT.

  wrong_stride   the same statistic computed on the mis-strided read. If the
                 correct read and the wrong read give the same answer, the
                 measurement was never about the video.
  cheek_box      a region of the same face at the same rows, to the right of
                 the mouth. The mouth opens; the cheek does not. A signal that
                 is the same in both is a signal about the person or the room,
                 not about speech.
  empty_room     `empty1.nv12`: nobody in it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import nv12  # noqa: E402

CAPTURES = Path("/mnt/c/proart_capture")
W, H = 160, 96

# The mouth, read off a 5x-upscaled frame with a 10-pixel coordinate grid
# (docs/research/NV12-STRIDE.md records the grid render). The cheek control
# is the same rows, to the right of the mouth.
MOUTH_BOX = (46, 58, 62, 86)
CHEEK_BOX = (46, 58, 92, 116)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def region_motion(luma: np.ndarray, box: tuple) -> np.ndarray:
    """Per-frame mean absolute change inside a region."""
    r0, r1, c0, c1 = box
    d = np.abs(np.diff(luma[:, r0:r1, c0:c1].astype(np.int32), axis=0))
    return d.mean(axis=(1, 2))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random positive sample exceeds a random negative one)."""
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    n, m = len(pos), len(neg)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def effect_size(a: np.ndarray, b: np.ndarray) -> float:
    sd = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--captures", default=str(CAPTURES))
    ap.add_argument("--out", default=str(ROOT / "experiments" /
                                         "nv12_stride_audit_results.json"))
    args = ap.parse_args(argv)
    caps = Path(args.captures)

    results: dict = {
        "question": ("the captures were read at 23,040 bytes per frame; the "
                     "files are 19,200. What did that cost?"),
        "frame_width": W, "frame_height": H,
        "layouts": {name: nv12.stride_for(name, W, H)
                    for name in nv12.LAYOUT_NAMES},
        "captures": {},
        "payoff": {},
    }

    # --- 1. the layout, per capture ------------------------------------
    for name in ("A", "B", "C", "empty1"):
        nv = caps / f"{name}.nv12"
        wv = caps / f"{name}.wav"
        if not nv.is_file():
            continue
        data = np.frombuffer(nv.read_bytes(), dtype=np.uint8)
        seconds = wav_seconds(wv) if wv.is_file() else None
        entry = {
            "path": str(nv),
            "size_bytes": len(data),
            "sha256": sha256_file(nv),
            "wav_seconds": None if seconds is None else round(seconds, 3),
            "candidates": {},
        }
        for layout in nv12.LAYOUT_NAMES:
            stride = nv12.stride_for(layout, W, H)
            if len(data) % stride:
                entry["candidates"][layout] = {"divides_evenly": False}
                continue
            n = len(data) // stride
            luma = nv12.luma_view(data, layout, W, H)
            diff = nv12.mean_consecutive_diff(luma)
            fps30 = n / 30.0
            entry["candidates"][layout] = {
                "divides_evenly": True,
                "stride_bytes": stride,
                "n_frames": n,
                "mean_consecutive_frame_diff": round(diff, 4),
                "duration_at_30fps_s": round(fps30, 3),
                "matches_audio_duration": (None if seconds is None else
                                            bool(abs(fps30 - seconds) < 0.25)),
            }
        found = nv12.detect_layout(data, W, H)
        entry["detected"] = found.as_dict()
        results["captures"][name] = entry

    # --- 2. the payoff: is the speaking signal in the mouth? -----------
    def load(name: str, layout: str = "y_uv_halfres"):
        luma, _ = nv12.read_luma(caps / f"{name}.nv12", W, H, layout=layout)
        return luma

    if all((caps / f"{n}.nv12").is_file() for n in ("A", "B", "C")):
        correct = {n: load(n) for n in ("A", "B", "C")}
        wrong = {n: load(n, "standard_nv12") for n in ("A", "B", "C")}
        payoff = {"regions": {"mouth": list(MOUTH_BOX),
                              "cheek": list(CHEEK_BOX)}}
        for label, frames in (("correct_stride", correct),
                              ("wrong_stride", wrong)):
            motion = {n: {"mouth": region_motion(frames[n], MOUTH_BOX),
                          "cheek": region_motion(frames[n], CHEEK_BOX)}
                      for n in frames}
            a, c = motion["A"]["mouth"], motion["C"]["mouth"]
            cheek_a, cheek_c = motion["A"]["cheek"], motion["C"]["cheek"]
            payoff[label] = {
                "mean_consecutive_frame_diff": {
                    n: round(nv12.mean_consecutive_diff(frames[n]), 4)
                    for n in frames},
                "mouth_A_minus_C": round(float(a.mean() - c.mean()), 4),
                "mouth_effect_size_sd": round(effect_size(a, c), 4),
                "mouth_auc_A_gt_C": round(auc(a, c), 4),
                "cheek_A_minus_C": round(float(cheek_a.mean() - cheek_c.mean()), 4),
                "cheek_effect_size_sd": round(effect_size(cheek_a, cheek_c), 4),
                "cheek_auc_A_gt_C": round(auc(cheek_c * 0 + cheek_a, cheek_c), 4),
            }
        # The permutation null for the mouth effect size.
        rng = np.random.default_rng(7)
        a = region_motion(correct["A"], MOUTH_BOX)
        c = region_motion(correct["C"], MOUTH_BOX)
        nulls = []
        for _ in range(400):
            idx = rng.permutation(len(a))
            half = len(a) // 2
            nulls.append(effect_size(a[idx[:half]], a[idx[half:]]))
        nulls = np.array(nulls)
        obs = effect_size(a, c)
        payoff["permutation_null"] = {
            "statistic": "effect size (A vs C) of the mouth region's motion",
            "observed": round(obs, 4),
            "null_mean": round(float(nulls.mean()), 4),
            "null_sd": round(float(nulls.std()), 4),
            "z": round(float((obs - nulls.mean()) / max(nulls.std(), 1e-12)), 2),
        }
        # How much of the effect is MOUTH-specific rather than head motion?
        for label, frames in (("correct_stride", correct),
                              ("wrong_stride", wrong)):
            if label not in payoff:
                continue
            m_a = region_motion(frames["A"], MOUTH_BOX)
            m_c = region_motion(frames["C"], MOUTH_BOX)
            k_a = region_motion(frames["A"], CHEEK_BOX)
            k_c = region_motion(frames["C"], CHEEK_BOX)
            mouth_e = effect_size(m_a, m_c)
            cheek_e = effect_size(k_a, k_c)
            payoff[label]["mouth_specific_excess_sd"] = round(mouth_e - cheek_e, 4)
            payoff[label]["mouth_over_cheek_ratio"] = (
                round(mouth_e / cheek_e, 3) if cheek_e > 1e-9 else None)
        # The empty room: nobody in it, so nothing should move.
        empty_nv = caps / "empty1.nv12"
        if empty_nv.is_file():
            empty, _ = nv12.read_luma(empty_nv, W, H, layout="y_uv_halfres")
            e_mouth = region_motion(empty, MOUTH_BOX)
            a_mouth = region_motion(correct["A"], MOUTH_BOX)
            c_mouth = region_motion(correct["C"], MOUTH_BOX)
            payoff["empty_room"] = {
                "mouth_motion": round(float(e_mouth.mean()), 6),
                "A_mouth_motion": round(float(a_mouth.mean()), 6),
                "C_mouth_motion": round(float(c_mouth.mean()), 6),
                "empty_over_A": round(float(e_mouth.mean() / a_mouth.mean()), 6),
                "empty_over_C": round(float(e_mouth.mean() / c_mouth.mean()), 6),
            }
        results["payoff"] = payoff

    results["provenance"] = {
        "script": "experiments/nv12_stride_audit.py",
        "script_sha256": sha256_file(Path(__file__)),
        "captures_dir": str(caps),
        "numpy": np.__version__,
    }
    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2) + "\n")

    for name, entry in results["captures"].items():
        det = entry["detected"]
        print(f"{name}: {entry['size_bytes']:,} bytes -> {det['layout']} "
              f"(stride {det['stride_bytes']}, {det['n_frames']} frames, "
              f"mean|d| {det['mean_consecutive_frame_diff']}, margin "
              f"{det['margin_over_runner_up']})")
        for cand, c in entry["candidates"].items():
            if not c.get("divides_evenly"):
                continue
            print(f"    {cand:16s} {c['n_frames']:5d} frames  "
                  f"{c['duration_at_30fps_s']:6.2f}s  "
                  f"mean|d| {c['mean_consecutive_frame_diff']:9.3f}  "
                  f"audio match: {c['matches_audio_duration']}")
    if results["payoff"]:
        p = results["payoff"]
        for label in ("correct_stride", "wrong_stride"):
            if label in p:
                v = p[label]
                print(f"{label}: mouth A-C {v['mouth_A_minus_C']:+.4f} "
                      f"(effect {v['mouth_effect_size_sd']:+.3f} sd)  "
                      f"cheek A-C {v['cheek_A_minus_C']:+.4f} "
                      f"(effect {v['cheek_effect_size_sd']:+.3f} sd)")
        if "permutation_null" in p:
            n = p["permutation_null"]
            print(f"mouth effect size {n['observed']:+.4f} vs a null of "
                  f"{n['null_mean']:+.4f} +/- {n['null_sd']:.4f}  "
                  f"z = {n['z']:+.1f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
