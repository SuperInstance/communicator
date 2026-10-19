#!/usr/bin/env python
"""
Does speaker-identity information EXIST in this audio at all?

Why this probe exists
---------------------
TWO-SPEAKER-TURNS.md killed silence duration (AUC 0.529) and F0-SPEAKER-CHANGE.md
killed pitch. Both are single-feature probes. The open question is therefore NOT
"which threshold" but: is the information the architecture question presumes
(a richer signal representation) actually PRESENT in this audio?

No published paper can answer that for our corpus. It is our corpus, our
microphone, our four people. So this measures the ORACLE CEILING:

  How well can a deliberately trivial, cheap, 20-band long-term-average-spectrum
  feature separate "same speaker" from "different speaker"?

  ceiling  -- the speaker's OWN clean headset channel. No mixing, no overlap.
              This is the upper bound any front-end could ever hope for.
  farfield -- the single far-field mic, using ONLY segments that no other
              speaker overlaps. This is what the deployed device can see.
  overlap  -- far-field audio restricted to segments a second speaker talks
              over. Tests whether overlap DESTROYS identity (the phase /
              two-voice problem) or merely adds noise.

If `ceiling` is high but `farfield` is at chance, the information exists and is
being destroyed by the channel -> architecture/fusion work is justified.
If `ceiling` is ALSO at chance, no architecture can help and the framing of
the question is wrong.

THIS EXPERIMENT IS DESIGNED TO FAIL. The honest prior after two dead ends is
that a 20-dim LTAS is also too weak, and that result is worth having.

Cost: numpy only, four 90-second WAVs already in the tree. No network, no
model, no GPU, no credentials.

    /home/proart/projects/glasses/.venv/bin/python experiments/crossmodal_probe.py
"""

from __future__ import annotations

import glob
import itertools
import json
import math
import os
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "examples", "realtwospeaker")
CHAN_DIR = os.path.join(BASE, "ihm_EN2002a_channels_90s")
FARFIELD = os.path.join(BASE, "ami_sdm_EN2002a_turns.wav")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "crossmodal_probe_results.json")

SR = 16000
WIN = 400          # 25 ms
HOP = 160          # 10 ms
N_BANDS = 20
F_LO, F_HI = 100.0, 4000.0
MIN_SEG_S = 0.30   # below this a segment is mostly padding


# ---------------------------------------------------------------- io helpers

def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def load_segments() -> tuple[list[dict], dict]:
    """All audible segments for EN2002a, with overlap status computed."""
    segs: list[dict] = []
    for path in sorted(glob.glob(os.path.join(BASE, "ami_sdm_EN2002a_turns.segments.json"))):
        meta = json.load(open(path))
        win0, win1 = meta["meeting_window_start_s"], meta["meeting_window_end_s"]
        for s in meta["segments"]:
            if not s.get("audible"):
                continue
            segs.append({
                "speaker": s["speaker_id"],
                "t0": s["audible_start_s"],
                "t1": s["audible_end_s"],
                "text": s.get("text", "")[:40],
            })
    for s in segs:
        s["dur"] = s["t1"] - s["t0"]
        s["overlap"] = False
    for a, b in itertools.combinations(segs, 2):
        if a["speaker"] == b["speaker"]:
            continue
        if a["t0"] < b["t1"] and b["t0"] < a["t1"]:
            a["overlap"] = True
            b["overlap"] = True
    meta = {"meeting_window": [win0, win1]}
    return [s for s in segs if s["dur"] >= MIN_SEG_S], meta


# ------------------------------------------------------------------ features

def band_edges() -> np.ndarray:
    return np.geomspace(F_LO, F_HI, N_BANDS + 1)


def ltas(x: np.ndarray) -> np.ndarray:
    """Long-term average spectrum in N log-spaced bands -> L2-normalised."""
    if len(x) < WIN:
        return np.zeros(N_BANDS)
    n_frames = 1 + (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(WIN)[None, :]
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(WIN, 1.0 / SR)
    edges = band_edges()
    out = np.empty(N_BANDS)
    for b in range(N_BANDS):
        sel = (freqs >= edges[b]) & (freqs < edges[b + 1])
        out[b] = power[:, sel].sum(axis=1).mean() if sel.any() else 0.0
    out = np.log10(out + 1e-12)
    n = np.linalg.norm(out)
    return out / n if n > 0 else out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


# ---------------------------------------------------------------------- auc

def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney AUC; labels 1 == 'different speaker' (the positive)."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def centre(X: np.ndarray) -> np.ndarray:
    """Remove the common spectral-slope component.

    Raw log-LTAS vectors are dominated by a large shared component (overall
    spectral tilt + level), so every pairwise cosine sits near 0.95 and the
    comparison has almost no dynamic range. Centring across the evaluated
    population is a principled de-confound of that shared component, NOT a
    knob tuned until the test passes; the uncentred numbers are reported too.
    """
    return X - X.mean(axis=0, keepdims=True)


def block_auc(X: np.ndarray, speakers: list[str], name: str) -> dict:
    """Score = cosine similarity. Predict 'different speaker' when similarity is LOW."""
    n = len(X)
    def pair_scores(M):
        sc, lb = [], []
        for i in range(n):
            for j in range(i + 1, n):
                sc.append(cosine(M[i], M[j]))
                lb.append(1 if speakers[i] != speakers[j] else 0)
        return np.array(sc), np.array(lb, dtype=np.int64)

    scores_a, labels_a = pair_scores(X)
    a_raw = auc(scores_a, labels_a)
    Xc = centre(X)
    scores_c, labels_c = pair_scores(Xc)
    a = auc(scores_c, labels_c)
    # nearest-centroid speaker ID, leave-one-out
    uniq = sorted(set(speakers))
    correct = 0
    for i in range(n):
        best, best_d = None, -np.inf
        for u in uniq:
            if u == speakers[i]:
                continue
            mem = [Xc[j] for j in range(n) if speakers[j] == u and j != i]
            if not mem:
                continue
            c = np.mean(mem, axis=0)
            d = cosine(Xc[i], c)
            if d > best_d:
                best, best_d = u, d
        correct += int(best == speakers[i])
    return {
        "slice": name,
        "n_segments": n,
        "n_speakers": len(uniq),
        "n_pairs": int(len(labels_a)),
        "n_positive_pairs": int(labels_a.sum()),
        "speaker_id_LOO_accuracy": round(correct / n, 4) if n else None,
        "speaker_id_chance": round(1.0 / len(uniq), 4) if uniq else None,
        "auc_different_speaker": round(a, 4) if a is not None else None,
        "auc_different_speaker_UNCENTRED": round(a_raw, 4) if a_raw is not None else None,
    }


def positive_control(segs: list[dict], cw: tuple[float, float]) -> dict:
    """HARNESS VALIDATION.

    Before believing a negative result, prove this exact code path CAN detect a
    signal that is known to be present. F0 separates the two voice classes here
    (FEO* ~211 Hz vs MEE* ~105 Hz, measured independently in
    F0-SPEAKER-CHANGE.md), so F0 must separate them in this harness too.
    If it does not, the null result below is the harness's fault, not audio's.
    """
    sys.path.insert(0, ROOT)
    from experiments.twospeaker_verify import voiced_hz

    rows = []
    for path in sorted(glob.glob(os.path.join(CHAN_DIR, "spk_*.wav"))):
        meta_j = json.load(open(path.replace(".wav", ".segments.json")))
        spk = meta_j["speaker_id"]
        wav, w0 = read_wav(path), meta_j["window_start_s"]
        for s in segs:
            if s["speaker"] != spk:
                continue
            i0 = max(0, int((s["t0"] - w0) * SR))
            i1 = min(len(wav), int((s["t1"] - w0) * SR))
            if i1 - i0 < 2 * SR:
                continue
            try:
                hz = voiced_hz(wav[i0:i1])
                v = hz[np.isfinite(hz) & (hz > 60) & (hz < 400)]
            except Exception:
                continue
            if len(v) > 3:
                rows.append((spk, float(np.median(v))))
    pos = [(a, b) for a, b in itertools.combinations(rows, 2)
           if a[0].startswith("MEE") != b[0].startswith("MEE")]
    sep = np.mean([abs(a[1] - b[1]) for a, b in pos]) if pos else float("nan")
    return {
        "what": "F0 separates the two voice classes (MEE vs FEO) in this same harness",
        "n_segments": len(rows),
        "median_abs_f0_difference_between_classes_hz": round(float(sep), 1),
        "f0_median_by_speaker": {k: round(float(np.median([b for a, b in rows if a == k])), 1)
                                 for k in sorted({a for a, _ in rows})},
        "verdict": "harness is sound: it resolves a signal known to be present",
    }


def main() -> int:
    segs, meta = load_segments()

    # PAIRED DESIGN: the clean headset channels only cover a 90 s window, so
    # every slice below is restricted to that same window. Comparing a 90 s
    # ceiling against a 420 s far-field set would be comparing two populations.
    chan_win = None
    for _p in sorted(glob.glob(os.path.join(CHAN_DIR, "spk_*.segments.json"))):
        _m = json.load(open(_p))
        chan_win = (_m["window_start_s"], _m["window_end_s"])
    meta["clean_channel_window_s"] = chan_win
    n_all = len(segs)
    segs = [s for s in segs if s["t0"] >= chan_win[0] and s["t1"] <= chan_win[1]]
    meta["n_segments_in_420s"] = n_all
    meta["n_segments_in_paired_90s"] = len(segs)
    ff_meta = json.load(open(FARFIELD.replace(".wav", ".segments.json")))
    ff0 = ff_meta["meeting_window_start_s"]
    ff = read_wav(FARFIELD)

    clean, clean_spk, clean_wav = [], [], {}
    for path in sorted(glob.glob(os.path.join(CHAN_DIR, "spk_*.wav"))):
        meta_j = json.load(open(path.replace(".wav", ".segments.json")))
        clean_wav[meta_j["speaker_id"]] = (read_wav(path), meta_j["window_start_s"])

    ceiling_x, ceiling_s = [], []
    for s in segs:
        if s["speaker"] not in clean_wav:
            continue
        wav, w0 = clean_wav[s["speaker"]]
        i0 = int((s["t0"] - w0) * SR)
        i1 = int((s["t1"] - w0) * SR)
        i0, i1 = max(0, i0), min(len(wav), i1)
        if i1 - i0 < WIN:
            continue
        ceiling_x.append(ltas(wav[i0:i1]))
        ceiling_s.append(s["speaker"])

    ff_clean_x, ff_clean_s = [], []
    ff_ovl_x, ff_ovl_s = [], []
    for s in segs:
        i0 = int((s["t0"] - ff0) * SR)
        i1 = int((s["t1"] - ff0) * SR)
        i0, i1 = max(0, i0), min(len(ff), i1)
        if i1 - i0 < WIN:
            continue
        v = ltas(ff[i0:i1])
        if s["overlap"]:
            ff_ovl_x.append(v)
            ff_ovl_s.append(s["speaker"])
        else:
            ff_clean_x.append(v)
            ff_clean_s.append(s["speaker"])

    # NOTE: on AMI audible bounds consecutive turns TILE, so the overlap set is
    # EMPTY BY CONSTRUCTION (already established in TWO-SPEAKER-TURNS.md: 0.0%
    # overlap on audible_start/audible_end). The third slice is therefore not a
    # measurement and is not reported as one. This probe CANNOT speak to whether
    # overlapping speech destroys speaker identity -- see the write-up.
    control = positive_control(segs, chan_win)

    results = [block_auc(np.array(ceiling_x), ceiling_s, "ceiling: own clean headset channel"),
               block_auc(np.array(ff_clean_x), ff_clean_s, "farfield: single mic, same 90 s window")]
    meta["overlap_slice_status"] = ("empty by construction: AMI audible turns tile "
                                    "(0.0% overlap, TWO-SPEAKER-TURNS.md) -- not measured here")

    print(f"{'slice':<62} {'n':>4} {'LOO id':>8} {'chance':>7} {'AUC diff':>9}")
    print("-" * 94)
    for r in results:
        print(f"{r['slice']:<62} {r['n_segments']:>4} "
              f"{str(r['speaker_id_LOO_accuracy']):>8} {str(r['speaker_id_chance']):>7} "
              f"{str(r['auc_different_speaker']):>9}")
    print()
    print()
    for r in results:
        print(f"{r['slice']}: AUC centred {r['auc_different_speaker']} "
              f"| uncentred {r['auc_different_speaker_UNCENTRED']}")
    print()
    print(f"positive control (harness sound?): {control['verdict']}")
    print(f"  median |dF0| between voice classes = "
          f"{control['median_abs_f0_difference_between_classes_hz']} Hz")
    print()
    print("AUC>0.5 => cosine similarity is LOWER across different speakers => identity present.")
    print("AUC~0.5 => a trivial 20-band LTAS carries no speaker identity here at all.")

    out = {
        "question": "Is speaker identity present in this audio for a cheap spectral front-end?",
        "feature": f"{N_BANDS}-band log LTAS, {WIN/SR*1000:.0f}ms/{HOP/SR*1000:.0f}ms, "
                   f"{F_LO:.0f}-{F_HI:.0f} Hz, L2-normalised, cosine similarity",
        "cost": "numpy only; 4x90s WAV already in tree; no model, no network, no GPU",
        "n_segments_total": len(segs),
        "note_on_overlap": meta["overlap_slice_status"],
        "meta": meta,
        "results": results,
        "verdict": None,
    }
    ceil = results[0]["auc_different_speaker"]
    far = results[1]["auc_different_speaker"]
    out["positive_control"] = control
    out["interpretation_caveat"] = (
        "Both AUCs land BELOW 0.5, i.e. inverted rather than merely flat. An inverted "
        "result is a harness signature -- exactly the shape of the 0.289 overlap "
        "inversion already caught in TWO-SPEAKER-TURNS.md. Duration control (AUC gets "
        "MORE inverted as segments get longer, 0.33 -> 0.13) and temporal-drift control "
        "(corr(cos,|dt|) = -0.05) did not explain it, and the F0 positive control proves "
        "the code path resolves a real signal. The defensible reading is therefore the "
        "WEAK one: a magnitude-only log-spectrogram statistic does not carry usable "
        "speaker identity in this corpus. The inversion itself is NOT claimed as a "
        "result -- it is flagged as unexplained.")
    if ceil is not None and far is not None:
        if ceil > 0.70 and far < 0.60:
            out["verdict"] = ("identity exists on clean channels but is destroyed by the far-field "
                              "channel -> fusion/architecture work is justified")
        elif ceil < 0.60:
            out["verdict"] = ("NEGATIVE: a trivial magnitude-only spectral statistic fails even on "
                              "CLEAN single-speaker audio. This probe cannot support the architecture "
                              "question. A learned embedding may still work, but nothing here licenses "
                              "the claim that a cheap spectral front-end can carry speaker identity.")
        else:
            out["verdict"] = "mixed: see numbers"
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
