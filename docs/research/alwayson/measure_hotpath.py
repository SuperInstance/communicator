#!/usr/bin/env python
"""
Hot-path measurement harness for the always-on research lane.

Measures the three things the SYNTHESIS has to decide between, on THIS
machine, rather than on a README:

  1. per-inference latency, single-threaded, cold and warm
  2. the model stack's actual on-disk size
  3. -- the one that decides the recommendation -- what silero-vad scores
     on the exact signals our own watcher is KNOWN to get wrong: steady
     periodic hum, keyboard clicks, music.

Run with a python that has onnxruntime:

    /tmp/ortenv/bin/python docs/research/alwayson/measure_hotpath.py

It deliberately does NOT import the system watcher by default, because the
system venv has numpy only. Pass --watcher to run the comparison half, which
needs a python with BOTH onnxruntime and the repo on sys.path.

Everything here is numpy + onnxruntime. No torch anywhere.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import wave

import numpy as np
import onnxruntime as ort

MODELS = os.environ.get("ALWAYSON_MODELS", "/tmp/models")
AUDIO = os.environ.get("ALWAYSON_AUDIO", "/tmp/audio")
SR = 16000
RESULTS = os.environ.get(
    "ALWAYSON_OUT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "hotpath_measurements.json"))

CPU_EP = ort.SessionOptions()
CPU_EP.inter_op_num_threads = 1
CPU_EP.intra_op_num_threads = 1


def sess(name):
    return ort.InferenceSession(os.path.join(MODELS, name),
                                sess_options=CPU_EP,
                                providers=["CPUExecutionProvider"])


# -- test signals ------------------------------------------------------------

def _norm16(x):
    x = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak
    return (x * 12000).astype(np.int16)


def signals():
    """
    The battery. Each is 4 s at 16 kHz, normalised to the same peak so the
    comparison is about SIGNAL SHAPE and not about loudness -- the whole
    point is that our watcher fails on a signal that is loud and periodic.
    """
    n = 4 * SR
    t = np.arange(n) / SR
    out = {}

    # A: steady periodic hum -- the declared failure of src/watcher.py.
    # 60 Hz + harmonics, constant amplitude, no gaps, no formants.
    hum = (0.50 * np.sin(2 * np.pi * 60 * t)
           + 0.25 * np.sin(2 * np.pi * 120 * t)
           + 0.15 * np.sin(2 * np.pi * 180 * t))
    out["hum_60hz"] = _norm16(hum)

    # B: fan / HVAC -- broadband noise with a periodic amplitude modulation.
    rng = np.random.default_rng(7)
    fan = rng.normal(0, 1, n) * (0.6 + 0.4 * np.sin(2 * np.pi * 3.5 * t))
    out["fan_noise"] = _norm16(fan)

    # C: keyboard clicks -- short broadband transients on a ~90 ms cadence.
    clicks = rng.normal(0, 0.02, n)
    for i in range(0, n - 400, int(0.09 * SR)):
        k = np.arange(400) / SR
        clicks[i:i + 400] += rng.normal(0, 1, 400) * np.exp(-k * 60)
    out["keyboard_clicks"] = _norm16(clicks)

    # D: music-ish -- a sustained triad with vibrato and slow swell.
    music = sum(np.sin(2 * np.pi * f * t + 1.5 * np.sin(2 * np.pi * 5 * t))
                for f in (220.0, 277.2, 329.6))
    music *= (0.7 + 0.3 * np.sin(2 * np.pi * 0.25 * t))
    out["music_triad"] = _norm16(music)

    # E: flat white noise at a constant level (the digital-silence control).
    out["white_noise"] = _norm16(rng.normal(0, 1, n))

    # F: near-silence.
    out["silence"] = _norm16(rng.normal(0, 0.001, n))

    # G: real speech, looped to 4 s. These are the repo's own test fixtures:
    # alexa_test.wav is a 0.62 s real utterance saying "alexa".
    for name, fn in (("speech_alexa", "alexa_test.wav"),
                     ("speech_mycroft", "hey_mycroft_test.wav"),
                     ("speech_jane", "hey_jane.wav")):
        with wave.open(os.path.join(AUDIO, fn)) as w:
            d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        reps = int(np.ceil(n / len(d)))
        out[name] = _norm16(np.tile(d, reps)[:n])

    return out


def silero_versions():
    """
    Two DIFFERENT silero models get compared, because they are not the same
    model and conflating them would be a measurement error:

      silero_vad.onnx        the CURRENT v5 in silero-vad master (state is
                             (2,1,128); opset 16; 2.220 MB)
      cur_silero_vad_op15    the 16 kHz-only opset-15 variant (1.230 MB)
      silero_vad.onnx        the OLDER copy VENDORED INSIDE openWakeWord's
                             release (state is (2,1,64); 1.724 MB)

    openWakeWord ships its own pinned copy (openwakeword/__init__.py:19-24,
    VAD_MODELS -> silero_vad.onnx), so if we take openWakeWord's VAD we take
    THAT copy, not current master. The comparison below is the reason to take
    current master instead.
    """
    rows = {}
    for label, fn in (("oww_vendored_v4_ish", "silero_vad.onnx"),
                      ("current_v5_opset16", "cur_silero_vad.onnx"),
                      ("current_v5_opset15_16k", "cur_silero_vad_16k_op15.onnx")):
        p = os.path.join(MODELS, fn)
        if not os.path.exists(p):
            continue
        s = sess(fn)
        meta = {i.name: (list(i.shape), i.type) for i in s.get_inputs()}
        rows[label] = {
            "file": fn,
            "bytes": os.path.getsize(p),
            "inputs": meta,
            "has_sr_input": "sr" in meta,
        }
    return rows


def silero_v5_scores(sigs):
    """
    Re-run the failure-mode battery against the CURRENT model, using the
    exact call convention silero's own OnnxWrapper uses (utils_vad.py:109-137):
    prepend a 64-sample context, feed exactly 512 samples, carry `state`.
    """
    rows = {}
    for label, fn in (("current_v5_opset16", "cur_silero_vad.onnx"),
                      ("current_v5_opset15_16k", "cur_silero_vad_16k_op15.onnx")):
        p = os.path.join(MODELS, fn)
        if not os.path.exists(p):
            continue
        m = sess(fn)
        sr = np.array(SR, dtype=np.int64)
        per = {}
        for name, x in sigs.items():
            xf = x.astype(np.float32) / 32768.0
            state = np.zeros((2, 1, 128), dtype=np.float32)
            ctx = np.zeros((1, 64), dtype=np.float32)
            scores = []
            for i in range(0, len(xf) - 512 + 1, 512):
                chunk = xf[i:i + 512][None, :]
                # utils_vad.py:130-137 -- feed context(64) + chunk(512) = 576,
                # then the NEXT context is the last 64 samples of what we fed.
                feed = np.concatenate([ctx, chunk], axis=1)
                ctx = feed[:, -64:].copy()
                out, state = m.run(None, {"input": feed, "state": state,
                                          "sr": sr})[:2]
                scores.append(float(out[0][0]))
            sc = np.array(scores)
            per[name] = {"mean": round(float(sc.mean()), 4),
                         "max": round(float(sc.max()), 4),
                         "p95": round(float(np.percentile(sc, 95)), 4),
                         "frac_over_0.5": round(float((sc > 0.5).mean()), 4)}
        rows[label] = per
    return rows


def onnx_param_count(path):
    """
    Count real parameters straight out of the ONNX artifact, so the "~1.8M
    params" claim can be checked rather than believed. Counts INITIALIZERS
    RECURSIVELY, which matters here: the current silero_vad.onnx has ZERO
    top-level initializers and puts its whole body inside `If` subgraphs and
    local functions, so a naive graph.initializer count returns 0.
    Needs `onnx` (not onnxruntime).
    """
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path, load_external_data=False)

    seen, total, fp32, n_init = {}, 0, 0, 0

    def add_inits(graph):
        nonlocal total, fp32, n_init
        for init in graph.initializer:
            n = 1
            for d in init.dims:
                n *= int(d)
            total += n
            n_init += 1
            if init.data_type == onnx.TensorProto.FLOAT:
                fp32 += n
        for node in graph.node:
            for a in list(node.attribute):
                for g in a.graphs:
                    add_inits(g)
                if a.type == onnx.AttributeProto.GRAPH:
                    add_inits(a.g)

    add_inits(m.graph)
    for fn in m.functions:
        add_inits(fn.node) if hasattr(fn, "node") and hasattr(
            fn.node, "attribute") else None
        for node in getattr(fn, "node", []):
            for a in node.attribute:
                for g in a.graphs:
                    add_inits(g)

    ops = set()

    def collect(graph):
        for node in graph.node:
            ops.add(node.op_type)
            for a in node.attribute:
                for g in a.graphs:
                    collect(g)

    collect(m.graph)
    return {"total_params": total, "float32_params": fp32,
            "n_initializers": n_init,
            "opset": m.opset_import[0].version if m.opset_import else None,
            "n_functions": len(m.functions),
            "ops_used": sorted(ops)}


# -- 1. silero-vad behaviour and latency -------------------------------------

def silero_scores(s):
    """
    Run silero exactly the way the package runs it: a STATELESS-per-call but
    state-carried LSTM over fixed chunks, accumulating h/c. That state is why
    a single-call score on a synthetic signal is meaningless -- we report the
    distribution over the whole clip instead.
    """
    m = sess("silero_vad.onnx")
    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    sr = np.array(SR, dtype=np.int64)

    rows = {}
    for chunk in (480, 512):          # 30 ms (v5) and 32 ms (v4)
        per = {}
        for name, x in s.items():
            xf = (x.astype(np.float32) / 32768.0)
            hh, cc = h.copy(), c.copy()
            scores = []
            for i in range(0, len(xf) - chunk + 1, chunk):
                seg = xf[i:i + chunk][None, :]
                out, hh, cc = m.run(None, {"input": seg, "sr": sr,
                                           "h": hh, "c": cc})
                scores.append(float(out[0][0]))
            sc = np.array(scores)
            per[name] = {"mean": round(float(sc.mean()), 4),
                         "max": round(float(sc.max()), 4),
                         "p90": round(float(np.percentile(sc, 90)), 4),
                         "frac_over_0.5": round(float((sc > 0.5).mean()), 4)}
        rows[f"chunk_{chunk}"] = per
    return rows


def silero_latency():
    """Per-30 ms-chunk cost, warm. This is the number the README claims."""
    m = sess("silero_vad.onnx")
    sr = np.array(SR, dtype=np.int64)

    def timeit(chunk, n=400, warmup=30):
        h = np.zeros((2, 1, 64), dtype=np.float32)
        c = np.zeros((2, 1, 64), dtype=np.float32)
        seg = np.zeros((1, chunk), dtype=np.float32)
        for _ in range(warmup):
            out, h, c = m.run(None, {"input": seg, "sr": sr, "h": h, "c": c})
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            out, h, c = m.run(None, {"input": seg, "sr": sr, "h": h, "c": c})
            ts.append(time.perf_counter() - t0)
        return {"chunk_samples": chunk, "chunk_ms": chunk / 16.0,
                "mean_ms": round(statistics.mean(ts) * 1000, 5),
                "median_ms": round(statistics.median(ts) * 1000, 5),
                "p95_ms": round(float(np.percentile(ts, 95)) * 1000, 5),
                "realtime_ratio": round((chunk / 16000) /
                                        statistics.median(ts), 1)}

    # Cold = a brand new session, which is what actually happens at wake-up.
    t0 = time.perf_counter()
    mcold = sess("silero_vad.onnx")
    cold_load_ms = (time.perf_counter() - t0) * 1000

    rows = {"cold_session_load_ms": round(cold_load_ms, 2)}
    for chunk in (480, 512, 1536, 15360):
        rows[str(chunk)] = timeit(chunk)
    rows["batch_30x_512"] = timeit(512, n=40)
    return rows


# -- 2. openWakeWord stack latency -------------------------------------------

def oww_latency():
    """
    The whole openWakeWord inference path on one 80 ms frame (1280 samples):
    melspectrogram -> embedding -> N wake-word heads. The embedding is SHARED,
    which is the architectural point that makes N models cheap.
    """
    ms = sess("melspectrogram.onnx")
    em = sess("embedding_model.onnx")
    # Only the openWakeWord classification heads. The directory also holds the
    # silero_vad copies fetched for this report, which are not heads and do not
    # have the (1, n_frames, 96) input.
    NOT_HEADS = {"silero_vad.onnx", "melspectrogram.onnx", "embedding_model.onnx"}
    heads = {n: sess(n) for n in sorted(os.listdir(MODELS))
             if n.endswith(".onnx") and n not in NOT_HEADS
             and not n.startswith("cur_")}
    sr_ = 1280

    def timeit(fn, n=60, warmup=10):
        for _ in range(warmup):
            fn()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return round(statistics.median(ts) * 1000, 4)

    # Shapes are reproduced from openwakeword/utils.py, not guessed:
    #   _get_melspectrogram  utils.py:202-208  squeeze(outputs[0]) -> (time, 32)
    #   _get_embeddings      utils.py:229-235  window 76 x 32, expand axis -1
    #   embedding_model_predict                utils.py:93   squeeze -> (n, 96)
    #   get_features          utils.py:454-460  last 16 rows -> (1, 16, 96)
    frame = np.zeros((1, sr_), dtype=np.float32)
    spec_one = np.squeeze(ms.run(None, {"input": frame})[0])
    # A single 80 ms frame yields only a handful of mel frames; openWakeWord
    # keeps a 10 s rolling buffer (utils.py:165-166: melspectrogram_max_len =
    # 10*97) and only runs the embedding every step_size=8 mel frames
    # (utils.py:229-234), i.e. roughly once per 80 ms of audio. So the steady
    # state per 80 ms of audio is 1 melspec + 1 embedding + N heads, which is
    # what the timings below measure. The embedding's INPUT is a fixed 76x32
    # window regardless, so it is built directly.
    mel76 = (np.random.default_rng(0)
             .normal(size=(1, 76, 32, 1)).astype(np.float32))
    emb = em.run(None, {"input_1": mel76})[0].squeeze()

    # Heads do NOT share one input shape: model_inputs[mdl] is read per model
    # (model.py:156) so a longer phrase reads more feature frames. timer wants
    # 34 frames, weather 22, the short wake words 16. Input NAME also varies
    # ('onnx::Flatten_0' vs 'x.1'), which is why this is built from the
    # session metadata instead of hard-coded.
    head_inputs = {}
    for name, s_ in heads.items():
        meta = s_.get_inputs()[0]
        head_inputs[name] = {
            meta.name: np.tile(emb, (int(meta.shape[1]), 1))
                       [None, :].astype(np.float32)}

    one_name = sorted(head_inputs)[0]
    one = heads[one_name]

    def full(k):
        def go():
            ms.run(None, {"input": frame})
            e = em.run(None, {"input_1": mel76})[0].squeeze()
            for _ in range(k):
                for name, sess_ in heads.items():
                    meta = sess_.get_inputs()[0]
                    hi = np.tile(e, (int(meta.shape[1]), 1))[None, :]\
                        .astype(np.float32)
                    sess_.run(None, {meta.name: hi})
        return go

    return {
        "n_heads": len(heads),
        "heads": sorted(heads),
        "melspec_ms": timeit(lambda: ms.run(None, {"input": frame})),
        "embedding_ms": timeit(lambda: em.run(None, {"input_1": mel76})),
        "shared_front_ms": timeit(lambda: (ms.run(None, {"input": frame}),
                                           em.run(None, {"input_1": mel76}))),
        "one_head_ms": timeit(lambda: one.run(None, head_inputs[one_name])),
        "stack_1_head_ms": timeit(full(1)),
        "stack_all_heads_ms": timeit(full(len(heads))),
        "budget_80ms_ms": 80.0,
    }


# -- 3. our own watcher, on the same signals ---------------------------------

def watcher_comparison(sigs):
    """
    Only runs when the repo is importable AND onnxruntime is present, because
    the same script does both halves. It answers the question the whole lane
    turns on: does OUR watcher call these non-speech signals speech?
    """
    sys.path.insert(0, "/home/proart/projects/communicator")
    from src import watcher as W  # noqa: E402

    rows = {}
    for name, x in sigs.items():
        w = W.Watcher(sample_rate=SR)
        evs, t0 = [], time.perf_counter()
        chunk = 1600                        # 100 ms pushes, as documented
        for i in range(0, len(x), chunk):
            evs.extend(w.push(x[i:i + chunk].tolist()))
        push_s = time.perf_counter() - t0
        utts = [e for e in evs if e.kind == W.UTTERANCE]
        onsets = [e for e in evs if e.kind == W.ONSET]
        pauses = [e for e in evs if e.kind == W.PAUSE]
        rows[name] = {
            "n_events": len(evs), "n_onset": len(onsets),
            "n_utterance": len(utts), "n_pause": len(pauses),
            "onsets_per_min": round(len(onsets) / (len(x) / SR) * 60, 2),
            "speech_seconds": round(sum(e.duration_s for e in utts), 3),
            "voiced_ratio": [e.data.get("voiced_ratio") for e in utts][:6],
            "open": [e.data.get("open") for e in utts][:6],
            "push_total_ms": round(push_s * 1000, 2),
            "push_us_per_100ms": round(push_s * 1e6 / (len(x) / SR * 10), 2),
            "realtime_ratio": round((len(x) / SR) / push_s, 1),
        }
    return rows


# -- 4. sizes ---------------------------------------------------------------

def sizes():
    rows = {}
    for n in sorted(os.listdir(MODELS)):
        p = os.path.join(MODELS, n)
        if os.path.isfile(p):
            rows[n] = os.path.getsize(p)
    return rows


def main():
    sigs = signals()
    out = {
        "host": {"cpu": _cpu_name(), "cores": os.cpu_count(),
                 "onnxruntime": ort.__version__,
                 "threads": 1, "numpy": np.__version__},
        "sizes_bytes": sizes(),
        "silero_latency": silero_latency(),
        "silero_scores": silero_scores(sigs),
        "silero_versions": silero_versions(),
        "silero_v5_scores": silero_v5_scores(sigs),
        "silero_v5_latency": _silero_v5_latency(),
        "onnx_param_counts": {n: onnx_param_count(os.path.join(MODELS, n))
                              for n in sorted(os.listdir(MODELS))
                              if n.endswith(".onnx")},
        "openwakeword_latency": oww_latency(),
        "signals_seconds": {k: len(v) / SR for k, v in sigs.items()},
    }
    if "--watcher" in sys.argv:
        out["watcher"] = watcher_comparison(sigs)

    with open(RESULTS, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"host: {out['host']['cpu']} x{out['host']['cores']}, "
          f"onnxruntime {ort.__version__}, 1 thread\n")
    print("== sizes ==")
    for k, v in out["sizes_bytes"].items():
        print(f"   {v:>12,d} B  {v/1048576:7.3f} MB  {k}")
    print("\n== silero-vad latency (1 thread) ==")
    for k, r in out["silero_latency"].items():
        if isinstance(r, dict) and "median_ms" in r:
            print(f"   chunk {r['chunk_samples']:>5d} ({r['chunk_ms']:.0f} ms audio)"
                  f"  median {r['median_ms']:.4f} ms  "
                  f"realtime x{r['realtime_ratio']}")
    print("\n== silero-vad scores by signal ==")
    print(f"   {'signal':18s} {'ch512 mean':>11s} {'max':>7s} {'>0.5':>7s}"
          f" | {'ch480 mean':>11s} {'max':>7s}")
    for name in sigs:
        a = out["silero_scores"]["chunk_512"][name]
        b = out["silero_scores"]["chunk_480"][name]
        print(f"   {name:18s} {a['mean']:>11.4f} {a['max']:>7.4f} "
              f"{a['frac_over_0.5']:>7.3f} | {b['mean']:>11.4f} {b['max']:>7.4f}")
    o = out["openwakeword_latency"]
    if "silero_v5_scores" in out and out["silero_v5_scores"]:
        print("\n== silero CURRENT v5 on the same signals ==")
        for label, per in out["silero_v5_scores"].items():
            print(f"   -- {label}")
            for name in sigs:
                r = per[name]
                print(f"      {name:18s} mean={r['mean']:>8.4f} max={r['max']:>7.4f}"
                      f" >0.5={r['frac_over_0.5']:>6.3f}")
    print("\n== parameter counts (from ONNX artifacts) ==")
    for k, v in out["onnx_param_counts"].items():
        print(f"   {k:34s} {v['total_params']:>9,d} params  opset {v['opset']}")
    print("\n== openWakeWord per 80 ms frame ==")
    print(f"   melspec {o['melspec_ms']} ms  embedding {o['embedding_ms']} ms  "
          f"shared front {o['shared_front_ms']} ms")
    print(f"   one head {o['one_head_ms']} ms -> stack of 1 "
          f"{o['stack_1_head_ms']} ms; of {o['n_heads']} "
          f"{o['stack_all_heads_ms']} ms  (budget 80 ms)")
    if "watcher" in out:
        print("\n== our watcher on the same signals ==")
        for name, r in out["watcher"].items():
            print(f"   {name:18s} onsets={r['n_onset']:>3d} "
                  f"({r['onsets_per_min']:>7.2f}/min)  "
                  f"speech_s={r['speech_seconds']:>6.3f}  "
                  f"rt=x{r['realtime_ratio']}  push={r['push_us_per_100ms']}us")
    print(f"\nwrote {RESULTS}")
    return 0


def _silero_v5_latency():
    """Latency of the CURRENT model, same conventions as their wiki table."""
    out = {}
    for fn in ("cur_silero_vad.onnx", "cur_silero_vad_16k_op15.onnx"):
        p = os.path.join(MODELS, fn)
        if not os.path.exists(p):
            continue
        m = sess(fn)
        sr = np.array(SR, dtype=np.int64)
        ctx = np.zeros((1, 64), dtype=np.float32)
        res = {}
        # v5 caps the chunk at 512 samples (32 ms) -- load_silero_vad's
        # max_frames=512 default, model.py:7 -- so a 1536-sample call is not a
        # faster path, it is an invalid input. Only 512 is timed.
        for chunk in (512,):
            st = np.zeros((2, 1, 128), dtype=np.float32)
            c0 = ctx

            def call():
                # 64 context + chunk = the 576-sample input the model expects
                # (utils_vad.py:130). Timing the whole call, context included,
                # is what a streaming caller actually pays.
                nonlocal st
                feed = np.concatenate(
                    [c0, np.zeros((1, chunk), dtype=np.float32)], axis=1)
                out_, stn = m.run(None, {"input": feed, "state": st,
                                         "sr": sr})[:2]
                st = stn
            for _ in range(30):
                call()
            ts = []
            for _ in range(300):
                t0 = time.perf_counter()
                call()
                ts.append(time.perf_counter() - t0)
            med = statistics.median(ts)
            res[str(chunk)] = {"chunk_samples": chunk,
                              "chunk_ms": round(chunk / 16.0, 2),
                              "median_ms": round(med * 1000, 5),
                              "realtime_ratio": round((chunk / 16000) / med, 1)}
        out[fn] = res
    return out


def _cpu_name():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())