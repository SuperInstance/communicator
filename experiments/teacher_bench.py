"""
teacher_bench.py -- benchmark a candidate TEACHER against ground truth.

WHY THIS FILE IS THE GATE, NOT A DETAIL.

Casey's plan: record him, use a model to analyse the recordings, and let that
model be the teacher for the on-device system. The plan is sound and the
corpus script (`corpus_capture_win.py`) exists to feed it. But a teacher that
is wrong teaches the wrong thing, and the failure is silent: a student trained
on bad labels is worse than no student, and it is worse in a way that is
invisible until the deployed system makes a confident wrong call.

SO THE TEACHER IS BENCHMARKED FIRST, ON LABELLED CONDITIONS, AND IT MUST BEAT
CHANCE BY A MARGIN.

The labelled conditions are the ones Casey performs deliberately: A (facing,
speaking), B (turned, speaking), C (facing, silent), D (turned, silent). Their
labels are ground truth by construction. The FREE sessions carry no ground
truth and exist only to be labelled BY the teacher -- so the free corpus is
worthless until the teacher has passed this benchmark. The order is not
negotiable: teacher first, then free corpus.

THE TWO CONTROLS, AND WHY THE PROJECT HAS ALREADY BEEN BURNED BY THEIR ABSENCE.

  BLANK. The same prompt with a featureless input. A model that answers from
    the prompt rather than the data will score well here.
  SHUFFLED. The real input with its FRAMES PERMUTED -- same characters, same
    length, same every-pixel count, destroyed temporal order. A model that is
    reading a spatial summary rather than motion will score the same as on the
    real input, and that difference is invisible without the control.

This is not hypothetical. zcode's spectrogram experiment found a model that
answered a constant "different" to EVERY spectrogram -- and to a blank
rectangle. Without a blank control that model reads as a working classifier.

THE REPORTING RULE, WHICH IS THE POINT OF THE WHOLE FILE.

A teacher's accuracy is reported TOGETHER WITH its control accuracies, and the
headline number is not the accuracy. It is the margin over the controls:

    usable_margin = accuracy(real) - max(accuracy(blank), accuracy(shuffled))

A model with 80% accuracy and 78% on the blank has a 2% margin and knows
nothing. A model with 62% accuracy and 50% on both controls has a 12% margin
and knows something. Reporting the first number alone is how this project
manufactured a false positive once already.

CANDIDATE TEACHERS, and why these.

The signal is MOTION -- facing-and-speaking differs from facing-and-silent by
4.914 counts against a within-condition split of 1.917, with 79% of the excess
in the top 5% of pixels. Single frames do not contain it. So the teacher reads
a SEQUENCE, rendered as text, which a text-only model can do. That is why no
vision model is required, and it is also why the SHUFFLE control is the
decisive one: it destroys exactly the thing the teacher is supposed to use.

Local ollama models (qwen3:0.6b, llama3.2:1b, gemma2:2b, tev1) and the
typesafe/Jev service are both text-first, so both are eligible.

WHAT IS NOT CLAIMED.

No teacher has been benchmarked on this representation yet, because the
labelled corpus does not exist until Casey records it. This file is the
instrument and the gate, not a result. A run that reports a teacher as usable
without a BLANK and a SHUFFLED control in the same table is not a result and
should be rejected.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from ascii_motion_render import load_session, render_grid, render_trace  # noqa: E402

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

QUESTIONS = {
    "speaking": (
        "This is a sequence of frames rendered as characters. Each frame shows "
        "where the picture CHANGED since the previous frame; brighter means "
        "more change. A person appears in the frame.\n\n"
        "Question: is this person SPEAKING (their mouth and face moving) or "
        "STILL (not talking)? Answer with exactly one word, SPEAKING or STILL."
    ),
    "facing": (
        "This is a sequence of frames rendered as characters. Each frame shows "
        "where the picture CHANGED since the previous frame; brighter means "
        "more change. A person appears in the frame.\n\n"
        "Question: is this person FACING the camera or FACED AWAY? Answer with "
        "exactly one word, FACING or AWAY."
    ),
}


def ollama_generate(model: str, prompt: str, timeout: float = 180.0) -> str:
    """One completion from a local ollama model.

    `think: false` is REQUIRED, and finding out why was itself instructive.
    The first run of this benchmark returned an EMPTY response for every
    single input -- real, shuffled and blank alike -- which looked like a
    broken harness. It was not: qwen3 is a reasoning model, and ollama puts
    its reasoning in a separate `thinking` field. With a small num_predict
    the entire budget went into the thinking preamble and `response` came
    back empty, so every parse returned None and the summary printed
    "NOT USABLE" for a reason that had nothing to do with the model's ability.

    That is the same failure shape as a model answering a constant, so it is
    worth being explicit about: an empty response and a constant response
    look identical in a summary table. Both were caught, but only because the
    raw text was kept and read.
    """
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 16},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    out = d.get("response") or ""
    if not out.strip():
        # A model that ignored think:false still puts its answer there.
        out = d.get("thinking") or ""
    return out


def parse_yes(text: str, positive: str, negative: str) -> bool | None:
    t = (text or "").strip().lower()
    p, n = positive.lower(), negative.lower()
    ip, inn = t.find(p), t.find(n)
    if ip < 0 and inn < 0:
        return None
    if ip < 0:
        return False
    if inn < 0:
        return True
    return ip < inn


def build_windows(maps: np.ndarray, k: int, stride: int, max_per: int = 12):
    """Windows of k consecutive frames, evenly spread over the session."""
    if len(maps) <= k:
        return []
    starts = list(range(0, len(maps) - k, stride))
    if not starts:
        return []
    if len(starts) > max_per:
        sel = np.linspace(0, len(starts) - 1, max_per).astype(int)
        starts = [starts[i] for i in sel]
    return [(s, maps[s:s + k]) for s in starts]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="directory of <name>_<stamp>.npz + .json sessions")
    ap.add_argument("--models", default="qwen3:0.6b",
                    help="comma-separated ollama models")
    ap.add_argument("--question", default="speaking",
                    choices=("speaking", "facing"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--max-per-session", type=int, default=10)
    ap.add_argument("--render", default="grid", choices=("grid", "trace"))
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    corpus = Path(a.corpus)
    sessions = []
    for meta_p in sorted(corpus.glob("*.json")):
        npz_p = meta_p.with_suffix(".npz")
        if not npz_p.is_file():
            continue
        meta = json.loads(meta_p.read_text())
        if meta.get("kind") != "condition":
            continue
        sessions.append((npz_p, meta))
    if not sessions:
        sys.exit(f"no CONDITION sessions in {corpus}. The free corpus cannot "
                 "benchmark a teacher; the labelled conditions are required.")
    print(f"condition sessions: {len(sessions)}")

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    positive, negative = (("SPEAKING", "STILL") if a.question == "speaking"
                          else ("FACING", "AWAY"))
    qhead = QUESTIONS[a.question]

    results = {}
    for model in models:
        rows = []
        for npz_p, meta in sessions:
            maps = load_session(npz_p)
            truth = (meta.get("speak") if a.question == "speaking"
                     else (meta.get("pose") == "facing"))
            for s, win in build_windows(maps, a.k, a.stride, a.max_per_session):
                real = (render_grid(win, len(win)) if a.render == "grid"
                        else render_trace(win, 0, len(win)))
                rng = np.random.default_rng(s + 17)
                perm = rng.permutation(len(win))
                shuf = (render_grid(win[perm], len(win)) if a.render == "grid"
                        else render_trace(win[perm], 0, len(win)))
                blank = ("." * 40 + "\n") * 24

                out = {}
                for variant, text in (("real", real), ("shuffled", shuf),
                                      ("blank", blank)):
                    t0 = time.time()
                    try:
                        resp = ollama_generate(model, f"{qhead}\n\n{text}")
                        out[variant] = parse_yes(resp, positive, negative)
                        out[variant + "_raw"] = resp.strip()[:60]
                    except (urllib.error.URLError, OSError) as exc:
                        out[variant] = None
                        out[variant + "_raw"] = f"ERROR {exc}"
                    out[variant + "_s"] = round(time.time() - t0, 2)
                out["truth"] = bool(truth)
                out["session"] = npz_p.name
                out["start"] = s
                rows.append(out)
                print(f"  {model:14s} {npz_p.stem:22s} truth={out['truth']!s:5s} "
                      f"real={out['real']!s:5s} shuf={out['shuffled']!s:5s} "
                      f"blank={out['blank']!s:5s}")
        results[model] = rows

    summary = {}
    for model, rows in results.items():
        def acc(key):
            vals = [r[key] for r in rows if r.get(key) is not None and r.get("truth") is not None]
            if not vals:
                return None
            correct = sum(1 for r in rows if r.get(key) is not None
                          and r.get("truth") is not None and r[key] == r["truth"])
            return correct / len(vals)
        ar, ash, abl = acc("real"), acc("shuffled"), acc("blank")
        base = max([v for v in (ash, abl) if v is not None], default=None)
        margin = (ar - base) if (ar is not None and base is not None) else None
        summary[model] = {"n": len(rows), "acc_real": ar, "acc_shuffled": ash,
                           "acc_blank": abl, "margin_over_best_control": margin,
                           "usable": bool(margin is not None and margin >= 0.10)}
        print(f"\n{model}:  real {ar}  shuffled {ash}  blank {abl}  "
              f"MARGIN {margin}  -> "
              f"{'USABLE' if summary[model]['usable'] else 'NOT USABLE'}")

    out = {
        "question": a.question,
        "question_head": qhead,
        "render": a.render,
        "k_frames": a.k,
        "models": models,
        "sessions": [str(p) for p, _ in sessions],
        "per_model": summary,
        "raw": results,
        "the_rule": ("the headline number is the MARGIN over the best control, "
                     "not the accuracy. A model with 80% accuracy and 78% on "
                     "the blank knows nothing; one with 62% and 50% on both "
                     "controls knows something. A teacher is usable at a "
                     "margin of 0.10 or more."),
        "limits": [
            "one labelled session per condition is not a benchmark; a model "
            "can score by memorising a session's look rather than its motion",
            "the shuffled control destroys temporal order but keeps spatial "
            "layout, so a teacher that passes it is genuinely reading motion",
            "the blank control catches a teacher that answers from the prompt",
            "a passing teacher still does not license training a student: the "
            "student is evaluated on held-out labelled sessions, which is a "
            "separate measurement",
        ],
    }
    p = Path(a.out) / f"teacher_bench_{a.question}_{a.render}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
