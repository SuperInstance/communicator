"""
calibration — the decisive number, on ground truth built by construction.

Stage 2 of the Phase 2 task brief: "CALIBRATION IS THE DECISIVE NUMBER. An
ECE of 0.36 means a stated 0.9 happens 60% of the time, and an agent that can
AUTONOMOUSLY ACT on this needs that number small."

The problem with calibrating a voice-character judge is ground truth: no
labelled corpus of utterance character exists here, and Jev cannot grade
itself. So every label in the corpus below is TRUE BY CONSTRUCTION -- the
utterance was written to embody the label, the way a tone at 200 Hz is a
ground-truth pitch fixture:

  * GESTURE: the text IS a question / command / statement / reaction /
    greeting by its grammatical form.
  * ATTENTION: the text names its addressee in the words themselves
    ("I", "you", the task, the machine). The "none" class is deliberately
    absent -- it cannot be constructed honestly from text.
  * INTENTION: four designed levels; the metric is ORDER (Spearman) and
    per-level bias, not a binary ECE, because a spectrum has no single
    correct bit to be right or wrong about.
  * TIMBRE: NO CALIBRATION CLAIM AT ALL. Warmth is in the delivery, not the
    words -- a text corpus cannot ground it. Measured here only as a
    consistency probe: do warm-WORDED and cold-WORDED texts separate at all?
    If they do not, TIMBRE-from-text carries zero signal and the track is
    waiting on the acoustic pipeline, honestly.

Metrics are pure functions over recorded rows so they are unit-testable
without a network; the live runner only collects rows and formats.

Platt scaling is deliberately NOT applied as a fix here: decision-ledger
measured that it HURTS an already-calibrated model (ECE 0.033 -> 0.053), so
the first move is always to measure, and only repair on evidence.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

from . import judged

# -- the corpus: labels are true by construction -----------------------------

GESTURE_ITEMS = {          # label = the speech act the grammar performs
    "question": ["What time is it?",
                 "Where did the build log go?",
                 "Is the gate on?"],
    "command": ["Run the tests.",
                "Delete the staging branch.",
                "Open the dashboard."],
    "statement": ["The sky is clear today.",
                  "Paris is the capital of France.",
                  "Water boils at 100 degrees."],
    "reaction": ["Oh wow, it actually worked.",
                 "Huh, that's odd.",
                 "Ah, there it is."],
    "greeting": ["Hey there.",
                 "Good morning.",
                 "Hello again."],
}

ATTENTION_ITEMS = {        # label = the addressee named in the words
    "self": ["Hmm, let me think this through.",
             "Okay, what do I actually want here?",
             "So that's what went wrong, interesting."],
    "listener": ["Can you hear me okay?",
                 "I need you to look at this.",
                 "Would you check the numbers for me?"],
    "task": ["This function is way too long.",
             "The migration needs a rollback plan.",
             "That comment is stale."],
    "tool": ["Why is the build red again?",
             "The linter is being noisy today.",
             "The server just restarted on its own."],
}

INTENTION_ITEMS = {        # label = designed position on the 0..3 scale
    0: ["Hmm.", "Oh.", "Eh, whatever."],
    1: ["Interesting, tell me more.", "Huh, what's that about?",
        "Oh really? Go on."],
    2: ["I need this done today.", "Please review the PR before five.",
        "Make sure the tests pass before you merge."],
    3: ["STOP THE DEPLOY RIGHT NOW.", "Drop everything, the site is down.",
        "Shut it off immediately!"],
}

# TIMBRE wording probe: not a label, a separation question.
TIMBRE_WARM_WORDS = ["Thank you so much, that really helped.",
                     "It's lovely to see you again.",
                     "You did a great job with this."]
TIMBRE_COLD_WORDS = ["Irrelevant. Continue.",
                     "The report is filed. Proceed.",
                     "Noted. Dismissed."]


# -- pure metrics ------------------------------------------------------------

def ece(rows: list[dict], n_bins: int = 10) -> float:
    """Expected calibration error over (p, outcome) rows, outcome in {0,1}.

    Sum over bins of (bin share × |mean predicted p − mean outcome|). A
    perfectly calibrated set scores 0.0; a set that says 0.9 and is right half
    the time scores ~0.4 in that bin.
    """
    if not rows:
        return float("nan")
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        in_bin = [r for r in rows if (lo <= r["p"] < hi)
                  or (b == n_bins - 1 and r["p"] == 1.0)]
        if not in_bin:
            continue
        mean_p = sum(r["p"] for r in in_bin) / len(in_bin)
        mean_y = sum(r["y"] for r in in_bin) / len(in_bin)
        total += (len(in_bin) / len(rows)) * abs(mean_p - mean_y)
    return total


def brier(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return sum((r["p"] - r["y"]) ** 2 for r in rows) / len(rows)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, average ranks on ties, pure standard library."""
    if len(xs) != len(ys) or not xs:
        return float("nan")

    def ranks(vs: list[float]) -> list[float]:
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        out = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


# -- live collection ---------------------------------------------------------

@dataclass
class TrackCalibration:
    track: str
    kind: str              # "choice" | "spectrum" | "wording-probe"
    n: int
    payload: dict


def _reading(readings: list, track: str):
    return next(r for r in readings if r.track == track)


def collect(client=None, *, model: str = judged.DEFAULT_MODEL,
            timeout: float = 8.0, sleep_s: float = 0.0):
    """
    One model call per unique text; every track's rows come off the same call
    set. Returns (results, n_calls, elapsed_s, resolved_model).
    """
    texts, seen = [], set()

    def add(items):
        for t in items:
            if t not in seen:
                seen.add(t)
                texts.append(t)

    add([t for ts in GESTURE_ITEMS.values() for t in ts])
    add([t for ts in ATTENTION_ITEMS.values() for t in ts])
    add([t for ts in INTENTION_ITEMS.values() for t in ts])
    add(TIMBRE_WARM_WORDS + TIMBRE_COLD_WORDS)

    t0 = time.monotonic()
    rows = {}
    resolved = model
    for i, text in enumerate(texts):
        readings, resolved = judged.read_tracks(text, model=model,
                                                client=client, timeout=timeout)
        rows[text] = {r.track: r for r in readings}
        if sleep_s:
            time.sleep(sleep_s)
    elapsed = time.monotonic() - t0
    return rows, len(texts), elapsed, resolved


def score(rows: dict) -> list[TrackCalibration]:
    out = []

    # Choice tracks: accuracy + ECE on p(chosen) vs correctness.
    #
    # p is PLACEMENT probability (probabilities[chosen]), deliberately NOT
    # the gated CC fraction: the presence gate (Phase 2.7) multiplies a
    # "is this property here at all" factor into the controller value, and
    # folding that in would conflate presence with placement. Placement is
    # the well-defined calibrated quantity; presence is question_health's.
    for track, items in (("GESTURE", GESTURE_ITEMS),
                         ("ATTENTION", ATTENTION_ITEMS)):
        truth = {t: label for label, ts in items.items() for t in ts}
        hits, cal_rows, per_class = 0, [], {}
        for text, label in truth.items():
            r = rows.get(text, {}).get(track)
            if r is None or not r.ok:
                continue
            chosen = r.detail.get("choice")
            probs = r.detail.get("probabilities") or {}
            y = 1 if chosen == label else 0
            hits += y
            p = float(probs.get(chosen)) if chosen in probs \
                else max(0.0, min(1.0, r.value / 127.0))
            cal_rows.append({"text": text, "p": p, "y": y,
                             "chosen": chosen, "label": label})
            c = per_class.setdefault(label, {"n": 0, "hit": 0})
            c["n"] += 1
            c["hit"] += y
        if cal_rows:
            out.append(TrackCalibration(track, "choice", len(cal_rows), {
                "accuracy": round(hits / len(cal_rows), 3),
                "ece": round(ece(cal_rows), 3),
                "brier": round(brier(cal_rows), 3),
                "per_class": {k: round(v["hit"] / v["n"], 3)
                              for k, v in sorted(per_class.items())},
                "rows": cal_rows,
            }))

    # INTENTION spectrum: order + per-level bias, in level units.
    xs, ys, by_level = [], [], {}
    for level, texts in INTENTION_ITEMS.items():
        for text in texts:
            r = rows.get(text, {}).get("INTENTION")
            if r is None or not r.ok:
                continue
            pos = r.detail.get("score")
            if pos is None:
                continue
            pos = float(pos)
            n_levels = len(judged.INTENTION_SCALE)
            xs.append(level)
            ys.append(pos)
            by_level.setdefault(level, []).append(pos)
    if xs:
        n_levels = len(judged.INTENTION_SCALE)
        bias = {lvl: round(sum(ps) / len(ps) - lvl, 2)
                for lvl, ps in sorted(by_level.items())}
        mae = sum(abs(b) * len(by_level[lvl]) for lvl, b in bias.items()) / len(xs)
        out.append(TrackCalibration("INTENTION", "spectrum", len(xs), {
            "spearman": round(spearman([float(x) for x in xs], ys), 3),
            "bias_by_level": bias,          # + reads more urgent than designed
            "mae_levels": round(mae, 2),
            "levels": n_levels,
        }))

    # TIMBRE wording probe: separation only, no calibration claim.
    def med_pos(texts):
        ps = []
        for t in texts:
            r = rows.get(t, {}).get("TIMBRE")
            if r is not None and r.ok and r.detail.get("score") is not None:
                ps.append(float(r.detail["score"]))
        ps.sort()
        return ps[len(ps) // 2] if ps else float("nan")

    warm, cold = med_pos(TIMBRE_WARM_WORDS), med_pos(TIMBRE_COLD_WORDS)
    out.append(TrackCalibration("TIMBRE", "wording-probe", 6, {
        "warm_worded_median_pos": round(warm, 2),
        "cold_worded_median_pos": round(cold, 2),
        "separates": (warm - cold) > 0.5 if not (math.isnan(warm) or math.isnan(cold)) else None,
        "scale": len(judged.TIMBRE_SCALE),
        "claim": "wording separation only; delivery warmth is unmeasurable from text",
    }))
    return out


def format_report(results: list[TrackCalibration], *, model: str,
                  n_calls: int, elapsed_s: float) -> str:
    lines = [
        "# Judged-track calibration — first numbers",
        "",
        f"Model `{model}` · {n_calls} calls (one per unique text, all four "
        f"questions per call) · {elapsed_s:.0f}s wall · "
        f"question fingerprint `{judged.question_fingerprint()}`.",
        "",
        "Ground truth is BY CONSTRUCTION (see src/calibration.py docstring): "
        "labels come from how each text was written, not from annotators.",
        "",
    ]
    for res in results:
        lines.append(f"## {res.track} ({res.kind}, n={res.n})")
        lines.append("")
        if res.kind == "choice":
            p = res.payload
            lines.append(f"accuracy **{p['accuracy']}** · ECE **{p['ece']}** · "
                         f"Brier **{p['brier']}** (on p(chosen option))")
            lines.append("")
            lines.append("per-class recall: " + ", ".join(
                f"`{k}` {v}" for k, v in p["per_class"].items()))
        elif res.kind == "spectrum":
            p = res.payload
            lines.append(f"Spearman(order) **{p['spearman']}** · MAE "
                         f"**{p['mae_levels']}** of {p['levels']} levels")
            lines.append("")
            lines.append("bias by designed level (+ = reads more urgent than "
                         "designed): " + ", ".join(
                             f"`{k}` {v:+.2f}" for k, v in p["bias_by_level"].items()))
        else:
            p = res.payload
            lines.append(f"warm-worded median position {p['warm_worded_median_pos']} "
                         f"vs cold-worded {p['cold_worded_median_pos']} of "
                         f"{p['scale']} — separates: **{p['separates']}**")
            lines.append("")
            lines.append(f"_{p['claim']}._")
        lines.append("")
    lines.append("Platt scaling NOT applied: measure first; repair only on "
                 "evidence (decision-ledger: it hurts an already-calibrated model).")
    lines.append("")
    return "\n".join(lines)


def main():
    rows, n_calls, elapsed, resolved = collect()
    results = score(rows)
    report = format_report(results, model=resolved, n_calls=n_calls,
                           elapsed_s=elapsed)
    with open("CALIBRATION.md", "w", encoding="utf-8") as fh:
        fh.write(report)
    ledger = [{"ts": time.time(), "model": resolved, "text": t,
               "tracks": {r.track: {"value": r.value, "ok": r.ok,
                                    "detail": {k: v for k, v in r.detail.items()
                                               if k != "probabilities"}}
                          for r in tr.values()}}
              for t, tr in rows.items()]
    with open("calibration-ledger.jsonl", "w", encoding="utf-8") as fh:
        for row in ledger:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(report)
    print(f"ledger: calibration-ledger.jsonl ({len(ledger)} rows)")


if __name__ == "__main__":
    if not (os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFEAI_KEY")):
        raise SystemExit("set TYPESAFE_API_KEY first")
    main()
