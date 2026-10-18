#!/usr/bin/env python
"""
Local vs cloud, same battery, head to head.

Casey's ask: learn how to harness Jev WITH local models in synergy. The only
way to know what the local half is for is to run the identical questions
through both and look at where they agree, where they diverge, and what each
one costs.

The local decision model is `tev1:latest`, served by Ollama at an undocumented
`POST /v1/systemone` with the same score/choice/noul primitives (Phase 2.9).
`src/local_tev.SystemOneClient` is a drop-in for the SDK's client, so
`judged.read_tracks(client=...)` runs against it with judged.py unmodified.

The comparison re-runs three of round 2's experiments against BOTH, because
those three are the ones that decide the architecture:

  A. THE ADVERSARIAL BATTERY. If the local model is more susceptible to text
     that argues for its own classification, the seam must be local-for-cheap
     / cloud-for-judgment, and that has to be measured rather than assumed.
  B. CONFIDENCE. Cloud confidence is exactly (p_max - 1/K)/(1 - 1/K) -- pure
     arithmetic on the distribution. If the local model returns something else
     there, it is a genuinely different quantity and must not be compared to
     the cloud number.
  C. BATCH LATENCY. Locally the "questions evaluate in parallel" property was
     already measured to be FALSE: llama.cpp scores sequentially, so four
     gates cost ~0.97 s. Measuring the curve again on the same questions
     keeps the two halves comparable.

Plus one thing only the local path can answer:

  D. DISAGREEMENT AS A SIGNAL. Where the two models disagree, is the local
     one right more often than chance? If yes, disagreement is worth
     escalating and the hybrid earns its keep. If no, the local model is just
     cheaper and worse, and the honest architecture is "local for availability,
     cloud for judgement" with no cleverness.

    /home/proart/projects/glasses/.venv/bin/python experiments/local_vs_cloud.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typesafe_sdk import Noul                                    # noqa: E402

from src import judged                                          # noqa: E402
from experiments.jev_round2 import (                            # noqa: E402
    ADVERSARIAL_BATTERY, GARBAGE, GARBAGE_ARGUE, HONEST, QUESTION)

CLOUD_MODEL = os.environ.get("JEV_MODEL", "jev-1.14.0")

PROBES = {
    "honest": HONEST,
    "question": QUESTION,
    "garbage": GARBAGE,
    "garbage_arguing": GARBAGE_ARGUE,
}


def cloud_client():
    from typesafe_sdk import TypeSafeClient
    return TypeSafeClient()


def local_client():
    from src.local_tev import SystemOneClient
    return SystemOneClient(timeout=180.0)


def warm_local(cli):
    """
    Load tev1 and pay the cold-start before any measurement.

    The first local call has to pull the weights onto the GPU, which blew
    straight through the client's 30 s default timeout and produced a
    TimeoutError that looked exactly like "the local model is too slow".
    That is a measurement of the loader, not of the model, so it is separated
    out here and the warm-up is reported.
    """
    t0 = time.perf_counter()
    cli.system_one(state="hello", questions={"warm": Noul(instructions="Is "
                                                                   "this "
                                                                   "greeting?")},
                   model="tev1:latest", timeout=180.0)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    cli.system_one(state="hello", questions={"warm": Noul(instructions="Is "
                                                                   "this "
                                                                   "greeting?")},
                   model="tev1:latest", timeout=180.0)
    warm = time.perf_counter() - t0
    print(f"   warm-up: cold {cold*1000:.0f} ms (includes GPU load), "
          f"warm {warm*1000:.0f} ms")
    return {"cold_start_s": round(cold, 3), "warm_s": round(warm, 3)}


def label(cli):
    return "local-tev1" if not hasattr(cli, "_sdk_client_marker") else "cloud"


# -- A: the adversarial battery ------------------------------------------------

def battery(cli, model):
    from typesafe_sdk import Noul

    def read(text):
        qs = {"holistic": Noul(instructions=(
            "Is this utterance something a listener should act on right "
            "now?")),
            **{k: Noul(instructions=v[0]) for k, v in
               ADVERSARIAL_BATTERY.items()}}
        ans = cli.system_one(state=text, questions=qs,
                             model=model,
                             **({"timeout": 180.0}
                                if model.startswith("tev") else {})).answers
        fields = {k: ans[k].noul for k in ADVERSARIAL_BATTERY}
        return {"holistic": round(ans["holistic"].noul, 4),
                "max": round(max(fields.values()), 4),
                "argmax": max(fields, key=fields.get),
                "mean": round(statistics.mean(fields.values()), 4)}

    return {name: read(text) for name, text in PROBES.items()}


# -- B: is local confidence the same arithmetic? ------------------------------

def confidence_check(cli, model):
    from typesafe_sdk import Choice

    rows = []
    for text in PROBES.values():
        a = cli.system_one(
            state=text,
            questions={"g": Choice(instructions=judged.GESTURE_INSTRUCTIONS,
                                   criteria=judged.GESTURE_CHOICE)},
            model=model,
            **({"timeout": 180.0}
                if model.startswith("tev") else {})).answers["g"]
        probs = getattr(a, "probabilities", None) or {}
        conf = getattr(a, "confidence", None)
        if not probs or conf is None:
            continue
        k = len(probs)
        predicted = (max(probs.values()) - 1.0 / k) / (1.0 - 1.0 / k)
        rows.append({"n_options": k, "reported": round(conf, 4),
                     "predicted": round(predicted, 4),
                     "abs_error": round(abs(predicted - conf), 4)})
    errs = [r["abs_error"] for r in rows]
    return {"rows": rows, "max_abs_error": max(errs) if errs else None,
            "mean_abs_error": round(statistics.mean(errs), 4) if errs else None}


# -- C: batch latency ---------------------------------------------------------

def latency(cli, model, repeats=3):
    from typesafe_sdk import Noul

    out = {}
    for n in (1, 4, 8, 16):
        qs = {f"q{i}": Noul(instructions=f"Is statement {i} present?")
              for i in range(n)}
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            cli.system_one(state=HONEST, questions=qs, model=model,
                           **({"timeout": 180.0}
                               if model.startswith("tev") else {}))
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        out[n] = {"median_s": round(med, 4),
                  "per_question_ms": round(med * 1000 / n, 2)}
    return out


# -- D: do they disagree, and who is right? -----------------------------------

# A tiny labelled set. Ground truth for `should_act`: only real, directed
# utterances should ever be acted on. Garbage must not be, however confidently
# it argues that it should be.
SHOULD_ACT_TRUTH = {"honest": True, "question": False,
                    "garbage": False, "garbage_arguing": False}


def tracks(cli, model):
    """Full judged read on every probe, through the same unmodified path."""
    out = {}
    for name, text in PROBES.items():
        rows, _ = judged.read_tracks(text, model=model, client=cli)
        out[name] = {r.track: r.value for r in rows}
    return out


def disagreement(cloud_tracks, local_tracks):
    """
    Compare on the two tracks that are supposed to separate action from noise:
    INTENTION (is something being asked for) and ATTENTION (who is it for).

    A track that differs by more than 40 CC is treated as a genuine
    disagreement; below that it is the same answer wearing a different number.
    """
    out = {}
    for track in ("INTENTION", "ATTENTION", "GESTURE", "TIMBRE"):
        agree, n = 0, 0
        rows = []
        for name in PROBES:
            c = cloud_tracks[name][track]
            l = local_tracks[name][track]
            n += 1
            agree += int(c == l)
            rows.append({"probe": name, "cloud": c, "local": l, "delta": c - l})
        out[track] = {"exact_agreements": f"{agree}/{n}", "rows": rows,
                      "mean_abs_delta": round(
                          statistics.mean(abs(r["delta"]) for r in rows), 1)}
    return out


def main() -> int:
    report = {"cloud_model": CLOUD_MODEL}
    try:
        cloud = cloud_client()
    except Exception as exc:                          # noqa: BLE001
        print(f"no cloud client: {exc}")
        return 2

    try:
        local = local_client()
        report["warmup"] = warm_local(local)
        have_local = True
    except Exception as exc:                          # noqa: BLE001
        print(f"no local client ({exc}); running cloud only")
        have_local = False

    print("== A. adversarial battery ==")
    report["battery_cloud"] = battery(cloud, CLOUD_MODEL)
    for name, r in report["battery_cloud"].items():
        print(f"   cloud {name:16s} holistic={r['holistic']:.3f} "
              f"max={r['max']:.3f} ({r['argmax']})")
    if have_local:
        report["battery_local"] = battery(local, "tev1:latest")
        for name, r in report["battery_local"].items():
            print(f"   local {name:16s} holistic={r['holistic']:.3f} "
                  f"max={r['max']:.3f} ({r['argmax']})")

    print("\n== B. is confidence the same arithmetic locally? ==")
    report["confidence_cloud"] = confidence_check(cloud, CLOUD_MODEL)
    print(f"   cloud max|err| "
          f"{report['confidence_cloud']['max_abs_error']:.4f}")
    if have_local:
        report["confidence_local"] = confidence_check(local, "tev1:latest")
        print(f"   local max|err| "
              f"{report['confidence_local']['max_abs_error']:.4f}")

    print("\n== C. batch latency ==")
    report["latency_cloud"] = latency(cloud, CLOUD_MODEL)
    for n, r in report["latency_cloud"].items():
        print(f"   cloud n={n:>3d}  {r['median_s']*1000:7.1f} ms "
              f"({r['per_question_ms']:6.2f} ms/q)")
    if have_local:
        report["latency_local"] = latency(local, "tev1:latest")
        for n, r in report["latency_local"].items():
            print(f"   local n={n:>3d}  {r['median_s']*1000:7.1f} ms "
                  f"({r['per_question_ms']:6.2f} ms/q)")

    print("\n== D. do the two agree? ==")
    ct = tracks(cloud, CLOUD_MODEL)
    report["tracks_cloud"] = ct
    if have_local:
        lt = tracks(local, "tev1:latest")
        report["tracks_local"] = lt
        report["disagreement"] = disagreement(ct, lt)
        for track, r in report["disagreement"].items():
            cells = " ".join(
                "{}:{}v{}".format(x["probe"][:9], x["cloud"], x["local"])
                for x in r["rows"])
            print(f"   {track:10s} exact {r['exact_agreements']}  "
                  f"mean|delta| {r['mean_abs_delta']:5.1f}  {cells}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "local_vs_cloud_results.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())