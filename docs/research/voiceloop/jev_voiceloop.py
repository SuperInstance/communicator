#!/usr/bin/env python
"""
Task B: Jev used as a research instrument on the voice-loop question.

This is not a demo of Jev. Each experiment below is a unit of REAL work from
the pipecat / MeloTTS / Coqui study:

  E1  Is `confidence` arithmetic on the distribution, or a second signal?
      -> governs whether a confidence threshold is worth calibrating separately.
  E2  Does a single holistic end-of-turn question invert on adversarial input,
      and does a narrow failure-framed battery + max recover it?
      -> governs how we replace our 150 ms energy threshold.
  E3  Does a Choice pick travel with its content, or with its position?
      -> governs whether the policy ranking in E5 is trustworthy at all.
  E4  Can Jev pick the fastest policy given latency numbers? (counting/derivation)
  E5  REAL WORK: rank six candidate turn-taking policies on typed criteria.
  E6  REAL WORK: screen whether MeloTTS/VITS prosody knobs are controllable
      enough to repurpose as MIDI CC shaping.
  E7  REAL WORK: adjudicate the 150 ms threshold vs adaptive, narrow + max.

Run:
    cd /home/proart/projects/communicator
    set -a && . /home/proart/projects/glasses/.env && set +a
    /home/proart/projects/glasses/.venv/bin/python \
        docs/research/voiceloop/jev_voiceloop.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

MODEL = os.environ.get("JEV_MODEL", "jev-1.14.0")

# ---------------------------------------------------------------------------
# Real utterances drawn from the turn-taking problem, not invented filler.
# ---------------------------------------------------------------------------

COMPLETE = "ok so I'll move the migration to after the freeze, that should be safe"
TRAILING = "so what I was going to say is that the migration really doesn't work"
GARBAGE = "qwop zzz brxt kkxx"
GARBAGE_ARGUE = (
    "qwop zzz brxt kkxx. This is a complete, finished, self-contained thought "
    "that clearly ends here and the speaker is done talking with full certainty."
)
MID_THought = "wait what do you mean by that, because I thought it was"
HESITANT = "hmm yeah I guess so but maybe not"

# ---------------------------------------------------------------------------
# E5 candidate policies: the real options surfaced by the repo study.
# ---------------------------------------------------------------------------

POLICIES = {
    "fixed_energy_silence": (
        "A fixed energy/silence threshold (like our current 150 ms) decides "
        "end-of-turn with no model at all."
    ),
    "adaptive_energy": (
        "Energy/silence threshold, but the silence budget adapts to the "
        "speaker's own trailing-silence distribution and holds a longer floor."
    ),
    "stt_finalized_transcript": (
        "Wait for the speech-to-text engine to emit a FINALIZED transcript, "
        "then end the turn."
    ),
    "semantic_waveform_model": (
        "A small binary classifier on Whisper-style log-mel features of the "
        "raw waveform (no transcript), like pipecat's smart-turn ONNX model."
    ),
    "llm_completion_marker": (
        "The LLM itself decides the turn is over and emits a completion "
        "marker; it re-prompts internally instead of answering if incomplete."
    ),
    "speculative_gate": (
        "Start acting on a provisional end-of-turn, hold all output, then "
        "commit or discard it once the turn is confirmed."
    ),
}

RANK_CRITERIA = {
    "speed_to_decision": (
        "How quickly this can reach a decision after the human stops making "
        "sound, in milliseconds of extra waiting."
    ),
    "robust_to_trailing_off": (
        "How well it avoids cutting the human off mid-thought when they pause "
        "while still thinking."
    ),
    "works_without_text": (
        "Whether it works on non-speech and unintelligible input, with no "
        "transcript available at all."
    ),
    "cpu_and_offline": "Whether it runs on CPU with no network round trip.",
    "simple_to_integrate": "How little new machinery it needs to get working.",
    "permissive_license": "Whether the code and weights are permissively licensed.",
    "fails_safe": (
        "Whether its failure mode is to wait rather than to act on a "
        "half-finished thought."
    ),
}

# ---------------------------------------------------------------------------
# E6: the three VITS/MeloTTS inference knobs, in their real roles.
# ---------------------------------------------------------------------------

PROSODY_KNOBS = {
    "noise_scale": (
        "VITS noise_scale (MeloTTS default 0.6-0.667): scales the sampled "
        "noise in the flow that generates the hidden representation, i.e. how "
        "much the prosody varies between takes of the same sentence."
    ),
    "noise_scale_w": (
        "VITS noise_scale_w (default 0.8): scales the noise in the duration "
        "predictor, i.e. how much phoneme and pause timing jitter varies "
        "between takes."
    ),
    "sdp_ratio": (
        "VITS sdp_ratio (MeloTTS default 0.2): scales the stochastic duration "
        "predictor; at 0 durations are the deterministic prediction, raising "
        "it makes each token's duration a stochastic draw."
    ),
}


def client():
    return TypeSafeClient()


def ask(cli, state, questions):
    return cli.system_one(state=state, questions=questions, model=MODEL).answers


def confirm(conf):
    """confidence is only defined on Choice/Score; Noul answers carry none."""
    return None if conf is None else round(float(conf), 4)


# ---------------------------------------------------------------------------
# E1 -- is confidence arithmetic on the distribution?
# ---------------------------------------------------------------------------

def e1_confidence_is_arithmetic(cli):
    """
    HYPOTHESIS (H1): confidence == (p_max - 1/K) / (1 - 1/K) exactly, i.e. it
    is a deterministic function of the returned distribution.

    If exact, a confidence threshold is not a second, separately-calibratable
    signal and tuning it is the same as tuning the distribution.
    """
    probes = [
        ("complete", COMPLETE),
        ("trailing", TRAILING),
        ("garbage", GARBAGE),
        ("mid_thought", MID_THought),
        ("hesitant", HESITANT),
    ]
    ks = {
        2: {"yes": "the thought is finished", "no": "the speaker may continue"},
        3: {"done": "finished", "maybe": "unclear", "not_done": "still going"},
        4: {
            "done": "finished",
            "trailing": "trailing off but could continue",
            "mid": "stopped to think mid-sentence",
            "garbage": "not speech at all",
        },
    }

    rows, errs = [], []
    for label, text in probes:
        for k, crit in ks.items():
            a = ask(cli, text, {
                "c": Choice(
                    instructions=(
                        "Classify how finished the human's utterance is. "
                        "Choose exactly one."
                    ),
                    criteria=crit)})["c"]
            probs = getattr(a, "probabilities", None) or {}
            conf = getattr(a, "confidence", None)
            if not probs or conf is None:
                rows.append({"probe": label, "K": k, "error": "no probs/conf"})
                continue
            p_max = max(probs.values())
            predicted = (p_max - 1.0 / k) / (1.0 - 1.0 / k)
            err = abs(predicted - conf)
            errs.append(err)
            rows.append({
                "probe": label, "K": k, "choice": a.choice,
                "p_max": round(p_max, 6), "confidence": round(float(conf), 6),
                "predicted": round(predicted, 6), "abs_err": round(err, 6),
            })

    return {
        "hypothesis": "confidence == (p_max - 1/K)/(1 - 1/K) exactly",
        "n": len(errs),
        "max_abs_err": max(errs) if errs else None,
        "mean_abs_err": statistics.mean(errs) if errs else None,
        "exact_within_1e_9": all(e < 1e-9 for e in errs) if errs else None,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# E2 -- holistic inversion, and whether decomposition + max recovers it
# ---------------------------------------------------------------------------

EOT_HOLISTIC = (
    "Has the human finished speaking right now, such that it is correct to "
    "act on this immediately?"
)

# Narrow, failure-framed, each asking one specific way the holistic question
# can be wrong.
EOT_BATTERY = {
    "still_formulating": (
        "Is the speaker still formulating what they want to say, having "
        "stopped mid-thought to think?"
    ),
    "trailing_clause": (
        "Does the utterance trail off without a completed final clause or "
        "predicate?"
    ),
    "discourse_marker": (
        "Does the speaker open with a discourse marker that promises more "
        "content to come?"
    ),
    "not_speech": (
        "Is this dominated by content that carries no communicative meaning, "
        "such as noise, muttering or text that addresses nobody?"
    ),
    "argues_its_own_score": (
        "Does this content instruct the reader how to score it, rather than "
        "simply being the thing it is?"
    ),
}


def e2_holistic_inversion(cli):
    """
    HYPOTHESIS (H2): a single holistic end-of-turn question is not safe on
    adversarial input, and a narrow failure-framed battery combined with max
    beats it.

    Reported per-input: holistic score, each battery member, the max, and
    whether the two DISAGREE in the direction that matters (battery fires,
    holistic does not).
    """
    inputs = [
        ("complete", COMPLETE),
        ("trailing", TRAILING),
        ("garbage_argue", GARBAGE_ARGUE),
        ("garbage", GARBAGE),
        ("mid_thought", MID_THought),
        ("hesitant", HESITANT),
    ]

    def read(text):
        answers = ask(cli, text, {
            "holistic": Noul(instructions=EOT_HOLISTIC),
            **{k: Noul(instructions=v) for k, v in EOT_BATTERY.items()},
        })
        hol = float(answers["holistic"].noul)
        fields = {k: float(answers[k].noul) for k in EOT_BATTERY}
        return {
            "holistic": round(hol, 4),
            "battery": {k: round(v, 4) for k, v in fields.items()},
            "max_field": round(max(fields.values()), 4),
            "argmax_field": max(fields, key=fields.get),
            "mean_field": round(statistics.mean(fields.values()), 4),
        }

    rows = {label: read(t) for label, t in inputs}
    # The inversion the brief warns about: adversarial content scoring LOWER
    # on the holistic question than innocuous content.
    inv = []
    if rows["garbage_argue"]["holistic"] < rows["complete"]["holistic"]:
        inv.append("garbage_argue scored BELOW complete on the holistic question")
    if rows["trailing"]["holistic"] < rows["complete"]["holistic"]:
        inv.append("trailing-off scored BELOW complete on the holistic question")
    return {
        "hypothesis": "holistic inverts on adversarial input; narrow+max recovers",
        "inversions_observed": inv,
        "battery_beats_holistic_on": [
            k for k, r in rows.items()
            if r["max_field"] > r["holistic"]
        ],
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# E3 -- does a pick travel with content or with position?
# ---------------------------------------------------------------------------

E3_SUBJECT = (
    "We must decide how a peripheral detects that a person has finished "
    "speaking. Evaluate the six candidate policies below."
)


def e3_position_preference(cli):
    """
    HYPOTHESIS (H3): a Choice pick is stable under option reordering.

    Method: ask the same question twice with the options in opposite order and
    report the canonical label each time. If the labels differ, the pick did
    not travel with the content and any ranking built on it is UNRESOLVED.
    """
    names = list(POLICIES)
    fwd = {n: POLICIES[n] for n in names}
    rev = {n: POLICIES[n] for n in reversed(names)}

    picks = {}
    for tag, order in (("forward", fwd), ("reversed", rev)):
        a = ask(cli, E3_SUBJECT, {
            "best": Choice(
                instructions=(
                    "Which single policy should we adopt as the primary "
                    "end-of-turn decision mechanism for a local peripheral "
                    "system? Choose exactly one."
                ),
                criteria=order)})["best"]
        picks[tag] = {
            "choice": a.choice,
            "p_winner": round((getattr(a, "probabilities", None) or {})
                              .get(a.choice, 0.0), 4),
            "confidence": confirm(getattr(a, "confidence", None)),
        }

    first_fwd = names[0]
    first_rev = names[-1]
    resolved = picks["forward"]["choice"] == picks["reversed"]["choice"]
    return {
        "hypothesis": "pick travels with content, not position",
        "first_option_forward": first_fwd,
        "first_option_reversed": first_rev,
        "forward": picks["forward"],
        "reversed": picks["reversed"],
        "RESOLVED": resolved,
        "picked_first_option_both_orders": (
            picks["forward"]["choice"] == first_fwd
            and picks["reversed"]["choice"] == first_rev
        ),
    }


# ---------------------------------------------------------------------------
# E4 -- counting / derivation
# ---------------------------------------------------------------------------

def e4_cannot_count(cli):
    """
    HYPOTHESIS (H4): Jev cannot count or derive, only compare. Asked to pick
    the policy with the LOWEST worst-case added latency, it will not reliably
    identify the numeric minimum.

    Method: embed four labelled latency options in the state text, ask it to
    pick the lowest. Record whether the pick is the true minimum.
    """
    options = {
        "adaptive_energy": 180.0,
        "fixed_energy_silence": 150.0,
        "semantic_waveform_model": 40.0,
        "stt_finalized_transcript": 620.0,
    }
    state_lines = ["Measured worst-case added latency after speech stops:"]
    for k, v in options.items():
        state_lines.append(f"- {k}: {v:.0f} ms")
    state = "\n".join(state_lines)

    # Ask it to pick the minimum.
    a = ask(cli, state, {
        "min_latency": Choice(
            instructions=(
                "Which single policy has the lowest worst-case added latency "
                "in milliseconds? Choose exactly one."
            ),
            criteria={k: f"worst-case added latency is {v:.0f} ms"
                      for k, v in options.items()})})["min_latency"]
    pick = a.choice
    true_min = min(options, key=options.get)

    # And ask it to do the same while explicitly naming the comparison.
    b = ask(cli, state, {
        "min_latency": Score(
            instructions=(
                "How much added latency in milliseconds does this policy add "
                "after the human stops speaking? Use the numeric values in "
                "the state."
            ),
            # NOTE: Score.criteria is a Sequence (list), unlike Choice's Mapping.
            criteria=[
                "adds less than 100 ms",
                "adds between 100 ms and 400 ms",
                "adds more than 400 ms",
            ])})["min_latency"]
    band_of = {k: (0 if v < 100 else 1 if v <= 400 else 2)
               for k, v in options.items()}

    return {
        "hypothesis": "Jev cannot count or derive",
        "options_ms": options,
        "true_minimum": true_min,
        "pick": pick,
        "correct": pick == true_min,
        "pick_p": round((getattr(a, "probabilities", None) or {})
                        .get(pick, 0.0), 4),
        "true_min_p": round((getattr(a, "probabilities", None) or {})
                            .get(true_min, 0.0), 4),
        "score_band_probe": {
            "score": float(b.score),
            "true_bands": band_of,
        },
    }


# ---------------------------------------------------------------------------
# E5 -- REAL WORK: rank the six policies on typed criteria
# ---------------------------------------------------------------------------

def e5_rank_policies(cli):
    """
    REAL WORK. The ranking the study actually needs: which turn-taking policy
    should a peripheral adopt, scored on seven typed criteria.

    Two readings are compared, because the brief's failure mode predicts they
    differ:
      (a) one holistic question per policy,
      (b) a narrow per-criterion question bank combined with max.
    """
    names = list(POLICIES)

    holistic = {}
    for n in names:
        a = ask(cli, f"Candidate policy: {POLICIES[n]}", {
            "q": Noul(
                instructions=(
                    "Is this policy a good primary end-of-turn decision "
                    "mechanism for a LOCAL peripheral system that produces "
                    "musical/prosodic output rather than speech, and that "
                    "must run continuously on CPU?"
                )
            )})["q"]
        holistic[n] = round(float(a.noul), 4)

    per_crit = {}
    for n in names:
        answers = ask(cli, f"Candidate policy: {POLICIES[n]}", {
            f"c{i}": Noul(instructions=f"Does this policy {desc}?")
            for i, (_, desc) in enumerate(RANK_CRITERIA.items())
        })
        vals = [round(float(answers[f"c{i}"].noul), 4)
                for i in range(len(RANK_CRITERIA))]
        per_crit[n] = vals

    crit_names = list(RANK_CRITERIA)
    agg = {}
    for n, vals in per_crit.items():
        agg[n] = {
            "max_crit": round(max(vals), 4),
            "mean_crit": round(statistics.mean(vals), 4),
            "weakest": crit_names[vals.index(min(vals))],
            "strongest": crit_names[vals.index(max(vals))],
        }

    hol_rank = sorted(names, key=lambda n: -holistic[n])
    max_rank = sorted(names, key=lambda n: -agg[n]["max_crit"])
    mean_rank = sorted(names, key=lambda n: -agg[n]["mean_crit"])

    return {
        "criteria": crit_names,
        "holistic": holistic,
        "per_criterion": per_crit,
        "aggregated": agg,
        "rank_by_holistic": hol_rank,
        "rank_by_max_of_criteria": max_rank,
        "rank_by_mean_of_criteria": mean_rank,
        "max_and_mean_agree_at_top": max_rank[0] == mean_rank[0],
        "holistic_and_max_agree_at_top": hol_rank[0] == max_rank[0],
    }


# ---------------------------------------------------------------------------
# E6 -- REAL WORK: prosody controllability screening
# ---------------------------------------------------------------------------

def e6_prosody_screen(cli):
    """
    REAL WORK. We do not want speech -- we want musical/prosodic response as
    MIDI CC. The question is whether the three VITS inference knobs give
    orthogonal, independently-usable control, or whether they are entangled.

    Method: ask, for each knob, whether it changes TIMING or PROSODY or both,
    and whether it is safe to drive continuously in real time.
    """
    state = (
        "MeloTTS/VITS inference knobs and their real roles:\n"
        + "\n".join(f"- {k}: {v}" for k, v in PROSODY_KNOBS.items())
    )
    axes = {
        "timing": "Does this knob change WHEN things happen (durations, pauses, jitter)?",
        "prosody": "Does this knob change HOW something sounds (intensity, contour, expressiveness)?",
        "realtime_safe": (
            "Can this knob be re-sampled continuously between utterances at "
            "audio-frame rate with no state carried between calls?"
        ),
        "orthogonal": (
            "Does this knob act independently of the other two, so that moving "
            "it does not also move them?"
        ),
    }
    answers = ask(cli, state, {
        f"{k}__{n}": Noul(instructions=q)
        for k in PROSODY_KNOBS
        for n, q in axes.items()
    })
    rows = {}
    for k in PROSODY_KNOBS:
        rows[k] = {n: round(float(answers[f"{k}__{n}"].noul), 4) for n in axes}

    # The entanglement measure: if every knob scores identically on every axis,
    # the three knobs are one control wearing three names.
    vectors = {k: tuple(rows[k][n] for n in axes) for k in PROSODY_KNOBS}
    distinct = len(set(vectors.values()))
    return {
        "axes": list(axes),
        "rows": rows,
        "distinct_control_vectors": distinct,
        "knobs_are_separate": distinct == len(PROSODY_KNOBS),
    }


# ---------------------------------------------------------------------------
# E7 -- REAL WORK: adjudicate our 150 ms threshold
# ---------------------------------------------------------------------------

def e7_threshold_adjudication(cli):
    """
    REAL WORK. Our watcher segments on a 150 ms energy silence threshold and
    we know that is naive. Adjudicate: does a longer/adaptive threshold buy
    enough robustness to justify the extra wait?

    Narrow failure-framed questions about the SYMPTOM, combined by max, then a
    single Choice for the verdict.
    """
    SHORT_TRAILING = (
        "A watcher that declares the turn over after 150 ms of silence cuts "
        "people off whenever they pause for a breath, a search for a word, or "
        "a change of mind."
    )
    LONG_TRAILING = (
        "A watcher that waits 600 ms of silence before declaring the turn over "
        "rarely cuts people off mid-thought, because a human thinking silence "
        "is almost always longer than 600 ms."
    )

    def symptom(state):
        answers = ask(cli, state, {
            "cuts_off": Noul(instructions=(
                "Will this cut the human off mid-thought during a pause while "
                "they are still thinking?")),
            "feels_slow": Noul(instructions=(
                "Will this feel slow or unresponsive to the human?")),
            "reacts_to_nothing": Noul(instructions=(
                "Will this also suppress reactions to genuine one-word "
                "acknowledgements and short utterances?")),
        })
        vals = {k: round(float(answers[k].noul), 4)
                for k in ("cuts_off", "feels_slow", "reacts_to_nothing")}
        return {
            **vals,
            "max": round(max(vals.values()), 4),
            "argmax": max(vals, key=vals.get),
        }

    short = symptom(SHORT_TRAILING)
    long = symptom(LONG_TRAILING)

    verdict = ask(cli, (
        f"Option A, the short threshold: {SHORT_TRAILING}\n"
        f"Option B, the long threshold: {LONG_TRAILING}"), {
        "pick": Choice(
            instructions=(
                "Which should we adopt as the silence threshold for a "
                "peripheral nervous system that should almost never interrupt "
                "and should react musically rather than speak?"
            ),
            criteria={
                "short_150ms": "keep the 150 ms threshold, react fast",
                "long_600ms": "raise it to 600 ms, never cut the human off",
                "adaptive": (
                    "make it adaptive -- track the speaker's own trailing-"
                    "silence distribution and hold a long floor"
                ),
                "none": "neither; energy alone cannot decide this",
            })})["pick"]

    return {
        "short": short,
        "long": long,
        "max_gap": round(abs(short["max"] - long["max"]), 4),
        "verdict_choice": verdict.choice,
        "verdict_p": round((getattr(verdict, "probabilities", None) or {})
                           .get(verdict.choice, 0.0), 4),
        "verdict_confidence": confirm(getattr(verdict, "confidence", None)),
        "verdict_all_p": {k: round(v, 4) for k, v in
                          (getattr(verdict, "probabilities", None) or {}).items()},
    }


def e8_position_under_indistinguishable(cli):
    """
    HYPOTHESIS (H5): the position preference appears ONLY when the options are
    genuinely indistinguishable. E3 found no position preference with six
    clearly-differentiated policies, so test the exact condition the brief
    describes: options whose descriptions are semantically empty.

    Method: two variants that are byte-identical as options except for a label,
    so nothing in the content can distinguish them. Ask in both orders, 3
    repeats each. If the pick lands on the first position regardless of which
    label is first, the pick is positional, not semantic -> UNRESOLVED.
    """
    # Variant A and B are semantically identical; only the label differs.
    label_sets = [
        ("variant_A", "variant_B"),
        ("variant_C", "variant_D"),
        ("variant_E", "variant_F"),
    ]
    desc = "a candidate component considered for adoption"

    rows = []
    for left, right in label_sets:
        for order in ("forward", "reversed"):
            crit = ({left: desc, right: desc} if order == "forward"
                    else {right: desc, left: desc})
            picks = []
            for _ in range(3):
                a = ask(cli, "We are comparing two indistinguishable candidate "
                             "components. Choose exactly one to adopt.", {
                    "pick": Choice(
                        instructions="Which component should we adopt?",
                        criteria=crit)})["pick"]
                picks.append(a.choice)
            rows.append({
                "pair": f"{left}/{right}",
                "order": order,
                "first_option": left if order == "forward" else right,
                "picks": picks,
                "all_first": all(p == (left if order == "forward" else right)
                                for p in picks),
            })

    n = len(rows)
    first_rate = sum(r["all_first"] for r in rows) / n
    resolved = first_rate < 1.0
    return {
        "hypothesis": "position preference appears only with indistinguishable options",
        "n_trials": n,
        "picked_first_option_rate": round(first_rate, 4),
        "RESOLVED": resolved,
        "verdict": ("UNRESOLVED - pick is positional, carries no information"
                    if not resolved else
                    "RESOLVED - pick is not purely positional"),
        "rows": rows,
    }


# ---------------------------------------------------------------------------

EXPERIMENTS = [
    ("1", "Is confidence arithmetic on the distribution?", e1_confidence_is_arithmetic),
    ("2", "Holistic inversion vs narrow battery + max", e2_holistic_inversion),
    ("3", "Does the pick travel with content or position?", e3_position_preference),
    ("4", "Can Jev count/derive a minimum latency?", e4_cannot_count),
    ("5", "REAL: rank six turn-taking policies", e5_rank_policies),
    ("6", "REAL: prosody knob controllability", e6_prosody_screen),
    ("7", "REAL: adjudicate the 150 ms threshold", e7_threshold_adjudication),
    ("8", "Position preference under indistinguishable options", e8_position_under_indistinguishable),
]


def main() -> int:
    if not (os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFEAI_KEY")):
        print("no TYPESAFE_API_KEY in env")
        return 2
    only = set(sys.argv[1:])
    cli = client()
    print(f"Jev model: {MODEL}")
    report = {"model": MODEL, "results": {}}

    for tag, title, fn in EXPERIMENTS:
        if only and tag not in only:
            continue
        print(f"\n== E{tag}. {title} ==")
        t0 = time.perf_counter()
        res = fn(cli)
        dt = time.perf_counter() - t0
        report["results"][f"E{tag}"] = res
        print(f"   ({dt:.1f}s)")
        for line in summarise(tag, res):
            print("   " + line)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "jev_voiceloop_results.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nwrote {out}")
    return 0


def summarise(tag, r):
    out = []
    if tag == "1":
        out.append(f"n={r['n']} max|err|={r['max_abs_err']:.3e} "
                   f"exact(<1e-9)={r['exact_within_1e_9']}")
    elif tag == "2":
        out.append(f"inversions: {r['inversions_observed'] or 'none observed'}")
        for k, v in r["rows"].items():
            out.append(f"  {k:15s} holistic={v['holistic']:.3f} "
                       f"max={v['max_field']:.3f} ({v['argmax_field']})")
    elif tag == "3":
        out.append(f"forward={r['forward']['choice']} "
                   f"reversed={r['reversed']['choice']} "
                   f"RESOLVED={r['RESOLVED']}")
        out.append(f"picked_first_option_both_orders="
                   f"{r['picked_first_option_both_orders']}")
    elif tag == "4":
        out.append(f"pick={r['pick']} true_min={r['true_minimum']} "
                   f"correct={r['correct']}")
    elif tag == "5":
        out.append(f"rank max   : {r['rank_by_max_of_criteria']}")
        out.append(f"rank mean  : {r['rank_by_mean_of_criteria']}")
        out.append(f"rank holis.: {r['rank_by_holistic']}")
        out.append(f"max==mean top? {r['max_and_mean_agree_at_top']}  "
                   f"holistic==max top? {r['holistic_and_max_agree_at_top']}")
    elif tag == "6":
        out.append(f"distinct control vectors={r['distinct_control_vectors']} "
                   f"knobs_are_separate={r['knobs_are_separate']}")
        for k, v in r["rows"].items():
            out.append(f"  {k:14s} " +
                       " ".join(f"{a}={v[a]:.2f}" for a in v))
    elif tag == "7":
        out.append(f"short max={r['short']['max']:.3f} ({r['short']['argmax']})")
        out.append(f"long  max={r['long']['max']:.3f} ({r['long']['argmax']})")
        out.append(f"verdict={r['verdict_choice']} p={r['verdict_p']} "
                   f"conf={r['verdict_confidence']}")
    elif tag == "8":
        out.append(f"picked_first_option_rate={r['picked_first_option_rate']} "
                   f"RESOLVED={r['RESOLVED']}")
        out.append(f"verdict: {r['verdict']}")
        for row in r["rows"]:
            out.append(f"  {row['pair']:22s} {row['order']:9s} "
                       f"picks={row['picks']} all_first={row['all_first']}")
    return out


if __name__ == "__main__":
    raise SystemExit(main())