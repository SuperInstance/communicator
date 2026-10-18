"""
rewrite_probe -- can a rewrite fix the three ill-posed questions, and if so,
which kind of rewrite?

THE FALSIFIED BASELINE (Phase 2.6, jev-1.14.0, src/question_health.py):

    track        garbage  real     verdict
    INTENTION      0.95   0.88    ill-posed  (scores HIGHER on "qwop zzz brxt")
    ATTENTION      0.80   0.70    ill-posed
    TIMBRE         0.04   0.27    degenerate
    GESTURE        0.47   0.99    healthy

THE HYPOTHESIS UNDER TEST, stated so it can fail:

    This is not a WORDING problem. Score and Choice have no way to decline --
    both primitives must commit to a position, so an utterance with no such
    property is still forced onto the scale, and it is placed there
    confidently. `Noul` is the only primitive here that can answer "I don't
    know" (0.5), because it returns the PROBABILITY OF PRESENCE rather than a
    chosen level.

    Therefore: adding an explicit presence GATE should fix the inversion, and
    re-wording the scale should NOT. If the reworded variant separates as well
    as the gated one, this hypothesis is wrong and says so.

The distinction matters because the two fixes have different costs. A gate costs
one extra question per track in the SAME single request. A reword needs a human
to keep tuning it, forever, against a model that may change.

WHAT IS MEASURED, and why it is the CC and not the confidence:

  question_health.py scores a track on SPREAD-CONFIDENCE, which answers "can the
  model tell something from nothing". That is the right question for an
  instrument, but it is not the question a synthesiser asks. A synthesiser is
  driven by the CC. So the decisive number here is the CC SEPARATION: median CC
  on garbage vs median CC on real. If a variant cannot move the CC, it has not
  fixed anything, no matter what its confidence looks like.

  Spread-confidence is still recorded, per track, so the numbers stay comparable
  with Phase 2.6 rather than being quietly replaced by a friendlier metric.

ONE CALL PER PROBE PER VARIANT. The one-call rule is architectural (a voice
loop pays latency per question, not per request), so all questions of a variant
ride together. Both gates and scale questions are asked in the same request --
the docs confirm adding questions "barely changes the response time".

THIS MODULE DOES NOT CHANGE judged.py. It is a probe. A rewrite is only adopted
after it has beaten the baseline on the live numbers, because the CC values in
judged.py are what the instrument is currently playing.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from . import judged

# NoulCriteria's keys are `true`/`false`, NOT `yes`/`no`. Writing the obvious
# `{"yes": ..., "no": ...}` raises
#   ValidationError: criteria.yes -- Extra inputs are not permitted
# which is a confusing way to learn a keyword. Guarded by a test.
NOUL_CRITERIA_KEYS = ("true", "false")

# -- the presence gates -----------------------------------------------------
#
# Each gate asks the EXISTENCE question, stripped of the level question. That
# separation is the whole point: confidence conflates "does this exist" with
# "can I tell which one", and garbage reliably fails only the first.

GATE_INTENTION = (
    "Does this utterance convey an intention to be acted upon by a listener "
    "or system? Answer no for text with no communicative content."
)
GATE_ATTENTION = (
    "Is this utterance's focus directed at an addressee -- a person, a "
    "system, or the work in front of the speaker? Answer no for text with no "
    "communicative content."
)
GATE_TIMBRE = (
    "Does this utterance, judged from its words alone, convey a deliberate "
    "vocal colour such as warm, cold or breathy?"
)
GATE_GESTURE = (
    "Does this utterance perform a deliberate speech act toward someone -- "
    "asking, instructing, greeting, reacting?"
)


# -- the reworded scales ----------------------------------------------------
#
# The most natural wording fix available: give the scale somewhere to PUT
# garbage. If the hypothesis is right this does not help, because the model is
# not refusing the scale -- it is using it. Kept in full so the negative result
# is inspectable rather than asserted.

INTENTION_SCALE_V2 = [
    "idle: no particular intent behind it, ambient or reflexive.",
    "attentive: mild interest, the speaker is considering something.",
    "focused: deliberate, the speaker means exactly what they said.",
    "urgent: needs acting on now, something is time-sensitive.",
    "not an utterance: no communicative content at all -- noise, gibberish, "
    "or strings that are not speech.",
]

ATTENTION_CHOICE_V2 = {
    "self": "The speaker is talking to themselves or thinking aloud.",
    "listener": "Directed at the person or system being spoken to.",
    "task": "About the work in front of them rather than a person.",
    "tool": "About the state or output of a tool or machine.",
    "none": "No clear addressee; neutral or incidental.",
    "not an utterance": "No communicative content at all -- noise or gibberish.",
}

TIMBRE_SCALE_V2 = [
    "cold: distant, flat, detached. Not warmth, not brightness.",
    "warm: close, gentle, affectionate. Not breathiness.",
    "bright: crisp, clear, energetic, lightly upbeat.",
    "nasal: resonant through the nose, slightly pinched or whiny.",
    "breathy: audible breath, soft edges, airy, little vocal fold.",
    "no vocal colour: the words alone carry no delivery at all.",
]

INTENTION_INSTRUCTIONS_V2 = (
    "How strongly does the speaker intend to be acted upon? Use `not an "
    "utterance` for anything that is not meaningful speech."
)
ATTENTION_INSTRUCTIONS_V2 = (
    "Who or what is this utterance focused on? Use `not an utterance` for "
    "anything that is not meaningful speech."
)
TIMBRE_INSTRUCTIONS_V2 = (
    "Where does this utterance sit on the warm-to-cold scale, judged from the "
    "words alone? Use `no vocal colour` when the words carry no delivery."
)


@dataclass
class Variant:
    """
    One candidate question set, and how to reduce its answers to CC values.

    `specs` is the reduction table: (answer_key, kind, track, n_levels). It is
    declared rather than introspected because the SDK answer objects are
    pydantic models with Optional fields, and a runtime `isinstance` check would
    silently mis-reduce a `score` that arrived as None.
    """
    name: str
    specs: list[tuple]
    gates: dict = field(default_factory=dict)   # answer_key -> track
    note: str = ""


def _baseline_specs():
    return [
        ("timbre", "score", "TIMBRE", len(judged.TIMBRE_SCALE)),
        ("intention", "score", "INTENTION", len(judged.INTENTION_SCALE)),
        ("attention", "choice", "ATTENTION", None),
        ("gesture", "choice", "GESTURE", None),
    ]


def _reworded_specs():
    return [
        ("timbre", "score", "TIMBRE", len(TIMBRE_SCALE_V2)),
        ("intention", "score", "INTENTION", len(INTENTION_SCALE_V2)),
        ("attention", "choice", "ATTENTION", None),
        ("gesture", "choice", "GESTURE", None),
    ]


def build_questions(variant: str):
    """
    Construct the SDK question dict for a variant name.

    The baseline is built HERE, from the scale constants, and deliberately
    does NOT call judged.build_questions(). judged.py adopted the gates after
    this experiment ran, so delegating would silently relabel the experiment's
    control: "baseline" would become today's production question set, the
    recorded Phase 2.6 numbers would stop being reproducible, and a future
    reader would be comparing the gate against itself. An experiment's control
    has to be pinned to the thing it measured, not to whatever the module now
    says.

    Raises KeyError on an unknown variant rather than defaulting to baseline --
    a typo in an experiment name must not silently measure the wrong thing.
    """
    from typesafe_sdk import Choice, Noul, Score

    if variant == "baseline":
        # The ungated Phase 2.6 question set, spelled out.
        return {
            "timbre": Score(instructions=judged.TIMBRE_INSTRUCTIONS,
                            criteria=judged.TIMBRE_SCALE),
            "intention": Score(instructions=judged.INTENTION_INSTRUCTIONS,
                               criteria=judged.INTENTION_SCALE),
            "attention": Choice(instructions=judged.ATTENTION_INSTRUCTIONS,
                                criteria=judged.ATTENTION_CHOICE),
            "gesture": Choice(instructions=judged.GESTURE_INSTRUCTIONS,
                              criteria=judged.GESTURE_CHOICE),
        }

    if variant == "reworded":
        return {
            "timbre": Score(instructions=TIMBRE_INSTRUCTIONS_V2,
                            criteria=TIMBRE_SCALE_V2),
            "intention": Score(instructions=INTENTION_INSTRUCTIONS_V2,
                               criteria=INTENTION_SCALE_V2),
            "attention": Choice(instructions=ATTENTION_INSTRUCTIONS_V2,
                                criteria=ATTENTION_CHOICE_V2),
            "gesture": build_questions("baseline")["gesture"],
        }

    if variant in ("gated", "gated_reworded"):
        qs = (build_questions("baseline") if variant == "gated"
              else build_questions("reworded"))
        # The gates ride in the SAME request. Four more questions, one round
        # trip -- the cost the one-call rule exists to avoid paying twice.
        qs = dict(qs)
        qs["gate_intention"] = Noul(
            instructions=GATE_INTENTION,
            criteria={"false": "No communicative content; nothing to act on."})
        qs["gate_attention"] = Noul(
            instructions=GATE_ATTENTION,
            criteria={"false": "No communicative content; no addressee."})
        qs["gate_timbre"] = Noul(
            instructions=GATE_TIMBRE,
            criteria={"false": "The words alone carry no deliberate delivery."})
        qs["gate_gesture"] = Noul(
            instructions=GATE_GESTURE,
            criteria={"false": "No deliberate speech act."})
        return qs

    raise KeyError(f"unknown variant {variant!r}")


GATE_KEYS = {
    "gate_intention": "INTENTION",
    "gate_attention": "ATTENTION",
    "gate_timbre": "TIMBRE",
    "gate_gesture": "GESTURE",
}


def variant(name: str) -> Variant:
    if name == "baseline":
        return Variant("baseline", _baseline_specs())
    if name == "reworded":
        return Variant("reworded", _reworded_specs())
    if name == "gated":
        return Variant("gated", _baseline_specs(),
                       gates=dict(GATE_KEYS),
                       note="baseline scales, explicit presence gate")
    if name == "gated_reworded":
        return Variant("gated_reworded", _reworded_specs(),
                       gates=dict(GATE_KEYS),
                       note="reworded scales AND presence gate")
    raise KeyError(f"unknown variant {name!r}")


def reduce_answers(answers, v: Variant) -> tuple[dict, dict]:
    """
    Answers -> ({track: CC}, {track: spread-confidence}).

    A gated track's CC is base_cc * gate. Multiplicative rather than a hard
    threshold: at gate 0.5 the controller gets half the value, which is a
    synthesiser fading a voice it is not sure it heard, rather than a hard mute
    that pops. It also degrades continuously, so an ambiguous real utterance is
    held back without being erased -- which is the failure under-driving would
    have caused, and the one the Phase 2 rewrite was guarding against.
    """
    ccs: dict = {}
    confs: dict = {}
    for key, kind, track, n_levels in v.specs:
        a = answers[key]
        if kind == "score":
            pos = getattr(a, "score", None)
            probs = getattr(a, "probabilities", None) or {}
            if pos is None:
                pos = (max(probs.values()) if probs else 0.0)
            base = judged._scale_position(pos, n_levels)
        else:
            probs = getattr(a, "probabilities", None) or {}
            p = probs.get(a.choice)
            base = judged._scale(p) if p is not None else 0
        confs[track] = float(getattr(a, "confidence", 0.0) or 0.0)
        ccs[track] = base

    gates: dict = {}
    if v.gates:
        for gkey, track in v.gates.items():
            g = getattr(answers[gkey], "noul", None)
            if g is None:
                continue
            g = float(g)
            gates[track] = g
            ccs[track] = ccs.get(track, 0.0) * g
    return ccs, confs


def measure(client, texts, v: Variant, *, model=None, timeout: float = 8.0):
    """
    One call per text. Returns list of {text, cc, conf, gate, ok}.

    `ok` is per text, not per track: a variant whose question dict raised would
    otherwise contribute zeros to a median and read as "the model is
    uncertain", which is a measurement artefact dressed as a result.
    """
    model = model or judged.DEFAULT_MODEL
    qs = build_questions(v.name)
    rows = []
    for text in texts:
        try:
            resp = client.system_one(state=text, questions=qs,
                                     model=model, timeout=timeout)
        except Exception as exc:                      # noqa: BLE001
            rows.append({"text": text, "cc": {}, "conf": {}, "gate": {},
                         "ok": False, "why": type(exc).__name__})
            continue
        try:
            ccs, confs = reduce_answers(resp.answers, v)
        except (KeyError, AttributeError) as exc:
            rows.append({"text": text, "cc": {}, "conf": {}, "gate": {},
                         "ok": False, "why": f"reduce:{type(exc).__name__}"})
            continue
        gate = {v.gates[k]: float(getattr(resp.answers[k], "noul", 0.0))
                for k in v.gates if k in resp.answers}
        rows.append({"text": text, "cc": ccs, "conf": confs, "gate": gate,
                     "ok": True,
                     "model": str(getattr(resp, "model", model))})
    return rows


def _median(rows, field_name, track):
    vals = [r[field_name][track] for r in rows
            if r["ok"] and track in r[field_name]]
    return statistics.median(vals) if vals else None


@dataclass
class TrackResult:
    track: str
    cc_garbage: float | None
    cc_real: float | None
    conf_garbage: float | None
    conf_real: float | None
    gate_garbage: float | None
    gate_real: float | None

    @property
    def cc_separation(self):
        if self.cc_garbage is None or self.cc_real is None:
            return None
        return self.cc_real - self.cc_garbage

    @property
    def conf_separation(self):
        if self.conf_garbage is None or self.conf_real is None:
            return None
        return self.conf_real - self.conf_garbage

    @property
    def gate_separation(self):
        if self.gate_garbage is None or self.gate_real is None:
            return None
        return self.gate_real - self.gate_garbage


def score_variant(v: Variant, garbage_rows, real_rows) -> list[TrackResult]:
    tracks = sorted({t for _, _, t, _ in v.specs})
    out = []
    for t in tracks:
        out.append(TrackResult(
            track=t,
            cc_garbage=_median(garbage_rows, "cc", t),
            cc_real=_median(real_rows, "cc", t),
            conf_garbage=_median(garbage_rows, "conf", t),
            conf_real=_median(real_rows, "conf", t),
            gate_garbage=_median(garbage_rows, "gate", t),
            gate_real=_median(real_rows, "gate", t),
        ))
    return out


# CC_FULL is the top of the 7-bit MIDI range. Separations are judged as a
# FRACTION OF THE FULL RANGE, because "separates by 1.0 CC" is not a real
# separation -- it is 0.8% of the instrument's travel, indistinguishable from
# noise. The first version of this function compared raw CC and called a
# 1.0-unit separation "healthy"; it was reading noise as a result.
CC_FULL = 127.0

GATE_HEALTHY_BAR = 0.40     # gate separation; garbage < 0.30, real > 0.70
CC_HEALTHY_BAR = 0.30       # 0.30 * 127 = 38 CC of travel between the batteries


def verdict(r: TrackResult) -> str:
    """
    HEALTHY / ILL-POSED / WEAK, judged on the quantity the synthesiser reads.

    For a gated variant the GATE carries the separation, because that is what
    suppresses garbage: a gated track whose gate says "no" returns CC 0 by
    construction, whatever the scale underneath happened to choose. For an
    ungated variant there is no absence signal at all, so the only evidence is
    whether the scale itself separates.

    question_health.py judges on spread-CONFIDENCE and reached different
    verdicts on the same data (Phase 2.6: INTENTION ill-posed). Both are
    recorded; the CC is the one a synthesiser plays.
    """
    if r.cc_separation is None:
        return "unmeasured"
    if r.gate_separation is not None:
        gs = r.gate_separation
        if gs >= GATE_HEALTHY_BAR:
            return "healthy"
        if gs < 0.0:
            return "ill-posed"
        return "weak"
    norm = r.cc_separation / CC_FULL
    if norm < 0.0:
        return "ill-posed"
    if norm >= CC_HEALTHY_BAR:
        return "healthy"
    return "weak"


def format_report(results: dict) -> str:
    lines = ["# Question rewrite probe", ""]
    for name, res in results.items():
        v = variant(name)
        lines.append(f"## {name}" + (f" — {v.note}" if v.note else ""))
        lines.append("")
        lines.append("| Track | CC garbage | CC real | CC sep | conf garbage | "
                     "conf real | conf sep | gate garbage | gate real | gate sep | verdict |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

        def f(x, pct=False):
            if x is None:
                return "-"
            return f"{x*100:.0f}%" if pct else f"{x:.3f}"

        for r in res:
            lines.append(
                f"| {r.track} | {f(r.cc_garbage)} | {f(r.cc_real)} | "
                f"{f(r.cc_separation)} | {f(r.conf_garbage)} | "
                f"{f(r.conf_real)} | {f(r.conf_separation)} | "
                f"{f(r.gate_garbage, True)} | {f(r.gate_real, True)} | "
                f"{f(r.gate_separation)} | {verdict(r)} |")
        lines.append("")
    return "\n".join(lines)