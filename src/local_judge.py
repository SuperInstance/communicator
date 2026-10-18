"""
local_judge -- the same questions, asked of a model on this machine.

WHY THIS EXISTS, and it is not "to avoid the cloud bill".

Two reasons, one practical and one about evidence:

1. The design says cloud / local / hybrid must be CONFIGURABLE, because the
   target device is an UNO Q with 6 GB of RAM and no GPU. Every judged track
   so far has been 100% cloud, so that requirement has never been tested even
   once. A requirement that has never been exercised is a wish.

2. A second model is an INDEPENDENT INSTRUMENT on question health. Jev said
   garbage was urgent (Phase 2.6). If a local model, given the same questions,
   also says garbage is urgent, the defect is in the QUESTION. If the local
   model separates cleanly where Jev did not, then what was measured was
   partly a property of one model rather than a property of the question.
   Nothing else in the repo can tell those two cases apart.

WHAT THIS IS NOT: a claim that a 7B local model matches a hosted one. It is
unlikely to, and the numbers are expected to be worse. The value is the
COMPARISON, measured on the same probes with the same reduction.

THE REDUCTION IS DELIBERATELY SHARED. Both paths call judged._scale_position
and judged._scale, so a difference between the local and cloud numbers is a
difference in the ANSWERS and never an accident of two implementations of
"how do I turn a score into a CC".

A LOCAL MODEL IS NOT TRAINED TO EMIT PROBABILITIES. It is asked for them in a
JSON schema and will produce a number that looks like a probability. Those
numbers are expected to be badly calibrated, and the calibration instrument
(calib.py) exists to say so rather than letting them look authoritative.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import judged

DEFAULT_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("LOCAL_LLM_MODEL", "qwen2.5-coder:7b")


class LocalJudgeUnavailable(RuntimeError):
    """The local model could not be reached or refused the request."""


# -- the schema the local model must fill -----------------------------------
#
# Built from the SAME scale constants the cloud path uses, so the two cannot
# drift apart by editing one file. If judged.INTENTION_SCALE gains a level,
# this schema gains a position to report and vice versa.

def build_schema() -> dict:
    score_specs = {"timbre": len(judged.TIMBRE_SCALE),
                   "intention": len(judged.INTENTION_SCALE)}
    choice_specs = {"attention": sorted(judged.ATTENTION_CHOICE),
                    "gesture": sorted(judged.GESTURE_CHOICE)}
    props: dict = {}
    for key, n in score_specs.items():
        props[key] = {
            "type": "object",
            "properties": {
                # POSITION on 0..(n-1), exactly what ScoreAnswer.score is, so
                # the shared reduction applies unchanged.
                "position": {"type": "number", "minimum": 0,
                             "maximum": n - 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["position", "confidence"],
        }
    for key, options in choice_specs.items():
        props[key] = {
            "type": "object",
            "properties": {
                "probabilities": {
                    "type": "object",
                    "properties": {o: {"type": "number", "minimum": 0,
                                       "maximum": 1} for o in options},
                    "required": options,
                },
            },
            "required": ["probabilities"],
        }
    for key in judged.GATES:
        props[key] = {"type": "number", "minimum": 0, "maximum": 1}
    return {
        "type": "object",
        "properties": props,
        "required": sorted(props),
    }


SYSTEM = (
    "You are a signal analyst for a voice-controlled instrument. For the "
    "utterance given, answer only these questions about it.\n\n"
    "For `timbre` and `intention` report `position`: a number from 0 to the "
    "highest index of the scale, as a decimal (0.0 is the first level, 1.0 "
    "the last). For `attention` and `gesture` report a probability for EVERY "
    "listed option, which must sum to about 1. For each `gate_*` report the "
    "probability that the property is genuinely present -- if the text is "
    "gibberish, noise or has no such property, report a LOW number near 0.\n\n"
    "Be honest about uncertainty. Do not invent structure that is not there."
)


def build_prompt(text: str) -> str:
    scales = {
        "timbre": judged.TIMBRE_SCALE,
        "intention": judged.INTENTION_SCALE,
        "attention": judged.ATTENTION_CHOICE,
        "gesture": judged.GESTURE_CHOICE,
    }
    parts = []
    for key, spec in scales.items():
        if isinstance(spec, dict):
            body = "\n".join(f"  - {k}: {v}" for k, v in spec.items())
        else:
            n = len(spec)
            body = "\n".join(f"  - {i}: {c}" for i, c in enumerate(spec))
            body += f"\n  (report position 0.0 to {n - 1}.0)"
        parts.append(f"{key}:\n{body}")
    gates = "\n".join(f"  - {k}" for k in judged.GATES)
    return (f"Utterance: {text}\n\nScales:\n" + "\n".join(parts)
            + f"\n\nPresence gates:\n{gates}\n\nAnswer as JSON.")


@dataclass
class LocalReading:
    ccs: dict = field(default_factory=dict)
    confs: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    ok: bool = True
    why: str = ""


def reduce_local(obj: dict) -> LocalReading:
    """
    Local JSON -> the same {track: CC} shape the cloud path produces.

    Any missing or malformed piece yields ok=False for the WHOLE reading. A
    partial reduction would put a real-looking number on half the tracks and
    zero on the rest, and a synthesiser cannot tell that from a genuine
    silence.
    """
    out = LocalReading(raw=obj)
    try:
        for key in ("timbre", "intention"):
            n = len(judged.TIMBRE_SCALE if key == "timbre"
                    else judged.INTENTION_SCALE)
            pos = float(obj[key]["position"])
            out.ccs[key.upper()] = judged._scale_position(pos, n)
            out.confs[key.upper()] = float(obj[key].get("confidence", 0.0))
        for key in ("attention", "gesture"):
            probs = {k: float(v) for k, v in obj[key]["probabilities"].items()}
            track = key.upper()
            chosen = max(probs, key=probs.get)
            # The CC is the winning option's probability, matching judged.py.
            out.ccs[track] = judged._scale(probs[chosen])
            out.confs[track] = 0.0
            out.raw.setdefault("choices", {})[track] = chosen
        for key, track in judged.GATES.items():
            g = float(obj[key])
            out.gates[track] = g
            # Multiplicative, exactly like the cloud path's Noul gate.
            out.ccs[track] = int(round(out.ccs[track] * g))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        out.ok = False
        out.why = f"reduce:{type(exc).__name__}"
    return out


class LocalJudge:
    """
    A client shaped like the SDK's, so callers can swap cloud for local by
    passing a different object rather than branching at every call site.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status == 200
        except Exception:                             # noqa: BLE001
            return False

    def judge(self, text: str) -> LocalReading:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(text)},
            ],
            "stream": False,
            "format": build_schema(),       # structured output: no prose to parse
            "options": {"temperature": 0},  # a controller must not jitter
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            return LocalReading(ok=False, why=f"http{exc.code}")
        except Exception as exc:                      # noqa: BLE001
            return LocalReading(ok=False, why=type(exc).__name__)

        if "message" not in body:
            # This is where a decision-only model lands: tev1 refuses both
            # /api/generate and /api/chat with "does not support", so the
            # error must be reported rather than silently read as silence.
            return LocalReading(ok=False,
                                why=str(body.get("error", "no message"))[:120])
        try:
            obj = json.loads(body["message"]["content"])
        except (json.JSONDecodeError, KeyError) as exc:
            return LocalReading(ok=False, why=f"json:{type(exc).__name__}")
        return reduce_local(obj)