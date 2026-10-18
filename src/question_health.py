"""
question_health — an ill-posed question still returns confident numbers.

decision-ledger's pass_detector exists because the 0.5-means-pass principle
caught ill-posed questions that a threshold could not: a question whose
answers cluster at "pass" is INVALID, not uncertain. The defect class it
catches there is a question the model declines to answer.

The defect class HERE is the mirror image, and it is worse for a controller:
a track question asked of SEMANTIC GARBAGE should return LOW spread-
confidence ("I cannot place this"), because garbage has no timbre, no
urgency, no addressee and no speech act. A question that confidently places
garbage on its scale is inventing structure -- and the same invention is
then applied to every ambiguous real utterance, which is exactly where a
controller must hold back (see judged.py: under-driving ambiguity is the
failure confidence-as-CC would have caused).

THE INSTRUMENT. Two batteries per question set:

  * garbage: strings engineered to carry none of the four properties.
  * real: plain utterances engineered to carry them.

A question is healthy when it separates: confident on real, unconfident on
garbage. EITHER collapse is a finding --

  * confident-on-garbage: the question invents structure (ill-posed).
  * unconfident-on-real: the question cannot see the property at all
    (degenerate -- and per the gate law this repo inherits, a question that
    does not separate may not steer).

Reported per track: question fingerprint (decision-ledger's question_hash,
via judged.question_fingerprint), median spread-confidence on each battery,
the separation gap, and the verdict. The fingerprint makes a rewrite
demonstrable: new wording, new hash, new numbers, both kept.

WHAT THIS DOES NOT MEASURE: whether the REAL utterances are placed CORRECTLY.
That is calibration's job (calibration.py) and it needs ground truth by
construction. Health asks only whether the question can tell something from
nothing.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from . import judged

# Strings engineered to carry no timbre, no urgency, no addressee, no speech
# act. Not "weird sentences" -- NON-sentences: no interrogative structure, no
# imperative mood, no addressee, no affect.
GARBAGE_PROBES = [
    "qwop zzz brxt",
    "vreeble the norvath",
    "zzz",
    "kxx ttt mmm",
    "blorp",
]

# Plain utterances that unambiguously carry the properties all four questions
# ask about (they deliberately overlap with the calibration corpus).
REAL_PROBES = [
    "Could you please pull up the last commit?",
    "Run the tests.",
    "Oh wow, it actually worked.",
    "The sky is clear today.",
    "Why is the build red again?",
]

HEALTH_BAR = 0.70        # median spread-confidence above this on garbage = invented structure
DEGENERATE_BAR = 0.50    # median below this on real = the question sees nothing


@dataclass
class TrackHealth:
    track: str
    fingerprint: str
    n_garbage: int
    n_real: int
    garbage_median: float
    real_median: float
    verdict: str          # "healthy" | "ill-posed" | "degenerate"

    @property
    def separation(self) -> float:
        return self.real_median - self.garbage_median


def _conf(reading) -> float:
    c = reading.detail.get("confidence")
    return float(c) if c is not None else 0.0


def _median_conf(readings_by_text, track: str) -> float:
    """
    Median spread-confidence for one track across a battery.

    `readings_by_text` is a list of {track: TrackReading} dicts -- one dict per
    probe. Two bugs lived here and both are the kind a caller never notices:

      * `all(r.ok for r in rs)` iterated the DICT, so `r` was the track NAME
        and every entry raised AttributeError: 'str' object has no attribute
        'ok'. The module had never been run; it had only been written.
      * `rs[track]` raises KeyError outright if one call dropped a track, which
        turns a partial network failure into a crash instead of a skip.

    So: iterate `.values()`, and skip a probe that is missing the track rather
    than pretending it measured zero.
    """
    vals = [_conf(rs[track]) for rs in readings_by_text
            if track in rs and all(r.ok for r in rs.values())]
    return statistics.median(vals) if vals else 0.0


def probe_questions(client=None, *, model: str = judged.DEFAULT_MODEL,
                    garbage=None, real=None, timeout: float = 8.0) -> list[TrackHealth]:
    """
    Ask all four questions of both batteries and score each track's health.

    One model call per probe string (all four questions ride in it -- the
    one-call rule is architectural, not a saving). n calls = len(garbage) +
    len(real), reported by the caller, because call counts are a first-class
    output.
    """
    garbage = garbage if garbage is not None else GARBAGE_PROBES
    real = real if real is not None else REAL_PROBES

    by_batt = {}
    for name, battery in (("garbage", garbage), ("real", real)):
        rows = []
        for text in battery:
            readings, _ = judged.read_tracks(text, model=model, client=client,
                                             timeout=timeout)
            by_track = {r.track: r for r in readings}
            rows.append(by_track)
        by_batt[name] = rows

    out = []
    for track in judged.CC:
        g = _median_conf(by_batt["garbage"], track)
        r = _median_conf(by_batt["real"], track)
        if g >= HEALTH_BAR:
            verdict = "ill-posed"
        elif r < DEGENERATE_BAR:
            verdict = "degenerate"
        else:
            verdict = "healthy"
        out.append(TrackHealth(track=track,
                               fingerprint=judged.question_fingerprint(),
                               n_garbage=len(garbage), n_real=len(real),
                               garbage_median=round(g, 3),
                               real_median=round(r, 3),
                               verdict=verdict))
    return out


def format_report(healths: list[TrackHealth], *, model: str) -> str:
    lines = [
        "# Question health",
        "",
        f"Model: `{model}` · fingerprint `{healths[0].fingerprint}` · "
        f"spread-confidence is the SDK's `confidence` (distribution shape), "
        f"NOT the CC value.",
        "",
        "| Track | garbage med | real med | separation | verdict |",
        "|---|---|---|---|---|",
    ]
    for h in healths:
        lines.append(f"| {h.track} | {h.garbage_median:.2f} | {h.real_median:.2f} "
                     f"| {h.separation:+.2f} | {h.verdict} |")
    lines += [
        "",
        f"Bars: ill-posed at garbage median ≥ {HEALTH_BAR}; degenerate at "
        f"real median < {DEGENERATE_BAR}.",
        "",
    ]
    return "\n".join(lines)
