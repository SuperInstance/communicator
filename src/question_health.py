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
PARK_BAR = 0.10          # Score tracks: garbage median position at/below this
                         # (in level units) drives CC ~0 -- parked, not steering


@dataclass
class TrackHealth:
    track: str
    fingerprint: str
    n_garbage: int
    n_real: int
    garbage_median: float
    real_median: float
    verdict: str          # "healthy" | "ill-posed" | "degenerate" | "safe-parked"
    # For Score tracks: median POSITION on garbage and real. A confident
    # answer on garbage is only damage if it drives the controller somewhere:
    # garbage that PARKS at position 0 reads CC 0 however confident the model
    # claims to be. Confidence says "I am sure"; position says where the CC
    # goes. For a controller the second question is the one that matters.
    # None for Choice tracks (their CC is p(chosen), always a "somewhere").
    garbage_pos: float | None = None
    real_pos: float | None = None

    @property
    def separation(self) -> float:
        return self.real_median - self.garbage_median


def _conf(reading) -> float:
    c = reading.detail.get("confidence")
    return float(c) if c is not None else 0.0


def _pos(reading) -> float | None:
    s = reading.detail.get("score")
    return float(s) if s is not None else None


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


def _median_pos(readings_by_text, track: str) -> float | None:
    """Median Score position for one track; None when nothing measurable."""
    vals = [p for rs in readings_by_text
            if track in rs and all(r.ok for r in rs.values())
            if (p := _pos(rs[track])) is not None]
    return statistics.median(vals) if vals else None


def _round2(x):
    """Round for display, preserving None as None rather than exploding."""
    return None if x is None else round(x, 2)


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
        is_score = track in ("TIMBRE", "INTENTION")
        # _median_pos returns None when nothing measurable came back -- a
        # dropped probe, or a reading with no `score`. Rounding it unguarded
        # raised TypeError: type NoneType doesn't define __round__ method,
        # which is why the verdict branches below check `gp is not None` while
        # these two lines did not. A helper that can answer "nothing" must be
        # rounded by a caller that can hear it.
        gp = _round2(_median_pos(by_batt["garbage"], track)) if is_score else None
        rp = _round2(_median_pos(by_batt["real"], track)) if is_score else None
        # Verdict order matters. For a Score track the controller-safety
        # question comes FIRST: garbage that parks low is harmless however
        # confidently it was placed, so "safe-parked" outranks "ill-posed".
        # A Choice track has no parked position -- p(chosen) always drives
        # SOMEWHERE -- so confident-on-garbage stays ill-posed.
        if g >= HEALTH_BAR and not (is_score and gp is not None and gp <= PARK_BAR):
            verdict = "ill-posed"
        elif r < DEGENERATE_BAR:
            verdict = "degenerate"
        elif is_score and gp is not None and gp <= PARK_BAR and g >= HEALTH_BAR:
            verdict = "safe-parked"
        else:
            verdict = "healthy"
        out.append(TrackHealth(track=track,
                               fingerprint=judged.question_fingerprint(),
                               n_garbage=len(garbage), n_real=len(real),
                               garbage_median=round(g, 3),
                               real_median=round(r, 3),
                               verdict=verdict,
                               garbage_pos=gp, real_pos=rp))
    return out


def format_report(healths: list[TrackHealth], *, model: str) -> str:
    lines = [
        "# Question health",
        "",
        f"Model: `{model}` · fingerprint `{healths[0].fingerprint}` · "
        f"spread-confidence is the SDK's `confidence` (distribution shape), "
        f"NOT the CC value.",
        "",
        "| Track | garbage med | real med | separation | verdict | garbage pos |",
        "|---|---|---|---|---|---|",
    ]
    for h in healths:
        gp = f"{h.garbage_pos:.2f}" if h.garbage_pos is not None else "-"
        lines.append(f"| {h.track} | {h.garbage_median:.2f} | {h.real_median:.2f} "
                     f"| {h.separation:+.2f} | {h.verdict} | {gp} |")
    lines += [
        "",
        f"Bars: ill-posed at garbage median ≥ {HEALTH_BAR}; degenerate at "
        f"real median < {DEGENERATE_BAR}. For Score tracks, `garbage pos` is "
        f"the controller-safety column: garbage parking at position ≤ "
        f"{PARK_BAR} drives CC ~0 however confident the model claims to be, "
        f"and the verdict says `safe-parked` instead of `ill-posed`.",
        "",
    ]
    return "\n".join(lines)
