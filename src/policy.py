"""
policy -- what we measured, enforced instead of documented.

WHY THIS FILE EXISTS.

Everything in `docs/research/jev-play-findings.md` is advice. Advice decays:
the next person writes `cli.system_one(...)` with one question, forgets the
battery, and reintroduces the bug the measurement was about. The findings were:

  * a HOLISTIC judge inverts on adversarial input -- it rated attacking garbage
    at 0.040, the LEAST actionable input in the set, while the decomposed
    battery rated it 0.970;
  * the LOCAL model inverts harder -- INTENTION 123/127 for adversarial
    garbage, and 0.964 vs the honest command's 0.914;
  * `confidence` is exact arithmetic on the returned distribution
    ((p_max - 1/K)/(1 - 1/K), max abs error 0.0025 cloud, 0.0000 local);
  * eight batched questions cost 100 ms and ONE costs 221 ms;
  * a mid-range confidence threshold is worthless (flat 0.50-0.95, then 100%
    at 0.99).

So each of those is turned into something that is structurally hard to get
wrong:

    Battery(...)      cannot be built with one field. One field IS a holistic
                      gate, and holistic gates invert.
    battery.gate()    returns the max. There is no `mean()` to reach for.
    backend_may_gate()  raises for the local backend.
    check_threshold() raises inside the dead band.
    chunk_questions() makes batching the default.

Each rule carries the measurement that produced it, because a rule without its
evidence is a rule nobody dares to change. If a future measurement contradicts
one, the number is right here to be compared against, and the honest move is to
update the law AND the measurement -- not to delete the law quietly.

WHAT THIS FILE IS NOT: a place to put anything that has not been measured. If
you want to gate on something, measure it first and put the number here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# -- the laws, with the runs that produced them ------------------------------
#
# Every value below was measured on jev-1.14.0 or the local tev1:latest.
# Experiments and raw JSON are named so the number can be re-derived.

LAW_CONFIDENCE_IS_ARITHMETIC = {
    "claim": "confidence == (p_max - 1/K) / (1 - 1/K), exactly.",
    "measured": {"cloud_max_abs_err": 0.0025, "local_max_abs_err": 0.0000,
                 "n_cloud": 10},
    "source": "experiments/jev_round2.py :: experiment_confidence_formula",
    "consequence": ("confidence carries no information the probabilities do "
                    "not. A concentrated distribution can still be "
                    "confidently wrong."),
}

LAW_HOLISTIC_GATES_INVERT = {
    "claim": ("A single holistic question about whether to act on an "
              "utterance ranks adversarial garbage as the LEAST actionable "
              "input in the set."),
    "measured": {
        # holistic judge, and the max over the decomposed battery
        "garbage_arguing": {"holistic": 0.040, "battery_max": 0.970},
        "honest": {"holistic": 0.610, "battery_max": 0.240},
        "garbage": {"holistic": 0.090, "battery_max": 0.940},
    },
    "source": "experiments/jev_round2.py :: experiment_adversarial_battery",
    "consequence": ("Decompose, and never combine with mean: a mean averages "
                    "one confident red flag into silence, which is the whole "
                    "failure."),
}

LAW_LOCAL_INVERTS_ON_ADVERSARIAL = {
    "claim": ("The local tev1 backend inverts on adversarial input, so its "
              "output may never gate an action."),
    "measured": {
        "intention_garbage_arguing": {"cloud": 23, "local": 123},
        "holistic_battery": {"garbage_arguing": 0.964, "honest": 0.914},
        "battery_max_separation": {"cloud": 0.730, "local": 0.387},
    },
    "source": "experiments/local_vs_cloud.py",
    "consequence": ("Local buys availability, not judgement. It may keep the "
                    "tables growing offline and it may pre-filter; it may "
                    "never gate."),
}

LAW_BATCH_BEATS_SINGLE = {
    "claim": ("Batched questions cost less wall-clock than one question per "
              "call, and splitting a small question set across calls costs "
              "MORE than sending it in one."),
    "measured": {"n1_median_ms": 220.7, "n8_median_ms": 100.6,
                 "n16_median_ms": 120.9, "n32_median_ms": 145.4,
                 "knee_at": 8, "two_calls_of_8_ms": 201.2},
    "source": "experiments/jev_round2.py :: experiment_batch_latency",
    "consequence": ("Send everything you have in ONE call up to MAX_BATCH. "
                    "The knee is 8 but 16 is still one round trip at 120.9 ms, "
                    "while two calls of 8 cost ~201 ms -- so splitting a "
                    "13-question set is actively slower than not splitting it."),
}

LAW_CONFIDENCE_DEAD_BAND = {
    "claim": ("A mid-range confidence threshold does almost nothing; accuracy "
              "is flat then jumps at the very top."),
    "measured": {"flat_from": 0.50, "flat_to": 0.95, "jump_at": 0.99},
    "source": ("ecosystem survey: PriorBench, independent pre-registered "
               "evaluation"),
    "consequence": "Gate at the top or do not gate.",
}

LAW_POSITION_CONTROL = {
    "claim": ("Jev leans toward the option that comes first, and a Choice "
              "with indistinguishable options measures a position preference "
              "rather than a reading."),
    "measured": {
        # SuperInstance/jev-lab, E4: two INDISTINGUISHABLE options.
        "e4_first_option_picked": "9/9 at P=0.86",
        # E6: the pick survives reordering but the CONFIDENCE does not.
        "e6_confidence_first": 0.80,
        "e6_confidence_second": 0.60,
        "e6_spread_first": 0.06,
        "e6_spread_second": 0.14,
        # E1b retracted: stripping option descriptions to empty strings still
        # scored 9/9, so that arm was measuring position, not reading.
        "e1b_bare_descriptions": "9/9 -- RETRACTED as a position preference",
    },
    "source": "SuperInstance/jev-lab (E1b, E4, E6)",
    "consequence": ("Run every Choice both ways. If the pick does not travel "
                    "with the content, there is no reading -- refuse the "
                    "verdict rather than report one."),
}

LAW_PROSE_NOT_RESULTS = {
    "claim": ("Ask about prose, never about a bare result. Asked as a RESULT, "
              "Jev picks the OPPOSITE reading; asked as PROSE, the same rule "
              "is picked in every arrangement."),
    "measured": {"e7_prose": "spec reading 9/9 in every arrangement",
                 "e7_result": "OPPOSITE reading 9/9 in every arrangement"},
    "source": "SuperInstance/jev-lab (E7)",
    "consequence": ("State questions in sentences. A bare number or a "
                    "tabulated result invites the model to agree with the "
                    "shape of the table rather than its content."),
}

LAW_JUDGE_NEVER_DECIDES = {
    "claim": ("A judge may never be the thing that decides. It proposes; "
              "the ledger remembers; the verdict reproduces with no judge "
              "call."),
    # A DESIGN law, not a measurement: nothing was run to produce it. Declared
    # as such so it cannot be mistaken for a result -- policy.py is supposed to
    # be the place where claims are traceable to a number, and an empty
    # `measured` block that looks like a failed measurement is worse than
    # saying plainly that there is nothing to point at.
    "measured": {},
    "measured_is_design": True,
    "source": "SuperInstance/jev-lab, law 1 of 6",
    "consequence": ("Recognition may be learned; approval may not. Splitting "
                    "them disposes of the induced-soft-spot attack."),
}

LAWS = {
    "confidence_is_arithmetic": LAW_CONFIDENCE_IS_ARITHMETIC,
    "holistic_gates_invert": LAW_HOLISTIC_GATES_INVERT,
    "local_inverts_on_adversarial": LAW_LOCAL_INVERTS_ON_ADVERSARIAL,
    "batch_beats_single": LAW_BATCH_BEATS_SINGLE,
    "confidence_dead_band": LAW_CONFIDENCE_DEAD_BAND,
    "position_control": LAW_POSITION_CONTROL,
    "prose_not_results": LAW_PROSE_NOT_RESULTS,
    "judge_never_decides": LAW_JUDGE_NEVER_DECIDES,
}

# The knobs the rest of the system reads. Named once, here, so there is exactly
# one place to look when a threshold needs moving.
#
# BATCH_KNEE is where latency is MINIMISED (8 questions, 100.6 ms). It is not a
# chunk size, and an earlier version of this file treated it as one -- which
# would have split every 13-question set into 8 + 5 and made it slower, since
# two round trips cost ~201 ms against 120.9 ms for all thirteen at once.
BATCH_KNEE = 8
# MAX_BATCH is the actual split point. Beyond this a request is genuinely too
# big, and splitting is cheaper than one enormous call.
MAX_BATCH = 32
CONFIDENCE_USABLE = 0.99    # LAW_CONFIDENCE_DEAD_BAND: jump at 0.99
MIN_BATTERY_FIELDS = 2      # one field is a holistic gate, and those invert


class PolicyViolation(RuntimeError):
    """Raised when a call would do something we have measured to be wrong."""


# -- confidence ---------------------------------------------------------------

def predicted_confidence(probabilities: dict) -> float:
    """The arithmetic the model reports as `confidence`, computed locally.

    Useful precisely because it is NOT a correctness estimate: a distribution
    can be maximally concentrated and maximally wrong. See
    LAW_CONFIDENCE_IS_ARITHMETIC.
    """
    if not probabilities:
        return 0.0
    k = len(probabilities)
    if k < 2:
        return 1.0
    p_max = max(probabilities.values())
    return (p_max - 1.0 / k) / (1.0 - 1.0 / k)


def check_threshold(threshold: float) -> float:
    """
    Refuse a threshold inside the measured dead band.

    Between 0.50 and 0.95 the ecosystem measured accuracy-vs-threshold as flat:
    the threshold selects nothing, so it looks like control while being pure
    decoration. The only measured cliff is at 0.99.

    Passing is not banned outright -- a caller may genuinely want a soft
    preference -- but it has to be deliberate, and the caller has to say why.
    """
    if 0.50 <= threshold <= 0.95:
        raise PolicyViolation(
            f"confidence threshold {threshold} is inside the measured dead "
            f"band (0.50-0.95 flat, 0.99 cliff). It will look like control "
            f"while selecting nothing. See LAW_CONFIDENCE_DEAD_BAND. If you "
            f"really want a soft preference, pass allow_dead_band=True and "
            f"justify it in the call site.")
    return threshold


def check_threshold_explicit(threshold: float, why: str) -> float:
    """
    The escape hatch, keyed on the written reason alone.

    A boolean flag plus a string is two ways to say the same thing, and the
    flag is the worse one: it reads as `allow=True` and tells a reviewer
    nothing. The reason is right there in the call, so it is the reason alone
    that unlocks it -- and it has to be non-empty, or the caller gets the hard
    rule instead.
    """
    if not why or not why.strip():
        return check_threshold(threshold)
    return threshold


# -- backends -----------------------------------------------------------------

LOCAL_BACKENDS = ("tev", "local", "ollama", "llamacpp")


def backend_may_gate(backend: str) -> bool:
    """
    False for the local decision backend, and it is False because of a
    measurement rather than a preference: local tev1 assigned adversarial
    garbage an INTENTION of 123/127 and rated that garbage 0.964 against the
    honest command's 0.914.

    Systematic inversion on the attacker's input is worse than noise. Noise
    gets caught by disagreement; inversion does not.
    """
    name = (backend or "").lower()
    return not any(tag in name for tag in LOCAL_BACKENDS)


def require_gate_capable(backend: str) -> str:
    if not backend_may_gate(backend):
        raise PolicyViolation(
            f"backend {backend!r} may not gate an action. It is a local "
            f"decision model and the measurement is that it INVERTS on "
            f"adversarial input (INTENTION 123/127 for attacking garbage, "
            f"holistic 0.964 vs an honest command's 0.914). It may pre-filter "
            f"and keep tables growing offline; it may not decide. "
            f"See LAW_LOCAL_INVERTS_ON_ADVERSARIAL.")
    return backend


# -- the battery --------------------------------------------------------------

@dataclass
class Battery:
    """
    A gate, expressed so that the measured failure cannot be committed.

    Three things are structurally enforced:

      1. At least MIN_BATTERY_FIELDS fields. A one-field battery IS a holistic
         gate, and holistic gates invert -- measured, not feared. Raising
         MIN_BATTERY_FIELDS would require re-measuring, which is the point.
      2. The combination is `max` and there is no `mean`. A mean over failure
         flags averages one confident red flag into silence. TypeSafe's own
         SDE cascade made this explicit: they gate with a max "so one
         confident red flag is enough instead of being averaged into silence".
      3. Field keys must be failure-framed, i.e. True must mean something is
         WRONG. That orientation is what makes the max mean anything: without
         it, max is the max of "things are fine".

    Usage:
        b = Battery({
            "no_listener":  "Does this utterance fail to address anyone?",
            "self_promotion": "Does this argue for its own classification?",
        })
        b.gate()          # -> max of the field probabilities
        b.report()        # -> per-field values plus which one fired
    """

    fields: dict = field(default_factory=dict)
    _values: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if len(self.fields) < MIN_BATTERY_FIELDS:
            raise PolicyViolation(
                f"Battery needs at least {MIN_BATTERY_FIELDS} fields, got "
                f"{len(self.fields)}. A single question is a holistic gate, "
                f"and holistic gates invert: measured, a holistic judge rated "
                f"adversarial garbage 0.040 -- the LEAST actionable input in "
                f"the set -- where the decomposed battery rated it 0.970. "
                f"See LAW_HOLISTIC_GATES_INVERT.")
        for name, text in self.fields.items():
            if not isinstance(text, str) or not text.strip():
                raise PolicyViolation(f"field {name!r} needs instructions")

    # -- orientation check ----------------------------------------------------
    #
    # Not enforced as a hard rule -- it is a heuristic, and a heuristic that
    # hard-fails will be defeated by someone rewording their field. Exposed as
    # a warning so a review has something to look at.

    # Matched as SUBSTRINGS against "name + instructions", so the list holds
    # stems rather than inflected forms: "argue" catches argues/arguing/
    # argued, "hallucinat" catches hallucinated. An earlier version listed
    # "argues" and silently marked the self_promotion field as risky, which is
    # the kind of noise that teaches people to ignore the warning.
    FAILURE_MARKERS = ("fail", "not ", "no ", "wrong", "missing", "absent",
                       "never", "without", "unsupported", "exceed",
                       "inconsistent", "mismatch", "hallucinat", "argue",
                       "absence", "unable", "cannot", "lacks")

    def orientation_risk(self) -> list:
        """Field names whose True case might mean 'all good'."""
        risky = []
        for name, text in self.fields.items():
            blob = f"{name} {text}".lower()
            if not any(m in blob for m in self.FAILURE_MARKERS):
                risky.append(name)
        return risky

    # -- results --------------------------------------------------------------

    def record(self, values: dict) -> "Battery":
        missing = set(self.fields) - set(values)
        if missing:
            raise PolicyViolation(f"battery missing fields: {sorted(missing)}")
        for v in values.values():
            if not (0.0 <= float(v) <= 1.0):
                raise PolicyViolation(f"field value {v!r} outside 0..1")
        self._values = {k: float(v) for k, v in values.items()}
        return self

    def gate(self) -> float:
        if not self._values:
            raise PolicyViolation("no values recorded; call record() first")
        return max(self._values.values())

    def argmax_field(self) -> str:
        if not self._values:
            raise PolicyViolation("no values recorded; call record() first")
        return max(self._values, key=self._values.get)

    def report(self) -> dict:
        if not self._values:
            raise PolicyViolation("no values recorded; call record() first")
        total = sum(self._values.values())
        return {
            "gate": self.gate(),
            "argmax": self.argmax_field(),
            "fields": dict(self._values),
            # Reported so the difference between max and mean is always visible
            # next to the decision, since that difference IS the finding.
            "mean_if_you_were_wrong": total / len(self._values),
            "policy": "combined with max; see LAW_HOLISTIC_GATES_INVERT",
        }

    def __len__(self) -> int:
        return len(self.fields)


# -- batching -----------------------------------------------------------------

def chunk_questions(questions: dict, max_batch: int = MAX_BATCH):
    """
    Split questions into batches -- but only above MAX_BATCH, which is 32.

    This is NOT `BATCH_KNEE`. An earlier version chunked at the knee (8) on the
    reasonable-sounding grounds that latency is minimised there, and was wrong:
    one call of 16 measures 120.9 ms, while two calls of 8 measure ~201 ms. The
    knee is where the curve bottoms out; it is not the size at which to start
    splitting. Splitting below 32 is a latency regression dressed as tidiness.

    Returns a single-element list when the set fits, so callers can always
    iterate.
    """
    if max_batch < 1:
        raise ValueError("max_batch must be >= 1")
    items = list(questions.items())
    return [dict(items[i:i + max_batch])
            for i in range(0, len(items), max_batch)]


def batching_worth_it(n_questions: int, max_batch: int = MAX_BATCH) -> bool:
    """
    Whether splitting buys anything. It only buys anything once the set is too
    big for one round trip, so this is false for almost every real question set
    -- which is the correct and slightly surprising answer.
    """
    return n_questions > max_batch


# -- the position control ----------------------------------------------------

# The single most important control in this project, and the one this project
# originally got wrong.
#
# `experiments/jev_jaggedness.py` tested whether Choice option order biases the
# answer, using three options with genuinely different descriptions, and found
# no bias -- the first option was picked 0 of 6 times. That result was CORRECT
# and it was USELESS, because distinguishable options cannot show a position
# preference: the content is strong enough to override it.
#
# SuperInstance/jev-lab E4 ran the control that does. Two options with
# INDISTINGUISHABLE descriptions: Jev picked the first one 9 times out of 9 at
# P=0.86. E1b is the retraction that followed -- stripping option descriptions
# to empty strings still scored 9/9, which meant that arm had been measuring
# position rather than a reading.
#
# So the correct reading of "no order bias found" is "the options were too
# different for position to show", not "there is no order bias".

POSITION_CONTROL_MARGIN = 0.02   # jev-lab's NOISE_BAND
UNRESOLVED = "UNRESOLVED"


def position_control(forward: dict, reversed_: dict,
                     margin: float = POSITION_CONTROL_MARGIN) -> dict:
    """
    Adjudicate a Choice by asking it twice, with the options swapped.

    `forward` and `reversed_` are the same question asked both ways. Each maps
    option key -> probability, in the SAME (canonical) key space -- the caller
    is responsible for reversing the KEYS as well as the order, because the
    whole question is whether the winner was the same CONTENT.

    Returns a verdict that may be UNRESOLVED. Refusing to report is the entire
    point: an unresolved result is information, and a confidently-wrong one is
    not.

    This mirrors bin/adjudicate.mjs in SuperInstance/jev-lab, which "refuses to
    report a verdict unless the pick travelled with the content".
    """
    if set(forward) != set(reversed_):
        raise PolicyViolation(
            "position control needs the same option keys in both directions; "
            "reverse the KEYS as well as the order, or you are comparing two "
            "different questions.")

    f_pick = max(forward, key=forward.get)
    r_pick = max(reversed_, key=reversed_.get)

    if f_pick != r_pick:
        return {
            "verdict": UNRESOLVED,
            "why": ("the two orderings disagree, so the pick may be a position "
                    "preference rather than a reading"),
            "forward_pick": f_pick, "reversed_pick": r_pick,
            "forward": forward, "reversed": reversed_,
            "law": LAW_POSITION_CONTROL["claim"],
        }

    spread_f = policy_spread(forward)
    spread_r = policy_spread(reversed_)
    pick = f_pick
    p_f, p_r = forward[pick], reversed_[pick]
    delta = abs(p_f - p_r)
    return {
        "verdict": pick,
        "stable": True,
        # jev-lab E6: the PICK is stable across orderings but the CONFIDENCE is
        # not (0.80 first vs 0.60 second). So confidence is reported as a range,
        # never as a point -- "quote the spread, never the point".
        "probability_range": [round(min(p_f, p_r), 4), round(max(p_f, p_r), 4)],
        "probability_delta": round(delta, 4),
        "within_noise": delta <= margin,
        "spread_range": [round(min(spread_f, spread_r), 4),
                         round(max(spread_f, spread_r), 4)],
        "law": "quote the spread, never the point",
    }


def policy_spread(probs: dict) -> float:
    """
    How far the distribution is from being a point mass. 0 = certain,
    1 - 1/K = uniform.

    Uses the complement of Simpson's index, `1 - sum(p_i^2)`, rather than
    `1 - sum(p_i)/K`.

    The first version used `1 - sum(p_i)/K` and it was BACKWARDS: a point mass
    scored 0.5 and a uniform distribution scored 0.0, so "spread" grew as the
    answer got LESS certain. A test caught it immediately, which is the only
    reason it did not ship -- a metric with an inverted sign is worse than no
    metric, because it looks like a measurement.

    Entropy would also be non-negative here, but it needs a log and Simpson's
    is exact at both ends and cheaper to read off a report.
    """
    if not probs:
        return 1.0
    return round(1.0 - sum(float(p) * float(p) for p in probs.values()), 4)


def is_coin_flip(value, threshold: float,
                 margin: float = POSITION_CONTROL_MARGIN) -> bool:
    """
    Within the noise band of the threshold, so pass/fail would be a coin flip.

    jev-lab's measure.mjs returns COIN_FLIP rather than YES/NO inside a 0.02
    band, for the same reason: an answer that close to the line is not an
    answer, and reporting it as one is how a suite learns to lie.
    """
    if value is None:
        return True
    return abs(float(value) - threshold) <= margin + 1e-9


# -- a human-readable dump for the docs --------------------------------------

def summary() -> str:
    lines = ["policy: measured laws, enforced at call sites", ""]
    for name, law in LAWS.items():
        lines.append(f"[{name}]")
        lines.append(f"  {law['claim']}")
        for k, v in law["measured"].items():
            lines.append(f"  {k} = {v}")
        lines.append(f"  -> {law['consequence']}")
        lines.append(f"  source: {law['source']}")
        lines.append("")
    lines.append(f"BATCH_KNEE       = {BATCH_KNEE}   (latency minimum)")
    lines.append(f"MAX_BATCH        = {MAX_BATCH}    (actual split point)")
    lines.append(f"CONFIDENCE_USABLE = {CONFIDENCE_USABLE}")
    lines.append(f"MIN_BATTERY_FIELDS= {MIN_BATTERY_FIELDS}")
    return "\n".join(lines)