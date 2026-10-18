"""
judged — the four tracks that are JUDGED rather than measured.

Phase 1 measures five tracks from the waveform. These four come from a model,
and they are the novel core of the system.

    TIMBRE     CC71-76   warm / cold / nasal / breathy / bright
    INTENTION  CC74      urgency and how strongly it is held
    ATTENTION  CC91      where focus is directed
    GESTURE    CC80-87   what triggered the utterance

THE CENTRAL IDEA:

    A calibrated probability IS a continuous controller value,
    and that is exactly what a MIDI CC is.

So this module is not routing a decision. It is a synthesiser's control
surface. "How warm is this utterance?" -> 0.73 -> CC71 = 73.

TWO CORRECTIONS THAT THE LIVE DOCS FORCED, recorded because both were wrong
in the first draft of this design and neither is obvious:

1. PRIMITIVE PER TRACK, NOT ONE PRIMITIVE FOR ALL FOUR.

   The docs are explicit: Choice is for "one of a fixed set of options",
   Score is for "a position on a spectrum", Noul is for "is this true".

   TIMBRE and INTENTION are spectra -- warm is the opposite of cold, and
   urgency has a position on a line. Asking for them as Choice throws away
   the ordering the model would otherwise exploit, and a Choice answer gives
   you a winner plus a distribution, not a position.

   ATTENTION and GESTURE are genuinely categorical, so they stay Choice.

2. THE CC VALUE IS A PROBABILITY, NOT `confidence`.

   This is the one that would have produced a plausible-looking, wrong system.
   A Choice answer carries `confidence`, and it is tempting to write
   `CC74 = answer.confidence`. Do not. The docs define confidence as a
   function of how the probability distribution is SPREAD -- a flat shape has
   low confidence, a single peak has high confidence. It answers "how
   unambiguous was this?", not "how likely is this?".

   Those are different questions, and for a MIDI controller we need the
   second one. `confidence` is invariant to the actual probabilities: a
   distribution of {0.34, 0.40, 0.24} is maximally ambiguous and scores 0.20,
   but it is ALSO the most likely state, and an agent steering voice
   character off "confidence" would systematically under-drive exactly the
   cases that are most ambiguous.

   So the CC value is taken from `probabilities[chosen]` for Choice, and from
   the Score/Noul position for the graded tracks.

ONE CALL, NOT FOUR. The docs say questions are evaluated in parallel and that
adding questions "barely changes the response time". Four tracks for the price
of roughly one is the whole reason this is affordable in a voice loop, where
latency is the product. A per-track call would be an architectural error.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Model generation is named EXPLICITLY, never via the floating "jev-latest"
# alias. On this account jev-latest resolves to jev-1.13.0 while jev-preview
# resolves to jev-1.14.0, so an alias is a number with no fixed referent.
DEFAULT_MODEL = "jev-1.14.0"
PREVIEW_MODEL = "jev-preview"

# CC numbers as slackwater-perception assigns them. Kept here so the mapping
# is in one place and a mismatch is a test failure rather than a surprise in a
# rendered file.
CC = {
    "INTENTION": 74,
    "TIMBRE": 71,      # CC71-76 is the spectral-colour bank
    "ATTENTION": 91,
    "GESTURE": 80,     # CC80-87
}

# Track -> MIDI channel, matching the 9-track convention.
CHANNEL = {
    "TIMBRE": 1, "INTENTION": 2, "ATTENTION": 3, "GESTURE": 4,
}


@dataclass
class TrackReading:
    """One judged track. `value` is the CC-scale 0-127 control position."""

    track: str
    value: int                 # 0..127
    detail: dict = field(default_factory=dict)
    model: str | None = None   # the build that actually answered
    ok: bool = True

    @property
    def cc(self) -> int:
        return CC[self.track]


def _scale(p: float) -> int:
    """Probability in [0,1] to a 0..127 CC value.

    Rounded, and clamped, because a control value outside 0..127 is a MIDI
    encoding error rather than a loud signal.
    """
    if p is None:
        return 0
    return max(0, min(127, int(round(float(p) * 127))))


def _scale_position(score: float, n_levels: int) -> int:
    """
    A Score's position to a 0..127 CC value.

    THIS IS NOT THE SAME AS _scale, and getting it wrong is silent.

    `ScoreAnswer.score` is a float POSITION on a 0..(n_levels-1) scale, not a
    probability in 0..1. Measured against the live API with 4 levels:

        "hey"                          -> score 0.44   (idle..attentive)
        "can you pull the last commit" -> score 2.34   (focused..urgent)
        "STOP THE DEPLOY RIGHT NOW"    -> score 3.00   (urgent)

    The first version of this module called _scale(score) directly. That
    multiplied 2.34 by 127, hit the clamp, and returned 127 -- so EVERY score of
    1.0 or above rendered as maximum intensity. A polite request and an
    emergency both came out at full tilt, and the test suite could not see it
    because both are plausible-looking numbers in range.

    So normalise by (n_levels - 1) to reach 0..1 first, then scale.
    """
    if score is None or n_levels < 2:
        return 0
    frac = float(score) / float(n_levels - 1)
    return _scale(max(0.0, min(1.0, frac)))


# -- the questions ----------------------------------------------------------
#
# Wording matters more than anything else here, and the docs are specific:
# "write descriptions that separate the options from each other". Every
# option below says what it is AND what it is not, because TIMBRE's
# neighbours genuinely overlap -- breathy is quiet AND airy, nasal is a
# resonance AND a voice quality.

TIMBRE_SCALE = [
    "cold: distant, flat, detached. Not warmth, not brightness.",
    "warm: close, gentle, affectionate. Not breathiness.",
    "bright: crisp, clear, energetic, lightly upbeat.",
    "nasal: resonant through the nose, slightly pinched or whiny.",
    "breathy: audible breath, soft edges, airy, little vocal fold.",
]

INTENTION_SCALE = [
    "idle: no particular intent behind it, ambient or reflexive.",
    "attentive: mild interest, the speaker is considering something.",
    "focused: deliberate, the speaker means exactly what they said.",
    "urgent: needs acting on now, something is time-sensitive.",
]

ATTENTION_CHOICE = {
    "self": "The speaker is talking to themselves or thinking aloud.",
    "listener": "Directed at the person or system being spoken to.",
    "task": "About the work in front of them rather than a person.",
    "tool": "About the state or output of a tool or machine.",
    "none": "No clear addressee; neutral or incidental.",
}

GESTURE_CHOICE = {
    "question": "Asking for information or an answer.",
    "command": "Instructing something to happen.",
    "statement": "Asserting something without asking for action.",
    "reaction": "An immediate response to something just perceived.",
    "greeting": "Opening or acknowledging contact.",
}

# The instructions live in named constants rather than inline in
# build_questions() so the ENTIRE question wording can be fingerprinted
# without constructing SDK objects. The fingerprint keys the verdict cache:
# change a word of a question and every cached verdict for it is invalid,
# by construction, with no manual invalidation anywhere.
TIMBRE_INSTRUCTIONS = (
    "Where does this utterance sit on the warm-to-cold scale? Judge the "
    "delivery as performed, not the sentiment of the words -- a "
    "cheerful statement can be delivered cold."
)
INTENTION_INSTRUCTIONS = (
    "How much intent stands behind this utterance? Judge how strongly "
    "the speaker means to be acted upon."
)
ATTENTION_INSTRUCTIONS = "Where is this utterance's focus directed?"
GESTURE_INSTRUCTIONS = "What kind of speech act is this utterance?"

# -- the presence gates -----------------------------------------------------
#
# ADDED AFTER MEASUREMENT (Phase 2.7), and they are the whole point of that
# phase. Three of the four questions were ill-posed: INTENTION and ATTENTION
# scored HIGH spread-confidence on semantic garbage ("qwop zzz brxt"), i.e.
# they were placing nonsense on the scale confidently rather than declining.
#
# The obvious fix -- re-word the scale, give it an escape hatch for non-speech
# -- was MEASURED AND IT MADE THINGS WORSE:
#
#     reworded alone   INTENTION CC separation  +82 -> -59
#                      ATTENTION  CC separation   -8 -> -27
#
# Because Score and Choice have no way to decline. Both primitives must commit
# to a position, so an utterance with no such property is still forced onto the
# scale, and adding a "not an utterance" level just gave the model somewhere to
# put the garbage at full tilt.
#
# Noul is the one primitive here that can answer "I don't know": it returns the
# PROBABILITY OF PRESENCE, and 0.5 is a legitimate answer. So each track gained
# a gate, asked in the SAME request (one call rule -- the cost is one round
# trip, not eight):
#
#     gated            ATTENTION  CC separation   -8 -> +53   garbage 107 -> 18
#                      GESTURE    CC separation  +46 -> +86   garbage  80 -> 24
#                      INTENTION  CC separation  +82 -> +70   garbage   3 ->  0.4
#
# The gate is MULTIPLIED into the CC, not thresholded: at gate 0.5 the
# synthesiser gets half the value, which fades a voice it is not sure it heard
# rather than clicking it off. Ambiguous real speech is held back, not erased.
#
# TIMBRE's gate reads ~0.15 on BOTH batteries -- the model correctly says the
# words alone carry no vocal colour. That is the honest answer: TIMBRE is not
# a text property, it lives in the measured acoustic tracks. Its gated CC is
# therefore near zero by design, not by failure.

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

# gate answer key -> track. The order matches judged.CC so a fingerprint diff
# reads as a diff of the same tracks.
GATES = {
    "gate_intention": "INTENTION",
    "gate_attention": "ATTENTION",
    "gate_timbre": "TIMBRE",
    "gate_gesture": "GESTURE",
}

# The wording, keyed the same way, so question_fingerprint() digests the text
# the model actually reads. Folding the key map in without the wording would
# mean a re-worded gate kept every cached verdict -- exactly the silent
# invalidation failure the fingerprint exists to prevent.
GATE_INSTRUCTIONS = {
    "gate_intention": GATE_INTENTION,
    "gate_attention": GATE_ATTENTION,
    "gate_timbre": GATE_TIMBRE,
    "gate_gesture": GATE_GESTURE,
}


def question_fingerprint() -> str:
    """
    A digest of exactly what the model is asked, per track.

    This is decision-ledger's question_hash, applied to a control surface: it
    makes question wording a first-class versioned input. The verdict cache
    and the calibration records both key on it, so a rewrite of a question can
    never be silently mixed with readings taken under the old wording.
    """
    import hashlib
    import json

    spec = {
        "timbre": {"instructions": TIMBRE_INSTRUCTIONS, "criteria": TIMBRE_SCALE},
        "intention": {"instructions": INTENTION_INSTRUCTIONS,
                      "criteria": INTENTION_SCALE},
        "attention": {"instructions": ATTENTION_INSTRUCTIONS,
                      "criteria": ATTENTION_CHOICE},
        "gesture": {"instructions": GESTURE_INSTRUCTIONS,
                    "criteria": GESTURE_CHOICE},
        # The gates are part of what the model is asked, so they belong in the
        # fingerprint. Adding them CHANGED the fingerprint, which is correct:
        # every verdict cached under the ungated questions is now invalid by
        # construction, with no manual cache invalidation anywhere.
        "gates": GATE_INSTRUCTIONS,
    }
    return hashlib.sha1(
        json.dumps(spec, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()[:12]


def build_questions():
    """
    All four scale questions AND the four presence gates, in ONE call.

    Parallel evaluation means this is roughly the price of one question, and
    splitting it would be an architectural error in a system whose latency
    budget is the reason it exists. The gates in particular must arrive WITH
    the values they suppress: a gate fetched in a second call would describe
    the previous utterance by the time it was used.

    Eight questions now ride in one request. That number is asserted by a test,
    because "let's split this for clarity" is exactly the kind of change that
    doubles a voice loop's latency and looks like a refactor in the diff.
    """
    from typesafe_sdk import Choice, Noul, Score

    qs = {
        "timbre": Score(instructions=TIMBRE_INSTRUCTIONS, criteria=TIMBRE_SCALE),
        "intention": Score(instructions=INTENTION_INSTRUCTIONS,
                           criteria=INTENTION_SCALE),
        "attention": Choice(instructions=ATTENTION_INSTRUCTIONS,
                            criteria=ATTENTION_CHOICE),
        "gesture": Choice(instructions=GESTURE_INSTRUCTIONS,
                          criteria=GESTURE_CHOICE),
    }
    # NoulCriteria's keys are `true`/`false`. Writing `yes`/`no` raises
    # ValidationError: criteria.yes -- Extra inputs are not permitted, which
    # reads like a library bug rather than a wrong keyword.
    for key, text in GATE_INSTRUCTIONS.items():
        qs[key] = Noul(instructions=text, criteria={
            "false": "No such property is present in this utterance."})
    return qs


def read_tracks(text: str, *, model: str = DEFAULT_MODEL,
                client=None, timeout: float = 8.0,
                cache=None) -> tuple[list[TrackReading], str]:
    """
    Score one utterance. Returns the four readings and the model that answered.

    Never raises: a voice loop that dies because a network call failed is
    worse than a voice loop that says nothing. Each track carries `ok=False`
    so the caller can distinguish "unvoiced" from "we did not get to ask".

    `cache` (a verdict_cache.VerdictCache) makes the SAME utterance under the
    SAME questions and model return the SAME control values. Jev is not
    bit-deterministic -- jev-lattice measured 7% of gates flipping on a pure
    re-ask at threshold 0.9 -- and a synthesiser fed a controller that jitters
    for no reason is an instrument nobody can play. Only successful readings
    are cached: a network failure must never become a sticky verdict.
    """
    if cache is not None:
        hit = cache.get(text, model)
        if hit is not None:
            readings = [TrackReading(r["track"], r["value"],
                                     detail=dict(r["detail"], cached=True),
                                     model=r["model"], ok=r["ok"])
                        for r in hit["readings"]]
            return readings, hit["model"]

    # The key check guards CONSTRUCTING a client, not calling one. An injected
    # client has already had its authentication decided by whoever injected it
    # -- a test double, a proxy, a cached session -- so gating on the ambient
    # environment would short-circuit a path the caller deliberately chose.
    if client is None and not (os.environ.get("TYPESAFE_API_KEY")
                               or os.environ.get("TYPESAFEAI_KEY")):
        return ([TrackReading(t, 0, ok=False, detail={"why": "no key"}) for t in CC],
                model)

    try:
        if client is None:
            from typesafe_sdk import TypeSafeClient
            client = TypeSafeClient()
        resp = client.system_one(state=text, questions=build_questions(),
                                 model=model, timeout=timeout)
    except Exception as exc:                      # noqa: BLE001
        return ([TrackReading(t, 0, ok=False, detail={"why": f"{type(exc).__name__}"})
                 for t in CC], model)

    resolved = str(getattr(resp, "model", model))
    out: list[TrackReading] = []
    gate_by_track = {t: getattr(resp.answers.get(k), "noul", None)
                     for k, t in GATES.items()}

    def _gate(track: str):
        """
        The presence value for a track, and whether it was actually answered.

        A MISSING gate is treated as 1.0 -- ungated -- because a network or
        schema hiccup must not silence a real utterance. But it is recorded as
        gated=False in the detail, so "this track ran ungated" is visible in
        the score rather than being indistinguishable from a gate of 1.0 that
        the model genuinely returned. Silently ungating is the direction that
        lets garbage drive the synthesiser, so it may not be invisible.
        """
        g = gate_by_track.get(track)
        if g is None:
            return 1.0, False
        return float(g), True

    for key, track, n_levels in (("timbre", "TIMBRE", len(TIMBRE_SCALE)),
                                 ("intention", "INTENTION", len(INTENTION_SCALE))):
        a = resp.answers[key]
        # ScoreAnswer exposes `score` -- a POSITION on a 0..(n-1) scale, not a
        # probability. _scale_position normalises it. Passing it to _scale
        # directly is the bug this function's docstring describes.
        pos = getattr(a, "score", None)
        probs = getattr(a, "probabilities", None) or {}
        if pos is None:
            pos = (max(probs.values()) if probs else 0.0)
        legend = getattr(a, "legend", None) or {}
        # The named level the mass actually sits on, kept for humans reading
        # the score. The CC uses the continuous position, not this label.
        nearest = max(probs, key=probs.get) if probs else None
        gate, gated = _gate(track)
        out.append(TrackReading(track,
                                int(round(_scale_position(pos, n_levels) * gate)),
                                detail={"score": pos,
                                        "nearest_level": legend.get(nearest),
                                        "confidence": getattr(a, "confidence", None),
                                        "gate": gate_by_track.get(track),
                                        "gated": gated},
                                model=resolved))

    for key, track in (("attention", "ATTENTION"), ("gesture", "GESTURE")):
        a = resp.answers[key]
        chosen = a.choice
        probs = getattr(a, "probabilities", None) or {}
        # The CC is the probability of the OPTION THAT WON, not confidence.
        # See the module docstring: confidence measures spread, and using it
        # here would systematically under-drive the ambiguous cases, which are
        # exactly the ones where a controller wants to hold back.
        p = probs.get(chosen)
        gate, gated = _gate(track)
        out.append(TrackReading(track,
                                int(round(_scale(p) * gate)) if p is not None else 0,
                                detail={"choice": chosen, "probabilities": probs,
                                        # spread-confidence is NOT the CC, but
                                        # it is the question-health signal --
                                        # stored as provenance, never driven.
                                        "confidence": getattr(a, "confidence", None),
                                        "gate": gate_by_track.get(track),
                                        "gated": gated},
                                model=resolved))

    if cache is not None and all(r.ok for r in out):
        cache.put(text, model, resolved, out)
    return out, resolved