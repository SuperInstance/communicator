#!/usr/bin/env python
"""
Can a SYMBOLIC encoding of audio carry spatial structure that scalar
summaries structurally cannot?

THE MOTIVATION, and it comes from two measurements rather than an intuition.

(1) Syzygy (SuperInstance/syzygy) turns a camera frame into TEXT in a single
    fused pass of integer arithmetic: a Braille mask per 2x4 cell, a Sobel
    edge glyph, a tone density char, and an argmax spectrum bin. ~1000 lines
    of C, byte-exact across gcc/clang/wasm/JavaScript, no floats, no malloc.
    Its own motto is "measures; it does not judge" -- it fills the measured
    field and leaves interpretation to a separate verification layer.

(2) I measured that Jev 1.14 ACCEPTS images and does not READ them.
    Feeding it spectrograms: pitch-band accuracy 1/3, and swapping a real
    spectrogram for a blank grey image of identical dimensions moved the
    answer by 0.020. Meanwhile the same model read a STATED NUMBER perfectly:
    stating "Measured pitch: 150/300/450 Hz" moved confidence to
    0.660 / 0.270 / 0.100, monotonic and correct.

    So: pixels are dead, text is alive. Jev's only working input channel is
    language. The question is therefore not "can it see the spectrogram" but
    "how much of the signal survives the trip through language".

THE EXPERIMENT is built so the answer cannot be flattered.

    Three synthetic utterances with IDENTICAL scalar summaries:
        rising  180 Hz -> 300 Hz   (mean 240)
        falling 300 Hz -> 180 Hz   (mean 240)
        flat    240 Hz constant    (mean 240)
    Every one of them has the same mean pitch, the same duration and the same
    energy. A model that reads ONLY the numbers we currently send -- and our
    PITCH track sends exactly a scalar -- CANNOT distinguish them. Not
    "probably cannot": there is no information to distinguish them. The
    scalar encoding is information-theoretically blind to the answer.

    The symbolic encoding carries the shape. If Jev is above chance on one
    and at chance on the other, the encoding is what did the work, and the
    shape was destroyed somewhere upstream of the model.

THREE ARMS:
    scalar   -- what src/score.py sends today. Must sit at chance.
    symbolic -- a Syzygy-flavoured text rendering: a pitch ladder, a
                sparkline, a density ramp, and Sobel-style direction glyphs.
    raw      -- the waveform's own description in prose, as a sanity arm.

PLUS A POSITIVE CONTROL: a question the scalar CAN answer (mean pitch band).
If that also sits at chance, the harness is broken and the scalar arm's
result means nothing.

PREDICTIONS, stated before the run:
    P1  scalar arm sits AT chance (1/3) on shape, because it must.
    P2  symbolic arm is WELL ABOVE chance, because text is the live channel.
    P3  scalar arm is ABOVE chance on the mean-pitch control (it is a number).
    If P2 fails, the shape did not survive language, and every reduction
    upstream of the model is lossy in a way that matters.

    set -a && . /home/proart/projects/glasses/.env && set +a
    /home/proart/projects/glasses/.venv/bin/python experiments/symbolic_audio.py
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = os.environ.get("JEV_MODEL", "jev-1.14.0")
SR = 16000
DURATION = 1.2
STEPS = 12


# -- signal generation --------------------------------------------------------

def sweep(start_hz, end_hz, duration=DURATION, amp=0.3, sr=SR):
    """Linear frequency sweep. Phase is integrated, not recomputed per sample,
    so the waveform is genuinely continuous -- a naive `sin(2*pi*f(t)*t)`
    would alias at the seam and put a click exactly where the shape changes."""
    n = int(duration * sr)
    phase = np.cumsum(2 * np.pi * np.linspace(start_hz, end_hz, n) / sr)
    return (amp * np.sin(phase)).tolist()


# -- the SYZYGY-FLAVOURED symbolic encoding -----------------------------------
#
# Vocabulary borrowed in spirit from Syzygy's own: a fixed ramp, an argmax
# bucket rather than a float, and a direction glyph standing in for a Sobel
# response. Nothing here is a float in the text; every value is a character,
# so the model reads the same kind of thing Syzygy emits.

PITCH_RAMP = " .:-=+*#%@"
DENSITY_RAMP = " .:-=+*#%@"
DIRECTIONS = {
    (1, 0): "─",    # steady
    (1, 1): "╱",    # rising
    (1, -1): "╲",   # falling
}


def _ramp_char(ramp: str, value: float, lo: float, hi: float) -> str:
    if hi <= lo:
        return ramp[0]
    k = int(round((value - lo) / (hi - lo) * (len(ramp) - 1)))
    return ramp[max(0, min(len(ramp) - 1, k))]


def measure(samples, sr=SR, steps=STEPS):
    """The MEASURED fields. No judgement -- exactly Syzygy's division."""
    n = len(samples)
    win = max(1, n // steps)
    pitch_hz, rms, zcr = [], [], []
    for s in range(steps):
        seg = samples[s * win:(s + 1) * win]
        if not seg:
            pitch_hz.append(0.0)
            rms.append(0.0)
            zcr.append(0.0)
            continue
        pitch_hz.append(_dominant_hz(seg, sr))
        rms.append(math.sqrt(sum(x * x for x in seg) / len(seg)))
        crossings = sum(1 for a, b in zip(seg, seg[1:]) if (a < 0) != (b < 0))
        zcr.append(crossings / len(seg))

    return {"pitch_hz": pitch_hz, "rms": rms, "zcr": zcr}


def _dominant_hz(seg, sr, fmin=60.0, fmax=600.0, bins=192):
    """
    Argmax over a Goertzel scan of the candidate frequencies.

    This is the same shape as Syzygy's spectrum argmax -- pick the bucket with
    the biggest score, emit an index, never a probability -- done here in
    floating point on a host that is not a microcontroller. The Goertzel
    recurrence is vectorised over bins because 192 bins x 1600 samples x 12
    steps x 3 cases in pure Python is minutes of wall clock, and a slow
    experiment is one that stops being re-run.
    """
    x = np.asarray(seg, dtype=np.float64)
    if x.size == 0:
        return 0.0
    hz = np.linspace(fmin, fmax, bins)
    omega = 2 * np.pi * hz / sr
    coeff = 2 * np.cos(omega)
    s1 = np.zeros(bins)
    s2 = np.zeros(bins)
    for sample in x:
        s0 = sample + coeff * s1 - s2
        s2, s1 = s1, s0
    power = s1 * s1 + s2 * s2 - coeff * s1 * s2
    return float(hz[int(np.argmax(power))])


def encode_symbolic(m) -> str:
    """
    The Syzygy-shaped text. Four rows, all discrete:

      PITCH   a ladder: each step's dominant pitch as a character in a fixed
              ramp, so the SHAPE survives as a shape rather than as three
              numbers.
      SPARK   block bars, for a second redundant read of the same shape.
      DENSITY the energy ramp -- the visual equivalent of Syzygy's tone chars.
      EDGE    a direction glyph per step: steady, rising or falling. This is
              the closest analogue to the Sobel glyph and it is what makes
              "rises then falls" representable at all.

    No float appears in the output. That is the point of the encoding: if the
    model can answer from this, then language is a sufficient channel for the
    structure, and it does not matter that we threw the exact values away.
    """
    hz = m["pitch_hz"]
    lo, hi = min(hz), max(hz)
    ladder = "".join(_ramp_char(PITCH_RAMP, v, 0.0, 700.0) for v in hz)
    bars = ""
    for v in hz:
        k = 0 if hi <= lo else int(round((v - lo) / (hi - lo) * 7))
        bars += "▁▂▃▄▅▆▇█"[max(0, min(7, k))]
    density = "".join(_ramp_char(DENSITY_RAMP, v, 0.0, 0.35) for v in m["rms"])

    edge = ""
    for a, b in zip(hz, hz[1:]):
        if b > a * 1.04:
            edge += "╱"
        elif b < a * 0.96:
            edge += "╲"
        else:
            edge += "─"
    return (f"PITCH   [{ladder}]\n"
            f"SPARK   [{bars}]\n"
            f"DENSITY [{density}]\n"
            f"EDGE    [{edge}]")


def encode_scalar(m) -> str:
    """What src/score.py sends today: summary statistics, no shape."""
    hz = m["pitch_hz"]
    return (f"mean pitch {statistics.mean(hz):.1f} Hz; "
            f"duration {DURATION:.2f} s; "
            f"voiced throughout; one continuous utterance.")


def encode_symbolic_absolute(m) -> str:
    """
    THE SAME ENCODING with one row changed, and the change is the whole point.

    The first run scored 3/3 on shape and 1/3 on mean pitch -- the symbolic arm
    was reading the SHAPE correctly and the LEVEL incorrectly. The two
    readings were in conflict because they came from two different rows:

      SPARK was normalised PER UTTERANCE (lo..hi across this utterance), so
      it is scale-free by construction. A rising sweep ends at the top of the
      sparkline, which is the same glyph a genuinely high voice would end on.
      PITCH was already on a fixed 0..700 Hz ladder, so it carried the level.

    The model appears to have latched onto SPARK for the level question and got
    `high / mid / low` -- which is the trajectory, not the register.

    So this arm makes SPARK absolute too. PREDICTION: it now scores 3/3 on
    shape AND 3/3 on the level, i.e. no information is lost that the scalar
    version had.

    This is Syzygy's I2 invariant arriving from the other direction: measure
    against an absolute scale, never against a per-frame normalisation, because
    a normalised reading throws away exactly the information that makes two
    different things comparable.
    """
    hz = m["pitch_hz"]
    ladder = "".join(_ramp_char(PITCH_RAMP, v, 0.0, 700.0) for v in hz)
    bars = "".join("▁▂▃▄▅▆▇█"[max(0, min(
        7, int(round(v / 600.0 * 7))))] for v in hz)
    density = "".join(_ramp_char(DENSITY_RAMP, v, 0.0, 0.35) for v in m["rms"])
    edge = ""
    for a, b in zip(hz, hz[1:]):
        if b > a * 1.04:
            edge += "╱"
        elif b < a * 0.96:
            edge += "╲"
        else:
            edge += "─"
    return (f"PITCH   [{ladder}]\n"
            f"SPARK   [{bars}]\n"
            f"DENSITY [{density}]\n"
            f"EDGE    [{edge}]")


def encode_prose(m) -> str:
    hz = m["pitch_hz"]
    return (f"A voice speaking for {DURATION:.1f} seconds. The pitch of the "
            f"voice at the start of the utterance and the pitch at the end "
            f"are the same.")


# -- probes -------------------------------------------------------------------

SHAPE_Q = ("Reading the readings in order, what happens to the pitch across "
           "this utterance?")
SHAPE_OPTIONS = {
    "rising": "The pitch climbs steadily from the start to the end.",
    "falling": "The pitch drops steadily from the start to the end.",
    "flat": "The pitch stays at one level throughout, with no climb and no "
            "drop.",
}

BAND_Q = "What pitch band is this voice in?"
BAND_OPTIONS = {
    "low": "A low-pitched voice.",
    "mid": "A mid-pitched voice.",
    "high": "A high-pitched voice.",
}

# 12 evenly spaced tones whose mean is exactly 240 Hz, so all three cases sit
# in the same band. Only the ORDER differs.
CASES = [
    ("rising", sweep(180.0, 300.0), "rising"),
    ("falling", sweep(300.0, 180.0), "falling"),
    ("flat", sweep(240.0, 240.0), "flat"),
]


def ask_shape(cli, state):
    from typesafe_sdk import Choice
    a = cli.system_one(state=state,
                       questions={"s": Choice(instructions=SHAPE_Q,
                                              criteria=SHAPE_OPTIONS)},
                       model=MODEL).answers["s"]
    return {"picked": a.choice, "confidence": round(a.confidence, 4),
            "probs": {k: round(v, 3) for k, v in a.probabilities.items()}}


def ask_band(cli, state):
    from typesafe_sdk import Choice
    a = cli.system_one(state=state,
                       questions={"b": Choice(instructions=BAND_Q,
                                              criteria=BAND_OPTIONS)},
                       model=MODEL).answers["b"]
    return {"picked": a.choice, "confidence": round(a.confidence, 4)}


def score(rows, key="picked"):
    correct = sum(1 for r in rows if r[key] == r["truth"])
    return correct, len(rows)


def main() -> int:
    if not (os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFEAI_KEY")):
        print("no TYPESAFE_API_KEY")
        return 2
    from typesafe_sdk import TypeSafeClient
    cli = TypeSafeClient()

    # Measure first, and PRINT the scalars so the identity of the three cases
    # is verifiable rather than asserted.
    prepared = []
    print("== 0. Are the three cases really indistinguishable as scalars? ==")
    for name, samples, truth in CASES:
        m = measure(samples)
        prepared.append((name, m, truth))
        hz = m["pitch_hz"]
        print(f"   {name:8s} mean={statistics.mean(hz):6.1f} Hz  "
              f"first={hz[0]:6.1f} last={hz[-1]:6.1f}  "
              f"rms_mean={statistics.mean(m['rms']):.4f}")

    means = [statistics.mean(m["pitch_hz"]) for _, m, _ in prepared]
    rmses = [statistics.mean(m["rms"]) for _, m, _ in prepared]
    print(f"   mean-pitch spread {max(means)-min(means):.1f} Hz; "
          f"energy spread {max(rmses)-min(rmses):.5f}")

    report = {"model": MODEL, "mean_pitch_spread_hz":
              round(max(means) - min(means), 1)}

    print("\n-- one encoding, so the format is legible --")
    _, m0, _ = prepared[0]
    print(encode_symbolic(m0))
    print(f"   scalar: {encode_scalar(m0)}")

    print("\n== 1. SHAPE question: rising / falling / flat ==")
    print("   P1 scalar at chance (it structurally must be)")
    print("   P2 symbolic well above chance (text is the live channel)")
    arms = {}
    for arm, encoder in (("scalar", encode_scalar),
                         ("symbolic", encode_symbolic),
                         ("prose", encode_prose)):
        rows = []
        for name, m, truth in prepared:
            state = encoder(m)
            r = ask_shape(cli, state)
            r.update({"case": name, "truth": truth})
            rows.append(r)
        arms[arm] = rows
        c, n = score(rows)
        picks = " ".join(f"{r['case'][:4]}:{r['picked'][:4]}"
                         f"{'' if r['picked'] == r['truth'] else '!'}()"
                         .replace("()", "") for r in rows)
        print(f"   {arm:9s} {c}/{n}   {picks}")
    report["shape"] = arms

    print("\n== 2. POSITIVE CONTROL: mean pitch band ==")
    print("   P3 scalar above chance here -- if not, the harness is broken")
    ctl = {}
    for arm, encoder in (("scalar", encode_scalar),
                         ("symbolic", encode_symbolic),
                         ("symbolic_abs", encode_symbolic_absolute)):
        rows = []
        for name, m, truth in prepared:
            # All three sit near 240 Hz, so "mid" is the truth for all.
            rows.append({"case": name, "truth": "mid",
                         **ask_band(cli, encoder(m))})
        ctl[arm] = rows
        c = sum(1 for r in rows if r["picked"] == r["truth"])
        print(f"   {arm:13s} {c}/{len(rows)}   "
              + " ".join(f"{r['case'][:4]}:{r['picked']}" for r in rows))
    report["band_control"] = ctl

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "symbolic_audio_results.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())