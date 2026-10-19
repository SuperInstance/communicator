"""
notch_ab.py -- does a mains notch fix the emitted F0, and what does it cost?

THIS IS THE EXPERIMENT `docs/research/IN-THE-WILD-AUDIO.md` §8 PROPOSED AND
NOBODY RAN. `docs/research/MIC-HARDWARE.md` priced the filter on our own
tracker (0.24 cents median F0 error for a four-notch bank) but never pushed a
filtered signal through the *watcher* and never looked at `hz_p05` on a mixture.
That is the gap this file closes, and it is the gap where the bug lives:
§4.3 of IN-THE-WILD-AUDIO localised the correctness failure INSIDE the pitch
tracker, on segments real speech opened, where no gate threshold can see.

THE NUMBER THAT BREAKS TODAY (reproduced by this file, see `controls.reproduced`):

    clean speech, emitted hz_p05   163.61  179.04  189.28  194.49 Hz
    150 Hz stack the gate PASSES   150.0   150.0   150.0   Hz
    200 Hz stack the gate PASSES   200.01  200.01  200.01  Hz

The emitted pitch IS the hum frequency, reported as though it were a voice.
On speech+hum mixtures at 10 dB, 150 Hz gives 74.97 / 74.99 / 79.86 Hz -- an
octave collapse to 150/2, because YIN is autocorrelation and a periodic
interferer inside an open segment puts a narrow peak at the hum period and its
subharmonics.

WHY A NOTCH IS THE ONLY CANDIDATE HERE. MIC-HARDWARE.md proved a high-pass
cannot work: a 200 Hz corner cuts a 100 Hz male F0 by 24.6 dB while cutting
200 Hz hum by only 6.0 dB -- four times more voice than hum. And IN-THE-WILD-AUDIO
§4.4 falsified per-device spectral subtraction against this gate at matched
amplitude. The notch is what is left, and it has never been run end to end.

WHERE THE FILTER IS APPLIED, AND WHY IT IS NOT EXACTLY §8's PLACE.

§8 asked for the notch AFTER the gate opens a segment and BEFORE
`pitch.PitchTracker`. `src/` is owned by another agent and must not be edited,
and `watcher._events_for` constructs its own `pitch.PitchTracker` with no seam
to substitute at. So the notch is applied to the SAMPLE ARRAY, ahead of the
gate. That is strictly downstream-inclusive for the metric that matters -- the
tracker sees exactly the audio §8's placement would give it -- and it has one
extra effect §8's placement would not have: it also perturbs the energy
segmenter and the gate input. That effect is not hidden, it is MEASURED, on
`utterances`, `voiced_ratio` and `gate_speech_prob`, which are reported per cell.
On a real device the input placement is also the honest one: firmware notch
before everything.

FIXTURES REUSED, NOT INVENTED (imported from `experiments/inthewild_probe.py`,
so these are the same numbers §4 of IN-THE-WILD-AUDIO.md reports):

  speech_clip()                      6 s of examples/realaudio/alice01_excerpt.wav,
                                     peak-normalised -> clean baseline
  harmonic_stack(f0, 1.0)            12-harmonic 1/k peak-normalised mains stack
  harmonic_stack(f0, 0.3, harms=(1,2,3), normalise=False)
                                     the tests/test_vad.py::TestDeclaredLimits
                                     fixture -- the one that PASSES the gate
  mix(speech, hum, snr_db)           speech RMS to hum RMS at the requested SNR
  FUNDAMENTALS = [50,60,100,120,150,175,200]
  SNRS_DB        = [20,10,0]
  fresh_gate()                      one ONNX session, reset between signals
  emitted_pitch()                    the same push/emitted protocol

Nothing here is a new generator. Two new families only in the sense that
`mix()` is now applied to the testvad fixture as well, which is the combination
the §8 proposal named ("50/60/100/120/150/200 Hz x {20,10,0} dB") but
inthewild_probe only ran over the 12-harmonic family.

FILTERS ARE RBJ BIQUADS, THE SAME CODE PATH AS `experiments/mic_filter_sim.py`,
imported rather than retyped, so a filter number in MIC-HARDWARE.md and a
filter number here are the same filter. Input is handed to it as a Python list
of floats because `mic_filter_sim` receives `wavio.read_wav().samples`, which is
a list of floats -- so the arithmetic is float64 in both files.


========================================================================
THE DECISION RULE. WRITTEN BEFORE ANY ARM WAS RUN. NOT EDITED AFTER.
========================================================================

Reference quantity, fixed from the UNFILTERED clean-speech run:

  CLEAN_MED = median of the emitted `hz_p05` over the utterances of clean
              6 s speech, arm `none`.

METRIC 1 -- RECOVERY (primary). For every emitted `hz_p05` value `v` on every
  mixture cell, in both families, `c(v) = 1200*log2(v / CLEAN_MED)`.
      R(A) = fraction of those values with |c(v)| <= 100 cents.
  100 cents is one semitone: the emitted pitch must come back to the clean
  speech band, not merely stop being an octave out.

METRIC 2 -- CLEAN REGRESSION, median. On clean speech, for each utterance that
  has an `hz_p05` under BOTH arms, `d_i = 1200*log2(hz_A,i / hz_none,i)`.
      SHIFT(A)   = median |d_i|
  Reported on the 6 s clip (n=4, the inthewild protocol) and on the full
  90 s clip (n~20, the mic_filter_sim protocol). Both must pass.

METRIC 3 -- CLEAN REGRESSION, maximum. `MAXSHIFT(A) = max |d_i|` over the same
  paired utterances. This is THE NO-INFORMATION CONTROL required of this
  experiment: the notch is applied to a signal that contains no hum at all, so
  if it moves a reported pitch by more than this the harness is wrong or the
  filter is not the narrow thing it claims to be.

METRIC 4 -- DELIVERY. `dUTT(A)` = (utterances carrying an `hz_p05` under A) -
  (the same under `none`), on clean speech. `dVOX(A)` = median difference of
  `voiced_ratio`. MIC-HARDWARE.md measured the high-pass's real cost here: at a
  200 Hz corner it kept the median cents error at 0.73 while quietly deleting 42
  voiced frames. A filter that "does not damage pitch" can still delete speech.

METRIC 5 -- MIXTURE COVERAGE. `dCELLS(A)` = (mixture cells emitting at least
  one `hz_p05` under A) - (the same under `none`). Without this a filter that
  makes the gate veto every mixture segment would score a perfect R by emitting
  nothing at all. R alone cannot see absence; this can.

    SHIP ARM A IFF ALL FIVE HOLD:

      S1  R(A) >= 0.90
      S2  SHIFT(A)  < 10.0 cents   on BOTH clean clips
      S3  MAXSHIFT(A) < 50.0 cents on BOTH clean clips
      S4  dUTT(A) == 0 and |dVOX(A)| < 0.02 on BOTH clean clips
      S5  dCELLS(A) >= 0

    AND the run as a whole is VOID (the numbers are untrustworthy, whatever
    they say) if any of:

      V1  R(hpf60) >= 0.90.
          `hpf60` is the DELIBERATELY LOSING CONTROL: a 4th-order 60 Hz
          high-pass, the "80 Hz rumble filter" folklore value. MIC-HARDWARE.md
          measured it removing 0.08 dB at 200 Hz. If it recovers the mixture
          pitch anyway, this harness cannot detect the failure it was built to
          detect and no arm of it may be believed.
      V2  the no-op arm does not reproduce arm `none` exactly, utterance for
          utterance, bit for bit. Plumbing is lossy.
      V3  any arm emits zero utterances on clean speech.

TWO DELIBERATE FAILURE MODES I ACCEPT AHEAD OF TIME.

  * A notch bank that removes only the fundamental leaves the upper harmonics.
    The 12-harmonic fixture has content at 2*f0, 3*f0 ... and YIN needs only
    periodicity, not the fundamental. Removing 150 Hz from a 150 Hz stack may
    simply hand YIN a clean 300 Hz tone and move the error from an octave DOWN
    to an octave UP. This file measures the residual hum with a Goertzel so
    that outcome is diagnosable rather than mysterious.
  * 150 and 200 Hz are inside the 85-255 Hz speech F0 range (IN-THE-WILD-AUDIO
    §7.5), so the six-notch bank is editing the band the tracker lives in. S3
    is the criterion that catches it and it is deliberately set on the MAXIMUM,
    not the median, because a filter that damages one utterance out of twenty
    is still a filter that damages a table.

WHAT THIS FILE CANNOT SHOW. Every hum here is a synthetic harmonic stack; there
is no audio hardware on this box and nothing here is evidence about a real
transformer, fan or charger. It is evidence about a frequency structure and
about this repo's own tracker.

RESULTS: experiments/notch_ab_results.json
WRITE-UP: docs/research/NOTCH-AB.md
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from experiments import inthewild_probe as probe  # noqa: E402
from experiments.mic_filter_sim import (  # noqa: E402
    NOTCH_Q, attenuation_db, biquad_notch, highpass,
)
from src import wavio, watcher  # noqa: E402

SR = probe.SR
DUR_S = probe.DUR_S
OUT = Path(__file__).resolve().parent / "notch_ab_results.json"
FULL_CLIP = Path(__file__).resolve().parents[1] / "examples/realaudio" / "alice01_excerpt.wav"

# --- the pre-registered rule, carried in the results file ------------------

RULE = {
    "written": "before any arm was run; not edited afterwards",
    "clean_med_reference": ("median emitted hz_p05 over utterances of clean 6 s "
                            "speech under arm `none`"),
    "recovery": "R(A) = frac of mixture emitted hz_p05 within 100 cents of CLEAN_MED",
    "clean_shift": "SHIFT(A) = median |cents(clean hz_p05 A vs none)|, both clean clips",
    "clean_max_shift": "MAXSHIFT(A) = max of the same, both clean clips",
    "delivery": "dUTT(A) = utterances with hz_p05 A - none; dVOX = median voiced_ratio diff",
    "coverage": "dCELLS(A) = mixture cells emitting >=1 hz_p05, A - none",
    "ship": {
        "S1": "R(A) >= 0.90",
        "S2": "SHIFT(A) < 10.0 cents on BOTH clean clips",
        "S3": "MAXSHIFT(A) < 50.0 cents on BOTH clean clips",
        "S4": "dUTT(A) == 0 and |dVOX(A)| < 0.02 on BOTH clean clips",
        "S5": "dCELLS(A) >= 0",
    },
    "void_if": {
        "V1": "R(hpf60) >= 0.90 -- the deliberately-losing control passed",
        "V2": "no-op arm does not reproduce arm `none` bit for bit",
        "V3": "any arm emits zero utterances on clean speech",
    },
}

# --- arms -------------------------------------------------------------------

MAINS_HZ = (50.0, 60.0, 100.0, 120.0)
MAINS_PLUS_SPEECH_HZ = (50.0, 60.0, 100.0, 120.0, 150.0, 200.0)

ARMS = {
    # name: (kind, spec)   kind is one of identity / notch / highpass
    "none": ("identity", ()),
    # Plumbing control. Goes through apply_arm() like every other arm but adds
    # no sections. If V2 trips, the harness is lossy and every number below is.
    "noop_control": ("identity", ()),
    "notch_mains": ("notch", MAINS_HZ),
    "notch_mains_plus_speech": ("notch", MAINS_PLUS_SPEECH_HZ),
    # The deliberately-losing control. MIC-HARDWARE.md measured -0.08 dB at
    # 200 Hz for a 4th-order 60 Hz high-pass. It is here to be shown failing.
    "hpf60": ("highpass", (60.0,)),
}

FILTER_ARMS = ["notch_mains", "notch_mains_plus_speech", "hpf60"]


def apply_arm(x: np.ndarray, arm: str) -> np.ndarray:
    """
    Filter a float32 sample array the way the firmware would, before push().

    Every arm goes through here, including `none`, so that `none` and
    `noop_control` are the same code path and V2 is a real test rather than a
    tautology about two literals.
    """
    kind, spec = ARMS[arm]
    # mic_filter_sim receives wavio.read_wav().samples, a list of Python
    # floats, so the biquad arithmetic there is float64. Hand it the same type
    # or the two files are not running the same arithmetic.
    s = x.astype(np.float64).tolist()
    if kind == "notch":
        for f in spec:
            s = biquad_notch(s, f, sr=SR, q=NOTCH_Q)
    elif kind == "highpass":
        s = highpass(s, spec[0], sections=2)
    return np.asarray(s, dtype=np.float32)


# --- measurement ------------------------------------------------------------

def emitted(samples: np.ndarray) -> list:
    """Every utterance record the watcher emits for this signal."""
    w = watcher.Watcher(SR, gate=probe.fresh_gate())
    for i in range(0, len(samples), probe.STFT_HOP * 8):
        w.push(samples[i:i + probe.STFT_HOP * 8])
    return [e for e in w.emitted if e.kind == watcher.UTTERANCE]


def records(samples: np.ndarray) -> dict:
    """The emitted numbers this experiment decides on, and nothing else."""
    utts = emitted(samples)
    hz = [e.data["hz_p05"] for e in utts if e.data.get("hz_p05")]
    return {
        "utterances": len(utts),
        "utts_with_hz": len(hz),
        "hz_p05_values": sorted(hz),
        "hz_p05_sequence": hz,          # order matters for the paired comparison
        "voiced_ratio_median": round(statistics.median(
            [e.data["voiced_ratio"] for e in utts]), 4) if utts else None,
        "gate_prob_max": max((e.data.get(watcher.GATE_FIELD) or 0.0)
                             for e in utts) if utts else None,
    }


def goertzel_mag(x: np.ndarray, f: float) -> float:
    """Magnitude of `x` at exactly `f`, normalised by length.

    Used to show how much of the ORIGINAL hum survives the filter inside the
    actual mixture, which is what makes a failure diagnosable instead of a
    shrug: a 150 Hz stack with its fundamental removed but 300/450/600 Hz
    intact is still a periodic interferer, and YIN only needs periodicity.
    """
    n = len(x)
    k = int(round(f * n / SR))
    w = 2.0 * math.pi * k / n
    c = 2.0 * math.cos(w)
    s1 = s2 = 0.0
    for v in x[:n]:
        s0 = float(v) + c * s1 - s2
        s2, s1 = s1, s0
    return math.sqrt(max(0.0, s1 * s1 + s2 * s2 - c * s1 * s2)) / n


def db(after: float, before: float) -> float | None:
    return round(20.0 * math.log10(after / before), 2) if before > 0 and after > 0 else None


def cents(a: float, b: float) -> float:
    return 1200.0 * math.log2(a / b)


# --- clean-speech regression ------------------------------------------------

def clean_pair(base: dict, arm: dict) -> dict:
    """
    Paired per-utterance cents shift of `arm` against `none` on clean speech.

    Paired on POSITION in the emitted sequence, not on matching timestamps:
    a filter that shifts an utterance boundary changes which segment is
    utterance i, and a timestamp join would silently drop exactly the
    utterances a boundary shift created. The count is compared in S4 instead.
    """
    b, a = base["hz_p05_sequence"], arm["hz_p05_sequence"]
    n = min(len(b), len(a))
    d = [cents(a[i], b[i]) for i in range(n)] if n else []
    ad = [abs(v) for v in d]
    return {
        "paired_utts": n,
        "shift_median_abs_cents": round(statistics.median(ad), 3) if ad else None,
        "shift_p90_abs_cents": round(sorted(ad)[min(len(ad) - 1,
                                                    int(round(0.9 * (len(ad) - 1))))], 3)
        if ad else None,
        "shift_max_abs_cents": round(max(ad), 3) if ad else None,
        "shift_signed_median_cents": round(statistics.median(d), 3) if d else None,
        "shift_max_signed_cents": round(max(d), 3) if d else None,
        "shift_min_signed_cents": round(min(d), 3) if d else None,
        "dUTT": arm["utts_with_hz"] - base["utts_with_hz"],
        "dVOX": (round(arm["voiced_ratio_median"] - base["voiced_ratio_median"], 4)
                 if arm["voiced_ratio_median"] is not None
                 and base["voiced_ratio_median"] is not None else None),
    }


# --- main -------------------------------------------------------------------

def main() -> None:
    speech = probe.speech_clip()
    clean_med = None

    # ---- spectral control: attenuation measured through the same code path ----
    # mic_filter_sim.attenuation_db(samples, fc, probe_hz, fn) calls fn(t, fc),
    # so each arm is wrapped in that signature.
    def arm_fn(a):
        kind, spec = ARMS[a]
        if kind == "highpass":
            return lambda t, _fc: highpass(t, spec[0], sections=2)
        return lambda t, _fc: _notch_all(t, spec)

    atten = {}
    for a in FILTER_ARMS:
        atten[a] = {}
        for f in (50, 60, 100, 120, 150, 200):
            tone = [math.sin(2.0 * math.pi * f * i / SR) for i in range(SR)]
            atten[a][str(f)] = round(attenuation_db(tone, ARMS[a][1], f, fn=arm_fn(a)), 2)

    # ---- control 1: reproduce the reported bug before touching any filter ----
    repro = {
        "clean_speech_hz_p05": records(speech)["hz_p05_sequence"],
        "testvad_150Hz_hz_p05": records(probe.harmonic_stack(
            150, 0.3, harms=probe.TEST_VAD_HARMS, normalise=False))["hz_p05_sequence"],
        "testvad_200Hz_hz_p05": records(probe.harmonic_stack(
            200, 0.3, harms=probe.TEST_VAD_HARMS, normalise=False))["hz_p05_sequence"],
        "mixture_150Hz_10dB_hz_p05": records(probe.mix(
            speech, probe.harmonic_stack(150, 1.0), 10.0))["hz_p05_sequence"],
        "note": ("arm `none`, the unmodified pipeline. These are the numbers "
                 "from IN-THE-WILD-AUDIO.md §4.3; if they do not reproduce, "
                 "the harness is not the harness that found the bug."),
    }

    # ---- clean speech, every arm, both clips --------------------------------
    full = np.asarray(wavio.read_wav(str(FULL_CLIP)).samples, dtype=np.float32)

    clean = {"speech6": {}, "alice_full": {}}
    for a in ARMS:
        clean["speech6"][a] = records(apply_arm(speech, a))
    for a in ARMS:
        clean["alice_full"][a] = records(apply_arm(full, a))

    clean_med = statistics.median(clean["speech6"]["none"]["hz_p05_sequence"])

    regress = {"speech6": {}, "alice_full": {}}
    for clip in ("speech6", "alice_full"):
        base = clean[clip]["none"]
        # `none` against itself is included so the table is complete and every
        # arm in the verdict block has a regression row to read.
        regress[clip]["none"] = clean_pair(base, base)
        for a in ARMS:
            if a == "none":
                continue
            regress[clip][a] = clean_pair(base, clean[clip][a])

    # V2 -- plumbing.
    v2 = (clean["speech6"]["none"]["hz_p05_sequence"]
          == clean["speech6"]["noop_control"]["hz_p05_sequence"]
          and clean["alice_full"]["none"]["hz_p05_sequence"]
          == clean["alice_full"]["noop_control"]["hz_p05_sequence"])

    # ---- mixtures -----------------------------------------------------------
    def cells(family, hum_of, label_of):
        out = []
        for f0 in label_of:
            for snr in probe.SNRS_DB:
                cell = {"family": family, "hum_hz": f0, "snr_db": snr,
                        "signal": f"{family}_hum{f0}Hz_{snr:g}dB", "arms": {}}
                mixed = probe.mix(speech, hum_of(f0), snr)
                ref_mag = goertzel_mag(mixed, float(f0))
                for a in ARMS:
                    x = apply_arm(mixed, a)
                    cell["arms"][a] = records(x)
                    cell["arms"][a]["hum_residual_db"] = db(
                        goertzel_mag(x, float(f0)), ref_mag)
                out.append(cell)
        return out

    fam_12harm = cells("mix12", lambda f: probe.harmonic_stack(f, 1.0),
                       probe.FUNDAMENTALS)
    fam_testvad = cells("mixtestvad",
                        lambda f: probe.harmonic_stack(
                            f, 0.3, harms=probe.TEST_VAD_HARMS, normalise=False),
                        (150, 175, 200))
    mixtures = fam_12harm + fam_testvad

    # ---- hum alone, for context ---------------------------------------------
    hum_alone = []
    for f0 in (150, 200):
        sig = probe.harmonic_stack(f0, 0.3, harms=probe.TEST_VAD_HARMS,
                                   normalise=False)
        hum_alone.append({"hum_hz": f0, "arms": {a: records(apply_arm(sig, a))
                                                  for a in ARMS}})

    # ---- score --------------------------------------------------------------
    def pool(a, rows=None):
        return [v for c in (rows if rows is not None else mixtures)
                for v in c["arms"][a]["hz_p05_sequence"]]

    def cells_with_hz(a, rows=None):
        return sum(1 for c in (rows if rows is not None else mixtures)
                   if c["arms"][a]["utts_with_hz"] > 0)

    base_cells = cells_with_hz("none")

    verdict = {}
    for a in ARMS:
        vals = pool(a)
        within = [v for v in vals if abs(cents(v, clean_med)) <= 100.0]
        r = len(within) / len(vals) if vals else 0.0
        checks, ok = {}, True
        checks["S1_recovery"] = r >= 0.90
        s2 = all(regress[c][a]["shift_median_abs_cents"] is not None
                 and regress[c][a]["shift_median_abs_cents"] < 10.0
                 for c in ("speech6", "alice_full"))
        checks["S2_clean_median"] = s2
        s3 = all(regress[c][a]["shift_max_abs_cents"] is not None
                 and regress[c][a]["shift_max_abs_cents"] < 50.0
                 for c in ("speech6", "alice_full"))
        checks["S3_clean_max"] = s3
        s4 = all(regress[c][a]["dUTT"] == 0
                 and regress[c][a]["dVOX"] is not None
                 and abs(regress[c][a]["dVOX"]) < 0.02
                 for c in ("speech6", "alice_full"))
        checks["S4_delivery"] = s4
        checks["S5_coverage"] = cells_with_hz(a) - base_cells >= 0
        verdict[a] = {
            "R_recovery": round(r, 4),
            "mixture_values": len(vals),
            "within_100c": len(within),
            "cents_vs_clean_median": {
                "median": round(statistics.median(
                    [cents(v, clean_med) for v in vals]), 1) if vals else None,
                "p10": round(sorted(cents(v, clean_med) for v in vals)[
                    int(0.10 * (len(vals) - 1))], 1) if vals else None,
                "p90": round(sorted(cents(v, clean_med) for v in vals)[
                    int(0.90 * (len(vals) - 1))], 1) if vals else None,
            },
            "cells_with_hz": cells_with_hz(a),
            "dCELLS": cells_with_hz(a) - base_cells,
            "checks": checks,
            "SHIPS": all(checks.values()),
        }

    v1 = verdict["hpf60"]["R_recovery"] >= 0.90
    v3 = any(clean[c][a]["utts_with_hz"] == 0
             for c in ("speech6", "alice_full") for a in ARMS)

    out = {
        "note": ("The §8 experiment from docs/research/IN-THE-WILD-AUDIO.md, run. "
                 "Primary metric: emitted hz_p05 on speech+hum mixtures. "
                 "Regression metric: emitted hz_p05 on clean real speech. "
                 "The notch is applied to the sample array before push() "
                 "because src/ must not be edited; see the module docstring."),
        "pre_registered_decision_rule": RULE,
        "methods": {
            "sr": SR, "dur_s": DUR_S,
            "fixtures_reused_from": "experiments/inthewild_probe.py",
            "fixtures": {
                "speech_clip": "6 s of examples/realaudio/alice01_excerpt.wav, peak-norm",
                "harm12": "harmonic_stack(f0, 1.0) -- 12 harmonics, 1/k, peak-norm",
                "testvad": ("harmonic_stack(f0, 0.3, harms=(1,2,3), normalise=False)"
                            " -- tests/test_vad.py::TestDeclaredLimits fixture"),
                "mix": "mix(speech, hum, snr_db) -- speech-RMS to hum-RMS",
                "fundamentals": probe.FUNDAMENTALS,
                "snrs_db": probe.SNRS_DB,
                "gate": "one ONNX session, reset per signal; GATE_SPEECH="
                        + str(watcher.GATE_SPEECH),
            },
            "filter": ("RBJ biquads imported from experiments/mic_filter_sim.py; "
                       "notch Q=%g, high-pass Q=0.707 x2 sections" % NOTCH_Q),
            "arms": {a: {"kind": ARMS[a][0],
                         "hz": ([] if ARMS[a][0] == "identity"
                                else list(ARMS[a][1]))}
                     for a in ARMS},
            "clean_clips": {
                "speech6": "the 6 s inthewild clip (n=4 utterances)",
                "alice_full": "the full 90 s alice01_excerpt.wav (n~20)",
            },
            "clean_med_hz": round(clean_med, 2),
        },
        "controls": {
            "reproduced": repro,
            "attenuation_db": atten,
            "attenuation_note": ("measured by pushing a unit sine through the "
                                 "same filter code, not asserted from the design"),
            "V2_noop_control_bit_identical": v2,
        },
        "clean": clean,
        "clean_regression": regress,
        "mixtures": {
            "mix12_12harm_hum": fam_12harm,
            "mix_testvad_fixture": fam_testvad,
        },
        "hum_alone_testvad": hum_alone,
        "verdict": verdict,
        "run_validity": {
            "V1_deliberately_losing_hpf60_passed": v1,
            "V2_noop_control_bit_identical": v2,
            "V3_any_arm_zero_clean_utterances": v3,
            "VOID": bool(v1 or not v2 or v3),
            "note": ("VOID means the numbers above exist but must not be used "
                     "to decide anything. V1 means the harness cannot detect "
                     "the failure it was built to detect."),
        },
        "shipped_arms": [a for a in ARMS if verdict[a]["SHIPS"]],
    }

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"clean_med = {clean_med:.2f} Hz")
    print(f"{'arm':<24} {'R':>7} {'S1':>5} {'S2':>5} {'S3':>5} {'S4':>5} {'S5':>5}  SHIP")
    for a in ARMS:
        v = verdict[a]
        print(f"{a:<24} {v['R_recovery']:>7.3f} "
              + " ".join(f"{str(v['checks'][k]):>5}" for k in
                         ("S1_recovery", "S2_clean_median", "S3_clean_max",
                          "S4_delivery", "S5_coverage"))
              + f"  {v['SHIPS']}")
    print(f"\nrun VOID: {out['run_validity']['VOID']}  "
          f"(V1={v1} V2={v2} V3={v3})")
    print(f"shipped: {out['shipped_arms'] or 'NONE'}")
    print(f"wrote {OUT}")


def _notch_all(samples, freqs):
    for f in freqs:
        samples = biquad_notch(samples, f, sr=SR, q=NOTCH_Q)
    return samples


if __name__ == "__main__":
    main()