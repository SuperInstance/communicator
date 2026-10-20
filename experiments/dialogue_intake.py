"""
dialogue_intake.py -- check whether the real-dialogue corpus is actually usable.

WHY THIS EXISTS.

`README.md` in this directory asks for eight specific scenes, because each one
answers a question that no existing corpus can (`docs/research/TWO-SPEAKER-TURNS.md`
records the three limits of the AMI reconstruction). The failure mode for a
hand-built corpus is arriving full of files that look like the thing asked for
and are not -- two channels recorded separately, so they drift; a "conversation"
that is two people taking turns for twenty minutes, which contains no overlap at
all; a scene recorded at a level that will be read as silence.

So this file does not count files. It measures the corpus and states plainly
which requirements are UNMET, because a check that cannot say "this is not
enough yet" is decoration.

WHAT IT MEASURES, AND WHY EACH ITEM IS A GATE RATHER THAN A STATISTIC.

    matched pairs   two channels from the same take, or the speaker labels are
                    fiction and every downstream number inherits the error
    level           a channel that is digitally quiet is not a recording, and a
                    silence that comes from gain rather than from a room is
                    indistinguishable from a real pause
    cross-talk      if channel B hears channel A clearly the "separate
                    microphones" design failed, and the overlap scenes are
                    contaminated at source
    signal          the fraction of audio above the noise floor, so "silence"
                    in these files can be trusted to be silence
    overlap         the one thing the AMI far-field channel cannot represent --
                    reported as a count because if it is zero the barge-in
                    scenes did not happen
    hand-offs       silences followed by the other channel speaking; this is
                    the positive class that was unrepresentable until now
    alignment       the two files were started by two people hitting a clap.
                    Until that offset is measured and removed, every
                    cross-channel number is a number about the recorders

WHAT CHANGED, AND WHY.  (2026-10-19)

The audit in `docs/research/DIALOGUE-INTAKE.md` found that this tool, as it
stood, would REJECT a recording made exactly to its own protocol, and would
report 100% overlap and 97% cross-talk on a clean recording made in an
ordinary room. Seven fixes landed; each is listed with the measurement that
motivates it, and each has a test that can fail.

    1. THE HAND-OFF RULE (P0). The old rule counted a hand-off when channel B
       was loud at the INSTANT channel A's silence ENDED. That misses the
       canonical hand-off -- A stops, B takes the floor, A stays quiet --
       because B's utterance starts a moment AFTER the silence ends and is
       over before the next silence closes. The rule now asks whether B is
       loud DURING A's silence, which is what "the other person took the
       floor" means. Measured on the protocol-strict corpus: 8 canonical
       hand-offs present, 0 counted by the old rule, 8 counted by the new one.

    2. THE CHAP (P0). The protocol asks for a clap before each scene, and
       nothing in this repo ever read it. The two files are now aligned on it
       before any cross-channel number is computed. Measured: a pair offset by
       300 ms had its hand-off count flip from 1 to 0; after the clap
       correction it is 1 again, at every offset tried (0/40/120/300/600 ms).

    3. THE EXIT CODE (P1). `main()` returned `None` and exited 0 while
       reporting "NOT YET USABLE", so a pipeline could not tell a rejected
       corpus from an accepted one. Exit 0 = USABLE, 1 = NOT YET USABLE,
       2 = NOTHING RECORDED YET.

    4. THE REPORTED-ONLY FAILURES (P1). An unmatched pair, an unreadable
       channel, a silent channel B and a low-level channel were all visible in
       the report and invisible in the verdict. They are gates now. The dead
       `ok` variable that computed the matched-pairs verdict and never used it
       is gone; the verdict it computed is what the gate now uses.

    5. THE CROSS-TALK STATISTIC (P1). The old statistic correlated the MEANS of
       the ten loudest windows, which separates clean from contaminated by
       0.0067 -- a margin too small to gate on. The new one asks the question
       the gate actually wants: of the time A is loud, how much of it is B
       loud too? If B is hearing A, that fraction goes to 1.

    6. THE LEVEL FLOOR (P2). The old floor was `energy.ABS_FLOOR` (1e-4 RMS,
       about -80 dBFS). An ordinary lived-in room is 23.5 dB ABOVE that, so
       the old floor reported a room with nobody in it as 100% signal and
       100% overlap. The floor is now measured, not assumed -- and it is
       measured from the place the PROTOCOL says to measure it from: the
       room-tone scene, which `examples/realdialogue/README.md` describes as
       "the noise floor every other number is measured against".

    7. THE LENGTH GATE (P2). The old gate wanted 600 s and the protocol's own
       eight scenes are 1230 s, so a corpus that satisfies the protocol trips
       the gate. The gate now wants the protocol's own number.

    8. THE WRITER (P3). `src/wavio.py` could read every file the protocol
       asks for and nothing in the repo could write one. It has a writer now,
       and it carries the file's REAL sample rate, which used to be
       unrecoverable -- `read_wav` resamples, so every file reported 16000.

WHAT IT STILL CANNOT DO.

It cannot judge the conversation, only the signal. A take can pass everything
here and still be two people politely alternating with no hand-off worth
measuring; that shows up as `hand_offs == 0` and is reported as such.

It cannot measure CLOCK DRIFT from one alignment point, and it says so rather
than implying a number it does not have. The clap fixes the offset AT the
clap. Two independent recorders also drift, and with one clap per pair there
is no second alignment point to see the drift with -- so the report carries a
BOUND derived from an assumed maximum clock error, and says that a bound is
what it is.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import energy, wavio  # noqa: E402

SR = 16000
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "examples/realdialogue/raw"

# energy.ABS_FLOOR is the repo's own definition of "not silent"; anything under
# it is not evidence of a pause, it is below measurement. It is the ABSOLUTE
# backstop, not the working floor: the working floor is measured (see
# `measure_noise`), because a real room is 23.5 dB above this.
SPEECH_FLOOR = energy.ABS_FLOOR

# The analysis grid: 25 ms frames at 16 kHz, the same grid the energy module
# uses.
HOP_S = 0.025
FRAME = int(SR * HOP_S)

# A silence shorter than this is inside a phrase, not a rest; a sound shorter
# than this is not speech. The repo's own constants.
MIN_SILENCE_S = energy.MIN_SILENCE_S
MIN_SPEECH_S = energy.MIN_SPEECH_S

# The protocol's eight scenes, at the durations `examples/realdialogue/README.md`
# gives them: 300 + 180 + 180 + 120 + 180 + 120 + 120 + 30 = 1230 s.
PROTOCOL_SECONDS = 1230.0

# The bar for speech, in multiples of the MEASURED noise floor. 3x is about
# 9.6 dB: far above the frame-to-frame spread of a stationary noise floor
# (about 0.3 dB), and low enough to keep a quiet speaker who is 10 dB over
# the room. Measured against both a quiet room (1e-5 RMS) and an ordinary one
# (1.5e-3 RMS); see `experiments/dialogue_intake_fixes_results.json`.
NOISE_MULTIPLE = 3.0

# How long the clap may sit before the first word, and how far either side of
# the peak the clap's body is taken to be.
CLAP_SEARCH_S = 2.0
CLAP_PRE_S = 0.030
CLAP_POST_S = 0.150

# A clap must stand this far above the measured noise floor, and must be a
# TRANSIENT rather than merely the loudest thing in the window. Measured on
# the synthetic corpus: a real clap scores about 15 on the ratio test and a
# Gaussian noise peak about 3.1. The threshold is 4.
CLAP_MIN_MULTIPLE = 3.0
CLAP_MIN_TRANSIENT_RATIO = 4.0

# The largest inter-channel offset the clap search will look for. The protocol
# says the two recorders are started by the same clap, so the offset is a
# human starting two devices; 600 ms is generous for that.
ALIGN_MAX_LAG_MS = 600

# The correlation the clap must reach before the alignment is believed.
ALIGN_MIN_R = 0.30

# Drift: the largest offset change between the start and the end of a pair
# that still leaves every gap in the pair meaningful.
DRIFT_MAX_MS = 50.0

# The clock error assumed for the drift BOUND. 100 ppm is far worse than a
# real crystal (a few ppm to tens of ppm) and is used only to turn "drift is
# unmeasurable here" into a number a reader can use. The audit's own 1.25%
# figure is also reported, because it is the magnitude that made the
# consequence visible.
BOUND_PPM = 100.0
BOUND_PPM_EXTREME = 12500.0

# Cross-talk: of the time A is loud, the fraction of that time B is loud too.
# A clean pair of separate microphones leaves this near the overlap fraction;
# a pair where B hears A drives it to 1.
CROSSTALK_MAX = 0.60

# A channel whose speech fraction is below this carries no voice.
MIN_SIGNAL_FRAC = 0.02

# The two channels of a pair may differ in level by this much before it is
# called a fault. 20 dB is a factor of ten: a mic on the table instead of at
# the mouth.
LEVEL_BALANCE_DB = 20.0

# The protocol's eight scenes, and which channels are expected to speak in
# each. Scene 7 is SPECIFIED to have a silent B and scene 8 is the room-tone
# scene, so "B is silent" cannot be an error on its own -- it has to be
# scoped to the scenes where B is supposed to speak.
SCENE_EXPECT = {
    1: (True, True),     # free conversation
    2: (True, True),     # question, then answer
    3: (True, True),     # thinking pause -- B thinks, then answers
    4: (True, False),    # turn-holding fillers -- A alone
    5: (True, True),     # backchannel
    6: (True, True),     # deliberate interruption
    7: (True, False),    # one speaker alone -- B silent throughout
    8: (False, False),   # room tone -- nobody speaks
}

PAIR_RE = re.compile(r"^(?P<scene>.+)_(?P<who>[AB])\.wav$", re.IGNORECASE)

# The protocol's scene names are ZERO-PADDED ("scene01_free", "scene08_
# roomtone"), so the number is one or two digits. A single-digit pattern
# finds NOTHING in a padded name -- "scene01" has no digit that is not
# followed by another digit -- and every scene silently fell through to the
# keyword rules, which know only three of the eight.
SCENE_NUM_RE = re.compile(r"(?:^|[^0-9])(?P<num>\d{1,2})(?![0-9])")


# ---------------------------------------------------------------- the level --

def frame_levels(samples) -> list:
    """RMS on the analysis grid."""
    n = len(samples) // FRAME
    out = []
    for i in range(n):
        s = samples[i * FRAME:(i + 1) * FRAME]
        out.append((sum(v * v for v in s) / FRAME) ** 0.5)
    return out


def median(values) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def percentile(values, q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[k]


def measure_noise(level_lists) -> dict:
    """The corpus noise floor, per channel.

    THE PROTOCOL ALREADY SAYS WHERE THIS COMES FROM: "Also record 30 s of
    nobody talking, at the end of any session. This is the noise floor every
    other number is measured against." The old intake ignored that and used
    an absolute floor 23.5 dB below an ordinary room, which is why it
    reported an empty room as two people talking over each other.

    `level_lists` is {"A": [frames...], "B": [frames...]} for the
    noise-reference scene. Returns the per-channel noise level and the bar
    that follows from it.
    """
    out = {}
    for side in ("A", "B"):
        levels = level_lists.get(side) or []
        noise = median(levels)
        out[side] = {
            "noise_rms": round(noise, 8),
            "bar_rms": round(max(SPEECH_FLOOR, NOISE_MULTIPLE * noise), 8),
        }
    return out


def fallback_noise(level_lists) -> dict:
    """No room-tone scene: estimate the floor from each channel's quiet frames.

    This is a WEAKER reference and the report says so. The 10th percentile
    of the frame levels, POOLED OVER EVERY SCENE, is a robust estimate of
    the quiet level of the recording as a whole.

    It used to be the quietest SCENE's median, which is wrong in a way that
    costs a real corpus: scene 7 of the protocol is one speaker alone, so
    its channel A is the quietest channel in the corpus for the wrong
    reason -- it is the median of a scene that is mostly SPEECH, and a bar
    derived from it is three times a voice.
    """
    out = {}
    for side in ("A", "B"):
        levels = level_lists.get(side) or []
        noise = percentile(levels, 0.10)
        out[side] = {
            "noise_rms": round(noise, 8),
            "bar_rms": round(max(SPEECH_FLOOR, NOISE_MULTIPLE * noise), 8),
        }
    return out


# ------------------------------------------------------------- segmentation --

def segment(levels, on: float, off: float) -> list:
    """Speech/silence segments over the frame grid, with hysteresis.

    The exit bar is HALF the entry bar, as in `src/energy.py`, so a decaying
    syllable tail cannot chatter the boundary.
    """
    segs = []
    if not levels:
        return segs
    state = levels[0] >= on
    start = 0.0
    for i, r in enumerate(levels[1:], start=1):
        if state:
            if r < off:
                segs.append((start, i * HOP_S, True))
                state = False
                start = i * HOP_S
        else:
            if r >= on:
                segs.append((start, i * HOP_S, False))
                state = True
                start = i * HOP_S
    segs.append((start, len(levels) * HOP_S, state))
    return segs


def merge_short(segs, min_silence_s: float, min_speech_s: float) -> list:
    """Drop silences shorter than min_silence_s and speech shorter than min_speech_s.

    A silence shorter than a rest is not a rest, so it becomes speech again;
    a sound shorter than a sound is not speech, so it becomes silence.

    NOTE THE DIFFERENCE FROM `src/energy.py`, AND WHY IT MATTERS HERE. That
    module keeps its FIRST segment whatever it is (`kept = [segs[0]]`), so a
    file that opens with a sliver of silence reports a sliver of silence. The
    measured blast radius of that defect is in
    `docs/research/SILENCE-LEADING-ARTEFACT.md` and it is NOT patched there.
    This function applies the rule to the first segment as well, because a
    leading sliver is not a rest and this tool's hand-off count would
    otherwise see one.
    """
    out = []
    for s0, s1, speech in segs:
        dur = s1 - s0
        if speech and dur < min_speech_s:
            speech = False
        elif (not speech) and dur < min_silence_s:
            speech = True
        if out and out[-1][2] == speech:
            out[-1] = (out[-1][0], s1, speech)
        else:
            out.append((s0, s1, speech))
    return out


def analyse_channel(samples, bar: float) -> dict:
    """Everything about one channel that the corpus-level measures need."""
    levels = frame_levels(samples)
    segs = merge_short(segment(levels, bar, bar / 2.0),
                       MIN_SILENCE_S, MIN_SPEECH_S)
    runs = []
    speech = []
    for s0, s1, is_speech in segs:
        if is_speech:
            speech.append((s0, s1))
        else:
            runs.append({"t0": s0, "t1": s1, "run_s": round(s1 - s0, 4)})
    n_frames = max(1, len(levels))
    mask = [False] * n_frames
    for t0, t1 in speech:
        i0 = max(0, int(t0 / HOP_S))
        i1 = min(n_frames - 1, int(t1 / HOP_S))
        for k in range(i0, i1 + 1):
            mask[k] = True
    return {
        "duration_s": round(len(samples) / SR, 2),
        "peak": round(max((abs(v) for v in samples), default=0.0), 5),
        "rms": round((sum(v * v for v in samples) / len(samples)) ** 0.5, 6)
               if samples else 0.0,
        "bar_rms": round(bar, 8),
        "signal_frame_frac": round(sum(mask) / n_frames, 4),
        "silent": not samples or max((abs(v) for v in samples), default=0.0) == 0.0,
        "silence_runs": runs,
        "speech_intervals": speech,
        "mask": mask,
        "frames": n_frames,
    }


# --------------------------------------------------------------- alignment --

def envelope_ms(samples, sr: int = SR, hop_ms: float = 1.0):
    """A 1 ms energy envelope. The clap is a transient; this is how it is seen."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float64)
    hop = max(1, int(round(hop_ms * sr / 1000.0)))
    n = len(x) // hop
    if n == 0:
        return np.zeros(0)
    frames = x[:n * hop].reshape(n, hop)
    return np.sqrt((frames ** 2).mean(axis=1))


def cross_correlate_lag(ea, eb, max_lag_ms: int = ALIGN_MAX_LAG_MS) -> dict:
    """The lag (in ms) at which two 1 ms envelopes match best.

    Sign convention, stated because it is the thing that gets a correction
    applied the wrong way round: a NEGATIVE lag means channel B is LATE
    relative to channel A. Advancing B by the detected lag (a negative shift,
    i.e. earlier) undoes the offset.
    """
    import numpy as np
    m = min(len(ea), len(eb))
    if m < 50:
        return {"lag_ms": 0, "peak_r": 0.0, "n_lags": 0,
                "note": "not enough material to correlate"}
    ea = np.asarray(ea[:m], dtype=np.float64)
    eb = np.asarray(eb[:m], dtype=np.float64)
    ea = ea - np.median(ea)
    eb = eb - np.median(eb)
    ea = ea / (float(np.linalg.norm(ea)) or 1.0)
    eb = eb / (float(np.linalg.norm(eb)) or 1.0)
    best_lag, best_r = 0, -2.0
    n_lags = 0
    for lag in range(-max_lag_ms, max_lag_ms + 1):
        if lag >= 0:
            x, y = ea[lag:], eb[:m - lag]
        else:
            x, y = ea[:m + lag], eb[-lag:]
        if len(x) < 50:
            continue
        r = float(np.dot(x, y))
        n_lags += 1
        if r > best_r:
            best_r, best_lag = r, lag
    return {"lag_ms": best_lag, "peak_r": round(best_r, 4), "n_lags": n_lags}


def clap_offset_ms(a_samples, b_samples, search_s: float = CLAP_SEARCH_S) -> dict:
    """The inter-channel offset, from the clap the protocol asks for.

    The clap is the loudest transient in the first seconds of BOTH files, so
    the 1 ms energy envelopes of the two files can be cross-correlated over a
    range of lags and the peak is the offset. This is the alignment the
    protocol describes and that no code in this repo performed.
    """
    env_a = envelope_ms(a_samples)[:int(search_s * 1000)]
    env_b = envelope_ms(b_samples)[:int(search_s * 1000)]
    return cross_correlate_lag(env_a, env_b)


def clap_region(samples, noise_rms: float = 0.0,
                search_s: float = CLAP_SEARCH_S):
    """Where the clap is, in seconds, as (t0, t1, peak_level) or None.

    The clap is the loudest thing in the first `search_s` of the file, by
    protocol -- it is the alignment point and it is clapped loudly, before
    anyone speaks.

    BUT "loudest thing" IS NOT ENOUGH, and the room-tone scene is what
    proves it: in a file that is nothing but room noise, the loudest thing in
    the first two seconds is a noise peak, and reading it as a clap produces
    a confident alignment between two files that were never aligned. So the
    clap must be a TRANSIENT -- its peak must stand well above the rest of
    the material, not merely above the floor.

    The test is the ratio of the peak to the 90th percentile of the same
    window. Measured on the synthetic corpus: a real clap scores about 15,
    and a Gaussian noise peak scores about 3.1 -- the difference between a
    transient and the top of a distribution. The threshold is 4.
    """
    env = envelope_ms(samples)[:int(search_s * 1000)]
    if len(env) == 0:
        return None
    k = int(env.argmax())
    peak = float(env[k])
    if peak <= SPEECH_FLOOR:
        return None
    if peak < CLAP_MIN_MULTIPLE * noise_rms:
        return None
    rest = float(np_percentile(env, 90))
    if rest > 0 and peak / rest < CLAP_MIN_TRANSIENT_RATIO:
        return None
    t = k / 1000.0
    return (max(0.0, t - CLAP_PRE_S), t + CLAP_POST_S, round(peak, 6))


def np_percentile(values, q: float) -> float:
    """The q-th percentile of a numpy array, without importing numpy here."""
    import numpy as np
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def shift_samples(samples, ms: float):
    """Delay a signal by `ms` (positive = later). Negative ADVANCES it."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float64)
    d = int(round(ms * SR / 1000.0))
    if d == 0:
        return x.copy()
    out = np.zeros(len(x))
    if d > 0:
        if d < len(x):
            out[d:] = x[:len(x) - d]
    else:
        adv = -d
        if adv < len(x):
            out[:len(x) - adv] = x[adv:]
    return out


def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


# ------------------------------------------------------------- the measures --

def cross_talk(a_stats, b_stats) -> dict:
    """Of the time A is loud, how much of it is B loud too?

    THE STATISTIC THE GATE WANTS. It does not depend on the two waveforms
    resembling each other at all: if channel B is hearing channel A, then B
    is loud whenever A is loud, and this fraction goes to 1. On separate
    microphones the fraction is the overlap fraction, which is small.

    The old statistic correlated the MEANS of the ten loudest windows, which
    asks a different question -- how similar are two loud moments -- and
    separates a contaminated corpus from a clean one by 0.0067 on the same
    signals where this one separates them by far more.
    """
    a_mask = a_stats["mask"]
    b_mask = b_stats["mask"]
    n = min(len(a_mask), len(b_mask))
    a_loud = both = 0
    for k in range(n):
        if a_mask[k]:
            a_loud += 1
            if b_mask[k]:
                both += 1
    frac = (both / a_loud) if a_loud else None
    return {
        "b_loud_when_a_loud": round(frac, 4) if frac is not None else None,
        "a_loud_frames": a_loud,
        "both_loud_frames": both,
        "note": None if frac is not None else "channel A never got loud",
    }


def legacy_corr(a_samples, b_samples) -> dict:
    """The OLD cross-talk statistic, kept for comparison on the same data.

    It correlates the MEANS of the ten loudest simultaneous windows. Kept
    because the report should show what the replacement replaced, not only
    what it now says.
    """
    n = min(len(a_samples), len(b_samples))
    W = 1600
    wins = []
    for i in range(0, n - W, W):
        wa = a_samples[i:i + W]
        wb = b_samples[i:i + W]
        ra = (sum(v * v for v in wa) / W) ** 0.5
        rb = (sum(v * v for v in wb) / W) ** 0.5
        if ra > SPEECH_FLOOR and rb > SPEECH_FLOOR:
            wins.append((ra + rb, wa, wb))
    if len(wins) < 3:
        return {"windows": len(wins), "corr": None}
    wins.sort(reverse=True, key=lambda w: w[0])
    top = wins[:10]
    ma = [sum(v for v in w[1]) / W for w in top]
    mb = [sum(v for v in w[2]) / W for w in top]
    num = sum((x - sum(ma) / len(ma)) * (y - sum(mb) / len(mb)) for x, y in zip(ma, mb))
    da = (sum((x - sum(ma) / len(ma)) ** 2 for x in ma)) ** 0.5
    db = (sum((y - sum(mb) / len(mb)) ** 2 for y in mb)) ** 0.5
    return {"windows": len(wins),
            "corr": round(num / (da * db), 4) if da and db else None}


def count_hand_offs(a_stats, b_stats, clap) -> dict:
    """Hand-offs: silences on A during which B took the floor.

    THE RULE, AND WHY THE OLD ONE MISSED THE CANONICAL CASE. The old rule
    asked whether B was loud at the instant A's silence ENDED. A canonical
    hand-off -- A stops, B takes the floor, A stays quiet -- has B starting
    just after that instant and finishing before the next silence closes, so
    the old rule scored it ZERO. It scored a BARGE-BACK instead (B is still
    talking when A's silence ends), which is the rare shape and not the
    common one.

    The rule now asks whether B is loud at ANY point DURING A's silence.

    Two conditions, each from a measurement:
      - the silence must be at least MIN_SILENCE_S long, because a 20 ms
        envelope dip inside a word is not a rest (the old rule counted one);
      - the clap must be excluded, because it is loud in BOTH channels by
        construction and sits inside the lead-in silence -- that is exactly
        one false hand-off per scene.
    """
    runs = a_stats["silence_runs"]
    b_mask = b_stats["mask"]
    n_frames = len(b_mask)
    hand_offs = 0
    silences = 0
    details = []
    for r in runs:
        if r["run_s"] < MIN_SILENCE_S:
            continue
        silences += 1
        if clap and overlaps(r["t0"], r["t1"], clap[0], clap[1]):
            continue                       # the clap is not a hand-off
        k0 = max(0, int(r["t0"] / HOP_S))
        k1 = min(n_frames - 1, int(r["t1"] / HOP_S))
        loud = False
        for k in range(k0, k1 + 1):
            if b_mask[k]:
                if clap and overlaps(k * HOP_S, (k + 1) * HOP_S,
                                     clap[0], clap[1]):
                    continue               # B loud only at the clap
                loud = True
                break
        if loud:
            hand_offs += 1
            details.append({"t0": round(r["t0"], 3), "t1": round(r["t1"], 3),
                            "run_s": r["run_s"]})
    return {"hand_offs": hand_offs, "silences": silences, "detail": details}


def both_talking_frac(a_stats, b_stats) -> dict:
    """Fraction of time both channels are loud: the overlap budget."""
    a_mask = a_stats["mask"]
    b_mask = b_stats["mask"]
    n = min(len(a_mask), len(b_mask))
    both = either = 0
    for k in range(n):
        la, lb = a_mask[k], b_mask[k]
        both += la and lb
        either += la or lb
    return {
        "frames": n,
        "both_loud_frac": round(both / n, 4) if n else 0.0,
        "either_loud_frac": round(either / n, 4) if n else 0.0,
        "a_loud_frac": round(sum(a_mask[:n]) / n, 4) if n else 0.0,
        "b_loud_frac": round(sum(b_mask[:n]) / n, 4) if n else 0.0,
    }


def drift_report(clap_lag_ms, duration_s) -> dict:
    """What can be said about clock drift between two recorders.

    THE HONEST ANSWER, WHICH IS MOSTLY A BOUND. The clap gives ONE alignment
    point. Two independent recorders also drift, so the offset measured at the
    clap is not the offset in the middle of the scene, and the gap errors that
    follow are not small: measured with a 1.25% time-base error, a 0.40 s gap
    is read as 0.21 s.

    With one alignment point there is nothing to MEASURE the drift against --
    the two channels speak different words, so their envelopes are
    uncorrelated everywhere except at the clap. So the report carries a BOUND
    derived from an assumed maximum clock error, and says that it is a bound.
    """
    bound_ms = BOUND_PPM * 1e-6 * duration_s * 1000.0
    extreme_ms = BOUND_PPM_EXTREME * 1e-6 * duration_s * 1000.0
    return {
        "measurable_from_one_alignment_point": False,
        "measured_ms": None,
        "clap_offset_ms": clap_lag_ms,
        "bound_ms_at_%gppm" % BOUND_PPM: round(bound_ms, 2),
        "bound_ms_at_%gppm" % BOUND_PPM_EXTREME: round(extreme_ms, 2),
        "method": ("the clap is the only alignment point in the pair, so the "
                   "drift cannot be measured; the bound above is what an "
                   "assumed clock error of that size would produce by the end "
                   "of the scene"),
    }


def scene_number(scene: str):
    m = SCENE_NUM_RE.search(scene)
    return int(m.group("num")) if m else None


def scene_expect(scene: str):
    """Which channels are expected to speak in this scene.

    Returns (a_speaks, b_speaks), each True, False, or None for "not known",
    so an unrecognised scene name is never gated on a silence it was never
    supposed to contain.
    """
    num = scene_number(scene)
    if num is not None and num in SCENE_EXPECT:
        return SCENE_EXPECT[num]
    low = scene.lower()
    if "roomtone" in low or "room_tone" in low or "room-tone" in low:
        return (False, False)
    if "alone" in low or "solo" in low:
        return (True, False)
    if "filler" in low:
        return (True, False)
    return (None, None)


# ------------------------------------------------------------------- main --

def read_pair(a_path: Path, b_path: Path):
    """Read both channels of a pair, with their REAL sample rates."""
    wav_a = wavio.read_wav(str(a_path))
    wav_b = wavio.read_wav(str(b_path))
    rate_a = getattr(wav_a, "source_sample_rate", None) or wav_a.sample_rate
    rate_b = getattr(wav_b, "source_sample_rate", None) or wav_b.sample_rate
    return wav_a.samples, wav_b.samples, rate_a, rate_b


def analyse_pair(scene: str, a_samples, b_samples, rate_a, rate_b,
                 bars: dict, noise_ref: dict) -> dict:
    """Everything measured about one matched pair."""
    entry = {"scene": scene, "has_A": True, "has_B": True,
             "source_rates_hz": {"A": rate_a, "B": rate_b}}

    # --- alignment, BEFORE any cross-channel number -----------------------
    det = clap_offset_ms(a_samples, b_samples)
    lag = det["lag_ms"]
    b_aligned = shift_samples(b_samples, lag).tolist()   # python floats
    clap = clap_region(a_samples, noise_ref["A"]["noise_rms"])
    clap_b = clap_region(b_aligned, noise_ref["B"]["noise_rms"])
    aligned = det["peak_r"] >= ALIGN_MIN_R
    if clap and clap_b:
        # Both channels have a clap: they must be the same clap.
        note = (None if aligned else
                "the two channels have a clap each, but they do not match: "
                "peak correlation %s is below %s, so the pair was not started "
                "together and every cross-channel number below is a number "
                "about two unsynchronised clocks" % (det["peak_r"], ALIGN_MIN_R))
    elif clap or clap_b:
        aligned = False
        note = ("the clap is in one channel only, so the two files were not "
                "started together")
    else:
        aligned = False
        note = ("no clap in either channel: the protocol asks for one before "
                "each scene, and without it the pair cannot be aligned")
    entry["alignment"] = {
        "clap_offset_ms": lag,
        "peak_r": det["peak_r"],
        "applied": aligned,
        "clap_region_s": [round(clap[0], 3), round(clap[1], 3)] if clap else None,
        "clap_peak_rms": clap[2] if clap else None,
        "clap_in_both": bool(clap and clap_b),
        "note": note,
    }

    sa = analyse_channel(a_samples, bars["A"])
    sb = analyse_channel(b_aligned, bars["B"])
    sa["source_rate_hz"] = rate_a
    sb["source_rate_hz"] = rate_b
    entry["A"] = {k: v for k, v in sa.items() if k not in ("mask",)}
    entry["B"] = {k: v for k, v in sb.items() if k not in ("mask",)}

    entry["levels"] = both_talking_frac(sa, sb)
    entry["crosstalk"] = cross_talk(sa, sb)
    entry["crosstalk"]["legacy_statistic"] = legacy_corr(a_samples, b_aligned)

    ho = count_hand_offs(sa, sb, clap)
    entry["hand_offs"] = ho["hand_offs"]
    entry["silences"] = ho["silences"]
    entry["hand_off_detail"] = ho["detail"]

    dur = min(sa["duration_s"], sb["duration_s"])
    entry["drift"] = drift_report(lag, dur)
    return entry


def main(argv=None) -> int:
    # `--raw` exists so the checker itself can be tested. A gate that has only
    # ever been observed rejecting things is not known to work: it could be
    # rejecting everything. `tests/test_dialogue_intake.py` builds synthetic
    # corpora that must PASS and synthetic ones that must fail, and this
    # argument is what lets it point at either.
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default=str(RAW),
                    help="directory of recorded _A/_B pairs")
    args = ap.parse_args(argv)
    raw = Path(args.raw)
    report = {"raw_dir": str(raw), "exists": raw.is_dir(), "pairs": [],
              "unmatched": [], "gates": {}, "protocol_seconds": PROTOCOL_SECONDS}

    if not raw.is_dir():
        report["gates"]["status"] = "NOTHING RECORDED YET"
        report["gates"]["detail"] = (
            f"{raw} does not exist. See examples/realdialogue/README.md for the "
            "eight scenes and the two-microphone rule.")
        print(json.dumps(report, indent=2))
        return 2

    found = sorted(p for p in raw.glob("*.wav"))
    scenes = {}
    for p in found:
        m = PAIR_RE.match(p.name)
        if not m:
            report["unmatched"].append(p.name)
            continue
        scenes.setdefault(m.group("scene"), {})[m.group("who").upper()] = p

    # ---- the noise reference, measured before anything else -------------
    # The protocol says the room-tone scene is the noise floor every other
    # number is measured against, so it is read FIRST and everything else is
    # measured against it.
    noise_ref = None
    levels_by_scene = {}
    for scene, sides in sorted(scenes.items()):
        a, b = sides.get("A"), sides.get("B")
        if not a or not b:
            continue
        try:
            sa, sb, _, _ = read_pair(a, b)
        except Exception:                                  # noqa: BLE001
            continue
        levels_by_scene[scene] = {"A": frame_levels(sa), "B": frame_levels(sb)}
    if levels_by_scene:
        pooled = {"A": [], "B": []}
        for lv in levels_by_scene.values():
            for side in ("A", "B"):
                pooled[side].extend(lv[side])
        room_tone = [s for s in levels_by_scene if scene_number(s) == 8]
        if room_tone:
            ref_scene = room_tone[0]
            noise_ref = measure_noise(levels_by_scene[ref_scene])
            noise_ref["source"] = ("the room-tone scene (%s), which the "
                                   "protocol names as the noise floor every "
                                   "other number is measured against"
                                   % ref_scene)
        else:
            ref_scene = None
            noise_ref = fallback_noise(pooled)
            noise_ref["source"] = (
                "NO room-tone scene: the floor is the 10th percentile of the "
                "frame levels pooled over every scene, which is a weaker "
                "reference -- the protocol asks for a room-tone recording "
                "precisely so this is not needed")
        # THE HIGHER OF THE TWO WINS, per channel. If the room-tone scene was
        # recorded at a lower gain than the rest of the session, its own
        # estimate is too low and the bar derived from it would let the other
        # scenes' room tone read as speech. The pooled estimate does not have
        # that failure mode, and taking the larger of the two is the
        # conservative choice: a slightly high bar misses a quiet speaker,
        # a slightly low bar invents speech out of a room.
        pooled_ref = fallback_noise(pooled)
        for side in ("A", "B"):
            if pooled_ref[side]["noise_rms"] > noise_ref[side]["noise_rms"]:
                noise_ref[side] = dict(pooled_ref[side])
        report["noise_reference"] = {"scene": ref_scene, **noise_ref}
        report["noise_reference"]["pooled_10th_percentile_rms"] = {
            s: pooled_ref[s]["noise_rms"] for s in ("A", "B")}
    else:
        noise_ref = {s: {"noise_rms": 0.0, "bar_rms": SPEECH_FLOOR}
                     for s in ("A", "B")}
        report["noise_reference"] = None

    # ---- the pairs -------------------------------------------------------
    total_s = 0.0
    hand_offs = 0
    silences = 0
    unmet = []
    for scene, sides in sorted(scenes.items()):
        a, b = sides.get("A"), sides.get("B")
        if not a or not b:
            report["pairs"].append({
                "scene": scene, "has_A": a is not None, "has_B": b is not None,
                "problem": "unmatched pair"})
            unmet.append(f"{scene}: unmatched pair (missing "
                         f"{'B' if a else 'A'})")
            continue
        try:
            sa, sb, rate_a, rate_b = read_pair(a, b)
        except Exception as exc:                          # noqa: BLE001
            report["pairs"].append({
                "scene": scene, "has_A": True, "has_B": True,
                "problem": "unreadable: %s: %s" % (type(exc).__name__, exc)})
            unmet.append(f"{scene}: unreadable channel")
            continue

        # A pair with no room-tone scene of its own uses the corpus reference.
        bars = {"A": noise_ref["A"]["bar_rms"], "B": noise_ref["B"]["bar_rms"]}
        entry = analyse_pair(scene, sa, sb, rate_a, rate_b, bars, noise_ref)
        report["pairs"].append(entry)
        if entry.get("problem"):
            unmet.append(f"{scene}: {entry['problem']}")
            continue
        total_s += min(entry["A"]["duration_s"], entry["B"]["duration_s"])
        hand_offs += entry.get("hand_offs", 0)
        silences += entry.get("silences", 0)

        want_a, want_b = scene_expect(scene)
        entry["A_expected_to_speak"] = want_a
        entry["B_expected_to_speak"] = want_b

        # -- the per-pair gates ------------------------------------------
        for side, want in (("A", want_a), ("B", want_b)):
            stats = entry[side]
            if stats.get("silent"):
                unmet.append(f"{scene}: channel {side} is digitally silent")
            elif want is True and stats.get("signal_frame_frac", 0.0) < \
                    MIN_SIGNAL_FRAC:
                unmet.append(f"{scene}: channel {side} carries almost no "
                             f"signal (frame fraction "
                             f"{stats.get('signal_frame_frac')}) but the "
                             f"protocol has {side} speaking in this scene")
            elif want is False and stats.get("signal_frame_frac", 1.0) >= \
                    MIN_SIGNAL_FRAC:
                unmet.append(f"{scene}: channel {side} carries signal but the "
                             f"protocol has {side} SILENT in this scene "
                             f"(frame fraction "
                             f"{stats.get('signal_frame_frac')})")
        # The two channels of one take should be at comparable levels: a mic
        # on the table instead of at the mouth is 20 dB down.
        rms_a = entry["A"]["rms"] or 1e-12
        rms_b = entry["B"]["rms"] or 1e-12
        imbalance_db = 20.0 * math.log10(rms_a / rms_b)
        entry["level_balance_db"] = round(imbalance_db, 2)
        if abs(imbalance_db) > LEVEL_BALANCE_DB:
            unmet.append(f"{scene}: the two channels differ by "
                         f"{imbalance_db:.1f} dB, past the "
                         f"{LEVEL_BALANCE_DB} dB limit")
        ct = entry["crosstalk"].get("b_loud_when_a_loud")
        if ct is not None and ct > CROSSTALK_MAX:
            unmet.append(f"{scene}: channel B hears channel A "
                         f"(b_loud_when_a_loud = {ct} > {CROSSTALK_MAX})")
        al = entry["alignment"]
        if not al.get("applied") and al.get("clap_in_both"):
            unmet.append(f"{scene}: {al.get('note')}")

    # -- the corpus gates --------------------------------------------------
    # "matched pairs" means pairs with BOTH channels. An unmatched pair is
    # still listed in `pairs`, but it is not a matched pair and counting it
    # as one would report a corpus as more complete than it is.
    matched_pairs = sum(1 for p in report["pairs"]
                         if p.get("has_A") and p.get("has_B"))
    if not matched_pairs:
        unmet.append("no matched _A/_B pairs yet")
    if report["unmatched"]:
        unmet.append(f"{len(report['unmatched'])} file(s) not matching NAME_A.wav")
    if total_s < PROTOCOL_SECONDS:
        unmet.append(f"only {total_s:.0f}s of audio; the protocol's eight "
                     f"scenes are {PROTOCOL_SECONDS:.0f}s")
    if matched_pairs and hand_offs == 0:
        unmet.append("zero hand-offs: no silence is followed by the other "
                     "speaker, so the positive class is still absent")

    gates = report["gates"]
    gates["matched_pairs"] = matched_pairs
    gates["total_audio_s"] = round(total_s, 1)
    gates["silences_seen"] = silences
    gates["hand_offs_seen"] = hand_offs
    gates["unmatched_files"] = report["unmatched"]
    gates["unmet"] = unmet
    gates["status"] = "USABLE" if not unmet else "NOT YET USABLE"
    print(json.dumps(report, indent=2))
    return 0 if not unmet else 1


def math_log10(x: float) -> float:
    import math
    return math.log10(x)


if __name__ == "__main__":
    sys.exit(main())
