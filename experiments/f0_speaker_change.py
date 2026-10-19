"""
f0_speaker_change.py -- can a change of speaker be detected as a change of F0?

MOTIVATION.

`docs/research/TURN-ROUTES.md` ends at a wall: the corpus is single-speaker, so
every acoustic turn test scores ~100% interrupt and the positive class (a
speaker yielding the floor) is unrepresentable. The obvious next move is "get a
second signal", and the cheapest second signal in an audio stream is pitch.

The hypothesis is attractive and testable:

    H0   a speaker change produces a large F0 jump, so longer pauses --
         where a hand-off is more likely -- should jump more than short
         (same-speaker) ones.

WHY THIS FILE IS NOT JUST "measure the jump".

Two failure modes had to be closed before any number means anything.

1. RATIO WITHOUT ABSOLUTE HZ IS NOT A VOICE CHANGE.
   |f2-f1|/f1 is dimensionless. A genuine adult-to-adult hand-off might move
   110 Hz -> 190 Hz (73%). A pitch tracker jumping one octave produces
   110 -> 220 (100%) from the SAME speaker with the SAME throat. The two are
   indistinguishable in ratio and opposite in meaning. Every observation here
   therefore carries both frequencies, and jumps are classified -- in
   particular the octave and subharmonic bands, which are instrument
   artifacts rather than evidence about anybody's larynx.

2. A HYPOTHESIS WITH NO NULL CANNOT FAIL.
   "Jumps look big" is not a finding; the corpus's intonation alone may
   produce equally big jumps between any two phrases of one speaker. The
   decisive control is WITHIN-UTTERANCE: the last second of an utterance
   compared to the first second of the SAME utterance. The speaker is
   identical by construction -- this file cannot be wrong about that -- and
   the window geometry is deliberately matched to the across-pause measure.
   Every number in that control is pure intonation.

   If across-pause jumps are no larger than within-utterance jumps, then F0
   is measuring phrasing, and no threshold of it can ever mean "the speaker
   changed".

WHAT IS MEASURED.

Per cell, over every pause the watcher reports (0.20 s < gap < 4.0 s, so
neither a word boundary nor a recording edit):

    bands            median |f2-f1|/f1 for short (<0.30 s), mid (0.30-0.60 s)
                     and long (>0.60 s) gaps, with n and a bootstrap CI --
                     the long bands are small (n = 6..18) and a median with
                     no interval invites reading noise as a trend
    control          the same statistic computed within single utterances
    jump_classes     how many jumps land in the octave / subharmonic bands
    spearman         rank correlation between gap length and jump size,
                     with a permutation p-value, because the raw medians
                     above can be monotone by luck

WHAT THIS FILE CANNOT SHOW.

It cannot find a speaker change that is not in the corpus: every file here is
one continuous single-speaker LibriVox reading, so a true hand-off may occur
zero times here. A null result is therefore "F0 does not do this on this
corpus", NOT "F0 can never detect a speaker change". What it CAN do is
falsify the specific claim that F0 jump size tracks pause length -- and that
claim is testable with the data present, so it is tested.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pitch, vad, wavio, watcher  # noqa: E402

SR = 16000
CORPUS = Path(__file__).resolve().parents[1] / "examples/realaudio"

FILES = {
    "narration": CORPUS / "alice01_excerpt.wav",
    "essay": CORPUS / "corpus/essay_perpetual_peace_librivox.wav",
    "dialogue": CORPUS / "corpus/shortstory_shortstory058_1405_librivox.wav",
    "shakespeare": CORPUS / "corpus/shakespeare_midsummernights_dream_1201_librivox.wav",
    "children": CORPUS / "corpus/children_english_fairy_tales_joy_librivox.wav",
    "oration": CORPUS / "corpus/oration_jfks19620912.wav",
}
FILES["shakespeare"] = CORPUS / "corpus/shakespeare_midsummernightsdream_1201_librivox.wav"

GAP_MIN, GAP_MAX = 0.20, 4.0
WIN_S = 1.0            # seconds of audio each F0 window covers
FRAME, HOP = 800, 160  # 50 ms / 10 ms -- src/pitch.py's own geometry

# Artifact bands, keyed on jump = |f2-f1|/f1 -- a RELATIVE MAGNITUDE, not a
# ratio. An octave (f2 = 2*f1) is jump = 1.00; a subharmonic (f2 = 0.5*f1) is
# jump = 0.50. Bands written as [1.8, 2.2] would miss every real octave error
# and file it as merely "large", which is precisely the confusion this file
# exists to prevent -- an earlier version of this script had that band, and
# would have reported the shakespeare p90 of 1.0454 (a near-perfect doubling)
# as ordinary movement rather than as a tracker artifact.
#
# The bands are wide enough to absorb intonation and tight enough not to
# swallow it: a genuine speaker change is not a power of two.
OCTAVE = (0.85, 1.15)
SUBHARMONIC = (0.42, 0.58)


def median_f0(samples, i0, i1):
    """Median voiced F0 over [i0, i1), or None if the window is not usable.

    A median over frames, not a mean: one bad frame should not move the
    answer, and voiced-frame counts here are routinely ~50.
    """
    if i0 < 0 or i1 > len(samples) or i1 - i0 < FRAME:
        return None
    tracker = pitch.PitchTracker(SR, frame_len=FRAME, hop=HOP)
    hz = [f.hz for f in tracker.track(samples[i0:i1]) if f.hz is not None]
    if len(hz) < 8:
        return None
    return sorted(hz)[len(hz) // 2]


def utterances(samples):
    """Utterance spans, using the same adaptive watcher the system runs."""
    w = watcher.Watcher(SR, adaptive=True)
    spans = []
    for i in range(0, len(samples), 1600):
        for e in w.push(samples[i:i + 1600]):
            if e.kind == "utterance":
                spans.append((e.t0, e.t1))
    return spans


def classify(jump):
    """Name a jump. Bins are disjoint so every jump gets exactly one name."""
    if OCTAVE[0] <= jump <= OCTAVE[1]:
        return "octave"
    if SUBHARMONIC[0] <= jump <= SUBHARMONIC[1]:
        return "subharmonic"
    if jump <= 0.20:
        return "small"
    if jump <= 0.50:
        return "moderate"
    return "large"


def bootstrap_ci(values, trials=2000, seed=12345):
    """Percentile CI on the median. Fixed seed: this number is cited."""
    import random
    if len(values) < 4:
        return None
    rng = random.Random(seed)
    n = len(values)
    meds = []
    for _ in range(trials):
        s = sorted(values[rng.randrange(n)] for _ in range(n))
        meds.append(s[len(s) // 2])
    meds.sort()
    return [round(meds[int(trials * 0.025)], 4), round(meds[int(trials * 0.975)], 4)]


def rank(xs):
    """Average ranks, ties shared."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(pairs):
    if len(pairs) < 5:
        return None, None
    rx, ry = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
    n = len(pairs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return None, None
    return num / (dx * dy), None


def permutation_p(pairs, obs, trials=20000, seed=999):
    """One-sided p for rho >= obs, by shuffling the pairing.

    Spearman assumes monotonicity; permutation only assumes exchangeability,
    which is all a rank correlation actually needs.
    """
    import random
    if len(pairs) < 5:
        return None
    rng = random.Random(seed)
    ys = [p[1] for p in pairs]
    hits = 0
    for _ in range(trials):
        sh = ys[:]
        rng.shuffle(sh)
        r, _ = spearman(list(zip([p[0] for p in pairs], sh)))
        if r is not None and r >= obs:
            hits += 1
    return (hits + 1) / (trials + 1)


def band_of(gap):
    if gap < 0.30:
        return "short"
    return "mid" if gap < 0.60 else "long"


def speech_runs(samples, min_s=2.2):
    """Maximal stretches of >=2.2 s of continuous speech, from the gate.

    The control cannot be built on watcher utterances. Measured on this corpus,
    the median utterance is 0.270 s and the LONGEST is 1.220 s, so there is not
    one utterance anywhere in six files long enough to hold two non-overlapping
    1 s windows. An earlier version of this file built its control that way and
    silently returned zero samples -- a control that cannot run is not a
    control, and it looked identical to "the control showed no difference".

    The gate is the right instrument because it is a speech/no-speech detector
    rather than a segmenter: it does not chop on breaths, so it exposes the
    long continuous stretches that a within-speaker control needs.
    """
    try:
        gate = vad.SileroGate()
    except Exception:
        return []
    flags = []
    for i in range(0, len(samples), 1600):
        flags.extend(p >= watcher.GATE_SPEECH
                     for p in gate.push(samples[i:i + 1600]))
    runs, start = [], None
    for i, on in enumerate(flags):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    need = int(min_s / (FRAME_SAMPLES_HZ))
    return [(a * FRAME_SAMPLES_HZ, b * FRAME_SAMPLES_HZ)
            for a, b in runs if (b - a) >= need]


FRAME_SAMPLES_HZ = 512 / SR   # one gate frame, in seconds


def within_speaker_control(samples, runs):
    """Median |f2-f1|/f1 between the first and last 1 s of one unbroken
    stretch of speech. The speaker is identical by construction -- nobody
    changes voice inside a continuous utterance."""
    out = []
    for t0, t1 in runs:
        fa = median_f0(samples, int(t0 * SR), int((t0 + WIN_S) * SR))
        fb = median_f0(samples, int((t1 - WIN_S) * SR), int(t1 * SR))
        if fa and fb:
            out.append(abs(fb - fa) / fa)
    return out


def observe(samples, spans):
    """Per-pause F0 records. This is the expensive part, so it is separated
    from the statistics and cached: re-deriving a band or a p-value should not
    cost another pass of YIN."""
    records = []
    for i in range(len(spans) - 1):
        _, t1 = spans[i]
        t2, _ = spans[i + 1]
        gap = t2 - t1
        if gap < GAP_MIN or gap > GAP_MAX:
            continue
        f1 = median_f0(samples, int((t1 - WIN_S) * SR), int(t1 * SR))
        f2 = median_f0(samples, int(t2 * SR), int((t2 + WIN_S) * SR))
        if not f1 or not f2:
            continue
        records.append({"t1": round(t1, 4), "t2": round(t2, 4),
                        "gap": round(gap, 4), "f1": round(f1, 2),
                        "f2": round(f2, 2), "jump": abs(f2 - f1) / f1})
    control = within_speaker_control(samples, speech_runs(samples))
    return records, control


def analyse(cell, records, control):
    bands = {"short": [], "mid": [], "long": []}
    classes = {}
    pairs, f1s, f2s = [], [], []
    for r in records:
        bands[band_of(r["gap"])].append(r["jump"])
        pairs.append((r["gap"], r["jump"]))
        f1s.append(round(r["f1"], 1))
        f2s.append(round(r["f2"], 1))
        classes[classify(r["jump"])] = classes.get(classify(r["jump"]), 0) + 1

    out = {"cell": cell, "n_pauses": len(records), "n_control": len(control),
           "jump_classes": classes,
           "f1_hz_median": round(sorted(f1s)[len(f1s) // 2], 1) if f1s else None,
           "f2_hz_median": round(sorted(f2s)[len(f2s) // 2], 1) if f2s else None}
    for k in ("short", "mid", "long"):
        v = bands[k]
        out[k] = {"n": len(v),
                  "med": round(sorted(v)[len(v) // 2], 4) if v else None,
                  "ci95": bootstrap_ci(v)}
    if control:
        cs = sorted(control)
        out["control"] = {"n": len(cs), "med": round(cs[len(cs) // 2], 4),
                          "ci95": bootstrap_ci(control)}
    rho, _ = spearman(pairs)
    if rho is not None:
        out["spearman_gap_vs_jump"] = {"rho": round(rho, 4),
                                       "p": round(permutation_p(pairs, rho), 4),
                                       "n": len(pairs)}
    return out


CACHE = Path(__file__).with_name("f0_observations.json")


def load_cache():
    if CACHE.is_file():
        try:
            return json.loads(CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    cells = [c for c in FILES if c in argv] or list(FILES)
    reuse = "--reuse" in sys.argv
    cache = load_cache()

    results, dirty = [], False
    for cell in cells:
        p = Path(FILES[cell])
        if not p.is_file():
            results.append({"cell": cell, "error": "absent"})
            continue
        if reuse and cell in cache:
            rec, ctl = cache[cell]["records"], cache[cell]["control"]
        else:
            samples = wavio.read_wav(str(p)).samples
            rec, ctl = observe(samples, utterances(samples))
            cache[cell] = {"records": rec, "control": ctl}
            dirty = True
        results.append(analyse(cell, rec, ctl))
    if dirty:
        CACHE.write_text(json.dumps(cache))

    pooled = [r for r in results if "error" not in r]
    ctl = sorted(r["control"]["med"] for r in pooled if "control" in r)
    verdicts = {
        "control_median": round(ctl[len(ctl) // 2], 4) if ctl else None,
        "cells_long_exceed_control": sum(
            1 for r in pooled
            if r.get("long", {}).get("med") is not None and "control" in r
            and r["long"]["med"] > r["control"]["med"]),
        "cells_long_under_control": sum(
            1 for r in pooled
            if r.get("long", {}).get("med") is not None and "control" in r
            and r["long"]["med"] <= r["control"]["med"]),
        "cells_monotone_increasing": sum(
            1 for r in pooled
            if r.get("short", {}).get("med") and r.get("mid", {}).get("med")
            and r.get("long", {}).get("med")
            and r["short"]["med"] < r["mid"]["med"] < r["long"]["med"]),
        "octave_artifacts_total": sum(
            r["jump_classes"].get("octave", 0) for r in pooled),
        "subharmonic_artifacts_total": sum(
            r["jump_classes"].get("subharmonic", 0) for r in pooled),
        "pauses_total": sum(r["n_pauses"] for r in pooled),
    }
    print(json.dumps({
        "note": ("Can F0 jump size identify a speaker change? Bands are median "
                 "|f2-f1|/f1 by pause length; 'control' is the same statistic "
                 "WITHIN one utterance, where the speaker is identical by "
                 "construction. A band that does not exceed the control is "
                 "phrasing, not a new speaker. Artifacts are counted on "
                 "jump = 1.0 (octave) and 0.5 (subharmonic), NOT on the ratio."),
        "verdict": verdicts,
        "cells": results,
    }, indent=2))


if __name__ == "__main__":
    main()