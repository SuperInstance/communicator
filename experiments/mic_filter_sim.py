"""
mic_filter_sim.py -- what a hardware high-pass does to OUR F0 output, measured
on the repo's own speech.

WHY THIS FILE EXISTS.

The hardware lane argues about a high-pass filter as the cheapest hum fix. Every
published number about that tradeoff is somebody else's speech and somebody
else's pitch tracker. Ours is unusual in a specific way: `src/pitch.py` tracks
F0 over 70-400 Hz, and mains hum sits at 150-200 Hz, INSIDE that band. So the
filter that removes the hum also edits the band the tracker lives in. Whether
that is acceptable is not a literature question here. It is measurable on
`examples/realaudio/alice01_excerpt.wav` with the tracker this repo ships.

WHAT IS MEASURED, PER HIGH-PASS CUT-OFF.

    attenuation   dB removed at 50, 100, 150 and 200 Hz -- the actual attack on
                  the band `tests/test_vad.py::TestDeclaredLimits` says the gate
                  cannot reject (pure tones 0.66-0.82, 200 Hz stack 0.97)
    pitch         median / p90 absolute cents error against the UNFILTERED
                  tracker on the SAME frames, and the octave-error rate
    voicing       frames the unfiltered tracker called voiced that the filtered
                  one does not -- the interaction with `pitch.ENERGY_FLOOR`,
                  which is the one people forget: a high-pass removes energy,
                  and the tracker gates on RMS before it ever runs YIN
    bias          median F0 before vs after, so a systematic pitch shift cannot
                  hide inside the error percentile

THE FILTER IS RBJ Biquad, TWO SECTIONS CASCADED.

Chosen because it is the filter an audio engineer actually reaches for and
because its response is exactly predictable, so the attenuation column is a
check on the implementation rather than an assumption. Attenuation is MEASURED
by pushing unit sines through the same code path, not asserted from the design.

WHAT THIS FILE CANNOT SHOW.

* Nothing about the gate. It has no onnxruntime here and never loads the model;
  the claim being audited is `vad.py`'s, and this only reports what the filter
  does to the signal underneath it.
* Nothing about hardware. No capsule, no mounting, no conducted hum. Those are
  argued in `docs/research/MIC-HARDWARE.md` from sources; this file supplies the
  one number those sources cannot: the cost on OUR tracker.
* Silence is filtered silence. The energy floor means unvoiced frames are
  mostly unchanged; the statistics are therefore computed over frames the
  UNFILTERED tracker called voiced, which is the population that matters.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pitch, wavio  # noqa: E402

SR = 16000
AUDIO = Path(__file__).resolve().parents[1] / "examples/realaudio" / "alice01_excerpt.wav"
OUT = Path(__file__).resolve().parent / "mic_filter_sim_results.json"

# Mains fundamentals and harmonics, in Hz: the band the gate is documented to
# fail in. 60 Hz is the project's own fixture; 50 Hz is the other grid.
HUM_HZ = (50, 100, 150, 200)

# Positions are subsampled by this stride so all five cut-offs plus the
# reference fit one run. Baseline and filtered runs use the SAME positions, so
# the comparison is paired and the stride cannot flatter any condition.
STRIDE = 3      # every 30 ms instead of every 10 ms

# Cut-offs to price. 60 Hz is the canonical "kill mains, keep everything"
# folklore value and is included so folklore can be scored against measurement.
# 200 Hz is included because it is the FIRST cut-off that meaningfully attacks
# the band the gate fails in -- and it sits inside the tracker's own range.
CUTOFFS = (60, 100, 150, 200)

# The aggressive alternative: notches. A high-pass cannot attenuate a
# component ABOVE its cut-off, and mains harmonics at 150-200 Hz sit inside
# the speaking range, so the only filter that removes them is one that carves
# holes at exactly those frequencies. Priced here rather than argued about.
# NOTCH_Q=10 at 16 kHz is a ~20 Hz-wide notch, so this is the BEST case for
# notching: a wide notch can only cost more pitch.
NOTCH_Q = 10.0
NOTCHES = {
    "notch_200Hz": (200.0,),
    "notch_mains_50_100_150_200Hz": (50.0, 100.0, 150.0, 200.0),
}


def biquad_hp(samples, fc, sr=SR, q=0.7071067811865476):
    """RBJ high-pass biquad, applied in place-order with state carried over."""
    w0 = 2.0 * math.pi * fc / sr
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    b0 = (1.0 + cw) / 2.0
    b1 = -(1.0 + cw)
    b2 = (1.0 + cw) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cw
    a2 = 1.0 - alpha
    b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
    a1, a2 = a1 / a0, a2 / a0
    x1 = x2 = y1 = y2 = 0.0
    out = []
    for x in samples:
        y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, x
        y2, y1 = y1, y
        out.append(y)
    return out


def highpass(samples, fc, sections=2):
    """Cascaded sections: `sections=2` is a 4th-order roll-off."""
    for _ in range(sections):
        samples = biquad_hp(samples, fc)
    return samples


def biquad_notch(samples, fc, sr=SR, q=NOTCH_Q):
    """RBJ band-stop biquad."""
    w0 = 2.0 * math.pi * fc / sr
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    b0, b1, b2 = 1.0, -2.0 * cw, 1.0
    a0, a1, a2 = 1.0 + alpha, -2.0 * cw, 1.0 - alpha
    b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
    a1, a2 = a1 / a0, a2 / a0
    x1 = x2 = y1 = y2 = 0.0
    out = []
    for x in samples:
        y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, x
        y2, y1 = y1, y
        out.append(y)
    return out


def notch_all(samples, freqs):
    for f in freqs:
        samples = biquad_notch(samples, f)
    return samples


def attenuation_db(samples, fc, probe_hz, fn=highpass):
    """Measured loss of a unit sine at `probe_hz` through the same filter."""
    n = SR  # one second is plenty for a steady tone
    t = [math.sin(2.0 * math.pi * probe_hz * i / SR) for i in range(n)]
    ref = math.sqrt(sum(v * v for v in t) / n)
    y = fn(t, fc)
    out = math.sqrt(sum(v * v for v in y) / n)
    return 20.0 * math.log10(out / ref) if out > 0 else -999.0


def rms(seq):
    return math.sqrt(sum(v * v for v in seq) / len(seq)) if seq else 0.0


def frame_stats(frames, base_hz, base_rms):
    """Compare one condition's per-frame results against the unfiltered run."""
    cents, octave_err, lost, bias = [], 0, 0, []
    for (f_hz, f_rms), b_hz, _b_rms in zip(frames, base_hz, base_rms):
        if b_hz is None:
            continue
        # Unvoiced AFTER filtering, either because YIN refused or because the
        # RMS fell under `pitch.ENERGY_FLOOR` -- both are the same operational
        # event, and the second is the one a high-pass actually causes.
        if f_hz is None or f_rms < pitch.ENERGY_FLOOR:
            lost += 1
            continue
        c = 1200.0 * math.log2(f_hz / b_hz)
        cents.append(abs(c))
        bias.append(f_hz - b_hz)
        if abs(abs(c) - 1200.0) < 60.0:
            octave_err += 1
    n = len(cents)
    cents.sort()
    def pct(p):
        if not cents:
            return None
        return round(cents[min(len(cents) - 1, int(round(p * (len(cents) - 1))))], 2)
    return {
        "compared_frames": n,
        "median_abs_cents": round(statistics.median(cents), 2) if cents else None,
        "p90_abs_cents": pct(0.90),
        "max_abs_cents": round(cents[-1], 2) if cents else None,
        "frac_over_20_cents": round(sum(1 for c in cents if c > 20.0) / n, 4) if n else None,
        "frac_over_50_cents": round(sum(1 for c in cents if c > 50.0) / n, 4) if n else None,
        "octave_error_rate": round(octave_err / n, 4) if n else None,
        "median_hz_shift": round(statistics.median(bias), 3) if bias else None,
        "voiced_frames_lost_to_filter": lost,
    }


def run_condition(samples, tracker):
    positions = list(range(0, tracker.n_positions(len(samples)), STRIDE))
    base_hz, base_rms = [], []
    for i in positions:
        fr = samples[i * tracker.hop:i * tracker.hop + tracker.frame_len]
        hz, _ = tracker.estimate(fr)
        base_hz.append(hz)
        base_rms.append(rms(fr))
    rows = [{
        "cutoff_hz": None, "note": "unfiltered reference",
        "voiced_frames": sum(1 for h in base_hz if h is not None),
        "median_hz": round(statistics.median([h for h in base_hz if h]), 3)
                     if any(base_hz) else None,
        "atten_db": {str(f): None for f in HUM_HZ},
    }]

    for fc in CUTOFFS:
        t0 = time.perf_counter()
        filt = highpass(samples, fc)
        tracker = pitch.PitchTracker(SR)
        after, after_hz = [], []
        for i in positions:
            fr = filt[i * tracker.hop:i * tracker.hop + tracker.frame_len]
            hz, _ = tracker.estimate(fr)
            r = rms(fr)
            after_hz.append(hz)
            after.append((hz, r))
        row = {"cutoff_hz": fc,
               "sections": 2,
               "voiced_frames": sum(1 for h in after_hz if h is not None),
               "median_hz": round(statistics.median([h for h in after_hz if h]), 3)
                            if any(after_hz) else None,
               "seconds": round(time.perf_counter() - t0, 1)}
        row.update(frame_stats(after, base_hz, base_rms))
        row["atten_db"] = {str(f): round(attenuation_db(samples, fc, f), 2)
                           for f in HUM_HZ}
        rows.append(row)
        print(f"  fc={fc:>3} Hz  median_hz={row['median_hz']}  "
              f"median_cents={row['median_abs_cents']}  "
              f"p90={row['p90_abs_cents']}  lost={row['voiced_frames_lost_to_filter']}  "
              f"atten@200={row['atten_db']['200']} dB", flush=True)

    for label, freqs in NOTCHES.items():
        t0 = time.perf_counter()
        filt = notch_all(samples, freqs)
        tr = pitch.PitchTracker(SR)
        after, after_hz = [], []
        for i in positions:
            fr = filt[i * tr.hop:i * tr.hop + tr.frame_len]
            hz, _ = tr.estimate(fr)
            after_hz.append(hz)
            after.append((hz, rms(fr)))
        row = {"cutoff_hz": None, "notch_hz": list(freqs), "q": NOTCH_Q,
               "sections": len(freqs),
               "voiced_frames": sum(1 for h in after_hz if h is not None),
               "median_hz": round(statistics.median([h for h in after_hz if h]), 3)
                            if any(after_hz) else None,
               "seconds": round(time.perf_counter() - t0, 1)}
        row.update(frame_stats(after, base_hz, base_rms))
        row["atten_db"] = {str(f): round(attenuation_db(samples, freqs, f,
                                                       fn=notch_all), 2)
                           for f in HUM_HZ}
        rows.append(row)
        print(f"  {label:<28} median_cents={row['median_abs_cents']}  "
              f"p90={row['p90_abs_cents']}  lost={row['voiced_frames_lost_to_filter']}  "
              f"atten@200={row['atten_db']['200']} dB", flush=True)
    return rows


def main():
    if not AUDIO.is_file():
        print(json.dumps({"error": f"missing fixture {AUDIO}"}))
        return
    audio = wavio.read_wav(str(AUDIO))
    if audio.sample_rate != SR:
        print(json.dumps({"error": f"fixture is {audio.sample_rate} Hz, "
                                  f"the tracker grid is {SR}"}))
        return
    print(f"fixture: {AUDIO.name}  {len(audio.samples)/SR:.1f} s", flush=True)
    tracker = pitch.PitchTracker(SR)
    rows = run_condition(audio.samples, tracker)
    out = {
        "note": ("Measured effect of a hardware high-pass on this repo's own F0 "
                 "tracker. Cut-off, measured attenuation at mains harmonics, "
                 "and pitch error against the UNFILTERED tracker on the same "
                 "frames. No gate is loaded and no hardware is modelled."),
        "audio": {"path": "examples/realaudio/alice01_excerpt.wav",
                  "sample_rate": SR,
                  "seconds": round(len(audio.samples) / SR, 2),
                  "frames_available": tracker.n_positions(len(audio.samples)),
                  "frame_stride": STRIDE,
                  "frames_analysed": len(range(0, tracker.n_positions(len(audio.samples)), STRIDE)),
                  "pitch_range_hz": [pitch.FMIN, pitch.FMAX],
                  "energy_floor_rms": pitch.ENERGY_FLOOR},
        "filter": "RBJ biquad high-pass, 2 cascaded sections (4th order), Q=0.707",
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
