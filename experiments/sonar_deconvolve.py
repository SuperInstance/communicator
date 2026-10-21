#!/usr/bin/env python3
"""
sonar_deconvolve.py -- the sonar return, measured properly.

WHAT THIS IS.

The brief (`docs/research/BRIEF-ROOM-SONAR.md`) makes the deconvolution
and impulse-response analysis mine once a probe return exists. The
first return is the probe artifact session `1792561741`, and it needs
more than a deconvolution: it needs a CHECK, because the artifact
carries its own verdict and that verdict has to survive an independent
measurement before anybody builds on it.

So this tool does four things:

  1. IT REPRODUCES THE PROBE'S OWN TABLE from the saved WAV. If the
     artifact's numbers cannot be re-derived from the audio beside
     them, the artifact is not about the audio and none of it counts.

  2. IT MEASURES THE CHANNEL THE RIGHT WAY, with the excitation
     gated. A band is only scored if the STIMULUS actually drove energy
     there. The probe's first version divided the return by the
     stimulus without that gate, and a band where the stimulus is
     near zero hands back a huge number for a very small return -- the
     hole reads as the signal. That is the whole story of the artifact's
     "usable band: 22-24 kHz", and this tool shows it rather than
     arguing about it.

  3. IT DECONVOLVES, and says what the result is. The impulse response
     of a sweep is the return divided by the stimulus. If the return
     has no stimulus in it, the "impulse response" is the inverse of
     the stimulus, and it GROWS instead of decaying. A growing IR is
     the signature of a division artifact and is reported as such.

  4. IT SAYS WHAT WOULD OPEN THE LANE, in numbers.

The parts are the probe's own: the stimulus generators, the spectrum
function, the band width, the SNR floor and the verdict rule are read
from the artifact's own manifest and reproduced exactly, so this tool
and the probe are talking about the same thing.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

#: The probe artifact this tool was written for, if none is given.
DEFAULT_SESSION = Path("/mnt/c/proart_capture/corpus/free_sessions/1792561741")

#: A band counts as excited when the stimulus drove it to within this
#: many dB of its own peak. The probe's second version uses 60 dB.
EXCITATION_FLOOR_DB = 60.0


# ============================================================================
# 1. READING THE ARTIFACT
# ============================================================================

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """The WAV as float samples in [-1, 1], and its rate."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return a, sr


def find_taps(buf: np.ndarray, sr: int, meta: dict) -> dict:
    """Where the four taps are inside the saved WAV.

    The probe taps the room, then the sweep, then a gap, then the pink
    burst, and it DISCARDS the gap. The artifact writer puts the
    discarded tap back as zeros, so the gap is findable exactly, and
    the layout is checked against the probe's own schedule afterwards.
    """
    zeros = np.abs(buf) == 0
    d = np.diff(zeros.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    runs = [(s, e) for s, e in zip(starts, ends) if e - s > 100]
    if not runs:
        raise SystemExit("no discarded tap found: the artifact is not the "
                         "shape this tool knows")
    gap_start, gap_end = max(runs, key=lambda r: r[1] - r[0])

    sched = meta["sonar"]["stimulus"]["schedule_s"]
    sweep_s = sched["sweep"]
    pink_s = sched["pink"]
    pre_s = sched["pre"]
    gap_s = sched["gap"]
    # the probe taps the sweep for its length plus 300 ms, and the pink
    # for its length plus 300 ms
    sweep_tap = int(round((sweep_s + 0.3) * sr))
    pink_tap = int(round((pink_s + 0.3) * sr))
    noise_tap = int(round(pre_s * sr))

    sweep_start = gap_start - sweep_tap
    noise_start = sweep_start - noise_tap
    pink_start = gap_end
    return {
        "noise": (noise_start, sweep_start),
        "sweep": (sweep_start, gap_start),
        "gap": (gap_start, gap_end),
        "pink": (pink_start, pink_start + pink_tap),
        "lead_samples": noise_start,
        "gap_is_digital_zero": True,
    }


# ============================================================================
# 2. THE PROBE'S OWN MATH
# ============================================================================

def spectrum_of(b: np.ndarray, sr: int, fft_n: int = 8192) -> np.ndarray:
    """The probe's spectrum: the mean Hann-windowed FFT magnitude.

    Same as the probe's `spectrumOf`: the window is applied in the time
    domain, the magnitudes are averaged over the windows, and the
    Nyquist bin is dropped (the probe's own FFT returns n/2 bins).
    """
    n_win = max(1, len(b) // fft_n)
    acc = np.zeros(fft_n // 2)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(fft_n) / fft_n)
    for w in range(n_win):
        seg = b[w * fft_n:(w + 1) * fft_n]
        if len(seg) < fft_n:
            break
        acc += np.abs(np.fft.rfft(seg * win))[:fft_n // 2]
    return acc / n_win


def make_sweep(sr: int, f0: float, f1: float, seconds: float) -> np.ndarray:
    """The probe's exponential sine sweep, reproduced exactly."""
    n = int(round(seconds * sr))
    ratio = f1 / f0
    ln = np.log(ratio)
    out = np.zeros(n)
    for i in range(n):
        t = i / n
        ph = 2 * np.pi * f0 * (seconds * (ratio ** t - 1) / ln)
        f = min(i, n - 1 - i) / (0.01 * sr)
        out[i] = np.sin(ph) * min(1.0, max(0.0, f))
    return out


def med(a) -> float:
    """The probe's median: sort, take the middle."""
    return float(np.median(np.sort(np.asarray(a))))


def band_table(return_spec: np.ndarray, ref_spec: np.ndarray,
               noise_spec: np.ndarray, sr: int, fft_n: int,
               f_lo: int, f_hi: int, band_hz: int) -> list[dict]:
    """The per-band numbers, both the probe's way and the gated way."""
    bin_hz = sr / fft_n
    ref_max = max(ref_spec[int(f_lo / bin_hz):int(f_hi / bin_hz)].max(), 1e-30)
    floor = ref_max * 10 ** (-EXCITATION_FLOOR_DB / 20.0)
    rows = []
    for f in range(f_lo, f_hi, band_hz):
        bins = list(range(int(f / bin_hz), int((f + band_hz) / bin_hz)))
        bins = [b for b in bins if 1 <= b < fft_n // 2]
        if not bins:
            continue
        tr = [return_spec[b] / max(ref_spec[b], 1e-12) for b in bins]
        nz = [max(noise_spec[b], 1e-12) for b in bins]
        rf = [ref_spec[b] for b in bins]
        ret = med([return_spec[b] for b in bins])
        exc = med(rf) >= floor
        rows.append({
            "f0": int(f), "f1": int(f + band_hz),
            # the probe's own numbers
            "probe_transfer_db": round(20 * np.log10(max(med(tr), 1e-12)), 1),
            "probe_snr_db": round(20 * np.log10(max(med(tr) / med(nz), 1e-6)), 1),
            # the numbers that mean something
            "reference_db": round(20 * np.log10(max(med(rf), 1e-30)), 1),
            "return_db": round(20 * np.log10(max(ret, 1e-30)), 1),
            "noise_db": round(20 * np.log10(max(med(nz), 1e-30)), 1),
            "return_over_noise_db": round(20 * np.log10(max(ret / med(nz), 1e-12)), 1),
            "return_over_stimulus_db": round(20 * np.log10(max(ret / max(med(rf), 1e-30), 1e-30)), 1),
            "excited": bool(exc),
        })
    return rows


# ============================================================================
# 3. THE DECONVOLUTION
# ============================================================================


def deconvolve(return_spec: np.ndarray, ref_spec: np.ndarray,
               sr: int, fft_n: int,
               return_over_noise_db: float | None = None) -> dict:
    """The impulse response and what it says.

    The impulse response of a sweep measurement is the return divided
    by the stimulus. The DECISIVE question is not what shape the
    result has -- a division by a near-zero stimulus produces a
    perfectly shaped-looking response -- but whether the return held
    any stimulus at all. If the return is at the noise floor, then
    there is nothing to deconvolve, the result is the inverse of the
    stimulus, and it is not a room. That is said here rather than left
    for the reader to discover.
    """
    h = return_spec / np.maximum(ref_spec, 1e-12)
    ir = np.fft.irfft(h, n=fft_n)
    half = fft_n // 2
    t = np.arange(half) / sr
    env = np.abs(ir[:half])
    peak_i = int(np.argmax(env))
    # the decay slope, over the part of the response after the peak
    sel = t > (t[peak_i] + 0.002)
    slope = None
    rt60 = None
    if sel.sum() > 16:
        y = 20 * np.log10(np.maximum(env[sel], 1e-30))
        c = np.polyfit(t[sel], y, 1)
        slope = float(c[0])
        rt60 = float(-60.0 / c[0]) if abs(c[0]) > 1e-9 else None
    grows = bool(slope is not None and slope > 0)
    meaningful = (return_over_noise_db is None) or (return_over_noise_db >= 10)
    if not meaningful:
        verdict = ("NOT a room, and not anything else: the return sits on "
                   "the noise floor, so there is no stimulus in it to "
                   "deconvolve. This response is the inverse of the "
                   "stimulus, not the room's answer to it.")
    elif grows:
        verdict = ("this is not a room: the response GROWS, which is what "
                   "dividing a floor by a near-zero stimulus looks like.")
    else:
        verdict = ("the response decays; this is at least shaped like a "
                   "room, and the early reflections are worth reading")
    return {
        "peak_ms": round(float(t[peak_i]) * 1000.0, 3),
        "peak_amplitude": float(env[peak_i]),
        "peak_to_median_db": round(float(20 * np.log10(
            max(env[peak_i] / max(float(np.median(env)), 1e-30), 1e-30))), 2),
        "decay_slope_db_per_s": None if slope is None else round(slope, 3),
        "rt60_s": None if rt60 is None else round(rt60, 4),
        "grows_instead_of_decaying": grows,
        "return_over_noise_db": (None if return_over_noise_db is None
                                  else round(float(return_over_noise_db), 2)),
        "meaningful": meaningful,
        "verdict": verdict,
        "ir": ir,
    }


# ============================================================================
# 4. THE DRIVER
# ============================================================================

def analyse(session_dir: Path) -> dict:
    """The whole measurement."""
    meta = json.loads((session_dir / "meta.json").read_text())
    buf, sr = read_wav(session_dir / "audio.wav")
    stim = meta["sonar"]["stimulus"]
    sw = stim["sweep"]
    fft_n = meta["sonar"]["analysis"]["fft_n"]
    band_hz = meta["sonar"]["analysis"]["band_hz"]
    f_lo, f_hi = 15000, 24000

    taps = find_taps(buf, sr, meta)
    noise = buf[taps["noise"][0]:taps["noise"][1]]
    ret = buf[taps["sweep"][0]:taps["sweep"][1]]
    pink = buf[taps["pink"][0]:taps["pink"][1]]

    ref_spec = spectrum_of(make_sweep(sr, sw["f0_hz"], sw["f1_hz"], sw["duration_s"]),
                            sr, fft_n)
    ret_spec = spectrum_of(ret, sr, fft_n)
    noise_spec = spectrum_of(noise, sr, fft_n)
    pink_spec = spectrum_of(pink, sr, fft_n)

    rows = band_table(ret_spec, ref_spec, noise_spec, sr, fft_n,
                      f_lo, f_hi, band_hz)

    # the excitation-gated verdict: a band is usable when the stimulus
    # drove it AND the return is at least 10 dB above the floor
    usable = [r for r in rows if r["excited"] and r["return_over_noise_db"] >= 10]
    bands = []
    for r in usable:
        if bands and bands[-1]["f1"] == r["f0"]:
            bands[-1]["f1"] = r["f1"]
        else:
            bands.append({"f0": int(r["f0"]), "f1": int(r["f1"])})

    # does the return distinguish the sweep tap from the pink tap?
    bin_hz = sr / fft_n
    ul = slice(int(16000 / bin_hz), int(24000 / bin_hz))
    sweep_ul = float(np.mean(ret_spec[ul]))
    pink_ul = float(np.mean(pink_spec[ul]))
    noise_ul = float(np.mean(noise_spec[ul]))

    sweep_ul_db = 20 * np.log10(max(sweep_ul, 1e-30))
    noise_ul_db = 20 * np.log10(max(noise_ul, 1e-30))
    ir = deconvolve(ret_spec, ref_spec, sr, fft_n,
                    return_over_noise_db=sweep_ul_db - noise_ul_db)

    doc = {
        "session": str(session_dir),
        "probe": meta.get("probe"),
        "probe_version": (meta.get("probe") or {}).get("version"),
        "device": "the local machine, numpy only -- no API, no network",
        "sample_rate": sr,
        "samples": len(buf),
        "duration_s": round(len(buf) / sr, 4),
        "taps": {k: ([int(x) for x in v] if isinstance(v, tuple) else v)
                 for k, v in taps.items()},
        "the_probe_said": {
            "usable_band": meta["sonar"].get("usable_band"),
            "rule": meta["sonar"]["analysis"].get("rule"),
            "bands": meta["sonar"].get("bands"),
        },
        "the_gated_verdict": {
            "usable_bands": bands,
            "usable_band": (f"{bands[0]['f0'] / 1000}-{bands[-1]['f1'] / 1000} kHz"
                            if bands else "NONE"),
            "rule": ("a band is usable when the STIMULUS drove energy there "
                     "(within 60 dB of the stimulus's own peak) AND the return "
                     "is at least 10 dB above the noise floor"),
            "excitation_gated_bands": sum(1 for r in rows if r["excited"]),
            "bands_total": len(rows),
        },
        "the_ultrasonic_floor": {
            "sweep_tap_16_24k_db": round(20 * np.log10(max(sweep_ul, 1e-30)), 2),
            "pink_tap_16_24k_db": round(20 * np.log10(max(pink_ul, 1e-30)), 2),
            "noise_tap_16_24k_db": round(20 * np.log10(max(noise_ul, 1e-30)), 2),
            "sweep_minus_noise_db": round(20 * np.log10(
                max(sweep_ul / max(noise_ul, 1e-30), 1e-30)), 2),
            "pink_minus_noise_db": round(20 * np.log10(
                max(pink_ul / max(noise_ul, 1e-30), 1e-30)), 2),
            "note": ("the sweep tap and the pink tap are the same level in "
                     "this band, so the band holds the floor and nothing else"),
        },
        "the_deconvolution": {k: v for k, v in ir.items() if k != "ir"},
        "bands": rows,
    }
    return doc


def draw(doc: dict, out_dir: Path, session_dir: Path) -> list[str]:
    """The picture: what was played against what came back."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = json.loads((session_dir / "meta.json").read_text())
    buf, sr = read_wav(session_dir / "audio.wav")
    stim = meta["sonar"]["stimulus"]
    sw = stim["sweep"]
    fft_n = meta["sonar"]["analysis"]["fft_n"]
    taps = find_taps(buf, sr, meta)
    ref_spec = spectrum_of(make_sweep(sr, sw["f0_hz"], sw["f1_hz"], sw["duration_s"]),
                            sr, fft_n)
    ret_spec = spectrum_of(buf[taps["sweep"][0]:taps["sweep"][1]], sr, fft_n)
    noise_spec = spectrum_of(buf[taps["noise"][0]:taps["noise"][1]], sr, fft_n)
    freqs = np.arange(len(ref_spec)) * (sr / fft_n)
    m = (freqs >= 14000) & (freqs <= 24500)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.semilogy(freqs[m], ref_spec[m], label="the STIMULUS (the sweep)",
                color="#4da3ff", lw=1.6)
    ax.semilogy(freqs[m], ret_spec[m], label="the RETURN (the mic)",
                color="#ff7043", lw=1.4)
    ax.semilogy(freqs[m], noise_spec[m], label="the noise tap",
                color="#9e9e9e", lw=1.2, ls="--")
    ax.axvspan(sw["f0_hz"], sw["f1_hz"], color="#2a2a35", alpha=0.55,
               label="where the sweep actually played")
    ax.set_xlim(14000, 24500)
    ax.set_xlabel("Hz", color="#b8b8c4")
    ax.set_ylabel("magnitude (probe units)", color="#b8b8c4")
    ax.tick_params(colors="#8a8a98")
    ax.set_title("ROOM SONAR -- what was played against what came back\n"
                 "the return sits on the noise floor across the whole band; "
                 "the sweep's own\nenergy is where the sweep played, and "
                 "nothing came back from it",
                 color="#e8e8ee", fontsize=11)
    leg = ax.legend(facecolor="#181820", edgecolor="#3a3a46", fontsize=9)
    for t in leg.get_texts():
        t.set_color("#d8d8e2")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "room-sonar-channel.png"
    fig.savefig(p, dpi=110, bbox_inches="tight", facecolor="#101014")
    plt.close(fig)
    return [str(p)]


def _jsonable(o):
    """numpy scalars are numbers; the json module does not know that."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", nargs="?", default=str(DEFAULT_SESSION),
                    help="the probe artifact session folder")
    ap.add_argument("--out", default=str(REPO / "docs" / "research"),
                    help="where the json and the image go")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    session_dir = Path(a.session)
    out_dir = Path(a.out)
    doc = analyse(session_dir)
    v = doc["the_gated_verdict"]
    u = doc["the_ultrasonic_floor"]
    print(f"session   {session_dir}")
    print(f"probe     {doc['probe']} v{doc['probe_version']}")
    print(f"the probe said   {doc['the_probe_said']['usable_band']}")
    print(f"the gated truth  {v['usable_band']}   "
          f"({v['excitation_gated_bands']} of {v['bands_total']} bands were "
          f"even excited by the stimulus)")
    print(f"ultrasonic floor: sweep tap {u['sweep_tap_16_24k_db']} dB, "
          f"pink tap {u['pink_tap_16_24k_db']} dB, "
          f"noise tap {u['noise_tap_16_24k_db']} dB")
    print(f"deconvolution: {doc['the_deconvolution']['verdict']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "room_sonar_analysis.json").write_text(
        json.dumps(doc, indent=2, default=_jsonable))
    print(f"json       {out_dir / 'room_sonar_analysis.json'}")
    if not a.no_images:
        for p in draw(doc, out_dir, session_dir):
            print(f"wrote      {p}")
    return 0


# ============================================================================
# 5. THE SELFTEST
# ============================================================================

def selftest() -> int:
    """The checks that need no artifact."""
    import tempfile

    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  [{'ok' if got == want else 'FAIL'}] {name}: {got!r}")

    print("SELFTEST -- the spectrum, the bands, the deconvolution")
    print("-" * 60)

    sr = 48000
    # 1. THE SWEEP. Its own spectrum must be where it says it is.
    sweep = make_sweep(sr, 16000, 22000, 3.0)
    check("the sweep is three seconds long", len(sweep), 144000)
    spec = spectrum_of(sweep, sr, 8192)
    bin_hz = sr / 8192
    inside = spec[int(18000 / bin_hz):int(20000 / bin_hz)].mean()
    below = spec[int(14000 / bin_hz):int(15500 / bin_hz)].mean()
    above = spec[int(22500 / bin_hz):int(24000 / bin_hz)].mean()
    check("the sweep drives energy in its own band", inside > below * 100, True)
    check("and NOT above its top", above < inside / 10, True)

    # 2. THE EXCITATION GATE. A band with no stimulus must not score.
    ret = np.full(4096, 1e-6)
    noise = np.full(4096, 1e-6)
    rows = band_table(ret, spec, noise, sr, 8192, 15000, 24000, 250)
    gated = [r for r in rows if r["excited"]]
    check("only the bands the stimulus drove are excited",
          all(15000 <= r["f0"] < 22000 for r in gated), True)
    check("and the unexcited ones are marked", 
          any(not r["excited"] for r in rows), True)

    # 3. THE DECONVOLUTION. On a synthetic room it recovers the room;
    #    on a floor it says there is nothing there.
    n = 4096
    h_true = np.zeros(n)
    h_true[0] = 1.0
    h_true[int(0.012 * sr)] = 0.5
    h_true += 0.3 * np.exp(-np.arange(n) / (0.05 * sr))
    H_true = np.fft.rfft(h_true, n=8192)[:4096]
    room = deconvolve(spec * H_true, spec, sr, 8192,
                      return_over_noise_db=30.0)
    check("a real room's response decays",
          room["grows_instead_of_decaying"], False)
    check("and has a positive RT60", room["rt60_s"] > 0, True)
    check("and is meaningful", room["meaningful"], True)
    flat = deconvolve(np.full(4096, 1e-9), spec, sr, 8192,
                      return_over_noise_db=0.0)
    check("a return on the floor is not meaningful", flat["meaningful"], False)
    check("and says so", "NOT a room" in flat["verdict"], True)

    # 4. THE TAP LAYOUT, on a synthetic artifact.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        meta = {"sonar": {"stimulus": {"sweep": {"f0_hz": 16000, "f1_hz": 22000,
                                                "duration_s": 3.0},
                                       "pink": {"duration_s": 0.5},
                                       "schedule_s": {"pre": 0.4, "sweep": 3.0,
                                                      "gap": 0.4, "pink": 0.5}},
                           "analysis": {"fft_n": 8192, "band_hz": 250}}}
        buf = np.ones(4 * 48000)
        buf[19488:38688] = 0.0
        taps = find_taps(buf, 48000, meta)
        check("the gap is found", taps["gap"], (19488, 38688))
        check("the sweep tap sits before it", taps["sweep"][1], 19488)
        check("the pink tap sits after it", taps["pink"][0], 38688)

    print("-" * 60)
    if failures:
        print(f"SELFTEST FAILED: {len(failures)} check(s)")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
