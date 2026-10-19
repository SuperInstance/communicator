"""
inthewild_probe.py -- does the gate's 150-200 Hz failure hurt USERS or only CPU?

WHY THIS FILE EXISTS, AND WHY IT IS NOT hum_sweep.py.

`hum_sweep.py` measures hum ALONE and speech ALONE. That leaves the question
that actually decides whether the gate is a safety net or a costume:

    a room has a hum AND a person talking AT THE SAME TIME.

hum_sweep cannot see that case. This file adds the mixture, and then asks the
question hum_sweep does not ask: when the gate lets the hum through at 150 Hz,
does the watcher emit a FALSE PITCH, or does it just burn CPU?

That distinction is the whole value. A gate that scores 0.97 on a 200 Hz stack
and the watcher then confidently reports "F0 = 200 Hz" is a CORRECTNESS bug --
a table built from hz_p05/hz_p95 now contains a mains harmonic posing as a
voice. A gate that scores 0.97 and the watcher reports nothing is only a speed
bug. ManiWAV measured exactly this shape of thing and got the uncomfortable
answer in the other direction: their test-time noise reduction "causes the
signal to be deprecated [degraded], resulting in a domain gap"
(https://arxiv.org/abs/2406.19464, Sec. 4.4). A cheaper signal is not the same
as a correct signal.

PART B TESTS THE TRANSFERABLE IDEA.

ManiWAV's per-robot noise calibration (ESC-50 + 10 clips of the robot's own
motor noise recorded at the contact-mic, overlaid p=0.5 each) is the one idea
from that paper with a plausible mechanism for a wearable. The wearable
analogue is a per-device noise profile: the fan, the frame resonance, the
charger buzz of THIS unit. Here that profile is applied as spectral
subtraction on a FROZEN gate -- the only kind of conditioning available to us,
since we do not train silero.

ManiWAV's counter-result predicts this will fail. This file is the test of that
prediction, not a demonstration of it.

PART C IS THE FAILURE MODE, MEASURED NOT ASSUMED.

A static per-device profile is only correct for the noise it was measured on.
The scenario that matters is a fan spinning up: the device's own noise gets
louder AFTER calibration. Profile calibrated on quiet hum, applied to hum 6x
louder. A method that only works at the calibration level is not a method.

WHAT THIS FILE CANNOT SHOW.

Everything about a real device. The hums here are synthetic harmonic stacks,
the "device profile" is synthetic, and no audio hardware exists on this box.
This measures the GATE'S RESPONSE to a frequency structure. It is evidence
about the mechanism, not a verdict about any real fan or charger.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src import vad, wavio, watcher  # noqa: E402

SR = 16000
DUR_S = 6.0
CORPUS = Path(__file__).resolve().parents[1] / "examples/realaudio"
SPEECH = CORPUS / "alice01_excerpt.wav"

# Mains band only. 250/300 are excluded on purpose: they are where the gate is
# known to work, and including them would dilute the boundary this is looking
# for.
FUNDAMENTALS = [50, 60, 100, 120, 150, 175, 200]

# 0.05 is the amplitude the project's hum fixture uses (src/vad.py docstring).
# 0.3 and 0.5 are the amplitudes tests/test_vad.py::TestDeclaredLimits uses for
# the "150-200 Hz is indistinguishable from speech" claim. Both are measured
# here because they are 20 dB apart and the project's two numbers were never
# taken at the same operating point.
AMPS = [0.05, 0.3, 0.5]
SNRS_DB = [20.0, 10.0, 0.0]

# hum_sweep.py and this file: 12 harmonics, 1/k, peak-normalised to amp.
N_HARM = 12
# tests/test_vad.py: 3 harmonics (1,2,3), NOT peak-normalised, amp is the
# fundamental coefficient. Reproduced exactly so the project's own regression
# fixture can be re-run beside the hum fixture instead of assumed comparable.
TEST_VAD_HARMS = (1, 2, 3)

STFT_FRAME = 512
STFT_HOP = 256

_GATE = None


def fresh_gate():
    """
    One loaded ONNX session, reset between signals.

    `SileroGate.reset()` clears the RNN state and the carry buffer, which is
    exactly what a new object would have, but it keeps the graph loaded.
    Building a session per measurement cost ~1 s each and made the first
    version of this file time out. Fresh state, shared weights.
    """
    global _GATE
    if _GATE is None:
        _GATE = vad.SileroGate(SR)
    _GATE.reset()
    return _GATE


def harmonic_stack(f0: float, amp: float, harms=range(1, N_HARM + 1),
                   normalise=True) -> np.ndarray:
    """
    A periodic buzz with 1/k amplitude rolloff -- a mains-driven device.

    Same generator as hum_sweep.py, deliberately: two files disagreeing about
    the fixture makes the comparison between them worthless. `src/` stays
    stdlib; this is an offline harness and onnxruntime already needs numpy.
    """
    n = int(DUR_S * SR)
    t = np.arange(n, dtype=np.float64) / SR
    out = np.zeros(n, dtype=np.float64)
    for k in harms:
        f = f0 * k
        if f >= SR / 2:
            break
        out += (amp / k) * np.sin(2.0 * np.pi * f * t)
    if not normalise:
        return out.astype(np.float32)
    peak = np.abs(out).max() or 1.0
    return (out * (amp / peak)).astype(np.float32)


def speech_clip() -> np.ndarray:
    """Real speech, peak-normalised to 1.0, truncated to DUR_S.

    Real speech, not a synthesised vowel. src/vad.py's own docstring records
    that the synthetic corpus flattered the gate, and repeating that mistake
    would make this whole file decorative.
    """
    audio = wavio.read_wav(str(SPEECH))
    x = np.asarray(audio.samples, dtype=np.float32)[:int(DUR_S * SR)]
    peak = np.abs(x).max() or 1.0
    return x / peak


def mix(speech: np.ndarray, hum: np.ndarray, snr_db: float) -> np.ndarray:
    """Speech at 1.0 peak, hum scaled to the requested SNR by speech RMS."""
    sp_rms = float(np.sqrt(np.mean(speech.astype(np.float64) ** 2))) or 1e-9
    hu_rms = float(np.sqrt(np.mean(hum.astype(np.float64) ** 2))) or 1e-9
    target = sp_rms / (10.0 ** (snr_db / 20.0))
    out = speech + hum * (target / hu_rms)
    peak = np.abs(out).max() or 1.0
    return (out / peak).astype(np.float32) if peak > 1.0 else out.astype(np.float32)


# --- Part B/C: per-device noise profile as spectral subtraction -------------

def _stft_mag(x: np.ndarray):
    win = np.hanning(STFT_FRAME).astype(np.float32)
    n = 1 + (len(x) - STFT_FRAME) // STFT_HOP
    if n <= 0:
        return None, win
    idx = np.arange(STFT_FRAME)[None, :] + STFT_HOP * np.arange(n)[:, None]
    frames = x[idx] * win[None, :]
    return np.abs(np.fft.rfft(frames, axis=1)), win


def noise_profile(noise: np.ndarray) -> np.ndarray:
    """Per-bin mean magnitude of a device's OWN idle noise. The calibration."""
    mag, _ = _stft_mag(noise)
    if mag is None:
        return None
    return mag.mean(axis=0)


def condition(x: np.ndarray, profile: np.ndarray, alpha: float) -> np.ndarray:
    """
    Subtract the measured device profile from the signal, keep phase.

    alpha is over-subtraction: alpha=1.0 removes exactly the calibrated level,
    alpha>1 also removes signal that looks like the noise. Sweeping alpha is
    the honest way to show the tradeoff between rejection and speech damage.
    """
    if profile is None:
        return x
    mag, win = _stft_mag(x)
    if mag is None:
        return x
    red = np.maximum(mag - alpha * profile[None, :], 0.0)
    # Phase is kept; only the magnitude is conditioned. Doing it explicitly
    # rather than cleverly, because a wrong phase stitch here would look like
    # "conditioning destroys the signal" when it is really an overlap-add bug.
    frames = x[idx_of(x)] * win[None, :]
    ph = np.angle(np.fft.rfft(frames, axis=1))
    spec = red * np.exp(1j * ph)
    out = np.zeros(len(x), dtype=np.float64)
    wsum = np.zeros(len(x), dtype=np.float64)
    wsq = (win ** 2).astype(np.float64)
    for i in range(spec.shape[0]):
        seg = np.fft.irfft(spec[i], n=STFT_FRAME)
        out[i * STFT_HOP:i * STFT_HOP + STFT_FRAME] += seg * wsq
        wsum[i * STFT_HOP:i * STFT_HOP + STFT_FRAME] += wsq
    nz = wsum > 1e-9
    out[nz] /= wsum[nz]
    return out.astype(np.float32)


def idx_of(x: np.ndarray) -> np.ndarray:
    n = 1 + (len(x) - STFT_FRAME) // STFT_HOP
    return (np.arange(STFT_FRAME)[None, :]
            + STFT_HOP * np.arange(n)[:, None])


# --- measurement ----------------------------------------------------------

def gate_stats(samples: np.ndarray) -> dict:
    """Fresh gate per signal: silero carries RNN state across push() calls."""
    g = fresh_gate()
    probs: list[float] = []
    for i in range(0, len(samples), STFT_HOP * 8):
        probs.extend(g.push(samples[i:i + STFT_HOP * 8]))
    if not probs:
        return {"gate_mean": None, "gate_frac": None, "gate_segmax": None}
    s = sorted(probs)
    # seg_max in tests/test_vad.py looks only at the FIRST 0.5 s. Both the
    # half-second peak and the whole-stream peak are reported because the
    # project's "0.0205" is a mean over 90 s and its "0.66-0.97" is a max over
    # 0.5 s. Comparing those two numbers is the mistake this file exists to
    # close, so it reports every statistic side by side instead of picking.
    first = int(0.5 * SR) // vad.FRAME_SAMPLES
    return {"gate_mean": round(sum(s) / len(s), 4),
            "gate_frac": round(
                sum(1 for p in s if p >= watcher.GATE_SPEECH) / len(s), 4),
            "gate_segmax": round(max(probs[:max(first, 1)]), 4),
            "gate_max": round(max(probs), 4)}


def cpu_pct(samples: np.ndarray, gated: bool) -> float:
    gate = fresh_gate() if gated else None
    w = watcher.Watcher(SR, gate=gate)
    t0 = time.perf_counter()
    for i in range(0, len(samples), STFT_HOP * 8):
        w.push(samples[i:i + STFT_HOP * 8])
    return (time.perf_counter() - t0) / DUR_S * 100.0


def emitted_pitch(samples: np.ndarray, gated: bool) -> dict:
    """
    What the watcher ACTUALLY reports. This is the correctness half.

    If a hum produces hz_p05 values inside the speech range, then the gate
    failure is not a speed bug, it is a table bug.
    """
    gate = fresh_gate() if gated else None
    w = watcher.Watcher(SR, gate=gate)
    for i in range(0, len(samples), STFT_HOP * 8):
        w.push(samples[i:i + STFT_HOP * 8])
    utts = [e for e in w.emitted if e.kind == watcher.UTTERANCE]
    hz = [e.data["hz_p05"] for e in utts if e.data.get("hz_p05")]
    return {"utterances": len(utts),
            "hz_p05_values": sorted(hz),
            "voiced_ratio_max": max((e.data.get("voiced_ratio") or 0.0)
                                    for e in utts) if utts else None}


def row(label: str, samples: np.ndarray, *, full: bool = False) -> dict:
    """
    `full` adds the ungated CPU number.

    Ungated is ~10 s of wall per row on this box -- an ungated watcher burns
    >100% of a core on a hum, which is itself the finding, but it is far too
    expensive to run on every row. It is run on a named subset instead of on
    everything, so the table says which rows carry the expensive number.
    """
    r = {"signal": label, **gate_stats(samples)}
    r["cpu_gated_pct"] = round(cpu_pct(samples, True), 2)
    if full:
        r["cpu_ungated_pct"] = round(cpu_pct(samples, False), 2)
        r["emitted_gated"] = emitted_pitch(samples, True)
    return r


def main() -> None:
    if vad.model_path() is None:
        print(json.dumps({"error": "gate weights absent"}))
        return
    speech = speech_clip()
    rows: list[dict] = []
    part_a: list[dict] = []

    # Reference: speech alone.
    sp = row("real_speech", speech, full=True)
    rows.append(sp)

    # PART A0 -- re-run the project's OWN regression fixture
    # (tests/test_vad.py::TestDeclaredLimits) next to the hum fixture. Same
    # frequency, same claim, two fixtures and two statistics. If these two
    # blocks disagree, the project's documented weakness was measured on a
    # signal that a real hum does not look like, and that is worth knowing
    # before anyone designs a fix around it.
    part_a0 = []
    for f0 in (150, 175, 200):
        part_a0.append(row(
            f"testvad_stack_{f0}Hz_amp0.3",
            harmonic_stack(f0, 0.3, harms=TEST_VAD_HARMS, normalise=False),
            full=(f0 in (150, 200))))
    for f0 in (150, 175, 200):
        pure = np.sin(2.0 * np.pi * f0 * np.arange(int(DUR_S * SR)) / SR)
        part_a0.append(row(
            f"testvad_tone_{f0}Hz_amp0.3",
            (0.3 * pure).astype(np.float32)))

    # PART A1 -- hum alone. Cost and correctness, per mains fundamental.
    for f0 in FUNDAMENTALS:
        for amp in AMPS:
            part_a.append(row(f"hum_stack_{f0}Hz_amp{amp}", harmonic_stack(f0, amp),
                              full=(f0 == 150 and amp in (0.05, 0.5))))

    # PART A2 -- the case hum_sweep cannot see: speech WITH the hum.
    for f0 in FUNDAMENTALS:
        for snr in SNRS_DB:
            h = harmonic_stack(f0, 1.0)
            part_a.append(row(f"speech+hum_{f0}Hz_{snr:g}dB", mix(speech, h, snr),
                              full=(f0 == 150 and snr in (10.0, 0.0))))

    # PART B -- per-device profile, over-subtraction sweep, on the band the
    # gate fails. Calibrated on 60 Hz hum at 0.05 (a quiet device).
    calib = harmonic_stack(60, 0.05)
    prof = noise_profile(calib)
    part_b = []
    for f0 in (150, 175, 200):
        hum = harmonic_stack(f0, 0.05)
        for alpha in (1.0, 2.0):
            cond = condition(hum, prof, alpha)
            part_b.append({"signal": f"hum_{f0}Hz_conditioned_a{alpha}",
                           **gate_stats(cond)})
        for snr in (20.0, 10.0):
            mixd = mix(speech, harmonic_stack(f0, 1.0), snr)
            for alpha in (1.0, 2.0):
                cond = condition(mixd, prof, alpha)
                part_b.append({"signal": f"speech+hum_{f0}Hz_{snr:g}dB_a{alpha}",
                               **gate_stats(cond)})
    part_b.append({"signal": "real_speech_conditioned_a1.0",
                   **gate_stats(condition(speech, prof, 1.0))})
    part_b.append({"signal": "real_speech_conditioned_a2.0",
                   **gate_stats(condition(speech, prof, 2.0))})

    # PART C -- the fan spins up. Same profile, 6x louder device noise.
    part_c = []
    for amp in (0.10, 0.30):
        hum = harmonic_stack(150, amp)
        for alpha in (1.0, 2.0):
            part_c.append({"signal": f"hum_150Hz_amp{amp}_profile_a{alpha}",
                           **gate_stats(condition(hum, prof, alpha))})

    def hz(rows_):
        v = [x for r in rows_ for x in r.get("hz_p05_values") or []]
        return {"n": len(v), "min": round(min(v), 1) if v else None,
                "max": round(max(v), 1) if v else None}

    out = {
        "note": ("Does the gate's 150-200 Hz failure hurt correctness or only "
                 "CPU? Part A: hum alone (cost+emitted pitch) and speech+hum "
                 "(the case hum_sweep.py cannot see). Part B: a per-device "
                 "noise profile applied as spectral subtraction to the frozen "
                 "gate -- the wearable analogue of ManiWAV's robot-noise "
                 "calibration. Part C: the same profile applied to a device "
                 "6x louder than calibration (the fan spinning up)."),
        "methods": {
            "sr": SR, "dur_s": DUR_S, "fundamentals": FUNDAMENTALS,
            "amps": AMPS, "snrs_db": SNRS_DB, "harmonics": N_HARM,
            "gate_speech_threshold": watcher.GATE_SPEECH,
            "stft": {"frame": STFT_FRAME, "hop": STFT_HOP},
            "fresh_gate_per_signal": True,
            "overlaps_hum_sweep": (
                "hum-alone rows deliberately reproduce hum_sweep.py's "
                "generator so the two files are comparable; the speech+hum "
                "mixture rows and Parts B/C are new"),
        },
        "part_a0_testvad_fixture": part_a0,
        "part_a": part_a,
        "part_b_device_conditioning": part_b,
        "part_c_louder_device": part_c,
        "summary": {
            "real_speech": {"gate": sp["gate_mean"], "frac": sp["gate_frac"],
                            "hz_p05": sp["emitted_gated"]["hz_p05_values"]},
            "hum_alone_hz_p05": hz(part_a[:len(FUNDAMENTALS) * len(AMPS)]),
            "testvad_fixture_rows": part_a0,
            "hum_that_beats_threshold": sorted(
                {r["signal"].rsplit("_amp", 1)[0] for r in part_a
                 if r.get("gate_frac")}),
            "mixture_that_beats_threshold": sorted(
                {r["signal"].rsplit("_", 1)[0] for r in part_a
                 if "dB" in r["signal"] and (r.get("gate_frac") or 0) > 0.0}),
            "conditioning_recovered_rejection": sorted(
                r["signal"] for r in part_b
                if r["signal"].startswith("hum_") and r.get("gate_frac") == 0.0),
            "conditioning_speech_damage": sorted(
                r["signal"] for r in part_b
                if r["signal"].startswith(("real_speech", "speech+hum"))
                and (r.get("gate_frac") or 0) < 0.3),
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
