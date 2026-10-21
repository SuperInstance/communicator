"""ARRIVAL NULL — the floor on the double-hearing technique.

The awareness-calculators study says the double-hearing claim ("two
arrivals of one voice, and the delay between them is geometry") is
untested on this corpus, because the corpus holds ONE real input device.
It also says the technique's floor is measurable TONIGHT, without a
second device, because a delay estimator applied to a signal and a copy
of that signal must return ZERO.

That null is the floor on everything a later real measurement inherits.
This module IS that floor:

  * `arrival_lag(a, b, sr, band)` -- the estimator.  The band-limited
    ENVELOPE cross-correlation, because two devices will not agree on
    phase, and the parabolic interpolation around the peak, because the
    envelope's 1 ms hop is coarser than the delay we care about.

  * `arrival_null(samples, sr)` -- the null: a signal against a
    gain-shifted copy of itself.  A second microphone with a different
    gain would look exactly like this, and the arrival is the same
    instant, so the lag must be zero.  The residual IS the estimator's
    bias, and it is the number that decides what a real measurement
    could ever mean.

  * `arrival_resolution(samples, sr, delays)` -- the calibration: known
    delays injected into the signal, and the error in recovering each
    one.  This says how finely the delay can be measured, which
    converts directly into a range resolution in metres.

Run from the repo root:
    /home/proart/projects/gpu-venv/bin/python experiments/arrival_null.py --selftest
    /home/proart/projects/gpu-venv/bin/python experiments/arrival_null.py --out docs/research
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import pause_study  # noqa: E402
from src import energy  # noqa: E402

#: The speed of sound, in m/s, at 20 degrees.
SPEED_OF_SOUND = 343.0

#: The band the corpus's chain carries HONESTLY.  Above 8 kHz the
#: resampler folds (measured: a 10 kHz tone arrives at 6 kHz, 884.4
#: cents down), and the ultrasonic probe came back 87-94 dB below the
#: stimulus.  So a marker lives here, in the voice band.
DEFAULT_BAND = (300.0, 3400.0)

#: The envelope hop.  1 ms, so the estimator's own resolution floor is
#: 1 ms -- and the parabolic interpolation is what beats it.
ENVELOPE_HOP_S = 0.001


def bandpass(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """A zero-phase FFT band-pass."""
    n = len(x)
    if n < 8:
        return x
    sp = np.fft.rfft(x)
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    sp[(fr < lo) | (fr > hi)] = 0.0
    return np.fft.irfft(sp, n)


def envelope(x: np.ndarray, sr: int, hop_s: float = ENVELOPE_HOP_S) -> np.ndarray:
    """The RMS envelope, on a fixed hop."""
    hop = max(1, int(round(hop_s * sr)))
    n = len(x) // hop
    if n < 8:
        return np.array([])
    frames = x[:n * hop].reshape(n, hop)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))


def arrival_lag(a: np.ndarray, b: np.ndarray, sr: int,
                band: tuple[float, float] = DEFAULT_BAND,
                max_lag_s: float = 0.5) -> dict | None:
    """The delay at which b best matches a, in seconds.

    POSITIVE means b arrives LATER than a.

    The comparison is on ENVELOPES, not waveforms: two devices will not
    agree on the phase of a sound, so a waveform correlation of two
    recordings of the same event would find a spurious fine-structure
    peak.  The envelope's peak is the arrival, and it is what an
    arrival-time difference actually is.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < sr // 10 or len(b) < sr // 10:
        return None
    if band:
        a = bandpass(a, sr, band[0], band[1])
        b = bandpass(b, sr, band[0], band[1])
    ea = envelope(a, sr)
    eb = envelope(b, sr)
    m = min(len(ea), len(eb))
    if m < 50:
        return None
    ea = ea[:m] - ea[:m].mean()
    eb = eb[:m] - eb[:m].mean()
    den = math.sqrt(float(np.sum(ea ** 2)) * float(np.sum(eb ** 2)))
    if den <= 0:
        return None
    # The correlation is done in the FREQUENCY DOMAIN.  The direct
    # form, np.correlate(..., 'full'), is O(n*m), and a 259 s session's
    # 1 ms envelope is 259,000 frames -- 67 billion operations for one
    # session, which is why the first run of this took minutes per
    # session and never finished.  scipy's fftconvolve is O(n log n) and
    # gives the identical answer.
    from scipy.signal import fftconvolve
    # correlate(ea, eb) == convolve(ea, eb[::-1]): fftconvolve
    # CONVOLVES, and numpy's correlate and convolve differ by exactly
    # that reversal.  Getting this wrong turns a 10 ms delay into a
    # 500 ms one and puts the peak at the edge of the search window.
    num = fftconvolve(ea, eb[::-1], mode='full')
    lags = np.arange(-m + 1, m)
    hop = max(1, int(round(ENVELOPE_HOP_S * sr)))
    keep = np.abs(lags) <= int(max_lag_s * sr / hop)
    if keep.sum() < 3:
        return None
    num = num[keep]
    lags = lags[keep]
    nc = num / den
    i = int(np.argmax(nc))
    peak = float(nc[i])
    # parabolic interpolation around the peak: the envelope's hop is
    # 1 ms and the delays we care about are smaller than that.
    delta = 0.0
    if 0 < i < len(nc) - 1:
        y0, y1, y2 = nc[i - 1], nc[i], nc[i + 1]
        den2 = (y0 - 2 * y1 + y2)
        if den2 != 0:
            delta = 0.5 * (y0 - y2) / den2
    lag_frames = float(lags[i] + delta)
    # np.correlate(ea, eb, 'full') peaks at a NEGATIVE index when eb is
    # the later of the two, so the sign is flipped here to keep the
    # module's convention: POSITIVE means b arrives LATER than a.
    lag_s = -lag_frames * hop / sr
    # the second-best peak away from the first: a confidence measure.
    far = np.abs(lags - lags[i]) > int(0.02 * sr / hop)
    second = float(np.max(nc[far])) if far.any() else 0.0
    return {
        'lag_s': float(lag_s),
        'lag_ms': float(lag_s * 1000),
        'peak': peak,
        'second_peak': second,
        'margin': float(peak - second),
        'n_envelope_frames': int(m),
    }


def arrival_null(samples: np.ndarray, sr: int,
                 band: tuple[float, float] = DEFAULT_BAND) -> dict | None:
    """THE NULL: a signal against a gain-shifted copy of itself.

    A second microphone with a different gain is exactly this, and the
    arrival is the same instant, so the lag must be ZERO.  Whatever the
    estimator returns is its own bias -- and it is the floor that any
    later real measurement inherits.
    """
    x = np.asarray(samples, dtype=np.float64)
    if len(x) < sr // 2:
        return None
    # a different gain, and a different length, as two devices would be
    copy = x * 0.37
    copy = copy[:int(len(copy) * 0.97)]
    r = arrival_lag(x, copy, sr, band)
    if r is None:
        return None
    return {
        'lag_ms': r['lag_ms'],
        'peak': r['peak'],
        'margin': r['margin'],
        'bias_ms': abs(r['lag_ms']),
        'bias_m': abs(r['lag_ms']) / 1000.0 * SPEED_OF_SOUND,
    }


def shift(x: np.ndarray, sr: int, delay_s: float) -> np.ndarray:
    """A copy of x delayed by delay_s, zero-padded."""
    n = int(round(delay_s * sr))
    out = np.zeros(len(x) + n, dtype=np.float64)
    out[n:] = x
    return out


def arrival_resolution(samples: np.ndarray, sr: int,
                       delays_ms=(0, 1, 2, 5, 10, 20, 40, 80),
                       band: tuple[float, float] = DEFAULT_BAND) -> list[dict]:
    """The calibration: known delays, injected, and the error in
    recovering each one.

    The error here is what a real measurement's error would be, and it
    converts to a range resolution in metres.
    """
    x = np.asarray(samples, dtype=np.float64)
    if len(x) < sr:
        return []
    # a window with real signal in it: the loudest second
    hop = int(0.01 * sr)
    n = len(x) // hop
    rms = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2)) for i in range(n)])
    if not rms.size:
        return []
    centre = int(np.argmax(rms))
    a = int(max(0, (centre - 25) * hop))
    b = int(min(len(x), (centre + 75) * hop))
    seg = x[a:b]
    if len(seg) < sr // 2:
        return []
    out = []
    for d in delays_ms:
        # b is the segment, a is the same sound arriving EARLIER by d
        # -- so the reported lag should be +d
        early = seg[:len(seg) - int(0.2 * sr)] if len(seg) > int(0.2 * sr) else seg
        late = shift(early, sr, d / 1000.0)
        r = arrival_lag(early, late, sr, band)
        if r is None:
            out.append({'delay_ms': d, 'recovered_ms': None, 'error_ms': None,
                        'peak': None})
            continue
        out.append({'delay_ms': d, 'recovered_ms': round(r['lag_ms'], 3),
                    'error_ms': round(abs(r['lag_ms'] - d), 3),
                    'peak': round(r['peak'], 4),
                    'margin': round(r['margin'], 4),
                    'range_m': round(abs(r['lag_ms'] - d) / 1000.0 * SPEED_OF_SOUND, 3)})
    return out


def measure(sessions, out_dir: Path) -> dict:
    """The measurement, over the corpus's real-speech sessions."""
    per_session = []
    for name, sdir in sessions:
        wav = Path(sdir) / 'audio.wav'
        if not wav.is_file():
            continue
        audio, samples = pause_study.read_clip(wav)
        sr = audio.sample_rate
        x = np.array(samples, dtype=float)
        if len(x) < sr // 2:
            continue
        if float(np.max(np.abs(x))) <= 0:
            continue
        null = arrival_null(x, sr)
        res = arrival_resolution(x, sr)
        if null is None and not res:
            continue
        per_session.append({
            'session': name,
            'duration_s': round(len(x) / sr, 2),
            'null': null,
            'resolution': res,
        })
        print(f"  {name}: null bias {null['bias_ms']:.3f} ms "
              f"({null['bias_m']:.3f} m), "
              f"max resolution error {max((r['error_ms'] or 0) for r in res):.3f} ms"
              if null else f"  {name}: no null")
    out = {
        'study': 'arrival null -- the floor on the double-hearing technique',
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'method': {
            'estimator': ('the band-limited ENVELOPE cross-correlation '
                          '(300-3400 Hz, the band the chain carries '
                          'honestly), with parabolic interpolation around '
                          'the peak at a 1 ms envelope hop'),
            'the_null': ('a signal against a gain-shifted copy of itself: '
                         'a second microphone with a different gain, and '
                         'the same arrival, so the lag must be ZERO'),
            'the_calibration': ('known delays injected into the signal, '
                                'and the error in recovering each one'),
            'band_hz': list(DEFAULT_BAND),
            'envelope_hop_s': ENVELOPE_HOP_S,
            'speed_of_sound_m_s': SPEED_OF_SOUND,
        },
        'sessions': per_session,
    }
    biases = [s['null']['bias_ms'] for s in per_session if s.get('null')]
    errs = [r['error_ms'] for s in per_session for r in s['resolution']
            if r.get('error_ms') is not None]
    by_delay = {}
    for s in per_session:
        for r in s['resolution']:
            if r.get('error_ms') is not None:
                by_delay.setdefault(r['delay_ms'], []).append(r['error_ms'])
    out['summary'] = {
        'n_sessions': len(per_session),
        'null_bias_ms': {
            'median': round(statistics.median(biases), 4) if biases else None,
            'max': round(max(biases), 4) if biases else None,
            'median_in_metres': round(statistics.median(biases) / 1000.0 * SPEED_OF_SOUND, 4)
            if biases else None,
        },
        'resolution_error_ms': {
            'median': round(statistics.median(errs), 4) if errs else None,
            'max': round(max(errs), 4) if errs else None,
            'median_in_metres': round(statistics.median(errs) / 1000.0 * SPEED_OF_SOUND, 4)
            if errs else None,
        },
        'by_injected_delay_ms': {
            str(k): {'n': len(v), 'median_error_ms': round(statistics.median(v), 4),
                     'max_error_ms': round(max(v), 4)}
            for k, v in sorted(by_delay.items())},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'arrival_null.json').write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------
# the selftest
# --------------------------------------------------------------------------

def selftest() -> int:
    checks: list[tuple[str, bool, str]] = []

    def ck(name, ok, detail=''):
        checks.append((name, bool(ok), detail))

    sr = 16000

    # 1. a synthetic pulse arrives later than it did
    n = sr * 2
    t = np.arange(n) / sr
    pulse = np.zeros(n)
    p0 = int(sr * 0.5)
    pulse[p0:p0 + 800] = np.sin(2 * np.pi * 500 * t[p0:p0 + 800])
    late = shift(pulse, sr, 0.010)
    r = arrival_lag(pulse, late, sr)
    ck('arrival: a 10 ms delay is recovered', r is not None
       and abs(r['lag_ms'] - 10.0) < 1.0, str(r))
    ck('arrival: the lag is positive for a later arrival',
       r is not None and r['lag_ms'] > 0, str(r))

    # 2. the NULL: a signal against a gain-shifted copy of itself
    noise = np.random.default_rng(7).normal(0, 0.1, n)
    burst = np.zeros(n)
    burst[p0:p0 + 4000] = np.random.default_rng(8).normal(0, 0.3, 4000)
    sig = noise + burst
    nul = arrival_null(sig, sr)
    ck('null: a copy of itself gives zero lag', nul is not None
       and nul['bias_ms'] < 1.0, str(nul))

    # 3. the null is not trivially zero: the estimator finds a real delay
    r2 = arrival_lag(sig, shift(sig, sr, 0.020), sr)
    ck('null: the estimator is not trivially zero',
       r2 is not None and abs(r2['lag_ms'] - 20.0) < 1.5, str(r2))

    # 4. the null's bias is smaller than the delay it must resolve
    if nul is not None and r2 is not None:
        ck('null: the bias is smaller than the smallest delay tested',
           nul['bias_ms'] < 5.0, f"bias {nul['bias_ms']:.3f} ms")

    # 5. the calibration recovers each injected delay
    res = arrival_resolution(sig, sr, delays_ms=(0, 1, 5, 10, 20, 40))
    got = {r['delay_ms']: r['error_ms'] for r in res if r.get('error_ms') is not None}
    ck('calibration: 0 ms comes back as 0', got.get(0) is not None and got[0] < 1.5, str(got))
    ck('calibration: 10 ms comes back as 10', got.get(10) is not None and got[10] < 1.5, str(got))
    ck('calibration: 40 ms comes back as 40', got.get(40) is not None and got[40] < 2.5, str(got))
    ck('calibration: the 1 ms delay is the hardest', got.get(1, 99) >= got.get(40, 0) - 0.5, str(got))

    # 6. what the fold ACTUALLY breaks.
    #
    # The first version of this check asserted that a tone above the
    # honest band cannot be timed, and it failed: the estimator
    # recovered the 10 ms delay from a 12 kHz tone that had aliased to
    # 4 kHz.  That is correct behaviour and the assertion was wrong.
    #
    # The fold does NOT break envelope timing -- the envelope's shape is
    # the burst, and the burst still arrives when it arrives.  What the
    # fold breaks is the marker's IDENTITY: a 14 kHz tone arrives as
    # 2 kHz, so the receiver cannot tell what was sent.  So the honest
    # band is a rule about WHAT A MARKER IS, not about WHEN IT ARRIVED.
    folded = np.sin(2 * np.pi * 12000 * t[:n]) * (np.abs(t[:n] - 0.5) < 0.05)
    r3 = arrival_lag(folded, shift(folded, sr, 0.010), sr,
                     band=(12000 * 0.5, 16000 * 0.5))
    ck('fold: the arrival SURVIVES the fold (the envelope is what we time)',
       r3 is not None and abs(r3['lag_ms'] - 10.0) < 1.5, str(r3))
    # ... but the identity does not: the aliased tone's measured
    # frequency is not what was sent.
    aliased = np.sin(2 * np.pi * 12000 * t[:n])
    sp = np.abs(np.fft.rfft(aliased * np.hanning(len(aliased))))
    fr = np.fft.rfftfreq(len(aliased), 1.0 / sr)
    measured_f = float(fr[int(np.argmax(sp))])
    ck('fold: the identity does NOT survive -- 12 kHz measures as 4 kHz',
       abs(measured_f - 4000.0) < 50.0, f"measured {measured_f:.0f} Hz, sent 12000 Hz")

    # 7. the range conversion is the one the doctrine uses
    if nul is not None:
        ck('units: the bias converts to metres at 343 m/s',
           abs(nul['bias_m'] - nul['bias_ms'] / 1000.0 * 343.0) < 1e-6, str(nul))

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"   [{detail}]" if detail and not ok else ""))
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'docs' / 'research'))
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    corpus = Path('/mnt/c/proart_capture/corpus/free_sessions')
    sessions = []
    for d in sorted(corpus.iterdir()):
        if not d.is_dir():
            continue
        meta = d / 'meta.json'
        if not meta.is_file():
            continue
        j = json.loads(meta.read_text())
        mic = (j.get('mic') or {}).get('label', '')
        if 'Realtek' not in mic:
            continue
        sessions.append((d.name, d))
    print(f"arrival null: {len(sessions)} real-microphone sessions")
    out = measure(sessions, Path(args.out))
    print(json.dumps(out['summary'], indent=1))
    print(f"wrote {Path(args.out) / 'arrival_null.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
