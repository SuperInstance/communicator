"""
pitch — YIN-style f0 tracker for the human speaking range.

Why YIN and not plain autocorrelation: autocorrelation alone answers "how
correlated is the signal with itself at lag tau" and happily locks onto a
harmonic or half-period. The cumulative-mean-normalized difference function
(YIN, de Cheveigné & Kawahara 2002) makes each dip's depth comparable and,
by taking the FIRST dip under an absolute threshold, prefers the true period
over its multiples. forge-soniqo's ±80% tolerance (design doc §0) is the
cost of not doing this properly.

Frame geometry: 50 ms window, 10 ms hop, at 16 kHz (800/160 samples). The
difference function is summed over a fixed W = frame_len - tau_max samples
so every lag sees the same number of terms (summing fewer terms at longer
lags is the classic normalization bug that biases toward slow periods).

Range: 70–400 Hz covers the speaking voice (male modal ~85–180 Hz, female
~165–255 Hz, plus creaky and head-register extremes). Content outside the
range is reported as unvoiced, not force-fit to the nearest in-range f0.

Emitting nothing beats guessing: every frame carries a confidence
(1 - d'(tau)); below `min_confidence` the frame is unvoiced and NO pitch is
emitted. PITCH is a track downstream consumers will trust, so a gap is
strictly better than a wrong number. Silence never even reaches YIN in the
pipeline — the energy gate in energy.py runs first — but this module also
refuses non-finite input per frame (the ring-buffer defect class: one NaN
contaminates everything after it).

The loudness gate lives HERE as well, not only in the pipeline, because YIN
confidence cannot provide it: the cumulative-mean normalisation makes d'(tau)
amplitude-invariant by construction, so a periodic signal at ANY amplitude
— a fan, a keyboard, a synth leaking from speakers — scores confidence 1.0.
A frame whose RMS is below `energy_floor` is unvoiced before YIN is ever
asked. The default, 1e-3 (≈ −60 dBFS), is ten times stricter than
energy.ABS_FLOOR: claiming a pitch is a stronger statement than "not
silent", so its floor must be stricter too.

Pure standard library. The difference function is evaluated as
d(tau) = e_head + e_tau - 2*c(tau) so the per-term arithmetic runs inside
math.fsum(map(operator.mul, ...)) at C speed; pure-Python inner loops over
~10^5 terms per frame would be needlessly slow.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from operator import mul

from .energy import frame_rms

FMIN = 70.0    # Hz, lower bound of the speaking range
FMAX = 400.0   # Hz, upper bound

# RMS below this is not voiced, whatever YIN thinks of its periodicity.
# −60 dBFS: practical room noise sits above this, real speech above it easily.
ENERGY_FLOOR = 1e-3


@dataclass
class PitchFrame:
    """One analysis position.

    hz is None when the frame is unvoiced — deliberately not 0.0, so that
    "no measurement" cannot be confused with "measured zero".
    """

    index: int
    time: float          # seconds, frame center in the source clip
    hz: float | None
    confidence: float    # 1 - d'(tau) at the chosen lag; 0.0 if unvoiced


class PitchTracker:
    """Fixed-grid YIN tracker over a whole clip, mono floats at 16 kHz."""

    def __init__(
        self,
        sample_rate: int,
        frame_len: int = 800,      # 50 ms at 16 kHz
        hop: int = 160,            # 10 ms at 16 kHz
        fmin: float = FMIN,
        fmax: float = FMAX,
        threshold: float = 0.15,   # YIN absolute threshold on d'(tau)
        min_confidence: float = 0.5,
        energy_floor: float = ENERGY_FLOOR,
    ):
        if sample_rate <= 0:
            raise ValueError(f"nonsense sample rate: {sample_rate}")
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self.hop = hop
        self.fmin = fmin
        self.fmax = fmax
        self.threshold = threshold
        self.min_confidence = min_confidence
        self.energy_floor = energy_floor

        self.tau_min = max(1, int(math.floor(sample_rate / fmax)))
        self.tau_max = int(math.ceil(sample_rate / fmin))
        if self.tau_max <= self.tau_min:
            raise ValueError("pitch range is empty at this sample rate")
        # Fixed comparison window: every lag sees the same number of terms.
        self.win = frame_len - self.tau_max
        if self.win < self.tau_min:
            raise ValueError(
                f"frame_len={frame_len} too short for fmin={fmin}Hz at "
                f"{sample_rate}Hz (need frame_len >= tau_max + tau_min)")
        self.poisoned_frames = 0
        self.quiet_frames = 0  # rejected by the energy floor, pre-YIN

    def track(self, samples: list[float]) -> list[PitchFrame]:
        """Frame the clip and estimate f0 at each position."""
        frames: list[PitchFrame] = []
        for i in range(self._n_positions(len(samples))):
            start = i * self.hop
            frame = samples[start:start + self.frame_len]
            center = (start + len(frame) / 2.0) / self.sample_rate
            hz, conf = self.estimate(frame)
            frames.append(PitchFrame(index=i, time=center, hz=hz,
                                     confidence=conf))
        return frames

    def n_positions(self, n_samples: int) -> int:
        return self._n_positions(n_samples)

    def _n_positions(self, n_samples: int) -> int:
        """Frames whose full window fits; the tail is dropped, not padded."""
        if n_samples < self.frame_len:
            return 0
        return 1 + (n_samples - self.frame_len) // self.hop

    def estimate(self, frame: list[float]) -> tuple[float | None, float]:
        """Estimate f0 for one window. Returns (hz_or_None, confidence)."""
        if len(frame) < self.frame_len:
            return None, 0.0
        if not self._finite(frame):
            return None, 0.0
        # The loudness gate, BEFORE periodicity is even computed: YIN
        # confidence is amplitude-invariant by construction (see module
        # docstring), so it can never be the thing that refuses a quiet fan.
        rms, _ = frame_rms(frame)
        if rms < self.energy_floor:
            self.quiet_frames += 1
            return None, 0.0
        return self._yin(frame)

    def _finite(self, frame: list[float]) -> bool:
        ok = all(math.isfinite(x) for x in frame)
        if not ok:
            self.poisoned_frames += 1
        return ok

    def _yin(self, frame: list[float]) -> tuple[float | None, float]:
        sr = self.sample_rate
        w = self.win
        head = frame[:w]  # the fixed comparison span x[0:w]

        # d(tau) = sum_{j<w} (x[j] - x[j+tau])^2
        #        = e_head + e_tau - 2 * c(tau)
        e_head = math.fsum(map(mul, head, head))
        d = [0.0] * (self.tau_max + 1)  # d[0] = 0 by definition (self-compare)
        for tau in range(1, self.tau_max + 1):
            seg = frame[tau:tau + w]
            c = math.fsum(map(mul, head, seg))
            e = math.fsum(map(mul, seg, seg))
            raw = e_head + e - 2.0 * c
            d[tau] = raw if raw > 0.0 else 0.0

        # Cumulative mean normalized difference d'(tau) = d(tau) / mean(d[1..tau])
        dp = [1.0] * len(d)
        running = 0.0
        for tau in range(1, len(d)):
            running += d[tau]
            dp[tau] = d[tau] * tau / running if running > 0.0 else 1.0

        tau = self._choose_tau(dp)
        if tau is None:
            return None, 0.0
        conf = max(0.0, min(1.0, 1.0 - dp[tau]))
        if conf < self.min_confidence:
            return None, conf
        hz = sr / self._parabolic(dp, tau)
        if not (self.fmin <= hz <= self.fmax):
            return None, conf
        return hz, conf

    def _choose_tau(self, dp: list[float]) -> int | None:
        """First local minimum of d' below the absolute threshold (YIN step 4).

        Falling back to the global minimum when nothing dips under threshold
        lets confidence decide emission: a shallow-but-clear dip still gets
        scored, and min_confidence turns weak maybes into honest gaps.
        """
        for tau in range(self.tau_min, self.tau_max + 1):
            if dp[tau] < self.threshold:
                while tau + 1 <= self.tau_max and dp[tau + 1] < dp[tau]:
                    tau += 1
                return tau
        best: int | None = None
        for tau in range(self.tau_min, self.tau_max + 1):
            if best is None or dp[tau] < dp[best]:
                best = tau
        return best

    def _parabolic(self, dp: list[float], tau: int) -> float:
        """Sub-sample lag refinement by parabolic fit around the dip."""
        if tau <= 0 or tau + 1 >= len(dp):
            return float(tau)
        denom = dp[tau - 1] - 2 * dp[tau] + dp[tau + 1]
        if abs(denom) < 1e-12:
            return float(tau)
        shift = 0.5 * (dp[tau - 1] - dp[tau + 1]) / denom
        return tau + max(-1.0, min(1.0, shift))


def smooth_octaves(
    frames: list[PitchFrame],
    max_semitone_jump: float = 9.0,
    hold: int = 3,
) -> list[PitchFrame]:
    """
    Suppress octave-scale outliers against the stable local median.

    The pYIN-lite post-pass (Mauch & Dixon 2014's temporal-smoothing idea,
    deterministic, pure stdlib; the reference implementation carries no
    license so only the idea was used). Day-0 real speech measured YIN's
    classic failure at 3.0% of adjacent voiced pairs; this is the
    conservative repair. WHAT IT DOES: within each contiguous voiced run
    of at least `hold` frames, every INTERIOR frame is compared to the
    median of its +-SMOOTH_HALF_WIDTH run-mates; a frame further than
    `max_semitone_jump` semitones out where exactly ONE octave of
    correction restores agreement is rescaled (f/2 or 2f -- the frame's
    own measurement, so within-octave contour motion survives, never a
    pasted-in median). Order-free: medians read originals, corrections
    write to the copy. Never touches run edges, gaps, short runs, or the
    tracker's voicedness; refuses corrections outside [FMIN, FMAX] and
    anything that is not a single-octave displacement (the f/3 error must
    NOT be snapped two octaves -- that maps it to 4f/3 and worsens it).
    Cannot see majority-octave runs (the error IS the consensus there) or
    bursts >= the window; those need pYIN's candidate prior, which this
    pass does not pretend to be.
    """
    out = list(frames)
    n = len(frames)
    i = 0
    while i < n:
        if frames[i].hz is None:
            i += 1
            continue
        j = i
        while j < n and frames[j].hz is not None:
            j += 1
        _smooth_run(frames, out, i, j, max_semitone_jump, hold)
        i = j
    return out


SMOOTH_HALF_WIDTH = 5    # +-5 frames = +-50 ms of context at the 10 ms hop


def _smooth_run(frames, out, start, end, max_semitone_jump, hold):
    """Correct octave outliers inside one voiced run [start, end)."""
    if end - start < hold:
        return
    vals = [f.hz for f in frames[start:end]]
    for k in range(1, len(vals) - 1):        # interior only: support both sides
        lo = max(0, k - SMOOTH_HALF_WIDTH)
        hi = min(len(vals), k + SMOOTH_HALF_WIDTH + 1)
        med = statistics.median(vals[lo:hi])
        val = vals[k]
        if not (val > 0.0 and med > 0.0):
            continue                          # defensive: hz is sr/tau > 0
        r = 12.0 * math.log2(val / med)       # semitones from local truth
        if abs(r) <= max_semitone_jump:
            continue                          # plausible, keep
        octaves = math.floor(r / 12.0 + 0.5)  # nearest whole-octave displacement
        if octaves not in (-1, 1):
            continue                          # not a single-octave error
        corrected = val / (2.0 ** octaves)
        if not (FMIN <= corrected <= FMAX):
            continue                          # the tracker could not have said this
        src = frames[start + k]
        out[start + k] = PitchFrame(index=src.index, time=src.time,
                                    hz=corrected, confidence=src.confidence)
