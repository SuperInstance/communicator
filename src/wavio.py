"""
wavio — WAV in, clean mono float samples out.

The analysis grid for the whole pipeline is mono 16 kHz. forge-soniqo's
defect (design doc §0) was parsing the sample rate and then analysing as if
it were something else; here the sample rate travels WITH the samples and
resampling to the analysis rate is explicit and testable.

Supported: PCM 8/16/32-bit, mono or stereo (stereo is averaged to mono —
forge-soniqo took channel 0 and lost the other half; averaging is the
deliberate difference). 24-bit and non-PCM encodings are rejected loudly
rather than silently misdecoded.

Non-finite guard: int PCM cannot encode NaN or inf, so non-finite values can
only enter through the in-memory API (a caller handing us a poisoned list —
the ring-buffer defect class, where one NaN contaminates every later frame).
`sanitize()` replaces them with 0.0 and COUNTS them, so the poison is
observable in the result instead of silently clipped.
"""

from __future__ import annotations

import array
import math
import wave
from dataclasses import dataclass
from pathlib import Path

ANALYSIS_SR = 16000  # the only sample rate the analysis stages ever see


class WavFormatError(ValueError):
    """The WAV file uses an encoding this reader does not support."""


@dataclass
class Audio:
    """Mono float samples in [-1, 1] plus the sample rate they are really at."""

    samples: list[float]
    sample_rate: int
    poisoned_samples: int = 0  # non-finite values replaced with 0.0 on read-in

    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def sanitize(samples: list[float]) -> tuple[list[float], int]:
    """Replace NaN/inf with 0.0 and report how many were found.

    Returns (clean_samples, poisoned_count). Not silent: the count is carried
    on the Audio object and into the score so a poisoned input is visible in
    the output, not just survived by it.
    """
    clean: list[float] = []
    poisoned = 0
    for s in samples:
        if math.isfinite(s):
            clean.append(s)
        else:
            clean.append(0.0)
            poisoned += 1
    return clean, poisoned


def read_wav(path: str | Path) -> Audio:
    """Read a WAV file to mono floats, resampled to ANALYSIS_SR.

    8-bit PCM is unsigned; 16/32-bit are little-endian signed. Stereo frames
    are averaged channel-by-channel. Anything else raises WavFormatError.
    """
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())

    if sr <= 0:
        raise WavFormatError(f"nonsense sample rate: {sr}")

    if sampwidth == 1:
        vals = [(b - 128) / 128.0 for b in raw]
    elif sampwidth == 2:
        arr = array.array("h")
        arr.frombytes(raw)
        if arr.itemsize != 2:
            raise WavFormatError("platform 'h' is not 2 bytes")
        vals = [v / 32768.0 for v in arr]
    elif sampwidth == 4:
        arr = array.array("i")
        arr.frombytes(raw)
        if arr.itemsize != 4:
            raise WavFormatError("platform 'i' is not 4 bytes")
        vals = [v / 2147483648.0 for v in arr]
    else:
        raise WavFormatError(
            f"{sampwidth * 8}-bit PCM not supported (8/16/32-bit only)")

    if n_channels < 1:
        raise WavFormatError(f"nonsense channel count: {n_channels}")
    if n_channels == 1:
        mono = vals
    else:
        # Average all channels. i-th frame's channels are contiguous.
        mono = [
            sum(vals[i * n_channels:(i + 1) * n_channels]) / n_channels
            for i in range(len(vals) // n_channels)
        ]

    clean, poisoned = sanitize(mono)
    audio = Audio(samples=clean, sample_rate=sr, poisoned_samples=poisoned)
    return resample(audio, ANALYSIS_SR)


def resample(audio: Audio, target_sr: int) -> Audio:
    """Linear-interpolation resample to target_sr (identity if already there).

    Linear interpolation is honest about what it is: it does NOT anti-alias.
    Fixtures and speech energy below ~7 kHz survive; content near the old
    Nyquist will alias. That limit is stated here rather than hidden.
    """
    if audio.sample_rate == target_sr:
        return audio
    n = len(audio.samples)
    if n == 0:
        return Audio(samples=[], sample_rate=target_sr,
                     poisoned_samples=audio.poisoned_samples)
    step = audio.sample_rate / target_sr
    out: list[float] = []
    pos = 0.0
    while pos < n - 1:
        i = int(pos)
        frac = pos - i
        out.append(audio.samples[i] * (1.0 - frac) + audio.samples[i + 1] * frac)
        pos += step
    return Audio(samples=out, sample_rate=target_sr,
                 poisoned_samples=audio.poisoned_samples)
