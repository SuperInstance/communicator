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
    # The rate the FILE was in, before resampling. `sample_rate` above is the
    # rate the samples are in NOW, which after read_wav() is always the
    # analysis rate. Without this field the original rate is unrecoverable,
    # and a caller that needs to know it -- the dialogue intake, checking that
    # two channels of one take came off devices with the same clock -- has
    # nothing to read. It was always 16000 before this field existed.
    source_sample_rate: int | None = None

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
    audio = Audio(samples=clean, sample_rate=sr, poisoned_samples=poisoned,
                  source_sample_rate=sr)
    return resample(audio, ANALYSIS_SR)


def write_wav(path: str | Path, samples, sample_rate: int = ANALYSIS_SR) -> Path:
    """Write mono float samples as a 16-bit PCM WAV.

    THE REPO HAD NO WRITER. `read_wav` could read every file the protocol
    asks for and nothing in the repo could produce one, so every synthetic
    corpus in the test suite carried its own copy of this function, and the
    protocol -- "Format: WAV, PCM 16-bit" -- described a file nothing here
    could write. This is that writer.

    The pair is verified on the round trip: what goes in is what comes back,
    to within the 16-bit quantisation the format imposes (1 LSB = 3.05e-5
    full scale). Samples outside [-1, 1] are CLIPPED, not wrapped, because a
    wrapped sample is a sign flip and a clipped one is at least audible as
    loud.
    """
    out_path = Path(path)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    vals = array.array("h")
    for s in samples:
        v = int(round(max(-1.0, min(1.0, float(s))) * 32767.0))
        vals.append(max(-32768, min(32767, v)))
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(vals.tobytes())
    return out_path


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
                     poisoned_samples=audio.poisoned_samples,
                     source_sample_rate=audio.source_sample_rate)
    step = audio.sample_rate / target_sr
    out: list[float] = []
    pos = 0.0
    while pos < n - 1:
        i = int(pos)
        frac = pos - i
        out.append(audio.samples[i] * (1.0 - frac) + audio.samples[i + 1] * frac)
        pos += step
    return Audio(samples=out, sample_rate=target_sr,
                 poisoned_samples=audio.poisoned_samples,
                 source_sample_rate=audio.source_sample_rate)
