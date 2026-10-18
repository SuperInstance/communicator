"""
energy — RMS loudness per frame, and the speech/silence boundary.

Two jobs, one module, because they must agree on a single question: "is
there voice here right now?" The RMS frames feed VELOCITY (CC7) and the
silence segmentation gates everything else (SILENCE track, speech duration
for TEMPO, and the pre-check that keeps YIN from ever being asked about a
silent frame).

Non-finite guard (hard, at this layer): a NaN or inf reaching an RMS
computation poisons it, and in a streaming design the poisoned value poisons
every later consumer — the ring-buffer defect class named in the design doc.
`frame_rms` therefore (a) counts and zeroes non-finite inputs and (b) asserts
its own output is finite before returning. Even a caller who bypasses
wavio.sanitize() cannot get a non-finite loudness out of this module.

Segmentation uses two thresholds with hysteresis (snapkit's deadband idea,
adopted): speech is ENTERED above `on` and EXITED below `off = on/2`, so a
decaying syllable tail cannot chatter the boundary. Silences shorter than
`min_silence_s` are not rests — a 40 ms stop consonant gap is part of the
word, not phrase punctuation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

ABS_FLOOR = 1e-4        # RMS below this is silence even in a loud file
RELATIVE_ON = 0.10      # speech "on" threshold: 10% of the file's peak RMS
MIN_SILENCE_S = 0.150   # shorter gaps are inside a phrase, not rests
MIN_SPEECH_S = 0.060    # blips shorter than this are not speech


@dataclass
class EnergyFrame:
    index: int
    time: float      # seconds, window center
    rms: float       # always finite


@dataclass
class Segment:
    """A run of one state. start/end are seconds, end exclusive."""
    start: float
    end: float
    speech: bool

    def duration(self) -> float:
        return self.end - self.start


def frame_rms(frame: list[float], hop_rate: float = 0.0) -> tuple[float, int]:
    """RMS of one window plus a count of the non-finite samples found.

    The output is finite NO MATTER WHAT: non-finite inputs are zeroed (and
    counted), and a computed non-finite result (which would mean an internal
    defect) is clamped to 0.0. Loudness feeds CC7 for every later frame of
    the score; this function is the wall it must not leak through.
    """
    poisoned = 0
    acc = 0.0
    n = 0
    for x in frame:
        if math.isfinite(x):
            acc += x * x
            n += 1
        else:
            poisoned += 1
    if n == 0:
        return 0.0, poisoned
    rms = math.sqrt(acc / n)
    if not math.isfinite(rms):  # unreachable in theory; the guard is the point
        return 0.0, poisoned
    return rms, poisoned


def rms_series(samples: list[float], frame_len: int, hop: int,
               sample_rate: int) -> tuple[list[EnergyFrame], int]:
    """RMS on the same grid the pitch tracker uses (window centers in time)."""
    frames: list[EnergyFrame] = []
    poisoned_total = 0
    if len(samples) < frame_len:
        return frames, poisoned_total
    n_pos = 1 + (len(samples) - frame_len) // hop
    for i in range(n_pos):
        start = i * hop
        window = samples[start:start + frame_len]
        center = (start + len(window) / 2.0) / sample_rate
        rms, poisoned = frame_rms(window)
        poisoned_total += poisoned
        frames.append(EnergyFrame(index=i, time=center, rms=rms))
    return frames, poisoned_total


def segment(frames: list[EnergyFrame], hop_s: float,
            min_silence_s: float = MIN_SILENCE_S,
            min_speech_s: float = MIN_SPEECH_S) -> list[Segment]:
    """Speech/silence segments over the frame grid, with hysteresis.

    The speech-on threshold is 10% of the file's peak frame RMS (never below
    ABS_FLOOR). Peak, not percentile, because Phase 1 clips are single
    utterances: the loudest frame is the voice by construction; a longer
    recording with applause would need a percentile instead — stated limit.

    Frames at or above `on` enter speech; frames below `on/2` exit it; the
    band in between holds the current state. Runs are then filtered by the
    minimum durations: tiny silences are merged into the surrounding speech,
    tiny speech blips into silence. `hop_s` is the frame step in seconds,
    used to place the final segment's end after the last frame center.
    """
    if not frames:
        return []
    peak_rms = max(f.rms for f in frames)
    on = max(ABS_FLOOR, RELATIVE_ON * peak_rms)
    off = on / 2.0

    segs: list[Segment] = []
    state = frames[0].rms >= on
    seg_start = frames[0].time

    def close(end_time: float) -> None:
        nonlocal state, seg_start
        segs.append(Segment(start=seg_start, end=end_time, speech=state))

    for f in frames[1:]:
        if state:
            if f.rms < off:
                close(f.time)
                state = False
                seg_start = f.time
        elif f.rms >= on:
            close(f.time)
            state = True
            seg_start = f.time
    close(frames[-1].time + hop_s)

    return _merge_short_runs(segs, min_silence_s, min_speech_s)


def _merge_short_runs(segs: list[Segment], min_silence_s: float,
                      min_speech_s: float) -> list[Segment]:
    """Drop silences shorter than min_silence_s and speech shorter than
    min_speech_s by merging them into their neighbours, then re-join."""
    if not segs:
        return []
    kept = [segs[0]]
    for s in segs[1:]:
        prev = kept[-1]
        if s.speech and s.duration() < min_speech_s:
            continue  # blip disappears into surrounding silence
        if (not s.speech) and s.duration() < min_silence_s:
            prev.end = s.end  # gap disappears into surrounding speech
            continue
        kept.append(s)
    # re-join adjacent segments of the same state created by the merges
    joined = [kept[0]]
    for s in kept[1:]:
        if s.speech == joined[-1].speech:
            joined[-1].end = s.end
        else:
            joined.append(s)
    return joined
