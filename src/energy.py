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

THE DEFAULT BAR IS A FRACTION OF THE LOUDEST FRAME ANYWHERE, which is a
global statistic answering a local question, and it fails in a way no
individual segmentation reveals:

    quiet speech alone (2.0 s, RMS 0.035)   speech found 4.00 s
    the same speech after one door slam       speech found 0.34 s

One door, slammed once, and the system never heard the person again.

`adaptive_threshold` replaces it with a local moving quantile, and the
quantile had to be chosen by measurement because the obvious answer was
wrong: a HIGH quantile sounds more principled and costs 94% of real speech,
because a voice is loud most of the time it exists and so a high quantile of
a window containing speech is a quantile OF THE SPEECH. See that function's
docstring for the grid, and `tests/test_adaptive_threshold.py` for the tests
that hold both directions at once.
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


LOCAL_WINDOW_S = 1.5         # half-width of the adaptive window, each side
LOCAL_QUANTILE = 0.20        # measured; see adaptive_threshold()
LOCAL_MULTIPLE = 2.5         # bar = floor x this, above ABS_FLOOR


def adaptive_threshold(values: list[float], window_s: float = LOCAL_WINDOW_S,
                       hop_s: float = 0.01, quantile: float = LOCAL_QUANTILE,
                       multiple: float = LOCAL_MULTIPLE) -> list[float]:
    """
    A speech threshold that follows the LOCAL noise floor.

    WHY, MEASURED. The threshold above is a fraction of the loudest frame
    anywhere in the material, which is a global statistic applied to a local
    question. Constructed and measured:

        quiet speech alone (2.0 s, RMS 0.035)   speech found 4.00 s
        the same speech after one door slam       speech found 0.34 s

    One door, slammed once, and the system never heard the person again. The
    slam raised the peak from 0.034 to 0.707, the bar rose with it, and every
    subsequent frame sat below it. Nothing in the code was wrong and every
    individual segmentation was self-consistent; the defect was only visible
    as an outcome.

    This is the fix: take the threshold from a moving quantile of the frames
    AROUND each one, so the bar rises with a transient and falls again when
    the transient passes.

    THE HYPOTHESIS I HAD WAS WRONG, AND MEASURING IT IS THE POINT.

    I reasoned first that the quantile should sit HIGH -- at the median, or
    above it -- so that the floor estimate would be drawn from the room rather
    than from the voice. I wrote that down, then measured it, and it was
    backwards.

        speech found in the 90 s fixture, adaptive bar at window 1.5 s
            global peak threshold      59.25 s
            q = 0.50 (median)          30.87 s
            q = 0.75                    3.62 s
            q = 0.90                    0.19 s

    A voice is LOUD most of the time it exists, so any high quantile of a
    window that contains speech is a quantile OF THE SPEECH, and the bar lands
    above the signal it is supposed to detect. At q = 0.75 the bar sat at
    2.5x the signal RMS and the adaptive threshold threw away 94% of real
    speech while fixing the door slam by 0.00 s. It was strictly worse than
    the thing it replaced, on the metric that matters.

    The reason a LOW quantile is right is that a noise floor is, by
    definition, low: the quietest fifth of a local window is room, and the
    voice is not. Measured across a grid of quantile x multiple, with two
    targets (find speech, survive a slam) and three guards (noise, hum,
    silence must all stay at zero):

        q     mult | speech(90s)  slam+quiet | noise .05  hum .2  silence
        0.15   2.5  |     67.97       2.34   |     0.00    0.00    0.00
        0.20   2.5  |     65.08       2.34   |     0.00    0.00    0.00
        0.25   2.5  |     61.13       2.03   |     0.00    0.00    0.00
        0.30   2.5  |     56.00       1.99   |     0.00    0.00    0.00
        global     |     59.25       0.34   |     2.96    2.96    0.00

    q = 0.20 wins on every axis at once: it finds MORE real speech than the
    global bar (65.08 vs 59.25 s -- the global bar was also under-reporting,
    which no test had caught), it recovers the slam case (2.34 vs 0.34 s of a
    2.0 s utterance), and it holds all three noise guards at zero where the
    global bar put 2.96 s of noise and 2.96 s of hum into the record as
    speech.

    THE WINDOW IS 1.5 s HALF-WIDTH, MEASURED TOO. Shorter windows cannot hold
    enough of a syllable run to estimate against: at 0.1 s the same q = 0.5
    finds 2.13 s of the fixture instead of 30.87 s.

    THE LIMIT, STATED RATHER THAN HIDDEN. This is a noise-floor estimator, not
    a VAD, and in a room that is loud AND continuous -- a fan, an engine -- the
    floor rises with the noise and so does the bar. Measured: on 3 s of 120 Hz
    hum at amplitude 0.2 it correctly reports 0.00 s, but only because the
    window is long enough to contain a settled floor; a hum whose level drifts
    faster than the window does not get that protection. That is the case
    `vad.py` exists for -- this function makes a TRANSIENT survivable, and a
    model makes SUSTAINED noise survivable. Neither handles both alone.

    COST. O(n * w log w) because each window is sorted. Measured 21.7 ms for
    90 s of audio, which is 0.024 of a core -- affordable beside a 2.4%-of-a-
    core watcher, and it is O(n) in memory because only the output bars are
    retained.
    """
    n = len(values)
    if n == 0:
        return []
    win = max(1, int(round(window_s / hop_s)))
    out: list[float] = []
    for i in range(n):
        lo = max(0, i - win)
        hi = min(n, i + win + 1)
        window = sorted(values[lo:hi])
        # Nearest-rank quantile: index ceil(q*n)-1, clamped. No interpolation,
        # so the returned bar is always a value that actually occurred.
        k = min(len(window) - 1, max(0, int(math.ceil(quantile * len(window))) - 1))
        out.append(max(ABS_FLOOR, multiple * window[k]))
    return out


def segment(frames: list[EnergyFrame], hop_s: float,
            min_silence_s: float = MIN_SILENCE_S,
            min_speech_s: float = MIN_SPEECH_S,
            peak_rms: float | None = None,
            adaptive: bool = False,
            quantile: float = LOCAL_QUANTILE,
            multiple: float = LOCAL_MULTIPLE) -> list[Segment]:
    """Speech/silence segments over the frame grid, with hysteresis.

    The speech-on threshold is 10% of the peak frame RMS (never below
    ABS_FLOOR). Peak, not percentile, because Phase 1 clips are single
    utterances: the loudest frame is the voice by construction; a longer
    recording with applause would need a percentile instead -- stated limit.

    `peak_rms` OVERRIDES that peak, for callers analysing a sliding window.
    The watcher (src/watcher.py) needs exactly that: it re-segments a moving
    buffer on every push, so a per-call peak would make the speech threshold
    drift with the window contents. A quiet stretch of buffer would lower the
    bar and turn the silence around it into "speech", and the drift would be
    invisible because every individual segmentation looked self-consistent.
    The watcher therefore keeps a running peak and passes it in, which is the
    reference-percentile idea above, implemented as the cheapest version that
    works. Default None keeps the original single-clip behaviour exactly.

    `adaptive=True` replaces the global bar with `adaptive_threshold`, which
    is the measured fix for a transient loud enough to silence the rest of the
    session. It is OFF by default so every existing caller keeps the exact
    behaviour it was written against, and the watcher opts in explicitly.
    """
    if not frames:
        return []
    if adaptive:
        on_local = adaptive_threshold([f.rms for f in frames], hop_s=hop_s,
                                      quantile=quantile, multiple=multiple)
    if peak_rms is None:
        peak_rms = max(f.rms for f in frames)
    on = max(ABS_FLOOR, RELATIVE_ON * peak_rms)
    off = on / 2.0

    segs: list[Segment] = []
    state = frames[0].rms >= (on_local[0] if adaptive else on)
    seg_start = frames[0].time

    def close(end_time: float) -> None:
        nonlocal state, seg_start
        segs.append(Segment(start=seg_start, end=end_time, speech=state))

    # The exit bar is HALF the entry bar, as above, so a decaying syllable tail
    # cannot chatter the boundary. It tracks the local entry bar rather than
    # the global one for the same reason the entry bar does: a half of a
    # global bar is still a global bar.
    for idx, f in enumerate(frames[1:], start=1):
        if adaptive:
            on = on_local[idx]
            off = on / 2.0
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
