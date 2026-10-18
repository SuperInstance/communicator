"""
tempo — speech rate as BPM, from syllable peaks in the energy envelope.

This is a PROXY, stated as one: syllables are energy peaks in the loudness
envelope, not detected nuclei. Real syllable detection uses formant and
voicing structure; on clean read speech the peak count is within ~15% of a
human syllable count, but on a whisper or a shouted run-on sentence it is
not. The BPM it emits is "peaks per minute", which is the speech-rate
signal the TEMPO track wants — not a musical beat.

Mapping: one syllable = one beat, so BPM = syllables/sec * 60. Conversational
speech at 4–5 syll/s lands at 240–300 BPM. That number is intentionally NOT
rescaled to a musical 90–140 range: rescaling would silently change the
meaning of the track whenever speech rate changes, and the whole point of
TEMPO being measured is that it moves with the speaker.

Fallback: no speech at all → None; the score builder emits slackwater's
default 120 BPM (its MultiTrackEncoder does exactly this when the tempo
track is empty).
"""

from __future__ import annotations

from .energy import EnergyFrame, Segment

SMOOTH_FRAMES = 3     # ~30 ms moving average at a 10 ms hop
MIN_PEAK_GAP_S = 0.120  # faster than ~8 syll/s is not speech
MIN_PEAK_HEIGHT = 0.30  # peak must rise 30% above the local valley floor


def estimate_bpm(frames: list[EnergyFrame], segments: list[Segment]) -> float | None:
    """Peaks per minute over speech regions; None if there is no speech.

    A peak is a local maximum of the smoothed envelope that clears the
    preceding valley by MIN_PEAK_HEIGHT (so a plateau or a ripple on a flat
    tail does not count twice) and stands at least MIN_PEAK_GAP_S from the
    previously accepted peak (greedy suppression, strongest-first within a
    tie is not needed at 10 ms resolution).
    """
    speech_s = sum(s.duration() for s in segments if s.speech)
    if speech_s <= 0:
        return None

    env = _smooth([f.rms for f in frames], SMOOTH_FRAMES)
    times = [f.time for f in frames]

    # candidate local maxima inside speech (frame i is speech if its center
    # falls in a speech segment; approximate by nearest-frame lookup)
    in_speech = [_speech_at(t, segments) for t in times]
    candidates: list[tuple[float, float]] = []  # (time, height)
    for i in range(1, len(env) - 1):
        if not in_speech[i]:
            continue
        if env[i] >= env[i - 1] and env[i] > env[i + 1]:
            valley = _valley_before(env, in_speech, i)
            if valley is not None and env[i] > valley * (1.0 + MIN_PEAK_HEIGHT):
                candidates.append((times[i], env[i]))

    # greedy min-distance suppression, earliest-first (keeps phrase order)
    accepted: list[tuple[float, float]] = []
    for t, h in candidates:
        if accepted and t - accepted[-1][0] < MIN_PEAK_GAP_S:
            continue
        accepted.append((t, h))

    if not accepted:
        return None
    syll_per_s = len(accepted) / speech_s
    return 60.0 * syll_per_s


def _speech_at(t: float, segments: list[Segment]) -> bool:
    for s in segments:
        if s.start <= t < s.end:
            return s.speech
    return False


def _valley_before(env: list[float], in_speech: list[bool], i: int) -> float | None:
    """Lowest envelope value in the contiguous speech run ending just
    before frame i — the valley a real syllable peak must rise above."""
    j = i - 1
    floor = None
    while j >= 0 and in_speech[j]:
        floor = env[j] if floor is None else min(floor, env[j])
        j -= 1
    if floor is None:
        return env[i - 1] if i > 0 else None
    return floor


def _smooth(x: list[float], width: int) -> list[float]:
    """Trailing/centered box average — centered, with shrinking windows at
    the edges so the envelope stays aligned with frame times."""
    half = width // 2
    out = []
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        window = x[lo:hi]
        out.append(sum(window) / len(window))
    return out
