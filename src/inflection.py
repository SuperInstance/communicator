"""
inflection — direction of pitch movement, for the pitch-bend track.

slackwater-perception classifies each frame against its predecessor with a
±15% ratio threshold (encoder.py `_classify_inflection`, threshold=0.15) and
maps RISING/FALLING/FLAT to pitch bend +4095/-4095/0. That SEMANTIC is
adopted exactly; the window is not, because slackwater's frames arrive at
musical cadence (~one per beat, 0.5 s at 120 BPM) while this pipeline's
frames arrive every 10 ms. A 15% move in 10 ms is a 240-semitone-per-second
glissando — nothing human — so on this grid every utterance would be FLAT
and the track would carry no information.

The comparable window at this frame rate is 100 ms (10 hops): 15% ≈ 2.4
semitones over a tenth of a second is a real intonational move (question
rise, terminal fall). Comparisons run over VOICED frames only — across an
unvoiced gap there is no movement, only absence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .pitch import PitchFrame

RATIO_THRESHOLD = 0.15   # slackwater's default, kept
SPAN_FRAMES = 10         # 100 ms at a 10 ms hop
BEND_RISING = 4095
BEND_FALLING = -4095
BEND_FLAT = 0


class Direction(Enum):
    RISING = "rising"
    FALLING = "falling"
    FLAT = "flat"
    UNDEFINED = "undefined"


@dataclass
class InflectionEvent:
    time: float        # seconds, at the frame the direction was decided
    direction: Direction
    bend: int          # +4095 / -4095 / 0, slackwater's mapping


def classify(current_hz: float, previous_hz: float,
             threshold: float = RATIO_THRESHOLD) -> Direction:
    """slackwater's ratio test, verbatim semantics."""
    if previous_hz <= 0 or current_hz <= 0:
        return Direction.UNDEFINED
    ratio = current_hz / previous_hz
    if ratio > 1.0 + threshold:
        return Direction.RISING
    if ratio < 1.0 - threshold:
        return Direction.FALLING
    return Direction.FLAT


def contour_events(frames: list[PitchFrame],
                   span: int = SPAN_FRAMES) -> list[InflectionEvent]:
    """One event per voiced frame that has a voiced comparison `span` frames
    back (or the nearest voiced frame within the span). Consecutive frames
    with the same direction are merged into one event at the run's start:
    slackwater emits per frame, and merging only changes density, not the
    direction-to-bend mapping that downstream consumers rely on.
    """
    events: list[InflectionEvent] = []
    voiced_idx = [f.index for f in frames if f.hz is not None]

    def hz_at(index: int) -> float | None:
        return frames[index].hz if 0 <= index < len(frames) else None

    # p walks voiced_idx: for the pos-th voiced frame it holds the NEAREST
    # earlier voiced frame at least `span` frame-indices back. Because both
    # lists are sorted, p only ever moves forward — O(V) total.
    p = 0
    last_direction: Direction | None = None
    for pos, i in enumerate(voiced_idx):
        while p + 1 < pos and i - voiced_idx[p + 1] >= span:
            p += 1
        if i - voiced_idx[p] < span:
            continue
        cur, prev = hz_at(i), hz_at(voiced_idx[p])
        if cur is None or prev is None:
            continue
        if not (math.isfinite(cur) and math.isfinite(prev)):
            continue
        direction = classify(cur, prev)
        if direction == last_direction:
            continue
        events.append(InflectionEvent(
            time=frames[i].time,
            direction=direction,
            bend={Direction.RISING: BEND_RISING,
                  Direction.FALLING: BEND_FALLING,
                  Direction.FLAT: BEND_FLAT,
                  Direction.UNDEFINED: BEND_FLAT}[direction],
        ))
        last_direction = direction
    return events
