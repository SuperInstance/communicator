"""
Octave smoothing — the measured fix for the Day-0 3% octave-jump rate.

Day-0 (EXPERIMENTS-AUDIO.md §3): 3.0% of adjacent voiced pairs (134/4477
on 90 s of LibriVox Alice) sit a full octave apart, split near-evenly
up/down. A real voice glides, dips and creaks, but it does not double or
halve inside 10 ms. `pitch.smooth_octaves` is the pYIN-lite response: a
deterministic median post-pass, not a Viterbi, deliberately additive.

THE CLAIM UNDER TEST is deliberately narrow: a frame an octave away from
a stable median supported by voiced neighbours on BOTH sides is an
error, rescaled to its octave-consistent value; everything else passes
through untouched.

WHAT THIS FILE DOES NOT PROVE: nothing about real speech (fixtures are
synthetic; the Alice before/after lives in the experiment log, not CI);
nothing about MAJORITY octave errors (a whole run uniformly doubled IS
the median — pYIN's candidate prior is the real fix, out of scope);
nothing about bursts near the window width.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pitch import PitchFrame, PitchTracker, smooth_octaves   # noqa: E402

SR = 16000


def frames_of(hz_values, start=0):
    return [PitchFrame(index=start + i, time=0.005 + (start + i) * 0.010,
                       hz=hz, confidence=0.9)
            for i, hz in enumerate(hz_values)]


def unvoiced(n, start=0):
    return [PitchFrame(index=start + i, time=0.005 + (start + i) * 0.010,
                       hz=None, confidence=0.0) for i in range(n)]


def test_single_frame_octave_spikes_are_corrected_both_directions():
    vals = [200.0] * 30
    vals[10] = 400.0
    vals[20] = 100.0
    out = smooth_octaves(frames_of(vals))
    assert abs(out[10].hz - 200.0) <= 0.5, out[10].hz
    assert abs(out[20].hz - 200.0) <= 0.5, out[20].hz
    for i, f in enumerate(out):
        if i not in (10, 20):
            assert f.hz == 200.0, (i, f.hz)
        assert f.confidence == 0.9 and f.index == i


def test_genuine_octave_melody_step_is_not_destroyed():
    vals = [100.0] * 40 + [400.0] * 40
    out = smooth_octaves(frames_of(vals))
    assert [f.hz for f in out] == vals, "the melody step was touched"
    assert abs(math.log2(out[40].hz / out[39].hz) - 2.0) < 1e-9


def test_semitone_movement_is_never_corrected():
    vals = [200.0] * 12 + [225.0] * 8
    out = smooth_octaves(frames_of(vals))
    assert [f.hz for f in out] == vals


def test_isolated_voiced_frame_is_untouched():
    frames = unvoiced(5) + frames_of([217.3], start=5) + unvoiced(5, start=6)
    out = smooth_octaves(frames)
    assert out == frames


def test_run_shorter_than_hold_is_untouched():
    frames = (unvoiced(3) + frames_of([200.0, 400.0, 200.0], start=3)
              + unvoiced(3, start=6))
    out = smooth_octaves(frames, hold=5)
    assert [f.hz for f in out] == [None] * 3 + [200.0, 400.0, 200.0] + [None] * 3


def test_unvoiced_gaps_preserved_and_never_bridged():
    frames = (frames_of([200.0] * 20) + unvoiced(4, start=20)
              + frames_of([400.0] * 20, start=24))
    out = smooth_octaves(frames)
    assert [f.hz for f in out] == [f.hz for f in frames]
    assert all(f.hz is None for f in out[20:24]), "the gap was altered"


def test_third_period_error_is_not_snapped_two_octaves():
    vals = [220.0] * 30
    vals[15] = 220.0 / 3.0
    out = smooth_octaves(frames_of(vals))
    assert out[15].hz == 220.0 / 3.0, "f/3 was snapped; the fix worsened it"


def test_correction_refused_outside_the_trackers_own_range():
    vals = [380.0] * 30
    vals[15] = 900.0
    out = smooth_octaves(frames_of(vals))
    assert out[15].hz == 900.0


def test_input_is_not_mutated():
    vals = [200.0] * 30
    vals[10] = 400.0
    frames = frames_of(vals)
    snapshot = list(frames)
    smooth_octaves(frames)
    assert frames == snapshot


def test_clean_tracker_output_passes_through_unchanged():
    samples = [0.5 * math.sin(2 * math.pi * 200 * i / SR)
               for i in range(int(0.5 * SR))]
    frames = PitchTracker(SR).track(samples)
    assert any(f.hz is not None for f in frames), "fixture tracked nothing"
    assert smooth_octaves(frames) == frames
