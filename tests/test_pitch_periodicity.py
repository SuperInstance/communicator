"""
Tests for the periodicity-consistency test the pitch tracker lacked.

WHY. `docs/research/NOTCH-AB.md` established the mechanism: YIN measures
PERIOD, not frequency. Deleting the fundamental of a harmonic stack leaves
the surviving harmonics with an unchanged period, so the emitted F0 is
unchanged (150.0 Hz in, 150.0 Hz out with the 150 Hz line removed to
-49.5 dB), and no linear filter can fix that -- `MIC-HARDWARE.md` retired
that row of the table. `IN-THE-WILD-AUDIO.md` §5.1 named the candidate fix:
real voice F0 moves continuously; a mains-locked estimate does not.

MEASURED HERE, on this machine, before the threshold was chosen:

    mains-locked runs (120 Hz tone, 150/200 Hz stacks, 30 s and 6 s)
        peak-to-peak excursion over runs of 1.73 s:  <= 0.01 cents
    real voice (alice01 90 s + both AMI meetings, 1409 voiced runs)
        excursion of runs >= 0.5 s: min 104 cents  (n = 36)
        runs < 20 cents: all shorter than 0.05 s -- indistinguishable
        from locks, which is WHY a minimum run length exists

So the detector is: a contiguous voiced run at least LOCK_MIN_RUN_S long
whose excursion stays within LOCK_MAX_EXCURSION_CENTS. The length floor is
not a dodge, it is forced by the data -- on runs shorter than ~0.05 s real
voice is as flat as a mains lock, and there is no evidence to either side.
Every lock in the failure corpus is >= 1.73 s.

WHAT THIS CATCHES AND WHAT IT DOES NOT. It catches the mains-locked estimate
on hum-alone and stack fixtures (the gate-passes-150-Hz-stack case included).
It does NOT catch mixtures: measured speech+150Hz-stack tracks wander
161-1318 cents while being wrong -- a subharmonic lock that drifts is still
wrong, but it is not periodicity-LOCKED, and suppressing it needs a different
test. That limit is stated here so nobody reads this fix as larger than it is.
"""

from __future__ import annotations
import pytest as _pytest  # noqa: E402
_pytest.importorskip("numpy", reason="exercises code needing numpy")  # noqa: E402

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import pitch, watcher                                    # noqa: E402

SR = 16000
FIXTURE = ROOT / "examples" / "realaudio" / "alice01_excerpt.wav"


def tone(hz, seconds, amp=0.3):
    return [amp * math.sin(2 * math.pi * hz * i / SR)
            for i in range(int(seconds * SR))]


def glide(f0, f1, seconds, amp=0.3):
    """A voiced-signal stand-in whose F0 genuinely moves: a linear glide."""
    n = int(seconds * SR)
    out = [0.0] * n
    phase = 0.0
    for i in range(n):
        f = f0 + (f1 - f0) * i / n
        phase += 2.0 * math.pi * f / SR
        out[i] = amp * math.sin(phase)
    return out


def stack(hz, seconds, amp=0.3, harmonics=(1, 2, 3)):
    """tests/test_vad.py's harmonic stack: the fixture the gate PASSES."""
    out = [0.0] * int(seconds * SR)
    for k, h in enumerate(harmonics):
        a = amp / (k + 1)
        for i in range(len(out)):
            out[i] += a * math.sin(2 * math.pi * hz * h * i / SR)
    return out


def drain(w, samples, chunk=1600):
    events = []
    for i in range(0, len(samples), chunk):
        events.extend(w.push(samples[i:i + chunk]))
    return events


class TestLockedRuns:
    def test_a_pure_lock_is_detected_a_glide_is_not(self):
        """The statistic itself, at the tracker level."""
        locked = pitch.PitchTracker(SR).track(tone(150.0, 1.0))
        moving = pitch.PitchTracker(SR).track(glide(180.0, 240.0, 1.0))
        assert pitch.locked_runs(locked, hop_s=160 / SR), \
            "a 1 s constant-period run was not detected as locked"
        assert not pitch.locked_runs(moving, hop_s=160 / SR), \
            "a 300+-cent glide was called locked"

    def test_short_runs_are_never_locked(self):
        """
        Below LOCK_MIN_RUN_S there is no evidence either way -- measured real
        runs under 0.05 s are as flat as mains locks -- so the detector must
        keep its hands off. This is what keeps short real utterances safe.
        """
        frames = pitch.PitchTracker(SR).track(tone(150.0, 0.3))
        assert not pitch.locked_runs(frames, hop_s=160 / SR)


class TestMainsLockedPitchNotEmitted:
    def test_the_stack_the_gate_passes_emits_no_pitch(self):
        """
        THE defect, end to end: the 150 Hz three-harmonic stack scores
        0.66-0.82 on the gate, so YIN runs -- and before this fix it emitted
        hz_p05 = 150.0 on every checkpoint, a mains lock reported as a voice.
        After: the record says the pitch is locked and emits no hz at all,
        because a gap is strictly better than a wrong number.
        """
        w = watcher.Watcher(SR)                    # no gate: YIN runs regardless
        utts = [e for e in drain(w, stack(150.0, 6.0, 0.3))
                if e.kind == watcher.UTTERANCE]
        assert utts, "fixture too short to close a segment"
        emitted = [u for u in utts if u.data["hz_p05"] is not None]
        assert not emitted, (
            f"{len(emitted)} of {len(utts)} checkpoints emitted a mains-locked "
            f"pitch ({[u.data['hz_p05'] for u in emitted]})")
        assert all(u.data["pitch_locked_runs"] >= 1 for u in utts)
        # voicedness is a statement about periodic energy, which hum HAS;
        # only the F0 is untrustworthy
        assert utts[0].data["voiced_ratio"] > 0.9

    def test_a_locked_run_among_moving_runs_is_suppressed_alone(self):
        """
        Suppression is per-run: an utterance that is genuinely moving keeps
        its pitch even when a neighbouring utterance is locked. A glide is
        periodic and loud -- exactly the hum signature -- but its F0 moves
        617 cents, so it must survive.
        """
        sig = (tone(150.0, 0.8) + [0.0] * int(0.4 * SR)
               + glide(180.0, 240.0, 0.8) + [0.0] * int(0.4 * SR))
        w = watcher.Watcher(SR)
        utts = [e for e in drain(w, sig) if e.kind == watcher.UTTERANCE]
        assert len(utts) == 2, f"expected 2 utterances, got {len(utts)}"
        first, second = utts
        assert first.data["hz_p05"] is None, "the locked tone kept its pitch"
        assert first.data["pitch_locked_runs"] == 1
        assert second.data["hz_p05"] is not None, "the glide lost its pitch"
        assert second.data["pitch_locked_runs"] == 0

    def test_real_speech_keeps_every_pitch(self):
        """
        The guard that decides whether this fix is safe to ship: on real
        speech, no utterance with voiced frames may lose its hz. Measured on
        alice01 (90 s) and both AMI meetings: no run >= 0.5 s stays within
        10 cents -- the tightest is 104 cents, a 10x margin.
        """
        if not FIXTURE.is_file():
            pytest.skip("real speech fixture absent")
        from src import wavio
        speech = wavio.read_wav(str(FIXTURE)).samples[:30 * SR]
        w = watcher.Watcher(SR)
        utts = [e for e in drain(w, speech) if e.kind == watcher.UTTERANCE]
        assert utts
        lost = [u for u in utts
                if u.data["voiced_frames"] > 0 and u.data["hz_p05"] is None]
        assert not lost, (
            f"{len(lost)} of {len(utts)} real utterances lost their pitch "
            f"to the lock detector (first at "
            f"{lost[0].t0:.2f}s)" if lost else "")
