"""
Tests for the gate's interval semantics on the watcher's decision path.

Two of the three defects this file was written against are real, and one --
the headline one -- turned out to be a misreading, kept visible here because
a measurement that corrects the record is worth as much as a fix:

1. GATE_FIELD "reads the retained buffer tail and understates by 14.5x"
   (docs/research/GATE-PEAK-COVERAGE.md). MEASURED PER RECORD: false. For
   every utterance record, GATE_FIELD equals the standalone gate's max over
   THAT RECORD's own [t0, t1] exactly -- worst ratio 1.00x on the doc's own
   200 Hz stack fixture at 4/8/16 s and on 90 s of real speech. The 14.5x
   came from dividing the whole-signal max by the LAST record's value: two
   different intervals. `_trim` drops probabilities only down to the last
   CLOSED segment's end, and the next segment starts exactly there, so the
   retained list always covers the record's segment. The guard test below
   pins that invariant so it cannot silently become false.

2. One transient gate frame clears a whole 2 s YIN window. REAL, measured to
   three decimals against WATCHER-COST.md: on the 120 Hz 12-harmonic hum
   fixture at peak 0.05, the gate passes exactly 1 frame in 937 (0.50388,
   index 2 -- a start-up transient), and because the decision statistic was
   "peak >= GATE_SPEECH", that single frame ran YIN on the first checkpoint
   of every run: gated hum cost 5.4% of a core against 2.6% for the pure
   tone where no transient occurs. The fix: the gate must say speech MORE
   THAN ONCE. GATE_MIN_FRAMES = 4 sits in the measured gap -- the weakest
   real-speech segment on the alice01 fixture holds 8 frames >= 0.5 at 0,
   -20 and -30 dB (45 segments each), the hum transient is 1 frame, and the
   150 Hz testvad stack the gate genuinely passes holds 187.

Everything deterministic here uses ScriptedGate so the suite needs no weights;
the fixture-level tests skip when the weights are absent, like test_vad.py.
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

from src import vad, watcher                                    # noqa: E402

SR = 16000
FIXTURE = ROOT / "examples" / "realaudio" / "alice01_excerpt.wav"

_weights = vad.model_path()
try:
    import onnxruntime                                           # noqa: F401
    _ORT = True
except ImportError:
    _ORT = False

needs_weights = pytest.mark.skipif(
    not (_ORT and _weights),
    reason="silero weights/onnxruntime absent; run src.vad.fetch_model()")


def tone(hz, seconds, amp=0.3):
    return [amp * math.sin(2 * math.pi * hz * i / SR)
            for i in range(int(seconds * SR))]


def session():
    """Three utterances, two pauses long enough to segment."""
    out = []
    for hz in (180.0, 240.0, 300.0):
        out += tone(hz, 0.40, 0.30)
        out += [0.0] * int(0.40 * SR)
    return out


def drain(w, samples, chunk=1600):
    events = []
    for i in range(0, len(samples), chunk):
        events.extend(w.push(samples[i:i + chunk]))
    return events


def mains_stack(f0, amp, seconds, n_harm=12):
    """hum_sweep.py's fixture: 1/k rolloff, normalised to peak == amp."""
    import numpy as np
    n = int(seconds * SR)
    t = np.arange(n, dtype=np.float64) / SR
    out = np.zeros(n, dtype=np.float64)
    for k in range(1, n_harm + 1):
        f = f0 * k
        if f >= SR / 2:
            break
        out += (amp / k) * np.sin(2.0 * math.pi * f * t)
    peak = float(np.abs(out).max()) or 1.0
    return (out * (amp / peak)).astype(np.float32).tolist()


# -- item 1: the invariant that was alleged broken and is not ---------------

@needs_weights
class TestGateFieldInterval:
    def test_gate_field_is_the_records_own_segment_peak(self):
        """
        For every utterance record, GATE_FIELD must equal the max of a
        standalone continuous gate over the RECORD's own [t0, t1].

        This is the invariant GATE-PEAK-COVERAGE.md alleged was broken with a
        14.5x understatement. Measured per record it holds exactly -- the
        14.5x divided the whole-signal max by the last record's value. If
        this test ever fails, probability retention HAS broken and the
        original allegation becomes true.
        """
        def check(samples, label):
            g = vad.SileroGate()
            probs = []
            for i in range(0, len(samples), 1600):
                probs.extend(g.push(samples[i:i + 1600]))

            def truth(t0, t1):
                i0 = max(0, int(t0 * SR) // vad.FRAME_SAMPLES)
                i1 = min(len(probs), -(-int(t1 * SR) // vad.FRAME_SAMPLES))
                return max(probs[i0:i1]) if i1 > i0 else None

            w = watcher.Watcher(SR, gate=vad.SileroGate())
            utts = [e for e in drain(w, samples) if e.kind == watcher.UTTERANCE]
            assert utts, f"{label}: nothing emitted"
            for u in utts:
                tr = truth(u.t0, u.t1)
                if tr is not None:
                    assert u.data[watcher.GATE_FIELD] == pytest.approx(
                        tr, abs=1e-6), (
                        f"{label}: record [{u.t0:.2f},{u.t1:.2f}]s reports "
                        f"{u.data[watcher.GATE_FIELD]} but its own segment "
                        f"peak is {tr}")

        check(mains_stack(200.0, 0.05, 4.0), "200Hz stack 4s")
        if FIXTURE.is_file():
            from src import wavio
            check(wavio.read_wav(str(FIXTURE)).samples, "real speech")

    def test_retained_probabilities_never_lag_the_open_segment(self):
        """
        The mechanism behind the invariant: `_trim` may drop probability
        frames no further back than the last CLOSED segment's end. If it ever
        drops past that, a closing segment's early frames are gone and
        GATE_FIELD silently becomes a tail statistic -- the exact defect
        GATE-PEAK-COVERAGE.md described, arriving for real this time.
        """
        g = vad.ScriptedGate([0.9])
        w = watcher.Watcher(SR, gate=g)
        stream = (tone(200.0, 0.3) + [0.0] * int(0.3 * SR)) * 20   # ~12 s
        emitted_until = 0.0
        for i in range(0, len(stream), 1600):
            w.push(stream[i:i + 1600])
            emitted_until = max(emitted_until, w._emitted_until)
            keep_from = int(emitted_until * SR)
            assert w._gate_probs_base <= keep_from + vad.FRAME_SAMPLES, (
                f"probability origin {w._gate_probs_base} dropped past the "
                f"emitted boundary {keep_from}; segment peaks are being lost")


# -- item 2: one transient frame must not clear a whole YIN window -----------

class TestGateMinFrames:
    def test_one_passing_frame_does_not_clear_yin(self):
        """
        THE defect, deterministically: a gate that says speech exactly once
        (a start-up transient) and never again. The old rule -- peak >=
        GATE_SPEECH -- ran YIN on that one frame's say-so for the entire 2 s
        checkpoint window. The measured instance: 1 frame in 937 at 0.50388.

        The gate must say it more than once: fewer than GATE_MIN_FRAMES
        passing frames means the segment is treated as not-speech for the
        purposes of spending CPU.
        """
        g = vad.ScriptedGate([0.6] + [0.01] * 4000)
        w = watcher.Watcher(SR, gate=g)
        utts = [e for e in drain(w, tone(200.0, 4.0, 0.3))
                if e.kind == watcher.UTTERANCE]
        assert utts, "fixture too short to close a segment"
        bad = [u for u in utts if not u.data["pitch_skipped"]]
        assert not bad, (
            f"{len(bad)} of {len(utts)} checkpoints ran YIN on a single "
            f"passing frame (gate_speech_prob "
            f"{bad[0].data[watcher.GATE_FIELD] if bad else '-'})")

    def test_steady_speech_frames_still_clear_yin(self):
        """The other direction: a gate that says speech constantly must not
        start skipping. This is the guard against over-tightening."""
        g = vad.ScriptedGate([0.9])
        w = watcher.Watcher(SR, gate=g)
        utts = [e for e in drain(w, session()) if e.kind == watcher.UTTERANCE]
        assert utts
        assert all(u.data["pitch_skipped"] is False for u in utts), \
            "segments the gate steadily accepts are being skipped"

    def test_the_decision_count_is_on_the_record(self):
        """
        With a count rule, `gate_speech_prob` alone can look contradictory
        (0.6 peak, skipped). The count must travel with the record or the
        audit trail GATE_FIELD exists for is broken in a new way.
        """
        g = vad.ScriptedGate([0.6] + [0.01] * 4000)
        w = watcher.Watcher(SR, gate=g)
        utts = [e for e in drain(w, tone(200.0, 4.0, 0.3))
                if e.kind == watcher.UTTERANCE]
        assert utts
        for u in utts:
            assert "gate_speech_frames" in u.data
        assert utts[0].data["gate_speech_frames"] == 1

    @needs_weights
    def test_the_measured_hum_transient_is_skipped(self):
        """
        The fixture from WATCHER-COST.md: 120 Hz, 12 harmonics, peak 0.05.
        The real gate passes exactly 1 frame in 937 (0.50388, index 2). The
        record must still carry that peak -- the audit trail -- while YIN is
        skipped, and gated cost must approach the pure-tone case rather than
        paying one YIN pass per run.
        """
        w = watcher.Watcher(SR, gate=vad.SileroGate())
        utts = [e for e in drain(w, mains_stack(120.0, 0.05, 8.0))
                if e.kind == watcher.UTTERANCE]
        assert utts, "fixture too short to close a segment"
        ran = [u for u in utts if not u.data["pitch_skipped"]]
        assert not ran, (
            f"{len(ran)} of {len(utts)} checkpoints ran YIN on the hum "
            f"fixture; the first record's count is "
            f"{utts[0].data.get('gate_speech_frames')}")
        # the audit trail still tells the truth about the transient
        assert utts[0].data[watcher.GATE_FIELD] >= 0.5, \
            "the transient peak disappeared from the record"

    @needs_weights
    def test_real_speech_still_runs_yin_everywhere(self):
        """No real utterance may be newly skipped by the count rule."""
        if not FIXTURE.is_file():
            pytest.skip("real speech fixture absent")
        from src import wavio
        speech = wavio.read_wav(str(FIXTURE)).samples[:30 * SR]
        w = watcher.Watcher(SR, gate=vad.SileroGate())
        utts = [e for e in drain(w, speech) if e.kind == watcher.UTTERANCE]
        assert utts
        skipped = [u for u in utts if u.data["pitch_skipped"]]
        assert not skipped, \
            f"{len(skipped)} of {len(utts)} real utterances were skipped"
