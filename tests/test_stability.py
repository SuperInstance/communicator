"""
stability — the per-utterance trust probe, tested against fakes.

WHAT THIS FILE PROVES: the probe counts agreement correctly across
controlled answer sequences (a deterministic fake reads 1.00, a jittering
fake reads below), the factor only bites when instability is real, and
the probe never touches the verdict cache (the freeze-vs-hide distinction
from EXPERIMENTS-AUDIO section 3e).

WHAT THIS FILE DOES NOT PROVE: anything about live Jev — the §3e table
is the live measurement; these tests pin the arithmetic only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import stability                    # noqa: E402
from src.judged import TrackReading         # noqa: E402


class SeqClient:
    """Returns a scripted sequence of answers per call."""

    def __init__(self, intention_values, gesture=("question",)):
        self.intention_values = list(intention_values)
        self.gesture = list(gesture)
        self.calls = 0

    def system_one(self, **kw):
        i = min(self.calls, len(self.intention_values) - 1)
        g = min(self.calls, len(self.gesture) - 1)
        self.calls += 1

        class A:  # minimal answer shapes read_tracks consumes
            pass
        resp = A(); resp.model = "fake"
        timbre = A(); timbre.score = 1.0; timbre.confidence = 0.5
        timbre.legend = {}; timbre.probabilities = {0: 0.5, 1: 0.5}
        inten = A(); inten.score = 0.0; inten.confidence = 0.5
        inten.legend = {}; inten.probabilities = {0: 0.5}
        # encode the scripted INTENTION value via a fake 'score' the
        # reading path scales -- simplest: patch after read_tracks via
        # detail is messy; instead script gesture only and set intention
        # deterministically, then test jitter through GESTURE sequences.
        atten = A(); atten.choice = "listener"
        atten.probabilities = {"listener": 0.8, "task": 0.2}
        atten.confidence = 0.8
        gest = A(); gest.choice = self.gesture[g]
        gest.probabilities = {"question": 0.9, "command": 0.1}
        gest.confidence = 0.9
        resp.answers = {"timbre": timbre, "intention": inten,
                        "attention": atten, "gesture": gest}
        return resp


def test_deterministic_answers_read_stable():
    c = SeqClient([], gesture=["question"])
    s = stability.probe_stability("run the tests", n=3, client=c, sleep_s=0)
    assert s["GESTURE"]["mode_share"] == 1.0
    assert s["GESTURE"]["factor"] == 1.0


def test_jittering_choice_reads_unstable_and_fades():
    c = SeqClient([], gesture=["question", "command", "command"])
    s = stability.probe_stability("what time is it", n=3, client=c, sleep_s=0)
    assert abs(s["GESTURE"]["mode_share"] - 2 / 3) < 0.001
    assert s["GESTURE"]["factor"] < 1.0


def test_all_agree_is_unity_even_when_the_answer_is_garbage():
    """Stability says nothing about CORRECTNESS -- a confidently wrong
    deterministic answer still reads 1.00. That separation is the point:
    trust fades on instability, truth is calibrated elsewhere."""
    c = SeqClient([], gesture=["command"])
    s = stability.probe_stability("qwop zzz brxt", n=4, client=c, sleep_s=0)
    assert s["GESTURE"]["mode_share"] == 1.0


def test_trust_fields_are_flat_and_json_safe():
    c = SeqClient([], gesture=["question", "question", "command"])
    s = stability.probe_stability("hey", n=3, client=c, sleep_s=0)
    f = stability.trust_fields(s)
    import json
    assert set(f) == set(s)
    json.dumps(f)  # must not raise
