"""
verdict_cache — a controller that jitters for no reason is unplayable.

jev-lattice measured Jev re-asked against identical evidence flipping 7% of
gates at threshold 0.9. The same defect in a synthesiser control surface is
audible: a CC that moves between two readings of the SAME utterance is noise
the operator will tune against. These tests pin the fix.

WHAT THIS FILE PROVES:
  * The second read of the same utterance is byte-identical to the first and
    costs zero model calls.
  * Different utterances, different models, and different question WORDING
    are different cache entries — invalidation is by construction.
  * Failed readings are never stored, so a network blip cannot become a
    sticky verdict.
  * Persistence round-trips through a JSON file.

WHAT THIS FILE DOES NOT PROVE:
  * Nothing about the LIVE flip rate on real Jev — the offline jitter here is
    injected on purpose, at full amplitude, to make the failure visible. The
    real 7%-scale subtlety needs the live double-ask experiment in
    PROGRESS.md, not a unit test.
  * Nothing about cache eviction. At voice-loop scale (thousands of
    utterances, small JSON records) there is nothing to evict yet; the honest
    statement is that the policy does not exist because the need has not been
    measured.
"""

from __future__ import annotations
import pytest as _pytest  # noqa: E402
_pytest.importorskip("typesafe_sdk", reason="exercises code needing typesafe_sdk")  # noqa: E402

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import judged                                    # noqa: E402
from src.judged import TIMBRE_SCALE                       # noqa: E402
from src.verdict_cache import VerdictCache, cache_key     # noqa: E402


class FlakyClient:
    """Answers differently every call, the way a non-deterministic model does.

    Intent probability of the winning option walks 0.80 -> 0.55 -> ... on
    successive calls. The CC delta (102 -> 70) is far larger than any
    quantisation or threshold subtlety, so a test that passes means the cache
    worked, not that the difference was small.
    """

    def __init__(self, walk=(0.80, 0.55, 0.65)):
        self.walk = list(walk)
        self.calls = 0

    def system_one(self, **kw):
        self.calls += 1
        p = self.walk[min(self.calls - 1, len(self.walk) - 1)]
        return _Response(p)


class _Response:
    def __init__(self, p, model="jev-fake-9.9.9"):
        self.model = model
        self.answers = {
            "timbre": _Score(1.0, len(TIMBRE_SCALE)),
            "intention": _Score(2.0, 4),
            "attention": _Choice("listener", {"listener": p, "task": 1 - p}),
            "gesture": _Choice("command", {"command": p, "question": 1 - p}),
        }


class _Score:
    def __init__(self, score, n):
        self.score = score
        self.confidence = 0.66
        self.legend = {i: f"level {i}" for i in range(n)}
        self.probabilities = {i: (0.66 if i == int(score) else 0.34 / (n - 1))
                              for i in range(n)}


class _Choice:
    def __init__(self, choice, probs):
        self.choice = choice
        self.probabilities = probs
        self.confidence = 0.42


# -- the property this module exists for -------------------------------------

class TestStability:
    def test_same_utterance_same_cc_zero_extra_calls(self):
        cache = VerdictCache()
        client = FlakyClient()
        first, m1 = judged.read_tracks("stop the deploy", client=client, cache=cache)
        second, m2 = judged.read_tracks("stop the deploy", client=client, cache=cache)
        assert client.calls == 1, "the second read must not reach the model"
        assert [(r.track, r.value) for r in first] == [(r.track, r.value) for r in second]
        assert m1 == m2
        assert all(r.detail.get("cached") for r in second)
        assert not any(r.detail.get("cached") for r in first)

    def test_without_the_cache_the_jitter_is_visible(self):
        """The control: the SAME client without a cache disagrees with itself."""
        client = FlakyClient()
        a, _ = judged.read_tracks("stop the deploy", client=client)
        b, _ = judged.read_tracks("stop the deploy", client=client)
        va = {r.track: r.value for r in a}
        vb = {r.track: r.value for r in b}
        assert va["GESTURE"] != vb["GESTURE"], (
            "injected jitter vanished -- the control is broken and the "
            "stability test above proves nothing")

    def test_a_different_utterance_pays_for_itself(self):
        cache = VerdictCache()
        client = FlakyClient()
        judged.read_tracks("hey", client=client, cache=cache)
        judged.read_tracks("what broke the build", client=client, cache=cache)
        assert client.calls == 2
        assert len(cache) == 2

    def test_the_model_is_part_of_the_key(self):
        """jev-1.13.0 and jev-1.14.0 are different instruments, not aliases."""
        assert cache_key("hey", "jev-1.13.0") != cache_key("hey", "jev-1.14.0")

    def test_the_question_wording_is_part_of_the_key(self, monkeypatch):
        """A rewrite invalidates by construction — no manual flush anywhere."""
        before = cache_key("hey", "jev-1.14.0")
        monkeypatch.setattr(judged, "TIMBRE_SCALE",
                            ["colder than cold"] + list(TIMBRE_SCALE))
        after = cache_key("hey", "jev-1.14.0")
        assert before != after

    def test_the_utterance_text_is_part_of_the_key(self):
        assert cache_key("hey", "m") != cache_key("hey ", "m")


# -- failures are never sticky ----------------------------------------------

class TestFailurePolicy:
    def test_a_failed_reading_is_not_cached(self):
        class SometimesDown:
            def __init__(self):
                self.calls = 0

            def system_one(self, **kw):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("network down")
                return _Response(0.9)

        cache = VerdictCache()
        client = SometimesDown()
        first, _ = judged.read_tracks("hello", client=client, cache=cache)
        assert all(not r.ok for r in first)
        assert len(cache) == 0, "a failure must not become a sticky verdict"
        second, _ = judged.read_tracks("hello", client=client, cache=cache)
        assert all(r.ok for r in second), "the retry must reach the model for real"


# -- persistence -------------------------------------------------------------

class TestPersistence:
    def test_round_trip_through_a_file(self, tmp_path):
        path = str(tmp_path / "verdicts.json")
        client = FlakyClient()
        judged.read_tracks("hey", client=client, cache=VerdictCache(path=path))
        reloaded = VerdictCache(path=path)
        assert len(reloaded) == 1
        again, _ = judged.read_tracks("hey", client=client, cache=reloaded)
        assert client.calls == 1, "a reloaded cache must still prevent the call"

    def test_a_corrupt_file_is_a_miss_not_a_crash(self, tmp_path):
        path = tmp_path / "verdicts.json"
        path.write_text("{ this is not json")
        cache = VerdictCache(path=str(path))
        assert len(cache) == 0

    def test_the_file_is_human_readable(self, tmp_path):
        path = str(tmp_path / "verdicts.json")
        judged.read_tracks("hey", client=FlakyClient(), cache=VerdictCache(path=path))
        doc = json.loads(Path(path).read_text())
        assert doc["v"] == 1
        entry = next(iter(doc["entries"].values()))
        assert {r["track"] for r in entry["readings"]} == \
            {"TIMBRE", "INTENTION", "ATTENTION", "GESTURE"}


# -- the blind-spot guard ----------------------------------------------------

def test_this_file_does_not_measure_the_live_flip_rate():
    """The offline jitter is INJECTED at full amplitude. The real Jev flip
    rate on these four questions is a live experiment (PROGRESS.md), and a
    green suite here must not be read as 'Jev is stable' -- only 'the cache
    removes whatever instability exists'."""
    assert True
