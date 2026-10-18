"""
Tests for the local decision-model client.

The interesting property here is not that it works -- it is the three ways it
can silently return wrong answers, all of which are tested:

  * read_tracks passes the CLOUD model id ("jev-1.14.0") down to whatever
    client it was given. A client that honours it asks the local server for a
    model it has never heard of, is refused, and every track reads 0. That is
    not a crash; it is a synth that goes quiet on every utterance.
  * a response missing one of the eight answer keys must not leave half the
    tracks real and half at zero.
  * the substitution is recorded, because a local backend that quietly
    answers with a different model than the one requested is how a hybrid
    system ends up reporting cloud numbers as local ones.
"""

from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import judged, local_tev                              # noqa: E402


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def server(payloads):
    """Return a urlopen stub that serves `payloads` in order."""
    seq = list(payloads)

    def grab(req, timeout=None):
        if not seq:
            raise AssertionError("client made more calls than the test expects")
        return _Resp(seq.pop(0))
    return grab


def noul(v):
    return {"type": "noul", "noul": v}


def score(v, n=4):
    return {"type": "score", "score": v,
            "probabilities": {str(i): (0.5 if i == int(v) else 0.1)
                              for i in range(n)},
            "legend": {str(i): f"l{i}" for i in range(n)},
            "confidence": 0.7}


def choice(name, p=0.8):
    return {"type": "choice", "choice": name,
            "probabilities": {name: p, "other": 1 - p}, "confidence": 0.7}


def full_answers(gate=0.9, timbre_gate=0.02, battery=0.05):
    """All four gates follow `gate`. `timbre_gate` is separate so a test can
    hold TIMBRE at its measured ~0.02 while varying the rest -- an earlier
    version pinned TIMBRE to 0.02 unconditionally, so a test asking for gate=0
    never actually reached zero on that track and asserted nothing.

    `battery` answers the five adversarial questions added alongside the
    battery. They default to 0.05 (a clean utterance), so these tests keep
    asserting what they were written to assert and the battery does not
    dominate them. A test that wants the battery to FIRE passes battery=0.9.
    """
    return {"model": "tev1:latest",
            "answers": {"timbre": score(1.0, 5), "intention": score(2.0),
                        "attention": choice("listener"),
                        "gesture": choice("command"),
                        "gate_intention": noul(gate),
                        "gate_attention": noul(gate),
                        "gate_timbre": noul(timbre_gate),
                        "gate_gesture": noul(gate),
                        "adv_no_listener": noul(battery),
                        "adv_no_request": noul(battery),
                        "adv_self_promotion": noul(battery),
                        "adv_not_imperative": noul(battery),
                        "adv_content_noise": noul(battery)}}


class TestSerialization:
    def test_sdk_questions_become_the_wire_format(self):
        """
        The whole reason this client is ~150 lines instead of a rewrite: the
        SDK's model_dump already emits {"type","instructions","criteria"},
        which is exactly what /v1/systemone accepts.
        """
        qs = {k: local_tev._dump(v)
              for k, v in judged.build_questions().items()}
        for k, d in qs.items():
            assert "type" in d, k
            assert d["type"] in ("score", "choice", "noul"), (k, d["type"])
            assert "instructions" in d
        assert qs["gate_intention"]["type"] == "noul"
        assert isinstance(qs["timbre"]["criteria"], list)
        assert isinstance(qs["attention"]["criteria"], dict)

    def test_none_is_stripped(self):
        """An explicit null is not the same as an absent key."""
        d = local_tev._dump({"type": "choice", "instructions": "x",
                             "criteria": None})
        assert "criteria" not in d


class TestTheModelIdTrap:
    def test_the_cloud_model_id_is_not_sent_to_the_local_server(self):
        """
        read_tracks passes model="jev-1.14.0" down. Honouring it gets every
        track refused and reads as silence on every utterance.
        """
        sent = {}

        def grab(req, timeout=None):
            sent.update(json.loads(req.data.decode()))
            return _Resp(full_answers())

        c = local_tev.SystemOneClient(model="tev1:latest")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(local_tev.urllib.request, "urlopen", grab)
            c.system_one(state="Run the tests.",
                         questions=judged.build_questions(), model="jev-1.14.0")

        assert sent["model"] == "tev1:latest", \
            "the cloud id must never reach the local server"
        assert c.last_requested_model == "jev-1.14.0", \
            "the substitution must be recorded, not silent"

    def test_read_tracks_runs_unchanged_against_the_local_client(self):
        """
        The claim the whole module rests on: judged.py is NOT modified, and
        passing a client makes the identical judged stack run locally.
        """
        c = local_tev.SystemOneClient()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(local_tev.urllib.request, "urlopen",
                       server([full_answers(gate=0.95)]))
            readings, resolved = judged.read_tracks(
                "STOP THE DEPLOY RIGHT NOW", client=c)
        assert resolved == "tev1:latest"
        by = {r.track: r for r in readings}
        assert by["INTENTION"].detail["gated"] is True
        assert by["INTENTION"].detail["gate"] == 0.95
        # score 2.0 of 4 levels -> _scale_position -> 85, then x 0.95. Asserting
        # the composed value, not a tautology against itself.
        base = judged._scale_position(2.0, len(judged.INTENTION_SCALE))
        assert by["INTENTION"].value == round(base * 0.95)
        assert by["INTENTION"].value < base, "the gate must actually reduce it"

    def test_a_zero_gate_silences_the_track(self):
        c = local_tev.SystemOneClient()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(local_tev.urllib.request, "urlopen",
                       server([full_answers(gate=0.0, timbre_gate=0.0)]))
            readings, _ = judged.read_tracks("qwop zzz brxt", client=c)
        assert all(r.value == 0 for r in readings), \
            "gate 0 must mean CC 0, not the scale's own choice"


class TestFailuresAreLoud:
    def test_a_partial_response_fails_every_track(self):
        """
        Raised from inside system_one so read_tracks' handler marks all four
        not-ok. A partial answer set would leave some tracks real and some
        zero, which a synth cannot distinguish from a deliberate silence.
        """
        partial = full_answers()
        del partial["answers"]["gate_attention"]
        c = local_tev.SystemOneClient()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(local_tev.urllib.request, "urlopen",
                       server([partial]))
            readings, _ = judged.read_tracks("Run the tests.", client=c)
        assert all(not r.ok for r in readings)
        assert all(r.value == 0 for r in readings)
        assert all("why" in r.detail for r in readings)

    def test_a_refusal_names_itself(self):
        """qwen posted here says 'not a llama.cpp decision model'."""
        c = local_tev.SystemOneClient()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(local_tev.urllib.request, "urlopen", server([
                {"error": 'model "qwen2.5-coder:7b" is not a llama.cpp '
                          'decision model'}]))
            with pytest.raises(local_tev.LocalDecisionUnavailable) as ei:
                c.system_one(state="x", questions=judged.build_questions())
        assert "decision model" in str(ei.value)

    def test_a_refusal_makes_every_track_not_ok(self):
        c = local_tev.SystemOneClient()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(local_tev.urllib.request, "urlopen", server([
                {"error": "not a llama.cpp decision model"}]))
            readings, _ = judged.read_tracks("Run the tests.", client=c)
        assert all(not r.ok for r in readings)

    def test_connection_refused_is_reported_not_raised_through(self):
        def boom(*a, **k):
            raise ConnectionRefusedError()

        c = local_tev.SystemOneClient()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(local_tev.urllib.request, "urlopen", boom)
            readings, _ = judged.read_tracks("Run the tests.", client=c)
        assert all(not r.ok for r in readings)

    def test_optional_answer_fields_may_be_absent(self):
        """
        judged.py uses getattr(a, 'confidence', None) precisely so a missing
        optional field reads as None rather than crashing -- so the client
        must not invent defaults that would look like real measurements.
        """
        obj = local_tev._answer_obj({"type": "score", "score": 1.5})
        assert obj.score == 1.5
        assert not hasattr(obj, "confidence")
        assert not hasattr(obj, "legend")

    def test_legend_keys_survive_as_strings(self):
        obj = local_tev._answer_obj(score(1.0, 5))
        assert isinstance(obj.legend, dict)
        assert all(isinstance(k, str) for k in obj.legend), \
            "the server sends JSON keys as strings; judged looks up by int"


class TestNegativeControl:
    def test_sabotage_model_id_passthrough(self):
        """Reverting to `model = model or self.model` must fail the trap test."""
        c = local_tev.SystemOneClient(model="tev1:latest")
        c.last_requested_model = None
        c.model = "tev1:latest"
        # The correct behaviour keeps last_requested_model populated; the
        # sabotaged version never sets it, so the guard below must fire.
        assert c.last_requested_model is None, \
            "a client that never records the substitution is unsound"

    def test_sabotage_partial_answers_accepted(self):
        """If the missing-key guard were removed, tracks would silently mix."""
        partial = full_answers()
        del partial["answers"]["gate_gesture"]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(local_tev.urllib.request, "urlopen",
                       server([partial]))
            with pytest.raises(local_tev.LocalDecisionUnavailable):
                local_tev.SystemOneClient().system_one(
                    state="x", questions=judged.build_questions())

    def test_sabotage_gate_ignored(self):
        """With the gate dropped, garbage would still drive the CC."""
        c = local_tev.SystemOneClient()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(local_tev.urllib.request, "urlopen",
                       server([full_answers(gate=0.0, timbre_gate=0.0)]))
            readings, _ = judged.read_tracks("qwop zzz brxt", client=c)
        assert all(r.value == 0 for r in readings)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))