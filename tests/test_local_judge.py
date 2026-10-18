"""
Tests for the local judge.

WHAT THIS PROVES: that the local path fails LOUDLY. A local model that is
unreachable, that refuses the request, or that returns half an answer must
produce ok=False with a reason -- never a track at CC 0 that a synthesiser
would read as a deliberate silence.

That distinction is the whole risk of a local path. In the cloud, a failure is
a visible exception with a stack trace. Locally it is a process that is not
running, and the natural failure mode is silence, which in a voice instrument
is indistinguishable from a person who stopped talking.

tev1:latest is the concrete case: it declares capability 'decision' and
Ollama 0.40.5 refuses it on BOTH /api/generate ("does not support generate")
and /api/chat ("does not support chat"). A judge that treats that as "no
answer" instead of "no model" reports a dead instrument as a quiet room.
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

from src import judged, local_judge as lj      # noqa: E402


def full_obj(**over):
    """A complete, well-formed local answer."""
    obj = {
        "timbre": {"position": 1.0, "confidence": 0.5},
        "intention": {"position": 3.0, "confidence": 0.9},
        "attention": {"probabilities": {"self": 0.1, "listener": 0.7,
                                        "task": 0.1, "tool": 0.05, "none": 0.05}},
        "gesture": {"probabilities": {"question": 0.1, "command": 0.8,
                                      "statement": 0.05, "reaction": 0.03,
                                      "greeting": 0.02}},
        "gate_intention": 0.95,
        "gate_attention": 0.9,
        "gate_timbre": 0.1,
        "gate_gesture": 0.93,
    }
    obj.update(over)
    return obj


class TestSchemaFollowsTheScales:
    def test_schema_is_built_from_the_live_constants(self):
        """
        The schema must be DERIVED from judged's scales, not copied. A copied
        schema drifts the moment a level is added, and the local model would
        silently answer a question the cloud path no longer asks -- two paths
        that look identical and measure different things.
        """
        s = lj.build_schema()["properties"]
        assert s["timbre"]["properties"]["position"]["maximum"] == \
            len(judged.TIMBRE_SCALE) - 1
        assert s["intention"]["properties"]["position"]["maximum"] == \
            len(judged.INTENTION_SCALE) - 1
        assert set(s["attention"]["properties"]["probabilities"]["properties"]) \
            == set(judged.ATTENTION_CHOICE)
        assert set(s["gesture"]["properties"]["probabilities"]["properties"]) \
            == set(judged.GESTURE_CHOICE)

    def test_schema_requires_every_gate(self):
        s = lj.build_schema()["properties"]
        for key in judged.GATES:
            assert key in s, f"{key} missing from the schema"
        assert set(lj.build_schema()["required"]) == set(s)

    def test_prompt_states_every_scale_and_gate(self):
        p = lj.build_prompt("Run the tests.")
        for key in ("timbre", "intention", "attention", "gesture"):
            assert key in p
        for key in judged.GATES:
            assert key in p
        assert "Run the tests." in p


class TestReductionMatchesTheCloudPath:
    def test_score_position_uses_the_shared_scale(self):
        r = lj.reduce_local(full_obj())
        assert r.ok
        # INTENTION position 3.0 of 4 levels -> the last level, CC 127,
        # then multiplied by gate 0.95.
        assert r.ccs["INTENTION"] == round(127 * 0.95)
        assert r.gates["INTENTION"] == 0.95

    def test_choice_cc_is_the_winning_probability(self):
        r = lj.reduce_local(full_obj())
        assert r.raw["choices"]["ATTENTION"] == "listener"
        assert r.ccs["ATTENTION"] == round(0.7 * 127 * 0.9)

    def test_gate_multiplication_matches_the_cloud_reduction(self):
        """
        Same rule, same direction: a local track silenced by gate 0 must read
        CC 0, or the local path would be a way to drive garbage.
        """
        r = lj.reduce_local(full_obj(gate_intention=0.0))
        assert r.ccs["INTENTION"] == 0

    def test_timbre_gate_near_zero_is_reported_not_hidden(self):
        """
        The expected value is base x gate, NOT gate alone. An earlier version
        of this test asserted round(127 * 0.1) == 13 and got 3, because the
        TIMBRE position 1.0 of 5 levels already scales to 32 before the gate
        multiplies it. The test was wrong about its own arithmetic, not the
        reduction -- worth fixing explicitly, because "the gate is the answer"
        is exactly the wrong mental model to encode here.
        """
        r = lj.reduce_local(full_obj())
        base = judged._scale_position(1.0, len(judged.TIMBRE_SCALE))
        assert base == 32, "position 1.0 of 5 levels is a quarter of the range"
        assert r.gates["TIMBRE"] == 0.1, "the gate itself is reported intact"
        assert r.ccs["TIMBRE"] == round(base * 0.1)


class TestFailuresAreLoud:
    @pytest.mark.parametrize("mutate,why", [
        (lambda o: o.pop("gesture"), "KeyError"),
        (lambda o: o.update(gate_intention="high"), "ValueError"),
        (lambda o: o.update(intention={"position": None}), "TypeError"),
        (lambda o: o.update(attention="listener"), "TypeError"),
        (lambda o: o.update(gate_timbre=[0.5]), "TypeError"),
    ])
    def test_malformed_input_fails_the_whole_reading(self, mutate, why):
        """
        Whole-reading failure, never partial. Half the tracks real and half at
        CC 0 is indistinguishable from a synthesiser that went quiet, and it
        would be read as data.
        """
        obj = full_obj()
        mutate(obj)
        r = lj.reduce_local(obj)
        assert r.ok is False
        assert r.why.startswith("reduce:")

    def test_a_decision_only_model_is_not_read_as_silence(self):
        """
        The tev1 case. Ollama answers with an `error` key and NO `message`;
        treating a missing message as "no answer" reports a dead instrument as
        a quiet room.
        """
        body = {"error": '"tev1:latest" does not support chat'}

        class _Resp:
            def read(self):
                return json.dumps(body).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(lj.urllib.request, "urlopen", lambda *a, **k: _Resp())
            out = lj.LocalJudge().judge("hello")
        assert out.ok is False
        assert "does not support" in out.why
        assert out.ccs == {}, "an error must not leave any CC behind"

    def test_non_json_content_fails(self):
        class _Resp:
            def read(self):
                return json.dumps({"message": {"content": "sorry!"}}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(lj.urllib.request, "urlopen", lambda *a, **k: _Resp())
            out = lj.LocalJudge().judge("hello")
        assert out.ok is False and out.why.startswith("json:")

    def test_http_error_is_reported_not_raised(self):
        def boom(*a, **k):
            raise urllib.error.HTTPError("u", 503, "busy", None, None)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(lj.urllib.request, "urlopen", boom)
            out = lj.LocalJudge().judge("hello")
        assert out.ok is False and out.why == "http503"

    def test_connection_refused_is_reported_not_raised(self):
        def boom(*a, **k):
            raise ConnectionRefusedError("nothing listening")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(lj.urllib.request, "urlopen", boom)
            out = lj.LocalJudge().judge("hello")
        assert out.ok is False and out.why == "ConnectionRefusedError"

    def test_available_is_false_when_nothing_listens(self):
        def boom(*a, **k):
            raise ConnectionRefusedError()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(lj.urllib.request, "urlopen", boom)
            assert lj.LocalJudge().available() is False

    def test_temperature_is_zero(self):
        """
        A controller that jitters for no reason is an instrument nobody can
        play. jev-lattice measured Jev itself flipping 7% of gates on a pure
        re-ask; a sampling temperature on top of that would be indefensible.
        """
        captured = {}

        class _Resp:
            def read(self):
                return json.dumps({"message": {"content":
                                               json.dumps(full_obj())}}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def grab(req, timeout=None):
            captured.update(json.loads(req.data.decode()))
            return _Resp()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(lj.urllib.request, "urlopen", grab)
            out = lj.LocalJudge().judge("hello")
        assert out.ok
        assert captured["options"]["temperature"] == 0
        assert captured["format"] == lj.build_schema(), \
            "structured output must be requested, or prose gets parsed"


# -- negative control -------------------------------------------------------

class TestNegativeControl:
    def test_sabotage_gate_ignored(self):
        """Dropping the gate multiplication must be visible."""
        r = lj.reduce_local(full_obj(gate_intention=0.0))
        assert r.ccs["INTENTION"] == 0
        ungated = lj.reduce_local(full_obj(gate_intention=1.0))
        assert r.ccs["INTENTION"] != ungated.ccs["INTENTION"]

    def test_sabotage_partial_reading_allowed(self):
        """
        If reduction ever returned ok=True with a missing track, this catches
        it: the tracks that DID parse would look like real data.
        """
        obj = full_obj()
        obj.pop("gesture")
        r = lj.reduce_local(obj)
        assert r.ok is False
        assert r.why
        # ok is the contract; ccs may be partial (the audit found the old
        # `assert r.ccs == {} or True` hiding exactly this: implementation
        # returns partial ccs on failure, so we assert the CONTRACT, not
        # an emptiness the code never guaranteed).

    def test_sabotage_argmax_to_first_key(self):
        """Taking probabilities[0] instead of the max must change the CC."""
        obj = full_obj(attention={"probabilities": {
            "self": 0.6, "listener": 0.2, "task": 0.1, "tool": 0.05,
            "none": 0.05}})
        r = lj.reduce_local(obj)
        assert r.raw["choices"]["ATTENTION"] == "self"
        assert r.ccs["ATTENTION"] != round(0.2 * 127 * 0.9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))