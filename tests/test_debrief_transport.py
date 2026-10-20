"""
test_debrief_transport.py -- a cut-off answer must not look like a finished one.

WHY THESE TESTS EXIST.

Three of the nine committed audit runs came back "unparseable": the model's
answer was not valid JSON. The record said only "unparseable", and the
natural reading is that the model failed to answer.

It did not. The API said so, and this client threw the evidence away.

`experiments/debrief.py::_http_json` rebuilt the response as
`{"text": text}` and DROPPED `finish_reason` and `usage`. A response that
the output budget cut off mid-answer and a response the model finished are
byte-for-byte the same shape on the wire, once you have thrown away
`finish_reason`. The symptom was qwen3 and gpt-oss writing a prose
preamble, running out of room before the JSON arrived, and the record
saying nothing about why.

`cloud_llm` already had a retry for reasoning models -- it repeats the call
with 4x the output budget when `content` comes back EMPTY. The truncated
case never reached it, because the content was not empty; it was cut.

A HYPOTHESIS THAT WAS REFUTED FIRST. The obvious explanation was that the
material budget was too large for the smaller models' context windows, and
that the fit ladder should have shrunk it. That was tested and it is
FALSE. Measured, same model, same mode, same key, at the committed budget
of 140,000 characters:

    qwen3-30b, blind mode
      140,000 chars -> parsed, headline +0.1111
       60,000 chars -> parsed, headline +0.1111
       40,000 chars -> UNPARSEABLE, and the raw text is a prose preamble

The failure happened at the SMALLEST budget. So the budget was not the
cause, and a ladder that treated "unparseable" as "does not fit" would have
walked DOWN past budgets that worked. The cause is the OUTPUT budget, not
the INPUT budget.

These tests pin the fix:
  1. `finish_reason` survives `_http_json`, and `truncated` is set from it.
  2. `cloud_llm` retries a truncated answer with 4x the output budget.
  3. `cloud_llm` does NOT retry a complete answer (no wasted calls).
  4. `cloud_llm` does NOT retry a context error (that is the fit ladder's
     job, and the two mechanisms must not fight each other).
  5. `run_audit` records the transport facts on a SUCCESSFUL run, so a
     truncation is visible even when the answer happened to parse.
  6. `run_audit` distinguishes the two unparseable shapes that used to
     share one label.
  7. A MUTATION TEST: if the truncation detection is removed, the suite
     fails. A test that cannot fail is not evidence.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import debrief  # noqa: E402


def _result(text, finish_reason="stop", usage=None):
    """A Workers AI response body, in the shape `_http_json` receives."""
    msg = {"content": text}
    out = {"choices": [{"message": msg, "finish_reason": finish_reason}]}
    if usage:
        out["usage"] = usage
    return {"result": out}


class _FakeResp:
    """A urlopen response stand-in that serves one canned body."""

    def __init__(self, raw):
        self._raw = raw.encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _serve(monkeypatch, body):
    """Make urlopen return `body`, so `_http_json` can be tested end to end."""
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp(json.dumps(body)))


# --------------------------------------------------------------- _http_json

def test_finish_reason_survives_the_transport(monkeypatch):
    """A response that was cut off must be distinguishable from a finished one."""
    _serve(monkeypatch, _result("partial text that got cut off",
                                 finish_reason="length"))
    out = debrief._http_json("https://example.invalid", {}, {})
    assert out["finish_reason"] == "length"
    assert out["truncated"] is True
    assert out["text"] == "partial text that got cut off"


def test_a_finished_answer_is_not_marked_truncated(monkeypatch):
    _serve(monkeypatch, _result('{"verdicts": []}', finish_reason="stop"))
    out = debrief._http_json("https://example.invalid", {}, {})
    assert out["finish_reason"] == "stop"
    assert out["truncated"] is False


def test_usage_is_kept_when_the_api_reports_it(monkeypatch):
    _serve(monkeypatch, _result("ok", finish_reason="stop",
                                usage={"total_tokens": 1234}))
    out = debrief._http_json("https://example.invalid", {}, {})
    assert out["usage"]["total_tokens"] == 1234


# --------------------------------------------------------------- cloud_llm

def _stub_llm(monkeypatch, responses):
    """Replace the HTTP call with a queue of canned responses."""
    calls = []
    monkeypatch.setenv("CF_TOKEN", "test-token")

    def fake_http(url, body, headers, timeout=240.0):
        calls.append({"url": url, "max_tokens": body.get("max_tokens")})
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(debrief, "_http_json", fake_http)
    monkeypatch.setattr(debrief, "cf_agree", lambda m: True)
    return calls


def test_a_truncated_answer_is_retried_with_more_room(monkeypatch):
    calls = _stub_llm(monkeypatch, [
        {"text": "cut off here", "finish_reason": "length", "truncated": True},
        {"text": '{"verdicts": []}', "finish_reason": "stop", "truncated": False},
    ])
    r = debrief.cloud_llm("@cf/test/model", "prompt", max_tokens=3000)
    assert len(calls) == 2, "the truncated answer must be retried, not accepted"
    assert calls[0]["max_tokens"] == 3000
    assert calls[1]["max_tokens"] == 12000, "the retry must ask for 4x the room"
    assert r["retried_with_more_tokens"] is True
    assert r["retried_because"] == "truncated"
    assert r["first_attempt_finish_reason"] == "length"
    assert r["text"] == '{"verdicts": []}'


def test_a_complete_answer_is_not_retried(monkeypatch):
    calls = _stub_llm(monkeypatch, [
        {"text": '{"verdicts": []}', "finish_reason": "stop", "truncated": False},
    ])
    r = debrief.cloud_llm("@cf/test/model", "prompt", max_tokens=3000)
    assert len(calls) == 1, "a finished answer must not spend another call"
    assert "retried_with_more_tokens" not in r


def test_a_context_error_is_not_retried_by_the_output_budget(monkeypatch):
    """A context error is the FIT LADDER's business, not the output budget's.

    Retrying it here would double every context rejection and would mask
    the ladder's own accounting.
    """
    calls = _stub_llm(monkeypatch, [
        {"error": "http413", "detail": "maximum context length is 24000 tokens"},
    ])
    r = debrief.cloud_llm("@cf/test/model", "prompt", max_tokens=3000)
    assert len(calls) == 1
    assert r["error"] == "http413"


def test_an_empty_answer_is_still_retried(monkeypatch):
    """The ORIGINAL retry must keep working. The new one is an addition."""
    calls = _stub_llm(monkeypatch, [
        {"text": "", "finish_reason": "stop", "content_empty": True},
        {"text": '{"verdicts": []}', "finish_reason": "stop", "truncated": False},
    ])
    r = debrief.cloud_llm("@cf/test/model", "prompt", max_tokens=3000)
    assert len(calls) == 2
    assert r["retried_because"] == "content_empty"


# --------------------------------------------------------------- run_audit

def test_run_audit_records_the_transport_facts_on_a_good_run(monkeypatch):
    verdicts = [{"index": i, "verdict": "supported", "reason": "x"}
                for i in range(10)]
    _stub_llm(monkeypatch, [
        {"text": json.dumps({"verdicts": verdicts}),
         "finish_reason": "stop", "truncated": False},
    ])
    monkeypatch.setattr(debrief, "audit_pairs",
                        lambda seed: [("claim %d" % i, i % 2 == 0)
                                      for i in range(10)])
    r = debrief.run_audit("@cf/test/model", "blind", materials_limit=1000)
    assert r.get("error") is None
    assert r["finish_reason"] == "stop"
    assert r["truncated"] is False
    assert r["retried_with_more_tokens"] is False


def test_the_two_unparseable_shapes_no_longer_share_one_label(monkeypatch):
    """A truncated answer and a malformed one are different facts.

    They used to be recorded as the same string, "unparseable", which is why
    the cause went unexamined for a whole run of the audit.
    """
    monkeypatch.setattr(debrief, "audit_pairs",
                        lambda seed: [("claim %d" % i, i % 2 == 0)
                                      for i in range(10)])
    _stub_llm(monkeypatch, [
        {"text": "prose preamble that never reached the json",
         "finish_reason": "length", "truncated": True},
    ])
    r = debrief.run_audit("@cf/test/model", "blind", materials_limit=1000)
    assert r["error"] == "unparseable"
    assert r["why"] == "response_was_truncated"
    assert r["truncated"] is True

    _stub_llm(monkeypatch, [
        {"text": "I would rather not answer in JSON.",
         "finish_reason": "stop", "truncated": False},
    ])
    r2 = debrief.run_audit("@cf/test/model", "blind", materials_limit=1000)
    assert r2["error"] == "unparseable"
    assert r2["why"] == "no_json_verdicts_object"
    assert r2["truncated"] is False


# ----------------------------------------------------------- mutation test

def test_the_fix_is_load_bearing(monkeypatch):
    """Remove the truncation detection and this test must fail.

    A suite that cannot fail is not evidence. This is the check that the
    other tests in this file are testing something real.
    """
    src = (ROOT / "experiments" / "debrief.py").read_text()
    assert '"truncated": ch[0].get("finish_reason") == "length"' in src, (
        "the truncation detection is gone; the rest of this file now tests "
        "a client that cannot see a cut-off answer")
    assert 'r.get("truncated")' in src, (
        "the retry no longer looks at the truncation flag")
