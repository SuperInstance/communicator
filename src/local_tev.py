"""
local_tev -- a drop-in client for the LOCAL decision model.

This is the piece the local-first design was missing, and it turned out to be
almost nothing.

`tev1:latest` on Ollama 0.40.5 declares `capabilities: ['decision']` and
refuses `/api/generate` and `/api/chat` ("does not support"), which is why
Phase 2.8 wrote it off as unservable and swept general-purpose models
instead. It is neither a fork nor a plugin. It is served by an undocumented
first-party route:

    POST /v1/systemone   ->   {"type": "noul"|"choice"|"score", ...}

...whose questions and answers are the SAME shapes as the TypeSafe SDK's:

    Score      criteria list   -> score, legend, probabilities, confidence
    Choice     criteria map    -> choice, probabilities, confidence
    Noul       no criteria     -> noul  (probability of yes)

The SDK's own `model_dump()` on judged.build_questions() therefore emits the
wire format byte for byte. There is no translation layer, no prompt
engineering and no second question set to keep in sync -- which is the only
reason this is cheap rather than a rewrite.

MEASURED, against the same batteries as Phases 2.7 and 2.8:

    tev1 via /v1/systemone   0.34 s per turn   (qwen2.5-coder:7b: 7.17 s)
    gate on real speech      0.95
    gate on "qwop zzz brxt" 0.16

So the judged half has a local backend that fits inside a voice loop, and it
answers the presence-gate question -- the one that fixed the three ill-posed
questions -- with the right sign.

THE HYBRID SEAM WAS ALREADY BUILT. judged.read_tracks takes `client=`, so
passing a SystemOneClient runs the entire judged stack locally with no change
to judged.py: same questions, same reduction, same CC scale, same gating. The
thing that made this possible is that the local model speaks the same
primitives as the cloud one rather than being a chat model that has to be
prompted into JSON -- which is exactly what the sweep of general models had
to do, and exactly why those models measured prompt-following instead of
capability.

ONE FAILURE MODE IS HANDLED HERE RATHER THAN IN judged.py. The answer loop in
read_tracks sits OUTSIDE its try/except, so a response missing one of the
eight keys raises KeyError uncaught. This client therefore validates that
every requested key came back and raises from inside `system_one`, where
read_tracks' existing handler turns it into `ok=False` on all four tracks --
no track reads as a real zero.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "tev1:latest"
ROUTE = "/v1/systemone"


class LocalDecisionUnavailable(RuntimeError):
    """The local decision model is absent, refused, or answered partially."""


def _answer_obj(payload: dict):
    """
    Rebuild the attribute surface judged.py reads.

    Not SimpleNamespace for every field: the fields are OMITTED when the
    server did not answer them, and judged.py uses `getattr(a, "confidence",
    None)` specifically so a missing optional field reads as None rather than
    as a crash. So only the fields the server actually sent become attributes.
    """
    kind = payload.get("type")
    fields = {}
    if kind == "noul":
        if "noul" in payload:
            fields["noul"] = float(payload["noul"])
    elif kind == "score":
        if "score" in payload:
            fields["score"] = float(payload["score"])
        if "probabilities" in payload:
            fields["probabilities"] = {int(k): float(v) for k, v
                                       in payload["probabilities"].items()}
        if "legend" in payload:
            fields["legend"] = payload["legend"]
    elif kind == "choice":
        if "choice" in payload:
            fields["choice"] = payload["choice"]
        if "probabilities" in payload:
            fields["probabilities"] = {k: float(v) for k, v
                                       in payload["probabilities"].items()}
    if "confidence" in payload:
        fields["confidence"] = float(payload["confidence"])
    return SimpleNamespace(**fields)


class _Response:
    def __init__(self, model: str, answers: dict):
        self.model = model
        self.answers = answers


class SystemOneClient:
    """
    Shaped like the TypeSafe SDK's client, deliberately.

    `system_one(state=..., questions=..., model=..., timeout=...)` is the same
    signature judged.read_tracks already calls, so this is substitutable
    without a single change upstream.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.calls = 0
        self.last_latency_s: float | None = None
        self.last_requested_model: str | None = None

    # -- introspection ------------------------------------------------------

    def available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as r:
                body = json.loads(r.read().decode())
            return any(m["name"] == self.model for m in body.get("models", []))
        except Exception:                                 # noqa: BLE001
            return False

    def is_decision_model(self) -> bool:
        """A model can be installed and still be refused by /v1/systemone."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/show",
                data=json.dumps({"name": self.model}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                body = json.loads(r.read().decode())
            return "decision" in (body.get("capabilities") or [])
        except Exception:                                 # noqa: BLE001
            return False

    # -- the call ------------------------------------------------------------

    def system_one(self, state, questions, model=None, timeout=None):
        # judged.read_tracks passes ITS `model` -- the cloud identifier,
        # "jev-1.14.0" -- down to whatever client it was handed. Honouring it
        # locally asks this server for a model it has never heard of and gets
        # "is not a llama.cpp decision model" back, which surfaces as every
        # track reading 0. That is the trap this line exists to close.
        #
        # The substitution is recorded rather than silent: `last_requested_model`
        # keeps the cloud id so a reader can see that the answer came from a
        # DIFFERENT model than the one that was asked for. A local backend that
        # quietly answers with something else is how a hybrid system ends up
        # reporting cloud numbers as if they were local ones.
        self.last_requested_model = model
        model = self.model
        payload = {
            "model": model,
            "state": state,
            # judged.build_questions() returns SDK models; their model_dump is
            # already {"type", "instructions", "criteria"} with None dropped.
            "questions": {k: _dump(q) for k, q in questions.items()},
        }
        req = urllib.request.Request(
            f"{self.base_url}{ROUTE}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req,
                                        timeout=timeout or self.timeout) as r:
                body = json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            raise LocalDecisionUnavailable(f"http{exc.code}") from None
        except Exception as exc:                          # noqa: BLE001
            raise LocalDecisionUnavailable(type(exc).__name__) from None
        self.last_latency_s = time.time() - t0
        self.calls += 1

        if "error" in body:
            # e.g. a general chat model posted here: "is not a llama.cpp
            # decision model". Surfaced, not swallowed into silence.
            raise LocalDecisionUnavailable(str(body["error"])[:140])

        raw = body.get("answers") or {}
        missing = [k for k in questions if k not in raw]
        if missing:
            # Raised from INSIDE system_one so judged.read_tracks' existing
            # except turns it into ok=False on every track. Returning a partial
            # answer set would leave some tracks real and some zero, which a
            # synthesiser cannot tell from a deliberate silence.
            raise LocalDecisionUnavailable(
                f"missing answers: {','.join(sorted(missing))}")
        answers = {k: _answer_obj(v) for k, v in raw.items()}
        return _Response(str(body.get("model", model)), answers)


def _dump(q) -> dict:
    """SDK question object -> wire dict, tolerating a plain dict already."""
    if isinstance(q, dict):
        return {k: v for k, v in q.items() if v is not None}
    d = q.model_dump(exclude_none=True)
    # criteria must not be empty for score/choice; the SDK enforces that.
    return d


def probe(model: str = DEFAULT_MODEL,
          base_url: str = DEFAULT_BASE_URL) -> dict:
    """
    One call, both batteries, on the gate that matters. The smallest honest
    demonstration that the local backend works and is not a constant.
    """
    from . import judged

    client = SystemOneClient(base_url=base_url, model=model)
    out = {"model": model, "available": client.available(),
           "decision": client.is_decision_model(), "rows": []}
    for text in ("qwop zzz brxt", "STOP THE DEPLOY RIGHT NOW"):
        t0 = time.time()
        readings, resolved = judged.read_tracks(text, client=client)
        row = {"text": text, "s": round(time.time() - t0, 3),
               "resolved": resolved}
        for r in readings:
            row[r.track] = {"cc": r.value, "gate": r.detail.get("gate"),
                            "gated": r.detail.get("gated")}
        out["rows"].append(row)
    return out