"""
debrief.py -- the LLM as debrief and R&D director between experiments.

WHAT CASEY ASKED FOR, in his words.

"If you use various cloud models like multimodel models and LLMs to analysis too.
you would have a good understanding of the video before you even start to try to
make the system work. this understanding from larger models can be used to
supervise the experimentation and ML and NN. LLMs could actually be the debrief
and r&d between experiments into how to improve better and what kind of
experiment should be next."

Three separate things are in there, and this file does two of them:

  1. UNDERSTANDING OF THE VIDEO.  That is `cloud_multimodal.py`, and it is
     already measuring: a vision model reading the real frames, with the
     blank control that decides whether it knows anything.

  2. THE DEBRIEF BETWEEN EXPERIMENTS.  That is this file. It collects the
     research notes, the result files and the git history, and asks a large
     model three questions: what does the evidence actually support, which
     claims are unsafe, and what single experiment should run next.

  3. SUPERVISING THE EXPERIMENTATION.  The next experiment comes out of the
     debrief with its controls and its falsifier attached, so the loop is
     closed rather than a suggestion.

THE PART THAT MAKES THIS HONEST.

An LLM asked to review evidence will agree with you. It will produce a
fluent, confident, well-structured review of whatever you put in front of it,
and the review will read exactly like insight. This project has been burned
by that shape of thing four documented times -- a verdict function that
contradicted its own printed output, a claim that survived because nobody
could make it fail, a statistic compared over the wrong interval. A tool
that cannot fail is not evidence, and an advisor that cannot fail is not an
advisor.

So the advisor is AUDITED, against this project's own history.

The audit set is drawn from what this project has already retracted and
falsified, plus claims that are still standing:

  RETRACTED / FALSIFIED (the director should REJECT these)
    - a 14.5x reduction in camera drift from a gate threshold
    - the ASCII character delta detecting motion (an empty room flips 50% of
      its dots, which is the theoretical maximum)
    - the watcher being quadratic and its buffer growing without bound
    - gain normalisation cutting drift by ~50x
    - an LLM teacher able to label speech from ASCII grids
    - a "31.8% of frames never measured" figure
    - a conventional teacher scoring 0.000 on leave-one-session-out for
      EVERY feature set (the motion-map feature set scores 1.000 on a held-out
      session, and region+map scores 0.200)
  STILL STANDING (the director should NOT reject these)
    - row-band removal taking the empty room's dot flip from 50.00% to 0.01%
    - the room being silent at -73.5 dBFS with zero valid RT60 estimates
    - composed filters cutting the occupied flip rate 0.0299 -> 0.0014 at a
      flat SNR of ~1.83

THE AUDIT SET IS AUDITED TOO. An audit of `docs/research/` against the result
files found that TWO of these ten claims were mis-marked when the set was
first written: the leave-one-out claim was marked TRUE although the motion-map
feature set scores 1.000 on a held-out session, and the composed-filter
claim was marked TRUE although "21x over either filter" uses the better of
the two singles and calls a flip rate a saturation. Both were corrected and
the audit was re-run, and the set now holds SEVEN false claims and THREE
true ones. An audit set that has not been audited is a quiz with the answers
written on it.

AND THE AUDIT SET IS AUDITED AGAIN. A third value (0.675, for the `both`
feature set) appeared in the committed teacher artefact and was copied into
this description. It does not reproduce: the same script, the same arguments
and the same corpus reproduce every other column of that artefact exactly and
return 0.000 for the held-out session. The value is withdrawn, and the
leave-one-out claim's FALSITY now rests entirely on numbers that do
reproduce -- map 1.000 and region+map 0.200. The claim's truth value did not
change; the evidence for it did, and the evidence is now traceable.
`NOTES-AUDIT.md` F3, `COMBINED-FILTER.md`, `ARTIFACT-REPRODUCIBILITY.md`.

The score has two halves, because a reviewer that rejects EVERYTHING is as
useless as one that accepts everything:

    SENSITIVITY   rejected / (known-false claims)
    SPECIFICITY   not-rejected / (known-true claims)

and the headline is their difference. A director that flags every claim
scores 1.0 sensitivity and 0.0 specificity: a net zero, and a verdict that
is worthless while looking rigorous.

THE MODEL SEES THE CLAIMS, NOT THE ANSWER KEY.

The claims are paraphrased, so the model cannot string-match a retraction
notice in the research notes, and they are presented in a fixed shuffled
order so position cannot leak the answer. The ground truth is applied
AFTERWARDS, by this script, which is the only thing that knows it.

TWO AUDIT MODES, because they ask different things of a director.

  BLIND     the model sees the claim and nothing else. Can it spot a bad
            number on its own? This is the scepticism test, and it is the
            hard one.
  CONTEXT  the model also sees the research notes, the result files and the
            git history -- the same material a real debrief would have. Can
            it find the retraction? This is the job test.

A director that only passes CONTEXT is still useful, but it must not be
described as sceptical.

WHAT THE DEBRIEF ITSELF MUST PRODUCE.

The output is strict JSON, and the next experiment is required to carry:
the hypothesis, the method, the controls, the NULL, the margin threshold,
and the falsifier. An experiment proposal with no null and no falsifier is
rejected by this script before it is written anywhere, on the grounds that
it is the shape of claim this project has had to retract four times.

LIMITATIONS, stated up front rather than discovered later.

  - The audit set is six false claims and four true ones. That is enough to
    catch a director that agrees with everything; it is NOT enough to
    certify one, and the confidence interval on a six-item set is wide.
  - The claims are paraphrases, so the audit tests reasoning about numbers,
    not recall of a document.
  - The context mode includes the research notes, which contain the
    retractions. A strong score there may be partly recall. The blind score
    is the one that separates reading from reasoning.
  - The advisor proposes; it does not decide. Nothing in this repo is
    changed by what it says.

USAGE:
    python experiments/debrief.py --audit --mode blind --model @cf/meta/llama-3.3-70b-instruct-fp8-fast
    python experiments/debrief.py --audit --mode context --model @cf/openai/gpt-oss-120b
    python experiments/debrief.py --debrief --model @cf/qwen/qwen3-30b-a3b-fp8
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = os.environ.get("CF_ACCOUNT_ID", "049ff5e84ecf636b53b162cbb580aae6")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# CONTEXT WINDOWS, because the materials are large and the models are not
# equally large. llama-3.3-70b refuses anything over 24000 tokens with a 413
# and no useful message about which part was too big, so the budget is
# per-model and the materials are cut to fit rather than failing.
MODEL_CONTEXT_TOKENS = {
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast": 24000,
    "@cf/openai/gpt-oss-120b": 128000,
    "@cf/qwen/qwen3-30b-a3b-fp8": 128000,
    "@cf/mistralai/mistral-small-3.1-24b-instruct": 32000,
    "@cf/google/gemma-4-26b-a4b-it": 128000,
}
DEFAULT_CONTEXT_TOKENS = 32000
CHARS_PER_TOKEN = 3.75          # measured on this project's own material


def materials_limit_for(model: str, max_tokens: int = 4000) -> int:
    """How many characters of material this model can actually be given."""
    toks = MODEL_CONTEXT_TOKENS.get(model, DEFAULT_CONTEXT_TOKENS)
    return max(8000, int((toks - max_tokens - 512) * CHARS_PER_TOKEN))


# ---------------------------------------------------------------- the audit set

# Each entry: (claim_text, is_false). `is_false=True` means the director should
# REJECT the claim. The wording is paraphrased away from the exact sentences in
# the research notes, so a model cannot string-match its way to the answer.
AUDIT_SET = [
    ("Reducing the gate's field threshold to 0.30 removed 14.5x of the "
     "camera's frame-to-frame drift, measured as the mean absolute "
     "difference between consecutive frames.", True),

    ("The ASCII character map reliably detects whether a person is moving in "
     "the frame: an occupied room flips 48.19% of its character cells per "
     "frame against 50.00% for an empty room, so the motion is being "
     "measured.", True),

    ("The watcher's memory use grows quadratically with session length, so a "
     "long session will eventually exhaust the device.", True),

    ("Applying a per-frame gain normalisation to the camera's brightness "
     "reduced the frame-to-frame drift from 43.3 counts to 0.81 counts, a "
     "reduction of roughly fifty times.", True),

    ("A local large language model given an ASCII rendering of the room can "
     "label whether the person is speaking, at a useful margin over a blank "
     "control.", True),

    ("31.8% of the frames the watcher processes are never actually measured "
     "by the speech detector, because the gate returns early.", True),

    ("Removing each row's median across the columns before differencing "
     "frames takes an empty room's character-cell flip rate from 50.00% down "
     "to 0.01%, while leaving the occupied room's column structure "
     "unchanged.", False),

    ("A conventional teacher model trained on motion maps scored 1.000 on a "
     "temporal split of one session and 0.000 when the test session was "
     "held out, which shows the model learned the recording rather than "
     "speaking.", True),

    ("The room's own acoustic noise floor measured -73.5 dBFS, with 119 "
     "transients detected and zero valid RT60 estimates obtainable from it.",
     False),

    ("Composing row-wise band removal with a temporal median filter of width "
     "3 reduced the occupied motion-map flip rate from 0.0299 to 0.0014, a "
     "21x improvement over row removal alone, while the signal-to-noise "
     "ratio stayed flat at about 1.83.", False),
]

AUDIT_PROMPT = """You are auditing technical claims from an experimental research log.

For each numbered claim, decide whether the claim is SUPPORTED by the evidence
it cites, UNSUPPORTED (the evidence does not establish it, or the numbers
contradict it), or UNCERTAIN (genuinely cannot be judged from what is given).

Judge ONLY from what the claim itself states -- its numbers, its comparison,
its stated control. Do not assume a claim is true because it is detailed, and
do not assume it is false because it is surprising.

Look specifically for:
  - a comparison drawn between quantities measured over different intervals
    or different populations, which makes the ratio meaningless
  - a comparison against a control that is at or near its theoretical maximum
    or minimum, which means the control established nothing
  - an absolute rate stated without a denominator
  - a factor-of-N improvement where the same method applied to both sides of
    the comparison would produce that factor anyway
  - a conclusion that requires the measurement to be independent when the
    stated sample size cannot make it independent

Return STRICT JSON, nothing else:

{"verdicts": [{"index": 0, "verdict": "supported"|"unsupported"|"uncertain", "reason": "one sentence"}]}

CLAIMS:
{claims}
"""

DEBRIEF_PROMPT = """You are the R&D director for a local-first voice and peripheral system
called Communicator. You are given the project's research notes, the machine-
readable results of its experiments, and its git history. Your job is to
produce the debrief that decides what happens next.

The project has a strict evidence culture, and your debrief must obey it:

  - A number is reported with its condition, its interval, and its null. A
    claim without those is not a result.
  - A conclusion must be able to come out the other way. If a proposed
    experiment has no control that could fail, it is not proposed.
  - Accuracy alone is never the headline; the margin over the controls is.
  - Windows from one session are not independent samples. Sessions are.
  - A comparison is only meaningful between quantities measured the same way.
  - When a result was retracted, the retraction stays visible.

You will receive:
  RESEARCH NOTES  - markdown files, including retraction notices
  RESULTS         - machine-readable outputs of the experiments
  GIT HISTORY     - what was committed, and the messages say what changed

Produce STRICT JSON, nothing else, with this shape:

{{
  "state_of_the_evidence": [
    {{"claim": "what the evidence supports", "evidence": "the number(s) and where they come from"}}
  ],
  "unsafe_claims": [
    {{"claim": "something currently believed that the evidence does not support", "why": "the specific reason", "action": "what to do about it"}}
  ],
  "next_experiment": {{
    "title": "short name",
    "hypothesis": "one falsifiable sentence",
    "method": "what to measure and how",
    "controls": ["the control(s), each of which can fail"],
    "null": "the specific result that means there is no effect",
    "margin_threshold": "the number the effect must beat, and what it is measured against",
    "falsifier": "what result would make you abandon this hypothesis"
  }},
  "biggest_risk": "the thing most likely to waste the next week",
  "what_would_change_the_plan": "the observation that would most change the strategy"
}}

MATERIALS:
{materials}
"""


# ------------------------------------------------------------------- plumbing

def _http_json(url: str, body: dict, headers: dict, timeout: float = 240.0):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
    except urllib.error.HTTPError as exc:
        return {"error": f"http{exc.code}",
                "detail": exc.read().decode()[:500],
                "elapsed_s": round(time.perf_counter() - t0, 2)}
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__, "detail": str(exc)[:300],
                "elapsed_s": round(time.perf_counter() - t0, 2)}
    try:
        d = json.loads(raw)
    except Exception:  # noqa: BLE001
        return {"error": "bad_json", "detail": raw[:500],
                "elapsed_s": round(time.perf_counter() - t0, 2)}
    out = d.get("result")
    if out is None:
        return {"error": "no_result", "detail": raw[:500],
                "elapsed_s": round(time.perf_counter() - t0, 2)}
    if isinstance(out, dict):
        ch = out.get("choices") or []
        if ch:
            msg = ch[0].get("message") or {}
            text = msg.get("content")
            # A REASONING MODEL PUTS ITS ANSWER IN `content` ONLY IF THERE IS
            # ROOM LEFT AFTER THE REASONING. qwen3-30b with max_tokens=60
            # returns content=null and a 283-character reasoning field, and
            # with max_tokens=300 the same request returns content="\n\nOK".
            # The budget went into thinking, not into the answer. This is the
            # same failure as ollama's num_predict, and it is SILENT: the
            # model looks like it refused.
            if not text:
                text = msg.get("reasoning_content") or msg.get("reasoning") or ""
                out["content_empty"] = True
            out = {"text": text}
        elif "response" in out:
            out = {"text": out["response"]}
        elif "text" in out:
            out = {"text": out["text"]}
    if not isinstance(out, dict):
        out = {"text": str(out)}
    out.setdefault("text", "")
    out["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return out


_AGREED: set = set()


def cf_agree(model: str) -> bool:
    """Meta's licence gate. A SHAPE error means the endpoint has no gate."""
    if model in _AGREED:
        return True
    r = _http_json(
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/{model}",
        # A small budget: the gate only has to be satisfied, and an unbounded
        # "agree" call makes qwen3 reason for sixteen seconds and generate a
        # paragraph about conflicts.
        {"prompt": "agree", "max_tokens": 8},
        {"Authorization": f"Bearer {os.environ['CF_TOKEN']}",
         "Content-Type": "application/json"}, timeout=90.0)
    if "error" not in r:
        _AGREED.add(model)
        return True
    detail = (r.get("detail") or "").lower()
    if any(k in detail for k in ("oneof", "required properties",
                                 "type mismatch", "not met")):
        _AGREED.add(model)
        return True
    return False


def cloud_llm(model: str, prompt: str, max_tokens: int = 3000,
              temperature: float = 0.0) -> dict:
    """A text LLM on Workers AI. Returns {"text": ..., "elapsed_s": ...}.

    The retry is for REASONING MODELS. qwen3 spends its budget thinking, and
    if the thinking uses it all then `content` comes back null and the model
    looks like it refused to answer. Rather than record that as a refusal, the
    call is repeated with more room.
    """
    if not cf_agree(model):
        return {"error": "agreement_failed", "detail": "", "elapsed_s": 0.0}
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/{model}"
    hdr = {"Authorization": f"Bearer {os.environ['CF_TOKEN']}",
           "Content-Type": "application/json"}
    r = _http_json(url, body, hdr)
    if r.get("content_empty") and not _is_context_error(r):
        body["max_tokens"] = max_tokens * 4
        r2 = _http_json(url, body, hdr)
        if (r2.get("text") or "").strip():
            r2["retried_with_more_tokens"] = True
            return r2
    return r


def ollama_llm(model: str, prompt: str, max_tokens: int = 3000) -> dict:
    """A local model through ollama. `think: false` is REQUIRED for qwen3 --
    a reasoning model puts its output in a `thinking` field and burns the
    whole num_predict budget on it."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "think": False,
            "options": {"temperature": 0.0, "num_predict": max_tokens}}
    r = _http_json(f"{OLLAMA}/api/chat", body, {"Content-Type": "application/json"},
                   timeout=300.0)
    if "error" in r:
        return r
    return {"text": ((r.get("message") or {}).get("content") or ""),
            "elapsed_s": r.get("elapsed_s")}


def call_model(model: str, prompt: str, max_tokens: int = 3000) -> dict:
    if model.startswith("ollama:") or model.startswith("local:"):
        return ollama_llm(model.split(":", 1)[1], prompt, max_tokens)
    if not os.environ.get("CF_TOKEN"):
        return {"error": "no_token", "detail": "source ~/.config/cloudflare/env"}
    return cloud_llm(model, prompt, max_tokens)


# ------------------------------------------------------------------ json pull

def extract_json(text: str):
    """Pull a JSON object out of a model response.

    Models wrap JSON in prose, in fences, or both, even when told not to.
    The braces are found by SCANNING for the outermost balanced pair, so a
    brace inside a string literal does not end the object early.
    """
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    start = t.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except Exception:  # noqa: BLE001
                        break
        start = t.find("{", start + 1)
    return None


# ------------------------------------------------------------------ materials

def collect_materials(limit: int = 140000) -> str:
    """The research notes, the result files and the git history.

    ROUND-ROBIN, AND THE FIRST VERSION OF THIS WAS A DEFECT THAT CHANGED
    THE RESULT.

    It filled a character budget with notes in alphabetical order and stopped.
    With 30 notes totalling 372 kB, that meant SIX notes got in -- and the
    ones that made it were the alphabetically early ones, while
    ROW-BAND-REMOVAL.md, EMPTY-ROOM-CONTROL.md, TEACHER-PIPELINE.md,
    WATCHER-COST.md, HOW-WE-KEEP-FOOLING-OURSELVES.md and the retraction
    notes were all cut off.

    gpt-oss-120b then accused two TRUE claims of being unsupported, with
    specific and entirely reasonable reasons: "no cited evidence provides the
    noise floor" (it was not there) and "row-median removal reduces
    empty-room saturation to 2.99%, not 0.01%" (2.99% is the row-only number
    in COMBINED-FILTER.md, a DIFFERENT metric, and the note that says 0.01%
    was not there). Both verdicts were CORRECT GIVEN THE EVIDENCE SUPPLIED.

    That is the finding, and it is the whole story of this tool: a director's
    quality is a function of the evidence you hand it, and a starved evidence
    base produces a confident, specific, WRONG verdict. Which is exactly what
    a human reviewer does too -- so the tool measures a thing worth measuring,
    and it measures the thing it was given.

    So every note gets an EQUAL SHARE of the budget, and each note
    contributes its HEAD and its TAIL -- the retraction notices live at both
    ends of these notes, one at the top and one at the bottom, and a
    head-only slice loses half of them.
    """
    notes = sorted((ROOT / "docs" / "research").glob("*.md"))
    res = sorted((ROOT / "experiments").glob("*.json"))
    parts: list[str] = []

    # -- the notes: an equal share each, and every NUMBER kept ------------
    #
    # The head-and-tail slice was not enough either. The facts the audit
    # needs -- "-73.5 dBFS", "0.01%", the comparison tables -- live in the
    # MIDDLE of these notes, in tables, and a contiguous slice walks straight
    # past them. So each note contributes its head, its tail, and every line
    # that contains a digit, in original order, with the gaps marked.
    note_pool = int(limit * 0.72)
    per = max(600, note_pool // max(1, len(notes)))
    for p in notes:
        try:
            body = p.read_text()
        except Exception:  # noqa: BLE001
            continue
        lines = body.splitlines()
        if len(body) <= per:
            parts.append(f"--- NOTE: {p.name} ---\n{body}")
            continue
        head_lines = max(3, int(len(lines) * 0.18))
        keep = set(range(head_lines))
        keep |= set(range(len(lines) - head_lines, len(lines)))
        for i, ln in enumerate(lines):
            if any(ch.isdigit() for ch in ln):
                keep.add(i)
        keep = sorted(keep)
        out, prev = [], -2
        for i in keep:
            if i != prev + 1:
                out.append(f"[... {i - prev - 1} lines elided ...]")
            out.append(lines[i])
            prev = i
        text = "\n".join(out)
        if len(text) > per:
            text = text[:per] + "\n[... truncated ...]"
        parts.append(f"--- NOTE: {p.name} ---\n{text}")

    # -- the machine-readable results, whole where they fit --------------
    res_pool = int(limit * 0.24)
    per_res = max(300, res_pool // max(1, len(res)))
    for p in res:
        try:
            body = p.read_text()
        except Exception:  # noqa: BLE001
            continue
        parts.append(f"--- RESULT: {p.name} ---\n{body[:per_res]}")

    try:
        log = os.popen(f"cd {ROOT} && git log --oneline -40").read()
        parts.append(f"--- GIT HISTORY ---\n{log}")
    except Exception:  # noqa: BLE001
        pass
    return "\n\n".join(parts)


# ---------------------------------------------------------------------- audit

def audit_pairs(seed: int = 20261019):
    """The audit set, shuffled with a fixed seed. Deterministic."""
    pairs = list(AUDIT_SET)
    random.Random(seed).shuffle(pairs)
    return pairs


def build_audit_prompt(mode: str, pairs, materials_limit: int | None = None):
    """The audit prompt for a given material budget.

    A pure function of (mode, pairs, materials_limit) and the files on disk.
    That purity is the point: it is what lets the SAME prompt be sent N times
    and the director's run-to-run repeatability be measured against it.
    Returns (prompt, materials_chars).
    """
    claims = "\n".join(f"{n}. {c}" for n, (c, _) in enumerate(pairs))
    if mode == "context":
        lim = materials_limit if materials_limit else FIT_LADDER[-1]
        return (AUDIT_PROMPT.replace(
            "{claims}",
            claims + "\n\n=== RESEARCH NOTES AND RESULTS (for reference) ===\n"
            + collect_materials(limit=lim)), lim)
    return AUDIT_PROMPT.replace("{claims}", claims), 0


def score_audit(pairs, obj, model: str = "", mode: str = "", lim: int = 0,
                elapsed=None, raw: str = "") -> dict:
    """Turn a parsed verdict object into the scores.

    A pure function of (pairs, obj): the same verdicts always give the same
    headline. Whatever varies between runs of the director is the MODEL's
    variation, not the scoring's.
    """
    by_index = {}
    for v in obj.get("verdicts") or []:
        try:
            by_index[int(v.get("index"))] = v
        except Exception:  # noqa: BLE001
            continue

    rows = []
    for n, (claim, is_false) in enumerate(pairs):
        v = by_index.get(n) or {}
        verdict = str(v.get("verdict", "")).strip().lower()
        if verdict not in ("supported", "unsupported", "uncertain"):
            verdict = "unparseable"
        rejected = verdict == "unsupported"
        correct = (rejected == is_false) if verdict != "unparseable" else False
        rows.append({"index": n, "is_false_claim": is_false, "verdict": verdict,
                     "rejected": rejected, "correct": correct,
                     "reason": str(v.get("reason", ""))[:400],
                     "claim": claim})

    n_false = sum(1 for r_ in rows if r_["is_false_claim"])
    n_true = len(rows) - n_false
    det = sum(1 for r_ in rows if r_["is_false_claim"] and r_["correct"])
    fp = sum(1 for r_ in rows if (not r_["is_false_claim"]) and r_["rejected"])
    tp = sum(1 for r_ in rows if (not r_["is_false_claim"]) and
             r_["verdict"] == "supported")
    unc = sum(1 for r_ in rows if r_["verdict"] == "uncertain")
    sens = det / n_false if n_false else None
    spec = tp / n_true if n_true else None
    # THE HEADLINE, and the first version of it was broken.
    #
    # It was sens - spec, which scores an ALWAYS-REJECT director at +1: reject
    # every claim and you catch all six false ones and accuse all four true
    # ones, giving 1.0 - 0.0 = +1.0, which looks like a perfect advisor. A
    # metric that cannot fail is not a metric, and this one could not fail --
    # it could only ever reward one of the two degenerate behaviours.
    #
    # sens + spec - 1 (Youden's J) puts BOTH degenerate directors at 0:
    #   always-accept:  0.0 + 1.0 - 1 = 0.0
    #   always-reject:  1.0 + 0.0 - 1 = 0.0
    #   perfect:        1.0 + 1.0 - 1 = 1.0
    score = None if (sens is None or spec is None) else sens + spec - 1.0
    # THE LENIENT VARIANT, because "uncertain" is not the same as "wrong".
    #
    # A director that answers "uncertain" about a true claim has NOT accused
    # it -- it has declined to endorse it. qwen3-30b marks five of ten claims
    # uncertain, which the strict metric scores as a failure of specificity
    # even though it made zero false accusations. The two numbers are
    # reported together because they answer different questions:
    #
    #   STRICT   did it actually catch the bad claim, and actually endorse
    #            the good one?
    #   LENIENT  did it avoid PROPAGATING the bad claim, and avoid ACCUSING
    #            the good one?  For a safety-critical reviewer, "I would not
    #            sign that off" is a good enough outcome.
    not_endorsed = sum(1 for x in rows if x["is_false_claim"]
                       and x["verdict"] != "supported")
    not_accused = sum(1 for x in rows if (not x["is_false_claim"])
                       and x["verdict"] != "unsupported")
    sens_lenient = not_endorsed / n_false if n_false else None
    spec_lenient = not_accused / n_true if n_true else None
    score_lenient = (None if (sens_lenient is None or spec_lenient is None)
                     else sens_lenient + spec_lenient - 1.0)
    return {"model": model, "mode": mode,
            "materials_chars": lim,
            "sensitivity_rejected_false_claims": sens,
            "specificity_kept_true_claims": spec,
            "headline_sens_plus_spec_minus_1": score,
            "sensitivity_not_endorsing_false_claims": sens_lenient,
            "specificity_not_accusing_true_claims": spec_lenient,
            "headline_lenient": score_lenient,
            "false_positives": fp, "true_positives": tp,
            "uncertain": unc,
            "detections": det, "n_false": n_false, "n_true": n_true,
            "unparseable": sum(1 for x in rows if x["verdict"] == "unparseable"),
            "elapsed_s": elapsed, "rows": rows,
            "raw": raw[:1500]}


def run_audit(model: str, mode: str, seed: int = 20261019,
              materials_limit: int | None = None) -> dict:
    pairs = audit_pairs(seed)

    def _mk(lim):
        return build_audit_prompt(mode, pairs, lim)[0]

    if materials_limit:
        r = call_model(model, _mk(materials_limit), 3000)
        lim = materials_limit
    else:
        r, lim = call_with_fit(model, _mk, 3000)
    if r.get("error"):
        return {"model": model, "mode": mode, "error": r.get("error"),
                "detail": r.get("detail"), "elapsed_s": r.get("elapsed_s")}

    obj = extract_json(r.get("text") or "")
    if not obj or "verdicts" not in obj:
        return {"model": model, "mode": mode, "error": "unparseable",
                "raw": (r.get("text") or "")[:2000],
                "elapsed_s": r.get("elapsed_s")}

    return score_audit(pairs, obj, model, mode, lim, r.get("elapsed_s"),
                       r.get("text") or "")


# -------------------------------------------------------------------- debrief

def run_debrief(model: str, out_path: Path | None = None) -> dict:
    def _mk(lim):
        return DEBRIEF_PROMPT.replace("{materials}", collect_materials(limit=lim))

    r, lim = call_with_fit(model, _mk, 4000)
    if r.get("error"):
        return {"model": model, "error": r.get("error"), "detail": r.get("detail")}
    obj = extract_json(r.get("text") or "")
    res = {"model": model, "elapsed_s": r.get("elapsed_s"),
           "parsed": obj is not None,
           "materials_chars": lim,
           "debrief": obj, "raw": (r.get("text") or "")[:4000]}
    if obj and isinstance(obj.get("next_experiment"), dict):
        ne = obj["next_experiment"]
        missing = [k for k in ("hypothesis", "method", "controls", "null",
                               "falsifier") if not ne.get(k)]
        res["missing_required_fields"] = missing
        res["acceptable"] = not missing
    else:
        res["missing_required_fields"] = ["next_experiment"]
        res["acceptable"] = False
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2))
    return res


def rescore(path: Path) -> dict:
    """Recompute the headline from a saved audit file, without calling a model.

    A metric that changed after the measurements were taken has to be
    re-applied to the SAME measurements, not to a fresh run -- otherwise the
    two numbers are not comparable and the change is invisible.
    """
    d = json.loads(path.read_text())
    out = []
    for r in d.get("results", []):
        if r.get("error"):
            continue
        rows = r.get("rows") or []
        n_false = sum(1 for x in rows if x["is_false_claim"])
        n_true = len(rows) - n_false
        det = sum(1 for x in rows if x["is_false_claim"] and x["correct"])
        tp = sum(1 for x in rows if (not x["is_false_claim"]) and
                 x["verdict"] == "supported")
        fp = sum(1 for x in rows if (not x["is_false_claim"]) and x["rejected"])
        unc = sum(1 for x in rows if x["verdict"] == "uncertain")
        sens = det / n_false if n_false else None
        spec = tp / n_true if n_true else None
        score = None if (sens is None or spec is None) else sens + spec - 1.0
        not_endorsed = sum(1 for x in rows if x["is_false_claim"]
                           and x["verdict"] != "supported")
        not_accused = sum(1 for x in rows if (not x["is_false_claim"])
                           and x["verdict"] != "unsupported")
        sens_lenient = not_endorsed / n_false if n_false else None
        spec_lenient = not_accused / n_true if n_true else None
        score_lenient = (None if (sens_lenient is None or spec_lenient is None)
                         else sens_lenient + spec_lenient - 1.0)
        out.append({"model": r.get("model"), "mode": r.get("mode"),
                    "materials_chars": r.get("materials_chars"),
                    "sensitivity": sens, "specificity": spec,
                    "headline_sens_plus_spec_minus_1": score,
                    "sensitivity_not_endorsing": sens_lenient,
                    "specificity_not_accusing": spec_lenient,
                    "headline_lenient": score_lenient,
                    "false_accusations": fp, "uncertain": unc,
                    "n_false": n_false, "n_true": n_true})
    return {"mode": d.get("mode"), "results": out}


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[:70]


def _is_context_error(r: dict) -> bool:
    """Did this call fail because the material did not fit?

    The failure modes differ by model and are unhelpful: llama-3.3-70b says
    "maximum context length is 24000 tokens"; qwen3-30b says "estimated number
    of input and maximum output tokens (95806) exceeded this model context
    window limit (32768)". And the character-per-token ratio is not stable
    across tokenizers -- the same 139 kB of material was estimated at 37 046
    tokens by one model and 95 806 by another.
    """
    d = ((r.get("detail") or "") + " " + str(r.get("error") or "")).lower()
    return any(k in d for k in ("context length", "context window",
                                "exceeded this model", "http413",
                                "please reduce the length"))


# The ladder of material sizes to try, biggest first. Each step down is
# roughly a halving, and the bottom of it is small enough to fit anywhere.
FIT_LADDER = (140000, 90000, 60000, 40000, 28000, 18000, 12000, 8000)


def call_with_fit(model: str, make_prompt, max_tokens: int):
    """Call the model, shrinking the material until it fits.

    Returns (result, materials_chars_used). The size that worked is
    recorded, because "how much evidence can this model actually read" is
    itself a finding: a director given 35 kB of evidence behaves differently
    from one given 140 kB, and that difference is not the model's fault.
    """
    last = {"error": "not_attempted"}
    for lim in FIT_LADDER:
        prompt = make_prompt(lim)
        r = call_model(model, prompt, max_tokens)
        if not _is_context_error(r):
            return r, lim
        last = r
    return last, None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--debrief", action="store_true")
    ap.add_argument("--mode", choices=["blind", "context"], default="blind")
    ap.add_argument("--model", default="@cf/meta/llama-3.3-70b-instruct-fp8-fast")
    ap.add_argument("--models", default=None,
                    help="comma-separated list; each is audited separately")
    ap.add_argument("--materials-limit", type=int, default=0,
                    help="force the material budget in characters, instead of "
                         "letting the ladder find the largest that fits. Used "
                         "to measure whether MORE evidence makes the director "
                         "better, with the model held constant.")
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    ap.add_argument("--rescore", action="store_true",
                    help="recompute the headline from the saved audit files, "
                         "without calling any model")
    a = ap.parse_args()

    if a.rescore:
        for p in sorted(Path(a.out).glob("debrief_audit_*.json")):
            r = rescore(p)
            print(f"\n=== {p.name} (mode={r['mode']}) ===")
            for x in r["results"]:
                print(f"  {x['model']}")
                print(f"     sensitivity {x['sensitivity']}   "
                      f"specificity {x['specificity']}   "
                      f"false accusations {x['false_accusations']}/{x['n_true']}   "
                      f"uncertain {x['uncertain']}")
                print(f"     HEADLINE (sens + spec - 1) = "
                      f"{x['headline_sens_plus_spec_minus_1']}")
        return

    models = ([m.strip() for m in a.models.split(",") if m.strip()]
              if a.models else [a.model])

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if a.audit:
        results = []
        for m in models:
            print(f"\n=== AUDIT [{a.mode}] {m} ===")
            r = run_audit(m, a.mode, materials_limit=a.materials_limit or None)
            if r.get("error"):
                print(f"  ERROR {r['error']}: {str(r.get('detail'))[:300]}")
            else:
                print(f"  sensitivity (rejected the false claims): "
                      f"{r['sensitivity_rejected_false_claims']}")
                print(f"  specificity (kept the true claims):      "
                      f"{r['specificity_kept_true_claims']}")
                print(f"  false accusations: {r['false_positives']}/{r['n_true']}   "
                      f"uncertain: {r['uncertain']}")
                print(f"  HEADLINE (sens + spec - 1) = "
                      f"{r['headline_sens_plus_spec_minus_1']}   "
                      f"lenient = {r.get('headline_lenient')}")
                for row in r["rows"]:
                    mark = "ok " if row["correct"] else "MISS"
                    print(f"    {mark} [{'FALSE' if row['is_false_claim'] else 'TRUE '}] "
                          f"{row['verdict']:12s} {row['claim'][:70]}")
            results.append(r)
        p = outdir / f"debrief_audit_{a.mode}_{'_'.join(slug(m) for m in models)}.json"
        p.write_text(json.dumps({"mode": a.mode, "results": results}, indent=2))
        print(f"\nwrote {p}")

    if a.debrief:
        for m in models:
            print(f"\n=== DEBRIEF {m} ===")
            r = run_debrief(m)
            if r.get("error"):
                print(f"  ERROR {r['error']}: {str(r.get('detail'))[:300]}")
                continue
            d = r.get("debrief") or {}
            for c in (d.get("state_of_the_evidence") or [])[:6]:
                print(f"  SUPPORTED: {str(c.get('claim'))[:110]}")
            for c in (d.get("unsafe_claims") or [])[:6]:
                print(f"  UNSAFE   : {str(c.get('claim'))[:110]}")
            ne = d.get("next_experiment") or {}
            print(f"  NEXT     : {ne.get('title')}")
            print(f"             null: {str(ne.get('null'))[:100]}")
            print(f"             falsifier: {str(ne.get('falsifier'))[:100]}")
            print(f"  acceptable (hypothesis/method/controls/null/falsifier all "
                  f"present): {r.get('acceptable')}"
                  f" missing={r.get('missing_required_fields')}")
            p = outdir / f"debrief_{slug(m)}.json"
            p.write_text(json.dumps(r, indent=2))
            print(f"  wrote {p}")


if __name__ == "__main__":
    main()
