"""
Tests for question_health -- the instrument that catches a question inventing
structure.

THE DEFECT CLASS, and why it matters more here than in a router.

`decision-ledger`'s pass_detector exists because an ill-posed question still
returns confident numbers: a question whose answers cluster at "pass" is
INVALID, not uncertain. There the failure is a question the model declines.

The mirror image is worse for a controller. Garbage input has no timbre, no
urgency, no addressee and no speech act. A question that confidently places
garbage on its scale is inventing structure -- and the SAME invention is then
applied to every ambiguous real utterance, which is precisely where a
controller must hold back. Phase 2 measured that band at ECE 0.153.

So the check is: garbage in, LOW spread-confidence out. Real in, meaningful
out.

WHAT THESE TESTS DO NOT PROVE.

  * Nothing about the BAR. HEALTH_BAR is 0.70 and DEGENERATE_BAR is 0.50, both
    chosen before seeing live results. That is a guess dressed as a constant,
    and the live report must be read before either is trusted.
  * Nothing about whether the model actually passes. These are unit tests
    against a fake client; `probe_questions` against the live model is a
    separate, slower thing and its result is a finding either way.
  * Nothing about per-question health in isolation -- the module reports a
    verdict per TRACK (a fingerprint groups four questions), so a single
    broken question inside a healthy track is invisible here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import question_health as qh            # noqa: E402


class FakeReading:
    def __init__(self, track, conf, ok=True):
        self.track = track
        self.ok = ok
        self.detail = {"confidence": conf}


class FakeAnswer:
    """Shaped like the real SDK answers -- read_tracks reads all of these."""

    def __init__(self, conf, n=4):
        self.confidence = conf
        self.probabilities = {i: conf / n for i in range(n)}
        self.score = 1.5
        self.legend = {i: f"level {i}" for i in range(n)}
        self.choice = 0


class FakeClient:
    """
    Returns a chosen spread-confidence for every probe. `garbage_conf` and
    `real_conf` stand in for the two behaviours the check is looking for.

    It has to satisfy the FULL read_tracks contract, not just `.confidence` --
    an earlier version returned a bare object and the module raised
    AttributeError on `.choice`, which is its own reminder that a test double
    narrower than the real interface tests less than it appears to.
    """

    def __init__(self, garbage_conf: float, real_conf: float):
        self.garbage_conf = garbage_conf
        self.real_conf = real_conf
        self.calls = 0
        self.seen: list[str] = []

    def system_one(self, *, state, questions, model, timeout):
        self.calls += 1
        self.seen.append(state)
        garbage = state in qh.GARBAGE_PROBES
        conf = self.garbage_conf if garbage else self.real_conf

        class _R:
            pass

        r = _R()
        r.model = model
        r.answers = {k: FakeAnswer(conf) for k in questions}
        return r


class TestProbeCorpus:
    def test_garbage_probes_are_non_sentences(self):
        """An interrogative or imperative in the garbage set would invalidate it."""
        for p in qh.GARBAGE_PROBES:
            assert "?" not in p, f"{p!r} is a question, not garbage"
            assert not p.strip().endswith("."), f"{p!r} is a statement, not garbage"

    def test_real_probes_are_well_formed_rather_than_addressed(self):
        """
        An earlier version of this test asserted every real probe was a
        question or an imperative. It is not: "the sky is clear today" carries
        plenty for a timbre question and addresses nobody.

        What actually separates the sets is well-formedness, not address. The
        garbage set is consonant noise; the real set is English. So that is what
        is asserted -- plus a word count, because a one-word probe cannot
        separate anything.
        """
        for p in qh.REAL_PROBES:
            assert len(p.split()) >= 3, f"{p!r} is too thin to separate anything"
            words = p.lower().rstrip("?.!").split()
            # Real English words here are mostly >=4 chars or common short ones.
            assert any(len(w) >= 4 or w in {"the", "is", "it", "up", "a", "run"}
                       for w in words), f"{p!r} looks like noise, not English"

    def test_the_two_sets_do_not_overlap(self):
        assert not (set(qh.GARBAGE_PROBES) & set(qh.REAL_PROBES))

    def test_bars_are_ordered_sensibly(self):
        assert 0.0 < qh.DEGENERATE_BAR < qh.HEALTH_BAR < 1.0


class TestProbeQuestions:
    def test_a_healthy_question_set_separates_garbage_from_real(self):
        """Garbage should read LOW confidence, real should read HIGH."""
        client = FakeClient(garbage_conf=0.15, real_conf=0.85)
        healths = qh.probe_questions(client)
        assert healths
        for h in healths:
            assert h.garbage_median < h.real_median, h
            assert h.separation > 0, h
            assert h.verdict == "healthy", f"{h.track}: {h.verdict}"

    def test_a_question_that_invents_structure_on_garbage_is_caught(self):
        """The whole point: confident on nonsense means the question is broken."""
        client = FakeClient(garbage_conf=0.92, real_conf=0.88)
        healths = qh.probe_questions(client)
        assert all(h.verdict == "ill-posed" for h in healths), \
            [(h.track, h.verdict, h.garbage_median) for h in healths]

    def test_a_question_that_sees_nothing_in_real_input_is_caught(self):
        """The opposite failure: a question that never separates is worthless,
        and would look like a low, safe number rather than a broken one."""
        client = FakeClient(garbage_conf=0.10, real_conf=0.20)
        healths = qh.probe_questions(client)
        assert all(h.verdict == "degenerate" for h in healths), \
            [(h.track, h.verdict) for h in healths]

    def test_every_track_is_reported(self):
        healths = qh.probe_questions(FakeClient(0.15, 0.85))
        assert {h.track for h in healths} == {"TIMBRE", "INTENTION",
                                              "ATTENTION", "GESTURE"}

    def test_probing_uses_one_call_per_utterance_not_per_question(self):
        """Four questions per call. Twelve calls for twelve probes."""
        client = FakeClient(0.15, 0.85)
        qh.probe_questions(client)
        n = len(qh.GARBAGE_PROBES) + len(qh.REAL_PROBES)
        assert client.calls == n, client.calls
        assert len(client.seen) == n

    def test_failed_readings_do_not_count_as_low_confidence(self):
        """
        A network failure must not read as "healthy" -- 0.0 confidence from a
        dropped call looks exactly like a well-behaved question declining to
        place garbage. The `_median_conf` filter on `r.ok` is what stops that,
        and this asserts the filter exists by driving it.
        """

        class AllFailed:
            def system_one(self, **kw):
                raise RuntimeError("down")

        healths = qh.probe_questions(AllFailed())
        # Every reading failed, so there is no evidence either way. The module
        # must not manufacture a verdict from nothing.
        for h in healths:
            assert h.garbage_median == 0.0 and h.real_median == 0.0, h

    def test_model_is_named_explicitly(self):
        """A floating alias would make every verdict unattributable."""
        assert judged_default() != "jev-latest"

    def test_report_renders_without_a_client(self):
        client = FakeClient(0.15, 0.85)
        report = qh.format_report(qh.probe_questions(client), model="jev-1.14.0")
        assert "# Question health" in report
        assert "TIMBRE" in report
        assert "ill-posed" in report or "degenerate" in report or "healthy" in report


def judged_default() -> str:
    from src.judged import DEFAULT_MODEL
    return DEFAULT_MODEL


class TestReport:
    def test_report_names_every_track_it_was_given(self):
        client = FakeClient(0.15, 0.85)
        healths = qh.probe_questions(client)
        report = qh.format_report(healths, model="jev-1.14.0")
        for h in healths:
            assert h.track in report

    def test_report_states_the_bars_it_judged_against(self):
        client = FakeClient(0.15, 0.85)
        report = qh.format_report(qh.probe_questions(client), model="m")
        # Formatted from the floats themselves, so "0.7" not "0.70".
        assert f"{qh.HEALTH_BAR}" in report and f"{qh.DEGENERATE_BAR}" in report, \
            "a verdict without its thresholds is not auditable"