"""
test_debrief_material_pool.py -- the evidence the director is given must not
lose the retraction notices.

WHY THESE TESTS EXIST.

`experiments/debrief.py` audits an LLM "director" against ten claims drawn
from this project's own retracted history. The auditor's docstring already
records the first version of the defect: the material pool was filled in
alphabetical order and stopped, so six notes got in and the retraction
notes were all cut off, and the director then confidently accused two true
claims of being unsupported -- correctly, given the evidence it was given.

The pool was rewritten to give every note an EQUAL SHARE and to keep each
note's HEAD and its TAIL, because the retraction notices live at both ends
of these notes.

AND THEN THE LAST LINE OF THAT REWRITE UNDID IT. The keep-set was built
correctly and the assembled text was then cut with `text[:per]` -- from the
FRONT. Every note whose slice exceeded its share therefore lost its TAIL,
which is where half the retraction notices are. The design was right and
one line of code undid it, invisibly: the prompt still looked reasonable
and the verdicts were still confidently wrong.

THE MEASUREMENT, ON THE REAL NOTES AT THE COMMITTED BUDGET:

    BEFORE the fix:   2 of 50 notes kept their tail   (48 lost it)
    AFTER  the fix:  50 of 50 notes keep their tail, and 50 of 50 keep
                      their head

These tests pin that behaviour, and pin that it is a BEHAVIOUR and not a
promise: the pool is measured by `collect_materials_stats()`, and a
regression that loses the tails again fails here.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import debrief  # noqa: E402


def _make_repo(tmp_path: Path, n_notes: int = 6, body_lines: int = 400) -> Path:
    """A fake repo with long notes that each have a distinctive head and
    tail, so a slice that keeps one end and drops the other is visible."""
    docs = tmp_path / "docs" / "research"
    docs.mkdir(parents=True)
    (tmp_path / "experiments").mkdir(parents=True)
    for i in range(n_notes):
        head = f"HEAD-MARKER-{i}"
        tail = f"TAIL-MARKER-{i}"
        middle = [f"filler line {j} with the number {j}" for j in range(body_lines)]
        (docs / f"note{i}.md").write_text(
            "\n".join([head, ""] + middle + [tail, ""]))
    return tmp_path


@pytest.fixture()
def fake_repo(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    monkeypatch.setattr(debrief, "ROOT", repo)
    return repo


class TestMaterialPoolKeepsBothEnds:
    def test_every_note_keeps_its_tail(self, fake_repo):
        """The retraction notices live in the tails. A pool that keeps only
        the head hides half of them, and the director then accuses a true
        claim of being unsupported -- correctly, given the evidence."""
        stats = debrief.collect_materials_stats(140000)
        assert stats["notes_total"] == 6
        assert stats["notes_keeping_tail"] == stats["notes_total"], (
            f"only {stats['notes_keeping_tail']} of {stats['notes_total']} "
            "notes kept their tail")

    def test_every_note_keeps_its_head(self, fake_repo):
        stats = debrief.collect_materials_stats(140000)
        assert stats["notes_keeping_head"] == stats["notes_total"]

    def test_the_markers_are_actually_in_the_prompt(self, fake_repo):
        """The stats are computed from the text; this checks the text
        itself, so a stats bug cannot make a broken pool look healthy."""
        text = debrief.collect_materials(140000)
        for i in range(6):
            assert f"HEAD-MARKER-{i}" in text, f"head of note{i} missing"
            assert f"TAIL-MARKER-{i}" in text, f"tail of note{i} missing"

    def test_both_ends_survive_a_budget_that_cannot_hold_the_whole_note(self,
                                                                        fake_repo):
        """A budget so small that the head and the tail cannot both fit
        whole still has to deliver both ends."""
        text = debrief.collect_materials(1200)
        assert "HEAD-MARKER-0" in text
        assert "TAIL-MARKER-0" in text


class TestTheOldBehaviourWouldFail:
    def test_front_truncation_loses_the_tail(self, fake_repo):
        """The bug, reproduced.

        This is the OLD assembly: build the keep-set, then cut the text from
        the front. It is kept in the test file so the fix cannot be
        reverted by someone who "simplifies" it back, and so the claim
        "the front-truncation loses the tails" is a measurement rather
        than an assertion about the past.
        """
        notes = sorted((fake_repo / "docs" / "research").glob("*.md"))
        # A budget small enough that the assembled slice cannot fit, which
        # is the only situation in which the front-truncation can bite.
        limit = 2000
        per = max(600, int(limit * 0.72) // max(1, len(notes)))
        tails_kept = 0
        for p in notes:
            body = p.read_text()
            lines = body.splitlines()
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
                text = text[:per] + "\n[... truncated ...]"   # THE OLD BUG
            if text.rstrip().endswith(lines[keep[-1]]):
                tails_kept += 1
        assert tails_kept < len(notes), (
            "the old front-truncation kept every tail, so this test no "
            "longer demonstrates the defect it is pinning")


class TestTheRealRepo:
    def test_the_real_pool_keeps_every_tail(self):
        """The real notes, at the committed budget. If the corpus grows and
        the pool stops keeping the tails, this fails."""
        stats = debrief.collect_materials_stats(140000)
        assert stats["notes_total"] >= 40, (
            f"only {stats['notes_total']} notes found -- did the path change?")
        assert stats["notes_keeping_tail"] == stats["notes_total"], (
            f"{stats['notes_total'] - stats['notes_keeping_tail']} of "
            f"{stats['notes_total']} notes lost their tail")

    def test_the_stats_are_internally_consistent(self):
        stats = debrief.collect_materials_stats(140000)
        assert stats["notes_kept_whole"] + stats["notes_truncated"] == \
            stats["notes_total"]
        assert stats["notes_keeping_both_ends"] <= stats["notes_keeping_tail"]
        assert stats["notes_keeping_both_ends"] <= stats["notes_keeping_head"]

    def test_a_retraction_notice_survives_to_the_prompt(self):
        """The concrete thing the fix is FOR. HOW-WE-KEEP-FOOLING-OURSELVES.md
        opens with a dated correction block, and the repo's retraction
        notices live at the tops and the bottoms of the notes. If the pool
        keeps neither, the director is blind to the repo's own history and
        there is no point running it."""
        text = debrief.collect_materials(140000)
        assert "CORRECTION" in text, (
            "no dated correction block reached the prompt -- the director "
            "cannot see the repo's own retractions")
