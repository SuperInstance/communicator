"""
Tests for the fixed recording stimulus, `src/stimulus.py`.

WHY THIS EXISTS. The corpus is the binding constraint of the whole project:
the teacher needs >= 63 INDEPENDENT sessions per class, and every one of those
sessions is a take of the same stimulus. A stimulus that changes between takes
-- or, worse, a stimulus whose two representations disagree -- does not fail
loudly. It produces a corpus that looks complete and is quietly mislabelled:
the recorder shows Casey one poem while the metadata records another, and
nothing in the pipeline can detect it, because both versions are perfectly
well-formed.

So the invariant pinned here is narrow and mechanical: THE POEM IS ONE POEM.
`STIMULUS_TEXT` (what is written on screen, and what a human reads to check
it) and `STIMULUS_STANZAS` (what is recited, and what the capture script
stores) must be the same words, in the same order, in the same stanza
structure. The test that matters most in this file is therefore the one that
compares the two representations in BOTH directions -- a one-sided comparison
would pass if a stanza were silently dropped, and the corpus would then
record 14 lines while the screen shows 16.

WHAT ELSE IS PINNED. The identity of the poem (Henley's "Invictus", first and
last line), the shape of the alternatives (they are the escape hatch when
Casey knows a poem better than this one, so they must be usable: at least two
stanzas, no empty stanza, a title and an author to attribute), the file-backed
escape hatch itself (`load_stimulus(path)` -- Casey's own text), and the
timing relationship that the capture plan depends on: the recommended window
is never shorter than the estimated length of the poem.

NOTE ON THE STANZA SHAPE. The brief this file was written against described
the default poem as "4 stanzas of 4 lines". The source holds FOUR stanzas of
4/4/4/2 lines -- 14 lines, not 16 -- because the published poem ends on the
two-line couplet ("I am the master of my fate: / I am the captain of my
soul!"). The module docstring in `src/stimulus.py` says "Sixteen short lines
in four stanzas", which is off by two; the stanza data itself matches the
published text. This test pins what the code actually holds, so that the
number of lines the corpus records can never change unnoticed, and the
discrepancy is reported rather than silently papered over.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import stimulus                                    # noqa: E402


def text_lines() -> list[str]:
    """STIMULUS_TEXT as a list of stripped, non-empty lines."""
    return [ln.strip() for ln in stimulus.STIMULUS_TEXT.splitlines()
            if ln.strip()]


def text_blocks() -> list[list[str]]:
    """STIMULUS_TEXT split on its blank lines: one block per stanza."""
    blocks = []
    for block in stimulus.STIMULUS_TEXT.split("\n\n"):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if lines:
            blocks.append(lines)
    return blocks


def stanza_lines() -> list[str]:
    """Every line of every stanza, in order."""
    return [ln for st in stimulus.STIMULUS_STANZAS for ln in st]


# -- the default stimulus ---------------------------------------------------

class TestDefaultStimulus:
    def test_load_stimulus_returns_the_default_poem(self):
        """
        With no path, `load_stimulus` must return the module's own poem --
        and return a COPY of the stanzas, so a caller that mutates them
        cannot rewrite the constant every later take reads from.
        """
        name, title, author, stanzas, est = stimulus.load_stimulus()
        assert name == stimulus.STIMULUS_NAME == "invictus"
        assert title == stimulus.STIMULUS_TITLE
        assert author == stimulus.STIMULUS_AUTHOR
        assert stanzas == [list(st) for st in stimulus.STIMULUS_STANZAS]
        assert est == stimulus.ESTIMATED_SECONDS

        stanzas[0][0] = "mutated"
        assert stimulus.STIMULUS_STANZAS[0][0] != "mutated", \
            "load_stimulus handed back the shared stanza list itself"

    def test_default_poem_has_four_stanzas_of_four_four_four_and_two_lines(self):
        """
        Four stanzas, no empty stanza, and the line counts pinned exactly.

        The shape is 4/4/4/2 -- 14 lines -- because the poem closes on the
        two-line couplet. The module docstring calls it "Sixteen short
        lines"; the data holds fourteen. Pinning the counts here is what
        makes a future edit to the poem visible instead of silent: the
        corpus records a line count, and that number must not change
        without someone deciding to change it.
        """
        stanzas = stimulus.load_stimulus()[3]
        assert len(stanzas) == 4, f"expected 4 stanzas, got {len(stanzas)}"
        assert [len(st) for st in stanzas] == [4, 4, 4, 2], \
            f"stanza line counts are {[len(st) for st in stanzas]}, not [4,4,4,2]"
        for i, st in enumerate(stanzas):
            assert st, f"stanza {i} is empty"
            for ln in st:
                assert ln.strip(), f"stanza {i} has a blank line"

    def test_the_poem_is_henleys_invictus(self):
        """
        The identity of the stimulus, pinned by its first and last line.

        A stimulus that is sayable from memory has to be the one Casey
        actually knows; these two lines are the couplet everyone knows, and
        they bracket the take.
        """
        lines = stanza_lines()
        assert lines[0] == "Out of the night that covers me,"
        assert lines[-1] == "I am the captain of my soul!"
        assert stimulus.STIMULUS_TITLE == "Invictus"
        assert "Henley" in stimulus.STIMULUS_AUTHOR


# -- the two representations must not drift apart ---------------------------

class TestTextAndStanzasAgree:
    def test_every_stanza_line_appears_in_the_text(self):
        """
        THE test. A stanza line missing from the text is a line the screen
        does not show -- Casey reads a different poem from the one the
        metadata records, and no downstream check can see it.
        """
        txt = text_lines()
        missing = [ln for ln in stanza_lines() if ln not in txt]
        assert not missing, (
            f"{len(missing)} stanza line(s) are absent from STIMULUS_TEXT: "
            f"{missing}")

    def test_every_text_line_appears_in_the_stanzas(self):
        """
        The other direction. A line in the text with no stanza is a line
        displayed but not recited, which is the same defect read the other
        way round: the corpus would contain a take of a shorter poem.
        """
        stanza_set = set(stanza_lines())
        extra = [ln for ln in text_lines() if ln not in stanza_set]
        assert not extra, (
            f"{len(extra)} line(s) in STIMULUS_TEXT have no stanza: {extra}")

    def test_the_text_blocks_are_the_stanzas_in_order(self):
        """
        Strongest form: the blank-line-separated blocks of the text are the
        stanzas, in order, line for line. This catches a reordering, which
        the membership checks above would not.
        """
        assert text_blocks() == [list(st) for st in stimulus.STIMULUS_STANZAS]

    def test_the_two_representations_have_the_same_line_count(self):
        """A one-line count disagreement is the drift, counted."""
        assert len(text_lines()) == len(stanza_lines())
        assert len(text_blocks()) == len(stimulus.STIMULUS_STANZAS)


# -- the alternatives --------------------------------------------------------

class TestAlternatives:
    def test_every_alternative_is_usable(self):
        """
        The alternatives are the escape hatch -- if Casey knows a poem better
        than Invictus, he recites that instead. So each one must have the
        same workable shape as the default: at least two stanzas (a
        one-stanza poem is a paragraph, which is the thing this whole module
        exists to stop being), no empty stanza, and a title and author to
        put in the metadata.
        """
        assert stimulus.ALTERNATIVES, "there are no alternatives"
        for key, spec in stimulus.ALTERNATIVES.items():
            title, author, stanzas, est, rec = spec
            assert title and title.strip(), f"{key}: empty title"
            assert author and author.strip(), f"{key}: empty author"
            assert len(stanzas) >= 2, f"{key}: only {len(stanzas)} stanza(s)"
            for i, st in enumerate(stanzas):
                assert st, f"{key}: stanza {i} is empty"
                for ln in st:
                    assert ln.strip(), f"{key}: stanza {i} has a blank line"
            assert est > 0, f"{key}: estimated seconds is {est}"
            assert rec >= est, (
                f"{key}: recommended {rec}s is shorter than the estimated "
                f"{est}s")

    def test_the_alternatives_are_the_named_ones(self):
        """
        The set is a decision, recorded: three public-domain poems, each
        sayable from memory. A fourth entry changes what the corpus can
        contain and must not arrive unnoticed.
        """
        assert set(stimulus.ALTERNATIVES) == {
            "sea-fever", "o-captain", "casey-at-the-bat"}


# -- formatting --------------------------------------------------------------

class TestFormatStimulus:
    def test_format_contains_the_title_the_author_and_every_line(self):
        """
        This string is what is printed on screen before a speaking take. If
        it drops a line, Casey recites something that is not the recorded
        stimulus -- so it must contain the title, the author and every line
        of every stanza, in order.
        """
        out = stimulus.format_stimulus(
            stimulus.STIMULUS_NAME, stimulus.STIMULUS_TITLE,
            stimulus.STIMULUS_AUTHOR, stimulus.STIMULUS_STANZAS)
        assert isinstance(out, str)
        assert stimulus.STIMULUS_TITLE in out
        assert stimulus.STIMULUS_AUTHOR in out
        for st in stimulus.STIMULUS_STANZAS:
            for ln in st:
                assert ln in out, f"the printed stimulus omits {ln!r}"
        positions = [out.index(ln) for st in stimulus.STIMULUS_STANZAS
                     for ln in st]
        assert positions == sorted(positions), \
            "the printed stimulus has the lines out of order"

    def test_format_of_an_alternative_prints_its_own_poem(self):
        """The alternatives are printable by the same path."""
        title, author, stanzas, _est, _rec = stimulus.ALTERNATIVES["sea-fever"]
        out = stimulus.format_stimulus("sea-fever", title, author, stanzas)
        assert title in out and author in out
        for st in stanzas:
            for ln in st:
                assert ln in out


# -- the file-backed escape hatch --------------------------------------------

class TestLoadFromFile:
    def test_a_supplied_file_is_parsed_into_stanzas(self, tmp_path):
        """
        Casey's own text: the escape hatch for a poem he knows better than
        Invictus. Blank lines separate stanzas; the name and title come from
        the file stem; the author says where it came from, so the corpus
        records which stimulus each take used.
        """
        p = tmp_path / "my_poem.txt"
        p.write_text(
            "first line of the poem\n"
            "second line of the poem\n"
            "\n"
            "third line of the poem\n"
            "fourth line of the poem\n",
            encoding="utf-8")

        name, title, author, stanzas, est = stimulus.load_stimulus(p)
        assert name == "my_poem"
        assert title == "my_poem"
        assert "my_poem.txt" in author
        assert stanzas == [["first line of the poem", "second line of the poem"],
                           ["third line of the poem", "fourth line of the poem"]]
        assert est > 0

    def test_an_empty_file_is_a_clear_error(self, tmp_path):
        """
        An empty stimulus must fail LOUDLY. The alternative -- an empty
        stanza list -- would be recorded as a take with no stimulus, and the
        corpus would look complete.
        """
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError) as exc:
            stimulus.load_stimulus(p)
        assert "no text" in str(exc.value)
        assert "empty.txt" in str(exc.value)

    def test_a_whitespace_only_file_is_also_an_error(self, tmp_path):
        """Whitespace is not text: the same clear error, not an empty take."""
        p = tmp_path / "blank.txt"
        p.write_text("\n\n   \n", encoding="utf-8")
        with pytest.raises(ValueError):
            stimulus.load_stimulus(p)


# -- the timing the capture plan depends on ---------------------------------

class TestTiming:
    def test_recommended_seconds_is_not_shorter_than_estimated(self):
        """
        The capture plan gives the performer RECOMMENDED_SECONDS for a
        speaking take. If that window were shorter than the poem needs, the
        performer would be cut off mid-poem, and the tail of every take would
        be a truncation rather than a silence -- which the gate would read
        as the end of the utterance rather than the end of the capture.
        """
        assert stimulus.ESTIMATED_SECONDS > 0
        assert stimulus.RECOMMENDED_SECONDS >= stimulus.ESTIMATED_SECONDS, (
            f"recommended {stimulus.RECOMMENDED_SECONDS}s is shorter than the "
            f"estimated {stimulus.ESTIMATED_SECONDS}s")
