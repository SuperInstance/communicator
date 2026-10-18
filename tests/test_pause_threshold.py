"""
Tests for `MIN_SILENCE_S` -- the constant the adoption plan wanted changed.

`docs/research/ADOPTION-PLAN.md` §3 recommended raising it from 150 ms to
200 ms, on the reasoning that pipecat triggers a VAD stop at ~200 ms and
gives up on a turn at 3 s, so our single 150 ms threshold is a fine trigger
and a 20x-too-eager decision.

MEASURED ON THE SEVEN-CELL REAL CORPUS, IT IS THE OTHER WAY ROUND.

Inter-utterance gaps in real human speech (`examples/realaudio/corpus/`):

    cell          gaps   shortest    p05   median
    narration        45     0.170  0.210    0.640
    poetry            7     0.150  0.150    0.510
    essay            33     0.150  0.160    0.760
    dialogue         35     0.130  0.160    0.450
    shakespeare       62    0.110  0.160    0.380
    children          78    0.110  0.160    0.300
    oration           2     0.310  0.310    0.380

The shortest pause a human voice takes is 110 ms. 150 ms already sits BELOW
the shortest real pause in every cell, so it is correctly placed for the
question it is actually answering.

Raising it to 200 ms costs:

    cell          @0.150  @0.200   lost    lost%
    narration         45      43      2     4.4%
    poetry             7       6      1    14.3%
    essay             33      24      9    27.3%
    dialogue          34      30      4    11.8%
    shakespeare        61      46     15    24.6%
    children           77      62     15    19.5%
    oration             2       2      0     0.0%
    TOTAL             259     213     46    17.8%

17.8% of all real pauses disappear, and the loss concentrates in dramatic and
children's speech -- the two cells where the pause structure IS the prosody
and the hardest to replace with anything.

A pause is not decoration. It is the event the §4.1 baseline table counts and
every TEMPO calculation divides by, so losing 46 of them is data loss wearing
the costume of a tuning change.

THE ERROR, NAMED. A trigger constant was treated as a decision constant.
150 ms answers "did the voice stop"; 3 s answers "is the turn over". They are
different questions with different evidence, and the 3 s one belongs in a
separate stage that CONSUMES pause events rather than filtering them.

These tests exist so the measurement cannot be quietly undone, and so the
3 s decision -- when it is built -- has to add a stage instead of moving this
number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import energy, wavio                                     # noqa: E402

SR = 16000
HOP = 160
HOP_S = HOP / SR
CORPUS = ROOT / "examples" / "realaudio"

# The seven cells from examples/realaudio/corpus/MANIFEST.md. Held out from
# every threshold decision made before the corpus existed, which is what makes
# them a test rather than a demonstration.
CELLS = {
    "narration": CORPUS / "alice01_excerpt.wav",
    "poetry": CORPUS / "corpus" / "poetry_weeds_by_the_wall_1702_librivox.wav",
    "essay": CORPUS / "corpus" / "essay_perpetual_peace_librivox.wav",
    "dialogue": CORPUS / "corpus" / "shortstory_shortstory058_1405_librivox.wav",
    "shakespeare": CORPUS / "corpus"
                   / "shakespeare_midsummernightsdream_1201_librivox.wav",
    "children": CORPUS / "corpus"
                / "children_english_fairy_tales_joy_librivox.wav",
    "oration": CORPUS / "corpus" / "oration_jfks19620912.wav",
}


def _gaps(path: Path):
    """Every silence run the segmenter finds, in seconds."""
    if not path.is_file():
        pytest.skip(f"corpus cell absent: {path.name}")
    samples = wavio.read_wav(str(path)).samples
    ef, _ = energy.rms_series(samples, 800, HOP, SR)
    if not ef:
        pytest.skip(f"{path.name} produced no frames")
    segs = energy.segment(ef, HOP_S, peak_rms=max(f.rms for f in ef))
    return [s.duration() for s in segs if not s.speech]


class TestRealPauseLength:
    def test_the_shortest_real_pause_is_under_the_trigger(self):
        """
        THE measurement. If this fails, the trigger is no longer below real
        speech and the plan's argument has to be reopened on evidence rather
        than on recollection.
        """
        shortest = {name: min(_gaps(p)) for name, p in CELLS.items()}
        assert min(shortest.values()) < energy.MIN_SILENCE_S, (
            f"the shortest real pause is now {min(shortest.values()):.3f}s, "
            f"at or above MIN_SILENCE_S={energy.MIN_SILENCE_S}; per cell: "
            + ", ".join(f"{k}={v:.3f}" for k, v in sorted(shortest.items())))

    def test_no_cell_has_its_pauses_eaten_at_the_shipped_value(self):
        """
        At 150 ms the segmenter keeps essentially every real pause. The point
        is not that the number is perfect -- it does merge a handful of the
        very shortest -- but that the shipped value is on the correct side of
        the distribution rather than inside it.
        """
        for name, path in CELLS.items():
            gaps = _gaps(path)
            eaten = sum(1 for g in gaps if g < energy.MIN_SILENCE_S)
            assert eaten / len(gaps) < 0.05, (
                f"{name}: {eaten}/{len(gaps)} real pauses are below "
                f"MIN_SILENCE_S={energy.MIN_SILENCE_S}")

    def test_dramatic_speech_has_the_shortest_pauses(self):
        """
        The reason a single global trigger is a compromise at all, and the
        reason the 200 ms figure from pipecat does not transfer.

        Shakespeare and children's speech sit at 110 ms; measured oratory sits
        at 310 ms. A threshold chosen from the loud end of that range -- which
        is what a system tested mostly on narration will produce -- throws away
        a fifth of the pauses in the cells that carry the most prosody.
        """
        dramatic = min(min(_gaps(CELLS["shakespeare"])),
                     min(_gaps(CELLS["children"])))
        oration = min(_gaps(CELLS["oration"]))
        assert dramatic < oration / 2, (
            f"dramatic {dramatic:.3f}s vs oration {oration:.3f}s -- the "
            f"expected 3x spread is gone; the corpus or the segmenter changed")


class TestTheRejectedChange:
    def test_raising_the_trigger_would_lose_a_lot_of_real_pauses(self):
        """
        The falsification, asserted rather than left in a document.

        Stated as a LOSS BUDGET rather than a magic number: a change to this
        constant that costs more than a fifth of the pauses is a change that
        needs new evidence, not a better rationale.
        """
        total = 0
        lost = 0
        for path in CELLS.values():
            gaps = _gaps(path)
            total += len(gaps)
            lost += sum(1 for g in gaps if g < 0.200)
        assert lost / total > 0.15, (
            f"raising MIN_SILENCE_S to 0.200 now costs only {lost}/{total} "
            f"pauses ({lost / total:.1%}) rather than the measured 17.8% -- "
            f"the corpus changed, so re-derive the argument")

    def test_the_prose_cost_is_what_makes_it_a_bad_trade(self):
        """
        The same loss, concentrated. The overall 17.8% understates it: the
        damage lands on dramatic and children's speech, where a pause carries
        meaning rather than breath.
        """
        for cell in ("shakespeare", "children"):
            gaps = _gaps(CELLS[cell])
            lost = sum(1 for g in gaps if g < 0.200)
            assert lost / len(gaps) > 0.15, (
                f"{cell}: only {lost}/{len(gaps)} pauses would be lost; the "
                f"case for keeping 150 ms was built on this cell")

    def test_the_three_second_decision_is_not_a_threshold_change(self):
        """
        The structural consequence, made executable.

        The 3 s end-of-turn decision must CONSUME pause events and emit turn
        boundaries. It must never FILTER them. So when it is built it has to
        add a kind of event and leave `MIN_SILENCE_S` alone -- and the cheapest
        way to keep that honest is to pin the constant now, before there is
        code that would want to move it.
        """
        assert energy.MIN_SILENCE_S == 0.150, (
            f"MIN_SILENCE_S is now {energy.MIN_SILENCE_S}; if this moved, the "
            f"3 s decision belongs in a new stage, not in this number")
        assert energy.MIN_SILENCE_S < 0.200, (
            "the shipped trigger is at or above the value measured to cost "
            "17.8% of real pauses")
