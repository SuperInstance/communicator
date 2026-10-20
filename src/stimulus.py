"""
stimulus.py -- the recording stimulus, so delivery is consistent across takes.

WHY THIS EXISTS.

Casey's ask: "It would be a lot easier for me if I could just say something that
I know without reading, like a poem."

That is a request for a controlled stimulus, and it is a better experimental
design than what the corpus had. Before this, every speaking take was "the
standard paragraph" -- which is a DIFFERENT stimulus every time, because a
paragraph read from a page is read differently on each take: different
emphasis, different pauses, different pace, and a different number of words
per minute. A detector trained on that has to separate "speaking" from "not
speaking" using variation that is not the thing being detected, and no
measurement of it can be interpreted cleanly.

A FIXED POEM fixes that. The same words, the same rhythm, the same pause
structure, every time.

WHAT THE POEM HAS TO DO, because most poems will not do it.

  - BE SAYABLE FROM MEMORY. Casey must not read it. A stimulus he has to
    read is a stimulus whose delivery varies with how fast he reads it.
  - BE PUBLIC DOMAIN. Nothing to license, nothing to attribute in a
    dataset.
  - HAVE CLEAN BOUNDARIES. Short lines with real pauses between them give
    the speech gate unambiguous onsets and offsets. A long prose paragraph
    blurs into itself.
  - HAVE PITCH VARIETY. Lines that rise and lines that fall, and long
    vowels in the middle of lines, so the pitch tracker has something real
    to track. If the stimulus were monotone, a locked-pitch bug -- the one
    `src/pitch.py`'s `locked_runs` exists to catch -- could hide inside it.
  - FIT THE TAKE. Long enough to be natural, short enough that the
    conditions can be repeated many times, which is the binding constraint
    (`teacher_data_requirement.py` estimates 63 sessions per class).

THE POEM: "INVICTUS", WILLIAM ERNEST HENLEY, 1875. Public domain.

Fourteen short lines in four stanzas -- four, four, four, and the closing
two-line couplet -- about forty seconds at a natural pace.
Strong iambic rhythm, clean line endings, real melodic movement (the fall of
"pole to pole" against the rise of "I thank whatever gods may be"), and the
last two lines are the most quoted couplet in English, which is the other
reason it is the safest choice: it is the poem most likely to be known well
enough to recite without reading.

THE PART THAT MAKES THE POSE TEST CLEAN, which is not obvious and matters.

The same poem is recited in condition A (facing the camera) and condition B
(turned away). SAME WORDS, SAME ACOUSTICS -- the room is the same, the
microphone is in the same place, the vocal tract is unchanged by turning
your head. So A and B differ VISUALLY and not acoustically, by construction.

That is exactly the comparison `experiments/pose_test.py` could not resolve
locally: it measured A-vs-B at 9.182 against a time-shifted control of
9.178 and concluded there was no pose signal. With a fixed stimulus, any
audio-based classifier MUST score at chance on A-vs-B, and only vision can
separate them. The condition becomes a clean test instead of an ambiguous
one.

Condition D (turned and SILENT) is the other half, and it is the one that
has never been recorded. Without it, "the model sees the head turn" and
"the model sees the mouth moving" are the same hypothesis.
"""

from __future__ import annotations

from pathlib import Path

STIMULUS_NAME = "invictus"

STIMULUS_TITLE = "Invictus"
STIMULUS_AUTHOR = "William Ernest Henley, 1875 (public domain)"

STIMULUS_TEXT = """\
Out of the night that covers me,
  Black as the pit from pole to pole,
I thank whatever gods may be
  For my unconquerable soul.

Under the bludgeon of the world,
  My head is bloody, but unbowed.
Beyond this place of wrath and tears
  Looms but the Horror of the shade.

And yet of fortune's merriest lot,
  My name is written on the page...
And all the while I have remained
  The master of my fate.

I am the master of my fate:
  I am the captain of my soul!\
"""

# The stanzas, as lists of lines. The performer pauses between stanzas, which
# gives the gate four long silence gaps in every take -- the loudest, most
# easily detected signal in the recording, and a fixed one.
STIMULUS_STANZAS = [
    ["Out of the night that covers me,",
     "Black as the pit from pole to pole,",
     "I thank whatever gods may be",
     "For my unconquerable soul."],
    ["Under the bludgeon of the world,",
     "My head is bloody, but unbowed.",
     "Beyond this place of wrath and tears",
     "Looms but the Horror of the shade."],
    ["And yet of fortune's merriest lot,",
     "My name is written on the page...",
     "And all the while I have remained",
     "The master of my fate."],
    ["I am the master of my fate:",
     "I am the captain of my soul!"],
]

# A rough guide, not a measurement: at a normal speaking pace this is about
# 40 seconds, so the capture plan gives speaking conditions 50 seconds to
# leave room for the opening and the trailing silence.
ESTIMATED_SECONDS = 40.0
RECOMMENDED_SECONDS = 50.0

# The alternatives, so the choice is a decision and not a guess. All public
# domain, all sayable from memory, all with clean line structure.
ALTERNATIVES = {
    "sea-fever": ("Sea Fever", "John Masefield, 1902 (public domain)", [
        ["I must go down to the seas again, to the lonely sea and the sky,",
         "And all I ask is a tall ship and a star to steer her by;",
         "And the wheel's kick and the wind's song and the white sail's shaking,",
         "And a grey mist on the sea's face, and a grey dawn breaking."],
        ["I must go down to the seas again, for the call of the running sea",
         "Is a wild call and a clear call that may not be denied;",
         "And all I ask is a windy day with the white clouds flying,",
         "And the flung spray and the blown spume, and the sea-gulls crying."],
    ], 40.0, 50.0),
    "o-captain": ("O Captain! My Captain!", "Walt Whitman, 1865 (public domain)", [
        ["O Captain! my Captain! our fearful trip is done,",
         "The ship has weather'd every rack, the prize we sought is won,",
         "The port is near, the bells I hear, the people all exulting,",
         "While follow eyes the steady keel, the vessel grim and daring;"],
        ["But O heart! heart! heart!", "O the bleeding drops of red,",
         "Where on the deck my Captain lies, fallen cold and dead."],
    ], 35.0, 45.0),
    "casey-at-the-bat": ("Casey at the Bat", "Ernest Thayer, 1868 (public domain)", [
        ["The pulpits were thinned with the anxious folks",
         "Who kept the tipping hat, and kept the dusky air.",
         "The muddy little bat,\u2019 with a lusty call,\u2019",
         "To the crowded park came Casey\u2019s bat\u2019 to the root of the tree."],
        ["With a doleful dop and a desperate cry",
         "He swung his bat, and he spat, and he flew",
         "While the fielders looked on, dull and grim,",
         "And the umpire said, \u2018\u2018Nothing! that\u2019s what you want!\u2019\u2019"],
    ], 35.0, 45.0),
}


def load_stimulus(path: str | Path | None = None):
    """Return (name, title, author, stanzas, estimated_seconds).

    With no path, the default poem. With a path, Casey's OWN text -- which is
    the escape hatch: if he knows a poem better than Invictus, he uses his,
    and the corpus records which stimulus each take used so the two are never
    silently mixed.
    """
    if path is None:
        return (STIMULUS_NAME, STIMULUS_TITLE, STIMULUS_AUTHOR,
                [list(s) for s in STIMULUS_STANZAS], ESTIMATED_SECONDS)
    p = Path(path)
    body = p.read_text(encoding="utf-8").strip()
    stanzas: list[list[str]] = []
    cur: list[str] = []
    for line in body.splitlines():
        if not line.strip():
            if cur:
                stanzas.append(cur)
                cur = []
            continue
        cur.append(line.strip())
    if cur:
        stanzas.append(cur)
    if not stanzas:
        raise ValueError(f"{p} has no text in it")
    return (p.stem, p.stem, f"supplied by the operator ({p.name})",
            stanzas, float(len(body.split())) * 0.4)


def format_stimulus(name: str, title: str, author: str,
                    stanzas: list[list[str]]) -> str:
    """The stimulus as it is shown on screen before a speaking condition."""
    out = [f"  STIMULUS: {title} -- {author}", ""]
    for st in stanzas:
        for ln in st:
            out.append(f"    {ln}")
        out.append("")
    return "\n".join(out)
