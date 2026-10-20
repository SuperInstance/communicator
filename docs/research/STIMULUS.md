# THE STIMULUS — one fixed poem, every take

**Code: `src/stimulus.py`. Wired into `experiments/corpus_capture_win.py`.**

---

## 1. What was asked for, and why it is a better experiment

Casey: *"It would be a lot easier for me if I could just say something that I
know without reading, like a poem."*

That is a request for a **controlled stimulus**, and it improves the science,
not just the ergonomics.

The corpus used to say "the standard paragraph". A paragraph read from a page
is read differently on every take — different emphasis, different pauses,
different pace, a different words-per-minute. A teacher trained on that has
to separate *speaking* from *not speaking* using variation that is not the
thing being detected, and no measurement of it can be interpreted cleanly. It
also costs attention: the performer is reading, not performing.

A fixed poem makes the **acoustic content identical across takes**. The only
thing that varies is whether he is speaking at all, and (for the pose
conditions) where his head is.

## 2. What the poem has to do, because most poems will not do it

  - **BE SAYABLE FROM MEMORY.** Casey must not read it. A stimulus he has to
    read is a stimulus whose delivery varies with how fast he reads it.
  - **BE PUBLIC DOMAIN.** Nothing to license, nothing to attribute in a
    dataset.
  - **HAVE CLEAN BOUNDARIES.** Short lines with real pauses between them give
    the speech gate unambiguous onsets and offsets. A long prose paragraph
    blurs into itself.
  - **HAVE PITCH VARIETY.** Lines that rise and lines that fall, and long
    vowels in the middle of lines, so the pitch tracker has something real to
    track. If the stimulus were monotone, a locked-pitch bug — the one
    `src/pitch.py`'s `locked_runs` exists to catch — could hide inside it.
  - **FIT THE TAKE.** Long enough to be natural, short enough that the
    conditions can be repeated many times, which is the binding constraint
    (`teacher_data_requirement.py` estimates 63 sessions per class).

## 3. The poem

### **INVICTUS** — William Ernest Henley, 1875 (public domain)

```
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
  I am the captain of my soul!
```

Fourteen short lines in four stanzas -- **4 / 4 / 4 / 2**, because the poem ends
on a couplet -- about **40 seconds** at a natural pace.

Why this one:

  - **Fourteen lines with clean endings.** Four long silence gaps, one
    between each stanza, in every take — the loudest and most reliably
    detectable signal in the recording, and a *fixed* one.
  - **Real melodic movement.** The fall of "pole to pole" against the rise of
    "I thank whatever gods may be"; long vowels in "unconquerable" and
    "captain".
  - **Memorable.** The last couplet is among the most quoted lines in
    English, which is the practical reason it is the safest choice: it is the
    poem most likely to be known well enough to recite without reading.
  - **Public domain since 1901.**

### The alternatives, if you know one better

`src/stimulus.py` ships three more, all public domain, all with clean line
structure:

| key | poem | length |
|---|---|---:|
| `sea-fever` | "Sea Fever", John Masefield, 1902 | 40 s |
| `o-captain` | "O Captain! My Captain!", Walt Whitman, 1865 | 35 s |
| `casey-at-the-bat` | "Casey at the Bat", Ernest Thayer, 1868 | 35 s |

And the escape hatch: `--stimulus-file <path>` takes **any text** and uses it
as the stimulus. If you know a poem better than Invictus, use yours — the
corpus records which stimulus each take used, so the two are never silently
mixed.

## 4. The part that makes the pose test clean, which is not obvious

**The same poem is recited in A (facing) and B (turned away).**

Same words, same room, same microphone, same vocal tract — turning your head
does not change how you sound. So **A and B differ visually and not
acoustically, by construction.**

That is exactly the comparison `experiments/pose_test.py` could not resolve
locally: it measured A-vs-B at **9.182** against a time-shifted control of
**9.178** and concluded there was no pose signal. With a fixed stimulus:

  - any **audio**-based classifier must score at chance on A-vs-B, because
    the audio is nominally identical;
  - only **vision** can separate them;
  - so the condition becomes a clean test instead of an ambiguous one.

**Condition D** (turned and silent) is the other half, and it has never been
recorded. Without it, "the model sees the head turn" and "the model sees the
mouth moving" are the same hypothesis. The cloud measurement this session
used exactly this control: it asks the pose question of condition C
(facing, silent) as well, and if `P(facing | facing AND SILENT)` collapses
towards `P(facing | turned)`, then what the model is reading is speech, not
pose.

## 5. How it is used in the capture

```
CONDITION A1: facing the camera, reciting the poem from memory
  duration 50s   pose=facing   speak=True

  STIMULUS: Invictus -- William Ernest Henley, 1875 (public domain)

    Out of the night that covers me,
    Black as the pit from pole to pole,
    ...
  RECITE FROM MEMORY. The same words every time -- that is the point.
```

The poem is shown so it can be checked, not read. The duration is 50 s for
speaking conditions (40 s of poem plus the opening and trailing silence) and
30 s for the silent ones.

Each take's metadata records `stimulus: "invictus"`, so the corpus is
self-describing and a future analysis can group by it.

## 6. The conditions, and what each one is for

| | pose | speech | what it isolates |
|---|---|---|---|
| **A** | facing | reciting | speech + facing |
| **B** | turned | reciting | speech, no facing — and *acoustically identical to A* |
| **C** | facing | silent | facing, no speech |
| **D** | turned | silent | neither — the baseline the other three are read against |
| **F** | any | any | no label; the teacher labels these, once it is trusted |

D has never been recorded. It is the one that makes the 2×2 complete, and
without it the project cannot tell pose from speech at all.

## 7. A caveat that belongs here

A fixed stimulus is a controlled stimulus, and controlled stimuli have a
cost: **a speech detector that has learned the poem's particular acoustic
signature will pass the conditions and fail in the world.** The silent
conditions catch the worst of this (C and D contain no poem at all, so a
poem-detector scores zero on them), and the free sessions F are the escape
hatch — they contain ordinary speech and no label. The teacher is only
allowed to label F once it beats chance on A–D, which is the gate the corpus
script enforces.

The honest summary: **the poem makes the controlled comparison possible, and
the free sessions are what keep the result honest.**
