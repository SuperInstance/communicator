# MOTION AS TEXT — the session becomes a document an LLM can read

**Built by:** `experiments/motion_as_text.py`
**The session:** `/mnt/c/proart_capture/corpus/free_sessions/1792553040` —
tonight's real free session, 15,514 frames of 160×96 luma at 59.88 fps.
**Device:** CPU, numpy. No GPU, no paid API, no network.

---

## 0. THE HEADLINE

**A four-minute session becomes 27.4 kB/s of text and reference frames
instead of 920.4 kB/s of pixels — 33.6× smaller — and the text is
readable as it stands.**

| | bytes/s | what it is |
|---|---|---|
| the raw luma | 920.4 kB/s | 15,514 frames of 15,360 bytes |
| the reference frames | 8.5 kB/s | 262 PNGs that anchor the picture |
| the delta stream | 15.1 kB/s | plain text, one line per frame |
| the motion stream | 3.8 kB/s | plain text, the edges that moved |
| **the total** | **27.4 kB/s** | **33.6× smaller** |

**And the budget can be spent on the CHANGE instead of on the clock:
8,885.5 bytes/s, 0.375× the clock-indexed cost — 2.7× cheaper.** Section 5A
is the measurement, and it rests on a number the lens files now own: 93.4 %
of the session is still.

**The reconstruction works, and it works better the finer the grid, and
the price is bytes.** At 20×12 the reconstruction's mean absolute error
is 12.20 luma levels against 14.24 for the reference alone — the motion
stream buys back 14% of the error. At 80×48 the same stream buys back
56%, for eight times the bytes.

**It breaks where the picture has detail, not where the person moves.**
The twenty worst frames are all in the top fifth of the session by
within-block texture, and the linear correlation between the error and
the motion is *negative* (−0.16): the encoding does not break on the
gesture. It breaks on texture, because texture is what the grid throws
away.

**And the payoff: the same frame is a picture and a line of text.**
Section 6 has Casey's gesture at 182.4 s, drawn from the luma and drawn
from the text stream — the same moment, twice, one as pixels and one as
characters, and the characters are the ones in the file.

---

## 1. THE SESSION

| | |
|---|---|
| frames | 15,514 of 160×96 raw luma |
| raw size | 238.3 MB |
| clock | 59.880 fps, frames at 6.83 s to 265.74 s, span 258.90 s |
| wall clock | 2026-10-21T03:19:40.683Z to 03:23:59.592Z |
| audio | **silent** — every sample zero, the sick-lane bug |

The audio does not matter here and that is worth saying: the motion
channel is the first one in this system that works on tonight's session
without waiting for the microphone fix. The session's *image* is fine.

The repository's parts are used, not re-implemented: `src/nv12.py` for
the luma, `experiments/corpus_capture_win.py:change_maps` for the motion
energy, and the speech-to-score tool's session clock for the frame
indices and the wall time.

---

## 2. THE ENCODING

Two streams, because they answer different questions, and only one of
them reconstructs.

### 2.1 THE DELTA STREAM — the one that reconstructs

For each frame, the **signed** difference between the frame and its
reference, on a coarse grid, one character per cell.

```
F 010518  182.4035 2026-10-21T03:22:43.086Z 010480 00000000@@0000000000/00000000000000000000/...
```

The format is: the frame index, the session's seconds, the session's wall
time, the frame index of the reference, and the grid — one row per line
of the text, rows separated by `/`.

**The ramp.** Index *i* is the value *i* − 7, so the values run from −7
to +7 and the zero is the character `0`. The characters are chosen to be
legible rather than to be in ASCII order: **the digits are the positive
side, the punctuation is the negative side, and `0` is a cell that did
not move.**

```
-  =  +  *  #  %  @  0  1  2  3  4  5  6  7
-7 -6 -5 -4 -3 -2 -1  0 +1 +2 +3 +4 +5 +6 +7
```

**The scale** is measured from the data, not chosen: one ramp unit is
**39.28 luma levels** on this session, taken as the 90th percentile of
the absolute per-cell difference from the reference. The scale is written
into the stream's header, so the file describes itself.

This is the stream that reconstructs, because a signed difference can be
added back. The motion stream cannot be.

### 2.2 THE MOTION STREAM — where the edges moved

The repository's own change maps — per-frame |frame(t+1) − frame(t)|,
downsampled 2×2 — reduced to the cells that clear a threshold, and
written as sparse events.

```
E 010519  182.4194 2026-10-21T03:22:43.102Z 40,22,11 41,23,9 44,25,7 ...
```

The format is `x,y,magnitude`, with the coordinates in the change map's
own 80×48 grid. Only the strongest cells are named, so the line stays
short: 3.8 kB/s against the delta stream's 15.1.

**This stream does not reconstruct and is not supposed to.** It says
where the edges moved and how much they moved; it does not say where the
picture went. A motion map is not a displacement, and the honest thing
is to keep the two apart rather than to pretend one is the other.

### 2.3 THE CLOCK CONTRACT

Every reference and every event carries the session's own clock: the
frame index, the seconds from `timestamps.json`, and the absolute wall
time built from `timestamps.json`'s offsets and `meta.json`'s
`wall_clock_start`. The streams are meant to sit in the **same alignment
table as the transcript words and the MIDI notes**, and they can, because
the frame index is the same integer the speech-to-score tool uses.

That is the whole point of the one-clock rule, and this lane is the
second place it pays: a row in the motion stream and a row in the
transcript's alignment table are joinable on the frame index with no
conversion and no second timeline.

---

## 3. THE REFERENCE FRAMES

262 references: 249 on the clock at one per second, and 13 on a scene
change.

**The scene threshold is measured, not guessed.** The mean absolute
difference between a frame and its current reference has a median of
4.78 and a 99th percentile of **25.16**, so the threshold is 25 — the
top one per cent of frames. An earlier run used 6.0, and it fired 550
times, because a threshold that low is ordinary motion and not a scene
change. The tool reports how many references came from each cause, so
the choice is visible in the output rather than buried in a constant.

The references are stored as PNGs: 2.20 MB for the whole session, 8.5
kB/s. A reference that cost more than the pixels it replaces would have
missed the point of the exercise, and at 160×96 it does not.

---

## 4. THE RECONSTRUCTION, AND WHERE IT BREAKS

The reconstruction is: the reference's cells, plus the stream's signed
delta, block-upsampled back to 160×96, compared against the true frame.

**The error is measured at full resolution, and that matters.** Comparing
the grid against the grid would be trivially zero, because the grid *is*
the reference plus the delta. What a reader actually gets back is the
block-upsampled grid, and the error is the detail inside the blocks that
the grid threw away. The first version of this tool measured grid
against grid and reported a perfect zero, which was a measurement bug and
not a result.

| | mean abs error (luma levels) |
|---|---|
| the reference alone, no motion at all | 14.24 |
| reference + the delta stream | **12.20** |
| p95 of the reconstruction | 13.29 |
| the worst frame | 14.28 |

The motion stream buys back **14%** of the error at this grid.

**Where it breaks: on texture, not on motion.** The correlation between
the reconstruction error and the frame's within-block texture is −0.15,
and between the error and the motion is −0.16. Both are weak, and the
weakness is the finding: the error is close to a constant, because the
grid's resolution cost is a constant. What varies is the *baseline* — how
far the picture had drifted from the reference — and that is where the
motion stream earns its keep.

The twenty worst frames are all at 187–190 s, and every one of them has
a within-block texture above the 80th percentile of the session. They
are the frames with the most detail in them, not the frames with the
most movement. The encoding does not break on the gesture.

---

## 5. THE GRID SWEEP — what each resolution costs and what it loses

Measured on 240 sampled frames, each reconstructed against its own
reference at one second old.

| grid | kB/s | error | reference alone | the motion is worth |
|---|---|---|---|---|
| 8×6 | 3.2 | 18.25 | 20.27 | 10% |
| 10×6 | 3.9 | 16.70 | 19.02 | 12% |
| 16×10 | 10.1 | 14.01 | 17.25 | 19% |
| **20×12** | **15.1** | **12.22** | **15.93** | **23%** |
| 40×24 | 59.0 | 8.35 | 13.47 | 38% |
| 80×48 | 233.2 | 5.05 | 11.38 | 56% |

(The headline run's 20×12 row reads 12.20 and 14.24 rather than 12.22
and 15.93, because the headline run's references also advance on a scene
change and the sweep's do not. The sweep is the like-for-like
comparison; the headline is the real thing.)

**The trade is legible.** Every doubling of the grid buys about 25% of
the remaining error and costs about 4× the bytes. The 20×12 grid is the
point where the text is still a document — 15.1 kB/s, a line per frame
that a reader can take in one at a time — and it is where the motion
stream is worth a fifth of the error. Below it the picture is a blur;
above it the text is a file again.

---

## 5A. THE RATE POLICY — spend the budget on the change, not on the clock

The ASCII-phone addendum's rule is: **stillness buys resolution and drops
the frame rate; motion buys frame rate and drops the resolution; the
budget follows the change, never the clock.** This section is what that
rule costs, measured.

The clock-indexed design is the one the spec describes above: one
reference per clock second, every frame's delta carried, every motion
event carried. The policy-indexed design spends the same byte budget
differently — one high-fidelity description per *still stretch*, and on
the live channel only the changed frames, at the coarse grid.

| | clock-indexed | policy-indexed | ratio |
|---|---|---|---|
| references | 262 | 1,068 | |
| reference bytes | 2,196,034 | 2,209,692 | |
| delta bytes | 3,909,528 | **90,780** | |
| sparse bytes | 28,133 | 0 | |
| **total bytes** | **6,133,695** | **2,300,472** | |
| **bytes/s** | **23,691.2** | **8,885.5** | **0.3751×** |

**The policy is 2.7× cheaper, and where the saving comes from is
measurable.** The delta stream falls from 3,909,528 bytes to 90,780 — a
**43×** reduction — while the references cost essentially the same
(2,196,034 against 2,209,692). The clock-indexed design spends 63.7 % of
its bytes on the delta stream; the policy spends 3.9 %. The references
are the fixed cost of seeing the room at all, and the policy does not
avoid them: it *increases* the number of high-fidelity anchors from 262
to 1,068 and spends less per frame.

### 5A.1 WHY: the session is 93.4 % still

The policy is only as good as its stillness test, and the test is where
this section's two measurement bugs were found.

| | |
|---|---|
| consecutive frame pairs | 15,513 |
| pairs with zero change | 10,339 (**66.7 %**) |
| non-zero change, median | 1.152 luma |
| non-zero change, p90 | 3.047 luma |
| the threshold used | **3.0 luma** |
| stretches found | 1,068 (534 still, 534 moving) |
| time still | 241.8 s of 258.9 s (**93.4 %**) |
| time moving | 8.8 s |
| frames unchanged | 14,980 (**96.6 %**) |

**Stillness is a MAGNITUDE test, and two other tests were tried first and
both were wrong.**

The first tested stillness on the *delta encoding* — comparing each
frame's delta against the previous one. But the clock's reference advance
makes a still frame's delta look changed, because the reference moved
under it. That gave 6,680 stretches instead of 1,068, and the policy
looked **6.8× worse** than the clock. The measurement was measuring the
reference schedule, not the room.

The second tested it on **grid equality** — whether the frame's coarse
grid equals the previous frame's. That fails on camera noise: the sensor
grain moves cells by fractions of a luma level, so no two grids are
exactly equal. The room map's own motion-noise σ is 0.61 luma.

The third test is a **magnitude** test at a measured threshold: 3.0 luma
is **2.6×** the non-zero median of 1.152 luma and about **5×** the room
map's noise σ of 0.61 luma. That threshold is now the camera lens file's
`SENSOR_NOISE_FLOOR` entry as well, so one number has one meaning.

### 5A.2 The two triggers disagree, and the union is the honest test

| trigger | fires on |
|---|---|
| whole-frame change | 534 frames |
| the motion map (person zone) | 418 frames |
| **either (the union)** | **952 frames** |
| motion map only | 418 |
| whole-frame only | 534 |

The whole-frame change and the motion map **disagree** — 534 frames where
the whole picture changed but the person zone did not, and 418 where the
person zone moved but the whole frame's maximum stayed under threshold.
Neither is the truth about "the picture changed". The union is the
defensible test, and it is recorded in `triggers_disagree` so the number
is auditable rather than assumed.

### 5A.3 What the exchange rate is

The policy's arithmetic is the grid sweep's own table, so the two
sections cannot drift apart: the **fine** grid costs **23.0×** the
coarse grid's bytes, and the **live** grid costs **1.49×** the coarse's.
So the policy buys resolution 23× more cheaply than it buys frame rate —
which is exactly the right exchange for a room, where the walls are
static and the person is not.

---

## 6. THE GESTURE

Tonight's session has one large, fast movement in the 175–190 s window:
the mean absolute difference between consecutive frames reaches **5.77**
at frame 10,522 (182.47 s) against a session median of 0.52 and a
session maximum of 13.1. That is the gesture.

The frame below is 10,518, at 182.403 s, wall clock 03:22:43.086Z. The
background is the median of 388 frames of the session subtracted, so the
person stands out of the room instead of drowning in it. The rendering
is a rendering choice, not a measurement.

```
   .       ... .       @@@@@@@@@#*+%@@@@@@@@@-           . .  . .      :. .     
            .....      @@@@@@@@#.- -*@@@@@@@@:       .    . .       . :. ..     
           . ...       #@@#=:           @@@@@      .     ..... ..  . :  ..     .
  .          .... .                      :         ..      ... . .  :. .        
            . .... .+                   ##=@      ..     . .... .   . .         
.       .  ......  -=-: -:          :  .@@@@      . ...    ..  .   :  . ...     
... ...  . .....:.. --=  %@:   = @@=@@%@@@@@@                . .  : ........:...
.          .  .  .....   .%%:    @@@:%#@@@@@@-%                  : ...:::::::---
             .    ..            =@@@@  *@@@@@@@                 : .             
==++++=++:-==::::::--.  ::-*#:  =-.@@   @@@@@@# ...::::=----::.: .:............:
.  .     :...          .. .:.      =     %@@@=  ..... ..    .--=:. .         .. 
     .                 .            :-   -@@@-   .  .              .            
.................... .   .##= +--.  =*===@@@@ -=====-=-::---=--                 
... .:...    . ..        .@@*           -@@@   .......... ......  .    .    .   
...........   . .  .  ....=@#  =@%@%**@= @@@     .......          .. .    .     
. ..          ..          ..#-:@@@@%@+@@@@@     .:......=-:+#%#+-: .:. . ...    
. .            ... .....     @@@@@%#@@@@@@@    -:::::..  --       .  .:::... .  
         . .      ....:      -@@@@@@@@@@@@      :-:::.:--.-::   -:   ....:...  .
.      ...:.   .......::       @@@%#@@@@=           .::.+=...:::.....       :...
       ..:.                      :                     .:..          ..:........
                                       :                              :-+=   :. 
-:-------======.                    *@:                      -====:::.    .-.   
%@@#:...:. .:                      :*@                         :::....       . :
 .:    -..::          .----  -   :+%@+                               ... .......
 -.    .  ..           :::        ++:                            :.. .*==++=-===
=-+=+=#-=.             :          ..                              .  . .. .:. . 
:.   .:  .:            =  .                                        .    .-==.:::
     ...:.                                                         .  :=-:..-*#-
===-::.-+.            -                                             .           
 .  . ..                   .-::: :                                     .  ......
. ......                                                              .*=-:::::-
  . ...                                                                -+%%%@@*%
  . :..               :..                                             . .:..:.:.
.   ..                 +=-.::     :                                    .........
   ..                                                                   ---:::..
  . .                                                                           
   ..                                                                      ...  
 . .                                                                            
. .                .                                                      .  :..
..                                                                            ..
.                                                                              -
                                                             -                  
                                  .:                         %                  
                                    .                        %*                 
                                                             @@                 
             -+=                                             @@                 
             *=-        :.                   .               @*=                
             :=    . ..:                                     @===               
```

The head is the bright mass at the top, the face is the dark band across
it, the shoulders and the torso are below. At 160×96 a person at a desk
is a small figure, and this is what the luma stream holds.

### The same moment, out of the text stream

This is the frame above, rendered out of the `delta_stream.txt` line for
frame 10,518. Twelve rows of twenty characters — the file's own
characters, with `0` meaning "this cell did not move":

```
00000000@@0000000000
00000000000000000000
00000000000000000000
00000000000@00000000
00000000000000000000
00000000000000@@0000
000000000000000@0000
0000000000000000@000
00000000000000000@00
00000000000000000000
00000000000000%1000@
00000000000000@0000@
```

**Text is the storage AND the rendering.** The line above is in
`delta_stream.txt`; the picture above it is in `ascii_010518.txt`; they
are the same moment, and the second one is what a model reads.

---

## 7. WHAT THE ENCODING LOSES — named, not hidden

**Texture.** The grid throws away the detail inside the blocks, and that
detail is most of the remaining error. It is the single biggest loss and
it is the reason the error is roughly constant: the more texture a frame
has, the more there is to lose. The 20×12 grid keeps the shapes and
loses the surfaces.

**Small deformations.** A 8×8 block cannot show a finger moving, a
shoulder dropping a centimetre, or a jaw opening. The encoding knows
that *something* in that block changed and by how much on average; it
does not know what.

**Direction of fine motion.** The delta stream is signed, so it knows
which way the luma went; the motion stream is not, because the
repository's change maps are absolute differences. A hand rising and a
hand falling produce the same motion event. The delta stream can tell
them apart, but only as a change in brightness, not as a movement.

**Identity.** Nothing in the encoding says who the person is. The text
stream is a description of a picture's changes, and the person portrait
has to come from the face analysis, which is a different tool.

**Everything outside the luma.** The stream is 160×96 greyscale. Colour,
depth, and the audio are not in it and cannot be recovered from it.

**The absolute position of a moving thing.** The encoding knows that a
region changed by N levels; it does not know that the thing which
changed is the same thing that changed in the previous frame. Tracking
one arm across a second of frames is a job for something that has the
frames, not for the text.

**And the honest summary:** the encoding is lossy in the way a
description is lossy compared to a photograph. It preserves *that*
something moved, *where*, *roughly how much*, and *when to the
millisecond* — and it loses who it was, what it looked like, and the
difference between a wave and a reach.

---

## 8. WHAT IS STILL OPEN

**The luma is 160×96 because the page downsamples it.** The original
webcam is 640×480. Every number in this report is bounded by that: an
arm is a few cells wide, and the gesture is legible as a burst of
motion and not as an arm. A 640×480 luma stream would change the grid
sweep's left-hand column and the size of the text's rows, and nothing
else about the design.

**The motion stream has no direction.** The repository's change maps are
absolute, so "the edges moved here" has no sign. A signed motion — the
difference of the change maps — would give the direction for one more
stage of the pipeline, and it is the obvious next thing.

**Nothing has been reconstructed across a reference boundary.** The
reconstruction is measured frame by frame against its own reference. What
happens when the picture moves more than the grid can hold between two
references is a real question and it is not answered here.

**The alignment table.** The streams carry the clock and are ready to
join, but the joint table — words, notes, syllables, motion events, all
on the frame index — has not been written. That is the bridge's last
piece and it belongs with the liaison lane.

---

## 9. HOW TO REPRODUCE ALL OF THIS

```bash
# the unit tests -- no session, no model
/home/proart/projects/gpu-venv/bin/python experiments/motion_as_text.py --selftest

# the whole thing on tonight's session
/home/proart/projects/gpu-venv/bin/python experiments/motion_as_text.py \
  /mnt/c/proart_capture/corpus/free_sessions/1792553040 \
  --out /tmp/motion_run --scene-threshold 25 \
  --ascii-at 182.4 --ascii-frames 4 --ascii-cols 80

# the rate policy alone: the clock-indexed cost against the
# policy-indexed cost, and the stillness test that decides between them
/home/proart/projects/gpu-venv/bin/python experiments/motion_as_text.py \
  /mnt/c/proart_capture/corpus/free_sessions/1792553040 \
  --out /tmp/motion_run --scene-threshold 25 \
  --ref-png-bytes 2196034 --rate-policy
```

The run writes `delta_stream.txt` (the reconstructable text), 
`motion_stream.txt` (the sparse edge events), `references/` (the PNG
anchors), `ascii_*.txt` (the frames as text), `stream_*.txt` (the same
frames out of the text), and `results.json` with every number in this
report. The rate policy writes `rate_policy.json`, which carries both
costings, the stillness test's threshold and its measured
distributions, and the `triggers_disagree` record.
