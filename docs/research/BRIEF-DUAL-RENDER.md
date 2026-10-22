# BRIEF: DUAL RENDER — the rendering play for tonight's dual capture

*Staged 2026-10-21 evening by the mcode seat. Companion to
`docs/research/BRIEF-DUAL-CAPTURE.md` (Casey's brief for the capture
itself). This is the lane that draws the two views once the pair has
landed, and it is the lane that tells zcode what it has to hand over.*

## 0. What this lane owns, and what it does not

Casey's brief puts three analyses on the dual capture: a TDOA position
track, a who-hears-what map, and **the renders**. The renders are this
lane's. The position track and the who-hears-what map need the two
audio tracks and the inter-mic delay, and the audio is zcode's.

**zcode owns the capture's liveness and the clap alignment. This lane
CONSUMES the alignment and never computes it.** The claps are the sync
events; this tool takes them as they are, fits the offset and the drift
from them, and prints what it used. If there is no sidecar, the tool
assumes the two streams started together and **says so in every single
output**, because an assumed alignment that is mistaken is a lie told
in a monospace font.

The three renders, all kept, all positional:

1. **THE DUAL-ASCII, SIDE BY SIDE.** The same moments from both views,
   one panel under the other, side by side.
2. **THE FUSED SCORE.** Two staves over time, one per channel, in
   which each channel **annotates the other** with a witness mark.
3. **THE DIFFERENCED VIEW.** What moved in the fixed frame, the roving
   frame that explains it, and the difference between them.

And the number that goes with the renders, because a picture of two
views is not yet a finding:

4. **THE WITNESS COUNT.** The score's marks are a 2×2 table of the same
   minutes seen two ways — both views moved, only the fixed moved, only
   the roving moved, neither. The number worth having is the **excess
   over chance**: if the two views were independent they would both
   move in `p_fixed × p_roving` of the slices, and the difference
   between that and what they actually do is the agreement. That
   agreement is not two sensors agreeing about a number; it is two eyes
   agreeing about a **movement** — the same person, seen from two
   places, at the same moment. It is the thing two sovereign logs of
   the same minutes give that neither gives alone.

   **AND WHEN THE BINARY COUNT IS DEGENERATE, A GRADED ONE.** The
   2×2 table throws away everything but the sign, and it is blind to a
   channel that is on most of the time: on the duet the roving view
   moved in 236 of 236 slices, so the excess over chance was exactly
   +0.0 points and no chi-square could be formed. The tool therefore
   also computes the **graded** reading — the correlation of the two
   channels' change energies — against a **slide null**: the roving
   channel is slid against the fixed one and the correlation recomputed
   at every offset, because the null here is the autocorrelation of two
   slow series rather than the correlation of two noises.

5. **THE CROSSING WITNESS.** The corpus already has the right
   instrument and this lane had not used it. A **crossing** is a
   connected blob of change that holds its shape while it
   **translates** — the person is one object, it keeps its size, and
   it moves. The grain is below the floor, a highlight is one frame, a
   fidget does not travel, and neither does a shake. So each view is
   asked the question it can answer: did a *thing* move across the
   picture? On the duet this is the only witness that produces a
   reading at all.

   **THE GAP IS COUNTED IN FRAMES, NOT IN SECONDS.** The corpus's
   detector took a flat 0.35 s, which is 21 frames at the 60 fps the
   corpus was captured at and **0.70 frames** at the 2 fps it is
   captured at now — so at 0.70 frames no blob ever chains and the
   walk-by study finds **nothing at all** in those sessions. The gap
   is now the corpus's own rule, `max(0.35 s, 1.5 x the session's own
   frame interval)`: the corpus's 92 fast sessions stay at exactly the
   **739** crossings they already had — so no filed result moves — and
   its 11 slow ones go from **2** to **25**. A defect every 2 fps
   session in the corpus has been carrying.

## 1. The two rules the renders do not break

**THE PANELS ARE ALIGNED IN TIME, NOT IN SPACE.** The fixed camera sits
on a stand and sees the room from one place; the roving eye is in
Casey's hand and sees the room from another. A cell in the left panel
and a cell in the right panel are **not the same part of the room**. The
grid is still positional — 40 by 24 cells over the 160 by 96 luma, four
pixels per cell, the corpus's own geometry — because a cell that moved
is a cell that moved *where it moved*. What the panels share is a
**moment**, not a place, and every output says so in its header.

**THE SCALE IS SHARED BETWEEN THE VIEWS.** A side-by-side of two
different quantisation scales says nothing about which view saw more,
and the reader is owed a comparison they can trust. The default is one
scale for both. The per-view scale is available (`--scale per-view-auto`)
and every output that uses it says the panels are **not comparable**.

**AND WHEN THE TWO SENSORS ARE IN DIFFERENT UNITS, THE SHARED SCALE
IS THE WRONG ONE.** The duet's two change maps are not the same
quantity, and one scale renders the ProArt's panel **black**. The
answer is `--scale floor-relative`: each view is divided by that
view's **own noise floor** and the two are then scaled together, so a
cell at the top of the ramp means the same thing on both sides — this
many times this sensor's own grain. The shared-scale render is still
filed, because the black panel is the honest picture of the mismatch.

## 2. What zcode hands over: THE SIDECAR

The renderer takes the alignment as a JSON sidecar, in either of two
forms.

**Form A — the clap pairs (preferred).** The raw evidence:

```json
{
  "claps": {
    "fixed_ms": [11834.7, 31834.7, 51834.7],
    "roving_ms": [5120.0, 25120.0, 45120.0]
  },
  "the_clock": "frame_offsets_ms"
}
```

**Form B — the fitted alignment.** When the fit is already done:

```json
{"offset_ms": -120.0, "drift_ppm": 0.0, "n_claps": 3}
```

**Form C — the same clap pairs, with the audio-to-frame skew.** A clap
is detected in the **audio**. The sidecar wants it in the **frame
clock**. Those are not the same clock: the corpus has already measured
the audio-to-video offset inside one capture at about **114 ms**, so
the conversion is not free — and the two streams need not share the
same skew. The sidecar may carry it and the renderer will apply it and
say so:

```json
{
  "claps": {"fixed_ms": [...], "roving_ms": [...]},
  "audio_to_frame_ms": {"fixed": 114.0, "roving": 0.0}
}
```

With the field absent, the clap times are taken as already being in the
frame clock. **Either way the output says which it assumed**, because
an unstated assumption in a clock is how a minute gets silently
misaligned.

**THE CLOCK IS THE TOOL'S BUSINESS, NOT THE SIDECAR'S.** A clap time is
a time in **one stream's clock** — the browser's `performance.now()`
for the ProArt, the video's own presentation time for the phone — and
those two clocks do not share an origin. So:

- Each list in Form A must be in **that stream's own
  `timestamps.json` `frame_offsets_ms` clock**. That is the list the
  bundle already carries; the clap's time is the timestamp of the frame
  the clap is in.
- The renderer **re-bases both sides** onto each stream's own first
  frame before it fits, and says in the output that it did.
- This matters because the ProArt's base is **not zero**: its sessions
  carry `performance.now()` minus the pre-roll base, and **all 92 of the
  corpus's sessions with timestamps have a non-zero base** — from
  **1259.9 ms** to **1534177.0 ms** (25 minutes). A sidecar written in
  that clock needs the base taken off, and the tool does it.
- If the sidecar's times are already relative to each stream's first
  frame, say so with `"the_clock": "rebased"` and they are taken at
  their word.

One clap pair gives an **offset and no drift** — the tool says so
rather than fitting a drift from one point. Three or more claps fit
both, and the output prints the **residual** (the largest error of the
fit) so a bad alignment is visible rather than hidden.

**THE CLAP TIMES SHOULD BE SUB-FRAME, AND THE TOOL NOW MEASURES
WHETHER THEY ARE.** The route for two uncalibrated views is
audio-sync **first**, and **sub-frame**: an alignment quantised to the
video cannot be finer than the video. So the tool measures each clap
time's distance from **its own stream's frame grid** and prints, in
words, whether the sidecar carried **FRAME TIMES** or **SUB-FRAME
times**:

```
THE ALIGNMENT RESOLUTION: the fixed clap times sit ON that stream's
frame grid (at most 0.000 ms from it), so the alignment is quantised to
that stream's frame interval (16.667 ms) and cannot be finer; the roving
clap times are SUB-FRAME (up to 3.412 ms from that stream's frame grid,
whose interval is 406.500 ms), so the alignment is finer than a frame on
that side
```

**THE RESOLUTION OF A FRAME-QUANTISED SYNC IS SET BY THE COARSER OF THE
TWO STREAMS**, and on the duet that is the fixed side: the merged
ProArt stream's frame interval is **499.75 ms** and the phone's room-lens
change maps are **406.5 ms** (the keyframe channel **2105.0 ms**). A
sidecar written from frame timestamps would therefore be quantised at
**half a second** on this pair. The corpus's other sessions are 60 fps
(**16.667 ms**), so that is a property of this capture and not of the
fleet.

**AND A CLAP TIME THAT IS TAKEN FROM A FRAME IS A DIFFERENT KIND OF
CLAIM FROM ONE THAT IS INTERPOLATED.** A sidecar that names a frame is
claiming the clap happened somewhere inside that frame's interval; one
that carries **1012.345 ms** is claiming a time. The renderer does not
judge which is right — that is zcode's detection — but it says which
kind it was given, because a claim and a guess must not look alike in
the output.

**A SPURIOUS CLAP DOES NOT TILT THE ALIGNMENT.** One door slammed in
the wrong place must not shift the whole minute. The tool finds the
outliers by **search** — every pair of claps proposes a line and the
line with the most claves on it wins — drops them, refits on the
inliers, and **names the ones it dropped** in the output. The search
is necessary because least squares *spreads* a bad clap's error over
the good ones: with four claps and one 500 ms outlier, every clap ends
up over the residual bar and nothing looks like an outlier. The
threshold alone would not have found it. This is the same lesson the
arrival-null study learned in the audio, where one bad arrival took the
implied speed of sound from 343 to 306 m/s.

**THE OFFSET IS CHECKED AGAINST THE CLAPS THEMSELVES.** A clap matched
to the **wrong** clap does not give a small offset — it gives one of
about a whole clap spacing, because that is how far the wrong clap is.
So the bar is not a number somebody liked: it is **the distance between
the claps**. An offset bigger than half the clap spacing is called
**SUSPECT** in the output, in words.

## 3. What zcode hands over: THE ROVING STREAM

The phone's recording is ingested into the corpus's own bundle format
by `experiments/dual_ingest.py`, so one reader serves both sides and no
new format is invented:

```bash
/home/proart/projects/gpu-venv/bin/python experiments/dual_ingest.py \
    --video <phone.mp4> \
    --out docs/research/dual/roving \
    --label "Casey's phone (rear camera)" \
    --wall-clock-start 2026-10-21T20:04:11.000Z
```

The output is `luma.raw` (uint8 BT.601 full-range luma, 160 by 96,
frames concatenated), `timestamps.json` and `meta.json` — exactly what
the ProArt session bundles are, so `experiments/lens.py` and the
renderer both read them with no changes.

**THE BUNDLE IS FIRST-CLASS, AND THAT IS MEASURED.** The corpus's own
walk-by study needs the luma **and** the audio, and it reads the
phone's bundle with no changes — the ingest's selftest runs it. For
that it needs `audio.wav` in the bundle, and the phone's audio track
**cannot be decoded on this machine** (no ffmpeg, no `av`, no
`imageio`). So zcode extracts the audio where there IS a decoder and
hands the file over:

```bash
/home/proart/projects/gpu-venv/bin/python experiments/dual_ingest.py \
    --video <phone.mp4> --out docs/research/dual/roving \
    --audio <phone.wav> --label "Casey's phone (rear camera)" \
    --wall-clock-start 2026-10-21T20:04:11.000Z
```

The audio is put in the corpus's own format: 16-bit PCM mono at the
recording's own rate, no resampling and no processing. A bundle with
no audio **says so** in its metadata rather than pretending to be
complete.

**`--wall-clock-start` is the only exact way to anchor a phone
capture.** A phone recording carries no wall clock of its own. Without
the flag the ingest falls back to the file's modification time, which
is **an approximation** (the file may have been uploaded hours later),
and the metadata labels it as one.

The picture is **cut to the corpus's aspect before it is resized**
(`--fit crop`, the default), because the grid is positional: a
stretched picture moves the body's cells around. The crop rectangle is
in `meta.json` so anyone can re-derive where the picture came from.

The ingest **measures the phone's own noise** rather than inheriting
the ProArt's 3.0 luma floor, because a phone in a hand is a different
sensor. The statistics are in `meta.json` under `the_noise_of_this_sensor`.

### 3a. THE PHONE WAS NOT A VIDEO: THE ROOM-LENS PATH

**THE DUET'S PHONE IS NOT A VIDEO, AND THAT CHANGES THE CONTRACT.**
The capture that landed (`~/dual-capture/dual-3.json`) is zcode's
room-lens session: one string of **CHARACTERS** per frame, with the
device's pose attached, and the grid's geometry **changing during the
session** (it alternates between 128×125 and 96×93 as the lens works).
There is nothing to decode with OpenCV and no aspect to crop — the
picture is already a grid — and the video ingest above does not apply.

So the ingest has a second path, and it writes a bundle the renderer
reads with the same reader:

```bash
/home/proart/projects/gpu-venv/bin/python experiments/dual_ingest.py \
    --room-lens ~/dual-capture/dual-3.json --out /tmp/duet-3 \
    --label "the phone room-lens (the roving eye)" \
    --fixed-label "the ProArt, merged" \
    --fixed-session 1792655768 --fixed-session 1792656069 \
    --fixed-session 1792656077 --fixed-session 1792656100 \
    --fixed-session 1792656118 --fixed-session 1792656155 \
    --fixed-session 1792656183 --fixed-session 1792656235
```

It writes `change.raw` (the change maps, already on the corpus's
40×24 grid), `luma_grid.raw` (the picture, on the same grid),
`timestamps.json` and `meta.json`, and it writes the **merged fixed
stream** and the **wall-clock sidecar** beside them.

**THE ROOM-LENS'S CLOCK IS NOT WHERE `meta.startedAt` SAYS IT IS.**
The room-lens stamps every frame `t = Date.now() - walkStart`, and the
session's `startedAt` is written when the **session** is created — which
can be minutes before the walkabout. The duet's session was created
**435.840 s** before the walk began. So the anchor is **derived** from
the bundle's own stamps: five audio parts and four walk-time room
measures each carry a wall clock and an offset from the walkabout's
start, and **all nine agree to 0.0 ms**. The tool files the estimators
and their spread, and says so when it has to fall back to
`meta.startedAt` because there is nothing to check it against.

**THE MERGE IS VERIFIED, NOT ASSUMED.** The ProArt's sessions share the
capture page's clock, and the page base is recoverable as
`wall_clock_start - frame_offsets_ms[0]`. The duet corpus's eight
sessions agree on that base to **1.2 ms across 12.8 minutes**, and the
tool **refuses** a merge whose bases disagree by more than 250 ms,
because merging two clocks invents a timeline.

**THE PHONE'S FLOOR IS ITS RESOLUTION, NOT A GRAIN ESTIMATE.** The
room-lens is a *deterministic renderer* of a noisy sensor: its change
map is quantised to brightness bands, so below **one band** a change
cannot be expressed at all — the graded fixture shows 12.8 % of its
pairs at **exactly zero**. The phone's floor is therefore one band
(15.9375), and the share of its frames that show any change at all —
its **duty cycle** — is measured and reported, because a channel that
is on most of the time cannot time a movement.

### 3b. THE KEYFRAME CHANNEL: THE PHONE'S OWN CAMERA LUMA

**THE ASCII IS THE LOSSY STEP, AND THE BUNDLE ALREADY HAS WHAT IS
BEHIND IT.** The room-lens computes a luma field for its own ASCII and
then posterises it, and it banks **103 keyframes** of the camera's
actual picture (320×569 JPEG). Those are the **camera's luma**, which
is the very quantity the ProArt's change map is in.

So the ingest builds a **second bundle** from them, in the corpus's own
format: 160×96 luma, cropped to the corpus's aspect rather than
stretched, with the corpus's own `timestamps.json` and `meta.json`. So
`experiments/lens.py` computes this channel's change map with **no
changes at all**, and the floor is the corpus's own 3.0 because the
construction *is* the corpus's. The two views are commensurable in
that bundle for the first time, and the render is filed beside the
others as `DUET-RENDER-KEYFRAMES.txt`.

**A NOISE FLOOR IS A PROPERTY OF A SENSOR AND AN INTERVAL.** The
keyframes are one per 2.105 s — 4.2× the ProArt's cadence — and the
corpus's 3.0 luma floor, measured on 0.5 s frames, is **saturated** at
that interval: the keyframe channel's duty cycle is 100 % at 3.0 and
89.2 % at its own measured floor of 94.938. The corpus's floor is a
property of the corpus's interval, and borrowing it across a 4× gap is
borrowing a threshold that has no meaning.

## 4. The invocations, ready to run

**Is the pair renderable?**

```bash
/home/proart/projects/gpu-venv/bin/python experiments/dual_render.py \
    --check \
    --fixed /mnt/c/proart_capture/corpus/free_sessions/<session> \
    --roving docs/research/dual/roving \
    --sync docs/research/dual/claps.json
```

A missing stream is a **precise blocker**, never an absence: the
output names what is missing.

**Draw the three renders.**

```bash
/home/proart/projects/gpu-venv/bin/python experiments/dual_render.py \
    --fixed /mnt/c/proart_capture/corpus/free_sessions/<session> \
    --roving docs/research/dual/roving \
    --sync docs/research/dual/claps.json \
    --out docs/research/dual \
    --mode change --scale shared-auto --map mean \
    --window 0.25 --slice 0.25 --moments 6
```

It writes `docs/research/dual/DUAL-RENDER.txt` (the render itself) and
`docs/research/dual/dual_render.json` (the numbers behind it).

**More than one way, all kept.** `--mode {change,luma,both}` is the
picture; `--map {mean,max,sum}` is the window statistic; `--scale
{shared-auto,shared-floor,shared-fixed,per-view-auto,floor-relative}`
is the quantisation. **Every one of those choices is printed in the
output**, because a rendering choice is not a measurement and the
reader is entitled to know which one was made.

**`--scale floor-relative` IS THE ONE FOR TWO SENSORS IN DIFFERENT
UNITS.** It divides each view by that view's **own noise floor** and
then uses one scale, so a cell at the top of the ramp means the same
thing on both sides: this many times this sensor's own grain. It is
the only comparison the duet makes possible, and the filed headline
render uses it. The `shared-auto` render is filed beside it, because
its black ProArt panel is the honest picture of the units mismatch.

**THE SLICE MUST EXCEED THE FRAME INTERVAL.** At 2.0 and 2.5 fps a
0.25 s slice holds at most one frame from each view, and the fused
score's slices are better at **1 s**. The window and the slice are
printed in the output, and the number of frames each window actually
held is printed beside each moment.

**KEEP ALL.** Every moment in the overlap is rendered. Nothing is
filtered for being dull.

## 5. What is already measured, and what is not

**Measured, on the tool's own fixture.** A synthetic pair — a person
walking a known path, seen by a fixed witness and a roving eye with a
known viewpoint transform — so the renders can be *graded* instead of
admired. The selftest is **65 of 65** and it grades:

- the fixed view's change centroid lands within **0.10** of the truth's
  position on both axes;
- the roving view lands more than **0.08** away from it in two
  dimensions, which is the geometry the two panels are supposed to
  have;
- the witness vocabulary is exercised — the fixture gives the fixed
  camera a **field of view** (room x from 0.25) so the person walks out
  of it while the phone still sees them, and the score's marks go
  `=` to `<` and back;
- the alignment: a 500 ppm drift is recovered as 500 ppm, one clap
  gives an offset and no drift, and the sidecar's clock is re-based
  onto each stream's first frame;
- **a spurious clap**: with four claps and one 500 ms outlier the
  offset still comes back as **−250.000 ms**, the outlier is **dropped
  and named**, and four good claps drop nothing;
- **the offset's plausibility**: a clap matched to the wrong clap is
  called **SUSPECT** and a sensible offset is not;
- **the audio-to-frame skew**: when the sidecar carries it, it is
  applied and the output says it was applied;
- **the alignment's resolution**: a sidecar whose clap times land on the
  stream's own frame grid is reported as **FRAME TIMES** and an
  alignment quantised to that stream's frame interval, and a sidecar
  whose clap times miss the grid by **12.345 ms** is reported as
  **SUB-FRAME**; both are in the output and not only in the code;
- the witness count adds up, its chance baseline is the product of the
  two rates, and its excess is signed.

**Measured, on the ingest.** The phone-shaped fixture (portrait, 270 by
480, 30 fps) lands in the corpus's shape at **30.000 fps** with the
container's own per-frame timestamps; its change map finds the fixture's
square to within **8 pixels** of where the fixture put it; its audio
lands in the corpus's own format (16-bit PCM mono at the recording's
own rate); and the corpus's own walk-by study — which needs the luma
**and** the audio — reads the phone's bundle with no changes.

**Measured, on a real corpus session.** The rehearsal
(`docs/research/dual/REHEARSAL-DUAL-RENDER.txt`) pairs a **real** ProArt
session (1792553329) with a **synthetic** phone recording of
**meaningless** content, ingested through the new tool. It is a
**negative control**, and it behaves like one:

- the injected **−175.000 ms** offset comes back as **−175.000 ms**
  with a **0.000 ms** residual;
- the two streams have nothing to do with each other, and the witness
  count says so: they both move in **29.4 %** of the 68 slices against
  a chance baseline of **30.4 %** — an excess of **−1.0 points**,
  chi-square **0.13** on 1 degree of freedom, **p = 0.72**.

That is the control's whole job: **the measure does not manufacture
agreement.** A witness count that found an excess between a real room
and a meaningless video would be a bug in the measure, not a finding.

**The selftests are 65 of 65** for the renderer and **29 of 29** for the
ingest.

**Measured, on the REAL pair.** The duet has landed and been rendered.
`docs/research/dual/DUET-RENDER.md` is the report and the three
`DUET-RENDER-*.txt` files are the renders. What it says:

- the phone's walkabout start is **07:59:28.268Z**, recovered from
  **nine independent stamps that agree to 0.0 ms**, and the session's
  own `meta.startedAt` is **435.840 s** earlier — believing it would
  have moved the whole duet and paired the phone with the wrong
  ProArt session;
- the ProArt's eight sessions merge into one stream because their
  recovered page bases agree to **1.2 ms** over 12.8 minutes;
- the shared minutes are **235.93 s**, aligned on the **wall clock**
  (+500538.0 ms), which is labelled as the wall clock in every output
  because the two devices' clocks have never been compared;
- the two change maps are **not the same quantity** — the ProArt's is
  in the camera's luma, the phone's in the room-lens's own brightness
  (a 2.2× contrast stretch, clipped, with **eight of seventeen**
  glyphs all `@`);
- the **witness count is DEGENERATE, not weak**: the roving view moved
  in **236 of 236** slices, so the contingency table has a constant row,
  the excess over chance is exactly **+0.0 points** and no chi-square
  can be formed. The cause is the phone's **99.8 % duty cycle**;
- the **graded** reading (the correlation of the two channels' change
  energies) is **+0.2553** against a **slide null** — the roving
  channel slid against the fixed one — whose own maximum is at **−44 s**
  and whose region within one null sd spans **58 s**. The duet
  therefore **cannot measure its own skew**, and the claps are the only
  way.

**NOT measured, and not to be filed as if it were:**

- **The clap alignment has never been measured on real claps.** The
  offset in the rehearsal was injected, not detected, and the duet's
  alignment is the **wall clock**. zcode's detection (cross-correlating
  the phone's opus against the ProArt's WAV) has not been run, and the
  phone's **audio track cannot be decoded on this machine** — there is
  no ffmpeg, no `av`, no `imageio` here. That is zcode's lane and it
  needs a different machine or a different decoder.
- **The duet has not yet shown the synergy the brief is after.** The
  two channels' energies are correlated at +0.2553 (raw) and +0.3599
  (log), which is suggestive and not significant against the honest
  null. The two causes are measured and both are fixable: the phone's
  channel is saturated because the room-lens posterises its own luma
  before the change is measured, and the alignment is unverified.
- **The device-to-device latency bias is still unmeasured on hardware.**
  The protocol and the calculator are filed
  (`docs/research/TWO-DEVICE-LATENCY-PROTOCOL.md`); the corpus holds one
  real device. The duet's wall-clock alignment carries exactly this
  unmeasured constant.

## 6. The three things that were wrong in the first build

Recorded because the fix is the point, and because the same three traps
are waiting for the next person who writes a renderer.

**A FIXTURE THAT PASSES FOR THE WRONG REASON.** The first synthetic
pair re-randomised its background on every frame, which flooded the
change map with noise: the fixed view's change centroid came out at
the centre of the frame for *every* moment. The check "the fixed view
finds the person where the truth puts them" **PASSED** — because the
truth happened to put the person at x = 0.5 at the moment it asked
about. A test that passes because everything is at the centre is not a
test. The background is now static, the grain is the only thing that
moves on its own, and the question is asked at a moment when the person
is off the centre and walking.

**A FIXTURE WHOSE BASE IS ZERO HIDES THE BUG.** The alignment bug
above was invisible in the selftest because the synthetic pair's
timestamps started at zero, and every one of the corpus's 92 sessions
starts at something else. The fixture now carries a **non-zero base on
purpose** (6834.7 ms and 120.5 ms, the first of them the real session's
own base).

**A SLOPE WRITTEN AS A DRIFT.** The fitted drift was written as `a *
1e6` where the drift is the slope's **departure from one**, so a 1 ppm
drift would have been reported as 1e6 ppm and the slope would have
doubled on the real pair.

And one that is not a bug but a decision that had to be made: **the
window statistic is the MEAN change per frame, not the sum.** The two
cameras will not run at the same frame rate, and a sum over the window
hands the faster camera a free factor of speed. The corpus's own 3.0
luma noise floor was measured on a single frame's change map, so it is
a threshold on a mean. The **witness** question is asked of the
**maximum** over the window instead, because a thing that moved for one
frame in five still moved.

## 7. The rehearsal, reproduced

The rehearsal is a **negative control**: a real ProArt session beside a
phone recording of meaningless content. It is reproducible with four
commands, and every number in
`docs/research/dual/REHEARSAL-DUAL-RENDER.txt` comes from them.

```bash
# 1. the meaningless phone recording (portrait, moves for one second and
#    rests for one, so half the slices have movement)
python experiments/dual_ingest.py --make-fixture /tmp/rehearsal/phone.mp4

# 2. ingest it into the corpus's bundle format
python experiments/dual_ingest.py --video /tmp/rehearsal/phone.mp4 \
    --out /tmp/rehearsal/roving --label "the rehearsal phone"

# 3. the sidecar: three claps, the roving's clock 175 ms ahead
#    (the files are in docs/research/dual/rehearsal_claps.json)

# 4. the renders
python experiments/dual_render.py \
    --fixed /mnt/c/proart_capture/corpus/free_sessions/1792553329 \
    --roving /tmp/rehearsal/roving \
    --sync docs/research/dual/rehearsal_claps.json \
    --out /tmp/rehearsal/out --moments 2 --slice 0.25 \
    --note "NEGATIVE CONTROL, NOT THE REAL PAIR"
```

## 8. THE ROUTE, AND THE SCALE THAT IS NOT HERE YET

**THE ROUTE FOR TWO UNCALIBRATED VIEWS HAS THREE STAGES, AND ONLY THE
FIRST OF THEM IS THIS LANE'S.**

1. **AUDIO-SYNC FIRST, AT SUB-FRAME PRECISION.** Two uncalibrated
   cameras have no common clock, so the only thing that puts them on one
   clock is a sound both of them heard. Sub-frame matters: an alignment
   quantised to the video is as coarse as the video. This is zcode's
   detection and this lane's consumption, and §2 now says which kind of
   sidecar it was given.
2. **THEN A RELATIVE POSE** — a DUSt3R/MASt3R-style relative
   reconstruction — **which is SCALE-FREE.** It says where the two
   viewpoints sit relative to each other and says nothing about how big
   anything is.
3. **AND THE SCALE COMES FROM THE RANGING/TDOA SIDE**, which is zcode's
   lane and not this one.

**SO THE FUSED RENDERS STAY IN RELATIVE GEOMETRY, AND NO SCALE IS BAKED
IN.** A guessed scale is not a small error in a render; it is a number
the reader cannot tell from a measured one, and it enters every
distance, every width and every position at once. This lane will not
invent one. What it has is the **temporal** alignment, and a temporal
alignment is already a relative quantity — a moment, not a place.

**AND THE PART WORTH HAVING IS THAT THE THREE STAGES HAVE INDEPENDENT
ERROR SOURCES.** The arrival-null study measured the TDOA estimator: it
resolves a delay to **0.004 ms**, which is **0.0014 m** of range, with a
null bias of **0.0002 ms**. And it found the thing that makes the scale
route independent of the sync route: **a TDOA measurement compares the
two signals to each other, so the offset between the two clocks cancels
exactly** — two signals with no clock relationship between them came
back 7.0 ms apart when they were 7.3 ms apart. So the scale can be
measured on a pair that is not synchronised at all, and a mistake in
the sync cannot corrupt the scale. That is the architecture this route
asks for and it is already measurable on this lane's own terms.

**WHAT IT MEANS FOR THE CONCRETE DELIVERABLE.** The three renders are
temporal, and they are already relative: a cell in the left panel and a
cell in the right panel are not the same part of the room, and every
output says so in its header. When a relative pose lands it lands
**scale-free**, and the scale lands with the TDOA. Until then the
renders carry no metric claim, and the duet's own alignment stays
labelled as the **wall clock** with its unmeasured device-to-device
skew.

**AND THE DUET IS A GOOD PLACE TO SEE WHY THE ORDER MATTERS.** Its two
streams are about **2 fps** — the fixed side's frame interval is
**499.75 ms** — so a frame-quantised sync on this pair would be
quantised at half a second, while the corpus's 60 fps sessions would
give **16.667 ms**. The sync route is the one that has to come first
and the one whose resolution depends on the capture.

## 9. The files

| file | what it is |
|---|---|
| `experiments/dual_render.py` | the renderer: the three renders, the witness count, the graded witness, the alignment consumer, the pair checker, the synthetic fixture, the selftest |
| `experiments/dual_ingest.py` | the phone recording → the corpus's bundle format, the room-lens bundle, the verified merge, and the two fixtures |
| `docs/research/dual/DUET-RENDER.md` | **the report on the real pair** — what the duet does and does not show, and why |
| `docs/research/dual/DUET-RENDER-FLOOR_RELATIVE.txt` | the real duet's three renders, each panel in times its own sensor's grain |
| `docs/research/dual/DUET-RENDER-SHARED_AUTO.txt` | the same, on one shared scale — the ProArt's panel is black, and that is the point |
| `docs/research/dual/DUET-RENDER-PER_VIEW_AUTO.txt` | the same, each view against its own maximum — legible, and not comparable |
| `docs/research/dual/DUET-RENDER-KEYFRAMES.txt` | the phone's **camera luma** (the keyframes) against the ProArt: the two views in the SAME units, so one scale is honest |
| `docs/research/dual/duet_render_*.json` | the numbers behind the four real renders |
| `docs/research/dual/duet_ingest.json` | what the ingest measured on the room-lens: the clock anchor, the geometry runs, the duty cycle |
| `docs/research/dual/duet_phone_meta.json` | the phone bundle's metadata: the ramp, the tiers, the pose ranges, the anchor estimators |
| `docs/research/dual/duet_keyframe_meta.json` | the keyframe channel's metadata: the crop, the interval, the corpus's own floor |
| `docs/research/dual/SYNTHETIC-DUAL-RENDER.txt` | the renders of the synthetic pair — the grade |
| `docs/research/dual/REHEARSAL-DUAL-RENDER.txt` | the negative control: a real corpus session beside a meaningless phone |
| `docs/research/dual/synthetic_dual_render.json` | the numbers behind the synthetic render |
| `docs/research/dual/rehearsal_dual_render.json` | the numbers behind the rehearsal render |
| `docs/research/dual/rehearsal_ingest.json` | what the ingest measured on the rehearsal phone recording |
| `docs/research/dual/rehearsal_claps.json` | the rehearsal sidecar, in the form zcode should use |

Run the selftests:

```bash
/home/proart/projects/gpu-venv/bin/python experiments/dual_render.py --selftest
/home/proart/projects/gpu-venv/bin/python experiments/dual_ingest.py --selftest
```

Staged for Muse to review; not merged.
