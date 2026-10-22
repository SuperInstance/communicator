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
{shared-auto,shared-floor,shared-fixed,per-view-auto}` is the
quantisation. **Every one of those choices is printed in the output**,
because a rendering choice is not a measurement and the reader is
entitled to know which one was made.

**KEEP ALL.** Every moment in the overlap is rendered. Nothing is
filtered for being dull.

## 5. What is already measured, and what is not

**Measured, on the tool's own fixture.** A synthetic pair — a person
walking a known path, seen by a fixed witness and a roving eye with a
known viewpoint transform — so the renders can be *graded* instead of
admired. The selftest is **37 of 37** and it grades:

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

**The selftests are 37 of 37** for the renderer and **16 of 16** for the
ingest.

**NOT measured, and not to be filed as if it were:**

- **The real dual pair has not been captured.** Every render in this
  repository is of a synthetic pair or a rehearsal. The tool has never
  seen two real cameras of the same minutes.
- **The clap alignment has never been measured on real claps.** The
  offset in the rehearsal was injected, not detected. zcode's detection
  (cross-correlating the phone's opus against the ProArt's WAV) has not
  been run, and the phone's **audio track cannot be decoded on this
  machine** — there is no ffmpeg, no `av`, no `imageio` here. That is
  zcode's lane and it needs a different machine or a different decoder.
- **The device-to-device latency bias is still unmeasured on hardware.**
  The protocol and the calculator are filed
  (`docs/research/TWO-DEVICE-LATENCY-PROTOCOL.md`); the corpus holds one
  real device.
- **The two views have never been compared on a real room**, so nothing
  is known yet about how the phone's change map and the ProArt's change
  map relate — only that they are different sensors with different
  noise.

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

## 8. The files

| file | what it is |
|---|---|
| `experiments/dual_render.py` | the renderer: the three renders, the witness count, the alignment consumer, the pair checker, the synthetic fixture, the selftest |
| `experiments/dual_ingest.py` | the phone recording → the corpus's bundle format, and the rehearsal fixture generator |
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
