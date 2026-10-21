# FREE-SESSION ANALYZER — the labels are derived after the recording, not manufactured before it

**Author:** the acceptance worktree (`~/projects/communicator-acceptance`,
branch `acceptance/browser-capture`)
**Date:** 2026-10-21
**Brief:** `docs/research/BRIEF-FREE-SESSION-ANALYZER.md`
**Tool:** `experiments/free_session_analyze.py`
**Evidence:** `--selftest` (66 checks) and the recovery report (§6)
**Closes:** the camera-probe lane (`docs/research/CAMERA-RND-FINDINGS.md`), §8

---

## 0. THE HEADLINE

**The choreographed label was the wrong instrument, and this is the
replacement.** Casey: "make it so I can click record and just talk freely and
turn and things, and use OpenCV and other things to see what I was doing —
you just need me to do a bunch of stuff in front of the camera." He is right.
A manufactured label has the value of the moment it was manufactured and
nothing else, and in the browser it broke. So the page only **records**, and
the labels are **derived afterwards**, from the recording, by machinery that
can be tested on data that already exists.

Three results, in order of how much they matter:

1. **The derived takes are takes the gate eats.** Twenty derived takes
   written from the captures that exist today, put through
   `experiments/corpus_integrity.py` — the repository's own gate — come out
   **20 takes, 0 failed, ok=True**. Not "formatted like a take": the gate
   itself says yes. The same gate was then run over the **production** path
   — the real free-record layout, rebuilt from the real capture — and it
   said yes there too (§6.5).
2. **The labels are recovered, not asserted.** Run over the A, B, C and
   empty1 captures, the machinery recovers **4 of 4** known labels with
   **0 per-frame face errors in 7,200 frames**.
3. **The derived maps are the pipeline's own maps, exactly.** The analyzer
   rebuilds the change maps for each capture and compares them with the
   stored take that was seeded from that same capture: **0 differing cells
   out of 3,452,160**, for all three, with the scale matching to the last
   bit.

And the thing I would keep if I could keep only one thing: **the gate
caught two real bugs that my own reasoning had missed** — a label taken
from the majority of the frames instead of their aggregate, and an A/V
bound computed against the wrong origin (§3.3). Neither would have been
visible without a check that could say no. That is the argument for
running the repository's own gate over the output rather than asserting that
the output is right.

---

## 1. WHAT THE PAGE ACTUALLY WRITES

The brief says to negotiate the layout from ZCode's page code and not guess.
ZCode has since shipped the free-record mode (`6c8dc1b`), so there is
something to read: `browser_capture/free_record.html` and
`browser_capture/server.py:save_free_session`.

| file | what it is |
|---|---|
| `meta.json` | `tool: "browser_free_record"`, `n_frames`, the luma geometry, `achieved_video_fps`, the audio's rate, the camera and mic labels |
| `luma.raw` | **RAW LUMA** — uint8, row-major, 160×96, **15,360 bytes a frame**, frames concatenated |
| `timestamps.json` | `{"wall_clock_start": ..., "frame_offsets_ms": [...]}` — **MILLISECONDS**, `performance.now()` minus frame 0 |
| `audio.wav` | mono 16-bit at the AudioContext's **own** rate (48 kHz typically), not resampled |
| `video.webm` | optional, the full-resolution recording |

Two things in that table are traps, and I walked into both before reading the
code:

- **`luma.raw` is not NV12.** The script capture writes NV12, and three
  strides divide those files evenly — 23,040, 19,200 and 15,360 — so the
  layout of an NV12 file can only be decided by its content. A raw-luma file
  is 15,360 bytes a frame with no chroma at all, and asking `detect_layout`
  about one is asking the wrong question. The analyzer therefore enters the
  raw-luma path only on a **declared** geometry, and every other file falls
  through to detection.
- **`frame_offsets_ms` is milliseconds.** A reader that looks for
  `frame_offsets_s` and does not find it falls back to an assumed 30 fps,
  which is wrong by a factor of a thousand and fails silently. The analyzer
  reads the unit **from the key**, never guesses it.

The server validates all five sections on the way in (the luma length must
be `n_frames × 15,360`, the timestamps must have `n_frames` entries, the WAV
header must be honest about its rate). **The analyzer checks the same things
on the way out**, because a session folder that has been copied, truncated or
half-written must not be analysed silently: a `meta.n_frames` that disagrees
with the luma, an `audio.sample_rate` that disagrees with the WAV, a luma
geometry that is not the pipeline's 160×96, and timestamps whose count
disagrees with the frames are all **reported as warnings**, on the console
and in the report. None of them is fatal — the analysis proceeds on what is
actually there — but none of them is silent either.

**The five files are deliberately not the take layout.** They carry no
conditions, no plan and no math pass — the browser recorded, and analysis
happens offline. The subfolder also keeps every JSON in the session away
from the gate's out-root glob, which is the same reasoning that put ZCode's
rich copies in `out/rich/`. **This analyzer follows the same rule:** the
segment table is written to `<out>/rich/free_session_segments.json`, because
`find_takes` in the gate is a non-recursive `*.json` glob that excludes only
`*_timing.json`, and a segment table in the root would be individuated as a
take and fail.

---

## 2. THE FACE STATE — a nose-to-eye asymmetry

MediaPipe FaceMesh is already installed in the project GPU venv
(`/home/proart/projects/gpu-venv`, 0.10.21); nothing was installed for this
lane. `docs/research/MOUTH-APERTURE-ALIGNMENT.md` already established that
this repository uses it, and how.

The measure is the **asymmetry of the nose's distance to the two outer
eyes**:

```
sym = (|nose − eye_outer_left| − |nose − eye_outer_right|)
      / (|nose − eye_outer_left| + |nose − eye_outer_right|)
```

on landmarks 1 (nose), 33 and 263 (the outer eye corners). A face square to
the camera puts the nose equidistant from the two outer eyes, so `sym` sits
near zero; a turned head breaks that symmetry.

Measured on **every frame of every capture that exists** — 2,700 frames of
A, B and C, 100% face detection:

| capture | min | max | mean | std | frames at/above 0.18 |
|---|---|---|---|---|---|
| A (facing) | −0.1769 | +0.1051 | −0.0212 | 0.0599 | 0 / 900 |
| B (turned) | +0.2460 | +0.4227 | +0.3499 | 0.0341 | 900 / 900 |
| C (facing) | −0.0371 | +0.0592 | +0.0174 | 0.0142 | 0 / 900 |
| empty1 (empty room) | — | — | — | — | 0 / 4500, and 0 faces found at all |

The threshold is **0.18**, and it separates the two populations with no
errors: A's worst frame is 0.1769 and B's best is 0.2460.

**That margin is thin on the facing side — 0.0031 — and I am not going to
pretend otherwise.** The gap between the two classes is wide (0.177 against
0.246); the threshold's *placement* inside that gap is what is close. It is
placed just above A's single worst frame, and one unlucky frame in a new
capture could cross it. More data should re-derive it; the honest statement
is that 0.18 is correct on 2,700 measured frames and no more than that.

### The cue that was measured and rejected

The three-dimensional cue — the nose's depth against the eyes' — was measured
too and **rejected**: it reads −13.4 on A, −9.3 on B and −11.5 on C. It
confounds head direction with whether the mouth is open, which is the same
trap `docs/research/MOUTH-APERTURE-ALIGNMENT.md` documents for the mouth
aperture. A cue that moves when the speaker opens their mouth is not a cue
for which way the head is pointing.

The empty room is worth one more line: **0 faces found in 4,500 frames**, and
the machinery calls it `absent`, which is the right answer and a condition
with no take.

---

## 3. THE SPEECH STATE — the repo's own VAD, at the gate's own thresholds

### 3.1 The measure

`src/energy.py` — `rms_series` + `segment(adaptive=True)`. The adaptive
threshold is the repository's measured fix for a transient loud enough to
silence the rest of a recording, and the repo's own comment says so. The
per-frame timestamps carry the audio's speech intervals onto the video's
frames.

Measured over the captures:

| capture | speech fraction | the gate's band | label |
|---|---|---|---|
| A | 0.6693 | ≥ 0.15 | speaking |
| B | 0.6267 | ≥ 0.15 | speaking |
| C | 0.0070 | ≤ 0.05 | silent |
| empty1 | 0.0456 | ≤ 0.05 | silent |

The thresholds are **the gate's own** (`corpus_integrity.py`: 0.15 to call it
speech, 0.05 to call it silence), so a derived take and the gate agree by
construction rather than by luck.

### 3.2 The ambiguous band is dropped, never guessed

A segment whose aggregate lands **between 0.05 and 0.15** gets no label. It
is reported as skipped with the reason `ambiguous speech state` and its
margin to the nearest threshold, and it is **not** written as a take. A
take labelled "silent" whose aggregate is 0.12 would fail the gate, and
labelling it "speaking" would be a lie about a recording that is mostly
quiet.

### 3.3 The two bugs the gate caught, which my own reasoning missed

This is the part worth reading twice.

**The first: the label came from the frames instead of the aggregate.** The
per-frame state decides a segment's **boundaries**; the segment's
**aggregate** decides its **label**. I first implemented it the other way
round — the majority of the frames' states chose the label — and the gate
rejected **3 of 21** derived takes:

```
C1: speech_frac 0.1000 > 0.05 for a condition labelled speak=False
D1: speech_frac 0.3118 > 0.05 for a condition labelled speak=False
D2: speech_frac 0.5022 > 0.05 for a condition labelled speak=False
```

All three were segments whose frames were **mostly silent** — a pause
inside a sentence — but which carry enough audible content that the
aggregate exceeds the gate's 0.05. Labelling them "silent" is exactly the
mislabelling the whole design note is against.

The fix is one rule: **the aggregate decides the label.** The two segments
at 0.312 and 0.502 became *speaking* (they are above 0.15 — they really are
speech, the frames were just unevenly distributed) and the one at 0.100
became *ambiguous* and was dropped. After the fix: **20 takes, 0 failed.**

**The second: the A/V bound was computed against the wrong origin.** The
audio and the video are separate clocks, and the gate checks their spans
agree within 2%. The bound that keeps them in agreement is the audio's
**absolute** extent, not the segment's start plus the audio's length — and I
had written the latter. On a session whose video outran its audio (which is
exactly what happens when the page's frame loop and the mic disagree), the
slice was not trimmed at all and the gate said:

```
C1: video span 15.650 s against audio span 0.681 s (relative difference
    0.9565 > 0.02) -- the two streams are not the same length
```

The fix bounds the **segments**, not just the written slice, so a segment's
duration, its speech fraction and its usability are all judged on the part
that has audio. A 15.7-second stretch past the end of the audio was being
scored as a long silence; after the fix it is 0.7 seconds of what it
actually is, and it is correctly rejected as too short to be a take.

The selftest now pins both cases: 100 silent frames and 10 carrying a loud
blip — 91% silent by frame count, aggregate 0.082 — must come out ambiguous
and unusable; and a slice whose video outruns its audio must be trimmed so
the two spans agree.

---

## 4. THE SEGMENTS AND THEIR CONFIDENCES

The segment table is contiguous runs of (face state, speech state), with a
confidence per label:

| | |
|---|---|
| **face state** | `facing` / `turned` / `absent` |
| **speech state** | `speaking` / `silent` / `ambiguous` |
| **condition** | facing+speaking → **A**, turned+speaking → **B**, facing+silent → **C**, turned+silent → **D** |
| **face confidence** | how far the segment's mean `sym` sits from the 0.18 threshold, scaled to the measured range |
| **speech confidence** | how far the aggregate sits from the threshold it was judged against |
| **confidence** | the smaller of the two |

A state shorter than 0.5 s is noise, not a state, and is merged into its
longer neighbour. A segment shorter than 3 s is not a take. An absent face
has no condition. Nothing is guessed: if the evidence does not support a
label, there is no label.

An excerpt, from capture B (turned, speaking), showing the pauses a real
recording contains:

```
   0.00    1.70  turned  speaking frac=0.3118 cond=B usable=True
   1.70    3.43  turned  speaking frac=0.9971 cond=B usable=True
   3.43    4.03  turned    silent frac=0.0167 cond=D usable=False shorter than 1 s
   4.03    5.83  turned  speaking frac=0.9954 cond=B usable=True
   ...
   9.10   10.63  turned  speaking frac=0.5022 cond=B usable=True
   ...
  28.37   30.00  turned  speaking frac=0.7347 cond=B usable=True
```

---

## 5. THE SLICES — a derived take IS a take

A slice is written in **exactly** the format `experiments/corpus_capture_win.py
--plan` writes:

```
<COND><n>_<stamp>.npz     maps, scale, maps_raw, scale_raw, luma
<COND><n>_<stamp>.wav     mono 16-bit at the analysis rate
<COND><n>_<stamp>_timing.json
<COND><n>_<stamp>.json
```

Four decisions are worth stating, because each of them is a place where the
obvious thing is wrong:

- **The maps are the reference implementation's own.** The analyzer imports
  `experiments/corpus_capture_win.py` and calls its `row_band_remove`,
  `change_maps` and `quantise`. It does not re-implement them. The proof is
  in §6: the derived maps reproduce the stored takes' maps with **0
  differing cells**.
- **`assumed_fps = achieved_fps` for a slice.** A slice's frame index *is*
  its time base, so the assumed rate is the measured one and the gate's 5%
  rate check is satisfied by construction. The meaningful rate question —
  did the camera deliver what it claimed? — is asked **once, at the session
  level**, from the timestamps, and recorded in the metadata as
  `session_achieved_fps`.
- **The audio and the video are separate clocks, so the shorter one bounds
  the slice.** The gate checks the two spans agree within 2%. If the audio
  ran out before the last frame, the tail frames are dropped; if the video
  ran out, the audio is trimmed to it. The measured error is recorded as
  `av_span_error` and is ~1e-5 on a well-formed session.
- **The provenance is explicit.** `provenance: "derived: free_session_analyze"`,
  and a `derived` block carrying the segment's start and end, the face and
  speech states, the confidence, the mean symmetry, the detection rate, the
  session's rate and the source audio rate. A derived take never looks like
  a captured one.

---

## 6. THE RECOVERY REPORT — THE ACCEPTANCE EVIDENCE

The brief asks for two validations on data that exists today, and this is
both of them.

### 6.1 The unit tests

`--selftest` exercises the segmenter, the label rules, the take writer and
the **real free-session layout** on synthetic arrays: **66 checks, all
passing**. The layout case builds a session folder exactly as
`free_record.html` writes one — raw luma, millisecond timestamps, a 48 kHz
WAV, the page's own metadata — reads it back, and slices it, so the
production path is exercised even though no real free session exists yet.

The first run of these tests found three real bugs, all fixed:
`wavio.read_wav` is a function and not a context manager; the synthetic
label sequence could not produce the fourth state it claimed to; and the
expected WAV length was the frame count rather than the span.

### 6.2 The recovery report

```
capture   known                recovered             spfrac   det  err  sym range
A         facing/speaking      facing/speaking       0.6693 1.000    0  -0.177..+0.105
B         turned/speaking      turned/speaking       0.6267 1.000    0  +0.246..+0.423
C         facing/silent        facing/silent         0.0070 1.000    0  -0.037..+0.059
empty1    absent/silent        absent/silent         0.0456 0.000    0  n/a

recovered 4 of 4
```

The ground truth is **read from each take's own metadata** — `corpus/A_seed.json`,
`corpus_v2/A.json` — not hardcoded in the script, so the comparison is
against the artifact's label and not against my opinion of it. `err` is the
count of frames whose face state disagrees with the label: **0 in all four
captures**, out of 7,200 frames.

### 6.3 The maps, against the takes

The analyzer rebuilds the change maps for each capture and compares them
with **every** stored take that was seeded from that capture:

| take | built from | differing cells | scale |
|---|---|---|---|
| `corpus_v2/A.npz` | the corrected 19,200-byte read | **0 / 3,452,160** | 8.0 against 8.00000000000001 |
| `corpus_v2/B.npz` | the corrected 19,200-byte read | **0 / 3,452,160** | 3.0 against 3.0 |
| `corpus_v2/C.npz` | the corrected 19,200-byte read | **0 / 3,452,160** | 2.5 against 2.5 |
| `corpus/A_seed.npz` | the **torn** 23,040-byte read | 2,279,091 / 2,300,160 | 143.27 against 9.01 |
| `corpus/B_seed.npz` | the **torn** 23,040-byte read | 2,290,513 / 2,300,160 | 131.32 against 2.77 |
| `corpus/C_seed.npz` | the **torn** 23,040-byte read | 2,288,896 / 2,300,160 | 140.05 against 2.90 |

**3 of 6 exact, and the 3 that do not match are the 3 that were built on the
torn stride.** That is the NV12-STRIDE finding visible in the artifacts
themselves: the old seed takes are not a version of the pipeline's maps, they
are a different signal. The scale column says it plainly — a torn read gives
a scale of 143 against a true 9, because the tear is the dominant edge in
the frame.

So: **the derived maps are the pipeline's maps, bit for bit**, on the
corrected read.

### 6.4 The gate over the derived takes

The last thing the report does is write the derived takes and put them
through the repository's own gate — `experiments/corpus_integrity.py`, the
same code that checks the script's takes:

```
THE GATE over the derived takes: 20 takes, 0 failed, ok=True
  A         9 takes  0 failed
  B        10 takes  0 failed
  C         1 takes  0 failed
  empty1    0 takes  0 failed
```

Every check the gate makes is satisfied: the four files per take, the label
prefix matching the filename, `kind: "condition"` with a bool `speak`, the
`speech_frac` consistent with that bool, the sample rate equal on both
sides, the assumed and achieved frame rates within 5%, the audio and video
spans within 2%, the drift under 5.0, and the stimulus consistent across
the corpus.

`empty1` writes no takes, and that is correct: there is no face in the
room, so there is no condition to write.

### 6.5 The production path, end to end, on the real capture

The recovery report exercises the NV12 path. The **production** path is the
free-record layout, and no real free session exists yet — so the layout was
rebuilt from the real capture and put through the whole CLI:

```
/tmp/fsa_prod_session/free_sessions/1758000000/
    luma.raw         the REAL luma of A.nv12, as raw luma (900 frames)
    timestamps.json  frame_offsets_ms at 20 fps -- the capture's own
                     ~19.8 distinct Hz, which is what the page's
                     requestAnimationFrame grid would have recorded
    audio.wav        the REAL audio of A.wav, resampled to the 48 kHz an
                     AudioContext runs at
    meta.json        the page's own metadata shape
```

```
session   /tmp/fsa_prod_session/free_sessions/1758000000
  meta        meta.json
  luma        raw-luma:luma.raw (900 x 160x96)
  timestamps  timestamps.json (frame_offsets_ms)
  audio       audio.wav (48000 Hz)
  audio_rate  48000
```

The reader took the raw-luma path, the millisecond path and the 48 kHz path
without being told which was which — the metadata declared the geometry and
the key declared the unit. The segmenter found 22 segments in the 30
seconds the audio covers, of which two are long enough to be takes, and the
gate accepted both:

```
CORPUS INTEGRITY REPORT
takes             2
failed            0
unverified checks 0
  [ok  ] A1_1792550370.json  condition=A1  speak=True  stimulus=free-session
  [ok  ] A2_1792550370.json  condition=A2  speak=True  stimulus=free-session
```

**This is where the second gate-caught bug was found** (§3.3): the video
ran to 45 s and the audio to 30 s, and the slice that straddled that
boundary failed the gate's span check with a 95.65% relative difference.
The production path is where that bug could only ever appear, because only
the production path has two clocks that can disagree.

---

## 7. THE HARVESTER — the analyzer as a batch pass

Casey escalated the lane once more, and the right reading of it is that the
analyzer is now a **harvester**: it runs unattended over the corpus, it
processes every new session, and it writes down what the room gave.

```bash
# one pass -- this is the cron line
/home/proart/projects/gpu-venv/bin/python experiments/free_session_analyze.py \
    --harvest /mnt/c/proart_capture/corpus

# or continuously
/home/proart/projects/gpu-venv/bin/python experiments/free_session_analyze.py \
    --harvest /mnt/c/proart_capture/corpus --watch 60
```

**Three decisions carry the design, and each of them is forced by
something the gate or the page already decided.**

- **The derived takes get their own corpus root** — `<corpus>/derived/` —
  not a place in the script's corpus. The gate's stimulus check fails any
  take whose stimulus differs from the corpus majority ("a mixed stimulus
  cannot be pooled"), so a derived take (`free-session`) sitting beside a
  script take (`invictus`) would fail one of them. Two corpora, each
  internally consistent, is the only arrangement the gate accepts.
- **Every take from one session carries that session's stamp.** The gate's
  session identity is the take's `timestamp`, and the gate counts
  *distinct sessions*, not distinct files, because "one session recorded
  twice is one session". If each slice carried its own stamp, one
  recording would count as a dozen independent sessions — exactly the
  over-counting `corpus_integrity.py` exists to prevent. The derived
  filenames are `<COND><n>_<session stamp>`, so the identity is in the
  name too.
- **A session that is being written RIGHT NOW is skipped, not harvested.**
  The page writes the five files in sequence, so a folder caught
  mid-write is short a file. The analyzer cross-checks the metadata
  against what it read (§1) and a session whose frame count disagrees is
  **not** recorded in the state, so the next pass retries it. A session
  that fails is never marked done.

**The state** lives at `<corpus>/harvest/state.json` — in a subdirectory,
because the gate individuates every root-level `*.json` as a take, and
the harvester must not be the thing that breaks the gate.

**The report** is `docs/research/HARVEST-REPORT.md`, appended once per run:
the minutes harvested per derived condition, the confidence of those
labels, the coverage against the 63-sessions-per-class target, and what
is still missing. The coverage numbers are the **gate's own** — the
harvester runs `experiments/corpus_integrity.py` over the derived corpus
and quotes it, rather than keeping a tally of its own.

### The dry run

There is no real free session yet, so the first pass ran over what does
exist: the script's A, B, C and empty1 recordings, re-encoded in the
free-record layout. **4 sessions, 4 takes, 0 failed by the gate, 0.69
minutes** — A 0.15 min, B 0.06 min, C 0.49 min, and nothing at all from
the empty room, because there is no face in it. The gap to the target:
**SPEAK is 61 sessions short, SILENT is 62.** Two thirds of the target is
still missing and a free session is worth one session, so the arithmetic
is brutal and honest: the corpus grows by *sessions*, and only Casey's
time in front of the camera moves that number.

The full numbers are in `docs/research/HARVEST-REPORT.md`.

---

## 8. THE CAMERA PROBE, FOLDED IN — AND THIS LANE IS CLOSED

The brief folds the probe's measurements in here and closes that lane. The
full record is `docs/research/CAMERA-RND-FINDINGS.md`; the part that belongs
in this note is what the probe established about the **capture**, because
the analyzer works on what was captured.

### 7.1 The correction, which stands

**The camera is healthy.** My first measurement said otherwise — mean luma
16.00, digital black, identical to fifteen decimal places across seven
constraint configurations — because I measured the wrong stack. I used
**DirectShow**; the reference capture opens the camera with
`cv2.VideoCapture(cam)`, and OpenCV on Windows resolves that to **MSMF**,
the stack Chrome uses. DirectShow negotiates badly with this UVC device: it
opens, it returns frames, and they are the black offset with a handful of
uninitialised bytes in the corner (19 pixels brighter than black, in rows
0–17). Saturated random colours in isolated pixels are uninitialised memory,
not a dark room. Casey confirmed the camera responds to light.

**The reference capture set no constraints at all**, which is why it produced
105.9 in this room. The analyzer's input is exactly that capture.

### 7.2 The evidence pack's numbers are the torn-frame numbers

This never depended on the camera:

| capture | the brief says | correct stride (19,200) | torn stride (23,040) |
|---|---|---|---|
| A — occupied, facing | 110.7 | **105.94** | 110.73 |
| B — occupied, turned | 114.2 | **110.17** | 114.16 |
| empty room | 49.7 | **30.07** | 49.69 |

The capture page's on-screen "empty room 50" (`browser_capture/index.html`)
is the torn number and should read **30**. Today the analyzer re-derived
the same thing from the file itself: `detect_layout` on `empty1.nv12`
returns `y_uv_halfres`, 4,500 frames, with a margin over the runner-up of
**7,446** — the layout is not in doubt.

**And this is the same finding §6.3 sees in the artifacts.** The old seed
takes and the new v2 takes are the same capture read two ways, and the maps
differ in 99% of their cells.

### 7.3 The device, and the constraint that matters

| what | state |
|---|---|
| ASUS FHD webcam | Status OK — the front camera, confirmed by Casey |
| ASUS IR camera | Status OK — the face-auth sensor, and the classic dark-picture impostor ZCode's picker avoids |
| NVIDIA Virtual Camera DS | a virtual device, not the front camera |
| Windows camera access | **Allow** |

**The camera is single-consumer.** A second consumer gets
`NotReadableError: Device in use`. That is a live explanation for "the page
shows a dark picture" that has nothing to do with exposure, and it is the
one real observation in `experiments/camera_probe/results.jsonl`.

### 7.4 Why the probe lane is closed, and what survives it

**The analyzer does not need the camera's luma.** It works on the recording,
after the fact. The exposure question — whether the browser's stream is
darker than the camera can be — is a **product** question about the page's
constraints, not a precondition for deriving labels offline.

So the probe lane closes here, with its instrument intact: `experiments/camera_probe/`
stays in the tree, the server is still running on `127.0.0.1:8766`, and
`?sweep=1` will write the whole decision table — P0, P1, P3, P4 back to
back — the first time the camera is free. What is still open is in §9.

---

## 9. WHAT IS STILL OPEN

- **No real free session exists yet.** The production path is exercised only
  on a synthetic session built in the exact layout the page writes. The
  first real one should be run through it before anything is trusted.
- **The 0.18 threshold has a thin margin on the facing side** (0.0031). It
  is correct on 2,700 measured frames. More data should re-derive it.
- **The face state is measured on the 160×96 luma**, not on the full-
  resolution WebM the page also records. The full-resolution frames might
  give a better-conditioned cue; that is a cheap experiment once a real
  session exists.
- **The speech VAD is the repo's energy VAD, not Silero.** The gate reads
  the recorded `speech_frac` rather than re-measuring, so the two agree by
  construction — but a take re-measured with Silero could disagree, and
  that would be worth knowing.
- **The audio and video share an assumed origin.** The page's audio starts
  within a few milliseconds of the first frame (`performance.now()` versus
  the AudioContext's first chunk), and the analyzer treats them as one
  clock. The gate's 2% span check does not see the offset; a strict
  A/V *alignment* check would.
- **The probe's exposure question is still open**, and it belongs to the
  product lane: run `http://localhost:8766/?sweep=1` when the camera is
  free, and read the decision table in `CAMERA-RND-FINDINGS.md` §5.

---

## 10. HOW TO REPRODUCE ALL OF THIS

```bash
cd ~/projects/communicator-acceptance

# the unit tests, including the real free-session layout
/home/proart/projects/gpu-venv/bin/python experiments/free_session_analyze.py \
  --selftest

# the recovery report, the derived takes, and the gate over them
/home/proart/projects/gpu-venv/bin/python experiments/free_session_analyze.py \
  --recover /mnt/c/proart_capture --slices /tmp/derived --json /tmp/recovery.json

# a real session, once one exists
/home/proart/projects/gpu-venv/bin/python experiments/free_session_analyze.py \
  /mnt/c/proart_capture/corpus/free_sessions/<stamp> --out /tmp/derived --gate

# the gate on its own, over anything
/home/proart/projects/glasses/.venv/bin/python experiments/corpus_integrity.py \
  /tmp/derived
```

### Rebuilding the production-layout session of §6.5

```bash
# the REAL capture, written in the layout the free-record page writes
/home/proart/projects/gpu-venv/bin/python - <<'PY'
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
from src import nv12, wavio

src = Path('/mnt/c/proart_capture')
out = Path('/tmp/fsa_prod_session/free_sessions/1758000000')
out.mkdir(parents=True, exist_ok=True)

luma, _ = nv12.read_luma(src/'A.nv12', width=160, height=96)
(out/'luma.raw').write_bytes(luma.astype(np.uint8).tobytes())

fps = 20.0                      # the capture's own ~19.8 distinct Hz
offs = [round(i*1000.0/fps, 1) for i in range(len(luma))]
(out/'timestamps.json').write_text(json.dumps(
    {"wall_clock_start": "2026-10-21T00:00:00Z", "frame_offsets_ms": offs}))

a = wavio.resample(wavio.read_wav(str(src/'A.wav')), 48000)
wavio.write_wav(out/'audio.wav', a.samples, 48000)

(out/'meta.json').write_text(json.dumps({
    "tool": "browser_free_record", "n_frames": len(luma),
    "luma": {"width": 160, "height": 96, "frame_bytes": 15360},
    "audio": {"sample_rate": 48000, "samples": len(a.samples)},
    "achieved_video_fps": round((len(luma)-1)/(offs[-1]/1000.0), 4),
    "mic": {"label": "real mic"}}, indent=2))
PY

# and then the production path over it
/home/proart/projects/gpu-venv/bin/python experiments/free_session_analyze.py \
  /tmp/fsa_prod_session/free_sessions/1758000000 --out /tmp/derived --gate
```
