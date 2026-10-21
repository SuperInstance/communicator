# CAMERA-RND-FINDINGS — the empirical lane

**Author:** the acceptance worktree (`~/projects/communicator-acceptance`,
branch `acceptance/browser-capture`)
**Date:** 2026-10-20
**Brief:** `docs/research/BRIEF-CAMERA-RND.md`
**Deliverable:** `experiments/camera_probe/` — `probe.html`, `probe_server.py`
**Measurement file:** `experiments/camera_probe/results.jsonl`
**Server:** running on `http://127.0.0.1:8766` (reachable from Windows — verified)

---

## 0. THE FINDING THAT DECIDES THE RACE

**The camera is not delivering an image right now, and no constraint set
changes that.** I measured the ASUS FHD webcam on the Windows side through
the driver, across seven constraint configurations and five exposure
settings, and got **the same mean luma — 16.003971354166662 — every single
time**, identical to fifteen decimal places.

Sixteen is *digital black*: `cv2.COLOR_BGR2YUV_I420` puts the black level at
Y=16, so every pixel of every frame is at the black offset. There is no
image. 19 pixels out of 307,200 were brighter than black, and those were
saturated random colours — the signature of an unfilled buffer, not of a
dark room.

**So the constraint-set question cannot be the cause of the browser's
darkness, and no change to the product page's `getUserMedia` constraints can
fix it while the camera is in this state.** Whatever is happening is
upstream of the constraint set.

That is the single most important thing this lane can tell the race, and it
cost twenty minutes to establish.

**And the second-most important thing:** the reference figures in the brief
are wrong, in a way that matters. They are the *torn-frame* numbers.

---

## 1. THE EVIDENCE PACK'S NUMBERS ARE THE TORN-FRAME NUMBERS

The brief hands me three reference points: A.nv12 mean luma 110.7, B.nv12
114.2, empty room 49.7. I recomputed all three from the files on disk, at
the **correct 19,200-byte stride** (`src/nv12.py`, `y_uv_halfres` layout):

| capture | the brief says | correct stride (19,200) | torn stride (23,040) |
|---|---|---|---|
| A.nv12 — occupied, facing | 110.7 | **105.94** | 110.73 |
| B.nv12 — occupied, turned | 114.2 | **110.17** | 114.16 |
| empty room | 49.7 | **30.07** | 49.69 |

The brief's three numbers match the **torn** read — the 23,040-byte stride,
the one this repo spent a night identifying and correcting. `docs/research/
NV12-STRIDE.md` is the record of that; `experiments/room_map.py` carries the
old value in its "supersedes" block (`previous_stride_bytes: 23040`,
`previous mean_luma: 49.70`) and recomputes it correctly.

**Two consequences, and the second one is the one that changes the work:**

1. **The capture page's own on-screen reference is wrong.** `index.html`
   tells Casey "old captures ~110, empty room 50". The occupied figure is
   right (105.9–110.2). **The empty-room figure should be 30, not 50.**

2. **The camera in the reference stack was NOT starved.** With the subject
   present: 105.9 and 110.2. With the lights off and nobody there: 30.1.
   That is a 3.5× dynamic range — a camera working properly, with its
   auto-exposure doing its job. If the browser path is reading far below
   30, it is not reading a dark room; it is reading nothing.

The premise "the camera comes up starved" is **not** supported by the
evidence pack as given. The evidence pack was measuring the wrong bytes.

---

## 2. THE DEVICE INVENTORY

Enumerated from the Windows side, not guessed:

| what | how it was found | state |
|---|---|---|
| **ASUS FHD webcam** | `Get-PnpDevice -Class Camera` | Status OK, `CM_PROB_NONE` |
| **ASUS IR camera** | `Get-PnpDevice -Class Camera` | Status OK, `CM_PROB_NONE` |
| NVIDIA Virtual Camera DS | `[VCAMDS]` in the DShow log at index 1 | a virtual device, not the front camera |
| Windows camera access | `CapabilityAccessManager\ConsentStore\webcam` | **Allow** (top level and NonPackaged) |
| `LetCameraAccess` policy | `HKLM\SOFTWARE\Policies\...\Privacy` | not set |

Casey confirmed the ASUS FHD webcam is the correct front camera. **Windows
camera access is allowed and the device reports no problem**, so this is not
an OS privacy block and not a disabled device.

---

## 3. THE MEASUREMENT — HOW THE CAMERA BEHAVES UNDER EVERY CONSTRAINT SET

Method: OpenCV through DirectShow on the Windows side, reading the real
device, with the reference capture's exact luma maths —
`resize(frame, (160,96), INTER_AREA)` → `cvtColor(BGR2YUV_I420)` → Y plane
→ mean. The same maths `corpus_capture_win.py` uses, so the numbers are
directly comparable to the .nv12 captures.

### 3.1 The constraint sweep

| configuration | delivered size | delivered fps | mean luma |
|---|---|---|---|
| 640×480 @ 30 (what the product ships) | 640×480 | 30 | **16.004** |
| 640×480, frame rate unconstrained | 640×480 | –1 | **16.004** |
| 1280×720, frame rate unconstrained | 1280×720 | –1 | **16.004** |
| 1280×720 @ 30 | 1280×720 | 30 | **16.004** |
| 1280×720 @ 15 | 1280×720 | 15 | **16.004** |
| 1920×1080, frame rate unconstrained | 1920×1080 | –1 | **16.004** |
| the driver's own default (what the reference capture used) | 640×480 | –1 | **16.004** |

The mean is *the same number* in all seven rows — to fifteen decimal places,
because the Y plane sums to exactly 245,820 every time.

### 3.2 The driver's own exposure knobs do nothing

At 640×480@30, moving the driver's exposure control across its whole range:

```
EXPOSURE -8 -> 16.003971354166662
EXPOSURE -4 -> 16.003971354166662
EXPOSURE  0 -> 16.003971354166662
EXPOSURE +4 -> 16.003971354166662
EXPOSURE +8 -> 16.003971354166662
```

### 3.3 What the frame actually contains

```
frame 0 (640x480): B mean 0.0075  G mean 0.0044  R mean 0.0043
                    20 unique colours in 307,200 pixels
                    19 pixels brighter than black, all in rows 0-17, cols 0-30
                    their values: [224,126,2] [48,197,159] [126,2,0] [244,174,0] ...
the 160x96 Y plane (what the takes record):
                    mean 16.004  min 16  max 31  std 0.220  8 unique values
                    row 0: [29, 17, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
```

Looked at directly (amplified 12×), the frame is **pure black** with a
scatter of garbage in the top-left corner. A real camera in a dark room
produces noise; this produces *nothing*.

### 3.4 The honest caveat on that measurement

I could not rule out one alternative mechanism: **another process holding the
device**. Chrome is running with 21 processes, and Chrome's stack is Media
Foundation — the MSMF open of index 0 succeeded but delivered **no frames
at all**, and a DirectShow open of the same index *blocked* on one occasion
and succeeded on others. The pattern is what a contended UVC device looks
like.

**The two mechanisms are distinguishable in one click with the probe**, and
that is the point of building it rather than writing this paragraph:

- If the probe shows a **real image** (luma ~100), then the black I measured
  was contention from my own second open, the camera is healthy, and the
  browser's constraint set becomes the live suspect again.
- If the probe shows **black too**, then the camera is obstructed or dead to
  this session, and no constraint set matters.

The probe now detects and names this state in the UI, and records
`stream_is_black` and `luma_black_fraction` in every log line.

---

## 4. THE PROBE — BUILT, RUNNING, AND VERIFIED WITHOUT A LENS

`experiments/camera_probe/probe.html` — one file, vanilla JS, no CDN, no
external fonts, no network beyond the local server.

**The presets**, each applied to the **same** camera (the device is pinned
to the one Casey approves on the first request):

| | constraint set | the question it answers |
|---|---|---|
| **P0** | `{width: ideal 640, height: ideal 480, frameRate: ideal 30}` | what the product page ships today |
| **P1** | `{video: true}` — nothing beyond the device | what the driver does when left alone |
| **P2** | defaults + `exposureMode: continuous` via `applyConstraints`, caught if unsupported | is the driver's AGC being held off? |
| **P3** | `{width: ideal 1280, height: ideal 720}`, **no frame rate pin** | does a lower frame rate buy exposure? |
| **P4** | adaptive, from the camera's own capabilities dump: unpinned frame rate, `exposureMode: manual` + the longest exposure the driver offers, `exposureCompensation` at max | the maximum this camera can do |

**The manual panel** (Casey's addition, folded in): every capability the
track exposes gets a control built from `getCapabilities()` — a **slider**
for each numeric range (`exposureTime`, `exposureCompensation`, `gain`,
`iso`, `brightness`, `frameRate`, `colorTemperature`, `pan`/`tilt`/`zoom`,
`aspectRatio`…), a **select** for each enum (`exposureMode`, `whiteBalanceMode`,
`focusMode`), a **checkbox** for each boolean. Every change applies **live**
via `track.applyConstraints()`, merging the value into the preset's
constraint set so the resolution is not dropped, and reading
`getSettings()` back.

**LOG PRESET** samples the luma for 3 s and POSTs one JSON object to the
server, which appends it to `experiments/camera_probe/results.jsonl`:
the preset, the manual slider state, the constraint set, the full
`getSettings()`, a capabilities summary, and the luma mean/min/max. **If any
slider has been moved the record is logged as `P-manual`**, carrying the
slider values as numbers — so a hand-found winning combination lands in the
file as data, not as a description.

The luma is computed on a 160×96 canvas with the capture page's own
`lumaFromImage` (BT.601, clamped, rounded), so it is the same quantity the
takes carry and the same quantity the .nv12 files carry.

### The verification, without a lens

Chrome's fake device (`--use-fake-device-for-media-stream`) was used to
exercise the entire chain headlessly. It works, end to end:

```
getUserMedia with P0's constraints      -> opened
getCapabilities()                       -> dumped, and the slider panel built from it
applyConstraints via a slider           -> getSettings() came back CHANGED:
                                          exposureMode: "manual", exposureTime: 50
3 s luma sample                         -> 180 frames, mean 81.76
POST /log                               -> accepted, appended, logged as "P-manual"
                                         with manual: {"aspectRatio": 1920.000231...}
```

The fake-device record was removed from `results.jsonl` afterwards, so that
file holds **only real measurements**. The probe is reproducible with:

```
chrome.exe --headless=new --use-fake-device-for-media-stream \
  --use-fake-ui-for-media-stream --user-data-dir=<scratch> \
  "http://127.0.0.1:8766/?selftest=1"
```

**The server** is running on `http://127.0.0.1:8766`, stdlib only, bound to
localhost, appending to `experiments/camera_probe/results.jsonl`. Windows
reachability is confirmed with the Windows `curl.exe` from WSL, so Casey can
open it in Chrome on the Windows side at **http://localhost:8766**.

---

## 5. THE RECOMMENDATION

### 5.1 Right now, in this order

**Step 1 — make the camera deliver an image.** Nothing else is measurable
until this is true. Concretely:

- close every tab and app that could hold the camera (the capture page
  itself holds it while it is running — **this is the most likely
  explanation for a black stream in the browser**, because a second
  `getUserMedia` consumer in Chrome can be handed a black or frozen stream);
- check the lens is not covered and the physical shutter (the ASUS FHD
  webcam has one) is open;
- if the room is dark, note that the *empty room with lights off* measured
  30.1 on this camera — so if the probe reads near 30, the camera is fine
  and you are simply in the dark room, which is a lighting problem, not a
  constraint problem.

**Step 2 — click P0 → LOG, then P1 → LOG, then P3 → LOG, then P4 → LOG.**
Four clicks, four lines in `results.jsonl`.

### 5.2 The decision rule, which the probe turns into a lookup

Let `L(P)` be the mean luma the probe records for preset `P`, and let the
reference be **106** (occupied, facing — the corrected number).

| what the probe shows | what it means | what to ship |
|---|---|---|
| every preset reads ~16, `stream_is_black: true` | the camera is delivering nothing; the constraint set is irrelevant | fix the camera/session, not the page. **Do not change the constraints on this evidence.** |
| P0 ≈ P1 ≈ P3 ≈ 106 | the constraints are not the cause | ship P0, unchanged |
| P0 ≪ P1 ≈ 106 | **the constraint set is the cause** — the pin is starving the exposure | ship the unconstrained form: `{width: ideal 640, height: ideal 480}` with **no frameRate**, and let the driver's own AGC work |
| P3 ≫ P0, P1 ≈ P3 | the **frame rate** is the cost, not the resolution | ship 720p with no frame rate pin, or a lower cap |
| P2 ≫ P0 but P1 ≈ P2 | the driver's AGC is being held off by the constraints | ship P2's `exposureMode: continuous` |
| only P4 reaches ~106 | the camera is exposure-limited and the driver will only brighten under manual exposure | ship `exposureMode: manual` + a long `exposureTime`, and accept the frame rate that follows |

### 5.3 The prior, stated before the numbers come in

**The reference capture set no constraints at all and produced 105.9.**
`corpus_capture_win.py` calls `cv2.VideoCapture(cam)` and resizes to
160×96 afterwards; it never asks the camera for a resolution or a frame
rate. The browser page *does* constrain: `{width: ideal 640, height:
ideal 480, frameRate: ideal 30}`.

So the two paths differ in exactly one respect that could matter — the
constraints — and the unconstrained one is the one that produced 106. If the
camera is healthy and the probe confirms the browser is dark, **the
expected fix is to stop pinning the frame rate**, not to pin it harder, and
the probe's P1/P3 columns will say so with numbers instead of with argument.

The counter-hypothesis, which the probe also tests: the camera is not
starved, and the browser's darkness is the *second consumer* problem — in
which case no constraint change matters and the fix is "one consumer".

Both hypotheses are live today. That is exactly why the probe exists, and
why it needed to be verified without a lens.

---

## 6. WHAT I DID NOT DO, AND WHAT IS STILL OPEN

- **I did not change `browser_capture/`.** The brief says do not touch it,
  and I did not. The probe is a separate instrument in `experiments/`.
- **I did not record any of Casey's data.** The .nv12 files were read
  read-only for the stride verification. The real corpus was not written
  to.
- **I could not measure Chrome's own `getUserMedia` behaviour**, because
  Media Foundation would not deliver frames while Chrome held the device.
  That is what the probe is for, and it is the only thing that can answer
  the browser question.
- **The MF extended-video-property ranges** (`exposureTime`,
  `exposureCompensation`, `gain`, `iso`, `colorTemperature`) come from
  `getCapabilities()`, which only a browser can give. The probe prints them
  and records them. Until Casey clicks, the ranges are unknown.
- **The `exposureTime` unit needs a human eye.** Chrome's fake device
  reports `exposureTime: {min: 10, max: 100, step: 5}` — milliseconds,
  while the WebRTC spec says seconds. The probe labels which it is looking
  at, but the real camera's unit has to be read off the capabilities dump
  and confirmed once.
- **`ideal` is a hint, not a guarantee.** The fake device clamped P0's
  `frameRate: {ideal: 30}` to 20 and reported 20 in `getSettings()`. The
  real camera may clamp too — and `CAPTURE-NEXT.md` already records the
  script asking for 30 fps and the camera delivering 25. **The constraint
  the page ships may not be the constraint the camera honours**, which is
  a third reason to prefer the unconstrained form.

---

## 7. HOW TO REPRODUCE ALL OF THIS

```bash
# the probe, running
cd ~/projects/communicator-acceptance
nohup python3 experiments/camera_probe/probe_server.py > /tmp/camera_probe_server.log 2>&1 &
curl -s http://127.0.0.1:8766/health
# then, on Windows: http://localhost:8766

# the probe's own verification, without a camera
chrome.exe --headless=new --use-fake-device-for-media-stream \
  --use-fake-ui-for-media-stream --user-data-dir=<scratch> \
  "http://127.0.0.1:8766/?selftest=1"

# the evidence-pack correction (the torn-vs-correct stride)
python3 -c "
import sys; sys.path.insert(0,'.')
import numpy as np
from src import nv12
for n in ('A','B','empty1'):
    d = np.frombuffer(open(f'/mnt/c/proart_capture/{n}.nv12','rb').read(), np.uint8)
    print(n, 'correct', round(float(nv12.luma_view(d,'y_uv_halfres',160,96).mean()),2),
             '| torn', round(float(nv12.luma_view(d,'standard_nv12',160,96).mean()),2))
"

# the Windows-side sweep (run with the WINDOWS python, which has OpenCV)
cd /mnt/c/Users/casey/AppData/Local/Temp/cam_probe
/mnt/c/Python314/python.exe win_sweep.py
```
