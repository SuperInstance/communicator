# CAMERA-RND-FINDINGS — the empirical lane

**Author:** the acceptance worktree (`~/projects/communicator-acceptance`,
branch `acceptance/browser-capture`)
**Date:** 2026-10-20
**Brief:** `docs/research/BRIEF-CAMERA-RND.md`
**Deliverable:** `experiments/camera_probe/` — `probe.html`, `probe_server.py`
**Measurement file:** `experiments/camera_probe/results.jsonl`
**Server:** running on `http://127.0.0.1:8766` (Windows reachability verified)

---

## 0. THE HEADLINE, INCLUDING WHERE I WAS WRONG FIRST

**The camera is healthy, and the first measurement I took said otherwise
because I measured the wrong stack.** That is the honest shape of this lane,
and the correction is the most useful thing in it.

| | what I said first | what is true |
|---|---|---|
| **The camera** | "delivers digital black; no constraint set changes it" | **the camera is fine.** Casey has confirmed at the page that it is the front camera and that it *responds to light* — a lamp brought into frame registers. The camera was never starved. |
| **Why I was wrong** | — | I measured through **DirectShow**. The reference capture opens the camera with `cv2.VideoCapture(cam)` and OpenCV 5.0.0 resolves that to **MSMF** — the same stack Chrome uses. DirectShow negotiates badly with this UVC device: it opens, it returns frames, and they are the black offset with a handful of uninitialised bytes in the corner. **I measured the broken stack and called it the camera.** |

Three things survive the correction, and all three are useful:

1. **The evidence pack's reference numbers are wrong** — they are the
   *torn-frame* numbers (§2). This is independent of the camera and is
   still fully valid.
2. **The camera is single-consumer**, and a second consumer gets
   `NotReadableError: Device in use` (§3). That is a real constraint on the
   product's use, and it is a plausible contributor to "REALLY dark".
3. **The instrument is built and verified** (§4): the probe measures the
   browser's stack, which is the only place the question can be answered.

**What is still unmeasured: the luma per constraint set on the real camera.**
That measurement needs the camera to be free of other consumers for the
moment it takes, and the probe is built to take it the first time that
happens — `?sweep=1` measures P0, P1, P3 and P4 back to back and writes the
whole decision table in one action.

---

## 1. WHERE I WENT WRONG, PRECISELY

The first sweep reported mean luma **16.003971354166662** — digital black —
for every one of seven constraint configurations, identical to fifteen
decimal places, with the driver's exposure knob changing nothing.

That was a *methodological* error, and it is worth writing down because the
same error is waiting for the next person:

```
corpus_capture_win.py:349     cap = cv2.VideoCapture(a.cam)     # no backend
```

OpenCV 5.0.0 on Windows resolves that to MSMF. Measured directly:

```
default cv2.VideoCapture(0)  opened=True  backend=MSMF
CAP_MSMF                     opened=True  backend=MSMF
CAP_DSHOW                    opened=True  backend=DSHOW
```

So the reference capture — the one that produced luma 105.9 and 110.2 with
Casey in frame — used **MSMF**. My sweep used **DSHOW**, which on this
device returns a black frame with garbage pixels:

```
frame 0 (640x480): B mean 0.0075  G mean 0.0044  R mean 0.0043
                    20 unique colours in 307,200 pixels
                    19 pixels brighter than black, in rows 0-17, cols 0-30
                    their values: [224,126,2] [48,197,159] [126,2,0] ...
the 160x96 Y plane: mean 16.004  min 16  max 31  std 0.220  8 unique values
```

Saturated random colours in isolated pixels are uninitialised memory, not a
dark room. A real sensor in a dark room produces noise; a broken stack
produces nothing.

**Two independent checks confirmed the camera was never the problem:**

- **The two-consumer test**: with one consumer holding the device and a
  second opening it, *both* saw black — so on the DirectShow path, black is
  what this stack returns regardless of contention.
- **The browser test**: headless Chrome on Windows, opening the **real**
  camera through the real browser stack, returned
  **`NotReadableError: Device in use`** — the camera is held by Casey's
  Chrome, and the browser stack does not silently return black for a
  contended device; it says so.

The corrected sweep, on the default (MSMF) backend, is in
`experiments/camera_probe/win_sweep.py`. **It cannot run while Chrome holds
the camera** — it returns no frames. That is the correct behaviour for the
right stack, and it is why the measurement belongs in the browser.

---

## 2. THE EVIDENCE PACK'S NUMBERS ARE THE TORN-FRAME NUMBERS

This finding never depended on the camera, and it stands.

The brief hands me three reference points: A.nv12 mean luma 110.7, B.nv12
114.2, empty room 49.7. Recomputed from the files on disk at the **correct
19,200-byte stride** (`src/nv12.py`, `y_uv_halfres`):

| capture | the brief says | correct stride (19,200) | torn stride (23,040) |
|---|---|---|---|
| A.nv12 — occupied, facing | 110.7 | **105.94** | 110.73 |
| B.nv12 — occupied, turned | 114.2 | **110.17** | 114.16 |
| empty room | 49.7 | **30.07** | 49.69 |

The brief's numbers match the **torn** read — the 23,040-byte stride this
repo spent a night identifying and correcting. `docs/research/NV12-STRIDE.md`
is the record; `experiments/room_map.py` still carries the old value in its
"supersedes" block (`previous_stride_bytes: 23040`, `previous mean_luma:
49.70`) and recomputes it correctly.

**Two consequences:**

1. **The product page's on-screen reference is wrong.** `index.html` tells
   Casey "old captures ~110, empty room 50". The occupied figure is right
   (105.9–110.2). **The empty-room figure should be 30, not 50.** The page's
   luma readout goes amber below ~80; against the correct empty-room figure
   of 30, that threshold is better placed than it looks, but the *number* on
   screen is the torn one. `browser_capture/DONE.md` repeats it ("111–114",
   "empty room: 50") — both should be 105.9/110.2 and 30.1.

2. **The camera was never starved.** Occupied 105.9, lights off and nobody
   there 30.1 — a 3.5× dynamic range, which is a working auto-exposure. So
   the question is not "why is the camera dark" but **"why is the browser's
   stream darker than the camera is capable of being"** — and the reference
   capture, which set **no constraints at all**, is the reference for what
   the camera can do.

---

## 3. THE DEVICE INVENTORY, AND THE SINGLE-CONSUMER CONSTRAINT

| what | how it was found | state |
|---|---|---|
| **ASUS FHD webcam** | `Get-PnpDevice -Class Camera` | Status OK, `CM_PROB_NONE` — the front camera, confirmed by Casey |
| **ASUS IR camera** | `Get-PnpDevice -Class Camera` | Status OK — the face-auth sensor, and the classic dark-picture impostor |
| NVIDIA Virtual Camera DS | `[VCAMDS]` in the DirectShow log at index 1 | a virtual device, not the front camera |
| Windows camera access | `CapabilityAccessManager\ConsentStore\webcam` | **Allow** (top level and NonPackaged) |
| `LetCameraAccess` policy | `HKLM\SOFTWARE\Policies\...\Privacy` | not set |

**ZCode independently found the IR impostor** and shipped a camera picker
that avoids it (`3c0b2fe`, "a picker that avoids the IR impostor"). My
device inventory corroborates it: there are two physical cameras on this
machine, and the IR one is the one that looks like a dead camera.

**And the single-consumer constraint is real.** A second consumer of the
device gets:

```
NotReadableError: Device in use
```

That is worth knowing for the product, not just for the measurement:

- The capture page holds the camera for the whole take.
- A second tab, a second instance of the page, or any other app that wants
  the camera **will fail**, and a failure that surfaces as a black picture
  in one of them is exactly the "REALLY dark" symptom.
- So "the page shows a dark picture" has at least two live explanations
  besides the exposure one: the wrong camera (IR) and a contended device.
  Both are now diagnosable in the page, and both are diagnosable in the
  probe.

---

## 4. THE PROBE — BUILT, RUNNING, VERIFIED WITHOUT A LENS

`experiments/camera_probe/probe.html` — one file, vanilla JS, no CDN, no
external fonts, nothing leaving localhost.

**The presets**, each applied to the **same** camera (the device is pinned
to the one Casey approves on the first request):

| | constraint set | the question it answers |
|---|---|---|
| **P0** | `{width: ideal 640, height: ideal 480, frameRate: ideal 30}` | what the product page shipped until the third patch |
| **P1** | `{video: true}` — nothing beyond the device | what the driver does when left alone — **and what the reference capture did, which produced 105.9** |
| **P2** | defaults + `exposureMode: continuous` via `applyConstraints`, caught if unsupported | is the driver's AGC being held off? |
| **P3** | `{width: ideal 1280, height: ideal 720}`, **no frame rate pin** | does a lower frame rate buy exposure? |
| **P4** | adaptive, from the camera's own capabilities dump: unpinned frame rate, `exposureMode: manual` + the longest exposure the driver offers, `exposureCompensation` at max | the maximum this camera can do |

**The manual panel** (Casey's addition): every capability the track exposes
gets a control built from `getCapabilities()` — a **slider** for each numeric
range (`exposureTime`, `exposureCompensation`, `gain`, `iso`, `brightness`,
`frameRate`, `colorTemperature`, `pan`/`tilt`/`zoom`, `aspectRatio`…), a
**select** for each enum, a **checkbox** for each boolean. Every change
applies **live** via `track.applyConstraints()`, merging the value into the
preset's constraint set so the resolution is not dropped, and reading
`getSettings()` back.

**LOG PRESET** samples the luma for 3 s and POSTs one JSON object to the
server, which appends it to `results.jsonl`: the preset, the manual slider
state, the constraint set, the full `getSettings()`, a capabilities summary,
and the luma mean/min/max — plus `stream_is_black` and
`luma_black_fraction`, because a stream that is *digitally black* is a
different fault from a stream that is merely dark. **If any slider has been
moved the record is logged as `P-manual`**, carrying the slider values as
numbers, so a hand-found winning combination lands in the file as data.

**`?sweep=1` measures P0, P1, P3 and P4 back to back** and writes the whole
decision table in one action — because the camera's free moments are
precious and nobody wants to click through four presets by hand.

**`?selftest=1`** exercises the entire chain headlessly, and
**`?preset=…&log=1`** measures one preset non-interactively.

### Verification, without a lens

Chrome's fake device (`--use-fake-device-for-media-stream`) was used to
exercise the whole chain. It works, end to end:

```
getUserMedia with P0's constraints      -> opened
getCapabilities()                       -> dumped, and the slider panel built from it
applyConstraints via a slider           -> getSettings() came back CHANGED:
                                          exposureMode: "manual", exposureTime: 50
?sweep=1                                -> four records in one run:
                                          P0 luma 88.14 (fps 20)
                                          P1 luma 88.19 (deviceId pinned)
                                          P3 luma 86.45
                                          P4 luma 86.27
POST /log                               -> accepted, appended, logged as "P-manual"
```

The fake device's numbers are not a measurement of anything; they are a
measurement of the *probe*. Every fake-device record was removed from
`results.jsonl`, which now holds only real observations — currently one, the
`NotReadableError: Device in use` from the real-camera attempt.

**The server** is running on `http://127.0.0.1:8766`, stdlib only, bound to
localhost. Windows reachability is confirmed with the Windows `curl.exe`, so
Casey can open it in Chrome on the Windows side at **http://localhost:8766**.

---

## 5. THE DECISION TABLE — WHAT THE PROBE WILL SETTLE

Let `L(P)` be the mean luma the probe records for preset `P`. The reference
is **106** (occupied, facing — the corrected number).

| what the probe shows | what it means | what to ship |
|---|---|---|
| every preset ≈ 16, `stream_is_black: true` | the camera is delivering nothing to *this* consumer | close every other camera consumer; check the lens. **Do not change the constraints on this evidence.** |
| every preset ≈ 106 | the constraints are not the cause | the current near-default form is fine |
| P0 ≪ P1 ≈ 106 | **the constraint set is the cause** | ship the unconstrained form — and note P1 *is* what the reference capture did |
| P3 ≫ P0, P1 ≈ P3 | the **frame rate** is the cost, not the resolution | ship 720p with no frame rate pin, or a lower cap |
| P2 ≫ P0 but P1 ≈ P2 | the driver's AGC is being held off by the constraints | keep `exposureMode: continuous` |
| only P4 reaches ~106 | the camera is exposure-limited; the driver only brightens under manual exposure | ship `exposureMode: manual` + a long `exposureTime`, and accept the frame rate that follows |
| every preset reads low but responds to light | the camera is fine and the *page* is starving it | this is the case ZCode's third patch targets, and the probe will say whether it worked |

**One new column the product needs, which is not a constraint at all:**
*which camera*. If the picker lands on the ASUS IR camera, the picture is
dark for a reason no constraint set can fix, and the page now has a picker
that avoids it. The probe reports `device_id` and `device_label` with every
record so the two are never confused.

---

## 6. THE RECOMMENDATION

**Step 1 — free the camera.** Close every other consumer: the capture page
holds it for the whole take, and a second consumer gets
`NotReadableError: Device in use` (or a black picture, depending on the
stack). Confirm the picker is on the ASUS **FHD** webcam, not the **IR**
camera.

**Step 2 — run the sweep.** One action:

```
http://localhost:8766/?sweep=1
```

or, by hand, P0 → LOG → P1 → LOG → P3 → LOG → P4 → LOG. Four records land in
`experiments/camera_probe/results.jsonl`.

**Step 3 — read the table above.**

**The prior, stated before the numbers arrive:** the reference capture set
**no constraints** and produced 105.9 in this room, on this camera. The
browser sets constraints. If the camera is healthy and the browser is dark,
the constraint set is the difference, and **the expected fix is to stop
pinning the frame rate**, not to pin it harder. ZCode's third patch already
moved to near-default constraints for exactly that reason, and its own
`DONE.md` says the expectation is unverified until this lane reports.

**It is verified now, in one direction:** the camera is *not* starved, so
there is nothing for a constraint change to fix at the hardware level; and
the live-stream path cannot be measured from outside the browser at all,
because the device is single-consumer and the browser holds it. The probe
is the instrument, and it is ready.

---

## 7. WHAT I DID NOT DO, AND WHAT IS STILL OPEN

- **I did not change `browser_capture/`.** The brief says do not touch it.
- **I did not write to the real corpus.** The .nv12 files were read
  read-only for the stride verification.
- **I could not measure the camera's luma under any constraint set**, because
  the camera is held by Casey's Chrome, and my fallback stack (DirectShow)
  is broken for this device. The one real-browser attempt is recorded in
  `results.jsonl` as the `Device in use` error.
- **The MF extended-video-property ranges** (`exposureTime`,
  `exposureCompensation`, `gain`, `iso`, `colorTemperature`) come from
  `getCapabilities()`, which only a browser can give. The probe prints them
  and records them. Until it is run, the ranges are unknown.
- **The `exposureTime` unit needs a human eye.** Chrome's fake device
  reports `exposureTime: {min: 10, max: 100, step: 5}` — milliseconds —
  while the WebRTC spec says seconds. The probe labels which it is looking
  at, but the real camera's unit has to be read off the capabilities dump
  and confirmed once.
- **`ideal` is a hint, not a guarantee.** The fake device clamped P0's
  `frameRate: {ideal: 30}` to 20 and reported 20 in `getSettings()`. The
  real camera may clamp too — and `CAPTURE-NEXT.md` already records the
  script asking for 30 fps and the camera delivering 25. **The constraint
  the page ships may not be the constraint the camera honours**, which is a
  third reason to prefer the unconstrained form.

---

## 8. HOW TO REPRODUCE ALL OF THIS

```bash
# the probe, running
cd ~/projects/communicator-acceptance
nohup python3 experiments/camera_probe/probe_server.py > /tmp/camera_probe_server.log 2>&1 &
curl -s http://127.0.0.1:8766/health
# then, on Windows: http://localhost:8766  (or http://localhost:8766/?sweep=1)

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

# which backend the reference capture actually used (it is the one that works)
cd /mnt/c/Users/casey/AppData/Local/Temp/cam_probe
/mnt/c/Python314/python.exe whichbackend.py
#   default cv2.VideoCapture(0) -> backend=MSMF

# the device inventory
powershell.exe -NoProfile -Command "Get-PnpDevice -Class Camera | Select FriendlyName,Status"
```
