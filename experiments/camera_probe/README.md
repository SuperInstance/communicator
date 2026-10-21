# camera_probe — the empirical lane for the dark-camera problem

Casey's front camera comes up REALLY dark in Chrome. This directory holds
the instruments that settle it by measurement rather than by argument.

**Findings:** `docs/research/CAMERA-RND-FINDINGS.md`
**Brief:** `docs/research/BRIEF-CAMERA-RND.md`

## The two halves

| file | runs on | what it does |
|---|---|---|
| `probe.html` | **Windows Chrome**, at http://localhost:8766 | the browser half: presets P0–P4, the live 160×96 luma, the `getSettings()` / `getCapabilities()` dump, the manual slider panel, and the LOG button |
| `probe_server.py` | **WSL**, stdlib only | hosts the page on `127.0.0.1:8766` and appends every measurement to `results.jsonl` |
| `win_sweep.py` | **Windows python** (has OpenCV) | the driver half: measures the camera across resolution × frame-rate × exposure-knob settings, with the reference capture's exact luma maths |
| `busy_test.py` | Windows python | is the camera busy, or dark, or broken? |
| `grab_frames.py` | Windows python | captures and characterises what the camera is actually delivering, and saves PNGs |

The browser half cannot answer the driver question, and the driver half
cannot answer the browser question. Both are needed, and both are here.

## Running it

```bash
# the server (WSL). Leave it running.
cd ~/projects/communicator-acceptance
nohup python3 experiments/camera_probe/probe_server.py > /tmp/camera_probe_server.log 2>&1 &
curl -s http://127.0.0.1:8766/health

# then, on Windows, open http://localhost:8766 in Chrome
```

Windows reaches the WSL server on localhost — verified with the Windows
`curl.exe`, so the page is reachable from Chrome on the Windows side.

## The measurement contract

The luma everywhere is the **reference capture's** luma, so every number is
comparable to the .nv12 files on disk:

```
frame -> cv2.resize(frame, (160, 96), INTER_AREA)
      -> cv2.cvtColor(frame, COLOR_BGR2YUV_I420) -> Y plane -> mean
```

and the browser half computes the same quantity in JavaScript with the
capture page's own `lumaFromImage` (BT.601, clamped, rounded), on a 160×96
canvas.

**The reference points, corrected.** The figures that circulated as
110.7 / 114.2 / 49.7 were read at the wrong 23,040-byte stride, on torn
frames. Read correctly (19,200 bytes), the occupied room is **105.9 / 110.2**
and the empty room is **30.1**. The empty-room figure is what the capture
page's own on-screen text gets wrong.

## The automation hooks

`?preset=P0..P4` applies a preset on load, `?log=1` logs a measurement, and
`?selftest=1` does the whole chain — camera, capabilities dump, slider
panel, `applyConstraints`, sample, POST. They exist so the probe can be
verified without a camera:

```
chrome.exe --headless=new --use-fake-device-for-media-stream \
  --use-fake-ui-for-media-stream --user-data-dir=<scratch> \
  "http://127.0.0.1:8766/?selftest=1"
```

That verifies the *probe*. It does not measure the *camera* — the fake
device's numbers are not a measurement of anything real, and any record it
produces is labelled `fake_device_0` so it can be excluded.

## `results.jsonl`

One JSON object per measurement, appended. The fields are the preset, the
manual slider state, the constraint set, `getSettings()`, a capabilities
summary, and the luma mean/min/max — plus `stream_is_black` and
`luma_black_fraction`, because a stream that is *digitally black* is a
different fault from a stream that is merely dark, and the two need
different fixes.
