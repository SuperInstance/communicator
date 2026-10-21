"""
win_sweep2.py -- the constraint sweep on the RIGHT backend.

THE CORRECTION.

The first sweep used DirectShow and reported digital black for every
constraint set. That was a measurement error, and the evidence is in this
repo:

  * `corpus_capture_win.py` opens the camera with `cv2.VideoCapture(a.cam)`
    -- no backend -- and OpenCV 5.0.0 on Windows resolves that to **MSMF**,
    Media Foundation, which is the SAME stack Chrome uses.
  * `cv2.VideoCapture(0).getBackendName()` returns "MSMF".
  * DirectShow on this UVC device negotiates badly: it opens, it returns
    frames, and those frames are the black offset with a handful of
    uninitialised bytes in the corner.

So the first sweep measured the WRONG STACK and was measuring the stack's
failure, not the camera's. This sweep uses the default backend (MSMF) and
reports which backend it actually got.

THE QUESTION, RESTATED.

Does the frame rate constraint cap the exposure? A UVC camera's maximum
exposure is one frame time, so pinning 30 fps caps the shutter near 33 ms.
If the driver behaves differently under an explicit constraint than under
its default, then the browser's `{frameRate: {ideal: 30}}` is what makes
the page dark, and dropping it is the fix. The reference capture set NO
constraints and got 105.9.

THE LUMA IS THE REFERENCE'S EXACT MATHS, so the numbers compare directly
with the .nv12 captures on disk.
"""

import json
import time

import cv2
import numpy as np

W, H = 160, 96
N_FRAMES = 25
OUT = "win_sweep2.json"

results = {"opencv": cv2.__version__, "sweep": [], "knobs": []}


def ref_luma(cap, n=N_FRAMES):
    vals = []
    t0 = time.perf_counter()
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        vals.append(float(yuv[:H, :W].mean()))
    if not vals:
        return None
    dt = time.perf_counter() - t0
    return {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
            "max": float(np.max(vals)), "n": len(vals),
            "delivered_fps": round(len(vals) / dt, 2) if dt > 0 else None}


def open_cam(backend=cv2.CAP_ANY):
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        return None
    return cap


def save():
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def sweep():
    configs = [
        {"tag": "reference-style (no constraints)", "w": None, "h": None, "fps": None},
        {"tag": "P0-style 640x480@30", "w": 640, "h": 480, "fps": 30},
        {"tag": "640x480, no fps pin", "w": 640, "h": 480, "fps": None},
        {"tag": "P3-style 1280x720, no fps pin", "w": 1280, "h": 720, "fps": None},
        {"tag": "1280x720@30", "w": 1280, "h": 720, "fps": 30},
        {"tag": "1280x720@15", "w": 1280, "h": 720, "fps": 15},
    ]
    for cfg in configs:
        cap = open_cam()
        if cap is None:
            results["sweep"].append({**cfg, "error": "did not open"})
            save()
            continue
        entry = dict(cfg)
        entry["backend"] = cap.getBackendName()
        if cfg["w"]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["w"])
        if cfg["h"]:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["h"])
        if cfg["fps"]:
            cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
        time.sleep(0.5)
        entry["properties"] = {
            "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
            "gain": cap.get(cv2.CAP_PROP_GAIN),
        }
        entry["luma"] = ref_luma(cap)
        cap.release()
        l = entry["luma"]
        print(f"{cfg['tag']:<38} [{entry['backend']}] -> " +
              (f"luma {l['mean']:7.2f}  frames {l['n']}  "
               f"delivered_fps {l['delivered_fps']}"
               if l else "NO FRAMES"), flush=True)
        results["sweep"].append(entry)
        save()


if __name__ == "__main__":
    print(f"openCV {cv2.__version__}", flush=True)
    sweep()
    save()
    print("wrote " + OUT, flush=True)
