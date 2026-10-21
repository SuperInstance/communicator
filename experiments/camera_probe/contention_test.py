"""
contention_test.py -- is my black frame the CAMERA, or the SECOND CONSUMER?

THE PROBLEM WITH THE EARLIER SWEEP.

The sweep measured mean luma 16.003971354166662 -- digital black -- through
DirectShow, for every constraint set. But Casey has confirmed, at the
browser page, that the camera IS the front camera and DOES respond to light
(a lamp brought into frame registers). A camera that responds to light and
a DirectShow path that returns digital black cannot both be describing the
same moment. So one of them is not seeing the camera.

This script settles it by running TWO consumers of the device at once:

  PRIMARY   opens the camera first and holds it, reading frames.
  SECONDARY waits, then opens a SECOND instance of the same device and
            measures it.

If the primary sees a real image and the secondary sees digital black, then
"black" is what a second consumer gets, the earlier sweep was measuring
contention rather than the camera, and the sweep's conclusion does not
transfer to the browser.

If BOTH see black, the camera really is delivering black right now.

Either way the answer is a measurement, not an argument.
"""

import json
import threading
import time

import cv2
import numpy as np

W, H = 160, 96
FRAMES = 40
OUT = "contention_test.json"


def luma_of(frame):
    s = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(s, cv2.COLOR_BGR2YUV_I420)
    y = yuv[:H, :W]
    return float(y.mean()), int(y.min()), int(y.max())


def consumer(tag, delay, backend, out, ready=None, go=None):
    if delay:
        time.sleep(delay)
    if go is not None:
        go.set()
    try:
        cap = cv2.VideoCapture(0, backend)
    except Exception as exc:
        out[tag] = {"error": f"open threw: {exc}"}
        return
    if not cap.isOpened():
        out[tag] = {"error": "did not open"}
        return
    vals = []
    info = {"opened": True}
    for _ in range(FRAMES):
        ok, frame = cap.read()
        if not ok:
            info["read_failed_at"] = len(vals)
            break
        m, lo, hi = luma_of(frame)
        vals.append(m)
        time.sleep(0.02)
    if vals:
        info.update({
            "luma_mean": float(np.mean(vals)),
            "luma_min": float(np.min(vals)),
            "luma_max": float(np.max(vals)),
            "first5": [round(v, 4) for v in vals[:5]],
            "last5": [round(v, 4) for v in vals[-5:]],
            "frames": len(vals),
            "is_black": float(np.mean(vals)) < 20.0,
        })
    out[tag] = info
    cap.release()


def run(backend, bname):
    out = {}
    primary_ready = threading.Event()
    t1 = threading.Thread(target=consumer,
                          args=("primary", 0, backend, out),
                          daemon=True)
    t1.start()
    time.sleep(3.0)          # let the primary take the device
    t2 = threading.Thread(target=consumer,
                          args=("secondary", 0, backend, out),
                          daemon=True)
    t2.start()
    t1.join(120)
    t2.join(120)
    print(f"=== {bname} ===", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


if __name__ == "__main__":
    results = {}
    for backend, bname in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")):
        try:
            results[bname] = run(backend, bname)
        except Exception as exc:
            results[bname] = {"error": str(exc)}
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    print("wrote " + OUT, flush=True)
