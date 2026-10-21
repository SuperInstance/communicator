"""
busy_test.py -- is the ASUS FHD webcam busy, or broken, or dark?

Three questions, one script:
  1. DSHOW index 0 -- does the OPEN block? (it blocks when another process
     holds the device)
  2. MSMF index 0  -- does it open, and can it deliver a frame?
  3. what error comes back, verbatim

The distinction matters for the R&D: "the camera is dark" and "the camera is
held by the browser that is measuring it" look identical from the outside.
"""

import json
import threading
import time

import cv2

out = {}


def dshow_open(idx, timeout=10.0):
    box = {}

    def worker():
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            box["opened"] = cap.isOpened()
            box["cap"] = cap
        except Exception as exc:
            box["error"] = str(exc)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return {"result": "TIMEOUT", "detail": f"open blocked > {timeout}s"}
    if box.get("opened"):
        cap = box["cap"]
        props = {n: cap.get(getattr(cv2, "CAP_PROP_" + n))
                 for n in ("FRAME_WIDTH", "FRAME_HEIGHT", "FPS")}
        ok, frame = cap.read()
        res = {"result": "OPENED", "properties": props,
               "first_read_ok": bool(ok),
               "frame_shape": list(frame.shape) if ok else None}
        cap.release()
        return res
    return {"result": "NOT_OPENED", "detail": box.get("error")}


def msmf_read(idx, timeout=10.0):
    box = {}

    def worker():
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
            box["opened"] = cap.isOpened()
            if not box["opened"]:
                return
            ok, frame = cap.read()
            box["read_ok"] = bool(ok)
            box["shape"] = list(frame.shape) if ok else None
            if ok:
                import numpy as np
                small = cv2.resize(frame, (160, 96), interpolation=cv2.INTER_AREA)
                yuv = cv2.cvtColor(small, cv2.COLOR_BGR2YUV_I420)
                box["luma_mean"] = float(yuv[:96, :160].mean())
            cap.release()
        except Exception as exc:
            box["error"] = str(exc)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return {"result": "TIMEOUT", "detail": f"read blocked > {timeout}s"}
    return box


out["dshow_index0"] = dshow_open(0)
print("DSHOW index 0:", json.dumps(out["dshow_index0"]), flush=True)
out["msmf_index0"] = msmf_read(0)
print("MSMF  index 0:", json.dumps(out["msmf_index0"]), flush=True)

with open("busy_test.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("wrote busy_test.json", flush=True)
