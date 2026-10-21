"""
win_sweep.py -- THE DECISIVE EXPERIMENT: how does this camera's brightness
respond to the constraint sets the browser could ship?

WHY THIS IS THE DECISIVE EXPERIMENT.

The reference capture on this machine -- corpus_capture_win.py -- put the
occupied room at mean luma 105.9 (A) and 110.2 (B), read at the CORRECT
19,200-byte stride. That capture set NO constraints: `cv2.VideoCapture(cam)`
and whatever the driver chose. The browser page DOES constrain:
{width: ideal 640, height: ideal 480, frameRate: ideal 30}.

Casey's hypothesis is that pinning the frame rate caps the exposure (a UVC
camera's maximum exposure is one frame time, so 30 fps caps the shutter at
33 ms), and that the driver behaves differently under explicit constraints
than under defaults.

This sweep measures the camera's luma across exactly that axis: resolution x
frame rate, plus the driver's own exposure knobs. The luma is the reference's
exact maths:

    frame -> cv2.resize(frame, (160,96), INTER_AREA)
          -> cv2.cvtColor(frame, COLOR_BGR2YUV_I420) -> Y plane -> mean

DSHOW is used, because that is the backend that works while Chrome holds the
Media Foundation source.

CAUTION, STATED UP FRONT: this is the driver, not the browser. It does not
prove what Chrome's getUserMedia will do -- that is what the probe page is
for. What it does is establish whether the camera CAN be this bright under
each constraint set, and what the driver's own knobs do. The probe page then
measures the browser's version of the same question.
"""

import json
import time

import cv2
import numpy as np

W, H = 160, 96
N_FRAMES = 20
OUT = "win_sweep.json"


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


def sweep():
    results = []
    # the constraint sets that matter: what the product ships now (P0),
    # 720p with no fps pin (P3), the driver's own default (what the
    # reference capture used), and the frame-rate axis on its own.
    configs = [
        {"tag": "P0-like 640x480@30", "w": 640, "h": 480, "fps": 30},
        {"tag": "640x480 unconstrained", "w": 640, "h": 480, "fps": None},
        {"tag": "P3-like 1280x720", "w": 1280, "h": 720, "fps": None},
        {"tag": "1280x720@30", "w": 1280, "h": 720, "fps": 30},
        {"tag": "1280x720@15", "w": 1280, "h": 720, "fps": 15},
        {"tag": "1920x1080 unconstrained", "w": 1920, "h": 1080, "fps": None},
        {"tag": "driver default", "w": None, "h": None, "fps": None},
    ]
    for cfg in configs:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            results.append({**cfg, "error": "did not open"})
            continue
        entry = dict(cfg)
        if cfg["w"]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["w"])
        if cfg["h"]:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["h"])
        if cfg["fps"]:
            cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
        time.sleep(0.4)  # let the driver settle into the new mode
        entry["properties"] = {
            "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "gain": cap.get(cv2.CAP_PROP_GAIN),
            "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        }
        entry["luma"] = ref_luma(cap)
        cap.release()
        lum = entry["luma"]
        print(f"{cfg['tag']:<26} -> "
              f"{entry['properties']['width']:.0f}x{entry['properties']['height']:.0f} "
              f"@ {entry['properties']['fps']:.0f}  "
              f"luma {lum['mean']:.1f}" if lum else
              f"{cfg['tag']:<26} -> no frames", flush=True)
        results.append(entry)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results


def knob_response():
    """What the driver's OWN exposure knobs do to the luma, at the shipped
    640x480@30 setting. If moving them does nothing, the browser has no
    lever and the constraint set is the only lever."""
    out = []
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return [{"error": "did not open"}]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    time.sleep(0.4)
    for prop, values in (("EXPOSURE", [-8, -4, 0, 4, 8]),
                         ("GAIN", [0, 32, 64, 128, 255]),
                         ("BRIGHTNESS", [-64, 0, 64])):
        cid = getattr(cv2, "CAP_PROP_" + prop, None)
        if cid is None:
            continue
        readings = []
        for v in values:
            cap.set(cid, float(v))
            time.sleep(0.35)
            r = ref_luma(cap)
            readings.append({"set": v, "readback": cap.get(cid),
                             "luma": r["mean"] if r else None})
        print(f"{prop}: " + ", ".join(f"{x['set']}->{x['luma']}"
                                      for x in readings), flush=True)
        out.append({"prop": prop, "readings": readings})
    cap.release()
    return out


if __name__ == "__main__":
    print("=== constraint sweep ===", flush=True)
    res = sweep()
    print("\n=== driver knob response at 640x480@30 ===", flush=True)
    knobs = knob_response()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"sweep": res, "knobs": knobs}, f, indent=2)
    print("wrote " + OUT, flush=True)
