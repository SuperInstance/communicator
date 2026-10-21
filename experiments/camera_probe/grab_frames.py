"""
grab_frames.py -- capture what the camera is ACTUALLY delivering, and save
the frames as PNGs so they can be looked at.

The sweep said: every resolution, every frame rate, every exposure setting
gives the SAME mean luma, to fifteen decimal places. That is not a camera
responding; that is a constant frame. So: what does the frame contain?
"""

import json

import cv2
import numpy as np

OUT = []
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

frames = []
for i in range(5):
    ok, frame = cap.read()
    if not ok:
        print(f"frame {i}: read failed", flush=True)
        break
    frames.append(frame)
    cv2.imwrite(f"frame_{i}.png", frame)

cap.release()
print(f"captured {len(frames)} frames at "
      f"{frames[0].shape if frames else None}", flush=True)

if frames:
    f0 = frames[0]
    b, g, r = f0[:, :, 0], f0[:, :, 1], f0[:, :, 2]
    print("--- frame 0 channel stats (BGR) ---", flush=True)
    for name, ch in (("B", b), ("G", g), ("R", r)):
        print(f"  {name}: min {ch.min():3d}  max {ch.max():3d}  "
              f"mean {ch.mean():8.4f}  std {ch.std():7.4f}", flush=True)
    print(f"  unique colours in frame 0: {len(np.unique(f0.reshape(-1, 3), axis=0))}",
          flush=True)
    # are consecutive frames identical?
    for i in range(1, len(frames)):
        same = np.array_equal(frames[0], frames[i])
        diff = float(np.abs(frames[0].astype(np.int32)
                             - frames[i].astype(np.int32)).mean())
        print(f"  frame0 vs frame{i}: identical={same}  mean abs diff {diff:.4f}",
              flush=True)
    # the reference luma, and what the Y plane looks like
    small = cv2.resize(f0, (160, 96), interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(small, cv2.COLOR_BGR2YUV_I420)
    y = yuv[:96, :160]
    print(f"--- the 160x96 Y plane (what the takes record) ---", flush=True)
    print(f"  mean {y.mean():.6f}  min {y.min()}  max {y.max()}  "
          f"std {y.std():.6f}", flush=True)
    print(f"  Y row 0, first 12 values: {list(y[0][:12])}", flush=True)
    print(f"  Y column 0, first 6 values: {list(y[:6, 0])}", flush=True)
    print(f"  Y unique values: {len(np.unique(y))}", flush=True)
    cv2.imwrite("y_plane.png", y)
    # and the small bgr for viewing
    cv2.imwrite("small_160x96.png", small)
    OUT.append({"captured": len(frames), "shape": list(f0.shape)})

with open("frames.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=2)
print("wrote frame_*.png, y_plane.png, small_160x96.png", flush=True)
