"""
corpus_capture_win.py -- collect LABELLED motion data on Windows.

WHY THIS EXISTS.

Casey's call: "could we simply record me and use a traditional model to analyze
and be the teacher for the system". Yes -- and the design below follows from
what the measurements have actually established, not from a generic wish.

WHAT THE MEASUREMENTS SAY THE DATA MUST LOOK LIKE.

1. THE SIGNAL IS MOTION, NOT APPEARANCE. `ROW-BAND-REMOVAL.md` measured
   facing-and-speaking against facing-and-silent: they differ by 4.914 counts
   against a within-condition split of 1.917, with 79% of the excess
   concentrated in the top 5% of pixels. A single still frame does not contain
   that information -- it is in the frame-to-frame DIFFERENCE. So the corpus
   stores differences, not frames. This also happens to make the corpus small
   enough to keep hundreds of sessions.

2. THE NUISANCE MUST BE REMOVED AT CAPTURE TIME, NOT AT TRAINING TIME. The
   camera's auto-exposure produces horizontal LED-flicker bands, and an empty
   room flips 50.00% of its dots per frame without it. Row-wise removal
   (subtract each row's median across the columns) takes that to 0.01% and
   leaves the subject's column structure untouched. Doing it here means the
   stored corpus is already clean, and the same code path is what the on-device
   system will run.

3. THE TEACHER MUST BE BENCHMARKED AGAINST GROUND TRUTH BEFORE IT IS TRUSTED.
   So the corpus has two kinds of session, and the distinction is in the
   label:
     - CONDITIONS (A, B, C, D, ...) are performed deliberately and their label
       is ground truth by construction. These are the evaluation set AND the
       teacher benchmark.
     - FREE (F...) is just Casey being in the room, with no condition
       asserted. These carry no ground truth; they exist to be labelled by the
       teacher, and their value is entirely dependent on the teacher being
       good.
   A teacher that cannot beat chance on the CONDITIONS set must not be allowed
   to label the FREE set. That is the gate.

4. THE STUDENT IS THE THING THAT SHIPS. The teacher runs offline, slowly, on
   a workstation. The student is small, local, and always-on in the glasses.
   The teacher's job is to make labels for data Casey will not sit and label
   by hand, so the corpus can be far larger than the condition set.

WHAT IS STORED, AND WHY IT IS THIS SMALL.

Per frame, the row-band-removed change map |frame(t+1) - frame(t)|, downsampled
to 80x48 and quantised to uint8. That is 3.8 kB per frame, about 7 MB per
minute of session, and it is the representation the student will actually
consume. Storing raw 640x480 video would be 400x larger and would preserve
information the pipeline has already decided to discard.

The raw audio is stored alongside, because the audio channel is not dead -- it
just failed to resolve INDIVIDUALS. It may still carry complementary
information, and a join is the thing the project is trying to build.

THE DRIVER, SO RECORDING IS NOT A BURDEN.

`--plan` runs the whole condition sequence automatically with a countdown
between conditions, so a session is one command and no clipboard work. Casey
turns his head, talks, or stays still; the script labels the take and checks
the label was actually achieved.

THE ACHIEVEMENT CHECK, because a mislabelled take is worse than no take.

Each condition declares how it should be verifiable, and the script measures it
rather than trusting the performer:
  - speaking conditions: the gate must report speech over a usable fraction of
    the window, else the take is marked VOICE_MISSING
  - silent conditions: the gate must report almost no speech
  - the pose conditions additionally get a change-map comparison against the
    calibration condition, so a head turn that did not happen is caught
A take that fails its check is still saved, but flagged, so it can be excluded
without being re-recorded blindly.

USAGE (Windows):

    C:\\Python314\\python.exe corpus_capture_win.py --plan --out C:\\proart_capture\\corpus
    C:\\Python314\\python.exe corpus_capture_win.py --condition A --seconds 30 --out C:\\proart_capture\\corpus
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

# The condition plan. `kind` drives the achievement check; `pose` and
# `speak` describe what the performer should do.
PLAN = [
    {"name": "A", "kind": "condition", "speak": True,  "pose": "facing",
     "note": "facing the camera, speaking the standard paragraph"},
    {"name": "B", "kind": "condition", "speak": True,  "pose": "turned",
     "note": "head turned fully away (90 deg, at the wall), same paragraph"},
    {"name": "C", "kind": "condition", "speak": False, "pose": "facing",
     "note": "facing the camera, completely silent"},
    {"name": "D", "kind": "condition", "speak": False, "pose": "turned",
     "note": "head turned fully away, completely silent"},
    {"name": "F1", "kind": "free", "speak": None, "pose": "any",
     "note": "free session: just be in the room, talk, read, move"},
]

W, H = 160, 96
MAP_W, MAP_H = 80, 48          # stored change-map resolution
RATE = 16000
FPS = 30

# THE CONDITION PLAN.
#
# REPETITIONS, AND WHY THEY ARE THE POINT.
#
# `docs/research/TEACHER-PIPELINE.md` measured why one session per condition
# cannot work: a conventional teacher scored 1.000 on a temporal split of a
# single session and 0.000 on leave-one-session-out, for every feature set
# tried. It had learned what the room looked like, not what speaking looks
# like. The only split that can tell those apart needs several INDEPENDENT
# sessions, so the plan records each condition REPEATS times, as separate
# recordings, with a rest between them.
#
# The repetition count is the parameter that unlocks everything downstream.
# Three is the minimum that makes leave-one-session-out meaningful; more is
# better, and the free sessions can run for as long as Casey likes.
#
# `kind` drives the achievement check; `pose` and `speak` describe what the
# performer should do.
REPEATS = 3

_CONDITIONS = [
    {"name": "A", "kind": "condition", "speak": True,  "pose": "facing",
     "note": "facing the camera, speaking the standard paragraph"},
    {"name": "B", "kind": "condition", "speak": True,  "pose": "turned",
     "note": "head turned fully away (90 deg, at the wall), same paragraph"},
    {"name": "C", "kind": "condition", "speak": False, "pose": "facing",
     "note": "facing the camera, completely silent"},
    {"name": "D", "kind": "condition", "speak": False, "pose": "turned",
     "note": "head turned fully away, completely silent"},
    {"name": "F", "kind": "free", "speak": None, "pose": "any",
     "note": "free session: just be in the room, talk, read, move"},
]

PLAN = []
for _rep in range(REPEATS):
    for _c in _CONDITIONS:
        _c = dict(_c)
        _c["rep"] = _rep + 1
        _c["name"] = f"{_c['name']}{_rep + 1}"
        PLAN.append(_c)
# One long free session at the end: the corpus the teacher will label.
_free = dict(_CONDITIONS[-1])
_free["name"] = "FREE"
_free["rep"] = 1
_free["long"] = True
PLAN.append(_free)


def row_band_remove(ys: np.ndarray) -> np.ndarray:
    """Remove the row-constant component: the LED flicker banding.

    Per-row MEDIAN across the columns, not mean -- a subject occupying a
    minority of a row's columns must not drag the estimate, and the median
    ignores it. Verified in experiments/row_band_removal.py: an empty room's
    row-profile std drops 45.98 -> 0.29 while the subject's column structure
    is untouched (7.08 -> 7.18).
    """
    prof = np.median(ys, axis=2)
    g = prof.mean(axis=1, keepdims=True)
    return ys - (prof - g)[:, :, None]


def change_maps(ys: np.ndarray) -> np.ndarray:
    """Per-frame |frame(t+1) - frame(t)|, row-band-removed, downsampled."""
    d = np.abs(np.diff(ys, axis=0))
    return d.reshape(d.shape[0], H // 2, 2, W // 2, 2).mean(axis=(2, 4))


def quantise(maps: np.ndarray) -> np.ndarray:
    """uint8 with a scale recorded alongside, so nothing is silently clipped."""
    hi = float(np.percentile(maps, 99.5)) if maps.size else 1.0
    hi = max(hi, 1e-6)
    q = np.clip(maps / hi * 255.0, 0, 255).astype(np.uint8)
    return q, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\proart_capture\corpus")
    ap.add_argument("--condition", default=None,
                    help="single condition name, e.g. A or F1")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--long-seconds", type=float, default=300.0,
                    help="length of the trailing free session")
    ap.add_argument("--plan", action="store_true",
                    help="run the whole condition sequence")
    ap.add_argument("--countdown", type=int, default=5)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--mic", type=int, default=None)
    ap.add_argument("--width", type=int, default=W)
    ap.add_argument("--height", type=int, default=H)
    args = ap.parse_args()

    import cv2
    import sounddevice as sd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plan = PLAN
    if args.condition:
        match = [p for p in PLAN if p["name"] == args.condition]
        if not match:
            sys.exit(f"unknown condition {args.condition!r}; "
                     f"known: {[p['name'] for p in PLAN]}")
        plan = match

    print("librosa/sounddevice available:", bool(sd))
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        sys.exit("camera did not open")
    native = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("native:", native)

    if args.mic is None:
        d = sd.query_devices(kind="input")
        args.mic = d["index"] if "index" in d else 0
    print("mic   :", sd.query_devices(args.mic)["name"])

    for spec in plan:
        name = spec["name"]
        secs = args.seconds
        if spec.get("long"):
            secs = args.long_seconds
        print("\n" + "=" * 62)
        print(f"CONDITION {name}: {spec['note']}")
        print("=" * 62)
        for t in range(args.countdown, 0, -1):
            print(f"  starting in {t}...", flush=True)
            time.sleep(1)

        ys, af = [], []
        n_target = int(secs * FPS)
        with sd.InputStream(device=args.mic, channels=1, samplerate=RATE,
                            dtype="int16", blocksize=0) as mic:
            for i in range(n_target):
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(frame, (args.width, args.height),
                                       interpolation=cv2.INTER_AREA)
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                ys.append(yuv[:args.height, :args.width].tobytes())
                blk = int(RATE / FPS)
                audio, _ = mic.read(blk)
                af.append(audio.tobytes())
        cap.release()

        if len(ys) < 10:
            print("  too few frames captured; skipping")
            continue

        Y = np.frombuffer(b"".join(ys), np.uint8).reshape(len(ys), args.height,
                                                          args.width).astype(np.float64)
        Yr = row_band_remove(Y)
        maps = change_maps(Yr)
        q, scale = quantise(maps)

        # -- the drift diagnostic, the number that decides usability ---------
        raw_mad = float(np.abs(np.diff(Y, axis=0)).mean())
        row_mad = float(np.abs(np.diff(Yr, axis=0)).mean())
        print(f"  frames {len(ys)}   drift raw {raw_mad:.2f}   "
              f"after row-band removal {row_mad:.2f}   "
              f"{'OK' if row_mad < 5 else 'TOO HIGH -- change the lighting'}")
        if row_mad >= 5:
            print("  warn: the banding is not the only problem. The take is "
                  "saved but the drift will dominate the motion field.")

        # -- the achievement check -------------------------------------------
        speech_frac = None
        try:
            pcm = b"".join(af)
            wave_path = out / f"{name}_{int(time.time())}.wav"
            with wave.open(str(wave_path), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(RATE)
                f.writeframes(pcm)
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from src.wavio import read_wav
            from src.vad import SileroGate
            a = read_wav(wave_path)
            probs = SileroGate().push(list(a.samples))
            if probs:
                speech_frac = float(sum(1 for v in probs if v >= 0.5) / len(probs))
                want = spec.get("speak")
                if want is True and speech_frac < 0.15:
                    print(f"  warn: VOICE_MISSING -- only {speech_frac * 100:.1f}% "
                          "of frames over the speech threshold")
                if want is False and speech_frac > 0.05:
                    print(f"  warn: NOT_SILENT -- {speech_frac * 100:.1f}% of frames "
                          "over the speech threshold")
                if want is False:
                    print(f"  silent condition, measured speech {speech_frac * 100:.1f}%")
                else:
                    print(f"  speech {speech_frac * 100:.1f}% of frames")
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: speech check unavailable: {exc}")

        stamp = int(time.time())
        npz = out / f"{name}_{stamp}.npz"
        np.savez_compressed(npz, maps=q, scale=np.float32(scale))
        meta = {
            "condition": name,
            "kind": spec["kind"],
            "note": spec["note"],
            "pose": spec["pose"],
            "speak": spec.get("speak"),
            "speech_frac": speech_frac,
            "n_frames": int(len(ys)),
            "seconds": len(ys) / FPS,
            "map_shape": list(q.shape[1:]),
            "map_scale": scale,
            "drift_raw": raw_mad,
            "drift_after_row_removal": row_mad,
            "timestamp": stamp,
            "rep": spec.get("rep"),
            "filter": "row+median3",
        }
        (out / f"{name}_{stamp}.json").write_text(json.dumps(meta, indent=2))
        print(f"  wrote {npz.name}  ({npz.stat().st_size / 1024:.0f} kB)")

    print("\ndone")


if __name__ == "__main__":
    main()
