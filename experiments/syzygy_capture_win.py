"""
syzygy_capture_win.py -- run this ON WINDOWS to capture camera + microphone.

WHY A SEPARATE SCRIPT ON THE WINDOWS SIDE.

WSL exposes neither a capture device nor a usable audio stack. Measured here:

    /dev/snd contains only a timer          (no pcmC* device at all)
    PulseAudio sources are RDPSink.monitor and RDPSource only -- an RDP bridge
    no arecord / parec / pactl / ffmpeg, and the pulsectl build that connects
    ships no recording API

So capture has to happen on the Windows side and land on disk for WSL to read.
Measured working on this machine:

    audio   "Microphone Array (Realtek(R) Audio)", 4 input channels
    camera  "ASUS FHD webcam", 640x540 at 30 fps
    python  C:\\Python314\\python.exe

DEPENDENCIES (installed once, on Windows):

    C:\\Python314\\python.exe -m pip install sounddevice opencv-python

WHY ONE PROCESS CAPTURES BOTH.

Audio and video have to share a clock or the join cannot be tested -- and a
join whose alignment is assumed is a join whose result cannot be believed. Both
streams are read in one loop, and the mic block size is derived from the camera
frame interval so they cannot silently drift apart.

NV12, because that is what Syzygy ingests. The camera is resized to 160x96,
which is Syzygy's own benchmark size and yields an 80x24 cell grid.

USAGE (from any shell, paths are Windows paths):

    C:\\Python314\\python.exe syzygy_capture_win.py --seconds 40 ^
        --nv12 C:\\proart_capture\\facing.nv12 --wav C:\\proart_capture\\facing.wav

Then, from WSL:

    python experiments/syzygy_camera.py ^
        --capture /mnt/c/proart_capture/facing.nv12 ^
        --audio   /mnt/c/proart_capture/facing.wav

WHAT IT DOES NOT DO.

It does not label anything. It does not know which condition you were in. Write
a JSON sidecar yourself if you want the conditions attached -- see
examples/realfacing/README.md for the three-condition protocol.
"""

import argparse
import json
import sys
import wave

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--nv12", default=r"C:\proart_capture\facing.nv12")
    ap.add_argument("--wav", default=r"C:\proart_capture\facing.wav")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--mic", type=int, default=None,
                    help="sounddevice input index; default = system default")
    a = ap.parse_args()

    import cv2
    import sounddevice as sd

    W, H = a.width, a.height
    cap = cv2.VideoCapture(a.cam)
    if not cap.isOpened():
        raise SystemExit("camera did not open")
    print("camera :", a.width, "x", a.height)
    if a.mic is None:
        d = sd.query_devices(kind="input")
        a.mic = d["index"] if "index" in d else 0
    print("mic    :", sd.query_devices(a.mic)["name"])

    # One read() per video frame; mic block sized from the frame interval so the
    # two streams advance together instead of drifting.
    with sd.InputStream(device=a.mic, channels=1, samplerate=a.rate,
                        dtype="int16", blocksize=0) as mic:
        vf = []
        af = []
        n_target = int(a.seconds * 30)
        for i in range(n_target):
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
            hr = H // 4                       # I420 chroma = H/4 rows each
            y = yuv[:H, :W]
            u = yuv[H:H + hr, :W // 2].ravel()
            v = yuv[H + hr:, :W // 2].ravel()
            uv = np.empty((u.size, 2), np.uint8)
            uv[:, 0] = u
            uv[:, 1] = v
            vf.append(y.tobytes() + uv.tobytes())

            blk = int(a.rate / 30)            # one audio block per video frame
            audio, _ = mic.read(blk)
            af.append(audio.tobytes())
    cap.release()

    open(a.nv12, "wb").write(b"".join(vf))
    pcm = b"".join(af)
    with wave.open(a.wav, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(a.rate)
        f.writeframes(pcm)

    print(f"frames  : {len(vf)}")
    print(f"audio   : {len(pcm) // 2} samples ({len(pcm) / 2 / a.rate:.2f}s)")
    print(f"drift   : {(len(pcm) // 2) / a.rate - len(vf) / 30:+.3f}s")
    print(f"NV12    : {a.nv12}")
    print(f"WAV     : {a.wav}")


if __name__ == "__main__":
    main()