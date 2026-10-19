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
    ap.add_argument("--lock-exposure", dest="lock_exposure",
                    action="store_true", default=True,
                    help="hold auto-exposure still (default on)")
    ap.add_argument("--no-lock-exposure", dest="lock_exposure",
                    action="store_false",
                    help="leave auto-exposure hunting -- this reproduces the "
                         "43.6-count per-frame noise that ruined the first capture")
    ap.add_argument("--exposure-mode", type=float, default=0.75,
                    help="DSHOW manual-exposure selector (default 0.75)")
    ap.add_argument("--exposure", type=float, default=-6.0,
                    help="exposure value once locked (default -6.0)")
    a = ap.parse_args()

    import cv2
    import sounddevice as sd

    W, H = a.width, a.height
    cap = cv2.VideoCapture(a.cam)
    if not cap.isOpened():
        raise SystemExit("camera did not open")
    print("camera :", a.width, "x", a.height)

    # EXPOSURE LOCK. Added 2026-10-18 after the first three-condition capture
    # came back unusable. Measured on that data: the mean absolute difference
    # between ADJACENT FRAMES was 43.6 luma counts, while the entire difference
    # between the facing and head-turned conditions was 14.4. The camera's
    # auto-exposure hunt was ~3x louder than the signal the experiment was
    # built to measure, and 0 of 96 spatial windows could separate the two
    # conditions above that noise floor.
    #
    # On Windows DSHOW, 0.75 selects manual exposure. Other backends use
    # different encodings, so the value actually applied is read back and
    # printed -- if the lock silently fails we want that in the output, not
    # discovered later.
    if a.lock_exposure:
        for prop, val in (("AUTO_EXPOSURE", a.exposure_mode),
                          ("EXPOSURE", a.exposure)):
            try:
                cap.set(getattr(cv2, f"CAP_PROP_{prop}"), val)
            except Exception as exc:  # noqa: BLE001
                print(f"warn   : could not set {prop}: {exc}")
        got_mode = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        got_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        print(f"exposure: lock requested (mode={a.exposure_mode} "
              f"value={a.exposure}); read back mode={got_mode} value={got_exp}")
        if abs(got_mode - a.exposure_mode) > 0.01:
            print("warn   : AUTO_EXPOSURE did not take -- this backend may not "
                  "support it, and per-frame brightness will drift again.")

    native = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("native :", native)

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

    # THE EXPOSURE CONTROL, reported with every capture.
    #
    # This is the single number that decides whether the capture is usable. The
    # failed three-condition run measured 43.6 mean absolute counts between
    # adjacent frames -- the AGC hunting -- while the entire facing-vs-turned
    # signal was 14.4. Anything at or above roughly the signal level means the
    # brightness drift will swamp the thing being measured, and no amount of
    # downstream analysis can recover it.
    #
    # Reported as a ratio rather than a raw count so it can be compared across
    # resolutions and lighting.
    if len(vf) >= 2:
        fr_nv12 = np.frombuffer(b"".join(vf), np.uint8)
        ys = fr_nv12[:len(vf) * W * H].reshape(len(vf), H, W).astype(np.int32)
        means = ys.reshape(len(vf), -1).mean(axis=1)
        mad = np.abs(np.diff(ys, axis=0)).mean()
        print(f"luma    : mean {means.mean():.2f}  std {means.std():.3f}  "
              f"range {means.max() - means.min():.2f}")
        print(f"drift   : adjacent-frame MAD {mad:.3f} counts"
              f"   {'OK - below ~5' if mad < 5 else 'TOO HIGH - exposure is hunting'}")
        if mad >= 5:
            print("warn   : brightness drift exceeds the usable threshold. Re-run "
                  "with --lock-exposure, or move out of the bright window.")


if __name__ == "__main__":
    main()