"""
corpus_capture_win.py -- collect LABELLED motion data on Windows.

WHY THIS EXISTS.

Casey's call: "could we simply record me and use a traditional model to analyze
and be the teacher for the system". Yes -- and the design below follows from
what the measurements have actually established, not from a generic wish.

WHAT THE MEASUREMENTS SAY THE DATA MUST LOOK LIKE.

1. THE SIGNAL IS MOTION, NOT APPEARANCE. `ROW-BAND-REMOVAL.md` measured
   facing-and-speaking against facing-and-silent: they differ by 4.914 counts
   against A's own within-condition split of 2.878 (1.7x), with 79% of the excess
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

   > ## CORRECTION (2026-10-20) -- THE NUISANCE WAS NEVER THERE, AND THE
   > FILTER IS NOW OPTIONAL
   >
   > The 50.00% was an ARTEFACT. Every reader in the repo read the captures at
   > a 23,040-byte frame stride when the files are 19,200, so 80% of every
   > "frame" was a tear between two real moments -- and a tear IS a horizontal
   > band. The full account is `docs/research/NV12-STRIDE.md`.
   >
   > On the corrected frames the empty room shows **0.0000% banded pixels with
   > NO filter at all**, where it had shown a perfect 50.00%. The occupied
   > rate falls 48.19% -> 0.62%. And because there is no longer a nuisance to
   > remove, the row-removal filter now makes the occupied rate **WORSE**
   > (1.79%). `experiments/combined_filter.py` puts it as: "THE FILTERS ARE
   > NOT NEEDED, AND THE PREMISE IS WITHDRAWN ... The honest recommendation
   > is to use the raw representation, keep the zero latency, and re-run the
   > teacher on it."
   >
   > So a filter whose justification is withdrawn must not be BAKED INTO THE
   > CAPTURE, because a capture cannot be re-filtered once the raw frames are
   > thrown away. This tool now stores, per take:
   >
   >   `maps`      the row-removed change maps   (unchanged key, unchanged
   >               meaning -- every existing consumer keeps working)
   >   `maps_raw`  the UNFILTERED change maps, with their own scale
   >   `luma`      the raw luma frames, unfiltered
   >
   > and the metadata records which is which, so the representation is a
   > POST-HOC DECISION rather than a permanent property of the recording.
   >
   > The raw luma is the part that matters most. The night's largest result --
   > the speaking signal in the mouth, +0.733 sd against a +0.477 sd cheek
   > control -- is measured on LUMA, not on change maps, and so are the
   > MediaPipe face detections, the pose test, the room map and the
   > empty-room control. A corpus of change maps alone could not carry the
   > repo's best current finding, so storing the luma is what makes this
   > corpus usable by the lanes that matter.

3. THE TEACHER MUST BE BENCHMARKED AGAINST GROUND TRUTH BEFORE IT IS TRUSTED.
   So the corpus has two kinds of session, and the distinction is in the
   label:
     - CONDITIONS (A, B, C, D, ...) are performed deliberately and their label
       is ground truth by construction. These are the evaluation set AND the
       teacher benchmark.
     - FREE (F...) is just Casey being in the room, with no condition
       asserted. These carry no ground truth; they exist to be labelled by
       the teacher, and their value is entirely dependent on the teacher being
       good.
   A teacher that cannot beat chance on the CONDITIONS set must not be allowed
   to label the FREE set. That is the gate.

4. THE STIMULUS IS FIXED, so delivery is consistent across takes. See
   `src/stimulus.py` -- Casey recites the same poem every time. "The standard
   paragraph" is not a controlled stimulus: a paragraph read from a page is
   read differently on every take, so a detector trained on it has to separate
   speech from not-speech using variation that is not the thing being
   detected.

5. THE AUDIO AND THE VIDEO MUST ACTUALLY CORRESPOND, AND THAT IS NOW
   CHECKED RATHER THAN ASSUMED. `experiments/av_sync.py` measured the
   existing captures and found NO correspondence at any lag, at any candidate
   sample rate, with any of three motion features -- the audio and the video
   of A/B/C are two separate recordings. The old captures were made by a
   script that recorded 30 s of audio and 25 s of video, separately, and
   nothing checked that they showed the same thing.

   So this script takes a video frame and an audio block in the SAME loop
   iteration, stamps every iteration with the wall clock, records the
   ACHIEVED frame rate rather than the assumed one, and writes the WAV header
   with the microphone's ACTUAL sample rate rather than the requested one.
   All three of those were wrong in the old captures:
     - 750 frames were assumed to be 25 s of video; the camera was delivering
       at a different rate, so the video's time base was stretched.
     - the WAV header claimed 16000 Hz when the captures imply 19188 Hz --
       a 1.2x error in every audio timestamp ever computed from them.
     - nothing recorded that the two streams had different lengths.

   A take is now self-describing: `timing.json` next to it holds the frame
   timestamps, so the correspondence is verifiable after the fact instead of
   taken on trust.

WHAT IS STORED, AND WHY IT IS THIS SMALL.

Per frame, the row-band-removed change map |frame(t+1) - frame(t)|, downsampled
to 80x48 and quantised to uint8. That is 3.8 kB per frame, about 7 MB per
minute of session, and it is the representation the student will actually
consume. Storing raw 640x480 video would be 400x larger and would preserve
information the pipeline has already decided to discard.

The raw audio is stored alongside, because the audio channel is not dead --
it just failed to resolve INDIVIDUALS. It may still carry complementary
information, and a join is the thing the project is trying to build. It is
stored with the sample rate it was ACTUALLY captured at.

THE DRIVER, so RECORDING IS NOT A BURDEN.

`--plan` runs the whole condition sequence automatically with a countdown
between conditions, so a session is one command and no clipboard work. Casey
turns his head, recites the poem, or stays still; the script labels the take
and checks the label was actually achieved.

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

THE CONDITION SET, and why D exists.

  A  facing,  speaking     the poem, to the camera
  B  turned,  speaking     the poem, head fully away -- SAME WORDS as A, so
                            the audio is nominally identical and ONLY the
                            picture differs. That makes A-vs-B a clean test.
  C  facing,  silent       the person present, mouth closed, nothing said
  D  turned,  silent       the person present and turned away, nothing said
  F  free                  just be in the room. No label. The teacher labels
                            these, once the teacher is trusted.

D has never been recorded, and without it "the model sees the head turn" and
"the model sees the mouth moving" are the same hypothesis.

USAGE (Windows):

    C:\\Python314\\python.exe corpus_capture_win.py --plan --out C:\\proart_capture\\corpus
    C:\\Python314\\python.exe corpus_capture_win.py --condition A --seconds 50 --out C:\\proart_capture\\corpus
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
REPEATS = 3

_CONDITIONS = [
    {"name": "A", "kind": "condition", "speak": True,  "pose": "facing",
     "seconds": 50,
     "note": "facing the camera, reciting the poem from memory"},
    {"name": "B", "kind": "condition", "speak": True,  "pose": "turned",
     "seconds": 50,
     "note": "head turned fully away (90 deg, at the wall), SAME poem"},
    {"name": "C", "kind": "condition", "speak": False, "pose": "facing",
     "seconds": 30,
     "note": "facing the camera, completely silent"},
    {"name": "D", "kind": "condition", "speak": False, "pose": "turned",
     "seconds": 30,
     "note": "head turned fully away, completely silent"},
    {"name": "F", "kind": "free", "speak": None, "pose": "any",
     "seconds": 60,
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

W, H = 160, 96
MAP_W, MAP_H = 80, 48          # stored change-map resolution
RATE = 16000
FPS = 30


def _find_src():
    """Locate the `src` package, so the achievement check actually runs.

    This was silently broken. The script did
    `sys.path.insert(0, Path(__file__).resolve().parents[1])`, which for
    C:\\proart_capture\\corpus_capture_win.py is C:\\ -- not the repo. So the
    import of src.vad failed, the exception was swallowed by a bare except,
    and every take was saved with "speech check unavailable" and NO
    VERIFICATION AT ALL. The bundle ships `src\\` next to this file, so it is
    found here; the repo checkout is the fallback.
    """
    here = Path(__file__).resolve().parent
    for cand in (here, here.parent, here.parent.parent):
        if (cand / "src" / "vad.py").is_file():
            return cand
    return None


def row_band_remove(ys: np.ndarray) -> np.ndarray:
    """Remove the row-constant component.

    Per-row MEDIAN across the columns, not mean -- a subject occupying a
    minority of a row's columns must not drag the estimate, and the median
    ignores it.

    > CORRECTION (2026-10-20). The original justification for this function
    > was LED flicker banding, and the numbers cited with it -- an empty
    > room's row-profile std dropping 45.98 -> 0.29 while the subject's
    > column structure went 7.08 -> 7.18 -- were measured on TORN frames. The
    > "banding" was the tear (see the module docstring and
    > `docs/research/NV12-STRIDE.md`). On the corrected frames there is no
    > flicker nuisance to remove, and applying this filter makes the occupied
    > banded rate slightly WORSE (1.79% against 0.62% with no filter).
    >
    > The function is KEPT, because it is cheap, because the on-device path
    > may still want it, and because removing it now would change the meaning
    > of the existing `maps` key. But it is no longer applied to the only
    > copy of the data: the raw change maps and the raw luma are stored too.
    """
    prof = np.median(ys, axis=2)
    g = prof.mean(axis=1, keepdims=True)
    return ys - (prof - g)[:, :, None]


def change_maps(ys: np.ndarray) -> np.ndarray:
    """Per-frame |frame(t+1) - frame(t)|, downsampled by 2x2.

    The caller decides whether `ys` has been row-band-removed first; this
    function does not filter anything, and the original docstring said it did,
    which was a description of the call site rather than of the function.
    """
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
                    help="single condition name, e.g. A1 or F1")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="override the per-condition duration")
    ap.add_argument("--long-seconds", type=float, default=300.0,
                    help="length of the trailing free session")
    ap.add_argument("--plan", action="store_true",
                    help="run the whole condition sequence")
    ap.add_argument("--countdown", type=int, default=5)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--mic", type=int, default=None)
    ap.add_argument("--width", type=int, default=W)
    ap.add_argument("--height", type=int, default=H)
    ap.add_argument("--stimulus-file", default=None,
                    help="use this text as the stimulus instead of the "
                         "default poem")
    a = ap.parse_args()

    import cv2
    import sounddevice as sd

    src_root = _find_src()
    SileroGate = read_wav = None
    if src_root:
        sys.path.insert(0, str(src_root))
        try:
            from src.vad import SileroGate
            from src.wavio import read_wav
        except Exception as exc:  # noqa: BLE001
            print(f"warn: could not import the gate ({exc}); the speech "
                  "achievement check will NOT run.")
    else:
        print("warn: could not find src/ -- the speech achievement check "
              "will NOT run, and every take will be unverified.")

    try:
        from src.stimulus import (load_stimulus, format_stimulus)
        stim_name, stim_title, stim_author, stim_stanzas, stim_secs = \
            load_stimulus(a.stimulus_file)
    except Exception as exc:  # noqa: BLE001
        print(f"warn: no stimulus module ({exc}); using a generic prompt")
        stim_name, stim_title, stim_author = "none", "none", "none"
        stim_stanzas, stim_secs = [], 40.0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    plan = PLAN
    if a.condition:
        match = [p for p in PLAN if p["name"] == a.condition]
        if not match:
            sys.exit(f"unknown condition {a.condition!r}; "
                     f"known: {[p['name'] for p in PLAN]}")
        plan = match

    print("librosa/sounddevice available:", bool(sd))
    cap = cv2.VideoCapture(a.cam)
    if not cap.isOpened():
        sys.exit("camera did not open")
    native = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("native:", native)

    if a.mic is None:
        d = sd.query_devices(kind="input")
        a.mic = d["index"] if "index" in d else 0
    print("mic   :", sd.query_devices(a.mic)["name"])
    print(f"stimulus: {stim_title} -- {stim_author}")

    for spec in plan:
        name = spec["name"]
        secs = a.seconds or spec.get("seconds", 30.0)
        if spec.get("long"):
            secs = a.long_seconds
        print("\n" + "=" * 62)
        print(f"CONDITION {name}: {spec['note']}")
        print(f"  duration {secs:.0f}s   pose={spec['pose']}   "
              f"speak={spec.get('speak')}")
        if spec.get("speak"):
            print(format_stimulus(stim_name, stim_title, stim_author,
                                  stim_stanzas))
            print("  RECITE FROM MEMORY. The same words every time -- that is "
                  "the point.\n")
        for t in range(a.countdown, 0, -1):
            print(f"  starting in {t}...", flush=True)
            time.sleep(1)

        ys, af, stamps = [], [], []
        n_target = int(secs * FPS)
        with sd.InputStream(device=a.mic, channels=1, samplerate=RATE,
                            dtype="int16", blocksize=0) as mic:
            real_rate = int(getattr(mic, "samplerate", RATE) or RATE)
            for i in range(n_target):
                ok, frame = cap.read()
                if not ok:
                    break
                # ONE video frame, ONE audio block, ONE clock reading. This
                # is what makes the two streams correspond.
                stamps.append(time.perf_counter())
                if frame.shape[1] != a.width or frame.shape[0] != a.height:
                    frame = cv2.resize(frame, (a.width, a.height),
                                       interpolation=cv2.INTER_AREA)
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                ys.append(yuv[:a.height, :a.width].tobytes())
                blk = int(real_rate / FPS)
                audio, _ = mic.read(blk)
                af.append(audio.tobytes())
        cap.release()

        if len(ys) < 10:
            print("  too few frames captured; skipping")
            continue

        # -- THE RATES, MEASURED, NOT ASSUMED ----------------------------
        elapsed = (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
        achieved_fps = ((len(stamps) - 1) / elapsed) if elapsed > 0 else 0.0
        n_samples = sum(len(b) for b in af) // 2
        audio_seconds = n_samples / real_rate if real_rate else 0.0
        video_seconds = len(ys) / FPS
        print(f"  frames {len(ys)}   video span {video_seconds:.2f}s "
              f"(assumed {FPS} fps)")
        print(f"  ACHIEVED camera rate {achieved_fps:.2f} fps   "
              f"mic rate {real_rate} Hz (requested {RATE})   "
              f"audio span {audio_seconds:.2f}s")
        if achieved_fps and abs(achieved_fps - FPS) / FPS > 0.05:
            print(f"  warn: the camera is delivering {achieved_fps:.2f} fps, "
                  f"not {FPS}. The video's TIME BASE is not what the frame "
                  f"index says it is. The timestamps in timing.json are the "
                  f"truth; use them, not the frame index.")
        if real_rate != RATE:
            print(f"  warn: the microphone is delivering {real_rate} Hz, not "
                  f"{RATE}. The WAV is written with the rate it was ACTUALLY "
                  f"captured at.")

        Y = np.ascontiguousarray(
            np.frombuffer(b"".join(ys), np.uint8).reshape(len(ys), a.height,
                                                           a.width))
        Yf = Y.astype(np.float64)
        Yr = row_band_remove(Yf)
        # `maps` keeps its meaning: the row-removed change maps. The
        # unfiltered maps and the raw luma are stored alongside so that the
        # representation is a post-hoc decision (see the module docstring).
        maps = change_maps(Yr)
        q, scale = quantise(maps)
        maps_raw = change_maps(Yf)
        q_raw, scale_raw = quantise(maps_raw)

        # -- the drift diagnostic, the number that decides usability ---------
        # The threshold is ~9x the measured OCCUPIED baseline and ~1000x the
        # measured EMPTY-ROOM baseline, on the CORRECTED stride: A.nv12
        # drifts 0.545 and empty1.nv12 drifts 0.005. It was set when the
        # tear made the number 43.560, so it has been re-derived against the
        # real baselines rather than left to look meaningful.
        DRIFT_OK = 5.0
        raw_mad = float(np.abs(np.diff(Yf, axis=0)).mean())
        row_mad = float(np.abs(np.diff(Yr, axis=0)).mean())
        print(f"  drift raw {raw_mad:.2f}   "
              f"after row-band removal {row_mad:.2f}   "
              f"{'OK' if row_mad < DRIFT_OK else 'TOO HIGH -- change the lighting'}")
        if row_mad >= DRIFT_OK:
            print("  warn: the banding is not the only problem. The take is "
                  "saved but the drift will dominate the motion field.")
        elif raw_mad > 2.0 * max(row_mad, 1e-6):
            # A row-constant component is present and the filter IS removing
            # something -- i.e. the case the filter exists for. Worth saying,
            # because the filter is now optional and this is the measurement
            # that would justify choosing it.
            print(f"  note: the row-constant component is {raw_mad / max(row_mad, 1e-6):.1f}x "
                  "the residual drift. The row-removed maps are the cleaner of "
                  "the two; the raw maps and the luma are stored as well.")

        stamp = int(time.time())
        npz = out / f"{name}_{stamp}.npz"
        np.savez_compressed(npz, maps=q, scale=np.float32(scale),
                            maps_raw=q_raw, scale_raw=np.float32(scale_raw),
                            luma=Y)

        # -- the audio, with the rate it was ACTUALLY captured at ----------
        pcm = b"".join(af)
        wave_path = out / f"{name}_{stamp}.wav"
        with wave.open(str(wave_path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(real_rate)
            f.writeframes(pcm)

        # -- the timing, so the correspondence is CHECKABLE ----------------
        t0 = stamps[0] if stamps else 0.0
        timing = {
            "condition": name,
            "n_frames": len(ys),
            "frame_offsets_s": [round(s - t0, 6) for s in stamps],
            "assumed_fps": FPS,
            "achieved_fps": round(achieved_fps, 4),
            "video_span_s": round(video_seconds, 3),
            "audio_span_s": round(audio_seconds, 3),
            "audio_sample_rate": real_rate,
            "requested_sample_rate": RATE,
            "mic": str(sd.query_devices(a.mic)["name"]),
            "capture_wallclock_s": round(time.time() - stamp, 3),
        }
        (out / f"{name}_{stamp}_timing.json").write_text(json.dumps(timing))

        # -- the achievement check -------------------------------------------
        speech_frac = None
        try:
            if SileroGate is None or read_wav is None:
                raise RuntimeError("the gate is not importable")
            a_ = read_wav(wave_path)
            probs = SileroGate().push(list(a_.samples))
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

        meta = {
            "condition": name,
            "kind": spec["kind"],
            "note": spec["note"],
            "pose": spec["pose"],
            "speak": spec.get("speak"),
            "stimulus": stim_name,
            "stimulus_title": stim_title,
            "stimulus_estimated_s": stim_secs,
            "speech_frac": speech_frac,
            "n_frames": int(len(ys)),
            "seconds": video_seconds,
            "achieved_fps": round(achieved_fps, 4),
            "audio_sample_rate": real_rate,
            "audio_seconds": round(audio_seconds, 3),
            "map_shape": list(q.shape[1:]),
            "map_scale": scale,
            "map_scale_raw": scale_raw,
            "drift_raw": raw_mad,
            "drift_after_row_removal": row_mad,
            "timestamp": stamp,
            "rep": spec.get("rep"),
            # The label said "row+median3", but no temporal median of width 3
            # was ever applied here -- the code subtracts each row's median
            # across the columns and then differences consecutive frames. A
            # label that asserts a falsehood is a landmine: a later reader
            # would assume a filter that is not in the data. (CORRECTION
            # 2026-10-20.)
            "filter": "row",
            # What is actually stored, per array. The representation is a
            # post-hoc decision now, so it is recorded per array rather than
            # as one property of the take.
            "filters": {"maps": "row", "maps_raw": "none", "luma": "none"},
            "arrays": ["maps", "maps_raw", "luma"],
        }
        (out / f"{name}_{stamp}.json").write_text(json.dumps(meta, indent=2))
        print(f"  wrote {npz.name}  ({npz.stat().st_size / 1024:.0f} kB)")

    print("\ndone")


if __name__ == "__main__":
    main()
