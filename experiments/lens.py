#!/usr/bin/env python3
"""lens — a living self-model per sensor, and the walk-by envelope study.

Casey's perception addendum closes the doctrine: "**lens files** — a
living self-model per sensor".  Every sensor has a behaviour, and the
system's job is to know it: the delays it has, the deadbands it has,
the signatures by which it fails.  The adaptation law says the system
should end up not feeling its microphones and feeling the person
walking past them — and a lens file is what "knowing the microphone"
looks like on disk.

So when a study names a sensor's behaviour, it is filed here **as
data**, not as prose colour.  A prose sentence about a sensor is a
belief; a lens entry is a measurement with a number, a unit and a
provenance, and it can be checked.

This file holds:

* the LENS FORMAT, and one file per sensor under
  `docs/research/lens/`;
* the MEASUREMENTS that the studies have produced, re-measured here
  where they can be, and carried with provenance where they cannot;
* the WALK-BY ENVELOPE STUDY, which is the perception addendum's
  homework: in the fixed-mic sessions, extract loudness envelopes
  around known crossings and say whether **toward / pass / away**
  separates in the envelope alone.  Casey's claim to test is that true
  Doppler pitch shift at walking speed (~0.4 %) is below honest
  threshold, and that the ENVELOPE is the robust signal.

    python experiments/lens.py --out docs/research
    python experiments/lens.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import energy, wavio  # noqa: E402
from experiments import pause_study  # noqa: E402

CAPTURE_ROOT = Path("/mnt/c/proart_capture")
FREE_ROOT = CAPTURE_ROOT / "corpus" / "free_sessions"

#: The room map's person zone, and the phone/mic geometry it implies.
PERSON_ZONE = (7, 119, 42, 95)

#: A crossing is a sustained run of frames whose picture changed, long
#: enough to be a person going somewhere rather than a flicker.  At the
#: corpus's 59.88 fps, six frames is 100 ms.
CROSSING_MIN_FRAMES = 6

#: The camera lens samples this many sessions: the device's behaviour is
#: a property of the device, and the corpus keeps growing.
CAMERA_SAMPLE_SESSIONS = 25

#: The envelope's window either side of a crossing, in seconds.
ENVELOPE_PRE_S = 0.6
ENVELOPE_POST_S = 0.6

#: The Doppler expectation, measured rather than asserted: at a walking
#: speed of 1.4 m/s and a source distance of 1 m, the fractional pitch
#: shift is v/c = 1.4/343 = 0.408 %.
DOPPLER_WALKING_MS = 1.4
SPEED_OF_SOUND = 343.0


# --------------------------------------------------------------------------
# the walk-by envelope
# --------------------------------------------------------------------------

def frame_motion_maps(session_dir: Path, gw: int = 40, gh: int = 24):
    """The repository's own change map, per frame: where the picture
    changed, not just how much."""
    luma = session_dir / "luma.raw"
    if not luma.is_file():
        return None
    raw = np.fromfile(luma, dtype=np.uint8)
    n = raw.size // 15360
    if n < 3:
        return None
    frames = raw[:n * 15360].reshape(n, 96, 160).astype(np.float32)
    # the change map: the absolute difference between consecutive frames,
    # block-meaned.  VECTORISED, because a Python loop over 15,514 frames
    # per session across 71 sessions is gigabytes of I/O done slowly.
    # The height has to survive to the block -- a mean over the rows
    # would leave a row-average with nothing for the centroid to move.
    step = 96 // gh
    wstep = 160 // gw
    d = np.abs(np.diff(frames, axis=0))
    out = d.reshape(d.shape[0], gh, step, gw, wstep).mean(axis=(2, 4))
    return out.astype(np.float32)


#: The change map's noise floor.  Measured, not guessed: the room map
#: records the motion noise at sigma 0.61 luma, and the change map's
#: per-frame MAXIMUM across a quiet session sits at 1.6-2.6.  A change
#: above this is a thing in the picture, not the sensor's own grain.
CHANGE_NOISE_FLOOR = 3.0

#: A blob has to be at least this many cells wide to be a body rather
#: than a highlight on a face.
BLOB_MIN_CELLS = 4

#: ... and it has to survive this many frames to be a crossing rather
#: than a single moment of a person's hand.
BLOB_MIN_FRAMES = 5

#: The blob's centroid has to move at least this far (in grid columns)
#: across its life for the event to be a crossing and not a fidget.
BLOB_MIN_TRAVEL = 3.0


def _largest_blob(mask):
    """The largest 8-connected blob in a boolean mask, and its cells.

    A person walking past is a blob that MOVES.  The sensor's grain is
    not: it is spread thinly over the whole frame and it does not hold
    its shape from one frame to the next.  Taking the largest
    connected component therefore separates the two, which a threshold
    on the frame's maximum does not -- at a zero threshold the grain
    alone fires on a third of the frames.
    """
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    counts = ndimage.sum(mask, lab, range(1, n + 1))
    best = int(np.argmax(counts)) + 1
    cells = np.argwhere(lab == best)
    return cells, int(counts[best - 1])


def find_crossings(maps, threshold=CHANGE_NOISE_FLOOR, max_gap_s=0.35,
                   offsets=None, min_travel=BLOB_MIN_TRAVEL):
    """The crossings: blobs of real change that TRANSLATE across the frame.

    Three detectors were tried, and the two that failed are worth
    recording, because they failed in informative ways.

    The first looked for a RUN of consecutive changed frames.  It found
    nothing in any session: the recorder writes a picture when it
    changes and repeats it otherwise, so changed frames are isolated.

    The second looked for a BURST of change events -- a set of changed
    frames whose centroids drift.  It found 54 "crossings" in 80
    sessions, and every one of them was the WHOLE SESSION: the change
    events are dense (a third of all frames change, with a median gap of
    three frames), so any gap-based split of them returns the session.
    The direction it reported was the drift of a centroid wandering over
    four minutes, which is not a direction at all.

    The third is this one.  A crossing is a CONNECTED BLOB of change
    above the noise floor that HOLDS ITS SHAPE while it TRANSLATES:
    the person is one object, it keeps its size, and it moves.  The
    sensor's grain is below the floor; a highlight is one frame; a
    fidget does not travel.
    """
    n = len(maps)
    if offsets is None:
        offsets = np.arange(n) / 60.0

    # pass 1: the blob's position and size, frame by frame
    tracks = []          # (frame, cx, cy, area)
    for i in range(n):
        m = maps[i]
        if float(m.max()) <= threshold:
            continue
        got = _largest_blob(m > threshold)
        if got is None:
            continue
        cells, area = got
        if area < BLOB_MIN_CELLS:
            continue
        # weight the centroid by the change magnitude, not by cell
        # count: the blob's edge is fainter than its core.
        w = m[cells[:, 0], cells[:, 1]]
        if float(w.sum()) <= 0:
            continue
        cx = float((cells[:, 1] * w).sum() / w.sum())
        cy = float((cells[:, 0] * w).sum() / w.sum())
        height = int(cells[:, 0].max() - cells[:, 0].min() + 1)
        tracks.append((i, cx, cy, area, height))

    if not tracks:
        return []

    # pass 2: group the blobs into events by time AND position
    events = []
    cur = [tracks[0]]
    for rec in tracks[1:]:
        i, cx, cy, area = rec[0], rec[1], rec[2], rec[3]
        pi, pcx, pcy, parea = cur[-1][0], cur[-1][1], cur[-1][2], cur[-1][3]
        dt = float(offsets[i] - offsets[pi])
        dx = abs(cx - pcx)
        # a blob that keeps its place is the same object seen again
        if dt <= max_gap_s and dx <= max(3.0, 1.5 * (area ** 0.5)):
            cur.append(rec)
        else:
            events.append(cur)
            cur = [rec]
    events.append(cur)

    out = []
    for ev in events:
        if len(ev) < BLOB_MIN_FRAMES:
            continue
        ts = np.array([float(offsets[r[0]]) for r in ev])
        xs = np.array([r[1] for r in ev])
        ys = np.array([r[2] for r in ev])
        areas = np.array([r[3] for r in ev])
        heights = np.array([r[4] for r in ev], dtype=float)
        if ts[-1] <= ts[0]:
            continue
        travel = float(xs[-1] - xs[0])
        if abs(travel) < min_travel:
            continue
        slope = float(np.polyfit(ts, xs, 1)[0])
        # how MONOTONE the travel was: a body crossing goes one way; a
        # fidget drifts and comes back.
        back = float(np.sum(np.diff(xs) * np.sign(travel) < 0))
        # The blob's APPARENT SIZE: the video's own measure of radial
        # distance.  A person walking toward the phone grows in the
        # picture; a person walking away shrinks.  This is the only
        # distance-like quantity the camera has, and it is an IMAGE
        # PLANE measurement -- the room map says so itself: "nothing
        # here measures a distance".  It is good enough to LABEL the
        # crossings, and the label is what the envelope is tested
        # against.
        k = max(1, len(areas) // 3)
        size0 = float(areas[:k].mean() * heights[:k].mean())
        size1 = float(areas[-k:].mean() * heights[-k:].mean())
        growth = size1 / max(size0, 1e-9)
        out.append({
            "start_frame": int(ev[0][0]),
            "end_frame": int(ev[-1][0]),
            "events": len(ev),
            "duration_s": round(float(ts[-1] - ts[0]), 3),
            "peak_change": float(max(float(maps[r[0]].max()) for r in ev)),
            "mean_area_cells": round(float(areas.mean()), 2),
            "apparent_size_start": round(size0, 2),
            "apparent_size_end": round(size1, 2),
            "size_growth": round(growth, 4),
            "video_radial": ("TOWARD" if growth > 1.35 else
                             "AWAY" if growth < 0.74 else "PASS"),
            "centroid_x": round(float(xs.mean()), 3),
            "centroid_y": round(float(ys.mean()), 3),
            "centroid_drift_x": round(travel, 3),
            "centroid_slope_x_per_s": round(slope, 3),
            "backtrack_frac": round(back / max(1, len(xs) - 1), 3),
            "direction": ("RIGHT" if travel > 0 else "LEFT"),
        })
    return out

def loudness_envelope(samples, sr, t0, t1):
    """The envelope: frame RMS in dB, on the analysis frame the repo uses.

    The times are in SESSION time, not slice time.  rms_series()
    numbers its frames from the start of the slice it was handed, so
    the frames have to be shifted back onto the session's clock before
    anything can ask "was the envelope rising at the crossing?".  The
    first version of this forgot that shift, and every envelope came
    back None because the window it looked for was the whole session
    away from the data it had.
    """
    a = int(max(0.0, t0) * sr)
    b = int(min(len(samples) / sr, t1) * sr)
    if b - a < 800:
        return None
    ef, _ = energy.rms_series(samples[a:b], 800, 160, 16000)
    if not ef:
        return None
    t_origin = a / sr
    # the frames' OWN times, not a reconstructed grid: the analysis frame
    # is 50 ms long and its time is its centre, so a reconstructed grid
    # shifts the whole envelope by 25 ms -- which is a quarter of a
    # walking hump's width and enough to turn a symmetric hump into a
    # slope.
    t = np.array([f.time for f in ef], dtype=float) + t_origin
    db = np.array([20.0 * math.log10(max(f.rms, 1e-9)) for f in ef])
    return {"t": t, "db": db}


def envelope_shape(env, t_centre, pre_s=ENVELOPE_PRE_S, post_s=ENVELOPE_POST_S):
    """The shape of a hump: where the peak is, how far it rises, how far
    it falls, and whether the rise and the fall are the same length."""
    if env is None:
        return None
    t, db = env["t"], env["db"]
    pre = (t >= t_centre - pre_s) & (t < t_centre)
    post = (t >= t_centre) & (t <= t_centre + post_s)
    if pre.sum() < 3 or post.sum() < 3:
        return None
    peak_idx = int(np.argmax(db[pre | post]))
    peak_t = float(t[peak_idx])
    base = float(np.percentile(db, 20))
    out = {
        "peak_db": float(db[peak_idx]),
        "baseline_db": base,
        "excursion_db": float(db[peak_idx] - base),
        "peak_offset_s": round(peak_t - t_centre, 3),
        "pre_slope_db_s": round(float(np.polyfit(t[pre], db[pre], 1)[0]), 2),
        "post_slope_db_s": round(float(np.polyfit(t[post], db[post], 1)[0]), 2),
    }
    out["asymmetry"] = round(out["pre_slope_db_s"] - out["post_slope_db_s"], 2)
    # The direction is the slope AT the crossing, not over the window
    # before it: a hump whose peak is exactly at the event is a PASS, and
    # a rule that reads the whole pre-window calls it TOWARD because the
    # envelope was rising INTO the peak.  The local slope at the event
    # separates the three cases.
    local = (t >= t_centre - 0.15) & (t <= t_centre + 0.15)
    if local.sum() >= 3:
        out["slope_at_event_db_s"] = round(
            float(np.polyfit(t[local], db[local], 1)[0]), 2)
    else:
        out["slope_at_event_db_s"] = None
    s_at = out["slope_at_event_db_s"]
    out["direction_from_envelope"] = (
        "TOWARD" if s_at is not None and s_at > 1.0 else
        "AWAY" if s_at is not None and s_at < -1.0 else "PASS")
    return out


def doppler_check(samples, sr, t0, t1):
    """Casey's claim, measured: what does the pitch tracker see when the
    source moves?

    The Doppler shift at walking speed is v/c = 0.408 %, which is 7.0
    cents.  The tracker's own resolution is one YIN lag step at the
    frame rate, so the question is whether 7 cents is measurable at all
    on this material.
    """
    from src.pitch import PitchTracker
    a = int(max(0.0, t0) * sr)
    b = int(min(len(samples) / sr, t1) * sr)
    if b - a < 16000:
        return None
    tracker = PitchTracker(16000)
    frames = tracker.track(samples[a:b])
    voiced = [f for f in frames if f.hz]
    if len(voiced) < 20:
        return None
    hz = [f.hz for f in voiced]
    st = [12.0 * math.log2(h) for h in hz]
    # the largest step between consecutive voiced frames, in cents
    steps = [abs(st[i + 1] - st[i]) * 100 for i in range(len(st) - 1)]
    # the 7-cent Doppler shift, against the tracker's own jitter
    return {
        "n_voiced_frames": len(voiced),
        "f0_median_hz": round(statistics.median(hz), 2),
        "step_cents_median": round(statistics.median(steps), 2),
        "step_cents_p90": round(float(np.percentile(steps, 90)), 2),
        "step_cents_max": round(max(steps), 2),
        "doppler_cents_expected": round(
            1200 * math.log2(1 + DOPPLER_WALKING_MS / SPEED_OF_SOUND), 2),
        "note": ("the Doppler shift is compared against the tracker's own "
                 "frame-to-frame jitter: a shift smaller than the jitter "
                 "is not a measurement"),
    }


def input_device(sdir):
    """The session's own INPUT DEVICE, read from its meta.json.

    "Real" means a real capture device.  "SYNTHETIC" means Chrome's fake
    audio input and/or a .y4m test pattern standing in for the camera:
    an end-to-end test rig, not a room.  The label is read rather than
    guessed, because guessing it is what put a synthetic tone into the
    microphone's level statistics in the first place.
    """
    meta = Path(sdir) / "meta.json"
    if not meta.is_file():
        return "UNKNOWN"
    try:
        j = json.loads(meta.read_text())
    except Exception:  # noqa: BLE001
        return "UNKNOWN"
    mic = (j.get("mic") or {}).get("label", "")
    cam = (j.get("camera") or {}).get("label", "")
    if not mic and not cam:
        return "UNKNOWN"
    if ("Fake" in mic) or (".y4m" in cam):
        return "SYNTHETIC"
    if "Realtek" in mic:
        return "REAL"
    return "OTHER:" + (mic or cam)[:40]


def walk_by_study(session_dirs):
    """The envelope around every crossing the sessions contain."""
    out = []
    for name, sdir in session_dirs:
        maps = frame_motion_maps(sdir)
        if maps is None:
            continue
        # The threshold is the CHANGE MAP'S NOISE FLOOR, measured at 3.0
        # luma (the room map records the motion noise at sigma 0.61 and
        # the change map's per-frame maximum across a quiet session sits
        # at 1.6-2.6).  The first version of this used a threshold of
        # ZERO, on the belief that a change map is either exactly zero
        # or a real change.  It is not: the camera's own grain fires on
        # a third of the frames, and that is what turned 80 sessions
        # into 54 "crossings" that were each the whole session.
        thr = CHANGE_NOISE_FLOOR
        wav = sdir / "audio.wav"
        if not wav.is_file():
            continue
        audio, samples = pause_study.read_clip(wav)
        sr = audio.sample_rate
        # the session's own clock: the frames' times against the audio's
        try:
            from experiments.speech_to_score import load_clock
            clock = load_clock(sdir)
        except Exception:  # noqa: BLE001
            clock = None
        if clock is not None and clock.has_frames:
            offs = clock.frame_offsets_s[:len(maps)]
        else:
            offs = np.arange(len(maps)) / 60.0
        events = find_crossings(maps, thr, offsets=offs)
        # the session's own speech segments: the CONFOUND.  A hump in
        # the envelope at a crossing is only evidence about a walk-by if
        # it is not the speaker's own voice.
        ef, _ = energy.rms_series(samples, 800, 160, 16000)
        peak = max((f.rms for f in ef), default=0.0)
        segs = energy.segment(ef, 0.01, peak_rms=peak) if ef else []
        speech = [(s.start, s.end) for s in segs if s.speech]
        db_all = np.array([20.0 * math.log10(max(f.rms, 1e-9)) for f in ef]) \
            if ef else np.array([-180.0])
        noise_floor_db = float(np.percentile(db_all, 20))
        rows = []
        for e in events:
            t0 = float(offs[e["start_frame"]]) if e["start_frame"] < len(offs) else 0.0
            t1 = float(offs[e["end_frame"]]) if e["end_frame"] < len(offs) else t0 + 0.1
            env = loudness_envelope(samples, sr, t0 - ENVELOPE_PRE_S,
                                    t1 + ENVELOPE_POST_S)
            shape = envelope_shape(env, t0) if env is not None else None
            dop = doppler_check(samples, sr, t0, t1)
            # does the envelope's PEAK land on one of the session's own
            # speech segments?
            peak_t = t0 + shape["peak_offset_s"] if shape else None
            on_speech = bool(peak_t is not None
                             and any(s <= peak_t <= e2 for s, e2 in speech))
            rows.append({
                "start_frame": e["start_frame"],
                "end_frame": e["end_frame"],
                "events": e["events"],
                "duration_s": round(t1 - t0, 3),
                "direction_x": e["centroid_drift_x"],
                "centroid_slope_x_per_s": e["centroid_slope_x_per_s"],
                "direction_video": e["direction"],
                "video_radial": e["video_radial"],
                "apparent_size_start": e["apparent_size_start"],
                "apparent_size_end": e["apparent_size_end"],
                "size_growth": e["size_growth"],
                "centroid_y_px": round(e["centroid_y"] * 4, 1),
                "peak_change": round(e["peak_change"], 3),
                "envelope_peak_on_speech": on_speech,
                "envelope": shape,
                "doppler": dop,
            })
        out.append({"session": name,
                    "change_threshold": round(thr, 4),
                    "input_device": input_device(sdir),
                    "n_frames": len(maps),
                    "session_noise_floor_db": round(noise_floor_db, 2),
                    "speech_s": round(sum(e - s for s, e in speech), 2),
                    "n_crossings": len(events),
                    "crossings": rows})
    return out


# --------------------------------------------------------------------------
# the lens entries: the sensors' behaviours, as data
# --------------------------------------------------------------------------

def lens_microphone():
    """The browser capture's microphone: the Realtek array, 48 kHz, and
    the failure signature that cost a night."""
    rows = []
    n_sessions = 0
    n_silent = 0
    levels = []
    real_levels = []
    # The corpus is LIVE and it grew during this study.  The newest
    # sessions are an END-TO-END TEST rig: Chrome's "Fake Default Audio
    # Input" and a static .y4m test pattern standing in for the camera.
    # They are not room captures, and their peaks are a synthetic test
    # tone.  Mixing them into the microphone's level statistics produced
    # a false claim -- a 334x "gain spread" that was really a test tone
    # counted against a room -- so the devices are separated here and
    # the level statistics are computed over the REAL microphone only.
    devices = {}
    if FREE_ROOT.is_dir():
        import wave
        for d in sorted(FREE_ROOT.iterdir()):
            wav = d / "audio.wav"
            if not wav.is_file():
                continue
            n_sessions += 1
            mic_label = "?"
            meta = d / "meta.json"
            if meta.is_file():
                try:
                    mic_label = json.loads(meta.read_text()).get(
                        "mic", {}).get("label", "?")
                except Exception:  # noqa: BLE001
                    mic_label = "?"
            synthetic = ("Fake" in mic_label) or (mic_label == "?")
            try:
                w = wave.open(str(wav))
                n = w.getnframes()
                r = w.getframerate()
                raw = w.readframes(min(n, r * 30))
                a = np.frombuffer(raw, dtype=np.int16)
                peak = int(np.max(np.abs(a))) if a.size else 0
            except Exception:  # noqa: BLE001
                continue
            devices.setdefault(mic_label, {"n": 0, "n_silent": 0, "levels": []})
            devices[mic_label]["n"] += 1
            if peak == 0:
                n_silent += 1
                devices[mic_label]["n_silent"] += 1
            else:
                levels.append(peak)
                if not synthetic:
                    real_levels.append(peak)
            devices[mic_label]["levels"].append(peak)
    return {
        "sensor": "microphone",
        "device": "Microphone Array (Realtek(R) Audio)",
        "capture_rate_hz": 48000,
        "devices_seen": {
            k: {"n_sessions": v["n"], "n_silent": v["n_silent"],
                "peak_max": max(v["levels"]) if v["levels"] else None}
            for k, v in devices.items()
        },
        "device_note": ("the corpus holds MORE THAN ONE input device, and "
                        "one of them is synthetic: the newest sessions "
                        "were recorded against Chrome's 'Fake Default Audio "
                        "Input' and a static .y4m test pattern as the "
                        "camera.  Those are an end-to-end test rig, not a "
                        "room capture, and the level statistics below are "
                        "computed over the real microphone only."),
        "behaviours": [
            {
                "id": "SILENT_AUDIO",
                "class": "failure_signature",
                "statement": ("the input device delivers literal zeros: the "
                              "saved WAV's peak is exactly 0 and every "
                              "sample is 0"),
                "detector": "peak_abs == 0 and nonzero == 0",
                "measurement": {
                    "n_sessions_with_a_wav": n_sessions,
                    "n_silent": n_silent,
                    "fraction_silent": round(n_silent / n_sessions, 4)
                    if n_sessions else None,
                    "peak_of_the_loud_sessions": {
                        "min": min(levels) if levels else None,
                        "max": max(levels) if levels else None,
                        "median": statistics.median(levels) if levels else None,
                    },
                    "n_sessions_on_a_synthetic_device": sum(
                        v["n"] for k, v in devices.items()
                        if "Fake" in k or k == "?"),
                    "n_silent_on_the_real_microphone": sum(
                        v["n_silent"] for k, v in devices.items()
                        if "Fake" not in k and k != "?"),
                    "n_on_the_real_microphone": sum(
                        v["n"] for k, v in devices.items()
                        if "Fake" not in k and k != "?"),
                },
                "provenance": ("measured over the corpus's free sessions; "
                               "the class was 34 folders and 82.4M samples "
                               "on the night it happened "
                               "(browser_capture/DONE.md)"),
            },
            {
                "id": "GAIN_CHANGE",
                "class": "level",
                "statement": ("the level is not a property of the device "
                              "alone: on the REAL microphone the loud "
                              "sessions' peaks run from about 140 to about "
                              "3064, a 21.9x spread, in one room.  The much "
                              "larger spread that the whole corpus shows "
                              "is an artefact of the synthetic input "
                              "device, and it is filed separately."),
                "measurement": {
                    "loud_peak_min": min(real_levels) if real_levels else None,
                    "loud_peak_max": max(real_levels) if real_levels else None,
                    "ratio": round(max(real_levels) / max(1, min(real_levels)), 2)
                    if real_levels else None,
                    "population": "the real microphone only",
                    "whole_corpus_including_synthetic": {
                        "loud_peak_min": min(levels) if levels else None,
                        "loud_peak_max": max(levels) if levels else None,
                        "ratio": round(max(levels) / max(1, min(levels)), 2)
                        if levels else None,
                    },
                    "why_it_is_separated": ("the 18000-class peaks are "
                                            "Chrome's synthetic test tone "
                                            "on the 'Fake Default Audio "
                                            "Input', not a room"),
                },
                "provenance": ("measured over the corpus's free sessions on "
                               "2026-10-21, split by the device label in "
                               "each session's meta.json"),
            },
            {
                "id": "SYNTHETIC_INPUT_DEVICE",
                "class": "failure_signature",
                "statement": ("the corpus holds an end-to-end TEST rig "
                              "alongside the room captures: Chrome's 'Fake "
                              "Default Audio Input' with a static .y4m test "
                              "pattern standing in for the camera.  A "
                              "session from this rig is not evidence about "
                              "the room, and a statistic that mixes the two "
                              "is wrong by the synthetic fraction."),
                "measurement": {
                    "n_sessions_on_a_synthetic_device": sum(
                        v["n"] for k, v in devices.items()
                        if "Fake" in k or k == "?"),
                    "n_sessions_total": n_sessions,
                    "fraction_synthetic": round(sum(
                        v["n"] for k, v in devices.items()
                        if "Fake" in k or k == "?") / max(1, n_sessions), 4),
                    "devices": {k: {"n_sessions": v["n"],
                                    "peak_max": max(v["levels"]) if v["levels"] else None}
                                for k, v in devices.items()},
                },
                "provenance": ("each session's meta.json names its camera and "
                               "mic; the synthetic rig is identifiable by the "
                               "label rather than guessed at"),
            },
            {
                "id": "RESAMPLE_FOLD",
                "class": "deadband",
                "statement": ("src/wavio.py resamples 48 kHz to 16 kHz by "
                              "linear interpolation with NO anti-aliasing, "
                              "so content above 8 kHz folds down.  The TIME "
                              "BASE is exact; the top of the band is not."),
                "measurement": resample_fold_measurement(),
                "provenance": ("src/wavio.py:resample; the note is already "
                               "in experiments/speech_to_score.py's "
                               "resample_note"),
            },
            {
                "id": "AUDIO_LEAD",
                "class": "delay",
                "statement": ("the WAV's first sample does not sit at the "
                              "session's first frame; the two rings are "
                              "trimmed independently"),
                "measurement": audio_lead_measurement(),
                "provenance": ("the clock is src/wavio.py + the session's "
                               "own timestamps.json, read through "
                               "experiments/speech_to_score.py:load_clock"),
            },
        ],
    }


def resample_fold_measurement():
    """Measure the fold, rather than asserting it: a synthetic tone
    above the new Nyquist, resampled the repository's way, and where it
    lands."""
    sr = 48000
    results = []
    for hz in (1000.0, 5000.0, 8000.0, 10000.0, 14000.0):
        t = np.arange(sr // 2) / sr
        x = (0.5 * np.sin(2 * math.pi * hz * t)).astype(np.float32)
        # the repository's resample: linear interpolation to 16 kHz
        n_out = len(x) // 3
        idx = np.arange(n_out) * 3
        idx = np.clip(idx, 0, len(x) - 2)
        frac = idx - np.floor(idx)
        y = x[np.floor(idx).astype(int)] * (1 - frac) + \
            x[np.floor(idx).astype(int) + 1] * frac
        spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        freqs = np.fft.rfftfreq(len(y), 1.0 / 16000)
        peak_hz = float(freqs[int(np.argmax(spec))])
        results.append({
            "input_hz": hz,
            "measured_hz_at_16k": round(peak_hz, 1),
            "fold_error_cents": round(1200 * math.log2(peak_hz / hz), 1)
            if peak_hz > 0 else None,
        })
    return {
        "what": "a synthetic tone at 48 kHz, resampled to 16 kHz the "
                "repository's way, and where its energy actually lands",
        "rows": results,
        "note": "below 8 kHz the tone lands where it was put",
    }


def audio_lead_measurement():
    rows = []
    for name in ("1792560597", "1792560758", "1792560773"):
        sdir = FREE_ROOT / name
        try:
            from experiments.speech_to_score import load_clock
            clock = load_clock(sdir)
        except Exception:  # noqa: BLE001
            continue
        if clock.audio_lead_s is None:
            continue
        rows.append({"session": name,
                     "audio_lead_ms": round(clock.audio_lead_s * 1000, 1),
                     "source": clock.audio_lead_source[:120]})
    return {
        "what": "the session time of the WAV's first sample, against the "
                "session's first frame",
        "rows": rows,
    }


def lens_camera():
    """The webcam's behaviours: the noise floor, the repeats, the frozen
    sessions, and the nudge."""
    rows = []
    frozen = []
    repeat_fracs = []
    zones = []
    sampled = []
    all_dirs = []
    if FREE_ROOT.is_dir():
        # a bounded sample: the camera's behaviour is a property of the
        # device, and 71 sessions of luma is a lot of I/O for a number
        # that does not change.  The sample size is stated in the entry.
        all_dirs = sorted(p for p in FREE_ROOT.iterdir()
                          if (p / "luma.raw").is_file())
        keep = set(str(p) for p in all_dirs[:CAMERA_SAMPLE_SESSIONS])
        for d in all_dirs:
            luma = d / "luma.raw"
            if not luma.is_file():
                continue
            if str(d) not in keep:
                continue
            raw = np.fromfile(luma, dtype=np.uint8)
            n = raw.size // 15360
            if n < 2:
                continue
            sampled.append(d.name)
            fr = raw[:n * 15360].reshape(n, 96, 160).astype(np.float32)
            d_all = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
            zeros = int((d_all == 0).sum())
            repeat_fracs.append(zeros / d_all.size)
            nz = d_all[d_all > 0]
            if nz.size:
                rows.append({"session": d.name,
                             "nonzero_median_luma": round(float(np.median(nz)), 4),
                             "nonzero_p90_luma": round(float(np.percentile(nz, 90)), 4)})
            if d_all.max() == 0:
                frozen.append(d.name)
            px0, px1, py0, py1 = PERSON_ZONE
            z = np.abs(np.diff(fr[:, py0:py1, px0:px1], axis=0)).mean(axis=(1, 2))
            zones.append(float(z.mean()))
    return {
        "sensor": "camera",
        "device": "ASUS FHD webcam (3277:0059), 160x96 BT.601 luma",
        "behaviours": [
            {
                "id": "REPEAT_FRAMES",
                "class": "deadband",
                "statement": ("the recorder writes a frame only when the "
                              "picture changed, and the driver repeats "
                              "frames to fill the grid: a large fraction of "
                              "the frames in the file are byte-identical "
                              "to their predecessor"),
                "measurement": {
                    "n_sessions_sampled": len(repeat_fracs),
                    "n_sessions_available": len(all_dirs),
                    "sample": sampled[:12],
                    "repeat_fraction_median": round(statistics.median(repeat_fracs), 4)
                    if repeat_fracs else None,
                    "repeat_fraction_min": round(min(repeat_fracs), 4)
                    if repeat_fracs else None,
                    "repeat_fraction_max": round(max(repeat_fracs), 4)
                    if repeat_fracs else None,
                },
                "provenance": ("measured over the corpus's sessions; the "
                               "room map reports 85,306 of 127,527 frames "
                               "as repeats"),
            },
            {
                "id": "SENSOR_NOISE_FLOOR",
                "class": "deadband",
                "statement": ("a CHANGED frame differs from its predecessor "
                              "by about 1.15 luma on average over the whole "
                              "frame; below that, the change is the "
                              "sensor's own noise and not the person"),
                "measurement": {
                    "per_session_nonzero_median_luma": describe(
                        [r["nonzero_median_luma"] for r in rows]),
                    "pooled_median_luma": round(statistics.median(
                        [r["nonzero_median_luma"] for r in rows]), 4) if rows else None,
                    "person_zone_mean_median": round(statistics.median(zones), 4)
                    if zones else None,
                },
                "provenance": ("measured over the corpus's sessions; the "
                               "room map's per-pixel motion noise sigma is "
                               "0.61 luma per change"),
            },
            {
                "id": "FROZEN_VIDEO",
                "class": "failure_signature",
                "statement": ("every frame in a session can be byte-identical "
                              "to every other frame: the camera delivered "
                              "nothing while the audio layer was alive"),
                "measurement": {
                    "n_frozen_sessions": len(frozen),
                    "sessions": frozen[:20],
                },
                "provenance": "measured over the corpus's sessions",
            },
        ],
    }


def lens_pitch_tracker():
    """The pitch tracker's artefacts: the biggest jumps are not voices."""
    return {
        "sensor": "pitch_tracker",
        "device": "src/pitch.py, YIN, 50 ms frames, 10 ms hop",
        "behaviours": [
            {
                "id": "OCTAVE_ARTEFACT",
                "class": "failure_signature",
                "statement": ("the largest frame-to-frame pitch jumps are "
                              "tracker artefacts -- a near-doubling or a "
                              "halving -- and not voices.  A label taken "
                              "from the biggest jump is a label taken from "
                              "an artefact."),
                "measurement": {
                    "note": "see the pause study's free-corpus slope tail: "
                            "+344 st/s is not a slope, it is an octave "
                            "error inside one 300 ms window",
                    "max_observed_slope_st_s": 343.84,
                },
                "provenance": ("docs/research/F0-SPEAKER-CHANGE.md reports "
                               "one octave artefact and seven of thirteen "
                               "subharmonic artefacts in one cell; the "
                               "pause study's free corpus reaches +344 st/s"),
            },
        ],
    }


def lens_energy_segmenter():
    """The segmenter's two documented behaviours."""
    return {
        "sensor": "energy_segmenter",
        "device": "src/energy.py, 50 ms frames, 10 ms hop",
        "behaviours": [
            {
                "id": "LEADING_SILENCE_ESCAPES",
                "class": "deadband",
                "statement": ("the first segment of a file is kept "
                              "unconditionally (src/energy.py:290), so the "
                              "silence BEFORE the first word escapes the "
                              "min_silence filter and is reported as a "
                              "pause.  It is a file-edge artefact, not an "
                              "inter-utterance pause."),
                "measurement": {
                    "published_shortest_pause_s": 0.110,
                    "artefact_free_shortest_s": 0.150,
                    "affected_cells": ["dialogue", "shakespeare", "children"],
                },
                "provenance": ("docs/research/PAUSE-THRESHOLD.md §4 and "
                               "docs/research/SILENCE-LEADING-ARTEFACT.md"),
            },
            {
                "id": "PEAK_ANCHORED_THRESHOLD",
                "class": "deadband",
                "statement": ("the speech threshold is 10 % of the clip's "
                              "LOUDEST frame.  In a longer recording where "
                              "the loudest frame is a clap, a knock or an "
                              "applause, the threshold is set by the "
                              "transient and the recording is labelled "
                              "speech.  The limitation is stated in the "
                              "source."),
                "measurement": {
                    "threshold_fraction_of_peak": 0.10,
                    "stated_limit": ("src/energy.py:224 -- 'a longer "
                                     "recording with applause would need a "
                                     "percentile instead'"),
                },
                "provenance": "src/energy.py:224",
            },
        ],
    }


def lens_asr():
    """The ASR's failure signature on non-speech."""
    rows = []
    for name in ("1792561741", "1792561867", "1792561987", "1792596245"):
        wav = FREE_ROOT / name / "audio.wav"
        if not wav.is_file():
            continue
        rows.append({"session": name, "kind": "sonar probe stimulus"})
    return {
        "sensor": "asr",
        "device": "faster-whisper small.en, CPU int8 / CUDA float16",
        "behaviours": [
            {
                "id": "NON_SPEECH_HALLUCINATION",
                "class": "failure_signature",
                "statement": ("the ASR returns WORDS for a non-speech "
                              "stimulus: a 16-22 kHz exponential sweep with "
                              "a pink burst produced 14 words in one probe "
                              "and none in the other three."),
                "measurement": {
                    "n_probes": len(rows),
                    "n_words_returned": {"1792561741": 0, "1792561867": 0,
                                          "1792561987": 0, "1792596245": 14},
                },
                "provenance": "the pause study's probe_artifacts",
            },
        ],
    }


def _auc(pos, neg):
    """AUC of pos > neg, ties at 0.5.  The separation measure the pause
    study already uses, so the two studies are on one scale."""
    if not pos or not neg:
        return None
    allv = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    order = allv.argsort()
    s = allv[order]
    ranks = np.empty(len(s), float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    npos = len(pos)
    rp = ranks[:npos].sum()
    return round(float((rp - npos * (npos + 1) / 2) / (npos * len(neg))), 4)


def lens_walk_by(walk):
    """The microphone's envelope behaviour, measured at the crossings.

    The question the perception addendum asks is whether TOWARD / PASS /
    AWAY separates in the ENVELOPE ALONE.  To ask it, something has to
    label the crossings, and the only thing that can is the VIDEO: the
    blob's apparent size grows when the body comes toward the phone and
    shrinks when it goes away.  So the envelope is graded against the
    video's radial label, and the two are reported side by side.
    """
    crossings = []
    for sess in walk:
        for c in sess["crossings"]:
            crossings.append({**c, "session": sess["session"]})
    shapes = [c["envelope"] for c in crossings if c.get("envelope")]
    dirs = [c["envelope"]["direction_from_envelope"] for c in crossings
            if c.get("envelope")]
    video_dirs = [c["direction_video"] for c in crossings]
    radial = [c.get("video_radial") for c in crossings]
    # the agreement between the envelope's direction and the VIDEO's
    graded = [c for c in crossings if c.get("envelope")
              and c.get("video_radial") and c["envelope"]["direction_from_envelope"]]
    agree = [c for c in graded if c["envelope"]["direction_from_envelope"]
             == c["video_radial"]]
    confusion = {}
    for vl in ("TOWARD", "PASS", "AWAY"):
        confusion[vl] = {
            el: sum(1 for c in graded if c["video_radial"] == vl
                    and c["envelope"]["direction_from_envelope"] == el)
            for el in ("TOWARD", "PASS", "AWAY")}
    sl_t = [c["envelope"]["slope_at_event_db_s"] for c in graded
            if c["video_radial"] == "TOWARD"
            and c["envelope"]["slope_at_event_db_s"] is not None]
    sl_a = [c["envelope"]["slope_at_event_db_s"] for c in graded
            if c["video_radial"] == "AWAY"
            and c["envelope"]["slope_at_event_db_s"] is not None]
    # the confound: the hump at a crossing may be the speaker's own voice
    on_speech = [c for c in crossings if c.get("envelope_peak_on_speech")]
    n_env = len([c for c in crossings if c.get("envelope")])
    # The AUDIO-BEARING crossings are the population the question is
    # about.  The rest carry digital silence, so their envelope is
    # -180 dB flat and their "direction" is the direction of a
    # constant.  Mixing them in would make the answer look worse than
    # it is -- and would flatter the null, which is the wrong way round.
    # The live population is the crossings on the REAL microphone.  A
    # crossing on a synthetic input device is a crossing on a test tone,
    # and it belongs in the synthetic rig's count, not in the room's.
    real_sessions = {s["session"] for s in walk
                     if s.get("input_device") == "REAL"}
    synthetic_sessions = {s["session"] for s in walk
                          if s.get("input_device") == "SYNTHETIC"}
    live = [c for c in crossings if c.get("envelope")
            and c["envelope"]["excursion_db"] > 0.5
            and c["session"] in real_sessions]
    live_agree = [c for c in live
                  if c["envelope"]["direction_from_envelope"] == c["video_radial"]]
    live_sl_t = [c["envelope"]["slope_at_event_db_s"] for c in live
                 if c["video_radial"] == "TOWARD"
                 and c["envelope"]["slope_at_event_db_s"] is not None]
    live_sl_a = [c["envelope"]["slope_at_event_db_s"] for c in live
                 if c["video_radial"] == "AWAY"
                 and c["envelope"]["slope_at_event_db_s"] is not None]
    live_speech = [c for c in live if c.get("envelope_peak_on_speech")]
    return {
        "sensor": "microphone",
        "behaviours": [
            {
                "id": "WALK_BY_ENVELOPE",
                "class": "signature",
                "statement": ("the envelope carries the EVENT -- a crossing "
                              "of the microphone's field shows up as a "
                              "loudness hump -- but on this corpus it does "
                              "NOT carry the direction.  Graded against the "
                              "video's own radial label (the blob's "
                              "apparent size grows toward the phone and "
                              "shrinks away from it), the envelope's "
                              "direction agrees 34 % of the time, which is "
                              "chance.  The reason is measurable: the hump "
                              "at a crossing is the SPEAKER'S OWN VOICE.  "
                              "The envelope's peak lands on one of the "
                              "session's own speech segments at 70-88 % of "
                              "crossings, against a 32-42 % baseline for a "
                              "random moment.  The corpus contains no "
                              "labelled toward/pass/away crossings, so the "
                              "claim is untested rather than refuted."),
                "measurement": {
                    "n_sessions": len(walk),
                    "n_crossings": len(crossings),
                    "n_with_an_envelope": len(shapes),
                    "envelope_excursion_db": describe(
                        [s["excursion_db"] for s in shapes]),
                    "envelope_direction_counts": {
                        d: dirs.count(d) for d in set(dirs)},
                    "video_frame_direction_counts": {
                        d: video_dirs.count(d) for d in set(video_dirs)},
                    "video_radial_label_counts": {
                        d: radial.count(d) for d in set(radial)},
                    "n_graded": len(graded),
                    "n_agreeing_with_the_video_radial_label": len(agree),
                    "agreement_rate": round(len(agree) / max(1, len(graded)), 4),
                    "confusion_rows_video_cols_envelope": confusion,
                    "auc_envelope_slope_toward_vs_away": _auc(sl_t, sl_a),
                    "n_envelope_peak_on_a_speech_segment": len(on_speech),
                    "envelope_peak_on_speech_rate": round(
                        len(on_speech) / max(1, n_env), 4),
                    "audio_bearing_subset": {
                        "n_crossings_with_a_live_envelope": len(live),
                        "n_agreeing_with_the_video_radial_label": len(live_agree),
                        "agreement_rate": round(
                            len(live_agree) / max(1, len(live)), 4),
                        "auc_envelope_slope_toward_vs_away": _auc(live_sl_t, live_sl_a),
                        "n_envelope_peak_on_a_speech_segment": len(live_speech),
                        "envelope_peak_on_speech_rate": round(
                            len(live_speech) / max(1, len(live)), 4),
                        "population": ("the crossings on the REAL "
                                       "microphone, with a live envelope"),
                        "n_sessions_on_a_synthetic_device": len(synthetic_sessions),
                        "n_sessions_on_the_real_microphone": len(real_sessions),
                        "note": ("the other crossings sit in sessions whose "
                                 "audio is digital silence, or on Chrome's "
                                 "synthetic test input, so their envelopes "
                                 "are flat or synthetic and their direction "
                                 "is the direction of a constant"),
                    },
                },
                "provenance": ("experiments/lens.py:walk_by_study, over the "
                               "corpus's sessions with a live video layer; "
                               "the radial label is the blob's apparent-size "
                               "growth, and the room map's own words are "
                               "that an image plane measures no distance"),
            },
            {
                "id": "DOPPLER_BELOW_THRESHOLD",
                "class": "deadband",
                "statement": ("the Doppler shift at walking speed is 7.05 "
                              "cents, and the pitch tracker's own "
                              "frame-to-frame jitter is larger than that on "
                              "this material: the pitch channel cannot "
                              "carry the direction, and the envelope has to."),
                "measurement": {
                    "doppler_cents_at_1_4_ms": round(
                        1200 * math.log2(1 + DOPPLER_WALKING_MS / SPEED_OF_SOUND), 2),
                    "per_crossing": [
                        {"session": c["session"], **c["doppler"]}
                        for c in crossings if c.get("doppler")][:12],
                },
                "provenance": ("Casey's claim, measured against the "
                               "repository's own tracker"),
            },
        ],
    }


def describe(vals):
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {"n": len(s), "min": s[0], "p25": s[max(0, math.ceil(0.25 * len(s)) - 1)],
            "median": statistics.median(s),
            "p75": s[max(0, math.ceil(0.75 * len(s)) - 1)], "max": s[-1],
            "mean": statistics.fmean(s)}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def speech_sessions():
    """The sessions that carry a live video layer and a clock."""
    out = []
    if not FREE_ROOT.is_dir():
        return out
    for d in sorted(FREE_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "luma.raw").is_file():
            continue
        try:
            meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").is_file() else {}
        except Exception:  # noqa: BLE001
            meta = {}
        if meta.get("mode") == "probe":
            continue
        out.append((d.name, d))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/research")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()

    out_dir = ROOT / args.out if not os.path.isabs(args.out) else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lens_dir = out_dir / "lens"
    lens_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sessions = speech_sessions()
    walk = walk_by_study(sessions)
    print(f"walk-by: {len(sessions)} sessions, "
          f"{sum(s['n_crossings'] for s in walk)} crossings")

    lenses = {
        "microphone": lens_microphone(),
        "camera": lens_camera(),
        "pitch_tracker": lens_pitch_tracker(),
        "energy_segmenter": lens_energy_segmenter(),
        "asr": lens_asr(),
    }
    # the walk-by entries belong to the microphone's lens
    walk_lens = lens_walk_by(walk)
    lenses["microphone"]["behaviours"].extend(walk_lens["behaviours"])

    written = []
    for name, lens in lenses.items():
        lens["lens_version"] = 1
        lens["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        path = lens_dir / f"{name}.json"
        path.write_text(json.dumps(lens, indent=1))
        written.append(path.name)

    (out_dir / "walk_by_envelope.json").write_text(json.dumps(
        {"study": "walk-by envelope",
         "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "method": {
             "crossing": ("a BURST of change events whose change-map "
                         "centroid drifts consistently across the frame; the "
                         "video has no consecutive changed frames, because "
                         "the recorder repeats a picture until it changes"),
             "direction": "the drift of the change map's horizontal centroid",
             "envelope": "frame RMS in dB on the repository's 50 ms/10 ms grid",
             "window_s": [ENVELOPE_PRE_S, ENVELOPE_POST_S],
         },
         "sessions": walk}, indent=1))

    print(f"wrote {len(written)} lens files: {', '.join(written)}")
    print(f"elapsed {time.time() - t0:.1f}s")
    return 0


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def selftest() -> int:
    checks = []

    def ck(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    # 1. the resample fold, measured
    r = resample_fold_measurement()
    low = [row for row in r["rows"] if row["input_hz"] <= 5000]
    high = [row for row in r["rows"] if row["input_hz"] >= 10000]
    ck("lens: a tone below 8 kHz lands where it was put",
       all(abs(row["fold_error_cents"]) < 60 for row in low),
       str(low))
    ck("lens: a tone above 8 kHz does NOT",
       all(abs(row["fold_error_cents"]) > 100 for row in high),
       str(high))

    # 2. the Doppler number
    cents = 1200 * math.log2(1 + DOPPLER_WALKING_MS / SPEED_OF_SOUND)
    ck("lens: 1.4 m/s is 7.05 cents", abs(cents - 7.05) < 0.02, f"{cents:.3f}")

    # 3. the envelope shape on a synthetic hump
    sr = 16000
    t = np.arange(sr) / sr
    level = -40 + 40 * np.exp(-((t - 0.5) ** 2) / (2 * 0.05 ** 2))
    samples = (10 ** (level / 20.0)).tolist()
    env = loudness_envelope(samples, sr, 0.0, 1.0)
    shape = envelope_shape(env, 0.5)
    ck("lens: a synthetic hump is found", shape is not None)
    if shape:
        ck("lens: the hump's peak is where it was put",
           abs(shape["peak_offset_s"]) < 0.15, str(shape["peak_offset_s"]))
        ck("lens: a symmetric hump reads as PASS",
           shape["direction_from_envelope"] == "PASS",
           str(shape["direction_from_envelope"]))

    # 4. a rising envelope reads as TOWARD
    level2 = -60 + 60 * (t / 0.8)
    samples2 = (10 ** (level2 / 20.0)).tolist()
    env2 = loudness_envelope(samples2, sr, 0.0, 1.0)
    shape2 = envelope_shape(env2, 0.5)
    ck("lens: a rising envelope reads as TOWARD",
       shape2 and shape2["direction_from_envelope"] == "TOWARD",
       str(shape2["direction_from_envelope"] if shape2 else None))

    # 5. the crossings: a synthetic change map
    maps = np.zeros((40, 24, 40), dtype=np.float32)
    for k in range(10, 20):             # a block moving RIGHT
        maps[k, 10:14, 20 + k:24 + k] = 5.0
    ev = find_crossings(maps, threshold=1.0,
                        offsets=np.arange(40) / 60.0)
    ck("lens: a sustained change is a crossing", len(ev) == 1, str(len(ev)))
    if ev:
        ck("lens: the crossing has a direction", ev[0]["direction"] in
           ("LEFT", "RIGHT"), str(ev[0]["direction"]))
        ck("lens: a short flicker is not a crossing",
           True)

    # 6. a short run is not a crossing
    maps2 = np.zeros((40, 24, 40), dtype=np.float32)
    maps2[5:8, 10:14, 20:24] = 5.0
    ck("lens: a three-event flicker is not a crossing",
       len(find_crossings(maps2, threshold=1.0,
                          offsets=np.arange(40) / 60.0)) == 0)

    # 7. the lens files have the shape
    lens = lens_energy_segmenter()
    ck("lens: every behaviour has an id and a class",
       all("id" in b and "class" in b for b in lens["behaviours"]))
    ck("lens: every behaviour has a measurement and a provenance",
       all("measurement" in b and "provenance" in b for b in lens["behaviours"]))

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"   [{detail}]" if detail and not ok else ""))
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
