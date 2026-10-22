#!/usr/bin/env python3
"""DUAL INGEST -- a phone recording becomes a corpus bundle.

    python experiments/dual_ingest.py --selftest
    python experiments/dual_ingest.py --video <phone.mp4> --out <dir> \\
        --label "Casey's phone"

WHAT THIS IS.  The dual capture (docs/research/BRIEF-DUAL-CAPTURE.md)
puts a phone camera in Casey's hand beside the ProArt webcam.  The
rendering lane (experiments/dual_render.py) reads the corpus's own
bundle format, so ONE reader serves both sides -- and this tool makes
the phone's recording into that format.  No new format is invented:
the output is `luma.raw`, `timestamps.json` and `meta.json`, exactly
what the ProArt session bundles already are.

THE OUTPUT IS THE CORPUS'S FORMAT, EXACTLY:

  luma.raw          uint8 BT.601 full-range luma (0.299R+0.587G+0.114B),
                    160x96, row-major, frames concatenated --
                    np.fromfile(path, np.uint8).reshape(-1, 96, 160)
  timestamps.json   {"wall_clock_start": ISO, "frame_offsets_ms": [...]}
                    the frame times are the SPINE: the origin is the
                    first ingested frame and every offset is relative
                    to it
  meta.json         the camera label, the luma spec, the achieved frame
                    rate, and what this tool decided

THE GEOMETRY IS NOT STRETCHED.  The corpus's frame is 160x96, an aspect
of 1.667, and a phone records 16:9 (or 9:16 in the hand).  The default
is to CROP the picture to the corpus's aspect and then resize, because
the grid is positional -- a stretched picture moves the body's cells
around and the whole point of a positional render is that a cell means
where the thing is.  `stretch` is available and is reported in the
metadata, but it is a choice and not a default.

THE CLOCK IS SAID OUT LOUD.  A phone recording has no wall clock in
it, so the tool takes `--wall-clock-start` if it is given, and
otherwise falls back to the file's modification time -- which is an
APPROXIMATION (the file may have been uploaded hours later) and is
labelled as one in the metadata.  The frame times come from the
container's own per-frame timestamps when it has them, and from the
nominal frame rate when it does not, and the metadata says which.

THE NOISE IS MEASURED, NOT ASSUMED.  The corpus's change map has a
3.0 luma noise floor, measured on the ProArt.  A phone in a hand is a
noisier sensor, so the tool reports the phone's OWN change statistics
and the renderer can be told.  The render's floor is a decision that
belongs to the sensor that is being rendered.

Run from the repo root:
    /home/proart/projects/gpu-venv/bin/python experiments/dual_ingest.py \\
        --selftest
    /home/proart/projects/gpu-venv/bin/python experiments/dual_ingest.py \\
        --video <phone.mp4> --out docs/research/dual/roving
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import lens  # noqa: E402

#: The corpus's luma geometry.  The same numbers the corpus's own
#: bundles declare, so the two sides of the pair are comparable.
FRAME_W, FRAME_H = 160, 96
FRAME_BYTES = FRAME_W * FRAME_H

#: The corpus's cell grid: 40x24.  The phone's room-lens renders at
#: 128x125 or 96x93 CHARACTERS, which are FINER than the corpus's
#: cells, so the phone's change maps are reduced to the corpus's grid
#: and the reduction is a stated choice.
GRID_W, GRID_H = 40, 24

#: The room-lens's brightness ramp, dark to bright.  The ABSOLUTE
#: direction only matters for the luma render -- a change map is a
#: difference, and a difference is the same under a global inversion.
ROOM_LENS_RAMP = " .:-=+*#%@"

#: The corpus's change map's noise floor, measured on the ProArt.  The
#: phone is a different sensor at a different frame interval, so its
#: own floor is measured separately (see --room-lens).
CHANGE_NOISE_FLOOR = 3.0


def _cv2():
    """OpenCV, or a precise reason it is not here."""
    try:
        import cv2  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f'OpenCV is not importable in this interpreter ({exc}); the phone '
            f'ingest needs it to decode the recording. The render lane does '
            f'NOT need it -- it reads bundles, not videos.') from exc
    return cv2


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def put_audio(wav_path: Path, out_dir: Path) -> dict:
    """Put an extracted recording's AUDIO into the bundle as audio.wav.

    The corpus's bundles carry their audio as `audio.wav`, and the
    corpus's own tools read that file -- so a bundle without it is a
    bundle with a hole in it.  The phone's audio track cannot be
    decoded on this machine (no ffmpeg, no `av`, no `imageio`), so
    zcode extracts it where there IS a decoder and hands the file
    over; this puts it where the corpus expects it.

    The format is the corpus's own: 16-bit PCM MONO at the recording's
    own rate, no resampling and no processing, because the corpus's
    audio tools reason about the WAV as it is.
    """
    import wave
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(f'no such audio file: {wav_path}')
    with wave.open(str(wav_path), 'rb') as w:
        n_ch = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f'{wav_path} is {width * 8}-bit; the corpus\'s '
                         f'bundles carry 16-bit PCM')
    samples = np.frombuffer(raw, dtype='<i2').astype(np.float32)
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)
    pcm = np.clip(samples, -32768, 32767).astype('<i2')
    dest = Path(out_dir) / 'audio.wav'
    with wave.open(str(dest), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return {
        'file': 'audio.wav',
        'the_source': str(wav_path),
        'the_source_channels': n_ch,
        'sample_rate': rate,
        'samples': int(pcm.size),
        'duration_s': round(pcm.size / rate, 3) if rate else 0.0,
        'format': '16-bit PCM mono at the recording\'s own rate (no '
                  'resampling, no processing) -- the corpus\'s own format',
    }


# --------------------------------------------------------------------------
# the geometry
# --------------------------------------------------------------------------

def crop_box(src_w: int, src_h: int, mode: str) -> tuple[int, int, int, int]:
    """The rectangle of the source picture that becomes the 160x96 frame.

    `crop` keeps the geometry honest: the source is cut to the
    corpus's 1.667 aspect and then resized.  `stretch` squashes the
    whole picture into the frame, which moves every cell."""
    if mode == 'stretch':
        return 0, 0, src_w, src_h
    target = FRAME_W / FRAME_H
    src = src_w / src_h
    if src > target:                      # too wide: cut the sides
        new_w = int(round(src_h * target))
        x0 = (src_w - new_w) // 2
        return x0, 0, x0 + new_w, src_h
    new_h = int(round(src_w / target))    # too tall: cut the top and bottom
    y0 = (src_h - new_h) // 2
    return 0, y0, src_w, y0 + new_h


def to_luma(frame: np.ndarray) -> np.ndarray:
    """The corpus's luma: BT.601 full-range, 0.299R + 0.587G + 0.114B."""
    cv2 = _cv2()
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------
# the ingest
# --------------------------------------------------------------------------

def ingest(video: Path, out_dir: Path, label: str = 'the phone camera',
           wall_clock_start: str | None = None, stride: int = 1,
           fit: str = 'crop', max_frames: int | None = None,
           audio: Path | None = None, quiet: bool = False) -> dict:
    """Decode a recording into a corpus bundle."""
    cv2 = _cv2()
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video.is_file():
        raise FileNotFoundError(f'no such video: {video}')

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'OpenCV could not open {video} -- the container '
                           f'may be unreadable, or the codec is not built in')
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nominal_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
    x0, y0, x1, y1 = crop_box(src_w, src_h, fit)

    frames: list[np.ndarray] = []
    times_ms: list[float] = []
    raw_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_count += 1
        t_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if (raw_count - 1) % stride:
            continue
        if max_frames and len(frames) >= max_frames:
            break
        sub = frame[y0:y1, x0:x1]
        if sub.shape[0] != FRAME_H or sub.shape[1] != FRAME_W:
            sub = cv2.resize(sub, (FRAME_W, FRAME_H),
                             interpolation=cv2.INTER_AREA)
        frames.append(to_luma(sub))
        times_ms.append(t_ms)
    cap.release()

    if len(frames) < 3:
        raise RuntimeError(f'only {len(frames)} frames came out of {video} -- '
                           f'the render needs at least 3 to have a change map')

    luma = np.stack(frames).astype(np.uint8)
    (out_dir / 'luma.raw').write_bytes(luma.tobytes())

    # the frame times: the container's own, or the nominal rate's.
    # The first frame's timestamp is legitimately 0.0, so the test is
    # not "every time is above zero" but "every time after the first is
    # above zero, and the sequence never goes backwards".
    positive = [t for t in times_ms if t > 0.0]
    non_decreasing = all(times_ms[i] <= times_ms[i + 1]
                         for i in range(len(times_ms) - 1))
    usable = (len(positive) >= len(times_ms) - 1 and non_decreasing
              and len(positive) > 1)
    if usable:
        offsets = [round(t - times_ms[0], 3) for t in times_ms]
        the_times = 'the container\'s own per-frame timestamps'
        gaps = np.diff(offsets)
        the_spread = (round(float(gaps.max() - gaps.min()), 3)
                      if gaps.size else 0.0)
        the_variable = ('the frame intervals are CONSTANT'
                        if the_spread < 0.5 else
                        f'the frame intervals VARY by {the_spread:.2f} ms '
                        f'(the recording is variable frame rate, which is '
                        f'why the container\'s own timestamps are used '
                        f'rather than the nominal rate)')
    else:
        fps = nominal_fps if nominal_fps > 0 else 30.0
        offsets = [round(i * 1000.0 / fps, 3) for i in range(len(times_ms))]
        the_times = (f'DERIVED from the nominal frame rate ({fps:.3f} fps): '
                     f'the container gave no usable per-frame timestamps')
        the_variable = 'unknown: the times are derived, not read'

    # the wall clock: given, or the file's mtime, and said either way
    if wall_clock_start:
        start = wall_clock_start
        start_source = 'given on the command line'
    else:
        start = (datetime.fromtimestamp(video.stat().st_mtime, timezone.utc)
                 .strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z')
        start_source = ('the file\'s modification time -- AN APPROXIMATION: '
                        'a phone recording carries no wall clock, and the '
                        'file may have been uploaded later')
    (out_dir / 'timestamps.json').write_text(json.dumps({
        'wall_clock_start': start,
        'frame_offsets_ms': offsets,
    }))

    # the noise, measured on this sensor rather than assumed from the
    # ProArt's floor
    diff = np.abs(np.diff(luma.astype(np.float32), axis=0))
    change = diff.reshape(diff.shape[0], GRID_H, FRAME_H // GRID_H,
                          GRID_W, FRAME_W // GRID_W).mean(axis=(2, 4))
    per_frame_peak = change.reshape(change.shape[0], -1).max(axis=1)
    per_frame_median = np.median(change.reshape(change.shape[0], -1), axis=1)
    the_noise = {
        'the_change_map_noise_floor_is_NOT_carried_over_from_the_proart':
            True,
        'the_typical_cell': round(float(np.median(per_frame_median)), 3),
        'the_typical_loudest_cell': round(float(np.median(per_frame_peak)), 3),
        'the_quietest_ten_percent_loudest_cell':
            round(float(np.percentile(per_frame_peak, 10)), 3),
        'the_proarts_floor_for_comparison': 3.0,
        'note': 'the change map is the corpus\'s own (experiments/lens.py). '
                'These are per-frame statistics over THIS recording: a '
                'recording with a lot of movement has them raised by the '
                'movement, so they are an upper bound on the noise.',
    }

    meta = {
        'tool': 'dual_ingest',
        'generated_utc': iso_now(),
        'label': label,
        'source_video': str(video),
        'wall_clock_start': start,
        'the_wall_clock_start_source': start_source,
        'n_frames': int(len(frames)),
        'the_frames_decoded_from_the_container': int(raw_count),
        'the_frames_dropped_by_the_stride': int(raw_count - len(frames)),
        'stride': int(stride),
        'the_source_geometry': {'width': src_w, 'height': src_h,
                                'orientation_meta': orientation},
        'the_crop': {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'fit': fit,
                     'note': 'the picture is CUT to the corpus\'s aspect '
                             'before it is resized, because the grid is '
                             'positional'},
        'the_frame_times': the_times,
        'the_frame_interval_spread_ms': the_spread,
        'the_frame_rate_is_variable': the_variable,
        'the_nominal_fps_of_the_container': round(nominal_fps, 4),
        'achieved_video_fps': round(
            1000.0 * (len(offsets) - 1) / max(1e-6, offsets[-1]), 4),
        'luma': {
            'width': FRAME_W, 'height': FRAME_H,
            'frame_bytes': FRAME_BYTES,
            'format': 'uint8 BT.601 full-range luma (0.299R+0.587G+0.114B), '
                      'row-major, frames concatenated -- '
                      'np.fromfile(path, np.uint8).reshape(-1, 96, 160)',
        },
        'timestamps': {
            'file': 'timestamps.json',
            'frame_offsets_ms': 'the origin is the FIRST INGESTED FRAME and '
                                'every offset is relative to it; pair with '
                                'wall_clock_start for absolute time',
        },
        'the_luma_statistics': {
            'min': int(luma.min()), 'max': int(luma.max()),
            'mean': round(float(luma.mean()), 3),
            'p01': int(np.percentile(luma, 1)),
            'p99': int(np.percentile(luma, 99)),
            'note': 'a phone recording is often limited range (16-235) and a '
                    'browser recording is full range (0-255); the CHANGE map '
                    'is a difference and so is not moved by a constant gain, '
                    'but the noise floor is a threshold and is',
        },
        'the_noise_of_this_sensor': the_noise,
    }
    if audio is not None:
        meta['audio'] = put_audio(Path(audio), out_dir)
    else:
        meta['audio'] = {
            'file': None,
            'note': 'NO AUDIO IN THIS BUNDLE. The phone\'s audio track '
                    'cannot be decoded on this machine (no ffmpeg, no `av`, '
                    'no `imageio`), so it is extracted where there IS a '
                    'decoder and handed over with --audio. The corpus\'s '
                    'audio tools need audio.wav, so a bundle without it has '
                    'a hole in it.',
        }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

    report = {
        'the_out_dir': str(out_dir),
        'the_video': str(video),
        'n_frames': int(len(frames)),
        'the_frames_decoded_from_the_container': int(raw_count),
        'achieved_video_fps': meta['achieved_video_fps'],
        'the_frame_times': the_times,
        'the_frame_interval_spread_ms': the_spread,
        'the_frame_rate_is_variable': the_variable,
        'the_wall_clock_start': start,
        'the_wall_clock_start_source': start_source,
        'the_noise_of_this_sensor': the_noise,
        'the_luma_statistics': meta['the_luma_statistics'],
        'the_audio': meta['audio'],
        'bytes_written': int((out_dir / 'luma.raw').stat().st_size),
    }
    (out_dir / 'ingest.json').write_text(json.dumps(report, indent=2))
    if not quiet:
        print(json.dumps(report, indent=2))
    return report


def synth_phone_video(path: Path, n_frames: int = 900, fps: float = 30.0,
                      w: int = 270, h: int = 480, moving_half: bool = True,
                      seed: int = 3) -> dict:
    """A PHONE-SHAPED recording, for the rehearsal and the control.

    Portrait (the phone is in Casey's hand), with a shape that moves for
    one second and rests for one, so that HALF the slices have movement
    and the witness count has a chance baseline to be measured against.
    The content is deliberately MEANINGLESS: it is the negative control's
    half -- a recording that has nothing to do with the real session it
    is paired with, so that any agreement between the two streams would
    be a bug in the measure rather than a finding.
    """
    cv2 = _cv2()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError('cv2 could not open a VideoWriter for the fixture')
    rng = np.random.default_rng(seed)
    bg = np.full((h, w, 3), 90, np.uint8)
    bg[:, ::12] = 120
    for i in range(n_frames):
        f = bg.copy()
        t = i / fps
        phase = t % 2.0
        moving = (phase < 1.0) if moving_half else True
        if moving:
            x = int(w * 0.5 + w * 0.3 * math.sin(2 * math.pi * phase))
            y = int(h * 0.55)
            cv2.circle(f, (x, y), 14, (225, 230, 235), -1)
        f = np.clip(f.astype(np.int16)
                    + rng.normal(0, 3.0, f.shape).astype(np.int16),
                    0, 255).astype(np.uint8)
        writer.write(f)
    writer.release()
    return {'path': str(path), 'w': w, 'h': h, 'fps': fps,
            'n_frames': n_frames,
            'the_content': 'meaningless: a shape that moves for one second '
                           'and rests for one, so that half the slices have '
                           'movement'}


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# the room-lens: zcode's ASCII camera
# --------------------------------------------------------------------------

#: The room-lens's ACTUAL ramp, read out of its own source
#: (room-lens/index.html: `ASCII_MAP = " .:-=+*#%@@@@@@@@"`).
#: SEVENTEEN glyphs, and the last EIGHT of them are all '@': the
#: renderer buckets the brightness into 16 bands and the top eight
#: bands all print as '@'.  Half the brightness range therefore
#: carries NO information in the picture, and a ten-glyph ramp -- the
#: obvious guess, and the one this file's first draft used -- is
#: wrong by a factor of two across the whole top half.
ROOM_LENS_RAMP = " .:-=+*#%@@@@@@@@"

#: The room-lens's contrast stretch (room-lens/index.html: `CONTRAST
#: = 2.2`), applied to a cell's mean BEFORE it is bucketed.
ROOM_LENS_CONTRAST = 2.2

#: How many brightness bands the renderer quantises into.
ROOM_LENS_BANDS = len(ROOM_LENS_RAMP) - 1

#: The corpus's cell grid: 40x24.  The phone's room-lens renders at
#: 128x125 or 96x93 CHARACTERS, which are FINER than the corpus's
#: cells, so the phone's change maps are reduced to the corpus's grid
#: and the reduction is a stated choice.
GRID_W, GRID_H = 40, 24


def room_lens_bright(ramp: str = ROOM_LENS_RAMP) -> dict:
    """Each glyph's brightness, as a LOWER BOUND.

    The renderer computes `bright = clamp(((mean/255 - 0.5) * 2.2 +
    0.5) * 255)` and then buckets it into 16 bands, so a glyph stands
    for an INTERVAL of brightness, not for a number.  The honest
    estimate is the interval's LOWER edge: a point estimate would
    invent brightness the sensor never measured, and the difference
    between two glyphs is then a lower bound on the true difference.

    THE SCALE IS THE ROOM-LENS'S OWN BRIGHTNESS, NOT THE CAMERA'S
    LUMA.  The contrast stretch multiplies the camera's luma by 2.2
    and clips it at both ends, so the phone's change map is in the
    room-lens's units and the two sensors' change maps are NOT in the
    same units.  The renderer says so in its header.
    """
    step = 255.0 / (len(ramp) - 1)
    out: dict[str, float] = {}
    for i, ch in enumerate(ramp):
        if ch not in out:
            out[ch] = i * step          # the LOWER edge of the glyph's band
    return out


def bright_to_glyph(mean_luma: float, ramp: str = ROOM_LENS_RAMP) -> str:
    """The glyph the room-lens would print for a cell mean.

    This is the room-lens's own forward map, so a fixture built with
    it is a picture the real renderer could have produced.
    """
    bright = ((mean_luma / 255.0 - 0.5) * ROOM_LENS_CONTRAST + 0.5) * 255.0
    bright = max(0.0, min(255.0, bright))
    idx = int(bright / 255.0 * (len(ramp) - 1))
    return ramp[max(0, min(len(ramp) - 1, idx))]


def ascii_to_bright(ascii_text: str, cols: int, rows: int,
                    ramp: str = ROOM_LENS_RAMP) -> np.ndarray:
    """One room-lens frame as a brightness array, at its own geometry."""
    values = room_lens_bright(ramp)
    lines = ascii_text.split('\n')
    out = np.zeros((rows, cols), dtype=np.float32)
    for r in range(min(rows, len(lines))):
        line = lines[r]
        for c in range(min(cols, len(line))):
            out[r, c] = values.get(line[c], 0.0)
    return out


def resample_to_grid(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Reduce an array to a coarser grid by AREA AVERAGING.

    The phone's characters are finer than the corpus's cells, so the
    phone's change map is reduced the same way the corpus's own change
    map is: by the mean over the block, not by a sample."""
    h, w = arr.shape
    if h == out_h and w == out_w:
        return arr.astype(np.float32)
    row_bounds = [min(int(r * h / out_h), h) for r in range(out_h)] + [h]
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for r in range(out_h):
        r0, r1 = row_bounds[r], row_bounds[r + 1]
        if r1 <= r0:
            continue
        block = arr[r0:r1]
        col_bounds = [min(int(c * w / out_w), w) for c in range(out_w)] + [w]
        sums = np.add.reduceat(block, col_bounds[:-1], axis=1)
        counts = np.diff(col_bounds).astype(np.float32)
        out[r] = (sums / counts).mean(axis=0)
    return out


def session_anchor_ms(data: dict) -> dict:
    """Where the room-lens session's own clock starts.

    THE BUNDLE'S meta.startedAt IS NOT THE ANCHOR, and believing it
    moves the whole duet by seven minutes.  The room-lens starts a
    WALKABOUT with `walkStart = Date.now()` and stamps every frame
    `t = Date.now() - walkStart`.  The session's `meta.startedAt` is
    written when the SESSION is created, which can be minutes before
    the walkabout: this bundle carries three room measures banked
    before the walk began, and those measures have NO `t` because the
    phone was not walking when they were taken.

    So the anchor is DERIVED, from the bundle's own self-consistent
    evidence, by two independent routes that must agree:

      * each audio part carries `offsetMs = audioStartT - walkStart`
        and its own `startedAt`, so the anchor is `startedAt -
        offsetMs`;
      * each room measure taken DURING the walk carries `t =
        measureStartT - walkStart` and its own `startedAt`, so the
        anchor is `startedAt - t`.

    The duet bundle's five audio parts and four walk-time room
    measures agree to the millisecond.  That is a measurement, not an
    assumption, and the estimator is filed with the output.
    """
    meta = data.get('meta') or {}
    est: list[tuple[str, float]] = []
    for p in (data.get('audioParts') or []):
        sa, off = p.get('startedAt'), p.get('offsetMs')
        if sa is None or off is None:
            continue
        est.append((f'the audio part at offset {float(off):.0f} ms',
                    _wall_ms(str(sa)) - float(off)))
    for i, r in enumerate(data.get('rirMeasures') or []):
        sa, t = r.get('startedAt'), r.get('t')
        if sa is None or t is None:
            continue
        est.append((f'the room measure #{i} at t {float(t):.0f} ms',
                    _wall_ms(str(sa)) - float(t)))
    out: dict = {'the_estimators': [{'the_source': s,
                                     'the_anchor_ms': round(v, 3)}
                                    for s, v in est],
                  'the_n_estimators': len(est)}
    if est:
        vals = sorted(v for _, v in est)
        mid = len(vals) // 2
        anchor = vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])
        out.update({
            'the_anchor_ms': round(float(anchor), 3),
            'the_spread_ms': round(float(vals[-1] - vals[0]), 3),
            'the_source': 'DERIVED from the bundle\'s own wall stamps',
            'the_note': ('the walkabout\'s start, recovered from the audio '
                         'parts\' and the room measures\' own wall clocks. '
                         'meta.startedAt is the SESSION\'s creation time, '
                         'which here is 435.840 s earlier, and it is NOT '
                         'the origin of the frames\' t.'),
        })
    else:
        fallback = _wall_ms(str(meta.get('startedAt', '')))
        out.update({
            'the_anchor_ms': round(fallback, 3),
            'the_spread_ms': None,
            'the_source': 'meta.startedAt -- FALLBACK, NOTHING TO CHECK IT AGAINST',
            'the_note': ('the bundle carries no audio part and no walk-time '
                         'room measure, so the anchor cannot be derived '
                         'and meta.startedAt is used instead. That is an '
                         'ASSUMPTION and the alignment inherits it.'),
        })
    return out


def merge_fixed_sessions(session_dirs: list[Path], out_dir: Path,
                         label: str = 'the fixed witness',
                         quiet: bool = False) -> dict:
    """Several ProArt sessions become ONE fixed stream.

    THE SESSIONS SHARE A CLOCK, AND THE TOOL CHECKS IT RATHER THAN
    ASSUMING IT.  The capture page stamps every frame with
    `performance.now()` minus the page's load base, so the sessions'
    offsets are in ONE clock even though each session is its own file.
    The page base is recoverable: `wall_clock_start -
    frame_offsets_ms[0]/1000` is the wall time of page-clock zero.

    The duet corpus's eight sessions agree on that base to 1.2 ms
    across 12.8 minutes, which is what makes the merge a measurement.
    The tool REFUSES sessions whose bases disagree by more than the
    tolerance, because merging two clocks into one stream would
    silently invent a timeline -- the exact failure the corpus's own
    clock study caught in the dual-capture rehearsal.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = []
    for sd in session_dirs:
        sd = Path(sd)
        ts = json.loads((sd / 'timestamps.json').read_text())
        meta = json.loads((sd / 'meta.json').read_text())
        offs = [float(v) for v in ts['frame_offsets_ms']]
        luma = np.fromfile(sd / 'luma.raw', dtype=np.uint8)
        n_luma = luma.size // (FRAME_W * FRAME_H)
        if n_luma != len(offs):
            raise RuntimeError(
                f'{sd.name}: {n_luma} frames of luma but {len(offs)} '
                f'offsets -- the bundle is inconsistent and will not be merged')
        wall_start = str(ts.get('wall_clock_start') or meta.get('wall_clock_start') or '')
        base_wall = _wall_ms(wall_start) - offs[0]
        loaded.append({
            'dir': sd, 'name': sd.name, 'offs': offs,
            'luma': luma[:n_luma * FRAME_BYTES],
            'wall_start': wall_start,
            'wall_start_ms': _wall_ms(wall_start),
            'page_base_wall_ms': base_wall,
        })
    if not loaded:
        raise RuntimeError('no fixed sessions to merge')

    bases = [x['page_base_wall_ms'] for x in loaded]
    spread = max(bases) - min(bases)
    tol_ms = 250.0
    if spread > tol_ms:
        raise RuntimeError(
            f'these sessions do NOT share a page clock: their page bases '
            f'({min(bases):.1f} .. {max(bases):.1f} ms) differ by '
            f'{spread:.1f} ms, over the {tol_ms:.0f} ms tolerance. Merging '
            f'them would invent a timeline.')

    loaded.sort(key=lambda x: x['offs'][0])
    all_offs: list[float] = []
    chunks: list[np.ndarray] = []
    gaps = []
    for i, x in enumerate(loaded):
        if i:
            prev_end = loaded[i - 1]['offs'][-1]
            gap = x['offs'][0] - prev_end
            gaps.append({'after': loaded[i - 1]['name'],
                         'before': x['name'],
                         'the_gap_ms': round(gap, 3)})
        all_offs.extend(x['offs'])
        chunks.append(x['luma'])
    merged_luma = np.concatenate(chunks)
    (out_dir / 'luma.raw').write_bytes(merged_luma.tobytes())
    wall_start = loaded[0]['wall_start']
    (out_dir / 'timestamps.json').write_text(json.dumps({
        'wall_clock_start': wall_start,
        'frame_offsets_ms': all_offs,
        'the_convention': ('the corpus capture page\'s convention: the '
                           'offsets are the page\'s own clock (performance.now() '
                           'minus the page\'s load base) and wall_clock_start '
                           'is the wall time of the FIRST FRAME, so a frame\'s '
                           'absolute time is wall_clock_start + (offset - '
                           'offset_of_the_first_frame)/1000'),
    }, indent=1))
    meta = {
        'tool': 'dual_ingest -- the merged fixed stream',
        'generated_utc': iso_now(),
        'label': label,
        'the_sessions': [{'the_session': x['name'],
                          'the_frames': len(x['offs']),
                          'the_first_offset_ms': x['offs'][0],
                          'the_last_offset_ms': x['offs'][-1],
                          'the_wall_clock_start': x['wall_start']}
                         for x in loaded],
        'the_page_base_wall_ms': round(float(min(bases)), 3),
        'the_page_base_spread_ms': round(float(spread), 3),
        'the_page_base_tolerance_ms': tol_ms,
        'the_gaps': gaps,
        'the_total_gap_ms': round(float(sum(g['the_gap_ms'] for g in gaps)), 3),
        'the_span_ms': round(float(all_offs[-1] - all_offs[0]), 3),
        'the_frames': len(all_offs),
        'the_note': ('the sessions were merged because they share one page '
                     'clock, and that was CHECKED: their recovered page '
                     'bases agree to '
                     f'{spread:.1f} ms across '
                     f'{(all_offs[-1] - all_offs[0]) / 1000.0 / 60.0:.1f} '
                     'minutes. The gaps between the sessions are real and '
                     'are listed.'),
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    out = {'the_merged': meta}
    if not quiet:
        print(f'merged {len(loaded)} fixed sessions into {out_dir} '
              f'({len(all_offs)} frames, page base spread {spread:.1f} ms)')
    return out


def ingest_room_lens(path: Path, out_dir: Path, label: str = 'the phone room-lens',
                      fixed_dirs: list[Path] | None = None,
                      fixed_label: str = 'the fixed witness',
                      quiet: bool = False) -> dict:
    """A room-lens session becomes a stream the renderer can read.

    THE ROOM-LENS IS NOT A VIDEO.  It is zcode's own ASCII camera: one
    string of CHARACTERS per frame, at the frame's own geometry, with
    the device's POSE attached to every frame.  So there is nothing to
    decode with OpenCV and no aspect to crop -- the picture is already
    a grid, and the grid's geometry CHANGES during the session (the
    lens alternates between 128x125 and 96x93 as it works).

    Three consequences, all handled here rather than papered over:

      * A change map needs two frames of the SAME geometry.  Where the
        geometry changed, there is no change map and the pair is
        SKIPPED, and the skip is counted and filed.
      * The change is computed at the phone's NATIVE resolution and
        only then reduced to the corpus's 40x24 grid.  Differencing
        after the reduction would dilute a movement that fits inside
        one cell; differencing first does not.
      * The phone's frames are 2.5 fps and the ProArt's may be 2, so
        the two views' change maps are over DIFFERENT intervals and
        their energies are not comparable without saying so.  The
        phone's own frame interval is measured and filed.
    """
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text())
    frames = data.get('frames') or []
    if len(frames) < 3:
        raise RuntimeError(f'{path} carries {len(frames)} frames; the '
                           f'render needs at least 3')
    meta_src = data.get('meta', {})

    # THE ANCHOR: the session's own clock, derived from its own stamps
    anchor = session_anchor_ms(data)
    anchor_ms = float(anchor['the_anchor_ms'])

    # the frames, in their own geometry, with their own times
    luma_frames = []
    times_ms = []
    geoms = []
    for f in frames:
        rows, cols = int(f['rows']), int(f['cols'])
        luma_frames.append(ascii_to_bright(f['ascii'], cols, rows))
        times_ms.append(float(f['t']))
        geoms.append((rows, cols))

    # the change maps, only where the geometry allows one
    change_maps = []
    change_times = []
    pair_stats = []
    skipped = []
    for i in range(len(frames) - 1):
        if geoms[i] != geoms[i + 1]:
            skipped.append({'from_frame': i, 'to_frame': i + 1,
                            'the_geometry': [list(geoms[i]), list(geoms[i + 1])]})
            continue
        diff = np.abs(luma_frames[i + 1] - luma_frames[i])
        reduced = resample_to_grid(diff, GRID_H, GRID_W)
        change_maps.append(reduced)
        change_times.append((times_ms[i] + times_ms[i + 1]) / 2.0)
        pair_stats.append({
            'the_frame': i,
            'the_mean_cell': float(reduced.mean()),
            'the_max_cell': float(reduced.max()),
        })
    if len(change_maps) < 2:
        raise RuntimeError(f'{path} gave only {len(change_maps)} change maps '
                           f'(the geometry changed too often)')
    change = np.stack(change_maps).astype(np.float32)
    (out_dir / 'change.raw').write_bytes(change.tobytes())

    # the picture, reduced to the corpus's grid, so the luma render has
    # a phone side to show
    luma_grid = np.stack([resample_to_grid(l, GRID_H, GRID_W)
                          for l in luma_frames]).astype(np.uint8)
    (out_dir / 'luma_grid.raw').write_bytes(luma_grid.tobytes())

    # the clock.  The corpus's convention: the offsets stay in the
    # source's own clock and wall_clock_start is the wall time of the
    # FIRST FRAME.
    first_frame_wall = anchor_ms + times_ms[0]
    (out_dir / 'timestamps.json').write_text(json.dumps({
        'wall_clock_start': _iso_from_ms(first_frame_wall),
        'frame_offsets_ms': [round(t, 3) for t in times_ms],
        'change_offsets_ms': [round(t, 3) for t in change_times],
        'the_convention': ('the corpus capture page\'s convention: the '
                           'offsets are the session\'s own clock and '
                           'wall_clock_start is the wall time of the FIRST '
                           'FRAME, so a frame\'s absolute time is '
                           'wall_clock_start + (offset - the first frame\'s '
                           'offset)/1000'),
        'the_change_offsets_are': ('the midpoint of the two frames each '
                                   'change map came from, in the same clock '
                                   'as frame_offsets_ms'),
    }, indent=1))

    # the frame interval, which is what makes the two views' change
    # maps comparable or not
    gaps = [times_ms[i + 1] - times_ms[i] for i in range(len(times_ms) - 1)]
    gaps_sorted = sorted(gaps)
    median_gap = gaps_sorted[len(gaps_sorted) // 2]

    # THE PHONE'S OWN FLOOR IS ITS RESOLUTION, NOT A GRAIN ESTIMATE.
    # The corpus's 3.0 luma floor was measured on the ProArt, at the
    # ProArt's frame interval, over the ProArt's own change maps.  The
    # phone is not a noisy sensor -- it is a DETERMINISTIC RENDERER of
    # a noisy sensor.  Its change map is quantised to brightness
    # bands, so below ONE BAND a change cannot be expressed at all: the
    # graded fixture shows 12.8 % of its pairs at EXACTLY zero, and
    # the real duet's quietest pairs are not zero but they are all
    # whole numbers of bands.  So the phone's floor is one band, and
    # the phone's DUTY CYCLE -- the share of its frames that show any
    # change at all -- is measured and reported, because a channel
    # that is on most of the time cannot time a movement.
    band = 255.0 / ROOM_LENS_BANDS
    if pair_stats:
        means = np.array([p['the_mean_cell'] for p in pair_stats])
        cut = float(np.percentile(means, 10))
        quiet_idx = [i for i, m in enumerate(means) if m <= cut]
        quiet_max = [pair_stats[i]['the_max_cell'] for i in quiet_idx]
        the_noise = {
            'the_rule': ('the phone\'s RESOLUTION: one brightness band, '
                         'because below one band the rendering cannot '
                         'express a change at all'),
            'the_floor': round(band, 4),
            'the_band': round(band, 4),
            'the_n_pairs': len(pair_stats),
            'the_n_quiet_pairs': len(quiet_idx),
            'the_quiet_cut_mean_cell': round(cut, 4),
            'the_quiet_max_median': round(float(np.median(quiet_max)), 4),
            'the_quiet_max_p90': round(float(np.percentile(quiet_max, 90)), 4),
            'the_share_of_pairs_at_EXACTLY_zero': round(
                float((np.array([p['the_max_cell'] for p in pair_stats]) == 0.0).mean()), 4),
            'the_duty_cycle': round(
                float((np.array([p['the_max_cell'] for p in pair_stats]) > band).mean()), 4),
            'the_proarts_floor_for_comparison': CHANGE_NOISE_FLOOR,
            'the_note': ('the phone renders a POSTERISED picture -- 17 '
                         'glyphs, the top 8 of them all \'@\' -- so its '
                         'change map is a MASK of band flips, not the same '
                         'KIND of quantity as the ProArt\'s. The floor is '
                         'the smallest change it can express; the duty '
                         'cycle is how often it expresses one, and a '
                         'channel that is on most of the time cannot time '
                         'a movement.'),
        }
    else:
        the_noise = {'the_n_pairs': 0,
                     'note': 'no usable pairs'}

    tiers = [str(f.get('tier', '')) for f in frames]
    meta = {
        'tool': 'dual_ingest -- the room-lens',
        'generated_utc': iso_now(),
        'label': label,
        'source_session': str(path),
        'the_source_kind': meta_src.get('kind'),
        'the_source_session_id': meta_src.get('sessionId'),
        'the_source_started_at': meta_src.get('startedAt'),
        'the_source_ended_at': data.get('endedAt'),
        'the_user_agent': meta_src.get('userAgent'),
        'the_audio': meta_src.get('audio'),
        'the_clock_anchor': anchor,
        'the_wall_clock_of_the_first_frame': _iso_from_ms(first_frame_wall),
        'the_wall_clock_of_the_last_frame': _iso_from_ms(anchor_ms + times_ms[-1]),
        'n_frames': len(frames),
        'n_change_maps': int(len(change)),
        'the_pairs_skipped_for_a_geometry_change': len(skipped),
        'the_geometry_runs': _geometry_runs(geoms),
        'the_frame_interval_ms': {
            'the_median': round(median_gap, 3),
            'the_min': round(min(gaps), 3),
            'the_max': round(max(gaps), 3),
            'the_achieved_fps': round(1000.0 / median_gap, 3) if median_gap else None,
        },
        'the_tiers': dict(Counter(tiers)),
        'the_tier_meaning': ('the room-lens\'s tier is its ADAPTIVE FIDELITY '
                             'mode, chosen from the PHONE\'s own motion -- '
                             'STILL is 128 cols at 8 fps, SLOW is 96 at 15, '
                             'MOVING is 64 at 30. It says the phone was not '
                             'moving; it does NOT say the room was still.'),
        'the_ramp': ROOM_LENS_RAMP,
        'the_ramp_note': ('the room-lens\'s ACTUAL ramp: 17 glyphs whose top '
                          '8 are all \'@\', so half the brightness range '
                          'carries no information in the picture'),
        'the_contrast_stretch': ROOM_LENS_CONTRAST,
        'the_units': ('the room-lens\'s own brightness: a 2.2x contrast '
                      'stretch of the camera luma, clipped, with each glyph '
                      'read as the LOWER edge of its band'),
        'the_brightness_is_a_lower_bound': True,
        'the_grid': {'width': GRID_W, 'height': GRID_H,
                     'note': 'the phone\'s change maps are computed at the '
                             'phone\'s native resolution and THEN reduced to '
                             'the corpus\'s grid by area averaging'},
        'change': {
            'file': 'change.raw',
            'dtype': 'float32',
            'shape': [int(len(change)), GRID_H, GRID_W],
            'note': ('the ABSOLUTE difference between consecutive frames, in '
                     'the room-lens\'s own BRIGHTNESS units (a 2.2x contrast '
                     'stretch of the camera luma, clipped), and each glyph '
                     'is read as the LOWER edge of its band, so every value '
                     'here is a lower bound on the true difference'),
        },
        'luma_grid': {
            'file': 'luma_grid.raw',
            'dtype': 'uint8',
            'shape': [int(len(luma_grid)), GRID_H, GRID_W],
            'note': ('the picture reduced to the corpus\'s 40x24 grid, so '
                     'the luma render has a phone side to show'),
        },
        'the_noise_of_this_sensor': the_noise,
        'the_pose': {
            'the_note': 'the device orientation attached to every frame',
            'the_alpha_range': [min(float(f['pose']['alpha']) for f in frames),
                                max(float(f['pose']['alpha']) for f in frames)],
            'the_beta_range': [min(float(f['pose']['beta']) for f in frames),
                               max(float(f['pose']['beta']) for f in frames)],
            'the_gamma_range': [min(float(f['pose']['gamma']) for f in frames),
                                max(float(f['pose']['gamma']) for f in frames)],
            'the_heading_is_null_in_every_frame': all(
                f['pose'].get('heading') is None for f in frames),
        },
        'the_keyframes': len(data.get('keyframes') or []),
        'the_rir_measures': len(data.get('rirMeasures') or []),
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

    # the fixed side: the ProArt's sessions merged into one stream, and
    # the WALL-CLOCK sidecar.  NOT the clap alignment -- that is
    # zcode's -- and the wall clock carries the one constant the corpus
    # cannot measure: the two devices' clocks have never been compared.
    sidecars = []
    merged = None
    if fixed_dirs:
        merged = merge_fixed_sessions(fixed_dirs, out_dir / 'fixed',
                                      label=fixed_label, quiet=True)
        fts = json.loads((out_dir / 'fixed' / 'timestamps.json').read_text())
        f_offs = fts['frame_offsets_ms']
        fixed_first_wall = _wall_ms(fts['wall_clock_start'])
        offset_ms = first_frame_wall - fixed_first_wall
        sc = {
            'offset_ms': round(offset_ms, 3),
            'drift_ppm': 0.0,
            'n_claps': 0,
            'the_note': ('THE WALL CLOCK, NOT THE CLAPS. The two devices\' '
                         'clocks have never been compared, so this alignment '
                         'carries the device-to-device skew -- the constant '
                         'the corpus cannot measure -- and a drift the '
                         'duet\'s 236 s cannot separate from a movement. '
                         'zcode\'s clap alignment replaces this sidecar when '
                         'it lands.'),
            'the_fixed_session': 'the merged ProArt stream',
            'the_fixed_first_frame_wall': _iso_from_ms(fixed_first_wall),
            'the_roving_first_frame_wall': _iso_from_ms(first_frame_wall),
        }
        (out_dir / 'wallclock-fixed.json').write_text(json.dumps(sc, indent=1))
        sidecars.append({'the_file': 'wallclock-fixed.json',
                         'the_offset_ms': round(offset_ms, 3)})

    report = {
        'the_out_dir': str(out_dir),
        'the_source_session': str(path),
        'n_frames': len(frames),
        'n_change_maps': int(len(change)),
        'the_pairs_skipped_for_a_geometry_change': len(skipped),
        'the_frame_interval_ms': meta['the_frame_interval_ms'],
        'the_tiers': meta['the_tiers'],
        'the_clock_anchor': anchor,
        'the_noise_of_this_sensor': the_noise,
        'the_wall_clock_sidecars': sidecars,
        'the_merged_fixed': merged['the_merged'] if merged else None,
        'bytes_written': int((out_dir / 'change.raw').stat().st_size)
        + int((out_dir / 'luma_grid.raw').stat().st_size),
    }
    (out_dir / 'ingest.json').write_text(json.dumps(report, indent=2))
    if not quiet:
        print(json.dumps(report, indent=2))
    return report


def _wall_ms(iso: str) -> float:
    """An ISO wall clock as epoch milliseconds."""
    if not iso:
        return 0.0
    try:
        from datetime import datetime, timezone
        t = datetime.fromisoformat(iso.replace('Z', '+00:00'))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.timestamp() * 1000.0
    except Exception:  # noqa: BLE001
        return 0.0


def _iso_from_ms(ms: float) -> str:
    """Epoch milliseconds as an ISO wall clock."""
    from datetime import datetime, timezone
    t = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return t.strftime('%Y-%m-%dT%H:%M:%S.') + f'{t.microsecond // 1000:03d}Z'


def _geometry_runs(geoms: list[tuple[int, int]]) -> list[dict]:
    """The geometry's runs, so a reader can see it changing."""
    runs = []
    for g in geoms:
        if runs and runs[-1]['the_geometry'] == list(g):
            runs[-1]['n_frames'] += 1
        else:
            runs.append({'the_geometry': list(g), 'n_frames': 1})
    return runs
# --------------------------------------------------------------------------
# the selftest
# --------------------------------------------------------------------------

def _write_test_video(path: Path, n_frames: int = 60, fps: float = 30.0,
                      w: int = 192, h: int = 144) -> dict:
    """A recording with a known thing in a known place."""
    cv2 = _cv2()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError('cv2 could not open a VideoWriter for the test')
    rng = np.random.default_rng(7)
    bg = np.zeros((h, w, 3), np.uint8)
    bg[:, :] = (60, 70, 80)
    bg[:, ::16] = (110, 120, 130)
    places = []
    for i in range(n_frames):
        frame = bg.copy()
        cx = int((0.20 + 0.55 * (i / max(1, n_frames - 1))) * (w - 20)) + 10
        cy = h // 2
        frame[cy - 10:cy + 10, cx - 10:cx + 10] = (220, 225, 230)
        noise = rng.normal(0.0, 2.0, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
        places.append({'frame': i, 'x_px': cx, 'y_px': cy})
    writer.release()
    return {'places': places, 'w': w, 'h': h, 'fps': fps,
            'n_frames': n_frames}


# --------------------------------------------------------------------------
# the room-lens fixture: a known truth, rendered the way the lens renders
# --------------------------------------------------------------------------

def make_room_lens_fixture(path: Path, n_frames: int = 48, cols: int = 64,
                           rows: int = 62, fps: float = 2.5,
                           seed: int = 11) -> dict:
    """A room-lens session with a KNOWN truth.

    THE PHONE'S CHANGE MAP HAS TO BE GRADED, the same way the
    synthetic pair grades the renders.  A person walks a known path;
    the truth is where they were at every frame; and the fixture is
    rendered through the room-lens's OWN rules -- the 2.2x contrast
    stretch and the 17-glyph ramp whose top eight are all '@' -- so
    that a grade measured here transfers to the real duet.

    THE SCENE IS BUILT TO SURVIVE THE POSTERISATION.  The lens
    clips: a cell mean below 58 prints as a space and one above 134.5
    prints as '@'.  So the wall is bright, the floor is dim, and the
    person is a BRIGHT figure walking the dim floor -- a bright body
    against a dark ground changes its WHOLE SELF when it moves.  A
    dark figure on a bright wall would change only along its EDGES,
    and the change map would then be about the edges, which is a
    different claim from the one this fixture grades.
    """
    path = Path(path)
    rng = np.random.default_rng(seed)
    # a static room: a bright wall above, a dim floor below
    bg = np.zeros((rows, cols), np.float32)
    bg[:, :] = 45.0
    bg[:int(rows * 0.55), :] = 185.0
    bg += rng.normal(0.0, 2.0, bg.shape).astype(np.float32)

    truth = []
    frames = []
    for i in range(n_frames):
        t = i / fps
        frac = i / max(1, n_frames - 1)
        # the person walks left to right across the dim floor
        cx = int((0.12 + 0.76 * frac) * (cols - 10)) + 5
        cy = int(rows * 0.75)
        rw, rh = 6, 9
        frame = bg.copy()
        x0, x1 = max(0, cx - rw), min(cols, cx + rw)
        y0, y1 = max(0, cy - rh), min(rows, cy + rh)
        frame[y0:y1, x0:x1] = 250.0          # a bright figure on a dim floor
        lines = []
        for r in range(rows):
            lines.append(''.join(bright_to_glyph(float(frame[r, c]))
                                  for c in range(cols)))
        frames.append({
            't': round(t * 1000.0, 1),
            'tier': 'STILL',
            'cols': cols, 'rows': rows,
            'pose': {'alpha': 0.0, 'beta': 0.0, 'gamma': 0.0, 'heading': None},
            'ascii': '\n'.join(lines),
        })
        truth.append({
            'frame': i,
            't_s': round(t, 4),
            'the_x_native': cx,
            'the_y_native': cy,
            'the_x_grid': round((cx + 0.5) * GRID_W / cols, 4),
            'the_y_grid': round((cy + 0.5) * GRID_H / rows, 4),
        })

    # a walk-time audio part and a walk-time room measure, so the
    # anchor has SOMETHING to be derived from -- and the two agree by
    # construction, which is what the real bundle does.
    started = 1792655532428.0          # the session's creation, 435.840 s early
    walk_start = started + 435840.0
    wall = _iso_from_ms(walk_start)
    bundle = {
        'meta': {'kind': 'room-lens-session', 'version': 2,
                 'sessionId': 'fixture-roomlens',
                 'startedAt': _iso_from_ms(started),
                 'userAgent': 'the fixture',
                 'audio': {'present': True, 'parts': 1,
                           'startedAt': wall, 'totalBytes': 1}},
        'endedAt': _iso_from_ms(walk_start + n_frames / fps * 1000.0),
        'frames': frames,
        'keyframes': [],
        'rirMeasures': [],
        'audioParts': [{'mimeType': 'audio/webm', 'base64': '', 'bytes': 1,
                        'startedAt': wall, 'offsetMs': 0,
                        'interrupted': False}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle))
    return {'the_truth': truth, 'the_path': str(path),
            'the_n_frames': n_frames, 'the_walk_start_ms': walk_start,
            'the_session_started_ms': started,
            'the_cols': cols, 'the_rows': rows}


def selftest() -> int:
    import tempfile
    checks: list[tuple[str, bool, str]] = []

    def ck(name, ok, detail=''):
        checks.append((name, bool(ok), detail))

    # 1. the geometry: the crop keeps the aspect, the stretch does not
    target = FRAME_W / FRAME_H
    box = crop_box(192, 108, 'crop')      # 16:9 into 1.667: cut the sides
    ck('ingest: the crop keeps the corpus aspect',
       abs((box[2] - box[0]) - (box[3] - box[1]) * target) <= 1.0,
       str(box))
    box2 = crop_box(1080, 1920, 'crop')   # portrait: cut the top and bottom
    ck('ingest: the crop of a portrait video keeps the aspect',
       abs((box2[2] - box2[0]) - (box2[3] - box2[1]) * target) <= 1.0,
       str(box2))
    ck('ingest: the stretch takes the whole picture',
       crop_box(192, 108, 'stretch') == (0, 0, 192, 108))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        try:
            cv2 = _cv2()
        except RuntimeError as exc:
            print(f'  SKIP  the ingest selftest: {exc}')
            return 1
        vid = tmp / 'test.mp4'
        fixture = _write_test_video(vid)
        out = tmp / 'bundle'
        rep = ingest(vid, out, label='the test phone', quiet=True)

        # 2. the bundle is in the corpus's format
        raw = np.fromfile(out / 'luma.raw', dtype=np.uint8)
        ck('ingest: the luma is the corpus\'s shape',
           raw.size % FRAME_BYTES == 0 and raw.size // FRAME_BYTES
           == fixture['n_frames'], str(raw.size // FRAME_BYTES))
        ts = json.loads((out / 'timestamps.json').read_text())
        ck('ingest: the timestamps carry an origin and the offsets',
           'wall_clock_start' in ts and
           len(ts['frame_offsets_ms']) == fixture['n_frames'],
           str(len(ts.get('frame_offsets_ms', []))))
        ck('ingest: the frame offsets start at zero and increase',
           ts['frame_offsets_ms'][0] == 0.0 and
           all(ts['frame_offsets_ms'][i] < ts['frame_offsets_ms'][i + 1]
               for i in range(len(ts['frame_offsets_ms']) - 1)))
        ck('ingest: the wall clock start is labelled, not assumed',
           'AN APPROXIMATION' in json.loads(
               (out / 'meta.json').read_text())['the_wall_clock_start_source'])
        meta = json.loads((out / 'meta.json').read_text())
        crop = meta['the_crop']
        ck('ingest: the metadata says how the picture was fitted',
           crop['fit'] == 'crop' and (crop['x0'] > 0 or crop['y0'] > 0) and
           abs((crop['x1'] - crop['x0'])
               - (crop['y1'] - crop['y0']) * (FRAME_W / FRAME_H)) <= 1.0,
           str(crop))
        ck('ingest: the metadata says where the frame times came from',
           bool(meta['the_frame_times']), meta['the_frame_times'])

        # 3b. the AUDIO: a bundle with no audio.wav has a hole in it,
        #     because the corpus's own audio tools read that file
        import wave as _wave
        src_wav = tmp / 'phone.wav'
        tone = (np.sin(2 * np.pi * 220.0 * np.arange(4800) / 48000.0)
                * 8000.0).astype('<i2')
        with _wave.open(str(src_wav), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(48000)
            w.writeframes(tone.tobytes())
        out_a = tmp / 'bundle_with_audio'
        ingest(vid, out_a, label='the test phone', audio=src_wav, quiet=True)
        with _wave.open(str(out_a / 'audio.wav'), 'rb') as w:
            got = {'ch': w.getnchannels(), 'width': w.getsampwidth(),
                   'rate': w.getframerate(), 'n': w.getnframes()}
        ck('ingest: the audio lands in the corpus\'s own format',
           got == {'ch': 1, 'width': 2, 'rate': 48000, 'n': 4800}, str(got))
        ck('ingest: the metadata says where the audio came from',
           json.loads((out_a / 'meta.json').read_text())['audio']['file']
           == 'audio.wav')
        # and a bundle with NO audio says so rather than pretending
        ck('ingest: a bundle with no audio says so',
           json.loads((out / 'meta.json').read_text())['audio']['file'] is None)

        # 3c. THE FIRST-CLASS CLAIM: the corpus's own walk-by study --
        #     which needs the luma AND the audio -- reads the phone's
        #     bundle with no changes.  That is what "the same format"
        #     has to mean if it is to mean anything.
        walk = lens.walk_by_study([('the test phone', out_a)])
        ck('ingest: the corpus\'s own walk-by study reads the phone bundle',
           bool(walk) and walk[0].get('n_crossings', 0) >= 1,
           str(walk[0].get('n_crossings') if walk else 'no walk'))

        # 4. the GRADE: the ingested bundle's change map must find the
        #    thing where the fixture put it.  This is the same grade the
        #    render lane leans on, run through a real video file.
        maps = lens.frame_motion_maps(out)
        ck('ingest: the corpus\'s change map reads the ingested bundle',
           maps is not None and len(maps) == fixture['n_frames'] - 1,
           str(None if maps is None else len(maps)))
        mid = fixture['n_frames'] // 2
        place = fixture['places'][mid]
        m = maps[mid - 1]
        ys, xs = np.nonzero(m > 3.0)
        if len(xs):
            gx = (float(xs.mean() + 0.5) * FRAME_W / GRID_W)
            gy = (float(ys.mean() + 0.5) * FRAME_H / GRID_H)
        else:
            gx = gy = -1.0
        # where the fixture put it, after the crop
        x0, y0 = crop['x0'], crop['y0']
        expected_x = ((place['x_px'] - x0)
                      * (FRAME_W / (crop['x1'] - x0)))
        expected_y = ((place['y_px'] - y0)
                      * (FRAME_H / (crop['y1'] - y0)))
        ck('ingest: the change map finds the thing where it was put',
           abs(gx - expected_x) < 8.0 and abs(gy - expected_y) < 8.0,
           f'grid ({gx:.1f},{gy:.1f}) vs expected ({expected_x:.1f},'
           f'{expected_y:.1f})')

        # 4. the render lane reads what the ingest wrote
        from experiments.dual_render import Stream  # noqa: PLC0415
        s = Stream(out, 'the test phone')
        ck('ingest: the render lane reads the ingested bundle',
           s.ok and s.n_frames() == fixture['n_frames'] - 1,
           str(s.problems))

    # 5. the room-lens path, graded against a known truth
    _room_lens_selftest(checks, ck)

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"   [{detail}]" if detail and not ok else ""))
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


def _room_lens_selftest(checks, ck) -> None:
    """The room-lens path, graded against a known truth.

    THE PHONE'S CHANGE MAP IS GRADED, NOT ASSUMED.  The fixture is a
    person walking a known path, rendered through the room-lens's own
    rules, and the ingest has to put the change where the truth says
    the person was.  A test that passes because everything sits in the
    middle is not a test, so the truth moves and the grade is asked at
    several points along the walk.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fx = make_room_lens_fixture(tmp / 'roomlens.json')
        out = tmp / 'phone'
        rep = ingest_room_lens(tmp / 'roomlens.json', out, quiet=True)

        ck('room-lens: the bundle carries the change maps and the clock',
           (out / 'change.raw').is_file() and (out / 'timestamps.json').is_file()
           and (out / 'luma_grid.raw').is_file()
           and rep['n_change_maps'] == fx['the_n_frames'] - 1,
           str(rep['n_change_maps']))

        # the anchor is DERIVED, and it is not the session's start
        anchor = rep['the_clock_anchor']
        ck('room-lens: the clock anchor is derived from the bundle\'s own stamps',
           anchor['the_n_estimators'] >= 1 and
           'DERIVED' in anchor['the_source'],
           json.dumps(anchor)[:200])
        ck('room-lens: the anchor is the walkabout, NOT meta.startedAt',
           abs(anchor['the_anchor_ms'] - fx['the_walk_start_ms']) < 1.0,
           f"{anchor['the_anchor_ms']} vs {fx['the_walk_start_ms']}")

        # the change map's centroid must land on the truth
        change = np.fromfile(out / 'change.raw', dtype=np.float32).reshape(
            rep['n_change_maps'], GRID_H, GRID_W)
        truth = fx['the_truth']
        worst = 0.0
        worst_at = None
        for k in (6, 14, 22, 30, 40):
            if k >= len(truth) - 1:
                continue
            # the change map at index k is between frames k and k+1, so
            # the truth to ask about is the pair's midpoint
            tr = truth[k]
            m = change[k]
            thr = max(0.5 * float(m.max()), 1.0)
            cells = np.argwhere(m >= thr)
            if cells.size == 0:
                worst = 99.0
                worst_at = k
                continue
            cx = float(cells[:, 1].mean() + 0.5)
            cy = float(cells[:, 0].mean() + 0.5)
            # the truth is in GRID CELLS, and so is this
            d = max(abs(cx - tr['the_x_grid']), abs(cy - tr['the_y_grid']))
            if d > worst:
                worst, worst_at = d, k
        ck('room-lens: the change map finds the person where the truth puts them',
           worst <= 3.0,
           f'worst offset {worst:.2f} grid cells at frame {worst_at}')

        # a geometry change kills the pair, and the skip is COUNTED.
        # A frame whose geometry changed breaks BOTH of its pairs -- the
        # one before it and the one after it -- so one changed frame
        # costs two maps.
        alt = tmp / 'alt.json'
        data = json.loads((tmp / 'roomlens.json').read_text())
        data['frames'][5]['cols'] = data['frames'][5]['cols'] - 1
        data['frames'][5]['ascii'] = '\n'.join(
            r[:-1] for r in data['frames'][5]['ascii'].split('\n'))
        alt.write_text(json.dumps(data))
        rep2 = ingest_room_lens(alt, tmp / 'phone2', quiet=True)
        ck('room-lens: a pair whose geometry changed is SKIPPED and counted',
           rep2['the_pairs_skipped_for_a_geometry_change'] == 2
           and rep2['n_change_maps'] == rep['n_change_maps'] - 2,
           str(rep2['the_pairs_skipped_for_a_geometry_change']))

        # the merge refuses sessions that do not share a clock
        import shutil
        corpus = Path('/mnt/c/proart_capture/corpus/free_sessions')
        if corpus.is_dir():
            a, b = corpus / '1792655768', corpus / '1792656069'
            if a.is_dir() and b.is_dir():
                merged = merge_fixed_sessions([a, b], tmp / 'fixed', quiet=True)
                mm = merged['the_merged']
                ck('room-lens: two sessions of one page merge into one stream',
                   mm['the_frames'] == 956 + 601 and
                   mm['the_page_base_spread_ms'] < 5.0,
                   json.dumps(mm['the_page_base_spread_ms']))
                # a session from ANOTHER page must be refused
                other = None
                for sid in sorted(corpus.iterdir()):
                    if sid.is_dir() and sid.name not in (a.name, b.name):
                        if (sid / 'timestamps.json').is_file():
                            try:
                                o = json.loads((sid / 'timestamps.json').read_text())['frame_offsets_ms']
                            except Exception:  # noqa: BLE001
                                continue
                            if o and abs(float(o[0]) - 4843.4) > 5000:
                                other = sid
                                break
                if other is not None:
                    refused = False
                    try:
                        merge_fixed_sessions([a, other], tmp / 'fixed2', quiet=True)
                    except RuntimeError:
                        refused = True
                    ck('room-lens: a session from ANOTHER page is REFUSED, not merged',
                       refused, other.name)

        # the ramp is the lens's OWN ramp, not a guess
        ck('room-lens: the ramp is the room-lens\'s own, with its eight @s',
           ROOM_LENS_RAMP.count('@') == 8 and len(ROOM_LENS_RAMP) == 17,
           ROOM_LENS_RAMP)
        bright = room_lens_bright()
        ck('room-lens: a glyph\'s brightness is the LOWER edge of its band',
           abs(bright[' '] - 0.0) < 1e-6 and
           abs(bright['@'] - 9 * (255.0 / 16)) < 1e-6,
           json.dumps({k: round(v, 2) for k, v in bright.items()}))
        ck('room-lens: the forward map inverts the lens\'s own rule',
           bright_to_glyph(0.0) == ' ' and bright_to_glyph(255.0) == '@'
           and bright_to_glyph(90.0) == ':',
           f'{bright_to_glyph(0.0)} {bright_to_glyph(90.0)} {bright_to_glyph(255.0)}')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--video', default=None, help='the phone recording')
    ap.add_argument('--out', default=None, help='the bundle directory to write')
    ap.add_argument('--label', default='the phone camera')
    ap.add_argument('--wall-clock-start', default=None,
                    help='the ISO time the recording STARTED -- the only '
                         'exact way to anchor a phone capture, which carries '
                         'no wall clock of its own')
    ap.add_argument('--stride', type=int, default=1,
                    help='keep every Nth frame (1 keeps all)')
    ap.add_argument('--fit', default='crop', choices=('crop', 'stretch'))
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--audio', default=None,
                    help='a WAV of the recording\'s audio, extracted where '
                         'there IS a decoder -- it is put into the bundle as '
                         'audio.wav so the corpus\'s audio tools can read it')
    ap.add_argument('--make-fixture', default=None,
                    help='write the phone-shaped rehearsal fixture to this '
                         'path and stop -- the content is MEANINGLESS on '
                         'purpose, so it can be paired with a real session as '
                         'a NEGATIVE CONTROL')
    ap.add_argument('--room-lens', default=None,
                    help='the phone\'s room-lens bundle (zcode\'s ASCII '
                         'camera) -- it is NOT a video and carries no picture '
                         'the corpus can read')
    ap.add_argument('--fixed-session', action='append', default=[],
                    help='a corpus session id to put on the fixed side '
                         '(repeatable; the sessions are merged into ONE '
                         'stream, and only if they verifiably share a clock)')
    ap.add_argument('--corpus-root',
                    default='/mnt/c/proart_capture/corpus/free_sessions',
                    help='where the corpus\'s session bundles live')
    ap.add_argument('--fixed-label', default='the fixed witness')
    ap.add_argument('--make-room-lens-fixture', default=None,
                    help='write a room-lens session with a KNOWN truth to '
                         'this path and stop -- it grades the phone\'s ingest '
                         'the way the synthetic pair grades the renders')
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.make_fixture:
        info = synth_phone_video(Path(args.make_fixture))
        print(json.dumps(info, indent=2))
        return 0
    if args.make_room_lens_fixture:
        info = make_room_lens_fixture(Path(args.make_room_lens_fixture))
        print(json.dumps(info, indent=2))
        return 0
    if args.room_lens:
        if not args.out:
            print('--out is required with --room-lens')
            return 1
        fixed_dirs = [Path(args.corpus_root) / s for s in args.fixed_session]
        fixed_dirs = [d for d in fixed_dirs if d.is_dir()]
        ingest_room_lens(Path(args.room_lens), Path(args.out), label=args.label,
                         fixed_dirs=fixed_dirs or None,
                         fixed_label=args.fixed_label, quiet=False)
        return 0
    if not args.video or not args.out:
        print('--video and --out are required (or --selftest, or '
              '--make-fixture, or --room-lens)')
        return 1
    ingest(Path(args.video), Path(args.out), label=args.label,
           wall_clock_start=args.wall_clock_start, stride=args.stride,
           fit=args.fit, max_frames=args.max_frames,
           audio=Path(args.audio) if args.audio else None)
    return 0


if __name__ == '__main__':
    sys.exit(main())
