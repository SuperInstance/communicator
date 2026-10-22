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
GRID_W, GRID_H = 40, 24


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

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"   [{detail}]" if detail and not ok else ""))
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


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
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.make_fixture:
        info = synth_phone_video(Path(args.make_fixture))
        print(json.dumps(info, indent=2))
        return 0
    if not args.video or not args.out:
        print('--video and --out are required (or --selftest, or '
              '--make-fixture)')
        return 1
    ingest(Path(args.video), Path(args.out), label=args.label,
           wall_clock_start=args.wall_clock_start, stride=args.stride,
           fit=args.fit, max_frames=args.max_frames,
           audio=Path(args.audio) if args.audio else None)
    return 0


if __name__ == '__main__':
    sys.exit(main())
