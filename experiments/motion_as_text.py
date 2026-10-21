"""Motion as text: a session becomes a document an LLM can read.

Casey's bridge from the communicator to syzygy. The idea is that video is
heavy and edges are cheap: instead of saving a wall of video, save a few
REFERENCE FRAMES to anchor the picture and, between them, a TEXT STREAM
of where the edges moved and how much. A text stream runs at frame rates
that shipping pixels could never afford, and an LLM can read it.

This tool does the whole thing on tonight's real free session, and it
measures what it cost and what it lost:

    REFERENCE FRAMES   one per second, plus one whenever the picture has
                       moved too far from the current reference to be
                       described by it
    THE DELTA STREAM  the signed difference between each frame and its
                       reference, on a coarse grid, one character per
                       cell -- this one RECONSTRUCTS
    THE MOTION STREAM the per-frame motion energy, the repository's own
                       change maps, as sparse events -- this one says
                       where the edges moved without saying where the
                       picture went
    THE MEASUREMENT   bytes per second against the raw stream, the
                       reconstruction error per frame, and where it
                       breaks
    THE RENDERING     one real gesture as ASCII, because text is the
                       storage AND the rendering

The repository's parts are used, not re-implemented: src/nv12.py for the
luma, experiments/corpus_capture_win.py for the change maps, and the
speech-to-score tool's session clock for the frame indices and the wall
time. Every reference and every event carries the session's own clock,
so the stream sits in the SAME alignment table as the transcript words
and the MIDI notes.

    python experiments/motion_as_text.py <session dir> --out /tmp/motion
    python experiments/motion_as_text.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import speech_to_score as s2s  # noqa: E402

try:
    import corpus_capture_win as ccw  # noqa: E402
except Exception:  # noqa: BLE001
    ccw = None

#: The signed ramp. Index i is the value i - 7, so index 7 is '0' and
#: index 14 is '7'. The INDEX is monotonic in the value; the characters
#: are not in ASCII order, and they are chosen to be legible instead:
#: the digits are the positive side, the punctuation is the negative
#: side, and '0' is a cell that did not change.
RAMP = "-=+*#%@0123456789"
RAMP_ZERO = 7
RAMP_MIN, RAMP_MAX = -7, 7

#: The luminance ramp for the ASCII rendering.
LUMA_RAMP = " .:-=+*#%@"


# ============================================================================
# 1. THE SESSION
# ============================================================================

def load_session(session_dir: Path) -> dict:
    """The session's own facts: the clock, the metadata and the luma."""
    session_dir = Path(session_dir)
    meta_path = session_dir / "meta.json"
    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:  # noqa: BLE001
            meta = {}

    clock = s2s.load_clock(session_dir)

    luma_meta = meta.get("luma") or {}
    height = int(luma_meta.get("height", 96))
    width = int(luma_meta.get("width", 160))

    raw_path = None
    for cand in ("luma.raw", "luma.nv12"):
        p = session_dir / cand
        if p.is_file():
            raw_path = p
            break
    if raw_path is None:
        raise SystemExit(f"no luma.raw in {session_dir}")

    data = np.fromfile(raw_path, dtype=np.uint8)
    stride = width * height
    if len(data) % stride:
        raise SystemExit(f"luma.raw is {len(data)} bytes, which is not a whole "
                         f"number of {width}x{height} frames")
    frames = data.reshape(-1, height, width)
    n = int(meta.get("n_frames") or len(frames))
    frames = frames[:n]
    return {"dir": session_dir, "meta": meta, "clock": clock,
            "frames": frames, "width": width, "height": height,
            "luma_path": raw_path}


def wall_time(clock, frame_index: int) -> str:
    """The session's wall clock, for a frame.

    `timestamps.json` carries the offsets from the pre-roll base and
    `meta.json` carries the base itself, so the two together give the
    absolute time. The frame index is the primary key and the wall time
    is the human-readable face of it.
    """
    if not clock.has_frames or frame_index >= len(clock.frame_offsets_s):
        return ""
    offset_s = float(clock.frame_offsets_s[frame_index])
    start = clock.wall_clock_start
    if start:
        try:
            import datetime
            base = datetime.datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ") \
                if "." in start else \
                datetime.datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ")
            t = base + datetime.timedelta(seconds=offset_s)
            return t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        except Exception:  # noqa: BLE001
            pass
    return f"{offset_s:.4f}s"


# ============================================================================
# 2. THE GRID
# ============================================================================

def _blocks(frames: np.ndarray, gw: int, gh: int):
    """The frames split into gw x gh blocks, padded at the edge when the
    frame does not divide evenly.

    A grid of 16x10 over a 96-row frame is 10 rows of 9 plus 6 rows left
    over, and dropping them would quietly lose the bottom of the picture.
    The remainder is padded with the edge rows instead.
    """
    n, h, w = frames.shape
    bh = -(-h // gh)
    bw = -(-w // gw)
    if gh * bh != h or gw * bw != w:
        frames = np.pad(frames, ((0, 0), (0, gh * bh - h), (0, gw * bw - w)),
                        mode="edge")
    return frames.reshape(n, gh, bh, gw, bw)


def block_mean(frames: np.ndarray, gw: int, gh: int) -> np.ndarray:
    """The frames on a coarse grid, each cell the mean of its block."""
    return _blocks(frames, gw, gh).mean(axis=(2, 4))


def delta_grid(frame: np.ndarray, reference: np.ndarray, gw: int,
               gh: int) -> np.ndarray:
    """The signed difference between a frame and its reference, per cell.

    This is the quantity that RECONSTRUCTS: adding it back to the
    reference's cells gives the frame's cells back, as well as a coarse
    grid can. The motion energy does not have that property, which is
    the difference between the two streams this tool writes.
    """
    a = _blocks(frame[None], gw, gh).mean(axis=(2, 4))[0]
    b = _blocks(reference[None], gw, gh).mean(axis=(2, 4))[0]
    return a - b


def encode_grid(deltas: np.ndarray, scale: float) -> str:
    """The delta grid as one line of text.

    Each cell is one character from the signed ramp, and the rows are
    separated by '/'. One line per frame, so a reader can take a line
    at a time and a grep can find a moment.
    """
    q = np.clip(np.round(deltas / scale), RAMP_MIN, RAMP_MAX).astype(int)
    rows = []
    for r in range(q.shape[0]):
        rows.append("".join(RAMP[v + RAMP_ZERO] for v in q[r]))
    return "/".join(rows)


def decode_grid(text: str, gh: int, gw: int, scale: float) -> np.ndarray:
    """The inverse of encode_grid, so the reconstruction is proven rather
    than asserted."""
    rows = text.split("/")
    out = np.zeros((gh, gw), dtype=np.float64)
    for r, row in enumerate(rows[:gh]):
        for c, ch in enumerate(row[:gw]):
            out[r][c] = (RAMP.index(ch) - RAMP_ZERO) * scale
    return out


def upsample(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    """A coarse grid back to the frame's size, by block repetition.

    The block sizes are the same ones the grid was made with, so the
    picture comes back at the size it went in at.
    """
    gh, gw = grid.shape
    bh = -(-h // gh)
    bw = -(-w // gw)
    big = np.repeat(np.repeat(grid, bh, axis=0), bw, axis=1)
    return big[:h, :w]


# ============================================================================
# 3. THE MOTION STREAM -- sparse events
# ============================================================================

def motion_maps(frames: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """The repository's own change maps: per-frame |frame(t+1) - frame(t)|,
    downsampled 2x2. Reused rather than reimplemented, because this is
    the machinery the corpus takes are built on.

    The session is walked in chunks, because the whole float32 copy of a
    four-minute session is nearly a gigabyte and the change maps are
    another one.
    """
    out = []
    for a in range(0, len(frames) - 1, chunk):
        block = frames[a:a + chunk + 1]
        if ccw is not None:
            out.append(ccw.change_maps(block.astype(np.float32)))
        else:
            d = np.abs(np.diff(block.astype(np.int16), axis=0)).astype(np.float32)
            out.append(d.reshape(d.shape[0], d.shape[1] // 2, 2,
                                 d.shape[2] // 2, 2).mean(axis=(2, 4)))
    if not out:
        return np.zeros((0, 1, 1), dtype=np.float32)
    return np.concatenate(out, axis=0)


def encode_sparse(motion: np.ndarray, threshold: float, limit: int = 24) -> str:
    """The motion, as sparse events: where the edges moved and how much.

    Only the cells whose motion clears the threshold are named, and only
    the strongest of them, so the line stays short. This is the stream
    that answers "where did it move"; it does not answer "where did the
    picture go", and the report says so.
    """
    gy, gx = motion.shape
    ys, xs = np.nonzero(motion >= threshold)
    if len(xs) == 0:
        return ""
    mags = motion[ys, xs]
    order = np.argsort(mags)[::-1][:limit]
    parts = []
    for i in order:
        parts.append(f"{int(xs[i])},{int(ys[i])},{int(round(float(mags[i])))}")
    return " ".join(parts)


# ============================================================================
# 4. THE REFERENCES
# ============================================================================

def reference_indices(frames: np.ndarray, clock, interval_s: float,
                      scene_threshold: float) -> tuple[list, dict]:
    """Which frames are the references.

    One every `interval_s` seconds, plus one whenever the picture has
    moved too far from the current reference to be described by it. The
    scene threshold is on the mean absolute difference from the current
    reference, so it is a statement about the picture and not about a
    particular pixel.
    """
    n = len(frames)
    if n == 0:
        return [], {"interval_s": interval_s, "scene_threshold": scene_threshold}

    offsets = (clock.frame_offsets_s if clock.has_frames
               else np.arange(n) / 60.0)
    refs = [0]
    last_ref = 0
    cur = frames[0].astype(np.int16)
    scene_hits = 0
    for t in range(1, n):
        d = float(np.abs(frames[t].astype(np.int16) - cur).mean())
        due = (offsets[t] - offsets[last_ref]) >= interval_s
        if due:
            refs.append(t)
            last_ref = t
            cur = frames[t].astype(np.int16)
        elif d >= scene_threshold:
            refs.append(t)
            last_ref = t
            cur = frames[t].astype(np.int16)
            scene_hits += 1
    return refs, {"interval_s": interval_s, "scene_threshold": scene_threshold,
                  "n_references": len(refs), "n_from_scene_change": scene_hits,
                  "n_from_interval": len(refs) - scene_hits}


def write_reference_png(frame: np.ndarray, path: Path) -> int:
    """A reference frame as a PNG, because a reference that costs more
    than the pixels it replaces has missed the point."""
    try:
        from PIL import Image
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame, mode="L").save(path, optimize=True)
        return path.stat().st_size
    except Exception:  # noqa: BLE001
        # no PIL: a PNG is written by hand, grayscale, no filter
        path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape
        raw = b"".join(b"\x00" + frame[r].tobytes() for r in range(h))

        def chunk(tag: bytes, data: bytes) -> bytes:
            body = tag + data
            return (len(data).to_bytes(4, "big") + body
                    + zlib.crc32(body).to_bytes(4, "big"))

        ihdr = (w.to_bytes(4, "big") + h.to_bytes(4, "big")
                + bytes([8, 0, 0, 0, 0]))
        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", ihdr)
        png += chunk(b"IDAT", zlib.compress(raw, 9))
        png += chunk(b"IEND", b"")
        path.write_bytes(png)
        return len(png)


# ============================================================================
# 5. THE RENDERING
# ============================================================================

def ascii_render(frame: np.ndarray, cols: int = 80, background=None,
                 lo: float | None = None, hi: float | None = None) -> list[str]:
    """A frame as text.

    With a background, the background is subtracted first, so the person
    stands out of the room instead of drowning in it. That is a
    rendering choice and not a measurement, and the report says which
    one it used.
    """
    h, w = frame.shape
    rows = h // (w // cols)
    block = w // cols
    small = frame[:rows * (h // rows)].reshape(
        rows, h // rows, cols, block).mean(axis=(1, 3))
    if background is not None:
        bh, bw = background.shape
        bsmall = background[:rows * (h // rows), :cols * block].reshape(
            rows, h // rows, cols, block).mean(axis=(1, 3))
        small = small - bsmall
    a = lo if lo is not None else float(small.min())
    b = hi if hi is not None else float(small.max())
    if b - a < 1e-6:
        return ["".join(LUMA_RAMP[0] * cols) for _ in range(rows)]
    q = np.clip(((small - a) / (b - a)) * (len(LUMA_RAMP) - 1), 0,
                len(LUMA_RAMP) - 1).astype(int)
    return ["".join(LUMA_RAMP[v] for v in row) for row in q]


def render_delta_text(line: str) -> list[str]:
    """A line of the delta stream, rendered back as a picture.

    This is the syzygy payoff: the line of text IS the picture. The
    characters in the file are the characters in the rendering, and the
    zero cells are the ones that did not move.
    """
    return line.split("/")


def background_of(frames: np.ndarray, every: int = 40) -> np.ndarray:
    """The median of a subsample of the session: the room, without the
    person in it."""
    sub = frames[::every].astype(np.float32)
    return np.median(sub, axis=0)


# ============================================================================
# 6. THE DRIVER
# ============================================================================

def run(session_dir: Path, out_dir: Path, interval_s: float = 1.0,
        scene_threshold: float = 6.0, gw: int = 20, gh: int = 12,
        sparse_threshold: float = 4.0, ascii_at: float | None = None,
        ascii_frames: int = 3, ascii_cols: int = 80) -> dict:
    """The whole thing, on one session."""
    t0 = time.time()
    sess = load_session(session_dir)
    frames = sess["frames"]
    clock = sess["clock"]
    n = len(frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = out_dir / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)

    print(f"session   {session_dir}")
    print(f"luma      {n} frames of {sess['width']}x{sess['height']}, "
          f"{n * sess['width'] * sess['height'] / 1e6:.1f} MB")
    if clock.has_frames:
        print(f"clock     {clock.n_frames} frames, "
              f"{clock.describe()['median_fps']:.3f} fps, "
              f"{clock.describe()['first_frame_s']:.2f}.."
              f"{clock.describe()['last_frame_s']:.2f} s")

    # 1. the references
    refs, ref_info = reference_indices(frames, clock, interval_s, scene_threshold)
    print(f"refs      {len(refs)} references "
          f"({ref_info['n_from_interval']} on the clock, "
          f"{ref_info['n_from_scene_change']} on a scene change)")

    ref_bytes = 0
    ref_index = []
    for i, t in enumerate(refs):
        p = ref_dir / f"ref_{i:05d}_f{t:06d}.png"
        ref_bytes += write_reference_png(frames[t], p)
        ref_index.append({"ref_index": i, "frame": t,
                           "t_s": (float(clock.frame_offsets_s[t])
                                   if clock.has_frames else t / 60.0),
                           "wall_time": wall_time(clock, t),
                           "file": p.name, "bytes": p.stat().st_size})
    print(f"          {ref_bytes / 1e6:.2f} MB of references")

    # 2. the scale of the ramp, from the data and not from taste
    sample = frames[:: max(1, n // 400)]
    ref0 = refs[0]
    d0 = block_mean(sample, gw, gh) - block_mean(frames[ref0][None], gw, gh)
    scale = max(1.0, float(np.percentile(np.abs(d0), 90)))
    print(f"scale     1 ramp unit = {scale:.2f} luma levels")

    # 3. the two streams
    delta_path = out_dir / "delta_stream.txt"
    sparse_path = out_dir / "motion_stream.txt"
    motion = motion_maps(frames) if n > 1 else np.zeros((0, 1, 1))
    gy, gx = motion.shape[1], motion.shape[2]
    if gx % gw or gy % gh:
        gx = (gx // gw) * gw
        gy = (gy // gh) * gh

    ref_i = 0
    ref_frame = frames[refs[0]]
    errors = []
    base_errors = []
    textures = []
    delta_bytes = 0
    sparse_bytes = 0
    delta_lines: dict = {}
    with delta_path.open("w", encoding="utf-8") as fd, \
            sparse_path.open("w", encoding="utf-8") as fs:
        fd.write("# MOTION AS TEXT -- the delta stream\n")
        fd.write(f"# session {session_dir}\n")
        fd.write(f"# grid {gw}x{gh} cells, each {sess['width'] // gw}x"
                 f"{sess['height'] // gh} luma pixels\n")
        fd.write(f"# scale 1 unit = {scale:.4f} luma levels, "
                 f"values {RAMP_MIN}..{RAMP_MAX}\n")
        fd.write(f"# ramp index i is value i-{RAMP_ZERO}; "
                 f"'{RAMP[RAMP_ZERO]}' is zero\n")
        fd.write("# format: F <frame> <seconds> <wall time> <ref frame> "
                 "<rows separated by />\n")
        fs.write("# MOTION AS TEXT -- the motion stream (sparse events)\n")
        fs.write(f"# session {session_dir}\n")
        fs.write(f"# grid {gx}x{gy} cells of the repository's own change maps\n")
        fs.write(f"# threshold {sparse_threshold}\n")
        fs.write("# format: E <frame> <seconds> <wall time> "
                 "<x,y,magnitude> ...\n")
        for t in range(n):
            while ref_i + 1 < len(refs) and refs[ref_i + 1] <= t:
                ref_i += 1
                ref_frame = frames[refs[ref_i]]
            dg = delta_grid(frames[t].astype(np.float64), ref_frame.astype(np.float64),
                            gw, gh)
            line = encode_grid(dg, scale)
            delta_lines[t] = line
            ts = (float(clock.frame_offsets_s[t]) if clock.has_frames
                  else t / 60.0)
            fd.write(f"F {t:06d} {ts:9.4f} {wall_time(clock, t):26s} "
                     f"{refs[ref_i]:06d} {line}\n")
            delta_bytes += len(line) + 1
            if t < len(motion):
                ev = encode_sparse(motion[t][:gy, :gx], sparse_threshold)
                if ev:
                    fs.write(f"E {t:06d} {ts:9.4f} {wall_time(clock, t):26s} {ev}\n")
                    sparse_bytes += len(ev) + 1

            # the reconstruction error, measured at FULL RESOLUTION.
            # Comparing the grid against the grid would be trivially zero,
            # because the grid IS the reference plus the delta. What the
            # reader actually gets back is the block-upsampled grid, and
            # the error is the detail inside the blocks that the grid
            # threw away.
            if t % 7 == 0:
                truth = frames[t].astype(np.float64)
                rec = upsample(block_mean(ref_frame[None].astype(np.float64),
                                          gw, gh)[0] + dg, sess["height"],
                              sess["width"])[:sess["height"], :sess["width"]]
                base = upsample(block_mean(ref_frame[None].astype(np.float64),
                                           gw, gh)[0], sess["height"],
                                sess["width"])[:sess["height"], :sess["width"]]
                errors.append((t, float(np.abs(rec - truth).mean())))
                base_errors.append((t, float(np.abs(base - truth).mean())))
                # the TEXTURE: the detail inside the blocks, which is the
                # thing the grid throws away. The reconstruction error
                # should track this and not the motion, and saying so is
                # the difference between "it breaks here" and "it breaks
                # HERE, for this reason".
                blocks = _blocks(truth[None], gw, gh)[0]
                textures.append((t, float(blocks.var(axis=(1, 2)).mean())))
    print(f"streams   delta {delta_bytes / 1e6:.2f} MB, "
          f"sparse {sparse_bytes / 1e6:.2f} MB")

    # 4. the measurement
    span = (float(clock.frame_offsets_s[-1] - clock.frame_offsets_s[0])
            if clock.has_frames and n > 1 else n / 60.0)
    raw_bytes = n * sess["width"] * sess["height"]
    err = np.array([e[1] for e in errors])
    bas = np.array([e[1] for e in base_errors])
    motion_series = (motion.mean(axis=(1, 2)) if len(motion) else np.zeros(1))

    # where does it break: the frames with the largest error, and what
    # the motion was doing there
    tex = np.array([x[1] for x in textures])
    order = np.argsort(err)[::-1][:20]
    worst = []
    for i in order:
        t = errors[i][0]
        m = float(motion_series[min(t, len(motion_series) - 1)])
        worst.append({"frame": t, "error": round(errors[i][1], 4),
                      "baseline_error": round(base_errors[i][1], 4),
                      "motion_mean": round(m, 3),
                      "texture_within_blocks": round(float(tex[i]), 3),
                      "t_s": round(float(clock.frame_offsets_s[t])
                                  if clock.has_frames else t / 60.0, 3)})
    if len(err) > 2 and float(err.std()) > 1e-9 and float(bas.std()) > 1e-9:
        corr = float(np.corrcoef(err, bas)[0, 1])
    else:
        corr = 0.0
    if len(err) > 2 and float(err.std()) > 1e-9 and float(tex.std()) > 1e-9:
        corr_tex = float(np.corrcoef(err, tex)[0, 1])
    else:
        corr_tex = 0.0
    # and the same for the motion, which is the thing one would expect
    if len(err) > 2:
        mv = np.array([motion_series[min(errors[i][0], len(motion_series) - 1)]
                       for i in range(len(errors))])
        if float(err.std()) > 1e-9 and float(mv.std()) > 1e-9:
            corr_motion = float(np.corrcoef(err, mv)[0, 1])
        else:
            corr_motion = 0.0
    else:
        corr_motion = 0.0

    # 5. the rendering
    renders = []
    if ascii_at is not None:
        target = ascii_at
        if clock.has_frames:
            idx = int(np.argmin(np.abs(clock.frame_offsets_s - target)))
        else:
            idx = int(target * 60)
        bg = background_of(frames)
        picks = [idx + k for k in range(ascii_frames)
                 if 0 <= idx + k < n]
        for p in picks:
            ts = (float(clock.frame_offsets_s[p]) if clock.has_frames
                  else p / 60.0)
            art = ascii_render(frames[p], cols=ascii_cols, background=bg,
                               lo=-4, hi=20)
            out = out_dir / f"ascii_{p:06d}.txt"
            out.write_text("\n".join(art) + "\n")
            # and the SAME moment rendered out of the text stream itself
            stream_art = render_delta_text(delta_lines.get(p, ""))
            sout = out_dir / f"stream_{p:06d}.txt"
            sout.write_text("\n".join(stream_art) + "\n")
            renders.append({"frame": p, "t_s": round(ts, 3),
                            "wall_time": wall_time(clock, p),
                            "file": out.name,
                            "stream_file": sout.name,
                            "stream_text": delta_lines.get(p, ""),
                            "motion_mean": round(float(
                                motion_series[min(p, len(motion_series) - 1)]), 3)})
        print(f"ascii     {len(renders)} renderings around {target:.1f}s")

    # the grid sweep: what each resolution costs and what it loses
    sweep = grid_sweep(frames, clock, interval_s,
                       [(8, 6), (10, 6), (16, 10), (20, 12), (40, 24), (80, 48)])
    for row in sweep:
        print(f"grid {row['grid']:>6s}  {row['bytes_per_second_at_60fps']:>6.1f} kB/s"
              f"  error {row['error_mean_abs']:>6.3f}"
              f"  (reference alone {row['reference_only_error_mean_abs']:>6.3f},"
              f" the motion is worth {row['improvement'] * 100:.0f}%)")

    results = {
        "tool": "experiments/motion_as_text.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session": str(session_dir),
        "n_frames": n,
        "width": sess["width"], "height": sess["height"],
        "clock": clock.describe(),
        "references": ref_info,
        "reference_index": ref_index,
        "grid": {"gw": gw, "gh": gh,
                 "cell_px": [sess["width"] // gw, sess["height"] // gh]},
        "scale_luma_per_unit": round(scale, 4),
        "ramp": RAMP,
        "bytes": {
            "raw_luma": raw_bytes,
            "reference_frames": ref_bytes,
            "delta_stream_text": delta_bytes,
            "motion_stream_text": sparse_bytes,
            "total": raw_bytes and (ref_bytes + delta_bytes + sparse_bytes),
        },
        "bytes_per_second": {
            "raw_luma": round(raw_bytes / span, 1),
            "reference_frames": round(ref_bytes / span, 1),
            "delta_stream_text": round(delta_bytes / span, 1),
            "motion_stream_text": round(sparse_bytes / span, 1),
            "total": round((ref_bytes + delta_bytes + sparse_bytes) / span, 1),
        },
        "span_s": round(span, 3),
        "reconstruction": {
            "what": "the reference's cells plus the stream's signed delta, "
                    "against the truth's cells on the same grid",
            "error_mean_abs": round(float(err.mean()), 4),
            "error_p95": round(float(np.percentile(err, 95)), 4),
            "error_max": round(float(err.max()), 4),
            "reference_only_error_mean_abs": round(float(bas.mean()), 4),
            "improvement_over_reference_only": round(
                float(1.0 - err.mean() / max(1e-9, bas.mean())), 4),
            "error_vs_reference_error_correlation": round(corr, 4),
            "error_vs_texture_correlation": round(corr_tex, 4),
            "error_vs_motion_correlation": round(corr_motion, 4),
            "n_sampled_frames": len(err),
        },
        "grid_sweep": sweep,
        "where_it_breaks": worst,
        "ascii_renders": renders,
        "elapsed_s": round(time.time() - t0, 2),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print()
    print(f"raw           {results['bytes_per_second']['raw_luma'] / 1000:.1f} kB/s")
    print(f"references    {results['bytes_per_second']['reference_frames'] / 1000:.1f} kB/s")
    print(f"delta text    {results['bytes_per_second']['delta_stream_text'] / 1000:.1f} kB/s")
    print(f"motion text   {results['bytes_per_second']['motion_stream_text'] / 1000:.1f} kB/s")
    print(f"total         {results['bytes_per_second']['total'] / 1000:.1f} kB/s "
          f"({raw_bytes / max(1, ref_bytes + delta_bytes + sparse_bytes):.1f}x smaller)")
    print(f"error         {results['reconstruction']['error_mean_abs']} "
          f"(reference alone: {results['reconstruction']['reference_only_error_mean_abs']})")
    print(f"results       {out_dir / 'results.json'}")
    return results


# ============================================================================
# 7. THE SELFTEST
# ============================================================================

def selftest() -> int:
    """The checks that need no session."""
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  [{'ok' if got == want else 'FAIL'}] {name}: {got!r}")

    print("SELFTEST -- the grid, the ramp, the reconstruction")
    print("-" * 60)

    # 1. THE GRID. A block is the mean of its pixels.
    f = np.arange(96 * 160, dtype=np.float64).reshape(96, 160)
    g = block_mean(f[None], 4, 3)
    check("the grid has the right shape", g.shape, (1, 3, 4))
    check("a cell is the mean of its block", round(float(g[0][0][0]), 4),
          round(float(f[0:32, 0:40].mean()), 4))

    # 2. THE RAMP. Zero is '0', the sign is in the order.
    check("the zero of the ramp", RAMP[RAMP_ZERO], "0")
    check("the ramp reads: digits up, punctuation down",
          RAMP[RAMP_ZERO + 1].isdigit() and not RAMP[RAMP_ZERO - 1].isdigit(),
          True)
    d = np.array([[-7.0, -1.0, 0.0, 1.0, 7.0]])
    enc = encode_grid(d, 1.0)
    check("a row of cells encodes", enc, "-@01 7".replace(" ", ""))
    dec = decode_grid(enc, 1, 5, 1.0)
    check("and decodes back", [round(v, 3) for v in dec[0]],
          [-7.0, -1.0, 0.0, 1.0, 7.0])

    # 3. THE RECONSTRUCTION. A frame and its own reference reconstruct
    # exactly; a frame and a DIFFERENT reference do not.
    ref = np.zeros((96, 160), dtype=np.float64)
    frame = ref.copy()
    frame[40:60, 60:90] = 30.0
    dg = delta_grid(frame, ref, 20, 12)
    rec = upsample(block_mean(ref[None], 20, 12)[0] + dg, 96, 160)
    truth = block_mean(frame[None], 20, 12)[0]
    check("the delta reconstructs the grid",
          round(float(np.abs(rec[40:60, 60:90].mean() - truth.mean())), 6)
          if False else round(float(np.abs(
              (block_mean(ref[None], 20, 12)[0] + dg) - truth).max()), 6), 0.0)
    check("a frame equal to its reference needs no delta",
          float(np.abs(delta_grid(ref, ref, 20, 12)).max()), 0.0)

    # 4. THE MOTION STREAM. A cell that moved is named; one that did not
    # is not.
    m = np.zeros((4, 6), dtype=np.float32)
    m[1][2] = 9.0
    m[3][5] = 12.0
    ev = encode_sparse(m, 4.0)
    check("the moved cells are named", "2,1,9" in ev, True)
    check("and the still ones are not", "0,0" in ev, False)
    check("a still frame has no events", encode_sparse(np.zeros((4, 6)), 4.0), "")

    # 5. THE RENDERING. A bright block is bright in the text.
    art = ascii_render(np.full((96, 160), 255.0), cols=40, lo=0, hi=255)
    check("the rendering has rows", len(art), 24)
    check("a bright frame is bright", art[12][:20], "@" * 20)
    mid = ascii_render(np.full((96, 160), 128.0), cols=40, lo=0, hi=255)
    check("a middling frame is middling",
          mid[12][0] in LUMA_RAMP[3:6], True)
    art2 = ascii_render(np.zeros((96, 160)), cols=40, lo=0, hi=255)
    check("a dark frame is dark", art2[12][:20], " " * 20)

    # 6. THE CLOCK. The wall time comes from the frame index.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "sess"
        d.mkdir()
        offs = [round(i * 1000.0 / 60.0, 3) for i in range(120)]
        (d / "timestamps.json").write_text(json.dumps(
            {"wall_clock_start": "2026-01-01T00:00:00Z",
             "frame_offsets_ms": offs}))
        clock = s2s.load_clock(d)
        check("the wall time of frame 60", wall_time(clock, 60),
              "2026-01-01T00:00:01.000Z")

    print("-" * 60)
    if failures:
        print(f"SELFTEST FAILED: {len(failures)} check(s)")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SELFTEST PASSED")
    return 0


def grid_sweep(frames: np.ndarray, clock, ref_interval_s: float,
               grids: list, sample: int = 240) -> list:
    """What each grid size costs and what it loses.

    The same measurement at several resolutions: the bytes per frame the
    text stream would cost, the reconstruction error against the true
    frame, and the error of the reference alone with no motion at all.
    The third column is the one that matters, because it says what the
    motion stream is worth.
    """
    n = len(frames)
    if n < 4:
        return []
    step = max(1, n // sample)
    picks = list(range(0, n, step))[:sample]
    # the reference for each picked frame: the last one at the interval
    offsets = (clock.frame_offsets_s if clock.has_frames
               else np.arange(n) / 60.0)
    out = []
    for gw, gh in grids:
        errs = []
        bases = []
        for t in picks:
            # the reference: the most recent frame that is at least
            # `ref_interval_s` old
            ref = t
            while ref > 0 and (offsets[t] - offsets[ref]) < ref_interval_s:
                ref -= 1
            r = frames[ref].astype(np.float64)
            f = frames[t].astype(np.float64)
            dg = delta_grid(f, r, gw, gh)
            rec = upsample(block_mean(r[None], gw, gh)[0] + dg,
                           frames.shape[1], frames.shape[2])
            base = upsample(block_mean(r[None], gw, gh)[0],
                            frames.shape[1], frames.shape[2])
            errs.append(np.abs(rec - f).mean())
            bases.append(np.abs(base - f).mean())
        # the bytes one frame of text would cost
        chars = gw * gh + max(0, gh - 1)
        out.append({
            "grid": f"{gw}x{gh}",
            "cell_px": [frames.shape[2] // gw, frames.shape[1] // gh],
            "bytes_per_frame": chars,
            "bytes_per_second_at_60fps": round(chars * 60 / 1000, 2),
            "error_mean_abs": round(float(np.mean(errs)), 4),
            "reference_only_error_mean_abs": round(float(np.mean(bases)), 4),
            "improvement": round(float(1.0 - np.mean(errs) / max(1e-9, np.mean(bases))), 4),
            "n_sampled": len(picks),
        })
    return out


def rate_policy(session_dir: Path, out_dir: Path,
                live_grid=(20, 12), coarse_grid=(16, 10),
                fine_grid=(80, 48), interval_s: float = 1.0,
                scene_threshold: float = 6.0, sparse_threshold: float = 4.0,
                ref_png_bytes: int | None = None,
                still_threshold: float = 3.0) -> dict:
    """THE RATE POLICY, measured on one session.

    Casey's phone addendum: "adaptive fidelity (motion-indexed spend)".
    A STILL phone drops its frame rate and RAISES its resolution -- a
    static scene is described ONCE at high fidelity and the live channel
    then carries only changes.  A MOVING phone scales its resolution back
    to afford motion.  "The budget follows the change, never the clock."

    So the question this answers is concrete: the bytes-per-second number
    the stream costs today is a CLOCK-indexed rate, and the policy turns
    it into a CHANGE-indexed one.  What does that do to the number?

    The measurement, on the real session:
      * every frame is encoded at the live grid and compared with the
        previous frame's encoding: an identical encoding means the
        picture did not change, and under the policy that frame costs
        NOTHING in the live channel;
      * a maximal run of unchanged frames is a STILL STRETCH: it gets
        ONE description at the fine grid, not one per clock second;
      * a frame that did change is carried at the COARSE grid, because
        motion buys frame rate and resolution scales back;
      * the sparse motion events are carried only for frames that
        changed, for the same reason.

    The exchange rate is not invented here: it is the grid sweep's own
    table.  Resolution is expensive (80x48 is 73x the bytes of 8x6) and
    the fidelity it buys is comparatively cheap (the error falls by half
    while the bytes rise 15x), which is exactly why the policy spends
    resolution on STILLNESS and frame rate on MOTION.
    """
    t0 = time.time()
    sess = load_session(session_dir)
    frames = sess["frames"]
    clock = sess["clock"]
    n = len(frames)
    if n < 2:
        return {"ok": False, "why": "session too short"}

    lgw, lgh = live_grid
    cgw, cgh = coarse_grid
    fgw, fgh = fine_grid

    refs, ref_info = reference_indices(frames, clock, interval_s,
                                       scene_threshold)
    motion = motion_maps(frames)

    # the SAME scale the live stream uses, so the byte counts here are the
    # bytes the stream would really cost
    sample = frames[:: max(1, n // 400)]
    ref0 = refs[0]
    d0 = block_mean(sample, lgw, lgh) - block_mean(frames[ref0][None], lgw, lgh)
    scale = max(1.0, float(np.percentile(np.abs(d0), 90)))

    live_chars = lgw * lgh + max(0, lgh - 1)
    coarse_chars = cgw * cgh + max(0, cgh - 1)
    fine_chars = fgw * fgh + max(0, fgh - 1)

    changed = [False] * n
    delta_bytes_clock = 0
    delta_bytes_policy = 0
    sparse_bytes_clock = 0
    sparse_bytes_policy = 0
    ref_i = 0
    # STILLNESS IS A PROPERTY OF THE PICTURE, and it has to be measured
    # in the units the test uses.
    #
    # The first version compared the encoded delta line with the
    # previous frame's, and it was wrong twice over: the reference
    # advances on a clock, so every reference advance changed the line
    # even when the picture had not moved, and a BYTE comparison calls
    # the camera's own noise "movement".  The second version compared
    # the picture's grid for equality and was still wrong, because the
    # camera's noise makes every changed frame differ and 66.6 % of the
    # frames in this session are byte-identical repeats with the
    # remaining ones a continuum with no valley in it.
    #
    # So stillness is a MAGNITUDE test, and the threshold is measured:
    # the whole-frame consecutive change has a non-zero median of 1.152
    # luma (the camera's own change at a changed frame) and the room map
    # measured the per-pixel noise sigma at 0.61 luma.  A threshold of
    # `still_threshold` (default 3.0 luma, about 2.6x the non-zero
    # median and about 5x the noise sigma) puts the sensor's own noise
    # below the line and leaves the person above it.
    px0, px1, py0, py1 = 7, 119, 42, 95
    zone = frames[:, py0:py1, px0:px1].astype(np.float32)
    cons = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2))
    cons_zone = np.abs(np.diff(zone, axis=0)).mean(axis=(1, 2))
    nz = cons[cons > 0]
    noise = {
        "n_consecutive": int(cons.size),
        "n_zero": int((cons == 0).sum()),
        "zero_fraction": round(float((cons == 0).mean()), 4),
        "nonzero_median_luma": float(np.median(nz)) if nz.size else None,
        "nonzero_p90_luma": float(np.percentile(nz, 90)) if nz.size else None,
        "still_threshold_luma": still_threshold,
    }
    # The motion map and the whole-frame change DISAGREE, and the
    # disagreement is the finding: the motion map measures WHERE the
    # change is (edges) and the whole-frame mean measures HOW MUCH
    # changed.  A person crossing a corner of the frame moves edges
    # without moving the frame's mean, so a trigger on either measure
    # alone misses something the other catches.  The union is the
    # defensible "the picture changed" test.
    motion_fired = np.zeros(n, dtype=bool)
    for t in range(n):
        if t < len(motion):
            ev = encode_sparse(motion[t][:cgh, :cgw], sparse_threshold)
            motion_fired[t] = bool(ev)
    for t in range(n):
        # frame t's change is cons[t-1]
        changed[t] = (t == 0) or (cons[t - 1] > still_threshold)
    changed_motion_only = [bool(motion_fired[t]) for t in range(n)]
    changed_union = [bool(changed[t] or changed_motion_only[t]) for t in range(n)]
    zone_moving = int((cons_zone > still_threshold).sum())

    for t in range(n):
        while ref_i + 1 < len(refs) and refs[ref_i + 1] <= t:
            ref_i += 1
        ref_frame = frames[refs[ref_i]].astype(np.float64)
        dg = delta_grid(frames[t].astype(np.float64), ref_frame, lgw, lgh)
        line = encode_grid(dg, scale)
        delta_bytes_clock += len(line) + 1
        if changed[t]:
            # the coarse carry: the same change, at a resolution that
            # affords more of them
            dgc = delta_grid(frames[t].astype(np.float64), ref_frame, cgw, cgh)
            delta_bytes_policy += len(encode_grid(dgc, scale)) + 1
        if t < len(motion):
            ev = encode_sparse(motion[t][:cgh, :cgw], sparse_threshold)
            if ev:
                sparse_bytes_clock += len(ev) + 1
                if changed[t]:
                    sparse_bytes_policy += len(ev) + 1

    # the stretches
    stretches = []
    t = 0
    while t < n:
        j = t
        while j + 1 < n and changed[j + 1] == changed[t]:
            j += 1
        stretches.append({"start_frame": t, "end_frame": j,
                          "frames": j - t + 1, "kind": "MOVING" if changed[t] else "STILL"})
        t = j + 1
    still = [s for s in stretches if s["kind"] == "STILL"]
    moving = [s for s in stretches if s["kind"] == "MOVING"]

    offsets = (clock.frame_offsets_s if clock.has_frames
               else np.arange(n) / 60.0)
    span = float(offsets[-1] - offsets[0]) if n > 1 else n / 60.0

    frame_dt = (float(offsets[1] - offsets[0]) if n > 1 and clock.has_frames
                else 1.0 / 60.0)

    def dur(s):
        # a one-frame stretch has zero span between its first and last
        # frame's timestamps; it still occupies one frame's worth of time
        span = float(offsets[s["end_frame"]] - offsets[s["start_frame"]])
        return span + (frame_dt if s["frames"] == 1 else 0.0)

    still_s = sum(dur(s) for s in still)
    moving_s = sum(dur(s) for s in moving)

    # the reference channel under the policy: the clock cadence while the
    # phone is moving, and ONE high-fidelity description per still
    # stretch.  A PNG reference costs what it costs; the policy's still
    # description is a fine-grid ASCII frame.
    png_ref_bytes = ref_png_bytes
    policy_refs = len(moving) + len(still)
    policy_ref_bytes = 0
    reanchors = 0
    for s_ in moving:
        reanchors += max(0, int(dur(s_) // interval_s))
    policy_refs += reanchors
    # the descriptions are ASCII: a still stretch is described ONCE at the
    # fine grid, a moving stretch is anchored at the live grid and
    # re-anchored on the clock because its deltas accumulate
    policy_ref_bytes = (len(still) * fine_chars
                         + (len(moving) + reanchors) * live_chars)
    clock_ref_bytes = png_ref_bytes if png_ref_bytes is not None else 0

    out = {
        "tool": "experiments/motion_as_text.py --rate-policy",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session": str(session_dir),
        "n_frames": n,
        "span_s": round(span, 3),
        "policy": ("stillness buys resolution and drops the frame rate; "
                   "motion buys frame rate and drops the resolution; the "
                   "budget follows the change, never the clock"),
        "grids": {
            "live": {"gw": lgw, "gh": lgh, "chars_per_frame": live_chars},
            "coarse": {"gw": cgw, "gh": cgh, "chars_per_frame": coarse_chars},
            "fine": {"gw": fgw, "gh": fgh, "chars_per_frame": fine_chars},
        },
        "stillness_test": noise,
        "stretches": {
            "n": len(stretches),
            "person_zone_moving_frames": zone_moving,
            "n_still": len(still),
            "n_moving": len(moving),
            "still_s": round(still_s, 3),
            "moving_s": round(moving_s, 3),
            "still_fraction_of_time": round(still_s / span, 4) if span else None,
            "unchanged_frames": int(sum(1 for c in changed if not c)),
            "unchanged_fraction": round(sum(1 for c in changed if not c) / n, 4),
        },
        "clock_indexed": {
            "what": "today's design: one reference per clock second, every "
                    "frame's delta carried, every motion event carried",
            "n_references": len(refs),
            "reference_bytes": clock_ref_bytes,
            "delta_bytes": delta_bytes_clock,
            "sparse_bytes": sparse_bytes_clock,
            "total_bytes": clock_ref_bytes + delta_bytes_clock + sparse_bytes_clock,
            "bytes_per_second": round(
                (clock_ref_bytes + delta_bytes_clock + sparse_bytes_clock) / span, 1)
            if span else None,
        },
        "policy_indexed": {
            "what": "one high-fidelity description per still stretch; the "
                    "live channel carries only changed frames, at the coarse "
                    "grid",
            "n_descriptions": policy_refs,
            "reference_bytes": policy_ref_bytes,
            "delta_bytes": delta_bytes_policy,
            "sparse_bytes": sparse_bytes_policy,
            "total_bytes": policy_ref_bytes + delta_bytes_policy + sparse_bytes_policy,
            "bytes_per_second": round(
                (policy_ref_bytes + delta_bytes_policy + sparse_bytes_policy) / span, 1)
            if span else None,
        },
        "triggers_disagree": {
            "note": "the whole-frame change and the motion map disagree; the "
                    "union is the defensible 'the picture changed' test",
            "n_whole_frame_moving": int(sum(1 for c in changed if c)),
            "n_motion_map_firing": int(sum(1 for c in changed_motion_only if c)),
            "n_either": int(sum(1 for c in changed_union if c)),
            "n_motion_only": int(sum(1 for t in range(n)
                                     if changed_motion_only[t] and not changed[t])),
            "n_whole_only": int(sum(1 for t in range(n)
                                    if changed[t] and not changed_motion_only[t])),
        },
        "exchange_rate": {
            "note": "the policy's arithmetic is the grid sweep's own table",
            "fine_over_coarse_bytes": round(fine_chars / coarse_chars, 2),
            "live_over_coarse_bytes": round(live_chars / coarse_chars, 2),
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    if out["clock_indexed"]["bytes_per_second"] and \
            out["policy_indexed"]["bytes_per_second"]:
        out["policy_indexed"]["ratio_to_clock"] = round(
            out["policy_indexed"]["bytes_per_second"]
            / out["clock_indexed"]["bytes_per_second"], 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rate_policy.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", nargs="?", default=None,
                    help="the free-session folder")
    ap.add_argument("--out", default="/tmp/motion_as_text")
    ap.add_argument("--interval-s", type=float, default=1.0,
                    help="how often a reference frame is emitted")
    ap.add_argument("--scene-threshold", type=float, default=6.0,
                    help="the mean absolute difference from the current "
                         "reference that forces a new one")
    ap.add_argument("--grid", default="20x12",
                    help="the text grid, GWxGH")
    ap.add_argument("--sparse-threshold", type=float, default=4.0)
    ap.add_argument("--ascii-at", type=float, default=None,
                    help="the second to render in ASCII")
    ap.add_argument("--ascii-frames", type=int, default=3)
    ap.add_argument("--ascii-cols", type=int, default=80)
    ap.add_argument("--rate-policy", action="store_true",
                    help="measure the adaptive-fidelity policy instead of "
                         "rendering the session")
    ap.add_argument("--still-threshold", type=float, default=3.0,
                    help="the loudness of a frame-to-frame change, in luma, "
                         "above which the picture counts as MOVING")
    ap.add_argument("--ref-png-bytes", type=int, default=None,
                    help="the measured total bytes of this session's PNG "
                         "references, so the policy's still descriptions can "
                         "be compared with them")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if not a.session:
        ap.error("give a session folder, or --selftest")
    gw, gh = (int(v) for v in a.grid.lower().split("x"))
    if a.rate_policy:
        out = rate_policy(Path(a.session), Path(a.out), live_grid=(gw, gh),
                          interval_s=a.interval_s,
                          scene_threshold=a.scene_threshold,
                          sparse_threshold=a.sparse_threshold,
                          ref_png_bytes=a.ref_png_bytes,
                          still_threshold=a.still_threshold)
        print(json.dumps({k: v for k, v in out.items()
                          if k in ("stretches", "clock_indexed",
                                   "policy_indexed", "exchange_rate")}, indent=2))
        return 0
    run(Path(a.session), Path(a.out), interval_s=a.interval_s,
        scene_threshold=a.scene_threshold, gw=gw, gh=gh,
        sparse_threshold=a.sparse_threshold, ascii_at=a.ascii_at,
        ascii_frames=a.ascii_frames, ascii_cols=a.ascii_cols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
