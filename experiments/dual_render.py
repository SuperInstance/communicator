#!/usr/bin/env python3
"""DUAL RENDER -- two views of the same minutes, rendered three ways.

    python experiments/dual_render.py --selftest
    python experiments/dual_render.py --check --fixed <dir> --roving <dir>
    python experiments/dual_render.py --fixed <dir> --roving <dir> \\
        --sync <claps.json> --out docs/research/dual

WHAT THIS IS.  Casey's brief (docs/research/BRIEF-DUAL-CAPTURE.md, the
evening of 2026-10-21) asks for two cameras in one room: the ProArt
webcam+mic as the FIXED WITNESS and the phone camera+mic as the ROVING
EYE.  zcode owns the capture's liveness and the clap alignment; this
tool is the RENDERING LANE, and it is staged before the pair lands so
it can run the moment the pair does.

THE THREE RENDERS, because the brief asks for all three and keeps all:

  1. THE DUAL-ASCII, SIDE BY SIDE.  The same MOMENTS from both views,
     one under the other, side by side in the terminal.
  2. THE FUSED SCORE.  Two staves over time -- one per channel -- in
     which each channel ANNOTATES THE OTHER: a witness mark under each
     staff says whether the OTHER view saw something move at that
     moment.
  3. THE DIFFERENCED VIEW.  What moved in the fixed frame, the roving
     frame that explains it, and the difference between them: which of
     the fixed frame's movements the roving frame witnessed and which
     it did not.

RENDER POSITIONALLY.  The grid IS the position: 40x24 cells over the
160x96 luma, four pixels per cell, the same geometry the corpus's own
`experiments/lens.py` computes.  A cell that moved is a cell that moved
WHERE IT MOVED, and the renders never summarise that away.

MORE THAN ONE WAY.  Every render comes in more than one form: the
motion ramp or the luma ramp or both; the window statistic is the
MEAN change over the window (default) or the MAXIMUM; and the
quantisation scale is either SHARED between the two views (the default,
because a side-by-side of two different scales is a lie) or per view
(available, and labelled as not-comparable).  The choice is printed in
the output, because a rendering choice is not a measurement and the
reader is entitled to know which one was made.

THE WINDOW IS A MEAN, NOT A SUM.  The two cameras will not run at the
same frame rate, and a sum over the window hands the faster camera a
free factor of speed.  The mean change per frame is the same quantity
for both, and the corpus's own noise floor (3.0 luma) was measured on
a single frame's change map, so it is a threshold on a MEAN.  The
WITNESS question -- "did the other eye see something move at this
moment" -- is asked of the MAXIMUM over the window instead, because a
thing that moved for one frame in five still moved.

KEEP ALL.  Every moment in the window is rendered.  Nothing is
filtered, nothing is dropped for being dull, and the refusals state
their reason rather than silently shrinking the window.

THE ALIGNMENT IS NOT THIS TOOL'S BUSINESS.  The claps are the sync
events and zcode owns them.  This tool CONSUMES an alignment sidecar:

    {"claps": {"fixed_ms": [...], "roving_ms": [...]}}

and fits the offset and the drift from the clap pairs, or accepts
them directly as {"offset_ms": ..., "drift_ppm": ...}.  A clap time is
a time in ONE stream's clock, and the two streams do not share an
origin, so the tool RE-BASES both sides onto each stream's own first
frame before it fits.  With no sidecar the tool assumes the two
streams started together, PRINTS THAT ASSUMPTION IN EVERY OUTPUT, and
refuses to describe a sub-frame alignment it does not have.

Run from the repo root:
    /home/proart/projects/gpu-venv/bin/python experiments/dual_render.py --selftest
    /home/proart/projects/gpu-venv/bin/python experiments/dual_render.py \\
        --fixed <proart-session> --roving <phone-ingest> --out docs/research/dual
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import lens  # noqa: E402

#: The corpus's grid: 40x24 over 160x96, four pixels per cell.  The
#: SAME geometry `experiments/lens.py` computes, because the two
#: programs must agree on where a thing is.
GRID_W, GRID_H = 40, 24
FRAME_W, FRAME_H = 160, 96

#: The change map's noise floor, measured by the corpus's own lens
#: study: 3.0 luma is 2.6x the non-zero median change of 1.152 and
#: about 5x the room's motion-noise sigma of 0.61.
CHANGE_NOISE_FLOOR = 3.0

#: The motion ramp: nothing moved, up to the most that moved.  The
#: same ramp the splat-room's glyph grid ships, so a reader who has
#: seen one has seen the other.
MOTION_RAMP = " .:-=+*#%@"

#: The luma ramp, for the "what the picture looks like" rendering.
LUMA_RAMP = " .:-=+*#%@"


# --------------------------------------------------------------------------
# a stream: the corpus's own format
# --------------------------------------------------------------------------

class Stream:
    """One camera's frames, in the corpus's format.

    A stream is a directory with `luma.raw` (160x96 uint8, frames
    concatenated) and `timestamps.json` (the spine: `wall_clock_start`
    and `frame_offsets_ms`).  That is what the ProArt session bundles
    already are, and what `experiments/dual_ingest.py` makes of a
    phone recording -- so ONE reader serves both sides and no new
    format is invented.

    THE PHONE IS NOT A VIDEO, SO IT IS NOT THE SAME BUNDLE.  The
    room-lens renders CHARACTERS, so the phone's bundle carries
    `change.raw` (the change maps, already reduced to the corpus's
    grid) and `luma_grid.raw` (the picture, at the corpus's grid)
    instead of `luma.raw`.  One reader still serves both, because the
    change map is the thing the renders are made of and it is the same
    KIND of quantity in both cases: the mean absolute luma difference
    over a cell, per frame pair.

    EVERY STREAM CARRIES ITS OWN FLOOR.  The corpus's 3.0 luma floor
    was measured on the ProArt, at the ProArt's frame interval, over
    the ProArt's own change maps.  A different sensor rendering a
    posterised picture has a different floor, and a witness threshold
    borrowed from another sensor is a threshold with no meaning.
    """

    def __init__(self, path: Path, label: str):
        self.path = Path(path)
        self.label = label
        self.problems: list[str] = []

        luma = self.path / 'luma.raw'
        change = self.path / 'change.raw'
        ts = self.path / 'timestamps.json'
        if not luma.is_file() and not change.is_file():
            self.problems.append(f'no luma.raw or change.raw in {self.path}')
        if not ts.is_file():
            self.problems.append(f'no timestamps.json in {self.path}')
        self.frames: np.ndarray = np.zeros((0, FRAME_H, FRAME_W), np.float32)
        #: the PICTURE, always on the corpus's cell grid, so the two
        #: panels of a luma render are the same SIZE
        self.picture: np.ndarray = np.zeros((0, GRID_H, GRID_W), np.float32)
        self.motion: np.ndarray | None = None
        #: the time of each change map, in this stream's own clock
        self.motion_times_s: np.ndarray = np.zeros(0)
        self.offsets_s: np.ndarray = np.zeros(0)
        self.wall_clock_start = ''
        #: The file's OWN first frame offset, before this class re-bases
        #: it to zero.  A sidecar's clap times live in the FILE's clock,
        #: so this is what has to come off them -- and it is NOT zero in
        #: the corpus: the ProArt sessions start at 6834.7 ms because
        #: their offsets are performance.now() minus the pre-roll base.
        self.base_offset_ms: float = 0.0
        #: this sensor's own noise floor, in this sensor's own units
        self.floor: float = CHANGE_NOISE_FLOOR
        #: what the change map's numbers MEAN, in words
        self.units: str = 'luma (the camera\'s own)'
        #: the median interval between this stream's change maps
        self.interval_s: float = 0.0
        #: the share of this stream's frames that show a change above
        #: its OWN floor.  A channel that is on most of the time cannot
        #: TIME a movement, and the witness count is blind to it.
        self.duty_cycle: float = 0.0
        self.is_change_map: bool = False

        data: dict = {}
        if ts.is_file():
            try:
                data = json.loads(ts.read_text())
            except Exception:  # noqa: BLE001
                self.problems.append('timestamps.json is not readable JSON')
        self.wall_clock_start = str(data.get('wall_clock_start', ''))

        # the frame offsets, in the file's clock and then re-based
        offs = data.get('frame_offsets_ms')
        if isinstance(offs, list) and len(offs) > 1:
            raw_offs = np.asarray(offs, dtype=np.float64) / 1000.0
            self.base_offset_ms = float(raw_offs[0] * 1000.0)
            self.offsets_s = raw_offs - raw_offs[0]
        elif not change.is_file():
            self.problems.append('timestamps.json carries no frame_offsets_ms')

        if change.is_file():
            # THE CHANGE-MAP STREAM: the phone's room-lens bundle
            self.is_change_map = True
            raw = np.fromfile(change, dtype=np.float32)
            n = raw.size // (GRID_H * GRID_W)
            if n < 2:
                self.problems.append(f'only {n} change maps in {self.path}')
                self.motion = None
            else:
                self.motion = raw[:n * GRID_H * GRID_W].reshape(
                    n, GRID_H, GRID_W).astype(np.float32)
            c_offs = data.get('change_offsets_ms')
            if isinstance(c_offs, list) and len(c_offs) == len(self.motion):
                ct = np.asarray(c_offs, dtype=np.float64) / 1000.0
                self.motion_times_s = ct - ct[0]
            elif len(self.motion):
                self.motion_times_s = np.arange(len(self.motion), dtype=np.float64)
                self.problems.append(
                    'timestamps.json carries no change_offsets_ms, so the '
                    'change maps are placed on an ARBITRARY clock')
            lg = self.path / 'luma_grid.raw'
            if lg.is_file():
                praw = np.fromfile(lg, dtype=np.uint8)
                pn = praw.size // (GRID_H * GRID_W)
                self.picture = praw[:pn * GRID_H * GRID_W].reshape(
                    pn, GRID_H, GRID_W).astype(np.float32)
        elif luma.is_file():
            # THE CORPUS STREAM
            raw = np.fromfile(luma, dtype=np.uint8)
            n = raw.size // (FRAME_W * FRAME_H)
            if n < 3:
                self.problems.append(f'only {n} frames of luma in {self.path}')
            else:
                self.frames = raw[:n * FRAME_W * FRAME_H].reshape(
                    n, FRAME_H, FRAME_W).astype(np.float32)
                # the picture on the corpus's grid, so the luma render
                # can put two views side by side at the same SIZE
                self.picture = self.frames.reshape(
                    n, GRID_H, FRAME_H // GRID_H, GRID_W,
                    FRAME_W // GRID_W).mean(axis=(2, 4)).astype(np.float32)
            self.motion = (lens.frame_motion_maps(self.path)
                           if luma.is_file() else None)
            if self.motion is not None and self.offsets_s.size >= 2:
                self.motion_times_s = (self.offsets_s[:-1]
                                       + self.offsets_s[1:]) / 2.0
        else:
            self.problems.append('the change map could not be computed')

        # this sensor's own floor, and its own frame interval
        meta_path = self.path / 'meta.json'
        if meta_path.is_file():
            try:
                smeta = json.loads(meta_path.read_text())
            except Exception:  # noqa: BLE001
                smeta = {}
            noise = smeta.get('the_noise_of_this_sensor') or {}
            if isinstance(noise, dict) and noise.get('the_floor') is not None:
                self.floor = float(noise['the_floor'])
            if smeta.get('the_units'):
                self.units = str(smeta['the_units'])
        if self.motion is not None and len(self.motion_times_s) >= 2:
            gaps = np.diff(self.motion_times_s)
            self.interval_s = float(np.median(gaps)) if gaps.size else 0.0
        if self.motion is not None and len(self.motion):
            mx = self.motion.reshape(len(self.motion), -1).max(axis=1)
            self.duty_cycle = float((mx > self.floor).mean())

    @property
    def ok(self) -> bool:
        return not self.problems and self.motion is not None

    def n_frames(self) -> int:
        return 0 if self.motion is None else len(self.motion)

    def time_of_motion_frame(self, i: int) -> float:
        """The time of the change between frames i and i+1."""
        if self.motion_times_s.size == 0:
            return 0.0
        i = max(0, min(i, len(self.motion_times_s) - 1))
        return float(self.motion_times_s[i])

    def motion_index_at(self, t_s: float) -> int | None:
        """The change-map frame nearest a time."""
        if self.motion is None or self.motion_times_s.size == 0:
            return None
        idx = int(np.argmin(np.abs(self.motion_times_s - t_s)))
        return idx

    def picture_index_at(self, t_s: float) -> int:
        """The PICTURE frame nearest a time.

        The picture and the change map are different lengths -- a
        change map needs two frames and a picture needs one -- so they
        are indexed separately."""
        if self.picture.size == 0 or self.offsets_s.size == 0:
            return 0
        idx = int(np.argmin(np.abs(self.offsets_s - t_s)))
        return max(0, min(idx, len(self.picture) - 1))

    def motion_window(self, t_s: float, half_window_s: float,
                      how: str = 'mean') -> np.ndarray:
        """The change map around a time, over the window.

        A single frame pair's change map is one frame of sensor noise
        with a body in it; the window is what makes a MOVEMENT legible
        without pretending it happened at an instant.

        `how` is the statistic over the window's frames:

          'mean'  the mean change per frame -- frame-rate independent,
                  and the statistic the corpus's 3.0 luma noise floor
                  was measured on;
          'max'   the strongest single frame in the window -- what the
                  WITNESS question needs, because a thing that moved for
                  one frame in five still moved;
          'sum'   the raw sum, kept because it is what the first draft
                  used and the difference is worth seeing.
        """
        if self.motion is None:
            return np.zeros((GRID_H, GRID_W), np.float32)
        mid = self.motion_times_s
        sel = (mid >= t_s - half_window_s) & (mid <= t_s + half_window_s)
        if not sel.any():
            idx = self.motion_index_at(t_s)
            if idx is None:
                return np.zeros((GRID_H, GRID_W), np.float32)
            return self.motion[idx]
        block = self.motion[sel]
        if how == 'mean':
            return block.mean(axis=0)
        if how == 'max':
            return block.max(axis=0)
        if how == 'sum':
            return block.sum(axis=0)
        raise ValueError(f'unknown window statistic {how}')

    def motion_window_count(self, t_s: float, half_window_s: float) -> int:
        """How many frames the window holds -- the sample count that
        says how much evidence a moment has behind it."""
        if self.motion is None or self.motion_times_s.size == 0:
            return 0
        mid = self.motion_times_s
        sel = (mid >= t_s - half_window_s) & (mid <= t_s + half_window_s)
        return int(sel.sum())


# --------------------------------------------------------------------------
# the alignment
# --------------------------------------------------------------------------

class Alignment:
    """The clap alignment, CONSUMED rather than computed.

    The claps are zcode's lane.  This class fits the offset and the
    drift from the clap pairs, or takes them directly, and -- when
    there is no sidecar at all -- says so."""

    def __init__(self, offset_ms: float = 0.0, drift_ppm: float = 0.0,
                 source: str = 'none (the two streams are ASSUMED to start together)',
                 n_claps: int = 0, residual_ms: float | None = None,
                 note: str = ''):
        self.offset_ms = float(offset_ms)
        self.drift_ppm = float(drift_ppm)
        self.source = source
        self.n_claps = int(n_claps)
        self.residual_ms = residual_ms
        #: the sidecar's OWN note, carried through into every output --
        #: a sidecar that says what it is (the wall clock, not the
        #: claps) must have that said wherever the alignment is shown
        self.note = note
        #: the per-clap residuals, and which claps the fit used -- so a
        #: reader can see WHICH clap was wrong, not just that something
        #: was
        self.the_clap_residuals_ms: list[float] = []
        self.the_claps_used: list[int] = []
        self.the_claps_dropped: list[int] = []

    @property
    def assumed(self) -> bool:
        return self.source.startswith('none')

    def roving_time_for_fixed(self, t_fixed_s: float) -> float:
        """The roving-stream time of the instant the fixed stream calls
        t_fixed_s."""
        slope = 1.0 + self.drift_ppm * 1e-6
        return (t_fixed_s - self.offset_ms / 1000.0) / slope

    def fixed_time_for_roving(self, t_roving_s: float) -> float:
        slope = 1.0 + self.drift_ppm * 1e-6
        return t_roving_s * slope + self.offset_ms / 1000.0

    def describe(self) -> str:
        if self.assumed:
            return ('NO ALIGNMENT: ' + self.source +
                    ' -- the two views are aligned only to the frame')
        out = (f'alignment: {self.n_claps} clap pairs, offset '
               f'{self.offset_ms:+.3f} ms, drift {self.drift_ppm:+.1f} ppm'
               + (f', residual {self.residual_ms:.3f} ms'
                  if self.residual_ms is not None else ''))
        if self.the_clap_residuals_ms:
            per = ', '.join(f'{v:.1f}' for v in self.the_clap_residuals_ms)
            out += f'  [the per-clap residuals in ms: {per}]'
        if self.the_claps_dropped:
            out += (f'  [claps {self.the_claps_dropped} were DROPPED as '
                    f'outliers]')
        if self.note:
            out += f'\n  THE SIDECAR SAYS: {self.note}'
        return out


#: A clap whose residual from the fit is worse than this is not a clap
#: the fit should believe.  The arrival-null study met the same thing
#: in the audio: one bad arrival tilted the implied speed of sound from
#: 343 to 306 m/s, and the fix was a robust refit on the inliers.
CLAP_ROBUST_MS = 40.0


def load_alignment(path: Path | None, fixed: Stream | None = None,
                   roving: Stream | None = None) -> Alignment:
    """Read the alignment sidecar, in either accepted form.

    THE CLOCK IS THE TOOL'S BUSINESS, NOT THE SIDECAR'S.  A clap time
    in the sidecar is a time in ONE stream's clock -- the phone's
    picture timestamp for one, the browser's performance.now() for the
    other -- and those clocks do not share an origin.  The tool knows
    each stream's own origin (the first frame), so it re-bases both
    sides itself and then fits.  A sidecar that says
    `"the_clock": "rebased"` is taken at its word instead, and the
    output says which it did.

    Accepted forms:

        {"claps": {"fixed_ms": [...], "roving_ms": [...]}}
        {"offset_ms": ..., "drift_ppm": ...}

    A clap is detected in the AUDIO.  The sidecar wants it in the
    FRAME clock.  Those are not the same clock: the corpus has already
    measured the audio-to-video offset inside one capture at about
    114 ms, so the conversion is not free.  A sidecar may therefore
    carry the skew and let this tool apply it:

        {"claps": {...},
         "audio_to_frame_ms": {"fixed": 114.0, "roving": 0.0}}

    When it is absent the clap times are taken as already being in the
    frame clock, and the output says which it assumed.
    """
    if path is None:
        return Alignment()
    data = json.loads(Path(path).read_text())
    skew = data.get('audio_to_frame_ms')
    skew_note = ''
    if isinstance(skew, dict):
        sk_f = float(skew.get('fixed', 0.0))
        sk_r = float(skew.get('roving', 0.0))
        skew_note = (f'the audio-to-frame skew was applied (fixed '
                     f'{sk_f:+.3f} ms, roving {sk_r:+.3f} ms)')
    claps = data.get('claps')
    if isinstance(claps, dict) and claps.get('fixed_ms') and claps.get('roving_ms'):
        f = np.asarray(claps['fixed_ms'], dtype=np.float64)
        r = np.asarray(claps['roving_ms'], dtype=np.float64)
        n = min(len(f), len(r))
        f, r = f[:n], r[:n]
        if data.get('the_clock', 'frame_offsets_ms') == 'rebased':
            base_f = base_r = 0.0
            clock_note = 'the clap times are already relative to each stream\'s first frame'
        else:
            base_f = (float(fixed.base_offset_ms)
                      if fixed is not None else 0.0)
            base_r = (float(roving.base_offset_ms)
                      if roving is not None else 0.0)
            clock_note = (f'the clap times were re-based onto each stream\'s '
                          f'first frame (fixed -{base_f:.1f} ms, roving '
                          f'-{base_r:.1f} ms)')
        f = f - base_f
        r = r - base_r
        if isinstance(skew, dict):
            f = f + float(skew.get('fixed', 0.0))
            r = r + float(skew.get('roving', 0.0))
        if skew_note:
            clock_note = f'{clock_note}; {skew_note}'
        if n < 1:
            return Alignment(source=f'{path.name} (no clap pairs)')
        if n == 1:
            # one clap gives the offset and no drift
            return Alignment(offset_ms=float(f[0] - r[0]), drift_ppm=0.0,
                             source=f'{path.name} (1 clap pair: offset only, '
                                    f'no drift can be fitted from one clap; '
                                    f'{clock_note})',
                             n_claps=n)
        # fixed_ms = a * roving_ms + b  ->  a is the slope, b the offset.
        # The drift is the slope's DEPARTURE FROM ONE, so it is
        # (a - 1) in parts per million.  Writing a * 1e6 instead would
        # report a 1 ppm drift as 1e6 ppm and turn the slope into 2.
        #
        # A SPURIOUS CLAP.  One door slammed must not tilt the whole
        # alignment -- the arrival-null study met exactly this in the
        # audio, where one bad arrival took the implied speed of sound
        # from 343 to 306 m/s.  The threshold cannot be applied to the
        # ordinary fit, because least squares SPREADS a bad clap's
        # error over the good ones: with four claps and one 500 ms
        # outlier, every clap ends up over the bar and nothing looks
        # like an outlier.  So the outliers are FOUND by search --
        # every pair of claps proposes a line, and the line with the
        # most claps on it wins, ties going to the line with the
        # smallest total error.
        a, b = np.polyfit(r, f, 1)
        resid = np.abs(a * r + b - f)
        used = list(range(n))
        dropped: list[int] = []
        if n >= 3 and float(resid.max()) > CLAP_ROBUST_MS:
            best = None
            for i in range(n):
                for j in range(i + 1, n):
                    if r[j] == r[i]:
                        continue
                    ca = (f[j] - f[i]) / (r[j] - r[i])
                    cb = f[i] - ca * r[i]
                    cres = np.abs(ca * r + cb - f)
                    key = (int((cres <= CLAP_ROBUST_MS).sum()),
                           -float(cres.sum()))
                    if best is None or key > best[0]:
                        best = (key, ca, cb, cres)
            if best is not None:
                inliers = [i for i in range(n)
                           if best[3][i] <= CLAP_ROBUST_MS]
                if len(inliers) >= 2 and len(inliers) < n:
                    a, b = np.polyfit(r[inliers], f[inliers], 1)
                    resid = np.abs(a * r + b - f)
                    used = inliers
                    dropped = [i for i in range(n) if i not in inliers]
        out = Alignment(offset_ms=b, drift_ppm=(a - 1.0) * 1e6,
                        source=f'{path.name} (fitted from the clap pairs; '
                               f'{clock_note})',
                        n_claps=n, residual_ms=float(resid.max()))
        out.the_clap_residuals_ms = [round(float(v), 3) for v in resid]
        out.the_claps_used = used
        out.the_claps_dropped = dropped
        # THE PLAUSIBILITY OF THE OFFSET, GROUNDED IN THE CLAPS
        # THEMSELVES.  A clap matched to the WRONG clap does not give a
        # small offset -- it gives one of about half the clap spacing,
        # because that is how far the wrong clap is.  So the bar is not
        # a number somebody liked: it is the distance between the
        # claps.
        gaps = np.diff(np.sort(r))
        if gaps.size and abs(b) > float(np.median(gaps)) / 2.0:
            out.source += (f'; THE OFFSET IS SUSPECT: {b:+.1f} ms is more '
                           f'than half the clap spacing '
                           f'({float(np.median(gaps)):.1f} ms), which is what '
                           f'a clap matched to the WRONG clap looks like')
        if dropped:
            out.source += (f'; {len(dropped)} clap(s) DROPPED as outliers '
                           f'(residual over {CLAP_ROBUST_MS:.0f} ms) and the '
                           f'fit was made on the {len(used)} inliers')
        if float(resid.max()) > CLAP_ROBUST_MS:
            out.source += ('; THE ALIGNMENT IS SUSPECT: a residual of '
                           f'{float(resid.max()):.3f} ms is over the '
                           f'{CLAP_ROBUST_MS:.0f} ms bar')
        return out
    if 'offset_ms' in data or 'drift_ppm' in data:
        return Alignment(offset_ms=float(data.get('offset_ms', 0.0)),
                         drift_ppm=float(data.get('drift_ppm', 0.0)),
                         source=f'{path.name} (offset and drift as given, in '
                                f'the re-based clock)',
                         n_claps=int(data.get('n_claps', 0)),
                         note=str(data.get('the_note', '')))
    return Alignment(source=f'{path.name} carries neither clap pairs nor an offset',
                     note=str(data.get('the_note', '')))


# --------------------------------------------------------------------------
# the rendering
# --------------------------------------------------------------------------

def quantise(values: np.ndarray, scale: float, ramp: str = MOTION_RAMP) -> str:
    """A grid as text, one character per cell.

    The scale is the value that maps to the TOP of the ramp, so the
    brightest thing in the window is the last glyph.  Rounding is
    HALF-UP (`floor(x + 0.5)`), not numpy's default: numpy rounds a
    half to the nearest EVEN, which maps a mid-scale value one step
    low and silently biases every panel toward the dimmer end.

    The scale is SHARED between the two views by default: a side-by-
    side of two different scales says nothing about which view saw
    more, and the reader is owed a comparison they can trust."""
    top = max(scale, 1e-6)
    idx = np.floor(values / top * (len(ramp) - 1) + 0.5).astype(int)
    q = np.clip(idx, 0, len(ramp) - 1)
    return '\n'.join(''.join(ramp[v] for v in row) for row in q)


def side_by_side(left: list[str], right: list[str], gap: str = '   |   ') -> list[str]:
    """Two blocks of text, side by side."""
    n = max(len(left), len(right))
    out = []
    for i in range(n):
        a = left[i] if i < len(left) else ''
        b = right[i] if i < len(right) else ''
        out.append(a + gap + b)
    return out


def shared_scale(a: np.ndarray, b: np.ndarray, mode: str,
                 floor_a: float = CHANGE_NOISE_FLOOR,
                 floor_b: float = CHANGE_NOISE_FLOOR) -> float:
    """The scale both views are quantised with.

    THE FLOORS ARE THE TWO SENSORS' OWN.  With one scale and two
    sensors, the scale has to be at least the HIGHER of the two floors
    -- otherwise a view's own "this is a thing, not grain" reading
    would be squashed into the bottom of the ramp and would look like
    nothing moved.
    """
    if mode == 'shared-auto':
        return float(max(a.max(), b.max(), floor_a, floor_b))
    if mode == 'shared-floor':
        return float(max(floor_a, floor_b))
    if mode == 'shared-fixed':
        return 12.0
    raise ValueError(f'unknown scale mode {mode}')


def per_view_scale(a: np.ndarray, b: np.ndarray,
                   floor_a: float = CHANGE_NOISE_FLOOR,
                   floor_b: float = CHANGE_NOISE_FLOOR) -> tuple[float, float]:
    """Each view against its OWN maximum.

    This is the "more than one way" that is a lie: the two panels are
    then unreadable against each other, and every output that uses it
    says so in the header."""
    return (float(max(a.max(), floor_a)), float(max(b.max(), floor_b)))


def _scaled_pair(fa: np.ndarray, rb: np.ndarray, fixed: Stream,
                 roving: Stream, scale_mode: str
                 ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """The two panels' values, and the scales to quantise them with.

    THE THREE WAYS, AND WHAT EACH ONE ANSWERS.

      'shared-auto'      one scale for both views.  This is the
                         default because a side-by-side of two scales
                         says nothing about which view saw more.  It
                         is only meaningful when the two views are in
                         the SAME UNITS -- and the duet is not: the
                         ProArt's change map is in the camera's luma
                         and the phone's is in the room-lens's own
                         brightness, so one scale renders one of them
                         black.  Kept, and filed, because the black
                         panel is the honest picture of that mismatch.

      'per-view-auto'    each view against its own maximum.  The two
                         panels are then readable, and NOT comparable
                         in magnitude -- every output that uses it
                         says so.

      'floor-relative'   each view divided by that view's OWN noise
                         floor, then one scale for both.  This is the
                         only comparison the two sensors make
                         possible: the panels read in TIMES THIS
                         SENSOR'S OWN GRAIN, so a cell at the top of
                         the ramp means the same thing on both sides.
    """
    if scale_mode == 'floor-relative':
        na = fa / max(fixed.floor, 1e-6)
        nb = rb / max(roving.floor, 1e-6)
        sc = float(max(na.max(), nb.max(), 1.0))
        return na, nb, sc, sc
    if scale_mode == 'per-view-auto':
        sc_l, sc_r = per_view_scale(fa, rb, fixed.floor, roving.floor)
        return fa, rb, sc_l, sc_r
    sc = shared_scale(fa, rb, scale_mode, fixed.floor, roving.floor)
    return fa, rb, sc, sc


def _picture_at(stream: Stream, t_s: float) -> np.ndarray:
    """The picture nearest a time, on the corpus's cell grid.

    BOTH PANELS LAND ON THE SAME GRID, so a luma render puts two
    views side by side at the same SIZE.  It does not make them the
    same VIEW: the two cameras see the room from different places, and
    the header says so.
    """
    if stream.picture.size == 0:
        return np.zeros((GRID_H, GRID_W), np.float32)
    return stream.picture[stream.picture_index_at(t_s)]


def _the_two_sensors(fixed: Stream, roving: Stream) -> list[str]:
    """What the two sensors ARE, in the units the panels are drawn in.

    THE TWO VIEWS ARE NOT IN THE SAME UNITS, and a reader who is not
    told that will read the panels as if they were.  The corpus's
    change map is in the camera's luma; the room-lens's is in the
    room-lens's own brightness -- a 2.2x contrast stretch of the
    camera's luma, clipped, with half its range collapsed into one
    glyph.  So each sensor's own noise floor is a different NUMBER,
    and the two sample the room at different rates.
    """
    out = [
        f'  the fixed view:  change in {fixed.units}; its own floor '
        f'{fixed.floor:.3f}; one frame every {fixed.interval_s * 1000.0:.1f} ms; '
        f'it shows a change in {fixed.duty_cycle * 100.0:.1f} % of its frames',
        f'  the roving view: change in {roving.units}; its own floor '
        f'{roving.floor:.3f}; one frame every {roving.interval_s * 1000.0:.1f} ms; '
        f'it shows a change in {roving.duty_cycle * 100.0:.1f} % of its frames',
    ]
    if fixed.duty_cycle > 0.5 or roving.duty_cycle > 0.5:
        out.append('  A CHANNEL THAT IS ON MOST OF THE TIME CANNOT TIME A '
                   'MOVEMENT: its witness marks are then close to constant, '
                   'and the')
        out.append('  contingency table has nothing left to measure. That is '
                   'what the duty cycle above is for.')
    if fixed.interval_s and roving.interval_s:
        ratio = roving.interval_s / fixed.interval_s
        out.append(
            f'  the two views sample the room at DIFFERENT rates: the '
            f'roving view\'s interval is {ratio:.2f}x the fixed view\'s, so '
            f'the same movement covers {ratio:.2f}x more of the picture per '
            f'frame in the roving view')
    out.append('  the two change maps are not in the same units and not in '
               'the same space: a cell on the left and a cell on the right '
               'are not the same part of the room')
    return out


def render_dual_ascii(fixed: Stream, roving: Stream, align: Alignment,
                       moments_s: list[float], scale_mode: str,
                       mode: str, half_window_s: float,
                       how: str = 'mean') -> tuple[list[str], dict]:
    """THE DUAL-ASCII, SIDE BY SIDE.

    The same MOMENTS from both views.  The panels are aligned in TIME
    and NOT in space -- the two cameras see the room from different
    places, so a cell in the left panel and a cell in the right panel
    are not the same part of the room, and the header says so."""
    blocks: list[str] = []
    left_blocks: list[list[str]] = []
    right_blocks: list[list[str]] = []
    numbers = []
    for t in moments_s:
        t_r = align.roving_time_for_fixed(t)
        fa = fixed.motion_window(t, half_window_s, how)
        rb = roving.motion_window(t_r, half_window_s, how)
        fa_q, rb_q, sc_l, sc_r = _scaled_pair(fa, rb, fixed, roving,
                                              scale_mode)
        if mode in ('change', 'both'):
            left_blocks.append(quantise(fa_q, sc_l).split('\n'))
            right_blocks.append(quantise(rb_q, sc_r).split('\n'))
        if mode in ('luma', 'both'):
            left_blocks.append(quantise(_picture_at(fixed, t), 255.0,
                                        LUMA_RAMP).split('\n'))
            right_blocks.append(quantise(_picture_at(roving, t_r), 255.0,
                                         LUMA_RAMP).split('\n'))
        numbers.append({
            't_fixed_s': round(t, 3),
            't_roving_s': round(t_r, 3),
            'fixed_energy': round(float(fa.sum()), 2),
            'roving_energy': round(float(rb.sum()), 2),
            'fixed_cells_above_floor': int((fa > fixed.floor).sum()),
            'roving_cells_above_floor': int((rb > roving.floor).sum()),
            'fixed_window_frames': fixed.motion_window_count(t, half_window_s),
            'roving_window_frames': roving.motion_window_count(t_r, half_window_s),
            'fixed_scale': round(float(sc_l), 3),
            'roving_scale': round(float(sc_r), 3),
        })

    blocks.append('THE DUAL-ASCII -- the same moments from both views')
    blocks.append(f'  fixed: {fixed.path}   roving: {roving.path}')
    blocks.append(f'  {align.describe()}')
    blocks.append(f'  the panels are aligned in TIME and NOT in space: the two '
                  f'cameras see the room from different places')
    blocks.append(f'  mode: {mode}   scale: {scale_mode}   window statistic: '
                  f'{how}   window: +/-{half_window_s:.2f} s')
    blocks.extend(_the_two_sensors(fixed, roving))
    if scale_mode == 'per-view-auto':
        blocks.append('  EACH VIEW HAS ITS OWN SCALE: the two panels are NOT '
                      'comparable against each other')
    if scale_mode == 'floor-relative':
        blocks.append('  EACH PANEL IS IN TIMES ITS OWN SENSOR\'S GRAIN: a '
                      'cell at the top of the ramp means the same thing on '
                      'both sides')
    header_left = f'FIXED ({fixed.label})'.ljust(GRID_W)
    header_right = f'ROVING ({roving.label})'.ljust(GRID_W)
    blocks.extend(side_by_side([header_left], [header_right]))
    blocks.append('')
    for k, t in enumerate(moments_s):
        n = numbers[k]
        blocks.append(f'--- t = {n["t_fixed_s"]:+.3f} s fixed / '
                      f'{n["t_roving_s"]:+.3f} s roving   '
                      f'fixed {n["fixed_energy"]:.1f} luma  '
                      f'roving {n["roving_energy"]:.1f} luma   '
                      f'({n["fixed_window_frames"]} vs '
                      f'{n["roving_window_frames"]} frames in the window)')
        blocks.extend(side_by_side(left_blocks[k], right_blocks[k]))
        blocks.append('')
    return blocks, {'moments': numbers, 'window_statistic': how,
                    'scale_mode': scale_mode}


def _fused_rows(fixed: Stream, roving: Stream, align: Alignment,
                t0: float, t1: float, slice_s: float) -> list[dict]:
    """The fused score's slices, as rows.

    Factored out of the render so that the SAME rows can be recomputed
    under a different alignment -- which is how the alignment's
    sensitivity is measured."""
    n_slices = max(1, int(round((t1 - t0) / slice_s)))
    rows = []
    fixed_energy = []
    roving_energy = []
    for i in range(n_slices):
        a = t0 + i * slice_s
        b = a + slice_s
        mid = (a + b) / 2.0
        fa = fixed.motion_window(mid, slice_s / 2.0, 'mean')
        rb = roving.motion_window(
            align.roving_time_for_fixed(mid), slice_s / 2.0, 'mean')
        fe = float(fa.sum())
        re = float(rb.sum())
        fixed_energy.append(fe)
        roving_energy.append(re)
        # the ENERGY is the mean over the window; the WITNESS asks
        # whether the other eye saw anything move at that moment, and
        # that is a question about the strongest single frame.  EACH
        # EYE IS ASKED AGAINST ITS OWN FLOOR -- a threshold borrowed
        # from the other sensor is a threshold with no meaning.
        fx = fixed.motion_window(mid, slice_s / 2.0, 'max')
        rx = roving.motion_window(
            align.roving_time_for_fixed(mid), slice_s / 2.0, 'max')
        f_moved = bool((fx > fixed.floor).any())
        r_moved = bool((rx > roving.floor).any())
        mark = '=' if (f_moved and r_moved) else ('>' if f_moved else
                                                  ('<' if r_moved else '.'))
        rows.append({'t_s': round(a, 3), 'fixed': fe, 'roving': re,
                     'mark': mark,
                     'fixed_cells': int((fx > fixed.floor).sum()),
                     'roving_cells': int((rx > roving.floor).sum())})
    return rows, fixed_energy, roving_energy


def render_fused_score(fixed: Stream, roving: Stream, align: Alignment,
                       t0: float, t1: float, slice_s: float) -> tuple[list[str], dict]:
    """THE FUSED SCORE -- two staves over time, each annotating the
    other.

    A score is quantized and it is read top to bottom, so the staves
    are time slices.  Each staff carries its own channel's energy; the
    row under it is the OTHER channel's witness: '=' when both moved,
    '>' when only this one moved, '<' when only the other did, '.' when
    neither did."""
    rows, fixed_energy, roving_energy = _fused_rows(
        fixed, roving, align, t0, t1, slice_s)
    n_slices = len(rows)

    top = max(max(fixed_energy), max(roving_energy), fixed.floor, roving.floor)
    f_chars = ''.join(MOTION_RAMP[min(len(MOTION_RAMP) - 1,
                                       int(round(v / top * (len(MOTION_RAMP) - 1))))]
                      for v in fixed_energy)
    r_chars = ''.join(MOTION_RAMP[min(len(MOTION_RAMP) - 1,
                                       int(round(v / top * (len(MOTION_RAMP) - 1))))]
                      for v in roving_energy)
    f_marks = ''.join(r['mark'] for r in rows)
    r_marks = ''.join('=' if r['mark'] == '=' else
                       ('<' if r['mark'] == '>' else
                        ('>' if r['mark'] == '<' else '.')) for r in rows)

    lines = []
    lines.append('THE FUSED SCORE -- two staves, each channel annotating the other')
    lines.append(f'  fixed: {fixed.path}   roving: {roving.path}')
    lines.append(f'  {align.describe()}')
    lines.append(f'  {n_slices} slices of {slice_s:.2f} s from {t0:+.2f} s; '
                 f'the energy scale is shared, maximum {top:.1f} luma '
                 f'(the mean change over the slice)')
    lines.append('  marks:  = both moved   > only this one   < only the other   '
                 '. neither')
    lines.append('  a mark is decided by the STRONGEST frame in the slice, so a '
                 'thing that moved for one frame still counts')
    lines.append(f'  each eye is asked against its OWN floor: the fixed view at '
                 f'{fixed.floor:.3f}, the roving view at {roving.floor:.3f}')
    lines.extend(_the_two_sensors(fixed, roving))
    lines.append('')
    lines.append('  time      ' + ''.join(('|' if i % 10 == 0 else ' ')
                                          for i in range(n_slices)))
    lines.append(f'  FIXED     {f_chars}')
    lines.append(f'  <- roving {f_marks}')
    lines.append(f'  ROVING    {r_chars}')
    lines.append(f'  <- fixed  {r_marks}')
    lines.append('')
    lines.append('  t(s)      fixed(luma)     roving(luma)     mark')
    for r in rows:
        lines.append(f'  {r["t_s"]:+8.3f}  {r["fixed"]:14.1f}  '
                     f'{r["roving"]:14.1f}  {r["mark"]}')
    lines.append('  the energy is the sum over the 960 cells of the mean change '
                 'per frame in the slice; the mark is decided by the strongest '
                 'single frame')
    return lines, {'slices': rows, 'n_slices': n_slices}


def crossing_witness(fixed: Stream, roving: Stream, align: Alignment,
                     t0: float, t1: float,
                     shift_max_s: float = 120.0) -> dict:
    """THE CROSSING WITNESS -- do the two views find the same CROSSINGS?

    THE BINARY COUNT CANNOT SEE A MOVEMENT.  It asks "did something
    change", and the roving eye is IN the room, so the answer is
    almost always yes: the phone's channel is on 99.8 % of the time,
    the contingency table has a constant row, and the excess over
    chance is exactly zero points.

    THE CORPUS'S OWN CROSSING IS THE RIGHT WITNESS.  A crossing is a
    CONNECTED BLOB of change that holds its shape while it TRANSLATES
    -- the person is one object, it keeps its size, and it moves.  The
    sensor's grain is below the floor, a highlight is one frame, a
    fidget does not travel, and neither does a SHAKE.  So each view
    is asked the question it can answer: did a THING move across the
    picture?

    **THE GAP IS THE CHANNEL'S OWN, IN FRAMES, NOT IN SECONDS.**  The
    corpus's detector defaults to a 0.35 s gap and to offsets at 60 fps,
    and the corpus's own sessions run at 2 fps -- so at 0.35 s no blob
    ever chains and the detector finds NOTHING AT ALL.  The gap here is
    **two frames of the channel**, so a blob that misses one frame
    still chains and one that misses two is a new event.
    """
    out: dict = {'the_shift_max_s': shift_max_s, 'the_readings': {}}
    gaps = {'fixed': max(0.04, fixed.interval_s * 2.0),
            'roving': max(0.04, roving.interval_s * 2.0)}
    try:
        from experiments import lens as _lens  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        out['the_note'] = 'the corpus lens could not be imported'
        return out

    def crossings(stream: Stream) -> list[tuple[float, float]]:
        ev = _lens.find_crossings(stream.motion, threshold=stream.floor,
                                  offsets=stream.motion_times_s,
                                  max_gap_s=gaps['fixed' if stream is fixed
                                                 else 'roving'])
        return [(float(stream.motion_times_s[e['start_frame']]),
                 float(stream.motion_times_s[e['end_frame']])) for e in ev]

    fx_all = crossings(fixed)
    rv_all = crossings(roving)
    fx = [(a, b) for a, b in fx_all if b >= t0 and a <= t1]
    out['the_fixed_crossings_total'] = len(fx_all)
    out['the_fixed_crossings_in_the_shared_minutes'] = len(fx)
    out['the_roving_crossings_total'] = len(rv_all)
    out['the_gaps_s'] = {k: round(v, 3) for k, v in gaps.items()}
    if not fx or not rv_all:
        which = []
        if not fx:
            which.append('the fixed view')
        if not rv_all:
            which.append('the roving view')
        out['the_note'] = (
            f'{" and ".join(which)} found no crossings at all. The corpus '
            f'detector needs a blob that holds its shape while it '
            f'TRANSLATES, and a channel that changes everywhere -- a hand-'
            f'held camera, whose whole picture moves -- has no such blob. '
            f'That is a finding about the channel, not a failure of the '
            f'render.')
        return out

    def _overlap(a, b, c, d, crit: str) -> bool:
        lo, hi = max(a, c), min(b, d)
        if hi <= lo:
            return False
        if crit == 'overlap':
            return True
        return (hi - lo) >= 0.5 * min(b - a, d - c)

    def count(dx: float, crit: str) -> int:
        n = 0
        for a, b in rv_all:
            A = align.fixed_time_for_roving(a) + dx
            B = align.fixed_time_for_roving(b) + dx
            if B < t0 or A > t1:
                continue
            for f0, f1 in fx:
                if _overlap(A, B, f0, f1, crit):
                    n += 1
                    break
        return n

    shifts = list(range(int(-shift_max_s), int(shift_max_s) + 1))
    for crit in ('overlap', 'half'):
        obs = count(0.0, crit)
        slides = np.asarray([count(float(d), crit) for d in shifts])
        sd = float(slides.std())
        out['the_readings'][crit] = {
            'the_criterion': ('the two crossings\' intervals overlap at all'
                              if crit == 'overlap' else
                              'they share at least half of the SHORTER one'),
            'the_observed': obs,
            'the_n_roving_crossings': len(rv_all),
            'the_null_mean': round(float(slides.mean()), 3),
            'the_null_sd': round(sd, 3),
            'the_null_max': int(slides.max()),
            'the_z': round((obs - float(slides.mean())) / sd, 3) if sd > 0 else None,
            'the_share_of_slides_that_match_it': int((slides >= obs).sum()),
            'the_n_slides': len(slides),
            'the_p_by_the_slide': round(float((slides >= obs).sum()) / len(slides), 4),
        }
    # how far the alignment can be wrong before the result goes
    sens = []
    for d in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0):
        obs = count(d, 'overlap')
        slides = np.asarray([count(float(x), 'overlap') for x in shifts])
        sd = float(slides.std())
        sens.append({'the_shift_s': d, 'the_observed': obs,
                     'the_null_mean': round(float(slides.mean()), 3),
                     'the_z': round((obs - float(slides.mean())) / sd, 3) if sd > 0 else None})
    out['the_alignment_sensitivity'] = sens
    out['the_note'] = (
        'the looser criterion is the one that was asked for first and the '
        'tighter one is reported beside it, not instead of it. The result '
        'is a statement that the two views find the same crossings WITHIN '
        'A FEW SECONDS; it is not a measurement of the alignment, because '
        'it survives a misalignment of several seconds.')
    return out


def render_crossing_witness(w: dict) -> list[str]:
    """The crossing witness, printed."""
    lines = []
    lines.append('THE CROSSING WITNESS -- do the two views find the same '
                 'CROSSINGS?')
    if not w.get('the_readings'):
        lines.append(f'  {w.get("the_note", "no reading")}')
        return lines
    lines.append('  a crossing is the corpus\'s own: a CONNECTED BLOB of change '
                 'that holds its shape while it')
    lines.append('  TRANSLATES.  The grain does not translate, a highlight is one '
                 'frame, a fidget does not travel,')
    lines.append('  and neither does a shake -- so this is the question each view '
                 'can actually answer.')
    lines.append(f'  the fixed view found {w["the_fixed_crossings_total"]} '
                 f'crossings, {w["the_fixed_crossings_in_the_shared_minutes"]} '
                 f'of them inside the shared minutes; the roving view found '
                 f'{w["the_roving_crossings_total"]}')
    g = w.get('the_gaps_s', {})
    lines.append(f'  the detector\'s gap is TWO FRAMES of each channel (the '
                 f'fixed {g.get("fixed", 0):.2f} s, the roving '
                 f'{g.get("roving", 0):.2f} s), not the corpus tool\'s 0.35 s default:')
    lines.append('  that default is calibrated for 60 fps and these channels '
                 'run at 2, so at 0.35 s no blob ever chains.')
    lines.append('')
    lines.append('  criterion                                 observed   null mean'
                 '   null sd       z   slides that match it      p')
    for name, r in w['the_readings'].items():
        lines.append(
            f'  {r["the_criterion"][:40]:40s}  {r["the_observed"]:4d} of '
            f'{r["the_n_roving_crossings"]:3d}   {r["the_null_mean"]:9.2f}  '
            f'{r["the_null_sd"]:8.2f}  {r["the_z"]:+5.2f}   '
            f'{r["the_share_of_slides_that_match_it"]:4d} of '
            f'{r["the_n_slides"]:4d}   {r["the_p_by_the_slide"]:.3f}')
    lines.append('')
    lines.append('  how much the alignment can be wrong before the result goes:')
    for s in w.get('the_alignment_sensitivity', []):
        lines.append(f'    the alignment moved {s["the_shift_s"]:+5.1f} s: the '
                     f'observed {s["the_observed"]:3d}, the null mean '
                     f'{s["the_null_mean"]:6.2f}, z {s["the_z"]:+5.2f}')
    lines.append('')
    lines.append('  A RESULT THAT SURVIVES A MISALIGNMENT OF SEVERAL SECONDS IS '
                 'NOT A MEASUREMENT OF THE')
    lines.append('  ALIGNMENT.  It is a statement that the two views find the '
                 'same crossings at the same')
    lines.append('  MOMENTS, which is the thing two sovereign logs of the same '
                 'minutes give that neither')
    lines.append('  gives alone.  The claps are still what settles the skew.')
    return lines


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """The Pearson correlation of two series."""
    if a.size < 3 or a.size != b.size:
        return 0.0
    x = a - a.mean()
    y = b - b.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den > 0 else 0.0


def graded_witness(fixed: Stream, roving: Stream, align: Alignment,
                   t0: float, t1: float, slice_s: float) -> dict:
    """THE GRADED WITNESS -- do the two channels' energies rise and fall
    together?

    THE BINARY COUNT IS BLUNT.  The 2x2 table asks "did something move
    in this slice", which is a yes/no question about a channel that
    may be SATURATED: when both cameras have something in view, both
    say yes for most of the time, and the table has nothing left to
    measure.  The GRADED question -- do the two channels' change
    energies rise and fall TOGETHER -- uses the whole signal instead
    of its sign.

    THE NULL IS THE SLIDE.  The roving channel is slid against the
    fixed one, slice by slice, and the correlation is recomputed at
    every offset.  A correlation that a slide can match was not
    measuring the alignment.  The null's spread is the
    AUTOCORRELATION of two slow series -- a person walking about a
    room produces bursts minutes long -- so the observed correlation
    has to beat THAT, not the correlation of two noises.

    AND THE SLIDE CANNOT FIND THE ALIGNMENT.  The sweep's largest
    value need not be at the truth, because two slow series correlate
    somewhere by luck.  So the sweep is reported with its shape, and
    a peak is only called a peak if it is SHARP: the spread of the
    region within one null standard deviation of the maximum is
    reported beside it, and a broad region means the sweep found
    nothing to localise.
    """
    rows, fe, re = _fused_rows(fixed, roving, align, t0, t1, slice_s)
    fe = np.asarray(fe, dtype=np.float64)
    re = np.asarray(re, dtype=np.float64)
    n = fe.size
    out: dict = {'n_slices': int(n), 'the_slice_s': slice_s,
                 'the_readings': {}}
    if n < 8:
        out['the_note'] = 'too few slices to correlate'
        return out
    for name, (a, b) in (('raw', (fe, re)),
                         ('log', (np.log1p(fe), np.log1p(re)))):
        obs = _corr(a, b)
        sweep = []
        for d in range(-(n // 2), n // 2 + 1):
            x, y = (a[:n - d], b[d:]) if d >= 0 else (a[-d:], b[:n + d])
            sweep.append((d, _corr(x, y)))
        vals = np.asarray([r for _, r in sweep])
        best_d, best_r = max(sweep, key=lambda kv: kv[1])
        sd = float(vals.std())
        near = [d for d, r in sweep if r >= best_r - sd]
        out['the_readings'][name] = {
            'the_correlation_at_the_alignment': round(obs, 4),
            'the_null_mean': round(float(vals.mean()), 4),
            'the_null_sd': round(sd, 4),
            'the_z_against_the_null': (round((obs - float(vals.mean())) / sd, 3)
                                       if sd > 0 else None),
            'the_share_of_slides_that_match_it': int((vals >= obs).sum()),
            'the_n_slides': len(sweep),
            'the_sweep_maximum': {'the_shift_slices': best_d,
                                  'the_shift_s': round(best_d * slice_s, 3),
                                  'the_correlation': round(best_r, 4)},
            'the_sweep_minimum': {'the_shift_slices': min(sweep, key=lambda kv: kv[1])[0],
                                  'the_correlation': round(min(r for _, r in sweep), 4)},
            'the_peak_sharpness_s': round((max(near) - min(near)) * slice_s, 3),
            'the_sweep': [{'the_shift_slices': d, 'the_shift_s': round(d * slice_s, 3),
                           'the_correlation': round(r, 4)} for d, r in sweep],
        }
    out['the_note'] = (
        'the raw reading is the one that was asked for first; the log '
        'reading was added after the raw one was seen, so it is a SECOND '
        'reading and not a replacement. Neither is called the answer '
        'here: the slide null says how much of each is the room and how '
        'much is two slow series correlating somewhere by luck.')
    return out


def render_graded_witness(w: dict, slice_s: float) -> list[str]:
    """The graded witness, printed."""
    lines = []
    lines.append('THE GRADED WITNESS -- do the two channels rise and fall '
                 'together?')
    if not w.get('n_slices'):
        return lines
    lines.append(f'  {w["n_slices"]} slices of {slice_s:.2f} s; the reading is '
                 f'the correlation of the two channels\' change energies')
    lines.append('  THE NULL IS THE SLIDE: the roving channel is slid against '
                 'the fixed one and the')
    lines.append('  correlation is recomputed at every offset.  The null\'s '
                 'spread is the AUTOCORRELATION of two slow')
    lines.append('  series, and the observed has to beat that, not the '
                 'correlation of two noises.')
    lines.append('')
    lines.append('  reading   r(at the alignment)   null mean   null sd      z'
                 '   slides that match it')
    for name, r in w.get('the_readings', {}).items():
        lines.append(
            f'  {name:7s}  {r["the_correlation_at_the_alignment"]:+19.4f}  '
            f'{r["the_null_mean"]:+10.4f}  {r["the_null_sd"]:8.4f}  '
            f'{r["the_z_against_the_null"]:+6.2f}   '
            f'{r["the_share_of_slides_that_match_it"]:4d} of '
            f'{r["the_n_slides"]}')
    lines.append('')
    for name, r in w.get('the_readings', {}).items():
        m = r['the_sweep_maximum']
        lines.append(
            f'  the {name} reading: the slide\'s own maximum is at '
            f'{m["the_shift_s"]:+.1f} s with r = {m["the_correlation"]:+.4f}; '
            f'the region within one null sd of it spans '
            f'{r["the_peak_sharpness_s"]:.0f} s')
    lines.append('')
    lines.append('  A SHARP PEAK WOULD BE AN ALIGNMENT MEASURED FROM THE DATA. '
                 'A BROAD ONE IS NOT:')
    lines.append('  two slow series correlate somewhere by luck, and a sweep '
                 'that cannot say WHERE is saying')
    lines.append('  nothing about the skew.  The claps are not optional.')
    return lines


def alignment_sensitivity(fixed: Stream, roving: Stream, align: Alignment,
                          t0: float, t1: float, slice_s: float,
                          shifts_ms: list[float]) -> list[dict]:
    """HOW MUCH THE WITNESS COUNT DEPENDS ON THE ALIGNMENT.

    THE WALL CLOCK IS NOT THE CLAPS.  The two devices' clocks have
    never been compared, so the wall-clock alignment carries a
    device-to-device skew that nobody has measured, and the count's
    excess over chance is worth something only if it survives a
    misalignment.  So the count is recomputed with the alignment
    DELIBERATELY moved, and the table says how much it moves.

    A count that collapses under a shift of a few hundred
    milliseconds was measuring the alignment, not the room.  A count
    that does not move is measuring something both cameras see
    whatever the clock says -- a different claim, and the table is
    what tells the two apart.
    """
    out = []
    for d in shifts_ms:
        a2 = Alignment(offset_ms=align.offset_ms + d,
                       drift_ppm=align.drift_ppm,
                       source=f'{align.source} (SHIFTED {d:+.0f} ms on purpose)',
                       n_claps=align.n_claps, note=align.note)
        rows, _, _ = _fused_rows(fixed, roving, a2, t0, t1, slice_s)
        cnt = witness_count(rows)
        out.append({
            'the_shift_ms': d,
            'n_slices': cnt.get('n_slices'),
            'both_moved': cnt.get('both_moved'),
            'p_fixed': cnt.get('p_fixed'),
            'p_roving': cnt.get('p_roving'),
            'p_both': cnt.get('p_both'),
            'p_both_if_independent': cnt.get('p_both_if_independent'),
            'the_excess_over_chance': cnt.get('the_excess_over_chance'),
            'the_chi_square': cnt.get('the_chi_square'),
            'the_p_value': cnt.get('the_p_value'),
        })
    return out


def render_alignment_sensitivity(table: list[dict], slice_s: float) -> list[str]:
    """The sensitivity table, printed."""
    lines = []
    lines.append('THE ALIGNMENT\'S SENSITIVITY -- the witness count with the '
                 'alignment moved on purpose')
    lines.append('  the wall clock is NOT the claps: the two devices\' clocks '
                 'have never been compared, so the')
    lines.append('  alignment carries a skew nobody has measured.  This table '
                 'asks what the count does when the')
    lines.append('  alignment is DELIBERATELY wrong, so the reader can see how '
                 'much of the count is the room')
    lines.append('  and how much is the clock.')
    lines.append('')
    lines.append('  shift(ms)   both   p_fixed  p_roving   p_both   chance   '
                 'excess   chi-sq      p')
    for r in table:
        if r.get('p_both') is None:
            continue
        chi = r.get('the_chi_square')
        pv = r.get('the_p_value')
        chi_s = f'{chi:6.2f}' if chi is not None else '     -'
        pv_s = f'{pv:8.4g}' if pv is not None else '       -'
        lines.append(
            f'  {r["the_shift_ms"]:+9.0f}  {r["both_moved"]:4d}  '
            f'{r["p_fixed"]:7.3f}  {r["p_roving"]:8.3f}  '
            f'{r["p_both"]:7.3f}  {r["p_both_if_independent"]:7.3f}  '
            f'{r["the_excess_over_chance"]:+6.3f}  '
            f'{chi_s}  {pv_s}')
    lines.append('')
    lines.append('  a count that collapses under a shift of a few hundred '
                 'milliseconds was measuring the ALIGNMENT, not the room')
    lines.append('  a count that does not move is measuring something both '
                 'cameras see whatever the clock says')
    return lines


def witness_count(rows: list[dict]) -> dict:
    """WHAT THE TWO VIEWS SAY TOGETHER THAT NEITHER SAYS ALONE.

    The fused score's marks are a 2x2 table of the same minutes seen
    two ways: both views moved, only the fixed moved, only the roving
    moved, neither.  The number worth having is the EXCESS over
    chance: if the two views were independent, they would both move in
    p_fixed * p_roving of the slices, and the difference between that
    and what they actually do is the agreement -- the two eyes seeing
    the same movement from two places.

    The slices TILE the time (each window is exactly one slice), so
    they do not overlap.  They are still autocorrelated in time -- one
    person's walk spans several slices -- so the chi-square's p-value
    is OPTIMISTIC and is reported with that said.
    """
    n = len(rows)
    if not n:
        return {'n_slices': 0}
    both = sum(1 for r in rows if r['mark'] == '=')
    fixed_only = sum(1 for r in rows if r['mark'] == '>')
    roving_only = sum(1 for r in rows if r['mark'] == '<')
    neither = sum(1 for r in rows if r['mark'] == '.')
    p_fixed = (both + fixed_only) / n
    p_roving = (both + roving_only) / n
    p_both = both / n
    chance = p_fixed * p_roving
    out = {
        'n_slices': n,
        'both_moved': both, 'only_fixed_moved': fixed_only,
        'only_roving_moved': roving_only, 'neither_moved': neither,
        'the_fixed_view_moved_in': both + fixed_only,
        'the_roving_view_moved_in': both + roving_only,
        'p_fixed': round(p_fixed, 4), 'p_roving': round(p_roving, 4),
        'p_both': round(p_both, 4),
        'p_both_if_independent': round(chance, 4),
        'the_excess_over_chance': round(p_both - chance, 4),
        'the_excess_in_slices': round((p_both - chance) * n, 2),
    }
    # chi-square on the 2x2 table, with the autocorrelation caveat
    try:
        from scipy.stats import chi2  # noqa: PLC0415
        table = [[both, fixed_only], [roving_only, neither]]
        total = n
        row1 = both + fixed_only
        row2 = roving_only + neither
        col1 = both + roving_only
        col2 = fixed_only + neither
        denom = row1 * row2 * col1 * col2
        if denom > 0:
            chi = (total * (both * neither - fixed_only * roving_only) ** 2
                   / denom)
            out['the_chi_square'] = round(float(chi), 4)
            out['the_p_value'] = float(chi2.sf(chi, 1))
            out['the_p_value_caveat'] = (
                'the p-value assumes the slices are independent of one '
                'another; a movement spans slices, so this is OPTIMISTIC')
    except Exception:  # noqa: BLE001
        out['the_chi_square'] = None
    return out


def render_witness_count(count: dict, slice_s: float) -> list[str]:
    """The witness count, printed."""
    lines = []
    lines.append('THE WITNESS COUNT -- what the two views say together that '
                 'neither says alone')
    if not count.get('n_slices'):
        return lines
    n = count['n_slices']
    pct = lambda k: 100.0 * k / n  # noqa: E731
    lines.append(f'  {n} slices of {slice_s:.2f} s; the windows TILE the '
                 f'time, so they do not overlap')
    lines.append(f'  both moved            {count["both_moved"]:6d}  '
                 f'({pct(count["both_moved"]):5.1f} %)')
    lines.append(f'  only the fixed moved  {count["only_fixed_moved"]:6d}  '
                 f'({pct(count["only_fixed_moved"]):5.1f} %)')
    lines.append(f'  only the roving moved {count["only_roving_moved"]:6d}  '
                 f'({pct(count["only_roving_moved"]):5.1f} %)')
    lines.append(f'  neither moved         {count["neither_moved"]:6d}  '
                 f'({pct(count["neither_moved"]):5.1f} %)')
    lines.append('')
    lines.append(f'  the fixed view moved in  {count["the_fixed_view_moved_in"]} '
                 f'slices ({pct(count["the_fixed_view_moved_in"]):.1f} %)')
    lines.append(f'  the roving view moved in {count["the_roving_view_moved_in"]}'
                 f' slices ({pct(count["the_roving_view_moved_in"]):.1f} %)')
    lines.append(f'  if the two views were INDEPENDENT they would both move '
                 f'in {count["p_both_if_independent"] * 100:.1f} % of the '
                 f'slices')
    lines.append(f'  they both move in {count["p_both"] * 100:.1f} % -- '
                 f'{count["the_excess_over_chance"] * 100:+.1f} points over '
                 f'chance')
    if count.get('the_chi_square') is not None:
        lines.append(f'  chi-square {count["the_chi_square"]:.2f} on 1 degree '
                     f'of freedom, p = {count["the_p_value"]:.4g}')
        lines.append(f'  {count["the_p_value_caveat"]}')
    lines.append('')
    lines.append('  THE EXCESS IS NOT TWO SENSORS AGREEING ABOUT A NUMBER. It '
                 'is two eyes')
    lines.append('  agreeing about a MOVEMENT -- the same person, seen from '
                 'two places, at')
    lines.append('  the same moment. That is the thing two sovereign logs of '
                 'the same minutes')
    lines.append('  can give that neither gives alone.')
    return lines


def render_differenced(fixed: Stream, roving: Stream, align: Alignment,
                       t: float, scale_mode: str,
                       half_window_s: float,
                       how: str = 'mean') -> tuple[list[str], dict]:
    """THE DIFFERENCED VIEW -- what moved in the fixed frame, the
    roving frame that explains it, and the difference between them.

    The third panel is the difference: for each cell of the FIXED
    frame, '+' where the fixed frame moved and the roving frame saw
    nothing at that moment, '#' where the fixed frame moved AND the
    roving frame witnessed movement at that moment, '.' where the
    fixed frame did not move.

    The witness is a statement about TIME, not about space: the roving
    frame saw movement at that moment, from somewhere else, and cannot
    say which cell of the fixed frame it was."""
    t_r = align.roving_time_for_fixed(t)
    fa = fixed.motion_window(t, half_window_s, how)
    rb = roving.motion_window(t_r, half_window_s, how)
    fa_q, rb_q, sc_l, sc_r = _scaled_pair(fa, rb, fixed, roving, scale_mode)
    f_moved = fa > fixed.floor
    # the witness is asked of the strongest single frame in the window
    rx = roving.motion_window(t_r, half_window_s, 'max')
    witnessed = bool((rx > roving.floor).any())

    diff = np.full((GRID_H, GRID_W), '.', dtype='<U1')
    for r in range(GRID_H):
        for c in range(GRID_W):
            if f_moved[r, c]:
                diff[r, c] = '#' if witnessed else '+'
    diff_lines = [''.join(row) for row in diff]

    n_fixed = int(f_moved.sum())
    n_roving = int((rb > roving.floor).sum())
    lines = []
    lines.append('THE DIFFERENCED VIEW -- the fixed frame, the roving frame, '
                 'and the difference')
    lines.append(f'  fixed: {fixed.path}   roving: {roving.path}')
    lines.append(f'  {align.describe()}')
    lines.append(f'  t = {t:+.3f} s fixed / {t_r:+.3f} s roving   '
                 f'window +/-{half_window_s:.2f} s   scale: {scale_mode}   '
                 f'window statistic: {how}')
    lines.append(f'  the fixed frame moved in {n_fixed} cells (its own floor '
                 f'{fixed.floor:.3f}); the roving frame saw movement in '
                 f'{n_roving} cells (its own floor {roving.floor:.3f})')
    lines.extend(_the_two_sensors(fixed, roving))
    if scale_mode == 'per-view-auto':
        lines.append('  EACH VIEW HAS ITS OWN SCALE: the two panels are NOT '
                     'comparable against each other')
    if scale_mode == 'floor-relative':
        lines.append('  EACH PANEL IS IN TIMES ITS OWN SENSOR\'S GRAIN: a '
                     'cell at the top of the ramp means the same thing on '
                     'both sides')
    lines.append('')
    lines.extend(side_by_side(
        [f'FIXED CHANGE ({fixed.label})'.ljust(GRID_W),
         f'ROVING CHANGE ({roving.label})'.ljust(GRID_W)],
        [f'DIFFERENCE (witness mask)'.ljust(GRID_W)]))
    lines.append('')
    lines.extend(side_by_side(quantise(fa_q, sc_l).split('\n'),
                               quantise(rb_q, sc_r).split('\n'), gap='   |   '))
    lines.append('')
    lines.append('THE DIFFERENCE PANEL, alone (it is the point of the render):')
    lines.extend(diff_lines)
    lines.append('')
    lines.append('  . nothing moved in the fixed frame')
    lines.append('  + the fixed frame moved and the roving frame saw NOTHING '
                 'at that moment')
    lines.append('  # the fixed frame moved and the roving frame WITNESSED '
                 'movement at that moment')
    return lines, {
        't_fixed_s': round(t, 3), 't_roving_s': round(t_r, 3),
        'n_fixed_cells_moved': n_fixed, 'n_roving_cells_moved': n_roving,
        'witnessed': witnessed,
    }


# --------------------------------------------------------------------------
# the pair
# --------------------------------------------------------------------------

def check_pair(fixed: Stream, roving: Stream, align: Alignment) -> dict:
    """Is this pair renderable?  A precise blocker, never an
    appearance."""
    out = {
        'fixed': {'path': str(fixed.path), 'label': fixed.label,
                  'frames': len(fixed.frames), 'ok': fixed.ok,
                  'problems': fixed.problems,
                  'wall_clock_start': fixed.wall_clock_start,
                  'the_change_maps': fixed.n_frames(),
                  'the_floor': fixed.floor, 'the_units': fixed.units,
                  'the_frame_interval_ms': round(fixed.interval_s * 1000.0, 3)},
        'roving': {'path': str(roving.path), 'label': roving.label,
                   'frames': len(roving.frames), 'ok': roving.ok,
                   'problems': roving.problems,
                   'wall_clock_start': roving.wall_clock_start,
                   'the_change_maps': roving.n_frames(),
                   'the_floor': roving.floor, 'the_units': roving.units,
                   'the_frame_interval_ms': round(roving.interval_s * 1000.0, 3)},
        'alignment': {'source': align.source, 'assumed': align.assumed,
                      'offset_ms': align.offset_ms, 'drift_ppm': align.drift_ppm,
                      'n_claps': align.n_claps, 'the_note': align.note},
    }
    blockers = list(fixed.problems) + list(roving.problems)
    # THE OVERLAP IS IN THE ALIGNED CLOCK.  The two streams have
    # DIFFERENT origins -- in the duet the phone's walkabout started
    # 500 s after the ProArt's page was loaded -- so their raw time
    # ranges never meet, and the ALIGNMENT is what brings them
    # together.  Taking the overlap of the raw ranges would report
    # that the pair has no shared minutes at all.
    def _span(stream: Stream):
        if stream.offsets_s.size == 0:
            return None
        return (float(stream.offsets_s[0]), float(stream.offsets_s[-1]))

    fs = _span(fixed)
    rs = _span(roving)
    if fs is None or rs is None:
        t0 = t1 = 0.0
    else:
        rs_in_fixed = (align.fixed_time_for_roving(rs[0]),
                       align.fixed_time_for_roving(rs[1]))
        t0 = max(fs[0], rs_in_fixed[0])
        t1 = min(fs[1], rs_in_fixed[1])
    out['the_overlap_s'] = [round(t0, 3), round(t1, 3)]
    out['the_fixed_span_s'] = [round(fs[0], 3), round(fs[1], 3)] if fs else None
    out['the_roving_span_s_in_the_fixed_clock'] = (
        [round(align.fixed_time_for_roving(rs[0]), 3),
         round(align.fixed_time_for_roving(rs[1]), 3)] if rs else None)
    if t1 <= t0:
        blockers.append('the two streams do not overlap in time')
    out['renderable'] = (not blockers)
    out['blockers'] = blockers
    return out


# --------------------------------------------------------------------------
# the synthetic pair -- the fixture the selftest grades against
# --------------------------------------------------------------------------

#: The roving eye's viewpoint, as a similarity transform of the room
#: coordinates: a scale, a rotation in radians, and a translation, both
#: in room units (0..1).  The fixed eye is the identity -- it is the
#: reference the room's coordinates are written in.
ROVING_VIEW = {'scale_x': 0.62, 'scale_y': 1.15, 'rotation_rad': 0.55,
               'shift_x': 0.10, 'shift_y': -0.03}

#: The fixed witness's field of view, in room coordinates.  A camera on
#: a stand does not see the whole room: past this edge the person
#: walks out of the fixed view while the phone, held in the hand and
#: pointing where the person is, still sees them.  That asymmetry is
#: what the witness marks are FOR -- without it every slice would be
#: '=' and the score would never say anything.
FIXED_FOV_X = (0.25, 1.0)


def project(pt: dict, view: dict | None = None) -> tuple[float, float]:
    """Where a point of the room appears in a view.

    The identity view is the fixed witness: it sees the room's
    coordinates as they are.  The roving eye sees the same person from
    another place, so its apparent position differs -- which is the
    whole point of a roving eye and the reason the two grids are NOT
    spatially comparable."""
    if view is None:
        return float(pt['x']), float(pt['y'])
    dx = float(pt['x']) - 0.5
    dy = float(pt['y']) - 0.5
    c = math.cos(view['rotation_rad'])
    s = math.sin(view['rotation_rad'])
    rx = dx * c - dy * s
    ry = dx * s + dy * c
    return (0.5 + rx * view['scale_x'] + view['shift_x'],
            0.5 + ry * view['scale_y'] + view['shift_y'])


def synth_pair(root: Path, seed: int = 5) -> dict:
    """A synthetic dual pair: a fixed view and a roving view of a
    person walking a KNOWN path, plus the truth.

    The point is that the renders can be GRADED.  A render with no
    truth behind it is a picture; a render with a truth behind it is
    a measurement, and this is how the render lane proves it can be
    one before the real pair lands.

    THE BACKGROUND IS STATIC.  The first draft re-randomised the
    background on every frame, which flooded the change map with
    noise: the fixed view's change centroid came out at the centre of
    the frame for every moment, and the "the fixed view finds the
    person where the truth puts them" check PASSED -- because the
    truth happened to put the person at x = 0.5 at the moment it
    asked about.  A test that passes because everything is at the
    centre is not a test.  The background is now drawn once per view
    and the grain is the only thing that moves on its own."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)

    n_frames = 120
    fps = 20.0
    duration = n_frames / fps

    # the truth: the person walks left to right and back, in the room's
    # own coordinates (0..1 across, 0..1 down)
    truth = []
    for i in range(n_frames):
        t = i / fps
        x = 0.5 + 0.35 * math.sin(2 * math.pi * t / 6.0)
        y = 0.55 + 0.08 * math.cos(2 * math.pi * t / 9.0)
        truth.append({'t_s': round(t, 4), 'x': round(x, 4), 'y': round(y, 4)})

    # the ROOM, seen from two viewpoints
    fixed = np.zeros((n_frames, FRAME_H, FRAME_W), np.float32)
    roving = np.zeros((n_frames, FRAME_H, FRAME_W), np.float32)
    for img, view in ((fixed, None), (roving, ROVING_VIEW)):
        # ONE static background for this view: a gradient, a few soft
        # patches of texture, and a horizon line to look at
        bg = np.linspace(30.0, 78.0, FRAME_H, dtype=np.float32)[:, None]
        bg = np.repeat(bg, FRAME_W, axis=1)
        bg += 14.0 * rng.random((FRAME_H, FRAME_W)).astype(np.float32)
        for _ in range(6):
            cy = int(rng.integers(4, FRAME_H - 4))
            cx = int(rng.integers(4, FRAME_W - 4))
            ry = int(rng.integers(3, 10))
            rx = int(rng.integers(3, 10))
            yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
            blob = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
            bg = np.where(blob, bg + 18.0, bg)
        for i, tr in enumerate(truth):
            frame = bg.copy()
            rx_p, ry_p = project(tr, view)
            px = int(rx_p * FRAME_W)
            py = int(ry_p * FRAME_H)
            rr = 5
            y0, y1 = max(0, py - rr), min(FRAME_H, py + rr + 1)
            x0, x1 = max(0, px - rr), min(FRAME_W, px + rr + 1)
            in_view = True
            if view is None and not (FIXED_FOV_X[0] <= tr['x'] <= FIXED_FOV_X[1]):
                in_view = False      # out of the fixed camera's field of view
            if in_view and y1 > y0 and x1 > x0:
                yy, xx = np.mgrid[y0:y1, x0:x1]
                mask = ((yy - py) ** 2 + (xx - px) ** 2) <= rr * rr
                frame[y0:y1, x0:x1] = np.where(mask, 190.0,
                                               frame[y0:y1, x0:x1])
            img[i] = frame
    # the fixed camera's own grain, and the roving camera's (which is
    # noisier, because a phone in a hand is noisier than a camera on
    # a stand)
    fixed += rng.normal(0.0, 1.2, fixed.shape)
    roving += rng.normal(0.0, 1.8, roving.shape)

    for name, frames in (('fixed', fixed), ('roving', roving)):
        d = root / name
        d.mkdir(exist_ok=True)
        (d / 'luma.raw').write_bytes(
            np.clip(frames, 0, 255).astype(np.uint8).tobytes())
        (d / 'timestamps.json').write_text(json.dumps({
            'wall_clock_start': '2026-10-21T00:00:00.000Z',
            'frame_offsets_ms': [round(i * 1000.0 / fps, 3)
                                 for i in range(n_frames)],
        }))
        (d / 'meta.json').write_text(json.dumps({
            'tool': 'dual_render_synthetic',
            'label': name,
            'wall_clock_start': '2026-10-21T00:00:00.000Z',
            'n_frames': n_frames,
            'luma': {'width': FRAME_W, 'height': FRAME_H},
        }, indent=2))

    (root / 'the_truth.json').write_text(json.dumps({
        'note': 'the person\'s path, in room coordinates, for grading the '
                'renders against',
        'frames': truth,
        'the_fixed_view': 'the identity: the fixed witness sees the room\'s '
                          'coordinates as they are',
        'the_fixed_field_of_view_x': list(FIXED_FOV_X),
        'the_fixed_sees_the_person': [
            {'t_s': tr['t_s'],
             'seen': bool(FIXED_FOV_X[0] <= tr['x'] <= FIXED_FOV_X[1])}
            for tr in truth],
        'the_roving_view': ROVING_VIEW,
        'the_roving_apparent': [
            {'t_s': tr['t_s'],
             'x': round(project(tr, ROVING_VIEW)[0], 4),
             'y': round(project(tr, ROVING_VIEW)[1], 4)}
            for tr in truth],
    }, indent=2))
    return {'root': str(root), 'duration_s': duration, 'n_frames': n_frames}


# --------------------------------------------------------------------------
# the selftest
# --------------------------------------------------------------------------

def selftest() -> int:
    import shutil
    import tempfile
    checks: list[tuple[str, bool, str]] = []

    def ck(name, ok, detail=''):
        checks.append((name, bool(ok), detail))

    # 1. the quantiser: the scale is shared, so the two views are
    #    comparable, and the ramp's ends are honoured.  The scale is
    #    the value that maps to the TOP of the ramp, so with a scale of
    #    9.0 and a ten-glyph ramp the index IS the value: 0 -> ' ',
    #    3 -> '-', 6 -> '*', and anything at or above the scale -> '@'.
    g = np.array([[0.0, 3.0], [6.0, 30.0]])
    txt = quantise(g, 9.0)
    ck('render: the quantiser honours the ramp ends',
       txt.split('\n')[0] == ' -' and txt.split('\n')[1] == '*@', repr(txt))
    ck('render: the quantiser rounds half UP, not to even',
       quantise(np.array([[4.5]]), 9.0) == '+',
       repr(quantise(np.array([[4.5]]), 9.0)))
    ck('render: a shared scale is used for both views',
       shared_scale(np.array([0.0, 4.0]), np.array([0.0, 40.0]),
                    'shared-auto') == 40.0)
    ck('render: the per-view scale is offered and labelled',
       per_view_scale(np.array([0.0, 4.0]), np.array([0.0, 40.0])) == (4.0, 40.0))

    # 2. the side-by-side keeps the two blocks aligned
    sb = side_by_side(['abcd', 'efgh'], ['1', '23'])
    ck('render: the side-by-side keeps the rows aligned',
       sb[0] == 'abcd   |   1' and sb[1] == 'efgh   |   23', str(sb))

    # 3. the alignment: a known offset and drift are inverted correctly
    al = Alignment(offset_ms=100.0, drift_ppm=1000.0)
    t_r = al.roving_time_for_fixed(1.0)
    ck('align: the offset and drift invert correctly',
       abs(al.fixed_time_for_roving(t_r) - 1.0) < 1e-9,
       f'{t_r} -> {al.fixed_time_for_roving(t_r)}')
    ck('align: an assumed alignment says so',
       Alignment().assumed and 'NO ALIGNMENT' in Alignment().describe())

    # 3b. the sidecar: the clap times come in each stream's OWN clock
    #     and the tool re-bases them.  This is the part a hand-written
    #     sidecar gets wrong, so it is tested rather than assumed.
    #     The fixture's base is made NON-ZERO on purpose: the corpus's
    #     own sessions start at 6834.7 ms, and a fixture whose base is
    #     zero hides the bug where the base is never taken off.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        synth_pair(tmp / 'pair', seed=5)
        for name, base in (('fixed', 6834.7), ('roving', 120.5)):
            p = tmp / 'pair' / name / 'timestamps.json'
            data = json.loads(p.read_text())
            data['frame_offsets_ms'] = [v + base for v in
                                        data['frame_offsets_ms']]
            p.write_text(json.dumps(data))
        fixed = Stream(tmp / 'pair' / 'fixed', 'synthetic fixed')
        roving = Stream(tmp / 'pair' / 'roving', 'synthetic roving')
        base_f = fixed.base_offset_ms
        base_r = roving.base_offset_ms
        ck('align: the stream keeps the file\'s own base',
           abs(base_f - 6834.7) < 1e-6 and abs(base_r - 120.5) < 1e-6,
           f'{base_f}, {base_r}')
        delta_ms = 250.0
        side = tmp / 'claps.json'
        side.write_text(json.dumps({'claps': {
            'fixed_ms': [1000.0 + base_f, 4000.0 + base_f],
            'roving_ms': [1000.0 + base_r + delta_ms,
                          4000.0 + base_r + delta_ms]}}))
        al3 = load_alignment(side, fixed, roving)
        ck('align: the sidecar is re-based onto each stream\'s first frame',
           abs(al3.offset_ms + delta_ms) < 1e-6 and not al3.assumed,
           f'offset {al3.offset_ms:+.3f} ms (expected {-delta_ms:+.3f})')
        ck('align: the re-based alignment inverts to the same instant',
           abs(al3.fixed_time_for_roving(al3.roving_time_for_fixed(1.0))
               - 1.0) < 1e-9)
        ck('align: the offset moves the roving clock the right way',
           abs(al3.roving_time_for_fixed(1.0) - (1.0 + delta_ms / 1000.0))
           < 1e-9, str(al3.roving_time_for_fixed(1.0)))
        # one clap gives an offset and NO drift
        one = tmp / 'one.json'
        one.write_text(json.dumps({'claps': {
            'fixed_ms': [1000.0 + base_f],
            'roving_ms': [1000.0 + base_r + delta_ms]}}))
        al4 = load_alignment(one, fixed, roving)
        ck('align: one clap gives an offset and no drift',
           abs(al4.offset_ms + delta_ms) < 1e-6 and al4.drift_ppm == 0.0,
           f'offset {al4.offset_ms:+.3f}, drift {al4.drift_ppm}')
        # a real drift: the phone's clock runs 500 ppm fast, so the
        # clap interval is longer in the fixed clock
        drift_ppm = 500.0
        a = 1.0 + drift_ppm * 1e-6
        r_ms = [1000.0, 4000.0]
        f_ms = [a * v - delta_ms for v in r_ms]
        drift_side = tmp / 'drift.json'
        drift_side.write_text(json.dumps({'claps': {
            'fixed_ms': [v + base_f for v in f_ms],
            'roving_ms': [v + base_r for v in r_ms]}}))
        al5 = load_alignment(drift_side, fixed, roving)
        ck('align: a 500 ppm drift is recovered as 500 ppm',
           abs(al5.drift_ppm - drift_ppm) < 1e-3,
           f'drift {al5.drift_ppm:+.3f} ppm')
        ck('align: the drifted alignment inverts to the same instant',
           abs(al5.fixed_time_for_roving(al5.roving_time_for_fixed(1.0))
               - 1.0) < 1e-9,
           f"{al5.fixed_time_for_roving(al5.roving_time_for_fixed(1.0)):.9f}")

        # 3c. A SPURIOUS CLAP.  One door slammed must not tilt the whole
        #     alignment -- the arrival-null study met exactly this in
        #     the audio, where one bad arrival took the implied speed
        #     of sound from 343 to 306 m/s.
        r_ms = [1000.0, 4000.0, 7000.0, 10000.0]
        f_ms = [v - delta_ms for v in r_ms]
        f_ms_clean = list(f_ms)
        f_ms[2] += 500.0          # the third clap is a door, not a clap
        spur = tmp / 'spurious.json'
        spur.write_text(json.dumps({'claps': {
            'fixed_ms': [v + base_f for v in f_ms],
            'roving_ms': [v + base_r for v in r_ms]}}))
        al6 = load_alignment(spur, fixed, roving)
        ck('align: one spurious clap does not tilt the offset',
           abs(al6.offset_ms + delta_ms) < 1.0,
           f'offset {al6.offset_ms:+.3f} ms (expected {-delta_ms:+.3f})')
        ck('align: the spurious clap is DROPPED and named',
           al6.the_claps_dropped == [2] and al6.the_claps_used == [0, 1, 3],
           f'dropped {al6.the_claps_dropped}, used {al6.the_claps_used}')
        ck('align: the output says the alignment was suspect',
           'SUSPECT' in al6.source or 'DROPPED' in al6.source)
        ck('align: the per-clap residuals are reported',
           len(al6.the_clap_residuals_ms) == 4 and
           max(al6.the_clap_residuals_ms) > CLAP_ROBUST_MS,
           str(al6.the_clap_residuals_ms))
        # and with all four claps good, nothing is dropped
        clean = tmp / 'clean.json'
        clean.write_text(json.dumps({'claps': {
            'fixed_ms': [v + base_f for v in f_ms_clean],
            'roving_ms': [v + base_r for v in r_ms]}}))
        al7 = load_alignment(clean, fixed, roving)
        ck('align: four good claps drop nothing',
           al7.the_claps_dropped == [] and
           abs(al7.offset_ms + delta_ms) < 1e-6,
           f'dropped {al7.the_claps_dropped}, offset {al7.offset_ms:+.6f}')

        # 3d. THE PLAUSIBILITY OF THE OFFSET, grounded in the clap
        #     spacing: a clap matched to the WRONG clap gives an
        #     offset of about a whole spacing, and that must be said.
        wrong = tmp / 'wrong_clap.json'
        wrong.write_text(json.dumps({'claps': {
            'fixed_ms': [v + base_f for v in [1000.0, 10000.0, 20000.0]],
            'roving_ms': [v + base_r for v in [10000.0, 20000.0, 30000.0]]}}))
        al8 = load_alignment(wrong, fixed, roving)
        ck('align: a clap matched to the wrong clap is called SUSPECT',
           'SUSPECT' in al8.source and 'WRONG clap' in al8.source,
           al8.source[-160:])
        ck('align: a sensible offset is not called SUSPECT',
           'SUSPECT' not in al7.source, al7.source[-160:])

        # 3e. THE AUDIO-TO-FRAME SKEW.  A clap is detected in the
        #     AUDIO and the sidecar wants it in the FRAME clock, and
        #     the corpus has measured those two apart by about 114 ms
        #     inside one capture.  The skew may be carried in the
        #     sidecar and applied here, and the output must say it did.
        skew_side = tmp / 'skewed.json'
        skew_side.write_text(json.dumps({
            'claps': {'fixed_ms': [1000.0 + base_f, 4000.0 + base_f],
                      'roving_ms': [1000.0 + base_r + delta_ms,
                                    4000.0 + base_r + delta_ms]},
            'audio_to_frame_ms': {'fixed': 114.0, 'roving': 0.0}}))
        al9 = load_alignment(skew_side, fixed, roving)
        ck('align: the audio-to-frame skew is applied when it is given',
           abs(al9.offset_ms + delta_ms - 114.0) < 1e-6,
           f'offset {al9.offset_ms:+.3f} ms (expected {-delta_ms + 114.0:+.3f})')
        ck('align: the output says the skew was applied',
           'skew was applied' in al9.source, al9.source[-140:])

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # 4. the synthetic pair: the fixed view's change map must find
        #    the person where the TRUTH says they are
        info = synth_pair(tmp / 'pair', seed=5)
        fixed = Stream(tmp / 'pair' / 'fixed', 'synthetic fixed')
        roving = Stream(tmp / 'pair' / 'roving', 'synthetic roving')
        ck('pair: both synthetic streams read',
           fixed.ok and roving.ok, str(fixed.problems + roving.problems))
        truth = json.loads((tmp / 'pair' / 'the_truth.json').read_text())['frames']

        # the moment asked about: the person is OFF the centre of the
        # frame and MOVING.  The first draft asked about t = 3.0 s,
        # where the truth's x is exactly 0.5 -- and the broken fixture
        # put the change centroid at the centre of the frame for
        # every moment, so the check passed for the wrong reason.
        t_mid = 0.75
        half = 0.12
        truth_at = [tr for tr in truth
                    if abs(tr['t_s'] - t_mid) <= half + 1e-9]
        tx = float(np.mean([tr['x'] for tr in truth_at]))
        ty = float(np.mean([tr['y'] for tr in truth_at]))

        def centroid(m: np.ndarray) -> tuple[float, float]:
            ys, xs = np.nonzero(m > CHANGE_NOISE_FLOOR)
            if not len(xs):
                return -1.0, -1.0
            return (float(xs.mean() + 0.5) / GRID_W,
                    float(ys.mean() + 0.5) / GRID_H)

        fa = fixed.motion_window(t_mid, half, 'mean')
        cx, cy = centroid(fa)
        ck('pair: the fixed view finds the person where the truth puts them',
           abs(cx - tx) < 0.10 and abs(cy - ty) < 0.10,
           f'grid ({cx:.3f},{cy:.3f}) vs truth ({tx:.3f},{ty:.3f})')
        ck('pair: the fixed view sees something at all',
           int((fa > CHANGE_NOISE_FLOOR).sum()) > 0,
           str(int((fa > CHANGE_NOISE_FLOOR).sum())))
        ck('pair: the fixed view is NOT simply the centre of the frame',
           abs(cx - 0.5) > 0.05, f'{cx:.3f}')

        # 5. the two views are NOT spatially comparable -- the roving
        #    view sees the person somewhere else, in TWO dimensions
        rb = roving.motion_window(t_mid, half, 'mean')
        rcx, rcy = centroid(rb)
        ck('pair: the roving view sees the person somewhere ELSE',
           math.hypot(rcx - cx, rcy - cy) > 0.08,
           f'fixed ({cx:.3f},{cy:.3f}) vs roving ({rcx:.3f},{rcy:.3f})')
        ck('pair: the roving view sees something at all',
           int((rb > CHANGE_NOISE_FLOOR).sum()) > 0,
           str(int((rb > CHANGE_NOISE_FLOOR).sum())))

        # 5b. the window statistic: the mean is frame-rate independent
        #     and the max is what the witness question needs
        fmax = fixed.motion_window(t_mid, half, 'max')
        ck('render: the window statistic is a mean, not a sum',
           float(fixed.motion_window(t_mid, half, 'mean').sum())
           <= float(fixed.motion_window(t_mid, half, 'sum').sum()),
           f"{float(fixed.motion_window(t_mid, half, 'mean').sum()):.1f} vs "
           f"{float(fixed.motion_window(t_mid, half, 'sum').sum()):.1f}")
        ck('render: the max window is at least the mean window',
           float(fmax.sum()) >= float(fa.sum()),
           f'{float(fmax.sum()):.1f} vs {float(fa.sum()):.1f}')

        # 6. the renders RUN and are shaped right
        al2 = Alignment(offset_ms=0.0, drift_ppm=0.0,
                        source='the synthetic pair (no claps)')
        moments = [0.5, 1.0, 1.5]
        dual, dnum = render_dual_ascii(fixed, roving, al2, moments,
                                        'shared-auto', 'change', 0.25)
        panel_lines = [ln for ln in dual if '   |   ' in ln]
        ck('render: the dual-ASCII has a panel per view per moment',
           len(panel_lines) == GRID_H * len(moments) + 1 and
           all(len(ln) > GRID_W + 7 for ln in panel_lines),
           f'{len(panel_lines)} panel lines, {len(dual)} lines')
        ck('render: the dual-ASCII states that the panels are not spatial',
           any('NOT in space' in line for line in dual))
        ck('render: the dual-ASCII states the window statistic',
           any('window statistic' in line for line in dual))

        # the whole overlap, so the slices cover the window where the
        # person walks OUT of the fixed camera's field of view and the
        # roving eye is the only one that still sees them
        fused, fnum = render_fused_score(fixed, roving, al2, 0.0, 5.0, 0.25)
        ck('render: the fused score has both staves and both annotations',
           any(line.strip().startswith('FIXED') for line in fused) and
           any(line.strip().startswith('ROVING') for line in fused) and
           any('<- roving' in line for line in fused) and
           any('<- fixed' for line in fused))
        ck('render: the fused score has one mark per slice',
           fnum['n_slices'] == 20, str(fnum['n_slices']))
        ck('render: the fused score annotates the other channel',
           all(r['mark'] in '=<>.' for r in fnum['slices']) and
           len(fnum['slices']) == fnum['n_slices'])
        marks = {r['mark'] for r in fnum['slices']}
        ck('render: the witness vocabulary is exercised, not just =',
           '=' in marks and ('>' in marks or '<' in marks),
           ''.join(sorted(marks)))

        # 6b. THE WITNESS COUNT: the two views together, against chance
        count = witness_count(fnum['slices'])
        ck('render: the witness count adds up',
           (count['both_moved'] + count['only_fixed_moved']
            + count['only_roving_moved'] + count['neither_moved'])
           == count['n_slices'], str(count['n_slices']))
        ck('render: the chance baseline is the product of the two rates',
           abs(count['p_both_if_independent']
               - count['p_fixed'] * count['p_roving']) < 1e-6)
        ck('render: the excess over chance is signed',
           abs(count['the_excess_over_chance']
               - (count['p_both'] - count['p_both_if_independent'])) < 1e-6)
        wtext = render_witness_count(count, 0.25)
        ck('render: the witness count says what the excess is',
           any('INDEPENDENT' in ln for ln in wtext) and
           any('points over' in ln for ln in wtext))
        # the printed rates must be the rates, not the counts run
        # through a percentage converter
        ck('render: the witness count prints the rates as rates',
           any(f'{count["p_both"] * 100:.1f} %' in ln for ln in wtext) and
           not any(f'{100.0 * count["p_both"] / count["n_slices"]:.1f} %'
                   in ln for ln in wtext),
           'the rate and the count must not be confused')

        diff, dnum2 = render_differenced(fixed, roving, al2, t_mid,
                                         'shared-auto', 0.25)
        ck('render: the differenced view names the witness',
           any('WITNESSED' in line for line in diff) and
           any('witness mask' in line for line in diff))

        # 6b. the per-view scale is offered, and says it is a lie
        dual_pv, _ = render_dual_ascii(fixed, roving, al2, moments,
                                       'per-view-auto', 'change', 0.25)
        ck('render: the per-view scale says the panels are not comparable',
           any('NOT comparable' in line for line in dual_pv))

        # 7. the pair check: a missing stream is a precise blocker
        empty = Stream(tmp / 'nothing', 'missing')
        report = check_pair(fixed, empty, al2)
        ck('check: a missing stream is a precise blocker, not an absence',
           not report['renderable'] and report['blockers'], str(report))

        # 8. the OVERLAP IS IN THE ALIGNED CLOCK.  Two streams with
        #    DIFFERENT origins share no minutes until the alignment
        #    brings them together, and taking the overlap of the raw
        #    ranges would report that the pair has nothing in common.
        s1 = Stream(fixed.path, 'a')
        s2 = Stream(roving.path, 'b')
        al_shift = Alignment(offset_ms=500000.0)
        rep2 = check_pair(s1, s2, al_shift)
        ck('check: the overlap is computed in the ALIGNED clock',
           rep2['the_overlap_s'][0] > 100.0, str(rep2['the_overlap_s']))
        rep3 = check_pair(s1, s2, Alignment())
        ck('check: with no alignment the overlap is the RAW one',
           rep3['the_overlap_s'][1] <= rep2['the_overlap_s'][1] + 1.0,
           str(rep3['the_overlap_s']))

        # 9. the FLOOR-RELATIVE scale: each view in times its OWN
        #    sensor's grain, which is the only comparison two sensors
        #    in different units can make.
        fa = np.array([[6.0, 0.0]])
        rb = np.array([[60.0, 0.0]])
        f_stream = Stream(fixed.path, 'f')
        r_stream = Stream(roving.path, 'r')
        f_stream.floor, r_stream.floor = 3.0, 30.0
        na, nb, sc_a, sc_b = _scaled_pair(fa, rb, f_stream, r_stream,
                                          'floor-relative')
        ck('render: the floor-relative scale divides by each own floor',
           abs(na[0, 0] - 2.0) < 1e-9 and abs(nb[0, 0] - 2.0) < 1e-9,
           f'{na[0, 0]} {nb[0, 0]}')
        ck('render: the floor-relative scale is ONE scale for both',
           abs(sc_a - sc_b) < 1e-9, f'{sc_a} {sc_b}')

        # 10. the GRADED WITNESS and its slide null
        gw = graded_witness(fixed, roving, al2, 0.0, 6.0, 0.25)
        raw = gw.get('the_readings', {}).get('raw', {})
        ck('render: the graded witness reports the reading and its null',
           'the_correlation_at_the_alignment' in raw
           and 'the_null_sd' in raw and 'the_z_against_the_null' in raw,
           str(sorted(raw.keys())))
        ck('render: the graded witness slides the roving channel',
           raw.get('the_n_slides', 0) > 3
           and abs(raw.get('the_correlation_at_the_alignment', 0.0)) <= 1.0,
           str(raw.get('the_n_slides')))
        ck('render: the graded witness says whether the peak is sharp',
           'the_peak_sharpness_s' in raw, str(sorted(raw.keys())))

        # 12. the CROSSING WITNESS: the corpus's own crossing detector,
        #     with the gap set to each channel's OWN interval.  The
        #     tool's 0.35 s default is shorter than these channels'
        #     frame intervals, and then no blob ever chains.
        cw = crossing_witness(fixed, roving, al2, 0.0, 6.0)
        ck('render: the crossing witness sets the gap to TWO FRAMES of the '
           'channel',
           abs(cw.get('the_gaps_s', {}).get('fixed', 0)
               - 2.0 * fixed.interval_s) < 1e-6
           and abs(cw.get('the_gaps_s', {}).get('roving', 0)
                   - 2.0 * roving.interval_s) < 1e-6,
           str(cw.get('the_gaps_s')))
        ck('render: the crossing witness slides the roving channel',
           all(r.get('the_n_slides', 0) > 3
               for r in cw.get('the_readings', {}).values()),
           str(sorted(cw.get('the_readings', {}).keys())))
        ck('render: the crossing witness says how wrong the alignment can be',
           len(cw.get('the_alignment_sensitivity', [])) >= 3,
           str(len(cw.get('the_alignment_sensitivity', []))))

        # 11. the sidecar's OWN note is carried into the output
        note_path = tmp / 'note.json'
        note_path.write_text(json.dumps({'offset_ms': 12.0, 'drift_ppm': 0.0,
                                         'the_note': 'THE WALL CLOCK, NOT '
                                         'THE CLAPS'}))
        a_note = load_alignment(note_path, fixed, roving)
        ck('align: the sidecar\'s own note is carried into the output',
           'THE WALL CLOCK' in a_note.describe(),
           a_note.describe()[:120])

        # 12. a CHANGE-MAP STREAM is read: the phone's room-lens bundle
        from experiments import dual_ingest as di  # noqa: PLC0415
        fx = di.make_room_lens_fixture(tmp / 'roomlens.json')
        ph = tmp / 'phone'
        di.ingest_room_lens(tmp / 'roomlens.json', ph, quiet=True)
        ps = Stream(ph, 'the phone')
        ck('render: the change-map stream is read by the render lane',
           ps.ok and ps.is_change_map and ps.n_frames() == fx['the_n_frames'] - 1,
           str(ps.problems))
        ck('render: the change-map stream carries its own floor and units',
           abs(ps.floor - 255.0 / 16.0) < 1e-6 and 'room-lens' in ps.units,
           f'{ps.floor} {ps.units}')
        ck('render: the change-map stream\'s picture is on the corpus grid',
           ps.picture.shape[1:] == (GRID_H, GRID_W), str(ps.picture.shape))
        ck('render: the two change maps are not in the same units',
           'not in the same units' in ' '.join(_the_two_sensors(fixed, ps)),
           ' '.join(_the_two_sensors(fixed, ps))[:120])

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"   [{detail}]" if detail and not ok else ""))
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--fixed', default=None,
                    help='the fixed-witness stream directory (a session bundle)')
    ap.add_argument('--roving', default=None,
                    help='the roving-eye stream directory (a phone ingest)')
    ap.add_argument('--sync', default=None,
                    help='the clap alignment sidecar (zcode\'s lane)')
    ap.add_argument('--out', default=str(ROOT / 'docs' / 'research' / 'dual'))
    ap.add_argument('--mode', default='change', choices=('change', 'luma', 'both'))
    ap.add_argument('--scale', default='shared-auto',
                    choices=('shared-auto', 'shared-floor', 'shared-fixed',
                             'per-view-auto', 'floor-relative'),
                    help='shared-auto: one scale for both views (the default, '
                         'because a side-by-side of two scales is a lie). '
                         'per-view-auto: each view against its own maximum, '
                         'which the output labels as not comparable. '
                         'floor-relative: each view divided by its OWN noise '
                         'floor, so the two panels read in times this '
                         "sensor's own grain -- the only comparison two "
                         'sensors in different units can make')
    ap.add_argument('--map', default='mean', choices=('mean', 'max', 'sum'),
                    help='the window statistic: the mean change per frame '
                         '(the default), the strongest single frame, or the '
                         'raw sum')
    ap.add_argument('--window', type=float, default=0.25,
                    help='the half-window, in seconds, for a motion map')
    ap.add_argument('--slice', type=float, default=0.25,
                    help='the slice length, in seconds, for the fused score')
    ap.add_argument('--moments', type=int, default=6,
                    help='how many moments the dual-ASCII shows')
    ap.add_argument('--check', action='store_true',
                    help='report whether the pair is renderable and stop')
    ap.add_argument('--shifts', default='0,-250,-500,-1000,-2000,500,1000,2000',
                    help='the DELIBERATE alignment shifts, in ms, for the '
                         'sensitivity table -- how much of the witness count '
                         'is the room and how much is the clock')
    ap.add_argument('--synthetic', action='store_true',
                    help='render the SYNTHETIC pair (a person walking a known '
                         'path) instead of a real one -- this is what grades '
                         'the renders before the real pair lands')
    ap.add_argument('--note', default=None,
                    help='a line to print in the header of every output, for '
                         'saying what this pair IS and is not')
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    if args.synthetic:
        pair_dir = Path(args.out) / 'synthetic-pair'
        synth_pair(pair_dir)
        args.fixed = str(pair_dir / 'fixed')
        args.roving = str(pair_dir / 'roving')
        print(f'the synthetic pair is at {pair_dir} '
              f'(a person walking a known path; the truth is in '
              f'the_truth.json)')
    elif not args.fixed or not args.roving:
        print('both --fixed and --roving are required (or --selftest, or '
              '--synthetic)')
        return 1

    fixed = Stream(Path(args.fixed), 'fixed witness')
    roving = Stream(Path(args.roving), 'roving eye')
    align = load_alignment(Path(args.sync) if args.sync else None, fixed, roving)
    report = check_pair(fixed, roving, align)
    if args.check:
        print(json.dumps(report, indent=2))
        return 0 if report['renderable'] else 1
    if not report['renderable']:
        print('NOT RENDERABLE:')
        for b in report['blockers']:
            print('  -', b)
        print(json.dumps(report, indent=2))
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, t1 = report['the_overlap_s']
    span = max(0.0, t1 - t0)
    moments = [t0 + span * (i + 0.5) / max(1, args.moments)
               for i in range(args.moments)]

    blocks = []
    blocks.append('=' * 78)
    if args.synthetic:
        blocks.append('THE DUAL RENDER -- THE SYNTHETIC PAIR, NOT THE REAL ONE')
        blocks.append('  This is the tool\'s own fixture: a person walking a')
        blocks.append('  KNOWN path, seen by a fixed witness and a roving eye')
        blocks.append('  with a known viewpoint transform.  The truth is in')
        blocks.append('  synthetic-pair/the_truth.json.  It is here so the')
        blocks.append('  renders can be GRADED before the real pair lands --')
        blocks.append('  the real capture has NOT been made.')
    else:
        blocks.append('THE DUAL RENDER -- two cameras, two minutes, three ways')
    blocks.append(f'  generated {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}')
    if args.note:
        for line in args.note.splitlines():
            blocks.append(f'  {line}')
    blocks.append(f'  {align.describe()}')
    blocks.append(f'  overlap: {t0:+.2f} s .. {t1:+.2f} s '
                  f'({span:.2f} s of shared minutes)')
    blocks.append('=' * 78)
    blocks.append('')

    dual, dnum = render_dual_ascii(fixed, roving, align, moments,
                                    args.scale, args.mode, args.window,
                                    args.map)
    blocks.extend(dual)
    fused, fnum = render_fused_score(fixed, roving, align, t0, t1, args.slice)
    blocks.extend(fused)
    count = witness_count(fnum['slices'])
    blocks.extend(render_witness_count(count, args.slice))
    blocks.append('')
    graded = graded_witness(fixed, roving, align, t0, t1, args.slice)
    blocks.extend(render_graded_witness(graded, args.slice))
    blocks.append('')
    cross = crossing_witness(fixed, roving, align, t0, t1)
    blocks.extend(render_crossing_witness(cross))
    blocks.append('')
    shifts = [float(v) for v in str(args.shifts).split(',') if v.strip()]
    sens = alignment_sensitivity(fixed, roving, align, t0, t1, args.slice, shifts)
    blocks.extend(render_alignment_sensitivity(sens, args.slice))
    blocks.append('')
    mid = moments[len(moments) // 2] if moments else t0
    diff, dn2 = render_differenced(fixed, roving, align, mid, args.scale,
                                   args.window, args.map)
    blocks.extend(diff)

    text = '\n'.join(blocks)
    (out_dir / 'DUAL-RENDER.txt').write_text(text)
    summary = {
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'the_pair_is_synthetic': bool(args.synthetic),
        'the_pair': report,
        'the_dual_ascii': dnum,
        'the_fused_score': fnum,
        'the_witness_count': count,
        'the_graded_witness': graded,
        'the_crossing_witness': cross,
        'the_alignment_sensitivity': sens,
        'the_differenced_view': dn2,
        'the_rendering_choices': {
            'mode': args.mode, 'scale': args.scale, 'map': args.map,
            'window_s': args.window, 'slice_s': args.slice,
            'moments': len(moments),
        },
    }
    (out_dir / 'dual_render.json').write_text(json.dumps(summary, indent=2))
    print(text)
    print(f"\nwrote {out_dir / 'DUAL-RENDER.txt'}")
    print(f"wrote {out_dir / 'dual_render.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
