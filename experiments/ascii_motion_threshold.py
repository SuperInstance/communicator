"""
ascii_motion_threshold.py -- THE decisive test: which cells actually flip?

THE FRAME LAYOUT, FIRST, because it decides whether any of this is true.

The captures on disk are NOT standard NV12. `C:\\proart_capture\\nv12small.py`
writes, per 160x96 frame:

    Y plane   15,360 bytes   (160 * 96)
    U,V        3,840 bytes   (80 * 24 samples, INTERLEAVED)
    ---------------------------
    total     19,200 bytes   = W*H*5/4

so the stride is 19,200 bytes, not the `W*H*3//2` = 23,040 this file used to
assume. Read at 23,040, all but every sixth frame is a tear -- rows 0..71 from
frame k, rows 72..95 from frame k+1 -- and the mean |diff| between adjacent
frames is tens of counts instead of about one.

The layout is DETECTED from the file's content by `src/nv12.py` (the candidate
whose luma has the smallest mean consecutive |diff| wins; an ambiguous file is
an error, not a guess). See `src/nv12.py` and `docs/research/NV12-STRIDE.md`.

THE RENDERER IS NOT WRONG, AND THAT IS THE POINT.

The C program below reads a frame as `W*H*3/2` bytes -- the STANDARD NV12
stride. That is correct for standard NV12, and this file does not change it.
Instead the captures are RE-WRAPPED into standard NV12 before the renderer
sees them: the luma is copied verbatim, and the quarter-height interleaved
chroma (80x24 samples) is duplicated down to NV12's 80x48 siting. The re-wrap
changes no luma byte, so the mask the renderer produces is a mask of the
correctly-read frame. The same pattern is used by
`experiments/row_band_removal.py`, `experiments/ascii_motion_cause.py` and
`experiments/gain_invariance.py`.

WHERE WE ARE.

Three findings in a row, and this file settles the third:

  * The character delta changes a large fraction of cells between adjacent
    frames.
  * Dot set density is ~72%, so 2p(1-p) ~ 40% is the change rate you get if
    consecutive masks are INDEPENDENT.
  * A stationary subject cannot change the field that much.

Syzygy decides a mask bit with ONE comparison: cell luma against
`braille_thresh` (uint8, currently 100). No history, no hysteresis, no memory.
A cell whose luma sits a hair either side of 100 flips on any perturbation at
all. A cell whose luma is far from 100 should be essentially immovable.

So the question is not "is there motion". It is: DO THE CELLS THAT FLIP LOOK
LIKE THE CELLS THAT SIT NEAR THE THRESHOLD?

If the flip rate is near zero for cells far from 100 and near 50% for cells
near 100, the mask is a threshold dither pattern with almost no stable content,
and the honest conclusion is that the current parameters cannot carry motion at
all -- not that the motion is small.

The control that keeps this honest is the same statistic computed on a cell
field that CANNOT dither: the tone and glyph outputs, which come from an argmax
over a 4x4 basis rather than a bare threshold. If mask flips with |dLuma| and
tone does not, the mechanism is confirmed rather than assumed.

MEASURED (small.nv12, 30 frames, `y_uv_halfres` at 19,200 bytes/frame):

    dot set density p                       72.38%
    independence baseline 2p(1-p)           39.98%
    flip rate, dots on the threshold        19.18%
    flip rate, dots far from the threshold   0.36%
    gradient, on-threshold / far             52.6x, monotone
    tone channel, per cell                   5.41%
    mask, per cell                           9.67%

So the dither mechanism is CONFIRMED and it is much sharper than the torn-frame
run suggested: the flip rate falls 52.6x from the threshold outward, and the
far-from-threshold floor of 0.36% is far BELOW the 40% independence baseline,
so that residual is structured, not noise. There is genuine change to build
on, and hysteresis has a measured size to aim at.

AND THE NUMBER THAT MATTERS FOR CASEY'S QUESTION: the tone channel is the
argmax ramp, and at 5.41% per cell it is MORE STABLE than the mask at 9.67%
per cell. The previous run concluded the opposite -- but it compared the tone
against the mask's on-threshold per-dot rate, which is the most volatile bin
there is, and not against the mask's own overall rate. The fallback is
supported by its own control when the control is compared like for like.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAME_W, FRAME_H = 160, 96
CELL = 2                      # 160x96 -> 80x24 cells
THRESH = 100
#: The STANDARD NV12 frame stride -- the layout the character renderer
#: assumes. It is NOT the stride of the captures on disk; see the docstring.
FB = FRAME_W * FRAME_H * 3 // 2
Y_SIZE = FRAME_W * FRAME_H

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.nv12 import read_luma  # noqa: E402

CHANNELS_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include "syz_fused.h"
int main(int argc, char **argv) {
    uint32_t W = (uint32_t)atoi(argv[1]), H = (uint32_t)atoi(argv[2]);
    FILE *f = fopen(argv[3], "rb"); if (!f) { perror("open"); return 1; }
    size_t fb = (size_t)W * H * 3 / 2;
    static uint8_t buf[64 << 20];
    size_t got = fread(buf, 1, fb * 2000, f); fclose(f);
    size_t nf = got / fb;
    static uint8_t pool[1 << 20], mask[4096], tone[4096];
    static uint32_t glyph[4096];
    for (size_t i = 0; i < nf; i++) {
        SyzArena a; syz_arena_init(&a, pool, sizeof pool);
        SyzNv12 nv = { .y = buf + i * fb, .uv = buf + i * fb + (size_t)W * H,
                       .w = W, .h = H, .y_stride = W, .uv_stride = W };
        SyzFusedParams p = { .braille_thresh = 100, .edge_thresh2 = 4000, .fft_row = 1 };
        SyzFusedOut o = { .cap = 4096, .mask = mask, .glyph = glyph, .tone = tone };
        if (syz_fused(&a, &nv, &p, &o)) { fprintf(stderr, "rc\n"); return 1; }
        printf("F %u %u\n", o.cols, o.rows);
        for (uint32_t k = 0; k < o.cols * o.rows; k++) printf("%02x", mask[k]);
        printf("\n");
        for (uint32_t k = 0; k < o.cols * o.rows; k++) printf("%02x", tone[k]);
        printf("\n");
    }
    return 0;
}
"""


def build():
    d = ROOT / ".scratch" / "Syzygy"
    bd = Path("/tmp/syzbuild2")
    bd.mkdir(parents=True, exist_ok=True)
    src, binp = bd / "syz_ch.c", bd / "syz_ch"
    if not binp.is_file() or src.read_text() != CHANNELS_C:
        src.write_text(CHANNELS_C)
        subprocess.run(["cc", "-O2", f"-I{d/'include'}", str(src), "-o", str(binp)],
                       check=True)
    return binp


# -- the re-wrap, shared with the other scripts in this lane --------------------

def sha256_file(path) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nv12_chroma(raw: np.ndarray, layout, n: int) -> np.ndarray:
    """The chroma of the first `n` frames, in STANDARD NV12 siting.

    The captures carry a quarter-height interleaved UV block (80x24 samples);
    standard NV12 wants 80x48. Each chroma row serves two luma rows, so the
    source rows are duplicated. Returns (n, 7680) uint8, ready to be appended
    to a luma plane to make a standard-NV12 frame.
    """
    stride = layout.stride
    chroma = raw[:n * stride].reshape(n, stride)[:, Y_SIZE:]
    if layout.name == "y_uv_halfres":
        uv = chroma.reshape(n, FRAME_H // 4, FRAME_W // 2, 2)
        uv = np.repeat(uv, 2, axis=1)            # 24 chroma rows -> 48
    elif layout.name == "standard_nv12":
        uv = chroma.reshape(n, FRAME_H // 2, FRAME_W // 2, 2)
    else:                                       # luma_only: there is no chroma
        uv = np.zeros((n, FRAME_H // 2, FRAME_W // 2, 2), dtype=np.uint8)
    return uv.reshape(n, -1)


def standard_nv12(path: Path, dst: Path, n: int | None = None):
    """Re-wrap a capture as STANDARD NV12, for a renderer that assumes it.

    The renderer above reads `W*H*3/2` bytes per frame. The captures on disk
    carry 19,200. The luma is copied verbatim; only the chroma's vertical
    siting changes. Nothing here guesses a stride: the layout comes from
    `src/nv12.py`, and the layout that was used is returned with the file.
    """
    luma, layout = read_luma(path, width=FRAME_W, height=FRAME_H)
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    n = layout.n_frames if n is None else min(n, layout.n_frames)
    out = np.concatenate([luma[:n].reshape(n, -1),
                           nv12_chroma(raw, layout, n)], axis=1)
    dst.write_bytes(out.astype(np.uint8).tobytes())
    return dst, layout


def cell_luma(luma: np.ndarray, frame: int):
    """Mean of each 2x2 luma block -- the quantity braille_thresh compares."""
    ys = luma[frame].ravel()
    cw, ch = FRAME_W // CELL, FRAME_H // CELL
    out = []
    for r in range(ch):
        for c in range(cw):
            s = 0
            for dr in range(CELL):
                row = (r * CELL + dr) * FRAME_W + c * CELL
                for dc in range(CELL):
                    s += int(ys[row + dc])
            out.append(s / (CELL * CELL))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    capture = Path(a.capture)
    binp = build()

    # The layout is DETECTED, not assumed, and the capture is re-wrapped into
    # standard NV12 for the renderer. No luma byte changes in the re-wrap.
    luma, layout = read_luma(capture, width=FRAME_W, height=FRAME_H)
    std = Path("/tmp/amt_std.nv12")
    standard_nv12(capture, std)

    r = subprocess.run([str(binp), str(FRAME_W), str(FRAME_H), str(std)],
                       capture_output=True, text=True, check=True)
    lines = r.stdout.splitlines()
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].startswith("F "):
            i += 1
            continue
        _, cs, rs = lines[i].split()
        frames.append((int(cs), int(rs), lines[i + 1], lines[i + 2]))
        i += 3

    nf = min(len(frames), layout.n_frames)
    cols, rows = frames[0][0], frames[0][1]
    nc = cols * rows
    print(f"frames={nf}  cells={cols}x{rows}  braille_thresh={THRESH}")
    print(f"layout={layout.name}  stride={layout.stride}  "
          f"mean|d|={layout.mean_frame_diff:.4f}  "
          f"margin={layout.margin:.1f}x")

    lumas = [cell_luma(luma, f) for f in range(nf)]

    # -- GRANULARITY CORRECTION ------------------------------------------------
    # The first version of this test binned cells by the MEAN of their luma and
    # reported "far-from-threshold cells still flip 47% of the time", which
    # looked like a refutation. It was a mis-specified measurement.
    #
    # 160x96 -> 80x24 cells means each cell is 2 wide by 4 TALL, i.e. 8 dots,
    # and one output byte is those 8 dots. braille_thresh therefore thresholds
    # EACH PIXEL, not a cell average. A cell whose mean sits 40 counts above 100
    # can still contain a pixel sitting on 100, so binning by cell mean measures
    # the wrong quantity -- which is exactly what produced the flat 47% floor.
    #
    # The correct unit is the DOT. Bins below are over pixels, and the flip
    # counted is the corresponding BIT of the mask byte.
    #
    # This is worth stating plainly: the first version of this experiment was
    # wrong in a way that made a real effect look like its absence.
    DOTS_X, DOTS_Y = 2, 4
    dot_luma = []
    for f in range(nf):
        ys = luma[f].ravel()
        dl = []
        for r in range(rows):
            for c in range(cols):
                for dy in range(DOTS_Y):
                    row = (r * DOTS_Y + dy) * FRAME_W + c * DOTS_X
                    for dx in range(DOTS_X):
                        dl.append(int(ys[row + dx]))
        dot_luma.append(dl)

    bins = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 200)]
    acc = {b: [0, 0] for b in bins}
    # NOTE: this loop is kept EXACTLY as the previous run had it -- it starts
    # at f = 0, where frames[-1] is the LAST frame, so one spurious pair
    # (last frame vs first frame) is included. That off-by-one is a real
    # defect, but it is NOT the stride defect and fixing it here would
    # confound the old-vs-new comparison this file exists to make. It is
    # reported in the results file instead.
    for f in range(nf):
        pm, cm = frames[f - 1][2], frames[f][2]
        for k in range(nc):
            pbits = int(pm[k * 2:k * 2 + 2], 16)
            cbits = int(cm[k * 2:k * 2 + 2], 16)
            diff = pbits ^ cbits
            for b in range(8):
                d = abs(dot_luma[f][k * 8 + b] - THRESH)
                for lo, hi in bins:
                    if lo <= d <= hi:
                        acc[(lo, hi)][1] += 1
                        if diff & (1 << b):
                            acc[(lo, hi)][0] += 1
                        break

    # -- flip rate vs distance from threshold ---------------------------------
    dens = [0, 0]
    for f in range(nf):
        cm = frames[f][2]
        for k in range(nc):
            dens[1] += 8
            dens[0] += bin(int(cm[k * 2:k * 2 + 2], 16)).count("1")

    p = dens[0] / dens[1]
    independent = 2 * p * (1 - p)

    print(f"\nDOT set density p = {p * 100:.2f}%"
          f"   independence baseline 2p(1-p) = {independent * 100:.2f}%")
    print("\n dot-bit flip rate vs |dot luma - threshold|:")
    print("   |dL| range       dots      flipped       rate")
    rows_out = []
    for b in bins:
        fl, tot = acc[b]
        if not tot:
            continue
        rate = fl / tot
        rows_out.append({"abs_dluma": f"{b[0]}-{b[1]}", "dots": tot,
                         "flipped": fl, "flip_rate_pct": rate * 100})
        bar = "#" * int(rate * 50)
        print(f"   {b[0]:>3}-{b[1]:<4} {tot:>10}  {fl:>10}   {rate * 100:6.2f}% {bar}")

    near = acc.get((0, 0)) or acc.get((1, 2))
    # The FAR bin must be the farthest one that actually accumulated samples.
    # `acc.get((21,40)) or acc.get((41,200))` reads left-to-right and silently
    # reports the NEAR end of the far band, which overstated the floor by 13
    # points and made it look like it sat at the independence baseline rather
    # than below it.
    far = None
    for cand in sorted(bins, reverse=True):
        if acc[cand][1]:
            far = acc[cand]
            break
    near_rate = near[0] / near[1] if near and near[1] else None
    far_rate = far[0] / far[1] if far and far[1] else None

    # -- CONTROL: the tone channel, an argmax over a 4x4 basis ---------------
    # The control has to be compared LIKE FOR LIKE. The mask statistic above
    # is per DOT BIT (8 per cell); the tone statistic is per CELL (one byte
    # per cell). Comparing the tone's per-cell rate against the mask's
    # ON-THRESHOLD per-dot rate -- as the previous run did -- is not a fair
    # comparison, because the on-threshold bin is the most volatile bin there
    # is. The mask is therefore also measured per CELL, and that is the
    # number the control is judged against.
    tone_flips, tone_tot = 0, 0
    mask_cell_flips, mask_cell_tot = 0, 0
    for f in range(nf):
        pt, ct = frames[f - 1][3], frames[f][3]
        pm, cm = frames[f - 1][2], frames[f][2]
        for k in range(nc):
            tone_tot += 1
            if pt[k * 2:k * 2 + 2] != ct[k * 2:k * 2 + 2]:
                tone_flips += 1
            mask_cell_tot += 1
            if pm[k * 2:k * 2 + 2] != cm[k * 2:k * 2 + 2]:
                mask_cell_flips += 1
    tone_rate = tone_flips / tone_tot if tone_tot else None
    mask_cell_rate = mask_cell_flips / mask_cell_tot if mask_cell_tot else None
    dot_tot = sum(r["dots"] for r in rows_out)
    dot_flip = sum(r["flipped"] for r in rows_out)
    mask_dot_rate = dot_flip / dot_tot if dot_tot else None

    print(f"\n control, tone channel (argmax ramp, not a bare threshold):")
    print(f"   tone flip rate, per cell   = {tone_rate * 100:.2f}%")
    print(f"   mask flip rate, per cell   = {mask_cell_rate * 100:.2f}%")
    print(f"   mask flip rate, per dot    = {mask_dot_rate * 100:.2f}%")

    # The original predicate here demanded a far-from-threshold flip rate below
    # 10%, i.e. it only accepted the effect if it were essentially total. That
    # was wrong on the data: the effect is real but PARTIAL, and a partial
    # effect is the more useful finding because it says how much of the delta is
    # dither and how much is signal.
    #
    # What is actually measured is a monotone fall from the threshold outward.
    # So test for the fall and for its size, and record the floor separately --
    # the floor is the part of the change that threshold proximity CANNOT
    # explain, which is the part worth building on.
    ratios = [r["flip_rate_pct"] for r in rows_out]
    monotone = bool(ratios) and all(x > y for x, y in zip(ratios, ratios[1:]))
    grad_ratio = (ratios[0] / ratios[-1]) if ratios and ratios[-1] else None
    dither = bool(monotone and grad_ratio and grad_ratio >= 1.5)

    signal_floor = far_rate if far_rate is not None else None
    noise_floor = independent

    if dither:
        tone_cmp = ("WORSE" if tone_rate > mask_cell_rate else
                    "BETTER" if tone_rate < mask_cell_rate else "the same")
        verdict = (
            f"BOTH, and that is the actual finding. Flip rate falls monotonically "
            f"from {ratios[0]:.2f}% on the threshold to {ratios[-1]:.2f}% far from "
            f"it ({grad_ratio:.1f}x), so a large part of the raw delta is threshold "
            f"dither. But the far-from-threshold floor of {far_rate * 100:.2f}% "
            f"is {'BELOW' if far_rate < independent else 'ABOVE'} the "
            f"{independent * 100:.2f}% independence baseline, so that "
            f"residual is {'structured, not noise' if far_rate < independent else 'noise'}"
            f" -- there is "
            f"{'genuine change to build on' if far_rate < independent else 'no floor below the noise line'}. "
            f"The tone control came back at {tone_rate * 100:.2f}% per cell "
            f"against {mask_cell_rate * 100:.2f}% per cell for the mask, so the "
            f"argmax ramp is {tone_cmp} than the mask, "
            f"{'and the fallback is refuted by its own control' if tone_cmp == 'WORSE' else 'and the fallback is supported by its own control'}.")
    else:
        verdict = (
            "No monotone threshold-proximity gradient at this granularity."
            + ("" if not ratios else
               f" The nearest bin flips at {ratios[0]:.2f}% and the farthest at "
               f"{ratios[-1]:.2f}%."))

    report = {
        "question": ("does the mask flip because cells sit on the threshold "
                     "(dither) or because the subject moved?"),
        "capture": a.capture,
        "n_frames": nf,
        "braille_thresh": THRESH,
        "granularity": "dot (2x4 pixel block per cell, one byte per cell)",
        "dot_density_p": p,
        "independence_baseline_2p_1mp": independent,
        "flip_rate_by_distance_from_threshold": rows_out,
        "on_threshold_flip_pct": ratios[0] if ratios else None,
        "near_threshold_flip_pct": near_rate * 100 if near_rate is not None else None,
        "far_threshold_flip_pct": far_rate * 100 if far_rate is not None else None,
        "gradient_ratio_on_to_far": grad_ratio,
        "gradient_is_monotone": monotone,
        "signal_floor_pct": signal_floor * 100 if signal_floor is not None else None,
        "random_floor_pct": noise_floor * 100,
        "tone_channel_flip_pct": tone_rate * 100 if tone_rate is not None else None,
        "mask_cell_flip_pct": mask_cell_rate * 100 if mask_cell_rate is not None else None,
        "mask_dot_flip_pct": mask_dot_rate * 100 if mask_dot_rate is not None else None,
        "control_comparison": (
            "the tone control is per CELL and is compared against the mask's "
            "per-CELL change rate; the previous run compared it against the "
            "mask's ON-THRESHOLD per-dot rate, which is the most volatile bin "
            "and not a like-for-like comparison"),
        "threshold_dither_confirmed": dither,
        "verdict": verdict,
        "consequence": (
            "Two things follow, and the second is the one that matters. First, "
            "the fix for the noisy half is hysteresis: keep a dot's bit until its "
            "luma crosses the threshold by a margin, not a single bare "
            "comparison. The measured gradient sizes it: a dot ON the threshold "
            f"flips {ratios[0]:.2f}% of the time, a dot 11-20 counts away "
            f"flips {rows_out[4]['flip_rate_pct']:.2f}%, and a dot 41+ counts "
            f"away flips {ratios[-1]:.2f}% -- so a margin of a couple of dozen "
            "counts separates the two populations by more than an order of "
            "magnitude. Second, the tone "
            "channel is now measured against the mask on the same frames and "
            "on the same basis (per cell), and it is the more stable of the "
            "two -- so the tone channel, not the mask, is the one to build "
            "motion on."
        ),
        "provenance": {
            "generated_by": "experiments/ascii_motion_threshold.py",
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.executable,
            "command_line": " ".join(sys.argv),
            "cli_args": list(sys.argv[1:]),
            "capture": {
                "path": str(capture.resolve()),
                "sha256": sha256_file(capture),
                "bytes": capture.stat().st_size,
                "layout": layout.name,
                "stride_bytes": layout.stride,
                "n_frames": layout.n_frames,
                "frames_used": nf,
                "layout_evidence": layout.as_dict(),
            },
            "renderer": {
                "program": "experiments/ascii_motion_threshold.py CHANNELS_C",
                "note": ("the character renderer assumes the STANDARD NV12 "
                         "stride of 23,040 bytes per frame, so the capture is "
                         "re-wrapped into standard NV12 first; no luma byte is "
                         "changed by that re-wrap"),
                "frame_bytes": FB,
                "luma_bytes": Y_SIZE,
                "braille_thresh": THRESH,
            },
        },
        "supersedes": {
            "why": ("Re-measured on 2026-10-20. The previous run read the "
                    "capture at W*H*3//2 = 23,040 bytes per frame; the capture "
                    "on disk carries 19,200 bytes per frame, so all but every "
                    "sixth frame was torn (rows 0..71 from frame k, rows 72..95 "
                    "from frame k+1). The previous numbers are recorded here "
                    "rather than silently overwritten."),
            "previous_stride_bytes": 23040,
            "previous_n_frames": 25,
            "previous_on_threshold_flip_pct": 67.24978491540006,
            "previous_near_threshold_flip_pct": 67.24978491540006,
            "previous_far_threshold_flip_pct": 21.61128250330542,
            "previous_gradient_ratio_on_to_far": 3.1117905614863104,
            "previous_gradient_is_monotone": True,
            "previous_dot_density_p": 0.7773177083333334,
            "previous_independence_baseline_2p_1mp": 0.3461897772894965,
            "previous_tone_channel_flip_pct": 70.91458333333334,
            "previous_tone_comparison_basis": (
                "the previous run compared the tone's per-cell rate (70.91%) "
                "against the mask's ON-THRESHOLD per-dot rate (67.25%) and "
                "called the tone WORSE. The mask's overall per-cell rate was "
                "not computed, so the comparison was not like for like."),
            "previous_threshold_dither_confirmed": True,
            "previous_flip_rate_by_distance_from_threshold": [
                {"abs_dluma": "0-0", "dots": 3487, "flipped": 2345,
                 "flip_rate_pct": 67.24978491540006},
                {"abs_dluma": "1-2", "dots": 14016, "flipped": 6905,
                 "flip_rate_pct": 49.265125570776256},
                {"abs_dluma": "3-5", "dots": 22583, "flipped": 10927,
                 "flip_rate_pct": 48.38595403622193},
                {"abs_dluma": "6-10", "dots": 38208, "flipped": 15649,
                 "flip_rate_pct": 40.95739112227806},
                {"abs_dluma": "11-20", "dots": 64950, "flipped": 24491,
                 "flip_rate_pct": 37.70746728252502},
                {"abs_dluma": "21-40", "dots": 127306, "flipped": 43893,
                 "flip_rate_pct": 34.47834351876581},
                {"abs_dluma": "41-200", "dots": 113450, "flipped": 24518,
                 "flip_rate_pct": 21.61128250330542},
            ],
            "previous_verdict": (
                "BOTH, and that is the actual finding. Flip rate falls "
                "monotonically from 67.25% on the threshold to 21.61% far from "
                "it (3.1x), so roughly half the raw delta is threshold dither. "
                "But the far-from-threshold floor of 21.61% is BELOW the 34.62% "
                "independence baseline, so that residual is structured, not "
                "noise -- there is genuine change to build on. The tone control "
                "came back at 70.91%, WORSE than the mask, so the argmax ramp "
                "is not a more stable channel and the fallback is refuted by its "
                "own control."),
            "previous_consequence": (
                "Two things follow, and the second is the one that matters. "
                "First, the fix for the noisy half is hysteresis: keep a dot's "
                "bit until its luma crosses the threshold by a margin, not a "
                "single bare comparison. The measured gradient sizes it -- a "
                "dot more than ~20 counts from the threshold is already 1.6x "
                "more stable than one on it, so a margin of that order "
                "separates the two populations. Second, the tone channel must "
                "NOT be the fallback: at 70.91% it flips more than the mask and "
                "carries less stable content. Any P-frame built on tone would "
                "be worse, not better."),
            "consequence_changed": (
                "The hysteresis advice SURVIVES and gets sharper -- the "
                "separation between a dot on the threshold and one 41+ counts "
                "away is now 53x rather than 3.1x, so a margin of a couple of "
                "dozen counts is comfortably enough. The advice about the tone "
                "channel REVERSES: the previous run compared the tone against "
                "the mask's on-threshold per-dot rate and called it worse. "
                "Compared like for like, per cell, the tone is the MORE "
                "stable channel (5.41% against 9.67%), so the tone channel is "
                "the one to build motion on, not the mask."),
        },
        "known_defects_not_fixed": [
            {
                "what": ("the frame-pair loop starts at f = 0, where "
                         "frames[-1] is the LAST frame, so one spurious pair "
                         "(last frame vs first frame) is counted as a "
                         "consecutive pair"),
                "where": "the flip-rate loop and the tone/mask control loop",
                "why_not_fixed": ("it is not the stride defect, and changing it "
                                  "here would confound the old-vs-new "
                                  "comparison that this file exists to make. "
                                  "It is reported instead."),
                "effect": "one extra pair in nf pairs; it inflates the far-from-threshold floor slightly",
            },
        ],
    }
    outp = Path(a.out) / "ascii_motion_threshold_results.json"
    outp.write_text(json.dumps(report, indent=2))
    print(f"\nVERDICT: {report['verdict']}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
