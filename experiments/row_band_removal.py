"""
row_band_removal.py -- remove the banding without blurring time.

THE FRAME LAYOUT, BEFORE ANYTHING ELSE.

The captures this reads are NOT standard NV12. `C:\\proart_capture\\nv12small.py`
writes, per 160x96 frame:

    Y plane    15,360 bytes   (160 * 96)
    U,V         3,840 bytes   (80 x 24 samples, INTERLEAVED)
    -------------------------
    total      19,200 bytes   = W*H*5/4

so the stride is 19,200 bytes, not the `W*H*3//2` = 23,040 this file used to
assume. Read at 23,040, a 19,200-byte file yields 750 frames of which only
every sixth is intact -- rows 0..71 from frame k and rows 72..95 from frame
k+1 -- and the "banding" that measured was that tear.

The layout is therefore DETECTED from the file's content by `src/nv12.py` (the
candidate whose luma has the smallest mean consecutive |diff| wins; an
ambiguous file is an error, not a guess) and not computed here. See
`src/nv12.py` and `docs/research/NV12-STRIDE.md`.

The character renderer is a separate program whose frame stride IS the
standard-NV12 23,040, so the frames handed to it are re-wrapped into standard
NV12 first. That re-wrap changes no luma byte.

WHAT CHANGED, AND WHAT THE EXPERIMENT NOW SAYS.

The premise of this experiment was a saturated flip rate: 50.00% of dots
changing per frame in an EMPTY room, which no real scene produces. Re-measured
at the correct stride, the empty room flips 0.00% of dots per frame with NO
removal at all, and the occupied capture flips 0.62%. The saturation was the
torn-frame artefact, not the camera. So the conclusion of the previous run --
"row removal works and beats the temporal median" -- does NOT survive: there
is no longer a nuisance term for row removal to beat, and applying it makes
the occupied rate worse rather than better.

The predictions further down were stated against the old numbers and are kept
as written, because they are the record of what was believed and why the
belief was wrong. The numbers this file prints are the ones in
`experiments/row_band_removal_results.json`, and that file records the
superseded values alongside them.

WHY THIS IS THE RIGHT SHAPE OF FIX.

`CAMERA-FLICKER-BANDING.md` measured the nuisance term precisely: in the empty
room the residual between adjacent frames has a COLUMN ENERGY SHARE OF 0.00%.
The banding is identical in every column. That is a very strong structural
statement, and it says the nuisance lives in a space we can measure
separately from the signal:

    frame(x, y) = content(x, y) + band(y)

`band(y)` is a per-row offset (or per-row gain). `content(x, y)` is everything
else. A room feature, a face, a body -- all of those vary across x. The
banding does not vary across x AT ALL.

So the per-row statistic ACROSS COLUMNS is a direct estimate of `band(y)`:
take the median of each row over the 160 columns. The median rather than the
mean because a subject occupies a minority of a row's columns and must not drag
the estimate; the median ignores it. Then subtract (or divide out) the row
offset and the banding is gone while the content is untouched.

WHY THIS IS BETTER THAN THE TEMPORAL MEDIAN, AND NOT JUST DIFFERENT.

The temporal median (`temporal_median_fix.py`) removes the banding -- empty room
50.00% -> 0.00% -- but it does so by averaging over TIME, and the price is
visible and measured:

  * it costs (k-1)/30 s of latency, on a system whose premise is watching
    continuously;
  * it smears real fast motion by the same amount;
  * and at k=5 it drives the OCCUPIED rate to 0.48%, i.e. it destroys the
    signal as thoroughly as it destroys the nuisance. That collapse is the
    proof that it is not discriminating -- it is cancelling whatever is
    periodic, whether nuisance or signal.

Row-wise removal is not a temporal filter at all. It has NO latency, it does
not blur motion, and it cannot smear a subject because it never combines
frames. If it works, it is strictly better on every axis the temporal median
loses on.

THE PREDICTION, STATED FIRST.

  * EMPTY room: flip rate collapses to near zero, as the temporal median did.
  * OCCUPIED: stays HIGH -- unlike the temporal median, which fell to 0.48% at
    k=5. If occupied also collapses, then the occupied signal is itself
    row-constant and this approach has the same failure.
  * FACING vs HEAD-TURNED: if the two are separable after row removal, the
    original three-condition capture is RECOVERED, because the difference
    between them is a column-structured change and this method is designed to
    keep exactly that.

Both additive and multiplicative correction are tested, because rolling-shutter
flicker is arguably a per-row GAIN change (each row is exposed for a different
length of time) and the correct model is not obvious a priori. Measuring both
is cheaper than arguing about it.

HONEST LIMITS, STATED BEFORE THE NUMBERS.

  * The row estimate is contaminated by the subject: where a subject fills more
    than half a row's columns, the median drifts toward the subject's own
    brightness. A person close to the camera could therefore be partly removed
    BY this filter. That is a real risk and it is why the occupied result
    matters as much as the empty one.
  * The two captures are hours apart and differ in overall lighting. Only
    within-condition comparisons are clean.
  * One capture per condition. No error bars.
  * Restoring the original absolute luma is not possible, only the row-
    structured part of the change. The threshold therefore applies to a
    reconstructed image, and the reconstruction is reported.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAME_W, FRAME_H = 160, 96
#: The STANDARD NV12 frame stride -- the layout the character renderer assumes.
#: It is NOT the stride of the captures on disk; see the module docstring.
FB = FRAME_W * FRAME_H * 3 // 2
Y_SIZE = FRAME_W * FRAME_H
THRESH = 100
sys.path.insert(0, str(ROOT / "experiments"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ascii_motion_threshold import build  # noqa: E402
from src.nv12 import read_luma  # noqa: E402


def sha256_file(path) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nv12_chroma(raw: np.ndarray, layout, n: int) -> np.ndarray:
    """The chroma of the first `n` frames, in STANDARD NV12 siting.

    The captures carry a quarter-height interleaved UV block; NV12 wants
    80x48 UV pairs. Each chroma row serves two luma rows, so the source rows
    are duplicated. Returns (n, 7680) uint8, ready to be appended to a luma
    plane to make a standard-NV12 frame.
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


def load(path: Path, nf: int):
    """(luma, chroma, layout) for the first `nf` frames of a capture.

    The layout is detected, not assumed -- see `src/nv12.py`.
    """
    luma, layout = read_luma(path, width=FRAME_W, height=FRAME_H)
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    n = min(nf, layout.n_frames)
    return luma[:n], nv12_chroma(raw, layout, n), layout


def row_profile(F):
    """Per-row median across columns: an estimate of band(y)."""
    return np.median(F.astype(np.float64), axis=2)          # (n, H)


def remove_rows(F, mode="subtract", clip_to=None):
    """Remove the row-constant component. F is (n, H, W) float."""
    prof = row_profile(F)
    gmean = prof.mean(axis=1, keepdims=True)                # (n, 1)
    if mode == "subtract":
        out = F - (prof - gmean)[:, :, None]
    elif mode == "divide":
        # per-row gain: F(x,y) ~ content * gain(y) -> divide by gain(y)
        gain = prof / (gmean + 1e-6)
        out = F / gain[:, :, None]
    else:
        out = F
    if clip_to is not None:
        out = np.clip(out, 0, clip_to)
    return out, prof


def flip(binp: Path, tmp: Path, nf: int):
    r = subprocess.run([str(binp), str(FRAME_W), str(FRAME_H), str(tmp)],
                       capture_output=True, text=True, check=True)
    lines = r.stdout.splitlines()
    masks, i = [], 0
    while i < len(lines):
        if not lines[i].startswith("F "):
            i += 1
            continue
        masks.append(lines[i + 1])
        i += 2
    parts = lines[0].split()
    nc = int(parts[1]) * int(parts[2])
    B = np.stack([((np.frombuffer(bytes.fromhex(h), np.uint8)[:, None]
                    >> np.arange(8)) & 1).astype(np.uint8) for h in masks[:nf]])
    x = B[1:] ^ B[:-1]
    per = x.reshape(len(x), -1).mean(axis=1)
    return float(np.median(per) * 100), float(per.min() * 100), float(per.max() * 100)


def run(binp: Path, F, UV, mode, ref_mean):
    nf = len(F)
    base = F.astype(np.float64)
    if mode == "none":
        rec = base
    else:
        rec, _ = remove_rows(base, mode)
        # rescale so the mean luma is back at a comparable level; the
        # reconstruction has removed an offset, and the threshold needs a
        # sensible absolute scale.
        m = rec.reshape(nf, -1).mean(axis=1)[:, None, None]
        rec = rec * (ref_mean / (m + 1e-9))
    rec = np.clip(rec, 0, 255).astype(np.uint8)
    tmp = Path("/tmp/rb.nv12")
    tmp.write_bytes(np.concatenate(
        [rec.reshape(nf, Y_SIZE), UV[:nf].reshape(nf, FB - Y_SIZE)], axis=1
    ).astype(np.uint8).tobytes())
    return flip(binp, tmp, nf), rec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--occupied", default="/mnt/c/proart_capture/A.nv12")
    ap.add_argument("--turned", default="/mnt/c/proart_capture/B.nv12")
    ap.add_argument("--empty", default="/home/proart/projects/glasses/.capture_keep/empty1.nv12")
    ap.add_argument("--nf", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    binp = build()
    caps = {}
    inputs = {}
    for p, lab in ((a.occupied, "occupied"), (a.turned, "turned"),
                   (a.empty, "empty")):
        if Path(p).is_file():
            F, UV, layout = load(Path(p), a.nf)
            caps[lab] = (F, UV)
            inputs[lab] = {
                "path": str(Path(p).resolve()),
                "sha256": sha256_file(p),
                "bytes": Path(p).stat().st_size,
                "layout": layout.name,
                "stride_bytes": layout.stride,
                "n_frames": layout.n_frames,
                "frames_used": int(len(F)),
                "layout_evidence": layout.as_dict(),
            }

    res = {}
    for mode in ("none", "subtract", "divide"):
        row = {}
        for lab, (F, UV) in caps.items():
            ref = float(np.median(F.reshape(len(F), -1).mean(axis=1)))
            (fl, lo, hi), _ = run(binp, F, UV, mode, ref)
            row[lab] = {"flip_pct": fl, "min": lo, "max": hi}
        res[mode] = row
        e = row.get("empty", {}).get("flip_pct")
        o = row.get("occupied", {}).get("flip_pct")
        t = row.get("turned", {}).get("flip_pct")
        gap = abs(o - t) if (o is not None and t is not None) else None
        print(f"  {mode:9s} empty {e if e is None else round(e,2):>6}"
              f"   occupied {o if o is None else round(o,2):>6}"
              f"   turned {t if t is None else round(t,2):>6}"
              f"   face/turn gap {gap if gap is None else round(gap,2):>6}")

    best = None
    for mode in ("subtract", "divide"):
        e = res[mode].get("empty", {}).get("flip_pct")
        o = res[mode].get("occupied", {}).get("flip_pct")
        if e is None or not o:
            continue
        if e < 5 and o > 5:
            if best is None or o > best[1]:
                best = (mode, o, e)

    out = {"question": ("can row-wise removal remove the banding without "
                        "blurring time or erasing content?"),
           "method": ("per-row MEDIAN across the 160 columns estimates "
                      "band(y); the row-constant component is removed, then the "
                      "frame is rescaled to the capture's median mean luma"),
           "results": res,
           "best_mode": best,
           "limits": [
               "the row estimate is contaminated where a subject fills more "
               "than half a row's columns -- the filter can partly remove a "
               "close subject, which is why the occupied number matters",
               "the empty capture is later and darker; only within-condition "
               "comparisons are clean",
               "one capture per condition, no error bars",
               "the reconstruction cannot restore the original absolute luma, "
               "only the row-structured part of the change, so the threshold "
               "is applied to a reconstructed image",
               "the temporal-median comparison numbers quoted in the verdict "
               "(0.48% at k=5) come from experiments/temporal_median_fix.py, "
               "which reads its capture at the OLD 23,040 stride and has not "
               "been re-run. They are quoted as they stand, not re-measured "
               "here.",
           ],
           "supersedes": {
               "why": ("These numbers were RE-MEASURED on 2026-10-20. The "
                       "previous run read the captures at W*H*3//2 = 23,040 "
                       "bytes per frame, but the captures on disk carry 19,200 "
                       "bytes per frame, so all but every sixth frame was torn "
                       "(rows 0..71 from frame k, rows 72..95 from frame k+1). "
                       "The previous numbers are recorded here rather than "
                       "silently overwritten."),
               "previous_stride_bytes": 23040,
               "previous_results": {
                   "none": {"occupied": 48.190104166666664,
                            "turned": 40.345052083333336,
                            "empty": 50.0},
                   "subtract": {"occupied": 48.56770833333333,
                                "turned": 44.921875,
                                "empty": 0.013020833333333334},
                   "divide": {"occupied": 49.303385416666664,
                              "turned": 46.184895833333336,
                              "empty": 0.05859375},
               },
               "previous_best_mode": ["divide", 49.303385416666664, 0.05859375],
               "previous_verdict": (
                   "ROW REMOVAL WORKS AND BEATS THE TEMPORAL MEDIAN. Mode "
                   "'divide': empty room 0.06%, occupied 49.30%. The occupied "
                   "rate STAYS HIGH, which is the thing the temporal median "
                   "could not do -- at k=5 it fell to 0.48%. No latency, no "
                   "motion smearing, no frame averaging. Facing vs head-turned "
                   "gap is 3.12 pp."),
           },
           "provenance": {
               "generated_by": "experiments/row_band_removal.py",
               "script_path": str(Path(__file__).resolve()),
               "script_sha256": sha256_file(Path(__file__).resolve()),
               "python": sys.executable,
               "command_line": " ".join(sys.argv),
               "cli_args": list(sys.argv[1:]),
               "captures": inputs,
               "renderer": {
                   "program": "experiments/ascii_motion_threshold.py CHANNELS_C",
                   "note": ("the character renderer assumes the STANDARD NV12 "
                            "stride of 23,040 bytes per frame, so the frames it "
                            "is given are re-wrapped into standard NV12 first; "
                            "no luma byte is changed by that re-wrap"),
                   "frame_bytes": FB,
                   "luma_bytes": Y_SIZE,
                   "braille_thresh": THRESH,
               },
               "frames_requested": a.nf,
           }}

    if best:
        mode, o, e = best
        gap = None
        t = res[mode].get("turned", {}).get("flip_pct")
        if t is not None:
            gap = abs(o - t)
        out["verdict"] = (
            f"ROW REMOVAL WORKS AND BEATS THE TEMPORAL MEDIAN. Mode "
            f"'{mode}': empty room {e:.2f}%, occupied {o:.2f}%. The occupied "
            f"rate STAYS HIGH, which is the thing the temporal median could "
            f"not do -- at k=5 it fell to 0.48%. No latency, no motion "
            f"smearing, no frame averaging."
            + (f" Facing vs head-turned gap is {gap:.2f} pp."
               if gap is not None else ""))
    else:
        # THE PREMISE CHECK, and it is the whole result.
        #
        # This experiment was written to remove a saturated flip rate: 50.00%
        # of dots per frame in an EMPTY room, which no real scene produces.
        # Re-measured at the correct frame stride, the empty room flips
        # 0.00% of dots per frame with NO removal at all. There is no banding
        # to remove: the saturation was the torn-frame artefact of reading a
        # 19,200-byte frame at a 23,040-byte stride. The conclusion of the
        # previous run -- "row removal works and beats the temporal median" --
        # does not survive; there is no longer a nuisance term for it to beat.
        none_res = res.get("none", {})
        e0 = none_res.get("empty", {}).get("flip_pct")
        o0 = none_res.get("occupied", {}).get("flip_pct")
        t0 = none_res.get("turned", {}).get("flip_pct")
        if e0 is not None and e0 < 1.0:
            sub = res.get("subtract", {}).get("occupied", {}).get("flip_pct")
            div = res.get("divide", {}).get("occupied", {}).get("flip_pct")
            out["verdict"] = (
                f"THE PREMISE WAS A STRIDE ARTEFACT, AND THE CONCLUSION OF THE "
                f"PREVIOUS RUN DOES NOT SURVIVE. With NO removal at all the "
                f"empty room flips {e0:.2f}% of dots per frame, the occupied "
                f"capture {o0:.2f}%"
                + (f", the turned capture {t0:.2f}%" if t0 is not None else "")
                + ". There is no saturated flip rate and therefore no "
                "row-constant banding for this filter to remove: the 50.00% "
                "empty-room figure that motivated the experiment was a "
                "torn-frame artefact of reading a 19,200-byte frame at a "
                "23,040-byte stride. Row removal now makes the occupied rate "
                "WORSE rather than better"
                + (f" ({sub:.2f}% subtract, {div:.2f}% divide, against "
                   f"{o0:.2f}% with none)" if (sub is not None and div is not None)
                   else "")
                + ", because there is no banding to subtract and the filter "
                "only adds its own noise. The character delta is not "
                "saturated in this capture, so the whole 'the character field "
                "flips 50% with nobody in the room' line of reasoning is "
                "withdrawn.")
        else:
            out["verdict"] = (
                "Row removal did not produce a clean split (empty near zero AND "
                "occupied high). The banding may be column-varying as well as "
                "row-constant, in which case a row-only model is insufficient.")
    p = Path(a.out) / "row_band_removal_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nVERDICT: {out['verdict']}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()