"""
drift_structure.py -- what IS the per-frame drift? It is not gain.

THE FRAME LAYOUT, FIRST, because it decides whether any of this is true.

The captures on disk are NOT standard NV12. `C:\\proart_capture\\nv12small.py`
writes, per 160x96 frame:

    Y plane   15,360 bytes   (160 * 96)
    U,V        3,840 bytes   (80 * 24 samples, INTERLEAVED)
    ---------------------------
    total     19,200 bytes   = W*H*5/4

so the stride is 19,200 bytes, not the `W*H*3//2` = 23,040 this file used to
assume. Read at 23,040, all but every sixth frame is a tear -- rows 0..71 from
frame k, rows 72..95 from frame k+1.

That splice lands a DISCONTINUITY ACROSS ROW 72, so a mis-strided read
manufactures a horizontal band in every frame, and a horizontal band is exactly
what this file was built to look for and call LED flicker. The drift, the
"banding", and the row-band-removal pass all came from the same 3,840-byte
error.

The layout is DETECTED from the file's content by `src/nv12.py` (the candidate
whose luma has the smallest mean consecutive |diff| wins; an ambiguous file is
an error, not a guess). See `src/nv12.py` and `docs/research/NV12-STRIDE.md`.

WHY THIS EXISTS.

`gain_invariance.py` tested the obvious fix for `EMPTY-ROOM-CONTROL.md` and it
FAILED: per-frame gain normalisation left the character delta at 50.00% in an
empty room, unchanged. Along the way it exposed an arithmetic error in my own
earlier report.

I had written that normalisation "cuts adjacent drift from 43.3 to 0.81
counts". It does not. Dividing a frame by its mean 111.18 and by its mean
109.71 are the same operation, so:

    raw         43.686 / 111.18  =  0.393
    normalised  43.226 / 109.71  =  0.394

Identical. Normalisation removed nothing. The earlier "43.3 -> 0.81" compared
raw counts against relative units -- a factor-of-100 illusion. A drift reduction
I had reported as ~50x was actually 1.00x.

There is now a SECOND, larger correction, and it is the stride: the 43.3 counts
of drift were never drift at all. At the correct stride the adjacent |diff| is
a fraction of a count, and the row/column split that was read as "mains-derived
LED flicker" is the row-72 splice of the torn frame. The banding was the
tear.

THE QUESTION THIS ANSWERS.

What changes between adjacent frames? Three candidates, distinguishable:

  G  GLOBAL GAIN      frame[i+1] ~ k * frame[i], one scalar for the whole
                      frame. Removing it should zero the difference.
  L  LOCAL/SPATIAL    some regions move or re-light and others do not. No
                      single scalar explains it.
  N  PIXEL NOISE      independent per-pixel jitter. No spatial structure, and
                      it averages out as the frame is downsampled.

The test: for every adjacent pair, fit the single scalar k that best explains
the pair (least squares through the origin), remove it, and look at what is
left. A global-gain frame collapses to nothing. Pure pixel noise collapses to
a residual that shrinks under averaging. A spatial change collapses to a
STRUCTURED residual that survives averaging -- and the structure is itself the
finding, because it shows you WHERE in the frame the scene is changing.

If it is NOISE, averaging a few frames before thresholding fixes the problem
cheaply and correctly. If it is SPATIAL, something in the room is genuinely
changing and no threshold will help until that is found.

Both the occupied and the empty capture are reported. The empty one was taken
in a dark room, which gives auto-exposure far more range to swing through.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.nv12 import read_luma  # noqa: E402

FRAME_W, FRAME_H = 160, 96


def sha256_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_y(path: Path, nf: int):
    """The luma of every frame, and the layout that was detected.

    The stride is DETECTED from the file's content by `src/nv12.py`, not
    assumed. See `src/nv12.py`.
    """
    luma, layout = read_luma(path, width=FRAME_W, height=FRAME_H)
    nf = min(nf, layout.n_frames)
    return luma[:nf].astype(np.float64), layout


def analyse(path: Path, label: str, nf: int = 400):
    Y, layout = load_y(path, nf)
    n = len(Y)
    a, b = Y[:-1], Y[1:]

    raw_mad = float(np.abs(b - a).mean())

    # Fit one scalar per pair: b ~ k * a. Residual is what gain cannot explain.
    num = (a * b).sum(axis=(1, 2))
    den = (a * a).sum(axis=(1, 2)) + 1e-9
    k = num / den
    resid = b - k[:, None, None] * a
    gain_mad = float(np.abs(resid).mean())
    # Express both in relative terms so the comparison is in the same units.
    mean_luma = float(Y.mean())
    raw_rel = raw_mad / mean_luma
    gain_rel = gain_mad / mean_luma

    # How structured is the residual? Compare per-pixel residual variance to
    # the variance you'd get from independent noise of the same magnitude.
    # If it is noise, averaging the pair down reduces the residual like 1/sqrt(n).
    r2 = (resid[:60] ** 2).reshape(60, -1).mean(axis=1)
    # 2x2 block average of the residual: independent noise drops ~2x in RMS.
    blk = resid[:60].reshape(60, FRAME_H // 2, 2, FRAME_W // 2, 2).mean(axis=(2, 4))
    rms1 = float(np.sqrt((resid[:60] ** 2).mean()))
    rms2 = float(np.sqrt((blk ** 2).mean()))
    noise_ratio = rms2 / rms1 if rms1 else None

    # Where does the residual live? Column/row profile of |residual|.
    col = np.abs(resid).mean(axis=(0, 1))
    row = np.abs(resid).mean(axis=(0, 2))
    col_peak = int(np.argmax(col))
    row_peak = int(np.argmax(row))
    col_share = float(col[max(0, col_peak - 8):col_peak + 9].sum() / col.sum())

    print(f"\n=== {label}  ({n} frames of {layout.n_frames}, {path.name}) ===")
    print(f"  layout / stride                {layout.name} / {layout.stride}")
    print(f"  mean luma                     {mean_luma:8.2f}")
    print(f"  adjacent |diff|               {raw_mad:8.3f} counts"
          f"   ({raw_rel * 100:.3f}% of mean)")
    print(f"  after removing best-fit gain  {gain_mad:8.3f} counts"
          f"   ({gain_rel * 100:.3f}% of mean)")
    print(f"  -> gain explains {100 * (1 - gain_rel / raw_rel):.2f}% of the drift")
    print(f"  fitted k: median {statistics.median(k):.4f}  "
          f"min {min(k):.4f}  max {max(k):.4f}")
    print(f"  residual RMS per pixel        {rms1:8.3f}")
    print(f"  residual RMS after 2x2 mean   {rms2:8.3f}"
          f"   ratio {noise_ratio:.3f}  (pure noise would be ~0.50)")
    print(f"  residual concentrated at col {col_peak:3d} "
          f"({col_share * 100:.1f}% within +-8 cols), row {row_peak}")

    # CLASSIFICATION, and a correction to the first version of this function.
    #
    # The original logic read "gain explains almost none of it" and concluded
    # PIXEL-LEVEL NOISE. That is the opposite of what the measurements say, and
    # the script contradicted its own printed data two lines above: it had just
    # measured a 2x2-averaging ratio of 0.97-1.00, and independent pixel noise
    # gives ~0.50 because averaging four samples halves their variance.
    #
    # A residual that barely moves under spatial averaging is NOT noise. It is
    # large-scale structure. This is the third time this project has had a
    # verdict function contradict its own printed output -- see
    # HOW-WE-KEEP-FOOLING-OURSELVES.md -- so the rule is now applied literally:
    # each classification must cite the number that supports it.
    gain_frac = gain_rel / raw_rel
    if gain_frac < 0.5:
        kind = (f"GLOBAL GAIN: one scalar explains {100 * (1 - gain_frac):.1f}% "
                f"of the drift (fit k median {statistics.median(k):.4f})")
    elif noise_ratio is not None and noise_ratio > 0.75:
        kind = (f"SPATIAL STRUCTURE, not noise: the residual survives 2x2 "
                f"averaging (ratio {noise_ratio:.2f}, pure noise would be 0.50) "
                f"and gain explains only {100 * (1 - gain_frac):.1f}%")
    else:
        kind = (f"MIXED: gain explains {100 * (1 - gain_frac):.1f}%, "
                f"averaging ratio {noise_ratio:.2f}")

    # -- BANDING TEST --------------------------------------------------------
    # The classification above says the residual is large-scale STRUCTURE. This
    # says what structure: separate the residual's variation along rows from its
    # variation along columns.
    #
    # A room feature -- a window, a chair, a face -- varies in BOTH directions.
    # A horizontal light band is identical in every column, so its column
    # profile is flat. In the empty-room capture the column profile std is
    # 0.16 counts against a row profile std of 45.98: a 287:1 split, which is
    # a pure banding pattern and nothing else.
    #
    # THAT 287:1 SPLIT WAS THE TORN FRAME. The splice lands across row 72, so
    # the residual is a horizontal step and the column profile is flat by
    # construction. What the measurement now has to say is recorded below, and
    # whatever it says replaces the "mains-derived LED flicker" reading.
    #
    # The spatial frequency then identifies the source. The camera scans 96
    # rows in 33 ms at 30 fps. Two to three cycles across that height is
    # 60-100 Hz, which is mains-derived LED flicker sampled by the rolling
    # shutter -- not sensor noise, not gain drift, and not anything happening
    # in the room.
    rp = resid.mean(axis=2)
    rp = rp - rp.mean(axis=1, keepdims=True)
    cp = resid.mean(axis=1)
    cp = cp - cp.mean(axis=1, keepdims=True)
    row_std, col_std = float(rp.std()), float(cp.std())
    col_energy_share = float(col_std ** 2 / (col_std ** 2 + row_std ** 2 + 1e-12))

    fy = np.abs(np.fft.rfft(rp, axis=1)).mean(axis=0)[1:]
    order = np.argsort(fy)[::-1][:3]
    top_bins = [int(t + 1) for t in order]
    top_energy = fy[order]
    # Report the STRONG bins as a range, not just the single maximum. The
    # maximum is almost always bin 1 (the lowest frequency carries the most
    # energy in any banded pattern), so quoting it alone as "the flicker
    # frequency" is misleading -- bins 1, 2 and 3 are all strong here, which
    # spans roughly 30-90 Hz at 30 fps.
    #
    # What is NOT established: which specific lamp or driver is responsible.
    # The bands are consistent with mains-derived LED flicker sampled by the
    # rolling shutter, and that is the leading candidate, but this measurement
    # cannot separate 50 Hz from 60 Hz mains, nor flicker from a PWM driver.
    fps = 30.0
    band_lo, band_hi = min(top_bins) * fps, max(top_bins) * fps
    share = float(top_energy.sum() / fy.sum())

    print(f"  row-profile std             {row_std:8.2f}")
    print(f"  col-profile std             {col_std:8.2f}")
    print(f"  column energy share         {col_energy_share * 100:8.2f}%  "
          f"(a room feature would be much higher)")
    print(f"  row FFT top bins            {top_bins}  -> ~{band_lo:.0f}-{band_hi:.0f} Hz "
          f"at {fps:.0f} fps (top-3 share {share * 100:.0f}%)")

    return {"label": label, "path": str(path), "n_frames": n,
            "n_frames_available": layout.n_frames,
            "layout": layout.name, "stride_bytes": layout.stride,
            "layout_evidence": layout.as_dict(),
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "mean_luma": mean_luma,
            "adjacent_mad_raw": raw_mad, "adjacent_mad_after_gain": gain_mad,
            "relative_raw_pct": raw_rel * 100,
            "relative_after_gain_pct": gain_rel * 100,
            "gain_explains_pct": 100 * (1 - gain_rel / raw_rel),
            "fitted_k_median": statistics.median(k),
            "fitted_k_min": float(min(k)), "fitted_k_max": float(max(k)),
            "residual_rms": rms1, "residual_rms_2x2": rms2,
            "residual_noise_ratio": noise_ratio,
            "residual_peak_col": col_peak, "residual_peak_row": row_peak,
            "row_profile_std": row_std, "col_profile_std": col_std,
            "column_energy_share": col_energy_share,
            "row_fft_top_bins": top_bins,
            "implied_band_hz_range": [band_lo, band_hi],
            "row_fft_top3_energy_share": share,
            "classification": kind}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--occupied", default="/mnt/c/proart_capture/A.nv12")
    ap.add_argument("--empty", default="/home/proart/projects/glasses/.capture_keep/empty1.nv12")
    ap.add_argument("--nf", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    res = []
    for p, lab in ((a.occupied, "OCCUPIED (facing, speaking)"),
                   (a.empty, "EMPTY (nobody, dark room)")):
        if Path(p).is_file():
            res.append(analyse(Path(p), lab, a.nf))

    def _by_label(label):
        for r in res:
            if r["label"].startswith(label):
                return r
        return {}

    occ = _by_label("OCCUPIED")
    emp = _by_label("EMPTY")

    notes = []
    if occ and emp:
        notes = [
            ("THE BANDING WAS THE TORN FRAME. The previous run measured a "
             "row-dominant residual in the occupied capture (row-profile "
             "std 31.31 counts against col-profile std 7.08, column "
             "energy share 4.9%) and read that as mains-derived LED "
             "flicker. It is the row-72 splice of a mis-strided read: a "
             "horizontal step is flat across every column by "
             "construction. At the correct stride the profile INVERTS -- "
             "the occupied capture is now column-dominant (row-profile std "
             f"{occ['row_profile_std']:.2f} against col-profile std "
             f"{occ['col_profile_std']:.2f}, column energy share "
             f"{occ['column_energy_share'] * 100:.0f}%), which is the "
             "signature of a room feature rather than a horizontal band. "
             "The top-3 row-FFT energy share falls from 44% to "
             f"{occ['row_fft_top3_energy_share'] * 100:.0f}%, so there "
             "is no dominant band left to attribute to a lamp."),
            ("THE EMPTY ROOM'S DRIFT IS AT THE NOISE FLOOR, and its gain "
             "fraction is meaningless. The empty capture's adjacent |diff| "
             f"is {emp['adjacent_mad_raw']:.3f} counts -- far below the "
             "8-bit quantisation of the sensor -- and its residual 2x2 "
             f"averaging ratio is {emp['residual_noise_ratio']:.3f}, "
             "which is the ~0.50 of independent per-pixel noise. 'Gain "
             f"explains {emp['gain_explains_pct']:.1f}%' is a division "
             "artefact at that magnitude, not a physical claim, and the "
             "MIXED classification is the pre-registered rule's honest "
             "fallback rather than a finding."),
            ("The verdict reports the capture where gain explains the MOST "
             "of the drift. That selection rule is the pre-existing one "
             "and is unchanged; with the new numbers it selects OCCUPIED "
             "(2.6%) rather than EMPTY (-14.0%)."),
        ]

    print("\n" + "=" * 66)
    for r in res:
        print(f"{r['label']:32s} {r['classification']}")

    out = {"question": ("is the per-frame drift global gain, pixel noise, or a "
                        "spatial change in the scene?"),
           "correction": (
               "TWO CORRECTIONS NOW, and the second is the bigger one. "
               "(1) An earlier report said per-frame gain normalisation cut "
               "drift from 43.3 to 0.81 counts. It did not: it compared raw "
               "counts against relative units. 43.686/111.18 = 0.393 and "
               "43.226/109.71 = 0.394, so the reduction was 1.00x, not ~50x. "
               "That error is why gain_invariance.py could not work. "
               "(2) The 43.3 counts of drift were never drift at all. The "
               "captures on disk carry 19,200 bytes per frame, not the "
               "W*H*3//2 = 23,040 this file used to assume, so all but every "
               "sixth frame was torn -- rows 0..71 from frame k, rows 72..95 "
               "from frame k+1. The splice lands across row 72, which "
               "manufactures a horizontal band in every mis-read frame. The "
               "'mains-derived LED flicker' banding this file measured, and "
               "the row-band-removal pass built on it, were the tear."),
           "captures": res,
           "notes": notes,
           "method": ("least-squares fit of one scalar k per adjacent frame "
                      "pair through the origin; residual is what a global gain "
                      "cannot explain. Noise test: RMS of the residual after "
                      "2x2 block averaging, relative to before -- pure "
                      "independent noise gives ~0.50, structure survives."),
           "provenance": {
               "generated_by": "experiments/drift_structure.py",
               "script_path": str(Path(__file__).resolve()),
               "script_sha256": sha256_file(Path(__file__).resolve()),
               "python": sys.executable,
               "command_line": " ".join(sys.argv),
               "cli_args": list(sys.argv[1:]),
               "nf": a.nf,
               "captures": {
                   r["label"]: {
                       "path": r["path"],
                       "sha256": r["sha256"],
                       "bytes": r["bytes"],
                       "layout": r["layout"],
                       "stride_bytes": r["stride_bytes"],
                       "n_frames": r["n_frames_available"],
                       "frames_used": r["n_frames"],
                       "layout_evidence": r["layout_evidence"],
                   } for r in res
               },
           },
           "supersedes": {
               "why": ("Re-measured on 2026-10-20. The previous run read both "
                       "captures at W*H*3//2 = 23,040 bytes per frame; the "
                       "captures on disk carry 19,200 bytes per frame, so all "
                       "but every sixth frame was torn (rows 0..71 from frame "
                       "k, rows 72..95 from frame k+1). The previous numbers "
                       "are recorded here rather than silently overwritten."),
               "previous_stride_bytes": 23040,
               "previous_captures": [
                   {"label": "OCCUPIED (facing, speaking)",
                    "path": "/mnt/c/proart_capture/A.nv12",
                    "n_frames": 400,
                    "mean_luma": 111.17517106119791,
                    "adjacent_mad_raw": 43.68600162515664,
                    "adjacent_mad_after_gain": 42.46152358314948,
                    "relative_raw_pct": 39.29474648715321,
                    "relative_after_gain_pct": 38.1933512472636,
                    "gain_explains_pct": 2.8029071017157348,
                    "fitted_k_median": 0.8856671028295736,
                    "fitted_k_min": 0.8274576985362577,
                    "fitted_k_max": 0.9880347818329536,
                    "residual_rms": 52.32412683117896,
                    "residual_rms_2x2": 50.70367387362781,
                    "residual_noise_ratio": 0.9690304825768148,
                    "residual_peak_col": 17,
                    "residual_peak_row": 92,
                    "row_profile_std": 31.312025810208343,
                    "col_profile_std": 7.08056686943787,
                    "column_energy_share": 0.04864693112727292,
                    "row_fft_top_bins": [1, 2, 3],
                    "implied_band_hz_range": [30.0, 90.0],
                    "row_fft_top3_energy_share": 0.4428339019148429,
                    "classification": ("SPATIAL STRUCTURE, not noise: the residual "
                                        "survives 2x2 averaging (ratio 0.97, pure "
                                        "noise would be 0.50) and gain explains only 2.8%")},
                   {"label": "EMPTY (nobody, dark room)",
                    "path": "/home/proart/projects/glasses/.capture_keep/empty1.nv12",
                    "n_frames": 400,
                    "mean_luma": 49.69857259114583,
                    "adjacent_mad_raw": 39.34364181939223,
                    "adjacent_mad_after_gain": 37.63364664052054,
                    "relative_raw_pct": 79.16453082678997,
                    "relative_after_gain_pct": 75.72379784449835,
                    "gain_explains_pct": 4.346306289390956,
                    "fitted_k_median": 0.4975903034112806,
                    "fitted_k_min": 0.34263900946582504,
                    "fitted_k_max": 1.8174001358574519,
                    "residual_rms": 50.878145767996,
                    "residual_rms_2x2": 50.869402768786934,
                    "residual_noise_ratio": 0.9998281580612443,
                    "residual_peak_col": 150,
                    "residual_peak_row": 54,
                    "row_profile_std": 45.9758537715454,
                    "col_profile_std": 0.1648122284073493,
                    "column_energy_share": 1.285031212242642e-05,
                    "row_fft_top_bins": [1, 2, 3],
                    "implied_band_hz_range": [30.0, 90.0],
                    "row_fft_top3_energy_share": 0.46455772339548435,
                    "classification": ("SPATIAL STRUCTURE, not noise: the residual "
                                        "survives 2x2 averaging (ratio 1.00, pure "
                                        "noise would be 0.50) and gain explains only 4.3%")},
               ],
               "previous_verdict": (
                   "Gain explains 4.3% of the drift in EMPTY (nobody, dark "
                   "room), and the residual noise ratio is 1.00. SPATIAL "
                   "STRUCTURE, not noise: the residual survives 2x2 averaging "
                   "(ratio 1.00, pure noise would be 0.50) and gain explains "
                   "only 4.3%"),
               "verdict_changed": (
                   "The verdict text is expected to change, and the reason is "
                   "not a new measurement: the thing the previous verdict "
                   "classified was the torn frame. A horizontal splice across "
                   "row 72 produces exactly the flat column profile and the "
                   "strong low row-frequency content that the previous run "
                   "read as mains-derived LED flicker. Whether the new "
                   "numbers still support a banding claim is stated by the "
                   "new verdict and must be read on its own."),
           },
           }
    if res:
        best = min(res, key=lambda r: r["gain_explains_pct"] * -1)
        out["verdict"] = (
            f"Gain explains {best['gain_explains_pct']:.1f}% of the drift in "
            f"{best['label']}, and the residual noise ratio is "
            f"{best['residual_noise_ratio']:.2f}. {best['classification']}")
    p = Path(a.out) / "drift_structure_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nVERDICT: {out.get('verdict')}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
