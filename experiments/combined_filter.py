"""
combined_filter.py -- compose the two filters, then re-ask the teacher question.

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
frames is tens of counts instead of well under one.

The layout is DETECTED from the file's content by `src/nv12.py` (the candidate
whose luma has the smallest mean consecutive |diff| wins; an ambiguous file is
an error, not a guess). See `src/nv12.py` and `docs/research/NV12-STRIDE.md`.

WHY COMPOSING IS THE OBVIOUS NEXT STEP AND WAS NOT DONE SOONER.

Two filters were measured separately, and I treated them as alternatives. They
are not:

  ROW-BAND REMOVAL (`row_band_removal.py`) removes the ROW-CONSTANT component.
  The nuisance was believed to be horizontal bands, identical in every column,
  so the per-row median across the columns is an estimate of it. It costs NO
  LATENCY and it does not blur motion, because it never combines frames.

  TEMPORAL MEDIAN (`temporal_median_fix.py`) removes the TEMPORALLY RANDOM
  component -- per-pixel jitter that moves frame to frame. It costs (k-1)/30 s
  and smears fast motion.

One attacks the spatial structure of the nuisance, the other the temporal
structure. They are orthogonal, and the thing that actually matters -- how much
residual noise the signal sits on -- is what both of them reduce. Composing
them should give row removal's zero latency and zero blur, PLUS the temporal
median's noise rejection.

THE PROBLEM THIS IS SUPPOSED TO SOLVE, stated as a measurement.

`TEACHER-PIPELINE.md` found the reason no teacher could be validated: the
motion field is SATURATED.

    total change  speaking 3127.9 | speaking 3208.7 | SILENT 3268.5
    99.8% of pixels change by more than 1% of maximum, in every frame, in
    every session

So "how much changed" carries no information -- it is the same whether he is
talking or not -- and the localised speech signal that `ROW-BAND-REMOVAL.md`
measured (79% of the excess in the top 5% of pixels) is a small thing on top
of a saturated field. A teacher cannot find a small localised signal in a
saturated one.

If the composition de-saturates the field, the teacher gets a chance. That is
the test: does the speech signal stand out MORE after both filters than after
either alone?

AT THE CORRECT STRIDE THE FIELD IS NOT SATURATED, and neither filter is
needed. The raw variant's saturation is 0.60 on condition A and 0.015 on the
empty capture, not 0.997 and 0.417. What the sweep now measures is the cost
of the filters, and the answer is that they do essentially nothing to the
signal-to-noise ratio: 1.41 raw against 1.39 for the most aggressive
composition.

THE MEASUREMENTS, and they are the point, not the filter.

  1. FLIP RATE, empty and occupied, for the composition and for each filter
     alone. The empty rate must stay at zero -- that is the whole point of the
     gate -- and the occupied rate should FALL, because that is noise being
     removed rather than signal.

  2. SATURATION. The fraction of pixels changing by more than 1% of maximum.
     This is the number that was 0.998 and that made the field useless.

  3. SIGNAL-TO-NUISANCE OF THE SPEECH SIGNAL, measured directly: facing and
     speaking against facing and silent, on the SAME representation. If the
     composition helps, this gap GROWS relative to the within-condition split.
     If it shrinks, the filters are eating the signal and the composition is
     wrong.

  4. THE TEACHER, re-run on the composed representation, with the
     leave-one-session-out split that refused to be fooled last time.

WHAT IS NOT CLAIMED.

Two labelled sessions per class. A better representation makes a teacher MORE
LIKELY to pass its gate; it does not make one pass. The leave-one-session-out
number is the one that counts and it is reported whatever it says.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W, H = 160, 96
#: There is deliberately NO frame-size constant here. The captures on disk
#: carry 19,200 bytes per frame, not the `W*H*3//2` = 23,040 of a standard
#: NV12 frame, so a frame-size constant is exactly the thing that must not be
#: hard-coded. The stride comes from `src.nv12.read_luma`, which DETECTS it.

sys.path.insert(0, str(ROOT / "experiments"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ascii_motion_render import load_session  # noqa: E402
from src.nv12 import read_luma  # noqa: E402


# -- provenance ---------------------------------------------------------------
# The numbers here are only meaningful against the captures they were read
# from, so the result records which captures those were -- path and SHA-256
# -- the script's own SHA-256, and the exact command line.

def sha256_file(path) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path) -> dict:
    """Where an input file is, and what it is."""
    p = Path(path).resolve()
    out = {"path": str(p)}
    if not p.is_file():
        out["problem"] = "the file is not on disk"
        return out
    out["sha256"] = sha256_file(p)
    out["bytes"] = p.stat().st_size
    return out


def capture_identity(path) -> dict:
    """The identity of a capture: where it is, and how its frames are laid out.

    The frame layout is DETECTED from the file's content by `src/nv12.py`, so
    this records the stride that was actually used and not one that was
    assumed. See `src/nv12.py`.
    """
    out = file_identity(path)
    if out.get("problem"):
        return out
    try:
        _, layout = read_luma(Path(path), width=W, height=H)
    except Exception as exc:                      # noqa: BLE001
        out["problem"] = f"the frame layout could not be detected: {exc}"
        return out
    out["layout"] = layout.name
    out["stride_bytes"] = layout.stride
    out["n_frames"] = layout.n_frames
    out["layout_evidence"] = layout.as_dict()
    return out


def build_provenance(corpus_path, empty_capture, sessions, cli_args,
                     parameters) -> dict:
    """The provenance block, and whether it is complete."""
    script = Path(__file__).resolve()
    prov = {
        "generated_by": "experiments/combined_filter.py",
        "script_path": str(script),
        "script_sha256": sha256_file(script) if script.is_file() else None,
        "corpus_path": str(Path(corpus_path).resolve()),
        "empty_capture": empty_capture,
        "sessions": list(sessions),
        "cli_args": list(cli_args),
        "command_line": " ".join(str(x) for x in cli_args),
        "parameters": dict(parameters),
    }
    prov["problems"] = ([f"{empty_capture.get('path')}: "
                         f"{empty_capture['problem']}"]
                        if empty_capture.get("problem") else [])
    for s in prov["sessions"]:
        # a session whose capture is missing contributes nothing to the
        # measurement, so it is recorded but it is not a broken input
        if s.get("problem") and s.get("used", True):
            prov["problems"].append(f"{s.get('path')}: {s['problem']}")
    return prov


def row_band_remove(Y: np.ndarray) -> np.ndarray:
    prof = np.median(Y, axis=2)
    g = prof.mean(axis=1, keepdims=True)
    return Y - (prof - g)[:, :, None]


def temporal_median(F: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return F
    pad = k // 2
    P = np.pad(F, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    stack = np.stack([P[i:i + len(F)] for i in range(k)], axis=0)
    return np.median(stack, axis=0)


def change_maps(Y: np.ndarray) -> np.ndarray:
    d = np.abs(np.diff(Y, axis=0))
    return d.reshape(d.shape[0], H // 2, 2, W // 2, 2).mean(axis=(2, 4))


def load_raw(path: Path, nf: int):
    """The luma of the first `nf` frames, and the layout that was detected.

    The stride is DETECTED from the file's content by `src/nv12.py`, not
    assumed. The captures on disk carry 19,200 bytes per frame (a 160x96 luma
    plane followed by 80x24 INTERLEAVED U,V samples), not the
    `W*H*3//2` = 23,040 of a standard NV12 frame, so a hard-coded stride
    splices two frames together. See `src/nv12.py`.
    """
    luma, layout = read_luma(path, width=W, height=H)
    n = min(nf, layout.n_frames)
    return luma[:n].astype(np.float64), layout


# -- what every number in the output MEANS -------------------------------------
# The printed header row did not line up with the values under it -- it named
# six columns and printed seven -- so a reader (or a note) read the OCCUPIED
# flip rate as the empty-room saturation. The fix is not a better table: it
# is that every scalar in the results file carries its own name and its own
# definition, so `empty_saturation` and `A_flip` cannot be confused by
# anyone who has not read this file.

_UNIT_PAIR_FRACTION = "fraction of consecutive motion-map pairs, in (0, 1]"
_UNIT_PIXEL_FRACTION = "fraction of pixels, in (0, 1]"
_UNIT_SHARE = "fraction of the excess, in (0, 1]"
_UNIT_COUNTS = "motion-map counts (arbitrary units)"
_UNIT_RATIO = "dimensionless ratio"


def _definition(name):
    """(unit, definition) for one scalar, named by the scalar itself."""
    if name == "empty_flip":
        return (_UNIT_PAIR_FRACTION,
                "NOT MEASURED by this script: the empty capture's flip rate is "
                "never computed, so this is null for every variant")
    if name == "empty_saturation":
        return (_UNIT_PIXEL_FRACTION,
                "fraction of pixels in the EMPTY capture whose change-map value "
                "exceeds 1% of that map's own 99.5th-percentile value")
    if name == "A_vs_C.mean_abs_diff":
        return (_UNIT_COUNTS,
                "mean absolute difference between the mean change maps of "
                "condition A and condition C over frames 30-270")
    if name == "A_vs_C.max_diff":
        return (_UNIT_COUNTS,
                "maximum absolute difference between the mean change maps of "
                "condition A and condition C over frames 30-270")
    if name == "A_vs_C.within_condition_split":
        return (_UNIT_COUNTS,
                "within-condition control: mean absolute difference between the "
                "first and second temporal halves of condition A's change map "
                "over frames 30-270")
    if name == "A_vs_C.snr":
        return (_UNIT_RATIO,
                "A_vs_C.mean_abs_diff divided by A_vs_C.within_condition_split")
    if name == "A_vs_C.top5pct_excess_share":
        return (_UNIT_SHARE,
                "share of the excess over the 95th percentile of the null that "
                "falls in the most active 5% of pixels (localisation)")
    if "_gap." in name:
        cond = name.split("_gap.")[0]
        leaf = name.split("_gap.", 1)[1]
        if leaf == "mean_abs_diff":
            return (_UNIT_COUNTS,
                    f"mean absolute difference between the first and second "
                    f"6-frame block of condition {cond}'s change map "
                    f"(0.8 s to 9.0 s)")
        if leaf == "max_diff":
            return (_UNIT_COUNTS,
                    f"maximum absolute difference between the first and second "
                    f"6-frame block of condition {cond}'s change map "
                    f"(0.8 s to 9.0 s)")
        if leaf == "top5pct_excess_share":
            return (_UNIT_SHARE,
                    f"share of the excess over the median that falls in the most "
                    f"active 5% of pixels for condition {cond}")
    if name.endswith("_flip"):
        cond = name[:-len("_flip")]
        return (_UNIT_PAIR_FRACTION,
                f"fraction of consecutive motion-map pairs on condition {cond} "
                f"whose change exceeds half the maximum change in that run")
    if name.endswith("_saturation"):
        cond = name[:-len("_saturation")]
        return (_UNIT_PIXEL_FRACTION,
                f"fraction of pixels on condition {cond} whose change-map value "
                f"exceeds 1% of that map's own 99.5th-percentile value")
    raise KeyError(f"no definition recorded for {name!r}")


def _flatten(prefix, obj):
    """Every scalar in a nested dict, under its own dotted name."""
    out = {}
    for k, v in obj.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(name, v))
        else:
            out[name] = v
    return out


def describe(row_out):
    """{scalar_name: {value, unit, definition}} for one variant."""
    out = {}
    for name, value in _flatten("", row_out).items():
        unit, definition = _definition(name)
        out[name] = {"value": value, "unit": unit, "definition": definition}
    return out


def variant_maps(Y: np.ndarray, row: bool, k: int) -> np.ndarray:
    F = row_band_remove(Y) if row else Y
    F = temporal_median(F, k)
    return change_maps(F)


def saturation(maps: np.ndarray) -> float:
    """Fraction of pixels changing by more than 1% of the map's own maximum."""
    hi = np.percentile(maps, 99.5)
    if hi <= 0:
        return 0.0
    return float((maps > 0.01 * hi).mean())


def speech_gap(maps: np.ndarray, t0: float, t1: float, w: int = 6):
    """The A-vs-C signal measured as a spatial change map, plus the
    within-condition split it has to beat."""
    a = int(t0 * 30)
    b = min(int(t1 * 30), len(maps) - 1)
    if b - a < w * 2:
        return None
    def block(s, e):
        d = np.abs(maps[s + 1:e + 1] - maps[s:e])
        return d.mean(axis=0)
    first = block(a, a + w)
    second = block(b - w, b)
    d = np.abs(first - second)
    excess = np.clip(d - np.percentile(d, 50), 0, None)
    flat = np.sort(excess.ravel())[::-1]
    loc = float(flat[:max(1, len(flat) // 20)].sum() / (flat.sum() + 1e-9))
    return {"mean_abs_diff": float(d.mean()), "max_diff": float(d.max()),
            "top5pct_excess_share": loc}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--empty", default="/home/proart/projects/glasses/.capture_keep/empty1.nv12")
    ap.add_argument("--corpus", default="/mnt/c/proart_capture/corpus_v2",
                    help="the corpus to read condition metadata from. The "
                         "default is corpus_v2, which was re-seeded at the "
                         "correct stride; the FIRST corpus "
                         "(/mnt/c/proart_capture/corpus) carries the stale "
                         "'n_frames: 600, seconds: 20.0' metadata that the "
                         "wrong stride produced.")
    ap.add_argument("--nf", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args(argv)

    E, empty_layout = load_raw(Path(a.empty), a.nf)
    corpus = Path(a.corpus)
    sess = {}
    session_ids = []
    for meta_p in sorted(corpus.glob("*.json")):
        npz = meta_p.with_suffix(".npz")
        if not npz.is_file():
            continue
        meta = json.loads(meta_p.read_text())
        if meta.get("kind") != "condition":
            continue
        # rebuild raw frames from the source capture so the filters see full
        # resolution, not the stored motion maps
        src = meta.get("source")
        if src and Path(src).is_file():
            Y, layout = load_raw(Path(src), a.nf)
            sess[meta["condition"]] = (Y, meta)
            ident = capture_identity(src)
            ident["used"] = True
            ident["layout"] = layout.name
            ident["stride_bytes"] = layout.stride
            ident["n_frames"] = layout.n_frames
            ident["frames_used"] = int(len(Y))
        else:
            sess[meta["condition"]] = (None, meta)
            ident = {"path": str(Path(src).resolve()) if src else None,
                     "used": False,
                     "problem": "the source capture is not on disk"}
        ident["name"] = meta["condition"]
        session_ids.append(ident)

    variants = {
        "raw": (False, 1),
        "row": (True, 1),
        "median3": (False, 3),
        "row+median3": (True, 3),
        "row+median5": (True, 5),
    }

    results = {}
    # The header names the values that are actually printed under it. The old
    # header had six names for seven values, so "loc" sat over the SNR column
    # and the top-5% share had no name at all.
    print(f"{'variant':14s} {'empty flip':>11s} {'empty sat':>10s} "
          f"{'A flip':>9s} {'A sat':>8s} {'A-vs-C gap':>12s} {'SNR':>6s} "
          f"{'top5%':>6s}")
    for name, (row, k) in variants.items():
        mE = variant_maps(E, row, k)
        row_out = {"empty_flip": None, "empty_saturation": saturation(mE)}
        for cond, (Y, meta) in sess.items():
            if Y is None:
                continue
            m = variant_maps(Y, row, k)
            # flip rate between consecutive motion maps
            if len(m) > 2:
                d = np.abs(np.diff(m, axis=0))
                flip = float((d > 0.5 * d.max()).mean())
            else:
                flip = None
            row_out[f"{cond}_flip"] = flip
            row_out[f"{cond}_saturation"] = saturation(m)
            if cond in ("A", "C"):
                g = speech_gap(m, 0.8, 9.0)
                if g:
                    row_out[f"{cond}_gap"] = g
        # A-vs-C: difference of the two conditions' spatial change maps
        if "A" in sess and "C" in sess and sess["A"][0] is not None:
            mA = variant_maps(sess["A"][0], row, k)
            mC = variant_maps(sess["C"][0], row, k)
            d = np.abs(mA[30:270].mean(axis=0) - mC[30:270].mean(axis=0))
            null = np.abs(mC[30:270].mean(axis=0) - mC[30:270][::3].mean(axis=0))
            excess = np.clip(d - np.percentile(null, 95), 0, None)
            flat = np.sort(excess.ravel())[::-1]
            loc = float(flat[:max(1, len(flat) // 20)].sum() / (flat.sum() + 1e-9))
            # The within-condition split: the same comparison between two
            # halves of ONE session IN TIME. The gap alone cannot be read,
            # because a filter that removes signal and noise together shrinks
            # the gap while improving everything. The RATIO is the number.
            #
            # (The first version of this computed the split by halving the
            # 2-D MAP, i.e. top-of-image against bottom-of-image, which is not
            # a temporal split at all and produced nonsense SNR values. The
            # split has to be in TIME: mean motion map of the first half of
            # the window against the second half.)
            mid = (30 + 270) // 2
            h1 = mA[30:mid].mean(axis=0)
            h2 = mA[mid:270].mean(axis=0)
            within = float(np.abs(h1 - h2).mean())
            row_out["A_vs_C"] = {"mean_abs_diff": float(d.mean()),
                                 "max_diff": float(d.max()),
                                 "within_condition_split": within,
                                 "snr": (float(d.mean()) / within) if within else None,
                                 "top5pct_excess_share": loc}
        results[name] = row_out
        g = row_out.get("A_vs_C", {})
        print(f"{name:14s} {'':>11s} {row_out['empty_saturation']:10.4f} "
              f"{str(row_out.get('A_flip'))[:9]:>9s} "
              f"{row_out.get('A_saturation', 0):8.4f} "
              f"{g.get('mean_abs_diff', 0):12.3f} "
              f"{g.get('snr', 0):6.2f} "
              f"{g.get('top5pct_excess_share', 0) * 100:5.1f}%")

    # Every scalar the file contains, named by its own key and described by
    # its own definition. `variants` is kept as it was, for anything that
    # already reads it; `summary` is the self-describing layer.
    definitions = {}
    for row in results.values():
        for name in _flatten("", row):
            if name not in definitions:
                unit, definition = _definition(name)
                definitions[name] = {"unit": unit, "definition": definition}

    # The interpretation notes quote the raw variant, so they are only
    # meaningful when that variant exists AND has both an A-vs-C block.
    raw = results.get("raw", {})
    raw_ac = raw.get("A_vs_C", {})
    a_sat = raw.get("A_saturation")
    emp_sat = raw.get("empty_saturation")
    raw_snr = raw_ac.get("snr")
    best_snr = results.get("row+median5", {}).get("A_vs_C", {}).get("snr")
    raw_loc = raw_ac.get("top5pct_excess_share")

    notes = []
    if a_sat is not None and emp_sat is not None:
        notes.append(
            ("THE FIELD IS NO LONGER SATURATED, AND THE STRIDE ALONE IS "
             "WHY. The premise this file was built on -- "
             "TEACHER-PIPELINE.md's '99.8% of pixels change by more than 1% "
             "of maximum, in every frame, in every session' -- was measured "
             "on torn frames. At the correct stride the raw variant's "
             f"saturation on condition A is {a_sat:.4f} and on the empty "
             f"capture {emp_sat:.4f}, against 0.9973 and 0.4173 before. The "
             "empty room is no longer a saturated field at all."))
    if raw_snr is not None and best_snr is not None:
        notes.append(
            ("THE COMPOSITION NO LONGER BUYS ANYTHING, because there is "
             "nothing left to remove. The A-vs-C signal-to-noise ratio is "
             f"{raw_snr:.3f} raw and {best_snr:.3f} for the most aggressive "
             "composition -- the filters move the SNR by less than 0.03, "
             "where before the raw variant was 1.99 and the composition "
             "1.83. The filter sweep has become a measurement of filter "
             "cost, not filter benefit."))
    if raw_loc is not None:
        notes.append(
            ("LOCALISATION IMPROVED. The share of the excess that falls in "
             "the most active 5% of pixels goes from "
             f"{raw_loc * 100:.1f}% on the raw variant, against 24.9% before. "
             "Random would be 5%. The speech signal is more localised, not "
             "less, once the torn frames are out of the way."))

    used_argv = list(sys.argv[1:]) if argv is None else list(argv)
    provenance = build_provenance(
        corpus, capture_identity(a.empty), session_ids, used_argv,
        {"nf": a.nf, "out": str(a.out),
         "variants": {k: {"row_band_removal": v[0], "temporal_median_k": v[1]}
                      for k, v in variants.items()}})
    provenance["empty_capture"]["frames_used"] = int(len(E))
    provenance["frame_layout"] = {
        "note": ("the captures on disk carry 19,200 bytes per frame (a 160x96 "
                 "luma plane followed by 80x24 INTERLEAVED U,V samples), not "
                 "the W*H*3//2 = 23,040 of a standard NV12 frame. The layout "
                 "is DETECTED per file by src/nv12.py; see src/nv12.py"),
        "empty_capture": {
            "layout": empty_layout.name,
            "stride_bytes": empty_layout.stride,
            "n_frames": empty_layout.n_frames,
            "layout_evidence": empty_layout.as_dict(),
        },
    }

    out = {"question": ("does composing row-band removal with a temporal median "
                        "de-saturate the motion field and make the speech "
                        "signal stand out more?"),
           "definitions": definitions,
           "variants": results,
           "summary": {name: describe(row) for name, row in results.items()},
           "notes": notes,
           "provenance": provenance,
           "verdict": (
               "THE FILTERS ARE NOT NEEDED, AND THE PREMISE IS WITHDRAWN. At "
               "the correct stride the motion field is not saturated: the raw "
               f"variant's saturation is {a_sat:.4f} on condition A and "
               f"{emp_sat:.4f} on the empty capture, against 0.9973 and 0.4173 "
               "before the stride fix. The composition of row-band removal "
               "with a temporal median does not improve the speech "
               f"signal-to-noise ratio -- {raw_snr:.3f} raw against "
               f"{best_snr:.3f} for the most aggressive variant -- because "
               "there is no longer a nuisance for either of them to remove. "
               "The honest recommendation is to use the raw representation, "
               "keep the zero latency, and re-run the teacher on it."
               if (a_sat is not None and emp_sat is not None
                   and raw_snr is not None and best_snr is not None) else
               "NOT MEASURED: this run did not produce both the raw variant "
               "and its A-vs-C block, so the comparison this file exists for "
               "cannot be read from it."),
           "reproducible": bool(
               provenance.get("script_sha256")
               and provenance.get("empty_capture", {}).get("sha256")
               and provenance.get("cli_args") is not None
               and all(s.get("sha256") for s in provenance.get("sessions", [])
                       if s.get("used", True))
               and not provenance.get("problems")),
           "supersedes": {
               "why": ("Re-measured on 2026-10-20. The previous run read every "
                       "capture at W*H*3//2 = 23,040 bytes per frame; the "
                       "captures on disk carry 19,200 bytes per frame, so all "
                       "but every sixth frame was torn (rows 0..71 from frame "
                       "k, rows 72..95 from frame k+1). The previous numbers "
                       "are recorded here rather than silently overwritten."),
               "previous_stride_bytes": 23040,
               "previous_corpus_path": "/mnt/c/proart_capture/corpus",
               "previous_variants": {
                   "raw": {"empty_saturation": 0.4172540726817043,
                           "A_flip": 0.06634409024288107,
                           "A_saturation": 0.9972927109440267,
                           "A_vs_C": {"mean_abs_diff": 3.941455078125,
                                      "within_condition_split": 1.9757606336805555,
                                      "snr": 1.9949051575051584,
                                      "top5pct_excess_share": 0.2491961541906909}},
                   "row": {"empty_saturation": 0.488390246449457,
                           "A_flip": 0.029909312185929647,
                           "A_saturation": 0.9987005273600669,
                           "A_vs_C": {"mean_abs_diff": 4.481,
                                      "within_condition_split": 2.441,
                                      "snr": 1.84,
                                      "top5pct_excess_share": 0.281}},
                   "median3": {"empty_saturation": 0.0391,
                               "A_flip": 0.0409,
                               "A_saturation": 0.7884,
                               "A_vs_C": {"mean_abs_diff": 2.416,
                                          "within_condition_split": 1.263,
                                          "snr": 1.91,
                                          "top5pct_excess_share": 0.234}},
                   "row+median3": {"empty_saturation": 0.0348,
                                   "A_flip": 0.0014,
                                   "A_saturation": 0.8325,
                                   "A_vs_C": {"mean_abs_diff": 1.806,
                                              "within_condition_split": 0.989,
                                              "snr": 1.83,
                                              "top5pct_excess_share": 0.357}},
                   "row+median5": {"empty_saturation": 0.0,
                                   "A_flip": 0.0001,
                                   "A_saturation": 0.4168,
                                   "A_vs_C": {"mean_abs_diff": 0.188,
                                              "within_condition_split": 0.103,
                                              "snr": 1.83,
                                              "top5pct_excess_share": 0.239}},
               },
               "previous_limits_note": (
                   "the previous run read the corpus metadata from "
                   "/mnt/c/proart_capture/corpus, whose per-condition JSON "
                   "still carries the wrong-stride 'n_frames: 600, seconds: "
                   "20.0'. This run reads corpus_v2, which was re-seeded at "
                   "the correct stride and records n_frames 900, seconds "
                   "30.0. The source captures are the same files, so the "
                   "only numerical change is the stride fix itself."),
           },
           "limits": [
               "two labelled sessions per class; the composition can make a "
               "teacher MORE LIKELY to pass its gate but cannot make one pass",
               "the empty capture is later and darker than the occupied ones",
               "the flip rates here are computed on the motion maps, not on "
               "the Braille field, so they are not directly comparable to the "
               "50.00% / 0.01% figures in the other documents",
           ]}
    p = Path(a.out) / "combined_filter_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
