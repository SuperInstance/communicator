"""
combined_filter.py -- compose the two filters, then re-ask the teacher question.

WHY COMPOSING IS THE OBVIOUS NEXT STEP AND WAS NOT DONE SOONER.

Two filters were measured separately, and I treated them as alternatives. They
are not:

  ROW-BAND REMOVAL (`row_band_removal.py`) removes the ROW-CONSTANT component.
  The nuisance is horizontal bands, identical in every column, so the per-row
  median across the columns is an estimate of it. It costs NO LATENCY and it
  does not blur motion, because it never combines frames. Empty room
  50.00% -> 0.01%, occupied 48.19% -> 48.57% (unchanged).

  TEMPORAL MEDIAN (`temporal_median_fix.py`) removes the TEMPORALLY RANDOM
  component -- per-pixel jitter that moves frame to frame. Empty room
  50.00% -> 0.00%, occupied 48.19% -> 21.43% at k=3. It costs (k-1)/30 s and
  smears fast motion.

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
FB = W * H * 3 // 2
YS = W * H

sys.path.insert(0, str(ROOT / "experiments"))
from ascii_motion_render import load_session  # noqa: E402


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
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    n = min(nf, len(raw) // FB)
    Y = raw[:n * FB].reshape(n, FB)[:, :YS].reshape(n, H, W).astype(np.float64)
    return Y


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
    ap.add_argument("--corpus", default="/mnt/c/proart_capture/corpus")
    ap.add_argument("--nf", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    a = ap.parse_args(argv)

    E = load_raw(Path(a.empty), a.nf)
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
            sess[meta["condition"]] = (load_raw(Path(src), a.nf), meta)
            ident = file_identity(src)
            ident["used"] = True
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

    used_argv = list(sys.argv[1:]) if argv is None else list(argv)
    provenance = build_provenance(
        corpus, file_identity(a.empty), session_ids, used_argv,
        {"nf": a.nf, "out": str(a.out),
         "variants": {k: {"row_band_removal": v[0], "temporal_median_k": v[1]}
                      for k, v in variants.items()}})

    out = {"question": ("does composing row-band removal with a temporal median "
                        "de-saturate the motion field and make the speech "
                        "signal stand out more?"),
           "definitions": definitions,
           "variants": results,
           "summary": {name: describe(row) for name, row in results.items()},
           "provenance": provenance,
           "reproducible": bool(
               provenance.get("script_sha256")
               and provenance.get("empty_capture", {}).get("sha256")
               and provenance.get("cli_args") is not None
               and all(s.get("sha256") for s in provenance.get("sessions", [])
                       if s.get("used", True))
               and not provenance.get("problems")),
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
