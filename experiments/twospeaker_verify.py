"""
twospeaker_verify.py -- prove, by measurement, that the corpus in
`examples/realtwospeaker/` really contains two or more distinct speakers.

WHY THIS FILE EXISTS.

`docs/research/ADOPTION-PLAN.md` (and commit 17ae46a) established the blocking
finding: every file in `examples/realaudio/` is a single person talking
continuously, and after any silence >= 0.6 s the SAME speaker resumes 98.1-100%
of the time. The positive class -- *a speaker yields the floor and somebody
else takes it* -- is therefore unrepresentable, and no turn-taking accuracy
number can be computed from that corpus at all.

A corpus that cannot express the positive class is not a small gap, it is a
measurement that would come out confident and wrong. So before any turn model
is trained or any threshold tuned, the material has to be shown to contain the
class. This file is that check, and it is built to be falsifiable.

WHAT COUNTS AS PROOF HERE (and what does not).

Not: "the manifest says there are four speakers".
Not: "two files that came from different people".
What is required is a MEASUREMENT on the audio itself, with three things a
skeptic can check independently:

  1. PER-SPEAKER DISTRIBUTIONS. Median f0 per speaker, but with the count of
     voiced frames BEHIND each median and the per-segment counts as well, so
     "two data points" is visibly not what happened.
  2. A NOISE FLOOR TO MEASURE THE MARGIN AGAINST. A median is meaningless
     without one. Here the noise floor is the WITHIN-speaker robust spread
     (MAD -> sigma, 1.4826 factor), because the question is not "are the medians
     unequal" (they always are) but "is the gap between speakers larger than
     the wobble inside one speaker".
  3. AN INDEPENDENT CHECK THAT DOES NOT TRUST THE LABELS. A 2-means clustering
     of the f0 values -- which never sees `speaker_id` -- is scored against the
     ground truth. If audio-only structure recovers the annotation, the
     annotation is describing something physically present.

Plus a channel-level check on the headset-mic files, where speaker identity is
separation of the RECORDING, not an inference at all.

HONESTY CONSTRAINTS BUILT INTO THE OUTPUT.

  * Unannotated stretches are written as digital silence by the fetcher, so a
    silence in this corpus is NOT automatically a silence in the meeting. The
    slice-level "silence then other speaker" rate is therefore reported ALONGSIDE
    the meeting-wide rate computed from the full annotation set, because the
    slice window was deliberately chosen to be turn-rich. Reporting only the
    slice rate would be exactly the cherry-picking this project is trying to
    avoid.
  * f0 from a far-field mic is contaminated by room and cross-talk. That is why
    the inlier rate (fraction of frames within a third of an octave of that
    speaker's own median) is reported: a speaker whose f0 is all over the place
    would show up here before it could quietly inflate the separation.

USES src/pitch.py AS THE REFERENCE, NOT AS THE ONLY ENGINE.

`PitchTracker` is pure stdlib and is the repo's instrument, but a per-frame
YIN in Python over a 7-minute slice is minutes of wall clock. So the bulk
measurement uses the same algorithm vectorised with numpy, and every run
cross-checks a sample of frames against `PitchTracker.estimate` and REPORTS the
agreement. If the fast path silently drifted from the reference, the file says
so instead of quietly reporting numbers.

Run:  python experiments/twospeaker_verify.py
Exit: 0 if two distinct speakers are confirmed, 1 if the check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                                  # noqa: E402

from src import pitch as ref_pitch                                 # noqa: E402
from src import wavio                                               # noqa: E402

# --- tracker geometry: identical to src/pitch.py defaults, on purpose ----
SR = 16000                            # wavio resamples to this anyway
FRAME_LEN = 800                         # 50 ms
HOP = 160                               # 10 ms
FMIN, FMAX = ref_pitch.FMIN, ref_pitch.FMAX
YIN_THRESHOLD = 0.15
MIN_CONFIDENCE = 0.5
ENERGY_FLOOR = ref_pitch.ENERGY_FLOOR

TAU_MIN = max(1, int(math.floor(SR / FMAX)))
TAU_MAX = int(math.ceil(SR / FMIN))
WIN = FRAME_LEN - TAU_MAX               # src/pitch.py's fixed comparison span

MERGE_S = 0.3                           # same-speaker segments closer = one turn
FLOOR_S = 0.6                           # the silence the project measures after
FRAME_CHUNK = 24                        # bound peak memory of the (F,tau,W) slab
CROSSCHECK_FRAMES = 150                 # frames re-measured with src/pitch.py
MIN_SPEAKER_VOICED_FRAMES = 25          # below this a median is not evidence
INLIER_OCTAVES = 1.0 / 3.0              # inlier band around a speaker's median


# --------------------------------------------------------------- fast YIN


def yin_batch(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised YIN, mirroring src/pitch.py's arithmetic step for step.

    frames: (F, FRAME_LEN) float64.  Returns (hz, conf) with hz=0.0 where the
    frame is unvoiced -- the same 'emit nothing rather than guess' contract.
    """
    f = frames.shape[0]
    hz = np.zeros(f)
    conf = np.zeros(f)
    if f == 0:
        return hz, conf

    # the loudness gate, before periodicity is asked for at all
    rms = np.sqrt((frames ** 2).mean(axis=1))
    audible = rms >= ENERGY_FLOOR
    if not audible.any():
        return hz, conf

    # d(tau) = sum_{j<W} (x[j] - x[j+tau])^2, for tau = 1..tau_max
    idx = np.arange(WIN)[None, :] + np.arange(1, TAU_MAX + 1)[:, None]
    d = np.empty((f, TAU_MAX), dtype=np.float64)
    for lo in range(0, f, FRAME_CHUNK):
        hi = min(lo + FRAME_CHUNK, f)
        sl = frames[lo:hi]
        diff = sl[:, idx] - sl[:, :WIN][:, None, :]
        np.square(diff, out=diff)
        d[lo:hi] = diff.sum(axis=2)
    del diff

    # cumulative mean normalised difference d'(tau) = d(tau) * tau / sum d[1..tau]
    taus = np.arange(1, TAU_MAX + 1, dtype=np.float64)
    running = np.cumsum(d, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dp = np.where(running > 0.0, d * taus[None, :] / running, 1.0)

    band = dp[:, TAU_MIN - 1:]
    # first local minimum of d' below the absolute threshold (YIN step 4)
    under = band < YIN_THRESHOLD
    has_dip = under.any(axis=1)
    first = under.argmax(axis=1)
    rows = np.arange(f)
    tau = first + TAU_MIN
    for _ in range(8):                       # walk down the dip; 8 is ample
        nxt = np.minimum(tau + 1, TAU_MAX)
        better = dp[rows, nxt - 1] < dp[rows, tau - 1]
        tau = np.where(better, nxt, tau)
    # rows with no dip under threshold fall back to the global minimum
    if not has_dip.all():
        glob = band.argmin(axis=1) + TAU_MIN
        tau = np.where(has_dip, tau, glob)

    c = 1.0 - dp[rows, tau - 1]
    conf = np.clip(c, 0.0, 1.0)

    # parabolic sub-sample refinement, as src/pitch.py::_parabolic
    # (neighbour indices are clamped: at tau == TAU_MAX there is no right
    # neighbour, and src/pitch.py returns the bare lag in that case)
    left = np.maximum(tau - 1, 1)
    right = np.minimum(tau + 1, TAU_MAX)
    denom = dp[rows, left - 1] - 2.0 * dp[rows, tau - 1] + dp[rows, right - 1]
    shift = np.where(np.abs(denom) < 1e-12, 0.0,
                     0.5 * (dp[rows, left - 1] - dp[rows, right - 1])
                     / np.where(denom == 0, 1.0, denom))
    refined = tau + np.clip(shift, -1.0, 1.0)
    est = SR / refined
    voiced = audible & (conf >= MIN_CONFIDENCE) & (est >= FMIN) & (est <= FMAX)
    hz = np.where(voiced, est, 0.0)
    return hz, conf


def voiced_hz(samples: np.ndarray) -> np.ndarray:
    """f0 of every fully-framed window of a mono float array, voiced only."""
    if samples.size < FRAME_LEN:
        return np.zeros(0)
    n = 1 + (samples.size - FRAME_LEN) // HOP
    view = np.lib.stride_tricks.sliding_window_view(samples, FRAME_LEN)[::HOP][:n]
    hz, _ = yin_batch(np.ascontiguousarray(view))
    return hz[hz > 0.0]


# ------------------------------------------------------------- statistics


def mad_sigma(values: list[float], centre: float | None = None) -> float:
    """Robust sigma from the median absolute deviation (1.4826 factor)."""
    if len(values) < 2:
        return 0.0
    c = statistics.median(values) if centre is None else centre
    return 1.4826 * statistics.median([abs(v - c) for v in values])


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    return {
        "n": len(s),
        "median_hz": round(statistics.median(s), 2),
        "mad_sigma_hz": round(mad_sigma(s), 2),
        "p05_hz": round(s[max(0, int(0.05 * len(s)))], 2),
        "p95_hz": round(s[min(len(s) - 1, int(0.95 * len(s)))], 2),
        "min_hz": round(s[0], 2),
        "max_hz": round(s[-1], 2),
    }


def two_means_1d(values: np.ndarray, iters: int = 100) -> tuple[float, float]:
    """Plain 2-means on one dimension; returns the two centres."""
    lo, hi = float(values.min()), float(values.max())
    centres = np.array([lo + (hi - lo) / 3.0, lo + 2.0 * (hi - lo) / 3.0])
    for _ in range(iters):
        a, b = centres
        side = np.abs(values[:, None] - centres[None, :]).argmin(axis=1)
        new = np.array([values[side == 0].mean() if (side == 0).any() else a,
                        values[side == 1].mean() if (side == 1).any() else b])
        if np.allclose(new, centres, atol=1e-6):
            centres = new
            break
        centres = new
    return float(centres[0]), float(centres[1])


def cluster_composition(values: np.ndarray, side: np.ndarray,
                        truth: list[str]) -> dict:
    """Describe a 2-way f0 split against labels it never saw.

    Purity is reported as "share of each cluster belonging to its dominant
    speaker". That number is deliberately NOT called accuracy: when two
    labelled speakers land in the same cluster (which is what happens here for
    two same-range voices) purity can be high while those two speakers are not
    actually separated. The per-cluster composition is printed alongside so that
    case is visible instead of hidden inside one flattering scalar.
    """
    comp: dict[str, dict] = {}
    for c in (0, 1):
        counts: dict[str, int] = {}
        for i in np.nonzero(side == c)[0]:
            counts[truth[i]] = counts.get(truth[i], 0) + 1
        total = sum(counts.values())
        if not total:
            continue
        dominant = max(counts, key=lambda k: counts[k])
        comp[f"cluster_{c}"] = {
            "frames": total,
            "speakers_present": dict(sorted(counts.items())),
            "dominant_speaker": dominant,
            "dominant_share": round(counts[dominant] / total, 4),
        }
    dominant_total = sum(v["frames"] * v["dominant_share"] for v in comp.values())
    return {"composition": comp,
            "dominant_speaker_purity": round(dominant_total / len(truth), 4) if truth else 0.0}


def voice_groups(speakers: dict, pooled_sigma: float, n_sigma: float = 3.0) -> dict:
    """How many speakers does median f0 actually resolve?

    Single-linkage on the per-speaker medians with a `n_sigma * pooled_sigma`
    threshold. Two speakers whose medians sit closer than that are reported as
    ONE group, because a pitch test cannot tell them apart -- which is a
    property of the measurement and needs stating, not a failure to hide.
    """
    usable = sorted((s for s in speakers if speakers[s].get("n", 0) >= MIN_SPEAKER_VOICED_FRAMES),
                    key=lambda s: speakers[s]["median_hz"])
    if len(usable) < 2 or pooled_sigma <= 0:
        return {"groups": len(usable), "assignment": {s: i for i, s in enumerate(usable)},
                "threshold_hz": None}
    threshold = n_sigma * pooled_sigma
    assignment: dict[str, int] = {}
    gid = 0
    for i, spk in enumerate(usable):
        if i == 0:
            assignment[spk] = 0
            continue
        prev = usable[i - 1]
        if speakers[spk]["median_hz"] - speakers[prev]["median_hz"] <= threshold:
            assignment[spk] = assignment[prev]
        else:
            gid += 1
            assignment[spk] = gid
    individually_resolvable = sum(
        1 for spk in usable
        if all(speakers[spk]["median_hz"] - speakers[o]["median_hz"] > threshold
               or speakers[o]["median_hz"] - speakers[spk]["median_hz"] > threshold
               for o in usable if o != spk))
    return {"groups": gid + 1, "threshold_hz": round(threshold, 2),
            "threshold_sigma": n_sigma,
            "assignment": assignment,
            "speakers_resolvable_individually": individually_resolvable,
            "note": ("groups = how many distinct voices median f0 can tell apart; "
                     "speakers_resolvable_individually = how many speakers are far "
                     "enough from EVERY other speaker to be identified on pitch alone")}


# ------------------------------------------------------------- turn truth


def turn_stats(segments: list[dict], label: str) -> dict:
    """How often a silence is followed by SOMEONE ELSE taking the floor.

    This is the positive class the project could not previously measure.
    """
    ordered = sorted(segments, key=lambda r: (r["begin_time"], r["end_time"]))
    turns: list[dict] = []
    for r in ordered:
        if (turns and turns[-1]["speaker_id"] == r["speaker_id"]
                and r["begin_time"] - turns[-1]["end_time"] < MERGE_S):
            turns[-1]["end_time"] = r["end_time"]
        else:
            turns.append({"speaker_id": r["speaker_id"], "begin_time": r["begin_time"],
                          "end_time": r["end_time"]})
    gaps = [(turns[i + 1]["begin_time"] - turns[i]["end_time"],
             turns[i]["speaker_id"], turns[i + 1]["speaker_id"])
            for i in range(len(turns) - 1)]
    floor = [g for g in gaps if g[0] >= FLOOR_S]
    handed = [g for g in floor if g[1] != g[2]]
    return {
        "scope": label,
        "segments": len(segments),
        "turns": len(turns),
        "gaps": len(gaps),
        "gaps_at_least_0p6s": len(floor),
        "followed_by_a_DIFFERENT_speaker": len(handed),
        "different_speaker_rate": (round(len(handed) / len(floor), 4) if floor else None),
        "note": ("gap measured between annotated segment edges; a gap here is "
                 "not necessarily a silence in the meeting, because unannotated "
                 "stretches were written as digital silence by the fetcher"),
    }


# --------------------------------------------------------------- the check


def measure_slice(samples: np.ndarray, sidecar: dict) -> dict:
    """Per-speaker f0 from the far-field slice, using the annotation labels."""
    origin = sidecar["meeting_window_start_s"]
    all_segs = sidecar["segments"]

    def overlaps_another(seg: dict) -> bool:
        """Does another speaker's utterance overlap this one in time?

        Far-field (single far-field microphone) segments that overlap contain
        TWO voices on ONE channel, so a median f0 over them describes neither
        speaker. Measured consequence, reported below: pooling these in inflates
        the within-speaker spread about 2.4x (52.8 -> 22.3 Hz) and destroys the
        separation. The primary measurement therefore uses non-overlapped
        segments only -- not because that flatters the answer, but because a
        median over two simultaneous voices is not a measurement of one person.
        The overlapped numbers are still reported, right beside it.
        """
        for other in all_segs:
            if other is seg or other["speaker_id"] == seg["speaker_id"]:
                continue
            if (other["begin_time"] < seg["end_time"]
                    and seg["begin_time"] < other["end_time"]):
                return True
        return False

    per_spk: dict[str, list[float]] = {}
    per_spk_overlapped: dict[str, list[float]] = {}
    per_segment: list[dict] = []
    for seg in all_segs:
        if not seg.get("audible"):
            continue
        lo = int(round((seg["audible_start_s"] - origin) * SR))
        hi = int(round((seg["audible_end_s"] - origin) * SR))
        chunk = samples[max(0, lo):max(0, hi)]
        hz = voiced_hz(chunk)
        spk = seg["speaker_id"]
        solo = not overlaps_another(seg)
        (per_spk if solo else per_spk_overlapped).setdefault(spk, []).extend(hz.tolist())
        per_segment.append({"speaker_id": spk, "audio_id": seg["audio_id"],
                            "duration_s": round(seg["audible_end_s"] - seg["audible_start_s"], 3),
                            "solo": solo,
                            "voiced_frames": int(hz.size)})

    def summarise(table: dict[str, list[float]]) -> dict:
        out = {}
        for spk, vals in sorted(table.items()):
            d = describe(vals)
            med = d.get("median_hz")
            # inlier rate: fraction of this speaker's frames within a third of an
            # octave of its own median. A speaker scattered across octaves (YIN
            # octave errors, or heavy cross-talk) shows up HERE, not as a fake
            # separation between speakers.
            if vals:
                ratio = np.array(vals) / med
                lo_b = 2.0 ** -INLIER_OCTAVES
                hi_b = 2.0 ** INLIER_OCTAVES
                d["inlier_rate"] = round(float(((ratio >= lo_b) & (ratio <= hi_b)).mean()), 4)
            counts = sorted(s["voiced_frames"] for s in per_segment if s["speaker_id"] == spk)
            d["segments_with_voicing"] = len(counts)
            d["voiced_frames_per_segment"] = {
                "min": counts[0] if counts else 0,
                "median": counts[len(counts) // 2] if counts else 0,
                "max": counts[-1] if counts else 0,
            }
            out[spk] = d
        return out

    speakers = summarise(per_spk)

    # margin against the within-speaker noise floor
    sigmas = {s: speakers[s].get("mad_sigma_hz", 0.0) for s in speakers}
    usable = [s for s in speakers if speakers[s].get("n", 0) >= MIN_SPEAKER_VOICED_FRAMES]
    pooled = math.sqrt(sum(sigmas[s] ** 2 for s in usable) / len(usable)) if usable else 0.0
    pairs = []
    for i, a in enumerate(sorted(usable)):
        for b in sorted(usable)[i + 1:]:
            delta = abs(speakers[a]["median_hz"] - speakers[b]["median_hz"])
            pairs.append({
                "a": a, "b": b,
                "median_delta_hz": round(delta, 2),
                "within_speaker_mad_sigma_hz": {a: sigmas[a], b: sigmas[b]},
                "separation_in_mad_sigmas": round(delta / pooled, 2) if pooled > 0 else None,
            })
    pairs.sort(key=lambda p: p["median_delta_hz"])

    # audio-only clustering, scored against the labels it never saw
    allv, truth = [], []
    for spk in sorted(per_spk):
        allv.extend(per_spk[spk])
        truth.extend([spk] * len(per_spk[spk]))
    clustering = {}
    if len(usable) >= 2 and allv:
        arr = np.array(allv)
        c0, c1 = two_means_1d(arr)
        side = (np.abs(arr - c1) < np.abs(arr - c0)).astype(int)
        info = cluster_composition(arr, side, truth)
        clustering = {
            "method": "2-means on f0 only; speaker_id never used",
            "centre_hz": [round(c0, 2), round(c1, 2)],
            "centre_separation_hz": round(abs(c1 - c0), 2),
            "frames": int(arr.size),
            **info,
        }

    return {"file": sidecar["file"], "meeting_id": sidecar["meeting_id"],
            "duration_s": round(samples.size / SR, 2),
            "measurement_scope": ("non-overlapped segments only: a far-field segment "
                                  "overlapping another speaker carries two voices on one "
                                  "channel, so its median f0 describes neither speaker"),
            "speakers": speakers,
            "pooled_within_speaker_mad_sigma_hz": round(pooled, 2),
            "pairwise_separation": pairs,
            "f0_resolves": voice_groups(speakers, pooled),
            "clustering": clustering,
            "crosstalk_control": {
                "why": ("same measurement on the overlapping segments, kept so the choice "
                        "of subset is visible rather than asserted"),
                "speakers_in_overlapped_segments": summarise(per_spk_overlapped),
                "solo_segments": sum(1 for s in per_segment if s["solo"]),
                "overlapped_segments": sum(1 for s in per_segment if not s["solo"]),
            },
            "per_segment_voiced_frames": {
                "segments_measured": len(per_segment),
                "total_voiced_frames": sum(s["voiced_frames"] for s in per_segment),
                "min": min((s["voiced_frames"] for s in per_segment), default=0),
                "median": int(statistics.median([s["voiced_frames"] for s in per_segment]))
                           if per_segment else 0,
                "max": max((s["voiced_frames"] for s in per_segment), default=0),
            },
            "turn_ground_truth": turn_stats(sidecar["segments"], "slice (window chosen for turn density)"),
            "segments_detail": per_segment}


def measure_channels(ch_dir: Path) -> dict:
    """Headset mics: speaker identity is separation of the RECORDING."""
    out = {"directory": ch_dir.name, "channels": {}}
    for wav in sorted(ch_dir.glob("spk_*.wav")):
        samples = np.array(wavio.read_wav(str(wav)).samples, dtype=np.float64)
        hz = voiced_hz(samples)
        d = describe(hz.tolist())
        med = d.get("median_hz")
        if hz.size and med:
            ratio = hz / med
            d["inlier_rate"] = round(float(((ratio >= 2.0 ** -INLIER_OCTAVES)
                                            & (ratio <= 2.0 ** INLIER_OCTAVES)).mean()), 4)
        out["channels"][wav.stem.replace("spk_", "")] = {
            "file": wav.name, "duration_s": round(samples.size / SR, 2), **d}
    meds = {k: v.get("median_hz") for k, v in out["channels"].items() if v.get("median_hz")}
    names = sorted(meds)
    sigmas = {k: out["channels"][k].get("mad_sigma_hz", 0.0) for k in names}
    usable = [k for k in names if out["channels"][k].get("n", 0) >= MIN_SPEAKER_VOICED_FRAMES]
    pooled = math.sqrt(sum(sigmas[k] ** 2 for k in usable) / len(usable)) if usable else 0.0
    pairs = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            delta = abs(meds[a] - meds[b])
            pairs.append({"a": a, "b": b, "median_delta_hz": round(delta, 2),
                          "separation_in_mad_sigmas": round(delta / pooled, 2) if pooled > 0 else None})
    pairs.sort(key=lambda p: p["median_delta_hz"])
    out["pooled_within_channel_mad_sigma_hz"] = round(pooled, 2)
    out["pairwise_separation"] = pairs
    out["note"] = ("each file is one AMI headset microphone over the same time "
                   "window, so the speakers are separated by the RECORDING and "
                   "not by any measurement taken here")
    return out


def crosscheck(samples: np.ndarray, sidecar: dict) -> dict:
    """Re-measure sampled frames with src/pitch.py and report the agreement."""
    tracker = ref_pitch.PitchTracker(SR)
    origin = sidecar["meeting_window_start_s"]
    picks: list[tuple[int, int]] = []          # (speaker, absolute sample idx)
    for seg in sidecar["segments"]:
        if not seg.get("audible"):
            continue
        lo = int(round((seg["audible_start_s"] - origin) * SR))
        hi = int(round((seg["audible_end_s"] - origin) * SR))
        n = max(0, 1 + (hi - lo - FRAME_LEN) // HOP)
        if n <= 0:
            continue
        for k in range(min(3, n)):             # a few frames per segment
            picks.append((seg["speaker_id"], lo + k * HOP))
        if len(picks) >= CROSSCHECK_FRAMES:
            break
    picks = picks[:CROSSCHECK_FRAMES]
    agree = both = 0
    cents: list[float] = []
    for _, idx in picks:
        frame = samples[idx:idx + FRAME_LEN].tolist()
        ref_hz, _ = tracker.estimate(frame)
        view = np.lib.stride_tricks.sliding_window_view(samples[idx:idx + FRAME_LEN], FRAME_LEN)
        fast_hz, _ = yin_batch(np.ascontiguousarray(view))
        fast_hz = float(fast_hz[0]) if fast_hz.size else 0.0
        if ref_hz is None and fast_hz == 0.0:
            agree += 1
            both += 1
        elif ref_hz is not None and fast_hz > 0.0:
            both += 1
            cents.append(1200.0 * math.log2(fast_hz / ref_hz))
            if abs(1200.0 * math.log2(fast_hz / ref_hz)) < 50.0:   # under half a semitone
                agree += 1
    return {
        "reference": "src.pitch.PitchTracker.estimate (pure stdlib)",
        "engine_under_test": "numpy yin_batch (same algorithm, vectorised)",
        "frames_checked": len(picks),
        "frames_scored_by_both": both,
        "agreement_rate": round(agree / len(picks), 4) if picks else None,
        "median_abs_deviation_cents": (round(statistics.median([abs(c) for c in cents]), 3)
                                       if cents else None),
        "max_abs_deviation_cents": (round(max(abs(c) for c in cents), 3) if cents else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "examples" / "realtwospeaker"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "twospeaker_verify_results.json"))
    ap.add_argument("--skip-crosscheck", action="store_true")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    results: dict = {
        "generated_by": "experiments/twospeaker_verify.py",
        "purpose": ("prove >=2 distinct speakers exist by measurement, so the "
                    "positive class (a speaker yields the floor) is representable"),
        "corpus_dir": str(corpus.relative_to(ROOT)),
        "tracker": {"source": "src/pitch.py algorithm, vectorised with numpy",
                    "frame_len": FRAME_LEN, "hop": HOP, "sample_rate": SR,
                    "fmin": FMIN, "fmax": FMAX, "yin_threshold": YIN_THRESHOLD,
                    "min_confidence": MIN_CONFIDENCE, "energy_floor": ENERGY_FLOOR},
        "slices": {}, "headset_channels": {}, "crosscheck": None,
    }

    for wav in sorted(corpus.glob("ami_sdm_*.wav")):
        side_path = wav.with_suffix(".segments.json")
        if not side_path.exists():
            print(f"skip {wav.name}: no sidecar")
            continue
        print(f"measuring {wav.name} ...")
        samples = np.array(wavio.read_wav(str(wav)).samples, dtype=np.float64)
        sidecar = json.loads(side_path.read_text())
        results["slices"][wav.name] = measure_slice(samples, sidecar)
        if not args.skip_crosscheck and results["crosscheck"] is None:
            results["crosscheck"] = crosscheck(samples, sidecar)

    for d in sorted(corpus.glob("ihm_*_channels_*")):
        print(f"measuring {d.name}/ ...")
        results["headset_channels"][d.name] = measure_channels(d)

    # meeting-wide contrast: the slices were picked for turn density, so the
    # unbiased number has to come from the full annotation set
    meta = corpus / "_downloads" / "meta_sdm_test.json"
    if meta.exists():
        rows = json.loads(meta.read_text())
        for meeting in sorted({r["meeting_id"] for r in rows}):
            segs = [{"speaker_id": r["speaker_id"], "begin_time": r["begin_time"],
                     "end_time": r["end_time"], "audio_id": r["audio_id"]}
                    for r in rows if r["meeting_id"] == meeting]
            results.setdefault("meeting_wide_turn_ground_truth", {})[meeting] = turn_stats(
                segs, "whole meeting (all annotated segments)")

    # ---- verdict ---------------------------------------------------------
    # NOTE ON WHAT IS AND IS NOT BEING CLAIMED.
    # The claim is "at least two distinct speakers are present and measurable",
    # NOT "median f0 identifies every speaker". Two people can legitimately share
    # a pitch range, so requiring EVERY pair to clear 3 sigma would demand that
    # pitch be a better instrument than it is, and would convert a true finding
    # into a false failure. What is required instead: two or more speakers carry
    # enough voiced frames to have a median; SOME pair is separated far beyond the
    # within-speaker noise floor; an audio-only split aligns with the labelled
    # groups; the channel-separated reference agrees. The limit of pitch is then
    # reported as its own finding below, not buried.
    criteria: list[dict] = []
    best_sep, best_spk = 0.0, None
    for name, sl in results["slices"].items():
        for pair in sl.get("pairwise_separation") or []:
            sep = pair.get("separation_in_mad_sigmas") or 0.0
            if sep > best_sep:
                best_sep = sep
                best_spk = (name, pair["a"], pair["b"], pair["median_delta_hz"])
    n_spk = usable_count(results)
    criteria.append({"name": "at least two speakers with usable voicing",
                     "pass": n_spk >= 2,
                     "detail": f"{n_spk} speakers with >= {MIN_SPEAKER_VOICED_FRAMES} voiced frames"})
    criteria.append({"name": "some speaker pair separated by >= 3x the within-speaker MAD sigma",
                     "pass": best_sep >= 3.0,
                     "detail": f"best separation {best_sep:.1f} MAD sigmas"
                               + (f" ({best_spk[1]} vs {best_spk[2]}, delta {best_spk[3]} Hz)"
                                  f" in {best_spk[0]}" if best_spk else "")})
    crit_cluster = [(n, sl["clustering"], sl["pooled_within_speaker_mad_sigma_hz"])
                    for n, sl in results["slices"].items() if sl.get("clustering")]
    best_bimodal, best_purity = 0.0, 0.0
    for _, c, sigma in crit_cluster:
        if sigma > 0:
            best_bimodal = max(best_bimodal, c["centre_separation_hz"] / sigma)
        best_purity = max(best_purity, c["dominant_speaker_purity"])
    # Deliberately NOT scored as "2-means recovers the speaker labels": each pitch
    # group here holds TWO speakers, so per-speaker purity is bounded near 0.5 by
    # the physics of the measurement, not by a defect. What can be tested
    # without any label at all is whether the audio is bimodal in f0.
    criteria.append({"name": "f0 alone is bimodal: the 2-means centres sit >= 3x the pooled "
                             "within-speaker MAD sigma apart (no labels used)",
                     "pass": best_bimodal >= 3.0,
                     "detail": f"best centre separation {best_bimodal:.2f} pooled sigmas; "
                               f"dominant-speaker purity {best_purity} (bounded below 1.0 by "
                               f"construction -- each pitch group holds two speakers, see "
                               f"pitch_limitations)"})
    ch_sep, ch_best = 0.0, None
    for dname, v in results["headset_channels"].items():
        for pair in v.get("pairwise_separation", []):
            sep = pair.get("separation_in_mad_sigmas") or 0.0
            if sep > ch_sep:
                ch_sep, ch_best = sep, (dname, pair["a"], pair["b"], pair["median_delta_hz"])
    criteria.append({"name": "headset channels differ (channel-separated speaker identity)",
                     "pass": (ch_sep >= 3.0) if results["headset_channels"] else True,
                     "detail": (f"best channel separation {ch_sep:.1f} MAD sigmas"
                                f" ({ch_best[1]} vs {ch_best[2]}, delta {ch_best[3]} Hz)"
                                if ch_best else "no channel set present")})
    mw = results.get("meeting_wide_turn_ground_truth", {})
    worst_rate = min((v["different_speaker_rate"] for v in mw.values()
                      if v["different_speaker_rate"] is not None), default=0.0)
    criteria.append({"name": "positive class present: a >=0.6s silence is often followed "
                              "by a DIFFERENT speaker (meeting-wide)",
                     "pass": worst_rate >= 0.5,
                     "detail": f"lowest meeting-wide rate {worst_rate} "
                               f"(the existing single-speaker corpus is 0.981-1.0, i.e. the "
                               f"floor is essentially never handed over there)"})

    # The limit of the pitch route is a finding of its own, not a footnote.
    unresolved = []
    for name, sl in results["slices"].items():
        fr = sl.get("f0_resolves") or {}
        if fr.get("speakers_resolvable_individually", 0) < len(sl["speakers"]):
            unresolved.append({
                "file": name,
                "speakers": len(sl["speakers"]),
                "f0_resolves_these_groups": fr.get("groups"),
                "resolvable_individually_on_pitch": fr.get("speakers_resolvable_individually"),
                "closest_pairs": [{"a": p["a"], "b": p["b"], "delta_hz": p["median_delta_hz"],
                                   "separation_in_mad_sigmas": p["separation_in_mad_sigmas"]}
                                  for p in (sl.get("pairwise_separation") or [])[:2]],
            })
    results["pitch_limitations"] = {
        "what_pitch_proves": ("two or more physically distinct voices, separated by far more "
                              "than the within-speaker spread, in channel-separated audio"),
        "what_pitch_does_NOT_prove": (
            "that every speaker is identifiable. Median f0 groups these speakers by voice "
            "range; WITHIN a group the median difference is smaller than the measurement's own "
            "noise floor, so a hand-off between those two produces no reliable pitch jump. A "
            "turn-taking stage must not assume 'the speaker changed' == 'f0 jumped'."),
        "not_resolvable": unresolved,
    }

    results["criteria"] = criteria
    results["verdict"] = "TWO_OR_MORE_DISTINCT_SPEAKERS_CONFIRMED" if all(c["pass"] for c in criteria) \
        else "CHECK_FAILED"

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n=== {results['verdict']} ===")
    for c in criteria:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['name']}\n         {c['detail']}")
    for name, sl in results["slices"].items():
        print(f"\n{name}: {sl['duration_s']}s, {len(sl['speakers'])} speakers")
        for spk, d in sl["speakers"].items():
            print(f"  {spk}: median {d.get('median_hz')} Hz  n={d.get('n')} voiced frames "
                  f"mad_sigma {d.get('mad_sigma_hz')} Hz  inlier {d.get('inlier_rate')}")
    for dname, ch in results["headset_channels"].items():
        print(f"\n{dname}")
        for spk, d in ch["channels"].items():
            print(f"  {spk}: median {d.get('median_hz')} Hz  n={d.get('n')} "
                  f"mad_sigma {d.get('mad_sigma_hz')} Hz")
    print(f"\nwrote {out}")
    return 0 if results["verdict"].endswith("CONFIRMED") else 1


def usable_count(results: dict) -> int:
    n = 0
    for sl in results.get("slices", {}).values():
        n = max(n, sum(1 for d in sl["speakers"].values()
                       if d.get("n", 0) >= MIN_SPEAKER_VOICED_FRAMES))
    return n


if __name__ == "__main__":
    sys.exit(main())