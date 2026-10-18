"""
stability — the per-utterance trust probe.

EXPERIMENTS-AUDIO.md §3e measured it: Jev's Choice tracks are
deterministic (mode-share 1.00 across every item tested) while its Score
tracks jitter on ambiguous mid-scale texts — six distinct INTENTION
values in ten asks on "Why is the build red again?". PAPERS.md §14
records why no per-item probability guarantee is possible (conformal
sets are the ceiling). The honest controller behavior is therefore
fade-don't-promise: measure an item's stability cheaply, stamp it in the
provenance, and let the presence gate multiply it in.

`probe_stability` is the cheap version of the §3e experiment: n
uncached resamples (NEVER the verdict cache — the cache exists to freeze
exactly the jitter this measures), a mode-share per track, and a
stability factor in [0,1] per track suitable for multiplying into a
gate. The default n=3 catches the strong instability (a 3-ask
disagreement is already evidence); n=10 is the measurement-grade
default of the experiment, not the runtime one.

Pure stdlib; reuses judged.read_tracks' never-raise contract.
"""

from __future__ import annotations

from collections import Counter

from . import judged

DEFAULT_N = 3
STABLE_BAR = 0.995     # mode-share at or above this reads as deterministic


def _answer(reading):
    """The comparable per-track answer: the chosen option for Choice
    tracks (deterministic in practice), the CC value for Score tracks
    (the continuous position whose jitter we are measuring)."""
    if reading.track in ("ATTENTION", "GESTURE"):
        return reading.detail.get("choice")
    return reading.value


def probe_stability(text: str, *, n: int = DEFAULT_N, model: str = judged.DEFAULT_MODEL,
                    client=None, timeout: float = 8.0,
                    sleep_s: float = 0.1) -> dict:
    """
    Resample `text` n times uncached; return per-track stability.

    Returns {track: {"n": asks that succeeded, "distinct": distinct
    answers, "mode_share": agreement fraction, "factor": mode_share if
    unstable enough to matter, else 1.0}}. Tracks that never answered are
    absent. Never raises.
    """
    import time

    samples: dict[str, list] = {}
    for _ in range(max(1, n)):
        readings, _ = judged.read_tracks(text, model=model, client=client,
                                         timeout=timeout)
        for r in readings:
            if r.ok:
                samples.setdefault(r.track, []).append(_answer(r))
        if sleep_s:
            time.sleep(sleep_s)

    out = {}
    for track, answers in samples.items():
        counts = Counter(map(str, answers))
        mode_share = counts.most_common(1)[0][1] / len(answers)
        out[track] = {
            "n": len(answers),
            "distinct": len(counts),
            "mode_share": round(mode_share, 3),
            # factor: 1.0 when effectively deterministic (the measured
            # Choice behavior), else the agreement fraction — multiplies
            # into a gate exactly like a presence probability.
            "factor": 1.0 if mode_share >= STABLE_BAR else round(mode_share, 3),
        }
    return out


def trust_fields(stability: dict) -> dict:
    """The provenance-sidecar shape: flat, json-safe, one row per track."""
    return {t: {"mode_share": v["mode_share"], "distinct": v["distinct"]}
            for t, v in stability.items()}
