"""
calib -- expected calibration error for the judged tracks.

THE GAP THIS CLOSES.

Phase 2 verified that the CC value is sourced from `probabilities[chosen]` and
not from `confidence`. It did NOT verify that either is a CALIBRATED
probability. Those are different claims:

    sourced correctly   -- the right number is being read
    calibrated          -- a stated 0.9 happens about 90% of the time

For an agent that can act autonomously on these numbers, the second claim is
the load-bearing one. An uncalibrated 0.9 that means 0.55 is worse than no
number at all, because a threshold on it will fire at the wrong rate and the
failure is invisible -- every individual decision looks reasonable.

WHY THE NAIVE VERSION IS NOT ENOUGH.

Two traps, both of which this repo has already hit elsewhere:

1. Platt scaling is NOT a safe fix. decision-ledger measured it making an
   already-calibrated model WORSE (ECE 0.033 -> 0.053). So this module
   measures and reports, and does not "correct" anything by default.

2. The model is a SAMPLER. This was measured directly on the Jev gate: five
   identical runs of one model over a fixed corpus gave AUC 0.913, 0.917,
   0.921, 0.921, 0.940. So a single-pass ECE is a sample from a distribution,
   not a number. Anything below the repeat-to-repeat spread is noise, and
   reporting one pass as "the ECE" would be the same false-green move as
   reporting one AUC run.

WHAT ECE IS NOT.

It measures aggregate agreement between stated probability and observed
frequency across a labelled set. It does NOT tell you:

  * whether the probe set represents real conversation -- the labels here are
    one author's judgements, not a user's, so this measures agreement with a
    reasonable annotator, not truth;
  * whether the model is right per-item. A model can be well calibrated and
    systematically wrong, if it is confidently wrong at the right rate.
  * anything about per-utterance reliability, which is what an agent acting on
    one decision actually needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Bin:
    lo: float
    hi: float
    n: int = 0
    conf_sum: float = 0.0
    correct: int = 0

    @property
    def confidence(self) -> float:
        return self.conf_sum / self.n if self.n else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def gap(self) -> float:
        return self.accuracy - self.confidence


@dataclass
class CalibrationResult:
    ece: float
    n: int
    bins: list[Bin] = field(default_factory=list)
    base_rate: float = 0.0
    mean_confidence: float = 0.0

    def worst_bin(self) -> Bin | None:
        populated = [b for b in self.bins if b.n]
        return max(populated, key=lambda b: abs(b.gap)) if populated else None


def expected_calibration_error(pairs: list[tuple[float, int]],
                               n_bins: int = 10) -> CalibrationResult:
    """
    Standard binned ECE over (stated confidence, binary outcome) pairs.

    Equal-width bins. There is a real argument for equal-mass bins here, and
    against: equal-width puts most of a skewed distribution into the top bin,
    so the ECE ends up describing one region. This reports the bins
    explicitly so that is visible rather than hidden inside one number, and
    `worst_bin()` exists so a reader can check it.
    """
    if not pairs:
        return CalibrationResult(0.0, 0)

    bins = [Bin(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    for conf, correct in pairs:
        p = max(0.0, min(1.0, float(conf)))
        idx = min(n_bins - 1, int(p * n_bins))
        b = bins[idx]
        b.n += 1
        b.conf_sum += p
        b.correct += 1 if correct else 0

    total = len(pairs)
    ece = sum((b.n / total) * abs(b.gap) for b in bins if b.n)
    return CalibrationResult(
        ece=ece,
        n=total,
        bins=bins,
        base_rate=sum(c for _, c in pairs) / total,
        mean_confidence=sum(p for p, _ in pairs) / total,
    )


def adaptive_calibration_error(pairs: list[tuple[float, int]],
                               n_bins: int = 10) -> CalibrationResult:
    """
    Equal-MASS variant: bins of equal COUNT, not equal width.

    Reported alongside the equal-width version because on a distribution as
    skewed as these probabilities, equal-width can put almost everything in
    the last bin and call the result calibrated on the strength of one region.
    Two numbers disagreeing is more information than one number.
    """
    if not pairs:
        return CalibrationResult(0.0, 0)
    ordered = sorted(pairs)
    per = max(1, len(ordered) // n_bins)
    bins: list[Bin] = []
    for i in range(0, len(ordered), per):
        chunk = ordered[i:i + per]
        b = Bin(chunk[0][0], chunk[-1][0], n=len(chunk),
                conf_sum=sum(p for p, _ in chunk),
                correct=sum(c for _, c in chunk))
        bins.append(b)
    total = len(pairs)
    ece = sum((b.n / total) * abs(b.gap) for b in bins if b.n)
    return CalibrationResult(ece, total, bins,
                             sum(c for _, c in pairs) / total,
                             sum(p for p, _ in pairs) / total)


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def worst_of(results: list[CalibrationResult]) -> tuple[float, float, int]:
    """
    Given repeated measurements of the SAME thing, report the MEAN and the
    SPREAD.

    Reporting the best pass is the false-green move. Reporting a single pass
    when the model is a sampler is the same mistake in subtler clothing, so
    ECE is always reported as `mean +/- stdev` with the number of passes
    stated.
    """
    vals = [r.ece for r in results]
    return sum(vals) / len(vals), stdev(vals), len(vals)