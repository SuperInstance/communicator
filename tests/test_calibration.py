"""
Tests for the calibration harness itself.

The harness is what every future calibration claim rests on, so it gets the
same treatment the broker got: prove it can say "bad", not just "good".

A calibration metric that returns 0.05 for everything would make Phase 2's
headline look fine forever. These tests use synthetic probability sets whose
calibration is known by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calib import (adaptive_calibration_error, expected_calibration_error,  # noqa: E402
                        stdev, worst_of)


class TestKnownAnswers:
    """Probability sets whose calibration is known without running a model."""

    def test_perfectly_calibrated_is_nearly_zero(self):
        """Every stated 0.5 comes true exactly half the time."""
        pairs = [(0.5, 1)] * 50 + [(0.5, 0)] * 50
        r = expected_calibration_error(pairs)
        assert r.ece < 0.01, r.ece
        assert r.n == 100

    def test_maximally_overconfident_is_large(self):
        """Says 0.99 and is right 10% of the time. This must NOT read as good."""
        pairs = [(0.99, 1)] * 10 + [(0.99, 0)] * 90
        r = expected_calibration_error(pairs)
        assert r.ece > 0.5, r.ece

    def test_maximally_underconfident_is_also_large(self):
        """Says 0.01 and is right 90% of the time. Underconfidence is a real
        error, not a safe one -- it wastes the threshold's whole range."""
        pairs = [(0.01, 1)] * 90 + [(0.01, 0)] * 10
        r = expected_calibration_error(pairs)
        assert r.ece > 0.5, r.ece

    def test_ece_cannot_detect_a_symmetrically_inverted_model(self):
        """
        A BLIND SPOT, documented rather than papered over.

        This test was originally written to assert that ECE catches a model
        which is exactly backwards. It does not, and the reason matters:

            bin 0.2-0.3:  5 right / 50     -> accuracy 0.10, states 0.2
            bin 0.9-1.0: 45 right / 50     -> accuracy 0.90, states 0.9

        The errors cancel. ECE reads 0.05 -- "well calibrated" -- for a model
        that is exactly wrong, because within each confidence band the right
        fraction happened to be right.

        So ECE is a statement about the DISTRIBUTION of errors, not about any
        of them. A good ECE is necessary for trusting a threshold and is not
        sufficient: this system would also need an accuracy or ranking check
        alongside it, and until that exists the calibration number in
        PROGRESS.md must not be quoted as if it stood alone.
        """
        pairs = ([(0.2, 1)] * 5 + [(0.2, 0)] * 45
                 + [(0.9, 0)] * 5 + [(0.9, 1)] * 45)
        r = expected_calibration_error(pairs)
        assert r.ece < 0.10, r.ece          # looks calibrated
        assert r.base_rate == 0.50          # and is not even guessing
        # ...while being maximally wrong in both tails.
        low = next(b for b in r.bins if b.lo < 0.3)
        high = next(b for b in r.bins if b.lo > 0.8)
        assert low.accuracy < 0.2           # stated 0.2, actually 0.10
        assert high.accuracy > 0.8          # stated 0.9, actually 0.90
        assert abs(low.gap) + abs(high.gap) < 0.15, \
            "errors cancel in aggregate -- that IS the blind spot"

    def test_ece_is_bounded(self):
        pairs = [(i / 100, i % 2) for i in range(100)]
        assert 0.0 <= expected_calibration_error(pairs).ece <= 1.0

    def test_empty_input_is_zero_not_a_crash(self):
        r = expected_calibration_error([])
        assert r.ece == 0.0 and r.n == 0
        assert r.worst_bin() is None

    def test_out_of_range_confidence_is_clamped_into_the_top_bin(self):
        """A model reporting 1.4 must not silently create bin 14."""
        pairs = [(1.4, 1)] * 5 + [(-0.2, 0)] * 5
        r = expected_calibration_error(pairs)
        assert len(r.bins) == 10
        assert sum(b.n for b in r.bins) == 10


class TestBinning:
    def test_equal_mass_separates_a_skewed_distribution(self):
        """
        The reason both estimators exist. When most mass piles into one
        equal-width bin, the headline ECE describes that one region and the
        rest is invisible -- which is exactly the shape of the measured
        distribution, where 120 of 192 pairs landed under 0.1.
        """
        skewed = [(0.02, 0)] * 90 + [(0.85, 1)] * 5 + [(0.85, 0)] * 5
        eq = expected_calibration_error(skewed, n_bins=10)
        am = adaptive_calibration_error(skewed, n_bins=10)
        # Equal-mass must use more bins on a 90/10 split.
        assert len([b for b in am.bins if b.n]) > len([b for b in eq.bins if b.n])

    def test_both_estimators_agree_on_a_fine_grained_set(self):
        pairs = [(i / 200, int(i / 200 > 0.5)) for i in range(200)]
        eq = expected_calibration_error(pairs).ece
        am = adaptive_calibration_error(pairs).ece
        assert abs(eq - am) < 0.05, (eq, am)

    def test_worst_bin_points_at_the_largest_gap(self):
        pairs = [(0.5, 1)] * 40 + [(0.5, 0)] * 10 + [(0.05, 0)] * 50
        r = expected_calibration_error(pairs)
        w = r.worst_bin()
        assert w is not None
        assert abs(w.gap) == max(abs(b.gap) for b in r.bins if b.n)


class TestRepeatReporting:
    def test_worst_of_reports_mean_and_spread(self):
        results = [expected_calibration_error([(0.5, 1), (0.5, 0)]) for _ in range(3)]
        mean, sd, n = worst_of(results)
        assert n == 3
        assert mean == 0.0 and sd == 0.0

    def test_worst_of_reports_the_spread_not_the_best(self):
        """A model measured three times must not be reported as its best pass."""
        results = [expected_calibration_error([(0.5, 1), (0.5, 0)]) for _ in range(2)]
        results.append(expected_calibration_error([(0.9, 1), (0.9, 0)]))
        mean, sd, n = worst_of(results)
        assert n == 3
        assert sd > 0, "a spread across repeats must be visible"
        assert mean != max(r.ece for r in results), \
            "reported the best pass instead of the mean"

    def test_stdev_of_one_sample_is_zero_not_a_crash(self):
        assert stdev([0.5]) == 0.0
        assert stdev([]) == 0.0

    def test_stdev_matches_the_sample_formula(self):
        """Sample stdev, n-1. For [0.1,0.2,0.3] that is exactly 0.1."""
        assert abs(stdev([0.1, 0.2, 0.3]) - 0.1) < 1e-9
        # and differs from the population form, which would give ~0.0816
        assert abs(stdev([0.1, 0.2, 0.3]) - 0.08164965809277261) > 1e-3