#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ds4_gate_stats import (  # noqa: E402
    StatsError,
    bca_interval,
    bca_mean,
    exact_sign_test,
    paired_log_ratio_interval,
    paired_log_ratio_repeated_interval,
    stratified_resample_indices,
    wilson_one_sided,
)


class GateStatsTests(unittest.TestCase):
    def test_constant_bca_is_exact(self) -> None:
        result = bca_mean([0.25] * 12, n_resamples=1999, seed=7)
        self.assertEqual(result["estimate"], 0.25)
        self.assertEqual(result["one_sided_lower"], 0.25)
        self.assertEqual(result["one_sided_upper"], 0.25)

    def test_bca_tracks_a_shift_without_crossing_the_mean(self) -> None:
        first = bca_mean([-2.0, -1.0, 0.0, 1.0, 2.0],
                         n_resamples=3999, seed=11)
        second = bca_mean([8.0, 9.0, 10.0, 11.0, 12.0],
                          n_resamples=3999, seed=11)
        self.assertAlmostEqual(second["estimate"] - first["estimate"], 10.0)
        self.assertAlmostEqual(
            second["one_sided_lower"] - first["one_sided_lower"], 10.0)
        self.assertAlmostEqual(
            second["one_sided_upper"] - first["one_sided_upper"], 10.0)
        self.assertLess(first["one_sided_lower"], first["estimate"])
        self.assertGreater(first["one_sided_upper"], first["estimate"])

    def test_ratio_of_totals_resamples_whole_cases(self) -> None:
        numerator = np.asarray([1.0, 2.0, 9.0, 18.0])
        denominator = np.asarray([2.0, 4.0, 10.0, 20.0])
        result = bca_interval(
            4,
            lambda indices: float(numerator[indices].sum() /
                                  denominator[indices].sum()),
            strata=["short", "short", "long", "long"],
            n_resamples=3999,
            seed=13,
        )
        self.assertAlmostEqual(result["estimate"], 30.0 / 36.0)
        self.assertLessEqual(result["one_sided_lower"], result["estimate"])
        self.assertGreaterEqual(result["one_sided_upper"], result["estimate"])

    def test_stratified_bootstrap_preserves_group_counts(self) -> None:
        sampled = stratified_resample_indices(
            ["a", "a", "b", "b", "b"], 1000, 17)
        self.assertEqual(sampled.shape, (1000, 5))
        self.assertTrue(np.all(sampled[:, :2] < 2))
        self.assertTrue(np.all(sampled[:, 2:] >= 2))

    def test_wilson_zero_failures_still_has_uncertainty(self) -> None:
        success = wilson_one_sided(300, 300)
        failure = wilson_one_sided(0, 300)
        self.assertLess(success["one_sided_lower"], 1.0)
        self.assertGreater(failure["one_sided_upper"], 0.0)

    def test_paired_log_ratio_interval(self) -> None:
        timing = paired_log_ratio_interval(
            [10.0, 10.1, 9.9], [12.0, 12.12, 11.88])
        self.assertEqual(timing["pairs"], 3)
        self.assertAlmostEqual(timing["geometric_mean_change"], 0.2)
        self.assertGreater(timing["one_sided_lower"], 0.19)
        noisy = paired_log_ratio_interval(
            [10.0, 10.0, 10.0], [10.8, 9.6, 10.4])
        self.assertLess(noisy["one_sided_lower"], 0.0)
        self.assertGreater(noisy["geometric_mean_change"], 0.0)

    def test_paired_timing_supports_bounded_escalation(self) -> None:
        for count in (5, 7, 9):
            timing = paired_log_ratio_interval(
                [10.0] * count, [11.0] * count)
            self.assertEqual(timing["pairs"], count)
            self.assertAlmostEqual(timing["one_sided_lower"], 0.1)

    def test_paired_timing_rejects_unregistered_sample_count(self) -> None:
        with self.assertRaises(StatsError):
            paired_log_ratio_interval([1.0] * 4, [1.1] * 4)

    def test_repeated_student_uses_one_boundary_at_every_formal_look(self) -> None:
        for count in (5, 7, 9):
            result = paired_log_ratio_repeated_interval(
                [10.0] * count, [12.0] * count)
            self.assertEqual(result["pairs"], count)
            self.assertEqual(result["boundary"], 2.52)
            self.assertAlmostEqual(result["one_sided_lower"], 0.2)
        with self.assertRaises(StatsError):
            paired_log_ratio_repeated_interval([10.0] * 3, [12.0] * 3)

    def test_exact_sign_fallback_requires_eight_of_nine_and_fails_ties(self) -> None:
        passed = exact_sign_test([0.02] * 8 + [0.0], 0.01)
        failed = exact_sign_test([0.02] * 7 + [0.01, 0.0], 0.01)
        self.assertEqual(passed["positive_adjusted_effects"], 8)
        self.assertLess(passed["p_value"], 0.05)
        self.assertEqual(failed["positive_adjusted_effects"], 7)
        self.assertEqual(failed["ties"], 1)
        self.assertGreater(failed["p_value"], 0.05)
        rounded_ties = exact_sign_test(
            [math.nextafter(0.01, math.inf)] * 9, 0.01)
        self.assertEqual(rounded_ties["positive_adjusted_effects"], 0)
        self.assertEqual(rounded_ties["ties"], 9)

    def test_invalid_coverage_fails_closed(self) -> None:
        with self.assertRaises(StatsError):
            bca_mean([1.0], n_resamples=1999, seed=1)
        with self.assertRaises(StatsError):
            wilson_one_sided(2, 1)


if __name__ == "__main__":
    unittest.main()
