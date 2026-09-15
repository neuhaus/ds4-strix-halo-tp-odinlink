#!/usr/bin/env python3
"""Deterministic, dependency-light statistics for DS4 promotion gates."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Callable, Sequence

import numpy as np


class StatsError(ValueError):
    pass


# One-sided Student-t quantiles. Promotion timing uses odd-sized, order-balanced
# escalation steps from three through nine matched pairs, so keeping the
# audited values local avoids a heavyweight scipy dependency in the gate.
_T_CRITICAL = {
    0.95: {2: 2.919986, 4: 2.131847, 6: 1.943180, 8: 1.859548},
    0.99: {2: 6.964557, 4: 3.746947, 6: 3.142668, 8: 2.896459},
}

REPEATED_STUDENT_BOUNDARY = 2.52


def _probability(value: float, label: str) -> float:
    if not math.isfinite(value) or not 0.5 < value < 1.0:
        raise StatsError(f"{label} must be between 0.5 and 1")
    return value


def stratified_resample_indices(strata: Sequence[str], n_resamples: int,
                                seed: int) -> np.ndarray:
    """Resample whole observations within each fixed stratum."""
    if not isinstance(n_resamples, int) or n_resamples < 1000:
        raise StatsError("n_resamples must be an integer of at least 1000")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise StatsError("seed must be a nonnegative integer")
    labels = np.asarray(list(strata), dtype=object)
    if labels.ndim != 1 or labels.size < 2 or any(not str(item) for item in labels):
        raise StatsError("at least two nonempty observation strata are required")
    groups = [np.flatnonzero(labels == item)
              for item in dict.fromkeys(str(item) for item in labels)]
    rng = np.random.default_rng(seed)
    sampled = np.empty((n_resamples, labels.size), dtype=np.int64)
    offset = 0
    for group in groups:
        width = group.size
        sampled[:, offset:offset + width] = group[
            rng.integers(0, width, size=(n_resamples, width))]
        offset += width
    return sampled


def _bca_quantile(observed: float, bootstrap: np.ndarray,
                  jackknife: np.ndarray, probability: float) -> float:
    if (bootstrap.ndim != 1 or bootstrap.size < 1000 or
            jackknife.ndim != 1 or jackknife.size < 2 or
            not np.all(np.isfinite(bootstrap)) or
            not np.all(np.isfinite(jackknife)) or
            not math.isfinite(observed)):
        raise StatsError("BCa inputs must be finite and sufficiently covered")
    if np.all(bootstrap == bootstrap[0]):
        return float(bootstrap[0])

    less = float(np.count_nonzero(bootstrap < observed))
    equal = float(np.count_nonzero(bootstrap == observed))
    rank = (less + 0.5 * equal) / bootstrap.size
    epsilon = 0.5 / bootstrap.size
    rank = min(1.0 - epsilon, max(epsilon, rank))
    normal = NormalDist()
    bias = normal.inv_cdf(rank)

    jack_mean = float(jackknife.mean(dtype=np.float64))
    centered = jack_mean - jackknife
    denominator = 6.0 * float(np.sum(centered * centered) ** 1.5)
    acceleration = (float(np.sum(centered * centered * centered)) / denominator
                    if denominator else 0.0)
    z = normal.inv_cdf(probability)
    divisor = 1.0 - acceleration * (bias + z)
    adjusted = normal.cdf(bias + (bias + z) / divisor) if divisor else probability
    adjusted = min(1.0, max(0.0, adjusted))
    return float(np.quantile(bootstrap, adjusted, method="linear"))


def bca_interval(sample_count: int, statistic: Callable[[np.ndarray], float],
                 *, strata: Sequence[str] | None = None,
                 confidence_level: float = 0.95,
                 n_resamples: int = 9999, seed: int = 0) -> dict[str, float]:
    """Return paired/clustered BCa one-sided bounds for an arbitrary statistic.

    ``statistic`` receives integer observation indices. Resampling therefore
    preserves all values belonging to one case and can recompute ratios of
    totals instead of treating token alternatives as independent samples.
    """
    _probability(confidence_level, "confidence_level")
    if not isinstance(sample_count, int) or sample_count < 2:
        raise StatsError("sample_count must be an integer of at least two")
    labels = list(strata) if strata is not None else ["all"] * sample_count
    if len(labels) != sample_count:
        raise StatsError("strata length differs from sample_count")
    sampled = stratified_resample_indices(labels, n_resamples, seed)
    observed = float(statistic(np.arange(sample_count, dtype=np.int64)))
    bootstrap = np.asarray([statistic(indices) for indices in sampled],
                           dtype=np.float64)
    jackknife = np.asarray([
        statistic(np.delete(np.arange(sample_count, dtype=np.int64), index))
        for index in range(sample_count)
    ], dtype=np.float64)
    alpha = 1.0 - confidence_level
    return {
        "estimate": observed,
        "one_sided_lower": _bca_quantile(
            observed, bootstrap, jackknife, alpha),
        "one_sided_upper": _bca_quantile(
            observed, bootstrap, jackknife, confidence_level),
    }


def bca_mean(values: Sequence[float], **kwargs: object) -> dict[str, float]:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size < 2 or not np.all(np.isfinite(sample)):
        raise StatsError("mean sample must contain at least two finite values")
    strata = kwargs.pop("strata", None)
    return bca_interval(
        int(sample.size),
        lambda indices: float(sample[indices].mean(dtype=np.float64)),
        strata=strata,
        **kwargs,
    )


def wilson_one_sided(successes: int, trials: int,
                     confidence_level: float = 0.95) -> dict[str, float]:
    """Return one-sided Wilson score bounds for a binomial rate."""
    _probability(confidence_level, "confidence_level")
    if (not isinstance(successes, int) or not isinstance(trials, int) or
            trials <= 0 or not 0 <= successes <= trials):
        raise StatsError("Wilson inputs require 0 <= successes <= trials")
    z = NormalDist().inv_cdf(confidence_level)
    rate = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (rate + z2 / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        rate * (1.0 - rate) / trials + z2 / (4.0 * trials * trials)
    ) / denominator
    return {
        "estimate": rate,
        "one_sided_lower": max(0.0, center - half),
        "one_sided_upper": min(1.0, center + half),
    }


def paired_log_ratio_interval(control: Sequence[float],
                              candidate: Sequence[float], *,
                              confidence_level: float = 0.95) -> dict[str, float]:
    """Infer a multiplicative effect from 3/5/7/9 matched run pairs.

    Ratios are analyzed in log space so reciprocal slowdowns and speedups are
    treated symmetrically. Returned changes are fractions: 0.02 means +2%.
    """
    left = np.asarray(control, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if (left.ndim != 1 or right.ndim != 1 or left.size != right.size or
            left.size not in {3, 5, 7, 9} or not np.all(np.isfinite(left)) or
            not np.all(np.isfinite(right)) or np.any(left <= 0) or
            np.any(right <= 0)):
        raise StatsError(
            "paired timing requires 3, 5, 7, or 9 positive finite matched pairs")
    degrees_freedom = int(left.size - 1)
    if degrees_freedom not in _T_CRITICAL.get(confidence_level, {}):
        raise StatsError(
            "paired timing confidence is not registered for this sample count")
    effects = np.log(right / left)
    estimate = float(effects.mean(dtype=np.float64))
    standard_error = float(effects.std(ddof=1)) / math.sqrt(effects.size)
    radius = _T_CRITICAL[confidence_level][degrees_freedom] * standard_error
    return {
        "pairs": int(effects.size),
        "confidence_level": confidence_level,
        "geometric_mean_change": math.exp(estimate) - 1.0,
        "median_change": math.exp(float(np.median(effects))) - 1.0,
        "one_sided_lower": math.exp(estimate - radius) - 1.0,
        "one_sided_upper": math.exp(estimate + radius) - 1.0,
        "log_standard_error": standard_error,
    }


def paired_log_ratio_repeated_interval(
        control: Sequence[float], candidate: Sequence[float], *,
        boundary: float = REPEATED_STUDENT_BOUNDARY) -> dict[str, float]:
    """Return a repeated interval for the predeclared nested 5/7/9 looks.

    The constant is calibrated jointly across all three looks by
    ``calibrate-promotion-boundary.py``. It must not be interpreted as an
    ordinary fixed-sample Student-t confidence level.
    """
    left = np.asarray(control, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if (left.ndim != 1 or right.ndim != 1 or left.size != right.size or
            left.size not in {5, 7, 9} or not np.all(np.isfinite(left)) or
            not np.all(np.isfinite(right)) or np.any(left <= 0) or
            np.any(right <= 0) or not math.isfinite(boundary) or boundary <= 0):
        raise StatsError(
            "repeated timing requires 5, 7, or 9 positive matched pairs and "
            "a positive boundary")
    effects = np.log(right / left)
    estimate = float(effects.mean(dtype=np.float64))
    standard_error = float(effects.std(ddof=1)) / math.sqrt(effects.size)
    radius = boundary * standard_error
    return {
        "pairs": int(effects.size),
        "boundary": boundary,
        "geometric_mean_change": math.exp(estimate) - 1.0,
        "median_change": math.exp(float(np.median(effects))) - 1.0,
        "one_sided_lower": math.exp(estimate - radius) - 1.0,
        "one_sided_upper": math.exp(estimate + radius) - 1.0,
        "log_standard_error": standard_error,
    }


def exact_sign_test(changes: Sequence[float], required: float) -> dict[str, float | int]:
    """One-sided fixed-nine sign test; exact-boundary ties count as failures."""
    values = np.asarray(changes, dtype=np.float64)
    if (values.ndim != 1 or values.size != 9 or
            not np.all(np.isfinite(values)) or not math.isfinite(required)):
        raise StatsError("exact sign promotion requires nine finite paired changes")
    ties_mask = np.isclose(values, required, rtol=1e-12, atol=1e-12)
    positives = int(np.count_nonzero((values > required) & ~ties_mask))
    ties = int(np.count_nonzero(ties_mask))
    p_value = sum(math.comb(9, index) for index in range(positives, 10)) / 512.0
    return {
        "pairs": 9,
        "positive_adjusted_effects": positives,
        "ties": ties,
        "minimum_positive": 8,
        "p_value": p_value,
    }
