#!/usr/bin/env python3
"""Paired, fail-closed quality comparison for DS4 score_official TSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

from ds4_gate_stats import StatsError, bca_interval, bca_mean


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path}: empty score table")
    required = {
        "id", "target_tokens", "nll", "avg_nll", "api_top1_count",
        "api_top1_match", "api_pair_total", "api_pair_agree",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(sorted(missing))}")
    return rows


def manifest_for(path: Path) -> Path:
    return path.with_suffix(".manifest")


def validate_pair(reference: Path, candidate: Path) -> None:
    manifests = []
    for path in (reference, candidate):
        metadata = dict(line.split("=", 1) for line in
                        manifest_for(path).read_text().splitlines() if "=" in line)
        manifests.append(metadata)
    for key in ("model_size", "model_sample_sha256", "quality_input_sha256"):
        if not manifests[0].get(key) or manifests[0][key] != manifests[1].get(key):
            raise ValueError(f"reference and candidate {key} differ or are missing")


def number(row: dict[str, str], key: str, integer: bool = False) -> float | int:
    try:
        value = int(row[key]) if integer else float(row[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"case {row.get('id', '?')}: invalid {key}") from error
    if not integer and not math.isfinite(value):
        raise ValueError(f"case {row.get('id', '?')}: non-finite {key}")
    return value


def load_thresholds(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") == 2:
        required = {
            "schema_version", "baseline_id", "min_cases", "min_target_tokens",
            "bootstrap", "nll", "api",
        }
        if set(value) != required:
            raise ValueError(f"{path}: invalid v2 quality threshold fields")
        if not isinstance(value["baseline_id"], str) or not value["baseline_id"]:
            raise ValueError(f"{path}: invalid baseline_id")
        for key in ("min_cases", "min_target_tokens"):
            if not isinstance(value[key], int) or value[key] <= 0:
                raise ValueError(f"{path}: {key} must be a positive integer")
        bootstrap = value["bootstrap"]
        nll = value["nll"]
        api = value["api"]
        if not all(isinstance(item, dict) for item in (bootstrap, nll, api)):
            raise ValueError(f"{path}: v2 quality sections must be objects")
        if set(bootstrap) != {
                "method", "resamples", "seed", "nll_confidence_level",
                "api_confidence_level"} or bootstrap.get("method") != "bca":
            raise ValueError(f"{path}: invalid v2 bootstrap contract")
        if (not isinstance(bootstrap["resamples"], int) or
                bootstrap["resamples"] < 1000 or
                not isinstance(bootstrap["seed"], int) or
                isinstance(bootstrap["seed"], bool) or bootstrap["seed"] < 0):
            raise ValueError(f"{path}: invalid deterministic bootstrap settings")
        for key in ("nll_confidence_level", "api_confidence_level"):
            item = bootstrap[key]
            if not isinstance(item, (int, float)) or not 0.5 < item < 1.0:
                raise ValueError(f"{path}: {key} must be between 0.5 and 1")
        if set(nll) != {"max_delta_upper", "max_case_delta"}:
            raise ValueError(f"{path}: invalid v2 NLL threshold fields")
        if set(api) != {
                "required", "min_cases", "min_top1_delta_lower",
                "min_pair_delta_lower"} or type(api.get("required")) is not bool:
            raise ValueError(f"{path}: invalid v2 API threshold fields")
        if not isinstance(api["min_cases"], int) or api["min_cases"] < 0:
            raise ValueError(f"{path}: api.min_cases must be nonnegative")
        for label, item in {
                "nll.max_delta_upper": nll["max_delta_upper"],
                "nll.max_case_delta": nll["max_case_delta"],
                "api.min_top1_delta_lower": api["min_top1_delta_lower"],
                "api.min_pair_delta_lower": api["min_pair_delta_lower"],
        }.items():
            if not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ValueError(f"{path}: {label} must be finite")
        return value

    required = {
        "baseline_id", "min_cases", "min_target_tokens",
        "max_mean_nll_delta", "max_ci95_high_nll_delta",
        "min_api_top1_rate_delta", "min_api_pair_rate_delta",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"{path}: missing thresholds: {', '.join(sorted(missing))}")
    if not isinstance(value["baseline_id"], str) or not value["baseline_id"]:
        raise ValueError(f"{path}: invalid baseline_id")
    for key in ("min_cases", "min_target_tokens"):
        if not isinstance(value[key], int) or value[key] <= 0:
            raise ValueError(f"{path}: {key} must be a positive integer")
    for key in required - {"baseline_id", "min_cases", "min_target_tokens"}:
        if not isinstance(value[key], (int, float)) or not math.isfinite(value[key]):
            raise ValueError(f"{path}: {key} must be finite")
    if "max_case_nll_delta" in value and (
            not isinstance(value["max_case_nll_delta"], (int, float)) or
            not math.isfinite(value["max_case_nll_delta"])):
        raise ValueError(f"{path}: max_case_nll_delta must be finite")
    return value


def ratio(rows: list[dict[str, str]], numerator: str,
          denominator: str) -> float | None:
    top = sum(number(row, numerator, True) for row in rows)
    bottom = sum(number(row, denominator, True) for row in rows)
    if bottom <= 0:
        return None
    return top / bottom


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        thresholds = load_thresholds(args.thresholds)
        reference = load_rows(args.reference)
        candidate = load_rows(args.candidate)
        validate_pair(args.reference, args.candidate)
        if [row["id"] for row in reference] != [row["id"] for row in candidate]:
            raise ValueError("reference and candidate case ids/order differ")
        if len({row["id"] for row in reference}) != len(reference):
            raise ValueError("duplicate case ids do not count as independent coverage")
        v2 = thresholds.get("schema_version") == 2
        deltas: list[float] = []
        nll_differences: list[float] = []
        token_counts: list[int] = []
        strata: list[str] = []
        ref_top1_matches: list[int] = []
        cand_top1_matches: list[int] = []
        top1_counts: list[int] = []
        ref_pair_matches: list[int] = []
        cand_pair_matches: list[int] = []
        pair_counts: list[int] = []
        ref_nll = cand_nll = 0.0
        target_tokens = 0
        for ref_row, cand_row in zip(reference, candidate):
            for denominator, numerator in (("api_top1_count", "api_top1_match"),
                                           ("api_pair_total", "api_pair_agree")):
                ref_count = number(ref_row, denominator, True)
                cand_count = number(cand_row, denominator, True)
                if ref_count < 0 or cand_count != ref_count:
                    raise ValueError(f"case {ref_row['id']}: API coverage differs")
                for row in (ref_row, cand_row):
                    if not 0 <= number(row, numerator, True) <= ref_count:
                        raise ValueError(f"case {row['id']}: invalid API agreement count")
            ref_tokens = number(ref_row, "target_tokens", True)
            cand_tokens = number(cand_row, "target_tokens", True)
            if ref_tokens <= 0 or cand_tokens != ref_tokens:
                raise ValueError(f"case {ref_row['id']}: target-token counts differ")
            ref_value = number(ref_row, "nll")
            cand_value = number(cand_row, "nll")
            ref_avg = number(ref_row, "avg_nll")
            cand_avg = number(cand_row, "avg_nll")
            if not math.isclose(ref_value / ref_tokens, ref_avg, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"case {ref_row['id']}: reference avg_nll is inconsistent")
            if not math.isclose(cand_value / cand_tokens, cand_avg, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"case {ref_row['id']}: candidate avg_nll is inconsistent")
            ref_nll += ref_value
            cand_nll += cand_value
            target_tokens += ref_tokens
            deltas.append(cand_avg - ref_avg)
            nll_differences.append(cand_value - ref_value)
            token_counts.append(ref_tokens)
            ref_stratum = ref_row.get("stratum", "all") or "all"
            cand_stratum = cand_row.get("stratum", "all") or "all"
            if ref_stratum != cand_stratum:
                raise ValueError(f"case {ref_row['id']}: quality strata differ")
            strata.append(ref_stratum)
            ref_top1_matches.append(number(ref_row, "api_top1_match", True))
            cand_top1_matches.append(number(cand_row, "api_top1_match", True))
            top1_counts.append(number(ref_row, "api_top1_count", True))
            ref_pair_matches.append(number(ref_row, "api_pair_agree", True))
            cand_pair_matches.append(number(cand_row, "api_pair_agree", True))
            pair_counts.append(number(ref_row, "api_pair_total", True))
        if len(deltas) < 2:
            raise ValueError("at least two paired cases are required for an uncertainty bound")
        mean_delta = statistics.fmean(deltas)
        ci_half = 1.96 * statistics.stdev(deltas) / math.sqrt(len(deltas))
        weighted_delta = (cand_nll - ref_nll) / target_tokens
        max_case_delta = max(deltas)
        ref_top1 = ratio(reference, "api_top1_match", "api_top1_count")
        cand_top1 = ratio(candidate, "api_top1_match", "api_top1_count")
        ref_pair = ratio(reference, "api_pair_agree", "api_pair_total")
        cand_pair = ratio(candidate, "api_pair_agree", "api_pair_total")
        api_metrics_available = all(
            value is not None
            for value in (ref_top1, cand_top1, ref_pair, cand_pair))
        api_top1_delta = (
            cand_top1 - ref_top1 if cand_top1 is not None and
            ref_top1 is not None else None)
        api_pair_delta = (
            cand_pair - ref_pair if cand_pair is not None and
            ref_pair is not None else None)
        bootstrap_result = None
        if v2:
            bootstrap = thresholds["bootstrap"]
            nll_difference_array = np.asarray(nll_differences, dtype=np.float64)
            token_array = np.asarray(token_counts, dtype=np.float64)
            try:
                weighted_nll_interval = bca_interval(
                    len(deltas),
                    lambda indices: float(
                        nll_difference_array[indices].sum(dtype=np.float64) /
                        token_array[indices].sum(dtype=np.float64)),
                    strata=strata,
                    confidence_level=bootstrap["nll_confidence_level"],
                    n_resamples=bootstrap["resamples"],
                    seed=bootstrap["seed"],
                )
                case_mean_interval = bca_mean(
                    deltas,
                    strata=strata,
                    confidence_level=bootstrap["nll_confidence_level"],
                    n_resamples=bootstrap["resamples"],
                    seed=bootstrap["seed"] + 1,
                )

                def api_interval(reference_values: list[int],
                                 candidate_values: list[int],
                                 denominators: list[int], seed: int) -> tuple[dict | None, int]:
                    covered = [index for index, count in enumerate(denominators)
                               if count > 0]
                    if len(covered) < 2:
                        return None, len(covered)
                    reference_array = np.asarray(
                        [reference_values[index] for index in covered],
                        dtype=np.float64)
                    candidate_array = np.asarray(
                        [candidate_values[index] for index in covered],
                        dtype=np.float64)
                    denominator_array = np.asarray(
                        [denominators[index] for index in covered],
                        dtype=np.float64)
                    covered_strata = [strata[index] for index in covered]
                    return bca_interval(
                        len(covered),
                        lambda indices: float(
                            candidate_array[indices].sum(dtype=np.float64) /
                            denominator_array[indices].sum(dtype=np.float64) -
                            reference_array[indices].sum(dtype=np.float64) /
                            denominator_array[indices].sum(dtype=np.float64)),
                        strata=covered_strata,
                        confidence_level=bootstrap["api_confidence_level"],
                        n_resamples=bootstrap["resamples"],
                        seed=seed,
                    ), len(covered)

                top1_interval, top1_cases = api_interval(
                    ref_top1_matches, cand_top1_matches, top1_counts,
                    bootstrap["seed"] + 2)
                pair_interval, pair_cases = api_interval(
                    ref_pair_matches, cand_pair_matches, pair_counts,
                    bootstrap["seed"] + 3)
            except StatsError as error:
                raise ValueError(f"invalid paired bootstrap: {error}") from error
            bootstrap_result = {
                "method": bootstrap["method"],
                "resamples": bootstrap["resamples"],
                "seed": bootstrap["seed"],
                "nll_confidence_level": bootstrap["nll_confidence_level"],
                "api_confidence_level": bootstrap["api_confidence_level"],
                "weighted_nll_delta": weighted_nll_interval,
                "case_mean_nll_delta": case_mean_interval,
                "api_top1_rate_delta": top1_interval,
                "api_top1_cases": top1_cases,
                "api_pair_rate_delta": pair_interval,
                "api_pair_cases": pair_cases,
            }
        metrics = {
            "cases": len(deltas),
            "target_tokens": target_tokens,
            "reference_avg_nll": ref_nll / target_tokens,
            "candidate_avg_nll": cand_nll / target_tokens,
            "weighted_nll_delta": weighted_delta,
            "paired_mean_avg_nll_delta": mean_delta,
            "paired_ci95_low": mean_delta - ci_half,
            "paired_ci95_high": mean_delta + ci_half,
            "max_case_avg_nll_delta": max_case_delta,
            "api_metrics_available": api_metrics_available,
            "api_top1_rate_delta": api_top1_delta,
            "api_pair_rate_delta": api_pair_delta,
            "paired_bootstrap": bootstrap_result,
        }
        if v2:
            assert bootstrap_result is not None
            nll_screen_passed = bool(
                metrics["cases"] >= thresholds["min_cases"] and
                metrics["target_tokens"] >= thresholds["min_target_tokens"] and
                bootstrap_result["weighted_nll_delta"]["one_sided_upper"] <=
                    thresholds["nll"]["max_delta_upper"] and
                max_case_delta <= thresholds["nll"]["max_case_delta"])
            api_required = thresholds["api"]["required"]
            api_screen_passed = bool(
                bootstrap_result["api_top1_rate_delta"] is not None and
                bootstrap_result["api_pair_rate_delta"] is not None and
                bootstrap_result["api_top1_cases"] >=
                    thresholds["api"]["min_cases"] and
                bootstrap_result["api_pair_cases"] >=
                    thresholds["api"]["min_cases"] and
                bootstrap_result["api_top1_rate_delta"]["one_sided_lower"] >=
                    thresholds["api"]["min_top1_delta_lower"] and
                bootstrap_result["api_pair_rate_delta"]["one_sided_lower"] >=
                    thresholds["api"]["min_pair_delta_lower"])
            if not api_required and not api_metrics_available:
                api_screen_passed = True
        else:
            nll_screen_passed = bool(
                metrics["cases"] >= thresholds["min_cases"] and
                metrics["target_tokens"] >= thresholds["min_target_tokens"] and
                metrics["weighted_nll_delta"] <= thresholds["max_mean_nll_delta"] and
                metrics["paired_ci95_high"] <= thresholds["max_ci95_high_nll_delta"] and
                ("max_case_nll_delta" not in thresholds or
                 max_case_delta <= thresholds["max_case_nll_delta"])
            )
            api_screen_passed = bool(
                api_metrics_available and
                api_top1_delta is not None and api_pair_delta is not None and
                api_top1_delta >= thresholds["min_api_top1_rate_delta"] and
                api_pair_delta >= thresholds["min_api_pair_rate_delta"])
        passed = nll_screen_passed and api_screen_passed
        blockers = []
        if not api_metrics_available and (not v2 or thresholds["api"]["required"]):
            blockers.append("hosted API top-1/pair logprob metrics unavailable")
        result = {
            "schema_version": 2 if v2 else 1,
            "baseline_id": thresholds["baseline_id"],
            "sources": {
                "reference": str(args.reference.resolve()),
                "reference_sha256": sha256(args.reference),
                "reference_manifest": str(manifest_for(args.reference).resolve()),
                "reference_manifest_sha256": sha256(manifest_for(args.reference)),
                "candidate": str(args.candidate.resolve()),
                "candidate_sha256": sha256(args.candidate),
                "candidate_manifest": str(manifest_for(args.candidate).resolve()),
                "candidate_manifest_sha256": sha256(manifest_for(args.candidate)),
            },
            "thresholds_sha256": sha256(args.thresholds),
            "metrics": metrics,
            "nll_screen_passed": nll_screen_passed,
            "api_screen_passed": (
                api_screen_passed if api_metrics_available or
                (v2 and not thresholds["api"]["required"]) else None),
            "blockers": blockers,
            "passed": passed,
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"quality-scores: FAIL {error}", file=sys.stderr)
        return 1
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    if not passed:
        if not metrics["api_metrics_available"]:
            print("quality-scores: FAIL hosted API metrics unavailable; "
                  "paired NLL screen was still reported", file=sys.stderr)
        else:
            print("quality-scores: FAIL paired non-inferiority gate", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
