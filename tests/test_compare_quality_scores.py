#!/usr/bin/env python3
import csv
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts" / "compare-quality-scores.py"
FIELDS = [
    "id", "target_tokens", "nll", "avg_nll", "api_top1_count",
    "api_top1_match", "api_pair_total", "api_pair_agree",
]


def write_scores(path: Path, averages: list[float], top1: int | list[int] = 9,
                 api_count: int | list[int] = 10) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for index, average in enumerate(averages):
            case_top1 = top1[index] if isinstance(top1, list) else top1
            case_count = api_count[index] if isinstance(api_count, list) else api_count
            writer.writerow({
                "id": f"case_{index:03d}", "target_tokens": 10,
                "nll": average * 10, "avg_nll": average,
                "api_top1_count": case_count,
                "api_top1_match": case_top1 if case_count else 0,
                "api_pair_total": case_count,
                "api_pair_agree": min(9, case_count) if case_count else 0,
            })
    path.with_suffix(".manifest").write_text(
        "model=/model.gguf\nmodel_size=1\nmodel_sample_sha256=" + "a" * 64 +
        "\nquality_input_sha256=" + "b" * 64 +
        "\nsource_commit=" + "0" * 40 + "\nsource_dirty=0\ndspark=0\n",
        encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        reference = root / "reference.tsv"
        candidate = root / "candidate.tsv"
        thresholds = root / "thresholds.json"
        write_scores(reference, [0.5, 0.6, 0.7])
        write_scores(candidate, [0.49, 0.59, 0.69])
        thresholds.write_text(json.dumps({
            "baseline_id": "sha256:" + "1" * 64,
            "min_cases": 3,
            "min_target_tokens": 30,
            "max_mean_nll_delta": 0.0,
            "max_ci95_high_nll_delta": 0.02,
            "min_api_top1_rate_delta": 0.0,
            "min_api_pair_rate_delta": 0.0,
        }), encoding="utf-8")
        passed = subprocess.run(
            [str(TOOL), str(reference), str(candidate), "--thresholds", str(thresholds)],
            text=True, capture_output=True, check=False)
        assert passed.returncode == 0, passed.stderr
        assert json.loads(passed.stdout)["passed"] is True

        write_scores(candidate, [0.48, 0.61, 0.68])
        noisy_better = subprocess.run(
            [str(TOOL), str(reference), str(candidate), "--thresholds", str(thresholds)],
            text=True, capture_output=True, check=False)
        assert noisy_better.returncode == 0, noisy_better.stderr
        assert json.loads(noisy_better.stdout)["metrics"]["paired_ci95_high"] > 0.0

        write_scores(candidate, [0.8, 0.9, 1.0])
        failed = subprocess.run(
            [str(TOOL), str(reference), str(candidate), "--thresholds", str(thresholds)],
            text=True, capture_output=True, check=False)
        assert failed.returncode != 0
        assert json.loads(failed.stdout)["passed"] is False

        write_scores(candidate, [0.49, 0.59, 0.69], top1=8)
        quality_drop = subprocess.run(
            [str(TOOL), str(reference), str(candidate), "--thresholds", str(thresholds)],
            text=True, capture_output=True, check=False)
        assert quality_drop.returncode != 0

        # A smaller/easier API denominator must not improve the apparent rate.
        write_scores(candidate, [0.49, 0.59, 0.69], top1=9, api_count=9)
        coverage_drop = subprocess.run(
            [str(TOOL), str(reference), str(candidate), "--thresholds", str(thresholds)],
            text=True, capture_output=True, check=False)
        assert coverage_drop.returncode != 0
        assert "API coverage differs" in coverage_drop.stderr

        write_scores(candidate, [0.49, 0.59, 0.69], top1=11)
        invalid_count = subprocess.run(
            [str(TOOL), str(reference), str(candidate), "--thresholds", str(thresholds)],
            text=True, capture_output=True, check=False)
        assert invalid_count.returncode != 0
        assert "invalid API agreement" in invalid_count.stderr

        for key in ("model_size", "model_sample_sha256", "quality_input_sha256"):
            write_scores(candidate, [0.49, 0.59, 0.69])
            manifest = candidate.with_suffix(".manifest")
            manifest.write_text(manifest.read_text() + f"{key}=mismatch\n")
            mismatch = subprocess.run(
                [str(TOOL), str(reference), str(candidate), "--thresholds", str(thresholds)],
                text=True, capture_output=True, check=False)
            assert mismatch.returncode != 0
            assert key in mismatch.stderr

        write_scores(reference, [0.5, 0.6, 0.7], api_count=0)
        write_scores(candidate, [0.49, 0.59, 0.69], api_count=0)
        no_api = subprocess.run(
            [str(TOOL), str(reference), str(candidate),
             "--thresholds", str(thresholds)],
            text=True, capture_output=True, check=False)
        assert no_api.returncode != 0
        no_api_result = json.loads(no_api.stdout)
        assert no_api_result["nll_screen_passed"] is True
        assert no_api_result["api_screen_passed"] is None
        assert no_api_result["passed"] is False
        assert no_api_result["blockers"]
        assert "paired NLL screen was still reported" in no_api.stderr

        v2_thresholds = root / "thresholds-v2.json"
        v2_thresholds.write_text(json.dumps({
            "schema_version": 2,
            "baseline_id": "sha256:" + "2" * 64,
            "min_cases": 20,
            "min_target_tokens": 200,
            "bootstrap": {
                "method": "bca",
                "resamples": 1999,
                "seed": 23,
                "nll_confidence_level": 0.95,
                "api_confidence_level": 0.99,
            },
            "nll": {"max_delta_upper": 0.05, "max_case_delta": 0.2},
            "api": {
                "required": True,
                "min_cases": 20,
                "min_top1_delta_lower": -0.02,
                "min_pair_delta_lower": -0.02,
            },
        }))
        base = [0.5 + 0.01 * (index % 5) for index in range(20)]
        write_scores(reference, base)
        write_scores(candidate, base)
        v2_exact = subprocess.run(
            [str(TOOL), str(reference), str(candidate),
             "--thresholds", str(v2_thresholds)],
            text=True, capture_output=True, check=False)
        assert v2_exact.returncode == 0, v2_exact.stderr
        v2_result = json.loads(v2_exact.stdout)
        assert v2_result["schema_version"] == 2
        assert v2_result["metrics"]["paired_bootstrap"][
            "weighted_nll_delta"]["one_sided_upper"] == 0.0

        # A large number of correlated alternatives in one degraded case does
        # not masquerade as a precise token-level estimate: whole cases are
        # resampled and the one-sided paired lower bound fails.
        large_counts = [100000] * 20
        reference_matches = [90000] * 20
        candidate_matches = list(reference_matches)
        candidate_matches[0] = 0
        write_scores(reference, base, top1=reference_matches,
                     api_count=large_counts)
        write_scores(candidate, base, top1=candidate_matches,
                     api_count=large_counts)
        clustered_drop = subprocess.run(
            [str(TOOL), str(reference), str(candidate),
             "--thresholds", str(v2_thresholds)],
            text=True, capture_output=True, check=False)
        assert clustered_drop.returncode != 0
        clustered_result = json.loads(clustered_drop.stdout)
        assert clustered_result["metrics"]["paired_bootstrap"][
            "api_top1_rate_delta"]["one_sided_lower"] < -0.02

        # A local loss cannot hide behind an acceptable corpus average.
        write_scores(reference, base)
        localized = list(base)
        localized[0] += 0.5
        write_scores(candidate, localized)
        local_loss = subprocess.run(
            [str(TOOL), str(reference), str(candidate),
             "--thresholds", str(v2_thresholds)],
            text=True, capture_output=True, check=False)
        assert local_loss.returncode != 0
        assert json.loads(local_loss.stdout)["metrics"][
            "max_case_avg_nll_delta"] > 0.2

        optional = json.loads(v2_thresholds.read_text())
        optional["api"]["required"] = False
        optional["api"]["min_cases"] = 0
        v2_thresholds.write_text(json.dumps(optional))
        write_scores(reference, base, api_count=0)
        write_scores(candidate, [value - 0.01 for value in base], api_count=0)
        optional_api = subprocess.run(
            [str(TOOL), str(reference), str(candidate),
             "--thresholds", str(v2_thresholds)],
            text=True, capture_output=True, check=False)
        assert optional_api.returncode == 0, optional_api.stderr
        assert json.loads(optional_api.stdout)["api_screen_passed"] is True
    print("test_compare_quality_scores: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
