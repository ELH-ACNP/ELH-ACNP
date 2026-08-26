#!/usr/bin/env python3
"""Run the raw two-sided paired Wilcoxon tests reported in Appendix C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ALGORITHMS = [
    "ELH-ACNP",
    "LNS-CNP",
    "VNS-CNP",
    "VND-CNP",
    "H-CNP",
    "HPFS-CNP",
    "FCFS-CNP",
    "LLF-CNP",
]
BASELINES = ALGORITHMS[1:]
SCENARIO_LABELS = {
    "section_5_3": "Five DRSs and 1,500 tasks",
    "section_5_4": "Three DRSs and 1,500 tasks",
}
METRIC_LABELS = {
    "bcr": "BCR",
    "runtime_reported_s": "Running time",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section-5-3", type=Path, required=True)
    parser.add_argument("--section-5-4", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate(rows: list[dict[str, object]]) -> dict[str, int | float]:
    if any(row.get("status") != "ok" for row in rows):
        raise ValueError("The raw result files contain failed records")

    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = str(row["scenario"]), int(row["run"])
        groups.setdefault(key, []).append(row)

    if len(groups) != 40:
        raise ValueError(f"Expected 40 scenario-run pairs, found {len(groups)}")

    maximum_total_range = 0.0
    for key, group in groups.items():
        names = sorted(str(row["algorithm"]) for row in group)
        if names != sorted(ALGORITHMS):
            raise ValueError(f"Incomplete algorithm set for {key}: {names}")
        totals = np.asarray([float(row["total_benefit"]) for row in group])
        maximum_total_range = max(maximum_total_range, float(np.ptp(totals)))

    if maximum_total_range > 1e-10:
        raise ValueError("Algorithms within at least one pair used different task instances")

    return {
        "records": len(rows),
        "scenario_run_pairs": len(groups),
        "runs_per_scenario": 20,
        "algorithms_per_run": len(ALGORITHMS),
        "maximum_within_pair_total_benefit_range": maximum_total_range,
    }


def values(
    rows: list[dict[str, object]],
    scenario: str,
    algorithm: str,
    metric: str,
) -> np.ndarray:
    selected = [
        row
        for row in rows
        if row["scenario"] == scenario and row["algorithm"] == algorithm
    ]
    selected.sort(key=lambda row: int(row["run"]))
    if len(selected) != 20:
        raise ValueError(f"Expected 20 values for {scenario}/{algorithm}/{metric}")
    return np.asarray([float(row[metric]) for row in selected], dtype=float)


def descriptive_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for scenario, scenario_label in SCENARIO_LABELS.items():
        for algorithm in ALGORITHMS:
            bcr = values(rows, scenario, algorithm, "bcr")
            runtime = values(rows, scenario, algorithm, "runtime_reported_s")
            result.append({
                "scenario": scenario,
                "scenario_label": scenario_label,
                "algorithm": algorithm,
                "n": 20,
                "bcr_mean": float(np.mean(bcr)),
                "bcr_sd": float(np.std(bcr, ddof=1)),
                "runtime_mean_s": float(np.mean(runtime)),
                "runtime_sd_s": float(np.std(runtime, ddof=1)),
            })
    return result


def wilcoxon_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for scenario, scenario_label in SCENARIO_LABELS.items():
        for metric, metric_label in METRIC_LABELS.items():
            proposed = values(rows, scenario, "ELH-ACNP", metric)
            for baseline in BASELINES:
                comparison = values(rows, scenario, baseline, metric)
                difference = proposed - comparison
                test = wilcoxon(
                    proposed,
                    comparison,
                    alternative="two-sided",
                    zero_method="pratt",
                    method="auto",
                )
                result.append({
                    "scenario": scenario,
                    "scenario_label": scenario_label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "reference": "ELH-ACNP",
                    "baseline": baseline,
                    "n_pairs": 20,
                    "median_paired_difference": float(np.median(difference)),
                    "wilcoxon_w": float(test.statistic),
                    "p_value": float(test.pvalue),
                    "significant_at_0_05": bool(test.pvalue < 0.05),
                })
    return result


def format_p(value: float) -> str:
    return f"{value:.2e}" if value < 0.001 else f"{value:.4f}"


def format_p_latex(value: float) -> str:
    if value >= 0.001:
        return f"{value:.4f}"
    coefficient, exponent = f"{value:.2e}".split("e")
    return f"${coefficient} \\times 10^{{{int(exponent)}}}$"


def make_report(test_rows: list[dict[str, object]]) -> str:
    lines = [
        "# Paired Wilcoxon signed-rank tests",
        "",
        "Two-sided paired Wilcoxon signed-rank tests were applied, and the reported p-values are unadjusted.",
        "Each comparison contains 20 paired observations, and ELH-ACNP is the reference algorithm.",
        "",
    ]
    for scenario, label in SCENARIO_LABELS.items():
        lines.extend([
            f"## {label}",
            "",
            "| Baseline | BCR W | BCR p | Running-time W | Running-time p |",
            "|---|---:|---:|---:|---:|",
        ])
        lookup = {
            (str(row["metric"]), str(row["baseline"])): row
            for row in test_rows
            if row["scenario"] == scenario
        }
        for baseline in BASELINES:
            bcr = lookup[("bcr", baseline)]
            runtime = lookup[("runtime_reported_s", baseline)]
            lines.append(
                f"| {baseline} | {bcr['wilcoxon_w']:.0f} | {format_p(float(bcr['p_value']))} | "
                f"{runtime['wilcoxon_w']:.0f} | {format_p(float(runtime['p_value']))} |"
            )
        lines.append("")
    return "\n".join(lines)


def make_latex_table(test_rows: list[dict[str, object]]) -> str:
    lookup = {
        (str(row["scenario"]), str(row["metric"]), str(row["baseline"])): row
        for row in test_rows
    }

    def cells(scenario: str, baseline: str) -> str:
        bcr = lookup[(scenario, "bcr", baseline)]
        runtime = lookup[(scenario, "runtime_reported_s", baseline)]
        return (
            f"{bcr['wilcoxon_w']:.0f} & {format_p_latex(float(bcr['p_value']))} & "
            f"{runtime['wilcoxon_w']:.0f} & {format_p_latex(float(runtime['p_value']))}"
        )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Paired Wilcoxon signed-rank test results for the representative scenarios.}",
        r"\label{tab:paired_wilcoxon}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "& \\multicolumn{2}{c}{BCR} & \\multicolumn{2}{c}{Running time} \\\\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        "Baseline & $W$ & $p$ & $W$ & $p$ \\\\",
        r"\midrule",
    ]
    for scenario, label in SCENARIO_LABELS.items():
        lines.append(f"\\multicolumn{{5}}{{l}}{{{label}}} \\\\")
        for baseline in BASELINES:
            lines.append(f"{baseline} & {cells(scenario, baseline)} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.section_5_3) + load_jsonl(args.section_5_4)
    validation = validate(rows)
    descriptive = descriptive_results(rows)
    tests = wilcoxon_results(rows)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "validation": validation,
        "descriptive_results": descriptive,
        "wilcoxon_tests": tests,
    }
    (output_dir / "analysis_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "raw_results_combined.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "statistical_report.md").write_text(
        make_report(tests),
        encoding="utf-8",
    )
    (output_dir / "table_c1.tex").write_text(
        make_latex_table(tests),
        encoding="utf-8",
    )
    print(json.dumps({"validation": validation, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
