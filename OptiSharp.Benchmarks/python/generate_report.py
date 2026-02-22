"""
Generate comparison report from benchmark JSON results.
Reads all results from .temp/runs/ and generates/updates benchmark-results/comparison_report.md
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class BenchmarkResult:
    def __init__(self, data: Dict):
        self.framework = data["framework"]
        self.sampler = data["sampler"]
        self.objective = data["objective"]
        self.n_params = data["n_params"]
        self.n_trials = data["n_trials"]
        self.pruner = data["pruner"]
        self.tier = data["tier"]
        self.best_value = data["best_value"]
        self.elapsed_ms = data["elapsed_ms"]
        self.trials_per_second = data["trials_per_second"]
        self.pruned_trials = data["pruned_trials"]
        self.pruning_rate = data["pruning_rate"]
        self.convergence = data["convergence"]
        self.seed = data["seed"]

    def key(self) -> Tuple:
        """Unique key for this configuration (both frameworks)."""
        return (self.sampler, self.objective, self.n_params, self.n_trials, self.pruner, self.tier)

    @staticmethod
    def key_from_parts(sampler, objective, n_params, n_trials, pruner, tier) -> Tuple:
        return (sampler, objective, n_params, n_trials, pruner, tier)


def load_results(input_dir: str) -> List[BenchmarkResult]:
    """Load all benchmark results from JSON files."""
    results = []
    input_path = Path(input_dir)

    if not input_path.exists():
        return results

    for json_file in sorted(input_path.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                results.append(BenchmarkResult(data))
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}", file=sys.stderr)

    return results


def group_by_config(results: List[BenchmarkResult]) -> Dict[Tuple, Dict[str, BenchmarkResult]]:
    """Group results by configuration, with frameworks as values."""
    grouped = defaultdict(dict)
    for result in results:
        key = result.key()
        grouped[key][result.framework] = result
    return grouped


def format_value(value: float, is_time=False) -> str:
    """Format a float value for display."""
    if value == float("inf"):
        return "N/A"
    if is_time:
        if value >= 60000:
            return f"{value/1000:.1f}s"
        else:
            return f"{value:.0f}ms"
    else:
        if abs(value) < 0.001:
            return f"{value:.2e}"
        elif abs(value) < 1:
            return f"{value:.4f}"
        else:
            return f"{value:.2f}"


def generate_summary_tables(grouped: Dict) -> str:
    """Generate summary tables comparing quality and throughput."""
    if not grouped:
        return ""

    lines = []
    lines.append("## Summary\n")

    # Best Value Quality (lower = better)
    lines.append("### Best Value Quality (lower = better)\n")
    lines.append("| Objective | Sampler | Params | Trials | Pruner | OptiSharp | Optuna | Winner |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")

    for (sampler, objective, n_params, n_trials, pruner, tier), frameworks in sorted(grouped.items()):
        cs = frameworks.get("optisharp")
        py = frameworks.get("optuna")

        if cs and py:
            cs_val = format_value(cs.best_value)
            py_val = format_value(py.best_value)
            winner = "OptiSharp" if (cs.best_value != float("inf") and py.best_value != float("inf") and cs.best_value < py.best_value) else ("Optuna" if (py.best_value != float("inf") and cs.best_value != float("inf") and py.best_value < cs.best_value) else "Tie")
            lines.append(f"| {objective} | {sampler} | {n_params} | {n_trials} | {pruner} | {cs_val} | {py_val} | {winner} |\n")

    lines.append("")

    # Throughput (trials/sec, higher = better)
    lines.append("### Throughput (trials/sec, higher = better)\n")
    lines.append("| Objective | Sampler | Params | Trials | OptiSharp | Optuna | Winner |\n")
    lines.append("|---|---|---|---|---|---|---|\n")

    for (sampler, objective, n_params, n_trials, pruner, tier), frameworks in sorted(grouped.items()):
        if pruner == "none":  # Only show throughput for non-pruned runs
            cs = frameworks.get("optisharp")
            py = frameworks.get("optuna")

            if cs and py:
                cs_tps = format_value(cs.trials_per_second)
                py_tps = format_value(py.trials_per_second)
                winner = "OptiSharp" if cs.trials_per_second > py.trials_per_second else ("Optuna" if py.trials_per_second > cs.trials_per_second else "Tie")
                lines.append(f"| {objective} | {sampler} | {n_params} | {n_trials} | {cs_tps} | {py_tps} | {winner} |\n")

    lines.append("")
    return "".join(lines)


def generate_per_objective_section(grouped: Dict) -> str:
    """Generate per-objective comparison sections."""
    if not grouped:
        return ""

    lines = []
    lines.append("## Per-Objective Results\n")

    objectives = sorted(set(k[1] for k in grouped.keys()))

    for obj in objectives:
        lines.append(f"### {obj.capitalize()}\n")

        # Filter configs for this objective
        obj_configs = {k: v for k, v in grouped.items() if k[1] == obj}

        if not obj_configs:
            continue

        lines.append("#### Convergence (best value at checkpoints)\n")
        lines.append("| Config | 20% | 40% | 60% | 80% | 100% |\n")
        lines.append("|---|---|---|---|---|---|\n")

        for (sampler, _, n_params, n_trials, pruner, tier), frameworks in sorted(obj_configs.items()):
            cs = frameworks.get("optisharp")
            py = frameworks.get("optuna")

            if cs:
                config_name = f"OptiSharp/{sampler}/{n_params}p/{n_trials}t/{pruner}"
                conv_str = " / ".join(format_value(v) for v in cs.convergence)
                lines.append(f"| {config_name} | {conv_str.replace(' / ', ' | ')} |\n")

            if py:
                config_name = f"Optuna/{sampler}/{n_params}p/{n_trials}t/{pruner}"
                conv_str = " / ".join(format_value(v) for v in py.convergence)
                lines.append(f"| {config_name} | {conv_str.replace(' / ', ' | ')} |\n")

        lines.append("")

    return "".join(lines)


def generate_pruning_section(grouped: Dict) -> str:
    """Generate pruning effectiveness section."""
    if not grouped:
        return ""

    lines = []

    # Filter for pruned runs
    pruned_configs = {k: v for k, v in grouped.items() if k[4] != "none"}

    if not pruned_configs:
        return ""

    lines.append("## Pruning Effectiveness\n")
    lines.append("| Pruner | Sampler | Objective | OptiSharp Rate | Optuna Rate | OptiSharp TPS | Optuna TPS |\n")
    lines.append("|---|---|---|---|---|---|---|\n")

    for (sampler, objective, n_params, n_trials, pruner, tier), frameworks in sorted(pruned_configs.items()):
        cs = frameworks.get("optisharp")
        py = frameworks.get("optuna")

        if cs and py:
            cs_rate = f"{cs.pruning_rate * 100:.1f}%"
            py_rate = f"{py.pruning_rate * 100:.1f}%"
            cs_tps = format_value(cs.trials_per_second)
            py_tps = format_value(py.trials_per_second)
            lines.append(f"| {pruner} | {sampler} | {objective} | {cs_rate} | {py_rate} | {cs_tps} | {py_tps} |\n")

    lines.append("")
    return "".join(lines)


def generate_report(input_dir: str) -> str:
    """Generate the full comparison report."""
    results = load_results(input_dir)
    grouped = group_by_config(results)

    lines = []

    # Header
    lines.append("# OptiSharp vs Optuna — Benchmark Comparison\n\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")

    # Count completed runs
    completed = sum(1 for frameworks in grouped.values() if len(frameworks) == 2)
    total_expected = len(grouped)
    lines.append(f"**Progress:** {completed}/{total_expected} configuration pairs completed\n\n")

    # Summary tables
    lines.append(generate_summary_tables(grouped))

    # Per-objective sections
    lines.append(generate_per_objective_section(grouped))

    # Pruning section
    lines.append(generate_pruning_section(grouped))

    # Footer
    lines.append("---\n\n")
    lines.append("*Lower is better for quality metrics (best value). Higher is better for throughput.*\n")

    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison report")
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing benchmark JSON results"
    )
    parser.add_argument(
        "--output", required=True, help="Output Markdown file path"
    )
    args = parser.parse_args()

    report = generate_report(args.input_dir)

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)

    print(f"[OK] Report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
