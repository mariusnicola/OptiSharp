"""
Optuna benchmark runner for comparison with OptiSharp.
Same CLI interface and output format as C# MatrixBenchmarkRunner.
"""

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List

import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler

from objectives import (
    get_objective,
    get_range,
    get_intermediate_value,
)

# Suppress Optuna logging
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class BenchmarkResult:
    def __init__(
        self,
        sampler: str,
        objective: str,
        n_params: int,
        n_trials: int,
        pruner: str,
        tier: str,
        best_value: float,
        elapsed_ms: float,
        trials_per_second: float,
        pruned_trials: int,
        pruning_rate: float,
        convergence: List[float],
        seed: int = 42,
    ):
        self.framework = "optuna"
        self.sampler = sampler
        self.objective = objective
        self.n_params = n_params
        self.n_trials = n_trials
        self.pruner = pruner
        self.tier = tier
        self.best_value = best_value
        self.elapsed_ms = elapsed_ms
        self.trials_per_second = trials_per_second
        self.pruned_trials = pruned_trials
        self.pruning_rate = pruning_rate
        self.convergence = convergence
        self.seed = seed

    def to_dict(self) -> Dict:
        return {
            "framework": self.framework,
            "sampler": self.sampler,
            "objective": self.objective,
            "n_params": self.n_params,
            "n_trials": self.n_trials,
            "pruner": self.pruner,
            "tier": self.tier,
            "best_value": self.best_value,
            "elapsed_ms": self.elapsed_ms,
            "trials_per_second": self.trials_per_second,
            "pruned_trials": self.pruned_trials,
            "pruning_rate": self.pruning_rate,
            "convergence": self.convergence,
            "seed": self.seed,
        }

    def save(self, path: str):
        """Save result to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def run_benchmark(
    sampler_name: str,
    n_params: int,
    n_trials: int,
    objective_name: str,
    pruner_name: str,
    tier: str,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""

    # Get objective function
    objective_fn = get_objective(objective_name)
    low, high = get_range(objective_name)

    # Create search space
    sampler_obj = None
    if sampler_name == "tpe":
        sampler_obj = TPESampler(seed=42)
    elif sampler_name == "random":
        sampler_obj = RandomSampler(seed=42)
    elif sampler_name == "cmaes":
        sampler_obj = CmaEsSampler(seed=42)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    # Create pruner
    pruner_obj = None
    if pruner_name == "none":
        pruner_obj = optuna.pruners.NopPruner()
    elif pruner_name == "median":
        pruner_obj = MedianPruner()
    elif pruner_name == "sha":
        pruner_obj = SuccessiveHalvingPruner()
    else:
        raise ValueError(f"Unknown pruner: {pruner_name}")

    # Create study
    study = optuna.create_study(
        sampler=sampler_obj,
        pruner=pruner_obj,
        direction="minimize",
    )

    # Track convergence at checkpoints
    convergence = []
    checkpoints = [0.2, 0.4, 0.6, 0.8, 1.0]
    checkpoint_idx = 0
    pruned_count = 0

    # Run optimization
    start_time = time.time()

    def objective(trial: optuna.trial.Trial) -> float:
        nonlocal checkpoint_idx, pruned_count

        # Suggest parameters
        params = {f"x{i}": trial.suggest_float(f"x{i}", low, high) for i in range(n_params)}

        # Get true value
        true_value = objective_fn(params)

        # Report intermediate values if pruning is enabled
        if pruner_name != "none":
            for step in range(10):
                intermediate_value = get_intermediate_value(
                    true_value, step, trial.number, seed=42
                )
                trial.report(intermediate_value, step)

                # Check if should prune
                if trial.should_prune():
                    pruned_count += 1
                    raise optuna.TrialPruned()

        # Record convergence at checkpoints
        if checkpoint_idx < len(checkpoints):
            checkpoint_trial = int(n_trials * checkpoints[checkpoint_idx])
            if trial.number >= checkpoint_trial - 1:
                convergence.append(study.best_value)
                checkpoint_idx += 1

        return true_value

    # Run trials
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    except Exception:
        pass  # Continue even if some errors occur

    elapsed_time = time.time() - start_time
    elapsed_ms = int(elapsed_time * 1000)

    # Ensure we have exactly 5 convergence points
    while len(convergence) < 5:
        convergence.append(study.best_value)
    convergence = convergence[:5]

    result = BenchmarkResult(
        sampler=sampler_name,
        objective=objective_name,
        n_params=n_params,
        n_trials=n_trials,
        pruner=pruner_name,
        tier=tier,
        best_value=study.best_value,
        elapsed_ms=elapsed_ms,
        trials_per_second=n_trials / elapsed_time if elapsed_time > 0 else 0,
        pruned_trials=pruned_count,
        pruning_rate=pruned_count / n_trials if n_trials > 0 else 0,
        convergence=convergence,
    )

    return result


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optuna matrix benchmark runner for OptiSharp comparison"
    )
    parser.add_argument(
        "--sampler", required=True, choices=["tpe", "random", "cmaes"], help="Sampler to use"
    )
    parser.add_argument("--params", type=int, required=True, help="Number of parameters")
    parser.add_argument("--trials", type=int, required=True, help="Number of trials")
    parser.add_argument(
        "--objective",
        required=True,
        choices=["sphere", "rosenbrock", "rastrigin", "ackley"],
        help="Objective function",
    )
    parser.add_argument(
        "--pruner", required=True, choices=["none", "median", "sha"], help="Pruner to use"
    )
    parser.add_argument("--tier", required=True, choices=["fast", "extended"], help="Tier")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    try:
        result = run_benchmark(
            sampler_name=args.sampler,
            n_params=args.params,
            n_trials=args.trials,
            objective_name=args.objective,
            pruner_name=args.pruner,
            tier=args.tier,
        )
        result.save(args.output)
        # Use ASCII-safe output for Windows console compatibility
        print(f"[OK] Result saved to {args.output}")
        return 0
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
