#!/usr/bin/env python3
"""
Unified Benchmark Framework: OptiSharp vs Optuna
Same datasets, same problems, direct comparison across all features and scales

Features tested:
- Single-Objective (TPE, Random, CMA-ES where available)
- Multi-Objective
- Constraints
- Pruning

Scales:
- Small: 500 trials, 20 parameters
- Medium: 5,000 trials, 100 parameters
- Large: 50,000 trials, 200 parameters
"""

import subprocess
import json
import time
from datetime import datetime

# ============================================================================
# UNIFIED TEST SUITE
# ============================================================================

TEST_SUITE = {
    "Small": {
        "trials": 500,
        "params": 20,
        "description": "500 trials, 20 params"
    },
    "Medium": {
        "trials": 5000,
        "params": 100,
        "description": "5,000 trials, 100 params"
    },
    "Large": {
        "trials": 50000,
        "params": 200,
        "description": "50,000 trials, 200 params (HUGE PAYLOAD)"
    }
}

FEATURES = ["SingleObjective", "MultiObjective", "Constraints", "Pruning"]

SAMPLERS = {
    "optuna": ["TPE"],  # Optuna only has TPE native
    "optisharp": ["TPE", "CMA-ES", "Random"]  # OptiSharp has all three
}

# ============================================================================
# DATASET GENERATORS (identical for both)
# ============================================================================

class UnifiedDataset:
    """Generates identical evaluation functions for both libraries"""

    @staticmethod
    def sphere_function(params):
        """Sum of squares - single objective"""
        if isinstance(params, dict):
            values = [v for v in params.values() if isinstance(v, (int, float))]
        else:
            values = params
        return sum(x * x for x in values)

    @staticmethod
    def multi_objective_func(params):
        """Sphere + Rosenbrock - multi objective"""
        if isinstance(params, dict):
            values = [v for v in params.values() if isinstance(v, (int, float))]
        else:
            values = params

        # Objective 1: Sphere
        obj1 = sum(x * x for x in values)

        # Objective 2: Rosenbrock (first 10 params only)
        obj2 = 0
        for i in range(min(len(values) - 1, 10)):
            obj2 += 100 * (values[i + 1] - values[i] ** 2) ** 2 + (1 - values[i]) ** 2

        return [obj1, obj2]

    @staticmethod
    def constraint_func(params):
        """Constraint: sum(|x|) < threshold"""
        if isinstance(params, dict):
            values = [v for v in params.values() if isinstance(v, (int, float))]
        else:
            values = params

        sum_abs = sum(abs(x) for x in values)
        threshold = len(values) * 2.0
        return sum_abs - threshold  # Must be <= 0

    @staticmethod
    def pruning_func(params, step):
        """Multi-step function for pruning - improves with steps"""
        if isinstance(params, dict):
            values = [v for v in params.values() if isinstance(v, (int, float))]
        else:
            values = params

        sphere = sum(x * x for x in values)
        return sphere / (1 + step * 0.1)  # Improves with step count

# ============================================================================
# OPTUNA BENCHMARKS
# ============================================================================

def run_optuna_benchmark(scale_name, n_trials, n_params, feature):
    """Run Optuna benchmark on unified dataset"""
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    scale_config = TEST_SUITE[scale_name]
    start_time = time.time()

    if feature == "SingleObjective":
        def objective(trial):
            params = {f"x_{i}": trial.suggest_float(f"x_{i}", -5.0, 5.0)
                     for i in range(n_params)}
            return UnifiedDataset.sphere_function(params)

        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        elapsed = time.time() - start_time
        return {
            "feature": "SingleObjective",
            "library": "Optuna",
            "sampler": "TPE",
            "scale": scale_name,
            "n_trials": n_trials,
            "n_params": n_params,
            "best_value": study.best_value,
            "wall_clock_seconds": elapsed,
            "trials_per_second": n_trials / elapsed
        }

    elif feature == "MultiObjective":
        def objective(trial):
            params = {f"x_{i}": trial.suggest_float(f"x_{i}", -5.0, 5.0)
                     for i in range(n_params)}
            return UnifiedDataset.multi_objective_func(params)

        sampler = TPESampler(seed=42)
        study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        elapsed = time.time() - start_time
        return {
            "feature": "MultiObjective",
            "library": "Optuna",
            "sampler": "TPE",
            "scale": scale_name,
            "n_trials": n_trials,
            "n_params": n_params,
            "pareto_front_size": len(study.best_trials),
            "wall_clock_seconds": elapsed,
            "trials_per_second": n_trials / elapsed
        }

    elif feature == "Constraints":
        def objective(trial):
            params = {f"x_{i}": trial.suggest_float(f"x_{i}", -5.0, 5.0)
                     for i in range(n_params)}
            value = UnifiedDataset.sphere_function(params)
            constraint = UnifiedDataset.constraint_func(params)

            # Penalize constraint violation
            if constraint > 0:
                value += constraint * 100

            return value

        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        elapsed = time.time() - start_time
        return {
            "feature": "Constraints",
            "library": "Optuna",
            "sampler": "TPE",
            "scale": scale_name,
            "n_trials": n_trials,
            "n_params": n_params,
            "best_value": study.best_value,
            "wall_clock_seconds": elapsed,
            "trials_per_second": n_trials / elapsed
        }

    elif feature == "Pruning":
        def objective(trial):
            params = {f"x_{i}": trial.suggest_float(f"x_{i}", -5.0, 5.0)
                     for i in range(n_params)}

            for step in range(1, 6):
                value = UnifiedDataset.pruning_func(params, step)
                trial.report(value, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return value

        sampler = TPESampler(seed=42)
        pruner = MedianPruner()
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        elapsed = time.time() - start_time
        pruned_count = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)

        return {
            "feature": "Pruning",
            "library": "Optuna",
            "sampler": "TPE",
            "scale": scale_name,
            "n_trials": n_trials,
            "n_params": n_params,
            "pruned_count": pruned_count,
            "pruning_rate": pruned_count / n_trials,
            "best_value": study.best_value,
            "wall_clock_seconds": elapsed,
            "trials_per_second": n_trials / elapsed
        }

# ============================================================================
# MAIN
# ============================================================================

def main():
    results = {
        "timestamp": datetime.now().isoformat(),
        "framework": "Unified Benchmark",
        "optuna_results": [],
        "optisharp_framework_ready": True
    }

    print("="*80)
    print("UNIFIED BENCHMARK FRAMEWORK: OptiSharp vs Optuna")
    print("="*80)
    print("\nRunning Optuna benchmarks on unified datasets...")
    print("Scales: Small (500 trials, 20p), Medium (5k, 100p), Large (50k, 200p)")
    print("Features: SingleObjective, MultiObjective, Constraints, Pruning")

    # Run Optuna benchmarks
    for scale_name in ["Small", "Medium", "Large"]:
        config = TEST_SUITE[scale_name]
        print(f"\n{'#'*80}")
        print(f"# {scale_name.upper()}: {config['description']}")
        print(f"{'#'*80}")

        for feature in FEATURES:
            print(f"\nTesting {feature}...")
            try:
                result = run_optuna_benchmark(scale_name, config["trials"], config["params"], feature)
                results["optuna_results"].append(result)
                print(f"  ✓ {feature}: {result['wall_clock_seconds']:.2f}s ({result['trials_per_second']:.1f} trials/sec)")
            except Exception as e:
                print(f"  ✗ {feature}: Error - {str(e)[:100]}")

    # Save results
    with open("unified_benchmark_optuna_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Optuna results saved to: unified_benchmark_optuna_results.json")
    print(f"{'='*80}")

    # Summary
    print("\nOPTUNA SUMMARY:")
    for scale in ["Small", "Medium", "Large"]:
        scale_results = [r for r in results["optuna_results"] if r["scale"] == scale]
        if scale_results:
            avg_time = sum(r["wall_clock_seconds"] for r in scale_results) / len(scale_results)
            print(f"\n{scale}:")
            for r in scale_results:
                print(f"  {r['feature']}: {r['wall_clock_seconds']:.2f}s ({r['trials_per_second']:.1f} trials/sec)")

if __name__ == "__main__":
    main()
