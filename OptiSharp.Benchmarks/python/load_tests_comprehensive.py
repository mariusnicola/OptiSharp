#!/usr/bin/env python3
"""
OptiSharp vs Optuna: Comprehensive Load Test Suite
Tests all features (single-obj, multi-obj, constraints, pruning) across 5 size scales.

Scales:
1. Tiny:    100 trials,  10 params
2. Small:   1k trials,   50 params
3. Medium:  5k trials,   100 params
4. Large:   25k trials,  150 params
5. XXL:     100k trials, 200 params

Run with:
    python load_tests_comprehensive.py [1-5] [feature]
    or: python load_tests_comprehensive.py all
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import json
import time
import numpy as np
from typing import Dict
import sys
from datetime import datetime

# ============================================================================
# TEST CONFIGURATIONS
# ============================================================================

TEST_SCALES = {
    1: {"name": "Tiny", "trials": 100, "params": 10},
    2: {"name": "Small", "trials": 1000, "params": 50},
    3: {"name": "Medium", "trials": 5000, "params": 100},
    4: {"name": "Large", "trials": 25000, "params": 150},
    5: {"name": "XXL", "trials": 100000, "params": 200},
}

# ============================================================================
# SINGLE-OBJECTIVE TESTS
# ============================================================================

def test_single_objective(scale_level: int, seed: int = 42) -> Dict:
    """Test single-objective optimization at given scale."""
    config = TEST_SCALES[scale_level]
    n_trials = config["trials"]
    n_params = config["params"]

    print(f"\n{'='*70}")
    print(f"SINGLE-OBJECTIVE: {config['name']} ({n_trials} trials, {n_params} params)")
    print(f"{'='*70}")

    def objective(trial):
        # High-dimensional sphere function
        params = [trial.suggest_float(f"x_{i}", -5.0, 5.0) for i in range(n_params)]
        value = sum(x * x for x in params)
        return value

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"single_obj_{config['name']}"
    )

    start_time = time.time()
    for trial_num in range(n_trials):
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        if (trial_num + 1) % max(1, n_trials // 4) == 0:
            elapsed = time.time() - start_time
            print(f"  {trial_num + 1}/{n_trials} trials: {elapsed:.1f}s, Best={study.best_value:.6f}")

    elapsed = time.time() - start_time

    return {
        "feature": "SingleObjective",
        "scale": config["name"],
        "n_trials": n_trials,
        "n_params": n_params,
        "best_value": study.best_value,
        "wall_clock_seconds": elapsed,
        "trials_per_second": n_trials / elapsed if elapsed > 0 else 0
    }

# ============================================================================
# MULTI-OBJECTIVE TESTS
# ============================================================================

def test_multi_objective(scale_level: int, seed: int = 42) -> Dict:
    """Test multi-objective optimization at given scale."""
    config = TEST_SCALES[scale_level]
    n_trials = config["trials"]
    n_params = config["params"]

    print(f"\n{'='*70}")
    print(f"MULTI-OBJECTIVE: {config['name']} ({n_trials} trials, {n_params} params)")
    print(f"{'='*70}")

    def objective(trial):
        # Multi-objective: minimize sphere & Rosenbrock
        params = [trial.suggest_float(f"x_{i}", -5.0, 5.0) for i in range(n_params)]

        # Objective 1: sphere
        obj1 = sum(x * x for x in params)

        # Objective 2: Rosenbrock-like
        obj2 = sum(100 * (params[i+1] - params[i]**2)**2 + (1 - params[i])**2
                   for i in range(min(n_params - 1, 10)))

        return [obj1, obj2]

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=sampler,
        study_name=f"multi_obj_{config['name']}"
    )

    start_time = time.time()
    for trial_num in range(n_trials):
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        if (trial_num + 1) % max(1, n_trials // 4) == 0:
            elapsed = time.time() - start_time
            pareto_size = len(study.best_trials)
            print(f"  {trial_num + 1}/{n_trials} trials: {elapsed:.1f}s, Pareto={pareto_size}")

    elapsed = time.time() - start_time

    return {
        "feature": "MultiObjective",
        "scale": config["name"],
        "n_trials": n_trials,
        "n_params": n_params,
        "pareto_front_size": len(study.best_trials),
        "wall_clock_seconds": elapsed,
        "trials_per_second": n_trials / elapsed if elapsed > 0 else 0
    }

# ============================================================================
# CONSTRAINT TESTS
# ============================================================================

def test_constraints(scale_level: int, seed: int = 42) -> Dict:
    """Test optimization with constraints at given scale."""
    config = TEST_SCALES[scale_level]
    n_trials = config["trials"]
    n_params = config["params"]

    print(f"\n{'='*70}")
    print(f"CONSTRAINTS: {config['name']} ({n_trials} trials, {n_params} params)")
    print(f"{'='*70}")

    def objective(trial):
        params = [trial.suggest_float(f"x_{i}", -5.0, 5.0) for i in range(n_params)]

        # Objective
        value = sum(x * x for x in params)

        # Constraint: sum must be < threshold
        constraint_value = sum(abs(x) for x in params) - (n_params * 2.0)

        # Return tuple (objective, constraint)
        if constraint_value <= 0:
            return value
        else:
            # Penalize constraint violation
            return value + constraint_value * 100

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"constraints_{config['name']}"
    )

    start_time = time.time()
    feasible_count = 0

    for trial_num in range(n_trials):
        study.optimize(objective, n_trials=1, show_progress_bar=False)

        # Check feasibility (simplified)
        best_trial = study.best_trial
        if best_trial and best_trial.value < 1000:  # Heuristic for feasibility
            feasible_count += 1

        if (trial_num + 1) % max(1, n_trials // 4) == 0:
            elapsed = time.time() - start_time
            feas_rate = (feasible_count / (trial_num + 1)) * 100
            print(f"  {trial_num + 1}/{n_trials} trials: {elapsed:.1f}s, Feasible={feas_rate:.1f}%")

    elapsed = time.time() - start_time

    return {
        "feature": "Constraints",
        "scale": config["name"],
        "n_trials": n_trials,
        "n_params": n_params,
        "best_value": study.best_value,
        "feasible_rate": (feasible_count / n_trials) * 100,
        "wall_clock_seconds": elapsed,
        "trials_per_second": n_trials / elapsed if elapsed > 0 else 0
    }

# ============================================================================
# PRUNING TESTS
# ============================================================================

def test_pruning(scale_level: int, seed: int = 42) -> Dict:
    """Test pruning effectiveness at given scale."""
    config = TEST_SCALES[scale_level]
    n_trials = config["trials"]
    n_params = config["params"]

    print(f"\n{'='*70}")
    print(f"PRUNING: {config['name']} ({n_trials} trials, {n_params} params)")
    print(f"{'='*70}")

    def objective(trial):
        params = [trial.suggest_float(f"x_{i}", -5.0, 5.0) for i in range(n_params)]

        # Multi-step evaluation with pruning checkpoints
        for step in range(1, 11):
            value = sum(x * x for x in params) / (1 + step * 0.1)
            trial.report(value, step)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return value

    pruner = MedianPruner()
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"pruning_{config['name']}"
    )

    start_time = time.time()
    pruned_count = 0

    for trial_num in range(n_trials):
        try:
            study.optimize(objective, n_trials=1, show_progress_bar=False)
        except:
            pass

        # Count pruned trials
        if len(study.trials) > 0:
            last_trial = study.trials[-1]
            if last_trial.state == optuna.trial.TrialState.PRUNED:
                pruned_count += 1

        if (trial_num + 1) % max(1, n_trials // 4) == 0:
            elapsed = time.time() - start_time
            pruning_rate = (pruned_count / (trial_num + 1)) * 100
            print(f"  {trial_num + 1}/{n_trials} trials: {elapsed:.1f}s, Pruned={pruning_rate:.1f}%")

    elapsed = time.time() - start_time

    return {
        "feature": "Pruning",
        "scale": config["name"],
        "n_trials": n_trials,
        "n_params": n_params,
        "pruned_count": pruned_count,
        "pruning_rate": (pruned_count / n_trials) * 100,
        "best_value": study.best_value if study.best_value else 0,
        "wall_clock_seconds": elapsed,
        "trials_per_second": n_trials / elapsed if elapsed > 0 else 0
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("OPTUNA COMPREHENSIVE LOAD TEST SUITE")
    print("="*70)
    print("Scales: Tiny (100 trials, 10p), Small (1k, 50p), Medium (5k, 100p),")
    print("        Large (25k, 150p), XXL (100k, 200p)")
    print("Features: SingleObjective, MultiObjective, Constraints, Pruning")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "library": "Optuna",
        "results": []
    }

    # Parse arguments
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        scales = [1, 2, 3, 4, 5]
        features = ["single", "multi", "constraints", "pruning"]
    else:
        scales = [1, 2, 3]  # Default: Tiny, Small, Medium only
        features = ["single", "multi", "constraints", "pruning"]

    for scale_level in scales:
        config = TEST_SCALES[scale_level]
        print(f"\n\n{'#'*70}")
        print(f"# SCALE: {config['name'].upper()} ({config['trials']} trials, {config['params']} params)")
        print(f"{'#'*70}")

        try:
            if "single" in features:
                result = test_single_objective(scale_level)
                all_results["results"].append(result)
        except Exception as e:
            print(f"  ERROR in SingleObjective: {e}")

        try:
            if "multi" in features:
                result = test_multi_objective(scale_level)
                all_results["results"].append(result)
        except Exception as e:
            print(f"  ERROR in MultiObjective: {e}")

        try:
            if "constraints" in features:
                result = test_constraints(scale_level)
                all_results["results"].append(result)
        except Exception as e:
            print(f"  ERROR in Constraints: {e}")

        try:
            if "pruning" in features:
                result = test_pruning(scale_level)
                all_results["results"].append(result)
        except Exception as e:
            print(f"  ERROR in Pruning: {e}")

    # Save results
    output_file = "optuna_load_tests.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*70}")
    print(f"Results saved to {output_file}")
    print(f"{'='*70}")

    # Print summary
    print("\nSUMMARY:")
    for result in all_results["results"]:
        print(f"\n{result['feature']} - {result['scale']}:")
        print(f"  Trials: {result['n_trials']}")
        print(f"  Time: {result['wall_clock_seconds']:.2f}s")
        print(f"  Throughput: {result['trials_per_second']:.1f} trials/sec")

if __name__ == "__main__":
    main()
