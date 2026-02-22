#!/usr/bin/env python3
"""
OptiSharp vs Optuna Empirical Benchmark Suite

This script runs identical benchmark problems using Optuna (Python) to compare
against OptiSharp (C#) on a level playing field.

Run with:
    python3 benchmarks_optuna.py

Requirements:
    pip install optuna scikit-learn pandas numpy
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import json
import time
import numpy as np
from typing import Dict, List, Tuple
import sys

# ============================================================================
# BENCHMARK 1: CASH (Algorithm Selection + Hyperparameter Optimization)
# ============================================================================

def benchmark_cash_optuna(n_trials: int = 100, seed: int = 42) -> Dict:
    """
    Benchmark 1: CASH problem - simulate classifier selection + tuning
    """
    print("\n" + "="*70)
    print("BENCHMARK 1: CASH (Algorithm Selection + HPO)")
    print("="*70)

    def objective(trial):
        classifier = trial.suggest_categorical('classifier', ['rf', 'svm', 'gb'])
        max_depth = trial.suggest_int('max_depth', 3, 30)
        n_estimators = trial.suggest_int('n_estimators', 10, 500)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 1.0, log=True)
        regularization = trial.suggest_float('regularization', 0.0001, 10.0, log=True)

        # Simulate accuracy based on hyperparameters
        base_accuracy = {
            'rf': 0.92,
            'svm': 0.90,
            'gb': 0.91
        }[classifier]

        depth_boost = min(0.05, (max_depth - 3) / 27.0 * 0.05)
        lr_penalty = abs(np.log10(learning_rate) + 2.5) * 0.02

        accuracy = min(1.0, base_accuracy + depth_boost - lr_penalty + np.random.random() * 0.03)

        return accuracy

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='cash_optuna'
    )

    start_time = time.time()
    convergence_history = {}

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    elapsed = time.time() - start_time
    best_value = study.best_value

    # Find trials to target 95%
    trials_to_95 = None
    for trial in study.trials:
        best_so_far = max(t.value for t in study.trials[:study.trials.index(trial)+1] if t.value is not None)
        if best_so_far >= 0.95 and trials_to_95 is None:
            trials_to_95 = trial.number + 1

    return {
        'benchmark': 'CASH',
        'library': 'Optuna',
        'n_trials': n_trials,
        'best_value': best_value,
        'wall_clock_seconds': elapsed,
        'trials_to_target_95': trials_to_95 if trials_to_95 else n_trials,
        'seed': seed,
        'sampler': 'TPE'
    }

# ============================================================================
# BENCHMARK 2: Neural Network Tuning
# ============================================================================

def benchmark_nn_optuna(n_trials: int = 50, seed: int = 100) -> Dict:
    """
    Benchmark 2: Neural Network hyperparameter tuning
    """
    print("\n" + "="*70)
    print("BENCHMARK 2: Neural Network Tuning")
    print("="*70)

    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
        batch_size = trial.suggest_int('batch_size', 16, 256)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        hidden_units = trial.suggest_int('hidden_units', 32, 512)
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])

        base_accuracy = 0.75
        lr_optimal = 0.001
        lr_penalty = abs(np.log10(learning_rate) - np.log10(lr_optimal)) * 0.15
        batch_boost = 0.02 * np.log(batch_size / 16.0)
        dropout_boost = dropout * 0.05

        accuracy = min(1.0, base_accuracy + batch_boost + dropout_boost - lr_penalty + np.random.random() * 0.02)

        return accuracy

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='nn_optuna'
    )

    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - start_time
    best_value = study.best_value

    return {
        'benchmark': 'NN_Tuning',
        'library': 'Optuna',
        'n_trials': n_trials,
        'best_value': best_value,
        'wall_clock_seconds': elapsed,
        'seed': seed,
        'sampler': 'TPE'
    }

# ============================================================================
# BENCHMARK 3: Multi-Objective with Constraints
# ============================================================================

def benchmark_mo_constraints_optuna(n_trials: int = 75, seed: int = 200) -> Dict:
    """
    Benchmark 3: Multi-objective (accuracy vs latency) with constraints
    """
    print("\n" + "="*70)
    print("BENCHMARK 3: Multi-Objective with Constraints")
    print("="*70)

    def objective(trial):
        batch_size = trial.suggest_int('batch_size', 16, 128)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
        model_size = trial.suggest_int('model_size', 10, 100)
        regularization = trial.suggest_float('regularization', 0.0, 0.5)

        # Constraints
        estimated_memory = 200 + batch_size * 2
        estimated_model_size = 5 + batch_size / 50.0

        if estimated_memory > 500 or estimated_model_size > 10:
            # Penalize constraint violations
            return [0.5, 200]  # Low accuracy, high latency

        # Objectives
        base_acc = 0.80
        lr_boost = max(0, 0.1 - abs(np.log10(learning_rate) + 2.5) * 0.3)
        batch_boost = np.log(batch_size / 16.0) * 0.02
        accuracy = min(1.0, base_acc + lr_boost + batch_boost + np.random.random() * 0.05)

        latency = 10 + np.log(batch_size) * 5 + np.sqrt(model_size) * 0.5 + np.random.random() * 5

        return [accuracy, latency]

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        directions=['maximize', 'minimize'],
        sampler=sampler,
        study_name='mo_constraints_optuna'
    )

    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - start_time

    # Count feasible solutions (simplified: those with accuracy > 0.75)
    feasible = sum(1 for trial in study.trials if trial.values and trial.values[0] > 0.75)

    return {
        'benchmark': 'MO_Constraints',
        'library': 'Optuna',
        'n_trials': n_trials,
        'pareto_front_size': len(study.best_trials),
        'feasible_solutions': feasible,
        'wall_clock_seconds': elapsed,
        'seed': seed,
        'sampler': 'TPE'
    }

# ============================================================================
# BENCHMARK 4: Pruning Effectiveness
# ============================================================================

def benchmark_pruning_optuna(n_trials: int = 100, seed: int = 300) -> Dict:
    """
    Benchmark 4: Test pruning effectiveness with different pruners
    """
    print("\n" + "="*70)
    print("BENCHMARK 4: Pruning Effectiveness")
    print("="*70)

    results = {}

    for pruner_type in ['median', 'none']:
        if pruner_type == 'median':
            pruner = MedianPruner()
        else:
            pruner = optuna.pruners.NopPruner()

        def objective(trial):
            x = trial.suggest_float('x', -5.0, 5.0)
            y = trial.suggest_float('y', -5.0, 5.0)

            # Multi-step evaluation with pruning checkpoints
            for step in range(1, 11):
                # Rastrigin-like function
                value = 20 + x*x + y*y - 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))
                value = value / (1 + step * 0.1)  # Improve with steps

                trial.report(value, step)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            return value

        sampler = TPESampler(seed=seed)
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            study_name=f'pruning_{pruner_type}'
        )

        start_time = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        elapsed = time.time() - start_time

        pruned_count = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)

        results[pruner_type] = {
            'pruner': pruner_type,
            'pruned_count': pruned_count,
            'pruning_rate': pruned_count / n_trials,
            'best_value': study.best_value,
            'wall_clock_seconds': elapsed
        }

    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("=" + " "*68 + "=")
    print("=" + "    OPTUNA EMPIRICAL BENCHMARK SUITE - REAL METRICS".center(68) + "=")
    print("=" + " "*68 + "=")
    print("="*70)

    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'library': 'Optuna',
        'benchmarks': []
    }

    # Run all benchmarks
    try:
        # Benchmark 1: CASH (run 3 times)
        for run in range(3):
            print(f"\nRun {run + 1}/3...")
            result = benchmark_cash_optuna(n_trials=100, seed=42 + run)
            all_results['benchmarks'].append(result)
            print(f"  * Best accuracy: {result['best_value']:.4f}")
            print(f"  * Wall-clock: {result['wall_clock_seconds']:.2f}s")
            print(f"  * Trials to 95%: {result['trials_to_target_95']}")

        # Benchmark 2: NN Tuning (run 3 times)
        for run in range(3):
            print(f"\nRun {run + 1}/3...")
            result = benchmark_nn_optuna(n_trials=50, seed=100 + run)
            all_results['benchmarks'].append(result)
            print(f"  * Best accuracy: {result['best_value']:.4f}")
            print(f"  * Wall-clock: {result['wall_clock_seconds']:.2f}s")

        # Benchmark 3: Multi-Objective (run 2 times)
        for run in range(2):
            print(f"\nRun {run + 1}/2...")
            result = benchmark_mo_constraints_optuna(n_trials=75, seed=200 + run)
            all_results['benchmarks'].append(result)
            print(f"  * Pareto front: {result['pareto_front_size']}")
            print(f"  * Feasible: {result['feasible_solutions']}/{result['n_trials']}")
            print(f"  * Wall-clock: {result['wall_clock_seconds']:.2f}s")

        # Benchmark 4: Pruning
        print("\nBenchmarking pruning strategies...")
        pruning_results = benchmark_pruning_optuna(n_trials=100, seed=300)
        for pruner_type, result in pruning_results.items():
            all_results['benchmarks'].append({
                'benchmark': 'Pruning',
                'library': 'Optuna',
                **result
            })
            print(f"  * {pruner_type}: {result['pruned_count']} pruned, {result['wall_clock_seconds']:.2f}s")

    except Exception as e:
        print(f"\nERROR during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save results
    output_file = 'optuna_benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print(f"Results saved to {output_file}")
    print("="*70)

    # Print summary
    cash_results = [r for r in all_results['benchmarks'] if r.get('benchmark') == 'CASH']
    if cash_results:
        avg_accuracy = np.mean([r['best_value'] for r in cash_results])
        avg_time = np.mean([r['wall_clock_seconds'] for r in cash_results])
        print(f"\nCASH Summary (3 runs):")
        print(f"  Average accuracy: {avg_accuracy:.4f}")
        print(f"  Average wall-clock: {avg_time:.2f}s")

# ============================================================================
# LOAD TEST 1: High-Dimensional Optimization (100 parameters)
# ============================================================================

def benchmark_high_dimensional_optuna(n_trials: int = 1000, n_params: int = 100, seed: int = 500) -> Dict:
    """
    Load Test 1: High-dimensional optimization with 100 continuous parameters
    """
    print("\n" + "="*70)
    print("LOAD TEST 1: High-Dimensional Optimization (100 parameters)")
    print("="*70)

    def objective(trial):
        # 100 continuous parameters
        params = [trial.suggest_float(f'x_{i}', -5.0, 5.0) for i in range(n_params)]

        # Sphere function (sum of squares)
        value = sum(x*x for x in params)
        return value

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name=f'high_dim_{n_params}_{n_trials}'
    )

    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - start_time

    return {
        'benchmark': 'HighDimensional',
        'library': 'Optuna',
        'n_trials': n_trials,
        'n_parameters': n_params,
        'best_value': study.best_value,
        'wall_clock_seconds': elapsed,
        'seed': seed,
        'sampler': 'TPE',
        'trials_per_second': n_trials / elapsed if elapsed > 0 else 0
    }

# ============================================================================
# LOAD TEST 2: Scale Test - Progressive Trial Counts
# ============================================================================

def benchmark_scale_progression_optuna(seed: int = 600) -> Dict:
    """
    Load Test 2: Measure performance at different trial scales
    """
    print("\n" + "="*70)
    print("LOAD TEST 2: Scale Progression (1k, 5k, 10k, 25k trials)")
    print("="*70)

    trial_counts = [1000, 5000, 10000, 25000]
    results = {}

    for n_trials in trial_counts:
        print(f"\nTesting {n_trials} trials...")

        def objective(trial):
            # 20 parameters
            x = trial.suggest_float('x', -10.0, 10.0)
            y = trial.suggest_float('y', -10.0, 10.0)
            z = trial.suggest_float('z', -10.0, 10.0)

            # Ackley function approximation
            value = (x*x + y*y + z*z) / 3.0 + np.sin(x) + np.cos(y) + np.sin(z)
            return value

        sampler = TPESampler(seed=seed + n_trials)
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            study_name=f'scale_{n_trials}'
        )

        start_time = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        elapsed = time.time() - start_time

        results[str(n_trials)] = {
            'n_trials': n_trials,
            'wall_clock_seconds': elapsed,
            'best_value': study.best_value,
            'trials_per_second': n_trials / elapsed if elapsed > 0 else 0
        }

        print(f"  Completed {n_trials} trials in {elapsed:.2f}s ({n_trials/elapsed:.1f} trials/sec)")

    return {
        'benchmark': 'ScaleProgression',
        'library': 'Optuna',
        'results': results
    }

if __name__ == '__main__':
    main()
