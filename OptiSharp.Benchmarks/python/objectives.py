"""
Objective functions for OptiSharp vs Optuna comparison.
Identical math to C# version for fair comparison.
All functions minimize; range varies per function.
"""

import math
import numpy as np
from typing import Dict, Tuple, Callable


def sphere(params: Dict[str, float]) -> float:
    """
    Sphere function: Σ xᵢ²
    Global minimum: 0
    Range: [-5, 5]
    """
    values = list(params.values())
    return sum(x**2 for x in values)


def rosenbrock(params: Dict[str, float]) -> float:
    """
    Rosenbrock function: Σ [100(xᵢ₊₁ - xᵢ²)² + (1 - xᵢ)²]
    Global minimum: 0
    Range: [-2, 2]
    """
    values = list(params.values())
    total = 0.0
    for i in range(len(values) - 1):
        xi = values[i]
        xi1 = values[i + 1]
        term1 = 100 * (xi1 - xi**2)**2
        term2 = (1 - xi)**2
        total += term1 + term2
    return total


def rastrigin(params: Dict[str, float]) -> float:
    """
    Rastrigin function: 10n + Σ [xᵢ² - 10·cos(2π·xᵢ)]
    Global minimum: 0
    Range: [-5.12, 5.12]
    Highly multimodal - tests exploration.
    """
    values = list(params.values())
    n = len(values)
    total = 0.0
    for xi in values:
        total += xi**2 - 10 * math.cos(2 * math.pi * xi)
    return 10 * n + total


def ackley(params: Dict[str, float]) -> float:
    """
    Ackley function: -20·exp(-0.2·√(Σxᵢ²/n)) - exp(Σcos(2πxᵢ)/n) + 20 + e
    Global minimum: 0
    Range: [-32.768, 32.768]
    Smooth with local noise - tests balance.
    """
    values = np.array(list(params.values()))
    n = len(values)

    sum_sq = np.sum(values**2)
    sum_cos = np.sum(np.cos(2 * math.pi * values))

    term1 = -20 * math.exp(-0.2 * math.sqrt(sum_sq / n))
    term2 = -math.exp(sum_cos / n)
    e = math.e

    return term1 + term2 + 20 + e


def get_objective(name: str) -> Callable[[Dict[str, float]], float]:
    """Get objective function by name."""
    name_lower = name.lower()
    if name_lower == "sphere":
        return sphere
    elif name_lower == "rosenbrock":
        return rosenbrock
    elif name_lower == "rastrigin":
        return rastrigin
    elif name_lower == "ackley":
        return ackley
    else:
        raise ValueError(f"Unknown objective: {name}")


def get_range(objective_name: str) -> Tuple[float, float]:
    """Get the search range for an objective."""
    name_lower = objective_name.lower()
    if name_lower == "sphere":
        return (-5.0, 5.0)
    elif name_lower == "rosenbrock":
        return (-2.0, 2.0)
    elif name_lower == "rastrigin":
        return (-5.12, 5.12)
    elif name_lower == "ackley":
        return (-32.768, 32.768)
    else:
        raise ValueError(f"Unknown objective: {objective_name}")


def get_intermediate_value(
    true_value: float,
    step: int,  # 0..9
    trial_number: int,  # used to seed noise
    seed: int = 42
) -> float:
    """
    Simulate intermediate pruning values.
    At step k (0-9), report: objective(params) * (1 + (9-k) * 0.5 * noise)
    where noise is seeded for reproducibility.
    Step 9 (final) returns the exact value.
    """
    if step == 9:
        return true_value  # Final step: exact value

    # Seeded pseudo-random noise for reproducibility
    rng = np.random.RandomState(seed ^ trial_number ^ step)
    noise = rng.uniform(0, 1)  # [0, 1)
    factor = 1 + (9 - step) * 0.5 * noise
    return true_value * factor
