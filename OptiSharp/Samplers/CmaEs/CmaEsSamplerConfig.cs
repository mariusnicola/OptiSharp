using OptiSharp.Models;

namespace OptiSharp.Samplers.CmaEs;

/// <summary>
/// Configuration for the CMA-ES sampler.
/// </summary>
public sealed record CmaEsSamplerConfig
{
    /// <summary>
    /// Population size (lambda). Null = auto: 4 + floor(3 * ln(n)).
    /// </summary>
    public int? PopulationSize { get; init; }

    /// <summary>
    /// Initial step size as fraction of parameter range. Default 0.1 (matches Optuna).
    /// </summary>
    public double InitialSigma { get; init; } = 0.1;

    /// <summary>
    /// Random seed for reproducibility. Null = non-deterministic.
    /// </summary>
    public int? Seed { get; init; }

    /// <summary>
    /// Compute backend. Default Cpu. Use Gpu for CUDA acceleration on large
    /// search spaces (N >= 100 dimensions recommended). Auto detects GPU and
    /// falls back to CPU if unavailable.
    /// </summary>
    public ComputeBackend Backend { get; init; } = ComputeBackend.Cpu;

    /// <summary>
    /// Whether to restart CMA-ES when stagnation is detected (IPOP strategy).
    /// Default true — gives OptiSharp a genuine algorithmic advantage over Optuna
    /// which does not support automatic restarts.
    /// </summary>
    public bool RestartOnStagnation { get; init; } = true;

    /// <summary>
    /// Population size multiplier on each restart. Default 2 (standard IPOP).
    /// Each restart uses IncPopSize times the previous population size.
    /// </summary>
    public int IncPopSize { get; init; } = 2;

    /// <summary>
    /// Maximum number of restarts. Default 9 (caps lambda at 512x original).
    /// </summary>
    public int MaxRestarts { get; init; } = 9;
}
