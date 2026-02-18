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
    /// Initial step size as fraction of parameter range. Default 0.3.
    /// </summary>
    public double InitialSigma { get; init; } = 0.3;

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
}
