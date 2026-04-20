namespace OptiSharp.Samplers.Sobol;

/// <summary>
/// Configuration for the Sobol quasi-random sampler.
/// </summary>
public sealed record SobolSamplerConfig
{
    /// <summary>
    /// Random seed for Owen scrambling. Null = unscrambled deterministic sequence.
    /// </summary>
    public int? Seed { get; init; }

    /// <summary>
    /// Number of initial points to skip in the sequence.
    /// Skipping a power-of-2 number of points can improve uniformity.
    /// Default 0 (start from the beginning).
    /// </summary>
    public int Skip { get; init; } = 0;
}
