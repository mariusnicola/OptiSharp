namespace OptiSharp.Samplers.Tpe;

/// <summary>
/// Configuration for the TPE sampler.
/// </summary>
public sealed record TpeSamplerConfig
{
    /// <summary>
    /// Number of random trials before TPE kicks in.
    /// </summary>
    public int NStartupTrials { get; init; } = 10;

    /// <summary>
    /// Number of EI candidates to evaluate per parameter.
    /// </summary>
    public int NEiCandidates { get; init; } = 24;

    /// <summary>
    /// Weight of the uniform prior in the Parzen estimator.
    /// </summary>
    public double PriorWeight { get; init; } = 1.0;

    /// <summary>
    /// Include running trials in "above" group to avoid duplicate suggestions.
    /// </summary>
    public bool ConstantLiar { get; init; } = true;

    /// <summary>
    /// Enforce minimum bandwidth in Parzen estimator.
    /// </summary>
    public bool ConsiderMagicClip { get; init; } = true;

    /// <summary>
    /// Random seed for reproducibility. Null = non-deterministic.
    /// </summary>
    public int? Seed { get; init; }

    /// <summary>
    /// Maximum number of observations in the "above" (bad) group for KDE fitting.
    /// Caps LogPdf cost to O(MaxAboveTrials) instead of O(n_completed).
    /// 0 = no limit (all completed trials used). Default: 200.
    /// </summary>
    public int MaxAboveTrials { get; init; } = 200;
}
