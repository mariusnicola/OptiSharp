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
    /// Default false — matches Optuna's default (constant_liar=False).
    /// Enable for parallel/async optimization to reduce duplicate suggestions.
    /// </summary>
    public bool ConstantLiar { get; init; } = false;

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
    /// 0 = auto: max(25, n_below). Matches Optuna behavior.
    /// </summary>
    public int MaxAboveTrials { get; init; } = 0;

    /// <summary>
    /// Use Sobol quasi-random sequences for startup trials instead of uniform random.
    /// Sobol sequences have better space-filling properties (lower discrepancy),
    /// giving the KDE better initial coverage for faster convergence.
    /// </summary>
    public bool SobolStartup { get; init; } = false;

    /// <summary>
    /// DEPRECATED: Gamma formula is now hardcoded to match Optuna: n_below = min(ceil(0.1 * n), 25).
    /// This property is kept for backward compatibility but is no longer used.
    /// </summary>
    [Obsolete("GammaFactor is deprecated. Gamma formula is now hardcoded to 0.1*n per Optuna spec.", false)]
    public double GammaFactor { get; init; } = 0.25;

    /// <summary>
    /// Use multivariate Gaussian KDE for the "below" (good) estimator instead of independent 1D KDEs.
    /// Captures parameter correlations in high-dimensional spaces, improving TPE quality at p50+.
    /// The "above" (bad) estimator remains factored (independent per parameter).
    /// </summary>
    public bool Multivariate { get; init; } = true;
}
