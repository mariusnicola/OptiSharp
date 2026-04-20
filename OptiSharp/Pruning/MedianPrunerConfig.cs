namespace OptiSharp.Pruning;

/// <summary>
/// Configuration for MedianPruner.
/// </summary>
public sealed record MedianPrunerConfig
{
    /// <summary>
    /// Minimum number of completed trials before pruning is enabled.
    /// </summary>
    public int NStartupTrials { get; init; } = 5;

    /// <summary>
    /// Number of initial steps to skip before pruning is evaluated.
    /// Default: 0. For low-dimensional problems (p10), consider setting higher (e.g., 10-20).
    /// </summary>
    public int NWarmupSteps { get; init; } = 0;

    /// <summary>
    /// Interval at which to check for pruning (every Nth step).
    /// </summary>
    public int IntervalSteps { get; init; } = 1;

    /// <summary>
    /// Optional: number of dimensions for auto-tuning warmup steps.
    /// If set, NWarmupSteps is overridden with dimension-aware defaults:
    /// - p10: NWarmupSteps = 15 (relax pruning for smooth functions)
    /// - p50: NWarmupSteps = 3
    /// - p100+: NWarmupSteps = 1
    /// </summary>
    public int? NDimensions { get; init; } = null;

    /// <summary>
    /// Get the effective NWarmupSteps, considering dimension-aware tuning.
    /// </summary>
    public int GetEffectiveNWarmupSteps()
    {
        if (NDimensions == null)
            return NWarmupSteps;

        return NDimensions switch
        {
            <= 10 => 15,   // p10: long warmup (sphere is smooth)
            <= 50 => 3,    // p50: moderate warmup
            _ => 1         // p100+: minimal warmup (more trials per budget)
        };
    }
}
