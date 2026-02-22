namespace OptiSharp.Pruning;

/// <summary>
/// Configuration for PercentilePruner.
/// </summary>
public sealed record PercentilePrunerConfig
{
    /// <summary>
    /// Percentile threshold (0-100). Trials worse than this percentile are pruned.
    /// </summary>
    public double Percentile { get; init; } = 25.0;

    /// <summary>
    /// Minimum number of completed trials before pruning is enabled.
    /// </summary>
    public int NStartupTrials { get; init; } = 5;

    /// <summary>
    /// Number of initial steps to skip before pruning is evaluated.
    /// </summary>
    public int NWarmupSteps { get; init; } = 0;

    /// <summary>
    /// Interval at which to check for pruning (every Nth step).
    /// </summary>
    public int IntervalSteps { get; init; } = 1;
}
