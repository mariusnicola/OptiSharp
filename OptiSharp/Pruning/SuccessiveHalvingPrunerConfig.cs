namespace OptiSharp.Pruning;

/// <summary>
/// Configuration for SuccessiveHalvingPruner (Asynchronous Successive Halving Algorithm - SHA).
/// </summary>
public sealed record SuccessiveHalvingPrunerConfig
{
    /// <summary>
    /// Minimum number of trials per rung.
    /// </summary>
    public int MinResource { get; init; } = 1;

    /// <summary>
    /// Reduction factor eta: top ceil(n/eta) trials advance to next rung.
    /// </summary>
    public double ReductionFactor { get; init; } = 3.0;

    /// <summary>
    /// Minimum early stopping rate (not currently used; reserved for future).
    /// </summary>
    public double MinEarlyStoppingRate { get; init; } = 0;
}
