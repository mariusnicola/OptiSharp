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
    public double ReductionFactor { get; init; } = 4.0;

    /// <summary>
    /// Minimum early stopping rate (not currently used; reserved for future).
    /// </summary>
    public double MinEarlyStoppingRate { get; init; } = 0;

    /// <summary>
    /// Minimum number of completed trials that must have reached a rung before
    /// any pruning decisions are made at that rung.
    /// Prevents premature elimination when only 1-2 trials have reported.
    /// Default: 1 (matches Optuna's bootstrap_count=0 behavior).
    /// Can be overridden via GetEffectiveMinTrialsBeforePruning() if NDimensions is set.
    /// </summary>
    public int MinTrialsBeforePruning { get; init; } = 1;

    /// <summary>
    /// Optional: number of dimensions for auto-tuning min trials before pruning.
    /// If set, MinTrialsBeforePruning is overridden with dimension-aware defaults:
    /// - p10: MinTrialsBeforePruning = 5 (relax pruning, allow more trials to complete)
    /// - p50: MinTrialsBeforePruning = 2
    /// - p100+: MinTrialsBeforePruning = 1 (original default)
    /// </summary>
    public int? NDimensions { get; init; } = null;

    /// <summary>
    /// Get the effective MinTrialsBeforePruning, considering dimension-aware tuning.
    /// </summary>
    public int GetEffectiveMinTrialsBeforePruning()
    {
        if (NDimensions == null)
            return MinTrialsBeforePruning;

        return NDimensions switch
        {
            <= 10 => 5,    // p10: require 5 completed trials before pruning (relax aggressiveness)
            <= 50 => 2,    // p50: require 2 completed trials
            _ => 1         // p100+: keep original default (more trials needed to fit budget)
        };
    }
}
