using OptiSharp.Models;

namespace OptiSharp.Pruning;

/// <summary>
/// Asynchronous Successive Halving Algorithm (SHA) pruner.
/// Groups trials into rungs; keeps only top 1/eta trials from each rung.
/// </summary>
public sealed class SuccessiveHalvingPruner : IPruner
{
    private readonly SuccessiveHalvingPrunerConfig _config;
    private double _effectiveReductionFactor;

    public SuccessiveHalvingPruner(SuccessiveHalvingPrunerConfig? config = null)
    {
        _config = config ?? new SuccessiveHalvingPrunerConfig();

        if (_config.ReductionFactor <= 1)
            throw new ArgumentException("ReductionFactor must be > 1", nameof(config));

        // Apply dimension-aware reduction factor if NDimensions is set
        _effectiveReductionFactor = _config.NDimensions == null
            ? _config.ReductionFactor
            : _config.NDimensions switch
            {
                <= 10 => 2.0,  // p10: keep 50% at each rung (eta=2 instead of 25%)
                <= 50 => 3.0,  // p50: keep 33% (eta=3)
                _ => _config.ReductionFactor  // p100+: use original (4.0)
            };

        if (_effectiveReductionFactor <= 1)
            throw new ArgumentException("Effective ReductionFactor must be > 1");
    }

    public bool ShouldPrune(Trial trial, IReadOnlyList<Trial> trials)
    {
        // Only prune running trials; completed/pruned trials can't be stopped
        if (trial.State != TrialState.Running)
            return false;

        var intermediates = trial.IntermediateValues;
        if (intermediates.Count == 0)
            return false;

        var lastStep = intermediates.Keys.Max();
        var lastValue = intermediates[lastStep];

        // Determine which rung this trial is in based on number of steps
        var rungIndex = GetRungIndex(lastStep);
        var rungResource = GetRungResource(rungIndex);

        // Get all trials that have completed at least this rung
        var trialsAtRung = new List<(int TrialNumber, double Value)>();
        foreach (var other in trials)
        {
            // Include both Complete and Pruned trials (any that reported at rung step)
            if (other.State != TrialState.Complete && other.State != TrialState.Pruned)
                continue;

            var otherLastStep = other.IntermediateValues.Count > 0
                ? other.IntermediateValues.Keys.Max()
                : 0;

            var otherRungIndex = GetRungIndex(otherLastStep);

            // Include trials at this rung or beyond
            if (otherRungIndex >= rungIndex && other.IntermediateValues.TryGetValue(rungResource, out var value))
            {
                trialsAtRung.Add((other.Number, value));
            }
        }

        // Also include the current running trial at rungResource for comparison
        if (intermediates.TryGetValue(rungResource, out var currentValue))
            trialsAtRung.Add((trial.Number, currentValue));

        if (trialsAtRung.Count == 0)
            return false;

        // Require minimum completed trials at rung before deciding to prune.
        // trialsAtRung includes the current running trial we just added, so subtract 1.
        var completedAtRung = trialsAtRung.Count - 1;
        var effectiveMinTrials = _config.GetEffectiveMinTrialsBeforePruning();
        if (completedAtRung < effectiveMinTrials)
            return false;

        // Sort by value (ascending = best first for minimize)
        trialsAtRung.Sort((a, b) => a.Value.CompareTo(b.Value));

        // Calculate how many trials should survive this rung (uses dimension-aware reduction factor)
        var survivingCount = (int)Math.Ceiling(trialsAtRung.Count / _effectiveReductionFactor);
        survivingCount = Math.Max(1, survivingCount);

        // Check if this trial is in the surviving set
        if (trialsAtRung.Count <= survivingCount)
            return false;

        // Find this trial's rank
        var trialRank = trialsAtRung.FindIndex(x => x.TrialNumber == trial.Number);
        if (trialRank < 0)
            return false;

        // Prune if not in top survivingCount
        return trialRank >= survivingCount;
    }

    private int GetRungIndex(int steps)
    {
        if (steps <= _config.MinResource)
            return 0;

        return (int)Math.Floor(Math.Log(steps / _config.MinResource, _effectiveReductionFactor));
    }

    private int GetRungResource(int rungIndex)
    {
        return (int)(_config.MinResource * Math.Pow(_effectiveReductionFactor, rungIndex));
    }
}
