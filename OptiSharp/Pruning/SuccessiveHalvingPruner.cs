using OptiSharp.Models;

namespace OptiSharp.Pruning;

/// <summary>
/// Asynchronous Successive Halving Algorithm (SHA) pruner.
/// Groups trials into rungs; keeps only top 1/eta trials from each rung.
/// </summary>
public sealed class SuccessiveHalvingPruner : IPruner
{
    private readonly SuccessiveHalvingPrunerConfig _config;

    public SuccessiveHalvingPruner(SuccessiveHalvingPrunerConfig? config = null)
    {
        _config = config ?? new SuccessiveHalvingPrunerConfig();

        if (_config.ReductionFactor <= 1)
            throw new ArgumentException("ReductionFactor must be > 1", nameof(config));
    }

    public bool ShouldPrune(Trial trial, IReadOnlyList<Trial> trials)
    {
        // Only prune completed or pruned trials
        if (trial.State == TrialState.Running)
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
            if (other.State != TrialState.Complete)
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

        if (trialsAtRung.Count == 0)
            return false;

        // Sort by value (ascending = best first for minimize)
        trialsAtRung.Sort((a, b) => a.Value.CompareTo(b.Value));

        // Calculate how many trials should survive this rung
        var survivingCount = (int)Math.Ceiling(trialsAtRung.Count / _config.ReductionFactor);
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

        return (int)Math.Floor(Math.Log(steps / _config.MinResource, _config.ReductionFactor));
    }

    private int GetRungResource(int rungIndex)
    {
        return (int)(_config.MinResource * Math.Pow(_config.ReductionFactor, rungIndex));
    }
}
