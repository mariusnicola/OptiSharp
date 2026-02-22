using OptiSharp.Models;

namespace OptiSharp.Pruning;

/// <summary>
/// Prunes trials that exceed the specified percentile of completed trials at the same step.
/// </summary>
public sealed class PercentilePruner : IPruner
{
    private readonly PercentilePrunerConfig _config;

    public PercentilePruner(PercentilePrunerConfig? config = null)
    {
        _config = config ?? new PercentilePrunerConfig();

        if (_config.Percentile < 0 || _config.Percentile > 100)
            throw new ArgumentException("Percentile must be between 0 and 100", nameof(config));
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

        // Skip warmup phase
        if (lastStep <= _config.NWarmupSteps)
            return false;

        // Check interval
        if (lastStep % _config.IntervalSteps != 0)
            return false;

        var lastValue = intermediates[lastStep];

        // Get completed trials with intermediate values at the same step
        var completedValues = new List<double>();
        foreach (var other in trials)
        {
            if (other.Number == trial.Number)
                continue;

            if (other.State != TrialState.Complete)
                continue;

            if (other.IntermediateValues.TryGetValue(lastStep, out var value))
                completedValues.Add(value);
        }

        // Need enough completed trials for meaningful comparison
        if (completedValues.Count < _config.NStartupTrials)
            return false;

        completedValues.Sort();

        // Calculate percentile value
        var index = (int)Math.Ceiling(_config.Percentile / 100.0 * completedValues.Count) - 1;
        index = Math.Max(0, Math.Min(index, completedValues.Count - 1));
        var percentileValue = completedValues[index];

        // Assume higher is worse (minimize mode)
        return lastValue > percentileValue;
    }
}
