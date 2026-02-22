using OptiSharp.Models;

namespace OptiSharp.Pruning;

/// <summary>
/// Prunes trials that exceed the median of completed trials at the same step.
/// </summary>
public sealed class MedianPruner : IPruner
{
    private readonly MedianPrunerConfig _config;

    public MedianPruner(MedianPrunerConfig? config = null)
    {
        _config = config ?? new MedianPrunerConfig();
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
        var median = completedValues.Count % 2 == 0
            ? (completedValues[completedValues.Count / 2 - 1] + completedValues[completedValues.Count / 2]) / 2.0
            : completedValues[completedValues.Count / 2];

        // Assume higher is worse (minimize mode)
        return lastValue > median;
    }
}
