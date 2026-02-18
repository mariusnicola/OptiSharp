namespace OptiSharp.Models;

/// <summary>
/// A single trial in the optimization study.
/// </summary>
public sealed class Trial
{
    public Trial(int number, Dictionary<string, object> parameters)
    {
        Number = number;
        Parameters = parameters;
    }

    public int Number { get; }
    public TrialState State { get; internal set; } = TrialState.Running;
    public double? Value { get; internal set; }
    public IReadOnlyDictionary<string, object> Parameters { get; }
}

/// <summary>
/// Result of a trial evaluation, used in TellBatch.
/// </summary>
public readonly record struct TrialResult(int TrialNumber, double? Value, TrialState State);
