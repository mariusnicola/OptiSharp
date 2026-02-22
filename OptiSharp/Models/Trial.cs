using System.Collections.Concurrent;

namespace OptiSharp.Models;

/// <summary>
/// A single trial in the optimization study.
/// </summary>
public sealed class Trial
{
    private readonly ConcurrentDictionary<int, double> _intermediateValues = new();

    public Trial(int number, Dictionary<string, object> parameters)
    {
        Number = number;
        Parameters = parameters;
    }

    public int Number { get; }
    public TrialState State { get; internal set; } = TrialState.Running;
    public double? Value { get; internal set; }
    public double[]? Values { get; internal set; }
    public double[]? ConstraintValues { get; internal set; }
    public IReadOnlyDictionary<string, object> Parameters { get; }
    public IReadOnlyDictionary<int, double> IntermediateValues => _intermediateValues;

    /// <summary>
    /// Report an intermediate value at a specific step.
    /// Thread-safe for concurrent reporting from evaluation threads.
    /// </summary>
    public void Report(double value, int step)
    {
        _intermediateValues[step] = value;
    }
}

/// <summary>
/// Result of a trial evaluation, used in TellBatch.
/// </summary>
public readonly record struct TrialResult(int TrialNumber, double? Value, TrialState State);
