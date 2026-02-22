using OptiSharp.Models;

namespace OptiSharp.MultiObjective;

/// <summary>
/// Result of a multi-objective trial evaluation, used in TellBatch.
/// </summary>
public readonly record struct MoTrialResult(int TrialNumber, double[]? Values, TrialState State);
