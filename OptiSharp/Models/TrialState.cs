namespace OptiSharp.Models;

/// <summary>
/// State of a trial in the optimization study.
/// </summary>
public enum TrialState
{
    Running,
    Complete,
    Fail,
    Pruned
}
