using OptiSharp.Models;

namespace OptiSharp.Pruning;

/// <summary>
/// Pruner that never prunes. Default when no pruner is specified.
/// </summary>
public sealed class NopPruner : IPruner
{
    public bool ShouldPrune(Trial trial, IReadOnlyList<Trial> trials) => false;
}
