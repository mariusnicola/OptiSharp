using OptiSharp.Models;

namespace OptiSharp.Pruning;

/// <summary>
/// Decides whether to prune (early stop) a trial based on its performance history.
/// </summary>
public interface IPruner
{
    /// <summary>
    /// Determines if the given trial should be pruned.
    /// </summary>
    /// <param name="trial">The trial to evaluate for pruning.</param>
    /// <param name="trials">All trials in the study (for comparison).</param>
    /// <returns>True if the trial should be pruned, false otherwise.</returns>
    bool ShouldPrune(Trial trial, IReadOnlyList<Trial> trials);
}
