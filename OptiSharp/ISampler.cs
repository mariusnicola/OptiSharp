using OptiSharp.Models;

namespace OptiSharp;

/// <summary>
/// Suggests parameter values for the next trial based on study history.
/// </summary>
public interface ISampler
{
    /// <summary>
    /// Sample parameters for a single-objective optimization.
    /// </summary>
    Dictionary<string, object> Sample(
        IReadOnlyList<Trial> trials,
        StudyDirection direction,
        SearchSpace searchSpace);

    /// <summary>
    /// Sample parameters for multi-objective optimization.
    /// Default implementation scalarizes to the first objective.
    /// </summary>
    Dictionary<string, object> SampleMultiObjective(
        IReadOnlyList<Trial> trials,
        StudyDirection[] directions,
        SearchSpace searchSpace)
    {
        // Default: scalarize to first objective
        return Sample(trials, directions[0], searchSpace);
    }
}
