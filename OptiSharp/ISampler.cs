using OptiSharp.Models;

namespace OptiSharp;

/// <summary>
/// Suggests parameter values for the next trial based on study history.
/// </summary>
public interface ISampler
{
    Dictionary<string, object> Sample(
        IReadOnlyList<Trial> trials,
        StudyDirection direction,
        SearchSpace searchSpace);
}
