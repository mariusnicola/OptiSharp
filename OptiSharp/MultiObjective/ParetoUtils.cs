using OptiSharp.Models;

namespace OptiSharp.MultiObjective;

/// <summary>
/// Utilities for Pareto front computation and multi-objective analysis.
/// </summary>
public static class ParetoUtils
{
    /// <summary>
    /// Checks if one solution dominates another.
    /// Higher values are better if direction is Maximize, lower if Minimize.
    /// </summary>
    public static bool Dominates(double[] a, double[] b, StudyDirection[] directions)
    {
        if (a.Length != b.Length || a.Length != directions.Length)
            throw new ArgumentException("Array lengths must match");

        bool atLeastOneBetter = false;
        for (int i = 0; i < a.Length; i++)
        {
            bool aBetter = directions[i] == StudyDirection.Minimize
                ? a[i] < b[i]
                : a[i] > b[i];

            bool bBetter = directions[i] == StudyDirection.Minimize
                ? b[i] < a[i]
                : b[i] > a[i];

            if (bBetter)
                return false; // b is better in at least one objective

            if (aBetter)
                atLeastOneBetter = true;
        }

        return atLeastOneBetter;
    }

    /// <summary>
    /// Computes the Pareto front from a set of completed trials.
    /// O(n^2 * m) complexity where n = trials, m = objectives.
    /// </summary>
    public static IReadOnlyList<Trial> ComputeParetoFront(IEnumerable<Trial> trials, StudyDirection[] directions)
    {
        var completedTrials = trials
            .Where(t => t.State == TrialState.Complete && t.Values != null && t.Values.Length == directions.Length)
            .ToList();

        if (completedTrials.Count == 0)
            return [];

        var front = new List<Trial>();

        foreach (var candidate in completedTrials)
        {
            // Check if candidate is dominated by any trial in front
            bool dominated = false;
            var indicesToRemove = new List<int>();

            for (int i = 0; i < front.Count; i++)
            {
                if (Dominates(front[i].Values!, candidate.Values!, directions))
                {
                    dominated = true;
                    break;
                }

                // Remove front members dominated by candidate
                if (Dominates(candidate.Values!, front[i].Values!, directions))
                {
                    indicesToRemove.Add(i);
                }
            }

            if (!dominated)
            {
                // Remove dominated trials in reverse order
                for (int i = indicesToRemove.Count - 1; i >= 0; i--)
                    front.RemoveAt(indicesToRemove[i]);

                front.Add(candidate);
            }
        }

        return front.AsReadOnly();
    }

    /// <summary>
    /// Computes crowding distances for trials in a front.
    /// Used for diversity preservation in multi-objective optimization.
    /// </summary>
    public static Dictionary<int, double> CrowdingDistances(
        IReadOnlyList<Trial> front,
        StudyDirection[] directions)
    {
        var distances = new Dictionary<int, double>();

        if (front.Count <= 2)
        {
            // All trials get infinite distance
            foreach (var trial in front)
                distances[trial.Number] = double.PositiveInfinity;
            return distances;
        }

        // Initialize distances
        foreach (var trial in front)
            distances[trial.Number] = 0;

        // For each objective
        for (int m = 0; m < directions.Length; m++)
        {
            // Sort by m-th objective
            var sorted = front.OrderBy(t => t.Values![m]).ToList();

            // Boundary points get infinite distance
            distances[sorted[0].Number] = double.PositiveInfinity;
            distances[sorted[^1].Number] = double.PositiveInfinity;

            // Compute distances for intermediate points
            var min = sorted[0].Values![m];
            var max = sorted[^1].Values![m];
            var range = max - min;

            if (range > 0)
            {
                for (int i = 1; i < sorted.Count - 1; i++)
                {
                    var distance = (sorted[i + 1].Values![m] - sorted[i - 1].Values![m]) / range;
                    distances[sorted[i].Number] += distance;
                }
            }
        }

        return distances;
    }
}
