using OptiSharp.Models;
using OptiSharp.Samplers.CmaEs;

namespace OptiSharp.Tests;

/// <summary>
/// Shared test utilities to avoid duplication across test classes.
/// </summary>
internal static class TestHelpers
{
    // ── Search spaces ─────────────────────────────────────────────────

    /// <summary>
    /// Creates a continuous N-dimensional search space with FloatRange parameters.
    /// </summary>
    public static SearchSpace MakeSpace(int dims, double low = -5, double high = 5)
        => new(Enumerable.Range(0, dims).Select(i => new FloatRange($"x{i}", low, high)).ToArray());

    /// <summary>
    /// Creates the standard 62-parameter mixed space (55 float, 5 int, 2 categorical).
    /// </summary>
    public static SearchSpace Create62ParamSpace()
    {
        var ranges = new List<ParameterRange>();
        for (var i = 0; i < 55; i++)
            ranges.Add(new FloatRange($"f{i}", 0, 10));
        for (var i = 0; i < 5; i++)
            ranges.Add(new IntRange($"i{i}", 0, 100));
        for (var i = 0; i < 2; i++)
            ranges.Add(new CategoricalRange($"c{i}", ["a", "b", "c"]));
        return new SearchSpace(ranges);
    }

    // ── Standard objective functions ──────────────────────────────────

    public static double Sphere(IReadOnlyDictionary<string, object> p, int dims)
        => Enumerable.Range(0, dims).Sum(i => Math.Pow((double)p[$"x{i}"], 2));

    public static double Rosenbrock(IReadOnlyDictionary<string, object> p)
    {
        var x = (double)p["x0"];
        var y = (double)p["x1"];
        return Math.Pow(1 - x, 2) + 100 * Math.Pow(y - x * x, 2);
    }

    public static double Rastrigin(IReadOnlyDictionary<string, object> p, int dims)
        => 10.0 * dims + Enumerable.Range(0, dims).Sum(i =>
        {
            var xi = (double)p[$"x{i}"];
            return xi * xi - 10.0 * Math.Cos(2.0 * Math.PI * xi);
        });

    public static double Ackley(IReadOnlyDictionary<string, object> p, int dims)
    {
        var sumSq = Enumerable.Range(0, dims).Sum(i => Math.Pow((double)p[$"x{i}"], 2));
        var sumCos = Enumerable.Range(0, dims).Sum(i => Math.Cos(2.0 * Math.PI * (double)p[$"x{i}"]));
        return -20.0 * Math.Exp(-0.2 * Math.Sqrt(sumSq / dims))
            - Math.Exp(sumCos / dims) + 20.0 + Math.E;
    }

    // ── Reflection helpers ────────────────────────────────────────────

    public static CmaEsSampler? GetSamplerFromStudy(Study study)
    {
        var field = typeof(Study).GetField("_sampler",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        return field?.GetValue(study) as CmaEsSampler;
    }

    // ── Statistics ────────────────────────────────────────────────────

    public static double Median(double[] values)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        var mid = sorted.Length / 2;
        return sorted.Length % 2 == 0
            ? (sorted[mid - 1] + sorted[mid]) / 2.0
            : sorted[mid];
    }

    // ── Test-only types ──────────────────────────────────────────────

    /// <summary>
    /// Dummy ParameterRange subclass for testing unknown-type handling.
    /// </summary>
    public sealed record TestOnlyParameterRange(string Name) : ParameterRange(Name);
}
