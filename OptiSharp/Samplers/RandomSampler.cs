using OptiSharp.Models;

namespace OptiSharp.Samplers;

/// <summary>
/// Uniform random sampler. Used as baseline and during TPE startup phase.
/// </summary>
public sealed class RandomSampler : ISampler
{
    private readonly Random _rng;

    public RandomSampler(int? seed = null)
    {
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    internal RandomSampler(Random rng)
    {
        _rng = rng;
    }

    public Dictionary<string, object> Sample(
        IReadOnlyList<Trial> trials,
        StudyDirection direction,
        SearchSpace searchSpace)
    {
        var parameters = new Dictionary<string, object>(searchSpace.Count);
        foreach (var range in searchSpace)
        {
            parameters[range.Name] = SampleParameter(range);
        }
        return parameters;
    }

    internal object SampleParameter(ParameterRange range) => range switch
    {
        IntRange ir => SampleInt(ir),
        FloatRange fr => SampleFloat(fr),
        CategoricalRange cr => cr.Choices[_rng.Next(cr.Choices.Length)],
        _ => throw new ArgumentException($"Unknown parameter range type: {range.GetType().Name}")
    };

    private int SampleInt(IntRange range)
    {
        if (range.Step <= 1)
            return _rng.Next(range.Low, range.High + 1);

        var steps = (range.High - range.Low) / range.Step;
        return range.Low + _rng.Next(steps + 1) * range.Step;
    }

    private double SampleFloat(FloatRange range)
    {
        if (range.Log)
        {
            var logLow = Math.Log(range.Low);
            var logHigh = Math.Log(range.High);
            return Math.Exp(logLow + _rng.NextDouble() * (logHigh - logLow));
        }

        return range.Low + _rng.NextDouble() * (range.High - range.Low);
    }
}
