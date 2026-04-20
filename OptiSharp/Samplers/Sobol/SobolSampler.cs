using OptiSharp.Models;

namespace OptiSharp.Samplers.Sobol;

/// <summary>
/// Quasi-random Sobol sequence sampler. Provides better space-filling coverage
/// than uniform random sampling due to low-discrepancy properties.
/// Categorical parameters are handled by uniform quantization of the Sobol coordinate.
/// </summary>
public sealed class SobolSampler : ISampler
{
    private readonly SobolSamplerConfig _config;
    private SobolSequence? _sobolSequence;
    private HaltonSequence? _haltonSequence;
    private int _dimensionCount;
    private bool _useHalton;  // True if using Halton for high dimensions

    public SobolSampler(SobolSamplerConfig? config = null)
    {
        _config = config ?? new SobolSamplerConfig();
    }

    public Dictionary<string, object> Sample(
        IReadOnlyList<Trial> trials,
        StudyDirection direction,
        SearchSpace searchSpace)
    {
        EnsureInitialized(searchSpace);

        var point = _useHalton
            ? _haltonSequence!.NextPoint()
            : _sobolSequence!.NextPoint();
        return MapToParameters(point, searchSpace);
    }

    private void EnsureInitialized(SearchSpace searchSpace)
    {
        if (_sobolSequence is not null || _haltonSequence is not null) return;

        _dimensionCount = searchSpace.Count;

        // Use Halton for p50+ (better low-discrepancy in medium+ dimensions)
        // Sobol for p10-50 (better 2D/3D/low-D properties)
        if (_dimensionCount >= 50 && _dimensionCount <= HaltonSequence.MaxDimensions)
        {
            _useHalton = true;
            _haltonSequence = new HaltonSequence(_dimensionCount, _config.Seed);
            if (_config.Skip > 0)
                _haltonSequence.Skip(_config.Skip);
        }
        else if (_dimensionCount < 50 && _dimensionCount <= SobolSequence.MaxDimensions)
        {
            _useHalton = false;
            _sobolSequence = new SobolSequence(_dimensionCount, _config.Seed);
            if (_config.Skip > 0)
                _sobolSequence.Skip(_config.Skip);
        }
        else
        {
            throw new ArgumentException(
                $"SobolSampler supports up to {SobolSequence.MaxDimensions} dimensions (or {HaltonSequence.MaxDimensions} via Halton), " +
                $"but search space has {_dimensionCount} parameters.");
        }
    }

    private Dictionary<string, object> MapToParameters(double[] point, SearchSpace searchSpace)
    {
        var parameters = new Dictionary<string, object>(searchSpace.Count);

        for (var i = 0; i < searchSpace.Count; i++)
        {
            var u = point[i]; // Value in [0, 1)
            var range = searchSpace[i];

            parameters[range.Name] = range switch
            {
                FloatRange fr => MapFloat(u, fr),
                IntRange ir => MapInt(u, ir),
                CategoricalRange cr => MapCategorical(u, cr),
                _ => throw new ArgumentException($"Unknown parameter range type: {range.GetType().Name}")
            };
        }

        return parameters;
    }

    private static object MapFloat(double u, FloatRange range)
    {
        if (range.Log)
        {
            var logLow = Math.Log(range.Low);
            var logHigh = Math.Log(range.High);
            return Math.Exp(logLow + u * (logHigh - logLow));
        }

        return range.Low + u * (range.High - range.Low);
    }

    private static object MapInt(double u, IntRange range)
    {
        if (range.Step <= 1)
        {
            // Map [0,1) to [Low, High] inclusive
            var value = (int)Math.Floor(range.Low + u * (range.High - range.Low + 1));
            return Math.Clamp(value, range.Low, range.High);
        }

        // Stepped integer: pick one of the valid steps
        var steps = (range.High - range.Low) / range.Step;
        var stepIndex = (int)Math.Floor(u * (steps + 1));
        stepIndex = Math.Clamp(stepIndex, 0, steps);
        return range.Low + stepIndex * range.Step;
    }

    private static object MapCategorical(double u, CategoricalRange range)
    {
        var index = (int)Math.Floor(u * range.Choices.Length);
        index = Math.Clamp(index, 0, range.Choices.Length - 1);
        return range.Choices[index];
    }
}
