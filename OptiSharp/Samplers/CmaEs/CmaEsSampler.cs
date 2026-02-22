using MathNet.Numerics.LinearAlgebra;
using OptiSharp.Models;

namespace OptiSharp.Samplers.CmaEs;

/// <summary>
/// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) sampler.
/// Population-based optimizer that handles parameter correlations.
/// Categorical parameters are sampled randomly (CMA-ES only handles continuous).
/// Supports optional GPU acceleration via CUDA for large search spaces.
/// </summary>
public sealed class CmaEsSampler : ISampler, IDisposable
{
    /// <summary>
    /// Recommended minimum dimensions for GPU to outperform CPU.
    /// Below this threshold, GPU kernel launch + transfer overhead exceeds CPU time.
    /// </summary>
    public const int GpuRecommendedMinDimensions = 100;

    private readonly CmaEsSamplerConfig _config;
    private readonly Random _rng;

    // Lazy init — dimension count unknown until first Sample()
    private CmaEsState? _state;
    private GpuCmaEsProvider? _gpu;
    private int _continuousDimCount;

    // Generation management
    private readonly List<(int TrialNumber, Vector<double> Candidate)> _currentGeneration = [];
    private int _issuedCount;
    private Vector<double>[]? _population;

    // Parameter mapping (order-sensitive — maps to vector indices)
    private string[]? _continuousParamNames;
    private double[]? _lows;
    private double[]? _highs;
    private bool[]? _isLog;
    private CategoricalRange[]? _categoricalRanges;

    public CmaEsSampler(CmaEsSamplerConfig? config = null)
    {
        _config = config ?? new CmaEsSamplerConfig();
        _rng = _config.Seed.HasValue ? new Random(_config.Seed.Value) : new Random();
    }

    /// <summary>
    /// Current metrics snapshot. Null before first generation completes.
    /// </summary>
    public CmaEsMetrics? Metrics { get; private set; }

    /// <summary>
    /// Warning about GPU usage with small dimension count.
    /// Null if no warning applicable (CPU mode or dimensions >= threshold).
    /// </summary>
    public string? GpuDimensionWarning { get; private set; }

    /// <summary>
    /// Whether GPU acceleration is active for this sampler.
    /// </summary>
    public bool IsGpuActive => _gpu?.IsGpu == true;

    /// <summary>
    /// Name of the active compute device.
    /// </summary>
    public string? DeviceName => _gpu?.DeviceName;

    public Dictionary<string, object> Sample(
        IReadOnlyList<Trial> trials,
        StudyDirection direction,
        SearchSpace searchSpace)
    {
        EnsureInitialized(searchSpace);

        // Check if previous generation is fully evaluated → trigger CMA-ES update
        TryCompleteGeneration(trials, direction);

        // Generate new population if needed
        if (_population is null || _issuedCount >= _population.Length)
            GeneratePopulation();

        // Take next candidate from population buffer
        var candidate = _population![_issuedCount];

        // trials.Count = the trial number Study will assign to this trial
        // (Study.AskCore calls Sample first, then assigns sequential number)
        var trialNumber = trials.Count;
        _currentGeneration.Add((trialNumber, candidate));
        _issuedCount++;

        return MapToParameters(candidate, searchSpace);
    }

    private void EnsureInitialized(SearchSpace searchSpace)
    {
        if (_state is not null) return;

        try
        {
            var names = new List<string>();
            var lows = new List<double>();
            var highs = new List<double>();
            var isLog = new List<bool>();
            var categoricals = new List<CategoricalRange>();

            foreach (var range in searchSpace)
            {
                switch (range)
                {
                    case FloatRange fr:
                        names.Add(fr.Name);
                        lows.Add(fr.Log ? Math.Log(fr.Low) : fr.Low);
                        highs.Add(fr.Log ? Math.Log(fr.High) : fr.High);
                        isLog.Add(fr.Log);
                        break;
                    case IntRange ir:
                        names.Add(ir.Name);
                        lows.Add(ir.Low);
                        highs.Add(ir.High);
                        isLog.Add(false);
                        break;
                    case CategoricalRange cr:
                        categoricals.Add(cr);
                        break;
                }
            }

            _continuousParamNames = names.ToArray();
            _lows = lows.ToArray();
            _highs = highs.ToArray();
            _isLog = isLog.ToArray();
            _categoricalRanges = categoricals.ToArray();
            _continuousDimCount = _continuousParamNames.Length;

            if (_continuousDimCount == 0)
                throw new ArgumentException("CMA-ES requires at least one continuous parameter (Float or Int).");

            // Initialize GPU if requested
            if (_config.Backend != ComputeBackend.Cpu)
            {
                _gpu = GpuCmaEsProvider.TryCreate(_config.Backend);

                // Emit warning for small dimensions
                if (_gpu?.IsGpu == true && _continuousDimCount < GpuRecommendedMinDimensions)
                {
                    GpuDimensionWarning =
                        $"GPU compute active ({_gpu.DeviceName}) but dimensions ({_continuousDimCount}) " +
                        $"< recommended minimum ({GpuRecommendedMinDimensions}). " +
                        "GPU kernel launch + PCIe transfer overhead likely exceeds CPU computation time. " +
                        "Consider ComputeBackend.Cpu for better performance at this scale.";
                }
            }

            // Initial mean = center of search space
            var initialMean = new double[_continuousDimCount];
            for (var i = 0; i < _continuousDimCount; i++)
                initialMean[i] = (_lows[i] + _highs[i]) * 0.5;

            // Initial sigma scaled to average range
            var avgRange = 0.0;
            for (var i = 0; i < _continuousDimCount; i++)
                avgRange += _highs[i] - _lows[i];
            avgRange /= _continuousDimCount;
            var sigma = _config.InitialSigma * avgRange;

            var lambda = _config.PopulationSize
                ?? 4 + (int)Math.Floor(3.0 * Math.Log(_continuousDimCount));

            _state = new CmaEsState(_continuousDimCount, lambda, sigma, initialMean, _gpu);
        }
        catch
        {
            _gpu?.Dispose();
            _gpu = null;
            throw;
        }
    }

    private void TryCompleteGeneration(IReadOnlyList<Trial> trials, StudyDirection direction)
    {
        if (_currentGeneration.Count < _state!.Lambda) return;

        // Check if all lambda trials from current generation are done
        var results = new List<(double Value, Vector<double> Candidate)>();

        foreach (var (trialNumber, candidate) in _currentGeneration)
        {
            if (trialNumber < 0 || trialNumber >= trials.Count) return;

            var trial = trials[trialNumber];

            if (trial.State == TrialState.Complete && trial.Value.HasValue)
            {
                results.Add((trial.Value.Value, candidate));
            }
            else if (trial.State is TrialState.Fail or TrialState.Pruned)
            {
                var worstValue = direction == StudyDirection.Minimize ? double.MaxValue : double.MinValue;
                results.Add((worstValue, candidate));
            }
            else
            {
                return; // Still running — can't update yet
            }
        }

        // All done — sort by fitness (best first) and update
        if (direction == StudyDirection.Minimize)
            results.Sort((a, b) => a.Value.CompareTo(b.Value));
        else
            results.Sort((a, b) => b.Value.CompareTo(a.Value));

        var ranked = results.Select(r => r.Candidate).ToArray();
        _state.Update(ranked);

        // Update metrics
        var bestInGen = direction == StudyDirection.Minimize
            ? results.Min(r => r.Value)
            : results.Max(r => r.Value);

        Metrics = new CmaEsMetrics
        {
            Generation = _state.Generation,
            Sigma = _state.Sigma,
            ConditionNumber = _state.ConditionNumber,
            BestValue = bestInGen,
            EvaluatedTrials = trials.Count(t => t.State == TrialState.Complete)
        };

        // Reset for next generation
        _currentGeneration.Clear();
        _population = null;
        _issuedCount = 0;
    }

    private void GeneratePopulation()
    {
        var (candidates, _) = _state!.SamplePopulation(_rng);

        for (var i = 0; i < candidates.Length; i++)
            for (var j = 0; j < _continuousDimCount; j++)
                candidates[i][j] = MirrorClip(candidates[i][j], _lows![j], _highs![j]);

        _population = candidates;
        _issuedCount = 0;
    }

    private Dictionary<string, object> MapToParameters(Vector<double> candidate, SearchSpace searchSpace)
    {
        var parameters = new Dictionary<string, object>(searchSpace.Count);

        for (var i = 0; i < _continuousDimCount; i++)
        {
            var name = _continuousParamNames![i];
            var value = candidate[i];
            var range = searchSpace[name];

            if (range is FloatRange fr)
            {
                var mapped = _isLog![i] ? Math.Exp(value) : value;
                parameters[name] = Math.Clamp(mapped, fr.Low, fr.High);
            }
            else if (range is IntRange ir)
            {
                var rounded = (int)Math.Round(value);
                if (ir.Step > 1)
                    rounded = ir.Low + (int)Math.Round((double)(rounded - ir.Low) / ir.Step) * ir.Step;
                parameters[name] = Math.Clamp(rounded, ir.Low, ir.High);
            }
        }

        foreach (var cr in _categoricalRanges!)
            parameters[cr.Name] = cr.Choices[_rng.Next(cr.Choices.Length)];

        return parameters;
    }

    /// <summary>
    /// Mirror-reflection boundary handling (standard CMA-ES technique).
    /// Preserves the shape of the search distribution near boundaries, unlike
    /// simple clamping which creates bias at the edges. 10 iterations suffice
    /// because each reflection halves the overshoot distance.
    /// </summary>
    private static double MirrorClip(double x, double low, double high)
    {
        var iterations = 0;
        while ((x < low || x > high) && iterations++ < 10)
        {
            if (x < low) x = low + (low - x);
            if (x > high) x = high - (x - high);
        }
        return Math.Clamp(x, low, high);
    }

    public void Dispose()
    {
        _gpu?.Dispose();
    }
}

/// <summary>
/// Observability metrics for CMA-ES sampler state.
/// </summary>
public sealed record CmaEsMetrics
{
    public int Generation { get; init; }
    public double Sigma { get; init; }
    public double ConditionNumber { get; init; }
    public double BestValue { get; init; }
    public int EvaluatedTrials { get; init; }
}
