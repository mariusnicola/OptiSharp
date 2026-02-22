using OptiSharp.Models;

namespace OptiSharp.Samplers.Tpe;

/// <summary>
/// Tree-structured Parzen Estimator sampler.
/// Fits two kernel density estimates (good/bad trials) and maximizes EI = l(x)/g(x).
/// </summary>
public sealed class TpeSampler : ISampler
{
    private readonly TpeSamplerConfig _config;
    private readonly RandomSampler _randomFallback;
    private readonly Random _rng;

    public TpeSampler(TpeSamplerConfig? config = null)
    {
        _config = config ?? new TpeSamplerConfig();
        _rng = _config.Seed.HasValue ? new Random(_config.Seed.Value) : new Random();
        _randomFallback = new RandomSampler(_rng);
    }

    public Dictionary<string, object> Sample(
        IReadOnlyList<Trial> trials,
        StudyDirection direction,
        SearchSpace searchSpace)
    {
        // Get completed trials with values
        var completed = GetCompletedTrials(trials);

        // Startup phase: use random sampling
        if (completed.Count < _config.NStartupTrials)
            return _randomFallback.Sample(trials, direction, searchSpace);

        // Sort by value: best first
        completed.Sort((a, b) => direction == StudyDirection.Minimize
            ? a.Value!.Value.CompareTo(b.Value!.Value)
            : b.Value!.Value.CompareTo(a.Value!.Value));

        // Split into below (good) and above (bad) groups
        var (belowTrials, aboveTrials) = SplitBelowAbove(completed, trials);

        // Sample each parameter independently
        var parameters = new Dictionary<string, object>(searchSpace.Count);
        foreach (var range in searchSpace)
        {
            parameters[range.Name] = SampleParameter(range, belowTrials, aboveTrials);
        }

        return parameters;
    }

    /// <summary>
    /// Batch-optimized sampling: pre-computes sort/split/extract/estimators once,
    /// then samples count times from the same KDE models.
    /// ~Nx faster than calling Sample() in a loop for large completed trial counts.
    /// </summary>
    internal Dictionary<string, object>[] SampleBatch(
        IReadOnlyList<Trial> trials,
        StudyDirection direction,
        SearchSpace searchSpace,
        int count)
    {
        var completed = GetCompletedTrials(trials);

        // Startup phase: random sampling for all
        if (completed.Count < _config.NStartupTrials)
        {
            var results = new Dictionary<string, object>[count];
            for (var i = 0; i < count; i++)
                results[i] = _randomFallback.Sample(trials, direction, searchSpace);
            return results;
        }

        // Sort once
        completed.Sort((a, b) => direction == StudyDirection.Minimize
            ? a.Value!.Value.CompareTo(b.Value!.Value)
            : b.Value!.Value.CompareTo(a.Value!.Value));

        // Split once
        var (belowTrials, aboveTrials) = SplitBelowAbove(completed, trials);

        // Pre-extract per-parameter observations and build estimators ONCE
        var paramEstimators = new Dictionary<string, object>(searchSpace.Count);
        foreach (var range in searchSpace)
        {
            paramEstimators[range.Name] = PreBuildEstimator(range, belowTrials, aboveTrials);
        }

        // Sample count times from pre-built estimators (FAST path)
        var batch = new Dictionary<string, object>[count];
        for (var i = 0; i < count; i++)
        {
            var parameters = new Dictionary<string, object>(searchSpace.Count);
            foreach (var range in searchSpace)
            {
                parameters[range.Name] = SampleFromCached(range, paramEstimators[range.Name]);
            }
            batch[i] = parameters;
        }

        return batch;
    }

    private object SampleParameter(ParameterRange range, List<Trial> below, List<Trial> above)
    {
        return range switch
        {
            FloatRange fr => SampleFloat(fr, below, above),
            IntRange ir => SampleInt(ir, below, above),
            CategoricalRange cr => SampleCategorical(cr, below, above),
            _ => throw new ArgumentException($"Unknown range type: {range.GetType().Name}")
        };
    }

    private double SampleFloat(FloatRange range, List<Trial> below, List<Trial> above)
    {
        var isLog = range.Log;
        var low = isLog ? Math.Log(range.Low) : range.Low;
        var high = isLog ? Math.Log(range.High) : range.High;

        var belowObs = ExtractSortedDoubles(below, range.Name, isLog);
        var aboveObs = ExtractSortedDoubles(above, range.Name, isLog);

        var result = SampleWithEi(belowObs, aboveObs, low, high);
        return isLog ? Math.Clamp(Math.Exp(result), range.Low, range.High) : Math.Clamp(result, range.Low, range.High);
    }

    private int SampleInt(IntRange range, List<Trial> below, List<Trial> above)
    {
        var low = (double)range.Low;
        var high = (double)range.High;

        var belowObs = ExtractSortedDoubles(below, range.Name, log: false);
        var aboveObs = ExtractSortedDoubles(above, range.Name, log: false);

        var result = SampleWithEi(belowObs, aboveObs, low, high);
        var rounded = (int)Math.Round(Math.Clamp(result, low, high));

        // Align to step
        if (range.Step > 1)
            rounded = range.Low + (int)Math.Round((double)(rounded - range.Low) / range.Step) * range.Step;

        return Math.Clamp(rounded, range.Low, range.High);
    }

    private object SampleCategorical(CategoricalRange range, List<Trial> below, List<Trial> above)
    {
        var choices = range.Choices;
        var nChoices = choices.Length;

        var belowObs = ExtractCategoricalIndices(below, range.Name, choices);
        var aboveObs = ExtractCategoricalIndices(above, range.Name, choices);

        var estBelow = new CategoricalEstimator(belowObs, nChoices, _config.PriorWeight);
        var estAbove = new CategoricalEstimator(aboveObs, nChoices, _config.PriorWeight);

        // Sample candidates from below estimator, pick max EI
        var bestIdx = 0;
        var bestEi = double.NegativeInfinity;

        for (var i = 0; i < _config.NEiCandidates; i++)
        {
            var candidate = estBelow.Sample(_rng);
            var ei = estBelow.LogPdf(candidate) - estAbove.LogPdf(candidate);
            if (ei > bestEi)
            {
                bestEi = ei;
                bestIdx = candidate;
            }
        }

        return choices[bestIdx];
    }

    /// <summary>
    /// Core EI maximization for continuous parameters.
    /// Builds Parzen estimators for below/above, samples candidates from below, returns max EI candidate.
    /// </summary>
    private double SampleWithEi(double[] belowObs, double[] aboveObs, double low, double high)
    {
        var estBelow = new ParzenEstimator(belowObs, low, high, _config.PriorWeight, _config.ConsiderMagicClip);
        var estAbove = new ParzenEstimator(aboveObs, low, high, _config.PriorWeight, _config.ConsiderMagicClip);

        // Sample candidates from below (good) estimator
        var candidates = estBelow.Sample(_rng, _config.NEiCandidates);

        // Compute EI = log l(x) - log g(x)
        var logL = estBelow.LogPdf(candidates);
        var logG = estAbove.LogPdf(candidates);

        // Pick candidate with maximum EI
        var bestIdx = 0;
        var bestEi = double.NegativeInfinity;
        for (var i = 0; i < candidates.Length; i++)
        {
            var ei = logL[i] - logG[i];
            if (ei > bestEi)
            {
                bestEi = ei;
                bestIdx = i;
            }
        }

        return candidates[bestIdx];
    }

    // ========== SPLIT & SUBSAMPLE ==========

    /// <summary>
    /// Split completed trials into below (good) and above (bad) groups.
    /// Applies constant-liar and MaxAboveTrials subsampling.
    /// </summary>
    private (List<Trial> Below, List<Trial> Above) SplitBelowAbove(
        List<Trial> sortedCompleted, IReadOnlyList<Trial> allTrials)
    {
        // Separate feasible and infeasible trials (based on ConstraintValues)
        var feasible = new List<Trial>();
        var infeasible = new List<Trial>();

        foreach (var trial in sortedCompleted)
        {
            if (IsFeasible(trial))
                feasible.Add(trial);
            else
                infeasible.Add(trial);
        }

        List<Trial> belowTrials, aboveTrials;

        if (feasible.Count >= _config.NStartupTrials)
        {
            // Enough feasible trials: use them for below/above split
            var nBelow = Gamma(feasible.Count);
            belowTrials = feasible.GetRange(0, nBelow);
            aboveTrials = feasible.GetRange(nBelow, feasible.Count - nBelow);

            // All infeasible trials go to above (penalize infeasibility)
            aboveTrials.AddRange(infeasible);
        }
        else
        {
            // Insufficient feasible trials: sort all by violation magnitude
            var allSorted = new List<(double Violation, Trial Trial)>();

            foreach (var trial in sortedCompleted)
            {
                var violation = GetConstraintViolation(trial);
                allSorted.Add((violation, trial));
            }

            allSorted.Sort((a, b) => a.Violation.CompareTo(b.Violation));

            var nBelow = Gamma(allSorted.Count);
            belowTrials = allSorted.GetRange(0, nBelow).Select(x => x.Trial).ToList();
            aboveTrials = allSorted.GetRange(nBelow, allSorted.Count - nBelow).Select(x => x.Trial).ToList();
        }

        // Constant liar: include running trials in above group
        if (_config.ConstantLiar)
        {
            foreach (var trial in allTrials)
            {
                if (trial.State == TrialState.Running)
                    aboveTrials.Add(trial);
            }
        }

        // Subsample above group if it exceeds MaxAboveTrials (caps O(n) LogPdf cost)
        if (_config.MaxAboveTrials > 0 && aboveTrials.Count > _config.MaxAboveTrials)
        {
            // Reservoir sampling to get a uniform subsample
            var sampled = new List<Trial>(_config.MaxAboveTrials);
            for (var i = 0; i < aboveTrials.Count; i++)
            {
                if (i < _config.MaxAboveTrials)
                {
                    sampled.Add(aboveTrials[i]);
                }
                else
                {
                    var j = _rng.Next(i + 1);
                    if (j < _config.MaxAboveTrials)
                        sampled[j] = aboveTrials[i];
                }
            }
            aboveTrials = sampled;
        }

        return (belowTrials, aboveTrials);
    }

    private bool IsFeasible(Trial trial)
    {
        return trial.ConstraintValues == null || trial.ConstraintValues.All(v => v <= 0.0);
    }

    private double GetConstraintViolation(Trial trial)
    {
        if (trial.ConstraintValues == null || trial.ConstraintValues.Length == 0)
            return 0.0;

        return trial.ConstraintValues.Sum(v => Math.Max(0.0, v));
    }

    // ========== BATCH OPTIMIZATION HELPERS ==========

    private sealed record CachedFloatEstimator(
        ParzenEstimator Below, ParzenEstimator Above,
        double Low, double High, bool IsLog);

    private sealed record CachedCatEstimator(
        CategoricalEstimator Below, CategoricalEstimator Above,
        object[] Choices);

    /// <summary>
    /// Pre-build Parzen estimators for a parameter (extract + sort + build — done once per batch).
    /// </summary>
    private object PreBuildEstimator(ParameterRange range, List<Trial> below, List<Trial> above)
    {
        switch (range)
        {
            case FloatRange fr:
            {
                var isLog = fr.Log;
                var low = isLog ? Math.Log(fr.Low) : fr.Low;
                var high = isLog ? Math.Log(fr.High) : fr.High;
                var belowObs = ExtractSortedDoubles(below, fr.Name, isLog);
                var aboveObs = ExtractSortedDoubles(above, fr.Name, isLog);
                return new CachedFloatEstimator(
                    new ParzenEstimator(belowObs, low, high, _config.PriorWeight, _config.ConsiderMagicClip),
                    new ParzenEstimator(aboveObs, low, high, _config.PriorWeight, _config.ConsiderMagicClip),
                    low, high, isLog);
            }
            case IntRange ir:
            {
                var low = (double)ir.Low;
                var high = (double)ir.High;
                var belowObs = ExtractSortedDoubles(below, ir.Name, log: false);
                var aboveObs = ExtractSortedDoubles(above, ir.Name, log: false);
                return new CachedFloatEstimator(
                    new ParzenEstimator(belowObs, low, high, _config.PriorWeight, _config.ConsiderMagicClip),
                    new ParzenEstimator(aboveObs, low, high, _config.PriorWeight, _config.ConsiderMagicClip),
                    low, high, false);
            }
            case CategoricalRange cr:
            {
                var choices = cr.Choices;
                var belowObs = ExtractCategoricalIndices(below, cr.Name, choices);
                var aboveObs = ExtractCategoricalIndices(above, cr.Name, choices);
                return new CachedCatEstimator(
                    new CategoricalEstimator(belowObs, choices.Length, _config.PriorWeight),
                    new CategoricalEstimator(aboveObs, choices.Length, _config.PriorWeight),
                    choices);
            }
            default:
                throw new ArgumentException($"Unknown range type: {range.GetType().Name}");
        }
    }

    /// <summary>
    /// Sample a parameter value from pre-built cached estimators (fast path — no extraction or sorting).
    /// </summary>
    private object SampleFromCached(ParameterRange range, object cached)
    {
        switch (cached)
        {
            case CachedFloatEstimator fe when range is FloatRange fr:
            {
                var result = SampleFromEstimatorPair(fe.Below, fe.Above);
                return fe.IsLog ? Math.Clamp(Math.Exp(result), fr.Low, fr.High) : Math.Clamp(result, fr.Low, fr.High);
            }
            case CachedFloatEstimator fe when range is IntRange ir:
            {
                var result = SampleFromEstimatorPair(fe.Below, fe.Above);
                var rounded = (int)Math.Round(Math.Clamp(result, fe.Low, fe.High));
                if (ir.Step > 1)
                    rounded = ir.Low + (int)Math.Round((double)(rounded - ir.Low) / ir.Step) * ir.Step;
                return Math.Clamp(rounded, ir.Low, ir.High);
            }
            case CachedCatEstimator ce:
            {
                var bestIdx = 0;
                var bestEi = double.NegativeInfinity;
                for (var i = 0; i < _config.NEiCandidates; i++)
                {
                    var candidate = ce.Below.Sample(_rng);
                    var ei = ce.Below.LogPdf(candidate) - ce.Above.LogPdf(candidate);
                    if (ei > bestEi) { bestEi = ei; bestIdx = candidate; }
                }
                return ce.Choices[bestIdx];
            }
            default:
                throw new ArgumentException($"Unexpected cached estimator type: {cached.GetType().Name}");
        }
    }

    /// <summary>
    /// EI maximization using pre-built estimator pair (shared by SampleFromCached).
    /// </summary>
    private double SampleFromEstimatorPair(ParzenEstimator below, ParzenEstimator above)
    {
        var candidates = below.Sample(_rng, _config.NEiCandidates);
        var logL = below.LogPdf(candidates);
        var logG = above.LogPdf(candidates);

        var bestIdx = 0;
        var bestEi = double.NegativeInfinity;
        for (var i = 0; i < candidates.Length; i++)
        {
            var ei = logL[i] - logG[i];
            if (ei > bestEi) { bestEi = ei; bestIdx = i; }
        }
        return candidates[bestIdx];
    }

    /// <summary>
    /// Gamma function: determines how many trials go in the "below" (good) group.
    /// Default: min(ceil(0.1 * n), 25) — Bergstra et al. (2011) / Optuna.
    /// </summary>
    private static int Gamma(int completedCount)
    {
        return Math.Min((int)Math.Ceiling(0.1 * completedCount), 25);
    }

    private static List<Trial> GetCompletedTrials(IReadOnlyList<Trial> trials)
    {
        var completed = new List<Trial>();
        foreach (var trial in trials)
        {
            if (trial.State == TrialState.Complete && trial.Value.HasValue)
                completed.Add(trial);
        }
        return completed;
    }

    private static double[] ExtractSortedDoubles(List<Trial> trials, string paramName, bool log)
    {
        var values = new double[trials.Count];
        var count = 0;

        foreach (var trial in trials)
        {
            if (trial.Parameters.TryGetValue(paramName, out var val))
            {
                var d = Convert.ToDouble(val);
                values[count++] = log ? Math.Log(d) : d;
            }
        }

        if (count < values.Length)
            Array.Resize(ref values, count);

        Array.Sort(values);
        return values;
    }

    private static Span<int> ExtractCategoricalIndices(List<Trial> trials, string paramName, object[] choices)
    {
        var indices = new int[trials.Count];
        var count = 0;

        foreach (var trial in trials)
        {
            if (trial.Parameters.TryGetValue(paramName, out var val))
            {
                var idx = Array.IndexOf(choices, val);
                if (idx >= 0)
                    indices[count++] = idx;
            }
        }

        return indices.AsSpan(0, count);
    }
}
