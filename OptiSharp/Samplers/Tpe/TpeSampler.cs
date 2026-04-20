using OptiSharp.Models;
using OptiSharp.Samplers.Sobol;

namespace OptiSharp.Samplers.Tpe;

/// <summary>
/// Tree-structured Parzen Estimator sampler.
/// Fits two kernel density estimates (good/bad trials) and maximizes EI = l(x)/g(x).
/// </summary>
public sealed class TpeSampler : ISampler
{
    private readonly TpeSamplerConfig _config;
    private readonly RandomSampler _randomFallback;
    private readonly SobolSampler _sobolFallback;
    private readonly Random _rng;

    public TpeSampler(TpeSamplerConfig? config = null)
    {
        _config = config ?? new TpeSamplerConfig();
        _rng = _config.Seed.HasValue ? new Random(_config.Seed.Value) : new Random();
        _randomFallback = new RandomSampler(_rng);
        _sobolFallback = new SobolSampler(new SobolSamplerConfig { Seed = _config.Seed });
    }

    public Dictionary<string, object> Sample(
        IReadOnlyList<Trial> trials,
        StudyDirection direction,
        SearchSpace searchSpace)
    {
        // Get completed + pruned trials (pruned trials with intermediate values contribute to KDE)
        var evaluatedTrials = GetEvaluatedTrials(trials);

        // Startup phase: use Sobol or random sampling
        if (evaluatedTrials.Count < _config.NStartupTrials)
        {
            return _config.SobolStartup
                ? _sobolFallback.Sample(trials, direction, searchSpace)
                : _randomFallback.Sample(trials, direction, searchSpace);
        }

        // Sort by effective value: best first
        // For complete trials: use trial.Value
        // For pruned trials: use last intermediate value; if none, score as worst (infinity)
        evaluatedTrials.Sort((a, b) =>
        {
            var valA = GetEffectiveValue(a) ?? double.PositiveInfinity;
            var valB = GetEffectiveValue(b) ?? double.PositiveInfinity;
            return direction == StudyDirection.Minimize
                ? valA.CompareTo(valB)
                : valB.CompareTo(valA);
        });

        // Split into below (good) and above (bad) groups
        var (belowTrials, aboveTrials) = SplitBelowAbove(evaluatedTrials, trials, searchSpace.Count);

        // Try multivariate TPE if enabled and feasible (min 10 dims — below that univariate is more reliable)
        if (_config.Multivariate && searchSpace.Count >= 10)
        {
            var multiResult = TryMultivariateSampling(belowTrials, aboveTrials, searchSpace);
            if (multiResult != null)
                return multiResult;
            // Fall back to univariate if multivariate fails
        }

        // Sample each parameter independently (univariate fallback or default)
        var parameters = new Dictionary<string, object>(searchSpace.Count);
        foreach (var range in searchSpace)
        {
            parameters[range.Name] = SampleParameter(range, belowTrials, aboveTrials, searchSpace.Count);
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
        var evaluated = GetEvaluatedTrials(trials);

        // Startup phase: Sobol or random sampling for all
        if (evaluated.Count < _config.NStartupTrials)
        {
            var results = new Dictionary<string, object>[count];
            for (var i = 0; i < count; i++)
            {
                results[i] = _config.SobolStartup
                    ? _sobolFallback.Sample(trials, direction, searchSpace)
                    : _randomFallback.Sample(trials, direction, searchSpace);
            }
            return results;
        }

        // Sort once
        evaluated.Sort((a, b) =>
        {
            var valA = GetEffectiveValue(a) ?? double.PositiveInfinity;
            var valB = GetEffectiveValue(b) ?? double.PositiveInfinity;
            return direction == StudyDirection.Minimize
                ? valA.CompareTo(valB)
                : valB.CompareTo(valA);
        });

        // Split once
        var (belowTrials, aboveTrials) = SplitBelowAbove(evaluated, trials, searchSpace.Count);

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
                parameters[range.Name] = SampleFromCached(range, paramEstimators[range.Name], searchSpace.Count);
            }
            batch[i] = parameters;
        }

        return batch;
    }

    private object SampleParameter(ParameterRange range, List<Trial> below, List<Trial> above, int nDims)
    {
        return range switch
        {
            FloatRange fr => SampleFloat(fr, below, above, nDims),
            IntRange ir => SampleInt(ir, below, above, nDims),
            CategoricalRange cr => SampleCategorical(cr, below, above),
            _ => throw new ArgumentException($"Unknown range type: {range.GetType().Name}")
        };
    }

    private double SampleFloat(FloatRange range, List<Trial> below, List<Trial> above, int nDims)
    {
        var isLog = range.Log;
        var low = isLog ? Math.Log(range.Low) : range.Low;
        var high = isLog ? Math.Log(range.High) : range.High;

        var belowObs = ExtractSortedDoubles(below, range.Name, isLog);
        var aboveObs = ExtractSortedDoubles(above, range.Name, isLog);

        var result = SampleWithEi(belowObs, aboveObs, low, high, nDims);
        return isLog ? Math.Clamp(Math.Exp(result), range.Low, range.High) : Math.Clamp(result, range.Low, range.High);
    }

    private int SampleInt(IntRange range, List<Trial> below, List<Trial> above, int nDims)
    {
        var low = (double)range.Low;
        var high = (double)range.High;

        var belowObs = ExtractSortedDoubles(below, range.Name, log: false);
        var aboveObs = ExtractSortedDoubles(above, range.Name, log: false);

        var result = SampleWithEi(belowObs, aboveObs, low, high, nDims);
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
    private double SampleWithEi(double[] belowObs, double[] aboveObs, double low, double high, int nDims)
    {
        var estBelow = new ParzenEstimator(belowObs, low, high, _config.PriorWeight, _config.ConsiderMagicClip, isBelowEstimator: true);
        var estAbove = new ParzenEstimator(aboveObs, low, high, _config.PriorWeight, _config.ConsiderMagicClip, isBelowEstimator: false);

        // Use dimension-aware number of EI candidates
        var nEi = EffectiveEiCandidates(nDims);

        // Sample candidates from below (good) estimator
        var candidates = estBelow.Sample(_rng, nEi);

        // Compute EI = log l(x) - log g(x)
        var logL = estBelow.LogPdf(candidates);
        var logG = estAbove.LogPdf(candidates);

        // Pick candidate with maximum EI
        var bestIdx = 0;
        var bestEi = double.NegativeInfinity;
        for (var i = 0; i < candidates.Length; i++)
        {
            var ei = logL[i] - logG[i];

            // Skip NaN scores
            if (double.IsNaN(ei))
                continue;

            // If x never seen in bad trials (logG = -∞), always prefer it
            if (double.IsNegativeInfinity(logG[i]))
                ei = double.MaxValue;

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
        List<Trial> sortedEvaluated, IReadOnlyList<Trial> allTrials, int nDims)
    {
        // Separate by feasibility and state — mirrors Optuna's _split_trials logic exactly.
        var completeTrials = new List<Trial>();
        var prunedTrials = new List<Trial>();
        var infeasible = new List<Trial>();

        foreach (var trial in sortedEvaluated)
        {
            if (!IsFeasible(trial))
                infeasible.Add(trial);
            else if (trial.State == TrialState.Complete)
                completeTrials.Add(trial);
            else if (trial.State == TrialState.Pruned)
                prunedTrials.Add(trial);
        }

        // completeTrials already sorted best-first (by objective value) from the caller.
        // prunedTrials: sort by (-lastStep, lastValue) ascending — Optuna's _get_pruned_trial_score.
        // Higher last step = survived longer pruning = better signal quality (comes first).
        // Precompute scores to avoid repeated ConcurrentDictionary traversal inside comparator.
        if (prunedTrials.Count > 1)
        {
            var prunedScores = new (double NegStep, double Value)[prunedTrials.Count];
            for (var i = 0; i < prunedTrials.Count; i++)
                prunedScores[i] = GetPrunedScore(prunedTrials[i]);

            // Index-sort using precomputed scores
            var indices = new int[prunedTrials.Count];
            for (var i = 0; i < indices.Length; i++) indices[i] = i;
            Array.Sort(indices, (a, b) =>
            {
                var cmp = prunedScores[a].NegStep.CompareTo(prunedScores[b].NegStep);
                return cmp != 0 ? cmp : prunedScores[a].Value.CompareTo(prunedScores[b].Value);
            });

            var sortedPruned = new List<Trial>(prunedTrials.Count);
            foreach (var idx in indices) sortedPruned.Add(prunedTrials[idx]);
            prunedTrials = sortedPruned;
        }

        List<Trial> belowTrials, aboveTrials;

        var totalFeasible = completeTrials.Count + prunedTrials.Count;
        if (totalFeasible >= _config.NStartupTrials)
        {
            // Gamma based on total feasible count (complete + pruned), matching Optuna.
            var nBelow = Gamma(totalFeasible, nDims);

            // Complete trials fill below FIRST (clean signal has priority).
            // Only use pruned to fill remaining slots if complete count < nBelow.
            var belowComplete = completeTrials.GetRange(0, Math.Min(nBelow, completeTrials.Count));
            var remaining = nBelow - belowComplete.Count;
            var belowPruned = prunedTrials.GetRange(0, Math.Min(remaining, prunedTrials.Count));

            belowTrials = [.. belowComplete, .. belowPruned];
            aboveTrials = [
                .. completeTrials.GetRange(belowComplete.Count, completeTrials.Count - belowComplete.Count),
                .. prunedTrials.GetRange(belowPruned.Count, prunedTrials.Count - belowPruned.Count),
                .. infeasible
            ];
        }
        else
        {
            // Insufficient feasible trials: sort all by violation magnitude
            var allSorted = sortedEvaluated
                .Select(t => (Violation: GetConstraintViolation(t), Trial: t))
                .OrderBy(x => x.Violation)
                .ToList();

            var nBelow = Gamma(allSorted.Count, nDims);
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

        // Subsample above group if it exceeds effective cap (caps O(n) LogPdf cost)
        // Dynamic cap: max(25, n_below) — matches Optuna behavior exactly
        var effectiveMaxAbove = _config.MaxAboveTrials > 0
            ? _config.MaxAboveTrials
            : Math.Max(25, belowTrials.Count);

        if (aboveTrials.Count > effectiveMaxAbove)
        {
            // Reservoir sampling to get a uniform subsample
            var sampled = new List<Trial>(effectiveMaxAbove);
            for (var i = 0; i < aboveTrials.Count; i++)
            {
                if (i < effectiveMaxAbove)
                {
                    sampled.Add(aboveTrials[i]);
                }
                else
                {
                    var j = _rng.Next(i + 1);
                    if (j < effectiveMaxAbove)
                        sampled[j] = aboveTrials[i];
                }
            }
            aboveTrials = sampled;
        }

        return (belowTrials, aboveTrials);
    }

    /// <summary>
    /// Scoring for pruned trials: (-lastStep, lastIntermediateValue).
    /// Matches Optuna's _get_pruned_trial_score — higher step (survived longer) comes first.
    /// </summary>
    private static (double NegStep, double Value) GetPrunedScore(Trial trial)
    {
        if (trial.IntermediateValues.Count == 0)
            return (1.0, 0.0); // no data — treat as worst

        var lastStep = trial.IntermediateValues.Keys.Max();
        var lastValue = trial.IntermediateValues[lastStep];
        return double.IsNaN(lastValue) ? (-lastStep, double.PositiveInfinity) : (-lastStep, lastValue);
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
                    new ParzenEstimator(belowObs, low, high, _config.PriorWeight, _config.ConsiderMagicClip, isBelowEstimator: true),
                    new ParzenEstimator(aboveObs, low, high, _config.PriorWeight, _config.ConsiderMagicClip, isBelowEstimator: false),
                    low, high, isLog);
            }
            case IntRange ir:
            {
                var low = (double)ir.Low;
                var high = (double)ir.High;
                var belowObs = ExtractSortedDoubles(below, ir.Name, log: false);
                var aboveObs = ExtractSortedDoubles(above, ir.Name, log: false);
                return new CachedFloatEstimator(
                    new ParzenEstimator(belowObs, low, high, _config.PriorWeight, _config.ConsiderMagicClip, isBelowEstimator: true),
                    new ParzenEstimator(aboveObs, low, high, _config.PriorWeight, _config.ConsiderMagicClip, isBelowEstimator: false),
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
    private object SampleFromCached(ParameterRange range, object cached, int nDims)
    {
        switch (cached)
        {
            case CachedFloatEstimator fe when range is FloatRange fr:
            {
                var result = SampleFromEstimatorPair(fe.Below, fe.Above, nDims);
                return fe.IsLog ? Math.Clamp(Math.Exp(result), fr.Low, fr.High) : Math.Clamp(result, fr.Low, fr.High);
            }
            case CachedFloatEstimator fe when range is IntRange ir:
            {
                var result = SampleFromEstimatorPair(fe.Below, fe.Above, nDims);
                var rounded = (int)Math.Round(Math.Clamp(result, fe.Low, fe.High));
                if (ir.Step > 1)
                    rounded = ir.Low + (int)Math.Round((double)(rounded - ir.Low) / ir.Step) * ir.Step;
                return Math.Clamp(rounded, ir.Low, ir.High);
            }
            case CachedCatEstimator ce:
            {
                var nEi = EffectiveEiCandidates(nDims);
                var bestIdx = 0;
                var bestEi = double.NegativeInfinity;
                for (var i = 0; i < nEi; i++)
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
    private double SampleFromEstimatorPair(ParzenEstimator below, ParzenEstimator above, int nDims)
    {
        var nEi = EffectiveEiCandidates(nDims);
        var candidates = below.Sample(_rng, nEi);
        var logL = below.LogPdf(candidates);
        var logG = above.LogPdf(candidates);

        var bestIdx = 0;
        var bestEi = double.NegativeInfinity;
        for (var i = 0; i < candidates.Length; i++)
        {
            var ei = logL[i] - logG[i];

            // Skip NaN scores
            if (double.IsNaN(ei))
                continue;

            // If x never seen in bad trials (logG = -∞), always prefer it
            if (double.IsNegativeInfinity(logG[i]))
                ei = double.MaxValue;

            if (ei > bestEi) { bestEi = ei; bestIdx = i; }
        }
        return candidates[bestIdx];
    }

    /// <summary>
    /// Attempt multivariate Gaussian TPE: sample joint vectors from below multivariate KDE,
    /// score via logL(x) - Σ logG_d(x_d), return best vector.
    /// Returns null if multivariate build fails (fall back to univariate).
    /// </summary>
    private Dictionary<string, object>? TryMultivariateSampling(List<Trial> belowTrials, List<Trial> aboveTrials, SearchSpace searchSpace)
    {
        try
        {
            // Multivariate KDE only handles continuous/int dims — fall back if any categorical is present
            if (searchSpace.Any(r => r is CategoricalRange))
                return null;

            // Extract parameter vectors from below trials
            var belowVectors = new double[belowTrials.Count][];
            var lows = new double[searchSpace.Count];
            var highs = new double[searchSpace.Count];

            for (var i = 0; i < searchSpace.Count; i++)
            {
                var range = searchSpace[i];
                lows[i] = range switch
                {
                    FloatRange fr => fr.Log ? Math.Log(fr.Low) : fr.Low,
                    IntRange ir => (double)ir.Low,
                    CategoricalRange cr => 0.0, // Not supported in multivariate
                    _ => 0.0
                };
                highs[i] = range switch
                {
                    FloatRange fr => fr.Log ? Math.Log(fr.High) : fr.High,
                    IntRange ir => (double)ir.High,
                    CategoricalRange cr => cr.Choices.Length - 1.0,
                    _ => 1.0
                };
            }

            // Build trial vectors (skip categorical for now - multivariate only handles continuous/int)
            for (var t = 0; t < belowTrials.Count; t++)
            {
                var trial = belowTrials[t];
                var vec = new double[searchSpace.Count];
                for (var i = 0; i < searchSpace.Count; i++)
                {
                    var range = searchSpace[i];
                    if (!trial.Parameters.TryGetValue(range.Name, out var val))
                        val = 0.0;

                    vec[i] = range switch
                    {
                        FloatRange fr => fr.Log ? Math.Log((double)val) : (double)val,
                        IntRange ir => (double)(int)val,
                        CategoricalRange => 0.0, // Skip for now
                        _ => 0.0
                    };
                }
                belowVectors[t] = vec;
            }

            // Build multivariate KDE from below vectors
            var kde = new MultivariateKde(belowVectors, lows, highs);

            // Build univariate above estimators for each parameter
            var aboveEstimators = new ParzenEstimator[searchSpace.Count];
            for (var i = 0; i < searchSpace.Count; i++)
            {
                var range = searchSpace[i];
                var aboveObs = ExtractSortedDoubles(aboveTrials, range.Name,
                    range is FloatRange fr && fr.Log);
                aboveEstimators[i] = new ParzenEstimator(aboveObs, lows[i], highs[i],
                    _config.PriorWeight, _config.ConsiderMagicClip, isBelowEstimator: false);
            }

            // Sample EI candidates from multivariate below
            var nEi = EffectiveEiCandidates(searchSpace.Count);
            var bestVec = belowVectors[0]; // Fallback
            var bestEi = double.NegativeInfinity;

            for (var i = 0; i < nEi; i++)
            {
                var candidate = kde.Sample(_rng);

                // Score: logL(x) from multivariate - Σ logG_d(x_d) from univariate
                var logL = kde.LogPdf(candidate);
                var logG = 0.0;
                for (var d = 0; d < searchSpace.Count; d++)
                    logG += aboveEstimators[d].LogPdf(new[] { candidate[d] })[0];

                var ei = logL - logG;

                // Skip NaN scores
                if (double.IsNaN(ei))
                    continue;

                // If logG = -∞, always prefer
                if (double.IsNegativeInfinity(logG))
                    ei = double.MaxValue;

                if (ei > bestEi)
                {
                    bestEi = ei;
                    bestVec = candidate;
                }
            }

            // Convert best vector back to parameter dict
            var result = new Dictionary<string, object>(searchSpace.Count);
            for (var i = 0; i < searchSpace.Count; i++)
            {
                var range = searchSpace[i];
                var val = bestVec[i];

                result[range.Name] = range switch
                {
                    FloatRange fr => fr.Log
                        ? Math.Clamp(Math.Exp(val), fr.Low, fr.High)
                        : Math.Clamp(val, fr.Low, fr.High),
                    IntRange ir =>
                        Math.Clamp((int)Math.Round(val), ir.Low, ir.High),
                    CategoricalRange cr =>
                        cr.Choices[Math.Clamp((int)Math.Round(val), 0, cr.Choices.Length - 1)],
                    _ => val
                };
            }

            return result;
        }
        catch
        {
            // Fall back to univariate if anything fails
            return null;
        }
    }

    /// <summary>
    /// Gamma function: determines how many trials go in the "below" (good) group.
    /// Formula: min(ceil(0.1 * n), cap) — Optuna's default formula (linear, not quadratic).
    /// This matches Optuna's default_gamma() implementation exactly.
    /// </summary>
    private int Gamma(int completedCount, int nDims)
    {
        var cap = nDims >= 100 ? 50 : 25;
        return Math.Min((int)Math.Ceiling(0.1 * completedCount), cap);
    }

    private int EffectiveEiCandidates(int nDims)
    {
        var scaled = (int)Math.Ceiling(_config.NEiCandidates * Math.Sqrt(nDims / 10.0));
        return Math.Min(scaled, 256);
    }

    /// <summary>
    /// Get trials that have evaluation data: completed trials and pruned trials with intermediate values.
    /// This includes trials that couldn't complete but reported progress via Report().
    /// </summary>
    private static List<Trial> GetEvaluatedTrials(IReadOnlyList<Trial> trials)
    {
        var evaluated = new List<Trial>();
        foreach (var trial in trials)
        {
            if (trial.State == TrialState.Complete && trial.Value.HasValue)
            {
                // Complete trial with final value
                evaluated.Add(trial);
            }
            else if (trial.State == TrialState.Pruned && trial.IntermediateValues.Count > 0)
            {
                // Pruned trial with intermediate values reported
                evaluated.Add(trial);
            }
        }
        return evaluated;
    }

    /// <summary>
    /// Get the effective objective value for a trial.
    /// For complete trials: return trial.Value.
    /// For pruned trials: return the value at the highest step number.
    /// Note: IntermediateValues is a ConcurrentDictionary — iteration order is NOT insertion order,
    /// so LastOrDefault() is wrong. We must find the key with the maximum step explicitly.
    /// </summary>
    private static double? GetEffectiveValue(Trial trial)
    {
        if (trial.Value.HasValue)
            return trial.Value;

        if (trial.IntermediateValues.Count > 0)
        {
            // Find value at highest step (ConcurrentDictionary iteration order is not guaranteed)
            var maxStep = -1;
            var maxVal = 0.0;
            foreach (var kv in trial.IntermediateValues)
            {
                if (kv.Key > maxStep) { maxStep = kv.Key; maxVal = kv.Value; }
            }
            return maxVal;
        }

        return null;
    }

    private static double[] ExtractSortedDoubles(List<Trial> trials, string paramName, bool log)
    {
        // First pass: count valid entries to avoid Array.Resize
        var count = 0;
        foreach (var trial in trials)
            if (trial.Parameters.ContainsKey(paramName)) count++;

        // Second pass: fill exact-sized array
        var values = new double[count];
        var idx = 0;
        foreach (var trial in trials)
        {
            if (trial.Parameters.TryGetValue(paramName, out var val))
            {
                var d = Convert.ToDouble(val);
                values[idx++] = log ? Math.Log(d) : d;
            }
        }

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
