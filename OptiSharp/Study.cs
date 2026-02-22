using OptiSharp.Models;
using OptiSharp.MultiObjective;
using OptiSharp.Pruning;
using OptiSharp.Samplers.Tpe;
using OptiSharp.Storage;

namespace OptiSharp;

/// <summary>
/// Thread-safe optimization study coordinator.
/// Manages the ask/tell lifecycle between sampler and external evaluators.
/// </summary>
public sealed class Study : IDisposable
{
    private readonly ISampler _sampler;
    private readonly SearchSpace _searchSpace;
    private readonly IPruner _pruner;
    private readonly List<Trial> _trials = [];
    private readonly Dictionary<int, Trial> _trialIndex = [];
    private readonly SemaphoreSlim _lock = new(1, 1);
    private readonly StudyDirection[]? _directions;
    private Func<Trial, double[]>? _constraintFunc;
    private int _nextNumber;

    public Study(string name, ISampler sampler, SearchSpace searchSpace, StudyDirection direction, IPruner? pruner = null)
    {
        Name = name;
        _sampler = sampler;
        _searchSpace = searchSpace;
        Direction = direction;
        _pruner = pruner ?? new NopPruner();
        _directions = null;
    }

    public Study(string name, ISampler sampler, SearchSpace searchSpace, StudyDirection[] directions, IPruner? pruner = null)
    {
        if (directions.Length == 0)
            throw new ArgumentException("directions must not be empty", nameof(directions));

        Name = name;
        _sampler = sampler;
        _searchSpace = searchSpace;
        Direction = directions[0]; // For backward compatibility
        _pruner = pruner ?? new NopPruner();
        _directions = directions;
    }

    public string Name { get; }
    public StudyDirection Direction { get; }
    public bool IsMultiObjective => _directions != null;

    public IReadOnlyList<Trial> Trials => WithLock(() => _trials.ToArray());

    public Trial? BestTrial => WithLock(FindBest);

    /// <summary>
    /// Returns the Pareto front for multi-objective studies.
    /// For single-objective studies, returns the best trial in a list.
    /// O(n^2 * m) complexity; not recommended for >5k trials.
    /// </summary>
    public IReadOnlyList<Trial> ParetoFront => WithLock(() =>
    {
        if (!IsMultiObjective)
        {
            var best = FindBest();
            return best != null ? [best] : [];
        }

        return ParetoUtils.ComputeParetoFront(_trials, _directions!);
    });

    private Trial? FindBest()
    {
        Trial? best = null;
        foreach (var trial in _trials)
        {
            if (trial.State != TrialState.Complete || !trial.Value.HasValue)
                continue;

            if (best is null)
            {
                best = trial;
                continue;
            }

            var isBetter = Direction == StudyDirection.Minimize
                ? trial.Value.Value < best.Value!.Value
                : trial.Value.Value > best.Value!.Value;

            if (isBetter)
                best = trial;
        }
        return best;
    }

    /// <summary>
    /// Ask the sampler for a new trial with suggested parameters.
    /// </summary>
    public Trial Ask() => WithLock(AskCore);

    /// <summary>
    /// Ask for multiple trials at once. Single lock acquisition.
    /// Uses batch-optimized sampling when the sampler supports it (TpeSampler).
    /// </summary>
    public IReadOnlyList<Trial> AskBatch(int count)
    {
        if (count <= 0) return [];

        return WithLock(() =>
        {
            // Batch-optimized path: pre-build estimators once, sample N times
            if (_sampler is TpeSampler tpe)
            {
                var paramsBatch = tpe.SampleBatch(_trials, Direction, _searchSpace, count);
                var trials = new Trial[count];
                for (var i = 0; i < count; i++)
                {
                    var trial = new Trial(_nextNumber++, paramsBatch[i]);
                    _trials.Add(trial);
                    _trialIndex[trial.Number] = trial;
                    trials[i] = trial;
                }
                return trials;
            }

            // Fallback: call AskCore individually
            var fallbackTrials = new Trial[count];
            for (var i = 0; i < count; i++)
                fallbackTrials[i] = AskCore();
            return fallbackTrials;
        });
    }

    /// <summary>
    /// Report the result of a completed trial (single-objective).
    /// </summary>
    public void Tell(int trialNumber, double value) => WithLock(() =>
    {
        if (!_trialIndex.TryGetValue(trialNumber, out var trial))
            throw new ArgumentException($"Trial {trialNumber} not found");

        trial.Value = value;
        if (_constraintFunc != null)
            trial.ConstraintValues = _constraintFunc(trial);
        trial.State = TrialState.Complete;
    });

    /// <summary>
    /// Report the results of a completed trial (multi-objective).
    /// Also sets trial.Value = values[0] for backward compatibility.
    /// </summary>
    public void Tell(int trialNumber, double[] values) => WithLock(() =>
    {
        if (!_trialIndex.TryGetValue(trialNumber, out var trial))
            throw new ArgumentException($"Trial {trialNumber} not found");

        trial.Values = values;
        trial.Value = values[0]; // Backward compatibility
        if (_constraintFunc != null)
            trial.ConstraintValues = _constraintFunc(trial);
        trial.State = TrialState.Complete;
    });

    /// <summary>
    /// Report a failed or pruned trial.
    /// </summary>
    public void Tell(int trialNumber, TrialState state)
    {
        if (state == TrialState.Complete)
            throw new ArgumentException("Use Tell(trialNumber, value) for complete trials");

        if (state == TrialState.Running)
            throw new ArgumentException("Cannot report Running state");

        WithLock(() =>
        {
            if (!_trialIndex.TryGetValue(trialNumber, out var trial))
                throw new ArgumentException($"Trial {trialNumber} not found");

            trial.State = state;
        });
    }

    /// <summary>
    /// Check if a trial should be pruned based on its current performance.
    /// </summary>
    public bool ShouldPrune(Trial trial) => WithLock(() => _pruner.ShouldPrune(trial, _trials));

    /// <summary>
    /// Set a constraint function that evaluates feasibility.
    /// The function should return constraint values where all values <= 0 means feasible.
    /// </summary>
    public void SetConstraintFunc(Func<Trial, double[]> func) => WithLock(() => _constraintFunc = func);

    /// <summary>
    /// Check if a trial is feasible (all constraint values <= 0, or no constraints).
    /// </summary>
    public bool IsFeasible(Trial trial) => trial.ConstraintValues == null || trial.ConstraintValues.All(v => v <= 0.0);

    /// <summary>
    /// Report results for multiple trials at once (single-objective). Single lock acquisition.
    /// Unknown trial numbers are silently skipped (batch-tolerant semantics).
    /// </summary>
    public void TellBatch(IReadOnlyList<TrialResult> results)
    {
        if (results.Count == 0) return;

        WithLock(() =>
        {
            foreach (var result in results)
            {
                if (!_trialIndex.TryGetValue(result.TrialNumber, out var trial))
                    continue;

                trial.State = result.State;
                trial.Value = result.Value;
            }
        });
    }

    /// <summary>
    /// Report results for multiple trials at once (multi-objective). Single lock acquisition.
    /// Unknown trial numbers are silently skipped (batch-tolerant semantics).
    /// </summary>
    public void TellBatch(IReadOnlyList<MoTrialResult> results)
    {
        if (results.Count == 0) return;

        WithLock(() =>
        {
            foreach (var result in results)
            {
                if (!_trialIndex.TryGetValue(result.TrialNumber, out var trial))
                    continue;

                trial.State = result.State;
                trial.Values = result.Values;
                if (result.Values != null && result.Values.Length > 0)
                    trial.Value = result.Values[0]; // Backward compatibility
            }
        });
    }

    /// <summary>
    /// Pre-populate the study with warm-start trials from a previous optimization.
    /// Only Complete trials are imported; their parameters, values, and constraints are copied.
    /// Trial numbers are assigned sequentially starting from _nextNumber.
    /// </summary>
    internal void PrePopulateWarmTrials(IEnumerable<Trial> warmTrials) => WithLock(() =>
    {
        foreach (var trial in warmTrials)
        {
            if (trial.State != TrialState.Complete)
                continue;

            var newTrial = new Trial(_nextNumber++, new Dictionary<string, object>(trial.Parameters))
            {
                Value = trial.Value,
                Values = trial.Values,
                ConstraintValues = trial.ConstraintValues,
                State = TrialState.Complete
            };

            // Copy intermediate values
            foreach (var (step, value) in trial.IntermediateValues)
                newTrial.Report(value, step);

            _trials.Add(newTrial);
            _trialIndex[newTrial.Number] = newTrial;
        }
    });

    private Trial AskCore()
    {
        var parameters = IsMultiObjective
            ? _sampler.SampleMultiObjective(_trials, _directions!, _searchSpace)
            : _sampler.Sample(_trials, Direction, _searchSpace);

        var trial = new Trial(_nextNumber++, parameters);
        _trials.Add(trial);
        _trialIndex[trial.Number] = trial;
        return trial;
    }

    private T WithLock<T>(Func<T> action)
    {
        _lock.Wait();
        try { return action(); }
        finally { _lock.Release(); }
    }

    private void WithLock(Action action)
    {
        _lock.Wait();
        try { action(); }
        finally { _lock.Release(); }
    }

    /// <summary>
    /// Save the study to a JSON file.
    /// Only Complete and Pruned trials are saved.
    /// </summary>
    public void Save(string filePath) => WithLock(() =>
    {
        var json = StudySerializer.Serialize(Name, Direction, _directions, _trials);
        System.IO.File.WriteAllText(filePath, json);
    });

    /// <summary>
    /// Load a study from a JSON file.
    /// Reconstructs the study state without evaluating trials.
    /// </summary>
    public static Study Load(
        string filePath,
        SearchSpace searchSpace,
        ISampler sampler,
        StudyDirection direction = StudyDirection.Minimize,
        IPruner? pruner = null)
    {
        var json = System.IO.File.ReadAllText(filePath);
        var (name, directions, trials) = StudySerializer.Deserialize(json, searchSpace);

        Study study = directions != null
            ? new Study(name, sampler, searchSpace, directions, pruner)
            : new Study(name, sampler, searchSpace, direction, pruner);

        study.PrePopulateWarmTrials(trials);
        return study;
    }

    public void Dispose()
    {
        (_sampler as IDisposable)?.Dispose();
        _lock.Dispose();
    }
}
