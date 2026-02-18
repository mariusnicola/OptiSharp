using OptiSharp.Models;
using OptiSharp.Samplers.Tpe;

namespace OptiSharp;

/// <summary>
/// Thread-safe optimization study coordinator.
/// Manages the ask/tell lifecycle between sampler and external evaluators.
/// </summary>
public sealed class Study : IDisposable
{
    private readonly ISampler _sampler;
    private readonly SearchSpace _searchSpace;
    private readonly List<Trial> _trials = [];
    private readonly Dictionary<int, Trial> _trialIndex = [];
    private readonly SemaphoreSlim _lock = new(1, 1);
    private int _nextNumber;

    public Study(string name, ISampler sampler, SearchSpace searchSpace, StudyDirection direction)
    {
        Name = name;
        _sampler = sampler;
        _searchSpace = searchSpace;
        Direction = direction;
    }

    public string Name { get; }
    public StudyDirection Direction { get; }

    public IReadOnlyList<Trial> Trials => WithLock(() => _trials.ToArray());

    public Trial? BestTrial => WithLock(FindBest);

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
    /// Report the result of a completed trial.
    /// </summary>
    public void Tell(int trialNumber, double value) => WithLock(() =>
    {
        if (!_trialIndex.TryGetValue(trialNumber, out var trial))
            throw new ArgumentException($"Trial {trialNumber} not found");

        trial.Value = value;
        trial.State = TrialState.Complete;
    });

    /// <summary>
    /// Report a failed trial.
    /// </summary>
    public void Tell(int trialNumber, TrialState state)
    {
        if (state == TrialState.Complete)
            throw new ArgumentException("Use Tell(trialNumber, value) for complete trials");

        WithLock(() =>
        {
            if (!_trialIndex.TryGetValue(trialNumber, out var trial))
                throw new ArgumentException($"Trial {trialNumber} not found");

            trial.State = state;
        });
    }

    /// <summary>
    /// Report results for multiple trials at once. Single lock acquisition.
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

    private Trial AskCore()
    {
        var parameters = _sampler.Sample(_trials, Direction, _searchSpace);
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

    public void Dispose()
    {
        (_sampler as IDisposable)?.Dispose();
        _lock.Dispose();
    }
}
