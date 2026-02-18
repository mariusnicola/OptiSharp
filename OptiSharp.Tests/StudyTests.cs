using OptiSharp.Models;
using OptiSharp.Samplers.Tpe;

namespace OptiSharp.Tests;

public sealed class StudyTests
{
    private static readonly SearchSpace SimpleSpace = new([
        new FloatRange("x", 0, 10)
    ]);

    [Fact]
    public void Ask_ReturnsRunningTrial()
    {
        using var study = Optimizer.CreateStudy("test", SimpleSpace,
            config: new TpeSamplerConfig { Seed = 42 });

        var trial = study.Ask();

        Assert.Equal(TrialState.Running, trial.State);
        Assert.Null(trial.Value);
        Assert.True(trial.Parameters.ContainsKey("x"));
    }

    [Fact]
    public void Tell_CompleteTrial_UpdatesState()
    {
        using var study = Optimizer.CreateStudy("test", SimpleSpace,
            config: new TpeSamplerConfig { Seed = 42 });

        var trial = study.Ask();
        study.Tell(trial.Number, 5.0);

        Assert.Equal(TrialState.Complete, trial.State);
        Assert.Equal(5.0, trial.Value);
    }

    [Fact]
    public void Tell_FailTrial_UpdatesState()
    {
        using var study = Optimizer.CreateStudy("test", SimpleSpace,
            config: new TpeSamplerConfig { Seed = 42 });

        var trial = study.Ask();
        study.Tell(trial.Number, TrialState.Fail);

        Assert.Equal(TrialState.Fail, trial.State);
        Assert.Null(trial.Value);
    }

    [Fact]
    public void BestTrial_ReturnsLowestValue_Minimize()
    {
        using var study = Optimizer.CreateStudy("test", SimpleSpace,
            config: new TpeSamplerConfig { Seed = 42 });

        for (var i = 0; i < 5; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, 10.0 - i); // Values: 10, 9, 8, 7, 6
        }

        Assert.NotNull(study.BestTrial);
        Assert.Equal(6.0, study.BestTrial!.Value);
    }

    [Fact]
    public void BestTrial_ReturnsHighestValue_Maximize()
    {
        using var study = Optimizer.CreateStudy("test", SimpleSpace,
            direction: StudyDirection.Maximize,
            config: new TpeSamplerConfig { Seed = 42 });

        for (var i = 0; i < 5; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, (double)i);
        }

        Assert.NotNull(study.BestTrial);
        Assert.Equal(4.0, study.BestTrial!.Value);
    }

    [Fact]
    public void BestTrial_NullBeforeAnyComplete()
    {
        using var study = Optimizer.CreateStudy("test", SimpleSpace,
            config: new TpeSamplerConfig { Seed = 42 });

        Assert.Null(study.BestTrial);

        study.Ask(); // Running, not complete
        Assert.Null(study.BestTrial);
    }

    [Fact]
    public void AskBatch_ReturnsRequestedCount()
    {
        using var study = Optimizer.CreateStudy("test", SimpleSpace,
            config: new TpeSamplerConfig { Seed = 42 });

        var trials = study.AskBatch(10);

        Assert.Equal(10, trials.Count);
        foreach (var t in trials)
            Assert.Equal(TrialState.Running, t.State);
    }

    [Fact]
    public void TellBatch_UpdatesAllTrials()
    {
        using var study = Optimizer.CreateStudy("test", SimpleSpace,
            config: new TpeSamplerConfig { Seed = 42 });

        var trials = study.AskBatch(5);
        var results = trials.Select((t, i) =>
            new TrialResult(t.Number, (double)i, TrialState.Complete)).ToList();

        study.TellBatch(results);

        for (var i = 0; i < 5; i++)
        {
            Assert.Equal(TrialState.Complete, trials[i].State);
            Assert.Equal((double)i, trials[i].Value);
        }
    }

    [Fact]
    public void Ask_IncrementingTrialNumbers()
    {
        using var study = Optimizer.CreateStudy("test", SimpleSpace,
            config: new TpeSamplerConfig { Seed = 42 });

        for (var i = 0; i < 10; i++)
        {
            var trial = study.Ask();
            Assert.Equal(i, trial.Number);
        }
    }

    [Fact]
    public void ThreadSafety_ConcurrentAskTell_NoCorruption()
    {
        using var study = Optimizer.CreateStudy("test", SimpleSpace,
            config: new TpeSamplerConfig { Seed = 42 });

        var exceptions = new List<Exception>();
        var threads = new Thread[10];
        var trialsPerThread = 50;

        for (var t = 0; t < threads.Length; t++)
        {
            threads[t] = new Thread(() =>
            {
                try
                {
                    for (var i = 0; i < trialsPerThread; i++)
                    {
                        var trial = study.Ask();
                        study.Tell(trial.Number, trial.Number * 1.0);
                    }
                }
                catch (Exception ex)
                {
                    lock (exceptions)
                        exceptions.Add(ex);
                }
            });
        }

        foreach (var t in threads) t.Start();
        foreach (var t in threads) t.Join(TimeSpan.FromSeconds(30));

        Assert.Empty(exceptions);
        Assert.Equal(threads.Length * trialsPerThread, study.Trials.Count);

        // All trials should be complete
        foreach (var trial in study.Trials)
            Assert.Equal(TrialState.Complete, trial.State);
    }

    [Fact]
    public void Tell_InvalidTrialNumber_Throws()
    {
        using var study = Optimizer.CreateStudy("test", SimpleSpace,
            config: new TpeSamplerConfig { Seed = 42 });

        Assert.Throws<ArgumentException>(() => study.Tell(999, 1.0));
    }
}
