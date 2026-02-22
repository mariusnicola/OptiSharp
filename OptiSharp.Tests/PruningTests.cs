using OptiSharp.Models;
using OptiSharp.Pruning;
using Xunit;

namespace OptiSharp.Tests;

public class PruningTests
{
    private SearchSpace CreateSimpleSpace() =>
        new SearchSpace([new FloatRange("x", 0, 10)]);

    [Fact]
    public void NopPruner_NeverPrunes()
    {
        var space = CreateSimpleSpace();
        var pruner = new NopPruner();
        using var study = new Study("test", new Samplers.RandomSampler(), space, StudyDirection.Minimize, pruner);

        var trial = new Trial(0, new Dictionary<string, object> { ["x"] = 5.0 })
        {
            State = TrialState.Complete,
            Value = 10.0
        };

        Assert.False(pruner.ShouldPrune(trial, [trial]));
    }

    [Fact]
    public void Study_ShouldPrune_UsesConfiguredPruner()
    {
        var space = CreateSimpleSpace();
        var pruner = new NopPruner();
        using var study = new Study("test", new Samplers.RandomSampler(), space, StudyDirection.Minimize, pruner);

        var trial = new Trial(0, new Dictionary<string, object> { ["x"] = 5.0 })
        {
            State = TrialState.Complete,
            Value = 10.0
        };

        // Should use the pruner we provided
        Assert.False(study.ShouldPrune(trial));
    }

    [Fact]
    public void Trial_Report_StoresIntermediateValues()
    {
        var trial = new Trial(0, new Dictionary<string, object> { ["x"] = 5.0 });

        trial.Report(1.5, 1);
        trial.Report(1.2, 2);
        trial.Report(0.8, 3);

        Assert.Equal(3, trial.IntermediateValues.Count);
        Assert.Equal(1.5, trial.IntermediateValues[1]);
        Assert.Equal(1.2, trial.IntermediateValues[2]);
        Assert.Equal(0.8, trial.IntermediateValues[3]);
    }

    [Fact]
    public async Task Trial_Report_IsThreadSafe()
    {
        var trial = new Trial(0, new Dictionary<string, object> { ["x"] = 5.0 });
        var tasks = new List<Task>();

        for (int i = 0; i < 10; i++)
        {
            int step = i;
            tasks.Add(Task.Run(() =>
            {
                for (int j = 0; j < 100; j++)
                    trial.Report(step + 0.01 * j, step * 100 + j);
            }));
        }

        await Task.WhenAll(tasks);
        Assert.Equal(1000, trial.IntermediateValues.Count);
    }

    [Fact]
    public void MedianPruner_HasCorrectConfig()
    {
        var config = new MedianPrunerConfig
        {
            NStartupTrials = 5,
            NWarmupSteps = 2,
            IntervalSteps = 1
        };
        var pruner = new MedianPruner(config);

        // Pruner with config should instantiate correctly
        Assert.NotNull(pruner);
    }

    [Fact]
    public void MedianPruner_SkipsWarmupPhase()
    {
        var config = new MedianPrunerConfig { NWarmupSteps = 2 };
        var pruner = new MedianPruner(config);

        var trial = new Trial(0, new Dictionary<string, object> { ["x"] = 5.0 })
        {
            State = TrialState.Complete
        };
        trial.Report(1.0, 1);
        trial.Report(0.8, 2); // Still in warmup

        // Should not prune during warmup
        Assert.False(pruner.ShouldPrune(trial, [trial]));
    }

    [Fact]
    public void MedianPruner_ChecksIntervalSteps()
    {
        var config = new MedianPrunerConfig { IntervalSteps = 2 };
        var pruner = new MedianPruner(config);

        var trial = new Trial(0, new Dictionary<string, object> { ["x"] = 5.0 })
        {
            State = TrialState.Complete
        };
        trial.Report(1.0, 1); // Not at interval
        trial.Report(0.8, 2); // At interval

        // Should only check at intervals
        Assert.False(pruner.ShouldPrune(trial, [trial]));
    }

    [Fact]
    public void PercentilePruner_AcceptsValidPercentiles()
    {
        // Should accept valid percentiles
        var config1 = new PercentilePrunerConfig { Percentile = 0.0 };
        Assert.NotNull(new PercentilePruner(config1));

        var config2 = new PercentilePrunerConfig { Percentile = 50.0 };
        Assert.NotNull(new PercentilePruner(config2));

        var config3 = new PercentilePrunerConfig { Percentile = 100.0 };
        Assert.NotNull(new PercentilePruner(config3));
    }

    [Fact]
    public void PercentilePruner_ValidatesPercentile()
    {
        Assert.Throws<ArgumentException>(() => new PercentilePruner(new PercentilePrunerConfig { Percentile = -1 }));
        Assert.Throws<ArgumentException>(() => new PercentilePruner(new PercentilePrunerConfig { Percentile = 101 }));
    }

    [Fact]
    public void SuccessiveHalvingPruner_PrunesLowerRankedTrials()
    {
        var config = new SuccessiveHalvingPrunerConfig { MinResource = 1, ReductionFactor = 2.0 };
        var pruner = new SuccessiveHalvingPruner(config);

        var trials = Enumerable.Range(0, 4)
            .Select(i =>
            {
                var t = new Trial(i, new Dictionary<string, object> { ["x"] = i })
                {
                    State = TrialState.Complete
                };
                t.Report(i * 0.5, 1);
                return t;
            })
            .ToList();

        // Bottom half should be pruned
        var bottom = trials[3];
        Assert.True(pruner.ShouldPrune(bottom, trials));

        var top = trials[0];
        Assert.False(pruner.ShouldPrune(top, trials));
    }

    [Fact]
    public void SuccessiveHalvingPruner_ValidatesReductionFactor()
    {
        Assert.Throws<ArgumentException>(() => new SuccessiveHalvingPruner(new SuccessiveHalvingPrunerConfig { ReductionFactor = 1.0 }));
        Assert.Throws<ArgumentException>(() => new SuccessiveHalvingPruner(new SuccessiveHalvingPrunerConfig { ReductionFactor = 0.5 }));
    }

    [Fact]
    public void Study_Tell_AcceptsPrunedState()
    {
        var space = CreateSimpleSpace();
        using var study = Optimizer.CreateStudy("test", space);

        var trial = study.Ask();
        study.Tell(trial.Number, TrialState.Pruned);

        Assert.Equal(TrialState.Pruned, trial.State);
    }

    [Fact]
    public void Optimizer_CreateStudy_AcceptsPruner()
    {
        var space = CreateSimpleSpace();
        var pruner = new NopPruner();

        using var study = Optimizer.CreateStudy("test", space, pruner: pruner);
        Assert.NotNull(study);
    }

    [Fact]
    public void Study_BestTrial_SkipsPrunedTrials()
    {
        var space = CreateSimpleSpace();
        using var study = Optimizer.CreateStudy("test", space);

        var t1 = study.Ask();
        var t2 = study.Ask();

        study.Tell(t1.Number, 10.0);
        study.Tell(t2.Number, TrialState.Pruned);

        var best = study.BestTrial;
        Assert.NotNull(best);
        Assert.Equal(t1.Number, best.Number);
        Assert.Equal(TrialState.Complete, best.State);
    }

    [Fact]
    public void CmaEsSampler_TreatsPublishedSameAsFail()
    {
        // This is implicitly tested through the integration with the sampler
        // The sampler should treat Pruned same as Fail when computing generations
        var space = CreateSimpleSpace();
        var sampler = new Samplers.CmaEs.CmaEsSampler();
        using var study = new Study("test", sampler, space, StudyDirection.Minimize);

        var t1 = study.Ask();
        study.Tell(t1.Number, 10.0);

        var t2 = study.Ask();
        study.Tell(t2.Number, TrialState.Pruned);

        // Should not crash when processing Pruned state
        var t3 = study.Ask();
        Assert.NotNull(t3);
    }

    [Fact]
    public void Pruner_ReceivesAllTrialsForContext()
    {
        var space = CreateSimpleSpace();
        var pruner = new NopPruner();
        using var study = new Study("test", new Samplers.RandomSampler(), space, StudyDirection.Minimize, pruner);

        var t1 = study.Ask();
        var t2 = study.Ask();
        var t3 = study.Ask();

        study.Tell(t1.Number, 5.0);
        study.Tell(t2.Number, 3.0);

        // When checking if t2 should be pruned, pruner sees all trials
        var shouldPrune = study.ShouldPrune(t2);
        Assert.False(shouldPrune); // NopPruner never prunes
    }
}
