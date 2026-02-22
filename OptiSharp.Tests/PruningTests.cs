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
        var config = new MedianPrunerConfig { NStartupTrials = 5, NWarmupSteps = 2 };
        var pruner = new MedianPruner(config);

        // 5 completed background trials with value at step 2
        var completedTrials = new List<Trial>();
        for (int i = 0; i < 5; i++)
        {
            var t = new Trial(i, new Dictionary<string, object> { ["x"] = (double)i })
            {
                State = TrialState.Complete,
                Value = 1.0
            };
            t.Report(1.0, 2);
            completedTrials.Add(t);
        }

        // Running trial at step 2 (still in warmup: 2 <= 2), much worse value
        var runningTrial = new Trial(5, new Dictionary<string, object> { ["x"] = 5.0 });
        runningTrial.Report(100.0, 2);

        var allTrials = completedTrials.Concat(new[] { runningTrial }).ToList();
        Assert.False(pruner.ShouldPrune(runningTrial, allTrials));
    }

    [Fact]
    public void MedianPruner_ChecksIntervalSteps()
    {
        var config = new MedianPrunerConfig { NStartupTrials = 5, IntervalSteps = 2 };
        var pruner = new MedianPruner(config);

        // 5 completed background trials with value at step 3
        var completedTrials = new List<Trial>();
        for (int i = 0; i < 5; i++)
        {
            var t = new Trial(i, new Dictionary<string, object> { ["x"] = (double)i })
            {
                State = TrialState.Complete,
                Value = 1.0
            };
            t.Report(1.0, 3);
            completedTrials.Add(t);
        }

        // Running trial at step 3 (not at interval: 3 % 2 != 0), much worse value
        var runningTrial = new Trial(5, new Dictionary<string, object> { ["x"] = 5.0 });
        runningTrial.Report(100.0, 3);

        var allTrials = completedTrials.Concat(new[] { runningTrial }).ToList();
        Assert.False(pruner.ShouldPrune(runningTrial, allTrials));
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

        // 3 completed background trials
        var completedTrials = Enumerable.Range(0, 3)
            .Select(i =>
            {
                var t = new Trial(i, new Dictionary<string, object> { ["x"] = (double)i })
                {
                    State = TrialState.Complete,
                    Value = i * 0.5
                };
                t.Report(i * 0.5, 1);
                return t;
            })
            .ToList();

        // Running trial with worst value → should be pruned
        var bottom = new Trial(3, new Dictionary<string, object> { ["x"] = 3.0 });
        bottom.Report(1.5, 1);
        var allTrialsWithBottom = completedTrials.Concat(new[] { bottom }).ToList();
        Assert.True(pruner.ShouldPrune(bottom, allTrialsWithBottom));

        // Running trial with best value → should NOT be pruned
        var top = new Trial(4, new Dictionary<string, object> { ["x"] = 0.0 });
        top.Report(0.0, 1);
        var allTrialsWithTop = completedTrials.Concat(new[] { top }).ToList();
        Assert.False(pruner.ShouldPrune(top, allTrialsWithTop));
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

    // --- Running-state pruning tests (benchmark scenario) ---

    [Fact]
    public void MedianPruner_PrunesRunningTrial_WhenWorseThanMedian()
    {
        var pruner = new MedianPruner(); // default: NStartupTrials=5, NWarmupSteps=0, IntervalSteps=1

        var completedTrials = new List<Trial>();
        for (int i = 0; i < 5; i++)
        {
            var t = new Trial(i, new Dictionary<string, object> { ["x"] = (double)i })
            {
                State = TrialState.Complete,
                Value = 1.0 + i * 0.1
            };
            t.Report(1.0 + i * 0.1, 3);
            completedTrials.Add(t);
        }

        // Running trial with much worse value at same step
        var runningTrial = new Trial(5, new Dictionary<string, object> { ["x"] = 5.0 });
        runningTrial.Report(100.0, 3);

        var allTrials = completedTrials.Concat(new[] { runningTrial }).ToList();
        Assert.True(pruner.ShouldPrune(runningTrial, allTrials));
    }

    [Fact]
    public void MedianPruner_DoesNotPrune_RunningTrial_WhenBetterThanMedian()
    {
        var pruner = new MedianPruner();

        var completedTrials = new List<Trial>();
        for (int i = 0; i < 5; i++)
        {
            var t = new Trial(i, new Dictionary<string, object> { ["x"] = (double)i })
            {
                State = TrialState.Complete,
                Value = 10.0 + i
            };
            t.Report(10.0 + i, 3);
            completedTrials.Add(t);
        }

        // Running trial with value well below median
        var runningTrial = new Trial(5, new Dictionary<string, object> { ["x"] = 5.0 });
        runningTrial.Report(0.5, 3);

        var allTrials = completedTrials.Concat(new[] { runningTrial }).ToList();
        Assert.False(pruner.ShouldPrune(runningTrial, allTrials));
    }

    [Fact]
    public void MedianPruner_DoesNotPrune_RunningTrial_InsufficientHistory()
    {
        var pruner = new MedianPruner(); // NStartupTrials = 5

        // Only 3 completed trials (< 5)
        var completedTrials = new List<Trial>();
        for (int i = 0; i < 3; i++)
        {
            var t = new Trial(i, new Dictionary<string, object> { ["x"] = (double)i })
            {
                State = TrialState.Complete,
                Value = 1.0
            };
            t.Report(1.0, 3);
            completedTrials.Add(t);
        }

        var runningTrial = new Trial(3, new Dictionary<string, object> { ["x"] = 5.0 });
        runningTrial.Report(100.0, 3);

        var allTrials = completedTrials.Concat(new[] { runningTrial }).ToList();
        Assert.False(pruner.ShouldPrune(runningTrial, allTrials));
    }

    [Fact]
    public void MedianPruner_DoesNotPrune_RunningTrial_DuringWarmup()
    {
        var config = new MedianPrunerConfig { NStartupTrials = 5, NWarmupSteps = 5 };
        var pruner = new MedianPruner(config);

        var completedTrials = new List<Trial>();
        for (int i = 0; i < 5; i++)
        {
            var t = new Trial(i, new Dictionary<string, object> { ["x"] = (double)i })
            {
                State = TrialState.Complete,
                Value = 1.0
            };
            t.Report(1.0, 3);
            completedTrials.Add(t);
        }

        // Running trial at step 3 which is within warmup (<= 5)
        var runningTrial = new Trial(5, new Dictionary<string, object> { ["x"] = 5.0 });
        runningTrial.Report(100.0, 3);

        var allTrials = completedTrials.Concat(new[] { runningTrial }).ToList();
        Assert.False(pruner.ShouldPrune(runningTrial, allTrials));
    }

    [Fact]
    public void MedianPruner_BenchmarkScenario_ActuallyPrunes()
    {
        var space = CreateSimpleSpace();
        var pruner = new MedianPruner(new MedianPrunerConfig { NStartupTrials = 5 });
        using var study = new Study("test", new Samplers.RandomSampler(42), space, StudyDirection.Minimize, pruner);

        int prunedCount = 0;
        int totalTrials = 30;

        for (int i = 0; i < totalTrials; i++)
        {
            var trial = study.Ask();
            var x = Convert.ToDouble(trial.Parameters["x"]);
            double value = x * x;

            bool wasPruned = false;
            for (int step = 1; step <= 10; step++)
            {
                var rng = new Random(42 ^ trial.Number ^ step);
                double noise = rng.NextDouble();
                double intermediate = value * (1 + (10 - step) * 0.5 * noise);
                trial.Report(intermediate, step);

                if (study.ShouldPrune(trial))
                {
                    study.Tell(trial.Number, TrialState.Pruned);
                    prunedCount++;
                    wasPruned = true;
                    break;
                }
            }

            if (!wasPruned)
                study.Tell(trial.Number, value);
        }

        Assert.True(prunedCount > 0, $"Expected some trials to be pruned, but prunedCount was {prunedCount}");
    }

    [Fact]
    public void PercentilePruner_PrunesRunningTrial_WhenWorseThanPercentile()
    {
        var config = new PercentilePrunerConfig { Percentile = 50.0 };
        var pruner = new PercentilePruner(config);

        var completedTrials = new List<Trial>();
        for (int i = 0; i < 5; i++)
        {
            var t = new Trial(i, new Dictionary<string, object> { ["x"] = (double)i })
            {
                State = TrialState.Complete,
                Value = 1.0 + i * 0.1
            };
            t.Report(1.0 + i * 0.1, 3);
            completedTrials.Add(t);
        }

        var runningTrial = new Trial(5, new Dictionary<string, object> { ["x"] = 5.0 });
        runningTrial.Report(100.0, 3);

        var allTrials = completedTrials.Concat(new[] { runningTrial }).ToList();
        Assert.True(pruner.ShouldPrune(runningTrial, allTrials));
    }

    [Fact]
    public void PercentilePruner_BenchmarkScenario_ActuallyPrunes()
    {
        var space = CreateSimpleSpace();
        var pruner = new PercentilePruner(new PercentilePrunerConfig { Percentile = 50.0, NStartupTrials = 5 });
        using var study = new Study("test", new Samplers.RandomSampler(42), space, StudyDirection.Minimize, pruner);

        int prunedCount = 0;
        for (int i = 0; i < 30; i++)
        {
            var trial = study.Ask();
            var x = Convert.ToDouble(trial.Parameters["x"]);
            double value = x * x;

            bool wasPruned = false;
            for (int step = 1; step <= 10; step++)
            {
                var rng = new Random(42 ^ trial.Number ^ step);
                double noise = rng.NextDouble();
                double intermediate = value * (1 + (10 - step) * 0.5 * noise);
                trial.Report(intermediate, step);

                if (study.ShouldPrune(trial))
                {
                    study.Tell(trial.Number, TrialState.Pruned);
                    prunedCount++;
                    wasPruned = true;
                    break;
                }
            }

            if (!wasPruned)
                study.Tell(trial.Number, value);
        }

        Assert.True(prunedCount > 0, $"Expected some trials to be pruned, but prunedCount was {prunedCount}");
    }

    [Fact]
    public void SuccessiveHalvingPruner_PrunesRunningTrial_WhenLowerRanked()
    {
        var config = new SuccessiveHalvingPrunerConfig { MinResource = 1, ReductionFactor = 2.0 };
        var pruner = new SuccessiveHalvingPruner(config);

        // 3 completed background trials
        var completedTrials = Enumerable.Range(0, 3)
            .Select(i =>
            {
                var t = new Trial(i, new Dictionary<string, object> { ["x"] = (double)i })
                {
                    State = TrialState.Complete,
                    Value = i * 0.5
                };
                t.Report(i * 0.5, 1);
                return t;
            })
            .ToList();

        // Running trial with worst value at same rung
        var worstRunning = new Trial(3, new Dictionary<string, object> { ["x"] = 3.0 });
        worstRunning.Report(1.5, 1);
        var allTrials = completedTrials.Concat(new[] { worstRunning }).ToList();
        Assert.True(pruner.ShouldPrune(worstRunning, allTrials));
    }

    [Fact]
    public void SuccessiveHalvingPruner_BenchmarkScenario_ActuallyPrunes()
    {
        var space = CreateSimpleSpace();
        var pruner = new SuccessiveHalvingPruner(new SuccessiveHalvingPrunerConfig { MinResource = 1, ReductionFactor = 3.0 });
        using var study = new Study("test", new Samplers.RandomSampler(42), space, StudyDirection.Minimize, pruner);

        int prunedCount = 0;
        for (int i = 0; i < 30; i++)
        {
            var trial = study.Ask();
            var x = Convert.ToDouble(trial.Parameters["x"]);
            double value = x * x;

            bool wasPruned = false;
            for (int step = 1; step <= 10; step++)
            {
                var rng = new Random(42 ^ trial.Number ^ step);
                double noise = rng.NextDouble();
                double intermediate = value * (1 + (10 - step) * 0.5 * noise);
                trial.Report(intermediate, step);

                if (study.ShouldPrune(trial))
                {
                    study.Tell(trial.Number, TrialState.Pruned);
                    prunedCount++;
                    wasPruned = true;
                    break;
                }
            }

            if (!wasPruned)
                study.Tell(trial.Number, value);
        }

        Assert.True(prunedCount > 0, $"Expected some trials to be pruned, but prunedCount was {prunedCount}");
    }

    [Fact]
    public void Study_PruningWorkflow_ActuallyPrunes()
    {
        var space = CreateSimpleSpace();
        var pruner = new MedianPruner(new MedianPrunerConfig { NStartupTrials = 5 });
        using var study = Optimizer.CreateStudy("test", space, pruner: pruner);

        int prunedCount = 0;
        int completedCount = 0;

        for (int i = 0; i < 30; i++)
        {
            var trial = study.Ask();
            var x = Convert.ToDouble(trial.Parameters["x"]);
            double value = x * x;

            bool wasPruned = false;
            for (int step = 1; step <= 10; step++)
            {
                var rng = new Random(42 ^ trial.Number ^ step);
                double noise = rng.NextDouble();
                double intermediate = value * (1 + (10 - step) * 0.5 * noise);
                trial.Report(intermediate, step);

                if (study.ShouldPrune(trial))
                {
                    study.Tell(trial.Number, TrialState.Pruned);
                    prunedCount++;
                    wasPruned = true;
                    break;
                }
            }

            if (!wasPruned)
            {
                study.Tell(trial.Number, value);
                completedCount++;
            }
        }

        Assert.True(prunedCount > 0, $"Expected some trials to be pruned, but prunedCount was {prunedCount}");
        Assert.True(completedCount > 0, "Expected some trials to complete");

        var best = study.BestTrial;
        Assert.NotNull(best);
        Assert.Equal(TrialState.Complete, best.State);

        var trials = study.Trials;
        Assert.Contains(trials, t => t.State == TrialState.Pruned);
        Assert.Contains(trials, t => t.State == TrialState.Complete);
    }
}
