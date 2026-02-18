using OptiSharp.Models;
using OptiSharp.Samplers.Tpe;

namespace OptiSharp.Tests;

public sealed class IntegrationTests
{
    [Fact]
    public void EndToEnd_CreateStudy_RunTrials_GetBest()
    {
        // Realistic search space matching our hyperopt (62 params)
        var ranges = new List<ParameterRange>();
        for (var i = 0; i < 55; i++)
            ranges.Add(new FloatRange($"f{i}", 0, 10));
        for (var i = 0; i < 5; i++)
            ranges.Add(new IntRange($"i{i}", 0, 100));
        for (var i = 0; i < 2; i++)
            ranges.Add(new CategoricalRange($"c{i}", ["a", "b", "c"]));
        var space = new SearchSpace(ranges);

        // Create study
        using var study = Optimizer.CreateStudy("e2e_test", space,
            config: new TpeSamplerConfig { NStartupTrials = 20, Seed = 42 });

        // Run 100 trials with a simple objective
        double Objective(IReadOnlyDictionary<string, object> p)
            => Enumerable.Range(0, 55).Sum(i => Math.Pow((double)p[$"f{i}"] - 5.0, 2));

        for (var i = 0; i < 100; i++)
        {
            var trial = study.Ask();
            var loss = Objective(trial.Parameters);
            study.Tell(trial.Number, loss);
        }

        // Verify study state
        Assert.Equal(100, study.Trials.Count);
        Assert.All(study.Trials, t => Assert.Equal(TrialState.Complete, t.State));
        Assert.All(study.Trials, t => Assert.True(t.Value.HasValue));

        // Best trial should exist and have lower loss than average
        var best = study.BestTrial;
        Assert.NotNull(best);
        Assert.True(best.Value!.Value < study.Trials.Average(t => t.Value!.Value));
    }

    [Fact]
    public void EndToEnd_BatchWorkflow()
    {
        var space = new SearchSpace([
            new FloatRange("x", -10, 10),
            new FloatRange("y", -10, 10),
            new IntRange("n", 1, 20),
            new CategoricalRange("mode", ["fast", "slow", "balanced"])
        ]);

        using var study = Optimizer.CreateStudy("e2e_batch", space,
            config: new TpeSamplerConfig { NStartupTrials = 10, Seed = 42 });

        // Wave-based workflow (matches production pattern)
        for (var wave = 0; wave < 10; wave++)
        {
            var batch = study.AskBatch(10);
            Assert.Equal(10, batch.Count);

            // All trials should be running
            Assert.All(batch, t => Assert.Equal(TrialState.Running, t.State));

            // Evaluate
            var results = batch.Select(t =>
            {
                var x = (double)t.Parameters["x"];
                var y = (double)t.Parameters["y"];
                var loss = x * x + y * y;
                return new TrialResult(t.Number, loss, TrialState.Complete);
            }).ToList();

            study.TellBatch(results);
        }

        Assert.Equal(100, study.Trials.Count);
        Assert.All(study.Trials, t => Assert.Equal(TrialState.Complete, t.State));

        // Best should have small x,y values
        var best = study.BestTrial!;
        var bestX = (double)best.Parameters["x"];
        var bestY = (double)best.Parameters["y"];
        Assert.True(Math.Abs(bestX) < 5, $"Best x={bestX:F2} — expected |x| < 5");
        Assert.True(Math.Abs(bestY) < 5, $"Best y={bestY:F2} — expected |y| < 5");
    }

    [Fact]
    public void EndToEnd_FailedTrials_Handled()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        using var study = Optimizer.CreateStudy("e2e_fail", space,
            config: new TpeSamplerConfig { NStartupTrials = 5, Seed = 42 });

        for (var i = 0; i < 50; i++)
        {
            var trial = study.Ask();

            // Simulate 20% failure rate
            if (i % 5 == 0)
            {
                study.Tell(trial.Number, TrialState.Fail);
            }
            else
            {
                var x = (double)trial.Parameters["x"];
                study.Tell(trial.Number, Math.Pow(x - 3, 2));
            }
        }

        Assert.Equal(50, study.Trials.Count);
        Assert.Equal(10, study.Trials.Count(t => t.State == TrialState.Fail));
        Assert.Equal(40, study.Trials.Count(t => t.State == TrialState.Complete));

        // Best should still be found among completed trials
        var best = study.BestTrial;
        Assert.NotNull(best);
        Assert.Equal(TrialState.Complete, best.State);
    }

    [Fact]
    public void EndToEnd_Maximize_Direction()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        using var study = Optimizer.CreateStudy("e2e_max", space,
            direction: StudyDirection.Maximize,
            config: new TpeSamplerConfig { NStartupTrials = 10, Seed = 42 });

        for (var i = 0; i < 100; i++)
        {
            var trial = study.Ask();
            var x = (double)trial.Parameters["x"];
            // Maximize: peak at x=7
            study.Tell(trial.Number, -Math.Pow(x - 7, 2) + 100);
        }

        var best = study.BestTrial!;
        Assert.True(best.Value!.Value > 90, $"Best value {best.Value.Value:F2} — expected > 90");
    }

    [Fact]
    public void EndToEnd_RandomSampler()
    {
        var space = new SearchSpace([
            new FloatRange("x", -5, 5),
            new IntRange("n", 0, 10),
            new CategoricalRange("c", ["a", "b"])
        ]);

        using var study = Optimizer.CreateStudyWithRandomSampler("e2e_random", space, seed: 42);

        for (var i = 0; i < 100; i++)
        {
            var trial = study.Ask();
            var x = (double)trial.Parameters["x"];
            study.Tell(trial.Number, x * x);
        }

        Assert.Equal(100, study.Trials.Count);
        Assert.NotNull(study.BestTrial);
        // Random should still find something reasonable in 100 trials on 1D
        Assert.True(study.BestTrial.Value!.Value < 1.0,
            $"Random best: {study.BestTrial.Value.Value:F4} — expected < 1.0");
    }

    [Fact]
    public void EndToEnd_LogScaleParameters()
    {
        var space = new SearchSpace([
            new FloatRange("lr", 0.0001, 1.0, Log: true),
            new FloatRange("weight_decay", 1e-6, 1e-2, Log: true)
        ]);

        using var study = Optimizer.CreateStudy("e2e_log", space,
            config: new TpeSamplerConfig { NStartupTrials = 10, Seed = 42 });

        for (var i = 0; i < 100; i++)
        {
            var trial = study.Ask();
            var lr = (double)trial.Parameters["lr"];
            var wd = (double)trial.Parameters["weight_decay"];

            // Verify log-scale params are in bounds
            Assert.InRange(lr, 0.0001, 1.0);
            Assert.InRange(wd, 1e-6, 1e-2);

            // Objective: optimal at lr=0.01, wd=0.0001
            var loss = Math.Pow(Math.Log(lr) - Math.Log(0.01), 2) +
                       Math.Pow(Math.Log(wd) - Math.Log(0.0001), 2);
            study.Tell(trial.Number, loss);
        }

        var bestLr = (double)study.BestTrial!.Parameters["lr"];
        var bestWd = (double)study.BestTrial.Parameters["weight_decay"];

        // Should be within order of magnitude of optimum
        Assert.InRange(bestLr, 0.001, 0.1);
        Assert.InRange(bestWd, 1e-5, 1e-3);
    }
}
