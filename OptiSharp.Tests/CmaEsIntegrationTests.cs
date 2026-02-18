using OptiSharp.Models;
using OptiSharp.Samplers.CmaEs;

namespace OptiSharp.Tests;

public sealed class CmaEsIntegrationTests
{
    [Fact]
    public void EndToEnd_CreateStudy_RunTrials_GetBest()
    {
        var space = new SearchSpace(
            Enumerable.Range(0, 10).Select(i => (ParameterRange)new FloatRange($"f{i}", 0, 10)).ToArray());

        using var study = Optimizer.CreateStudyWithCmaEs("e2e_cma", space,
            config: new CmaEsSamplerConfig { Seed = 42 });

        double Objective(IReadOnlyDictionary<string, object> p)
            => Enumerable.Range(0, 10).Sum(i => Math.Pow((double)p[$"f{i}"] - 5.0, 2));

        for (var i = 0; i < 100; i++)
        {
            var trial = study.Ask();
            var loss = Objective(trial.Parameters);
            study.Tell(trial.Number, loss);
        }

        Assert.Equal(100, study.Trials.Count);
        Assert.All(study.Trials, t => Assert.Equal(TrialState.Complete, t.State));
        Assert.All(study.Trials, t => Assert.True(t.Value.HasValue));

        var best = study.BestTrial;
        Assert.NotNull(best);
        Assert.True(best.Value!.Value < study.Trials.Average(t => t.Value!.Value));
    }

    [Fact]
    public void EndToEnd_BatchWorkflow()
    {
        var space = new SearchSpace([
            new FloatRange("x", -10, 10),
            new FloatRange("y", -10, 10)
        ]);

        var config = new CmaEsSamplerConfig { PopulationSize = 10, Seed = 42 };
        using var study = Optimizer.CreateStudyWithCmaEs("e2e_batch_cma", space, config: config);

        // Wave-based: ask batch, evaluate, tell batch
        for (var wave = 0; wave < 10; wave++)
        {
            var batch = study.AskBatch(10);
            Assert.Equal(10, batch.Count);
            Assert.All(batch, t => Assert.Equal(TrialState.Running, t.State));

            var results = batch.Select(t =>
            {
                var x = (double)t.Parameters["x"];
                var y = (double)t.Parameters["y"];
                return new TrialResult(t.Number, x * x + y * y, TrialState.Complete);
            }).ToList();

            study.TellBatch(results);
        }

        Assert.Equal(100, study.Trials.Count);
        Assert.All(study.Trials, t => Assert.Equal(TrialState.Complete, t.State));

        var best = study.BestTrial!;
        var bestX = (double)best.Parameters["x"];
        var bestY = (double)best.Parameters["y"];
        Assert.True(Math.Abs(bestX) < 8, $"Best x={bestX:F2}");
        Assert.True(Math.Abs(bestY) < 8, $"Best y={bestY:F2}");
    }

    [Fact]
    public void EndToEnd_FailedTrials_Handled()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        var config = new CmaEsSamplerConfig { PopulationSize = 6, Seed = 42 };
        using var study = Optimizer.CreateStudyWithCmaEs("e2e_fail_cma", space, config: config);

        for (var i = 0; i < 60; i++)
        {
            var trial = study.Ask();

            if (i % 6 == 0) // ~17% failure rate
            {
                study.Tell(trial.Number, TrialState.Fail);
            }
            else
            {
                var x = (double)trial.Parameters["x"];
                study.Tell(trial.Number, Math.Pow(x - 3, 2));
            }
        }

        Assert.Equal(60, study.Trials.Count);
        Assert.Equal(10, study.Trials.Count(t => t.State == TrialState.Fail));
        Assert.Equal(50, study.Trials.Count(t => t.State == TrialState.Complete));

        var best = study.BestTrial;
        Assert.NotNull(best);
        Assert.Equal(TrialState.Complete, best.State);
    }

    [Fact]
    public void EndToEnd_Maximize_Direction()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        var config = new CmaEsSamplerConfig { PopulationSize = 6, Seed = 42 };
        using var study = Optimizer.CreateStudyWithCmaEs("e2e_max_cma", space,
            direction: StudyDirection.Maximize, config: config);

        for (var i = 0; i < 100; i++)
        {
            var trial = study.Ask();
            var x = (double)trial.Parameters["x"];
            study.Tell(trial.Number, -Math.Pow(x - 7, 2) + 100);
        }

        var best = study.BestTrial!;
        Assert.True(best.Value!.Value > 80, $"Best value {best.Value.Value:F2} — expected > 80");
    }

    [Fact]
    public void EndToEnd_LogScaleParameters()
    {
        var space = new SearchSpace([
            new FloatRange("lr", 0.0001, 1.0, Log: true),
            new FloatRange("wd", 1e-6, 1e-2, Log: true)
        ]);

        var config = new CmaEsSamplerConfig { Seed = 42 };
        using var study = Optimizer.CreateStudyWithCmaEs("e2e_log_cma", space, config: config);

        for (var i = 0; i < 100; i++)
        {
            var trial = study.Ask();
            var lr = (double)trial.Parameters["lr"];
            var wd = (double)trial.Parameters["wd"];

            Assert.InRange(lr, 0.0001, 1.0);
            Assert.InRange(wd, 1e-6, 1e-2);

            var loss = Math.Pow(Math.Log(lr) - Math.Log(0.01), 2) +
                       Math.Pow(Math.Log(wd) - Math.Log(0.0001), 2);
            study.Tell(trial.Number, loss);
        }

        Assert.NotNull(study.BestTrial);
    }
}
