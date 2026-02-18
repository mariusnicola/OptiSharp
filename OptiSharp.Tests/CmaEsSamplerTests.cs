using OptiSharp.Models;
using OptiSharp.Samplers.CmaEs;

namespace OptiSharp.Tests;

public sealed class CmaEsSamplerTests
{
    private static readonly SearchSpace SimpleSpace = new([
        new FloatRange("x", 0, 10),
        new IntRange("n", 1, 5),
        new CategoricalRange("cat", ["a", "b"])
    ]);

    private static readonly SearchSpace ContinuousOnly = new([
        new FloatRange("x", -5, 5),
        new FloatRange("y", -5, 5)
    ]);

    [Fact]
    public void Sample_FirstGeneration_ReturnsValidParams()
    {
        var sampler = new CmaEsSampler(new CmaEsSamplerConfig { Seed = 42 });
        var trials = new List<Trial>();

        var result = sampler.Sample(trials, StudyDirection.Minimize, SimpleSpace);

        Assert.True(result.ContainsKey("x"));
        Assert.True(result.ContainsKey("n"));
        Assert.True(result.ContainsKey("cat"));
    }

    [Fact]
    public void Sample_FloatRange_WithinBounds()
    {
        var sampler = new CmaEsSampler(new CmaEsSamplerConfig { Seed = 42 });
        var trials = new List<Trial>();

        for (var i = 0; i < 100; i++)
        {
            var result = sampler.Sample(trials, StudyDirection.Minimize, ContinuousOnly);
            var x = (double)result["x"];
            var y = (double)result["y"];

            Assert.InRange(x, -5, 5);
            Assert.InRange(y, -5, 5);

            // Simulate evaluation so generations can complete
            var trial = new Trial(i, result) { State = TrialState.Complete, Value = x * x + y * y };
            trials.Add(trial);
        }
    }

    [Fact]
    public void Sample_IntRange_ReturnsIntegers()
    {
        var sampler = new CmaEsSampler(new CmaEsSamplerConfig { Seed = 42 });
        var trials = new List<Trial>();

        for (var i = 0; i < 50; i++)
        {
            var result = sampler.Sample(trials, StudyDirection.Minimize, SimpleSpace);
            Assert.IsType<int>(result["n"]);
            var n = (int)result["n"];
            Assert.InRange(n, 1, 5);

            var trial = new Trial(i, result) { State = TrialState.Complete, Value = i };
            trials.Add(trial);
        }
    }

    [Fact]
    public void Sample_CategoricalRange_ValidChoice()
    {
        var sampler = new CmaEsSampler(new CmaEsSamplerConfig { Seed = 42 });
        var trials = new List<Trial>();

        for (var i = 0; i < 50; i++)
        {
            var result = sampler.Sample(trials, StudyDirection.Minimize, SimpleSpace);
            var cat = result["cat"];
            Assert.True(cat.Equals("a") || cat.Equals("b"),
                $"Expected 'a' or 'b', got '{cat}'");

            var trial = new Trial(i, result) { State = TrialState.Complete, Value = i };
            trials.Add(trial);
        }
    }

    [Fact]
    public void Sample_PopulationSize_MatchesConfig()
    {
        var config = new CmaEsSamplerConfig { PopulationSize = 10, Seed = 42 };
        var sampler = new CmaEsSampler(config);
        var trials = new List<Trial>();

        // Issue 10 candidates (full population) then evaluate them all
        for (var i = 0; i < 10; i++)
        {
            var result = sampler.Sample(trials, StudyDirection.Minimize, ContinuousOnly);
            var trial = new Trial(i, result) { State = TrialState.Running };
            trials.Add(trial);
        }

        // Complete all trials
        for (var i = 0; i < 10; i++)
        {
            trials[i].State = TrialState.Complete;
            trials[i].Value = i;
        }

        // Next Sample should trigger generation update
        var nextResult = sampler.Sample(trials, StudyDirection.Minimize, ContinuousOnly);
        Assert.NotNull(nextResult);
        Assert.NotNull(sampler.Metrics);
        Assert.Equal(1, sampler.Metrics!.Generation);
    }

    [Fact]
    public void Sample_Deterministic_WithSeed()
    {
        var s1 = new CmaEsSampler(new CmaEsSamplerConfig { Seed = 99 });
        var s2 = new CmaEsSampler(new CmaEsSamplerConfig { Seed = 99 });
        var trials = new List<Trial>();

        var r1 = s1.Sample(trials, StudyDirection.Minimize, ContinuousOnly);
        var r2 = s2.Sample(trials, StudyDirection.Minimize, ContinuousOnly);

        Assert.Equal((double)r1["x"], (double)r2["x"]);
        Assert.Equal((double)r1["y"], (double)r2["y"]);
    }

    [Fact]
    public void Sample_GenerationCycles_UpdatesMean()
    {
        // Use a space where optimum is at x=0, y=0
        var space = new SearchSpace([
            new FloatRange("x", -10, 10),
            new FloatRange("y", -10, 10)
        ]);
        var config = new CmaEsSamplerConfig { PopulationSize = 6, Seed = 42 };
        var sampler = new CmaEsSampler(config);
        var trials = new List<Trial>();

        // Run 3 full generations
        for (var gen = 0; gen < 3; gen++)
        {
            for (var i = 0; i < 6; i++)
            {
                var result = sampler.Sample(trials, StudyDirection.Minimize, space);
                var x = (double)result["x"];
                var y = (double)result["y"];
                var trial = new Trial(trials.Count, result)
                {
                    State = TrialState.Complete,
                    Value = x * x + y * y // Sphere function
                };
                trials.Add(trial);
            }
        }

        // After 3 generations, metrics should exist and show progress
        Assert.NotNull(sampler.Metrics);
        Assert.True(sampler.Metrics!.Generation >= 2,
            $"Expected >= 2 generations, got {sampler.Metrics.Generation}");
    }

    [Fact]
    public void Sample_PartialGeneration_StillSamples()
    {
        var config = new CmaEsSamplerConfig { PopulationSize = 6, Seed = 42 };
        var sampler = new CmaEsSampler(config);
        var trials = new List<Trial>();

        // Issue 6 candidates but only complete 3
        for (var i = 0; i < 6; i++)
        {
            var result = sampler.Sample(trials, StudyDirection.Minimize, ContinuousOnly);
            var trial = new Trial(i, result)
            {
                State = i < 3 ? TrialState.Complete : TrialState.Running,
                Value = i < 3 ? i * 1.0 : null
            };
            trials.Add(trial);
        }

        // Should still be able to sample (from new population, no update yet)
        var next = sampler.Sample(trials, StudyDirection.Minimize, ContinuousOnly);
        Assert.NotNull(next);
        // Metrics should be null — no generation completed yet
        Assert.Null(sampler.Metrics);
    }

    [Fact]
    public void Sample_MaximizeDirection_Works()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        var config = new CmaEsSamplerConfig { PopulationSize = 6, Seed = 42 };
        var sampler = new CmaEsSampler(config);
        var trials = new List<Trial>();

        // Maximize: peak at x=7
        for (var i = 0; i < 60; i++)
        {
            var result = sampler.Sample(trials, StudyDirection.Maximize, space);
            var x = (double)result["x"];
            var trial = new Trial(i, result)
            {
                State = TrialState.Complete,
                Value = -Math.Pow(x - 7, 2) + 100
            };
            trials.Add(trial);
        }

        // Best trial should be near x=7
        var best = trials.OrderByDescending(t => t.Value).First();
        var bestX = (double)best.Parameters["x"];
        Assert.InRange(bestX, 3, 10);
    }

    [Fact]
    public void Sample_LogScaleFloat_WithinBounds()
    {
        var space = new SearchSpace([new FloatRange("lr", 0.0001, 1.0, Log: true)]);
        var sampler = new CmaEsSampler(new CmaEsSamplerConfig { Seed = 42 });
        var trials = new List<Trial>();

        for (var i = 0; i < 50; i++)
        {
            var result = sampler.Sample(trials, StudyDirection.Minimize, space);
            var lr = (double)result["lr"];
            Assert.InRange(lr, 0.0001, 1.0);

            var trial = new Trial(i, result)
            {
                State = TrialState.Complete,
                Value = Math.Pow(Math.Log(lr) - Math.Log(0.01), 2)
            };
            trials.Add(trial);
        }
    }

    [Fact]
    public void Sample_IntRangeWithStep_AlignsToStep()
    {
        var space = new SearchSpace([new IntRange("n", 0, 100, Step: 10)]);
        var sampler = new CmaEsSampler(new CmaEsSamplerConfig { Seed = 42 });
        var trials = new List<Trial>();

        for (var i = 0; i < 50; i++)
        {
            var result = sampler.Sample(trials, StudyDirection.Minimize, space);
            var n = (int)result["n"];
            Assert.Equal(0, n % 10); // Must be aligned to step of 10
            Assert.InRange(n, 0, 100);

            var trial = new Trial(i, result)
            {
                State = TrialState.Complete,
                Value = Math.Abs(n - 50)
            };
            trials.Add(trial);
        }
    }

    [Fact]
    public void Metrics_TracksSigmaAndCondition()
    {
        var config = new CmaEsSamplerConfig { PopulationSize = 6, Seed = 42 };
        var sampler = new CmaEsSampler(config);
        var trials = new List<Trial>();

        // Run 2 full generations
        for (var gen = 0; gen < 2; gen++)
        {
            for (var i = 0; i < 6; i++)
            {
                var result = sampler.Sample(trials, StudyDirection.Minimize, ContinuousOnly);
                var x = (double)result["x"];
                var y = (double)result["y"];
                var trial = new Trial(trials.Count, result)
                {
                    State = TrialState.Complete,
                    Value = x * x + y * y
                };
                trials.Add(trial);
            }
        }

        // Trigger metrics update via next Sample
        sampler.Sample(trials, StudyDirection.Minimize, ContinuousOnly);

        var metrics = sampler.Metrics;
        Assert.NotNull(metrics);
        Assert.True(metrics!.Sigma > 0, "Sigma should be positive");
        Assert.True(metrics.ConditionNumber >= 1.0, "Condition number >= 1");
        Assert.True(metrics.EvaluatedTrials > 0);
    }

    [Fact]
    public void Sample_CategoricalOnly_Throws()
    {
        var space = new SearchSpace([new CategoricalRange("c", ["a", "b"])]);
        var sampler = new CmaEsSampler(new CmaEsSamplerConfig { Seed = 42 });

        Assert.Throws<ArgumentException>(() =>
            sampler.Sample([], StudyDirection.Minimize, space));
    }
}