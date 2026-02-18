using OptiSharp.Models;
using OptiSharp.Samplers.Tpe;

namespace OptiSharp.Tests;

public sealed class TpeSamplerTests
{
    private static readonly SearchSpace SimpleSpace = new([
        new FloatRange("x", 0, 10),
        new IntRange("n", 1, 5),
        new CategoricalRange("cat", ["a", "b"])
    ]);

    private static List<Trial> CreateCompletedTrials(int count, SearchSpace space, int seed = 42)
    {
        var rng = new Random(seed);
        var trials = new List<Trial>();
        for (var i = 0; i < count; i++)
        {
            var parameters = new Dictionary<string, object>
            {
                ["x"] = rng.NextDouble() * 10,
                ["n"] = rng.Next(1, 6),
                ["cat"] = rng.Next(2) == 0 ? "a" : (object)"b"
            };
            var trial = new Trial(i, parameters)
            {
                State = TrialState.Complete,
                Value = rng.NextDouble() * 100
            };
            trials.Add(trial);
        }
        return trials;
    }

    [Fact]
    public void Sample_BelowStartup_ReturnsRandom()
    {
        var config = new TpeSamplerConfig { NStartupTrials = 20, Seed = 42 };
        var sampler = new TpeSampler(config);

        // Only 5 completed trials — should use random sampling
        var trials = CreateCompletedTrials(5, SimpleSpace);
        var result = sampler.Sample(trials, StudyDirection.Minimize, SimpleSpace);

        Assert.True(result.ContainsKey("x"));
        Assert.True(result.ContainsKey("n"));
        Assert.True(result.ContainsKey("cat"));
    }

    [Fact]
    public void Sample_AfterStartup_UsesTpe()
    {
        var config = new TpeSamplerConfig { NStartupTrials = 10, Seed = 42 };
        var sampler = new TpeSampler(config);

        // 50 completed trials — should use TPE
        var trials = CreateCompletedTrials(50, SimpleSpace);
        var result = sampler.Sample(trials, StudyDirection.Minimize, SimpleSpace);

        Assert.True(result.ContainsKey("x"));
        var x = (double)result["x"];
        Assert.InRange(x, 0, 10);
    }

    [Fact]
    public void Sample_WithinSearchBounds_AllParams()
    {
        var config = new TpeSamplerConfig { NStartupTrials = 10, Seed = 42 };
        var sampler = new TpeSampler(config);
        var trials = CreateCompletedTrials(50, SimpleSpace);

        for (var i = 0; i < 100; i++)
        {
            var result = sampler.Sample(trials, StudyDirection.Minimize, SimpleSpace);

            var x = (double)result["x"];
            Assert.InRange(x, 0, 10);

            var n = (int)result["n"];
            Assert.InRange(n, 1, 5);

            var cat = result["cat"];
            Assert.True(cat.Equals("a") || cat.Equals("b"));
        }
    }

    [Fact]
    public void Sample_IntRange_ReturnsIntegers()
    {
        var config = new TpeSamplerConfig { NStartupTrials = 10, Seed = 42 };
        var sampler = new TpeSampler(config);
        var trials = CreateCompletedTrials(50, SimpleSpace);

        for (var i = 0; i < 50; i++)
        {
            var result = sampler.Sample(trials, StudyDirection.Minimize, SimpleSpace);
            Assert.IsType<int>(result["n"]);
        }
    }

    [Fact]
    public void Sample_FloatLogRange_ReturnsPositive()
    {
        var space = new SearchSpace([new FloatRange("lr", 0.0001, 1.0, Log: true)]);
        var config = new TpeSamplerConfig { NStartupTrials = 10, Seed = 42 };
        var sampler = new TpeSampler(config);

        var rng = new Random(42);
        var trials = new List<Trial>();
        for (var i = 0; i < 50; i++)
        {
            var lr = Math.Exp(Math.Log(0.0001) + rng.NextDouble() * (Math.Log(1.0) - Math.Log(0.0001)));
            var trial = new Trial(i, new Dictionary<string, object> { ["lr"] = lr })
            {
                State = TrialState.Complete,
                Value = rng.NextDouble()
            };
            trials.Add(trial);
        }

        for (var i = 0; i < 50; i++)
        {
            var result = sampler.Sample(trials, StudyDirection.Minimize, space);
            var val = (double)result["lr"];
            Assert.InRange(val, 0.0001, 1.0);
        }
    }

    [Fact]
    public void Sample_ConstantLiar_RunningTrialsInAbove()
    {
        var config = new TpeSamplerConfig { NStartupTrials = 10, ConstantLiar = true, Seed = 42 };
        var sampler = new TpeSampler(config);

        var trials = CreateCompletedTrials(50, SimpleSpace);

        // Add some running trials
        for (var i = 0; i < 5; i++)
        {
            var t = new Trial(100 + i, new Dictionary<string, object>
            {
                ["x"] = 5.0, ["n"] = 3, ["cat"] = "a"
            });
            // State defaults to Running
            trials.Add(t);
        }

        // Should not throw and should still produce valid results
        var result = sampler.Sample(trials, StudyDirection.Minimize, SimpleSpace);
        Assert.True(result.ContainsKey("x"));
    }

    [Fact]
    public void Sample_Deterministic_WithSeed()
    {
        var trials = CreateCompletedTrials(50, SimpleSpace);

        var sampler1 = new TpeSampler(new TpeSamplerConfig { NStartupTrials = 10, Seed = 99 });
        var sampler2 = new TpeSampler(new TpeSamplerConfig { NStartupTrials = 10, Seed = 99 });

        var r1 = sampler1.Sample(trials, StudyDirection.Minimize, SimpleSpace);
        var r2 = sampler2.Sample(trials, StudyDirection.Minimize, SimpleSpace);

        Assert.Equal((double)r1["x"], (double)r2["x"]);
        Assert.Equal((int)r1["n"], (int)r2["n"]);
    }
}
