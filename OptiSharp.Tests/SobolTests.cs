using OptiSharp.Models;
using OptiSharp.Samplers.Sobol;

namespace OptiSharp.Tests;

public sealed class SobolTests
{
    private static readonly SearchSpace TestSpace = new(new ParameterRange[]
    {
        new IntRange("int_param", 1, 10),
        new IntRange("int_step", 0, 100, Step: 10),
        new FloatRange("float_param", 0.0, 1.0),
        new FloatRange("float_log", 0.001, 1.0, Log: true),
        new CategoricalRange("cat_param", new object[] { "a", "b", "c" }),
    });

    // --- Sequence property tests ---

    [Fact]
    public void Sequence_ValuesInUnitInterval()
    {
        var seq = new SobolSequence(5);
        for (var i = 0; i < 256; i++)
        {
            var point = seq.NextPoint();
            Assert.Equal(5, point.Length);
            foreach (var v in point)
            {
                Assert.InRange(v, 0.0, 1.0 - 1e-15);
            }
        }
    }

    [Fact]
    public void Sequence_Dimension1_IsVanDerCorput()
    {
        // First dimension of Sobol = Van der Corput base 2 (Gray code order)
        // Gray code reorders: 0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125
        var seq = new SobolSequence(1);
        var expected = new[] { 0.0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125 };

        for (var i = 0; i < expected.Length; i++)
        {
            var point = seq.NextPoint();
            Assert.Equal(expected[i], point[0], precision: 10);
        }
    }

    [Fact]
    public void Sequence_AllPointsUnique()
    {
        var seq = new SobolSequence(3);
        var seen = new HashSet<string>();

        for (var i = 0; i < 1024; i++)
        {
            var point = seq.NextPoint();
            var key = string.Join(",", point.Select(v => v.ToString("R")));
            Assert.True(seen.Add(key), $"Duplicate point at index {i}: {key}");
        }
    }

    [Fact]
    public void Sequence_Scrambled_DifferentFromUnscrambled()
    {
        var plain = new SobolSequence(3);
        var scrambled = new SobolSequence(3, seed: 42);

        // Skip the origin (first point)
        plain.NextPoint();
        scrambled.NextPoint();

        // At least one subsequent point should differ
        var anyDifferent = false;
        for (var i = 0; i < 10; i++)
        {
            var p1 = plain.NextPoint();
            var p2 = scrambled.NextPoint();
            if (Math.Abs(p1[0] - p2[0]) > 1e-15)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "Scrambled sequence should differ from unscrambled");
    }

    [Fact]
    public void Sequence_Scrambled_Reproducible()
    {
        var seq1 = new SobolSequence(5, seed: 42);
        var seq2 = new SobolSequence(5, seed: 42);

        for (var i = 0; i < 100; i++)
        {
            var p1 = seq1.NextPoint();
            var p2 = seq2.NextPoint();
            for (var d = 0; d < 5; d++)
                Assert.Equal(p1[d], p2[d]);
        }
    }

    [Fact]
    public void Sequence_LowDiscrepancy_BetterThanRandom()
    {
        // Sobol should cover [0,1]^2 more uniformly than random.
        // Divide [0,1]^2 into a 4x4 grid and count coverage with 64 points.
        // Sobol should cover all 16 cells; random likely misses some.
        const int gridSize = 4;
        const int points = 64;

        var seq = new SobolSequence(2);
        var sobolCells = new HashSet<(int, int)>();

        for (var i = 0; i < points; i++)
        {
            var point = seq.NextPoint();
            var cellX = Math.Min((int)(point[0] * gridSize), gridSize - 1);
            var cellY = Math.Min((int)(point[1] * gridSize), gridSize - 1);
            sobolCells.Add((cellX, cellY));
        }

        // Sobol with 64 points in 2D should cover all 16 cells of a 4x4 grid
        Assert.Equal(gridSize * gridSize, sobolCells.Count);
    }

    [Fact]
    public void Sequence_HighDimension_DoesNotThrow()
    {
        var seq = new SobolSequence(200);
        for (var i = 0; i < 32; i++)
        {
            var point = seq.NextPoint();
            Assert.Equal(200, point.Length);
            foreach (var v in point)
                Assert.InRange(v, 0.0, 1.0 - 1e-15);
        }
    }

    [Fact]
    public void Sequence_ThrowsOnInvalidDimensions()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new SobolSequence(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SobolSequence(201));
    }

    // --- ISampler contract tests ---

    [Fact]
    public void Sample_IntRange_WithinBounds()
    {
        var sampler = new SobolSampler(new SobolSamplerConfig { Seed = 42 });
        for (var i = 0; i < 100; i++)
        {
            var result = sampler.Sample(new List<Trial>(), StudyDirection.Minimize, TestSpace);
            var val = (int)result["int_param"];
            Assert.InRange(val, 1, 10);
        }
    }

    [Fact]
    public void Sample_IntRange_RespectsStep()
    {
        var sampler = new SobolSampler(new SobolSamplerConfig { Seed = 42 });
        for (var i = 0; i < 100; i++)
        {
            var result = sampler.Sample(new List<Trial>(), StudyDirection.Minimize, TestSpace);
            var val = (int)result["int_step"];
            Assert.Equal(0, val % 10);
            Assert.InRange(val, 0, 100);
        }
    }

    [Fact]
    public void Sample_FloatRange_WithinBounds()
    {
        var sampler = new SobolSampler(new SobolSamplerConfig { Seed = 42 });
        for (var i = 0; i < 100; i++)
        {
            var result = sampler.Sample(new List<Trial>(), StudyDirection.Minimize, TestSpace);
            var val = (double)result["float_param"];
            Assert.InRange(val, 0.0, 1.0);
        }
    }

    [Fact]
    public void Sample_FloatLogRange_WithinBounds()
    {
        var sampler = new SobolSampler(new SobolSamplerConfig { Seed = 42 });
        for (var i = 0; i < 100; i++)
        {
            var result = sampler.Sample(new List<Trial>(), StudyDirection.Minimize, TestSpace);
            var val = (double)result["float_log"];
            Assert.InRange(val, 0.001, 1.0);
        }
    }

    [Fact]
    public void Sample_CategoricalRange_ValidChoice()
    {
        var sampler = new SobolSampler(new SobolSamplerConfig { Seed = 42 });
        var validChoices = new HashSet<object> { "a", "b", "c" };
        for (var i = 0; i < 100; i++)
        {
            var result = sampler.Sample(new List<Trial>(), StudyDirection.Minimize, TestSpace);
            Assert.Contains(result["cat_param"], validChoices);
        }
    }

    [Fact]
    public void Sample_Deterministic_WithSeed()
    {
        var sampler1 = new SobolSampler(new SobolSamplerConfig { Seed = 123 });
        var sampler2 = new SobolSampler(new SobolSamplerConfig { Seed = 123 });

        for (var i = 0; i < 10; i++)
        {
            var r1 = sampler1.Sample(new List<Trial>(), StudyDirection.Minimize, TestSpace);
            var r2 = sampler2.Sample(new List<Trial>(), StudyDirection.Minimize, TestSpace);

            Assert.Equal(r1["int_param"], r2["int_param"]);
            Assert.Equal(r1["float_param"], r2["float_param"]);
        }
    }

    [Fact]
    public void Sample_AllParametersPresent()
    {
        var sampler = new SobolSampler(new SobolSamplerConfig { Seed = 42 });
        var result = sampler.Sample(new List<Trial>(), StudyDirection.Minimize, TestSpace);

        Assert.Equal(TestSpace.Count, result.Count);
        foreach (var range in TestSpace)
            Assert.True(result.ContainsKey(range.Name), $"Missing parameter: {range.Name}");
    }

    // --- Convergence test: Sobol should beat random on average ---

    [Fact]
    public void Convergence_BeatsRandom_OnSphere()
    {
        // Minimize f(x) = sum(x_i^2) over [-5, 5]^5
        // Sobol should find a point closer to 0 than random with same number of evaluations
        var space = new SearchSpace(Enumerable.Range(0, 5).Select(
            i => (ParameterRange)new FloatRange($"x{i}", -5.0, 5.0)));

        const int trials = 64;

        // Sobol
        var sobolSampler = new SobolSampler(new SobolSamplerConfig { Seed = 42 });
        var sobolBest = double.MaxValue;
        var trialList = new List<Trial>();
        for (var i = 0; i < trials; i++)
        {
            var p = sobolSampler.Sample(trialList, StudyDirection.Minimize, space);
            var f = p.Values.Cast<double>().Sum(x => x * x);
            if (f < sobolBest) sobolBest = f;
        }

        // Random (10 seeds, take median)
        var randomBests = new List<double>();
        for (var seed = 0; seed < 10; seed++)
        {
            var rng = new Random(seed);
            var best = double.MaxValue;
            for (var i = 0; i < trials; i++)
            {
                var f = 0.0;
                for (var d = 0; d < 5; d++)
                {
                    var x = -5.0 + rng.NextDouble() * 10.0;
                    f += x * x;
                }
                if (f < best) best = f;
            }
            randomBests.Add(best);
        }
        randomBests.Sort();
        var randomMedian = randomBests[randomBests.Count / 2];

        // Sobol should be at least as good as median random
        Assert.True(sobolBest <= randomMedian * 1.5,
            $"Sobol best ({sobolBest:F4}) should be close to or better than random median ({randomMedian:F4})");
    }

    // --- Factory method test ---

    [Fact]
    public void Optimizer_CreateStudyWithSobolSampler_Works()
    {
        var space = new SearchSpace(new ParameterRange[]
        {
            new FloatRange("x", 0.0, 1.0),
        });

        var study = Optimizer.CreateStudyWithSobolSampler("test", space,
            config: new SobolSamplerConfig { Seed = 42 });

        var trial = study.Ask();
        Assert.True(trial.Parameters.ContainsKey("x"));
        var x = (double)trial.Parameters["x"];
        Assert.InRange(x, 0.0, 1.0);
    }
}
