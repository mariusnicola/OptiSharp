using OptiSharp.Models;
using OptiSharp.Samplers;

namespace OptiSharp.Tests;

public sealed class RandomSamplerTests
{
    private static readonly SearchSpace TestSpace = new([
        new IntRange("int_param", 1, 10),
        new IntRange("int_step", 0, 100, Step: 10),
        new FloatRange("float_param", 0.0, 1.0),
        new FloatRange("float_log", 0.001, 1.0, Log: true),
        new CategoricalRange("cat_param", ["a", "b", "c"])
    ]);

    [Fact]
    public void Sample_IntRange_WithinBounds()
    {
        var sampler = new RandomSampler(seed: 42);
        for (var i = 0; i < 100; i++)
        {
            var result = sampler.Sample([], StudyDirection.Minimize, TestSpace);
            var val = (int)result["int_param"];
            Assert.InRange(val, 1, 10);
        }
    }

    [Fact]
    public void Sample_IntRange_RespectsStep()
    {
        var sampler = new RandomSampler(seed: 42);
        for (var i = 0; i < 100; i++)
        {
            var result = sampler.Sample([], StudyDirection.Minimize, TestSpace);
            var val = (int)result["int_step"];
            Assert.Equal(0, val % 10);
            Assert.InRange(val, 0, 100);
        }
    }

    [Fact]
    public void Sample_FloatRange_WithinBounds()
    {
        var sampler = new RandomSampler(seed: 42);
        for (var i = 0; i < 100; i++)
        {
            var result = sampler.Sample([], StudyDirection.Minimize, TestSpace);
            var val = (double)result["float_param"];
            Assert.InRange(val, 0.0, 1.0);
        }
    }

    [Fact]
    public void Sample_FloatLogRange_WithinBounds()
    {
        var sampler = new RandomSampler(seed: 42);
        for (var i = 0; i < 100; i++)
        {
            var result = sampler.Sample([], StudyDirection.Minimize, TestSpace);
            var val = (double)result["float_log"];
            Assert.InRange(val, 0.001, 1.0);
        }
    }

    [Fact]
    public void Sample_CategoricalRange_ValidChoice()
    {
        var sampler = new RandomSampler(seed: 42);
        var validChoices = new HashSet<object> { "a", "b", "c" };
        for (var i = 0; i < 100; i++)
        {
            var result = sampler.Sample([], StudyDirection.Minimize, TestSpace);
            Assert.Contains(result["cat_param"], validChoices);
        }
    }

    [Fact]
    public void Sample_Deterministic_WithSeed()
    {
        var sampler1 = new RandomSampler(seed: 123);
        var sampler2 = new RandomSampler(seed: 123);

        for (var i = 0; i < 10; i++)
        {
            var r1 = sampler1.Sample([], StudyDirection.Minimize, TestSpace);
            var r2 = sampler2.Sample([], StudyDirection.Minimize, TestSpace);

            Assert.Equal(r1["int_param"], r2["int_param"]);
            Assert.Equal(r1["float_param"], r2["float_param"]);
        }
    }
}
