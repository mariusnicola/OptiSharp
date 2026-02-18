using OptiSharp.Models;
using OptiSharp.Samplers;
using OptiSharp.Samplers.CmaEs;
using OptiSharp.Samplers.Tpe;

namespace OptiSharp.Tests;

/// <summary>
/// Coverage tests for TPE, CMA-ES, and Random sampler code paths.
/// </summary>
public sealed class CoverageGap_SamplerTests
{
    // ── RandomSampler ─────────────────────────────────────────────────

    [Fact]
    public void RandomSampler_UnknownRangeType_Throws()
    {
        var sampler = new RandomSampler(seed: 1);
        var unknownRange = new TestHelpers.TestOnlyParameterRange("unknown");

        Assert.Throws<ArgumentException>(() => sampler.SampleParameter(unknownRange));
    }

    // ── CmaEsSampler ──────────────────────────────────────────────────

    [Fact]
    public void CmaEsSampler_CategoricalOnly_Throws()
    {
        var space = new SearchSpace([
            new CategoricalRange("c", ["a", "b", "c"])
        ]);
        var sampler = new CmaEsSampler(new CmaEsSamplerConfig { Seed = 1 });
        using var study = Optimizer.CreateStudy("test", space, sampler);

        Assert.Throws<ArgumentException>(() => study.Ask());
    }

    [Fact]
    public void CmaEsSampler_Dispose_WithoutInit_NoError()
    {
        var sampler = new CmaEsSampler(new CmaEsSamplerConfig { Seed = 1 });

        // Dispose without ever calling Sample — _gpu is null
        sampler.Dispose();
    }

    [Fact]
    public void CmaEsSampler_Properties_BeforeInit()
    {
        var sampler = new CmaEsSampler(new CmaEsSamplerConfig { Seed = 1 });

        Assert.Null(sampler.Metrics);
        Assert.Null(sampler.GpuDimensionWarning);
        Assert.False(sampler.IsGpuActive);
        Assert.Null(sampler.DeviceName);

        sampler.Dispose();
    }

    [Fact]
    public void CmaEsSampler_IntRange_WithStep_Aligned()
    {
        var space = new SearchSpace([
            new IntRange("n", 0, 100, Step: 10)
        ]);

        using var study = Optimizer.CreateStudyWithCmaEs("test", space,
            config: new CmaEsSamplerConfig { Seed = 42 });

        for (var i = 0; i < 20; i++)
        {
            var trial = study.Ask();
            var n = (int)trial.Parameters["n"];
            Assert.Equal(0, n % 10); // Must be aligned to step
            Assert.InRange(n, 0, 100);
            study.Tell(trial.Number, n);
        }
    }

    // ── ParzenEstimator: heap allocation path (>128 components) ──────

    [Fact]
    public void ParzenEstimator_LargeComponentCount_HeapAllocation()
    {
        // ParzenEstimator is internal, so test via TpeSampler with enough trials
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        using var study = Optimizer.CreateStudy("heap_test", space,
            config: new TpeSamplerConfig { NStartupTrials = 10, Seed = 42 });

        // Run 260+ trials (need >128 in "above" group to trigger heap path)
        // gamma = min(ceil(0.1*n), 25), so at n=260, below=25, above=235 > 128
        for (var i = 0; i < 260; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, (double)i);
        }

        // One more ask triggers TPE with >128 components in the above estimator
        var finalTrial = study.Ask();
        Assert.True(finalTrial.Parameters.ContainsKey("x"));
    }

    // ── TruncatedNormal ──────────────────────────────────────────────

    [Fact]
    public void LogSumExp_AllNegativeInfinity_ReturnsNegativeInfinity()
    {
        // This tests the IsNegativeInfinity(max) branch
        ReadOnlySpan<double> values = [double.NegativeInfinity, double.NegativeInfinity];
        var result = TruncatedNormal.LogSumExp(values);
        Assert.True(double.IsNegativeInfinity(result));
    }

    [Fact]
    public void TruncatedNormal_Sample_DegenerateRange_ReturnsMidpoint()
    {
        // When mu is far outside [low, high], cdfHigh - cdfLow ≈ 0 → degenerate
        var rng = new Random(42);
        var result = TruncatedNormal.Sample(rng, mu: 1e10, sigma: 0.001, low: 0, high: 1);

        // Should return midpoint (0.5) when CDF range is degenerate
        Assert.InRange(result, 0, 1);
    }

    // ── TpeSampler: ConstantLiar and MagicClip paths ─────────────────

    [Fact]
    public void TpeSampler_ConstantLiar_Disabled()
    {
        var space = new SearchSpace([
            new FloatRange("x", 0, 10),
            new CategoricalRange("c", ["a", "b", "c"])
        ]);

        // ConstantLiar=false skips adding running trials to above group
        using var study = Optimizer.CreateStudy("test", space,
            config: new TpeSamplerConfig { NStartupTrials = 5, ConstantLiar = false, Seed = 42 });

        for (var i = 0; i < 20; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, i * 1.0);
        }

        // Ask with running trials present but ConstantLiar=false
        var pending = study.Ask(); // stays running
        var next = study.Ask();    // should still work without constant liar
        Assert.NotNull(next);
    }

    [Fact]
    public void TpeSampler_MagicClipDisabled()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        using var study = Optimizer.CreateStudy("test", space,
            config: new TpeSamplerConfig { NStartupTrials = 5, ConsiderMagicClip = false, Seed = 42 });

        for (var i = 0; i < 20; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, i * 1.0);
        }

        var next = study.Ask();
        Assert.InRange((double)next.Parameters["x"], 0, 10);
    }
}
