using OptiSharp.Models;
using OptiSharp.Samplers.CmaEs;

namespace OptiSharp.Tests;

/// <summary>
/// Coverage tests for GpuCmaEsProvider direct calls, buffer management, and GPU code paths.
/// </summary>
public sealed class CoverageGap_GpuTests
{
    // ── GpuCmaEsProvider: factory and availability ────────────────────

    [GpuFact]
    public void GpuCmaEsProvider_IsCudaAvailable_ReturnsTrue()
    {
        // RTX 3090 is available
        var available = GpuCmaEsProvider.IsCudaAvailable();
        Assert.True(available);
    }

    [Fact]
    public void GpuCmaEsProvider_TryCreate_CpuBackend_ReturnsNull()
    {
        var result = GpuCmaEsProvider.TryCreate(ComputeBackend.Cpu);
        Assert.Null(result);
    }

    [GpuFact]
    public void GpuCmaEsProvider_TryCreate_GpuBackend_Succeeds()
    {
        var provider = GpuCmaEsProvider.TryCreate(ComputeBackend.Gpu);
        Assert.NotNull(provider);
        Assert.True(provider!.IsGpu);
        Assert.NotNull(provider.DeviceName);
        provider.Dispose();
    }

    [GpuFact]
    public void GpuCmaEsProvider_TryCreate_AutoBackend_Succeeds()
    {
        var provider = GpuCmaEsProvider.TryCreate(ComputeBackend.Auto);
        Assert.NotNull(provider);
        Assert.True(provider!.IsGpu);
        provider.Dispose();
    }

    // ── CMA-ES with GPU: integration paths ────────────────────────────

    [GpuFact]
    public void CmaEsSampler_Gpu_BufferReuse_DifferentDimensions()
    {
        // Run CMA-ES with GPU on two different space sizes to trigger buffer reallocation
        var space1 = new SearchSpace(
            Enumerable.Range(0, 10).Select(i => new FloatRange($"x{i}", -5, 5)).ToList());

        using var study1 = Optimizer.CreateStudyWithCmaEs("gpu_reuse1", space1,
            config: new CmaEsSamplerConfig { Seed = 42, Backend = ComputeBackend.Gpu });

        // Run enough trials to complete a generation and trigger GPU paths
        for (var i = 0; i < 30; i++)
        {
            var trial = study1.Ask();
            var value = Enumerable.Range(0, 10).Sum(j => Math.Pow((double)trial.Parameters[$"x{j}"], 2));
            study1.Tell(trial.Number, value);
        }

        Assert.NotNull(study1.BestTrial);
    }

    [GpuFact]
    public void CmaEsSampler_Gpu_FullGenerationCycle()
    {
        // Ensure GPU rank-mu update path is exercised by completing full generations
        var dims = 20;
        var space = new SearchSpace(
            Enumerable.Range(0, dims).Select(i => new FloatRange($"x{i}", -5, 5)).ToList());

        var config = new CmaEsSamplerConfig { Seed = 42, Backend = ComputeBackend.Gpu };
        using var study = Optimizer.CreateStudyWithCmaEs("gpu_gen", space, config: config);

        // Run multiple complete generations
        for (var i = 0; i < 100; i++)
        {
            var trial = study.Ask();
            var value = Enumerable.Range(0, dims).Sum(j => Math.Pow((double)trial.Parameters[$"x{j}"], 2));
            study.Tell(trial.Number, value);
        }

        // Verify CMA-ES updated through multiple generations
        var sampler = TestHelpers.GetSamplerFromStudy(study);
        Assert.NotNull(sampler?.Metrics);
        Assert.True(sampler!.Metrics!.Generation > 1);
        Assert.True(sampler.IsGpuActive);
    }

    [GpuFact]
    public void CmaEsSampler_LogScale_WithGpu()
    {
        var space = new SearchSpace([
            new FloatRange("lr", 0.0001, 1.0, Log: true),
            new FloatRange("wd", 1e-6, 1e-2, Log: true)
        ]);

        using var study = Optimizer.CreateStudyWithCmaEs("gpu_log", space,
            config: new CmaEsSamplerConfig { Seed = 42, Backend = ComputeBackend.Gpu });

        for (var i = 0; i < 30; i++)
        {
            var trial = study.Ask();
            var lr = (double)trial.Parameters["lr"];
            var wd = (double)trial.Parameters["wd"];
            Assert.InRange(lr, 0.0001, 1.0);
            Assert.InRange(wd, 1e-6, 1e-2);
            study.Tell(trial.Number, lr + wd);
        }
    }

    // ── GpuCmaEsProvider: direct SamplePopulation ─────────────────────

    [GpuFact]
    public void GpuCmaEsProvider_SamplePopulation_DirectCall()
    {
        var provider = GpuCmaEsProvider.TryCreate(ComputeBackend.Gpu);
        Assert.NotNull(provider);

        const int n = 5;
        const int lambda = 3;

        // BD = identity (flattened row-major)
        var bd = new double[n * n];
        for (var i = 0; i < n; i++) bd[i * n + i] = 1.0;

        // Z = random normals (flattened)
        var z = new double[lambda * n];
        var rng = new Random(42);
        for (var i = 0; i < z.Length; i++) z[i] = rng.NextDouble() * 2 - 1;

        var mean = new double[n];
        var output = new double[lambda * n];

        provider!.SamplePopulation(bd, z, mean, 1.0, output, lambda, n);

        // GPU should have computed non-zero output
        Assert.True(output.Any(v => Math.Abs(v) > 1e-10));
        provider.Dispose();
    }

    // ── GpuCmaEsProvider: direct ComputeRankMu ────────────────────────

    [GpuFact]
    public void GpuCmaEsProvider_ComputeRankMu_DirectCall()
    {
        var provider = GpuCmaEsProvider.TryCreate(ComputeBackend.Gpu);
        Assert.NotNull(provider);

        const int n = 4;
        const int mu = 3;

        var artmp = new double[mu * n];
        var rng = new Random(42);
        for (var i = 0; i < artmp.Length; i++) artmp[i] = rng.NextDouble();

        var weights = new double[mu];
        for (var i = 0; i < mu; i++) weights[i] = 1.0 / mu;

        var output = new double[n * n];

        provider!.ComputeRankMu(artmp, weights, output, mu, n);

        Assert.True(output.Any(v => Math.Abs(v) > 1e-10));
        provider.Dispose();
    }

    // ── GpuCmaEsProvider: buffer reallocation ─────────────────────────

    [GpuFact]
    public void GpuCmaEsProvider_PopulationBufferReallocation()
    {
        var provider = GpuCmaEsProvider.TryCreate(ComputeBackend.Gpu);
        Assert.NotNull(provider);

        // First call with n=3, lambda=2 — initial buffer allocation
        var bd3 = new double[9];
        bd3[0] = bd3[4] = bd3[8] = 1.0;
        var z3 = new double[6];
        z3[0] = 1; z3[3] = -1;
        var mean3 = new double[3];
        var out3 = new double[6];
        provider!.SamplePopulation(bd3, z3, mean3, 1.0, out3, 2, 3);

        // Second call with n=5, lambda=4 — triggers dispose-old + allocate-new path
        var bd5 = new double[25];
        for (var i = 0; i < 5; i++) bd5[i * 5 + i] = 1.0;
        var z5 = new double[20];
        for (var i = 0; i < 20; i++) z5[i] = 0.5;
        var mean5 = new double[5];
        var out5 = new double[20];
        provider.SamplePopulation(bd5, z5, mean5, 1.0, out5, 4, 5);

        Assert.True(out5.Any(v => Math.Abs(v) > 1e-10));
        provider.Dispose();
    }

    [GpuFact]
    public void GpuCmaEsProvider_RankMuBufferReallocation()
    {
        var provider = GpuCmaEsProvider.TryCreate(ComputeBackend.Gpu);
        Assert.NotNull(provider);

        // First call: mu=2, n=3
        var artmp3 = new double[6];
        for (var i = 0; i < 6; i++) artmp3[i] = (i + 1) * 0.1;
        var w3 = new[] { 0.6, 0.4 };
        var out3 = new double[9];
        provider!.ComputeRankMu(artmp3, w3, out3, 2, 3);

        // Second call: mu=3, n=5 — triggers buffer reallocation
        var artmp5 = new double[15];
        for (var i = 0; i < 15; i++) artmp5[i] = (i + 1) * 0.05;
        var w5 = new[] { 0.5, 0.3, 0.2 };
        var out5 = new double[25];
        provider.ComputeRankMu(artmp5, w5, out5, 3, 5);

        Assert.True(out5.Any(v => Math.Abs(v) > 1e-10));
        provider.Dispose();
    }

    // ── GpuCmaEsProvider: buffer cache hit ────────────────────────────

    [GpuFact]
    public void GpuCmaEsProvider_PopulationBufferCacheHit()
    {
        var provider = GpuCmaEsProvider.TryCreate(ComputeBackend.Gpu);
        Assert.NotNull(provider);

        const int n = 4;
        const int lambda = 2;

        var bd = new double[n * n];
        for (var i = 0; i < n; i++) bd[i * n + i] = 1.0;
        var mean = new double[n];

        // First call — allocates buffers
        var z1 = new double[lambda * n];
        z1[0] = 1.0;
        var out1 = new double[lambda * n];
        provider!.SamplePopulation(bd, z1, mean, 1.0, out1, lambda, n);

        // Second call with same dimensions — hits cache (early return in EnsurePopulationBuffers)
        var z2 = new double[lambda * n];
        z2[1] = -1.0;
        var out2 = new double[lambda * n];
        provider.SamplePopulation(bd, z2, mean, 1.0, out2, lambda, n);

        // Both should produce results
        Assert.True(out1.Any(v => Math.Abs(v) > 1e-10));
        provider.Dispose();
    }

    [GpuFact]
    public void GpuCmaEsProvider_RankMuBufferCacheHit()
    {
        var provider = GpuCmaEsProvider.TryCreate(ComputeBackend.Gpu);
        Assert.NotNull(provider);

        const int n = 3;
        const int mu = 2;

        var weights = new[] { 0.6, 0.4 };

        // First call — allocates buffers
        var a1 = new double[mu * n];
        for (var i = 0; i < a1.Length; i++) a1[i] = 0.5;
        var out1 = new double[n * n];
        provider!.ComputeRankMu(a1, weights, out1, mu, n);

        // Second call same dimensions — hits cache
        var a2 = new double[mu * n];
        for (var i = 0; i < a2.Length; i++) a2[i] = 0.3;
        var out2 = new double[n * n];
        provider.ComputeRankMu(a2, weights, out2, mu, n);

        Assert.True(out1.Any(v => Math.Abs(v) > 1e-10));
        Assert.True(out2.Any(v => Math.Abs(v) > 1e-10));
        provider.Dispose();
    }

    // ── GpuCmaEsProvider: Dispose paths ───────────────────────────────

    [GpuFact]
    public void GpuCmaEsProvider_Dispose_AllBuffersPopulated()
    {
        var provider = GpuCmaEsProvider.TryCreate(ComputeBackend.Gpu);
        Assert.NotNull(provider);

        const int n = 3;

        // Call SamplePopulation to populate _bdBuf, _zBuf, _meanBuf, _popOutputBuf
        var bd = new double[n * n];
        bd[0] = bd[4] = bd[8] = 1.0;
        var z = new double[2 * n];
        z[0] = 1;
        var mean = new double[n];
        var popOut = new double[2 * n];
        provider!.SamplePopulation(bd, z, mean, 1.0, popOut, 2, n);

        // Call ComputeRankMu to populate _artmpBuf, _weightsBuf, _rankMuBuf
        var artmp = new double[2 * n];
        artmp[0] = 1;
        var w = new[] { 0.6, 0.4 };
        var rankOut = new double[n * n];
        provider.ComputeRankMu(artmp, w, rankOut, 2, n);

        // Dispose should clean up all 7 buffers + accelerator + context
        provider.Dispose();
    }

    [GpuFact]
    public void GpuCmaEsProvider_Dispose_NoBuffersAllocated()
    {
        var provider = GpuCmaEsProvider.TryCreate(ComputeBackend.Gpu);
        Assert.NotNull(provider);

        // Dispose immediately — all buffer fields are null
        provider!.Dispose();
    }
}
