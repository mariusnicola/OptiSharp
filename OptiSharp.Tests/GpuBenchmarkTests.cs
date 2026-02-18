using System.Diagnostics;
using OptiSharp.Models;
using OptiSharp.Samplers.CmaEs;
using Xunit.Abstractions;

namespace OptiSharp.Tests;

/// <summary>
/// CPU vs GPU (CUDA) benchmarks for CMA-ES.
/// Measures wall-clock time at various dimensions to identify the crossover point
/// where GPU becomes faster than CPU.
///
/// Results guide: use these benchmark outputs to decide CPU vs GPU for your search space size.
/// </summary>
public sealed class GpuBenchmarkTests(ITestOutputHelper output)
{
    // ----------------------------------------------------------------
    // Core benchmark: CPU vs GPU at different dimension counts
    // ----------------------------------------------------------------

    [Fact]
    public void CpuVsGpu_DimensionScaling()
    {
        var gpuAvailable = CheckGpuAvailable();
        var dims = new[] { 10, 20, 50, 100, 200, 500 };
        const int trialsPerConfig = 200;

        output.WriteLine("╔══════════════════════════════════════════════════════════════════════════╗");
        output.WriteLine("║              CMA-ES: CPU vs GPU Dimension Scaling                       ║");
        output.WriteLine("║              Sphere function, 200 trials per config                     ║");
        output.WriteLine("╚══════════════════════════════════════════════════════════════════════════╝");
        output.WriteLine("");

        if (!gpuAvailable)
        {
            output.WriteLine("⚠ CUDA GPU not detected — showing CPU-only results.");
            output.WriteLine("  To enable GPU comparison, ensure NVIDIA CUDA drivers are installed.");
            output.WriteLine("");
        }

        output.WriteLine($"{"Dims",-6} {"CPU ms/ask",-14} {"GPU ms/ask",-14} {"Speedup",-12} {"CPU best",-14} {"GPU best",-14} {"Verdict",-20}");
        output.WriteLine(new string('-', 94));

        foreach (var d in dims)
        {
            var space = TestHelpers.MakeSpace(d);

            // CPU baseline
            var (cpuBest, cpuMs) = RunCmaEsWithTiming(space, d, trialsPerConfig, ComputeBackend.Cpu);

            // GPU (if available)
            double gpuMs = -1;
            double gpuBest = -1;
            string verdict;

            if (gpuAvailable)
            {
                (gpuBest, gpuMs) = RunCmaEsWithTiming(space, d, trialsPerConfig, ComputeBackend.Gpu);
                var speedup = cpuMs / gpuMs;

                verdict = speedup > 1.2 ? $"GPU {speedup:F1}x faster"
                    : speedup < 0.8 ? $"CPU {1.0 / speedup:F1}x faster"
                    : "~same";

                output.WriteLine($"{d,-6} {cpuMs,-14:F3} {gpuMs,-14:F3} {speedup,-12:F2}x {cpuBest,-14:F4} {gpuBest,-14:F4} {verdict,-20}");
            }
            else
            {
                output.WriteLine($"{d,-6} {cpuMs,-14:F3} {"N/A",-14} {"N/A",-12} {cpuBest,-14:F4} {"N/A",-14} {"GPU unavailable",-20}");
            }
        }

        output.WriteLine("");
        output.WriteLine("RECOMMENDATION:");
        output.WriteLine($"  GPU recommended for N >= {CmaEsSampler.GpuRecommendedMinDimensions} continuous parameters.");
        output.WriteLine("  Below that threshold, CPU is faster due to GPU kernel launch + PCIe transfer overhead.");
        output.WriteLine("  The exact crossover depends on your GPU (kernel launch ~10-50μs, PCIe ~12 GB/s).");
    }

    // ----------------------------------------------------------------
    // GPU convergence correctness — same results as CPU
    // ----------------------------------------------------------------

    [Fact]
    public void Gpu_ConvergenceCorrectness_Sphere20D()
    {
        var gpuAvailable = CheckGpuAvailable();
        if (!gpuAvailable)
        {
            output.WriteLine("CUDA GPU not detected — skipping GPU correctness test.");
            return;
        }

        var space = TestHelpers.MakeSpace(20);
        const int trials = 200;
        const int runs = 5;

        var cpuBests = new double[runs];
        var gpuBests = new double[runs];

        for (var run = 0; run < runs; run++)
        {
            var seed = run * 100;
            cpuBests[run] = RunCmaEsOpt(space, 20, trials, ComputeBackend.Cpu, seed);
            gpuBests[run] = RunCmaEsOpt(space, 20, trials, ComputeBackend.Gpu, seed);
        }

        output.WriteLine("=== GPU Convergence Correctness — Sphere 20D ===");
        output.WriteLine($"CPU median best: {TestHelpers.Median(cpuBests):F4}");
        output.WriteLine($"GPU median best: {TestHelpers.Median(gpuBests):F4}");

        // GPU should converge comparably (not exact same due to floating point order)
        var cpuMedian = TestHelpers.Median(cpuBests);
        var gpuMedian = TestHelpers.Median(gpuBests);
        Assert.True(gpuMedian < cpuMedian * 5,
            $"GPU convergence ({gpuMedian:F4}) should be comparable to CPU ({cpuMedian:F4})");
    }

    // ----------------------------------------------------------------
    // GPU warning test
    // ----------------------------------------------------------------

    [Fact]
    public void Gpu_DimensionWarning_SmallSpace()
    {
        var gpuAvailable = CheckGpuAvailable();
        if (!gpuAvailable)
        {
            output.WriteLine("CUDA GPU not detected — skipping GPU warning test.");
            return;
        }

        var space = TestHelpers.MakeSpace(10); // well below recommended threshold
        var config = new CmaEsSamplerConfig { Seed = 42, Backend = ComputeBackend.Gpu };
        using var study = Optimizer.CreateStudyWithCmaEs("gpu_warn_test", space, config: config);

        // Ask once to trigger initialization
        var trial = study.Ask();
        study.Tell(trial.Number, 1.0);

        // Check that sampler emits a warning
        var sampler = TestHelpers.GetSamplerFromStudy(study);
        Assert.NotNull(sampler);

        output.WriteLine($"GPU active: {sampler.IsGpuActive}");
        output.WriteLine($"Device: {sampler.DeviceName}");
        output.WriteLine($"Warning: {sampler.GpuDimensionWarning ?? "(none)"}");

        if (sampler.IsGpuActive)
        {
            Assert.NotNull(sampler.GpuDimensionWarning);
            Assert.Contains("recommended minimum", sampler.GpuDimensionWarning, StringComparison.OrdinalIgnoreCase);
        }
    }

    // ----------------------------------------------------------------
    // Generation-level GPU timing (isolates kernel time from study overhead)
    // ----------------------------------------------------------------

    [Fact]
    public void Gpu_GenerationTiming()
    {
        var gpuAvailable = CheckGpuAvailable();

        var dims = new[] { 20, 50, 100, 200, 500 };
        const int generations = 20;

        output.WriteLine("╔══════════════════════════════════════════════════════════════════╗");
        output.WriteLine("║           CMA-ES Generation Timing (CPU vs GPU)                 ║");
        output.WriteLine("║           Measures Ask+Tell per full generation cycle            ║");
        output.WriteLine("╚══════════════════════════════════════════════════════════════════╝");
        output.WriteLine("");
        output.WriteLine($"{"Dims",-6} {"CPU ms/gen",-14} {"GPU ms/gen",-14} {"Speedup",-12}");
        output.WriteLine(new string('-', 46));

        foreach (var d in dims)
        {
            var space = TestHelpers.MakeSpace(d);

            var cpuGenMs = MeasureGenerationTime(space, d, generations, ComputeBackend.Cpu);

            if (gpuAvailable)
            {
                var gpuGenMs = MeasureGenerationTime(space, d, generations, ComputeBackend.Gpu);
                var speedup = cpuGenMs / gpuGenMs;
                output.WriteLine($"{d,-6} {cpuGenMs,-14:F3} {gpuGenMs,-14:F3} {speedup,-12:F2}x");
            }
            else
            {
                output.WriteLine($"{d,-6} {cpuGenMs,-14:F3} {"N/A",-14} {"N/A",-12}");
            }
        }

        if (!gpuAvailable)
        {
            output.WriteLine("");
            output.WriteLine("⚠ CUDA GPU not detected — GPU columns show N/A.");
        }
    }

    // ----------------------------------------------------------------
    // Memory benchmark with GPU
    // ----------------------------------------------------------------

    [Fact]
    public void Gpu_Memory_1000Trials_100Params()
    {
        var gpuAvailable = CheckGpuAvailable();
        if (!gpuAvailable)
        {
            output.WriteLine("CUDA GPU not detected — skipping GPU memory test.");
            return;
        }

        var space = TestHelpers.MakeSpace(100);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        var memBefore = GC.GetTotalMemory(true);

        using var study = Optimizer.CreateStudyWithCmaEs("gpu_mem", space,
            config: new CmaEsSamplerConfig { Seed = 42, Backend = ComputeBackend.Gpu });

        for (var i = 0; i < 1000; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, TestHelpers.Sphere(trial.Parameters, 100));
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        var memAfter = GC.GetTotalMemory(true);

        var usedMb = (memAfter - memBefore) / (1024.0 * 1024.0);
        output.WriteLine($"GPU CMA-ES memory (1000 trials, 100 params): {usedMb:F1} MB");
        Assert.True(usedMb < 200, $"GPU memory: {usedMb:F1}MB — expected < 200MB");
    }

    // ----------------------------------------------------------------
    // Summary reference card
    // ----------------------------------------------------------------

    [Fact]
    public void ReferenceCard_WhenToUseGpu()
    {
        output.WriteLine("╔══════════════════════════════════════════════════════════════════╗");
        output.WriteLine("║              CMA-ES Backend Selection Guide                     ║");
        output.WriteLine("╚══════════════════════════════════════════════════════════════════╝");
        output.WriteLine("");
        output.WriteLine("  Dimensions (N)  │  Recommended Backend  │  Reason");
        output.WriteLine("  ────────────────┼───────────────────────┼──────────────────────────────");
        output.WriteLine("  N < 50          │  CPU                  │  GPU overhead >> compute time");
        output.WriteLine("  50 <= N < 100   │  CPU (usually)        │  GPU might break even");
        output.WriteLine("  100 <= N < 200  │  Auto                 │  GPU starts to help");
        output.WriteLine("  N >= 200        │  GPU                  │  GPU clearly faster");
        output.WriteLine("  N >= 500        │  GPU                  │  GPU significantly faster");
        output.WriteLine("");
        output.WriteLine("  Config example:");
        output.WriteLine("    new CmaEsSamplerConfig { Backend = ComputeBackend.Auto }");
        output.WriteLine("");
        output.WriteLine("  Auto mode: tries CUDA first, falls back to CPU silently.");
        output.WriteLine("  GPU mode: throws if no CUDA device found.");
        output.WriteLine("");
        output.WriteLine($"  Warning threshold: N < {CmaEsSampler.GpuRecommendedMinDimensions} " +
                          "will emit GpuDimensionWarning on the sampler.");
        output.WriteLine("");
        output.WriteLine("  Key factors affecting crossover point:");
        output.WriteLine("    - GPU kernel launch overhead: ~10-50μs");
        output.WriteLine("    - PCIe transfer: ~12 GB/s (RTX 3090)");
        output.WriteLine("    - CMA-ES compute: O(lambda * N²) per generation");
        output.WriteLine("    - Eigendecomposition: O(N³) — always on CPU (MathNet.Numerics)");

        // Just ensure this test runs — it's a documentation test
        Assert.True(true);
    }

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    private static bool CheckGpuAvailable()
    {
        try
        {
            // Quick check without full ILGPU initialization
            var config = new CmaEsSamplerConfig { Backend = ComputeBackend.Auto };
            var sampler = new CmaEsSampler(config);
            var space = TestHelpers.MakeSpace(2);
            var trials = new List<Trial>();
            sampler.Sample(trials, StudyDirection.Minimize, space);
            var isGpu = sampler.IsGpuActive;
            sampler.Dispose();
            return isGpu;
        }
        catch
        {
            return false;
        }
    }

    private (double BestValue, double MsPerAsk) RunCmaEsWithTiming(
        SearchSpace space, int dims, int trials, ComputeBackend backend)
    {
        var config = new CmaEsSamplerConfig { Seed = 42, Backend = backend };
        using var study = Optimizer.CreateStudyWithCmaEs("bench", space, config: config);

        // Warmup: 1 generation
        var warmupTrial = study.Ask();
        study.Tell(warmupTrial.Number, TestHelpers.Sphere(warmupTrial.Parameters, dims));

        var sw = Stopwatch.StartNew();
        for (var i = 1; i < trials; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, TestHelpers.Sphere(trial.Parameters, dims));
        }
        sw.Stop();

        var best = study.BestTrial!.Value!.Value;
        var msPerAsk = sw.Elapsed.TotalMilliseconds / (trials - 1);
        return (best, msPerAsk);
    }

    private static double RunCmaEsOpt(
        SearchSpace space, int dims, int trials, ComputeBackend backend, int seed)
    {
        var config = new CmaEsSamplerConfig { Seed = seed, Backend = backend };
        using var study = Optimizer.CreateStudyWithCmaEs("opt", space, config: config);

        for (var i = 0; i < trials; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, TestHelpers.Sphere(trial.Parameters, dims));
        }

        return study.BestTrial!.Value!.Value;
    }

    private static double MeasureGenerationTime(
        SearchSpace space, int dims, int generations, ComputeBackend backend)
    {
        var config = new CmaEsSamplerConfig { Seed = 42, Backend = backend };
        using var study = Optimizer.CreateStudyWithCmaEs("gen_bench", space, config: config);

        // Warmup: 1 full generation
        var lambda = config.PopulationSize ?? 4 + (int)Math.Floor(3.0 * Math.Log(dims));
        for (var i = 0; i < lambda; i++)
        {
            var t = study.Ask();
            study.Tell(t.Number, TestHelpers.Sphere(t.Parameters, dims));
        }

        // Measure
        var sw = Stopwatch.StartNew();
        for (var gen = 0; gen < generations; gen++)
        {
            for (var i = 0; i < lambda; i++)
            {
                var t = study.Ask();
                study.Tell(t.Number, TestHelpers.Sphere(t.Parameters, dims));
            }
        }
        sw.Stop();

        return sw.Elapsed.TotalMilliseconds / generations;
    }

}
