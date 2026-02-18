using System.Diagnostics;
using OptiSharp.Models;
using OptiSharp.Samplers;
using OptiSharp.Samplers.CmaEs;
using OptiSharp.Samplers.Tpe;
using Xunit.Abstractions;

namespace OptiSharp.Tests;

/// <summary>
/// Head-to-head benchmarks: TPE vs CMA-ES vs Random.
/// All tests use standard optimization test functions — no external dependency.
/// Output goes to ITestOutputHelper for human-readable comparison tables.
/// </summary>
public sealed class SamplerBenchmarkTests(ITestOutputHelper output)
{
    // --- Convergence benchmarks ---

    [Fact]
    public void Convergence_Sphere5D()
    {
        var space = TestHelpers.MakeSpace(5);
        const int trials = 200;
        const int runs = 8;

        var tpeWins = 0;
        var cmaWins = 0;
        var tpeBests = new double[runs];
        var cmaBests = new double[runs];
        var rndBests = new double[runs];

        for (var run = 0; run < runs; run++)
        {
            var seed = run * 100;
            tpeBests[run] = RunOpt(new TpeSampler(new TpeSamplerConfig { NStartupTrials = 15, Seed = seed }),
                space, trials, p => TestHelpers.Sphere(p, 5));
            cmaBests[run] = RunOpt(new CmaEsSampler(new CmaEsSamplerConfig { Seed = seed }),
                space, trials, p => TestHelpers.Sphere(p, 5));
            rndBests[run] = RunOpt(new RandomSampler(seed), space, trials, p => TestHelpers.Sphere(p, 5));

            if (tpeBests[run] < rndBests[run]) tpeWins++;
            if (cmaBests[run] < rndBests[run]) cmaWins++;
        }

        output.WriteLine("=== Sphere 5D (200 trials, 8 runs) ===");
        output.WriteLine($"TPE    median best: {TestHelpers.Median(tpeBests):F4}  wins vs random: {tpeWins}/{runs}");
        output.WriteLine($"CMA-ES median best: {TestHelpers.Median(cmaBests):F4}  wins vs random: {cmaWins}/{runs}");
        output.WriteLine($"Random median best: {TestHelpers.Median(rndBests):F4}");

        Assert.True(cmaWins >= 4, $"CMA-ES won {cmaWins}/{runs} vs random — expected >= 4");
        Assert.True(tpeWins >= 4, $"TPE won {tpeWins}/{runs} vs random — expected >= 4");
    }

    [Fact]
    public void Convergence_Rosenbrock2D()
    {
        var space = TestHelpers.MakeSpace(2);
        const int trials = 200;
        const int runs = 8;

        var tpeWins = 0;
        var cmaWins = 0;
        var tpeBests = new double[runs];
        var cmaBests = new double[runs];
        var rndBests = new double[runs];

        for (var run = 0; run < runs; run++)
        {
            var seed = run * 100;
            tpeBests[run] = RunOpt(new TpeSampler(new TpeSamplerConfig { NStartupTrials = 15, Seed = seed }),
                space, trials, TestHelpers.Rosenbrock);
            cmaBests[run] = RunOpt(new CmaEsSampler(new CmaEsSamplerConfig { Seed = seed }),
                space, trials, TestHelpers.Rosenbrock);
            rndBests[run] = RunOpt(new RandomSampler(seed), space, trials, TestHelpers.Rosenbrock);

            if (tpeBests[run] < rndBests[run]) tpeWins++;
            if (cmaBests[run] < rndBests[run]) cmaWins++;
        }

        output.WriteLine("=== Rosenbrock 2D (200 trials, 8 runs) ===");
        output.WriteLine($"TPE    median best: {TestHelpers.Median(tpeBests):F4}  wins vs random: {tpeWins}/{runs}");
        output.WriteLine($"CMA-ES median best: {TestHelpers.Median(cmaBests):F4}  wins vs random: {cmaWins}/{runs}");
        output.WriteLine($"Random median best: {TestHelpers.Median(rndBests):F4}");

        Assert.True(cmaWins >= 4, $"CMA-ES won {cmaWins}/{runs} vs random — expected >= 4");
    }

    [Fact]
    public void Convergence_Rastrigin5D()
    {
        var space = TestHelpers.MakeSpace(5, -5.12, 5.12);
        const int trials = 300;
        const int runs = 8;

        var tpeBests = new double[runs];
        var cmaBests = new double[runs];
        var rndBests = new double[runs];

        for (var run = 0; run < runs; run++)
        {
            var seed = run * 100;
            tpeBests[run] = RunOpt(new TpeSampler(new TpeSamplerConfig { NStartupTrials = 20, Seed = seed }),
                space, trials, p => TestHelpers.Rastrigin(p, 5));
            cmaBests[run] = RunOpt(new CmaEsSampler(new CmaEsSamplerConfig { Seed = seed }),
                space, trials, p => TestHelpers.Rastrigin(p, 5));
            rndBests[run] = RunOpt(new RandomSampler(seed), space, trials, p => TestHelpers.Rastrigin(p, 5));
        }

        output.WriteLine("=== Rastrigin 5D (300 trials, 8 runs) ===");
        output.WriteLine($"TPE    median best: {TestHelpers.Median(tpeBests):F4}");
        output.WriteLine($"CMA-ES median best: {TestHelpers.Median(cmaBests):F4}");
        output.WriteLine($"Random median best: {TestHelpers.Median(rndBests):F4}");

        // Rastrigin is multimodal — neither TPE nor CMA-ES guaranteed to dominate
        // Just report results, no strict assertion
        Assert.True(TestHelpers.Median(cmaBests) < 200, "CMA-ES should find something reasonable");
    }

    [Fact]
    public void Convergence_Ackley5D()
    {
        var space = TestHelpers.MakeSpace(5);
        const int trials = 200;
        const int runs = 8;

        var tpeBests = new double[runs];
        var cmaBests = new double[runs];
        var rndBests = new double[runs];

        for (var run = 0; run < runs; run++)
        {
            var seed = run * 100;
            tpeBests[run] = RunOpt(new TpeSampler(new TpeSamplerConfig { NStartupTrials = 15, Seed = seed }),
                space, trials, p => TestHelpers.Ackley(p, 5));
            cmaBests[run] = RunOpt(new CmaEsSampler(new CmaEsSamplerConfig { Seed = seed }),
                space, trials, p => TestHelpers.Ackley(p, 5));
            rndBests[run] = RunOpt(new RandomSampler(seed), space, trials, p => TestHelpers.Ackley(p, 5));
        }

        output.WriteLine("=== Ackley 5D (200 trials, 8 runs) ===");
        output.WriteLine($"TPE    median best: {TestHelpers.Median(tpeBests):F4}");
        output.WriteLine($"CMA-ES median best: {TestHelpers.Median(cmaBests):F4}");
        output.WriteLine($"Random median best: {TestHelpers.Median(rndBests):F4}");
    }

    // --- Wall-clock time benchmarks ---

    [Fact]
    public void WallClock_Ask_10Params()
    {
        var space = TestHelpers.MakeSpace(10);
        const int prefill = 100;
        const int measure = 50;

        var tpeMs = MeasureAskTime(
            () => new TpeSampler(new TpeSamplerConfig { NStartupTrials = 10, Seed = 42 }),
            space, prefill, measure);
        var cmaMs = MeasureAskTime(
            () => new CmaEsSampler(new CmaEsSamplerConfig { Seed = 42 }),
            space, prefill, measure);

        output.WriteLine("=== Ask() latency — 10 params, 100 prior trials ===");
        output.WriteLine($"TPE:    {tpeMs:F3} ms/ask");
        output.WriteLine($"CMA-ES: {cmaMs:F3} ms/ask");

        Assert.True(tpeMs < 50, $"TPE: {tpeMs:F2}ms — expected < 50ms");
        Assert.True(cmaMs < 50, $"CMA-ES: {cmaMs:F2}ms — expected < 50ms");
    }

    [Fact]
    public void WallClock_Ask_50Params()
    {
        var space = TestHelpers.MakeSpace(50);
        const int prefill = 300;
        const int measure = 30;

        var tpeMs = MeasureAskTime(
            () => new TpeSampler(new TpeSamplerConfig { NStartupTrials = 10, Seed = 42 }),
            space, prefill, measure);
        var cmaMs = MeasureAskTime(
            () => new CmaEsSampler(new CmaEsSamplerConfig { Seed = 42 }),
            space, prefill, measure);

        output.WriteLine("=== Ask() latency — 50 params, 300 prior trials ===");
        output.WriteLine($"TPE:    {tpeMs:F3} ms/ask");
        output.WriteLine($"CMA-ES: {cmaMs:F3} ms/ask");

        Assert.True(tpeMs < 100, $"TPE: {tpeMs:F2}ms — expected < 100ms");
        Assert.True(cmaMs < 100, $"CMA-ES: {cmaMs:F2}ms — expected < 100ms");
    }

    // --- Dimension scaling ---

    [Fact]
    public void Scaling_Dimensions()
    {
        var dims = new[] { 5, 10, 20, 50 };
        const int trials = 200;

        output.WriteLine("=== Dimension scaling (Sphere, 200 trials) ===");
        output.WriteLine($"{"Dims",-6} {"TPE best",-14} {"CMA best",-14} {"TPE ms/ask",-14} {"CMA ms/ask",-14}");

        foreach (var d in dims)
        {
            var space = TestHelpers.MakeSpace(d);

            var (tpeBest, tpeMs) = RunWithTiming(
                new TpeSampler(new TpeSamplerConfig { NStartupTrials = Math.Min(20, trials / 2), Seed = 42 }),
                space, trials, p => TestHelpers.Sphere(p, d));
            var (cmaBest, cmaMs) = RunWithTiming(
                new CmaEsSampler(new CmaEsSamplerConfig { Seed = 42 }),
                space, trials, p => TestHelpers.Sphere(p, d));

            output.WriteLine($"{d,-6} {tpeBest,-14:F4} {cmaBest,-14:F4} {tpeMs,-14:F3} {cmaMs,-14:F3}");
        }
    }

    // --- Trial scaling ---

    [Fact]
    public void Scaling_Trials()
    {
        var trialCounts = new[] { 50, 100, 200, 500 };
        const int dims = 20;
        var space = TestHelpers.MakeSpace(dims);

        output.WriteLine("=== Trial scaling (Sphere 20D) ===");
        output.WriteLine($"{"Trials",-8} {"TPE best",-14} {"CMA best",-14} {"TPE ms/ask",-14} {"CMA ms/ask",-14}");

        foreach (var n in trialCounts)
        {
            var (tpeBest, tpeMs) = RunWithTiming(
                new TpeSampler(new TpeSamplerConfig { NStartupTrials = Math.Min(20, n / 2), Seed = 42 }),
                space, n, p => TestHelpers.Sphere(p, dims));
            var (cmaBest, cmaMs) = RunWithTiming(
                new CmaEsSampler(new CmaEsSamplerConfig { Seed = 42 }),
                space, n, p => TestHelpers.Sphere(p, dims));

            output.WriteLine($"{n,-8} {tpeBest,-14:F4} {cmaBest,-14:F4} {tpeMs,-14:F3} {cmaMs,-14:F3}");
        }
    }

    // --- Memory ---

    [Fact]
    public void Memory_CmaEs_2000Trials_20Params()
    {
        var space = TestHelpers.MakeSpace(20);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        var memBefore = GC.GetTotalMemory(true);

        using var study = Optimizer.CreateStudyWithCmaEs("mem_bench", space,
            config: new CmaEsSamplerConfig { Seed = 42 });

        for (var i = 0; i < 2000; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, i * 0.01);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        var memAfter = GC.GetTotalMemory(true);

        var usedMb = (memAfter - memBefore) / (1024.0 * 1024.0);
        output.WriteLine($"CMA-ES memory (2000 trials, 20 params): {usedMb:F1} MB");
        Assert.True(usedMb < 100, $"Memory: {usedMb:F1}MB — expected < 100MB");
    }

    // --- Helpers ---

    private static double RunOpt(
        ISampler sampler, SearchSpace space, int trials,
        Func<IReadOnlyDictionary<string, object>, double> objective)
    {
        var trialList = new List<Trial>();
        var bestValue = double.MaxValue;

        for (var i = 0; i < trials; i++)
        {
            var parameters = sampler.Sample(trialList, StudyDirection.Minimize, space);
            var value = objective(parameters);
            var trial = new Trial(i, parameters) { State = TrialState.Complete, Value = value };
            trialList.Add(trial);
            if (value < bestValue) bestValue = value;
        }

        return bestValue;
    }

    private static (double BestValue, double MsPerAsk) RunWithTiming(
        ISampler sampler, SearchSpace space, int trials,
        Func<IReadOnlyDictionary<string, object>, double> objective)
    {
        var trialList = new List<Trial>();
        var bestValue = double.MaxValue;

        var sw = Stopwatch.StartNew();
        for (var i = 0; i < trials; i++)
        {
            var parameters = sampler.Sample(trialList, StudyDirection.Minimize, space);
            var value = objective(parameters);
            var trial = new Trial(i, parameters) { State = TrialState.Complete, Value = value };
            trialList.Add(trial);
            if (value < bestValue) bestValue = value;
        }
        sw.Stop();

        return (bestValue, sw.Elapsed.TotalMilliseconds / trials);
    }

    private static double MeasureAskTime(
        Func<ISampler> createSampler, SearchSpace space, int prefill, int measure)
    {
        var sampler = createSampler();
        var trials = new List<Trial>();
        var rng = new Random(42);

        // Prefill
        for (var i = 0; i < prefill; i++)
        {
            var parameters = sampler.Sample(trials, StudyDirection.Minimize, space);
            var trial = new Trial(i, parameters) { State = TrialState.Complete, Value = rng.NextDouble() * 100 };
            trials.Add(trial);
        }

        // Measure
        var sw = Stopwatch.StartNew();
        for (var i = 0; i < measure; i++)
        {
            var parameters = sampler.Sample(trials, StudyDirection.Minimize, space);
            var trial = new Trial(prefill + i, parameters) { State = TrialState.Complete, Value = rng.NextDouble() * 100 };
            trials.Add(trial);
        }
        sw.Stop();

        return sw.Elapsed.TotalMilliseconds / measure;
    }

}
