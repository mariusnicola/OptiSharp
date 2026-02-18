using System.Diagnostics;
using OptiSharp.Models;
using OptiSharp.Samplers.Tpe;

namespace OptiSharp.Tests;

[Trait("Category", "Slow")]
public sealed class LoadTests
{

    [SlowFact]
    public void FullRun_2000Trials_62Params_Sequential()
    {
        var space = TestHelpers.Create62ParamSpace();
        using var study = Optimizer.CreateStudy("load_seq", space,
            config: new TpeSamplerConfig { NStartupTrials = 50, Seed = 42 });

        var sw = Stopwatch.StartNew();
        for (var i = 0; i < 2000; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, i * 0.01);
        }
        sw.Stop();

        Assert.Equal(2000, study.Trials.Count);
        Assert.NotNull(study.BestTrial);
        Assert.True(sw.Elapsed.TotalSeconds < 30,
            $"Full 2000 sequential: {sw.Elapsed.TotalSeconds:F1}s — expected < 30s");
    }

    [SlowFact]
    public void FullRun_2000Trials_WavePattern()
    {
        var space = TestHelpers.Create62ParamSpace();
        using var study = Optimizer.CreateStudy("load_wave", space,
            config: new TpeSamplerConfig { NStartupTrials = 50, Seed = 42 });

        var sw = Stopwatch.StartNew();
        const int waves = 50;
        const int batchSize = 40;

        for (var wave = 0; wave < waves; wave++)
        {
            var batch = study.AskBatch(batchSize);
            var results = batch.Select((t, idx) =>
                new TrialResult(t.Number, (wave * batchSize + idx) * 0.01, TrialState.Complete)).ToList();
            study.TellBatch(results);
        }
        sw.Stop();

        Assert.Equal(waves * batchSize, study.Trials.Count);
        Assert.NotNull(study.BestTrial);
        Assert.True(sw.Elapsed.TotalSeconds < 30,
            $"Wave 2000: {sw.Elapsed.TotalSeconds:F1}s — expected < 30s");
    }

    [SlowFact]
    public async Task FullRun_500Trials_ConcurrentAskTell()
    {
        var space = TestHelpers.Create62ParamSpace();
        using var study = Optimizer.CreateStudy("load_concurrent", space,
            config: new TpeSamplerConfig { NStartupTrials = 50, Seed = 42 });

        const int totalTrials = 500;
        var told = 0;
        var exceptions = new List<Exception>();

        var sw = Stopwatch.StartNew();

        // 4 threads: ask and tell concurrently
        var threads = Enumerable.Range(0, 4).Select(threadId => Task.Run(() =>
        {
            try
            {
                for (var i = 0; i < totalTrials / 4; i++)
                {
                    var trial = study.Ask();
                    // Simulate some work
                    Thread.SpinWait(100);
                    study.Tell(trial.Number, trial.Number * 0.1);
                    Interlocked.Increment(ref told);
                }
            }
            catch (Exception ex)
            {
                lock (exceptions) exceptions.Add(ex);
            }
        })).ToArray();

        await Task.WhenAll(threads);
        sw.Stop();

        Assert.Empty(exceptions);
        Assert.Equal(totalTrials, told);
        Assert.Equal(totalTrials, study.Trials.Count);
        Assert.True(study.Trials.All(t => t.State == TrialState.Complete));
        Assert.True(sw.Elapsed.TotalSeconds < 30,
            $"Concurrent 500: {sw.Elapsed.TotalSeconds:F1}s — expected < 30s");
    }

    [SlowFact]
    public void FullRun_2000Trials_MemoryStable()
    {
        var space = TestHelpers.Create62ParamSpace();

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        using var study = Optimizer.CreateStudy("load_mem", space,
            config: new TpeSamplerConfig { NStartupTrials = 50, Seed = 42 });

        // Run 100 trials and measure baseline
        for (var i = 0; i < 100; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, i * 0.01);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        var memAt100 = GC.GetTotalMemory(true);

        // Run remaining 1900 trials
        for (var i = 100; i < 2000; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, i * 0.01);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        var memAt2000 = GC.GetTotalMemory(true);

        // Memory should grow sub-linearly — not more than 5x the baseline
        var ratio = (double)memAt2000 / memAt100;
        Assert.True(ratio < 5.0,
            $"Memory ratio: {ratio:F2}x (at 100: {memAt100 / 1024.0:F0}KB, at 2000: {memAt2000 / 1024.0:F0}KB) — expected < 5x");
    }

    [SlowFact]
    public void Sustained_10Runs_BackToBack()
    {
        var space = TestHelpers.Create62ParamSpace();
        var runTimes = new double[10];

        for (var run = 0; run < 10; run++)
        {
            using var study = Optimizer.CreateStudy($"sustained_{run}", space,
                config: new TpeSamplerConfig { NStartupTrials = 20, Seed = run });

            var sw = Stopwatch.StartNew();
            for (var i = 0; i < 200; i++)
            {
                var trial = study.Ask();
                study.Tell(trial.Number, i * 0.01);
            }
            sw.Stop();
            runTimes[run] = sw.Elapsed.TotalMilliseconds;
        }

        // Last run should be within 3x of first run (no degradation)
        var ratio = runTimes[9] / runTimes[0];
        Assert.True(ratio < 3.0,
            $"Run 10 vs Run 1: {ratio:F2}x ({runTimes[9]:F1}ms vs {runTimes[0]:F1}ms) — expected < 3x");
    }

    [SlowFact]
    public void HotPath_AskBatch40_100Iterations()
    {
        var space = TestHelpers.Create62ParamSpace();
        var rng = new Random(42);

        // Pre-fill 500 trials
        using var study = Optimizer.CreateStudy("hotpath", space,
            config: new TpeSamplerConfig { NStartupTrials = 50, Seed = 42 });

        for (var i = 0; i < 500; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, rng.NextDouble() * 100);
        }

        // Run AskBatch(40) 100 times
        var latencies = new double[100];
        for (var iter = 0; iter < 100; iter++)
        {
            var sw = Stopwatch.StartNew();
            var batch = study.AskBatch(40);
            sw.Stop();
            latencies[iter] = sw.Elapsed.TotalMilliseconds;

            // Tell results so next iteration has more data
            var results = batch.Select(t =>
                new TrialResult(t.Number, rng.NextDouble() * 100, TrialState.Complete)).ToList();
            study.TellBatch(results);
        }

        Array.Sort(latencies);
        var mean = latencies.Average();
        var p99 = latencies[98]; // 99th percentile

        Assert.True(mean < 500,
            $"AskBatch(40) mean: {mean:F1}ms — expected < 500ms");
        Assert.True(p99 < 2000,
            $"AskBatch(40) p99: {p99:F1}ms — expected < 2000ms");
    }

    [SlowFact]
    public void MixedLoad_AskAndTellInterleaved()
    {
        var space = TestHelpers.Create62ParamSpace();
        using var study = Optimizer.CreateStudy("mixed", space,
            config: new TpeSamplerConfig { NStartupTrials = 50, Seed = 42 });

        const int totalTrials = 1000;
        var sw = Stopwatch.StartNew();

        for (var i = 0; i < totalTrials; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, i * 0.01);
        }
        sw.Stop();

        var trialsPerSecond = totalTrials / sw.Elapsed.TotalSeconds;
        Assert.True(trialsPerSecond > 10,
            $"Throughput: {trialsPerSecond:F0} trials/s — expected > 10 trials/s");
        Assert.Equal(totalTrials, study.Trials.Count);
        Assert.True(study.Trials.All(t => t.State == TrialState.Complete));
    }
}
