using System.Diagnostics;
using OptiSharp.Models;
using OptiSharp.Samplers.Tpe;

namespace OptiSharp.Tests;

[Trait("Category", "Slow")]
public sealed class PerformanceBenchmarkTests
{

    private static Study PreFillStudy(SearchSpace space, int trialCount, int seed = 42)
    {
        var study = Optimizer.CreateStudy("bench", space,
            config: new TpeSamplerConfig { NStartupTrials = 10, Seed = seed });

        var rng = new Random(seed);
        for (var i = 0; i < trialCount; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, rng.NextDouble() * 100);
        }
        return study;
    }

    [SlowFact]
    public void Ask_StartupPhase_Fast()
    {
        var space = TestHelpers.Create62ParamSpace();
        using var study = Optimizer.CreateStudy("bench", space,
            config: new TpeSamplerConfig { NStartupTrials = 100, Seed = 42 });

        // Warmup
        for (var i = 0; i < 5; i++)
        {
            var t = study.Ask();
            study.Tell(t.Number, i * 1.0);
        }

        var sw = Stopwatch.StartNew();
        var count = 1000;
        for (var i = 0; i < count; i++)
        {
            var t = study.Ask();
            study.Tell(t.Number, i * 1.0);
        }
        sw.Stop();

        var msPerAsk = sw.Elapsed.TotalMilliseconds / count;
        Assert.True(msPerAsk < 0.1, $"Startup ask: {msPerAsk:F4}ms/ask — expected < 0.1ms");
    }

    [SlowFact]
    public void Ask_100Trials_62Params()
    {
        var space = TestHelpers.Create62ParamSpace();
        using var study = PreFillStudy(space, 100);

        var sw = Stopwatch.StartNew();
        var count = 100;
        for (var i = 0; i < count; i++)
        {
            var t = study.Ask();
            study.Tell(t.Number, t.Number * 1.0);
        }
        sw.Stop();

        var msPerAsk = sw.Elapsed.TotalMilliseconds / count;
        Assert.True(msPerAsk < 5, $"100-trial ask: {msPerAsk:F2}ms — expected < 5ms");
    }

    [SlowFact]
    public void Ask_500Trials_62Params()
    {
        var space = TestHelpers.Create62ParamSpace();
        using var study = PreFillStudy(space, 500);

        var sw = Stopwatch.StartNew();
        var count = 50;
        for (var i = 0; i < count; i++)
        {
            var t = study.Ask();
            study.Tell(t.Number, t.Number * 1.0);
        }
        sw.Stop();

        var msPerAsk = sw.Elapsed.TotalMilliseconds / count;
        Assert.True(msPerAsk < 20, $"500-trial ask: {msPerAsk:F2}ms — expected < 20ms");
    }

    [SlowFact]
    public void AskBatch40_500Trials()
    {
        var space = TestHelpers.Create62ParamSpace();
        using var study = PreFillStudy(space, 500);

        var sw = Stopwatch.StartNew();
        var batch = study.AskBatch(40);
        sw.Stop();

        Assert.Equal(40, batch.Count);
        Assert.True(sw.Elapsed.TotalMilliseconds < 500,
            $"AskBatch(40): {sw.Elapsed.TotalMilliseconds:F1}ms — expected < 500ms");
    }

    [SlowFact]
    public void Tell_Single_Trivial()
    {
        var space = TestHelpers.Create62ParamSpace();
        using var study = Optimizer.CreateStudy("bench", space,
            config: new TpeSamplerConfig { Seed = 42 });

        var trials = study.AskBatch(1000);

        var sw = Stopwatch.StartNew();
        foreach (var t in trials)
            study.Tell(t.Number, t.Number * 1.0);
        sw.Stop();

        var msPerTell = sw.Elapsed.TotalMilliseconds / trials.Count;
        Assert.True(msPerTell < 0.05, $"Tell: {msPerTell:F4}ms — expected < 0.05ms");
    }

    [SlowFact]
    public void TellBatch200_Fast()
    {
        var space = TestHelpers.Create62ParamSpace();
        using var study = Optimizer.CreateStudy("bench", space,
            config: new TpeSamplerConfig { Seed = 42 });

        var trials = study.AskBatch(200);
        var results = trials.Select((t, i) =>
            new TrialResult(t.Number, i * 1.0, TrialState.Complete)).ToList();

        var sw = Stopwatch.StartNew();
        study.TellBatch(results);
        sw.Stop();

        Assert.True(sw.Elapsed.TotalMilliseconds < 1,
            $"TellBatch(200): {sw.Elapsed.TotalMilliseconds:F3}ms — expected < 1ms");
    }

    [SlowFact]
    public void Memory_2000Trials_62Params()
    {
        var space = TestHelpers.Create62ParamSpace();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        var memBefore = GC.GetTotalMemory(true);

        using var study = Optimizer.CreateStudy("bench", space,
            config: new TpeSamplerConfig { NStartupTrials = 100, Seed = 42 });

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
        Assert.True(usedMb < 50, $"Memory: {usedMb:F1}MB — expected < 50MB");
    }
}
