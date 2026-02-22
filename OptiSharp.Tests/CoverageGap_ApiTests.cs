using System.Collections;
using OptiSharp.Models;
using OptiSharp.Samplers;
using OptiSharp.Samplers.CmaEs;
using OptiSharp.Samplers.Tpe;

namespace OptiSharp.Tests;

/// <summary>
/// Coverage tests for Optimizer factory, SearchSpace, Study edge cases, and record semantics.
/// </summary>
public sealed class CoverageGap_ApiTests
{
    // ── Optimizer.cs ──────────────────────────────────────────────────

    [Fact]
    public void Optimizer_CreateStudy_WithCustomSampler()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        var sampler = new RandomSampler(seed: 1);

        using var study = Optimizer.CreateStudy("custom", space, sampler);

        var trial = study.Ask();
        Assert.True(trial.Parameters.ContainsKey("x"));
        Assert.Equal(StudyDirection.Minimize, study.Direction);
    }

    [Fact]
    public void Optimizer_CreateStudy_WithCustomSampler_Maximize()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        var sampler = new RandomSampler(seed: 1);

        using var study = Optimizer.CreateStudy("custom_max", space, sampler, StudyDirection.Maximize);

        Assert.Equal(StudyDirection.Maximize, study.Direction);
    }

    [Fact]
    public void Optimizer_CreateStudyWithRandomSampler_Maximize()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        using var study = Optimizer.CreateStudyWithRandomSampler("test", space,
            direction: StudyDirection.Maximize, seed: 42);

        Assert.Equal(StudyDirection.Maximize, study.Direction);
        var trial = study.Ask();
        Assert.True(trial.Parameters.ContainsKey("x"));
    }

    [Fact]
    public void Optimizer_CreateStudyWithCmaEs_Maximize()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        using var study = Optimizer.CreateStudyWithCmaEs("test", space,
            direction: StudyDirection.Maximize,
            config: new CmaEsSamplerConfig { Seed = 42 });

        Assert.Equal(StudyDirection.Maximize, study.Direction);
    }

    // ── SearchSpace / Models ──────────────────────────────────────────

    [Fact]
    public void SearchSpace_Contains_ReturnsTrueForExisting()
    {
        var space = new SearchSpace([
            new FloatRange("x", 0, 10),
            new IntRange("n", 1, 5)
        ]);

        Assert.True(space.Contains("x"));
        Assert.True(space.Contains("n"));
        Assert.False(space.Contains("nonexistent"));
    }

    [Fact]
    public void SearchSpace_StringIndexer_ReturnsCorrectRange()
    {
        var space = new SearchSpace([
            new FloatRange("x", 0, 10),
            new IntRange("n", 1, 5)
        ]);

        var range = space["x"];
        Assert.IsType<FloatRange>(range);
        Assert.Equal("x", range.Name);

        var intRange = space["n"];
        Assert.IsType<IntRange>(intRange);
    }

    [Fact]
    public void SearchSpace_StringIndexer_ThrowsForMissing()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);

        Assert.Throws<KeyNotFoundException>(() => space["missing"]);
    }

    [Fact]
    public void SearchSpace_NonGenericEnumerator_Works()
    {
        var space = new SearchSpace([
            new FloatRange("x", 0, 10),
            new IntRange("n", 1, 5)
        ]);

        // Force non-generic IEnumerable.GetEnumerator()
        IEnumerable enumerable = space;
        var count = 0;
        foreach (var item in enumerable)
        {
            Assert.IsAssignableFrom<ParameterRange>(item);
            count++;
        }
        Assert.Equal(2, count);
    }

    [Fact]
    public void SearchSpace_IntIndexer_ReturnsCorrectRange()
    {
        var space = new SearchSpace([
            new FloatRange("x", 0, 10),
            new IntRange("n", 1, 5),
            new CategoricalRange("c", ["a", "b"])
        ]);

        Assert.Equal("x", space[0].Name);
        Assert.Equal("n", space[1].Name);
        Assert.Equal("c", space[2].Name);
        Assert.Equal(3, space.Count);
    }

    // ── Study.cs ──────────────────────────────────────────────────────

    [Fact]
    public void Study_AskBatch_ZeroCount_ReturnsEmpty()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        using var study = Optimizer.CreateStudy("test", space, config: new TpeSamplerConfig { Seed = 1 });

        var batch = study.AskBatch(0);

        Assert.Empty(batch);
    }

    [Fact]
    public void Study_AskBatch_NegativeCount_ReturnsEmpty()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        using var study = Optimizer.CreateStudy("test", space, config: new TpeSamplerConfig { Seed = 1 });

        var batch = study.AskBatch(-1);

        Assert.Empty(batch);
    }

    [Fact]
    public void Study_Tell_Complete_State_Throws()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        using var study = Optimizer.CreateStudy("test", space, config: new TpeSamplerConfig { Seed = 1 });

        var trial = study.Ask();

        // Tell(number, TrialState.Complete) should throw — must use Tell(number, value) instead
        Assert.Throws<ArgumentException>(() => study.Tell(trial.Number, TrialState.Complete));
    }

    [Fact]
    public void Study_Tell_InvalidTrialNumber_Fail_Throws()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        using var study = Optimizer.CreateStudy("test", space, config: new TpeSamplerConfig { Seed = 1 });

        Assert.Throws<ArgumentException>(() => study.Tell(999, TrialState.Fail));
    }

    [Fact]
    public void Study_TellBatch_EmptyList_NoOp()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        using var study = Optimizer.CreateStudy("test", space, config: new TpeSamplerConfig { Seed = 1 });

        // Should not throw - explicitly typed to TrialResult
        study.TellBatch(new List<TrialResult>());
        Assert.Empty(study.Trials);
    }

    [Fact]
    public void Study_TellBatch_UnknownTrialNumber_Skipped()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        using var study = Optimizer.CreateStudy("test", space, config: new TpeSamplerConfig { Seed = 1 });

        var trial = study.Ask();

        // Include one valid and one invalid trial number
        var results = new List<TrialResult>
        {
            new(trial.Number, 1.0, TrialState.Complete),
            new(999, 2.0, TrialState.Complete) // Unknown — should be skipped
        };

        study.TellBatch(results);

        Assert.Equal(TrialState.Complete, trial.State);
        Assert.Equal(1.0, trial.Value);
    }

    [Fact]
    public void Study_Dispose_WithNonDisposableSampler_NoError()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        var study = Optimizer.CreateStudy("test", space, config: new TpeSamplerConfig { Seed = 1 });

        // TpeSampler doesn't implement IDisposable — Dispose should still work
        study.Dispose();
    }

    [Fact]
    public void Study_Dispose_WithDisposableSampler_DisposesIt()
    {
        var space = new SearchSpace([
            new FloatRange("x", 0, 10),
            new FloatRange("y", 0, 10)
        ]);
        var sampler = new CmaEsSampler(new CmaEsSamplerConfig { Seed = 1 });
        var study = Optimizer.CreateStudy("test", space, sampler);

        // Initialize the sampler by asking
        var trial = study.Ask();
        study.Tell(trial.Number, 1.0);

        // CmaEsSampler implements IDisposable
        study.Dispose();
    }

    [Fact]
    public void Study_Name_Property()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        using var study = Optimizer.CreateStudy("my_study", space, config: new TpeSamplerConfig { Seed = 1 });

        Assert.Equal("my_study", study.Name);
    }

    [Fact]
    public void Study_BestTrial_SkipsFailedTrials()
    {
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        using var study = Optimizer.CreateStudy("test", space, config: new TpeSamplerConfig { Seed = 1 });

        // All trials failed — BestTrial should be null
        for (var i = 0; i < 3; i++)
        {
            var trial = study.Ask();
            study.Tell(trial.Number, TrialState.Fail);
        }

        Assert.Null(study.BestTrial);
    }

    // ── Record structs and semantics ──────────────────────────────────

    [Fact]
    public void TrialResult_Properties()
    {
        var result = new TrialResult(42, 3.14, TrialState.Complete);

        Assert.Equal(42, result.TrialNumber);
        Assert.Equal(3.14, result.Value);
        Assert.Equal(TrialState.Complete, result.State);
    }

    [Fact]
    public void TrialResult_NullValue_ForFail()
    {
        var result = new TrialResult(1, null, TrialState.Fail);

        Assert.Null(result.Value);
        Assert.Equal(TrialState.Fail, result.State);
    }

    [Fact]
    public void TpeSamplerConfig_DefaultValues()
    {
        var config = new TpeSamplerConfig();

        Assert.Equal(10, config.NStartupTrials);
        Assert.Equal(24, config.NEiCandidates);
        Assert.Equal(1.0, config.PriorWeight);
        Assert.True(config.ConstantLiar);
        Assert.True(config.ConsiderMagicClip);
        Assert.Null(config.Seed);
    }

    [Fact]
    public void TpeSamplerConfig_RecordSemantics()
    {
        var config1 = new TpeSamplerConfig { Seed = 42, NStartupTrials = 20 };
        var config2 = new TpeSamplerConfig { Seed = 42, NStartupTrials = 20 };
        var config3 = config1 with { NEiCandidates = 48 };

        // Equality
        Assert.Equal(config1, config2);
        Assert.NotEqual(config1, config3);

        // ToString (exercises PrintMembers)
        var str = config1.ToString();
        Assert.Contains("Seed", str);

        // GetHashCode
        Assert.Equal(config1.GetHashCode(), config2.GetHashCode());
    }

    [Fact]
    public void TpeSamplerConfig_AllInits()
    {
        var config = new TpeSamplerConfig
        {
            NStartupTrials = 5,
            NEiCandidates = 12,
            PriorWeight = 2.0,
            ConstantLiar = false,
            ConsiderMagicClip = false,
            Seed = 99
        };

        Assert.Equal(5, config.NStartupTrials);
        Assert.Equal(12, config.NEiCandidates);
        Assert.Equal(2.0, config.PriorWeight);
        Assert.False(config.ConstantLiar);
        Assert.False(config.ConsiderMagicClip);
        Assert.Equal(99, config.Seed);
    }

    [Fact]
    public void CmaEsSamplerConfig_DefaultValues()
    {
        var config = new CmaEsSamplerConfig();

        Assert.Null(config.PopulationSize);
        Assert.Equal(0.3, config.InitialSigma);
        Assert.Null(config.Seed);
        Assert.Equal(ComputeBackend.Cpu, config.Backend);
    }

    [Fact]
    public void CmaEsSamplerConfig_RecordSemantics()
    {
        var config1 = new CmaEsSamplerConfig { Seed = 42, InitialSigma = 0.5 };
        var config2 = new CmaEsSamplerConfig { Seed = 42, InitialSigma = 0.5 };
        var config3 = config1 with { PopulationSize = 30 };

        Assert.Equal(config1, config2);
        Assert.NotEqual(config1, config3);

        var str = config1.ToString();
        Assert.Contains("Seed", str);

        Assert.Equal(config1.GetHashCode(), config2.GetHashCode());
    }

    [Fact]
    public void CmaEsSamplerConfig_AllInits()
    {
        var config = new CmaEsSamplerConfig
        {
            PopulationSize = 50,
            InitialSigma = 0.5,
            Seed = 7,
            Backend = ComputeBackend.Auto
        };

        Assert.Equal(50, config.PopulationSize);
        Assert.Equal(0.5, config.InitialSigma);
        Assert.Equal(7, config.Seed);
        Assert.Equal(ComputeBackend.Auto, config.Backend);
    }

    [Fact]
    public void CmaEsMetrics_Properties()
    {
        var metrics = new CmaEsMetrics
        {
            Generation = 5,
            Sigma = 0.3,
            ConditionNumber = 1.5,
            BestValue = 0.001,
            EvaluatedTrials = 50
        };

        Assert.Equal(5, metrics.Generation);
        Assert.Equal(0.3, metrics.Sigma);
        Assert.Equal(1.5, metrics.ConditionNumber);
        Assert.Equal(0.001, metrics.BestValue);
        Assert.Equal(50, metrics.EvaluatedTrials);
    }

    [Fact]
    public void CmaEsMetrics_RecordSemantics()
    {
        var m1 = new CmaEsMetrics { Generation = 1, Sigma = 0.3, ConditionNumber = 1.0, BestValue = 0.5, EvaluatedTrials = 10 };
        var m2 = new CmaEsMetrics { Generation = 1, Sigma = 0.3, ConditionNumber = 1.0, BestValue = 0.5, EvaluatedTrials = 10 };

        Assert.Equal(m1, m2);
        Assert.Equal(m1.GetHashCode(), m2.GetHashCode());
        Assert.Contains("Generation", m1.ToString());
    }
}
