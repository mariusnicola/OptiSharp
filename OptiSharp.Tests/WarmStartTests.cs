using OptiSharp.Models;
using Xunit;

namespace OptiSharp.Tests;

public class WarmStartTests
{
    private SearchSpace CreateSimpleSpace() =>
        new SearchSpace([new FloatRange("x", 0, 10), new IntRange("y", 0, 5)]);

    [Fact]
    public void Optimizer_CreateStudy_WithWarmStartTrials()
    {
        var space = CreateSimpleSpace();

        // Create warm trials
        var warmTrials = new List<Trial>
        {
            new Trial(0, new Dictionary<string, object> { ["x"] = 1.0, ["y"] = 2 })
            {
                State = TrialState.Complete,
                Value = 1.5
            },
            new Trial(1, new Dictionary<string, object> { ["x"] = 3.0, ["y"] = 1 })
            {
                State = TrialState.Complete,
                Value = 3.5
            }
        };

        using var study = Optimizer.CreateStudy("test", space, warmStartTrials: warmTrials);

        var trials = study.Trials;
        Assert.Equal(2, trials.Count);
        Assert.Equal(1.5, trials[0].Value);
        Assert.Equal(3.5, trials[1].Value);
    }

    [Fact]
    public void Optimizer_CreateStudy_WithFromStudy()
    {
        var space = CreateSimpleSpace();

        // Create initial study
        using var study1 = Optimizer.CreateStudy("study1", space);
        var t1 = study1.Ask();
        var t2 = study1.Ask();
        study1.Tell(t1.Number, 2.0);
        study1.Tell(t2.Number, 4.0);

        // Create new study with warm start from study1
        using var study2 = Optimizer.CreateStudy("study2", space, fromStudy: study1);

        var trials = study2.Trials;
        Assert.Equal(2, trials.Count);
        Assert.Equal(TrialState.Complete, trials[0].State);
        Assert.Equal(TrialState.Complete, trials[1].State);
    }

    [Fact]
    public void WarmStart_OnlyImportsCompletedTrials()
    {
        var space = CreateSimpleSpace();

        var warmTrials = new List<Trial>
        {
            new Trial(0, new Dictionary<string, object> { ["x"] = 1.0, ["y"] = 2 })
            {
                State = TrialState.Complete,
                Value = 1.5
            },
            new Trial(1, new Dictionary<string, object> { ["x"] = 3.0, ["y"] = 1 })
            {
                State = TrialState.Running // Should be skipped
            }
        };

        using var study = Optimizer.CreateStudy("test", space, warmStartTrials: warmTrials);

        // Only the complete trial should be imported
        Assert.Single(study.Trials);
    }

    [Fact]
    public void WarmStart_AssignsNewTrialNumbers()
    {
        var space = CreateSimpleSpace();

        var warmTrials = new List<Trial>
        {
            new Trial(100, new Dictionary<string, object> { ["x"] = 1.0, ["y"] = 2 })
            {
                State = TrialState.Complete,
                Value = 1.5
            },
            new Trial(101, new Dictionary<string, object> { ["x"] = 3.0, ["y"] = 1 })
            {
                State = TrialState.Complete,
                Value = 3.5
            }
        };

        using var study = Optimizer.CreateStudy("test", space, warmStartTrials: warmTrials);

        var trials = study.Trials;
        Assert.Equal(0, trials[0].Number);
        Assert.Equal(1, trials[1].Number);
    }

    [Fact]
    public void WarmStart_PreservesParameterValues()
    {
        var space = CreateSimpleSpace();

        var warmTrials = new List<Trial>
        {
            new Trial(0, new Dictionary<string, object> { ["x"] = 5.5, ["y"] = 3 })
            {
                State = TrialState.Complete,
                Value = 1.5
            }
        };

        using var study = Optimizer.CreateStudy("test", space, warmStartTrials: warmTrials);

        var trial = study.Trials[0];
        Assert.Equal(5.5, trial.Parameters["x"]);
        Assert.Equal(3, trial.Parameters["y"]);
    }

    [Fact]
    public void WarmStart_CopiesIntermediateValues()
    {
        var space = CreateSimpleSpace();

        var warmTrial = new Trial(0, new Dictionary<string, object> { ["x"] = 1.0, ["y"] = 2 })
        {
            State = TrialState.Complete,
            Value = 1.5
        };
        warmTrial.Report(2.0, 1);
        warmTrial.Report(1.8, 2);
        warmTrial.Report(1.5, 3);

        using var study = Optimizer.CreateStudy("test", space, warmStartTrials: [warmTrial]);

        var trial = study.Trials[0];
        Assert.Equal(3, trial.IntermediateValues.Count);
        Assert.Equal(2.0, trial.IntermediateValues[1]);
        Assert.Equal(1.8, trial.IntermediateValues[2]);
        Assert.Equal(1.5, trial.IntermediateValues[3]);
    }

    [Fact]
    public void WarmStart_CopiesConstraintValues()
    {
        var space = CreateSimpleSpace();

        var warmTrial = new Trial(0, new Dictionary<string, object> { ["x"] = 1.0, ["y"] = 2 })
        {
            State = TrialState.Complete,
            Value = 1.5,
            ConstraintValues = new[] { -1.0, 0.5 }
        };

        using var study = Optimizer.CreateStudy("test", space, warmStartTrials: [warmTrial]);

        var trial = study.Trials[0];
        Assert.Equal(new[] { -1.0, 0.5 }, trial.ConstraintValues);
    }

    [Fact]
    public void WarmStart_CopiesMultiObjectiveValues()
    {
        var space = CreateSimpleSpace();
        var directions = new[] { Models.StudyDirection.Minimize, Models.StudyDirection.Maximize };

        var warmTrial = new Trial(0, new Dictionary<string, object> { ["x"] = 1.0, ["y"] = 2 })
        {
            State = TrialState.Complete,
            Values = new[] { 1.0, 5.0 }
        };

        using var study = Optimizer.CreateStudy("test", space, directions, warmStartTrials: [warmTrial]);

        var trial = study.Trials[0];
        Assert.Equal(new[] { 1.0, 5.0 }, trial.Values);
    }

    [Fact]
    public void WarmStart_TrialsCountTowardStartupPhase()
    {
        var space = CreateSimpleSpace();
        var config = new Samplers.Tpe.TpeSamplerConfig { NStartupTrials = 5, Seed = 42 };

        var warmTrials = Enumerable.Range(0, 3)
            .Select(i => new Trial(i, new Dictionary<string, object> { ["x"] = i, ["y"] = i })
            {
                State = TrialState.Complete,
                Value = i * 1.0
            })
            .ToList();

        using var study = Optimizer.CreateStudy("test", space, config: config, warmStartTrials: warmTrials);

        // Warm trials should be included in sampler's history
        Assert.Equal(3, study.Trials.Count);

        // Add 2 more trials to reach startup threshold
        var t1 = study.Ask();
        study.Tell(t1.Number, 3.0);
        var t2 = study.Ask();
        study.Tell(t2.Number, 4.0);

        // Now we should have enough for TPE to use (5 >= NStartupTrials)
        Assert.True(study.Trials.Count >= config.NStartupTrials);
    }

    [Fact]
    public void WarmStart_WithRandomSampler()
    {
        var space = CreateSimpleSpace();

        var warmTrial = new Trial(0, new Dictionary<string, object> { ["x"] = 5.0, ["y"] = 2 })
        {
            State = TrialState.Complete,
            Value = 1.5
        };

        using var study = Optimizer.CreateStudyWithRandomSampler("test", space, warmStartTrials: [warmTrial]);

        Assert.Single(study.Trials);
        Assert.Equal(5.0, study.Trials[0].Parameters["x"]);
    }

    [Fact]
    public void WarmStart_WithCmaEsSampler()
    {
        var space = CreateSimpleSpace();

        var warmTrials = Enumerable.Range(0, 3)
            .Select(i => new Trial(i, new Dictionary<string, object> { ["x"] = i * 2.0, ["y"] = i })
            {
                State = TrialState.Complete,
                Value = i * 1.0
            })
            .ToList();

        using var study = Optimizer.CreateStudyWithCmaEs("test", space, warmStartTrials: warmTrials);

        Assert.Equal(3, study.Trials.Count);
        var best = study.BestTrial;
        Assert.NotNull(best);
    }

    [Fact]
    public void WarmStart_MultipleWarmTrials_InOrder()
    {
        var space = CreateSimpleSpace();

        var warmTrials = Enumerable.Range(0, 5)
            .Select(i => new Trial(i, new Dictionary<string, object> { ["x"] = i, ["y"] = i })
            {
                State = TrialState.Complete,
                Value = i * 1.0
            })
            .ToList();

        using var study = Optimizer.CreateStudy("test", space, warmStartTrials: warmTrials);

        var trials = study.Trials;
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(i, trials[i].Number);
            Assert.Equal(i * 1.0, trials[i].Value);
        }
    }
}
