using OptiSharp.Models;
using OptiSharp.MultiObjective;
using Xunit;

namespace OptiSharp.Tests;

public class MultiObjectiveTests
{
    private SearchSpace CreateSimpleSpace() =>
        new SearchSpace([new FloatRange("x", 0, 10)]);

    [Fact]
    public void ParetoUtils_Dominates_WithMinimize()
    {
        var directions = new[] { StudyDirection.Minimize, StudyDirection.Minimize };

        var a = new[] { 1.0, 2.0 };
        var b = new[] { 2.0, 3.0 };

        Assert.True(ParetoUtils.Dominates(a, b, directions));
        Assert.False(ParetoUtils.Dominates(b, a, directions));
    }

    [Fact]
    public void ParetoUtils_Dominates_WithMaximize()
    {
        var directions = new[] { StudyDirection.Maximize, StudyDirection.Maximize };

        var a = new[] { 2.0, 3.0 };
        var b = new[] { 1.0, 2.0 };

        Assert.True(ParetoUtils.Dominates(a, b, directions));
        Assert.False(ParetoUtils.Dominates(b, a, directions));
    }

    [Fact]
    public void ParetoUtils_Dominates_MixedDirections()
    {
        var directions = new[] { StudyDirection.Minimize, StudyDirection.Maximize };

        var a = new[] { 1.0, 3.0 };
        var b = new[] { 2.0, 2.0 };

        // a dominates b: better in first (1 < 2), at least as good in second (3 > 2)
        Assert.True(ParetoUtils.Dominates(a, b, directions));
    }

    [Fact]
    public void ParetoUtils_Dominates_WithEqualValues()
    {
        var directions = new[] { StudyDirection.Minimize, StudyDirection.Minimize };

        var a = new[] { 1.0, 2.0 };
        var b = new[] { 1.0, 2.0 };

        // Neither dominates when equal
        Assert.False(ParetoUtils.Dominates(a, b, directions));
        Assert.False(ParetoUtils.Dominates(b, a, directions));
    }

    [Fact]
    public void ParetoUtils_ComputeParetoFront_SingleObjective()
    {
        var directions = new[] { StudyDirection.Minimize };

        var trials = new List<Trial>();
        for (int i = 0; i < 5; i++)
        {
            var t = new Trial(i, new Dictionary<string, object> { ["x"] = i })
            {
                State = TrialState.Complete,
                Values = new[] { 5.0 - i } // 5, 4, 3, 2, 1
            };
            trials.Add(t);
        }

        var front = ParetoUtils.ComputeParetoFront(trials, directions);

        // Only the best trial (value=1) should be on the front
        Assert.Single(front);
        Assert.Equal(4, front[0].Number);
    }

    [Fact]
    public void ParetoUtils_ComputeParetoFront_MultiObjective()
    {
        var directions = new[] { StudyDirection.Minimize, StudyDirection.Minimize };

        var trials = new List<Trial>();

        // Create a simple Pareto front: (1,5), (2,3), (4,1)
        var values = new[] { new[] { 1.0, 5.0 }, new[] { 2.0, 3.0 }, new[] { 4.0, 1.0 }, new[] { 3.0, 4.0 } };

        for (int i = 0; i < values.Length; i++)
        {
            var t = new Trial(i, new Dictionary<string, object> { ["x"] = i })
            {
                State = TrialState.Complete,
                Values = values[i]
            };
            trials.Add(t);
        }

        var front = ParetoUtils.ComputeParetoFront(trials, directions);

        // Should have 3 non-dominated solutions
        Assert.Equal(3, front.Count);
    }

    [Fact]
    public void ParetoUtils_ComputeParetoFront_EmptyList()
    {
        var directions = new[] { StudyDirection.Minimize };
        var front = ParetoUtils.ComputeParetoFront([], directions);
        Assert.Empty(front);
    }

    [Fact]
    public void ParetoUtils_ComputeParetoFront_IgnoresRunningTrials()
    {
        var directions = new[] { StudyDirection.Minimize };

        var trials = new List<Trial>
        {
            new Trial(0, new Dictionary<string, object> { ["x"] = 0 })
            {
                State = TrialState.Complete,
                Values = new[] { 1.0 }
            },
            new Trial(1, new Dictionary<string, object> { ["x"] = 1 })
            {
                State = TrialState.Running
            }
        };

        var front = ParetoUtils.ComputeParetoFront(trials, directions);
        Assert.Single(front);
    }

    [Fact]
    public void ParetoUtils_CrowdingDistances_CalculatesDistances()
    {
        var directions = new[] { StudyDirection.Minimize };

        var trials = new List<Trial>();
        for (int i = 0; i < 3; i++)
        {
            var t = new Trial(i, new Dictionary<string, object> { ["x"] = i })
            {
                State = TrialState.Complete,
                Values = new[] { (double)i }
            };
            trials.Add(t);
        }

        var distances = ParetoUtils.CrowdingDistances(trials, directions);

        // Boundary points should have infinite distance
        Assert.Equal(double.PositiveInfinity, distances[0]);
        Assert.Equal(double.PositiveInfinity, distances[2]);

        // Middle point should have finite distance
        Assert.IsType<double>(distances[1]);
        Assert.NotEqual(double.PositiveInfinity, distances[1]);
    }

    [Fact]
    public void Study_IsMultiObjective_ReturnsTrueForMultiObjectiveStudy()
    {
        var space = CreateSimpleSpace();
        var directions = new[] { StudyDirection.Minimize, StudyDirection.Maximize };
        using var study = Optimizer.CreateStudy("test", space, directions);

        Assert.True(study.IsMultiObjective);
    }

    [Fact]
    public void Study_Tell_WithDoubleArray_StoresValues()
    {
        var space = CreateSimpleSpace();
        var directions = new[] { StudyDirection.Minimize, StudyDirection.Maximize };
        using var study = Optimizer.CreateStudy("test", space, directions);

        var trial = study.Ask();
        var values = new[] { 1.0, 5.0 };
        study.Tell(trial.Number, values);

        Assert.Equal(values, trial.Values);
        Assert.Equal(values[0], trial.Value); // Backward compat
        Assert.Equal(TrialState.Complete, trial.State);
    }

    [Fact]
    public void Study_TellBatch_WithMoTrialResult()
    {
        var space = CreateSimpleSpace();
        var directions = new[] { StudyDirection.Minimize, StudyDirection.Maximize };
        using var study = Optimizer.CreateStudy("test", space, directions);

        var t1 = study.Ask();
        var t2 = study.Ask();

        var results = new[]
        {
            new MoTrialResult(t1.Number, new[] { 1.0, 5.0 }, TrialState.Complete),
            new MoTrialResult(t2.Number, new[] { 2.0, 4.0 }, TrialState.Complete)
        };

        study.TellBatch(results);

        Assert.Equal(new[] { 1.0, 5.0 }, t1.Values);
        Assert.Equal(new[] { 2.0, 4.0 }, t2.Values);
    }

    [Fact]
    public void Study_ParetoFront_ReturnsNonDominatedTrials()
    {
        var space = CreateSimpleSpace();
        var directions = new[] { StudyDirection.Minimize, StudyDirection.Minimize };
        using var study = Optimizer.CreateStudy("test", space, directions);

        var t1 = study.Ask();
        var t2 = study.Ask();
        var t3 = study.Ask();

        study.Tell(t1.Number, new[] { 1.0, 5.0 });
        study.Tell(t2.Number, new[] { 2.0, 3.0 });
        study.Tell(t3.Number, new[] { 4.0, 1.0 });

        var front = study.ParetoFront;

        // All three are non-dominated
        Assert.Equal(3, front.Count);
    }

    [Fact]
    public void Study_ParetoFront_SingleObjectiveReturnsOnlyBest()
    {
        var space = CreateSimpleSpace();
        using var study = Optimizer.CreateStudy("test", space, StudyDirection.Minimize);

        var t1 = study.Ask();
        var t2 = study.Ask();

        study.Tell(t1.Number, 10.0);
        study.Tell(t2.Number, 5.0);

        var front = study.ParetoFront;

        Assert.Single(front);
        Assert.Equal(t2.Number, front[0].Number);
    }

    [Fact]
    public void ISampler_SampleMultiObjective_DefaultImplementation()
    {
        ISampler sampler = new Samplers.RandomSampler();
        var space = CreateSimpleSpace();
        var directions = new[] { StudyDirection.Minimize, StudyDirection.Maximize };

        var parameters = sampler.SampleMultiObjective([], directions, space);

        Assert.NotNull(parameters);
        Assert.Single(parameters);
    }

    [Fact]
    public void Optimizer_CreateStudy_WithDirections_CreatesMultiObjectiveStudy()
    {
        var space = CreateSimpleSpace();
        var directions = new[] { StudyDirection.Minimize, StudyDirection.Maximize };

        using var study = Optimizer.CreateStudy("test", space, directions);

        Assert.True(study.IsMultiObjective);
    }

    [Fact]
    public void Study_BestTrial_InMultiObjectiveUsesFirstObjective()
    {
        var space = CreateSimpleSpace();
        var directions = new[] { StudyDirection.Minimize, StudyDirection.Maximize };
        using var study = Optimizer.CreateStudy("test", space, directions);

        var t1 = study.Ask();
        var t2 = study.Ask();

        study.Tell(t1.Number, new[] { 10.0, 5.0 });
        study.Tell(t2.Number, new[] { 5.0, 10.0 }); // Better in first objective

        var best = study.BestTrial;

        Assert.NotNull(best);
        Assert.Equal(t2.Number, best.Number); // Selected by first objective
    }

    [Fact]
    public void MoTrialResult_CanBeCreatedAndDestructured()
    {
        var result = new MoTrialResult(42, new[] { 1.0, 2.0 }, TrialState.Complete);

        Assert.Equal(42, result.TrialNumber);
        Assert.Equal(new[] { 1.0, 2.0 }, result.Values);
        Assert.Equal(TrialState.Complete, result.State);
    }

    [Fact]
    public void Study_Direction_SetsToFirstDirectionInMultiObjective()
    {
        var space = CreateSimpleSpace();
        var directions = new[] { StudyDirection.Minimize, StudyDirection.Maximize };
        using var study = Optimizer.CreateStudy("test", space, directions);

        // Direction should be set to the first objective for backward compatibility
        Assert.Equal(StudyDirection.Minimize, study.Direction);
    }
}
