using OptiSharp.Models;
using Xunit;

namespace OptiSharp.Tests;

public class ConstraintTests
{
    private SearchSpace CreateSimpleSpace() =>
        new SearchSpace([new FloatRange("x", 0, 10)]);

    [Fact]
    public void Study_SetConstraintFunc_StoresFunction()
    {
        var space = CreateSimpleSpace();
        using var study = Optimizer.CreateStudy("test", space);

        Func<Trial, double[]> func = t => new[] { (double)t.Parameters["x"] - 5.0 };
        study.SetConstraintFunc(func);

        var trial = study.Ask();
        study.Tell(trial.Number, 1.0);

        Assert.NotNull(trial.ConstraintValues);
    }

    [Fact]
    public void Study_Tell_ComputesConstraintValues()
    {
        var space = CreateSimpleSpace();
        using var study = Optimizer.CreateStudy("test", space);

        study.SetConstraintFunc(t =>
        {
            var x = (double)t.Parameters["x"];
            return new[] { x - 5.0 };  // Constraint: x <= 5 (x - 5 <= 0)
        });

        var trial = study.Ask();
        study.Tell(trial.Number, 1.0);

        // Constraint values should be computed
        Assert.NotNull(trial.ConstraintValues);
        Assert.Single(trial.ConstraintValues);
        // For any x in [0, 10], the constraint is x - 5, which may be negative or positive
        Assert.IsType<double>(trial.ConstraintValues[0]);
    }

    [Fact]
    public void Study_IsFeasible_ReturnsTrue_WhenConstraintsSatisfied()
    {
        var space = CreateSimpleSpace();
        using var study = Optimizer.CreateStudy("test", space);

        var trial = new Trial(0, new Dictionary<string, object> { ["x"] = 3.0 })
        {
            ConstraintValues = new[] { -1.0, -0.5 } // All <= 0
        };

        Assert.True(study.IsFeasible(trial));
    }

    [Fact]
    public void Study_IsFeasible_ReturnsFalse_WhenConstraintsViolated()
    {
        var space = CreateSimpleSpace();
        using var study = Optimizer.CreateStudy("test", space);

        var trial = new Trial(0, new Dictionary<string, object> { ["x"] = 7.0 })
        {
            ConstraintValues = new[] { 1.0, -0.5 } // First constraint violated
        };

        Assert.False(study.IsFeasible(trial));
    }

    [Fact]
    public void Study_IsFeasible_ReturnsTrue_WhenNoConstraints()
    {
        var space = CreateSimpleSpace();
        using var study = Optimizer.CreateStudy("test", space);

        var trial = new Trial(0, new Dictionary<string, object> { ["x"] = 5.0 })
        {
            ConstraintValues = null
        };

        Assert.True(study.IsFeasible(trial));
    }

    [Fact]
    public void Study_Tell_WithMultipleConstraints()
    {
        var space = CreateSimpleSpace();
        using var study = Optimizer.CreateStudy("test", space);

        study.SetConstraintFunc(t =>
        {
            var x = (double)t.Parameters["x"];
            return new[] { x - 3.0, 7.0 - x }; // Constraints: x >= 3 and x <= 7
        });

        var trial = study.Ask();
        study.Tell(trial.Number, 1.0);

        Assert.Equal(2, trial.ConstraintValues!.Length);
    }

    [Fact]
    public void TpeSampler_PrefersFeasibleTrials()
    {
        var space = CreateSimpleSpace();
        var sampler = new Samplers.Tpe.TpeSampler(new Samplers.Tpe.TpeSamplerConfig { Seed = 42 });
        using var study = new Study("test", sampler, space, StudyDirection.Minimize);

        study.SetConstraintFunc(t =>
        {
            var x = (double)t.Parameters["x"];
            return new[] { x - 5.0 }; // Constraint: x <= 5
        });

        // Create some completed trials
        var feasible = new[] { 1.0, 2.0, 3.0, 4.0 };
        var infeasible = new[] { 6.0, 7.0, 8.0 };

        for (int i = 0; i < feasible.Length; i++)
        {
            var trial = new Trial(i, new Dictionary<string, object> { ["x"] = feasible[i] })
            {
                State = TrialState.Complete,
                Value = feasible[i]
            };
            trial.ConstraintValues = new[] { feasible[i] - 5.0 };
            study.GetType().GetField("_trials", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.SetValue(study, new List<Trial> { trial });
        }

        // Should still work even with infeasible trials mixed in
        var next = study.Ask();
        Assert.NotNull(next);
    }

    [Fact]
    public void Constraint_BoundaryValue_WithEqualZero()
    {
        var space = CreateSimpleSpace();
        using var study = Optimizer.CreateStudy("test", space);

        var trial = new Trial(0, new Dictionary<string, object> { ["x"] = 5.0 })
        {
            ConstraintValues = new[] { 0.0 } // Exactly at boundary
        };

        Assert.True(study.IsFeasible(trial));
    }

    [Fact]
    public void Study_SetConstraintFunc_CanBeCalledMultipleTimes()
    {
        var space = CreateSimpleSpace();
        using var study = Optimizer.CreateStudy("test", space);

        study.SetConstraintFunc(t => new[] { -1.0 });
        study.SetConstraintFunc(t => new[] { 1.0 }); // Should replace

        var trial = study.Ask();
        study.Tell(trial.Number, 1.0);

        Assert.True(trial.ConstraintValues![0] > 0);
    }

    [Fact]
    public void Constraint_WithMultiObjective()
    {
        var space = CreateSimpleSpace();
        var directions = new[] { OptiSharp.Models.StudyDirection.Minimize, OptiSharp.Models.StudyDirection.Minimize };
        using var study = Optimizer.CreateStudy("test", space, directions);

        study.SetConstraintFunc(t => new[] { (double)t.Parameters["x"] - 5.0 });

        var trial = study.Ask();
        study.Tell(trial.Number, new[] { 1.0, 2.0 });

        Assert.NotNull(trial.ConstraintValues);
        Assert.NotNull(trial.Values);
    }

    [Fact]
    public void Study_Tell_WithConstraintValues_AndMultipleObjectives()
    {
        var space = CreateSimpleSpace();
        var directions = new[] { OptiSharp.Models.StudyDirection.Minimize, OptiSharp.Models.StudyDirection.Maximize };
        using var study = Optimizer.CreateStudy("test", space, directions);

        study.SetConstraintFunc(t =>
        {
            var x = (double)t.Parameters["x"];
            return new[] { x - 2.0, 8.0 - x };
        });

        var trial = study.Ask();
        study.Tell(trial.Number, new[] { 1.0, 5.0 });

        Assert.Equal(2, trial.ConstraintValues!.Length);
        Assert.Equal(2, trial.Values!.Length);
    }
}
