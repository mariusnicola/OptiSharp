using OptiSharp.Models;
using Xunit;

namespace OptiSharp.Tests;

public class StorageTests
{
    private SearchSpace CreateSimpleSpace() =>
        new SearchSpace([new FloatRange("x", 0, 10), new IntRange("y", 1, 5)]);

    [Fact]
    public void Study_Save_SerializesToJson()
    {
        var space = CreateSimpleSpace();
        using var study = Optimizer.CreateStudy("test-study", space);

        var t1 = study.Ask();
        study.Tell(t1.Number, 2.5);

        var tempFile = Path.GetTempFileName();
        try
        {
            study.Save(tempFile);

            var json = File.ReadAllText(tempFile);
            Assert.NotEmpty(json);
            Assert.Contains("test-study", json);
            Assert.Contains("Complete", json);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void Study_Load_ReconstructsStudy()
    {
        var space = CreateSimpleSpace();
        using var study1 = Optimizer.CreateStudy("test-study", space);

        var t1 = study1.Ask();
        var t2 = study1.Ask();
        study1.Tell(t1.Number, 2.5);
        study1.Tell(t2.Number, 3.5);

        var tempFile = Path.GetTempFileName();
        try
        {
            study1.Save(tempFile);

            var sampler = new Samplers.RandomSampler();
            using var study2 = Study.Load(tempFile, space, sampler);

            var trials = study2.Trials;
            Assert.Equal(2, trials.Count);
            Assert.Equal(2.5, trials[0].Value);
            Assert.Equal(3.5, trials[1].Value);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void Storage_RoundTrip_PreservesAllFields()
    {
        var space = CreateSimpleSpace();
        using var study1 = Optimizer.CreateStudy("test", space);

        var t1 = study1.Ask();
        study1.Tell(t1.Number, 1.5);

        var tempFile = Path.GetTempFileName();
        try
        {
            study1.Save(tempFile);

            using var study2 = Study.Load(tempFile, space, new Samplers.RandomSampler());
            var trial = study2.Trials[0];

            Assert.Equal(1.5, trial.Value);
            Assert.Equal(TrialState.Complete, trial.State);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void Storage_PreservesFloatParameter()
    {
        var space = CreateSimpleSpace();
        using var study1 = Optimizer.CreateStudy("test", space);

        var t1 = study1.Ask();
        var xValue = (double)t1.Parameters["x"];
        study1.Tell(t1.Number, 1.0);

        var tempFile = Path.GetTempFileName();
        try
        {
            study1.Save(tempFile);

            using var study2 = Study.Load(tempFile, space, new Samplers.RandomSampler());
            var trial = study2.Trials[0];

            Assert.Equal(xValue, trial.Parameters["x"]);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void Storage_PreservesIntParameter()
    {
        var space = CreateSimpleSpace();
        using var study1 = Optimizer.CreateStudy("test", space);

        var t1 = study1.Ask();
        var yValue = (int)t1.Parameters["y"];
        study1.Tell(t1.Number, 1.0);

        var tempFile = Path.GetTempFileName();
        try
        {
            study1.Save(tempFile);

            using var study2 = Study.Load(tempFile, space, new Samplers.RandomSampler());
            var trial = study2.Trials[0];

            Assert.Equal(yValue, trial.Parameters["y"]);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void Storage_PreservesIntermediateValues()
    {
        var space = CreateSimpleSpace();
        using var study1 = Optimizer.CreateStudy("test", space);

        var t1 = study1.Ask();
        t1.Report(2.0, 1);
        t1.Report(1.5, 2);
        t1.Report(1.0, 3);
        study1.Tell(t1.Number, 1.0);

        var tempFile = Path.GetTempFileName();
        try
        {
            study1.Save(tempFile);

            using var study2 = Study.Load(tempFile, space, new Samplers.RandomSampler());
            var trial = study2.Trials[0];

            Assert.Equal(3, trial.IntermediateValues.Count);
            Assert.Equal(2.0, trial.IntermediateValues[1]);
            Assert.Equal(1.5, trial.IntermediateValues[2]);
            Assert.Equal(1.0, trial.IntermediateValues[3]);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void Storage_PreservesConstraintValues()
    {
        var space = CreateSimpleSpace();
        using var study1 = Optimizer.CreateStudy("test", space);

        study1.SetConstraintFunc(t => new[] { (double)t.Parameters["x"] - 5.0 });

        var t1 = study1.Ask();
        study1.Tell(t1.Number, 1.0);

        var tempFile = Path.GetTempFileName();
        try
        {
            study1.Save(tempFile);

            using var study2 = Study.Load(tempFile, space, new Samplers.RandomSampler());
            var trial = study2.Trials[0];

            Assert.NotNull(trial.ConstraintValues);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void Storage_MultiObjective_PreservesDirections()
    {
        var space = CreateSimpleSpace();
        var directions = new[] { Models.StudyDirection.Minimize, Models.StudyDirection.Maximize };
        using var study1 = Optimizer.CreateStudy("test", space, directions);

        var t1 = study1.Ask();
        study1.Tell(t1.Number, new[] { 1.0, 5.0 });

        var tempFile = Path.GetTempFileName();
        try
        {
            study1.Save(tempFile);

            using var study2 = Study.Load(tempFile, space, new Samplers.RandomSampler(), Models.StudyDirection.Minimize);
            // When loading multi-objective, the directions are restored

            var trial = study2.Trials[0];
            Assert.Equal(new[] { 1.0, 5.0 }, trial.Values);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void Storage_OnlyPersistsCompletedTrials()
    {
        var space = CreateSimpleSpace();
        using var study1 = Optimizer.CreateStudy("test", space);

        var t1 = study1.Ask();
        var t2 = study1.Ask();
        var t3 = study1.Ask();

        study1.Tell(t1.Number, 1.0);
        study1.Tell(t2.Number, TrialState.Fail);
        // t3 is still Running

        var tempFile = Path.GetTempFileName();
        try
        {
            study1.Save(tempFile);

            using var study2 = Study.Load(tempFile, space, new Samplers.RandomSampler());

            // Only t1 should be persisted (Complete state)
            Assert.Single(study2.Trials);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void Storage_ParameterTypeReconstruction_CategoricalRange()
    {
        var space = new SearchSpace([
            new FloatRange("x", 0, 10),
            new CategoricalRange("cat", new object[] { "a", "b", "c" })
        ]);

        using var study1 = Optimizer.CreateStudy("test", space);

        var t1 = study1.Ask();
        study1.Tell(t1.Number, 1.0);

        var tempFile = Path.GetTempFileName();
        try
        {
            study1.Save(tempFile);

            using var study2 = Study.Load(tempFile, space, new Samplers.RandomSampler());
            var trial = study2.Trials[0];

            var catValue = trial.Parameters["cat"];
            Assert.IsType<string>(catValue);
            Assert.Contains((string)catValue, new[] { "a", "b", "c" });
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void Storage_InvalidFile_Throws()
    {
        var space = CreateSimpleSpace();
        var sampler = new Samplers.RandomSampler();

        Assert.Throws<FileNotFoundException>(() => Study.Load("/nonexistent/path.json", space, sampler));
    }

    [Fact]
    public void Storage_MalformedJson_Throws()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tempFile, "{ invalid json }");

            var space = CreateSimpleSpace();
            Assert.Throws<System.Text.Json.JsonException>(() => Study.Load(tempFile, space, new Samplers.RandomSampler()));
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void Storage_OnlyPersistsPrunedAndCompleted()
    {
        var space = CreateSimpleSpace();
        using var study1 = Optimizer.CreateStudy("test", space);

        var t1 = study1.Ask();
        var t2 = study1.Ask();
        var t3 = study1.Ask();

        study1.Tell(t1.Number, 1.0);
        study1.Tell(t2.Number, TrialState.Fail);  // Should not be persisted
        // t3 is still Running - should not be persisted

        var tempFile = Path.GetTempFileName();
        try
        {
            study1.Save(tempFile);

            using var study2 = Study.Load(tempFile, space, new Samplers.RandomSampler());

            // Only t1 (Complete) should be persisted
            Assert.Single(study2.Trials);
            Assert.Equal(1.0, study2.Trials[0].Value);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }
}
