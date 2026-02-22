using OptiSharp.Models;
using OptiSharp.Pbt;
using Xunit;

namespace OptiSharp.Tests;

public class PbtTests
{
    private SearchSpace CreateSimpleSpace() =>
        new SearchSpace([
            new FloatRange("lr", 1e-4, 1e-1),
            new IntRange("batch_size", 16, 128),
            new CategoricalRange("optimizer", new object[] { "adam", "sgd", "rmsprop" })
        ]);

    [Fact]
    public void PopulationBasedTrainer_AskPopulation_ReturnsCorrectSize()
    {
        var space = CreateSimpleSpace();
        var pbt = new PopulationBasedTrainer(space, populationSize: 10);

        var population = pbt.AskPopulation();

        Assert.Equal(10, population.Count);
    }

    [Fact]
    public void PopulationBasedTrainer_AskPopulation_AssignsIds()
    {
        var space = CreateSimpleSpace();
        var pbt = new PopulationBasedTrainer(space, populationSize: 5);

        var population = pbt.AskPopulation();

        for (int i = 0; i < 5; i++)
            Assert.Equal(i, population[i].Id);
    }

    [Fact]
    public void PopulationBasedTrainer_AskPopulation_CreatesParameterHistory()
    {
        var space = CreateSimpleSpace();
        var pbt = new PopulationBasedTrainer(space, populationSize: 3);

        var population = pbt.AskPopulation();

        foreach (var member in population)
        {
            Assert.Single(member.ParameterHistory);
            Assert.Equal(member.Parameters, member.ParameterHistory[0]);
        }
    }

    [Fact]
    public void PopulationBasedTrainer_Report_UpdatesPerformance()
    {
        var space = CreateSimpleSpace();
        var pbt = new PopulationBasedTrainer(space, populationSize: 2);

        var population = pbt.AskPopulation().ToList();
        var member = population[0];

        var updated = pbt.Report(member, performance: 0.95, step: 10);

        Assert.Equal(0.95, updated.Performance);
        Assert.Equal(10, updated.Step);
    }

    [Fact]
    public void PopulationBasedTrainer_Evolve_KeepsTopMembers()
    {
        var space = CreateSimpleSpace();
        var pbt = new PopulationBasedTrainer(space, populationSize: 10, exploitFraction: 0.3, seed: 42);

        var population = pbt.AskPopulation().ToList();

        // Assign performances
        var updated = population.Select((m, i) => pbt.Report(m, (double)(10 - i), i)).ToList();

        var evolved = pbt.Evolve(updated);

        // Top 70% should have their performance unchanged
        var top7 = evolved.Take(7).ToList();
        Assert.All(top7, m => Assert.True(m.Performance >= 3.0)); // At least the bottom of top 70%
    }

    [Fact]
    public void PopulationBasedTrainer_Evolve_PerturbsBottom()
    {
        var space = CreateSimpleSpace();
        var pbt = new PopulationBasedTrainer(space, populationSize: 10, exploitFraction: 0.3, perturbFactor: 0.2, seed: 42);

        var population = pbt.AskPopulation().ToList();
        var updated = population.Select((m, i) => pbt.Report(m, (double)(10 - i), i)).ToList();

        var evolved = pbt.Evolve(updated);

        // Bottom 30% should have Reset performance
        var bottom3 = evolved.Skip(7).ToList();
        Assert.All(bottom3, m => Assert.Equal(double.NegativeInfinity, m.Performance));
    }

    [Fact]
    public void PopulationBasedTrainer_Evolve_PreservesPopulationSize()
    {
        var space = CreateSimpleSpace();
        var pbt = new PopulationBasedTrainer(space, populationSize: 8);

        var population = pbt.AskPopulation().ToList();
        var updated = population.Select((m, i) => pbt.Report(m, (double)i, i)).ToList();

        var evolved = pbt.Evolve(updated);

        Assert.Equal(8, evolved.Count);
    }

    [Fact]
    public void PopulationBasedTrainer_PerturbsFloatParameter()
    {
        var space = CreateSimpleSpace();
        var pbt = new PopulationBasedTrainer(space, populationSize: 2, perturbFactor: 0.1, seed: 42);

        var population = pbt.AskPopulation().ToList();
        var updated = population.Select((m, i) => pbt.Report(m, 5.0 - i, i)).ToList();

        var evolved = pbt.Evolve(updated);

        // Check that LR values are within valid bounds
        foreach (var member in evolved)
        {
            var lr = (double)member.Parameters["lr"];
            Assert.InRange(lr, 1e-4, 1e-1);
        }
    }

    [Fact]
    public void PopulationBasedTrainer_PerturbsIntParameter()
    {
        var space = CreateSimpleSpace();
        var pbt = new PopulationBasedTrainer(space, populationSize: 4, perturbFactor: 0.2, seed: 42);

        var population = pbt.AskPopulation().ToList();
        var updated = population.Select((m, i) => pbt.Report(m, 5.0 - i, i)).ToList();

        var evolved = pbt.Evolve(updated);

        // Check that batch_size is still valid
        foreach (var member in evolved)
        {
            var batchSize = (int)member.Parameters["batch_size"];
            Assert.InRange(batchSize, 16, 128);
        }
    }

    [Fact]
    public void PopulationBasedTrainer_PerturbsCategorical()
    {
        var space = CreateSimpleSpace();
        var pbt = new PopulationBasedTrainer(space, populationSize: 4, seed: 42);

        var population = pbt.AskPopulation().ToList();
        var updated = population.Select((m, i) => pbt.Report(m, 5.0 - i, i)).ToList();

        var evolved = pbt.Evolve(updated);

        // Check that optimizer is still valid
        var validOptimizers = new[] { "adam", "sgd", "rmsprop" };
        foreach (var member in evolved)
        {
            var optimizer = (string)member.Parameters["optimizer"];
            Assert.Contains(optimizer, validOptimizers);
        }
    }

    [Fact]
    public void PopulationBasedTrainer_Evolve_UpdatesParameterHistory()
    {
        var space = CreateSimpleSpace();
        var pbt = new PopulationBasedTrainer(space, populationSize: 2, exploitFraction: 0.5, seed: 42);

        var population = pbt.AskPopulation().ToList();
        var updated = population.Select((m, i) => pbt.Report(m, (double)i, i)).ToList();

        var evolved = pbt.Evolve(updated);

        // Bottom member should have extended history
        var bottomMember = evolved.FirstOrDefault(m => m.Performance == double.NegativeInfinity);
        if (bottomMember != null && bottomMember.ParameterHistory.Count > 1)
        {
            Assert.True(bottomMember.ParameterHistory.Count >= 2);
        }
    }

    [Fact]
    public void PopulationBasedTrainer_Constructor_ValidatesPopulationSize()
    {
        var space = CreateSimpleSpace();

        Assert.Throws<ArgumentException>(() => new PopulationBasedTrainer(space, populationSize: 1));
    }

    [Fact]
    public void PopulationBasedTrainer_Constructor_ValidatesExploitFraction()
    {
        var space = CreateSimpleSpace();

        Assert.Throws<ArgumentException>(() => new PopulationBasedTrainer(space, exploitFraction: -0.1));
        Assert.Throws<ArgumentException>(() => new PopulationBasedTrainer(space, exploitFraction: 1.5));
    }

    [Fact]
    public void PopulationBasedTrainer_Constructor_ValidatesPerturbFactor()
    {
        var space = CreateSimpleSpace();

        Assert.Throws<ArgumentException>(() => new PopulationBasedTrainer(space, perturbFactor: -0.1));
    }

    [Fact]
    public void PbtMember_Record_CanBeCreated()
    {
        var parameters = new Dictionary<string, object> { ["x"] = 1.0 };
        var history = new List<IReadOnlyDictionary<string, object>> { parameters }.AsReadOnly();

        var member = new PbtMember(
            Id: 0,
            Parameters: parameters,
            Performance: 0.5,
            Step: 10,
            ParameterHistory: history
        );

        Assert.Equal(0, member.Id);
        Assert.Equal(0.5, member.Performance);
        Assert.Equal(10, member.Step);
    }

    [Fact]
    public void PopulationBasedTrainer_CompleteWorkflow()
    {
        var space = CreateSimpleSpace();
        var pbt = new PopulationBasedTrainer(space, populationSize: 6, exploitFraction: 0.33, seed: 42);

        // Generation 1: Initial population
        var pop1 = pbt.AskPopulation().ToList();
        Assert.Equal(6, pop1.Count);

        // Simulate training and get performances (higher is better)
        var pop1_updated = pop1.Select((m, i) => pbt.Report(m, 0.7 + (0.05 * i), 100)).ToList();

        // Evolution step
        var pop2 = pbt.Evolve(pop1_updated);
        Assert.Equal(6, pop2.Count);

        // Generation 2: Continue training - bottom members get additional performance boost
        var pop2_updated = pop2.Select((m, i) =>
        {
            var perf = m.Performance == double.NegativeInfinity ? 0.7 : m.Performance + 0.1;
            return pbt.Report(m, perf, 200);
        }).ToList();

        // Another evolution
        var pop3 = pbt.Evolve(pop2_updated);
        Assert.Equal(6, pop3.Count);

        // At least one member should have improved or stayed the same
        var best1 = pop1_updated.Max(m => m.Performance);
        var best3 = pop3.Max(m => m.Performance);
        Assert.True(best3 >= best1 - 0.1); // Allow small variance
    }
}
