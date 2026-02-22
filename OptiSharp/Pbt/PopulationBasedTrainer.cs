using OptiSharp.Models;
using OptiSharp.Samplers;

namespace OptiSharp.Pbt;

/// <summary>
/// Population-Based Training (PBT) coordinator.
/// Manages a population of trials that evolve based on performance and periodic checkpoints.
/// </summary>
public sealed class PopulationBasedTrainer
{
    private readonly SearchSpace _searchSpace;
    private readonly int _populationSize;
    private readonly double _exploitFraction;
    private readonly double _perturbFactor;
    private readonly Random _rng;

    public PopulationBasedTrainer(
        SearchSpace searchSpace,
        int populationSize = 10,
        double exploitFraction = 0.2,
        double perturbFactor = 0.2,
        int? seed = null)
    {
        if (populationSize < 2)
            throw new ArgumentException("populationSize must be >= 2", nameof(populationSize));

        if (exploitFraction < 0 || exploitFraction > 1)
            throw new ArgumentException("exploitFraction must be in [0, 1]", nameof(exploitFraction));

        if (perturbFactor < 0)
            throw new ArgumentException("perturbFactor must be >= 0", nameof(perturbFactor));

        _searchSpace = searchSpace;
        _populationSize = populationSize;
        _exploitFraction = exploitFraction;
        _perturbFactor = perturbFactor;
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Generate initial population with random hyperparameters.
    /// </summary>
    public IReadOnlyList<PbtMember> AskPopulation()
    {
        var sampler = new Samplers.RandomSampler(_rng);
        var population = new List<PbtMember>();

        for (int i = 0; i < _populationSize; i++)
        {
            var parameters = sampler.Sample([], StudyDirection.Minimize, _searchSpace);
            var member = new PbtMember(
                Id: i,
                Parameters: parameters,
                Performance: double.NegativeInfinity,
                Step: 0,
                ParameterHistory: [new Dictionary<string, object>(parameters)]
            );
            population.Add(member);
        }

        return population;
    }

    /// <summary>
    /// Report the performance of a member at a checkpoint.
    /// Returns an updated PbtMember record.
    /// </summary>
    public PbtMember Report(PbtMember member, double performance, int step)
    {
        return new PbtMember(
            member.Id,
            member.Parameters,
            performance,
            step,
            member.ParameterHistory
        );
    }

    /// <summary>
    /// Evolve the population based on performance (exploit/explore).
    /// Returns a new population list with updated members.
    ///
    /// Algorithm:
    /// 1. Sort by performance descending (higher is better)
    /// 2. Top members (1 - exploitFraction) remain unchanged
    /// 3. Bottom members are assigned to random top members and perturbed
    /// </summary>
    public IReadOnlyList<PbtMember> Evolve(IReadOnlyList<PbtMember> population)
    {
        if (population.Count == 0)
            return [];

        // Sort by performance descending (best first)
        var sorted = population.OrderByDescending(m => m.Performance).ToList();

        var nKeep = Math.Max(1, (int)Math.Floor((1.0 - _exploitFraction) * population.Count));
        var nExploit = population.Count - nKeep;

        var newPopulation = new List<PbtMember>(population.Count);

        // Keep top performers unchanged
        for (int i = 0; i < nKeep; i++)
            newPopulation.Add(sorted[i]);

        // Exploit bottom performers: copy from top, then perturb
        var topMembers = sorted.Take(nKeep).ToList();

        for (int i = 0; i < nExploit; i++)
        {
            var targetMember = topMembers[_rng.Next(topMembers.Count)];
            var newParameters = PerturbParameters(new Dictionary<string, object>(targetMember.Parameters));

            var history = new List<IReadOnlyDictionary<string, object>>(targetMember.ParameterHistory)
            {
                newParameters
            };

            var evolved = new PbtMember(
                sorted[nKeep + i].Id,
                newParameters,
                double.NegativeInfinity, // Reset performance
                0, // Reset step
                history.AsReadOnly()
            );

            newPopulation.Add(evolved);
        }

        return newPopulation.AsReadOnly();
    }

    private Dictionary<string, object> PerturbParameters(Dictionary<string, object> parameters)
    {
        var perturbed = new Dictionary<string, object>(parameters);

        foreach (var range in _searchSpace)
        {
            switch (range)
            {
                case FloatRange fr:
                    var floatValue = (double)parameters[range.Name];
                    var floatFactor = _rng.NextDouble() * 2 * _perturbFactor + (1 - _perturbFactor);
                    var newFloatValue = floatValue * floatFactor;
                    newFloatValue = Math.Max(fr.Low, Math.Min(fr.High, newFloatValue));
                    perturbed[range.Name] = newFloatValue;
                    break;

                case IntRange ir:
                    var intValue = (int)parameters[range.Name];
                    var intFactor = _rng.NextDouble() * 2 * _perturbFactor + (1 - _perturbFactor);
                    var newIntValue = (int)Math.Round(intValue * intFactor);
                    newIntValue = Math.Max(ir.Low, Math.Min(ir.High, newIntValue));
                    perturbed[range.Name] = newIntValue;
                    break;

                case CategoricalRange cr:
                    // Keep with 50% probability, otherwise resample
                    if (_rng.NextDouble() < 0.5)
                    {
                        var newChoice = cr.Choices[_rng.Next(cr.Choices.Length)];
                        perturbed[range.Name] = newChoice;
                    }
                    break;
            }
        }

        return perturbed;
    }
}
