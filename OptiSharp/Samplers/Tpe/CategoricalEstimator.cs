namespace OptiSharp.Samplers.Tpe;

/// <summary>
/// Estimates categorical distributions from observations using frequency counting with prior smoothing.
/// </summary>
internal sealed class CategoricalEstimator
{
    private readonly double[] _logWeights;
    private readonly double[] _cumulativeWeights;

    public int NumChoices { get; }

    public CategoricalEstimator(ReadOnlySpan<int> observations, int numChoices, double priorWeight)
    {
        NumChoices = numChoices;
        var weights = new double[numChoices];
        var priorPerChoice = priorWeight / numChoices;

        // Count observations + add prior
        for (var i = 0; i < numChoices; i++)
            weights[i] = priorPerChoice;

        foreach (var obs in observations)
            weights[obs] += 1.0;

        // Normalize
        var total = 0.0;
        foreach (var w in weights) total += w;
        for (var i = 0; i < numChoices; i++)
            weights[i] /= total;

        // Pre-compute log weights and cumulative for sampling
        _logWeights = new double[numChoices];
        _cumulativeWeights = new double[numChoices];
        var cumulative = 0.0;
        for (var i = 0; i < numChoices; i++)
        {
            _logWeights[i] = Math.Log(Math.Max(weights[i], double.Epsilon));
            cumulative += weights[i];
            _cumulativeWeights[i] = cumulative;
        }
    }

    /// <summary>
    /// Sample a choice index using weighted random selection.
    /// </summary>
    public int Sample(Random rng)
    {
        var u = rng.NextDouble();
        for (var i = 0; i < NumChoices; i++)
        {
            if (u <= _cumulativeWeights[i])
                return i;
        }
        return NumChoices - 1;
    }

    /// <summary>
    /// Log probability of a specific choice.
    /// </summary>
    public double LogPdf(int choiceIndex) => _logWeights[choiceIndex];
}
