namespace OptiSharp.Samplers.Tpe;

/// <summary>
/// Kernel density estimator using mixture of truncated Gaussians + uniform prior.
/// Reused for both l(x) (good trials) and g(x) (bad trials) in TPE.
/// </summary>
internal sealed class ParzenEstimator
{
    private readonly double[] _mus;       // Observation centers
    private readonly double[] _sigmas;    // Per-observation bandwidths
    private readonly double[] _logWeights; // Log of mixture weights (observations + prior)
    private readonly double _low;
    private readonly double _high;
    private readonly int _nComponents;    // observations.Length (prior is extra)

    public ParzenEstimator(
        double[] sortedObservations,
        double low,
        double high,
        double priorWeight,
        bool considerMagicClip)
    {
        _low = low;
        _high = high;
        _nComponents = sortedObservations.Length;
        _mus = sortedObservations;

        // Compute bandwidths
        _sigmas = ComputeBandwidths(sortedObservations, low, high, considerMagicClip);

        // Compute weights: equal weight for each observation + prior
        var totalComponents = _nComponents + 1; // +1 for uniform prior
        var rawWeights = new double[totalComponents];
        for (var i = 0; i < _nComponents; i++)
            rawWeights[i] = 1.0;
        rawWeights[_nComponents] = priorWeight;

        // Normalize and take log
        var sum = 0.0;
        foreach (var w in rawWeights) sum += w;

        _logWeights = new double[totalComponents];
        for (var i = 0; i < totalComponents; i++)
            _logWeights[i] = Math.Log(rawWeights[i] / sum);
    }

    /// <summary>
    /// Sample count values from the mixture distribution.
    /// </summary>
    public double[] Sample(Random rng, int count)
    {
        var samples = new double[count];
        var totalComponents = _nComponents + 1;

        // Build cumulative weights for component selection
        var cumWeights = new double[totalComponents];
        cumWeights[0] = Math.Exp(_logWeights[0]);
        for (var i = 1; i < totalComponents; i++)
            cumWeights[i] = cumWeights[i - 1] + Math.Exp(_logWeights[i]);

        for (var s = 0; s < count; s++)
        {
            // Pick component
            var u = rng.NextDouble();
            var k = 0;
            while (k < totalComponents - 1 && u > cumWeights[k])
                k++;

            if (k < _nComponents)
            {
                // Sample from truncated Gaussian centered at observation k
                samples[s] = TruncatedNormal.Sample(rng, _mus[k], _sigmas[k], _low, _high);
            }
            else
            {
                // Sample from uniform prior
                samples[s] = _low + rng.NextDouble() * (_high - _low);
            }
        }

        return samples;
    }

    /// <summary>
    /// Compute log probability density for each value under the mixture.
    /// </summary>
    public double[] LogPdf(double[] values)
    {
        var result = new double[values.Length];
        var totalComponents = _nComponents + 1;
        var logUniformPdf = -Math.Log(_high - _low);

        // Temp buffer for LogSumExp — reuse across values
        Span<double> componentLogPdfs = totalComponents <= 128
            ? stackalloc double[totalComponents]
            : new double[totalComponents];

        for (var v = 0; v < values.Length; v++)
        {
            var x = values[v];

            // Compute log(weight_k * pdf_k(x)) for each component
            for (var k = 0; k < _nComponents; k++)
            {
                componentLogPdfs[k] = _logWeights[k] +
                    TruncatedNormal.LogPdf(x, _mus[k], _sigmas[k], _low, _high);
            }

            // Uniform prior component
            componentLogPdfs[_nComponents] = _logWeights[_nComponents] + logUniformPdf;

            result[v] = TruncatedNormal.LogSumExp(componentLogPdfs[..totalComponents]);
        }

        return result;
    }

    /// <summary>
    /// Compute per-observation bandwidths using nearest-neighbor spacing.
    /// </summary>
    private static double[] ComputeBandwidths(
        double[] sortedObs, double low, double high, bool considerMagicClip)
    {
        var n = sortedObs.Length;
        var sigmas = new double[n];

        if (n == 0)
            return sigmas;

        if (n == 1)
        {
            // Single observation: bandwidth spans the full range
            sigmas[0] = high - low;
            return sigmas;
        }

        for (var i = 0; i < n; i++)
        {
            var leftDist = i == 0 ? sortedObs[0] - low : sortedObs[i] - sortedObs[i - 1];
            var rightDist = i == n - 1 ? high - sortedObs[n - 1] : sortedObs[i + 1] - sortedObs[i];
            sigmas[i] = Math.Max(leftDist, rightDist);
        }

        if (considerMagicClip)
        {
            // Enforce minimum bandwidth — from Optuna ParzenEstimator implementation
            var minSigma = (high - low) / Math.Min(100.0, 1.0 + n);
            for (var i = 0; i < n; i++)
                sigmas[i] = Math.Max(sigmas[i], minSigma);
        }

        // Ensure no zero/negative sigmas
        for (var i = 0; i < n; i++)
            sigmas[i] = Math.Max(sigmas[i], 1e-12);

        return sigmas;
    }
}
