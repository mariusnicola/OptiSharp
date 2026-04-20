using MathNet.Numerics.Distributions;

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
    private readonly double[] _logZ;      // log(Phi((high-mu_k)/sigma_k) - Phi((low-mu_k)/sigma_k)) per component
    private readonly double[] _logSigmas; // log(sigma_k), cached to avoid Math.Log in hot loop
    private readonly double[] _cdfLow;    // Phi((low - mu_k) / sigma_k) per component
    private readonly double[] _cdfRange;  // cdfHigh_k - cdfLow_k per component
    private readonly double _low;
    private readonly double _high;
    private readonly int _nComponents;    // observations.Length (prior is extra)

    public ParzenEstimator(
        double[] sortedObservations,
        double low,
        double high,
        double priorWeight,
        bool considerMagicClip,
        bool isBelowEstimator = false)
    {
        _low = low;
        _high = high;
        _nComponents = sortedObservations.Length;
        _mus = sortedObservations;

        // Compute bandwidths
        _sigmas = ComputeBandwidths(sortedObservations, low, high, considerMagicClip);

        // Compute weights: recency-weighted for below estimator (l(x)), equal for above (g(x))
        var totalComponents = _nComponents + 1; // +1 for uniform prior
        var rawWeights = new double[totalComponents];

        // Apply recency weighting only to l(x) (below) estimator
        if (isBelowEstimator && _nComponents >= 25)
        {
            // Optuna's default_weights: ramp from 1/n to 1.0 for oldest (n-25) trials, flat 1.0 for newest 25
            int rampCount = _nComponents - 25;
            for (int i = 0; i < rampCount; i++)
                rawWeights[i] = (i + 1.0) / _nComponents;  // 1/n, 2/n, ..., (n-25)/n
            for (int i = rampCount; i < _nComponents; i++)
                rawWeights[i] = 1.0;  // flat 1.0 for newest 25
        }
        else
        {
            // Equal weight for all observations (used for above estimator and small n_below)
            for (var i = 0; i < _nComponents; i++)
                rawWeights[i] = 1.0;
        }

        rawWeights[_nComponents] = priorWeight;

        // Normalize and take log
        var sum = 0.0;
        foreach (var w in rawWeights) sum += w;

        _logWeights = new double[totalComponents];
        for (var i = 0; i < totalComponents; i++)
            _logWeights[i] = Math.Log(rawWeights[i] / sum);

        // Pre-compute cached math constants to avoid redundant MathNet calls in hot loops
        _logZ = new double[_nComponents];
        _logSigmas = new double[_nComponents];
        _cdfLow = new double[_nComponents];
        _cdfRange = new double[_nComponents];

        for (var k = 0; k < _nComponents; k++)
        {
            _logSigmas[k] = Math.Log(_sigmas[k]);

            // Pre-compute CDF values for truncation boundaries (fixed per component)
            var cdfHigh = TruncatedNormal.Phi((_high - _mus[k]) / _sigmas[k]);
            var cdfLow = TruncatedNormal.Phi((_low - _mus[k]) / _sigmas[k]);
            _logZ[k] = Math.Log(Math.Max(cdfHigh - cdfLow, double.Epsilon));
            _cdfLow[k] = cdfLow;
            _cdfRange[k] = cdfHigh - cdfLow;
        }
    }

    /// <summary>
    /// Sample count values from the mixture distribution.
    /// </summary>
    public double[] Sample(Random rng, int count)
    {
        var samples = new double[count];
        var totalComponents = _nComponents + 1;

        // Build cumulative weights for component selection
        // Use stackalloc for small arrays (totalComponents ≤ 26 typical), heap fallback for large
        Span<double> cumWeights = totalComponents <= 128
            ? stackalloc double[totalComponents]
            : new double[totalComponents];
        cumWeights[0] = Math.Exp(_logWeights[0]);
        for (var i = 1; i < totalComponents; i++)
            cumWeights[i] = cumWeights[i - 1] + Math.Exp(_logWeights[i]);

        for (var s = 0; s < count; s++)
        {
            // Pick component: cumWeights is monotone non-decreasing, use binary search
            var u = rng.NextDouble();
            var searchSpan = cumWeights[..totalComponents];
            var k = MemoryExtensions.BinarySearch(searchSpan, u);
            if (k < 0) k = ~k; // BinarySearch returns bitwise complement when not found (position to insert)
            k = Math.Min(k, totalComponents - 1);

            if (k < _nComponents)
            {
                // Sample from truncated Gaussian using cached boundary CDFs
                var range = _cdfRange[k];
                if (range < double.Epsilon)
                {
                    samples[s] = (_low + _high) * 0.5;
                }
                else
                {
                    var p = Math.Clamp(_cdfLow[k] + rng.NextDouble() * range, 1e-15, 1.0 - 1e-15);
                    samples[s] = Math.Clamp(_mus[k] + _sigmas[k] * Normal.InvCDF(0, 1, p), _low, _high);
                }
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

            // Early exit if x is outside bounds — all components have log-pdf = -infinity
            if (x < _low || x > _high)
            {
                result[v] = double.NegativeInfinity;
                continue;
            }

            // Compute log(weight_k * pdf_k(x)) for each component using cached math
            for (var k = 0; k < _nComponents; k++)
            {
                // log N(x | mu_k, sigma_k) - log Z_k
                // where N(x|mu,sigma) = exp(-0.5*log(2pi) - log(sigma) - 0.5*(x-mu)^2/sigma^2)
                var z = (x - _mus[k]) / _sigmas[k];
                var logNorm = -0.9189385332046727 - _logSigmas[k] - 0.5 * z * z;
                componentLogPdfs[k] = _logWeights[k] + logNorm - _logZ[k];
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
            // Use neighbor distance for edge observations, not boundary distance (Optuna behavior).
            // n >= 2 guaranteed by early-return at line 130.
            var leftDist = i == 0
                ? sortedObs[1] - sortedObs[0]
                : sortedObs[i] - sortedObs[i - 1];
            var rightDist = i == n - 1
                ? sortedObs[n - 1] - sortedObs[n - 2]
                : sortedObs[i + 1] - sortedObs[i];
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
