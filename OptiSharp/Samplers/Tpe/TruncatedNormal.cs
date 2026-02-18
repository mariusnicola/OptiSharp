using MathNet.Numerics.Distributions;

namespace OptiSharp.Samplers.Tpe;

/// <summary>
/// Pure static math utilities for truncated normal distributions.
/// No state, no allocations on the hot path.
/// </summary>
internal static class TruncatedNormal
{
    /// <summary>
    /// Log probability density of x under a truncated normal N(mu, sigma²) on [low, high].
    /// </summary>
    public static double LogPdf(double x, double mu, double sigma, double low, double high)
    {
        if (x < low || x > high)
            return double.NegativeInfinity;

        var logNorm = Normal.PDFLn(mu, sigma, x);
        var cdfHigh = Phi((high - mu) / sigma);
        var cdfLow = Phi((low - mu) / sigma);
        var logZ = Math.Log(Math.Max(cdfHigh - cdfLow, double.Epsilon));
        return logNorm - logZ;
    }

    /// <summary>
    /// Sample from a truncated normal N(mu, sigma²) on [low, high] using inverse CDF.
    /// O(1) per sample — no rejection loops.
    /// </summary>
    public static double Sample(Random rng, double mu, double sigma, double low, double high)
    {
        var cdfLow = Phi((low - mu) / sigma);
        var cdfHigh = Phi((high - mu) / sigma);

        // Avoid degenerate case where cdf range is zero
        var range = cdfHigh - cdfLow;
        if (range < double.Epsilon)
            return (low + high) * 0.5;

        var u = rng.NextDouble();
        var p = cdfLow + u * range;

        // Clamp p to avoid infinities from InvCDF at 0 or 1
        p = Math.Clamp(p, 1e-15, 1.0 - 1e-15);

        var x = mu + sigma * Normal.InvCDF(0, 1, p);
        return Math.Clamp(x, low, high);
    }

    /// <summary>
    /// Standard normal CDF: Phi(z) = P(Z ≤ z).
    /// </summary>
    public static double Phi(double z) => Normal.CDF(0, 1, z);

    /// <summary>
    /// Log-sum-exp for numerical stability: log(Σ exp(x_i)).
    /// </summary>
    public static double LogSumExp(ReadOnlySpan<double> values)
    {
        if (values.Length == 0)
            return double.NegativeInfinity;

        var max = double.NegativeInfinity;
        foreach (var v in values)
            if (v > max) max = v;

        if (double.IsNegativeInfinity(max))
            return double.NegativeInfinity;

        var sum = 0.0;
        foreach (var v in values)
            sum += Math.Exp(v - max);

        return max + Math.Log(sum);
    }
}
