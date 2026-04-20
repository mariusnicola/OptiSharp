using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace OptiSharp.Samplers.Tpe;

/// <summary>
/// Multivariate kernel density estimator using a joint Gaussian distribution.
/// Fits a multivariate normal N(mu, C) to observed samples using Ledoit-Wolf shrinkage
/// for numerical stability when n &lt; D.
/// </summary>
internal sealed class MultivariateKde
{
    private readonly Vector<double> _mu;              // Mean vector (D)
    private readonly Matrix<double> _cholesky;        // Lower Cholesky factor of C + ridge
    private readonly double _logDetC;                 // log(det(C)) for PDF computation
    private readonly double[] _low;                   // Parameter lower bounds
    private readonly double[] _high;                  // Parameter upper bounds
    private readonly int _dim;

    /// <summary>
    /// Fit a multivariate Gaussian N(mu, C) to observations using Ledoit-Wolf shrinkage.
    /// C is regularized as: C_reg = (1-α)*C_sample + α*(tr(C_sample)/D)*I
    /// where α depends on conditioning: higher for ill-conditioned C.
    /// </summary>
    public MultivariateKde(double[][] observations, double[] low, double[] high)
    {
        _dim = low.Length;
        _low = low;
        _high = high;

        if (observations.Length < 2)
            throw new ArgumentException($"Need at least 2 observations, got {observations.Length}");

        if (observations[0].Length != _dim)
            throw new ArgumentException($"Observation dimension {observations[0].Length} != expected {_dim}");

        // Compute sample mean
        var muArray = new double[_dim];
        foreach (var obs in observations)
        {
            for (var i = 0; i < _dim; i++)
                muArray[i] += obs[i];
        }
        for (var i = 0; i < _dim; i++)
            muArray[i] /= observations.Length;
        _mu = Vector<double>.Build.DenseOfArray(muArray);

        // Compute sample covariance
        var covArray = new double[_dim, _dim];
        foreach (var obs in observations)
        {
            for (var i = 0; i < _dim; i++)
            {
                for (var j = 0; j < _dim; j++)
                {
                    covArray[i, j] += (obs[i] - muArray[i]) * (obs[j] - muArray[j]);
                }
            }
        }
        var n = observations.Length;
        for (var i = 0; i < _dim; i++)
        {
            for (var j = 0; j < _dim; j++)
                covArray[i, j] /= n;
        }

        // Ledoit-Wolf shrinkage toward identity
        var cov = Matrix<double>.Build.DenseOfArray(covArray);
        var trace = cov.Trace();
        var shrinkTarget = (trace / _dim) * Matrix<double>.Build.DenseIdentity(_dim);

        // Compute shrinkage intensity: higher when condition number is large
        var cond = EstimateConditionNumber(cov);
        var alpha = Math.Min(1.0, Math.Max(0.01, Math.Log10(cond) / 10.0)); // Scale from 0.01 to 1.0

        var covRegularized = (1.0 - alpha) * cov + alpha * shrinkTarget;

        // Add numerical ridge if needed (safeguard for Cholesky)
        var ridge = 1e-5;
        for (var i = 0; i < _dim; i++)
            covRegularized[i, i] += ridge;

        // Cholesky decomposition
        try
        {
            _cholesky = covRegularized.Cholesky().Factor;
            _logDetC = 2.0 * Enumerable.Range(0, _dim).Sum(i => Math.Log(Math.Abs(_cholesky[i, i])));
        }
        catch (Exception ex)
        {
            throw new ArgumentException($"Cholesky decomposition failed (possibly singular covariance): {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Sample from the multivariate Gaussian: mu + L*z where z ~ N(0,I) and LL^T = C.
    /// </summary>
    public double[] Sample(Random rng)
    {
        var z = new double[_dim];
        for (var i = 0; i < _dim; i++)
            z[i] = Normal.Sample(rng, 0, 1);

        // Multiply by Cholesky factor: y = L*z
        var y = new double[_dim];
        for (var i = 0; i < _dim; i++)
        {
            for (var j = 0; j <= i; j++)
                y[i] += _cholesky[i, j] * z[j];
        }

        // Return mu + y, clamped to bounds
        var result = new double[_dim];
        for (var i = 0; i < _dim; i++)
            result[i] = Math.Clamp(_mu[i] + y[i], _low[i], _high[i]);
        return result;
    }

    /// <summary>
    /// Log PDF of x under the multivariate Gaussian.
    /// log p(x) = -0.5*(D*log(2π) + log(det(C)) + (x-mu)^T C^-1 (x-mu))
    /// </summary>
    public double LogPdf(double[] x)
    {
        if (x.Length != _dim)
            throw new ArgumentException($"Input dimension {x.Length} != model dimension {_dim}");

        // Compute centered vector: dx = x - mu
        var dx = new double[_dim];
        for (var i = 0; i < _dim; i++)
            dx[i] = x[i] - _mu[i];

        // Solve L*L^T * z = dx for z via forward substitution: L*y = dx
        var y = new double[_dim];
        for (var i = 0; i < _dim; i++)
        {
            y[i] = dx[i];
            for (var j = 0; j < i; j++)
                y[i] -= _cholesky[i, j] * y[j];
            y[i] /= _cholesky[i, i];
        }

        // Compute Mahalanobis distance: ||y||^2 = dx^T C^-1 dx
        var mahal = 0.0;
        for (var i = 0; i < _dim; i++)
            mahal += y[i] * y[i];

        return -0.5 * (_dim * Math.Log(2.0 * Math.PI) + _logDetC + mahal);
    }

    /// <summary>
    /// Rough estimate of condition number for shrinkage tuning.
    /// Returns max(|λ|) / min(|λ|) where λ are eigenvalues.
    /// For speed, just use trace/min-diagonal as a heuristic.
    /// </summary>
    private static double EstimateConditionNumber(Matrix<double> mat)
    {
        var minDiag = double.MaxValue;
        for (var i = 0; i < mat.RowCount; i++)
            minDiag = Math.Min(minDiag, Math.Abs(mat[i, i]));

        var trace = mat.Trace();
        return trace / Math.Max(minDiag, 1e-10);
    }
}
