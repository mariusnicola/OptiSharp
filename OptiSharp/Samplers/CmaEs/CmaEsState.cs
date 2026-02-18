using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Factorization;

namespace OptiSharp.Samplers.CmaEs;

/// <summary>
/// Mutable internal state for the CMA-ES algorithm.
/// Holds the distribution parameters and strategy constants.
/// Optionally uses GPU acceleration for population sampling and rank-mu update.
/// </summary>
internal sealed class CmaEsState
{
    private readonly GpuCmaEsProvider? _gpu;

    // Distribution parameters
    private Vector<double> _mean;
    private Matrix<double> _c;          // Covariance matrix
    private double _sigma;              // Global step size
    private Vector<double> _pc;         // Evolution path for C
    private Vector<double> _ps;         // Evolution path for sigma

    // Eigendecomposition cache
    private Matrix<double> _b;          // Eigenvectors
    private Vector<double> _d;          // Sqrt of eigenvalues
    private bool _eigenDirty = true;

    // Strategy constants (derived from n and lambda)
    public int N { get; }               // Dimension count
    public int Lambda { get; }          // Population size
    public int Mu { get; }              // Parent count
    public double[] Weights { get; }    // Recombination weights
    public double MuEff { get; }        // Variance effective selection mass
    public double Cc { get; }           // Learning rate for pc
    public double Cs { get; }           // Learning rate for ps
    public double C1 { get; }           // Learning rate for rank-one update
    public double Cmu { get; }          // Learning rate for rank-mu update
    public double Dsigma { get; }       // Damping for step size
    public double ChiN { get; }         // E[||N(0,I)||]

    public int Generation { get; private set; }

    public CmaEsState(int n, int lambda, double sigma, double[]? initialMean = null, GpuCmaEsProvider? gpu = null)
    {
        N = n;
        Lambda = lambda;
        _sigma = sigma;
        _gpu = gpu;

        // Initialize mean
        _mean = initialMean is not null
            ? DenseVector.OfArray(initialMean)
            : DenseVector.Create(n, 0.5); // Center of [0,1] normalized space

        // Initialize covariance as identity
        _c = DenseMatrix.CreateIdentity(n);

        // Evolution paths
        _pc = DenseVector.Create(n, 0.0);
        _ps = DenseVector.Create(n, 0.0);

        // Eigendecomposition of identity
        _b = DenseMatrix.CreateIdentity(n);
        _d = DenseVector.Create(n, 1.0);
        _eigenDirty = false;

        // Derived constants
        Mu = lambda / 2;
        Weights = ComputeWeights(Mu);
        MuEff = ComputeMuEff(Weights);

        Cc = (4.0 + MuEff / n) / (n + 4.0 + 2.0 * MuEff / n);
        Cs = (MuEff + 2.0) / (n + MuEff + 5.0);
        C1 = 2.0 / ((n + 1.3) * (n + 1.3) + MuEff);
        Cmu = Math.Min(
            1.0 - C1,
            2.0 * (MuEff - 2.0 + 1.0 / MuEff) / ((n + 2.0) * (n + 2.0) + MuEff));
        Dsigma = 1.0 + 2.0 * Math.Max(0, Math.Sqrt((MuEff - 1.0) / (n + 1.0)) - 1.0) + Cs;

        // Expected length of a N(0,I) vector — Hansen & Ostermeier (2001), Eq. 21
        ChiN = Math.Sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));
    }

    public double Sigma => _sigma;
    public Vector<double> Mean => _mean;

    /// <summary>
    /// Condition number of the covariance matrix: max(D)/min(D).
    /// </summary>
    public double ConditionNumber
    {
        get
        {
            EnsureEigen();
            var max = _d.Maximum();
            var min = _d.Minimum();
            return min > 0 ? max / min : double.PositiveInfinity;
        }
    }

    /// <summary>
    /// Generate lambda candidate vectors from the current distribution.
    /// Uses GPU if available, otherwise CPU.
    /// </summary>
    public (Vector<double>[] Candidates, Vector<double>[] ZVectors) SamplePopulation(Random rng)
    {
        EnsureEigen();
        return _gpu is not null
            ? SamplePopulationGpu(rng)
            : SamplePopulationCpu(rng);
    }

    /// <summary>
    /// Update distribution after a complete generation.
    /// rankedCandidates: lambda vectors sorted by fitness (best first).
    /// </summary>
    public void Update(Vector<double>[] rankedCandidates)
    {
        var oldMean = _mean.Clone();

        // 1. New mean: weighted recombination of mu best
        Vector<double> newMean = DenseVector.Create(N, 0.0);
        for (var i = 0; i < Mu; i++)
            newMean = newMean + Weights[i] * rankedCandidates[i];
        _mean = newMean;

        var meanDiff = (_mean - oldMean) / _sigma;

        // 2. Update evolution path for sigma (ps)
        EnsureEigen();
        var cInvSqrt = ComputeCInvSqrt();
        _ps = (1.0 - Cs) * _ps + Math.Sqrt(Cs * (2.0 - Cs) * MuEff) * (cInvSqrt * meanDiff);

        // 3. Heaviside function
        var psNorm = _ps.L2Norm();
        var threshold = (1.4 + 2.0 / (N + 1.0)) * ChiN *
            Math.Sqrt(1.0 - Math.Pow(1.0 - Cs, 2.0 * (Generation + 1)));
        var hSigma = psNorm < threshold ? 1.0 : 0.0;

        // 4. Update evolution path for C (pc)
        _pc = (1.0 - Cc) * _pc + hSigma * Math.Sqrt(Cc * (2.0 - Cc) * MuEff) * meanDiff;

        // 5. Covariance matrix update
        var oldC = _c.Clone();

        // Rank-one: pc * pc^T
        var rankOne = _pc.OuterProduct(_pc);

        // Rank-mu: weighted sum of outer products
        var rankMu = ComputeRankMu(rankedCandidates, oldMean);

        // Correction term when hSigma = 0
        var correction = (1.0 - hSigma) * Cc * (2.0 - Cc);

        _c = (1.0 - C1 - Cmu) * oldC
            + C1 * (rankOne + correction * oldC)
            + Cmu * rankMu;

        // 6. Step size update (CSA)
        _sigma *= Math.Exp((Cs / Dsigma) * (psNorm / ChiN - 1.0));

        // Clamp sigma to prevent explosion
        _sigma = Math.Clamp(_sigma, 1e-20, 1e10);

        // 7. Mark eigen dirty
        _eigenDirty = true;
        Generation++;
    }

    // --- CPU paths ---

    private (Vector<double>[] Candidates, Vector<double>[] ZVectors) SamplePopulationCpu(Random rng)
    {
        var candidates = new Vector<double>[Lambda];
        var zVectors = new Vector<double>[Lambda];

        for (var i = 0; i < Lambda; i++)
        {
            var z = DenseVector.Create(N, _ => SampleStdNormal(rng));
            var y = _b * DiagTimesVector(_d, z);
            candidates[i] = _mean + _sigma * y;
            zVectors[i] = z;
        }

        return (candidates, zVectors);
    }

    private Matrix<double> ComputeRankMuCpu(Vector<double>[] rankedCandidates, Vector<double> oldMean)
    {
        Matrix<double> rankMu = DenseMatrix.Create(N, N, 0.0);
        for (var i = 0; i < Mu; i++)
        {
            var artmp = (rankedCandidates[i] - oldMean) / _sigma;
            rankMu = rankMu + Weights[i] * artmp.OuterProduct(artmp);
        }
        return rankMu;
    }

    // --- GPU paths ---

    private (Vector<double>[] Candidates, Vector<double>[] ZVectors) SamplePopulationGpu(Random rng)
    {
        // Precompute BD = B * diag(D) as flat row-major array
        var bd = new double[N * N];
        for (var i = 0; i < N; i++)
            for (var j = 0; j < N; j++)
                bd[i * N + j] = _b[i, j] * _d[j];

        // Generate random normals on CPU (cheap relative to matmul)
        var z = new double[Lambda * N];
        for (var i = 0; i < Lambda * N; i++)
            z[i] = SampleStdNormal(rng);

        var mean = _mean.ToArray();
        var output = new double[Lambda * N];

        _gpu!.SamplePopulation(bd, z, mean, _sigma, output, Lambda, N);

        // Convert flat arrays to Vector<double>[]
        var candidates = new Vector<double>[Lambda];
        var zVectors = new Vector<double>[Lambda];
        for (var i = 0; i < Lambda; i++)
        {
            var candidateData = new double[N];
            var zData = new double[N];
            Array.Copy(output, i * N, candidateData, 0, N);
            Array.Copy(z, i * N, zData, 0, N);
            candidates[i] = DenseVector.OfArray(candidateData);
            zVectors[i] = DenseVector.OfArray(zData);
        }

        return (candidates, zVectors);
    }

    /// <summary>
    /// Flatten artmp vectors (deviation from old mean, normalized by sigma)
    /// into a row-major array for GPU or reuse across paths.
    /// </summary>
    private double[] FlattenArtmp(Vector<double>[] rankedCandidates, Vector<double> oldMean)
    {
        var flat = new double[Mu * N];
        for (var i = 0; i < Mu; i++)
        {
            var artmp = (rankedCandidates[i] - oldMean) / _sigma;
            for (var j = 0; j < N; j++)
                flat[i * N + j] = artmp[j];
        }
        return flat;
    }

    private Matrix<double> ComputeRankMuGpu(Vector<double>[] rankedCandidates, Vector<double> oldMean)
    {
        var artmpFlat = FlattenArtmp(rankedCandidates, oldMean);

        var rankMuFlat = new double[N * N];
        _gpu!.ComputeRankMu(artmpFlat, Weights, rankMuFlat, Mu, N);

        return DenseMatrix.Create(N, N, (r, c) => rankMuFlat[r * N + c]);
    }

    private Matrix<double> ComputeRankMu(Vector<double>[] rankedCandidates, Vector<double> oldMean)
    {
        return _gpu is not null
            ? ComputeRankMuGpu(rankedCandidates, oldMean)
            : ComputeRankMuCpu(rankedCandidates, oldMean);
    }

    // --- Shared math ---

    private void EnsureEigen()
    {
        if (!_eigenDirty) return;

        // Force symmetry (numerical drift)
        _c = (_c + _c.Transpose()) * 0.5;

        var evd = _c.Evd(Symmetricity.Symmetric);
        _b = evd.EigenVectors;

        // Eigenvalues: take sqrt, clamp negatives to small positive
        var eigenValues = evd.EigenValues;
        _d = DenseVector.Create(N, i =>
            Math.Sqrt(Math.Max(eigenValues[i].Real, 1e-20)));

        _eigenDirty = false;
    }

    private Matrix<double> ComputeCInvSqrt()
    {
        EnsureEigen();
        // C^{-1/2} = B * D^{-1} * B^T
        var dInv = DenseVector.Create(N, i => 1.0 / _d[i]);
        return _b * DiagTimesMatrix(dInv, _b.Transpose());
    }

    private static Vector<double> DiagTimesVector(Vector<double> diag, Vector<double> vec)
    {
        var result = DenseVector.Create(vec.Count, 0.0);
        for (var i = 0; i < vec.Count; i++)
            result[i] = diag[i] * vec[i];
        return result;
    }

    private static Matrix<double> DiagTimesMatrix(Vector<double> diag, Matrix<double> mat)
    {
        var result = mat.Clone();
        for (var i = 0; i < diag.Count; i++)
            for (var j = 0; j < mat.ColumnCount; j++)
                result[i, j] = diag[i] * mat[i, j];
        return result;
    }

    private static double[] ComputeWeights(int mu)
    {
        var rawWeights = new double[mu];
        for (var i = 0; i < mu; i++)
            rawWeights[i] = Math.Log(mu + 0.5) - Math.Log(i + 1);

        var sum = rawWeights.Sum();
        for (var i = 0; i < mu; i++)
            rawWeights[i] /= sum;

        return rawWeights;
    }

    private static double ComputeMuEff(double[] weights)
    {
        var sumSq = weights.Sum(w => w * w);
        return 1.0 / sumSq;
    }

    internal static double SampleStdNormal(Random rng)
    {
        // Box-Muller transform
        var u1 = 1.0 - rng.NextDouble(); // avoid log(0)
        var u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
