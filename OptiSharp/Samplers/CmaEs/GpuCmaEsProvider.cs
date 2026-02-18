using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using OptiSharp.Models;

namespace OptiSharp.Samplers.CmaEs;

/// <summary>
/// ILGPU-based GPU compute provider for CMA-ES hot-path operations.
/// Provides GPU-accelerated population sampling (batched matmul) and
/// rank-mu covariance update (batched outer products).
/// </summary>
internal sealed class GpuCmaEsProvider : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;

    // Compiled kernels
    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>,
        double, ArrayView<double>, int, int> _populationKernel;

    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>,
        ArrayView<double>, int, int> _rankMuKernel;

    // Cached GPU buffers (reused across calls for same dimensions)
    private MemoryBuffer1D<double, Stride1D.Dense>? _bdBuf;
    private MemoryBuffer1D<double, Stride1D.Dense>? _zBuf;
    private MemoryBuffer1D<double, Stride1D.Dense>? _meanBuf;
    private MemoryBuffer1D<double, Stride1D.Dense>? _popOutputBuf;
    private MemoryBuffer1D<double, Stride1D.Dense>? _artmpBuf;
    private MemoryBuffer1D<double, Stride1D.Dense>? _weightsBuf;
    private MemoryBuffer1D<double, Stride1D.Dense>? _rankMuBuf;
    private int _cachedN;
    private int _cachedLambda;
    private int _cachedMu;

    public bool IsGpu { get; }
    public string DeviceName { get; }

    private GpuCmaEsProvider(Context context, Accelerator accelerator)
    {
        _context = context;
        _accelerator = accelerator;
        IsGpu = accelerator is CudaAccelerator;
        DeviceName = accelerator.Name;

        _populationKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>,
            double, ArrayView<double>, int, int>(PopulationSampleKernel);

        _rankMuKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<double>, ArrayView<double>,
            ArrayView<double>, int, int>(RankMuUpdateKernel);
    }

    /// <summary>
    /// Check if any CUDA device is available on this system.
    /// </summary>
    public static bool IsCudaAvailable()
    {
        try
        {
            using var ctx = Context.Create(b => b.Cuda());
            foreach (var device in ctx)
            {
                if (device.AcceleratorType == AcceleratorType.Cuda)
                    return true;
            }
            return false;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Try to create a GPU compute provider based on the requested backend.
    /// Returns null if Auto mode and no GPU found. Throws if Gpu mode and no GPU.
    /// </summary>
    public static GpuCmaEsProvider? TryCreate(ComputeBackend backend)
    {
        if (backend == ComputeBackend.Cpu)
            return null;

        try
        {
            var context = Context.Create(b => b.Cuda());

            Device? cudaDevice = null;
            foreach (var device in context)
            {
                if (device.AcceleratorType == AcceleratorType.Cuda)
                {
                    cudaDevice = device;
                    break;
                }
            }

            if (cudaDevice is not null)
            {
                var accelerator = cudaDevice.CreateAccelerator(context);
                return new GpuCmaEsProvider(context, accelerator);
            }

            context.Dispose();

            if (backend == ComputeBackend.Gpu)
                throw new InvalidOperationException(
                    "GPU compute requested but no CUDA device found. " +
                    "Ensure NVIDIA drivers are installed. Use ComputeBackend.Auto for fallback.");

            return null;
        }
        catch (Exception) when (backend == ComputeBackend.Auto)
        {
            return null;
        }
    }

    /// <summary>
    /// GPU-accelerated population sampling.
    /// Computes: output[i,j] = mean[j] + sigma * dot(BD[j,:], z[i,:])
    /// where BD = B * diag(D), Z = random normals.
    /// All arrays are flat row-major: bd[N*N], z[lambda*N], mean[N], output[lambda*N].
    /// </summary>
    public void SamplePopulation(
        double[] bd, double[] z, double[] mean, double sigma,
        double[] output, int lambda, int n)
    {
        EnsurePopulationBuffers(lambda, n);

        _bdBuf!.CopyFromCPU(bd);
        _zBuf!.CopyFromCPU(z);
        _meanBuf!.CopyFromCPU(mean);

        _populationKernel(lambda * n, _bdBuf!.View, _zBuf!.View, _meanBuf!.View,
            sigma, _popOutputBuf!.View, lambda, n);

        _accelerator.Synchronize();
        _popOutputBuf.CopyToCPU(output);
    }

    /// <summary>
    /// GPU-accelerated rank-mu covariance update.
    /// Computes: output[r,c] = sum(weights[i] * artmp[i,r] * artmp[i,c])
    /// All arrays are flat row-major: artmp[mu*N], weights[mu], output[N*N].
    /// </summary>
    public void ComputeRankMu(
        double[] artmp, double[] weights, double[] output, int mu, int n)
    {
        EnsureRankMuBuffers(mu, n);

        _artmpBuf!.CopyFromCPU(artmp);
        _weightsBuf!.CopyFromCPU(weights);

        _rankMuKernel(n * n, _artmpBuf!.View, _weightsBuf!.View,
            _rankMuBuf!.View, mu, n);

        _accelerator.Synchronize();
        _rankMuBuf.CopyToCPU(output);
    }

    // --- GPU Kernels ---

    /// <summary>
    /// Each thread computes one element of the output matrix.
    /// Thread index maps to (candidate, dimension) via division/modulo.
    /// </summary>
    private static void PopulationSampleKernel(
        Index1D index,
        ArrayView<double> bd,
        ArrayView<double> z,
        ArrayView<double> mean,
        double sigma,
        ArrayView<double> output,
        int lambda,
        int n)
    {
        int totalWork = lambda * n;
        if (index >= totalWork) return;

        int i = index / n;  // candidate index
        int j = index % n;  // dimension index

        double dot = 0;
        for (int k = 0; k < n; k++)
            dot += bd[j * n + k] * z[i * n + k];

        output[i * n + j] = mean[j] + sigma * dot;
    }

    /// <summary>
    /// Each thread computes one element of the N x N rank-mu matrix.
    /// Thread index maps to (row, col) via division/modulo.
    /// </summary>
    private static void RankMuUpdateKernel(
        Index1D index,
        ArrayView<double> artmp,
        ArrayView<double> weights,
        ArrayView<double> output,
        int mu,
        int n)
    {
        int totalWork = n * n;
        if (index >= totalWork) return;

        int r = index / n;
        int c = index % n;

        double sum = 0;
        for (int i = 0; i < mu; i++)
            sum += weights[i] * artmp[i * n + r] * artmp[i * n + c];

        output[r * n + c] = sum;
    }

    // --- Buffer management ---

    private void EnsurePopulationBuffers(int lambda, int n)
    {
        if (_cachedN == n && _cachedLambda == lambda
            && _bdBuf is not null && _zBuf is not null
            && _meanBuf is not null && _popOutputBuf is not null)
            return;

        _bdBuf?.Dispose();
        _zBuf?.Dispose();
        _meanBuf?.Dispose();
        _popOutputBuf?.Dispose();

        _bdBuf = _accelerator.Allocate1D<double>(n * n);
        _zBuf = _accelerator.Allocate1D<double>(lambda * n);
        _meanBuf = _accelerator.Allocate1D<double>(n);
        _popOutputBuf = _accelerator.Allocate1D<double>(lambda * n);

        _cachedN = n;
        _cachedLambda = lambda;
    }

    private void EnsureRankMuBuffers(int mu, int n)
    {
        if (_cachedMu == mu && _cachedN == n
            && _artmpBuf is not null && _weightsBuf is not null
            && _rankMuBuf is not null)
            return;

        _artmpBuf?.Dispose();
        _weightsBuf?.Dispose();
        _rankMuBuf?.Dispose();

        _artmpBuf = _accelerator.Allocate1D<double>(mu * n);
        _weightsBuf = _accelerator.Allocate1D<double>(mu);
        _rankMuBuf = _accelerator.Allocate1D<double>(n * n);

        _cachedMu = mu;
    }

    public void Dispose()
    {
        try
        {
            _bdBuf?.Dispose();
            _zBuf?.Dispose();
            _meanBuf?.Dispose();
            _popOutputBuf?.Dispose();
            _artmpBuf?.Dispose();
            _weightsBuf?.Dispose();
            _rankMuBuf?.Dispose();
        }
        finally
        {
            try { _accelerator.Dispose(); }
            finally { _context.Dispose(); }
        }
    }
}
