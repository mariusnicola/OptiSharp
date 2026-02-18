using OptiSharp.Samplers.CmaEs;

namespace OptiSharp.Tests;

/// <summary>
/// Marks a test as requiring a CUDA GPU. Skipped when no GPU is available.
/// </summary>
public sealed class GpuFactAttribute : FactAttribute
{
    public GpuFactAttribute()
    {
        if (!GpuCmaEsProvider.IsCudaAvailable())
            Skip = "No CUDA GPU available.";
    }
}
