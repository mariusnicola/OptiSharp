namespace OptiSharp.Models;

/// <summary>
/// Compute backend for GPU-accelerated optimization operations.
/// </summary>
public enum ComputeBackend
{
    /// <summary>CPU-only computation (default). Best for dimensions &lt; 100.</summary>
    Cpu,

    /// <summary>GPU computation via CUDA. Throws if no CUDA device found.</summary>
    Gpu,

    /// <summary>Auto-detect: use GPU if available, otherwise CPU.</summary>
    Auto
}
