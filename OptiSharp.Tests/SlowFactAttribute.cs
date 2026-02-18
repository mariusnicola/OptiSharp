namespace OptiSharp.Tests;

/// <summary>
/// Marks a test as slow. Skipped by default.
/// Set environment variable RUN_SLOW_TESTS=1 to run.
/// </summary>
public sealed class SlowFactAttribute : FactAttribute
{
    public SlowFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("RUN_SLOW_TESTS") != "1")
            Skip = "Slow test. Set RUN_SLOW_TESTS=1 to run.";
    }
}
