using OptiSharp.Models;
using OptiSharp.Samplers;
using OptiSharp.Samplers.CmaEs;
using OptiSharp.Samplers.Tpe;

namespace OptiSharp;

/// <summary>
/// Factory for creating optimization studies.
/// </summary>
public static class Optimizer
{
    /// <summary>
    /// Creates a study with TPE sampler (default).
    /// </summary>
    public static Study CreateStudy(
        string name,
        SearchSpace searchSpace,
        StudyDirection direction = StudyDirection.Minimize,
        TpeSamplerConfig? config = null)
    {
        var sampler = new TpeSampler(config);
        return new Study(name, sampler, searchSpace, direction);
    }

    /// <summary>
    /// Creates a study with random sampler.
    /// </summary>
    public static Study CreateStudyWithRandomSampler(
        string name,
        SearchSpace searchSpace,
        StudyDirection direction = StudyDirection.Minimize,
        int? seed = null)
    {
        var sampler = new RandomSampler(seed);
        return new Study(name, sampler, searchSpace, direction);
    }

    /// <summary>
    /// Creates a study with CMA-ES sampler.
    /// </summary>
    public static Study CreateStudyWithCmaEs(
        string name,
        SearchSpace searchSpace,
        StudyDirection direction = StudyDirection.Minimize,
        CmaEsSamplerConfig? config = null)
    {
        var sampler = new CmaEsSampler(config);
        return new Study(name, sampler, searchSpace, direction);
    }

    /// <summary>
    /// Creates a study with a custom sampler.
    /// </summary>
    public static Study CreateStudy(
        string name,
        SearchSpace searchSpace,
        ISampler sampler,
        StudyDirection direction = StudyDirection.Minimize)
    {
        return new Study(name, sampler, searchSpace, direction);
    }
}
