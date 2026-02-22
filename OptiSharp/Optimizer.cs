using OptiSharp.Models;
using OptiSharp.Pruning;
using OptiSharp.Samplers;
using OptiSharp.Samplers.CmaEs;
using OptiSharp.Samplers.Tpe;
using System;

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
        TpeSamplerConfig? config = null,
        IPruner? pruner = null,
        IEnumerable<Trial>? warmStartTrials = null,
        Study? fromStudy = null)
    {
        var sampler = new TpeSampler(config);
        var study = new Study(name, sampler, searchSpace, direction, pruner);

        // Pre-populate with warm trials
        var trials = warmStartTrials ?? fromStudy?.Trials;
        if (trials != null)
            study.PrePopulateWarmTrials(trials);

        return study;
    }

    /// <summary>
    /// Creates a study with random sampler.
    /// </summary>
    public static Study CreateStudyWithRandomSampler(
        string name,
        SearchSpace searchSpace,
        StudyDirection direction = StudyDirection.Minimize,
        int? seed = null,
        IPruner? pruner = null,
        IEnumerable<Trial>? warmStartTrials = null,
        Study? fromStudy = null)
    {
        var sampler = new RandomSampler(seed);
        var study = new Study(name, sampler, searchSpace, direction, pruner);

        // Pre-populate with warm trials
        var trials = warmStartTrials ?? fromStudy?.Trials;
        if (trials != null)
            study.PrePopulateWarmTrials(trials);

        return study;
    }

    /// <summary>
    /// Creates a study with CMA-ES sampler.
    /// </summary>
    public static Study CreateStudyWithCmaEs(
        string name,
        SearchSpace searchSpace,
        StudyDirection direction = StudyDirection.Minimize,
        CmaEsSamplerConfig? config = null,
        IPruner? pruner = null,
        IEnumerable<Trial>? warmStartTrials = null,
        Study? fromStudy = null)
    {
        var sampler = new CmaEsSampler(config);
        var study = new Study(name, sampler, searchSpace, direction, pruner);

        // Pre-populate with warm trials
        var trials = warmStartTrials ?? fromStudy?.Trials;
        if (trials != null)
            study.PrePopulateWarmTrials(trials);

        return study;
    }

    /// <summary>
    /// Creates a study with a custom sampler.
    /// </summary>
    public static Study CreateStudy(
        string name,
        SearchSpace searchSpace,
        ISampler sampler,
        StudyDirection direction = StudyDirection.Minimize,
        IPruner? pruner = null,
        IEnumerable<Trial>? warmStartTrials = null,
        Study? fromStudy = null)
    {
        var study = new Study(name, sampler, searchSpace, direction, pruner);

        // Pre-populate with warm trials
        var trials = warmStartTrials ?? fromStudy?.Trials;
        if (trials != null)
            study.PrePopulateWarmTrials(trials);

        return study;
    }

    /// <summary>
    /// Creates a multi-objective study with TPE sampler.
    /// </summary>
    public static Study CreateStudy(
        string name,
        SearchSpace searchSpace,
        StudyDirection[] directions,
        TpeSamplerConfig? config = null,
        IPruner? pruner = null,
        IEnumerable<Trial>? warmStartTrials = null,
        Study? fromStudy = null)
    {
        var sampler = new TpeSampler(config);
        var study = new Study(name, sampler, searchSpace, directions, pruner);

        // Pre-populate with warm trials
        var trials = warmStartTrials ?? fromStudy?.Trials;
        if (trials != null)
            study.PrePopulateWarmTrials(trials);

        return study;
    }

    /// <summary>
    /// Creates a multi-objective study with a custom sampler.
    /// </summary>
    public static Study CreateStudy(
        string name,
        SearchSpace searchSpace,
        ISampler sampler,
        StudyDirection[] directions,
        IPruner? pruner = null,
        IEnumerable<Trial>? warmStartTrials = null,
        Study? fromStudy = null)
    {
        var study = new Study(name, sampler, searchSpace, directions, pruner);

        // Pre-populate with warm trials
        var trials = warmStartTrials ?? fromStudy?.Trials;
        if (trials != null)
            study.PrePopulateWarmTrials(trials);

        return study;
    }
}
