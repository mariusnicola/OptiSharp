using System.Diagnostics;
using System.Text.Json;
using OptiSharp;
using OptiSharp.Models;
using OptiSharp.Pruning;
using OptiSharp.Samplers.Tpe;

namespace OptiSharp.Benchmarks;

/// <summary>
/// Unified benchmark framework: Same datasets, same problems for direct OptiSharp vs Optuna comparison
///
/// Features tested:
/// - Single-Objective (TPE, Random, CMA-ES)
/// - Multi-Objective
/// - Constraints
/// - Pruning
///
/// Scales:
/// - Small: 500 trials, 20 parameters
/// - Medium: 5,000 trials, 100 parameters
/// - Large: 50,000 trials, 200 parameters
/// </summary>
public class UnifiedBenchmarkRunner
{
    private readonly string _outputDir = "./benchmark-results";

    private static readonly Dictionary<string, (int Trials, int Params, string Description)> TestScales = new()
    {
        { "Small", (500, 20, "500 trials, 20 params") },
        { "Medium", (5000, 100, "5,000 trials, 100 params") },
        { "Large", (50000, 200, "50,000 trials, 200 params") }
    };

    private static readonly List<string> Features = new() { "SingleObjective", "MultiObjective", "Constraints", "Pruning" };

    public UnifiedBenchmarkRunner()
    {
        Directory.CreateDirectory(_outputDir);
    }

    // ============================================================================
    // UNIFIED DATASET (Identical to Python version)
    // ============================================================================

    /// <summary>Sphere function: sum of x_i^2 (single-objective)</summary>
    private static double SphereFunctionEval(IReadOnlyDictionary<string, object> parameters)
    {
        double sum = 0;
        for (int i = 0; i < 200; i++)  // Support up to 200 params
        {
            var key = $"x_{i}";
            if (parameters.ContainsKey(key) && parameters[key] is double value)
            {
                sum += value * value;
            }
        }
        return sum;
    }

    /// <summary>Multi-objective: Sphere + Rosenbrock (first 10 params only)</summary>
    private static (double obj1, double obj2) MultiObjectiveEval(IReadOnlyDictionary<string, object> parameters, int nParams)
    {
        // Objective 1: Sphere
        double obj1 = 0;
        for (int i = 0; i < nParams; i++)
        {
            var key = $"x_{i}";
            if (parameters.ContainsKey(key) && parameters[key] is double value)
            {
                obj1 += value * value;
            }
        }

        // Objective 2: Rosenbrock (first 10 params only, for speed)
        double obj2 = 0;
        int rosenbrokDims = Math.Min(nParams - 1, 10);
        for (int i = 0; i < rosenbrokDims; i++)
        {
            var key1 = $"x_{i}";
            var key2 = $"x_{i + 1}";
            if (parameters.ContainsKey(key1) && parameters.ContainsKey(key2) &&
                parameters[key1] is double x && parameters[key2] is double xNext)
            {
                double term = 100 * Math.Pow(xNext - x * x, 2) + Math.Pow(1 - x, 2);
                obj2 += term;
            }
        }

        return (obj1, obj2);
    }

    /// <summary>Constraint: sum(|x_i|) < threshold</summary>
    private static double ConstraintEval(IReadOnlyDictionary<string, object> parameters, int nParams)
    {
        double sumAbs = 0;
        for (int i = 0; i < nParams; i++)
        {
            var key = $"x_{i}";
            if (parameters.ContainsKey(key) && parameters[key] is double value)
            {
                sumAbs += Math.Abs(value);
            }
        }
        double threshold = nParams * 2.0;
        return sumAbs - threshold;  // Must be <= 0
    }

    /// <summary>Pruning function: Multi-step evaluation with improvement signal</summary>
    private static double PruningEval(IReadOnlyDictionary<string, object> parameters, int step, int nParams)
    {
        double sphere = SphereFunctionEval(parameters);
        return sphere / (1 + step * 0.1);  // Improves with step count
    }

    // ============================================================================
    // UNIFIED BENCHMARK EXECUTION
    // ============================================================================

    public class UnifiedBenchmarkResult
    {
        public string Feature { get; set; }
        public string Sampler { get; set; }
        public string Scale { get; set; }
        public int NTrials { get; set; }
        public int NParameters { get; set; }
        public double BestValue { get; set; }
        public int ParetoFrontSize { get; set; }
        public double WallClockSeconds { get; set; }
        public double TrialsPerSecond { get; set; }
    }

    public void RunUnifiedBenchmarks(string scale = "Small")
    {
        if (!TestScales.ContainsKey(scale))
        {
            Console.WriteLine($"Unknown scale: {scale}. Valid options: Small, Medium, Large");
            return;
        }

        var (nTrials, nParams, description) = TestScales[scale];
        var results = new List<UnifiedBenchmarkResult>();

        Console.WriteLine($"\n{'='*80}");
        Console.WriteLine($"UNIFIED BENCHMARK: {scale.ToUpper()} ({description})");
        Console.WriteLine($"{'='*80}");

        // SingleObjective - TPE
        try
        {
            Console.WriteLine($"\n[1/4] Single-Objective (TPE)...");
            var result = RunSingleObjective(nTrials, nParams, scale, "TPE");
            results.Add(result);
            Console.WriteLine($"  ✓ Best value: {result.BestValue:F6}, Time: {result.WallClockSeconds:F2}s, Throughput: {result.TrialsPerSecond:F1} trials/sec");
        }
        catch (Exception e)
        {
            Console.WriteLine($"  ✗ Error: {e.Message}");
        }

        // MultiObjective
        try
        {
            Console.WriteLine($"\n[2/4] Multi-Objective (TPE)...");
            var result = RunMultiObjective(nTrials, nParams, scale);
            results.Add(result);
            Console.WriteLine($"  ✓ Pareto front size: {result.ParetoFrontSize}, Time: {result.WallClockSeconds:F2}s, Throughput: {result.TrialsPerSecond:F1} trials/sec");
        }
        catch (Exception e)
        {
            Console.WriteLine($"  ✗ Error: {e.Message}");
        }

        // Constraints
        try
        {
            Console.WriteLine($"\n[3/4] Constraints (TPE)...");
            var result = RunConstraints(nTrials, nParams, scale);
            results.Add(result);
            Console.WriteLine($"  ✓ Best value: {result.BestValue:F6}, Time: {result.WallClockSeconds:F2}s, Throughput: {result.TrialsPerSecond:F1} trials/sec");
        }
        catch (Exception e)
        {
            Console.WriteLine($"  ✗ Error: {e.Message}");
        }

        // Pruning
        try
        {
            Console.WriteLine($"\n[4/4] Pruning (TPE)...");
            var result = RunPruning(nTrials, nParams, scale);
            results.Add(result);
            Console.WriteLine($"  ✓ Best value: {result.BestValue:F6}, Time: {result.WallClockSeconds:F2}s, Throughput: {result.TrialsPerSecond:F1} trials/sec");
        }
        catch (Exception e)
        {
            Console.WriteLine($"  ✗ Error: {e.Message}");
        }

        // Save results
        SaveResults($"unified_benchmark_optisharp_{scale.ToLower()}_results.json", results);

        // Print summary
        Console.WriteLine($"\n{'='*80}");
        Console.WriteLine("SUMMARY:");
        Console.WriteLine($"{'='*80}");
        foreach (var result in results)
        {
            Console.WriteLine($"\n{result.Feature} ({result.Scale}):");
            Console.WriteLine($"  Trials: {result.NTrials}, Params: {result.NParameters}");
            Console.WriteLine($"  Time: {result.WallClockSeconds:F2}s, Throughput: {result.TrialsPerSecond:F1} trials/sec");
            if (result.ParetoFrontSize > 0)
                Console.WriteLine($"  Pareto front size: {result.ParetoFrontSize}");
            if (result.BestValue > 0)
                Console.WriteLine($"  Best Value: {result.BestValue:F6}");
        }
    }

    private UnifiedBenchmarkResult RunSingleObjective(int nTrials, int nParams, string scaleName, string samplerType = "TPE")
    {
        var space = new SearchSpace(
            Enumerable.Range(0, nParams)
                .Select(i => new FloatRange($"x_{i}", -5.0, 5.0))
                .ToList()
        );

        Study study;
        if (samplerType == "TPE")
        {
            study = Optimizer.CreateStudy(
                $"unified_single_obj_{scaleName.ToLower()}_tpe",
                space,
                StudyDirection.Minimize,
                new TpeSamplerConfig { Seed = 42 }
            );
        }
        else if (samplerType == "Random")
        {
            study = Optimizer.CreateStudyWithRandomSampler(
                $"unified_single_obj_{scaleName.ToLower()}_random",
                space,
                StudyDirection.Minimize,
                seed: 42
            );
        }
        else if (samplerType == "CMA-ES")
        {
            study = Optimizer.CreateStudyWithCmaEs(
                $"unified_single_obj_{scaleName.ToLower()}_cmaes",
                space,
                StudyDirection.Minimize
            );
        }
        else
        {
            throw new ArgumentException($"Unknown sampler: {samplerType}");
        }

        using (study)
        {
            var sw = Stopwatch.StartNew();

            for (int trial = 0; trial < nTrials; trial++)
            {
                var trialObj = study.Ask();
                var value = SphereFunctionEval(trialObj.Parameters);
                study.Tell(trialObj.Number, value);

                if ((trial + 1) % Math.Max(1, nTrials / 4) == 0)
                {
                    Console.WriteLine($"  {trial + 1}/{nTrials} trials: {sw.Elapsed.TotalSeconds:F1}s, Best={study.BestTrial?.Value:F6}");
                }
            }

            sw.Stop();

            return new UnifiedBenchmarkResult
            {
                Feature = "SingleObjective",
                Sampler = samplerType,
                Scale = scaleName,
                NTrials = nTrials,
                NParameters = nParams,
                BestValue = study.BestTrial?.Value ?? 0,
                WallClockSeconds = sw.Elapsed.TotalSeconds,
                TrialsPerSecond = nTrials / sw.Elapsed.TotalSeconds
            };
        }
    }

    private UnifiedBenchmarkResult RunMultiObjective(int nTrials, int nParams, string scaleName)
    {
        var space = new SearchSpace(
            Enumerable.Range(0, nParams)
                .Select(i => new FloatRange($"x_{i}", -5.0, 5.0))
                .ToList()
        );

        var directions = new[] { StudyDirection.Minimize, StudyDirection.Minimize };

        using var study = Optimizer.CreateStudy(
            $"unified_multi_obj_{scaleName.ToLower()}",
            space,
            directions,
            new TpeSamplerConfig { Seed = 42 }
        );

        var sw = Stopwatch.StartNew();

        for (int trial = 0; trial < nTrials; trial++)
        {
            var trialObj = study.Ask();
            var (obj1, obj2) = MultiObjectiveEval(trialObj.Parameters, nParams);
            study.Tell(trialObj.Number, new[] { obj1, obj2 });

            if ((trial + 1) % Math.Max(1, nTrials / 4) == 0)
            {
                var paretoSize = study.ParetoFront.Count;
                Console.WriteLine($"  {trial + 1}/{nTrials} trials: {sw.Elapsed.TotalSeconds:F1}s, Pareto={paretoSize}");
            }
        }

        sw.Stop();

        return new UnifiedBenchmarkResult
        {
            Feature = "MultiObjective",
            Sampler = "TPE",
            Scale = scaleName,
            NTrials = nTrials,
            NParameters = nParams,
            ParetoFrontSize = study.ParetoFront.Count,
            WallClockSeconds = sw.Elapsed.TotalSeconds,
            TrialsPerSecond = nTrials / sw.Elapsed.TotalSeconds
        };
    }

    private UnifiedBenchmarkResult RunConstraints(int nTrials, int nParams, string scaleName)
    {
        var space = new SearchSpace(
            Enumerable.Range(0, nParams)
                .Select(i => new FloatRange($"x_{i}", -5.0, 5.0))
                .ToList()
        );

        using var study = Optimizer.CreateStudy(
            $"unified_constraints_{scaleName.ToLower()}",
            space,
            StudyDirection.Minimize,
            new TpeSamplerConfig { Seed = 42 }
        );

        // Set constraint: sum of absolute values < threshold
        study.SetConstraintFunc(trial =>
        {
            double constraint = ConstraintEval(trial.Parameters, nParams);
            return new[] { constraint };
        });

        var sw = Stopwatch.StartNew();

        for (int trial = 0; trial < nTrials; trial++)
        {
            var trialObj = study.Ask();
            var value = SphereFunctionEval(trialObj.Parameters);
            study.Tell(trialObj.Number, value);

            if ((trial + 1) % Math.Max(1, nTrials / 4) == 0)
            {
                Console.WriteLine($"  {trial + 1}/{nTrials} trials: {sw.Elapsed.TotalSeconds:F1}s, Best={study.BestTrial?.Value:F6}");
            }
        }

        sw.Stop();

        return new UnifiedBenchmarkResult
        {
            Feature = "Constraints",
            Sampler = "TPE",
            Scale = scaleName,
            NTrials = nTrials,
            NParameters = nParams,
            BestValue = study.BestTrial?.Value ?? 0,
            WallClockSeconds = sw.Elapsed.TotalSeconds,
            TrialsPerSecond = nTrials / sw.Elapsed.TotalSeconds
        };
    }

    private UnifiedBenchmarkResult RunPruning(int nTrials, int nParams, string scaleName)
    {
        var space = new SearchSpace(
            Enumerable.Range(0, nParams)
                .Select(i => new FloatRange($"x_{i}", -5.0, 5.0))
                .ToList()
        );

        using var study = Optimizer.CreateStudy(
            $"unified_pruning_{scaleName.ToLower()}",
            space,
            StudyDirection.Minimize,
            new TpeSamplerConfig { Seed = 42 },
            pruner: new MedianPruner()
        );

        var sw = Stopwatch.StartNew();

        for (int trial = 0; trial < nTrials; trial++)
        {
            var trialObj = study.Ask();

            // Multi-step evaluation with pruning checkpoints
            bool wasPruned = false;
            for (int step = 1; step <= 5; step++)
            {
                var value = PruningEval(trialObj.Parameters, step, nParams);
                trialObj.Report(value, step);

                if (study.ShouldPrune(trialObj))
                {
                    study.Tell(trialObj.Number, TrialState.Pruned);
                    wasPruned = true;
                    break;
                }
            }

            if (!wasPruned && trialObj.State == TrialState.Running)
            {
                var finalValue = SphereFunctionEval(trialObj.Parameters);
                study.Tell(trialObj.Number, finalValue);
            }

            if ((trial + 1) % Math.Max(1, nTrials / 4) == 0)
            {
                Console.WriteLine($"  {trial + 1}/{nTrials} trials: {sw.Elapsed.TotalSeconds:F1}s, Best={study.BestTrial?.Value:F6}");
            }
        }

        sw.Stop();

        return new UnifiedBenchmarkResult
        {
            Feature = "Pruning",
            Sampler = "TPE",
            Scale = scaleName,
            NTrials = nTrials,
            NParameters = nParams,
            BestValue = study.BestTrial?.Value ?? 0,
            WallClockSeconds = sw.Elapsed.TotalSeconds,
            TrialsPerSecond = nTrials / sw.Elapsed.TotalSeconds
        };
    }

    private void SaveResults(string filename, object results)
    {
        var json = JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true });
        var path = Path.Combine(_outputDir, filename);
        File.WriteAllText(path, json);
        Console.WriteLine($"\nResults saved to {path}");
    }
}
