using System.Diagnostics;
using System.Text.Json;
using OptiSharp;
using OptiSharp.Models;
using OptiSharp.Pruning;
using OptiSharp.Samplers.Tpe;

namespace OptiSharp.Benchmarks;

/// <summary>
/// Comprehensive load test runner comparing all features at 5 scales.
/// Scales: Tiny (100 trials, 10 params) -> XXL (100k trials, 200 params)
/// Features: SingleObjective, MultiObjective, Constraints, Pruning
/// </summary>
public class LoadTestRunner
{
    private readonly string _outputDir = "./benchmark-results";

    private static readonly Dictionary<int, (string Name, int Trials, int Params)> TestScales = new()
    {
        { 1, ("Tiny", 100, 10) },
        { 2, ("Small", 1000, 50) },
        { 3, ("Medium", 5000, 100) },
        { 4, ("Large", 25000, 150) },
        { 5, ("XXL", 100000, 200) }
    };

    public LoadTestRunner()
    {
        Directory.CreateDirectory(_outputDir);
    }

    // ============================================================================
    // SINGLE-OBJECTIVE TESTS
    // ============================================================================

    public class LoadTestResult
    {
        public string Feature { get; set; }
        public string Scale { get; set; }
        public int NTrials { get; set; }
        public int NParameters { get; set; }
        public double BestValue { get; set; }
        public double WallClockSeconds { get; set; }
        public double TrialsPerSecond { get; set; }
    }

    public void RunAllTests(int maxScale = 3, string samplerFilter = "all")
    {
        var allResults = new List<LoadTestResult>();

        for (int scale = 1; scale <= maxScale; scale++)
        {
            var (scaleName, nTrials, nParams) = TestScales[scale];
            Console.WriteLine($"\n{'#',-70}");
            Console.WriteLine($"# SCALE: {scaleName.ToUpper()} ({nTrials} trials, {nParams} params)");
            Console.WriteLine($"{'#',-70}");

            try
            {
                Console.WriteLine("\n[1/4] Single-Objective...");
                allResults.AddRange(TestSingleObjectiveAllSamplers(scale, samplerFilter));
            }
            catch (Exception e)
            {
                Console.WriteLine($"ERROR: {e.Message}");
            }

            try
            {
                Console.WriteLine("\n[2/4] Multi-Objective...");
                allResults.Add(TestMultiObjective(scale));
            }
            catch (Exception e)
            {
                Console.WriteLine($"ERROR: {e.Message}");
            }

            try
            {
                Console.WriteLine("\n[3/4] Constraints...");
                allResults.Add(TestConstraints(scale));
            }
            catch (Exception e)
            {
                Console.WriteLine($"ERROR: {e.Message}");
            }

            try
            {
                Console.WriteLine("\n[4/4] Pruning...");
                allResults.Add(TestPruning(scale));
            }
            catch (Exception e)
            {
                Console.WriteLine($"ERROR: {e.Message}");
            }

            // Break after first scale for sampler tests to save time
            if (scale >= 2)
                break;
        }

        // Save results
        SaveResults("load_tests_comprehensive.json", allResults);

        // Print summary
        Console.WriteLine($"\n\n{new string('=', 70)}");
        Console.WriteLine("SUMMARY:");
        Console.WriteLine(new string('=', 70));

        foreach (var result in allResults)
        {
            Console.WriteLine($"\n{result.Feature} - {result.Scale}:");
            Console.WriteLine($"  Trials: {result.NTrials}, Params: {result.NParameters}");
            Console.WriteLine($"  Time: {result.WallClockSeconds:F2}s, Throughput: {result.TrialsPerSecond:F1} trials/sec");
            if (result.BestValue > 0)
                Console.WriteLine($"  Best Value: {result.BestValue:F6}");
        }
    }

    // ============================================================================
    // SAMPLER-VARIANT TESTS (TPE, Random, CMA-ES)
    // ============================================================================

    private List<LoadTestResult> TestSingleObjectiveAllSamplers(int scaleLevel, string samplerFilter = "all")
    {
        var results = new List<LoadTestResult>();

        if (samplerFilter == "all" || samplerFilter == "tpe")
            results.Add(TestSingleObjective(scaleLevel, "TPE"));
        if (samplerFilter == "all" || samplerFilter == "random")
            results.Add(TestSingleObjective(scaleLevel, "Random"));
        if (samplerFilter == "all" || samplerFilter == "cma-es")
            results.Add(TestSingleObjective(scaleLevel, "CMA-ES"));

        return results;
    }

    private LoadTestResult TestSingleObjective(int scaleLevel, string samplerType = "TPE")
    {
        var (scaleName, nTrials, nParams) = TestScales[scaleLevel];

        Console.WriteLine($"\n{'='*70}");
        Console.WriteLine($"SINGLE-OBJECTIVE ({samplerType}): {scaleName} ({nTrials} trials, {nParams} params)");
        Console.WriteLine($"{'='*70}");

        // Create search space with nParams continuous parameters
        var space = new SearchSpace(
            Enumerable.Range(0, nParams)
                .Select(i => new FloatRange($"x_{i}", -5.0, 5.0))
                .ToList()
        );

        Study study;
        if (samplerType == "TPE")
        {
            study = Optimizer.CreateStudy(
                $"single_obj_{scaleName}_tpe",
                space,
                StudyDirection.Minimize,
                new TpeSamplerConfig { Seed = 42 }
            );
        }
        else if (samplerType == "Random")
        {
            study = Optimizer.CreateStudyWithRandomSampler(
                $"single_obj_{scaleName}_random",
                space,
                StudyDirection.Minimize,
                seed: 42
            );
        }
        else if (samplerType == "CMA-ES")
        {
            study = Optimizer.CreateStudyWithCmaEs(
                $"single_obj_{scaleName}_cmaes",
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
                var value = EvaluateSphere(trialObj.Parameters);
                study.Tell(trialObj.Number, value);

                if ((trial + 1) % Math.Max(1, nTrials / 4) == 0)
                {
                    Console.WriteLine($"  {trial + 1}/{nTrials} trials: {sw.Elapsed.TotalSeconds:F1}s, Best={study.BestTrial?.Value:F6}");
                }
            }

            sw.Stop();

            return new LoadTestResult
            {
                Feature = $"SingleObjective-{samplerType}",
                Scale = scaleName,
                NTrials = nTrials,
                NParameters = nParams,
                BestValue = study.BestTrial?.Value ?? 0,
                WallClockSeconds = sw.Elapsed.TotalSeconds,
                TrialsPerSecond = nTrials / sw.Elapsed.TotalSeconds
            };
        }
    }

    private LoadTestResult TestMultiObjective(int scaleLevel)
    {
        var (scaleName, nTrials, nParams) = TestScales[scaleLevel];

        Console.WriteLine($"\n{'='*70}");
        Console.WriteLine($"MULTI-OBJECTIVE: {scaleName} ({nTrials} trials, {nParams} params)");
        Console.WriteLine($"{'='*70}");

        var space = new SearchSpace(
            Enumerable.Range(0, nParams)
                .Select(i => new FloatRange($"x_{i}", -5.0, 5.0))
                .ToList()
        );

        var directions = new[] { StudyDirection.Minimize, StudyDirection.Minimize };

        using var study = Optimizer.CreateStudy(
            $"multi_obj_{scaleName}",
            space,
            directions,
            new TpeSamplerConfig { Seed = 42 }
        );

        var sw = Stopwatch.StartNew();

        for (int trial = 0; trial < nTrials; trial++)
        {
            var trialObj = study.Ask();
            var params_list = Enumerable.Range(0, nParams)
                .Select(i => (double)trialObj.Parameters[$"x_{i}"])
                .ToList();

            // Objective 1: Sphere
            var obj1 = params_list.Sum(x => x * x);

            // Objective 2: Rosenbrock-like (first 10 params only for speed)
            var obj2 = 0.0;
            for (int i = 0; i < Math.Min(nParams - 1, 10); i++)
            {
                var term = 100 * Math.Pow(params_list[i + 1] - params_list[i] * params_list[i], 2) +
                           Math.Pow(1 - params_list[i], 2);
                obj2 += term;
            }

            study.Tell(trialObj.Number, new[] { obj1, obj2 });

            if ((trial + 1) % Math.Max(1, nTrials / 4) == 0)
            {
                var paretoSize = study.ParetoFront.Count;
                Console.WriteLine($"  {trial + 1}/{nTrials} trials: {sw.Elapsed.TotalSeconds:F1}s, Pareto={paretoSize}");
            }
        }

        sw.Stop();

        return new LoadTestResult
        {
            Feature = "MultiObjective",
            Scale = scaleName,
            NTrials = nTrials,
            NParameters = nParams,
            BestValue = study.ParetoFront.Count,  // Store Pareto size as "best value"
            WallClockSeconds = sw.Elapsed.TotalSeconds,
            TrialsPerSecond = nTrials / sw.Elapsed.TotalSeconds
        };
    }

    private LoadTestResult TestConstraints(int scaleLevel)
    {
        var (scaleName, nTrials, nParams) = TestScales[scaleLevel];

        Console.WriteLine($"\n{'='*70}");
        Console.WriteLine($"CONSTRAINTS: {scaleName} ({nTrials} trials, {nParams} params)");
        Console.WriteLine($"{'='*70}");

        var space = new SearchSpace(
            Enumerable.Range(0, nParams)
                .Select(i => new FloatRange($"x_{i}", -5.0, 5.0))
                .ToList()
        );

        using var study = Optimizer.CreateStudy(
            $"constraints_{scaleName}",
            space,
            StudyDirection.Minimize,
            new TpeSamplerConfig { Seed = 42 }
        );

        // Set constraint: sum of absolute values < threshold
        double threshold = nParams * 2.0;
        study.SetConstraintFunc(trial =>
        {
            var sum = Enumerable.Range(0, nParams)
                .Sum(i => Math.Abs((double)trial.Parameters[$"x_{i}"]));
            return new[] { sum - threshold };
        });

        var sw = Stopwatch.StartNew();
        int feasibleCount = 0;

        for (int trial = 0; trial < nTrials; trial++)
        {
            var trialObj = study.Ask();
            var value = EvaluateSphere(trialObj.Parameters);
            study.Tell(trialObj.Number, value);

            if (study.IsFeasible(trialObj))
                feasibleCount++;

            if ((trial + 1) % Math.Max(1, nTrials / 4) == 0)
            {
                double feasRate = (feasibleCount / (double)(trial + 1)) * 100;
                Console.WriteLine($"  {trial + 1}/{nTrials} trials: {sw.Elapsed.TotalSeconds:F1}s, Feasible={feasRate:F1}%");
            }
        }

        sw.Stop();

        return new LoadTestResult
        {
            Feature = "Constraints",
            Scale = scaleName,
            NTrials = nTrials,
            NParameters = nParams,
            BestValue = feasibleCount / (double)nTrials,  // Store feasibility rate
            WallClockSeconds = sw.Elapsed.TotalSeconds,
            TrialsPerSecond = nTrials / sw.Elapsed.TotalSeconds
        };
    }

    private LoadTestResult TestPruning(int scaleLevel)
    {
        var (scaleName, nTrials, nParams) = TestScales[scaleLevel];

        Console.WriteLine($"\n{'='*70}");
        Console.WriteLine($"PRUNING: {scaleName} ({nTrials} trials, {nParams} params)");
        Console.WriteLine($"{'='*70}");

        var space = new SearchSpace(
            Enumerable.Range(0, nParams)
                .Select(i => new FloatRange($"x_{i}", -5.0, 5.0))
                .ToList()
        );

        using var study = Optimizer.CreateStudy(
            $"pruning_{scaleName}",
            space,
            StudyDirection.Minimize,
            new TpeSamplerConfig { Seed = 42 },
            pruner: new MedianPruner()
        );

        var sw = Stopwatch.StartNew();
        int prunedCount = 0;

        for (int trial = 0; trial < nTrials; trial++)
        {
            var trialObj = study.Ask();

            // Multi-step evaluation with pruning checkpoints
            bool wasPruned = false;
            for (int step = 1; step <= 5; step++)
            {
                var value = EvaluateSphereWithStep(trialObj.Parameters, step);
                trialObj.Report(value, step);

                if (study.ShouldPrune(trialObj))
                {
                    study.Tell(trialObj.Number, TrialState.Pruned);
                    prunedCount++;
                    wasPruned = true;
                    break;
                }
            }

            if (!wasPruned && trialObj.State == TrialState.Running)
            {
                var finalValue = EvaluateSphere(trialObj.Parameters);
                study.Tell(trialObj.Number, finalValue);
            }

            if ((trial + 1) % Math.Max(1, nTrials / 4) == 0)
            {
                double pruneRate = (prunedCount / (double)(trial + 1)) * 100;
                Console.WriteLine($"  {trial + 1}/{nTrials} trials: {sw.Elapsed.TotalSeconds:F1}s, Pruned={pruneRate:F1}%");
            }
        }

        sw.Stop();

        return new LoadTestResult
        {
            Feature = "Pruning",
            Scale = scaleName,
            NTrials = nTrials,
            NParameters = nParams,
            BestValue = prunedCount / (double)nTrials,  // Store pruning rate
            WallClockSeconds = sw.Elapsed.TotalSeconds,
            TrialsPerSecond = nTrials / sw.Elapsed.TotalSeconds
        };
    }

    // ============================================================================
    // EVALUATION FUNCTIONS
    // ============================================================================

    private double EvaluateSphere(IReadOnlyDictionary<string, object> parameters)
    {
        double sum = 0;
        foreach (var param in parameters.Values)
        {
            if (param is double d)
                sum += d * d;
        }
        return sum;
    }

    private double EvaluateSphereWithStep(IReadOnlyDictionary<string, object> parameters, int step)
    {
        var sphere = EvaluateSphere(parameters);
        return sphere / (1 + step * 0.1);  // Improve with steps for pruning signal
    }

    private void SaveResults(string filename, object results)
    {
        var json = JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true });
        var path = Path.Combine(_outputDir, filename);
        File.WriteAllText(path, json);
        Console.WriteLine($"\nResults saved to {path}");
    }
}
