using System.Diagnostics;
using System.Text.Json;
using OptiSharp;
using OptiSharp.Models;
using OptiSharp.Pruning;
using OptiSharp.Samplers.Tpe;

namespace OptiSharp.Benchmarks;

/// <summary>
/// Comprehensive benchmark harness for OptiSharp vs Optuna comparisons.
/// Measures wall-clock time, sample efficiency, convergence speed, and feature completeness.
/// </summary>
public class BenchmarkRunner
{
    private readonly string _outputDir = "./benchmark-results";

    public BenchmarkRunner()
    {
        Directory.CreateDirectory(_outputDir);
    }

    /// <summary>
    /// Benchmark 1: CASH (Combined Algorithm Selection + Hyperparameter optimization)
    /// Problem: Maximize 5-fold CV accuracy by selecting classifier and tuning hyperparameters.
    /// Budget: 100 trials, 60-second wall-clock limit
    /// </summary>
    public void BenchmarkCASH()
    {
        Console.WriteLine("\n=== BENCHMARK 1: CASH (Algorithm Selection + HPO) ===\n");

        var searchSpace = CreateCASHSearchSpace();
        var results = new List<BenchmarkResult>();

        for (int run = 0; run < 3; run++)
        {
            Console.WriteLine($"Run {run + 1}/3...");

            using var study = Optimizer.CreateStudy(
                $"cash-run-{run}",
                searchSpace,
                StudyDirection.Maximize,
                new TpeSamplerConfig { Seed = 42 + run }
            );

            var sw = Stopwatch.StartNew();
            var convergenceHistory = new Dictionary<int, double>();

            for (int trial = 0; trial < 100; trial++)
            {
                var trialObj = study.Ask();
                var accuracy = EvaluateCASH(trialObj.Parameters);

                study.Tell(trialObj.Number, accuracy);
                convergenceHistory[trial] = study.BestTrial?.Value ?? 0;

                if ((trial + 1) % 25 == 0)
                    Console.WriteLine($"  Trial {trial + 1}: Best accuracy = {study.BestTrial?.Value:F4}");
            }

            sw.Stop();

            results.Add(new BenchmarkResult
            {
                Run = run,
                Problem = "CASH",
                Algorithm = "OptiSharp-TPE",
                TrialCount = 100,
                WallClockSeconds = sw.Elapsed.TotalSeconds,
                BestValue = study.BestTrial?.Value ?? 0,
                TrialsToTarget95 = FindTrialsToTarget(convergenceHistory, 0.95),
                ConvergenceHistory = convergenceHistory,
                Seed = 42 + run
            });
        }

        SaveResults("cash_results.json", results);
        PrintCASHSummary(results);
    }

    /// <summary>
    /// Benchmark 2: Neural Network Hyperparameter Tuning
    /// Problem: Optimize MLP for binary classification
    /// Budget: 50 trials, 90-second wall-clock limit
    /// </summary>
    public void BenchmarkNeuralNetworkTuning()
    {
        Console.WriteLine("\n=== BENCHMARK 2: Neural Network Hyperparameter Tuning ===\n");

        var searchSpace = CreateNNSearchSpace();
        var results = new List<BenchmarkResult>();

        for (int run = 0; run < 3; run++)
        {
            Console.WriteLine($"Run {run + 1}/3...");

            using var study = Optimizer.CreateStudy(
                $"nn-tuning-run-{run}",
                searchSpace,
                StudyDirection.Maximize,
                new TpeSamplerConfig { Seed = 100 + run }
            );

            var sw = Stopwatch.StartNew();
            var convergenceHistory = new Dictionary<int, double>();

            for (int trial = 0; trial < 50; trial++)
            {
                var trialObj = study.Ask();
                var accuracy = EvaluateNNTuning(trialObj.Parameters);

                study.Tell(trialObj.Number, accuracy);
                convergenceHistory[trial] = study.BestTrial?.Value ?? 0;

                if ((trial + 1) % 10 == 0)
                    Console.WriteLine($"  Trial {trial + 1}: Best accuracy = {study.BestTrial?.Value:F4}");
            }

            sw.Stop();

            results.Add(new BenchmarkResult
            {
                Run = run,
                Problem = "NNTuning",
                Algorithm = "OptiSharp-TPE",
                TrialCount = 50,
                WallClockSeconds = sw.Elapsed.TotalSeconds,
                BestValue = study.BestTrial?.Value ?? 0,
                ConvergenceHistory = convergenceHistory,
                Seed = 100 + run
            });
        }

        SaveResults("nn_tuning_results.json", results);
        PrintNNTuningSummary(results);
    }

    /// <summary>
    /// Benchmark 3: Multi-Objective with Constraints
    /// Problem: Maximize accuracy while minimizing latency, with memory/storage constraints
    /// Budget: 75 trials
    /// </summary>
    public void BenchmarkMultiObjectiveConstraints()
    {
        Console.WriteLine("\n=== BENCHMARK 3: Multi-Objective with Constraints ===\n");

        var searchSpace = CreateMOSearchSpace();
        var results = new List<MOBenchmarkResult>();

        for (int run = 0; run < 2; run++)
        {
            Console.WriteLine($"Run {run + 1}/2...");

            var directions = new[] { StudyDirection.Maximize, StudyDirection.Minimize };
            using var study = Optimizer.CreateStudy(
                $"mo-constraints-run-{run}",
                searchSpace,
                directions,
                new TpeSamplerConfig { Seed = 200 + run },
                new NopPruner()
            );

            // Set constraint function: Memory <= 500MB, Size <= 10MB
            study.SetConstraintFunc(trial =>
            {
                var batchSize = (int)trial.Parameters["batch_size"];
                var modelSize = EstimateModelSize(batchSize);
                var memory = EstimateMemory(batchSize);

                return new[]
                {
                    memory - 500,     // Memory constraint
                    modelSize - 10.0  // Size constraint
                };
            });

            var sw = Stopwatch.StartNew();

            for (int trial = 0; trial < 75; trial++)
            {
                var trialObj = study.Ask();
                var accuracy = EvaluateMOAccuracy(trialObj.Parameters);
                var latency = EvaluateMOLatency(trialObj.Parameters);

                study.Tell(trialObj.Number, new[] { accuracy, latency });

                if ((trial + 1) % 15 == 0)
                {
                    var front = study.ParetoFront;
                    var feasible = study.Trials.Count(t => study.IsFeasible(t));
                    Console.WriteLine($"  Trial {trial + 1}: Pareto size = {front.Count}, " +
                                    $"Feasible = {feasible}");
                }
            }

            sw.Stop();

            var paretoFront = study.ParetoFront;
            var feasibleCount = study.Trials.Count(t => study.IsFeasible(t));
            var totalTrials = study.Trials.Count;

            results.Add(new MOBenchmarkResult
            {
                Run = run,
                Problem = "MOConstraints",
                TrialCount = 75,
                WallClockSeconds = sw.Elapsed.TotalSeconds,
                ParetoFrontSize = paretoFront.Count,
                FeasibleSolutions = feasibleCount,
                TotalTrials = totalTrials,
                ConstraintViolationRate = 1.0 - ((double)feasibleCount / totalTrials),
                Hypervolume = ComputeHypervolume(paretoFront),
                Seed = 200 + run
            });
        }

        SaveResults("mo_constraints_results.json", results);
        PrintMOConstraintsSummary(results);
    }

    /// <summary>
    /// Benchmark 4: Pruning Effectiveness
    /// Measure how much trials are pruned and impact on convergence
    /// </summary>
    public void BenchmarkPruning()
    {
        Console.WriteLine("\n=== BENCHMARK 4: Pruning Effectiveness ===\n");

        var searchSpace = CreateSimpleSearchSpace();
        var results = new List<PruningBenchmarkResult>();

        var prunerConfigs = new[]
        {
            ("NopPruner", new NopPruner() as IPruner),
            ("MedianPruner", new MedianPruner(new MedianPrunerConfig { NStartupTrials = 5 }) as IPruner),
            ("PercentilePruner", new PercentilePruner(new PercentilePrunerConfig { Percentile = 25.0 }) as IPruner)
        };

        foreach (var (name, pruner) in prunerConfigs)
        {
            Console.WriteLine($"\nTesting {name}...");

            using var study = Optimizer.CreateStudy(
                $"pruning-{name}",
                searchSpace,
                StudyDirection.Minimize,
                new TpeSamplerConfig { Seed = 42 },
                pruner
            );

            var sw = Stopwatch.StartNew();
            var pruned = 0;

            for (int trial = 0; trial < 100; trial++)
            {
                var trialObj = study.Ask();

                // Simulate multi-step evaluation with intermediate reporting
                for (int step = 1; step <= 10; step++)
                {
                    var intermediate = EvaluateWithIntermediates(trialObj.Parameters, step);
                    trialObj.Report(intermediate, step);

                    // Check if should prune
                    if (study.ShouldPrune(trialObj))
                    {
                        study.Tell(trialObj.Number, TrialState.Pruned);
                        pruned++;
                        break;
                    }
                }

                if (trialObj.State == TrialState.Running)
                {
                    var finalValue = EvaluatePruningProblem(trialObj.Parameters);
                    study.Tell(trialObj.Number, finalValue);
                }
            }

            sw.Stop();

            results.Add(new PruningBenchmarkResult
            {
                PrunerName = name,
                TotalTrials = 100,
                PrunedCount = pruned,
                PruningRate = (double)pruned / 100,
                WallClockSeconds = sw.Elapsed.TotalSeconds,
                BestValue = study.BestTrial?.Value ?? 0
            });
        }

        SaveResults("pruning_results.json", results);
        PrintPruningSummary(results);
    }

    // ============ Helper Methods ============

    private SearchSpace CreateCASHSearchSpace() =>
        new SearchSpace([
            new CategoricalRange("classifier", new object[] { "rf", "svm", "gb" }),
            new IntRange("max_depth", 3, 30),
            new IntRange("n_estimators", 10, 500),
            new FloatRange("learning_rate", 0.001, 1.0, Log: true),
            new FloatRange("regularization", 0.0001, 10.0, Log: true)
        ]);

    private SearchSpace CreateNNSearchSpace() =>
        new SearchSpace([
            new FloatRange("learning_rate", 0.0001, 0.1, Log: true),
            new IntRange("batch_size", 16, 256),
            new FloatRange("dropout", 0.0, 0.5),
            new IntRange("hidden_units", 32, 512),
            new CategoricalRange("optimizer", new object[] { "adam", "sgd", "rmsprop" })
        ]);

    private SearchSpace CreateMOSearchSpace() =>
        new SearchSpace([
            new IntRange("batch_size", 16, 128),
            new FloatRange("learning_rate", 0.0001, 0.1, Log: true),
            new IntRange("model_size", 10, 100),
            new FloatRange("regularization", 0.0, 0.5)
        ]);

    private SearchSpace CreateSimpleSearchSpace() =>
        new SearchSpace([
            new FloatRange("x", -5.0, 5.0),
            new FloatRange("y", -5.0, 5.0)
        ]);

    private double EvaluateCASH(IReadOnlyDictionary<string, object> parameters)
    {
        // Simulate CV accuracy: higher is better
        var classifier = (string)parameters["classifier"];
        var depth = (int)parameters["max_depth"];
        var lr = (double)parameters["learning_rate"];

        // Synthetic accuracy function
        var baseAccuracy = classifier switch
        {
            "rf" => 0.92,
            "svm" => 0.90,
            "gb" => 0.91,
            _ => 0.85
        };

        // Adjust for hyperparameters
        var depthBoost = Math.Min(0.05, (depth - 3) / 27.0 * 0.05);
        var lrPenalty = Math.Abs(Math.Log10(lr) + 2.5) * 0.02; // Prefer lr around 0.003

        return Math.Min(1.0, baseAccuracy + depthBoost - lrPenalty + Random.Shared.NextDouble() * 0.03);
    }

    private double EvaluateNNTuning(IReadOnlyDictionary<string, object> parameters)
    {
        // Simulate NN accuracy
        var lr = (double)parameters["learning_rate"];
        var batchSize = (int)parameters["batch_size"];
        var dropout = (double)parameters["dropout"];

        var baseAccuracy = 0.75;
        var lrOptimal = 0.001; // Optimal learning rate
        var lrPenalty = Math.Abs(Math.Log10(lr) - Math.Log10(lrOptimal)) * 0.15;
        var batchBoost = 0.02 * Math.Log(batchSize / 16.0);
        var dropoutBoost = dropout * 0.05; // Some dropout helps

        return Math.Min(1.0, baseAccuracy + batchBoost + dropoutBoost - lrPenalty + Random.Shared.NextDouble() * 0.02);
    }

    private double EvaluateMOAccuracy(IReadOnlyDictionary<string, object> parameters)
    {
        var batchSize = (int)parameters["batch_size"];
        var lr = (double)parameters["learning_rate"];

        var baseAcc = 0.80;
        var lrBoost = Math.Max(0, 0.1 - Math.Abs(Math.Log10(lr) + 2.5) * 0.3);
        var batchBoost = Math.Log(batchSize / 16.0) * 0.02;

        return Math.Min(1.0, baseAcc + lrBoost + batchBoost + Random.Shared.NextDouble() * 0.05);
    }

    private double EvaluateMOLatency(IReadOnlyDictionary<string, object> parameters)
    {
        var batchSize = (int)parameters["batch_size"];
        var modelSize = (int)parameters["model_size"];

        // Latency in ms: lower is better
        return 10 + Math.Log(batchSize) * 5 + Math.Sqrt(modelSize) * 0.5 + Random.Shared.NextDouble() * 5;
    }

    private double EvaluateWithIntermediates(IReadOnlyDictionary<string, object> parameters, int step)
    {
        var x = (double)parameters["x"];
        var y = (double)parameters["y"];

        // Rastrigin function with early stopping signal
        var value = 20 + x * x + y * y - 10 * (Math.Cos(2 * Math.PI * x) + Math.Cos(2 * Math.PI * y));
        return value / (1 + step * 0.1); // Improve with steps
    }

    private double EvaluatePruningProblem(IReadOnlyDictionary<string, object> parameters)
    {
        // Sphere function
        var x = (double)parameters["x"];
        var y = (double)parameters["y"];
        return x * x + y * y;
    }

    private double EstimateModelSize(int batchSize)
    {
        return 5 + batchSize / 50.0; // MB
    }

    private double EstimateMemory(int batchSize)
    {
        return 200 + batchSize * 2; // MB
    }

    private int FindTrialsToTarget(Dictionary<int, double> history, double target)
    {
        foreach (var (trial, best) in history.OrderBy(x => x.Key))
        {
            if (best >= target) return trial + 1;
        }
        return history.Count;
    }

    private double ComputeHypervolume(IReadOnlyList<Trial> paretoFront)
    {
        // Simplified: sum of normalized objectives
        if (paretoFront.Count == 0) return 0;

        double volume = 0;
        foreach (var solution in paretoFront)
        {
            if (solution.Values == null) continue;
            var accuracy = solution.Values[0]; // 0-1
            var latency = 100 - Math.Min(100, solution.Values[1]); // Invert & cap
            volume += (accuracy / 100) * (latency / 100);
        }

        return volume / paretoFront.Count;
    }

    // ============ Result Types ============

    private void SaveResults(string filename, object results)
    {
        var json = JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true });
        var path = Path.Combine(_outputDir, filename);
        File.WriteAllText(path, json);
        Console.WriteLine($"\n✓ Results saved to {path}");
    }

    private void PrintCASHSummary(List<BenchmarkResult> results)
    {
        var avgTime = results.Average(r => r.WallClockSeconds);
        var avgAccuracy = results.Average(r => r.BestValue);
        var avgTrialsTo95 = results.Average(r => r.TrialsToTarget95);

        Console.WriteLine($"\n--- CASH Summary ---");
        Console.WriteLine($"Average wall-clock time: {avgTime:F2}s");
        Console.WriteLine($"Average best accuracy: {avgAccuracy:F4}");
        Console.WriteLine($"Average trials to 95%: {avgTrialsTo95:F1}");
    }

    private void PrintNNTuningSummary(List<BenchmarkResult> results)
    {
        var avgTime = results.Average(r => r.WallClockSeconds);
        var avgAccuracy = results.Average(r => r.BestValue);

        Console.WriteLine($"\n--- NN Tuning Summary ---");
        Console.WriteLine($"Average wall-clock time: {avgTime:F2}s");
        Console.WriteLine($"Average best accuracy: {avgAccuracy:F4}");
    }

    private void PrintMOConstraintsSummary(List<MOBenchmarkResult> results)
    {
        var avgParetoSize = results.Average(r => r.ParetoFrontSize);
        var avgFeasible = results.Average(r => (double)r.FeasibleSolutions / r.TotalTrials) * 100;
        var avgHypervolume = results.Average(r => r.Hypervolume);

        Console.WriteLine($"\n--- MO Constraints Summary ---");
        Console.WriteLine($"Average Pareto front size: {avgParetoSize:F1}");
        Console.WriteLine($"Average feasible solutions: {avgFeasible:F1}%");
        Console.WriteLine($"Average hypervolume: {avgHypervolume:F4}");
    }

    private void PrintPruningSummary(List<PruningBenchmarkResult> results)
    {
        Console.WriteLine($"\n--- Pruning Summary ---");
        foreach (var result in results)
        {
            Console.WriteLine($"{result.PrunerName}:");
            Console.WriteLine($"  Pruned: {result.PrunedCount}/100 ({result.PruningRate:P1})");
            Console.WriteLine($"  Wall-clock: {result.WallClockSeconds:F2}s");
            Console.WriteLine($"  Best value: {result.BestValue:F4}");
        }
    }

    // ============ LOAD TESTS ============

    public class LoadTestResult
    {
        public string Benchmark { get; set; }
        public string Library { get; set; }
        public int NTrials { get; set; }
        public int NParameters { get; set; }
        public double WallClockSeconds { get; set; }
        public double BestValue { get; set; }
        public double TrialsPerSecond { get; set; }
        public long MemoryMBStart { get; set; }
        public long MemoryMBPeak { get; set; }
    }

    public class ScaleProgressionResult
    {
        public string Benchmark { get; set; }
        public string Library { get; set; }
        public Dictionary<int, ScalePoint> Results { get; set; }
    }

    public class ScalePoint
    {
        public int NTrials { get; set; }
        public double WallClockSeconds { get; set; }
        public double BestValue { get; set; }
        public double TrialsPerSecond { get; set; }
    }

    /// <summary>
    /// Load Test 1: High-Dimensional Optimization (100 parameters, 500 trials)
    /// </summary>
    public void BenchmarkHighDimensional()
    {
        Console.WriteLine("\n=== LOAD TEST 1: High-Dimensional Optimization (100 parameters) ===\n");

        var space = new SearchSpace(
            Enumerable.Range(0, 100)
                .Select(i => new FloatRange($"x_{i}", -5.0, 5.0))
                .ToList()
        );

        var result = new LoadTestResult
        {
            Benchmark = "HighDimensional",
            Library = "OptiSharp",
            NParameters = 100
        };

        // Warm up
        using var warmupStudy = Optimizer.CreateStudy("warmup", space);
        for (int i = 0; i < 10; i++)
        {
            var trial = warmupStudy.Ask();
            var value = EvaluateHighDimensional(trial.Parameters);
            warmupStudy.Tell(trial.Number, value);
        }

        // Actual test - 500 trials
        using var study = Optimizer.CreateStudy("high_dimensional", space);
        var initialMemory = GC.GetTotalMemory(false) / (1024 * 1024);
        var sw = Stopwatch.StartNew();

        for (int trial = 0; trial < 500; trial++)
        {
            var trialObj = study.Ask();
            var value = EvaluateHighDimensional(trialObj.Parameters);
            study.Tell(trialObj.Number, value);

            if ((trial + 1) % 125 == 0)
                Console.WriteLine($"  Trial {trial + 1}/500: Best = {study.BestTrial?.Value:F6}");
        }

        sw.Stop();
        var peakMemory = GC.GetTotalMemory(false) / (1024 * 1024);

        result.NTrials = 500;
        result.WallClockSeconds = sw.Elapsed.TotalSeconds;
        result.BestValue = study.BestTrial?.Value ?? 0;
        result.TrialsPerSecond = 1000 / result.WallClockSeconds;
        result.MemoryMBStart = initialMemory;
        result.MemoryMBPeak = peakMemory;

        SaveResults("load_test_high_dimensional.json", new[] { result });
        Console.WriteLine($"\n--- Load Test 1 Summary ---");
        Console.WriteLine($"Completed 500 trials (100 parameters) in {result.WallClockSeconds:F2}s");
        Console.WriteLine($"Throughput: {result.TrialsPerSecond:F1} trials/sec");
        Console.WriteLine($"Memory: {initialMemory}MB initial -> {peakMemory}MB peak");
    }

    /// <summary>
    /// Load Test 2: Scale Progression (1k, 5k, 10k trials)
    /// </summary>
    public void BenchmarkScaleProgression()
    {
        Console.WriteLine("\n=== LOAD TEST 2: Scale Progression (1k, 5k, 10k trials) ===\n");

        var space = new SearchSpace(new[]
        {
            new FloatRange("x", -10.0, 10.0),
            new FloatRange("y", -10.0, 10.0),
            new FloatRange("z", -10.0, 10.0),
        });

        var progressionResults = new Dictionary<int, ScalePoint>();
        var trialCounts = new[] { 1000, 5000, 10000 };

        foreach (var nTrials in trialCounts)
        {
            Console.WriteLine($"Testing {nTrials} trials...");

            using var study = Optimizer.CreateStudy($"scale_{nTrials}", space);
            var sw = Stopwatch.StartNew();

            for (int trial = 0; trial < nTrials; trial++)
            {
                var trialObj = study.Ask();
                var value = EvaluateScaleProblem(trialObj.Parameters);
                study.Tell(trialObj.Number, value);

                if ((trial + 1) % (nTrials / 4) == 0)
                    Console.WriteLine($"  Trial {trial + 1}/{nTrials}: Best = {study.BestTrial?.Value:F6}");
            }

            sw.Stop();

            progressionResults[nTrials] = new ScalePoint
            {
                NTrials = nTrials,
                WallClockSeconds = sw.Elapsed.TotalSeconds,
                BestValue = study.BestTrial?.Value ?? 0,
                TrialsPerSecond = nTrials / sw.Elapsed.TotalSeconds
            };

            Console.WriteLine($"  Completed in {sw.Elapsed.TotalSeconds:F2}s ({nTrials / sw.Elapsed.TotalSeconds:F1} trials/sec)\n");
        }

        var result = new ScaleProgressionResult
        {
            Benchmark = "ScaleProgression",
            Library = "OptiSharp",
            Results = progressionResults
        };

        SaveResults("load_test_scale_progression.json", new[] { result });

        Console.WriteLine($"\n--- Load Test 2 Summary ---");
        foreach (var (trials, point) in progressionResults)
        {
            Console.WriteLine($"{trials} trials: {point.WallClockSeconds:F2}s ({point.TrialsPerSecond:F1} trials/sec)");
        }
    }

    private double EvaluateHighDimensional(IReadOnlyDictionary<string, object> parameters)
    {
        // Sphere function: sum of squares
        double sum = 0;
        for (int i = 0; i < 100; i++)
        {
            var x = (double)parameters[$"x_{i}"];
            sum += x * x;
        }
        return sum;
    }

    private double EvaluateScaleProblem(IReadOnlyDictionary<string, object> parameters)
    {
        // Ackley function approximation
        var x = (double)parameters["x"];
        var y = (double)parameters["y"];
        var z = (double)parameters["z"];

        return (x * x + y * y + z * z) / 3.0 + Math.Sin(x) + Math.Cos(y) + Math.Sin(z);
    }
}

public class BenchmarkResult
{
    public int Run { get; set; }
    public string Problem { get; set; }
    public string Algorithm { get; set; }
    public int TrialCount { get; set; }
    public double WallClockSeconds { get; set; }
    public double BestValue { get; set; }
    public int TrialsToTarget95 { get; set; }
    public Dictionary<int, double> ConvergenceHistory { get; set; }
    public int Seed { get; set; }
}

public class MOBenchmarkResult
{
    public int Run { get; set; }
    public string Problem { get; set; }
    public int TrialCount { get; set; }
    public double WallClockSeconds { get; set; }
    public int ParetoFrontSize { get; set; }
    public int FeasibleSolutions { get; set; }
    public int TotalTrials { get; set; }
    public double ConstraintViolationRate { get; set; }
    public double Hypervolume { get; set; }
    public int Seed { get; set; }
}

public class PruningBenchmarkResult
{
    public string PrunerName { get; set; }
    public int TotalTrials { get; set; }
    public int PrunedCount { get; set; }
    public double PruningRate { get; set; }
    public double WallClockSeconds { get; set; }
    public double BestValue { get; set; }
}
