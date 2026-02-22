using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using OptiSharp;
using OptiSharp.Models;
using OptiSharp.Pruning;
using OptiSharp.Samplers.CmaEs;
using OptiSharp.Samplers.Tpe;

namespace OptiSharp.Benchmarks
{
    public class MatrixBenchmarkRunner
    {
        public class Config
        {
            public string Sampler { get; set; }      // tpe, random, cmaes
            public int Params { get; set; }          // 10, 50, 100, 200
            public int Trials { get; set; }          // 100, 300, 500, 1000, 10000, 100000
            public string Objective { get; set; }    // sphere, rosenbrock, rastrigin, ackley
            public string Pruner { get; set; }       // none, median, sha
            public string Tier { get; set; }         // fast, extended
            public string Output { get; set; }       // output JSON path
        }

        public static int Main(string[] args)
        {
            var config = ParseArgs(args);
            if (config == null)
            {
                PrintUsage();
                return 1;
            }

            try
            {
                var result = RunBenchmark(config);
                result.SerializeToFile(config.Output);
                Console.WriteLine($"✓ Result saved to {config.Output}");
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error: {ex.Message}");
                return 1;
            }
        }

        private static Config ParseArgs(string[] args)
        {
            var config = new Config();

            for (int i = 0; i < args.Length; i += 2)
            {
                if (i + 1 >= args.Length)
                    return null;

                string key = args[i].TrimStart('-').ToLower();
                string value = args[i + 1];

                switch (key)
                {
                    case "sampler":
                        config.Sampler = value.ToLower();
                        break;
                    case "params":
                        config.Params = int.Parse(value);
                        break;
                    case "trials":
                        config.Trials = int.Parse(value);
                        break;
                    case "objective":
                        config.Objective = value.ToLower();
                        break;
                    case "pruner":
                        config.Pruner = value.ToLower();
                        break;
                    case "tier":
                        config.Tier = value.ToLower();
                        break;
                    case "output":
                        config.Output = value;
                        break;
                    default:
                        return null;
                }
            }

            // Validate
            if (string.IsNullOrEmpty(config.Sampler) || string.IsNullOrEmpty(config.Objective)
                || string.IsNullOrEmpty(config.Pruner) || string.IsNullOrEmpty(config.Output))
                return null;

            return config;
        }

        private static void PrintUsage()
        {
            Console.WriteLine(@"
Usage: --matrix --sampler <tpe|random|cmaes> --params <10|50|100|200> --trials <N> \
  --objective <sphere|rosenbrock|rastrigin|ackley> --pruner <none|median|sha> \
  --tier <fast|extended> --output <path.json>
");
        }

        public class BenchmarkResult
        {
            [JsonPropertyName("framework")]
            public string Framework { get; set; } = "optisharp";

            [JsonPropertyName("sampler")]
            public string Sampler { get; set; }

            [JsonPropertyName("objective")]
            public string Objective { get; set; }

            [JsonPropertyName("n_params")]
            public int NParams { get; set; }

            [JsonPropertyName("n_trials")]
            public int NTrials { get; set; }

            [JsonPropertyName("pruner")]
            public string Pruner { get; set; }

            [JsonPropertyName("tier")]
            public string Tier { get; set; }

            [JsonPropertyName("best_value")]
            public double BestValue { get; set; }

            [JsonPropertyName("elapsed_ms")]
            public long ElapsedMs { get; set; }

            [JsonPropertyName("trials_per_second")]
            public double TrialsPerSecond { get; set; }

            [JsonPropertyName("pruned_trials")]
            public int PrunedTrials { get; set; }

            [JsonPropertyName("pruning_rate")]
            public double PruningRate { get; set; }

            [JsonPropertyName("convergence")]
            public double[] Convergence { get; set; }  // 5 checkpoints: 20%, 40%, 60%, 80%, 100%

            [JsonPropertyName("seed")]
            public int Seed { get; set; } = 42;

            public void SerializeToFile(string path)
            {
                string dir = Path.GetDirectoryName(path);
                if (!Directory.Exists(dir))
                    Directory.CreateDirectory(dir);

                var options = new JsonSerializerOptions { WriteIndented = true };
                string json = JsonSerializer.Serialize(this, options);
                File.WriteAllText(path, json);
            }
        }

        private static BenchmarkResult RunBenchmark(Config config)
        {
            var watch = Stopwatch.StartNew();

            // Build search space
            var (low, high) = Objectives.GetRange(config.Objective);
            var ranges = Enumerable.Range(0, config.Params)
                .Select(i => (ParameterRange)new FloatRange($"x{i}", low, high))
                .ToList();
            var searchSpace = new SearchSpace(ranges);

            // Get objective function
            var objectiveFn = Objectives.GetObjective(config.Objective);

            // Create pruner
            IPruner pruner = config.Pruner switch
            {
                "none" => new NopPruner(),
                "median" => new MedianPruner(),
                "sha" => new SuccessiveHalvingPruner(),
                _ => throw new ArgumentException($"Unknown pruner: {config.Pruner}")
            };

            // Create study
            Study study = config.Sampler switch
            {
                "tpe" => Optimizer.CreateStudy(
                    name: $"matrix_{config.Objective}_{config.Params}p",
                    searchSpace: searchSpace,
                    direction: StudyDirection.Minimize,
                    pruner: pruner),
                "random" => Optimizer.CreateStudyWithRandomSampler(
                    name: $"matrix_{config.Objective}_{config.Params}p",
                    searchSpace: searchSpace,
                    direction: StudyDirection.Minimize,
                    seed: 42,
                    pruner: pruner),
                "cmaes" => Optimizer.CreateStudyWithCmaEs(
                    name: $"matrix_{config.Objective}_{config.Params}p",
                    searchSpace: searchSpace,
                    direction: StudyDirection.Minimize,
                    config: new CmaEsSamplerConfig { Seed = 42 },
                    pruner: pruner),
                _ => throw new ArgumentException($"Unknown sampler: {config.Sampler}")
            };

            // Run optimization
            var convergence = new List<double>();
            var checkpoints = new[] { 0.2, 0.4, 0.6, 0.8, 1.0 };
            int checkpointIdx = 0;
            int prunedCount = 0;

            for (int i = 0; i < config.Trials; i++)
            {
                // Ask
                var trial = study.Ask();

                // Evaluate (convert IReadOnlyDictionary to Dictionary for objective function)
                var paramDict = new Dictionary<string, object>(trial.Parameters);
                double value = objectiveFn(paramDict);

                // Report intermediate values if pruning is enabled
                if (config.Pruner != "none")
                {
                    for (int step = 0; step < 10; step++)
                    {
                        double intermediateValue = Objectives.GetIntermediateValue(
                            value, step, trial.Number, seed: 42);
                        trial.Report(intermediateValue, step);

                        // Check if should prune
                        if (study.ShouldPrune(trial))
                        {
                            study.Tell(trial.Number, TrialState.Pruned);
                            prunedCount++;
                            goto next_trial;
                        }
                    }
                }

                // Tell final value
                study.Tell(trial.Number, value);

            next_trial:
                // Record convergence at checkpoints
                if (checkpointIdx < checkpoints.Length)
                {
                    int checkpointTrial = (int)(config.Trials * checkpoints[checkpointIdx]);
                    if (i >= checkpointTrial - 1)
                    {
                        convergence.Add(study.BestTrial?.Value ?? double.PositiveInfinity);
                        checkpointIdx++;
                    }
                }
            }

            watch.Stop();

            // Ensure we have exactly 5 convergence points
            while (convergence.Count < 5)
                convergence.Add(study.BestTrial?.Value ?? double.PositiveInfinity);
            convergence = convergence.Take(5).ToList();

            var result = new BenchmarkResult
            {
                Sampler = config.Sampler,
                Objective = config.Objective,
                NParams = config.Params,
                NTrials = config.Trials,
                Pruner = config.Pruner,
                Tier = config.Tier,
                BestValue = study.BestTrial?.Value ?? double.PositiveInfinity,
                ElapsedMs = watch.ElapsedMilliseconds,
                TrialsPerSecond = config.Trials / (watch.ElapsedMilliseconds / 1000.0),
                PrunedTrials = prunedCount,
                PruningRate = (double)prunedCount / config.Trials,
                Convergence = convergence.ToArray()
            };

            return result;
        }
    }
}
