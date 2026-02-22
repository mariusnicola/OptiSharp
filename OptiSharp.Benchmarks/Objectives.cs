using System;
using System.Collections.Generic;
using System.Linq;

namespace OptiSharp.Benchmarks
{
    /// <summary>
    /// Identical objective functions for C# and Python comparison.
    /// All functions minimize; range varies per function.
    /// </summary>
    public static class Objectives
    {
        /// <summary>
        /// Sphere function: Σ xᵢ²
        /// Global minimum: 0
        /// Range: [-5, 5]
        /// </summary>
        public static double Sphere(Dictionary<string, object> parameters)
        {
            return parameters.Values
                .Cast<double>()
                .Sum(x => x * x);
        }

        /// <summary>
        /// Rosenbrock function: Σ [100(xᵢ₊₁ - xᵢ²)² + (1 - xᵢ)²]
        /// Global minimum: 0
        /// Range: [-2, 2]
        /// </summary>
        public static double Rosenbrock(Dictionary<string, object> parameters)
        {
            var values = parameters.Values.Cast<double>().ToList();
            double sum = 0;
            for (int i = 0; i < values.Count - 1; i++)
            {
                double xi = values[i];
                double xi1 = values[i + 1];
                double term1 = 100 * Math.Pow(xi1 - xi * xi, 2);
                double term2 = Math.Pow(1 - xi, 2);
                sum += term1 + term2;
            }
            return sum;
        }

        /// <summary>
        /// Rastrigin function: 10n + Σ [xᵢ² - 10·cos(2π·xᵢ)]
        /// Global minimum: 0
        /// Range: [-5.12, 5.12]
        /// Highly multimodal - tests exploration.
        /// </summary>
        public static double Rastrigin(Dictionary<string, object> parameters)
        {
            var values = parameters.Values.Cast<double>().ToList();
            int n = values.Count;
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                double xi = values[i];
                sum += xi * xi - 10 * Math.Cos(2 * Math.PI * xi);
            }
            return 10 * n + sum;
        }

        /// <summary>
        /// Ackley function: -20·exp(-0.2·√(Σxᵢ²/n)) - exp(Σcos(2πxᵢ)/n) + 20 + e
        /// Global minimum: 0
        /// Range: [-32.768, 32.768]
        /// Smooth with local noise - tests balance.
        /// </summary>
        public static double Ackley(Dictionary<string, object> parameters)
        {
            var values = parameters.Values.Cast<double>().ToArray();
            int n = values.Length;

            double sum_sq = values.Sum(x => x * x);
            double sum_cos = values.Sum(x => Math.Cos(2 * Math.PI * x));

            double term1 = -20 * Math.Exp(-0.2 * Math.Sqrt(sum_sq / n));
            double term2 = -Math.Exp(sum_cos / n);
            double e = Math.E; // Euler's number ≈ 2.71828

            return term1 + term2 + 20 + e;
        }

        /// <summary>
        /// Get objective function by name.
        /// </summary>
        public static Func<Dictionary<string, object>, double> GetObjective(string name)
        {
            return name.ToLower() switch
            {
                "sphere" => Sphere,
                "rosenbrock" => Rosenbrock,
                "rastrigin" => Rastrigin,
                "ackley" => Ackley,
                _ => throw new ArgumentException($"Unknown objective: {name}")
            };
        }

        /// <summary>
        /// Get the search range for an objective.
        /// </summary>
        public static (double low, double high) GetRange(string objectiveName)
        {
            return objectiveName.ToLower() switch
            {
                "sphere" => (-5.0, 5.0),
                "rosenbrock" => (-2.0, 2.0),
                "rastrigin" => (-5.12, 5.12),
                "ackley" => (-32.768, 32.768),
                _ => throw new ArgumentException($"Unknown objective: {objectiveName}")
            };
        }

        /// <summary>
        /// Simulate intermediate pruning values.
        /// At step k (0-9), report: objective(params) * (1 + (9-k) * 0.5 * noise)
        /// where noise is seeded for reproducibility.
        /// Step 9 (final) returns the exact value.
        /// </summary>
        public static double GetIntermediateValue(
            double trueValue,
            int step,  // 0..9
            int trialNumber,  // used to seed noise
            int seed = 42)
        {
            if (step == 9)
                return trueValue;  // Final step: exact value

            // Seeded pseudo-random noise for reproducibility
            var rng = new Random(seed ^ trialNumber ^ step);
            double noise = rng.NextDouble();  // [0, 1)
            double factor = 1 + (9 - step) * 0.5 * noise;
            return trueValue * factor;
        }
    }
}
