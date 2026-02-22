using OptiSharp.Benchmarks;

// Check for --matrix CLI mode (for automated comparison benchmarks)
if (args.Length > 0 && args[0] == "--matrix")
{
    return MatrixBenchmarkRunner.Main(args.Skip(1).ToArray());
}

Console.WriteLine("╔════════════════════════════════════════════════════════════════╗");
Console.WriteLine("║         OptiSharp Comprehensive Benchmark Suite v0.1.0          ║");
Console.WriteLine("║  Real-world + Load Tests: CASH, NN, MO, Pruning, Scale         ║");
Console.WriteLine("║  Samplers: TPE, Random, CMA-ES                                 ║");
Console.WriteLine("╚════════════════════════════════════════════════════════════════╝");

var benchmarkRunner = new BenchmarkRunner();
var loadTestRunner = new LoadTestRunner();
var unifiedBenchmarkRunner = new UnifiedBenchmarkRunner();

Console.WriteLine("\nSelect benchmark to run:");
Console.WriteLine("STANDARD BENCHMARKS:");
Console.WriteLine("1. CASH (Algorithm Selection + HPO)");
Console.WriteLine("2. Neural Network Tuning");
Console.WriteLine("3. Multi-Objective with Constraints");
Console.WriteLine("4. Pruning Effectiveness");
Console.WriteLine("\nLOAD TESTS (5 Scales, All Features):");
Console.WriteLine("5. Load Test - Tiny (100 trials, 10 params)");
Console.WriteLine("6. Load Test - Small (1k trials, 50 params)");
Console.WriteLine("7. Load Test - Medium (5k trials, 100 params)");
Console.WriteLine("8. Load Test - All + Sampler Comparison (TPE, Random, CMA-ES)");
Console.WriteLine("\nUNIFIED BENCHMARKS (Identical datasets for OptiSharp vs Optuna):");
Console.WriteLine("11. Unified - Small (500 trials, 20 params)");
Console.WriteLine("12. Unified - Medium (5k trials, 100 params)");
Console.WriteLine("13. Unified - Large (50k trials, 200 params)");
Console.WriteLine("\nCOMBINED:");
Console.WriteLine("9. Run All Standard Benchmarks");
Console.WriteLine("10. Run All Load Tests (up to Medium)");
Console.Write("\nChoice (1-13): ");

var choice = Console.ReadLine();

var stopwatch = System.Diagnostics.Stopwatch.StartNew();

switch (choice?.Trim())
{
    case "1":
        benchmarkRunner.BenchmarkCASH();
        break;
    case "2":
        benchmarkRunner.BenchmarkNeuralNetworkTuning();
        break;
    case "3":
        benchmarkRunner.BenchmarkMultiObjectiveConstraints();
        break;
    case "4":
        benchmarkRunner.BenchmarkPruning();
        break;
    case "5":
        loadTestRunner.RunAllTests(maxScale: 1, samplerFilter: "tpe");
        break;
    case "6":
        loadTestRunner.RunAllTests(maxScale: 2, samplerFilter: "tpe");
        break;
    case "7":
        loadTestRunner.RunAllTests(maxScale: 3, samplerFilter: "tpe");
        break;
    case "8":
        loadTestRunner.RunAllTests(maxScale: 2, samplerFilter: "all");
        break;
    case "9":
        benchmarkRunner.BenchmarkCASH();
        benchmarkRunner.BenchmarkNeuralNetworkTuning();
        benchmarkRunner.BenchmarkMultiObjectiveConstraints();
        benchmarkRunner.BenchmarkPruning();
        break;
    case "10":
        loadTestRunner.RunAllTests(maxScale: 3, samplerFilter: "tpe");
        break;
    case "11":
        unifiedBenchmarkRunner.RunUnifiedBenchmarks("Small");
        break;
    case "12":
        unifiedBenchmarkRunner.RunUnifiedBenchmarks("Medium");
        break;
    case "13":
        unifiedBenchmarkRunner.RunUnifiedBenchmarks("Large");
        break;
    default:
        Console.WriteLine("Invalid choice.");
        break;
}

stopwatch.Stop();
Console.WriteLine($"\n╔════════════════════════════════════════════════════════════════╗");
Console.WriteLine($"║ Benchmarks complete in {stopwatch.Elapsed.TotalSeconds:F2} seconds");
Console.WriteLine($"║ Results saved to ./benchmark-results/");
Console.WriteLine($"╚════════════════════════════════════════════════════════════════╝");

return 0;
