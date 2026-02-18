using OptiSharp.Models;
using OptiSharp.Samplers;
using OptiSharp.Samplers.Tpe;

namespace OptiSharp.Tests;

public sealed class ConvergenceTests
{
    [Fact]
    public void Tpe_Quadratic_BeatRandom()
    {
        // f(x) = (x - 3)^2, optimum at x=3
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        var tpeWins = 0;
        var runs = 10;

        for (var run = 0; run < runs; run++)
        {
            var seed = run * 100;
            var tpeBest = RunOptimization(new TpeSampler(new TpeSamplerConfig { NStartupTrials = 10, Seed = seed }),
                space, 100, x => Math.Pow((double)x["x"] - 3.0, 2));
            var rndBest = RunOptimization(new RandomSampler(seed),
                space, 100, x => Math.Pow((double)x["x"] - 3.0, 2));

            if (tpeBest < rndBest) tpeWins++;
        }

        Assert.True(tpeWins >= 7, $"TPE won {tpeWins}/{runs} — expected >= 7");
    }

    [Fact]
    public void Tpe_Sphere5D_BeatRandom()
    {
        // f(x) = Σ xi^2, optimum at origin
        var space = new SearchSpace(
            Enumerable.Range(0, 5).Select(i => new FloatRange($"x{i}", -5, 5)).ToArray());

        var tpeWins = 0;
        var runs = 8;

        for (var run = 0; run < runs; run++)
        {
            var seed = run * 100;
            double Objective(IReadOnlyDictionary<string, object> p)
                => Enumerable.Range(0, 5).Sum(i => Math.Pow((double)p[$"x{i}"], 2));

            var tpeBest = RunOptimization(new TpeSampler(new TpeSamplerConfig { NStartupTrials = 15, Seed = seed }),
                space, 200, Objective);
            var rndBest = RunOptimization(new RandomSampler(seed), space, 200, Objective);

            if (tpeBest < rndBest) tpeWins++;
        }

        Assert.True(tpeWins >= 5, $"TPE won {tpeWins}/{runs} — expected >= 5");
    }

    [Fact]
    public void Tpe_Rosenbrock_BeatRandom()
    {
        // f(x,y) = (1-x)^2 + 100(y-x^2)^2, optimum at (1,1)
        var space = new SearchSpace([new FloatRange("x", -2, 2), new FloatRange("y", -2, 2)]);
        var tpeWins = 0;
        var runs = 8;

        for (var run = 0; run < runs; run++)
        {
            var seed = run * 100;
            double Objective(IReadOnlyDictionary<string, object> p)
            {
                var x = (double)p["x"];
                var y = (double)p["y"];
                return Math.Pow(1 - x, 2) + 100 * Math.Pow(y - x * x, 2);
            }

            var tpeBest = RunOptimization(new TpeSampler(new TpeSamplerConfig { NStartupTrials = 15, Seed = seed }),
                space, 200, Objective);
            var rndBest = RunOptimization(new RandomSampler(seed), space, 200, Objective);

            if (tpeBest < rndBest) tpeWins++;
        }

        Assert.True(tpeWins >= 5, $"TPE won {tpeWins}/{runs} — expected >= 5");
    }

    [Fact]
    public void Tpe_MixedSpace_BeatRandom()
    {
        // Mixed: int + float + categorical
        var space = new SearchSpace([
            new IntRange("a", 0, 10),
            new FloatRange("b", -5, 5),
            new FloatRange("c", 0.01, 10, Log: true),
            new CategoricalRange("d", [1.0, 2.0, 3.0])
        ]);

        var tpeWins = 0;
        var runs = 8;

        for (var run = 0; run < runs; run++)
        {
            var seed = run * 100;
            double Objective(IReadOnlyDictionary<string, object> p)
            {
                var a = (int)p["a"];
                var b = (double)p["b"];
                var c = (double)p["c"];
                var d = (double)p["d"];
                return Math.Pow(a - 5, 2) + Math.Pow(b - 1, 2) + Math.Pow(Math.Log(c) - Math.Log(0.1), 2) + (d == 2.0 ? 0 : 10);
            }

            var tpeBest = RunOptimization(new TpeSampler(new TpeSamplerConfig { NStartupTrials = 15, Seed = seed }),
                space, 150, Objective);
            var rndBest = RunOptimization(new RandomSampler(seed), space, 150, Objective);

            if (tpeBest < rndBest) tpeWins++;
        }

        Assert.True(tpeWins >= 5, $"TPE won {tpeWins}/{runs} — expected >= 5");
    }

    [Fact]
    public void Tpe_LogScale_FindsOptimum()
    {
        // f(x) = (log(x) - log(0.001))^2, optimum near 0.001
        var space = new SearchSpace([new FloatRange("x", 0.0001, 1.0, Log: true)]);
        var config = new TpeSamplerConfig { NStartupTrials = 10, Seed = 42 };

        using var study = Optimizer.CreateStudy("log_test", space, config: config);

        for (var i = 0; i < 100; i++)
        {
            var trial = study.Ask();
            var x = (double)trial.Parameters["x"];
            var loss = Math.Pow(Math.Log(x) - Math.Log(0.001), 2);
            study.Tell(trial.Number, loss);
        }

        var best = (double)study.BestTrial!.Parameters["x"];
        // Should be within 10x of 0.001 (i.e., 0.0001 to 0.01)
        Assert.InRange(best, 0.0001, 0.01);
    }

    [Fact]
    public void Tpe_62Params_Converges()
    {
        // Realistic: 62 parameters like our hyperopt space
        var ranges = Enumerable.Range(0, 62)
            .Select(i => (ParameterRange)new FloatRange($"p{i}", 0, 10))
            .ToArray();
        var space = new SearchSpace(ranges);
        var config = new TpeSamplerConfig { NStartupTrials = 50, Seed = 42 };

        // Target: random quadratic f(p) = Σ (pi - 5)^2
        double Objective(IReadOnlyDictionary<string, object> p)
            => Enumerable.Range(0, 62).Sum(i => Math.Pow((double)p[$"p{i}"] - 5.0, 2));

        using var study = Optimizer.CreateStudy("big_test", space, config: config);
        var losses = new List<double>();

        for (var i = 0; i < 500; i++)
        {
            var trial = study.Ask();
            var loss = Objective(trial.Parameters);
            study.Tell(trial.Number, loss);
            losses.Add(loss);
        }

        // In 62 dimensions, 500 trials can't dominate — but best should beat average
        var bestLoss = study.BestTrial!.Value!.Value;
        var avgLoss = losses.Average();
        Assert.True(bestLoss < avgLoss * 0.9,
            $"Best {bestLoss:F2} should be < 90% of avg {avgLoss:F2}");
    }

    private static double RunOptimization(
        ISampler sampler, SearchSpace space, int trials,
        Func<IReadOnlyDictionary<string, object>, double> objective)
    {
        var trialList = new List<Trial>();
        var bestValue = double.MaxValue;

        for (var i = 0; i < trials; i++)
        {
            var parameters = sampler.Sample(trialList, StudyDirection.Minimize, space);
            var value = objective(parameters);
            var trial = new Trial(i, parameters) { State = TrialState.Complete, Value = value };
            trialList.Add(trial);
            if (value < bestValue) bestValue = value;
        }

        return bestValue;
    }
}
