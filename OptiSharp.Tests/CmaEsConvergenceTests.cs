using OptiSharp.Models;
using OptiSharp.Samplers;
using OptiSharp.Samplers.CmaEs;

namespace OptiSharp.Tests;

public sealed class CmaEsConvergenceTests
{
    [Fact]
    public void CmaEs_Quadratic_BeatRandom()
    {
        // f(x) = (x - 3)^2, optimum at x=3
        var space = new SearchSpace([new FloatRange("x", 0, 10)]);
        var cmaWins = 0;
        var runs = 10;

        for (var run = 0; run < runs; run++)
        {
            var seed = run * 100;
            var cmaBest = RunOptimization(
                new CmaEsSampler(new CmaEsSamplerConfig { Seed = seed }),
                space, 100, x => Math.Pow((double)x["x"] - 3.0, 2));
            var rndBest = RunOptimization(
                new RandomSampler(seed),
                space, 100, x => Math.Pow((double)x["x"] - 3.0, 2));

            if (cmaBest < rndBest) cmaWins++;
        }

        Assert.True(cmaWins >= 6, $"CMA-ES won {cmaWins}/{runs} — expected >= 6");
    }

    [Fact]
    public void CmaEs_Sphere5D_BeatRandom()
    {
        // f(x) = sum(xi^2), optimum at origin
        var space = new SearchSpace(
            Enumerable.Range(0, 5).Select(i => new FloatRange($"x{i}", -5, 5)).ToArray());

        var cmaWins = 0;
        var runs = 8;

        for (var run = 0; run < runs; run++)
        {
            var seed = run * 100;
            double Objective(IReadOnlyDictionary<string, object> p)
                => Enumerable.Range(0, 5).Sum(i => Math.Pow((double)p[$"x{i}"], 2));

            var cmaBest = RunOptimization(
                new CmaEsSampler(new CmaEsSamplerConfig { Seed = seed }),
                space, 200, Objective);
            var rndBest = RunOptimization(
                new RandomSampler(seed), space, 200, Objective);

            if (cmaBest < rndBest) cmaWins++;
        }

        Assert.True(cmaWins >= 5, $"CMA-ES won {cmaWins}/{runs} — expected >= 5");
    }

    [Fact]
    public void CmaEs_Rosenbrock_BeatRandom()
    {
        // f(x,y) = (1-x)^2 + 100(y-x^2)^2, optimum at (1,1)
        // CMA-ES should excel here — handles the correlated narrow valley
        var space = new SearchSpace([new FloatRange("x", -2, 2), new FloatRange("y", -2, 2)]);
        var cmaWins = 0;
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

            var cmaBest = RunOptimization(
                new CmaEsSampler(new CmaEsSamplerConfig { Seed = seed }),
                space, 200, Objective);
            var rndBest = RunOptimization(
                new RandomSampler(seed), space, 200, Objective);

            if (cmaBest < rndBest) cmaWins++;
        }

        Assert.True(cmaWins >= 5, $"CMA-ES won {cmaWins}/{runs} — expected >= 5");
    }

    [Fact]
    public void CmaEs_20Params_Converges()
    {
        // 20-dimensional sphere — realistic medium search space
        var ranges = Enumerable.Range(0, 20)
            .Select(i => (ParameterRange)new FloatRange($"p{i}", 0, 10))
            .ToArray();
        var space = new SearchSpace(ranges);

        double Objective(IReadOnlyDictionary<string, object> p)
            => Enumerable.Range(0, 20).Sum(i => Math.Pow((double)p[$"p{i}"] - 5.0, 2));

        using var study = Optimizer.CreateStudyWithCmaEs("cma_20d", space,
            config: new CmaEsSamplerConfig { Seed = 42 });
        var losses = new List<double>();

        for (var i = 0; i < 300; i++)
        {
            var trial = study.Ask();
            var loss = Objective(trial.Parameters);
            study.Tell(trial.Number, loss);
            losses.Add(loss);
        }

        // Best should beat average significantly
        var bestLoss = study.BestTrial!.Value!.Value;
        var avgLoss = losses.Average();
        Assert.True(bestLoss < avgLoss * 0.5,
            $"Best {bestLoss:F2} should be < 50% of avg {avgLoss:F2}");
    }

    [Fact]
    public void CmaEs_MixedSpace_HandlesCategorials()
    {
        // Mixed: int + float + categorical
        var space = new SearchSpace([
            new IntRange("a", 0, 10),
            new FloatRange("b", -5, 5),
            new CategoricalRange("d", [1.0, 2.0, 3.0])
        ]);

        var cmaWins = 0;
        var runs = 8;

        for (var run = 0; run < runs; run++)
        {
            var seed = run * 100;
            double Objective(IReadOnlyDictionary<string, object> p)
            {
                var a = (int)p["a"];
                var b = (double)p["b"];
                return Math.Pow(a - 5, 2) + Math.Pow(b - 1, 2);
            }

            var cmaBest = RunOptimization(
                new CmaEsSampler(new CmaEsSamplerConfig { Seed = seed }),
                space, 150, Objective);
            var rndBest = RunOptimization(
                new RandomSampler(seed), space, 150, Objective);

            if (cmaBest < rndBest) cmaWins++;
        }

        Assert.True(cmaWins >= 4, $"CMA-ES won {cmaWins}/{runs} — expected >= 4");
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