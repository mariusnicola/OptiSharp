# OptiSharp

[![CI](https://github.com/mariusnicola/OptiSharp/actions/workflows/ci.yml/badge.svg)](https://github.com/mariusnicola/OptiSharp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![.NET](https://img.shields.io/badge/.NET-Standard%202.1%2B-purple.svg)](https://dotnet.microsoft.com/)

A pure C# optimization library for .NET. No Python. No subprocess. No interop headaches.

Three samplers (Random, TPE, CMA-ES), multi-objective Pareto optimization, early stopping/pruning, constraint-aware optimization, study persistence, warm start, Population-Based Training, optional NVIDIA CUDA GPU acceleration, thread-safe ask/tell API for distributed workloads. Used in large-scale hyperparameter optimization for trading strategy discovery.

Targets **.NET Standard 2.1** — works with .NET Core 3.0+, .NET 5/6/7/8/9, and Unity 2021+.

---

## Why This Exists

If you've ever needed hyperparameter optimization in .NET, you've hit the same wall:

- **Optuna** is Python-only. You end up shelling out to a Python subprocess, serializing parameters as JSON, parsing stdout. It works, but it's fragile, slow, and painful to debug.
- **ML.NET AutoML** is limited to ML.NET pipelines. If you're optimizing anything else — game parameters, simulation configs, engineering designs — it doesn't help.
- **Random search** works but wastes compute. You're leaving performance on the table.

OptiSharp brings **Optuna-style API and algorithms, implemented natively in C#** and validated with convergence + performance benchmarks — same core algorithms (TPE, CMA-ES), same ask/tell API pattern, zero Python dependency.

---

## Installation

**Project reference** (clone this repo):

```bash
git clone https://github.com/mariusnicola/OptiSharp.git
```

Then add a project reference in your `.csproj`:

```xml
<ProjectReference Include="..\OptiSharp\OptiSharp.csproj" />
```

### Dependencies

- `MathNet.Numerics` — Linear algebra for CMA-ES
- `ILGPU` + `ILGPU.Algorithms` — GPU acceleration (GPU code path is only used when `ComputeBackend.Gpu` or `Auto` is set)
- `PolySharp` — Polyfills for modern C# features on .NET Standard 2.1

---

## Quick Start — 30 Seconds

```csharp
using OptiSharp;
using OptiSharp.Models;

// 1. Define what you're searching over
var space = new SearchSpace([
    new FloatRange("learning_rate", 0.0001, 1.0, Log: true),
    new IntRange("num_layers", 1, 10),
    new CategoricalRange("activation", ["relu", "tanh", "gelu"])
]);

// 2. Create a study
using var study = Optimizer.CreateStudy("my_experiment", space);

// 3. Run optimization
for (int i = 0; i < 200; i++)
{
    var trial = study.Ask();

    // Extract parameters (strongly typed)
    var lr = (double)trial.Parameters["learning_rate"];
    var layers = (int)trial.Parameters["num_layers"];
    var activation = (string)trial.Parameters["activation"];

    // Evaluate your objective (whatever you're optimizing)
    var loss = TrainAndEvaluate(lr, layers, activation);

    // Report the result
    study.Tell(trial.Number, loss);
}

// 4. Get the best result
var best = study.BestTrial!;
Console.WriteLine($"Best loss: {best.Value:F6}");
Console.WriteLine($"Parameters: {string.Join(", ", best.Parameters.Select(p => $"{p.Key}={p.Value}"))}");
```

That's it. No configuration files. No servers. No Python environment.

---

## Core Concepts

### Search Space

Define what parameters you want to optimize and their bounds:

```csharp
var space = new SearchSpace([
    // Continuous parameter
    new FloatRange("temperature", 0.01, 2.0),

    // Continuous parameter with log-uniform sampling (spans orders of magnitude)
    new FloatRange("learning_rate", 1e-5, 1.0, Log: true),

    // Integer parameter
    new IntRange("batch_size", 16, 512),

    // Integer parameter with step (multiples of 8 only)
    new IntRange("hidden_dim", 64, 1024, Step: 8),

    // Categorical parameter (strings, numbers, anything)
    new CategoricalRange("optimizer", ["adam", "sgd", "adamw"]),
    new CategoricalRange("use_dropout", [true, false])
]);
```

### Study

A study manages the optimization loop. It holds the trial history, tracks the best result, and coordinates with the sampler:

```csharp
// Minimize (default) — lower is better
using var study = Optimizer.CreateStudy("minimize_loss", space);

// Maximize — higher is better
using var study = Optimizer.CreateStudy("maximize_accuracy", space, StudyDirection.Maximize);
```

### Ask / Tell

The API is intentionally simple. **Ask** for a trial, **tell** the result:

```csharp
var trial = study.Ask();           // Get next suggested parameters
// ... evaluate ...
study.Tell(trial.Number, 0.95);    // Report the objective value
```

This decouples parameter suggestion from evaluation. Your evaluation can be anything — a function call, a subprocess, a network request, a multi-hour simulation.

### Batch Trials

When your evaluations are expensive and you have multiple cores (or machines), request a batch and evaluate in parallel:

```csharp
using var study = Optimizer.CreateStudy("batch_example", space, config: new TpeSamplerConfig
{
    ConstantLiar = true  // Prevents suggesting duplicate parameters for concurrent trials
});

// Request 10 trials at once
var batch = study.AskBatch(10);

// Evaluate them all in parallel
var results = batch
    .AsParallel()
    .Select(trial =>
    {
        var value = Evaluate(trial.Parameters);
        return new TrialResult(trial.Number, value, TrialState.Complete);
    })
    .ToList();

// Report all results back
study.TellBatch(results);

// Best result across all trials
Console.WriteLine($"Best: {study.BestTrial!.Value:F6}");
```

You can also run batches in waves for iterative improvement — each wave learns from the previous:

```csharp
for (int wave = 0; wave < 20; wave++)
{
    var batch = study.AskBatch(Environment.ProcessorCount);

    var results = batch.AsParallel().Select(trial =>
    {
        try
        {
            var value = Evaluate(trial.Parameters);
            return new TrialResult(trial.Number, value, TrialState.Complete);
        }
        catch
        {
            return new TrialResult(trial.Number, null, TrialState.Fail);
        }
    }).ToList();

    study.TellBatch(results);
    Console.WriteLine($"Wave {wave}: best = {study.BestTrial?.Value:F6}");
}
```

**Key detail:** `ConstantLiar = true` (default) ensures that when multiple trials are in flight simultaneously, the sampler adds running trials to the "bad" group. This prevents it from suggesting the same region of the search space multiple times in one batch.

### Failed Trials

Sometimes evaluations fail. Report them so the sampler can learn from failures:

```csharp
var trial = study.Ask();
try
{
    var value = Evaluate(trial.Parameters);
    study.Tell(trial.Number, value);
}
catch
{
    study.Tell(trial.Number, TrialState.Fail);
}
```

---

## Multi-Objective Optimization

Optimize multiple conflicting objectives simultaneously. OptiSharp computes the Pareto front — the set of
solutions where no single objective can be improved without worsening another.

```csharp
using OptiSharp;
using OptiSharp.Models;
using OptiSharp.MultiObjective;

// Two objectives: minimize latency, minimize memory usage
var space = new SearchSpace([
    new IntRange("num_layers", 1, 8),
    new IntRange("hidden_size", 32, 512, Step: 32),
    new FloatRange("dropout", 0.0, 0.5),
    new CategoricalRange("activation", ["relu", "gelu", "tanh"])
]);

var directions = new[] { StudyDirection.Minimize, StudyDirection.Minimize };
using var study = Optimizer.CreateStudy("latency_vs_memory", space, directions);

for (int i = 0; i < 100; i++)
{
    var trial = study.Ask();

    var (latencyMs, memoryMb) = EvaluateModel(trial.Parameters);

    // Report both objectives as an array
    study.Tell(trial.Number, new[] { latencyMs, memoryMb });
}

// Get the Pareto front — all non-dominated solutions
var front = study.ParetoFront;
Console.WriteLine($"Pareto front: {front.Count} solutions");

foreach (var t in front)
    Console.WriteLine($"  Trial {t.Number}: latency={t.Values![0]:F1}ms, memory={t.Values[1]:F0}MB");
```

### Batch reporting for multi-objective

Use `MoTrialResult` when reporting multiple trials at once:

```csharp
using OptiSharp.MultiObjective;

var batch = study.AskBatch(10);

var results = batch.AsParallel().Select(trial =>
{
    var (latency, memory) = EvaluateModel(trial.Parameters);
    return new MoTrialResult(trial.Number, new[] { latency, memory }, TrialState.Complete);
}).ToList();

study.TellBatch(results);
```

### Crowding distance — spread of Pareto front

NSGA-II style crowding distance measures how isolated each solution is on the Pareto front.
Boundary solutions get `PositiveInfinity`. Useful for filtering solutions that are too clustered:

```csharp
var front = study.ParetoFront;
var distances = ParetoUtils.CrowdingDistances(front, directions);

// Sort by diversity — most isolated solutions first
var diverse = front.Zip(distances)
    .OrderByDescending(pair => pair.Second)
    .Select(pair => pair.First)
    .ToList();
```

### Pareto dominance check

```csharp
var a = new[] { 1.0, 5.0 };  // Fast but memory-hungry
var b = new[] { 2.0, 3.0 };  // Slower but leaner

bool aWins = ParetoUtils.Dominates(a, b, directions);  // false — different trade-offs
```

### Mixed minimize/maximize

```csharp
// Maximize accuracy, minimize latency
var directions = new[] { StudyDirection.Maximize, StudyDirection.Minimize };
using var study = Optimizer.CreateStudy("accuracy_vs_latency", space, directions);

var trial = study.Ask();
study.Tell(trial.Number, new[] { 0.94, 12.5 });  // 94% accuracy, 12.5ms latency
```

---

## Pruning / Early Stopping

Prune unpromising trials mid-evaluation based on intermediate results. This is especially valuable
for long training runs — stop a trial early if its progress is clearly worse than completed trials.

### The pruning loop pattern

```csharp
using OptiSharp;
using OptiSharp.Models;
using OptiSharp.Pruning;

var space = new SearchSpace([
    new FloatRange("learning_rate", 1e-5, 1e-1, Log: true),
    new IntRange("batch_size", 16, 256, Step: 16),
    new CategoricalRange("optimizer", ["adam", "adamw", "sgd"])
]);

// Create study with a pruner
using var study = Optimizer.CreateStudy("neural_net", space,
    pruner: new MedianPruner());

for (int i = 0; i < 100; i++)
{
    var trial = study.Ask();

    for (int epoch = 1; epoch <= 50; epoch++)
    {
        var validationLoss = TrainOneEpoch(trial.Parameters, epoch);

        // Report intermediate result for this step
        trial.Report(validationLoss, epoch);

        // Check if this trial should be pruned
        if (study.ShouldPrune(trial))
        {
            study.Tell(trial.Number, TrialState.Pruned);
            goto nextTrial;
        }
    }

    var finalLoss = EvaluateFinal(trial.Parameters);
    study.Tell(trial.Number, finalLoss);

    nextTrial:;
}

var best = study.BestTrial!;  // BestTrial skips pruned trials
Console.WriteLine($"Best loss: {best.Value:F6}");
```

### MedianPruner — prune trials below the median

Prunes if the current intermediate value exceeds the median of completed trials at the same step.

```csharp
var pruner = new MedianPruner(new MedianPrunerConfig
{
    NStartupTrials = 5,    // Wait for 5 completed trials before pruning
    NWarmupSteps = 3,      // Don't prune during the first 3 steps
    IntervalSteps = 1      // Check every step
});

using var study = Optimizer.CreateStudy("experiment", space, pruner: pruner);
```

### PercentilePruner — custom threshold

Prunes trials below an arbitrary percentile instead of the median:

```csharp
var pruner = new PercentilePruner(new PercentilePrunerConfig
{
    Percentile = 25.0,     // Prune bottom 25% — more aggressive than median
    NStartupTrials = 5,
    NWarmupSteps = 2
});
```

### SuccessiveHalvingPruner — SHA algorithm

Implements Asynchronous Successive Halving. At each "rung" (power-of-eta step boundary),
only the top `1/eta` fraction of trials survive:

```csharp
var pruner = new SuccessiveHalvingPruner(new SuccessiveHalvingPrunerConfig
{
    MinResource = 1,          // Minimum steps before first rung check
    ReductionFactor = 3.0     // Keep top 1/3 at each rung (eta=3)
});
```

### Comparing pruner effectiveness

```csharp
// No pruning baseline
using var noPruneStudy = Optimizer.CreateStudy("baseline", space, pruner: new NopPruner());

// Median pruning
using var medianStudy = Optimizer.CreateStudy("median", space, pruner: new MedianPruner());

// 25th-percentile pruning (most aggressive)
using var pctStudy = Optimizer.CreateStudy("percentile", space,
    pruner: new PercentilePruner(new PercentilePrunerConfig { Percentile = 25.0 }));
```

---

## Constraint-Aware Optimization

Define hard constraints on top of your objective. Infeasible trials are automatically deprioritized
by TPE — the sampler pushes them into the "bad" group regardless of their objective value.

```csharp
var space = new SearchSpace([
    new CategoricalRange("instance_type", ["t3.medium", "m5.large", "c5.xlarge"]),
    new IntRange("replicas", 1, 20),
    new IntRange("memory_mb", 512, 8192, Step: 512)
]);

using var study = Optimizer.CreateStudy("infra_opt", space, StudyDirection.Minimize);

// Constraints: negative = feasible, positive = infeasible
study.SetConstraintFunc(trial =>
{
    var memory = (int)trial.Parameters["memory_mb"];
    var replicas = (int)trial.Parameters["replicas"];

    return new[]
    {
        memory * replicas - 32_000.0,   // Total memory must be <= 32 GB
        replicas - 10.0                  // Replicas must be <= 10
    };
});

for (int i = 0; i < 100; i++)
{
    var trial = study.Ask();
    var result = RunLoadTest(trial.Parameters);

    // Report objective even for infeasible trials — TPE uses this to learn
    study.Tell(trial.Number, result.CostPerHour);
}

// Filter to feasible trials only
var feasible = study.Trials.Where(study.IsFeasible).ToList();
Console.WriteLine($"Feasible: {feasible.Count}/{study.Trials.Count}");
Console.WriteLine($"Best feasible cost: {feasible.Min(t => t.Value):F2}/hr");
```

### Multiple constraints

```csharp
study.SetConstraintFunc(trial =>
{
    var x = (double)trial.Parameters["x"];
    return new[]
    {
        x - 5.0,    // x <= 5  (feasible when <= 0)
        -x + 1.0    // x >= 1  (feasible when <= 0)
    };
});
```

### IsFeasible check

```csharp
var trial = study.BestTrial;
if (study.IsFeasible(trial))
    Console.WriteLine("Best trial satisfies all constraints");
```

### Constraints + multi-objective

```csharp
var directions = new[] { StudyDirection.Minimize, StudyDirection.Maximize };
using var study = Optimizer.CreateStudy("mo_constrained", space, directions);

study.SetConstraintFunc(t => new[] { ComputeConstraint(t.Parameters) });

var trial = study.Ask();
study.Tell(trial.Number, new[] { cost, quality });  // Both ConstraintValues and Values populated
```

---

## Study Persistence

Save and load study state to disk. Resume optimization after restarts without losing trial history.

### Save a study

```csharp
using var study = Optimizer.CreateStudy("my_experiment", space);

// Run some trials...
for (int i = 0; i < 50; i++)
{
    var trial = study.Ask();
    study.Tell(trial.Number, Evaluate(trial.Parameters));
}

// Checkpoint to disk
study.Save("my_experiment.json");
Console.WriteLine($"Saved {study.Trials.Count(t => t.State == TrialState.Complete)} trials");
```

### Load and resume

```csharp
var space = new SearchSpace([
    new FloatRange("x", 0, 10),
    new IntRange("y", 1, 5)
]);

// Load resumes from saved state — all previous trials are pre-populated
using var study = Study.Load("my_experiment.json", space, new Samplers.Tpe.TpeSampler());

Console.WriteLine($"Resumed with {study.Trials.Count} prior trials");

// Continue where you left off
for (int i = 0; i < 50; i++)
{
    var trial = study.Ask();
    study.Tell(trial.Number, Evaluate(trial.Parameters));
}
```

### What is preserved

| Data | Saved? |
|------|--------|
| Trial parameters (float, int, categorical) | Yes |
| Objective value (`Value`) | Yes |
| Multi-objective values (`Values[]`) | Yes |
| Constraint values (`ConstraintValues[]`) | Yes |
| Intermediate values (`IntermediateValues`) | Yes |
| TrialState (Complete, Pruned) | Yes |
| Running trials | No |
| Failed trials | No |

**Parameter types are fully reconstructed** from the SearchSpace — floats come back as `double`,
ints as `int`, categoricals as `string`.

### Periodic checkpointing

```csharp
for (int i = 0; i < 500; i++)
{
    var trial = study.Ask();
    study.Tell(trial.Number, Evaluate(trial.Parameters));

    // Checkpoint every 50 trials
    if ((i + 1) % 50 == 0)
        study.Save($"checkpoint_{i + 1}.json");
}
```

---

## Warm Start / Transfer Learning

Pre-populate a study with results from a previous run. The sampler treats warm trials as real trial
history — they count toward `NStartupTrials` and inform the first TPE/CMA-ES suggestions.

### From a list of trials

```csharp
// Previous experiment results
var warmTrials = new List<Trial>
{
    new Trial(0, new Dictionary<string, object> { ["lr"] = 0.01, ["layers"] = 3 })
    {
        State = TrialState.Complete,
        Value = 0.42
    },
    new Trial(1, new Dictionary<string, object> { ["lr"] = 0.001, ["layers"] = 5 })
    {
        State = TrialState.Complete,
        Value = 0.35
    }
};

using var study = Optimizer.CreateStudy("new_experiment", space,
    warmStartTrials: warmTrials);

// Trial numbers are re-assigned from 0 — no gaps
Console.WriteLine(study.Trials[0].Number);  // 0
Console.WriteLine(study.Trials[1].Number);  // 1
```

### From an existing study

Copy all completed trials from a previous study object:

```csharp
using var study1 = Optimizer.CreateStudy("experiment_v1", space);
// ... run study1 ...

// study2 inherits all of study1's completed trials
using var study2 = Optimizer.CreateStudy("experiment_v2", space,
    fromStudy: study1);

Console.WriteLine($"Warm started with {study2.Trials.Count} trials from v1");
```

### Works with all factory methods

```csharp
// With CMA-ES
using var cmaStudy = Optimizer.CreateStudyWithCmaEs("cma_warm", space,
    warmStartTrials: previousTrials);

// With Random sampler
using var randStudy = Optimizer.CreateStudyWithRandomSampler("rand_warm", space,
    warmStartTrials: previousTrials);
```

### Warm start counts toward startup phase

With `NStartupTrials = 5` and 3 warm trials, TPE activates after only 2 more random trials:

```csharp
var config = new TpeSamplerConfig { NStartupTrials = 5 };
using var study = Optimizer.CreateStudy("fast_start", space,
    config: config,
    warmStartTrials: previousTrials); // 3 trials already loaded
// TPE activates after 2 more trials instead of 5
```

---

## Population-Based Training

PBT maintains a population of models training in parallel. At regular intervals, underperforming
members copy hyperparameters from top performers and apply random perturbations — combining the
exploitation of good configurations with exploration of nearby variants.

`PopulationBasedTrainer` is a standalone coordinator, separate from `Study`.

```csharp
using OptiSharp;
using OptiSharp.Models;
using OptiSharp.Pbt;

var space = new SearchSpace([
    new FloatRange("lr", 1e-4, 1e-1),
    new IntRange("batch_size", 16, 128),
    new CategoricalRange("optimizer", new object[] { "adam", "sgd", "rmsprop" })
]);

var pbt = new PopulationBasedTrainer(
    space,
    populationSize: 10,
    exploitFraction: 0.3,   // Bottom 30% will exploit top performers
    perturbFactor: 0.2,     // 20% multiplicative perturbation on float/int params
    seed: 42
);

// Generation 1 — initialize population with random hyperparameters
var population = pbt.AskPopulation().ToList();

for (int generation = 0; generation < 5; generation++)
{
    // Train each member, report performance
    var updated = population.Select(member =>
    {
        var loss = Train(member.Parameters, steps: 100);
        return pbt.Report(member, performance: -loss, step: (generation + 1) * 100);
    }).ToList();

    // Evolve: top 70% kept, bottom 30% exploit + explore
    population = pbt.Evolve(updated);

    var best = population.Max(m => m.Performance);
    Console.WriteLine($"Generation {generation + 1}: best performance = {best:F4}");
}

// Inspect best member's parameter history across generations
var bestMember = population.OrderByDescending(m => m.Performance).First();
Console.WriteLine($"Parameter evolution ({bestMember.ParameterHistory.Count} snapshots):");
foreach (var snapshot in bestMember.ParameterHistory)
    Console.WriteLine($"  lr={snapshot["lr"]:G4}, batch={snapshot["batch_size"]}");
```

### PbtMember — immutable record

`PbtMember` is an immutable record. `Report` and `Evolve` return new instances:

```csharp
var member = population[0];
// member.Performance == double.NegativeInfinity initially

var updated = pbt.Report(member, performance: 0.92, step: 100);
// updated.Performance == 0.92, member is unchanged
```

### Constructor validation

```csharp
// These throw ArgumentException:
new PopulationBasedTrainer(space, populationSize: 1);      // Must be >= 2
new PopulationBasedTrainer(space, exploitFraction: 1.5);   // Must be in [0, 1]
new PopulationBasedTrainer(space, perturbFactor: -0.1);    // Must be >= 0
```

### Perturbation behavior

| Parameter type | Perturbation |
|---|---|
| `FloatRange` | Multiplicative: `value * (1 ± perturbFactor)`, clamped to `[Low, High]` |
| `IntRange` | Same, rounded and clamped to `[Low, High]` |
| `CategoricalRange` | 50% keep current value, 50% resample uniformly |

---

## Samplers — Which One to Use

| Sampler | Best For | Dimensions | Overhead |
|---------|----------|------------|----------|
| **TPE** (default) | General purpose, mixed parameter types | 1–100 | Low |
| **CMA-ES** | High-dimensional continuous problems | 10–1000+ | Medium |
| **Random** | Baselines, sanity checks, embarrassingly parallel | Any | None |

### TPE (Tree-structured Parzen Estimator) — The Default

Best general-purpose sampler. Learns from trial history to suggest better parameters over time.

```csharp
// Default configuration (works well for most problems)
using var study = Optimizer.CreateStudy("my_study", space);

// Custom configuration
using var study = Optimizer.CreateStudy("my_study", space, config: new TpeSamplerConfig
{
    NStartupTrials = 20,     // More random exploration before TPE kicks in
    NEiCandidates = 48,      // More candidates = better suggestions, slightly slower
    Seed = 42                // Reproducible results
});
```

**How it works:** Splits completed trials into "good" (best 10%) and "bad" (rest). Builds kernel density estimates (mixture of truncated Gaussians) for each group. Suggests parameters that maximize the Expected Improvement ratio l(x)/g(x). Based on [Bergstra et al., "Algorithms for Hyper-Parameter Optimization" (NeurIPS 2011)](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html), with bandwidth selection and magic clip following the Optuna implementation.

**When to use:** Start here. It handles mixed types (float + int + categorical), scales well to ~100 dimensions, and converges fast with low overhead.

### CMA-ES (Covariance Matrix Adaptation)

An evolution strategy that learns parameter correlations. Excels at high-dimensional continuous optimization.

```csharp
using var study = Optimizer.CreateStudyWithCmaEs("cma_study", space, config: new CmaEsSamplerConfig
{
    PopulationSize = 30,    // Null = auto: 4 + floor(3 * ln(n))
    InitialSigma = 0.3,    // Step size as fraction of range (0.0–1.0)
    Backend = ComputeBackend.Auto  // Use GPU if available
});
```

**How it works:** Maintains a multivariate Gaussian distribution. Each generation, samples a population of candidates, evaluates them, and updates the distribution (mean, covariance, step size) to move toward better regions. The covariance matrix learns which parameter combinations work together. Implements the (mu/mu_w, lambda)-CMA-ES variant from [Hansen & Ostermeier, "Completely Derandomized Self-Adaptation in Evolution Strategies" (Evolutionary Computation, 2001)](https://doi.org/10.1162/106365601750190398), including cumulative step-size adaptation (CSA) and rank-mu updates.

**When to use:** High-dimensional continuous problems (>20 params), problems where parameter correlations matter, situations where you can evaluate a full population per generation.

### Random

Pure random sampling. No learning, no overhead.

```csharp
using var study = Optimizer.CreateStudyWithRandomSampler("baseline", space, seed: 42);
```

**When to use:** Establishing baselines, verifying your pipeline works, problems where the search space is small enough that random search is competitive, or when you want maximum parallelism with zero coordination overhead.

---

## GPU Acceleration (NVIDIA CUDA)

CMA-ES can offload its two heaviest operations to your GPU:
- **Population sampling** — batched matrix-vector multiplications
- **Covariance update** — batched outer products

### Requirements

- NVIDIA GPU with CUDA support
- NVIDIA drivers installed (no CUDA SDK needed — ILGPU compiles at runtime)
- Dimensions >= 100 recommended (GPU kernel launch overhead dominates at smaller sizes)

### Usage

```csharp
// Explicit GPU (throws if no CUDA device)
var config = new CmaEsSamplerConfig { Backend = ComputeBackend.Gpu };

// Auto-detect (uses GPU if available, falls back to CPU)
var config = new CmaEsSamplerConfig { Backend = ComputeBackend.Auto };

// CPU only (default)
var config = new CmaEsSamplerConfig { Backend = ComputeBackend.Cpu };

using var study = Optimizer.CreateStudyWithCmaEs("gpu_opt", space, config: config);
```

### Performance (i9-13900 + RTX 3090)

| Dimensions | CPU (ms/gen) | GPU (ms/gen) | Speedup |
|------------|-------------|-------------|---------|
| 10 | 0.08 | 1.2 | 0.07x (CPU wins) |
| 20 | 0.15 | 0.10 | 1.5x |
| 50 | 0.9 | 0.7 | 1.3x |
| 100 | 3.2 | 1.9 | 1.7x |
| 200 | 12.0 | 8.2 | 1.5x |

**Key insight:** Even modest GPU speedups are valuable when CPU cores are needed elsewhere. If you're running optimization alongside other workloads (e.g., parallel simulation workers), offloading CMA-ES to the GPU frees CPU cores for the work that needs them.

### GPU Dimension Warning

The library automatically warns when GPU is active but dimensions are below the recommended minimum:

```
GPU compute active (NVIDIA GeForce RTX 3090) but dimensions (10) < recommended minimum (100).
CPU is likely faster for this problem size.
```

Check this via `CmaEsSampler.GpuDimensionWarning`.

---

## Use Cases

### 1. Machine Learning — Hyperparameter Tuning

The classic use case. Tune learning rate, architecture, and training parameters.

```csharp
var space = new SearchSpace([
    new FloatRange("learning_rate", 1e-5, 1e-1, Log: true),
    new FloatRange("weight_decay", 1e-6, 1e-2, Log: true),
    new IntRange("batch_size", 16, 256, Step: 16),
    new IntRange("num_layers", 2, 8),
    new IntRange("hidden_size", 64, 512, Step: 64),
    new FloatRange("dropout", 0.0, 0.5),
    new CategoricalRange("activation", ["relu", "gelu", "silu"]),
    new CategoricalRange("optimizer", ["adam", "adamw", "sgd"])
]);

using var study = Optimizer.CreateStudy("neural_net_tuning", space, StudyDirection.Minimize);

for (int i = 0; i < 200; i++)
{
    var trial = study.Ask();

    var config = new TrainingConfig
    {
        LearningRate = (double)trial.Parameters["learning_rate"],
        WeightDecay = (double)trial.Parameters["weight_decay"],
        BatchSize = (int)trial.Parameters["batch_size"],
        NumLayers = (int)trial.Parameters["num_layers"],
        HiddenSize = (int)trial.Parameters["hidden_size"],
        Dropout = (double)trial.Parameters["dropout"],
        Activation = (string)trial.Parameters["activation"],
        Optimizer = (string)trial.Parameters["optimizer"]
    };

    try
    {
        var validationLoss = TrainModel(config);
        study.Tell(trial.Number, validationLoss);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Trial {trial.Number} failed: {ex.Message}");
        study.Tell(trial.Number, TrialState.Fail);
    }
}

var best = study.BestTrial!;
Console.WriteLine($"Best validation loss: {best.Value:F6}");
```

### 2. Game Balance & AI Tuning

Tune game parameters — enemy difficulty, economy rates, AI behavior weights.

```csharp
var space = new SearchSpace([
    // Economy balance
    new FloatRange("gold_per_kill", 5.0, 50.0),
    new FloatRange("xp_curve_exponent", 1.2, 3.0),
    new IntRange("shop_reroll_cost", 1, 5),

    // AI behavior weights
    new FloatRange("aggression", 0.0, 1.0),
    new FloatRange("retreat_threshold", 0.1, 0.5),
    new FloatRange("teamfight_priority", 0.0, 1.0),
    new FloatRange("objective_priority", 0.0, 1.0),

    // Difficulty scaling
    new FloatRange("damage_multiplier", 0.5, 2.0),
    new FloatRange("health_multiplier", 0.5, 2.0),
    new CategoricalRange("targeting_strategy", ["nearest", "lowest_hp", "highest_threat", "random"])
]);

using var study = Optimizer.CreateStudy("game_balance", space, StudyDirection.Maximize);

for (int i = 0; i < 300; i++)
{
    var trial = study.Ask();

    // Simulate 100 games with these settings
    var results = SimulateGames(trial.Parameters, numGames: 100);

    // Score: we want close, exciting games
    var score = results.AverageEngagement
              - Math.Abs(results.WinRate - 0.5) * 10   // Penalize imbalance
              - results.AverageGameLength / 60.0;        // Penalize long games

    study.Tell(trial.Number, score);
}
```

### 3. Infrastructure & Cloud Cost Optimization

Find the cheapest cloud configuration that meets performance SLAs.

```csharp
var space = new SearchSpace([
    new CategoricalRange("instance_type", ["t3.medium", "t3.large", "m5.large", "m5.xlarge", "c5.large"]),
    new IntRange("replica_count", 2, 20),
    new IntRange("cpu_limit_millicores", 500, 4000, Step: 500),
    new IntRange("memory_limit_mb", 512, 8192, Step: 512),
    new IntRange("connection_pool_size", 10, 200, Step: 10),
    new FloatRange("cache_ttl_seconds", 5.0, 300.0),
    new CategoricalRange("cache_strategy", ["lru", "lfu", "fifo"]),
    new IntRange("batch_size", 10, 500, Step: 10)
]);

using var study = Optimizer.CreateStudy("infra_cost", space, StudyDirection.Minimize);

for (int i = 0; i < 100; i++)
{
    var trial = study.Ask();

    var config = MapToInfraConfig(trial.Parameters);
    var result = RunLoadTest(config, duration: TimeSpan.FromMinutes(5));

    if (result.P99Latency > TimeSpan.FromMilliseconds(200) || result.ErrorRate > 0.01)
    {
        // Doesn't meet SLA — mark as failed
        study.Tell(trial.Number, TrialState.Fail);
        continue;
    }

    // Minimize monthly cost for configs that meet the SLA
    var monthlyCost = CalculateMonthlyCost(config);
    study.Tell(trial.Number, monthlyCost);
}
```

### 4. Signal Processing & Filtering

Optimize filter parameters for noisy sensor data.

```csharp
var space = new SearchSpace([
    new CategoricalRange("filter_type", ["kalman", "butterworth", "savitzky_golay", "median"]),
    new IntRange("window_size", 3, 51, Step: 2),         // Odd numbers for symmetric windows
    new FloatRange("cutoff_frequency", 0.01, 0.5),
    new IntRange("polynomial_order", 1, 5),
    new FloatRange("process_noise", 1e-5, 1e-1, Log: true),
    new FloatRange("measurement_noise", 1e-4, 1.0, Log: true)
]);

using var study = Optimizer.CreateStudy("filter_opt", space, StudyDirection.Minimize);

for (int i = 0; i < 150; i++)
{
    var trial = study.Ask();

    var filtered = ApplyFilter(noisySignal, trial.Parameters);

    // Minimize: tracking error + smoothness penalty
    var trackingError = MeanSquaredError(filtered, groundTruth);
    var roughness = SecondDerivativeEnergy(filtered);
    var score = trackingError + 0.1 * roughness;

    study.Tell(trial.Number, score);
}
```

### 5. Compiler / Build Optimization

Find the best compiler flags and build settings for your performance-critical code.

```csharp
var space = new SearchSpace([
    new CategoricalRange("optimization_level", ["O1", "O2", "O3", "Os"]),
    new CategoricalRange("lto", ["none", "thin", "full"]),
    new CategoricalRange("target_cpu", ["generic", "native", "haswell", "skylake", "znver3"]),
    new IntRange("inline_threshold", 100, 1000, Step: 50),
    new CategoricalRange("vectorize", [true, false]),
    new CategoricalRange("unroll_loops", [true, false]),
    new IntRange("parallel_jobs", 1, 16)
]);

using var study = Optimizer.CreateStudy("build_opt", space, StudyDirection.Minimize);

for (int i = 0; i < 80; i++)
{
    var trial = study.Ask();

    var buildFlags = GenerateFlags(trial.Parameters);

    try
    {
        var buildTime = CompileProject(buildFlags);
        var benchResult = RunBenchmarkSuite();

        // Balance: fast runtime with acceptable build time
        var score = benchResult.MedianLatencyMs + buildTime.TotalSeconds * 0.1;
        study.Tell(trial.Number, score);
    }
    catch
    {
        study.Tell(trial.Number, TrialState.Fail);  // Invalid flag combination
    }
}
```

### 6. Robotics & Control Systems — PID Tuning

Tune PID controller gains for optimal response characteristics.

```csharp
var space = new SearchSpace([
    new FloatRange("kp", 0.1, 100.0, Log: true),     // Proportional gain
    new FloatRange("ki", 0.001, 50.0, Log: true),     // Integral gain
    new FloatRange("kd", 0.001, 20.0, Log: true),     // Derivative gain
    new FloatRange("filter_coefficient", 0.01, 1.0),   // Derivative filter
    new IntRange("control_frequency_hz", 50, 1000, Step: 50)
]);

// CMA-ES is excellent for continuous control problems
using var study = Optimizer.CreateStudyWithCmaEs("pid_tuning", space, StudyDirection.Minimize,
    config: new CmaEsSamplerConfig { InitialSigma = 0.4 });

for (int i = 0; i < 200; i++)
{
    var trial = study.Ask();

    var pid = new PidController(
        kp: (double)trial.Parameters["kp"],
        ki: (double)trial.Parameters["ki"],
        kd: (double)trial.Parameters["kd"],
        filterCoeff: (double)trial.Parameters["filter_coefficient"]
    );

    var sim = SimulateStepResponse(pid, duration: 10.0);

    // ITAE criterion: penalizes slow convergence heavily
    var itae = sim.TimePoints.Zip(sim.Errors)
        .Sum(pair => pair.First * Math.Abs(pair.Second));

    // Penalize overshoot and oscillation
    var overshoot = Math.Max(0, sim.MaxValue - sim.SetPoint) / sim.SetPoint;
    var score = itae + overshoot * 100 + sim.SettlingTime * 10;

    study.Tell(trial.Number, score);
}
```

### 7. A/B Testing Configuration

Optimize multiple experiment parameters simultaneously instead of testing one at a time.

```csharp
var space = new SearchSpace([
    new FloatRange("button_size_px", 32, 80),
    new CategoricalRange("button_color", ["#FF4444", "#44AA44", "#4444FF", "#FF8800"]),
    new CategoricalRange("cta_text", ["Buy Now", "Add to Cart", "Get It", "Order"]),
    new FloatRange("price_font_size", 14.0, 28.0),
    new CategoricalRange("layout", ["single_column", "grid_2x2", "carousel"]),
    new IntRange("items_per_page", 10, 50, Step: 5),
    new FloatRange("discount_badge_opacity", 0.5, 1.0)
]);

using var study = Optimizer.CreateStudy("landing_page", space, StudyDirection.Maximize);

// Each trial = deploy variant, collect data for N hours, measure conversion
for (int i = 0; i < 50; i++)
{
    var trial = study.Ask();

    var variant = DeployVariant(trial.Parameters);
    var metrics = await CollectMetrics(variant, duration: TimeSpan.FromHours(4));

    var conversionRate = metrics.Purchases / (double)metrics.Visitors;
    study.Tell(trial.Number, conversionRate);
}
```

### 8. Database Query Optimization

Tune database configuration for your specific workload.

```csharp
var space = new SearchSpace([
    new IntRange("shared_buffers_mb", 256, 8192, Step: 256),
    new IntRange("work_mem_mb", 4, 256, Step: 4),
    new IntRange("effective_cache_size_mb", 1024, 32768, Step: 1024),
    new FloatRange("random_page_cost", 1.0, 4.0),
    new IntRange("max_connections", 50, 500, Step: 50),
    new IntRange("max_parallel_workers", 1, 16),
    new CategoricalRange("wal_level", ["minimal", "replica", "logical"]),
    new IntRange("checkpoint_completion_target_pct", 50, 95, Step: 5)
]);

using var study = Optimizer.CreateStudy("postgres_tuning", space, StudyDirection.Minimize);

for (int i = 0; i < 60; i++)
{
    var trial = study.Ask();

    ApplyPostgresConfig(trial.Parameters);
    RestartPostgres();

    var benchmark = RunPgBench(duration: TimeSpan.FromMinutes(2));

    // Minimize: latency with throughput constraint
    if (benchmark.TransactionsPerSecond < 1000)
    {
        study.Tell(trial.Number, TrialState.Fail);  // Below throughput minimum
        continue;
    }

    study.Tell(trial.Number, benchmark.P95LatencyMs);
}
```

### 9. Image Processing Pipeline

Optimize image processing parameters for quality vs. speed.

```csharp
var space = new SearchSpace([
    new FloatRange("denoise_strength", 0.0, 1.0),
    new IntRange("denoise_template_size", 3, 11, Step: 2),
    new FloatRange("sharpen_amount", 0.0, 3.0),
    new FloatRange("sharpen_radius", 0.5, 5.0),
    new FloatRange("contrast_gamma", 0.5, 2.0),
    new FloatRange("saturation", 0.5, 1.5),
    new IntRange("jpeg_quality", 60, 98),
    new CategoricalRange("resize_algorithm", ["bilinear", "bicubic", "lanczos"])
]);

using var study = Optimizer.CreateStudy("image_pipeline", space, StudyDirection.Maximize);

for (int i = 0; i < 200; i++)
{
    var trial = study.Ask();

    var processed = ProcessImage(referenceImage, trial.Parameters);

    var ssim = ComputeSSIM(processed, groundTruth);    // Structural similarity
    var fileSize = GetFileSize(processed);
    var processingTime = MeasureLatency(trial.Parameters);

    // Composite: quality vs. size vs. speed
    var score = ssim
              - (fileSize / 1_000_000.0) * 0.1          // Penalize large files
              - processingTime.TotalMilliseconds * 0.001; // Penalize slow processing

    study.Tell(trial.Number, score);
}
```

### 10. Distributed / Parallel Optimization

Run evaluations across multiple workers using the batch API.

```csharp
var space = new SearchSpace([
    new FloatRange("param_a", -5.0, 5.0),
    new FloatRange("param_b", -5.0, 5.0),
    new FloatRange("param_c", -5.0, 5.0)
]);

using var study = Optimizer.CreateStudy("distributed", space, config: new TpeSamplerConfig
{
    ConstantLiar = true  // Prevents duplicate suggestions for concurrent trials
});

var workerCount = Environment.ProcessorCount;

for (int wave = 0; wave < 20; wave++)
{
    // Ask for a batch of trials
    var batch = study.AskBatch(workerCount);

    // Evaluate all trials in parallel
    var results = batch
        .AsParallel()
        .WithDegreeOfParallelism(workerCount)
        .Select(trial =>
        {
            try
            {
                var value = ExpensiveSimulation(trial.Parameters);
                return new TrialResult(trial.Number, value, TrialState.Complete);
            }
            catch
            {
                return new TrialResult(trial.Number, null, TrialState.Fail);
            }
        })
        .ToList();

    // Report all results
    study.TellBatch(results);

    Console.WriteLine($"Wave {wave}: Best so far = {study.BestTrial?.Value:F6}");
}
```

### 11. Simulation Parameter Calibration

Calibrate a physics simulation to match real-world observations.

```csharp
var space = new SearchSpace([
    new FloatRange("friction_coefficient", 0.01, 1.0),
    new FloatRange("restitution", 0.0, 1.0),
    new FloatRange("air_resistance", 0.0, 0.1),
    new FloatRange("spring_constant", 100.0, 10000.0, Log: true),
    new FloatRange("damping_ratio", 0.01, 1.0),
    new FloatRange("mass_scaling", 0.8, 1.2),
    new IntRange("solver_iterations", 5, 50)
]);

// CMA-ES excels at finding correlated parameter combinations
using var study = Optimizer.CreateStudyWithCmaEs("sim_calibration", space, StudyDirection.Minimize,
    config: new CmaEsSamplerConfig { InitialSigma = 0.5 });

for (int i = 0; i < 300; i++)
{
    var trial = study.Ask();

    var simulated = RunSimulation(trial.Parameters);
    var error = MeanSquaredError(simulated.Trajectory, realWorldData.Trajectory);

    study.Tell(trial.Number, error);
}

Console.WriteLine($"Calibration error: {study.BestTrial!.Value:E2}");
```

### 12. Feature Engineering Selection

Automatically find the best feature engineering pipeline.

```csharp
var space = new SearchSpace([
    new CategoricalRange("scaler", ["standard", "minmax", "robust", "none"]),
    new CategoricalRange("use_pca", [true, false]),
    new IntRange("pca_components", 5, 50),
    new CategoricalRange("interaction_features", [true, false]),
    new CategoricalRange("polynomial_degree", [1, 2, 3]),
    new FloatRange("feature_selection_threshold", 0.01, 0.5),
    new CategoricalRange("imputation", ["mean", "median", "knn", "iterative"]),
    new IntRange("knn_neighbors", 3, 15)
]);

using var study = Optimizer.CreateStudy("feature_eng", space, StudyDirection.Maximize);

for (int i = 0; i < 100; i++)
{
    var trial = study.Ask();

    var pipeline = BuildPipeline(trial.Parameters);
    var features = pipeline.FitTransform(trainingData);
    var cvScore = CrossValidate(model, features, labels, folds: 5);

    study.Tell(trial.Number, cvScore.Mean);
}
```

### 13. Network Protocol Tuning

Optimize TCP/network parameters for throughput or latency.

```csharp
var space = new SearchSpace([
    new IntRange("tcp_window_size_kb", 16, 4096, Step: 16),
    new IntRange("send_buffer_kb", 32, 2048, Step: 32),
    new IntRange("receive_buffer_kb", 32, 2048, Step: 32),
    new IntRange("max_concurrent_streams", 1, 256),
    new FloatRange("retry_backoff_base", 1.0, 5.0),
    new IntRange("max_retries", 1, 10),
    new IntRange("timeout_ms", 100, 30000),
    new CategoricalRange("congestion_algorithm", ["cubic", "bbr", "reno"])
]);

using var study = Optimizer.CreateStudy("network_opt", space, StudyDirection.Maximize);

for (int i = 0; i < 80; i++)
{
    var trial = study.Ask();

    ApplyNetworkConfig(trial.Parameters);
    var metrics = RunNetworkBenchmark(targetServer, duration: TimeSpan.FromSeconds(30));

    // Maximize goodput while keeping latency acceptable
    var score = metrics.GoodputMbps;
    if (metrics.P99LatencyMs > 100)
        score *= 0.5;  // Heavy penalty for high latency

    study.Tell(trial.Number, score);
}
```

### 14. High-Dimensional with GPU (CMA-ES Sweet Spot)

For problems with 100+ continuous parameters, CMA-ES with GPU shines.

```csharp
// Large continuous search space (200 dimensions)
var ranges = Enumerable.Range(0, 200)
    .Select(i => (ParameterRange)new FloatRange($"w_{i}", -1.0, 1.0))
    .ToList();
var space = new SearchSpace(ranges);

using var study = Optimizer.CreateStudyWithCmaEs("high_dim", space, StudyDirection.Minimize,
    config: new CmaEsSamplerConfig
    {
        Backend = ComputeBackend.Auto,   // GPU if available
        PopulationSize = 50,
        InitialSigma = 0.3
    });

// Check GPU status
var sampler = (CmaEsSampler)study.GetType()
    .GetField("_sampler", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!
    .GetValue(study)!;

for (int i = 0; i < 2000; i++)
{
    var trial = study.Ask();
    var weights = Enumerable.Range(0, 200)
        .Select(j => (double)trial.Parameters[$"w_{j}"])
        .ToArray();

    var loss = EvaluateNeuralNetWeights(weights);
    study.Tell(trial.Number, loss);
}

if (sampler.IsGpuActive)
    Console.WriteLine($"Ran on {sampler.DeviceName}");
```

---

## Sampler Selection Guide

```
Start here:
    |
    v
Mixed types (float + int + categorical)?
    |                   |
   YES                  NO (all continuous)
    |                   |
    v                   v
  Use TPE           How many dimensions?
                        |
              +---------+---------+
              |         |         |
            < 20     20–100     > 100
              |         |         |
              v         v         v
         TPE or      TPE       CMA-ES
         CMA-ES    (lower     (learns
         (both      overhead)  correlations)
          work)                   |
                                  v
                           GPU available?
                              |        |
                             YES       NO
                              |        |
                              v        v
                           CMA-ES    CMA-ES
                           + Auto    + Cpu
```

---

## Observability

### CMA-ES Metrics

Monitor the internal state of CMA-ES during optimization:

```csharp
var sampler = new CmaEsSampler(new CmaEsSamplerConfig());
using var study = Optimizer.CreateStudy("monitored", space, sampler);

// After some trials...
if (sampler.Metrics is { } m)
{
    Console.WriteLine($"Generation:       {m.Generation}");
    Console.WriteLine($"Step size (σ):    {m.Sigma:E2}");
    Console.WriteLine($"Condition number: {m.ConditionNumber:F1}");
    Console.WriteLine($"Best value:       {m.BestValue:F6}");
    Console.WriteLine($"Evaluated:        {m.EvaluatedTrials}");
}
```

### GPU Status

```csharp
if (sampler.IsGpuActive)
{
    Console.WriteLine($"GPU device: {sampler.DeviceName}");
    if (sampler.GpuDimensionWarning is { } warning)
        Console.WriteLine($"Warning: {warning}");
}
```

---

## Determinism & Thread Safety

### Reproducible Runs

All samplers accept an optional `Seed` parameter. When set, the same seed + same search space + same evaluation order produces identical trial sequences:

```csharp
// Deterministic — same results every time
var config = new TpeSamplerConfig { Seed = 42 };
using var study = Optimizer.CreateStudy("reproducible", space, config: config);
```

**Caveat:** Parallel evaluation (e.g., `AskBatch` + `AsParallel`) introduces non-deterministic `Tell` ordering, which affects subsequent suggestions even with a fixed seed. For fully reproducible runs, evaluate sequentially.

### Thread Safety

Each `Study` instance is thread-safe. All public methods (`Ask`, `Tell`, `AskBatch`, `TellBatch`, `BestTrial`, `Trials`) acquire an internal lock. You can safely call `Ask` and `Tell` from different threads on the same study.

**Scope:** Thread safety is per-study. Different `Study` instances are fully independent and can run concurrently without any shared state.

**What is NOT thread-safe:** Individual `ISampler` instances. If you create a sampler manually and share it across multiple studies, you must synchronize access yourself. The `Optimizer.Create*` factory methods create one sampler per study, so this is only relevant for custom setups.

---

## API Reference

### Search Space Types

| Type | Parameters | Description |
|------|-----------|-------------|
| `FloatRange` | `Name, Low, High, Log` | Continuous real-valued parameter |
| `IntRange` | `Name, Low, High, Step` | Discrete integer parameter |
| `CategoricalRange` | `Name, Choices` | One-of-N selection |

### Study Methods

| Method | Description |
|--------|-------------|
| `Ask()` | Get next suggested trial |
| `Tell(number, value)` | Report successful evaluation |
| `Tell(number, double[])` | Report multi-objective values |
| `Tell(number, TrialState.Fail)` | Report failed evaluation |
| `Tell(number, TrialState.Pruned)` | Report pruned trial |
| `AskBatch(count)` | Get multiple trials at once |
| `TellBatch(results)` | Report multiple results at once |
| `TellBatch(MoTrialResult[])` | Report multi-objective batch results |
| `BestTrial` | Best completed trial (null if none) |
| `ParetoFront` | Non-dominated trials (multi-objective studies) |
| `IsMultiObjective` | True if study has multiple objectives |
| `Direction` | Primary study direction (first for multi-objective) |
| `ShouldPrune(trial)` | Ask configured pruner if trial should stop |
| `SetConstraintFunc(fn)` | Register constraint evaluation function |
| `IsFeasible(trial)` | True if all constraint values ≤ 0 |
| `Save(filePath)` | Serialize study to JSON |
| `Study.Load(path, space, sampler)` | Restore study from JSON |
| `Trials` | All trial history |
| `Dispose()` | Clean up resources (GPU buffers, etc.) |

### Factory Methods (Optimizer class)

| Method | Notes |
|--------|-------|
| `CreateStudy(name, space)` | TPE (default) |
| `CreateStudy(name, space, direction)` | TPE with Maximize/Minimize |
| `CreateStudy(name, space, StudyDirection[])` | Multi-objective |
| `CreateStudy(name, space, config: tpeConfig)` | TPE with custom config |
| `CreateStudy(name, space, pruner: pruner)` | With early stopping |
| `CreateStudy(name, space, warmStartTrials: trials)` | Warm start from trial list |
| `CreateStudy(name, space, fromStudy: study)` | Warm start from existing study |
| `CreateStudyWithCmaEs(name, space, config: cmaConfig)` | CMA-ES |
| `CreateStudyWithCmaEs(name, space, warmStartTrials:)` | CMA-ES + warm start |
| `CreateStudyWithRandomSampler(name, space, seed: 42)` | Random |
| `CreateStudyWithRandomSampler(name, space, warmStartTrials:)` | Random + warm start |
| `CreateStudy(name, space, customSampler)` | Any ISampler |

### TpeSamplerConfig

| Property | Default | Description |
|----------|---------|-------------|
| `NStartupTrials` | 10 | Random trials before TPE |
| `NEiCandidates` | 24 | EI candidates per parameter |
| `PriorWeight` | 1.0 | Uniform prior weight |
| `ConstantLiar` | true | Deduplicate concurrent trials |
| `ConsiderMagicClip` | true | Enforce minimum bandwidth |
| `MaxAboveTrials` | 200 | Cap "bad" group size (0 = no limit) |
| `Seed` | null | Random seed |

### CmaEsSamplerConfig

| Property | Default | Description |
|----------|---------|-------------|
| `PopulationSize` | null (auto) | Lambda. Auto = `4 + floor(3 * ln(n))` |
| `InitialSigma` | 0.3 | Step size as fraction of range |
| `Seed` | null | Random seed |
| `Backend` | `Cpu` | `Cpu`, `Gpu`, or `Auto` |

### ComputeBackend

| Value | Behavior |
|-------|----------|
| `Cpu` | CPU-only (default, safest) |
| `Gpu` | CUDA required, throws if unavailable |
| `Auto` | Try CUDA, fall back to CPU silently |

---

## Compared to Optuna (Python)

| Feature | OptiSharp | Optuna |
|---------|-------------|--------|
| Language | C# / .NET Standard 2.1+ | Python |
| TPE | Yes | Yes |
| CMA-ES | Yes (with GPU) | Yes (no GPU) |
| Random | Yes | Yes |
| GPU Acceleration | CUDA via ILGPU | No |
| Thread-safe | Yes (built-in) | Partial |
| Batch API | Yes | Yes (v3+) |
| Storage Backend | JSON (Save/Load) + in-memory | SQLite, PostgreSQL, etc. |
| Pruning | Yes (Median, Percentile, SHA) | Yes |
| Multi-objective | Yes (Pareto front + crowding) | Yes |
| Constraints | Yes (per-trial function) | Yes |
| Warm Start | Yes (from trials or study) | Yes |
| Population-Based Training | Yes (PopulationBasedTrainer) | No |
| Visualization | No | Yes (plotly) |

**What Optuna has that this doesn't:** Built-in visualization, dozens of samplers, SQLite/PostgreSQL storage backends.

**What this has that Optuna doesn't:** Native .NET, zero Python interop, GPU-accelerated CMA-ES, Population-Based Training, single-file deployment.

---

## Current Limitations

Things OptiSharp does **not** support:

- **No conditional parameters** — All parameters are sampled independently. You can't express "only sample `dropout` if `use_dropout` is true."
- **No parameter constraints at sampling time** — Constraints are evaluated after sampling. The sampler learns to avoid infeasible regions over time, but cannot enforce them structurally.
- **No built-in visualization** — No trial plots, parameter importance, or optimization history charts.
- **CMA-ES handles categoricals randomly** — CMA-ES only optimizes continuous parameters. Categorical parameters in a CMA-ES study are sampled uniformly at random, not learned.
- **JSON storage only** — No SQLite, PostgreSQL, or other database backends.

---

## Validation & Benchmarks

The test suite covers correctness, convergence, and performance across **243+ tests**.

### Convergence

Both TPE and CMA-ES consistently beat random search on standard benchmark functions:

| Function | Dims | Trials | TPE vs Random | CMA-ES vs Random |
|----------|------|--------|---------------|------------------|
| Quadratic | 1D | 100 | >= 7/10 wins | >= 6/10 wins |
| Sphere | 5D | 200 | >= 5/8 wins | >= 5/8 wins |
| Rosenbrock | 2D | 200 | >= 5/8 wins | >= 5/8 wins |
| Mixed space | 4D | 150 | >= 5/8 wins | >= 4/8 wins |

TPE also converges on a 62-parameter space (500 trials, best < 90% of mean) and log-scale parameters (finds optimum near 0.001 in [0.0001, 1.0]). CMA-ES converges on 20-parameter spaces (best < 50% of mean).

### Ask / Tell Latency (62-parameter mixed space)

| Scenario | Threshold |
|----------|-----------|
| Ask — startup/random phase (1000 calls) | < 0.1 ms |
| Ask — TPE active, 100 prior trials | < 5 ms |
| Ask — TPE active, 500 prior trials | < 20 ms |
| AskBatch(40) — 500 prior trials | < 500 ms total |
| Tell — single call | < 0.05 ms |
| TellBatch(200) | < 1 ms total |

### Memory & Load

| Scenario | Threshold |
|----------|-----------|
| 2000 sequential trials, 62 params — heap growth | < 50 MB |
| 2000 sequential trials — wall time | < 30 seconds |
| 4 concurrent threads × 125 trials | < 30 seconds, 0 exceptions |
| Memory ratio (2000 trials / 100 trials) | < 5× |

---

## Running Tests

```bash
# Run all tests (slow tests are skipped by default)
dotnet test

# Run everything, including slow tests
RUN_SLOW_TESTS=1 dotnet test
```

Load tests (`LoadTests`, `PerformanceBenchmarkTests`) use `[SlowFact]` instead of `[Fact]`. They are skipped by default everywhere — CLI, VS Code Test Explorer, CI. Set `RUN_SLOW_TESTS=1` to opt in. These tests run thousands of trials and can take minutes.

---

## License

MIT — Use it however you want.
