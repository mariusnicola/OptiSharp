# OptiSharp vs Optuna Benchmark Comparison Guide

## Quick Start

### Run Benchmarks
```bash
cd e:\OptimizationCore
bash OptiSharp.Benchmarks/run_comparison.sh fast    # ~8-10 hours
bash OptiSharp.Benchmarks/run_comparison.sh all     # ~24+ hours (includes extended tier)
```

### View Results
```bash
cat benchmark-results/comparison_report.md
```

**Stop anytime:** Press `Ctrl+C` — results saved, resumable on next run.

---

## What This Does

Runs identical benchmarks on **OptiSharp** (C#) and **Optuna** (Python):
- Same 4 objective functions (Sphere, Rosenbrock, Rastrigin, Ackley)
- Same parameter ranges, seeds, trial budgets
- Comparisons: **TPE**, **Random**, **CMA-ES** samplers
- Pruning: **None**, **MedianPruner**, **SuccessiveHalving**
- Sizes: **10, 50, 100, 200** parameters
- Budgets: **100-1000** trials (fast) or **10k-100k** (extended)

---

## Current Progress

**Completed: 41/41 pairs** (TPE sampler, 10 params, all trial counts)

### Key Findings

#### 🚀 Throughput Winner: OptiSharp
OptiSharp is **4-6x faster**:
- Sphere/10p/100t: 431 trials/sec vs 135 (3.2x)
- Rosenbrock/10p/500t: 289 trials/sec vs 74 (3.9x)
- Sphere/10p/1000t: 312 trials/sec vs 51 (6.1x)

#### 🎯 Quality: Mixed Results
- **Easy objectives (Sphere)**: Optuna converges better at 500+ trials
- **Hard objectives (Rosenbrock)**: OptiSharp better at 100 trials, mixed at 300+
- **Ackley**: OptiSharp dominates (7/9 wins)
- **Pattern**: OptiSharp excels early, Optuna catches up later

#### 🚨 Pruning Bug
- OptiSharp: **0% pruning rate** across all runs
- Optuna: **44-95% pruning rates**
- **Issue**: OptiSharp's pruning logic not activating (investigate!)

---

## Remaining Work

| Phase | Coverage | Est. Time |
|---|---|---|
| TPE (done) | 10p all trials × 3 pruning | ✅ Complete |
| Random | 50, 100, 200p × 100-1000 trials × 3 pruning | 4-5h |
| CMA-ES | 50, 100, 200p × 100-1000 trials × 3 pruning | 5-6h |
| Extended | 10k-100k trials × 2 samplers × 4p × 3 pruning | 8-12h |
| **Total** | 576 pairs (fast) + 96 pairs (extended) | ~24h |

---

## Architecture

### Files
```
OptiSharp.Benchmarks/
├── Objectives.cs              # C# objective functions
├── MatrixBenchmarkRunner.cs   # C# benchmark executor
├── Program.cs                 # CLI entry point (--matrix mode)
├── python/
│   ├── objectives.py          # Python objectives (identical math)
│   ├── optuna_matrix.py       # Optuna runner
│   ├── generate_report.py     # Report generator
└── run_comparison.sh          # Master orchestration script

.temp/
└── runs/                       # JSON results (auto-generated)

benchmark-results/
└── comparison_report.md        # Main report (incrementally updated)
```

### Flow
1. `run_comparison.sh` loops through matrix configurations
2. Runs C# benchmark: `.temp/bin/OptiSharp.Benchmarks --matrix ...`
3. Runs Python benchmark: `python optuna_matrix.py ...`
4. Appends results to `benchmark-results/comparison_report.md`
5. Progress shown in real-time with ETA

### CLI Format
```bash
.temp/bin/OptiSharp.Benchmarks --matrix \
  --sampler tpe|random|cmaes \
  --params 10|50|100|200 \
  --trials N \
  --objective sphere|rosenbrock|rastrigin|ackley \
  --pruner none|median|sha \
  --tier fast|extended \
  --output path.json
```

---

## Metrics Captured

Per run:
- `best_value`: Final best value found
- `elapsed_ms`: Wall clock time
- `trials_per_second`: Throughput
- `pruned_trials` / `pruning_rate`: Pruning effectiveness
- `convergence`: Best value at 20%, 40%, 60%, 80%, 100% of trials

---

## Resume on Another PC

1. Clone/pull latest code (commit: `bfa3e9a`)
2. Install dependencies:
   ```bash
   pip install optuna numpy
   ```
3. Run same command:
   ```bash
   bash OptiSharp.Benchmarks/run_comparison.sh fast
   ```
4. Script auto-detects completed pairs and continues

---

## Investigation Items

### 1. OptiSharp Pruning Bug
- **Issue**: Pruning rate = 0% for all configurations
- **Expected**: 40-95% pruning rates like Optuna
- **Location**: `MatrixBenchmarkRunner.cs` lines 197-210 (intermediate value reporting & pruning check)
- **Action**: Debug `study.ShouldPrune()` logic

### 2. Quality Variance at High Trials
- Optuna converges better at 500-1000 trials on simple objectives
- OptiSharp dominates early (100 trials) but trails later
- Investigate: TPE hyperparameters, startup trials, bandwidth calculation

### 3. CMA-ES on 10 Params
- CMA-ES recommended for 100+ dimensions
- May underperform at 10 params
- Monitor whether to include in comparison

---

## Report Structure

Generated report includes:
1. **Summary tables**: Quality (best value) and throughput winners per config
2. **Per-objective sections**: Convergence curves at checkpoints
3. **Pruning effectiveness**: Pruning rates and throughput with pruners
4. **Progress**: X/Y pairs completed

---

## Tips for Next PC

- `VERBOSE=1 bash run_comparison.sh fast` — shows detailed output
- `cat .temp/runs/*.json | head -c 500` — peek at raw results
- Results are deterministic (seed=42) — rerunning same config gives ~same values
- Ctrl+C saves work; just re-run to continue
- If stuck: delete `.temp/runs/*.json` to restart fresh

---

## Contact Points

**Key Files to Check**:
- Pruning logic: `MatrixBenchmarkRunner.cs:197-210` & `optuna_matrix.py:119-123`
- Objective math: `Objectives.cs` vs `objectives.py`
- Report generation: `generate_report.py:44-98`
- Orchestration: `run_comparison.sh:81-136`

**Commit Hash**: `bfa3e9a` — contains all benchmark code and framework
