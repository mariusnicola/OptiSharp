# OptiSharp vs Optuna — Benchmark Comparison

**Generated:** 2026-02-22 17:22:39 UTC

**Progress:** 41/41 configuration pairs completed

## Summary
### Best Value Quality (lower = better)
| Objective | Sampler | Params | Trials | Pruner | OptiSharp | Optuna | Winner |
|---|---|---|---|---|---|---|---|
| ackley | tpe | 10 | 100 | median | 16.95 | 19.91 | OptiSharp |
| ackley | tpe | 10 | 100 | none | 18.48 | 17.36 | Optuna |
| ackley | tpe | 10 | 100 | sha | 17.48 | 21.09 | OptiSharp |
| ackley | tpe | 10 | 300 | median | 13.80 | 19.91 | OptiSharp |
| ackley | tpe | 10 | 300 | none | 14.37 | 12.76 | Optuna |
| ackley | tpe | 10 | 300 | sha | 14.27 | 19.85 | OptiSharp |
| ackley | tpe | 10 | 500 | median | 14.83 | 19.91 | OptiSharp |
| ackley | tpe | 10 | 500 | none | 11.87 | 12.07 | OptiSharp |
| ackley | tpe | 10 | 500 | sha | 13.52 | 19.85 | OptiSharp |
| rastrigin | tpe | 10 | 100 | median | 91.57 | 69.55 | Optuna |
| rastrigin | tpe | 10 | 100 | none | 91.20 | 90.27 | Optuna |
| rastrigin | tpe | 10 | 100 | sha | 71.36 | 138.14 | OptiSharp |
| rastrigin | tpe | 10 | 300 | median | 54.67 | 69.55 | OptiSharp |
| rastrigin | tpe | 10 | 300 | none | 62.10 | 49.16 | Optuna |
| rastrigin | tpe | 10 | 300 | sha | 56.74 | 100.11 | OptiSharp |
| rastrigin | tpe | 10 | 500 | median | 68.60 | 69.55 | OptiSharp |
| rastrigin | tpe | 10 | 500 | none | 85.89 | 44.14 | Optuna |
| rastrigin | tpe | 10 | 500 | sha | 53.98 | 89.03 | OptiSharp |
| rosenbrock | tpe | 10 | 100 | median | 143.58 | 336.10 | OptiSharp |
| rosenbrock | tpe | 10 | 100 | none | 193.18 | 562.47 | OptiSharp |
| rosenbrock | tpe | 10 | 100 | sha | 383.16 | 125.55 | Optuna |
| rosenbrock | tpe | 10 | 300 | median | 48.87 | 126.22 | OptiSharp |
| rosenbrock | tpe | 10 | 300 | none | 88.94 | 176.82 | OptiSharp |
| rosenbrock | tpe | 10 | 300 | sha | 106.63 | 38.77 | Optuna |
| rosenbrock | tpe | 10 | 500 | median | 49.26 | 126.22 | OptiSharp |
| rosenbrock | tpe | 10 | 500 | none | 66.60 | 87.49 | OptiSharp |
| rosenbrock | tpe | 10 | 500 | sha | 48.50 | 24.43 | Optuna |
| rosenbrock | tpe | 10 | 1000 | median | 29.50 | 47.95 | OptiSharp |
| rosenbrock | tpe | 10 | 1000 | none | 17.00 | 28.43 | OptiSharp |
| sphere | tpe | 10 | 100 | median | 9.60 | 13.80 | OptiSharp |
| sphere | tpe | 10 | 100 | none | 13.46 | 8.16 | Optuna |
| sphere | tpe | 10 | 100 | sha | 11.36 | 10.41 | Optuna |
| sphere | tpe | 10 | 300 | median | 5.64 | 6.60 | OptiSharp |
| sphere | tpe | 10 | 300 | none | 3.82 | 4.06 | OptiSharp |
| sphere | tpe | 10 | 300 | sha | 3.01 | 1.40 | Optuna |
| sphere | tpe | 10 | 500 | median | 5.29 | 2.86 | Optuna |
| sphere | tpe | 10 | 500 | none | 2.94 | 2.75 | Optuna |
| sphere | tpe | 10 | 500 | sha | 3.86 | 0.8671 | Optuna |
| sphere | tpe | 10 | 1000 | median | 1.31 | 1.56 | OptiSharp |
| sphere | tpe | 10 | 1000 | none | 2.24 | 1.02 | Optuna |
| sphere | tpe | 10 | 1000 | sha | 1.61 | 0.2291 | Optuna |
### Throughput (trials/sec, higher = better)
| Objective | Sampler | Params | Trials | OptiSharp | Optuna | Winner |
|---|---|---|---|---|---|---|
| ackley | tpe | 10 | 100 | 434.78 | 126.46 | OptiSharp |
| ackley | tpe | 10 | 300 | 298.51 | 89.98 | OptiSharp |
| ackley | tpe | 10 | 500 | 313.87 | 74.65 | OptiSharp |
| rastrigin | tpe | 10 | 100 | 469.48 | 128.33 | OptiSharp |
| rastrigin | tpe | 10 | 300 | 325.03 | 78.04 | OptiSharp |
| rastrigin | tpe | 10 | 500 | 319.28 | 77.45 | OptiSharp |
| rosenbrock | tpe | 10 | 100 | 467.29 | 121.36 | OptiSharp |
| rosenbrock | tpe | 10 | 300 | 389.11 | 84.51 | OptiSharp |
| rosenbrock | tpe | 10 | 500 | 288.52 | 73.87 | OptiSharp |
| rosenbrock | tpe | 10 | 1000 | 313.87 | 52.61 | OptiSharp |
| sphere | tpe | 10 | 100 | 431.03 | 135.08 | OptiSharp |
| sphere | tpe | 10 | 300 | 331.86 | 89.23 | OptiSharp |
| sphere | tpe | 10 | 500 | 314.27 | 73.80 | OptiSharp |
| sphere | tpe | 10 | 1000 | 312.40 | 50.56 | OptiSharp |
## Per-Objective Results
### Ackley
#### Convergence (best value at checkpoints)
| Config | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| OptiSharp/tpe/10p/100t/median | 19.50 | 19.05 | 19.05 | 16.95 | 16.95 |
| Optuna/tpe/10p/100t/median | 20.89 | 20.30 | 20.30 | 19.91 | 19.91 |
| OptiSharp/tpe/10p/100t/none | 20.65 | 18.96 | 18.96 | 18.48 | 18.48 |
| Optuna/tpe/10p/100t/none | 20.16 | 19.83 | 19.33 | 17.36 | 17.36 |
| OptiSharp/tpe/10p/100t/sha | 20.28 | 19.77 | 17.48 | 17.48 | 17.48 |
| Optuna/tpe/10p/100t/sha | 21.43 | 21.13 | 21.09 | 21.09 | 21.09 |
| OptiSharp/tpe/10p/300t/median | 19.78 | 17.36 | 15.42 | 14.38 | 13.80 |
| Optuna/tpe/10p/300t/median | 20.30 | 19.91 | 19.91 | 19.91 | 19.91 |
| OptiSharp/tpe/10p/300t/none | 19.15 | 17.74 | 16.25 | 15.66 | 14.37 |
| Optuna/tpe/10p/300t/none | 19.33 | 17.36 | 15.72 | 13.09 | 12.76 |
| OptiSharp/tpe/10p/300t/sha | 18.58 | 16.19 | 15.78 | 14.98 | 14.27 |
| Optuna/tpe/10p/300t/sha | 21.09 | 19.85 | 19.85 | 19.85 | 19.85 |
| OptiSharp/tpe/10p/500t/median | 18.98 | 17.17 | 15.80 | 15.21 | 14.83 |
| Optuna/tpe/10p/500t/median | 19.91 | 19.91 | 19.91 | 19.91 | 19.91 |
| OptiSharp/tpe/10p/500t/none | 17.68 | 14.80 | 14.03 | 12.24 | 11.87 |
| Optuna/tpe/10p/500t/none | 17.36 | 14.35 | 12.76 | 12.07 | 12.07 |
| OptiSharp/tpe/10p/500t/sha | 16.77 | 15.95 | 14.97 | 14.53 | 13.52 |
| Optuna/tpe/10p/500t/sha | 21.09 | 19.85 | 19.85 | 19.85 | 19.85 |
### Rastrigin
#### Convergence (best value at checkpoints)
| Config | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| OptiSharp/tpe/10p/100t/median | 111.66 | 95.13 | 95.13 | 91.57 | 91.57 |
| Optuna/tpe/10p/100t/median | 132.15 | 118.04 | 69.55 | 69.55 | 69.55 |
| OptiSharp/tpe/10p/100t/none | 124.16 | 91.20 | 91.20 | 91.20 | 91.20 |
| Optuna/tpe/10p/100t/none | 106.64 | 106.64 | 106.64 | 105.28 | 90.27 |
| OptiSharp/tpe/10p/100t/sha | 92.23 | 84.70 | 84.70 | 84.70 | 71.36 |
| Optuna/tpe/10p/100t/sha | 171.42 | 171.42 | 160.78 | 149.58 | 138.14 |
| OptiSharp/tpe/10p/300t/median | 104.90 | 65.32 | 63.65 | 54.67 | 54.67 |
| Optuna/tpe/10p/300t/median | 69.55 | 69.55 | 69.55 | 69.55 | 69.55 |
| OptiSharp/tpe/10p/300t/none | 98.49 | 77.60 | 77.60 | 62.10 | 62.10 |
| Optuna/tpe/10p/300t/none | 106.64 | 90.27 | 61.06 | 49.16 | 49.16 |
| OptiSharp/tpe/10p/300t/sha | 108.71 | 84.01 | 77.53 | 61.77 | 56.74 |
| Optuna/tpe/10p/300t/sha | 160.78 | 130.70 | 100.11 | 100.11 | 100.11 |
| OptiSharp/tpe/10p/500t/median | 73.74 | 73.74 | 68.60 | 68.60 | 68.60 |
| Optuna/tpe/10p/500t/median | 69.55 | 69.55 | 69.55 | 69.55 | 69.55 |
| OptiSharp/tpe/10p/500t/none | 113.12 | 85.89 | 85.89 | 85.89 | 85.89 |
| Optuna/tpe/10p/500t/none | 90.27 | 55.22 | 49.16 | 49.16 | 44.14 |
| OptiSharp/tpe/10p/500t/sha | 88.11 | 53.98 | 53.98 | 53.98 | 53.98 |
| Optuna/tpe/10p/500t/sha | 138.14 | 100.11 | 100.11 | 89.03 | 89.03 |
### Rosenbrock
#### Convergence (best value at checkpoints)
| Config | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| OptiSharp/tpe/10p/100t/median | 1019.00 | 337.12 | 337.12 | 166.95 | 143.58 |
| Optuna/tpe/10p/100t/median | 1904.44 | 1085.23 | 618.07 | 618.07 | 336.10 |
| OptiSharp/tpe/10p/100t/none | 918.24 | 428.05 | 193.18 | 193.18 | 193.18 |
| Optuna/tpe/10p/100t/none | 695.85 | 613.35 | 613.35 | 562.47 | 562.47 |
| OptiSharp/tpe/10p/100t/sha | 833.61 | 795.57 | 620.49 | 383.16 | 383.16 |
| Optuna/tpe/10p/100t/sha | 797.18 | 377.62 | 302.84 | 145.85 | 125.55 |
| OptiSharp/tpe/10p/300t/median | 282.52 | 141.98 | 83.78 | 66.91 | 48.87 |
| Optuna/tpe/10p/300t/median | 618.07 | 336.10 | 126.22 | 126.22 | 126.22 |
| OptiSharp/tpe/10p/300t/none | 386.27 | 225.11 | 122.84 | 90.93 | 88.94 |
| Optuna/tpe/10p/300t/none | 613.35 | 442.94 | 323.25 | 240.58 | 176.82 |
| OptiSharp/tpe/10p/300t/sha | 749.64 | 395.75 | 147.24 | 127.27 | 106.63 |
| Optuna/tpe/10p/300t/sha | 302.84 | 125.55 | 73.13 | 44.41 | 38.77 |
| OptiSharp/tpe/10p/500t/median | 361.81 | 113.40 | 102.49 | 96.94 | 49.26 |
| Optuna/tpe/10p/500t/median | 336.10 | 126.22 | 126.22 | 126.22 | 126.22 |
| OptiSharp/tpe/10p/500t/none | 294.34 | 136.35 | 98.02 | 81.07 | 66.60 |
| Optuna/tpe/10p/500t/none | 562.47 | 290.00 | 176.82 | 87.49 | 87.49 |
| OptiSharp/tpe/10p/500t/sha | 316.81 | 81.49 | 58.71 | 51.99 | 48.50 |
| Optuna/tpe/10p/500t/sha | 125.55 | 73.13 | 38.77 | 32.32 | 24.43 |
| OptiSharp/tpe/10p/1000t/median | 116.04 | 58.35 | 35.69 | 33.22 | 29.50 |
| Optuna/tpe/10p/1000t/median | 126.22 | 126.22 | 103.24 | 77.08 | 47.95 |
| OptiSharp/tpe/10p/1000t/none | 107.65 | 63.04 | 47.91 | 24.30 | 17.00 |
| Optuna/tpe/10p/1000t/none | 290.00 | 87.49 | 87.49 | 53.12 | 28.43 |
### Sphere
#### Convergence (best value at checkpoints)
| Config | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| OptiSharp/tpe/10p/100t/median | 38.32 | 25.67 | 17.05 | 12.34 | 9.60 |
| Optuna/tpe/10p/100t/median | 51.83 | 19.48 | 19.48 | 17.23 | 13.80 |
| OptiSharp/tpe/10p/100t/none | 56.21 | 27.08 | 22.40 | 13.46 | 13.46 |
| Optuna/tpe/10p/100t/none | 25.75 | 19.98 | 12.90 | 12.90 | 8.16 |
| OptiSharp/tpe/10p/100t/sha | 29.07 | 13.28 | 11.36 | 11.36 | 11.36 |
| Optuna/tpe/10p/100t/sha | 39.01 | 31.55 | 31.48 | 10.41 | 10.41 |
| OptiSharp/tpe/10p/300t/median | 18.91 | 11.18 | 9.63 | 8.25 | 5.64 |
| Optuna/tpe/10p/300t/median | 19.48 | 13.80 | 9.32 | 7.19 | 6.60 |
| OptiSharp/tpe/10p/300t/none | 14.83 | 7.38 | 5.12 | 3.83 | 3.82 |
| Optuna/tpe/10p/300t/none | 12.90 | 8.16 | 5.93 | 5.58 | 4.06 |
| OptiSharp/tpe/10p/300t/sha | 18.02 | 5.83 | 5.83 | 3.49 | 3.01 |
| Optuna/tpe/10p/300t/sha | 31.48 | 8.09 | 4.29 | 1.40 | 1.40 |
| OptiSharp/tpe/10p/500t/median | 13.89 | 11.86 | 8.82 | 6.78 | 5.29 |
| Optuna/tpe/10p/500t/median | 13.80 | 8.92 | 6.60 | 3.81 | 2.86 |
| OptiSharp/tpe/10p/500t/none | 8.44 | 5.09 | 2.94 | 2.94 | 2.94 |
| Optuna/tpe/10p/500t/none | 8.16 | 5.58 | 4.06 | 3.30 | 2.75 |
| OptiSharp/tpe/10p/500t/sha | 19.96 | 10.54 | 7.80 | 5.73 | 3.86 |
| Optuna/tpe/10p/500t/sha | 10.41 | 1.40 | 1.40 | 1.04 | 0.8671 |
| OptiSharp/tpe/10p/1000t/median | 4.88 | 3.36 | 2.03 | 1.31 | 1.31 |
| Optuna/tpe/10p/1000t/median | 8.92 | 3.81 | 2.40 | 1.63 | 1.56 |
| OptiSharp/tpe/10p/1000t/none | 6.37 | 3.65 | 2.87 | 2.24 | 2.24 |
| Optuna/tpe/10p/1000t/none | 5.58 | 3.30 | 1.78 | 1.49 | 1.02 |
| OptiSharp/tpe/10p/1000t/sha | 11.98 | 4.86 | 3.75 | 1.61 | 1.61 |
| Optuna/tpe/10p/1000t/sha | 1.40 | 1.04 | 0.7234 | 0.5503 | 0.2291 |
## Pruning Effectiveness
| Pruner | Sampler | Objective | OptiSharp Rate | Optuna Rate | OptiSharp TPS | Optuna TPS |
|---|---|---|---|---|---|---|
| median | tpe | ackley | 0.0% | 85.0% | 467.29 | 125.09 |
| sha | tpe | ackley | 0.0% | 92.0% | 450.45 | 106.74 |
| median | tpe | ackley | 0.0% | 93.0% | 299.70 | 77.20 |
| sha | tpe | ackley | 0.0% | 95.3% | 331.13 | 79.97 |
| median | tpe | ackley | 0.0% | 94.6% | 309.02 | 65.12 |
| sha | tpe | ackley | 0.0% | 94.8% | 315.66 | 61.46 |
| median | tpe | rastrigin | 0.0% | 79.0% | 485.44 | 83.19 |
| sha | tpe | rastrigin | 0.0% | 92.0% | 460.83 | 115.20 |
| median | tpe | rastrigin | 0.0% | 90.3% | 280.64 | 77.50 |
| sha | tpe | rastrigin | 0.0% | 93.3% | 298.80 | 79.79 |
| median | tpe | rastrigin | 0.0% | 93.4% | 318.67 | 61.90 |
| sha | tpe | rastrigin | 0.0% | 93.2% | 309.98 | 63.18 |
| median | tpe | rosenbrock | 0.0% | 62.0% | 490.20 | 105.36 |
| sha | tpe | rosenbrock | 0.0% | 72.0% | 487.80 | 123.17 |
| median | tpe | rosenbrock | 0.0% | 48.0% | 308.01 | 74.01 |
| sha | tpe | rosenbrock | 0.0% | 58.0% | 309.92 | 75.28 |
| median | tpe | rosenbrock | 0.0% | 52.4% | 294.46 | 61.50 |
| sha | tpe | rosenbrock | 0.0% | 63.4% | 294.64 | 65.70 |
| median | tpe | rosenbrock | 0.0% | 58.2% | 304.60 | 42.08 |
| median | tpe | sphere | 0.0% | 58.0% | 492.61 | 105.13 |
| sha | tpe | sphere | 0.0% | 76.0% | 446.43 | 118.63 |
| median | tpe | sphere | 0.0% | 52.7% | 320.17 | 76.78 |
| sha | tpe | sphere | 0.0% | 54.7% | 327.15 | 79.19 |
| median | tpe | sphere | 0.0% | 44.6% | 292.40 | 61.22 |
| sha | tpe | sphere | 0.0% | 57.6% | 297.80 | 62.64 |
| median | tpe | sphere | 0.0% | 46.8% | 308.45 | 40.52 |
| sha | tpe | sphere | 0.0% | 66.3% | 302.30 | 42.11 |
---

*Lower is better for quality metrics (best value). Higher is better for throughput.*
