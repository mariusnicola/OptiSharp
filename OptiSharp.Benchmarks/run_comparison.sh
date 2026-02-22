#!/bin/bash

# OptiSharp vs Optuna Benchmark Comparison
# Orchestrates side-by-side benchmarking with FULL observability

TIER=${1:-fast}
VERBOSE=${VERBOSE:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Create directories
mkdir -p .temp/runs .temp/bin benchmark-results

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         OptiSharp vs Optuna Comparison Benchmark              ║"
echo "║                                                                ║"
echo "║ Tier: $TIER                                                    ║"
echo "║ Results: .temp/runs/*.json                                    ║"
echo "║ Report: benchmark-results/comparison_report.md               ║"
echo "║                                                                ║"
echo "║ To stop: Press Ctrl+C                                         ║"
echo "║ For verbose output: VERBOSE=1 bash run_comparison.sh          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Build C# project once
echo "[INFO] Building OptiSharp.Benchmarks (Release)..."
cd "$ROOT_DIR"
if ! dotnet build OptiSharp.Benchmarks/OptiSharp.Benchmarks.csproj -c Release -o .temp/bin 2>&1 | grep -E "Build|error"; then
    echo "[ERROR] Build failed!"
    exit 1
fi
echo "✓ Build complete"
echo ""

# Define matrix parameters
SAMPLERS_FAST="tpe random cmaes"
SAMPLERS_EXT="tpe random"
PARAMS="10 50 100 200"
TRIALS_FAST="100 300 500 1000"
TRIALS_EXT="10000 100000"
OBJECTIVES="sphere rosenbrock rastrigin ackley"
PRUNERS="none median sha"

TOTAL_PAIRS=0
COMPLETED_PAIRS=0
START_TIME=$(date +%s)

# Count total pairs for progress
count_pairs() {
    local count=0
    if [ "$TIER" = "fast" ] || [ "$TIER" = "all" ]; then
        for S in $SAMPLERS_FAST; do
            for P in $PARAMS; do
                for T in $TRIALS_FAST; do
                    for O in $OBJECTIVES; do
                        for PR in $PRUNERS; do
                            count=$((count + 1))
                        done
                    done
                done
            done
        done
    fi

    if [ "$TIER" = "extended" ] || [ "$TIER" = "all" ]; then
        for S in $SAMPLERS_EXT; do
            for P in $PARAMS; do
                for T in $TRIALS_EXT; do
                    for O in $OBJECTIVES; do
                        for PR in $PRUNERS; do
                            count=$((count + 1))
                        done
                    done
                done
            done
        done
    fi

    echo $count
}

TOTAL_PAIRS=$(count_pairs)
echo "[INFO] Total configuration pairs to run: $TOTAL_PAIRS"
echo ""

run_pair() {
    local SAMPLER=$1 NPARAMS=$2 NTRIALS=$3 OBJ=$4 PRUNER=$5 T=$6
    local TAG="${SAMPLER}__${OBJ}__p${NPARAMS}__t${NTRIALS}__${PRUNER}"
    local CS_OUT=".temp/runs/optisharp__${TAG}.json"
    local PY_OUT=".temp/runs/optuna__${TAG}.json"

    COMPLETED_PAIRS=$((COMPLETED_PAIRS + 1))
    ELAPSED=$(($(date +%s) - START_TIME))
    PERCENT=$((COMPLETED_PAIRS * 100 / TOTAL_PAIRS))
    ETA=$((ELAPSED * (TOTAL_PAIRS - COMPLETED_PAIRS) / (COMPLETED_PAIRS + 1)))

    printf "[%3d%%] ETA: %3dm (%d/%d) %s/%s/%dp/%dt/%s\n" \
        "$PERCENT" "$((ETA/60))" "$COMPLETED_PAIRS" "$TOTAL_PAIRS" \
        "$SAMPLER" "$OBJ" "$NPARAMS" "$NTRIALS" "$PRUNER"

    # Skip if already completed
    if [ -f "$CS_OUT" ] && [ -f "$PY_OUT" ]; then
        echo "  [skip]"
        return
    fi

    # Run C# benchmark
    if [ ! -f "$CS_OUT" ]; then
        if [ $VERBOSE -eq 1 ]; then
            echo "  [running] OptiSharp..."
        fi
        if ! .temp/bin/OptiSharp.Benchmarks --matrix \
            --sampler $SAMPLER --params $NPARAMS --trials $NTRIALS \
            --objective $OBJ --pruner $PRUNER --tier $T --output "$CS_OUT" 2>&1 | grep -v "^$"; then
            echo "  [FAIL] OptiSharp"
            return 1
        fi
        echo "  [OK] OptiSharp"
    fi

    # Run Python benchmark
    if [ ! -f "$PY_OUT" ]; then
        if [ $VERBOSE -eq 1 ]; then
            echo "  [running] Optuna..."
        fi
        if ! python "$SCRIPT_DIR/python/optuna_matrix.py" \
            --sampler $SAMPLER --params $NPARAMS --trials $NTRIALS \
            --objective $OBJ --pruner $PRUNER --tier $T --output "$PY_OUT" 2>&1 | grep -v "^$"; then
            echo "  [FAIL] Optuna"
            return 1
        fi
        echo "  [OK] Optuna"
    fi

    # Update report incrementally
    if ! python "$SCRIPT_DIR/python/generate_report.py" \
        --input-dir .temp/runs \
        --output benchmark-results/comparison_report.md 2>&1 | grep -v "^$"; then
        echo "  [WARN] Report generation failed (continuing)"
    fi
}

# Trap Ctrl+C for graceful exit
trap 'echo ""; echo "[INTERRUPTED] Results saved to .temp/runs/ and benchmark-results/comparison_report.md"; exit 0' INT

# Run fast tier
if [ "$TIER" = "fast" ] || [ "$TIER" = "all" ]; then
    echo "═════════════════════════════════════════════════════════════"
    echo "FAST TIER: 100-1000 trials, all samplers (TPE/Random/CMA-ES)"
    echo "═════════════════════════════════════════════════════════════"
    echo ""

    for S in $SAMPLERS_FAST; do
        for P in $PARAMS; do
            for T in $TRIALS_FAST; do
                for O in $OBJECTIVES; do
                    for PR in $PRUNERS; do
                        run_pair $S $P $T $O $PR fast || true
                    done
                done
            done
        done
    done
fi

# Run extended tier
if [ "$TIER" = "extended" ] || [ "$TIER" = "all" ]; then
    echo ""
    echo "═════════════════════════════════════════════════════════════"
    echo "EXTENDED TIER: 10k-100k trials (TPE/Random only)"
    echo "═════════════════════════════════════════════════════════════"
    echo ""

    for S in $SAMPLERS_EXT; do
        for P in $PARAMS; do
            for T in $TRIALS_EXT; do
                for O in $OBJECTIVES; do
                    for PR in $PRUNERS; do
                        run_pair $S $P $T $O $PR extended || true
                    done
                done
            done
        done
    done
fi

TOTAL_TIME=$(($(date +%s) - START_TIME))
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Benchmark Complete!                        ║"
echo "║                                                                ║"
echo "║ Total time: ${TOTAL_TIME}s                                     ║"
echo "║ Results: .temp/runs/                                          ║"
echo "║ Report:  benchmark-results/comparison_report.md              ║"
echo "║                                                                ║"
echo "║ View results: cat benchmark-results/comparison_report.md      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
