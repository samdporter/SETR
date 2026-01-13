#!/bin/bash

# Quick status check for preconditioner experiments

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
OUTPUT_DIR="$SCRIPT_DIR/output"

echo "=== Preconditioner Experiment Status ==="
echo ""

# Check baselines
echo "BASELINE RECONSTRUCTIONS (Stage 1):"
echo "-----------------------------------"
BASELINE_DIR="$OUTPUT_DIR/baselines_1bpos"
if [ -d "$BASELINE_DIR" ]; then
    TOTAL_BASELINES=$(find "$BASELINE_DIR" -maxdepth 1 -type d -name "baseline_alpha_*" | wc -l)
    COMPLETED_BASELINES=$(find "$BASELINE_DIR" -name "baseline_metrics.json" | wc -l)
    
    echo "  Total baselines: $TOTAL_BASELINES"
    echo "  Completed: $COMPLETED_BASELINES"
    
    if [ $COMPLETED_BASELINES -gt 0 ]; then
        echo ""
        echo "  Completed baselines:"
        for metrics_file in "$BASELINE_DIR"/*/baseline_metrics.json; do
            if [ -f "$metrics_file" ]; then
                alpha=$(python3 -c "import json; print(json.load(open('$metrics_file'))['alpha'])")
                final_obj=$(python3 -c "import json; print(f\"{json.load(open('$metrics_file'))['final_objective']:.6f}\")")
                runtime=$(python3 -c "import json; r=json.load(open('$metrics_file'))['total_runtime']; print(f'{r/3600:.1f}h')")
                echo "    α=$alpha: obj=$final_obj, time=$runtime"
            fi
        done
    fi
    
    if [ $COMPLETED_BASELINES -lt $TOTAL_BASELINES ]; then
        echo ""
        echo "  ⚠️  Some baselines still running or failed"
        echo "      Check: ls -la $BASELINE_DIR/*/baseline_metrics.json"
    fi
else
    echo "  ❌ No baselines found"
    echo "      Run: ./launch_baseline_recons.sh config_1bpos_anthro_long.yaml 200 1.0 vtv_svd_principal_alpha full"
fi
echo ""

# Check sweep results
echo "PRECONDITIONER SWEEP (Stage 2):"
echo "-------------------------------"
SWEEP_DIR="$OUTPUT_DIR/precond_1bpos"
if [ -d "$SWEEP_DIR" ]; then
    TOTAL_SWEEP=$(find "$SWEEP_DIR" -maxdepth 1 -type d -name "precond_*" | wc -l)
    COMPLETED_SWEEP=$(find "$SWEEP_DIR" -name "result.csv" | wc -l)
    SUCCESSFUL_SWEEP=$(find "$SWEEP_DIR" -name "result.csv" -exec grep -l "success" {} \; 2>/dev/null | wc -l)
    
    echo "  Total sweep jobs: $TOTAL_SWEEP"
    echo "  Completed: $COMPLETED_SWEEP"
    echo "  Successful: $SUCCESSFUL_SWEEP"
    
    if [ $COMPLETED_SWEEP -gt 0 ]; then
        # Count by preconditioner type
        echo ""
        echo "  Completed by preconditioner type:"
        for precond in bsrem vtv_svd_principal_alpha vtv_frobenius_surrogate_pd vtv_vector_tv_per_modality; do
            count=$(find "$SWEEP_DIR" -name "result.csv" -path "*precond_${precond}_*" 2>/dev/null | wc -l)
            if [ $count -gt 0 ]; then
                echo "    $precond: $count"
            fi
        done
    fi
    
    if [ $COMPLETED_SWEEP -lt $TOTAL_SWEEP ]; then
        echo ""
        echo "  ⚠️  Some sweep jobs still running or incomplete"
        echo "      Progress: $COMPLETED_SWEEP / $TOTAL_SWEEP"
    fi
else
    echo "  ❌ No sweep results found"
    echo "      Run: ./launch_precond_sweep.sh precond_sweep_1bpos.yaml full"
fi
echo ""

# Check analysis results
echo "ANALYSIS RESULTS (Stage 3):"
echo "---------------------------"
ANALYSIS_DIR="$OUTPUT_DIR/precond_1bpos_analysis"
if [ -d "$ANALYSIS_DIR" ]; then
    if [ -f "$ANALYSIS_DIR/analysis_results.csv" ]; then
        NUM_ANALYZED=$(tail -n +2 "$ANALYSIS_DIR/analysis_results.csv" | wc -l)
        echo "  ✓ Analysis complete: $NUM_ANALYZED results analyzed"
        echo "  Results: $ANALYSIS_DIR/analysis_results.csv"
        echo "  Report: $ANALYSIS_DIR/summary_report.md"
        
        if [ -f "$ANALYSIS_DIR/summary_report.md" ]; then
            echo ""
            echo "  Quick summary:"
            head -20 "$ANALYSIS_DIR/summary_report.md" | grep -A 10 "Overall Statistics" || true
        fi
    else
        echo "  ⚠️  Analysis directory exists but no results yet"
    fi
else
    if [ -d "$SWEEP_DIR" ] && [ $COMPLETED_SWEEP -gt 0 ] && [ -d "$BASELINE_DIR" ] && [ $COMPLETED_BASELINES -gt 0 ]; then
        echo "  ⏳ Ready to analyze!"
        echo "      Run: python scripts/analyze_precond_sweep.py --sweep precond_1bpos --baseline baselines_1bpos"
    else
        echo "  ⏳ Waiting for baselines and sweep to complete"
    fi
fi
echo ""

# Check cluster jobs
echo "CLUSTER STATUS:"
echo "---------------"
if command -v qstat &> /dev/null; then
    BASELINE_JOBS=$(qstat -u $USER 2>/dev/null | grep -c "baseline" || true)
    PRECOND_JOBS=$(qstat -u $USER 2>/dev/null | grep -c "precond" || true)
    
    if [ $BASELINE_JOBS -gt 0 ] || [ $PRECOND_JOBS -gt 0 ]; then
        echo "  Active jobs:"
        [ $BASELINE_JOBS -gt 0 ] && echo "    Baselines: $BASELINE_JOBS"
        [ $PRECOND_JOBS -gt 0 ] && echo "    Sweep: $PRECOND_JOBS"
    else
        echo "  No active jobs"
    fi
else
    echo "  qstat not available (not on cluster?)"
fi
echo ""

echo "=== Next Steps ==="
if [ ! -d "$BASELINE_DIR" ] || [ $COMPLETED_BASELINES -eq 0 ]; then
    echo "1. Run baselines: ./launch_baseline_recons.sh config_1bpos_anthro_long.yaml 200 1.0 vtv_svd_principal_alpha full"
elif [ $COMPLETED_BASELINES -lt 3 ]; then
    echo "1. Wait for all baselines to complete ($COMPLETED_BASELINES/3 done)"
elif [ ! -d "$SWEEP_DIR" ] || [ $COMPLETED_SWEEP -eq 0 ]; then
    echo "2. Run sweep: ./launch_precond_sweep.sh precond_sweep_1bpos.yaml full"
elif [ $COMPLETED_SWEEP -lt 60 ]; then
    echo "2. Wait for sweep to complete ($COMPLETED_SWEEP/60 done)"
    echo "   Monitor: python scripts/analyze_precond_sweep.py --sweep precond_1bpos --baseline baselines_1bpos --watch"
else
    echo "3. Review results: cat $ANALYSIS_DIR/summary_report.md"
fi
echo ""
