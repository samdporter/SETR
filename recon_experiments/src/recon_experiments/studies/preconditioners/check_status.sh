#!/bin/bash

# Quick status check for preconditioner experiments

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
OUTPUT_DIR="$SCRIPT_DIR/output"
ALPHAS_FILE="$SCRIPT_DIR/parameters/alphas.csv"
PRECOND_TYPES_FILE="$SCRIPT_DIR/parameters/precond_types.csv"
STEP_SIZES_FILE="$SCRIPT_DIR/parameters/step_sizes.csv"

infer_bpos() {
    local value="${1:-1}"
    value="$(basename "$value" | tr '[:upper:]' '[:lower:]')"
    case "$value" in
        2|2bpos|*2bpos*) printf "2" ;;
        1|1bpos|*1bpos*) printf "1" ;;
        *)
            echo "Error: Could not infer bed positions from '$1'. Use 1, 2, 1bpos, 2bpos, or a matching config/sweep name." >&2
            exit 1
            ;;
    esac
}

BPOS="$(infer_bpos "${1:-1}")"
BASELINE_CONFIG_HINT="config_${BPOS}bpos.yaml"
if [[ "$BPOS" == "1" ]]; then
    BASELINE_CONFIG_HINT="config_1bpos_anthro.yaml"
fi
SWEEP_CONFIG_HINT="precond_sweep_${BPOS}bpos.yaml"
BASELINE_DIR="$OUTPUT_DIR/baselines_${BPOS}bpos"
SWEEP_DIR="$OUTPUT_DIR/precond_${BPOS}bpos"
ANALYSIS_DIR="$OUTPUT_DIR/precond_${BPOS}bpos_analysis"

SWEEP_REPEATS="${SWEEP_REPEATS:-1}"
if ! [[ "$SWEEP_REPEATS" =~ ^[0-9]+$ ]] || [ "$SWEEP_REPEATS" -lt 1 ]; then
    SWEEP_REPEATS=1
fi

echo "=== Preconditioner Experiment Status ==="
echo "Bed positions: $BPOS"
echo ""

# Check baselines
echo "BASELINE RECONSTRUCTIONS (Stage 1):"
echo "-----------------------------------"
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
    echo "      Run: ./launch_baseline_recons.sh $BASELINE_CONFIG_HINT 200 full"
fi
echo ""

# Check sweep results
echo "PRECONDITIONER SWEEP (Stage 2):"
echo "-------------------------------"
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
        if [ -f "$PRECOND_TYPES_FILE" ]; then
            mapfile -t PRECOND_LIST < <(tail -n +2 "$PRECOND_TYPES_FILE" | awk -F, 'NF{gsub(/^[ \t]+|[ \t]+$/,"",$1); if($1!="") print $1}' | sort -u)
            for precond in "${PRECOND_LIST[@]}"; do
                count=$(find "$SWEEP_DIR" -name "result.csv" -path "*precond_${precond}_*" 2>/dev/null | wc -l)
                if [ $count -gt 0 ]; then
                    echo "    $precond: $count"
                fi
            done
        fi
    fi
    
    if [ $COMPLETED_SWEEP -lt $TOTAL_SWEEP ]; then
        echo ""
        echo "  ⚠️  Some sweep jobs still running or incomplete"
        echo "      Progress: $COMPLETED_SWEEP / $TOTAL_SWEEP"
    fi
else
    echo "  ❌ No sweep results found"
    echo "      Run: ./launch_precond_sweep.sh $SWEEP_CONFIG_HINT full"
fi
echo ""

# Check analysis results
echo "ANALYSIS RESULTS (Stage 3):"
echo "---------------------------"
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
        echo "      Run: python scripts/analyze_precond_sweep.py --sweep precond_${BPOS}bpos --baseline baselines_${BPOS}bpos"
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
    echo "1. Run baselines: ./launch_baseline_recons.sh $BASELINE_CONFIG_HINT 200 full"
elif [ -f "$ALPHAS_FILE" ]; then
    EXPECTED_BASELINES=$(tail -n +2 "$ALPHAS_FILE" | wc -l)
    if [ $COMPLETED_BASELINES -lt $EXPECTED_BASELINES ]; then
        echo "1. Wait for all baselines to complete ($COMPLETED_BASELINES/$EXPECTED_BASELINES done)"
    fi
fi
if [ ! -d "$SWEEP_DIR" ] || [ $COMPLETED_SWEEP -eq 0 ]; then
    echo "2. Run sweep: ./launch_precond_sweep.sh $SWEEP_CONFIG_HINT full"
elif [ -f "$PRECOND_TYPES_FILE" ] && [ -f "$ALPHAS_FILE" ] && [ -f "$STEP_SIZES_FILE" ]; then
    BASE_SWEEP_COMBINATIONS=$(( $(tail -n +2 "$PRECOND_TYPES_FILE" | wc -l) * $(tail -n +2 "$ALPHAS_FILE" | wc -l) * $(tail -n +2 "$STEP_SIZES_FILE" | wc -l) ))
    DETECTED_REPEATS=$(find "$SWEEP_DIR" -maxdepth 1 -type d -name "precond_*_rep_*" 2>/dev/null | sed -n 's/.*_rep_\([0-9][0-9]*\)$/\1/p' | sort -n | tail -1)
    EFFECTIVE_REPEATS="$SWEEP_REPEATS"
    if [ -n "$DETECTED_REPEATS" ] && [ "$DETECTED_REPEATS" -gt "$EFFECTIVE_REPEATS" ]; then
        EFFECTIVE_REPEATS="$DETECTED_REPEATS"
    fi
    EXPECTED_SWEEP=$((BASE_SWEEP_COMBINATIONS * EFFECTIVE_REPEATS))
    if [ $COMPLETED_SWEEP -lt $EXPECTED_SWEEP ]; then
        echo "2. Wait for sweep to complete ($COMPLETED_SWEEP/$EXPECTED_SWEEP done)"
        echo "   Monitor: python scripts/analyze_precond_sweep.py --sweep precond_${BPOS}bpos --baseline baselines_${BPOS}bpos --watch"
    else
        echo "3. Review results: cat $ANALYSIS_DIR/summary_report.md"
    fi
else
    echo "3. Review results: cat $ANALYSIS_DIR/summary_report.md"
fi
echo ""
