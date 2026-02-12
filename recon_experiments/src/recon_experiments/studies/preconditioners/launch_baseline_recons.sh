#!/bin/bash

# Launch baseline reconstructions for all alpha values
# These establish reference solutions for preconditioner comparison

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
FUNC_DIR="$SCRIPT_DIR"
PARAM_DIR="$FUNC_DIR/parameters"
SCRIPTS_DIR="$FUNC_DIR/scripts"
RUNNER_DIR="$BASE_DIR/src/recon_experiments/runners/scripts"

# Configuration
BASE_CONFIG="${1:-config_1bpos_anthro.yaml}"
NUM_EPOCHS="${2:-200}"
PRECOND_TYPE="${PRECOND_TYPE:-mm_block_diag}"
PRECOND_COMBINE="${PRECOND_COMBINE:-harmonic}"
PRECOND_SCALAR_REDUCTION="${PRECOND_SCALAR_REDUCTION:-diag}"
# Support old interface: args 3 and 4 might be step_size and precond_type (ignored now)
# Last arg should be mode
if [ "$#" -eq 5 ]; then
    # Old interface: config epochs step_size precond_type mode
    MODE="${5:-local}"
elif [ "$#" -eq 3 ]; then
    # New interface: config epochs mode
    MODE="${3:-local}"
elif [ "$#" -eq 2 ]; then
    MODE="local"
elif [ "$#" -eq 1 ]; then
    MODE="local"
else
    MODE="${3:-local}"
fi

echo "=== Baseline Reconstruction Launcher ==="
echo "Base config: $BASE_CONFIG"
echo "Epochs: $NUM_EPOCHS"
echo "Preconditioner: $PRECOND_TYPE"
echo "Combine: $PRECOND_COMBINE"
echo "Block scalar reduction: $PRECOND_SCALAR_REDUCTION"
echo "Mode: $MODE"
echo ""

# Read alpha values from parameter file
ALPHAS_FILE="$PARAM_DIR/alphas.csv"
if [ ! -f "$ALPHAS_FILE" ]; then
    echo "Error: Alpha file not found: $ALPHAS_FILE"
    exit 1
fi

# Parse alphas
mapfile -t ALPHAS < <(tail -n +2 "$ALPHAS_FILE" | awk -F, 'NF{gsub(/^[ \t]+|[ \t]+$/,"",$1); if($1!="") print $1}')
NUM_ALPHAS=${#ALPHAS[@]}

echo "Found $NUM_ALPHAS alpha values: ${ALPHAS[*]}"
echo ""

# Output directory
OUTPUT_BASE="$FUNC_DIR/output/baselines_1bpos"
mkdir -p "$OUTPUT_BASE"

case "$MODE" in
    "local")
        # Run first alpha locally for testing
        ALPHA=${ALPHAS[0]}
        echo "Running baseline for alpha=$ALPHA locally..."
        
        OUTPUT_DIR="$OUTPUT_BASE/baseline_alpha_${ALPHA}"
        mkdir -p "$OUTPUT_DIR"
        
        cd "$BASE_DIR"
        python "$RUNNER_DIR/run_dtnv_1bpos.py" \
            --config "configs/$BASE_CONFIG" \
            --override "num_epochs=$NUM_EPOCHS" \
            --override "alpha=$ALPHA" \
            --override "beta=$ALPHA" \
            --override "precond_type=$PRECOND_TYPE" \
            --override "precond_combine=$PRECOND_COMBINE" \
            --override "block_scalar_reduction=$PRECOND_SCALAR_REDUCTION" \
            --override "output_path=$OUTPUT_DIR"
        
        # Create baseline metrics file for compatibility with analysis
        if [ -f "$OUTPUT_DIR/result.csv" ]; then
            python3 -c "
import json
import pandas as pd
df = pd.read_csv('$OUTPUT_DIR/result.csv')
metrics = {
    'alpha': $ALPHA,
    'final_objective': float(df['final_objective'].iloc[-1]) if 'final_objective' in df else 0.0,
    'total_runtime': float(df['runtime'].iloc[-1]) if 'runtime' in df else 0.0,
    'num_epochs': $NUM_EPOCHS,
    'precond_type': '$PRECOND_TYPE',
    'status': 'success'
}
with open('$OUTPUT_DIR/baseline_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
"
        fi
        
        echo ""
        echo "Local test complete!"
        echo "Output: $OUTPUT_DIR"
        ;;
    
    "local_all")
        echo "Running baselines locally for all alpha values..."
        echo ""

        cd "$BASE_DIR"

        for ALPHA in "${ALPHAS[@]}"; do
            echo "Running baseline for alpha=$ALPHA locally..."

            OUTPUT_DIR="$OUTPUT_BASE/baseline_alpha_${ALPHA}"
            mkdir -p "$OUTPUT_DIR"

            python "$RUNNER_DIR/run_dtnv_1bpos.py"                 --config "configs/$BASE_CONFIG"                 --override "num_epochs=$NUM_EPOCHS"                 --override "alpha=$ALPHA"                 --override "beta=$ALPHA"                 --override "precond_type=$PRECOND_TYPE"                 --override "precond_combine=$PRECOND_COMBINE"                 --override "block_scalar_reduction=$PRECOND_SCALAR_REDUCTION"                 --override "output_path=$OUTPUT_DIR"

            # Create baseline metrics file for compatibility with analysis
            if [ -f "$OUTPUT_DIR/result.csv" ]; then
                python3 -c "
import json
import pandas as pd
df = pd.read_csv('$OUTPUT_DIR/result.csv')
metrics = {
    'alpha': $ALPHA,
    'final_objective': float(df['final_objective'].iloc[-1]) if 'final_objective' in df else 0.0,
    'total_runtime': float(df['runtime'].iloc[-1]) if 'runtime' in df else 0.0,
    'num_epochs': $NUM_EPOCHS,
    'precond_type': '$PRECOND_TYPE',
    'status': 'success'
}
with open('$OUTPUT_DIR/baseline_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
"
            fi

            echo ""
        done

        echo "Local-all run complete!"
        echo "Output: $OUTPUT_BASE"
        ;;

    "test")
        # Submit one job to cluster
        ALPHA=${ALPHAS[0]}
        echo "Submitting test baseline job for alpha=$ALPHA to cluster..."
        
        OUTPUT_DIR="$OUTPUT_BASE/baseline_alpha_${ALPHA}"
        LOG_DIR="$OUTPUT_BASE/_logs"
        mkdir -p "$LOG_DIR"
        
        cd "$BASE_DIR"
        
        qsub \
            -r y \
            -l h_rt=72:00:00 \
            -l tmem=95G \
            -l gpu=true \
            -N "baseline_1bpos_test" \
            -o "$LOG_DIR" \
            -e "$LOG_DIR" \
            -v "BASE_DIR=$BASE_DIR,OUTPUT_DIR=$OUTPUT_DIR,BASE_CONFIG=$BASE_CONFIG,ALPHA=$ALPHA,NUM_EPOCHS=$NUM_EPOCHS,PRECOND_TYPE=$PRECOND_TYPE,PRECOND_COMBINE=$PRECOND_COMBINE,PRECOND_SCALAR_REDUCTION=$PRECOND_SCALAR_REDUCTION" \
            "$SCRIPTS_DIR/baseline_recon.qsub.sh"
        
        echo "Test job submitted!"
        echo "Logs: $LOG_DIR"
        ;;
    
    "full"|*)
        echo "Submitting $NUM_ALPHAS baseline jobs to cluster..."
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi
        
        LOG_DIR="$OUTPUT_BASE/_logs"
        mkdir -p "$LOG_DIR"
        
        cd "$BASE_DIR"
        
        for ALPHA in "${ALPHAS[@]}"; do
            OUTPUT_DIR="$OUTPUT_BASE/baseline_alpha_${ALPHA}"
            
            echo "Submitting baseline for alpha=$ALPHA..."
            
            qsub \
                -r y \
                -l h_rt=240:00:00 \
                -l tmem=95G \
                -l gpu=true \
                -N "baseline_alpha_${ALPHA}" \
                -o "$LOG_DIR" \
                -e "$LOG_DIR" \
                -v "BASE_DIR=$BASE_DIR,OUTPUT_DIR=$OUTPUT_DIR,BASE_CONFIG=$BASE_CONFIG,ALPHA=$ALPHA,NUM_EPOCHS=$NUM_EPOCHS,PRECOND_TYPE=$PRECOND_TYPE,PRECOND_COMBINE=$PRECOND_COMBINE,PRECOND_SCALAR_REDUCTION=$PRECOND_SCALAR_REDUCTION" \
                "$SCRIPTS_DIR/baseline_recon.qsub.sh"
        done
        
        echo ""
        echo "All baseline jobs submitted!"
        echo "Output: $OUTPUT_BASE"
        echo "Logs: $LOG_DIR"
        ;;
esac
