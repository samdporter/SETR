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
NUM_EPOCHS="${2:-1000}"
PRECOND_TYPE="${PRECOND_TYPE:-mm_diag_block_maj}"
PRECOND_COMBINE="${PRECOND_COMBINE:-majoriser}"
PRECOND_SCALAR_REDUCTION="${PRECOND_SCALAR_REDUCTION:-diag}"
PRECOND_SAFETY_SCALE="${PRECOND_SAFETY_SCALE:-0.8}"
EM_PRECOND_CAP_TO_INITIAL_MAX="${EM_PRECOND_CAP_TO_INITIAL_MAX:-true}"
PRECOND_CAP_TO_INITIAL_MAX="${PRECOND_CAP_TO_INITIAL_MAX:-true}"
GAMMA_TNV="${GAMMA_TNV:-0.01}"
SUPPORT_MASK_FROM_SENSITIVITY="${SUPPORT_MASK_FROM_SENSITIVITY:-true}"
SUPPORT_MASK_REL_THRESHOLD="${SUPPORT_MASK_REL_THRESHOLD:-1e-3}"
SUPPORT_MASK_ABS_THRESHOLD="${SUPPORT_MASK_ABS_THRESHOLD:-0.0}"
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

if [[ "$MODE" =~ ^local(_all)?$ ]]; then
    export LOCAL_RUN_ID="${LOCAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
fi

echo "=== Baseline Reconstruction Launcher ==="
echo "Base config: $BASE_CONFIG"
echo "Epochs: $NUM_EPOCHS"
echo "Preconditioner: $PRECOND_TYPE"
echo "Combine: $PRECOND_COMBINE"
echo "Block scalar reduction: $PRECOND_SCALAR_REDUCTION"
echo "Preconditioner safety scale: $PRECOND_SAFETY_SCALE"
echo "EM preconditioner cap to initial max: $EM_PRECOND_CAP_TO_INITIAL_MAX"
echo "Combined preconditioner cap to initial max: $PRECOND_CAP_TO_INITIAL_MAX"
echo "Gamma TNV: $GAMMA_TNV"
echo "Support mask from sensitivity: $SUPPORT_MASK_FROM_SENSITIVITY"
echo "Support mask relative threshold: $SUPPORT_MASK_REL_THRESHOLD"
echo "Support mask absolute threshold: $SUPPORT_MASK_ABS_THRESHOLD"
echo "Mode: $MODE"
if [[ "$MODE" =~ ^local(_all)?$ ]]; then
    echo "Local run ID: $LOCAL_RUN_ID"
fi
echo ""

infer_bpos() {
    local config_name
    config_name="$(basename "$1" | tr '[:upper:]' '[:lower:]')"
    if [[ "$config_name" == *"2bpos"* ]]; then
        printf "2"
    elif [[ "$config_name" == *"1bpos"* ]]; then
        printf "1"
    else
        return 1
    fi
}

if ! BPOS="$(infer_bpos "$BASE_CONFIG")"; then
    echo "Error: Could not infer bed positions from config name '$BASE_CONFIG'."
    echo "Expected the config filename to contain '1bpos' or '2bpos'."
    exit 1
fi

RUNNER_SCRIPT="$RUNNER_DIR/run_dtnv_${BPOS}bpos.py"
if [ ! -f "$RUNNER_SCRIPT" ]; then
    echo "Error: Reconstruction script not found: $RUNNER_SCRIPT"
    exit 1
fi

echo "Bed positions: $BPOS"
echo "Runner: $RUNNER_SCRIPT"
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
OUTPUT_ROOT="${PRECOND_OUTPUT_ROOT:-$FUNC_DIR/output}"
if [[ "$MODE" =~ ^local(_all)?$ && -n "${LOCAL_RUN_ID:-}" && -z "${PRECOND_OUTPUT_ROOT:-}" ]]; then
    OUTPUT_ROOT="$FUNC_DIR/output/local_runs/$LOCAL_RUN_ID"
fi
OUTPUT_BASE="$OUTPUT_ROOT/baselines_${BPOS}bpos"
mkdir -p "$OUTPUT_BASE"

RESTART_BASELINE="${RESTART_BASELINE:-false}"

find_latest_iteration_info() {
    local dir="$1"
    local modality="$2"
    local latest_file=""
    local latest_iter=-1
    shopt -s nullglob
    for img in "$dir"/image_"${modality}"_*".hv"; do
        [[ -f "$img" ]] || continue
        local base
        base=$(basename "$img")
        if [[ "$base" =~ _([0-9]+)\.hv$ ]]; then
            local iter="${BASH_REMATCH[1]}"
            if (( iter > latest_iter )); then
                latest_iter=$iter
                latest_file="$img"
            fi
        fi
    done
    shopt -u nullglob
    if (( latest_iter >= 0 )); then
        printf "%s|%s" "$latest_file" "$latest_iter"
    else
        printf ""
    fi
}

get_restart_images() {
    local dir="$1"
    local pet=""
    local spect=""
    local pet_iter=-1
    local spect_iter=-1
    local offset=0
    if [[ "$RESTART_BASELINE" == "true" && -d "$dir" ]]; then
        local pet_info spect_info
        pet_info=$(find_latest_iteration_info "$dir" 0)
        spect_info=$(find_latest_iteration_info "$dir" 1)
        if [[ -n "$pet_info" && -n "$spect_info" ]]; then
            pet="${pet_info%%|*}"
            pet_iter="${pet_info##*|}"
            spect="${spect_info%%|*}"
            spect_iter="${spect_info##*|}"
            local max_iter=$pet_iter
            if (( spect_iter > max_iter )); then
                max_iter=$spect_iter
            fi
            offset=$((max_iter + 1))
        fi
    fi
    printf "%s|%s|%s" "$pet" "$spect" "$offset"
}

case "$MODE" in
    "local")
        # Run first alpha locally for testing
        ALPHA=${ALPHAS[0]}
        echo "Running baseline for alpha=$ALPHA locally..."
        
        OUTPUT_DIR="$OUTPUT_BASE/baseline_alpha_${ALPHA}"
        mkdir -p "$OUTPUT_DIR"

        restart_pair="$(get_restart_images "$OUTPUT_DIR")"
        IFS="|" read -r RESTART_PET_IMAGE_PATH RESTART_SPECT_IMAGE_PATH RESTART_ITERATION_OFFSET <<< "$restart_pair"
        if [[ -n "$RESTART_PET_IMAGE_PATH" && -n "$RESTART_SPECT_IMAGE_PATH" ]]; then
            echo "Restart: using PET initial image $RESTART_PET_IMAGE_PATH and SPECT initial image $RESTART_SPECT_IMAGE_PATH (next iter=$RESTART_ITERATION_OFFSET)"
        fi
        EXTRA_OVERRIDES=()
        if [[ -n "$RESTART_PET_IMAGE_PATH" ]]; then
            EXTRA_OVERRIDES+=( "--override" "pet_initial_image_path=$RESTART_PET_IMAGE_PATH" )
        fi
        if [[ -n "$RESTART_SPECT_IMAGE_PATH" ]]; then
            EXTRA_OVERRIDES+=( "--override" "spect_initial_image_path=$RESTART_SPECT_IMAGE_PATH" )
        fi
        EXTRA_OVERRIDES+=( "--override" "initial_iteration_offset=${RESTART_ITERATION_OFFSET:-0}" )

        cd "$BASE_DIR"
        python "$RUNNER_SCRIPT" \
            --config "configs/$BASE_CONFIG" \
            --override "num_epochs=$NUM_EPOCHS" \
            --override "alpha=$ALPHA" \
            --override "beta=$ALPHA" \
            --override "gamma_tnv=$GAMMA_TNV" \
            --override "precond_type=$PRECOND_TYPE" \
            --override "precond_combine=$PRECOND_COMBINE" \
            --override "block_scalar_reduction=$PRECOND_SCALAR_REDUCTION" \
            --override "precond_safety_scale=$PRECOND_SAFETY_SCALE" \
            --override "em_precond_cap_to_initial_max=$EM_PRECOND_CAP_TO_INITIAL_MAX" \
            --override "precond_cap_to_initial_max=$PRECOND_CAP_TO_INITIAL_MAX" \
            --override "support_mask_from_sensitivity=$SUPPORT_MASK_FROM_SENSITIVITY" \
            --override "support_mask_rel_threshold=$SUPPORT_MASK_REL_THRESHOLD" \
            --override "support_mask_abs_threshold=$SUPPORT_MASK_ABS_THRESHOLD" \
            --override "output_path=$OUTPUT_DIR" \
            "${EXTRA_OVERRIDES[@]}"
        
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

            restart_pair="$(get_restart_images "$OUTPUT_DIR")"
            IFS="|" read -r RESTART_PET_IMAGE_PATH RESTART_SPECT_IMAGE_PATH RESTART_ITERATION_OFFSET <<< "$restart_pair"
            EXTRA_OVERRIDES=()
            if [[ -n "$RESTART_PET_IMAGE_PATH" ]]; then
                EXTRA_OVERRIDES+=( "--override" "pet_initial_image_path=$RESTART_PET_IMAGE_PATH" )
            fi
            if [[ -n "$RESTART_SPECT_IMAGE_PATH" ]]; then
                EXTRA_OVERRIDES+=( "--override" "spect_initial_image_path=$RESTART_SPECT_IMAGE_PATH" )
            fi
            EXTRA_OVERRIDES+=( "--override" "initial_iteration_offset=${RESTART_ITERATION_OFFSET:-0}" )

            python "$RUNNER_SCRIPT" \
                --config "configs/$BASE_CONFIG" \
                --override "num_epochs=$NUM_EPOCHS" \
                --override "alpha=$ALPHA" \
                --override "beta=$ALPHA" \
                --override "gamma_tnv=$GAMMA_TNV" \
                --override "precond_type=$PRECOND_TYPE" \
                --override "precond_combine=$PRECOND_COMBINE" \
                --override "block_scalar_reduction=$PRECOND_SCALAR_REDUCTION" \
                --override "precond_safety_scale=$PRECOND_SAFETY_SCALE" \
                --override "em_precond_cap_to_initial_max=$EM_PRECOND_CAP_TO_INITIAL_MAX" \
                --override "precond_cap_to_initial_max=$PRECOND_CAP_TO_INITIAL_MAX" \
                --override "support_mask_from_sensitivity=$SUPPORT_MASK_FROM_SENSITIVITY" \
                --override "support_mask_rel_threshold=$SUPPORT_MASK_REL_THRESHOLD" \
                --override "support_mask_abs_threshold=$SUPPORT_MASK_ABS_THRESHOLD" \
                --override "output_path=$OUTPUT_DIR" \
                "${EXTRA_OVERRIDES[@]}"

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

        restart_pair="$(get_restart_images "$OUTPUT_DIR")"
        IFS="|" read -r RESTART_PET_IMAGE_PATH RESTART_SPECT_IMAGE_PATH RESTART_ITERATION_OFFSET <<< "$restart_pair"

        cd "$BASE_DIR"

        QSUB_ENV="BASE_DIR=$BASE_DIR,OUTPUT_DIR=$OUTPUT_DIR,BASE_CONFIG=$BASE_CONFIG,BPOS=$BPOS,ALPHA=$ALPHA,GAMMA_TNV=$GAMMA_TNV,NUM_EPOCHS=$NUM_EPOCHS,PRECOND_TYPE=$PRECOND_TYPE,PRECOND_COMBINE=$PRECOND_COMBINE,PRECOND_SCALAR_REDUCTION=$PRECOND_SCALAR_REDUCTION,PRECOND_SAFETY_SCALE=$PRECOND_SAFETY_SCALE,EM_PRECOND_CAP_TO_INITIAL_MAX=$EM_PRECOND_CAP_TO_INITIAL_MAX,PRECOND_CAP_TO_INITIAL_MAX=$PRECOND_CAP_TO_INITIAL_MAX,SUPPORT_MASK_FROM_SENSITIVITY=$SUPPORT_MASK_FROM_SENSITIVITY,SUPPORT_MASK_REL_THRESHOLD=$SUPPORT_MASK_REL_THRESHOLD,SUPPORT_MASK_ABS_THRESHOLD=$SUPPORT_MASK_ABS_THRESHOLD"
        if [[ -n "$RESTART_PET_IMAGE_PATH" ]]; then
            QSUB_ENV+=",PET_INITIAL_IMAGE_PATH=$RESTART_PET_IMAGE_PATH"
        fi
        if [[ -n "$RESTART_SPECT_IMAGE_PATH" ]]; then
            QSUB_ENV+=",SPECT_INITIAL_IMAGE_PATH=$RESTART_SPECT_IMAGE_PATH"
        fi
        QSUB_ENV+=",INITIAL_ITERATION_OFFSET=${RESTART_ITERATION_OFFSET:-0}"
        
        qsub \
            -r y \
            -l h_rt=72:00:00 \
            -l tmem=95G \
            -l gpu=true \
            -N "baseline_${BPOS}bpos_test" \
            -o "$LOG_DIR" \
            -e "$LOG_DIR" \
            -v "$QSUB_ENV" \
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
            mkdir -p "$OUTPUT_DIR"
            restart_pair="$(get_restart_images "$OUTPUT_DIR")"
            IFS="|" read -r RESTART_PET_IMAGE_PATH RESTART_SPECT_IMAGE_PATH RESTART_ITERATION_OFFSET <<< "$restart_pair"
            QSUB_ENV="BASE_DIR=$BASE_DIR,OUTPUT_DIR=$OUTPUT_DIR,BASE_CONFIG=$BASE_CONFIG,BPOS=$BPOS,ALPHA=$ALPHA,GAMMA_TNV=$GAMMA_TNV,NUM_EPOCHS=$NUM_EPOCHS,PRECOND_TYPE=$PRECOND_TYPE,PRECOND_COMBINE=$PRECOND_COMBINE,PRECOND_SCALAR_REDUCTION=$PRECOND_SCALAR_REDUCTION,PRECOND_SAFETY_SCALE=$PRECOND_SAFETY_SCALE,EM_PRECOND_CAP_TO_INITIAL_MAX=$EM_PRECOND_CAP_TO_INITIAL_MAX,PRECOND_CAP_TO_INITIAL_MAX=$PRECOND_CAP_TO_INITIAL_MAX,SUPPORT_MASK_FROM_SENSITIVITY=$SUPPORT_MASK_FROM_SENSITIVITY,SUPPORT_MASK_REL_THRESHOLD=$SUPPORT_MASK_REL_THRESHOLD,SUPPORT_MASK_ABS_THRESHOLD=$SUPPORT_MASK_ABS_THRESHOLD"
            if [[ -n "$RESTART_PET_IMAGE_PATH" ]]; then
                QSUB_ENV+=",PET_INITIAL_IMAGE_PATH=$RESTART_PET_IMAGE_PATH"
            fi
            if [[ -n "$RESTART_SPECT_IMAGE_PATH" ]]; then
                QSUB_ENV+=",SPECT_INITIAL_IMAGE_PATH=$RESTART_SPECT_IMAGE_PATH"
            fi
            QSUB_ENV+=",INITIAL_ITERATION_OFFSET=${RESTART_ITERATION_OFFSET:-0}"

            echo "Submitting baseline for alpha=$ALPHA..."
            
            qsub \
                -r y \
                -l h_rt=240:00:00 \
                -l tmem=95G \
                -l gpu=true \
                -N "baseline_${BPOS}bpos_alpha_${ALPHA}" \
                -o "$LOG_DIR" \
                -e "$LOG_DIR" \
                -v "$QSUB_ENV" \
                "$SCRIPTS_DIR/baseline_recon.qsub.sh"
        done
        
        echo ""
        echo "All baseline jobs submitted!"
        echo "Output: $OUTPUT_BASE"
        echo "Logs: $LOG_DIR"
        ;;
esac
