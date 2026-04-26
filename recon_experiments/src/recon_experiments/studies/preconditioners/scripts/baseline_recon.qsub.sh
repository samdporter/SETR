#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
set -euo pipefail

# SGE job script for baseline reconstruction

HOSTNAME=$(hostname)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_with_timestamp "Starting baseline reconstruction"
log_with_timestamp "Host: $HOSTNAME"
log_with_timestamp "Start time: $START_TIME"
log_with_timestamp "Alpha: $ALPHA"
log_with_timestamp "Epochs: $NUM_EPOCHS"
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
log_with_timestamp "Preconditioner: $PRECOND_TYPE"
log_with_timestamp "Combine: $PRECOND_COMBINE"
log_with_timestamp "Block scalar reduction: $PRECOND_SCALAR_REDUCTION"
log_with_timestamp "Preconditioner safety scale: $PRECOND_SAFETY_SCALE"
log_with_timestamp "EM preconditioner cap to initial max: $EM_PRECOND_CAP_TO_INITIAL_MAX"
log_with_timestamp "Combined preconditioner cap to initial max: $PRECOND_CAP_TO_INITIAL_MAX"
log_with_timestamp "Gamma TNV: $GAMMA_TNV"
log_with_timestamp "Support mask from sensitivity: $SUPPORT_MASK_FROM_SENSITIVITY"
log_with_timestamp "Support mask relative threshold: $SUPPORT_MASK_REL_THRESHOLD"
log_with_timestamp "Support mask absolute threshold: $SUPPORT_MASK_ABS_THRESHOLD"

# --- Runtime env: activate venv + SIRF ---
log_with_timestamp "Setting up runtime environment..."

if [ ! -f "$HOME/sirf_venv/bin/activate" ]; then
    echo "Error: SIRF virtual environment not found"
    exit 1
fi

source "$HOME/sirf_venv/bin/activate"

export INSTALLDIR=/home/sporter/synergistic_Y90/devel/SIRF/SIRF_installs/Release-cuda12.0

if [ ! -f "${INSTALLDIR}/bin/env_sirf.sh" ]; then
    echo "Error: SIRF installation not found"
    exit 1
fi

source "${INSTALLDIR}/bin/env_sirf.sh"

# Remove any source-tree CIL path
if [ -n "${PYTHONPATH:-}" ]; then
  PYTHONPATH="$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v 'CIL/Wrappers/Python' | paste -sd':' -)"
  export PYTHONPATH
fi

# --- Run baseline reconstruction ---
cd "$BASE_DIR"

mkdir -p "$OUTPUT_DIR"

BPOS="${BPOS:-}"
if [[ -z "$BPOS" ]]; then
    config_name="$(basename "$BASE_CONFIG" | tr '[:upper:]' '[:lower:]')"
    if [[ "$config_name" == *"2bpos"* ]]; then
        BPOS=2
    elif [[ "$config_name" == *"1bpos"* ]]; then
        BPOS=1
    else
        echo "Error: Could not infer bed positions from BASE_CONFIG=$BASE_CONFIG"
        exit 1
    fi
fi

RUNNER_SCRIPT="$BASE_DIR/src/recon_experiments/runners/scripts/run_dtnv_${BPOS}bpos.py"
if [ ! -f "$RUNNER_SCRIPT" ]; then
    echo "Error: Reconstruction script not found: $RUNNER_SCRIPT"
    exit 1
fi
log_with_timestamp "Bed positions: $BPOS"

    EXTRA_OVERRIDES=()
    if [[ -n "${PET_INITIAL_IMAGE_PATH:-}" ]]; then
        EXTRA_OVERRIDES+=( "--override" "pet_initial_image_path=$PET_INITIAL_IMAGE_PATH" )
        log_with_timestamp "Restart override PET initial image: $PET_INITIAL_IMAGE_PATH"
    fi
    if [[ -n "${SPECT_INITIAL_IMAGE_PATH:-}" ]]; then
        EXTRA_OVERRIDES+=( "--override" "spect_initial_image_path=$SPECT_INITIAL_IMAGE_PATH" )
        log_with_timestamp "Restart override SPECT initial image: $SPECT_INITIAL_IMAGE_PATH"
    fi
    if [[ -n "${INITIAL_ITERATION_OFFSET:-}" ]]; then
        EXTRA_OVERRIDES+=( "--override" "initial_iteration_offset=$INITIAL_ITERATION_OFFSET" )
        if [[ "${INITIAL_ITERATION_OFFSET}" -ne 0 ]]; then
            log_with_timestamp "Restart iteration offset: $INITIAL_ITERATION_OFFSET"
        fi
    fi

    log_with_timestamp "Running baseline reconstruction..."

    if python "$RUNNER_SCRIPT" \
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
        "${EXTRA_OVERRIDES[@]}"; then
    RETURN_CODE=0
    log_with_timestamp "Baseline reconstruction completed successfully"
    
    # Create baseline metrics file for analysis compatibility
    if [ -f "$OUTPUT_DIR/result.csv" ]; then
        python3 -c "
import json
import pandas as pd
try:
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
except Exception as e:
    print(f'Warning: Could not create baseline metrics: {e}')
"
    fi
else
    RETURN_CODE=$?
    log_with_timestamp "Baseline reconstruction failed with code $RETURN_CODE"
fi

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

if [ $RETURN_CODE -eq 0 ]; then
    log_with_timestamp "Job completed successfully at: $END_TIME"
    
    cat > "$OUTPUT_DIR/job_completion.txt" <<EOF
alpha=$ALPHA,bpos=$BPOS,precond_type=$PRECOND_TYPE,epochs=$NUM_EPOCHS,status=completed,end_time=$END_TIME,host=$HOSTNAME,start_time=$START_TIME
EOF
else
    log_with_timestamp "Job failed with return code: $RETURN_CODE at: $END_TIME"
    
    cat > "$OUTPUT_DIR/job_completion.txt" <<EOF
alpha=$ALPHA,bpos=$BPOS,precond_type=$PRECOND_TYPE,epochs=$NUM_EPOCHS,status=failed,return_code=$RETURN_CODE,end_time=$END_TIME,host=$HOSTNAME,start_time=$START_TIME
EOF
fi

log_with_timestamp "Baseline job finished with return code: $RETURN_CODE"
exit $RETURN_CODE
