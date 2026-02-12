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
PRECOND_TYPE="${PRECOND_TYPE:-mm_block_diag}"
PRECOND_COMBINE="${PRECOND_COMBINE:-harmonic}"
PRECOND_SCALAR_REDUCTION="${PRECOND_SCALAR_REDUCTION:-diag}"
log_with_timestamp "Preconditioner: $PRECOND_TYPE"
log_with_timestamp "Combine: $PRECOND_COMBINE"
log_with_timestamp "Block scalar reduction: $PRECOND_SCALAR_REDUCTION"

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

RUNNER_SCRIPT="$BASE_DIR/src/recon_experiments/runners/scripts/run_dtnv_1bpos.py"
if [ ! -f "$RUNNER_SCRIPT" ]; then
    echo "Error: Reconstruction script not found: $RUNNER_SCRIPT"
    exit 1
fi

log_with_timestamp "Running baseline reconstruction..."

if python "$RUNNER_SCRIPT" \
    --config "configs/$BASE_CONFIG" \
    --override "num_epochs=$NUM_EPOCHS" \
    --override "alpha=$ALPHA" \
    --override "beta=$ALPHA" \
    --override "precond_type=$PRECOND_TYPE" \
    --override "precond_combine=$PRECOND_COMBINE" \
    --override "block_scalar_reduction=$PRECOND_SCALAR_REDUCTION" \
    --override "output_path=$OUTPUT_DIR"; then
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
alpha=$ALPHA,precond_type=$PRECOND_TYPE,epochs=$NUM_EPOCHS,status=completed,end_time=$END_TIME,host=$HOSTNAME,start_time=$START_TIME
EOF
else
    log_with_timestamp "Job failed with return code: $RETURN_CODE at: $END_TIME"
    
    cat > "$OUTPUT_DIR/job_completion.txt" <<EOF
alpha=$ALPHA,precond_type=$PRECOND_TYPE,epochs=$NUM_EPOCHS,status=failed,return_code=$RETURN_CODE,end_time=$END_TIME,host=$HOSTNAME,start_time=$START_TIME
EOF
fi

log_with_timestamp "Baseline job finished with return code: $RETURN_CODE"
exit $RETURN_CODE
