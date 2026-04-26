#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
set -euo pipefail

# Enhanced error handling and logging
HOSTNAME=$(hostname)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
TASK_ID=${SGE_TASK_ID:-1}

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Function to report job failure
report_failure() {
    exit_code="$1"
    failure_reason="$2"
    failure_type="${3:-unknown}"

    log_with_timestamp "JOB FAILURE: Exit code $exit_code"
    log_with_timestamp "Failure reason: $failure_reason"
    log_with_timestamp "Failure type: $failure_type"
    log_with_timestamp "Host: $HOSTNAME"
    log_with_timestamp "Task ID: $TASK_ID"

    # Write detailed failure info to completion file
    if [ -n "${OUTPUT_DIR:-}" ]; then
        cat > "$OUTPUT_DIR/job_completion.txt" <<EOF
precond_type=${PRECOND_TYPE:-unknown},combine=${PRECOND_COMBINE:-},alpha=${ALPHA:-unknown},step_size=${STEP_SIZE:-unknown},repeat=${REPEAT:-1},status=failed,return_code=$exit_code,end_time=$(date),host=$HOSTNAME,failure_reason=$failure_reason,failure_type=$failure_type,start_time=$START_TIME
EOF
    fi
}

# Check GPU health
check_gpu_health() {
    if ! command -v nvidia-smi >/dev/null; then
        return 0
    fi

    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        return 0
    fi

    log_with_timestamp "Checking GPU health for devices: $CUDA_VISIBLE_DEVICES"

    IFS=',' read -ra __GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    for __id in "${__GPU_IDS[@]}"; do
        ecc_errors=$(nvidia-smi -i "${__id}" --query-gpu=ecc.errors.uncorrected.total --format=csv,noheader 2>/dev/null || echo "N/A")

        log_with_timestamp "GPU ${__id} ECC errors: $ecc_errors"

        if [[ "$ecc_errors" =~ ^[0-9]+$ ]] && [ "$ecc_errors" -gt 0 ]; then
            report_failure 99 "GPU ${__id} reports uncorrectable ECC errors: $ecc_errors" "gpu_ecc_error"
            log_with_timestamp "GPU ${__id} reports uncorrectable ECC errors. Exiting."
            exit 99
        fi
    done

    log_with_timestamp "GPU health check passed"
}

# Perform initial health checks
log_with_timestamp "Starting preconditioner test array task: ${TASK_ID}"
log_with_timestamp "Host: $HOSTNAME"
log_with_timestamp "Start time: $START_TIME"

check_gpu_health

# --- Runtime env: activate venv + SIRF ---
log_with_timestamp "Setting up runtime environment..."

if [ ! -f "$HOME/sirf_venv/bin/activate" ]; then
    report_failure 1 "SIRF virtual environment not found" "environment"
    exit 1
fi

source "$HOME/sirf_venv/bin/activate"

export INSTALLDIR=/home/sporter/synergistic_Y90/devel/SIRF/SIRF_installs/Release-cuda12.0

if [ ! -f "${INSTALLDIR}/bin/env_sirf.sh" ]; then
    report_failure 1 "SIRF installation not found" "environment"
    exit 1
fi

source "${INSTALLDIR}/bin/env_sirf.sh"

# Remove any source-tree CIL path
if [ -n "${PYTHONPATH:-}" ]; then
  PYTHONPATH="$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v 'CIL/Wrappers/Python' | paste -sd':' -)"
  export PYTHONPATH
fi

# --- Layout from launcher ---
if [ -n "${SETR_BASE_DIR:-}" ]; then
    BASE_DIR="$SETR_BASE_DIR"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    BASE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi

FUNC_DIR="$BASE_DIR/src/recon_experiments/studies/preconditioners"
PARAM_DIR="$FUNC_DIR/parameters"
CONFIG_DIR="$FUNC_DIR/configs"
OUTPUT_BASE_DIR="$FUNC_DIR/output"
SCRIPTS_DIR="$FUNC_DIR/scripts"

SWEEP_NAME=${SWEEP_NAME:-precond_test}
BASE_CONFIG_FILE=${BASE_CONFIG_FILE:-config_2bpos.yaml}
PRECOND_TYPES_FILE=${PRECOND_TYPES_FILE:-precond_types.csv}
ALPHAS_FILE=${ALPHAS_FILE:-alphas.csv}
STEP_SIZES_FILE=${STEP_SIZES_FILE:-step_sizes.csv}
NUM_EPOCHS=${NUM_EPOCHS:-50}
RECON_SCRIPT=${RECON_SCRIPT:-run_precond_sweep_single.py}
SWEEP_REPEATS=${SWEEP_REPEATS:-1}

if ! [[ "$SWEEP_REPEATS" =~ ^[0-9]+$ ]] || [ "$SWEEP_REPEATS" -lt 1 ]; then
    report_failure 1 "Invalid SWEEP_REPEATS='$SWEEP_REPEATS' (must be positive integer)" "config"
    exit 1
fi

log_with_timestamp "Base config: $BASE_CONFIG_FILE"
log_with_timestamp "Epochs: $NUM_EPOCHS"
log_with_timestamp "Sweep repeats: $SWEEP_REPEATS"

# --- Read parameters (4-way combination: precond_type × alpha × step_size × repeat) ---
log_with_timestamp "Reading parameter files..."

if [ ! -f "$PARAM_DIR/$PRECOND_TYPES_FILE" ]; then
    report_failure 1 "Precond types file not found: $PARAM_DIR/$PRECOND_TYPES_FILE" "config"
    exit 1
fi

if [ ! -f "$PARAM_DIR/$ALPHAS_FILE" ]; then
    report_failure 1 "Alphas file not found: $PARAM_DIR/$ALPHAS_FILE" "config"
    exit 1
fi

if [ ! -f "$PARAM_DIR/$STEP_SIZES_FILE" ]; then
    report_failure 1 "Step sizes file not found: $PARAM_DIR/$STEP_SIZES_FILE" "config"
    exit 1
fi

mapfile -t PRECOND_TYPES < <(tail -n +2 "$PARAM_DIR/$PRECOND_TYPES_FILE" | awk -F, 'NF{gsub(/^[ \t]+|[ \t]+$/,"",$1); gsub(/^[ \t]+|[ \t]+$/,"",$2); if($1!="") print $1 "," $2}')
mapfile -t ALPHAS < <(tail -n +2 "$PARAM_DIR/$ALPHAS_FILE" | awk -F, 'NF{gsub(/^[ \t]+|[ \t]+$/,"",$1); if($1!="") print $1}')
mapfile -t STEP_SIZES < <(tail -n +2 "$PARAM_DIR/$STEP_SIZES_FILE" | awk -F, 'NF{gsub(/^[ \t]+|[ \t]+$/,"",$1); if($1!="") print $1}')

NUM_PRECOND_TYPES=${#PRECOND_TYPES[@]}
NUM_ALPHAS=${#ALPHAS[@]}
NUM_STEP_SIZES=${#STEP_SIZES[@]}

log_with_timestamp "Total precond types: $NUM_PRECOND_TYPES"
log_with_timestamp "Total alphas: $NUM_ALPHAS"
log_with_timestamp "Total step sizes: $NUM_STEP_SIZES"
TOTAL_BASE_COMBINATIONS=$((NUM_PRECOND_TYPES * NUM_ALPHAS * NUM_STEP_SIZES))
TOTAL_TASKS=$((TOTAL_BASE_COMBINATIONS * SWEEP_REPEATS))
log_with_timestamp "Total base combinations: $TOTAL_BASE_COMBINATIONS"
log_with_timestamp "Total tasks with repeats: $TOTAL_TASKS"

if [ "$NUM_PRECOND_TYPES" -eq 0 ] || [ "$NUM_ALPHAS" -eq 0 ] || [ "$NUM_STEP_SIZES" -eq 0 ]; then
    report_failure 1 "No parameters found in parameter files" "config"
    exit 0
fi

# Calculate 4-way indices
# Task layout: repeat (fastest), then step_sizes, then alphas, then precond_types (slowest)
if [ "$TASK_ID" -gt "$TOTAL_TASKS" ]; then
    log_with_timestamp "Task ID $TASK_ID exceeds available parameter combinations. Exiting."
    exit 0
fi

REPEAT_INDEX=$(( (TASK_ID - 1) % SWEEP_REPEATS ))
COMBINATION_INDEX=$(( (TASK_ID - 1) / SWEEP_REPEATS ))
STEP_SIZE_INDEX=$(( COMBINATION_INDEX % NUM_STEP_SIZES ))
ALPHA_INDEX=$(( (COMBINATION_INDEX / NUM_STEP_SIZES) % NUM_ALPHAS ))
PRECOND_TYPE_INDEX=$(( COMBINATION_INDEX / (NUM_STEP_SIZES * NUM_ALPHAS) ))

PRECOND_LINE=${PRECOND_TYPES[$PRECOND_TYPE_INDEX]}
IFS=',' read -r PRECOND_TYPE PRECOND_COMBINE <<< "$PRECOND_LINE"
PRECOND_TYPE="${PRECOND_TYPE//[$'\t\r\n ']/}"
PRECOND_COMBINE="${PRECOND_COMBINE//[$'\t\r\n ']/}"
ALPHA=${ALPHAS[$ALPHA_INDEX]}
STEP_SIZE=${STEP_SIZES[$STEP_SIZE_INDEX]}
REPEAT=$((REPEAT_INDEX + 1))

log_with_timestamp "Task $TASK_ID: precond_type=$PRECOND_TYPE, combine=$PRECOND_COMBINE, alpha=$ALPHA, step_size=$STEP_SIZE, repeat=$REPEAT/$SWEEP_REPEATS"

# --- Paths for this job ---
if [ -n "$PRECOND_COMBINE" ]; then
    OUTPUT_DIR="$OUTPUT_BASE_DIR/${SWEEP_NAME}/precond_${PRECOND_TYPE}_combine_${PRECOND_COMBINE}_alpha_${ALPHA}_step_${STEP_SIZE}"
else
    OUTPUT_DIR="$OUTPUT_BASE_DIR/${SWEEP_NAME}/precond_${PRECOND_TYPE}_alpha_${ALPHA}_step_${STEP_SIZE}"
fi
if [ "$SWEEP_REPEATS" -gt 1 ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_rep_${REPEAT}"
fi
WORKING_DIR="$OUTPUT_DIR/tmp"

log_with_timestamp "Creating output directories..."
if ! mkdir -p "$OUTPUT_DIR" "$WORKING_DIR"; then
    report_failure 1 "Failed to create output directories" "filesystem"
    exit 1
fi

# --- Run reconstruction ---
cd "$BASE_DIR"

if [ ! -f "$SCRIPTS_DIR/$RECON_SCRIPT" ]; then
    report_failure 1 "Test script not found: $SCRIPTS_DIR/$RECON_SCRIPT" "config"
    exit 1
fi

MAX_RETRIES=2
RETURN_CODE=1

for attempt in $(seq 1 $MAX_RETRIES); do
    log_with_timestamp "Test attempt $attempt of $MAX_RETRIES"

    attempt_start_time=$(date '+%Y-%m-%d %H:%M:%S')
    log_with_timestamp "Starting test at: $attempt_start_time"

    OVERRIDE_ARGS=()
    if [ -n "${FIXED_OVERRIDES:-}" ]; then
        IFS=';' read -ra FIXED_LIST <<< "$FIXED_OVERRIDES"
        for ov in "${FIXED_LIST[@]}"; do
            [ -n "$ov" ] && OVERRIDE_ARGS+=(--override "$ov")
        done
    fi
    COMBINE_ARGS=(--precond-combine "$PRECOND_COMBINE")

    if python "$SCRIPTS_DIR/$RECON_SCRIPT" \
        --config "$BASE_DIR/configs/$BASE_CONFIG_FILE" \
        --output "$OUTPUT_DIR" \
        --precond-type "$PRECOND_TYPE" \
        --alpha "$ALPHA" \
        --step-size "$STEP_SIZE" \
        --epochs "$NUM_EPOCHS" \
        "${COMBINE_ARGS[@]}" \
        "${OVERRIDE_ARGS[@]}"; then
        RETURN_CODE=0
        log_with_timestamp "Test completed successfully"
        break
    else
        RETURN_CODE=$?
        attempt_end_time=$(date '+%Y-%m-%d %H:%M:%S')
        log_with_timestamp "Attempt $attempt failed with code $RETURN_CODE at: $attempt_end_time"

        if [ $RETURN_CODE -eq 137 ]; then
            log_with_timestamp "Test was killed (likely OOM)"
            report_failure $RETURN_CODE "Test killed (likely out of memory)" "memory"
            break
        elif [ $attempt -eq $MAX_RETRIES ]; then
            report_failure $RETURN_CODE "Test failed after $MAX_RETRIES attempts" "test_error"
        fi

        if [ $attempt -lt $MAX_RETRIES ]; then
            log_with_timestamp "Waiting 60 seconds before retry..."
            sleep 60
        fi
    fi
done

# --- Post-processing ---
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

if [ $RETURN_CODE -eq 0 ]; then
    log_with_timestamp "Job completed successfully at: $END_TIME"

    cat > "$OUTPUT_DIR/job_completion.txt" <<EOF
precond_type=$PRECOND_TYPE,combine=$PRECOND_COMBINE,alpha=$ALPHA,step_size=$STEP_SIZE,repeat=$REPEAT,status=completed,end_time=$END_TIME,host=$HOSTNAME,start_time=$START_TIME
EOF
else
    log_with_timestamp "Job failed with return code: $RETURN_CODE at: $END_TIME"

    if [ ! -f "$OUTPUT_DIR/job_completion.txt" ]; then
        report_failure $RETURN_CODE "Final test failure" "test_error"
    fi
fi

log_with_timestamp "Task ${TASK_ID} finished with return code: $RETURN_CODE"
exit $RETURN_CODE
