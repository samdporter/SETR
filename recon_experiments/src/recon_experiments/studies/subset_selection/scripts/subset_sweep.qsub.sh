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

    if [ -n "${OUTPUT_DIR:-}" ]; then
        cat > "$OUTPUT_DIR/job_completion.txt" <<EOF
subset_mode=${SUBSET_MODE:-unknown},prior_mode=${PRIOR_MODE:-unknown},precond_type=${PRECOND_TYPE:-unknown},gamma=${GAMMA:-unknown},status=failed,return_code=$exit_code,end_time=$(date),host=$HOSTNAME,failure_reason=$failure_reason,failure_type=$failure_type,start_time=$START_TIME
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
log_with_timestamp "Starting subset selection array task: ${TASK_ID}"
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
    BASE_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
fi

FUNC_DIR="$BASE_DIR/studies/subset_selection"
PARAM_DIR="$FUNC_DIR/parameters"
CONFIG_DIR="$FUNC_DIR/configs"
OUTPUT_BASE_DIR="$FUNC_DIR/output"
SCRIPTS_DIR="$FUNC_DIR/scripts"

SWEEP_NAME=${SWEEP_NAME:-subset_test}
BASE_CONFIG_FILE=${BASE_CONFIG_FILE:-base_config_anthro.yaml}
NUM_EPOCHS=${NUM_EPOCHS:-100}

log_with_timestamp "Base config: $BASE_CONFIG_FILE"
log_with_timestamp "Epochs: $NUM_EPOCHS"

# --- Read parameter files and compute indices ---
log_with_timestamp "Reading parameter files..."

# Initialize arrays and counts
declare -a SUBSET_MODES=()
declare -a PRIOR_MODES=()
declare -a PRECOND_TYPES=()
declare -a GAMMAS=()

NUM_SUBSET_MODES=1
NUM_PRIOR_MODES=1
NUM_PRECOND_TYPES=1
NUM_GAMMAS=1

# Read subset modes if file specified
if [ -n "${SUBSET_MODES_FILE:-}" ]; then
    if [ ! -f "$PARAM_DIR/$SUBSET_MODES_FILE" ]; then
        report_failure 1 "Subset modes file not found: $PARAM_DIR/$SUBSET_MODES_FILE" "config"
        exit 1
    fi
    mapfile -t SUBSET_MODES < <(tail -n +2 "$PARAM_DIR/$SUBSET_MODES_FILE" | awk 'NF{print $1}')
    NUM_SUBSET_MODES=${#SUBSET_MODES[@]}
    log_with_timestamp "Loaded $NUM_SUBSET_MODES subset modes"
fi

# Read prior modes if file specified
if [ -n "${PRIOR_MODES_FILE:-}" ]; then
    if [ ! -f "$PARAM_DIR/$PRIOR_MODES_FILE" ]; then
        report_failure 1 "Prior modes file not found: $PARAM_DIR/$PRIOR_MODES_FILE" "config"
        exit 1
    fi
    mapfile -t PRIOR_MODES < <(tail -n +2 "$PARAM_DIR/$PRIOR_MODES_FILE" | awk 'NF{print $1}')
    NUM_PRIOR_MODES=${#PRIOR_MODES[@]}
    log_with_timestamp "Loaded $NUM_PRIOR_MODES prior modes"
fi

# Read precond types if file specified
if [ -n "${PRECOND_TYPES_FILE:-}" ]; then
    if [ ! -f "$PARAM_DIR/$PRECOND_TYPES_FILE" ]; then
        report_failure 1 "Precond types file not found: $PARAM_DIR/$PRECOND_TYPES_FILE" "config"
        exit 1
    fi
    mapfile -t PRECOND_TYPES < <(tail -n +2 "$PARAM_DIR/$PRECOND_TYPES_FILE" | awk 'NF{print $1}')
    NUM_PRECOND_TYPES=${#PRECOND_TYPES[@]}
    log_with_timestamp "Loaded $NUM_PRECOND_TYPES preconditioner types"
fi

# Read gammas if file specified
if [ -n "${GAMMAS_FILE:-}" ]; then
    if [ ! -f "$PARAM_DIR/$GAMMAS_FILE" ]; then
        report_failure 1 "Gammas file not found: $PARAM_DIR/$GAMMAS_FILE" "config"
        exit 1
    fi
    mapfile -t GAMMAS < <(tail -n +2 "$PARAM_DIR/$GAMMAS_FILE" | awk 'NF{print $1}')
    NUM_GAMMAS=${#GAMMAS[@]}
    log_with_timestamp "Loaded $NUM_GAMMAS gamma values"
fi

# Calculate multi-dimensional indices
# Iteration order: gamma (fastest) -> precond -> prior -> subset (slowest)
GAMMA_INDEX=$(( (TASK_ID - 1) % NUM_GAMMAS ))
PRECOND_INDEX=$(( ((TASK_ID - 1) / NUM_GAMMAS) % NUM_PRECOND_TYPES ))
PRIOR_INDEX=$(( ((TASK_ID - 1) / (NUM_GAMMAS * NUM_PRECOND_TYPES)) % NUM_PRIOR_MODES ))
SUBSET_INDEX=$(( (TASK_ID - 1) / (NUM_GAMMAS * NUM_PRECOND_TYPES * NUM_PRIOR_MODES) ))

# Check if task exceeds parameter space
TOTAL_COMBINATIONS=$((NUM_SUBSET_MODES * NUM_PRIOR_MODES * NUM_PRECOND_TYPES * NUM_GAMMAS))
if [ $TASK_ID -gt $TOTAL_COMBINATIONS ]; then
    log_with_timestamp "Task ID $TASK_ID exceeds available parameter combinations ($TOTAL_COMBINATIONS). Exiting."
    exit 0
fi

# Get parameter values (use fixed values if specified, otherwise use indexed values)
if [ -n "${FIXED_SUBSET_MODE:-}" ]; then
    SUBSET_MODE="$FIXED_SUBSET_MODE"
elif [ ${#SUBSET_MODES[@]} -gt 0 ]; then
    SUBSET_MODE=${SUBSET_MODES[$SUBSET_INDEX]}
else
    SUBSET_MODE="separate"  # default
fi

if [ -n "${FIXED_PRIOR_MODE:-}" ]; then
    PRIOR_MODE="$FIXED_PRIOR_MODE"
elif [ ${#PRIOR_MODES[@]} -gt 0 ]; then
    PRIOR_MODE=${PRIOR_MODES[$PRIOR_INDEX]}
else
    PRIOR_MODE="always"  # default
fi

if [ -n "${FIXED_PRECOND_TYPE:-}" ]; then
    PRECOND_TYPE="$FIXED_PRECOND_TYPE"
elif [ ${#PRECOND_TYPES[@]} -gt 0 ]; then
    PRECOND_TYPE=${PRECOND_TYPES[$PRECOND_INDEX]}
else
    PRECOND_TYPE="bsrem"  # default
fi

if [ ${#GAMMAS[@]} -gt 0 ]; then
    GAMMA=${GAMMAS[$GAMMA_INDEX]}
else
    GAMMA="50"  # default
fi

log_with_timestamp "Task $TASK_ID: subset_mode=$SUBSET_MODE, prior_mode=$PRIOR_MODE, precond_type=$PRECOND_TYPE, gamma=$GAMMA"

# --- Paths for this job ---
OUTPUT_DIR="$OUTPUT_BASE_DIR/${SWEEP_NAME}/subset_${SUBSET_MODE}_prior_${PRIOR_MODE}_precond_${PRECOND_TYPE}_gamma_${GAMMA}"
WORKING_DIR="$OUTPUT_DIR/tmp"

log_with_timestamp "Creating output directories..."
if ! mkdir -p "$OUTPUT_DIR" "$WORKING_DIR"; then
    report_failure 1 "Failed to create output directories" "filesystem"
    exit 1
fi

# --- Run reconstruction ---
cd "$BASE_DIR"

if [ ! -f "$SCRIPTS_DIR/run_subset_selection.py" ]; then
    report_failure 1 "Reconstruction script not found: $SCRIPTS_DIR/run_subset_selection.py" "config"
    exit 1
fi

MAX_RETRIES=2
RETURN_CODE=1

for attempt in $(seq 1 $MAX_RETRIES); do
    log_with_timestamp "Reconstruction attempt $attempt of $MAX_RETRIES"

    attempt_start_time=$(date '+%Y-%m-%d %H:%M:%S')
    log_with_timestamp "Starting reconstruction at: $attempt_start_time"

    if python "$SCRIPTS_DIR/run_subset_selection.py" \
        --config "$CONFIG_DIR/$BASE_CONFIG_FILE" \
        --override \
            output_path="$OUTPUT_DIR" \
            working_path="$WORKING_DIR" \
            num_epochs="$NUM_EPOCHS" \
            subset_mode="$SUBSET_MODE" \
            prior_mode="$PRIOR_MODE" \
            precond_type="$PRECOND_TYPE" \
            gamma_tnv="$GAMMA"; then
        RETURN_CODE=0
        log_with_timestamp "Reconstruction completed successfully"
        break
    else
        RETURN_CODE=$?
        attempt_end_time=$(date '+%Y-%m-%d %H:%M:%S')
        log_with_timestamp "Attempt $attempt failed with code $RETURN_CODE at: $attempt_end_time"

        if [ $RETURN_CODE -eq 137 ]; then
            log_with_timestamp "Reconstruction was killed (likely OOM)"
            report_failure $RETURN_CODE "Reconstruction killed (likely out of memory)" "memory"
            break
        elif [ $attempt -eq $MAX_RETRIES ]; then
            report_failure $RETURN_CODE "Reconstruction failed after $MAX_RETRIES attempts" "recon_error"
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
subset_mode=$SUBSET_MODE,prior_mode=$PRIOR_MODE,precond_type=$PRECOND_TYPE,gamma=$GAMMA,status=completed,end_time=$END_TIME,host=$HOSTNAME,start_time=$START_TIME
EOF
else
    log_with_timestamp "Job failed with return code: $RETURN_CODE at: $END_TIME"

    if [ ! -f "$OUTPUT_DIR/job_completion.txt" ]; then
        report_failure $RETURN_CODE "Final reconstruction failure" "recon_error"
    fi
fi

log_with_timestamp "Task ${TASK_ID} finished with return code: $RETURN_CODE"
exit $RETURN_CODE
