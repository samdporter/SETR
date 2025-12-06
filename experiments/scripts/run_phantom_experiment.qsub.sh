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

# Function to report job failure with detailed info
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
phantom=${PHANTOM:-unknown},algorithm=${ALGORITHM:-unknown},status=failed,return_code=$exit_code,end_time=$(date),host=$HOSTNAME,failure_reason=$failure_reason,failure_type=$failure_type,start_time=$START_TIME
EOF
    fi
}

# Enhanced GPU ECC check with better error reporting
check_gpu_health() {
    if ! command -v nvidia-smi >/dev/null; then
        return 0  # No GPU, skip check
    fi

    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        return 0  # No assigned GPUs
    fi

    log_with_timestamp "Checking GPU health for devices: $CUDA_VISIBLE_DEVICES"

    IFS=',' read -ra __GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    for __id in "${__GPU_IDS[@]}"; do
        ecc_errors=$(nvidia-smi -i "${__id}" --query-gpu=ecc.errors.uncorrected.total --format=csv,noheader 2>/dev/null || echo "N/A")

        log_with_timestamp "GPU ${__id} ECC errors: $ecc_errors"

        # Only check for ECC errors if we got a valid numeric result
        if [[ "$ecc_errors" =~ ^[0-9]+$ ]] && [ "$ecc_errors" -gt 0 ]; then
            report_failure 99 "GPU ${__id} reports uncorrectable ECC errors: $ecc_errors" "gpu_ecc_error"
            log_with_timestamp "GPU ${__id} reports uncorrectable ECC errors. Exiting."
            sleep 20
            exit 99
        fi
    done

    log_with_timestamp "GPU health check passed"
}

# Check system resources
check_system_resources() {
    log_with_timestamp "Checking system resources..."

    # Check memory
    mem_info=$(free -h | grep "^Mem:")
    log_with_timestamp "Memory: $mem_info"

    # Check disk space
    disk_info=$(df -h . | tail -1)
    log_with_timestamp "Disk space (working dir): $disk_info"

    # Check if we're running low on disk space (less than 1GB free)
    available_kb=$(df . | tail -1 | awk '{print $4}')
    if [ "$available_kb" -lt 1048576 ]; then  # 1GB in KB
        report_failure 1 "Low disk space: ${available_kb}KB available" "disk_space"
        log_with_timestamp "ERROR: Low disk space detected"
        exit 1
    fi
}

# Perform initial health checks
log_with_timestamp "Starting phantom experiment array task: ${TASK_ID}"
log_with_timestamp "Host: $HOSTNAME"
log_with_timestamp "Start time: $START_TIME"

check_gpu_health
check_system_resources

# --- Runtime env: activate venv + SIRF ---
log_with_timestamp "Setting up runtime environment..."

if [ ! -f "$HOME/sirf_venv/bin/activate" ]; then
    report_failure 1 "SIRF virtual environment not found" "environment"
    log_with_timestamp "ERROR: SIRF virtual environment not found at $HOME/sirf_venv/bin/activate"
    exit 1
fi

source "$HOME/sirf_venv/bin/activate"

export INSTALLDIR=/home/sporter/synergistic_Y90/devel/SIRF/SIRF_installs/Release-cuda12.0

if [ ! -f "${INSTALLDIR}/bin/env_sirf.sh" ]; then
    report_failure 1 "SIRF installation not found" "environment"
    log_with_timestamp "ERROR: SIRF installation not found at ${INSTALLDIR}/bin/env_sirf.sh"
    exit 1
fi

source "${INSTALLDIR}/bin/env_sirf.sh"

# Remove any source-tree CIL path to avoid shadowing the wheel install
if [ -n "${PYTHONPATH:-}" ]; then
  PYTHONPATH="$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v 'CIL/Wrappers/Python' | paste -sd':' -)"
  export PYTHONPATH
fi

# Pre-flight: confirm correct CIL and that libcilacc exists
log_with_timestamp "Verifying Python environment..."
if ! python - <<'PY'
import sys, pathlib, importlib
print("python:", sys.executable)
try:
    import cil
    libdir = pathlib.Path(cil.__file__).parent / "lib"
    print("cil   :", cil.__file__)
    cilacc_libs = list(libdir.glob("*cilacc*"))
    print("cilacc:", cilacc_libs)
    if not cilacc_libs:
        print("WARNING: No cilacc libraries found")
    else:
        print("Found cilacc libraries:", len(cilacc_libs))
except Exception as e:
    print("E: import cil failed ->", e)
    raise
PY
then
    report_failure 1 "Python environment verification failed" "environment"
    log_with_timestamp "ERROR: Python environment verification failed"
    exit 1
fi

# --- Layout from launcher (or infer) ---
if [ -n "${SETR_BASE_DIR:-}" ]; then
    BASE_DIR="$SETR_BASE_DIR"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    BASE_DIR="$(dirname "$SCRIPT_DIR")"
fi

EXPERIMENTS_DIR="$BASE_DIR/experiments"
SCRIPTS_DIR="$BASE_DIR/scripts"

# Get sweep configuration from environment variables
SWEEP_NAME=${SWEEP_NAME:-phantom_experiments}
PHANTOMS_STR=${PHANTOMS:-"manc,anthro,nema"}
ALGORITHMS_STR=${ALGORITHMS:-"hkem,dtnv,tnv,log_dtnv,log_tnv"}

# Parse comma-separated lists into arrays
IFS=',' read -ra PHANTOMS <<< "$PHANTOMS_STR"
IFS=',' read -ra ALGORITHMS <<< "$ALGORITHMS_STR"

NUM_PHANTOMS=${#PHANTOMS[@]}
NUM_ALGORITHMS=${#ALGORITHMS[@]}
TOTAL_JOBS=$((NUM_PHANTOMS * NUM_ALGORITHMS))

log_with_timestamp "Phantoms: ${PHANTOMS[*]}"
log_with_timestamp "Algorithms: ${ALGORITHMS[*]}"
log_with_timestamp "Total combinations: $NUM_PHANTOMS phantoms × $NUM_ALGORITHMS algorithms = $TOTAL_JOBS jobs"

# Calculate phantom and algorithm index for this task
PHANTOM_INDEX=$(( (TASK_ID - 1) / NUM_ALGORITHMS ))
ALGORITHM_INDEX=$(( (TASK_ID - 1) % NUM_ALGORITHMS ))

if [ $PHANTOM_INDEX -ge $NUM_PHANTOMS ]; then
    log_with_timestamp "Task ID $TASK_ID exceeds available combinations. Exiting."
    exit 0
fi

PHANTOM=${PHANTOMS[$PHANTOM_INDEX]}
ALGORITHM=${ALGORITHMS[$ALGORITHM_INDEX]}

log_with_timestamp "Task $TASK_ID: Running phantom=$PHANTOM, algorithm=$ALGORITHM"

# --- Paths for this job ---
OUTPUT_BASE_DIR="$EXPERIMENTS_DIR/output"
OUTPUT_DIR="$OUTPUT_BASE_DIR/$SWEEP_NAME/${PHANTOM}_${ALGORITHM}"
WORKING_DIR="$OUTPUT_DIR/tmp"

log_with_timestamp "Creating output directories..."
if ! mkdir -p "$OUTPUT_DIR" "$WORKING_DIR"; then
    report_failure 1 "Failed to create output directories" "filesystem"
    exit 1
fi

# --- Prepare configuration using the Python script ---
log_with_timestamp "Composing configuration for $PHANTOM + $ALGORITHM..."

# Build override arguments
OVERRIDE_ARGS=()
OVERRIDE_ARGS+=("output_path=$OUTPUT_DIR")
OVERRIDE_ARGS+=("working_path=$WORKING_DIR")

# Add any config overrides passed from the launcher
if [ -n "${CONFIG_OVERRIDES_JSON:-}" ] && [ "$CONFIG_OVERRIDES_JSON" != "{}" ]; then
    while IFS= read -r line; do
        OVERRIDE_ARGS+=("$line")
    done < <(python3 - <<'PY' "$CONFIG_OVERRIDES_JSON"
import json, sys
overrides = json.loads(sys.argv[1])
for key, value in overrides.items():
    # Format the value appropriately
    if isinstance(value, bool):
        print(f"{key}={str(value).lower()}")
    elif isinstance(value, str):
        print(f"{key}={value}")
    else:
        print(f"{key}={value}")
PY
    )
fi

log_with_timestamp "Configuration overrides: ${OVERRIDE_ARGS[*]}"

# --- Run the experiment ---
cd "$BASE_DIR"

MASTER_SCRIPT="$SCRIPTS_DIR/run_phantom_experiments.py"
if [ ! -f "$MASTER_SCRIPT" ]; then
    report_failure 1 "Master script not found: $MASTER_SCRIPT" "config"
    exit 1
fi

MAX_RETRIES=2
RETURN_CODE=1

for attempt in $(seq 1 $MAX_RETRIES); do
    log_with_timestamp "Experiment attempt $attempt of $MAX_RETRIES"

    # Check system resources before each attempt
    available_kb=$(df . | tail -1 | awk '{print $4}')
    log_with_timestamp "Available disk space: ${available_kb}KB"

    if [ "$available_kb" -lt 524288 ]; then  # 512MB in KB
        report_failure 1 "Insufficient disk space for experiment: ${available_kb}KB" "disk_space"
        log_with_timestamp "ERROR: Insufficient disk space for experiment"
        exit 1
    fi

    # Run the experiment
    attempt_start_time=$(date '+%Y-%m-%d %H:%M:%S')
    log_with_timestamp "Starting experiment at: $attempt_start_time"

    # Build command with overrides
    CMD_ARGS=("--phantom" "$PHANTOM" "--algorithm" "$ALGORITHM")
    for override in "${OVERRIDE_ARGS[@]}"; do
        CMD_ARGS+=("--override" "$override")
    done

    log_with_timestamp "Running: python $MASTER_SCRIPT ${CMD_ARGS[*]}"

    if python "$MASTER_SCRIPT" "${CMD_ARGS[@]}"; then
        RETURN_CODE=0
        log_with_timestamp "Experiment completed successfully"
        break
    else
        RETURN_CODE=$?
        attempt_end_time=$(date '+%Y-%m-%d %H:%M:%S')
        log_with_timestamp "Attempt $attempt failed with code $RETURN_CODE at: $attempt_end_time"

        # Analyze the failure
        if [ $RETURN_CODE -eq 124 ]; then
            log_with_timestamp "Experiment timed out"
            if [ $attempt -eq $MAX_RETRIES ]; then
                report_failure $RETURN_CODE "Experiment timeout after $MAX_RETRIES attempts" "timeout"
            fi
        elif [ $RETURN_CODE -eq 137 ]; then
            log_with_timestamp "Experiment was killed (likely OOM)"
            report_failure $RETURN_CODE "Experiment killed (likely out of memory)" "memory"
            break  # Don't retry OOM errors
        else
            log_with_timestamp "Experiment failed with error code $RETURN_CODE"
            if [ $attempt -eq $MAX_RETRIES ]; then
                report_failure $RETURN_CODE "Experiment failed after $MAX_RETRIES attempts" "experiment"
            fi
        fi

        if [ $attempt -lt $MAX_RETRIES ] && [ $RETURN_CODE -ne 137 ]; then
            log_with_timestamp "Waiting 60 seconds before retry..."
            sleep 60
        fi
    fi
done

# --- Post-processing ---
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

if [ $RETURN_CODE -eq 0 ]; then
    log_with_timestamp "Job completed successfully at: $END_TIME"

    # Verify output files were created
    output_files_count=0
    if [ -d "$OUTPUT_DIR" ]; then
        output_files_count=$(find "$OUTPUT_DIR" -name "*.hv" -o -name "*.nii*" -o -name "*.h5" -o -name "*.hdf5" 2>/dev/null | wc -l)
    fi

    cat > "$OUTPUT_DIR/job_completion.txt" <<EOF
phantom=$PHANTOM,algorithm=$ALGORITHM,status=completed,end_time=$END_TIME,host=$HOSTNAME,start_time=$START_TIME,output_files=$output_files_count
EOF

    log_with_timestamp "Created $output_files_count output files"
else
    log_with_timestamp "Job failed with return code: $RETURN_CODE at: $END_TIME"

    # If we haven't already reported the failure, do it now
    if [ ! -f "$OUTPUT_DIR/job_completion.txt" ]; then
        report_failure $RETURN_CODE "Final experiment failure" "experiment"
    fi
fi

# Final resource check
log_with_timestamp "Final system state:"
log_with_timestamp "Memory: $(free -h | grep '^Mem:' || echo 'unavailable')"
log_with_timestamp "Disk: $(df -h . | tail -1 || echo 'unavailable')"

log_with_timestamp "Task ${TASK_ID} finished with return code: $RETURN_CODE"
exit $RETURN_CODE
