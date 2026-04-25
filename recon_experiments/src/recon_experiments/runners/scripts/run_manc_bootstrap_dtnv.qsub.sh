#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
set -euo pipefail

# ================================
# MANC Bootstrap DTNV Reconstruction Script
# ================================
# This script runs DTNV reconstruction for MANC NEMA phantom bootstrap datasets
# with a specified alpha/beta pair across all bootstrap replicates
#
# Usage: Set ALPHA and BETA via environment or launcher script
# Each task processes one bootstrap dataset

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
bootstrap=${BOOTSTRAP_ID:-unknown},alpha=${ALPHA:-unknown},beta=${BETA:-unknown},status=failed,return_code=$exit_code,end_time=$(date),host=$HOSTNAME,failure_reason=$failure_reason,failure_type=$failure_type,start_time=$START_TIME
EOF
    fi
}

# Enhanced GPU ECC check
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

    mem_info=$(free -h | grep "^Mem:")
    log_with_timestamp "Memory: $mem_info"

    disk_info=$(df -h . | tail -1)
    log_with_timestamp "Disk space (working dir): $disk_info"

    available_kb=$(df . | tail -1 | awk '{print $4}')
    if [ "$available_kb" -lt 1048576 ]; then  # 1GB in KB
        report_failure 1 "Low disk space: ${available_kb}KB available" "disk_space"
        log_with_timestamp "ERROR: Low disk space detected"
        exit 1
    fi
}

# Perform initial health checks
log_with_timestamp "Starting MANC bootstrap DTNV reconstruction: Task ${TASK_ID}"
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

# --- Directory structure ---
if [ -n "${SETR_BASE_DIR:-}" ]; then
    BASE_DIR="$SETR_BASE_DIR"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    BASE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi

SCRIPTS_DIR="$BASE_DIR/src/recon_experiments/runners/scripts"
CONFIG_DIR="$BASE_DIR/configs"
REPO_ROOT="$(cd "$BASE_DIR/.." && pwd)"
RESULTS_BASE="${RESULTS_BASE:-$REPO_ROOT/results/manc_bootstraps/dtnv}"

# Bootstrap data configuration
BOOTSTRAP_BASE_DIR="/home/storage/prepared_data/phantom_data/manc_nema_phantom_data/PET/bootstraps/nonparametric"
SPECT_DATA_PATH="/home/storage/prepared_data/phantom_data/manc_nema_phantom_data/SPECT"

# Get alpha and beta from environment (set by launcher)
ALPHA=${ALPHA:-1.0}
BETA=${BETA:-1.0}

log_with_timestamp "Using alpha=$ALPHA, beta=$BETA"

# Create logs directory
mkdir -p "$HOME/setr_logs"

# --- Determine bootstrap ID from task ID ---
# Task IDs are 1-indexed, bootstraps are numbered 000-029
BOOTSTRAP_NUM=$((TASK_ID - 1))
BOOTSTRAP_ID=$(printf "prompts_nonparam_s73_sf0.05_%03d" $BOOTSTRAP_NUM)
BOOTSTRAP_PATH="$BOOTSTRAP_BASE_DIR/$BOOTSTRAP_ID"

log_with_timestamp "Processing bootstrap: $BOOTSTRAP_ID"
log_with_timestamp "Bootstrap path: $BOOTSTRAP_PATH"

# Check if bootstrap directory exists
if [ ! -d "$BOOTSTRAP_PATH" ]; then
    log_with_timestamp "Bootstrap directory not found: $BOOTSTRAP_PATH"
    log_with_timestamp "Task ID $TASK_ID exceeds available bootstraps. Exiting."
    exit 0
fi

# --- Paths for this job ---
OUTPUT_DIR="$RESULTS_BASE/alpha_${ALPHA}_beta_${BETA}/bootstrap_$(printf '%03d' $BOOTSTRAP_NUM)"
WORKING_DIR="$OUTPUT_DIR/tmp"

log_with_timestamp "Creating output directories..."
if ! mkdir -p "$OUTPUT_DIR" "$WORKING_DIR"; then
    report_failure 1 "Failed to create output directories" "filesystem"
    exit 1
fi

CONFIG_FILE="$CONFIG_DIR/.tmp_manc_bootstrap_alpha_${ALPHA}_beta_${BETA}_$(printf '%03d' $BOOTSTRAP_NUM).yaml"

# Copy base config
BASE_CONFIG="$CONFIG_DIR/config_manc_bootstrap_dtnv.yaml"
if [ ! -f "$BASE_CONFIG" ]; then
    report_failure 1 "Base config file not found: $BASE_CONFIG" "config"
    exit 1
fi

cp "$BASE_CONFIG" "$CONFIG_FILE"

# --- Edit YAML config ---
log_with_timestamp "Updating configuration file..."
if ! python3 - <<'PY' "$CONFIG_FILE" "$ALPHA" "$BETA" "$BOOTSTRAP_PATH" "$OUTPUT_DIR" "$WORKING_DIR" "$SPECT_DATA_PATH"
import sys, os, tempfile, re, yaml

cfg_file = sys.argv[1]
alpha = sys.argv[2]
beta = sys.argv[3]
bootstrap_path = sys.argv[4]
output_dir = sys.argv[5]
working_dir = sys.argv[6]
spect_path = sys.argv[7]

try:
    # Read original file
    with open(cfg_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace values with regex to preserve formatting
    content = re.sub(r'^alpha:\s*[\d\.]+.*$', f'alpha: {alpha}', content, flags=re.MULTILINE)
    content = re.sub(r'^beta:\s*[\d\.]+.*$', f'beta: {beta}', content, flags=re.MULTILINE)
    content = re.sub(r'^pet_data_path:\s*["\']?.*["\']?$', f'pet_data_path: "{bootstrap_path}"', content, flags=re.MULTILINE)
    content = re.sub(r'^spect_data_path:\s*["\']?.*["\']?$', f'spect_data_path: "{spect_path}"', content, flags=re.MULTILINE)
    content = re.sub(r'^output_path:\s*["\']?.*["\']?$', f'output_path: "{output_dir}"', content, flags=re.MULTILINE)
    content = re.sub(r'^working_path:\s*["\']?.*["\']?$', f'working_path: "{working_dir}"', content, flags=re.MULTILINE)

    # Write atomically
    dir_name = os.path.dirname(cfg_file)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix='.tmp_', suffix='.yaml', text=True)

    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        # Verify it's valid YAML
        with open(tmp_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)

        # Replace original
        os.replace(tmp_path, cfg_file)
        print(f"Modified config: {cfg_file}")

    except Exception as e:
        try:
            os.unlink(tmp_path)
        except:
            pass
        raise e

except Exception as e:
    print(f"Error updating config: {e}")
    sys.exit(1)
PY
then
    report_failure 1 "Failed to update configuration file" "config"
    exit 1
fi

log_with_timestamp "Generated config file: $CONFIG_FILE"

# --- Run reconstruction ---
cd "$BASE_DIR"

RECON_SCRIPT="$SCRIPTS_DIR/run_dtnv_1bpos.py"
if [ ! -f "$RECON_SCRIPT" ]; then
    report_failure 1 "Reconstruction script not found: $RECON_SCRIPT" "config"
    exit 1
fi

MAX_RETRIES=3
RETURN_CODE=1

for attempt in $(seq 1 $MAX_RETRIES); do
    log_with_timestamp "Reconstruction attempt $attempt of $MAX_RETRIES"

    # Check system resources before each attempt
    available_kb=$(df . | tail -1 | awk '{print $4}')
    log_with_timestamp "Available disk space: ${available_kb}KB"

    if [ "$available_kb" -lt 524288 ]; then  # 512MB in KB
        report_failure 1 "Insufficient disk space for reconstruction: ${available_kb}KB" "disk_space"
        log_with_timestamp "ERROR: Insufficient disk space for reconstruction"
        exit 1
    fi

    # Run the reconstruction
    attempt_start_time=$(date '+%Y-%m-%d %H:%M:%S')
    log_with_timestamp "Starting reconstruction at: $attempt_start_time"

    if python "$RECON_SCRIPT" --config "$CONFIG_FILE"; then
        RETURN_CODE=0
        log_with_timestamp "Reconstruction completed successfully"
        break
    else
        RETURN_CODE=$?
        attempt_end_time=$(date '+%Y-%m-%d %H:%M:%S')
        log_with_timestamp "Attempt $attempt failed with code $RETURN_CODE at: $attempt_end_time"

        # Analyze the failure
        if [ $RETURN_CODE -eq 124 ]; then
            log_with_timestamp "Reconstruction timed out (>2 hours)"
            if [ $attempt -eq $MAX_RETRIES ]; then
                report_failure $RETURN_CODE "Reconstruction timeout after $MAX_RETRIES attempts" "timeout"
            fi
        elif [ $RETURN_CODE -eq 137 ]; then
            log_with_timestamp "Reconstruction was killed (likely OOM)"
            report_failure $RETURN_CODE "Reconstruction killed (likely out of memory)" "memory"
            break  # Don't retry OOM errors
        else
            log_with_timestamp "Reconstruction failed with error code $RETURN_CODE"
            if [ $attempt -eq $MAX_RETRIES ]; then
                report_failure $RETURN_CODE "Reconstruction failed after $MAX_RETRIES attempts" "reconstruction"
            fi
        fi

        if [ $attempt -lt $MAX_RETRIES ] && [ $RETURN_CODE -ne 137 ]; then
            log_with_timestamp "Waiting 300 seconds before retry..."
            sleep 300
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
        output_files_count=$(find "$OUTPUT_DIR" -name "*.hv" -o -name "*.nii*" 2>/dev/null | wc -l)
    fi

    cat > "$OUTPUT_DIR/job_completion.txt" <<EOF
bootstrap=$BOOTSTRAP_ID,alpha=$ALPHA,beta=$BETA,status=completed,end_time=$END_TIME,host=$HOSTNAME,start_time=$START_TIME,output_files=$output_files_count
EOF

    log_with_timestamp "Created $output_files_count output files"

    # Clean up temporary config file
    rm -f "$CONFIG_FILE"
else
    log_with_timestamp "Job failed with return code: $RETURN_CODE at: $END_TIME"

    # If we haven't already reported the failure, do it now
    if [ ! -f "$OUTPUT_DIR/job_completion.txt" ]; then
        report_failure $RETURN_CODE "Final reconstruction failure" "reconstruction"
    fi
fi

# Final resource check
log_with_timestamp "Final system state:"
log_with_timestamp "Memory: $(free -h | grep '^Mem:' || echo 'unavailable')"
log_with_timestamp "Disk: $(df -h . | tail -1 || echo 'unavailable')"

log_with_timestamp "Task ${TASK_ID} finished with return code: $RETURN_CODE"
exit $RETURN_CODE
