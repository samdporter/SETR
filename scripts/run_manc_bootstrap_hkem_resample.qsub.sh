#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
set -euo pipefail

# ================================
# MANC Bootstrap HKEM - Resampling
# ================================
# This script resamples SPECT reconstruction to PET space for each bootstrap
# Each array task processes one bootstrap dataset

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
    if [ -n "${OUTPUT_FILE:-}" ]; then
        OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
        cat > "$OUTPUT_DIR/resample_completion.txt" <<EOF
bootstrap=${BOOTSTRAP_ID:-unknown},status=failed,return_code=$exit_code,end_time=$(date),host=$HOSTNAME,failure_reason=$failure_reason,failure_type=$failure_type,start_time=$START_TIME
EOF
    fi
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
log_with_timestamp "Starting MANC resampling: Task ${TASK_ID}"
log_with_timestamp "Host: $HOSTNAME"
log_with_timestamp "Start time: $START_TIME"

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

# Pre-flight: confirm correct CIL
log_with_timestamp "Verifying Python environment..."
if ! python - <<'PY'
import sys
print("python:", sys.executable)
try:
    import cil
    print("cil   :", cil.__file__)
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
    BASE_DIR="$(dirname "$SCRIPT_DIR")"
fi

SCRIPTS_DIR="$BASE_DIR/scripts"
CONFIG_DIR="$BASE_DIR/configs"
RESULTS_BASE="${RESULTS_BASE:-/home/sam/working/synergistic_recon/results/manc_bootstraps/hkem}"

# Data paths
DATA_BASE="/home/storage/prepared_data/phantom_data/manc_nema_phantom_data"
BOOTSTRAP_BASE_DIR="${DATA_BASE}/PET/bootstraps/nonparametric"
SPECT_DATA_PATH="${DATA_BASE}/SPECT"

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

# Paths for this job
SPECT_OUTPUT_DIR="$RESULTS_BASE/spect_hkem"
SPECT_RECONSTRUCTION="${SPECT_OUTPUT_DIR}/reconstruction_x.hv"
TRANSFORM_FILE="${SPECT_DATA_PATH}/spect2pet.nii"
OUTPUT_FILE="${BOOTSTRAP_PATH}/spect.hv"

log_with_timestamp "SPECT reconstruction: $SPECT_RECONSTRUCTION"
log_with_timestamp "Transform file: $TRANSFORM_FILE"
log_with_timestamp "Output file: $OUTPUT_FILE"

# Verify SPECT reconstruction exists
if [ ! -f "$SPECT_RECONSTRUCTION" ]; then
    report_failure 1 "SPECT reconstruction not found: $SPECT_RECONSTRUCTION" "missing_input"
    exit 1
fi

# Verify transform file exists
if [ ! -f "$TRANSFORM_FILE" ]; then
    report_failure 1 "Transform file not found: $TRANSFORM_FILE" "missing_input"
    exit 1
fi

# Config file
BASE_CONFIG="$CONFIG_DIR/config_manc_resample.yaml"
if [ ! -f "$BASE_CONFIG" ]; then
    report_failure 1 "Base config file not found: $BASE_CONFIG" "config"
    exit 1
fi

CONFIG_FILE="$CONFIG_DIR/.tmp_manc_resample_${JOB_ID}_$(printf '%03d' $BOOTSTRAP_NUM).yaml"
cp "$BASE_CONFIG" "$CONFIG_FILE"

# --- Edit YAML config ---
log_with_timestamp "Updating configuration file..."
if ! python3 - <<'PY' "$CONFIG_FILE" "$DATA_BASE" "$RESULTS_BASE" "$BOOTSTRAP_PATH" "$SPECT_DATA_PATH" "$SPECT_RECONSTRUCTION" "$TRANSFORM_FILE" "$OUTPUT_FILE"
import sys, os, tempfile, re, yaml

cfg_file = sys.argv[1]
data_base = sys.argv[2]
results_base = sys.argv[3]
pet_dir = sys.argv[4]
spect_dir = sys.argv[5]
spect_recon = sys.argv[6]
transform_file = sys.argv[7]
output_file = sys.argv[8]

try:
    # Read original file
    with open(cfg_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace values with regex to preserve formatting
    content = re.sub(r'^data_dir:\s*["\']?.*["\']?$', f'data_dir: "{data_base}"', content, flags=re.MULTILINE)
    content = re.sub(r'^results_dir:\s*["\']?.*["\']?$', f'results_dir: "{results_base}"', content, flags=re.MULTILINE)
    content = re.sub(r'^pet_dir:\s*["\']?.*["\']?$', f'pet_dir: "{pet_dir}"', content, flags=re.MULTILINE)
    content = re.sub(r'^spect_dir:\s*["\']?.*["\']?$', f'spect_dir: "{spect_dir}"', content, flags=re.MULTILINE)
    content = re.sub(r'^spect_reconstruction:\s*["\']?.*["\']?$', f'spect_reconstruction: "{spect_recon}"', content, flags=re.MULTILINE)
    content = re.sub(r'^transform_file:\s*["\']?.*["\']?$', f'transform_file: "{transform_file}"', content, flags=re.MULTILINE)
    content = re.sub(r'^output_file:\s*["\']?.*["\']?$', f'output_file: "{output_file}"', content, flags=re.MULTILINE)

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

# --- Run resampling ---
cd "$BASE_DIR"

RESAMPLE_SCRIPT="$SCRIPTS_DIR/resample_spect_to_pet_simple.py"
if [ ! -f "$RESAMPLE_SCRIPT" ]; then
    report_failure 1 "Resampling script not found: $RESAMPLE_SCRIPT" "config"
    exit 1
fi

MAX_RETRIES=2
RETURN_CODE=1

for attempt in $(seq 1 $MAX_RETRIES); do
    log_with_timestamp "Resampling attempt $attempt of $MAX_RETRIES"

    # Run the resampling
    attempt_start_time=$(date '+%Y-%m-%d %H:%M:%S')
    log_with_timestamp "Starting resampling at: $attempt_start_time"

    if python "$RESAMPLE_SCRIPT" --config "$CONFIG_FILE"; then
        RETURN_CODE=0
        log_with_timestamp "Resampling completed successfully"
        break
    else
        RETURN_CODE=$?
        attempt_end_time=$(date '+%Y-%m-%d %H:%M:%S')
        log_with_timestamp "Attempt $attempt failed with code $RETURN_CODE at: $attempt_end_time"

        if [ $attempt -eq $MAX_RETRIES ]; then
            report_failure $RETURN_CODE "Resampling failed after $MAX_RETRIES attempts" "resampling"
        else
            log_with_timestamp "Waiting 60 seconds before retry..."
            sleep 60
        fi
    fi
done

# --- Post-processing ---
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

if [ $RETURN_CODE -eq 0 ]; then
    log_with_timestamp "Job completed successfully at: $END_TIME"

    # Verify output file was created
    if [ -f "$OUTPUT_FILE" ]; then
        file_size=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE" 2>/dev/null || echo "0")
        log_with_timestamp "Created output file: $OUTPUT_FILE (${file_size} bytes)"

        OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
        cat > "$OUTPUT_DIR/resample_completion.txt" <<EOF
bootstrap=$BOOTSTRAP_ID,status=completed,end_time=$END_TIME,host=$HOSTNAME,start_time=$START_TIME,output_file=$OUTPUT_FILE,file_size=$file_size
EOF
    else
        log_with_timestamp "WARNING: Output file not found: $OUTPUT_FILE"
        report_failure 1 "Output file not created" "output_missing"
        RETURN_CODE=1
    fi

    # Clean up temporary config file
    rm -f "$CONFIG_FILE"
else
    log_with_timestamp "Job failed with return code: $RETURN_CODE at: $END_TIME"

    # If we haven't already reported the failure, do it now
    OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
    if [ ! -f "$OUTPUT_DIR/resample_completion.txt" ]; then
        report_failure $RETURN_CODE "Final resampling failure" "resampling"
    fi
fi

# Final resource check
log_with_timestamp "Final system state:"
log_with_timestamp "Memory: $(free -h | grep '^Mem:' || echo 'unavailable')"
log_with_timestamp "Disk: $(df -h . | tail -1 || echo 'unavailable')"

log_with_timestamp "Task ${TASK_ID} finished with return code: $RETURN_CODE"
exit $RETURN_CODE
