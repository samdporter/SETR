#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
set -euo pipefail

# SGE Job script for single debug test
# This script runs test_function_debug.py on the cluster with proper environment setup

echo "Starting debug test job at: $(date)"
echo "Job ID: ${JOB_ID:-unknown}"
echo "Host: $(hostname)"

# Fail fast on GPU issues (copied from sweep script)
if command -v nvidia-smi >/dev/null; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra __GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    for __id in "${__GPU_IDS[@]}"; do
      if nvidia-smi -i "${__id}" \
            --query-gpu=ecc.errors.uncorrected.total \
            --format=csv,noheader 2>/dev/null | grep -q '^[1-9]'; then
        echo "GPU ${__id} reports uncorrectable ECC errors. Exiting."
        sleep 20; exit 99
      fi
    done
  fi
fi

# --- Runtime env: activate venv + SIRF ---
source "$HOME/sirf_venv/bin/activate"

export INSTALLDIR=/home/sporter/synergistic_Y90/devel/SIRF/SIRF_installs/Release-cuda12.0
source "${INSTALLDIR}/bin/env_sirf.sh"

# Remove any source-tree CIL path to avoid shadowing the wheel install
if [ -n "${PYTHONPATH:-}" ]; then
  PYTHONPATH="$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v 'CIL/Wrappers/Python' | paste -sd':' -)"
  export PYTHONPATH
fi

# Pre-flight: confirm correct CIL and that libcilacc exists
echo "Checking Python environment..."
python - <<'PY'
import sys, pathlib, importlib
print("python:", sys.executable)
try:
    import cil
    libdir = pathlib.Path(cil.__file__).parent / "lib"
    print("cil   :", cil.__file__)
    print("cilacc:", list(libdir.glob("*cilacc*")))
except Exception as e:
    print("E: import cil failed ->", e)
    raise
PY

# --- Project layout ---
# Detect base directory from script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "Base directory: $BASE_DIR"
echo "Script directory: $SCRIPT_DIR"

# Set up paths
CONFIG_FILE="$BASE_DIR/configs/config_test_debug.yaml"
DEBUG_SCRIPT="$SCRIPT_DIR/test_function_debug.py"
OUTPUT_DIR="$BASE_DIR/output/debug_test"
LOG_DIR="$OUTPUT_DIR/logs"

# Create output directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Config file: $CONFIG_FILE"
echo "Debug script: $DEBUG_SCRIPT"
echo "Output directory: $OUTPUT_DIR"

# Check that required files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$DEBUG_SCRIPT" ]; then
    echo "Error: Debug script not found: $DEBUG_SCRIPT"
    exit 1
fi

# --- Run the debug test ---
echo "Starting debug test execution..."
cd "$BASE_DIR"

# Run with output redirection to capture all logs
python "$DEBUG_SCRIPT" --config "$CONFIG_FILE" 2>&1 | tee "$LOG_DIR/debug_execution.log"
RETURN_CODE=${PIPESTATUS[0]}

# --- Post-processing ---
if [ $RETURN_CODE -eq 0 ]; then
    echo "Debug test completed successfully at: $(date)"
    echo "status=completed,end_time=$(date),return_code=$RETURN_CODE" > "$OUTPUT_DIR/job_completion.txt"
    
    # Copy key logs to output directory for easy access
    if [ -f "$LOG_DIR/debug_execution.log" ]; then
        echo "Debug execution log saved to: $LOG_DIR/debug_execution.log"
    fi
    
    # List generated files
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"
else
    echo "Debug test failed with return code: $RETURN_CODE at: $(date)"
    echo "status=failed,return_code=$RETURN_CODE,end_time=$(date)" > "$OUTPUT_DIR/job_completion.txt"
    
    echo "Check logs in: $LOG_DIR"
fi

echo "Debug test job finished with return code: $RETURN_CODE"
exit $RETURN_CODE