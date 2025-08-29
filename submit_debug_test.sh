#!/bin/bash

# Submit debug test job to SGE cluster
# Usage: ./submit_debug_test.sh

set -euo pipefail

# Get base directory from script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$SCRIPT_DIR"

# Paths
SGE_SCRIPT="$BASE_DIR/scripts/run_debug_cluster.qsub.sh"
OUTPUT_DIR="$BASE_DIR/output/debug_test"
LOG_DIR="$OUTPUT_DIR/logs"

echo "=== SETR Debug Test Submission ==="
echo "Base directory: $BASE_DIR"
echo "SGE script: $SGE_SCRIPT"
echo "Output directory: $OUTPUT_DIR"

# Check if SGE script exists
if [ ! -f "$SGE_SCRIPT" ]; then
    echo "Error: SGE script not found: $SGE_SCRIPT"
    exit 1
fi

# Check if config exists
CONFIG_FILE="$BASE_DIR/configs/config_test_debug.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Make sure config_test_debug.yaml exists in configs/ directory"
    exit 1
fi

# Create output and log directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Creating directories..."
echo "  Output: $OUTPUT_DIR"
echo "  Logs: $LOG_DIR"

# SGE resource settings
SGE_RUNTIME="2:00:00"    # 2 hours
SGE_MEMORY="30G"         # 30GB memory  
SGE_CORES="1"            # Single core
SGE_GPU="true"           # GPU needed for acquisition models

# Build qsub command
echo ""
echo "Submitting debug test job to SGE..."

QSUB_CMD="qsub \
  -l h_rt=\"$SGE_RUNTIME\" \
  -l tmem=\"$SGE_MEMORY\" \
  -l gpu=true \
  -N \"setr_debug_test\" \
  -o \"$LOG_DIR/debug_test.out\" \
  -e \"$LOG_DIR/debug_test.err\" \
  \"$SGE_SCRIPT\""

echo "Command: $QSUB_CMD"
echo ""

# Submit the job
eval "$QSUB_CMD"

echo ""
echo "Job submitted successfully!"
echo ""
echo "Monitor with:"
echo "  qstat -u \$USER"
echo "  qstat -j <job_id>"
echo ""
echo "Check results after completion:"
echo "  cat $OUTPUT_DIR/job_completion.txt"
echo "  cat $OUTPUT_DIR/function_summary.txt"
echo "  ls $OUTPUT_DIR/"
echo ""
echo "Logs will be in:"
echo "  $LOG_DIR/debug_test.out  (stdout)"
echo "  $LOG_DIR/debug_test.err  (stderr)"
echo "  $LOG_DIR/debug_execution.log  (script output)"