#!/bin/bash

# --- SETR Parameter Sweep Launcher ---
# Usage: ./launch_sweep.sh <sweep_config> [test]
#
# Examples:
#   ./launch_sweep.sh sweep_1bpos.yaml       # Launch full sweep for 1 bed position
#   ./launch_sweep.sh sweep_2bpos.yaml       # Launch full sweep for 2 bed positions  
#   ./launch_sweep.sh sweep_1bpos.yaml test  # Test mode (single job)

set -e  # Exit on error

# --- Configuration ---
# Auto-detect base directory (assumes script is in BASE_DIR/sweeps/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
SWEEP_CONFIGS_DIR=$SCRIPT_DIR/configs
PARAM_DIR=$SCRIPT_DIR/parameters
JOB_SCRIPT=$SCRIPT_DIR/scripts/sweep_alpha_beta.qsub.sh

# --- Input validation ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 <sweep_config.yaml> [test]"
    echo ""
    echo "Available sweep configurations:"
    ls -1 $SWEEP_CONFIGS_DIR/sweep_*.yaml 2>/dev/null || echo "  No sweep configs found"
    exit 1
fi

SWEEP_CONFIG_FILE="$1"
TEST_MODE="${2:-""}"

# Check if config file exists
if [ ! -f "$SWEEP_CONFIGS_DIR/$SWEEP_CONFIG_FILE" ]; then
    echo "Error: Sweep config file not found: $SWEEP_CONFIGS_DIR/$SWEEP_CONFIG_FILE"
    echo ""
    echo "Available configurations:"
    ls -1 $SWEEP_CONFIGS_DIR/sweep_*.yaml 2>/dev/null || echo "  No sweep configs found"
    exit 1
fi

echo "=== SETR Parameter Sweep Launcher ==="
echo "Sweep config: $SWEEP_CONFIG_FILE"
echo "Mode: $([ "$TEST_MODE" == "test" ] && echo "TEST" || echo "FULL SWEEP")"
echo ""

# --- Parse YAML config ---
# Simple YAML parsing (assumes specific format)
SWEEP_NAME=$(grep "^sweep_name:" "$SWEEP_CONFIGS_DIR/$SWEEP_CONFIG_FILE" | cut -d'"' -f2)
BASE_CONFIG=$(grep "^base_config:" "$SWEEP_CONFIGS_DIR/$SWEEP_CONFIG_FILE" | cut -d'"' -f2)
SCRIPT=$(grep "^script:" "$SWEEP_CONFIGS_DIR/$SWEEP_CONFIG_FILE" | cut -d'"' -f2)
RUNTIME=$(grep "runtime:" "$SWEEP_CONFIGS_DIR/$SWEEP_CONFIG_FILE" | cut -d'"' -f2)
MEMORY=$(grep "memory:" "$SWEEP_CONFIGS_DIR/$SWEEP_CONFIG_FILE" | cut -d'"' -f2)
CORES=$(grep "cores:" "$SWEEP_CONFIGS_DIR/$SWEEP_CONFIG_FILE" | cut -d' ' -f2)
ALPHA_FILE=$(grep "alpha_file:" "$SWEEP_CONFIGS_DIR/$SWEEP_CONFIG_FILE" | cut -d'"' -f2)
BETA_FILE=$(grep "beta_file:" "$SWEEP_CONFIGS_DIR/$SWEEP_CONFIG_FILE" | cut -d'"' -f2)

# Set defaults if not found
SWEEP_NAME=${SWEEP_NAME:-"default_sweep"}
BASE_CONFIG=${BASE_CONFIG:-"config_1bpos.yaml"}
SCRIPT=${SCRIPT:-"run_dtnv_1bpos.py"}
RUNTIME=${RUNTIME:-"48:00:00"}
MEMORY=${MEMORY:-"60G"}
CORES=${CORES:-4}
ALPHA_FILE=${ALPHA_FILE:-"alphas.csv"}
BETA_FILE=${BETA_FILE:-"betas.csv"}

echo "Sweep configuration:"
echo "  Name: $SWEEP_NAME"
echo "  Base config: $BASE_CONFIG"
echo "  Script: $SCRIPT"
echo "  Runtime: $RUNTIME"
echo "  Memory: $MEMORY"
echo "  Cores: $CORES"
echo "  Alpha file: $ALPHA_FILE"
echo "  Beta file: $BETA_FILE"
echo ""

# --- Check parameter files ---
if [ ! -f "$PARAM_DIR/$ALPHA_FILE" ]; then
    echo "Error: Alpha parameter file not found: $PARAM_DIR/$ALPHA_FILE"
    exit 1
fi

if [ ! -f "$PARAM_DIR/$BETA_FILE" ]; then
    echo "Error: Beta parameter file not found: $PARAM_DIR/$BETA_FILE"
    exit 1
fi

# --- Calculate job count ---
NUM_ALPHAS=$(tail -n +2 "$PARAM_DIR/$ALPHA_FILE" | grep -c '^[^[:space:]]*[[:space:]]*$')
NUM_BETAS=$(tail -n +2 "$PARAM_DIR/$BETA_FILE" | grep -c '^[^[:space:]]*[[:space:]]*$')
TOTAL_JOBS=$((NUM_ALPHAS * NUM_BETAS))

echo "Parameter sweep details:"
echo "  Alphas: $NUM_ALPHAS"
echo "  Betas: $NUM_BETAS"
echo "  Total combinations: $TOTAL_JOBS"
echo ""

if [ "$TOTAL_JOBS" -le 0 ]; then
    echo "Error: No valid parameter combinations found"
    exit 1
fi

# --- Test Mode Check ---
if [ "$TEST_MODE" == "test" ]; then
    echo "=== TEST MODE ==="
    echo "Submitting a single job for the first α-β combination (TASK_ID=1)"
    TASK_RANGE="1-1"
else
    echo "=== FULL SWEEP MODE ==="
    echo "Submitting job array with $TOTAL_JOBS tasks"
    TASK_RANGE="1-$TOTAL_JOBS"
fi

# --- Create output directories ---
mkdir -p $SCRIPT_DIR/output/$SWEEP_NAME
mkdir -p $HOME/setr_logs

# --- Submit job ---
echo ""
echo "Submitting to SGE..."

# Export environment variables for the job script
export SWEEP_NAME
export BASE_CONFIG_FILE="$BASE_CONFIG"
export RECON_SCRIPT="$SCRIPT"
export ALPHA_FILE
export BETA_FILE
export SETR_BASE_DIR="$BASE_DIR"

# Submit with dynamic SGE parameters
qsub \
    -t $TASK_RANGE \
    -N "${SWEEP_NAME}" \
    -pe smp $CORES \
    -l h_rt=$RUNTIME \
    -l h_vmem=$MEMORY \
    -o $HOME/setr_logs \
    -e $HOME/setr_logs \
    "$JOB_SCRIPT"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Job array submitted successfully!"
    echo "  Job name: $SWEEP_NAME"
    echo "  Tasks: $TASK_RANGE"
    echo "  Log directory: $HOME/setr_logs"
    echo "  Output directory: $SCRIPT_DIR/output/$SWEEP_NAME"
    echo ""
    echo "Monitor progress with:"
    echo "  qstat -u $USER"
    echo "  ls -la $SCRIPT_DIR/output/$SWEEP_NAME/"
else
    echo "✗ Job submission failed!"
    exit 1
fi