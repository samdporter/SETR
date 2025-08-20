#!/bin/bash

# --- SETR Sweep Launcher Script ---
# Usage: ./launch_sweep.sh <sweep_config.yaml> [test]

set -euo pipefail

# Auto-detect base directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
SWEEPS_DIR="$SCRIPT_DIR"
CONFIG_DIR="$SWEEPS_DIR/configs"
PARAM_DIR="$SWEEPS_DIR/parameters"
SCRIPTS_DIR="$SWEEPS_DIR/scripts"

# Get sweep config from argument
SWEEP_CONFIG="$1"
TEST_MODE="${2:-}"

if [ -z "$SWEEP_CONFIG" ]; then
    echo "=== SETR Sweep Launcher ==="
    echo "Usage: $0 <sweep_config.yaml> [test]"
    echo ""
    echo "Available sweep configs:"
    ls -1 "$CONFIG_DIR"/*.yaml 2>/dev/null | sed 's/^/  /' || echo "  No configs found"
    echo ""
    echo "Use 'test' as second argument to submit only one test job"
    exit 1
fi

# Check if config exists
SWEEP_CONFIG_PATH="$CONFIG_DIR/$SWEEP_CONFIG"
if [ ! -f "$SWEEP_CONFIG_PATH" ]; then
    echo "Error: Sweep config not found: $SWEEP_CONFIG_PATH"
    exit 1
fi

echo "=== SETR Sweep Launcher ==="
echo "Config: $SWEEP_CONFIG"
echo "Base directory: $BASE_DIR"

# Parse YAML config using Python
CONFIG_VALUES=$(python3 -c "
import yaml, sys
with open('$SWEEP_CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)

print('SWEEP_NAME=' + config['sweep_name'])
print('BASE_CONFIG=' + config['base_config'])
print('RECON_SCRIPT=' + config['script'])
print('ALPHA_FILE=' + config['parameters']['alpha_file'])
print('BETA_FILE=' + config['parameters']['beta_file'])
print('SGE_RUNTIME=' + config['sge']['runtime'])
print('SGE_MEMORY=' + config['sge']['memory'])
print('SGE_CORES=' + str(config['sge']['cores']))
print('SGE_QUEUE=' + (config['sge']['queue'] or 'default'))
")

# Source the config values
eval "$CONFIG_VALUES"

echo "Sweep name: $SWEEP_NAME"
echo "Base config: $BASE_CONFIG"
echo "Script: $RECON_SCRIPT"
echo "SGE resources: $SGE_RUNTIME, $SGE_MEMORY, $SGE_CORES cores"

# Check parameter files exist
if [ ! -f "$PARAM_DIR/$ALPHA_FILE" ]; then
    echo "Error: Alpha parameter file not found: $PARAM_DIR/$ALPHA_FILE"
    exit 1
fi

if [ ! -f "$PARAM_DIR/$BETA_FILE" ]; then
    echo "Error: Beta parameter file not found: $PARAM_DIR/$BETA_FILE"
    exit 1
fi

# Count parameters (excluding header)
NUM_ALPHAS=$(tail -n +2 "$PARAM_DIR/$ALPHA_FILE" | wc -l)
NUM_BETAS=$(tail -n +2 "$PARAM_DIR/$BETA_FILE" | wc -l)
TOTAL_JOBS=$((NUM_ALPHAS * NUM_BETAS))

echo "Parameters: $NUM_ALPHAS alphas × $NUM_BETAS betas = $TOTAL_JOBS total jobs"

if [ "$TOTAL_JOBS" -eq 0 ]; then
    echo "Error: No parameter combinations found"
    exit 1
fi

# Prepare output directory
OUTPUT_DIR="$SWEEPS_DIR/output/$SWEEP_NAME"
mkdir -p "$OUTPUT_DIR"

# Set job array range
if [ "$TEST_MODE" = "test" ]; then
    JOB_RANGE="1"
    echo "TEST MODE: Submitting only 1 job"
else
    JOB_RANGE="1-$TOTAL_JOBS"
    echo "FULL MODE: Submitting $TOTAL_JOBS jobs"
fi

# Build qsub command
QSUB_SCRIPT="$SCRIPTS_DIR/sweep_alpha_beta.qsub.sh"
if [ ! -f "$QSUB_SCRIPT" ]; then
    echo "Error: SGE script not found: $QSUB_SCRIPT"
    exit 1
fi

# Set SGE queue option
QUEUE_OPTION=""
if [ "$SGE_QUEUE" != "default" ]; then
    QUEUE_OPTION="-q $SGE_QUEUE"
fi

# Set parallel environment option - only use smp if cores > 1
PE_OPTION=""
if [ "$SGE_CORES" -gt 1 ]; then
    PE_OPTION="-pe smp $SGE_CORES"
fi

# Submit to SGE
echo ""
echo "Submitting jobs to SGE..."
echo "Command: qsub -t $JOB_RANGE -l h_rt=$SGE_RUNTIME -l h_vmem=$SGE_MEMORY $PE_OPTION $QUEUE_OPTION ..."

qsub \
    -t "$JOB_RANGE" \
    -l h_rt="$SGE_RUNTIME" \
    -l h_vmem="$SGE_MEMORY" \
    $PE_OPTION \
    $QUEUE_OPTION \
    -N "setr_$SWEEP_NAME" \
    -o "$OUTPUT_DIR/logs" \
    -e "$OUTPUT_DIR/logs" \
    -v "SETR_BASE_DIR=$BASE_DIR,SWEEP_NAME=$SWEEP_NAME,BASE_CONFIG_FILE=$BASE_CONFIG,RECON_SCRIPT=$RECON_SCRIPT,ALPHA_FILE=$ALPHA_FILE,BETA_FILE=$BETA_FILE" \
    "$QSUB_SCRIPT"

echo ""
echo "Jobs submitted successfully!"
echo "Monitor with: ./monitor_sweep.sh $SWEEP_NAME"
echo "Output directory: $OUTPUT_DIR"