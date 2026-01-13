#!/bin/bash

# --- SETR Preconditioner Sweep Launcher ---
# Usage: ./launch_precond_sweep.sh <sweep_config.yaml> [test|local]

set -euo pipefail

# Auto-detect base directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
FUNC_DIR="$SCRIPT_DIR"
CONFIG_DIR="$FUNC_DIR/configs"
PARAM_DIR="$FUNC_DIR/parameters"
SCRIPTS_DIR="$FUNC_DIR/scripts"

# Get sweep config from argument
SWEEP_CONFIG="${1:-precond_sweep_1bpos.yaml}"
MODE="${2:-full}"

show_usage() {
    echo "=== SETR Preconditioner Sweep Launcher ==="
    echo "Usage: $0 [sweep_config.yaml] [mode]"
    echo ""
    echo "Modes:"
    echo "  full    - Submit all jobs to cluster (default)"
    echo "  test    - Submit only one test job to cluster"
    echo "  local   - Run one test locally (no SGE submission)"
    echo ""
    echo "Available sweep configs:"
    ls -1 "$CONFIG_DIR"/*.yaml 2>/dev/null | sed 's/^/  /' || echo "  No configs found"
}

if [ "$MODE" = "help" ] || [ "$MODE" = "--help" ]; then
    show_usage
    exit 0
fi

# Check if config exists
SWEEP_CONFIG_PATH="$CONFIG_DIR/$SWEEP_CONFIG"
if [ ! -f "$SWEEP_CONFIG_PATH" ]; then
    echo "Error: Sweep config not found: $SWEEP_CONFIG_PATH"
    show_usage
    exit 1
fi

echo "=== SETR Preconditioner Sweep Launcher ==="
echo "Config: $SWEEP_CONFIG"
echo "Mode: $MODE"
echo "Base directory: $BASE_DIR"

# Parse YAML config using Python
CONFIG_VALUES=$(python3 -c "
import yaml
with open('$SWEEP_CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)

print('SWEEP_NAME=' + config['sweep_name'])
print('BASE_CONFIG=' + config['base_config'])
print('RECON_SCRIPT=' + config['script'])
print('PRECOND_TYPES_FILE=' + config['parameters']['precond_types_file'])
print('ALPHAS_FILE=' + config['parameters']['alphas_file'])
print('STEP_SIZES_FILE=' + config['parameters']['step_sizes_file'])
print('SGE_RUNTIME=' + config['sge']['runtime'])
print('SGE_MEMORY=' + config['sge']['memory'])
print('SGE_CORES=' + str(config['sge']['cores']))
print('SGE_QUEUE=' + (config['sge']['queue'] or 'default'))
print('SGE_GPU=' + str(config['sge'].get('gpu', False)).lower())
print('NUM_EPOCHS=' + str(config['fixed_params'].get('num_epochs', 50)))
")

# Source the config values
eval "$CONFIG_VALUES"

echo "Sweep name: $SWEEP_NAME"
echo "Base config: $BASE_CONFIG"
echo "Epochs: $NUM_EPOCHS"
echo "SGE resources: $SGE_RUNTIME, $SGE_MEMORY, $SGE_CORES cores, GPU=$SGE_GPU"

# Check parameter files exist
if [ ! -f "$PARAM_DIR/$PRECOND_TYPES_FILE" ]; then
    echo "Error: Precond types file not found: $PARAM_DIR/$PRECOND_TYPES_FILE"
    exit 1
fi

if [ ! -f "$PARAM_DIR/$ALPHAS_FILE" ]; then
    echo "Error: Alphas file not found: $PARAM_DIR/$ALPHAS_FILE"
    exit 1
fi

if [ ! -f "$PARAM_DIR/$STEP_SIZES_FILE" ]; then
    echo "Error: Step sizes file not found: $PARAM_DIR/$STEP_SIZES_FILE"
    exit 1
fi

# Count parameters (excluding header)
NUM_PRECOND_TYPES=$(tail -n +2 "$PARAM_DIR/$PRECOND_TYPES_FILE" | wc -l)
NUM_ALPHAS=$(tail -n +2 "$PARAM_DIR/$ALPHAS_FILE" | wc -l)
NUM_STEP_SIZES=$(tail -n +2 "$PARAM_DIR/$STEP_SIZES_FILE" | wc -l)
TOTAL_JOBS=$((NUM_PRECOND_TYPES * NUM_ALPHAS * NUM_STEP_SIZES))

echo "Parameters: $NUM_PRECOND_TYPES precond types × $NUM_ALPHAS alphas × $NUM_STEP_SIZES step sizes = $TOTAL_JOBS total jobs"

if [ "$TOTAL_JOBS" -eq 0 ]; then
    echo "Error: No parameter combinations found"
    exit 1
fi

# Prepare sweep output + logs
SWEEP_OUT_DIR="$FUNC_DIR/output/$SWEEP_NAME"
LOG_DIR="$SWEEP_OUT_DIR/_logs"
mkdir -p "$LOG_DIR"

# Handle different modes
case "$MODE" in
    "local")
        echo ""
        echo "LOCAL TEST MODE: Running first parameter combination locally"
        echo ""

        # Read first parameter combination
        PRECOND_TYPE=$(tail -n +2 "$PARAM_DIR/$PRECOND_TYPES_FILE" | head -1 | awk -F, '{gsub(/^[ \t]+|[ \t]+$/,"",$1); print $1}')
        ALPHA=$(tail -n +2 "$PARAM_DIR/$ALPHAS_FILE" | head -1 | awk -F, '{gsub(/^[ \t]+|[ \t]+$/,"",$1); print $1}')
        STEP_SIZE=$(tail -n +2 "$PARAM_DIR/$STEP_SIZES_FILE" | head -1 | awk -F, '{gsub(/^[ \t]+|[ \t]+$/,"",$1); print $1}')

        echo "Testing: precond_type=$PRECOND_TYPE, alpha=$ALPHA, step_size=$STEP_SIZE"

        OUTPUT_DIR="$SWEEP_OUT_DIR/local_test"
        mkdir -p "$OUTPUT_DIR"

        cd "$BASE_DIR"
        python scripts/test_preconditioner_single.py \
            --config "configs/$BASE_CONFIG" \
            --output "$OUTPUT_DIR" \
            --precond-type "$PRECOND_TYPE" \
            --alpha "$ALPHA" \
            --step-size "$STEP_SIZE" \
            --epochs "$NUM_EPOCHS"

        echo ""
        echo "Local test complete!"
        echo "Output: $OUTPUT_DIR"
        ;;

    "test")
        JOB_RANGE="1"
        echo ""
        echo "TEST MODE: Submitting 1 job to cluster"
        echo ""

        # Build qsub command
        QSUB_SCRIPT="$SCRIPTS_DIR/precond_sweep.qsub.sh"
        if [ ! -f "$QSUB_SCRIPT" ]; then
            echo "Error: SGE script not found: $QSUB_SCRIPT"
            exit 1
        fi

        # Set GPU option
        GPU_OPTION=""
        if [ "$SGE_GPU" = "true" ]; then
            GPU_OPTION="-l gpu=true"
            MEM_OPTION="-l tmem=${SGE_MEMORY}"
        else
            MEM_OPTION="-l h_vmem=${SGE_MEMORY}"
        fi

        # Set parallel environment option
        PE_OPTION=""
        if [ "$SGE_CORES" -gt 1 ]; then
            PE_OPTION="-pe smp $SGE_CORES"
        fi

        qsub \
          -t "$JOB_RANGE" \
          -r y \
          -l h_rt="$SGE_RUNTIME" \
          $MEM_OPTION \
          $GPU_OPTION \
          $PE_OPTION \
          -N "precond_${SWEEP_NAME}_test" \
          -o "$LOG_DIR" \
          -e "$LOG_DIR" \
          -v "SETR_BASE_DIR=$BASE_DIR,SWEEP_NAME=${SWEEP_NAME},BASE_CONFIG_FILE=$BASE_CONFIG,PRECOND_TYPES_FILE=$PRECOND_TYPES_FILE,ALPHAS_FILE=$ALPHAS_FILE,STEP_SIZES_FILE=$STEP_SIZES_FILE,NUM_EPOCHS=$NUM_EPOCHS" \
          "$QSUB_SCRIPT"

        echo ""
        echo "Test job submitted!"
        echo "Output root: $SWEEP_OUT_DIR"
        echo "Logs: $LOG_DIR"
        ;;

    "full"|*)
        JOB_RANGE="1-$TOTAL_JOBS"
        echo ""
        echo "FULL MODE: Submitting $TOTAL_JOBS jobs to cluster"
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi

        # Build qsub command
        QSUB_SCRIPT="$SCRIPTS_DIR/precond_sweep.qsub.sh"
        if [ ! -f "$QSUB_SCRIPT" ]; then
            echo "Error: SGE script not found: $QSUB_SCRIPT"
            exit 1
        fi

        # Set GPU option
        GPU_OPTION=""
        if [ "$SGE_GPU" = "true" ]; then
            GPU_OPTION="-l gpu=true"
            MEM_OPTION="-l tmem=${SGE_MEMORY}"
        else
            MEM_OPTION="-l h_vmem=${SGE_MEMORY}"
        fi

        # Set parallel environment option
        PE_OPTION=""
        if [ "$SGE_CORES" -gt 1 ]; then
            PE_OPTION="-pe smp $SGE_CORES"
        fi

        qsub \
          -t "$JOB_RANGE" \
          -r y \
          -l h_rt="$SGE_RUNTIME" \
          $MEM_OPTION \
          $GPU_OPTION \
          $PE_OPTION \
          -N "precond_${SWEEP_NAME}" \
          -o "$LOG_DIR" \
          -e "$LOG_DIR" \
          -v "SETR_BASE_DIR=$BASE_DIR,SWEEP_NAME=${SWEEP_NAME},BASE_CONFIG_FILE=$BASE_CONFIG,PRECOND_TYPES_FILE=$PRECOND_TYPES_FILE,ALPHAS_FILE=$ALPHAS_FILE,STEP_SIZES_FILE=$STEP_SIZES_FILE,NUM_EPOCHS=$NUM_EPOCHS" \
          "$QSUB_SCRIPT"

        echo ""
        echo "Jobs submitted successfully!"
        echo "Output root: $SWEEP_OUT_DIR"
        echo "Logs: $LOG_DIR"
        ;;
esac
