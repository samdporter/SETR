#!/bin/bash

# --- SETR Subset Selection Sweep Launcher ---
# Usage: ./launch_sweep.sh <sweep_config.yaml> [test|local]

set -euo pipefail

# Auto-detect base directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
FUNC_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$FUNC_DIR/configs"
PARAM_DIR="$FUNC_DIR/parameters"
SCRIPTS_DIR="$FUNC_DIR/scripts"

# Get sweep config from argument
SWEEP_CONFIG="${1:-sweep_main_experiments.yaml}"
MODE="${2:-full}"

if [ "$MODE" = "local" ]; then
    export LOCAL_RUN_ID="${LOCAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
fi

show_usage() {
    echo "=== SETR Subset Selection Sweep Launcher ==="
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

echo "=== SETR Subset Selection Sweep Launcher ==="
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

# Check which parameters are swept
params = config['parameters']
param_files = []
if 'subset_modes_file' in params:
    param_files.append(('SUBSET_MODES_FILE', params['subset_modes_file']))
if 'prior_modes_file' in params:
    param_files.append(('PRIOR_MODES_FILE', params['prior_modes_file']))
if 'precond_types_file' in params:
    param_files.append(('PRECOND_TYPES_FILE', params['precond_types_file']))
if 'gammas_file' in params:
    param_files.append(('GAMMAS_FILE', params['gammas_file']))

for name, value in param_files:
    print(f'{name}={value}')

print('SGE_RUNTIME=' + config['sge']['runtime'])
print('SGE_MEMORY=' + config['sge']['memory'])
print('SGE_CORES=' + str(config['sge']['cores']))
print('SGE_QUEUE=' + (config['sge']['queue'] or 'default'))
print('SGE_GPU=' + str(config['sge'].get('gpu', False)).lower())
print('NUM_EPOCHS=' + str(config['fixed_params'].get('num_epochs', 100)))

# Fixed parameters for convergence reference
fixed = config.get('fixed_params', {})
if 'subset_mode' in fixed:
    print('FIXED_SUBSET_MODE=' + fixed['subset_mode'])
if 'prior_mode' in fixed:
    print('FIXED_PRIOR_MODE=' + fixed['prior_mode'])
if 'precond_type' in fixed:
    print('FIXED_PRECOND_TYPE=' + fixed['precond_type'])
")

# Source the config values
eval "$CONFIG_VALUES"

echo "Sweep name: $SWEEP_NAME"
echo "Base config: $BASE_CONFIG"
echo "Epochs: $NUM_EPOCHS"
echo "SGE resources: $SGE_RUNTIME, $SGE_MEMORY, $SGE_CORES cores, GPU=$SGE_GPU"

# Count parameters based on which files are defined
NUM_COMBINATIONS=1

if [ -n "${SUBSET_MODES_FILE:-}" ]; then
    if [ ! -f "$PARAM_DIR/$SUBSET_MODES_FILE" ]; then
        echo "Error: Subset modes file not found: $PARAM_DIR/$SUBSET_MODES_FILE"
        exit 1
    fi
    NUM_SUBSET_MODES=$(tail -n +2 "$PARAM_DIR/$SUBSET_MODES_FILE" | wc -l)
    NUM_COMBINATIONS=$((NUM_COMBINATIONS * NUM_SUBSET_MODES))
    echo "Subset modes: $NUM_SUBSET_MODES"
fi

if [ -n "${PRIOR_MODES_FILE:-}" ]; then
    if [ ! -f "$PARAM_DIR/$PRIOR_MODES_FILE" ]; then
        echo "Error: Prior modes file not found: $PARAM_DIR/$PRIOR_MODES_FILE"
        exit 1
    fi
    NUM_PRIOR_MODES=$(tail -n +2 "$PARAM_DIR/$PRIOR_MODES_FILE" | wc -l)
    NUM_COMBINATIONS=$((NUM_COMBINATIONS * NUM_PRIOR_MODES))
    echo "Prior modes: $NUM_PRIOR_MODES"
fi

if [ -n "${PRECOND_TYPES_FILE:-}" ]; then
    if [ ! -f "$PARAM_DIR/$PRECOND_TYPES_FILE" ]; then
        echo "Error: Precond types file not found: $PARAM_DIR/$PRECOND_TYPES_FILE"
        exit 1
    fi
    NUM_PRECOND_TYPES=$(tail -n +2 "$PARAM_DIR/$PRECOND_TYPES_FILE" | wc -l)
    NUM_COMBINATIONS=$((NUM_COMBINATIONS * NUM_PRECOND_TYPES))
    echo "Preconditioner types: $NUM_PRECOND_TYPES"
fi

if [ -n "${GAMMAS_FILE:-}" ]; then
    if [ ! -f "$PARAM_DIR/$GAMMAS_FILE" ]; then
        echo "Error: Gammas file not found: $PARAM_DIR/$GAMMAS_FILE. Are you in SETR dir?"
        exit 1
    fi
    NUM_GAMMAS=$(tail -n +2 "$PARAM_DIR/$GAMMAS_FILE" | wc -l)
    NUM_COMBINATIONS=$((NUM_COMBINATIONS * NUM_GAMMAS))
    echo "Gamma values: $NUM_GAMMAS"
fi

TOTAL_JOBS=$NUM_COMBINATIONS

echo "Total parameter combinations: $TOTAL_JOBS jobs"

if [ "$TOTAL_JOBS" -eq 0 ]; then
    echo "Error: No parameter combinations found"
    exit 1
fi

# Prepare sweep output + logs
OUTPUT_ROOT="${SUBSET_OUTPUT_ROOT:-$FUNC_DIR/output}"
if [ "$MODE" = "local" ] && [ -n "${LOCAL_RUN_ID:-}" ] && [ -z "${SUBSET_OUTPUT_ROOT:-}" ]; then
    OUTPUT_ROOT="$FUNC_DIR/output/local_runs/$LOCAL_RUN_ID"
fi
SWEEP_OUT_DIR="$OUTPUT_ROOT/$SWEEP_NAME"
LOG_DIR="$SWEEP_OUT_DIR/_logs"
mkdir -p "$LOG_DIR"

# Build environment variables string for qsub
ENV_VARS="SETR_BASE_DIR=$BASE_DIR,SWEEP_NAME=${SWEEP_NAME},BASE_CONFIG_FILE=$BASE_CONFIG,NUM_EPOCHS=$NUM_EPOCHS"
[ -n "${SUBSET_MODES_FILE:-}" ] && ENV_VARS="$ENV_VARS,SUBSET_MODES_FILE=$SUBSET_MODES_FILE"
[ -n "${PRIOR_MODES_FILE:-}" ] && ENV_VARS="$ENV_VARS,PRIOR_MODES_FILE=$PRIOR_MODES_FILE"
[ -n "${PRECOND_TYPES_FILE:-}" ] && ENV_VARS="$ENV_VARS,PRECOND_TYPES_FILE=$PRECOND_TYPES_FILE"
[ -n "${GAMMAS_FILE:-}" ] && ENV_VARS="$ENV_VARS,GAMMAS_FILE=$GAMMAS_FILE"
[ -n "${FIXED_SUBSET_MODE:-}" ] && ENV_VARS="$ENV_VARS,FIXED_SUBSET_MODE=$FIXED_SUBSET_MODE"
[ -n "${FIXED_PRIOR_MODE:-}" ] && ENV_VARS="$ENV_VARS,FIXED_PRIOR_MODE=$FIXED_PRIOR_MODE"
[ -n "${FIXED_PRECOND_TYPE:-}" ] && ENV_VARS="$ENV_VARS,FIXED_PRECOND_TYPE=$FIXED_PRECOND_TYPE"

# Handle different modes
case "$MODE" in
    "local")
        echo ""
        echo "LOCAL MODE: Running all parameter combinations sequentially"
        if [ -n "${LOCAL_RUN_ID:-}" ]; then
            echo "Local run ID: $LOCAL_RUN_ID"
        fi
        echo ""

        # Build arrays of values for each swept dimension.
        # Dimensions not present in this sweep get a single empty-string sentinel
        # so the loop runs once without contributing to OVERRIDES or RUN_NAME.
        if [ -n "${SUBSET_MODES_FILE:-}" ]; then
            mapfile -t _SUBSET_MODES < <(tail -n +2 "$PARAM_DIR/$SUBSET_MODES_FILE" | awk '{print $1}')
        else
            _SUBSET_MODES=("")
        fi
        if [ -n "${PRIOR_MODES_FILE:-}" ]; then
            mapfile -t _PRIOR_MODES < <(tail -n +2 "$PARAM_DIR/$PRIOR_MODES_FILE" | awk '{print $1}')
        else
            _PRIOR_MODES=("")
        fi
        if [ -n "${PRECOND_TYPES_FILE:-}" ]; then
            mapfile -t _PRECOND_TYPES < <(tail -n +2 "$PARAM_DIR/$PRECOND_TYPES_FILE" | awk '{print $1}')
        else
            _PRECOND_TYPES=("")
        fi
        if [ -n "${GAMMAS_FILE:-}" ]; then
            mapfile -t _GAMMAS < <(tail -n +2 "$PARAM_DIR/$GAMMAS_FILE" | awk '{print $1}')
        else
            _GAMMAS=("")
        fi

        TOTAL_LOCAL=$(( ${#_SUBSET_MODES[@]} * ${#_PRIOR_MODES[@]} * ${#_PRECOND_TYPES[@]} * ${#_GAMMAS[@]} ))
        JOB_NUM=0
        echo "Total combinations: $TOTAL_LOCAL"
        echo ""

        for SUBSET_MODE in "${_SUBSET_MODES[@]}"; do
        for PRIOR_MODE in "${_PRIOR_MODES[@]}"; do
        for PRECOND_TYPE in "${_PRECOND_TYPES[@]}"; do
        for GAMMA in "${_GAMMAS[@]}"; do
            JOB_NUM=$((JOB_NUM + 1))
            OVERRIDES=""
            RUN_NAME="subset"

            if [ -n "$SUBSET_MODE" ]; then
                OVERRIDES="$OVERRIDES subset_mode=$SUBSET_MODE"
                RUN_NAME="${RUN_NAME}_${SUBSET_MODE}"
            elif [ -n "${FIXED_SUBSET_MODE:-}" ]; then
                OVERRIDES="$OVERRIDES subset_mode=$FIXED_SUBSET_MODE"
            fi

            if [ -n "$PRIOR_MODE" ]; then
                OVERRIDES="$OVERRIDES prior_mode=$PRIOR_MODE"
                RUN_NAME="${RUN_NAME}_prior_${PRIOR_MODE}"
            elif [ -n "${FIXED_PRIOR_MODE:-}" ]; then
                OVERRIDES="$OVERRIDES prior_mode=$FIXED_PRIOR_MODE"
            fi

            if [ -n "$PRECOND_TYPE" ]; then
                OVERRIDES="$OVERRIDES precond_type=$PRECOND_TYPE"
                RUN_NAME="${RUN_NAME}_precond_${PRECOND_TYPE}"
            elif [ -n "${FIXED_PRECOND_TYPE:-}" ]; then
                OVERRIDES="$OVERRIDES precond_type=$FIXED_PRECOND_TYPE"
            fi

            if [ -n "$GAMMA" ]; then
                OVERRIDES="$OVERRIDES gamma_tnv=$GAMMA"
                RUN_NAME="${RUN_NAME}_gamma_${GAMMA}"
            fi

            RUN_NAME="${RUN_NAME//[^A-Za-z0-9._-]/_}"
            OUTPUT_DIR="$SWEEP_OUT_DIR/$RUN_NAME"
            mkdir -p "$OUTPUT_DIR"

            echo "[$JOB_NUM/$TOTAL_LOCAL] $RUN_NAME"
            cd "$BASE_DIR"
            python "$SCRIPTS_DIR/run_subset_selection.py" \
                --config "$CONFIG_DIR/$BASE_CONFIG" \
                --override output_path="$OUTPUT_DIR" num_epochs="$NUM_EPOCHS" $OVERRIDES
            echo "[$JOB_NUM/$TOTAL_LOCAL] Done: $OUTPUT_DIR"
            echo ""
        done
        done
        done
        done

        echo "All $TOTAL_LOCAL local jobs complete!"
        echo "Output: $SWEEP_OUT_DIR"
        ;;

    "test")
        JOB_RANGE="1"
        echo ""
        echo "TEST MODE: Submitting 1 job to cluster"
        echo ""

        QSUB_SCRIPT="$SCRIPTS_DIR/subset_sweep.qsub.sh"
        if [ ! -f "$QSUB_SCRIPT" ]; then
            echo "Error: SGE script not found: $QSUB_SCRIPT"
            exit 1
        fi

        GPU_OPTION=""
        if [ "$SGE_GPU" = "true" ]; then
            GPU_OPTION="-l gpu=true"
            MEM_OPTION="-l tmem=${SGE_MEMORY}"
        else
            MEM_OPTION="-l h_vmem=${SGE_MEMORY}"
        fi

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
          -N "subset_${SWEEP_NAME}_test" \
          -o "$LOG_DIR" \
          -e "$LOG_DIR" \
          -v "$ENV_VARS" \
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

        QSUB_SCRIPT="$SCRIPTS_DIR/subset_sweep.qsub.sh"
        if [ ! -f "$QSUB_SCRIPT" ]; then
            echo "Error: SGE script not found: $QSUB_SCRIPT"
            exit 1
        fi

        GPU_OPTION=""
        if [ "$SGE_GPU" = "true" ]; then
            GPU_OPTION="-l gpu=true"
            MEM_OPTION="-l tmem=${SGE_MEMORY}"
        else
            MEM_OPTION="-l h_vmem=${SGE_MEMORY}"
        fi

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
          -N "subset_${SWEEP_NAME}" \
          -o "$LOG_DIR" \
          -e "$LOG_DIR" \
          -v "$ENV_VARS" \
          "$QSUB_SCRIPT"

        echo ""
        echo "Jobs submitted successfully!"
        echo "Output root: $SWEEP_OUT_DIR"
        echo "Logs: $LOG_DIR"
        ;;
esac
