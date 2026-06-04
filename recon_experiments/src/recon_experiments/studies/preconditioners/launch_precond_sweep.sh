#!/bin/bash

# --- SETR Preconditioner Sweep Launcher ---
# Usage: ./launch_precond_sweep.sh <sweep_config.yaml> [local|local_all|test|full]

set -euo pipefail

# Auto-detect base directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
FUNC_DIR="$SCRIPT_DIR"
CONFIG_DIR="$FUNC_DIR/configs"
PARAM_DIR="$FUNC_DIR/parameters"
SCRIPTS_DIR="$FUNC_DIR/scripts"

# Get sweep config from argument
SWEEP_CONFIG="${1:-precond_sweep_1bpos.yaml}"
MODE="${2:-full}"
SWEEP_REPEATS="${SWEEP_REPEATS:-1}"

if [[ "$MODE" =~ ^local(_all)?$ ]]; then
    export LOCAL_RUN_ID="${LOCAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
fi

if ! [[ "$SWEEP_REPEATS" =~ ^[0-9]+$ ]] || [ "$SWEEP_REPEATS" -lt 1 ]; then
    echo "Error: SWEEP_REPEATS must be a positive integer (got '$SWEEP_REPEATS')"
    exit 1
fi

show_usage() {
    echo "=== SETR Preconditioner Sweep Launcher ==="
    echo "Usage: $0 [sweep_config.yaml] [mode]"
    echo ""
    echo "Modes:"
    echo "  full    - Submit all jobs to cluster (default)"
    echo "  test    - Submit only one test job to cluster"
    echo "  local   - Run one test locally (no SGE submission)"
    echo "  local_all - Run all parameter combinations locally, sequentially"
    echo ""
    echo "Env:"
    echo "  SWEEP_REPEATS - Number of repeats per parameter combination (default: 1)"
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
import shlex
import base64
with open('$SWEEP_CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)
fixed = config.get('fixed_params', {}) or {}
num_epochs = fixed.get('num_epochs', 50)
fixed_overrides = [
    f\"{k}={repr(v)}\" for k, v in fixed.items() if k != 'num_epochs'
]
fixed_overrides_text = ';'.join(fixed_overrides)

def emit(key, value):
    print(f\"{key}={shlex.quote(str(value))}\")

emit('SWEEP_NAME', config['sweep_name'])
emit('BASE_CONFIG', config['base_config'])
emit('RECON_SCRIPT', config['script'])
emit('PRECOND_TYPES_FILE', config['parameters']['precond_types_file'])
emit('ALPHAS_FILE', config['parameters']['alphas_file'])
emit('STEP_SIZES_FILE', config['parameters']['step_sizes_file'])
emit('SGE_RUNTIME', config['sge']['runtime'])
emit('SGE_MEMORY', config['sge']['memory'])
emit('SGE_CORES', config['sge']['cores'])
emit('SGE_QUEUE', config['sge']['queue'] or 'default')
emit('SGE_GPU', str(config['sge'].get('gpu', False)).lower())
emit('NUM_EPOCHS', num_epochs)
emit('FIXED_OVERRIDES', fixed_overrides_text)
emit('FIXED_OVERRIDES_B64', base64.b64encode(fixed_overrides_text.encode()).decode())
")

# Source the config values
eval "$CONFIG_VALUES"

echo "Sweep name: $SWEEP_NAME"
echo "Base config: $BASE_CONFIG"
echo "Epochs: $NUM_EPOCHS"
echo "SGE resources: $SGE_RUNTIME, $SGE_MEMORY, $SGE_CORES cores, GPU=$SGE_GPU"
echo "Sweep repeats: $SWEEP_REPEATS"

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
TOTAL_BASE_JOBS=$((NUM_PRECOND_TYPES * NUM_ALPHAS * NUM_STEP_SIZES))
TOTAL_CLUSTER_JOBS=$((TOTAL_BASE_JOBS * SWEEP_REPEATS))

echo "Parameters: $NUM_PRECOND_TYPES precond types × $NUM_ALPHAS alphas × $NUM_STEP_SIZES step sizes = $TOTAL_BASE_JOBS base combinations"
if [ "$SWEEP_REPEATS" -gt 1 ]; then
    echo "Total sweep jobs with repeats: $TOTAL_CLUSTER_JOBS"
fi

if [ "$TOTAL_BASE_JOBS" -eq 0 ]; then
    echo "Error: No parameter combinations found"
    exit 1
fi

# Prepare sweep output + logs
OUTPUT_ROOT="${PRECOND_OUTPUT_ROOT:-$FUNC_DIR/output}"
if [[ "$MODE" =~ ^local(_all)?$ && -n "${LOCAL_RUN_ID:-}" && -z "${PRECOND_OUTPUT_ROOT:-}" ]]; then
    OUTPUT_ROOT="$FUNC_DIR/output/local_runs/$LOCAL_RUN_ID"
fi
SWEEP_OUT_DIR="$OUTPUT_ROOT/$SWEEP_NAME"
LOG_DIR="$SWEEP_OUT_DIR/_logs"
mkdir -p "$LOG_DIR"

# Handle different modes
case "$MODE" in
    "local")
        echo ""
        echo "LOCAL TEST MODE: Running first parameter combination locally"
        if [ -n "${LOCAL_RUN_ID:-}" ]; then
            echo "Local run ID: $LOCAL_RUN_ID"
        fi
        echo ""

        # Read first parameter combination
        PRECOND_TYPE=$(tail -n +2 "$PARAM_DIR/$PRECOND_TYPES_FILE" | head -1 | awk -F, '{gsub(/^[ \t]+|[ \t]+$/,"",$1); print $1}')
        PRECOND_COMBINE=$(tail -n +2 "$PARAM_DIR/$PRECOND_TYPES_FILE" | head -1 | awk -F, '{gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2}')
        ALPHA=$(tail -n +2 "$PARAM_DIR/$ALPHAS_FILE" | head -1 | awk -F, '{gsub(/^[ \t]+|[ \t]+$/,"",$1); print $1}')
        STEP_SIZE=$(tail -n +2 "$PARAM_DIR/$STEP_SIZES_FILE" | head -1 | awk -F, '{gsub(/^[ \t]+|[ \t]+$/,"",$1); print $1}')

        echo "Testing: precond_type=$PRECOND_TYPE, combine=$PRECOND_COMBINE, alpha=$ALPHA, step_size=$STEP_SIZE"

        if [ -n "$PRECOND_COMBINE" ]; then
            OUTPUT_DIR="$SWEEP_OUT_DIR/precond_${PRECOND_TYPE}_combine_${PRECOND_COMBINE}_alpha_${ALPHA}_step_${STEP_SIZE}_local"
        else
            OUTPUT_DIR="$SWEEP_OUT_DIR/precond_${PRECOND_TYPE}_alpha_${ALPHA}_step_${STEP_SIZE}_local"
        fi
        mkdir -p "$OUTPUT_DIR"

        cd "$BASE_DIR"
        OVERRIDE_ARGS=()
        if [ -n "${FIXED_OVERRIDES:-}" ]; then
            IFS=';' read -ra FIXED_LIST <<< "$FIXED_OVERRIDES"
            for ov in "${FIXED_LIST[@]}"; do
                [ -n "$ov" ] && OVERRIDE_ARGS+=(--override "$ov")
            done
        fi
        COMBINE_ARGS=(--precond-combine "$PRECOND_COMBINE")
        python "$SCRIPTS_DIR/$RECON_SCRIPT" \
            --config "configs/$BASE_CONFIG" \
            --output "$OUTPUT_DIR" \
            --precond-type "$PRECOND_TYPE" \
            --alpha "$ALPHA" \
            --step-size "$STEP_SIZE" \
            --epochs "$NUM_EPOCHS" \
            "${COMBINE_ARGS[@]}" \
            "${OVERRIDE_ARGS[@]}"

        echo ""
        echo "Local test complete!"
        echo "Output: $OUTPUT_DIR"
        ;;
    "local_all")
        echo ""
        echo "LOCAL_ALL MODE: Running all parameter combinations locally"
        if [ -n "${LOCAL_RUN_ID:-}" ]; then
            echo "Local run ID: $LOCAL_RUN_ID"
        fi
        echo ""

        mapfile -t PRECOND_TYPES < <(tail -n +2 "$PARAM_DIR/$PRECOND_TYPES_FILE" | awk -F, 'NF{gsub(/^[ \t]+|[ \t]+$/,"",$1); gsub(/^[ \t]+|[ \t]+$/,"",$2); if($1!="") print $1 "," $2}')
        mapfile -t ALPHAS < <(tail -n +2 "$PARAM_DIR/$ALPHAS_FILE" | awk -F, 'NF{gsub(/^[ \t]+|[ \t]+$/,"",$1); if($1!="") print $1}')
        mapfile -t STEP_SIZES < <(tail -n +2 "$PARAM_DIR/$STEP_SIZES_FILE" | awk -F, 'NF{gsub(/^[ \t]+|[ \t]+$/,"",$1); if($1!="") print $1}')

        cd "$BASE_DIR"

        TOTAL_LOCAL_JOBS=$((TOTAL_BASE_JOBS * SWEEP_REPEATS))
        job_idx=0
        for PRECOND_LINE in "${PRECOND_TYPES[@]}"; do
            IFS=',' read -r PRECOND_TYPE PRECOND_COMBINE <<< "$PRECOND_LINE"
            PRECOND_TYPE="${PRECOND_TYPE//[$'\t\r\n ']}"
            PRECOND_COMBINE="${PRECOND_COMBINE//[$'\t\r\n ']}"

            for ALPHA in "${ALPHAS[@]}"; do
                for STEP_SIZE in "${STEP_SIZES[@]}"; do
                    for REPEAT in $(seq 1 "$SWEEP_REPEATS"); do
                        job_idx=$((job_idx + 1))
                        echo "[$job_idx/$TOTAL_LOCAL_JOBS] precond_type=$PRECOND_TYPE combine=$PRECOND_COMBINE alpha=$ALPHA step_size=$STEP_SIZE repeat=$REPEAT/$SWEEP_REPEATS"

                        if [ -n "$PRECOND_COMBINE" ]; then
                            OUTPUT_DIR="$SWEEP_OUT_DIR/precond_${PRECOND_TYPE}_combine_${PRECOND_COMBINE}_alpha_${ALPHA}_step_${STEP_SIZE}"
                        else
                            OUTPUT_DIR="$SWEEP_OUT_DIR/precond_${PRECOND_TYPE}_alpha_${ALPHA}_step_${STEP_SIZE}"
                        fi

                        if [ "$SWEEP_REPEATS" -gt 1 ]; then
                            OUTPUT_DIR="${OUTPUT_DIR}_rep_${REPEAT}"
                        fi

                        mkdir -p "$OUTPUT_DIR"

                        OVERRIDE_ARGS=()
                        if [ -n "${FIXED_OVERRIDES:-}" ]; then
                            IFS=';' read -ra FIXED_LIST <<< "$FIXED_OVERRIDES"
                            for ov in "${FIXED_LIST[@]}"; do
                                [ -n "$ov" ] && OVERRIDE_ARGS+=(--override "$ov")
                            done
                        fi
                        COMBINE_ARGS=(--precond-combine "$PRECOND_COMBINE")

                        python "$SCRIPTS_DIR/$RECON_SCRIPT" \
                            --config "configs/$BASE_CONFIG" \
                            --output "$OUTPUT_DIR" \
                            --precond-type "$PRECOND_TYPE" \
                            --alpha "$ALPHA" \
                            --step-size "$STEP_SIZE" \
                            --epochs "$NUM_EPOCHS" \
                            "${COMBINE_ARGS[@]}" \
                            "${OVERRIDE_ARGS[@]}"
                    done
                done
            done
        done

        echo ""
        echo "Local-all sweep complete!"
        echo "Output root: $SWEEP_OUT_DIR"
        ;;

    "test")
        JOB_RANGE="1"
        echo ""
        echo "TEST MODE: Submitting 1 job to cluster (task 1 of $TOTAL_CLUSTER_JOBS)"
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
          -v "SETR_BASE_DIR=$BASE_DIR,SWEEP_NAME=${SWEEP_NAME},BASE_CONFIG_FILE=$BASE_CONFIG,PRECOND_TYPES_FILE=$PRECOND_TYPES_FILE,ALPHAS_FILE=$ALPHAS_FILE,STEP_SIZES_FILE=$STEP_SIZES_FILE,NUM_EPOCHS=$NUM_EPOCHS,RECON_SCRIPT=$RECON_SCRIPT,FIXED_OVERRIDES_B64=$FIXED_OVERRIDES_B64,SWEEP_REPEATS=$SWEEP_REPEATS" \
          "$QSUB_SCRIPT"

        echo ""
        echo "Test job submitted!"
        echo "Output root: $SWEEP_OUT_DIR"
        echo "Logs: $LOG_DIR"
        ;;

    "full")
        JOB_RANGE="1-$TOTAL_CLUSTER_JOBS"
        echo ""
        echo "FULL MODE: Submitting $TOTAL_CLUSTER_JOBS jobs to cluster"
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
          -v "SETR_BASE_DIR=$BASE_DIR,SWEEP_NAME=${SWEEP_NAME},BASE_CONFIG_FILE=$BASE_CONFIG,PRECOND_TYPES_FILE=$PRECOND_TYPES_FILE,ALPHAS_FILE=$ALPHAS_FILE,STEP_SIZES_FILE=$STEP_SIZES_FILE,NUM_EPOCHS=$NUM_EPOCHS,RECON_SCRIPT=$RECON_SCRIPT,FIXED_OVERRIDES_B64=$FIXED_OVERRIDES_B64,SWEEP_REPEATS=$SWEEP_REPEATS" \
          "$QSUB_SCRIPT"

        echo ""
        echo "Jobs submitted successfully!"
        echo "Output root: $SWEEP_OUT_DIR"
        echo "Logs: $LOG_DIR"
        ;;

    *)
        echo "Error: Unknown mode '$MODE'"
        echo "Valid modes: full, test, local, local_all"
        show_usage
        exit 1
        ;;
esac
