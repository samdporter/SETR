#!/bin/bash

# --- SETR Experiment Launcher Script ---
# Usage: ./launch_experiments.sh <sweep_config.yaml> [mode]

set -euo pipefail

# Auto-detect base directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENTS_DIR="$SCRIPT_DIR"
CONFIG_DIR="$EXPERIMENTS_DIR/configs"
SCRIPTS_DIR="$EXPERIMENTS_DIR/scripts"
FAILED_NODES_FILE="$EXPERIMENTS_DIR/failed_nodes.txt"
PERMANENT_EXCLUSIONS="!hoots-207-2.local&!hoots-207-1.local"

# Get sweep config from argument
SWEEP_CONFIG="${1:-sweep_phantom_experiments.yaml}"
MODE="${2:-full}"

# Functions for node management
initialize_failed_nodes_file() {
    if [ ! -f "$FAILED_NODES_FILE" ]; then
        touch "$FAILED_NODES_FILE"
        echo "Created failed nodes tracking file: $FAILED_NODES_FILE"
    fi
}

get_hostname_exclusions() {
    if [ ! -f "$FAILED_NODES_FILE" ] || [ ! -s "$FAILED_NODES_FILE" ]; then
        echo ""
        return
    fi

    local exclusions=""
    while IFS= read -r node; do
        # Skip empty lines and comments
        [[ -z "$node" || "$node" =~ ^[[:space:]]* ]] && continue

        if [ -z "$exclusions" ]; then
            exclusions="!${node}"
        else
            exclusions="${exclusions}&!${node}"
        fi
    done < "$FAILED_NODES_FILE"

    if [ -n "$exclusions" ]; then
        echo "$exclusions"
    else
        echo ""
    fi
}

show_usage() {
    echo "=== SETR Experiment Launcher ==="
    echo "Usage: $0 [sweep_config.yaml] [mode]"
    echo ""
    echo "Modes:"
    echo "  full    - Run complete sweep (default)"
    echo "  test    - Submit only one test job"
    echo "  local   - Run first experiment locally (no qsub)"
    echo ""
    echo "Available sweep configs:"
    ls -1 "$CONFIG_DIR"/*.yaml 2>/dev/null | sed 's/^/  /' || echo "  No configs found"
    echo ""
    echo "Failed nodes management:"
    echo "  View failed nodes: cat $FAILED_NODES_FILE"
    echo "  Remove node from blacklist: grep -v 'nodename' $FAILED_NODES_FILE > tmp && mv tmp $FAILED_NODES_FILE"
}

if [ "$SWEEP_CONFIG" = "--help" ] || [ "$SWEEP_CONFIG" = "-h" ]; then
    show_usage
    exit 0
fi

# Check if config exists
SWEEP_CONFIG_PATH="$CONFIG_DIR/$SWEEP_CONFIG"
if [ ! -f "$SWEEP_CONFIG_PATH" ]; then
    echo "Error: Sweep config not found: $SWEEP_CONFIG_PATH"
    echo ""
    show_usage
    exit 1
fi

echo "=== SETR Experiment Launcher ==="
echo "Config: $SWEEP_CONFIG"
echo "Mode: $MODE"
echo "Base directory: $BASE_DIR"

# Initialize failed nodes tracking
initialize_failed_nodes_file

# Parse YAML config using Python
CONFIG_VALUES=$(python3 -c "
import json, yaml, sys
with open('$SWEEP_CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)

print('SWEEP_NAME=' + config['sweep_name'])
print('EXPERIMENT_TYPE=' + config.get('experiment_type', 'phantom_algorithm'))
print('PHANTOMS=' + ','.join(config.get('phantoms', [])))
print('ALGORITHMS=' + ','.join(config.get('algorithms', [])))
print('SGE_RUNTIME=' + config['sge']['runtime'])
print('SGE_MEMORY=' + config['sge']['memory'])
print('SGE_CORES=' + str(config['sge']['cores']))
print('SGE_QUEUE=' + (config['sge']['queue'] or 'default'))
print('SGE_GPU=' + str(config['sge'].get('gpu', False)).lower())
print('CONFIG_OVERRIDES_JSON=' + json.dumps(config.get('config_overrides', {})))
")

# Source the config values
eval "$CONFIG_VALUES"

echo "Sweep name: $SWEEP_NAME"
echo "Experiment type: $EXPERIMENT_TYPE"
echo "Phantoms: $PHANTOMS"
echo "Algorithms: $ALGORITHMS"
echo "SGE resources: $SGE_RUNTIME, $SGE_MEMORY, $SGE_CORES cores"

# Calculate total jobs
IFS=',' read -ra PHANTOM_ARRAY <<< "$PHANTOMS"
IFS=',' read -ra ALGORITHM_ARRAY <<< "$ALGORITHMS"
NUM_PHANTOMS=${#PHANTOM_ARRAY[@]}
NUM_ALGORITHMS=${#ALGORITHM_ARRAY[@]}
TOTAL_JOBS=$((NUM_PHANTOMS * NUM_ALGORITHMS))

echo "Combinations: $NUM_PHANTOMS phantoms × $NUM_ALGORITHMS algorithms = $TOTAL_JOBS total jobs"

if [ "$TOTAL_JOBS" -eq 0 ]; then
    echo "Error: No experiment combinations found"
    exit 1
fi

# Prepare output + logs
OUTPUT_BASE_DIR="$EXPERIMENTS_DIR/output"
SWEEP_OUT_DIR="$OUTPUT_BASE_DIR/$SWEEP_NAME"
LOG_DIR="$SWEEP_OUT_DIR/_logs"
mkdir -p "$LOG_DIR"

# Handle local mode early (no qsub submission)
if [ "$MODE" = "local" ]; then
    echo ""
    echo "LOCAL TEST MODE: Running first experiment locally"
    echo ""

    FIRST_PHANTOM=${PHANTOM_ARRAY[0]}
    FIRST_ALGORITHM=${ALGORITHM_ARRAY[0]}

    echo "Testing phantom=$FIRST_PHANTOM, algorithm=$FIRST_ALGORITHM"

    OUTPUT_DIR="$SWEEP_OUT_DIR/local_test_${FIRST_PHANTOM}_${FIRST_ALGORITHM}"
    WORKING_DIR="$OUTPUT_DIR/tmp"
    mkdir -p "$WORKING_DIR"

    LOCAL_OVERRIDE_ARGS=(
        "output_path=$OUTPUT_DIR"
        "working_path=$WORKING_DIR"
    )

    # Add config overrides if any
    if [ -n "${CONFIG_OVERRIDES_JSON:-}" ] && [ "$CONFIG_OVERRIDES_JSON" != "{}" ]; then
        while IFS= read -r line; do
            LOCAL_OVERRIDE_ARGS+=("$line")
        done < <(python3 - <<'PY' "$CONFIG_OVERRIDES_JSON"
import json, sys
overrides = json.loads(sys.argv[1])
for key, value in overrides.items():
    if isinstance(value, bool):
        print(f"{key}={str(value).lower()}")
    elif isinstance(value, str):
        print(f"{key}={value}")
    else:
        print(f"{key}={value}")
PY
        )
    fi

    cd "$BASE_DIR"

    MASTER_SCRIPT="$BASE_DIR/scripts/run_phantom_experiments.py"
    if [ ! -f "$MASTER_SCRIPT" ]; then
        echo "Error: Master script not found: $MASTER_SCRIPT"
        exit 1
    fi

    CMD_ARGS=("--phantom" "$FIRST_PHANTOM" "--algorithm" "$FIRST_ALGORITHM")
    for override in "${LOCAL_OVERRIDE_ARGS[@]}"; do
        CMD_ARGS+=("--override" "$override")
    done

    echo "Running: python $MASTER_SCRIPT ${CMD_ARGS[*]}"
    python "$MASTER_SCRIPT" "${CMD_ARGS[@]}"

    echo ""
    echo "Local test complete!"
    echo "Output: $OUTPUT_DIR"
    exit 0
fi

# Determine job range based on mode
JOB_RANGE=""
case "$MODE" in
    "test")
        JOB_RANGE="1"
        echo "TEST MODE: Submitting only 1 job"
        ;;
    "full"|*)
        JOB_RANGE="1-$TOTAL_JOBS"
        echo "FULL MODE: Submitting $TOTAL_JOBS jobs"
        ;;
esac

# Build qsub command
QSUB_SCRIPT="$SCRIPTS_DIR/run_phantom_experiment.qsub.sh"
if [ ! -f "$QSUB_SCRIPT" ]; then
    echo "Error: SGE script not found: $QSUB_SCRIPT"
    exit 1
fi

# Get dynamic hostname exclusions for failed nodes
DYNAMIC_EXCLUSIONS=$(get_hostname_exclusions)
if [ -n "$DYNAMIC_EXCLUSIONS" ]; then
    echo "Excluding failed nodes: $DYNAMIC_EXCLUSIONS"
    HOSTNAME_OPTION="-l hostname='$DYNAMIC_EXCLUSIONS&$PERMANENT_EXCLUSIONS'"
else
    HOSTNAME_OPTION="-l hostname='$PERMANENT_EXCLUSIONS'"
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

# Set GPU option
GPU_OPTION=""
if [ "$SGE_GPU" = "true" ]; then
    # GPU job → use tmem
    GPU_OPTION="-l gpu=true"
    MEM_OPTION="-l tmem=${SGE_MEMORY}"
else
    # CPU job → use h_vmem
    MEM_OPTION="-l h_vmem=${SGE_MEMORY}"
fi

# Convert config overrides to JSON for passing to qsub
if [ -n "${CONFIG_OVERRIDES_JSON:-}" ]; then
    CONFIG_OVERRIDES_ESCAPED=$(echo "$CONFIG_OVERRIDES_JSON" | sed 's/"/\\"/g')
else
    CONFIG_OVERRIDES_ESCAPED="{}"
fi

# Submit to SGE
CMD="qsub \
  -t \"$JOB_RANGE\" \
  -r y \
  -l h_rt=\"$SGE_RUNTIME\" \
  $HOSTNAME_OPTION \
  $MEM_OPTION \
  $GPU_OPTION \
  $PE_OPTION \
  $QUEUE_OPTION \
  -N \"setr_${SWEEP_NAME}\" \
  -o \"$LOG_DIR\" \
  -e \"$LOG_DIR\" \
  -v \"SETR_BASE_DIR=$BASE_DIR,SWEEP_NAME=$SWEEP_NAME,PHANTOMS=$PHANTOMS,ALGORITHMS=$ALGORITHMS,CONFIG_OVERRIDES_JSON=$CONFIG_OVERRIDES_ESCAPED\" \
  \"$QSUB_SCRIPT\""

echo ""
echo "Submitting jobs to SGE..."
echo "$CMD"
eval "$CMD"

echo ""
echo "Jobs submitted successfully!"
echo "Output root: $SWEEP_OUT_DIR"
echo "Logs: $LOG_DIR"
echo ""
echo "Monitor jobs with: qstat"
echo "Failed nodes file: $FAILED_NODES_FILE"
echo ""
echo "Experiment combinations:"
for phantom in "${PHANTOM_ARRAY[@]}"; do
    for algorithm in "${ALGORITHM_ARRAY[@]}"; do
        echo "  - $phantom + $algorithm"
    done
done
