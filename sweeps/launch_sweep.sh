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
FAILED_NODES_FILE="$SWEEPS_DIR/failed_nodes.txt"
PERMANENT_EXCLUSIONS="!hoots-207-2.local&!hoots-207-1.local"

# Get sweep config from argument
SWEEP_CONFIG="$1"
MODE="${2:-full}"
FORCE_FLAG="${3:-}"

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

check_for_node_failure() {
    log_file="$1"
    
    if [ ! -f "$log_file" ]; then
        return
    fi
    
    # Extract hostname from log
    hostname=$(grep "^Host:" "$log_file" | awk '{print $2}' | head -1)
    
    if [ -z "$hostname" ]; then
        return
    fi
    
    # Check for node-related failure patterns - MUCH MORE COMPREHENSIVE
    node_failure=false
    failure_reason=""
    
    # GPU/CUDA errors - IMPROVED PATTERNS
    if grep -q -E "(cudaMalloc.*error|CUDA.*error|GPU.*error)" "$log_file"; then
        echo "Detected CUDA/GPU errors on node: $hostname"
        node_failure=true
        failure_reason="CUDA/GPU error"
    fi
    
    # Specific ECC error patterns
    if grep -q -E "(uncorrectable ECC error|ECC.*error)" "$log_file"; then
        echo "Detected ECC errors on node: $hostname"
        node_failure=true
        failure_reason="ECC error"
    fi
    
    # GPU busy/unavailable
    if grep -q -E "(device.*busy.*unavailable|GPU.*busy|GPU.*unavailable)" "$log_file"; then
        echo "Detected GPU busy/unavailable on node: $hostname"
        node_failure=true
        failure_reason="GPU busy/unavailable"
    fi
    
    # CUDA memory allocation failures
    if grep -q -E "(cudaMalloc.*failed|CUDA memory|cuda.*out of memory)" "$log_file"; then
        echo "Detected CUDA memory issues on node: $hostname"
        node_failure=true
        failure_reason="CUDA memory error"
    fi
    
    # Out of memory (system level)
    if grep -q -E "(Out of memory|Cannot allocate memory|Killed.*memory)" "$log_file"; then
        echo "Detected memory issues on node: $hostname"
        node_failure=true
        failure_reason="System memory error"
    fi
    
    # Disk space issues
    if grep -q -E "(No space left|Disk quota exceeded)" "$log_file"; then
        echo "Detected disk space issues on node: $hostname"
        node_failure=true
        failure_reason="Disk space error"
    fi
    
    # GPU prolog/epilog failures
    if grep -q -E "(Not enough GPUs available|GPU.*failed|nvidia-smi.*failed)" "$log_file"; then
        echo "Detected GPU allocation/management issues on node: $hostname"
        node_failure=true
        failure_reason="GPU allocation error"
    fi
    
    if [ "$node_failure" = true ]; then
        echo "Adding $hostname to blacklist for: $failure_reason"
        add_failed_node "$hostname"
    fi
}

add_failed_node() {
    local hostname="$1"
    
    # Check if node is already in the failed list
    if [ -f "$FAILED_NODES_FILE" ] && grep -q "^${hostname}$" "$FAILED_NODES_FILE"; then
        echo "Node $hostname already in failed nodes list"
        return
    fi
    
    echo "$hostname" >> "$FAILED_NODES_FILE"
    echo "Added $hostname to failed nodes list"
}

show_usage() {
    echo "=== SETR Sweep Launcher ==="
    echo "Usage: $0 <sweep_config.yaml> [mode]"
    echo ""
    echo "Modes:"
    echo "  full    - Run complete sweep (default)"
    echo "  test    - Submit only one test job"
    echo "  local   - Run first parameter combination locally (no qsub)"
    echo ""
    echo "Available sweep configs:"
    ls -1 "$CONFIG_DIR"/*.yaml 2>/dev/null | sed 's/^/  /' || echo "  No configs found"
    echo ""
    echo "Failed nodes management:"
    echo "  View failed nodes: cat $FAILED_NODES_FILE"
    echo "  Remove node from blacklist: grep -v 'nodename' $FAILED_NODES_FILE > tmp && mv tmp $FAILED_NODES_FILE"
}

if [ -z "$SWEEP_CONFIG" ]; then
    show_usage
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
print('BASE_CONFIG=' + config['base_config'])
print('RECON_SCRIPT=' + config['script'])
print('ALPHA_FILE=' + config['parameters']['alpha_file'])
print('BETA_FILE=' + config['parameters']['beta_file'])
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
echo "Base config: $BASE_CONFIG"
echo "Script: $RECON_SCRIPT"
echo "SGE resources: $SGE_RUNTIME, $SGE_MEMORY, $SGE_CORES cores"

# Parse config overrides (if any) into CLI-friendly arguments
CONFIG_OVERRIDE_ARGS=()
if [ -n "${CONFIG_OVERRIDES_JSON:-}" ] && [ "$CONFIG_OVERRIDES_JSON" != "{}" ]; then
    while IFS= read -r line; do
        CONFIG_OVERRIDE_ARGS+=("$line")
    done < <(python3 - <<'PY' "$CONFIG_OVERRIDES_JSON"
import json, sys
overrides = json.loads(sys.argv[1])
for key, value in overrides.items():
    print(f"{key}={value!r}")
PY
    )
fi

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

# Prepare sweep output + logs
SWEEP_OUT_DIR="$SWEEPS_DIR/output/$SWEEP_NAME"
LOG_DIR="$SWEEP_OUT_DIR/_logs"
mkdir -p "$LOG_DIR"

# Handle local mode early (no qsub submission)
if [ "$MODE" = "local" ]; then
    echo ""
    echo "LOCAL TEST MODE: Running first parameter combination locally"
    echo ""

    BASE_CONFIG_PATH="$BASE_DIR/configs/$BASE_CONFIG"
    RECON_SCRIPT_PATH="$BASE_DIR/scripts/$RECON_SCRIPT"

    if [ ! -f "$RECON_SCRIPT_PATH" ]; then
        echo "Error: Reconstruction script not found: $RECON_SCRIPT_PATH"
        exit 1
    fi

    if [ ! -f "$BASE_CONFIG_PATH" ]; then
        echo "Error: Base config file not found: $BASE_CONFIG_PATH"
        exit 1
    fi

    FIRST_ALPHA=$(python3 - <<'PY' "$PARAM_DIR/$ALPHA_FILE"
import csv, sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open() as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        if not row:
            continue
        value = row[0].strip()
        if value:
            print(value)
            sys.exit(0)

raise SystemExit(1)
PY
    ) || {
        echo "Error: Failed to read first alpha value from $ALPHA_FILE"
        exit 1
    }

    FIRST_BETA=$(python3 - <<'PY' "$PARAM_DIR/$BETA_FILE"
import csv, sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open() as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        if not row:
            continue
        value = row[0].strip()
        if value:
            print(value)
            sys.exit(0)

raise SystemExit(1)
PY
    ) || {
        echo "Error: Failed to read first beta value from $BETA_FILE"
        exit 1
    }

    echo "Testing alpha=$FIRST_ALPHA, beta=$FIRST_BETA"

    OUTPUT_DIR="$SWEEP_OUT_DIR/local_test"
    WORKING_DIR="$OUTPUT_DIR/tmp"
    mkdir -p "$WORKING_DIR"

    LOCAL_OVERRIDE_ARGS=(
        "output_path=$OUTPUT_DIR"
        "working_path=$WORKING_DIR"
        "alpha=$FIRST_ALPHA"
        "beta=$FIRST_BETA"
    )
    if [ ${#CONFIG_OVERRIDE_ARGS[@]} -gt 0 ]; then
        LOCAL_OVERRIDE_ARGS+=("${CONFIG_OVERRIDE_ARGS[@]}")
    fi

    cd "$BASE_DIR"
    python "$RECON_SCRIPT_PATH" \
        --config "$BASE_CONFIG_PATH" \
        --override "${LOCAL_OVERRIDE_ARGS[@]}"

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
QSUB_SCRIPT="$SCRIPTS_DIR/sweep_alpha_beta.qsub.sh"
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
  -N \"setr_$SWEEP_NAME\" \
  -o \"$LOG_DIR\" \
  -e \"$LOG_DIR\" \
  -v \"SETR_BASE_DIR=$BASE_DIR,SWEEP_NAME=$SWEEP_NAME,BASE_CONFIG_FILE=$BASE_CONFIG,RECON_SCRIPT=$RECON_SCRIPT,ALPHA_FILE=$ALPHA_FILE,BETA_FILE=$BETA_FILE\" \
  \"$QSUB_SCRIPT\""

echo ""
echo "Submitting jobs to SGE..."
echo "$CMD"
eval "$CMD"

echo ""
echo "Jobs submitted successfully!"
echo "Monitor with: ./monitor_sweep.sh $SWEEP_NAME"
echo "Output root: $SWEEP_OUT_DIR"
echo "Logs: $LOG_DIR"
echo ""
echo "Failed nodes file: $FAILED_NODES_FILE"
