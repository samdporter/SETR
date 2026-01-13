#!/bin/bash

# --- SETR Experiment Monitor Script ---
# Usage: ./monitor_experiments.sh [sweep_name]

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SWEEP_NAME="${1:-phantom_experiments}"
OUTPUT_DIR="$SCRIPT_DIR/output/$SWEEP_NAME"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Sweep output directory not found: $OUTPUT_DIR"
    echo ""
    echo "Available sweeps:"
    ls -1 "$SCRIPT_DIR/output" 2>/dev/null || echo "  None"
    exit 1
fi

echo "=== SETR Experiment Monitor: $SWEEP_NAME ==="
echo ""

# Count total experiment directories (excluding _logs)
TOTAL_DIRS=$(find "$OUTPUT_DIR" -maxdepth 1 -type d ! -name "_logs" ! -name "$(basename "$OUTPUT_DIR")" | wc -l)

# Count completed experiments
COMPLETED=0
FAILED=0
INCOMPLETE=0

for exp_dir in "$OUTPUT_DIR"/*; do
    # Skip if not a directory or if it's the _logs directory
    if [ ! -d "$exp_dir" ] || [ "$(basename "$exp_dir")" = "_logs" ]; then
        continue
    fi

    exp_name=$(basename "$exp_dir")
    completion_file="$exp_dir/job_completion.txt"

    if [ -f "$completion_file" ]; then
        if grep -q "status=completed" "$completion_file"; then
            COMPLETED=$((COMPLETED + 1))
        elif grep -q "status=failed" "$completion_file"; then
            FAILED=$((FAILED + 1))
        else
            INCOMPLETE=$((INCOMPLETE + 1))
        fi
    else
        INCOMPLETE=$((INCOMPLETE + 1))
    fi
done

# Display summary
echo "Summary:"
echo "  Total experiments: $TOTAL_DIRS"
echo "  Completed:         $COMPLETED"
echo "  Failed:            $FAILED"
echo "  Incomplete:        $INCOMPLETE"
echo ""

# Calculate progress percentage
if [ $TOTAL_DIRS -gt 0 ]; then
    PERCENT=$((COMPLETED * 100 / TOTAL_DIRS))
    echo "Progress: $PERCENT% ($COMPLETED/$TOTAL_DIRS)"
else
    echo "Progress: No experiments found"
fi
echo ""

# Show running jobs
echo "Running SGE jobs:"
RUNNING_JOBS=$(qstat 2>/dev/null | grep "setr_${SWEEP_NAME}" | wc -l)
if [ "$RUNNING_JOBS" -gt 0 ]; then
    qstat | grep "setr_${SWEEP_NAME}" || true
else
    echo "  No jobs currently running"
fi
echo ""

# List failed experiments with details
if [ $FAILED -gt 0 ]; then
    echo "Failed experiments:"
    for exp_dir in "$OUTPUT_DIR"/*; do
        if [ ! -d "$exp_dir" ] || [ "$(basename "$exp_dir")" = "_logs" ]; then
            continue
        fi

        completion_file="$exp_dir/job_completion.txt"
        if [ -f "$completion_file" ] && grep -q "status=failed" "$completion_file"; then
            exp_name=$(basename "$exp_dir")
            failure_reason=$(grep -o "failure_reason=[^,]*" "$completion_file" | cut -d= -f2 || echo "unknown")
            failure_type=$(grep -o "failure_type=[^,]*" "$completion_file" | cut -d= -f2 || echo "unknown")
            host=$(grep -o "host=[^,]*" "$completion_file" | cut -d= -f2 || echo "unknown")
            echo "  - $exp_name: $failure_reason (type: $failure_type, host: $host)"
        fi
    done
    echo ""
fi

# List incomplete experiments
if [ $INCOMPLETE -gt 0 ]; then
    echo "Incomplete experiments:"
    for exp_dir in "$OUTPUT_DIR"/*; do
        if [ ! -d "$exp_dir" ] || [ "$(basename "$exp_dir")" = "_logs" ]; then
            continue
        fi

        completion_file="$exp_dir/job_completion.txt"
        if [ ! -f "$completion_file" ] || ! grep -q "status=" "$completion_file"; then
            exp_name=$(basename "$exp_dir")
            echo "  - $exp_name"
        fi
    done
    echo ""
fi

# Show recent log activity
echo "Recent log activity (last 5 modified):"
LOG_DIR="$OUTPUT_DIR/_logs"
if [ -d "$LOG_DIR" ]; then
    ls -lt "$LOG_DIR" 2>/dev/null | head -6 | tail -5 | awk '{print "  " $9 " (" $6, $7, $8 ")"}' || echo "  No logs found"
else
    echo "  Log directory not found"
fi
echo ""

# Optionally show details for specific experiment
if [ $# -ge 2 ]; then
    EXPERIMENT="$2"
    EXP_DIR="$OUTPUT_DIR/$EXPERIMENT"

    if [ -d "$EXP_DIR" ]; then
        echo "=== Details: $EXPERIMENT ==="
        if [ -f "$EXP_DIR/job_completion.txt" ]; then
            cat "$EXP_DIR/job_completion.txt"
        else
            echo "  No completion file found"
        fi
        echo ""

        echo "Output files:"
        find "$EXP_DIR" -type f \( -name "*.hv" -o -name "*.nii*" -o -name "objective.csv" \) 2>/dev/null | head -10 | sed 's/^/  /'
        echo ""
    else
        echo "Experiment not found: $EXPERIMENT"
    fi
fi

echo "=== Tips ==="
echo "View specific experiment: $0 $SWEEP_NAME <experiment_name>"
echo "Check logs: ls -lth $LOG_DIR | head"
echo "Tail specific log: tail -f $LOG_DIR/setr_${SWEEP_NAME}.o<job_id>.<task_id>"
