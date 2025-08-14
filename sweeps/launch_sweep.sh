#!/bin/bash

# --- SETR Sweep Monitoring Script ---
# Usage: ./monitor_sweep.sh [sweep_name]

# Auto-detect base directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SWEEP_OUTPUT_DIR="$SCRIPT_DIR/output"

# Get sweep name from argument or show help
SWEEP_NAME="$1"

if [ -z "$SWEEP_NAME" ]; then
    echo "=== SETR Sweep Monitor ==="
    echo "Usage: $0 <sweep_name>"
    echo ""
    echo "Available sweep names (from output directories):"
    ls -1 $SWEEP_OUTPUT_DIR/ 2>/dev/null | sed 's/^/  /' || echo "  No sweeps found"
    echo ""
    echo "Current SGE jobs:"
    qstat -u $USER 2>/dev/null || echo "  qstat not available"
    exit 0
fi

echo "=== Sweep Monitor: $SWEEP_NAME ==="
echo ""

# Show SGE queue status for this sweep
echo "SGE Queue Status:"
if command -v qstat >/dev/null 2>&1; then
    JOBS=$(qstat -u $USER 2>/dev/null | grep "$SWEEP_NAME")
    if [ -n "$JOBS" ]; then
        echo "$JOBS"
        echo ""
        echo "Job Summary:"
        echo "  Running (r): $(echo "$JOBS" | grep -c ' r ')"
        echo "  Queued (qw): $(echo "$JOBS" | grep -c ' qw ')"
        echo "  Error (Eqw): $(echo "$JOBS" | grep -c ' Eqw ')"
        echo "  Total: $(echo "$JOBS" | wc -l)"
    else
        echo "  No jobs found for sweep: $SWEEP_NAME"
    fi
else
    echo "  qstat not available"
fi
echo ""

# Check output directory if it exists
if [ -d "$SWEEP_OUTPUT_DIR/$SWEEP_NAME" ]; then
    echo "Output Directory Status:"
    RESULT_DIRS=$(find "$SWEEP_OUTPUT_DIR/$SWEEP_NAME" -name "alpha_*_beta_*" -type d 2>/dev/null | wc -l)
    COMPLETED=$(find "$SWEEP_OUTPUT_DIR/$SWEEP_NAME" -name "job_completion.txt" 2>/dev/null | wc -l)
    
    echo "  Results directories: $RESULT_DIRS"
    echo "  Completed jobs: $COMPLETED"
    echo "  Directory: $SWEEP_OUTPUT_DIR/$SWEEP_NAME"
    echo ""
    
    # Show disk usage
    echo "Disk Usage:"
    du -sh "$SWEEP_OUTPUT_DIR/$SWEEP_NAME" 2>/dev/null || echo "  Unable to calculate"
else
    echo "Output directory not found: $SWEEP_OUTPUT_DIR/$SWEEP_NAME"
    echo "Sweep may not have started yet or different name used."
fi