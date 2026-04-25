#!/bin/bash
# ================================
# Launcher for MANC Bootstrap HKEM Reconstruction
# ================================
# This script launches SGE jobs to run HKEM reconstruction for all MANC NEMA
# phantom bootstrap datasets. The workflow is:
#   1. Run SPECT HKEM reconstruction (single job, runs once)
#   2. Resample SPECT to PET space (array job, one per bootstrap, waits for SPECT)
#   3. Run PET HKEM reconstruction (array job, one per bootstrap, waits for resampling)
#
# Usage:
#   ./launch_manc_bootstrap_hkem.sh [num_bootstraps]
#
# Example:
#   ./launch_manc_bootstrap_hkem.sh 30
#
# This will run HKEM reconstruction for bootstraps 0-29

set -euo pipefail

# Check arguments
NUM_BOOTSTRAPS=${1:-30}

# Validate inputs
if ! [[ "$NUM_BOOTSTRAPS" =~ ^[0-9]+$ ]]; then
    echo "Error: num_bootstraps must be a positive integer"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
REPO_ROOT="$(cd "$BASE_DIR/.." && pwd)"

# Set paths
SPECT_QSUB_SCRIPT="$SCRIPT_DIR/run_manc_bootstrap_hkem_spect.qsub.sh"
RESAMPLE_QSUB_SCRIPT="$SCRIPT_DIR/run_manc_bootstrap_hkem_resample.qsub.sh"
PET_QSUB_SCRIPT="$SCRIPT_DIR/run_manc_bootstrap_hkem_pet.qsub.sh"
RESULTS_BASE="${RESULTS_BASE:-$REPO_ROOT/results/manc_bootstraps/hkem}"
LOG_DIR="$HOME/setr_logs"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_BASE"

# Verify qsub scripts exist
for script in "$SPECT_QSUB_SCRIPT" "$RESAMPLE_QSUB_SCRIPT" "$PET_QSUB_SCRIPT"; do
    if [ ! -f "$script" ]; then
        echo "Error: Submission script not found: $script"
        exit 1
    fi
done

echo "======================================"
echo "MANC Bootstrap HKEM Reconstruction"
echo "======================================"
echo "Number of bootstraps: $NUM_BOOTSTRAPS"
echo "Results directory:    $RESULTS_BASE"
echo "Log directory:        $LOG_DIR"
echo "======================================"
echo ""

# ================================
# Step 1: Submit SPECT HKEM job
# ================================
echo "Step 1: Submitting SPECT HKEM reconstruction job..."

SPECT_JOB_NAME="manc_hkem_spect"

SPECT_JOB_OUTPUT=$(qsub \
    -N "$SPECT_JOB_NAME" \
    -l h_rt=04:00:00 \
    -l tmem=16G \
    -l gpu=true \
    -o "$LOG_DIR/${SPECT_JOB_NAME}_\$JOB_ID.log" \
    -v SETR_BASE_DIR="$BASE_DIR",RESULTS_BASE="$RESULTS_BASE" \
    "$SPECT_QSUB_SCRIPT")

SPECT_JOB_SUBMIT_STATUS=$?

if [ $SPECT_JOB_SUBMIT_STATUS -ne 0 ]; then
    echo "Error: SPECT job submission failed with status $SPECT_JOB_SUBMIT_STATUS"
    exit 1
fi

# Extract job ID from qsub output (format: "Your job 123456 ...")
SPECT_JOB_ID=$(echo "$SPECT_JOB_OUTPUT" | grep -oP 'job \K[0-9]+')

echo "SPECT job submitted successfully! Job ID: $SPECT_JOB_ID"
echo ""

# ================================
# Step 2: Submit resampling array job (depends on SPECT)
# ================================
echo "Step 2: Submitting resampling array job..."

RESAMPLE_JOB_NAME="manc_hkem_resample"

RESAMPLE_JOB_OUTPUT=$(qsub \
    -N "$RESAMPLE_JOB_NAME" \
    -t 1-${NUM_BOOTSTRAPS} \
    -hold_jid "$SPECT_JOB_ID" \
    -l h_rt=00:30:00 \
    -l h_vmem=8G \
    -o "$LOG_DIR/${RESAMPLE_JOB_NAME}_\$JOB_ID_\$TASK_ID.log" \
    -v SETR_BASE_DIR="$BASE_DIR",RESULTS_BASE="$RESULTS_BASE" \
    "$RESAMPLE_QSUB_SCRIPT")

RESAMPLE_JOB_SUBMIT_STATUS=$?

if [ $RESAMPLE_JOB_SUBMIT_STATUS -ne 0 ]; then
    echo "Error: Resampling job submission failed with status $RESAMPLE_JOB_SUBMIT_STATUS"
    exit 1
fi

RESAMPLE_JOB_ID=$(echo "$RESAMPLE_JOB_OUTPUT" | grep -oP 'job-array \K[0-9]+')

echo "Resampling array job submitted successfully! Job ID: $RESAMPLE_JOB_ID"
echo "  - Will start after SPECT job $SPECT_JOB_ID completes"
echo "  - Array tasks: 1-$NUM_BOOTSTRAPS"
echo ""

# ================================
# Step 3: Submit PET HKEM array job (depends on resampling)
# ================================
echo "Step 3: Submitting PET HKEM array job..."

PET_JOB_NAME="manc_hkem_pet"

PET_JOB_OUTPUT=$(qsub \
    -N "$PET_JOB_NAME" \
    -t 1-${NUM_BOOTSTRAPS} \
    -hold_jid "$RESAMPLE_JOB_ID" \
    -l h_rt=04:00:00 \
    -l tmem=16G \
    -l gpu=true \
    -o "$LOG_DIR/${PET_JOB_NAME}_\$JOB_ID_\$TASK_ID.log" \
    -v SETR_BASE_DIR="$BASE_DIR",RESULTS_BASE="$RESULTS_BASE" \
    "$PET_QSUB_SCRIPT")

PET_JOB_SUBMIT_STATUS=$?

if [ $PET_JOB_SUBMIT_STATUS -ne 0 ]; then
    echo "Error: PET job submission failed with status $PET_JOB_SUBMIT_STATUS"
    exit 1
fi

PET_JOB_ID=$(echo "$PET_JOB_OUTPUT" | grep -oP 'job-array \K[0-9]+')

echo "PET HKEM array job submitted successfully! Job ID: $PET_JOB_ID"
echo "  - Will start after resampling job $RESAMPLE_JOB_ID completes"
echo "  - Array tasks: 1-$NUM_BOOTSTRAPS"
echo ""

# ================================
# Summary
# ================================
echo "======================================"
echo "All jobs submitted successfully!"
echo "======================================"
echo ""
echo "Job Pipeline:"
echo "  1. SPECT HKEM:    Job $SPECT_JOB_ID"
echo "  2. Resampling:    Job $RESAMPLE_JOB_ID (array, waits for $SPECT_JOB_ID)"
echo "  3. PET HKEM:      Job $PET_JOB_ID (array, waits for $RESAMPLE_JOB_ID)"
echo ""
echo "Monitor job status with:"
echo "  qstat -u \$USER"
echo ""
echo "View logs in:"
echo "  $LOG_DIR/${SPECT_JOB_NAME}_${SPECT_JOB_ID}.log"
echo "  $LOG_DIR/${RESAMPLE_JOB_NAME}_${RESAMPLE_JOB_ID}_*.log"
echo "  $LOG_DIR/${PET_JOB_NAME}_${PET_JOB_ID}_*.log"
echo ""
echo "Results will be saved to:"
echo "  SPECT HKEM:   $RESULTS_BASE/spect_hkem/"
echo "  PET HKEM:     $RESULTS_BASE/bootstrap_XXX/"
echo ""
