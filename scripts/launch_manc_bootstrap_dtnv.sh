#!/bin/bash
# ================================
# Launcher for MANC Bootstrap DTNV Reconstruction
# ================================
# This script launches an SGE array job to run DTNV reconstruction
# for all MANC NEMA phantom bootstrap datasets with specified alpha/beta values
#
# Usage:
#   ./launch_manc_bootstrap_dtnv.sh <alpha> <beta> [num_bootstraps]
#
# Example:
#   ./launch_manc_bootstrap_dtnv.sh 0.5 1.0 30
#
# This will run DTNV reconstruction for bootstraps 0-29 with alpha=0.5, beta=1.0

set -euo pipefail

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <alpha> <beta> [num_bootstraps]"
    echo ""
    echo "Arguments:"
    echo "  alpha          - Alpha parameter for data fidelity (PET weight)"
    echo "  beta           - Beta parameter for data fidelity (SPECT weight)"
    echo "  num_bootstraps - Number of bootstrap datasets to process (default: 30)"
    echo ""
    echo "Example:"
    echo "  $0 0.5 1.0 30"
    exit 1
fi

ALPHA=$1
BETA=$2
NUM_BOOTSTRAPS=${3:-30}

# Validate inputs
if ! [[ "$ALPHA" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "Error: Alpha must be a number"
    exit 1
fi

if ! [[ "$BETA" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "Error: Beta must be a number"
    exit 1
fi

if ! [[ "$NUM_BOOTSTRAPS" =~ ^[0-9]+$ ]]; then
    echo "Error: num_bootstraps must be a positive integer"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Set paths
QSUB_SCRIPT="$SCRIPT_DIR/run_manc_bootstrap_dtnv.qsub.sh"
RESULTS_BASE="/home/sam/working/synergistic_recon/results/manc_bootstraps/dtnv"
LOG_DIR="$HOME/setr_logs"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_BASE/alpha_${ALPHA}_beta_${BETA}"

# Verify qsub script exists
if [ ! -f "$QSUB_SCRIPT" ]; then
    echo "Error: Submission script not found: $QSUB_SCRIPT"
    exit 1
fi

# Job name
JOB_NAME="manc_dtnv_a${ALPHA}_b${BETA}"

echo "======================================"
echo "MANC Bootstrap DTNV Reconstruction"
echo "======================================"
echo "Alpha:               $ALPHA"
echo "Beta:                $BETA"
echo "Number of bootstraps: $NUM_BOOTSTRAPS"
echo "Results directory:    $RESULTS_BASE/alpha_${ALPHA}_beta_${BETA}"
echo "Log directory:        $LOG_DIR"
echo "======================================"
echo ""
echo "Submitting array job with $NUM_BOOTSTRAPS tasks..."

# Submit the job array
qsub \
    -N "$JOB_NAME" \
    -t 1-${NUM_BOOTSTRAPS} \
    -l h_rt=04:00:00 \
    -l tmem=16G \
    -l gpu=true \
    -o "$LOG_DIR/${JOB_NAME}_\$JOB_ID_\$TASK_ID.log" \
    -v ALPHA="$ALPHA",BETA="$BETA",SETR_BASE_DIR="$BASE_DIR",RESULTS_BASE="$RESULTS_BASE" \
    "$QSUB_SCRIPT"

JOB_SUBMIT_STATUS=$?

if [ $JOB_SUBMIT_STATUS -eq 0 ]; then
    echo ""
    echo "Job submitted successfully!"
    echo ""
    echo "Monitor job status with:"
    echo "  qstat -u \$USER"
    echo ""
    echo "View logs in:"
    echo "  $LOG_DIR/${JOB_NAME}_*.log"
    echo ""
    echo "Results will be saved to:"
    echo "  $RESULTS_BASE/alpha_${ALPHA}_beta_${BETA}/bootstrap_XXX/"
else
    echo ""
    echo "Error: Job submission failed with status $JOB_SUBMIT_STATUS"
    exit 1
fi
