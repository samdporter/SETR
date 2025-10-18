#!/bin/bash
# ================================
# Local Testing Script for MANC Bootstrap DTNV
# ================================
# Quick test of DTNV reconstruction on a single bootstrap with reduced parameters
#
# Usage:
#   ./test_manc_bootstrap_dtnv_local.sh [alpha] [beta] [bootstrap_num]
#
# Example:
#   ./test_manc_bootstrap_dtnv_local.sh 0.5 1.0 0

set -e

# Parse arguments
ALPHA=${1:-1.0}
BETA=${2:-1.0}
BOOTSTRAP_NUM=${3:-0}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$BASE_DIR/configs"

# Paths
BOOTSTRAP_ID=$(printf "prompts_nonparam_s73_sf0.05_%03d" $BOOTSTRAP_NUM)
BOOTSTRAP_PATH="/home/storage/prepared_data/phantom_data/manc_nema_phantom_data/PET/bootstraps/nonparametric/${BOOTSTRAP_ID}"
SPECT_PATH="/home/storage/prepared_data/phantom_data/manc_nema_phantom_data/SPECT"
OUTPUT_DIR="$BASE_DIR/test_output/dtnv_local/alpha_${ALPHA}_beta_${BETA}/bootstrap_$(printf '%03d' $BOOTSTRAP_NUM)"
WORKING_DIR="$OUTPUT_DIR/tmp"

# Check if bootstrap exists
if [ ! -d "$BOOTSTRAP_PATH" ]; then
    echo "ERROR: Bootstrap directory not found: $BOOTSTRAP_PATH"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR" "$WORKING_DIR"

echo "======================================"
echo "Local DTNV Test"
echo "======================================"
echo "Bootstrap:    $BOOTSTRAP_ID"
echo "Alpha:        $ALPHA"
echo "Beta:         $BETA"
echo "Output:       $OUTPUT_DIR"
echo "======================================"
echo ""

# Run reconstruction with reduced parameters for quick testing
# Override num_epochs to 5 for quick test
python3 "$SCRIPT_DIR/run_dtnv_1bpos.py" \
    --config "$CONFIG_DIR/config_manc_bootstrap_dtnv.yaml" \
    --override \
    "alpha=$ALPHA" \
    "beta=$BETA" \
    "pet_data_path=$BOOTSTRAP_PATH" \
    "spect_data_path=$SPECT_PATH" \
    "output_path=$OUTPUT_DIR" \
    "working_path=$WORKING_DIR" \
    "num_epochs=5" \
    "save_images=true" \
    "save_gradients=false"

echo ""
echo "======================================"
echo "Test completed!"
echo "======================================"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "To run the full 50-epoch reconstruction:"
echo "  Remove the num_epochs=5 override"
echo ""
