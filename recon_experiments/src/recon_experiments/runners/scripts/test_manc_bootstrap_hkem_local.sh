#!/bin/bash
# ================================
# Local Testing Script for MANC Bootstrap HKEM
# ================================
# Quick test of HKEM pipeline on a single bootstrap with reduced parameters
#
# Usage:
#   ./test_manc_bootstrap_hkem_local.sh [bootstrap_num]
#
# Example:
#   ./test_manc_bootstrap_hkem_local.sh 0

set -e

# Parse arguments
BOOTSTRAP_NUM=${1:-0}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$BASE_DIR/configs"

# Paths
BOOTSTRAP_ID=$(printf "prompts_nonparam_s73_sf0.05_%03d" $BOOTSTRAP_NUM)
DATA_BASE="/home/storage/prepared_data/phantom_data/manc_nema_phantom_data"
BOOTSTRAP_PATH="${DATA_BASE}/PET/bootstraps/nonparametric/${BOOTSTRAP_ID}"
SPECT_PATH="${DATA_BASE}/SPECT"
OUTPUT_BASE="$BASE_DIR/test_output/hkem_local"
SPECT_OUTPUT="$OUTPUT_BASE/spect_hkem"
PET_OUTPUT="$OUTPUT_BASE/bootstrap_$(printf '%03d' $BOOTSTRAP_NUM)"
WORKING_DIR="$OUTPUT_BASE/tmp"

# Check if bootstrap exists
if [ ! -d "$BOOTSTRAP_PATH" ]; then
    echo "ERROR: Bootstrap directory not found: $BOOTSTRAP_PATH"
    exit 1
fi

# Create output directories
mkdir -p "$SPECT_OUTPUT" "$PET_OUTPUT" "$WORKING_DIR"

echo "======================================"
echo "Local HKEM Test"
echo "======================================"
echo "Bootstrap:     $BOOTSTRAP_ID"
echo "SPECT output:  $SPECT_OUTPUT"
echo "PET output:    $PET_OUTPUT"
echo "======================================"
echo ""

# Step 1: SPECT HKEM (only if not already done)
if [ ! -f "$SPECT_OUTPUT/reconstruction_x.hv" ]; then
    echo "Step 1: SPECT HKEM reconstruction"
    echo "======================================"
    python3 "$SCRIPT_DIR/run_hkem_1bpos.py" \
        --config "$CONFIG_DIR/config_manc_hkem_spect.yaml" \
        --override \
        "data_path=$SPECT_PATH" \
        "spect_data_path=$SPECT_PATH" \
        "output_path=$SPECT_OUTPUT" \
        "working_path=$WORKING_DIR" \
        "num_epochs=5"
    echo ""
else
    echo "Step 1: SPECT HKEM reconstruction already exists, skipping..."
    echo ""
fi

# Step 2: Resample SPECT to PET space
echo "Step 2: Resampling SPECT to PET space"
echo "======================================"
TRANSFORM_FILE="${SPECT_PATH}/spect2pet.nii"
OUTPUT_SPECT_FILE="${BOOTSTRAP_PATH}/spect.hv"

python3 "$SCRIPT_DIR/resample_spect_to_pet_simple.py" \
    --config "$CONFIG_DIR/config_manc_resample.yaml" \
    --override \
    "pet_dir=$BOOTSTRAP_PATH" \
    "spect_dir=$SPECT_PATH" \
    "spect_reconstruction=$SPECT_OUTPUT/reconstruction_x.hv" \
    "transform_file=$TRANSFORM_FILE" \
    "output_file=$OUTPUT_SPECT_FILE" \
    "use_2bpos=false" \
    "flip=true"

echo ""

# Step 3: PET HKEM reconstruction
echo "Step 3: PET HKEM reconstruction"
echo "======================================"
python3 "$SCRIPT_DIR/run_hkem_1bpos.py" \
    --config "$CONFIG_DIR/config_manc_hkem_pet.yaml" \
    --override \
    "pet_data_path=$BOOTSTRAP_PATH" \
    "output_path=$PET_OUTPUT" \
    "working_path=$WORKING_DIR" \
    "num_epochs=5"

echo ""
echo "======================================"
echo "Test completed!"
echo "======================================"
echo "Results saved in:"
echo "  SPECT: $SPECT_OUTPUT"
echo "  PET:   $PET_OUTPUT"
echo ""
echo "To run the full 15-epoch reconstruction:"
echo "  Remove the num_epochs=5 overrides"
echo ""
