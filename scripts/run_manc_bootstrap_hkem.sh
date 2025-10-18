#!/bin/bash
set -e

# ================================
# MANC Bootstrap HKEM Reconstruction Pipeline
# ================================
# This script runs the complete HKEM reconstruction pipeline for MANC phantom bootstrap data:
# 1. SPECT HKEM reconstruction (once, shared across all bootstraps)
# 2. For each bootstrap:
#    a. Resample SPECT to PET space
#    b. PET HKEM reconstruction with SPECT emission guidance
#
# Usage:
#   ./run_manc_bootstrap_hkem.sh [num_bootstraps]
#
# Example:
#   ./run_manc_bootstrap_hkem.sh 30

# ================================
# CONFIGURATION
# ================================

# Base directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$BASE_DIR/configs"
DATA_BASE="/home/storage/prepared_data/phantom_data/manc_nema_phantom_data"
RESULTS_BASE="/home/sam/working/synergistic_recon/results/manc_bootstraps/hkem"

# Config files
SPECT_CONFIG="${CONFIG_DIR}/config_manc_hkem_spect.yaml"
PET_CONFIG="${CONFIG_DIR}/config_manc_hkem_pet.yaml"
RESAMPLE_CONFIG="${CONFIG_DIR}/config_manc_resample.yaml"

# Backup configs (to restore original state)
SPECT_BACKUP="${CONFIG_DIR}/config_manc_hkem_spect.yaml.backup"
PET_BACKUP="${CONFIG_DIR}/config_manc_hkem_pet.yaml.backup"
RESAMPLE_BACKUP="${CONFIG_DIR}/config_manc_resample.yaml.backup"

# Scripts
HKEM_SCRIPT_1BPOS="${SCRIPT_DIR}/run_hkem_1bpos.py"
RESAMPLE_SCRIPT="${SCRIPT_DIR}/resample_spect_to_pet_simple.py"

# Bootstrap configuration
NUM_BOOTSTRAPS=${1:-30}
BOOTSTRAP_BASE_DIR="${DATA_BASE}/PET/bootstraps/nonparametric"

# ================================
# CONTROL FLAGS
# ================================
DO_SPECT=true       # Run SPECT HKEM reconstruction (only once)
DO_RESAMPLE=true    # Resample SPECT to PET space for each bootstrap
DO_PET_HKEM=true    # Run PET HKEM reconstruction for each bootstrap

# ================================
# FUNCTIONS
# ================================

backup_configs() {
    echo "Creating config backups..."
    cp "$SPECT_CONFIG" "$SPECT_BACKUP"
    cp "$PET_CONFIG" "$PET_BACKUP"
    cp "$RESAMPLE_CONFIG" "$RESAMPLE_BACKUP"
}

restore_configs() {
    echo "Restoring original configs..."
    if [ -f "$SPECT_BACKUP" ]; then
        mv "$SPECT_BACKUP" "$SPECT_CONFIG"
    fi
    if [ -f "$PET_BACKUP" ]; then
        mv "$PET_BACKUP" "$PET_CONFIG"
    fi
    if [ -f "$RESAMPLE_BACKUP" ]; then
        mv "$RESAMPLE_BACKUP" "$RESAMPLE_CONFIG"
    fi
}

update_spect_config() {
    local spect_data_path="${DATA_BASE}/SPECT"
    local spect_output="${RESULTS_BASE}/spect_hkem"

    echo "Updating SPECT config..."

    # Ensure output directory exists
    mkdir -p "$spect_output"

    # Update SPECT config
    sed -i "s|^data_path: \"[^\"]*\"|data_path: \"$spect_data_path\"|g" "$SPECT_CONFIG"
    sed -i "s|^spect_data_path: \"[^\"]*\"|spect_data_path: \"$spect_data_path\"|g" "$SPECT_CONFIG"
    sed -i "s|^output_path: \"[^\"]*\"|output_path: \"$spect_output\"|g" "$SPECT_CONFIG"
}

update_pet_config_for_bootstrap() {
    local bootstrap_num=$1
    local bootstrap_id=$(printf "prompts_nonparam_s73_sf0.05_%03d" $bootstrap_num)
    local pet_data_path="${BOOTSTRAP_BASE_DIR}/${bootstrap_id}"
    local pet_output="${RESULTS_BASE}/bootstrap_$(printf '%03d' $bootstrap_num)"

    echo "Updating PET config for bootstrap $bootstrap_num..."

    # Ensure output directory exists
    mkdir -p "$pet_output"

    # Update PET config
    sed -i "s|^pet_data_path: \"[^\"]*\"|pet_data_path: \"$pet_data_path\"|g" "$PET_CONFIG"
    sed -i "s|^output_path: \"[^\"]*\"|output_path: \"$pet_output\"|g" "$PET_CONFIG"
}

update_resample_config_for_bootstrap() {
    local bootstrap_num=$1
    local bootstrap_id=$(printf "prompts_nonparam_s73_sf0.05_%03d" $bootstrap_num)
    local spect_data_path="${DATA_BASE}/SPECT"
    local pet_data_path="${BOOTSTRAP_BASE_DIR}/${bootstrap_id}"
    local spect_output="${RESULTS_BASE}/spect_hkem"
    local transform_file="${spect_data_path}/spect2pet.nii"
    local output_file="${pet_data_path}/spect.hv"

    echo "Updating resample config for bootstrap $bootstrap_num..."

    # Update resample config
    sed -i "s|^data_dir: \"[^\"]*\"|data_dir: \"$DATA_BASE\"|g" "$RESAMPLE_CONFIG"
    sed -i "s|^results_dir: \"[^\"]*\"|results_dir: \"$RESULTS_BASE\"|g" "$RESAMPLE_CONFIG"
    sed -i "s|^pet_dir: \"[^\"]*\"|pet_dir: \"$pet_data_path\"|g" "$RESAMPLE_CONFIG"
    sed -i "s|^spect_dir: \"[^\"]*\"|spect_dir: \"$spect_data_path\"|g" "$RESAMPLE_CONFIG"
    sed -i "s|^spect_reconstruction: \"[^\"]*\"|spect_reconstruction: \"$spect_output/reconstruction_x.hv\"|g" "$RESAMPLE_CONFIG"
    sed -i "s|^transform_file: \"[^\"]*\"|transform_file: \"$transform_file\"|g" "$RESAMPLE_CONFIG"
    sed -i "s|^output_file: \"[^\"]*\"|output_file: \"$output_file\"|g" "$RESAMPLE_CONFIG"
}

run_bootstrap_hkem() {
    local bootstrap_num=$1
    local bootstrap_id=$(printf "prompts_nonparam_s73_sf0.05_%03d" $bootstrap_num)

    echo "======================================"
    echo "Processing bootstrap: $bootstrap_id"
    echo "======================================"

    local bootstrap_path="${BOOTSTRAP_BASE_DIR}/${bootstrap_id}"
    local results_dir="${RESULTS_BASE}/bootstrap_$(printf '%03d' $bootstrap_num)"

    # Check if bootstrap directory exists
    if [ ! -d "$bootstrap_path" ]; then
        echo "WARNING: Bootstrap directory not found: $bootstrap_path"
        echo "Skipping bootstrap $bootstrap_num"
        return
    fi

    # Create results directory
    mkdir -p "$results_dir"

    echo "Bootstrap path: $bootstrap_path"
    echo "Results directory: $results_dir"
    echo "--------------------------------------"

    # 1. Resample SPECT to PET space for this bootstrap
    if [ "$DO_RESAMPLE" = true ]; then
        echo "=== Resampling SPECT to PET space for bootstrap $bootstrap_num ==="
        update_resample_config_for_bootstrap "$bootstrap_num"
        python3 "$RESAMPLE_SCRIPT" --config "$RESAMPLE_CONFIG"
        echo "SPECT resampling completed for bootstrap $bootstrap_num"
    fi

    # 2. PET HKEM reconstruction with SPECT guidance
    if [ "$DO_PET_HKEM" = true ]; then
        echo "=== Running PET HKEM for bootstrap $bootstrap_num ==="
        update_pet_config_for_bootstrap "$bootstrap_num"
        python3 "$HKEM_SCRIPT_1BPOS" --config "$PET_CONFIG"
        echo "PET HKEM reconstruction completed for bootstrap $bootstrap_num"
    fi

    echo "Bootstrap $bootstrap_num processing completed!"
    echo "Results saved in: $results_dir"
    echo ""
}

# ================================
# MAIN EXECUTION
# ================================

# Trap to ensure configs are restored on exit
trap restore_configs EXIT

echo "======================================"
echo "MANC Bootstrap HKEM Reconstruction Pipeline"
echo "======================================"
echo "Number of bootstraps: $NUM_BOOTSTRAPS"
echo "Data directory:       $DATA_BASE"
echo "Results directory:    $RESULTS_BASE"
echo "Config files:"
echo "  SPECT:    $SPECT_CONFIG"
echo "  PET:      $PET_CONFIG"
echo "  Resample: $RESAMPLE_CONFIG"
echo "Processing steps:"
echo "  SPECT HKEM:   $DO_SPECT"
echo "  Resample:     $DO_RESAMPLE"
echo "  PET HKEM:     $DO_PET_HKEM"
echo "======================================"

# Create backups of original configs
backup_configs

# 1. SPECT HKEM reconstruction (done once for all bootstraps)
if [ "$DO_SPECT" = true ]; then
    echo ""
    echo "======================================"
    echo "Step 1: SPECT HKEM Reconstruction"
    echo "======================================"
    update_spect_config
    python3 "$HKEM_SCRIPT_1BPOS" --config "$SPECT_CONFIG"
    echo "SPECT HKEM reconstruction completed"
    echo "======================================"
fi

# 2. Process each bootstrap
echo ""
echo "======================================"
echo "Step 2: Processing Bootstraps"
echo "======================================"

for bootstrap_num in $(seq 0 $((NUM_BOOTSTRAPS - 1))); do
    run_bootstrap_hkem "$bootstrap_num"
done

echo "======================================"
echo "All bootstraps completed!"
echo "======================================"
echo "Results saved in:"
echo "  SPECT HKEM:        ${RESULTS_BASE}/spect_hkem/"
for bootstrap_num in $(seq 0 $((NUM_BOOTSTRAPS - 1))); do
    if [ $bootstrap_num -lt 3 ] || [ $bootstrap_num -ge $((NUM_BOOTSTRAPS - 1)) ]; then
        printf "  Bootstrap %03d:      ${RESULTS_BASE}/bootstrap_%03d/\n" $bootstrap_num $bootstrap_num
    elif [ $bootstrap_num -eq 3 ]; then
        echo "  ..."
    fi
done
echo "======================================"

# Configs will be restored automatically via trap
