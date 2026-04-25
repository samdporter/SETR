#!/bin/bash
set -e

# ================================
# Multi-Patient HKEM Comparison Script with TOF Support
# ================================
# This script runs HKEM reconstruction for patients sirt2, sirt3, and sirt4
# Workflow for each patient:
# 1. SPECT HKEM reconstruction (1 bed position)
# 2. Resample SPECT to PET space for non-TOF guidance
# 3. PET HKEM reconstruction (2 bed positions with emission guidance) - non-TOF
# 4. Resample SPECT to PET space for TOF guidance
# 5. PET HKEM reconstruction (2 bed positions with emission guidance) - TOF

# ================================
# CONFIGURATION
# ================================

# Base directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
REPO_ROOT="$(cd "$BASE_DIR/.." && pwd)"
CONFIG_DIR="$BASE_DIR/configs"
DATA_BASE="/home/storage/prepared_data/oxford_patient_data"
RESULTS_BASE="${RESULTS_BASE:-$REPO_ROOT/results}"

# Config files
SPECT_CONFIG="${CONFIG_DIR}/config_hkem_2bpos_spect.yaml"
PET_CONFIG="${CONFIG_DIR}/config_hkem_2bpos.yaml"
RESAMPLE_CONFIG="${CONFIG_DIR}/config_resample_2bpos.yaml"

# Backup configs (to restore original state)
SPECT_BACKUP="${CONFIG_DIR}/config_hkem_2bpos_spect.yaml.backup"
PET_BACKUP="${CONFIG_DIR}/config_hkem_2bpos.yaml.backup"
RESAMPLE_BACKUP="${CONFIG_DIR}/config_resample_2bpos.yaml.backup"

# Scripts
HKEM_SCRIPT_1BPOS="${SCRIPT_DIR}/run_hkem_1bpos.py"
HKEM_SCRIPT_2BPOS="${SCRIPT_DIR}/run_hkem_2bpos.py"
RESAMPLE_SCRIPT="${SCRIPT_DIR}/resample_spect_to_pet.py"

# Patient list
PATIENTS=(
    "sirt1" "sirt2" "sirt4" "sirt5"
    "sirt6" "sirt7" "sirt8" "sirt9" "sirt10"
)


# ================================
# CONTROL FLAGS - Edit these to run specific parts
# ================================
DO_SPECT=false      # Run SPECT HKEM reconstruction (only once)
DO_RESAMPLE=false     # Resample SPECT to PET space for non-TOF
DO_PET_HKEM=true     # Run PET HKEM reconstruction (non-TOF)
DO_TOF_RESAMPLE=false # Resample SPECT to PET space for TOF
DO_TOF_PET_HKEM=false  # Run PET HKEM reconstruction (TOF)

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
    mv "$SPECT_BACKUP" "$SPECT_CONFIG"
    mv "$PET_BACKUP" "$PET_CONFIG"
    mv "$RESAMPLE_BACKUP" "$RESAMPLE_CONFIG"
}

update_spect_config_for_patient() {
    local patient=$1
    local data_dir="${DATA_BASE}/${patient}"
    local spect_data_path="${data_dir}/SPECT"
    local results_dir="${RESULTS_BASE}/${patient}"
    local spect_output="${results_dir}/hkem_spect_2bpos"

    echo "Updating SPECT config for $patient..."

    # Ensure output directory exists
    mkdir -p "$spect_output"

    # Update SPECT config
    sed -i "s|^data_path: \"[^\"]*\"|data_path: \"$spect_data_path\"|g" "$SPECT_CONFIG"
    sed -i "s|^spect_data_path: \"[^\"]*\"|spect_data_path: \"$spect_data_path\"|g" "$SPECT_CONFIG"
    sed -i "s|^output_path: \"[^\"]*\"|output_path: \"$spect_output\"|g" "$SPECT_CONFIG"
}

update_pet_config_for_patient() {
    local patient=$1
    local use_tof=$2  # "true" or "false"
    local data_dir="${DATA_BASE}/${patient}"
    local pet_data_path="${data_dir}/PET"
    local results_dir="${RESULTS_BASE}/${patient}"
    local out_subdir
    local pet_output

    if [ "$use_tof" = "true" ]; then
        out_subdir="tof"
    else
        out_subdir="non_tof"
    fi

    # Ensure separate output directory exists
    mkdir -p "${results_dir}/${out_subdir}"

    # Single base name, different parent dirs
    pet_output="${results_dir}/${out_subdir}/hkem_pet_2bpos"

    echo "Updating PET config for $patient (TOF: $use_tof) -> ${pet_output}"

    # Update PET config using sed for paths
    sed -i "s|^pet_data_path: \"[^\"]*\"|pet_data_path: \"$pet_data_path\"|g" "$PET_CONFIG"
    sed -i "s|^output_path: \"[^\"]*\"|output_path: \"$pet_output\"|g" "$PET_CONFIG"
    
    # Use awk for reliable TOF update (handles any whitespace/formatting issues)
    awk -v tof="$use_tof" '
        /^use_tof:/ { print "use_tof: " tof; next }
        { print }
    ' "$PET_CONFIG" > "${PET_CONFIG}.tmp" && mv "${PET_CONFIG}.tmp" "$PET_CONFIG"
    
    # Verify the update worked
    echo "TOF parameter after update:"
    grep "use_tof:" "$PET_CONFIG"
}
update_resample_config_for_patient() {
    local patient=$1
    local use_tof=$2  # "true" or "false"
    local data_dir="${DATA_BASE}/${patient}"
    local spect_data_path="${data_dir}/SPECT"
    local pet_data_path="${data_dir}/PET"
    local results_dir="${RESULTS_BASE}/${patient}"
    local spect_output="${results_dir}/hkem_spect_2bpos"
    local tof_folder

    if [ "$use_tof" = "true" ]; then
        tof_folder="tof"
    else
        tof_folder="non_tof"
    fi

    echo "Updating resample config for $patient (TOF: $use_tof)..."

    # Ensure PET guidance subfolder exists
    mkdir -p "${pet_data_path}/${tof_folder}"

    # Update resample config
    sed -i "s|^data_dir: \"[^\"]*\"|data_dir: \"$data_dir\"|g" "$RESAMPLE_CONFIG"
    sed -i "s|^results_dir: \"[^\"]*\"|results_dir: \"$RESULTS_BASE\"|g" "$RESAMPLE_CONFIG"
    sed -i "s|^spect_reconstruction: \"[^\"]*\"|spect_reconstruction: \"$spect_output/reconstruction_x.hv\"|g" "$RESAMPLE_CONFIG"
    sed -i "s|^transform_file: \"[^\"]*\"|transform_file: \"$spect_data_path/spect2pet_zoom_nonrigid.nii\"|g" "$RESAMPLE_CONFIG"
    sed -i "s|^output_file: \"[^\"]*\"|output_file: \"$pet_data_path/$tof_folder/spect.hv\"|g" "$RESAMPLE_CONFIG"
    sed -i "s|^use_tof: \(true\|false\)$|use_tof: $use_tof|g" "$RESAMPLE_CONFIG"
}

run_patient_hkem() {
    local patient=$1

    echo "======================================"
    echo "Processing patient: $patient"
    echo "======================================"

    # Define paths for this patient
    local data_dir="${DATA_BASE}/${patient}"
    local results_dir="${RESULTS_BASE}/${patient}"

    # Create results directory
    mkdir -p "$results_dir"

    echo "Data directory: $data_dir"
    echo "Results directory: $results_dir"
    echo "--------------------------------------"

    # 1. SPECT HKEM reconstruction (done once)
    if [ "$DO_SPECT" = true ]; then
        echo "=== Running SPECT HKEM for $patient ==="
        update_spect_config_for_patient "$patient"
        python3 "$HKEM_SCRIPT_1BPOS" --config "$SPECT_CONFIG"
        echo "SPECT HKEM reconstruction completed for $patient"
    fi

    # 2. Resample SPECT to non-TOF PET space
    if [ "$DO_RESAMPLE" = true ]; then
        echo "=== Resampling SPECT to PET space (non-TOF) for $patient ==="
        update_resample_config_for_patient "$patient" "false"
        python3 "$RESAMPLE_SCRIPT" --config "$RESAMPLE_CONFIG"
        echo "SPECT resampling (non-TOF) completed for $patient"
    fi

    # 3. PET HKEM reconstruction (non-TOF)
    if [ "$DO_PET_HKEM" = true ]; then
        echo "=== Running PET HKEM (non-TOF) for $patient ==="
        update_pet_config_for_patient "$patient" "false"
        python3 "$HKEM_SCRIPT_2BPOS" --config "$PET_CONFIG"
        echo "PET HKEM reconstruction (non-TOF) completed for $patient"
    fi

    # 4. Resample SPECT to TOF PET space
    if [ "$DO_TOF_RESAMPLE" = true ]; then
        echo "=== Resampling SPECT to PET space (TOF) for $patient ==="
        update_resample_config_for_patient "$patient" "true"
        python3 "$RESAMPLE_SCRIPT" --config "$RESAMPLE_CONFIG"
        echo "SPECT resampling (TOF) completed for $patient"
    fi

    # 5. PET HKEM reconstruction (TOF)
    if [ "$DO_TOF_PET_HKEM" = true ]; then
        echo "=== Running PET HKEM (TOF) for $patient ==="
        update_pet_config_for_patient "$patient" "true"
        python3 "$HKEM_SCRIPT_2BPOS" --config "$PET_CONFIG"
        echo "PET HKEM reconstruction (TOF) completed for $patient"
    fi

    echo "Patient $patient processing completed!"
    echo "Results saved in: $results_dir"
    echo ""
}

# ================================
# MAIN EXECUTION
# ================================

# Trap to ensure configs are restored on exit
trap restore_configs EXIT

echo "======================================"
echo "Multi-Patient HKEM Comparison Script with TOF Support"
echo "======================================"
echo "Patients to process: ${PATIENTS[*]}"
echo "Config files:"
echo "  SPECT:    $SPECT_CONFIG"
echo "  PET:      $PET_CONFIG"
echo "  Resample: $RESAMPLE_CONFIG"
echo "Processing steps:"
echo "  SPECT HKEM:       $DO_SPECT"
echo "  Resample non-TOF: $DO_RESAMPLE"
echo "  PET HKEM non-TOF: $DO_PET_HKEM"
echo "  Resample TOF:     $DO_TOF_RESAMPLE"
echo "  PET HKEM TOF:     $DO_TOF_PET_HKEM"
echo "======================================"

# Create backups of original configs
backup_configs

# Process each patient
for patient in "${PATIENTS[@]}"; do
    run_patient_hkem "$patient"
done

echo "======================================"
echo "All patients completed!"
echo "======================================"
echo "Results saved in:"
for patient in "${PATIENTS[@]}"; do
    echo "  $patient:"
    echo "    SPECT:        ${RESULTS_BASE}/${patient}/hkem_spect_2bpos/"
    echo "    PET non-TOF:  ${RESULTS_BASE}/${patient}/non_tof/hkem_pet_2bpos/"
    echo "    PET TOF:      ${RESULTS_BASE}/${patient}/tof/hkem_pet_2bpos/"
done
echo "======================================"

# Configs will be restored automatically via trap
