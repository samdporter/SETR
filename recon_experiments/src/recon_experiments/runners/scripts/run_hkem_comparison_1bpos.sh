#!/bin/bash
set -e

# ================================
# HKEM/KEM Comparison Script (1 Bed Position)
# ================================
# This script runs a complete comparison workflow:
# 1. SPECT HKEM reconstruction
# 2. Resample SPECT to PET space for guidance
# 3. PET HKEM reconstruction (with emission guidance)
# 4. PET KEM reconstruction (anatomical guidance only)
# 5. Optional KOSMAPOSL method testing

# ================================
# CONFIGURATION
# ================================

# Directories (modify these for your setup)
SCRIPT_DIR="/home/sam/working/synergistic_recon/scripts"
CONFIG_DIR="/home/sam/working/synergistic_recon/configs"
RESULTS_DIR="/home/sam/working/synergistic_recon/results"

# Config files
SPECT_CONFIG="${CONFIG_DIR}/config_hkem_1bpos_spect.yaml"
PET_CONFIG="${CONFIG_DIR}/config_hkem_1bpos_pet.yaml"
KEM_CONFIG="${CONFIG_DIR}/config_kem_1bpos_pet.yaml"
RESAMPLE_CONFIG="${CONFIG_DIR}/config_resample_1bpos.yaml"

# Scripts
HKEM_SCRIPT="${SCRIPT_DIR}/run_hkem_1bpos.py"
RESAMPLE_SCRIPT="${SCRIPT_DIR}/resample_spect_to_pet.py"

# ================================
# CONTROL FLAGS - Edit these to run specific parts
# ================================
DO_SPECT=false        # Run SPECT HKEM reconstruction
DO_RESAMPLE=false       # Resample SPECT to PET space - will fail if SPECT reconstruction is not done
DO_PET_HKEM=false       # Run PET HKEM (hybrid with emission guidance)
DO_PET_KEM=true        # Run PET KEM (anatomical guidance only)

# ================================
# EXECUTION
# ================================

echo "======================================"
echo "HKEM/KEM Comparison Script (1 Bed Pos)"
echo "======================================"
echo "Config files:"
echo "  SPECT:    $SPECT_CONFIG"
echo "  PET:      $PET_CONFIG"
echo "  Resample: $RESAMPLE_CONFIG"
echo "======================================"

if [ "$DO_SPECT" = true ]; then
    echo "=== Running HKEM with SPECT ==="
    python3 "$HKEM_SCRIPT" --config "$SPECT_CONFIG"
    echo "SPECT HKEM reconstruction completed"
fi

if [ "$DO_RESAMPLE" = true ]; then
    echo "=== Resampling SPECT reconstruction to PET space ==="
    python3 "$RESAMPLE_SCRIPT" --config "$RESAMPLE_CONFIG"
    echo "SPECT resampling completed"
fi

if [ "$DO_PET_HKEM" = true ]; then
    echo "=== Running HKEM with PET ==="
    python3 "$HKEM_SCRIPT" --config "$PET_CONFIG"
    echo "PET HKEM reconstruction completed"
fi

if [ "$DO_PET_KEM" = true ]; then
    echo "=== Running KEM with PET ==="
    # Create a temporary KEM config by overriding the hybrid parameter
    python3 "$HKEM_SCRIPT" --config "$KEM_CONFIG"
    echo "PET KEM reconstruction completed"
fi

echo "======================================"
echo "All reconstructions completed!"
echo "======================================"