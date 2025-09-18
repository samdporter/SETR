#!/bin/bash
set -e

# ================================
# HKEM/KEM Comparison Script (2 Bed Position PET)
# ================================
# This script runs a complete comparison workflow:
# 1. SPECT HKEM reconstruction (1 bed position)
# 2. Resample SPECT to PET space for guidance
# 3. PET HKEM reconstruction (2 bed positions with emission guidance)
# 4. PET KEM reconstruction (2 bed positions, anatomical guidance only)
# 5. Optional KOSMAPOSL method testing

# ================================
# CONFIGURATION
# ================================

# Directories (modify these for your setup)
SCRIPT_DIR="/home/sam/working/synergistic_recon/scripts"
CONFIG_DIR="/home/sam/working/synergistic_recon/configs"
RESULTS_DIR="/home/sam/working/synergistic_recon/results"

# Config files
SPECT_CONFIG="${CONFIG_DIR}/config_hkem_2bpos_spect.yaml"  # Always 1 bed position for SPECT
PET_CONFIG="${CONFIG_DIR}/config_hkem_2bpos.yaml"          # 2 bed positions for PET
RESAMPLE_CONFIG="${CONFIG_DIR}/config_resample_2bpos.yaml"

# Scripts
HKEM_SCRIPT_1BPOS="${SCRIPT_DIR}/run_hkem_1bpos.py"
HKEM_SCRIPT_2BPOS="${SCRIPT_DIR}/run_hkem_2bpos.py"
RESAMPLE_SCRIPT="${SCRIPT_DIR}/resample_spect_to_pet.py"

# Output directories (automatically determined from configs)
SPECT_RESULTS="${RESULTS_DIR}/hkem_spect_1bpos"
PET_HKEM_RESULTS="${RESULTS_DIR}/hkem_pet_2bpos"
PET_KEM_RESULTS="${RESULTS_DIR}/kem_pet_2bpos"
PET_KOSMAPOSL_RESULTS="${RESULTS_DIR}/hkem_pet_kosmaposl_2bpos"

# ================================
# CONTROL FLAGS - Edit these to run specific parts
# ================================
DO_SPECT=true         # Run SPECT HKEM reconstruction (1 bed position)
DO_RESAMPLE=true       # Resample SPECT to PET space
DO_PET_HKEM=true      # Run PET HKEM (2 bed positions, hybrid with emission guidance)
DO_PET_KEM=false        # Run PET KEM (2 bed positions, anatomical guidance only)
DO_KOSMAPOSL=false     # Run KOSMAPOSL method (experimental)

# ================================
# EXECUTION
# ================================

echo "======================================"
echo "HKEM/KEM Comparison Script (2 Bed Pos)"
echo "======================================"
echo "Config files:"
echo "  SPECT:    $SPECT_CONFIG (1 bed position)"
echo "  PET:      $PET_CONFIG (2 bed positions)"
echo "  Resample: $RESAMPLE_CONFIG"
echo "======================================"

if [ "$DO_SPECT" = true ]; then
    echo "=== Running HKEM with SPECT (1 bed position) ==="
    python3 "$HKEM_SCRIPT_1BPOS" --config "$SPECT_CONFIG"
    echo "SPECT HKEM reconstruction completed"
fi

if [ "$DO_RESAMPLE" = true ]; then
    echo "=== Resampling SPECT reconstruction to PET space ==="
    python3 "$RESAMPLE_SCRIPT" --config "$RESAMPLE_CONFIG"
    echo "SPECT resampling completed"
fi

if [ "$DO_PET_HKEM" = true ]; then
    echo "=== Running HKEM with PET (2 bed positions) ==="
    python3 "$HKEM_SCRIPT_2BPOS" --config "$PET_CONFIG"
    echo "PET HKEM reconstruction completed"
fi

if [ "$DO_PET_KEM" = true ]; then
    echo "=== Running KEM with PET (2 bed positions) ==="
    # Create a KEM config by overriding the hybrid parameter
    python3 "$HKEM_SCRIPT_2BPOS" \
        --config "$PET_CONFIG" \
        --override hybrid=false \
        --override guidance=attenuation \
        --override output_path="$PET_KEM_RESULTS"
    echo "PET KEM reconstruction completed"
fi

if [ "$DO_KOSMAPOSL" = true ]; then
    echo "=== Running HKEM with PET (KOSMAPOSL method, 2 bed positions) ==="
    python3 "$HKEM_SCRIPT_2BPOS" \
        --config "$PET_CONFIG" \
        --override method=kosmaposl \
        --override step_size=1.0 \
        --override output_path="$PET_KOSMAPOSL_RESULTS"
    echo "PET HKEM KOSMAPOSL reconstruction completed"
fi

echo "======================================"
echo "All reconstructions completed!"
echo "======================================"
echo "Results saved in:"
echo "  SPECT HKEM: $SPECT_RESULTS"
echo "  PET HKEM:   $PET_HKEM_RESULTS"
echo "  PET KEM:    $PET_KEM_RESULTS"
if [ "$DO_KOSMAPOSL" = true ]; then
    echo "  KOSMAPOSL:  $PET_KOSMAPOSL_RESULTS"
fi
echo "======================================"