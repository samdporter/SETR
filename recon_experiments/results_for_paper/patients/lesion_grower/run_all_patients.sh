#!/bin/bash
# Run lesion growth for all patients (sirt1-sirt10) with both methods and both source images
# Then generate comparison plots

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Array of all patients
PATIENTS=(sirt1 sirt2 sirt3 sirt4 sirt5 sirt6 sirt7 sirt8 sirt9 sirt10)

# Source images to test
SOURCE_IMAGES=(vendor)

# Methods to test
METHODS=(radial_gradient)

echo "========================================="
echo "STEP 1: Growing lesion masks"
echo "========================================="
echo ""

# Loop over all combinations
for patient in "${PATIENTS[@]}"; do
    echo "Processing patient: $patient"

    for source in "${SOURCE_IMAGES[@]}"; do
        echo "  Source image: $source"

        for method in "${METHODS[@]}"; do
            echo "    Method: $method"

            # Run lesion growth
            python grow_lesions.py \
                --patient "$patient" \
                --method "$method" \
                --source-image "$source" \
                --num-lesions 5 \
                --min-size 10 \
                --force \
                2>&1 | tee "logs/${patient}_${method}_${source}_growth.log"

            if [ $? -eq 0 ]; then
                echo "      ✓ Success"
            else
                echo "      ✗ Failed (check logs/${patient}_${method}_${source}_growth.log)"
            fi
        done
    done
    echo ""
done

echo ""
echo "========================================="
echo "STEP 2: Generating comparison plots"
echo "========================================="
echo ""

# Generate comparison plots for each patient and source image
for patient in "${PATIENTS[@]}"; do
    echo "Generating plots for patient: $patient"

    for source in "${SOURCE_IMAGES[@]}"; do
        echo "  Source image: $source"

        python plot_method_comparison.py \
            --patients "$patient" \
            --num-lesions 5 \
            --source-image "$source" \
            2>&1 | tee "logs/${patient}_${source}_plots.log"

        if [ $? -eq 0 ]; then
            echo "    ✓ Success"
        else
            echo "    ✗ Failed (check logs/${patient}_${source}_plots.log)"
        fi
    done
    echo ""
done

echo ""
echo "========================================="
echo "STEP 3: Running method comparisons"
echo "========================================="
echo ""

# Run method comparison metrics for each source image
for source in "${SOURCE_IMAGES[@]}"; do
    echo "Computing comparison metrics for source: $source"

    python compare_methods.py \
        --patients "${PATIENTS[@]}" \
        --num-lesions 5 \
        --source-image "$source" \
        --output "method_comparison_results_${source}.csv" \
        2>&1 | tee "logs/comparison_${source}.log"

    if [ $? -eq 0 ]; then
        echo "  ✓ Success - saved to method_comparison_results_${source}.csv"
    else
        echo "  ✗ Failed (check logs/comparison_${source}.log)"
    fi
    echo ""
done

echo ""
echo "========================================="
echo "ALL PROCESSING COMPLETE"
echo "========================================="
echo ""
echo "Results locations:"
echo "  - Lesion masks: /home/storage/cluster/patient_sweeps/lesion_masks/{patient}/{method}_{source}/"
echo "  - Comparison plots: /home/storage/cluster/patient_sweeps/lesion_grower/comparison_plots/"
echo "  - Comparison metrics: /home/storage/cluster/patient_sweeps/lesion_grower/method_comparison_results_{vendor,combined_recon}.csv"
echo "  - Logs: /home/storage/cluster/patient_sweeps/lesion_grower/logs/"
echo ""
