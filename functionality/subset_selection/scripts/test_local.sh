#!/bin/bash

# Quick local test script for subset selection experiments
# Usage: ./test_local.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
FUNC_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$FUNC_DIR/configs"
SCRIPTS_DIR="$FUNC_DIR/scripts"

echo "=== Local Test for Subset Selection ==="
echo "Base directory: $BASE_DIR"

# Test configuration
OUTPUT_DIR="$FUNC_DIR/output/local_test"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run a quick test with minimal epochs
echo "Running test reconstruction..."
echo "Config: separate subsets, always prior, bsrem precond, gamma=10, 2 epochs"
echo ""

cd "$BASE_DIR"

python "$SCRIPTS_DIR/run_subset_selection.py" \
    --config "$CONFIG_DIR/base_config_anthro.yaml" \
    --override \
        output_path="$OUTPUT_DIR" \
        num_epochs=2 \
        subset_mode=separate \
        prior_mode=always \
        precond_type=bsrem \
        gamma_tnv=10

echo ""
echo "Test complete!"
echo "Check output in: $OUTPUT_DIR"
echo ""
echo "To test other configurations, modify the overrides above or run:"
echo "  python $SCRIPTS_DIR/run_subset_selection.py --config ... --override ..."
