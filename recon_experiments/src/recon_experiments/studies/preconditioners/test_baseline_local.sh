#!/bin/bash

# Quick local test of baseline reconstruction
# Usage: ./test_baseline_local.sh [alpha]

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

ALPHA="${1:-0.005}"
EPOCHS="${2:-5}"  # Short test

echo "=== Testing Baseline Reconstruction Locally ==="
echo "Alpha: $ALPHA"
echo "Epochs: $EPOCHS (short test)"
echo ""

OUTPUT_DIR="$SCRIPT_DIR/output/test_baseline_alpha_${ALPHA}"
mkdir -p "$OUTPUT_DIR"

cd "$BASE_DIR"

python scripts/run_dtnv_1bpos.py \
    --config configs/config_1bpos_anthro_long.yaml \
    --override "num_epochs=$EPOCHS" \
    --override "alpha=$ALPHA" \
    --override "beta=$ALPHA" \
    --override "output_path=$OUTPUT_DIR"

# Create baseline metrics for compatibility
if [ -f "$OUTPUT_DIR/result.csv" ]; then
    python3 -c "
import json
import pandas as pd
df = pd.read_csv('$OUTPUT_DIR/result.csv')
metrics = {
    'alpha': $ALPHA,
    'final_objective': float(df['final_objective'].iloc[-1]) if 'final_objective' in df else 0.0,
    'total_runtime': float(df['runtime'].iloc[-1]) if 'runtime' in df else 0.0,
    'num_epochs': $EPOCHS,
    'precond_type': 'baseline',
    'status': 'success'
}
with open('$OUTPUT_DIR/baseline_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
"
fi

echo ""
echo "=== Test Complete ==="
echo "Output: $OUTPUT_DIR"
echo ""
echo "Check results:"
echo "  cat $OUTPUT_DIR/result.csv"
echo "  cat $OUTPUT_DIR/baseline_metrics.json"
echo ""
