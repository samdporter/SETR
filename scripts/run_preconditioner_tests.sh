#!/bin/bash
# Run preconditioner comparison tests
#
# Usage:
#   ./scripts/run_preconditioner_tests.sh [quick|full]

set -e

MODE="${1:-full}"

echo "=========================================="
echo "VTV Preconditioner Comparison Tests"
echo "=========================================="
echo ""

case "$MODE" in
  quick)
    echo "Running QUICK test with 5 epochs (2 alphas × 3 step sizes × 2 preconditioners = 12 runs)"
    echo "Estimated time: ~1 hour"
    python scripts/test_preconditioners.py \
      --config configs/config_2bpos.yaml \
      --output results/preconditioner_tests_quick \
      --alphas 100 500 \
      --step-sizes 0.01 0.1 1.0 \
      --precond-types bsrem vtv_fast \
      --epochs 5

    echo ""
    echo "Running analysis..."
    python scripts/analyze_preconditioner_tests.py \
      --results results/preconditioner_tests_quick/results.csv
    ;;

  full)
    echo "Running FULL test (50 epochs) (4 alphas × 4 step sizes × 3 preconditioners = 48 runs)"
    echo "Estimated time: ~3-5 days"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Aborted."
      exit 1
    fi

    python scripts/test_preconditioners.py \
      --config configs/config_2bpos.yaml \
      --output results/preconditioner_tests \
      --alphas 50 100 500 1000 \
      --step-sizes 0.05 0.1 0.5 1.0 \
      --precond-types bsrem vtv_fast vtv_slow \
      --epochs 50

    echo ""
    echo "Running analysis..."
    python scripts/analyze_preconditioner_tests.py \
      --results results/preconditioner_tests/results.csv
    ;;

  *)
    echo "Usage: $0 [quick|full]"
    echo ""
    echo "  quick: Fast test with 12 runs (~6-8 hours)"
    echo "  full:  Complete test with 72 runs (~3-5 days)"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo "Tests complete!"
echo "=========================================="
