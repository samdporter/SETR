#!/bin/bash
# Run preconditioner comparison tests (LOCAL VERSION)
#
# For cluster execution, use: functionality/preconditioners/launch_precond_sweep.sh
#
# Usage:
#   ./scripts/run_preconditioner_tests.sh [quick|full]

set -e

MODE="${1:-full}"

echo "=========================================="
echo "VTV Preconditioner Comparison Tests"
echo "LOCAL EXECUTION (shared setup)"
echo ""
echo "For cluster execution with independent"
echo "setups per job, use:"
echo "  functionality/preconditioners/launch_precond_sweep.sh"
echo "=========================================="
echo ""

case "$MODE" in
  quick)
    echo "Running QUICK test with 10 epochs"
    echo "  3 alphas × 1 step size × 1 preconditioner = 3 runs"
    echo "  Estimated time: ~1-2 hours"
    python scripts/test_preconditioners.py \
      --config configs/config_2bpos.yaml \
      --output results/preconditioner_tests_quick \
      --alphas 500 5000 50000 \
      --step-sizes 0.1 1 \
      --precond-types bsrem vtv_svd_principal_alpha vtv_mm_jensen \
      --epochs 10

    echo ""
    echo "Running analysis..."
    python scripts/analyze_preconditioner_tests.py \
      --results results/preconditioner_tests_quick/results.csv
    ;;

  full)
    echo "Running FULL test (50 epochs)"
    echo "  4 alphas × 4 step sizes × 5 preconditioners = 80 runs"
    echo "  Estimated time: ~4-6 days"
    echo ""
    echo "Preconditioner methods to test:"
    echo "  1. bsrem (baseline, no VTV preconditioning)"
    echo "  2. vtv_svd_principal_alpha (SVD principal + isotropic α)"
    echo "  3. vtv_mm_jensen (MM/Jensen, SVD-free)"
    echo "  4. vtv_frobenius_surrogate_pd (Frobenius surrogate, PD)"
    echo "  5. vtv_vector_tv_per_modality (Per-modality vector TV)"
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
      --precond-types bsrem vtv_svd_principal_alpha vtv_mm_jensen vtv_frobenius_surrogate_pd vtv_vector_tv_per_modality \
      --epochs 50

    echo ""
    echo "Running analysis..."
    python scripts/analyze_preconditioner_tests.py \
      --results results/preconditioner_tests/results.csv
    ;;

  *)
    echo "Usage: $0 [quick|full]"
    echo ""
    echo "  quick: Fast test with 18 runs (~1-2 hours)"
    echo "  full:  Complete test with 80 runs (~4-6 days)"
    echo ""
    echo "Available preconditioner methods:"
    echo "  - bsrem: Baseline (no VTV preconditioning)"
    echo "  - vtv_svd_principal_alpha: SVD principal + isotropic α"
    echo "  - vtv_mm_jensen: Jensen-bound MM, SVD-free"
    echo "  - vtv_frobenius_surrogate_pd: Frobenius surrogate (PD)"
    echo "  - vtv_vector_tv_per_modality: Per-modality vector TV"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo "Tests complete!"
echo "=========================================="
