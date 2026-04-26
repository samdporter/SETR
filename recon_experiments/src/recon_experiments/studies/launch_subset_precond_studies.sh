#!/usr/bin/env bash

# Launch the subset-selection and preconditioner studies for both datasets.
#
# Usage:
#   ./launch_subset_precond_studies.sh [test|full|local]
#
# Environment overrides:
#   RUN_SUBSET=true|false
#   RUN_PRECOND=true|false
#   PHANTOM_BASELINE_EPOCHS=1000
#   PATIENT_BASELINE_EPOCHS=1000
#   POLL_SECONDS=300
#   SWEEP_REPEATS=5
#   LOCAL_RUN_ID=20260426_120000

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SUBSET_DIR="$SCRIPT_DIR/subset_selection"
PRECOND_DIR="$SCRIPT_DIR/preconditioners"

MODE="${1:-test}"
RUN_SUBSET="${RUN_SUBSET:-true}"
RUN_PRECOND="${RUN_PRECOND:-true}"
PHANTOM_BASELINE_EPOCHS="${PHANTOM_BASELINE_EPOCHS:-1000}"
PATIENT_BASELINE_EPOCHS="${PATIENT_BASELINE_EPOCHS:-1000}"
POLL_SECONDS="${POLL_SECONDS:-300}"

if [[ "$MODE" == "local" ]]; then
    export LOCAL_RUN_ID="${LOCAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
fi

show_usage() {
    echo "Usage: $0 [test|full|local]"
    echo ""
    echo "Launches subset-selection and preconditioner studies for:"
    echo "  - 1-bed anthropomorphic phantom"
    echo "  - 2-bed patient data"
    echo ""
    echo "Modes:"
    echo "  test   Submit one cluster test job per study stage"
    echo "  full   Submit the full cluster run"
    echo "  local  Run one local test job for each stage"
    echo ""
    echo "Environment overrides:"
    echo "  RUN_SUBSET=true|false"
    echo "  RUN_PRECOND=true|false"
    echo "  PHANTOM_BASELINE_EPOCHS=1000"
    echo "  PATIENT_BASELINE_EPOCHS=1000"
    echo "  POLL_SECONDS=300"
    echo "  SWEEP_REPEATS=5"
    echo "  LOCAL_RUN_ID=20260426_120000"
}

case "$MODE" in
    test|full|local)
        ;;
    help|--help|-h)
        show_usage
        exit 0
        ;;
    *)
        echo "Error: unsupported mode '$MODE'. Expected test, full, or local."
        exit 1
        ;;
esac

if [[ ! -d "$SUBSET_DIR" ]]; then
    echo "Error: subset study directory not found: $SUBSET_DIR"
    exit 1
fi

if [[ ! -d "$PRECOND_DIR" ]]; then
    echo "Error: preconditioner study directory not found: $PRECOND_DIR"
    exit 1
fi

run_launcher() {
    if [[ "$MODE" == "full" ]]; then
        yes y | "$@"
    else
        "$@"
    fi
}

expected_baselines() {
    local mode="$1"
    local alphas_file="$PRECOND_DIR/parameters/alphas.csv"

    if [[ "$mode" == "test" ]]; then
        printf "1"
        return
    fi

    tail -n +2 "$alphas_file" | awk -F, 'NF{gsub(/^[ \t]+|[ \t]+$/, "", $1); if($1!="") c++} END{print c+0}'
}

wait_for_baselines() {
    local bpos="$1"
    local expected="$2"
    local out_dir="$PRECOND_DIR/output/baselines_${bpos}bpos"

    mkdir -p "$out_dir"

    while true; do
        completed=$(find "$out_dir" -name "baseline_metrics.json" 2>/dev/null | wc -l)
        echo "Baselines ${bpos}bpos complete: $completed/$expected"

        if [[ "$completed" -ge "$expected" ]]; then
            break
        fi

        sleep "$POLL_SECONDS"
    done
}

echo "=== Subset + Preconditioner Study Launcher ==="
echo "Mode: $MODE"
echo "Run subset studies: $RUN_SUBSET"
echo "Run preconditioner studies: $RUN_PRECOND"
echo "Phantom baseline epochs: $PHANTOM_BASELINE_EPOCHS"
echo "Patient baseline epochs: $PATIENT_BASELINE_EPOCHS"
echo "Poll seconds: $POLL_SECONDS"
if [[ "$MODE" == "local" ]]; then
    echo "Local run ID: $LOCAL_RUN_ID"
fi
echo ""

if [[ "$RUN_SUBSET" == "true" ]]; then
    echo "[1/4] Launching subset-selection studies"
    run_launcher "$SUBSET_DIR/scripts/launch_sweep.sh" sweep_main_experiments.yaml "$MODE"
    run_launcher "$SUBSET_DIR/scripts/launch_sweep.sh" sweep_convergence_ref.yaml "$MODE"
    run_launcher "$SUBSET_DIR/scripts/launch_sweep.sh" sweep_main_experiments_2bpos.yaml "$MODE"
    run_launcher "$SUBSET_DIR/scripts/launch_sweep.sh" sweep_convergence_ref_2bpos.yaml "$MODE"
else
    echo "[1/4] Skipping subset-selection studies"
fi

if [[ "$RUN_PRECOND" != "true" ]]; then
    echo "[2/4] Skipping preconditioner studies"
    echo "Done."
    exit 0
fi

echo ""
echo "[2/4] Launching preconditioner baselines"
run_launcher "$PRECOND_DIR/launch_baseline_recons.sh" config_1bpos_anthro.yaml "$PHANTOM_BASELINE_EPOCHS" "$MODE"
run_launcher "$PRECOND_DIR/launch_baseline_recons.sh" config_2bpos.yaml "$PATIENT_BASELINE_EPOCHS" "$MODE"

echo ""
echo "[3/4] Waiting for baseline availability"
case "$MODE" in
    test|full)
        EXPECTED_BASELINES="$(expected_baselines "$MODE")"
        wait_for_baselines 1 "$EXPECTED_BASELINES"
        wait_for_baselines 2 "$EXPECTED_BASELINES"
        ;;
    local)
        echo "Local baseline runs are blocking; continuing directly to sweeps."
        ;;
esac

echo ""
echo "[4/4] Launching preconditioner sweeps"
if [[ -z "${SWEEP_REPEATS:-}" && "$MODE" != "local" ]]; then
    export SWEEP_REPEATS=5
    echo "Defaulting SWEEP_REPEATS=$SWEEP_REPEATS for $MODE mode."
fi
run_launcher "$PRECOND_DIR/launch_precond_sweep.sh" precond_sweep_1bpos.yaml "$MODE"
run_launcher "$PRECOND_DIR/launch_precond_sweep.sh" precond_sweep_2bpos.yaml "$MODE"

echo ""
echo "Done."
if [[ "$MODE" == "local" ]]; then
    echo "Subset outputs: $SUBSET_DIR/output/local_runs/$LOCAL_RUN_ID"
    echo "Preconditioner outputs: $PRECOND_DIR/output/local_runs/$LOCAL_RUN_ID"
else
    echo "Subset outputs: $SUBSET_DIR/output"
    echo "Preconditioner outputs: $PRECOND_DIR/output"
fi
