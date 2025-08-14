#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
set -euo pipefail

# Fail fast on *visible/assigned* GPU(s) with uncorrectable ECC
if command -v nvidia-smi >/dev/null; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra __GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    for __id in "${__GPU_IDS[@]}"; do
      if nvidia-smi -i "${__id}" \
            --query-gpu=ecc.errors.uncorrected.total \
            --format=csv,noheader 2>/dev/null | grep -q '^[1-9]'; then
        echo "GPU ${__id} reports uncorrectable ECC errors. Exiting."
        sleep 20; exit 99
      fi
    done
  fi
fi


# --- Runtime env: activate venv + SIRF ---
source "$HOME/sirf_venv/bin/activate"

export INSTALLDIR=/home/sporter/synergistic_Y90/devel/SIRF/SIRF_installs/Release-cuda12.0
source "${INSTALLDIR}/bin/env_sirf.sh"

# Remove any source-tree CIL path to avoid shadowing the wheel install
if [ -n "${PYTHONPATH:-}" ]; then
  PYTHONPATH="$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v 'CIL/Wrappers/Python' | paste -sd':' -)"
  export PYTHONPATH
fi

# Pre-flight: confirm correct CIL and that libcilacc exists
python - <<'PY'
import sys, pathlib, importlib
print("python:", sys.executable)
try:
    import cil
    libdir = pathlib.Path(cil.__file__).parent / "lib"
    print("cil   :", cil.__file__)
    print("cilacc:", list(libdir.glob("*cilacc*")))
except Exception as e:
    print("E: import cil failed ->", e)
    raise
PY

# --- Layout from launcher (or infer) ---
if [ -n "${SETR_BASE_DIR:-}" ]; then
    BASE_DIR="$SETR_BASE_DIR"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
fi

SWEEPS_DIR="$BASE_DIR/sweeps"
PARAM_DIR="$SWEEPS_DIR/parameters"
CONFIG_DIR="$SWEEPS_DIR/configs"
OUTPUT_BASE_DIR="$SWEEPS_DIR/output"
SCRIPTS_DIR="$BASE_DIR/scripts"

SWEEP_NAME=${SWEEP_NAME:-default_sweep}
BASE_CONFIG_FILE=${BASE_CONFIG_FILE:-config_1bpos.yaml}
RECON_SCRIPT=${RECON_SCRIPT:-run_dtnv_1bpos.py}
ALPHA_FILE=${ALPHA_FILE:-alphas.csv}
BETA_FILE=${BETA_FILE:-betas.csv}

mkdir -p "$HOME/setr_logs"

echo "Starting job array task: ${SGE_TASK_ID:-1} for sweep: $SWEEP_NAME"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Base config: $BASE_CONFIG_FILE"
echo "Script: $RECON_SCRIPT"

# --- Read parameters (ignore blanks/header) --
mapfile -t ALPHAS < <(tail -n +2 "$PARAM_DIR/$ALPHA_FILE" | awk -F, 'NF{gsub(/^[ \t]+|[ \t]+$/,"",$1); if($1!="") print $1}')
mapfile -t BETAS  < <(tail -n +2 "$PARAM_DIR/$BETA_FILE"  | awk -F, 'NF{gsub(/^[ \t]+|[ \t]+$/,"",$1); if($1!="") print $1}')

NUM_ALPHAS=${#ALPHAS[@]}
NUM_BETAS=${#BETAS[@]}
echo "Total alphas: $NUM_ALPHAS"
echo "Total betas : $NUM_BETAS"

if [ "$NUM_ALPHAS" -eq 0 ] || [ "$NUM_BETAS" -eq 0 ]; then
    echo "No parameters to run. Exiting."
    exit 0
fi

TASK_ID=${SGE_TASK_ID:-1}
ALPHA_INDEX=$(( (TASK_ID - 1) / NUM_BETAS ))
BETA_INDEX=$(( (TASK_ID - 1) % NUM_BETAS ))

if [ $ALPHA_INDEX -ge $NUM_ALPHAS ]; then
    echo "Task ID $TASK_ID exceeds available parameter combinations. Exiting."
    exit 0
fi

ALPHA=${ALPHAS[$ALPHA_INDEX]}
BETA=${BETAS[$BETA_INDEX]}
echo "Task $TASK_ID: Using alpha=$ALPHA, beta=$BETA"

# --- Paths for this job ---
OUTPUT_DIR="$OUTPUT_BASE_DIR/${SWEEP_NAME}/alpha_${ALPHA}_beta_${BETA}"
WORKING_DIR="$OUTPUT_DIR/tmp"
mkdir -p "$OUTPUT_DIR" "$WORKING_DIR"

CONFIG_FILE="$CONFIG_DIR/${SWEEP_NAME}_alpha_${ALPHA}_beta_${BETA}.yaml"
cp "$BASE_DIR/configs/$BASE_CONFIG_FILE" "$CONFIG_FILE"

# --- Edit YAML config (requires PyYAML in the venv) ---
python - <<'PY' "$CONFIG_FILE" "$ALPHA" "$BETA" "$OUTPUT_DIR" "$WORKING_DIR"
import sys, yaml
cfg, alpha, beta, outd, workd = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
with open(cfg, 'r') as f:
    data = yaml.safe_load(f)
data['alpha'] = alpha
data['beta'] = beta
data['output_path'] = outd
data['working_path'] = workd
with open(cfg, 'w') as f:
    yaml.dump(data, f, default_flow_style=False)
PY

echo "Generated config file: $CONFIG_FILE"

# --- Run reconstruction ---
cd "$BASE_DIR"
python "$SCRIPTS_DIR/$RECON_SCRIPT" --config "$CONFIG_FILE"
RETURN_CODE=$?

# --- Post-processing ---
if [ $RETURN_CODE -eq 0 ]; then
    echo "Job completed successfully at: $(date)"
    echo "alpha=$ALPHA,beta=$BETA,status=completed,end_time=$(date)" > "$OUTPUT_DIR/job_completion.txt"
else
    echo "Job failed with return code: $RETURN_CODE at: $(date)"
    echo "alpha=$ALPHA,beta=$BETA,status=failed,return_code=$RETURN_CODE,end_time=$(date)" > "$OUTPUT_DIR/job_completion.txt"
fi

echo "Task ${TASK_ID} finished with return code: $RETURN_CODE"
exit $RETURN_CODE
