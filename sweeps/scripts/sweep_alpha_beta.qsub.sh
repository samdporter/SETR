#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -j y
# SGE parameters will be set dynamically by launcher script

# --- Configuration ---
# Get base directory from environment (set by launcher) or detect it
if [ -n "$SETR_BASE_DIR" ]; then
    BASE_DIR="$SETR_BASE_DIR"
else
    # Fallback: assume we're in BASE_DIR/sweeps/scripts/
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
fi

SWEEPS_DIR="$BASE_DIR/sweeps"
PARAM_DIR="$SWEEPS_DIR/parameters"
CONFIG_DIR="$SWEEPS_DIR/configs"
OUTPUT_BASE_DIR="$SWEEPS_DIR/output"
SCRIPTS_DIR="$BASE_DIR/scripts"

# Sweep settings from environment (set by launcher)
SWEEP_NAME=${SWEEP_NAME:-"default_sweep"}
BASE_CONFIG_FILE=${BASE_CONFIG_FILE:-"config_1bpos.yaml"}
RECON_SCRIPT=${RECON_SCRIPT:-"run_dtnv_1bpos.py"}
ALPHA_FILE=${ALPHA_FILE:-"alphas.csv"}
BETA_FILE=${BETA_FILE:-"betas.csv"}

# Create log directory if it doesn't exist
mkdir -p $HOME/setr_logs

# --- Load Environment ---
# Add any necessary module loads or environment activation here
# module load python/3.11
# source /path/to/your/conda/activate setr_env

# --- Job Array Setup ---
echo "Starting job array task: $SGE_TASK_ID for sweep: $SWEEP_NAME"
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Base config: $BASE_CONFIG_FILE"
echo "Script: $RECON_SCRIPT"

# --- Parameter Extraction ---
# Read alpha and beta values from CSV files
ALPHAS=($(tail -n +2 "$PARAM_DIR/$ALPHA_FILE" | cut -d',' -f1))
BETAS=($(tail -n +2 "$PARAM_DIR/$BETA_FILE" | cut -d',' -f1))

NUM_ALPHAS=${#ALPHAS[@]}
NUM_BETAS=${#BETAS[@]}

echo "Total alphas: $NUM_ALPHAS"
echo "Total betas: $NUM_BETAS"

# Calculate which alpha and beta to use for this task
ALPHA_INDEX=$(( (SGE_TASK_ID - 1) / NUM_BETAS ))
BETA_INDEX=$(( (SGE_TASK_ID - 1) % NUM_BETAS ))

# Handle case where task ID exceeds parameter combinations
if [ $ALPHA_INDEX -ge $NUM_ALPHAS ]; then
    echo "Task ID $SGE_TASK_ID exceeds available parameter combinations. Exiting."
    exit 0
fi

ALPHA=${ALPHAS[$ALPHA_INDEX]}
BETA=${BETAS[$BETA_INDEX]}

echo "Task $SGE_TASK_ID: Using alpha=$ALPHA, beta=$BETA"

# --- Output Directory Setup ---
OUTPUT_DIR="$OUTPUT_BASE_DIR/${SWEEP_NAME}/alpha_${ALPHA}_beta_${BETA}"
WORKING_DIR="$OUTPUT_DIR/tmp"
mkdir -p "$OUTPUT_DIR" "$WORKING_DIR"

# --- Configuration File Generation ---
CONFIG_FILE="$CONFIG_DIR/${SWEEP_NAME}_alpha_${ALPHA}_beta_${BETA}.yaml"

# Copy base config and modify parameters
cp "$BASE_DIR/configs/$BASE_CONFIG_FILE" "$CONFIG_FILE"

# Use Python to safely update YAML parameters
python3 -c "
import yaml
import sys

config_file = sys.argv[1]
alpha = float(sys.argv[2])
beta = float(sys.argv[3])
output_dir = sys.argv[4]
working_dir = sys.argv[5]

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

config['alpha'] = alpha
config['beta'] = beta
config['output_path'] = output_dir
config['working_path'] = working_dir

# Optionally adjust other parameters based on alpha/beta
# config['num_epochs'] = 100  # You might want different epochs for sweeps

with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
" "$CONFIG_FILE" "$ALPHA" "$BETA" "$OUTPUT_DIR" "$WORKING_DIR"

echo "Generated config file: $CONFIG_FILE"

# --- Job Execution ---
echo "Starting SETR reconstruction..."
echo "Config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"

cd "$BASE_DIR"

# Run the reconstruction using the script specified by launcher
python3 "$SCRIPTS_DIR/$RECON_SCRIPT" --config "$CONFIG_FILE"

RETURN_CODE=$?

# --- Post-processing ---
if [ $RETURN_CODE -eq 0 ]; then
    echo "Job completed successfully at: $(date)"
    
    # Create a completion marker
    echo "alpha=$ALPHA,beta=$BETA,status=completed,end_time=$(date)" > "$OUTPUT_DIR/job_completion.txt"
    
    # Optional: compress intermediate files to save space
    # cd "$WORKING_DIR" && tar -czf ../intermediate_files.tar.gz *.hv *.v *.ahv 2>/dev/null || true
    
else
    echo "Job failed with return code: $RETURN_CODE at: $(date)"
    echo "alpha=$ALPHA,beta=$BETA,status=failed,return_code=$RETURN_CODE,end_time=$(date)" > "$OUTPUT_DIR/job_completion.txt"
fi

echo "Task $SGE_TASK_ID finished with return code: $RETURN_CODE"
exit $RETURN_CODE