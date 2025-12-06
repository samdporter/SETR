# Array Job Submission for Phantom Experiments

This guide explains how to submit phantom experiments as SGE array jobs for parallel execution on the cluster.

## Quick Start

```bash
# Navigate to experiments directory
cd experiments

# Submit all 15 experiments (3 phantoms × 5 algorithms) as an array job
./launch_experiments.sh

# Test with a single job first
./launch_experiments.sh sweep_phantom_experiments.yaml test

# Run locally without qsub (for debugging)
./launch_experiments.sh sweep_phantom_experiments.yaml local
```

## Overview

The array job system allows you to submit all phantom experiments simultaneously to the cluster, with each combination running as a separate task in an SGE array job.

**Total experiments**: 15 (3 phantoms × 5 algorithms)

| Phantom       | Algorithms |
|--------------|------------|
| Manchester NEMA (manc)   | hkem, dtnv, tnv, log_dtnv, log_tnv |
| Anthropomorphic (anthro) | hkem, dtnv, tnv, log_dtnv, log_tnv |
| NEMA                     | hkem, dtnv, tnv, log_dtnv, log_tnv |

## Directory Structure

```
experiments/
├── launch_experiments.sh              # Main launcher script
├── configs/
│   └── sweep_phantom_experiments.yaml # Array job configuration
├── scripts/
│   └── run_phantom_experiment.qsub.sh # SGE worker script
├── output/
│   └── phantom_experiments/           # Results (created on submission)
│       ├── _logs/                     # SGE job logs
│       ├── manc_hkem/                 # Individual experiment outputs
│       ├── manc_dtnv/
│       └── ...
└── failed_nodes.txt                   # Auto-managed blacklist
```

## Configuration File

The sweep configuration is defined in [configs/sweep_phantom_experiments.yaml](configs/sweep_phantom_experiments.yaml):

```yaml
sweep_name: "phantom_experiments"
experiment_type: "phantom_algorithm"

# SGE resource requirements (per experiment)
sge:
  runtime: "24:00:00"  # 24 hours
  memory: "32G"        # 32GB RAM
  cores: 2             # 2 CPU cores
  queue: null          # Use default queue
  gpu: true            # Enable GPU

# Experiment combinations
phantoms:
  - manc
  - anthro
  - nema

algorithms:
  - hkem
  - dtnv
  - tnv
  - log_dtnv
  - log_tnv

# Optional parameter overrides
config_overrides: {}
  # num_epochs: 100
  # alpha: 0.02
```

### Customizing Resources

Edit [configs/sweep_phantom_experiments.yaml](configs/sweep_phantom_experiments.yaml) to adjust:

- **Runtime**: Increase for longer experiments
- **Memory**: Adjust based on phantom size
- **GPU**: Set to `false` for CPU-only execution
- **Config overrides**: Modify reconstruction parameters globally

## Usage Modes

### Full Mode (Default)

Submit all 15 experiments as an array job:

```bash
./launch_experiments.sh
# or explicitly:
./launch_experiments.sh sweep_phantom_experiments.yaml full
```

This creates an SGE array job with tasks 1-15:
- Task 1: manc + hkem
- Task 2: manc + dtnv
- Task 3: manc + tnv
- Task 4: manc + log_dtnv
- Task 5: manc + log_tnv
- Task 6: anthro + hkem
- ...
- Task 15: nema + log_tnv

### Test Mode

Submit only the first experiment (manc + hkem) as a test:

```bash
./launch_experiments.sh sweep_phantom_experiments.yaml test
```

Use this to verify:
- SGE configuration is correct
- Environment setup works
- Output paths are writable

### Local Mode

Run the first experiment locally without qsub (for debugging):

```bash
./launch_experiments.sh sweep_phantom_experiments.yaml local
```

This runs `manc + hkem` directly on your current machine, useful for:
- Testing configuration changes
- Debugging script issues
- Verifying data paths

## Monitoring Jobs

### Check job status

```bash
# View all jobs
qstat

# View jobs for this sweep
qstat | grep setr_phantom_experiments

# Check specific task
qstat -j <job_id> -t <task_id>
```

### View logs

Logs are written to `output/phantom_experiments/_logs/`:

```bash
# List all logs
ls output/phantom_experiments/_logs/

# View specific task log
cat output/phantom_experiments/_logs/setr_phantom_experiments.o<job_id>.<task_id>

# Monitor running task
tail -f output/phantom_experiments/_logs/setr_phantom_experiments.o<job_id>.<task_id>
```

### Check experiment outputs

```bash
# List all experiment directories
ls output/phantom_experiments/

# Check specific experiment
ls output/phantom_experiments/manc_dtnv/

# View completion status
cat output/phantom_experiments/manc_dtnv/job_completion.txt
```

## Output Structure

Each experiment creates its own directory:

```
output/phantom_experiments/
├── _logs/                          # SGE job logs
│   ├── setr_phantom_experiments.o<job_id>.1  # Task 1 log (manc + hkem)
│   ├── setr_phantom_experiments.o<job_id>.2  # Task 2 log (manc + dtnv)
│   └── ...
├── manc_hkem/                      # HKEM has subdirectories
│   ├── spect/                      # Stage 1 outputs
│   ├── spect_resampled/            # Resampled outputs
│   ├── pet/                        # Stage 2 outputs
│   ├── job_completion.txt          # Job status
│   └── tmp/                        # Working directory
├── manc_dtnv/                      # dTNV outputs
│   ├── image_0_*.hv                # PET reconstructions
│   ├── image_1_*.hv                # SPECT reconstructions
│   ├── objective.csv               # Convergence tracking
│   ├── job_completion.txt
│   └── tmp/
└── ...
```

## Parameter Overrides

Override configuration parameters for all experiments:

### Edit config file

Modify [configs/sweep_phantom_experiments.yaml](configs/sweep_phantom_experiments.yaml):

```yaml
config_overrides:
  num_epochs: 50        # Reduce epochs for faster testing
  alpha: 0.05           # Change regularization
  beta: 0.05
  save_interval: 10     # Save images every 10 epochs
```

Then submit:

```bash
./launch_experiments.sh
```

## Error Handling and Node Blacklisting

The system automatically tracks failed nodes and excludes them from future submissions.

### How it works

1. If a job fails due to node issues (GPU errors, ECC errors, etc.), the hostname is logged
2. The launcher script reads `failed_nodes.txt` and excludes those nodes
3. Future submissions avoid problematic nodes automatically

### View failed nodes

```bash
cat experiments/failed_nodes.txt
```

### Manually add a node to blacklist

```bash
echo "problematic-node.local" >> experiments/failed_nodes.txt
```

### Remove a node from blacklist

```bash
# Remove specific node
grep -v "node-name.local" experiments/failed_nodes.txt > tmp && mv tmp experiments/failed_nodes.txt

# Clear all
rm experiments/failed_nodes.txt
```

### Permanently excluded nodes

The following nodes are permanently excluded (hardcoded in launcher):
- `hoots-207-1.local`
- `hoots-207-2.local`

## Advanced Usage

### Submit subset of experiments

Create a custom sweep config:

```yaml
# configs/sweep_custom.yaml
sweep_name: "custom_subset"
experiment_type: "phantom_algorithm"

sge:
  runtime: "24:00:00"
  memory: "32G"
  cores: 2
  gpu: true

# Only 2 phantoms × 2 algorithms = 4 experiments
phantoms:
  - manc
  - anthro

algorithms:
  - dtnv
  - log_dtnv

config_overrides:
  num_epochs: 200
```

Submit:

```bash
./launch_experiments.sh sweep_custom.yaml
```

### Different resources per algorithm

Currently, all experiments use the same resources. To vary resources, create separate sweep configs:

```bash
# High memory for HKEM
./launch_experiments.sh sweep_hkem_highmem.yaml

# Standard for TNV variants
./launch_experiments.sh sweep_tnv.yaml
```

### Resubmit failed experiments

Check which experiments failed:

```bash
# Find experiments without completion files or with failed status
for dir in output/phantom_experiments/*/; do
    if [ -f "$dir/job_completion.txt" ]; then
        if grep -q "status=failed" "$dir/job_completion.txt"; then
            echo "Failed: $(basename $dir)"
        fi
    else
        echo "Incomplete: $(basename $dir)"
    fi
done
```

Then create a custom sweep config with only the failed combinations.

## Task-to-Experiment Mapping

Array tasks are mapped to experiments in this order:

```
Task  Phantom  Algorithm
----  -------  ---------
1     manc     hkem
2     manc     dtnv
3     manc     tnv
4     manc     log_dtnv
5     manc     log_tnv
6     anthro   hkem
7     anthro   dtnv
8     anthro   tnv
9     anthro   log_dtnv
10    anthro   log_tnv
11    nema     hkem
12    nema     dtnv
13    nema     tnv
14    nema     log_dtnv
15    nema     log_tnv
```

Formula: `task_id = phantom_index * num_algorithms + algorithm_index + 1`

## Troubleshooting

### Job fails immediately

Check the log file for errors:

```bash
cat output/phantom_experiments/_logs/setr_phantom_experiments.o<job_id>.<task_id>
```

Common issues:
- **Environment not found**: SIRF venv or installation path incorrect
- **Config file not found**: Check experiment configs exist
- **Disk space**: Verify output directory is writable

### GPU errors

If you see CUDA/GPU errors:

1. Check GPU availability: `nvidia-smi`
2. The node will be auto-blacklisted
3. Resubmit to run on different nodes

### Out of memory

Increase memory in sweep config:

```yaml
sge:
  memory: "64G"  # Increase from 32G
```

Or disable GPU and use CPU:

```yaml
sge:
  gpu: false
  memory: "48G"  # Use h_vmem instead of tmem
```

### Experiment not found

Ensure the master script exists:

```bash
ls -l scripts/run_phantom_experiments.py
```

If missing, the array job cannot run.

## Comparison with Manual Execution

### Manual execution

```bash
# Run each experiment one by one
python scripts/run_phantom_experiments.py --phantom manc --algorithm hkem
python scripts/run_phantom_experiments.py --phantom manc --algorithm dtnv
# ... repeat 15 times
```

**Time**: Sequential, ~15 × 24h = 15 days

### Array job execution

```bash
# Submit all experiments simultaneously
./launch_experiments.sh
```

**Time**: Parallel, ~24h (assuming 15 nodes available)

## Notes

- Each experiment runs independently
- Failed experiments don't affect others
- Results are written to separate directories
- Job completion status is tracked per experiment
- Automatic retry on transient failures (up to 2 attempts)
- GPU health checks before execution
- Disk space verification before and during execution

## See Also

- [QUICKSTART.md](QUICKSTART.md) - Manual execution guide
- [README.md](README.md) - Full experiment documentation
- [../sweeps/README.md](../sweeps/README.md) - Parameter sweep documentation (alpha/beta)
