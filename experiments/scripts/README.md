# Experiment Scripts

This directory contains SGE qsub scripts for running experiments as array jobs.

## Files

### run_phantom_experiment.qsub.sh

SGE worker script that runs individual phantom experiment tasks as part of an array job.

**Purpose**: Executed by SGE for each array task (1-15 for the full sweep).

**What it does**:
1. Sets up SIRF environment
2. Performs GPU and system health checks
3. Determines phantom + algorithm from task ID
4. Calls the master Python script with appropriate arguments
5. Handles retries and error reporting
6. Tracks node failures for blacklisting

**Used by**: `launch_experiments.sh`

**Environment variables** (passed by launcher):
- `SETR_BASE_DIR`: Repository root
- `SWEEP_NAME`: Name of the sweep
- `PHANTOMS`: Comma-separated list of phantoms
- `ALGORITHMS`: Comma-separated list of algorithms
- `CONFIG_OVERRIDES_JSON`: JSON string of config overrides
- `SGE_TASK_ID`: Array task ID (1-15)

**Task mapping**:
```
task_id = phantom_index * num_algorithms + algorithm_index + 1

Examples:
  Task 1  = phantoms[0] + algorithms[0] = manc + hkem
  Task 2  = phantoms[0] + algorithms[1] = manc + dtnv
  Task 6  = phantoms[1] + algorithms[0] = anthro + hkem
  Task 15 = phantoms[2] + algorithms[4] = nema + log_tnv
```

**Logs**: Written to `experiments/output/<sweep_name>/_logs/`

**Output**: Each task writes to `experiments/output/<sweep_name>/<phantom>_<algorithm>/`

## Usage

These scripts are not meant to be called directly. Use the launcher:

```bash
cd experiments
./launch_experiments.sh
```

## See Also

- [../ARRAY_JOBS.md](../ARRAY_JOBS.md) - Full array job documentation
- [../launch_experiments.sh](../launch_experiments.sh) - Main launcher script
- [../configs/sweep_phantom_experiments.yaml](../configs/sweep_phantom_experiments.yaml) - Sweep configuration
