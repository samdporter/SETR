# SETR Parameter Sweep System

The scripts in the `sweeps/` directory drive parameter sweeps of SETR reconstructions on SGE clusters. This guide consolidates the workflow documentation in a single place.

## Quick Start

1. **Set up parameters** (edit if needed):
   ```bash
   # Edit parameter ranges
   nano sweeps/parameters/alphas.csv
   nano sweeps/parameters/betas.csv
   ```

2. **Launch a sweep**:
   ```bash
   # Test with single job first
   cd sweeps
   ./launch_sweep.sh sweep_1bpos.yaml test
   
   # Launch full sweep
   ./launch_sweep.sh sweep_1bpos.yaml
   ```

3. **Monitor progress**:
   ```bash
   # Check SGE queue
   qstat -u $USER
   
   # Monitor specific sweep
   ./monitor_sweep.sh 1bpos_alpha_beta
   ```

4. **Collect results** (after jobs complete):
   ```bash
   ./collect_results.sh 1bpos_alpha_beta
   ```

## Directory Structure

```
sweeps/
├── configs/                    # Sweep configurations
│   ├── sweep_1bpos.yaml       # Single bed position sweep
│   └── sweep_2bpos.yaml       # Multiple bed position sweep
├── parameters/                 # Parameter files
│   ├── alphas.csv             # Alpha values to sweep
│   └── betas.csv              # Beta values to sweep  
├── scripts/                    # SGE job scripts
│   └── sweep_alpha_beta.qsub.sh
├── output/                     # Results (created during runs)
│   └── <sweep_name>/
│       └── alpha_X_beta_Y/    # Individual job outputs
└── launch_sweep.sh            # Main launcher script
```

## Sweep Configurations

### Single Bed Position (`sweep_1bpos.yaml`)
- Memory: 60GB
- Runtime: 48 hours  
- Script: `run_dtnv_1bpos.py`
- Base config: `config_1bpos.yaml`

### Multiple Bed Position (`sweep_2bpos.yaml`)
- Memory: 95GB
- Runtime: 72 hours
- Script: `run_dtnv_2bpos.py` 
- Base config: `config_2bpos.yaml`

## Parameter Files

Edit `parameters/alphas.csv` and `parameters/betas.csv` to set the parameter ranges:

```csv
alpha
0.1
0.5
1.0
2.0
5.0
```

**Note**: The sweep will run **all combinations** (alphas × betas). With 9 alphas and 9 betas = 81 total jobs.

## Usage Examples

### Test Mode (Single Job)
```bash
./launch_sweep.sh sweep_1bpos.yaml test
```
Runs only the first parameter combination to test the setup.

### Full Sweep
```bash
./launch_sweep.sh sweep_1bpos.yaml
```
Runs all parameter combinations.

### Monitor Running Jobs
```bash
# Basic SGE status
qstat -u $USER

# Detailed view of specific job
qstat -j <job_id>

# Monitor sweep progress
./monitor_sweep.sh 1bpos_alpha_beta
```

### Cancel Jobs
```bash
# Cancel specific job
qdel <job_id>

# Cancel all jobs for a sweep
qstat -u $USER | grep "1bpos_alpha_beta" | awk '{print $1}' | xargs qdel
```

## Results Collection

After jobs complete, collect results into a summary CSV:

```bash
./collect_results.sh 1bpos_alpha_beta
```

This creates `output/1bpos_alpha_beta_summary.csv` with:
- Parameter combinations (alpha, beta)
- Job status (completed/failed/pending)
- Final objective values
- Number of iterations
- Output directories
- Completion times

### Analyzing Results

```bash
# View summary
column -t -s',' output/1bpos_alpha_beta_summary.csv | less

# Find best results (lowest objective)
tail -n +2 output/1bpos_alpha_beta_summary.csv | sort -t',' -k4 -n | head -10
```

## Output Structure

Each job creates an output directory:
```
output/1bpos_alpha_beta/alpha_1.0_beta_2.0/
├── image_*.hv              # Reconstructed images
├── objective.csv           # Objective function values
├── bsrem_objective_*.csv   # BSREM-specific objectives  
├── gradient_*.hv           # Gradient images
├── preconditioner_*.hv     # Preconditioner images
├── kappa_sq_*.hv          # Kappa squared images
├── args.csv               # Job arguments/parameters
├── job_completion.txt     # Completion status
└── tmp/                   # Working files
```

## Customization

### Adding New Parameters
1. Create new parameter CSV files in `parameters/`
2. Modify the job script to read additional parameters
3. Update the sweep configuration YAML

### Custom Sweep Configurations
1. Copy an existing sweep YAML (e.g., `sweep_1bpos.yaml`)
2. Modify SGE resources, base config, and parameters
3. Launch with: `./launch_sweep.sh your_sweep.yaml`

### Resource Requirements
- **1bpos**: 60GB memory, 48h runtime (recommended for single bed position)
- **2bpos**: 95GB memory, 72h runtime (recommended for multiple bed positions)

Adjust in the YAML files based on your cluster and data size.

## Troubleshooting

### Jobs Stuck in Queue
- Check cluster status: `qstat -g c`
- Verify resource requests aren't too high
- Check queue limits: `qstat -q`

### Jobs Failing
- Check log files in `$HOME/setr_logs/`
- Verify data paths in base config files
- Test single job first: `./launch_sweep.sh sweep_1bpos.yaml test`

### Out of Memory
- Increase memory in sweep YAML: `memory: "120G"`
- Consider reducing number of epochs or image size

### Missing Results
- Check if jobs completed: `./monitor_sweep.sh <sweep_name>`
- Verify output paths in job scripts match expectations
- Look for error messages in SGE log files
