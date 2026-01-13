# Cluster and HPC Usage Guide

This guide covers running SETR reconstructions on HPC clusters using Sun Grid Engine (SGE) or similar batch systems.

## Table of Contents

- [Quick Start](#quick-start)
- [Job Submission Basics](#job-submission-basics)
- [Common Workflows](#common-workflows)
  - [Bootstrap Reconstructions (MANC Dataset)](#bootstrap-reconstructions-manc-dataset)
  - [Parameter Sweeps](#parameter-sweeps)
  - [Preconditioner Comparison](#preconditioner-comparison)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Resource Requirements](#resource-requirements)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

1. **Install packages on cluster**:
   ```bash
   # Load required modules (cluster-specific)
   module load cuda/11.8
   module load python/3.10

   # Install SETR packages
   pip install -e recon_core/
   pip install -e recon_experiments/
   ```

2. **Verify installation**:
   ```bash
   python -c "import recon_core; print(recon_core.__version__)"
   python -c "import recon_experiments; print('OK')"
   ```

3. **Test locally first**:
   ```bash
   # Run a quick test reconstruction (5 epochs)
   python -m recon_experiments.runners.scripts.run_dtnv_1bpos \
       --config recon_experiments/configs/config_1bpos.yaml \
       --override num_epochs=5
   ```

## Job Submission Basics

### SGE Job Script Template

```bash
#!/bin/bash
#$ -N setr_job              # Job name
#$ -l h_rt=48:00:00         # Runtime (48 hours)
#$ -l h_vmem=95G            # Memory
#$ -pe smp 4                # Cores
#$ -l gpu=1                 # GPU (if available)
#$ -cwd                     # Run in current directory
#$ -o logs/job_$JOB_ID.out  # Output log
#$ -e logs/job_$JOB_ID.err  # Error log

# Load environment
module load cuda/11.8
module load python/3.10
source /path/to/venv/bin/activate

# Run reconstruction
python -m recon_experiments.runners.scripts.run_dtnv_2bpos \
    --config recon_experiments/configs/config_2bpos.yaml \
    --output results/my_reconstruction
```

### Submitting Jobs

```bash
# Create logs directory
mkdir -p logs

# Submit job
qsub my_job.qsub.sh

# Check queue
qstat -u $USER

# Cancel job
qdel <job_id>
```

## Common Workflows

### Bootstrap Reconstructions (MANC Dataset)

Bootstrap reconstructions for uncertainty quantification on the Manchester bootstrap dataset.

#### DTNV (Dual Total Nuclear Variation)

Test different alpha/beta regularisation parameters across 30 bootstrap replicates.

**Local test (5 epochs, bootstrap 0)**:
```bash
cd recon_experiments/src/recon_experiments/runners/scripts
./test_manc_bootstrap_dtnv_local.sh 0.5 1.0 0
```

**Submit cluster job array (30 bootstraps)**:
```bash
./launch_manc_bootstrap_dtnv.sh 0.5 1.0 30
```

**Monitor progress**:
```bash
qstat -u $USER
tail -f ~/setr_logs/manc_dtnv_a0.5_b1.0_*.log
```

**Output structure**:
```
results/manc_bootstraps/dtnv/
└── alpha_0.5_beta_1.0/
    ├── bootstrap_000/
    │   ├── reconstruction_x.hv
    │   ├── objective.csv
    │   └── job_completion.txt
    ├── bootstrap_001/
    └── ...
```

**Key parameters**:
- Alpha/Beta: Set via command line (e.g., 0.5, 1.0)
- Epochs: 50 (configurable in config)
- Gamma_TNV: 50.0 (can override with `--override`)
- Flip: Enabled (required for MANC data)

#### HKEM (Hybrid KEM with SPECT Guidance)

Run HKEM reconstruction using SPECT emission as anatomical guidance for PET.

**Local test**:
```bash
cd recon_experiments/src/recon_experiments/runners/scripts
./test_manc_bootstrap_hkem_local.sh 0
```

**Full pipeline (30 bootstraps)**:
```bash
./run_manc_bootstrap_hkem.sh 30
```

This runs sequentially:
1. SPECT reconstruction (shared across bootstraps)
2. PET reconstructions with SPECT guidance (one per bootstrap)

**Output structure**:
```
results/manc_bootstraps/hkem/
├── spect_hkem/
│   └── reconstruction_x.hv      # Shared SPECT recon
├── bootstrap_000/
│   └── reconstruction_x.hv      # PET recon with guidance
├── bootstrap_001/
└── ...
```

**Key parameters**:
- SPECT: 15 epochs, 12 subsets
- PET: 15 epochs, 9 subsets
- Guidance: SPECT emission image
- Flip: Enabled

**Configuration files**:
- DTNV: `recon_experiments/configs/config_manc_bootstrap_dtnv.yaml`
- HKEM SPECT: `recon_experiments/configs/config_manc_hkem_spect.yaml`
- HKEM PET: `recon_experiments/configs/config_manc_hkem_pet.yaml`

### Parameter Sweeps

Run reconstruction parameter sweeps using the sweep framework in `recon_experiments/src/recon_experiments/sweeps/`.

**Sweep configuration** (`sweeps/my_sweep.yaml`):
```yaml
sweep_name: "alpha_beta_sweep"
base_config: "config_2bpos.yaml"
script: "run_dtnv_2bpos.py"

sge:
  runtime: "48:00:00"
  memory: "95G"
  cores: 4
  gpu: true

parameters:
  alpha: [0.1, 0.5, 1.0, 5.0]
  beta: [0.5, 1.0, 2.0]
  gamma_tnv: [10.0, 50.0, 100.0]

fixed_params:
  num_epochs: 50
```

**Submit sweep**:
```bash
cd recon_experiments/src/recon_experiments/sweeps
python run_sweep.py --config my_sweep.yaml --mode cluster
```

This submits 4 × 3 × 3 = 36 jobs to the cluster.

### Preconditioner Comparison

Test different VTV preconditioner methods to find optimal performance.

**Location**: `recon_experiments/src/recon_experiments/studies/preconditioners/`

**Preconditioner methods**:
1. `bsrem` - Baseline (no VTV preconditioning)
2. `vtv_svd_principal_alpha` - SVD principal rank-one terms
3. `vtv_mm_jensen` - MM/Jensen surrogate (SVD-free, ~3× faster)
4. `vtv_frobenius_surrogate_pd` - Frobenius surrogate (guaranteed positive)
5. `vtv_vector_tv_per_modality` - Per-modality vector TV (exact)

**Quick start**:

1. **Local test** (single configuration):
   ```bash
   cd recon_experiments/src/recon_experiments/studies/preconditioners
   ./launch_precond_sweep.sh precond_sweep_2bpos.yaml local
   ```

2. **Cluster test** (single job):
   ```bash
   ./launch_precond_sweep.sh precond_sweep_2bpos.yaml test
   ```

3. **Full sweep** (all combinations):
   ```bash
   ./launch_precond_sweep.sh precond_sweep_2bpos.yaml full
   ```

   This submits: 5 precond types × 7 alphas × 4 step sizes = **140 jobs**

**Parameter files** (`parameters/`):
- `precond_types.csv` - Preconditioner methods to test
- `alphas.csv` - Prior strength values
- `step_sizes.csv` - Step sizes to test

**Output**:
```
output/precond_2bpos/
├── precond_vtv_mm_jensen_alpha_500.0_step_0.1/
│   ├── image_*.hv
│   ├── objective.csv
│   ├── preconditioner_*.hv
│   ├── result.csv
│   └── preconditioner_test.log
└── ...
```

**Collect results**:
```bash
python collect_precond_results.py --sweep precond_2bpos
# Creates: output/precond_2bpos_summary.csv
```

**Resource requirements** (2 bed positions):
- Memory: 95GB per job
- Runtime: 48 hours per job
- GPU: Required
- Disk: ~10GB per job

## Monitoring and Debugging

### Check Job Status

```bash
# List your jobs
qstat -u $USER

# Count running/pending jobs
qstat -u $USER | grep setr | wc -l

# View specific job details
qstat -j <job_id>

# View job history
qacct -j <job_id>
```

### View Logs

```bash
# Check job output logs
tail -f logs/job_*.out

# Check error logs
tail -f logs/job_*.err

# Search for errors
grep -i "error\|fail" logs/*.err
```

### Check Reconstruction Progress

```bash
# View objective function values
tail -f results/my_reconstruction/objective.csv

# Count saved images (iterations)
ls -1 results/my_reconstruction/image_*.hv | wc -l

# Check if job completed
cat results/my_reconstruction/job_completion.txt
```

### GPU Usage

```bash
# Check GPU availability on nodes
qhost -F gpu

# Monitor GPU usage (if logged into compute node)
nvidia-smi -l 5
```

## Resource Requirements

### Typical Requirements by Dataset Size

| Dataset | Memory | Runtime | GPU | Disk |
|---------|--------|---------|-----|------|
| 1 bed position (PET only) | 30-40G | 12-24h | Recommended | ~5GB |
| 1 bed position (PET+SPECT) | 40-60G | 24-36h | Recommended | ~8GB |
| 2 bed positions (PET+SPECT) | 80-100G | 36-48h | Required | ~15GB |
| Bootstrap (30 replicates) | 60-80G each | 24-36h each | Recommended | ~300GB total |

### GPU Acceleration

GPU acceleration speeds up:
- VTV gradient computations (~10× faster)
- Schatten norm calculations
- Some preconditioner methods

**CPU-only option**: Set `no_gpu: true` in config, but expect significantly longer runtimes.

### Optimal Job Configuration

For 2 bed position data:
```bash
#$ -l h_rt=48:00:00    # 48 hours (safer than 36h)
#$ -l h_vmem=95G       # 95GB (safer than 80GB)
#$ -pe smp 4           # 4 cores (for data loading/saving)
#$ -l gpu=1            # 1 GPU
```

## Troubleshooting

### Jobs Fail Immediately

**Check**:
1. Module loading: `module list`
2. Python environment: `which python`
3. Package installation: `python -c "import recon_core"`
4. Data paths: Check paths in config files exist

**Common fixes**:
- Add `module load` commands to job script
- Activate correct virtual environment
- Use absolute paths in configs

### Out of Memory Errors

**Symptoms**:
- Job killed with "memory limit exceeded"
- Error logs show OOM messages
- Reconstructions fail during projection operations

**Fixes**:
1. Increase memory: `#$ -l h_vmem=120G`
2. Reduce subsets (uses less memory per iteration)
3. Enable GPU (offloads memory to GPU)
4. Use single bed position data for testing

### Job Timeout

**Symptoms**:
- Job killed after reaching runtime limit
- Partial output but no completion marker

**Fixes**:
1. Increase runtime: `#$ -l h_rt=72:00:00`
2. Reduce epochs for testing
3. Use faster preconditioner (e.g., `vtv_mm_jensen`)
4. Check if GPU is actually being used

### GPU Errors

**Symptoms**:
- CUDA errors in logs
- "GPU not available" messages
- Slower than expected (running on CPU)

**Check**:
1. GPU requested: `#$ -l gpu=1` in job script
2. CUDA loaded: `module load cuda`
3. PyTorch sees GPU: `python -c "import torch; print(torch.cuda.is_available())"`
4. Config setting: `no_gpu: false` in YAML

**Common issues**:
- Node GPU failure → Try different queue
- CUDA version mismatch → Check module versions
- Out of GPU memory → Reduce batch size or use CPU

### Data Loading Errors

**Symptoms**:
- "File not found" errors
- SIRF crashes during setup
- Empty acquisition data

**Fixes**:
1. Use absolute paths in configs
2. Check file permissions
3. Verify SIRF installation: `python -c "import sirf.STIR"`
4. Test data loading separately:
   ```bash
   python -c "from recon_core.utils import get_pet_data; get_pet_data('/path/to/data')"
   ```

### Partial Results

**Symptoms**:
- Some images saved but not all
- Job completed but results incomplete
- `job_completion.txt` missing or shows failure

**Check**:
1. Disk quota: `quota -s`
2. Disk space on compute node: `df -h`
3. Error logs for crash/failure messages
4. Objective function CSV for where it stopped

**Recovery**:
- Resume from checkpoint (if implemented)
- Reduce save frequency to save disk
- Clean up old results

### Networking Issues (Multi-Node)

**Symptoms**:
- Jobs hang indefinitely
- MPI errors (if using distributed computing)
- Data loading extremely slow

**Fixes**:
1. Use local scratch space for data: `#$ -l tmpfree=50G`
2. Copy data to node before processing
3. Check cluster file system status
4. Avoid I/O during peak hours

## Best Practices

1. **Always test locally first** - Run 5-10 epochs locally before submitting large jobs
2. **Use checkpoints** - Save intermediate results in case of failure
3. **Monitor early** - Check first few minutes to catch quick failures
4. **Batch similar jobs** - Group parameter variations into job arrays
5. **Clean up** - Remove old results to save disk space
6. **Document parameters** - Save config files with results
7. **Version control** - Tag code version used for experiments

## Additional Resources

- **Sweep framework**: See `recon_experiments/src/recon_experiments/sweeps/README.md`
- **Experiment structure**: See `recon_experiments/src/recon_experiments/experiments/README.md`
- **Configuration guide**: See `recon_core/README.md`
- **Preconditioner details**: See `docs/guides/preconditioner_testing.md`

## Support

If you encounter issues not covered here:
1. Check experiment-specific README files
2. Review job logs carefully
3. Test with minimal configuration (1 epoch, small data)
4. Contact cluster support for infrastructure issues
