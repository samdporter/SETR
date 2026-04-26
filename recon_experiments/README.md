# recon_experiments

Experiment framework and analysis tools for multimodal PET/SPECT reconstruction research.

## Overview

`recon_experiments` provides experiment runners, parameter sweeps, analysis scripts, and study workflows for reconstruction research using the `recon_core` library.

### Key Features

- **Experiment Runners**: Pre-configured scripts for DTNV, HKEM, RDP, and other reconstruction methods
- **Parameter Sweeps**: Infrastructure for systematic parameter exploration
- **HPC Integration**: Job submission scripts for cluster/HPC environments
- **Analysis Tools**: Plotting, metrics computation, and result aggregation
- **Study Workflows**: Organised studies on preconditioners, subset selection, etc.

## Installation

### Prerequisites

First, install the core reconstruction library:

```bash
cd ../recon_core
pip install -e .
```

Then install the experiments package:

```bash
cd ../recon_experiments
pip install -e .

# For development with Jupyter notebooks:
pip install -e .[dev]

# For HPC experiment tracking:
pip install -e .[hpc]
```

## Quick Start

### Running a Reconstruction

```bash
# Run DTNV reconstruction for single bed position
python -m recon_experiments.runners.scripts.run_dtnv_1bpos \
    --config configs/config_1bpos_manc.yaml

# Override config parameters
python -m recon_experiments.runners.scripts.run_dtnv_1bpos \
    --config configs/config_1bpos_manc.yaml \
    --override num_epochs=50 \
    --override alpha=1.5
```

### Running Parameter Sweeps

```bash
# Launch parameter sweep on cluster
cd recon_experiments/src/recon_experiments/sweeps
./launch_sweep.sh configs/sweep_config.yaml

# Analyse sweep results
python scripts/analyze_sweep_results.py --sweep-dir output/sweep_name
```

### Running Current Studies

To launch the subset-selection and preconditioner studies on both the phantom
and patient datasets:

```bash
cd recon_experiments/src/recon_experiments/studies

# Submit one cluster test job per study stage
./launch_subset_precond_studies.sh test

# Submit the full phantom + patient study set
./launch_subset_precond_studies.sh full
```

The combined launcher submits subset-selection runs immediately, runs
preconditioner baselines for the 1-bed phantom and 2-bed patient configs, waits
for those baselines, then submits the preconditioner sweeps.

### HPC/Cluster Workflows

```bash
# Submit array job for multiple experiments
qsub -t 1-10 scripts/experiment.qsub.sh

# Monitor running jobs
./scripts/monitor_experiments.sh
```

## Package Structure

```
recon_experiments/
├── runners/                 # Experiment runner scripts
│   ├── common.py            # Shared experiment utilities
│   ├── dtnv_common.py       # DTNV-specific helpers
│   ├── hkem_common.py       # HKEM-specific helpers
│   └── scripts/             # Main experiment runners
│       ├── run_dtnv_1bpos.py
│       ├── run_dtnv_2bpos.py
│       ├── run_hkem_1bpos.py
│       └── ...
├── sweeps/                  # Parameter sweep infrastructure
│   ├── launch_sweep.sh
│   ├── configs/
│   ├── parameters/
│   └── scripts/
├── studies/                 # Organised research studies
│   ├── preconditioners/     # Preconditioner comparison study
│   └── subset_selection/    # Subset selection study
├── experiments/             # Experiment orchestration
│   └── scripts/
│       └── run_phantom_experiments.py
└── configs/                 # Experiment configurations
    ├── config_1bpos_*.yaml
    ├── config_2bpos_*.yaml
    └── ...
```

## Configuration Files

Experiment configurations are in YAML format. Example:

```yaml
# config_1bpos_manc.yaml
pet_data_path: /path/to/pet/data
spect_data_path: /path/to/spect/data
output_path: output/experiment_name
working_path: tmp/experiment_name

alpha: 1.0          # PET weight
beta: 0.25          # SPECT weight
num_subsets: [7, 7] # [PET, SPECT] subsets
num_epochs: 100     # Reconstruction iterations

pet_gauss_fwhm: 4.0     # PET resolution (mm)
spect_gauss_fwhm: 8.0   # SPECT resolution (mm)
use_kappa: true         # Cross-modal weighting
```

## Available Experiments

### Single Bed Position

- `run_dtnv_1bpos.py` - Directional Total Nuclear Variation
- `run_hkem_1bpos.py` - Hybrid Kernel EM
- `run_dtv_1bpos_pet.py` - Directional TV (PET only)
- `run_dtv_1bpos_spect.py` - Directional TV (SPECT only)
- `run_rdp_1bpos.py` - Relative Difference Prior

### Multiple Bed Positions

- `run_dtnv_2bpos.py` - DTNV for multi-bed
- `run_hkem_2bpos.py` - HKEM for multi-bed
- `run_dtv_2bpos_pet.py` - DTV for multi-bed PET

### Analysis and Testing

- `test_preconditioners.py` - Test different preconditioner strategies
- `test_spatial_ops.py` - Validate spatial operators
- `analyze_precond_sweep.py` - Analyse preconditioner sweep results

## HPC Usage

### Submitting Jobs

The experiment runners are designed to work on clusters:

```bash
# Interactive test
python -m recon_experiments.runners.scripts.run_dtnv_1bpos \
    --config configs/config_test_debug.yaml \
    --override num_epochs=1

# Submit to queue
qsub -l select=1:ncpus=8:ngpus=1:mem=32gb \
     -v CONFIG=configs/config_1bpos_manc.yaml \
     scripts/run_experiment.qsub.sh
```

### Array Jobs

For running multiple experiments:

```bash
# Launch array job (see ARRAY_JOBS.md for details)
qsub -t 1-20 scripts/array_job.qsub.sh

# Each task gets different parameters from CSV files
```

See [experiments/ARRAY_JOBS.md](src/recon_experiments/experiments/ARRAY_JOBS.md) for detailed array job documentation.

## Analysis Notebooks

Jupyter notebooks for analysis are in `notebooks/`:

```bash
# Start Jupyter
jupyter notebook notebooks/

# Open image_viewer.ipynb for interactive visualisation
```

## Output Structure

Experiment outputs are saved to `output_path`:

```
output/experiment_name/
├── image_*.hv              # Reconstructed images at iterations
├── gradient_*.hv           # Gradient updates
├── preconditioner_*.hv     # Preconditioner images
├── kappa_sq_*.hv           # Cross-modal weights
├── objective.csv           # Objective function values
└── *.png                   # Visualisation plots
```

## Development

### Adding New Experiments

1. Create new runner script in `runners/scripts/`
2. Use common utilities from `runners/common.py`
3. Add configuration template in `configs/`
4. Test locally before submitting to cluster

### Parameter Sweeps

Create sweep configuration:

```yaml
# sweeps/configs/my_sweep.yaml
base_config: ../../configs/config_1bpos_manc.yaml
sweep_parameters:
  - name: alpha
    values: [0.5, 1.0, 1.5, 2.0]
  - name: beta
    values: [0.1, 0.25, 0.5]
output_base: output/my_sweep
```

Run sweep:

```bash
./sweeps/launch_sweep.sh sweeps/configs/my_sweep.yaml
```

## Dependencies

**Core dependencies:**
- `recon-core` - Core reconstruction library
- `numpy`, `pandas`, `matplotlib` - Data processing and plotting
- `seaborn>=0.11.0` - Statistical visualisation
- `tqdm>=4.60.0` - Progress bars
- `pyyaml>=5.4.0` - Configuration parsing

**Optional:**
- `jupyter>=1.0.0` - Interactive notebooks
- `ipykernel>=6.0.0` - Jupyter kernel
- `wandb>=0.12.0` - Experiment tracking (HPC)

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure recon_core is installed first
pip install -e ../recon_core
pip install -e .
```

**Path issues in configs:**
- Use absolute paths or paths relative to where you run the script
- Set environment variables for data directories if needed

**GPU memory errors:**
- Reduce batch size or number of subsets
- Use `no_gpu: true` in config to force CPU mode

**Cluster job failures:**
- Check `.e` and `.o` files in job output directory
- Verify module loading in job scripts
- Test interactively before submitting to queue

## Support

- **Issues**: [GitHub Issues](https://github.com/samdporter/setr/issues)
- **Documentation**: [Read the Docs](https://setr.readthedocs.io)
- **HPC Guide**: See [CLUSTER_DEBUG_GUIDE.md](../CLUSTER_DEBUG_GUIDE.md)

## License

MIT License - see LICENSE file for details.
