# Quick Start Guide - Phantom Experiments

## TL;DR - Cluster Array Jobs (Fastest)

Submit all 20 experiments as parallel array jobs on the cluster:

```bash
cd experiments
./launch_experiments.sh
```

Monitor progress:

```bash
./monitor_experiments.sh
```

See [ARRAY_JOBS.md](ARRAY_JOBS.md) for details.

## TL;DR - Local Sequential (Slower)

Run all 20 phantom reconstruction experiments sequentially on your local machine:

```bash
python experiments/scripts/run_phantom_experiments.py --batch-all
```

## Individual Experiments

```bash
# Manchester NEMA
python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm hkem
python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm dtnv
python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm tnv
python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm log_dtnv
python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm log_tnv

# Manchester NEMA (bootstrap)
python experiments/scripts/run_phantom_experiments.py --phantom manc_short --algorithm hkem
python experiments/scripts/run_phantom_experiments.py --phantom manc_short --algorithm dtnv
python experiments/scripts/run_phantom_experiments.py --phantom manc_short --algorithm tnv
python experiments/scripts/run_phantom_experiments.py --phantom manc_short --algorithm log_dtnv
python experiments/scripts/run_phantom_experiments.py --phantom manc_short --algorithm log_tnv

# Anthropomorphic
python experiments/scripts/run_phantom_experiments.py --phantom anthro --algorithm hkem
python experiments/scripts/run_phantom_experiments.py --phantom anthro --algorithm dtnv
python experiments/scripts/run_phantom_experiments.py --phantom anthro --algorithm tnv
python experiments/scripts/run_phantom_experiments.py --phantom anthro --algorithm log_dtnv
python experiments/scripts/run_phantom_experiments.py --phantom anthro --algorithm log_tnv

# NEMA
python experiments/scripts/run_phantom_experiments.py --phantom nema --algorithm hkem
python experiments/scripts/run_phantom_experiments.py --phantom nema --algorithm dtnv
python experiments/scripts/run_phantom_experiments.py --phantom nema --algorithm tnv
python experiments/scripts/run_phantom_experiments.py --phantom nema --algorithm log_dtnv
python experiments/scripts/run_phantom_experiments.py --phantom nema --algorithm log_tnv
```

## Preview Before Running

```bash
# Dry-run to see configs without executing
python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm dtnv --dry-run
python experiments/scripts/run_phantom_experiments.py --batch-all --dry-run
```

## Results Location

```
output/
├── manc/{hkem,dtnv,tnv,log_dtnv,log_tnv}/
├── manc_short/{hkem,dtnv,tnv,log_dtnv,log_tnv}/
├── anthro/{hkem,dtnv,tnv,log_dtnv,log_tnv}/
└── nema/{hkem,dtnv,tnv,log_dtnv,log_tnv}/
```

## Algorithm Comparison

| Algorithm | Type | CT Guidance | Domain | Modalities |
|-----------|------|-------------|--------|------------|
| HKEM | Sequential | Kernel + Emission | Standard | SPECT → PET |
| dTNV | Joint | Directional gradient | Standard | PET + SPECT |
| TNV | Joint | None (isotropic) | Standard | PET + SPECT |
| Log-dTNV | Joint | Directional gradient | Log | PET + SPECT |
| Log-TNV | Joint | None (isotropic) | Log | PET + SPECT |

## Key Differences

**dTNV vs TNV**:
- **dTNV**: `directional_tnv: true` - Uses CT anatomical edges to guide regularization
- **TNV**: `directional_tnv: false` - Pure isotropic vectorial TV, no anatomical guidance

**Log variants vs Standard**:
- **Log-domain**: `use_log_tnv: true` - Log transformation for better dynamic range handling
- **Standard**: `use_log_tnv: false` - Reconstruction in standard intensity domain

All dTNV/TNV variants use the same script ([run_dtnv_1bpos.py](../scripts/run_dtnv_1bpos.py)) with different configs.

## Parameter Overrides

```bash
# Change regularization strength
python experiments/scripts/run_phantom_experiments.py --phantom manc --algorithm dtnv \
    --override alpha=0.05 --override beta=0.05

# Reduce epochs for testing
python experiments/scripts/run_phantom_experiments.py --phantom anthro --algorithm tnv \
    --override num_epochs=10
```

## Need Help?

See full documentation: [README.md](README.md)
