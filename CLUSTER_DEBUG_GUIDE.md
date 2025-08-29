# Running Debug Test on Cluster

Simple guide for running the function debug test as a single cluster job.

## Quick Start

1. **Submit the job**:
   ```bash
   ./submit_debug_test.sh
   ```

2. **Monitor the job**:
   ```bash
   # Check job status
   qstat -u $USER
   
   # Get detailed job info  
   qstat -j <job_id>
   ```

3. **Check results** (after completion):
   ```bash
   # Check if job completed successfully
   cat output/debug_test/job_completion.txt
   
   # View main analysis
   cat output/debug_test/function_summary.txt
   
   # View all generated files
   ls -la output/debug_test/
   ```

## Output Structure

```
output/debug_test/
├── logs/
│   ├── debug_test.out           # SGE stdout log
│   ├── debug_test.err           # SGE stderr log  
│   └── debug_execution.log      # Script execution log
├── function_summary.txt         # Main debugging report
├── test_image_*.png             # Test image visualizations
├── gradient_*.png               # Gradient visualizations
├── preconditioner_*.png         # Preconditioner visualizations
├── test_image_*.hv              # Test images (STIR format)
├── initial_image_*.hv           # Initial images from data
├── s_inv_*.hv                   # Sensitivity images
├── kappa_sq_*.hv               # Kappa squared images
├── args.csv                     # Job parameters
└── job_completion.txt           # Job status and timing
```

## Key Files to Check

- **`function_summary.txt`**: Main debugging analysis with function values, gradient norms, and warnings for numerical issues
- **`logs/debug_execution.log`**: Detailed execution log from the debug script
- **`job_completion.txt`**: Job status (completed/failed) and return code

## Resource Usage

- **Runtime**: 2 hours (should complete much faster)
- **Memory**: 30GB
- **Cores**: 1
- **GPU**: Required (for acquisition models)

## Troubleshooting

### Job Won't Start
```bash
# Check cluster status
qstat -g c

# Check queue availability
qstat -q

# Check resource availability
qhost
```

### Job Failed
```bash
# Check SGE logs
cat output/debug_test/logs/debug_test.err

# Check execution log
cat output/debug_test/logs/debug_execution.log

# Check job completion status
cat output/debug_test/job_completion.txt
```

### Common Issues

1. **Config file not found**: Make sure `configs/config_test_debug.yaml` exists
2. **Data path issues**: Check data paths in the config file match cluster paths
3. **Out of memory**: Increase memory in `submit_debug_test.sh`: `SGE_MEMORY="40G"`
4. **GPU issues**: Check if GPUs are available with `nvidia-smi`

## Manual Submission (Alternative)

If the submission script doesn't work, submit manually:

```bash
qsub \
  -l h_rt="2:00:00" \
  -l tmem="30G" \
  -l gpu=true \
  -N "setr_debug_test" \
  -o "output/debug_test/logs/debug_test.out" \
  -e "output/debug_test/logs/debug_test.err" \
  scripts/run_debug_cluster.qsub.sh
```

## What the Job Does

1. **Environment Setup**: Activates SIRF virtual environment and sets up paths
2. **Data Loading**: Loads your actual PET/SPECT data using same paths as reconstruction
3. **Function Setup**: Creates data fidelity + prior + preconditioner functions identically to 2bpos
4. **Testing**: Evaluates functions on:
   - Initial images from data loading
   - Scaled ellipsoid test phantoms  
   - Simple uniform test images
5. **Analysis**: Calculates function values, gradients, preconditioners for each test
6. **Visualization**: Generates PNG images showing coronal slices of gradients and preconditioners
7. **Report**: Creates summary with numerical analysis and warnings

## Comparing with Local Results

After running on both local and cluster:

1. Compare function values in `function_summary.txt`
2. Look for differences in gradient magnitudes  
3. Check for NaN/infinite values that appear only on cluster
4. Compare visualization PNGs side by side
5. Look for environment-specific numerical issues