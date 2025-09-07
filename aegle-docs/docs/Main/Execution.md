---
sidebar_position: 2
---

# Execution Guide

This guide provides comprehensive instructions for running the Main pipeline, from single experiment execution to large-scale batch processing.

## Prerequisites

Before running the Main pipeline, ensure:

1. **Data Preprocessing Complete**: All input data has been processed through the [Data Preprocessing](../DataPreprocess/Overview.md) pipeline
2. **Configuration Files Ready**: Experiment configurations generated using the [Experiment Configuration](../ExperimentConfiguration/intro.md) system
3. **Environment Setup**: Required dependencies installed and environment activated
4. **Directory Structure**: Proper directory structure established

## Execution Methods

### 1. Single Experiment Execution

For running individual experiments or testing configurations:

```bash
python src/main.py \
  --data_dir /path/to/data \
  --config_file /path/to/config.yaml \
  --out_dir /path/to/output \
  --log_level INFO
```

#### Parameters:
- `--data_dir`: Directory containing preprocessed input data
- `--config_file`: Path to experiment configuration YAML file
- `--out_dir`: Output directory for results
- `--log_level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

#### Example:
```bash
python src/main.py \
  --data_dir ../data \
  --config_file exps/configs/main/test/D18_Scan1_0/config.yaml \
  --out_dir out/main/test/D18_Scan1_0 \
  --log_level INFO
```

### 2. Batch Processing with Shell Scripts

For processing multiple experiments efficiently:

#### Main Analysis Pipeline

```bash
./run_main_ft.sh
```

This script processes multiple experiments sequentially with the following features:
- **Automatic Directory Management**: Creates output and log directories
- **Progress Tracking**: Shows current experiment progress
- **Comprehensive Logging**: Individual log files for each experiment
- **Timing Information**: Records execution time for each experiment

#### Configuration in `run_main_ft.sh`:

```bash
# Define the experiment set name
EXP_SET_NAME="main_ft"

# Define the base directory
ROOT_DIR="/workspaces/codex-analysis"

# Define logging level
LOG_LEVEL="INFO"

# Define experiment array
declare -a EXPERIMENTS=(
  "D11_0"
  "D11_1" 
  "D11_2"
  "D11_3"
  # Add more experiments as needed
)
```

#### Directory Structure:
```
ROOT_DIR/0-phenocycler-penntmc-pipeline/
├── exps/configs/main/main_ft/     # Configuration files
├── logs/main/main_ft/             # Log files
└── out/main/main_ft/              # Output files
```

### 3. Downstream Analysis Execution

After main analysis completion, run downstream analysis:

```bash
./run_analysis_ft.sh
```

#### Configuration in `run_analysis_ft.sh`:

```bash
# Set experiment set name
EXP_SET_NAME="test_analysis"

# Define input data directory (output from main pipeline)
DATA_DIR="${ROOT_DIR}/out/main/test0206_main"

# Define configuration directory for analysis
CONFIG_DIR="${ROOT_DIR}/exps/configs/analysis/${EXP_SET_NAME}"
```

## Execution Workflow

### Complete Pipeline Execution

1. **Prepare Configurations**:
   ```bash
   cd exps/
   python config_generator.py
   ```

2. **Run Main Analysis**:
   ```bash
   ./run_main_ft.sh
   ```

3. **Monitor Progress**:
   ```bash
   # Check logs in real-time
   tail -f logs/main/main_ft/D11_0.log
   
   # Check overall progress
   ls -la out/main/main_ft/
   ```

4. **Run Downstream Analysis**:
   ```bash
   ./run_analysis_ft.sh
   ```

## Monitoring and Logging

### Log File Structure

Each experiment generates detailed logs:

```
logs/main/main_ft/
├── D11_0.log          # Individual experiment logs
├── D11_1.log
└── ...
```

### Log Content Example:

```
Current time: Mon Dec  9 10:30:00 UTC 2024
Running experiment D11_0 (1 of 4)
Log level: INFO
Starting pipeline with log level: INFO
Data directory: /workspaces/codex-analysis/data
Output directory: /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/main/main_ft/D11_0
Running the CODEX image analysis pipeline.
Pipeline execution completed in 1245.67 seconds.
Experiment D11_0 completed.
```

### Real-time Monitoring

```bash
# Monitor current experiment
tail -f logs/main/main_ft/D11_0.log

# Check system resources
htop

# Monitor disk usage
df -h

# Check memory usage
free -h
```

## Performance Optimization

### Memory Management

1. **Adjust Patch Size**:
   ```yaml
   patching:
     patch_size: 1000  # Reduce if memory issues occur
   ```

2. **Limit Parallel Processes**:
   ```yaml
   processing:
     max_workers: 4  # Adjust based on available RAM
   ```

3. **Monitor Memory Usage**:
   ```bash
   # Add memory monitoring to scripts
   python debug_memory.py
   ```

### Execution Speed

1. **Optimize Logging Level**:
   ```bash
   LOG_LEVEL="WARNING"  # Reduce logging overhead
   ```

2. **Parallel Execution** (Advanced):
   ```bash
   # Modify script for background execution
   for EXP_ID in "${EXPERIMENTS[@]}"; do
     bash ${RUN_FILE} "$EXP_SET_NAME" "$EXP_ID" ... &
   done
   wait  # Wait for all to complete
   ```

3. **Resource Allocation**:
   ```bash
   # Set CPU affinity for large experiments
   taskset -c 0-7 python src/main.py ...
   ```

## Common Execution Patterns

### 1. Development and Testing

```bash
# Test single experiment with debug logging
python src/main.py \
  --data_dir ../data \
  --config_file exps/configs/main/test/test_exp/config.yaml \
  --out_dir out/test \
  --log_level DEBUG
```

### 2. Production Batch Processing

```bash
# Run full experiment set
./run_main_ft.sh > batch_execution.log 2>&1 &

# Monitor progress
tail -f batch_execution.log
```

### 3. Resume Interrupted Processing

```bash
# Check completed experiments
ls out/main/main_ft/

# Edit EXPERIMENTS array to exclude completed ones
vim run_main_ft.sh

# Resume processing
./run_main_ft.sh
```

## Error Handling

### Common Issues and Solutions

1. **Memory Errors**:
   ```bash
   # Reduce patch size in configuration
   # Increase system swap space
   # Process fewer experiments simultaneously
   ```

2. **Configuration Errors**:
   ```bash
   # Validate configuration
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

3. **File Permission Issues**:
   ```bash
   # Fix permissions
   chmod +x run_main_ft.sh
   chmod -R 755 out/
   ```

4. **Disk Space Issues**:
   ```bash
   # Check disk usage
   df -h
   
   # Clean up temporary files
   find out/ -name "*.tmp" -delete
   ```

## Best Practices

### Before Execution

1. **Validate Configurations**: Test with a small experiment first
2. **Check Resources**: Ensure sufficient memory and disk space
3. **Backup Important Data**: Create backups of critical configurations
4. **Plan Execution Time**: Estimate total execution time based on data size

### During Execution

1. **Monitor Resources**: Keep an eye on system resources
2. **Check Logs Regularly**: Watch for errors or warnings
3. **Document Issues**: Keep notes of any problems encountered
4. **Backup Intermediate Results**: Save important intermediate outputs

### After Execution

1. **Validate Outputs**: Check that all expected outputs are generated
2. **Review Logs**: Look for any warnings or performance issues
3. **Archive Results**: Move completed results to long-term storage
4. **Clean Up**: Remove unnecessary temporary files

## Integration with Other Systems

### Slurm/PBS Integration

For HPC environments:

```bash
#!/bin/bash
#SBATCH --job-name=aegle_main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

module load python/3.8
source activate aegle_env

python src/main.py \
  --data_dir $DATA_DIR \
  --config_file $CONFIG_FILE \
  --out_dir $OUT_DIR \
  --log_level INFO
```

### Docker Integration

```bash
# Run in Docker container
docker run -v /data:/data -v /output:/output \
  aegle:latest python src/main.py \
  --data_dir /data \
  --config_file /config.yaml \
  --out_dir /output
```

## Next Steps

After successful execution:

1. **Review Results**: Check the [Outputs documentation](Outputs.md)
2. **Run Analysis**: Proceed to downstream analysis
3. **Troubleshoot Issues**: Consult [Troubleshooting guide](Troubleshooting.md)
4. **Optimize Performance**: Adjust configurations based on results
