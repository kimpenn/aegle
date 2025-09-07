---
sidebar_position: 6
---

# Troubleshooting

This guide helps you diagnose and resolve common issues encountered when running the Main pipeline.

## Quick Diagnostic Checklist

Before diving into specific issues, run through this quick checklist:

- [ ] All prerequisites completed ([Data Preprocessing](../DataPreprocess/Overview.md))
- [ ] Configuration files valid and accessible
- [ ] Sufficient disk space available
- [ ] Required memory available
- [ ] Input data files exist and are readable
- [ ] Output directories have write permissions

## Common Issues and Solutions

### Memory-Related Issues

#### Out of Memory Errors

**Symptoms:**
```
MemoryError: Unable to allocate array
RuntimeError: CUDA out of memory
Process killed (signal 9)
```

**Solutions:**

1. **Reduce Patch Size:**
   ```yaml
   patching:
     patch_height: 1000  # Reduce from default
     patch_width: 1000
   ```

2. **Disable Memory-Intensive Features:**
   ```yaml
   visualization:
     visualize_patches: false
     save_all_channel_patches: false
   
   segmentation:
     segmentation_analysis: false
   ```

3. **Increase System Swap:**
   ```bash
   # Check current swap
   free -h
   
   # Add temporary swap file
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

4. **Process Smaller Batches:**
   ```bash
   # Edit experiment array to process fewer at once
   declare -a EXPERIMENTS=(
     "D11_0"  # Process one at a time
   )
   ```

#### Memory Leaks

**Symptoms:**
- Memory usage continuously increases
- System becomes progressively slower
- Later experiments fail with memory errors

**Solutions:**

1. **Monitor Memory Usage:**
   ```bash
   # Run memory monitoring during execution
   python debug_memory.py &
   
   # Watch memory in real-time
   watch -n 1 'free -h && ps aux --sort=-%mem | head -10'
   ```

2. **Restart Between Experiments:**
   ```bash
   # Modify batch script to restart Python process
   for EXP_ID in "${EXPERIMENTS[@]}"; do
     python src/main.py ... # Run single experiment
     sleep 5  # Allow cleanup
   done
   ```

### Configuration Issues

#### Invalid Configuration Files

**Symptoms:**
```
yaml.scanner.ScannerError: while scanning for the next token
KeyError: 'required_parameter'
FileNotFoundError: config.yaml not found
```

**Solutions:**

1. **Validate YAML Syntax:**
   ```bash
   # Check YAML syntax
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   
   # Use online YAML validator
   # Copy-paste config to yamllint.com
   ```

2. **Check Required Parameters:**
   ```python
   # Validate required fields
   import yaml
   config = yaml.safe_load(open('config.yaml'))
   
   required = ['exp_id', 'data', 'channels', 'segmentation']
   for field in required:
       assert field in config, f"Missing required field: {field}"
   ```

3. **Regenerate Configuration:**
   ```bash
   cd exps/
   python config_generator.py  # Regenerate from template
   ```

#### File Path Issues

**Symptoms:**
```
FileNotFoundError: No such file or directory: 'path/to/file'
PermissionError: [Errno 13] Permission denied
```

**Solutions:**

1. **Use Absolute Paths:**
   ```yaml
   data:
     file_name: /full/path/to/image.qptiff
     antibodies_file: /full/path/to/antibodies.tsv
   ```

2. **Check File Permissions:**
   ```bash
   # Check file exists and is readable
   ls -la /path/to/file
   
   # Fix permissions if needed
   chmod 644 /path/to/file
   ```

3. **Verify Directory Structure:**
   ```bash
   # Check expected directory structure
   find data/ -name "*.qptiff" -o -name "*.tsv"
   ```

### Segmentation Issues

#### Segmentation Model Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'deepcell'
RuntimeError: Could not load segmentation model
CUDA device-side assert triggered
```

**Solutions:**

1. **Check Model Path:**
   ```bash
   # Verify model directory exists
   ls -la /path/to/segmentation/model
   
   # Check required files
   find /path/to/model -name "*.h5" -o -name "*.pb"
   ```

2. **Install Required Dependencies:**
   ```bash
   # Install segmentation dependencies
   pip install deepcell tensorflow
   
   # For GPU support
   pip install tensorflow-gpu
   ```

3. **Test Model Loading:**
   ```python
   # Test model loading separately
   from deepcell.applications import MultiplexSegmentation
   model = MultiplexSegmentation('/path/to/model')
   ```

#### Poor Segmentation Results

**Symptoms:**
- Very few cells detected
- Over-segmentation (too many small objects)
- Under-segmentation (cells merged together)

**Solutions:**

1. **Check Channel Configuration:**
   ```yaml
   channels:
     nuclear_channel: DAPI  # Must match antibodies file exactly
     wholecell_channel: Pan-Cytokeratin  # Strong membrane marker
   ```

2. **Adjust Image Quality:**
   ```yaml
   patch_qc:
     non_zero_perc_threshold: 0.1  # Increase for better tissue
     mean_intensity_threshold: 5   # Increase for brighter images
   ```

3. **Enable Segmentation Analysis:**
   ```yaml
   segmentation:
     segmentation_analysis: true  # Get detailed quality metrics
   ```

### Performance Issues

#### Slow Execution

**Symptoms:**
- Processing takes much longer than expected
- High CPU usage but low progress
- Disk I/O bottlenecks

**Solutions:**

1. **Optimize Logging:**
   ```bash
   # Reduce logging verbosity
   LOG_LEVEL="WARNING"  # In run script
   ```

2. **Disable Unnecessary Features:**
   ```yaml
   data:
     generate_channel_stats: false
   
   visualization:
     visualize_whole_sample: false
     enhance_contrast: false
   ```

3. **Optimize Disk I/O:**
   ```bash
   # Use faster storage for temporary files
   export TMPDIR=/fast/storage/tmp
   
   # Monitor disk usage
   iotop -ao
   ```

#### High Disk Usage

**Symptoms:**
```
OSError: [Errno 28] No space left on device
```

**Solutions:**

1. **Clean Up Temporary Files:**
   ```bash
   # Remove old outputs
   find out/ -name "*.tmp" -delete
   find out/ -name "*.log" -mtime +7 -delete
   
   # Clean up disrupted patches
   find out/ -name "*disrupted*" -delete
   ```

2. **Reduce Output Size:**
   ```yaml
   visualization:
     save_all_channel_patches: false
   
   segmentation:
     save_segmentation_images: false
   
   testing:
     data_disruption:
       save_disrupted_patches: false
   ```

3. **Monitor Disk Usage:**
   ```bash
   # Check disk usage regularly
   df -h
   du -sh out/*/
   ```

### Data Issues

#### Corrupted Input Files

**Symptoms:**
```
RuntimeError: Could not read TIFF file
ValueError: Invalid image dimensions
PIL.UnidentifiedImageError: cannot identify image file
```

**Solutions:**

1. **Validate Input Files:**
   ```bash
   # Check file integrity
   file /path/to/image.qptiff
   
   # Test with ImageJ/Fiji
   # Open file in ImageJ to verify readability
   ```

2. **Check File Format:**
   ```python
   # Test file reading
   from tifffile import TiffFile
   with TiffFile('image.qptiff') as tif:
       print(tif.pages[0].shape)
   ```

3. **Verify Antibodies File:**
   ```bash
   # Check TSV format
   head -5 antibodies.tsv
   
   # Validate structure
   python -c "import pandas as pd; print(pd.read_csv('antibodies.tsv', sep='\t'))"
   ```

#### Channel Mismatch

**Symptoms:**
```
KeyError: 'DAPI' not found in antibodies
ValueError: Channel index out of range
```

**Solutions:**

1. **Check Channel Names:**
   ```bash
   # List available channels
   cat antibodies.tsv | cut -f2
   
   # Compare with configuration
   grep -A5 "channels:" config.yaml
   ```

2. **Update Configuration:**
   ```yaml
   channels:
     nuclear_channel: "DAPI"  # Use exact name from TSV
     wholecell_channel: "Pan-Cytokeratin"
   ```

## Debugging Techniques

### Enable Debug Logging

```bash
# Run with maximum logging
python src/main.py \
  --config_file config.yaml \
  --log_level DEBUG \
  --out_dir debug_output
```

### Isolate Issues

1. **Test Single Patch:**
   ```yaml
   patching:
     split_mode: patches
     patch_height: 500
     patch_width: 500
   ```

2. **Disable Complex Features:**
   ```yaml
   visualization:
     visualize_whole_sample: false
     visualize_patches: false
   
   segmentation:
     segmentation_analysis: false
   
   testing:
     data_disruption:
       type: null
   ```

3. **Run Components Separately:**
   ```python
   # Test image loading only
   from aegle.codex_image import CodexImage
   codex_image = CodexImage(config, args)
   ```

### Memory Profiling

```bash
# Profile memory usage
python -m memory_profiler src/main.py --config_file config.yaml

# Use pympler for detailed analysis
pip install pympler
```

### Performance Profiling

```bash
# Profile execution time
python -m cProfile -o profile.stats src/main.py --config_file config.yaml

# Analyze profile
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

## Getting Help

### Log Analysis

When reporting issues, include:

1. **Configuration File:**
   ```bash
   cat config.yaml
   ```

2. **Error Messages:**
   ```bash
   tail -50 logs/main/experiment_name.log
   ```

3. **System Information:**
   ```bash
   # System specs
   free -h
   df -h
   python --version
   pip list | grep -E "(numpy|tensorflow|deepcell)"
   ```

### Useful Commands

```bash
# Check pipeline status
ps aux | grep python

# Monitor resources
htop
nvidia-smi  # For GPU systems

# Check logs in real-time
tail -f logs/main/*/D11_0.log

# Find error patterns
grep -r "ERROR\|Exception\|Failed" logs/main/
```

## Frequently Asked Questions

### Q: Pipeline runs but produces no output
**A:** Check patch quality control settings. If all patches are marked as "bad", no processing occurs. Lower the QC thresholds:

```yaml
patch_qc:
  non_zero_perc_threshold: 0.01
  mean_intensity_threshold: 0.1
```

### Q: Segmentation produces too many/too few cells
**A:** This usually indicates channel configuration issues. Verify:
1. Channel names match exactly between config and antibodies file
2. Nuclear channel shows clear nuclear staining
3. Wholecell channel shows cell boundaries

### Q: Process killed without error message
**A:** Usually indicates out-of-memory condition. Check system logs:
```bash
dmesg | grep -i "killed process"
journalctl -u your-service-name
```

### Q: Results inconsistent between runs
**A:** Enable reproducible processing:
```yaml
# Add to configuration
random_seed: 42
deterministic: true
```

### Q: Can I resume interrupted processing?
**A:** Yes, the pipeline skips completed experiments. Edit the experiment array in your run script to exclude finished experiments:

```bash
# Check completed experiments
ls out/main/experiment_set/

# Edit EXPERIMENTS array in run script
vim run_main_ft.sh
```

### Q: How to process just one patch for testing?
**A:** Use single patch mode:
```yaml
patching:
  split_mode: patches
  patch_height: 1000
  patch_width: 1000

# Then manually crop your image to test size
```

## Performance Optimization Tips

### For Large Images (&gt;50K x 50K)
- Use `patch_height: 2000, patch_width: 2000`
- Set `overlap: 0.05` (reduce overlap)
- Disable visualization features
- Process experiments sequentially, not in parallel

### For Many Small Experiments
- Use `split_mode: full_image`
- Enable parallel processing
- Use higher quality thresholds to skip empty images

### For Limited Memory Systems
- Reduce patch size to 1000x1000 or smaller
- Disable `save_all_channel_patches`
- Set `segmentation_analysis: false`
- Process one experiment at a time

This troubleshooting guide should help you resolve most common issues. For additional support, check the project's issue tracker or contact the development team.
