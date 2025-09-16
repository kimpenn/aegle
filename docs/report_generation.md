# Pipeline Report Generation

The PhenoCycler pipeline can automatically generate comprehensive analysis reports at the end of each run. These reports provide a complete summary of the analysis results, quality metrics, and visualizations.

## Features

### 1. **Executive Summary**
- Experiment ID and metadata
- Key metrics at a glance
- Total cell counts
- Quality scores
- Processing status

### 2. **Input Data Summary**
- Image information (dimensions, channels, resolution)
- Antibody panel details
- Configuration parameters used

### 3. **Quality Assessment**
- Patch-level quality metrics
- Channel intensity statistics
- Signal-to-noise ratios
- Data coverage analysis

### 4. **Segmentation Results**
- Cell detection statistics
- Morphology distributions
- Nucleus-cell matching rates
- Error detection summaries
- Spatial distribution analysis

### 5. **Expression Analysis**
- Marker expression statistics
- Positive cell percentages
- Expression heatmaps
- Co-expression patterns

### 6. **Visualizations**
All key visualizations are embedded in the report:
- Quality dashboards
- Segmentation overlays
- Morphology distributions
- Expression heatmaps
- Spatial analysis plots

### 7. **Performance Metrics**
- Pipeline execution time
- Memory usage statistics
- Processing bottlenecks
- Resource utilization

## Configuration

Enable report generation in your pipeline configuration:

```yaml
# Report generation settings
report:
  generate_report: true     # Enable/disable report generation
  report_format: html       # Output format (currently only HTML)
```

## Usage

### Automatic Generation
Reports are automatically generated at the end of pipeline runs when enabled in the configuration.

### Manual Generation
Generate a report for existing pipeline results:

```bash
python scripts/generate_report.py /path/to/pipeline/output
```

Options:
- `--format html|pdf`: Choose output format (default: html)
- `--output <path>`: Custom output path for report
- `--verbose`: Enable verbose logging

### Example:
```bash
# Generate report for tile_0 results
python scripts/generate_report.py \
    /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/main/test/ft_small_tiles/tile_0

# The report will be saved as:
# pipeline_report.html in the output directory
```

## Report Contents

### Standard Sections

1. **Header Information**
   - Report generation timestamp
   - Pipeline version
   - Experiment ID

2. **Data Quality Dashboard**
   - Patch statistics
   - Channel quality metrics
   - Coverage maps

3. **Segmentation Summary**
   - Total cells detected
   - Cell size statistics
   - Morphology distributions
   - Quality scores

4. **Expression Analysis**
   - Per-marker statistics
   - Expression distributions
   - Positive cell counts

5. **Warnings and Issues**
   - Low quality regions
   - Processing errors
   - Parameter recommendations

### Customization

The report template can be customized by modifying:
- `aegle/report_generator.py`: Report logic
- HTML template within the generator
- CSS styles for appearance

## Output

Reports are saved in the pipeline output directory:
```
output_dir/
├── pipeline_report.html     # Main report file
├── pipeline_report_data/    # Supporting data (future)
│   ├── figures/            # High-res figures
│   └── tables/             # Data tables
```

## Future Enhancements

Planned features for report generation:

1. **PDF Export**
   - High-quality PDF reports
   - Print-friendly formatting
   - Embedded vector graphics

2. **Interactive Reports**
   - Zoomable plots
   - Sortable tables
   - Collapsible sections

3. **Batch Reports**
   - Multi-sample comparisons
   - Aggregate statistics
   - Trend analysis

4. **Custom Templates**
   - User-defined report templates
   - Branding options
   - Section selection

5. **Data Export**
   - Downloadable data tables
   - Figure export options
   - Raw data links

## Troubleshooting

### Common Issues

1. **Missing Data**
   - Ensure pipeline completed successfully
   - Check that all output files exist
   - Verify configuration was saved

2. **Visualization Errors**
   - Check matplotlib backend
   - Ensure sufficient memory
   - Verify image data integrity

3. **Large Reports**
   - For many patches, consider sampling
   - Reduce figure resolution
   - Enable compression

### Debug Mode

Enable verbose logging for troubleshooting:
```bash
python scripts/generate_report.py --verbose /path/to/output
```

## Best Practices

1. **Regular Generation**
   - Generate reports for all runs
   - Archive reports with data
   - Use for quality control

2. **Report Review**
   - Check quality metrics
   - Identify processing issues
   - Verify expected results

3. **Sharing Results**
   - HTML reports are self-contained
   - Can be opened in any browser
   - Easy to share via email/web
