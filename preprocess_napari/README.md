# QPTIFF Manual Annotation Tool

This tool is used for manual annotation of QPTIFF files in napari and saving annotation results.

This module is used in local MacBook and upload the annotation files to the remote server.

## Features

- **Batch Processing**: Process QPTIFF files in the list sequentially
- **Auto Save**: Automatically save annotations in JSON format after completion
- **Resume Support**: Continue unfinished annotation work
- **View Annotations**: Reload and view existing annotations
- **Interactive Interface**: User-friendly command-line interface
- **Detailed Logging**: Automatically log the entire annotation process to log files
- **Area Calculation**: Automatically calculate area for each ROI (pixels and mm²)
- **Thumbnail Generation**: Automatically generate annotation overview thumbnails
- **Thumbnail Regeneration**: Regenerate thumbnails from existing annotations (fix coordinate issues, etc.)

## Usage

### 1. Run the Script Example
```bash
cd /Users/kuangda/Developer/1-projects/4-codex-analysis/0-phenocycler-penntmc-pipeline/preprocess_napari
conda activate napari-env
python napari_annotation.py
```

### 2. Select Operation Mode
After starting the script, a menu will be displayed:
```
QPTIFF File Annotation Tool
Please select operation:
1. Start annotating all files
2. View existing annotations for a file
3. Continue unfinished annotations
4. View annotation thumbnails
5. Regenerate thumbnail from existing annotations
```

### 3. Annotation Workflow
- Select "1" to start annotating all files
- For each file, napari will open displaying the image
- Use napari's polygon tool to draw tissue regions
- Close the napari window after completing annotations
- Annotations are automatically saved, then proceed to the next file

### 4. Regenerate Thumbnails
- Select "5" to regenerate thumbnails from existing annotations
- Option to select individual files or batch regenerate all thumbnails
- Useful for fixing coordinate errors or updating thumbnail styles

## Annotation Guide

### Annotating in napari:
1. Images are displayed in grayscale mode. If the DAPI channel dynamic range is too small,
2. Select the shapes layer (`tissue_masks`) in the left toolbar
3. Ensure the shape type is set to "polygon"
4. Click on image boundaries to draw polygons to delineate tissue regions
5. Multiple regions can be drawn
6. Close the napari window when finished to save and continue

### Keyboard Shortcuts:
- `Shift + Mouse Click`: Add polygon vertices
- `Esc`: Complete current polygon
- `Delete`: Delete selected shape

## Output Files

Annotation results are saved in the same directory as the original image:
- `{filename}_annotations.json` - Annotation data
- `{filename}_annotation_overview.png` - Thumbnail preview
- `annotation_log_YYYYMMDD_HHMMSS.txt` - Operation log

### Annotation JSON File Contains:
- Source file path
- Number of annotations
- Polygon coordinate points
- Thumbnail filename
- Annotation timestamp

### JSON File Structure Example:
```json
{
  "source_file": "/path/to/img0003.qptiff",
  "file_name": "img0003", 
  "num_shapes": 2,
  "shapes": [
    [[x1, y1], [x2, y2], ...],
    [[x1, y1], [x2, y2], ...]
  ],
  "thumbnail_file": "img0003_annotation_overview.png",
  "annotation_timestamp": "2024-01-15T14:30:25.123456"
}
```

### Thumbnail Features:
- Automatically scaled to appropriate size (maximum 800 pixels)
- Red borders show annotated regions
- Yellow labels mark ROI numbers
- Display original image dimensions and scaling ratio information

## Important Notes

1. **File Checking**: Script checks if files exist and automatically skips non-existent files
2. **Duplicate Annotations**: If a file already has annotations, it will ask whether to re-annotate
3. **Resume from Interruption**: Can be interrupted at any time, use option 3 to continue unfinished annotations
4. **Memory Usage**: Large files may require more memory, recommend processing individually
5. **Logging**: All operations are logged to log files for tracking and debugging
6. **Area Calculation**: Assumes pixel size of 0.325μm, modify code if adjustment needed
7. **Thumbnails**: Automatically generate PNG format thumbnails, can be quickly viewed through option 4

## Dependencies

Ensure the required Python packages are installed:
```bash
pip install napari tifffile numpy matplotlib pillow scikit-image
```

## File List

Currently files are listed in `file_paths` in `napari_annotation.py`

## Upload to Remote Server

Use the `upload_annotations.sh` script to upload annotation files to the remote server:

### Usage

```bash
# Check local files (no upload)
./upload_annotations.sh --dry-run

# Upload all annotation files
./upload_annotations.sh

# View help
./upload_annotations.sh --help
```

### Configuration

The script will upload local files:
- `{filename}_annotations.json`
- `{filename}_annotation_overview.png`

To corresponding remote locations:
- Remote server: `derek@thelio-kim`
- Remote path: `/mnt/data-3/0-projects/codex-analysis/data/FallopianTube/`

### Features

- **Auto Detection**: Only upload existing annotation files
- **Error Handling**: Connection testing and detailed error reporting
- **Progress Display**: Colored output and upload status
- **Batch Processing**: Process all 8 files at once
- **Safety Check**: Dry-run mode for pre-checking files
