

# The working environment is: napari-env
# conda activate napari-env
import napari
from tifffile import TiffFile
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from skimage.draw import polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# file_paths = [
#     "/Users/kuangda/Developer/1-projects/4-codex-analysis/data/FallopianTube/D10/img0003/img0003.qptiff",
#     "/Users/kuangda/Developer/1-projects/4-codex-analysis/data/FallopianTube/D11/img0005/img0005.qptiff",
#     "/Users/kuangda/Developer/1-projects/4-codex-analysis/data/FallopianTube/D13/img0007/img0007.qptiff",
#     "/Users/kuangda/Developer/1-projects/4-codex-analysis/data/FallopianTube/D14/img0009/img0009.qptiff",
#     "/Users/kuangda/Developer/1-projects/4-codex-analysis/data/FallopianTube/D15/img0011/img0011.qptiff",
#     "/Users/kuangda/Developer/1-projects/4-codex-analysis/data/FallopianTube/D16/img0001/img0001.qptiff",
#     "/Users/kuangda/Developer/1-projects/4-codex-analysis/data/FallopianTube/D17/img0013/img0013.qptiff",
#     "/Users/kuangda/Developer/1-projects/4-codex-analysis/data/FallopianTube/D18/img0015/img0015.qptiff"
# ]

file_paths = [
    # "/Users/kuangda/Developer/1-projects/4-codex-analysis/data/Ovary/D11_13_Scan1.er.qptiff",
    # "/Users/kuangda/Developer/1-projects/4-codex-analysis/data/Ovary/D14_15_Scan1.er.qptiff",
    "/Users/kuangda/Developer/1-projects/4-codex-analysis/data/Ovary/D17_18_Scan1.er.qptiff"
]

# Create log file
log_file = Path(__file__).parent / f"annotation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_message(message):
    """Write message to both console and log file"""
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def calculate_polygon_area(polygon_coords, image_shape):
    """Calculate area of polygon in pixels"""
    # Using shoelace formula for polygon area
    x = polygon_coords[:, 0]
    y = polygon_coords[:, 1]
    
    # Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

def generate_annotation_thumbnail(image, shapes_data, file_path, thumbnail_size=800):
    """Generate a thumbnail showing annotation overlays"""
    # Create thumbnail of original image
    height, width = image.shape
    
    # Calculate scale factor to fit within thumbnail_size
    scale_factor = min(thumbnail_size / width, thumbnail_size / height)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize image for thumbnail
    pil_image = Image.fromarray((image / image.max() * 255).astype(np.uint8))
    thumbnail = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    thumbnail_array = np.array(thumbnail)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(thumbnail_array, cmap='gray')
    
    # Draw annotations on thumbnail
    for i, shape in enumerate(shapes_data):
        # napari coordinates are (row, col) = (y, x), need to convert to (x, y) for matplotlib
        # Also scale coordinates to thumbnail size
        scaled_coords = shape * scale_factor
        
        # Convert from (y, x) to (x, y) for matplotlib
        matplotlib_coords = np.column_stack([scaled_coords[:, 1], scaled_coords[:, 0]])
        
        # Create polygon patch
        polygon_patch = patches.Polygon(matplotlib_coords, linewidth=2, 
                                      edgecolor='red', facecolor='none', alpha=0.8)
        ax.add_patch(polygon_patch)
        
        # Add ROI label at centroid
        centroid_x = np.mean(matplotlib_coords[:, 0])  # matplotlib x coordinate
        centroid_y = np.mean(matplotlib_coords[:, 1])  # matplotlib y coordinate
        ax.text(centroid_x, centroid_y, f'ROI {i+1}', 
                color='yellow', fontsize=12, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Set title and formatting
    file_name = Path(file_path).stem
    ax.set_title(f'Annotation Overview - {file_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Original size: {width} x {height} pixels', fontsize=10)
    ax.set_ylabel(f'Thumbnail scale: {scale_factor:.3f}x', fontsize=10)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save thumbnail
    thumbnail_path = Path(file_path).parent / f"{file_name}_annotation_overview.png"
    plt.tight_layout()
    plt.savefig(thumbnail_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log_message(f"Annotation thumbnail saved to: {thumbnail_path}")
    return thumbnail_path

def load_qptiff_image(file_path):
    """Load QPTIFF file and return image data"""
    with TiffFile(file_path) as tif:
        log_message(f"Series count: {len(tif.series)}")
        page = tif.series[0]
        data = page.asarray()
    
    log_message(f"Data shape: {data.shape}")
    
    # If multichannel, select first channel or do max projection
    if data.ndim == 3:
        channels, height, width = data.shape
        log_message(f"Number of channels: {channels}")
        log_message(f"Image dimensions: {width} x {height} pixels")
        log_message(f"Total image size: {width * height:,} pixels")
        
        # Option 1: single channel
        # image = data[0]
        # Option 2: max projection for annotation (recommended for tissue outline)
        image = np.max(data, axis=0)
        log_message("Using max projection across all channels for better tissue visualization")
    else:
        height, width = data.shape
        log_message(f"Image dimensions: {width} x {height} pixels")
        log_message(f"Total image size: {width * height:,} pixels")
        image = data
    
    return image

def save_annotations(shapes_data, file_path):
    """Save annotation data to the same directory as input image"""
    file_path_obj = Path(file_path)
    file_name = file_path_obj.stem
    annotation_dir = file_path_obj.parent
    
    # Save as JSON format in the same directory as the input image
    json_path = annotation_dir / f"{file_name}_annotations.json"
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_shapes = []
    for shape in shapes_data:
        serializable_shapes.append(shape.tolist())
    
    annotation_data = {
        "source_file": str(file_path),
        "file_name": file_name,
        "num_shapes": len(shapes_data),
        "shapes": serializable_shapes,
        "thumbnail_file": f"{file_name}_annotation_overview.png",
        "annotation_timestamp": datetime.now().isoformat()
    }
    
    with open(json_path, 'w') as f:
        json.dump(annotation_data, f, indent=2)
    
    log_message(f"Annotations saved to: {json_path}")
    return json_path

def process_all_files():
    """Main function to process all files"""
    log_message("="*60)
    log_message("QPTIFF ANNOTATION SESSION STARTED")
    log_message("="*60)
    
    for i, file_path in enumerate(file_paths):
        log_message(f"\n{'='*50}")
        log_message(f"Processing file {i+1}/{len(file_paths)}: {Path(file_path).name}")
        log_message(f"{'='*50}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            log_message(f"File does not exist, skipping: {file_path}")
            continue
        
        # Check if annotations already exist
        file_name = Path(file_path).stem
        annotation_dir = Path(file_path).parent
        json_path = annotation_dir / f"{file_name}_annotations.json"
        if json_path.exists():
            response = input(f"File {file_name} already has annotations, re-annotate? (y/n): ")
            if response.lower() != 'y':
                log_message("Skipping this file")
                continue
        
        # Load image
        try:
            image = load_qptiff_image(file_path)
            image_shape = image.shape
        except Exception as e:
            log_message(f"Failed to load file: {e}")
            continue
        
        # Create napari viewer
        viewer = napari.Viewer(title=f"Annotate - {Path(file_path).name}")
        viewer.add_image(image, name='tissue_scan', colormap='gray')
        viewer.add_shapes(name='tissue_masks', shape_type='polygon', 
                         edge_color='red', face_color='transparent', edge_width=2)
        
        log_message("Please annotate in napari...")
        log_message("After completing annotation, close napari window to continue to next file")
        
        # Wait for user to complete annotation and close window
        napari.run()
        
        # Get annotation data
        shapes_data = viewer.layers["tissue_masks"].data
        
        if len(shapes_data) > 0:
            # Save annotations
            saved_path = save_annotations(shapes_data, file_path)
            log_message(f"Successfully annotated {len(shapes_data)} regions")
            
            # Calculate and log ROI areas
            log_message("\nROI Area Analysis:")
            total_area = 0
            for j, shape in enumerate(shapes_data):
                area_pixels = calculate_polygon_area(shape, image_shape)
                total_area += area_pixels
                area_mm2 = area_pixels * 0.325 * 0.325 / 1000000  # Assuming 0.325 μm/pixel, convert to mm²
                log_message(f"  ROI {j+1}: {shape.shape[0]} points, Area = {area_pixels:.0f} pixels ({area_mm2:.3f} mm²)")
            
            log_message(f"Total annotated area: {total_area:.0f} pixels ({total_area * 0.325 * 0.325 / 1000000:.3f} mm²)")
            coverage_percent = (total_area / (image_shape[0] * image_shape[1])) * 100
            log_message(f"Coverage of total image: {coverage_percent:.2f}%")
            
            # Generate annotation thumbnail
            log_message("\nGenerating annotation overview thumbnail...")
            try:
                thumbnail_path = generate_annotation_thumbnail(image, shapes_data, file_path)
                log_message("Thumbnail generation completed successfully")
            except Exception as e:
                log_message(f"Failed to generate thumbnail: {e}")
        else:
            log_message("No annotations found, skipping save")
        
        # Ask whether to continue
        if i < len(file_paths) - 1:  # Not the last file
            response = input("\nContinue to next file? (y/n, press Enter to continue): ")
            if response.lower() == 'n':
                log_message("User chose to stop")
                break
    
    log_message("="*60)
    log_message("ANNOTATION SESSION COMPLETED")
    log_message("="*60)

def load_saved_annotations(file_path):
    """Load saved annotations (for viewing)"""
    file_path_obj = Path(file_path)
    file_name = file_path_obj.stem
    annotation_dir = file_path_obj.parent
    json_path = annotation_dir / f"{file_name}_annotations.json"
    
    if not json_path.exists():
        log_message(f"Annotation file not found: {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        annotation_data = json.load(f)
    
    # Convert back to numpy arrays
    shapes = [np.array(shape) for shape in annotation_data['shapes']]
    return shapes, annotation_data

def regenerate_thumbnail_from_json(file_path):
    """Regenerate thumbnail from existing JSON annotation file"""
    log_message(f"Regenerating thumbnail for: {Path(file_path).name}")
    
    # Load existing annotations
    result = load_saved_annotations(file_path)
    if not result:
        log_message("No annotation file found, cannot regenerate thumbnail")
        return False
    
    shapes, annotation_data = result
    
    # Load the image
    try:
        image = load_qptiff_image(file_path)
    except Exception as e:
        log_message(f"Failed to load image: {e}")
        return False
    
    # Generate new thumbnail with corrected coordinates
    try:
        thumbnail_path = generate_annotation_thumbnail(image, shapes, file_path)
        log_message("Thumbnail regeneration completed successfully")
        return True
    except Exception as e:
        log_message(f"Failed to regenerate thumbnail: {e}")
        return False

if __name__ == "__main__":
    # Initialize log file
    log_message("QPTIFF File Annotation Tool Started")
    log_message(f"Log file created: {log_file}")
    
    print("QPTIFF File Annotation Tool")
    print("Please select operation:")
    print("1. Start annotating all files")
    print("2. View existing annotations for a file")
    print("3. Continue unfinished annotations")
    print("4. View annotation thumbnails")
    print("5. Regenerate thumbnail from existing annotations")
    
    choice = input("Enter choice (1/2/3/4/5): ")
    log_message(f"User selected option: {choice}")
    
    if choice == "1":
        process_all_files()
    elif choice == "2":
        print("Available files:")
        for i, fp in enumerate(file_paths):
            print(f"{i+1}. {Path(fp).name}")
        
        try:
            file_idx = int(input("Select file number: ")) - 1
            if 0 <= file_idx < len(file_paths):
                file_path = file_paths[file_idx]
                log_message(f"Viewing annotations for: {Path(file_path).name}")
                result = load_saved_annotations(file_path)
                if result:
                    shapes, annotation_data = result
                    log_message(f"File: {annotation_data['file_name']}")
                    log_message(f"Number of annotations: {annotation_data['num_shapes']}")
                    
                    # Display in napari
                    image = load_qptiff_image(file_path)
                    viewer = napari.Viewer(title=f"View annotations - {Path(file_path).name}")
                    viewer.add_image(image, name='tissue_scan', colormap='gray')
                    shapes_layer = viewer.add_shapes(shapes, name='tissue_masks', 
                                                   shape_type='polygon', edge_color='red', 
                                                   face_color='transparent', edge_width=2)
                    napari.run()
            else:
                log_message("Invalid file number entered")
        except ValueError:
            log_message("Invalid input - not a number")
    elif choice == "3":
        # Find unprocessed files
        unprocessed = []
        for fp in file_paths:
            file_name = Path(fp).stem
            annotation_dir = Path(fp).parent
            json_path = annotation_dir / f"{file_name}_annotations.json"
            if not json_path.exists():
                unprocessed.append(fp)
        
        if unprocessed:
            log_message(f"Found {len(unprocessed)} unprocessed files")
            # Temporarily update file_paths to process only unfinished ones
            original_paths = file_paths.copy()
            file_paths[:] = unprocessed
            process_all_files()
            file_paths[:] = original_paths
        else:
            log_message("All files have been annotated")
    elif choice == "4":
        # View annotation thumbnails
        print("Available files with annotations:")
        annotated_files = []
        for i, fp in enumerate(file_paths):
            file_name = Path(fp).stem
            annotation_dir = Path(fp).parent
            json_path = annotation_dir / f"{file_name}_annotations.json"
            thumbnail_path = annotation_dir / f"{file_name}_annotation_overview.png"
            if json_path.exists():
                annotated_files.append((i, fp, thumbnail_path.exists()))
                status = "with thumbnail" if thumbnail_path.exists() else "no thumbnail"
                print(f"{len(annotated_files)}. {Path(fp).name} ({status})")
        
        if annotated_files:
            try:
                file_idx = int(input("Select file number to view thumbnail: ")) - 1
                if 0 <= file_idx < len(annotated_files):
                    _, file_path, has_thumbnail = annotated_files[file_idx]
                    file_name = Path(file_path).stem
                    thumbnail_path = Path(file_path).parent / f"{file_name}_annotation_overview.png"
                    
                    if has_thumbnail:
                        log_message(f"Opening thumbnail: {thumbnail_path}")
                        # Open thumbnail with default system viewer
                        import subprocess
                        import platform
                        
                        if platform.system() == "Darwin":  # macOS
                            subprocess.run(["open", str(thumbnail_path)])
                        elif platform.system() == "Windows":
                            subprocess.run(["start", str(thumbnail_path)], shell=True)
                        else:  # Linux
                            subprocess.run(["xdg-open", str(thumbnail_path)])
                    else:
                        log_message("No thumbnail found for this file. You can re-annotate to generate one.")
                else:
                    log_message("Invalid file number entered")
            except ValueError:
                log_message("Invalid input - not a number")
        else:
            log_message("No annotated files found")
    elif choice == "5":
        # Regenerate thumbnails from existing annotations
        print("Available files with annotations:")
        annotated_files = []
        for i, fp in enumerate(file_paths):
            file_name = Path(fp).stem
            annotation_dir = Path(fp).parent
            json_path = annotation_dir / f"{file_name}_annotations.json"
            if json_path.exists():
                annotated_files.append((i, fp))
                print(f"{len(annotated_files)}. {Path(fp).name}")
        
        if annotated_files:
            print(f"{len(annotated_files)+1}. Regenerate all thumbnails")
            
            try:
                choice_idx = int(input("Select file number (or choose 'all'): ")) - 1
                if choice_idx == len(annotated_files):  # "All" option
                    log_message("Regenerating all thumbnails...")
                    success_count = 0
                    for _, file_path in annotated_files:
                        if regenerate_thumbnail_from_json(file_path):
                            success_count += 1
                    log_message(f"Successfully regenerated {success_count}/{len(annotated_files)} thumbnails")
                elif 0 <= choice_idx < len(annotated_files):
                    _, file_path = annotated_files[choice_idx]
                    regenerate_thumbnail_from_json(file_path)
                else:
                    log_message("Invalid file number entered")
            except ValueError:
                log_message("Invalid input - not a number")
        else:
            log_message("No annotated files found")
    else:
        log_message("Invalid choice entered")


