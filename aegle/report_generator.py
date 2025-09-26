"""
Report generator for PhenoCycler pipeline results.
Creates comprehensive HTML reports with analysis summaries and visualizations.
"""

import os
import json
import datetime
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from jinja2 import Template
import pickle
import base64
from io import BytesIO
from collections import OrderedDict
try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

logger = logging.getLogger(__name__)


class PipelineReportGenerator:
    """Generate comprehensive analysis reports for pipeline runs."""
    
    def __init__(self, output_dir: str, config: Dict, codex_patches=None):
        """
        Initialize report generator.
        
        Args:
            output_dir: Pipeline output directory
            config: Pipeline configuration dictionary
            codex_patches: Optional CodexPatches object with results
        """
        self.output_dir = Path(output_dir)
        self.config = config
        self.codex_patches = codex_patches
        self.report_data = {}
        self.figures = {}
        self.patch_quality_df = None
        
    def generate_report(self, report_path: Optional[str] = None) -> str:
        """
        Generate complete pipeline report.
        
        Args:
            report_path: Optional custom path for report
            
        Returns:
            Path to generated report
        """
        if report_path is None:
            report_path = self.output_dir / "pipeline_report.html"
            
        logger.info("Generating pipeline report...")

        split_mode = self.config.get('patching', {}).get('split_mode', 'patches')
        self.report_data['split_mode'] = split_mode

        
        # Collect all data
        self._collect_metadata()
        self._collect_input_info()
        self._collect_sample_preview()
        self._collect_segmentation_visualization()
        self._collect_segmentation_channel_info()
        # self._collect_quality_metrics()
        self._collect_segmentation_stats()
        self._collect_mask_repair_stats()
        self._collect_expression_stats()
        self._collect_performance_metrics()

        if split_mode == "patches":
            self._collect_patch_mode_quality_data()
            self._generate_patch_mode_figures()
            html_content = self._render_patch_mode_html_report()
        else:
            self._generate_summary_figures()
            html_content = self._render_html_report()

        # HTML content rendered above based on split mode
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"Report saved to: {report_path}")
        return str(report_path)
    
    def _collect_metadata(self):
        """Collect basic metadata about the run."""
        self.report_data['metadata'] = {
            'report_generated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'experiment_id': self.config.get('exp_id', 'Unknown'),
            'output_directory': str(self.output_dir),
            'pipeline_version': '0.1',  # TODO: Get from version file
        }
        
    def _collect_input_info(self):
        """Collect information about input data."""
        data_config = self.config.get('data', {})
        
        self.report_data['input_info'] = {
            'image_file': data_config.get('file_name', 'Unknown'),
            'antibodies_file': data_config.get('antibodies_file', 'Unknown'),
            'microns_per_pixel': data_config.get('image_mpp', 0.5),
        }
        
        # Try to get image dimensions directly from image file
        image_file = data_config.get('file_name', '')
        if image_file and os.path.exists(image_file):
            try:
                image_info = self._get_image_dimensions(image_file)
                if image_info:
                    self.report_data['input_info'].update(image_info)
                    logger.info(f"Got image dimensions from file: {image_info}")
            except Exception as e:
                logger.warning(f"Failed to get image dimensions from file: {e}")
        
        # Get image dimensions from CodexPatches if available (this may override file info)
        if self.codex_patches:
            codex_info = {
                'image_height': getattr(self.codex_patches, 'img_height', self.report_data['input_info'].get('image_height', 'Unknown')),
                'image_width': getattr(self.codex_patches, 'img_width', self.report_data['input_info'].get('image_width', 'Unknown')),
                'n_channels': getattr(self.codex_patches, 'n_channels', self.report_data['input_info'].get('n_channels', 'Unknown')),
            }
            self.report_data['input_info'].update(codex_info)
            
            # Also try to get from tif_image_details
            if hasattr(self.codex_patches, 'tif_image_details'):
                details = self.codex_patches.tif_image_details
                if isinstance(details, dict):
                    if 'ImageLength' in details:
                        self.report_data['input_info']['image_height'] = details['ImageLength']
                    if 'ImageWidth' in details:
                        self.report_data['input_info']['image_width'] = details['ImageWidth']
        else:
            # If no CodexPatches, set defaults if not already set
            if 'image_height' not in self.report_data['input_info']:
                self.report_data['input_info']['image_height'] = 'Unknown'
            if 'image_width' not in self.report_data['input_info']:
                self.report_data['input_info']['image_width'] = 'Unknown'
            if 'n_channels' not in self.report_data['input_info']:
                self.report_data['input_info']['n_channels'] = 'Unknown'
            
        # Load antibody information
        antibodies_path = self.output_dir / "cell_profiling" / "patches_metadata.csv"
        if antibodies_path.exists():
            # Get from saved metadata
            pass
        elif self.codex_patches and hasattr(self.codex_patches, 'antibody_df'):
            self.report_data['antibodies'] = self.codex_patches.antibody_df.to_dict('records')
            
    def _collect_sample_preview(self):
        """Collect sample preview images."""
        data_config = self.config.get('data', {})
        image_file = data_config.get('file_name', '')
        
        preview_images = []
        
        if image_file:
            # Convert image file path to preview image path
            # Example: /workspaces/codex-analysis/data/test/ft_small_tiles/tile_000.ome.tiff
            # -> /workspaces/codex-analysis/data/test/ft_small_tiles/tile_000_ch0.jpg
            image_path = Path(image_file)
            if image_path.exists():
                # Get the base name without extension
                base_name = image_path.stem
                if base_name.endswith('.ome'):
                    base_name = base_name[:-4]  # Remove .ome suffix
                
                # Look for preview images in the same directory
                preview_dir = image_path.parent
                
                # Look for channel preview images (ch0, ch1, etc.)
                for i in range(10):  # Check first 10 channels
                    preview_file = preview_dir / f"{base_name}_ch{i}.jpg"
                    if preview_file.exists():
                        img_data = self._load_preview_image(preview_file)
                        if img_data:
                            preview_images.append({
                                'channel': i,
                                'filename': preview_file.name,
                                'data': img_data,
                            })
                            logger.info(f"Found preview image: {preview_file}")
                
                # If no channel-specific images found, look for a general preview
                if not preview_images:
                    general_preview = preview_dir / f"{base_name}_preview.jpg"
                    if general_preview.exists():
                        img_data = self._load_preview_image(general_preview)
                        if img_data:
                            preview_images.append({
                                'channel': 'general',
                                'filename': general_preview.name,
                                'data': img_data,
                            })
                            logger.info(f"Found general preview image: {general_preview}")
            else:
                logger.warning(f"Image file not found: {image_file}")
        
        self.report_data['sample_preview'] = {
            'images': preview_images,
            'total_previews': len(preview_images)
        }
        
        if preview_images:
            logger.info(f"Collected {len(preview_images)} preview images")
        else:
            logger.info("No preview images found")
            
    def _load_preview_image(self, preview_file: Path) -> Optional[str]:
        """Load preview image, apply contrast enhancement, return base64 data URI."""
        try:
            if PIL_AVAILABLE:
                with Image.open(preview_file) as img:
                    # Convert to RGB to ensure consistent output
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    elif img.mode == "L":
                        # Keep grayscale but operations expect single channel
                        pass

                    # Apply auto-contrast to expand dynamic range
                    img = ImageOps.autocontrast(img, cutoff=1)

                    # For very low dynamic range grayscale, equalize as well
                    if img.mode == "L":
                        img = ImageOps.equalize(img)

                    buffer = BytesIO()
                    img.convert("RGB").save(buffer, format='JPEG', quality=90)
                    buffer.seek(0)
                    img_data = base64.b64encode(buffer.read()).decode()
                    return f"data:image/jpeg;base64,{img_data}"
            # Fallback: raw bytes without enhancement
            with open(preview_file, 'rb') as f:
                return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
        except Exception as e:
            logger.warning(f"Failed to process preview image {preview_file}: {e}")
            return None

    def _get_image_dimensions(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Get image dimensions directly from image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image dimensions or None if failed
        """
        image_path = Path(image_path)
        
        try:
            # Try with tifffile first for TIFF files
            if TIFFFILE_AVAILABLE and image_path.suffix.lower() in ['.tiff', '.tif']:
                with tifffile.TiffFile(image_path) as tif:
                    # For OME-TIFF and multi-page TIFF files
                    if hasattr(tif, 'series') and len(tif.series) > 0:
                        series = tif.series[0]
                        if len(series.shape) == 3:  # (channels, height, width) or (height, width, channels)
                            # For OME-TIFF, usually (channels, height, width)
                            n_channels, height, width = series.shape
                        elif len(series.shape) == 2:  # Single channel (height, width)
                            height, width = series.shape
                            n_channels = 1
                        else:
                            # Fallback: use first page dimensions
                            page = tif.pages[0]
                            height, width = page.shape[:2]
                            n_channels = len(tif.pages)  # Each page is a channel
                    else:
                        # Single page or no series info
                        page = tif.pages[0]
                        height, width = page.shape[:2]
                        
                        # Try to get number of channels
                        if len(page.shape) >= 3:
                            n_channels = page.shape[2]
                        elif len(tif.pages) > 1:
                            # Multi-page TIFF: assume each page is a channel
                            n_channels = len(tif.pages)
                        else:
                            n_channels = 1
                    
                    return {
                        'image_height': int(height),
                        'image_width': int(width),
                        'n_channels': int(n_channels)
                    }
            
            # Try with PIL for other formats
            elif PIL_AVAILABLE:
                with Image.open(image_path) as img:
                    width, height = img.size
                    
                    # Try to get number of channels
                    n_channels = 'Unknown'
                    if hasattr(img, 'mode'):
                        if img.mode == 'RGB':
                            n_channels = 3
                        elif img.mode == 'RGBA':
                            n_channels = 4
                        elif img.mode == 'L':
                            n_channels = 1
                        elif img.mode == 'LA':
                            n_channels = 2
                    
                    return {
                        'image_height': int(height),
                        'image_width': int(width),
                        'n_channels': n_channels
                    }
            
        except Exception as e:
            logger.warning(f"Failed to read image dimensions from {image_path}: {e}")
            return None
        
        logger.warning(f"No suitable library available to read {image_path}")
        return None
    
    def _collect_segmentation_visualization(self):
        """Collect segmentation visualization images and metadata for the report."""
        visualization_dir = self.output_dir / "visualization" / "segmentation"

        overlay_cards: List[Dict] = []
        nucleus_cards: List[Dict] = []
        wholecell_cards: List[Dict] = []
        overview_image = None
        morphology_image = None
        patch_summaries: List[Dict] = []
        large_sample_note = None
        is_large_sample = False
        total_images = 0

        if not visualization_dir.exists():
            logger.info(f"Segmentation visualization directory not found: {visualization_dir}")
            self.report_data['segmentation_visualization'] = {
                'overlay_cards': overlay_cards,
                'nucleus_cards': nucleus_cards,
                'wholecell_cards': wholecell_cards,
                'overview_image': overview_image,
                'is_large_sample': is_large_sample,
                'large_sample_note': large_sample_note,
                'selected_patch_count': 0,
                'total_images': total_images,
            }
            self.report_data['segmentation_results_details'] = {'morphology_image': morphology_image}
            return

        summary_path = visualization_dir / "segmentation_patch_summary.json"
        summary_data: Dict = {}
        if summary_path.exists():
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to load segmentation patch summary: {exc}")
                summary_data = {}
        else:
            logger.warning(f"Segmentation patch summary not found at {summary_path}")

        patches = summary_data.get('patches', []) if isinstance(summary_data.get('patches'), list) else []
        is_large_sample = bool(summary_data.get('is_large_sample'))
        selected_patch_count = summary_data.get('selected_patch_count', len(patches))
        target_patch_count = summary_data.get('target_patch_count', selected_patch_count)
        total_pixels = summary_data.get('total_pixels')

        if is_large_sample and total_pixels:
            large_sample_note = (
                f"Sample contains approximately {int(total_pixels):,} pixels. "
                f"Displaying {selected_patch_count} representative patch"
                f"{'es' if selected_patch_count != 1 else ''} (target {target_patch_count})."
            )

        def _load_image_data(path: Path) -> Optional[str]:
            try:
                with open(path, 'rb') as img_file:
                    encoded = base64.b64encode(img_file.read()).decode()
                suffix = path.suffix.lower()
                mime = 'image/png' if suffix == '.png' else 'image/jpeg'
                return f"data:{mime};base64,{encoded}"
            except FileNotFoundError:
                logger.debug(f"Visualization image missing: {path}")
            except Exception as exc:
                logger.warning(f"Failed to encode visualization image {path}: {exc}")
            return None

        def _build_image_dict(path: Path, title: str, description: str) -> Optional[Dict]:
            data = _load_image_data(path)
            if not data:
                return None
            return {
                'filename': path.name,
                'title': title,
                'description': description,
                'data': data,
            }

        def _format_metrics(entry: Dict) -> str:
            parts: List[str] = []
            source_idx = entry.get('source_patch_index')
            if source_idx is not None and source_idx != -1:
                try:
                    parts.append(f"Source segmentation patch #{int(source_idx)}")
                except (TypeError, ValueError):
                    pass
            cell_count = entry.get('cell_count')
            if cell_count is not None:
                parts.append(f"Cells: {int(cell_count)}")
            matched_fraction = entry.get('matched_fraction')
            if matched_fraction is not None:
                try:
                    parts.append(f"Match rate: {float(matched_fraction) * 100:.1f}%")
                except (TypeError, ValueError):
                    pass
            unmatched_cells = entry.get('unmatched_cells')
            unmatched_nuclei = entry.get('unmatched_nuclei')
            if (unmatched_cells or unmatched_nuclei) and (
                (unmatched_cells or 0) > 0 or (unmatched_nuclei or 0) > 0
            ):
                parts.append(
                    f"Unmatched cells/nuclei: {int(unmatched_cells or 0)}/{int(unmatched_nuclei or 0)}"
                )
            return " | ".join(parts)

        overview_filename = summary_data.get('overview_image')
        if overview_filename:
            overview_path = visualization_dir / overview_filename
            overview_image = _build_image_dict(
                overview_path,
                'Patch Selection Overview',
                'Highlighted rectangles indicate sampled patches used for visualization.',
            )
            if overview_image:
                total_images += 1

        overlay_tab_defs = [
            ('overlay_repaired', 'Repaired', None),
            ('overlay_pre_repair', 'Pre-Repair', 'Original segmentation prior to repair.'),
            ('overlay_unmatched_nucleus', 'Unmatched Nuclei', 'Highlights nuclei removed during repair.'),
            ('overlay_unmatched_cell', 'Unmatched Cells', 'Highlights cells removed during repair.'),
        ]
        nucleus_tab_defs = [
            ('nucleus_mask', 'Repaired', 'Repaired nucleus mask.'),
            ('nucleus_mask_pre_repair', 'Pre-Repair', 'Original nucleus mask.'),
        ]
        wholecell_tab_defs = [
            ('wholecell_mask', 'Repaired', 'Repaired whole-cell mask.'),
            ('wholecell_mask_pre_repair', 'Pre-Repair', 'Original whole-cell mask.'),
        ]
        for patch in patches:
            files = patch.get('files') or {}
            display_idx = patch.get('display_index', 0)
            display_name = patch.get('display_name', f"Patch {display_idx + 1}")
            reason_label = patch.get('selection_reason_label')
            metrics_summary = _format_metrics(patch)

            card_title_suffix = f" — {reason_label}" if reason_label else ''
            card_title = f"{display_name}{card_title_suffix}"

            overlay_tabs = []
            for key, tab_title, extra_desc in overlay_tab_defs:
                filename = files.get(key)
                if not filename:
                    continue
                image = _build_image_dict(
                    visualization_dir / filename,
                    tab_title,
                    extra_desc or metrics_summary or tab_title,
                )
                if not image:
                    continue
                if metrics_summary and extra_desc is None:
                    image['description'] = metrics_summary
                overlay_tabs.append(
                    {
                        'title': tab_title,
                        'image': image,
                    }
                )
                total_images += 1

            if overlay_tabs:
                overlay_cards.append(
                    {
                        'card_id': f"patch_{display_idx}_overlay",
                        'card_title': f"{card_title} — Segmentation Overlay",
                        'tabs': overlay_tabs,
                    }
                )

            nucleus_tabs = []
            for key, tab_title, desc in nucleus_tab_defs:
                filename = files.get(key)
                if not filename:
                    continue
                image = _build_image_dict(
                    visualization_dir / filename,
                    tab_title,
                    desc,
                )
                if not image:
                    continue
                nucleus_tabs.append({'title': tab_title, 'image': image})
                total_images += 1

            if nucleus_tabs:
                nucleus_cards.append(
                    {
                        'card_id': f"patch_{display_idx}_nucleus",
                        'card_title': f"{card_title} — Nucleus Mask",
                        'tabs': nucleus_tabs,
                    }
                )

            wholecell_tabs = []
            for key, tab_title, desc in wholecell_tab_defs:
                filename = files.get(key)
                if not filename:
                    continue
                image = _build_image_dict(
                    visualization_dir / filename,
                    tab_title,
                    desc,
                )
                if not image:
                    continue
                wholecell_tabs.append({'title': tab_title, 'image': image})
                total_images += 1

            if wholecell_tabs:
                wholecell_cards.append(
                    {
                        'card_id': f"patch_{display_idx}_wholecell",
                        'card_title': f"{card_title} — Whole-cell Mask",
                        'tabs': wholecell_tabs,
                    }
                )

            if metrics_summary:
                patch_summaries.append(
                    {
                        'display_name': display_name,
                        'selection_reason_label': reason_label,
                        'metrics': metrics_summary,
                        'source_patch_index': patch.get('source_patch_index'),
                    }
                )

        morphology_path = visualization_dir / 'cell_morphology_stats.png'
        if morphology_path.exists():
            morphology_image = _build_image_dict(
                morphology_path,
                'Cell Morphology Statistics',
                'Distribution of key morphology metrics across matched cells.',
            )

        self.report_data['segmentation_visualization'] = {
            'overlay_cards': overlay_cards,
            'nucleus_cards': nucleus_cards,
            'wholecell_cards': wholecell_cards,
            'overview_image': overview_image,
            'is_large_sample': is_large_sample,
            'large_sample_note': large_sample_note,
            'selected_patch_count': selected_patch_count,
            'target_patch_count': target_patch_count,
            'patch_summaries': patch_summaries,
            'total_images': total_images,
        }

        if total_images:
            logger.info(f"Collected {total_images} segmentation visualization image(s)")
        else:
            logger.info("No segmentation visualization images found")

        self.report_data['segmentation_results_details'] = {
            'morphology_image': morphology_image
        }

    def _collect_segmentation_channel_info(self):
        """Collect segmentation channel usage information from CodexImage target_channels_dict."""
        segmentation_channel_info = {
            'nucleus_channel': 'Unknown',
            'wholecell_channels': [],
            'total_wholecell_channels': 0,
            'channel_details': {}
        }
        
        # Try to get channel information from CodexPatches object if available
        if self.codex_patches and hasattr(self.codex_patches, 'target_channels_dict'):
            try:
                target_channels = self.codex_patches.target_channels_dict
                logger.info(f"Found target_channels_dict: {target_channels}")
                
                # Extract nucleus channel with safe access
                nucleus_channel = target_channels.get('nucleus', 'Unknown')
                segmentation_channel_info['nucleus_channel'] = nucleus_channel
                
                # Extract wholecell channels with safe access
                wholecell_channels = target_channels.get('wholecell', [])
                if isinstance(wholecell_channels, str):
                    wholecell_channels = [wholecell_channels]
                segmentation_channel_info['wholecell_channels'] = wholecell_channels
                segmentation_channel_info['total_wholecell_channels'] = len(wholecell_channels)
                
                # Create detailed channel information
                segmentation_channel_info['channel_details'] = {
                    'nucleus': {
                        'channel_name': nucleus_channel,
                        'purpose': 'Nuclear segmentation',
                        'description': 'Used for identifying cell nuclei boundaries'
                    },
                    'wholecell': []
                }
                
                for i, channel in enumerate(wholecell_channels):
                    segmentation_channel_info['channel_details']['wholecell'].append({
                        'channel_name': channel,
                        'purpose': 'Whole-cell segmentation',
                        'description': f'Used for identifying whole-cell boundaries (channel {i+1})'
                    })
                    
                logger.info(f"Collected segmentation channel info: nucleus={nucleus_channel}, wholecell={wholecell_channels}")
                
            except (KeyError, AttributeError, TypeError) as e:
                logger.warning(f"Error accessing target_channels_dict: {e}")
                logger.warning("Falling back to configuration for channel information")
            
        # Also try to get from antibody information if available
        if (self.codex_patches and hasattr(self.codex_patches, 'antibody_df') and 
            'channel_details' in segmentation_channel_info):
            try:
                antibody_df = self.codex_patches.antibody_df
                
                # Add antibody details if available
                nucleus_channel = segmentation_channel_info['nucleus_channel']
                if (nucleus_channel != 'Unknown' and 
                    nucleus_channel in antibody_df['antibody_name'].values and
                    'nucleus' in segmentation_channel_info['channel_details']):
                    nucleus_row = antibody_df[antibody_df['antibody_name'] == nucleus_channel].iloc[0]
                    segmentation_channel_info['channel_details']['nucleus']['antibody_info'] = nucleus_row.to_dict()
                    
                for wc_info in segmentation_channel_info['channel_details'].get('wholecell', []):
                    channel_name = wc_info['channel_name']
                    if channel_name in antibody_df['antibody_name'].values:
                        channel_row = antibody_df[antibody_df['antibody_name'] == channel_name].iloc[0]
                        wc_info['antibody_info'] = channel_row.to_dict()
            except Exception as e:
                logger.warning(f"Error adding antibody information: {e}")
        
        # Try to get from configuration if CodexPatches is not available
        if (segmentation_channel_info['nucleus_channel'] == 'Unknown' and 
            'channels' in self.config):
            channels_config = self.config.get('channels', {})
            # Try both possible naming conventions
            nucleus_channel = (channels_config.get('nucleus_channel') or 
                             channels_config.get('nuclear_channel', 'Unknown'))
            wholecell_channel = channels_config.get('wholecell_channel', [])
            
            if not isinstance(wholecell_channel, list):
                wholecell_channel = [wholecell_channel]
                
            segmentation_channel_info['nucleus_channel'] = nucleus_channel
            segmentation_channel_info['wholecell_channels'] = wholecell_channel
            segmentation_channel_info['total_wholecell_channels'] = len(wholecell_channel)
            
            # Create channel details for config-based info
            segmentation_channel_info['channel_details'] = {
                'nucleus': {
                    'channel_name': nucleus_channel,
                    'purpose': 'Nuclear segmentation',
                    'description': 'Used for identifying cell nuclei boundaries'
                },
                'wholecell': []
            }
            
            for i, channel in enumerate(wholecell_channel):
                segmentation_channel_info['channel_details']['wholecell'].append({
                    'channel_name': channel,
                    'purpose': 'Whole-cell segmentation',
                    'description': f'Used for identifying whole-cell boundaries (channel {i+1})'
                })
            
            logger.info(f"Got segmentation channels from config: nucleus={nucleus_channel}, wholecell={wholecell_channel}")
        
        self.report_data['segmentation_channel_info'] = segmentation_channel_info
        
        if segmentation_channel_info['nucleus_channel'] != 'Unknown':
            logger.info(f"Collected segmentation channel information successfully")
        else:
            logger.warning("No segmentation channel information found")
            
    def _collect_quality_metrics(self):
        """Collect patch quality metrics."""
        patches_metadata_path = self.output_dir / "cell_profiling" / "patches_metadata.csv"
        
        if patches_metadata_path.exists():
            patches_df = pd.read_csv(patches_metadata_path)
            
            self.report_data['quality_metrics'] = {
                'total_patches': len(patches_df),
                'informative_patches': patches_df['is_informative'].sum() if 'is_informative' in patches_df else 0,
                'mean_nucleus_intensity': patches_df['nucleus_mean'].mean() if 'nucleus_mean' in patches_df else 0,
                'mean_wholecell_intensity': patches_df['wholecell_mean'].mean() if 'wholecell_mean' in patches_df else 0,
            }
        else:
            self.report_data['quality_metrics'] = {
                'total_patches': 0,
                'informative_patches': 0,
            }
            
    def _load_segmentation_metrics(self):
        """Load segmentation evaluation metrics from available pickle artefacts."""
        candidate_names = (
            "seg_evaluation_metrics.pkl.gz",
            "seg_evaluation_metrics.pkl",
        )
        for name in candidate_names:
            path = self.output_dir / name
            if not path.exists():
                continue
            opener = gzip.open if path.suffix == '.gz' else open
            try:
                with opener(path, 'rb') as handle:
                    metrics = pickle.load(handle)
                return metrics, path
            except Exception as exc:
                logger.warning(f"Failed to load {name}: {exc}")
        return None, None

    def _collect_segmentation_stats(self):
        """Collect segmentation statistics."""
        # Try to load from saved files
        stats = {}
        
        # Cell counts from expression matrix
        # Try multiple possible file names
        possible_exp_files = [
            "cell_by_marker.csv",
            "cell_expression_matrix.csv",
            "expression_matrix.csv"
        ]
        
        exp_df = None
        for filename in possible_exp_files:
            exp_matrix_path = self.output_dir / "cell_profiling" / filename
            if exp_matrix_path.exists():
                logger.info(f"Found expression matrix: {filename}")
                exp_df = pd.read_csv(exp_matrix_path)
                stats['total_cells'] = len(exp_df)
                break
        
        if exp_df is None:
            logger.warning("No expression matrix file found")
        
        # Morphology stats from metadata
        cell_metadata_path = self.output_dir / "cell_profiling" / "cell_metadata.csv"
        if cell_metadata_path.exists():
            meta_df = pd.read_csv(cell_metadata_path)
            if 'area' in meta_df:
                stats['mean_cell_area'] = meta_df['area'].mean()
                stats['median_cell_area'] = meta_df['area'].median()
                stats['cell_area_std'] = meta_df['area'].std()
                self._convert_cell_area_units(stats)
            
        # Load evaluation metrics if available
        eval_metrics, eval_path = self._load_segmentation_metrics()
        if eval_metrics is not None:
            quality_scores = [m.get('QualityScore', np.nan)
                              for m in eval_metrics if m is not None]
            stats['mean_quality_score'] = np.nanmean(quality_scores)
            logger.info(f"Loaded segmentation metrics from {eval_path.name}")
        
        # If we still don't have cell count, try to get from patches metadata
        if 'total_cells' not in stats:
            patches_meta_path = self.output_dir / "cell_profiling" / "patches_metadata.csv"
            if patches_meta_path.exists():
                patches_df = pd.read_csv(patches_meta_path)
                # Sum up cells from all patches if available
                if 'n_cells' in patches_df.columns:
                    stats['total_cells'] = patches_df['n_cells'].sum()
                    
        # Try to get matched fraction from patches metadata
        patches_meta_path = self.output_dir / "cell_profiling" / "patches_metadata.csv"
        if patches_meta_path.exists():
            patches_df = pd.read_csv(patches_meta_path)
            if 'matched_fraction' in patches_df.columns:
                # Calculate weighted average of matched fraction
                valid_patches = patches_df[patches_df['is_informative'] == True]
                if len(valid_patches) > 0:
                    stats['matched_fraction'] = valid_patches['matched_fraction'].mean()
                    logger.info(f"Average matched fraction: {stats['matched_fraction']:.3f}")
        
        # Try to get from CodexPatches object if available
        if 'total_cells' not in stats and self.codex_patches:
            if hasattr(self.codex_patches, 'repaired_seg_res_batch'):
                total_cells = 0
                for seg_result in self.codex_patches.repaired_seg_res_batch:
                    if seg_result is not None:
                        cell_mask = seg_result.get('cell_matched_mask', seg_result.get('cell'))
                        if cell_mask is not None:
                            n_cells = len(np.unique(cell_mask)) - 1  # Subtract background
                            total_cells += n_cells
                if total_cells > 0:
                    stats['total_cells'] = total_cells
                    logger.info(f"Got cell count from CodexPatches: {total_cells}")
        
        self.report_data['segmentation_stats'] = stats

    def _convert_cell_area_units(self, stats: Dict[str, Any]) -> None:
        """Augment cell area statistics with micron-squared conversions when possible."""
        input_info = self.report_data.get('input_info', {})
        mpp = input_info.get('microns_per_pixel')
        try:
            mpp = float(mpp)
        except (TypeError, ValueError):
            return

        if mpp <= 0:
            return

        factor = mpp ** 2
        for key in ('mean_cell_area', 'median_cell_area'):
            if key in stats and stats[key] is not None:
                value = stats[key]
                if pd.notna(value):
                    stats[f"{key}_um2"] = float(value) * factor

    def _collect_mask_repair_stats(self):
        """Collect mask repair statistics."""
        stats = {}
        
        # Try to get mask repair stats from codex_patches if available
        if self.codex_patches and hasattr(self.codex_patches, 'repaired_seg_res_batch'):
            total_repair_stats = {
                'nucleus': {'total': 0, 'matched': 0, 'unmatched': 0},
                'whole_cell': {'total': 0, 'matched': 0, 'unmatched': 0}
            }
            
            patches_with_stats = 0
            repair_counts = []
            
            logger.info(f"Found {len(self.codex_patches.repaired_seg_res_batch)} patches in repaired_seg_res_batch")
            
            for idx, seg_result in enumerate(self.codex_patches.repaired_seg_res_batch):
                if seg_result is not None:
                    logger.debug(f"Patch {idx} keys: {seg_result.keys() if isinstance(seg_result, dict) else 'Not a dict'}")
                    
                    if 'matching_stats' in seg_result:
                        matching_stats = seg_result['matching_stats']
                        patches_with_stats += 1
                        
                        # Aggregate statistics
                        for mask_type in ['nucleus', 'whole_cell']:
                            if mask_type in matching_stats:
                                total_repair_stats[mask_type]['total'] += matching_stats[mask_type]['total'][0]
                                total_repair_stats[mask_type]['matched'] += matching_stats[mask_type]['matched'][0]
                                total_repair_stats[mask_type]['unmatched'] += matching_stats[mask_type]['unmatched'][0]
                        
                        # Track repair counts per patch
                        if 'matched_fraction' in seg_result:
                            repair_counts.append(seg_result['matched_fraction'])
            
            if patches_with_stats > 0:
                # Calculate overall percentages
                for mask_type in ['nucleus', 'whole_cell']:
                    total = total_repair_stats[mask_type]['total']
                    if total > 0:
                        stats[f'{mask_type}_total'] = total
                        stats[f'{mask_type}_matched'] = total_repair_stats[mask_type]['matched']
                        stats[f'{mask_type}_unmatched'] = total_repair_stats[mask_type]['unmatched']
                        stats[f'{mask_type}_matched_pct'] = (total_repair_stats[mask_type]['matched'] / total) * 100
                        stats[f'{mask_type}_unmatched_pct'] = (total_repair_stats[mask_type]['unmatched'] / total) * 100
                
                # Calculate repair statistics
                if repair_counts:
                    stats['mean_matched_fraction'] = np.mean(repair_counts)
                    stats['min_matched_fraction'] = np.min(repair_counts)
                    stats['max_matched_fraction'] = np.max(repair_counts)
                    stats['patches_with_repair_stats'] = patches_with_stats
                    
                logger.info(f"Collected mask repair stats from {patches_with_stats} patches")
        
        # Also try to get from patches metadata if available
        patches_meta_path = self.output_dir / "cell_profiling" / "patches_metadata.csv"
        if patches_meta_path.exists() and not stats:
            patches_df = pd.read_csv(patches_meta_path)
            if 'matched_fraction' in patches_df.columns:
                valid_patches = patches_df[patches_df['is_informative'] == True]
                if len(valid_patches) > 0:
                    stats['mean_matched_fraction'] = valid_patches['matched_fraction'].mean()
                    stats['min_matched_fraction'] = valid_patches['matched_fraction'].min()
                    stats['max_matched_fraction'] = valid_patches['matched_fraction'].max()
                    
        self.report_data['mask_repair_stats'] = stats
        
    def _collect_expression_stats(self):
        """Collect marker expression statistics."""
        # Try multiple possible file names
        possible_exp_files = [
            "cell_by_marker.csv",
            "cell_expression_matrix.csv",
            "expression_matrix.csv"
        ]
        
        exp_df = None
        for filename in possible_exp_files:
            exp_matrix_path = self.output_dir / "cell_profiling" / filename
            if exp_matrix_path.exists():
                logger.info(f"Loading expression data from: {filename}")
                exp_df = pd.read_csv(exp_matrix_path)
                break
        
        if exp_df is not None:
            
            # Calculate statistics for each marker
            marker_stats = []
            # For cell_by_marker.csv, all columns are markers
            # For other formats, exclude metadata columns
            exclude_cols = ['cell_id', 'patch_id', 'global_cell_id', 'patch_x', 'patch_y']
            marker_cols = [col for col in exp_df.columns if col not in exclude_cols]
            
            for marker in marker_cols:
                if marker in exp_df:
                    values = exp_df[marker]
                    
                    # Define positive threshold based on marker type
                    # For nuclear markers like DAPI, use a lower threshold
                    # For other markers, use median + 1*std as a more reasonable cutoff
                    if 'DAPI' in marker.upper() or 'HOECHST' in marker.upper():
                        # For nuclear stains, use median as threshold (most cells should be positive)
                        positive_threshold = values.median()
                    else:
                        # For other markers, use median + 1*std (more sensitive than mean + 2*std)
                        positive_threshold = values.median() + values.std()
                    
                    positive_mask = values > positive_threshold
                    
                    marker_stats.append({
                        'marker': marker,
                        'mean': values.mean(),
                        'median': values.median(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'positive_threshold': positive_threshold,
                        'positive_cells': positive_mask.sum(),
                        'positive_percentage': positive_mask.mean() * 100
                    })
            
            self.report_data['expression_stats'] = marker_stats
            
    def _collect_performance_metrics(self):
        """Collect performance and resource usage metrics."""
        # Parse log file for timing information
        log_files = list((self.output_dir.parent.parent / "logs").glob("*.log"))
        
        performance = {
            'total_runtime': 'Unknown',
            'peak_memory': 'Unknown',
        }
        
        # TODO: Parse actual log files for timing and memory info
        
        self.report_data['performance'] = performance
        
    def _collect_patch_mode_quality_data(self):
        """Collect patch-level quality metrics for the patch split mode."""
        patch_quality = {
            'available': False,
            'summary_rows': [],
            'top_patches': [],
            'bottom_patches': [],
            'has_quality_scores': False,
            'distribution_metric_label': None,
            'map_metric_label': None,
            'table_columns': [],
        }
        patches_path = self.output_dir / "cell_profiling" / "patches_metadata.csv"
        if not patches_path.exists():
            logger.warning("patches_metadata.csv not found; skipping patch-quality summary.")
            self.report_data['patch_quality'] = patch_quality
            self.patch_quality_df = None
            return
        try:
            df = pd.read_csv(patches_path)
        except Exception as exc:
            logger.warning(f"Failed to load patches_metadata.csv: {exc}")
            self.report_data['patch_quality'] = patch_quality
            self.patch_quality_df = None
            return

        if 'patch_index' not in df.columns:
            df['patch_index'] = df.index

        if 'is_informative' not in df.columns:
            if 'is_infomative' in df.columns:
                df['is_informative'] = df['is_infomative']
                logger.info("Renamed 'is_infomative' column to 'is_informative'.")
            else:
                df['is_informative'] = True
        df['is_informative'] = df['is_informative'].fillna(False).astype(bool)

        if 'quality_score' not in df.columns:
            df['quality_score'] = np.nan

        informative_indices = df.index[df['is_informative']].tolist()
        eval_metrics, eval_path = self._load_segmentation_metrics()
        if eval_metrics is not None:
            for idx, patch_idx in enumerate(informative_indices):
                if idx >= len(eval_metrics):
                    break
                metrics = eval_metrics[idx]
                score = np.nan
                if isinstance(metrics, dict):
                    score = metrics.get('QualityScore', np.nan)
                df.at[patch_idx, 'quality_score'] = score
            if len(informative_indices) > len(eval_metrics):
                logger.info("Fewer evaluation entries than informative patches; some patches lack quality scores.")
        else:
            logger.info("Segmentation metrics artefact not found; quality scores unavailable for patch-mode report.")

        informative_df = df[df['is_informative']].copy()
        quality_series = informative_df['quality_score'] if 'quality_score' in informative_df else pd.Series(dtype=float)
        valid_quality = quality_series.dropna()
        has_quality = not valid_quality.empty

        def _safe_stat(series, func):
            if series is None:
                return None
            series = series.dropna()
            if series.empty:
                return None
            return func(series)

        summary = {
            'total_patches': int(len(df)),
            'informative_patches': int(informative_df.shape[0]),
            'evaluated_patches': int(valid_quality.shape[0]),
            'quality_score_mean': _safe_stat(valid_quality, np.mean) if has_quality else None,
            'quality_score_median': _safe_stat(valid_quality, np.median) if has_quality else None,
            'quality_score_std': _safe_stat(valid_quality, np.std) if has_quality else None,
            'quality_score_min': _safe_stat(valid_quality, np.min) if has_quality else None,
            'quality_score_max': _safe_stat(valid_quality, np.max) if has_quality else None,
        }
        if 'matched_fraction' in informative_df.columns:
            summary['matched_fraction_mean'] = _safe_stat(informative_df['matched_fraction'], np.mean)
            summary['matched_fraction_std'] = _safe_stat(informative_df['matched_fraction'], np.std)

        def _fmt_number(value, digits=3):
            if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
                return "N/A"
            if isinstance(value, (int, np.integer)):
                return str(int(value))
            return f"{float(value):.{digits}f}"

        summary_rows = [
            {'label': 'Total patches', 'value': _fmt_number(summary['total_patches'])},
            {'label': 'Informative patches', 'value': _fmt_number(summary['informative_patches'])},
            {'label': 'Evaluated patches', 'value': _fmt_number(summary['evaluated_patches'])},
            {'label': 'Mean quality score', 'value': _fmt_number(summary['quality_score_mean'])},
            {'label': 'Median quality score', 'value': _fmt_number(summary['quality_score_median'])},
            {'label': 'Std quality score', 'value': _fmt_number(summary['quality_score_std'])},
            {'label': 'Min quality score', 'value': _fmt_number(summary['quality_score_min'])},
            {'label': 'Max quality score', 'value': _fmt_number(summary['quality_score_max'])},
        ]
        if 'matched_fraction_mean' in summary:
            summary_rows.append({'label': 'Mean matched fraction', 'value': _fmt_number(summary['matched_fraction_mean'])})
            summary_rows.append({'label': 'Std matched fraction', 'value': _fmt_number(summary['matched_fraction_std'])})

        table_columns = ['patch_index']
        for candidate in ['quality_score', 'matched_fraction', 'nucleus_non_zero_perc', 'wholecell_non_zero_perc']:
            if candidate in informative_df.columns and candidate not in table_columns:
                table_columns.append(candidate)

        top_patches = []
        bottom_patches = []
        if has_quality:
            sorted_df = informative_df.dropna(subset=['quality_score']).sort_values('quality_score', ascending=False)
            if not sorted_df.empty:
                top_patches = sorted_df.head(3)[table_columns].to_dict('records')
                bottom_df = sorted_df.sort_values('quality_score', ascending=True)
                bottom_patches = bottom_df.head(3)[table_columns].to_dict('records')
        elif 'matched_fraction' in informative_df.columns:
            sorted_df = informative_df.dropna(subset=['matched_fraction']).sort_values('matched_fraction', ascending=False)
            if not sorted_df.empty:
                top_patches = sorted_df.head(3)[table_columns].to_dict('records')
                bottom_df = sorted_df.sort_values('matched_fraction', ascending=True)
                bottom_patches = bottom_df.head(3)[table_columns].to_dict('records')

        for collection in (top_patches, bottom_patches):
            for entry in collection:
                for key, value in list(entry.items()):
                    if isinstance(value, (float, int, np.floating, np.integer)):
                        entry[key] = _fmt_number(value)

        patch_quality.update({
            'available': True,
            'summary_rows': summary_rows,
            'top_patches': top_patches,
            'bottom_patches': bottom_patches,
            'has_quality_scores': has_quality,
            'table_columns': table_columns,
        })
        self.report_data['patch_quality'] = patch_quality
        self.patch_quality_df = df

    def _generate_patch_mode_figures(self):
        """Generate figures specific to patch-mode reporting."""
        patch_quality = self.report_data.get('patch_quality', {})
        df = getattr(self, 'patch_quality_df', None)

        if not patch_quality.get('available') or df is None or df.empty:
            return

        informative_df = df[df['is_informative']]

        distribution_label = None
        metric_series = pd.Series(dtype=float)
        if 'quality_score' in informative_df and informative_df['quality_score'].dropna().any():
            metric_series = informative_df['quality_score'].dropna()
            distribution_label = 'Quality Score'
        elif 'matched_fraction' in informative_df and informative_df['matched_fraction'].dropna().any():
            metric_series = informative_df['matched_fraction'].dropna()
            distribution_label = 'Matched Fraction'

        if distribution_label is not None and not metric_series.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            bins = min(20, max(5, int(np.sqrt(metric_series.size))))
            sns.histplot(metric_series, bins=bins, ax=ax, kde=True)
            ax.set_title(f"{distribution_label} Distribution (Informative Patches)")
            ax.set_xlabel(distribution_label)
            ax.set_ylabel('Patch Count')
            self.figures['patch_quality_distribution'] = self._fig_to_base64(fig)
            plt.close(fig)
        else:
            self.figures.pop('patch_quality_distribution', None)

        patch_quality['distribution_metric_label'] = distribution_label

        width_col = 'patch_width' if 'patch_width' in df.columns else ('width' if 'width' in df.columns else None)
        height_col = 'patch_height' if 'patch_height' in df.columns else ('height' if 'height' in df.columns else None)
        spatial_ready = all(col in df.columns for col in ['x_start', 'y_start']) and width_col and height_col

        map_metric_col = None
        map_metric_label = None
        if 'quality_score' in df and df['quality_score'].dropna().any():
            map_metric_col = 'quality_score'
            map_metric_label = 'Quality Score'
        elif 'matched_fraction' in df and df['matched_fraction'].dropna().any():
            map_metric_col = 'matched_fraction'
            map_metric_label = 'Matched Fraction'

        if spatial_ready and map_metric_col is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            cmap = plt.cm.viridis
            scores = df[map_metric_col]
            scores = scores[pd.notna(scores)]
            if not scores.empty:
                vmin = float(scores.min())
                vmax = float(scores.max())
                if vmin == vmax:
                    vmin -= 0.5
                    vmax += 0.5
                norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = None

            x_max = (df['x_start'] + df[width_col]).max()
            y_max = (df['y_start'] + df[height_col]).max()

            for _, row in df.iterrows():
                width = row.get(width_col, 0)
                height = row.get(height_col, 0)
                if width == 0 or height == 0:
                    continue
                score = row.get(map_metric_col, np.nan)
                informative = bool(row.get('is_informative', False))
                alpha = 0.8 if informative else 0.25
                facecolor = (0.85, 0.85, 0.85, 1.0)
                if pd.notna(score) and norm is not None:
                    facecolor = cmap(norm(score))
                rect = Rectangle((row['x_start'], row['y_start']), width, height,
                                 facecolor=facecolor, edgecolor='black', linewidth=0.3, alpha=alpha)
                ax.add_patch(rect)

            ax.set_xlim(df['x_start'].min(), x_max)
            ax.set_ylim(df['y_start'].min(), y_max)
            ax.set_aspect('equal', adjustable='box')
            ax.invert_yaxis()
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_title(f"{map_metric_label} Across Patches")

            if norm is not None:
                sm = ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=map_metric_label)

            self.figures['patch_quality_map'] = self._fig_to_base64(fig)
            plt.close(fig)
        else:
            self.figures.pop('patch_quality_map', None)

        patch_quality['map_metric_label'] = map_metric_label
        self.report_data['patch_quality'] = patch_quality

    def _generate_summary_figures(self):
        """Generate summary visualization figures."""
        # 1. Quality metrics dashboard
        if 'quality_metrics' in self.report_data:
            fig, ax = plt.subplots(figsize=(8, 6))
            metrics = self.report_data['quality_metrics']
            
            # Create bar plot
            keys = ['total_patches', 'informative_patches']
            values = [metrics.get(k, 0) for k in keys]
            ax.bar(keys, values)
            ax.set_title('Patch Quality Summary')
            ax.set_ylabel('Count')
            
            self.figures['quality_dashboard'] = self._fig_to_base64(fig)
            plt.close(fig)
        
        # 2. Expression heatmap
        if 'expression_stats' in self.report_data:
            stats_df = pd.DataFrame(self.report_data['expression_stats'])
            if not stats_df.empty:
                # Create improved heatmap with better scaling and layout
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(stats_df) * 0.4), 8))
                
                # Separate heatmaps for mean and positive percentage
                # 1. Mean expression (log-scaled for better visualization)
                mean_data = stats_df.set_index('marker')[['mean']].T
                # Apply log10 transformation for better visualization (add 1 to avoid log(0))
                mean_data_log = np.log10(mean_data + 1)
                sns.heatmap(mean_data_log, annot=False, ax=ax1, cmap='viridis', 
                           cbar_kws={'label': 'Log10(Mean Expression + 1)'})
                ax1.set_title('Mean Expression Levels (Log Scale)')
                ax1.set_xlabel('')
                ax1.tick_params(axis='x', rotation=45, labelsize=8)
                
                # 2. Positive percentage
                pos_data = stats_df.set_index('marker')[['positive_percentage']].T
                sns.heatmap(pos_data, annot=True, fmt='.1f', ax=ax2, cmap='Reds',
                           cbar_kws={'label': 'Positive Cells (%)'})
                ax2.set_title('Positive Cell Percentage')
                ax2.set_xlabel('Markers')
                ax2.tick_params(axis='x', rotation=45, labelsize=8)
                
                plt.tight_layout()
                
            else:
                # Fallback for empty data
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.text(0.5, 0.5, 'No expression data available', 
                       ha='center', va='center', fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
            self.figures['expression_heatmap'] = self._fig_to_base64(fig)
            plt.close(fig)
            
            # 3. Additional bar chart for better comparison
            if not stats_df.empty:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Sort by mean expression for better visualization
                stats_sorted = stats_df.sort_values('mean', ascending=True)
                
                # Bar chart for mean expression (log scale)
                bars1 = ax1.barh(range(len(stats_sorted)), np.log10(stats_sorted['mean'] + 1))
                ax1.set_yticks(range(len(stats_sorted)))
                ax1.set_yticklabels(stats_sorted['marker'], fontsize=8)
                ax1.set_xlabel('Log10(Mean Expression + 1)')
                ax1.set_title('Mean Expression Levels by Marker')
                ax1.grid(axis='x', alpha=0.3)
                
                # Color bars by value
                for i, bar in enumerate(bars1):
                    bar.set_color(plt.cm.viridis(i / len(bars1)))
                
                # Bar chart for positive percentage
                stats_sorted_pos = stats_df.sort_values('positive_percentage', ascending=True)
                bars2 = ax2.barh(range(len(stats_sorted_pos)), stats_sorted_pos['positive_percentage'])
                ax2.set_yticks(range(len(stats_sorted_pos)))
                ax2.set_yticklabels(stats_sorted_pos['marker'], fontsize=8)
                ax2.set_xlabel('Positive Cells (%)')
                ax2.set_title('Positive Cell Percentage by Marker')
                ax2.grid(axis='x', alpha=0.3)
                
                # Color bars by value
                for i, bar in enumerate(bars2):
                    bar.set_color(plt.cm.Reds(stats_sorted_pos.iloc[i]['positive_percentage'] / 100))
                
                plt.tight_layout()
                self.figures['expression_bars'] = self._fig_to_base64(fig)
                plt.close(fig)
            
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string for embedding in HTML."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        return f"data:image/png;base64,{img_str}"
        
    def _render_html_report(self) -> str:
        """Render HTML report using template."""
        template_str = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PhenoCycler Pipeline Report - {{ metadata.experiment_id }}</title>
    
    <style>
        *, *::before, *::after { box-sizing: border-box; }
        .summary-box { display: flow-root; }
        .segmentation-gallery { align-items: start; }
        .segmentation-card { height: auto; }     /* 覆盖原来的 height:100% */
        .segmentation-card img {
        display: block;
        max-width: 100%;
        height: auto;
        }    
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        h3 {
            color: #666;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 18px;
        }
        .summary-box {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }
        .metrics-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 15px 0;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .figure {
            text-align: center;
            margin: 20px 0;
        }
        .figure img {
            max-width: 100%;
            border: 1px solid #ddd;
        }
        .warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .segmentation-section {
            margin-bottom: 30px;
        }
        .segmentation-section + .segmentation-section {
            border-top: 1px solid #e0e0e0;
            padding-top: 20px;
        }
        .segmentation-section h3 {
            margin: 0 0 12px;
            color: #2c3e50;
        }
        .segmentation-gallery {
            display: grid;
            gap: 24px;
        }
        .segmentation-gallery--single {
            grid-template-columns: minmax(0, 1fr);
        }
        .segmentation-gallery--paired {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        @media (max-width: 1100px) {
            .segmentation-gallery--paired {
                grid-template-columns: minmax(0, 1fr);
            }
        }
        .segmentation-gallery--wide {
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
        }
        .segmentation-card {
            background-color: #fafafa;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .segmentation-card__title {
            font-weight: 600;
            font-size: 15px;
            color: #333;
            margin-bottom: 12px;
            text-align: left;
        }
        .segmentation-card img {
            width: 100%;
            height: auto;
            border-radius: 5px;
            border: 1px solid #ccc;
            margin-bottom: 12px;
        }
        .segmentation-card figcaption {
            font-size: 12px;
            color: #666;
            line-height: 1.4;
            margin-top: auto;
        }
        .segmentation-card .segmentation-filename {
            display: block;
            font-weight: 600;
            color: #444;
            margin: 0 0 2px;
        }
        .segmentation-card__title + img {
            margin-bottom: 8px;
        }
        .segmentation-card figcaption {
            margin-top: 6px;
        }
        .segmentation-card--tabbed {
            position: relative;
        }
        .segmentation-tab-bar {
            display: inline-flex;
            gap: 8px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }
        .segmentation-tab {
            border: 1px solid #b3b3b3;
            background: #f3f3f3;
            border-radius: 4px;
            padding: 4px 12px;
            font-size: 12px;
            cursor: pointer;
            transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease;
        }
        .segmentation-tab:hover {
            background: #e4e4e4;
        }
        .segmentation-tab.is-active {
            background: #1b73d1;
            color: #ffffff;
            border-color: #1b73d1;
        }
        .segmentation-tab-panel {
            display: none;
        }
        .segmentation-tab-panel.is-active {
            display: block;
        }
        .segmentation-card--tabbed img {
            margin-bottom: 10px;
        }
        .segmentation-caption {
            font-size: 12px;
            color: #666;
            line-height: 1.4;
        }
        .segmentation-caption .segmentation-filename {
            display: block;
            font-weight: 600;
            color: #444;
            margin-bottom: 2px;
        }
        .segmentation-note {
            font-size: 13px;
            color: #555;
            margin: 8px 0 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PhenoCycler Pipeline Analysis Report</h1>
        
        <div class="summary-box">
            <h2>Executive Summary</h2>
            <div class="metric">
                <div class="metric-value">{{ metadata.experiment_id }}</div>
                <div class="metric-label">Experiment ID</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ segmentation_stats.total_cells|default(0) }}</div>
                <div class="metric-label">Total Cells</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ (quality_metrics.informative_patches if quality_metrics else 0)|default(0) }}/{{ (quality_metrics.total_patches if quality_metrics else 0)|default(0) }}</div>
                <div class="metric-label">Valid Patches</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.2f"|format(segmentation_stats.mean_quality_score|default(0)) }}</div>
                <div class="metric-label">Quality Score</div>
            </div>
            {% if segmentation_stats.matched_fraction %}
            <div class="metric">
                <div class="metric-value">{{ "%.1f"|format(segmentation_stats.matched_fraction * 100) }}%</div>
                <div class="metric-label">Nucleus-Cell Match</div>
            </div>
            {% endif %}
        </div>
        
        <h2>Input Data Information</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Image File</td>
                <td>{{ input_info.image_file }}</td>
            </tr>
            <tr>
                <td>Image Dimensions</td>
                <td>{{ input_info.image_width }} × {{ input_info.image_height }} × {{ input_info.n_channels }}</td>
            </tr>
            <tr>
                <td>Resolution</td>
                <td>{{ input_info.microns_per_pixel }} μm/pixel</td>
            </tr>
            <tr>
                <td>Antibodies File</td>
                <td>{{ input_info.antibodies_file }}</td>
            </tr>
        </table>
        
        {% if sample_preview.images %}
        <h2>Sample Preview</h2>
        <div class="summary-box">
            <div class="metrics-grid">
                {% for image in sample_preview.images %}
                <div style="text-align: center; margin: 20px;">
                    <h4>Channel {{ image.channel }}</h4>
                    <img src="{{ image.data }}" alt="{{ image.filename }}" style="max-width: 300px; max-height: 300px; border: 2px solid #ddd; border-radius: 5px;">
                    <p style="font-size: 12px; color: #666; margin-top: 5px;">{{ image.filename }}</p>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        <h2>Segmentation Channel Configuration</h2>
        {% if segmentation_channel_info.nucleus_channel != 'Unknown' %}
        <div class="summary-box">
            <h3>Channel Usage Summary</h3>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{{ segmentation_channel_info.nucleus_channel }}</div>
                    <div class="metric-label">Nucleus Channel</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ segmentation_channel_info.total_wholecell_channels }}</div>
                    <div class="metric-label">Whole-cell Channels</div>
                </div>
            </div>
            
            <h3>Channel Details</h3>
            <table>
                <tr>
                    <th>Channel Type</th>
                    <th>Channel Name</th>
                    <th>Purpose</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td><strong>Nucleus</strong></td>
                    <td>{{ segmentation_channel_info.channel_details.nucleus.channel_name }}</td>
                    <td>{{ segmentation_channel_info.channel_details.nucleus.purpose }}</td>
                    <td>{{ segmentation_channel_info.channel_details.nucleus.description }}</td>
                </tr>
                {% for wc_channel in segmentation_channel_info.channel_details.wholecell %}
                <tr>
                    <td><strong>Whole-cell</strong></td>
                    <td>{{ wc_channel.channel_name }}</td>
                    <td>{{ wc_channel.purpose }}</td>
                    <td>{{ wc_channel.description }}</td>
                </tr>
                {% endfor %}
            </table>
            
            {% if segmentation_channel_info.wholecell_channels|length > 1 %}
            <div class="warning">
                <strong>Multiple Whole-cell Channels Detected:</strong><br>
                The pipeline is using {{ segmentation_channel_info.total_wholecell_channels }} whole-cell channels: 
                {{ segmentation_channel_info.wholecell_channels|join(', ') }}. 
                These channels are combined using maximum projection for segmentation.
            </div>
            {% endif %}
        </div>
        {% else %}
        <div class="warning">
            <strong>Segmentation channel information not available.</strong><br>
            Channel configuration could not be retrieved from the pipeline data.
        </div>
        {% endif %}
        <h2>Segmentation Overview</h2>
        <p class="segmentation-note">
            Metrics and figures below summarize matched/repaired cells. See <em>Mask Repair Details</em> for additional repair statistics.
        </p>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Cells Detected</td>
                <td>{{ segmentation_stats.total_cells|default('N/A') }}</td>
            </tr>
            <tr>
                <td>Mean Cell Area</td>
                <td>
                    {% if segmentation_stats.mean_cell_area is defined and segmentation_stats.mean_cell_area is not none %}
                        {{ "%.1f"|format(segmentation_stats.mean_cell_area) }} px
                        {% if segmentation_stats.mean_cell_area_um2 is defined %}
                            (approx. {{ "%.1f"|format(segmentation_stats.mean_cell_area_um2) }} μm²)
                        {% endif %}
                    {% else %}
                        N/A
                    {% endif %}
                </td>
            </tr>
            <tr>
                <td>Median Cell Area</td>
                <td>
                    {% if segmentation_stats.median_cell_area is defined and segmentation_stats.median_cell_area is not none %}
                        {{ "%.1f"|format(segmentation_stats.median_cell_area) }} px
                        {% if segmentation_stats.median_cell_area_um2 is defined %}
                            (approx. {{ "%.1f"|format(segmentation_stats.median_cell_area_um2) }} μm²)
                        {% endif %}
                    {% else %}
                        N/A
                    {% endif %}
                </td>
            </tr>
            <tr>
                <td>Quality Score</td>
                <td>{{ "%.3f"|format(segmentation_stats.mean_quality_score|default(0)) }}</td>
            </tr>
        </table>
        {% if segmentation_results_details.morphology_image %}
        <div class="figure">
            <img src="{{ segmentation_results_details.morphology_image.data }}" alt="{{ segmentation_results_details.morphology_image.filename }}">
            <p class="figure-caption">
                <span class="segmentation-filename">{{ segmentation_results_details.morphology_image.filename }}</span>
                {{ segmentation_results_details.morphology_image.description }}.
                <br>Solidity = cell area divided by its convex hull area (values near 1 indicate compact shapes).
                <br>Eccentricity ranges from 0 for circular cells to 1 for elongated, line-like cells.
            </p>
        </div>
        {% endif %}
        
        <h2>Mask Repair Details</h2>
        {% if mask_repair_stats %}
        <div class="summary-box">
            <h3>Summary Statistics</h3>
            <div class="metrics-grid">
                {% if mask_repair_stats.mean_matched_fraction is defined %}
                <div class="metric">
                    <div class="metric-value">{{ "%.1f"|format(mask_repair_stats.mean_matched_fraction * 100) }}%</div>
                    <div class="metric-label">Nucleus-Cell Match</div>
                </div>
                {% endif %}
                {% if mask_repair_stats.nucleus_total %}
                <div class="metric">
                    <div class="metric-value">{{ mask_repair_stats.nucleus_matched }}/{{ mask_repair_stats.nucleus_total }}</div>
                    <div class="metric-label">Matched Nuclei</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ "%.1f"|format(mask_repair_stats.nucleus_matched_pct) }}%</div>
                    <div class="metric-label">Nucleus Match Rate</div>
                </div>
                {% endif %}
                {% if mask_repair_stats.whole_cell_total %}
                <div class="metric">
                    <div class="metric-value">{{ mask_repair_stats.whole_cell_matched }}/{{ mask_repair_stats.whole_cell_total }}</div>
                    <div class="metric-label">Matched Cells</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ "%.1f"|format(mask_repair_stats.whole_cell_matched_pct) }}%</div>
                    <div class="metric-label">Cell Match Rate</div>
                </div>
                {% endif %}
            </div>
            {% if mask_repair_stats.mean_matched_fraction is defined %}
            <p style="font-size: 12px; color: #666; margin: 5px 0 0 0;">
                Nucleus-Cell Match indicates the proportion of nuclei and whole cells that remain paired after repair.<br>
                Formula: matched_fraction = matched_cells / (matched_cells + unmatched_cells + unmatched_nuclei), averaged across informative patches.
            </p>
            {% endif %}
        </div>
        {% else %}
        <div class="warning">
            <strong>No mask repair statistics available.</strong><br>
            This may indicate that mask repair was not performed or statistics were not collected during processing.
        </div>
        {% endif %}
        
        <h2>Segmentation Visualization</h2>
        <div class="summary-box">
            <p class="segmentation-note">
                {% if segmentation_visualization.large_sample_note %}
                    {{ segmentation_visualization.large_sample_note }}
                {% else %}
                    Displaying {{ segmentation_visualization.selected_patch_count }} patch(es) with {{ segmentation_visualization.total_images }} generated visualization image(s).
                {% endif %}
            </p>
            {% if segmentation_visualization.patch_summaries %}
            <div class="segmentation-caption">
                {% for patch in segmentation_visualization.patch_summaries %}
                <div>
                    <strong>{{ patch.display_name }}</strong>{% if patch.selection_reason_label %} — {{ patch.selection_reason_label }}{% endif %}
                    {% if patch.metrics %}: {{ patch.metrics }}{% endif %}
                </div>
                {% endfor %}
            </div>
            {% endif %}
            {% if segmentation_visualization.overview_image %}
            <figure class="segmentation-card">
                <div class="segmentation-card__title">{{ segmentation_visualization.overview_image.title }}</div>
                <img src="{{ segmentation_visualization.overview_image.data }}" alt="{{ segmentation_visualization.overview_image.filename }}">
                <figcaption>
                    <span class="segmentation-filename">{{ segmentation_visualization.overview_image.filename }}</span>
                    {{ segmentation_visualization.overview_image.description }}
                </figcaption>
            </figure>
            {% endif %}
            {% if segmentation_visualization.overlay_cards %}
            <div class="segmentation-section">
                <h3>Segmentation Overlays</h3>
                <div class="segmentation-gallery segmentation-gallery--wide">
                    {% for card in segmentation_visualization.overlay_cards %}
                    <div class="segmentation-card segmentation-card--tabbed" data-card-id="{{ card.card_id }}">
                        <div class="segmentation-card__title">{{ card.card_title }}</div>
                        <div class="segmentation-tab-bar" role="tablist" aria-label="{{ card.card_title }}">
                            {% for tab in card.tabs %}
                            <button type="button" class="segmentation-tab{% if loop.first %} is-active{% endif %}" role="tab" data-target="{{ card.card_id }}-{{ loop.index0 }}">
                                {{ tab.title }}
                            </button>
                            {% endfor %}
                        </div>
                        {% for tab in card.tabs %}
                        <div class="segmentation-tab-panel{% if loop.first %} is-active{% endif %}" id="{{ card.card_id }}-{{ loop.index0 }}" role="tabpanel">
                            <img src="{{ tab.image.data }}" alt="{{ tab.image.filename }}">
                            <div class="segmentation-caption">
                                <span class="segmentation-filename">{{ tab.image.filename }}</span>
                                {{ tab.image.description }}
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
            {% if segmentation_visualization.nucleus_cards %}
            <div class="segmentation-section">
                <h3>Nucleus Masks</h3>
                <div class="segmentation-gallery {{ 'segmentation-gallery--paired' if segmentation_visualization.nucleus_cards|length > 1 else 'segmentation-gallery--single' }}">
                    {% for card in segmentation_visualization.nucleus_cards %}
                    <div class="segmentation-card segmentation-card--tabbed" data-card-id="{{ card.card_id }}">
                        <div class="segmentation-card__title">{{ card.card_title }}</div>
                        <div class="segmentation-tab-bar" role="tablist" aria-label="{{ card.card_title }}">
                            {% for tab in card.tabs %}
                            <button type="button" class="segmentation-tab{% if loop.first %} is-active{% endif %}" role="tab" data-target="{{ card.card_id }}-{{ loop.index0 }}">
                                {{ tab.title }}
                            </button>
                            {% endfor %}
                        </div>
                        {% for tab in card.tabs %}
                        <div class="segmentation-tab-panel{% if loop.first %} is-active{% endif %}" id="{{ card.card_id }}-{{ loop.index0 }}" role="tabpanel">
                            <img src="{{ tab.image.data }}" alt="{{ tab.image.filename }}">
                            <div class="segmentation-caption">
                                <span class="segmentation-filename">{{ tab.image.filename }}</span>
                                {{ tab.image.description }}
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
            {% if segmentation_visualization.wholecell_cards %}
            <div class="segmentation-section">
                <h3>Whole-cell Masks</h3>
                <div class="segmentation-gallery {{ 'segmentation-gallery--paired' if segmentation_visualization.wholecell_cards|length > 1 else 'segmentation-gallery--single' }}">
                    {% for card in segmentation_visualization.wholecell_cards %}
                    <div class="segmentation-card segmentation-card--tabbed" data-card-id="{{ card.card_id }}">
                        <div class="segmentation-card__title">{{ card.card_title }}</div>
                        <div class="segmentation-tab-bar" role="tablist" aria-label="{{ card.card_title }}">
                            {% for tab in card.tabs %}
                            <button type="button" class="segmentation-tab{% if loop.first %} is-active{% endif %}" role="tab" data-target="{{ card.card_id }}-{{ loop.index0 }}">
                                {{ tab.title }}
                            </button>
                            {% endfor %}
                        </div>
                        {% for tab in card.tabs %}
                        <div class="segmentation-tab-panel{% if loop.first %} is-active{% endif %}" id="{{ card.card_id }}-{{ loop.index0 }}" role="tabpanel">
                            <img src="{{ tab.image.data }}" alt="{{ tab.image.filename }}">
                            <div class="segmentation-caption">
                                <span class="segmentation-filename">{{ tab.image.filename }}</span>
                                {{ tab.image.description }}
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
        </div>

        {% if figures.quality_dashboard %}
        <h2>Quality Metrics</h2>
        <div class="figure">
            <img src="{{ figures.quality_dashboard }}" alt="Quality Dashboard">
        </div>
        {% endif %}
        
        {# Marker Expression Statistics temporarily removed #}
        
        <h2>Pipeline Metadata</h2>
        <table>
            <tr>
                <td>Report Generated</td>
                <td>{{ metadata.report_generated }}</td>
            </tr>
            <tr>
                <td>Output Directory</td>
                <td>{{ metadata.output_directory }}</td>
            </tr>
            <tr>
                <td>Pipeline Version</td>
                <td>{{ metadata.pipeline_version }}</td>
            </tr>
        </table>
        
        <div class="success">
            <strong>Report generation completed successfully!</strong>
        </div>
    </div>
    <script>
document.addEventListener('DOMContentLoaded', function () {
    var tabbedCards = document.querySelectorAll('.segmentation-card--tabbed');
    tabbedCards.forEach(function (card) {
        var tabs = card.querySelectorAll('.segmentation-tab');
        var panels = card.querySelectorAll('.segmentation-tab-panel');
        tabs.forEach(function (tab, tabIndex) {
            var targetId = tab.getAttribute('data-target');
            tab.setAttribute('aria-selected', tabIndex === 0 ? 'true' : 'false');
            tab.addEventListener('click', function () {
                tabs.forEach(function (t) {
                    t.classList.remove('is-active');
                    t.setAttribute('aria-selected', 'false');
                });
                panels.forEach(function (panel) {
                    panel.classList.remove('is-active');
                });
                tab.classList.add('is-active');
                tab.setAttribute('aria-selected', 'true');
                var targetPanel = card.querySelector('#' + targetId);
                if (targetPanel) {
                    targetPanel.classList.add('is-active');
                }
            });
        });
    });
});
</script>
</body>
</html>
        '''
        
        template = Template(template_str)
        return template.render(**self.report_data, figures=self.figures)


    def _render_patch_mode_html_report(self) -> str:
        """Render HTML report tailored for patch-mode outputs."""
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PhenoCycler Patch Report - {{ metadata.experiment_id }}</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 24px;
            box-shadow: 0 0 12px rgba(0, 0, 0, 0.08);
            border-radius: 8px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 12px;
        }
        h2 {
            color: #34495e;
            margin-top: 32px;
            border-bottom: 1px solid #dfe6e9;
            padding-bottom: 8px;
        }
        h3 {
            color: #2c3e50;
            margin-top: 24px;
            margin-bottom: 12px;
        }
        .summary-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            margin: 16px 0 24px 0;
        }
        .summary-item {
            flex: 1 1 220px;
            background-color: #f8fbff;
            border: 1px solid #d6e4ff;
            border-radius: 6px;
            padding: 12px 16px;
        }
        .summary-label {
            display: block;
            color: #5f6c7b;
            font-size: 13px;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .summary-value {
            font-size: 18px;
            font-weight: 600;
            color: #1e3d59;
            word-break: break-word;
        }
        .summary-box {
            background-color: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 16px;
            margin: 16px 0 24px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
        }
        th, td {
            border: 1px solid #dfe6e9;
            padding: 10px 12px;
            text-align: left;
        }
        th {
            background-color: #ecf5ff;
            color: #2c3e50;
        }
        tr:nth-child(even) td {
            background-color: #fafcfe;
        }
        .figure {
            text-align: center;
            margin: 20px 0;
        }
        .figure img {
            max-width: 100%;
            border: 1px solid #dfe6e9;
            border-radius: 4px;
            background: white;
        }
        .patch-tables {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
        }
        .patch-table-card {
            border: 1px solid #dfe6e9;
            border-radius: 6px;
            background-color: white;
            padding: 12px 16px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        }
        .figure-caption {
            color: #5f6c7b;
            font-size: 13px;
            margin-top: 6px;
        }
        .segmentation-section {
            margin-bottom: 32px;
        }
        .segmentation-section + .segmentation-section {
            border-top: 1px solid #e0e0e0;
            padding-top: 20px;
        }
        .segmentation-card {
            background-color: #ffffff;
            border: 1px solid #dfe6e9;
            border-radius: 8px;
            box-shadow: 0 3px 6px rgba(0, 0, 0, 0.04);
            padding: 12px;
        }
        .segmentation-card__title {
            font-weight: 600;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        .segmentation-gallery {
            display: grid;
            gap: 24px;
        }
        .segmentation-gallery--paired {
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        }
        .segmentation-gallery--single {
            grid-template-columns: minmax(320px, 520px);
        }
        .segmentation-card img {
            display: block;
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .segmentation-caption {
            margin-top: 8px;
            font-size: 13px;
            color: #5f6c7b;
        }
        .segmentation-tab-bar {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }
        .segmentation-tab {
            border: 1px solid #b2bec3;
            background-color: #ecf0f1;
            color: #34495e;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            cursor: pointer;
        }
        .segmentation-tab.is-active {
            background-color: #4CAF50;
            color: white;
            border-color: #4CAF50;
        }
        .segmentation-tab-panel {
            display: none;
        }
        .segmentation-tab-panel.is-active {
            display: block;
        }
        .note {
            background-color: #fff9e6;
            border: 1px solid #ffe599;
            color: #8a6d3b;
            padding: 10px 12px;
            border-radius: 4px;
            margin: 12px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PhenoCycler Patch Report - {{ metadata.experiment_id }}</h1>

        <div class="summary-grid">
            <div class="summary-item">
                <span class="summary-label">Report Generated</span>
                <span class="summary-value">{{ metadata.report_generated }}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Split Mode</span>
                <span class="summary-value">{{ split_mode }}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Image Dimensions</span>
                <span class="summary-value">{{ input_info.image_width }} × {{ input_info.image_height }}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Microns Per Pixel</span>
                <span class="summary-value">{{ input_info.microns_per_pixel }}</span>
            </div>
        </div>

        {% if patch_quality.summary_rows %}
        <h2>Patch Quality Overview</h2>
        <div class="summary-box">
            <table>
                <tbody>
                {% for row in patch_quality.summary_rows %}
                    <tr>
                        <th>{{ row.label }}</th>
                        <td>{{ row.value }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        {% if segmentation_stats %}
        <h2>Segmentation Statistics</h2>
        <table>
            <tbody>
            {% for key, value in segmentation_stats.items() %}
                <tr>
                    <th>{{ key.replace('_', ' ') | title }}</th>
                    <td>{{ value }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
        {% endif %}

        {% if patch_quality.distribution_metric_label and figures.patch_quality_distribution %}
        <h2>{{ patch_quality.distribution_metric_label }} Distribution</h2>
        <div class="figure">
            <img src="{{ figures.patch_quality_distribution }}" alt="{{ patch_quality.distribution_metric_label }} Distribution">
        </div>
        {% endif %}

        {% if patch_quality.map_metric_label and figures.patch_quality_map %}
        <h2>{{ patch_quality.map_metric_label }} Across Patches</h2>
        <div class="figure">
            <img src="{{ figures.patch_quality_map }}" alt="{{ patch_quality.map_metric_label }} Map">
        </div>
        {% endif %}

        <div class="patch-tables">
            {% if patch_quality.top_patches %}
            <div class="patch-table-card">
                <h3>Top Patches</h3>
                <table>
                    <thead>
                        <tr>
                        {% for col in patch_quality.table_columns %}
                            <th>{{ col.replace('_', ' ') | title }}</th>
                        {% endfor %}
                        </tr>
                    </thead>
                    <tbody>
                    {% for patch in patch_quality.top_patches %}
                        <tr>
                        {% for col in patch_quality.table_columns %}
                            <td>{{ patch.get(col, 'N/A') }}</td>
                        {% endfor %}
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endif %}

            {% if patch_quality.bottom_patches %}
            <div class="patch-table-card">
                <h3>Lowest Patches</h3>
                <table>
                    <thead>
                        <tr>
                        {% for col in patch_quality.table_columns %}
                            <th>{{ col.replace('_', ' ') | title }}</th>
                        {% endfor %}
                        </tr>
                    </thead>
                    <tbody>
                    {% for patch in patch_quality.bottom_patches %}
                        <tr>
                        {% for col in patch_quality.table_columns %}
                            <td>{{ patch.get(col, 'N/A') }}</td>
                        {% endfor %}
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endif %}
        </div>

        {% if segmentation_visualization.overlay_cards or segmentation_visualization.nucleus_cards or segmentation_visualization.wholecell_cards %}
        <h2>Segmentation Highlights</h2>
        {% if segmentation_visualization.large_sample_note %}
        <div class="note">{{ segmentation_visualization.large_sample_note }}</div>
        {% endif %}

        {% if segmentation_visualization.overlay_cards %}
        <div class="segmentation-section">
            <h3>Overlay Visualizations</h3>
            <div class="segmentation-gallery {{ 'segmentation-gallery--paired' if segmentation_visualization.overlay_cards|length > 1 else 'segmentation-gallery--single' }}">
                {% for card in segmentation_visualization.overlay_cards %}
                <div class="segmentation-card segmentation-card--tabbed" data-card-id="{{ card.card_id }}">
                    <div class="segmentation-card__title">{{ card.card_title }}</div>
                    <div class="segmentation-tab-bar" role="tablist" aria-label="{{ card.card_title }}">
                        {% for tab in card.tabs %}
                        <button type="button" class="segmentation-tab{% if loop.first %} is-active{% endif %}" role="tab" data-target="{{ card.card_id }}-{{ loop.index0 }}">
                            {{ tab.title }}
                        </button>
                        {% endfor %}
                    </div>
                    {% for tab in card.tabs %}
                    <div class="segmentation-tab-panel{% if loop.first %} is-active{% endif %}" id="{{ card.card_id }}-{{ loop.index0 }}" role="tabpanel">
                        <img src="{{ tab.image.data }}" alt="{{ tab.image.filename }}">
                        <div class="segmentation-caption">
                            <span class="segmentation-filename">{{ tab.image.filename }}</span>
                            {{ tab.image.description }}
                        </div>
                    </div>
                    {% endfor %}
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        {% if segmentation_visualization.nucleus_cards %}
        <div class="segmentation-section">
            <h3>Nucleus Masks</h3>
            <div class="segmentation-gallery {{ 'segmentation-gallery--paired' if segmentation_visualization.nucleus_cards|length > 1 else 'segmentation-gallery--single' }}">
                {% for card in segmentation_visualization.nucleus_cards %}
                <div class="segmentation-card segmentation-card--tabbed" data-card-id="{{ card.card_id }}">
                    <div class="segmentation-card__title">{{ card.card_title }}</div>
                    <div class="segmentation-tab-bar" role="tablist" aria-label="{{ card.card_title }}">
                        {% for tab in card.tabs %}
                        <button type="button" class="segmentation-tab{% if loop.first %} is-active{% endif %}" role="tab" data-target="{{ card.card_id }}-{{ loop.index0 }}">
                            {{ tab.title }}
                        </button>
                        {% endfor %}
                    </div>
                    {% for tab in card.tabs %}
                    <div class="segmentation-tab-panel{% if loop.first %} is-active{% endif %}" id="{{ card.card_id }}-{{ loop.index0 }}" role="tabpanel">
                        <img src="{{ tab.image.data }}" alt="{{ tab.image.filename }}">
                        <div class="segmentation-caption">
                            <span class="segmentation-filename">{{ tab.image.filename }}</span>
                            {{ tab.image.description }}
                        </div>
                    </div>
                    {% endfor %}
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        {% if segmentation_visualization.wholecell_cards %}
        <div class="segmentation-section">
            <h3>Whole-cell Masks</h3>
            <div class="segmentation-gallery {{ 'segmentation-gallery--paired' if segmentation_visualization.wholecell_cards|length > 1 else 'segmentation-gallery--single' }}">
                {% for card in segmentation_visualization.wholecell_cards %}
                <div class="segmentation-card segmentation-card--tabbed" data-card-id="{{ card.card_id }}">
                    <div class="segmentation-card__title">{{ card.card_title }}</div>
                    <div class="segmentation-tab-bar" role="tablist" aria-label="{{ card.card_title }}">
                        {% for tab in card.tabs %}
                        <button type="button" class="segmentation-tab{% if loop.first %} is-active{% endif %}" role="tab" data-target="{{ card.card_id }}-{{ loop.index0 }}">
                            {{ tab.title }}
                        </button>
                        {% endfor %}
                    </div>
                    {% for tab in card.tabs %}
                    <div class="segmentation-tab-panel{% if loop.first %} is-active{% endif %}" id="{{ card.card_id }}-{{ loop.index0 }}" role="tabpanel">
                        <img src="{{ tab.image.data }}" alt="{{ tab.image.filename }}">
                        <div class="segmentation-caption">
                            <span class="segmentation-filename">{{ tab.image.filename }}</span>
                            {{ tab.image.description }}
                        </div>
                    </div>
                    {% endfor %}
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        {% endif %}

        <h2>Pipeline Metadata</h2>
        <table>
            <tbody>
                <tr>
                    <th>Output Directory</th>
                    <td>{{ metadata.output_directory }}</td>
                </tr>
                <tr>
                    <th>Pipeline Version</th>
                    <td>{{ metadata.pipeline_version }}</td>
                </tr>
                <tr>
                    <th>Image File</th>
                    <td>{{ input_info.image_file }}</td>
                </tr>
            </tbody>
        </table>

        <div class="note">Report generation completed successfully.</div>
    </div>

    <script>
    document.addEventListener('DOMContentLoaded', function () {
        var tabbedCards = document.querySelectorAll('.segmentation-card--tabbed');
        tabbedCards.forEach(function (card) {
            var tabs = card.querySelectorAll('.segmentation-tab');
            var panels = card.querySelectorAll('.segmentation-tab-panel');
            tabs.forEach(function (tab, tabIndex) {
                var targetId = tab.getAttribute('data-target');
                tab.setAttribute('aria-selected', tabIndex === 0 ? 'true' : 'false');
                tab.addEventListener('click', function () {
                    tabs.forEach(function (t) {
                        t.classList.remove('is-active');
                        t.setAttribute('aria-selected', 'false');
                    });
                    panels.forEach(function (panel) {
                        panel.classList.remove('is-active');
                    });
                    tab.classList.add('is-active');
                    tab.setAttribute('aria-selected', 'true');
                    var targetPanel = card.querySelector('#' + targetId);
                    if (targetPanel) {
                        targetPanel.classList.add('is-active');
                    }
                });
            });
        });
    });
    </script>
</body>
</html>
"""
        template = Template(template_str)
        context = {
            'metadata': self.report_data.get('metadata', {}),
            'input_info': self.report_data.get('input_info', {}),
            'patch_quality': self.report_data.get('patch_quality', {}),
            'figures': self.figures,
            'sample_preview': self.report_data.get('sample_preview', {'images': []}),
            'segmentation_visualization': self.report_data.get('segmentation_visualization', {}),
            'segmentation_stats': self.report_data.get('segmentation_stats', {}),
            'split_mode': self.report_data.get('split_mode', 'patches'),
        }
        return template.render(**context)

def generate_pipeline_report(output_dir: str, config: Dict, codex_patches=None) -> str:
    """
    Convenience function to generate a pipeline report.
    
    Args:
        output_dir: Pipeline output directory
        config: Pipeline configuration
        codex_patches: Optional CodexPatches object
        
    Returns:
        Path to generated report
    """
    generator = PipelineReportGenerator(output_dir, config, codex_patches)
    return generator.generate_report()
