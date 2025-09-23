"""
Report generator for PhenoCycler pipeline results.
Creates comprehensive HTML reports with analysis summaries and visualizations.
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from jinja2 import Template
import base64
from io import BytesIO
from collections import OrderedDict
try:
    from PIL import Image
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
        self._generate_summary_figures()
        
        # Generate HTML report
        html_content = self._render_html_report()
        
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
                        # Convert to base64 for embedding in HTML
                        try:
                            with open(preview_file, 'rb') as f:
                                img_data = base64.b64encode(f.read()).decode()
                                preview_images.append({
                                    'channel': i,
                                    'filename': preview_file.name,
                                    'data': f"data:image/jpeg;base64,{img_data}"
                                })
                                logger.info(f"Found preview image: {preview_file}")
                        except Exception as e:
                            logger.warning(f"Failed to load preview image {preview_file}: {e}")
                
                # If no channel-specific images found, look for a general preview
                if not preview_images:
                    general_preview = preview_dir / f"{base_name}_preview.jpg"
                    if general_preview.exists():
                        try:
                            with open(general_preview, 'rb') as f:
                                img_data = base64.b64encode(f.read()).decode()
                                preview_images.append({
                                    'channel': 'general',
                                    'filename': general_preview.name,
                                    'data': f"data:image/jpeg;base64,{img_data}"
                                })
                                logger.info(f"Found general preview image: {general_preview}")
                        except Exception as e:
                            logger.warning(f"Failed to load preview image {general_preview}: {e}")
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
        """Collect segmentation visualization images."""
        visualization_dir = self.output_dir / "visualization" / "segmentation"

        segmentation_images = []

        categories = OrderedDict([
            ('overview', {'title': 'Segmentation Overview', 'layout': 'wide'}),
            ('quality', {'title': 'Quality Diagnostics', 'layout': 'single'}),
            ('nucleus', {'title': 'Nucleus Mask Details', 'layout': 'paired'}),
            ('wholecell', {'title': 'Whole Cell Mask Details', 'layout': 'paired'}),
            ('statistics', {'title': 'Morphology Statistics', 'layout': 'single'}),
            ('other', {'title': 'Additional Visualizations', 'layout': 'paired'}),
        ])

        if visualization_dir.exists():
            # Define expected visualization files with descriptions
            expected_files = {
                'segmentation_overlay_patch_0.png': {
                    'title': 'Segmentation Overlay',
                    'description': 'Overlay of cell and nucleus masks (viridis/magma, area-scaled)',
                    'category': 'overview',
                    'tab_group': 'overlay_patch_0',
                    'tab_title': 'Repaired',
                    'card_title': 'Segmentation Overlay',
                    'tab_order': 0
                },
                'segmentation_overlay_pre_repair_patch_0.png': {
                    'title': 'Pre-Repair Segmentation Overlay',
                    'description': 'Original segmentation before mask repair (viridis/magma)',
                    'category': 'overview',
                    'tab_group': 'overlay_patch_0',
                    'tab_title': 'Pre-Repair',
                    'card_title': 'Segmentation Overlay',
                    'tab_order': 1
                },
                'segmentation_overlay_unmatched_nucleus_patch_0.png': {
                    'title': 'Unmatched Nuclei Overlay',
                    'description': 'Highlights nuclei removed during repair (yellow overlay)',
                    'category': 'overview',
                    'tab_group': 'overlay_patch_0',
                    'tab_title': 'Unmatched Nuclei',
                    'card_title': 'Segmentation Overlay',
                    'tab_order': 2
                },
                'segmentation_overlay_unmatched_cell_patch_0.png': {
                    'title': 'Unmatched Cell Overlay',
                    'description': 'Highlights cells removed during repair (cyan overlay)',
                    'category': 'overview',
                    'tab_group': 'overlay_patch_0',
                    'tab_title': 'Unmatched Cells',
                    'card_title': 'Segmentation Overlay',
                    'tab_order': 3
                },
                'segmentation_errors_patch_0.png': {
                    'title': 'Segmentation Errors',
                    'description': 'Visualization of segmentation quality and potential errors',
                    'category': 'quality'
                },
                'nucleus_mask_patch_0.png': {
                    'title': 'Nucleus Mask Visualization',
                    'description': 'Nucleus mask boundaries with color mapped to nucleus area',
                    'category': 'nucleus',
                    'tab_group': 'nucleus_patch_0',
                    'tab_title': 'Repaired',
                    'card_title': 'Nucleus Mask Visualization',
                    'tab_order': 0
                },
                'nucleus_mask_pre_repair_patch_0.png': {
                    'title': 'Pre-Repair Nucleus Mask',
                    'description': 'Original nucleus segmentation before mask repair',
                    'category': 'nucleus',
                    'tab_group': 'nucleus_patch_0',
                    'tab_title': 'Pre-Repair',
                    'card_title': 'Nucleus Mask Visualization',
                    'tab_order': 1
                },
                'wholecell_mask_patch_0.png': {
                    'title': 'Whole Cell Mask Visualization',
                    'description': 'Whole-cell mask boundaries with color mapped to cell area',
                    'category': 'wholecell',
                    'tab_group': 'wholecell_patch_0',
                    'tab_title': 'Repaired',
                    'card_title': 'Whole Cell Mask Visualization',
                    'tab_order': 0
                },
                'wholecell_mask_pre_repair_patch_0.png': {
                    'title': 'Pre-Repair Whole Cell Mask',
                    'description': 'Original whole-cell segmentation before mask repair',
                    'category': 'wholecell',
                    'tab_group': 'wholecell_patch_0',
                    'tab_title': 'Pre-Repair',
                    'card_title': 'Whole Cell Mask Visualization',
                    'tab_order': 1
                },
                'cell_morphology_stats.png': {
                    'title': 'Cell Morphology Statistics',
                    'description': 'Distribution of cell morphology metrics',
                    'category': 'statistics'
                }
            }

            for filename, info in expected_files.items():
                file_path = visualization_dir / filename
                if file_path.exists():
                    try:
                        # Convert to base64 for embedding in HTML
                        with open(file_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode()
                            segmentation_images.append({
                                'filename': filename,
                                'title': info['title'],
                                'description': info['description'],
                                'data': f"data:image/png;base64,{img_data}",
                                'category': info.get('category', 'other'),
                                'tab_group': info.get('tab_group'),
                                'tab_title': info.get('tab_title'),
                                'card_title': info.get('card_title', info['title']),
                                'tab_order': info.get('tab_order', 0)
                            })
                            logger.info(f"Found segmentation visualization: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to load visualization image {file_path}: {e}")
                else:
                    logger.debug(f"Visualization file not found: {filename}")
        else:
            logger.info(f"Segmentation visualization directory not found: {visualization_dir}")

        grouped_images = []
        for key, meta in categories.items():
            group_items = [img for img in segmentation_images if img.get('category', 'other') == key]

            tab_groups = OrderedDict()
            regular_images = []

            for img in group_items:
                tab_group = img.get('tab_group')
                if tab_group:
                    tab_entry = tab_groups.setdefault(
                        tab_group,
                        {
                            'card_title': img.get('card_title', img['title']),
                            'tabs': []
                        }
                    )
                    tab_entry['tabs'].append(
                        {
                            'title': img.get('tab_title', img['title']),
                            'image': img,
                            'order': img.get('tab_order', 0)
                        }
                    )
                else:
                    regular_images.append(img)

            tab_cards = []
            for tab_id, tab_info in tab_groups.items():
                sorted_tabs = sorted(tab_info['tabs'], key=lambda t: t['order'])
                card_title = tab_info.get('card_title') or sorted_tabs[0]['image'].get('title')
                tab_cards.append(
                    {
                        'card_id': tab_id,
                        'card_title': card_title,
                        'tabs': [
                            {
                                'title': tab['title'],
                                'image': tab['image']
                            }
                            for tab in sorted_tabs
                        ]
                    }
                )

            if regular_images or tab_cards:
                grouped_images.append({
                    'key': key,
                    'title': meta['title'],
                    'layout': meta['layout'],
                    'images': regular_images,
                    'tab_cards': tab_cards
                })

        global_tab_groups = OrderedDict()
        global_regular_images = []

        for img in segmentation_images:
            tab_group = img.get('tab_group')
            if tab_group:
                tab_entry = global_tab_groups.setdefault(
                    tab_group,
                    {
                        'card_title': img.get('card_title', img['title']),
                        'tabs': []
                    }
                )
                tab_entry['tabs'].append(
                    {
                        'title': img.get('tab_title', img['title']),
                        'image': img,
                        'order': img.get('tab_order', 0)
                    }
                )
            else:
                global_regular_images.append(img)

        global_tab_cards = []
        for tab_id, tab_info in global_tab_groups.items():
            sorted_tabs = sorted(tab_info['tabs'], key=lambda t: t['order'])
            card_title = tab_info.get('card_title') or sorted_tabs[0]['image'].get('title')
            global_tab_cards.append(
                {
                    'card_id': tab_id,
                    'card_title': card_title,
                    'tabs': [
                        {
                            'title': tab['title'],
                            'image': tab['image']
                        }
                        for tab in sorted_tabs
                    ]
                }
            )

        self.report_data['segmentation_visualization'] = {
            'images': segmentation_images,
            'groups': grouped_images,
            'total_images': len(segmentation_images),
            'visualization_dir': str(visualization_dir),
            'tab_cards': global_tab_cards,
            'regular_images': global_regular_images
        }

        if segmentation_images:
            logger.info(f"Collected {len(segmentation_images)} segmentation visualization images")
        else:
            logger.info("No segmentation visualization images found")
            
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
                        'description': f'Used for identifying whole cell boundaries (channel {i+1})'
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
                    'description': f'Used for identifying whole cell boundaries (channel {i+1})'
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
            
        # Load evaluation metrics if available
        eval_metrics_path = self.output_dir / "seg_evaluation_metrics.pkl"
        if eval_metrics_path.exists():
            import pickle
            with open(eval_metrics_path, 'rb') as f:
                eval_metrics = pickle.load(f)
                # Extract quality scores
                quality_scores = [m.get('QualityScore', np.nan) 
                                 for m in eval_metrics if m is not None]
                stats['mean_quality_score'] = np.nanmean(quality_scores)
        
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
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
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
        {% if segmentation_visualization.groups %}
        <h2>Segmentation Visualization</h2>
        <div class="summary-box">
            <p>Found {{ segmentation_visualization.total_images }} segmentation visualization image(s) from pipeline analysis.</p>
            {% for group in segmentation_visualization.groups %}
            <div class="segmentation-section">
                <h3>{{ group.title }}</h3>
                <div class="segmentation-gallery segmentation-gallery--{{ group.layout }}">
                    {% if group.tab_cards %}
                    {% for card in group.tab_cards %}
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
                    {% endif %}
                    {% for image in group.images %}
                    <figure class="segmentation-card">
                        <div class="segmentation-card__title">{{ image.title }}</div>
                        <img src="{{ image.data }}" alt="{{ image.filename }}">
                        <figcaption>
                            <span class="segmentation-filename">{{ image.filename }}</span>
                            {{ image.description }}
                        </figcaption>
                    </figure>
                    {% endfor %}
                </div>
            </div>
            {% endfor %}
        </div>
        {% elif segmentation_visualization.images %}
        <h2>Segmentation Visualization</h2>
        <div class="summary-box">
            <p>Found {{ segmentation_visualization.total_images }} segmentation visualization image(s) from pipeline analysis.</p>
            <div class="segmentation-gallery segmentation-gallery--paired">
                {% if segmentation_visualization.tab_cards %}
                {% for card in segmentation_visualization.tab_cards %}
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
                {% endif %}
                {% for image in segmentation_visualization.regular_images %}
                <figure class="segmentation-card">
                    <div class="segmentation-card__title">{{ image.title }}</div>
                    <img src="{{ image.data }}" alt="{{ image.filename }}">
                    <figcaption>
                        <span class="segmentation-filename">{{ image.filename }}</span>
                        {{ image.description }}
                    </figcaption>
                </figure>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        
        {% if figures.quality_dashboard %}
        <h2>Quality Metrics</h2>
        <div class="figure">
            <img src="{{ figures.quality_dashboard }}" alt="Quality Dashboard">
        </div>
        {% endif %}
        
        <h2>Segmentation Results</h2>
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
                <td>{{ "%.1f"|format(segmentation_stats.mean_cell_area|default(0)) }} pixels²</td>
            </tr>
            <tr>
                <td>Median Cell Area</td>
                <td>{{ "%.1f"|format(segmentation_stats.median_cell_area|default(0)) }} pixels²</td>
            </tr>
            <tr>
                <td>Quality Score</td>
                <td>{{ "%.3f"|format(segmentation_stats.mean_quality_score|default(0)) }}</td>
            </tr>
        </table>
        
        <h2>Mask Repair Details</h2>
        {% if mask_repair_stats %}
        <div class="summary-box">
            <h3>Summary Statistics</h3>
            <div class="metrics-grid">
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
        </div>
        {% else %}
        <div class="warning">
            <strong>No mask repair statistics available.</strong><br>
            This may indicate that mask repair was not performed or statistics were not collected during processing.
        </div>
        {% endif %}
        
        {% if expression_stats %}
        <h2>Marker Expression Statistics</h2>        
        {% if figures.expression_bars %}
        <div class="figure">
            <img src="{{ figures.expression_bars }}" alt="Expression Bar Charts">
            <p style="color: #666; font-size: 12px; margin-top: 10px;">
                Left: Mean expression levels (log-transformed) ranked from lowest to highest<br>
                Right: Positive cell percentages ranked from lowest to highest
            </p>
        </div>
        {% endif %}
        
        <div class="warning" style="background-color: #e8f4fd; border: 1px solid #bee5eb; color: #0c5460;">
            <strong>Positive Cell Definition:</strong><br>
            • <strong>Nuclear markers (DAPI, Hoechst):</strong> Cells with expression > median (most cells should be positive)<br>
            • <strong>Other markers:</strong> Cells with expression > median + 1×standard deviation<br>
            This approach provides more biologically meaningful thresholds than the previous mean + 2×std method.
        </div>
        
        <table>
            <tr>
                <th>Marker</th>
                <th>Mean</th>
                <th>Median</th>
                <th>Std Dev</th>
                <th>Positive Threshold</th>
                <th>Positive Cells (%)</th>
            </tr>
            {% for stat in expression_stats[:10] %}
            <tr>
                <td>{{ stat.marker }}</td>
                <td>{{ "%.2f"|format(stat.mean) }}</td>
                <td>{{ "%.2f"|format(stat.median) }}</td>
                <td>{{ "%.2f"|format(stat.std) }}</td>
                <td>{{ "%.2f"|format(stat.positive_threshold|default(0)) }}</td>
                <td>{{ "%.1f"|format(stat.positive_percentage) }}%</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        
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
