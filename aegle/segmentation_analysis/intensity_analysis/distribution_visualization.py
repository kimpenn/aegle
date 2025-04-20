import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import re
from scipy import ndimage

# Configure logging
logger = logging.getLogger(__name__)

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string to be used as a filename by replacing unsafe characters with underscores.
    
    Args:
        filename: The string to sanitize
        
    Returns:
        A sanitized string safe for use as a filename
    """
    # Replace any character that's not alphanumeric, underscore, or hyphen with underscore
    return re.sub(r'[^\w\-]', '_', filename)

def visualize_intensity_distributions(
    intensity_data: Dict,
    output_dir: Optional[str] = None,
    use_log_scale: bool = True,
    channel_names: Optional[List[str]] = None,
) -> None:
    """
    Visualize intensity distributions for cells and nuclei across channels.
    
    Args:
        intensity_data: Dictionary containing intensity analysis results from extract_intensity_data_across_channels
        output_dir: Directory to save plots (if None, plots will be displayed)
        use_log_scale: Whether to use log scale for intensity values
        channel_names: Optional list of channel names to visualize. If None, all channels will be used.
    """
    logger.info("Starting intensity distribution visualization")
    
    # Create output directory if needed
    if output_dir:
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # Get list of channel names
    if channel_names is None:
        channel_names = sorted(intensity_data.keys())
        logger.info(f"Using all available channels: {len(channel_names)} channels")
    else:
        # Validate that all requested channels exist in results
        missing_channels = [ch for ch in channel_names if ch not in intensity_data]
        if missing_channels:
            logger.warning(f"Some requested channels not found in results: {missing_channels}")
            channel_names = [ch for ch in channel_names if ch in intensity_data]
            if not channel_names:
                logger.error("No valid channels found in results")
                return
        logger.info(f"Using specified channels: {len(channel_names)} channels: {channel_names}")

    num_channels = len(channel_names)
    logger.info(f"Processing {num_channels} channels")

    for channel_idx, channel_name in enumerate(channel_names):
        logger.info(f"Visualizing distribution for channel {channel_idx + 1}/{num_channels}: {channel_name}")
        
        # Get intensity data for this channel
        channel_intensities = intensity_data[channel_name]["intensities"]
        
        # Check if unmatched nuclei intensities are available
        has_unmatched_nuclei = "nucleus_unmatched" in channel_intensities
        
        # Create figure with two or three subplots
        if has_unmatched_nuclei:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        fig.suptitle(f'Intensity Distribution - {channel_name}', fontsize=16)
        
        # Plot cell intensity distributions
        logger.debug(f"Plotting cell distributions for {channel_name}")
        if use_log_scale:
            # Cell subplot
            if len(channel_intensities["cell"]) > 0:
                sns.kdeplot(np.log1p(channel_intensities["cell"]), 
                            label="Original Cell", fill=True, alpha=0.3, ax=ax1)
            if len(channel_intensities["cell_matched"]) > 0:
                sns.kdeplot(np.log1p(channel_intensities["cell_matched"]), 
                            label="Repaired Cell", fill=True, alpha=0.3, ax=ax1)
            ax1.set_xlabel('Log(Intensity + 1)')
            
            # Nucleus subplot
            if len(channel_intensities["nucleus"]) > 0:
                sns.kdeplot(np.log1p(channel_intensities["nucleus"]), 
                            label="Original Nucleus", fill=True, alpha=0.3, ax=ax2)
            if len(channel_intensities["nucleus_matched"]) > 0:
                sns.kdeplot(np.log1p(channel_intensities["nucleus_matched"]), 
                            label="Repaired Nucleus", fill=True, alpha=0.3, ax=ax2)
            ax2.set_xlabel('Log(Intensity + 1)')
            
            # Unmatched nucleus subplot if available
            if has_unmatched_nuclei:
                if len(channel_intensities["nucleus_unmatched"]) > 0:
                    sns.kdeplot(np.log1p(channel_intensities["nucleus_unmatched"]), 
                                label="Unmatched Nucleus", fill=True, alpha=0.3, ax=ax3)
                ax3.set_xlabel('Log(Intensity + 1)')
        else:
            # Cell subplot
            if len(channel_intensities["cell"]) > 0:
                sns.kdeplot(channel_intensities["cell"], 
                            label="Original Cell", fill=True, alpha=0.3, ax=ax1)
            if len(channel_intensities["cell_matched"]) > 0:
                sns.kdeplot(channel_intensities["cell_matched"], 
                            label="Repaired Cell", fill=True, alpha=0.3, ax=ax1)
            ax1.set_xlabel('Intensity')
            
            # Nucleus subplot
            if len(channel_intensities["nucleus"]) > 0:
                sns.kdeplot(channel_intensities["nucleus"], 
                            label="Original Nucleus", fill=True, alpha=0.3, ax=ax2)
            if len(channel_intensities["nucleus_matched"]) > 0:
                sns.kdeplot(channel_intensities["nucleus_matched"], 
                            label="Repaired Nucleus", fill=True, alpha=0.3, ax=ax2)
            ax2.set_xlabel('Intensity')
            
            # Unmatched nucleus subplot if available
            if has_unmatched_nuclei:
                if len(channel_intensities["nucleus_unmatched"]) > 0:
                    sns.kdeplot(channel_intensities["nucleus_unmatched"], 
                                label="Unmatched Nucleus", fill=True, alpha=0.3, ax=ax3)
                ax3.set_xlabel('Intensity')

        # Set titles and labels
        ax1.set_title('Cell Intensity Distribution')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        ax2.set_title('Nucleus Intensity Distribution')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        if has_unmatched_nuclei:
            ax3.set_title('Unmatched Nucleus Intensity Distribution')
            ax3.set_ylabel('Density')
            ax3.legend()
        
        plt.tight_layout()
        
        if output_dir:
            # Sanitize the channel name for use in filename
            safe_channel_name = sanitize_filename(channel_name)
            file_name = f"{channel_idx:02d}-intensity_distribution_{safe_channel_name}.png"
            file_path = os.path.join(output_dir, file_name)
            logger.info(f"Saving plot to {file_path}")
            plt.savefig(file_path, dpi=300)
            plt.close()
        else:
            logger.info("Displaying plot (no output directory specified)")
            plt.show()

    logger.info("Completed intensity distribution visualization") 