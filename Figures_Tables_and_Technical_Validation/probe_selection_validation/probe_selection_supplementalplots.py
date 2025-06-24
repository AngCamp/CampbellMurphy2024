#!/usr/bin/env python3
"""
Probe Selection Validation Visualization Script

This script analyzes channel selection metadata from Neuropixels probes
and creates visualization plots showing ripple band and sharp wave features
as a function of depth relative to selected channels.
"""

import os
import json
import gzip
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION CONTROL
# =============================================================================

# Set this to True to use environment variables, False to use hardcoded values
USE_ENV_VARS = False

# =============================================================================
# GLOBAL CONFIGURATION SETTINGS
# =============================================================================

if USE_ENV_VARS:
    # Use environment variables (set these in your shell: export BASE_DIR="/path/to/data")
    BASE_DIRECTORY = os.environ.get('BASE_DIR', '/default/fallback/path')
    RUN_NAME = os.environ.get('RUN_NAME', 'default_run')
else:
    # Use hardcoded values - EDIT THESE SETTINGS
    BASE_DIRECTORY = "yourpath/SWR_final_pipeline/osf_campbellmurphy2025_swr_data_backup"  # CHANGE THIS
    RUN_NAME = "supplemental_channel_selection_plots"  # CHANGE THIS

# Rest of your existing configuration...
MIN_PROBE_COUNT = 3

DATASETS = [
    'allen_visbehave_swr_murphylab2024',
    'allen_viscoding_swr_murphylab2024', 
    'ibl_swr_murphylab2024'
]

OUTPUT_FORMATS = ['png', 'svg']
DEPTH_BIN_SIZE = 20
FIGURE_SIZE = (15, 10)
DPI = 300

# =============================================================================
# NEW DEPTH LIMIT CONFIGURATION
# =============================================================================

# Enable/disable depth limits
USE_DEPTH_LIMITS = True

# Depth limits in micrometers (relative to selected channel)
# For ripple features: both positive and negative directions from selected channel
RIPPLE_DEPTH_LIMIT_UP = 1000    # Maximum depth above selected channel (positive values)
RIPPLE_DEPTH_LIMIT_DOWN = 1000  # Maximum depth below selected channel (negative values, will be checked as absolute value)

# For sharp wave features: typically only below the ripple channel (negative values)
SHARP_WAVE_DEPTH_LIMIT = 800    # Maximum depth below ripple channel (negative values, will be checked as absolute value)

# Bar thickness for plots (in units of the y-axis, i.e., micrometers)
BAR_HEIGHT = DEPTH_BIN_SIZE * 0.8  # Make bars slightly smaller than bin size to show gaps

# =============================================================================
# CHANNEL SMOOTHING CONFIGURATION
# =============================================================================

# Enable/disable anatomical channel smoothing (applied before z-scoring)
USE_CHANNEL_SMOOTHING = False

# Smoothing parameter (sigma in micrometers for Gaussian kernel)
CHANNEL_SMOOTHING_SIGMA = 50.0

# =============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# =============================================================================

def load_channel_metadata(file_path):
    """Load and parse channel selection metadata from json.gz file."""
    try:
        with gzip.open(file_path, 'rt') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def find_metadata_files(base_directory):
    """Find all channel selection metadata files in the specified datasets."""
    metadata_files = []
    base_path = Path(base_directory)
    
    for dataset in DATASETS:
        dataset_path = base_path / dataset
        if not dataset_path.exists():
            print(f"Warning: Dataset directory {dataset_path} not found")
            continue
            
        # Find session directories
        session_dirs = glob.glob(str(dataset_path / "swrs_session_*"))
        
        for session_dir in session_dirs:
            # Find metadata files in each session
            metadata_pattern = os.path.join(session_dir, "probe_*_channel_selection_metadata.json.gz")
            session_files = glob.glob(metadata_pattern)
            
            for file_path in session_files:
                # Extract identifiers from filename
                filename = os.path.basename(file_path)
                probe_id = filename.split('_')[1]
                session_id = os.path.basename(session_dir).replace('swrs_session_', '')
                
                metadata_files.append({
                    'file_path': file_path,
                    'dataset': dataset,
                    'session_id': session_id,
                    'probe_id': probe_id,
                    'filename': filename
                })
    
    return metadata_files

def bin_depth(depth, bin_size=DEPTH_BIN_SIZE):
    """Bin depth values to specified resolution."""
    return int(np.round(depth / bin_size) * bin_size)

def apply_depth_filter(relative_depths, use_limits=USE_DEPTH_LIMITS, 
                      limit_up=RIPPLE_DEPTH_LIMIT_UP, limit_down=RIPPLE_DEPTH_LIMIT_DOWN):
    """Apply depth limits to filter out unrealistic depth values."""
    if not use_limits:
        return np.ones(len(relative_depths), dtype=bool)
    
    mask = np.ones(len(relative_depths), dtype=bool)
    
    for i, depth in enumerate(relative_depths):
        if depth > 0:  # Above selected channel
            if depth > limit_up:
                mask[i] = False
        else:  # Below selected channel
            if abs(depth) > limit_down:
                mask[i] = False
    
    return mask

def process_ripple_data(metadata_files):
    """Process ripple band data from all metadata files."""
    ripple_data = defaultdict(list)
    processed_probes = []
    skipped_probes = []
    depth_filtered_count = 0
    
    for file_info in metadata_files:
        data = load_channel_metadata(file_info['file_path'])
        
        if data is None or 'ripple_band' not in data:
            skipped_probes.append(f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']}")
            continue
            
        try:
            ripple_band = data['ripple_band']
            selected_channel_id = ripple_band['selected_channel_id']
            
            # Find the depth of the selected channel
            channel_ids = ripple_band['channel_ids']
            depths = ripple_band['depths']
            skewness = ripple_band['skewness']
            net_power = ripple_band['net_power']
            
            if selected_channel_id not in channel_ids:
                skipped_probes.append(f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']} - selected channel not in list")
                continue
                
            selected_idx = channel_ids.index(selected_channel_id)
            selected_depth = depths[selected_idx]
            
            # Calculate relative depths and z-score within probe
            relative_depths = [d - selected_depth for d in depths]
            z_skewness = stats.zscore(skewness)
            z_net_power = stats.zscore(net_power)
            
            # Apply depth filter if enabled
            depth_mask = apply_depth_filter(relative_depths, USE_DEPTH_LIMITS, 
                                          RIPPLE_DEPTH_LIMIT_UP, RIPPLE_DEPTH_LIMIT_DOWN)
            depth_filtered_count += np.sum(~depth_mask)
            
            # Bin and collect data for channels that pass depth filter
            for i, (rel_depth, z_skew, z_power) in enumerate(zip(relative_depths, z_skewness, z_net_power)):
                if depth_mask[i]:  # Only include if passes depth filter
                    binned_depth = bin_depth(rel_depth)
                    ripple_data[binned_depth].append({
                        'skewness': z_skew,
                        'net_power': z_power,
                        'probe_info': f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']}"
                    })
            
            processed_probes.append(f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']}")
            
        except Exception as e:
            skipped_probes.append(f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']} - {str(e)}")
    
    if USE_DEPTH_LIMITS and depth_filtered_count > 0:
        print(f"Ripple data: Filtered out {depth_filtered_count} channels due to depth limits (±{RIPPLE_DEPTH_LIMIT_UP}/{RIPPLE_DEPTH_LIMIT_DOWN} μm)")
    
    return ripple_data, processed_probes, skipped_probes

def process_sharp_wave_data(metadata_files):
    """Process sharp wave band data from all metadata files."""
    sw_data = defaultdict(list)
    sw_selections = {
        'net_sw_power': [],
        'modulation_index': [], 
        'circular_linear_corr': []
    }  # Track selected channel depths for each metric
    processed_probes = []
    skipped_probes = []
    depth_filtered_count = 0
    
    for file_info in metadata_files:
        data = load_channel_metadata(file_info['file_path'])
        
        if data is None or 'sharp_wave_band' not in data or 'ripple_band' not in data:
            skipped_probes.append(f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']}")
            continue
            
        try:
            ripple_band = data['ripple_band']
            sharp_wave_band = data['sharp_wave_band']
            
            # Get ripple channel depth as reference
            ripple_selected_id = ripple_band['selected_channel_id']
            ripple_channel_ids = ripple_band['channel_ids']
            ripple_depths = ripple_band['depths']
            
            if ripple_selected_id not in ripple_channel_ids:
                skipped_probes.append(f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']} - ripple channel not found")
                continue
                
            ripple_idx = ripple_channel_ids.index(ripple_selected_id)
            ripple_depth = ripple_depths[ripple_idx]
            
            # Process sharp wave data
            sw_depths = sharp_wave_band['depths']
            sw_net_power = sharp_wave_band['net_sw_power']
            sw_mod_index = sharp_wave_band['modulation_index']
            sw_corr = sharp_wave_band['circular_linear_corrs']
            
            # Get selected sharp wave channel info if available
            selected_sw_channel_id = sharp_wave_band.get('selected_channel_id', None)
            selected_sw_depth = None
            if selected_sw_channel_id is not None:
                sw_channel_ids = sharp_wave_band.get('channel_ids', [])
                if selected_sw_channel_id in sw_channel_ids:
                    selected_idx = sw_channel_ids.index(selected_sw_channel_id)
                    selected_sw_depth = sw_depths[selected_idx] - ripple_depth  # Relative to ripple
            
            # Calculate relative depths
            relative_depths = [d - ripple_depth for d in sw_depths]
            
            # Apply anatomical smoothing if enabled (before z-scoring)
            if USE_CHANNEL_SMOOTHING:
                sw_net_power = anatomical_smooth_values(sw_net_power, sw_depths, CHANNEL_SMOOTHING_SIGMA)
                sw_mod_index = anatomical_smooth_values(sw_mod_index, sw_depths, CHANNEL_SMOOTHING_SIGMA)
                sw_corr = anatomical_smooth_values(sw_corr, sw_depths, CHANNEL_SMOOTHING_SIGMA)
            
            # Z-score within probe (after smoothing if enabled)
            z_net_power = stats.zscore(sw_net_power)
            z_mod_index = stats.zscore(sw_mod_index)
            z_corr = stats.zscore(sw_corr)
            
            # Apply depth filter using absolute values (accept both positive and negative relative depths)
            for i, (rel_depth, z_power, z_mod, z_cor) in enumerate(zip(relative_depths, z_net_power, z_mod_index, z_corr)):
                # Apply depth limit filter using absolute value
                if USE_DEPTH_LIMITS and abs(rel_depth) > SHARP_WAVE_DEPTH_LIMIT:
                    depth_filtered_count += 1
                    continue
                
                # Channel passes depth filter - add to data
                binned_depth = bin_depth(rel_depth)
                sw_data[binned_depth].append({
                    'net_sw_power': z_power,
                    'modulation_index': z_mod,
                    'circular_linear_corr': z_cor,
                    'probe_info': f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']}"
                })
            
            # Simulate selections for each metric (find channel with highest z-scored value)
            probe_info = f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']}"
            
            # For net_sw_power: select channel with highest z-scored net SW power
            max_power_idx = np.argmax(z_net_power)
            selected_power_depth = relative_depths[max_power_idx]
            sw_selections['net_sw_power'].append({
                'depth': selected_power_depth,
                'probe_info': probe_info
            })
            
            # For modulation_index: select channel with highest z-scored modulation index
            max_mod_idx = np.argmax(z_mod_index)
            selected_mod_depth = relative_depths[max_mod_idx]
            sw_selections['modulation_index'].append({
                'depth': selected_mod_depth,
                'probe_info': probe_info
            })
            
            # For circular_linear_corr: select channel with highest z-scored correlation
            max_corr_idx = np.argmax(z_corr)
            selected_corr_depth = relative_depths[max_corr_idx]
            sw_selections['circular_linear_corr'].append({
                'depth': selected_corr_depth,
                'probe_info': probe_info
            })
            
            processed_probes.append(f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']}")
            
        except Exception as e:
            skipped_probes.append(f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']} - {str(e)}")
    
    # Report filtering results
    if USE_DEPTH_LIMITS and depth_filtered_count > 0:
        print(f"Sharp wave data: Filtered out {depth_filtered_count} channels due to depth limits (>{SHARP_WAVE_DEPTH_LIMIT} μm from ripple channel)")
    
    # Debug: Show final data summary
    if sw_data:
        all_sw_depths = list(sw_data.keys())
        print(f"Sharp wave data depth range: {min(all_sw_depths)} to {max(all_sw_depths)} μm")
        print(f"Total depth bins collected: {len(all_sw_depths)}")
    else:
        print(f"WARNING: No sharp wave data collected!")
    
    return sw_data, sw_selections, processed_probes, skipped_probes

def filter_by_probe_count(data_dict, min_count=MIN_PROBE_COUNT):
    """Filter depths that don't meet minimum probe count threshold."""
    filtered_data = {}
    for depth, values in data_dict.items():
        if len(values) >= min_count:
            filtered_data[depth] = values
    return filtered_data

def calculate_mean_values(data_dict):
    """Calculate mean values for each depth."""
    means = {}
    for depth, values in data_dict.items():
        means[depth] = {}
        if values:  # Check if values exist
            # Get all keys from first value
            keys = values[0].keys()
            for key in keys:
                if key != 'probe_info':  # Skip metadata
                    means[depth][key] = np.mean([v[key] for v in values])
    return means

def anatomical_smooth_values(raw_values, channel_depths, sigma=CHANNEL_SMOOTHING_SIGMA):
    """
    Apply anatomical smoothing using Gaussian weights based on channel depth distances.
    
    Parameters:
    - raw_values: Array of raw feature values for each channel
    - channel_depths: Array of depth positions for each channel (in micrometers)
    - sigma: Standard deviation of Gaussian kernel (in micrometers)
    
    Returns:
    - smoothed_values: Array of smoothed feature values
    """
    raw_values = np.array(raw_values)
    channel_depths = np.array(channel_depths)
    smoothed_values = np.zeros_like(raw_values)
    
    for i in range(len(raw_values)):
        # Calculate anatomical distances from current channel
        anatomical_distances = np.abs(channel_depths - channel_depths[i])
        
        # Calculate Gaussian weights
        weights = np.exp(-anatomical_distances**2 / (2 * sigma**2))
        
        # Apply weighted averaging
        smoothed_values[i] = np.sum(weights * raw_values) / np.sum(weights)
    
    return smoothed_values

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_ripple_features(ripple_means, output_dir):
    """Create ripple band feature plots."""
    if not ripple_means:
        print("No ripple data to plot")
        return
        
    # Multiply depths by -1 for anatomical orientation (deeper = more negative)
    depths = sorted([-d for d in ripple_means.keys()])  # Convert to negative and sort
    skewness_values = [ripple_means[-d]['skewness'] for d in depths]  # Use original keys
    net_power_values = [ripple_means[-d]['net_power'] for d in depths]  # Use original keys
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharey=True)
    
    # Skewness plot
    ax1.barh(depths, skewness_values, height=BAR_HEIGHT, color='steelblue', alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Z-scored Skewness')
    ax1.set_ylabel('Depth Relative to Selected Channel (μm)')
    ax1.set_title('Skewness')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # Net Power plot
    ax2.barh(depths, net_power_values, height=BAR_HEIGHT, color='coral', alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax2.set_xlabel('Z-scored Net Power')
    ax2.set_title('Net Ripple Power')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # Add selected channel line label
    ax1.text(0.02, 0.98, 'Selected Channel', transform=ax1.transAxes, 
             verticalalignment='top', fontsize=10, color='red')
    ax2.text(0.02, 0.98, 'Selected Channel', transform=ax2.transAxes, 
             verticalalignment='top', fontsize=10, color='red')
    
    # Add depth limit info to title if enabled
    title = 'Mean Z-scored Ripple Features by Depth (Relative to Selected Channel)'
    if USE_DEPTH_LIMITS:
        title += f'\nDepth limits: ±{RIPPLE_DEPTH_LIMIT_UP}/{RIPPLE_DEPTH_LIMIT_DOWN} μm'
    plt.suptitle(title, fontsize=14, y=0.95)
    plt.tight_layout()
    
    # Save in specified formats
    for fmt in OUTPUT_FORMATS:
        output_file = os.path.join(output_dir, f'ripple_features_by_depth.{fmt}')
        plt.savefig(output_file, format=fmt, dpi=DPI, bbox_inches='tight')
        print(f"Saved ripple plot: {output_file}")
    
    plt.show()

def plot_sharp_wave_features(sw_means, output_dir):
    """Create sharp wave feature plots."""
    print(f"Sharp wave plotting: {len(sw_means)} depth bins available")
    
    if not sw_means:
        print("No sharp wave data to plot")
        return
    
    # Multiply depths by -1 for anatomical orientation (deeper = more negative)
    original_depths = list(sw_means.keys())
    depths = sorted([-d for d in original_depths])  # Convert to negative and sort
    print(f"Sharp wave depth range: {min(depths)} to {max(depths)} μm relative to ripple channel")
    
    try:
        net_power_values = [sw_means[-d]['net_sw_power'] for d in depths]  # Use original keys
        mod_index_values = [sw_means[-d]['modulation_index'] for d in depths]  # Use original keys  
        corr_values = [sw_means[-d]['circular_linear_corr'] for d in depths]  # Use original keys
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
        
        # Net SW Power plot
        ax1.barh(depths, net_power_values, height=BAR_HEIGHT, color='darkgreen', alpha=0.7)
        ax1.set_xlabel('Z-scored Net SW Power')
        ax1.set_ylabel('Depth Relative to Selected Channel (μm)')
        ax1.set_title('Net SW Power')
        ax1.grid(True, alpha=0.3)
        
        # Modulation Index plot
        ax2.barh(depths, mod_index_values, height=BAR_HEIGHT, color='purple', alpha=0.7)
        ax2.set_xlabel('Z-scored Modulation Index')
        ax2.set_title('Modulation Index')
        ax2.grid(True, alpha=0.3)
        
        # Circular-Linear Correlation plot
        ax3.barh(depths, corr_values, height=BAR_HEIGHT, color='orange', alpha=0.7)
        ax3.set_xlabel('Z-scored Circular-Linear Correlation')
        ax3.set_title('Circular-Linear Correlation')
        ax3.grid(True, alpha=0.3)
        
        # Add a reference line at y=0 to show the ripple channel
        for ax in [ax1, ax2, ax3]:
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # Add reference line labels
        ax1.text(0.02, 0.98, 'Ripple Channel (Reference)', transform=ax1.transAxes, 
                 verticalalignment='top', fontsize=10, color='red')
        
        # Add depth limit info to title if enabled
        title = 'Mean Z-scored Sharp Wave Features by Depth (Relative to Selected Ripple Channel)'
        if USE_DEPTH_LIMITS:
            title += f'\nDepth limit: ±{SHARP_WAVE_DEPTH_LIMIT} μm from ripple channel'
        plt.suptitle(title, fontsize=14, y=0.95)
        plt.tight_layout()
        
        # Save in specified formats
        for fmt in OUTPUT_FORMATS:
            output_file = os.path.join(output_dir, f'sharp_wave_features_by_depth.{fmt}')
            plt.savefig(output_file, format=fmt, dpi=DPI, bbox_inches='tight')
            print(f"Saved sharp wave plot: {output_file}")
        
        plt.show()
        print(f"Sharp wave plot completed successfully")
        
    except Exception as e:
        print(f"Error during sharp wave plotting: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def plot_sharp_wave_selections(sw_selections, output_dir):
    """Create histograms showing selected sharp wave channel depths for each metric."""
    print(f"Sharp wave selection plotting for 3 metrics...")
    
    if not sw_selections or not any(sw_selections.values()):
        print("No sharp wave selection data to plot")
        return
    
    # Define colors and titles for each metric
    metric_info = {
        'net_sw_power': {'color': 'darkgreen', 'title': 'Net SW Power'},
        'modulation_index': {'color': 'purple', 'title': 'Modulation Index'},
        'circular_linear_corr': {'color': 'orange', 'title': 'Circular-Linear Correlation'}
    }
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharey=True)
    
    for i, (metric, info) in enumerate(metric_info.items()):
        ax = axes[i]
        selections = sw_selections[metric]
        
        if not selections:
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=16)
            ax.set_title(info['title'])
            continue
        
        # Extract depths and bin them
        depths = [sel['depth'] for sel in selections]
        binned_depths = [bin_depth(d) for d in depths]
        
        # Count frequency of each binned depth
        from collections import Counter
        depth_counts = Counter(binned_depths)
        
        if not depth_counts:
            ax.text(0.5, 0.5, 'No Depth Counts', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=16)
            ax.set_title(info['title'])
            continue
        
        # Convert to sorted lists for plotting (anatomical orientation)
        original_depths = list(depth_counts.keys())
        plot_depths = sorted([-d for d in original_depths])  # Convert to negative and sort
        counts = [depth_counts[-d] for d in plot_depths]  # Use original keys for counts
        
        # Create horizontal bar chart
        bars = ax.barh(plot_depths, counts, height=BAR_HEIGHT, 
                      color=info['color'], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add reference line at y=0 to show the ripple channel
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Formatting
        ax.set_xlabel('Number of Selected Channels')
        if i == 0:  # Only add ylabel to leftmost plot
            ax.set_ylabel('Depth Relative to Ripple Channel (μm)')
        ax.set_title(info['title'])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add reference line label
        ax.text(0.02, 0.98, 'Ripple Channel', transform=ax.transAxes, 
                 verticalalignment='top', fontsize=9, color='red', fontweight='bold')
        
        # Add statistics
        total_selections = len(selections)
        depth_range = max(plot_depths) - min(plot_depths) if len(plot_depths) > 1 else 0
        median_depth = np.median(depths)
        
        stats_text = f'N: {total_selections}\n'
        stats_text += f'Range: {depth_range:.0f} μm\n'
        stats_text += f'Median: {median_depth:.0f} μm'
        
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
                verticalalignment='bottom', horizontalalignment='right', 
                fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        print(f"{info['title']}: {len(plot_depths)} depth bins, median = {median_depth:.0f} μm")
    
    # Add overall title
    title = 'Sharp Wave Channel Selection by Metric\n(Channels with Highest Z-scored Values)'
    if USE_DEPTH_LIMITS:
        title += f'\nDepth limit: ±{SHARP_WAVE_DEPTH_LIMIT} μm from ripple channel'
    if USE_CHANNEL_SMOOTHING:
        title += f' | Smoothing: σ={CHANNEL_SMOOTHING_SIGMA} μm'
    
    plt.suptitle(title, fontsize=14, y=0.95)
    plt.tight_layout()
    
    # Save in specified formats
    for fmt in OUTPUT_FORMATS:
        output_file = os.path.join(output_dir, f'sharp_wave_selection_by_metric.{fmt}')
        plt.savefig(output_file, format=fmt, dpi=DPI, bbox_inches='tight')
        print(f"Saved selection by metric plot: {output_file}")
    
    plt.show()
    print(f"Sharp wave selection by metric plots completed successfully")

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main(base_directory, output_directory=None, run_name=None):
    """Main function to run the analysis."""
    
    if output_directory is None:
        # Create timestamped run folder in current working directory
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name is None:
            run_name = RUN_NAME
        folder_name = f"{run_name}_{timestamp}"
        output_directory = os.path.join(os.getcwd(), folder_name)
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"Analyzing data from: {base_directory}")
    print(f"Looking for datasets: {DATASETS}")
    print(f"Minimum probe count threshold: {MIN_PROBE_COUNT}")
    print(f"Output formats: {OUTPUT_FORMATS}")
    print(f"Depth bin size: {DEPTH_BIN_SIZE} μm")
    print(f"Bar height: {BAR_HEIGHT} μm")
    
    # Print depth limit settings
    if USE_DEPTH_LIMITS:
        print(f"Depth limits enabled:")
        print(f"  Ripple features: ±{RIPPLE_DEPTH_LIMIT_UP}/{RIPPLE_DEPTH_LIMIT_DOWN} μm from selected channel")
        print(f"  Sharp wave features: {SHARP_WAVE_DEPTH_LIMIT} μm below ripple channel")
    else:
        print("Depth limits disabled")
    
    # Print channel smoothing settings
    if USE_CHANNEL_SMOOTHING:
        print(f"Channel smoothing enabled: σ = {CHANNEL_SMOOTHING_SIGMA} μm")
    else:
        print("Channel smoothing disabled")
    
    print("-" * 60)
    
    # Find all metadata files
    metadata_files = find_metadata_files(base_directory)
    print(f"Found {len(metadata_files)} metadata files")
    
    if not metadata_files:
        print("No metadata files found. Check your base directory and dataset names.")
        return
    
    # Process ripple data
    print("\nProcessing ripple band data...")
    ripple_data, ripple_processed, ripple_skipped = process_ripple_data(metadata_files)
    
    print(f"Successfully processed {len(ripple_processed)} probes for ripple analysis")
    if ripple_skipped:
        print(f"Skipped {len(ripple_skipped)} probes:")
        for skipped in ripple_skipped:
            print(f"  - {skipped}")
    
    # Filter and calculate means for ripple data
    ripple_filtered = filter_by_probe_count(ripple_data)
    ripple_means = calculate_mean_values(ripple_filtered)
    
    print(f"Ripple data: {len(ripple_data)} total depths, {len(ripple_filtered)} after filtering (≥{MIN_PROBE_COUNT} probes)")
    
    # Process sharp wave data
    print("\nProcessing sharp wave band data...")
    sw_data, sw_selections, sw_processed, sw_skipped = process_sharp_wave_data(metadata_files)
    
    print(f"Successfully processed {len(sw_processed)} probes for sharp wave analysis")
    if sw_skipped:
        print(f"Skipped {len(sw_skipped)} probes:")
        for skipped in sw_skipped:
            print(f"  - {skipped}")
    
    # Filter and calculate means for sharp wave data
    sw_filtered = filter_by_probe_count(sw_data)
    sw_means = calculate_mean_values(sw_filtered)
    
    print(f"Sharp wave data: {len(sw_data)} total depths, {len(sw_filtered)} after filtering (≥{MIN_PROBE_COUNT} probes)")
    
    if sw_means:
        depth_range = f"{min(sw_means.keys())} to {max(sw_means.keys())} μm"
        print(f"Sharp wave final data: {len(sw_means)} depth bins, range: {depth_range}")
    
    # Create plots
    print(f"\nGenerating plots in {output_directory}...")
    plot_ripple_features(ripple_means, output_directory)
    plot_sharp_wave_features(sw_means, output_directory)
    plot_sharp_wave_selections(sw_selections, output_directory)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    import sys
    import os
    
    # Check if command line arguments are provided
    if len(sys.argv) >= 2:
        # Use command line arguments
        base_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        run_name = sys.argv[3] if len(sys.argv) > 3 else None
        
        print("Running with command line arguments...")
        print(f"Base directory: {base_dir}")
        main(base_dir, output_dir, run_name)
        
    else:
        # Use configuration settings from top of file
        print("No command line arguments provided.")
        print(f"USE_ENV_VARS = {USE_ENV_VARS}")
        
        if USE_ENV_VARS:
            print("Using environment variables...")
            if 'BASE_DIR' not in os.environ:
                print("WARNING: BASE_DIR environment variable not set!")
                print("Set it with: export BASE_DIR='/path/to/your/data'")
        else:
            print("Using hardcoded configuration settings...")
        
        print(f"BASE_DIRECTORY = {BASE_DIRECTORY}")
        print(f"RUN_NAME = {RUN_NAME}")
        
        # Check if directory exists and provide helpful feedback
        if os.path.exists(BASE_DIRECTORY):
            print(f"✓ Base directory exists: {BASE_DIRECTORY}")
            main(BASE_DIRECTORY, None, RUN_NAME)
        else:
            print(f"✗ Base directory does not exist: {BASE_DIRECTORY}")
            
            # Try to find similar directories
            parent_dir = os.path.dirname(BASE_DIRECTORY)
            if os.path.exists(parent_dir):
                print(f"Parent directory exists: {parent_dir}")
                print("Contents:")
                try:
                    for item in os.listdir(parent_dir):
                        print(f"  - {item}")
                except PermissionError:
                    print("  (Permission denied to list contents)")
            else:
                print(f"Parent directory also doesn't exist: {parent_dir}")
            
            print("\nOptions to fix this:")
            print("1. Update BASE_DIRECTORY in the script to the correct path")
            print("2. Create the missing directory structure")
            print("3. Use command line: python script.py /correct/path/to/data")
            if USE_ENV_VARS:
                print("4. Set environment variable: export BASE_DIR='/correct/path'") 