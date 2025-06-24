#!/usr/bin/env python3
"""
Simplified Probe Selection Validation Visualization Script

This script analyzes channel selection metadata from Neuropixels probes
and creates simplified visualization plots showing only net ripple power 
and net sharp wave power as a function of depth relative to selected channels.
"""

import os
import json
import gzip
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
mpl.rcParams['svg.fonttype'] = 'none' # Ensure SVG text stays as editable text, not paths
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
    BASE_DIRECTORY = "yourpath/SWR_final_pipeline/osf_campbellmurphy2025_v2_final"  # CHANGE THIS
    RUN_NAME = "channel_selection_plots_figure_5b_and_c"  # CHANGE THIS

# Rest of configuration...
MIN_PROBE_COUNT = 10  # Require 10 channels minimum

DATASETS = [
    'allen_visbehave_swr_murphylab2024',
    'allen_viscoding_swr_murphylab2024', 
    'ibl_swr_murphylab2024'
]

OUTPUT_FORMATS = ['svg', 'png']  # Include SVG for font size control
DEPTH_BIN_SIZE = 20
DPI = 300

# =============================================================================
# DEPTH LIMIT CONFIGURATION
# =============================================================================

# Enable depth limits
USE_DEPTH_LIMITS = True

# Depth limits in micrometers (corrected as requested)
RIPPLE_DEPTH_LIMIT_UP = 500     # Maximum depth above selected channel for ripple
RIPPLE_DEPTH_LIMIT_DOWN = 500   # Maximum depth below selected channel for ripple
SHARP_WAVE_DEPTH_LIMIT = 500    # Maximum depth below ripple channel (only negative)

# Bar thickness for plots
BAR_HEIGHT = DEPTH_BIN_SIZE * 0.8

# Plot formatting
FONT_SIZE = 8
RIPPLE_Y_AXIS_LIMITS = [-500, 500]  # Ripple plot: full range (both positive/negative)
SW_Y_AXIS_LIMITS = [-500, -DEPTH_BIN_SIZE]        # Sharp wave plot: only negative values (below ripple)
Y_TICK_INTERVAL = 250  # Axis ticks every 250 μm

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
            net_power = ripple_band['net_power']
            
            if selected_channel_id not in channel_ids:
                skipped_probes.append(f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']} - selected channel not in list")
                continue
                
            selected_idx = channel_ids.index(selected_channel_id)
            selected_depth = depths[selected_idx]
            
            # Calculate relative depths and z-score within probe
            relative_depths = [d - selected_depth for d in depths]
            z_net_power = stats.zscore(net_power)
            
            # Apply depth filter if enabled
            depth_mask = apply_depth_filter(relative_depths, USE_DEPTH_LIMITS, 
                                          RIPPLE_DEPTH_LIMIT_UP, RIPPLE_DEPTH_LIMIT_DOWN)
            depth_filtered_count += np.sum(~depth_mask)
            
            # Bin and collect data for channels that pass depth filter
            for i, (rel_depth, z_power) in enumerate(zip(relative_depths, z_net_power)):
                if depth_mask[i]:  # Only include if passes depth filter
                    binned_depth = bin_depth(rel_depth)
                    ripple_data[binned_depth].append({
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
    sw_selections = []  # Track selected channel depths for Net SW Power within 500μm
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
            sw_net_power = sharp_wave_band['modulation_index'] #'net_sw_power', 'modulation_index', or 'circular_linear_corr'
            
            # Calculate relative depths
            relative_depths = [d - ripple_depth for d in sw_depths]
            
            # Z-score within probe
            z_net_power = stats.zscore(sw_net_power)
            
            # Apply depth filter using absolute values (accept both positive and negative relative depths)
            for i, (rel_depth, z_power) in enumerate(zip(relative_depths, z_net_power)):
                # Apply depth limit filter using absolute value
                if USE_DEPTH_LIMITS and abs(rel_depth) > SHARP_WAVE_DEPTH_LIMIT:
                    depth_filtered_count += 1
                    continue
                
                # Channel passes depth filter - add to data
                binned_depth = bin_depth(rel_depth)
                sw_data[binned_depth].append({
                    'net_sw_power': z_power,
                    'probe_info': f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']}"
                })
            
            # Modified selection: find channel with highest net SW power within 500μm of ripple channel
            probe_info = f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']}"
            
            # Filter channels within 500μm of ripple channel
            within_500um_mask = [abs(rel_depth) <= 500 for rel_depth in relative_depths]
            if any(within_500um_mask):
                # Get z-scored powers for channels within 500μm
                filtered_powers = [z_net_power[i] for i, mask in enumerate(within_500um_mask) if mask]
                filtered_depths = [relative_depths[i] for i, mask in enumerate(within_500um_mask) if mask]
                
                # Find the channel with highest power within this subset
                max_power_idx = np.argmax(filtered_powers)
                selected_depth = filtered_depths[max_power_idx]
                
                sw_selections.append({
                    'depth': selected_depth,
                    'probe_info': probe_info
                })
            
            processed_probes.append(f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']}")
            
        except Exception as e:
            skipped_probes.append(f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']} - {str(e)}")
    
    if USE_DEPTH_LIMITS and depth_filtered_count > 0:
        print(f"Sharp wave data: Filtered out {depth_filtered_count} channels due to depth limits (>{SHARP_WAVE_DEPTH_LIMIT} μm from ripple channel)")
    
    # Debug: Print some stats about what data we collected
    if sw_data:
        all_depths = list(sw_data.keys())
        print(f"DEBUG: Sharp wave raw data collected at {len(all_depths)} depth bins")
        print(f"DEBUG: Depth range: {min(all_depths)} to {max(all_depths)} μm")
        total_channels = sum(len(values) for values in sw_data.values())
        print(f"DEBUG: Total channels collected: {total_channels}")
    else:
        print(f"DEBUG: No sharp wave data collected at all!")
    
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

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_ripple_power(ripple_means, output_dir):
    """Create ripple band power plot."""
    
    # Set matplotlib font size globally for SVG output
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'svg.fonttype': 'none'  # Ensure SVG text stays as editable text, not paths
    })
    
    # Create figure for ripple plot (wider format)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot ripple data if available
    if ripple_means:
        # Multiply depths by -1 for anatomical orientation (deeper = more negative)
        depths = sorted([-d for d in ripple_means.keys()])  # Convert to negative and sort
        net_power_values = [ripple_means[-d]['net_power'] for d in depths]  # Use original keys
        
        # Create ripple plot with black bars
        ax.barh(depths, net_power_values, height=BAR_HEIGHT, 
                color='black', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_xlabel('Z-scored Net Power', fontsize=FONT_SIZE)
        ax.set_ylabel('Depth Relative to Selected Channel (μm)', fontsize=FONT_SIZE)
        ax.set_title('Ripple Band Power (150-250Hz)', fontsize=FONT_SIZE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Invert y-axis for anatomical orientation
        
        # Set y-axis limits and ticks
        ax.set_ylim(RIPPLE_Y_AXIS_LIMITS)
        ax.set_yticks(range(RIPPLE_Y_AXIS_LIMITS[0], RIPPLE_Y_AXIS_LIMITS[1] + 1, Y_TICK_INTERVAL))
        ax.tick_params(axis='both', labelsize=FONT_SIZE)
        
        # Add selected channel reference
        ax.text(0.02, 0.98, 'Selected Channel', transform=ax.transAxes, 
                verticalalignment='top', fontsize=FONT_SIZE, color='red')
    else:
        ax.text(0.5, 0.5, 'No Ripple Data', transform=ax.transAxes, 
                ha='center', va='center', fontsize=FONT_SIZE)
        ax.set_title('Ripple Band Power (150-250Hz)', fontsize=FONT_SIZE, fontweight='bold')
        ax.set_ylim(RIPPLE_Y_AXIS_LIMITS)
        ax.set_yticks(range(RIPPLE_Y_AXIS_LIMITS[0], RIPPLE_Y_AXIS_LIMITS[1] + 1, Y_TICK_INTERVAL))
        ax.tick_params(axis='both', labelsize=FONT_SIZE)
        ax.set_xlabel('Z-scored Net Power', fontsize=FONT_SIZE)
        ax.set_ylabel('Depth Relative to Selected Channel (μm)', fontsize=FONT_SIZE)
        ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save in specified formats
    for fmt in OUTPUT_FORMATS:
        output_file = os.path.join(output_dir, f'ripple_band_power.{fmt}')
        plt.savefig(output_file, format=fmt, dpi=DPI, bbox_inches='tight')
        print(f"Saved ripple band power plot: {output_file}")
    
    plt.show()
    plt.close()

def plot_sharp_wave_power(sw_means, output_dir):
    """Create sharp wave power plot."""
    
    # Set matplotlib font size globally for SVG output
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'svg.fonttype': 'none'  # Ensure SVG text stays as editable text, not paths
    })
    
    # Create figure for sharp wave plot (narrower format)
    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    
    # Plot sharp wave data if available
    if sw_means:
        # Multiply depths by -1 for anatomical orientation (deeper = more negative)
        depths = sorted([-d for d in sw_means.keys()])  # Convert to negative and sort
        net_power_values = [sw_means[-d]['net_sw_power'] for d in depths]  # Use original keys
        
        # Create sharp wave plot with blue bars
        ax.barh(depths, net_power_values, height=BAR_HEIGHT, 
                color='blue', alpha=0.7, edgecolor='blue', linewidth=0.5)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_xlabel('Z-scored Net Power', fontsize=FONT_SIZE)
        ax.set_ylabel('Depth Relative to Ripple Channel (μm)', fontsize=FONT_SIZE)
        #ax.set_title('Concurrent SW Band Power (8-40Hz)', fontsize=FONT_SIZE, fontweight='bold')
        ax.set_title('Modulation Index (SW to Ripple)', fontsize=FONT_SIZE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits and ticks
        ax.set_ylim(SW_Y_AXIS_LIMITS)
        ax.set_yticks(range(SW_Y_AXIS_LIMITS[0], SW_Y_AXIS_LIMITS[1] + 1, Y_TICK_INTERVAL))
        ax.tick_params(axis='both', labelsize=FONT_SIZE)
        
    else:
        ax.text(0.5, 0.5, 'No Sharp Wave Data', transform=ax.transAxes, 
                ha='center', va='center', fontsize=FONT_SIZE)
        ax.set_title('Concurrent SW Band Power (8-40Hz)', fontsize=FONT_SIZE, fontweight='bold')
        ax.set_ylim(SW_Y_AXIS_LIMITS)
        ax.set_yticks(range(SW_Y_AXIS_LIMITS[0], SW_Y_AXIS_LIMITS[1] + 1, Y_TICK_INTERVAL))
        ax.tick_params(axis='both', labelsize=FONT_SIZE)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save in specified formats
    for fmt in OUTPUT_FORMATS:
        output_file = os.path.join(output_dir, f'sharp_wave_band_power.{fmt}')
        plt.savefig(output_file, format=fmt, dpi=DPI, bbox_inches='tight')
        print(f"Saved sharp wave band power plot: {output_file}")
    
    plt.show()
    plt.close()

def plot_sharp_wave_selection(sw_selections, output_dir):
    """Create histogram showing selected sharp wave channel depths (modified selection within 500μm)."""
    
    # Set matplotlib font size globally for SVG output
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'svg.fonttype': 'none'  # Ensure SVG text stays as editable text, not paths
    })
    
    # Create figure for sharp wave selection plot (half width of ripple plot)
    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    
    if not sw_selections:
        ax.text(0.5, 0.5, 'No Sharp Wave Selection Data', transform=ax.transAxes, 
                ha='center', va='center', fontsize=FONT_SIZE)
        ax.set_title('Concurrent SW Band Power (8-40Hz)', fontsize=FONT_SIZE, fontweight='bold')
        ax.set_ylim(SW_Y_AXIS_LIMITS)
        ax.set_yticks(range(SW_Y_AXIS_LIMITS[0], SW_Y_AXIS_LIMITS[1] + 1, Y_TICK_INTERVAL))
        ax.tick_params(axis='both', labelsize=FONT_SIZE)
        ax.set_xlabel('Number of Selected Channels', fontsize=FONT_SIZE)
        ax.set_ylabel('Depth Relative to Ripple Channel (μm)', fontsize=FONT_SIZE)
        ax.grid(True, alpha=0.3)
    else:
        # Extract depths and bin them
        depths = [sel['depth'] for sel in sw_selections]
        binned_depths = [bin_depth(d) for d in depths]
        
        # Count frequency of each binned depth
        from collections import Counter
        depth_counts = Counter(binned_depths)
        
        if depth_counts:
            # Convert to sorted lists for plotting (anatomical orientation)
            original_depths = list(depth_counts.keys())
            plot_depths = sorted([-d for d in original_depths])  # Convert to negative and sort
            counts = [depth_counts[-d] for d in plot_depths]  # Use original keys for counts
            
            # Create horizontal bar chart
            ax.barh(plot_depths, counts, height=BAR_HEIGHT, 
                    color='blue', alpha=0.7, edgecolor='blue', linewidth=0.5)
            
            # Add reference line at y=0 to show the ripple channel
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # Formatting
        ax.set_xlabel('Number of Selected Channels', fontsize=FONT_SIZE)
        ax.set_ylabel('Depth Relative to Ripple Channel (μm)', fontsize=FONT_SIZE)
        ax.set_title('Sharp Wave Channel Selection', fontsize=FONT_SIZE, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Set y-axis limits and ticks
        ax.set_ylim(SW_Y_AXIS_LIMITS)
        ax.set_yticks(range(SW_Y_AXIS_LIMITS[0], SW_Y_AXIS_LIMITS[1] + 1, Y_TICK_INTERVAL))
        ax.tick_params(axis='both', labelsize=FONT_SIZE)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save in specified formats
    for fmt in OUTPUT_FORMATS:
        output_file = os.path.join(output_dir, f'concurrent_sw_band_selection.{fmt}')
        plt.savefig(output_file, format=fmt, dpi=DPI, bbox_inches='tight')
        print(f"Saved concurrent SW band selection plot: {output_file}")
    
    plt.show()
    plt.close()

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main(base_directory, output_directory=None, run_name=None):
    """Main function to run the simplified analysis."""
    
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
    print(f"Depth limits: ±{RIPPLE_DEPTH_LIMIT_UP}/{RIPPLE_DEPTH_LIMIT_DOWN} μm (ripple), ±{SHARP_WAVE_DEPTH_LIMIT} μm (sharp wave)")
    print(f"Ripple Y-axis range: {RIPPLE_Y_AXIS_LIMITS[0]} to {RIPPLE_Y_AXIS_LIMITS[1]} μm")
    print(f"Sharp wave Y-axis range: {SW_Y_AXIS_LIMITS[0]} to {SW_Y_AXIS_LIMITS[1]} μm")
    print(f"Y-axis ticks every: {Y_TICK_INTERVAL} μm")
    print(f"Font size: {FONT_SIZE}")
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
        print(f"Skipped {len(ripple_skipped)} probes")
    
    # Filter and calculate means for ripple data
    ripple_filtered = filter_by_probe_count(ripple_data)
    ripple_means = calculate_mean_values(ripple_filtered)
    
    print(f"Ripple data: {len(ripple_data)} total depths, {len(ripple_filtered)} after filtering (≥{MIN_PROBE_COUNT} probes)")
    if ripple_means:
        ripple_depth_range = f"{min(ripple_means.keys())} to {max(ripple_means.keys())} μm"
        print(f"Ripple depth range: {ripple_depth_range}")
    
    # Process sharp wave data
    print("\nProcessing sharp wave band data...")
    sw_data, sw_selections, sw_processed, sw_skipped = process_sharp_wave_data(metadata_files)
    
    print(f"Successfully processed {len(sw_processed)} probes for sharp wave analysis")
    if sw_skipped:
        print(f"Skipped {len(sw_skipped)} probes")
    
    # Filter and calculate means for sharp wave data
    sw_filtered = filter_by_probe_count(sw_data)
    sw_means = calculate_mean_values(sw_filtered)
    
    print(f"Sharp wave data: {len(sw_data)} total depths, {len(sw_filtered)} after filtering (≥{MIN_PROBE_COUNT} probes)")
    if sw_means:
        sw_depth_range = f"{min(sw_means.keys())} to {max(sw_means.keys())} μm"
        print(f"Sharp wave depth range: {sw_depth_range}")
    
    # Create separate plots
    print(f"\nGenerating separate plots in {output_directory}...")
    plot_ripple_power(ripple_means, output_directory)
    plot_sharp_wave_power(sw_means, output_directory)
    plot_sharp_wave_selection(sw_selections, output_directory)
    
    print("\nSimplified analysis complete!")

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