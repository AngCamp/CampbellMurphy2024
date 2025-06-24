"""
Verify that there are actually channels below the chosen CA1 channel in the channel selection metadata.
This helps identify potential errors in channel selection where no suitable channels were found below the CA1 channel.
"""

import os
import json
import gzip
import glob
from pathlib import Path
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data configuration
BASE_DIRECTORY = "/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_v2_final"
OUTPUT_DIR = "channel_selection_verification"

DATASETS = [
    'allen_visbehave_swr_murphylab2024',
    'allen_viscoding_swr_murphylab2024', 
    'ibl_swr_murphylab2024'
]

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
    
    for dataset in DATASETS:
        dataset_path = os.path.join(base_directory, dataset)
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset directory not found: {dataset_path}")
            continue
            
        # Find all session directories
        session_dirs = [d for d in glob.glob(os.path.join(dataset_path, "*")) if os.path.isdir(d)]
        
        for session_dir in session_dirs:
            session_id = os.path.basename(session_dir)
            
            # Find all probe metadata files in this session
            probe_files = glob.glob(os.path.join(session_dir, "probe_*_channel_selection_metadata.json.gz"))
            
            for file_path in probe_files:
                # Extract probe ID from filename
                probe_id = os.path.basename(file_path).split('_')[1]
                
                metadata_files.append({
                    'dataset': dataset,
                    'session_id': session_id,
                    'probe_id': probe_id,
                    'file_path': file_path
                })
    
    return metadata_files

def verify_channel_selection(metadata_files):
    """Verify that there are channels below the chosen CA1 channel."""
    results = defaultdict(lambda: {'total_probes': 0, 'no_channels_below_count': 0})
    problematic_probes = []
    
    for file_info in metadata_files:
        data = load_channel_metadata(file_info['file_path'])
        
        if data is None or 'sharp_wave_band' not in data or 'ripple_band' not in data:
            continue
            
        dataset = file_info['dataset']
        results[dataset]['total_probes'] += 1
        
        # Get ripple channel depth as reference
        ripple_band = data['ripple_band']
        sharp_wave_band = data['sharp_wave_band']
        
        ripple_selected_id = ripple_band['selected_channel_id']
        ripple_channel_ids = ripple_band['channel_ids']
        ripple_depths = ripple_band['depths']
        
        if ripple_selected_id not in ripple_channel_ids:
            continue
            
        ripple_idx = ripple_channel_ids.index(ripple_selected_id)
        ripple_depth = ripple_depths[ripple_idx]
        
        # Get sharp wave channel depths
        sw_depths = sharp_wave_band['depths']
        sw_channel_ids = sharp_wave_band['channel_ids']
        
        # Check if any channels are below the ripple channel
        has_channels_below = False
        min_distance = float('inf')
        channels_below = []
        
        for sw_depth, sw_id in zip(sw_depths, sw_channel_ids):
            distance_from_ref = sw_depth - ripple_depth
            if distance_from_ref > 0:  # Channel is below ripple channel
                has_channels_below = True
                min_distance = min(min_distance, distance_from_ref)
                channels_below.append((sw_id, distance_from_ref))
        
        # If no channels found below, this is a potential issue
        if not has_channels_below:
            results[dataset]['no_channels_below_count'] += 1
            probe_info = f"{dataset}/{file_info['session_id']}/{file_info['probe_id']}"
            problematic_probes.append({
                'probe_info': probe_info,
                'ripple_depth': ripple_depth,
                'channel_depths': [(sw_id, sw_depth - ripple_depth) for sw_id, sw_depth in zip(sw_channel_ids, sw_depths)]
            })
    
    return results, problematic_probes

def generate_report(results, problematic_probes, output_dir):
    """Generate a report of the verification results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary report
    with open(os.path.join(output_dir, 'channel_selection_verification_report.txt'), 'w') as f:
        f.write("Channel Selection Verification Report\n")
        f.write("===================================\n\n")
        
        total_probes = sum(r['total_probes'] for r in results.values())
        total_problems = sum(r['no_channels_below_count'] for r in results.values())
        
        f.write(f"Overall Statistics:\n")
        f.write(f"Total probes analyzed: {total_probes}\n")
        f.write(f"Probes with no channels below CA1: {total_problems}\n")
        f.write(f"Problem rate: {(total_problems/total_probes)*100:.1f}%\n\n")
        
        f.write("Per-Dataset Statistics:\n")
        f.write("----------------------\n")
        for dataset, stats in results.items():
            problem_rate = (stats['no_channels_below_count'] / stats['total_probes']) * 100 if stats['total_probes'] > 0 else 0
            f.write(f"\n{dataset}:\n")
            f.write(f"  Total probes: {stats['total_probes']}\n")
            f.write(f"  No channels below CA1: {stats['no_channels_below_count']}\n")
            f.write(f"  Problem rate: {problem_rate:.1f}%\n")
        
        f.write("\n\nDetailed Analysis of Problematic Probes:\n")
        f.write("----------------------------------------\n")
        for probe_data in sorted(problematic_probes, key=lambda x: x['probe_info']):
            f.write(f"\n{probe_data['probe_info']}:\n")
            f.write(f"  CA1 channel depth: {probe_data['ripple_depth']:.1f} μm\n")
            f.write("  All channel depths relative to CA1:\n")
            for chan_id, rel_depth in sorted(probe_data['channel_depths'], key=lambda x: x[1]):
                f.write(f"    Channel {chan_id}: {rel_depth:+.1f} μm\n")

def main():
    """Main function to run the verification."""
    print("Starting channel selection verification...")
    
    # Find all metadata files
    metadata_files = find_metadata_files(BASE_DIRECTORY)
    print(f"Found {len(metadata_files)} metadata files to analyze")
    
    # Verify channel selection
    results, problematic_probes = verify_channel_selection(metadata_files)
    
    # Generate report
    generate_report(results, problematic_probes, OUTPUT_DIR)
    print(f"Verification complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 