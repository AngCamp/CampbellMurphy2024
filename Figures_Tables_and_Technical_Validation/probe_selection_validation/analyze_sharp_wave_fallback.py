"""
Analyze how often the sharp wave channel selection falls back to using the ripple channel
due to no suitable channels being found within 500 microns below the ripple channel.
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
BASE_DIRECTORY = "yourpath/SWR_final_pipeline/osf_campbellmurphy2025_swr_data_backup"
OUTPUT_DIR = "sharp_wave_fallback_analysis"

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

def analyze_fallback_usage(metadata_files):
    """Analyze when the sharp wave channel selection should have fallen back to using the ripple channel."""
    results = defaultdict(lambda: {'total_probes': 0, 'should_fallback_count': 0})
    fallback_probes = []
    
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
        
        # Check if any channels are within 500 microns below ripple channel
        has_channels_below = False
        min_distance = float('inf')
        for sw_depth, sw_id in zip(sw_depths, sw_channel_ids):
            distance_from_ref = sw_depth - ripple_depth
            if 0 < distance_from_ref <= 500:  # Within 500 microns below
                has_channels_below = True
                break
            elif distance_from_ref > 0:  # Track minimum distance for reporting
                min_distance = min(min_distance, distance_from_ref)
        
        # If no channels found within 500 microns below, this should have used fallback
        if not has_channels_below:
            results[dataset]['should_fallback_count'] += 1
            probe_info = f"{dataset}/{file_info['session_id']}/{file_info['probe_id']}"
            fallback_probes.append({
                'probe_info': probe_info,
                'ripple_depth': ripple_depth,
                'min_distance': min_distance if min_distance != float('inf') else None,
                'channel_depths': [(sw_id, sw_depth - ripple_depth) for sw_id, sw_depth in zip(sw_channel_ids, sw_depths)]
            })
    
    return results, fallback_probes

def generate_report(results, fallback_probes, output_dir):
    """Generate a report of the analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary report
    with open(os.path.join(output_dir, 'fallback_analysis_report.txt'), 'w') as f:
        f.write("Sharp Wave Channel Selection Fallback Analysis\n")
        f.write("=============================================\n\n")
        
        total_probes = sum(r['total_probes'] for r in results.values())
        total_fallbacks = sum(r['should_fallback_count'] for r in results.values())
        
        f.write(f"Overall Statistics:\n")
        f.write(f"Total probes analyzed: {total_probes}\n")
        f.write(f"Probes that should use fallback: {total_fallbacks}\n")
        f.write(f"Overall fallback rate: {(total_fallbacks/total_probes)*100:.1f}%\n\n")
        
        f.write("Per-Dataset Statistics:\n")
        f.write("----------------------\n")
        for dataset, stats in results.items():
            fallback_rate = (stats['should_fallback_count'] / stats['total_probes']) * 100 if stats['total_probes'] > 0 else 0
            f.write(f"\n{dataset}:\n")
            f.write(f"  Total probes: {stats['total_probes']}\n")
            f.write(f"  Should use fallback: {stats['should_fallback_count']}\n")
            f.write(f"  Fallback rate: {fallback_rate:.1f}%\n")
        
        f.write("\n\nDetailed Analysis of Probes Needing Fallback:\n")
        f.write("--------------------------------------------\n")
        for probe_data in sorted(fallback_probes, key=lambda x: x['probe_info']):
            f.write(f"\n{probe_data['probe_info']}:\n")
            f.write(f"  Ripple channel depth: {probe_data['ripple_depth']:.1f} μm\n")
            if probe_data['min_distance'] is not None:
                f.write(f"  Minimum distance to any channel below: {probe_data['min_distance']:.1f} μm\n")
            else:
                f.write("  No channels found below ripple channel\n")
            f.write("  All channel depths relative to ripple:\n")
            for chan_id, rel_depth in sorted(probe_data['channel_depths'], key=lambda x: x[1]):
                f.write(f"    Channel {chan_id}: {rel_depth:+.1f} μm\n")

def main():
    """Main function to run the analysis."""
    print("Starting sharp wave fallback analysis...")
    
    # Find all metadata files
    metadata_files = find_metadata_files(BASE_DIRECTORY)
    print(f"Found {len(metadata_files)} metadata files to analyze")
    
    # Analyze fallback usage
    results, fallback_probes = analyze_fallback_usage(metadata_files)
    
    # Generate report
    generate_report(results, fallback_probes, OUTPUT_DIR)
    print(f"Analysis complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 