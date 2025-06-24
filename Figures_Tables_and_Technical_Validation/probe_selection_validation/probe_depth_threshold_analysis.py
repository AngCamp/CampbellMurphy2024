#!/usr/bin/env python3
"""
Probe Depth Threshold Analysis Script

This script analyzes channel selection metadata from Neuropixels probes
to identify sessions where sharp wave channels are selected beyond a depth threshold
relative to the ripple channel.
"""

import os
import json
import gzip
import glob
from pathlib import Path
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Depth threshold in micrometers (relative to ripple channel)
# Channels selected deeper than this will be flagged
DEPTH_THRESHOLD = 500

# Metric to use for threshold violations
# Choose one of the following:
VIOLATION_METRIC = 'modulation_index'
# VIOLATION_METRIC = 'net_sw_power'
# VIOLATION_METRIC = 'circular_linear_corr'

# Data configuration (update these paths as needed)
BASE_DIRECTORY = "yourpath/SWR_final_pipeline/osf_campbellmurphy2025_swr_data_backup"
OUTPUT_DIR = "depth_threshold_analysis4"

DATASETS = [
    'allen_visbehave_swr_murphylab2024',
    'allen_viscoding_swr_murphylab2024', 
    'ibl_swr_murphylab2024'
]

# Channel smoothing settings (match the original analysis)
USE_CHANNEL_SMOOTHING = True
CHANNEL_SMOOTHING_SIGMA = 50.0

# =============================================================================
# UTILITY FUNCTIONS
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

def anatomical_smooth_values(raw_values, channel_depths, sigma=CHANNEL_SMOOTHING_SIGMA):
    """
    Apply anatomical smoothing using Gaussian weights based on channel depth distances.
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

def create_failure_histogram(violations, all_selections, output_dir, depth_threshold):
    """Create histogram of sessions binned by number of failed probes (ABI datasets only)."""
    
    # Count failed probes per session for ABI datasets only (including 0 failures)
    abi_session_failures = []
    
    # First, get all ABI sessions and initialize with 0 failures
    abi_sessions = {}
    for dataset, sessions in all_selections.items():
        # Skip IBL dataset
        if 'ibl' in dataset.lower():
            continue
        
        for session_id in sessions.keys():
            abi_sessions[f"{dataset}/{session_id}"] = 0
    
    # Then count actual failures
    for dataset, sessions in violations.items():
        # Skip IBL dataset
        if 'ibl' in dataset.lower():
            continue
            
        for session_id, probes in sessions.items():
            session_key = f"{dataset}/{session_id}"
            if session_key in abi_sessions:
                abi_sessions[session_key] = len(probes)
    
    # Convert to list for histogram
    abi_session_failures = list(abi_sessions.values())
    
    if not abi_session_failures:
        print("No ABI dataset sessions found for histogram")
        return
    
    # Create histogram
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    bins = range(0, max(abi_session_failures) + 2)  # 0 to max+1 (include 0 failures)
    n, bins, patches = plt.hist(abi_session_failures, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Color the 0 failures bar differently to highlight it
    if len(patches) > 0:
        patches[0].set_color('lightgreen')  # Sessions with no failures in green
    
    # Formatting
    plt.xlabel('Number of Failed Probes per Session')
    plt.ylabel('Number of Sessions')
    plt.title(f'Distribution of Failed Probes per Session (ABI Datasets Only)\nThreshold: {depth_threshold}μm ({VIOLATION_METRIC})')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, count in enumerate(n):
        if count > 0:
            plt.text(bins[i] + 0.5, count + 0.05, str(int(count)), ha='center', va='bottom')
    
    # Set x-axis to show integer ticks
    plt.xticks(range(0, max(abi_session_failures) + 1))
    
    # Add statistics text box
    total_sessions = len(abi_session_failures)
    sessions_with_failures = sum(1 for x in abi_session_failures if x > 0)
    sessions_no_failures = sum(1 for x in abi_session_failures if x == 0)
    total_failed_probes = sum(abi_session_failures)
    mean_failures = np.mean(abi_session_failures)
    
    stats_text = f'Total Sessions: {total_sessions}\n'
    stats_text += f'Sessions with Failures: {sessions_with_failures}\n'
    stats_text += f'Sessions with No Failures: {sessions_no_failures}\n'
    stats_text += f'Total Failed Probes: {total_failed_probes}\n'
    stats_text += f'Mean Failures/Session: {mean_failures:.1f}'
    
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    hist_path = os.path.join(output_dir, f'session_failure_histogram_{depth_threshold}um.png')
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print(f"Histogram saved: {hist_path}")
    plt.show()

def analyze_probe_selections(metadata_files, depth_threshold):
    """Analyze sharp wave channel selections and identify MI threshold violations."""
    
    violations = defaultdict(lambda: defaultdict(dict))  # dataset -> session -> probe -> metrics
    all_selections = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))  # same structure for all data
    
    processed_probes = []
    skipped_probes = []
    
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
            
            # Find selected channels for each metric (highest z-scored values)
            selections = {
                'net_sw_power': {
                    'idx': np.argmax(z_net_power),
                    'name': 'Net SW Power'
                },
                'modulation_index': {
                    'idx': np.argmax(z_mod_index), 
                    'name': 'Modulation Index'
                },
                'circular_linear_corr': {
                    'idx': np.argmax(z_corr),
                    'name': 'Circular-Linear Correlation'
                }
            }
            
            dataset = file_info['dataset']
            session_id = file_info['session_id']
            probe_id = file_info['probe_id']
            
            # Store all selections for metadata
            probe_selections = {}
            for metric, sel_info in selections.items():
                selected_depth = relative_depths[sel_info['idx']]
                probe_selections[metric] = selected_depth
            
            all_selections[dataset][session_id][probe_id] = probe_selections
            
            # Check selected metric for threshold violations
            violation_depth = probe_selections[VIOLATION_METRIC]
            has_violation = abs(violation_depth) > depth_threshold
            
            if has_violation:
                violations[dataset][session_id][probe_id] = probe_selections
                processed_probes.append(f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']} - {VIOLATION_METRIC.upper()} VIOLATION ({violation_depth:.1f}μm)")
            else:
                processed_probes.append(f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']} - OK")
            
        except Exception as e:
            skipped_probes.append(f"{file_info['dataset']}/{file_info['session_id']}/{file_info['probe_id']} - {str(e)}")
    
    return violations, all_selections, processed_probes, skipped_probes

def generate_outputs(violations, all_selections, output_dir, depth_threshold):
    """Generate text file and JSON outputs."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect session lists for pipeline re-running (separate ABI datasets)
    abi_visbehave_sessions = set()
    abi_viscoding_sessions = set()
    ibl_sessions = set()
    
    # Count violations by dataset and get totals
    dataset_stats = defaultdict(lambda: {'sessions': set(), 'probes': 0, 'total_sessions': set(), 'total_probes': 0})
    total_violated_probes = 0
    total_violated_sessions = set()
    total_probes = 0
    total_sessions = set()
    
    # First, count all probes and sessions
    for dataset, sessions in all_selections.items():
        for session_id, probes in sessions.items():
            dataset_stats[dataset]['total_sessions'].add(session_id)
            total_sessions.add(f"{dataset}/{session_id}")
            for probe_id, metrics in probes.items():
                dataset_stats[dataset]['total_probes'] += 1
                total_probes += 1
    
    # Then count violations
    for dataset, sessions in violations.items():
        for session_id, probes in sessions.items():
            total_violated_sessions.add(f"{dataset}/{session_id}")
            dataset_stats[dataset]['sessions'].add(session_id)
            
            # Add to appropriate session list
            if 'ibl' in dataset.lower():
                ibl_sessions.add(session_id)
            elif 'visbehave' in dataset.lower():
                try:
                    abi_visbehave_sessions.add(int(session_id))
                except ValueError:
                    abi_visbehave_sessions.add(session_id)
            elif 'viscoding' in dataset.lower():
                try:
                    abi_viscoding_sessions.add(int(session_id))
                except ValueError:
                    abi_viscoding_sessions.add(session_id)
            
            for probe_id, metrics in probes.items():
                total_violated_probes += 1
                dataset_stats[dataset]['probes'] += 1
    
    # Generate text file
    txt_path = os.path.join(output_dir, f'depth_threshold_{depth_threshold}um_violations.txt')
    with open(txt_path, 'w') as f:
        metric_names = {
            'net_sw_power': 'Net SW Power',
            'modulation_index': 'Modulation Index',
            'circular_linear_corr': 'Circular-Linear Correlation'
        }
        metric_display = metric_names.get(VIOLATION_METRIC, VIOLATION_METRIC)
        
        f.write(f"Sharp Wave Channel Depth Threshold Analysis ({metric_display} Only)\n")
        f.write(f"Threshold: {depth_threshold} μm (relative to ripple channel)\n")
        f.write(f"="*60 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY VIOLATION INFORMATION:\n")
        f.write(f"Total violated probes: {total_violated_probes} of {total_probes} probes ({100*total_violated_probes/total_probes:.1f}%)\n")
        f.write(f"Total violated sessions: {len(total_violated_sessions)} of {len(total_sessions)} sessions ({100*len(total_violated_sessions)/len(total_sessions):.1f}%)\n\n")
        
        # Show all datasets (including those with no violations)
        all_datasets = set(dataset_stats.keys())
        for dataset in sorted(all_datasets):
            stats = dataset_stats[dataset]
            if stats['probes'] > 0:
                f.write(f"{dataset.upper()}: {stats['probes']} of {stats['total_probes']} probes ({100*stats['probes']/stats['total_probes']:.1f}%), {len(stats['sessions'])} of {len(stats['total_sessions'])} sessions ({100*len(stats['sessions'])/len(stats['total_sessions']):.1f}%)\n")
            else:
                f.write(f"{dataset.upper()}: All {stats['total_probes']} probes passed, all {len(stats['total_sessions'])} sessions passed\n")
        f.write("\n")
        
        # Session lists for pipeline re-running
        f.write("SESSION LISTS FOR PIPELINE RE-RUNNING:\n\n")
        
        if abi_visbehave_sessions:
            visbehave_list = sorted(list(abi_visbehave_sessions))
            f.write(f"ABI Visbehave Sessions: {visbehave_list}\n\n")
        
        if abi_viscoding_sessions:
            viscoding_list = sorted(list(abi_viscoding_sessions))
            f.write(f"ABI Viscoding Sessions: {viscoding_list}\n\n")
        
        if ibl_sessions:
            ibl_list = sorted(list(ibl_sessions))
            f.write(f"IBL Sessions: {ibl_list}\n\n")
        
        f.write("-"*60 + "\n\n")
        
        # Detailed violation information
        f.write("DETAILED VIOLATION INFORMATION:\n\n")
        
        for dataset, sessions in violations.items():
            f.write(f"{dataset.upper()}\n")
            
            for session_id, probes in sorted(sessions.items()):
                f.write(f"Session: {session_id}\n")
                
                for probe_id, metrics in sorted(probes.items()):
                    metric_info = []
                    for metric, depth in metrics.items():
                        metric_names = {
                            'net_sw_power': 'Net SW Power',
                            'modulation_index': 'Modulation Index', 
                            'circular_linear_corr': 'Circular-Linear Corr'
                        }
                        metric_info.append(f"{metric_names[metric]}: {depth:.1f}μm")
                    
                    f.write(f"  Probes: {probe_id}, {', '.join(metric_info)}\n")
                f.write("\n")
            f.write("\n")
    
    # Generate JSON metadata with failed/passed separation
    json_path = os.path.join(output_dir, f'depth_threshold_{depth_threshold}um_metadata.json')
    
    # Convert all_selections to JSON-serializable format
    json_data = {}
    for dataset, sessions in all_selections.items():
        json_data[dataset] = {}
        for session_id, probes in sessions.items():
            json_data[dataset][session_id] = {
                'failed_probes': {},
                'passed_probes': {}
            }
            
            for probe_id, metrics in probes.items():
                # Check if this probe failed (violation in selected metric)
                violation_depth = metrics[VIOLATION_METRIC]
                if abs(violation_depth) > depth_threshold:
                    # Failed probe
                    json_data[dataset][session_id]['failed_probes'][probe_id] = {}
                    for metric, depth in metrics.items():
                        json_data[dataset][session_id]['failed_probes'][probe_id][metric] = str(depth)
                else:
                    # Passed probe
                    json_data[dataset][session_id]['passed_probes'][probe_id] = {}
                    for metric, depth in metrics.items():
                        json_data[dataset][session_id]['passed_probes'][probe_id][metric] = str(depth)
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Create histogram of sessions by number of failed probes (ABI datasets only)
    create_failure_histogram(violations, all_selections, output_dir, depth_threshold)
    
    # Print summary with dataset breakdown
    metric_names = {
        'net_sw_power': 'Net SW Power',
        'modulation_index': 'Modulation Index',
        'circular_linear_corr': 'Circular-Linear Correlation'
    }
    metric_display = metric_names.get(VIOLATION_METRIC, VIOLATION_METRIC)
    
    print(f"\nANALYSIS COMPLETE:")
    print(f"Depth threshold: {depth_threshold} μm ({metric_display} only)")
    print(f"Total violated probes: {total_violated_probes} of {total_probes} probes ({100*total_violated_probes/total_probes:.1f}%)")
    print(f"Total violated sessions: {len(total_violated_sessions)} of {len(total_sessions)} sessions ({100*len(total_violated_sessions)/len(total_sessions):.1f}%)")
    print(f"\nDataset breakdown:")
    all_datasets = set(dataset_stats.keys())
    for dataset in sorted(all_datasets):
        stats = dataset_stats[dataset]
        if stats['probes'] > 0:
            print(f"  {dataset}: {stats['probes']} of {stats['total_probes']} probes ({100*stats['probes']/stats['total_probes']:.1f}%), {len(stats['sessions'])} of {len(stats['total_sessions'])} sessions ({100*len(stats['sessions'])/len(stats['total_sessions']):.1f}%)")
        else:
            print(f"  {dataset}: All {stats['total_probes']} probes passed, all {len(stats['total_sessions'])} sessions passed")
    print(f"\nOutputs saved to:")
    print(f"  Text file: {txt_path}")
    print(f"  JSON file: {json_path}")
    
    return total_violated_probes, len(total_violated_sessions)

def main():
    """Main analysis function."""
    
    metric_names = {
        'net_sw_power': 'Net SW Power',
        'modulation_index': 'Modulation Index',
        'circular_linear_corr': 'Circular-Linear Correlation'
    }
    metric_display = metric_names.get(VIOLATION_METRIC, VIOLATION_METRIC)
    
    print(f"Sharp Wave Channel Depth Threshold Analysis")
    print(f"Violation metric: {metric_display}")
    print(f"Depth threshold: {DEPTH_THRESHOLD} μm")
    print(f"Base directory: {BASE_DIRECTORY}")
    print(f"Datasets: {DATASETS}")
    print("-" * 60)
    
    # Find metadata files
    metadata_files = find_metadata_files(BASE_DIRECTORY)
    print(f"Found {len(metadata_files)} metadata files")
    
    if not metadata_files:
        print("No metadata files found. Check your base directory and dataset names.")
        return
    
    # Run analysis
    print(f"\nAnalyzing sharp wave channel selections...")
    violations, all_selections, processed, skipped = analyze_probe_selections(metadata_files, DEPTH_THRESHOLD)
    
    print(f"Processed {len(processed)} probes")
    if skipped:
        print(f"Skipped {len(skipped)} probes")
    
    # Debug: Show processing breakdown by dataset
    dataset_processing = defaultdict(lambda: {'total': 0, 'violations': 0, 'ok': 0})
    for probe_info in processed:
        dataset = probe_info.split('/')[0]
        dataset_processing[dataset]['total'] += 1
        if 'VIOLATION' in probe_info:
            dataset_processing[dataset]['violations'] += 1
        else:
            dataset_processing[dataset]['ok'] += 1
    
    print(f"\nProcessing breakdown by dataset:")
    for dataset, stats in dataset_processing.items():
        print(f"  {dataset}: {stats['total']} total ({stats['violations']} violations, {stats['ok']} OK)")
    
    # Debug: Show selected metric depth ranges for each dataset
    metric_names = {
        'net_sw_power': 'Net SW Power',
        'modulation_index': 'Modulation Index', 
        'circular_linear_corr': 'Circular-Linear Correlation'
    }
    metric_display = metric_names.get(VIOLATION_METRIC, VIOLATION_METRIC)
    
    print(f"\n{metric_display} depth ranges by dataset:")
    for dataset, sessions in all_selections.items():
        metric_depths = []
        for session_id, probes in sessions.items():
            for probe_id, metrics in probes.items():
                metric_depths.append(abs(metrics[VIOLATION_METRIC]))
        
        if metric_depths:
            print(f"  {dataset}: {len(metric_depths)} probes, {metric_display} depths {min(metric_depths):.1f}-{max(metric_depths):.1f}μm (mean: {np.mean(metric_depths):.1f}μm)")
        else:
            print(f"  {dataset}: No data found")
    
    # Generate outputs
    print(f"\nGenerating outputs...")
    total_violated_probes, total_violated_sessions = generate_outputs(
        violations, all_selections, OUTPUT_DIR, DEPTH_THRESHOLD
    )

if __name__ == "__main__":
    main() 