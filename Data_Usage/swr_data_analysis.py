"""
# SWR Data Analysis for ABI Visual Behavior Dataset

This notebook is designed for analyzing Sharp Wave Ripple (SWR) events in the ABI Visual Behavior dataset.
It requires the allensdk_env conda environment to be loaded.

The analysis includes:
1. Loading and filtering SWR events
2. Aligning events with CA1 units
3. Analyzing unit responses to ripples
4. Aligning events with running speed

Note: Time is the unifying factor across all analyses - all data can be aligned using timestamps.
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
from scipy.stats import mannwhitneyu
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache

# Configuration
CACHE_DIR = "/space/scratch/allen_visbehave_data"
SWR_INPUT_DIR = "/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_v2_final"
DATASET_NAME = "allen_visbehave_swr_murphylab2024"
SESSION_ID = 1047969464  # Example session ID

"""
## 1. Loading and Filtering SWR Events

The SWR events are stored in compressed CSV files within session-specific folders.
The folder structure is:
```
SWR_INPUT_DIR/
└── DATASET_NAME/
    └── swrs_session_{SESSION_ID}/
        ├── probe_*_channel_*_putative_swr_events.csv.gz
        ├── probe_*_channel_*_gamma_band_events.csv.gz
        └── ...
```

Let's load and filter the events based on our criteria:
- 3SD < power_max_zscore < 10SD
- No gamma or movement overlap
- Minimum sw_peak_power > 1SD
"""

def load_and_filter_swr_events(session_id):
    """Load and filter SWR events for a given session."""
    session_path = os.path.join(SWR_INPUT_DIR, DATASET_NAME, f"swrs_session_{session_id}")
    event_files = glob.glob(os.path.join(session_path, "probe_*_channel_*_putative_swr_events.csv.gz"))
    
    all_events = []
    for event_file in event_files:
        try:
            events_df = pd.read_csv(event_file, compression='gzip')
            if not events_df.empty:
                all_events.append(events_df)
        except Exception as e:
            print(f"Warning: Could not load {event_file}: {e}")
            continue
    
    if not all_events:
        return pd.DataFrame([])
    
    events = pd.concat(all_events, ignore_index=True)
    
    # Filter events based on criteria
    filtered = events[
        (events['power_max_zscore'] > 3) & 
        (events['power_max_zscore'] < 10) &
        (~events['overlaps_with_gamma']) &
        (~events['overlaps_with_movement']) &
        (events['sw_peak_power'] > 1)
    ].copy()
    
    print(f"Total events: {len(events)}")
    print(f"Filtered events: {len(filtered)}")
    return filtered

"""
## 2. Aligning with CA1 Units

We'll analyze how CA1 units respond to SWR events using a simple Mann-Whitney U test.
This is a relatively simple method that compares firing rates during SWR events vs baseline periods.
Note: This analysis can take a few minutes to run.
"""

def calculate_unit_significance(session, unit_id, events_df, bin_size=0.01):
    """Calculate Mann-Whitney U test for a unit's firing during vs outside SWR events."""
    spike_times = session.spike_times[unit_id]
    
    if events_df.empty:
        return np.nan, np.nan
    
    session_start = events_df['start_time'].min()
    session_end = events_df['end_time'].max()
    time_bins = np.arange(session_start, session_end + bin_size, bin_size)
    spike_counts = np.histogram(spike_times, bins=time_bins)[0]
    
    # Create during mask
    during_mask = np.zeros_like(spike_counts, dtype=bool)
    for _, event in events_df.iterrows():
        start_bin = int((event['start_time'] - session_start) / bin_size)
        end_bin = int((event['end_time'] - session_start) / bin_size)
        during_mask[start_bin:end_bin] = True
    
    during_samples = spike_counts[during_mask]
    baseline_samples = spike_counts[~during_mask]
    
    try:
        stat, pval = mannwhitneyu(during_samples, baseline_samples, alternative='two-sided')
        effect = np.mean(during_samples) - np.mean(baseline_samples)
        return effect, pval
    except:
        return np.nan, np.nan

def find_responding_units(session, events_df, target_region='CA1'):
    """Find units that respond significantly to SWR events."""
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=CACHE_DIR)
    units = cache.get_unit_table()
    
    # Filter for good units in target region
    region_units = units[
        (units['ecephys_session_id'] == session.ecephys_session_id) & 
        (units['structure_acronym'] == target_region) &
        (units['quality'] == 'good') &
        (units['valid_data'] == True)
    ]
    
    results = []
    for unit_id in region_units.index:
        effect, pval = calculate_unit_significance(session, unit_id, events_df)
        if not np.isnan(effect):
            results.append({
                'unit_id': unit_id,
                'effect': effect,
                'pval': pval
            })
    
    return pd.DataFrame(results)

"""
## 3. Aligning with Running Speed

We'll analyze how SWR events relate to running speed.
Note: Running speed data may have gaps that need to be checked session by session.
"""

def plot_swr_running_alignment(session, events_df, window=0.5):
    """Plot SWR events aligned with running speed."""
    running_speed = session.running_speed
    
    # Print running speed data info
    print("\nRunning Speed Data Info:")
    print(f"Number of samples: {len(running_speed)}")
    print(f"Time range: {running_speed['timestamps'].min():.2f} to {running_speed['timestamps'].max():.2f}")
    
    # Check for gaps in running speed data
    run_times = running_speed['timestamps'].values
    run_gaps = np.diff(run_times)
    median_gap = np.median(run_gaps)
    large_gaps = np.where(run_gaps > 5 * median_gap)[0]
    if len(large_gaps) > 0:
        print(f"\nFound {len(large_gaps)} large gaps in running speed data")
        print("Large gaps at times:", run_times[large_gaps])
    
    # Plot running speed with SWR events
    plt.figure(figsize=(12, 4))
    plt.plot(running_speed['timestamps'], running_speed['speed'], 'b-', alpha=0.5, label='Running Speed')
    
    # Mark SWR events
    for _, event in events_df.iterrows():
        plt.axvspan(event['start_time'], event['end_time'], 
                   color='red', alpha=0.2, label='SWR Event' if _ == 0 else "")
        if not np.isnan(event['power_peak_time']):
            plt.axvline(event['power_peak_time'], color='red', linestyle=':', 
                       label='SWR Peak' if _ == 0 else "")
    
    plt.xlabel('Time (s)')
    plt.ylabel('Running Speed (cm/s)')
    plt.title('SWR Events Aligned with Running Speed')
    plt.legend()
    plt.show()

"""
## Example Usage

Here's how to use the functions above:
"""

# Load session
cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=CACHE_DIR)
session = cache.get_ecephys_session(SESSION_ID)

# Load and filter SWR events
swr_events = load_and_filter_swr_events(SESSION_ID)

# Find responding CA1 units
responding_units = find_responding_units(session, swr_events)
print("\nResponding Units:")
print(responding_units.sort_values('pval').head())

# Plot alignment with running speed
plot_swr_running_alignment(session, swr_events) 