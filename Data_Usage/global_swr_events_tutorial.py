# Global Sharp Wave Ripple Events Tutorial
"""
This tutorial demonstrates how to work with the global SWR events data from the 
Campbell, Murphy 2025 dataset. We'll walk through the analysis step by step,
showing results and visualizations as we go.
"""

# ## What are Global SWR Events?
"""
Global SWR events are detected when putative SWR events occur simultaneously 
across multiple probes within a specified time window (typically 50ms). These 
represent network-level hippocampal oscillations that span across recording sites, 
providing insights into the spatial organization and propagation of sharp wave 
ripples in the mouse hippocampus.

Each session folder contains:
- Individual probe SWR event files: `probe_{ID}_channel_{ID}_putative_swr_events.csv.gz`
- Global SWR events file: `session_{ID}_global_swr_events.csv.gz`
- Probe metadata: `session_{ID}_probe_metadata.csv.gz`
- Run settings: `session_{ID}_run_settings.json.gz`

The global events file links back to individual probe events through the 
`probe_event_file_index` column, allowing you to cross-reference between 
network-level and probe-specific data.
"""

# ## Step 1: Import Required Libraries and Set Up Paths

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import json
import ast
from pathlib import Path
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d

# AllenSDK imports for data loading
from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)

# Set up paths - modify these for your local setup
BASE_DATA_PATH = "/path/to/your/osf_campbellmurphy2025_data"  # Update this path
ALLENSDK_CACHE_DIR = "/path/to/your/allen_cache"  # Update this path
FILTER_PATH = "/path/to/NeuropixelsLFPOnRamp/SWR_Neuropixels_Detector/Filters/sharpwave_componenet_8to40band_1500hz_band.npz"

print("✓ Libraries imported successfully")

# ## Step 2: Set Up Data Access

def setup_allensdk_cache():
    """Set up the AllenSDK cache for accessing LFP data"""
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=ALLENSDK_CACHE_DIR)
    return cache

# Initialize the cache
cache = setup_allensdk_cache()
print("✓ AllenSDK cache initialized successfully")
print(f"Cache directory: {ALLENSDK_CACHE_DIR}")

# ## Step 3: Load Session Data

"""
Now let's load data for an example session. We'll use session 1044385384 which has 
4 probes with good CA1 coverage and contains interesting global events.

### Session Folder Structure:
```
swrs_session_1044385384/
├── probe_1044506933_channel_1049370115_putative_swr_events.csv.gz
├── probe_1044506935_channel_1049370909_putative_swr_events.csv.gz  
├── probe_1044506936_channel_1049371349_putative_swr_events.csv.gz
├── probe_1044506937_channel_1049372363_putative_swr_events.csv.gz
├── session_1044385384_global_swr_events.csv.gz
├── session_1044385384_probe_metadata.csv.gz
└── session_1044385384_run_settings.json.gz
```
"""

def load_session_data(session_id, base_path=BASE_DATA_PATH):
    """Load all data files for a given session"""
    session_dir = Path(base_path) / "allen_visbehave_swr_murphylab2024" / f"swrs_session_{session_id}"
    
    # Load global events
    global_events_path = session_dir / f"session_{session_id}_global_swr_events.csv.gz"
    with gzip.open(global_events_path, 'rt') as f:
        global_events = pd.read_csv(f, index_col=0)
    
    # Load probe metadata
    probe_metadata_path = session_dir / f"session_{session_id}_probe_metadata.csv.gz"
    with gzip.open(probe_metadata_path, 'rt') as f:
        probe_metadata = pd.read_csv(f)
    
    # Load individual probe events
    probe_events = {}
    for probe_file in session_dir.glob("probe_*_putative_swr_events.csv.gz"):
        # Extract probe ID from filename
        probe_id = probe_file.name.split('_')[1]
        with gzip.open(probe_file, 'rt') as f:
            events_df = pd.read_csv(f, index_col=0)
            probe_events[probe_id] = events_df
    
    # Load run settings
    run_settings_path = session_dir / f"session_{session_id}_run_settings.json.gz"
    with gzip.open(run_settings_path, 'rt') as f:
        run_settings = json.load(f)
    
    return global_events, probe_events, probe_metadata, run_settings

# Load the example session
session_id = "1044385384"
global_events, probe_events, probe_metadata, run_settings = load_session_data(session_id)

print(f"✓ Loaded session {session_id}")
print(f"  - {len(global_events)} global events")
print(f"  - {len(probe_events)} probes with individual events")
print(f"  - Probes: {list(probe_events.keys())}")

# Show basic statistics for each probe
for probe_id, events in probe_events.items():
    print(f"  - Probe {probe_id}: {len(events)} individual events")

# ## Step 4: Explore Global Events Data Structure

"""
Let's examine the structure of the global events data to understand what information
is available and how it's organized.
"""

print("Global Events DataFrame Shape:", global_events.shape)
print("\nColumn Names and Types:")
for col in global_events.columns:
    print(f"  {col}: {global_events[col].dtype}")

# Display the first few global events
print("\nFirst 3 Global Events:")
print("="*80)
display_cols = ['start_time', 'end_time', 'duration', 'probe_count', 'global_peak_power']
print(global_events[display_cols].head(3))

# Show the full structure of one event
print(f"\nDetailed view of Global Event 0:")
print("="*80)
event_0 = global_events.iloc[0]
for col in global_events.columns:
    value = event_0[col]
    if isinstance(value, str) and value.startswith('['):
        # Parse array columns for better display
        try:
            parsed = ast.literal_eval(value)
            print(f"  {col}: {parsed}")
        except:
            print(f"  {col}: {value}")
    else:
        print(f"  {col}: {value}")

# ## Step 5: Understanding Cross-Referencing

"""
The key insight is how global events link back to individual probe events through
the `probe_event_file_index` column. Let's demonstrate this cross-referencing.
"""

def demonstrate_cross_referencing(global_events, probe_events, event_idx=0):
    """Show how to cross-reference global and individual events"""
    event = global_events.iloc[event_idx]
    print(f"Cross-referencing Global Event {event_idx}:")
    print("="*60)
    print(f"Global event time: {event['start_time']:.3f} - {event['end_time']:.3f} seconds")
    print(f"Duration: {event['duration']:.3f} seconds")
    print(f"Number of participating probes: {event['probe_count']}")
    
    # Parse the array columns (stored as strings)
    participating_probes = ast.literal_eval(event['participating_probes'])
    peak_times = ast.literal_eval(event['peak_times'])
    peak_powers = ast.literal_eval(event['peak_powers'])
    probe_indices = ast.literal_eval(event['probe_event_file_index'])
    
    print(f"\nParticipating probes: {participating_probes}")
    print(f"Peak times: {[f'{t:.3f}' for t in peak_times]}")
    print(f"Peak powers: {[f'{p:.2f}' for p in peak_powers]}")
    print(f"Probe event indices: {probe_indices}")
    
    print("\nCorresponding Individual Probe Events:")
    print("-" * 60)
    
    # Cross-reference with individual probe events
    for i, (probe_id, probe_idx, peak_time, peak_power) in enumerate(
        zip(participating_probes, probe_indices, peak_times, peak_powers)
    ):
        if probe_id in probe_events:
            individual_event = probe_events[probe_id].iloc[probe_idx]
            print(f"\nProbe {probe_id} (Event #{probe_idx}):")
            print(f"  Individual event time: {individual_event['start_time']:.3f} - {individual_event['end_time']:.3f}")
            print(f"  Duration: {individual_event['duration']:.3f} seconds")
            print(f"  Power max z-score: {individual_event['power_max_zscore']:.2f}")
            print(f"  Peak time matches: {abs(peak_time - individual_event['power_peak_time']) < 0.001}")
            
            # Show a few key metrics from the individual event
            print(f"  Sharp wave exceeds threshold: {individual_event['sw_exceeds_threshold']}")
            print(f"  Overlaps with gamma: {individual_event['overlaps_with_gamma']}")
            print(f"  Overlaps with movement: {individual_event['overlaps_with_movement']}")
    
    return participating_probes, peak_times, peak_powers, probe_indices

# Demonstrate cross-referencing with the first global event
participating_probes, peak_times, peak_powers, probe_indices = demonstrate_cross_referencing(
    global_events, probe_events, event_idx=0
)

# ## Step 6: Visualize Global Events Distribution

"""
Let's create some visualizations to understand the characteristics of global events
in this session.
"""

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Distribution of global event durations
axes[0, 0].hist(global_events['duration'], bins=30, alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Duration (seconds)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Global Event Durations')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Distribution of probe counts
probe_counts = global_events['probe_count'].value_counts().sort_index()
axes[0, 1].bar(probe_counts.index, probe_counts.values, alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Number of Participating Probes')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Probe Participation in Global Events')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Global peak power distribution
axes[1, 0].hist(global_events['global_peak_power'], bins=30, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Global Peak Power (z-score)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Global Event Peak Powers')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Timeline of global events
axes[1, 1].scatter(global_events['global_peak_time'], global_events['global_peak_power'], 
                   c=global_events['probe_count'], cmap='viridis', alpha=0.6)
axes[1, 1].set_xlabel('Time (seconds)')
axes[1, 1].set_ylabel('Global Peak Power (z-score)')
axes[1, 1].set_title('Global Events Timeline')
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label('Probe Count')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle(f'Global Events Overview - Session {session_id}', fontsize=16, y=1.02)
plt.show()

print(f"\nSummary Statistics:")
print(f"  Total global events: {len(global_events)}")
print(f"  Mean duration: {global_events['duration'].mean():.3f} ± {global_events['duration'].std():.3f} seconds")
print(f"  Mean probe count: {global_events['probe_count'].mean():.1f}")
print(f"  Mean peak power: {global_events['global_peak_power'].mean():.2f} ± {global_events['global_peak_power'].std():.2f} z-score")

# ## Step 7: LFP Data Loading Functions

"""
Now let's set up functions to load the actual LFP data for visualization.
This requires the AllenSDK to access the raw electrophysiology data.
"""

def load_lfp_data_for_probe(cache, session_id, probe_id):
    """Load LFP data for a specific probe"""
    try:
        # Get the session from AllenSDK
        session = cache.get_ecephys_session(ecephys_session_id=int(session_id))
        
        # Load LFP data for the specific probe
        lfp = session.get_lfp(int(probe_id))
        
        # Get CA1 channels for this probe
        channels = session.get_channels()
        probe_channels = channels[channels.probe_id == int(probe_id)]
        ca1_channels = probe_channels[probe_channels.structure_acronym == "CA1"]
        
        if len(ca1_channels) == 0:
            print(f"No CA1 channels found for probe {probe_id}")
            return None, None, None
        
        # Select CA1 channels from LFP data
        ca1_channel_ids = ca1_channels.index.values
        ca1_lfp_data = lfp.sel(channel=ca1_channel_ids)
        
        return ca1_lfp_data, lfp.time.values, ca1_channels
        
    except Exception as e:
        print(f"Error loading LFP data for probe {probe_id}: {e}")
        return None, None, None

def resample_signal(signal_data, time_vals, target_fs=1500.0):
    """Resample the LFP signal to match the processing pipeline sampling rate"""
    from scipy import interpolate
    
    original_fs = 1.0 / np.mean(np.diff(time_vals))
    
    if abs(original_fs - target_fs) < 0.1:  # Already at target rate
        return signal_data, time_vals
    
    # Create new time vector at target sampling rate
    duration = time_vals[-1] - time_vals[0]
    n_samples = int(duration * target_fs)
    new_time = np.linspace(time_vals[0], time_vals[-1], n_samples)
    
    # Resample each channel
    if signal_data.ndim == 1:
        f = interpolate.interp1d(time_vals, signal_data, kind='linear', 
                               bounds_error=False, fill_value=0)
        resampled = f(new_time)
    else:
        resampled = np.zeros((len(new_time), signal_data.shape[1]))
        for ch in range(signal_data.shape[1]):
            f = interpolate.interp1d(time_vals, signal_data[:, ch], kind='linear',
                                   bounds_error=False, fill_value=0)
            resampled[:, ch] = f(new_time)
    
    return resampled, new_time

# Test loading LFP data for one probe
test_probe_id = participating_probes[0]
print(f"Testing LFP data loading for probe {test_probe_id}...")

lfp_data, time_vals, channels_info = load_lfp_data_for_probe(cache, session_id, test_probe_id)

if lfp_data is not None:
    print(f"✓ Successfully loaded LFP data")
    print(f"  Shape: {lfp_data.shape}")
    print(f"  Time range: {time_vals[0]:.1f} - {time_vals[-1]:.1f} seconds")
    print(f"  CA1 channels: {len(channels_info)}")
    print(f"  Sampling rate: {1.0 / np.mean(np.diff(time_vals)):.1f} Hz")
else:
    print("✗ Failed to load LFP data")

# ## Step 8: Signal Processing Functions

"""
Now let's implement the signal processing functions needed to filter the ripple band
and create the same visualizations used in the detection pipeline.
"""

def filter_ripple_band_signal(lfp_data, sampling_rate=1500.0):
    """Filter LFP signal in the ripple band (150-250 Hz)"""
    # Butterworth bandpass filter for ripple band
    nyquist = sampling_rate / 2
    low_freq = 150.0 / nyquist
    high_freq = 250.0 / nyquist
    
    if high_freq >= 1.0:
        high_freq = 0.99
    
    # Design the filter
    b, a = butter(4, [low_freq, high_freq], btype='band')
    
    # Apply filter
    if lfp_data.ndim == 1:
        filtered = filtfilt(b, a, lfp_data)
    else:
        filtered = np.zeros_like(lfp_data)
        for ch in range(lfp_data.shape[1]):
            filtered[:, ch] = filtfilt(b, a, lfp_data[:, ch])
    
    # Compute envelope using Hilbert transform
    if lfp_data.ndim == 1:
        envelope = np.abs(hilbert(filtered))
    else:
        envelope = np.zeros_like(filtered)
        for ch in range(filtered.shape[1]):
            envelope[:, ch] = np.abs(hilbert(filtered[:, ch]))
    
    # Smooth envelope (4ms Gaussian kernel at 1500 Hz)
    sigma = 0.004 * sampling_rate  # 4ms in samples
    if lfp_data.ndim == 1:
        smoothed = gaussian_filter1d(envelope, sigma)
    else:
        smoothed = np.zeros_like(envelope)
        for ch in range(envelope.shape[1]):
            smoothed[:, ch] = gaussian_filter1d(envelope[:, ch], sigma)
    
    # Square for power and z-score
    power = smoothed ** 2
    if lfp_data.ndim == 1:
        power_z = zscore(power)
    else:
        power_z = np.zeros_like(power)
        for ch in range(power.shape[1]):
            power_z[:, ch] = zscore(power[:, ch])
    
    return filtered, envelope, smoothed, power_z

def select_best_ripple_channel(lfp_data, channels_info):
    """Select the channel with the highest ripple band power"""
    if lfp_data.ndim == 1:
        return 0, channels_info.iloc[0]
    
    # Calculate total ripple power for each channel
    _, _, _, power_z = filter_ripple_band_signal(lfp_data)
    total_power = np.sum(power_z ** 2, axis=0)
    
    # Select channel with highest power
    best_channel_idx = np.argmax(total_power)
    best_channel_info = channels_info.iloc[best_channel_idx]
    
    print(f"Selected channel {best_channel_info.name} at depth {best_channel_info['probe_vertical_position']:.0f} μm")
    
    return best_channel_idx, best_channel_info

# Test the signal processing with our loaded data
if lfp_data is not None:
    print("Testing signal processing...")
    
    # Convert to numpy and resample
    lfp_array = lfp_data.to_numpy()
    lfp_resampled, time_resampled = resample_signal(lfp_array, time_vals, 1500.0)
    
    # Select best channel for ripple detection
    best_ch_idx, best_ch_info = select_best_ripple_channel(lfp_resampled, channels_info)
    
    # Process a short segment for demonstration
    segment_start = 5000  # Start at 5000 samples
    segment_end = 10000   # End at 10000 samples
    test_segment = lfp_resampled[segment_start:segment_end, best_ch_idx]
    
    # Apply ripple band filtering
    filtered, envelope, smoothed, power_z = filter_ripple_band_signal(test_segment)
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    
    time_segment = time_resampled[segment_start:segment_end]
    
    axes[0].plot(time_segment, test_segment, 'gray', linewidth=0.8)
    axes[0].set_ylabel('Raw LFP (μV)')
    axes[0].set_title(f'Signal Processing Demo - Probe {test_probe_id}, Channel {best_ch_info.name}')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time_segment, filtered, 'blue', linewidth=0.8)
    axes[1].set_ylabel('Filtered\n150-250 Hz (μV)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time_segment, envelope, 'red', linewidth=0.8)
    axes[2].set_ylabel('Envelope (μV)')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(time_segment, power_z, 'black', linewidth=1.0)
    axes[3].set_ylabel('Power\n(z-score)')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Signal processing completed successfully")

# ## Step 9: Plot a Global Event

"""
Now let's create the main visualization: plotting a global SWR event across
multiple probes, showing both the raw LFP and the processed ripple power.
"""

def plot_global_event_simple(cache, session_id, global_events, event_idx, window=0.15):
    """Plot a global SWR event across multiple probes"""
    event = global_events.iloc[event_idx]
    
    # Parse event data
    participating_probes = ast.literal_eval(event['participating_probes'])
    peak_times = ast.literal_eval(event['peak_times'])
    peak_powers = ast.literal_eval(event['peak_powers'])
    
    global_peak_time = event['global_peak_time']
    event_start = event['start_time']
    event_end = event['end_time']
    
    print(f"Plotting Global Event {event_idx}:")
    print(f"  Time: {event_start:.3f} - {event_end:.3f} seconds")
    print(f"  Global peak: {global_peak_time:.3f} seconds")
    print(f"  Participating probes: {participating_probes}")
    
    # Create figure
    fig, axes = plt.subplots(len(participating_probes), 1, 
                           figsize=(12, 2.5 * len(participating_probes)),
                           sharex=True)
    if len(participating_probes) == 1:
        axes = [axes]
    
    # Time window around global peak
    plot_start = global_peak_time - window
    plot_end = global_peak_time + window
    
    for i, (probe_id, peak_time, peak_power) in enumerate(
        zip(participating_probes, peak_times, peak_powers)
    ):
        ax = axes[i]
        
        # Load LFP data for this probe
        lfp_data, time_vals, channels_info = load_lfp_data_for_probe(
            cache, session_id, probe_id
        )
        
        if lfp_data is None:
            ax.text(0.5, 0.5, f'No data for probe {probe_id}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Convert to numpy and resample
        lfp_array = lfp_data.to_numpy()
        lfp_resampled, time_resampled = resample_signal(lfp_array, time_vals, 1500.0)
        
        # Select best channel for ripple detection
        best_ch_idx, best_ch_info = select_best_ripple_channel(lfp_resampled, channels_info)
        
        # Get time indices for plotting window
        time_mask = (time_resampled >= plot_start) & (time_resampled <= plot_end)
        plot_time = time_resampled[time_mask]
        plot_lfp = lfp_resampled[time_mask, best_ch_idx]
        
        # Convert to relative time (relative to global peak)
        plot_time_rel = plot_time - global_peak_time
        
        # Filter ripple band for this channel
        filtered, envelope, smoothed, power_z = filter_ripple_band_signal(plot_lfp)
        
        # Plot raw LFP
        ax.plot(plot_time_rel, plot_lfp, 'gray', alpha=0.7, linewidth=0.8, 
                label='LFP (μV)')
        
        # Plot ripple power (scaled for visualization)
        power_scaled = power_z * 50  # Adjust scaling as needed
        ax.plot(plot_time_rel, power_scaled, 'black', linewidth=1.5, 
                label='Ripple Power (z-scored)')
        
        # Mark global event duration
        event_start_rel = event_start - global_peak_time
        event_end_rel = event_end - global_peak_time
        ax.axvspan(event_start_rel, event_end_rel, alpha=0.3, color='green', 
                   label='Global Event')
        
        # Mark individual probe peak
        peak_time_rel = peak_time - global_peak_time
        ax.axvline(peak_time_rel, color='black', linestyle='--', alpha=0.8,
                   label=f'Peak (z={peak_power:.1f})')
        
        # Mark global peak
        ax.axvline(0, color='red', linestyle='-', alpha=0.5, linewidth=2,
                   label='Global Peak')
        
        # Formatting
        ax.set_ylabel(f'Probe {probe_id}\n(Depth: {best_ch_info["probe_vertical_position"]:.0f} μm)')
        ax.grid(True, alpha=0.3)
        
        if i == 0:  # Only show legend on first subplot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Final formatting
    axes[-1].set_xlabel('Time from Global Peak (s)')
    plt.suptitle(f'Global SWR Event {event_idx} - Session {session_id}\n'
                f'Duration: {event["duration"]:.3f}s, Peak Power: {event["global_peak_power"]:.1f}z',
                fontsize=14)
    
    plt.tight_layout()
    return fig

# Find a good event to plot (high power, multiple probes, reasonable duration)
good_events = global_events[
    (global_events['probe_count'] >= 3) & 
    (global_events['global_peak_power'] >= 4) &
    (global_events['duration'] <= 0.15)
]

if len(good_events) > 0:
    event_to_plot = good_events.index[0]
    print(f"\nPlotting global event {event_to_plot}:")
    print(f"  Peak power: {good_events.loc[event_to_plot, 'global_peak_power']:.1f} z-score")
    print(f"  Duration: {good_events.loc[event_to_plot, 'duration']:.3f} seconds")
    print(f"  Probe count: {good_events.loc[event_to_plot, 'probe_count']}")
    
    fig = plot_global_event_simple(cache, session_id, global_events, event_to_plot, window=0.15)
    plt.show()
else:
    print("No suitable events found for plotting")

# ## Step 10: Analyze Event Directionality

"""
Finally, let's analyze the directionality of global SWR events. This examines
whether events propagate from anterior to posterior probes or vice versa.
"""

def analyze_event_directionality(global_events, cache, session_id):
    """Analyze the directionality of global SWR events"""
    # Get channel table for AP coordinates
    channel_table = cache.get_channel_table()
    
    directional_stats = []
    
    for idx, event in global_events.iterrows():
        participating_probes = ast.literal_eval(event['participating_probes'])
        peak_times = ast.literal_eval(event['peak_times'])
        
        if len(participating_probes) < 2:
            continue
            
        # Get AP coordinates for each probe (using first CA1 channel as proxy)
        probe_ap_coords = []
        for probe_id in participating_probes:
            # Find a CA1 channel for this probe to get AP coordinate
            probe_channels = channel_table[channel_table.probe_id == int(probe_id)]
            ca1_channels = probe_channels[probe_channels.structure_acronym == "CA1"]
            
            if len(ca1_channels) > 0:
                ap_coord = ca1_channels.iloc[0]['anterior_posterior_ccf_coordinate']
                probe_ap_coords.append((probe_id, ap_coord))
        
        if len(probe_ap_coords) < 2:
            continue
            
        # Sort probes by AP coordinate (anterior first)
        probe_ap_coords.sort(key=lambda x: x[1], reverse=True)  # Higher AP = more anterior
        sorted_probes = [p[0] for p in probe_ap_coords]
        
        # Get peak times in AP order
        sorted_peak_times = []
        for probe_id in sorted_probes:
            probe_idx = participating_probes.index(probe_id)
            sorted_peak_times.append(peak_times[probe_idx])
        
        # Analyze temporal gradient
        time_diffs = np.diff(sorted_peak_times)
        
        # Classify directionality
        if np.all(time_diffs > 0.005):  # 5ms minimum delay
            direction = 'anterior_to_posterior'
        elif np.all(time_diffs < -0.005):
            direction = 'posterior_to_anterior'  
        else:
            direction = 'non_directional'
        
        directional_stats.append({
            'event_idx': idx,
            'direction': direction,
            'probe_count': len(participating_probes),
            'max_time_diff': max(abs(t) for t in time_diffs) if len(time_diffs) > 0 else 0,
            'duration': event['duration'],
            'peak_power': event['global_peak_power']
        })
    
    return pd.DataFrame(directional_stats)

# Analyze directionality for our session
print("Analyzing event directionality...")
directional_df = analyze_event_directionality(global_events, cache, session_id)

if len(directional_df) > 0:
    direction_counts = directional_df['direction'].value_counts()
    print(f"\nDirectionality distribution for {len(directional_df)} events:")
    for direction, count in direction_counts.items():
        percentage = (count / len(directional_df)) * 100
        print(f"  {direction}: {count} events ({percentage:.1f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Direction distribution
    direction_counts.plot(kind='bar', ax=axes[0], color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0].set_title('Event Directionality Distribution')
    axes[0].set_ylabel('Number of Events')
    axes[0].set_xlabel('Direction')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Time differences by direction
    for direction in directional_df['direction'].unique():
        subset = directional_df[directional_df['direction'] == direction]
        axes[1].scatter(subset['peak_power'], subset['max_time_diff'], 
                       label=direction, alpha=0.6, s=50)
    
    axes[1].set_xlabel('Global Peak Power (z-score)')
    axes[1].set_ylabel('Max Time Difference (s)')
    axes[1].set_title('Directionality vs Peak Power')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ## Summary

"""
This tutorial has demonstrated:

1. **Data Structure**: How global SWR events are organized and how they link to individual probe events
2. **Cross-referencing**: Using `probe_event_file_index` to connect global and individual events
3. **Signal Processing**: Filtering ripple band signals for visualization
4. **Visualization**: Creating publication-quality plots of global events across multiple probes
5. **Analysis**: Examining event directionality and propagation patterns

### Key Takeaways:

- Global events capture network-level hippocampal activity across multiple probes
- The dataset allows for quick regeneration of global events with different parameters
- Cross-referencing enables detailed analysis of individual probe contributions
- Directionality analysis reveals propagation patterns in the hippocampal network
- The visualization approach matches the methods used in the original publication

### Pipeline Flexibility:

The global event detection can be quickly rerun with different parameters using
the `run_pipeline.sh` script with the `-g` flag, allowing researchers to explore:
- Different simultaneity windows (merge_window)
- Different minimum probe counts
- Different artifact exclusion criteria
- Different power thresholds

This makes the dataset highly flexible for testing different definitions of
"global" hippocampal events without reprocessing the computationally expensive
LFP filtering steps.
"""

print("✓ Tutorial completed successfully!")
print(f"Session {session_id} analysis summary:")
print(f"  - Total global events: {len(global_events)}")
print(f"  - Events with directionality info: {len(directional_df) if len(directional_df) > 0 else 0}")
print(f"  - Participating probes: {list(probe_events.keys())}")