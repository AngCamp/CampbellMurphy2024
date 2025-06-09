#!/usr/bin/env python3

import os
import re
import gzip
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from pathlib import Path
import logging
from tqdm import tqdm
from ripple_detection import filter_ripple_band
from ripple_detection.core import gaussian_smooth
from scipy import stats


def extending_edeno_event_stats(events_df, time_values, lfp_array):
    """
    Extend event statistics with power-based metrics and envelope 90th percentile.
    Primarily computes power metrics from the ripple band LFP data, but also includes
    the 90th percentile of the smoothed envelope.
    """
    # Rename envelope-based metrics from Karlsson detector
    envelope_columns = {
        'max_thresh': 'envelope_max_thresh',
        'mean_zscore': 'envelope_mean_zscore',
        'median_zscore': 'envelope_median_zscore',
        'max_zscore': 'envelope_max_zscore',
        'min_zscore': 'envelope_min_zscore',
        'area': 'envelope_area',
        'total_energy': 'envelope_total_energy'
    }
    events_df = events_df.rename(columns=envelope_columns)
    
    # Filter LFP to ripple band
    ripple_filtered = filter_ripple_band(lfp_array[:, None])
    
    # Compute envelope and smooth it
    envelope = np.abs(scipy_signal.hilbert(ripple_filtered))
    smoothing_sigma = 0.004  # 4ms smoothing
    envelope_smoothed = gaussian_smooth(envelope, sigma=smoothing_sigma, sampling_frequency=1500.0)  # 1500 Hz sampling rate
    envelope_smoothed_z = stats.zscore(envelope_smoothed)
    # Compute power from smoothed envelope
    ripple_power = envelope_smoothed ** 2
    
    # Z-score the power using global statistics
    power_z = (ripple_power - np.mean(ripple_power)) / np.std(ripple_power)
    
    # Initialize lists for metrics
    power_metrics = []
    
    # Process each event
    for idx, event in events_df.iterrows():
        # Get event window
        mask = (time_values >= event['start_time']) & (time_values <= event['end_time'])
        
        # Debug print
        print(f"\nProcessing event {idx}")
        print(f"Event times: {event['start_time']} to {event['end_time']}")
        print(f"Time values range: {time_values[0]} to {time_values[-1]}")
        print(f"Number of points in mask: {np.sum(mask)}")
        
        if np.sum(mask) == 0:
            print(f"WARNING: No points found in time range for event {idx}")
            # Use the original values for this event
            metrics = {
                'power_peak_time': event['power_peak_time'],
                'power_max_zscore': event['power_max_zscore'],
                'power_median_zscore': event['power_median_zscore'],
                'power_mean_zscore': event['power_mean_zscore'],
                'power_min_zscore': event['power_min_zscore'],
                'power_90th_percentile': event['power_90th_percentile'],
                'envelope_90th_percentile': event['envelope_90th_percentile'],
                'envelope_peak_time': event['envelope_peak_time']
            }
        else:
            event_power_z = power_z[mask]
            event_envelope = envelope_smoothed[mask]
            event_times = time_values[mask]
            event_envelope_smoothed_z = envelope_smoothed_z[mask]
            
            # Compute metrics
            metrics = {
                'power_peak_time': event_times[np.argmax(event_power_z)],
                'power_max_zscore': np.max(event_power_z),
                'power_median_zscore': np.median(event_power_z),
                'power_mean_zscore': np.mean(event_power_z),
                'power_min_zscore': np.min(event_power_z),
                'power_90th_percentile': np.percentile(event_power_z, 90),
                'envelope_90th_percentile': np.percentile(event_envelope, 90),
                'envelope_peak_time': event_times[np.argmax(event_envelope_smoothed_z)]
            }
            
            # Debug print
            print(f"Computed metrics for event {idx}:")
            print(f"Power peak time: {metrics['power_peak_time']}")
            print(f"Original power peak time: {event['power_peak_time']}")
        
        power_metrics.append(metrics)
    
    # Convert to DataFrame
    power_df = pd.DataFrame(power_metrics)
    
    # Update existing columns or add new ones
    for col in power_df.columns:
        events_df[col] = power_df[col]
    
    return events_df

def process_session(swr_session_dir, lfp_session_dir, logger):
    """Process a single session directory."""
    session_id = swr_session_dir.name.split('_')[-1]
    logger.info(f"Processing session {session_id}")
    
    # Get all event files in the session directory
    event_files = list(swr_session_dir.glob("probe_*_channel_*_putative_swr_events.csv.gz"))
    
    for event_file in tqdm(event_files, desc=f"Processing session {session_id}"):
        # Extract probe_id and channel_id from filename
        match = re.match(r'probe_(\d+)_channel_(\d+)_putative_swr_events\.csv\.gz', event_file.name)
        if not match:
            logger.warning(f"Could not parse probe and channel IDs from {event_file.name}")
            continue
            
        probe_id = match.group(1)
        channel_id = match.group(2)
        
        # Construct paths for LFP data
        lfp_file = lfp_session_dir / f"probe_{probe_id}_channel_{channel_id}_lfp_ca1_putative_pyramidal_layer.npz"
        time_file = lfp_session_dir / f"probe_{probe_id}_channel_{channel_id}_lfp_time_index_1500hz.npz"
        
        if not (lfp_file.exists() and time_file.exists()):
            logger.warning(f"Missing LFP data for probe {probe_id} channel {channel_id}")
            continue
            
        try:
            # Load event data
            events_df = pd.read_csv(event_file, compression='gzip')
            
            # Load LFP data and print available keys
            lfp_data = np.load(lfp_file)
            time_data = np.load(time_file)
            
            logger.info(f"Available keys in LFP file: {lfp_data.files}")
            logger.info(f"Available keys in time file: {time_data.files}")
            
            # Get the LFP array and time index
            lfp_array = lfp_data[lfp_data.files[0]]  # Use first key
            time_index = time_data[time_data.files[0]]  # Use first key
            
            # Debug print
            print(f"\nProcessing file: {event_file.name}")
            print(f"LFP array shape: {lfp_array.shape}")
            print(f"Time index shape: {time_index.shape}")
            print(f"Number of events: {len(events_df)}")
            
            # Recompute extended values
            updated_events = extending_edeno_event_stats(events_df, time_index, lfp_array)
            
            # Save updated events back to the original file
            updated_events.to_csv(event_file, compression='gzip', index=False)
            logger.info(f"Updated {event_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {event_file.name}: {str(e)}")
            continue

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Base directory for SWR data
    #swr_base_dir = Path("/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_swr_data_backup/allen_visbehave_swr_murphylab2024")
    swr_base_dir = Path("/space/scratch/SWR_final_pipeline/muckingabout/allen_visbehave_swr_murphylab2024")
    
    # Construct LFP base directory by appending _lfp_data
    lfp_base_dir = Path(str(swr_base_dir) + "_lfp_data")
    
    # Get all session directories
    swr_session_dirs = list(swr_base_dir.glob("swrs_session_*"))
    
    for swr_session_dir in swr_session_dirs:
        session_id = swr_session_dir.name.split('_')[-1]
        lfp_session_dir = lfp_base_dir / f"lfp_session_{session_id}"
        
        if not lfp_session_dir.exists():
            logger.warning(f"Missing LFP directory for session {session_id}")
            continue
            
        process_session(swr_session_dir, lfp_session_dir, logger)

if __name__ == "__main__":
    main() 