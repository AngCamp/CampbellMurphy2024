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
from scipy import ndimage
from multiprocessing import Pool


def extending_edeno_event_stats_optimized(events_df, time_values, lfp_array):
    """
    Optimized version using vectorized operations and advanced indexing.
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
    envelope_smoothed = gaussian_smooth(envelope, sigma=smoothing_sigma, sampling_frequency=1500.0)
    envelope_smoothed_z = stats.zscore(envelope_smoothed)
    
    # Compute power from smoothed envelope
    ripple_power = envelope_smoothed ** 2
    power_z = (ripple_power - np.mean(ripple_power)) / np.std(ripple_power)
    
    # Convert time bounds to indices for faster processing
    start_indices = np.searchsorted(time_values, events_df['start_time'].values)
    end_indices = np.searchsorted(time_values, events_df['end_time'].values, side='right')
    
    # Ensure indices are within bounds
    start_indices = np.clip(start_indices, 0, len(time_values) - 1)
    end_indices = np.clip(end_indices, 0, len(time_values))
    
    # Method 1: Using list comprehension with advanced indexing (fastest for most cases)
    def compute_event_metrics_vectorized(data, start_idx, end_idx):
        """Compute metrics for a single event using vectorized operations."""
        if start_idx >= end_idx:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        event_data = data[start_idx:end_idx]
        return (
            np.max(event_data),
            np.median(event_data), 
            np.mean(event_data),
            np.min(event_data),
            np.percentile(event_data, 90),
            start_idx + np.argmax(event_data)  # Index of peak
        )
    
    # Vectorized computation of all metrics
    power_metrics = np.array([
        compute_event_metrics_vectorized(power_z.flatten(), start, end)
        for start, end in zip(start_indices, end_indices)
    ])
    
    envelope_metrics = np.array([
        compute_event_metrics_vectorized(envelope_smoothed.flatten(), start, end)
        for start, end in zip(start_indices, end_indices)
    ])
    
    envelope_z_peak_indices = np.array([
        start + np.argmax(envelope_smoothed_z.flatten()[start:end]) if start < end else start
        for start, end in zip(start_indices, end_indices)
    ])
    
    # Extract metrics and convert peak indices back to times
    power_max_zscores = power_metrics[:, 0]
    power_median_zscores = power_metrics[:, 1]
    power_mean_zscores = power_metrics[:, 2]
    power_min_zscores = power_metrics[:, 3]
    power_90th_percentiles = power_metrics[:, 4]
    power_peak_indices = power_metrics[:, 5].astype(int)
    
    envelope_90th_percentiles = envelope_metrics[:, 4]
    
    # Convert indices back to times (handle NaN indices)
    valid_power_indices = ~np.isnan(power_peak_indices)
    valid_envelope_indices = ~np.isnan(envelope_z_peak_indices)
    
    power_peak_times = np.full(len(events_df), np.nan)
    envelope_peak_times = np.full(len(events_df), np.nan)
    
    power_peak_times[valid_power_indices] = time_values[power_peak_indices[valid_power_indices]]
    envelope_peak_times[valid_envelope_indices] = time_values[envelope_z_peak_indices[valid_envelope_indices]]
    
    # Update the DataFrame with new values
    events_df['power_peak_time'] = power_peak_times
    events_df['power_max_zscore'] = power_max_zscores
    events_df['power_median_zscore'] = power_median_zscores
    events_df['power_mean_zscore'] = power_mean_zscores
    events_df['power_min_zscore'] = power_min_zscores
    events_df['power_90th_percentile'] = power_90th_percentiles
    events_df['envelope_90th_percentile'] = envelope_90th_percentiles
    events_df['envelope_peak_time'] = envelope_peak_times
    
    return events_df


def extending_edeno_event_stats_ultra_optimized(events_df, time_values, lfp_array):
    """
    Alternative ultra-optimized version using segment-based processing.
    Best for cases with many events and when memory usage is not a concern.
    """
    # [Same preprocessing as above...]
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
    envelope = np.abs(scipy_signal.hilbert(ripple_filtered))
    smoothing_sigma = 0.004
    envelope_smoothed = gaussian_smooth(envelope, sigma=smoothing_sigma, sampling_frequency=1500.0)
    envelope_smoothed_z = stats.zscore(envelope_smoothed)
    ripple_power = envelope_smoothed ** 2
    power_z = (ripple_power - np.mean(ripple_power)) / np.std(ripple_power)
    
    # Convert to indices
    start_indices = np.searchsorted(time_values, events_df['start_time'].values)
    end_indices = np.searchsorted(time_values, events_df['end_time'].values, side='right')
    start_indices = np.clip(start_indices, 0, len(time_values) - 1)
    end_indices = np.clip(end_indices, 0, len(time_values))
    
    # Method 2: Using scipy.ndimage for segment statistics (very fast for large datasets)
    # Create a label array where each event gets a unique label
    labels = np.zeros(len(time_values), dtype=int)
    for i, (start, end) in enumerate(zip(start_indices, end_indices)):
        if start < end:
            labels[start:end] = i + 1  # Labels start from 1
    
    # Use ndimage functions for fast segment-wise operations
    from scipy import ndimage
    
    # Get unique labels (excluding 0 which represents background)
    unique_labels = np.unique(labels[labels > 0])
    
    if len(unique_labels) > 0:
        # Compute all statistics at once using ndimage
        power_maxima = ndimage.maximum(power_z.flatten(), labels, unique_labels)
        power_means = ndimage.mean(power_z.flatten(), labels, unique_labels)
        power_mins = ndimage.minimum(power_z.flatten(), labels, unique_labels)
        
        # For median and percentiles, we need a custom approach
        def compute_percentile(data, labels, label_list, percentile):
            result = np.full(len(label_list), np.nan)
            for i, label in enumerate(label_list):
                mask = labels == label
                if np.any(mask):
                    result[i] = np.percentile(data[mask], percentile)
            return result
        
        power_medians = compute_percentile(power_z.flatten(), labels, unique_labels, 50)
        power_90th = compute_percentile(power_z.flatten(), labels, unique_labels, 90)
        envelope_90th = compute_percentile(envelope_smoothed.flatten(), labels, unique_labels, 90)
        
        # Find peak times
        power_peak_indices = ndimage.maximum_position(power_z.flatten(), labels, unique_labels)
        envelope_peak_indices = ndimage.maximum_position(envelope_smoothed_z.flatten(), labels, unique_labels)
        
        # Convert to arrays and handle the case where some events might be missing
        power_peak_times = np.full(len(events_df), np.nan)
        envelope_peak_times = np.full(len(events_df), np.nan)
        
        # Map results back to original event indices
        for i, label in enumerate(unique_labels):
            event_idx = label - 1  # Convert back to 0-based indexing
            if event_idx < len(events_df):
                power_peak_times[event_idx] = time_values[power_peak_indices[i]]
                envelope_peak_times[event_idx] = time_values[envelope_peak_indices[i]]
        
        # Initialize result arrays
        power_max_zscores = np.full(len(events_df), np.nan)
        power_median_zscores = np.full(len(events_df), np.nan)
        power_mean_zscores = np.full(len(events_df), np.nan)
        power_min_zscores = np.full(len(events_df), np.nan)
        power_90th_percentiles = np.full(len(events_df), np.nan)
        envelope_90th_percentiles = np.full(len(events_df), np.nan)
        
        # Map results back
        for i, label in enumerate(unique_labels):
            event_idx = label - 1
            if event_idx < len(events_df):
                power_max_zscores[event_idx] = power_maxima[i]
                power_median_zscores[event_idx] = power_medians[i]
                power_mean_zscores[event_idx] = power_means[i]
                power_min_zscores[event_idx] = power_mins[i]
                power_90th_percentiles[event_idx] = power_90th[i]
                envelope_90th_percentiles[event_idx] = envelope_90th[i]
    else:
        # Handle case with no valid events
        power_max_zscores = np.full(len(events_df), np.nan)
        power_median_zscores = np.full(len(events_df), np.nan)
        power_mean_zscores = np.full(len(events_df), np.nan)
        power_min_zscores = np.full(len(events_df), np.nan)
        power_90th_percentiles = np.full(len(events_df), np.nan)
        envelope_90th_percentiles = np.full(len(events_df), np.nan)
        power_peak_times = np.full(len(events_df), np.nan)
        envelope_peak_times = np.full(len(events_df), np.nan)
    
    # Update DataFrame
    events_df['power_peak_time'] = power_peak_times
    events_df['power_max_zscore'] = power_max_zscores
    events_df['power_median_zscore'] = power_median_zscores
    events_df['power_mean_zscore'] = power_mean_zscores
    events_df['power_min_zscore'] = power_min_zscores
    events_df['power_90th_percentile'] = power_90th_percentiles
    events_df['envelope_90th_percentile'] = envelope_90th_percentiles
    events_df['envelope_peak_time'] = envelope_peak_times
    
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
            updated_events = extending_edeno_event_stats_ultra_optimized(events_df, time_index, lfp_array)
            
            # Save updated events back to the original file
            updated_events.to_csv(event_file, compression='gzip', index=False)
            logger.info(f"Updated {event_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {event_file.name}: {str(e)}")
            continue

def process_session_wrapper(args):
    """Wrapper function for process_session to work with multiprocessing."""
    swr_session_dir, lfp_session_dir = args
    logger = logging.getLogger(__name__)
    return process_session(swr_session_dir, lfp_session_dir, logger)

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Base directory for SWR data
    swr_base_dir = Path("/space/scratch/SWR_final_pipeline/muckingabout/allen_visbehave_swr_murphylab2024")
    
    # Construct LFP base directory by appending _lfp_data
    lfp_base_dir = Path(str(swr_base_dir) + "_lfp_data")
    
    # Get all session directories
    swr_session_dirs = list(swr_base_dir.glob("swrs_session_*"))
    
    # Prepare arguments for parallel processing
    process_args = []
    for swr_session_dir in swr_session_dirs:
        session_id = swr_session_dir.name.split('_')[-1]
        lfp_session_dir = lfp_base_dir / f"lfp_session_{session_id}"
        
        if not lfp_session_dir.exists():
            logger.warning(f"Missing LFP directory for session {session_id}")
            continue
            
        process_args.append((swr_session_dir, lfp_session_dir))
    
    # Set pool size
    pool_size = 10
    
    # Process sessions in parallel
    with Pool(pool_size) as pool:
        pool.map(process_session_wrapper, process_args)

if __name__ == "__main__":
    main() 