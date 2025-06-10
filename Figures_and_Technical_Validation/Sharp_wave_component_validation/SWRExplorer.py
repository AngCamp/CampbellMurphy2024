#!/usr/bin/env python3
import os
import gzip
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import fftconvolve, butter, filtfilt, hilbert, convolve
from scipy.signal.windows import gaussian
from scipy.stats import zscore
from pathlib import Path
from tqdm import tqdm
import re
import glob
from ripple_detection import filter_ripple_band
from ripple_detection.core import gaussian_smooth
import matplotlib as mpl
from itertools import chain
import math
import ast
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import ast
import glob
import tempfile
import subprocess
try:
    from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
except ImportError:
    VisualBehaviorNeuropixelsProjectCache = None


class SWRExplorer:
    """
    Class for exploring and plotting Sharp Wave Ripple (SWR) events.
    
    The class provides methods to:
    1. Load and organize SWR data from multiple datasets
    2. Filter events based on various metrics
    3. Plot individual SWR events with detailed visualizations
    4. List available sessions and probes
    """
    
    def __init__(self, base_path=None, allensdk_cache_dir="/space/scratch/allen_visbehave_data"):
        """
        Initialize the SWRExplorer.
        
        Parameters:
        -----------
        base_path : str, optional
            Base path to the data directory. If None, uses default path.
        """
        if base_path is None:
            raise ValueError("base_path must be specified explicitly")
        self.base_path = Path(base_path)
            
        self.data_sources = [
            "allen_visbehave_swr_murphylab2024",
            "ibl_swr_murphylab2024",
            "allen_viscoding_swr_murphylab2024"
        ]
        
        self.lfp_sources = {
            "allen_visbehave_swr_murphylab2024": "allen_visbehave_swr_murphylab2024_lfp_data",
            "ibl_swr_murphylab2024": "ibl_swr_murphylab2024_lfp_data",
            "allen_viscoding_swr_murphylab2024": "allen_viscoding_swr_murphylab2024_lfp_data"
        }
        
        self.data = {}
        self.allensdk_cache_dir = allensdk_cache_dir
        self.allensdk_cache = None
        self.channel_table = None
        self._load_data()
        self._load_allensdk_cache_and_channels()
        
    def _load_data(self):
        """Load all available data from the specified sources."""
        for source in self.data_sources:
            source_path = self.base_path / source
            if not source_path.exists():
                print(f"Warning: Source path does not exist: {source_path}")
                continue
                
            self.data[source] = {}
            
            # Find all session directories
            for session_dir in source_path.glob("swrs_session_*"):
                if not session_dir.is_dir():
                    continue
                    
                session_id = session_dir.name.split('_')[-1]
                self.data[source][session_id] = {}
                
                # Find all probe event files
                for event_file in session_dir.glob("probe_*_channel_*_putative_swr_events.csv.gz"):
                    try:
                        # Extract probe ID from filename
                        filename = event_file.name
                        probe_match = re.search(r'probe_([^_]+)_channel_[^_]+_putative_swr_events', filename)
                        if not probe_match:
                            print(f"Warning: Could not extract probe ID from filename: {filename}")
                            continue
                            
                        probe_id = probe_match.group(1)
                        if probe_id not in self.data[source][session_id]:
                            self.data[source][session_id][probe_id] = {}
                        
                        print(f"\n[DEBUG] Loading events for {source}/{session_id}/probe_{probe_id}")
                        with gzip.open(event_file, 'rt') as f:
                            events_df = pd.read_csv(f)
                            print(f"[DEBUG] Raw events DataFrame info:")
                            print(events_df.info())
                            print("\n[DEBUG] First few rows of raw events:")
                            print(events_df.head())
                            print("\n[DEBUG] Index values (first 10):")
                            print(events_df.index.values[:10])
                            
                            # Check for 'Unnamed: 0' column
                            if 'Unnamed: 0' in events_df.columns:
                                print(f"\n[DEBUG] Found 'Unnamed: 0' column, renaming to 'event_id'")
                                events_df = events_df.rename(columns={'Unnamed: 0': 'event_id'})
                                events_df = events_df.set_index('event_id')
                                print("\n[DEBUG] After setting index:")
                                print(events_df.head())
                                print("\n[DEBUG] New index values (first 10):")
                                print(events_df.index.values[:10])
                            
                            self.data[source][session_id][probe_id]['events'] = events_df
                            
                    except Exception as e:
                        print(f"Error loading {event_file}: {e}")
                        continue
    
    def _load_allensdk_cache_and_channels(self):
        if VisualBehaviorNeuropixelsProjectCache is not None:
            self.allensdk_cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=self.allensdk_cache_dir)
            self.channel_table = self.allensdk_cache.get_channel_table()
        else:
            print("[WARNING] AllenSDK not available. Channel sorting by AP will not work.")

    def get_channel_ap_coordinate(self, channel_id):
        if self.channel_table is not None and channel_id in self.channel_table.index:
            return self.channel_table.loc[channel_id, 'anterior_posterior_ccf_coordinate']
        return None

    def list_available_data(self):
        """Print available sessions and probes for each dataset."""
        for source in self.data_sources:
            print(f"\nDataset: {source}")
            if source not in self.data:
                print("  No data available")
                continue
                
            for session_id in self.data[source]:
                print(f"  Session: {session_id}")
                for probe_id in self.data[source][session_id]:
                    event_count = len(self.data[source][session_id][probe_id].get('events', []))
                    print(f"    Probe: {probe_id} - {event_count} events")
    
    def get_session_probe_stats(self, dataset=None):
        """
        Get statistics about events for each session and probe.
        
        Parameters:
        -----------
        dataset : str, optional
            Specific dataset to analyze. If None, analyze all datasets.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with session and probe statistics
        """
        stats = []
        datasets = [dataset] if dataset else self.data_sources
        
        for source in datasets:
            if source not in self.data:
                continue
                
            for session_id in self.data[source]:
                for probe_id in self.data[source][session_id]:
                    events = self.data[source][session_id][probe_id].get('events', pd.DataFrame())
                    if len(events) > 0:
                        stats.append({
                            'dataset': source,
                            'session_id': session_id,
                            'probe_id': probe_id,
                            'total_events': len(events),
                            'mean_sw_power': events['sw_peak_power'].mean(),
                            'mean_duration': events['duration'].mean(),
                            'mean_clcorr': events['sw_ripple_clcorr'].mean(),
                            'max_clcorr': events['sw_ripple_clcorr'].max(),
                            'events_no_gamma': len(events[~events['overlaps_with_gamma']]),
                            'events_no_movement': len(events[~events['overlaps_with_movement']])
                        })
        
        return pd.DataFrame(stats)
    
    def find_best_events(self, dataset, session_id, probe_id, 
                        min_sw_power=1.5, min_duration=0.08, max_duration=0.1,
                        min_clcorr=0.8, exclude_gamma=True, exclude_movement=True):
        """
        Find events matching specific criteria.
        
        Parameters:
        -----------
        dataset : str
            Dataset name
        session_id : str
            Session ID
        probe_id : str
            Probe ID
        min_sw_power : float
            Minimum sharp wave power threshold
        min_duration : float
            Minimum event duration in seconds
        max_duration : float
            Maximum event duration in seconds
        min_clcorr : float
            Minimum circular-linear correlation
        exclude_gamma : bool
            Whether to exclude events overlapping with gamma
        exclude_movement : bool
            Whether to exclude events overlapping with movement
            
        Returns:
        --------
        pd.DataFrame
            Filtered events sorted by circular-linear correlation
        """
        if dataset not in self.data or session_id not in self.data[dataset] or probe_id not in self.data[dataset][session_id]:
            raise ValueError("Invalid dataset, session_id, or probe_id")
            
        events = self.data[dataset][session_id][probe_id]['events'].copy()
        print(f"[DEBUG] {dataset} {session_id} {probe_id} events columns: {list(events.columns)}, shape: {events.shape}")
        if events.empty:
            print(f"[INFO] No events for dataset={dataset}, session={session_id}, probe={probe_id}")
        if 'overlaps_with_gamma' not in events.columns or 'overlaps_with_movement' not in events.columns:
            print(f"[ERROR] Missing expected columns in events for dataset={dataset}, session={session_id}, probe={probe_id}")
            print(f"Columns found: {list(events.columns)}")
            print(f"File may be corrupted or from an old pipeline run.")
            raise KeyError("Missing 'overlaps_with_gamma' or 'overlaps_with_movement' in event file!")
        
        # Apply filters
        mask = (
            (events['sw_peak_power'] > min_sw_power) & 
            (events['duration'] >= min_duration) & 
            (events['duration'] <= max_duration) &
            (events['sw_ripple_clcorr'] >= min_clcorr)
        )
        
        if exclude_gamma:
            mask &= ~events['overlaps_with_gamma']
        if exclude_movement:
            mask &= ~events['overlaps_with_movement']
            
        filtered_events = events[mask].copy()
        
        # Add dataset, session, and probe information as columns
        filtered_events['dataset'] = dataset
        filtered_events['session_id'] = session_id
        filtered_events['probe_id'] = probe_id
        
        # Ensure 'Unnamed: 0' is preserved as 'event_id'
        if 'Unnamed: 0' in filtered_events.columns:
            filtered_events = filtered_events.rename(columns={'Unnamed: 0': 'event_id'})
            filtered_events = filtered_events.set_index('event_id')
        
        return filtered_events.sort_values('sw_ripple_clcorr', ascending=False)
    
    def get_event_speed(self, speed_data, start_time, end_time):
        """
        Get the average speed during an event.
        
        Parameters:
        -----------
        speed_data : dict
            Dictionary containing 'time' and 'velocity' arrays
        start_time : float
            Event start time
        end_time : float
            Event end time
            
        Returns:
        --------
        float
            Average speed during the event
        """
        mask = (speed_data['time'] >= start_time) & (speed_data['time'] <= end_time)
        
        if mask.size > 0 and not np.isnan(speed_data['velocity'][mask]).all():
            return np.nanmean(np.abs(speed_data['velocity'][mask]))
        return 0
    
    def filter_events_by_speed(self, events, speed_data, max_speed=5.0):
        """
        Filter events based on speed criteria.
        
        Parameters:
        -----------
        events : pd.DataFrame
            Events to filter
        speed_data : dict
            Dictionary containing 'time' and 'velocity' arrays
        max_speed : float
            Maximum allowed speed during event
            
        Returns:
        --------
        pd.DataFrame
            Events with speed below threshold
        """
        speeds = []
        for _, event in events.iterrows():
            speed = self.get_event_speed(speed_data, event['start_time'], event['end_time'])
            speeds.append(speed)
        
        events['speed'] = speeds
        return events[events['speed'] <= max_speed]
    
    def filter_events(self, dataset, session_id, probe_id, 
                     min_sw_power=1.0, min_duration=0.05,
                     max_speed=None, speed_data=None,
                     exclude_gamma=True, exclude_movement=True):
        """
        Filter events based on specified criteria.
        
        Parameters:
        -----------
        dataset : str
            Dataset name
        session_id : str
            Session ID
        probe_id : str
            Probe ID
        min_sw_power : float
            Minimum sharp wave power threshold
        min_duration : float
            Minimum event duration in seconds
        max_speed : float, optional
            Maximum allowed speed during event
        speed_data : dict, optional
            Dictionary containing 'time' and 'velocity' arrays for speed filtering
        exclude_gamma : bool
            Whether to exclude events overlapping with gamma
        exclude_movement : bool
            Whether to exclude events overlapping with movement
            
        Returns:
        --------
        pd.DataFrame
            Filtered events
        """
        if dataset not in self.data or session_id not in self.data[dataset] or probe_id not in self.data[dataset][session_id]:
            raise ValueError("Invalid dataset, session_id, or probe_id")
            
        events = self.data[dataset][session_id][probe_id]['events'].copy()
        
        print(f"\n[DEBUG] Filtering events for {dataset}/{session_id}/probe_{probe_id}")
        print(f"[DEBUG] Initial events shape: {events.shape}")
        print(f"[DEBUG] Initial events index values (first 10):")
        print(events.index.values[:10])
        
        # Apply basic filters
        mask = (events['sw_peak_power'] > min_sw_power) & (events['duration'] > min_duration)
        
        if exclude_gamma:
            mask &= ~events['overlaps_with_gamma']
        if exclude_movement:
            mask &= ~events['overlaps_with_movement']
            
        filtered_events = events[mask].copy()
        
        print(f"\n[DEBUG] After basic filtering:")
        print(f"Shape: {filtered_events.shape}")
        print(f"Index values (first 10):")
        print(filtered_events.index.values[:10])
        
        # Apply speed filter if requested
        if max_speed is not None and speed_data is not None:
            speeds = []
            for _, event in filtered_events.iterrows():
                speed = self.get_event_speed(speed_data, event['start_time'], event['end_time'])
                speeds.append(speed)
            filtered_events['speed'] = speeds
            filtered_events = filtered_events[filtered_events['speed'] <= max_speed]
            
            print(f"\n[DEBUG] After speed filtering:")
            print(f"Shape: {filtered_events.shape}")
            print(f"Index values (first 10):")
            print(filtered_events.index.values[:10])
            
        return filtered_events.sort_values('sw_ripple_clcorr', ascending=False)
    
    def get_event_by_csv_index(self, dataset, session_id, probe_id, event_id):
        """
        Fetch an event by its index in the probe's event CSV file.
        Parameters:
        -----------
        dataset : str
            Dataset name
        session_id : str
            Session ID
        probe_id : str
            Probe ID
        event_id : int
            Index of the event in the probe's event CSV (row number)
        Returns:
        --------
        pd.Series
            The event row as a Series
        """
        print(f"\n[DEBUG] Getting event by index for {dataset}/{session_id}/probe_{probe_id}")
        print(f"[DEBUG] Requested event_id: {event_id}")
        
        events = self.data[dataset][session_id][probe_id]['events']
        print(f"[DEBUG] Available events index values (first 10):")
        print(events.index.values[:10])
        
        if event_id not in events.index:
            print(f"[DEBUG] Event ID {event_id} not found in index")
            print(f"[DEBUG] Index contains values: {events.index.values}")
            raise ValueError(f"Event ID {event_id} not found in events for probe {probe_id}.")
            
        event = events.loc[event_id]
        print(f"[DEBUG] Found event:")
        print(event)
        return event

    def plot_swr_event(self, events_df, event_idx, filter_path=None, show_info_title=False, window_padding=None, figsize_mm=None, panels_to_plot=None, time_per_mm=None, envelope_mode='zscore', show_peak_dots=False, **kwargs):
        """
        Plot a detailed visualization of a specific SWR event, with modular options.
        Parameters:
        -----------
        events_df : pd.DataFrame
            DataFrame of events (from the probe's event CSV)
        event_idx : int
            Index of the event in the DataFrame (should match CSV index)
        filter_path : str
            Path to the sharp wave filter .npz file
        show_info_title : bool, optional
            Whether to show info title (default: False)
        window_padding : float, optional
            Padding (in seconds) before/after event (overrides default logic)
        figsize_mm : tuple, optional
            Figure size in mm (width, height). Default: (200, 200)
        panels_to_plot : list of str, optional
            List of panels to plot. Options: ['ripple_band_zscore', 'sharp_wave_band_zscore', 'ripple_envelope_zscore', 'sharp_wave_envelope_zscore', 'power_zscore'].
            Default: all five panels.
        time_per_mm : float, optional
            Seconds per mm for time axis scaling. If set, overrides figsize_mm width to match time window.
        envelope_mode : str, optional
            How to plot the envelope. Options: 'zscore' (default) or 'raw'.
            'zscore' plots the z-scored envelope, 'raw' plots the smoothed envelope.
        show_peak_dots : bool, optional
            Whether to plot a dot at the peak (max z-score) for each envelope and power trace (default: False)
        kwargs : dict
            Additional arguments (e.g., save_path)
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if filter_path is None:
            raise ValueError("You must provide filter_path to plot_swr_event. Pass it from your workflow file.")
        event = events_df.loc[event_idx]
        dataset = event['dataset']
        session_id = event['session_id']
        probe_id = event['probe_id']
        base_dir = str(self.base_path)
        lfp_dir = os.path.join(base_dir, self.lfp_sources[dataset], f"lfp_session_{session_id}")
        ripple_pattern = os.path.join(lfp_dir, f"probe_{probe_id}_channel*_lfp_ca1_putative_pyramidal_layer.npz")
        sharp_wave_pattern = os.path.join(lfp_dir, f"probe_{probe_id}_channel*_lfp_ca1_putative_str_radiatum.npz")
        time_pattern = os.path.join(lfp_dir, f"probe_{probe_id}_channel*_lfp_time_index_1500hz.npz")
        ripple_files = glob.glob(ripple_pattern)
        sharp_wave_files = glob.glob(sharp_wave_pattern)
        time_files = glob.glob(time_pattern)
        if not ripple_files or not sharp_wave_files or not time_files:
            raise FileNotFoundError(f"Could not find required LFP files for probe {probe_id} in session {session_id}")
        ripple_file = ripple_files[0]
        sharp_wave_file = sharp_wave_files[0]
        time_file = time_files[0]
        ripple_data = np.load(ripple_file)
        sharp_wave_data = np.load(sharp_wave_file)
        time_data = np.load(time_file)
        raw_lfp_signal = ripple_data['array'] if 'array' in ripple_data else ripple_data[ripple_data.files[0]]
        sharp_wave_signal = sharp_wave_data['array'] if 'array' in sharp_wave_data else sharp_wave_data[sharp_wave_data.files[0]]
        time_stamps = time_data['array'] if 'array' in time_data else time_data[time_data.files[0]]
        ripple_band_signal = filter_ripple_band(raw_lfp_signal[:, None])
        sw_filter_data = np.load(filter_path)
        sw_filter = sw_filter_data['sharpwave_componenet_8to40band_1500hz_band']
        sharp_wave_filtered_full = fftconvolve(sharp_wave_signal, sw_filter, mode='same')
        # Modular window padding
        if window_padding is not None:
            pad = window_padding
        else:
            pad = max(0.02, event['duration'] * 0.5)
        start_time = max(time_stamps[0], event['start_time'] - pad)
        end_time = min(time_stamps[-1], event['end_time'] + pad)
        start_idx = np.searchsorted(time_stamps, start_time)
        end_idx = np.searchsorted(time_stamps, end_time)
        time_window = time_stamps[start_idx:end_idx]
        ripple_band_window = ripple_band_signal[start_idx:end_idx]
        sharp_wave_window = sharp_wave_signal[start_idx:end_idx]
        sharp_wave_filtered_window = sharp_wave_filtered_full[start_idx:end_idx]
        raw_lfp_window = raw_lfp_signal[start_idx:end_idx]
        raw_lfp_window_uv = raw_lfp_window * 1e6
        sharp_wave_window_uv = sharp_wave_window * 1e6
        ripple_band_window_uv = ripple_band_window * 1e6
        sharp_wave_filtered_window_uv = sharp_wave_filtered_window * 1e6
        lfp_min = min(raw_lfp_window_uv.min(), sharp_wave_window_uv.min())
        lfp_max = max(raw_lfp_window_uv.max(), sharp_wave_window_uv.max())
        # Calculate y_margin for LFP plots early
        y_margin = (lfp_max - lfp_min) * 0.05

        ripple_envelope = np.abs(hilbert(ripple_band_window_uv))
        sharp_wave_envelope = np.abs(hilbert(sharp_wave_filtered_window_uv))
        
        # Smooth envelopes using gaussian_smooth from ripple_detection
        smoothing_sigma = 0.004  # 4ms smoothing
        sampling_frequency = 1500
        ripple_envelope_smooth = gaussian_smooth(ripple_envelope, sigma=smoothing_sigma, sampling_frequency=sampling_frequency)
        sharp_wave_envelope_smooth = gaussian_smooth(sharp_wave_envelope, sigma=smoothing_sigma, sampling_frequency=sampling_frequency)
        
        # --- Compute full-session signals and z-score them ---
        # Ripple band
        full_ripple_file = os.path.join(lfp_dir, f"probe_{probe_id}_channel*_lfp_ca1_putative_pyramidal_layer.npz")
        full_ripple_files = glob.glob(full_ripple_file)
        full_ripple_data = np.load(full_ripple_files[0])
        full_ripple_signal = full_ripple_data['array'] if 'array' in full_ripple_data else full_ripple_data[full_ripple_data.files[0]]
        full_ripple_band = filter_ripple_band(full_ripple_signal[:, None])
        
        # Compute envelope and smooth it for full session
        full_envelope = np.abs(hilbert(full_ripple_band))
        smoothing_sigma = 0.004  # 4ms smoothing
        sampling_frequency = 1500
        full_envelope_smooth = gaussian_smooth(full_envelope, sigma=smoothing_sigma, sampling_frequency=sampling_frequency)
        full_envelope_smooth_z = zscore(full_envelope_smooth)
        
        # Compute power from smoothed envelope for full session
        full_ripple_power = full_envelope_smooth ** 2
        # Z-score the power using global statistics
        full_ripple_power_z = (full_ripple_power - np.mean(full_ripple_power)) / np.std(full_ripple_power)

        # Sharp wave band
        full_sharp_wave_file = os.path.join(lfp_dir, f"probe_{probe_id}_channel*_lfp_ca1_putative_str_radiatum.npz")
        full_sharp_wave_files = glob.glob(full_sharp_wave_file)
        full_sharp_wave_data = np.load(full_sharp_wave_files[0])
        full_sharp_wave_signal = full_sharp_wave_data['array'] if 'array' in full_sharp_wave_data else full_sharp_wave_data[full_sharp_wave_data.files[0]]
        sw_filter_data = np.load(filter_path)
        sw_filter = sw_filter_data['sharpwave_componenet_8to40band_1500hz_band']
        full_sharp_wave_filtered = fftconvolve(full_sharp_wave_signal, sw_filter, mode='same')
        full_sharp_wave_band_z = zscore(full_sharp_wave_filtered)
        full_sharp_wave_envelope = np.abs(hilbert(full_sharp_wave_filtered))
        full_sharp_wave_envelope_smooth = gaussian_smooth(full_sharp_wave_envelope, sigma=smoothing_sigma, sampling_frequency=sampling_frequency)
        full_sharp_wave_envelope_z = zscore(full_sharp_wave_envelope_smooth)
        full_sharp_wave_power = full_sharp_wave_envelope_smooth ** 2
        full_sharp_wave_power_z = (full_sharp_wave_power - np.mean(full_sharp_wave_power)) / np.std(full_sharp_wave_power)

        # --- Extract event window from z-scored full-session arrays ---
        start_idx = np.searchsorted(time_stamps, start_time)
        end_idx = np.searchsorted(time_stamps, end_time)
        time_window = time_stamps[start_idx:end_idx]
        time_rel = time_window - (event['start_time'] + event['duration']/2)
        ripple_band_window_z = full_ripple_band_z[start_idx:end_idx].squeeze()
        sharp_wave_band_window_z = full_sharp_wave_band_z[start_idx:end_idx].squeeze()
        ripple_envelope_window_z = full_envelope_smooth_z[start_idx:end_idx].squeeze()
        sharp_wave_envelope_window_z = full_sharp_wave_envelope_z[start_idx:end_idx].squeeze()
        ripple_power_window_z = full_ripple_power_z[start_idx:end_idx].squeeze()
        sharp_wave_power_window_z = full_sharp_wave_power_z[start_idx:end_idx].squeeze()

        # Modular panel selection
        all_panels = [
            'raw_pyramidal_lfp',
            'raw_s_radiatum_lfp',
            'bandpass_signals',
            'envelope',
            'power',
        ]
        if panels_to_plot is None:
            panels_to_plot = all_panels
        n_panels = len(panels_to_plot)
        # Modular time-to-mm scaling
        time_window_size = end_time - start_time
        if show_info_title:
            height_mm = 450
        else:
            height_mm = 200
        width_mm = time_window_size * 500  # Proportional width
        figsize = (width_mm / 25.4, height_mm / 25.4)  # Convert to inches for matplotlib

        fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True, constrained_layout=show_info_title)
        if n_panels == 1:
            axes = [axes]
        panel_map = {name: i for i, name in enumerate(panels_to_plot)}

        # Use event['peak_time'] if available, else midpoint
        event_peak = event['peak_time'] if 'peak_time' in event else event['start_time'] + event['duration']/2
        # 1. Raw LFP (Pyramidal)
        if 'raw_pyramidal_lfp' in panels_to_plot:
            ax = axes[panel_map['raw_pyramidal_lfp']]
            ax.plot(time_rel, raw_lfp_window_uv, color='black', linewidth=0.5, label='Raw LFP (Pyramidal)')
            ax.axvspan(event['start_time']-event_peak, event['end_time']-event_peak, alpha=0.3, color='green')
            ax.set_ylabel('Pyramidal Layer LFP (μV)', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', frameon=True, fancybox=False)
            ax.margins(x=0)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

        # 2. Raw LFP (Sharp Wave)
        if 'raw_s_radiatum_lfp' in panels_to_plot:
            ax = axes[panel_map['raw_s_radiatum_lfp']]
            ax.plot(time_rel, sharp_wave_window_uv, color='blue', linewidth=0.5, label='Raw LFP (Sharp Wave)')
            ax.axvspan(event['start_time']-event_peak, event['end_time']-event_peak, alpha=0.3, color='green')
            ax.set_ylabel('S. Radiatum LFP (μV)', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', frameon=True, fancybox=False)
            ax.margins(x=0)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

        # 3. Bandpass signals (z-scored)
        if 'bandpass_signals' in panels_to_plot:
            ax = axes[panel_map['bandpass_signals']]
            ax.plot(time_rel, ripple_band_window_z, color='black', linewidth=1.0, label='Ripple Band (Z-scored)')
            ax.plot(time_rel, sharp_wave_band_window_z, color='blue', linewidth=1.0, label='Sharp Wave Band (Z-scored)')
            ax.axvspan(event['start_time']-event_peak, event['end_time']-event_peak, alpha=0.3, color='green')
            ax.set_ylabel('Bandpass (Z-scored)', fontsize=12, fontweight='bold')
            ax.margins(x=0)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

        # 4. Envelope (z-scored)
        if 'envelope' in panels_to_plot:
            ax = axes[panel_map['envelope']]
            ax.plot(time_rel, ripple_envelope_window_z, color='black', linewidth=1.0, label='Ripple Envelope (Z-scored)')
            ax.plot(time_rel, sharp_wave_envelope_window_z, color='blue', linewidth=1.0, label='Sharp Wave Envelope (Z-scored)')
            # Compute local envelope peak in the event window
            local_env_peak_idx = np.argmax(ripple_envelope_window_z)
            local_env_peak_time = time_rel[local_env_peak_idx]
            local_env_peak_value = ripple_envelope_window_z[local_env_peak_idx]
            ax.plot(local_env_peak_time, local_env_peak_value, 'o', color='black', markersize=7, markeredgecolor='white', label='Envelope Max Z-score')
            # Add 2 SD threshold line
            ax.axhline(2, color='black', linestyle='--', label='Envelope Threshold (2 SD)')
            ax.axvspan(event['start_time']-event_peak, event['end_time']-event_peak, alpha=0.3, color='green')
            ax.set_ylabel('Envelope (Z-scored)', fontsize=12, fontweight='bold')
            ax.margins(x=0)  # Remove x margin only
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

        # 5. Power (z-scored)
        if 'power' in panels_to_plot:
            ax = axes[panel_map['power']]
            ax.plot(time_rel, ripple_power_window_z, color='black', linewidth=1.0, label='Ripple Power (Z-scored)')
            ax.plot(time_rel, sharp_wave_power_window_z, color='blue', linewidth=1.0, label='Sharp Wave Power (Z-scored)')
            if show_peak_dots:
                # Ripple power peak: use power_peak_time and power_max_zscore
                ripple_power_peak_time = event.get('power_peak_time', None)
                ripple_power_peak_z = event.get('power_max_zscore', None)
                if ripple_power_peak_time is not None and ripple_power_peak_z is not None:
                    ax.plot(ripple_power_peak_time - (event['peak_time'] if 'peak_time' in event else event['start_time'] + event['duration']/2), ripple_power_peak_z, 'o', color='black', markersize=7, markeredgecolor='white', label='Ripple Power Peak')
                # Sharp wave power peak: use sw_peak_time and sw_peak_power (these should be referenced to event_peak)
                sharp_wave_power_peak_time = event.get('sw_peak_time', None)
                sharp_wave_power_peak_z = event.get('sw_peak_power', None)
                if sharp_wave_power_peak_time is not None and sharp_wave_power_peak_z is not None:
                    ax.plot(sharp_wave_power_peak_time - event_peak, sharp_wave_power_peak_z, 'o', color='blue', markersize=7, markeredgecolor='white', label='Sharp Wave Power Peak')
            ax.axhline(2.5, color='grey', linestyle='--', label='Ripple Threshold')
            ax.axhline(1, color='blue', linestyle='--', label='Sharp Wave Threshold')
            ax.axvspan(event['start_time']-event_peak, event['end_time']-event_peak, alpha=0.3, color='green')
            ax.set_ylabel('Power (Z-scored)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time from Ripple Peak (s)', fontsize=12, fontweight='bold')
            ax.margins(x=0)  # Remove x margin only
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            # Only add legend to the bottom panel, below the plot
            if show_info_title:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=8, frameon=True)
            else:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9, frameon=True)

        # Display key values at the bottom of the figure
        # Use event DataFrame columns for info block values
        envelope_max_zscore = event.get('envelope_max_zscore', np.nan)
        power_max_zscore = event.get('power_max_zscore', np.nan)
        sw_peak_power = event.get('sw_peak_power', np.nan)
        fig.text(0.5, 0.01, f"envelope_max_zscore: {envelope_max_zscore:.2f}   power_max_zscore: {power_max_zscore:.2f}   sw_peak_power: {sw_peak_power:.2f}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add detailed info title if requested
        if show_info_title:
            info_text = (
                f"Duration: {event['duration']*1000:.1f} ms\n"
                f"Max {'Z-score' if envelope_mode=='zscore' else 'Envelope'}: {event.get('envelope_max_zscore', np.nan):.2f}\n"
                f"SW-Ripple PLV: {event.get('sw_ripple_plv', np.nan):.3f}, SW-Ripple MI: {event.get('sw_ripple_mi', np.nan):.3f}, SW-Ripple CLCorr: {event.get('sw_ripple_clcorr', np.nan):.3f}"
            )
            fig.suptitle(info_text, fontsize=12, fontweight='bold')
            plt.subplots_adjust(top=0.72)
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        else:
            plt.tight_layout(rect=[0, 0.03, 1, 1])
        if 'save_path' in kwargs and kwargs['save_path']:
            dpi = kwargs.get('dpi', 300)  # Default to 300 DPI for high resolution
            fig.savefig(kwargs['save_path'], format='svg', dpi=dpi, bbox_inches='tight')
            if kwargs.get('save_png', False):
                png_path = kwargs['save_path'].rsplit('.', 1)[0] + '.png'
                fig.savefig(png_path, format='png', dpi=dpi, bbox_inches='tight')
        return fig
    
    def get_allensdk_speed(self, running_speed_df, events_df, event_index='all', window=0.5, agg='mean'):
        """
        Get speed statistics for events using Allen SDK speed data.
        
        Parameters:
        -----------
        running_speed_df : pd.DataFrame
            DataFrame with 'timestamps' and 'speed' columns from Allen SDK
        events_df : pd.DataFrame
            DataFrame containing events with 'start_time' and 'end_time' columns
        event_index : str, list, or bool, optional
            - 'all' (default): process all events
            - list: indices of events to process
            - bool: boolean mask for events to process
        window : float
            Time window (in seconds) to consider before and after the event
        agg : str
            Aggregation method: 'mean', 'median', 'max', or 'min'
            
        Returns:
        --------
        pd.Series
            Speed statistics for the specified events
        """
        # Validate aggregation method
        valid_aggs = ['mean', 'median', 'max', 'min']
        if agg not in valid_aggs:
            raise ValueError(f"agg must be one of {valid_aggs}")
            
        # Get aggregation function
        agg_func = getattr(np, agg)
        
        def get_speed_stats(start, end):
            # Calculate window boundaries
            window_start = start - window
            window_end = end + window
            
            # Filter speed data within the window
            mask = (running_speed_df['timestamps'] >= window_start) & (running_speed_df['timestamps'] <= window_end)
            window_speeds = running_speed_df.loc[mask, 'speed']
            
            if len(window_speeds) > 0:
                return agg_func(np.abs(window_speeds))
            return 0
        
        # Handle event selection
        if event_index == 'all':
            event_mask = slice(None)
        elif isinstance(event_index, pd.Series) and event_index.dtype == bool:
            # A boolean Series mask
            event_mask = event_index
        elif isinstance(event_index, list):
            event_mask = event_index
        else:
            raise ValueError("event_index must be 'all', a list, or a boolean Series mask")

            
        # Get speeds for selected events
        speeds = []
        for _, event in events_df.iloc[event_mask].iterrows():
            speed = get_speed_stats(event['start_time'], event['end_time'])
            speeds.append(speed)
            
        return pd.Series(speeds, index=events_df.iloc[event_mask].index)
    
    def filter_events_by_allensdk_speed(self, events_df, running_speed_df, max_speed=5.0, window=0.5, agg='max'):
        """
        Filter events based on Allen SDK speed data.
        
        Parameters:
        -----------
        events_df : pd.DataFrame
            DataFrame containing events with 'start_time' and 'end_time' columns
        running_speed_df : pd.DataFrame
            DataFrame with 'timestamps' and 'speed' columns from Allen SDK
        max_speed : float
            Maximum allowed speed during event window
        window : float
            Time window (in seconds) to consider before and after each event
        agg : str
            Aggregation method: 'mean', 'median', 'max', or 'min'
            
        Returns:
        --------
        pd.DataFrame
            Events with speed below threshold in their windows
        """
        # Get speeds for all events
        speeds = self.get_allensdk_speed(
            running_speed_df=running_speed_df,
            events_df=events_df,
            window=window,
            agg=agg
        )
        
        # Add speed column and filter
        events_df['max_speed_in_window'] = speeds
        filtered_events = events_df[events_df['max_speed_in_window'] <= max_speed]
        
        # The filtered events will automatically maintain the same index structure
        # as the original events_df, so we don't need to manually set it
        return filtered_events

    def plot_global_swr_event(self, dataset, session_id, global_event_idx, filter_path=None, window=0.15, output_dir=None, file_ext='png', show_ap_in_ylabel=False, additional_value='envelope', relative_time=True):
        """
        Plot a global SWR event, showing LFP (grey) and a selected additional value for all probes in the session, sorted by anterior-posterior coordinate.
        The output file extension is controlled by the file_ext parameter (default 'png').
        The function constructs the save_path using event/session info and file_ext.
        Only one file is saved, with the requested extension.

        Parameters:
        - dataset: str
        - session_id: str
        - global_event_idx: int
        - filter_path: str
        - window: float (seconds, half-width of window around center)
        - output_dir: str or None
        - file_ext: str
        - show_ap_in_ylabel: bool
        - additional_value: str
        - relative_time: bool (if True, plot time relative to global peak; if False, plot absolute time)
        """
        print(f"\n[DEBUG] --- Starting plot_global_swr_event for global_event_idx={global_event_idx} ---")
        print(f"[DEBUG] relative_time: {relative_time}")
        # Settings for smoothing
        smoothing_sigma = 0.004  # 4ms
        sampling_frequency = 1500

        # Load global events
        global_events_path = self.base_path / dataset / f"swrs_session_{session_id}" / f"session_{session_id}_global_swr_events.csv.gz"
        if not global_events_path.exists():
            print(f"[ERROR] Global events file not found: {global_events_path}")
            return None
        global_events = pd.read_csv(global_events_path, index_col=0)
        
        # Get the event
        if global_event_idx not in global_events.index:
            print(f"[ERROR] Global event {global_event_idx} not found in events DataFrame")
            return None
        event = global_events.loc[global_event_idx]
        
        # Precompute probe info for all probes in the session
        all_probes = []
        probe_ap_dict = {}
        lfp_dir = self.base_path / self.lfp_sources[dataset] / f"lfp_session_{session_id}"
        for lfp_file in lfp_dir.glob("probe_*_channel*_lfp_ca1_putative_pyramidal_layer.npz"):
            m_probe = lfp_file.name.split('_')[1]
            m_chan = lfp_file.name.split('_')[3]
            ap = self.get_channel_ap_coordinate(int(m_chan)) if m_chan.isdigit() else None
            probe_ap_dict[m_probe] = ap
            all_probes.append(m_probe)
        
        # Sort all probes by AP coordinate
        probe_ap_list = [(pid, probe_ap_dict.get(pid, float('inf'))) for pid in all_probes]
        probe_ap_list.sort(key=lambda x: x[1])
        sorted_probe_ids = [p[0] for p in probe_ap_list]
        
        # Prepare participation info
        participating_probes = ast.literal_eval(event['participating_probes'])
        peak_times = ast.literal_eval(event['peak_times'])
        probe_to_peak_time = {pid: pt for pid, pt in zip(participating_probes, peak_times)}
        global_start_time = event['start_time']
        global_end_time = event['end_time']
        global_peak_time = event['global_peak_time']
        
        # Always work in absolute time for windowing
        plot_start = global_peak_time - window
        plot_end = global_peak_time + window
        
        min_x, max_x = float('inf'), float('-inf')
        all_time_windows = []
        all_lfp_traces = []
        all_additional_traces = []
        all_probe_labels = []
        
        for probe_id in sorted_probe_ids:
            lfp_files = list(lfp_dir.glob(f"probe_{probe_id}_channel*_lfp_ca1_putative_pyramidal_layer.npz"))
            time_files = list(lfp_dir.glob(f"probe_{probe_id}_channel*_lfp_time_index_1500hz.npz"))
            
            if not lfp_files or not time_files:
                print(f"[WARNING] Missing LFP or time index files for probe {probe_id}")
                continue
                
            # Load LFP data and time index
            lfp_data = np.load(lfp_files[0])
            time_data = np.load(time_files[0])
            
            # Get the first key from the time index file
            time_key = time_data.files[0]
            time_vals = time_data[time_key]
            lfp_vals = lfp_data['lfp_ca1'] * 1e6  # Convert to microvolts
            
            # Compute additional signal
            ripple_band = filter_ripple_band(lfp_vals[:, None]).squeeze()
            if additional_value == 'ripple_band':
                additional = zscore(ripple_band)
            elif additional_value == 'envelope':
                envelope = np.abs(hilbert(ripple_band))
                envelope_smooth = gaussian_smooth(envelope, sigma=smoothing_sigma, sampling_frequency=sampling_frequency)
                additional = zscore(envelope_smooth)
            elif additional_value == 'power':
                envelope = np.abs(hilbert(ripple_band))
                envelope_smooth = gaussian_smooth(envelope, sigma=smoothing_sigma, sampling_frequency=sampling_frequency)
                power = envelope_smooth ** 2
                additional = zscore(power)
            else:
                raise ValueError(f"Unknown additional_value: {additional_value}")
            
            # Define window for plotting (always in absolute time)
            mask = (time_vals >= plot_start) & (time_vals <= plot_end)
            t_window = time_vals[mask]
            lfp_window = lfp_vals[mask]
            additional_window = additional[mask]
            
            # Convert to relative time only for display if needed
            if relative_time:
                t_plot = t_window - global_peak_time
            else:
                t_plot = t_window
            
            # Store for axis limits
            all_time_windows.append(t_plot)
            all_lfp_traces.append(lfp_window)
            all_additional_traces.append(additional_window)
            all_probe_labels.append(f"Probe {probe_id} (AP: {probe_ap_dict.get(probe_id, np.nan):.1f})" if show_ap_in_ylabel else f"Probe {probe_id}")
            min_x = min(min_x, t_plot.min())
            max_x = max(max_x, t_plot.max())
        
        # Convert global event boundaries for display
        if relative_time:
            global_start_plot = global_start_time - global_peak_time
            global_end_plot = global_end_time - global_peak_time
        else:
            global_start_plot = global_start_time
            global_end_plot = global_end_time
        
        fig, axes = plt.subplots(len(all_lfp_traces), 1, figsize=(12, 2.5*len(all_lfp_traces)), sharex=True)
        if len(all_lfp_traces) == 1:
            axes = [axes]
        
        for i, (ax, t, lfp, add, label, probe_id) in enumerate(zip(axes, all_time_windows, all_lfp_traces, all_additional_traces, all_probe_labels, sorted_probe_ids)):
            ax.plot(t, lfp, color='gray', linewidth=0.7, label='LFP (μV)')
            ax.set_ylabel('LFP (μV)')
            
            # Plot additional signal (right y-axis)
            ax2 = ax.twinx()
            ax2.plot(t, add, color='black', linewidth=1.0, label=additional_value)
            ax2.set_ylabel(f'{additional_value.replace("_", " ").title()} (z-scored)')
            
            if probe_id in probe_to_peak_time:
                peak_time = probe_to_peak_time[probe_id]
                probe_event_files = list((self.base_path / dataset / f"swrs_session_{session_id}").glob(f"probe_{probe_id}_channel*_putative_swr_events.csv.gz"))
                if probe_event_files:
                    probe_events = pd.read_csv(probe_event_files[0])
                    match_rows = probe_events[probe_events['power_peak_time'] == peak_time]
                    if not match_rows.empty:
                        matching_event = match_rows.iloc[0]
                        
                        # Convert probe event times for display
                        if relative_time:
                            event_start_plot = matching_event['start_time'] - global_peak_time
                            event_end_plot = matching_event['end_time'] - global_peak_time
                            peak_time_plot = matching_event['power_peak_time'] - global_peak_time
                        else:
                            event_start_plot = matching_event['start_time']
                            event_end_plot = matching_event['end_time']
                            peak_time_plot = matching_event['power_peak_time']
                        
                        # Plot green shaded area for probe-level event window
                        ax.axvspan(event_start_plot, event_end_plot, color='green', alpha=0.3)
                        # Plot black dotted line for probe-level event peak
                        ax.axvline(peak_time_plot, color='black', linestyle='--', linewidth=1)
                        # Plot green dotted lines for global event window
                        ax.axvline(global_start_plot, color='green', linestyle=':', linewidth=1)
                        ax.axvline(global_end_plot, color='green', linestyle=':', linewidth=1)
            
            ax.set_title(label)
            ax.margins(x=0)
            ax2.margins(x=0)
            
            # Legend
            if i == 0:
                ax.legend(loc='upper right')
                ax2.legend(loc='upper left')
        
        axes[-1].set_xlabel('Time from Global Peak (s)' if relative_time else 'Time (s)')
        plt.xlim(min_x, max_x)
        plt.tight_layout()
        
        # Save figure
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"global_swr_event_{global_event_idx}_session_{session_id}.{file_ext}"
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved figure to {save_path}")
        return fig

    def plot_global_event_CSD_slice(self, dataset, session_id, global_event_idx, filter_path=None, 
                                    window=0.15, save_path=None, ca1_only=True, show_scale=True, file_ext='png'):
        """
        UNDER CONSTRUCTION

        Plot a CSD slice for a global event, overlaying the CSD from CA1 channels (or all hippocampus) on a sagittal view.
        - For each probe in CA1, extract a window of LFP data around the event peak.
        - Compute the CSD (second spatial derivative) across channels, smoothing in time.
        - Overlay the CSD as a heatmap on a schematic sagittal view (vertical vs. horizontal position).
        - Add a colorbar and axis labels. Save or show the figure.
        
        This method runs the plotting in a separate process using the brainglobe conda environment
        to avoid dependency conflicts with allensdk.
        
        Parameters:
            dataset: str
            session_id: str
            global_event_idx: int
            filter_path: str or None
            window: float (seconds, window around event peak)
            save_path: str or None
            ca1_only: bool (default True, only plot CA1 channels)
            show_scale: bool (default True, show colorbar)
            file_ext: str (default 'png') - file extension for the saved image
        """

        # 1. Load global event and get peak time
        session_dir = self.base_path / dataset / f"swrs_session_{session_id}"
        global_csv = list(session_dir.glob(f"session_{session_id}_global_swr_events.csv.gz"))
        if not global_csv:
            raise FileNotFoundError(f"No global event CSV found for session {session_id}")
        global_events = pd.read_csv(global_csv[0], index_col=0)
        event = global_events.loc[global_event_idx]
        peak_time = np.median(ast.literal_eval(event['peak_times']))

        # 2. Find all CA1 probe/channel files for this session
        lfp_dir = self.base_path / self.lfp_sources[dataset] / f"lfp_session_{session_id}"
        probe_files = list(lfp_dir.glob("probe_*_channel*_lfp_ca1_putative_pyramidal_layer.npz"))

        # 3. For each probe, load LFP and channel info
        csd_matrices = []
        vert_pos_list = []
        horiz_pos_list = []
        ap_list = []
        ml_list = []
        time_axis = None
        pyramidal_layer_pos = []
        sharpwave_layer_pos = []

        # Use AllenSDK precomputed CSD for each probe
        for lfp_file in probe_files:
            # Extract probe and channel info
            m_probe = re.search(r'probe_(\d+)_channel', lfp_file.name)

            probe_id = m_probe.group(1)
            # Get the session object for this session
            session = self.allensdk_cache.get_ecephys_session(ecephys_session_id=int(session_id))
            # Debug: check if get_current_source_density is a method of the session object
            has_csd_method = hasattr(session, 'get_current_source_density')
            print(f"Session object has get_current_source_density: {has_csd_method}")
            if not has_csd_method:
                print(f"Session object methods: {dir(session)}")
            csd = session.get_current_source_density(int(probe_id))
            print("[DEBUG] csd xarray object:")
            print(csd)
            # Get the time window indices
            time = csd['time'].values
            mask = (time >= (peak_time - window)) & (time <= (peak_time + window))
            csd_window = csd.data[:, mask]
            time_axis = time[mask]
            vertical_positions = csd['vertical_position'].values
            ap_coord = None
            ml_coord = None
            # Try to get AP/ML from channel_table if available
            if self.channel_table is not None and int(probe_id) in self.channel_table['ecephys_probe_id'].values:
                probe_rows = self.channel_table[self.channel_table['ecephys_probe_id'] == int(probe_id)]
                if not probe_rows.empty:
                    ap_coord = probe_rows['anterior_posterior_ccf_coordinate'].iloc[0]
                    ml_coord = probe_rows['medial_lateral_ccf_coordinate'].iloc[0] if 'medial_lateral_ccf_coordinate' in probe_rows else 0
            # Save for plotting
            csd_matrices.append(csd_window)
            vert_pos_list.append(vertical_positions)
            horiz_pos_list.append(np.full_like(vertical_positions, ap_coord if ap_coord is not None else 0))
            ap_list.append(ap_coord if ap_coord is not None else 0)
            ml_list.append(ml_coord if ml_coord is not None else 0)

            # Mark pyramidal and sharpwave layers if available (optional, placeholder)
            pyramidal_layer_pos.append(np.nanmin(vertical_positions))
            sharpwave_layer_pos.append(np.nanmax(vertical_positions))

        # 4. Stack CSDs and positions for plotting
        if not csd_matrices:
            print("No CSD data found for this event.")
            return None

        csd_all = np.concatenate(csd_matrices, axis=0)
        vert_all = np.concatenate(vert_pos_list, axis=0)
        horiz_all = np.concatenate(horiz_pos_list, axis=0)
        
        # Debug: print shapes and types before writing
        print("[DEBUG] Preparing to write CSD data to temp file...")
        print(f"[DEBUG] csd_all shape: {np.array(csd_all).shape}, dtype: {np.array(csd_all).dtype}")
        print(f"[DEBUG] vert_all shape: {np.array(vert_all).shape}, dtype: {np.array(vert_all).dtype}")
        print(f"[DEBUG] horiz_all shape: {np.array(horiz_all).shape}, dtype: {np.array(horiz_all).dtype}")
        print(f"[DEBUG] time_axis shape: {np.array(time_axis).shape}, dtype: {np.array(time_axis).dtype}")
        print(f"[DEBUG] pyramidal_layer_pos: {pyramidal_layer_pos}")
        print(f"[DEBUG] sharpwave_layer_pos: {sharpwave_layer_pos}")
        with tempfile.NamedTemporaryFile(prefix='temp_csdslice_', suffix='.json', delete=False, mode='w') as temp_file:
            data = {
                'csd_all': np.array(csd_all).tolist(),
                'vert_all': np.array(vert_all).tolist(),
                'horiz_all': np.array(horiz_all).tolist(),
                'time_axis': np.array(time_axis).tolist(),
                'pyramidal_layer_pos': [float(x) for x in pyramidal_layer_pos],
                'sharpwave_layer_pos': [float(x) for x in sharpwave_layer_pos]
            }
            json.dump(data, temp_file)
            temp_path = temp_file.name
        print(f"[DEBUG] CSD data written to temp file: {temp_path}")

        try:
            # Create a dedicated folder for CSD and brainglobe images
            csd_folder = os.path.join(os.path.dirname(save_path), 'csd_brainglobe_images')
            os.makedirs(csd_folder, exist_ok=True)
            
            # Define paths for CSD and brainglobe images
            csd_save_path = os.path.join(csd_folder, f'csd_heatmap_{session_id}_{global_event_idx}.{file_ext}')
            brainglobe_save_path = os.path.join(csd_folder, f'brainglobe_sagittal_{session_id}_{global_event_idx}.{file_ext}')
            
            # Update the plot_script to save the CSD heatmap
            plot_script = f'''
import json
import numpy as np
import matplotlib.pyplot as plt
import brainglobe_heatmap as bgh
print("[PLOT DEBUG] Loading CSD data from file...")
with open("{temp_path}", "r") as f:
    data = json.load(f)
print("[PLOT DEBUG] Data loaded. Converting to numpy arrays...")
def safe_shape(arr):
    try:
        return arr.shape
    except Exception:
        return 'None/empty'
def safe_dtype(arr):
    try:
        return arr.dtype
    except Exception:
        return 'None/empty'
csd = np.array(data.get('csd_all', []))
vert = np.array(data.get('vert_all', []))
horiz = np.array(data.get('horiz_all', []))
time = np.array(data.get('time_axis', []))
print(f"[PLOT DEBUG] csd shape: {{safe_shape(csd)}}, dtype: {{safe_dtype(csd)}}")
print(f"[PLOT DEBUG] vert shape: {{safe_shape(vert)}}, dtype: {{safe_dtype(vert)}}")
print(f"[PLOT DEBUG] horiz shape: {{safe_shape(horiz)}}, dtype: {{safe_dtype(horiz)}}")
print(f"[PLOT DEBUG] time shape: {{safe_shape(time)}}, dtype: {{safe_dtype(time)}}")
print("[PLOT DEBUG] Creating heatmap...")
heatmap = bgh.Heatmap(
    {{}},
    position=None,
    orientation="sagittal",
    thickness=1000,
    atlas_name="allen_mouse_25um",
    format="2D"
)
heatmap.show(figsize=(8, 6))
ax = plt.gca()
print("[PLOT DEBUG] Plotting overlays...")
if csd.size == 0 or vert.size == 0 or time.size == 0:
    print("No CSD data to plot for this event/probe.")
    exit(0)
extent = [time[0], time[-1], vert[0], vert[-1]]
ax.imshow(csd, aspect='auto', extent=extent, cmap='bwr', alpha=0.7, origin='lower')
for pyr in data.get('pyramidal_layer_pos', []):
    ax.axhline(pyr, color='magenta', linestyle='--', label='Pyramidal Layer')
for sw in data.get('sharpwave_layer_pos', []):
    ax.axhline(sw, color='cyan', linestyle='--', label='Sharpwave Layer')
print("[PLOT DEBUG] Saving CSD heatmap...")
plt.savefig("{csd_save_path}", dpi=300, bbox_inches='tight', format='{file_ext}')
print("[PLOT DEBUG] Done.")
'''
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as script_file:
                script_file.write(plot_script)
                script_path = script_file.name

            try:
                cmd = f"conda run -n brainglobe-env python {script_path}"
                subprocess.run(cmd, shell=True, check=True)

            finally:
                # Clean up temporary script file
                os.unlink(script_path)

        finally:
            # Clean up temporary data file
            os.unlink(temp_path)

        return None  # Figure is saved or shown by the subprocess

    def CSD_average_session(self, session_id, dataset='allen_visbehave_swr_murphylab2024', output_dir=None, window=0.25, plot_window=0.15, plot_brainglobe=True, file_ext='png'):
        """
        UNDER CONSTRUCTION
        
        Compute and plot the average CSD across all CA1 channels and all events in a session.
        - For each probe with CA1, load LFP and event peaks.
        - For each event, extract a window (±window s), compute CSD, and average.
        - Plot the average CSD (±plot_window s) as a heatmap.
        - Optionally, plot a brainglobe sagittal slice with CA1 channel locations.
        """
        import gzip, json, numpy as np, os
        import matplotlib.pyplot as plt
        from scipy.signal import savgol_filter
        from scipy.ndimage import gaussian_filter
        from pathlib import Path
        import pandas as pd
        
        if output_dir is None:
            output_dir = f"./csd_average_session_{session_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Find all probes with CA1 for this session
        session_dir = self.base_path / dataset / f"swrs_session_{session_id}"
        lfp_dir = self.base_path / self.lfp_sources[dataset] / f"lfp_session_{session_id}"
        probe_metadata_path = session_dir / f"session_{session_id}_probe_metadata.csv.gz"
        if not probe_metadata_path.exists():
            print(f"[CSD AVG] Probe metadata not found: {probe_metadata_path}")
            return
        probe_metadata = pd.read_csv(probe_metadata_path)
        ca1_probes = probe_metadata[probe_metadata['ca1_channel_count'] > 0]['probe_id'].astype(str).tolist()
        print(f"[CSD AVG] Probes with CA1: {ca1_probes}")
        
        all_csd = []
        all_vert = []
        all_time = None
        pyramidal_coords = []
        radiatum_coords = []
        ca1_coords = []
        for probe_id in ca1_probes:
            # Load channel selection metadata
            chan_meta_path = session_dir / f"probe_{probe_id}_channel_selection_metadata.json.gz"
            if not chan_meta_path.exists():
                print(f"[CSD AVG] Channel selection metadata not found for probe {probe_id}")
                continue
            with gzip.open(chan_meta_path, 'rt') as f:
                chan_meta = json.load(f)
            ca1_chan_ids = chan_meta['ripple_band']['channel_ids']
            ca1_depths = chan_meta['ripple_band']['depths']
            # Check contiguity
            diffs = np.diff(sorted(ca1_depths))
            if np.any(diffs > 50):
                print(f"[CSD AVG] Warning: CA1 channels for probe {probe_id} are not contiguous (depth jumps: {diffs})")
            # Load LFP and time
            lfp_file = lfp_dir / f"probe_{probe_id}_channel_{chan_meta['ripple_band']['selected_channel_id']}_lfp_ca1_putative_pyramidal_layer.npz"
            time_file = lfp_dir / f"probe_{probe_id}_channel_{chan_meta['ripple_band']['selected_channel_id']}_lfp_time_index_1500hz.npz"
            if not lfp_file.exists() or not time_file.exists():
                print(f"[CSD AVG] LFP or time file missing for probe {probe_id}")
                continue
            lfp = np.load(lfp_file)['lfp_ca1']
            time_vals = np.load(time_file)['lfp_time_index']
            # Load events
            events_file = session_dir / f"probe_{probe_id}_channel_{chan_meta['ripple_band']['selected_channel_id']}_putative_swr_events.csv.gz"
            if not events_file.exists():
                print(f"[CSD AVG] Events file missing for probe {probe_id}")
                continue
            events = pd.read_csv(events_file)
            if 'power_peak_time' not in events.columns:
                print(f"[CSD AVG] No power_peak_time in events for probe {probe_id}")
                continue
            # For each event, extract window and compute CSD
            probe_csds = []
            for peak_time in events['power_peak_time']:
                mask = (time_vals >= (peak_time - window)) & (time_vals <= (peak_time + window))
                if np.sum(mask) < 10:
                    continue
                lfp_win = lfp[mask, :]
                # Compute CSD (second spatial derivative along channel axis)
                csd = -np.diff(lfp_win, n=2, axis=1)
                # Pad to match channel count
                csd = np.pad(csd, ((0,0),(1,1)), mode='edge')
                probe_csds.append(csd)
            if not probe_csds:
                print(f"[CSD AVG] No valid CSD windows for probe {probe_id}")
                continue
            avg_csd = np.mean(probe_csds, axis=0)
            # For plotting, restrict to ±plot_window
            center_idx = avg_csd.shape[0] // 2
            plot_pts = int(plot_window * 1500)
            t0 = np.searchsorted(time_vals, events['power_peak_time'].iloc[0])
            t_idxs = np.arange(t0 - plot_pts, t0 + plot_pts)
            t_idxs = t_idxs[(t_idxs >= 0) & (t_idxs < avg_csd.shape[0])]
            avg_csd_plot = avg_csd[t_idxs, :]
            # Save for stacking
            all_csd.append(avg_csd_plot)
            all_vert.append(ca1_depths)
            if all_time is None:
                all_time = time_vals[t_idxs] - time_vals[t0]
            # For anatomical plot
            ca1_coords.extend([(d, probe_id) for d in ca1_depths])
            # Pyramidal and radiatum
            pyramidal_coords.append((min(ca1_depths), probe_id))
            radiatum_coords.append((max(ca1_depths), probe_id))
        if not all_csd:
            print("[CSD AVG] No CSD data found for this session.")
            return
        # Stack and average
        csd_stack = np.stack(all_csd, axis=0)
        csd_avg = np.mean(csd_stack, axis=0)
        # Plot heatmap
        plt.figure(figsize=(8,6))
        plt.imshow(csd_avg.T, aspect='auto', cmap='bwr', origin='lower', extent=[all_time[0], all_time[-1], np.min(all_vert), np.max(all_vert)])
        plt.colorbar(label='CSD (a.u.)')
        plt.xlabel('Time (s)')
        plt.ylabel('CA1 Channel Depth (μm)')
        plt.title(f'Average CSD (Session {session_id})')
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, f'csd_average_session_{session_id}.{file_ext}')
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"[CSD AVG] Saved average CSD heatmap: {heatmap_path}")
        # Optionally plot brainglobe sagittal slice
        if plot_brainglobe:
            try:
                import brainglobe_heatmap as bgh
                # Use mean dorsal-ventral for CA1
                dv_vals = [d for d, _ in ca1_coords]
                mean_dv = np.mean(dv_vals)
                heatmap = bgh.Heatmap({}, position=None, orientation="sagittal", thickness=1000, atlas_name="allen_mouse_25um", format="2D")
                heatmap.show(figsize=(8, 6))
                ax = plt.gca()
                # Plot pyramidal (green) and radiatum (blue) dots
                for d, pid in pyramidal_coords:
                    ax.plot(0, d, 'o', color='green', label='Pyramidal' if pid==pyramidal_coords[0][1] else "")
                for d, pid in radiatum_coords:
                    ax.plot(0, d, 'o', color='blue', label='Str. Radiatum' if pid==radiatum_coords[0][1] else "")
                plt.legend()
                plt.title(f'CA1 Channel Locations (Session {session_id})')
                plt.tight_layout()
                anat_path = os.path.join(output_dir, f'ca1_locations_session_{session_id}.{file_ext}')
                plt.savefig(anat_path, dpi=300)
                plt.close()
                print(f"[CSD AVG] Saved CA1 anatomical plot: {anat_path}")
            except ImportError:
                print("[CSD AVG] brainglobe_heatmap not available, skipping anatomical plot.")