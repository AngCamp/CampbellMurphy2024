#!/usr/bin/env python3
import os
import gzip
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
            self.base_path = Path("/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_swr_data")
        else:
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
                for event_file in session_dir.glob("probe_*_karlsson_detector_events.csv.gz"):
                    try:
                        # Extract probe ID from filename (e.g., "probe_1098236048_channel_1100921954_karlsson_detector_events.csv.gz" or "probe_abc123_channel_xyz_karlsson_detector_events.csv.gz")
                        filename = event_file.name
                        probe_match = re.search(r'probe_([^_]+)_channel_[^_]+_karlsson_detector_events', filename)
                        if not probe_match:
                            print(f"Warning: Could not extract probe ID from filename: {filename}")
                            continue
                            
                        probe_id = probe_match.group(1)
                        if probe_id not in self.data[source][session_id]:
                            self.data[source][session_id][probe_id] = {}
                        
                        with gzip.open(event_file, 'rt') as f:
                            self.data[source][session_id][probe_id]['events'] = pd.read_csv(f)
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
        
        # Apply basic filters
        mask = (events['sw_peak_power'] > min_sw_power) & (events['duration'] > min_duration)
        
        if exclude_gamma:
            mask &= ~events['overlaps_with_gamma']
        if exclude_movement:
            mask &= ~events['overlaps_with_movement']
            
        filtered_events = events[mask].copy()
        
        # Apply speed filter if requested
        if max_speed is not None and speed_data is not None:
            speeds = []
            for _, event in filtered_events.iterrows():
                speed = self.get_event_speed(speed_data, event['start_time'], event['end_time'])
                speeds.append(speed)
            filtered_events['speed'] = speeds
            filtered_events = filtered_events[filtered_events['speed'] <= max_speed]
            
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
        events = self.data[dataset][session_id][probe_id]['events']
        if event_id not in events.index:
            raise ValueError(f"Event ID {event_id} not found in events for probe {probe_id}.")
        return events.loc[event_id]

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
        full_ripple_band_z = zscore(full_ripple_band, axis=0)
        full_ripple_envelope = np.abs(hilbert(full_ripple_band))
        full_ripple_envelope_smooth = gaussian_smooth(full_ripple_envelope, sigma=smoothing_sigma, sampling_frequency=sampling_frequency)
        full_ripple_envelope_z = zscore(full_ripple_envelope_smooth, axis=0)
        full_ripple_power = full_ripple_envelope_smooth ** 2
        full_ripple_power_z = zscore(full_ripple_power, axis=0)

        # Sharp wave band
        full_sharp_wave_file = os.path.join(lfp_dir, f"probe_{probe_id}_channel*_lfp_ca1_putative_str_radiatum.npz")
        full_sharp_wave_files = glob.glob(full_sharp_wave_file)
        full_sharp_wave_data = np.load(full_sharp_wave_files[0])
        full_sharp_wave_signal = full_sharp_wave_data['array'] if 'array' in full_sharp_wave_data else full_sharp_wave_data[full_sharp_wave_data.files[0]]
        sw_filter_data = np.load(filter_path)
        sw_filter = sw_filter_data['sharpwave_componenet_8to40band_1500hz_band']
        full_sharp_wave_filtered = fftconvolve(full_sharp_wave_signal, sw_filter, mode='same')
        full_sharp_wave_band_z = zscore(full_sharp_wave_filtered, axis=0)
        full_sharp_wave_envelope = np.abs(hilbert(full_sharp_wave_filtered))
        full_sharp_wave_envelope_smooth = gaussian_smooth(full_sharp_wave_envelope, sigma=smoothing_sigma, sampling_frequency=sampling_frequency)
        full_sharp_wave_envelope_z = zscore(full_sharp_wave_envelope_smooth, axis=0)
        full_sharp_wave_power = full_sharp_wave_envelope_smooth ** 2
        full_sharp_wave_power_z = zscore(full_sharp_wave_power, axis=0)

        # --- Extract event window from z-scored full-session arrays ---
        start_idx = np.searchsorted(time_stamps, start_time)
        end_idx = np.searchsorted(time_stamps, end_time)
        time_window = time_stamps[start_idx:end_idx]
        time_rel = time_window - (event['start_time'] + event['duration']/2)
        ripple_band_window_z = full_ripple_band_z[start_idx:end_idx].squeeze()
        sharp_wave_band_window_z = full_sharp_wave_band_z[start_idx:end_idx].squeeze()
        ripple_envelope_window_z = full_ripple_envelope_z[start_idx:end_idx].squeeze()
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

    def plot_global_swr_event(self, dataset, session_id, global_event_idx, filter_path=None, window=0.15, save_path=None, save_png=False, show_ap_in_ylabel=False, additional_value='envelope'):
        """
        Plot a global SWR event, showing LFP (grey) and a selected additional value for all probes in the session, sorted by anterior-posterior coordinate.
        Only show global event start/end (dotted green lines) and probe-level event shading (green) on probes in participating_probes.
        Y-axis margin is 5% of the data range, with a minimum range to avoid excessive whitespace. Min/max are computed only on the plotted window.
        If show_ap_in_ylabel is True, display the AP coordinate in the y-label for each subplot.
        The plot title will include the computed direction label (anterior, posterior, or non-directional).
        additional_value: one of:
            'ripple_band': z-scored ripple band (no smoothing)
            'envelope': z-scored, smoothed ripple envelope (abs(Hilbert), Gaussian smooth, z-score)
            'power': z-scored, smoothed ripple envelope squared
        """
        # Load global event CSV
        session_dir = self.base_path / dataset / f"swrs_session_{session_id}"
        global_csv = list(session_dir.glob(f"session_{session_id}_global_swr_events.csv.gz"))
        if not global_csv:
            raise FileNotFoundError(f"No global event CSV found for session {session_id}")
        global_events = pd.read_csv(global_csv[0], index_col=0)
        event = global_events.loc[global_event_idx]
        # Parse probe/channel info from global event
        participating_probes = ast.literal_eval(event['participating_probes'])
        peak_times = ast.literal_eval(event['peak_times'])
        peak_powers = ast.literal_eval(event['peak_powers'])
        global_start = event['start_time']
        global_end = event['end_time']
        # Determine event direction (same logic as filter)
        # Get AP coordinates for all probes in this session
        lfp_dir = self.base_path / self.lfp_sources[dataset] / f"lfp_session_{session_id}"
        probe_ap_dict = {}
        for lfp_file in glob.glob(str(lfp_dir / "probe_*_channel*_lfp_ca1_putative_pyramidal_layer.npz")):
            m_probe = re.search(r'probe_(\d+)_channel', lfp_file)
            m_chan = re.search(r'channel_(\d+)_lfp', lfp_file)
            if not m_probe or not m_chan:
                continue
            probe_id = m_probe.group(1)
            channel_id = int(m_chan.group(1))
            ap = self.get_channel_ap_coordinate(channel_id) if channel_id is not None else None
            probe_ap_dict[probe_id] = ap
        
        # Sort participating probes by AP coordinate
        participating_probes_ap = [(pid, probe_ap_dict.get(pid, float('inf'))) for pid in participating_probes]
        participating_probes_ap.sort(key=lambda x: x[1])
        sorted_probe_ids = [p[0] for p in participating_probes_ap]
        sorted_peak_times = [peak_times[participating_probes.index(pid)] for pid in sorted_probe_ids]
        tolerance = 0.001
        diffs = np.diff(sorted_peak_times)
        increasing = np.all(diffs >= -tolerance)
        decreasing = np.all(diffs <= tolerance)
        if increasing:
            direction = "Anterior → Posterior"
        elif decreasing:
            direction = "Posterior → Anterior"
        else:
            direction = "Non-directional"
        # Print debug info for this event
        print(f"[PLOT] Event {global_event_idx}:")
        print(f"  Probes: {participating_probes}")
        print(f"  APs: {[probe_ap_dict.get(pid, None) for pid in participating_probes]}")
        print(f"  Sorted Probes: {sorted_probe_ids}")
        print(f"  Sorted Peak Times: {sorted_peak_times}")
        print(f"  Diffs: {diffs}")
        print(f"  Increasing: {increasing}, Decreasing: {decreasing}")
        print(f"  Computed Direction: {direction}")
        # Find all probe/channel files for this session
        lfp_dir = self.base_path / self.lfp_sources[dataset] / f"lfp_session_{session_id}"
        probe_patterns = glob.glob(str(lfp_dir / "probe_*_channel*_lfp_ca1_putative_pyramidal_layer.npz"))
        probe_infos = []
        for lfp_file in probe_patterns:
            m_probe = re.search(r'probe_(\d+)_channel', lfp_file)
            m_chan = re.search(r'channel_(\d+)_lfp', lfp_file)
            if not m_probe or not m_chan:
                continue
            probe_id = m_probe.group(1)
            channel_id = int(m_chan.group(1))
            ap = self.get_channel_ap_coordinate(channel_id) if channel_id is not None else None
            probe_infos.append({
                'probe_id': probe_id,
                'channel_id': channel_id,
                'lfp_file': lfp_file,
                'ap': ap
            })
        # Sort by AP (anterior-most first)
        probe_infos = sorted(probe_infos, key=lambda x: (x['ap'] if x['ap'] is not None else float('inf')))
        
        # Load LFP and additional value for each probe
        lfp_traces = []
        additional_traces = []
        time_traces = []
        probe_event_windows = []
        probe_is_participant = []
        windowed_lfp_minmax = []
        windowed_additional_minmax = []
        power_peak_times = []  # Store power peak times for each probe
        middle_peak = np.median(peak_times)
        start_time = middle_peak - window
        end_time = middle_peak + window
        
        for info in probe_infos:
            # Load LFP and time for this probe
            lfp_data = np.load(info['lfp_file'])
            lfp = lfp_data['array'] if 'array' in lfp_data else lfp_data[lfp_data.files[0]]
            # Find time file
            time_pattern = info['lfp_file'].replace('_lfp_ca1_putative_pyramidal_layer.npz', '_lfp_time_index_1500hz.npz')
            time_data = np.load(time_pattern)
            time_stamps = time_data['array'] if 'array' in time_data else time_data[time_data.files[0]]
            # Ripple band
            ripple_band = filter_ripple_band(lfp[:, None]).squeeze()
            # Compute values as in plot_swr_event
            smoothing_sigma = 0.004  # 4ms smoothing
            sampling_frequency = 1500
            if additional_value == 'ripple_band':
                # Z-score ripple band, no smoothing
                additional_traces.append(zscore(ripple_band))
            else:
                ripple_envelope = np.abs(hilbert(ripple_band))
                ripple_envelope_smooth = gaussian_smooth(ripple_envelope, sigma=smoothing_sigma, sampling_frequency=sampling_frequency)
                ripple_envelope_z = zscore(ripple_envelope_smooth)
                if additional_value == 'envelope':
                    additional_traces.append(ripple_envelope_z)
                elif additional_value == 'power':
                    ripple_power = ripple_envelope_smooth ** 2
                    ripple_power_z = zscore(ripple_power)
                    additional_traces.append(ripple_power_z)
                else:
                    raise ValueError(f"Invalid additional_value: {additional_value}. Use 'ripple_band', 'envelope', or 'power'.")
            lfp_traces.append(lfp.squeeze() * 1e6)
            time_traces.append(time_stamps)
            
            # Only mark as participant if in participating_probes
            is_participant = info['probe_id'] in participating_probes
            probe_is_participant.append(is_participant)
            
            # For participating probes, find the first probe-level event that overlaps the global event
            probe_event = None
            ev_start, ev_end = None, None
            power_peak_time = None
            
            if is_participant:
                # Use glob to find the correct karlsson_detector_events file (with channel id)
                event_files = list(session_dir.glob(f"probe_{info['probe_id']}_channel*_karlsson_detector_events.csv.gz"))
                if event_files:
                    event_file = event_files[0]
                    with gzip.open(event_file, 'rt') as f:
                        probe_events_df = pd.read_csv(f)
                    # Find the first event that overlaps the global event window
                    overlap_mask = (probe_events_df['start_time'] <= global_end) & (probe_events_df['end_time'] >= global_start)
                    if overlap_mask.any():
                        probe_event = probe_events_df[overlap_mask].iloc[0]
                        # Clip probe event times to be within global event window
                        ev_start = max(probe_event['start_time'], global_start)
                        ev_end = min(probe_event['end_time'], global_end)
                        # Get power peak time if available
                        if 'power_peak_time' in probe_event:
                            power_peak_time = probe_event['power_peak_time']
                    else:
                        print(f"[DEBUG] No overlapping probe event found for probe {info['probe_id']} in session {session_id} (global event {global_event_idx})")
                else:
                    print(f"[DEBUG] No karlsson_detector_events.csv.gz for probe {info['probe_id']} in session {session_id}")
            
            probe_event_windows.append((ev_start, ev_end))
            power_peak_times.append(power_peak_time)
            
            # Compute min/max for the windowed data for y-axis scaling
            mask = (time_stamps >= start_time) & (time_stamps <= end_time)
            if np.any(mask):
                windowed_lfp_minmax.append((lfp[mask].min() * 1e6, lfp[mask].max() * 1e6))
                windowed_additional_minmax.append((additional_traces[-1][mask].min(), additional_traces[-1][mask].max()))
            else:
                windowed_lfp_minmax.append((0, 0))
                windowed_additional_minmax.append((0, 0))
        
        # Find global min/max for axes (tight, 5% margin, min range) on plotted window only
        lfp_min = min([v[0] for v in windowed_lfp_minmax])
        lfp_max = max([v[1] for v in windowed_lfp_minmax])
        add_min = min([np.min(additional_traces[i][(time_traces[i] >= start_time) & (time_traces[i] <= end_time)]) for i in range(len(additional_traces))])
        add_max = max([np.max(additional_traces[i][(time_traces[i] >= start_time) & (time_traces[i] <= end_time)]) for i in range(len(additional_traces))])
        min_add_range = 2.0
        add_range = max(add_max - add_min, min_add_range)
        add_margin = add_range * 0.05
        add_center = (add_max + add_min) / 2
        add_min = add_center - add_range / 2 - add_margin
        add_max = add_center + add_range / 2 + add_margin
        lfp_range = max(lfp_max - lfp_min, 100.0)  # μV
        add_range = max(add_max - add_min, 2.0)  # Z
        lfp_margin = lfp_range * 0.05
        add_margin = add_range * 0.05
        lfp_center = (lfp_max + lfp_min) / 2
        add_center = (add_max + add_min) / 2
        lfp_min = lfp_center - lfp_range / 2 - lfp_margin
        lfp_max = lfp_center + lfp_range / 2 + lfp_margin
        add_min = add_center - add_range / 2 - add_margin
        add_max = add_center + add_range / 2 + add_margin
        
        n_probes = len(probe_infos)
        fig, axes = plt.subplots(n_probes, 1, figsize=(10, 2.5 * n_probes), sharex=True)
        if n_probes == 1:
            axes = [axes]
        
        # Add direction label to the figure title
        fig.suptitle(f"Global SWR Event - {direction}", fontsize=14, fontweight='bold')
        
        legend_handles = []
        for i, (ax, lfp, add_trace, t, info, (ev_start, ev_end), is_participant, power_peak_time) in enumerate(zip(
            axes, lfp_traces, additional_traces, time_traces, probe_infos, probe_event_windows, probe_is_participant, power_peak_times)):
            
            # Restrict to window for plotting
            mask = (t >= start_time) & (t <= end_time)
            t_rel = t[mask] - middle_peak
            lfp_win = lfp[mask]
            add_win = add_trace[mask]
            
            # LFP: grey, thin
            lfp_line, = ax.plot(t_rel, lfp_win, color='grey', lw=0.8, alpha=0.8, label='LFP (μV)')
            
            # Additional value: black, thicker
            add_ax = ax.twinx()
            if additional_value == 'ripple_band':
                add_label = 'Ripple Band (Z)'
            elif additional_value == 'envelope':
                add_label = 'Ripple Envelope (Z)'
            else:
                add_label = 'Ripple Power (Z)'
            add_line, = add_ax.plot(t_rel, add_win, color='black', lw=1.2, alpha=0.7, label=add_label)
            
            ax.set_ylim(lfp_min, lfp_max)
            add_ax.set_ylim(add_min, add_max)
            ax.margins(x=0)
            add_ax.margins(x=0)
            
            # Only shade probe-level event and show global event lines if participant
            if is_participant:
                # Shade probe-level event (only if exists)
                if ev_start is not None and ev_end is not None:
                    # Shade from probe event's start_time to end_time
                    ax.axvspan(ev_start - middle_peak, ev_end - middle_peak, color='green', alpha=0.25, zorder=1, label='Probe Event')
                
                # Add power peak marker if available
                if power_peak_time is not None:
                    ax.axvline(power_peak_time - middle_peak, color='black', linestyle='--', lw=1, zorder=3, label='Power Peak')
                
                # Dotted green lines for global event (only for participants)
                ax.axvline(global_start - middle_peak, color='green', linestyle=':', lw=3, zorder=2, label='Global Event Start')
                ax.axvline(global_end - middle_peak, color='green', linestyle=':', lw=3, zorder=2, label='Global Event End')
            
            # Both y-axes labeled with units
            if show_ap_in_ylabel:
                ap_str = f" (AP: {info['ap']:.1f})" if info['ap'] is not None else ""
                ax.set_ylabel(f'Probe {info["probe_id"]}{ap_str}\nLFP (μV)', color='grey', fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel(f'Probe {info["probe_id"]}\nLFP (μV)', color='grey', fontsize=10, fontweight='bold')
            add_ax.set_ylabel(add_label, color='black', fontsize=10, fontweight='bold')
            
            for label in ax.get_yticklabels() + add_ax.get_yticklabels():
                label.set_fontweight('bold')
            
            # For legend (only once)
            if i == 0:
                legend_handles = [lfp_line, add_line]
        
        axes[-1].set_xlabel('Time from Middle Peak (s)', fontsize=12, fontweight='bold')
        
        # Add a single legend for the overall figure
        fig.legend(handles=legend_handles + [
            mpl.lines.Line2D([], [], color='green', lw=8, alpha=0.25, label='Probe Event'),
            mpl.lines.Line2D([], [], color='green', linestyle=':', lw=3, label='Global Event Start/End'),
            mpl.lines.Line2D([], [], color='black', linestyle='--', lw=1, label='Power Peak')
        ], loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=11, frameon=True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if save_path:
            ext = save_path.split('.')[-1]
            fig.savefig(save_path, format=ext, dpi=300, bbox_inches='tight')
            if save_png and not save_path.endswith('.png'):
                fig.savefig(save_path.replace('.svg', '.png'), format='png', dpi=300, bbox_inches='tight')
        
        return fig

    def plot_global_event_CSD_slice(self, dataset, session_id, global_event_idx, filter_path=None, 
                                    window=0.15, save_path=None, ca1_only=True, show_scale=True):
        """
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
            time_axis = time[mask]
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
print("[PLOT DEBUG] Saving figure...")
plt.savefig("{save_path}", dpi=300, bbox_inches='tight')
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