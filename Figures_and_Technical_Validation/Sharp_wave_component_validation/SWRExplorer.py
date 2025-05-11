#!/usr/bin/env python3
import os
import gzip
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, butter, filtfilt, hilbert, gaussian, convolve
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


class SWRExplorer:
    """
    Class for exploring and plotting Sharp Wave Ripple (SWR) events.
    
    The class provides methods to:
    1. Load and organize SWR data from multiple datasets
    2. Filter events based on various metrics
    3. Plot individual SWR events with detailed visualizations
    4. List available sessions and probes
    """
    
    def __init__(self, base_path=None):
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
        self._load_data()
        
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

    def plot_swr_event(self, events_df, event_idx, filter_path=None, show_info_title=False, window_padding=None, figsize_mm=None, panels_to_plot=None, time_per_mm=None, **kwargs):
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
            List of panels to plot. Options: ['raw_pyramidal_lfp','raw_s_radiatum_lfp','bandpassed_signals','power'].
            Default: all four panels.
        time_per_mm : float, optional
            Seconds per mm for time axis scaling. If set, overrides figsize_mm width to match time window.
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
        ripple_pattern = os.path.join(lfp_dir, f"probe_{probe_id}_channel*_lfp_ca1_peakripplepower.npz")
        sharp_wave_pattern = os.path.join(lfp_dir, f"probe_{probe_id}_channel*_lfp_ca1_sharpwave.npz")
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

        ripple_band_zscore = zscore(ripple_band_window_uv)
        sharp_wave_filtered_zscore = zscore(sharp_wave_filtered_window_uv)
        ripple_power = np.abs(hilbert(ripple_band_window_uv))**2
        sharp_wave_power = np.abs(hilbert(sharp_wave_filtered_window_uv))**2
        smoothing_sigma = 0.004
        sampling_frequency = 1500
        ripple_power_smooth = gaussian_smooth(ripple_power, sigma=smoothing_sigma, sampling_frequency=sampling_frequency)
        sharp_wave_power_smooth = gaussian_smooth(sharp_wave_power, sigma=smoothing_sigma, sampling_frequency=sampling_frequency)
        ripple_power_z = zscore(ripple_power_smooth)
        sharp_wave_power_z = zscore(sharp_wave_power_smooth)
        power_min = min(ripple_power_z.min(), sharp_wave_power_z.min())
        power_max = max(ripple_power_z.max(), sharp_wave_power_z.max())
        marginfudge = 1.10
        ytop = max(0.25, math.ceil(power_max * marginfudge / 0.25) * 0.25)
        peak_time = event['Peak_time'] if 'Peak_time' in event else event['start_time'] + event['duration']/2
        time_rel = time_window - peak_time
        # Modular panel selection
        all_panels = ['raw_pyramidal_lfp','raw_s_radiatum_lfp','bandpassed_signals','power']
        if panels_to_plot is None:
            panels_to_plot = all_panels
        n_panels = len(panels_to_plot)
        # Modular time-to-mm scaling
        time_window_size = end_time - start_time
        # Use 500mm/second as the standard ratio (like the original plot showing 0.4s in 200mm)
        width_mm = time_window_size * 500  # This gives us the proportional width
        height_mm = 200  # Keep height constant
        figsize = (width_mm / 25.4, height_mm / 25.4)  # Convert to inches for matplotlib

        fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
        if n_panels == 1:
            axes = [axes]
        panel_map = {name: i for i, name in enumerate(panels_to_plot)}

        # Panel plotting with fixed y-axis limits
        if 'raw_pyramidal_lfp' in panels_to_plot:
            ax = axes[panel_map['raw_pyramidal_lfp']]
            ax.plot(time_rel, raw_lfp_window_uv, color='black', linewidth=0.5, label='Raw LFP (Pyramidal)')
            ax.axvspan(event['start_time']-peak_time, event['end_time']-peak_time, alpha=0.3, color='green')
            ax.set_ylabel('Pyramidal Layer LFP (μV)', fontsize=12, fontweight='bold')
            if show_info_title:
                ax.set_title(
                    f'SWR Event {event_idx} - SW Peak Power: {event["sw_peak_power"]:.2f}, SW-Ripple MI: {event.get("sw_ripple_mi", np.nan):.3f}\n'
                    f'Start: {event["start_time"]:.3f}s, End: {event["end_time"]:.3f}s',
                    fontsize=14, fontweight='bold'
                )
            ax.legend(loc='upper right', frameon=True, fancybox=False)
            # Ensure y-axis limits include the full signal range plus a small margin
            ax.set_ylim(lfp_min - y_margin, lfp_max + y_margin)
            ax.tick_params(axis='both', which='major', labelsize=8, width=1.2)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

        if 'raw_s_radiatum_lfp' in panels_to_plot:
            ax = axes[panel_map['raw_s_radiatum_lfp']]
            ax.plot(time_rel, sharp_wave_window_uv, color='blue', linewidth=0.5, label='Raw LFP (Sharp Wave)')
            ax.axvspan(event['start_time']-peak_time, event['end_time']-peak_time, alpha=0.3, color='green')
            ax.set_ylabel('S. Radiatum LFP (μV)', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', frameon=True, fancybox=False)
            # Use same y-axis limits as pyramidal LFP
            ax.set_ylim(lfp_min - y_margin, lfp_max + y_margin)
            ax.tick_params(axis='both', which='major', labelsize=8, width=1.2)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

        if 'bandpassed_signals' in panels_to_plot:
            ax = axes[panel_map['bandpassed_signals']]
            ax.plot(time_rel, ripple_band_zscore, color='black', linewidth=1.5, label='Ripple Band (Z-scored)')
            ax.plot(time_rel, sharp_wave_filtered_zscore, color='blue', linewidth=1.5, label='Sharp Wave Band (Z-scored)')
            ax.axvspan(event['start_time']-peak_time, event['end_time']-peak_time, alpha=0.3, color='green')
            ax.set_ylabel('Bandpass (Z-scored)', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', frameon=True, fancybox=False)
            # Set y-axis limits for z-scored signals
            zscore_max = max(abs(ripple_band_zscore.max()), abs(ripple_band_zscore.min()),
                           abs(sharp_wave_filtered_zscore.max()), abs(sharp_wave_filtered_zscore.min()))
            zscore_margin = zscore_max * 0.05
            ax.set_ylim(-zscore_max - zscore_margin, zscore_max + zscore_margin)
            ax.tick_params(axis='both', which='major', labelsize=8, width=1.2)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

        if 'power' in panels_to_plot:
            ax = axes[panel_map['power']]
            line1, = ax.plot(time_rel, ripple_power_z, color='black', linewidth=1.5, label='Ripple Power (Z-scored)')
            line2, = ax.plot(time_rel, sharp_wave_power_z, color='blue', linewidth=1.5, label='Sharp Wave Power (Z-scored)')
            ax.axvspan(event['start_time']-peak_time, event['end_time']-peak_time, alpha=0.3, color='green')
            ax.set_ylabel('Smoothed Power (Z-scored)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time from Ripple Peak (s)', fontsize=12, fontweight='bold')
            # Set y-axis limits for power plot
            power_margin = (power_max - power_min) * 0.05
            ax.set_ylim(power_min - power_margin, ytop + power_margin)

            # Add peak time markers
            dot_handles = []
            dot_labels = []
            if 'Peak_time' in event and not pd.isna(event['Peak_time']):
                peak_idx = np.argmin(np.abs(time_window - event['Peak_time']))
                dot1 = ax.plot(time_rel[peak_idx], ripple_power_z[peak_idx], 'o', color='black', markersize=7, markeredgecolor='white', label='Ripple Peak Power')[0]
                dot_handles.append(dot1)
                dot_labels.append('Ripple Peak Power')
            if 'sw_peak_time' in event and not pd.isna(event['sw_peak_time']):
                sw_peak_idx = np.argmin(np.abs(time_window - event['sw_peak_time']))
                dot2 = ax.plot(time_rel[sw_peak_idx], sharp_wave_power_z[sw_peak_idx], 'o', color='blue', markersize=7, markeredgecolor='white', label='SW Peak Power')[0]
                dot_handles.append(dot2)
                dot_labels.append('SW Peak Power')

            # Create unified legend with unique entries
            handles, labels = ax.get_legend_handles_labels()
            unique = {}
            for h, l in zip(handles, labels):
                if l not in unique:
                    unique[l] = h
            ax.legend(list(unique.values()), list(unique.keys()), loc='upper right', frameon=True, fancybox=False)
            ax.tick_params(axis='both', which='major', labelsize=8, width=1.2)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

            # Add detailed info title if requested
            if show_info_title:
                info_text = (f"Duration: {event['duration']*1000:.1f} ms, "
                           f"Max Z-score: {event.get('max_zscore', np.nan):.2f}, "
                           f"SW-Ripple PLV: {event.get('sw_ripple_plv', np.nan):.3f}, "
                           f"SW exceeds threshold: {event.get('sw_exceeds_threshold', np.nan)}")
                fig.suptitle(info_text, fontsize=12, fontweight='bold')
        plt.tight_layout()
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