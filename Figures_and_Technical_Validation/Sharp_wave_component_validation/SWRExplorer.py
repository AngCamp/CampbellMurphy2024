#!/usr/bin/env python3
import os
import gzip
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.stats import zscore
from pathlib import Path
from tqdm import tqdm
import re

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
        
        # Set the index to be a combination of dataset, session, and probe
        filtered_events.index.name = f"{dataset}_{session_id}_{probe_id}"
        
        # Ensure 'Unnamed: 0' is preserved as 'event_id'
        if 'Unnamed: 0' in filtered_events.columns:
            filtered_events = filtered_events.rename(columns={'Unnamed: 0': 'event_id'})
            filtered_events = filtered_events.set_index('event_id', append=True)
        
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
    
    def plot_swr_event(self, events_df, event_idx, channel_pyramidal=None, channel_radiatum=None):
        """
        Plot a single SWR event with detailed visualization.
        
        Parameters:
        -----------
        events_df : pd.DataFrame
            DataFrame containing filtered events with metadata
        event_idx : int or str
            Index of the event to plot in the DataFrame (can be either position or event_id)
        channel_pyramidal : str, optional
            Channel ID for pyramidal layer
        channel_radiatum : str, optional
            Channel ID for radiatum layer
        """
        # Get the event data
        if isinstance(event_idx, int):
            event_data = events_df.iloc[event_idx]
        else:
            event_data = events_df.loc[event_idx]
        
        # Extract dataset and session info from the index
        if isinstance(events_df.index, pd.MultiIndex):
            dataset, session_id, probe_id = events_df.index.names[0].split('_')
        else:
            dataset, session_id, probe_id = events_df.index.name.split('_')
        
        # Set up paths
        swr_dir = self.base_path / dataset / f"swrs_session_{session_id}"
        lfp_dir = self.base_path / self.lfp_sources[dataset] / f"lfp_session_{session_id}"
        filter_path = "/home/acampbell/NeuropixelsLFPOnRamp/SWR_Neuropixels_Detector/Filters/sharpwave_componenet_8to40band_1500hz_band.npz"
        
        # Load LFP data
        ripple_file = lfp_dir / f"probe_{probe_id}_channel_{channel_pyramidal}_lfp_ca1_peakripplepower.npz"
        sharp_wave_file = lfp_dir / f"probe_{probe_id}_channel_{channel_radiatum}_lfp_ca1_sharpwave.npz"
        time_file = lfp_dir / f"probe_{probe_id}_channel_{channel_pyramidal}_lfp_time_index_1500hz.npz"
        
        ripple_data = np.load(ripple_file)
        sharp_wave_data = np.load(sharp_wave_file)
        time_data = np.load(time_file)
        
        # Extract data arrays
        ripple_signal = ripple_data['array'] if 'array' in ripple_data else ripple_data[ripple_data.files[0]]
        sharp_wave_signal = sharp_wave_data['array'] if 'array' in sharp_wave_data else sharp_wave_data[sharp_wave_data.files[0]]
        time_stamps = time_data['array'] if 'array' in time_data else time_data[time_data.files[0]]
        
        # Load and apply sharp wave filter
        sw_filter_data = np.load(filter_path)
        sw_filter = sw_filter_data['sharpwave_componenet_8to40band_1500hz_band']
        sharp_wave_filtered = fftconvolve(sharp_wave_signal, sw_filter, mode='same')
        
        # Calculate time window
        window_padding = max(0.05, (event_data['end_time'] - event_data['start_time']) * 3)
        start_time = max(time_stamps[0], event_data['start_time'] - window_padding)
        end_time = min(time_stamps[-1], event_data['end_time'] + window_padding)
        
        # Find sample indices
        start_idx = np.searchsorted(time_stamps, start_time)
        end_idx = np.searchsorted(time_stamps, end_time)
        
        # Extract window data
        time_window = time_stamps[start_idx:end_idx]
        ripple_window = ripple_signal[start_idx:end_idx]
        sharp_wave_window = sharp_wave_signal[start_idx:end_idx]
        sharp_wave_filtered_window = sharp_wave_filtered[start_idx:end_idx]
        
        # Create plot
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Plot pyramidal layer LFP
        axes[0].plot(time_window, ripple_window, color='grey', linewidth=0.8)
        axes[0].axvspan(event_data['start_time'], event_data['end_time'], alpha=0.2, color='red')
        axes[0].set_ylabel('Putative Pyramidal Layer LFP', fontsize=12)
        
        # Add title and event info
        title = (f"SWR Event {event_idx} - Session: {session_id}, Probe: {probe_id}\n"
                f"SW Peak Power: {event_data['sw_peak_power']:.2f}, "
                f"SW-Ripple CLCORR: {event_data['sw_ripple_clcorr']:.3f}, "
                f"Speed: {event_data['max_speed_in_window']:.2f} cm/s")
        axes[0].set_title(title, fontsize=14)
        
        if 'Peak_time' in event_data and not pd.isna(event_data['Peak_time']):
            axes[0].axvline(event_data['Peak_time'], color='orange', linestyle='--', label='Ripple Peak')
            axes[0].legend(loc='upper right')
        
        # Plot radiatum layer LFP
        axes[1].plot(time_window, sharp_wave_window, color='grey', linewidth=0.8)
        axes[1].axvspan(event_data['start_time'], event_data['end_time'], alpha=0.2, color='red')
        axes[1].set_ylabel('Putative S. Radiatum Layer LFP', fontsize=12)
        
        # Plot filtered signals
        axes[2].plot(time_window, ripple_window, color='black', linewidth=1.2, 
                    label='Pyramidal Layer')
        axes[2].plot(time_window, sharp_wave_filtered_window, color='blue', linewidth=1.2, 
                    label='Bandpassed S. Radiatum')
        axes[2].axvspan(event_data['start_time'], event_data['end_time'], alpha=0.2, color='red')
        axes[2].set_ylabel('Filtered Signals', fontsize=12)
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].legend(loc='upper right')
        
        # Add horizontal zero lines
        for ax in axes:
            ax.axhline(y=0, color='lightgrey', linestyle='-', alpha=0.5)
            ax.set_xlim(time_window[0], time_window[-1])
        
        # Add info text
        info_text = (f"Duration: {event_data['duration']*1000:.1f} ms, "
                    f"Max Z-score: {event_data['max_zscore']:.2f}, "
                    f"SW-Ripple PLV: {event_data['sw_ripple_plv']:.3f}, "
                    f"SW exceeds threshold: {event_data['sw_exceeds_threshold']}")
        fig.suptitle(info_text, fontsize=12, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
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
        
        # Preserve the multi-index structure
        if isinstance(events_df.index, pd.MultiIndex):
            filtered_events.index = events_df.index[filtered_events.index]
        
        return filtered_events

