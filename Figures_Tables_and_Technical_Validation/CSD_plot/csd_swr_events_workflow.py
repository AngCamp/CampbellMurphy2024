# =============================================================================
# IMPORTS
# =============================================================================
import os
import json
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert, savgol_filter
from scipy.stats import zscore
import yaml
import logging
from tqdm import tqdm
import ast
from typing import Dict, List, Tuple, Optional
import tempfile
import subprocess
import traceback
from scipy.ndimage import gaussian_filter1d

# AllenSDK imports
try:
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorNeuropixelsProjectCache,
    )
    ALLENSDK_AVAILABLE = True
except ImportError:
    ALLENSDK_AVAILABLE = False
    print("Warning: AllenSDK not available. Some functionality may be limited.")


#!/usr/bin/env python3
"""
CSD SWR Events Workflow for ABI Visual Behaviour Dataset

This script implements the three-step process for creating CSD plots of SWR events:
1. Identifying good recordings (probes in middle of CA1 with buffer)
2. Identifying good events (immobile, duration 75-150ms)
3. Plotting CSD for each probe centered around peak global ripple time

Based on instructions in CURSOR_INSTRUCTIONS.md
"""

# =============================================================================
# USER-CONFIGURABLE CSD PARAMETERS
# =============================================================================
MIN_CHANNELS_BELOW_PYRAMIDAL = 10
MIN_CHANNELS_ABOVE_PYRAMIDAL = 5
MIN_CONSECUTIVE_40MICRON_CHANNELS = 10

CSD_COMPUTE_DEPTH_RANGE = 500  # microns from pyramidal layer for CSD computation
# Set as (min, max) tuple for plotting range
CSD_PLOT_DEPTH_RANGE = (-200, 100)  # microns from pyramidal layer for plotting (min, max)

# Number of top putative events per probe
MAX_EVENTS_PER_PROBE = 5

SMOOTHING_METHOD = "sav-gol" # "exponential" or "gaussian" or "sav-gol" or "none"

# Smoothing parameters
if SMOOTHING_METHOD == "exponential":
    CSD_SMOOTHING_ENABLED = True  # Set to True to enable smoothing
else:
    CSD_SMOOTHING_ENABLED = False
CSD_SMOOTH_ALPHA_SPATIAL = 0.7  # Exponential smoothing alpha for spatial (channels) direction
CSD_SMOOTH_ALPHA_TEMPORAL = 0  # Exponential smoothing alpha for temporal (time) direction

# Gaussian smoothing parameters
if SMOOTHING_METHOD == "gaussian":
    CSD_GAUSSIAN_SMOOTHING_ENABLED = True
else:
    CSD_GAUSSIAN_SMOOTHING_ENABLED = False
CSD_GAUSSIAN_SIGMA = 1.0  # in channels
CSD_GAUSSIAN_TRUNCATE = 3.0

# Savitzky-Golay smoothing parameters
if SMOOTHING_METHOD == "sav-gol":
    CSD_SAVGOL_SMOOTHING_ENABLED = True
else:
    CSD_SAVGOL_SMOOTHING_ENABLED = False  # Set to True to enable Savitzky-Golay smoothing
CSD_SAVGOL_WINDOW_LENGTH = 3  # Must be odd and <= number of channels
CSD_SAVGOL_POLYORDER = 2 

# =============================================================================
# INPUT/OUTPUT PATHS - MODIFY THESE AS NEEDED
# =============================================================================
BASE_DATA_PATH = "/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_v2_final"  # Base path to data directory
OUTPUT_DIR = "/home/acampbell/NeuropixelsLFPOnRamp/Figures_Tables_and_Technical_Validation/CSD_plot/csd_results"  # Output directory

# Session and event search parameters
MAX_SESSIONS_TO_PROCESS = 1  # Number of top sessions to process
MAX_EVENTS_PER_SESSION = 1  # Number of top events per session to plot
MIN_PROBE_COUNT = 3  # Minimum number of probes for a session to be considered
MAX_PROBE_COUNT = 6  # Maximum number of probes for a session to be considered
MIN_PEAK_POWER = 5.0  # Minimum global peak power for events
MAX_PEAK_POWER = 10.0  # Maximum global peak power for events
MIN_PARTICIPATING_PROBES = 2  # Minimum number of probes participating in global event

# Depth range for plotting (microns from pyramidal layer)
CSD_CLIP_RANGE = (-2, 2)  # (min, max) for CSD color scale

# =============================================================================
# PUTATIVE EVENT FILTERING THRESHOLDS
# =============================================================================
PUTATIVE_MIN_POWER_MAX_ZSCORE = 3.0
PUTATIVE_MAX_POWER_MAX_ZSCORE = 10.0
PUTATIVE_MIN_SW_PEAK_POWER = 2.0
PUTATIVE_SPEED_THRESHOLD = 2.0  # cm/s
PUTATIVE_SPEED_EXCLUSION_WINDOW = 2.0  # seconds


# =============================================================================
# CSD FUNCTIONS (from instructions)
# =============================================================================
def average_channels(electrode_data):
    """Calculate the average of every pair of even and odd channels"""
    nchan = electrode_data.shape[1]
    mask = np.full(nchan, False, dtype=bool)
    # Set True at even indices
    mask[::2] = True
    even_channels = electrode_data[:, mask]
    odd_channels = electrode_data[:, ~mask]
    
    # Handle case where even and odd channels have different lengths
    min_length = min(even_channels.shape[1], odd_channels.shape[1])
    even_channels = even_channels[:, :min_length]
    odd_channels = odd_channels[:, :min_length]
    
    # Calculate the average
    averaged_data = (even_channels + odd_channels) / 2
    return averaged_data

def butterworth_filter_for_csd(
    LFP_array, lowcut=1.0, highcut=250.0, samplingfreq=2500.0, order=6
):
    nyquist = 0.5 * samplingfreq
    # Design the Butterworth bandpass filter
    b, a = signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    # Apply the filter to all channels simultaneously using vectorized operations
    filtered_signal = signal.lfilter(b, a, LFP_array, axis=0)
    return filtered_signal

def finitimpresp_filter_for_csd(
    LFP_array, lowcut=1, highcut=250, samplingfreq=2500, filter_order=101
):
    nyquist = 0.5 * samplingfreq
    # Design the FIR bandpass filter using scipy.signal.firwin
    fir_coeff = signal.firwin(
        filter_order,
        [lowcut / nyquist, highcut / nyquist],
        pass_zero=False,
        fs=samplingfreq,
    )
    # Apply the FIR filter to your signal_array
    filtered_signal = signal.lfilter(fir_coeff, 1.0, LFP_array, axis=0)
    return filtered_signal

def compute_csd(lfp_data, spacing_between_channels):
    """
    Compute Current Source Density (CSD) from Local Field Potential (LFP) data.

    Parameters:
        lfp_data (numpy.ndarray): 2D array of LFP data, shape (n_channels, n_timepoints).
        spacing_between_channels (float): Distance (in micrometers) between adjacent channels.

    Returns:
        csd_data (numpy.ndarray): 2D array of CSD data, shape (n_channels, n_timepoints).
    """
    # Optionally filter LFP if needed (uncomment if desired)
    # lfp_data = finitimpresp_filter_for_csd(lfp_data)

    # Compute second spatial derivative (across channels, for each timepoint)
    csd_data = np.zeros_like(lfp_data)
    csd_data[1:-1, :] = (lfp_data[2:, :] - 2 * lfp_data[1:-1, :] + lfp_data[:-2, :]) / (spacing_between_channels ** 2)
    # Optionally, set edges to zero or nan
    csd_data[0, :] = 0
    csd_data[-1, :] = 0
    return csd_data

def exponential_smoothing_2d(data, alpha, axis):
    """
    Apply exponential smoothing to a 2D NumPy array along the specified axis.

    Parameters:
    - data: Input 2D NumPy array
    - alpha: Smoothing parameter (0 < alpha < 1)
    - axis: Axis along which to apply exponential smoothing (0 for rows, 1 for columns)

    Returns:
    - smoothed_data: Exponentially smoothed array
    """
    if axis not in (0, 1):
        raise ValueError("Axis must be 0 (rows) or 1 (columns).")

    smoothed_data = np.copy(data)

    if axis == 0:
        for i in range(data.shape[1]):
            smoothed_data[:, i] = exponential_smoothing(data[:, i], alpha)
    elif axis == 1:
        for i in range(data.shape[0]):
            smoothed_data[i, :] = exponential_smoothing(data[i, :], alpha)

    return smoothed_data

def exponential_smoothing(series, alpha):
    """
    Apply exponential smoothing to a 1D NumPy array.

    Parameters:
    - series: Input 1D NumPy array
    - alpha: Smoothing parameter (0 < alpha < 1)

    Returns:
    - smoothed_series: Exponentially smoothed array
    """
    smoothed_series = np.zeros_like(series)
    smoothed_series[0] = series[0]

    for t in range(1, len(series)):
        smoothed_series[t] = alpha * series[t] + (1 - alpha) * smoothed_series[t - 1]

    return smoothed_series

def gaussian_smoothing_2d(data, sigma, axis, truncate=3.0):
    """Apply Gaussian smoothing to a 2D array along the specified axis."""
    return gaussian_filter1d(data, sigma=sigma, axis=axis, truncate=truncate, mode='nearest')

# =============================================================================
# MAIN WORKFLOW CLASS
# =============================================================================
class CSDSWREventsWorkflow:
    """
    Workflow for creating CSD plots of SWR events in ABI Visual Behaviour dataset.
    """
    
    def __init__(self, base_path: str, output_dir: str):
        """
        Initialize the workflow.
        
        Parameters:
        -----------
        base_path : str
            Base path to the data directory
        output_dir : str
            Output directory for results
        """
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Dataset paths
        self.dataset = "allen_visbehave_swr_murphylab2024"
        self.lfp_source = "allen_visbehave_swr_murphylab2024_lfp_data"
        
        # Parameters
        self.min_duration = 0.06  # 60ms minimum
        self.max_duration = 0.150  # 150ms maximum
        self.speed_threshold = 2.0  # cm/s
        self.speed_exclusion_window = 2.0  # seconds
        self.plot_padding = 0.25  # seconds on either side
        self.channel_spacing = 40.0  # microns between channels
        
        # AllenSDK cache
        self.cache = None
        if ALLENSDK_AVAILABLE:
            self._setup_allensdk_cache()
        
        # Create subdirectories
        self.data_dir = self.output_dir / "csd_data"
        self.plots_dir = self.output_dir / "csd_plots"
        self.data_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
    def _setup_allensdk_cache(self):
        """Set up AllenSDK cache for accessing raw data."""
        try:
            # Use the cache directory that's been used for everything else
            cache_dir = "/space/scratch/allen_visbehave_data"
            
            self.cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)
            self.logger.info(f"AllenSDK cache initialized from {cache_dir}")
        except Exception as e:
            self.logger.warning(f"Could not initialize AllenSDK cache: {e}")
            self.cache = None
    
    def load_channel_selection_metadata(self, session_id: str, probe_id: str) -> dict:
        """
        Load channel selection metadata for a specific probe.
        
        Parameters:
        -----------
        session_id : str
            Session ID
        probe_id : str
            Probe ID
            
        Returns:
        --------
        dict
            Channel selection metadata
        """
        session_dir = self.base_path / self.dataset / f"swrs_session_{session_id}"
        metadata_file = session_dir / f"probe_{probe_id}_channel_selection_metadata.json.gz"
        
        if not metadata_file.exists():
            self.logger.warning(f"Metadata file not found: {metadata_file}")
            return None
            
        with gzip.open(metadata_file, 'rt') as f:
            return json.load(f)
    
    def analyze_probe_spacing(self, metadata: dict) -> dict:
        """
        Analyze the spacing characteristics of a probe's channels.
        
        Parameters:
        -----------
        metadata : dict
            Channel selection metadata
            
        Returns:
        --------
        dict
            Analysis results including spacing info and failure reasons
        """
        if not metadata:
            return {'is_good': False, 'failure_reason': 'No metadata'}
            
        ripple_band = metadata.get('ripple_band', {})
        sharp_wave_band = metadata.get('sharp_wave_band', {})
        
        if not ripple_band or not sharp_wave_band:
            return {'is_good': False, 'failure_reason': 'Missing ripple or sharp wave band data'}
            
        # Get channel depths and IDs
        depths = ripple_band.get('depths', [])
        channel_ids = ripple_band.get('channel_ids', [])
        selected_ripple_channel_id = ripple_band.get('selected_channel_id')
        selected_sw_channel_id = sharp_wave_band.get('selected_channel_id')
        
        if not depths or not channel_ids or selected_ripple_channel_id is None or selected_sw_channel_id is None:
            return {'is_good': False, 'failure_reason': 'Missing channel data or selected channels'}
            
        # Find selected channel indices
        try:
            ripple_idx = channel_ids.index(selected_ripple_channel_id)
            ripple_depth = depths[ripple_idx]
        except ValueError:
            return {'is_good': False, 'failure_reason': 'Selected ripple channel not found in channel list'}
            
        # Sort depths and find spacing
        sorted_depths = sorted(depths)
        depth_diffs = np.diff(sorted_depths)
        
        # Find longest stretch of 40-micron spacing
        expected_spacing = 40.0
        spacing_tolerance = 2.0  # More lenient tolerance
        
        # Find consecutive channels with 40-micron spacing
        good_spacing_mask = np.abs(depth_diffs - expected_spacing) <= spacing_tolerance
        
        # Find longest consecutive stretch
        max_consecutive = 0
        current_consecutive = 0
        longest_stretch_start = 0
        longest_stretch_end = 0
        
        for i, is_good in enumerate(good_spacing_mask):
            if is_good:
                current_consecutive += 1
                if current_consecutive > max_consecutive:
                    max_consecutive = current_consecutive
                    longest_stretch_start = i - current_consecutive + 1
                    longest_stretch_end = i
            else:
                current_consecutive = 0
        
        # Calculate how far the longest stretch extends around pyramidal layer
        # Find the pyramidal layer depth in the sorted depths
        sorted_ripple_idx = sorted_depths.index(ripple_depth)
        
        # Calculate distance from pyramidal layer to longest stretch
        if max_consecutive > 0:
            stretch_center = (longest_stretch_start + longest_stretch_end) / 2
            distance_from_pyramidal = abs(sorted_ripple_idx - stretch_center)
        else:
            distance_from_pyramidal = float('inf')
        
        # Check if pyramidal layer is in middle third
        min_depth = min(depths)
        max_depth = max(depths)
        depth_range = max_depth - min_depth
        middle_start = min_depth + depth_range * 0.33
        middle_end = min_depth + depth_range * 0.67
        in_middle_third = middle_start <= ripple_depth <= middle_end
        
        # Count channels around pyramidal layer
        channels_below = ripple_idx
        channels_above = len(channel_ids) - ripple_idx - 1
        
        # Determine if this probe would be good for CSD
        is_good = True
        failure_reasons = []
        
        if not in_middle_third:
            is_good = False
            failure_reasons.append("Pyramidal layer not in middle third")
            
        if channels_below < 10:
            is_good = False
            failure_reasons.append(f"Insufficient channels below pyramidal layer ({channels_below} < 10)")
            
        if channels_above < 5:
            is_good = False
            failure_reasons.append(f"Insufficient channels above pyramidal layer ({channels_above} < 5)")
            
        if max_consecutive < 10:  # Require at least 10 consecutive channels with good spacing
            is_good = False
            failure_reasons.append(f"Insufficient consecutive channels with 40-micron spacing ({max_consecutive} < 10)")
        
        # Create analysis result
        analysis = {
            'is_good': is_good,
            'failure_reasons': failure_reasons if not is_good else [],
            'ripple_channel_depth': ripple_depth,
            'ripple_channel_idx': ripple_idx,
            'total_channels': len(channel_ids),
            'channels_above_pyramidal': channels_above,
            'channels_below_pyramidal': channels_below,
            'in_middle_third': in_middle_third,
            'max_consecutive_40micron_channels': max_consecutive,
            'longest_stretch_start_idx': longest_stretch_start,
            'longest_stretch_end_idx': longest_stretch_end,
            'distance_from_pyramidal_to_stretch': distance_from_pyramidal,
            'mean_spacing': np.mean(depth_diffs) if len(depth_diffs) > 0 else 0,
            'spacing_std': np.std(depth_diffs) if len(depth_diffs) > 0 else 0,
            'min_spacing': np.min(depth_diffs) if len(depth_diffs) > 0 else 0,
            'max_spacing': np.max(depth_diffs) if len(depth_diffs) > 0 else 0
        }
        
        return analysis
    
    def check_good_recording(self, metadata: dict) -> Tuple[bool, dict]:
        """
        Check if a recording is good for CSD plotting.
        """
        analysis = self.analyze_probe_spacing(metadata)
        if not analysis['is_good']:
            return False, {}
        ripple_band = metadata.get('ripple_band', {})
        sharp_wave_band = metadata.get('sharp_wave_band', {})
        ripple_channel_id = ripple_band.get('selected_channel_id')
        sharp_wave_channel_id = sharp_wave_band.get('selected_channel_id')
        depths = ripple_band.get('depths', [])
        channel_ids = ripple_band.get('channel_ids', [])
        # Find indices and depths
        try:
            ripple_idx = channel_ids.index(ripple_channel_id)
            ripple_depth = depths[ripple_idx]
            sw_idx = channel_ids.index(sharp_wave_channel_id)
            sw_depth = depths[sw_idx]
        except Exception:
            return False, {}
        # Require stratum radiatum channel within PLOT_DEPTH_RANGE of pyramidal
        if abs(sw_depth - ripple_depth) > CSD_PLOT_DEPTH_RANGE[1]:
            return False, {'failure_reason': f'Stratum radiatum channel not within {CSD_PLOT_DEPTH_RANGE[1]}um of pyramidal'}
        recording_info = {
            'ripple_channel_id': ripple_channel_id,
            'ripple_channel_depth': ripple_depth,
            'ripple_channel_idx': ripple_idx,
            'sharp_wave_channel_id': sharp_wave_channel_id,
            'sharp_wave_channel_depth': sw_depth,
            'channel_depths': depths,
            'channel_ids': channel_ids,
            'channels_above_pyramidal': analysis['channels_above_pyramidal'],
            'channels_below_pyramidal': analysis['channels_below_pyramidal'],
            'max_consecutive_40micron_channels': analysis['max_consecutive_40micron_channels'],
            'spacing_analysis': analysis
        }
        return True, recording_info
    
    def load_global_events(self, session_id: str) -> pd.DataFrame:
        """
        Load global SWR events for a session.
        
        Parameters:
        -----------
        session_id : str
            Session ID
            
        Returns:
        --------
        pd.DataFrame
            Global events DataFrame
        """
        session_dir = self.base_path / self.dataset / f"swrs_session_{session_id}"
        global_csv = list(session_dir.glob(f"session_{session_id}_global_swr_events.csv.gz"))
        
        if not global_csv:
            return pd.DataFrame()
            
        return pd.read_csv(global_csv[0], index_col=0)
    
    def filter_good_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter events to find good ones for CSD plotting.
        
        Criteria:
        - Duration between 75ms and 150ms
        - Peak power between MIN_PEAK_POWER and MAX_PEAK_POWER
        - At least MIN_PARTICIPATING_PROBES participating probes
        
        Parameters:
        -----------
        events_df : pd.DataFrame
            Global events DataFrame
            
        Returns:
        --------
        pd.DataFrame
            Filtered events DataFrame
        """
        if events_df.empty:
            return events_df
            
        # Duration filter
        duration_mask = (events_df['duration'] >= self.min_duration) & \
                       (events_df['duration'] <= self.max_duration)
        
        # Peak power filter
        if 'global_peak_power' in events_df.columns:
            power_mask = (events_df['global_peak_power'] >= MIN_PEAK_POWER) & \
                        (events_df['global_peak_power'] <= MAX_PEAK_POWER)
        else:
            power_mask = pd.Series([True] * len(events_df), index=events_df.index)
        
        # Probe count filter
        if 'probe_count' in events_df.columns:
            probe_mask = events_df['probe_count'] >= MIN_PARTICIPATING_PROBES
        else:
            probe_mask = pd.Series([True] * len(events_df), index=events_df.index)
            
        # Apply filters
        filtered_events = events_df[duration_mask & power_mask & probe_mask].copy()
        
        # Sort by global peak power (descending)
        if 'global_peak_power' in filtered_events.columns:
            filtered_events = filtered_events.sort_values('global_peak_power', ascending=False)
            
        return filtered_events
    
    def load_lfp_data_from_allensdk(self, session_id: str, probe_id: str, 
                                   peak_time: float, window: float,
                                   ca1_channel_ids: list) -> tuple:
        """
        Load LFP data for selected CA1 channels from AllenSDK, in the order given by ca1_channel_ids.
        Only returns data for those channels, in that order.
        Returns (lfp_data, time_axis, used_channel_ids)
        """
        if not self.cache:
            raise RuntimeError("AllenSDK cache not available")
        session = self.cache.get_ecephys_session(ecephys_session_id=int(session_id))
        lfp = session.get_lfp(int(probe_id))
        # Only use channel IDs present in the LFP object
        lfp_channel_ids = set(lfp.channel.values)
        valid_ca1_channel_ids = [cid for cid in ca1_channel_ids if cid in lfp_channel_ids]
        missing = set(ca1_channel_ids) - lfp_channel_ids
        if missing:
            self.logger.warning(f"Probe {probe_id}: {len(missing)} channel IDs missing from LFP. Skipping these channels.")
        if not valid_ca1_channel_ids:
            raise ValueError(f"No valid CA1 channel IDs found in LFP for probe {probe_id}.")
        lfp_sel = lfp.sel(channel=valid_ca1_channel_ids)
        lfp_data = lfp_sel.to_numpy().T  # shape: (n_channels, n_time)
        time_axis = lfp_sel.time.values
        # Restrict to window around peak_time (symmetric, always centered)
        t0 = peak_time - window
        t1 = peak_time + window
        mask = (time_axis >= t0) & (time_axis <= t1)
        lfp_data = lfp_data[:, mask]
        time_axis = time_axis[mask]
        # Ensure window is always 2*window seconds, pad if needed
        expected_len = int(round((2 * window) / np.median(np.diff(time_axis)))) + 1 if len(time_axis) > 1 else 0
        if len(time_axis) > 1 and len(time_axis) < expected_len:
            # Pad with NaNs if not enough data (should be rare)
            pad_len = expected_len - len(time_axis)
            lfp_data = np.pad(lfp_data, ((0,0),(0,pad_len)), mode='constant', constant_values=np.nan)
            time_step = np.median(np.diff(time_axis))
            new_times = np.arange(time_axis[-1] + time_step, time_axis[-1] + (pad_len+1)*time_step, time_step)
            time_axis = np.concatenate([time_axis, new_times])
        # Ensure LFP and time_axis are always the same length
        if lfp_data.shape[1] != len(time_axis):
            min_len = min(lfp_data.shape[1], len(time_axis))
            lfp_data = lfp_data[:, :min_len]
            time_axis = time_axis[:min_len]
        return lfp_data, time_axis, valid_ca1_channel_ids
    
    def compute_csd_for_event(self, lfp_data: np.ndarray) -> np.ndarray:
        """
        Compute CSD for LFP data.
        
        Parameters:
        -----------
        lfp_data : np.ndarray
            LFP data, shape (n_channels, n_samples)
            
        Returns:
        --------
        np.ndarray
            CSD data, shape (n_channels, n_samples)
        """
        # Use all channels for CSD computation
        chan_rows = list(range(lfp_data.shape[1]))
        
        # Compute CSD
        csd_data = compute_csd(lfp_data, self.channel_spacing)
        
        return csd_data
    
    def plot_csd_event(self, csd_data: np.ndarray, time_axis: np.ndarray, depth_range: tuple,
                      channel_depths: list, event_info: dict,
                      metadata: dict, save_path_base: str,
                      clip_range: tuple) -> None:
        """
        Plot CSD event with optional exponential smoothing in spatial and temporal directions and adjustable clipping.

        Parameters:
        - csd_data: np.ndarray, CSD data (channels x time)
        - time_axis: np.ndarray, time points
        - depth_range: tuple, (min, max) range for plotting (microns)
        - channel_depths: list, depths for each channel
        - event_info: dict, event metadata
        - metadata: dict, probe/channel metadata
        - save_path_base: str, base path for saving plots
        - clip_range: tuple, (min, max) for CSD color scale
        """
        import numpy as np
        import matplotlib.pyplot as plt
        # Ensure dimensions match
        if csd_data.shape[0] != len(channel_depths):
            self.logger.warning(f"CSD data has {csd_data.shape[0]} channels but {len(channel_depths)} depths provided. Truncating to match.")
            min_channels = min(csd_data.shape[0], len(channel_depths))
            csd_data = csd_data[:min_channels, :]
            channel_depths = channel_depths[:min_channels]
        if csd_data.shape[1] != len(time_axis):
            self.logger.warning(f"CSD data has {csd_data.shape[1]} time points but {len(time_axis)} time values provided. Truncating to match.")
            min_time = min(csd_data.shape[1], len(time_axis))
            csd_data = csd_data[:, :min_time]
            time_axis = time_axis[:min_time]
        # Correct relative depth: positive = superficial, negative = deep
        ripple_depth = metadata['ripple_band']['depths'][metadata['ripple_band']['channel_ids'].index(metadata['ripple_band']['selected_channel_id'])]
        rel_channel_depths = ripple_depth - np.array(channel_depths)
        # Only plot channels within (min, max) microns of pyramidal layer
        min_depth, max_depth = depth_range
        mask = (rel_channel_depths >= min_depth) & (rel_channel_depths <= max_depth)
        rel_channel_depths_plot = rel_channel_depths[mask]
        channel_ids = np.array(metadata['ripple_band']['channel_ids'])[:csd_data.shape[0]]
        channel_ids_plot = channel_ids[mask]
        channel_depths_plot = np.array(channel_depths)[mask]
        csd_data_masked = csd_data[mask, :]
        # Sort by rel_channel_depths_plot (most positive/superficial at top)
        sort_idx = np.argsort(-rel_channel_depths_plot)
        rel_channel_depths_plot = rel_channel_depths_plot[sort_idx]
        csd_data_plot = csd_data_masked[sort_idx, :]
        channel_ids_plot = channel_ids_plot[sort_idx]
        channel_depths_plot = channel_depths_plot[sort_idx]
        # --- Smoothing ---
        if CSD_SAVGOL_SMOOTHING_ENABLED:
            # Ensure window length is odd and does not exceed number of channels
            win_len = min(CSD_SAVGOL_WINDOW_LENGTH, csd_data_plot.shape[0] if csd_data_plot.shape[0] % 2 == 1 else csd_data_plot.shape[0] - 1)
            if win_len < 3:
                win_len = 3
            if win_len % 2 == 0:
                win_len += 1
            if win_len > csd_data_plot.shape[0]:
                win_len = csd_data_plot.shape[0] if csd_data_plot.shape[0] % 2 == 1 else csd_data_plot.shape[0] - 1
            csd_data_plot = savgol_filter(csd_data_plot, window_length=win_len, polyorder=CSD_SAVGOL_POLYORDER, axis=0, mode='interp')
        elif CSD_GAUSSIAN_SMOOTHING_ENABLED:
            csd_data_plot = gaussian_smoothing_2d(
                csd_data_plot, sigma=CSD_GAUSSIAN_SIGMA, axis=0, truncate=CSD_GAUSSIAN_TRUNCATE
            )
        elif CSD_SMOOTHING_ENABLED:
            if CSD_SMOOTH_ALPHA_SPATIAL > 0:
                csd_data_plot = exponential_smoothing_2d(csd_data_plot, CSD_SMOOTH_ALPHA_SPATIAL, axis=0)
            if CSD_SMOOTH_ALPHA_TEMPORAL > 0:
                csd_data_plot = exponential_smoothing_2d(csd_data_plot, CSD_SMOOTH_ALPHA_TEMPORAL, axis=1)
        # --- Clipping ---
        csd_plot = np.clip(csd_data_plot, *clip_range)
        # For the y-axis, use rel_channel_depths_plot (depth relative to pyramidal layer)
        # Center time axis on peak
        peak_time = event_info.get('peak_time', 0)
        time_rel = time_axis - peak_time
        fig, ax = plt.subplots(figsize=(12, 8))
        time_mesh, depth_mesh = np.meshgrid(time_rel, rel_channel_depths_plot)
        im = ax.pcolormesh(time_mesh, depth_mesh, csd_plot, cmap='RdBu_r', shading='gouraud')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Current Source Density (μA/mm³)', rotation=270, labelpad=20)
        ax.set_xlabel('Time relative to peak (s)')
        ax.set_ylabel('Depth relative to pyramidal layer (μm)')
        # Optionally, set y-ticks to show both rel depth and actual depth
        yticks = rel_channel_depths_plot
        yticklabels = [f"{int(rd)} ({int(cd)})" for rd, cd in zip(rel_channel_depths_plot, channel_depths_plot)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        title = f"CSD - Session {event_info.get('session_id', 'Unknown')}, "
        title += f"Probe {event_info.get('probe_id', 'Unknown')}, "
        title += f"Event {event_info.get('event_id', 'Unknown')}\n"
        title += f"Duration: {event_info.get('duration', 0):.3f}s, "
        title += f"Power Z: {event_info.get('power_max_zscore', 0):.2f}"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        # Mark the peak time (black line at t=0)
        ax.axvline(0, color='black', linestyle='--', alpha=0.7, label='Peak Time')
        # Markers for pyramidal and str. radiatum channels (stars)
        pyr_rel_depth = 0
        sw_channel_id = metadata.get('sharp_wave_band', {}).get('selected_channel_id')
        sw_rel_depth = None
        # get str. radiatum depth
        if sw_channel_id is not None and 'sharp_wave_band' in metadata:
            sw_band_ids = metadata['sharp_wave_band']['channel_ids']
            sw_band_depths = metadata['sharp_wave_band']['depths']
            if sw_channel_id in sw_band_ids:
                sw_idx = sw_band_ids.index(sw_channel_id)
                sw_depth = sw_band_depths[sw_idx]
                sw_rel_depth = ripple_depth - sw_depth
            else:
                sw_rel_depth = None
        else:
            sw_rel_depth = None
        x_left = time_rel[0]
        x_right = time_rel[-1]
        y_offset = (rel_channel_depths_plot.max() - rel_channel_depths_plot.min()) * 0.01
        # Pyramidal: only if within window
        if np.abs(pyr_rel_depth) <= max_depth:
            ax.plot(x_left, pyr_rel_depth, marker='*', color='green', markersize=18, label='Pyr. Channel')
            ax.plot(x_right, pyr_rel_depth, marker='*', color='green', markersize=18)
            ax.text(x_left, pyr_rel_depth-y_offset*4, 'Pyr. Channel', color='green', va='top', ha='left', fontweight='bold')
            ax.text(x_right, pyr_rel_depth-y_offset*4, 'Pyr. Channel', color='green', va='top', ha='right', fontweight='bold')
        # Str. radiatum: only if within window
        if sw_rel_depth is not None and np.abs(sw_rel_depth) <= max_depth:
            ax.plot(x_left, sw_rel_depth, marker='*', color='blue', markersize=18, label='Str. Rad. Channel')
            ax.plot(x_right, sw_rel_depth, marker='*', color='blue', markersize=18)
            ax.text(x_left, sw_rel_depth-y_offset*4, 'Str. Rad. Channel', color='blue', va='top', ha='left', fontweight='bold')
            ax.text(x_right, sw_rel_depth-y_offset*4, 'Str. Rad. Channel', color='blue', va='top', ha='right', fontweight='bold')
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        plt.tight_layout()
        plt.savefig(f"{save_path_base}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path_base}.svg", bbox_inches='tight')
        self.logger.info(f"CSD plots saved to {save_path_base}.png and {save_path_base}.svg")
        plt.close(fig)
        # Debug list and file
        channel_depth_debug_list = list(zip(channel_ids_plot, channel_depths_plot, rel_channel_depths_plot))
        try:
            debug_file = save_path_base + "_channel_depths_debug.txt"
            with open(debug_file, 'w') as f:
                f.write("channel_id\tchannel_depth\trel_depth_to_pyramidal\n")
                for tup in channel_depth_debug_list:
                    f.write(f"{tup[0]}\t{tup[1]}\t{tup[2]}\n")
            self.logger.info(f"Channel depth debug file written: {debug_file}")
        except Exception as e:
            self.logger.error(f"Failed to write channel depth debug file {debug_file}: {e}")
        self.logger.info(f"CSD plot used clip_range: {clip_range}")
    
    def load_putative_events(self, session_id: str, probe_id: str, channel_id: str) -> pd.DataFrame:
        """
        Load putative SWR events for a specific probe and channel.
        """
        session_dir = self.base_path / self.dataset / f"swrs_session_{session_id}"
        putative_csv = session_dir / f"probe_{probe_id}_channel_{channel_id}_putative_swr_events.csv.gz"
        if not putative_csv.exists():
            self.logger.warning(f"Putative SWR events file not found: {putative_csv}")
            return pd.DataFrame()
        return pd.read_csv(putative_csv, index_col=0)

    def process_session(self, session_id: str, max_events: int = 1) -> list:
        print(f"Processing session {session_id}")
        print(f"Looking in directory: {self.base_path / self.dataset / f'swrs_session_{session_id}'}")
        session_dir = self.base_path / self.dataset / f"swrs_session_{session_id}"
        probe_metadata_files = list(session_dir.glob("probe_*_channel_selection_metadata.json.gz"))
        print(f"Found {len(probe_metadata_files)} probe metadata files for session {session_id}: {[f.name for f in probe_metadata_files]}")
        if len(probe_metadata_files) == 0:
            raise RuntimeError(f"No probe metadata files found for session {session_id} in {self.base_path / self.dataset / f'swrs_session_{session_id}'}")
        recordings = []
        probe_ids = []
        for metadata_file in probe_metadata_files:
            probe_id = metadata_file.name.split('_')[1]
            metadata = self.load_channel_selection_metadata(session_id, probe_id)
            recordings.append({
                'probe_id': probe_id,
                'metadata': metadata
            })
            probe_ids.append(probe_id)
        print(f"Number of probes for session {session_id}: {len(recordings)}; probe_ids: {probe_ids}")
        if len(recordings) == 0:
            raise RuntimeError(f"No probe metadata files found for session {session_id}.")
        # Select the probe with the most putative events
        max_events = 0
        best_recording = None
        for recording in recordings:
            probe_id = recording['probe_id']
            metadata = recording['metadata']
            ripple_channel_id = metadata['ripple_band']['selected_channel_id']
            putative_events = self.load_putative_events(session_id, probe_id, ripple_channel_id)
            if len(putative_events) > max_events:
                max_events = len(putative_events)
                best_recording = recording
        if best_recording is None:
            raise RuntimeError(f"No probes with putative events found for session {session_id}.")
        # Only process the best probe
        processed_events = []
        probe_id = best_recording['probe_id']
        metadata = best_recording['metadata']
        ripple_channel_id = metadata['ripple_band']['selected_channel_id']
        putative_csv_path = str((self.base_path / self.dataset / f"swrs_session_{session_id}" / f"probe_{probe_id}_channel_{ripple_channel_id}_putative_swr_events.csv.gz"))
        putative_events = self.load_putative_events(session_id, probe_id, ripple_channel_id)
        print(f"Probe {probe_id}: loaded {len(putative_events)} putative events from CSV: {putative_csv_path}")
        if putative_events.empty:
            raise RuntimeError(f"No putative events found for probe {probe_id} in session {session_id} at {putative_csv_path}")
        # Filter events by power, gamma, and movement overlap
        before_filter = len(putative_events)
        putative_events = putative_events[
            (putative_events['power_max_zscore'] >= PUTATIVE_MIN_POWER_MAX_ZSCORE) &
            (putative_events['power_max_zscore'] <= PUTATIVE_MAX_POWER_MAX_ZSCORE) &
            (putative_events['sw_peak_power'] >= PUTATIVE_MIN_SW_PEAK_POWER) &
            (putative_events['overlaps_with_gamma'] == False) &
            (putative_events['overlaps_with_movement'] == False)
        ]
        print(f"Probe {probe_id}: {len(putative_events)} events after power/gamma/movement filter (was {before_filter})")
        if putative_events.empty:
            print(f"No events left after initial filtering for probe {probe_id} in session {session_id}")
            return []
        # Load speed data for the session (AllenSDK Visual Behaviour)
        if not hasattr(self, 'cache') or self.cache is None:
            raise RuntimeError("AllenSDK cache not initialized for speed data loading.")
        session_obj = self.cache.get_ecephys_session(ecephys_session_id=int(session_id))
        wheel_velocity = session_obj.running_speed['speed'].values
        wheel_time = session_obj.running_speed['timestamps'].values
        # Interpolate speed to regular time grid (optional, but for now use as is)
        speed_mask = []
        for idx, row in putative_events.iterrows():
            window_mask = (wheel_time >= row['start_time'] - 2) & (wheel_time <= row['end_time'] + 2)
            if np.any(np.abs(wheel_velocity[window_mask]) > 2.0):
                speed_mask.append(False)  # Exclude this event
            else:
                speed_mask.append(True)   # Keep this event
        n_before_speed = len(putative_events)
        putative_events = putative_events[speed_mask]
        print(f"Probe {probe_id}: {len(putative_events)} events after speed mask (was {n_before_speed})")
        if putative_events.empty:
            print(f"No events left after speed filtering for probe {probe_id} in session {session_id}")
            return []
        if 'power_max_zscore' in putative_events.columns:
            putative_events = putative_events.sort_values('power_max_zscore', ascending=False)
        ca1_channel_ids = metadata['ripple_band']['channel_ids']
        channel_depths = metadata['ripple_band']['depths']
        ripple_depth = channel_depths[ca1_channel_ids.index(ripple_channel_id)]
        for i, (_, best_event) in enumerate(putative_events.iterrows()):
            if i >= MAX_EVENTS_PER_PROBE:
                break
            peak_time = best_event['power_peak_time'] if 'power_peak_time' in best_event else (best_event['start_time'] + best_event['end_time']) / 2
            duration = best_event['duration']
            window = 0.25
            rel_channel_depths = ripple_depth - np.array(channel_depths)
            csd_mask = np.abs(rel_channel_depths) <= CSD_COMPUTE_DEPTH_RANGE
            ca1_channel_ids_csd = list(np.array(ca1_channel_ids)[csd_mask])
            channel_depths_csd = list(np.array(channel_depths)[csd_mask])
            lfp_data, time_axis, used_channel_ids = self.load_lfp_data_from_allensdk(
                session_id, probe_id, peak_time, window, ca1_channel_ids_csd
            )
            csd_data = self.compute_csd_for_event(lfp_data)
            event_info = {
                'session_id': session_id,
                'probe_id': probe_id,
                'event_id': int(best_event.name),
                'duration': duration,
                'power_max_zscore': best_event.get('power_max_zscore', 0),
                'peak_time': peak_time,
            }
            event_folder = self.data_dir / f"session_{session_id}_event_{best_event.name}"
            event_folder.mkdir(exist_ok=True)
            output_file = event_folder / f"probe_{probe_id}_csd_data.npz"
            np.savez(output_file, 
                    csd_data=csd_data,
                    time_axis=time_axis,
                    channel_depths=channel_depths_csd,
                    event_info=event_info,
                    metadata=metadata)
            plots_event_folder = self.plots_dir / f"session_{session_id}_event_{best_event.name}"
            plots_event_folder.mkdir(exist_ok=True)
            plot_file_base = plots_event_folder / f"probe_{probe_id}_csd_plot"
            self.plot_csd_event(csd_data, time_axis, CSD_PLOT_DEPTH_RANGE, channel_depths_csd, event_info, 
                              metadata, str(plot_file_base), clip_range=CSD_CLIP_RANGE)
            processed_events.append({
                'session_id': session_id,
                'probe_id': probe_id,
                'event_id': int(best_event.name),
                'duration': duration,
                'power_max_zscore': best_event.get('power_max_zscore', 0),
                'peak_time': peak_time,
                'csd_file': str(output_file),
                'plot_files': [f"{plot_file_base}.png", f"{plot_file_base}.svg"],
                'event_folder': str(event_folder),
                'plots_folder': str(plots_event_folder)
            })
        return processed_events
    
    def find_best_sessions_and_events(self) -> List[dict]:
        # Only select sessions based on probe/channel density, not global events
        dataset_dir = self.base_path / self.dataset
        session_dirs = list(dataset_dir.glob("swrs_session_*"))
        if not session_dirs:
            self.logger.error(f"No session directories found in {dataset_dir}")
            return []
        all_processed_events = []
        for session_dir in session_dirs[:MAX_SESSIONS_TO_PROCESS]:
            session_id = session_dir.name.split('_')[-1]
            self.logger.info(f"Processing session {session_id}")
            processed_events = self.process_session(session_id, MAX_EVENTS_PER_PROBE)
            all_processed_events.extend(processed_events)
        # Save summary
        summary_file = self.output_dir / "csd_events_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_processed_events, f, indent=2)
        self.logger.info(f"Processed {len(all_processed_events)} events from {len(session_dirs[:MAX_SESSIONS_TO_PROCESS])} sessions")
        self.logger.info(f"Summary saved to {summary_file}")
        return all_processed_events

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main function to run the CSD SWR events workflow."""
    print("CSD SWR Events Workflow")
    print("=" * 50)
    print(f"Base data path: {BASE_DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max sessions to process: {MAX_SESSIONS_TO_PROCESS}")
    print(f"Max events per session: {MAX_EVENTS_PER_SESSION}")
    print(f"Peak power range: {MIN_PEAK_POWER} - {MAX_PEAK_POWER}")
    print(f"Min participating probes: {MIN_PARTICIPATING_PROBES}")
    print("=" * 50)
    
    # Create workflow
    workflow = CSDSWREventsWorkflow(BASE_DATA_PATH, OUTPUT_DIR)
    
    # Run workflow
    processed_events = workflow.find_best_sessions_and_events()
    
    print(f"\nWorkflow completed successfully!")
    print(f"Processed {len(processed_events)} events.")
    print(f"Results saved to {OUTPUT_DIR}")
    print(f"  - CSD data: {workflow.data_dir}")
    print(f"  - CSD plots: {workflow.plots_dir}")
    print(f"  - Summary: {workflow.output_dir}/csd_events_summary.json")

if __name__ == "__main__":
    main() 