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
CSD Trial Average Workflow for ABI Visual Behaviour Dataset

This script implements a modified version of the CSD workflow that computes trial-averaged CSD
across all events that pass the threshold, rather than plotting individual events.

Based on the original csd_swr_events_workflow.py but modified for trial averaging.
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
MAX_EVENTS_PER_PROBE = 100

# DEBUG PARAMETER - Limit number of events for quick testing
DEBUG_EVENT_LIMIT = 100  # Set to None to use all events, or set to a number for debugging

# PROBE SELECTION PARAMETER
PROBE_RANK_TO_PROCESS = 3  # 0 = best probe, 1 = second best, 2 = third best, etc.
# Set to None to process the best probe automatically

# TRIAL AVERAGING PARAMETERS
ENABLE_TRIAL_AVERAGING = True  # Set to False to process individual events like original
TRIAL_AVERAGE_METHOD = "mean"  # "mean" or "median"

# Smoothing parameters (turned off for trial averaging)
SMOOTHING_METHOD = "none"  # "exponential" or "gaussian" or "sav-gol" or "none"

# Smoothing parameters
if SMOOTHING_METHOD == "exponential":
    CSD_SMOOTHING_ENABLED = True
else:
    CSD_SMOOTHING_ENABLED = False
CSD_SMOOTH_ALPHA_SPATIAL = 0.7
CSD_SMOOTH_ALPHA_TEMPORAL = 0

# Gaussian smoothing parameters
if SMOOTHING_METHOD == "gaussian":
    CSD_GAUSSIAN_SMOOTHING_ENABLED = True
else:
    CSD_GAUSSIAN_SMOOTHING_ENABLED = False
CSD_GAUSSIAN_SIGMA = 1.0
CSD_GAUSSIAN_TRUNCATE = 3.0

# Savitzky-Golay smoothing parameters
if SMOOTHING_METHOD == "sav-gol":
    CSD_SAVGOL_SMOOTHING_ENABLED = True
else:
    CSD_SAVGOL_SMOOTHING_ENABLED = False
CSD_SAVGOL_WINDOW_LENGTH = 3
CSD_SAVGOL_POLYORDER = 2

# =============================================================================
# INPUT/OUTPUT PATHS - MODIFY THESE AS NEEDED
# =============================================================================
BASE_DATA_PATH = "/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_v2_final"
OUTPUT_DIR = "/home/acampbell/NeuropixelsLFPOnRamp/Figures_Tables_and_Technical_Validation/CSD_plot/csd_trial_mean"

# Session and event search parameters
MAX_SESSIONS_TO_PROCESS = 1
MAX_EVENTS_PER_SESSION = 1
MIN_PROBE_COUNT = 3
MAX_PROBE_COUNT = 6
MIN_PEAK_POWER = 5.0
MAX_PEAK_POWER = 10.0
MIN_PARTICIPATING_PROBES = 2

# Depth range for plotting (microns from pyramidal layer)
CSD_CLIP_RANGE = (-2, 2)  # (min, max) for CSD color scale

# =============================================================================
# PUTATIVE EVENT FILTERING THRESHOLDS
# =============================================================================
PUTATIVE_MIN_POWER_MAX_ZSCORE = 3.0
PUTATIVE_MAX_POWER_MAX_ZSCORE = 10.0
PUTATIVE_MIN_SW_PEAK_POWER = 1.0
PUTATIVE_SPEED_THRESHOLD = 2.0  # cm/s
PUTATIVE_SPEED_EXCLUSION_WINDOW = 0.5  # seconds


# =============================================================================
# CSD FUNCTIONS (from original workflow)
# =============================================================================
def average_channels(electrode_data):
    """Calculate the average of every pair of even and odd channels"""
    nchan = electrode_data.shape[1]
    mask = np.full(nchan, False, dtype=bool)
    mask[::2] = True
    even_channels = electrode_data[:, mask]
    odd_channels = electrode_data[:, ~mask]
    
    min_length = min(even_channels.shape[1], odd_channels.shape[1])
    even_channels = even_channels[:, :min_length]
    odd_channels = odd_channels[:, :min_length]
    
    averaged_data = (even_channels + odd_channels) / 2
    return averaged_data

def butterworth_filter_for_csd(
    LFP_array, lowcut=1.0, highcut=250.0, samplingfreq=2500.0, order=6
):
    nyquist = 0.5 * samplingfreq
    b, a = signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    filtered_signal = signal.lfilter(b, a, LFP_array, axis=0)
    return filtered_signal

def finitimpresp_filter_for_csd(
    LFP_array, lowcut=1, highcut=250, samplingfreq=2500, filter_order=101
):
    nyquist = 0.5 * samplingfreq
    fir_coeff = signal.firwin(
        filter_order,
        [lowcut / nyquist, highcut / nyquist],
        pass_zero=False,
        fs=samplingfreq,
    )
    filtered_signal = signal.lfilter(fir_coeff, 1.0, LFP_array, axis=0)
    return filtered_signal

def compute_csd(lfp_data, spacing_between_channels):
    """
    Compute Current Source Density (CSD) from Local Field Potential (LFP) data.
    """
    csd_data = np.zeros_like(lfp_data)
    csd_data[1:-1, :] = (lfp_data[2:, :] - 2 * lfp_data[1:-1, :] + lfp_data[:-2, :]) / (spacing_between_channels ** 2)
    csd_data[0, :] = 0
    csd_data[-1, :] = 0
    return csd_data

def exponential_smoothing_2d(data, alpha, axis):
    """Apply exponential smoothing to a 2D NumPy array along the specified axis."""
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
    """Apply exponential smoothing to a 1D NumPy array."""
    smoothed_series = np.zeros_like(series)
    smoothed_series[0] = series[0]

    for t in range(1, len(series)):
        smoothed_series[t] = alpha * series[t] + (1 - alpha) * smoothed_series[t - 1]

    return smoothed_series

def gaussian_smoothing_2d(data, sigma, axis, truncate=3.0):
    """Apply Gaussian smoothing to a 2D array along the specified axis."""
    return gaussian_filter1d(data, sigma=sigma, axis=axis, truncate=truncate, mode='nearest')

# =============================================================================
# TRIAL AVERAGING FUNCTIONS
# =============================================================================
def compute_trial_averaged_csd(csd_events_3d, method="mean"):
    """
    Compute trial-averaged CSD across the events dimension.
    
    Parameters:
    -----------
    csd_events_3d : np.ndarray
        3D array of shape (n_events, n_channels, n_timepoints)
    method : str
        "mean" or "median"
        
    Returns:
    --------
    np.ndarray
        2D array of shape (n_channels, n_timepoints)
    """
    if method == "mean":
        return np.mean(csd_events_3d, axis=0)
    elif method == "median":
        return np.median(csd_events_3d, axis=0)
    else:
        raise ValueError(f"Unknown averaging method: {method}")

def align_events_to_peak(csd_events_3d, time_axes, peak_times):
    """
    Align CSD events to their peak times by interpolating to a common time grid.
    
    Parameters:
    -----------
    csd_events_3d : np.ndarray
        3D array of shape (n_events, n_channels, n_timepoints)
    time_axes : list
        List of time axes for each event
    peak_times : list
        List of peak times for each event
        
    Returns:
    --------
    tuple
        (aligned_csd_3d, common_time_axis)
    """
    # Find the common time range and resolution
    all_time_ranges = []
    for time_axis, peak_time in zip(time_axes, peak_times):
        rel_time = time_axis - peak_time
        all_time_ranges.append((rel_time.min(), rel_time.max()))
    
    # Use the most restrictive range
    min_rel_time = max([t[0] for t in all_time_ranges])
    max_rel_time = min([t[1] for t in all_time_ranges])
    
    # Use the finest time resolution
    all_dts = []
    for time_axis in time_axes:
        if len(time_axis) > 1:
            dt = np.median(np.diff(time_axis))
            all_dts.append(dt)
    
    if not all_dts:
        raise ValueError("No valid time axes found")
    
    dt = min(all_dts)
    
    # Create common time axis
    common_time_axis = np.arange(min_rel_time, max_rel_time + dt, dt)
    
    # Interpolate each event to common time axis
    n_events, n_channels, _ = csd_events_3d.shape
    aligned_csd_3d = np.zeros((n_events, n_channels, len(common_time_axis)))
    
    for i in range(n_events):
        rel_time = time_axes[i] - peak_times[i]
        for j in range(n_channels):
            # Interpolate this channel's data
            aligned_csd_3d[i, j, :] = np.interp(
                common_time_axis, 
                rel_time, 
                csd_events_3d[i, j, :],
                left=np.nan, 
                right=np.nan
            )
    
    return aligned_csd_3d, common_time_axis

# =============================================================================
# MAIN WORKFLOW CLASS
# =============================================================================
class CSDTrialAverageWorkflow:
    """
    Workflow for creating trial-averaged CSD plots of SWR events.
    """
    
    def __init__(self, base_path: str, output_dir: str):
        """Initialize the workflow."""
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
        self.min_duration = 0.06
        self.max_duration = 0.150
        self.speed_threshold = 2.0
        self.speed_exclusion_window = 2.0
        self.plot_padding = 0.25
        self.channel_spacing = 40.0
        
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
            cache_dir = "/space/scratch/allen_visbehave_data"
            self.cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)
            self.logger.info(f"AllenSDK cache initialized from {cache_dir}")
        except Exception as e:
            self.logger.warning(f"Could not initialize AllenSDK cache: {e}")
            self.cache = None
    
    def load_channel_selection_metadata(self, session_id: str, probe_id: str) -> dict:
        """Load channel selection metadata for a specific probe."""
        session_dir = self.base_path / self.dataset / f"swrs_session_{session_id}"
        metadata_file = session_dir / f"probe_{probe_id}_channel_selection_metadata.json.gz"
        
        if not metadata_file.exists():
            self.logger.warning(f"Metadata file not found: {metadata_file}")
            return None
            
        with gzip.open(metadata_file, 'rt') as f:
            return json.load(f)
    
    def load_putative_events(self, session_id: str, probe_id: str, channel_id: str) -> pd.DataFrame:
        """Load putative SWR events for a specific probe and channel."""
        session_dir = self.base_path / self.dataset / f"swrs_session_{session_id}"
        putative_csv = session_dir / f"probe_{probe_id}_channel_{channel_id}_putative_swr_events.csv.gz"
        if not putative_csv.exists():
            self.logger.warning(f"Putative SWR events file not found: {putative_csv}")
            return pd.DataFrame()
        return pd.read_csv(putative_csv, index_col=0)
    
    def load_lfp_data_from_allensdk(self, session_id: str, probe_id: str, 
                                   peak_time: float, window: float,
                                   ca1_channel_ids: list) -> tuple:
        """Load LFP data for selected CA1 channels from AllenSDK."""
        if not self.cache:
            raise RuntimeError("AllenSDK cache not available")
        session = self.cache.get_ecephys_session(ecephys_session_id=int(session_id))
        lfp = session.get_lfp(int(probe_id))
        
        lfp_channel_ids = set(lfp.channel.values)
        valid_ca1_channel_ids = [cid for cid in ca1_channel_ids if cid in lfp_channel_ids]
        missing = set(ca1_channel_ids) - lfp_channel_ids
        if missing:
            self.logger.warning(f"Probe {probe_id}: {len(missing)} channel IDs missing from LFP.")
        if not valid_ca1_channel_ids:
            raise ValueError(f"No valid CA1 channel IDs found in LFP for probe {probe_id}.")
        
        lfp_sel = lfp.sel(channel=valid_ca1_channel_ids)
        lfp_data = lfp_sel.to_numpy().T
        time_axis = lfp_sel.time.values
        
        # Restrict to window around peak_time
        t0 = peak_time - window
        t1 = peak_time + window
        mask = (time_axis >= t0) & (time_axis <= t1)
        lfp_data = lfp_data[:, mask]
        time_axis = time_axis[mask]
        
        # Ensure window is always 2*window seconds, pad if needed
        expected_len = int(round((2 * window) / np.median(np.diff(time_axis)))) + 1 if len(time_axis) > 1 else 0
        if len(time_axis) > 1 and len(time_axis) < expected_len:
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
        """Compute CSD for LFP data."""
        return compute_csd(lfp_data, self.channel_spacing)
    
    def plot_trial_averaged_csd(self, csd_data: np.ndarray, time_axis: np.ndarray, 
                               depth_range: tuple, channel_depths: list, 
                               event_info: dict, metadata: dict, 
                               save_path_base: str, clip_range: tuple,
                               n_events: int) -> None:
        """Plot trial-averaged CSD event."""
        # Ensure dimensions match
        if csd_data.shape[0] != len(channel_depths):
            self.logger.warning(f"CSD data has {csd_data.shape[0]} channels but {len(channel_depths)} depths provided.")
            min_channels = min(csd_data.shape[0], len(channel_depths))
            csd_data = csd_data[:min_channels, :]
            channel_depths = channel_depths[:min_channels]
        
        if csd_data.shape[1] != len(time_axis):
            self.logger.warning(f"CSD data has {csd_data.shape[1]} time points but {len(time_axis)} time values provided.")
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
        
        # Clipping
        csd_plot = np.clip(csd_data_plot, *clip_range)
        
        # Center time axis on peak (should be 0 for trial-averaged data)
        time_rel = time_axis
        
        fig, ax = plt.subplots(figsize=(12, 8))
        time_mesh, depth_mesh = np.meshgrid(time_rel, rel_channel_depths_plot)
        im = ax.pcolormesh(time_mesh, depth_mesh, csd_plot, cmap='RdBu_r', shading='gouraud')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Current Source Density (μA/mm³)', rotation=270, labelpad=20)
        ax.set_xlabel('Time relative to peak (s)')
        ax.set_ylabel('Depth relative to pyramidal layer (μm)')
        
        # Set y-ticks to show both rel depth and actual depth
        yticks = rel_channel_depths_plot
        yticklabels = [f"{int(rd)} ({int(cd)})" for rd, cd in zip(rel_channel_depths_plot, channel_depths_plot)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        
        title = f"Trial-Averaged CSD - Session {event_info.get('session_id', 'Unknown')}, "
        title += f"Probe {event_info.get('probe_id', 'Unknown')}\n"
        title += f"Method: {TRIAL_AVERAGE_METHOD.capitalize()}, "
        title += f"N Events: {n_events}"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Mark the peak time (black line at t=0)
        ax.axvline(0, color='black', linestyle='--', alpha=0.7, label='Peak Time')
        
        # Markers for pyramidal and str. radiatum channels
        pyr_rel_depth = 0
        sw_channel_id = metadata.get('sharp_wave_band', {}).get('selected_channel_id')
        sw_rel_depth = None
        
        if sw_channel_id is not None and 'sharp_wave_band' in metadata:
            sw_band_ids = metadata['sharp_wave_band']['channel_ids']
            sw_band_depths = metadata['sharp_wave_band']['depths']
            if sw_channel_id in sw_band_ids:
                sw_idx = sw_band_ids.index(sw_channel_id)
                sw_depth = sw_band_depths[sw_idx]
                sw_rel_depth = ripple_depth - sw_depth
        
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
        self.logger.info(f"Trial-averaged CSD plots saved to {save_path_base}.png and {save_path_base}.svg")
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
        
        self.logger.info(f"Trial-averaged CSD plot used clip_range: {clip_range}")

    def process_session_trial_average(self, session_id: str) -> dict:
        """Process a session and compute trial-averaged CSD."""
        print(f"Processing session {session_id} for trial averaging")
        session_dir = self.base_path / self.dataset / f"swrs_session_{session_id}"
        probe_metadata_files = list(session_dir.glob("probe_*_channel_selection_metadata.json.gz"))
        
        if len(probe_metadata_files) == 0:
            raise RuntimeError(f"No probe metadata files found for session {session_id}")
        
        # Rank all probes by number of putative events
        probe_rankings = []
        
        for metadata_file in probe_metadata_files:
            probe_id = metadata_file.name.split('_')[1]
            metadata = self.load_channel_selection_metadata(session_id, probe_id)
            if not metadata:
                continue
                
            ripple_channel_id = metadata['ripple_band']['selected_channel_id']
            putative_events = self.load_putative_events(session_id, probe_id, ripple_channel_id)
            
            probe_rankings.append({
                'probe_id': probe_id,
                'metadata': metadata,
                'n_events': len(putative_events),
                'ripple_channel_id': ripple_channel_id
            })
        
        # Sort by number of events (descending)
        probe_rankings.sort(key=lambda x: x['n_events'], reverse=True)
        
        if not probe_rankings:
            raise RuntimeError(f"No probes with putative events found for session {session_id}")
        
        # Select probe based on rank
        if PROBE_RANK_TO_PROCESS is None:
            selected_rank = 0  # Default to best probe
        else:
            selected_rank = PROBE_RANK_TO_PROCESS
            
        if selected_rank >= len(probe_rankings):
            raise RuntimeError(f"Probe rank {selected_rank} requested but only {len(probe_rankings)} probes available")
        
        selected_probe = probe_rankings[selected_rank]
        probe_id = selected_probe['probe_id']
        metadata = selected_probe['metadata']
        ripple_channel_id = selected_probe['ripple_channel_id']
        
        print(f"Selected probe {probe_id} (rank {selected_rank}) with {selected_probe['n_events']} putative events")
        print(f"Available probes: {[(p['probe_id'], p['n_events']) for p in probe_rankings]}")
        
        putative_events = self.load_putative_events(session_id, probe_id, ripple_channel_id)
        print(f"Probe {probe_id}: loaded {len(putative_events)} putative events")
        
        if putative_events.empty:
            raise RuntimeError(f"No putative events found for probe {probe_id} in session {session_id}")
        
        # Filter events
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
            return {}
        
        # Load speed data and filter by movement
        if not hasattr(self, 'cache') or self.cache is None:
            raise RuntimeError("AllenSDK cache not initialized for speed data loading.")
        
        session_obj = self.cache.get_ecephys_session(ecephys_session_id=int(session_id))
        wheel_velocity = session_obj.running_speed['speed'].values
        wheel_time = session_obj.running_speed['timestamps'].values
        
        speed_mask = []
        for idx, row in putative_events.iterrows():
            window_mask = (wheel_time >= row['start_time'] - 2) & (wheel_time <= row['end_time'] + 2)
            if np.any(np.abs(wheel_velocity[window_mask]) > 2.0):
                speed_mask.append(False)
            else:
                speed_mask.append(True)
        
        n_before_speed = len(putative_events)
        putative_events = putative_events[speed_mask]
        print(f"Probe {probe_id}: {len(putative_events)} events after speed mask (was {n_before_speed})")
        
        if putative_events.empty:
            print(f"No events left after speed filtering for probe {probe_id} in session {session_id}")
            return {}
        
        # Sort by power and take top events
        if 'power_max_zscore' in putative_events.columns:
            putative_events = putative_events.sort_values('power_max_zscore', ascending=False)
        
        # Limit number of events for processing
        putative_events = putative_events.head(MAX_EVENTS_PER_PROBE)
        
        # Apply debug limit if specified
        if DEBUG_EVENT_LIMIT is not None:
            putative_events = putative_events.head(DEBUG_EVENT_LIMIT)
            print(f"DEBUG: Limiting to {DEBUG_EVENT_LIMIT} events for testing")
        
        # Prepare for trial averaging
        ca1_channel_ids = metadata['ripple_band']['channel_ids']
        channel_depths = metadata['ripple_band']['depths']
        ripple_depth = channel_depths[ca1_channel_ids.index(ripple_channel_id)]
        
        # Calculate relative depths and create CSD mask
        rel_channel_depths = ripple_depth - np.array(channel_depths)
        csd_mask = np.abs(rel_channel_depths) <= CSD_COMPUTE_DEPTH_RANGE
        ca1_channel_ids_csd = list(np.array(ca1_channel_ids)[csd_mask])
        channel_depths_csd = list(np.array(channel_depths)[csd_mask])
        
        # Collect CSD data for all events
        csd_events_list = []
        time_axes_list = []
        peak_times_list = []
        event_info_list = []
        
        window = 0.25
        
        for i, (_, event) in enumerate(putative_events.iterrows()):
            peak_time = event['power_peak_time'] if 'power_peak_time' in event else (event['start_time'] + event['end_time']) / 2
            duration = event['duration']
            
            try:
                lfp_data, time_axis, used_channel_ids = self.load_lfp_data_from_allensdk(
                    session_id, probe_id, peak_time, window, ca1_channel_ids_csd
                )
                csd_data = self.compute_csd_for_event(lfp_data)
                
                csd_events_list.append(csd_data)
                time_axes_list.append(time_axis)
                peak_times_list.append(peak_time)
                
                event_info_list.append({
                    'session_id': session_id,
                    'probe_id': probe_id,
                    'event_id': int(event.name),
                    'duration': duration,
                    'power_max_zscore': event.get('power_max_zscore', 0),
                    'peak_time': peak_time,
                })
                
                print(f"Processed event {i+1}/{len(putative_events)}")
                
            except Exception as e:
                self.logger.warning(f"Failed to process event {event.name}: {e}")
                continue
        
        if not csd_events_list:
            raise RuntimeError(f"No events successfully processed for probe {probe_id}")
        
        # Convert to 3D array
        csd_events_3d = np.stack(csd_events_list, axis=0)  # Shape: (n_events, n_channels, n_timepoints)
        
        # Align events to peak time
        aligned_csd_3d, common_time_axis = align_events_to_peak(csd_events_3d, time_axes_list, peak_times_list)
        
        # Compute trial average
        trial_averaged_csd = compute_trial_averaged_csd(aligned_csd_3d, TRIAL_AVERAGE_METHOD)
        
        # Create event info for plotting
        avg_event_info = {
            'session_id': session_id,
            'probe_id': probe_id,
            'n_events': len(csd_events_list),
            'method': TRIAL_AVERAGE_METHOD,
            'avg_duration': np.mean([info['duration'] for info in event_info_list]),
            'avg_power_zscore': np.mean([info['power_max_zscore'] for info in event_info_list]),
        }
        
        # Save data
        output_file = self.data_dir / f"session_{session_id}_probe_{probe_id}_trial_averaged_csd.npz"
        np.savez(output_file, 
                csd_data=trial_averaged_csd,
                time_axis=common_time_axis,
                channel_depths=channel_depths_csd,
                event_info=avg_event_info,
                metadata=metadata,
                n_events=len(csd_events_list),
                individual_csd_data=aligned_csd_3d)
        
        # Create plot
        plot_file_base = self.plots_dir / f"session_{session_id}_probe_{probe_id}_trial_averaged_csd"
        self.plot_trial_averaged_csd(trial_averaged_csd, common_time_axis, CSD_PLOT_DEPTH_RANGE, 
                                   channel_depths_csd, avg_event_info, metadata, 
                                   str(plot_file_base), clip_range=CSD_CLIP_RANGE,
                                   n_events=len(csd_events_list))
        
        return {
            'session_id': session_id,
            'probe_id': probe_id,
            'n_events': len(csd_events_list),
            'method': TRIAL_AVERAGE_METHOD,
            'csd_file': str(output_file),
            'plot_files': [f"{plot_file_base}.png", f"{plot_file_base}.svg"],
            'avg_duration': avg_event_info['avg_duration'],
            'avg_power_zscore': avg_event_info['avg_power_zscore'],
        }
    
    def find_best_sessions_and_events(self) -> List[dict]:
        """Find best sessions and compute trial-averaged CSD."""
        dataset_dir = self.base_path / self.dataset
        session_dirs = list(dataset_dir.glob("swrs_session_*"))
        
        if not session_dirs:
            self.logger.error(f"No session directories found in {dataset_dir}")
            return []
        
        all_processed_events = []
        
        for session_dir in session_dirs[:MAX_SESSIONS_TO_PROCESS]:
            session_id = session_dir.name.split('_')[-1]
            self.logger.info(f"Processing session {session_id}")
            
            try:
                if ENABLE_TRIAL_AVERAGING:
                    processed_event = self.process_session_trial_average(session_id)
                    if processed_event:
                        all_processed_events.append(processed_event)
                else:
                    # Fall back to individual event processing (not implemented here)
                    self.logger.warning("Individual event processing not implemented in this version")
                    
            except Exception as e:
                self.logger.error(f"Failed to process session {session_id}: {e}")
                continue
        
        # Save summary
        summary_file = self.output_dir / "trial_averaged_csd_events_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_processed_events, f, indent=2)
        
        self.logger.info(f"Processed {len(all_processed_events)} trial-averaged CSD results from {len(session_dirs[:MAX_SESSIONS_TO_PROCESS])} sessions")
        self.logger.info(f"Summary saved to {summary_file}")
        
        return all_processed_events

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main function to run the trial-averaged CSD workflow."""
    print("CSD Trial Average Workflow")
    print("=" * 50)
    print(f"Base data path: {BASE_DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Trial averaging enabled: {ENABLE_TRIAL_AVERAGING}")
    print(f"Averaging method: {TRIAL_AVERAGE_METHOD}")
    print(f"Max sessions to process: {MAX_SESSIONS_TO_PROCESS}")
    print(f"Max events per probe: {MAX_EVENTS_PER_PROBE}")
    if DEBUG_EVENT_LIMIT is not None:
        print(f"DEBUG: Event limit for testing: {DEBUG_EVENT_LIMIT}")
    if PROBE_RANK_TO_PROCESS is not None:
        print(f"Processing probe rank: {PROBE_RANK_TO_PROCESS} (0 = best, 1 = second best, etc.)")
    else:
        print("Processing best probe automatically")
    print("=" * 50)
    
    # Create workflow
    workflow = CSDTrialAverageWorkflow(BASE_DATA_PATH, OUTPUT_DIR)
    
    # Run workflow
    processed_events = workflow.find_best_sessions_and_events()
    
    print(f"\nWorkflow completed successfully!")
    print(f"Processed {len(processed_events)} trial-averaged CSD results.")
    print(f"Results saved to {OUTPUT_DIR}")
    print(f"  - CSD data: {workflow.data_dir}")
    print(f"  - CSD plots: {workflow.plots_dir}")
    print(f"  - Summary: {workflow.output_dir}/trial_averaged_csd_events_summary.json")

if __name__ == "__main__":
    main() 