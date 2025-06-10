# Standard library imports
# swr_neuropixels_collection_core
import os
import time
import sys
import subprocess
import traceback
import json
import gzip
import string
import glob
import re

# Third-party data processing libraries
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

# SciPy imports
from scipy import io, signal, stats, interpolate
from scipy.signal import lfilter, hilbert, fftconvolve
import scipy.ndimage
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import pearsonr, skew


# Multiprocessing
from multiprocessing import Pool, Process, Queue, Manager, set_start_method

# AWS
import boto3
from botocore.config import Config

# Ripple detection
import ripple_detection
from ripple_detection import filter_ripple_band
import ripple_detection.simulate as ripsim
from ripple_detection.core import gaussian_smooth

# Logging
import logging
import logging.handlers

# United SWR detector
import os
import subprocess
import numpy as np
import pandas as pd
from scipy import io, signal, stats
from scipy.signal import lfilter
import scipy.ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
import matplotlib.pyplot as plt
import ripple_detection
from ripple_detection import filter_ripple_band
import ripple_detection.simulate as ripsim  # for making our time vectors
from tqdm import tqdm
import time
import traceback
import logging
import logging.handlers
import sys
from multiprocessing import Pool, Process, Queue, Manager, set_start_method
import yaml
import json
import gzip
import string
from botocore.config import Config
import boto3
import shutil


# ===================================
# Helper functions
# ===================================

def read_json_file(file_path):
    """Read a JSON file and return the loaded data."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_filter(filter_path):
    """Load filter coefficients from a file."""
    filter_data = np.load(filter_path)
    return filter_data["arr_0"]

# ===================================
# BASE LOADER CLASS
# ===================================
# swr_neuropixels_collection_core.py

# Standard library imports
import os
import time
import sys
import importlib

# Common numerical/scientific imports
import numpy as np
from scipy import signal, interpolate, stats
from scipy.signal import hilbert, fftconvolve

class BaseLoader:
    """
    Base class for all dataset loaders with common functionality.
    """
    def __init__(self, session_id):
        self.session_id = session_id

        # Initialize dictionary to store channel selection metadata
        self.channel_selection_metadata_dict = {
            'probe_id': str(session_id), # Store probe ID here during setup/processing
            'ripple_band': { # Metrics related to ripple channel selection
                'channel_ids': [], # All CA1 channel IDs evaluated
                'depths': [],      # Corresponding depths
                'skewness': [],    # Skewness of ripple power per channel
                'net_power': [],   # Net ripple power per channel
                'selected_channel_id': None,
                'selection_method': None
            },
            'sharp_wave_band': { # Metrics related to SW channel selection
                'channel_ids': [], # Channel IDs below ripple chan evaluated
                'depths': [],      # Corresponding depths
                'net_sw_power': [],# Net SW power during co-activity
                'modulation_index': [],
                'circular_linear_corrs': [],
                'selected_channel_id': None,
                'selection_method': None
            }
            # Note: We might rename 'channel idx' and 'depths' from instructions
            # to be nested under ripple/sw bands for clarity.
        }

    @staticmethod
    def create(dataset_type, session_id):
        """
        Factory method to dynamically load and instantiate the appropriate loader.
        
        Parameters
        ----------
        dataset_type : str
            Type of dataset ('ibl', 'abi_visual_behaviour', or 'abi_visual_coding')
        session_id : str or int
            Session identifier
            
        Returns
        -------
        object
            Instance of the appropriate loader class
        """
        if dataset_type == 'ibl':
            # Dynamically import IBL loader
            try:
                from IBL_loader import ibl_loader
                return ibl_loader(session_id)
            except ImportError:
                print("IBL dependencies not available in this environment")
                raise
                
        elif dataset_type == 'abi_visual_behaviour':
            # Dynamically import ABI visual behaviour loader
            try:
                from ABI_visual_behaviour_loader import abi_visual_behaviour_loader
                return abi_visual_behaviour_loader(session_id)
            except ImportError:
                print("ABI Visual Behaviour dependencies not available in this environment")
                raise
                
        elif dataset_type == 'abi_visual_coding':
            # Dynamically import ABI visual coding loader
            try:
                from ABI_visual_coding_loader import abi_visual_coding_loader
                return abi_visual_coding_loader(session_id)
            except ImportError:
                print("ABI Visual Coding dependencies not available in this environment")
                raise
                
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def resample_signal(self, signal_data, time_values, target_fs=1500.0):
        """
        Resamples a signal to the target sampling frequency.
        
        Parameters
        ----------
        signal_data : numpy.ndarray
            Signal data to resample
        time_values : numpy.ndarray
            Time values corresponding to the signal data
        target_fs : float, optional
            Target sampling frequency
            
        Returns
        -------
        tuple
            (resampled_signal, new_time_values)
        """
        # Create new time index
        t_start = time_values[0]
        t_end = time_values[-1]
        dt_new = 1.0 / target_fs
        n_samples = int(np.ceil((t_end - t_start) / dt_new))
        new_time_values = t_start + np.arange(n_samples) * dt_new
        
        # Resample signal
        if signal_data.ndim == 1:
            # For 1D signals
            interp_func = interpolate.interp1d(
                time_values, signal_data, bounds_error=False, fill_value="extrapolate"
            )
            resampled = interp_func(new_time_values)
        else:
            # For multi-channel signals
            # Check orientation - assume we want (time, channels)
            if signal_data.shape[0] > signal_data.shape[1] and len(time_values) == signal_data.shape[0]:
                # Data is already in (time, channels) format
                resampled = np.zeros((len(new_time_values), signal_data.shape[1]))
                for i in range(signal_data.shape[1]):
                    interp_func = interpolate.interp1d(
                        time_values, signal_data[:, i], bounds_error=False, fill_value="extrapolate"
                    )
                    resampled[:, i] = interp_func(new_time_values)
            else:
                # Data is in (channels, time) format
                resampled = np.zeros((signal_data.shape[0], len(new_time_values)))
                for i in range(signal_data.shape[0]):
                    interp_func = interpolate.interp1d(
                        time_values, signal_data[i, :], bounds_error=False, fill_value="extrapolate"
                    )
                    resampled[i, :] = interp_func(new_time_values)
                
                # Transpose to our standard format (time, channels)
                resampled = resampled.T
        
        return resampled, new_time_values

    def _compute_modulation_index(self, sw_phase, ripple_amp, valid_mask, n_bins=18):
        """
        Compute the modulation index between sharp wave phase and ripple amplitude.

        Parameters
        ----------
        sw_phase : np.ndarray
            Instantaneous phase of the sharp wave filtered signal (in radians).
        ripple_amp : np.ndarray
            Instantaneous amplitude envelope of the ripple filtered signal.
        valid_mask : np.ndarray of bool
            Boolean mask selecting valid time points to include (e.g., high power, no artifacts).

        Returns
        -------
        float
            Modulation index. Returns np.nan if insufficient data is available.
        """
        sw_phase = sw_phase[valid_mask]
        ripple_amp = ripple_amp[valid_mask]
        if ripple_amp.size < 10:
            return np.nan

        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        digitized = np.digitize(sw_phase, bins) - 1
        amp_by_bin = np.array([
            ripple_amp[digitized == j].mean() if np.any(digitized == j) else 0
            for j in range(n_bins)
        ])
        if amp_by_bin.sum() == 0:
            return np.nan

        P = amp_by_bin / amp_by_bin.sum()
        H = -np.sum(P * np.log(P + 1e-10))
        H_max = np.log(n_bins)
        return (H_max - H) / H_max

    def _compute_circular_linear_corr(self, sw_phase, ripple_amp, valid_mask):
        """
        Compute the circular-linear correlation between sharp wave phase and ripple amplitude.

        Parameters
        ----------
        sw_phase : np.ndarray
            Instantaneous phase of the sharp wave filtered signal (in radians).
        ripple_amp : np.ndarray
            Instantaneous amplitude envelope of the ripple filtered signal.
        valid_mask : np.ndarray of bool
            Boolean mask selecting valid time points to include (e.g., high power, no artifacts).

        Returns
        -------
        float
            Circular-linear correlation coefficient. Returns np.nan if insufficient data is available.
        """
        sw_phase = sw_phase[valid_mask]
        ripple_amp = ripple_amp[valid_mask]
        if ripple_amp.size < 10:
            return np.nan

        sin_phi = np.sin(sw_phase)
        cos_phi = np.cos(sw_phase)

        r_s, _ = pearsonr(ripple_amp, sin_phi)
        r_c, _ = pearsonr(ripple_amp, cos_phi)
        r_sc, _ = pearsonr(sin_phi, cos_phi)

        denominator = 1 - r_sc**2
        if denominator <= 0:
            return 0.0  # Safely assume no meaningful correlation

        numerator = r_c**2 + r_s**2 - 2 * r_c * r_s * r_sc
        r_cl = numerator / denominator
        r_cl = max(0, r_cl)

        return np.sqrt(r_cl)

    def select_sharpwave_channel(
            self,
            ca1_lfp,
            lfp_time_index,
            ca1_chan_ids,
            peak_ripple_chan_id,
            channel_positions,
            ripple_filtered,
            config=None,
            filter_path=None,
            running_exclusion_periods=None,
            selection_metric='modulation_index'):
        """
        Selects the optimal sharp wave channel below the ripple channel based on 
        phase-amplitude coupling and return metadata for all evaluated channels.

        This function aims to find the channel electrode that best captures the 
        phase-amplitude coupling with the ripple activity. It calculates metrics 
        for all CA1 channels below the reference channel and returns the best 
        channel ID, its LFP, and the metadata for all evaluated channels.

        Parameters:
        ----------
        ca1_lfp : np.ndarray
            LFP data array containing signals from channels located in the CA1 region.
            Expected shape: (time, channels).
        lfp_time_index : np.ndarray
            1D array of timestamps (in seconds) corresponding to the rows of `ca1_lfp`.
        ca1_chan_ids : list of int
            List of unique channel identifiers corresponding to the columns in `ca1_lfp`.
        peak_ripple_chan_id : int
            The channel ID previously identified as having the strongest ripple power.
            This serves as the reference channel.
        channel_positions : pd.Series
            A pandas Series where the index contains channel IDs (including those in 
            `ca1_chan_ids` and potentially others) and the values are the vertical
            positions (depths) of those channels on the probe. Typically, larger
            values mean deeper positions.
        ripple_filtered : np.ndarray
            The ripple-band filtered signal (e.g., 150-250 Hz) from the 
            `peak_ripple_chan_id`. Shape: (time,)
        config : dict, optional
            A configuration dictionary potentially containing parameters like 
            filter paths (`config['filter_paths']['sharpwave_filter']`), 
            analysis parameters (`config['analysis_params']` like thresholds, 
            time buffers, MI bins).
        filter_path : str, optional
            Direct path to the sharp wave filter coefficients (.mat file). Overrides
            path found in config.
        running_exclusion_periods : list of tuple, optional
            A list where each tuple contains (start_time, end_time) in seconds, 
            indicating periods (e.g., due to animal movement) that should be excluded 
            from the phase-amplitude coupling analysis. Overrides any found in config.
        selection_metric : {'modulation_index', 'circular_linear'}, optional
            Specifies the metric used to determine the "best" sharp wave channel from 
            the candidates below the ripple channel. Overrides config.
            - 'modulation_index': Uses the Tort Modulation Index (MI).
            - 'circular_linear': Uses the circular-linear correlation.
            Default is 'modulation_index'.

        Returns:
        -------
        tuple
            - best_sw_channel_id (int or None): The channel ID selected as the best 
              sharp wave channel. None if no suitable channel is found below the 
              reference or if errors occur.
            - best_sw_channel_lfp (np.ndarray or None): The raw LFP signal (1D array)
              from the `best_sw_channel_id`. None if no channel is selected.
            Note: This method now directly populates the 'sharp_wave_band' section of
            `self.channel_selection_metadata_dict` instead of returning separate metadata.

        Raises
        ------
        ValueError
            If no suitable channel is found below the reference channel or if errors occur.
        """
        # Get the filter - implementers must override this if filter_path is None
        filter_path = self.sw_component_filter_path
                       
        filter_data = np.load(filter_path)
        sharpwave_filter = filter_data['sharpwave_componenet_8to40band_1500hz_band']

        mask = (lfp_time_index > 3.5) & (lfp_time_index < lfp_time_index[-1] - 3.5)
        if running_exclusion_periods:
            for start, end in running_exclusion_periods:
                mask &= ~((lfp_time_index >= start) & (lfp_time_index <= end))

        ripple_an = hilbert(ripple_filtered)
        ripple_phase = np.angle(ripple_an)
        ripple_amp = np.abs(ripple_an)
        ripple_power = ripple_amp ** 2
        ripple_power_z = (ripple_power - ripple_power[mask].mean()) / ripple_power[mask].std()

        ref_depth = channel_positions.loc[peak_ripple_chan_id]
        below_ids = channel_positions[channel_positions > ref_depth].index
        id_to_idx = {cid: i for i, cid in enumerate(ca1_chan_ids)}
        below_idx = [id_to_idx[cid] for cid in below_ids if cid in id_to_idx]

        # Create dictionary to store detailed results for selection logic
        results = {}
        
        # Initialize lists in the target metadata dictionary
        self.channel_selection_metadata_dict['sharp_wave_band']['channel_ids'] = []
        self.channel_selection_metadata_dict['sharp_wave_band']['depths'] = []
        self.channel_selection_metadata_dict['sharp_wave_band']['net_sw_power'] = []
        self.channel_selection_metadata_dict['sharp_wave_band']['modulation_index'] = []
        self.channel_selection_metadata_dict['sharp_wave_band']['circular_linear_corrs'] = []

        for cid, idx in zip(below_ids, below_idx):
            sw_filt = fftconvolve(ca1_lfp[:, idx], sharpwave_filter, mode='same')
            sw_an = hilbert(sw_filt)
            sw_phase = np.angle(sw_an)
            sw_power = np.abs(sw_an) ** 2
            sw_power_z = (sw_power - sw_power[mask].mean()) / sw_power[mask].std()

            high_mask = (ripple_power_z > 1) & (sw_power_z > 1) & mask

            modulation_index = self._compute_modulation_index(sw_phase, ripple_amp, high_mask)
            circular_linear_corr = self._compute_circular_linear_corr(sw_phase, ripple_amp, high_mask)
            
            # Calculate net SW power across whole recording
            net_sw_power_val = np.sum(sw_power)

            # Store detailed results for selection logic
            results[cid] = {
                'modulation_index': modulation_index,
                'circular_linear_corr': circular_linear_corr,
                'idx': idx,
                'sw_power_z': sw_power_z
            }
            
            # Append results to the correct metadata dictionary
            self.channel_selection_metadata_dict['sharp_wave_band']['channel_ids'].append(int(cid))
            self.channel_selection_metadata_dict['sharp_wave_band']['depths'].append(float(channel_positions.loc[cid]))
            self.channel_selection_metadata_dict['sharp_wave_band']['net_sw_power'].append(float(net_sw_power_val))
            self.channel_selection_metadata_dict['sharp_wave_band']['modulation_index'].append(float(modulation_index) if not np.isnan(modulation_index) else None)
            self.channel_selection_metadata_dict['sharp_wave_band']['circular_linear_corrs'].append(float(circular_linear_corr) if not np.isnan(circular_linear_corr) else None)

        # Select best channel
        best_cid = max(
            results,
            key=lambda k: results[k][selection_metric] if not np.isnan(results[k][selection_metric]) else -np.inf
        )

        best_idx = results[best_cid]['idx']
        best_lfp = ca1_lfp[:, best_idx]
        
        # Store selection info in the metadata dictionary
        self.channel_selection_metadata_dict['sharp_wave_band']['selected_channel_id'] = int(best_cid)
        self.channel_selection_metadata_dict['sharp_wave_band']['selection_method'] = selection_metric
        
        # Return the best channel ID and its raw LFP
        return best_cid, best_lfp

    def select_ripple_channel(self, ca1_lfp, ca1_chan_ids, channel_positions, ripple_filter_func, config=None):
        """Select the putative pyramidal layer (ripple band) channel for a given probe.
        
        Parameters
        ----------
        ca1_lfp : np.ndarray
            LFP data array containing signals from channels located in the CA1 region.
            Expected shape: (time, channels).
        ca1_chan_ids : list of int
            List of unique channel identifiers corresponding to the columns in `ca1_lfp`.
        channel_positions : pd.Series
            A pandas Series where the index contains channel IDs (including those in 
            `ca1_chan_ids` and potentially others) and the values are the vertical
            positions (depths) of those channels on the probe.
        ripple_filter_func : callable
            Function to filter the LFP data into the ripple band.
        config : dict, optional
            Configuration dictionary (not used in this implementation).
        
        Returns
        -------
        tuple
            - peak_chan_id (int): The channel ID selected as the best ripple channel.
            - peak_ripple_band_lfp (np.ndarray): The ripple-band filtered signal from the selected channel.
            - peak_ripple_chan_lfp (np.ndarray): The raw LFP signal from the selected channel.
        """
        # Initialize lists to store metrics for each channel
        channel_metrics = []
        
        # Process each CA1 channel
        for chan_idx, chan_id in enumerate(ca1_chan_ids):
            # Get raw LFP for this channel
            chan_lfp = ca1_lfp[:, chan_idx]
            
            # Filter to ripple band
            ripple_band = ripple_filter_func(chan_lfp[:, None])
            ripple_band = ripple_band.flatten()  # Remove extra dimension
            
            # Calculate power and skewness
            ripple_power = np.abs(hilbert(ripple_band)) ** 2
            net_power = np.sum(ripple_power)
            skewness = skew(ripple_power)
            
            # Store metrics
            channel_metrics.append({
                'channel_id': chan_id,
                'depth': channel_positions.loc[chan_id],
                'net_power': net_power,
                'skewness': skewness,
                'raw_lfp': chan_lfp,
                'ripple_band': ripple_band
            })
            
            # Update metadata dictionary
            self.channel_selection_metadata_dict['ripple_band']['channel_ids'].append(int(chan_id))
            self.channel_selection_metadata_dict['ripple_band']['depths'].append(float(channel_positions.loc[chan_id]))
            self.channel_selection_metadata_dict['ripple_band']['skewness'].append(float(skewness))
            self.channel_selection_metadata_dict['ripple_band']['net_power'].append(float(net_power))
            
        # Convert to DataFrame for easier selection
        metrics_df = pd.DataFrame(channel_metrics)
        
        # Select channel with highest net power
        best_channel = metrics_df.loc[metrics_df['net_power'].idxmax()]
            
        # Update metadata with selection info
        self.channel_selection_metadata_dict['ripple_band']['selected_channel_id'] = int(best_channel['channel_id'])
        self.channel_selection_metadata_dict['ripple_band']['selection_method'] = 'max_power'
        
        return (
            best_channel['channel_id'],
            best_channel['ripple_band'],
            best_channel['raw_lfp']
        )

    def global_events_probe_info(self):
        """
        Get probe-level information needed for global SWR detection.
        This is a base implementation that should be overridden by each dataset's loader.
        
        Returns
        -------
        dict
            Dictionary mapping probe IDs to probe information dictionaries
        """
        raise NotImplementedError("Each loader must implement global_events_probe_info")

    def get_metadata_for_probe(self, probe_id, config=None):
        """
        Generates metadata for a single specified probe, focusing on unit counts
        and CA1 properties.

        Parameters
        ----------
        probe_id : str or int
            The unique identifier for the probe being processed.
        config : dict, optional
            Configuration dictionary, potentially needed for specific dataset logic.

        Returns
        -------
        dict
            A dictionary containing standardized probe metadata:
            - 'probe_id': Copied from input.
            - 'has_ca1_channels': bool
            - 'ca1_channel_count': int
            - 'ca1_span_microns': float
            - 'total_unit_count': int
            - 'good_unit_count': int
            - 'ca1_total_unit_count': int
            - 'ca1_good_unit_count': int

        Raises
        ------
        NotImplementedError
            This base method must be implemented by each dataset-specific loader subclass.
        """
        raise NotImplementedError("Subclasses must implement get_metadata_for_probe method")

    def extending_edeno_event_stats(self, events_df, time_values, ripple_filtered):
        """
        Extend event statistics with power-based metrics and envelope 90th percentile.
        Primarily computes power metrics from the ripple band LFP data, but also includes
        the 90th percentile of the smoothed envelope.
        
        Parameters
        ----------
        events_df : pd.DataFrame
            Must contain 'start_time' and 'end_time' columns (seconds).
        time_values : np.ndarray
            Time vector (seconds) aligned to the LFP traces.
        ripple_filtered : np.ndarray
            Ripple-band filtered signal (same length as time_values).
            
        Returns
        -------
        pd.DataFrame
            Copy of events_df with new power-based metrics and envelope percentile columns.
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
        
        # Compute envelope and smooth it
        envelope = np.abs(signal.hilbert(ripple_filtered))
        smoothing_sigma = 0.004  # 4ms smoothing
        envelope_smoothed = gaussian_smooth(envelope, sigma=smoothing_sigma, sampling_frequency=1500.0)  # 1500 Hz sampling rate
        envelope_smoothed_z = stats.zscore(envelope_smoothed)
        # Compute power from smoothed envelope
        ripple_power = envelope_smoothed ** 2
        
        # Z-score the power using global statistics
        power_z = (ripple_power - np.mean(ripple_power)) / np.std(ripple_power)
        
        # Initialize lists for metrics
        power_peak_times = []
        power_max_zscores = []
        power_median_zscores = []
        power_mean_zscores = []
        power_min_zscores = []
        power_90th_percentiles = []
        envelope_90th_percentiles = []
        envelope_peak_times = []

        # Process each event
        for _, event in events_df.iterrows():
            # Get event window
            mask = (time_values >= event['start_time']) & (time_values <= event['end_time'])
            event_power_z = power_z[mask]
            event_envelope = envelope_smoothed[mask]
            event_times = time_values[mask]
            event_envelope_smoothed_z = envelope_smoothed_z[mask]
            
            # Append metrics to lists
            power_peak_times.append(event_times[np.argmax(event_power_z)])
            power_max_zscores.append(np.max(event_power_z))
            power_median_zscores.append(np.median(event_power_z))
            power_mean_zscores.append(np.mean(event_power_z))
            power_min_zscores.append(np.min(event_power_z))
            power_90th_percentiles.append(np.percentile(event_power_z, 90))
            envelope_90th_percentiles.append(np.percentile(event_envelope, 90))
            envelope_peak_times.append(event_times[np.argmax(event_envelope_smoothed_z)])

        # ---------- add to dataframe -------------------------------------------------
        events_df['power_peak_time'] = power_peak_times
        events_df['power_max_zscore'] = power_max_zscores
        events_df['power_median_zscore'] = power_median_zscores
        events_df['power_mean_zscore'] = power_mean_zscores
        events_df['power_min_zscore'] = power_min_zscores
        events_df['power_90th_percentile'] = power_90th_percentiles
        events_df['envelope_90th_percentile'] = envelope_90th_percentiles
        events_df['envelope_peak_time'] = envelope_peak_times
        
        return events_df

    def incorporate_sharp_wave_component_info(
            self,
            events_df: pd.DataFrame,
            time_values: np.ndarray,
            ripple_filtered: np.ndarray,
            sharp_wave_lfp: np.ndarray,
            sharpwave_filter: np.ndarray,
            n_bins: int = 18
    ) -> pd.DataFrame:
        """
        Add sharp‑wave metrics to each ripple event.

        For every ripple (row) in *events_df* the function computes:
        1. sw_exceeds_threshold      – True if sharp‑wave power (z‑score) > +1 SD at any point.
        2. sw_peak_power             – Median sharp‑wave z‑score of samples ≥ 90th percentile
                                       (robust peak estimate).
        3. sw_peak_time              – Time (s) of the absolute peak sharp‑wave power.
        4. sw_ripple_plv             – Phase‑locking value (PLV) between ripple and SW phase.
        5. sw_ripple_mi              – Modulation index (Tort MI) between SW phase and ripple amp.
        6. sw_ripple_clcorr          – Circular‑linear correlation between SW phase and ripple amp.

        Parameters
        ----------
        events_df : pd.DataFrame
            Must contain 'start_time' and 'end_time' columns (seconds).
        time_values : np.ndarray
            Time vector (seconds) aligned to the LFP traces.
        ripple_filtered : np.ndarray
            Ripple‑band filtered signal (same length as time_values).
        sharp_wave_lfp : np.ndarray
            Raw LFP from the selected sharp‑wave channel.
        sharpwave_filter : np.ndarray
            FIR/IIR kernel to isolate the 8–40 Hz sharp‑wave component.
        n_bins : int, optional
            Number of phase bins for MI calculation (default 18).

        Returns
        -------
        pd.DataFrame
            Copy of *events_df* with the six new columns listed above.
        """
        # ---------- analytic signals -------------------------------------------------
        sw_filtered = signal.fftconvolve(sharp_wave_lfp, sharpwave_filter, mode="same")
        sw_an       = signal.hilbert(sw_filtered)
        sw_phase    = np.angle(sw_an)
        sw_power    = np.abs(sw_an) ** 2
        sw_power_z  = stats.zscore(sw_power)

        ripple_an   = signal.hilbert(ripple_filtered)
        ripple_phase = np.angle(ripple_an)
        ripple_amp   = np.abs(ripple_an)

        # ---------- helpers ----------------------------------------------------------
        def _plv(mask: np.ndarray) -> float:
            """Phase‑locking value."""
            if mask.sum() < 10:
                return np.nan
            dphi = ripple_phase[mask] - sw_phase[mask]
            return np.abs(np.mean(np.exp(1j * dphi)))

        def _mi(mask: np.ndarray) -> float:
            """Tort modulation index."""
            if mask.sum() < 10:
                return np.nan
            phi = sw_phase[mask]
            amp = ripple_amp[mask]
            bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            digitized = np.digitize(phi, bins) - 1
            amp_by_bin = np.array([
                amp[digitized == j].mean() if np.any(digitized == j) else 0
                for j in range(n_bins)
            ])
            if amp_by_bin.sum() == 0:
                return np.nan
            P = amp_by_bin / amp_by_bin.sum()
            H = -np.sum(P * np.log(P + 1e-10))
            return (np.log(n_bins) - H) / np.log(n_bins)

        def _clcorr(mask: np.ndarray) -> float:
            """Circular‑linear correlation (Zar-style, bounded between 0 and 1)."""
            if mask.sum() < 10:
                return np.nan

            phi = sw_phase[mask]
            amp = ripple_amp[mask]

            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            r_s, _ = pearsonr(amp, sin_phi)
            r_c, _ = pearsonr(amp, cos_phi)
            r_sc, _ = pearsonr(sin_phi, cos_phi)

            denominator = 1 - r_sc**2
            if denominator <= 1e-8 or not np.isfinite(denominator):
                return 0.0  # No meaningful circular variation

            numerator = r_c**2 + r_s**2 - 2 * r_c * r_s * r_sc
            r_cl = numerator / denominator
            r_cl = max(0, r_cl)

            return np.sqrt(r_cl)

        # ---------- iterate events ---------------------------------------------------
        out_df = events_df.copy()
        exceeds, peak_pwr, peak_time, plv, mi, clcorr = ([] for _ in range(6))

        for _, ev in events_df.iterrows():
            mask = (time_values >= ev['start_time']) & (time_values <= ev['end_time'])
            sw_z = sw_power_z[mask]

            # 1. threshold flag
            exceeds.append(bool(np.any(sw_z > 1)))

            # 2‑3. robust peak power & its time
            if sw_z.size:
                q90 = np.quantile(sw_z, 0.9)
                peak_pwr.append(np.median(sw_z[sw_z >= q90]))
                peak_idx = np.argmax(sw_z)
                peak_time.append(time_values[mask][peak_idx])
            else:
                peak_pwr.append(np.nan)
                peak_time.append(np.nan)

            # 4‑6. coupling metrics
            plv.append(_plv(mask))
            mi.append(_mi(mask))
            clcorr.append(_clcorr(mask))

        # ---------- add to dataframe -------------------------------------------------
        out_df['sw_exceeds_threshold'] = exceeds
        out_df['sw_peak_power']        = peak_pwr
        out_df['sw_peak_time']         = peak_time
        out_df['sw_ripple_plv']        = plv
        out_df['sw_ripple_mi']         = mi
        out_df['sw_ripple_clcorr']     = clcorr

        return out_df


    def events_dict_from_files(self, session_subfolder, session_id, logger):
        """Load probe events from existing files in the session directory.
        
        Args:
            session_subfolder (str): Path to the session directory
            session_id (str): Session ID for logging
            logger: Logger instance for logging messages
            
        Returns:
            dict: Dictionary mapping probe IDs to their event DataFrames
        """
        probe_events_dict = {}
        # Find all probe event files in the session directory
        event_files = [f for f in os.listdir(session_subfolder) if f.endswith('_putative_swr_events.csv.gz')]
        for event_file in event_files:
            # Extract probe_id from filename: probe_<probe_id>_channel_...
            match = re.match(r'probe_(.*?)_channel_.*_putative_swr_events\.csv\.gz', event_file)
            if match:
                probe_id = match.group(1)
            else:
                logger.warning(f"Session {session_id}: Could not extract probe_id from {event_file}")
                continue
            full_path = os.path.join(session_subfolder, event_file)
            try:
                events_df = pd.read_csv(full_path, compression='gzip', index_col=0)
                probe_events_dict[str(probe_id)] = events_df
                logger.info(f"✓ Successfully loaded events for probe {probe_id} ({len(events_df)} events)")
            except Exception as e:
                logger.error(f"Session {session_id}: Error loading events for probe {probe_id}: {str(e)}")
                continue
        logger.info(f"\nLoaded events for {len(probe_events_dict)} probes")
        return probe_events_dict

    def save_settings_metadata(self, output_path, config, logger):
        """
        Save the detector settings metadata as a compressed JSON file.
        
        Args:
            output_path: Full path where to save the settings file
            config: The configuration dictionary containing all settings
            logger: Logger instance to use for logging
        """
        try:
            # Extract relevant settings
            settings = {
                "run_name": config["run_details"]["run_name"],
                "thresholds": {
                    "gamma_event_thresh": config["artifact_detection"]["gamma_event_thresh"],
                    "ripple_band_threshold": config["ripple_detection"]["ripple_band_threshold"],
                    "movement_artifact_ripple_band_threshold": config["ripple_detection"]["movement_artifact_ripple_band_threshold"],
                    "merge_events_offset": 0.025  # Hardcoded in config
                },
                "global_swr_detection": config["global_swr"],
                "dataset": config["run_details"]["dataset_to_process"],
                "sampling_rates": config["sampling_rates"]
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as compressed JSON
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)
                
            logger.info(f"Saved settings metadata to {output_path}")
        except Exception as e:
            logger.error(f"Error saving settings metadata: {str(e)}")
            raise
    
    # Abstract method stubs that must be implemented by subclasses
    # Due to dependency conflicts in the apis, differences in metadata labels or conventions
    # the code is sufficiently different to implement all of these steps
    # that they have implmented in their own seperate classes, see the _loader.py files
    # for the code there
    def cleanup(self):
        """Clean up resources."""
        raise NotImplementedError("Subclasses must implement cleanup method")
    
    def set_up(self):
        """Setup the loader and initialize connections."""
        raise NotImplementedError("Subclasses must implement set_up method")
        
    def has_ca1_channels(self):
        """Check if the session has CA1 channels."""
        raise NotImplementedError("Subclasses must implement has_ca1_channels method")
        
    def get_probes_with_ca1(self):
        """Get list of probes with CA1 channels."""
        raise NotImplementedError("Subclasses must implement get_probes_with_ca1 method")
        
    def process_probe(self, probe_id, filter_ripple_band_func=None):
        """Process a single probe."""
        raise NotImplementedError("Subclasses must implement process_probe method")  

    def get_metadata_for_probe(self, probe_id, config=None):
        """Get metadata for a specific probe."""
        raise NotImplementedError("Subclasses must implement get_metadata_for_probe")


# ===================================
#  PROBE LEVEL EVENT DETECTOR
#  -Channel selection, artifact, and putative ripple event detection
#  -intentionally overly permissive settings are used here
#  -most computationally expensive
# ===================================



# Assuming you have your signal_array, b, and a defined as before
def finitimpresp_filter_for_LFP(
    LFP_array, samplingfreq, lowcut=1, highcut=250, filter_order=101
):
    """
    Filter the LFP array using a finite impulse response filter.

    Parameters
    ----------
    LFP_array : np.array
        The LFP array.
    samplingfreq : float
        The sampling frequency of the LFP array.
    lowcut : float
        The lowcut frequency.
    highcut : float
        The highcut frequency.
    filter_order : int
        The filter order.

    Returns
    -------
    np.array
        The filtered LFP array.
    """
    nyquist = 0.5 * samplingfreq

    # Design the FIR bandpass filter using scipy.signal.firwin
    fir_coeff = signal.firwin(
        filter_order,
        [lowcut / nyquist, highcut / nyquist],
        pass_zero=False,
        fs=samplingfreq,
    )

    # Apply the FIR filter to your signal_array
    # filtered_signal = signal.convolve(LFP_array, fir_coeff, mode='same', method='auto')
    filtered_signal = signal.lfilter(fir_coeff, 1.0, LFP_array, axis=0)
    return filtered_signal


def event_boundary_detector(
    time,
    five_to_fourty_band_power_df,
    envelope=True,
    minimum_duration=0.02,
    maximum_duration=0.4,
    threshold_sd=2.5,
    envelope_threshold_sd=1,
):
    """
    For detecting gamma events.
    Parameters
    ----------
    time : np.array
        The time values for the signal.
    five_to_fourty_band_power_df : np.array
        The power of the signal in the 5-40 Hz band.
    envelope : bool
        Whether to use the envelope threshold.
    minimum_duration : float
        The minimum duration of an event.
    maximum_duration : float
        The maximum duration of an event.
    threshold_sd : float
        The threshold in standard deviations.
    envelope_threshold_sd : float
        The envelope threshold in standard deviations.

    Returns
    -------
    pd.DataFrame
        A dataframe with the start and end times of the events.

    """
    # make df to fill
    row_of_info = {
        "start_time": [],
        "end_time": [],
        "duration": [],
    }

    # sharp_wave_events_df = pd.DataFrame()
    # scored_wave_power = stats.zscore(five_to_fourty_band_df)

    # compute our power threshold
    # wave_band_sd_thresh = np.std(five_to_fourty_band_df)*threshold_sd
    five_to_fourty_band_power_df = stats.zscore(five_to_fourty_band_power_df)
    past_thresh = five_to_fourty_band_power_df >= threshold_sd

    # now we expand the sections that are past thresh up to the points that
    # are past the envelope thresh, so not all sections above envelope thresh are true
    # but those sections which alse contain a region past the detection threshold are included
    def expand_sections(z_scores, boolean_array, thresh):
        # Find indices where boolean_array is True
        true_indices = np.where(boolean_array)[0]

        # Initialize an array to keep track of expanded sections
        expanded_sections = np.zeros_like(z_scores, dtype=bool)

        # Iterate over true_indices and expand sections
        for index in true_indices:
            # Find the start and end of the current section
            start = index
            end = index

            # Expand section to the left (while meeting conditions)
            while start > 0 and z_scores[start - 1] > thresh:
                start -= 1

            # Expand section to the right (while meeting conditions)
            while end < len(z_scores) - 1 and z_scores[end + 1] > thresh:
                end += 1

            # Check if the expanded section contains a point above envelope_threshold_sd in z_scores
            if any(z_scores[start : end + 1] > thresh):
                expanded_sections[start : end + 1] = True

        # Update the boolean_array based on expanded_sections
        boolean_array = boolean_array | expanded_sections

        return boolean_array

    if envelope == True:
        past_thresh = expand_sections(
            z_scores=five_to_fourty_band_power_df,
            boolean_array=past_thresh,
            thresh=envelope_threshold_sd,
        )

    # Find the indices where consecutive True values start
    starts = np.where(past_thresh & ~np.roll(past_thresh, 1))[0]
    row_of_info["start_time"] = time[starts]
    # Find the indices where consecutive True values end
    ends = np.where(past_thresh & ~np.roll(past_thresh, -1))[0]
    row_of_info["end_time"] = time[ends]

    row_of_info["duration"] = [
        row_of_info["end_time"][i] - row_of_info["start_time"][i]
        for i in range(0, len(row_of_info["start_time"]))
    ]

    # turn the dictionary into adataframe
    sharp_wave_events_df = pd.DataFrame(row_of_info)

    # filter for the duration range we want
    in_duration_range = (sharp_wave_events_df.duration > minimum_duration) & (
        sharp_wave_events_df.duration < maximum_duration
    )
    sharp_wave_events_df = sharp_wave_events_df[in_duration_range]

    return sharp_wave_events_df


def event_boundary_times(time, past_thresh):
    """
    Finds the times of a vector of true statements and returns values from another
    array representing the times

    Parameters
    ----------
    time : np.array
        The time values for the signal.
    past_thresh : np.array
        The boolean array of the signal.

    Returns
    -------
    pd.DataFrame
        A dataframe with the start and end times of the events.
    """
    # Find the indices where consecutive True values start
    starts = np.where(past_thresh & ~np.roll(past_thresh, 1))[0]
    row_of_info["start_time"] = time[starts]
    # Find the indices where consecutive True values end
    ends = np.where(past_thresh & ~np.roll(past_thresh, -1))[0]
    row_of_info["end_time"] = time[ends]

    row_of_info["duration"] = [
        row_of_info["end_time"][i] - row_of_info["start_time"][i]
        for i in range(0, len(row_of_info["start_time"]))
    ]

    # turn the dictionary into adataframe
    events_df = pd.DataFrame(row_of_info)

    return events_df


def peaks_time_of_events(events, time_values, signal_values):
    """
    Computes the times when ripple power peaks in the events

    Parameters
    ----------
    events : pd.DataFrame
        The events dataframe.
    time_values : np.array
        The time values for the signal.
    signal_values : np.array
        The signal values for the signal.

    Returns
    -------
    np.array
        The times of the peaks in the ripple power signal.
    """

    # looks for the peaks in the ripple power signal, value of zscored raw lfp peak and returns time of peak
    signal_values_zscore = stats.zscore(signal_values)
    peak_times = []
    for start, end in zip(events["start_time"], events["end_time"]):
        window_idx = (time_values >= start) & (time_values <= end)
        ripple_lfp_zscore_signal = signal_values_zscore[window_idx]
        maxpoint = np.argmax(ripple_lfp_zscore_signal)
        rippletimepoints = time_values[window_idx]
        peak_times.append(rippletimepoints[maxpoint])
    return np.array(peak_times)


def resample_signal(signal, times, new_rate):
    """
    Resample a 2D signal array to a new sampling rate.

    Parameters:
    signal (np.array): 2D array where each column is a source and each row is a time point.
    times (np.array): 1D array of times corresponding to the rows of the signal array.
    new_rate (float): The new sampling rate in Hz.

    Returns:
    new_signal (np.array): The resampled signal array.
    new_times (np.array): The times corresponding to the rows of the new signal array.
    """
    nsamples_new = int(len(times) * new_rate / (len(times) / times[-1]))
    new_times = np.linspace(times[0], times[-1], nsamples_new)
    new_signal = np.zeros((signal.shape[0], nsamples_new))

    for i in range(signal.shape[0]):
        interp_func = interpolate.interp1d(
            times, signal[i, :], bounds_error=False, fill_value="extrapolate"
        )
        new_signal[i, :] = interp_func(new_times)

    return new_signal, new_times


def check_gamma_overlap(events_df, gamma_df):
    """
    Uses vectorized operations to find overlaps in gamma bursts and putative events.
    """
    annotated = []
    
    for _, event in events_df.iterrows():
        event_start, event_end = event['start_time'], event['end_time']
        event_duration = event_end - event_start
        
        # Check overlap using interval logic
        overlaps = ((gamma_df['start_time'] <= event_end) & 
                    (gamma_df['end_time'] >= event_start))
        
        has_overlap = overlaps.any()
        
        if has_overlap:
            # Calculate overlapping segments
            overlap_segments = gamma_df[overlaps].copy()
            # Clip each segment to the event boundaries
            overlap_segments['overlap_start'] = np.maximum(overlap_segments['start_time'], event_start)
            overlap_segments['overlap_end'] = np.minimum(overlap_segments['end_time'], event_end)
            overlap_segments['overlap_duration'] = overlap_segments['overlap_end'] - overlap_segments['overlap_start']
            
            # Calculate total overlap duration (accounting for potential overlapping segments)
            # Sort segments by start time
            overlap_segments = overlap_segments.sort_values('overlap_start')
            
            # Merge overlapping segments
            merged_segments = []
            for _, segment in overlap_segments.iterrows():
                if not merged_segments or merged_segments[-1][1] < segment['overlap_start']:
                    merged_segments.append([segment['overlap_start'], segment['overlap_end']])
                else:
                    merged_segments[-1][1] = max(merged_segments[-1][1], segment['overlap_end'])
            
            total_overlap = sum(end - start for start, end in merged_segments)
            overlap_pct = 100.0 * total_overlap / event_duration
        else:
            overlap_pct = 0.0
        
        annotated.append({
            'overlaps_with_gamma': has_overlap,
            'gamma_overlap_percent': overlap_pct
        })
    
    return pd.concat([events_df.reset_index(drop=True), pd.DataFrame(annotated)], axis=1)

def check_movement_overlap(events_df, move_df1, move_df2):
    """
    Uses vectorized operations to find overlap in putative events and movement artifacts.
    """
    annotated = []
    
    for _, event in events_df.iterrows():
        event_start, event_end = event['start_time'], event['end_time']
        event_duration = event_end - event_start
        
        # Check overlap using interval logic for both movement dataframes
        overlaps1 = ((move_df1['start_time'] <= event_end) & 
                     (move_df1['end_time'] >= event_start))
        overlaps2 = ((move_df2['start_time'] <= event_end) & 
                     (move_df2['end_time'] >= event_start))
        
        has_overlap1 = overlaps1.any()
        has_overlap2 = overlaps2.any()
        both_overlap = has_overlap1 and has_overlap2
        
        # Calculate overlap percentage with either channel
        segments_list = []  # Use a clearer name to avoid confusion
        
        # Process overlaps from first movement dataframe
        if has_overlap1:
            df1_overlaps = move_df1[overlaps1].copy()
            df1_overlaps['overlap_start'] = np.maximum(df1_overlaps['start_time'], event_start)
            df1_overlaps['overlap_end'] = np.minimum(df1_overlaps['end_time'], event_end)
            for _, row in df1_overlaps.iterrows():
                segments_list.append([row['overlap_start'], row['overlap_end']])
        
        # Process overlaps from second movement dataframe
        if has_overlap2:
            df2_overlaps = move_df2[overlaps2].copy()
            df2_overlaps['overlap_start'] = np.maximum(df2_overlaps['start_time'], event_start)
            df2_overlaps['overlap_end'] = np.minimum(df2_overlaps['end_time'], event_end)
            for _, row in df2_overlaps.iterrows():
                segments_list.append([row['overlap_start'], row['overlap_end']])
        
        if segments_list:
            # Sort segments by start time using key function
            segments_list.sort(key=lambda x: x[0])
            
            # Merge overlapping segments
            merged_segments = []
            for start, end in segments_list:
                if not merged_segments or merged_segments[-1][1] < start:
                    merged_segments.append([start, end])
                else:
                    merged_segments[-1][1] = max(merged_segments[-1][1], end)
            
            total_overlap = sum(end - start for start, end in merged_segments)
            overlap_pct = 100.0 * total_overlap / event_duration
        else:
            overlap_pct = 0.0
        
        annotated.append({
            'overlaps_with_movement': both_overlap,
            'movement_overlap_percent': overlap_pct
        })
    
    return pd.concat([events_df.reset_index(drop=True), pd.DataFrame(annotated)], axis=1)



# ===================================
#  GLOBAL EVENT DETECTOR
#  -Probe level events are merged into 
# ===================================

def add_participating_probes(global_events, probe_events_dict, offset=0.02):
    """
    Add information about participating probes to global events.
    
    Parameters
    ----------
    global_events : pandas.DataFrame
        Global events DataFrame
    probe_events_dict : dict
        Dictionary mapping probe IDs to event DataFrames
    offset : float, optional
        Time window (in seconds) to consider for participation
        
    Returns
    -------
    pandas.DataFrame
        Global events DataFrame with probe information
    """
    # Initialize columns
    global_events['participating_probes'] = [[] for _ in range(len(global_events))]
    global_events['peak_times'] = [[] for _ in range(len(global_events))]
    global_events['peak_powers'] = [[] for _ in range(len(global_events))]
    global_events['probe_event_file_index'] = [[] for _ in range(len(global_events))]
    
    # Check each probe against each global event
    for probe_id, probe_df in probe_events_dict.items():
        for i, event in global_events.iterrows():
            # Check if any probe event overlaps with this global event
            overlap = ((probe_df['start_time'] <= event['end_time'] + offset) & 
                      (probe_df['end_time'] >= event['start_time'] - offset))
            
            if any(overlap):
                # Get peak time and power from the strongest overlapping event
                overlapping_events = probe_df[overlap]
                if not overlapping_events.empty:
                    max_idx = overlapping_events['power_max_zscore'].idxmax()
                    if pd.notna(max_idx):  # Check if max_idx is not NaN
                        # Add this probe to the participating probes
                        global_events.at[i, 'participating_probes'].append(probe_id)
                        global_events.at[i, 'probe_event_file_index'].append(max_idx)  # Store the probe event index
                        global_events.at[i, 'peak_times'].append(overlapping_events.loc[max_idx, 'power_peak_time'])
                        global_events.at[i, 'peak_powers'].append(overlapping_events.loc[max_idx, 'power_max_zscore'])
                else:
                    print("WARNING: No overlapping events found despite overlap=True")

    # Add count of participating probes
    global_events['probe_count'] = global_events['participating_probes'].apply(len)

    # Add global peak information
    global_events['global_peak_time'] = global_events.apply(
        lambda row: row['peak_times'][row['peak_powers'].index(max(row['peak_powers']))] 
        if row['peak_powers'] else np.nan, axis=1)
    
    global_events['global_peak_power'] = global_events['peak_powers'].apply(
        lambda x: max(x) if x else np.nan)
    
    global_events['peak_probe'] = global_events.apply(
        lambda row: row['participating_probes'][row['peak_powers'].index(max(row['peak_powers']))]
        if row['peak_powers'] else '', axis=1)
    
    return global_events

def merge_probe_events(probe_events_dict, merge_window, min_probe_count=1):
    """
    Identify potential multiprobe 'global' events by merging co-occurring events across probes.
    
    Parameters
    ----------
    probe_events_dict : dict
        Dictionary mapping probe IDs to event DataFrames
    merge_window : float, optional
        Time window (in seconds) to merge nearby events
    min_probe_count : int, optional
        Minimum number of probes required to define a global event
        
    Returns
    -------
    pandas.DataFrame
        Global events DataFrame
        
    Raises
    ------
    ValueError
        If no events are created or no probe events are provided
    """
    if not probe_events_dict:
        raise ValueError("No probe events provided")
    
    # Convert to list of DataFrames
    event_dfs = list(probe_events_dict.values())
    
    # Create initial intervals from probes
    all_intervals = []
    for df in event_dfs:
        intervals = [(row['start_time'], row['end_time'], i) 
                    for i, (_, row) in enumerate(df.iterrows())]
        all_intervals.extend(intervals)
    
    if not all_intervals:
        raise ValueError("No intervals found in any probe events")
    
    # Sort by start time
    all_intervals.sort(key=lambda x: x[0])
    
    # Merge overlapping intervals
    merged_intervals = []
    current = list(all_intervals[0])  # Convert tuple to list so we can modify it
    current_events = [current[2]]  # Track all events in this interval
    
    for interval in all_intervals[1:]:
        if interval[0] <= current[1] + merge_window:
            # Merge intervals
            current[1] = max(current[1], interval[1])  # Update end time
            current_events.append(interval[2])  # Add this event's index
        else:
            merged_intervals.append((current[0], current[1], current_events))  # Store all event indices
            current = list(interval)
            current_events = [current[2]]
    
    merged_intervals.append((current[0], current[1], current_events))  # Add the last interval
    
    # Create DataFrame for global events
    global_events = pd.DataFrame({
        'start_time': [interval[0] for interval in merged_intervals],
        'end_time': [interval[1] for interval in merged_intervals],
        'contributing_probe_events_idx': [interval[2] for interval in merged_intervals]  # Now contains list of all contributing events
    })
    
    global_events['duration'] = global_events['end_time'] - global_events['start_time']
    
    # Add information about which probes participate in each event
    global_events = add_participating_probes(global_events, probe_events_dict, merge_window)
    
    # Filter events based on minimum probe count
    global_events = global_events[global_events['probe_count'] >= min_probe_count]
    
    return global_events

def save_global_events(global_events, session_dir, session_id, label="global"):
    """
    Save global events to a CSV file.
    
    Parameters
    ----------
    global_events : pandas.DataFrame
        Global events DataFrame
    session_dir : str
        Path to session directory
    session_id : str
        Session ID
    label : str, optional
        Label to append to filename
        
    Returns
    -------
    str
        Path to saved file
    """
    filename = f"session_{session_id}_global_swrs_{label}.csv.gz"
    filepath = os.path.join(session_dir, filename)
    global_events.to_csv(filepath, index=False, compression='gzip')
    logging.info(f"Saved global events to {filepath}")
    return filepath

def create_global_swr_events(probe_events_dict, global_swr_config, probe_metadata_df, session_subfolder, session_id, logger):
    """
    Create global SWR events from probe-level events, filtering probes based on metadata.
    
    Parameters
    ----------
    probe_events_dict : dict
        Dictionary mapping probe IDs (str) to event DataFrames
    global_swr_config : dict
        Configuration parameters for global SWR detection (e.g., min_sw_power, 
        min_filtered_events, min_ca1_units, merge_window, min_probe_count, 
        global_rip_label)
    probe_metadata_df : pd.DataFrame
        DataFrame loaded from probe_metadata.csv.gz, containing columns like 
        'probe_id' (str), 'ca1_good_unit_count'.
    session_subfolder : str
        Path to session directory
    session_id : str
        Session ID
    logger : logging.Logger
        Logger instance to use for logging
        
    Returns
    -------
    pd.DataFrame or None
        Global events DataFrame, or None if no events pass filtering criteria.
    
    Raises
    ------
    ValueError
        If no probe events are provided or if required columns are missing from metadata.
    """
    if not probe_events_dict:  # Pythonic way to check for empty dict
        raise ValueError("probe_events_dict is empty - no probe events provided to create_global_swr_events")
    
    if not all(col in probe_metadata_df.columns for col in ['probe_id', 'ca1_good_unit_count']):
        raise ValueError("Probe metadata DataFrame is missing required columns ('probe_id', 'ca1_good_unit_count')")

    # Apply SW power filters to all collected events
    filtered_probe_events_dict = {}
    for probe_id_str, events in probe_events_dict.items():
        # Filter by SW power
        filtered_events = events[events['sw_peak_power'] >= global_swr_config.get('min_sw_power', -np.inf)]
        
        # Filter by event count
        if len(filtered_events) >= global_swr_config.get('min_filtered_events', 0):
            filtered_probe_events_dict[probe_id_str] = filtered_events
        else:
            logger.info(f"Session {session_id}: Probe {probe_id_str} excluded by event count ({len(filtered_events)} < {global_swr_config.get('min_filtered_events', 0)}).")

    if not filtered_probe_events_dict:
        logger.warning(f"Session {session_id}: No probes have enough events after SW power filtering (min_filtered_events={global_swr_config.get('min_filtered_events', 1)}).")
        return None # Changed from raise ValueError

    # Filter probes based on unit count using the metadata DataFrame
    unit_filtered_probe_events_dict = {}
    min_ca1_units_req = global_swr_config.get('min_ca1_units', 1) # Default min 1 unit
    
    for probe_id_str, events in filtered_probe_events_dict.items():
        # Find the row for this probe in the metadata DataFrame
        probe_row = probe_metadata_df[probe_metadata_df['probe_id'] == probe_id_str]
        
        if probe_row.empty:
            logger.warning(f"Session {session_id}: Probe {probe_id_str} not found in metadata CSV. Skipping for unit filtering.")
            continue
        
        # Get the unit count (handle potential multiple rows? Take first)
        ca1_good_units = probe_row['ca1_good_unit_count'].iloc[0]
            
        # Check if probe has enough CA1 units
        if ca1_good_units >= min_ca1_units_req:
            unit_filtered_probe_events_dict[probe_id_str] = events
        else:
             logger.info(f"Session {session_id}: Probe {probe_id_str} excluded by unit count ({ca1_good_units} < {min_ca1_units_req}).")

    if not unit_filtered_probe_events_dict:
        logger.warning(f"Session {session_id}: No probes meet the minimum CA1 unit count criteria (min_ca1_units={min_ca1_units_req}).")
        return None # Changed from raise ValueError

    # Create global events using the unit-filtered probes
    try:
        global_events = merge_probe_events(
            unit_filtered_probe_events_dict,
            merge_window=global_swr_config.get('merge_window'),
            min_probe_count=global_swr_config.get('min_probe_count', 1)
        )
    except ValueError as e_create:
        # merge_probe_events raises ValueError if no intervals or merged events found
        logger.warning(f"Session {session_id}: Error during global event creation: {e_create}")
        return None
    
    # Note: Saving is handled back in process_session now
    # save_global_events(
    #     global_events, 
    #     session_subfolder, 
    #     session_id, 
    #     label=swr_config.get('global_rip_label', 'global')
    # )
    
    return global_events

# --- Paste process_session here and modify --- 
def process_session(session_id, config):
    """
    Process a single session using parameters passed in config dictionary.
    
    Parameters
    ----------
    session_id : str
        The ID of the session to process
    config : dict
        Dictionary containing all necessary settings including:
        - paths: containing output directories
        - run_details: containing dataset info and run parameters
        - flags: containing processing options like save_lfp, overwrite_existing
        - ripple_detection: containing detection thresholds
        - filters: containing filter arrays and paths
    
    Returns
    -------
    None
        Results are saved to files in the specified output directory
    """
    # Set up logging
    logger = logging.getLogger(f"session_{session_id}")
    logger.setLevel(logging.INFO)
    
    # Log the session ID and dataset being processed
    logger.info(f"Processing session {session_id} for dataset {config['run_details']['dataset_to_process']}")
    
    # Extract necessary paths and settings from config
    swr_output_dir_path = config['paths']['swr_output_dir']
    lfp_output_dir_path = config['paths']['lfp_output_dir']
    dataset_to_process = config['run_details']['dataset_to_process']
    run_name = config['run_details']['run_name']
    
    # Extract processing flags
    save_lfp = config['flags']['save_lfp']
    save_channel_metadata = config['flags'].get('save_channel_metadata', False)
    overwrite_existing = config['flags'].get('overwrite_existing', False)
    cleanup_after = config['flags'].get('cleanup_cache', False)
    run_putative = config['flags'].get('run_putative', False)
    run_filter = config['flags'].get('run_filter', False)
    run_global = config['flags'].get('run_global', False)
    
    # Extract ripple detection settings
    ripple_band_threshold = config['ripple_detection']['ripple_band_threshold']
    movement_artifact_ripple_band_threshold = config['ripple_detection']['movement_artifact_ripple_band_threshold']
    
    # Extract artifact detection settings
    gamma_event_thresh = config['artifact_detection']['gamma_event_thresh']
    
    # Extract filters
    gamma_filter = config['filters']['gamma_filter']
    sharp_wave_component_path = config['filters']['sharp_wave_component_path']
    
    # Global SWR detection settings
    global_swr_config = config.get('global_swr', {})
    
    # Setup AWS client with extended timeout
    my_config = Config(connect_timeout=1200, read_timeout=1200)
    s3 = boto3.client('s3', config=my_config)
    
    # Initialize variables
    process_stage = f"Starting processing for session {session_id}"
    probe_id = None
    probe_id_log = "Not Loaded Yet"
    probe_events_dict = {}  # Initialize dictionary to store probe events
    all_probe_metadata = [] # Initialize list to store metadata from each processed probe
    loader = None  # Initialize loader to None
    
    try:
        # Check if we're in find_global mode and the session folder doesn't exist
        if config['flags'].get('find_global', False):
            session_subfolder = os.path.join(swr_output_dir_path, f"swrs_session_{str(session_id)}")
            if not os.path.exists(session_subfolder):
                logger.info(f"Session {session_id}: No folder for Find Globals to run in, prior probe level detection likely failed, skipping")
                return

        # Create session subfolder paths
        session_subfolder = os.path.join(swr_output_dir_path, f"swrs_session_{str(session_id)}")
        
        # Create LFP subfolder path
        if save_lfp:
            session_lfp_subfolder = os.path.join(lfp_output_dir_path, f"lfp_session_{str(session_id)}")
        
        # Create directories
        os.makedirs(session_subfolder, exist_ok=True)
        if save_lfp:
            os.makedirs(session_lfp_subfolder, exist_ok=True)
        
        # Set up logging
        process_stage = "Setting up"
        logger.info(f"Session {session_id}: Beginning processing, dataset {dataset_to_process}")
        
        # Initialize and set up the loader using the BaseLoader factory
        process_stage = "Setting up loader"
        loader = BaseLoader.create(dataset_to_process, session_id)
        loader.set_up()
        
        # Get probe IDs and names
        process_stage = "Getting probe IDs and names"
        if dataset_to_process == 'abi_visual_coding' or dataset_to_process == 'abi_visual_behaviour':
            probenames = None
            probelist = loader.get_probes_with_ca1()
        elif dataset_to_process == 'ibl':
            # Get probes with CA1 directly - this will handle both getting all probes and filtering
            probelist, probenames = loader.get_probes_with_ca1()
        
        # If no probes with CA1, log and return
        if not probelist:
            logger.warning(f"Session {session_id}: No probes with CA1 found, skipping.")
            return
            
        # Process probes
        process_stage = "Running through the probes in the session"
        
        # Normal probe processing
        if not config['flags'].get('find_global', False):
            logger.info(f"\n{'='*80}\nProcessing probes normally (find_global=False)\n{'='*80}")
            for this_probe in range(len(probelist)):
                if dataset_to_process == 'ibl':
                    probe_name = probenames[this_probe]
                probe_id = probelist[this_probe]
                probe_id_log = str(probe_id)
                logger.info(f"Session {session_id}: Processing probe {probe_id_log}")
                
                # Process the probe and get results
                process_stage = f"Processing probe with id {probe_id_log}"
                if dataset_to_process == 'abi_visual_coding' or dataset_to_process == 'abi_visual_behaviour':
                    results = loader.process_probe(probe_id, filter_ripple_band)
                elif dataset_to_process == 'ibl':
                    results = loader.process_probe(this_probe, filter_ripple_band)
               
                # Extract results using the standardized key names
                peakripple_chan_raw_lfp = results['peak_ripple_raw_lfp']
                lfp_time_index = results['lfp_time_index']
                ca1_chans = results['ca1_channel_ids'] # Use standardized key
                outof_hp_chans_lfp = results['control_lfps']
                take_two = results['control_channel_ids'] # Use standardized key
                peakrippleband = results['ripple_band_filtered'] # Use standardized key
                peak_ripple_chan_id = results['peak_ripple_chan_id']
                
                # Save channel selection metadata only if the flag is enabled
                if save_channel_metadata:
                    channel_metadata_path = os.path.join(
                        session_subfolder,
                        f"probe_{probe_id_log}_channel_selection_metadata.json.gz"
                    )
                    with gzip.open(channel_metadata_path, 'wt', encoding='utf-8') as f:
                        json.dump(loader.channel_selection_metadata_dict, f)
                    logger.info(f"Session {session_id}: Saved channel selection metadata for probe {probe_id_log}")
                
                # Save LFP data if enabled
                if save_lfp:
                    np.savez(
                        os.path.join(
                            session_lfp_subfolder,
                            f"probe_{probe_id_log}_channel_{peak_ripple_chan_id}_lfp_ca1_putative_pyramidal_layer.npz",
                        ),
                        lfp_ca1=peakripple_chan_raw_lfp,
                    )
                    np.savez(
                        os.path.join(
                            session_lfp_subfolder,
                            f"probe_{probe_id_log}_channel_{peak_ripple_chan_id}_lfp_time_index_1500hz.npz",
                        ),
                        lfp_time_index=lfp_time_index,
                    )
                    
                    # Save control channel data
                    for i in range(2):
                        channel_outside_hp = take_two[i]
                        np.savez(
                            os.path.join(
                                session_lfp_subfolder,
                                f"probe_{probe_id_log}_channel_{channel_outside_hp}_lfp_control_channel.npz",
                            ),
                            lfp_control_channel=outof_hp_chans_lfp[i],
                        )
                
                # Create dummy speed vector for ripple detection
                dummy_speed = np.zeros_like(peakrippleband)
                
                # Detect putative ripples
                process_stage = f"Detecting Putative Ripples on probe with id {probe_id_log}"
                logger.info(f"Session {session_id}: Detecting putative ripples on probe {probe_id_log}")
                
                Karlsson_ripple_times = ripple_detection.Karlsson_ripple_detector(
                    time=lfp_time_index,
                    zscore_threshold=ripple_band_threshold,
                    filtered_lfps=peakrippleband[:, None],
                    speed=dummy_speed,
                    sampling_frequency=1500.0,
                )
                
                # Filter by duration
                Karlsson_ripple_times = Karlsson_ripple_times[
                    Karlsson_ripple_times.duration < 0.25
                ]
                
                # Remove speed columns
                speed_cols = [
                    col for col in Karlsson_ripple_times.columns if "speed" in col
                ]
                Karlsson_ripple_times = Karlsson_ripple_times.drop(columns=speed_cols)
                
                # Extract sharp wave component info
                sharp_wave_lfp = results['sharpwave_chan_raw_lfp']
                sw_chan_id = results['sharpwave_chan_id']
                
                # Save sharp wave LFP if enabled
                if save_lfp:
                    np.savez(
                        os.path.join(
                            session_lfp_subfolder,
                            f"probe_{probe_id_log}_channel_{sw_chan_id}_lfp_ca1_putative_str_radiatum.npz",
                        ),
                        lfp_ca1=sharp_wave_lfp,
                    )
                
                # Incorporate power metrics 
                logger.info(f"Session {session_id}: Incorporating power metrics for probe {probe_id_log}")
                Karlsson_ripple_times = loader.extending_edeno_event_stats(
                    events_df=Karlsson_ripple_times,
                    time_values=lfp_time_index,
                    ripple_filtered=peakrippleband
                )
                
                # Then incorporate sharp wave component info
                logger.info(f"Session {session_id}: Incorporating sharp wave component information for probe {probe_id_log}")
                sw_filter_data = np.load(sharp_wave_component_path)
                sharpwave_filter = sw_filter_data['sharpwave_componenet_8to40band_1500hz_band']
                
                Karlsson_ripple_times = loader.incorporate_sharp_wave_component_info(
                    events_df=Karlsson_ripple_times,
                    time_values=lfp_time_index,
                    ripple_filtered=peakrippleband,
                    sharp_wave_lfp=sharp_wave_lfp,
                    sharpwave_filter=sharpwave_filter
                )
                           
                # Detect gamma events
                process_stage = f"Detecting Gamma Events on probe with id {probe_id_log}"
                logger.info(f"Session {session_id}: Detecting gamma events on probe {probe_id_log}")
                
                # Filter raw lfp to get gamma band
                gamma_band_ca1 = np.convolve(
                    peakripple_chan_raw_lfp.reshape(-1), gamma_filter, mode="same"
                )
                
                gamma_power = np.abs(signal.hilbert(gamma_band_ca1)) ** 2
                gamma_times = event_boundary_detector(
                    time=lfp_time_index,
                    threshold_sd=gamma_event_thresh,
                    envelope=False,
                    minimum_duration=0.015,
                    maximum_duration=float("inf"),
                    five_to_fourty_band_power_df=gamma_power,
                )
                
                # Save gamma events
                csv_filename = f"probe_{probe_id_log}_channel_{peak_ripple_chan_id}_gamma_band_events.csv.gz"
                csv_path = os.path.join(session_subfolder, csv_filename)
                gamma_times.to_csv(csv_path, index=True, compression="gzip")
                
                # Detect movement artifacts
                process_stage = f"Detecting Movement Artifacts on probe with id {probe_id_log}"
                logger.info(f"Session {session_id}: Detecting movement artifacts on probe {probe_id_log}")
                
                movement_control_list = []
                for i in [0, 1]:
                    channel_outside_hp = take_two[i]
                    process_stage = f"Detecting Movement Artifacts on control channel {channel_outside_hp} on probe {probe_id_log}"
                    
                    # Process control channel for movement artifacts
                    ripple_band_control = outof_hp_chans_lfp[i]
                    dummy_speed = np.zeros_like(ripple_band_control)
                    ripple_band_control = filter_ripple_band(ripple_band_control)
                    rip_power_controlchan = np.abs(signal.hilbert(ripple_band_control)) ** 2
                    
                    # Reshape arrays as needed for different datasets
                    if dataset_to_process == 'abi_visual_behaviour':
                        lfp_time_index = lfp_time_index.reshape(-1)
                        dummy_speed = dummy_speed.reshape(-1)
                    if dataset_to_process == 'ibl':
                        # Reshape to ensure consistent (n_samples, n_channels) format for detector
                        rip_power_controlchan = rip_power_controlchan.reshape(-1,1)
                    
                    # Detect movement artifacts
                    movement_controls = ripple_detection.Karlsson_ripple_detector(
                        time=lfp_time_index.reshape(-1),
                        filtered_lfps=rip_power_controlchan,
                        speed=dummy_speed.reshape(-1),
                        zscore_threshold=movement_artifact_ripple_band_threshold,
                        sampling_frequency=1500.0,
                    )
                    
                    # Remove speed columns
                    speed_cols = [
                        col for col in movement_controls.columns if "speed" in col
                    ]
                    movement_controls = movement_controls.drop(columns=speed_cols)
                    
                    # Save movement artifact events
                    channel_outside_hp_str = f"channelsrawInd_{str(channel_outside_hp)}"
                    csv_filename = f"probe_{probe_id_log}_channel_{channel_outside_hp_str}_movement_artifacts.csv.gz"
                    csv_path = os.path.join(session_subfolder, csv_filename)
                    movement_controls.to_csv(csv_path, index=True, compression="gzip")
                    movement_control_list.append(movement_controls)
                
                # Apply filtering to label events with gamma/movement overlap
                logger.info(f"Session {session_id}: Filtering events for probe {probe_id_log}")
                Karlsson_ripple_times = check_gamma_overlap(Karlsson_ripple_times, gamma_times)
                Karlsson_ripple_times = check_movement_overlap(Karlsson_ripple_times, movement_control_list[0], movement_control_list[1])
                
                # Define the desired column order
                column_order = [
                    'start_time', 'end_time', 'duration', 
                    'power_peak_time', 'power_max_zscore', 'power_median_zscore',
                    'power_mean_zscore', 'power_min_zscore', 'power_90th_percentile',
                    'sw_exceeds_threshold', 'sw_peak_power', 'sw_peak_time',
                    'sw_ripple_plv', 'sw_ripple_mi', 'sw_ripple_clcorr',
                    'envelope_peak_time','envelope_max_thresh', 'envelope_mean_zscore', 
                    'envelope_median_zscore', 'envelope_max_zscore', 'envelope_min_zscore',
                    'envelope_area', 'envelope_total_energy', 'envelope_90th_percentile',
                    'overlaps_with_gamma', 'gamma_overlap_percent',
                    'overlaps_with_movement', 'movement_overlap_percent'
                ]
                
                # Reorder the DataFrame
                Karlsson_ripple_times = Karlsson_ripple_times[column_order]
                
                # Save filtered ripple events
                csv_filename = f"probe_{probe_id_log}_channel_{peak_ripple_chan_id}_putative_swr_events.csv.gz"
                csv_path = os.path.join(session_subfolder, csv_filename)
                Karlsson_ripple_times.to_csv(csv_path, index=True, compression="gzip")
                probe_events_dict[probe_id_log] = Karlsson_ripple_times
                logger.info(f"Session {session_id}: Saved {len(Karlsson_ripple_times)} filtered events for probe {probe_id_log}")
                
        else:
            logger.info(f"\n{'='*80}\nSkipping probe processing (find_global=True)\n{'='*80}")

        # Process global ripples if we have probe events
        if config['flags'].get('find_global', False):
            logger.info(f"\n{'='*80}\nLoading existing probe events for global detection\n{'='*80}")
            probe_events_dict = loader.events_dict_from_files(session_subfolder, session_id, logger)
        else:
            logger.info(f"\n{'='*80}\nSkipping probe processing (find_global=True)\n{'='*80}")

        # Generate metadata fresh from cache for all probes
        logger.info(f"Session {session_id}: Generating metadata for all probes")
        all_probe_metadata = []
        for probe_id in probelist:
            probe_metadata = loader.get_metadata_for_probe(probe_id, config=config)
            all_probe_metadata.append(probe_metadata)
        
        # Create metadata DataFrame
        probe_metadata_df = pd.DataFrame(all_probe_metadata)
        if probe_metadata_df.empty:
            logger.error(f"Session {session_id}: No valid probe metadata generated")
            return
            
        # Save metadata for future reference
        metadata_filepath = os.path.join(session_subfolder, f"session_{session_id}_probe_metadata.csv.gz")
        if os.path.exists(metadata_filepath) and not overwrite_existing:
            logger.info(f"Session {session_id}: Metadata file exists and overwrite not enabled. Skipping metadata save.")
        else:
            logger.info(f"Session {session_id}: Overwritting old probe metadata CSV ({len(all_probe_metadata)} entries).")
            probe_metadata_df.to_csv(metadata_filepath, index=False, compression="gzip")
            logger.info(f"Session {session_id}: Saved probe metadata to {metadata_filepath}")

        # Process global events if we have probe events
        if probe_events_dict:
            process_stage = "Detecting Global Ripples"
            logger.info(f"Session {session_id}: Detecting global ripples")
            
            # Validate probe metadata
            if 'probe_id' not in probe_metadata_df.columns:
                logger.error(f"Session {session_id}: 'probe_id' column missing from metadata DataFrame.  File corrupted")
                return
                
            # Convert probe_id column to string to ensure consistent type for matching keys
            probe_metadata_df['probe_id'] = probe_metadata_df['probe_id'].astype(str)
            
            # Create global events, passing the DataFrame instead of probe_info dict
            global_events = create_global_swr_events(
                probe_events_dict,
                global_swr_config,
                probe_metadata_df,
                session_subfolder, 
                session_id,
                logger
            )
            
            # Save global events if created successfully
            if global_events is not None and not global_events.empty:
                csv_filename = f"session_{session_id}_global_swr_events.csv.gz"
                csv_path = os.path.join(session_subfolder, csv_filename)
                
                # Check if file exists and handle overwrite
                if os.path.exists(csv_path):
                    if not overwrite_existing:
                        logger.warning(f"Session {session_id}: Global events file exists and overwrite not enabled. Skipping save.")
                    else:
                        logger.warning(f"Session {session_id}: Global events file exists and will be overwritten.")
                        global_events.to_csv(csv_path, index=True, compression="gzip")
                        logger.info(f"Session {session_id}: Created {len(global_events)} global events")
                else:
                    global_events.to_csv(csv_path, index=True, compression="gzip")
                    logger.info(f"Session {session_id}: Created {len(global_events)} global events")
            else:
                logger.warning(f"Session {session_id}: No global events were created - check probe criteria and event counts")
        
        else:
            logger.warning(f"Session {session_id}: Insufficient probe events available for global detection")
        
        # Save settings metadata for this run
        run_settings_path = os.path.join(session_subfolder, f"session_{session_id}_run_settings.json.gz")
        loader.save_settings_metadata(run_settings_path, config, logger)
        
        # Cleanup resources
        if loader is not None:
            loader.cleanup()
            logger.info(f"Session {session_id}: Loader cleanup finished")
            
            # Optional cache cleanup based on flag
            if cleanup_after:
                logger.info(f"Session {session_id}: Running cache cleanup")
                try:
                    loader.cleanup_cache(config=config)
                except Exception as e_cache:
                    logger.error(f"Session {session_id}: Error during cache cleanup - {e_cache}")
        
        logger.info(f"Session {session_id}: Processing completed successfully")
        
    except Exception as e_main:
        # Log the error
        tb_str = traceback.format_exc()
        logger.error(f"Session {session_id}: Error during processing at stage '{process_stage}' for probe '{probe_id_log}': {e_main}")
        logger.error(f"Traceback:\n{tb_str}")
        
        # Attempt loader cleanup if it exists
        if loader is not None:
            try:
                loader.cleanup()
                logger.info(f"Session {session_id}: Cleanup completed after error")
            except Exception as e_cleanup:
                logger.error(f"Session {session_id}: Error during loader cleanup after main exception: {e_cleanup}")
        
        # Check if session directory has existing probe metadata before deciding to delete
        if 'session_subfolder' in locals() and os.path.exists(session_subfolder):
            # Look for probe metadata files
            has_existing_data = any(
                f.startswith('probe_') and f.endswith('_putative_swr_events.csv.gz')
                for f in os.listdir(session_subfolder)
            )
            
            if not has_existing_data:
                try:
                    shutil.rmtree(session_subfolder)
                    logger.warning(f"Session {session_id}: Removed empty session folder after error")
                except Exception as e_rm:
                    logger.error(f"Session {session_id}: Failed to remove session folder: {e_rm}")
            else:
                logger.warning(f"Session {session_id}: Preserving session folder with existing probe data")
        
        # Check if LFP directory has existing data before deciding to delete
        if save_lfp and 'session_lfp_subfolder' in locals() and os.path.exists(session_lfp_subfolder):
            # Look for LFP files
            has_existing_lfp = any(
                f.endswith('_lfp_ca1_putative_pyramidal_layer.npz')
                for f in os.listdir(session_lfp_subfolder)
            )
            
            if not has_existing_lfp:
                try:
                    shutil.rmtree(session_lfp_subfolder)
                    logger.warning(f"Session {session_id}: Removed empty LFP folder after error")
                except Exception as e_rm:
                    logger.error(f"Session {session_id}: Failed to remove LFP folder: {e_rm}")
            else:
                logger.warning(f"Session {session_id}: Preserving LFP folder with existing data")

        # Handle overwrite and skipping logic for global detection
        if config['flags'].get('find_global', False):
            # Only require probe metadata for global detection if -fg flag is present
            metadata_file = os.path.join(session_subfolder, f"session_{session_id}_probe_metadata.csv.gz")
            if not os.path.exists(metadata_file):
                logger.info(f"Session {session_id}: Probe metadata missing, skipping global detection.")
                return

            # Check for existing global SWR files
            global_swr_files = [f for f in os.listdir(session_subfolder) if f.startswith(f"session_{session_id}_global_swrs_") and f.endswith(".csv.gz")]
            if global_swr_files:
                if not overwrite_existing:
                    logger.info(f"Session {session_id}: Global SWR file(s) exist and overwrite not enabled. Skipping global detection.")
                    return
                else:
                    # Overwrite: delete existing global SWR files
                    for f in global_swr_files:
                        os.remove(os.path.join(session_subfolder, f))
                    logger.info(f"Session {session_id}: Deleted existing global SWR files for overwrite.")
            # (Continue to run global detection after this block)


