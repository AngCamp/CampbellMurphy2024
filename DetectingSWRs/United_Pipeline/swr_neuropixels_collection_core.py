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

# Multiprocessing
from multiprocessing import Pool, Process, Queue, Manager, set_start_method

# AWS
import boto3
from botocore.config import Config

# Ripple detection
import ripple_detection
from ripple_detection import filter_ripple_band
import ripple_detection.simulate as ripsim

# Logging
import logging
import logging.handlers



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

        weighted_cos = np.sum(ripple_amp * np.cos(sw_phase))
        weighted_sin = np.sum(ripple_amp * np.sin(sw_phase))
        R = np.sqrt(weighted_cos ** 2 + weighted_sin ** 2)
        norm = np.sqrt(np.sum(ripple_amp ** 2))
        return R / norm

    def select_sharpwave_channel(
            self,
            ca1_lfp,
            lfp_time_index,
            ca1_chan_ids,
            this_chan_id,
            channel_positions,
            ripple_filtered,
            filter_path=None,
            running_exclusion_periods=None,
            selection_metric='modulation_index'):
        """
        Select the optimal sharp wave channel based on phase-amplitude coupling with ripple activity.

        Parameters
        ----------
        ca1_lfp : np.ndarray
            LFP data array (time x channels) for CA1 region.
        lfp_time_index : np.ndarray
            Time index array corresponding to ca1_lfp (in seconds).
        ca1_chan_ids : list of int
            List of channel IDs corresponding to columns in ca1_lfp.
        this_chan_id : int
            Channel ID of the previously selected ripple-detection channel.
        channel_positions : pd.Series
            Series mapping channel IDs to vertical probe positions (larger = deeper).
        ripple_filtered : np.ndarray
            Ripple-band filtered signal from the selected ripple detection channel.
        filter_path : str, optional
            Path to the sharp wave filter file. If None, must be provided by the loader implementation.
        running_exclusion_periods : list of tuple, optional
            List of (start_time, end_time) tuples for periods to exclude from analysis.
        selection_metric : {'modulation_index', 'circular_linear'}, optional
            Metric used to select the best sharp wave channel. Default is 'modulation_index'.

        Returns
        -------
        tuple
            (best_channel_id, best_channel_lfp) - Best channel ID and its raw LFP
        """
        # Get the filter - implementers must override this if filter_path is None
        if hasattr(self, 'sw_component_filter_path') and self.sw_component_filter_path is not None:
            filter_path = self.sw_component_filter_path
            
        if filter_path is None:
            raise ValueError("Sharp wave filter path must be provided or set in the loader instance")
            
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

        ref_depth = channel_positions.loc[this_chan_id]
        below_ids = channel_positions[channel_positions > ref_depth].index
        id_to_idx = {cid: i for i, cid in enumerate(ca1_chan_ids)}
        below_idx = [id_to_idx[cid] for cid in below_ids if cid in id_to_idx]

        # Create dictionary to store detailed results for selection logic
        results = {}
        
        # Create a separate dictionary for JSON-serializable summary stats
        self.sw_component_summary_stats_dict = {
            'channel_ids': [],
            'modulation_indices': [],
            'circular_linear_corrs': [],
            'depths': [],
            'selection_metric': selection_metric,
            'best_channel_id': None  # Will be filled in after selection
        }

        for cid, idx in zip(below_ids, below_idx):
            sw_filt = fftconvolve(ca1_lfp[:, idx], sharpwave_filter, mode='same')
            sw_an = hilbert(sw_filt)
            sw_phase = np.angle(sw_an)
            sw_power = np.abs(sw_an) ** 2
            sw_power_z = (sw_power - sw_power[mask].mean()) / sw_power[mask].std()

            high_mask = (ripple_power_z > 1) & (sw_power_z > 1) & mask

            modulation_index = self._compute_modulation_index(sw_phase, ripple_amp, high_mask)
            circular_linear_corr = self._compute_circular_linear_corr(sw_phase, ripple_amp, high_mask)

            # Store detailed results for selection
            results[cid] = {
                'modulation_index': modulation_index,
                'circular_linear_corr': circular_linear_corr,
                'idx': idx,
                'sw_power_z': sw_power_z
            }
            
            # Store JSON-serializable summary stats
            self.sw_component_summary_stats_dict['channel_ids'].append(int(cid))
            self.sw_component_summary_stats_dict['modulation_indices'].append(float(modulation_index) if not np.isnan(modulation_index) else None)
            self.sw_component_summary_stats_dict['circular_linear_corrs'].append(float(circular_linear_corr) if not np.isnan(circular_linear_corr) else None)
            self.sw_component_summary_stats_dict['depths'].append(float(channel_positions.loc[cid]))

        # Select best channel
        best_cid = max(
            results,
            key=lambda k: results[k][selection_metric] if not np.isnan(results[k][selection_metric]) else -np.inf
        )

        best_idx = results[best_cid]['idx']
        best_lfp = ca1_lfp[:, best_idx]
        
        # Store best channel ID in summary stats
        self.sw_component_summary_stats_dict['best_channel_id'] = int(best_cid)
        
        # Return the best channel ID and its raw LFP
        return best_cid, best_lfp

    def standardize_results(self, results, dataset_type):
        """
        Standardize the results dictionary format across different loaders.
        
        Parameters
        ----------
        results : dict
            Results dictionary from process_probe
        dataset_type : str
            Type of dataset
            
        Returns
        -------
        dict
            Standardized results dictionary
        """
        # Add dataset type to results
        results['dataset_type'] = dataset_type
        
        # Add sampling rate
        if 'sampling_rate' not in results:
            results['sampling_rate'] = 1500.0
            
        # Rename keys for consistency if needed
        key_mapping = {
            'ca1_chans': 'ca1_channel_ids',
            'control_channels': 'control_channel_ids',
            'peak_ripple_chan_raw_lfp': 'peak_ripple_raw_lfp',
            'rippleband': 'ripple_band_filtered'
        }
        
        for old_key, new_key in key_mapping.items():
            if old_key in results and new_key not in results:
                results[new_key] = results[old_key]
                
        # Create dataset-specific section for non-standard data
        dataset_specific = {}
        for key in list(results.keys()):
            if key not in [
                'probe_id', 'dataset_type', 'lfp_time_index', 'sampling_rate',
                'ca1_channel_ids', 'peak_ripple_chan_id', 'peak_ripple_raw_lfp',
                'ripple_band_filtered', 'sharpwave_chan_id', 'sharpwave_chan_raw_lfp',
                'sharpwave_power_z', 'control_channel_ids', 'control_lfps',
                'ca1_chans', 'control_channels', 'peak_ripple_chan_raw_lfp', 'rippleband'
            ]:
                dataset_specific[key] = results[key]
                results.pop(key)
                
        results['dataset_specific'] = dataset_specific
        
        return results

    # Abstract method stubs that must be implemented by subclasses
    # Due to dependency conflicts in the apis, differences in metadata labels or conventions
    # the code is sufficiently different to implement all of these steps
    # that they have implmented in their own seperate classes, see the _loader.py files
    # for the code there
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
    # Add to BaseLoader class in swr_neuropixels_collection_core.py

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

    def cleanup(self):
        """Clean up resources."""
        raise NotImplementedError("Subclasses must implement cleanup method")

# ===================================
#  PROBE LEVEL EVENT DETECTOR
#  -Channel selection, artifact, and putative ripple event detection
#  -intentionally overly permissive settings are used here
#  -most computationally expensive
# ===================================
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


def incorporate_sharp_wave_component_info(
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
    1. sw_exceeds_threshold      – True if sharp‑wave power (z‑score) > +1 SD at any point.
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
        FIR/IIR kernel to isolate the 8–40 Hz sharp‑wave component.
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
        """Circular‑linear correlation (vector‑strength form)."""
        if mask.sum() < 10:
            return np.nan
        phi = sw_phase[mask]
        amp = ripple_amp[mask]
        wc  = np.sum(amp * np.cos(phi))
        ws  = np.sum(amp * np.sin(phi))
        R   = np.sqrt(wc**2 + ws**2)
        return R / np.sqrt(np.sum(amp**2))

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


# ===================================
#  PROBE LEVEL EVENT FILTERING
#  -Thresholds applied here to probe level events
# ===================================

import pandas as pd
import numpy as np

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
    
    # Check each probe against each global event
    for probe_id, probe_df in probe_events_dict.items():
        for i, event in global_events.iterrows():
            # Check if any probe event overlaps with this global event
            overlap = ((probe_df['start_time'] <= event['end_time'] + offset) & 
                      (probe_df['end_time'] >= event['start_time'] - offset))
            
            if any(overlap):
                # Add this probe to the participating probes
                global_events.at[i, 'participating_probes'].append(probe_id)
                
                # Get peak time and power from the strongest overlapping event
                overlapping_events = probe_df[overlap]
                if 'Peak_time' in overlapping_events.columns and 'max_zscore' in overlapping_events.columns:
                    max_idx = overlapping_events['max_zscore'].idxmax()
                    global_events.at[i, 'peak_times'].append(overlapping_events.loc[max_idx, 'Peak_time'])
                    global_events.at[i, 'peak_powers'].append(overlapping_events.loc[max_idx, 'max_zscore'])
                elif 'Peak_time' in overlapping_events.columns:
                    max_idx = overlapping_events.index[0]
                    global_events.at[i, 'peak_times'].append(overlapping_events.loc[max_idx, 'Peak_time'])
                    global_events.at[i, 'peak_powers'].append(3.0)  # Default value
    
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

def create_global_events(probe_events_dict, merge_window=0.02, min_probe_count=1):
    """
    Create global events by merging events across probes.
    
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
    current = all_intervals[0]
    for interval in all_intervals[1:]:
        if interval[0] <= current[1] + merge_window:
            # Merge intervals
            current = (current[0], max(current[1], interval[1]), current[2])
        else:
            merged_intervals.append(current)
            current = interval
    merged_intervals.append(current)
    
    # Create DataFrame for global events
    global_events = pd.DataFrame({
        'start_time': [interval[0] for interval in merged_intervals],
        'end_time': [interval[1] for interval in merged_intervals],
        'original_event_idx': [interval[2] for interval in merged_intervals]
    })
    
    global_events['duration'] = global_events['end_time'] - global_events['start_time']
    
    # Add information about which probes participate in each event
    global_events = add_participating_probes(global_events, probe_events_dict, merge_window)
    
    # Filter events based on minimum probe count
    global_events = global_events[global_events['probe_count'] >= min_probe_count]
    
    if len(global_events) == 0:
        raise ValueError(f"No events meet the minimum probe count criteria (min_probe_count={min_probe_count})")
    
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

def create_global_swr_events(probe_events_dict, swr_config, probe_info, session_subfolder, session_id):
    """
    Create global SWR events from probe-level events.
    
    Parameters
    ----------
    probe_events_dict : dict
        Dictionary mapping probe IDs to event DataFrames
    swr_config : dict
        Configuration parameters for global SWR detection
    probe_info : dict
        Dictionary mapping probe IDs to probe information dictionaries
    session_subfolder : str
        Path to session directory
    session_id : str
        Session ID
        
    Returns
    -------
    pandas.DataFrame
        Global events DataFrame
    
    Raises
    ------
    ValueError
        If no events pass filtering criteria
    """
    if not probe_events_dict:
        raise ValueError("No probe events provided - probe_events_dict is empty")
    
    # Apply SW power filters to all collected events
    filtered_probe_events_dict = {}
    for probe_id, events in probe_events_dict.items():
        # Apply SW power threshold
        if 'sw_peak_power' in events.columns:
            filtered_events = events[events['sw_peak_power'] >= swr_config['min_sw_power']]
        elif 'SW_max_zscore' in events.columns:
            filtered_events = events[events['SW_max_zscore'] >= swr_config['min_sw_power']]
        else:
            filtered_events = events
            
        # Store events if we have enough after filtering
        if len(filtered_events) >= swr_config['min_filtered_events']:
            filtered_probe_events_dict[probe_id] = filtered_events
    
    if not filtered_probe_events_dict:
        raise ValueError(f"No probes have enough events after SW power filtering (min_filtered_events={swr_config['min_filtered_events']})")

    # Filter probes based on unit count
    unit_filtered_probe_events_dict = {}
    for probe_id, events in filtered_probe_events_dict.items():
        # Check if probe has enough CA1 units
        if probe_id in probe_info and probe_info[probe_id].get('ca1_good_unit_count', 0) >= swr_config['min_ca1_units']:
            unit_filtered_probe_events_dict[probe_id] = events

    if not unit_filtered_probe_events_dict:
        raise ValueError(f"No probes have enough CA1 units (min_ca1_units={swr_config['min_ca1_units']})")

    # Create global events
    global_events = create_global_events(
        unit_filtered_probe_events_dict,
        merge_window=swr_config['merge_window'],
        min_probe_count=swr_config['min_probe_count']
    )
    
    if global_events is None or len(global_events) == 0:
        raise ValueError(f"No global events created after merging (min_probe_count={swr_config['min_probe_count']})")
    
    # Save global events
    save_global_events(
        global_events, 
        session_subfolder, 
        session_id, 
        label=swr_config['global_rip_label']
    )
    
    return global_events