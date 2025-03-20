# pipeline that works on the 1500 Hz data from all the 
# probes

# IBL SWR detector
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
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from one.api import ONE
import spikeglx
from brainbox.io.one import load_channel_locations
from brainbox.io.spikeglx import Streamer
from brainbox.io.one import SpikeSortingLoader
from neurodsp.voltage import destripe_lfp
from ibllib.plots import Density
import time
import traceback
import logging
import logging.handlers
import sys
from multiprocessing import Pool, Process, Queue, Manager, set_start_method
import yaml

# Load the configuration from a YAML file
with open("ibl_swr_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Get the values from the configuration
pool_size = config["pool_size"]
gamma_filters_path = config["gamma_filters_path"]
run_name = config["run_name"]
oneapi_cache_dir = config["oneapi_cache_dir"]
output_dir = config["output_dir"]
swr_output_dir = config["swr_output_dir"]
gamma_event_thresh = config["gamma_event_thresh"]
ripple_band_threshold = config["ripple_band_threshold"]
movement_artifact_ripple_band_threshold = config[
    "movement_artifact_ripple_band_threshold"
]
dont_wipe_these_sessions = config["dont_wipe_these_sessions"]
session_npz_filepath = config["session_npz_filepath"]
save_lfp = config["save_lfp"]


# FUNCTIONS
# subprocess is a default module
def call_bash_function(bash_command=""):
    # example bash comand:
    # bash_command = "source /path/to/your/bash_script.sh && your_bash_function"
    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    if process.returncode == 0:
        print("Bash function executed successfully.")
        print("Output:", output.decode("utf-8"))
    else:
        print("Error:", error.decode("utf-8"))


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

def listener_process(queue):
    """
    This function listens for messages from the logging module and writes them to a log file.
    It sets the logging level to MESSAGE so that only messages with level MESSAGE or higher are written to the log file.
    This is a level we created to be between INFO and WARNING, so to see messages from this code and errors  but not other
    messages that are mostly irrelevant and make the log file too large and uninterpretable.

    Parameters
    ----------
    queue : multiprocessing.Queue
        The queue to get messages from.

    Returns
    -------
    None

    """
    root = logging.getLogger()
    h = logging.FileHandler(
        f"ibl_detector_{swr_output_dir}_{run_name}_app.log", mode="w"
    )
    f = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    h.setFormatter(f)
    root.addHandler(h)
    root.setLevel(MESSAGE)  # Set logging level to MESSAGE

    while True:
        message = queue.get()
        if message == "kill":
            break
        logger = logging.getLogger(message.name)
        logger.handle(message)

def init_pool(*args):
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(MESSAGE)  # Set logging level to MESSAGE

class ibl_loader:
    def __init__(self, session_id, br=None):
        """
        Initialize the IBL loader with a session ID.
        
        Parameters
        ----------
        session_id : str
            The IBL session ID
        br : BrainRegions, optional
            An existing BrainRegions object to use. If None, brain regions
            will need to be added externally.
        """
        self.session_id = session_id
        self.probe_id = "Not Loaded Yet"
        self.one_exists = False
        self.probelist = None
        self.probenames = None
        self.one = None
        self.data_files = None
        self.br = br
        
    def set_up(self):
        """
        Sets up the ONE API connection.
        """
        ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
        self.one = ONE(password="international")
        self.one_exists = True
        return self
        
    def get_probe_ids_and_names(self):
        """
        Gets the probe IDs and names for the session.        
        """
        if not self.one_exists:
            self.set_up()
        self.probelist, self.probenames = self.one.eid2pid(self.session_id)
        print(f"Probe IDs: {self.probelist}, Probe names: {self.probenames}")
        return self.probelist, self.probenames
    
    def load_channels(self, probe_idx):
        """
        Loads channel data for a specific probe.
        
        Parameters
        ----------
        probe_idx : int
            Index of the probe in the probelist.
            
        Returns
        -------
        tuple
            (channels, probe_name, probe_id)
        """
        if self.probelist is None:
            self.get_probe_ids_and_names()
            
        probe_name = self.probenames[probe_idx]
        probe_id = self.probelist[probe_idx]
        print(f"Loading channel data for probe: {probe_id}")
        
        # Get channels data
        collectionname = f"alf/{probe_name}/pykilosort"
        channels = self.one.load_object(self.session_id, "channels", collection=collectionname)
        
        return channels, probe_name, probe_id
    
    def has_ca1_channels(self, channels):
        """
        Checks if the channels include CA1.
        
        Parameters
        ----------
        channels : object
            Channel information object
            
        Returns
        -------
        bool
            True if CA1 channels exist, False otherwise
        """
        if self.br is None:
            raise ValueError("BrainRegions object (br) must be set to check for CA1 channels")
            
        channels.allen2017_25um_acronym = self.br.id2acronym(
            channels["brainLocationIds_ccf_2017"]
        )
        
        regions_on_probe = np.unique(channels.allen2017_25um_acronym)
        has_ca1 = "CA1" in regions_on_probe
        
        if not has_ca1:
            print(f"No CA1 channels on probe, skipping...")
            
        return has_ca1
    
    def load_bin_file(self, probe_name):
        """
        Loads the binary file for a probe.
        
        Parameters
        ----------
        probe_name : str
            Name of the probe
            
        Returns
        -------
        pathlib.Path or None
            Path to the binary file
        """
        # Find the relevant datasets and download them
        dsets = self.one.list_datasets(
            self.session_id, collection=f"raw_ephys_data/{probe_name}", filename="*.lf.*"
        )
        print(f"Found {len(dsets)} datasets")
        
        self.data_files, _ = self.one.load_datasets(self.session_id, dsets, download_only=False)
        bin_file = next((df for df in self.data_files if df.suffix == ".cbin"), None)
        
        if bin_file is None:
            print(f"No .cbin file found for probe {probe_name}, skipping...")
            
        return bin_file
    
    def create_time_index(self, sr, probe_id):
        """
        Creates a time index for the LFP data.
        
        Parameters
        ----------
        sr : spikeglx.Reader
            SpikeGLX reader object
        probe_id : str
            Probe ID
            
        Returns
        -------
        numpy.ndarray
            Time index for the LFP data
        """
        # Make time index
        start_time = time.time()
        ssl = SpikeSortingLoader(pid=probe_id, one=self.one)
        t0 = ssl.samples2times(0, direction="forward")
        dt = (ssl.samples2times(1, direction="forward") - t0) * 12
        lfp_time_index_og = np.arange(0, sr.shape[0]) * dt + t0
        del ssl
        print(f"Time index created, time elapsed: {time.time() - start_time}")
        
        return lfp_time_index_og
    
    def extract_raw_data(self, sr):
        """
        Extracts raw data from the SpikeGLX reader.
        
        Parameters
        ----------
        sr : spikeglx.Reader
            SpikeGLX reader object
            
        Returns
        -------
        tuple
            (raw_data, sampling_rate)
        """
        # Extract raw data
        start_time = time.time()
        raw = sr[:, : -sr.nsync].T
        fs_from_sr = sr.fs
        print(f"Raw data extracted, time elapsed: {time.time() - start_time}")
        
        return raw, fs_from_sr
    
    def destripe_data(self, raw, fs):
        """
        Applies destriping to the raw data.
        
        Parameters
        ----------
        raw : numpy.ndarray
            Raw data
        fs : float
            Sampling rate
            
        Returns
        -------
        numpy.ndarray
            Destriped data
        """
        start_time = time.time()
        destriped = destripe_lfp(raw, fs=fs)
        print(f"Destriped shape: {destriped.shape}")
        print(f"Destriping done, time elapsed: {time.time() - start_time}")
        
        return destriped
    
    def get_ca1_channels(self, channels, destriped):
        """
        Gets the CA1 channels from the destriped data.
        
        Parameters
        ----------
        channels : object
            Channel information object
        destriped : numpy.ndarray
            Destriped data
            
        Returns
        -------
        tuple
            (ca1_lfp, ca1_channel_indices)
        """
        ca1_chans = channels.rawInd[channels.allen2017_25um_acronym == "CA1"]
        lfp_ca1 = destriped[ca1_chans, :]
        
        return lfp_ca1, ca1_chans
    
    def get_non_hippocampal_channels(self, channels, destriped):
        """
        Gets two non-hippocampal channels for artifact detection.
        
        Parameters
        ----------
        channels : object
            Channel information object
        destriped : numpy.ndarray
            Destriped data
            
        Returns
        -------
        tuple
            (non_hippocampal_lfp_list, non_hippocampal_channel_indices)
        """
        # Find channels outside the hippocampal formation
        not_a_hp_chan = np.logical_not(
            np.isin(
                channels.allen2017_25um_acronym,
                ["CA3", "CA2", "CA1", "HPF", "EC", "DG"],
            )
        )
        
        # Select two random non-hippocampal channels
        control_channels = np.random.choice(
            channels.rawInd[not_a_hp_chan], 2, replace=False
        )
        
        # Extract data for these channels
        control_data = []
        for channel_idx in control_channels:
            control_data.append(destriped[channel_idx, :])
            
        return control_data, control_channels
    
    def resample_signal(self, lfp_data, time_index_og, target_fs=1500.0):
        """
        Resamples the signal to a target frequency.
        
        Parameters
        ----------
        lfp_data : numpy.ndarray
            LFP data
        time_index_og : numpy.ndarray
            Original time index
        target_fs : float, optional
            Target sampling frequency
            
        Returns
        -------
        tuple
            (resampled_data, new_time_index)
        """
        # Create new time index at target sampling rate
        t_start = time_index_og[0]
        t_end = time_index_og[-1]
        dt_new = 1.0 / target_fs
        n_samples = int(np.ceil((t_end - t_start) / dt_new))
        new_time_index = t_start + np.arange(n_samples) * dt_new
        
        # Check if lfp_data is 1D or 2D
        if lfp_data.ndim == 1:
            # For 1D array
            interp_func = interpolate.interp1d(
                time_index_og,
                lfp_data,
                bounds_error=False,
                fill_value="extrapolate",
            )
            resampled = interp_func(new_time_index)
            resampled = resampled.T  # Transpose for standard orientation
        else:
            # For 2D array (multiple channels)
            resampled = np.zeros((lfp_data.shape[0], len(new_time_index)))
            for i in range(lfp_data.shape[0]):
                interp_func = interpolate.interp1d(
                    time_index_og,
                    lfp_data[i, :],
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                resampled[i, :] = interp_func(new_time_index)
            
            resampled = resampled.T  # Transpose for standard orientation
        
        return resampled, new_time_index
    
    def find_peak_ripple_channel(self, lfp_ca1, ca1_chans, filter_ripple_band_func):
        """
        Finds the CA1 channel with the highest ripple power.
        
        Parameters
        ----------
        lfp_ca1 : numpy.ndarray
            LFP data from CA1 channels
        ca1_chans : numpy.ndarray
            Indices of CA1 channels
        filter_ripple_band_func : function
            Function to filter signal to ripple band
            
        Returns
        -------
        tuple
            (peak_channel_index, peak_channel_id, peak_channel_raw_lfp)
        """
        lfp_ca1_rippleband = filter_ripple_band_func(lfp_ca1)
        highest_rip_power = np.abs(signal.hilbert(lfp_ca1_rippleband)) ** 2
        highest_rip_power = highest_rip_power.max(axis=0)
        
        peak_channel_idx = highest_rip_power.argmax()
        peak_channel_id = ca1_chans[peak_channel_idx]
        peak_channel_raw_lfp = lfp_ca1[:, peak_channel_idx]
        
        return peak_channel_idx, peak_channel_id, peak_channel_raw_lfp
    
    def process_probe(self, probe_idx, filter_ripple_band_func=None):
        """
        Processes a single probe completely.
        
        Parameters
        ----------
        probe_idx : int
            Index of the probe in the probelist
        filter_ripple_band_func : function, optional
            Function to filter for ripple band
            
        Returns
        -------
        dict
            Dictionary with processing results
        """
        if self.probelist is None:
            self.get_probe_ids_and_names()
            
        # Step 1: Load channels
        channels, probe_name, probe_id = self.load_channels(probe_idx)
        
        # Step 2: Check for CA1 channels
        if self.br is not None and not self.has_ca1_channels(channels):
            return None
            
        # Step 3: Load bin file
        bin_file = self.load_bin_file(probe_name)
        if bin_file is None:
            return None
            
        # Step 4: Read the data
        print(f"Reading LFP data for probe {probe_id}...")
        sr = spikeglx.Reader(bin_file)
        
        # Step 5: Create time index
        lfp_time_index_og = self.create_time_index(sr, probe_id)
        
        # Step 6: Extract raw data
        raw, fs_from_sr = self.extract_raw_data(sr)
        del sr  # Free memory
        
        # Step 7: Destripe data
        destriped = self.destripe_data(raw, fs_from_sr)
        del raw  # Free memory
        
        # Step 8: Get CA1 channels
        lfp_ca1, ca1_chans = self.get_ca1_channels(channels, destriped)
        
        # Step 9: Get non-hippocampal control channels for artifact detection
        print(f"Getting control channels for artifact detection...")
        control_data, control_channels = self.get_non_hippocampal_channels(channels, destriped)
        
        # Step 10: Resample CA1 channels to 1500 Hz
        print(f"Resampling CA1 channels to 1.5kHz...")
        lfp_ca1, lfp_time_index = self.resample_signal(lfp_ca1, lfp_time_index_og, 1500.0)
        
        # Step 11: Resample control channels
        outof_hp_chans_lfp = []
        for channel_data in control_data:
            # Reshape to 2D array with shape (1, n_samples)
            channel_data = channel_data.reshape(1, -1)
            
            # Resample
            lfp_control, _ = self.resample_signal(channel_data, lfp_time_index_og, 1500.0)
            
            # Append to list, ensuring correct shape
            outof_hp_chans_lfp.append(lfp_control[:, None])
            del lfp_control  # Free memory
        
        del destriped  # Free memory for large array
        del control_data  # Free memory
        
        # Step 12: Find channel with highest ripple power if function provided
        if filter_ripple_band_func is not None:
            peak_idx, peak_id, peak_lfp = self.find_peak_ripple_channel(
                lfp_ca1, ca1_chans, filter_ripple_band_func
            )
            
            # Create a channel ID string for naming files
            this_chan_id = f"channelsrawInd_{peak_id}"
        else:
            peak_idx = None
            peak_id = None
            peak_lfp = None
            this_chan_id = None
            
        # Collect results
        results = {
            'probe_id': probe_id,
            'probe_name': probe_name,
            'lfp_ca1': lfp_ca1,
            'lfp_time_index': lfp_time_index,
            'channels': channels,
            'ca1_chans': ca1_chans,
            'control_lfps': outof_hp_chans_lfp,
            'control_channels': control_channels,
            'peak_ripple_chan_idx': peak_idx,
            'peak_ripple_chan_id': peak_id,
            'peak_ripple_chan_raw_lfp': peak_lfp,
            'chan_id_string': this_chan_id
        }
        
        return results
    
    def cleanup(self):
        """
        Cleans up resources to free memory.
        """
        self.data_files = None


def process_session(session_id):
    """
    This function takes in a session_id (eid in the IBL) and loops through the probes in that session,
    for each probe it finds the CA1 channel with the highest ripple power and uses that
    channel to detect SWR events.  It also detects gamma events and movement artifacts
    on two channels outside of the brain.
    
    Parameters
    ----------
    session_id : int
        The session id for the session to be processed.
    queue : multiprocessing.Queue
        The queue to send messages to the listener process for recording errors.
    
    Returns
    -------
    None
    but...
    Saves the following files to the folder specified by swr_output_dir_path.
    
    Notes:
    - The LFP is interpolated to 1500 Hz for all channels used.
    - The SWR detector used is the Karlsson ripple detector from the ripple_detection module.
    - The folders are titled by session and all files contain the name of the probe and the channel they originated from
    """
    
    process_stage = "Starting the process"  # for debugging
    probe_id = "Not Loaded Yet"
    one_exists = False
    
    # Create session subfolder
    session_subfolder = "swrs_session_" + str(session_id)
    session_subfolder = os.path.join(swr_output_dir_path, session_subfolder)
    
    try:
        # Set up brain atlas
        process_stage = "Setting up brain atlas"
        ba = AllenAtlas()
        br = BrainRegions()
        
        process_stage = "Session loaded, checking if directory exists"
        # Check if directory already exists
        if os.path.exists(session_subfolder):
            raise FileExistsError(f"The directory {session_subfolder} already exists.")
        else:
            os.makedirs(session_subfolder)
            
        if save_lfp == True:
            # Create subfolder for lfp data
            session_lfp_subfolder = "lfp_session_" + str(session_id)
            session_lfp_subfolder = os.path.join(lfp_output_dir_path, session_lfp_subfolder)
            os.makedirs(session_lfp_subfolder, exist_ok=True)
        
        # Initialize and set up the IBL loader
        process_stage = "Setting up IBL loader"
        loader = ibl_loader(session_id, br=br)
        loader.set_up()
        one_exists = True  # Mark that we have a connection for error handling
        
        # Get probe IDs and names
        process_stage = "Getting probe IDs and names"
        probelist, probenames = loader.get_probe_ids_and_names()
        
        process_stage = "Running through the probes in the session"
        # Process each probe
        for this_probe in range(len(probelist)):
            probe_name = probenames[this_probe]
            probe_id = probelist[this_probe]
            print(f"Processing probe: {probe_id}")
            
            # Create a function for ripple band filtering
            def filter_ripple_band(signal_data):
                # Your existing filter_ripple_band implementation
                return filter_ripple_band(signal_data)
            
            # Process the probe and get results
            process_stage = f"Processing probe {probe_name} with id {probe_id}"
            results = loader.process_probe(this_probe, filter_ripple_band)
            
            # Skip if no results (no CA1 channels or no bin file)
            if results is None:
                print(f"No results for probe {probe_id}, skipping...")
                continue
            
            # Extract results
            lfp_ca1 = results['lfp_ca1']
            lfp_time_index = results['lfp_time_index']
            ca1_chans = results['ca1_chans']
            outof_hp_chans_lfp = results['control_lfps']
            take_two = results['control_channels']
            
            # From here, continue with your existing processing steps:
            # - Filter for ripple band
            lfp_ca1_ripppleband = filter_ripple_band(lfp_ca1)
            highest_rip_power = np.abs(signal.hilbert(lfp_ca1_ripppleband)) ** 2
            highest_rip_power = highest_rip_power.max(axis=0)
            
            # Get channel ID and peak ripple data
            this_chan_id = "channelsrawInd_" + str(ca1_chans[highest_rip_power.argmax()])
            peakrippleband = lfp_ca1_ripppleband[:, highest_rip_power.argmax()]
            peakripple_chan_raw_lfp = lfp_ca1[:, highest_rip_power.argmax()]
            
            # Filter to gamma band
            gamma_band_ca1 = np.convolve(
                peakripple_chan_raw_lfp.reshape(-1), gamma_filter, mode="same"
            )

            # write our lfp to file
            np.savez(
                os.path.join(
                    session_lfp_subfolder,
                    f"probe_{probe_id}_channel_{this_chan_id}_lfp_ca1_peakripplepower.npz",
                ),
                lfp_ca1=peakripple_chan_raw_lfp,
            )
            np.savez(
                os.path.join(
                    session_lfp_subfolder,
                    f"probe_{probe_id}_channel_{this_chan_id}_lfp_time_index_1500hz.npz",
                ),
                lfp_time_index=lfp_time_index,
            )
            for i in range(2):
                channel_outside_hp = take_two[i]
                channel_outside_hp = "channelsrawInd_" + str(channel_outside_hp)
                np.savez(
                    os.path.join(
                        session_lfp_subfolder,
                        f"probe_{probe_id}_channel_{channel_outside_hp}_lfp_control_channel.npz",
                    ),
                    lfp_control_channel=outof_hp_chans_lfp[i],
                )

            del lfp_ca1  # clear up some memory

            # create a dummy speed vector
            dummy_speed = np.zeros_like(peakrippleband)
            print("Detecting Putative Ripples")
            # we add a dimension to peakrippleband because the ripple detector needs it
            process_stage = (
                f"Detecting Putative Ripples on probe {probe_name} with id {probe_id}"
            )
            Karlsson_ripple_times = ripple_detection.Karlsson_ripple_detector(
                time=lfp_time_index,
                zscore_threshold=ripple_band_threshold,
                filtered_lfps=peakrippleband[:, None],
                speed=dummy_speed,
                sampling_frequency=1500.0,
            )

            Karlsson_ripple_times = Karlsson_ripple_times[
                Karlsson_ripple_times.duration < 0.25
            ]
            print("Done")
            # adds some stuff we want to the file

            # ripple band power
            peakrippleband_power = np.abs(signal.hilbert(peakrippleband)) ** 2
            Karlsson_ripple_times["Peak_time"] = peaks_time_of_events(
                events=Karlsson_ripple_times,
                time_values=lfp_time_index,
                signal_values=peakrippleband_power,
            )
            speed_cols = [
                col for col in Karlsson_ripple_times.columns if "speed" in col
            ]
            Karlsson_ripple_times = Karlsson_ripple_times.drop(columns=speed_cols)
            csv_filename = (
                f"probe_{probe_id}_channel_{this_chan_id}_karlsson_detector_events.csv"
            )
            csv_path = os.path.join(session_subfolder, csv_filename)
            Karlsson_ripple_times.to_csv(csv_path, index=True, compression="gzip")
            print("Writing to file.")
            print("Detecting gamma events.")

            # compute this later, I will have a seperate script called SWR filtering which will do this
            process_stage = (
                f"Detecting Gamma Events on probe {probe_name} with id {probe_id}"
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
            print("Done")
            csv_filename = (
                f"probe_{probe_id}_channel_{this_chan_id}_gamma_band_events.csv"
            )
            csv_path = os.path.join(session_subfolder, csv_filename)
            gamma_times.to_csv(csv_path, index=True, compression="gzip")

            # movement artifact detection
            process_stage = (
                f"Detecting Movement Artifacts on probe {probe_name} with id {probe_id}"
            )
            for i in [0, 1]:
                channel_outside_hp = take_two[i]
                process_stage = f"Detecting Movement Artifacts on control channel {channel_outside_hp} on probe {probe_name} with id {probe_id}"
                # process control channel ripple times
                ripple_band_control = outof_hp_chans_lfp[i]

                ripple_band_control = filter_ripple_band(ripple_band_control)
                rip_power_controlchan = np.abs(signal.hilbert(ripple_band_control)) ** 2
                movement_controls = ripple_detection.Karlsson_ripple_detector(
                    time=lfp_time_index,  # if this doesnt work try adding .reshape(-1)
                    filtered_lfps=rip_power_controlchan,  # indexing [:,None] is not needed here, rip_power_controlchan is already 2d (nsamples, 1)
                    speed=dummy_speed,  # if this doesnt work try adding .reshape(-1)
                    zscore_threshold=movement_artifact_ripple_band_threshold,
                    sampling_frequency=1500.0,
                )
                speed_cols = [
                    col for col in movement_controls.columns if "speed" in col
                ]
                movement_controls = movement_controls.drop(columns=speed_cols)
                # write to file name
                channel_outside_hp = "channelsrawInd_" + str(
                    channel_outside_hp
                )  # no cjannel id in IBL dataset, so this will do instead
                csv_filename = f"probe_{probe_id}_channel_{channel_outside_hp}_movement_artifacts.csv"
                csv_path = os.path.join(session_subfolder, csv_filename)
                movement_controls.to_csv(csv_path, index=True, compression="gzip")
                print("Done Probe id " + str(probe_id))

        # deleting the session folder
        del one  # so that we can delete the session folder, note sr and ssl need to be deleted as well, already done earlier
        process_stage = "All processing done, Deleting the session folder"
        one_exists = False
        # Get the file path of the session folder trim it and then use it to delete the folder
        s = str(data_files[0])

        # Find the index of "raw_ephys_data" in the string
        index = s.find("raw_ephys_data")

        # Remove everything after "raw_ephys_data" including that string
        s = s[:index]

        # Remove the substring "PosixPath('"
        s = s.replace("PosixPath('", "")

        # Remove the trailing slash
        s = s.rstrip("/")

        # Define the bash command to delete the folder
        cmd = f"rm -r {s}"

        # Execute the bash command
        os.system(cmd)

        # in the session
        logging.log(MESSAGE, f"Processing complete for id {session_id}.")
    except Exception:

        # we still need to clear the session
        # if an error occured in deleting the session folder one will have already been deleted
        if one_exists:
            del one  # so that we can delete the session folder, note sr and ssl need to be deleted as well, already done earlier

        if data_files is not None:
            # to avoid an exception where data files have not been created yet
            # Get the file path of the session folder trim it and then use it to delete the folder
            s = str(data_files[0])

            # Find the index of "raw_ephys_data" in the string
            index = s.find("raw_ephys_data")

            # Remove everything after "raw_ephys_data" including that string
            s = s[:index]

            # Remove the substring "PosixPath('"
            s = s.replace("PosixPath('", "")

            # Remove the trailing slash
            s = s.rstrip("/")

            # Define the bash command to delete the folder
            cmd = f"rm -rf {s}"

            # Execute the bash command
            os.system(cmd)
        # Check if the session subfolder is empty
        if os.path.exists(session_subfolder) and not os.listdir(session_subfolder):
            # If it is, delete it
            os.rmdir(session_subfolder)
            logging.log(
                MESSAGE,
                "PROCESSING FAILED REMOVING EMPTY SESSION SWR DIR :  session id %s ",
                session_id,
            )
        # if there is an error we want to know about it, but we dont want it to stop the loop
        # so we will print the error to a file and continue
        logging.error(
            "Error in session: %s, probe id: %s, Process Error at : ",
            session_id,
            probe_id,
            process_stage,
        )
        logging.error(traceback.format_exc())


# set up the logging
MESSAGE = 25  # Define a custom logging level, between INFO (20) and WARNING (30)

# loading filters (crates artifacts in first and last ~ 3.5 seconds of recordings, remember to clip these off)
# I don't think I need this it's at the start of my files
gamma_filter = np.load(gamma_filters_path)
gamma_filter = gamma_filter["arr_0"]

# load in the brain atlas and the brain region object for working with the ccf and ABI region id's in channels objects
ba = AllenAtlas()
br = BrainRegions()  # br is also an attribute of ba so could to br = ba.regions

# Searching for datasets
brain_acronym = "CA1"
# query sessions endpoint
# sessions, sess_details = one.search(atlas_acronym=brain_acronym, query_type='remote', details=True)

swr_output_dir_path = os.path.join(output_dir, swr_output_dir)
os.makedirs(swr_output_dir_path, exist_ok=True)
sessions_without_ca1 = np.array([])
# from multiprocessing import Pool

if save_lfp == True:
    lfp_output_dir_path = os.path.join(output_dir, swr_output_dir + "_lfp_data")
    os.makedirs(lfp_output_dir_path, exist_ok=True)

queue = Queue()
listener = Process(target=listener_process, args=(queue,))
listener.start()

data = np.load(session_npz_filepath)
all_sesh_with_ca1_eid = data["all_sesh_with_ca1_eid_unique"]
del data


# run the processes with the specified number of cores:
with Pool(pool_size, initializer=init_pool, initargs=(queue,)) as p:
    p.map(process_session, all_sesh_with_ca1_eid)

queue.put("kill")
listener.join()
