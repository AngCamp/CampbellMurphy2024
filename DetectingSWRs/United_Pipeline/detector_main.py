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
#from iblatlas.atlas import AllenAtlas
#from iblatlas.regions import BrainRegions

#from one.api import ONE
#import spikeglx
#from brainbox.io.one import load_channel_locations
#from brainbox.io.spikeglx import Streamer
#from brainbox.io.one import SpikeSortingLoader
#from neurodsp.voltage import destripe_lfp
#from ibllib.plots import Density
import time
import traceback
import logging
import logging.handlers
import sys
from multiprocessing import Pool, Process, Queue, Manager, set_start_method
import yaml
import string

# Get loader type from environment variable with a default value
#DATASET_TO_PROCESS = os.environ.get('DATASET_TO_PROCESS', 'ibl').lower()
#DATASET_TO_PROCESS = os.environ.get('DATASET_TO_PROCESS', 'abi').lower()  
DATASET_TO_PROCESS = 'ibl'



# Lazy loading of the appropriate loader class
if DATASET_TO_PROCESS == 'ibl':
    from IBL_loader import ibl_loader
elif DATASET_TO_PROCESS == 'abi':
    from ABI_loader import abi_loader
else:
    raise ValueError(f"Unknown dataset type: {DATASET_TO_PROCESS}")

# Load the configuration from a YAML file
config_path = os.environ.get('CONFIG_PATH', 'united_detector_config.yaml')
with open(config_path, "r") as f:
    config_content = f.read()
    full_config = yaml.safe_load(config_content)

# Extract the unified output directory first
output_dir = full_config.get("output_dir", "")

# Load common settings
pool_size = full_config["pool_sizes"][DATASET_TO_PROCESS]
gamma_event_thresh = full_config["gamma_event_thresh"]
ripple_band_threshold = full_config["ripple_band_threshold"]
movement_artifact_ripple_band_threshold = full_config["movement_artifact_ripple_band_threshold"]
run_name = full_config["run_name"]
save_lfp = full_config["save_lfp"]

# Load dataset-specific settings
if DATASET_TO_PROCESS == 'ibl':
    # IBL specific settings
    dataset_config = full_config["ibl"]
    gamma_filters_path = full_config["filters"]["gamma_filters"]["ibl"]
    oneapi_cache_dir = dataset_config["oneapi_cache_dir"]
    swr_output_dir = dataset_config["swr_output_dir"]
    dont_wipe_these_sessions = dataset_config["dont_wipe_these_sessions"]
    session_npz_filepath = dataset_config["session_npz_filepath"]
    # Additional IBL-specific variables if needed
    
elif DATASET_TO_PROCESS == 'abi':
    # ABI (Allen) specific settings
    dataset_config = full_config["abi"]
    gamma_filters_path = full_config["filters"]["gamma_filters"]["allen"]
    theta_filter_path = full_config["filters"]["theta_filter"]
    #sdk_cache_dir = dataset_config["sdk_cache_dir"]
    swr_output_dir = dataset_config["swr_output_dir"]
    dont_wipe_these_sessions = dataset_config["dont_wipe_these_sessions"]
    only_brain_observatory_sessions = dataset_config["only_brain_observatory_sessions"]
    # Setting up the ABI Cache (where data is held, what is present or absent)
    #manifest_path = os.path.join(sdk_cache_dir, "manifest.json")
    # There's no session_npz_filepath for ABI in the consolidated config

print(f"Configured for dataset: {DATASET_TO_PROCESS}")
print(f"Pool size: {pool_size}")
print(f"Output directory: {output_dir}")
print(f"SWR output directory: {swr_output_dir}")


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
    
    process_stage = f"Starting the process, session{str(session_id)}"  # for debugging
    probe_id = "Not Loaded Yet"
    one_exists = False
    # Add this near the beginning of the function
    data_files = None
    process_stage = "Starting the process"  # for debugging
    probe_id = "Not Loaded Yet"
    
    # Create session subfolder
    session_subfolder = "swrs_session_" + str(session_id)
    session_subfolder = os.path.join(swr_output_dir_path, session_subfolder)
    
    try:
        # Set up brain atlas
        process_stage = "Setting up brain atlas"
        #ba = AllenAtlas()
        #br = BrainRegions()
        
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
        
        if DATASET_TO_PROCESS == 'ibl':
            loader = ibl_loader(session_id)
        elif DATASET_TO_PROCESS == 'abi':
            loader = abi_loader(session_id)
        loader.set_up()
        one_exists = True  # Mark that we have a connection for error handling
        
        # Get probe IDs and names
        process_stage = "Getting probe IDs and names"
        if DATASET_TO_PROCESS == 'ibl':
            probelist, probenames = loader.get_probe_ids_and_names()
        elif DATASET_TO_PROCESS == 'abi':
            probenames = None
            probelist = loader.get_probes_with_ca1()

        process_stage = "Running through the probes in the session"
        # Process each probe
        for this_probe in range(len(probelist)):
            if DATASET_TO_PROCESS == 'ibl':
                probe_name = probenames[this_probe]
            probe_id = probelist[this_probe]  # Always get the probe_id from probelist
            print(f"Processing probe: {str(probe_id)}")

            # Process the probe and get results
            process_stage = f"Processing probe with id {str(probe_id)}"
            if DATASET_TO_PROCESS == 'ibl':
                results = loader.process_probe(this_probe, filter_ripple_band)  # Use probe_id, not this_probe
            elif DATASET_TO_PROCESS == 'abi':
                results = loader.process_probe(probe_id, filter_ripple_band)  # Use probe_id, not this_probe

            # Skip if no results (no CA1 channels or no bin file)
            if results is None:
                print(f"No results for probe {probe_id}, skipping...")
                continue

            # Extract results
            #lfp_ca1 = results['lfp_ca1']
            peakripple_chan_raw_lfp = results['peak_ripple_chan_raw_lfp']
            lfp_time_index = results['lfp_time_index']
            ca1_chans = results['ca1_chans']
            outof_hp_chans_lfp = results['control_lfps']
            take_two = results['control_channels']
            peakrippleband = results['rippleband']
            this_chan_id = results['peak_ripple_chan_id']

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
                lfp_time_index = lfp_time_index,
            )
            print(f"outof_hp_chans_lfp : {outof_hp_chans_lfp}")
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

            # create a dummy speed vector
            dummy_speed = np.zeros_like(peakrippleband)
            print("Detecting Putative Ripples")
            # we add a dimension to peakrippleband because the ripple detector needs it
            process_stage = f"Detecting Putative Ripples on probe with id {str(probe_id)}"
            
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
            process_stage = f"Detecting Gamma Events on probe with id {str(probe_id)}"
            
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
            process_stage = f"Detecting Movement Artifacts on probe with id {probe_id}"
            
            for i in [0, 1]:
                channel_outside_hp = take_two[i]
                process_stage = f"Detecting Movement Artifacts on control channel {channel_outside_hp} on probe {probe_id}"
                # process control channel ripple times
                ripple_band_control = outof_hp_chans_lfp[i]
                dummy_speed = np.zeros_like(ripple_band_control)
                ripple_band_control = filter_ripple_band(ripple_band_control)
                rip_power_controlchan = np.abs(signal.hilbert(ripple_band_control)) ** 2
                
                print(f"ripple_band_control shape: {ripple_band_control.shape}, length: {len(ripple_band_control)}")
                print(f"lfp_time_index shape: {lfp_time_index.shape}, length: {len(lfp_time_index)}")
                print(f"dummy_speed shape: {dummy_speed.shape}, length: {len(dummy_speed)}")
                
                if DATASET_TO_PROCESS == 'abi':
                    lfp_time_index = lfp_time_index.reshape(-1)
                    dummy_speed = dummy_speed.reshape(-1)
                
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
                channel_outside_hp = "channelsrawInd_" + str(channel_outside_hp)  # no cjannel id in IBL dataset, so this will do instead
                csv_filename = f"probe_{probe_id}_channel_{channel_outside_hp}_movement_artifacts.csv"
                csv_path = os.path.join(session_subfolder, csv_filename)
                movement_controls.to_csv(csv_path, index=True, compression="gzip")
                print("Done Probe id " + str(probe_id))

        # deleting the session folder
        # del one  # so that we can delete the session folder, note sr and ssl need to be deleted as well, already done earlier
        if 'loader' in locals() and loader is not None:
            loader.cleanup()
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


"""
from one.api import ONE
ata = np.load(session_npz_filepath)
all_sesh_with_ca1_eid = data["all_sesh_with_ca1_eid_unique"]
session_id = all_sesh_with_ca1_eid[0]
ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
one = ONE(password="international", silent=True)
print(one.search_terms())
probelist, probenames = one.eid2pid(session_id)
probe_name = probenames[0]
collectionname = f"alf/{probe_name}/pykilosort"
channels = one.load_object(session_id, "channels", collection=collectionname)
channels
"""


# set up the logging.0
MESSAGE = 25  # Define a custom logging level, between INFO (20) and WARNING (30)

# loading filters (crates artifacts in first and last ~ 3.5 seconds of recordings, remember to clip these off)
# I don't think I need this it's at the start of my files
gamma_filter = np.load(gamma_filters_path)
gamma_filter = gamma_filter["arr_0"]

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
"""
queue = Queue()
listener = Process(target=listener_process, args=(queue,))
listener.start()
"""

if DATASET_TO_PROCESS == "abi":
    # If processing Allen data
    data = np.load("allen_visbehave_ca1_session_ids.npz")
    all_sesh_with_ca1_eid = data["data"]
    del data
    print(f"Loaded {len(all_sesh_with_ca1_eid)} sessions from allen_visbehave_ca1_session_ids.npz")
elif DATASET_TO_PROCESS == "ibl":
    # If processing IBL data
    data = np.load(session_npz_filepath)
    all_sesh_with_ca1_eid = data["all_sesh_with_ca1_eid_unique"]
    del data
    print(f"Loaded {len(all_sesh_with_ca1_eid)} sessions from {session_npz_filepath}")


process_session(all_sesh_with_ca1_eid[0])
"""
# run the processes with the specified number of cores:
with Pool(pool_size, initializer=init_pool, initargs=(queue,)) as p:
    p.map(process_session, all_sesh_with_ca1_eid[0:2])

queue.put("kill")
listener.join()
"""