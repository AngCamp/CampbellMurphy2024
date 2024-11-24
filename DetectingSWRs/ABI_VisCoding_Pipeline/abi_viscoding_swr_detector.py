# Allen Visual Behaviour SWR detection script

# libraries
import os
import subprocess
import numpy as np
import pandas as pd
from scipy import signal
import ripple_detection
from ripple_detection import filter_ripple_band
from scipy import stats
from tqdm import tqdm
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from scipy import interpolate
from scipy.signal import firwin, lfilter
from multiprocessing import Pool, Queue, Process
import time
import traceback
import logging
import logging.handlers
import yaml

# start timing
start_time_outer = time.time()  # start timing


# Load the configuration from a YAML file
with open("abi_viscoding_swr_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Get the values from the configuration
pool_size = config["pool_size"]
sdk_cache_dir = config["sdk_cache_dir"]
output_dir = config["output_dir"]
swr_output_dir = config["swr_output_dir"]
run_name = config["run_name"]
select_these_sessions = config["select_these_sessions"]
only_brain_observatory_sessions = config["only_brain_observatory_sessions"]
dont_wipe_these_sessions = config["dont_wipe_these_sessions"]
gamma_event_thresh = config["gamma_event_thresh"]
gamma_filter_path = config["gamma_filter_path"]
theta_filter_path = config["theta_filter_path"]
ripple_band_threshold = config["ripple_band_threshold"]
movement_artifact_ripple_band_threshold = config[
    "movement_artifact_ripple_band_threshold"
]
save_lfp = config["save_lfp"]


# Functions
def call_bash_function(bash_command=""):
    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    if process.returncode == 0:
        print("Bash function executed successfully.")
        print("Output:", output.decode("utf-8"))
    else:
        print("Error:", error.decode("utf-8"))


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
    new_signal = np.zeros((nsamples_new, signal.shape[1]))

    for i in range(signal.shape[1]):
        interp_func = interpolate.interp1d(
            times, signal[:, i], bounds_error=False, fill_value="extrapolate"
        )
        new_signal[:, i] = interp_func(new_times)

    return new_signal, new_times


# Set up error logging
MESSAGE = 25  # Define a custom logging level, between INFO (20) and WARNING (30)


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
        f"abi_detector_{swr_output_dir}_{run_name}_app.log", mode="w"
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


# loading filters (craetes artifacts in first and last ~ 3.5 seconds of recordings, remember to clip these off)
gamma_filter = np.load(gamma_filter_path)
gamma_filter = gamma_filter["arr_0"]

theta_filter = np.load(theta_filter_path)
theta_filter = theta_filter["arr_0"]

# Setting up the ABI Cache (where data is held, what is present or absent)
manifest_path = os.path.join(sdk_cache_dir, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# we start by calling and filtering our dataframe of the sessions we will be working with
sessions = cache.get_session_table()

if only_brain_observatory_sessions:
    sessions = sessions[sessions.session_type == "brain_observatory_1.1"]

if len(select_these_sessions) > 0:
    sessions = sessions.loc[sessions.index.intersection(select_these_sessions)]


# Create main folder
swr_output_dir_path = os.path.join(output_dir, swr_output_dir)
os.makedirs(swr_output_dir_path, exist_ok=True)

if save_lfp == True:
    lfp_output_dir_path = os.path.join(output_dir, swr_output_dir + "_lfp_data")
    os.makedirs(lfp_output_dir_path, exist_ok=True)


# this is the process we will run for each session over multiple cores
def process_session(session_id):
    """
    This function takes in a session id and loops through the probes in that session,
    for each probe it finds the CA1 channel with the highest ripple power and uses that
    channel to detect SWR events.  It also detects gamma events and movement artifacts
    on two channels outside of the brain.

    Parameters
    ----------
    session_id : int
        The session id for the session to be processed.

    Returns
    -------
    None
    but...
    Saves the following files to the folder specified by swr_output_dir_path:
        - a csv file for each probe with the SWR events detected on the CA1 channel with the highest ripple power
        - a csv file for each probe with the gamma events detected on the CA1 channel with the highest ripple power
        - a csv file for each probe with the movement artifacts detected on the two channels outside of the brain
        - a numpy array of the LFP from the CA1 channels used for SWR detection
        - a numpy array of the LFP from the two channels outside of the brain used for movement artifact detection
        - a numpy array of the times of samples in the interpolated LFP (from original rate to 1500 Hz) for all channels used

    Notes:
    - The LFP is interpolated to 1500 Hz for all channels used.
    - The SWR detector used is the Karlsson ripple detector from the ripple_detection module.
    - The folders are titled by session and all files contain the name of the probe and the channel they originated from
    """
    session = cache.get_session_data(session_id)
    print("Starting Session id " + str(session_id))
    probe_id = "Not selected yet"
    # check if this session even has CA1 channels in it, if not skip this iteration and add the name to the list
    sesh_has_ca1 = np.isin(
        "CA1", list(session.channels.ecephys_structure_acronym.unique())
    )
    if not sesh_has_ca1:
        print("Session id " + str(session_id) + "Does not have CA1")
        return  # end the process
    try:
        # timing the probe run
        start_time_sesh = time.time()

        # Create subfolder for session, will contain all csvs for events detected and .npy of ca1 channels and control channels
        session_subfolder = "swrs_session_" + str(session_id)
        session_subfolder = os.path.join(swr_output_dir_path, session_subfolder)
        if os.path.exists(session_subfolder):
            raise FileExistsError(f"The directory {session_subfolder} already exists.")
        else:
            os.makedirs(session_subfolder)

        if save_lfp == True:
            # Create subfolder for lfp data, will contain all npz files for lfp data
            session_lfp_subfolder = "lfp_session_" + str(session_id)
            session_lfp_subfolder = os.path.join(
                lfp_output_dir_path, session_lfp_subfolder
            )
            os.makedirs(session_lfp_subfolder, exist_ok=True)

        # get probes with CA1 recordings out of recording
        probe_id_list = list(session.channels.probe_id.unique())
        probes_of_interest = []

        # find probes which contain channels from CA1

        for probe_id in probe_id_list:
            has_ca1_and_exists = np.isin(
                "CA1",
                list(
                    session.channels[
                        session.channels.probe_id == probe_id
                    ].ecephys_structure_acronym.unique()
                ),
            )
            has_ca1_and_exists = (
                has_ca1_and_exists & session.probes.has_lfp_data[probe_id]
            )
            if has_ca1_and_exists:
                probes_of_interest.append(probe_id)
        # create an arraey to be filled with channel ids fro ca1
        ca1_chans_arr = np.array([], dtype=int)

        # get lfp for each probe
        for probe_id in probes_of_interest:

            # timing the probe...
            start_time_probe = time.time()

            # pull or laod the lfp for this probe
            print("Probe id " + str(probe_id))
            lfp = session.get_lfp(probe_id)

            print("Selecting CA1 channel...")
            # fetching channels in ca1 on this probe for this recording
            ca1_chans = session.channels.probe_channel_number[
                (session.channels.probe_id == probe_id)
                & (session.channels.ecephys_structure_acronym == "CA1")
            ]
            ca1_idx = np.isin(lfp.channel.values, ca1_chans.index.values)
            ca1_idx = lfp.channel.values[ca1_idx]

            # select ca1 channels
            lfp_ca1 = lfp.sel(channel=ca1_idx)
            lfp_ca1 = lfp_ca1.to_pandas()
            lfp_ca1_chans = lfp_ca1.columns
            lfp_ca1 = lfp_ca1.to_numpy()

            # check for nans indicating this is a bad probe
            try:
                if np.isnan(lfp_ca1).any():  # Check if there is any NaN in lfp_ca1
                    del lfp_ca1, lfp  # Delete lfp_ca1 and lfp from memory
                    raise ValueError(
                        f"During session {session_id} processing : Nan in lfp of Probe {probe_id}, probe skipped."
                    )  # Raise error
            except ValueError as e:
                logging.error(e)  # Log the error message, skip to next probe
                del lfp  # Delete lfp from memory to save RAM
                continue

            # get the timestamps for this lfp recording
            # lfp_time_index = lfp_ca1.index.values
            lfp_ca1, lfp_time_index = resample_signal(
                lfp_ca1, lfp.time.values, 1500.0
            )  # note the original samplig rate is infered from the times object

            # identify channel on probe with highest ripple power
            # lfp_ca1_ripppleband = finitimpresp_filter_for_LFP(lfp_ca1, samplingfreq = sampling_rate_this_probe,  lowcut = 120, highcut = 250)
            lfp_ca1_ripppleband = filter_ripple_band(lfp_ca1)
            highest_rip_power = np.abs(signal.hilbert(lfp_ca1_ripppleband)) ** 2
            highest_rip_power = highest_rip_power.max(axis=0)

            # store channel identity in ca1_chans_arr and pull it for analysis of that channel
            this_chan_id = int(lfp_ca1_chans[highest_rip_power.argmax()])

            # ideally we would store the channels for later use, but each lfp has it's own time and sampling rate that it goes through
            # used_channels_xarray_dict[this_chan_id] = lfp.channel.values[this_chan_id]

            # GET LFP FOR ALL CHANNELS NEEDED, DELETE ARRAYS TO CLEAN UP MEMORY
            peakripchan_lfp_ca1 = lfp_ca1[:, lfp_ca1_chans == this_chan_id]

            # as detailed in supplementry methods in Nitzan et al., (2022) on page 2 under Event Detection
            # we will take two control channels from the same probe rather than just one
            # Bute we will take two control channels from the same probe rather than just one
            idx = session.channels.probe_id == probe_id
            organisedprobechans = session.channels[idx].sort_values(
                by="probe_vertical_position"
            )
            organisedprobechans = organisedprobechans[
                np.isin(organisedprobechans.index.values, lfp.channel.values)
            ]

            # code for identifying first  and last ca1 channel, not used now but can be later to pick channels above or below ca1
            # first_ca1 = organisedprobechans.probe_vertical_position[organisedprobechans.ecephys_structure_acronym == 'CA1'].tolist()[-1]
            # last_ca1 = organisedprobechans.probe_vertical_position[organisedprobechans.ecephys_structure_acronym == 'CA1'].tolist()[0]

            not_a_ca1_chan = np.logical_not(
                np.isin(
                    organisedprobechans.ecephys_structure_acronym,
                    ["CA3", "CA2", "CA1", "HPF", "EC", "DG"],
                )
            )

            # Find the indices of the blocks of False i.e. the channels that are ca1
            take_two = np.random.choice(
                organisedprobechans.index[not_a_ca1_chan], 2, replace=False
            )
            control_channels = []

            # movement control
            for channel_outside_hp in take_two:
                movement_control_channel = lfp.sel(channel=channel_outside_hp)
                movement_control_channel = movement_control_channel.to_numpy()
                # select ca1 channels
                interp_func = interpolate.interp1d(
                    lfp.time.values, movement_control_channel
                )
                movement_control_channel = interp_func(lfp_time_index)
                control_channels.append(movement_control_channel)

            # Saving LFP for all channels used
            if save_lfp == True:
                np.savez(
                    os.path.join(
                        session_lfp_subfolder,
                        f"probe_{probe_id}_channel_{this_chan_id}_lfp_ca1_peakripplepower.npz",
                    ),
                    lfp_ca1=peakripchan_lfp_ca1,
                )
                np.savez(
                    os.path.join(
                        session_lfp_subfolder,
                        f"probe_{probe_id}_channel_{this_chan_id}_lfp_time_index_1500hz.npz",
                    ),
                    lfp_time_index=lfp_time_index,
                )
                for i in [0, 1]:
                    channel_outside_hp = str(take_two[i])
                    np.savez(
                        os.path.join(
                            session_lfp_subfolder,
                            f"probe_{probe_id}_channel_{channel_outside_hp}_lfp_control_channel.npz",
                        ),
                        lfp_control_channel=control_channels[i],
                    )

            # clean memory...
            del lfp
            del lfp_ca1

            # COMPUTING LFP EVENTS
            ca1_chans_arr = np.append(ca1_chans_arr, this_chan_id)
            peakrippleband = lfp_ca1_ripppleband[:, highest_rip_power.argmax()]
            # make fake speed variable, we can use this for now and fix it later
            dummy_speed = np.zeros_like(peakrippleband)
            print("Detecting Putative Ripples")
            # we add a dimension to peakrippleband because the ripple detector needs it
            Karlsson_ripple_times = ripple_detection.Karlsson_ripple_detector(
                time=lfp_time_index,
                zscore_threshold=ripple_band_threshold,
                filtered_lfps=peakrippleband[:, None],
                speed=dummy_speed,
                sampling_frequency=1500.0,  # reinterploate to 1500 Hz, for edeno code
            )
            # there is no need for this criteria (Karlsson_ripple_times.duration>0.015)&(Karlsson_ripple_times.duration<0.25)
            # because they are already filtered for minimum duration
            # but we need to do it for maximum duration
            Karlsson_ripple_times = Karlsson_ripple_times[
                Karlsson_ripple_times.duration < 0.25
            ]
            print("Done")
            # adds some stuff we want to the file
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
            # gamma power
            # compute this later, I will have a seperate script called SWR filtering which will do this
            # gamma_band = finitimpresp_filter_for_LFP(lfp_ca1[:,lfp_ca1_chans == this_chan_id], samplingfreq =  1500.0, lowcut = 20, highcut = 80)
            # gamma_band = gamma_band_1500hzsig_filter(interpolated_1500hz_signal = peakripchan_lfp_ca1, filters_path = gamma_filters_paths)
            gamma_band = np.convolve(
                peakripchan_lfp_ca1.reshape(-1), gamma_filter, mode="same"
            )  # reshape is needed to prevent "to deep" error
            gamma_power = np.abs(signal.hilbert(gamma_band)) ** 2
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
            print("Writing to file.")
            print("Selecting reference channel for movement artifact filtering.")

            # as detailed in supplementry methods in Nitzan et al., (2022) on page 2 under Event Detection
            """"
            An additional ‘noise’ signal from a channel outside of the hippocampus was provided to exclude
            simultaneously occurring high frequency events. 
            """

            # movement control
            for i in [0, 1]:
                channel_outside_hp = take_two[i]
                movement_control_channel = control_channels[i]
                movement_control_channel = filter_ripple_band(
                    movement_control_channel[:, None]
                )

                dummy_speed = np.zeros_like(movement_control_channel)

                movement_controls = ripple_detection.Karlsson_ripple_detector(
                    time=lfp_time_index.reshape(-1),
                    filtered_lfps=movement_control_channel,
                    speed=dummy_speed.reshape(-1),
                    zscore_threshold=movement_artifact_ripple_band_threshold,
                    sampling_frequency=1500.0,
                )
                print("Done")
                speed_cols = [
                    col for col in movement_controls.columns if "speed" in col
                ]
                movement_controls = movement_controls.drop(columns=speed_cols)
                csv_filename = f"probe_{probe_id}_channel_{channel_outside_hp}_movement_artifacts.csv"
                csv_path = os.path.join(session_subfolder, csv_filename)
                movement_controls.to_csv(csv_path, index=True, compression="gzip")
                print("Done Probe id " + str(probe_id))

            # write these two to a numpy array, finish loop
            # write channel number and session id to a pandas array tracking where each channel came from

            # At the end of the probe
            end_time_probe = time.time()
            elapsed_time_probe = end_time_probe - start_time_probe
            hours_probe, rem_probe = divmod(elapsed_time_probe, 3600)
            minutes_probe, seconds_probe = divmod(rem_probe, 60)
            print(
                f"Elapsed time for probe: {int(hours_probe)}:{int(minutes_probe)}:{seconds_probe:.2f}"
            )

        print("Done Session id " + str(session_id))
        # loop over global channels

        # removing files
        # replace path/to/directory with cache and session info for this loop
        if session_id not in dont_wipe_these_sessions:
            remove_from_path_command = (
                "find "
                + sdk_cache_dir
                + "/session_"
                + str(session_id)
                + " -type f -name '*lfp*' -exec rm {} +"
            )
            call_bash_function(remove_from_path_command)

        # At the end of the session
        end_time_sesh = time.time()
        elapsed_time_sesh = end_time_sesh - start_time_sesh
        hours_sesh, rem_sesh = divmod(elapsed_time_sesh, 3600)
        minutes_sesh, seconds_sesh = divmod(rem_sesh, 60)
        print(
            f"Elapsed time for session: {int(hours_sesh)}:{int(minutes_sesh)}:{seconds_sesh:.2f}"
        )

    except Exception:

        # if there is an error we want to know about it, but we dont want it to stop the loop
        # so we will print the error to a file and continue
        logging.error("Error in session: %s, probe id: %s", session_id, probe_id)
        logging.error(traceback.format_exc())


queue = Queue()
listener = Process(target=listener_process, args=(queue,))
listener.start()

pool_size = 6

# already filterd for only brain observatory sessions
session_list = sessions.index.values
session_list = session_list  # for testing

# run the processes with the specified number of cores:
with Pool(pool_size, initializer=init_pool, initargs=(queue,)) as p:
    # There is no point in passing queue to this worker function
    # since it is not used (Booboo):
    p.map(process_session, session_list)

queue.put("kill")
listener.join()

print("Done! Results in " + swr_output_dir_path)

# remove any empty directories
# Bash command to find and remove all empty directories
command_empty_swrdir_clear = f"find {swr_output_dir_path} -type d -empty -delete"
command_empty_lfpdir_clear = f"find {lfp_output_dir_path} -type d -empty -delete"

# Run the command
subprocess.run(command_empty_swrdir_clear, shell=True, check=True)
# Run the command
subprocess.run(command_empty_lfpdir_clear, shell=True, check=True)

end_time_outer = time.time()  # end timing
elapsed_time_outer = end_time_outer - start_time_outer  # calculate elapsed time

hours, rem = divmod(elapsed_time_outer, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Elapsed time for whole script: {int(hours)}:{int(minutes)}:{seconds:.2f}")
