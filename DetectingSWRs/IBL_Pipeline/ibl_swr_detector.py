#IBL SWR detector
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
import ripple_detection.simulate as ripsim # for making our time vectors
from tqdm import tqdm
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from one.api import ONE
import spikeglx
from brainbox.io.one import load_channel_locations
from brainbox.io.spikeglx import Streamer
from brainbox.io.one import SpikeSortingLoader
from neurodsp.voltage import destripe_lfp
from neurodsp.voltage import destripe_lfp
from ibllib.plots import Density
import time 
import traceback
import logging
import logging.handlers
import sys
from multiprocessing import Pool, Process, Queue, Manager, set_start_method
from one.api import ONE
import yaml

# Load the configuration from a YAML file
with open('ibl_swr_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get the values from the configuration
pool_size = config['pool_size']
gamma_filters_path = config['gamma_filters_path']
run_name = config['run_name']
oneapi_cache_dir = config['oneapi_cache_dir']
output_dir = config['output_dir']
swr_output_dir = config['swr_output_dir']
gamma_event_thresh = config['gamma_event_thresh']
ripple_band_threshold = config['ripple_band_threshold']
movement_artifact_ripple_band_threshold = config['movement_artifact_ripple_band_threshold']
dont_wipe_these_sessions = config['dont_wipe_these_sessions']
session_npz_filepath = config['session_npz_filepath']
save_lfp = config['save_lfp']

# FUNCTIONS
# subprocess is a default module
def call_bash_function(bash_command = ""):
    #example bash comand:
    #bash_command = "source /path/to/your/bash_script.sh && your_bash_function"
    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    if process.returncode == 0:
        print("Bash function executed successfully.")
        print("Output:", output.decode('utf-8'))
    else:
        print("Error:", error.decode('utf-8'))

# Assuming you have your signal_array, b, and a defined as before
def finitimpresp_filter_for_LFP(LFP_array, samplingfreq, lowcut = 1, highcut = 250,
                    filter_order = 101):
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
    fir_coeff = signal.firwin(filter_order, [lowcut / nyquist, highcut / nyquist],
                              pass_zero=False, fs=samplingfreq)

    # Apply the FIR filter to your signal_array
    #filtered_signal = signal.convolve(LFP_array, fir_coeff, mode='same', method='auto')
    filtered_signal = signal.lfilter(fir_coeff, 1.0, LFP_array, axis=0)
    return(filtered_signal)


def event_boundary_detector(time, five_to_fourty_band_power_df, envelope=True, minimum_duration = 0.02, maximum_duration = 0.4,
                       threshold_sd=2.5, envelope_threshold_sd=1):
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
    row_of_info =  {
        'start_time': [],
        'end_time': [],
        'duration': [],
        }

    #sharp_wave_events_df = pd.DataFrame()
    #scored_wave_power = stats.zscore(five_to_fourty_band_df)
    
    # compute our power threshold
    #wave_band_sd_thresh = np.std(five_to_fourty_band_df)*threshold_sd
    five_to_fourty_band_power_df = stats.zscore(five_to_fourty_band_power_df)
    past_thresh = five_to_fourty_band_power_df>=threshold_sd
    
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
            while start > 0 and z_scores[start - 1] >  thresh:
                start -= 1

            # Expand section to the right (while meeting conditions)
            while end < len(z_scores) - 1 and z_scores[end + 1] >  thresh:
                end += 1

            # Check if the expanded section contains a point above envelope_threshold_sd in z_scores
            if any(z_scores[start:end + 1] >  thresh):
                expanded_sections[start:end + 1] = True

        # Update the boolean_array based on expanded_sections
        boolean_array = boolean_array | expanded_sections

        return boolean_array
    
    if envelope==True:
        past_thresh = expand_sections(z_scores=five_to_fourty_band_power_df,
                                  boolean_array= past_thresh,
                                  thresh = envelope_threshold_sd)
    
    
    # Find the indices where consecutive True values start
    starts = np.where(past_thresh & ~np.roll(past_thresh, 1))[0]
    row_of_info['start_time'] = time[starts]
    # Find the indices where consecutive True values end
    ends = np.where(past_thresh & ~np.roll(past_thresh, -1))[0]
    row_of_info['end_time'] = time[ends]
    
    row_of_info['duration'] = [row_of_info['end_time'][i]-row_of_info['start_time'][i] for i in range(0,len(row_of_info['start_time']))]
    
    #turn the dictionary into adataframe
    sharp_wave_events_df = pd.DataFrame(row_of_info)
    
    # filter for the duration range we want
    in_duration_range = (sharp_wave_events_df.duration>minimum_duration)&(sharp_wave_events_df.duration<maximum_duration)
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
    row_of_info['start_time'] = time[starts]
    # Find the indices where consecutive True values end
    ends = np.where(past_thresh & ~np.roll(past_thresh, -1))[0]
    row_of_info['end_time'] = time[ends]
    
    row_of_info['duration'] = [row_of_info['end_time'][i]-row_of_info['start_time'][i] for i in range(0,len(row_of_info['start_time']))]
    
    #turn the dictionary into adataframe
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
    for start, end in zip(events['start_time'], events['end_time']):
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
        interp_func = interpolate.interp1d(times, signal[i,:], bounds_error=False, fill_value="extrapolate")
        new_signal[i,:] = interp_func(new_times)

    return new_signal, new_times


# set up the logging
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
    h = logging.FileHandler(f'ibl_detector_{swr_output_dir}_{run_name}_app.log', mode='w')
    f = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    h.setFormatter(f)
    root.addHandler(h)
    root.setLevel(MESSAGE)  # Set logging level to MESSAGE

    while True:
        message = queue.get()
        if message == 'kill':
            break
        logger = logging.getLogger(message.name)
        logger.handle(message)

def init_pool(*args):
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(MESSAGE)  # Set logging level to MESSAGE


# loading filters (crates artifacts in first and last ~ 3.5 seconds of recordings, remember to clip these off)
gamma_filter = np.load(gamma_filters_path)
gamma_filter = gamma_filter['arr_0']

# load in the brain atlas and the brain region object for working with the ccf and ABI region id's in channels objects
ba = AllenAtlas()
br = BrainRegions() # br is also an attribute of ba so could to br = ba.regions

#Searching for datasets
brain_acronym = 'CA1'
# query sessions endpoint
#sessions, sess_details = one.search(atlas_acronym=brain_acronym, query_type='remote', details=True)

swr_output_dir_path = os.path.join(output_dir, swr_output_dir)
os.makedirs(swr_output_dir_path, exist_ok=True)
sessions_without_ca1 = np.array([])
#from multiprocessing import Pool

if save_lfp == True:
    lfp_output_dir_path = os.path.join(output_dir, swr_output_dir+'_lfp_data')
    os.makedirs(lfp_output_dir_path, exist_ok=True)

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

    Process used to process a single session insode a multiprocessing call.
    Remember to set up the brain atlas first:
       ba = AllenAtlas()
       br = BrainRegions()
    """
    process_stage = "Starting the process" # for debugging, to see where the process is at in the log
    probe_id = 'Not Loaded Yet' # needs to be created here outside of the try statement so it can be checked in the except statement
    one_exists = False # needs to be created here outside of the try statement so it can be checked in the except statement
    data_files = None  # Initialize data_files to None
    
    # Create a string for the subfolder for the session, needs to be done here otherwise an error happens in except block
    session_subfolder = "swrs_session_" + str(session_id)
    session_subfolder = os.path.join(swr_output_dir_path, session_subfolder)
    try:
        process_stage = "Loading the session from ONE, getting probe ids and names"
        ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
        one = ONE(password='international')
        one_exists = True # this is used for the exception to avoid an error when deleting the session folder
        #one = ONE(password='international') # restarting one for this loop
        eid = session_id # just to keep the naming similar to the IBL example scripts, bit silly but helps to write the code
        probelist, probenames = one.eid2pid(eid) # probe_id is pid in the IBL tutorials
        
        print(f'Probe IDs: {probelist}, Probe names: {probenames}')
        
        band = 'lf' # either 'ap','lf'
        process_stage = "Session loaded, checking if already directory exists"
        # Create subfolder for session, will contain all csvs for events detected and .npy of ca1 channels and control channels 
        if os.path.exists(session_subfolder):
            raise FileExistsError(f"The directory {session_subfolder} already exists.")
        else:
            os.makedirs(session_subfolder)
            
        if save_lfp == True:
            # Create subfolder for lfp data, will contain all npz files for lfp data
            session_lfp_subfolder = "lfp_session_" + str(session_id)
            session_lfp_subfolder = os.path.join(lfp_output_dir_path, session_lfp_subfolder)
            os.makedirs(session_lfp_subfolder, exist_ok=True)
        
        process_stage = "Running through the probes in the session"
        #for probe_id in pid:
        for this_probe in range(len(probelist)):
            probe_name = probenames[this_probe]
            probe_id = probelist[this_probe]
            print(probe_id)
            # first check if this probe even has CA1 channels on it, no need to process if not
            print("getting channels data")
            process_stage = f"Getting channels data for probe {probe_name}, with id : {probe_id}"
            collectionname = f'alf/{probe_name}/pykilosort' # ensures channels are all from this probe
            channels = one.load_object(eid, 'channels', collection=collectionname)
            channels.allen2017_25um_acronym = br.id2acronym( channels['brainLocationIds_ccf_2017'] )
            
            regions_on_probe = np.unique(channels.allen2017_25um_acronym)
            if 'CA1' not in regions_on_probe:
                print(f'No CA1 channels on probe {probe_id}, skipping...')
                continue
            process_stage = f"Getting LFP data for the probe {probe_name} with id {probe_id}"
            # Find the relevant datasets and download them
            dsets = one.list_datasets(eid, collection=f'raw_ephys_data/{probe_name}', filename='*.lf.*')
            print(type(dsets))
            print(len(dsets))
            data_files, _ = one.load_datasets(eid, dsets, download_only=False)
            bin_file = next(df for df in data_files if df.suffix == '.cbin')
            
            # Use spikeglx reader to read in the whole raw data
            print("sr = spikeglx.Reader(bin_file)")
            start_time = time.time()
            sr = spikeglx.Reader(bin_file)

            # make time index
            ssl = SpikeSortingLoader(pid=probe_id, one=one)
            t0 = ssl.samples2times(0, direction='forward') # get the time of the first sample
            dt = (ssl.samples2times(1, direction='forward') - t0)*12 # get the time difference between samples at 2500Hz
            lfp_time_index_og = np.arange(0, sr.shape[0])*dt + t0 # get the time index for the LFP data
            del ssl 
            print(f"done, time elapsed: {time.time() - start_time}")
            # Important: remove sync channel from raw data, and transpose
            print("raw = sr[:, :-sr.nsync].T")
            start_time = time.time()
            raw = sr[:, :-sr.nsync].T
            print(f"done, time elapsed: {time.time() - start_time}")
            # Reminder : If not done before, remove first the sync channel from raw data
            # Apply destriping algorithm to data
            fs_from_sr = sr.fs
            del sr
            # code to be used in final version but in the meantime we will load in the destriped data from the saved folder just to save time
            print("destriped = destripe(raw, fs=sr.fs)")
            process_stage = f"Destriping the LFP data for probe {probe_name} with id {probe_id}"
            start_time = time.time()
            destriped = destripe_lfp(raw, fs=fs_from_sr)
            del raw

            print(f"destripped shape : {destriped.shape}")
            print(f"done, time elapsed: {time.time() - start_time}")
            
            # You can access your saved array with the keyword you used while saving
            #destriped = data['destriped']
            print("destriped loaded...")
            
            #GETTING THE CA1 CHANNEL WITH PEAK RIPPLE POWER AND CONTROL CHANNELS
            # get hippocampal channel
            ca1_chans = channels.rawInd[channels.allen2017_25um_acronym == 'CA1']
            
            # select ca1 channels 
            lfp_ca1  = destriped[ca1_chans,:]
            process_stage = f"Destriping done, resampling {probe_name} with id {probe_id} to 1.5kHz "
            # interpolate to 1500 Hz
            lfp_ca1, lfp_time_index = resample_signal(lfp_ca1, lfp_time_index_og, 1500.0) # note the original samplig rate is infered from the times object
            lfp_ca1 = lfp_ca1.T # transpose to get the right shape for the rest of the code
            
            lfp_ca1_ripppleband = 
            highest_rip_power = np.abs(signal.hilbert(lfp_ca1_ripppleband))**2
            highest_rip_power = highest_rip_power.max(axis=0)
            
            # ideally we would store the channels for later use, but each lfp has it's own time and sampling rate that it goes through
            # note channels do not have unique ids in the IBL unlike in the allen institute so instead we use their rawInd value
            this_chan_id = "channelsrawInd_"+str(ca1_chans[highest_rip_power.argmax()]) # for naming the file later, this way we know what object its from so we dont go looking for an id
            peakrippleband = lfp_ca1_ripppleband[:,highest_rip_power.argmax()]
            
            peakripple_chan_raw_lfp = lfp_ca1[:,highest_rip_power.argmax()]
            
            # filter it to gamma band
            gamma_band_ca1 = np.convolve(peakripple_chan_raw_lfp.reshape(-1), gamma_filter, mode='same') # reshape is needed to prevent "to deep" error
            
            # get the control channels here, store their index so you can name them later
            not_a_ca1_chan = np.logical_not(np.isin(channels.allen2017_25um_acronym,[ "CA3", "CA2", "CA1", "HPF", "EC", "DG"]))

            # Find the indices of the blocks of False i.e. the channels that are ca1
            take_two = np.random.choice(channels.rawInd [not_a_ca1_chan], 2, replace=False)
            outof_hp_chans_lfp = []
            
            process_stage = f"Resampling done selecting non Hp control channels on probe {probe_name} with id {probe_id} "
            # making list to hold the arrays of the control channels
            for channel_outside_hp in take_two:
                lfp_control = destriped[channel_outside_hp,:]
                interp_func = interpolate.interp1d(lfp_time_index_og, lfp_control, bounds_error=False, fill_value="extrapolate")
                lfp_control = interp_func(lfp_time_index)
                lfp_control = lfp_control.T # transpose to get the right shape for the rest of the code
                
                outof_hp_chans_lfp.append(lfp_control[:,None])
                # clean it up
                del lfp_control

            del destriped # clear up some memory
            
            # write our lfp to file
            np.savez(os.path.join(session_lfp_subfolder, f"probe_{probe_id}_channel_{this_chan_id}_lfp_ca1_peakripplepower.npz"), lfp_ca1 = peakripple_chan_raw_lfp)
            np.savez(os.path.join(session_lfp_subfolder, f"probe_{probe_id}_channel_{this_chan_id}_lfp_time_index_1500hz.npz"), lfp_time_index = lfp_time_index)
            for i in range(2):
                channel_outside_hp = take_two[i]
                channel_outside_hp = "channelsrawInd_"+ str(channel_outside_hp)
                np.savez(os.path.join(session_lfp_subfolder, f"probe_{probe_id}_channel_{channel_outside_hp}_lfp_control_channel.npz"), lfp_control_channel = outof_hp_chans_lfp[i])
            
            del lfp_ca1 # clear up some memory
            
            # create a dummy speed vector             
            dummy_speed = np.zeros_like(peakrippleband)
            print("Detecting Putative Ripples")
            # we add a dimension to peakrippleband because the ripple detector needs it
            process_stage = f"Detecting Putative Ripples on probe {probe_name} with id {probe_id}"
            Karlsson_ripple_times = ripple_detection.Karlsson_ripple_detector(
                time = lfp_time_index, 
                zscore_threshold= ripple_band_threshold,
                filtered_lfps = peakrippleband[:,None], 
                speed = dummy_speed, 
                sampling_frequency = 1500.0
            )

            Karlsson_ripple_times = Karlsson_ripple_times[Karlsson_ripple_times.duration<0.25]
            print("Done")
            # adds some stuff we want to the file
            
            # ripple band power
            peakrippleband_power = np.abs(signal.hilbert(peakrippleband))**2
            Karlsson_ripple_times['Peak_time'] = peaks_time_of_events(events=Karlsson_ripple_times, 
                                                                 time_values=lfp_time_index, 
                                                                signal_values=peakrippleband_power)
            speed_cols = [col for col in Karlsson_ripple_times.columns if 'speed' in col]
            Karlsson_ripple_times = Karlsson_ripple_times.drop(columns=speed_cols)
            csv_filename = f"probe_{probe_id}_channel_{this_chan_id}_karlsson_detector_events.csv"
            csv_path = os.path.join(session_subfolder, csv_filename)
            Karlsson_ripple_times.to_csv(csv_path, index=True, compression='gzip')
            print("Writing to file.")
            print("Detecting gamma events.")

            # compute this later, I will have a seperate script called SWR filtering which will do this
            process_stage = f"Detecting Gamma Events on probe {probe_name} with id {probe_id}"
            gamma_power = np.abs(signal.hilbert(gamma_band_ca1))**2
            gamma_times = event_boundary_detector(time = lfp_time_index, threshold_sd = gamma_event_thresh, envelope=False, 
                                        minimum_duration = 0.015, maximum_duration = float('inf'),
                                    five_to_fourty_band_power_df = gamma_power)
            print("Done")
            csv_filename = f"probe_{probe_id}_channel_{this_chan_id}_gamma_band_events.csv"
            csv_path = os.path.join(session_subfolder, csv_filename)
            gamma_times.to_csv(csv_path, index=True, compression='gzip')
            
            # movement artifact detection
            process_stage = f"Detecting Movement Artifacts on probe {probe_name} with id {probe_id}"
            for i in [0,1]:
                channel_outside_hp = take_two[i]
                process_stage = f"Detecting Movement Artifacts on control channel {channel_outside_hp} on probe {probe_name} with id {probe_id}"
                # process control channel ripple times
                ripple_band_control = outof_hp_chans_lfp[i]
                
                ripple_band_control = filter_ripple_band(ripple_band_control)
                rip_power_controlchan = np.abs(signal.hilbert(ripple_band_control))**2
                movement_controls = ripple_detection.Karlsson_ripple_detector(
                    time = lfp_time_index, # if this doesnt work try adding .reshape(-1)
                    filtered_lfps = rip_power_controlchan, # indexing [:,None] is not needed here, rip_power_controlchan is already 2d (nsamples, 1)
                    speed = dummy_speed, # if this doesnt work try adding .reshape(-1)
                    zscore_threshold= movement_artifact_ripple_band_threshold,
                    sampling_frequency = 1500.0
                )
                speed_cols = [col for col in movement_controls.columns if 'speed' in col]
                movement_controls = movement_controls.drop(columns=speed_cols)
                # write to file name
                channel_outside_hp = "channelsrawInd_"+ str(channel_outside_hp) # no cjannel id in IBL dataset, so this will do instead
                csv_filename = f"probe_{probe_id}_channel_{channel_outside_hp}_movement_artifacts.csv"
                csv_path = os.path.join(session_subfolder, csv_filename)
                movement_controls.to_csv(csv_path, index=True, compression='gzip')
                print("Done Probe id " + str(probe_id))
                
        # deleting the session folder
        del one # so that we can delete the session folder, note sr and ssl need to be deleted as well, already done earlier
        process_stage = "All processing done, Deleting the session folder"
        one_exists = False
        # Get the file path of the session folder trim it and then use it to delete the folder
        s = str(data_files[0])

        # Find the index of "raw_ephys_data" in the string
        index = s.find('raw_ephys_data')

        # Remove everything after "raw_ephys_data" including that string
        s = s[:index]

        # Remove the substring "PosixPath('"
        s = s.replace("PosixPath('", '')

        # Remove the trailing slash
        s = s.rstrip('/')

        # Define the bash command to delete the folder
        cmd = f'rm -r {s}'

        # Execute the bash command
        os.system(cmd)
        
        # in the session
        logging.log(MESSAGE, f'Processing complete for id {session_id}.')
    except Exception:
        
        # we still need to clear the session
        # if an error occured in deleting the session folder one will have already been deleted
        if one_exists:
            del one # so that we can delete the session folder, note sr and ssl need to be deleted as well, already done earlier
        
        if data_files is not None:
            # to avoid an exception where data files have not been created yet
            # Get the file path of the session folder trim it and then use it to delete the folder
            s = str(data_files[0])

            # Find the index of "raw_ephys_data" in the string
            index = s.find('raw_ephys_data')

            # Remove everything after "raw_ephys_data" including that string
            s = s[:index]

            # Remove the substring "PosixPath('"
            s = s.replace("PosixPath('", '')

            # Remove the trailing slash
            s = s.rstrip('/')

            # Define the bash command to delete the folder
            cmd = f'rm -rf {s}'

            # Execute the bash command
            os.system(cmd)
        # Check if the session subfolder is empty
        if os.path.exists(session_subfolder) and not os.listdir(session_subfolder):
            # If it is, delete it
            os.rmdir(session_subfolder)
            logging.log(MESSAGE,'PROCESSING FAILED REMOVING EMPTY SESSION SWR DIR :  session id %s ', session_id)
        # if there is an error we want to know about it, but we dont want it to stop the loop
        # so we will print the error to a file and continue
        logging.error('Error in session: %s, probe id: %s, Process Error at : ', session_id, probe_id, process_stage)
        logging.error(traceback.format_exc())

queue = Queue()
listener = Process(target=listener_process, args=(queue,))
listener.start()

data = np.load(session_npz_filepath)
all_sesh_with_ca1_eid = data['all_sesh_with_ca1_eid_unique']
del data
all_sesh_with_ca1_eid[4:7]

# run the processes with the specified number of cores:
with Pool(pool_size, initializer=init_pool, initargs=(queue,)) as p:
    p.map(process_session, all_sesh_with_ca1_eid)

queue.put('kill')
listener.join()
