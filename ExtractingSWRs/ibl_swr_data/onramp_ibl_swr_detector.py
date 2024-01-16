#IBL SWR detector
import os
import subprocess 
import numpy as np
import pandas as pd
from scipy import io, signal, stats
from scipy.signal import lfilter
#from fitter import Fitter, get_common_distributions, get_distributions
import scipy.ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
import matplotlib.pyplot as plt
# for ripple detection
import ripple_detection
from ripple_detection import filter_ripple_band
import ripple_detection.simulate as ripsim # for making our time vectors
import piso #can be difficult to install, https://piso.readthedocs.io/en/latest/
from tqdm import tqdm
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from one.api import ONE
import spikeglx
from brainbox.io.one import load_channel_locations
from brainbox.io.spikeglx import Streamer
from neurodsp.voltage import destripe_lfp

#THIS CODE WORKS THIS CODE LOOPS THROUGH THE SESSIONS AND DOWNLOADS THE DATA, WE NEED TO ADD THE RIPPLE DETECTION CODE TO REMOVE THE DATA AFTER 
from neurodsp.voltage import destripe_lfp
from ibllib.plots import Density
import time # for debugging
import traceback
import logging

from multiprocessing import Pool, Process, Queue

from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')
# Parameters, (input output file paths and thresholds)


pool_size =3
gamma_filters_path = ["/home/acampbell/Stienmetz2019Reanalyzed/ExtractingSWRs/PowerBandFilters/Gamma_Band_withlowpass_Filter/lowpass_prefilter.npy",
                                                "/home/acampbell/Stienmetz2019Reanalyzed/ExtractingSWRs/PowerBandFilters/Gamma_Band_withlowpass_Filter/gamma_band_filter.npy"]

# cache location
oneapi_cache_dir = '/space/scratch/IBL_data_cache'# path to where the cache for the allensdk is (wehre the lfp is going)
output_dir = '/space/scratch/IBL_swr_data'
swr_output_dir = 'IBL_swr_2sd_envelope' # directory specifying the output

# THRESHOLDS
gamma_event_thresh = 3 # zscore threshold for gamma events
ripple_band_threshold = 2 # note this defines the threshold for envelopes, from these events identify ones with peaks that pass a peak-power threshold as well
movement_artifact_ripple_band_threshold = 2

# see the notebook SearchingForSubsetforTesting.ipynb to see how this list was generated, this will be our testing list
# first fourteen are from Fingling et al., (2023) via..
# one.load_cache(tag = '2022_Q2_IBL_et_al_RepeatedSite',)
# sessions_rep_sites = one.search()
# then clear the cache...

testing_list = ['0c828385-6dd6-4842-a702-c5075f5f5e81','111c1762-7908-47e0-9f40-2f2ee55b6505','8a3a0197-b40a-449f-be55-c00b23253bbf','1a507308-c63a-4e02-8f32-3239a07dc578','1a507308-c63a-4e02-8f32-3239a07dc578','73918ae1-e4fd-4c18-b132-00cb555b1ad2',
 '73918ae1-e4fd-4c18-b132-00cb555b1ad2','09b2c4d1-058d-4c84-9fd4-97530f85baf6','5339812f-8b91-40ba-9d8f-a559563cc46b','034e726f-b35f-41e0-8d6c-a22cc32391fb','83e77b4b-dfa0-4af9-968b-7ea0c7a0c7e4','83e77b4b-dfa0-4af9-968b-7ea0c7a0c7e4','931a70ae-90ee-448e-bedb-9d41f3eda647',
 'd2832a38-27f6-452d-91d6-af72d794136c','dda5fc59-f09a-4256-9fb5-66c67667a466','e2b845a1-e313-4a08-bc61-a5f662ed295e','a4a74102-2af5-45dc-9e41-ef7f5aed88be','572a95d1-39ca-42e1-8424-5c9ffcb2df87','781b35fd-e1f0-4d14-b2bb-95b7263082bb',
 'b01df337-2d31-4bcc-a1fe-7112afd50c50','e535fb62-e245-4a48-b119-88ce62a6fe67','614e1937-4b24-4ad3-9055-c8253d089919','7f6b86f9-879a-4ea2-8531-294a221af5d0','824cf03d-4012-4ab1-b499-c83a92c5589e','4b00df29-3769-43be-bb40-128b1cba6d35','ff96bfe1-d925-4553-94b5-bf8297adf259']

#dont_wipe_these_sessions =['0c828385-6dd6-4842-a702-c5075f5f5e81']
# testing_list = np.load('testing_list.npy')

dont_wipe_these_sessions =[]

# query insertions endpoint
#insertions = one.search_insertions(atlas_acronym=brain_acronym)
#session_list = [x for x in sessions] # when we need to loop through all the sessions
session_list = testing_list
dont_wipe_these_sessions =[]



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
    Power threshold event detector, includes an envelope as well if wanted
    
    Originally for detecting sharp waves in the striatum radiatum, takes in power signal from 
    
    From Fernández-Ruiz, A., Oliva, A., Fermino de Oliveira, E., Rocha-Almeida, F., Tingley, D., 
    & Buzsáki, G. (2019). Long-duration hippocampal sharp wave ripples improve memory. Science, 364(6445), 1082-1086.
    
    
    Sharp waves were detected separately using LFP from a CA1 str. radiatum channel, filtered with band-pass filter boundaries
   (5-40 Hz). LFP events of a minimum duration of 20 ms and maximum 400 ms exceeding 2.5 SD of the
   background signal were included as candidate SPWs. Only if a SPW was simultaneously detected with
   a ripple, a CA1 SPW-R event was retained for further analysis. SPW-R bursts were classified when more
   than one event was detected in a 400 ms time window.
    
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
    finds the times of a vector of true statements and returns values from another
    array representing the times
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

def peaks_in_events(events, time_values, signal_values):
    # looks for the peask in the lfp signal, value of zscored raw lfp peak and returns time of peak
    signal_values_zscore = stats.zscore(signal_values)
    max_values = []
    max_lfp_zscore_values = []
    peak_times = []
    for start, end in zip(events['start_time'], events['end_time']):
        window_idx = (time_values >= start) & (time_values <= end)
        ripplesignal = signal_values[window_idx]
        ripple_lfp_zscore_signal = signal_values_zscore[window_idx]
        maxpoint = np.argmax(ripplesignal)
        max_values.append(ripplesignal[maxpoint])
        max_lfp_zscore_values.append(ripple_lfp_zscore_signal[maxpoint])
        rippletimepoints = time_values[window_idx]
        peak_times.append(rippletimepoints[maxpoint])
    return np.array(max_values), np.array(max_lfp_zscore_values),  np.array(peak_times)

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

def gamma_band_1500hzsig_filter(interpolated_1500hz_signal, 
                                filters_path = ["/home/acampbell/Stienmetz2019Reanalyzed/ExtractingSWRs/PowerBandFilters/Gamma_Band_withlowpass_Filter/lowpass_prefilter.npy",
                                                "/home/acampbell/Stienmetz2019Reanalyzed/ExtractingSWRs/PowerBandFilters/Gamma_Band_withlowpass_Filter/gamma_band_filter.npy"]):
    """
    Takes in a signal interpolated to 1500 Hz and returns the signal filtered in the gamma band, using filters designed for 1500 Hz.
    
    Parameters
    ----------
    interpolated_1500hz_signal : array_like
        The signal interpolated to 1500 Hz.
    filters_path : list of str
        The path to the filters to be used for filtering the signal.
        
    Returns
    -------
    bandpassed_signal : array_like
        The filtered signal.
        
    Notes:    
    How filters were made:
        # Define the order of the low-pass filter
        numtaps = 101

        # Define the cutoff frequency (in Hz) for the low-pass filter
        cutoff_hz = 625.0
        # Create the low-pass filter
        low_pass_taps = firwin(numtaps, cutoff_hz/(0.5*1500), window='hamming')
    
        # Now create your bandpass filter
        bandpass_taps = make_bandpass_filter(sampling_frequency=1500, BAND_OF_INTEREST = [20, 80],
                                TRANSITION_BAND = 10, ORDER = 250)
                                
    
    """
    # Create the low-pass filter
    low_pass_taps = np.load(filters_path[0])

    # Apply the low-pass filter to your signal
    low_passed_signal = lfilter(low_pass_taps, 1.0, interpolated_1500hz_signal)

    # Now create your bandpass filter
    bandpass_taps = np.load(filters_path[1])

    # Apply the bandpass filter to the low-passed signal
    bandpassed_signal = lfilter(bandpass_taps, 1.0, low_passed_signal)
    
    return bandpassed_signal

def listener_process(queue):
    """ 
    This function is run by the listener process. It listens for messages on the queue
    and writes them to the log file.
    
    Parameters
    ----------
    queue : multiprocessing.Queue
        The queue to listen on.
        
    Returns
    -------
    None
    
    """
    # Configure the logger
    logging.basicConfig(filename='abi_detector_app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    while True:
        message = queue.get()
        if message == 'kill':
            break
        logging.error(message)


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
    try:
        #one = ONE(password='international') # restarting one for this loop
        eid = session_id # just to keep the naming similar to the IBL example scripts, bit silly but helps to write the code
        probelist, probenames = one.eid2pid(eid) # probe_id is pid in the IBL tutorials
        
        print(f'Probe IDs: {probelist}, Probe names: {probenames}')
        
        band = 'lf' # either 'ap','lf'
        
        # Create subfolder for session, will contain all csvs for events detected and .npy of ca1 channels and control channels 
        session_subfolder = "swrs_session_" + str(session_id)
        session_subfolder = os.path.join(swr_output_dir_path, session_subfolder)
        os.makedirs(session_subfolder, exist_ok=True)
        
        
        #for probe_id in pid:
        for this_probe in range(len(probelist)):
            probe_name = probenames[this_probe]
            probe_id = probelist[this_probe]
            print(probe_id)
            # first check if this probe even has CA1 channels on it, no need to process if not
            print("getting channels data")
            collectionname = f'alf/{probe_name}/pykilosort' # ensures channels are all from this probe
            channels = one.load_object(eid, 'channels', collection=collectionname)
            channels.allen2017_25um_acronym = br.id2acronym( channels['brainLocationIds_ccf_2017'] )
            
            regions_on_probe = np.unique(channels.allen2017_25um_acronym)
            if 'CA1' not in regions_on_probe:
                print(f'No CA1 channels on probe {probe_id}, skipping...')
                continue
                    
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
            print(f"done, time elapsed: {time.time() - start_time}")
            # Important: remove sync channel from raw data, and transpose
            print("raw = sr[:, :-sr.nsync].T")
            start_time = time.time()
            raw = sr[:, :-sr.nsync].T
            print(f"done, time elapsed: {time.time() - start_time}")
            # Reminder : If not done before, remove first the sync channel from raw data
            # Apply destriping algorithm to data
            
            # code to be used in final version but in the meantime we will load in the destriped data from the saved folder just to save time
            print("destriped = destripe(raw, fs=sr.fs)")
            start_time = time.time()
            destriped = destripe_lfp(raw, fs=sr.fs)
            del raw
            print(f"destripped shape : {destriped.shape}")
            print(f"done, time elapsed: {time.time() - start_time}")
            
            # just for debugging
            # np.load function will load the .npz file
            #data = np.load("/space/scratch/test_destripe_save/debugging_destriped_save.npz")

            # You can access your saved array with the keyword you used while saving
            #destriped = data['destriped']
            print("destriped loaded...")


            
            #GETTING THE CA1 CHANNEL WITH PEAK RIPPLE POWER AND CONTROL CHANNELS
            # get hippocampal channel
            ca1_chans = channels.rawInd[channels.allen2017_25um_acronym == 'CA1']
            
            # select ca1 channels 
            lfp_ca1  = destriped[ca1_chans,:]
            
            # get the timestamps for this lfp recording
            #2 columns file containing time synchronisation information for the AP binary file: 
            # sample index in the first column and session time in the second column. Note that sample indices may not be integers
            lfp_time_timestamps_path = one.list_datasets(eid, collection=f'raw_ephys_data/{probe_name}', filename='*timestamps.npy')
            lfp_time_timestamps, _ = one.load_datasets(eid, lfp_time_timestamps_path, download_only=False)

            # Divide the first column by 12
            lfp_time_timestamps = lfp_time_timestamps[1]

            adjusted_samples = lfp_time_timestamps[:, 0] / 12

            # Create an array of sample numbers for your signal
            sample_numbers = np.arange(destriped.shape[1])

            # Interpolate to get lfp_time_index_og
            lfp_time_index_og = np.interp(sample_numbers, adjusted_samples, lfp_time_timestamps[:, 1])
            lfp_ca1, lfp_time_index = resample_signal(lfp_ca1, lfp_time_index_og, 1500.0) # note the original samplig rate is infered from the times object
            lfp_ca1 = lfp_ca1.T # transpose to get the right shape for the rest of the code
            
            lfp_ca1_ripppleband = filter_ripple_band(lfp_ca1)
            highest_rip_power = np.abs(signal.hilbert(lfp_ca1_ripppleband))**2
            highest_rip_power = highest_rip_power.max(axis=0)
            
            # ideally we would store the channels for later use, but each lfp has it's own time and sampling rate that it goes through
            # note channels do not have unique ids in the IBL unlike in the allen institute so instead we use their rawInd value
            this_chan_id = "channelsrawInd_"+str(ca1_chans[highest_rip_power.argmax()]) # for naming the file later, this way we know what object its from so we dont go looking for an id
            peakrippleband = lfp_ca1_ripppleband[:,highest_rip_power.argmax()]
            
            peakripple_chan_raw_lfp = lfp_ca1[:,highest_rip_power.argmax()]
            del lfp_ca1 # clear up some memory
            # filter it to gamma band
            gamma_band_ca1 = gamma_band_1500hzsig_filter(peakripple_chan_raw_lfp, filters_path = gamma_filters_path)
            
            # get the control channels here, store their index so you can name them later
            not_a_ca1_chan = np.logical_not(np.isin(channels.allen2017_25um_acronym,[ "CA3", "CA2", "CA1", "HPF", "EC", "DG"]))

            # Find the indices of the blocks of False i.e. the channels that are ca1
            take_two = np.random.choice(channels.rawInd [not_a_ca1_chan], 2, replace=False)
            outof_hp_chans_lfp = []
            
            # making list to hold the arrays of the control channels
            for channel_outside_hp in take_two:
                lfp_control = destriped[channel_outside_hp,:]
                interp_func = interpolate.interp1d(lfp_time_index_og, lfp_control, bounds_error=False, fill_value="extrapolate")
                lfp_control = interp_func(lfp_time_index)
                lfp_control = lfp_control.T # transpose to get the right shape for the rest of the code
                
                
                outof_hp_chans_lfp.append(lfp_control[:,None])
                # clean it up
                del lfp_control

            # write our lfp to file
            np.savez(os.path.join(session_subfolder, f"probe_{probe_id}_channel_{this_chan_id}_lfp_ca1_peakripplepower.npz"), lfp_ca1 = peakripple_chan_raw_lfp)
            np.savez(os.path.join(session_subfolder, f"probe_{probe_id}_channel_{this_chan_id}_lfp_time_index_1500hz.npz"), lfp_time_index = lfp_time_index)
            for i in range(2):
                channel_outside_hp = take_two[i]
                channel_outside_hp = "channelsrawInd_"+ str(channel_outside_hp)
                np.savez(os.path.join(session_subfolder, f"probe_{probe_id}_channel_{channel_outside_hp}_lfp_control_channel.npz"), lfp_control_channel = outof_hp_chans_lfp[i])
            # delete the dataframes we don't need to clear up some memory
            del destriped
            
            # FILTERING FOR EVENTS IN THE PROCESSED CHANNELS, WRITING TO FILE
            
            # ripples
            # make fake speed variable, we can use this for now and fix it later              
            dummy_speed = np.zeros_like(peakrippleband)
            print("Detecting Putative Ripples")
            # we add a dimension to peakrippleband because the ripple detector needs it
            Karlsson_ripple_times = ripple_detection.Karlsson_ripple_detector(
                time = lfp_time_index, 
                zscore_threshold= ripple_band_threshold,
                filtered_lfps = peakrippleband[:,None], 
                speed = dummy_speed, 
                sampling_frequency = 1500.0
            )
            # there is no need for this criteria (Karlsson_ripple_times.duration>0.015)&(Karlsson_ripple_times.duration<0.25)
            # because they are already filtered for minimum duration
            # but we need to do it for maximum duration
            Karlsson_ripple_times = Karlsson_ripple_times[Karlsson_ripple_times.duration<0.25]
            print("Done")
            # adds some stuff we want to the file
            
            # ripple band power
            peakrippleband_power = np.abs(signal.hilbert(peakrippleband))**2
            Karlsson_ripple_times['Peak_Amp_RipBandPower'], Karlsson_ripple_times['Peak_Amp_RipBandPower_zscore'],  Karlsson_ripple_times['Peak_time'] = peaks_in_events(events=Karlsson_ripple_times, 
                                                                                                                            time_values=lfp_time_index, 
                                                                                                                            signal_values=peakrippleband_power)
            
            csv_filename = f"probe_{probe_id}_channel_{this_chan_id}_karlsson_detector_events.csv"
            csv_path = os.path.join(session_subfolder, csv_filename)
            Karlsson_ripple_times.to_csv(csv_path, index=True)
            print("Writing to file.")
            print("Detecting gamma events.")
            # gamma events
            # gamma power
            # compute this later, I will have a seperate script called SWR filtering which will do this
            #gamma_band = finitimpresp_filter_for_LFP(lfp_ca1[:,lfp_ca1_chans == this_chan_id], samplingfreq =  1500.0, lowcut = 20, highcut = 80)
            gamma_power = np.abs(signal.hilbert(gamma_band_ca1))**2
            gamma_times = event_boundary_detector(time = lfp_time_index, threshold_sd = gamma_event_thresh, envelope=False, 
                                        minimum_duration = 0.015, maximum_duration = float('inf'),
                                    five_to_fourty_band_power_df = gamma_power)
            print("Done")
            csv_filename = f"probe_{probe_id}_channel_{this_chan_id}_gamma_band_events.csv"
            csv_path = os.path.join(session_subfolder, csv_filename)
            gamma_times.to_csv(csv_path, index=True)
            
            # movement artifact detection
            for i in [0,1]:
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
                
                # write to file name
                channel_outside_hp = take_two[i]
                channel_outside_hp = "channelsrawInd_"+ str(channel_outside_hp) # no cjannel id in IBL dataset, so this will do instead
                csv_filename = f"probe_{probe_id}_channel_{channel_outside_hp}_movement_artifacts.csv"
                csv_path = os.path.join(session_subfolder, csv_filename)
                movement_controls.to_csv(csv_path, index=True)
                print("Done Probe id " + str(probe_id))
    except (IndexError, NameError) as e:
        # if there is an error we want to know about it, but we dont want it to stop the loop
        # so we will print the error to a file and continue
        logging.error('Error in session: %s', 'probe id: %s', session_id, probe_id)
        logging.error(traceback.format_exc())


# create a queue to share data between processes for logging errors
queue = Queue()
listener = Process(target=listener_process, args=(queue,))
listener.start()

# run the processes with the specified number of cores
with Pool(pool_size) as p:
    p.map(process_session, session_list)

queue.put('kill')
listener.join()
