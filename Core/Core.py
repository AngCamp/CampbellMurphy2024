# Core.py 
# The core functions used across this repository
# Functions
# libraries
import os
import re
import subprocess 
import numpy as np
import pandas as pd
from scipy import io, signal
#from fitter import Fitter, get_common_distributions, get_distributions
import scipy.ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
# for ripple detection
import ripple_detection
from ripple_detection.core import filter_ripple_band
from ripple_detection.simulate import simulate_time
from scipy import signal
import seaborn as sns
#import KernelRegDraft as kreg # custom module, not needed
#import 'Stienmetz2019Reanalyzed/KernelRegDraft.py' as kreg
import piso #can be difficult to install, https://piso.readthedocs.io/en/latest/
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from tqdm import tqdm
import piso
from ripple_detection.simulate import simulate_time
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import xarray as xr
from scipy import interpolate
from scipy.signal import firwin, lfilter



def check_overlap(df1, df2):
    # returns true or false if there is overlap between the two dataframes with start_time and end_time columns
    result = []

    for i in range(len(df1)):
        start_time_1, end_time_1 = df1.loc[i, 'start_time'], df1.loc[i, 'end_time']
        overlap = any((start_time_2 < end_time_1 and end_time_2 > start_time_1) for start_time_2, end_time_2 in zip(df2['start_time'], df2['end_time']))
        result.append(overlap)

    return result


def merge_intervals(intervals, x_offset=0):
    """
    Merges overlapping intervals within a list of intervals
    Convert intervals into sets of tuples then merge with this function
    
    parameters
    intervals : list of tuples
    x_offset : int, optional, time offset to merge intervals
    
    Returns a list of merged intervals
    """
    intervals.sort(key=lambda interval: interval[0])

    stack = [intervals[0]]
    for current in intervals[1:]:
        last = stack[-1]
        if current[0] <= last[1] + x_offset:
            stack[-1] = (last[0], max(last[1], current[1]))
        else:
            stack.append(current)
    
    return stack

# Function to convert DataFrame to set of tuples (start_time, end_time)
def dataframe_to_interval_set(df):
    interval_set = set()
    for index, row in df.iterrows():
        interval_set.add((row['start_time'], row['end_time']))
    return interval_set

def union_of_lists_of_interval_sets(list_of_interval_sets):
    """
    Takes a list of lists of interval sets and returns the union of all
    interval sets in the list
    
    parameters
    list_of_interval_sets : list of lists of interval sets
    
    Returns a list of merged intervals
    """
    union = set()
    for interval_set in list_of_interval_sets:
        union = union.union(interval_set)
    return union


def union_of_event_times(list_of_event_intervals_df, offset=0):
    """
    Returns the union of all event times in a list of DataFrames
    Parameters
    ----------
    list_of_event_intervals_df : list of DataFrames
        Each DataFrame should have columns 'start_time' and 'end_time'
    offset : int, optional, time offset to merge intervals
    Returns
    -------     
    list : list of tuples
        Each tuple is an event time interval
    """
    set_list = []
    for df in list_of_event_intervals_df:
        set_list.append(dataframe_to_interval_set(df))
    
    union = union_of_lists_of_interval_sets(set_list)
    
    return merge_intervals(list(union), x_offset=offset)

def dataframe_union_of_event_times(list_of_event_intervals_df, offset=0, event_name='event'):
    """
    Returns the union of all event times in a list of DataFrames
    Parameters
    ----------
    list_of_event_intervals_df : list of DataFrames
        Each DataFrame should have columns 'start_time' and 'end_time'
    offset : int, optional, time offset to merge intervals
    Returns
    -------     
    DataFrame : DataFrame of event times
        Columns are 'start_time' and 'end_time'
    """
    union = union_of_event_times(list_of_event_intervals_df, offset=offset)
    union_df = pd.DataFrame(union, columns=['start_time', 'end_time'])
    union_df[event_name] = union_df.index
    return union_df

def add_overlap_probes(df, df_dict):
    """
    Adds a column to a DataFrame that lists the probes that overlap with putative global event created from
    the union of the DataFrames in df_dict
    Parameters
    ----------
    df : DataFrame
        DataFrame with columns 'start_time' and 'end_time'
    df_dict : dict
        Dictionary of DataFrames with columns 'start_time' and 'end_time'
    Returns
    -------
    DataFrame : DataFrame with new column 'overlap_probes'
    """
    overlap_probes = []
    for i, row in df.iterrows():
        overlap_probes_row = []
        for key, df_probe in df_dict.items():
            overlap = any((df_probe['start_time'] < row['end_time']) & (df_probe['end_time'] > row['start_time']))
            if overlap:
                overlap_probes_row.append(key)
        overlap_probes.append(overlap_probes_row)
    df['overlap_probes'] = overlap_probes
    return df

def unit_spike_times_alyx_format(session_obj, unit_ids, start_time=0, stop_time=np.inf, as_array=False):
    """
    Returns a DataFrame of spike times for multiple units in a session in the format
    according to the Alyx format
    
    Parameters
    ----------
    session : allensdk session object
    unit_ids : list of unit ids (obtained by session.units.index)
    start_time : float, optional, start time of spike times
    stop_time : float, optional, stop time of spike times
    as_array : bool, optional, if True returns a numpy array instead of a DataFrame
    
    Returns
    -------
    DataFrame or numpy array of spike times
    """
    df = pd.DataFrame(columns=['units', 'times'])

    for unit_id in unit_ids:
        ca1_unit_time = np.array(session_obj.spike_times[unit_id])
        filtered_time = ca1_unit_time[(ca1_unit_time > start_time) & (ca1_unit_time < stop_time)]
        unit_ids_array = np.full(filtered_time.shape, unit_id)
        
        df_temp = pd.DataFrame({
            'units': unit_ids_array,
            'times': filtered_time
        })
        
        df = pd.concat([df, df_temp], ignore_index=True)

    df = df.sort_values(by='times')

    if as_array:
        return df.values
    else:
        return df
    
    #For smoothing we make halfguassian_kernel1d and halfgaussian_filter1d
def halfgaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Half-Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(0, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    return phi_x

def halfgaussian_filter1d(input, sigma, axis=-1, output=None,
                      mode="constant", cval=0.0, truncate=4.0):
    """
    Convolves a 1-D Half-Gaussian convolution kernel.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = halfgaussian_kernel1d(sigma, lw)
    origin = -lw // 2
    return scipy.ndimage.convolve1d(input, weights, axis, output, mode, cval, origin)


def resample_signal_v1(signal, times, new_rate):
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
    # Calculate the number of samples for the new rate
    nsamples_new = int(len(times) * new_rate / (len(times) / times[-1]))

    # Create a new time array for the new rate
    new_times = np.linspace(times[0], times[-1], nsamples_new)

    # Initialize an empty array for the new signal
    new_signal = np.zeros((nsamples_new, signal.shape[1]))

    # Interpolate each source separately
    for i in range(signal.shape[1]):
        interp_func = interpolate.interp1d(times, signal[:, i])
        new_signal[:, i] = interp_func(new_times)

    return new_signal, new_times


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
        interp_func = interpolate.interp1d(times, signal[:, i], bounds_error=False, fill_value="extrapolate")
        new_signal[:, i] = interp_func(new_times)

    return new_signal, new_times

# some functions for signal processing and time series manipulation
# from Visualizing_Ripple_Modulation.ipynb, for making plots

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

"""
# how to reannotate ccf values...
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
atlas = BrainGlobeAtlas("allen_mouse_25um") # or "allen_mouse_10um" or other atalases

COORD = (500, 300, 400) # can be a list of them in tupples as well
from bg_atlasapi import BrainGlobeAtlas
atlas = BrainGlobeAtlas("allen_mouse_10um")
atlas.structure_from_coords(COORD, as_acronym=True)

# of with a list of tuples form a dataframe...
import pandas as pd
import numpy as np

# Create a DataFrame
df = pd.DataFrame({
    'col1': np.random.randint(495, 506, 3),
    'col2': np.random.randint(295, 306, 3),
    'col3': np.random.randint(395, 406, 3),
})

coordslist = [tuple(x) for x in df.to_numpy()]

[atlas.structure_from_coords(coord, as_acronym=True) for coord in coordslist]
"""