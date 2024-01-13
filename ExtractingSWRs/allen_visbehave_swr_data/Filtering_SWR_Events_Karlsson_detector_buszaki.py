# this code filters the events provided based on certain criteria
# draft code is in /home/acampbell/AllenLFP/AllenLFP_test.ipynb


# change this as needed:
sdk_cache_dir='/space/scratch/allen_visbehave_data'# path to where the cache for the allensdk is (wehre the lfp is going)
input_dir = '/space/scratch/allen_visbehave_swr_data/allen_visbehave_swr_5sdthresh'
output_dir = '/space/scratch/allen_visbehave_swr_data/allen_visbehave_swr_5sdthresh'
swr_output_dir = 'filtered_swrs_buszaki' # directory specifying the 

# thresholding parameters

select_these_sessions = []
# test set list
#select_these_sessions = [715093703, 719161530, 721123822, 743475441, 744228101, 746083955, 750332458, 750749662, 751348571, 754312389, 754829445, 755434585, 756029989, 757216464, 757970808, 758798717, 759883607, 760345702, 761418226, 762602078, 763673393]
#select_these_sessions = [715093703]

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
import ripple_detection.simulate as ripsim # for making our time vectors
from scipy import signal
# %%
#import KernelRegDraft as kreg # custom module, not needed
#import 'Stienmetz2019Reanalyzed/KernelRegDraft.py' as kreg
import piso #can be difficult to install, https://piso.readthedocs.io/en/latest/
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from tqdm import tqdm
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# Functions

def get_session_id_numbers_from_swr_event_directories(directory_path):
    # Get a list of all directories in the specified path that contain "swrs_session_"
    all_directories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d)) and "swrs_session_" in d]

    # Remove the substring "swrs_session_" and convert to int
    directory_numbers = [int(d.replace("swrs_session_", "")) for d in all_directories]

    return directory_numbers

def get_files_with_substrings(directory_path, substring1, substring2):
    # Get a list of all items (files) in the specified path
    all_files = [f for f in os.listdir(directory_path)
                 if os.path.isfile(os.path.join(directory_path, f))
                 and substring1 in f
                 and substring2 in f]
    return all_files

def get_numbers_between_substrings(directory_path, substring1, substring2):
    all_files = os.listdir(directory_path)
    numbers = []

    for filename in all_files:
        if os.path.isfile(os.path.join(directory_path, filename)) and substring1 in filename and substring2 in filename:
            # Extract the number between the substrings
            pattern = f"{substring1}_(\d+)_{substring2}"
            match = re.search(pattern, filename)
            if match:
                numbers.append(int(match.group(1)))

    return numbers

def find_channel_num_in_filename(unfilterd_swr_path, criteria1, criteria2):
    """
    Finds the channel number in the filename of the unfiltered swr file
    Parameters
    ----------
    unfilterd_swr_path : str
        Path to the unfiltered swr file (e.g. '/space/scratch/allen_visbehave_swr_data_test/filtered_lfp/filtered_lfp_0.mat')
    criteria1 : str 
        Criteria to find in the filename (e.g. 'filtered_lfp')
    criteria2 : str
        Criteria to find in the filename (e.g. 'channel')
    Returns
    -------
    int : channel number
    """
    filenames = np.array(os.listdir(unfilterd_swr_path))
    mask = np.array([], dtype=bool)
    for filename in filenames:
        if criteria1 in filename and criteria2 in filename:
            mask = np.append(mask,True)
        else:
            mask = np.append(mask,False)
    filename = str(filenames[mask])
    start = filename.find('channel_') + len('channel_')
    end = filename.find('_', start)
    result = filename[start:end] if start != -1 and end != -1 else None
    return int(result)


def probe_ids_fromfilenames(filenames):
    probe_ids = []
    for filename in filenames:
        match = re.search(r'probe_(\d+)', filename)
        if match:
            probe_ids.append(int(match.group(1)))
    return probe_ids



def check_overlap(df1, df2):
    # returns true or false if there is overlap between the two dataframes with start_time and end_time columns
    result = []

    for i in range(len(df1)):
        start_time_1, end_time_1 = df1.loc[i, 'start_time'], df1.loc[i, 'end_time']
        overlap = any((start_time_2 < end_time_1 and end_time_2 > start_time_1) for start_time_2, end_time_2 in zip(df2['start_time'], df2['end_time']))
        result.append(overlap)

    return result

# Replace 'your_directory_path' with the actual path of the directory you want to search
if len(select_these_sessions)==0:
    select_these_sessions = get_session_id_numbers_from_swr_event_directories(input_dir)



# creating the path to our output directory for the filtered swrs
swr_output_dir_path = os.path.join(output_dir, swr_output_dir)

# we start by calling and filtering our dataframe of the sessions we will be working with

# number of events in total 
eventspersession_df = pd.DataFrame(columns=['session_id', 'probe_id', 'ripple_number'])

for seshnum in tqdm(range(0, len(select_these_sessions)), desc="Processing", unit="iteration"):
    session_id = select_these_sessions[seshnum]
    
    # making the input path for this session
    session_path = os.path.join(input_dir, 'swrs_session_' + str(session_id))
    
    # making the output path for this session and the subfolder for the session
    session_subfolder = "swrs_session_" + str(session_id)
    session_subfolder = os.path.join(swr_output_dir_path, session_subfolder)
    os.makedirs(session_subfolder, exist_ok=True)
    
    # make probe list
    probe_file_list = get_files_with_substrings(session_path, substring1='probe', substring2='karlsson_detector_events')
    probe_id_list = probe_ids_fromfilenames(probe_file_list)
    
    for probe_num in range(0, len(probe_file_list)):
        # load the dataframe containing the putative ripples, filter for ones matching cirteria
        putative_ripples_df = pd.read_csv(os.path.join(session_path, probe_file_list[probe_num]))
        print('putative ripples df')

        probe_id = probe_id_list[probe_num]
        # check if the events overlap with each other
        # check if they overlap with gamma events
    
        # now check if the overlapping non hippocampal HFEs also overlap with the hippocampal HFEs
        # if they do we mark them in the dataframe
        # check if the HFE events in the non hippocampal channels overlap
        movement_channels_files = get_files_with_substrings(session_path, substring1='probe_' + str(probe_id), substring2='movement_artifacts')
        movement_channel_1 = pd.read_csv(os.path.join(session_path, movement_channels_files[0]))
        print('Check for overlapping movement artifacts')
        putative_ripples_df['Overlaps_with_movement'] = check_overlap(putative_ripples_df, movement_channel_1)
        
        print('Filtering the putative ripples...')
        #filtered_df = putative_ripples_df[(putative_ripples_df['Overlaps_with_gamma'] == False) & (putative_ripples_df['Overlaps_with_movement'] == False)]
        filtered_df = putative_ripples_df[putative_ripples_df['Overlaps_with_movement'] == False]
        # this line filtered by max lfp ampiplitude which is pointless, should be removed
        #filtered_df = filtered_df[filtered_df.Peak_Amplitude_lfpzscore > lfp_amplidude_threshold]
        print('Writing filtered events to file')
        csv_filename = f"session_{session_id}_probe_{probe_id}_filtered_swrs.csv"
        csv_path = os.path.join(session_subfolder, csv_filename)
        filtered_df.to_csv(csv_path, index=True)
        new_row = {'session_id': session_id, 'probe_id': probe_id, 'ripple_number': filtered_df.shape[0]}
        eventspersession_df = eventspersession_df.append(new_row, ignore_index=True)
        print('Done probe ' + str(probe_id))
        
    # make list of global ripples
    # with peak ripple power detected in the hippocampal channels
    
    
    print('Done session ' + str(session_id))
"""
# compute QC thresholds
eventspersession_df['ripple_number_logep1'] = np.log10(eventspersession_df['ripple_number']+1)

def compute_kurtosis_threshold(data):
    threshold = np.min(data)
    while True:
        mask = data > threshold
        filtered_data = data[mask]
        kurt = kurtosis(filtered_data)
        if kurt<3:
            return threshold
        threshold += 0.1

thresh = compute_kurtosis_threshold(eventspersession_df.ripple_number_logep1)

threshold_log10_df = eventspersession_df[eventspersession_df.ripple_number_logep1>thresh]
lower_bound_5th = threshold_log10_df.ripple_number_logep1.quantile(0.05)
upper_bound_95th = threshold_log10_df.ripple_number_logep1.quantile(0.95)
mask = (eventspersession_df.ripple_number_logep1>lower_bound_5th)&(eventspersession_df.ripple_number_logep1<upper_bound_95th)
eventspersession_df['within_lognormal_bounds'] = mask

eventspersession_df.to_csv('/space/scratch/allen_visbehave_swr_data_test/filtered_swrs/eventspersession_df.csv')

eventspersession_df.to_csv(os.path.join(swr_output_dir_path, 'eventspersession_df.csv'), index=True)
"""

eventspersession_df.to_csv(os.path.join(swr_output_dir_path, 'eventspersession_df.csv'), index=True)
print('Done all sessions.')