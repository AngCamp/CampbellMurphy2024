#IBL SWR detector
import os
import subprocess 
import numpy as np
import pandas as pd
from scipy import io, signal, stats
#from fitter import Fitter, get_common_distributions, get_distributions
import scipy.ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
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
from neurodsp.voltage import destripe
#THIS CODE WORKS THIS CODE LOOPS THROUGH THE SESSIONS AND DOWNLOADS THE DATA, WE NEED TO ADD THE RIPPLE DETECTION CODE TO REMOVE THE DATA AFTER 
from neurodsp.voltage import destripe_lfp
from ibllib.plots import Density
import time # for debugging


# begine timing whole script
start_time = time.time()

from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

# NOTE:  Unline the ABI data we can pull all the experiments that have the specific brain region of interest, in this case CA1
# so we don't need to produce the sessions_without_ca1.npy file, debateable if we need it there either, could create it a different way as well


# see the notebook SearchingForSubsetforTesting.ipynb to see how this list was generated, this will be our testing list
# first fourteen are from Fingling et al., (2023) via..
# one.load_cache(tag = '2022_Q2_IBL_et_al_RepeatedSite',)
# sessions_rep_sites = one.search()
# then clear the cache...

testing_list = ['0c828385-6dd6-4842-a702-c5075f5f5e81','111c1762-7908-47e0-9f40-2f2ee55b6505','8a3a0197-b40a-449f-be55-c00b23253bbf','1a507308-c63a-4e02-8f32-3239a07dc578','1a507308-c63a-4e02-8f32-3239a07dc578','73918ae1-e4fd-4c18-b132-00cb555b1ad2',
 '73918ae1-e4fd-4c18-b132-00cb555b1ad2','09b2c4d1-058d-4c84-9fd4-97530f85baf6','5339812f-8b91-40ba-9d8f-a559563cc46b','034e726f-b35f-41e0-8d6c-a22cc32391fb','83e77b4b-dfa0-4af9-968b-7ea0c7a0c7e4','83e77b4b-dfa0-4af9-968b-7ea0c7a0c7e4','931a70ae-90ee-448e-bedb-9d41f3eda647',
 'd2832a38-27f6-452d-91d6-af72d794136c','dda5fc59-f09a-4256-9fb5-66c67667a466','e2b845a1-e313-4a08-bc61-a5f662ed295e','a4a74102-2af5-45dc-9e41-ef7f5aed88be','572a95d1-39ca-42e1-8424-5c9ffcb2df87','781b35fd-e1f0-4d14-b2bb-95b7263082bb',
 'b01df337-2d31-4bcc-a1fe-7112afd50c50','e535fb62-e245-4a48-b119-88ce62a6fe67','614e1937-4b24-4ad3-9055-c8253d089919','7f6b86f9-879a-4ea2-8531-294a221af5d0','824cf03d-4012-4ab1-b499-c83a92c5589e','4b00df29-3769-43be-bb40-128b1cba6d35','ff96bfe1-d925-4553-94b5-bf8297adf259']

# testing_list = np.load('testing_list.npy')

# load brain atlas stuff

# load in the brain atlas and the brain region object for working with the ccf and ABI region id's in channels objects
ba = AllenAtlas()
br = BrainRegions() # br is also an attribute of ba so could to br = ba.regions

#Searching for datasets
brain_acronym = 'CA1'

# WE WILL USE THIS EVENTUALLY FOR THE FULL RUN, FOR NEW WE CAN JUST USE THE TESTING LIST    
"""
# query sessions endpoint
sessions, sess_details = one.search(atlas_acronym=brain_acronym, query_type='remote', details=True)

# query insertions endpoint, DON'T THINK I NEED THIS
insertions = one.search_insertions(atlas_acronym=brain_acronym)
session_list = [x for x in sessions] # when we need to loop through all the sessions
"""
session_list = testing_list


one = ONE(password='international')
def process_session(session_id):
     # restarting one for this loop
    eid = session_id # just to keep the naming similar to the IBL example scripts, bit silly but helps to write the code
    pid, probename = one.eid2pid(eid)
    print(f'Probe: {pid}, Probe name: {probename}')
    
    band = 'lf' # either 'ap','lf'
    #for this_probe in pid:
    for this_probe in probename:
        print(this_probe)
        # Find the relevant datasets and download them
        dsets = one.list_datasets(eid, collection=f'raw_ephys_data/{this_probe}', filename='*.lf.*')
        print(type(dsets))
        print(len(dsets))
        data_files, _ = one.load_datasets(eid, dsets, download_only=False)
        bin_file = next(df for df in data_files if df.suffix == '.cbin')

        # Use spikeglx reader to read in the whole raw data
        print("sr = spikeglx.Reader(bin_file)")
        
        sr = spikeglx.Reader(bin_file)
        
        # Important: remove sync channel from raw data, and transpose
        print("raw = sr[:, :-sr.nsync].T")
        
        raw = sr[:, :-sr.nsync].T
        
        # Reminder : If not done before, remove first the sync channel from raw data
        # Apply destriping algorithm to data
        
        # code to be used in final version but in the meantime we will load in the destriped data from the saved folder just to save time
        print("destriped = destripe(raw, fs=sr.fs)")
        
        destriped = destripe(raw, fs=sr.fs)
        del raw
        print(f"destripped shape : {destriped.shape}")
        
        
        # just for debugging
        # np.load function will load the .npz file
        #data = np.load("/space/scratch/test_destripe_save/debugging_destriped_save.npz")

        # You can access your saved array with the keyword you used while saving
        #destriped = data['destriped']
        #del raw
        #del data
        print("destriped loaded...")



# print time of script
print(f"done, time elapsed: {time.time() - start_time}")  
