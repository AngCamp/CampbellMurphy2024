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