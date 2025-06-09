# IBL SWR detector
import os
import re
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
from neurodsp.voltage import destripe_lfp
from ibllib.plots import Density
import time
import traceback
import logging
import logging.handlers
import sys
from multiprocessing import Pool, Process, Queue, Manager, set_start_method
import pickle
from one.api import ONE

ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
one = ONE(password="international")

from brainbox.io.one import load_wheel_reaction_times
import brainbox.behavior.wheel as wh
from ibllib.io.extractors.ephys_fpga import extract_wheel_moves
from ibllib.io.extractors.training_wheel import extract_first_movement_times

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ibl_ripples_path = "/space/scratch/IBL_swr_data/ibl_swr_murphylab2024"
ibl_lfp_path = "/space/scratch/IBL_swr_data/ibl_swr_murphylab2024_lfp_data"
theta_filter_path = "/home/acampbell/NeuropixelsLFPOnRamp/PowerBandFilters/swr_detection_script_filters_1500Hz/theta_1500hz_bandpass_filter.npz"
theta_filter = np.load(theta_filter_path)
theta_filter = theta_filter["arr_0"]

#
speeds = []
theta_powers = {
    "peakwindow_med": [],
    "peakwindow_mean": [],
    "eventwindow_med": [],
    "eventwindow_mean": [],
}
theta_compute_halfwindow = 0.125
failed_sesh = []


for sessionfolder in os.listdir(ibl_ripples_path)[0:1]:
    try:
        if ".csv" in sessionfolder:
            continue
        folder_path = os.path.join(ibl_ripples_path, sessionfolder)
        folderfiles = os.listdir(folder_path)
        try:
            global_ripples_filename = [
                file for file in folderfiles if "global_swrs" in file
            ][0]
        except:
            continue
        global_ripples_df = pd.read_csv(
            os.path.join(folder_path, global_ripples_filename), compression="gzip"
        )
        session_id = sessionfolder.split("_")[-1]

        # getting speeds
        wheel = one.load_object(session_id, "wheel", collection="alf")
        pos, t = wh.interpolate_position(wheel.timestamps, wheel.position)
        delta_t = 1 / np.array([t[i] - t[i - 1] for i in range(1, t.shape[0])]).mean()
        wh_vel, wh_accel = wh.velocity_filtered(pos, delta_t)

        average_speeds = []
        for _, row in global_ripples_df.iterrows():
            start_time = row["start_time"]
            end_time = row["end_time"]
            mask = (t >= start_time) & (t <= end_time)
            average_speed = np.abs(wh_vel[mask]).mean()
            average_speeds.append(average_speed)

        global_ripples_df["average_speed"] = average_speeds
        speeds.extend(average_speeds)

        lfp_session_path = f"{ibl_lfp_path}/lfp_session_{session_id}"
        lfp_files = os.listdir(lfp_session_path)
        for file in folderfiles:
            # we load lfp data for the karlsson files, then we compute theta power (zscored) and take that values
            # in a window around the peak of the swr
            if not "putative_swr_events" in file:
                continue

            # load the data
            events_df = pd.read_csv(os.path.join(folder_path, file), compression="gzip")
            events_df = events_df[
                (events_df.Overlaps_with_gamma == True)
                & (events_df.Overlaps_with_movement == True)
            ]
            probe_id = re.search(r"probe_(.*?)_", file).group(1)
            channel_indx = re.search(r"channelsrawInd_(.*?)_", file).group(1)
            lfp_data = [
                file
                for file in lfp_files
                if f"channelsrawInd_{channel_indx}" in file
                and probe_id in file
                and "ca1_peakripplepower.npz" in file
            ]
            lfp_data = np.load(os.path.join(lfp_session_path, lfp_data[0]))
            lfp_data = lfp_data["lfp_ca1"]
            lfp_times = [
                file
                for file in lfp_files
                if f"channelsrawInd_{channel_indx}" in file
                and probe_id in file
                and "time_index_1500hz.npz" in file
            ]
            lfp_times = np.load(os.path.join(lfp_session_path, lfp_times[0]))
            lfp_times = lfp_times["lfp_time_index"]

            # compute theta power
            theta_pow_zscore = np.convolve(lfp_data, theta_filter, mode="same")
            theta_pow_zscore = scipy.stats.zscore(
                np.abs(signal.hilbert(theta_pow_zscore)) ** 2
            )

            for _, event in events_df.iterrows():
                # Compute median and mean for peak window
                peak_start = event["Peak_time"] - theta_compute_halfwindow
                peak_end = event["Peak_time"] + theta_compute_halfwindow
                peak_window_data = theta_pow_zscore[
                    (lfp_times >= peak_start) & (lfp_times <= peak_end)
                ]
                theta_powers["peakwindow_med"].append(np.median(peak_window_data))
                theta_powers["peakwindow_mean"].append(np.mean(peak_window_data))

                # Compute median and mean for event window
                event_window_data = theta_pow_zscore[
                    (lfp_times >= event["start_time"])
                    & (lfp_times <= event["end_time"])
                ]
                theta_powers["eventwindow_med"].append(np.median(event_window_data))
                theta_powers["eventwindow_mean"].append(np.mean(event_window_data))
    except:
        failed_sesh.append(sessionfolder)
        continue
