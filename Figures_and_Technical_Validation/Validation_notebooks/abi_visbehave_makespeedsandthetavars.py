# libraries
import os
import subprocess
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import ripple_detection
from scipy.signal import hilbert
from ripple_detection import filter_ripple_band
from scipy.interpolate import interp1d
from scipy.signal import windows, firwin, lfilter
from scipy.ndimage import convolve, filters
from scipy import stats
from tqdm import tqdm
from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)
from scipy import interpolate
from fitter import Fitter, get_common_distributions, get_distributions
import time
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import zscore
from multiprocessing import Manager, Pool
import glob
import pickle

output_dir = "/home/acampbell"
# os.chdir(output_dir)

abi_ripples_path = (
    "/space/scratch/allen_visbehave_swr_data/allen_visbehave_swr_murphylab2024"
)
abi_lfp_path = (
    "/space/scratch/allen_visbehave_swr_data/allen_visbehave_swr_murphylab2024_lfp_data"
)
theta_filter_path = "/home/acampbell/NeuropixelsLFPOnRamp/PowerBandFilters/swr_detection_script_filters_1500Hz/theta_1500hz_bandpass_filter.npz"
theta_filter = np.load(theta_filter_path)
theta_filter = theta_filter["arr_0"]

speeds = []
theta_powers = {
    "peakwindow_med": [],
    "peakwindow_mean": [],
    "eventwindow_med": [],
    "eventwindow_mean": [],
}
theta_compute_halfwindow = 0.125
failed_sesh = []


# Assuming process_speed_and_theta is defined elsewhere and returns average_speeds, event_window_data
def process_speed_and_theta(
    sessionfolder,
    abi_ripples_path,
    abi_lfp_path,
    sdk_cache_dir,
    theta_filter,
    theta_compute_halfwindow,
):
    try:
        if ".csv" in sessionfolder:
            return [], {}

        cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
            cache_dir=output_dir
        )
        sdk_cache_dir = "/space/scratch/allen_visbehave_data"
        # Setting up the ABI Cache (where data is held, what is present or absent)
        manifest_path = os.path.join(sdk_cache_dir, "manifest.json")

        cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
            cache_dir=sdk_cache_dir
        )

        folder_path = os.path.join(abi_ripples_path, sessionfolder)
        folderfiles = os.listdir(folder_path)

        global_ripples_filename = [
            file for file in folderfiles if "global_swrs" in file
        ][0]
        global_ripples_df = pd.read_csv(
            os.path.join(folder_path, global_ripples_filename), compression="gzip"
        )
        session_id = int(sessionfolder.split("_")[-1])

        session = cache.get_ecephys_session(session_id)
        wheel_velocity = session.running_speed["speed"].values
        wheel_time = session.running_speed["timestamps"].values

        interp_func = interp1d(wheel_time, wheel_velocity)
        wheel_time = np.linspace(
            wheel_time[0], wheel_time[-1], int(len(wheel_time) * 1500 / len(wheel_time))
        )
        wheel_velocity = interp_func(wheel_time)
        []
        average_speeds = []
        for _, row in global_ripples_df.iterrows():
            start_time = row["start_time"]
            end_time = row["end_time"]
            mask = (wheel_time >= start_time) & (wheel_time <= end_time)
            # Before calculating the average speed, check if the masked array is empty or contains only NaNs
            if mask.size > 0 and not np.isnan(wheel_velocity[mask]).all():
                average_speed = np.nanmean(np.abs(wheel_velocity[mask]))
            elif mask.size == 0:
                average_speed = 0
                # Handle the case where the calculation cannot be performed meaningfully
                average_speed = (
                    0  # or np.nan, depending on how you want to handle this case
                )
            average_speed = np.abs(wheel_velocity[mask]).mean()
            average_speeds.append(average_speed)

        theta_powers = {
            "peakwindow_med": [],
            "peakwindow_mean": [],
            "eventwindow_med": [],
            "eventwindow_mean": [],
        }

        lfp_session_path = f"{abi_lfp_path}/lfp_session_{str(session_id)}"
        lfp_files = os.listdir(lfp_session_path)
        for file in folderfiles:
            if not "putative_swr_events" in file:
                continue

            events_df = pd.read_csv(os.path.join(folder_path, file), compression="gzip")
            events_df = events_df[
                (events_df.Overlaps_with_gamma == True)
                & (events_df.Overlaps_with_movement == True)
            ]
            probe_id = re.search(r"probe_(.*?)_", file).group(1)
            channel_indx = re.search(r"channel_(.*?)_", file).group(1)

            lfp_data_file = [
                f
                for f in lfp_files
                if f"channel_{channel_indx}" in f
                and probe_id in f
                and "ca1_peakripplepower.npz" in f
            ][0]
            lfp_data = np.load(os.path.join(lfp_session_path, lfp_data_file))["lfp_ca1"]

            lfp_times_file = [
                f
                for f in lfp_files
                if f"channel_{channel_indx}" in f
                and probe_id in f
                and "time_index_1500hz.npz" in f
            ][0]
            lfp_times = np.load(os.path.join(lfp_session_path, lfp_times_file))[
                "lfp_time_index"
            ]

            theta_pow_zscore = np.convolve(
                lfp_data.flatten(), theta_filter.flatten(), mode="same"
            )
            theta_pow_zscore = scipy.stats.zscore(
                np.abs(hilbert(theta_pow_zscore)) ** 2
            )

            for _, event in events_df.iterrows():
                peak_start = event["Peak_time"] - theta_compute_halfwindow
                peak_end = event["Peak_time"] + theta_compute_halfwindow
                peak_window_data = theta_pow_zscore[
                    (lfp_times >= peak_start) & (lfp_times <= peak_end)
                ]
                theta_powers["peakwindow_med"].append(np.median(peak_window_data))
                theta_powers["peakwindow_mean"].append(np.mean(peak_window_data))

                event_window_data = theta_pow_zscore[
                    (lfp_times >= event["start_time"])
                    & (lfp_times <= event["end_time"])
                ]
                theta_powers["eventwindow_med"].append(np.median(event_window_data))
                theta_powers["eventwindow_mean"].append(np.mean(event_window_data))

        return average_speeds, theta_powers
    except Exception as e:
        print(f"Error processing {sessionfolder}: {e}")
        return [], {}


def worker_function(
    shared_speeds,
    shared_theta_powers,
    sessionfolder,
    abi_ripples_path,
    abi_lfp_path,
    cache,
    theta_filter,
    theta_compute_halfwindow,
):
    # Call the process_speed_and_theta function to get data
    average_speeds, theta_powers = process_speed_and_theta(
        sessionfolder,
        abi_ripples_path,
        abi_lfp_path,
        cache,
        theta_filter,
        theta_compute_halfwindow,
    )

    # Use the shared data structures to append/extend data
    shared_speeds.extend(average_speeds)
    for key in theta_powers:
        shared_theta_powers[key].extend(theta_powers[key])


if __name__ == "__main__":
    abi_ripples_path = (
        "/space/scratch/allen_visbehave_swr_data/allen_visbehave_swr_murphylab2024"
    )
    abi_lfp_path = "/space/scratch/allen_visbehave_swr_data/allen_visbehave_swr_murphylab2024_lfp_data"
    theta_filter_path = "/home/acampbell/NeuropixelsLFPOnRamp/PowerBandFilters/swr_detection_script_filters_1500Hz/theta_1500hz_bandpass_filter.npz"
    theta_filter = np.load(theta_filter_path)
    theta_filter = theta_filter["arr_0"]

    sdk_cache_dir = "/space/scratch/allen_visbehave_data"
    # Setting up the ABI Cache (where data is held, what is present or absent)

    sessionfolders = [
        folder for folder in os.listdir(abi_ripples_path) if not ".csv" in folder
    ]  # Adjust as needed

    with Manager() as manager:
        shared_speeds = manager.list()
        shared_theta_powers = manager.dict(
            {
                "peakwindow_med": manager.list(),
                "peakwindow_mean": manager.list(),
                "eventwindow_med": manager.list(),
                "eventwindow_mean": manager.list(),
            }
        )

        args_list = [
            (
                shared_speeds,
                shared_theta_powers,
                folder,
                abi_ripples_path,
                abi_lfp_path,
                sdk_cache_dir,
                theta_filter,
                theta_compute_halfwindow,
            )
            for folder in sessionfolders
        ]
        pool_size = 10
        with Pool(pool_size) as pool:
            pool.starmap(worker_function, args_list)

        # Convert shared data structures back to regular lists/dicts if necessary
        speeds = list(shared_speeds)
        theta_powers = {key: list(value) for key, value in shared_theta_powers.items()}


theta_save_path = os.path.join(output_dir, "abi_visbehave_theta.pkl")
with open(theta_save_path, "wb") as f:
    pickle.dump(theta_powers, f)

speeds_save_path = os.path.join(output_dir, "abi_visbehave_speeds.npy")
# Save the numpy array
np.save(speeds_save_path, speeds)
