# united_swr_detector.py
import os
import subprocess
import numpy as np
import pandas as pd
from scipy import io, signal, stats
from scipy.signal import lfilter
import scipy.ndimage
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy import interpolate
import matplotlib.pyplot as plt
import ripple_detection
from ripple_detection import filter_ripple_band
import ripple_detection.simulate as ripsim
from tqdm import tqdm
import time
import traceback
import logging
import logging.handlers
import sys
from multiprocessing import Pool, Process, Queue, Manager, set_start_method
import yaml
import json
import gzip
import string
from botocore.config import Config
import boto3

# Import utility functions from core
from swr_neuropixels_collection_core import (
    BaseLoader,
    finitimpresp_filter_for_LFP,
    event_boundary_detector,
    event_boundary_times,
    peaks_time_of_events,
    incorporate_sharp_wave_component_info
)

# Custom logging level
MESSAGE = 25
logging.addLevelName(MESSAGE, "MESSAGE")

def listener_process(queue):
    """
    This function listens for messages from the logging module and writes them to a log file.
    It sets the logging level to MESSAGE so that only messages with level MESSAGE or higher are written to the log file.
    This is a level we created to be between INFO and WARNING, so to see messages from this code and errors but not other
    messages that are mostly irrelevant and make the log file too large and uninterpretable.
    """
    root = logging.getLogger()
    h = logging.FileHandler(
        f"ibl_detector_{swr_output_dir}_{run_name}_app.log", mode="w"
    )
    f = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    h.setFormatter(f)
    root.addHandler(h)
    root.setLevel(MESSAGE)

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
    root.setLevel(MESSAGE)

def process_session(session_id):
    """
    This function takes in a session_id (eid in the IBL) and loops through the probes in that session,
    for each probe it finds the CA1 channel with the highest ripple power and uses that
    channel to detect SWR events. It also detects gamma events and movement artifacts
    on two channels outside of the brain.
    """
    # to avoid time outs in some of the libraries
    my_config = Config(connect_timeout=1200, read_timeout=1200)
    s3 = boto3.client('s3', config=my_config)
    
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
        
        # Initialize and set up the loader using the BaseLoader factory
        process_stage = "Setting up loader"
        loader = BaseLoader.create(DATASET_TO_PROCESS, session_id)
        loader.set_up()
        
        # Get probe IDs and names
        process_stage = "Getting probe IDs and names"
        if DATASET_TO_PROCESS == 'abi_visual_coding':
            probenames = None
            probelist = loader.get_probes_with_ca1()
        elif DATASET_TO_PROCESS == 'abi_visual_behaviour':
            probenames = None
            probelist = loader.get_probes_with_ca1()
        elif DATASET_TO_PROCESS == 'ibl':
            probelist, probenames = loader.get_probe_ids_and_names()

        process_stage = "Running through the probes in the session"
        #icount = 0
        # Process each probe
        for this_probe in range(len(probelist)):
            #if icount > 0:
            #    break
            #icount = icount + 1
            
            if DATASET_TO_PROCESS == 'ibl':
                probe_name = probenames[this_probe]
            probe_id = probelist[this_probe]  # Always get the probe_id from probelist
            print(f"Processing probe: {str(probe_id)}")

            # Process the probe and get results
            process_stage = f"Processing probe with id {str(probe_id)}"
            if DATASET_TO_PROCESS == 'abi_visual_coding':
                results = loader.process_probe(probe_id, filter_ripple_band)  # Use probe_id, not this_probe
            elif DATASET_TO_PROCESS == 'abi_visual_behaviour':
                results = loader.process_probe(probe_id, filter_ripple_band)  # Use probe_id, not this_probe
            elif DATASET_TO_PROCESS == 'ibl':
                results = loader.process_probe(this_probe, filter_ripple_band)  # Use probe_id, not this_probe
            # Skip if no results (no CA1 channels or no bin file)
            if results is None:
                print(f"No results for probe {probe_id}, skipping...")
                continue

            # Extract results
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
            # Extract the sharp wave channel data
            sharp_wave_lfp = results['sharpwave_chan_raw_lfp']
            sw_chan_id = results['sharpwave_chan_id']

            # Save the sharp wave LFP to file
            if save_lfp == True:
                np.savez(
                    os.path.join(
                        session_lfp_subfolder,
                        f"probe_{probe_id}_channel_{sw_chan_id}_lfp_ca1_sharpwave.npz",
                    ),
                    lfp_ca1=sharp_wave_lfp,
                )

            print("Incorporating sharp wave component information...")
            # Load the sharp wave filter
            sw_filter_data = np.load(sharp_wave_component_path)
            sharpwave_filter = sw_filter_data['sharpwave_componenet_8to40band_1500hz_band']

            # Analyze sharp wave components for each ripple
            Karlsson_ripple_times = incorporate_sharp_wave_component_info(
                events_df=Karlsson_ripple_times,
                time_values=lfp_time_index,
                ripple_filtered=peakrippleband,
                sharp_wave_lfp=sharp_wave_lfp,
                sharpwave_filter=sharpwave_filter
            )
            
            # save the info about sw band relation to the chosen channel for
            # validation of the choices made 
            if save_lfp == True:
                np.savez(
                    os.path.join(
                        session_lfp_subfolder,
                        f"probe_{probe_id}_channel_{sw_chan_id}_lfp_ca1_sharpwave.npz",
                    ),
                    lfp_ca1=sharp_wave_lfp,
                )
                
                # Save loader.sw_channel_info as compressed JSON
                channel_info_path = os.path.join(
                    session_lfp_subfolder,
                    f"probe_{probe_id}_sw_component_summary.json.gz"
                )
                with gzip.open(channel_info_path, 'wt', encoding='utf-8') as f:
                    json.dump(loader.sw_component_summary_stats_dict, f)

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
                
                if DATASET_TO_PROCESS == 'abi_visual_behaviour':
                    lfp_time_index = lfp_time_index.reshape(-1)
                    dummy_speed = dummy_speed.reshape(-1)
                if DATASET_TO_PROCESS == 'ibl':
                    # Reshape to ensure consistent (n_samples, n_channels) format for detector
                    # Prevents memory error when pd.notnull() creates boolean arrays with shape (n, n)
                    rip_power_controlchan = rip_power_controlchan.reshape(-1,1)
                
                movement_controls = ripple_detection.Karlsson_ripple_detector(
                    time=lfp_time_index.reshape(-1),
                    filtered_lfps=rip_power_controlchan,
                    speed=dummy_speed.reshape(-1),
                    zscore_threshold=movement_artifact_ripple_band_threshold,
                    sampling_frequency=1500.0,
                )
                speed_cols = [
                    col for col in movement_controls.columns if "speed" in col
                ]
                movement_controls = movement_controls.drop(columns=speed_cols)
                # write to file name
                channel_outside_hp = "channelsrawInd_" + str(channel_outside_hp)
                csv_filename = f"probe_{probe_id}_channel_{channel_outside_hp}_movement_artifacts.csv"
                csv_path = os.path.join(session_subfolder, csv_filename)
                movement_controls.to_csv(csv_path, index=True, compression="gzip")
                print("Done Probe id " + str(probe_id))

        # Cleanup resources
        if 'loader' in locals() and loader is not None:
            loader.cleanup()
        process_stage = "All processing done, Deleting the session folder"

        # in the session
        logging.log(MESSAGE, f"Processing complete for id {session_id}.")
    except Exception:
        loader = None
        # removes saved files to save memory
        if 'loader' in locals() and loader is not None:
            loader.cleanup() 
        
        # Check if the session subfolder is empty
        if os.path.exists(session_subfolder) and not os.listdir(session_subfolder):
            # If it is, delete it
            os.rmdir(session_subfolder)
            logging.log(
                MESSAGE,
                "PROCESSING FAILED REMOVING EMPTY SESSION SWR DIR: session id %s ",
                session_id,
            )
        # if there is an error we want to know about it, but we dont want it to stop the loop
        # so we will print the error to a file and continue
        logging.error(
            "Error in session: %s, probe id: %s, Process Error at: ",
            session_id,
            probe_id,
            process_stage,
        )
        logging.error(traceback.format_exc())

def main():
    """Main function to handle configuration and orchestrate processing"""
    global queue, swr_output_dir_path, lfp_output_dir_path, swr_output_dir
    global gamma_filter, save_lfp, gamma_event_thresh, ripple_band_threshold
    global movement_artifact_ripple_band_threshold, run_name, sharp_wave_component_path
    global DATASET_TO_PROCESS

    # Configure AWS timeout settings
    my_config = Config(connect_timeout=1200, read_timeout=1200)
    s3 = boto3.client('s3', config=my_config)

    # Get loader type from environment variable
    DATASET_TO_PROCESS = os.environ.get('DATASET_TO_PROCESS', '').lower()
    valid_datasets = ['ibl', 'abi_visual_behaviour', 'abi_visual_coding']
    if DATASET_TO_PROCESS not in valid_datasets:
        raise ValueError(f"DATASET_TO_PROCESS must be one of {valid_datasets}, got '{DATASET_TO_PROCESS}'")

    # Create path to config file in the same directory
    config_path = os.environ.get('CONFIG_PATH', 'expanded_config.yaml')

    with open(config_path, "r") as f:
        # Parse the YAML content
        raw_content = f.read()
        # Replace environment variables
        for key, value in os.environ.items():
            raw_content = raw_content.replace(f"${key}", value)
        # Load the YAML
        full_config = yaml.safe_load(raw_content)

    # Extract the unified output directory first
    output_dir = full_config.get("output_dir", "")

    # Load common settings
    pool_size = full_config["pool_sizes"][DATASET_TO_PROCESS]
    gamma_event_thresh = full_config["gamma_event_thresh"]
    ripple_band_threshold = full_config["ripple_band_threshold"]
    movement_artifact_ripple_band_threshold = full_config["movement_artifact_ripple_band_threshold"]
    run_name = full_config["run_name"]
    save_lfp = full_config["save_lfp"]
    gamma_filters_path = full_config["filters"]["gamma_filter"]
    sharp_wave_component_path = full_config["filters"]["sw_component_filter"]

    # Load dataset-specific settings
    if DATASET_TO_PROCESS == 'ibl':
        # IBL specific settings
        dataset_config = full_config["ibl"]
        oneapi_cache_dir = dataset_config["oneapi_cache_dir"]
        swr_output_dir = dataset_config["swr_output_dir"]
        dont_wipe_these_sessions = dataset_config["dont_wipe_these_sessions"]
        session_npz_filepath = dataset_config["session_npz_filepath"]
        
    elif DATASET_TO_PROCESS == 'abi_visual_behaviour' or DATASET_TO_PROCESS == 'abi_visual_coding':
        # ABI (Allen) specific settings
        dataset_config = full_config[DATASET_TO_PROCESS]
        swr_output_dir = dataset_config["swr_output_dir"]
        dont_wipe_these_sessions = dataset_config["dont_wipe_these_sessions"]
        only_brain_observatory_sessions = dataset_config["only_brain_observatory_sessions"]

    print(f"Configured for dataset: {DATASET_TO_PROCESS}")
    print(f"Pool size: {pool_size}")
    print(f"Output directory: {output_dir}")
    print(f"SWR output directory: {swr_output_dir}")

    # Setup logging
    log_file = os.environ.get('LOG_FILE', f"{DATASET_TO_PROCESS}_detector_{swr_output_dir}_{run_name}_app.log")

    # Set up file handler for logging
    file_handler = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Set up root logger - but don't remove existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(MESSAGE)  # Only log MESSAGE level and above
    root_logger.addHandler(file_handler)

    # Prevent propagation of lower-level warnings to the root logger
    for logger_name in ['hdmf', 'pynwb', 'spikeglx', 'ripple_detection']:
        logger = logging.getLogger(logger_name)
        logger.propagate = False  # Don't send these to the root logger

    # loading filters
    gamma_filter = np.load(gamma_filters_path)
    gamma_filter = gamma_filter["arr_0"]

    # Create output directories
    swr_output_dir_path = os.path.join(output_dir, swr_output_dir)
    os.makedirs(swr_output_dir_path, exist_ok=True)
    
    if save_lfp:
        lfp_output_dir_path = os.path.join(output_dir, swr_output_dir + "_lfp_data")
        os.makedirs(lfp_output_dir_path, exist_ok=True)

    # Set up multiprocessing queue for logging
    queue = Queue()
    listener = Process(target=listener_process, args=(queue,))
    listener.start()

    # Load session IDs based on dataset type
    if DATASET_TO_PROCESS == "abi_visual_coding":
        data_file_path = os.path.join("session_id_lists", "allen_viscoding_ca1_session_ids.npz")
        data = np.load(data_file_path)
        all_sesh_with_ca1_eid = data["data"]
        del data
        print(f"Loaded {len(all_sesh_with_ca1_eid)} sessions from {data_file_path}")

    elif DATASET_TO_PROCESS == "abi_visual_behaviour":
        data_file_path = os.path.join("session_id_lists", "allen_visbehave_ca1_session_ids.npz")
        data = np.load(data_file_path)
        all_sesh_with_ca1_eid = data["data"]
        del data
        print(f"Loaded {len(all_sesh_with_ca1_eid)} sessions from {data_file_path}")

    elif DATASET_TO_PROCESS == "ibl":
        session_file_path = os.path.join("session_id_lists", session_npz_filepath)
        data = np.load(session_file_path)
        all_sesh_with_ca1_eid = data["all_sesh_with_ca1_eid_unique"]
        del data
        print(f"Loaded {len(all_sesh_with_ca1_eid)} sessions from {session_file_path}")

    # Run the processing with the specified number of cores
    with Pool(pool_size, initializer=init_pool) as p:
        p.map(process_session, all_sesh_with_ca1_eid[0:6])

    # Signal listener to terminate and wait for it to complete
    queue.put("kill")
    listener.join()

    # Find and clean up empty session folders
    print(f"Checking for empty session folders in {swr_output_dir_path}")
    empty_folder_count = 0

    for folder_name in os.listdir(swr_output_dir_path):
        folder_path = os.path.join(swr_output_dir_path, folder_name)
        
        # Check if it's a directory and starts with the session prefix
        if os.path.isdir(folder_path) and folder_name.startswith("swrs_session_"):
            # Check if the directory is empty
            if not os.listdir(folder_path):
                session_id = folder_name.replace("swrs_session_", "")
                logging.log(MESSAGE, f"Empty session folder found and removed: {session_id}")
                print(f"Removing empty session folder: {folder_path}")
                
                # Remove the empty directory
                os.rmdir(folder_path)
                empty_folder_count += 1

    print(f"Removed {empty_folder_count} empty session folders")
    logging.log(MESSAGE, f"Processing complete. Removed {empty_folder_count} empty session folders.")

if __name__ == "__main__":
    main()