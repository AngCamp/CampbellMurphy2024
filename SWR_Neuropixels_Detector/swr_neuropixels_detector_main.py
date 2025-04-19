#!/usr/bin/env python

# debugging
import multiprocessing as mp
import os

if os.getenv("DEBUG_MODE") == "true" and mp.current_process().name == "MainProcess":
    import debugpy
    print("Debug mode: starting debugpy listener on port 5678")
    debugpy.listen(("0.0.0.0", 5678))      # or ("localhost", 5678)
    print("Waiting for VS Code to attach…")
    debugpy.wait_for_client()
    print("Debugger attached!")

# united_swr_detector.py
import os
import subprocess
import sys
import time
import traceback
import logging
import logging.handlers
import json
import gzip
import string
import argparse
from multiprocessing import Pool, Process, Queue, Manager, set_start_method
from functools import partial

# Third-party imports
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
    process_session,
    event_boundary_detector,
    event_boundary_times,
    peaks_time_of_events,
    incorporate_sharp_wave_component_info,
    check_gamma_overlap,
    check_movement_overlap,
    create_global_swr_events,
    filter_ripple_band,
    read_json_file,
    get_filter
)
from functools import partial
import argparse

# Custom logging level
MESSAGE = 25
logging.addLevelName(MESSAGE, "MESSAGE")

def listener_process(queue, log_dir, dataset_name, run_name):
    """
    This function listens for messages from the logging module and writes them to a log file.
    It sets the logging level to MESSAGE so that only messages with level MESSAGE or higher are written to the log file.
    This is a level we created to be between INFO and WARNING, so to see messages from this code and errors but not other
    messages that are mostly irrelevant and make the log file too large and uninterpretable.
    """
    root = logging.getLogger()
    # Use passed arguments for dynamic log file path
    log_file_path = os.path.join(log_dir, f"{dataset_name}_detector_{run_name}_app.log")
    h = logging.FileHandler(log_file_path, mode="w")
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

def init_pool(queue):
    """Initialize logging in the worker process."""
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(MESSAGE)

def main():
    """Main function to handle configuration and orchestrate processing"""
    #global queue, swr_output_dir_path, lfp_output_dir_path, swr_output_dir
    #global gamma_filter, save_lfp, gamma_event_thresh, ripple_band_threshold
    #global movement_artifact_ripple_band_threshold, run_name, sharp_wave_component_path
    #global DATASET_TO_PROCESS, full_config

    # Configure AWS timeout settings
    my_aws_config = Config(connect_timeout=1200, read_timeout=1200)
    s3 = boto3.client('s3', config=my_aws_config)

    # Get loader type from environment variable
    DATASET_TO_PROCESS = os.environ.get('DATASET_TO_PROCESS', '').lower()
    valid_datasets = ['ibl', 'abi_visual_behaviour', 'abi_visual_coding']
    if DATASET_TO_PROCESS not in valid_datasets:
        raise ValueError(f"DATASET_TO_PROCESS must be one of {valid_datasets}, got '{DATASET_TO_PROCESS}'")

    # Parse command-line arguments for flags
    parser = argparse.ArgumentParser(description="Run SWR Pipeline Stages.")
    parser.add_argument("-p", "--run-putative", action="store_true", help="Run Putative event detection stage.")
    parser.add_argument("-f", "--run-filter", action="store_true", help="Run event Filtering stage.")
    parser.add_argument("-g", "--run-global", action="store_true", help="Run Global event consolidation stage.")
    parser.add_argument("-C", "--cleanup-cache", action="store_true", help="Run cache cleanup after processing each session.")
    parser.add_argument("-s", "--save-lfp", action="store_true", help="Enable saving of LFP data.")
    parser.add_argument("-m", "--save-channel-metadata", action="store_true", help="Enable saving of channel selection metadata.")
    parser.add_argument("-o", "--overwrite-existing", action="store_true", help="Overwrite existing session output.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode (for internal script use).")
    # Add argument for config path if needed, overriding env var
    parser.add_argument("--config", type=str, default=os.environ.get('CONFIG_PATH', 'united_detector_config.yaml'), help="Path to configuration YAML file.")
    args = parser.parse_args()

    # Assign flags from args to variables used later
    save_lfp = args.save_lfp

    # Determine dataset from environment (still needed for config loading)
    DATASET_TO_PROCESS = os.environ.get('DATASET_TO_PROCESS', '').lower()
    if not DATASET_TO_PROCESS or DATASET_TO_PROCESS not in valid_datasets:
        raise ValueError(f"DATASET_TO_PROCESS environment variable must be set to one of {valid_datasets}, got '{DATASET_TO_PROCESS}'")

    # Create path to config file in the same directory
    config_path = args.config # Use path from args or default env var

    with open(config_path, "r") as f:
        # Read the raw YAML content as a string
        raw_yaml_content = f.read()
        
        # Expand environment variables (handles $VAR or ${VAR})
        expanded_yaml_content = os.path.expandvars(raw_yaml_content)
        
        # Load the YAML from the expanded string
        full_config = yaml.safe_load(expanded_yaml_content)

    # Extract the unified output directory first
    output_dir = full_config.get("output_dir", "")
    # --- Path Setup based on Config/Env Vars --- 
    # Use environment variables if set, otherwise use config values as defaults
    output_dir = os.environ.get('OUTPUT_DIR', full_config.get("output_dir", "./swr_output"))
    log_dir = os.environ.get('LOG_DIR', os.path.join(output_dir, 'logs')) # Default logs subdir
    swr_output_dir_path = output_dir # Main output is SWR output base
    lfp_output_dir_path = os.path.join(output_dir, f"{DATASET_TO_PROCESS}_lfp_data") # LFP subdir

    # Load common settings
    pool_size = full_config["pool_sizes"][DATASET_TO_PROCESS]
    gamma_event_thresh = full_config["gamma_event_thresh"]
    ripple_band_threshold = full_config["ripple_band_threshold"]
    movement_artifact_ripple_band_threshold = full_config["movement_artifact_ripple_band_threshold"]
    run_name = full_config["run_name"]
    # save_lfp = full_config["save_lfp"] # Replaced by flag
    gamma_filters_path = full_config["filters"]["gamma_filter"]
    sharp_wave_component_path = full_config["filters"]["sw_component_filter"]

    # Load dataset-specific settings
    if DATASET_TO_PROCESS == 'ibl':
        # IBL specific settings
        dataset_config = full_config["ibl"]
        oneapi_cache_dir = os.environ.get('IBL_ONEAPI_CACHE') # Rely only on Env Var
        swr_output_dir = dataset_config["swr_output_dir"]
        dont_wipe_these_sessions = dataset_config["dont_wipe_these_sessions"]
        session_npz_filepath = dataset_config["session_npz_filepath"]
    elif DATASET_TO_PROCESS == 'abi_visual_behaviour' or DATASET_TO_PROCESS == 'abi_visual_coding':
        # ABI (Allen) specific settings
        dataset_config = full_config[DATASET_TO_PROCESS]
        # Use Env Vars for cache paths
        if DATASET_TO_PROCESS == 'abi_visual_behaviour':
            abi_cache_dir = os.environ.get('ABI_VISUAL_BEHAVIOUR_SDK_CACHE') # Rely only on Env Var
        else: # abi_visual_coding
            abi_cache_dir = os.environ.get('ABI_VISUAL_CODING_SDK_CACHE') # Rely only on Env Var
        swr_output_dir = dataset_config["swr_output_dir"]
        dont_wipe_these_sessions = dataset_config["dont_wipe_these_sessions"]
        only_brain_observatory_sessions = dataset_config["only_brain_observatory_sessions"]

    print(f"Configured for dataset: {DATASET_TO_PROCESS}")
    print(f"Pool size: {pool_size}")
    print(f"Output directory: {output_dir}")
    print(f"SWR output directory: {swr_output_dir}")

    # Setup logging
    queue = Queue()
    # Ensure log directory exists (must be done before listener starts)
    os.makedirs(log_dir, exist_ok=True) 
    # Pass necessary arguments to the listener process
    listener = Process(target=listener_process, args=(queue, log_dir, DATASET_TO_PROCESS, run_name))
    listener.start()

    # loading filters
    gamma_filter = np.load(gamma_filters_path)
    gamma_filter = gamma_filter["arr_0"]

    # Create output directories
    swr_output_dir_path = os.path.join(output_dir, swr_output_dir)
    os.makedirs(swr_output_dir_path, exist_ok=True)
    
    # LFP output dir creation moved here, depends on save_lfp flag
    if save_lfp:
        lfp_output_dir_path = os.path.join(output_dir, swr_output_dir + "_lfp_data")
        os.makedirs(lfp_output_dir_path, exist_ok=True)
    else:
        lfp_output_dir_path = None # Ensure it's None if not saving LFP

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

    # for debugging..
    all_sesh_with_ca1_eid = all_sesh_with_ca1_eid[1:3]
    # --- Consolidate Config Dictionary --- 
    config = {
        "paths": {
            "output_dir": output_dir,
            "log_dir": log_dir,
            "swr_output_dir": swr_output_dir_path, # Base path for SWR output
            "lfp_output_dir": lfp_output_dir_path, # Base path for LFP output
            "ibl_cache_dir": oneapi_cache_dir if DATASET_TO_PROCESS == 'ibl' else None,
            "abi_vb_cache_dir": abi_cache_dir if DATASET_TO_PROCESS == 'abi_visual_behaviour' else None,
            "abi_vc_cache_dir": abi_cache_dir if DATASET_TO_PROCESS == 'abi_visual_coding' else None,
        },
        "run_details": {
            "dataset_to_process": DATASET_TO_PROCESS,
            "run_name": run_name,
            # Add other relevant details from dataset_config?
            "dont_wipe_these_sessions": dont_wipe_these_sessions,
            "only_brain_observatory_sessions": only_brain_observatory_sessions if DATASET_TO_PROCESS != 'ibl' else None,
        },
        "flags": {
            "save_lfp": args.save_lfp, # From command line args
            "save_channel_metadata": args.save_channel_metadata,
            "overwrite_existing": args.overwrite_existing,
            "cleanup_cache": args.cleanup_cache,
            "run_putative": args.run_putative,
            "run_filter": args.run_filter,
            "run_global": args.run_global,
        },
        "pool_size": pool_size,
        "ripple_detection": {
            "ripple_band_threshold": ripple_band_threshold,
            "movement_artifact_ripple_band_threshold": movement_artifact_ripple_band_threshold,
            # Add other ripple detection params from full_config if needed
        },
        "artifact_detection": {
            "gamma_event_thresh": gamma_event_thresh,
            # Add other artifact params if needed
        },
        "filters": {
            "gamma_filter": gamma_filter, # Loaded filter array
            "sharp_wave_component_path": sharp_wave_component_path, # Path to SW filter
            # Add other filter details if needed
        },
        "global_swr": full_config.get("global_swr_detection", {}), # Pass the whole sub-dict
        # Add other top-level or dataset-specific configs from full_config if needed by process_session
        # e.g., sampling rates:
        "sampling_rates": full_config.get("sampling_rates", {"target_fs": 1500.0}) # Example default
    }

    # Start multiprocessing
    # ==============================================================================
    # Create a partially applied function with the consolidated configuration
    process_func_partial = partial(process_session, config=config)
    
    # Run the processing with the specified number of cores
    print(f"Starting processing pool with {config['pool_size']} workers...")
    with Pool(processes=config['pool_size'], initializer=init_pool, initargs=(queue,)) as p:
        # Use imap for progress bar with tqdm
        list(tqdm(p.imap(process_func_partial, all_sesh_with_ca1_eid), total=len(all_sesh_with_ca1_eid)))

    # Clean up
    # ==============================================================================
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