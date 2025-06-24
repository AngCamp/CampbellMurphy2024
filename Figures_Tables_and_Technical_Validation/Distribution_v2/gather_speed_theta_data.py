import os
import argparse
import numpy as np
import pandas as pd
import scipy
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from scipy.signal import windows, convolve
from scipy.stats import zscore
import re
import multiprocessing as mp
import time
import json
import fcntl
from pathlib import Path

# Lazy loading for API dependencies
def load_abi_visbehave_api():
    """Lazy load ABI Visual Behaviour API"""
    from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
    return VisualBehaviorNeuropixelsProjectCache

def load_abi_viscoding_api():
    """Lazy load ABI Visual Coding API"""
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    return EcephysProjectCache

def load_ibl_api():
    """Lazy load IBL ONE API"""
    from one.api import ONE
    import brainbox.behavior.wheel as wh
    return ONE, wh

# Paths
DATA_ROOT = "/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_v2_final"
OUTPUT_DIR = "/home/acampbell/NeuropixelsLFPOnRamp/Figures_Tables_and_Technical_Validation/Distribution_v2/distributions_for_plotting"
THETA_FILTER_PATH = "../../Figures_Tables_and_Technical_Validation/Distribution_v2/theta_1500hz_bandpass_filter.npz"

# Dataset configurations
DATASET_CONFIGS = {
    "abi_visbehave": {
        "swr_dir": "allen_visbehave_swr_murphylab2024",
        "lfp_dir": "allen_visbehave_swr_murphylab2024_lfp_data",
        "cache_dir": "/space/scratch/allen_visbehave_data",
        "api_loader": load_abi_visbehave_api,
        "session_loader": "get_ecephys_session",
        "speed_data": "running_speed"
    },
    "abi_viscoding": {
        "swr_dir": "allen_viscoding_swr_murphylab2024", 
        "lfp_dir": "allen_viscoding_swr_murphylab2024_lfp_data",
        "cache_dir": "/space/scratch/allen_viscoding_data",
        "api_loader": load_abi_viscoding_api,
        "session_loader": "get_session_data",
        "speed_data": "running_speed"
    },
    "ibl": {
        "swr_dir": "ibl_swr_murphylab2024",
        "lfp_dir": "ibl_swr_murphylab2024_lfp_data", 
        "cache_dir": None,  # Uses ONE API
        "api_loader": load_ibl_api,
        "session_loader": "load_object",
        "speed_data": "wheel"
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Gather mean theta power and speed for SWR events.")
    parser.add_argument('--dataset', choices=['abi_visbehave', 'abi_viscoding', 'ibl'], 
                       required=True, help='Dataset to process')
    parser.add_argument('--max_sessions', type=int, default=None, help='Limit number of sessions (for debugging)')
    parser.add_argument('--n_processes', type=int, default=4, help='Number of parallel processes')
    return parser.parse_args()

def get_speed_data_abi(session_obj, dataset_type):
    """Get speed data for ABI datasets"""
    if dataset_type == "abi_visbehave":
        # Visual Behaviour uses 'speed' and 'timestamps' fields
        wheel_velocity = session_obj.running_speed['speed'].values
        wheel_time = session_obj.running_speed['timestamps'].values
    else:  # abi_viscoding
        # Visual Coding uses 'start_time' and 'velocity' fields
        running_speed_times = session_obj.running_speed['start_time'].values
        running_speed_velocity = session_obj.running_speed['velocity'].values
        
        # Interpolate to 1500 Hz to match LFP sampling rate
        # Create a regular time grid at 1500 Hz
        session_start = running_speed_times[0]
        session_end = running_speed_times[-1]
        wheel_time = np.arange(session_start, session_end, 1/1500)  # 1500 Hz sampling
        
        # Interpolate velocity to the regular time grid
        interp_func = interp1d(running_speed_times, running_speed_velocity, 
                              bounds_error=False, fill_value=0)
        wheel_velocity = interp_func(wheel_time)
        return wheel_velocity, wheel_time
    
    # For Visual Behaviour: Interpolate to 1500 Hz
    interp_func = interp1d(wheel_time, wheel_velocity)
    wheel_time = np.linspace(wheel_time[0], wheel_time[-1], int(len(wheel_time) * 1500 / len(wheel_time)))
    wheel_velocity = interp_func(wheel_time)
    return wheel_velocity, wheel_time

def get_speed_data_ibl(one, session_id):
    """Get speed data for IBL dataset"""
    # Import wheel module here to ensure it's available
    import brainbox.behavior.wheel as wh
    
    wheel = one.load_object(session_id, 'wheel', collection='alf')
    pos, t = wh.interpolate_position(wheel.timestamps, wheel.position)
    
    # Calculate velocity using simple differentiation instead of filtered velocity
    # This avoids the frequency band filtering issue
    dt = np.diff(t)
    vel = np.diff(pos) / dt
    
    # Interpolate to 1500 Hz
    interp_func = interp1d(t[1:], vel, bounds_error=False, fill_value=0)
    wheel_time = np.linspace(t[1], t[-1], int(len(t) * 1500 / len(t)))
    wheel_velocity = interp_func(wheel_time)
    
    return wheel_velocity, wheel_time

def process_session(args):
    """Process a single session - this function will be called by multiprocessing"""
    session, dataset_type, config, theta_filter = args
    
    print(f"  Processing session: {session}")
    session_id = session.split('_')[-1]
    
    # Convert session_id to int for ABI datasets
    if dataset_type in ["abi_visbehave", "abi_viscoding"]:
        try:
            session_id = int(session_id)
        except ValueError:
            print(f"    Could not convert session_id '{session_id}' to int")
            return None
    
    lfp_session_path = os.path.join(DATA_ROOT, config["lfp_dir"], session)
    swr_session_path = os.path.join(DATA_ROOT, config["swr_dir"], f"swrs_session_{session_id}")
    
    if not os.path.exists(swr_session_path):
        print(f"    SWR session path not found: {swr_session_path}")
        return None
    
    folderfiles = os.listdir(swr_session_path)
    
    # Initialize API connections for this process
    try:
        if dataset_type in ["abi_visbehave", "abi_viscoding"]:
            API_Class = config["api_loader"]()
            cache = API_Class.from_s3_cache(cache_dir=config["cache_dir"]) if dataset_type == "abi_visbehave" else API_Class.from_warehouse(manifest=os.path.join(config["cache_dir"], "manifest.json"))
            one = None
            wh = None
        else:  # ibl
            ONE, wh = config["api_loader"]()
            ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
            one = ONE(password='international', silent=True)
            cache = None
    except Exception as e:
        print(f"    Error initializing API for session {session_id}: {e}")
        return None
    
    # Load speed data based on dataset type (only once per session)
    try:
        if dataset_type in ["abi_visbehave", "abi_viscoding"]:
            session_obj = getattr(cache, config["session_loader"])(session_id)
            wheel_velocity, wheel_time = get_speed_data_abi(session_obj, dataset_type)
        else:  # ibl
            wheel_velocity, wheel_time = get_speed_data_ibl(one, session_id)
    except Exception as e:
        print(f"    Error loading speed data for session {session_id}: {e}")
        return None
    
    lfp_files = os.listdir(lfp_session_path)
    
    # Process each putative_swr_events file
    putative_event_files = [f for f in folderfiles if 'putative_swr_events' in f]
    if not putative_event_files:
        print(f"    No putative SWR event files found in {swr_session_path}")
        return None
    
    session_results = []
    
    for putative_file in putative_event_files:
        print(f"    Processing putative events file: {putative_file}")
        
        try:
            events_df = pd.read_csv(os.path.join(swr_session_path, putative_file), compression='gzip')
            
            # Filter events with correct criteria
            events_df = events_df[
                (events_df['power_max_zscore'] >= 3) & 
                (events_df['power_max_zscore'] <= 10) &
                (events_df['sw_peak_power'] > 1) &
                (events_df['overlaps_with_gamma'] == False) &
                (events_df['overlaps_with_movement'] == False) &
                (~events_df['start_time'].isna()) &
                (~events_df['end_time'].isna()) &
                (events_df['end_time'] > events_df['start_time']) &
                ((events_df['end_time'] - events_df['start_time']) <= 0.150)
            ]
            
            if len(events_df) == 0:
                print(f"      No events pass filters for {putative_file}")
                continue
            
            # Extract probe and channel info from filename
            probe_match = re.search(r"probe_(.*?)_", putative_file)
            channel_match = re.search(r"channel_(.*?)_", putative_file)
            
            if not probe_match or not channel_match:
                print(f"      Could not extract probe/channel info from {putative_file}")
                continue
            
            probe_id = probe_match.group(1)
            channel_indx = channel_match.group(1)
            
            # Find corresponding LFP data files
            lfp_data_file = [f for f in lfp_files if f"channel_{channel_indx}" in f and probe_id in f and "ca1_putative_pyramidal_layer.npz" in f]
            if not lfp_data_file:
                print(f"      No LFP data file for channel {channel_indx} probe {probe_id}")
                continue
            
            lfp_times_file = [f for f in lfp_files if f"channel_{channel_indx}" in f and probe_id in f and "time_index_1500hz.npz" in f]
            if not lfp_times_file:
                print(f"      No LFP time index file for channel {channel_indx} probe {probe_id}")
                continue
            
            # Load LFP data
            lfp_data = np.load(os.path.join(lfp_session_path, lfp_data_file[0]))['lfp_ca1']
            lfp_times = np.load(os.path.join(lfp_session_path, lfp_times_file[0]))['lfp_time_index']
            
            # Compute theta power
            theta_pow = np.convolve(lfp_data.flatten(), theta_filter.flatten(), mode='same')
            theta_pow_zscore = zscore(np.abs(hilbert(theta_pow)) ** 2)
            
            # Fully vectorized processing of all events
            start_times = events_df['start_time'].values
            end_times = events_df['end_time'].values
            durations = end_times - start_times
            peak_ripple_powers = events_df['power_max_zscore'].values
            
            # Vectorized theta power calculation using numpy operations
            mean_theta_powers = np.array([
                np.nanmean(theta_pow_zscore[(lfp_times >= start) & (lfp_times <= end)]) 
                if np.any((lfp_times >= start) & (lfp_times <= end)) else np.nan
                for start, end in zip(start_times, end_times)
            ])
            
            # Vectorized speed calculation using numpy operations
            mean_speeds = np.array([
                np.nanmean(np.abs(wheel_velocity[(wheel_time >= start) & (wheel_time <= end)])) 
                if np.any((wheel_time >= start) & (wheel_time <= end)) else np.nan
                for start, end in zip(start_times, end_times)
            ])
            
            # Create results DataFrame for this file
            file_results_df = pd.DataFrame({
                'session': session,
                'probe_id': probe_id,
                'channel_indx': channel_indx,
                'start_time': start_times,
                'end_time': end_times,
                'duration': durations,
                'mean_theta_power': mean_theta_powers,
                'mean_speed': mean_speeds,
                'power_max_zscore': peak_ripple_powers
            })
            
            # Convert to list of dicts for compatibility
            file_results = file_results_df.to_dict('records')
            session_results.extend(file_results)
            print(f"      Processed {len(events_df)} events from {putative_file}")
            
        except Exception as e:
            print(f"      Error processing {putative_file}: {e}")
            continue
    
    # Return session data as arrays for easy combination
    if session_results:
        session_df = pd.DataFrame(session_results)
        return {
            'theta_power': session_df['mean_theta_power'].dropna().values,
            'speed': session_df['mean_speed'].dropna().values,
            'duration': session_df['duration'].dropna().values,
            'peak_power': session_df['power_max_zscore'].dropna().values
        }
    else:
        return None

def main():
    args = parse_args()
    dataset_type = args.dataset
    config = DATASET_CONFIGS[dataset_type]
    
    print(f"Processing dataset: {dataset_type} with {args.n_processes} processes")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    theta_filter = np.load(THETA_FILTER_PATH)
    theta_filter = theta_filter[list(theta_filter.keys())[0]]
    
    # Data folders
    lfp_dir = os.path.join(DATA_ROOT, config["lfp_dir"])
    session_dirs = sorted(os.listdir(lfp_dir))
    if args.max_sessions:
        session_dirs = session_dirs[:args.max_sessions]
    
    print(f"Found {len(session_dirs)} sessions to process")
    
    # Prepare arguments for multiprocessing (only pass picklable data)
    mp_args = [(session, dataset_type, config, theta_filter) for session in session_dirs]
    
    # Process sessions in parallel
    with mp.Pool(processes=args.n_processes) as pool:
        session_results_list = pool.map(process_session, mp_args)
    
    # Collect all results
    all_session_results = [r for r in session_results_list if r is not None]
    
    if all_session_results:
        print(f"Successfully processed {len(all_session_results)} sessions")
        
        # Combine all session data
        combined_data = {
            'theta_power': np.concatenate([r['theta_power'] for r in all_session_results]),
            'speed': np.concatenate([r['speed'] for r in all_session_results]),
            'duration': np.concatenate([r['duration'] for r in all_session_results]),
            'peak_power': np.concatenate([r['peak_power'] for r in all_session_results])
        }
        
        # Save all data in single write operations
        print("Saving combined data...")
        for metric, data in combined_data.items():
            npz_file = os.path.join(OUTPUT_DIR, f"{dataset_type}_{metric}.npz")
            np.savez(npz_file, data=data)
            print(f"  Saved {len(data)} values to {metric}")
        
        print(f"Completed processing for {dataset_type}")
        print(f"Total events processed:")
        for metric, data in combined_data.items():
            print(f"  - {metric}: {len(data)} events")
    else:
        print(f"No results to save for {dataset_type}")

if __name__ == "__main__":
    main() 